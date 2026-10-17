"""Public retrieval API: candidate cosine search, exact explain and persistence.

``SearchEngine`` is the only object the rest of the project (CLI, Streamlit app,
pooling, evaluation, benchmarks) is supposed to use. It owns the tokenizer and
stopword configuration, delegates the mathematics to :class:`TfidfEngine` and
the candidate enumeration to :class:`InvertedIndex`, and never touches
``data/raw`` itself: the corpus arrives in memory from ``data/corpus.jsonl``.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .inverted_index import INDEX_FILE, InvertedIndex
from .tfidf_engine import (SCHEMA_VERSION, STATE_FILE, QueryVector, TfidfEngine,
                           _read_json, _write_json, sha256_file, verify_file_hashes)

MANIFEST_FILE = "manifest.json"
DOCUMENTS_FILE = "doc_metadata.json"
DEFAULT_SNIPPET_WIDTH = 240

MATH_PROVENANCE = {
    "term_frequency": "1 + ln(f)",
    "idf": "ln((N + 1) / (df + 1)) + 1",
    "idf_scope": "corpus_only",
    "normalization": "l2",
    "similarity": "cosine",
    "tie_break": "doc_id_ascending",
}


def _whitespace_tokenize(text: str) -> list[str]:
    """Development baseline: whitespace splitting only (no M1 dependency)."""
    return text.split()


def _make_snippet(content: str, width: int) -> str:
    text = " ".join(str(content).split())
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    return text[:width].rstrip() + "…"


def _callable_source(callable_object: Any) -> bytes | None:
    try:
        path = inspect.getsourcefile(callable_object)
    except TypeError:
        path = None
    if not path:
        return None
    try:
        return Path(path).read_bytes()
    except OSError:
        return None


def tokenizer_descriptor(callable_object: Callable) -> dict:
    """Identity of a tokenizer, used to refuse silent tokenizer swaps on load.

    ``fingerprint`` covers the callable's own source; ``state_fingerprint``
    covers instance configuration when the tokenizer exposes ``metadata()`` with
    a ``fingerprint`` key (this is how ``engine.preprocessing.Preprocessor``
    reports its tokenizer/tier1/resources configuration).
    """
    if inspect.isfunction(callable_object) or inspect.ismethod(callable_object):
        kind = "function"
        module = getattr(callable_object, "__module__", None)
        qualname = getattr(callable_object, "__qualname__", None)
        source = _callable_source(callable_object)
        spec = f"{module}:{qualname}" if module and qualname else None
        label = spec or repr(callable_object)
    else:
        kind = "callable_object"
        owner = type(callable_object)
        module = getattr(owner, "__module__", None)
        qualname = getattr(owner, "__qualname__", owner.__name__)
        source = _callable_source(owner)
        spec = None
        label = f"{module}:{qualname}" if module and qualname else repr(callable_object)
    fingerprint = None
    if source is not None:
        fingerprint = hashlib.sha256(
            json.dumps({"kind": kind, "label": label,
                        "source_sha256": hashlib.sha256(source).hexdigest()},
                       sort_keys=True).encode("utf-8")).hexdigest()
    state_fingerprint = None
    metadata_method = getattr(callable_object, "metadata", None)
    if callable(metadata_method):
        try:
            reported = metadata_method()
        except Exception:  # noqa: BLE001 - a tokenizer must never break persistence
            reported = None
        if isinstance(reported, Mapping):
            value = reported.get("fingerprint")
            if isinstance(value, str):
                state_fingerprint = value
    return {"kind": kind, "label": label, "spec": spec, "fingerprint": fingerprint,
            "state_fingerprint": state_fingerprint,
            "resolvable": spec is not None and kind == "function"}


def _resolve_spec(spec: str | None) -> Callable | None:
    if not spec or ":" not in spec:
        return None
    module_name, _, qualname = spec.partition(":")
    try:
        target: Any = importlib.import_module(module_name)
    except ImportError:
        return None
    for part in qualname.split("."):
        target = getattr(target, part, None)
        if target is None:
            return None
    return target if callable(target) else None


def _validate_top_k(top_k: Any) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)):
        raise TypeError("top_k must be an integer")
    value = int(top_k)
    if value < 0:
        raise ValueError("top_k must be >= 0")
    return value


class SearchEngine:
    """TF-IDF cosine retrieval over a fitted corpus.

    Parameters
    ----------
    tokenizer:
        Callable ``str -> list[str]``. ``None`` selects the whitespace baseline
        that the contract marks as development only.
    stopwords:
        Iterable of tokens removed from corpus *and* query tokens alike.
    metadata:
        Provenance dict (corpus hash, configuration, preprocessing status ...).
        Stored verbatim and restored by :meth:`load`.
    """

    def __init__(self, tokenizer: Callable[[str], list[str]] | None = None,
                 stopwords: Iterable[str] | None = None,
                 metadata: Mapping | None = None) -> None:
        if tokenizer is not None and not callable(tokenizer):
            raise TypeError("tokenizer must be callable or None")
        self.tokenizer: Callable[[str], list[str]] = tokenizer or _whitespace_tokenize
        self.tokenizer_explicit = tokenizer is not None
        self.stopwords_: frozenset[str] = frozenset(stopwords or ())
        self.metadata: dict = dict(metadata) if metadata else {}
        self.snippet_width: int = DEFAULT_SNIPPET_WIDTH
        self.tfidf: TfidfEngine | None = None
        self.index: InvertedIndex | None = None
        self.doc_ids_: list[str] = []
        self.titles_: list[str] = []
        self.categories_: list[str] = []
        self.snippets_: list[str] = []
        self._doc_position: dict[str, int] = {}
        self._doc_id_rank = np.zeros(0, dtype=np.int64)
        self._fitted = False

    # ------------------------------------------------------------------- state

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def n_documents(self) -> int:
        return len(self.doc_ids_)

    def _require_fitted(self) -> None:
        if not self._fitted or self.tfidf is None or self.index is None:
            raise RuntimeError("SearchEngine.fit must be called before searching")

    # -------------------------------------------------------------- tokenizing

    def tokenize(self, text: str) -> list[str]:
        """Apply the index tokenizer only (stopword filtering happens later)."""
        if not isinstance(text, str):
            raise TypeError("tokenizer input must be a string")
        return list(self.tokenizer(text))

    def _coerce_query(self, query: str | Sequence[str]) -> list[str]:
        if isinstance(query, str):
            return self.tokenize(query)
        if isinstance(query, (list, tuple)) or (
                isinstance(query, Sequence) and not isinstance(query, (str, bytes))):
            tokens = list(query)
            for token in tokens:
                if not isinstance(token, str):
                    raise TypeError("A token list query must contain only str tokens")
            return tokens
        raise TypeError("query must be a string or a list of tokens")

    def _query_vector(self, query: str | Sequence[str]) -> QueryVector:
        self._require_fitted()
        return self.tfidf.vectorize(self._coerce_query(query))  # type: ignore[union-attr]

    # -------------------------------------------------------------------- fit

    def fit(self, documents: Sequence[Mapping],
            tokenized_docs: Sequence[Sequence[str]] | None = None) -> "SearchEngine":
        """Index ``content`` of every document.

        ``tokenized_docs``, when given, must be aligned with ``documents`` and is
        used as-is (that is how ``engine.preprocessing`` output enters the
        index); otherwise ``content`` is tokenized with ``self.tokenizer``. The
        title is never tokenized separately: it already lives inside ``content``.
        """
        if isinstance(documents, Mapping):
            raise TypeError("documents must be a sequence of document mappings")
        records = list(documents)
        if tokenized_docs is None:
            tokens_per_doc = [self.tokenize(str(document.get("content", "")))
                              for document in records]
            source = "engine_tokenizer"
        else:
            tokens_per_doc = list(tokenized_docs)
            source = "caller"
            if len(tokens_per_doc) != len(records):
                raise ValueError(
                    f"tokenized_docs has {len(tokens_per_doc)} rows but documents has "
                    f"{len(records)}")

        doc_ids: list[str] = []
        titles: list[str] = []
        categories: list[str] = []
        snippets: list[str] = []
        for document in records:
            if not isinstance(document, Mapping):
                raise TypeError("each document must be a mapping")
            doc_id = document.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("each document needs a non-empty string doc_id")
            doc_ids.append(doc_id)
            titles.append(str(document.get("title") or ""))
            categories.append(str(document.get("category") or ""))
            snippets.append(_make_snippet(document.get("content", ""), self.snippet_width))
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("doc_id values must be unique")

        tfidf = TfidfEngine(stopwords=self.stopwords_)
        tfidf.fit(tokens_per_doc)
        index = InvertedIndex()
        index.fit(tfidf.matrix_, vocabulary=tfidf.vocabulary_)

        self.tfidf = tfidf
        self.index = index
        self.doc_ids_ = doc_ids
        self.titles_ = titles
        self.categories_ = categories
        self.snippets_ = snippets
        self._install_lookup()
        self._fitted = True
        self.metadata.setdefault("schema_version", 1)
        self.metadata["retrieval"] = {
            "engine": "engine.indexer.search_engine.SearchEngine",
            "schema_version": SCHEMA_VERSION,
            "math": dict(MATH_PROVENANCE),
            "n_documents": len(doc_ids),
            "vocabulary_size": tfidf.n_features_,
            "stopword_count": len(self.stopwords_),
            "tokenized_source": source,
            "snippet_width": self.snippet_width,
            "tokenizer": tokenizer_descriptor(self.tokenizer),
        }
        return self

    def _install_lookup(self) -> None:
        self._doc_position = {doc_id: row for row, doc_id in enumerate(self.doc_ids_)}
        ranks = {doc_id: rank for rank, doc_id in enumerate(sorted(self.doc_ids_))}
        self._doc_id_rank = np.asarray([ranks[doc_id] for doc_id in self.doc_ids_], dtype=np.int64)

    def _install_documents(self, records: Sequence[Mapping]) -> None:
        self.doc_ids_ = [str(record["doc_id"]) for record in records]
        self.titles_ = [str(record.get("title", "")) for record in records]
        self.categories_ = [str(record.get("category", "")) for record in records]
        self.snippets_ = [str(record.get("snippet", "")) for record in records]
        self._install_lookup()

    def document(self, doc_id: str) -> dict:
        """Stored display fields for one document (no scoring)."""
        return {
            "doc_id": doc_id,
            "title": self.titles_[self._row_for(doc_id)],
            "category": self.categories_[self._row_for(doc_id)],
            "snippet": self.snippets_[self._row_for(doc_id)],
        }

    def _row_for(self, doc_id: str) -> int:
        row = self._doc_position.get(doc_id) if isinstance(doc_id, str) else None
        if row is None:
            raise KeyError(f"Unknown doc_id: {doc_id!r}")
        return row

    # ----------------------------------------------------------------- search

    def search(self, query: str | Sequence[str], top_k: int = 10) -> list[dict]:
        """Top-``k`` documents with a strictly positive cosine score.

        Ordering is score descending, then ``doc_id`` ascending for ties. An
        empty, stopword-only or out-of-vocabulary query returns ``[]`` because it
        yields a zero query vector.
        """
        self._require_fitted()
        limit = _validate_top_k(top_k)
        if limit == 0:
            return []
        vector = self._query_vector(query)
        if vector.is_empty:
            return []
        rows, scores = self.index.score(vector)  # type: ignore[union-attr]
        if rows.size == 0:
            return []
        positive = scores > 0.0
        rows, scores = rows[positive], scores[positive]
        if rows.size == 0:
            return []
        order = np.lexsort((self._doc_id_rank[rows], -scores))[:limit]
        return [self._hit(int(rows[position]), float(scores[position])) for position in order]

    def _hit(self, row: int, score: float) -> dict:
        return {
            "doc_id": self.doc_ids_[row],
            "title": self.titles_[row],
            "category": self.categories_[row],
            "score": score,
            "snippet": self.snippets_[row],
        }

    # ---------------------------------------------------------------- explain

    def explain(self, query: str | Sequence[str], doc_id: str) -> dict:
        """Exact per-term decomposition of the cosine score of one document.

        ``query_tf``/``doc_tf`` are raw frequencies, ``idf`` is the corpus IDF
        and ``query_weight``/``doc_weight`` are the L2-normalised weights that
        the cosine actually uses; ``contribution = query_weight * doc_weight``
        and the contributions sum to ``score``.
        """
        self._require_fitted()
        row = self._row_for(doc_id)
        tokens = self._coerce_query(query)
        vector = self.tfidf.vectorize(self.tfidf.filter_tokens(tokens))  # type: ignore[union-attr]
        contributions = []
        for position, term in enumerate(vector.terms):
            term_id = int(vector.term_ids[position])
            doc_weight = float(self.tfidf.doc_term_weight(row, term_id))  # type: ignore[union-attr]
            contributions.append({
                "term": term,
                "query_tf": int(vector.tfs[position]),
                "doc_tf": int(self.tfidf.doc_term_tf(row, term_id)),  # type: ignore[union-attr]
                "idf": float(vector.idf[position]),
                "query_weight": float(vector.weights[position]),
                "doc_weight": doc_weight,
                "contribution": float(vector.weights[position]) * doc_weight,
            })
        contributions.sort(key=lambda item: (-abs(item["contribution"]), item["term"]))
        score = math.fsum(item["contribution"] for item in contributions)
        return {
            "doc_id": doc_id,
            "title": self.titles_[row],
            "category": self.categories_[row],
            "score": score,
            "contributions": contributions,
            "query_terms": list(vector.terms),
            "oov_terms": list(vector.oov_terms),
            "dropped_terms": self.tfidf.dropped_tokens(tokens),  # type: ignore[union-attr]
            "query_norm": vector.norm,
            "doc_norm": float(self.tfidf.doc_norms_[row]),  # type: ignore[union-attr]
        }

    def score(self, query: str | Sequence[str], doc_id: str) -> float:
        """Cosine score of one document (identical to ``explain``'s ``score``)."""
        return float(self.explain(query, doc_id)["score"])

    # ------------------------------------------------------------- persistence

    def save(self, directory: str | Path) -> dict:
        """Write the complete state and a hashed manifest into ``directory``."""
        self._require_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        files = dict(self.tfidf.save(directory))  # type: ignore[union-attr]
        files[INDEX_FILE] = self.index.save(directory)  # type: ignore[union-attr]
        _write_json(directory / DOCUMENTS_FILE, {
            "schema_version": SCHEMA_VERSION,
            "documents": [
                {"doc_id": doc_id, "title": title, "category": category, "snippet": snippet}
                for doc_id, title, category, snippet
                in zip(self.doc_ids_, self.titles_, self.categories_, self.snippets_)
            ],
        })
        files[DOCUMENTS_FILE] = sha256_file(directory / DOCUMENTS_FILE)
        files[STATE_FILE] = sha256_file(directory / STATE_FILE)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "component": "engine.indexer.search_engine.SearchEngine",
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            "n_documents": len(self.doc_ids_),
            "n_features": self.tfidf.n_features_,  # type: ignore[union-attr]
            "math": dict(MATH_PROVENANCE),
            "tokenizer": tokenizer_descriptor(self.tokenizer),
            "stopwords": {
                "count": len(self.stopwords_),
                "sha256": hashlib.sha256(
                    "\n".join(sorted(self.stopwords_)).encode("utf-8")).hexdigest(),
                "terms": sorted(self.stopwords_),
            },
            "snippet_width": self.snippet_width,
            "metadata": self.metadata,
            "files": files,
        }
        _write_json(directory / MANIFEST_FILE, manifest)
        return manifest

    @classmethod
    def load(cls, directory: str | Path, tokenizer: Callable[[str], list[str]] | None = None
             ) -> "SearchEngine":
        """Rebuild an engine from ``save`` output.

        Without ``tokenizer`` the saved tokenizer is re-imported from its
        ``module:qualname`` spec. Passing a different tokenizer raises instead of
        silently changing the analysis of the index.
        """
        directory = Path(directory)
        manifest = _read_json(directory / MANIFEST_FILE)
        version = int(manifest.get("schema_version", -1))
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported index schema version: {version}")
        verify_file_hashes(directory, manifest.get("files", {}))
        resolved = cls._resolve_saved_tokenizer(manifest.get("tokenizer", {}), tokenizer)
        stopwords = manifest.get("stopwords", {}).get("terms", [])
        engine = cls(tokenizer=resolved, stopwords=stopwords, metadata=manifest.get("metadata", {}))
        engine.snippet_width = int(manifest.get("snippet_width", DEFAULT_SNIPPET_WIDTH))
        engine.tfidf = TfidfEngine.load(directory)
        engine.index = InvertedIndex.load(directory, vocabulary=engine.tfidf.vocabulary_)
        engine._install_documents(_read_json(directory / DOCUMENTS_FILE)["documents"])
        stored_documents = len(engine.doc_ids_)
        if stored_documents != engine.tfidf.n_documents_ or \
                stored_documents != engine.index.n_documents:
            raise ValueError(
                "Stored documents, term counts and postings disagree on the corpus size")
        if engine.tfidf.n_features_ != int(manifest["n_features"]):
            raise ValueError("Stored vocabulary size does not match the manifest")
        if frozenset(stopwords) != engine.tfidf.stopwords_:
            raise ValueError("Stopword set differs between the manifest and the TF-IDF state")
        engine._fitted = True
        return engine

    @staticmethod
    def _resolve_saved_tokenizer(saved: Mapping, tokenizer: Callable | None) -> Callable:
        if tokenizer is not None:
            if not callable(tokenizer):
                raise TypeError("tokenizer must be callable or None")
            current = tokenizer_descriptor(tokenizer)
            saved_fingerprint = saved.get("fingerprint")
            current_fingerprint = current.get("fingerprint")
            if not saved_fingerprint or not current_fingerprint:
                raise ValueError(
                    "Cannot verify that the given tokenizer is the tokenizer of this index "
                    f"(index={saved.get('label')!r}, given={current.get('label')!r}); "
                    "rebuild the index instead of silently switching tokenizer")
            if saved_fingerprint != current_fingerprint:
                raise ValueError(
                    "Tokenizer source does not match the saved index; rebuild the index instead of "
                    "silently switching tokenizer")
            saved_state = saved.get("state_fingerprint")
            current_state = current.get("state_fingerprint")
            if saved_state != current_state and (saved_state or current_state):
                raise ValueError(
                    "Tokenizer configuration does not match the saved index; rebuild the index "
                    "of silently switching tokenizer")
            return tokenizer
        if not saved.get("resolvable") or not saved.get("spec"):
            raise ValueError(
                "The tokenizer of this index cannot be re-imported "
                f"({saved.get('label')!r}); pass tokenizer=... explicitly to SearchEngine.load")
        resolved = _resolve_spec(saved["spec"])
        if resolved is None:
            raise ValueError(
                f"Cannot import the saved tokenizer {saved['spec']!r}; "
                "pass tokenizer=... explicitly")
        return resolved

    def __repr__(self) -> str:
        state = "fitted" if self._fitted else "unfitted"
        return (f"SearchEngine({state}, n_documents={self.n_documents}, "
                f"stopwords={len(self.stopwords_)})")
