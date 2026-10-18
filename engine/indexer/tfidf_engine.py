"""Sparse TF-IDF weighting over a corpus-frozen vocabulary.

Mathematical contract (``docs/api_contracts.md``, section "API retrieval")::

    f(t, d)   raw frequency of term ``t`` in document ``d``
    TF(t, d)  = 1 + ln f(t, d)                    (sublinear TF)
    IDF(t)    = ln((N + 1) / (df(t) + 1)) + 1     (smoothed corpus IDF)
    w(t, d)   = TF(t, d) * IDF(t)
    w~(t, d)  = w(t, d) / ||d||_2                 (L2 row normalisation, zero rows stay zero)

The vocabulary, the IDF table and the document norms are derived once from the
corpus and then frozen: a query is always weighted with the *corpus* IDF, can
never extend the vocabulary, and the same stopword set is applied to corpus
tokens and query tokens.

Nothing in this module writes to ``data/raw``; the corpus is only ever received
in memory as a sequence of token sequences.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

SCHEMA_VERSION = 1

STATE_FILE = "tfidf_state.json"
VOCABULARY_FILE = "vocabulary.json"
IDF_FILE = "idf_weights.json"
COUNTS_FILE = "tf_counts.npz"
NORM_FILE = "doc_norms.npy"

TokenizedDocs = Sequence[Sequence[str]]


def sha256_file(path: str | Path) -> str:
    """Streaming SHA256 of a file, used for index integrity checks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_file_hashes(directory: str | Path, expected: dict[str, str]) -> None:
    """Raise ``ValueError`` when an index file is missing or its SHA256 changed."""
    directory = Path(directory)
    for name, digest in sorted(expected.items()):
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Index file is missing: {path}")
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"Index file hash mismatch for {name}: "
                            f"expected {digest}, found {actual}")


@dataclass(frozen=True, eq=False)
class QueryVector:
    """A query represented in the frozen corpus space.

    ``tfs`` are raw query term frequencies, ``raw_weights`` are
    ``(1 + ln tf) * idf`` and ``weights`` are the L2-normalised weights that the
    cosine similarity actually multiplies with the document weights.
    """

    terms: tuple[str, ...]
    term_ids: np.ndarray
    tfs: np.ndarray
    idf: np.ndarray
    raw_weights: np.ndarray
    weights: np.ndarray
    norm: float
    oov_terms: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.terms)

    def __repr__(self) -> str:
        return (f"QueryVector(terms={list(self.terms)}, oov={list(self.oov_terms)}, "
                f"norm={self.norm:.6f})")

    @property
    def is_empty(self) -> bool:
        return len(self.terms) == 0

    def weights_by_term(self) -> dict[str, float]:
        return {term: float(weight) for term, weight in zip(self.terms, self.weights)}


class TfidfEngine:
    """Fits the corpus vocabulary/IDF table and weights documents or queries.

    Parameters
    ----------
    stopwords:
        Iterable of tokens removed *identically* from corpus and query tokens.
        Empty or whitespace-only tokens are removed as well, matching the
        ``engine.preprocessing`` boundary.
    sublinear_tf, smooth_idf, norm:
        Kept as explicit state because they are serialised with the index; the
        M2 contract fixes them to ``True / True / "l2"``.
    """

    def __init__(self, stopwords: Iterable[str] | None = None, sublinear_tf: bool = True,
                 smooth_idf: bool = True, norm: str = "l2") -> None:
        if norm != "l2":
            raise ValueError("Only L2 row normalisation is part of the M2 contract")
        self.sublinear_tf = bool(sublinear_tf)
        self.smooth_idf = bool(smooth_idf)
        self.norm = norm
        self.stopwords_: frozenset[str] = frozenset(stopwords or ())
        self._fitted = False
        self.vocabulary_: dict[str, int] = {}
        self.n_features_: int = 0
        self.n_documents_: int = 0
        self.df_: np.ndarray = np.zeros(0, dtype=np.int64)
        self.idf_: np.ndarray = np.zeros(0, dtype=np.float64)
        self.counts_: csr_matrix = csr_matrix((0, 0), dtype=np.int64)
        self.matrix_: csr_matrix = csr_matrix((0, 0), dtype=np.float64)
        self.doc_norms_: np.ndarray = np.zeros(0, dtype=np.float64)

    # ------------------------------------------------------------------ state

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TfidfEngine.fit must be called before this operation")

    # -------------------------------------------------------------- filtering

    def filter_tokens(self, tokens: Iterable[str]) -> list[str]:
        """Drop stopwords and empty tokens; identical rule for corpus and query."""
        stopwords = self.stopwords_
        kept: list[str] = []
        for token in tokens:
            if not isinstance(token, str):
                raise TypeError(
                    f"tokenizer must return str tokens, got {type(token).__name__}")
            if not token or token.isspace() or token in stopwords:
                continue
            kept.append(token)
        return kept

    def dropped_tokens(self, tokens: Iterable[str]) -> list[str]:
        """Multiset of tokens removed by :meth:`filter_tokens` (for provenance)."""
        tokens = list(tokens)
        remaining = Counter(self.filter_tokens(tokens))
        dropped = Counter(tokens)
        dropped.subtract(remaining)
        return sorted(token for token, count in dropped.items() for _ in range(max(count, 0)))

    # -------------------------------------------------------------------- fit

    def fit(self, tokenized_docs: TokenizedDocs) -> "TfidfEngine":
        documents = [self.filter_tokens(tokens) for tokens in tokenized_docs]
        vocabulary = sorted({token for tokens in documents for token in tokens})
        self.vocabulary_ = {term: position for position, term in enumerate(vocabulary)}
        self.n_features_ = len(vocabulary)
        self.n_documents_ = len(documents)
        self.df_ = np.zeros(self.n_features_, dtype=np.int64)
        rows: list[list[tuple[int, int]]] = []
        for tokens in documents:
            counts = Counter(tokens)
            for term in counts:
                self.df_[self.vocabulary_[term]] += 1
            rows.append(sorted((self.vocabulary_[term], count) for term, count in counts.items()))
        self.idf_ = self._idf(self.df_)
        self.counts_ = self._counts_matrix(rows, self.n_documents_)
        self.matrix_, self.doc_norms_ = self._weight(self.counts_)
        self._fitted = True
        return self

    def _idf(self, document_frequency: np.ndarray) -> np.ndarray:
        n_documents = float(self.n_documents_)
        document_frequency = document_frequency.astype(np.float64)
        if self.smooth_idf:
            return np.log((1.0 + n_documents) / (1.0 + document_frequency)) + 1.0
        with np.errstate(divide="ignore"):
            return np.log(n_documents / document_frequency) + 1.0

    def _counts_matrix(self, rows: Sequence[Sequence[tuple[int, int]]], n_rows: int) -> csr_matrix:
        indices: list[int] = []
        data: list[int] = []
        indptr: list[int] = [0]
        for row in rows:
            indices.extend(column for column, _ in row)
            data.extend(count for _, count in row)
            indptr.append(len(indices))
        return csr_matrix(
            (
                np.asarray(data, dtype=np.int64),
                np.asarray(indices, dtype=np.int32),
                np.asarray(indptr, dtype=np.int32),
            ),
            shape=(n_rows, self.n_features_),
        )

    def _weight(self, counts: csr_matrix) -> tuple[csr_matrix, np.ndarray]:
        """Apply sublinear TF * corpus IDF and L2-normalise every non-zero row."""
        term_frequency = counts.astype(np.float64)
        if self.sublinear_tf:
            term_frequency.data = 1.0 + np.log(term_frequency.data)
        weights = term_frequency.multiply(self.idf_).tocsr()
        norms = np.sqrt(np.asarray(weights.multiply(weights).sum(axis=1)).ravel())
        inverse = np.zeros(norms.shape[0], dtype=np.float64)
        nonzero = norms > 0.0
        inverse[nonzero] = 1.0 / norms[nonzero]
        normalized = weights.tocsr()
        normalized.data = normalized.data * np.repeat(inverse, np.diff(normalized.indptr))
        normalized.sort_indices()
        return normalized, norms

    # -------------------------------------------------------------- transform

    def transform_documents(self, tokenized_docs: TokenizedDocs) -> csr_matrix:
        """L2-normalised TF-IDF rows for new token sequences (OOV terms dropped)."""
        self._require_fitted()
        return self._transform_rows(tokenized_docs)[1]

    def transform_full(self, tokenized_docs: TokenizedDocs
                        ) -> tuple[csr_matrix, csr_matrix, np.ndarray]:
        """Same as :meth:`transform_documents` but also returns counts and norms."""
        self._require_fitted()
        return self._transform_rows(tokenized_docs)

    def _transform_rows(self, tokenized_docs: TokenizedDocs):
        rows: list[list[tuple[int, int]]] = []
        for tokens in tokenized_docs:
            counts = Counter(self.filter_tokens(tokens))
            rows.append(sorted((self.vocabulary_[term], count)
                              for term, count in counts.items() if term in self.vocabulary_))
        counts = self._counts_matrix(rows, len(rows))
        matrix, norms = self._weight(counts)
        return counts, matrix, norms

    def vectorize(self, tokens: Iterable[str]) -> QueryVector:
        """Weight a query with the frozen vocabulary and the corpus IDF."""
        self._require_fitted()
        filtered = self.filter_tokens(tokens)
        counts = Counter(filtered)
        oov = tuple(sorted(token for token in counts if token not in self.vocabulary_))
        in_vocabulary = sorted(
            (self.vocabulary_[token], token) for token in counts if token in self.vocabulary_)
        term_ids = np.asarray([column for column, _ in in_vocabulary], dtype=np.int32)
        terms = tuple(token for _, token in in_vocabulary)
        tfs = np.asarray([float(counts[token]) for _, token in in_vocabulary], dtype=np.float64)
        if term_ids.size == 0:
            empty = np.zeros(0, dtype=np.float64)
            return QueryVector(terms, term_ids, empty, empty, empty, empty, 0.0, oov)
        idf = self.idf_[term_ids]
        raw_weights = (1.0 + np.log(tfs)) * idf if self.sublinear_tf else tfs * idf
        norm = float(math.sqrt(float(raw_weights @ raw_weights)))
        weights = raw_weights / norm if norm > 0.0 else np.zeros_like(raw_weights)
        return QueryVector(terms, term_ids, tfs, idf, raw_weights, weights, norm, oov)

    def cosine_scores(self, query: QueryVector) -> np.ndarray:
        """Cosine of every document against a normalised query vector.

        Only the query's own columns are multiplied: ``query.weights`` has one
        entry per *query* term, not one per vocabulary column.
        """
        self._require_fitted()
        if query.is_empty:
            return np.zeros(self.n_documents_, dtype=np.float64)
        selected = self.matrix_[:, query.term_ids]
        return np.asarray(selected.dot(query.weights)).ravel()

    def unnormalized(self) -> csr_matrix:
        """TF * IDF before L2 normalisation (kept for TDV / inspection)."""
        self._require_fitted()
        term_frequency = self.counts_.astype(np.float64)
        if self.sublinear_tf:
            term_frequency.data = 1.0 + np.log(term_frequency.data)
        return term_frequency.multiply(self.idf_).tocsr()

    def to_dense(self) -> np.ndarray:
        self._require_fitted()
        return self.matrix_.toarray()

    # ------------------------------------------------------------ vocabulary

    def term_id(self, term: str) -> int | None:
        return self.vocabulary_.get(term)

    def term(self, term_id: int) -> str:
        position = int(term_id)
        if not 0 <= position < self.n_features_:
            raise IndexError(f"term_id out of range: {term_id}")
        return self._terms()[position]

    def _terms(self) -> list[str]:
        return sorted(self.vocabulary_, key=self.vocabulary_.get)

    def terms(self) -> list[str]:
        return self._terms()

    def term_stats(self, term: str) -> dict:
        self._require_fitted()
        position = self.vocabulary_.get(term)
        if position is None:
            return {"term": term, "term_id": None, "df": 0, "idf": 0.0}
        return {
            "term": term,
            "term_id": position,
            "df": int(self.df_[position]),
            "idf": float(self.idf_[position]),
            "document_frequency_ratio": (int(self.df_[position]) / self.n_documents_
                                         if self.n_documents_ else 0.0),
        }

    def doc_term_tf(self, document: int, term_id: int) -> int:
        self._require_fitted()
        row = self.counts_.getrow(int(document))
        column = row.indices
        position = int(np.searchsorted(column, int(term_id)))
        if position < column.size and int(column[position]) == int(term_id):
            return int(row.data[position])
        return 0

    def doc_term_weight(self, document: int, term_id: int) -> float:
        self._require_fitted()
        position = int(term_id)
        return float(self.matrix_[int(document), position])

    def doc_raw_weight(self, document: int, term_id: int) -> float:
        self._require_fitted()
        return float(self.unnormalized()[int(document), int(term_id)])

    # ----------------------------------------------------------- persistence

    def save(self, directory: str | Path) -> dict[str, str]:
        self._require_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_npz(directory / COUNTS_FILE, self.counts_)
        np.save(directory / NORM_FILE, self.doc_norms_)
        _write_json(directory / VOCABULARY_FILE,
                    {"schema_version": SCHEMA_VERSION, "terms": self._terms()})
        _write_json(directory / IDF_FILE, {
            "schema_version": SCHEMA_VERSION,
            "formula": "ln((N + 1) / (df + 1)) + 1" if self.smooth_idf else "ln(N / df) + 1",
            "n_documents": self.n_documents_,
            "weights": self.idf_.tolist(),
        })
        _write_json(directory / STATE_FILE, {
            "schema_version": SCHEMA_VERSION,
            "component": "engine.indexer.tfidf_engine.TfidfEngine",
            "sublinear_tf": self.sublinear_tf,
            "smooth_idf": self.smooth_idf,
            "norm": self.norm,
            "n_documents": self.n_documents_,
            "n_features": self.n_features_,
            "stopwords": sorted(self.stopwords_),
            "files": {
                COUNTS_FILE: sha256_file(directory / COUNTS_FILE),
                NORM_FILE: sha256_file(directory / NORM_FILE),
                VOCABULARY_FILE: sha256_file(directory / VOCABULARY_FILE),
                IDF_FILE: sha256_file(directory / IDF_FILE),
            },
        })
        return {
            COUNTS_FILE: sha256_file(directory / COUNTS_FILE),
            NORM_FILE: sha256_file(directory / NORM_FILE),
            VOCABULARY_FILE: sha256_file(directory / VOCABULARY_FILE),
            IDF_FILE: sha256_file(directory / IDF_FILE),
        }

    @classmethod
    def load(cls, directory: str | Path) -> "TfidfEngine":
        directory = Path(directory)
        state = _read_json(directory / STATE_FILE)
        verify_file_hashes(directory, state.get("files", {}))
        engine = cls(stopwords=state.get("stopwords"), sublinear_tf=state["sublinear_tf"],
                     smooth_idf=state["smooth_idf"], norm=state["norm"])
        terms = _read_json(directory / VOCABULARY_FILE)["terms"]
        engine.vocabulary_ = {term: position for position, term in enumerate(terms)}
        engine.n_features_ = len(terms)
        engine.n_documents_ = int(state["n_documents"])
        engine.idf_ = np.asarray(_read_json(directory / IDF_FILE)["weights"], dtype=np.float64)
        engine.counts_ = load_npz(directory / COUNTS_FILE).tocsr()
        if engine.counts_.shape != (engine.n_documents_, engine.n_features_):
            raise ValueError("Stored term counts do not match the stored vocabulary shape")
        engine.df_ = np.asarray(engine.counts_.getnnz(axis=0)).ravel().astype(np.int64)
        expected_idf = engine._idf(engine.df_)
        if not np.allclose(expected_idf, engine.idf_, rtol=0, atol=1e-12):
            raise ValueError(
                "Stored IDF weights are inconsistent with the stored document frequencies")
        stored_norms = np.load(directory / NORM_FILE)
        engine.matrix_, norms = engine._weight(engine.counts_)
        if norms.shape != stored_norms.shape or not np.allclose(
                norms, stored_norms, rtol=0, atol=1e-12):
            raise ValueError("Stored document norms are inconsistent with the stored counts/IDF")
        engine.doc_norms_ = stored_norms
        engine._fitted = True
        return engine

    def __repr__(self) -> str:
        return (f"TfidfEngine(n_documents={self.n_documents_}, n_features={self.n_features_}, "
                f"stopwords={len(self.stopwords_)})")
