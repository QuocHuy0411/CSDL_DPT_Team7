"""Build the TREC-style judgment pool for the 20 locked queries.

Pooling method: union of the Top-20 rankings of three baselines computed on the
same corpus and the same tokenizer. Every pooled pair keeps the rank and score
each baseline gave it, so a human annotator (or a later report) can see where a
document came from.

All baselines index `content` only and use the same supplied tokenizer.

1. tfidf_cosine
   TF(t,d) = 1 + ln(f)                    for f > 0, else 0
   IDF(t)  = ln((N + 1)/(df + 1)) + 1
   w(t,d)  = TF(t,d) * IDF(t), L2 normalisation on documents AND query, cosine.
2. binary_cosine  (cosine between binary presence vectors, NO idf)
   w(t,d) = 1 if the term occurs in d, else 0; L2 on documents AND query.
   This is deliberately different from tfidf_cosine: no term weighting at all.
3. bm25  (Robertson/Sparck-Jones, standard qtf handling)
   IDF(t)  = ln(1 + (N - df + 0.5)/(df + 0.5))
   w(t,d)  = IDF(t) * f(t,d)(k1 + 1) / (f(t,d) + k1(1 - b + b*dl/avgdl))
   k1 = 1.2, b = 0.75. Each unique query term contributes its IDF exactly once;
   no query term frequency and no length normalisation of the query. No L2.

Outputs (data/ground_truth/):
    pool.json        pooled pairs with per-baseline provenance and document content
    pool_stats.json  sizes, baseline overlap, and the SearchEngine parity result
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.preprocessing import Preprocessor, baseline_tokenize  # noqa: E402

DEFAULT_CORPUS = ROOT / "data/corpus.jsonl"
DEFAULT_MANIFEST = ROOT / "data/corpus_manifest.json"
DEFAULT_QUERIES = ROOT / "data/queries.json"
DEFAULT_POOL = ROOT / "data/ground_truth/mm/pool.json"
DEFAULT_STATS = ROOT / "data/ground_truth/mm/pool_stats.json"

SCHEMA_VERSION = 1
POOL_DEPTH = 20
BM25_K1 = 1.2
BM25_B = 0.75
TOKENIZER_SPEC = "engine.preprocessing:baseline_tokenize"
SCORE_TOLERANCE = 1e-8
BASELINES = ("tfidf_cosine", "binary_cosine", "bm25")


class PoolError(RuntimeError):
    """Raised when the pool cannot be built against the locked inputs."""


class ParityError(PoolError):
    """Raised when the tfidf baseline disagrees with the public SearchEngine."""


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_corpus(corpus_path: Path, manifest_path: Path | None) -> tuple[list[dict], str]:
    payload = Path(corpus_path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    documents = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if manifest_path is not None:
        manifest = load_json(manifest_path)
        expected = manifest.get("corpus_sha256") or manifest.get("sha256")
        if expected != digest:
            raise PoolError(f"corpus sha256 {digest} does not match manifest {expected}")
    return documents, digest


def build_vocabulary(token_lists: list[list[str]]) -> dict[str, int]:
    vocabulary: dict[str, int] = {}
    for tokens in token_lists:
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    return vocabulary


def count_matrix(token_lists: list[list[str]], vocabulary: dict[str, int]) -> sparse.csr_matrix:
    """Document term frequency matrix in CSR form."""
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for tokens in token_lists:
        counts = Counter(token for token in tokens if token in vocabulary)
        for token, frequency in sorted(counts.items()):
            indices.append(vocabulary[token])
            data.append(float(frequency))
        indptr.append(len(indices))
    return sparse.csr_matrix((data, indices, indptr),
                            shape=(len(token_lists), len(vocabulary)), dtype=np.float64)


def row_of_each_nonzero(matrix: sparse.csr_matrix) -> np.ndarray:
    """Row index for every stored value; used to broadcast document lengths."""
    counts_per_row = np.diff(matrix.indptr)
    return np.repeat(np.arange(matrix.shape[0], dtype=np.int64), counts_per_row)


def l2_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0.0] = 1.0
    return sparse.csr_matrix(sparse.diags(1.0 / norms) @ matrix)


def sublinear_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """IDF ln((N+1)/(df+1))+1, the smoothing locked by the retrieval contract."""
    return np.log((n_docs + 1.0) / (df + 1.0)) + 1.0


def bm25_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    """Robertson/Sparck-Jones IDF with the standard +1 inside the logarithm."""
    return np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def query_vector(vocabulary: dict[str, int], tokens: list[str], vocabulary_size: int,
                 weight, normalize: bool) -> sparse.csr_matrix:
    """Build one query row. `weight(qtf, column_index) -> value`."""
    counts = Counter(token for token in tokens if token in vocabulary)
    if not counts:
        return sparse.csr_matrix((1, vocabulary_size), dtype=np.float64)
    pairs = sorted((vocabulary[token], token) for token in counts)
    indices = np.asarray([index for index, _ in pairs], dtype=np.int64)
    values = [weight(counts[token], index) for index, token in pairs]
    vector = sparse.csr_matrix(
        (np.asarray(values, dtype=np.float64),
         (np.zeros(len(indices), dtype=np.int64), indices)),
        shape=(1, vocabulary_size))
    return l2_normalize(vector) if normalize else vector


def rank(scores: np.ndarray, doc_ids: list[str], depth: int) -> list[tuple[str, float]]:
    """Top-`depth` strictly positive scores; ties broken by ascending doc_id."""
    positive = [(index, float(value)) for index, value in enumerate(scores) if value > 0.0]
    positive.sort(key=lambda item: (-item[1], doc_ids[item[0]]))
    return [(doc_ids[index], value) for index, value in positive[:depth]]


class Pooler:
    """The three baseline weightings over one corpus, all kept sparse."""

    def __init__(self, token_lists: list[list[str]]):
        self.vocabulary = build_vocabulary(token_lists)
        self.vocabulary_size = len(self.vocabulary)
        self.counts = count_matrix(token_lists, self.vocabulary).tocsr()
        n_docs, _ = self.counts.shape
        self.df = np.asarray((self.counts > 0).sum(axis=0)).ravel().astype(np.float64)
        self.idf = sublinear_idf(self.df, n_docs)
        self.bm25_idf_values = bm25_idf(self.df, n_docs)
        self.row = row_of_each_nonzero(self.counts)
        self.columns = self.counts.indices.astype(np.int64)
        frequency = self.counts.data

        # 1. tfidf cosine
        tfidf_values = (1.0 + np.log(frequency)) * self.idf[self.columns]
        self.tfidf = l2_normalize(self._csr(tfidf_values))

        # 2. binary cosine, presence only, no idf
        self.binary = l2_normalize(self._csr(np.ones_like(frequency)))

        # 3. BM25, worked out on the stored values only
        doc_lengths = np.asarray(self.counts.sum(axis=1)).ravel()
        average = float(doc_lengths.mean()) if n_docs else 0.0
        length_per_entry = doc_lengths[self.row]
        normalisation = 1.0 - BM25_B + BM25_B * (length_per_entry / (average or 1.0))
        denominator = frequency + BM25_K1 * normalisation
        bm25_values = self.bm25_idf_values[self.columns] * (
            frequency * (BM25_K1 + 1.0) / denominator)
        self.bm25 = self._csr(bm25_values)

    def _csr(self, values: np.ndarray) -> sparse.csr_matrix:
        return sparse.csr_matrix((values, self.counts.indices.copy(), self.counts.indptr.copy()),
                                 shape=self.counts.shape)

    def tfidf_query(self, tokens):
        return query_vector(self.vocabulary, tokens, self.vocabulary_size,
                            lambda qtf, index: (1.0 + np.log(qtf)) * self.idf[index], True)

    def binary_query(self, tokens):
        return query_vector(self.vocabulary, tokens, self.vocabulary_size,
                            lambda qtf, index: 1.0, True)

    def bm25_query(self, tokens):
        # The document matrix already carries IDF(t), so each unique query term
        # contributes a single unit. Standard BM25 ignores query term frequency.
        return query_vector(self.vocabulary, tokens, self.vocabulary_size,
                            lambda qtf, index: 1.0, False)

    @staticmethod
    def _scores(matrix, vector) -> np.ndarray:
        if vector.nnz == 0:
            return np.zeros(matrix.shape[0])
        return np.asarray((matrix @ vector.T).todense()).ravel()

    def rank_tfidf(self, tokens, doc_ids, depth):
        return rank(self._scores(self.tfidf, self.tfidf_query(tokens)), doc_ids, depth)

    def rank_binary(self, tokens, doc_ids, depth):
        return rank(self._scores(self.binary, self.binary_query(tokens)), doc_ids, depth)

    def rank_bm25(self, tokens, doc_ids, depth):
        return rank(self._scores(self.bm25, self.bm25_query(tokens)), doc_ids, depth)


def check_search_engine_parity(documents: list[dict], pooler: Pooler, doc_ids: list[str],
                               queries_payload: dict, depth: int,
                               required: bool = True, tokenizer=baseline_tokenize) -> dict:
    """The tfidf_cosine baseline must equal the public SearchEngine.

    Retrieval is a separate deliverable. When SearchEngine cannot be imported the
    check is reported as skipped; when it can, a mismatch is a hard failure
    unless required is False.
    """
    try:
        from engine.indexer.search_engine import SearchEngine
    except Exception as error:  # noqa: BLE001
        result = {"status": "skipped", "reason": f"{type(error).__name__}: {error}"}
        if required:
            raise ParityError(f"SearchEngine parity is required but unavailable: {result['reason']}")
        return result
    engine = SearchEngine(tokenizer=tokenizer, stopwords=None,
                          metadata={"purpose": "pool tfidf parity"})
    engine.fit(documents)
    checked = 0
    worst = 0.0
    for query in queries_payload["queries"]:
        expected = pooler.rank_tfidf(tokenizer(query["text"]), doc_ids, depth)
        produced = engine.search(query["text"], top_k=depth)
        got = [(item["doc_id"], float(item["score"])) for item in produced]
        if [doc for doc, _ in expected] != [doc for doc, _ in got]:
            mismatch = ParityError(
                f"{query['query_id']}: ranking differs from SearchEngine; "
                f"pool={[d for d, _ in expected][:5]} engine={[d for d, _ in got][:5]}")
            if required:
                raise mismatch
            return {"status": "failed", "error": str(mismatch)}
        worst = max(worst, max((abs(a[1] - b[1]) for a, b in zip(expected, got)), default=0.0))
        checked += 1
    if worst > SCORE_TOLERANCE:
        mismatch = ParityError(f"max score delta {worst} exceeds {SCORE_TOLERANCE}")
        if required:
            raise mismatch
        return {"status": "failed", "error": str(mismatch)}
    return {"status": "passed", "engine": "engine.indexer.search_engine",
            "queries_checked": checked, "depth": depth,
            "max_score_delta": worst, "tolerance": SCORE_TOLERANCE}


def build_pool(corpus_path: Path = DEFAULT_CORPUS, manifest_path: Path | None = DEFAULT_MANIFEST,
               queries_path: Path = DEFAULT_QUERIES, pool_path: Path = DEFAULT_POOL,
               stats_path: Path = DEFAULT_STATS, depth: int = POOL_DEPTH,
               embed_content: bool = True, require_parity: bool = True,
               tokenizer=baseline_tokenize, tokenizer_metadata: dict | None = None) -> dict:
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise PoolError("pooling depth must be a positive integer")
    documents, corpus_sha = load_corpus(corpus_path, manifest_path)
    by_id = {document["doc_id"]: document for document in documents}
    doc_ids = [document["doc_id"] for document in documents]
    queries_sha = hashlib.sha256(Path(queries_path).read_bytes()).hexdigest()
    queries_payload = load_json(queries_path)
    if queries_payload.get("corpus_sha256") != corpus_sha:
        raise PoolError("queries.json corpus_sha256 does not match the corpus")

    token_lists = [tokenizer(document["content"]) for document in documents]
    pooler = Pooler(token_lists)
    parity = check_search_engine_parity(documents, pooler, doc_ids, queries_payload,
                                        depth, required=require_parity, tokenizer=tokenizer)

    entries = []
    retrieved = Counter()
    sizes = []
    for query in queries_payload["queries"]:
        tokens = tokenizer(query["text"])
        rankings = {
            "tfidf_cosine": pooler.rank_tfidf(tokens, doc_ids, depth),
            "binary_cosine": pooler.rank_binary(tokens, doc_ids, depth),
            "bm25": pooler.rank_bm25(tokens, doc_ids, depth),
        }
        provenance: dict[str, dict] = {}
        for name in BASELINES:
            results = rankings[name]
            retrieved[name] += len(results)
            for position, (doc_id, score) in enumerate(results, start=1):
                slot = provenance.setdefault(doc_id, {"ranks": {}, "scores": {}})
                slot["ranks"][name] = position
                slot["scores"][name] = round(float(score), 9)
        union = sorted(provenance, key=lambda value: (min(provenance[value]["ranks"].values()),
                                                      value))
        sizes.append(len(union))
        for doc_id in union:
            document = by_id[doc_id]
            entry = {
                "query_id": query["query_id"],
                "doc_id": doc_id,
                "title": document["title"],
                "category": document["category"],
                "source_file": document["metadata"]["source_file"],
                "retrieved_by": sorted(provenance[doc_id]["ranks"]),
                "best_rank": min(provenance[doc_id]["ranks"].values()),
                "ranks": provenance[doc_id]["ranks"],
                "scores": provenance[doc_id]["scores"],
                "status": "pending_human_annotation",
            }
            if embed_content:
                entry["content"] = document["content"]
            entries.append(entry)

    pool = {
        "schema_version": SCHEMA_VERSION,
        "corpus_sha256": corpus_sha,
        "queries_sha256": queries_sha,
        "queries_file": Path(queries_path).name,
        "status": "pending_human_annotation",
        "pooling": {
            "method": f"union of Top-{depth} from three baselines",
            "depth_per_baseline": depth,
            "baselines": list(BASELINES),
            "tokenizer": (tokenizer_metadata or {}).get("tokenizer", TOKENIZER_SPEC),
            "tokenizer_status": (tokenizer_metadata or {}).get("status", "development_only"),
            "preprocessing": tokenizer_metadata,
            "indexed_field": "content",
            "tie_break": "ascending doc_id",
            "empty_or_oov_query_returns": [],
            "definitions": {
                "tfidf_cosine": {"tf": "1+ln(f)", "idf": "ln((N+1)/(df+1))+1",
                                 "doc_norm": "L2", "query_norm": "L2", "idf_in_query": True},
                "binary_cosine": {"tf": "1 when f>0 else 0", "idf": "none",
                                  "doc_norm": "L2", "query_norm": "L2",
                                  "note": "presence-only cosine, deliberately no idf"},
                "bm25": {"idf": "ln(1+(N-df+0.5)/(df+0.5))", "k1": BM25_K1, "b": BM25_B,
                         "doc_norm": "none", "query_norm": "none",
                         "qtf": "each unique query term contributes weight 1.0",
                         "idf_placement": "document side only; query weights are not idf"},
            },
        },
        "search_engine_parity": parity,
        "queries": [
            {"query_id": query["query_id"], "text": query["text"],
             "category": query["category"], "types": query["types"],
             "pooled_documents": sum(1 for e in entries if e["query_id"] == query["query_id"])}
            for query in queries_payload["queries"]
        ],
        "pairs": entries,
    }
    stats = {
        "schema_version": SCHEMA_VERSION,
        "corpus_sha256": corpus_sha,
        "queries_sha256": queries_sha,
        "vocabulary_size": pooler.vocabulary_size,
        "pooled_pairs": len(entries),
        "queries": len(queries_payload["queries"]),
        "depth_per_baseline": depth,
        "pool_size_min": min(sizes),
        "pool_size_max": max(sizes),
        "pool_size_mean": round(sum(sizes) / len(sizes), 2),
        "max_possible_pool_size": 3 * depth,
        "queries_with_maximal_pool": sum(1 for size in sizes if size == 3 * depth),
        "retrieved_counts": {name: retrieved[name] for name in sorted(retrieved)},
        "documents_in_pool": len({entry["doc_id"] for entry in entries}),
        "pairs_by_number_of_baselines": dict(sorted(Counter(
            len(entry["retrieved_by"]) for entry in entries).items())),
        "search_engine_parity": parity,
        "label_status": "pending_human_annotation",
        "notes": [
            "A pooled pair is a candidate only. Relevance must be judged per query.",
            "Category is never a substitute for a relevance judgment.",
            "Evidence ids in queries.json are authoring support, not relevance.",
        ],
    }
    previous_qrels = Path(pool_path).parent / "qrels.json"
    if Path(pool_path).exists() and previous_qrels.exists():
        previous_pool = load_json(pool_path)
        old_pairs = {(p["query_id"], p["doc_id"]) for p in previous_pool["pairs"]}
        new_pairs = {(p["query_id"], p["doc_id"]) for p in entries}
        if (old_pairs != new_pairs or any(previous_pool.get(k) != pool.get(k)
                                         for k in ("corpus_sha256", "queries_sha256"))):
            raise PoolError("Pool already has qrels and its locked pairs/meaning would change. "
                            "Use new --pool and --stats paths, then migrate compatible judgments.")
    Path(pool_path).parent.mkdir(parents=True, exist_ok=True)
    Path(pool_path).write_text(json.dumps(pool, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    Path(stats_path).write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    return {"pool": pool, "stats": stats}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--depth", type=int, default=POOL_DEPTH)
    parser.add_argument("--development-baseline", action="store_true",
                        help="Use the old syllable baseline, explicitly development only")
    parser.add_argument("--no-content", action="store_true")
    parser.add_argument("--allow-missing-search-engine", action="store_true",
                        help="Record the SearchEngine parity check as skipped instead of failing")
    args = parser.parse_args(argv)
    try:
        from engine.preprocessing import M1_RESOURCES, M1_TOKENIZER, M1_TIER1
        processor = Preprocessor() if args.development_baseline else Preprocessor(
            M1_TOKENIZER, M1_TIER1, False, True, M1_RESOURCES)
        result = build_pool(corpus_path=args.corpus, manifest_path=args.manifest,
                            queries_path=args.queries, pool_path=args.pool,
                            stats_path=args.stats, depth=args.depth,
                            embed_content=not args.no_content,
                            require_parity=not args.allow_missing_search_engine,
                            tokenizer=processor, tokenizer_metadata=processor.metadata())
    except PoolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    stats = result["stats"]
    print(f"pooled {stats['pooled_pairs']} pairs for {stats['queries']} queries "
          f"(min {stats['pool_size_min']}, max {stats['pool_size_max']}, "
          f"mean {stats['pool_size_mean']})")
    print(f"SearchEngine parity: {stats['search_engine_parity']['status']}")
    print(f"wrote {args.pool} and {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
