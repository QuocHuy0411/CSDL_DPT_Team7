"""Tests for three-baseline pooling, qrels handling and the offline annotation tool."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.preprocessing import baseline_tokenize  # noqa: E402
from scripts import annotate as ann  # noqa: E402
from scripts.build_pool import (  # noqa: E402
    BASELINES,
    BM25_B,
    BM25_K1,
    POOL_DEPTH,
    ParityError,
    Pooler,
    PoolError,
    bm25_idf,
    build_pool,
    build_vocabulary,
    count_matrix,
    rank,
    sublinear_idf,
)

CORPUS = ROOT / "data/corpus.jsonl"
MANIFEST = ROOT / "data/corpus_manifest.json"
QUERIES = ROOT / "data/queries.json"
POOL = ROOT / "data/ground_truth/pool.json"
POOL_STATS = ROOT / "data/ground_truth/pool_stats.json"
QRELS = ROOT / "data/ground_truth/qrels.json"
SAMPLE = ROOT / "data/ground_truth/m1_review_sample.json"

SAMPLE_DOCS = [
    "mỡ trong máu cao gây xơ vữa động mạch và bệnh tim mạch ở người bệnh",
    "bệnh nhân tiểu đường cần chế độ ăn hợp lý và tập thể dục đều đặn",
    "cholesterol toàn phần cao thì phải kiểm soát lipid máu thường xuyên",
    "xơ vữa mạch máu nguy hiểm với người bệnh tim mạch trẻ tuổi",
]
SAMPLE_QUERY = "mỡ trong máu cao gây biến chứng tim mạch"


def tokenize(documents):
    return [baseline_tokenize(document) for document in documents]


# --------------------------------------------------------------------------- #
# sparse construction
# --------------------------------------------------------------------------- #

def test_idf_row_shapes_are_valid_sparse():
    tokens = tokenize(SAMPLE_DOCS)
    vocabulary = build_vocabulary(tokens)
    counts = count_matrix(tokens, vocabulary)
    pooler = Pooler(tokens)
    assert counts.shape == (4, len(vocabulary))
    # a wrong (columns, columns) row index would raise or produce nnz != V
    for matrix in (pooler.tfidf, pooler.binary, pooler.bm25):
        assert matrix.shape[0] == 4
        assert matrix.shape[1] == len(vocabulary)
        assert matrix.nnz == counts.nnz


def test_l2_normalize_gives_unit_rows_and_keeps_sparsity():
    tokens = tokenize(SAMPLE_DOCS)
    pooler = Pooler(tokens)
    norms = np.sqrt(np.asarray(pooler.tfidf.multiply(pooler.tfidf).sum(axis=1)).ravel())
    assert np.allclose(norms, 1.0, atol=1e-12)
    assert pooler.tfidf.nnz <= pooler.counts.nnz


def test_sublinear_and_bm25_idf_definitions():
    df = np.array([1.0, 3.0])
    assert np.allclose(sublinear_idf(df, 10), np.log((10 + 1) / (df + 1)) + 1)
    assert np.allclose(bm25_idf(df, 10), np.log(1 + (10 - df + 0.5) / (df + 0.5)))


# --------------------------------------------------------------------------- #
# sklearn parity
# --------------------------------------------------------------------------- #

def test_tfidf_baseline_matches_sklearn_including_the_query():
    sklearn = pytest.importorskip("sklearn.feature_extraction.text")
    vectorizer = sklearn.TfidfVectorizer(analyzer=baseline_tokenize, sublinear_tf=True,
                                         smooth_idf=True, norm="l2", use_idf=True)
    reference = sparse.csr_matrix(vectorizer.fit_transform(SAMPLE_DOCS))
    pooler = Pooler(tokenize(SAMPLE_DOCS))
    positions = dict(vectorizer.vocabulary_)
    inverse = {index: term for term, index in pooler.vocabulary.items()}
    order = [positions[inverse[index]] for index in range(pooler.vocabulary_size)]
    assert abs(reference[:, order] - pooler.tfidf).max() < 1e-12

    query_reference = sparse.csr_matrix(vectorizer.transform([SAMPLE_QUERY]))[:, order]
    query_mine = pooler.tfidf_query(baseline_tokenize(SAMPLE_QUERY))
    assert abs(query_reference - query_mine).max() < 1e-12
    assert query_mine.nnz > 0


def test_query_vector_applies_corpus_idf_not_raw_tf():
    pooler = Pooler(tokenize(SAMPLE_DOCS))
    tokens = [term for term in baseline_tokenize(SAMPLE_QUERY)
              if term in pooler.vocabulary]
    assert tokens, "sample query must contain in-vocabulary terms"
    query = pooler.tfidf_query(baseline_tokenize(SAMPLE_QUERY)).toarray().ravel()
    raw = np.array([(1.0 + np.log(baseline_tokenize(SAMPLE_QUERY).count(term)))
                    * pooler.idf[pooler.vocabulary[term]] for term in tokens])
    expected = raw / np.linalg.norm(raw)
    got = np.array([query[pooler.vocabulary[term]] for term in tokens])
    assert np.allclose(got, expected, atol=1e-12)
    # a query built from tf alone (no idf) would differ from the idf-weighted one
    plain = np.array([1.0 + np.log(baseline_tokenize(SAMPLE_QUERY).count(term))
                      for term in tokens])
    plain = plain / np.linalg.norm(plain)
    assert abs(plain - expected).max() > 1e-6


def test_binary_cosine_has_no_idf_and_is_distinct_from_tfidf():
    pooler = Pooler(tokenize(SAMPLE_DOCS))
    assert pooler.binary.nnz == pooler.counts.nnz
    assert abs(pooler.binary - pooler.tfidf).max() > 0.1


# --------------------------------------------------------------------------- #
# BM25 against a hand written reference
# --------------------------------------------------------------------------- #

def hand_bm25(doc_tokens, query_tokens, k1=BM25_K1, b=BM25_B):
    vocabulary = build_vocabulary(doc_tokens)
    n_docs = len(doc_tokens)
    df = {}
    for tokens in doc_tokens:
        for term in set(tokens):
            if term in vocabulary:
                df[term] = df.get(term, 0) + 1
    lengths = [len([t for t in tokens if t in vocabulary]) for tokens in doc_tokens]
    average = sum(lengths) / n_docs
    scores = []
    for row in range(n_docs):
        tf = {}
        for term in doc_tokens[row]:
            if term in vocabulary:
                tf[term] = tf.get(term, 0) + 1
        total = 0.0
        for term in set(query_tokens):          # unique terms only, no query tf
            if term not in vocabulary:
                continue
            idf = np.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            frequency = tf.get(term, 0.0)
            total += idf * (frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * lengths[row] / average)))
        scores.append(total)
    return np.array(scores)


def test_bm25_matches_the_hand_reference_and_does_not_square_idf():
    doc_tokens = tokenize(SAMPLE_DOCS)
    pooler = Pooler(doc_tokens)
    query_tokens = baseline_tokenize(SAMPLE_QUERY)
    produced = pooler._scores(pooler.bm25, pooler.bm25_query(query_tokens))
    assert abs(produced - hand_bm25(doc_tokens, query_tokens)).max() < 1e-12
    # a squared idf would inflate every score well beyond the reference
    assert produced.max() < hand_bm25(doc_tokens, query_tokens).max() * 1.001


def test_bm25_query_ignores_query_term_frequency():
    pooler = Pooler(tokenize(SAMPLE_DOCS))
    once = pooler._scores(pooler.bm25, pooler.bm25_query(baseline_tokenize("máu")))
    twice = pooler._scores(pooler.bm25, pooler.bm25_query(baseline_tokenize("máu máu máu")))
    assert np.allclose(once, twice)


def hand_binary(doc_tokens, query_tokens):
    vocabulary = build_vocabulary(doc_tokens)
    unique = {t for t in query_tokens if t in vocabulary}
    out = []
    for tokens in doc_tokens:
        present = {t for t in tokens if t in vocabulary}
        denominator = np.sqrt(len(present) * len(unique))
        out.append(len(present & unique) / denominator if denominator else 0.0)
    return np.array(out)


def test_binary_cosine_matches_a_hand_presence_cosine():
    doc_tokens = tokenize(SAMPLE_DOCS)
    pooler = Pooler(doc_tokens)
    query_tokens = baseline_tokenize(SAMPLE_QUERY)
    produced = pooler._scores(pooler.binary, pooler.binary_query(query_tokens))
    assert abs(produced - hand_binary(doc_tokens, query_tokens)).max() < 1e-12


# --------------------------------------------------------------------------- #
# ranking rules
# --------------------------------------------------------------------------- #

def test_rank_drops_zero_scores_and_breaks_ties_by_doc_id():
    doc_ids = ["d3", "d1", "d2", "d0"]
    scores = np.array([0.5, 0.5, 0.0, -0.2])
    assert rank(scores, doc_ids, 10) == [("d1", 0.5), ("d3", 0.5)]


def test_rank_respects_depth():
    doc_ids = [f"d{i}" for i in range(50)]
    scores = np.linspace(1.0, 0.1, 50)
    assert len(rank(scores, doc_ids, 20)) == 20


def test_empty_and_oov_queries_return_nothing():
    pooler = Pooler(tokenize(SAMPLE_DOCS))
    doc_ids = [f"d{i}" for i in range(4)]
    assert pooler.rank_tfidf([], doc_ids, 20) == []
    assert pooler.rank_tfidf(["zzzzz", "qqqqq"], doc_ids, 20) == []
    assert pooler.rank_bm25([], doc_ids, 20) == []


# --------------------------------------------------------------------------- #
# pool structure against the real corpus
# --------------------------------------------------------------------------- #

requires_pool = pytest.mark.skipif(not POOL.is_file(), reason="pool.json has not been built")


@pytest.fixture(scope="module")
def pool_payload():
    return json.loads(POOL.read_text(encoding="utf-8"))


@requires_pool
def test_pool_matches_the_locked_corpus_and_query_hashes(pool_payload):
    corpus_sha = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    queries_sha = hashlib.sha256(QUERIES.read_bytes()).hexdigest()
    assert pool_payload["corpus_sha256"] == corpus_sha
    assert pool_payload["queries_sha256"] == queries_sha


@requires_pool
def test_every_pair_is_pending_and_carries_three_baseline_provenance(pool_payload):
    assert pool_payload["status"] == "pending_human_annotation"
    for pair in pool_payload["pairs"]:
        assert pair["status"] == "pending_human_annotation"
        assert pair["retrieved_by"]
        assert set(pair["ranks"]) <= set(BASELINES)
        assert set(pair["scores"]) == set(pair["ranks"])
        assert pair["best_rank"] == min(pair["ranks"].values())
        assert "content" in pair and pair["content"]


@requires_pool
def test_pool_depth_is_twenty_per_baseline(pool_payload):
    for query in pool_payload["queries"]:
        rows = [pair for pair in pool_payload["pairs"]
                if pair["query_id"] == query["query_id"]]
        for name in BASELINES:
            depths = [pair["ranks"][name] for pair in rows if name in pair["ranks"]]
            assert len(depths) <= POOL_DEPTH
            assert max(depths) == len(depths)  # ranks are contiguous from 1


@requires_pool
def test_pool_documents_exist_in_the_corpus(pool_payload):
    known = {json.loads(line)["doc_id"] for line in
             CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert {pair["doc_id"] for pair in pool_payload["pairs"]} <= known


@requires_pool
def test_pool_stats_agree_with_the_pool_file(pool_payload):
    stats = json.loads(POOL_STATS.read_text(encoding="utf-8"))
    assert stats["pooled_pairs"] == len(pool_payload["pairs"])
    assert stats["search_engine_parity"]["status"] == "passed"


@requires_pool
def test_queries_are_grounded_and_balanced():
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    assert queries["frozen"] is True
    assert len(queries["queries"]) == 20
    known = {json.loads(line)["doc_id"] for line in
             CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()}
    categories = {}
    for query in queries["queries"]:
        assert query["evidence_doc_ids"]
        assert set(query["evidence_doc_ids"]) <= known
        categories.setdefault(query["category"], []).append(query)
    assert len(categories) == 10
    assert {len(value) for value in categories.values()} == {2}
    kinds = {kind for query in queries["queries"] for kind in query["types"]}
    assert kinds == {"short", "long", "negation", "technical"}


# --------------------------------------------------------------------------- #
# annotation round trip on temporary fixtures
# --------------------------------------------------------------------------- #

def make_pool_fixture(tmp_path: Path, corpus_sha="a" * 64, queries_sha="b" * 64) -> Path:
    pairs = []
    for query_id in ("Q01", "Q02"):
        for index in range(4):
            pairs.append({
                "query_id": query_id,
                "doc_id": f"doc{index}",
                "title": f"Tiêu đề {index}",
                "category": "Khoa học",
                "source_file": f"Khoa hoc/{index}.txt",
                "retrieved_by": ["tfidf_cosine"],
                "best_rank": index + 1,
                "ranks": {"tfidf_cosine": index + 1},
                "scores": {"tfidf_cosine": 0.5 - index / 10},
                "status": "pending_human_annotation",
                "content": f"Nội dung bài viết số {index} về mỡ máu và tim mạch.",
            })
    pool = {
        "schema_version": 1, "corpus_sha256": corpus_sha, "queries_sha256": queries_sha,
        "status": "pending_human_annotation", "pooling": {"baselines": list(BASELINES)},
        "queries": [{"query_id": "Q01", "text": "mỡ máu", "category": "Khoa học",
                     "types": ["short"], "pooled_documents": 4},
                    {"query_id": "Q02", "text": "tim mạch", "category": "Khoa học",
                     "types": ["long"], "pooled_documents": 4}],
        "pairs": pairs,
    }
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    return path


def run_init(pool_path: Path, qrels_path: Path, sample_path: Path, *extra):
    return ann.main(["--pool", str(pool_path), "--qrels", str(qrels_path),
                     "--sample", str(sample_path), "init", *extra])


def test_init_creates_all_pending_labels_and_the_m1_sample(tmp_path):
    pool_path = make_pool_fixture(tmp_path)
    qrels_path = tmp_path / "qrels.json"
    sample_path = tmp_path / "sample.json"
    assert run_init(pool_path, qrels_path, sample_path) == 0
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    assert qrels["status"] == ann.PENDING
    assert qrels["queries_sha256"] == "b" * 64
    assert len(qrels["judgments"]) == 8
    assert all(row["status"] == "pending" and row["relevance"] is None
               and row["assessor"] is None for row in qrels["judgments"])
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    assert sample["seed"] == 2026
    assert sample["sample_fraction"] == 0.20
    assert sample["sample_size"] == 2  # round(8 * 0.2)


def test_m1_sample_is_deterministic_and_twenty_percent(tmp_path):
    pool = json.loads(make_pool_fixture(tmp_path).read_text(encoding="utf-8"))
    first = ann.build_m1_sample(pool["pairs"])
    second = ann.build_m1_sample(pool["pairs"])
    assert first == second
    assert first["sample_size"] == round(len(pool["pairs"]) * 0.20)
    assert len(first["pairs"]) == first["sample_size"]


def test_init_refuses_to_mix_corpora(tmp_path):
    pool_path = make_pool_fixture(tmp_path)
    qrels_path = tmp_path / "qrels.json"
    assert run_init(pool_path, qrels_path, tmp_path / "s.json") == 0
    other = make_pool_fixture(tmp_path, corpus_sha="c" * 64)
    assert run_init(other, qrels_path, tmp_path / "s.json") == 2
    assert json.loads(qrels_path.read_text(encoding="utf-8"))["corpus_sha256"] == "a" * 64


def test_init_refuses_to_mix_query_sets(tmp_path):
    pool_path = make_pool_fixture(tmp_path)
    qrels_path = tmp_path / "qrels.json"
    assert run_init(pool_path, qrels_path, tmp_path / "s.json") == 0
    stale = make_pool_fixture(tmp_path, queries_sha="e" * 64)
    assert run_init(stale, qrels_path, tmp_path / "s.json") == 2
    assert json.loads(qrels_path.read_text(encoding="utf-8"))["queries_sha256"] == "b" * 64


def test_export_writes_offline_html_with_full_text_and_hashes(tmp_path):
    pool_path = make_pool_fixture(tmp_path)
    qrels_path = tmp_path / "qrels.json"
    sheet = tmp_path / "sheet.html"
    assert run_init(pool_path, qrels_path, tmp_path / "s.json") == 0
    assert ann.main(["--pool", str(pool_path), "--qrels", str(qrels_path),
                     "--sheet", str(sheet), "export", "--assessor", "Lan"]) == 0
    document = sheet.read_text(encoding="utf-8")
    assert document.count("<section") == 8
    assert document.count("<select data-qid=") == 8
    assert "queryFilter" in document and "localStorage" in document
    assert document.count("<details") == 8
    assert "Nội dung bài viết số 3" in document      # full article is available
    assert "%s" not in document
    assert "http://" not in document and "https://" not in document
    assert '"a" * 64'.replace("'", "") not in document
    assert "a" * 64 in document and "b" * 64 in document


def test_import_round_trip_marks_labels_confirmed(tmp_path):
    pool_path = make_pool_fixture(tmp_path)
    qrels_path = tmp_path / "qrels.json"
    assert run_init(pool_path, qrels_path, tmp_path / "s.json") == 0
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({
        "corpus_sha256": "a" * 64, "queries_sha256": "b" * 64, "assessor": "Lan",
        "labels": [
            {"query_id": "Q01", "doc_id": "doc0", "relevance": 2},
            {"query_id": "Q01", "doc_id": "doc1", "relevance": 0},
        ]}), encoding="utf-8")
    assert ann.main(["--pool", str(pool_path), "--qrels", str(qrels_path),
                     "import", "--input", str(labels)]) == 0
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    assert qrels["status"] == ann.PENDING  # still pending overall
    assert qrels["annotation_progress"]["confirmed"] == 2
    graded = {(row["query_id"], row["doc_id"]): row for row in qrels["judgments"]}
    assert graded[("Q01", "doc0")]["relevance"] == 2
    assert graded[("Q01", "doc0")]["assessor"] == "Lan"   # global assessor applied
    assert graded[("Q01", "doc1")]["relevance"] == 0
    assert graded[("Q02", "doc0")]["relevance"] is None


def test_import_rejects_stale_corpus_or_query_hashes(tmp_path):
    pool_path = make_pool_fixture(tmp_path)
    qrels_path = tmp_path / "qrels.json"
    assert run_init(pool_path, qrels_path, tmp_path / "s.json") == 0
    labels = tmp_path / "labels.json"

    labels.write_text(json.dumps({
        "corpus_sha256": "z" * 64, "queries_sha256": "b" * 64,
        "labels": [{"query_id": "Q01", "doc_id": "doc0", "relevance": 2,
                    "assessor": "Lan"}]}), encoding="utf-8")
    assert ann.main(["--pool", str(pool_path), "--qrels", str(qrels_path),
                     "import", "--input", str(labels)]) == 2

    labels.write_text(json.dumps({
        "corpus_sha256": "a" * 64, "queries_sha256": "z" * 64,
        "labels": [{"query_id": "Q01", "doc_id": "doc0", "relevance": 2,
                    "assessor": "Lan"}]}), encoding="utf-8")
    assert ann.main(["--pool", str(pool_path), "--qrels", str(qrels_path),
                     "import", "--input", str(labels)]) == 2

    labels.write_text(json.dumps(
        {"labels": [{"query_id": "Q01", "doc_id": "doc0", "relevance": 2,
                     "assessor": "Lan"}]}), encoding="utf-8")
    assert ann.main(["--pool", str(pool_path), "--qrels", str(qrels_path),
                     "import", "--input", str(labels)]) == 2

    # A legacy bare list must not bypass the version hashes required above.
    labels.write_text(json.dumps([
        {"query_id": "Q01", "doc_id": "doc0", "relevance": 2, "assessor": "Lan"}
    ]), encoding="utf-8")
    assert ann.main(["--pool", str(pool_path), "--qrels", str(qrels_path),
                     "import", "--input", str(labels)]) == 2

    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    assert qrels["annotation_progress"]["confirmed"] == 0
    assert all(row["relevance"] is None for row in qrels["judgments"])


def test_import_rejects_missing_assessor_and_bad_grades(tmp_path):
    pool_path = make_pool_fixture(tmp_path)
    qrels_path = tmp_path / "qrels.json"
    assert run_init(pool_path, qrels_path, tmp_path / "s.json") == 0
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({
        "corpus_sha256": "a" * 64, "queries_sha256": "b" * 64,
        "labels": [
            {"query_id": "Q01", "doc_id": "doc0", "relevance": 2},
            {"query_id": "Q01", "doc_id": "doc1", "relevance": 7, "assessor": "Lan"},
            {"query_id": "Q99", "doc_id": "docX", "relevance": 1, "assessor": "Lan"},
        ]}), encoding="utf-8")
    assert ann.main(["--pool", str(pool_path), "--qrels", str(qrels_path),
                     "import", "--input", str(labels)]) == 0
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    assert qrels["annotation_progress"]["confirmed"] == 0
    assert all(row["relevance"] is None for row in qrels["judgments"])


def test_confirm_is_refused_while_labels_are_pending(tmp_path):
    pool_path = make_pool_fixture(tmp_path)
    qrels_path = tmp_path / "qrels.json"
    assert run_init(pool_path, qrels_path, tmp_path / "s.json") == 0
    assert ann.main(["--pool", str(pool_path), "--qrels", str(qrels_path), "confirm"]) == 2
    assert json.loads(qrels_path.read_text(encoding="utf-8"))["status"] == ann.PENDING


def test_confirm_succeeds_only_when_every_pair_is_labeled(tmp_path):
    pool_path = make_pool_fixture(tmp_path)
    qrels_path = tmp_path / "qrels.json"
    assert run_init(pool_path, qrels_path, tmp_path / "s.json") == 0
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    labels = [{"query_id": row["query_id"], "doc_id": row["doc_id"],
               "relevance": 0} for row in qrels["judgments"]]
    labels_path = tmp_path / "all.json"
    labels_path.write_text(json.dumps({"corpus_sha256": "a" * 64,
                                       "queries_sha256": "b" * 64,
                                       "assessor": "Lan", "labels": labels}), encoding="utf-8")
    assert ann.main(["--pool", str(pool_path), "--qrels", str(qrels_path),
                     "import", "--input", str(labels_path)]) == 0
    assert ann.main(["--pool", str(pool_path), "--qrels", str(qrels_path), "confirm"]) == 0
    assert json.loads(qrels_path.read_text(encoding="utf-8"))["status"] == ann.COMPLETE


def test_status_and_review_run_on_a_fresh_fixture(tmp_path):
    pool_path = make_pool_fixture(tmp_path)
    qrels_path = tmp_path / "qrels.json"
    sample_path = tmp_path / "s.json"
    assert run_init(pool_path, qrels_path, sample_path) == 0
    assert ann.main(["--pool", str(pool_path), "--qrels", str(qrels_path),
                     "status"]) == 0
    assert ann.main(["--pool", str(pool_path), "--qrels", str(qrels_path),
                     "--sample", str(sample_path), "review"]) == 0


def test_qrels_and_pool_hash_mismatch_is_rejected(tmp_path):
    pool_path = make_pool_fixture(tmp_path)
    qrels_path = tmp_path / "qrels.json"
    assert run_init(pool_path, qrels_path, tmp_path / "s.json") == 0
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    pool["corpus_sha256"] = "d" * 64
    pool_path.write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    assert ann.main(["--pool", str(pool_path), "--qrels", str(qrels_path), "status"]) == 2


def test_queries_hash_mismatch_between_pool_and_qrels_is_rejected(tmp_path):
    pool_path = make_pool_fixture(tmp_path)
    qrels_path = tmp_path / "qrels.json"
    assert run_init(pool_path, qrels_path, tmp_path / "s.json") == 0
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    qrels["queries_sha256"] = "f" * 64
    qrels_path.write_text(json.dumps(qrels, ensure_ascii=False), encoding="utf-8")
    assert ann.main(["--pool", str(pool_path), "--qrels", str(qrels_path), "status"]) == 2


def test_pool_refuses_queries_from_another_corpus(tmp_path):
    documents = [{"doc_id": f"vn_{i:016x}", "title": f"T{i}", "content": text,
                  "category": "Khoa học",
                  "metadata": {"source_file": f"Khoa hoc/{i}.txt"}} for i, text
                 in enumerate(SAMPLE_DOCS)]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in documents),
                      encoding="utf-8")
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"corpus_sha256": digest}), encoding="utf-8")
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps({"corpus_sha256": "0" * 64,
                                   "queries": [{"query_id": "Q01", "text": "máu",
                                                "category": "Khoa học",
                                                "types": ["short"]}]}), encoding="utf-8")
    with pytest.raises(PoolError):
        build_pool(corpus_path=corpus, manifest_path=manifest, queries_path=queries,
                   pool_path=tmp_path / "p.json", stats_path=tmp_path / "s.json",
                   require_parity=False)


def test_search_engine_parity_is_enforced_when_requested(tmp_path, monkeypatch):
    import scripts.build_pool as pool_module

    class BrokenEngine:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, documents):
            return self

        def search(self, query, top_k=10):
            return []

    monkeypatch.setitem(sys.modules, "engine.indexer.search_engine",
                        type("m", (), {"SearchEngine": BrokenEngine}))
    documents = [{"doc_id": f"vn_{i:016x}", "title": f"T{i}", "content": text,
                  "category": "Khoa học",
                  "metadata": {"source_file": f"Khoa hoc/{i}.txt"}} for i, text
                 in enumerate(SAMPLE_DOCS)]
    doc_ids = [d["doc_id"] for d in documents]
    pooler = Pooler(tokenize(SAMPLE_DOCS))
    payload = {"queries": [{"query_id": "Q01", "text": SAMPLE_QUERY}]}
    with pytest.raises(ParityError):
        pool_module.check_search_engine_parity(documents, pooler, doc_ids, payload,
                                               20, required=True)
    assert pool_module.check_search_engine_parity(
        documents, pooler, doc_ids, payload, 20, required=False)["status"] == "failed"


requires_qrels = pytest.mark.skipif(not QRELS.is_file(), reason="qrels.json has not been created")


@requires_qrels
def test_real_qrels_are_all_pending():
    qrels = json.loads(QRELS.read_text(encoding="utf-8"))
    assert qrels["status"] == ann.PENDING
    assert qrels["corpus_sha256"] == hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    assert all(row["status"] == "pending" and row["relevance"] is None
               for row in qrels["judgments"])


@requires_qrels
def test_real_m1_sample_is_twenty_percent_of_the_pool():
    qrels = json.loads(QRELS.read_text(encoding="utf-8"))
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    assert sample["seed"] == 2026
    assert sample["sample_size"] == round(len(qrels["judgments"]) * 0.20)
    flagged = [row for row in qrels["judgments"] if row["in_m1_sample"]]
    assert len(flagged) == sample["sample_size"]
