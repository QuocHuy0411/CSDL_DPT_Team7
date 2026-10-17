"""Kiểm thử bộ đánh giá: công thức chỉ số, cổng chặn chính thức và chống rò fixture.

Bộ test này dùng fixture trong thư mục tạm của pytest. Số liệu fixture chỉ phục
vụ kiểm thử và không bao giờ được ghi vào kết quả chính thức — có test riêng cho
điều đó (`test_*` với `results/ablation`).
"""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator import metrics  # noqa: E402
from evaluator.evaluate import (  # noqa: E402
    OFFICIAL_RESULT_DIRECTORIES,
    ResultTargetRefused,
    TruncatedRankingRefused,
    assert_result_target,
    check_ranking_depth,
    measure_index_bytes,
    metrics_readiness,
    resolve_corpus_sha256,
)
from evaluator.metrics import MAP_SCOPE  # noqa: E402
from evaluator.qrels import (  # noqa: E402
    ALLOWED_RELEVANCE_GRADES,
    load_queries,
    QRELS_CONFIRMED_STATUS,
    SUPPORTED_SCHEMA_VERSION,
    Judgment,
    OfficialEvaluationRefused,
    Qrels,
    QrelsQueryHashMismatch,
    QuerySet,
    load_qrels,
    preprocessing_status,
    validate_official_run,
)

CORPUS_HASH = "f" * 64
QUERIES_HASH = "a" * 64


# --------------------------------------------------------------- chỉ số cơ bản


def test_precision_at_k_uses_k_as_denominator_even_with_short_results():
    assert metrics.precision_at_k(["a", "b"], {"a"}, 10) == pytest.approx(1 / 10)
    assert metrics.precision_at_k(["a"], {"a"}, 1) == pytest.approx(1.0)
    assert metrics.precision_at_k([], {"a"}, 10) == 0.0
    with pytest.raises(ValueError):
        metrics.precision_at_k(["a"], {"a"}, 0)


def test_recall_uses_confirmed_relevant_total_from_qrels_not_returned():
    assert metrics.recall_at_k(["a", "b"], {"a", "b", "c", "d"}, 2) == pytest.approx(0.5)
    assert metrics.recall_at_k(["a", "b", "c", "d"], {"a", "b", "c", "d"}, 2) == 0.5
    assert metrics.recall_at_k(["a"], set(), 10) == 0.0


def test_f1_is_harmonic_mean():
    assert metrics.f1_at_k(1.0, 1.0) == pytest.approx(1.0)
    assert metrics.f1_at_k(0.5, 1.0) == pytest.approx(2 * 0.5 / 1.5)
    assert metrics.f1_at_k(0.0, 0.0) == 0.0
    assert metrics.f1_at_k(0.0, 0.7) == 0.0


def test_average_precision_uses_full_ranking_not_top_k():
    # Relevant ở vị trí 1 và 3; AP = (1/1 + 2/3) / 2
    ranking = ["r1", "x", "r2", "x", "x"]
    assert metrics.average_precision(ranking, {"r1", "r2"}) == pytest.approx((1.0 + 2 / 3) / 2)
    # Cắt Top-2 sẽ cho kết quả khác hẳn, chứng minh tầm quan trọng
    assert metrics.average_precision(ranking[:2], {"r1", "r2"}) == pytest.approx(0.5)
    assert metrics.average_precision(ranking, set()) == 0.0


def test_ndcg_uses_exponential_gain_and_qrels_ideal():
    ranking = ["d1", "d2", "d3"]
    judgments = {"d1": 2, "d2": 0, "d3": 1}
    dcg = 3.0 + 0.0 + 1.0 / metrics.math.log2(4)
    idcg = 3.0 + 1.0 / metrics.math.log2(3) + 0.0
    assert metrics.ndcg_at_k(ranking, judgments, 3) == pytest.approx(dcg / idcg)
    # IDCG lấy từ qrels, không lấy từ lần chạy: bỏ d2 (gain 0) không đổi kết quả
    assert metrics.ndcg_at_k(["d1", "d3"], judgments, 2) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(ranking, {"d1": 0}, 3) == 0.0


def test_relevant_documents_threshold():
    assert metrics.relevant_documents({"a": 0, "b": 1, "c": 2}) == frozenset({"b", "c"})


def test_macro_average_keeps_zero_relevant_queries_and_reports_them():
    rows = [
        {"query_id": "Q1", "k": 10, "precision_at_k": 1.0, "recall_at_k": 1.0,
         "f1_at_k": 1.0, "average_precision": 1.0, "ndcg_at_k": 1.0, "zero_relevant": False},
        {"query_id": "Q2", "k": 10, "precision_at_k": 0.0, "recall_at_k": 0.0,
         "f1_at_k": 0.0, "average_precision": 0.0, "ndcg_at_k": 0.0, "zero_relevant": True},
    ]
    summary = metrics.macro_average(rows)
    assert summary["map"] == pytest.approx(0.5)
    assert summary["macro_precision_at_k"] == pytest.approx(0.5)
    assert summary["n_queries"] == 2
    assert summary["n_zero_relevant_queries"] == 1
    assert summary["zero_relevant_query_ids"] == "Q2"
    assert summary["map_scope"] == MAP_SCOPE


def test_map_name_does_not_claim_a_cut():
    summary = metrics.macro_average([
        {"query_id": "Q1", "k": 10, "precision_at_k": 0.1, "recall_at_k": 0.1,
         "f1_at_k": 0.1, "average_precision": 0.1, "ndcg_at_k": 0.1, "zero_relevant": False}])
    assert "map_at_k" not in summary
    assert summary["map_scope"] == MAP_SCOPE
    assert not any("ap_at_k" in column for column in metrics.PER_QUERY_COLUMNS)


def test_percentile_uses_fraction_not_percent():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert metrics.percentile(values, 0.95) == pytest.approx(4.8)
    with pytest.raises(ValueError):
        metrics.percentile(values, 95.0)


def test_summarize_pairs_groups_mean_and_p95():
    summary = metrics.summarize_pairs([("C1", "a", 1.0), ("C1", "b", 3.0), ("C2", "c", 5.0)])
    assert summary["C1"][0] == pytest.approx(2.0)
    assert summary["C1"][1] == pytest.approx(metrics.percentile([1.0, 3.0], 0.95))
    assert summary["C2"][0] == pytest.approx(5.0)


def test_empty_ranking_is_a_valid_zero_not_an_error():
    row = metrics.query_metrics([], {"d1": 1}, 10)
    assert row["precision_at_k"] == 0.0
    assert row["recall_at_k"] == 0.0
    assert row["average_precision"] == 0.0
    assert row["n_returned"] == 0
    assert row["n_relevant"] == 1
    assert row["zero_relevant"] is False


# ------------------------------------------------------- cổng chặn chính thức


def _query_set(n=20, corpus=CORPUS_HASH, schema=SUPPORTED_SCHEMA_VERSION,
               queries_hash=QUERIES_HASH):
    queries = tuple({"query_id": f"Q{i:02d}", "text": f"truy vấn {i}",
                     "category": "Lĩnh vực", "types": ["short"]} for i in range(n))
    return QuerySet(schema_version=schema, corpus_sha256=corpus, queries=queries,
                    file_sha256=queries_hash, queries_sha256=queries_hash)


def _qrels(n=20, corpus=CORPUS_HASH, status=QRELS_CONFIRMED_STATUS, relevance=1,
           judge="human_a", judgment_status="confirmed", schema=SUPPORTED_SCHEMA_VERSION,
           queries_hash=QUERIES_HASH):
    judgments = tuple(Judgment(f"Q{i:02d}", f"D{i}", relevance, judgment_status, judge)
                      for i in range(n))
    return Qrels(schema_version=schema, corpus_sha256=corpus, status=status,
                 judgments=judgments, source_path="fixture",
                 queries_sha256=queries_hash)


def _official_index(**overrides):
    metadata = {
        "configuration": "C1",
        "corpus_sha256": CORPUS_HASH,
        "preprocessing_status": "official",
        "official_preprocessing": True,
        "preprocessing": {"status": "official", "tokenizer": "engine.m1:tokenize",
                          "tier1_filter": "engine.m1:tier1", "tier1_enabled": False,
                          "m1_verified": True, "fingerprint": "abc"},
    }
    metadata.update(overrides)
    return metadata


def test_official_run_is_allowed_when_everything_is_clean():
    report = validate_official_run(_qrels(), _query_set(), corpus_sha256=CORPUS_HASH,
                                   index_metadata=_official_index())
    assert report.official_allowed, report.issues
    assert report.raise_if_blocked() is None


@pytest.mark.parametrize("status", ["pending_human_annotation", "unknown"])
def test_pending_qrels_status_blocks_official(status):
    report = validate_official_run(_qrels(status=status), _query_set(),
                                   corpus_sha256=CORPUS_HASH,
                                   index_metadata=_official_index())
    assert not report.official_allowed
    assert "qrels_status_not_final" in report.codes


def test_pending_judgments_block_official():
    qrels = _qrels(judgment_status="pending", relevance=None)
    report = validate_official_run(qrels, _query_set(), corpus_sha256=CORPUS_HASH,
                                   index_metadata=_official_index())
    assert "pending_judgments" in report.codes


def test_confirmed_without_relevance_blocks_official():
    qrels = _qrels(relevance=None)
    report = validate_official_run(qrels, _query_set(), corpus_sha256=CORPUS_HASH,
                                   index_metadata=_official_index())
    assert "confirmed_without_relevance" in report.codes
    assert "query_without_confirmed_judgments" in report.codes


def test_fewer_than_twenty_queries_blocks_official():
    report = validate_official_run(_qrels(n=5), _query_set(n=5), corpus_sha256=CORPUS_HASH,
                                   index_metadata=_official_index())
    assert "insufficient_queries" in report.codes


def test_corpus_hash_mismatch_blocks_official():
    report = validate_official_run(_qrels(corpus="a" * 64), _query_set(),
                                   corpus_sha256=CORPUS_HASH,
                                   index_metadata=_official_index())
    assert "qrels_corpus_hash_missing" not in report.codes
    assert "qrels_corpus_hash_mismatch" in report.codes
    index_report = validate_official_run(_qrels(), _query_set(), corpus_sha256=CORPUS_HASH,
                                         index_metadata=_official_index(corpus_sha256="b" * 64))
    assert "index_corpus_hash_mismatch" in index_report.codes


def test_missing_corpus_hash_blocks_official():
    report = validate_official_run(_qrels(), _query_set(), corpus_sha256=None,
                                   index_metadata=_official_index())
    assert "expected_corpus_hash_missing" in report.codes
    index_report = validate_official_run(_qrels(), _query_set(), corpus_sha256=CORPUS_HASH,
                                         index_metadata=_official_index(corpus_sha256=None))
    assert "index_corpus_hash_missing" in index_report.codes


def test_development_tokenizer_blocks_even_when_metadata_claims_official():
    metadata = _official_index(
        preprocessing={"status": "official", "tokenizer": "engine.preprocessing:baseline_tokenize",
                       "tier1_filter": "engine.m1:tier1", "tier1_enabled": False,
                       "m1_verified": True, "fingerprint": "abc"},
        tokenizer="engine.preprocessing:baseline_tokenize")
    report = validate_official_run(_qrels(), _query_set(), corpus_sha256=CORPUS_HASH,
                                   index_metadata=metadata)
    assert "development_tokenizer" in report.codes
    assert not report.official_allowed


def test_fixture_tier1_filter_blocks_even_when_metadata_claims_official():
    metadata = _official_index(
        preprocessing={"status": "official", "tokenizer": "engine.m1:tokenize",
                       "tier1_filter": "engine.preprocessing:fixture_tier1",
                       "tier1_enabled": True, "m1_verified": True, "fingerprint": "abc"},
        tier1_filter="engine.preprocessing:fixture_tier1")
    report = validate_official_run(_qrels(), _query_set(), corpus_sha256=CORPUS_HASH,
                                   index_metadata=metadata)
    assert "fixture_tier1_filter" in report.codes


def test_official_flag_requires_consistent_status_and_flag():
    only_status = _official_index(
        official_preprocessing=False,
        preprocessing={"status": "official", "tokenizer": "engine.m1:tokenize",
                       "tier1_filter": "engine.m1:tier1", "tier1_enabled": False,
                       "m1_verified": True, "fingerprint": "abc"})
    only_flag = _official_index(preprocessing_status="development_only",
                                official_preprocessing=True)
    neither = _official_index(preprocessing_status="development_only",
                              official_preprocessing=False)
    assert preprocessing_status(only_status)["official"] is False
    assert preprocessing_status(only_flag)["official"] is False
    assert preprocessing_status(neither)["official"] is False
    assert preprocessing_status(_official_index())["official"] is True
    report = validate_official_run(_qrels(), _query_set(), corpus_sha256=CORPUS_HASH,
                                   index_metadata=only_status)
    assert "preprocessing_metadata_inconsistent" in report.codes


def test_unknown_schema_version_blocks_official():
    report = validate_official_run(_qrels(schema=2), _query_set(), corpus_sha256=CORPUS_HASH,
                                   index_metadata=_official_index())
    assert "qrels_schema_version_unsupported" in report.codes
    query_report = validate_official_run(_qrels(), _query_set(schema=None),
                                         corpus_sha256=CORPUS_HASH,
                                         index_metadata=_official_index())
    assert "queries_schema_version_unsupported" in query_report.codes


def test_duplicate_judgments_block_official():
    qrels = Qrels(schema_version=1, corpus_sha256=CORPUS_HASH, status=QRELS_CONFIRMED_STATUS,
                  judgments=(_query_pair(), _query_pair()), queries_sha256=QUERIES_HASH)
    report = validate_official_run(qrels, _query_set(n=1), corpus_sha256=CORPUS_HASH,
                                   index_metadata=_official_index())
    assert "duplicate_judgments" in report.codes


def _query_pair():
    return Judgment("Q00", "D0", 1, "confirmed", "human_a")


def test_invalid_relevance_grade_blocks_official():
    qrels = _qrels(n=1, relevance=7)
    report = validate_official_run(qrels, _query_set(n=1), corpus_sha256=CORPUS_HASH,
                                   index_metadata=_official_index())
    assert "invalid_relevance_grade" in report.codes
    assert 7 not in ALLOWED_RELEVANCE_GRADES


def test_confirmed_without_assessor_blocks_official():
    qrels = _qrels(n=1, judge="")
    report = validate_official_run(qrels, _query_set(n=1), corpus_sha256=CORPUS_HASH,
                                   index_metadata=_official_index())
    assert "confirmed_without_assessor" in report.codes


def test_unknown_query_id_in_qrels_blocks_official():
    judgments = (Judgment("Q99", "D0", 1, "confirmed", "human_a"),)
    qrels = Qrels(schema_version=1, corpus_sha256=CORPUS_HASH, status=QRELS_CONFIRMED_STATUS,
                  judgments=judgments, queries_sha256=QUERIES_HASH)
    report = validate_official_run(qrels, _query_set(n=1), corpus_sha256=CORPUS_HASH,
                                   index_metadata=_official_index())
    assert "unknown_query_id_in_qrels" in report.codes


def test_raise_if_blocked_lists_every_issue():
    report = validate_official_run(_qrels(n=3), _query_set(n=3), corpus_sha256=None,
                                   index_metadata=None)
    with pytest.raises(OfficialEvaluationRefused) as error:
        report.raise_if_blocked()
    message = str(error.value)
    assert "insufficient_queries" in message
    assert "expected_corpus_hash_missing" in message


# ------------------------------------------------- khoá hash của bộ truy vấn


def test_qrels_queries_hash_mismatch_blocks_official():
    report = validate_official_run(_qrels(queries_hash="b" * 64), _query_set(),
                                   corpus_sha256=CORPUS_HASH,
                                   index_metadata=_official_index())
    assert "queries_hash_mismatch" in report.codes
    detail = next(issue for issue in report.issues if issue.code == "queries_hash_mismatch")
    assert detail.detail["qrels"] == "b" * 64
    assert detail.detail["actual"] == QUERIES_HASH


def test_missing_qrels_queries_hash_blocks_official():
    report = validate_official_run(_qrels(queries_hash=None), _query_set(),
                                   corpus_sha256=CORPUS_HASH,
                                   index_metadata=_official_index())
    assert "qrels_queries_hash_missing" in report.codes


def test_unavailable_queries_hash_blocks_official():
    queries = _query_set()
    unhashed = QuerySet(schema_version=queries.schema_version,
                        corpus_sha256=queries.corpus_sha256, queries=queries.queries,
                        file_sha256=None, queries_sha256=None)
    report = validate_official_run(_qrels(), unhashed, corpus_sha256=CORPUS_HASH,
                                   index_metadata=_official_index())
    assert "queries_hash_unavailable" in report.codes


def test_empty_qrels_only_warns_about_missing_queries_hash():
    empty = Qrels(schema_version=1, corpus_sha256=CORPUS_HASH, status=QRELS_CONFIRMED_STATUS,
                  judgments=(), queries_sha256=None)
    report = validate_official_run(empty, _query_set(), corpus_sha256=CORPUS_HASH,
                                   index_metadata=_official_index())
    assert "qrels_queries_hash_missing" not in report.codes
    assert "qrels_empty" in [issue.code for issue in report.warnings]
    assert "query_without_confirmed_judgments" in report.codes


def test_load_queries_records_actual_file_digest(tmp_path):
    from evaluator.qrels import file_sha256

    path = tmp_path / "queries.json"
    path.write_text(json.dumps({"schema_version": 1, "corpus_sha256": CORPUS_HASH,
                                "queries": [{"query_id": "Q01", "text": "bóng đá"}]}),
                    encoding="utf-8")
    loaded = load_queries(path)
    assert loaded.file_sha256 == file_sha256(path)
    assert loaded.queries_sha256 is None
    assert loaded.summary()["file_sha256"] == loaded.file_sha256


def test_load_qrels_keeps_queries_sha256(tmp_path):
    path = _write_qrels(tmp_path / "qrels.json", [
        {"query_id": "Q01", "doc_id": "D1", "relevance": 1, "status": "confirmed",
         "assessor": "a"},
    ], queries_hash=QUERIES_HASH)
    loaded = load_qrels(path)
    assert loaded.queries_sha256 == QUERIES_HASH
    assert loaded.summary()["queries_sha256"] == QUERIES_HASH


def test_edited_query_text_with_same_id_is_detected(frozen_inputs):
    """Đổi nội dung truy vấn nhưng giữ nguyên query_id thì qrels phải bị chặn."""
    from evaluator.evaluate import evaluate

    root, _digest = frozen_inputs
    payload = json.loads((root / "queries.json").read_text(encoding="utf-8"))
    payload["queries"][0]["text"] = "văn bản đã sửa sau khi gán nhãn"
    (root / "queries_edited.json").write_text(json.dumps(payload, ensure_ascii=False),
                                              encoding="utf-8")
    with pytest.raises(QrelsQueryHashMismatch, match="khác SHA256 thực tế"):
        evaluate(indexes=root / "indexes", queries_path=root / "queries_edited.json",
                 qrels_path=root / "qrels.json", out=root / "edited_out",
                 manifest_path=root / "manifest.json", corpus_path=root / "corpus.jsonl",
                 allow_development=True)


# ------------------------------------------------- sẵn sàng đo và độ sâu


def test_metrics_readiness_blocks_partial_coverage():
    qrels = _qrels(n=2)
    readiness = metrics_readiness(_query_set(n=2), qrels, ["Q00", "Q01"])
    assert readiness["ready"] is True
    partial = metrics_readiness(_query_set(n=2), qrels, ["Q00", "Q01", "Q02"])
    assert partial["ready"] is False
    assert "Q02" in partial["queries_without_confirmed_labels"]


def test_ranking_depth_must_cover_the_corpus():
    assert check_ranking_depth(2500, 2500, allow_truncated=False)["truncated"] is False
    with pytest.raises(TruncatedRankingRefused):
        check_ranking_depth(100, 2500, allow_truncated=False)
    with pytest.raises(TruncatedRankingRefused):
        check_ranking_depth(0, 2500, allow_truncated=False)
    allowed = check_ranking_depth(100, 2500, allow_truncated=True)
    assert allowed["truncated"] is True
    assert "AP@100" in allowed["average_precision_scope"]
    assert allowed["average_precision_scope"] != MAP_SCOPE


def test_resolve_corpus_sha256_prefers_locked_manifest(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"doc_id":"D1"}\n', encoding="utf-8")
    actual = metrics_hash(corpus)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"corpus_sha256": actual}), encoding="utf-8")
    resolved = resolve_corpus_sha256(manifest, corpus)
    assert resolved["corpus_sha256"] == actual
    assert resolved["source"] == "locked_manifest"
    from_file = resolve_corpus_sha256(tmp_path / "missing.json", corpus)
    assert from_file["source"] == "corpus_file_sha256"
    assert resolve_corpus_sha256(tmp_path / "missing.json", tmp_path / "missing.jsonl")[
        "corpus_sha256"] is None


def test_resolve_corpus_sha256_rejects_tampered_corpus(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"doc_id":"D1"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"corpus_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest khóa"):
        resolve_corpus_sha256(manifest, corpus)


def metrics_hash(path):
    from evaluator.qrels import file_sha256
    return file_sha256(path)


# ------------------------------------------------- chống lẫn kết quả fixture


def test_non_official_results_cannot_be_written_to_official_directories(tmp_path):
    official_dir = tmp_path / "results" / "ablation"
    assert assert_result_target(official_dir, official=True) == official_dir.resolve()
    with pytest.raises(ResultTargetRefused):
        assert_result_target(official_dir, official=False)
    for name in OFFICIAL_RESULT_DIRECTORIES:
        with pytest.raises(ResultTargetRefused):
            assert_result_target(tmp_path / "results" / name, official=False)
    allowed = assert_result_target(tmp_path / "results" / "development_eval", official=False)
    assert allowed.name == "development_eval"
    assert assert_result_target(tmp_path / "results" / "fixtures" / "run1", official=False)


def test_measure_index_bytes_separates_serving_from_tdv_analysis(tmp_path):
    directory = tmp_path / "C4"
    directory.mkdir()
    (directory / "inverted_index.npz").write_bytes(b"x" * 1000)
    (directory / "build_metadata.json").write_bytes(b"{}")
    (directory / "tdv_scores.json").write_bytes(b"y" * 400)
    sizes = measure_index_bytes(directory)
    assert sizes["tdv_analysis_bytes"] == 400
    assert sizes["serving_bytes"] == 1002
    assert sizes["total_bytes"] == 1402


# ----------------------------------------------------- nạp qrels thật từ file


def _write_qrels(path, judgments, corpus=CORPUS_HASH,
                 status=QRELS_CONFIRMED_STATUS, schema=SUPPORTED_SCHEMA_VERSION,
                 queries_hash=None):
    payload = {"schema_version": schema, "corpus_sha256": corpus, "status": status,
               "judgments": judgments}
    if queries_hash is not None:
        payload["queries_sha256"] = queries_hash
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_qrels_keeps_pending_and_relevance_none(tmp_path):
    path = _write_qrels(tmp_path / "qrels.json", [
        {"query_id": "Q01", "doc_id": "D1", "relevance": 2, "status": "confirmed",
         "assessor": "a"},
        {"query_id": "Q01", "doc_id": "D2", "relevance": None, "status": "pending",
         "assessor": None},
        {"query_id": "Q01", "doc_id": "D3", "relevance": 0, "status": "confirmed",
         "assessor": "b"},
    ], status="pending_human_annotation")
    qrels = load_qrels(path)
    assert len(qrels.judgments) == 3
    assert len(qrels.pending_judgments()) == 1
    assert qrels.relevance_map("Q01") == {"D1": 2, "D3": 0}
    assert qrels.relevant_map("Q01") == frozenset({"D1"})
    assert qrels.summary()["n_pending"] == 1
    assert qrels.summary()["assessors"] == ["a", "b"]


def test_pending_labels_are_never_used_as_relevance(tmp_path):
    path = _write_qrels(tmp_path / "qrels.json", [
        {"query_id": "Q01", "doc_id": "D1", "relevance": 2, "status": "pending",
         "assessor": None},
    ])
    qrels = load_qrels(path)
    assert qrels.relevant_map("Q01") == frozenset()
    assert qrels.relevance_map("Q01") == {}


# ------------------------------------------------------ kiểm thử end-to-end


@pytest.fixture()
def built_indexes(tmp_path):
    """Bốn chỉ mục C1–C4 thật, dựng bằng chính `scripts.build_indexes`."""
    from scripts.build_indexes import build_indexes

    documents = [{"doc_id": f"D{i}", "title": f"Tài liệu {i}", "content": text,
                  "category": "fixture"} for i, text in enumerate([
                      "và chung bóng đá Việt Nam", "và chung máy tính và bóng đá",
                      "chung thị trường chứng khoán", "chung bóng đá thị trường",
                      "máy tính chứng khoán bóng đá", "và của các chung bóng đá máy tính"])]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents),
                      encoding="utf-8")
    build_indexes(corpus, tmp_path / "indexes")
    return tmp_path, documents


@pytest.fixture()
def frozen_inputs(tmp_path, built_indexes):
    """Bộ query/qrels/manifest đã confirmed đủ để đánh giá chính thức."""
    root, documents = built_indexes
    from evaluator.qrels import file_sha256

    digest = file_sha256(root / "corpus.jsonl")
    (root / "manifest.json").write_text(json.dumps({"corpus_sha256": digest}), encoding="utf-8")
    query_ids = [f"Q{i:02d}" for i in range(20)]
    queries = [{"query_id": query_id, "text": "bóng đá", "category": "Thể thao",
                "types": ["short"]} for query_id in query_ids]
    (root / "queries.json").write_text(json.dumps({
        "schema_version": 1, "corpus_sha256": digest, "queries": queries}, ensure_ascii=False),
        encoding="utf-8")
    judgments = []
    for query_id in query_ids:
        for index in range(3):
            judgments.append({"query_id": query_id, "doc_id": documents[index]["doc_id"],
                              "relevance": 1, "status": "confirmed", "assessor": "fixture_a"})
    (root / "qrels.json").write_text(json.dumps({
        "schema_version": 1, "corpus_sha256": digest,
        "queries_sha256": file_sha256(root / "queries.json"),
        "status": "human_annotation_confirmed", "judgments": judgments}, ensure_ascii=False),
        encoding="utf-8")
    return root, digest


def test_end_to_end_evaluate_refuses_official_on_development_preprocessing(frozen_inputs):
    from evaluator.evaluate import evaluate

    root, _digest = frozen_inputs
    with pytest.raises(OfficialEvaluationRefused) as error:
        evaluate(indexes=root / "indexes", queries_path=root / "queries.json",
                 qrels_path=root / "qrels.json", out=root / "official_out",
                 manifest_path=root / "manifest.json", corpus_path=root / "corpus.jsonl",
                 mode="official")
    assert "preprocessing_not_official" in str(error.value)
    assert not (root / "official_out").exists()


def test_end_to_end_evaluate_measures_four_configurations(frozen_inputs):
    from evaluator.evaluate import FALLBACK_LABEL, evaluate

    root, digest = frozen_inputs
    payload = evaluate(indexes=root / "indexes", queries_path=root / "queries.json",
                       qrels_path=root / "qrels.json", out=root / "ablation_out",
                       manifest_path=root / "manifest.json", corpus_path=root / "corpus.jsonl",
                       k=10, allow_development=True)
    assert payload["metrics_status"] == "measured"
    assert payload["official"] is False
    assert payload["corpus_sha256"] == digest
    assert payload["ranking_depth"] == 6
    assert set(payload["configurations"]) == {"C1", "C2", "C3", "C4", FALLBACK_LABEL}
    for entry in payload["configurations"].values():
        assert entry["summary"]["n_queries"] == 20
        assert entry["summary"]["n_zero_relevant_queries"] == 0
        assert 0.0 <= entry["summary"]["map"] <= 1.0
    per_query = list(csv.DictReader((root / "ablation_out" / "per_query.csv")
                                    .open(encoding="utf-8")))
    assert len(per_query) == 100
    assert {row["configuration"] for row in per_query} == {"C1", "C2", "C3", "C4", FALLBACK_LABEL}
    assert (root / "ablation_out" / "summary_by_category.csv").is_file()
    assert (root / "ablation_out" / "run.json").is_file()


def test_end_to_end_evaluate_refuses_to_leak_into_official_directory(frozen_inputs):
    from evaluator.evaluate import evaluate

    root, _digest = frozen_inputs
    with pytest.raises(ResultTargetRefused):
        evaluate(indexes=root / "indexes", queries_path=root / "queries.json",
                 qrels_path=root / "qrels.json", out=root / "results" / "ablation",
                 manifest_path=root / "manifest.json", corpus_path=root / "corpus.jsonl",
                 allow_development=True)


def test_end_to_end_evaluate_refuses_truncated_depth(frozen_inputs):
    from evaluator.evaluate import evaluate

    root, _digest = frozen_indexes_tuple(frozen_inputs)
    with pytest.raises(TruncatedRankingRefused):
        evaluate(indexes=root / "indexes", queries_path=root / "queries.json",
                 qrels_path=root / "qrels.json", out=root / "truncated_out",
                 manifest_path=root / "manifest.json", corpus_path=root / "corpus.jsonl",
                 ranking_depth=3, allow_development=True)


def frozen_indexes_tuple(frozen_inputs):
    return frozen_inputs


def test_end_to_end_evaluate_keeps_empty_rankings_as_zeros(frozen_inputs, tmp_path):
    from evaluator.evaluate import evaluate
    from engine.preprocessing import Preprocessor

    root, _digest = frozen_inputs
    payload = evaluate(indexes=root / "indexes", queries_path=root / "queries.json",
                       qrels_path=root / "qrels.json", out=root / "empty_out",
                       manifest_path=root / "manifest.json", corpus_path=root / "corpus.jsonl",
                       allow_development=True)
    c4 = payload["configurations"]["C4"]
    assert "n_queries_with_empty_result" in c4["statistics"]
    assert c4["statistics"]["empty_results_are_valid_outcome"] is True
    assert 0 <= c4["summary"]["map"] <= 1
    assert Preprocessor is not None


def test_unmeasured_run_writes_no_metric_csvs(frozen_inputs):
    """Qrels pending -> không được xuất CSV chỉ số toàn số 0."""
    from evaluator.evaluate import evaluate

    root, _digest = frozen_inputs
    qrels_path = root / "pending_qrels.json"
    payload_qrels = json.loads((root / "qrels.json").read_text(encoding="utf-8"))
    payload_qrels["status"] = "pending_human_annotation"
    for judgment in payload_qrels["judgments"]:
        judgment["status"] = "pending"
        judgment["relevance"] = None
    qrels_path.write_text(json.dumps(payload_qrels, ensure_ascii=False), encoding="utf-8")
    payload = evaluate(indexes=root / "indexes", queries_path=root / "queries.json",
                       qrels_path=qrels_path, out=root / "pending_out",
                       manifest_path=root / "manifest.json",
                       corpus_path=root / "corpus.jsonl", allow_development=True)
    assert payload["metrics_status"] == "unmeasured"
    assert payload["official"] is False
    assert not (root / "pending_out" / "per_query.csv").exists()
    assert not (root / "pending_out" / "summary.csv").exists()
    assert (root / "pending_out" / "run.json").is_file()
    recorded = json.loads((root / "pending_out" / "run.json").read_text(encoding="utf-8"))
    assert all(entry["summary"] is None for entry in recorded["configurations"].values())


def test_cli_exit_codes_report_the_state(frozen_inputs):
    from evaluator.evaluate import main

    root, _digest = frozen_inputs
    assert main(["--indexes", str(root / "indexes"), "--queries", str(root / "queries.json"),
                 "--qrels", str(root / "qrels.json"), "--corpus-manifest",
                 str(root / "manifest.json"), "--corpus", str(root / "corpus.jsonl"),
                 "--out", str(root / "cli_out"), "--mode", "official"]) == 2
    assert main(["--indexes", str(root / "indexes"), "--queries", str(root / "queries.json"),
                 "--qrels", str(root / "qrels.json"), "--corpus-manifest",
                 str(root / "manifest.json"), "--corpus", str(root / "corpus.jsonl"),
                 "--out", str(root / "cli_out"), "--allow-development"]) == 0


def test_benchmark_cli_runs_on_built_indexes(frozen_inputs, tmp_path):
    from benchmarks.benchmark_latency import main as benchmark_main

    root, digest = frozen_inputs
    out = root / "latency_out"
    assert benchmark_main(["--indexes", str(root / "indexes"),
                           "--queries", str(root / "queries.json"),
                           "--qrels", str(root / "qrels.json"),
                           "--corpus-manifest", str(root / "manifest.json"),
                           "--repeat", "1", "--warmup", "0", "--out", str(out)]) == 0
    payload = json.loads((out / "latency.json").read_text(encoding="utf-8"))
    assert payload["corpus_sha256"] == digest
    assert set(payload["configurations"]) == {"C1", "C2", "C3", "C4"}
    assert "C4_with_fallback" in payload["deployment_bundles"]
    bundle = payload["deployment_bundles"]["C4_with_fallback"]
    assert bundle["components"] == ["C3", "C4"]
    c3 = payload["configurations"]["C3"]["index_bytes"]["serving_bytes"]
    c4 = payload["configurations"]["C4"]["index_bytes"]["serving_bytes"]
    assert bundle["serving_bytes"] == c3 + c4
    c4 = payload["configurations"]["C4"]
    assert "fallback_measurement" in c4
    assert c4["fallback_measurement"]["fallback_enabled"] is True
    assert payload["metrics_available"] is False
    per_query = list(csv.DictReader((out / "latency_per_query.csv").open(encoding="utf-8")))
    assert len({row["query_id"] for row in per_query}) == 20
    assert all(row["metrics_available"] == "False" for row in per_query)
