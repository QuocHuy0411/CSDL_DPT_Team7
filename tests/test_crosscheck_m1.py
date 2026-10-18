"""The M1 second-reviewer workflow stays blind and never rewrites qrels."""

import hashlib
import json

import pytest

from scripts.annotate import build_m1_sample
from scripts.crosscheck_m1 import CrosscheckError, compare, export_sheet


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def review_paths(tmp_path):
    payload = {
        "schema_version": 1, "corpus_sha256": "corpus", "queries_sha256": "queries",
        "queries": [{"query_id": "Q1", "text": "tình hình nghiên cứu"}],
        "pairs": [
            {"query_id": "Q1", "doc_id": "D1", "title": "Bài một", "content": "Toàn văn một."},
            {"query_id": "Q1", "doc_id": "D2", "title": "Bài hai", "content": "Toàn văn hai."},
        ],
    }
    pool = _json(tmp_path / "pool.json", payload)
    sample = _json(tmp_path / "sample.json", build_m1_sample(payload["pairs"], fraction=1.0))
    qrels = _json(tmp_path / "qrels.json", {
        "schema_version": 1, "corpus_sha256": "corpus", "queries_sha256": "queries",
        "judgments": [
            {"query_id": "Q1", "doc_id": "D1", "relevance": 2,
             "status": "confirmed", "assessor": "Original"},
            {"query_id": "Q1", "doc_id": "D2", "relevance": 0,
             "status": "confirmed", "assessor": "Original"},
        ],
    })
    return pool, sample, qrels


def _second(sample, grades=(2, 1), reviewer="M1 Person"):
    return {
        "schema_version": 1, "corpus_sha256": "corpus",
        "queries_sha256": "queries",
        "sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
        "reviewer_role": "M1", "reviewer": reviewer,
        "independent_review_confirmed": True,
        "labels": [{"query_id": "Q1", "doc_id": doc, "relevance": grade}
                   for doc, grade in zip(("D1", "D2"), grades)],
    }


def test_blind_sheet_has_full_text_but_no_original_grades(review_paths, tmp_path):
    pool, sample, _ = review_paths
    out = tmp_path / "sheet.html"
    assert export_sheet(pool, sample, out) == 2
    body = out.read_text(encoding="utf-8")
    assert "Toàn văn một." in body and "Toàn văn hai." in body
    assert "tình hình nghiên cứu" in body
    assert "Original" not in body
    assert "<option value=''>-- chọn --</option>" in body
    assert "independent_review_confirmed:true" in body


def test_compare_reports_disagreement_without_changing_qrels(review_paths, tmp_path):
    pool, sample, qrels = review_paths
    before = qrels.read_bytes()
    labels = _json(tmp_path / "m1.json", _second(sample))
    result = compare(pool, sample, qrels, labels)
    assert result["sample_size"] == 2
    assert result["agreements"] == 1
    assert result["observed_agreement"] == 0.5
    assert result["grade_confusion_matrix"]["0"]["1"] == 1
    assert result["disagreements"] == [{
        "query_id": "Q1", "doc_id": "D2", "original_grade": 0,
        "m1_grade": 1, "resolution": "pending_human_adjudication",
    }]
    assert qrels.read_bytes() == before


@pytest.mark.parametrize("change", [
    lambda p: p.update(reviewer="Original"),
    lambda p: p.update(independent_review_confirmed=False),
    lambda p: p.update(independent_review_confirmed=1),
    lambda p: p["labels"].pop(),
    lambda p: p["labels"][0].update(relevance=True),
    lambda p: p.update(sample_sha256="stale"),
    lambda p: p["labels"][1].update(doc_id="D1"),
])
def test_compare_rejects_unreliable_input(review_paths, tmp_path, change):
    pool, sample, qrels = review_paths
    payload = _second(sample)
    change(payload)
    labels = _json(tmp_path / "m1.json", payload)
    with pytest.raises(CrosscheckError):
        compare(pool, sample, qrels, labels)


def test_compare_accepts_full_agreement(review_paths, tmp_path):
    pool, sample, qrels = review_paths
    labels = _json(tmp_path / "m1.json", _second(sample, grades=(2, 0)))
    result = compare(pool, sample, qrels, labels)
    assert result["status"] == "agreement_recorded"
    assert result["disagreement_count"] == 0


def test_export_rejects_changed_deterministic_sample(review_paths, tmp_path):
    pool, sample, _ = review_paths
    payload = json.loads(sample.read_text(encoding="utf-8"))
    payload["pairs"].reverse()
    _json(sample, payload)
    with pytest.raises(CrosscheckError, match="deterministic"):
        export_sheet(pool, sample, tmp_path / "sheet.html")


def test_compare_rejects_duplicate_qrels(review_paths, tmp_path):
    pool, sample, qrels = review_paths
    payload = json.loads(qrels.read_text(encoding="utf-8"))
    payload["judgments"].append(payload["judgments"][0].copy())
    _json(qrels, payload)
    labels = _json(tmp_path / "m1.json", _second(sample))
    with pytest.raises(CrosscheckError, match="duplicate"):
        compare(pool, sample, qrels, labels)
