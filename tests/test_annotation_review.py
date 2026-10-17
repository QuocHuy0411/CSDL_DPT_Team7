"""A focused semantic review must not inherit old grades or assessor names."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.annotate import AnnotationError, cmd_export, cmd_import
from scripts.annotate_segmentation import export as export_gold
from evaluator.segmentation_eval import DEFAULT_DATA, SegmentationError


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def test_focused_sheet_is_blind_and_contains_only_flagged_pair(tmp_path):
    pool = _write(tmp_path / "pool.json", {
        "corpus_sha256": "corpus", "queries_sha256": "queries",
        "queries": [{"query_id": "Q1", "text": "việc nghiên cứu"}],
        "pairs": [
            {"query_id": "Q1", "doc_id": "D1", "title": "T1", "content": "Toàn bài 1",
             "category": "A", "best_rank": 1, "retrieved_by": ["tfidf"]},
            {"query_id": "Q1", "doc_id": "D2", "title": "T2", "content": "Toàn bài 2",
             "category": "A", "best_rank": 2, "retrieved_by": ["tfidf"]},
        ],
    })
    qrels = _write(tmp_path / "qrels.json", {
        "corpus_sha256": "corpus", "queries_sha256": "queries",
        "judgments": [
            {"query_id": "Q1", "doc_id": "D1", "relevance": 2,
             "assessor": "Original", "status": "confirmed"},
            {"query_id": "Q1", "doc_id": "D2", "relevance": 0,
             "assessor": "Original", "status": "confirmed"},
        ],
    })
    review = _write(tmp_path / "review.json", {
        "qrels_sha256_at_review": hashlib.sha256(qrels.read_bytes()).hexdigest(),
        "qrels_flags": [{"query_id": "Q1", "doc_id": "D1"}],
    })
    sheet = tmp_path / "sheet.html"
    args = SimpleNamespace(pool=pool, qrels=qrels, sheet=sheet, assessor="",
                           review_flags=review)
    assert cmd_export(args) == 0
    body = sheet.read_text(encoding="utf-8")
    assert body.count("<section") == 1
    assert "Toàn bài 1" in body and "Toàn bài 2" not in body
    assert "Original" not in body and "selected='2'" not in body
    assert "id='reviewed'" in body
    assert "review_qrels_sha256" in body
    before = qrels.read_bytes()
    _write(review, {"qrels_sha256_at_review": "stale",
                    "qrels_flags": [{"query_id": "Q1", "doc_id": "D1"}]})
    with pytest.raises(AnnotationError, match="stale"):
        cmd_export(args)
    assert qrels.read_bytes() == before

    payload = {"corpus_sha256": "corpus", "queries_sha256": "queries",
               "review_qrels_sha256": hashlib.sha256(before).hexdigest(),
               "human_review_confirmed": False,
               "labels": [{"query_id": "Q1", "doc_id": "D1", "relevance": 0,
                           "assessor": "Second"}]}
    source = _write(tmp_path / "decisions.json", payload)
    importer = SimpleNamespace(pool=pool, qrels=qrels, input=source, assessor=None)
    with pytest.raises(AnnotationError, match="human-reading"):
        cmd_import(importer)
    assert qrels.read_bytes() == before
    payload["human_review_confirmed"] = True
    payload["review_qrels_sha256"] = "stale"
    _write(source, payload)
    with pytest.raises(AnnotationError, match="older qrels"):
        cmd_import(importer)
    assert qrels.read_bytes() == before
    payload["review_qrels_sha256"] = hashlib.sha256(before).hexdigest()
    _write(source, payload)
    assert cmd_import(importer) == 0
    updated = json.loads(qrels.read_text(encoding="utf-8"))["judgments"]
    assert updated[0]["relevance"] == 0 and updated[0]["assessor"] == "Second"
    assert updated[1]["relevance"] == 0 and updated[1]["assessor"] == "Original"


def test_focused_gold_sheet_is_blank_and_separate(tmp_path):
    out = tmp_path / "gold_review.html"
    assert export_gold(DEFAULT_DATA, out, ["SG010"]) == 1
    body = out.read_text(encoding="utf-8")
    assert body.count("<section>") == 1
    assert "<textarea data-id='SG010'" in body
    assert "độc_đáo_để</textarea>" not in body
    assert "review:SG010" in body
    with pytest.raises(SegmentationError, match="Unknown"):
        export_gold(DEFAULT_DATA, out, ["SG999"])
