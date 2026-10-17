"""Regression tests for versioned pooling and preservation of human judgments."""

import copy
import json

import pytest

from scripts.annotate import AnnotationError, build_m1_sample, new_qrels
from scripts.migrate_qrels import migrate


def fixture_paths(tmp_path):
    old = {"schema_version": 1, "corpus_sha256": "a" * 64, "queries_sha256": "b" * 64,
           "queries": [{"query_id": "Q1", "text": "fixture"}],
           "pairs": [{"query_id": "Q1", "doc_id": "D1", "content": "same document",
                      "category": "fixture", "retrieved_by": ["tfidf_cosine"], "best_rank": 1}]}
    new = copy.deepcopy(old)
    new["pairs"].append({**new["pairs"][0], "doc_id": "D2", "content": "new document"})
    qrels = new_qrels(old, build_m1_sample(old["pairs"]), "fixture assessor")
    qrels["judgments"][0].update(relevance=2, status="confirmed", assessor="real fixture name")
    paths = [tmp_path / name for name in ("old_pool.json", "old_qrels.json", "new_pool.json",
                                         "new_qrels.json", "sample.json")]
    for path, payload in zip(paths, (old, qrels, new)):
        path.write_text(json.dumps(payload), encoding="utf-8")
    return paths


def test_migration_preserves_common_confirmed_and_source_bytes(tmp_path):
    paths = fixture_paths(tmp_path)
    source_bytes = paths[1].read_bytes()
    assert migrate(*paths)["carried_confirmed"] == 1
    out = json.loads(paths[3].read_text(encoding="utf-8"))
    assert out["judgments"][0]["relevance"] == 2
    assert out["judgments"][0]["assessor"] == "real fixture name"
    assert out["judgments"][1]["relevance"] is None
    assert paths[1].read_bytes() == source_bytes
    assert migrate(*paths)["carried_confirmed"] == 0


@pytest.mark.parametrize("field,value", [("corpus_sha256", "changed"),
                                         ("queries_sha256", "changed"), ("schema_version", 99)])
def test_migration_rejects_incompatible_versions(tmp_path, field, value):
    paths = fixture_paths(tmp_path)
    pool = json.loads(paths[2].read_text())
    pool[field] = value
    paths[2].write_text(json.dumps(pool))
    with pytest.raises(AnnotationError):
        migrate(*paths)
    assert not paths[3].exists()


@pytest.mark.parametrize("mutation", ["content", "query", "conflict"])
def test_migration_does_not_trust_claimed_hash_or_overwrite_conflicts(tmp_path, mutation):
    paths = fixture_paths(tmp_path)
    if mutation == "conflict":
        migrate(*paths)
        payload = json.loads(paths[3].read_text())
        payload["judgments"][0]["relevance"] = 0
        paths[3].write_text(json.dumps(payload))
        saved = paths[3].read_bytes()
    else:
        payload = json.loads(paths[2].read_text())
        if mutation == "content":
            payload["pairs"][0]["content"] = "altered"
        else:
            payload["queries"][0]["text"] = "altered"
        paths[2].write_text(json.dumps(payload))
    with pytest.raises(AnnotationError):
        migrate(*paths)
    if mutation == "conflict":
        assert paths[3].read_bytes() == saved


@pytest.mark.parametrize("grade", [True, 1.5, "1"])
def test_qrels_loader_refuses_lossy_grade_coercion(tmp_path, grade):
    from evaluator.qrels import load_qrels
    path = tmp_path / "qrels.json"
    path.write_text(json.dumps({"judgments": [{"query_id": "Q1", "doc_id": "D1",
                                              "relevance": grade}]}))
    with pytest.raises(ValueError, match="integer"):
        load_qrels(path)
