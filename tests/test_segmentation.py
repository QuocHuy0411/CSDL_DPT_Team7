"""Segmentation metrics, human gates and atomic annotation import (temporary fixtures)."""

import hashlib
import json
from pathlib import Path
import re
import shutil

import pytest

from evaluator.segmentation_eval import (DEFAULT_DATA, SegmentationError, evaluate,
                                          jsonl, load_dataset, scores, span_counts, word_spans)
from scripts.annotate_segmentation import export as export_gold, import_labels


def test_word_span_metric_is_not_syllable_accuracy():
    assert span_counts("học sinh học", ["học_sinh", "học"], ["học", "sinh_học"]) == (0, 2, 2)
    assert span_counts("học sinh học", ["học_sinh", "học"], ["học_sinh", "học"]) == (2, 2, 2)
    assert scores(1, 2, 4)["f1"] == pytest.approx(1 / 3)
    assert scores(0, 0, 0)["f1"] == 0


@pytest.mark.parametrize("tokens", [["học_sinh"], ["sinh", "học"], ["học", "sinh", "thêm"],
                                     ["học__sinh"], ["học sinh"], "học sinh", [True]])
def test_spans_reject_missing_reordered_invalid_or_boundary_crossing_words(tokens):
    with pytest.raises(SegmentationError):
        word_spans("học, sinh", tokens)


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "gold"
    path.mkdir()
    for name in ("segmentation_manifest.json", "segmentation_sample.jsonl"):
        shutil.copy2(DEFAULT_DATA / name, path / name)
    records = jsonl(DEFAULT_DATA / "segmentation_sample.jsonl")
    pending = [{"sample_id": r["sample_id"], "sentence_sha256": r["sentence_sha256"],
                "tokens": None, "assessor": None, "label_status": "pending_human_annotation"}
               for r in records]
    (path / "segmentation_labels.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pending), encoding="utf-8")
    return path


def label_payload(dataset):
    manifest, records, _ = load_dataset(dataset)
    record = records[0]
    return {**{k: manifest[k] for k in ("schema_version", "corpus_sha256", "sample_sha256", "normalizer_sha256")},
            "human_confirmed": True, "assessor": "fixture human (not real labels)",
            "labels": [{"sample_id": record["sample_id"], "sentence_sha256": record["sentence_sha256"],
                        "tokens": record["normalized_text"].split()}]}


def test_real_sample_is_balanced_sourced_and_official_quality_still_gated(dataset, tmp_path):
    from collections import Counter
    _, records, _ = load_dataset(dataset)
    assert len(records) == 150 and set(Counter(r["category"] for r in records).values()) == {15}
    payload = evaluate(dataset, tmp_path / "results")
    assert payload["metrics_status"] == "unmeasured" and "metrics" not in payload
    assert payload["gold_labels_sha256"] == hashlib.sha256(
        (dataset / "segmentation_labels.jsonl").read_bytes()).hexdigest()


@pytest.mark.parametrize("bad", ["corpus_sha256", "sample_sha256", "normalizer_sha256", "human_confirmed", "tokens", "assessor", "duplicate", "non_object"])
def test_import_rejects_stale_automatic_or_malformed_labels_atomically(dataset, tmp_path, bad):
    payload = label_payload(dataset)
    if bad.endswith("sha256"):
        payload[bad] = "changed"
    elif bad == "human_confirmed":
        payload[bad] = False
    elif bad == "tokens":
        payload["labels"][0]["tokens"] = ["lost_syllables"]
    elif bad == "assessor":
        payload[bad] = ""
    elif bad == "non_object":
        payload["labels"][0] = None
    else:
        payload["labels"].append(payload["labels"][0])
    source = tmp_path / "import.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    before = (dataset / "segmentation_labels.jsonl").read_bytes()
    with pytest.raises(SegmentationError):
        import_labels(dataset, source)
    assert (dataset / "segmentation_labels.jsonl").read_bytes() == before


def test_focused_gold_export_hides_old_gold_and_machine_suggestion(dataset, tmp_path):
    payload = label_payload(dataset)
    source = tmp_path / "reviewed.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    assert import_labels(dataset, source) == 1
    out = tmp_path / "focused.html"
    assert export_gold(dataset, out, ["SG001", "SG010"]) == 2
    html = out.read_text(encoding="utf-8")
    assert html.count("<section>") == 2
    assert "Phiếu rà soát riêng không hiển thị nhãn cũ hoặc gợi ý máy" in html
    assert "<details>" not in html
    assert "<textarea data-id='SG001'" in html
    assert "<textarea data-id='SG010'" in html
    assert re.findall(r"<textarea[^>]+>(.*?)</textarea>", html) == ["", ""]
    assert "review:SG001,SG010" in html


def test_import_preserves_other_labels_and_one_label_does_not_unlock_official(dataset, tmp_path):
    payload = label_payload(dataset)
    source = tmp_path / "labels.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    assert import_labels(dataset, source) == 1
    _, _, labels = load_dataset(dataset)
    assert sum(r["label_status"] == "confirmed" for r in labels.values()) == 1
    result = evaluate(dataset, tmp_path / "results")
    assert not result["official"] and "metrics" not in result


def test_complete_fixture_gold_measures_all_methods_and_categories_only_in_temp(dataset, tmp_path):
    manifest, records, _ = load_dataset(dataset)
    payload = label_payload(dataset)
    payload["labels"] = [{"sample_id": r["sample_id"], "sentence_sha256": r["sentence_sha256"],
                           "tokens": r["normalized_text"].split()} for r in records]
    source = tmp_path / "fixture_labels.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    assert import_labels(dataset, source) == 150
    result = evaluate(dataset, tmp_path / "results")
    assert result["official"]
    assert result["gold_labels_sha256"] == hashlib.sha256(
        (dataset / "segmentation_labels.jsonl").read_bytes()).hexdigest()
    assert set(result["metrics"]) == {"fmm", "bmm", "bimm"}
    assert all(len(groups) == 10 for groups in result["by_category"].values())
    assert all(0 < row["f1"] < 1 for row in result["metrics"].values())
