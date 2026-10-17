"""Word-span M1 evaluation. Corpus gold must be explicitly confirmed by people."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from engine.preprocessing import ROOT
from engine.tokenizer.max_matching import tokenize
from engine.tokenizer.preprocessor import normalize, normalize_segments

DEFAULT_DATA = ROOT / "data/gold"


class SegmentationError(ValueError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def word_spans(text: str, tokens: list[str]) -> frozenset[tuple[int, int]]:
    """Validate complete ordered syllable coverage; reject words crossing hard boundaries."""
    if not isinstance(tokens, list) or any(not isinstance(t, str) or not t or
            any(c.isspace() for c in t) or any(not s for s in t.split("_")) for t in tokens):
        raise SegmentationError("Tokens must be nonempty words joined with underscores")
    segments = [s.split() for s in normalize_segments(text)]
    expected = [s for segment in segments for s in segment]
    actual = [s for token in tokens for s in token.split("_")]
    if actual != expected:
        raise SegmentationError("Gold tokens must preserve all normalized syllables in order")
    boundaries, end = set(), 0
    for segment in segments[:-1]:
        end += len(segment)
        boundaries.add(end)
    spans, start = set(), 0
    for token in tokens:
        end = start + len(token.split("_"))
        if any(start < b < end for b in boundaries):
            raise SegmentationError("A word crosses punctuation/removed-text boundary")
        spans.add((start, end))
        start = end
    return frozenset(spans)


def span_counts(text, predicted, gold):
    p, g = word_spans(text, predicted), word_spans(text, gold)
    return len(p & g), len(p), len(g)


def scores(correct, predicted, gold):
    precision = correct / predicted if predicted else 0.0
    recall = correct / gold if gold else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "correct_spans": correct, "predicted_spans": predicted, "gold_spans": gold}


def load_dataset(directory=DEFAULT_DATA):
    directory = Path(directory)
    manifest = json.loads((directory / "segmentation_manifest.json").read_text(encoding="utf-8"))
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise SegmentationError("Unsupported segmentation schema")
    if digest(directory / "segmentation_sample.jsonl") != manifest["sample_sha256"]:
        raise SegmentationError("Sample hash mismatch")
    if digest(ROOT / "data/corpus.jsonl") != manifest["corpus_sha256"]:
        raise SegmentationError("Gold corpus is not the current locked corpus")
    locked = json.loads((ROOT / "data/corpus_manifest.json").read_text(encoding="utf-8"))
    if locked["corpus_sha256"] != manifest["corpus_sha256"]:
        raise SegmentationError("Locked corpus manifest mismatch")
    if digest(ROOT / "engine/tokenizer/preprocessor.py") != manifest.get("normalizer_sha256"):
        raise SegmentationError("Normalizer changed; review alignment before evaluating gold")
    records = jsonl(directory / "segmentation_sample.jsonl")
    labels = jsonl(directory / "segmentation_labels.jsonl")
    rows = {r["sample_id"]: r for r in records}
    judgments = {r["sample_id"]: r for r in labels}
    if len(rows) != len(records) or len(judgments) != len(labels) or set(rows) != set(judgments):
        raise SegmentationError("Duplicate or mismatching sample/label IDs")
    documents = {r["doc_id"]: r for r in jsonl(ROOT / "data/corpus.jsonl")}
    for sample_id, record in rows.items():
        text = record["sentence_text"]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != record["sentence_sha256"]:
            raise SegmentationError("Sentence hash mismatch")
        document = documents.get(record["doc_id"])
        if not document or text not in document["content"] or record["category"] != document["category"]:
            raise SegmentationError("Sentence source/category mismatch")
        if normalize(text) != record.get("normalized_text"):
            raise SegmentationError("Normalized text mismatch")
        label = judgments[sample_id]
        if label.get("sentence_sha256") != record["sentence_sha256"]:
            raise SegmentationError("Label sentence hash mismatch")
        if label.get("label_status") == "confirmed":
            if not str(label.get("assessor") or "").strip():
                raise SegmentationError("Confirmed gold requires a human assessor")
            word_spans(text, label.get("tokens"))
    return manifest, records, judgments


def evaluate(directory=DEFAULT_DATA, out=ROOT / "results/segmentation"):
    manifest, records, labels = load_dataset(directory)
    confirmed = sum(r.get("label_status") == "confirmed" for r in labels.values())
    counts = Counter(r["category"] for r in records)
    eligible = (confirmed == len(records) == 150 and len(counts) == 10 and set(counts.values()) == {15})
    payload = {"schema_version": 1, "corpus_sha256": manifest["corpus_sha256"],
               "sample_sha256": manifest["sample_sha256"], "n_samples": len(records),
               "confirmed": confirmed, "official": eligible,
               "metrics_status": "measured" if eligible else "unmeasured",
               "reason": None if eligible else "Need all 150 human-confirmed labels (15/category)"}
    if eligible:
        totals, by_category, errors = {}, {}, []
        for method in ("fmm", "bmm", "bimm"):
            sums = [0, 0, 0]
            grouped = defaultdict(lambda: [0, 0, 0])
            for record in records:
                gold = labels[record["sample_id"]]["tokens"]
                predicted = tokenize(record["sentence_text"], method)
                row = span_counts(record["sentence_text"], predicted, gold)
                sums = [a + b for a, b in zip(sums, row)]
                grouped[record["category"]] = [a + b for a, b in zip(grouped[record["category"]], row)]
                if predicted != gold and len(errors) < 15:
                    errors.append({"method": method, "sample_id": record["sample_id"],
                                   "predicted": predicted, "gold": gold})
            totals[method] = scores(*sums)
            by_category[method] = {category: scores(*values) for category, values in grouped.items()}
        payload.update(metrics=totals, by_category=by_category, error_examples=errors,
                       aggregation="micro word-span counts; punctuation boundaries preserved")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "evaluation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=ROOT / "results/segmentation")
    args = parser.parse_args(argv)
    try:
        payload = evaluate(args.data, args.out)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["official"] else 5
    except (SegmentationError, KeyError, ValueError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
