"""The AI review is a traceable proposal, never a replacement for human labels."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


ADJUDICATION_FILE = ROOT / "data/ground_truth/mm/m1_adjudication_2026-10-18.json"
GOLD_ADJUDICATION_FILE = ROOT / "data/gold/segmentation_adjudication_2026-10-18.json"


def test_review_decisions_and_amendment_preserve_the_original_snapshot():
    review = json.loads((ROOT / "results/label_review.json").read_text(encoding="utf-8"))
    resolution = json.loads((ROOT / "results/label_review_resolution.json").read_text(encoding="utf-8"))
    submitted_path = ROOT / resolution["source"]
    before_path = ROOT / resolution["qrels_before_path"]
    after_37_path = ROOT / resolution["qrels_after_37_path"]
    amendment_path = ROOT / resolution["amendment_source"]
    qrels_path = ROOT / "data/ground_truth/mm/qrels.json"
    gold_path = ROOT / "data/gold/segmentation_labels.jsonl"
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    gold = {row["sample_id"]: row for row in
            (json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines())}

    assert review["status"] == "needs_human_adjudication"
    assert "not human judgments" in review["review_type"]
    if before_path.exists():
        assert hashlib.sha256(before_path.read_bytes()).hexdigest() == review["qrels_sha256_at_review"]
    assert resolution["qrels_before_sha256"] == review["qrels_sha256_at_review"]
    if after_37_path.exists():
        assert hashlib.sha256(after_37_path.read_bytes()).hexdigest() == resolution["qrels_after_37_sha256"]
    if amendment_path.exists():
        assert hashlib.sha256(amendment_path.read_bytes()).hexdigest() == resolution["amendment_source_sha256"]
    if submitted_path.exists():
        assert hashlib.sha256(submitted_path.read_bytes()).hexdigest() == resolution["source_sha256"]
    gold_after = json.loads(GOLD_ADJUDICATION_FILE.read_text(encoding="utf-8"))
    final_release = json.loads((ROOT / "results/final_release.json").read_text(encoding="utf-8"))
    gold_current_sha256 = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    assert gold_current_sha256 == final_release["sha256"]["data/gold/segmentation_labels.jsonl"]
    assert gold_current_sha256 != review["gold_sha256_at_review"], (
        "gold adjudication is stale: gold moved on without a new record")
    assert gold_after["gold_before_sha256"] == review["gold_sha256_at_review"]
    gold_decisions = {row["sample_id"]: row for row in gold_after["decisions"]}
    assert set(gold_decisions) == {proposal["sample_id"]
                                   for proposal in review["gold_flags"]}
    for sample_id, decision in gold_decisions.items():
        joined = " ".join(gold[sample_id]["tokens"])
        assert decision["final_tokens_fragment"] in joined
        assert decision["before_tokens_fragment"] not in joined
        assert gold[sample_id]["label_status"] == "confirmed"
    assert review["corpus_sha256"] == qrels["corpus_sha256"]
    assert review["queries_sha256"] == qrels["queries_sha256"]

    adjudication = json.loads(ADJUDICATION_FILE.read_text(encoding="utf-8"))
    judgments = {(row["query_id"], row["doc_id"]): row for row in qrels["judgments"]}
    current_sha256 = hashlib.sha256(qrels_path.read_bytes()).hexdigest()
    assert current_sha256 == final_release["sha256"]["data/ground_truth/mm/qrels.json"]
    assert current_sha256 != resolution["qrels_after_sha256"], (
        "adjudication is stale: qrels moved on without a new record")
    adjudications = {(row["query_id"], row["doc_id"]): row
                     for row in final_release["relevance_decisions"]}
    assert len(adjudications) == adjudication["summary"]["n_pairs"] == 8
    for key, decision in adjudications.items():
        assert judgments[key]["relevance"] == decision["final_grade"]
        assert judgments[key]["status"] == "confirmed"

    if after_37_path.exists() and submitted_path.exists() and amendment_path.exists() and before_path.exists():
        original = json.loads(before_path.read_text(encoding="utf-8"))
        original_judgments = {(row["query_id"], row["doc_id"]): row
                              for row in original["judgments"]}
        after_37 = json.loads(after_37_path.read_text(encoding="utf-8"))
        after_37_judgments = {(row["query_id"], row["doc_id"]): row
                              for row in after_37["judgments"]}
        submitted = json.loads(submitted_path.read_text(encoding="utf-8"))
        decisions = {(row["query_id"], row["doc_id"]): row for row in submitted["labels"]}
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        amendments = {(row["query_id"], row["doc_id"]): row for row in amendment["labels"]}
        assert amendment["review_qrels_sha256"] == resolution["qrels_after_37_sha256"]
        assert amendment["human_review_confirmed"] is True
        assert set(amendments) == {("Q16", "vn_f4f2b5ae9443c172")}
        flagged = set()
        for proposal in review["qrels_flags"]:
            key = proposal["query_id"], proposal["doc_id"]
            assert key not in flagged
            flagged.add(key)
            assert proposal["current_grade"] == original_judgments[key]["relevance"]
            assert proposal["suggested_grade"] in {0, 1, 2}
            assert proposal["suggested_grade"] != proposal["current_grade"]
            assert after_37_judgments[key]["relevance"] == decisions[key]["relevance"]
            assert judgments[key]["relevance"] == (
                amendments[key]["relevance"] if key in amendments else decisions[key]["relevance"])
            assert judgments[key]["status"] == "confirmed"
        assert len(flagged) == 37
        assert set(decisions) == flagged
        assert resolution["n_submitted_pairs"] == len(decisions)
    assert len(review["gold_flags"]) == 3

    for proposal in review["gold_flags"]:
        tokens = gold[proposal["sample_id"]]["tokens"]
        if proposal["sample_id"] in gold_decisions:
            # Fragment-level before/after checks above already verify that the
            # adjudicated replacement was applied for this sample.
            pass
        else:
            assert proposal["current_token"] in tokens
        assert gold[proposal["sample_id"]]["label_status"] == "confirmed"
