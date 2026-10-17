"""Carry human judgments into a new pool without changing locked query/document meaning.

Defaults migrate the retained syllable pool to ground_truth/mm. Source files are
read-only. Conflicting confirmed judgments require human adjudication; no force reset.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from scripts.annotate import (AnnotationError, ROOT, build_m1_sample, load_context,
                              load_json, new_qrels, refresh_progress, write_json)


def migrate(source_pool: Path, source_qrels: Path, target_pool: Path,
            target_qrels: Path, target_sample: Path) -> dict:
    old_pool, old_qrels, old_rows = load_context(source_pool, source_qrels)
    pool = load_json(target_pool)
    for field in ("schema_version", "corpus_sha256", "queries_sha256"):
        if old_pool.get(field) != pool.get(field) or pool.get(field) is None:
            raise AnnotationError(f"Cannot reuse judgments: {field} mismatch or missing")
    if pool["schema_version"] != 1:
        raise AnnotationError("Unsupported pool schema")
    meaning = lambda p: {q["query_id"]: q["text"] for q in p["queries"]}
    if meaning(old_pool) != meaning(pool):
        raise AnnotationError("Query meaning differs even though the claimed hash matches")
    pairs = {(p["query_id"], p["doc_id"]): p for p in pool["pairs"]}
    if len(pairs) != len(pool["pairs"]):
        raise AnnotationError("Target pool contains duplicate pairs")
    old_pairs = {(p["query_id"], p["doc_id"]): p for p in old_pool["pairs"]}
    for key in pairs.keys() & old_pairs.keys():
        if pairs[key].get("content") != old_pairs[key].get("content"):
            raise AnnotationError(f"Document content mismatch for {key}")
        if not pairs[key].get("content"):
            raise AnnotationError("Embedded content required to verify reused judgments")
    sample = build_m1_sample(pool["pairs"])
    if target_qrels.exists():
        _, existing, _ = load_context(target_pool, target_qrels)
        qrels = deepcopy(existing)
    else:
        qrels = new_qrels(pool, sample, "M2, M1 cross-check")
    carried = 0
    for row in qrels["judgments"]:
        key = (row["query_id"], row["doc_id"])
        previous = old_rows.get(key)
        if previous is None or previous.get("status") != "confirmed":
            continue
        grade = previous.get("relevance")
        if type(grade) is not int or grade not in {0, 1, 2} or not str(previous.get("assessor") or "").strip():
            raise AnnotationError(f"Invalid confirmed source judgment: {key}")
        if row.get("status") == "confirmed":
            if row.get("relevance") != grade:
                raise AnnotationError(f"Conflicting confirmed judgments for {key}; adjudicate manually")
            continue
        for field in ("relevance", "status", "assessor", "notes"):
            row[field] = deepcopy(previous.get(field))
        row["carried_from"] = str(source_qrels)
        carried += 1
    audit = {"source_qrels": str(source_qrels), "source_pool": str(source_pool),
             "carried_confirmed": carried,
             "source_confirmed_outside_new_pool": sum(
                 r.get("status") == "confirmed" and k not in pairs for k, r in old_rows.items()),
             "source_preserved": True}
    qrels["migration"] = audit
    refresh_progress(qrels)
    write_json(target_qrels, qrels)
    write_json(target_sample, sample)
    return audit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "data/ground_truth"
    parser.add_argument("--source-pool", type=Path, default=base / "pool.json")
    parser.add_argument("--source-qrels", type=Path, default=base / "qrels.json")
    parser.add_argument("--pool", type=Path, default=base / "mm/pool.json")
    parser.add_argument("--qrels", type=Path, default=base / "mm/qrels.json")
    parser.add_argument("--sample", type=Path, default=base / "mm/m1_review_sample.json")
    args = parser.parse_args(argv)
    try:
        if args.source_qrels.resolve() == args.qrels.resolve():
            raise AnnotationError("Source and destination qrels must be different files")
        print(migrate(args.source_pool, args.source_qrels, args.pool, args.qrels, args.sample))
    except AnnotationError as error:
        print(f"error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
