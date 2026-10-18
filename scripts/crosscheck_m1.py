"""Export a blind M1 worksheet and compare independently supplied grades.

This module never changes qrels. A second person's identity, independence and
actual reading cannot be proven by software; the form records their declaration.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

from scripts.annotate import (
    DEFAULT_POOL, DEFAULT_QRELS, DEFAULT_SAMPLE, build_m1_sample, load_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHEET = ROOT / "data/ground_truth/mm/m1_review_sheet.html"
DEFAULT_COMPARISON = ROOT / "results/m1_crosscheck.json"


class CrosscheckError(ValueError):
    """Invalid, stale or incomplete second-reviewer input."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pairs(rows: list[dict]) -> list[tuple[str, str]]:
    return [(row["query_id"], row["doc_id"]) for row in rows]


def _context(pool_path: Path, sample_path: Path) -> tuple[dict, dict, dict]:
    pool, sample = load_json(pool_path), load_json(sample_path)
    if pool.get("schema_version") != 1 or sample.get("schema_version") != 1:
        raise CrosscheckError("unsupported pool or M1 sample schema")
    pool_keys = _pairs(pool["pairs"])
    sample_keys = _pairs(sample["pairs"])
    if len(pool_keys) != len(set(pool_keys)):
        raise CrosscheckError("pool has duplicate query/document pairs")
    if not sample_keys or len(sample_keys) != len(set(sample_keys)):
        raise CrosscheckError("M1 sample is empty or has duplicate pairs")
    if len(pool_keys) != sample.get("population_size"):
        raise CrosscheckError("M1 sample population_size does not match pool")
    if len(sample_keys) != sample.get("sample_size"):
        raise CrosscheckError("M1 sample_size does not match its pairs")
    if not set(sample_keys) <= set(pool_keys):
        raise CrosscheckError("M1 sample includes a pair outside the locked pool")
    if sample.get("reviewer_role") != "M1":
        raise CrosscheckError("M1 sample has the wrong reviewer role")
    seed, fraction = sample.get("seed"), sample.get("sample_fraction")
    if type(seed) is not int or type(fraction) not in {int, float} or not 0 < fraction <= 1:
        raise CrosscheckError("M1 sample has an invalid seed or sample_fraction")
    expected = build_m1_sample(
        pool["pairs"], seed=seed, fraction=fraction,
    )
    if sample_keys != _pairs(expected["pairs"]) or sample["sample_size"] != expected["sample_size"]:
        raise CrosscheckError("M1 sample differs from declared deterministic selection")
    queries = {row["query_id"]: row["text"] for row in pool["queries"]}
    if len(queries) != len(pool["queries"]) or any(qid not in queries for qid, _ in sample_keys):
        raise CrosscheckError("pool queries are missing or duplicated")
    return pool, sample, dict(zip(pool_keys, pool["pairs"]))


def export_sheet(pool_path: Path, sample_path: Path, out: Path) -> int:
    """Write an offline sheet with full content but no first-reviewer grades."""
    pool, sample, pairs = _context(pool_path, sample_path)
    queries = {row["query_id"]: row["text"] for row in pool["queries"]}
    blocks = []
    for qid, did in _pairs(sample["pairs"]):
        pair = pairs[qid, did]
        if not pair.get("content"):
            raise CrosscheckError(f"full article is missing for {(qid, did)}")
        blocks.append(
            f"<section><h3>{html.escape(qid)} · {html.escape(did)}</h3>"
            f"<p><b>Truy vấn:</b> {html.escape(queries[qid])}</p>"
            f"<p><b>Tiêu đề:</b> {html.escape(pair['title'])}</p>"
            f"<details><summary>Đọc toàn bài</summary><p class='article'>"
            f"{html.escape(pair['content'])}</p></details>"
            f"<label>Nhãn <select data-qid='{html.escape(qid, quote=True)}' "
            f"data-did='{html.escape(did, quote=True)}'>"
            "<option value=''>-- chọn --</option><option value='0'>0</option>"
            "<option value='1'>1</option><option value='2'>2</option></select></label></section>"
        )
    version = {
        "schema_version": 1,
        "corpus_sha256": pool["corpus_sha256"],
        "queries_sha256": pool["queries_sha256"],
        "sample_sha256": _sha256(sample_path),
        "reviewer_role": "M1",
    }
    document = (
        "<!doctype html><html lang='vi'><meta charset='utf-8'>"
        "<title>Phiếu M1 chấm độc lập</title><style>body{font-family:Arial;max-width:65rem;"
        "margin:auto}section{border:1px solid #aaa;padding:1rem;margin:1rem 0}"
        ".article{white-space:pre-wrap;line-height:1.5}details{margin:1rem 0}</style>"
        f"<h1>Phiếu M1 chấm độc lập — {sample['sample_size']} cặp</h1>"
        "<p>Không hiển thị nhãn/người gán ban đầu. Đọc toàn bài và đánh giá theo "
        "nội dung truy vấn (0: không liên quan; 1: một phần; 2: trả lời trực tiếp), "
        "không dùng lĩnh vực làm nhãn. Chỉ xuất sau khi điền đủ và tự xác nhận "
        "đã chấm độc lập.</p>"
        "<label>Tên người chấm M1 <input id='reviewer'></label> "
        "<label><input id='independent' type='checkbox'>Tôi tự đọc và chấm độc lập, "
        "không sao chép nhãn gốc</label> <button onclick='download()'>Xuất JSON</button>"
        + "".join(blocks)
        + "<script>const VERSION=" + json.dumps(version, ensure_ascii=False) + ";"
        "function download(){const name=document.getElementById('reviewer').value.trim();"
        "const independent=document.getElementById('independent').checked;"
        "const fields=[...document.querySelectorAll('select[data-qid]')];"
        "if(!name||!independent||fields.some(x=>x.value==='')){alert('Cần tên, xác nhận "
        "độc lập và đủ tất cả nhãn.');return;}"
        "const labels=fields.map(x=>({query_id:x.dataset.qid,doc_id:x.dataset.did,"
        "relevance:Number(x.value)}));"
        "const payload={...VERSION,reviewer:name,independent_review_confirmed:true,labels};"
        "const a=document.createElement('a');a.href=URL.createObjectURL(new Blob("
        "[JSON.stringify(payload,null,2)],{type:'application/json'}));"
        "a.download='m1_review_labels.json';a.click();}</script></html>"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    return len(blocks)


def compare(pool_path: Path, sample_path: Path, qrels_path: Path,
            labels_path: Path) -> dict:
    """Compare two human-provided sets; never claim to verify independence."""
    pool, sample, _ = _context(pool_path, sample_path)
    qrels, second = load_json(qrels_path), load_json(labels_path)
    expected = set(_pairs(sample["pairs"]))
    first_rows = qrels["judgments"]
    first_keys = _pairs(first_rows)
    if len(first_keys) != len(set(first_keys)):
        raise CrosscheckError("qrels contain duplicate query/document pairs")
    first = dict(zip(first_keys, first_rows))
    if qrels.get("schema_version") != 1:
        raise CrosscheckError("unsupported qrels schema")
    if (qrels.get("corpus_sha256") != pool["corpus_sha256"]
            or qrels.get("queries_sha256") != pool["queries_sha256"]):
        raise CrosscheckError("qrels do not match the locked pool")
    required = (
        ("schema_version", 1),
        ("corpus_sha256", pool["corpus_sha256"]),
        ("queries_sha256", pool["queries_sha256"]),
        ("sample_sha256", _sha256(sample_path)),
        ("reviewer_role", "M1"),
    )
    for field, value in required:
        if second.get(field) != value:
            raise CrosscheckError(f"M1 review has missing/stale field: {field}")
    if second.get("independent_review_confirmed") is not True:
        raise CrosscheckError("M1 review lacks an independence declaration")
    reviewer = second.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise CrosscheckError("M1 reviewer name is required")
    rows = second.get("labels")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise CrosscheckError(f"M1 review needs exactly {len(expected)} labels")
    second_by_key = {}
    for row in rows:
        if not isinstance(row, dict):
            raise CrosscheckError("each M1 label must be an object")
        key = row.get("query_id"), row.get("doc_id")
        if key in second_by_key or key not in expected:
            raise CrosscheckError(f"duplicate or unknown pair: {key}")
        if type(row.get("relevance")) is not int or row["relevance"] not in {0, 1, 2}:
            raise CrosscheckError(f"invalid grade for {key}")
        second_by_key[key] = row["relevance"]
    if set(second_by_key) != expected:
        raise CrosscheckError("M1 review does not cover the exact sample")
    differences = []
    matrix = {str(old): {str(new): 0 for new in (0, 1, 2)} for old in (0, 1, 2)}
    for key in sorted(expected):
        original = first.get(key)
        if (original is None or original.get("status") != "confirmed"
                or type(original.get("relevance")) is not int
                or original["relevance"] not in {0, 1, 2}):
            raise CrosscheckError(f"original qrels label is not confirmed for {key}")
        if str(original.get("assessor") or "").strip().casefold() == reviewer.strip().casefold():
            raise CrosscheckError("M1 reviewer name matches original assessor")
        old, new = original["relevance"], second_by_key[key]
        matrix[str(old)][str(new)] += 1
        if old != new:
            differences.append({
                "query_id": key[0], "doc_id": key[1],
                "original_grade": old, "m1_grade": new,
                "resolution": "pending_human_adjudication",
            })
    agreements = len(expected) - len(differences)
    return {
        "schema_version": 1,
        "status": "requires_human_adjudication" if differences else "agreement_recorded",
        "note": "Self-declared independence only; software cannot verify identity or reading.",
        "corpus_sha256": pool["corpus_sha256"],
        "queries_sha256": pool["queries_sha256"],
        "sample_sha256": _sha256(sample_path),
        "qrels_sha256": _sha256(qrels_path),
        "m1_labels_sha256": _sha256(labels_path),
        "m1_reviewer": reviewer.strip(),
        "sample_size": len(expected),
        "agreements": agreements,
        "observed_agreement": agreements / len(expected),
        "disagreement_count": len(differences),
        "grade_confusion_matrix": matrix,
        "disagreements": differences,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--out", type=Path, default=DEFAULT_SHEET)
    check = sub.add_parser("compare")
    check.add_argument("--labels", type=Path, required=True)
    check.add_argument("--out", type=Path, default=DEFAULT_COMPARISON)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            count = export_sheet(args.pool, args.sample, args.out)
            print(f"Exported {count} blind pairs to {args.out}")
        else:
            result = compare(args.pool, args.sample, args.qrels, args.labels)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"Compared {result['sample_size']} pairs: "
                f"{result['disagreement_count']} disagreements; {args.out}"
            )
        return 0
    except (CrosscheckError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
