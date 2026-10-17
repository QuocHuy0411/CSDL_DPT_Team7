"""Offline relevance annotation for the pooled query/document pairs.

This tool never invents a label. A judgment only becomes `confirmed` when a
human assessor types a relevance grade and their name. Everything else stays
`pending` with relevance `null`, and official evaluation must ignore pending
rows.

Workflow
    init      create qrels.json with every pooled pair pending, build the M1 sample
    review    print the deterministic 20% M1 cross-check sample (seed 2026)
    label     interactive terminal session for one query or one whole category
    export    write a self-contained offline HTML worksheet for offline annotators
    import    merge labels exported from the worksheet
    status    progress summary

Outputs (data/ground_truth/):
    qrels.json             judgments, status pending_human_annotation until complete
    m1_review_sample.json  the deterministic M1 cross-check sample
    annotation_sheet.html  offline worksheet (git-ignored, regenerate as needed)
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_POOL = ROOT / "data/ground_truth/mm/pool.json"
DEFAULT_QRELS = ROOT / "data/ground_truth/mm/qrels.json"
DEFAULT_SAMPLE = ROOT / "data/ground_truth/mm/m1_review_sample.json"
DEFAULT_SHEET = ROOT / "data/ground_truth/mm/annotation_sheet.html"

SCHEMA_VERSION = 1
PENDING = "pending_human_annotation"
COMPLETE = "human_annotation_confirmed"
REVIEW_SEED = 2026
REVIEW_FRACTION = 0.20
GRADES = {0: "không liên quan", 1: "liên quan", 2: "rất liên quan"}
SNIPPET_UNITS = 90


class AnnotationError(RuntimeError):
    """Raised when annotation input is inconsistent with the locked pool."""


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def build_m1_sample(pairs: list[dict], seed: int = REVIEW_SEED,
                    fraction: float = REVIEW_FRACTION) -> dict:
    """Deterministic 20% cross-check sample over the sorted pool."""
    ordered = sorted(pairs, key=lambda item: (item["query_id"], item["doc_id"]))
    size = round(len(ordered) * fraction)
    chooser = random.Random(seed)
    chosen = chooser.sample(ordered, size)
    return {
        "schema_version": SCHEMA_VERSION,
        "reviewer_role": "M1",
        "sample_fraction": fraction,
        "sample_size": size,
        "population_size": len(ordered),
        "seed": seed,
        "selection": "random.Random(seed).sample over pairs sorted by (query_id, doc_id)",
        "pairs": [{"query_id": item["query_id"], "doc_id": item["doc_id"]} for item in chosen],
    }


def new_qrels(pool: dict, sample: dict, assessor_pool: str) -> dict:
    judgments = [
        {
            "query_id": pair["query_id"],
            "doc_id": pair["doc_id"],
            "category": pair["category"],
            "relevance": None,
            "status": "pending",
            "assessor": None,
            "notes": "",
            "in_m1_sample": any(entry["query_id"] == pair["query_id"]
                                and entry["doc_id"] == pair["doc_id"]
                                for entry in sample["pairs"]),
            "retrieved_by": pair["retrieved_by"],
            "best_rank": pair["best_rank"],
        }
        for pair in pool["pairs"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_sha256": pool["corpus_sha256"],
        "queries_sha256": pool.get("queries_sha256"),
        "status": PENDING,
        "grading": {
            "scale": [0, 1, 2],
            "meaning": {str(k): v for k, v in GRADES.items()},
            "null": "not judged yet; a null row is never relevant",
            "source": "pool.json, union of three baseline Top-20 lists",
        },
        "rules": [
            "Judge each pair on its own, per query.",
            "Category is never a relevance signal.",
            "evidence_doc_ids in queries.json are authoring support, not judgments.",
            "Only rows with status 'confirmed' and relevance > 0 count as relevant.",
        ],
        "assessor_pool": assessor_pool,
        "annotation_progress": {"total": len(judgments), "confirmed": 0},
        "judgments": judgments,
    }


def cmd_init(args) -> int:
    pool = load_json(args.pool)
    if not pool.get("pairs"):
        raise AnnotationError("pool.json has no pairs; run scripts/build_pool.py first")
    sample = build_m1_sample(pool["pairs"], args.seed, args.fraction)
    qrels = new_qrels(pool, sample, args.assessor_pool)
    if args.qrels.exists() and not args.force:
        existing = load_json(args.qrels)
        if existing.get("corpus_sha256") != pool["corpus_sha256"]:
            raise AnnotationError(
                f"{args.qrels} was built against corpus {existing.get('corpus_sha256')} "
                f"but the pool uses {pool['corpus_sha256']}")
        if existing.get("queries_sha256") != pool.get("queries_sha256"):
            raise AnnotationError(
                f"{args.qrels} was built against queries {existing.get('queries_sha256')} "
                f"but the pool uses {pool.get('queries_sha256')}; the queries changed, so "
                "start a fresh qrels instead of mixing labels")
        print(f"{args.qrels} already exists; keeping it. Use --force to reset every label.")
        return 0
    write_json(args.qrels, qrels)
    write_json(args.sample, sample)
    print(f"wrote {args.qrels} with {len(qrels['judgments'])} pending judgments (status {PENDING})")
    print(f"wrote {args.sample} with {sample['sample_size']} pairs "
          f"({args.fraction:.0%} of {sample['population_size']}, seed {args.seed})")
    return 0


def load_context(pool_path: Path, qrels_path: Path) -> tuple[dict, dict, dict]:
    pool = load_json(pool_path)
    qrels = load_json(qrels_path)
    if pool["corpus_sha256"] != qrels["corpus_sha256"]:
        raise AnnotationError("qrels and pool disagree on corpus_sha256")
    if qrels.get("queries_sha256") != pool.get("queries_sha256"):
        raise AnnotationError(
            f"qrels was built against queries {qrels.get('queries_sha256')} but the pool "
            f"uses {pool.get('queries_sha256')}; rebuild the pool and re-run init")
    pool_by_pair = {(item["query_id"], item["doc_id"]): item for item in pool["pairs"]}
    index = {(item["query_id"], item["doc_id"]): item for item in qrels["judgments"]}
    if set(index) != set(pool_by_pair):
        raise AnnotationError("qrels judgments do not match the pool pairs exactly")
    return pool, qrels, index


def snippet(text: str, units: int = SNIPPET_UNITS) -> str:
    parts = text.split()
    if len(parts) <= units:
        return text
    return " ".join(parts[:units]) + " …"


def select_pairs(qrels: dict, query_id: str | None, category: str | None,
                 only_pending: bool) -> list[dict]:
    rows = qrels["judgments"]
    if query_id:
        rows = [row for row in rows if row["query_id"] == query_id]
    if category:
        rows = [row for row in rows if row["category"] == category]
    if only_pending:
        rows = [row for row in rows if row["status"] != "confirmed"]
    return rows


def cmd_label(args) -> int:
    pool, qrels, index = load_context(args.pool, args.qrels)
    query_text = {item["query_id"]: item for item in pool["queries"]}
    category_of = {item["doc_id"]: item["category"] for item in pool["pairs"]}
    rows = select_pairs(qrels, args.query, args.category, not args.all)
    if not rows:
        print("nothing to label for that selection")
        return 0
    assessor = args.assessor
    if not assessor:
        assessor = input("assessor name (recorded with every label): ").strip()
    if not assessor:
        print("error: an assessor name is required before labels can be confirmed")
        return 2
    pool_by_pair = {(item["query_id"], item["doc_id"]): item for item in pool["pairs"]}
    confirmed = 0
    for row in rows:
        pair = pool_by_pair[(row["query_id"], row["doc_id"])]
        text = query_text[row["query_id"]]["text"]
        print("=" * 100)
        print(f"{row['query_id']} [{query_text[row['query_id']]['category']}] {text}")
        print(f"doc {pair['doc_id']}  rank {pair['best_rank']} via {','.join(pair['retrieved_by'])}"
              f"  category={category_of[pair['doc_id']]}")
        print(f"title: {pair['title']}")
        print(f"text : {snippet(pair.get('content', ''))}")
        if pair.get("content") is None:
            print("      (rebuild the pool without --no-content to see the text)")
        while True:
            raw = input("relevance [0/1/2/skip]: ").strip().lower()
            if raw in {"0", "1", "2"}:
                grade = int(raw)
                break
            if raw in {"skip", "s", ""}:
                grade = None
                break
            print("   please type 0, 1, 2 or skip")
        if grade is None:
            continue
        row["relevance"] = grade
        row["status"] = "confirmed"
        row["assessor"] = assessor
        confirmed += 1
    refresh_progress(qrels)
    write_json(args.qrels, qrels)
    print(f"confirmed {confirmed} judgments; wrote {args.qrels}")
    return 0


def refresh_progress(qrels: dict) -> dict:
    judged = [row for row in qrels["judgments"]
              if row["status"] == "confirmed" and row["relevance"] is not None]
    qrels["annotation_progress"] = {
        "total": len(qrels["judgments"]),
        "confirmed": len(judged),
        "pending": len(qrels["judgments"]) - len(judged),
        "graded_distribution": dict(sorted(Counter(
            str(row["relevance"]) for row in judged).items())),
    }
    qrels["status"] = COMPLETE if len(judged) == len(qrels["judgments"]) and judged else PENDING
    return qrels


def cmd_review(args) -> int:
    sample = load_json(args.sample)
    pool, qrels, index = load_context(args.pool, args.qrels)
    pool_by_pair = {(item["query_id"], item["doc_id"]): item for item in pool["pairs"]}
    counts = Counter()
    print(f"M1 sample: {sample['sample_size']} of {sample['population_size']} pairs "
          f"({sample['sample_fraction']:.0%}, seed {sample['seed']})")
    for entry in sample["pairs"]:
        row = index[(entry["query_id"], entry["doc_id"])]
        counts[row["status"]] += 1
        pair = pool_by_pair[(entry["query_id"], entry["doc_id"])]
        state = f"{row['relevance']}/{row['status']}" if row["status"] == "confirmed" else "-"
        print(f"{entry['query_id']} {entry['doc_id']} [{state:>9}] {pair['title'][:70]}")
    print("status counts:", dict(counts))
    return 0


def cmd_status(args) -> int:
    _, qrels, _ = load_context(args.pool, args.qrels)
    refresh_progress(qrels)
    progress = qrels["annotation_progress"]
    print(f"status: {qrels['status']}")
    print(f"judgments: {progress['confirmed']}/{progress['total']} confirmed, "
          f"{progress['pending']} pending")
    print("graded distribution:", progress["graded_distribution"])
    per_query = {}
    for row in qrels["judgments"]:
        slot = per_query.setdefault(row["query_id"], [0, 0])
        if row["status"] == "confirmed":
            slot[0] += 1
        else:
            slot[1] += 1
    for query_id in sorted(per_query):
        done, pending = per_query[query_id]
        print(f"  {query_id}: {done} confirmed, {pending} pending")
    assessors = Counter(row["assessor"] for row in qrels["judgments"]
                        if row["assessor"])
    print("assessors:", dict(assessors))
    return 0


def _pair_section(row: dict, pair: dict, query_text: str) -> str:
    """One worksheet block. Full article text is available under <details>."""
    text = pair.get("content")
    if text:
        preview = (f"<details><summary>Xem toàn bộ bài "
                   f"({len(text.split())} đơn vị khoảng trắng)</summary>"
                   f"<p class='t'>{html.escape(text)}</p></details>")
        lead = f"<p class='t'>{html.escape(snippet(text))}</p>"
    else:
        preview = "<p class='t'>(pool built with --no-content; rebuild it for the text)</p>"
        lead = ""
    qid = html.escape(row["query_id"])
    did = html.escape(pair["doc_id"])
    selected = "" if row["relevance"] is None else f" selected='{row['relevance']}'"
    options = (f"<option value=''{'' if row['relevance'] is not None else ' selected'}>--</option>"
               f"<option value='0'{selected if row['relevance'] == 0 else ''}>0 - không liên quan</option>"
               f"<option value='1'{selected if row['relevance'] == 1 else ''}>1 - liên quan</option>"
               f"<option value='2'{selected if row['relevance'] == 2 else ''}>2 - rất liên quan</option>")
    return (
        f"<section class='pair' data-query='{qid}'>"
        f"<h3>{qid} &middot; {did} &middot; rank {pair['best_rank']}</h3>"
        f"<p class='q'><b>Query:</b> {html.escape(query_text)}</p>"
        f"<p class='q'><b>Doc title:</b> {html.escape(pair['title'])}</p>"
        f"<p class='q'><b>Category:</b> {html.escape(pair['category'])}"
        f" &middot; found by {html.escape(', '.join(pair['retrieved_by']))}</p>"
        f"{lead}{preview}"
        f"<p><label>relevance "
        f"<select data-qid='{qid}' data-did='{did}'>{options}</select></label>"
        f"<label>assessor "
        f"<input data-qid='{qid}' data-did='{did}' "
        f"value='{html.escape(row['assessor'] or '')}'></label></p>"
        f"</section>"
    )


def cmd_export(args) -> int:
    pool, qrels, index = load_context(args.pool, args.qrels)
    pool_by_pair = {(item["query_id"], item["doc_id"]): item for item in pool["pairs"]}
    query_text = {item["query_id"]: item["text"] for item in pool["queries"]}
    body = [_pair_section(row, pool_by_pair[(row["query_id"], row["doc_id"])],
                          query_text[row["query_id"]]) for row in qrels["judgments"]]
    document = (
        "<!doctype html><html lang='vi'><head><meta charset='utf-8'>"
        "<title>Phiếu gán nhãn liên quan (offline)</title><style>"
        "body{font-family:sans-serif;margin:1rem;max-width:62rem}"
        ".pair{border:1px solid #ccc;padding:.6rem;margin:.6rem 0;border-radius:4px}"
        ".t{background:#f6f6f6;padding:.5rem;font-size:.9rem}"
        "label{margin-right:1rem}select,input{margin-left:.3rem}"
        "</style></head><body>"
        "<h1>Phiếu gán nhãn liên quan</h1>"
        f"<p>Corpus: <code>{html.escape(pool['corpus_sha256'])}</code><br>"
        f"Queries: <code>{html.escape(str(pool.get('queries_sha256')))}</code><br>"
        "Điền 0/1/2 cho từng cặp. Công cụ KHÔNG tự tạo nhãn confirmed. "
        "Bấm nút xuất JSON rồi chạy "
        "<code>python scripts/annotate.py import --input labels.json</code>. "
        "JSON xuất ra kèm hash corpus và queries; import sẽ từ chối nếu hash không khớp.</p>"
        f"<label>Tên người gán (dùng cho mọi cặp chưa điền): "
        f"<input id='assessor' value='{html.escape(args.assessor or '')}'></label> "
        "<button onclick='download()'>Xuất JSON nhãn</button>"
        "<p><label>Lọc query <select id='queryFilter' onchange='filterRows()'>"
        "<option value=''>Tất cả</option>"
        + "".join(f"<option value='{html.escape(q)}'>{html.escape(q)}</option>" for q in query_text)
        + "</select></label> <span id='progress'></span></p>"
        "<p>Bản nháp tự lưu trên trình duyệt này nếu được phép. "
        "Vẫn cần xuất JSON để sao lưu và nhập vào qrels; bản nháp không tự xác nhận nhãn.</p>"
        + "".join(body) +
        "<script>"
        f"var CORPUS_SHA={json.dumps(pool['corpus_sha256'])};"
        f"var QUERIES_SHA={json.dumps(pool.get('queries_sha256'))};"
        f"var PAIRS_KEY={json.dumps(hashlib.sha256(json.dumps(sorted(pool_by_pair)).encode()).hexdigest())};"
        "var STORAGE_KEY='team7-labels:'+CORPUS_SHA+':'+QUERIES_SHA+':'+PAIRS_KEY;"
        "function collect(){var g=document.getElementById('assessor').value.trim();"
        "var labels=[];document.querySelectorAll('select[data-qid]').forEach(function(s){"
        "var field=document.querySelector(\"input[data-qid='\"+s.dataset.qid+\"'][data-did='\"+s.dataset.did+\"']\");"
        "if(s.value!=='')labels.push({query_id:s.dataset.qid,doc_id:s.dataset.did,"
        "relevance:parseInt(s.value,10),assessor:(field&&field.value.trim())||g});});"
        "return {corpus_sha256:CORPUS_SHA,queries_sha256:QUERIES_SHA,assessor:g,labels:labels};}"
        "function filterRows(){var q=document.getElementById('queryFilter').value;"
        "document.querySelectorAll('section.pair').forEach(function(s){s.hidden=!!q&&s.dataset.query!==q;});"
        "document.getElementById('progress').textContent=collect().labels.length+' / '+"
        "document.querySelectorAll('select[data-qid]').length+' cặp đã chọn nhãn';}"
        "function saveDraft(){try{localStorage.setItem(STORAGE_KEY,JSON.stringify(collect()));}catch(e){}filterRows();}"
        "try{var saved=JSON.parse(localStorage.getItem(STORAGE_KEY)||'null');if(saved){"
        "document.getElementById('assessor').value=saved.assessor||'';"
        "(saved.labels||[]).forEach(function(r){"
        "document.querySelectorAll('select[data-qid]').forEach(function(s){"
        "if(s.dataset.qid===r.query_id&&s.dataset.did===r.doc_id&&s.value===''){s.value=String(r.relevance);"
        "var a=document.querySelector(\"input[data-qid='\"+s.dataset.qid+\"'][data-did='\"+s.dataset.did+\"']\");"
        "if(a)a.value=r.assessor||'';}});});}}catch(e){}"
        "document.querySelectorAll('input,select[data-qid]').forEach(function(e){e.addEventListener('change',saveDraft);});"
        "filterRows();"
        "function download(){var g=document.getElementById('assessor').value.trim();"
        "var out={corpus_sha256:CORPUS_SHA,queries_sha256:QUERIES_SHA,assessor:g,labels:[]};"
        "document.querySelectorAll('select[data-qid]').forEach(function(s){"
        "var sel=\"input[data-qid='\"+s.dataset.qid+\"'][data-did='\"+s.dataset.did+\"']\";"
        "var a=document.querySelector(sel);var name=(a&&a.value.trim())||g;"
        "if(s.value!=='')out.labels.push({query_id:s.dataset.qid,doc_id:s.dataset.did,"
        "relevance:parseInt(s.value,10),assessor:name});});"
        "var blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});"
        "var a=document.createElement('a');a.href=URL.createObjectURL(blob);"
        "a.download='labels.json';a.click();}</script></body></html>")
    Path(args.sheet).parent.mkdir(parents=True, exist_ok=True)
    Path(args.sheet).write_text(document, encoding="utf-8")
    print(f"wrote offline worksheet {args.sheet} ({len(body)} pairs)")
    print(f"embedded corpus_sha256={pool['corpus_sha256']}")
    print(f"embedded queries_sha256={pool.get('queries_sha256')}")
    return 0


def cmd_import(args) -> int:
    pool, qrels, index = load_context(args.pool, args.qrels)
    payload = load_json(args.input)
    if isinstance(payload, list):
        raise AnnotationError(
            "labels must be an object carrying corpus_sha256, queries_sha256 and labels; "
            "a bare list cannot identify the corpus/query version")
    elif isinstance(payload, dict):
        labels = payload.get("labels") or []
        label_corpus = payload.get("corpus_sha256")
        label_queries = payload.get("queries_sha256")
        if label_corpus is None or label_queries is None:
            raise AnnotationError(
                "labels file must carry corpus_sha256 and queries_sha256 so stale "
                "worksheet labels cannot be applied to a changed corpus or query set")
        if label_corpus != pool["corpus_sha256"]:
            raise AnnotationError(
                f"labels were made against corpus {label_corpus} but the pool uses "
                f"{pool['corpus_sha256']}; re-export the worksheet")
        if label_queries != pool.get("queries_sha256"):
            raise AnnotationError(
                f"labels were made against queries {label_queries} but the pool uses "
                f"{pool.get('queries_sha256')}; the query texts changed, so re-export "
                "and re-annotate")
    else:
        raise AnnotationError("labels file must be a JSON object with a 'labels' list")
    applied = skipped = 0
    for label in labels:
        if not isinstance(label, dict) or "query_id" not in label or "doc_id" not in label:
            print(f"warning: malformed label {label!r}; skipped")
            skipped += 1
            continue
        key = (label["query_id"], label["doc_id"])
        row = index.get(key)
        if row is None:
            print(f"warning: {key} is not in the pool; skipped")
            skipped += 1
            continue
        grade = label.get("relevance")
        assessor = (label.get("assessor") or payload_assessor(payload)
                    or args.assessor or "").strip()
        if type(grade) is not int or grade not in {0, 1, 2}:
            print(f"warning: {key} has invalid relevance {grade!r}; left pending")
            skipped += 1
            continue
        if not assessor:
            print(f"warning: {key} has no assessor; left pending")
            skipped += 1
            continue
        row["relevance"] = int(grade)
        row["status"] = "confirmed"
        row["assessor"] = assessor
        applied += 1
    refresh_progress(qrels)
    write_json(args.qrels, qrels)
    print(f"imported {applied} labels, skipped {skipped}; status {qrels['status']}")
    return 0


def payload_assessor(payload) -> str:
    if isinstance(payload, dict):
        return str(payload.get("assessor") or "")
    return ""


def cmd_confirm(args) -> int:
    _, qrels, _ = load_context(args.pool, args.qrels)
    refresh_progress(qrels)
    progress = qrels["annotation_progress"]
    if progress["pending"]:
        raise AnnotationError(
            f"{progress['pending']} judgments are still pending; status stays {PENDING}. "
            "Official evaluation must refuse pending labels.")
    write_json(args.qrels, qrels)
    print(f"all {progress['total']} judgments confirmed; status {qrels['status']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create pending qrels and the M1 sample")
    init.add_argument("--seed", type=int, default=REVIEW_SEED)
    init.add_argument("--fraction", type=float, default=REVIEW_FRACTION)
    init.add_argument("--assessor-pool", default="M2, M1 cross-check")
    init.add_argument("--force", action="store_true", help="discard existing labels")
    init.set_defaults(func=cmd_init)

    review = sub.add_parser("review", help="show the deterministic M1 cross-check sample")
    review.set_defaults(func=cmd_review)

    label = sub.add_parser("label", help="interactive labeling session")
    label.add_argument("--query", help="query_id, e.g. Q07")
    label.add_argument("--category", help="Vietnamese display category")
    label.add_argument("--assessor")
    label.add_argument("--all", action="store_true", help="relabel confirmed rows too")
    label.set_defaults(func=cmd_label)

    export = sub.add_parser("export", help="write the offline HTML worksheet")
    export.add_argument("--assessor", default="",
                        help="prefill the single global assessor field")
    export.set_defaults(func=cmd_export)

    importer = sub.add_parser("import", help="merge labels exported from the worksheet")
    importer.add_argument("--input", type=Path, required=True)
    importer.add_argument("--assessor", help="default assessor when a label omits one")
    importer.set_defaults(func=cmd_import)

    status = sub.add_parser("status", help="progress summary")
    status.set_defaults(func=cmd_status)

    confirm = sub.add_parser("confirm", help="set the final status once nothing is pending")
    confirm.set_defaults(func=cmd_confirm)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except AnnotationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
