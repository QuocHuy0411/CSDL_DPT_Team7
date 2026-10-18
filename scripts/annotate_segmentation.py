"""Offline human word-segmentation annotation: export, import, status.

Write words separated by spaces and syllables within each word joined by '_'.
Suggestions are separate from the empty answer field and never self-confirm.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import unicodedata

from evaluator.segmentation_eval import (DEFAULT_DATA, SegmentationError, load_dataset, word_spans)


def import_labels(directory, input_path, assessor=""):
    manifest, records, current = load_dataset(directory)
    by_id = {r["sample_id"]: r for r in records}
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise SegmentationError("Labels must be a versioned JSON object")
    for field in ("corpus_sha256", "sample_sha256", "normalizer_sha256"):
        if payload.get(field) != manifest[field]:
            raise SegmentationError(f"Label version mismatch: {field}")
    if payload.get("human_confirmed") is not True:
        raise SegmentationError("Explicit human review confirmation is required; machine suggestions are not gold")
    updates, seen = [], set()
    labels = payload.get("labels")
    if not isinstance(labels, list):
        raise SegmentationError("labels must be a list")
    for label in labels:
        if not isinstance(label, dict):
            raise SegmentationError("Each label must be an object")
        sample_id = label.get("sample_id")
        if sample_id not in by_id or sample_id in seen:
            raise SegmentationError("Unknown or duplicate sample ID")
        seen.add(sample_id)
        record = by_id[sample_id]
        if label.get("sentence_sha256") != record["sentence_sha256"]:
            raise SegmentationError("Stale sentence hash")
        name = str(label.get("assessor") or payload.get("assessor") or assessor).strip()
        if not name:
            raise SegmentationError("A human assessor name is required")
        tokens = label.get("tokens")
        if isinstance(tokens, str):
            tokens = tokens.split()
        if not isinstance(tokens, list) or any(not isinstance(t, str) for t in tokens):
            raise SegmentationError("Invalid gold token list")
        tokens = [unicodedata.normalize("NFC", t).lower() for t in tokens]
        word_spans(record["sentence_text"], tokens)
        updates.append((sample_id, tokens, name))
    # Validate every row before any write; unmentioned confirmed rows are preserved.
    for sample_id, tokens, name in updates:
        current[sample_id].update(tokens=tokens, assessor=name, label_status="confirmed")
    target = Path(directory) / "segmentation_labels.jsonl"
    target.write_text("".join(json.dumps(current[r["sample_id"]], ensure_ascii=False) + "\n"
                              for r in records), encoding="utf-8")
    return len(updates)


def export(directory, out, only_ids=None):
    manifest, records, labels = load_dataset(directory)
    focused_review = only_ids is not None
    if only_ids is not None:
        requested = set(only_ids)
        if not requested or len(requested) != len(only_ids) or not requested <= {r["sample_id"] for r in records}:
            raise SegmentationError("Unknown, duplicate or empty sample ID selection")
        records = [r for r in records if r["sample_id"] in requested]
    esc = html.escape
    sections = []
    for r in records:
        label = labels[r["sample_id"]]
        answer = (" ".join(label["tokens"])
                  if only_ids is None and label.get("label_status") == "confirmed" else "")
        suggestion = ("<details><summary>Gợi ý máy, chưa phải gold</summary><code>"
                      + esc(" ".join(r["suggestion"]["tokens"])) + "</code></details>"
                      if not focused_review else "")
        sections.append(
            f"<section><h3>{esc(r['sample_id'])} · {esc(r['category'])}</h3>"
            f"<p>{esc(r['sentence_text'])}</p><p>Chuỗi âm tiết: {esc(r['normalized_text'])}</p>"
            f"{suggestion}"
            f"<textarea data-id='{esc(r['sample_id'])}' data-hash='{esc(r['sentence_sha256'])}' rows='3'>{esc(answer)}</textarea>"
            f"<p>Nguồn: {esc(r['doc_id'])}, câu {r['sentence_index']}</p></section>")
    version = {field: manifest[field] for field in ("schema_version", "corpus_sha256", "sample_sha256", "normalizer_sha256")}
    document = (
        "<!doctype html><html lang='vi'><meta charset='utf-8'><title>Gán nhãn tách từ M1</title>"
        "<style>body{font-family:Arial;max-width:60rem;margin:auto}section{border:1px solid #aaa;padding:1rem;margin:1rem 0}"
        f"textarea{{width:98%}}code{{white-space:pre-wrap}}</style><h1>{len(records)} câu — phân đoạn từ do người gán</h1>"
        "<p>Điền các từ cách nhau bằng khoảng trắng, âm tiết trong từ nối dấu _. "
        "Không thêm/bỏ/đổi thứ tự âm tiết, không nối từ qua dấu câu. "
        + ("Phiếu rà soát riêng không hiển thị nhãn cũ hoặc gợi ý máy.</p>"
           if focused_review else "Gợi ý máy có thể sai.</p>")
        + "<label>Tên người gán <input id='assessor'></label><p><label>"
        "<input type='checkbox' id='reviewed'>Tôi đã tự đọc và kiểm tra các câu đã điền</label></p>"
        "<button onclick='download()'>Xuất JSON để nhập nhãn</button>"
        + "".join(sections) + "<script>var VERSION=" + json.dumps(version) + ";"
        "function data(){var p=Object.assign({},VERSION);p.assessor=document.getElementById('assessor').value.trim();"
        "p.human_confirmed=document.getElementById('reviewed').checked;p.labels=[];"
        "document.querySelectorAll('textarea').forEach(function(t){if(t.value.trim())p.labels.push({"
        "sample_id:t.dataset.id,sentence_sha256:t.dataset.hash,tokens:t.value.trim().split(/\\s+/)});});return p;}"
        "function download(){var p=data();if(!p.assessor||!p.human_confirmed||!p.labels.length){"
        "alert('Điền ít nhất một nhãn, tên và xác nhận đã tự kiểm tra.');return;}"
        "var a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(p,null,2)],"
        "{type:'application/json'}));a.download='segmentation_labels.json';a.click();}"
        "var KEY='team7-seg:'+VERSION.sample_sha256+':' + " +
        json.dumps('review:' + ','.join(sorted(only_ids)) if only_ids else 'all') + ";"
        "try{var d=JSON.parse(localStorage.getItem(KEY)||'null');if(d){document.getElementById('assessor').value=d.assessor||'';"
        "(d.labels||[]).forEach(function(r){document.querySelectorAll('textarea').forEach(function(t){"
        "if(t.dataset.id===r.sample_id&&!t.value)t.value=r.tokens.join(' ');});});}}catch(e){}"
        "document.querySelectorAll('textarea,input').forEach(function(t){t.addEventListener('change',function(){"
        "try{localStorage.setItem(KEY,JSON.stringify(data()));}catch(e){}});});</script></html>")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(document, encoding="utf-8")
    return len(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    exp = sub.add_parser("export")
    exp.add_argument("--out", type=Path, default=DEFAULT_DATA / "annotation_sheet.html")
    exp.add_argument("--only", nargs="+", help="export only selected IDs with blank answers")
    imp = sub.add_parser("import")
    imp.add_argument("--input", type=Path, required=True)
    imp.add_argument("--assessor", default="")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            print(f"Exported {export(args.data, args.out, args.only)} sentences to {args.out}")
        elif args.command == "import":
            print(f"Imported {import_labels(args.data, args.input, args.assessor)} human labels")
        else:
            _, records, labels = load_dataset(args.data)
            print(f"Human confirmed: {sum(r.get('label_status') == 'confirmed' for r in labels.values())}/{len(records)}")
        return 0
    except (SegmentationError, ValueError, KeyError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
