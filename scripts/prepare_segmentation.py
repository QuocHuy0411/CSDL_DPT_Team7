"""Chuẩn bị mẫu 150 câu (15/lĩnh vực) để đo chất lượng tách từ M1.

Câu được rút từ corpus đã khóa (``data/corpus.jsonl`` + ``corpus_manifest.json``);
corpus không bị sửa. Token gợi ý do Maximum Matching sinh ra và **luôn ở trạng
thái pending**: bộ này không phải gold do người xác nhận, nên
``evaluator/segmentation_eval.py`` sẽ từ chối đánh giá chính thức cho tới khi con
người gán nhãn.

    python -m scripts.prepare_segmentation                # ghi data/gold/**
    python -m scripts.prepare_segmentation --check        # chỉ kiểm tra, không ghi
    python -m scripts.prepare_segmentation --force        # ghi lại (giữ nhãn cũ)

Mỗi câu mang ``sample_id`` ổn định, ``sentence_sha256``, ``doc_id``, ``category``
và ``sentence_index`` trong tài liệu, nên mẫu tái lập được bằng cùng seed và
đổi từ điển/tiền xử lý sẽ bị phát hiện qua hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data/corpus.jsonl"
DEFAULT_MANIFEST = ROOT / "data/corpus_manifest.json"
DEFAULT_OUTPUT = ROOT / "data/gold"

SAMPLE_FILE = "segmentation_sample.jsonl"
MANIFEST_FILE = "segmentation_manifest.json"
LABELS_FILE = "segmentation_labels.jsonl"

SCHEMA_VERSION = 1
SEED = 2026
PER_CATEGORY = 15
TARGET_TOTAL = 150
#: A sentence must be annotatable: long enough to be informative, short enough to read.
MIN_UNITS = 12
MAX_UNITS = 60
SUGGESTION_STATUS = "pending"
SUGGESTION_SOURCE = "m1_suggestion_bimm"

#: Punctuation and line breaks end a sentence; a sentence never spans them.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;:])\s+|\n+")
#: Artefacts that make a sentence unusable for segmentation annotation.
SKIP_PATTERN = re.compile(r"(https?://|www\.|@|\.com|\.vn|\.org|\.net)", re.IGNORECASE)
LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sentence_text(text: str) -> str:
    """NFC + collapsed whitespace. Punctuation is kept: it is an annotation anchor."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def split_sentences(content: str) -> List[Tuple[int, str]]:
    """Split on punctuation/newline boundaries; returns ``(index, sentence)``."""
    sentences: List[Tuple[int, str]] = []
    position = 0
    for index, chunk in enumerate(SENTENCE_BOUNDARY.split(content)):
        cleaned = stable_sentence_text(chunk)
        if cleaned:
            sentences.append((index, cleaned))
        position += len(chunk)
    return sentences


def is_annotatable(sentence: str, min_units: int = MIN_UNITS, max_units: int = MAX_UNITS) -> bool:
    units = sentence.split()
    if not min_units <= len(units) <= max_units:
        return False
    if SKIP_PATTERN.search(sentence):
        return False
    letters = LETTER.findall(sentence)
    if len(letters) < 0.6 * sum(1 for unit in units for _ in unit):
        return False
    return True


def load_corpus(corpus_path: Path, manifest_path: Path) -> Tuple[List[Mapping], str]:
    """Read the locked corpus and refuse if it no longer matches the manifest hash."""
    digest = sha256_file(corpus_path)
    if not manifest_path.is_file():
        raise SystemExit("Locked corpus manifest is required")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("corpus_sha256") or manifest.get("sha256")
        if not expected or expected != digest:
            raise SystemExit(
                f"Corpus SHA256 {digest} does not match the locked manifest {expected}; "
                "refusing to build a sample from an unlocked corpus")
    documents = [json.loads(line) for line
                 in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return documents, digest


def tokenizer_metadata() -> Dict[str, object]:
    """Provenance of the tokenizer that produced the suggestions."""
    from engine.tokenizer.max_matching import dictionary_metadata, tokenize

    info = dictionary_metadata()
    probe = tokenize("Trí tuệ nhân tạo trong y tế")
    return {
        "tokenizer": "engine.tokenizer.max_matching:tokenize",
        "method": "bimm",
        "dictionary_path": info.get("dictionary_path"),
        "dictionary_sha256": info.get("dictionary_sha256"),
        "dictionary_entries": info.get("dictionary_entries"),
        "probe": probe,
        "tier1_applied": False,
    }


def sample_per_category(documents: Sequence[Mapping], per_category: int = PER_CATEGORY,
                        seed: int = SEED) -> List[Dict[str, object]]:
    """Pick ``per_category`` sentences from each of the ten categories, deterministically.

    Documents are visited in a stable order (category, doc_id); one sentence is
    taken from as many distinct documents as possible before a second sentence of
    the same document is used.
    """
    by_category: Dict[str, List[Mapping]] = {}
    for document in documents:
        by_category.setdefault(str(document["category"]), []).append(document)

    selected: List[Dict[str, object]] = []
    for category in sorted(by_category):
        candidates: List[Tuple[str, int, str]] = []
        for document in sorted(by_category[category], key=lambda item: str(item["doc_id"])):
            for sentence_index, sentence in split_sentences(str(document.get("content", ""))):
                if is_annotatable(sentence):
                    candidates.append((str(document["doc_id"]), sentence_index, sentence))
        if not candidates:
            raise SystemExit(f"No annotatable sentence in category {category!r}")
        generator = random.Random(f"{seed}:{category}")
        order = list(range(len(candidates)))
        generator.shuffle(order)
        chosen: List[Tuple[str, int, str]] = []
        used_documents: Dict[str, int] = {}
        for index in order:
            doc_id, _, _ = candidates[index]
            if used_documents.get(doc_id, 0) >= 1:
                continue
            chosen.append(candidates[index])
            used_documents[doc_id] = used_documents.get(doc_id, 0) + 1
            if len(chosen) == per_category:
                break
        if len(chosen) < per_category:
            for index in order:
                if candidates[index] in chosen:
                    continue
                chosen.append(candidates[index])
                if len(chosen) == per_category:
                    break
        chosen.sort(key=lambda item: (item[0], item[1]))
        for doc_id, sentence_index, sentence in chosen:
            selected.append({
                "doc_id": doc_id,
                "category": category,
                "sentence_index": sentence_index,
                "sentence_text": sentence,
            })
    return selected


def build_records(selected: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Attach stable ids, hashes and pending M1 suggestions to each sentence."""
    from engine.tokenizer.max_matching import tokenize
    from engine.tokenizer.preprocessor import normalize

    records: List[Dict[str, object]] = []
    seen: Dict[str, str] = {}
    for position, item in enumerate(selected, start=1):
        sentence = str(item["sentence_text"])
        digest = sha256_text(sentence)
        if digest in seen:
            raise SystemExit(
                f"Duplicate sentence {digest} in {item['category']} "
                f"({item['doc_id']}#{item['sentence_index']} and {seen[digest]})")
        seen[digest] = f"{item['doc_id']}#{item['sentence_index']}"
        sample_id = f"SG{position:03d}"
        tokens = tokenize(sentence)
        records.append({
            "sample_id": sample_id,
            "doc_id": str(item["doc_id"]),
            "category": str(item["category"]),
            "sentence_index": int(item["sentence_index"]),
            "sentence_text": sentence,
            "sentence_sha256": digest,
            "normalized_text": normalize(sentence),
            "whitespace_units": len(sentence.split()),
            "suggestion": {
                "tokens": tokens,
                "source": SUGGESTION_SOURCE,
                "status": SUGGESTION_STATUS,
                "note": "Machine suggestion, NOT gold: must be replaced by a human label.",
            },
            "label": None,
            "label_status": "pending_human_annotation",
        })
    return records


def build_manifest(records: Sequence[Mapping[str, object]], corpus_sha256: str,
                   sample_path: Path, tokenizer: Mapping[str, object],
                   per_category: int, seed: int) -> Dict[str, object]:
    categories: Dict[str, int] = {}
    for record in records:
        categories[str(record["category"])] = categories.get(str(record["category"]), 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "component": "scripts.prepare_segmentation",
        "corpus_file": "data/corpus.jsonl",
        "corpus_sha256": corpus_sha256,
        "sample_file": sample_path.name,
        "sample_sha256": sha256_file(sample_path),
        "normalizer_sha256": sha256_file(ROOT / "engine/tokenizer/preprocessor.py"),
        "labels_file": LABELS_FILE,
        "n_samples": len(records),
        "per_category_target": per_category,
        "seed": seed,
        "category_counts": dict(sorted(categories.items())),
        "sentence_length_units": {"min": MIN_UNITS, "max": MAX_UNITS},
        "tokenizer": dict(tokenizer),
        "gold_status": SUGGESTION_STATUS,
        "status": "pending_human_annotation",
        "official_evaluation_allowed": False,
        "eligibility_rules": {
            "label_status_must_be": "confirmed",
            "assessor_required": True,
            "sentence_sha256_must_match": True,
            "tokens_must_align_to_spans": True,
            "auto_suggestions_never_eligible": True,
        },
        "note": ("Token suggestions are machine output. Official segmentation accuracy may only be "
                 "reported after a human confirms labels; hand-authored unit fixtures are not a "
                 "substitute for this corpus."),
    }


def prepare(corpus_path: Path = DEFAULT_CORPUS, manifest_path: Path = DEFAULT_MANIFEST,
            output: Path = DEFAULT_OUTPUT, per_category: int = PER_CATEGORY,
            seed: int = SEED, force: bool = False) -> Dict[str, object]:
    documents, corpus_sha256 = load_corpus(corpus_path, manifest_path)
    selected = sample_per_category(documents, per_category=per_category, seed=seed)
    records = build_records(selected)
    output.mkdir(parents=True, exist_ok=True)
    sample_path = output / SAMPLE_FILE
    labels_path = output / LABELS_FILE

    if sample_path.exists() and not force:
        raise SystemExit(
            f"{sample_path} already exists; pass --force to rebuild. Existing labels in "
            f"{labels_path} are never deleted by a rebuild.")
    if labels_path.exists() and force:
        existing = [json.loads(line) for line in
                    labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(row.get("label_status") == "confirmed" for row in existing):
            raise SystemExit("Confirmed labels exist; choose a new --output, never overwrite a labeled sample")

    sample_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8")
    manifest = build_manifest(records, corpus_sha256, sample_path, tokenizer_metadata(),
                              per_category, seed)
    (output / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    labels_path.write_text("".join(json.dumps({"sample_id": r["sample_id"],
        "sentence_sha256": r["sentence_sha256"], "tokens": None,
        "label_status": "pending", "assessor": None}, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8")
    return manifest


def check(output: Path = DEFAULT_OUTPUT) -> Dict[str, object]:
    """Validate an existing sample against its manifest without writing anything."""
    manifest_path = output / MANIFEST_FILE
    sample_path = output / SAMPLE_FILE
    if not manifest_path.is_file() or not sample_path.is_file():
        raise SystemExit(f"No sample in {output}; run this script without --check first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line
               in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    problems: List[str] = []
    if sha256_file(sample_path) != manifest["sample_sha256"]:
        problems.append("sample_sha256 does not match the manifest")
    if len(records) != manifest["n_samples"]:
        problems.append(f"n_samples {len(records)} != {manifest['n_samples']}")
    categories: Dict[str, int] = {}
    for record in records:
        categories[record["category"]] = categories.get(record["category"], 0) + 1
        if sha256_text(record["sentence_text"]) != record["sentence_sha256"]:
            problems.append(f"{record['sample_id']}: sentence_sha256 mismatch")
        if record.get("label_status") != "pending_human_annotation" and \
                record.get("label") is not None:
            problems.append(f"{record['sample_id']}: sample file must not carry confirmed labels")
    if categories != manifest["category_counts"]:
        problems.append("category_counts do not match the manifest")
    return {"ok": not problems, "problems": problems,
            "n_samples": len(records), "category_counts": dict(sorted(categories.items())),
            "status": manifest["status"], "corpus_sha256": manifest["corpus_sha256"]}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-category", type=int, default=PER_CATEGORY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="overwrite an existing sample")
    parser.add_argument("--check", action="store_true", help="validate only, write nothing")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.check:
        report = check(args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1

    manifest = prepare(args.corpus, args.manifest, args.output,
                       per_category=args.per_category, seed=args.seed, force=args.force)
    print(f"Wrote {manifest['n_samples']} sentences to {args.output} "
          f"({manifest['status']}); corpus {manifest['corpus_sha256'][:12]}")
    print(f"Suggestions come from {manifest['tokenizer']['tokenizer']} "
          f"(dictionary {str(manifest['tokenizer']['dictionary_sha256'])[:12]}) and are PENDING.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
