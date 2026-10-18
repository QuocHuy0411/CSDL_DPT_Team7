"""Deterministically build the locked VNTC corpus, manifest and statistics.

Raw data under data/raw is read-only. This script never writes there.

Outputs (data/):
    corpus.jsonl          2500 UTF-8 JSONL records, 250 per category
    corpus_manifest.json  provenance + corpus_sha256 lock
    corpus_stats.json     descriptive statistics of the selected corpus

Determinism: every decision is derived from relative source path order and the
CLI seed. No wall-clock time is written into the manifest, so rebuilding with
the same inputs reproduces byte-identical files.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import random
import re
import statistics
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data/raw"
DEFAULT_CORPUS = ROOT / "data/corpus.jsonl"
DEFAULT_MANIFEST = ROOT / "data/corpus_manifest.json"
DEFAULT_STATS = ROOT / "data/corpus_stats.json"

SCHEMA_VERSION = 1
DEFAULT_SEED = 2026
DEFAULT_PER_CATEGORY = 250
MIN_WHITESPACE_UNITS = 100
DOC_ID_PREFIX = "vn_"
DOC_ID_HEX_LENGTH = 16

SOURCE_DATASET = "VNTC"
SOURCE_DATASET_VERSION = "10Topics/Ver1.1"

# Raw folder name -> Vietnamese display label. Folder order is the contract order.
CATEGORIES: tuple[tuple[str, str], ...] = (
    ("Chinh tri Xa hoi", "Chính trị – Xã hội"),
    ("Doi song", "Đời sống"),
    ("Khoa hoc", "Khoa học"),
    ("Kinh doanh", "Kinh doanh"),
    ("Phap luat", "Pháp luật"),
    ("Suc khoe", "Sức khỏe"),
    ("The gioi", "Thế giới"),
    ("The thao", "Thể thao"),
    ("Van hoa", "Văn hóa"),
    ("Vi tinh", "Vi tính"),
)
CATEGORY_BY_FOLDER = dict(CATEGORIES)
CATEGORY_ORDER = [label for _, label in CATEGORIES]

# Headline rule H1 parameters (see docs/data_preparation.md).
HEADLINE_MAX_WHITESPACE_UNITS = 20
HEADLINE_MAX_RATIO_TO_MEDIAN_BODY = 0.75

# Excerpt rule parameters, used when H1 does not hold.
EXCERPT_MAX_WHITESPACE_UNITS = 20
EXCERPT_MIN_SENTENCE_WHITESPACE_UNITS = 4
EXCERPT_SENTENCE_END = re.compile(r"(?<=[.!?…])[ \t]+")
TRUNCATION_MARK = "…"

BOM_SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"\xff\xfe", "utf-16-le-bom", "utf-16"),
    (b"\xfe\xff", "utf-16-be-bom", "utf-16"),
    (b"\xef\xbb\xbf", "utf-8-bom", "utf-8-sig"),
)
# (codec, reported label). Labels are metadata only; codecs are the real ones.
# No legacy codepage fallback: cp1258 accepts any byte string and would silently
# mangle Vietnamese diacritics, so BOM-less unreadable files are excluded instead.
DECODE_ATTEMPTS: tuple[tuple[str, str], ...] = (
    ("utf-16", "utf-16"),
    ("utf-8-sig", "utf-8-sig"),
    ("utf-8", "utf-8"),
)
BOM_CODEC = {"utf-16-le-bom": "utf-16", "utf-16-be-bom": "utf-16", "utf-8-bom": "utf-8-sig"}
WHITESPACE_RUN = re.compile(r"\s+")


class RawDataError(RuntimeError):
    """Raised when the raw tree does not match the audited expectations."""


class CorpusLockedError(RuntimeError):
    """Raised when an existing locked corpus would be overwritten with different content."""


@dataclass(frozen=True)
class RawFile:
    relative_path: str
    folder: str


@dataclass
class RawScan:
    files: list[RawFile] = field(default_factory=list)
    encodings: Counter = field(default_factory=Counter)
    folder_counts: Counter = field(default_factory=Counter)
    folder_bytes: Counter = field(default_factory=Counter)
    unreadable: list[dict] = field(default_factory=list)


@dataclass
class Candidate:
    relative_path: str
    category: str
    title: str
    content: str
    title_origin: str
    content_md5: str
    whitespace_unit_count: int
    source_encoding: str
    title_truncated: bool


class DecodeError(RuntimeError):
    """Raised when raw bytes cannot be decoded without replacement characters."""


def normalize_text(text: str) -> str:
    """NFC normalization plus whitespace collapsing, as locked by the contract."""
    return WHITESPACE_RUN.sub(" ", unicodedata.normalize("NFC", text)).strip()


def detect_encoding(head: bytes) -> str:
    for signature, label, _ in BOM_SIGNATURES:
        if head.startswith(signature):
            return label
    return "none"


def decode_bytes(payload: bytes) -> tuple[str, str]:
    """Decode raw bytes strictly. Returns (text, reported_encoding_label).

    The BOM decides the codec; the fallback chain is only used when no BOM is
    present. Replacement decoding is never used: corrupt bytes raise DecodeError
    so the caller can exclude and report the file instead of silently mangling
    Vietnamese diacritics.
    """
    detected = detect_encoding(payload[:4])
    if detected != "none":
        codec = BOM_CODEC[detected]
        try:
            text = payload.decode(codec)
        except (UnicodeDecodeError, LookupError) as error:
            raise DecodeError(f"{detected}: {error}") from error
        if text.startswith("\ufeff"):
            text = text[1:]
        return text, detected
    for codec, label in DECODE_ATTEMPTS:
        try:
            text = payload.decode(codec)
        except (UnicodeDecodeError, LookupError):
            continue
        if text.startswith("\ufeff"):
            text = text[1:]
        return text, label
    raise DecodeError("no strict codec accepted these bytes")


def split_lines(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.split("\n")


def source_headline(lines: list[str]) -> str | None:
    """Return the headline line when rule H1 holds, else None.

    H1 holds only when the first raw line is a plausible standalone headline:
    at least two non-empty lines, no leading whitespace (VNTC merges headline
    into the first line for some sources), a short line, and clearly shorter
    than the median following paragraph.
    """
    non_empty = [line for line in lines if line.strip()]
    if len(non_empty) < 2:
        return None
    first = lines[0]
    if not first.strip() or first[:1].isspace():
        return None
    headline_units = len(first.split())
    if headline_units > HEADLINE_MAX_WHITESPACE_UNITS:
        return None
    body_units = [len(line.split()) for line in non_empty[1:]]
    median_body = statistics.median(body_units) if body_units else 0.0
    if median_body <= 0:
        return None
    if headline_units > HEADLINE_MAX_RATIO_TO_MEDIAN_BODY * median_body:
        return None
    return normalize_text(first)


def excerpt_title(content: str) -> tuple[str, bool]:
    """Derive a title from the content itself. Never invents wording."""
    window = content.split(" ")[:EXCERPT_MAX_WHITESPACE_UNITS]
    truncated = len(content.split(" ")) > EXCERPT_MAX_WHITESPACE_UNITS
    if not truncated:
        return content, False
    for index, unit in enumerate(window):
        if index < EXCERPT_MIN_SENTENCE_WHITESPACE_UNITS:
            continue
        if EXCERPT_SENTENCE_END.fullmatch(unit):
            return " ".join(window[: index + 1]), False
    return " ".join(window) + " " + TRUNCATION_MARK, True


def derive_title(lines: list[str], content: str) -> tuple[str, str, bool]:
    headline = source_headline(lines)
    if headline is not None:
        return headline, "source_line", False
    excerpt, truncated = excerpt_title(content)
    return excerpt, "inferred_excerpt", truncated


def content_md5(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def doc_id_for(content: str) -> str:
    return DOC_ID_PREFIX + content_md5(content)[:DOC_ID_HEX_LENGTH]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_raw(raw_root: Path) -> RawScan:
    """Enumerate and byte-audit every raw TXT without modifying anything."""
    scan = RawScan()
    if not raw_root.is_dir():
        raise RawDataError(f"Raw directory not found: {raw_root}")
    known_folders = set(CATEGORY_BY_FOLDER)
    for directory, subdirectories, filenames in os.walk(raw_root):
        subdirectories.sort()
        for filename in sorted(filenames):
            if not filename.lower().endswith(".txt"):
                continue
            path = Path(directory) / filename
            relative = path.relative_to(raw_root).as_posix()
            folder = relative.split("/")[0]
            if folder not in known_folders:
                continue
            payload = path.read_bytes()
            scan.encodings[detect_encoding(payload[:4])] += 1
            scan.folder_counts[folder] += 1
            scan.folder_bytes[folder] += len(payload)
            scan.files.append(RawFile(relative, folder))
    scan.files.sort(key=lambda item: item.relative_path)
    return scan


def load_candidate(item: RawFile, raw_root: Path) -> Candidate:
    """Decode and normalize one raw file. Raises DecodeError on corrupt bytes."""
    payload = (raw_root / item.relative_path).read_bytes()
    text, encoding = decode_bytes(payload)
    content = normalize_text(text)
    title, origin, truncated = derive_title(split_lines(text), content)
    return Candidate(
        relative_path=item.relative_path,
        category=CATEGORY_BY_FOLDER[item.folder],
        title=title,
        content=content,
        title_origin=origin,
        content_md5=content_md5(content),
        whitespace_unit_count=len(content.split()),
        source_encoding=encoding,
        title_truncated=truncated,
    )


def category_seed(seed: int, folder: str) -> int:
    """Explicit, reproducible per-category seed derived from the CLI seed."""
    digest = hashlib.sha256(f"{seed}|{folder}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _folder_of(category: str) -> str:
    for folder, label in CATEGORIES:
        if label == category:
            return folder
    raise KeyError(category)


def global_dedup(eligible: list[Candidate]) -> tuple[list[Candidate], list[dict]]:
    """Deduplicate the whole eligible input by MD5 in ascending source path order.

    The first path that carries a given MD5 owns the document. Every later copy
    is recorded, including copies that sit in a different category folder, so the
    report shows the complete duplicate picture of the input rather than only the
    duplicates that happened to fall inside the 250-document sample.
    """
    ordered = sorted(eligible, key=lambda item: item.relative_path)
    owner_by_md5: dict[str, str] = {}
    unique: list[Candidate] = []
    duplicates: list[dict] = []
    for candidate in ordered:
        owner = owner_by_md5.get(candidate.content_md5)
        if owner is None:
            owner_by_md5[candidate.content_md5] = candidate.relative_path
            unique.append(candidate)
            continue
        duplicates.append({
            "source_file": candidate.relative_path,
            "duplicate_of": owner,
            "content_md5": candidate.content_md5,
            "source_folder": _folder_of(candidate.category),
            "kept_folder": _folder_of(CATEGORY_BY_FOLDER[owner.split("/")[0]]),
            "cross_category": _folder_of(candidate.category) != owner.split("/")[0],
        })
    return unique, duplicates


def duplicate_report(duplicates: list[dict], eligible_count: int, unique_count: int) -> dict:
    cross = [item for item in duplicates if item["cross_category"]]
    pairs = Counter(
        tuple(sorted((item["kept_folder"], item["source_folder"]))) for item in duplicates)
    return {
        "scope": "all eligible input documents, before sampling",
        "eligible_documents": eligible_count,
        "unique_documents": unique_count,
        "duplicate_copies_removed": len(duplicates),
        "cross_category_duplicate_copies": len(cross),
        "within_category_duplicate_copies": len(duplicates) - len(cross),
        "cross_category_folder_pairs": {f"{a} <-> {b}": count
                                       for (a, b), count in sorted(pairs.items())},
        "examples": duplicates[:10],
    }


def rank_candidates(unique: list[Candidate], seed: int) -> dict[str, list[Candidate]]:
    """Order each category's deduplicated pool deterministically."""
    by_folder: dict[str, list[Candidate]] = {folder: [] for folder, _ in CATEGORIES}
    for candidate in unique:
        by_folder[_folder_of(candidate.category)].append(candidate)
    ranked: dict[str, list[Candidate]] = {}
    for folder, _ in CATEGORIES:
        pool = sorted(by_folder[folder], key=lambda item: item.relative_path)
        order = list(range(len(pool)))
        random.Random(category_seed(seed, folder)).shuffle(order)
        ranked[folder] = [pool[index] for index in order]
    return ranked


def select_corpus(candidates: list[Candidate], seed: int,
                  per_category: int) -> tuple[list[Candidate], dict]:
    """Filter, deduplicate globally, then sample per category."""
    eligible = [item for item in candidates if item.whitespace_unit_count >= MIN_WHITESPACE_UNITS]
    rejected_short = [item for item in candidates
                      if item.whitespace_unit_count < MIN_WHITESPACE_UNITS]
    unique, duplicates = global_dedup(eligible)
    if len({item.content_md5 for item in unique}) != len(unique):
        raise RawDataError("Global deduplication failed to produce unique content")
    ranked = rank_candidates(unique, seed)
    accepted: list[Candidate] = []
    per_category_accepted: Counter = Counter()
    for folder, label in CATEGORIES:
        taken = ranked[folder][:per_category]
        if len(taken) < per_category:
            raise RawDataError(
                f"Category {label!r} only produced {len(taken)} unique documents, "
                f"need {per_category}")
        accepted.extend(taken)
        per_category_accepted[label] = len(taken)
    selection = {
        "eligible_documents": len(eligible),
        "rejected_below_min_whitespace_units": len(rejected_short),
        "rejected_below_min_by_folder": dict(sorted(
            Counter(_folder_of(item.category) for item in rejected_short).items())),
        "rejected_below_min_max_whitespace_units": max(
            (item.whitespace_unit_count for item in rejected_short), default=0),
        "deduplication": duplicate_report(duplicates, len(eligible), len(unique)),
        "pool_size_after_dedup": {label: sum(1 for item in unique
                                             if item.category == label)
                                  for _, label in CATEGORIES},
        "selected_per_category": {label: per_category_accepted[label] for _, label in CATEGORIES},
        "seed": seed,
        "per_category_target": per_category,
    }
    return accepted, selection


def build_records(selected: list[Candidate]) -> list[dict]:
    records = []
    for candidate in sorted(selected, key=lambda item: item.relative_path):
        records.append({
            "doc_id": doc_id_for(candidate.content),
            "title": candidate.title,
            "content": candidate.content,
            "category": candidate.category,
            "metadata": {
                "source_file": candidate.relative_path,
                "source_dataset": SOURCE_DATASET,
                "source_dataset_version": SOURCE_DATASET_VERSION,
                "source_encoding": candidate.source_encoding,
                "content_md5": candidate.content_md5,
                "whitespace_unit_count": candidate.whitespace_unit_count,
                "title_origin": candidate.title_origin,
                "title_truncated": candidate.title_truncated,
            },
        })
    return records


def serialize_corpus(records: list[dict]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n"
                   for record in records)


def describe(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": ordered[len(ordered) // 4],
        "median": int(statistics.median(ordered)),
        "mean": round(statistics.fmean(ordered), 2),
        "p75": ordered[3 * len(ordered) // 4],
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
        "total": sum(ordered),
    }


def build_stats(records: list[dict], scan: RawScan, selection: dict, corpus_bytes: int) -> dict:
    units = [record["metadata"]["whitespace_unit_count"] for record in records]
    titles = [len(record["title"].split()) for record in records]
    encodings = Counter(record["metadata"]["source_encoding"] for record in records)
    origins = Counter(record["metadata"]["title_origin"] for record in records)
    per_category = Counter(record["category"] for record in records)
    prefixes = Counter(record["metadata"]["source_file"].split("/")[0] for record in records)
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_documents": len(records),
        "corpus_bytes": corpus_bytes,
        "raw_input": {
            "txt_files_scanned": len(scan.files),
            "folder_file_counts": dict(sorted(scan.folder_counts.items())),
            "folder_bytes": dict(sorted(scan.folder_bytes.items())),
            "detected_encodings": dict(sorted(scan.encodings.items())),
            "undecodable_files": selection["undecodable_documents_skipped"],
            "undecodable_examples": selection["undecodable_examples"],
            "empty_after_normalization": selection["empty_documents_skipped"],
        },
        "selection": selection,
        "length": {
            "definition": "whitespace_unit_count == len(content.split()) after NFC + whitespace collapsing",
            "min_whitespace_units_threshold": MIN_WHITESPACE_UNITS,
            "whitespace_units": describe(units),
            "title_whitespace_units": describe(titles),
            "documents_below_threshold": sum(1 for value in units if value < MIN_WHITESPACE_UNITS),
        },
        "category_counts": {label: per_category[label] for label in CATEGORY_ORDER},
        "raw_folder_counts_in_corpus": {folder: prefixes[folder] for folder, _ in CATEGORIES},
        "title_origin_counts": dict(sorted(origins.items())),
        "title_truncated_count": sum(1 for record in records
                                     if record["metadata"]["title_truncated"]),
        "source_encoding_counts": dict(sorted(encodings.items())),
        "duplicates": {
            "dedup_algorithm": "MD5 of NFC + whitespace-collapsed content, global across categories",
            "unique_content_md5": len({record["metadata"]["content_md5"] for record in records}),
            "duplicate_records": len(records) - len({record["metadata"]["content_md5"]
                                                      for record in records}),
        },
    }


def locked_sha256(manifest_path: Path) -> str | None:
    """Return the corpus SHA256 an existing manifest has locked, if any."""
    if not manifest_path.is_file():
        return None
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return previous.get("corpus_sha256") or previous.get("sha256")


def guard_locked_outputs(corpus_path: Path, manifest_path: Path, allow_overwrite: bool) -> dict | None:
    """Phase 1: refuse to run against an already inconsistent locked pair."""
    if allow_overwrite or not manifest_path.is_file():
        return None
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock = previous.get("lock", previous)
    locked_sha = lock.get("locked_sha256") or previous.get("corpus_sha256")
    if not locked_sha:
        return None
    if not corpus_path.is_file():
        raise CorpusLockedError(
            f"{manifest_path} locks corpus_sha256={locked_sha} but {corpus_path} is missing; "
            "refusing to write an unlocked corpus over a locked manifest")
    current_sha = sha256_file(corpus_path)
    if current_sha != locked_sha:
        raise CorpusLockedError(
            f"{corpus_path} sha256={current_sha} does not match locked {locked_sha}; "
            "rebuild into a new path or pass --allow-overwrite-lock")
    return previous


def guard_new_lock(candidate_sha256: str, manifest_path: Path, allow_overwrite: bool) -> dict | None:
    """Phase 2: compare the NEW corpus hash with the previously locked hash.

    This is the real overwrite guard: a locked corpus may only be replaced by
    byte-identical content, or by an explicit --allow-overwrite-lock.
    """
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    if allow_overwrite or previous is None:
        return previous
    locked_sha = previous.get("corpus_sha256") or previous.get("sha256")
    if locked_sha and locked_sha != candidate_sha256:
        raise CorpusLockedError(
            f"new corpus sha256={candidate_sha256} differs from locked {locked_sha} in "
            f"{manifest_path}; relock explicitly with --allow-overwrite-lock")
    return previous


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def build_corpus(raw_root: Path = DEFAULT_RAW, corpus_path: Path = DEFAULT_CORPUS,
                 manifest_path: Path = DEFAULT_MANIFEST, stats_path: Path = DEFAULT_STATS,
                 seed: int = DEFAULT_SEED, per_category: int = DEFAULT_PER_CATEGORY,
                 min_whitespace_units: int = MIN_WHITESPACE_UNITS,
                 allow_overwrite_lock: bool = False, audit_only: bool = False,
                 quiet: bool = False) -> dict:
    """Scan raw, select documents deterministically and emit corpus artifacts."""
    if min_whitespace_units != MIN_WHITESPACE_UNITS:
        raise RawDataError("This build is locked to the contract threshold of 100 whitespace units")
    previous = guard_locked_outputs(corpus_path, manifest_path, allow_overwrite_lock)
    scan = scan_raw(raw_root)
    if not scan.files:
        raise RawDataError(f"No .txt files found under {raw_root}")
    missing = [folder for folder, _ in CATEGORIES if folder not in scan.folder_counts]
    if missing:
        raise RawDataError(f"Missing raw category folders: {missing}")

    def log(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    log(f"Scanned {len(scan.files)} raw TXT files in {len(scan.folder_counts)} folders")
    log(f"Detected encodings: {dict(scan.encodings)}")
    if audit_only:
        return {"audit": {"txt_files": len(scan.files),
                          "folder_file_counts": dict(sorted(scan.folder_counts.items())),
                          "detected_encodings": dict(sorted(scan.encodings.items()))}}

    candidates: list[Candidate] = []
    empty: list[str] = []
    unreadable: list[dict] = []
    for item in scan.files:
        try:
            candidate = load_candidate(item, raw_root)
        except DecodeError as error:
            unreadable.append({"source_file": item.relative_path, "error": str(error)})
            continue
        if not candidate.content:
            empty.append(item.relative_path)
            continue
        candidates.append(candidate)
    if unreadable:
        log(f"Excluded {len(unreadable)} files that no strict codec could decode")
    log(f"Decoded {len(candidates)} documents "
        f"({len(empty)} empty after normalization, {len(unreadable)} unreadable)")

    selected, selection = select_corpus(candidates, seed, per_category)
    selection["empty_documents_skipped"] = len(empty)
    selection["undecodable_documents_skipped"] = len(unreadable)
    selection["undecodable_examples"] = unreadable[:10]
    records = build_records(selected)
    corpus_text = serialize_corpus(records)
    corpus_bytes = corpus_text.encode("utf-8")
    corpus_sha256 = hashlib.sha256(corpus_bytes).hexdigest()
    previous = guard_new_lock(corpus_sha256, manifest_path, allow_overwrite_lock)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus_file": corpus_path.name,
        "corpus_sha256": corpus_sha256,
        "corpus_bytes": len(corpus_bytes),
        "corpus_documents": len(records),
        "locked": True,
        "lock": {
            "locked": True,
            "locked_sha256": corpus_sha256,
            "locked_by": "scripts/build_corpus.py",
            "locked_before_annotation": True,
            "note": "qrels and query artifacts must record this exact corpus_sha256",
        },
        "supersedes_sha256": previous.get("corpus_sha256") if previous else None,
        "source": {
            "dataset": SOURCE_DATASET,
            "version": SOURCE_DATASET_VERSION,
            "raw_root": raw_root.relative_to(ROOT).as_posix() if raw_root.is_relative_to(ROOT)
                        else raw_root.as_posix(),
            "raw_is_read_only": True,
            "txt_files_scanned": len(scan.files),
            "detected_encodings": dict(sorted(scan.encodings.items())),
            "undecodable_files_skipped": selection["undecodable_documents_skipped"],
            "empty_files_skipped": selection["empty_documents_skipped"],
        },
        "build": {
            "seed": seed,
            "per_category_target": per_category,
            "categories": [{"raw_folder": folder, "category": label} for folder, label in CATEGORIES],
            "min_whitespace_units": min_whitespace_units,
            "whitespace_unit_definition": "len(content.split()) after NFC + whitespace collapsing",
            "normalization": "unicode NFC, all whitespace runs collapsed to a single space, trimmed",
            "line_endings": "CRLF or LF in raw, collapsed away by normalization",
            "dedup": "MD5 of normalized content over all eligible input, "
                     "ascending source_file order, before per-category sampling",
            "sampling": "per-category shuffle of the deduplicated pool with "
                        "seed derived as sha256('<seed>|<raw folder>')",
            "record_order": "ascending source_file relative path",
            "doc_id_scheme": f"{DOC_ID_PREFIX} + first {DOC_ID_HEX_LENGTH} hex chars of content MD5",
            "tokenizer": "none; corpus keeps untokenized text for M1",
        },
        "title_rule": {
            "source_line": {
                "description": "first raw line accepted as a headline",
                "requires": [
                    "at least two non-empty lines",
                    "first raw line does not start with whitespace",
                    f"first line <= {HEADLINE_MAX_WHITESPACE_UNITS} whitespace units",
                    f"first line <= {HEADLINE_MAX_RATIO_TO_MEDIAN_BODY} x median following line units",
                ],
            },
            "inferred_excerpt": {
                "description": "no reliable headline; title derived from the content itself",
                "method": "first sentence within the first "
                          f"{EXCERPT_MAX_WHITESPACE_UNITS} whitespace units, else that prefix",
                "truncation_mark": TRUNCATION_MARK,
                "never_fabricated": True,
            },
        },
        "counts": {
            "by_category": {label: per_category for _, label in CATEGORIES},
            "eligible_documents": selection["eligible_documents"],
            "rejected_below_min_whitespace_units":
                selection["rejected_below_min_whitespace_units"],
            "duplicate_copies_removed":
                selection["deduplication"]["duplicate_copies_removed"],
            "cross_category_duplicate_copies":
                selection["deduplication"]["cross_category_duplicate_copies"],
            "pool_size_after_dedup": selection["pool_size_after_dedup"],
        },
        "deduplication": selection["deduplication"],
        "license_note": "Local VNTC-format news data supplied for this project. "
                        "Redistribution rights for the news texts have not been verified; "
                        "data/raw stays local and untracked as requested.",
    }
    stats = build_stats(records, scan, selection, len(corpus_bytes))

    write_text(corpus_path, corpus_text)
    write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_text(stats_path, json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    log(f"Wrote {corpus_path} ({len(records)} docs, sha256={corpus_sha256})")
    log(f"Title origins: {stats['title_origin_counts']}")
    return {"records": records, "manifest": manifest, "stats": stats,
            "corpus_sha256": corpus_sha256}


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page that cannot print Vietnamese.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--per-category", type=int, default=DEFAULT_PER_CATEGORY)
    parser.add_argument("--audit-only", action="store_true",
                        help="Only byte-audit the raw tree, do not write anything")
    parser.add_argument("--allow-overwrite-lock", action="store_true",
                        help="Permit replacing a locked corpus whose SHA256 differs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        build_corpus(raw_root=args.raw, corpus_path=args.corpus, manifest_path=args.manifest,
                     stats_path=args.stats, seed=args.seed, per_category=args.per_category,
                     allow_overwrite_lock=args.allow_overwrite_lock,
                     audit_only=args.audit_only, quiet=args.quiet)
    except (RawDataError, CorpusLockedError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
