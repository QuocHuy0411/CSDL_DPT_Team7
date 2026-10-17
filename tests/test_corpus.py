"""Tests for the locked corpus build: raw audit, dedup, sampling, lock."""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_corpus import (  # noqa: E402
    CATEGORIES,
    CATEGORY_ORDER,
    DEFAULT_SEED,
    MIN_WHITESPACE_UNITS,
    CorpusLockedError,
    DecodeError,
    RawDataError,
    build_corpus,
    decode_bytes,
    derive_title,
    excerpt_title,
    global_dedup,
    guard_new_lock,
    load_candidate,
    normalize_text,
    scan_raw,
    select_corpus,
    sha256_file,
    source_headline,
    split_lines,
)
from scripts.build_corpus import Candidate  # noqa: E402

BUILT_CORPUS = ROOT / "data/corpus.jsonl"
BUILT_MANIFEST = ROOT / "data/corpus_manifest.json"
BUILT_STATS = ROOT / "data/corpus_stats.json"


def long_text(marker: str, units: int = 140) -> str:
    body = " ".join(f"{marker}{index}" for index in range(units))
    return f"Tiêu đề {marker}\n{body}\n"


def write_raw(root: Path, folder: str, name: str, text: str, encoding: str = "utf-16") -> Path:
    path = root / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode(encoding))
    return path


def candidate(folder: str, name: str, marker: str) -> Candidate:
    text = long_text(marker)
    return Candidate(
        relative_path=f"{folder}/{name}",
        category=dict(CATEGORIES)[folder],
        title=f"Tiêu đề {marker}",
        content=normalize_text(text),
        title_origin="source_line",
        content_md5=__import__("hashlib").md5(normalize_text(text).encode()).hexdigest(),
        whitespace_unit_count=len(normalize_text(text).split()),
        source_encoding="utf-16-le-bom",
        title_truncated=False,
    )


# --------------------------------------------------------------------------- #
# normalization and decoding
# --------------------------------------------------------------------------- #

def test_normalize_collapses_all_whitespace_runs_to_one_space():
    raw = "  Tiêu đề\r\n\r\nĐoạn\t một   hai \n\n"
    assert normalize_text(raw) == "Tiêu đề Đoạn một hai"


def test_normalize_is_nfc_and_idempotent():
    decomposed = "Tiêu đề"  # NFC input stays stable
    once = normalize_text(decomposed)
    assert once == unicodedata.normalize("NFC", once)
    assert normalize_text(once) == once


def test_decode_bytes_reports_bom_label_and_strips_bom():
    text, label = decode_bytes("Xin chào".encode("utf-16"))
    assert text == "Xin chào"
    assert label == "utf-16-le-bom"
    assert not text.startswith("\ufeff")


def test_decode_bytes_uses_real_codec_for_utf8_bom():
    text, label = decode_bytes(b"\xef\xbb\xbfXin ch\xc3\xa0o")
    assert text == "Xin chào"
    assert label == "utf-8-bom"


def test_decode_bytes_rejects_corrupt_bytes_instead_of_replacing():
    # Odd byte count after the BOM cannot form a UTF-16 code unit.
    with pytest.raises(DecodeError):
        decode_bytes(b"\xff\xfe\x41\x00\x42")
    # Unpaired high surrogate.
    with pytest.raises(DecodeError):
        decode_bytes(b"\xff\xfe\x00\xd8\x41\x00")


def test_decode_bytes_never_returns_replacement_characters():
    # BOM-declared but invalid code units must raise, never be silently repaired.
    with pytest.raises(DecodeError):
        decode_bytes(b"\xff\xfe\x00\xd8\x41\x00")
    # BOM-less invalid bytes are excluded rather than decoded as a legacy codepage.
    with pytest.raises(DecodeError):
        decode_bytes(b"\x81\x82\x83")
    for payload in (b"\xff\xfeXin chao".ljust(14, b"\x00"), "Tiếng Việt".encode("utf-8")):
        text, _ = decode_bytes(payload)
        assert "�" not in text


# --------------------------------------------------------------------------- #
# title provenance
# --------------------------------------------------------------------------- #

def test_source_headline_requires_a_separate_short_first_line():
    lines = split_lines("Một tiêu đề ngắn\n" + "\n".join(["câu " * 40] * 3))
    assert source_headline(lines) == "Một tiêu đề ngắn"


def test_source_headline_rejects_leading_whitespace_merged_line():
    lines = split_lines(" Tiêu đề bị nối vào đoạn văn " + "câu " * 40 + "\n" + "thân " * 40)
    assert source_headline(lines) is None


def test_source_headline_rejects_first_line_as_long_as_the_body():
    lines = split_lines("câu " * 40 + "\n" + "thân " * 40)
    assert source_headline(lines) is None


def test_source_headline_rejects_single_line_documents():
    assert source_headline(split_lines("chỉ có một dòng")) is None


def test_excerpt_title_is_a_verbatim_prefix_and_marks_truncation():
    content = normalize_text("Alpha beta gamma. delta epsilon zeta eta theta iota kappa lambda mu nu")
    title, truncated = excerpt_title(content)
    assert title.startswith("Alpha beta gamma.")
    assert truncated is False
    long_content = normalize_text(" ".join(f"w{index}" for index in range(60)))
    title, truncated = excerpt_title(long_content)
    assert truncated is True
    assert title.endswith("…")
    assert title.rstrip(" …").split() == long_content.split()[:20]


def test_derive_title_marks_inferred_excerpt_when_no_headline():
    text = "không có tiêu đề riêng " + "nội dung " * 60
    title, origin, _ = derive_title(split_lines(text), normalize_text(text))
    assert origin == "inferred_excerpt"
    assert title.rstrip(" …").strip() in normalize_text(text)


# --------------------------------------------------------------------------- #
# global deduplication over the full eligible input
# --------------------------------------------------------------------------- #

def test_global_dedup_is_first_path_wins_and_reports_cross_category():
    items = [
        candidate("Chinh tri Xa hoi", "a.txt", "A"),
        candidate("Chinh tri Xa hoi", "b.txt", "A"),  # same content, later path
        candidate("Phap luat", "c.txt", "A"),  # cross-category duplicate
        candidate("Phap luat", "d.txt", "D"),
    ]
    unique, duplicates = global_dedup(items)
    assert [item.relative_path for item in unique] == [
        "Chinh tri Xa hoi/a.txt", "Phap luat/d.txt"]
    assert len(duplicates) == 2
    assert all(item["duplicate_of"] == "Chinh tri Xa hoi/a.txt" for item in duplicates)
    assert [item["cross_category"] for item in duplicates] == [False, True]
    assert duplicates[1]["kept_folder"] == "Chinh tri Xa hoi"
    assert duplicates[1]["source_folder"] == "Phap luat"


def filler_for_all_categories(folder_counts: dict[str, int] | None = None) -> list[Candidate]:
    """Build filler candidates for every category that needs a full pool."""
    items = []
    for folder, _ in CATEGORIES:
        count = (folder_counts or {}).get(folder, 260)
        for index in range(count):
            items.append(candidate(folder, f"f{index:03d}.txt", f"{folder}-{index}"))
    return items


def test_global_dedup_runs_before_sampling_not_inside_it():
    # 400 identical documents in one folder must not consume the 250 quota.
    items = [candidate("Khoa hoc", f"dup{index:03d}.txt", "SAME") for index in range(400)]
    items += [candidate("Khoa hoc", f"u{index:03d}.txt", f"U{index}") for index in range(300)]
    items += [item for item in filler_for_all_categories() if item.category != "Khoa học"]
    selected, selection = select_corpus(items, DEFAULT_SEED, 250)
    md5s = [item.content_md5 for item in selected]
    assert len(selected) == 2500
    assert len(set(md5s)) == 2500
    dedup = selection["deduplication"]
    assert dedup["eligible_documents"] == 700 + 9 * 260
    assert dedup["duplicate_copies_removed"] == 399
    assert dedup["scope"] == "all eligible input documents, before sampling"


def test_global_dedup_counts_duplicates_even_when_they_never_get_sampled():
    # Duplicates that lose the sampling race must still appear in the report.
    items = [candidate("Suc khoe", f"u{index:03d}.txt", f"U{index}") for index in range(400)]
    items += [candidate("Suc khoe", f"dup{index:03d}.txt", "SAME") for index in range(50)]
    items += [item for item in filler_for_all_categories() if item.category != "Sức khỏe"]
    selected, selection = select_corpus(items, DEFAULT_SEED, 250)
    assert len(selected) == 2500
    assert selection["deduplication"]["duplicate_copies_removed"] == 49


def test_select_corpus_reports_every_category_pool_after_dedup():
    items = filler_for_all_categories()
    _, selection = select_corpus(items, DEFAULT_SEED, 250)
    assert set(selection["pool_size_after_dedup"].values()) == {260}
    assert set(selection["selected_per_category"].values()) == {250}


def test_select_corpus_refuses_when_a_category_is_too_small():
    items = filler_for_all_categories({"Khoa hoc": 10})
    with pytest.raises(RawDataError):
        select_corpus(items, DEFAULT_SEED, 250)


def test_select_corpus_refuses_when_a_category_folder_is_absent():
    items = [item for item in filler_for_all_categories()
             if item.category != "Chính trị – Xã hội"]
    with pytest.raises(RawDataError):
        select_corpus(items, DEFAULT_SEED, 250)


# --------------------------------------------------------------------------- #
# length threshold
# --------------------------------------------------------------------------- #

def test_threshold_is_whitespace_units_not_segmented_words():
    # 97 body units plus a 3-unit headline line == 100 whitespace units.
    content = normalize_text(long_text("A", units=MIN_WHITESPACE_UNITS - 3))
    assert len(content.split()) == MIN_WHITESPACE_UNITS
    assert content == " ".join(content.split())
    short = " ".join(["w"] * (MIN_WHITESPACE_UNITS - 1))
    items = [Candidate("Khoa hoc/x.txt", "Khoa học", "t", short, "inferred_excerpt",
                       "d" * 32, len(short.split()), "utf-16-le-bom", False)]
    _, selection = select_corpus(items, DEFAULT_SEED, 0)
    assert selection["rejected_below_min_whitespace_units"] == 1
    assert selection["rejected_below_min_max_whitespace_units"] == MIN_WHITESPACE_UNITS - 1
    assert selection["eligible_documents"] == 0


# --------------------------------------------------------------------------- #
# seed and lock behaviour
# --------------------------------------------------------------------------- #

def test_same_seed_reproduces_the_same_selection():
    items = [candidate(folder, f"d{index:03d}.txt", f"{folder}-{index}")
             for folder, _ in CATEGORIES for index in range(260)]
    first, _ = select_corpus(items, DEFAULT_SEED, 250)
    second, _ = select_corpus(items, DEFAULT_SEED, 250)
    assert [item.relative_path for item in first] == [item.relative_path for item in second]


def test_changing_the_seed_changes_the_selection_but_not_the_sizes():
    items = [candidate(folder, f"d{index:03d}.txt", f"{folder}-{index}")
             for folder, _ in CATEGORIES for index in range(260)]
    base, _ = select_corpus(items, DEFAULT_SEED, 250)
    other, _ = select_corpus(items, DEFAULT_SEED + 1, 250)
    assert [item.relative_path for item in base] != [item.relative_path for item in other]
    assert sorted(item.category for item in base) == sorted(item.category for item in other)


def test_guard_new_lock_refuses_a_different_corpus_hash(tmp_path):
    manifest = tmp_path / "corpus_manifest.json"
    manifest.write_text(json.dumps({"corpus_sha256": "locked-hash", "locked": True}),
                        encoding="utf-8")
    with pytest.raises(CorpusLockedError):
        guard_new_lock("different-hash", manifest, allow_overwrite=False)
    assert guard_new_lock("locked-hash", manifest, allow_overwrite=False)
    assert guard_new_lock("different-hash", manifest, allow_overwrite=True)


def test_end_to_end_build_is_reproducible_and_locked(tmp_path):
    raw = tmp_path / "raw"
    for folder, _ in CATEGORIES:
        for index in range(6):
            write_raw(raw, folder, f"f{index}.txt", long_text(f"{folder}-{index}", units=130))
    # one cross-category duplicate of an existing document
    write_raw(raw, "Phap luat", "dup.txt", long_text(f"{CATEGORIES[0][0]}-0", units=130))
    corpus = tmp_path / "corpus.jsonl"
    manifest = tmp_path / "corpus_manifest.json"
    stats = tmp_path / "corpus_stats.json"
    result = build_corpus(raw_root=raw, corpus_path=corpus, manifest_path=manifest,
                          stats_path=stats, per_category=3, quiet=True)
    assert len(result["records"]) == 30
    assert sha256_file(corpus) == result["corpus_sha256"]
    assert result["manifest"]["corpus_sha256"] == result["corpus_sha256"]
    assert result["stats"]["selection"]["deduplication"]["duplicate_copies_removed"] == 1
    first_digest = sha256_file(corpus)
    build_corpus(raw_root=raw, corpus_path=corpus, manifest_path=manifest,
                 stats_path=stats, per_category=3, quiet=True)
    assert sha256_file(corpus) == first_digest
    with pytest.raises(CorpusLockedError):
        build_corpus(raw_root=raw, corpus_path=corpus, manifest_path=manifest,
                     stats_path=stats, per_category=2, quiet=True)


def test_scan_raw_audits_every_txt_and_ignores_non_txt(tmp_path):
    raw = tmp_path / "raw"
    write_raw(raw, "Khoa hoc", "a.txt", long_text("A"))
    write_raw(raw, "Khoa hoc", "b.txt", long_text("B"))
    (raw / "Khoa hoc" / "readme.md").write_text("ignore me", encoding="utf-8")
    scan = scan_raw(raw)
    assert len(scan.files) == 2
    assert scan.encodings == {"utf-16-le-bom": 2}
    assert scan.folder_counts == {"Khoa hoc": 2}
    assert [item.relative_path for item in scan.files] == ["Khoa hoc/a.txt", "Khoa hoc/b.txt"]


def test_load_candidate_keeps_provenance(tmp_path):
    raw = tmp_path / "raw"
    path = write_raw(raw, "Khoa hoc", "a.txt", long_text("A"))
    loaded = load_candidate(scan_raw(raw).files[0], raw)
    assert loaded.relative_path == "Khoa hoc/a.txt"
    assert loaded.category == "Khoa học"
    assert loaded.source_encoding == "utf-16-le-bom"
    assert loaded.content_md5 == __import__("hashlib").md5(loaded.content.encode()).hexdigest()
    assert loaded.whitespace_unit_count == len(loaded.content.split())
    assert path.exists()


# --------------------------------------------------------------------------- #
# the real locked corpus
# --------------------------------------------------------------------------- #

requires_corpus = pytest.mark.skipif(not BUILT_CORPUS.is_file(),
                                    reason="data/corpus.jsonl has not been built")

CORPUS = None


def load_built_corpus():
    global CORPUS
    if CORPUS is None:
        CORPUS = [json.loads(line) for line in
                  BUILT_CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    return CORPUS


@requires_corpus
def test_manifest_lock_matches_the_corpus_file():
    manifest = json.loads(BUILT_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["corpus_sha256"] == sha256_file(BUILT_CORPUS)
    assert manifest["locked"] is True
    assert manifest["build"]["seed"] == DEFAULT_SEED


@requires_corpus
def test_corpus_has_2500_documents_250_per_category():
    records = load_built_corpus()
    assert len(records) == 2500
    counts = {label: 0 for label in CATEGORY_ORDER}
    for record in records:
        counts[record["category"]] += 1
    assert set(counts) == set(CATEGORY_ORDER)
    assert set(counts.values()) == {250}


@requires_corpus
def test_corpus_normalization_invariants():
    import re
    for record in load_built_corpus():
        content = record["content"]
        assert content == unicodedata.normalize("NFC", content)
        assert content == " ".join(content.split())
        assert not re.search(r"[\n\r\t]", content)
        assert record["metadata"]["whitespace_unit_count"] == len(content.split())
        assert record["metadata"]["whitespace_unit_count"] >= MIN_WHITESPACE_UNITS


@requires_corpus
def test_corpus_is_globally_deduplicated():
    records = load_built_corpus()
    digests = [record["metadata"]["content_md5"] for record in records]
    assert len(set(digests)) == 2500
    assert len({record["doc_id"] for record in records}) == 2500


@requires_corpus
def test_corpus_doc_ids_are_content_hash_derived_and_stable():
    import hashlib
    for record in load_built_corpus():
        digest = hashlib.md5(record["content"].encode("utf-8")).hexdigest()
        assert record["metadata"]["content_md5"] == digest
        assert record["doc_id"] == "vn_" + digest[:16]


@requires_corpus
def test_corpus_record_order_is_stable_relative_path_order():
    paths = [record["metadata"]["source_file"] for record in load_built_corpus()]
    assert paths == sorted(paths)


@requires_corpus
def test_corpus_titles_are_never_invented():
    allowed = {"source_line", "inferred_excerpt"}
    for record in load_built_corpus():
        metadata = record["metadata"]
        assert metadata["title_origin"] in allowed
        assert record["title"].rstrip(" …").strip() in record["content"]


@requires_corpus
def test_corpus_metadata_has_no_fabricated_urls():
    for record in load_built_corpus():
        keys = set(record) | set(record["metadata"])
        assert not any("url" in key.lower() for key in keys)


@requires_corpus
def test_corpus_stats_agree_with_the_corpus_file():
    stats = json.loads(BUILT_STATS.read_text(encoding="utf-8"))
    records = load_built_corpus()
    assert stats["corpus_documents"] == len(records)
    assert stats["duplicates"]["unique_content_md5"] == len(records)
    assert stats["length"]["documents_below_threshold"] == 0
    assert stats["raw_input"]["txt_files_scanned"] == 33759
    assert stats["raw_input"]["undecodable_files"] == 0
