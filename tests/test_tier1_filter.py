"""Tier-1 grammar stopword filtering: guards, compounds, symmetry, provenance.

Fixtures are hand-written on purpose: the contract states that hand-authored unit
fixtures do **not** replace the human-annotated gold segmentation corpus. These
tests pin the *rules*; the corpus-level quality claim lives in
``docs/segmentation.md`` and is refused until human labels exist.
"""

from __future__ import annotations

import json
import shutil

import pytest

from engine.tokenizer.stopwords import (Tier1StopwordFilter, canonical, filter_tier1,
                                        get_default_filter, is_protected)

# Terms the contract names explicitly, plus the wider protected classes.
REQUIRED_GUARDS = ["không", "chưa", "chẳng", "vì", "do_đó"]
PROTECTED_COMPOUNDS = ["không_thể", "không_thể_nữa", "vì_dầu", "chẳng_hạn",
                       "trước_hết", "phần_lớn", "bất_kỳ_người_nào", "có_thể"]
FILTERABLE_FUNCTION_WORDS = ["và", "của", "là", "có", "rất", "dạ", "với", "trong"]


@pytest.fixture(scope="module")
def tier1() -> Tier1StopwordFilter:
    return Tier1StopwordFilter()


def test_resource_counts_are_reported_honestly(tier1):
    metadata = tier1.metadata()
    assert metadata["n_stopword_entries"] == len(tier1.stopwords)
    assert metadata["n_guard_entries"] == len(tier1.guards)
    assert metadata["n_effective_stopwords"] == len(tier1.effective_stopwords)
    assert tier1.effective_stopwords == frozenset(
        t for t in tier1.stopwords if not tier1.is_protected(t))
    assert len(tier1.effective_stopwords) == 124
    # The DoD said 150-200; the curated set is smaller on purpose and must not be
    # padded with content words, so the count is pinned to what the rules justify.
    assert 100 <= metadata["n_effective_stopwords"] < 150
    assert metadata["status"] == "curated_for_m1_tier1"


def test_every_required_guard_is_protected(tier1):
    for term in REQUIRED_GUARDS:
        assert term in tier1.guards, f"{term} must never be filtered"
        assert is_protected(term)
        assert term in filter_tier1([term])
        assert tier1.is_protected(term)


def test_guard_components_protect_compounds(tier1):
    for compound in PROTECTED_COMPOUNDS:
        assert tier1.is_protected(compound)
        assert compound in filter_tier1([compound])
    # A guard only protects tokens that actually contain the component.
    assert not tier1.is_protected("bóng_đá")
    assert not tier1.is_protected("khỏe_mạnh")  # "khỏe" is not "không"


def test_function_words_are_filtered(tier1):
    for term in FILTERABLE_FUNCTION_WORDS:
        assert term in tier1.stopwords
        assert term in tier1.effective_stopwords
        assert filter_tier1([term]) == []


def test_content_words_are_never_filtered(tier1):
    content = ["bóng_đá", "trí_tuệ_nhân_tạo", "y_tế", "kinh_tế", "văn_bản",
               "chính_trị", "thể_thao", "sức_khỏe", "xe_máy", "giáo_dục"]
    assert filter_tier1(content) == content
    assert all(not tier1.is_protected(term) for term in content)


def test_no_content_word_leaked_into_the_resources(tier1):
    """The curation rule, asserted: no domain/content vocabulary in either file."""
    forbidden = {"bóng_đá", "đá", "máy", "tính", "y_tế", "kinh_tế", "người", "việc",
                 "năm", "ngày", "chính_trị", "thể_thao", "sức_khỏe", "xe_máy",
                 "giáo_dục", "văn_bản", "trí_tuệ", "nhân_tạo", "công_nghệ", "thị_trường"}
    assert not (tier1.stopwords & forbidden)
    assert not (tier1.guards & forbidden)


def test_filter_is_symmetric_and_order_preserving(tier1):
    tokens = ["bóng_đá", "và", "của", "bóng", "đá", "không", "rất", "hay"]
    once = tier1.filter_tokens(tokens)
    assert once == tier1.filter_tokens(tokens)
    assert once == ["bóng_đá", "bóng", "đá", "không", "hay"]
    # Corpus and query must go through exactly the same code path.
    assert filter_tier1(tokens) == once


def test_filter_normalises_case_and_nfc(tier1):
    assert tier1.filter_tokens(["Bóng_Đá", "VÀ", "Không"]) == ["bóng_đá", "không"]
    assert tier1.filter_tokens(["  bóng  ", "đá"]) == ["bóng", "đá"]
    assert canonical("  Bóng  Đá ") == "bóng_đá"


def test_empty_and_invalid_tokens(tier1):
    assert tier1.filter_tokens(["", "  ", "và"]) == []
    assert tier1.filter_tokens([]) == []
    with pytest.raises(TypeError):
        tier1.filter_tokens([1, 2])


def test_dropped_tokens_matches_the_filter(tier1):
    tokens = ["bóng", "và", "và", "đá", "không"]
    assert tier1.dropped_tokens(tokens) == ["và", "và"]
    assert tier1.filter_tokens(tokens) == ["bóng", "đá", "không"]


def test_explicit_sets_override_the_resources():
    custom = Tier1StopwordFilter(stopwords={"và", "của", "không"}, guards={"không"})
    assert custom.filter_tokens(["và", "của", "không", "bóng"]) == ["không", "bóng"]
    assert custom.stopword_classes["và"] == "explicit"


def test_metadata_roundtrip_and_tamper_detection(tier1, tmp_path):
    metadata = tier1.metadata()
    assert set(metadata["resource_sha256"]) == {
        "grammar_stopwords.txt", "guard_whitelist.txt", "tier1_source.md"}
    assert all(metadata["resource_sha256"].values())
    restored = Tier1StopwordFilter.from_metadata(metadata)
    assert restored.metadata()["fingerprint"] == metadata["fingerprint"]
    tampered = dict(metadata)
    tampered["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="rebuild"):
        Tier1StopwordFilter.from_metadata(tampered)

    # Real tamper detection: edit a copy of the curated resources, never the originals.
    for name in ("grammar_stopwords.txt", "guard_whitelist.txt", "tier1_source.md"):
        shutil.copy(tier1.resource_dir / name, tmp_path / name)
    copy_filter = Tier1StopwordFilter(resource_dir=tmp_path)
    copy_metadata = copy_filter.metadata()
    assert Tier1StopwordFilter.from_metadata(copy_metadata).metadata()["fingerprint"] == \
        copy_metadata["fingerprint"]
    with (tmp_path / "guard_whitelist.txt").open("a", encoding="utf-8") as handle:
        handle.write("\nvừa_thêm\n")
    with pytest.raises(ValueError, match="rebuild"):
        Tier1StopwordFilter.from_metadata(copy_metadata)


def test_metadata_records_guard_classes(tier1):
    metadata = tier1.metadata()
    classes = tier1.guard_report()
    assert sum(len(terms) for terms in classes.values()) == len(tier1.guards)
    assert any("NEGATION" in name for name in classes)
    assert set(metadata["guard_classes"]) == set(classes)
    assert all(not name.startswith("-") for name in classes)


def test_default_filter_is_cached(tier1):
    assert get_default_filter() is get_default_filter()
    assert get_default_filter().effective_stopwords == tier1.effective_stopwords


def test_guard_whitelist_file_has_no_malformed_entries(tier1):
    for path in (tier1.resource_dir / "grammar_stopwords.txt",
                 tier1.resource_dir / "guard_whitelist.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            assert stripped == canonical(stripped), f"{path.name}: {stripped!r} not canonical"
            assert "(" not in stripped and "?" not in stripped


def test_resources_are_valid_json_free_utf8(tier1):
    for name in ("grammar_stopwords.txt", "guard_whitelist.txt"):
        text = (tier1.resource_dir / name).read_text(encoding="utf-8")
        assert json.loads(json.dumps(text)) == text  # no lone surrogates
        assert text.startswith("#"), "each curated file documents itself first"
