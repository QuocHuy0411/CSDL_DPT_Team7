"""Unit tests for the M1 normalizer. Deterministic fixtures only.

These fixtures check that the normalizer obeys its stated policy. They say
nothing about segmentation quality: that is measured against a human confirmed
gold set, which builder-retrieval owns.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.tokenizer.preprocessor import (  # noqa: E402
    VIETNAMESE_LETTERS_SET,
    normalize,
    normalize_segments,
    policy,
    syllables,
)


# --------------------------------------------------------------------------- #
# unicode
# --------------------------------------------------------------------------- #

def test_nfd_input_becomes_nfc():
    decomposed = unicodedata.normalize("NFD", "Tiếng Việt")
    assert decomposed != unicodedata.normalize("NFC", "Tiếng Việt")
    assert normalize(decomposed) == "tiếng việt"


def test_output_is_nfc():
    for text in ("tiếng việt", "đường sắt", "chuyển giao"):
        assert normalize(text) == unicodedata.normalize("NFC", normalize(text))


def test_bom_is_stripped():
    assert normalize("﻿tiếng việt") == "tiếng việt"


def test_case_is_folded_to_lower():
    assert normalize("BỆNH NHÂN") == "bệnh nhân"


# --------------------------------------------------------------------------- #
# noise removal
# --------------------------------------------------------------------------- #

def test_urls_are_removed_whole():
    # the whole URL including its path is removed; "tin" is part of /tin, not a word
    assert normalize("xem https://vnexpress.net/tin abc") == "xem abc"
    assert normalize("truy cập www.vn.vn ngay") == "truy cập ngay"


def test_email_glued_to_a_vietnamese_word_is_not_eaten():
    # the address is only recognised at a boundary, so "họca@b.com" keeps its letters
    assert normalize("gửi cho a@b.com nhé") == "gửi cho nhé"
    assert normalize_segments("họca@b.com") == ["họca", "b", "com"]


def test_html_tags_are_removed():
    assert normalize("<b>in đậm</b> và <i>nghiêng</i>") == "in đậm và nghiêng"


def test_nested_html_tags_are_removed():
    assert normalize("<p><b><i>đậm nghiêng</i></b></p>") == "đậm nghiêng"


def test_emoji_are_removed():
    assert normalize("vui quá 😀😂 nhé") == "vui quá nhé"


def test_digits_are_removed_as_boundaries():
    assert normalize("câu 1 tiếp 2") == "câu tiếp"


def test_non_string_input_returns_empty():
    assert normalize(None) == ""
    assert normalize(123) == ""
    assert normalize_segments(None) == []


# --------------------------------------------------------------------------- #
# punctuation is a boundary, whitespace is not
# --------------------------------------------------------------------------- #

def test_punctuation_splits_segments():
    assert normalize_segments("tôi, đi!") == ["tôi", "đi"]


def test_plain_space_does_not_split_a_segment():
    assert normalize_segments("cơ sở dữ liệu") == ["cơ sở dữ liệu"]


def test_newline_is_a_boundary():
    assert normalize_segments("một\ntư") == ["một", "tư"]


def test_tab_is_a_boundary():
    assert normalize_segments("một\ttư") == ["một", "tư"]


def test_complex_punctuation_sequence():
    assert normalize_segments("a--b;;c??d!!e") == ["a", "b", "c", "d", "e"]


def test_punctuation_is_never_hidden_by_the_display_form():
    text = "tôi, đi!"
    assert normalize(text) == "tôi đi"
    assert normalize(text).split(" ") == ["tôi", "đi"]
    # the display form collapses the boundary, the segments do not
    assert normalize_segments(text) == ["tôi", "đi"]


def test_display_form_and_segments_agree_on_letters():
    samples = [
        "Bệnh nhân không mắc bệnh, người già!",
        "Cổ phiếu tăng? <b>Mạnh</b> lắm... 100%",
        "một\t hai\n ba",
        "abc",
        "!?.,;:",
    ]
    for text in samples:
        assert " ".join(normalize_segments(text)) == normalize(text)


def test_whitespace_only_input():
    assert normalize("   \n\t ") == ""
    assert normalize_segments("   \n\t ") == []


def test_empty_input():
    assert normalize("") == ""
    assert normalize_segments("") == []


# --------------------------------------------------------------------------- #
# vocabulary of the normalizer itself
# --------------------------------------------------------------------------- #

def test_letters_cover_the_vietnamese_alphabet():
    for letter in "aăâbcdđeêghiklmnoôơpqrstuưvxy":
        assert letter in VIETNAMESE_LETTERS_SET
    for letter in "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụưứừửữựýỳỷỹỵ":
        assert letter in VIETNAMESE_LETTERS_SET


def test_ascii_and_loanword_names_are_not_mangled():
    assert normalize("abc Windows Fed Java") == "abc windows fed java"


def test_syllables_helper_flattens_segments():
    assert syllables("tôi, đi! làm việc") == ["tôi", "đi", "làm", "việc"]


def test_policy_is_self_describing():
    described = policy()
    assert described["punctuation_is_a_boundary"] is True
    assert described["whitespace_is_a_boundary"] is False
    assert described["alphabet_size"] == len(VIETNAMESE_LETTERS_SET)


@pytest.mark.parametrize("text", [
    "Hà Nội", "Đà Nẵng", "Bình Dương", "Hồ Chí Minh", "Quảng Ninh",
    "Nghệ An", "Thừa Thiên Huế", "Bắc Ninh", "Phú Thọ",
])
def test_real_place_names_survive_normalization(text):
    assert normalize(text) == text.lower()
    assert normalize_segments(text) == [text.lower()]
