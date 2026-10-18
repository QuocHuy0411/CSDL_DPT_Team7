"""Trie and Maximum Matching (FMM, BMM) behaviour on a declared dictionary.

Every test states the dictionary it uses, so an expectation can be checked by hand
against rule H-like reasoning rather than trusted blindly. These tests prove
deterministic rule behaviour on a fixture dictionary. They do NOT measure
segmentation quality; that needs a human confirmed gold set.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.tokenizer.max_matching import MaximumMatching  # noqa: E402
from engine.tokenizer.preprocessor import syllables  # noqa: E402
from engine.tokenizer.trie import Trie, TrieNode, split_syllables  # noqa: E402

# A fixture dictionary small enough to verify by eye.
FIXTURE = ["học", "sinh", "học sinh", "sinh học", "cơ sở", "dữ liệu",
           "cơ sở dữ liệu", "trí tuệ", "nhân tạo", "trí tuệ nhân tạo",
           "bệnh", "bệnh nhân", "mắc", "người", "người già", "già",
           "làm", "việc", "làm việc", "máu", "mỡ", "kiểm soát"]


@pytest.fixture(scope="module")
def matcher():
    return MaximumMatching(FIXTURE)


# --------------------------------------------------------------------------- #
# trie
# --------------------------------------------------------------------------- #

def test_trie_node_defaults():
    node = TrieNode()
    assert node.children == {}
    assert node.is_end_of_word is False
    assert node.depth == 0


def test_split_syllables_accepts_both_forms():
    assert split_syllables("cơ sở dữ liệu") == ("cơ", "sở", "dữ", "liệu")
    assert split_syllables(["cơ", "sở"]) == ("cơ", "sở")
    assert split_syllables("") == ()


def test_trie_insert_and_search():
    trie = Trie()
    assert trie.insert("cơ sở dữ liệu") is True
    assert trie.search("cơ sở dữ liệu") is True
    assert trie.search("cơ sở") is False          # prefix, not an entry
    assert "cơ sở dữ liệu" in trie
    assert len(trie) == 1


def test_trie_insert_is_idempotent_for_the_same_entry():
    trie = Trie()
    assert trie.insert("máu") is True
    assert trie.insert("máu") is False
    assert len(trie) == 1


def test_trie_rejects_empty_and_blank_entries():
    trie = Trie()
    assert trie.insert("") is False
    assert trie.insert("   ") is False
    assert trie.insert("máu  mắc") is False       # double space inside
    assert len(trie) == 0


def test_trie_tracks_the_longest_entry():
    trie = Trie(FIXTURE)
    assert trie.max_syllables == 4
    assert trie.size == len(FIXTURE)


def test_trie_longest_match_returns_exclusive_end():
    trie = Trie(FIXTURE)
    haystack = ("cơ", "sở", "dữ", "liệu", "x")
    assert trie.longest_match(haystack, 0, 4) == 4
    assert trie.longest_match(haystack, 0, 2) == 2
    assert trie.longest_match(haystack, 0, 1) == 0   # no dictionary entry within one syllable
    assert trie.longest_match(("x", "y"), 0, 4) == 0  # no match returns start


# --------------------------------------------------------------------------- #
# FMM
# --------------------------------------------------------------------------- #

def test_fmm_takes_the_longest_entry_starting_here(matcher):
    assert matcher.tokenize("cơ sở dữ liệu", "fmm") == ["cơ_sở_dữ_liệu"]


def test_fmm_is_left_to_right_and_greedy(matcher):
    # at 0 the longest entry is "học sinh"; the window 4..2 is never reconsidered
    assert matcher.tokenize("học sinh học sinh học", "fmm") == \
        ["học_sinh", "học_sinh", "học"]


def test_fmm_falls_back_to_single_syllables(matcher):
    assert matcher.tokenize("mắc bệnh", "fmm") == ["mắc", "bệnh"]


def test_fmm_emits_oov_syllables_verbatim(matcher):
    assert matcher.tokenize("mắc xyzzy bệnh", "fmm") == ["mắc", "xyzzy", "bệnh"]


# --------------------------------------------------------------------------- #
# BMM
# --------------------------------------------------------------------------- #

def test_bmm_takes_the_longest_entry_ending_here(matcher):
    assert matcher.tokenize("cơ sở dữ liệu", "bmm") == ["cơ_sở_dữ_liệu"]


def test_bmm_is_right_to_left_and_can_differ_from_fmm(matcher):
    # BMM sees "sinh học" before "học sinh" and commits to it
    assert matcher.tokenize("học sinh học sinh học", "bmm") == \
        ["học", "sinh_học", "sinh_học"]
    assert matcher.tokenize("học sinh học sinh học", "fmm") == \
        ["học_sinh", "học_sinh", "học"]


def test_bmm_keeps_token_order_correct_with_mixed_oov(matcher):
    tokens = matcher.tokenize("mắc xyzzy bệnh nhân", "bmm")
    assert tokens == ["mắc", "xyzzy", "bệnh_nhân"]


# --------------------------------------------------------------------------- #
# hard boundaries: punctuation and removed spans
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("separator", [
    ".", ";", "!", "?", ",", ":", "-", "…", "\n", "\t", "123",
    "https://example.org ", "<br>", " a@b.com ", "()", "\"\"",
])
@pytest.mark.parametrize("method", ["fmm", "bmm", "bimm"])
def test_no_fusion_across_any_boundary(separator, method):
    matcher = MaximumMatching(["học", "sinh", "học sinh"])
    assert matcher.tokenize("học sinh", method) == ["học_sinh"]
    assert matcher.tokenize(f"học{separator} sinh", method) == ["học", "sinh"]


def test_removal_creates_a_boundary_not_a_syllable_separator():
    # "học<br> sinh": the tag is removed, and the two sides must not fuse
    matcher = MaximumMatching(["học", "sinh", "học sinh"])
    assert matcher.tokenize("học<br> sinh", "bimm") == ["học", "sinh"]
    assert matcher.tokenize("họchttps://vn.vn  sinh", "bimm") == ["học", "sinh"]


def test_multiple_segments_and_mixed_oov_keep_order(matcher):
    tokens = matcher.tokenize("mắc xyzzy, bệnh nhân; người già", "bmm")
    assert tokens == ["mắc", "xyzzy", "bệnh_nhân", "người_già"]


def test_newline_separates_words_that_would_otherwise_fuse():
    matcher = MaximumMatching(["học", "sinh", "học sinh"])
    assert matcher.tokenize("học\nsinh", "bimm") == ["học", "sinh"]


# --------------------------------------------------------------------------- #
# invariants over random inputs
# --------------------------------------------------------------------------- #

def random_sentence(rng):
    pool = ["học", "sinh", "cơ", "sở", "dữ", "liệu", "mắc", "bệnh", "máu",
            "zzz", "qqq", "trí", "tuệ", "làm", "việc"]
    separators = [", ", ". ", "\n", "; ", " 1 ", " <b> ", " https://x.vn "]
    parts = []
    for _ in range(rng.randint(1, 8)):
        for _ in range(rng.randint(1, 4)):
            parts.append(rng.choice(pool))
        parts.append(rng.choice(separators))
    return " ".join(parts).strip()


@pytest.mark.parametrize("method", ["fmm", "bmm", "bimm"])
def test_tokens_reconstruct_the_input_syllables(matcher, method):
    rng = random.Random(20261016)
    for _ in range(300):
        text = random_sentence(rng)
        tokens = matcher.tokenize(text, method)
        rebuilt = []
        for token in tokens:
            rebuilt.extend(token.split("_"))
        assert rebuilt == syllables(text), (text, tokens)


@pytest.mark.parametrize("method", ["fmm", "bmm", "bimm"])
def test_multi_syllable_tokens_are_always_dictionary_entries(matcher, method):
    rng = random.Random(2026)
    for _ in range(300):
        tokens = matcher.tokenize(random_sentence(rng), method)
        for token in tokens:
            if "_" in token:
                assert matcher.contains(token), token


@pytest.mark.parametrize("method", ["fmm", "bmm", "bimm"])
def test_results_are_deterministic(matcher, method):
    rng = random.Random(99)
    for _ in range(100):
        text = random_sentence(rng)
        assert matcher.tokenize(text, method) == matcher.tokenize(text, method)


# --------------------------------------------------------------------------- #
# window bound and argument handling
# --------------------------------------------------------------------------- #

def test_max_syllables_override_limits_the_window():
    matcher = MaximumMatching(FIXTURE, max_syllables=2)
    assert matcher.tokenize("cơ sở dữ liệu", "fmm") == ["cơ_sở", "dữ_liệu"]


def test_max_syllables_defaults_to_the_longest_entry():
    assert MaximumMatching(FIXTURE).max_syllables == 4


def test_max_syllables_must_be_positive():
    with pytest.raises(ValueError):
        MaximumMatching(FIXTURE, max_syllables=0)


def test_unknown_method_is_rejected(matcher):
    with pytest.raises(ValueError):
        matcher.tokenize("máu", "nope")
    with pytest.raises(ValueError):
        matcher.trace("máu", "nope")


def test_empty_and_non_string_input(matcher):
    assert matcher.tokenize("") == []
    assert matcher.tokenize(None) == []
    assert matcher.tokenize("   ,,.  ") == []
    assert matcher.trace("") == []


def test_missing_dictionary_file_raises():
    with pytest.raises(FileNotFoundError):
        MaximumMatching(Path("does-not-exist-dict.txt"))
