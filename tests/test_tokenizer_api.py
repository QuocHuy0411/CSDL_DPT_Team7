"""Public API surface of engine.tokenizer: entry points, trace schema, provenance.

These tests pin the contract other builders and the lead's index build rely on.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import engine.tokenizer as tokenizer_package  # noqa: E402
from engine.tokenizer import max_matching  # noqa: E402
from engine.tokenizer.max_matching import (  # noqa: E402
    ACTIONS,
    MaximumMatching,
    default_matcher,
    tokenize,
    tokenize_bimm,
    tokenize_bmm,
    tokenize_fmm,
    trace,
)

FIXTURE = ["học", "sinh", "học sinh", "sinh học", "cơ sở dữ liệu",
           "bệnh", "bệnh nhân", "mắc", "người", "người già", "già",
           "của", "và", "là", "không", "chưa", "chẳng", "vì", "do đó"]


# --------------------------------------------------------------------------- #
# module surface
# --------------------------------------------------------------------------- #

def test_required_symbols_are_exported():
    for name in ("normalize", "normalize_segments", "syllables", "Trie", "TrieNode",
                 "MaximumMatching", "tokenize", "tokenize_fmm", "tokenize_bmm",
                 "tokenize_bimm", "trace"):
        assert hasattr(tokenizer_package, name), name


def test_preprocessing_spec_matches_lead_contract():
    assert tokenizer_package.PREPROCESSING_SPEC == \
        "engine.tokenizer.max_matching:tokenize"
    assert tokenizer_package.TIER1_SPEC == \
        "engine.tokenizer.stopwords:filter_tier1"


def test_spec_is_resolvable_by_the_shared_resolver():
    from engine.preprocessing import resolve_callable
    resolved = resolve_callable(tokenizer_package.PREPROCESSING_SPEC)
    assert resolved is tokenize


def test_module_level_functions_use_the_shared_default_instance():
    assert default_matcher() is default_matcher()
    text = "bệnh nhân không mắc bệnh"
    assert tokenize(text) == tokenize_bimm(text) == default_matcher().tokenize(text)
    assert tokenize_fmm(text) == default_matcher().tokenize(text, "fmm")
    assert tokenize_bmm(text) == default_matcher().tokenize(text, "bmm")


def test_default_method_is_bimm():
    text = "học sinh học sinh học"
    assert tokenize(text) == tokenize_bimm(text)


def test_method_keyword_is_accepted():
    assert tokenize("máu", method="fmm") == tokenize_fmm("máu")
    assert tokenize("máu", method="bmm") == tokenize_bmm("máu")


def test_tokens_are_underscore_joined():
    tokens = tokenize("cơ sở dữ liệu")
    assert tokens == ["cơ_sở_dữ_liệu"]
    assert all(" " not in token and token.strip("_") == token for token in tokens)


def test_no_tier1_filtering_is_applied_by_the_core():
    # function words and guard words survive the core tokenizer untouched
    tokens = MaximumMatching(FIXTURE).tokenize("và của là không chưa chẳng vì do đó", "bimm")
    assert tokens == ["và", "của", "là", "không", "chưa", "chẳng", "vì", "do_đó"]


def test_guards_are_not_swallowed_into_neighbours():
    # the core tokenizer keeps guards as their own tokens; isolating them from
    # compounds is the guard-list owner's policy, not this module's
    matcher = MaximumMatching(FIXTURE)
    assert "không" in matcher.tokenize("không mắc bệnh", "bimm")


def test_importing_the_package_does_not_pull_in_stopwords():
    # stopwords is owned by builder-retrieval and must stay an optional import
    code = (
        "import sys; import engine.tokenizer; "
        "print('engine.tokenizer.stopwords' in sys.modules)"
    )
    output = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd=str(ROOT), check=True)
    assert output.stdout.strip() == "False", output.stdout


# --------------------------------------------------------------------------- #
# trace schema
# --------------------------------------------------------------------------- #

REQUIRED_KEYS = {"step", "direction", "start", "end", "candidate", "size",
                 "matched", "action"}


@pytest.mark.parametrize("method", ["fmm", "bmm", "bimm"])
def test_trace_rows_carry_the_contract_schema(method):
    rows = MaximumMatching(FIXTURE).trace("học sinh học sinh học", method)
    assert rows
    for row in rows:
        assert REQUIRED_KEYS <= set(row), row
        assert row["action"] in ACTIONS, row
        assert row["direction"] in {"fmm", "bmm"}, row
        assert isinstance(row["matched"], bool)
        assert isinstance(row["size"], int) and row["size"] >= 1
        assert row["end"] > row["start"], row
        assert row["end"] - row["start"] == row["size"], row
        assert row["candidate"] == " ".join(row["candidate"].split())


@pytest.mark.parametrize("method", ["fmm", "bmm", "bimm"])
def test_trace_step_numbers_are_strictly_increasing_in_time(method):
    rows = MaximumMatching(FIXTURE).trace("học sinh học sinh học", method)
    assert [row["step"] for row in rows] == list(range(len(rows)))


def test_trace_indices_are_normalized_syllable_indices_end_exclusive():
    rows = MaximumMatching(FIXTURE).trace("cơ sở dữ liệu", "fmm")
    first = rows[0]
    assert (first["start"], first["end"]) == (0, 4)
    assert first["candidate"] == "cơ sở dữ liệu"
    assert first["action"] == "MATCHED"
    assert first["matched"] is True


def test_trace_records_shrink_then_single_for_unknown_windows():
    rows = MaximumMatching(FIXTURE).trace("xyzzy", "fmm")
    assert [row["action"] for row in rows] == ["SINGLE"]
    assert rows[0]["matched"] is False
    assert rows[0]["size"] == 1


def test_trace_reports_shrink_before_a_shorter_match():
    rows = MaximumMatching(FIXTURE).trace("học sinh học sinh học", "fmm")
    actions = [row["action"] for row in rows]
    assert actions[0] in {"SHRINK", "MATCHED"}
    assert "SHRINK" in actions


def test_bmm_trace_keeps_right_to_left_chronology():
    rows = MaximumMatching(FIXTURE).trace("xyzzy bệnh nhân", "bmm")
    assert rows
    # Cursor walks left; shrink attempts at the same cursor have decreasing size.
    assert [r["end"] for r in rows] == sorted([r["end"] for r in rows], reverse=True)
    assert rows[0]["action"] == "SHRINK" and rows[1]["action"] == "MATCHED"
    assert rows[0]["size"] > rows[1]["size"]


def test_bimm_trace_marks_the_selected_pass():
    rows = MaximumMatching(FIXTURE).trace("học sinh học sinh học", "bimm")
    directions = {row["direction"] for row in rows}
    assert directions == {"fmm", "bmm"}
    selected = {row["direction"] for row in rows if row["selected"]}
    assert len(selected) == 1
    chosen = selected.pop()
    expected = MaximumMatching(FIXTURE).tokenize("học sinh học sinh học", chosen)
    tokens_from_trace = _tokens_from_trace([row for row in rows if row["selected"]])
    assert tokens_from_trace == expected


def _tokens_from_trace(rows):
    terminals = sorted((r for r in rows if r["action"] in {"MATCHED", "SINGLE"}),
                       key=lambda r: r["start"])
    return [r["candidate"].replace(" ", "_") for r in terminals]


@pytest.mark.parametrize("method", ["fmm", "bmm", "bimm"])
def test_trace_reconstructs_the_tokenize_result(method):
    text = "bệnh nhân, không mắc bệnh của người già"
    matcher = MaximumMatching(FIXTURE)
    tokens = matcher.tokenize(text, method)
    rows = [row for row in matcher.trace(text, method) if row["selected"]]
    assert _tokens_from_trace(rows) == tokens


def test_trace_on_empty_input():
    assert trace("") == []
    assert trace(None) == []


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

def test_metadata_records_dictionary_provenance():
    metadata = MaximumMatching(FIXTURE).metadata()
    assert metadata["tokenizer"] == "engine.tokenizer.max_matching"
    assert metadata["method"] == "bimm"
    assert metadata["tier1_filter"] is None
    assert metadata["dictionary_entries"] == len(FIXTURE)
    assert metadata["bimm_order"] == [
        "fewer_tokens", "fewer_single_tokens", "fewer_oov", "tie_goes_to_bmm"]
    assert metadata["punctuation_is_boundary"] is True


def test_default_dictionary_provenance_is_populated():
    metadata = MaximumMatching().metadata()
    assert metadata["dictionary_path"].endswith("resources/dict.txt")
    assert len(metadata["dictionary_sha256"]) == 64
    assert metadata["dictionary_entries"] > 65000
    assert metadata["max_syllables"] >= 4


def test_default_dictionary_is_bundled_and_offline():
    assert max_matching.DEFAULT_DICTIONARY.is_file()
    assert max_matching.DEFAULT_DICTIONARY.parent.name == "resources"


def test_dictionary_entries_are_lowercase_nfc_vietnamese():
    import unicodedata
    from engine.tokenizer.preprocessor import VIETNAMESE_LETTERS_SET
    sample = max_matching.DEFAULT_DICTIONARY.read_text(encoding="utf-8").splitlines()
    assert len(sample) > 65000
    for entry in sample[:2000] + sample[-2000:]:
        assert entry == entry.strip().lower()
        assert entry == unicodedata.normalize("NFC", entry)
        for character in entry:
            assert character == " " or character in VIETNAMESE_LETTERS_SET, entry


def test_real_corpus_style_text_tokenizes_without_error():
    sentence = "Bệnh nhân không mắc bệnh tiểu đường nhưng vẫn phải điều trị cao huyết áp."
    tokens = tokenize(sentence)
    assert tokens
    assert "bệnh_nhân" in tokens
    assert "không" in tokens
    from engine.tokenizer.preprocessor import syllables
    assert [s for token in tokens for s in token.split("_")] == syllables(sentence)
    # Controlled fixture tests phrase joining, not an invented default lexicon entry.
    assert MaximumMatching(["cao huyết áp"]).tokenize("cao huyết áp") == ["cao_huyết_áp"]
