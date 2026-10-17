"""BiMM decision rule, tie breaks and parity with an independent set oracle.

The rule under test is the one in the contract: prefer fewer tokens, then fewer
single syllable tokens, then fewer out-of-vocabulary tokens, and on an exact tie
prefer BMM. The oracle below is written a second time, deliberately differently
from the production code: it uses plain Python set membership and string slicing
instead of a trie walk, so agreement between the two is real evidence.
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

FIXTURE = ["học", "sinh", "học sinh", "sinh học", "cơ sở", "dữ liệu",
           "cơ sở dữ liệu", "trí tuệ", "nhân tạo", "trí tuệ nhân tạo",
           "bệnh", "bệnh nhân", "mắc", "người", "người già", "già",
           "làm", "việc", "làm việc", "máu", "mỡ", "kiểm soát",
           "kiểm", "soát"]


@pytest.fixture(scope="module")
def matcher():
    return MaximumMatching(FIXTURE)


# --------------------------------------------------------------------------- #
# independent oracle: set based, no trie
# --------------------------------------------------------------------------- #

def segments_of(text):
    from engine.tokenizer.preprocessor import normalize_segments
    return [segment.split(" ") for segment in normalize_segments(text)]


def oracle_fmm(segments, vocabulary, window):
    tokens = []
    for syllables in segments:
        index = 0
        while index < len(syllables):
            limit = min(window, len(syllables) - index)
            found = None
            for size in range(limit, 0, -1):
                candidate = tuple(syllables[index:index + size])
                if candidate in vocabulary:
                    found = size
                    break
            if found is None:
                tokens.append(syllables[index])
                index += 1
            else:
                tokens.append("_".join(syllables[index:index + found]))
                index += found
    return tokens


def oracle_bmm(segments, vocabulary, window):
    tokens = []
    for syllables in segments:
        collected = []
        end = len(syllables)
        while end > 0:
            limit = min(window, end)
            found = None
            for size in range(limit, 0, -1):
                start = end - size
                if tuple(syllables[start:end]) in vocabulary:
                    found = start
                    break
            if found is None:
                collected.append(syllables[end - 1])
                end -= 1
            else:
                collected.append("_".join(syllables[found:end]))
                end = found
        collected.reverse()
        tokens.extend(collected)
    return tokens


def key_of(tokens, vocabulary):
    singles = sum(1 for token in tokens if "_" not in token)
    oov = sum(1 for token in tokens
              if tuple(token.split("_")) not in vocabulary)
    return (len(tokens), singles, oov)


def oracle_bimm(segments, vocabulary, window):
    forward = oracle_fmm(segments, vocabulary, window)
    backward = oracle_bmm(segments, vocabulary, window)
    # a tie goes to BMM, so <= is the correct comparison
    return backward if key_of(backward, vocabulary) <= key_of(forward, vocabulary) else forward


# --------------------------------------------------------------------------- #
# the rule itself
# --------------------------------------------------------------------------- #

def test_bimm_prefers_fewer_tokens(matcher):
    # FMM commits to "học sinh" then repeats it; BMM takes three compound-friendly
    # pieces; both produce 3 tokens here, so this case only pins the ordering rule
    text = "học sinh học sinh học"
    assert matcher.tokenize(text, "bimm") in (
        matcher.tokenize(text, "fmm"), matcher.tokenize(text, "bmm"))


def test_bimm_never_worse_than_either_pass_on_token_count(matcher):
    rng = random.Random(20261016)
    for _ in range(400):
        text = random_text(rng)
        chosen = matcher.tokenize(text, "bimm")
        forward = matcher.tokenize(text, "fmm")
        backward = matcher.tokenize(text, "bmm")
        assert len(chosen) == min(len(forward), len(backward)), text


def test_bimm_respects_the_full_ladder(matcher):
    rng = random.Random(4242)
    checked = 0
    for _ in range(400):
        text = random_text(rng)
        forward = matcher.tokenize(text, "fmm")
        backward = matcher.tokenize(text, "bmm")
        chosen = matcher.tokenize(text, "bimm")
        key_fmm = matcher.decision_key(forward)
        key_bmm = matcher.decision_key(backward)
        expected = backward if key_bmm <= key_fmm else forward
        assert chosen == expected, text
        checked += 1
    assert checked == 400


def test_tie_goes_to_bmm():
    # "học sinh học sinh học": both passes emit 3 tokens, 1 single and 1 OOV-free set
    matcher = MaximumMatching(FIXTURE)
    text = "học sinh học sinh học"
    forward = matcher.tokenize(text, "fmm")
    backward = matcher.tokenize(text, "bmm")
    assert matcher.decision_key(forward) == matcher.decision_key(backward)
    assert matcher.tokenize(text, "bimm") == backward == ["học", "sinh_học", "sinh_học"]


def test_decision_key_components_are_distinct_and_ordered(matcher):
    tokens = matcher.tokenize("học sinh học sinh học", "fmm")
    length, singles, oov = matcher.decision_key(tokens)
    assert length == 3
    assert singles == 1
    assert oov == 0


def test_oov_is_counted_as_such(matcher):
    tokens = matcher.tokenize("mắc xyzzy bệnh", "fmm")
    length, singles, oov = matcher.decision_key(tokens)
    assert length == 3
    assert singles == 3      # mắc, xyzzy, bệnh are all single-syllable tokens
    assert oov == 1          # xyzzy only


# --------------------------------------------------------------------------- #
# parity with the independent oracle
# --------------------------------------------------------------------------- #

def random_text(rng):
    pool = ["học", "sinh", "cơ", "sở", "dữ", "liệu", "mắc", "bệnh", "máu",
            "zzz", "qqq", "trí", "tuệ", "làm", "việc", "kiểm", "soát"]
    separators = [", ", ". ", "\n", "; ", " 1 ", " <b> ", " https://x.vn "]
    parts = []
    for _ in range(rng.randint(1, 7)):
        for _ in range(rng.randint(1, 4)):
            parts.append(rng.choice(pool))
        parts.append(rng.choice(separators))
    return " ".join(parts).strip()


@pytest.mark.parametrize("method", ["fmm", "bmm", "bimm"])
def test_parity_with_independent_set_oracle(method):
    vocabulary = {tuple(entry.split(" ")) for entry in FIXTURE}
    matcher = MaximumMatching(FIXTURE)
    oracle = {"fmm": oracle_fmm, "bmm": oracle_bmm, "bimm": oracle_bimm}[method]
    rng = random.Random(20261016)
    for _ in range(400):
        text = random_text(rng)
        assert matcher.tokenize(text, method) == \
            oracle(segments_of(text), vocabulary, matcher.max_syllables), text


def test_oracle_agrees_with_both_passes_on_the_contract_examples():
    vocabulary = {tuple(entry.split(" ")) for entry in FIXTURE}
    matcher = MaximumMatching(FIXTURE)
    samples = [
        "học sinh học sinh học",
        "cơ sở dữ liệu",
        "trí tuệ nhân tạo",
        "bệnh nhân không mắc bệnh của người già",
        "mắc xyzzy, bệnh nhân; người già",
        "máu, mỡ; kiểm soát",
    ]
    for text in samples:
        for method in ("fmm", "bmm", "bimm"):
            oracle = {"fmm": oracle_fmm, "bmm": oracle_bmm, "bimm": oracle_bimm}[method]
            assert matcher.tokenize(text, method) == \
                oracle(segments_of(text), vocabulary, matcher.max_syllables), (text, method)


def test_bimm_reconstructs_input_syllables():
    matcher = MaximumMatching(FIXTURE)
    rng = random.Random(7)
    for _ in range(300):
        text = random_text(rng)
        tokens = matcher.tokenize(text, "bimm")
        rebuilt = [part for token in tokens for part in token.split("_")]
        assert rebuilt == syllables(text), text
