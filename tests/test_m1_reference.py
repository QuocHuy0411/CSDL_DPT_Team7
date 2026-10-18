"""Independent direct-set oracle: word order, hard boundaries and BiMM arbitration."""

import itertools
import random

import pytest


def direct(segment, vocabulary, reverse):
    syllables = segment.split()
    words = []
    while syllables:
        match = None
        for size in range(len(syllables), 0, -1):
            candidate = syllables[-size:] if reverse else syllables[:size]
            if " ".join(candidate) in vocabulary:
                match = candidate
                break
        match = match or ([syllables[-1]] if reverse else [syllables[0]])
        words.append("_".join(match))
        syllables = syllables[:-len(match)] if reverse else syllables[len(match):]
    return list(reversed(words)) if reverse else words


def arbitration(tokens, vocabulary):
    return (len(tokens), sum("_" not in t for t in tokens),
            sum(t.replace("_", " ") not in vocabulary for t in tokens))


def test_random_mm_against_independent_set_oracle():
    from engine.tokenizer.max_matching import MaximumMatching
    rng = random.Random(20261016)
    universe = [" ".join(t) for n in range(1, 5)
                for t in itertools.product("abcd", repeat=n)]
    for _ in range(120):
        vocab = set(rng.sample(universe, rng.randint(0, 60)))
        matcher = MaximumMatching(vocab)
        segments = [" ".join(rng.choices("abcd", k=rng.randint(1, 12)))
                    for _ in range(rng.randint(1, 4))]
        text = " ; ".join(segments)
        forward = [t for s in segments for t in direct(s, vocab, False)]
        backward = [t for s in segments for t in direct(s, vocab, True)]
        chosen = backward if arbitration(backward, vocab) <= arbitration(forward, vocab) else forward
        assert matcher.tokenize(text, "fmm") == forward
        assert matcher.tokenize(text, "bmm") == backward
        assert matcher.tokenize(text, "bimm") == chosen
        trace = matcher.trace(text, "bimm")
        assert [row["step"] for row in trace] == list(range(len(trace)))
        for row in trace:
            assert row["end"] - row["start"] == row["size"]
            assert row["candidate"] in vocab if row["matched"] else row["candidate"] not in vocab
        # Winning terminal actions must reproduce the exact emitted word order.
        terminal = [r for r in trace if r["selected"] and r["action"] in {"MATCHED", "SINGLE"}]
        terminal.sort(key=lambda r: r["start"])
        assert [r["candidate"].replace(" ", "_") for r in terminal] == chosen


@pytest.mark.parametrize("method", ["fmm", "bmm", "bimm"])
def test_no_fusion_across_punctuation_or_removed_nontext(method):
    from engine.tokenizer.max_matching import MaximumMatching
    matcher = MaximumMatching(["học sinh", "cơ sở dữ liệu", "học", "sinh"])
    assert matcher.tokenize("học sinh", method) == ["học_sinh"]
    for separator in (".", ";", "!", "?", ",", "\n", "123", "https://example.org ", "<br>"):
        assert matcher.tokenize(f"học{separator} sinh", method) == ["học", "sinh"]


def test_m1_adapter_protects_compounds_and_tracks_full_resource_graph():
    from engine.preprocessing import M1_RESOURCES, M1_TIER1, M1_TOKENIZER, Preprocessor
    processor = Preprocessor(M1_TOKENIZER, M1_TIER1, True, True)
    tokens = processor("Bệnh nhân không thể và chưa mắc bệnh của người già.")
    assert any("không" in t.split("_") for t in tokens)
    assert any("chưa" in t.split("_") for t in tokens)
    assert "của" not in tokens
    metadata = processor.metadata()
    assert metadata["status"] == "official"
    assert set(M1_RESOURCES) <= set(metadata["resource_hashes"])
    assert Preprocessor.from_metadata(metadata)("bệnh nhân không thể") == processor("bệnh nhân không thể")
