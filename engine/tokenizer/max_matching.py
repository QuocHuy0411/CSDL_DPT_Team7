"""Maximum Matching word segmentation for Vietnamese (FMM, BMM, BiMM).

Contract surface
----------------
    MaximumMatching(words=None, max_syllables=None)
    MaximumMatching.tokenize(text, method="bimm") -> list[str]
    tokenize(text, method="bimm") -> list[str]        (module level, shared instance)
    tokenize_fmm / tokenize_bmm / tokenize_bimm(text) -> list[str]
    trace(text, method="bimm") -> list[dict]

`tokenize` returns underscore joined words and applies **no** tier 1 stopword
filtering. Filtering is a separate owner (`engine.tokenizer.stopwords`).

Boundaries
----------
Punctuation is a hard boundary. The text is first split into segments by
`preprocessor.normalize_segments`; a sliding window never crosses a segment
boundary, so no word is ever fused across punctuation even though the normalized
display form looks like one continuous space separated string.

BiMM decision rule, applied in this order
-----------------------------------------
  1. fewer tokens
  2. fewer single syllable tokens
  3. fewer out-of-vocabulary tokens
  4. tie -> BMM

With a pure trie implementation criterion 3 is implied by criterion 2 (every OOV
token is emitted as a single syllable), so it is kept as an explicit safety net
rather than dropped, and `decision_key` is exposed so a caller can see the full
comparison instead of trusting the summary.

trace() rows
------------
Each row carries `step`, `direction` ("fmm"/"bmm"), `start`, `end` (normalized
syllable indices, end exclusive), `candidate`, `size`, `matched` and `action`
("MATCHED" / "SHRINK" / "SINGLE"). Two extra fields are added: `segment`, the
index of the text segment the window belongs to, and `selected`, which marks the
rows of the winning pass when BiMM emits both passes.
"""

from __future__ import annotations

from bisect import bisect_left
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

from engine.tokenizer.preprocessor import normalize_segments
from engine.tokenizer.trie import Trie

__all__ = [
    "MaximumMatching",
    "DEFAULT_DICTIONARY",
    "DICTIONARY_PATH",
    "tokenize",
    "tokenize_fmm",
    "tokenize_bmm",
    "tokenize_bimm",
    "trace",
    "default_matcher",
    "dictionary_metadata",
    "ACTIONS",
]

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DICTIONARY = ROOT / "resources/dict.txt"
DICTIONARY_PATH = DEFAULT_DICTIONARY

ACTIONS = ("MATCHED", "SHRINK", "SINGLE")
METHODS = ("fmm", "bmm", "bimm")
JOINER = "_"


def _load_words(source: str | Path | Iterable[str] | None) -> list[str]:
    if source is None:
        source = DEFAULT_DICTIONARY
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"dictionary not found: {path}")
        return [line.strip() for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [str(word).strip() for word in source if str(word).strip()]


class MaximumMatching:
    """Trie backed Maximum Matching segmenter.

    Parameters
    ----------
    words:
        Path to a dictionary file, or an iterable of entries. `None` loads the
        bundled `resources/dict.txt`.
    max_syllables:
        Upper bound for the sliding window. `None` uses the longest entry in the
        dictionary, which is the smallest correct bound.
    """

    def __init__(self, words: str | Path | Iterable[str] | None = None,
                 max_syllables: int | None = None) -> None:
        if words is None:
            # provenance must be recorded even for the bundled default
            self.dictionary_path: Path | None = DEFAULT_DICTIONARY
            entries = _load_words(None)
        else:
            self.dictionary_path = Path(words) if isinstance(words, (str, Path)) else None
            entries = _load_words(words)
        self.trie = Trie(entries)
        self.words_count = len(entries)
        if max_syllables is None:
            self.max_syllables = self.trie.max_syllables or 1
        else:
            if max_syllables < 1:
                raise ValueError("max_syllables must be >= 1")
            self.max_syllables = int(max_syllables)
        self._sha256: str | None = None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def tokenize(self, text: str, method: str = "bimm") -> list[str]:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if not isinstance(text, str) or not text:
            return []
        if method == "fmm":
            return self._tokens(self._run_fmm(self._layout(text)))
        if method == "bmm":
            return self._tokens(self._run_bmm(self._layout(text)))
        return self._tokens(self._run_bimm(self._layout(text)))

    def tokenize_fmm(self, text: str) -> list[str]:
        return self.tokenize(text, "fmm")

    def tokenize_bmm(self, text: str) -> list[str]:
        return self.tokenize(text, "bmm")

    def tokenize_bimm(self, text: str) -> list[str]:
        return self.tokenize(text, "bimm")

    def trace(self, text: str, method: str = "bimm") -> list[dict]:
        """Step by step record of the sliding window. See module docstring."""
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if not isinstance(text, str) or not text:
            return []
        layout = self._layout(text)
        if method == "fmm":
            rows, _ = self._fmm(layout, record=True)
            return rows
        if method == "bmm":
            rows, _ = self._bmm(layout, record=True)
            return rows
        chosen, rows = self._bimm(layout, record=True)
        for row in rows:
            row["selected"] = row["direction"] == chosen
        return rows

    def decision_key(self, tokens: Sequence[str]) -> tuple[int, int, int]:
        """The ordered BiMM comparison key for one candidate segmentation."""
        singles = sum(1 for token in tokens if JOINER not in token)
        oov = sum(1 for token in tokens if not self.contains(token))
        return (len(tokens), singles, oov)

    def contains(self, token: str) -> bool:
        return self.trie.search(token.replace(JOINER, " "))

    @property
    def dictionary_sha256(self) -> str | None:
        if self._sha256 is None and self.dictionary_path is not None:
            if self.dictionary_path.is_file():
                self._sha256 = hashlib.sha256(
                    self.dictionary_path.read_bytes()).hexdigest()
        return self._sha256

    def metadata(self) -> dict:
        """Provenance a retrieval index can record next to its own metadata."""
        return {
            "tokenizer": "engine.tokenizer.max_matching",
            "method": "bimm",
            "dictionary_path": (self.dictionary_path.as_posix()
                                if self.dictionary_path else None),
            "dictionary_sha256": self.dictionary_sha256,
            "dictionary_entries": self.words_count,
            "max_syllables": self.max_syllables,
            "tier1_filter": None,
            "bimm_order": ["fewer_tokens", "fewer_single_tokens", "fewer_oov", "tie_goes_to_bmm"],
            "punctuation_is_boundary": True,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"MaximumMatching(entries={self.words_count}, "
                f"max_syllables={self.max_syllables})")

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _layout(self, text: str) -> tuple[list[str], list[int]]:
        """Flatten segments into one syllable list plus each segment's end index."""
        syllables: list[str] = []
        segment_ends: list[int] = []
        for segment in normalize_segments(text):
            syllables.extend(segment.split(" "))
            segment_ends.append(len(syllables))
        return syllables, segment_ends

    def _window_limit(self, start: int, segment_end: int, total: int) -> int:
        room = min(self.max_syllables, segment_end - start, total - start)
        return max(room, 1)

    def _fmm(self, layout, record: bool = False) -> tuple[list[dict], list[str]]:
        syllables, segment_ends = layout
        total = len(syllables)
        rows: list[dict] = []
        tokens: list[str] = []
        start = 0
        while start < total:
            segment_index = bisect_left(segment_ends, start + 1)
            segment_end = segment_ends[segment_index]
            limit = self._window_limit(start, segment_end, total)
            matched_end = None
            for size in range(limit, 0, -1):
                end = start + size
                candidate = " ".join(syllables[start:end])
                hit = self.trie.search(candidate)
                if record:
                    rows.append({
                        "step": len(rows),
                        "direction": "fmm",
                        "start": start,
                        "end": end,
                        "candidate": candidate,
                        "size": size,
                        "matched": hit,
                        "action": "MATCHED" if hit else ("SINGLE" if size == 1 else "SHRINK"),
                        "segment": segment_index,
                        "selected": True,
                    })
                if hit:
                    matched_end = end
                    break
            if matched_end is None:
                tokens.append(syllables[start])
                start += 1
            else:
                tokens.append(JOINER.join(syllables[start:matched_end]))
                start = matched_end
        return rows, tokens

    def _bmm(self, layout, record: bool = False) -> tuple[list[dict], list[str]]:
        syllables, segment_ends = layout
        total = len(syllables)
        rows: list[dict] = []
        backwards: list[str] = []
        # end_cursor walks right to left; start/end stay left-to-right indices
        end_cursor = total
        while end_cursor > 0:
            segment_index = bisect_left(segment_ends, end_cursor)
            segment_start = segment_ends[segment_index - 1] if segment_index else 0
            limit = self._window_limit(segment_start, end_cursor, total)
            matched_start = None
            for size in range(limit, 0, -1):
                start = end_cursor - size
                if start < segment_start:
                    break
                candidate = " ".join(syllables[start:end_cursor])
                hit = self.trie.search(candidate)
                if record:
                    rows.append({
                        "step": len(rows),
                        "direction": "bmm",
                        "start": start,
                        "end": end_cursor,
                        "candidate": candidate,
                        "size": size,
                        "matched": hit,
                        "action": "MATCHED" if hit else ("SINGLE" if size == 1 else "SHRINK"),
                        "segment": segment_index,
                        "selected": True,
                    })
                if hit:
                    matched_start = start
                    break
            if matched_start is None:
                backwards.append(syllables[end_cursor - 1])
                end_cursor -= 1
            else:
                backwards.append(JOINER.join(syllables[matched_start:end_cursor]))
                end_cursor = matched_start
        # one single reversal; trace rows keep the real right-to-left chronology
        backwards.reverse()
        return rows, backwards

    def _bimm(self, layout, record: bool = False):
        fmm_rows, fmm_tokens = self._fmm(layout, record=record)
        bmm_rows, bmm_tokens = self._bmm(layout, record=record)
        key_fmm = self.decision_key(fmm_tokens)
        key_bmm = self.decision_key(bmm_tokens)
        chosen = "bmm" if key_bmm <= key_fmm else "fmm"
        tokens = bmm_tokens if chosen == "bmm" else fmm_tokens
        if not record:
            return chosen, tokens
        rows = fmm_rows + bmm_rows
        for row in rows:
            row["selected"] = row["direction"] == chosen
        for position, row in enumerate(rows):
            row["step"] = position
        return chosen, rows

    def _run_fmm(self, layout) -> list[str]:
        return self._fmm(layout, record=False)[1]

    def _run_bmm(self, layout) -> list[str]:
        return self._bmm(layout, record=False)[1]

    def _run_bimm(self, layout) -> list[str]:
        return self._bimm(layout, record=False)[1]

    def _tokens(self, tokens: Sequence[str]) -> list[str]:
        return list(tokens)


_DEFAULT: MaximumMatching | None = None


def default_matcher() -> MaximumMatching:
    """Shared instance backed by the bundled dictionary, created on first use."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MaximumMatching()
    return _DEFAULT


def dictionary_metadata() -> dict:
    """Provenance of the bundled dictionary without building a Trie."""
    digest = hashlib.sha256(DEFAULT_DICTIONARY.read_bytes()).hexdigest() \
        if DEFAULT_DICTIONARY.is_file() else None
    entries = 0
    if DEFAULT_DICTIONARY.is_file():
        entries = sum(1 for line in DEFAULT_DICTIONARY.read_text(encoding="utf-8")
                      .splitlines() if line.strip())
    return {
        "dictionary_path": DICTIONARY_PATH.as_posix(),
        "dictionary_sha256": digest,
        "dictionary_entries": entries,
    }


def tokenize(text: str, method: str = "bimm") -> list[str]:
    """Contract entry point: `engine.tokenizer.max_matching:tokenize`."""
    return default_matcher().tokenize(text, method)


def tokenize_fmm(text: str) -> list[str]:
    return default_matcher().tokenize(text, "fmm")


def tokenize_bmm(text: str) -> list[str]:
    return default_matcher().tokenize(text, "bmm")


def tokenize_bimm(text: str) -> list[str]:
    return default_matcher().tokenize(text, "bimm")


def trace(text: str, method: str = "bimm") -> list[dict]:
    return default_matcher().trace(text, method)