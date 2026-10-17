"""Syllable level prefix trie for Vietnamese Maximum Matching.

Why syllable level and not character level: a Vietnamese dictionary entry is a
sequence of syllables ("cơ sở dữ liệu"), and the algorithm slides a window over
syllables. A character trie forces a string slice plus a hash lookup at every
window, which re-scans the same characters many times. A syllable trie walks one
edge per syllable, so a window of k syllables costs exactly k dictionary steps.

The public shape still matches the project plan (`insert(word)`, `search(word)`,
`is_end_of_word`); a "word" here is the space separated syllable string that
`resources/dict.txt` stores, one entry per line.
"""

from __future__ import annotations

from typing import Iterable, Sequence

__all__ = ["TrieNode", "Trie", "split_syllables"]

SEPARATOR = " "


def split_syllables(word: str | Iterable[str]) -> tuple[str, ...]:
    """Accept 'cơ sở dữ liệu' or ('cơ', 'sở', 'dữ', 'liệu') -> the same tuple."""
    if isinstance(word, str):
        return tuple(word.split(SEPARATOR)) if word else ()
    return tuple(word)


class TrieNode:
    """One syllable position. `children` maps a syllable to the next node."""

    __slots__ = ("children", "is_end_of_word", "depth")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False
        self.depth: int = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TrieNode(end={self.is_end_of_word}, children={len(self.children)})"


class Trie:
    """Prefix trie over syllable sequences."""

    def __init__(self, words: Iterable[str] | None = None) -> None:
        self.root = TrieNode()
        self.size = 0
        self.max_syllables = 0
        if words is not None:
            for word in words:
                self.insert(word)

    def insert(self, word: str | Iterable[str]) -> bool:
        """Insert one entry. Returns False when it was already present."""
        syllables = split_syllables(word)
        if not syllables or any(not syllable for syllable in syllables):
            return False
        node = self.root
        for syllable in syllables:
            child = node.children.get(syllable)
            if child is None:
                child = TrieNode()
                child.depth = node.depth + 1
                node.children[syllable] = child
            node = child
        if node.is_end_of_word:
            return False
        node.is_end_of_word = True
        self.size += 1
        self.max_syllables = max(self.max_syllables, len(syllables))
        return True

    def search(self, word: str | Iterable[str]) -> bool:
        """Exact match on the whole syllable sequence."""
        node = self._walk(split_syllables(word))
        return node is not None and node.is_end_of_word

    def __contains__(self, word: object) -> bool:
        return isinstance(word, (str, tuple, list)) and self.search(word)

    def __len__(self) -> int:
        return self.size

    def longest_match(self, syllables: Sequence[str], start: int,
                      max_length: int) -> int:
        """Longest entry starting at `start`; returns its exclusive end index.

        Returns `start` when nothing matches, which means "no match" for the
        caller. `max_length` bounds the window so the caller never walks past the
        end of the text or past a hard boundary.
        """
        limit = min(len(syllables), start + max_length)
        best = start
        node = self.root
        for end in range(start + 1, limit + 1):
            child = node.children.get(syllables[end - 1])
            if child is None:
                break
            node = child
            if node.is_end_of_word:
                best = end
        return best

    def _walk(self, syllables: Sequence[str]) -> TrieNode | None:
        node = self.root
        for syllable in syllables:
            child = node.children.get(syllable)
            if child is None:
                return None
            node = child
        return node

    def stats(self) -> dict:
        return {
            "entries": self.size,
            "max_syllables": self.max_syllables,
            "root_children": len(self.root.children),
        }