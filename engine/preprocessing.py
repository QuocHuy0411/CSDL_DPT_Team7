"""Fingerprint-checked M1 integration; the syllable baseline remains development only."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "engine.preprocessing:baseline_tokenize"
M1_TOKENIZER = "engine.tokenizer.max_matching:tokenize"
M1_TIER1 = "engine.tokenizer.stopwords:filter_tier1"
M1_RESOURCES = (
    "resources/dict.txt", "resources/grammar_stopwords.txt", "resources/guard_whitelist.txt",
    "engine/tokenizer/__init__.py", "engine/tokenizer/preprocessor.py",
    "engine/tokenizer/trie.py", "engine/tokenizer/max_matching.py",
    "engine/tokenizer/stopwords.py",
)
# A small integration fixture, NOT an approved M1 grammar list.
TIER1_FIXTURE = frozenset("và của là các những một ở tại với được bị bởi thì mà để này đó".split())
GUARDS = frozenset({"không", "chưa", "chẳng", "vì", "do_đó"})


def canonical(token: str) -> str:
    return "_".join(unicodedata.normalize("NFC", token).casefold().split())


def baseline_tokenize(text: str) -> list[str]:
    """Unicode syllable/word-character baseline; no Vietnamese word segmentation."""
    text = unicodedata.normalize("NFC", text).casefold()
    return re.findall(r"[^\W_]+(?:_[^\W_]+)*", text, flags=re.UNICODE)


def fixture_tier1(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in TIER1_FIXTURE or t in GUARDS]


def resolve_callable(spec: str) -> Callable:
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("Callable must use module:function syntax")
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise ValueError(f"Not callable: {spec}")
    return function


def read_stoplist(path: str | Path) -> set[str]:
    return {canonical(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")}


def protected_terms(m1: bool = False) -> frozenset[str]:
    """M1's complete pinned whitelist also protects dimensions from TDV removal."""
    return GUARDS | frozenset(read_stoplist(ROOT / "resources/guard_whitelist.txt")) if m1 else GUARDS


@dataclass
class Preprocessor:
    tokenizer_spec: str = BASELINE
    tier1_spec: str = "engine.preprocessing:fixture_tier1"
    tier1_enabled: bool = False
    m1_verified: bool = False
    resources: tuple[str, ...] = ()
    _tokenizer: Callable = field(init=False, repr=False)
    _tier1: Callable = field(init=False, repr=False)
    _protected: frozenset[str] = field(init=False, repr=False)
    _protected_components: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self):
        if self.m1_verified and self.tokenizer_spec == M1_TOKENIZER:
            # Track the entire implementation, not just the thin exported callable.
            self.resources = tuple(dict.fromkeys((*M1_RESOURCES, *self.resources)))
        self._tokenizer = resolve_callable(self.tokenizer_spec)
        self._tier1 = resolve_callable(self.tier1_spec)
        if self.m1_verified and (self.tokenizer_spec == BASELINE or
                               self.tier1_spec == "engine.preprocessing:fixture_tier1"):
            raise ValueError("Official preprocessing requires real M1 tokenizer and tier1 filter")
        self._protected = protected_terms(self.m1_verified)
        self._protected_components = frozenset(
            part for guard in self._protected for part in guard.split("_") if part in self._protected)

    def __call__(self, text: str) -> list[str]:
        tokens = self._tokenizer(unicodedata.normalize("NFC", text))
        if isinstance(tokens, str) or not isinstance(tokens, (list, tuple)):
            raise TypeError("M1 tokenizer must return a list of tokens, not a string")
        tokens = [canonical(t) for t in tokens if isinstance(t, str) and t.strip()]
        if self.tier1_enabled:
            filtered = self._tier1(tokens.copy())
            if isinstance(filtered, str) or not isinstance(filtered, (list, tuple)):
                raise TypeError("M1 tier1 filter must return a list of tokens")
            # The negation guard is an invariant at the M2 adapter boundary.
            retained = [canonical(t) for t in filtered]
            from collections import Counter
            counts = Counter(retained)
            output = []
            for token in tokens:
                if (token in self._protected or (self.m1_verified and
                        any(p in self._protected_components for p in token.split("_")))):
                    output.append(token)
                elif counts[token] > 0:
                    output.append(token)
                    counts[token] -= 1
            tokens = output
        return tokens

    def metadata(self) -> dict:
        files = {}
        for spec, function in ((self.tokenizer_spec, self._tokenizer),
                               (self.tier1_spec, self._tier1)):
            source = inspect.getsourcefile(function)
            files[spec] = hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None
        resource_hashes = {}
        for resource in self.resources:
            path = Path(resource)
            if not path.is_absolute():
                path = ROOT / path
            resource_hashes[resource] = hashlib.sha256(path.read_bytes()).hexdigest()
        payload = {"tokenizer": self.tokenizer_spec, "tier1_filter": self.tier1_spec,
                   "tier1_enabled": self.tier1_enabled, "m1_verified": self.m1_verified,
                   "resources": list(self.resources), "source_hashes": files,
                   "resource_hashes": resource_hashes,
                   "status": "official" if self.m1_verified else "development_only"}
        payload["fingerprint"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return payload

    @classmethod
    def from_metadata(cls, metadata: dict) -> "Preprocessor":
        instance = cls(metadata["tokenizer"], metadata["tier1_filter"],
                       metadata["tier1_enabled"], metadata["m1_verified"],
                       tuple(metadata.get("resources", [])))
        if instance.metadata()["fingerprint"] != metadata["fingerprint"]:
            raise ValueError("Preprocessing source/resources changed; rebuild the index")
        return instance
