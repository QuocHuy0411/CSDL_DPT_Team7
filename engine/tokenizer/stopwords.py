"""Tier-1 grammar stopword filtering for M1 (Maximum Matching preprocessing).

What this module is
-------------------
Tier-1 filtering removes *function* words - pronouns, prepositions, auxiliaries,
particles, degree/focus adverbs, discourse connectives - that carry almost no
discriminative power for retrieval, while keeping every term whose removal could
change the truth conditions, the polarity or the scope of a query.

Two curated resources drive it, both versioned in-repo:

``resources/grammar_stopwords.txt``
    The grammar inventory (function-like terms only).
``resources/guard_whitelist.txt``
    Terms that must never be removed, grouped by class.

The effective stopword set is ``grammar_stopwords - guard_whitelist``. Guards win
over stopwords, and a guard also protects every compound that contains one of its
components: guarding ``không`` protects ``không_thể`` and ``chẳng_hạn``,
guarding ``vì`` protects ``vì_dầu``. That is why negation and logic survive
filtering intact, including inside multi-syllable words.

``resources/tier1_source.md`` documents the curation rules, the deliberate
exclusions and the actual item counts. No external dictionary is repackaged.

Protection predicate (stable, reused by the TDV dimensional guard)
-------------------------------------------------------------------
With ``G`` the guard set, ``C = {c for g in G for c in components(g) if c in G}``
and ``components(x) = x.split("_")``::

    protected(t) := t in G  or  any(c in C for c in components(t))

A guard protects itself *and* every token that contains a guard **as a full
component**: guarding ``không`` keeps ``không``, ``không_thể`` and
``không_thể_nữa``; guarding ``vì`` keeps ``vì`` and ``vì_dầu``; guarding
``chẳng`` keeps ``chẳng_hạn``; guarding ``trước`` keeps ``trước_hết``.

Only components that are themselves guards are protectable. That is deliberate:
if every syllable of every guard were protectable, ``có_thể`` would make ``thể``
protectable and ``thể_thao`` would be falsely protected. Compounds that must be
protected for a different reason are listed explicitly in the guard file.
``is_protected`` implements exactly this and is exported at module level so the
TDV dimensional protection can mirror the same rule.

Usage
-----
::

    from engine.tokenizer.stopwords import filter_tier1, is_protected, tokenize_and_filter_t1

    filter_tier1(["bóng", "đá", "và", "của", "bóng_đá"])   # -> ["bóng", "đá", "bóng_đá"]
    is_protected("vì_dầu")                                 # -> True
    tokenize_and_filter_t1("Bóng đá và bóng đá!")          # lazy BiMM tokenizer + filter

``tokenize_and_filter_t1`` imports ``engine.tokenizer.max_matching`` lazily, so
this module stays importable (and unit-testable) before the M1 tokenizer exists.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
RESOURCE_DIR = ROOT / "resources"
STOPWORD_RESOURCE = "grammar_stopwords.txt"
GUARD_RESOURCE = "guard_whitelist.txt"
RULES_RESOURCE = "tier1_source.md"

SCHEMA_VERSION = 1

#: Module-level singleton, built from the curated resources on first use.
DEFAULT_FILTER: "Tier1StopwordFilter | None" = None


def canonical(token: str) -> str:
    """NFC + lowercase + stable separators; compound words keep their ``_``."""
    return "_".join(unicodedata.normalize("NFC", token).casefold().split())


def _parse_resource(path: Path) -> Tuple[List[str], Dict[str, str]]:
    """Return ``(terms, class_of_term)`` for a curated resource file."""
    terms: List[str] = []
    classes: Dict[str, str] = {}
    current_class = "uncategorised"
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip().strip("-").strip()
            if heading:
                current_class = heading
            continue
        term = canonical(line)
        if not term:
            continue
        terms.append(term)
        classes[term] = current_class
    return terms, classes


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Tier1StopwordFilter:
    """Tier-1 stopword filter with negation/logic guards and full provenance.

    Parameters
    ----------
    stopwords, guards:
        Explicit term sets. ``None`` (default) loads the curated resources.
    resource_dir:
        Directory holding the two curated files; defaults to ``resources/``.
    """

    def __init__(self, stopwords: Iterable[str] | None = None,
                 guards: Iterable[str] | None = None,
                 resource_dir: str | Path | None = None) -> None:
        self.resource_dir = Path(resource_dir) if resource_dir else RESOURCE_DIR
        if stopwords is None and guards is None:
            stopword_terms, stopword_classes = _parse_resource(
                self.resource_dir / STOPWORD_RESOURCE)
            guard_terms, guard_classes = _parse_resource(self.resource_dir / GUARD_RESOURCE)
            self.stopword_classes = stopword_classes
            self.guard_classes = guard_classes
        else:
            stopword_terms = [canonical(t) for t in (stopwords or ())]
            guard_terms = [canonical(t) for t in (guards or ())]
            self.stopword_classes = {term: "explicit" for term in stopword_terms}
            self.guard_classes = {term: "explicit" for term in guard_terms}
        self.stopwords: frozenset[str] = frozenset(stopword_terms)
        self.guards: frozenset[str] = frozenset(guard_terms)
        # Only components that are themselves guards protect compounds: that keeps
        # "không_thể" (via "không") and "vì_dầu" (via "vì") while leaving
        # "thể_thao" unprotected (its "thể" only appears inside the guard "có_thể").
        self.guard_components: frozenset[str] = frozenset(
            component for guard in self.guards for component in guard.split("_")
            if component in self.guards)
        self.effective_stopwords = frozenset(t for t in self.stopwords if not self.is_protected(t))

    # ------------------------------------------------------------------ filter

    def is_protected(self, token: str) -> bool:
        """``t in guards or any(component in guard_components)``.

        This is the single protection predicate of the M1 tier-1 layer; the TDV
        dimensional guard mirrors it so a protected term is never dropped on
        either side of the pipeline.
        """
        token = canonical(token)
        if token in self.guards:
            return True
        return any(component in self.guard_components for component in token.split("_"))

    def filter_tokens(self, tokens: Sequence[str]) -> List[str]:
        """Drop tier-1 stopwords; keep guards, guard-bearing compounds and content words."""
        kept: List[str] = []
        for token in tokens:
            if not isinstance(token, str):
                raise TypeError(
                    f"tokenizer must return str tokens, got {type(token).__name__}")
            normalized = canonical(token)
            if not normalized:
                continue
            if normalized in self.effective_stopwords and not self.is_protected(normalized):
                continue
            kept.append(normalized)
        return kept

    def dropped_tokens(self, tokens: Sequence[str]) -> List[str]:
        """Which input tokens tier-1 filtering would remove (audit helper)."""
        kept = self.filter_tokens(tokens)
        kept_counts: Dict[str, int] = {}
        for token in kept:
            kept_counts[token] = kept_counts.get(token, 0) + 1
        dropped: List[str] = []
        for token in tokens:
            normalized = canonical(token)
            if kept_counts.get(normalized, 0) > 0:
                kept_counts[normalized] -= 1
            else:
                dropped.append(normalized)
        return dropped

    # -------------------------------------------------------------- provenance

    def metadata(self) -> dict:
        """Provenance block: source files, actual counts and SHA256 digests."""
        resources: Dict[str, str] = {}
        for name in (STOPWORD_RESOURCE, GUARD_RESOURCE, RULES_RESOURCE):
            path = self.resource_dir / name
            resources[name] = sha256_file(path) if path.is_file() else None
        payload = {
            "component": "engine.tokenizer.stopwords.Tier1StopwordFilter",
            "schema_version": SCHEMA_VERSION,
            "tier": 1,
            "resource_dir": str(self.resource_dir),
            "stopword_source": STOPWORD_RESOURCE,
            "guard_source": GUARD_RESOURCE,
            "curation_rules_source": RULES_RESOURCE,
            "n_stopword_entries": len(self.stopwords),
            "n_guard_entries": len(self.guards),
            "n_effective_stopwords": len(self.effective_stopwords),
            "n_guard_components": len(self.guard_components),
            "n_protected_stopword_entries": len(self.stopwords & self.guards),
            "guard_classes": sorted(set(self.guard_classes.values())),
            "resource_sha256": resources,
            "status": "curated_for_m1_tier1",
        }
        payload["fingerprint"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return payload

    @classmethod
    def from_metadata(cls, metadata: Mapping) -> "Tier1StopwordFilter":
        """Rebuild a filter and refuse if the curated resources changed on disk."""
        instance = cls(resource_dir=metadata.get("resource_dir"))
        current = instance.metadata()
        if current["fingerprint"] != metadata.get("fingerprint"):
            raise ValueError(
                "Tier-1 resources changed (counts or SHA256 differ); rebuild the index")
        return instance

    def guard_report(self) -> Dict[str, List[str]]:
        """Guards grouped by curation class, for documentation and audits."""
        grouped: Dict[str, List[str]] = {}
        for term, group in sorted(self.guard_classes.items()):
            if term in self.guards:
                grouped.setdefault(group, []).append(term)
        return grouped

    def __repr__(self) -> str:
        return (f"Tier1StopwordFilter(stopwords={len(self.stopwords)}, "
                f"guards={len(self.guards)}, effective={len(self.effective_stopwords)})")


def get_default_filter() -> Tier1StopwordFilter:
    """Process-wide filter built from the curated resources (cached)."""
    global DEFAULT_FILTER
    if DEFAULT_FILTER is None:
        DEFAULT_FILTER = Tier1StopwordFilter()
    return DEFAULT_FILTER


def filter_tier1(tokens: Sequence[str]) -> List[str]:
    """Module-level tier-1 filter over already segmented tokens."""
    return get_default_filter().filter_tokens(tokens)


def is_protected(token: str) -> bool:
    """Module-level guard predicate: guard itself, or contains a guard component.

    Mirrors ``Tier1StopwordFilter.is_protected`` on the curated resources and is
    the predicate other roles (TDV dimensional protection) should reuse.
    """
    return get_default_filter().is_protected(canonical(token))


def tokenize_and_filter_t1(text: str) -> List[str]:
    """Segment ``text`` with the M1 tokenizer (default BiMM) then filter tier-1.

    The tokenizer import is lazy on purpose: this module must stay usable when
    ``engine.tokenizer.max_matching`` is absent (for example while only the
    stopword half of M1 is being developed or tested).
    """
    try:
        from engine.tokenizer.max_matching import tokenize
    except ImportError as error:  # pragma: no cover - depends on builder-data
        raise ImportError(
            "tokenize_and_filter_t1 needs engine.tokenizer.max_matching.tokenize; "
            "use filter_tier1(tokens) when only segmentation is not available."
        ) from error
    return filter_tier1(tokenize(text))


__all__ = [
    "GUARD_RESOURCE",
    "RESOURCE_DIR",
    "RULES_RESOURCE",
    "ROOT",
    "SCHEMA_VERSION",
    "STOPWORD_RESOURCE",
    "Tier1StopwordFilter",
    "canonical",
    "filter_tier1",
    "get_default_filter",
    "is_protected",
    "sha256_file",
    "tokenize_and_filter_t1",
]
