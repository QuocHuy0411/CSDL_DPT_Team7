"""M1 Vietnamese text processing: normalizer, trie and Maximum Matching.

Import policy for this package: `stopwords` belongs to builder-retrieval and is
**not** imported here, eagerly or lazily, because tier 1 filtering and the bundled
dictionary have to be usable without pulling in stopword resources. Code that
wants the filter imports `engine.tokenizer.stopwords` itself and uses the lazy
`tokenize_and_filter_t1` helper there.

The standard final preprocessing spec for this project is:

    tokenizer = engine.tokenizer.max_matching:tokenize
    tier1     = engine.tokenizer.stopwords:filter_tier1
"""

from __future__ import annotations

from engine.tokenizer import max_matching, preprocessor, trie
from engine.tokenizer.max_matching import (
    MaximumMatching,
    default_matcher,
    tokenize,
    tokenize_bimm,
    tokenize_bmm,
    tokenize_fmm,
    trace,
)
from engine.tokenizer.preprocessor import normalize, normalize_segments, syllables
from engine.tokenizer.trie import Trie, TrieNode

__all__ = [
    "max_matching",
    "preprocessor",
    "trie",
    "MaximumMatching",
    "Trie",
    "TrieNode",
    "default_matcher",
    "normalize",
    "normalize_segments",
    "syllables",
    "tokenize",
    "tokenize_fmm",
    "tokenize_bmm",
    "tokenize_bimm",
    "trace",
    "PREPROCESSING_SPEC",
]

#: The tokenizer spec lead wires into the index build and fingerprints.
PREPROCESSING_SPEC = "engine.tokenizer.max_matching:tokenize"
#: The tier 1 spec, owned by builder-retrieval.
TIER1_SPEC = "engine.tokenizer.stopwords:filter_tier1"