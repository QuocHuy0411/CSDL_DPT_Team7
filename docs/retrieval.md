# Retrieval (builder-retrieval)

Owner: `engine/indexer/tfidf_engine.py`, `engine/indexer/inverted_index.py`,
`engine/indexer/search_engine.py`, `tests/test_ir_math.py`,
`tests/test_search_engine.py`.

This document is the reference for the retrieval layer: the exact mathematics,
the public `SearchEngine` API, and the internal fields other roles (TDV,
evaluation, benchmarks, app) may rely on. It follows `docs/api_contracts.md`
("API retrieval"); where the two disagree the contract wins.

Nothing in this layer reads or writes `data/raw`. The corpus is only ever
received in memory as a sequence of records built by `builder-data`.

---

## 1. Mathematics

For a document `d` and a term `t`:

| Quantity | Definition | Notes |
| --- | --- | --- |
| `f(t, d)` | raw frequency of `t` in `d` | stored as the CSR integer matrix `counts_` |
| `TF(t, d)` | `1 + ln f(t, d)`, and `0` when `f = 0` | sublinear TF |
| `df(t)` | number of documents containing `t` | `df_`, one entry per vocabulary column |
| `IDF(t)` | `ln((N + 1) / (df(t) + 1)) + 1` | smoothed, `N = n_documents_` |
| `w(t, d)` | `TF(t, d) * IDF(t)` | raw weight, see `TfidfEngine.unnormalized()` |
| `w~(t, d)` | `w(t, d) / \|\|d||_2` | stored in `matrix_`; rows with `\|\|d\|\| = 0` stay zero |
| `cos(q, d)` | `sum_t w~(t, q) * w~(t, d)` | both sides already L2-normalised |

Frozen-at-fit-time rules (they are the contract, not tuning knobs):

* The **vocabulary is built from the corpus only**. A query can never extend it;
  unknown query terms are dropped and reported in `QueryVector.oov_terms`.
* The **IDF table is computed from the corpus only**. A query is weighted with
  the corpus IDF, never with a document frequency computed from the query
  itself. `test_query_uses_corpus_idf_even_when_the_query_is_itself_a_document_term`
  pins this.
* A document that has no surviving token (only stopwords, or nothing at all) is
  a **zero row**: it is never a candidate and never returned by `search`.
* **The same stopword set** removes tokens from corpus and query alike
  (`TfidfEngine.filter_tokens` is the single entry point used by both paths).
* Empty and whitespace-only tokens are dropped before counting, matching the
  `engine.preprocessing` boundary. Any reference tokenizer used for comparison
  must apply the same rule.

### Parity with scikit-learn

`TfidfVectorizer(analyzer="word", tokenizer=<callable>, lowercase=False,
token_pattern=None, sublinear_tf=True, norm="l2")` reproduces `matrix_`,
`idf_` and the query vector to ~1e-16 (asserted in `tests/test_ir_math.py`).
Every numeric claim is checked **twice**: against scikit-learn and against a
plain-Python reference implementation written independently inside the test file
(`reference_weights`, `reference_cosine`).

---

## 2. Public API

```python
from engine.indexer.search_engine import SearchEngine

engine = SearchEngine(tokenizer=None, stopwords=None, metadata=None)
engine.fit(documents, tokenized_docs=None) -> SearchEngine
engine.search(query, top_k=10) -> list[dict]
engine.explain(query, doc_id) -> dict
engine.score(query, doc_id) -> float          # convenience, equals explain()["score"]
engine.document(doc_id) -> dict               # stored display fields
engine.save(directory) -> manifest dict
SearchEngine.load(directory, tokenizer=None) -> SearchEngine
```

### Constructor

| Argument | Meaning |
| --- | --- |
| `tokenizer` | callable `str -> list[str]`. `None` selects the whitespace baseline (`engine.indexer.search_engine:_whitespace_tokenize`), which the contract marks as development only. In production pass the `engine.preprocessing.Preprocessor` instance. |
| `stopwords` | iterable of tokens, stored as `frozenset` in `stopwords_`, applied identically to corpus and query. |
| `metadata` | provenance dict, copied into `engine.metadata` and restored verbatim by `load`. The retrieval layer adds one key of its own, `metadata["retrieval"]` (see §5). |

### `fit(documents, tokenized_docs=None)`

* `documents` is a sequence of mappings with `doc_id` (unique, non-empty `str`),
  `title`, `content`, `category`. Only `content` is indexed: the title already
  lives inside the content and is never counted a second time.
* `tokenized_docs`, when given, must be aligned row-for-row with `documents` and
  is used verbatim (this is how `Preprocessor` output enters the index); a length
  mismatch raises `ValueError`. When omitted, `content` is tokenized with
  `self.tokenizer`.
* Raises `TypeError` for non-mapping rows, non-`str` tokens, a non-callable
  tokenizer or a mapping passed instead of a sequence; `ValueError` for missing /
  duplicate `doc_id` or misaligned tokens.
* The input documents and the input token lists are never mutated.
* `fit` is also what sets `metadata["retrieval"]["tokenized_source"]`
  (`"engine_tokenizer"` or `"caller"`).

### `search(query, top_k=10)`

* `query` is a `str` (tokenized with `self.tokenizer`) or a list of tokens (used
  as-is). Non-string, non-token input raises `TypeError`.
* Each result has **exactly** the keys `doc_id`, `title`, `category`, `score`,
  `snippet`. Rank is the 1-based position in the returned list; it is not a
  field.
* Ordering: score descending, then **`doc_id` ascending** for ties.
* Only strictly positive scores are returned. An empty, whitespace-only,
  stopword-only or fully out-of-vocabulary query yields `[]`, because it produces
  a zero query vector.
* `top_k=0` returns `[]`; a negative `top_k` raises `ValueError`, a non-integer
  raises `TypeError`.
* Zero-vector documents are never candidates, so they can never appear.

### `explain(query, doc_id)`

```python
{
  "doc_id": "D1", "title": ..., "category": ...,
  "score": 0.71,                      # == sum(contribution)
  "contributions": [
    {"term": "đá", "query_tf": 1, "doc_tf": 2, "idf": 1.2876820724517808,
     "query_weight": 0.7071067811865476, "doc_weight": 0.7071067811865476,
     "contribution": 0.5000000000000001}
  ],
  "query_terms": ["đá"],              # in-vocabulary query terms, by term id
  "oov_terms": ["zzz"],               # dropped: not in the corpus vocabulary
  "dropped_terms": ["và"],            # dropped: stopword / empty token
  "query_norm": 1.82,                 # L2 norm of the raw query weights
  "doc_norm": 3.08                    # L2 norm of the raw document weights
}
```

Contract of the numbers:

* `query_tf` / `doc_tf` are **raw integer frequencies**.
* `idf` is the **corpus** IDF.
* `query_weight` / `doc_weight` are the **L2-normalised** weights, i.e. exactly
  the numbers multiplied by the cosine. Raw weights are
  `query_weight * query_norm` and `doc_weight * doc_norm`
  (also available as `TfidfEngine.doc_raw_weight`).
* `contribution = query_weight * doc_weight`, and
  `sum(contributions) == score` (`math.fsum`, so the identity holds to ~1e-15 and
  matches `search`'s score within 1e-12).
* Contributions are sorted by `|contribution|` descending, then term ascending;
  the sort does not affect the sum.
* **A term the document does not contain is still reported**, with
  `doc_tf = 0`, `doc_weight = 0.0` and `contribution = 0.0`. This is
  intentional: an explanation shows which query terms were missed instead of
  silently hiding them. Such rows contribute exactly `0.0` to the total, so the
  sum identity is unaffected.
* A document with no query term at all therefore yields `score == 0.0` with one
  zero row per query term.
* An unknown `doc_id` raises `KeyError`; calling `explain` before `fit` raises
  `RuntimeError`.

---

## 3. Persistence

`save(directory)` writes a self-contained, hashed index:

| File | Content |
| --- | --- |
| `manifest.json` | schema version, sizes, maths provenance, tokenizer descriptor, stopwords, `metadata`, SHA256 of every other file |
| `tfidf_state.json` | TF-IDF parameters, `stopwords`, shapes, SHA256 of the four files below |
| `vocabulary.json` | vocabulary terms in column order |
| `idf_weights.json` | IDF per column, formula string, `n_documents` |
| `tf_counts.npz` | raw frequencies, CSR (`int64`) |
| `doc_norms.npy` | L2 norm of each raw document vector |
| `inverted_index.npz` | posting offsets, document rows, posting weights |
| `doc_metadata.json` | `doc_id`, `title`, `category`, precomputed `snippet` per document |

Design choices worth knowing:

* The normalised matrix, the IDF table and `df` are **derived** from
  `tf_counts.npz` + `idf_weights.json` on load instead of being duplicated on
  disk. `load` recomputes them and cross-checks the result against
  `doc_norms.npy`, so a hand-edited IDF table or count matrix is rejected even
  when its SHA256 was also updated.
* `df_` is restored on `load` (regression-tested), so `term_stats()` is accurate
  for a loaded index.
* Only the **snippet** of each document is stored, not the full content: the
  search API never needs the rest, and the index stays small enough for the
  size-reduction target. Re-derive richer snippets from `data/corpus.jsonl`.
* `save` never deletes foreign files in the target directory, so
  `scripts/build_indexes.py` can write `build_metadata.json`, `tdv_*.json` next
  to the index.
* Every file listed in a manifest is verified with SHA256 on load; a mismatch or
  a missing file raises `ValueError`.

### Tokenizer safety on load

`load` never silently changes how queries are analysed:

* **Without `tokenizer=`**: the tokenizer is re-imported from the saved
  `module:qualname` spec. If the saved tokenizer was not a plain importable
  function (for example a `Preprocessor` *instance*), `load` raises and asks for
  an explicit `tokenizer=` instead of guessing.
* **With `tokenizer=`**: the fingerprint must match. The fingerprint is the
  SHA256 of the callable's source file combined with its `module:qualname`
  (for a callable instance: its class). Configuration that the tokenizer
  reports itself is compared as well: if it exposes `metadata()["fingerprint"]`
  — this is how `Preprocessor` reports tokenizer / tier1 / resource
  configuration — the two values must be equal. A tokenizer whose fingerprint
  cannot be computed (lambda in a REPL, dynamically built callable) is refused.
* All mismatches raise `ValueError` telling the caller to rebuild.

This is what makes `scripts/search.py::load_engine` safe: the `Preprocessor` is
reconstructed from `build_metadata.json`, and a configuration mismatch (for
example `tier1_enabled=False` for a C3 index) is refused instead of producing
silently different results.

---

## 4. Internal fields available to other roles

These are internal (no stability guarantee beyond this milestone) but they are
the intended hand-off, especially for `engine.indexer.tdv_engine.TDVEngine`,
`evaluator/**` and `benchmarks/**`.

### `engine.tfidf` (a `TfidfEngine`)

| Attribute | Meaning |
| --- | --- |
| `vocabulary_` | `dict[str, int]`, term to column, sorted lexicographically (same convention as scikit-learn) |
| `terms()` / `term(term_id)` / `term_id(term)` | vocabulary in both directions |
| `n_features_`, `n_documents_` | shapes |
| `idf_` | `float64[n_features_]`, corpus IDF |
| `df_` | `int64[n_features_]`, document frequency |
| `counts_` | CSR `int64[n_documents_, n_features_]`, raw frequencies |
| `matrix_` | CSR `float64`, L2-normalised TF-IDF. Rows of all-zero documents stay zero |
| `doc_norms_` | `float64[n_documents_]`, L2 norm of the raw weights |
| `unnormalized()` | rebuilds `tf * idf` before L2 (on demand, not stored) |
| `transform_documents(tokenized_docs)` / `transform_full(...)` | weights new token rows with the frozen vocabulary and corpus IDF (OOV dropped) |
| `vectorize(tokens)` | a `QueryVector` (below) |
| `cosine_scores(query_vector)` | cosine against every document, `float64[n_documents_]` |
| `to_dense()` | dense copy of `matrix_`, for tests and inspection only |
| `term_stats(term)` | `{"term", "term_id", "df", "idf", "document_frequency_ratio"}`; unknown terms give `term_id=None, df=0, idf=0.0` |
| `doc_term_tf(doc, term_id)`, `doc_term_weight(doc, term_id)`, `doc_raw_weight(doc, term_id)` | per-pair access |
| `filter_tokens(tokens)` / `dropped_tokens(tokens)` | the one stopword rule, usable for reporting |
| `stopwords_` | `frozenset` actually applied |
| `save(directory)` / `load(directory)` | standalone persistence (also verifies hashes) |

`QueryVector` fields: `terms`, `term_ids`, `tfs` (raw), `idf`, `raw_weights`,
`weights` (L2-normalised), `norm`, `oov_terms`; `is_empty`, `__len__`,
`weights_by_term()`.

### `engine.index` (an `InvertedIndex`)

CSR-like over the **term** axis, so a posting list is an O(1) slice:

* `offsets` `int64[n_features + 1]`, `docs` `int32[nnz]`, `weights` `float64[nnz]`
  (the L2-normalised weight of that document/term pair).
* `posting_list(term_or_id) -> tuple[int, ...]` document rows containing the term.
* `posting_slice(term_or_id) -> (docs, weights)`, `posting_length(...)`.
* `candidates(term_ids) -> np.ndarray` sorted unique union of posting lists.
* `score(query_vector) -> (rows, scores)` exact cosine of the candidates only;
  every stored weight is strictly positive, so every candidate scores `> 0`.
* `cosine_scores_full(query_vector) -> np.ndarray` dense per-document view.
* `stats()`: documents, features, total/empty postings, longest and mean posting
  length, and the byte size of the two posting arrays.
* `save(directory)` / `load(directory, vocabulary=...)`.

### `engine` (the `SearchEngine` itself)

`doc_ids_`, `titles_`, `categories_`, `snippets_` (all aligned with the row
order of `matrix_`), `stopwords_`, `tokenizer`, `tokenizer_explicit`,
`snippet_width` (default 240 characters, whitespace-collapsed head of the
content, `…` when truncated), `metadata`, `is_fitted`, `n_documents`,
`document(doc_id)`, `tokenize(text)`.

Snippet policy: deterministic head of the content. It is intentionally *not*
query-dependent, so that a snippet never leaks a term-matching rule that differs
from the tokenizer.

---

## 5. `metadata["retrieval"]` provenance

Written by `fit`, restored verbatim by `load`:

```python
{"engine": "engine.indexer.search_engine.SearchEngine", "schema_version": 1,
 "math": {"term_frequency": "1 + ln(f)", "idf": "ln((N + 1) / (df + 1)) + 1",
          "idf_scope": "corpus_only", "normalization": "l2",
          "similarity": "cosine", "tie_break": "doc_id_ascending"},
 "n_documents": 2500, "vocabulary_size": 17744, "stopword_count": 0,
 "tokenized_source": "caller", "snippet_width": 240,
 "tokenizer": {"kind": "callable_object", "label": "engine.preprocessing:Preprocessor",
               "spec": None, "fingerprint": "...", "state_fingerprint": "...",
               "resolvable": false}}
```

This block is additive: any other key the caller passed (configuration, corpus
hash, preprocessing status) is preserved byte-for-byte, which is what
`scripts/search.py` relies on.

---

## 6. Tests

```bash
python -m pytest tests/test_ir_math.py tests/test_search_engine.py -q
```

* `tests/test_ir_math.py` — vocabulary/IDF/TF formulas, L2 rows, zero rows,
  corpus-IDF-for-query, frozen vocabulary, cosine against scikit-learn **and**
  against an independent Python reference, posting lists vs the CSR matrix,
  candidate set == positive-score set, randomised parity (seed 2026, 40 and 120
  documents), plus the persistence regressions (restored `df_`, SHA256
  verification, IDF/count consistency).
* `tests/test_search_engine.py` — result contract, ordering and tie-breaks,
  `top_k` validation, empty/stopword/OOV/zero-vector queries, symmetric stopword
  handling, `explain` field contract and sum identity, input immutability,
  validation errors, and the persistence/tokenizer-safety matrix.

Fixtures deliberately include single-character tokens, a repeated term, a
stopword-only (zero vector) document, a document that never matches a query, and
out-of-vocabulary query terms. No fixture reads `data/raw`; all artefacts are
written to `tmp_path`.

Measured on this machine against the built `data/indexes/C1..C4`
(development tokenizer, 20 queries sampled from the corpus, 5 repeats each —
**not** the official benchmark, which belongs to `builder-experiment`):

| Config | docs | vocabulary | stopwords | load | mean | P95 | bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | 2500 | 17744 | 0 | 112 ms | 1.10 ms | 1.62 ms | 10.13 MB |
| C2 | 2500 | 17404 | 1942 | 93 ms | 0.52 ms | 1.01 ms | 7.37 MB |
| C3 | 2500 | 17727 | 0 | 105 ms | 1.10 ms | 1.92 ms | 9.72 MB |
| C4 | 2500 | 17233 | 494 | 68 ms | 0.24 ms | 0.66 ms | 6.16 MB |

`sum(contribution) == score` held on every configuration. These numbers are
informational only; the reportable latency and index-size figures must come from
`benchmarks/**`.

---

## 7. Integration notes for the lead

* **Index format**: `schema_version = 1`, unchanged since the first build of
  `data/indexes/C1..C4`. Any change to the on-disk layout must bump
  `engine.indexer.tfidf_engine.SCHEMA_VERSION` and require a rebuild; `load`
  refuses unknown versions.
* **Post-index source edits**: editing the retrieval sources does *not* corrupt
  existing indexes (the manifest only hashes index files, not source), but it can
  change scoring or tokenisation. Any such edit after a build must be announced
  so the four configurations can be rebuilt together; the retrieval tests must be
  re-run against the rebuilt artefacts.
* **Postings memory**: `inverted_index.npz` (uncompressed, `float64` weights +
  `int32` rows) is the largest file and dominates index size. If the 30 %
  reduction target needs more, the levers are, in order: `float32` posting
  weights (halves the biggest array, cosine stays within ~1e-7), and compressed
  `np.savez_compressed` (smaller on disk, slower to load). Neither is applied
  here because both trade the latency target against the size target, and that
  trade-off is `builder-experiment`'s to measure, not this layer's to assume.
* **What is not implemented here**: BM25 and binary-cosine baselines needed by
  the pooling step (`data/ground_truth/qrels.json`) are not part of this layer;
  `builder-data`/`builder-experiment` currently only have TF-IDF. TDV is
  `builder-experiment`'s file and only consumes `engine.tfidf` (§4).
