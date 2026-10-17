"""IR mathematics: TF-IDF weights, corpus IDF, L2 norms, postings and persistence.

Every number asserted here is derived twice: once against
``sklearn.feature_extraction.text.TfidfVectorizer`` configured with the same
callable tokenizer, and once against a plain-Python reference implementation
written independently in this file. Fixtures deliberately include single
character tokens, a repeated term, a document that becomes a zero vector after
stopword removal, and out-of-vocabulary query terms.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from engine.indexer.inverted_index import InvertedIndex
from engine.indexer.tfidf_engine import TfidfEngine

SEED = 2026
TOLERANCE = 1e-12


# --------------------------------------------------------------------- fixtures

# 1-character tokens, a repeated term, a stopword-only (zero vector) document.
TOKEN_DOCS = [
    ["a", "b", "b"],
    ["b", "c"],
    ["a"],
    ["the", "the"],
    ["x"],
    ["y", "y", "y", "a", "the"],
    ["c", "c", "a", "b"],
]
STOPWORDS = {"the"}
DOC_TEXTS = [" ".join(tokens) for tokens in TOKEN_DOCS]


def whitespace_tokens(text: str) -> list[str]:
    return text.split()


def punctuation_tokens(text: str) -> list[str]:
    return [token.strip(".,") for token in text.split() if token.strip(".,")]


def sklearn_vectorizer(**kwargs) -> TfidfVectorizer:
    """sklearn configured exactly like the contract: sublinear TF, L2, smooth IDF."""
    return TfidfVectorizer(analyzer="word", tokenizer=whitespace_tokens, lowercase=False,
                           token_pattern=None, sublinear_tf=True, norm="l2", **kwargs)


def reference_weights(docs, stopwords=frozenset()):
    """Independent pure-Python reference for the contract formulas."""
    vocabulary = sorted({token for tokens in docs for token in tokens if token not in stopwords})
    n_documents = len(docs)
    frequency = {term: sum(1 for tokens in docs if term in set(tokens))
                 for term in vocabulary}
    idf = {term: math.log((n_documents + 1) / (frequency[term] + 1)) + 1.0
           for term in vocabulary}
    rows = []
    for tokens in docs:
        counts = Counter(token for token in tokens if token not in stopwords)
        weights = {term: (1.0 + math.log(counts[term])) * idf[term] for term in counts}
        norm = math.sqrt(sum(value * value for value in weights.values()))
        rows.append({term: value / norm for term, value in weights.items()} if norm > 0 else {})
    return vocabulary, idf, frequency, rows


def reference_cosine(query_tokens, documents, stopwords=frozenset()):
    vocabulary, idf, _, rows = reference_weights(documents, stopwords)
    counts = Counter(token for token in query_tokens if token not in stopwords)
    weights = {term: (1.0 + math.log(counts[term])) * idf[term]
               for term in counts if term in vocabulary}
    norm = math.sqrt(sum(value * value for value in weights.values()))
    if norm == 0:
        return [0.0] * len(rows)
    normalized = {term: value / norm for term, value in weights.items()}
    return [sum(normalized.get(term, 0.0) * value for term, value in row.items())
            for row in rows]


def fitted_engine(docs=TOKEN_DOCS, stopwords=STOPWORDS) -> TfidfEngine:
    return TfidfEngine(stopwords=stopwords).fit(docs)


# ------------------------------------------------------------------- TF / IDF

def test_vocabulary_is_sorted_and_excludes_stopwords():
    engine = fitted_engine()
    assert engine.vocabulary_ == {"a": 0, "b": 1, "c": 2, "x": 3, "y": 4}
    assert "the" not in engine.vocabulary_
    assert engine.n_features_ == 5
    assert engine.n_documents_ == len(TOKEN_DOCS)


def test_idf_matches_sklearn_callable_tokenizer_fixture():
    engine = fitted_engine()
    sklearn = sklearn_vectorizer(stop_words=sorted(STOPWORDS)).fit(DOC_TEXTS)
    assert engine.vocabulary_ == sklearn.vocabulary_
    assert np.allclose(engine.idf_, sklearn.idf_, rtol=0, atol=1e-15)


def test_idf_values_follow_the_smoothed_formula():
    engine = fitted_engine()
    for term, position in engine.vocabulary_.items():
        expected = math.log((len(TOKEN_DOCS) + 1) / (int(engine.df_[position]) + 1)) + 1.0
        assert engine.idf_[position] == pytest.approx(expected, abs=TOLERANCE)
        # "b" occurs in documents 0, 1 and 6 -> df = 3 with N = 7
    assert int(engine.df_[engine.vocabulary_["b"]]) == 3
    assert engine.idf_[engine.vocabulary_["b"]] == pytest.approx(math.log(8 / 4) + 1.0,
                                                                 abs=TOLERANCE)
    assert engine.idf_[engine.vocabulary_["y"]] == pytest.approx(math.log(8 / 2) + 1.0,
                                                                 abs=TOLERANCE)


def test_sublinear_term_frequency_is_not_linear():
    engine = fitted_engine()
    tf = engine.doc_term_tf(5, engine.vocabulary_["y"])  # "y y y" -> f = 3
    assert tf == 3
    raw = engine.doc_raw_weight(5, engine.vocabulary_["y"])
    expected = (1.0 + math.log(3.0)) * engine.idf_[engine.vocabulary_["y"]]
    assert raw == pytest.approx(expected, abs=TOLERANCE)
    assert raw < 3.0 * engine.idf_[engine.vocabulary_["y"]]  # linear TF would be larger


def test_document_matrix_matches_sklearn():
    engine = fitted_engine()
    sklearn = sklearn_vectorizer(stop_words=sorted(STOPWORDS)).fit(DOC_TEXTS)
    assert np.allclose(engine.to_dense(), sklearn.transform(DOC_TEXTS).toarray(),
                       rtol=0, atol=TOLERANCE)


def test_document_matrix_matches_independent_reference():
    engine = fitted_engine()
    vocabulary, idf, frequency, rows = reference_weights(TOKEN_DOCS, STOPWORDS)
    assert engine.vocabulary_ == {term: index for index, term in enumerate(vocabulary)}
    for position, term in enumerate(vocabulary):
        assert int(engine.df_[position]) == frequency[term]
        assert engine.idf_[position] == pytest.approx(idf[term], abs=TOLERANCE)
    for row, expected in enumerate(rows):
        for position, term in enumerate(vocabulary):
            assert engine.to_dense()[row, position] == pytest.approx(
                expected.get(term, 0.0), abs=TOLERANCE)


def test_rows_are_l2_normalised_and_zero_documents_stay_zero():
    engine = fitted_engine()
    dense = engine.to_dense()
    norms = np.sqrt((dense ** 2).sum(axis=1))
    assert np.allclose(np.delete(norms, 3), 1.0, rtol=0, atol=TOLERANCE)
    assert norms[3] == 0.0  # document 3 contains only stopwords
    assert np.all(dense[3] == 0.0)
    assert engine.doc_norms_[3] == 0.0
    assert engine.counts_.getrow(3).nnz == 0


def test_counts_matrix_holds_raw_frequencies():
    engine = fitted_engine()
    assert isinstance(engine.counts_, csr_matrix)
    assert engine.counts_[1, engine.vocabulary_["b"]] == 1
    assert engine.counts_[0, engine.vocabulary_["b"]] == 2
    assert "the" not in engine.vocabulary_  # the stopword column never exists


# ------------------------------------------------------------- query vector

def test_query_vector_matches_sklearn_transform():
    engine = fitted_engine()
    sklearn = sklearn_vectorizer(stop_words=sorted(STOPWORDS)).fit(DOC_TEXTS)
    for query in ("a b", "y y y", "a b c", "the a"):
        expected = sklearn.transform([query]).toarray()[0]
        vector = engine.vectorize(query.split())
        actual = np.zeros(engine.n_features_)
        actual[vector.term_ids] = vector.weights
        assert np.allclose(actual, expected, rtol=0, atol=TOLERANCE)


def test_query_uses_corpus_idf_even_when_the_query_is_itself_a_document_term():
    engine = fitted_engine()
    vector = engine.vectorize(["b"])
    corpus_idf = engine.idf_[engine.vocabulary_["b"]]
    query_local_idf = math.log((len(TOKEN_DOCS) + 1) / 2) + 1.0  # df inside the query itself
    assert query_local_idf != pytest.approx(corpus_idf, abs=TOLERANCE)
    assert vector.weights[0] == pytest.approx(1.0, abs=TOLERANCE)  # single term normalises to 1
    assert vector.raw_weights[0] == pytest.approx(corpus_idf, abs=TOLERANCE)


def test_query_normalisation_and_sublinear_tf():
    engine = fitted_engine()
    vector = engine.vectorize(["b", "b", "b"])
    assert vector.terms == ("b",)
    assert vector.tfs[0] == 3.0
    assert vector.raw_weights[0] == pytest.approx(
        (1.0 + math.log(3.0)) * engine.idf_[engine.vocabulary_["b"]], abs=TOLERANCE)
    assert vector.norm == pytest.approx(abs(vector.raw_weights[0]), abs=TOLERANCE)
    assert vector.weights[0] == pytest.approx(1.0, abs=TOLERANCE)


def test_query_cannot_extend_the_corpus_vocabulary():
    engine = fitted_engine()
    vocabulary = dict(engine.vocabulary_)
    vector = engine.vectorize(["unknown_term", "a"])
    assert engine.vocabulary_ == vocabulary
    assert engine.n_features_ == len(vocabulary)
    assert vector.terms == ("a",)
    assert vector.oov_terms == ("unknown_term",)


@pytest.mark.parametrize("tokens", [[], ["the"], ["the", "the"], ["zzz"], ["zzz", "the"]])
def test_empty_stopword_and_oov_queries_give_a_zero_vector(tokens):
    engine = fitted_engine()
    vector = engine.vectorize(tokens)
    assert vector.is_empty
    assert vector.norm == 0.0
    assert np.allclose(engine.cosine_scores(vector), 0.0)


# --------------------------------------------------------- cosine / postings

def test_cosine_scores_have_one_value_per_document():
    """Regression: the query has fewer terms than the vocabulary."""
    engine = fitted_engine()
    vector = engine.vectorize(["a"])
    scores = engine.cosine_scores(vector)
    assert scores.shape == (len(TOKEN_DOCS),)
    assert np.allclose(scores, reference_cosine(["a"], TOKEN_DOCS, STOPWORDS),
                       rtol=0, atol=TOLERANCE)


def test_cosine_scores_match_sklearn_similarities():
    engine = fitted_engine()
    sklearn = sklearn_vectorizer(stop_words=sorted(STOPWORDS)).fit(DOC_TEXTS)
    for query in ("a b", "y", "a b c"):
        expected = (sklearn.transform([query]) @ sklearn.transform(DOC_TEXTS).T).toarray()[0]
        actual = engine.cosine_scores(engine.vectorize(query.split()))
        assert np.allclose(actual, expected, rtol=0, atol=TOLERANCE)


def test_posting_lists_match_the_csr_matrix():
    engine = fitted_engine()
    index = InvertedIndex().fit(engine.matrix_, vocabulary=engine.vocabulary_)
    dense = engine.matrix_.toarray()
    for term, position in engine.vocabulary_.items():
        expected = tuple(int(row) for row in np.flatnonzero(dense[:, position] > 0))
        assert index.posting_list(term) == expected
        assert index.posting_list(position) == expected
        assert index.posting_length(term) == len(expected)
        assert term in index
    assert "the" not in index  # stopword column is not in the vocabulary at all
    assert engine.n_features_ not in index  # out-of-range term ids are never present


def test_postings_store_the_normalised_weight():
    engine = fitted_engine()
    index = InvertedIndex().fit(engine.matrix_, vocabulary=engine.vocabulary_)
    term = "b"
    position = engine.vocabulary_[term]
    rows, weights = index.posting_slice(term)
    assert list(rows) == list(index.posting_list(term))
    assert weights == pytest.approx(engine.to_dense()[list(rows), position], abs=TOLERANCE)


def test_candidates_are_exactly_the_positive_scoring_documents():
    engine = fitted_engine()
    index = InvertedIndex().fit(engine.matrix_, vocabulary=engine.vocabulary_)
    for query in ("a b", "y", "c", "a c y"):
        vector = engine.vectorize(query.split())
        candidates = index.candidates(vector.term_ids)
        rows, scores = index.score(vector)
        assert list(rows) == list(candidates)
        assert np.all(scores > 0.0)
        dense_scores = engine.cosine_scores(vector)
        assert np.allclose(scores, dense_scores[rows], rtol=0, atol=TOLERANCE)
        assert set(int(row) for row in candidates) == {
            int(row) for row in np.flatnonzero(dense_scores > 0.0)}


def test_index_scores_match_the_independent_reference():
    engine = fitted_engine()
    index = InvertedIndex().fit(engine.matrix_, vocabulary=engine.vocabulary_)
    query = ["b", "y", "zzz"]
    rows, scores = index.score(engine.vectorize(query))
    expected = reference_cosine(query, TOKEN_DOCS, STOPWORDS)
    for row, score in zip(rows, scores):
        assert score == pytest.approx(expected[int(row)], abs=TOLERANCE)


def test_empty_query_scores_nothing():
    engine = fitted_engine()
    index = InvertedIndex().fit(engine.matrix_, vocabulary=engine.vocabulary_)
    rows, scores = index.score(engine.vectorize(["zzz"]))
    assert rows.size == 0
    assert scores.size == 0
    assert np.allclose(index.cosine_scores_full(engine.vectorize(["zzz"])), 0.0)


def test_index_stats_and_roundtrip(tmp_path):
    engine = fitted_engine()
    index = InvertedIndex().fit(engine.matrix_, vocabulary=engine.vocabulary_)
    stats = index.stats()
    assert stats["n_documents"] == len(TOKEN_DOCS)
    assert stats["total_postings"] == int(engine.matrix_.nnz)
    index.save(tmp_path)
    restored = InvertedIndex.load(tmp_path, vocabulary=engine.vocabulary_)
    assert restored.n_documents == index.n_documents
    assert restored.n_features == index.n_features
    assert np.array_equal(restored.docs, index.docs)
    assert np.allclose(restored.weights, index.weights, rtol=0, atol=TOLERANCE)
    assert restored.posting_list("b") == index.posting_list("b")


# ------------------------------------------------------- transform new docs

def test_transform_documents_uses_the_frozen_vocabulary_and_matches_sklearn():
    engine = fitted_engine()
    sklearn = sklearn_vectorizer(stop_words=sorted(STOPWORDS)).fit(DOC_TEXTS)
    new_docs = [["a", "a", "the"], ["zzz", "b"], ["the"]]
    texts = [" ".join(tokens) for tokens in new_docs]
    actual = engine.transform_documents(new_docs)
    assert np.allclose(actual.toarray(), sklearn.transform(texts).toarray(), rtol=0, atol=TOLERANCE)
    assert actual.shape == (len(new_docs), engine.n_features_)
    row_norms = np.sqrt((actual.toarray() ** 2).sum(axis=1))
    assert np.allclose(row_norms[:2], 1.0, rtol=0, atol=TOLERANCE)
    assert np.all(actual.toarray()[2] == 0.0)


def test_transform_and_unnormalized_reconstruct_the_matrix():
    engine = fitted_engine()
    raw = engine.unnormalized()
    dense_raw = raw.toarray()
    dense = engine.to_dense()
    for row in range(engine.n_documents_):
        norm = np.sqrt((dense_raw[row] ** 2).sum())
        if norm > 0:
            assert dense[row] == pytest.approx(dense_raw[row] / norm, abs=TOLERANCE)


def test_term_stats_expose_df_and_idf():
    engine = fitted_engine()
    stats = engine.term_stats("b")
    assert stats == {"term": "b", "term_id": engine.vocabulary_["b"], "df": 3,
                     "idf": pytest.approx(engine.idf_[stats_term_id(engine, "b")], abs=TOLERANCE),
                     "document_frequency_ratio": pytest.approx(3 / len(TOKEN_DOCS))}
    assert engine.term_stats("unknown") == {"term": "unknown", "term_id": None, "df": 0, "idf": 0.0}
    assert engine.term(engine.vocabulary_["c"]) == "c"
    assert engine.terms() == ["a", "b", "c", "x", "y"]


def stats_term_id(engine, term):
    return engine.vocabulary_[term]


# ------------------------------------------------------------- persistence

def test_tfidf_save_load_roundtrip(tmp_path):
    engine = fitted_engine()
    engine.save(tmp_path)
    restored = TfidfEngine.load(tmp_path)
    assert restored.vocabulary_ == engine.vocabulary_
    assert np.array_equal(restored.df_, engine.df_)
    assert np.allclose(restored.idf_, engine.idf_, rtol=0, atol=TOLERANCE)
    assert np.allclose(restored.doc_norms_, engine.doc_norms_, rtol=0, atol=TOLERANCE)
    assert (restored.counts_ != engine.counts_).nnz == 0  # counts restored verbatim
    assert restored.counts_.nnz == engine.counts_.nnz
    assert np.allclose(restored.to_dense(), engine.to_dense(), rtol=0, atol=1e-15)
    assert restored.stopwords_ == engine.stopwords_
    assert restored.is_fitted
    # The loaded state must be usable for scoring, not only readable.
    vector = restored.vectorize(["a", "b"])
    assert np.allclose(restored.cosine_scores(vector), engine.cosine_scores(vector),
                       rtol=0, atol=TOLERANCE)


def test_load_restores_document_frequency_used_by_term_stats(tmp_path):
    """Regression: df_ used to be empty after load, breaking term_stats()."""
    engine = fitted_engine()
    engine.save(tmp_path)
    restored = TfidfEngine.load(tmp_path)
    for term in ("a", "b", "c", "x", "y"):
        assert restored.term_stats(term)["df"] == engine.term_stats(term)["df"]
        assert restored.term_stats(term)["idf"] == pytest.approx(
            engine.term_stats(term)["idf"], abs=TOLERANCE)
    assert restored.term_stats("b")["df"] == 3


def test_load_rejects_tampered_files(tmp_path):
    engine = fitted_engine()
    engine.save(tmp_path)
    counts = tmp_path / "tf_counts.npz"
    counts.write_bytes(counts.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="hash"):
        TfidfEngine.load(tmp_path)
    (tmp_path / "doc_norms.npy").unlink()
    with pytest.raises(ValueError, match="missing"):
        TfidfEngine.load(tmp_path)


def test_load_rejects_an_idf_table_that_contradicts_the_counts(tmp_path):
    engine = fitted_engine()
    engine.save(tmp_path)
    idf_path = tmp_path / "idf_weights.json"
    payload = json.loads(idf_path.read_text(encoding="utf-8"))
    payload["weights"][0] += 0.5
    idf_path.write_text(json.dumps(payload), encoding="utf-8")
    state_path = tmp_path / "tfidf_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["files"]["idf_weights.json"] = __import__("hashlib").sha256(
        idf_path.read_bytes()).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="IDF"):
        TfidfEngine.load(tmp_path)


# ------------------------------------------------------- randomised parity

@pytest.mark.parametrize("documents", [40, 120])
def test_randomised_parity_against_sklearn(documents):
    generator = random.Random(SEED)
    vocabulary = ["a", "b", "c", "x", "y", "z", "một", "hai"]
    docs = [[generator.choice(vocabulary) for _ in range(generator.randint(0, 12))]
            for _ in range(documents)]
    stopwords = {"the", "z"}
    docs = [[token for token in tokens if token != "z"] or ["z"] for tokens in docs]
    engine = TfidfEngine(stopwords=stopwords).fit(docs)
    sklearn = sklearn_vectorizer(stop_words=sorted(stopwords)).fit(
        [" ".join(tokens) for tokens in docs])
    assert engine.vocabulary_ == sklearn.vocabulary_
    assert np.allclose(engine.idf_, sklearn.idf_, rtol=0, atol=1e-15)
    dense = engine.to_dense()
    assert np.allclose(dense, sklearn.transform([" ".join(t) for t in docs]).toarray(),
                       rtol=0, atol=1e-12)
    index = InvertedIndex().fit(engine.matrix_, vocabulary=engine.vocabulary_)
    for _ in range(5):
        query = [generator.choice(vocabulary + ["oov"]) for _ in range(generator.randint(1, 5))]
        vector = engine.vectorize(query)
        rows, scores = index.score(vector)
        dense_scores = engine.cosine_scores(vector)
        assert np.allclose(scores, dense_scores[rows], rtol=0, atol=1e-12)
        expected = reference_cosine(query, docs, stopwords)
        for row, score in zip(rows, scores):
            assert score == pytest.approx(expected[int(row)], abs=1e-12)
        assert np.allclose(dense_scores, expected, rtol=0, atol=1e-12)
