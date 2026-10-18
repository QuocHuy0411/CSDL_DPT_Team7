"""Public ``SearchEngine`` contract: results, ordering, explain and persistence.

The fixtures use callable tokenizers defined at module level so that the
tokenizer fingerprint survives a ``save`` / ``load`` round trip. Nothing here
reads ``data/raw``: documents are built in memory and every artefact is written
to pytest's ``tmp_path``.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

import numpy as np
import pytest

from engine.indexer.search_engine import (DOCUMENTS_FILE, MANIFEST_FILE, SearchEngine,
                                           tokenizer_descriptor)

TOLERANCE = 1e-12
STOPWORDS = {"và", "của"}

DOCUMENTS = [
    {"doc_id": "D3", "title": "Máy tính", "content": "bóng đá và máy tính", "category": "Tin"},
    {"doc_id": "D1", "title": "Bóng đá", "content": "bóng bóng đá đá", "category": "Thể thao"},
    {"doc_id": "D2", "title": "Chung", "content": "bóng đá và chung máy", "category": "Tin"},
    {"doc_id": "D4", "title": "Rỗng", "content": "và của chỉ", "category": "Khác"},
]


def regex_tokenizer(text: str) -> list[str]:
    """Callable token fixture: keeps Vietnamese syllables and digits."""
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def ascii_tokenizer(text: str) -> list[str]:
    return text.lower().split()


def other_tokenizer(text: str) -> list[str]:
    return sorted(text.lower().split())


def fitted(stopwords=STOPWORDS, tokenizer=regex_tokenizer, documents=None) -> SearchEngine:
    return SearchEngine(tokenizer=tokenizer, stopwords=stopwords).fit(
        list(documents if documents is not None else DOCUMENTS))


def reference_cosine(query_tokens, documents, stopwords) -> dict[str, float]:
    """Cosine recomputed with plain Python, independent of the engine."""
    cleaned = [[token for token in regex_tokenizer(document["content"])
                if token not in stopwords] for document in documents]
    vocabulary = sorted({token for tokens in cleaned for token in tokens})
    n_documents = len(cleaned)
    frequency = {term: sum(1 for tokens in cleaned if term in set(tokens)) for term in vocabulary}
    idf = {term: math.log((n_documents + 1) / (frequency[term] + 1)) + 1.0 for term in vocabulary}
    query = Counter(token for token in query_tokens if token not in stopwords)
    query_weights = {term: (1.0 + math.log(query[term])) * idf[term]
                     for term in query if term in vocabulary}
    norm = math.sqrt(sum(value * value for value in query_weights.values()))
    scores = {}
    for document, tokens in zip(documents, cleaned):
        counts = Counter(tokens)
        weights = {term: (1.0 + math.log(counts[term])) * idf[term] for term in counts}
        length = math.sqrt(sum(value * value for value in weights.values()))
        if norm == 0.0 or length == 0.0:
            scores[document["doc_id"]] = 0.0
            continue
        scores[document["doc_id"]] = sum(
            (query_weights[term] / norm) * (value / length)
            for term, value in weights.items() if term in query_weights)
    return scores


# ----------------------------------------------------------------- basic API

def test_fit_returns_self_and_indexes_content_only():
    engine = SearchEngine(tokenizer=regex_tokenizer, stopwords=STOPWORDS)
    assert engine.fit(DOCUMENTS) is engine
    assert engine.is_fitted
    assert engine.n_documents == len(DOCUMENTS)
    # The title is already inside the content and must not be counted twice.
    assert engine.tfidf.term_stats("đá")["df"] == 3
    assert engine.tfidf.term_stats("tin")["df"] == 0
    assert engine.metadata["retrieval"]["n_documents"] == len(DOCUMENTS)
    assert engine.metadata["retrieval"]["math"]["idf"] == "ln((N + 1) / (df + 1)) + 1"


def test_search_result_contract_and_ordering():
    engine = fitted()
    results = engine.search("đá", top_k=10)
    assert [set(result) for result in results] == [
        {"doc_id", "title", "category", "score", "snippet"}] * len(results)
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0.0 for score in scores)
    assert all(result["doc_id"] in {document["doc_id"] for document in DOCUMENTS}
               for result in results)


def test_scores_match_an_independent_python_cosine():
    engine = fitted()
    for query in ("đá", "bóng đá", "máy", "đá zzz"):
        expected = reference_cosine(regex_tokenizer(query), DOCUMENTS, STOPWORDS)
        results = {result["doc_id"]: result["score"] for result in engine.search(query, top_k=99)}
        for doc_id, score in results.items():
            assert score == pytest.approx(expected[doc_id], abs=1e-12)
        assert set(results) == {doc_id for doc_id, score in expected.items() if score > 0}


def test_top_k_is_respected():
    engine = fitted()
    assert len(engine.search("đá", top_k=1)) == 1
    assert len(engine.search("đá", top_k=2)) == 2
    assert engine.search("đá", top_k=0) == []
    assert len(engine.search("đá", top_k=99)) == 3
    with pytest.raises(ValueError):
        engine.search("đá", top_k=-1)
    with pytest.raises(TypeError):
        engine.search("đá", top_k=1.5)


def test_ties_are_broken_by_ascending_doc_id():
    documents = [{"doc_id": doc_id, "title": doc_id, "content": "chung bóng", "category": "x"}
                 for doc_id in ("D9", "D2", "D10", "D1")]
    engine = fitted(documents=documents)
    results = engine.search("bóng", top_k=10)
    assert [result["doc_id"] for result in results] == ["D1", "D10", "D2", "D9"]
    assert len({result["score"] for result in results}) == 1


def test_zero_vector_document_is_never_returned():
    engine = fitted()
    results = engine.search("đá", top_k=10)
    assert "D4" not in {result["doc_id"] for result in results}
    explanation = engine.explain("đá", "D4")
    assert explanation["score"] == 0.0
    # The missing term is still reported, with an explicit zero contribution.
    assert [(row["term"], row["doc_tf"], row["contribution"])
            for row in explanation["contributions"]] == [("đá", 0, 0.0)]


@pytest.mark.parametrize("query", ["", "   ", "và của", "zzzzzz", "và zzzzzz"])
def test_empty_stopword_and_oov_queries_return_no_results(query):
    engine = fitted()
    assert engine.search(query) == []


def test_query_accepts_string_or_token_list():
    engine = fitted()
    assert engine.search("bóng đá") == engine.search(regex_tokenizer("bóng đá"))
    assert engine.search(["bóng", "đá"]) == engine.search("bóng đá")
    assert engine.search(["bóng", "bóng", "đá"]) == engine.search("bóng bóng đá")
    with pytest.raises(TypeError):
        engine.search(42)
    with pytest.raises(TypeError):
        engine.search(["bóng", 7])


def test_stopwords_are_applied_symmetrically_to_corpus_and_query():
    engine = fitted()
    assert engine.tfidf.term_stats("và")["df"] == 0
    assert engine.search("và") == []
    # A term that is only ever a stopword cannot be rescued by a query.
    documents = [{"doc_id": "D1", "title": "t", "content": "chung bóng", "category": "x"},
                 {"doc_id": "D2", "title": "t", "content": "chung máy", "category": "x"}]
    engine = fitted(documents=documents, stopwords={"chung"})
    assert engine.search("chung") == []
    assert [result["doc_id"] for result in engine.search("bóng")] == ["D1"]


def test_zero_matrix_row_never_becomes_a_candidate():
    engine = fitted()
    vector = engine.tfidf.vectorize(["đá"])
    rows, scores = engine.index.score(vector)
    assert 3 not in set(int(row) for row in rows)
    assert all(score > 0 for score in scores)


# ------------------------------------------------------------------- explain

def test_explain_contributions_sum_to_the_score():
    engine = fitted()
    for query in ("đá", "bóng đá", "máy tính"):
        expected = reference_cosine(regex_tokenizer(query), DOCUMENTS, STOPWORDS)
        for hit in engine.search(query, top_k=99):
            explanation = engine.explain(query, hit["doc_id"])
            total = sum(row["contribution"] for row in explanation["contributions"])
            assert total == pytest.approx(explanation["score"], abs=1e-12)
            assert explanation["score"] == pytest.approx(hit["score"], abs=TOLERANCE)
            assert explanation["score"] == pytest.approx(expected[hit["doc_id"]], abs=1e-12)


def test_explain_rows_carry_the_documented_fields():
    engine = fitted()
    explanation = engine.explain("bóng đá", "D1")
    assert explanation["doc_id"] == "D1"
    assert set(explanation) >= {"doc_id", "score", "contributions"}
    assert [row["term"] for row in explanation["contributions"]] == ["bóng", "đá"]
    for row in explanation["contributions"]:
        assert set(row) == {"term", "query_tf", "doc_tf", "idf", "query_weight", "doc_weight",
                            "contribution"}
        assert row["doc_tf"] == 2  # "bóng bóng đá đá"
        assert row["query_tf"] == 1
        assert row["contribution"] == pytest.approx(row["query_weight"] * row["doc_weight"],
                                                    abs=TOLERANCE)
        assert row["query_weight"] == pytest.approx(
            row["idf"] / math.sqrt(row["idf"] ** 2 * 2), abs=TOLERANCE)
        assert row["doc_weight"] <= 1.0 + TOLERANCE


def test_explain_reports_missing_terms_and_oov_terms():
    engine = fitted()
    explanation = engine.explain("bóng zzzz và", "D1")
    assert [row["term"] for row in explanation["contributions"]] == ["bóng"]
    assert explanation["oov_terms"] == ["zzzz"]
    assert explanation["dropped_terms"] == ["và"]
    assert explanation["query_terms"] == ["bóng"]


def test_explain_handles_a_document_without_any_query_term():
    engine = fitted()
    explanation = engine.explain("máy", "D1")
    assert explanation["score"] == 0.0
    assert [row["doc_tf"] for row in explanation["contributions"]] == [0]
    assert explanation["contributions"][0]["contribution"] == 0.0


def test_explain_rejects_unknown_documents_and_unfitted_engines():
    engine = fitted()
    with pytest.raises(KeyError):
        engine.explain("đá", "D404")
    with pytest.raises(RuntimeError):
        SearchEngine(tokenizer=regex_tokenizer).explain("đá", "D1")


def test_score_helper_matches_explain():
    engine = fitted()
    assert engine.score("bóng", "D1") == engine.explain("bóng", "D1")["score"]


# ---------------------------------------------------------------- validation

def test_fit_rejects_misaligned_or_invalid_input():
    engine = SearchEngine(tokenizer=regex_tokenizer)
    with pytest.raises(ValueError, match="tokenized_docs"):
        engine.fit(DOCUMENTS, tokenized_docs=[["bóng"]])
    with pytest.raises(ValueError, match="doc_id"):
        engine.fit([{"content": "bóng"}])
    with pytest.raises(ValueError, match="unique"):
        engine.fit([{"doc_id": "D1", "content": "bóng"}, {"doc_id": "D1", "content": "đá"}])
    with pytest.raises(TypeError):
        engine.fit({"D1": {"content": "bóng"}})
    with pytest.raises(TypeError):
        engine.fit(DOCUMENTS, tokenized_docs=[[1, 2], ["a"], ["b"], ["c"]])
    with pytest.raises(TypeError):
        SearchEngine(tokenizer="not callable")


def test_search_requires_fit():
    with pytest.raises(RuntimeError):
        SearchEngine(tokenizer=regex_tokenizer).search("đá")


def test_inputs_are_never_mutated():
    documents = [dict(document) for document in DOCUMENTS]
    tokens = [[token for token in regex_tokenizer(document["content"])] for document in documents]
    documents_before = json.dumps(documents, ensure_ascii=False, sort_keys=True)
    tokens_before = json.dumps(tokens, ensure_ascii=False, sort_keys=True)
    engine = fitted(documents=documents)
    engine.fit(documents, tokenized_docs=tokens)
    engine.search("bóng đá")
    assert json.dumps(documents, ensure_ascii=False, sort_keys=True) == documents_before
    assert json.dumps(tokens, ensure_ascii=False, sort_keys=True) == tokens_before


def test_tokenizer_is_used_for_queries_but_not_for_caller_supplied_tokens():
    calls = []

    def counting_tokenizer(text: str) -> list[str]:
        calls.append(text)
        return regex_tokenizer(text)

    engine = SearchEngine(tokenizer=counting_tokenizer, stopwords=STOPWORDS)
    engine.fit(DOCUMENTS, tokenized_docs=[["bóng", "đá"], ["đá"], ["máy"], ["và"]])
    assert calls == []
    engine.search("bóng")
    assert calls == ["bóng"]


# --------------------------------------------------------------- persistence

def test_save_load_round_trip_preserves_search_and_explain(tmp_path):
    engine = fitted()
    manifest = engine.save(tmp_path)
    assert (tmp_path / MANIFEST_FILE).is_file()
    assert manifest["n_documents"] == len(DOCUMENTS)
    assert manifest["math"]["similarity"] == "cosine"

    restored = SearchEngine.load(tmp_path)
    for query in ("đá", "bóng đá", "máy tính", "và"):
        assert restored.search(query, top_k=10) == engine.search(query, top_k=10)
    assert restored.explain("bóng đá", "D1") == engine.explain("bóng đá", "D1")
    assert restored.metadata == engine.metadata
    assert restored.stopwords_ == engine.stopwords_
    assert restored.tfidf.vocabulary_ == engine.tfidf.vocabulary_
    assert np.allclose(restored.tfidf.to_dense(), engine.tfidf.to_dense(), rtol=0, atol=1e-15)
    assert restored.tfidf.df_.tolist() == engine.tfidf.df_.tolist()
    assert restored.index.posting_list("đá") == engine.index.posting_list("đá")
    assert restored.snippet_width == engine.snippet_width


def test_save_load_keeps_user_metadata_verbatim(tmp_path):
    metadata = {"configuration": "C3", "corpus_sha256": "abc", "preprocessing": {"status": "dev"}}
    engine = SearchEngine(tokenizer=regex_tokenizer, stopwords=STOPWORDS, metadata=metadata).fit(
        DOCUMENTS)
    engine.save(tmp_path)
    restored = SearchEngine.load(tmp_path)
    assert restored.metadata["configuration"] == "C3"
    assert restored.metadata["corpus_sha256"] == "abc"
    assert restored.metadata["preprocessing"] == {"status": "dev"}
    assert restored.metadata["retrieval"]["stopword_count"] == len(STOPWORDS)


def test_load_accepts_the_same_tokenizer_explicitly(tmp_path):
    engine = fitted(tokenizer=ascii_tokenizer)
    engine.save(tmp_path)
    restored = SearchEngine.load(tmp_path, tokenizer=ascii_tokenizer)
    assert restored.search("bong") == engine.search("bong")


def test_load_refuses_a_different_tokenizer(tmp_path):
    engine = fitted(tokenizer=regex_tokenizer)
    engine.save(tmp_path)
    with pytest.raises(ValueError, match="tokenizer"):
        SearchEngine.load(tmp_path, tokenizer=ascii_tokenizer)
    with pytest.raises(ValueError, match="tokenizer"):
        SearchEngine.load(tmp_path, tokenizer=other_tokenizer)


def test_load_refuses_a_tokenizer_it_cannot_fingerprint(tmp_path):
    """A REPL-only callable has no comparable source: refuse it, never guess."""
    engine = fitted(tokenizer=regex_tokenizer)
    engine.save(tmp_path)
    with pytest.raises(ValueError, match="(?i)tokenizer"):
        SearchEngine.load(tmp_path, tokenizer=lambda text: text.split())
    # ...and the same rule holds for a callable whose source cannot be located at all.
    descriptor = tokenizer_descriptor(eval(compile("lambda text: text.split()", "<stdin>", "eval")))
    assert descriptor["fingerprint"] is None


def test_load_detects_a_tampered_index_file(tmp_path):
    engine = fitted()
    engine.save(tmp_path)
    counts = tmp_path / "tf_counts.npz"
    counts.write_bytes(counts.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="hash"):
        SearchEngine.load(tmp_path)


def test_load_detects_a_tampered_document_list(tmp_path):
    engine = fitted()
    engine.save(tmp_path)
    path = tmp_path / DOCUMENTS_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["documents"][0]["title"] = "Sai"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        SearchEngine.load(tmp_path)


def test_load_rejects_an_unknown_schema_version(tmp_path):
    engine = fitted()
    engine.save(tmp_path)
    path = tmp_path / MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 99
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        SearchEngine.load(tmp_path)


def test_load_resolves_the_whitespace_baseline_without_arguments(tmp_path):
    engine = SearchEngine(stopwords=STOPWORDS).fit(DOCUMENTS)
    engine.save(tmp_path)
    restored = SearchEngine.load(tmp_path)
    assert restored.tokenizer("a  b\tc") == ["a", "b", "c"]
    assert restored.search("bóng đá") == engine.search("bóng đá")


def test_load_requires_an_explicit_tokenizer_for_callable_instances(tmp_path):
    from engine.preprocessing import Preprocessor

    processor = Preprocessor(tier1_enabled=True)
    metadata = {"configuration": "C3", "preprocessing": processor.metadata()}
    documents = [{"doc_id": "D1", "title": "Một", "content": "và chung bóng đá", "category": "f"},
                 {"doc_id": "D2", "title": "Hai", "content": "chung máy tính", "category": "f"}]
    engine = SearchEngine(tokenizer=processor, stopwords={"chung"}, metadata=metadata)
    engine.fit(documents)
    engine.save(tmp_path)
    same = Preprocessor.from_metadata(metadata["preprocessing"])
    assert SearchEngine.load(tmp_path, tokenizer=same).search("bóng") == engine.search("bóng")
    with pytest.raises(ValueError, match="explicitly"):
        SearchEngine.load(tmp_path)
    with pytest.raises(ValueError, match="tokenizer"):
        SearchEngine.load(tmp_path, tokenizer=Preprocessor(tier1_enabled=False))


def test_saved_manifest_records_the_configuration_provenance(tmp_path):
    engine = SearchEngine(tokenizer=regex_tokenizer, stopwords=STOPWORDS,
                          metadata={"configuration": "C1"}).fit(DOCUMENTS)
    manifest = engine.save(tmp_path)
    stored = json.loads((tmp_path / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert stored["metadata"]["configuration"] == "C1"
    assert stored["stopwords"]["count"] == len(STOPWORDS)
    assert set(stored["stopwords"]["terms"]) == STOPWORDS
    assert stored["tokenizer"]["label"].endswith("regex_tokenizer")
    assert set(stored["files"]) >= {"tf_counts.npz", "idf_weights.json", "inverted_index.npz",
                                    "doc_metadata.json", "manifest.json"} - {MANIFEST_FILE}
    assert manifest["n_features"] == engine.tfidf.n_features_


# ------------------------------------------------------------------ snippets

def test_snippets_are_deterministic_and_bounded(tmp_path):
    long_content = " ".join(f"từ{index}" for index in range(200))
    documents = [{"doc_id": "D1", "title": "t", "content": long_content, "category": "x"}]
    engine = fitted(documents=documents, stopwords=set())
    assert engine.snippet_width == 240
    snippet = engine.search("từ1", top_k=1)[0]["snippet"]
    assert len(snippet) <= 241
    assert snippet.endswith("…")
    assert "\n" not in snippet
    short = [{"doc_id": "D1", "title": "t", "content": "ngắn gọn", "category": "x"}]
    short_engine = fitted(documents=short, stopwords=set())
    assert short_engine.search("ngắn", top_k=1)[0]["snippet"] == "ngắn gọn"

    engine.save(tmp_path)
    assert SearchEngine.load(tmp_path).search("từ1", top_k=1)[0]["snippet"] == snippet


def test_tfidf_state_is_exposed_for_tdv_and_evaluation():
    """`.tfidf` is the documented hand-off for TDV/evaluator code."""
    engine = fitted()
    tfidf = engine.tfidf
    assert set(tfidf.vocabulary_) == {"bóng", "đá", "máy", "tính", "chung", "chỉ"}
    assert tfidf.n_documents_ == len(DOCUMENTS)
    assert tfidf.n_features_ == len(tfidf.vocabulary_)
    assert tfidf.matrix_.format == "csr"
    assert tfidf.matrix_.dtype == np.float64
    assert tfidf.idf_.shape == (tfidf.n_features_,)
    assert tfidf.df_.shape == (tfidf.n_features_,)
    assert tfidf.doc_norms_.shape == (tfidf.n_documents_,)
    assert isinstance(engine.index.posting_list("đá"), tuple)
    assert engine.index.stats()["total_postings"] == tfidf.matrix_.nnz
