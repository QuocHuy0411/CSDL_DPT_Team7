"""Kiểm chứng TDV Salton: công thức, tái chuẩn hóa, one-shot và ngưỡng lọc.

Nguyên tắc của bộ test này: mọi khẳng định về giá trị TDV đều dựa trên một
trong hai nguồn độc lập — (1) số học viết tay trong docstring của fixture, hoặc
(2) :func:`engine.indexer.tdv_engine.reference_scores` dựng lại ma trận cosine
đầy đủ cho từng từ. Ngoài ra có một biến thể "không tái chuẩn hóa" được cài ở
đây để chứng minh implementation thật sự tái chuẩn hóa chứ không vô tình khớp với
cách làm sai.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.indexer.tdv_engine import (  # noqa: E402
    DEFAULT_TOLERANCE,
    TDVEngine,
    _build_preliminary_tfidf,
    _dense_preliminary_tfidf,
    reference_scores,
    stopwords_from_scores,
)

MATCH_TOLERANCE = 1e-9


# ---------------------------------------------------------------- fixture chuẩn

#: Số học viết tay cho 2 tài liệu.
#:
#: N = 2, df(a) = df(c) = 1 nên idf = ln(3/2) + 1 = 1.40546511,
#: df(b) = 2 nên idf(b) = ln(1) + 1 = 1. Sau chuẩn L2:
#: d1 = (0.81480, 0.57974, 0), d2 = (0, 0.57974, 0.81480).
#: D_base = cos(d1, d2) = 0.57974^2 = 0.33609693.
#: Bỏ "b": d1 -> (1,0,0), d2 -> (0,0,1) nên D_{-b} = 0  -> TDV(b) = -0.33609693.
#: Bỏ "a": d1 -> (0,1,0), d2 giữ nguyên nên D_{-a} = 0.57974 -> TDV(a) = +0.24364174.
PAIR_FIXTURE = [["a", "b"], ["b", "c"]]
PAIR_EXPECTED = {"a": 0.24364174426140817, "b": -0.33609692727625706, "c": 0.2436417442614086}

#: Fixture có tài liệu chỉ chứa đúng một từ: sau khi bỏ chính từ đó vectơ bằng
#: không nên cosine với mọi tài liệu khác bằng 0 (mẫu số vẫn là N(N-1)).
SINGLE_TOKEN_FIXTURE = [["a"], ["a", "b"], ["b", "c"], ["c", "a"]]

#: Fixture có tần suất lặp khác nhau để sublinear TF và tái chuẩn hóa cùng ảnh hưởng.
REPEATED_FIXTURE = [["a", "a", "a", "b"], ["b", "b", "c"], ["c", "c", "a"], ["a", "b", "c"]]

#: Có cả tài liệu rỗng (vectơ tất cả bằng không) và từ xuất hiện ở mọi tài liệu.
EDGE_FIXTURE = [[], ["a", "b"], [], ["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]]


def _naive_scores_without_renormalization(tokenized_docs):
    """TDV sai lệch: bỏ chiều nhưng KHÔNG chuẩn hóa lại tài liệu.

    Chỉ tồn tại để chứng minh :class:`TDVEngine` khác biệt thật sự với cách làm
    mà đề cương cảnh báo (bỏ chiều mà quên cập nhật trọng tâm/chuẩn).
    """
    dense, terms, _ = _dense_preliminary_tfidf(tokenized_docs)
    n_documents = dense.shape[0]
    denominator = float(n_documents * (n_documents - 1))
    if denominator <= 0:
        return {term: 0.0 for term in terms}
    gram = dense @ dense.T
    base = (gram.sum() - np.trace(gram)) / denominator
    scores = {}
    for column, term in enumerate(terms):
        reduced = dense.copy()
        reduced[:, column] = 0.0
        norms = np.sqrt((dense * dense).sum(axis=1))
        usable = norms > 0
        naive = np.zeros_like(dense)
        naive[usable] = reduced[usable] / norms[usable][:, None]
        gram_reduced = naive @ naive.T
        scores[term] = float((gram_reduced.sum() - np.trace(gram_reduced)) / denominator - base)
    return scores


def _sequential_scores(tokenized_docs):
    """TDV loại nối tiếp, cố ý sai so với hợp đồng one-shot."""
    remaining = [list(document) for document in tokenized_docs]
    scores = {}
    for term in sorted({token for document in tokenized_docs for token in document}):
        engine = TDVEngine().fit(remaining)
        if term in engine.scores_:
            scores[term] = engine.scores_[term]
        remaining = [[token for token in document if token != term] for document in remaining]
    return scores


# ------------------------------------------------------------ khớp tham chiếu


def test_hand_computed_pair_fixture_pins_the_formula():
    engine = TDVEngine().fit(PAIR_FIXTURE)
    for term, expected in PAIR_EXPECTED.items():
        assert engine.scores_[term] == pytest.approx(expected, abs=MATCH_TOLERANCE)


@pytest.mark.parametrize("fixture", [PAIR_FIXTURE, SINGLE_TOKEN_FIXTURE, REPEATED_FIXTURE,
                                     EDGE_FIXTURE, [["a"], ["a"]], [["a"], ["b"]],
                                     [[], ["a", "b"], []]], ids=lambda f: f"n{len(f)}")
def test_fast_path_matches_direct_reference(fixture):
    engine = TDVEngine().fit(fixture)
    reference = reference_scores(fixture)
    assert set(engine.scores_) == set(reference)
    for term, expected in reference.items():
        assert engine.scores_[term] == pytest.approx(expected, abs=MATCH_TOLERANCE)


@pytest.mark.parametrize("size", [2, 3, 5, 8, 13])
@pytest.mark.parametrize("density", [0.15, 0.35, 0.6, 0.85])
def test_fast_path_matches_reference_on_random_fixtures(size, density):
    rng = random.Random(2026 + size * 100 + int(density * 100))
    vocabulary = [f"t{i}" for i in range(9)]
    fixture = [[term for term in vocabulary if rng.random() < density] for _ in range(size)]
    engine = TDVEngine().fit(fixture)
    reference = reference_scores(fixture)
    assert set(engine.scores_) == set(reference)
    for term, expected in reference.items():
        assert engine.scores_[term] == pytest.approx(expected, abs=MATCH_TOLERANCE)


def test_repeated_tokens_fixture_really_needs_renormalization():
    engine = TDVEngine().fit(REPEATED_FIXTURE)
    naive = _naive_scores_without_renormalization(REPEATED_FIXTURE)
    reference = reference_scores(REPEATED_FIXTURE)
    differences = [abs(engine.scores_[term] - naive[term]) for term in engine.scores_]
    assert max(differences) > 1e-3, "fixture không còn phân biệt được cách làm đúng và cách làm sai"
    for term, expected in reference.items():
        assert naive[term] != pytest.approx(expected, abs=MATCH_TOLERANCE)
        assert engine.scores_[term] == pytest.approx(expected, abs=MATCH_TOLERANCE)


def test_single_token_document_becomes_zero_vector_and_cosine_is_zero():
    engine = TDVEngine().fit(SINGLE_TOKEN_FIXTURE)
    reference = reference_scores(SINGLE_TOKEN_FIXTURE)
    for term, expected in reference.items():
        assert engine.scores_[term] == pytest.approx(expected, abs=MATCH_TOLERANCE)
    assert engine.metadata_["zero_vector_cosine"] == 0.0
    assert engine.metadata_["density_denominator"] == "N*(N-1)"


# ------------------------------------------------------------------ one-shot


def test_scoring_is_one_shot_not_successive_removal():
    fixture = [["a", "b", "c"], ["a", "b"], ["b", "c"], ["c", "d"], ["a", "d"]]
    one_shot = TDVEngine().fit(fixture).scores_
    sequential = _sequential_scores(fixture)
    assert set(one_shot) == set(sequential)
    assert any(abs(one_shot[term] - sequential[term]) > 1e-6 for term in one_shot), \
        "fixture này phải cho kết quả khác giữa one-shot và loại nối tiếp"
    for term, expected in reference_scores(fixture).items():
        assert one_shot[term] == pytest.approx(expected, abs=MATCH_TOLERANCE)


def test_metadata_declares_one_shot_and_renormalization():
    metadata = TDVEngine().fit(REPEATED_FIXTURE).metadata_
    assert metadata["one_shot"] is True
    assert metadata["successive_removal"] is False
    assert metadata["renormalize_after_dimension_removal"] is True
    assert metadata["centroid_update"].startswith("exact")
    assert metadata["self_pairs_included"] is False


# ------------------------------------------------------------ trường hợp biên


@pytest.mark.parametrize("fixture", [[], [["a"]], [[], []]])
def test_fewer_than_two_documents_gives_zero_density(fixture):
    engine = TDVEngine().fit(fixture)
    assert engine.metadata_["base_density"] == 0.0
    assert set(engine.stopwords_) == set()
    assert all(value == 0.0 for value in engine.scores_.values())
    assert reference_scores(fixture) == {term: 0.0 for term in engine.scores_}


def test_identical_documents_have_unit_density():
    engine = TDVEngine().fit([["a", "b"], ["a", "b"], ["a", "b"]])
    assert engine.metadata_["base_density"] == pytest.approx(1.0, abs=1e-12)
    assert engine.scores_["a"] == pytest.approx(0.0, abs=MATCH_TOLERANCE)


def test_empty_documents_contribute_nothing_to_density():
    engine = TDVEngine().fit([[], ["a", "b"], []])
    assert engine.metadata_["base_density"] == pytest.approx(0.0, abs=1e-12)
    for term, expected in reference_scores([[], ["a", "b"], []]).items():
        assert engine.scores_[term] == pytest.approx(expected, abs=MATCH_TOLERANCE)


def test_empty_corpus_produces_empty_but_valid_result():
    engine = TDVEngine().fit([])
    assert engine.scores_ == {}
    assert engine.stopwords_ == set()
    assert engine.metadata_["n_documents"] == 0
    assert engine.metadata_["n_filtered_terms"] == 0


# -------------------------------------------------------------- ngưỡng lọc


def test_tolerance_is_a_numerical_epsilon_not_a_quality_threshold():
    engine = TDVEngine()
    assert engine.tolerance == DEFAULT_TOLERANCE
    assert DEFAULT_TOLERANCE == 1e-12
    with pytest.raises(ValueError):
        TDVEngine(tolerance=-1e-6)


def test_filter_rule_is_strictly_below_negative_tolerance():
    scores = {"keep_zero": 0.0, "keep_small": -5e-13, "drop_mid": -1e-11, "drop_big": -0.5}
    filtered = stopwords_from_scores(scores)
    assert filtered == {"drop_mid", "drop_big"}


def test_stopwords_match_the_declared_tolerance():
    engine = TDVEngine(tolerance=1e-12).fit(REPEATED_FIXTURE)
    expected = {term for term, value in engine.scores_.items() if value < -1e-12}
    assert engine.stopwords_ == expected
    assert engine.metadata_["filter_rule"] == "score < -tolerance"
    assert engine.metadata_["tolerance"] == 1e-12
    assert engine.metadata_["n_filtered_terms"] == len(expected)


def test_custom_tolerance_changes_only_the_filter():
    documents = [["a", "a", "b", "b", "c"], ["a", "b", "c"], ["a", "b"], ["a", "c"]]
    default = TDVEngine().fit(documents)
    wide = TDVEngine(tolerance=0.5).fit(documents)
    assert default.scores_ == wide.scores_
    assert wide.stopwords_ <= default.stopwords_


def test_metadata_is_json_serializable_and_describes_preliminary_weighting():
    import json

    engine = TDVEngine().fit(REPEATED_FIXTURE)
    payload = json.dumps(engine.metadata_, ensure_ascii=False)
    restored = json.loads(payload)
    assert restored["preliminary_weighting"]["tf"] == "1+ln(f)"
    assert restored["preliminary_weighting"]["idf"] == "ln((N+1)/(df+1))+1"
    assert restored["preliminary_weighting"]["normalization"] == "L2"
    assert restored["preliminary_weighting"]["idf_policy"] == (
        "preliminary_idf_frozen_while_measuring_tdv")
    assert restored["n_documents"] == 4
    assert restored["n_scored_terms"] == len(engine.scores_)


# -------------------------------------------------------------- tính lặp lại


def test_fit_is_deterministic():
    first = TDVEngine().fit(REPEATED_FIXTURE)
    second = TDVEngine().fit(REPEATED_FIXTURE)
    assert first.scores_ == second.scores_
    assert first.stopwords_ == second.stopwords_


def test_preliminary_matrix_uses_real_document_frequency_and_sublinear_tf():
    matrix, vocabulary, df, idf = _build_preliminary_tfidf(REPEATED_FIXTURE)
    assert vocabulary == {"a": 0, "b": 1, "c": 2}
    assert list(df) == [3, 3, 3]
    assert idf[vocabulary["a"]] == pytest.approx(np.log(5 / 4) + 1)
    expected = (1 + np.log(3)) * idf[vocabulary["a"]]
    assert matrix[0, vocabulary["a"]] / np.linalg.norm(matrix[0].toarray()) == pytest.approx(
        expected / np.sqrt(expected ** 2 + (1 + np.log(1)) ** 2 * idf[vocabulary["b"]] ** 2))
    rows = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    nonzero = rows[rows > 0]
    assert np.allclose(nonzero, 1.0)


def test_single_character_and_underscored_tokens_are_supported():
    documents = [["a"], ["a_1", "b"], ["b", "c_1"], ["trí_tuệ", "nhân_tạo"]]
    engine = TDVEngine().fit(documents)
    reference = reference_scores(documents)
    assert set(engine.scores_) == {"a", "a_1", "b", "c_1", "trí_tuệ", "nhân_tạo"}
    for term, expected in reference.items():
        assert engine.scores_[term] == pytest.approx(expected, abs=MATCH_TOLERANCE)


def test_duplicate_tokens_inside_one_document_do_not_inflate_document_frequency():
    documents = [["x", "x", "x", "y"], ["y", "z"], ["z", "w"]]
    matrix, vocabulary, df, _ = _build_preliminary_tfidf(documents)
    assert df[vocabulary["x"]] == 1
    assert df[vocabulary["y"]] == 2
