"""Term Discrimination Value (Salton) tầng 2 cho bộ lọc stopword theo miền.

Định nghĩa dùng trong module này (khớp với `docs/api_contracts.md`):

1. Mật độ không gian tài liệu là **mật độ cosine pairwise chính xác**, loại trừ
   self-pair, trên toàn bộ corpus đã chọn (một lần duy nhất, không loại nối tiếp):

   .. math::

       D(X) = \\frac{1}{N(N-1)} \\sum_{i \\neq j} \\cos(x_i, x_j)

   Mẫu số cố định là :math:`N(N-1)` (số cặp có thứ tự). Với :math:`N < 2` thì
   :math:`D = 0`. Vector tất cả bằng không thì :math:`\\cos = 0`.

2. Trọng số sơ bộ dùng để đo TDV là TF-IDF: sublinear TF
   :math:`1+\\ln f_{t,d}`, IDF làm mượt :math:`\\ln\\frac{N+1}{df(t)+1}+1`
   và chuẩn L2 **trước** khi lọc TDV. IDF sơ bộ được **giữ nguyên** khi loại một
   chiều để đo TDV, tránh trộn hai hiệu ứng (đổi trọng số và đổi hình học).

3. TDV của một từ là hiệu mật độ khi xóa chiều đó:

   .. math::  TDV(t) = D_{-t} - D_{base}

   Khi xóa chiều :math:`t` thì **mỗi tài liệu phải được chuẩn hóa L2 lại** và
   tổng vector (centroid) phải được cập nhật; cài đặt ở đây làm đúng hai việc đó
   (xem `TDVEngine.fit`).

4. Ngưỡng lọc: ``score < -tolerance`` với ``tolerance = 1e-12`` mặc định, đây là
   dung sai số học chứ không phải ngưỡng nghiệp vụ; nó được ghi lại trong
   ``metadata_``. Không ép đường cong hình chữ U và không suy ra từ đó rằng
   cấu hình nào "tốt nhất": chất lượng phải đo bằng evaluator.

Sau khi chọn được stoplist, TF-IDF chính thức của C4 được dựng lại từ đầu với
stoplist đó (việc này do ``scripts/build_indexes.py`` thực hiện), không dùng lại
ma trận đo TDV.

Hai đường tính được cung cấp và phải khớp nhau trên fixture nhỏ:

* :meth:`TDVEngine.fit` — đường nhanh, dùng đẳng thức đóng dựa trên
  :math:`\\|X_{-t}u\\|` (không dựng lại ma trận cosine cho từng từ);
* :func:`reference_scores` — đường trực tiếp, xóa cột, chuẩn hóa lại từng dòng,
  nhân ma trận cosine đầy đủ, dùng độc lập để kiểm chứng.

Ví dụ::

    >>> engine = TDVEngine().fit([["a", "b"], ["b", "c"], ["c", "a"]])
    >>> engine.metadata_["n_documents"]
    3
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy import sparse

#: Dung sai số học cho ngưỡng lọc stopword tầng 2. Không phải siêu tham số IR.
DEFAULT_TOLERANCE = 1e-12

#: Phiên bản công thức, ghi vào metadata để truy vết kết quả.
FORMULA_VERSION = "exact_pairwise_cosine_density_tdv/one-shot/1.0"


class TDVEngine:
    """Tính TDV Salton chính xác trên toàn bộ corpus đã token hoá.

    Sau :meth:`fit`::

        engine.scores_     # dict[str, float] mọi từ trong từ vựng
        engine.stopwords_  # set[str] gồm các từ có score < -tolerance
        engine.metadata_   # dict JSON-serializable để lưu cạnh chỉ mục
    """

    def __init__(self, tolerance: float = DEFAULT_TOLERANCE):
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.tolerance = float(tolerance)
        self.scores_: Dict[str, float] = {}
        self.stopwords_: set = set()
        self.metadata_: Dict[str, object] = {}

    def fit(self, tokenized_docs: Sequence[Sequence[str]] | Sequence[str]) -> "TDVEngine":
        """Tính TDV một lần trên toàn bộ ``tokenized_docs``.

        ``tokenized_docs`` là danh sách token của từng tài liệu, cùng thứ tự với
        corpus. Nếu phần tử là ``str`` thì được tách theo khoảng trắng (chỉ phục
        vụ tiện ích/fixture; tiền xử lý chính thức do M1 cung cấp).
        """
        documents = _as_token_lists(tokenized_docs)
        matrix, vocabulary, df, idf = _build_preliminary_tfidf(documents)
        base = _base_density(matrix)
        scores = _tdv_scores(matrix, base)
        self.scores_ = {term: float(scores[column]) for term, column in vocabulary.items()}
        self.stopwords_ = {term for term, value in self.scores_.items() if value < -self.tolerance}
        self.metadata_ = {
            "formula": FORMULA_VERSION,
            "density": "mean pairwise cosine over ordered distinct pairs",
            "density_denominator": "N*(N-1)",
            "self_pairs_included": False,
            "zero_vector_cosine": 0.0,
            "one_shot": True,
            "successive_removal": False,
            "renormalize_after_dimension_removal": True,
            "centroid_update": "exact via X^T u after re-normalization",
            "preliminary_weighting": {
                "tf": "1+ln(f)",
                "idf": "ln((N+1)/(df+1))+1",
                "normalization": "L2",
                "idf_policy": "preliminary_idf_frozen_while_measuring_tdv",
            },
            "filter_rule": "score < -tolerance",
            "tolerance": self.tolerance,
            "tolerance_meaning": "numerical tolerance only, not an IR hyper-parameter",
            "n_documents": int(matrix.shape[0]),
            "n_vocabulary": int(matrix.shape[1]),
            "n_scored_terms": len(self.scores_),
            "n_filtered_terms": len(self.stopwords_),
            "base_density": float(base),
            "density_min": min(self.scores_.values()) if self.scores_ else 0.0,
            "density_max": max(self.scores_.values()) if self.scores_ else 0.0,
            "document_frequency_min": int(df.min()) if df.size else 0,
            "document_frequency_max": int(df.max()) if df.size else 0,
            "idf_min": float(idf.min()) if idf.size else 0.0,
            "idf_max": float(idf.max()) if idf.size else 0.0,
            "filtered_terms": sorted(self.stopwords_),
        }
        return self

    def reference_scores(self, tokenized_docs: Sequence[Sequence[str]] | Sequence[str]
                         ) -> Dict[str, float]:
        """Xem :func:`reference_scores`."""
        return reference_scores(tokenized_docs, tolerance=self.tolerance)


def reference_scores(tokenized_docs: Sequence[Sequence[str]] | Sequence[str],
                     tolerance: float = DEFAULT_TOLERANCE) -> Dict[str, float]:
    """TDV tính trực tiếp, độc lập với đường nhanh, dùng để kiểm chứng.

    Cách làm cố ý "chậm nhưng hiển nhiên đúng": với mỗi từ, xóa hẳn cột khỏi ma
    trận TF-IDF, chuẩn hóa L2 lại từng tài liệu, nhân ma trận cosine đầy đủ rồi
    lấy tổng ngoài đường chéo. Chỉ dùng cho fixture nhỏ vì chi phí là O(V·N²).
    """
    documents = _as_token_lists(tokenized_docs)
    dense, vocabulary, _ = _dense_preliminary_tfidf(documents)
    n_documents = dense.shape[0]
    denominator = float(n_documents * (n_documents - 1))
    if denominator <= 0:
        return {term: 0.0 for term in vocabulary}
    gram = dense @ dense.T
    base = (gram.sum() - np.trace(gram)) / denominator
    scores: Dict[str, float] = {}
    for column, term in enumerate(vocabulary):
        reduced = dense.copy()
        reduced[:, column] = 0.0
        norms = np.sqrt((reduced * reduced).sum(axis=1))
        normalized = np.zeros_like(reduced)
        usable = norms > 0
        normalized[usable] = reduced[usable] / norms[usable][:, None]
        gram_reduced = normalized @ normalized.T
        density = (gram_reduced.sum() - np.trace(gram_reduced)) / denominator
        scores[term] = float(density - base)
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    return scores


def _as_token_lists(tokenized_docs: Iterable) -> List[List[str]]:
    documents: List[List[str]] = []
    for item in tokenized_docs:
        if isinstance(item, str):
            documents.append(item.split())
        else:
            documents.append([str(token) for token in item])
    return documents


def _build_preliminary_tfidf(documents: Sequence[Sequence[str]]):
    """Dựng CSR TF-IDF sơ bộ đã chuẩn L2, kèm từ vựng, df và idf."""
    n_documents = len(documents)
    document_frequency: Counter = Counter()
    for tokens in documents:
        document_frequency.update(set(tokens))
    vocabulary: Dict[str, int] = {}
    rows: List[int] = []
    columns: List[int] = []
    frequencies: List[float] = []
    for row, tokens in enumerate(documents):
        for term, count in Counter(tokens).items():
            if term not in vocabulary:
                vocabulary[term] = len(vocabulary)
            rows.append(row)
            columns.append(vocabulary[term])
            frequencies.append(float(count))
    n_vocabulary = len(vocabulary)
    if n_documents == 0 or n_vocabulary == 0:
        return (sparse.csr_matrix((n_documents, n_vocabulary), dtype=np.float64),
                vocabulary, np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64))
    counts = np.asarray(frequencies, dtype=np.float64)
    df = np.zeros(n_vocabulary, dtype=np.float64)
    for term, column in vocabulary.items():
        df[column] = document_frequency[term]
    idf = np.log((n_documents + 1.0) / (df + 1.0)) + 1.0
    weights = (1.0 + np.log(counts)) * idf[np.asarray(columns, dtype=np.int64)]
    matrix = sparse.csr_matrix(
        (weights, (np.asarray(rows), np.asarray(columns))),
        shape=(n_documents, n_vocabulary), dtype=np.float64)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    scale = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    matrix = sparse.diags(scale) @ matrix
    matrix = sparse.csr_matrix(matrix)
    return matrix, vocabulary, df, idf


def _dense_preliminary_tfidf(documents: Sequence[Sequence[str]]):
    """Bản dày của :func:`_build_preliminary_tfidf`, dùng cho đường tham chiếu."""
    n_documents = len(documents)
    terms = sorted({term for tokens in documents for term in tokens})
    vocabulary = {term: column for column, term in enumerate(terms)}
    dense = np.zeros((n_documents, len(terms)), dtype=np.float64)
    for row, tokens in enumerate(documents):
        for term, count in Counter(tokens).items():
            dense[row, vocabulary[term]] = float(count)
    if not terms or n_documents == 0:
        return dense, terms, np.zeros(0, dtype=np.float64)
    frequency = (dense > 0).sum(axis=0).astype(np.float64)
    idf = np.log((n_documents + 1.0) / (frequency + 1.0)) + 1.0
    present = dense > 0
    weighted = np.zeros_like(dense)
    weighted[present] = (1.0 + np.log(dense[present])) * np.broadcast_to(idf, dense.shape)[present]
    dense = weighted
    norms = np.sqrt((dense * dense).sum(axis=1))
    usable = norms > 0
    dense[usable] /= norms[usable][:, None]
    return dense, terms, idf


def _base_density(matrix) -> float:
    """D_base chính xác: (1^T S 1 - tr S) / N(N-1), S = X X^T do X chuẩn L2."""
    n_documents = matrix.shape[0]
    denominator = float(n_documents * (n_documents - 1))
    if denominator <= 0:
        return 0.0
    centroid_sum = np.asarray(matrix.sum(axis=0)).ravel()
    row_square_sums = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    return float(centroid_sum @ centroid_sum - row_square_sums.sum()) / denominator


def _tdv_scores(matrix, base: float) -> np.ndarray:
    """TDV(t) = D_{-t} - D_base cho mọi cột, bằng đẳng thức đóng chính xác.

    Với :math:`X` đã chuẩn L2 theo dòng và cột :math:`t` có giá trị :math:`v_i`
    trên dòng :math:`i`, xóa cột đó cho vectơ :math:`x_i - v_i e_t` có chuẩn
    :math:`n_i = \\sqrt{\\|x_i\\|^2 - v_i^2}`. Với :math:`u_i = 1/n_i` (và
    :math:`u_i = 0` khi vectơ sau khi xóa bằng không) thì

    .. math::

        \\sum_{i \\neq j} u_i u_j (S_{ij} - v_i v_j)
          = \\|X^T u\\|^2 - \\sum_i u_i^2 S_{ii} - \\big(\\langle u, v\\rangle^2
            - \\sum_i u_i^2 v_i^2\\big)

    nên chỉ cần một phép tích lũy theo posting list của từ, không dựng lại ma
    trận cosine cho từng từ. Mọi tài liệu đều được chuẩn hóa lại (nhiều hơn 1)
    và centroid được cập nhật qua :math:`X^T u`.
    """
    n_documents, n_vocabulary = matrix.shape
    scores = np.zeros(n_vocabulary, dtype=np.float64)
    denominator = float(n_documents * (n_documents - 1))
    if denominator <= 0:
        return scores
    csr = sparse.csr_matrix(matrix)
    csc = csr.tocsc()
    indptr, indices, values = csr.indptr, csr.indices, csr.data
    row_ids = np.repeat(np.arange(n_documents, dtype=np.int64), np.diff(indptr))
    row_square_sums = np.bincount(row_ids, weights=values * values, minlength=n_documents)
    total_row_square_sums = float(row_square_sums.sum())
    centroid_sum = np.asarray(csr.sum(axis=0)).ravel()
    for column in range(n_vocabulary):
        start, end = csc.indptr[column], csc.indptr[column + 1]
        if end <= start:
            continue
        rows = csc.indices[start:end]
        column_values = csc.data[start:end]
        lengths = (indptr[rows + 1] - indptr[rows]).astype(np.int64)
        begins = indptr[rows].astype(np.int64)
        offsets = np.zeros(rows.size, dtype=np.int64)
        if rows.size > 1:
            np.cumsum(lengths[:-1], out=offsets[1:])
        total = int(lengths.sum())
        gather = (np.arange(total, dtype=np.int64)
                  - np.repeat(offsets, lengths) + np.repeat(begins, lengths))
        reduced_square = np.maximum(row_square_sums[rows] - column_values * column_values, 0.0)
        usable = reduced_square > 0
        scale = np.zeros(rows.size, dtype=np.float64)
        scale[usable] = 1.0 / np.sqrt(reduced_square[usable])
        delta = scale - 1.0
        weights = values[gather] * np.repeat(delta, lengths)
        updated_centroid = centroid_sum + np.bincount(indices[gather], weights=weights,
                                                      minlength=n_vocabulary)
        diagonal = (total_row_square_sums - float(row_square_sums[rows].sum())
                    + float(np.sum(scale[usable] ** 2 * row_square_sums[rows][usable])))
        projected = float(scale @ column_values)
        gram_term = float(updated_centroid @ updated_centroid) - diagonal
        overlap_term = projected * projected - float(np.sum((scale * column_values) ** 2))
        scores[column] = (gram_term - overlap_term) / denominator - base
    return scores


def stopwords_from_scores(scores: Mapping[str, float],
                          tolerance: float = DEFAULT_TOLERANCE) -> set:
    """Áp đúng ngưỡng ``score < -tolerance`` lên một dict điểm TDV."""
    return {term for term, value in scores.items() if float(value) < -tolerance}


def density_of_matrix(matrix) -> float:
    """Mật độ cosine pairwise chính xác của một ma trận đã chuẩn L2 theo dòng."""
    return _base_density(matrix)


__all__ = [
    "DEFAULT_TOLERANCE",
    "FORMULA_VERSION",
    "TDVEngine",
    "density_of_matrix",
    "reference_scores",
    "stopwords_from_scores",
]
