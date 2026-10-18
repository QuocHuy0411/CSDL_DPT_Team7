"""Chỉ số đánh giá truy xuất văn bản (M2).

Toàn bộ công thức ở đây bám đúng `docs/api_contracts.md`, mục "Đánh giá và thực
nghiệm":

* ``P@K`` luôn lấy mẫu số đúng ``K``, kể cả khi hệ thống trả về ít hơn ``K``
  kết quả (thiếu kết quả là sai, không phải là "không tính").
* ``Recall@K`` lấy mẫu số là **tổng số tài liệu relevant đã được xác nhận trong
  qrels**, không phải số kết quả trả về.
* ``F1@K`` là trung bình điều hòa của precision và recall cùng cấp.
* ``AP`` tính trên **toàn bộ ranking có sẵn** chia cho tổng số relevant; ``MAP`` là
  trung bình macro của AP đó trên mọi query. Tên ``AP@K`` không được dùng để chỉ
  MAP toàn ranking, và ở đây cũng không có "MAP@K".
* ``NDCG@K`` dùng gain ``2^rel - 1``, giảm giám ``log2(rank + 1)`` với ``rank``
  bắt đầu từ 1, còn IDCG lấy từ qrels chứ không lấy từ lần chạy đó.
* Query không có tài liệu relevant nào được tính ``AP = 0`` và vẫn giữ trong bình
  quân macro, đồng thời được đếm riêng để báo cáo không giấu.

Module này thuần toán học, không đọc file và không phụ thuộc engine: có thể kiểm
thử bằng fixture, nhưng kết quả fixture tuyệt đối không được ghi vào kết quả
chính (chặn ở `evaluator/evaluate.py`).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: K mặc định của đồ án, có thể cấu hình lại từ CLI.
DEFAULT_K = 10

#: Ngưỡng coi một tài liệu là relevant.
RELEVANCE_THRESHOLD = 0

#: Tên metric dùng đúng nhất trong báo cáo và CSV.
PER_QUERY_COLUMNS = (
    "configuration",
    "query_id",
    "category",
    "query_type",
    "k",
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "average_precision",
    "average_precision_depth",
    "ndcg_at_k",
    "n_relevant",
    "n_returned",
    "zero_relevant",
)

SUMMARY_COLUMNS = (
    "configuration",
    "k",
    "map",
    "map_scope",
    "macro_precision_at_k",
    "macro_recall_at_k",
    "macro_f1_at_k",
    "macro_ndcg_at_k",
    "n_queries",
    "n_zero_relevant_queries",
    "zero_relevant_query_ids",
    "median_average_precision",
)

#: Ghi rõ phạm vi của MAP để không ai đọc nhầm thành AP@K.
MAP_SCOPE = "mean_average_precision_over_full_available_ranking"


def relevant_documents(judgments: Mapping[str, int],
                       threshold: int = RELEVANCE_THRESHOLD) -> frozenset:
    """Tập tài liệu relevant: ``relevance > threshold`` trong qrels đã xác nhận."""
    return frozenset(doc_id for doc_id, grade in judgments.items() if int(grade) > threshold)


def precision_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """P@K với mẫu số cố định bằng ``K``."""
    if k <= 0:
        raise ValueError("k must be positive")
    hits = sum(1 for doc_id in list(ranking)[:k] if doc_id in relevant)
    return hits / k


def recall_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Recall@K với mẫu số là tổng số relevant trong qrels."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in list(ranking)[:k] if doc_id in relevant)
    return hits / len(relevant)


def f1_at_k(precision: float, recall: float) -> float:
    """F1@K là trung bình điều hòa."""
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def average_precision(ranking: Sequence[str], relevant: Iterable[str]) -> float:
    """AP trên toàn bộ ranking có sẵn, mẫu số là tổng số relevant.

    ``AP = (1/R) * sum_{i=1..L} P@i * rel_i`` với ``L`` là độ dài ranking thực tế
    mà hệ thống trả về, không cắt theo K. ``R = 0`` thì trả ``0.0``.
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = 0.0
    total = 0.0
    for position, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            hits += 1.0
            total += hits / position
    return total / len(relevant)


def dcg(gains: Iterable[float]) -> float:
    """Discounted cumulative gain với gain ``g`` ở thứ hạng ``i`` (i bắt đầu từ 1)."""
    return float(sum(gain / math.log2(position + 1)
                     for position, gain in enumerate(gains, start=1)))


def ndcg_at_k(ranking: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    """NDCG@K: gain ``2^rel - 1``, IDCG lấy từ qrels, không lấy từ lần chạy."""
    if k <= 0:
        raise ValueError("k must be positive")
    ordered_relevances = [int(judgments.get(doc_id, 0)) for doc_id in list(ranking)[:k]]
    ideal_relevances = sorted((int(grade) for grade in judgments.values()), reverse=True)[:k]
    ideal = dcg(2.0 ** grade - 1.0 for grade in ideal_relevances)
    if ideal <= 0.0:
        return 0.0
    return dcg(2.0 ** grade - 1.0 for grade in ordered_relevances) / ideal


def query_metrics(ranking: Sequence[str], judgments: Mapping[str, int], k: int = DEFAULT_K, *,
                  threshold: int = RELEVANCE_THRESHOLD) -> Dict[str, object]:
    """Tính toàn bộ chỉ số của một query trên một ranking đã sắp xếp."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = relevant_documents(judgments, threshold)
    precision = precision_at_k(ranking, relevant, k)
    recall = recall_at_k(ranking, relevant, k)
    return {
        "k": k,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1_at_k(precision, recall),
        "average_precision": average_precision(ranking, relevant),
        "average_precision_depth": len(ranking),
        "ndcg_at_k": ndcg_at_k(ranking, judgments, k),
        "n_relevant": len(relevant),
        "n_returned": len(ranking),
        "zero_relevant": not relevant,
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def macro_average(rows: Sequence[Mapping[str, object]],
                  query_ids: Sequence[str] = ()) -> Dict[str, object]:
    """Bình quân macro trên **mọi** query, kể cả query không có relevant.

    Query không có relevant vẫn giữ giá trị 0 trong bình quân (không bị loại khỏi
    mẫu) và được đếm riêng để người đọc biết MAP bị kéo xuống bao nhiêu.
    """
    if not rows:
        raise ValueError("macro_average requires at least one query row")
    zero_relevant = [str(row.get("query_id", "")) for row in rows if row.get("zero_relevant")]
    if not query_ids:
        query_ids = [str(row.get("query_id", "")) for row in rows]
    return {
        "k": int(rows[0].get("k", DEFAULT_K)),
        "map": _mean([float(row["average_precision"]) for row in rows]),
        "map_scope": MAP_SCOPE,
        "macro_precision_at_k": _mean([float(row["precision_at_k"]) for row in rows]),
        "macro_recall_at_k": _mean([float(row["recall_at_k"]) for row in rows]),
        "macro_f1_at_k": _mean([float(row["f1_at_k"]) for row in rows]),
        "macro_ndcg_at_k": _mean([float(row["ndcg_at_k"]) for row in rows]),
        "median_average_precision": _median([float(row["average_precision"]) for row in rows]),
        "n_queries": len(rows),
        "n_zero_relevant_queries": len(zero_relevant),
        "zero_relevant_query_ids": ";".join(zero_relevant),
    }


def group_by_category(rows: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    """Bình quân macro theo category để phục vụ bảng 10 lĩnh vực."""
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("category", "unknown")), []).append(row)
    return {category: macro_average(members) for category, members in sorted(grouped.items())}


def summarize_pairs(pairs: Iterable[Tuple[str, str, float]]) -> Dict[str, Tuple[float, float]]:
    """Gom (mean, p95) từ một dãy cặp nhãn–giá trị, dùng cho benchmark."""
    grouped: Dict[str, List[Tuple[str, float]]] = {}
    for label, name, value in pairs:
        grouped.setdefault(label, []).append((name, float(value)))
    summary: Dict[str, Tuple[float, float]] = {}
    for label, entries in grouped.items():
        values = sorted(value for _, value in entries)
        mean = _mean(values)
        summary[label] = (mean, percentile(values, 0.95))
    return summary


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Phân vị gần nhất theo quy ước ``numpy.percentile`` (nội suy tuyến tính)."""
    if not sorted_values:
        return 0.0
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[int(position)])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def ranking_from_results(results: Sequence[Mapping[str, object]]) -> List[str]:
    """Lấy danh sách doc_id từ kết quả :meth:`SearchEngine.search`."""
    return [str(hit["doc_id"]) for hit in results]


def optional_depth(n_documents: Optional[int], requested: Optional[int]) -> int:
    """Độ sâu ranking dùng cho AP toàn ranking (mặc định: toàn bộ corpus)."""
    if requested:
        return int(requested)
    if n_documents:
        return int(n_documents)
    return 1000


__all__ = [
    "DEFAULT_K",
    "MAP_SCOPE",
    "PER_QUERY_COLUMNS",
    "RELEVANCE_THRESHOLD",
    "SUMMARY_COLUMNS",
    "average_precision",
    "dcg",
    "f1_at_k",
    "group_by_category",
    "macro_average",
    "ndcg_at_k",
    "optional_depth",
    "percentile",
    "precision_at_k",
    "query_metrics",
    "ranking_from_results",
    "recall_at_k",
    "relevant_documents",
    "summarize_pairs",
]
