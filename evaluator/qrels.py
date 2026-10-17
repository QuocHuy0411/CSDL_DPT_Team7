"""Nạp và kiểm tra qrels, cùng cổng chặn đánh giá chính thức.

Qrels của dự án có dạng::

    {
      "schema_version": 1,
      "corpus_sha256": "<sha256 của corpus đã khóa>",
      "status": "pending_human_annotation",
      "judgments": [
        {"query_id": "Q01", "doc_id": "D0123",
         "relevance": null, "status": "pending", "assessor": null}
      ]
    }

Quy tắc bắt buộc (xem `docs/api_contracts.md`):

* Chỉ nhãn ``status == "confirmed"`` do người gán mới hợp lệ cho đánh giá.
  Nhãn ``pending`` hoặc ``relevance == null`` **không** được coi là relevant và
  cũng không được dùng để "lấp chỗ trống".
* ``category`` của corpus và ``evidence_doc_ids`` của query chỉ hỗ trợ soạn
  query, không bao giờ được dùng thay cho nhãn relevance.
* Đánh giá **chính thức** bị từ chối nếu bất kỳ điều nào sau đây xảy ra:
  dưới 20 query, lệch SHA256 corpus giữa qrels/query/chỉ mục, còn nhãn
  pending, hoặc metadata tiền xử lý chưa ở trạng thái official.
* Fixture vẫn dùng được để kiểm thử, nhưng kết quả fixture không được ghi vào
  kết quả chính (chặn ở `evaluator/evaluate.py`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: Số query tối thiểu cho một lần đánh giá chính thức.
REQUIRED_OFFICIAL_QUERIES = 20

#: Phiên bản lược đồ qrels/queries mà bộ đánh giá này hiểu.
SUPPORTED_SCHEMA_VERSION = 1

#: Trạng thái qrels cho phép coi là đã gán xong.
QRELS_CONFIRMED_STATUS = "human_annotation_confirmed"

#: Trạng thái tiền xử lý duy nhất được phép xuất hiện trong kết quả chính thức.
OFFICIAL_PREPROCESSING_STATUS = "official"

#: Các mức relevance hợp lệ trong qrels của đồ án.
ALLOWED_RELEVANCE_GRADES = frozenset({0, 1, 2})

#: Bộ lọc tầng 1 tạm của M1: dùng được để tích hợp nhưng không phải Maximum
#: Matching chính thức, nên kết quả chỉ được coi là provisional.
FIXTURE_TIER1_FILTERS = frozenset({"engine.preprocessing:fixture_tier1"})

#: Baseline tokenizer âm thầm của M1, không phải tách từ Maximum Matching.
#: Dùng nó luôn chặn kết quả chính thức, kể cả khi metadata bị ghi sai là official.
DEVELOPMENT_TOKENIZERS = frozenset({"engine.preprocessing:baseline_tokenize"})


class OfficialEvaluationRefused(RuntimeError):
    """Ném ra khi một lần chạy được yêu cầu là chính thức nhưng dữ liệu chưa đủ."""


class QrelsQueryHashMismatch(RuntimeError):
    """Ném ra khi nhãn người được gán cho một bộ truy vấn khác bộ đang chạy.

    Đây là lỗi toàn vẹn dữ liệu chứ không phải chuyện trạng thái chính thức: nếu
    ``queries.json`` đã bị sửa mà ``query_id`` giữ nguyên, nhãn người sẽ bị gán
    nhầm cho nội dung khác. Vì vậy nó chặn **mọi** lần chạy, kể cả development, và
    bị từ chối trước cả khi ghi bất kỳ tệp kết quả nào.
    """


@dataclass(frozen=True)
class Judgment:
    """Một phán quyết của người gán cho một cặp (query, tài liệu)."""

    query_id: str
    doc_id: str
    relevance: Optional[int]
    status: str
    assessor: Optional[str] = None

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"

    @property
    def usable(self) -> bool:
        """Chỉ nhãn confirmed có điểm số mới dùng được cho đánh giá."""
        return self.confirmed and self.relevance is not None

    @property
    def relevant(self) -> bool:
        return self.usable and int(self.relevance) > 0


@dataclass(frozen=True)
class Qrels:
    """Toàn bộ qrels đã nạp, giữ nguyên trạng thái để kiểm chứng được."""

    schema_version: Optional[int]
    corpus_sha256: Optional[str]
    status: str
    judgments: Tuple[Judgment, ...] = ()
    source_path: Optional[str] = None
    queries_sha256: Optional[str] = None

    def query_ids(self) -> List[str]:
        seen: List[str] = []
        for judgment in self.judgments:
            if judgment.query_id not in seen:
                seen.append(judgment.query_id)
        return sorted(seen)

    def for_query(self, query_id: str, *, confirmed_only: bool = False) -> List[Judgment]:
        return [j for j in self.judgments
                if j.query_id == query_id and (j.confirmed or not confirmed_only)]

    def pending_judgments(self, query_ids: Optional[Iterable[str]] = None) -> List[Judgment]:
        scope = set(query_ids) if query_ids is not None else None
        return [j for j in self.judgments
                if not j.confirmed and (scope is None or j.query_id in scope)]

    def ungraded_confirmed(self, query_ids: Optional[Iterable[str]] = None) -> List[Judgment]:
        """Nhãn đã confirm nhưng điểm relevance còn null: dữ liệu mâu thuẫn."""
        scope = set(query_ids) if query_ids is not None else None
        return [j for j in self.judgments
                if j.confirmed and j.relevance is None and (scope is None or j.query_id in scope)]

    def relevance_map(self, query_id: str, *, confirmed_only: bool = True
                      ) -> Dict[str, int]:
        """Nhãn relevance dạng dict; mặc định chỉ lấy nhãn đã confirmed."""
        return {j.doc_id: int(j.relevance) for j in self.for_query(query_id)
                if (j.confirmed or not confirmed_only) and j.relevance is not None}

    def relevant_map(self, query_id: str) -> frozenset:
        return frozenset(doc_id for doc_id, grade in self.relevance_map(query_id).items()
                         if grade > 0)

    def assessor_names(self) -> List[str]:
        return sorted({j.assessor for j in self.judgments if j.assessor})

    def summary(self) -> Dict[str, object]:
        relevant_pairs = sum(1 for j in self.judgments if j.relevant)
        return {
            "source": self.source_path,
            "schema_version": self.schema_version,
            "corpus_sha256": self.corpus_sha256,
            "queries_sha256": self.queries_sha256,
            "status": self.status,
            "n_judgments": len(self.judgments),
            "n_confirmed": sum(1 for j in self.judgments if j.confirmed),
            "n_pending": sum(1 for j in self.judgments if not j.confirmed),
            "n_relevant_pairs": relevant_pairs,
            "n_query_ids": len(self.query_ids()),
            "assessors": self.assessor_names(),
        }


@dataclass(frozen=True)
class QuerySet:
    """`data/queries.json` đã nạp."""

    schema_version: Optional[int]
    corpus_sha256: Optional[str]
    queries: Tuple[Mapping[str, object], ...] = ()
    source_path: Optional[str] = None
    file_sha256: Optional[str] = None
    queries_sha256: Optional[str] = None

    def query_ids(self) -> List[str]:
        return [str(entry["query_id"]) for entry in self.queries]

    def text_of(self, query_id: str) -> str:
        for entry in self.queries:
            if str(entry["query_id"]) == query_id:
                return str(entry["text"])
        raise KeyError(query_id)

    def metadata_of(self, query_id: str) -> Mapping[str, object]:
        for entry in self.queries:
            if str(entry["query_id"]) == query_id:
                return entry
        raise KeyError(query_id)

    def category_of(self, query_id: str) -> str:
        return str(self.metadata_of(query_id).get("category", "unknown"))

    def types_of(self, query_id: str) -> str:
        types = self.metadata_of(query_id).get("types") or []
        return ";".join(str(value) for value in types)

    def summary(self) -> Dict[str, object]:
        return {
            "source": self.source_path,
            "schema_version": self.schema_version,
            "corpus_sha256": self.corpus_sha256,
            "queries_sha256": self.queries_sha256,
            "file_sha256": self.file_sha256,
            "n_queries": len(self.queries),
            "query_ids": self.query_ids(),
        }


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    detail: object = None

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}" + (f" ({self.detail})" if self.detail else "")


@dataclass(frozen=True)
class ValidationReport:
    """Kết quả kiểm tra điều kiện để coi một lần chạy là *chính thức*."""

    issues: Tuple[ValidationIssue, ...] = ()
    warnings: Tuple[ValidationIssue, ...] = ()
    checked: Mapping[str, object] = field(default_factory=dict)

    @property
    def official_allowed(self) -> bool:
        return not self.issues

    @property
    def codes(self) -> Tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def raise_if_blocked(self, action: str = "đánh giá chính thức") -> None:
        if self.official_allowed:
            return
        details = "\n".join(f"  - {issue}" for issue in self.issues)
        raise OfficialEvaluationRefused(
            f"Không thể coi kết quả là {action} vì các vấn đề sau:\n{details}")

    def as_dict(self) -> Dict[str, object]:
        return {
            "official_allowed": self.official_allowed,
            "issues": [{"code": issue.code, "message": issue.message, "detail": issue.detail}
                       for issue in self.issues],
            "warnings": [{"code": issue.code, "message": issue.message, "detail": issue.detail}
                         for issue in self.warnings],
            "checked": dict(self.checked),
        }


def load_qrels(path: str | Path) -> Qrels:
    """Nạp ``qrels.json``; không tự suy diễn thêm nhãn nào."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source} must contain a JSON object")
    judgments = []
    for entry in payload.get("judgments", []) or []:
        relevance = entry.get("relevance")
        if relevance is not None and type(relevance) is not int:
            raise ValueError("qrels relevance must be an integer or null, not bool/float/string")
        judgments.append(Judgment(
            query_id=str(entry["query_id"]),
            doc_id=str(entry["doc_id"]),
            relevance=None if relevance is None else int(relevance),
            status=str(entry.get("status", "pending")),
            assessor=entry.get("assessor"),
        ))
    return Qrels(
        schema_version=payload.get("schema_version"),
        corpus_sha256=payload.get("corpus_sha256"),
        status=str(payload.get("status", "unknown")),
        judgments=tuple(judgments),
        source_path=str(source),
        queries_sha256=payload.get("queries_sha256"),
    )


def load_queries(path: str | Path) -> QuerySet:
    """Nạp ``queries.json``."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source} must contain a JSON object")
    queries = payload.get("queries") or []
    missing = [q for q in queries if "query_id" not in q or "text" not in q]
    if missing:
        raise ValueError(f"{source} has {len(missing)} query objects without query_id/text")
    return QuerySet(
        schema_version=payload.get("schema_version"),
        corpus_sha256=payload.get("corpus_sha256"),
        queries=tuple(queries),
        source_path=str(source),
        file_sha256=file_sha256(source),
        queries_sha256=payload.get("queries_sha256"),
    )


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preprocessing_status(index_metadata: Optional[Mapping[str, object]]) -> Dict[str, object]:
    """Tóm tắt trạng thái tiền xử lý từ ``build_metadata.json`` của một chỉ mục.

    ``official`` chỉ đúng khi **cả hai** điều kiện cùng đúng: trạng thái ghi trong
    metadata là ``official`` *và* cờ ``official_preprocessing``/``m1_verified``
    cũng là true. Hai nguồn này lệch nhau tức là metadata không nhất quán và phải
    chặn, không được chọn bừa một bên.
    """
    if not index_metadata:
        return {"known": False, "status": None, "official": False,
                "status_is_official": False, "flag_is_official": False, "consistent": False,
                "tokenizer": None, "tier1_filter": None, "tier1_enabled": None,
                "m1_verified": None, "fingerprint": None, "corpus_sha256": None,
                "reason": "missing index metadata"}
    preprocessing = index_metadata.get("preprocessing")
    preprocessing = preprocessing if isinstance(preprocessing, Mapping) else {}
    status = index_metadata.get("preprocessing_status") or preprocessing.get("status")
    m1_verified = preprocessing.get("m1_verified")
    official_preprocessing = index_metadata.get("official_preprocessing")
    flag_is_official = official_preprocessing is True and m1_verified is True
    status_is_official = status == OFFICIAL_PREPROCESSING_STATUS
    return {
        "known": True,
        "status": status,
        "status_is_official": status_is_official,
        "flag_is_official": flag_is_official,
        "official_preprocessing": official_preprocessing,
        "consistent": status_is_official and flag_is_official,
        "official": status_is_official and flag_is_official,
        "tokenizer": preprocessing.get("tokenizer") or index_metadata.get("tokenizer"),
        "tier1_filter": preprocessing.get("tier1_filter") or index_metadata.get("tier1_filter"),
        "tier1_enabled": preprocessing.get("tier1_enabled"),
        "m1_verified": m1_verified,
        "fingerprint": preprocessing.get("fingerprint"),
        "corpus_sha256": index_metadata.get("corpus_sha256"),
    }


def judgment_issues(qrels: Qrels, known_query_ids: Sequence[str]) -> List[ValidationIssue]:
    """Các lỗi cấu trúc của qrels, kiểm tra trên toàn bộ file chứ không riêng phạm vi.

    * Trùng cặp ``(query_id, doc_id)`` – pooling hợp lệ không sinh trùng.
    * ``relevance`` ngoài tập ``{0, 1, 2}``.
    * Nhãn ``confirmed`` mà không có người gán (``assessor`` rỗng).
    * ``query_id`` trong qrels không tồn tại trong ``queries.json``.
    """
    issues: List[ValidationIssue] = []
    known = set(known_query_ids)

    seen: Dict[Tuple[str, str], int] = {}
    for judgment in qrels.judgments:
        key = (judgment.query_id, judgment.doc_id)
        seen[key] = seen.get(key, 0) + 1
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    if duplicates:
        issues.append(ValidationIssue(
            "duplicate_judgments",
            f"{len(duplicates)} cặp (query_id, doc_id) bị gán trùng trong qrels",
            [f"{query_id}/{doc_id}" for query_id, doc_id in duplicates[:5]]))

    invalid = [f"{j.query_id}/{j.doc_id}={j.relevance}" for j in qrels.judgments
               if j.relevance is not None and int(j.relevance) not in ALLOWED_RELEVANCE_GRADES]
    if invalid:
        issues.append(ValidationIssue(
            "invalid_relevance_grade",
            f"{len(invalid)} phán quyết có relevance ngoài tập "
            f"{sorted(ALLOWED_RELEVANCE_GRADES)}", invalid[:5]))

    anonymous = [f"{j.query_id}/{j.doc_id}" for j in qrels.judgments
                 if j.confirmed and not (j.assessor or "").strip()]
    if anonymous:
        issues.append(ValidationIssue(
            "confirmed_without_assessor",
            f"{len(anonymous)} phán quyết confirmed nhưng không ghi người gán (assessor)",
            anonymous[:5]))

    if known:
        unknown = sorted({j.query_id for j in qrels.judgments} - known)
        if unknown:
            issues.append(ValidationIssue(
                "unknown_query_id_in_qrels",
                "qrels chứa query_id không có trong queries.json", unknown[:10]))
    return issues


def validate_official_run(qrels: Qrels, queries: QuerySet, *,
                          corpus_sha256: Optional[str],
                          index_metadata: Optional[Mapping[str, object]] = None,
                          evaluated_query_ids: Optional[Sequence[str]] = None,
                          required_queries: int = REQUIRED_OFFICIAL_QUERIES) -> ValidationReport:
    """Kiểm tra mọi điều kiện chặn một lần đánh giá chính thức.

    Trả về :class:`ValidationReport`; ``official_allowed`` là ``True`` chỉ khi
    không có vấn đề chặn nào. Cảnh báo không chặn nhưng luôn được ghi kèm vào kết
    quả để người đọc không bỏ sót.
    """
    issues: List[ValidationIssue] = []
    warnings: List[ValidationIssue] = []
    scope = list(evaluated_query_ids) if evaluated_query_ids else queries.query_ids()
    status = preprocessing_status(index_metadata)

    for name, version in (("qrels", qrels.schema_version), ("queries", queries.schema_version)):
        if version != SUPPORTED_SCHEMA_VERSION:
            issues.append(ValidationIssue(
                f"{name}_schema_version_unsupported",
                f"{name}.schema_version={version!r}, bộ đánh giá chỉ hỗ trợ "
                f"{SUPPORTED_SCHEMA_VERSION}", version))

    if not corpus_sha256:
        issues.append(ValidationIssue(
            "expected_corpus_hash_missing",
            "Không xác định được SHA256 corpus cần đánh giá nên không thể chứng minh "
            "qrels, queries và chỉ mục cùng trỏ tới một corpus"))
    else:
        for name, value in (("qrels", qrels.corpus_sha256), ("queries", queries.corpus_sha256)):
            if value is None:
                issues.append(ValidationIssue(f"{name}_corpus_hash_missing",
                                              f"{name} không khai báo corpus_sha256"))
            elif value != corpus_sha256:
                issues.append(ValidationIssue(
                    f"{name}_corpus_hash_mismatch",
                    f"SHA256 corpus trong {name} khác corpus đang đánh giá",
                    {"qrels_or_queries": value, "expected": corpus_sha256}))
        index_hash = status.get("corpus_sha256")
        if index_hash is None:
            issues.append(ValidationIssue(
                "index_corpus_hash_missing",
                "build_metadata.json của chỉ mục không ghi corpus_sha256"))
        elif index_hash != corpus_sha256:
            issues.append(ValidationIssue(
                "index_corpus_hash_mismatch",
                "SHA256 corpus của chỉ mục khác corpus đang đánh giá",
                {"index": index_hash, "expected": corpus_sha256}))

    if len(queries.query_ids()) < required_queries:
        issues.append(ValidationIssue(
            "insufficient_queries",
            f"Bộ query chỉ có {len(queries.query_ids())} query, cần tối thiểu {required_queries}",
            {"actual": len(queries.query_ids()), "required": required_queries}))

    actual_queries_sha256 = queries.file_sha256 or queries.queries_sha256
    if not qrels.judgments:
        warnings.append(ValidationIssue(
            "qrels_empty", "qrels không có phán quyết nào nên mọi chỉ số đều là 0"))
    elif qrels.queries_sha256 is None:
        issues.append(ValidationIssue(
            "qrels_queries_hash_missing",
            "qrels không ghi queries_sha256 nên không chứng minh được nhãn được gán cho "
            "đúng bộ truy vấn đang chạy; nếu nội dung truy vấn đổi mà giữ nguyên "
            "query_id thì nhãn người sẽ bị dùng sai"))
    elif not actual_queries_sha256:
        issues.append(ValidationIssue(
            "queries_hash_unavailable",
            "Không tính được SHA256 của queries.json để đối chiếu với qrels"))
    elif qrels.queries_sha256 != actual_queries_sha256:
        issues.append(ValidationIssue(
            "queries_hash_mismatch",
            "SHA256 của queries.json khác queries_sha256 ghi trong qrels; nhãn người "
            "được gán cho một bộ truy vấn khác",
            {"actual": actual_queries_sha256, "qrels": qrels.queries_sha256}))

    known_ids = set(queries.query_ids())
    if known_ids:
        unknown = [query_id for query_id in scope if query_id not in known_ids]
        if unknown:
            issues.append(ValidationIssue("unknown_query_id",
                                          "Truy vấn đánh giá không có trong queries.json",
                                          unknown))

    if qrels.status != QRELS_CONFIRMED_STATUS:
        issues.append(ValidationIssue(
            "qrels_status_not_final",
            f"qrels đang ở trạng thái '{qrels.status}', chưa phải '{QRELS_CONFIRMED_STATUS}'",
            qrels.status))

    issues.extend(judgment_issues(qrels, queries.query_ids()))

    pending = qrels.pending_judgments(scope)
    if pending:
        issues.append(ValidationIssue(
            "pending_judgments",
            f"Còn {len(pending)} phán quyết chưa confirmed trong phạm vi đánh giá",
            {"count": len(pending),
             "example": [f"{j.query_id}/{j.doc_id}" for j in pending[:5]]}))

    ungraded = qrels.ungraded_confirmed(scope)
    if ungraded:
        issues.append(ValidationIssue(
            "confirmed_without_relevance",
            f"{len(ungraded)} phán quyết confirmed nhưng relevance vẫn null",
            [f"{j.query_id}/{j.doc_id}" for j in ungraded[:5]]))

    if scope:
        without = [query_id for query_id in scope if not qrels.relevance_map(query_id)]
        if without:
            issues.append(ValidationIssue(
                "query_without_confirmed_judgments",
                "Query chưa có nhãn confirmed nào nên không thể tính Recall/MAP",
                without))

    if not status["official"]:
        code = ("preprocessing_metadata_inconsistent" if status["known"]
                and (status["status_is_official"] or status["flag_is_official"])
                else "preprocessing_not_official")
        issues.append(ValidationIssue(
            code,
            "Metadata tiền xử lý chưa nhất quán ở trạng thái official nên "
            "kết quả chỉ là provisional",
            {"preprocessing_status": status["status"],
             "official_preprocessing": status.get("official_preprocessing"),
             "m1_verified": status["m1_verified"], "tokenizer": status["tokenizer"],
             "tier1_filter": status["tier1_filter"], "tier1_enabled": status["tier1_enabled"]}))

    if status["tokenizer"] in DEVELOPMENT_TOKENIZERS:
        issues.append(ValidationIssue(
            "development_tokenizer",
            "Tokenizer baseline của M1 không phải tách từ Maximum Matching chính thức",
            {"tokenizer": status["tokenizer"],
             "note": "Chặn tuyệt đối, kể cả khi metadata ghi nhầm là official"}))
    if status["tier1_filter"] in FIXTURE_TIER1_FILTERS:
        issues.append(ValidationIssue(
            "fixture_tier1_filter",
            "C3/C4 đang dùng fixture tầng 1, không phải Maximum Matching chính thức của M1",
            {"tier1_filter": status["tier1_filter"], "tier1_enabled": status["tier1_enabled"],
             "note": "Chặn tuyệt đối, kể cả khi metadata ghi nhầm là official"}))
    if not status.get("fingerprint"):
        warnings.append(ValidationIssue(
            "preprocessing_fingerprint_missing",
            "Chỉ mục không ghi fingerprint tiền xử lý nên không kiểm chứng được tính tái lập"))

    if index_metadata and index_metadata.get("configuration") == "C4":
        if not index_metadata.get("stoplist_count"):
            warnings.append(ValidationIssue(
                "c4_stoplist_not_recorded", "C4 không ghi số phần tử stoplist trong metadata"))
        if not index_metadata.get("tdv_seconds"):
            warnings.append(ValidationIssue(
                "c4_tdv_time_not_recorded", "C4 không ghi thời gian chạy TDV"))

    return ValidationReport(
        issues=tuple(issues),
        warnings=tuple(warnings),
        checked={
            "corpus_sha256": corpus_sha256,
            "actual_queries_sha256": actual_queries_sha256,
            "qrels_queries_sha256": qrels.queries_sha256,
            "qrels": qrels.summary(),
            "queries": queries.summary(),
            "evaluated_query_ids": scope,
            "required_queries": required_queries,
            "preprocessing": status,
            "allowed_relevance_grades": sorted(ALLOWED_RELEVANCE_GRADES),
        },
    )


def assert_queries_hash(qrels: Qrels, queries: QuerySet) -> Optional[str]:
    """Chặn nếu qrels không khớp với bộ truy vấn đang chạy.

    Trả về SHA256 thực tế của ``queries.json`` khi hợp lệ. Qrels rỗng (chưa có
    phán quyết nào) thì không có gì để đối chiếu nên được bỏ qua.
    """
    if not qrels.judgments:
        return queries.file_sha256 or queries.queries_sha256
    actual = queries.file_sha256 or queries.queries_sha256
    if qrels.queries_sha256 is None:
        raise QrelsQueryHashMismatch(
            "qrels không ghi queries_sha256 nên không chứng minh được nhãn người được "
            "gán cho đúng bộ truy vấn đang chạy. Sửa nội dung truy vấn nhưng giữ "
            "nguyên query_id sẽ khiến nhãn bị dùng sai.")
    if not actual:
        raise QrelsQueryHashMismatch("Không tính được SHA256 của queries.json để đối chiếu")
    if qrels.queries_sha256 != actual:
        raise QrelsQueryHashMismatch(
            f"qrels.queries_sha256={qrels.queries_sha256} khác SHA256 thực tế của "
            f"queries.json={actual}. Nhãn người được gán cho một bộ truy vấn khác; "
            f"không được đo chỉ số nào trên cặp này.")
    return actual


def development_run_report(report: ValidationReport, reason: str) -> Dict[str, object]:
    """Ghi rõ một lần chạy *không chính thức* và lý do cụ thể."""
    payload = report.as_dict()
    payload["official"] = False
    payload["mode"] = "development"
    payload["mode_reason"] = reason
    return payload


__all__ = [
    "ALLOWED_RELEVANCE_GRADES",
    "DEVELOPMENT_TOKENIZERS",
    "FIXTURE_TIER1_FILTERS",
    "Judgment",
    "OFFICIAL_PREPROCESSING_STATUS",
    "OfficialEvaluationRefused",
    "QRELS_CONFIRMED_STATUS",
    "Qrels",
    "QuerySet",
    "REQUIRED_OFFICIAL_QUERIES",
    "SUPPORTED_SCHEMA_VERSION",
    "ValidationIssue",
    "ValidationReport",
    "development_run_report",
    "file_sha256",
    "judgment_issues",
    "load_qrels",
    "load_queries",
    "preprocessing_status",
    "validate_official_run",
]
