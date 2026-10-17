"""Chạy đánh giá ablation C1–C4 và ghi kết quả có dấu vết kiểm chứng được.

Cách dùng (chạy như module từ thư mục gốc repo)::

    # Chính thức: chỉ chạy được khi qrels đã confirmed và preprocessing là official
    python -m evaluator.evaluate --out results/ablation

    # Tập nối/phát triển: qrels còn pending, kết quả ghi ra thư mục KHÔNG phải
    # kết quả chính thức để không lẫn vào báo cáo
    python -m evaluator.evaluate --allow-development --out results/development_eval

Nguyên tắc của CLI này:

* Không bao giờ tự nâng một lần chạy lên "chính thức". Trạng thái
  official/development được quyết định bởi :func:`evaluator.qrels.validate_official_run`
  rồi ghi kèm vào *mọi* file đầu ra.
* Cấu hình C4 luôn được đo **thuần** (tắt fallback) và fallback được đo thành
  một cấu hình riêng có nhãn riêng, không trộn vào số của C4.
* AP dùng toàn bộ ranking có sẵn, nên độ sâu truy vấn mặc định bằng số tài liệu
  trong corpus thay vì K.
* Kết quả kiểm thử bằng fixture không được ghi vào thư mục kết quả chính thức.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator import metrics  # noqa: E402
from evaluator.qrels import (  # noqa: E402
    OfficialEvaluationRefused,
    QrelsQueryHashMismatch,
    assert_queries_hash,
    file_sha256,
    Qrels,
    QuerySet,
    ValidationReport,
    load_qrels,
    load_queries,
    validate_official_run,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = ROOT / "data/queries.json"
DEFAULT_QRELS = ROOT / "data/ground_truth/mm/qrels.json"
DEFAULT_MANIFEST = ROOT / "data/corpus_manifest.json"
DEFAULT_CORPUS = ROOT / "data/corpus.jsonl"
DEFAULT_INDEXES = ROOT / "data/indexes"
DEFAULT_RESULTS = ROOT / "results/ablation"

#: Các thư mục kết quả chính thức; kết quả development/fixture bị chặn ở đây.
OFFICIAL_RESULT_DIRECTORIES = frozenset({"ablation", "latency", "ablation_fallback"})

#: Nhãn riêng cho lần đo có fallback, để không nhầm với C4 thuần.
FALLBACK_LABEL = "C4_fallback"

#: Các tệp TDV chỉ phục vụ phân tích, không cần khi phục vụ truy vấn.
TDV_ANALYSIS_FILES = ("tdv_scores.json", "tdv_stopwords.json", "tdv_metadata.json", "tdv_df.json")


class ResultTargetRefused(RuntimeError):
    """Ném ra khi định điểm ghi kết quả làm sai nguyên tắc tách fixture."""


def assert_result_target(directory: str | Path, official: bool) -> Path:
    """Chặn ghi kết quả không chính thức vào thư mục kết quả chính thức."""
    target = Path(directory).resolve()
    if not official and OFFICIAL_RESULT_DIRECTORIES & set(target.parts):
        raise ResultTargetRefused(
            f"Không được ghi kết quả development/fixture vào {target}: "
            f"đó là vùng kết quả chính thức. Dùng ví dụ results/development_eval.")
    return target


def read_build_metadata(directory: Path) -> Dict[str, object]:
    path = directory / "build_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path}; cần build chỉ mục trước khi đánh giá")
    return json.loads(path.read_text(encoding="utf-8"))


def measure_index_bytes(directory: Path) -> Dict[str, int]:
    """Đo bytes theo một định nghĩa nhất quán và ghi rõ định nghĩa đó.

    * ``serving_bytes`` – mọi tệp chỉ mục cần để phục vụ truy vấn, tức là toàn bộ
      tệp trong thư mục trừ các tệp phân tích TDV.
    * ``tdv_analysis_bytes`` – riêng các tệp ``tdv_*.json`` (chỉ có ở C4).
    * ``total_bytes`` – toàn bộ thư mục, dùng để đối chiếu ``build_summary.json``.
    """
    serving = 0
    analysis = 0
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if path.name in TDV_ANALYSIS_FILES:
            analysis += size
        else:
            serving += size
    return {"serving_bytes": serving, "tdv_analysis_bytes": analysis,
            "total_bytes": serving + analysis}


def _resolve_ranking(results: Sequence[Mapping[str, object]]) -> List[str]:
    return metrics.ranking_from_results(results)


class MetricsUnavailable(RuntimeError):
    """Ném ra nếu không đủ nhãn confirmed để sinh bất kỳ chỉ số nào có ý nghĩa."""


class TruncatedRankingRefused(ValueError):
    """Ném ra nếu độ sâu ranking nhỏ hơn corpus, khiến AP/MAP không còn là toàn ranking."""


def resolve_corpus_sha256(manifest_path: Optional[Path] = None,
                          corpus_path: Optional[Path] = None) -> Dict[str, object]:
    """Xác định SHA256 corpus cần đánh giá.

    Nguồn được phép, theo thứ tự ưu tiên:

    1. manifest đã khóa (``data/corpus_manifest.json``), và nếu có
       ``data/corpus.jsonl`` thì còn phải khớp với hash thật của file;
    2. hash tính trực tiếp từ ``data/corpus.jsonl``.

    Hash trong qrels **không bao giờ** được dùng làm nguồn: nếu không xác định
    được từ manifest hay file corpus thì đánh giá chính thức bị chặn.
    """
    if manifest_path and Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        locked = manifest.get("corpus_sha256") or manifest.get("sha256")
        if locked:
            if corpus_path and Path(corpus_path).is_file():
                actual = file_sha256(corpus_path)
                if actual != locked:
                    raise ValueError(
                        f"SHA256 của {corpus_path} là {actual} nhưng manifest khóa {locked}")
            return {"corpus_sha256": locked, "source": "locked_manifest",
                    "manifest": str(manifest_path), "corpus_file": str(corpus_path or "")}
    if corpus_path and Path(corpus_path).is_file():
        return {"corpus_sha256": file_sha256(corpus_path), "source": "corpus_file_sha256",
                "manifest": str(manifest_path or ""), "corpus_file": str(corpus_path)}
    return {"corpus_sha256": None, "source": "missing", "manifest": str(manifest_path or ""),
            "corpus_file": str(corpus_path or "")}


def check_ranking_depth(depth: int, n_documents: int, *,
                        allow_truncated: bool) -> Dict[str, object]:
    """Bắt buộc lấy ranking đủ sâu để AP/MAP là *toàn ranking*.

    Cắt cụt thì AP không còn là AP toàn ranking và MAP sẽ bị ghi sai tên, nên mặc
    định bị từ chối. Nếu người dùng cố ý chấp nhận, kết quả được ghi nhãn lại là
    ``AP@depth`` để không ai đọc nhầm thành MAP toàn ranking.
    """
    if depth <= 0:
        raise TruncatedRankingRefused("ranking_depth phải là số nguyên dương")
    if n_documents and depth < n_documents:
        message = (f"ranking_depth={depth} nhỏ hơn số tài liệu corpus ({n_documents}); "
                   f"AP/MAP sẽ chỉ tính trên {depth} kết quả đầu, không phải toàn ranking. "
                   f"Đặt --ranking-depth {n_documents} trở lên.")
        if not allow_truncated:
            raise TruncatedRankingRefused(message)
        return {"ranking_depth": depth, "n_documents": n_documents, "truncated": True,
                "average_precision_scope": f"AP@{depth} (cắt cụt, KHÔNG phải MAP toàn ranking)",
                "warning": message}
    return {"ranking_depth": depth, "n_documents": n_documents, "truncated": False,
            "average_precision_scope": metrics.MAP_SCOPE, "warning": None}


def metrics_readiness(queries: QuerySet, qrels: Qrels,
                      scope: Sequence[str]) -> Dict[str, object]:
    """Kiểm tra đã đủ nhãn confirmed để *đo* chỉ số chưa.

    Cố ý nghiêm: chỉ số chỉ được sinh khi **mọi** query trong phạm vi đều có ít
    nhất một phán quyết confirmed. Nếu không, kết quả sẽ là hàng loạt số 0 giả
    tạo ra từ nhãn pending, và bộ đánh giá sẽ từ chối ghi ra thay vì xuất số 0.
    """
    missing = [query_id for query_id in scope if not qrels.relevance_map(query_id)]
    n_confirmed = sum(1 for judgment in qrels.judgments if judgment.usable)
    return {
        "ready": not missing and bool(scope),
        "n_evaluated_queries": len(scope),
        "n_queries_without_confirmed_labels": len(missing),
        "queries_without_confirmed_labels": missing,
        "n_usable_judgments": n_confirmed,
        "reason": (None if not missing and scope else
                   ("không có query nào trong phạm vi đánh giá" if not scope else
                    f"{len(missing)}/{len(scope)} query chưa có nhãn relevance confirmed")),
    }


def configuration_facts(directory: Path) -> Dict[str, object]:
    """Các sự kiện có thật của một chỉ mục, không cần nhãn relevance."""
    metadata = read_build_metadata(directory)
    preprocessing = metadata.get("preprocessing")
    preprocessing = preprocessing if isinstance(preprocessing, Mapping) else {}
    return {
        "configuration": metadata.get("configuration", directory.name),
        "corpus_sha256": metadata.get("corpus_sha256"),
        "n_documents": metadata.get("n_documents"),
        "stoplist_count": metadata.get("stoplist_count"),
        "stoplist_matched_vocabulary_count": metadata.get("stoplist_matched_vocabulary_count"),
        "fallback_policy": metadata.get("fallback_policy"),
        "tdv_seconds": metadata.get("tdv_seconds"),
        "preprocessing_status": metadata.get("preprocessing_status")
        or preprocessing.get("status"),
        "official_preprocessing": bool(metadata.get("official_preprocessing", False)),
        "tokenizer": preprocessing.get("tokenizer") or metadata.get("tokenizer"),
        "tier1_filter": preprocessing.get("tier1_filter") or metadata.get("tier1_filter"),
        "build_seconds": metadata.get("build_seconds"),
        **measure_index_bytes(directory),
    }


def run_configuration(directory: Path, configuration: str, queries: QuerySet, qrels: Qrels, *,
                      k: int, depth: int, enable_fallback: bool,
                      expected_corpus_sha256: Optional[str] = None) -> Dict[str, object]:
    """Chạy một cấu hình trên toàn bộ query và trả về các hàng + thống kê.

    ``expected_corpus_sha256`` phải được truyền vào tường minh (không dùng trạng
    thái toàn cục) để lời gọi độc lập cũng tự kiểm tra hash. Khi không truyền,
    hàm vẫn dùng hash trong ``build_metadata.json`` nhưng ghi rõ là chỉ tự kiểm
    chéo, không phải xác minh với manifest.

    Ranking rỗng là một kết quả **hợp lệ** (ví dụ truy vấn mà C4 thuần không còn
    token trong từ vựng): nó vẫn sinh ra dòng chỉ số bằng 0 và vẫn được tính vào bình
    quân macro. Không loại bỏ vì sẽ làm đẹp số.
    """
    metadata = read_build_metadata(directory)
    index_hash = metadata.get("corpus_sha256")
    if expected_corpus_sha256 is None:
        if index_hash is None:
            raise ValueError(
                f"{configuration}: không có expected_corpus_sha256 và build_metadata cũng thiếu "
                f"corpus_sha256 nên không thể xác minh chỉ mục")
        expected_corpus_sha256 = index_hash
    if index_hash != expected_corpus_sha256:
        raise ValueError(
            f"{configuration}: SHA256 corpus của chỉ mục ({index_hash}) khác corpus đang "
            f"đánh giá ({expected_corpus_sha256})")
    if enable_fallback:
        _check_fallback_peer(directory, configuration, expected_corpus_sha256)
        from scripts.search import SearchService
        service = SearchService(directory, enable_fallback=True)
        engine = None
    else:
        from scripts.search import load_engine
        service = None
        engine = load_engine(directory)
    rows: List[Dict[str, object]] = []
    fallback_used = 0
    empty_results: List[str] = []
    for query_id in queries.query_ids():
        if enable_fallback:
            response = service.search(queries.text_of(query_id), top_k=depth)
            results = response["results"]
            if response["fallback_used"]:
                fallback_used += 1
        else:
            results = engine.search(queries.text_of(query_id), top_k=depth)
        ranking = _resolve_ranking(results)
        if not ranking:
            empty_results.append(query_id)
        judgments = qrels.relevance_map(query_id)
        if not judgments:
            raise MetricsUnavailable(
                f"Query {query_id} chưa có nhãn relevance confirmed; không được xuất chỉ số 0 giả")
        row = metrics.query_metrics(ranking, judgments, k)
        row.update({
            "configuration": configuration,
            "query_id": query_id,
            "category": queries.category_of(query_id),
            "query_type": queries.types_of(query_id),
            "empty_result": not ranking,
        })
        rows.append(row)
    return {
        "configuration": configuration,
        "rows": rows,
        "summary": metrics.macro_average(rows, queries.query_ids()),
        "by_category": metrics.group_by_category(rows),
        "statistics": {
            "n_queries": len(rows),
            "n_queries_with_fallback_used": fallback_used,
            "n_queries_with_empty_result": len(empty_results),
            "empty_result_query_ids": empty_results,
            "empty_results_are_valid_outcome": True,
            "fallback_enabled": enable_fallback,
            "fallback_policy": metadata.get("fallback_policy"),
            "stoplist_count": metadata.get("stoplist_count"),
            "ranking_depth_requested": depth,
            **measure_index_bytes(directory),
        },
        "metadata": metadata,
    }


def _check_fallback_peer(directory: Path, configuration: str,
                         expected_corpus_sha256: str) -> None:
    """Kiểm tra chỉ mục C3 đi kèm trước khi đo fallback của C4."""
    if configuration != "C4":
        return
    peer = directory.parent / "C3"
    if not peer.is_dir():
        raise FileNotFoundError(
            f"Đo fallback của C4 cần chỉ mục C3 riêng tại {peer}; hãy build C3 trước "
            f"hoặc tắt bằng --no-fallback-measurement")
    peer_metadata = read_build_metadata(peer)
    if peer_metadata.get("corpus_sha256") != expected_corpus_sha256:
        raise ValueError(
            f"C3 dùng corpus {peer_metadata.get('corpus_sha256')} khác C4 "
            f"({expected_corpus_sha256}); fallback sẽ trộn hai corpus khác nhau")
    own = read_build_metadata(directory)
    if peer_metadata.get("preprocessing") != own.get("preprocessing"):
        raise ValueError(
            "C3 và C4 dùng khác tiền xử lý nên fallback sẽ không nhất quán: "
            f"C3={peer_metadata.get('preprocessing')} C4={own.get('preprocessing')}")


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def environment() -> Dict[str, object]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
    }


def evaluate(*, indexes: Path, queries_path: Path, qrels_path: Path, out: Path,
             manifest_path: Optional[Path] = None, corpus_path: Optional[Path] = None,
             k: int = metrics.DEFAULT_K,
             configurations: Sequence[str] = ("C1", "C2", "C3", "C4"),
             ranking_depth: Optional[int] = None, mode: str = "auto",
             allow_development: bool = False, measure_fallback: bool = True,
             allow_truncated_depth: bool = False) -> Dict[str, object]:
    """Điểm vào chính của evaluator; trả về payload kết quả đầy đủ."""
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)
    actual_queries_sha256 = assert_queries_hash(qrels, queries)
    resolved = resolve_corpus_sha256(manifest_path, corpus_path)
    expected_corpus_sha256 = resolved["corpus_sha256"]

    per_configuration: Dict[str, Dict[str, object]] = {}
    facts: Dict[str, Dict[str, object]] = {}
    reports: Dict[str, ValidationReport] = {}
    for configuration in configurations:
        directory = Path(indexes) / configuration
        if not directory.is_dir():
            raise FileNotFoundError(f"Thiếu thư mục chỉ mục {directory}")
        facts[configuration] = configuration_facts(directory)
        reports[configuration] = validate_official_run(
            qrels, queries, corpus_sha256=expected_corpus_sha256,
            index_metadata=read_build_metadata(directory),
            evaluated_query_ids=queries.query_ids())
    combined = ValidationReport(
        issues=tuple(issue for report in reports.values() for issue in report.issues),
        warnings=tuple(issue for report in reports.values() for issue in report.warnings),
        checked={name: dict(report.checked) for name, report in reports.items()})

    n_documents = max((int(fact["n_documents"] or 0) for fact in facts.values()), default=0)
    depth_check = check_ranking_depth(
        ranking_depth if ranking_depth else (n_documents or 1000), n_documents,
        allow_truncated=allow_truncated_depth)
    depth = depth_check["ranking_depth"]

    readiness = metrics_readiness(queries, qrels, queries.query_ids())
    metrics_status = "measured" if readiness["ready"] else "unmeasured"

    if mode == "official" or (mode == "auto" and combined.official_allowed):
        combined.raise_if_blocked("đánh giá chính thức")
        official = True
    elif mode == "auto" and not allow_development:
        combined.raise_if_blocked("đánh giá chính thức")
        official = False
    else:
        official = False
    destination = assert_result_target(out, official)
    reasons = [str(issue) for issue in combined.issues]

    if metrics_status == "unmeasured":
        payload = {
            "official": False,
            "mode": "development",
            "mode_reason": "; ".join(reasons) or "chạy ở chế độ development",
            "metrics_status": metrics_status,
            "metrics_reason": readiness["reason"],
            "metrics_readiness": readiness,
            "k": k,
            "ranking_depth": depth,
            "ranking_depth_check": depth_check,
            "average_precision_scope": depth_check["average_precision_scope"],
            "corpus_sha256": expected_corpus_sha256,
            "corpus_sha256_source": resolved["source"],
            "queries_sha256": actual_queries_sha256,
            "queries": queries.summary(),
            "qrels": qrels.summary(),
            "validation": combined.as_dict(),
            "environment": environment(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "configurations": {
                name: {"summary": None, "by_category": None, "statistics": fact,
                       "metrics_status": metrics_status}
                for name, fact in facts.items()},
            "note": ("Chưa đủ nhãn relevance confirmed nên KHÔNG ghi bất kỳ chỉ số nào. "
                     "Các số 0 ở đây sẽ là số giả; bộ đánh giá từ chối xuất chúng."),
        }
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "run.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload

    for configuration in configurations:
        directory = Path(indexes) / configuration
        per_configuration[configuration] = run_configuration(
            directory, configuration, queries, qrels, k=k, depth=depth, enable_fallback=False,
            expected_corpus_sha256=expected_corpus_sha256)
        if measure_fallback and configuration == "C4":
            per_configuration[FALLBACK_LABEL] = run_configuration(
                directory, FALLBACK_LABEL, queries, qrels, k=k, depth=depth,
                enable_fallback=True, expected_corpus_sha256=expected_corpus_sha256)

    per_query_rows = [row for entry in per_configuration.values() for row in entry["rows"]]
    summary_rows = [{"configuration": name, **entry["summary"]}
                    for name, entry in per_configuration.items()]
    category_rows = []
    for name, entry in per_configuration.items():
        for category, values in entry["by_category"].items():
            category_rows.append({"configuration": name, "category": category,
                                  **{key: value for key, value in values.items()
                                     if key not in {"zero_relevant_query_ids"}}})

    payload = {
        "official": official,
        "mode": "official" if official else "development",
        "mode_reason": None if official else ("; ".join(reasons) or "chạy ở chế độ development"),
        "metrics_status": metrics_status,
        "metrics_reason": readiness["reason"],
        "metrics_readiness": readiness,
        "k": k,
        "ranking_depth": depth,
        "ranking_depth_check": depth_check,
        "average_precision_scope": depth_check["average_precision_scope"],
        "corpus_sha256": expected_corpus_sha256,
        "corpus_sha256_source": resolved["source"],
        "queries_sha256": actual_queries_sha256,
        "queries": queries.summary(),
        "qrels": qrels.summary(),
        "validation": combined.as_dict(),
        "environment": environment(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configurations": {
            name: {"summary": entry["summary"], "by_category": entry["by_category"],
                   "statistics": entry["statistics"],
                   "preprocessing_status": entry["metadata"].get("preprocessing_status"),
                   "stoplist_count": entry["metadata"].get("stoplist_count"),
                   "metrics_status": metrics_status}
            for name, entry in per_configuration.items()},
    }

    _write_csv(destination / "per_query.csv", metrics.PER_QUERY_COLUMNS, per_query_rows)
    _write_csv(destination / "summary.csv", metrics.SUMMARY_COLUMNS, summary_rows)
    _write_csv(destination / "summary_by_category.csv",
               ("configuration", "category") + metrics.SUMMARY_COLUMNS[1:], category_rows)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Đánh giá ablation C1–C4 (P@K, R@K, F1@K, MAP, NDCG@K)")
    parser.add_argument("--indexes", type=Path, default=DEFAULT_INDEXES)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="Manifest corpus đã khóa; nguồn SHA256 ưu tiên")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                        help="data/corpus.jsonl; dùng để tự kiểm tra hash khi thiếu manifest")
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--k", type=int, default=metrics.DEFAULT_K)
    parser.add_argument("--configurations", nargs="+", default=["C1", "C2", "C3", "C4"])
    parser.add_argument("--ranking-depth", type=int, default=None,
                        help="Độ sâu lấy ranking cho AP toàn ranking; "
                             "mặc định = số tài liệu corpus")
    parser.add_argument("--allow-truncated-depth", action="store_true",
                        help="Cho phép AP trên ranking cắt cụt; kết quả sẽ được ghi nhãn AP@depth")
    parser.add_argument("--mode", choices=["auto", "official", "development"], default="auto")
    parser.add_argument("--allow-development", action="store_true",
                        help="Cho phép chạy và ghi ra thư mục khi điều kiện chính thức chưa đủ")
    parser.add_argument("--no-fallback-measurement", dest="measure_fallback",
                        action="store_false", default=True,
                        help="Bỏ lần đo riêng cho fallback của C4")
    args = parser.parse_args(argv)
    try:
        payload = evaluate(
            indexes=args.indexes, queries_path=args.queries, qrels_path=args.qrels,
            out=args.out, manifest_path=args.corpus_manifest, corpus_path=args.corpus, k=args.k,
            configurations=args.configurations, ranking_depth=args.ranking_depth,
            mode=args.mode, allow_development=args.allow_development,
            measure_fallback=args.measure_fallback,
            allow_truncated_depth=args.allow_truncated_depth)
    except OfficialEvaluationRefused as error:
        print(str(error), file=sys.stderr)
        print("Dùng --allow-development --out <thư mục không phải kết quả chính thức> "
              "để chạy thử mà không ghi đè kết quả chính thức.", file=sys.stderr)
        return 2
    except ResultTargetRefused as error:
        print(str(error), file=sys.stderr)
        return 3
    except TruncatedRankingRefused as error:
        print(str(error), file=sys.stderr)
        return 6
    except QrelsQueryHashMismatch as error:
        print(f"Từ chối đo do lệch hash tập truy vấn: {error}", file=sys.stderr)
        return 7
    print(f"mode={payload['mode']} k={payload['k']} depth={payload['ranking_depth']} "
          f"corpus_sha256={payload['corpus_sha256']} "
          f"({payload.get('corpus_sha256_source', 'unknown')})")
    if payload["metrics_status"] != "measured":
        print(f"CHỈ SỐ CHẤT LƯỢNG CHƯA ĐO: {payload['metrics_reason']}")
        print("Không ghi per_query.csv/summary.csv vì mọi số 0 ở đây đều là số giả.")
        for name, entry in payload["configurations"].items():
            statistics = entry["statistics"]
            print(f"  {name:12s} serving_bytes={statistics['serving_bytes']:>9d} "
                  f"stoplist={statistics.get('stoplist_count')} "
                  f"preprocessing={statistics.get('preprocessing_status')}")
        print(f"-> {args.out}/run.json")
        return 5
    for name, entry in payload["configurations"].items():
        summary = entry["summary"]
        if not summary:
            continue
        print(f"  {name:12s} MAP={summary['map']:.4f} "
              f"P@{payload['k']}={summary['macro_precision_at_k']:.4f} "
              f"R@{payload['k']}={summary['macro_recall_at_k']:.4f} "
              f"F1={summary['macro_f1_at_k']:.4f} "
              f"NDCG@{payload['k']}={summary['macro_ndcg_at_k']:.4f} "
              f"(rỗng {entry['statistics']['n_queries_with_empty_result']} query)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CORPUS",
    "DEFAULT_INDEXES",
    "DEFAULT_MANIFEST",
    "DEFAULT_QRELS",
    "DEFAULT_QUERIES",
    "DEFAULT_RESULTS",
    "FALLBACK_LABEL",
    "MetricsUnavailable",
    "QrelsQueryHashMismatch",
    "OFFICIAL_RESULT_DIRECTORIES",
    "ResultTargetRefused",
    "TDV_ANALYSIS_FILES",
    "TruncatedRankingRefused",
    "assert_result_target",
    "check_ranking_depth",
    "configuration_facts",
    "evaluate",
    "main",
    "measure_index_bytes",
    "metrics_readiness",
    "read_build_metadata",
    "resolve_corpus_sha256",
    "run_configuration",
]
