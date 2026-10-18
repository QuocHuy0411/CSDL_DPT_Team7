"""Đo độ trễ truy vấn và kích thước chỉ mục trên bộ chỉ mục đã build.

Cách dùng (từ thư mục gốc repo)::

    python -m benchmarks.benchmark_latency --out results/latency
    python -m benchmarks.benchmark_latency --repeat 200 --warmup 5 --top-k 10

Nguyên tắc báo cáo trung thực:

* Khi qrels còn pending **hoặc** tiền xử lý còn ``development_only``, kết quả vẫn
  chạy được (đo thời gian không cần nhãn) nhưng bắt buộc ghi
  ``qrels_status``, ``tokenizer_status`` và ``official_eligible = false`` để không
  ai đọc nhầm là số đo chính thức.
* Cấu hình C4 được đo **thuần** với fallback tắt; fallback được đo thành một mục
  riêng có nhãn riêng, không gộp vào số của C4.
* Kích thước chỉ mục đo bằng ba số có định nghĩa ghi rõ trong báo cáo:
  ``serving_bytes`` (tệp cần để phục vụ), ``tdv_analysis_bytes`` (tệp phân tích
  TDV) và ``total_bytes``. Mục tiêu "giảm 30% chỉ mục" chỉ được kết luận trên một
  định nghĩa duy nhất, và báo cáo nói rõ đang dùng định nghĩa nào.
* Mỗi số đo đều kèm môi trường chạy, số lần lặp, số query và thời điểm ghi.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.evaluate import (  # noqa: E402
    DEFAULT_INDEXES,
    DEFAULT_MANIFEST,
    DEFAULT_QRELS,
    DEFAULT_QUERIES,
    FALLBACK_LABEL,
    OFFICIAL_RESULT_DIRECTORIES,
    ResultTargetRefused,
    assert_result_target,
    measure_index_bytes,
    read_build_metadata,
)
from evaluator.metrics import percentile  # noqa: E402
from evaluator.qrels import load_qrels, load_queries  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/latency"

DEFAULT_REPEAT = 100
DEFAULT_WARMUP = 5
DEFAULT_TOP_K = 10

#: Mục tiêu định hướng của đồ án; chỉ báo đạt/không đạt theo số đo, không mặc định là đạt.
TARGET_MEAN_MS = 30.0
TARGET_P95_MS = 50.0
TARGET_INDEX_REDUCTION = 0.30


def environment() -> Dict[str, object]:
    import os

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "timer": "time.perf_counter_ns",
        "timer_resolution_ns": time.get_clock_info("perf_counter").resolution * 1e9,
    }


def _summarize(latencies_ms: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(latencies_ms)
    return {
        "mean_ms": float(statistics.fmean(ordered)) if ordered else 0.0,
        "median_ms": float(statistics.median(ordered)) if ordered else 0.0,
        "p95_ms": percentile(ordered, 0.95),
        "p99_ms": percentile(ordered, 0.99),
        "min_ms": float(ordered[0]) if ordered else 0.0,
        "max_ms": float(ordered[-1]) if ordered else 0.0,
        "stdev_ms": float(statistics.pstdev(ordered)) if len(ordered) > 1 else 0.0,
    }


def _measure(service, items: Sequence[tuple], *, repeat: int, warmup: int,
             top_k: int, use_fallback: bool) -> Dict[str, object]:
    """Đo một dịch vụ tìm kiếm trên các cặp ``(query_id, text)``.

    ``service`` là đối tượng có ``search(query, top_k=...)`` (đã nạp sẵn để không
    tính thời gian nạp chỉ mục vào phép đo). ``warmup`` là số lượt chạy **toàn bộ**
    bộ query trước khi đo, và kết quả trả về gồm số tổng hợp lẫn số riêng cho từng
    query để có thể soi từng truy vấn và từng lĩnh vực.
    """
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    def once(text: str) -> tuple:
        if use_fallback:
            response = service.search(text, top_k=top_k)
            return response["results"], bool(response["fallback_used"])
        return service.search(text, top_k=top_k), False

    for _ in range(max(warmup, 0)):
        for _query_id, text in items:
            once(text)

    latencies: List[float] = []
    empty_results = 0
    fallback_used = 0
    per_query: Dict[str, Dict[str, object]] = {}
    for query_id, text in items:
        query_latencies: List[float] = []
        query_empty = 0
        query_fallback = 0
        for _ in range(repeat):
            started = time.perf_counter_ns()
            results, fell_back = once(text)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            query_latencies.append(elapsed)
            latencies.append(elapsed)
            if not results:
                query_empty += 1
                empty_results += 1
            if fell_back:
                query_fallback += 1
                fallback_used += 1
        per_query[query_id] = {
            **_summarize(query_latencies),
            "n_measurements": len(query_latencies),
            "n_empty_results": query_empty,
            "n_fallback_used": query_fallback,
        }

    total = len(items) * repeat
    summary = _summarize(latencies)
    summary.update({
        "n_queries": len(items),
        "repeat_per_query": repeat,
        "warmup_iterations": max(warmup, 0),
        "n_measurements": total,
        "n_empty_results": empty_results,
        "n_fallback_used": fallback_used,
        "fallback_enabled": use_fallback,
        "queries_per_second": (total / (summary["mean_ms"] * total / 1000.0)) if total else 0.0,
        "top_k": top_k,
    })
    return {"aggregate": summary, "per_query": per_query}


def _preprocessing_status(metadata: Dict[str, object]) -> Dict[str, object]:
    preprocessing = metadata.get("preprocessing")
    preprocessing = preprocessing if isinstance(preprocessing, dict) else {}
    return {
        "preprocessing_status": metadata.get("preprocessing_status")
        or preprocessing.get("status"),
        "official_preprocessing": bool(metadata.get("official_preprocessing", False)),
        "tokenizer": preprocessing.get("tokenizer") or metadata.get("tokenizer"),
        "tier1_filter": preprocessing.get("tier1_filter") or metadata.get("tier1_filter"),
        "tokenizer_status": ("official" if metadata.get("official_preprocessing")
                             else "development_only"),
    }


def run(*, indexes: Path = DEFAULT_INDEXES, queries_path: Path = DEFAULT_QUERIES,
        qrels_path: Path = DEFAULT_QRELS, manifest_path: Path = DEFAULT_MANIFEST,
        configurations: Sequence[str] = ("C1", "C2", "C3", "C4"),
        repeat: int = DEFAULT_REPEAT, warmup: int = DEFAULT_WARMUP,
        top_k: int = DEFAULT_TOP_K, out: Path = DEFAULT_OUT) -> Dict[str, object]:
    """Đo toàn bộ cấu hình và trả về payload kết quả."""
    queries = load_queries(queries_path)
    query_ids = queries.query_ids()
    items = [(query_id, queries.text_of(query_id)) for query_id in query_ids]
    qrels_path = Path(qrels_path)
    qrels_status = "not_read"
    n_pending = 0
    n_confirmed = 0
    if qrels_path.is_file():
        qrels = load_qrels(qrels_path)
        qrels_status = qrels.status
        n_pending = len(qrels.pending_judgments())
        n_confirmed = sum(1 for judgment in qrels.judgments if judgment.confirmed)
    corpus_sha256 = None
    if Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        corpus_sha256 = manifest.get("corpus_sha256") or manifest.get("sha256")

    from scripts.search import SearchService, load_engine
    hash_checks = _check_hashes(indexes, configurations, queries, qrels_path, corpus_sha256)

    entries: Dict[str, Dict[str, object]] = {}
    for configuration in configurations:
        directory = Path(indexes) / configuration
        metadata = read_build_metadata(directory)
        pure = load_engine(directory)
        latency = _measure(pure, items, repeat=repeat, warmup=warmup,
                           top_k=top_k, use_fallback=False)
        sizes = measure_index_bytes(directory)
        status = _preprocessing_status(metadata)
        official_eligible = (bool(status["official_preprocessing"])
                             and qrels_status == "human_annotation_confirmed")
        aggregate = latency["aggregate"]
        entry = {
            "configuration": configuration,
            "latency": aggregate,
            "per_query_latency": latency["per_query"],
            "index_bytes": sizes,
            "corpus_sha256": metadata.get("corpus_sha256"),
            "preprocessing_fingerprint": (metadata.get("preprocessing") or {}).get("fingerprint"),
            "stoplist_count": metadata.get("stoplist_count"),
            "fallback_policy": metadata.get("fallback_policy"),
            "tdv_seconds": metadata.get("tdv_seconds"),
            "official_eligible": official_eligible,
            **status,
            "meets_mean_target_ms": aggregate["mean_ms"] < TARGET_MEAN_MS,
            "meets_p95_target_ms": aggregate["p95_ms"] < TARGET_P95_MS,
        }
        if configuration == "C4" and "C3" in configurations:
            service = SearchService(directory, enable_fallback=True)
            fallback = _measure(service, items, repeat=repeat, warmup=warmup, top_k=top_k,
                                use_fallback=True)
            entry["fallback_measurement"] = fallback["aggregate"]
            entry["fallback_per_query_latency"] = fallback["per_query"]
            entry["fallback_measurement"]["configuration"] = FALLBACK_LABEL
        entries[configuration] = entry

    baseline = configurations[0]
    baseline_bytes = entries[baseline]["index_bytes"]["serving_bytes"]
    for configuration, entry in entries.items():
        serving = entry["index_bytes"]["serving_bytes"]
        reduction = (baseline_bytes - serving) / baseline_bytes
        entry["reduction_vs_baseline"] = {
            "baseline_configuration": baseline,
            "serving_bytes": serving,
            "reduction": reduction,
            "meets_30_percent_reduction_target": reduction >= TARGET_INDEX_REDUCTION,
        }

    bundles = _deployment_bundles(entries, baseline, baseline_bytes)

    eligible = all(entry["official_eligible"] for entry in entries.values())
    payload = {
        "official_eligible": eligible,
        "measurement_status": "official" if eligible else "provisional",
        "metrics_available": False,
        "metrics_note": ("Độ trễ và dung lượng đo được không cần nhãn; chỉ số chất lượng "
                         "(P@K, R@K, F1@K, MAP, NDCG@K) chưa đo vì qrels chưa confirmed. "
                         "Chạy evaluator/evaluate.py để có chỉ số."),
        "qrels_status": qrels_status,
        "n_qrels_pending": n_pending,
        "n_qrels_confirmed": n_confirmed,
        "queries_file": str(queries_path),
        "queries_sha256": queries.file_sha256,
        "qrels_file": str(qrels_path) if qrels_path.is_file() else None,
        "qrels_queries_sha256": hash_checks.get("qrels_queries_sha256"),
        "corpus_sha256": corpus_sha256,
        "hash_checks": hash_checks,
        "environment": environment(),
        "parameters": {
            "repeat_per_query": repeat,
            "warmup_iterations": warmup,
            "top_k": top_k,
            "n_queries": len(items),
            "query_ids": query_ids,
        },
        "targets": {
            "mean_latency_ms": TARGET_MEAN_MS,
            "p95_latency_ms": TARGET_P95_MS,
            "index_reduction": TARGET_INDEX_REDUCTION,
            "note": "Mục tiêu định hướng; trạng thái đạt/không đạt lấy từ số đo thật.",
        },
        "byte_definitions": {
            "serving_bytes": "Mọi tệp trong thư mục chỉ mục trừ tdv_*.json; số dùng cho "
                             "mục tiêu giảm kích thước của một chỉ mục đơn.",
            "tdv_analysis_bytes": "Riêng các tệp tdv_scores/tdv_stopwords/tdv_metadata/tdv_df.",
            "total_bytes": "Toàn bộ thư mục, đối chiếu được với build_summary.json.",
            "bundle_serving_bytes": "Tổng serving_bytes của mọi chỉ mục "
                                    "phải cài cùng nhau để phục vụ.",
        },
        "baseline_configuration": baseline,
        "deployment_bundles": bundles,
        "configurations": entries,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    destination = assert_result_target(out, official=True)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "latency.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_latency_csv(destination / "latency.csv", payload)
    _write_per_query_csv(destination / "latency_per_query.csv", payload, queries)
    return payload


def _check_hashes(indexes: Path, configurations: Sequence[str], queries, qrels_path: Path,
                  corpus_sha256: Optional[str]) -> Dict[str, object]:
    """Đối chiếu SHA256 giữa manifest, queries.json, qrels và từng chỉ mục trước khi đo."""
    checks: Dict[str, object] = {"corpus_sha256": corpus_sha256,
                                 "queries_sha256": queries.file_sha256,
                                 "queries_sha256_declared": queries.queries_sha256,
                                 "per_configuration": {}, "mismatches": []}
    sources = {"manifest": corpus_sha256, "queries": queries.corpus_sha256}
    if not queries.file_sha256:
        checks["mismatches"].append("queries: không tính được SHA256 của queries.json")
    if Path(qrels_path).is_file():
        qrels = load_qrels(qrels_path)
        sources["qrels"] = qrels.corpus_sha256
        checks["qrels_queries_sha256"] = qrels.queries_sha256
        if qrels.judgments and qrels.queries_sha256 != queries.file_sha256:
            checks["mismatches"].append(
                f"qrels: queries_sha256 {qrels.queries_sha256} != {queries.file_sha256}")
    for name, value in sources.items():
        if value is None:
            checks["mismatches"].append(f"{name}: thiếu corpus_sha256")
        elif corpus_sha256 and value != corpus_sha256:
            checks["mismatches"].append(f"{name}: {value} != {corpus_sha256}")
    for configuration in configurations:
        metadata = read_build_metadata(Path(indexes) / configuration)
        value = metadata.get("corpus_sha256")
        checks["per_configuration"][configuration] = value
        if value is None:
            checks["mismatches"].append(f"{configuration}: build_metadata thiếu corpus_sha256")
        elif corpus_sha256 and value != corpus_sha256:
            checks["mismatches"].append(f"{configuration}: {value} != {corpus_sha256}")
    checks["consistent"] = not checks["mismatches"]
    if not checks["consistent"]:
        raise ValueError("Từ chối đo vì SHA256 không khớp: " + "; ".join(checks["mismatches"]))
    return checks


def _deployment_bundles(entries: Dict[str, Dict[str, object]], baseline: str,
                        baseline_bytes: int) -> Dict[str, Dict[str, object]]:
    """Dung lượng phải cài cho từng cách triển khai thực tế.

    C4 có fallback thì hệ thống phải giữ **cả** chỉ mục C4 và một chỉ mục C3 riêng,
    nên gói triển khai tính cả hai. Báo cáo phải tách rõ giảm kích thước của *chỉ
    mục C4 thuần* với dung lượng của *gói C4 + fallback*, và không được dùng số của
    chỉ mục thuần để tuyên bố giảm 30% cho cả hệ thống.
    """
    bundles: Dict[str, Dict[str, object]] = {}
    for configuration, entry in entries.items():
        sizes = entry["index_bytes"]
        bundles[configuration] = {
            "components": [configuration],
            "serving_bytes": sizes["serving_bytes"],
            "tdv_analysis_bytes": sizes["tdv_analysis_bytes"],
            "total_bytes": sizes["total_bytes"],
            "reduction_vs_baseline": (baseline_bytes - sizes["serving_bytes"]) / baseline_bytes,
            "meets_30_percent_reduction_target":
                (baseline_bytes - sizes["serving_bytes"]) / baseline_bytes
                >= TARGET_INDEX_REDUCTION,
            "note": "Chỉ mục đơn, phục vụ không dùng fallback.",
        }
    if "C4" in entries and "C3" in entries:
        c3 = entries["C3"]["index_bytes"]
        c4 = entries["C4"]["index_bytes"]
        serving = c3["serving_bytes"] + c4["serving_bytes"]
        reduction = (baseline_bytes - serving) / baseline_bytes
        bundles["C4_with_fallback"] = {
            "components": ["C3", "C4"],
            "serving_bytes": serving,
            "tdv_analysis_bytes": c3["tdv_analysis_bytes"] + c4["tdv_analysis_bytes"],
            "total_bytes": c3["total_bytes"] + c4["total_bytes"],
            "reduction_vs_baseline": reduction,
            "meets_30_percent_reduction_target": reduction >= TARGET_INDEX_REDUCTION,
            "note": ("Gói triển khai thật của C4 khi bật fallback: phải giữ cả C4 và C3. "
                     "Số này lớn hơn số của C4 thuần và có thể lớn hơn baseline."),
        }
    bundles[baseline]["baseline"] = True
    return bundles


def _write_latency_csv(path: Path, payload: Dict[str, object]) -> None:
    import csv

    columns = ("kind", "label", "mean_ms", "median_ms", "p95_ms", "p99_ms", "min_ms", "max_ms",
               "stdev_ms", "n_queries", "repeat_per_query", "n_measurements", "top_k",
               "serving_bytes", "tdv_analysis_bytes", "total_bytes",
               "reduction_vs_baseline", "meets_30_percent_reduction_target",
               "meets_mean_target_ms", "meets_p95_target_ms",
               "tokenizer_status", "qrels_status", "official_eligible")
    qrels_status = payload.get("qrels_status")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for configuration, entry in payload["configurations"].items():
            row = {**entry["latency"], **entry["index_bytes"],
                   "kind": "configuration", "label": configuration,
                   "tokenizer_status": entry["tokenizer_status"],
                   "qrels_status": qrels_status,
                   "official_eligible": entry["official_eligible"],
                   "reduction_vs_baseline": entry["reduction_vs_baseline"]["reduction"],
                   "meets_30_percent_reduction_target":
                       entry["reduction_vs_baseline"]["meets_30_percent_reduction_target"]}
            writer.writerow(row)
            fallback = entry.get("fallback_measurement")
            if fallback:
                writer.writerow({**fallback,
                                 "kind": "latency_only", "label": FALLBACK_LABEL,
                                 "tokenizer_status": entry["tokenizer_status"],
                                 "qrels_status": qrels_status,
                                 "official_eligible": entry["official_eligible"],
                                 "meets_mean_target_ms": fallback["mean_ms"] < TARGET_MEAN_MS,
                                 "meets_p95_target_ms": fallback["p95_ms"] < TARGET_P95_MS})
        for label, bundle in payload["deployment_bundles"].items():
            writer.writerow({"kind": "bundle", "label": label, **bundle,
                             "qrels_status": qrels_status})


def _write_per_query_csv(path: Path, payload: Dict[str, object], queries) -> None:
    """Ghi độ trễ theo từng query để soi được cả 20 truy vấn và cả 10 lĩnh vực.

    Cột ``metrics_available`` nói rõ số chỉ số chất lượng có được tính ở lần chạy này
    hay không; khi qrels còn pending thì cột này là ``False`` cho mọi dòng.
    """
    import csv

    columns = ("configuration", "query_id", "category", "query_types", "mean_ms", "median_ms",
               "p95_ms", "p99_ms", "min_ms", "max_ms", "n_measurements", "n_empty_results",
               "n_fallback_used", "top_k", "official_eligible", "metrics_available")
    metrics_available = payload.get("metrics_available", False)
    rows = []
    for configuration, entry in payload["configurations"].items():
        sources = [(configuration, entry.get("per_query_latency") or {})]
        if entry.get("fallback_per_query_latency"):
            sources.append((FALLBACK_LABEL, entry["fallback_per_query_latency"]))
        for label, per_query in sources:
            for query_id, values in per_query.items():
                rows.append({"configuration": label, "query_id": query_id,
                             "category": queries.category_of(query_id),
                             "query_types": queries.types_of(query_id),
                             "official_eligible": entry["official_eligible"],
                             "metrics_available": metrics_available, **values})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Đo độ trễ truy vấn và kích thước chỉ mục")
    parser.add_argument("--indexes", type=Path, default=DEFAULT_INDEXES)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--configurations", nargs="+", default=["C1", "C2", "C3", "C4"])
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT,
                        help="Số lần lặp mỗi query (mặc định 100)")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="Số lượt chạy toàn bộ bộ query trước khi đo (mặc định 5)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        payload = run(indexes=args.indexes, queries_path=args.queries, qrels_path=args.qrels,
                      manifest_path=args.corpus_manifest, configurations=args.configurations,
                      repeat=args.repeat, warmup=args.warmup, top_k=args.top_k, out=args.out)
    except FileNotFoundError as error:
        print(f"Không đo được: {error}", file=sys.stderr)
        return 4
    except ResultTargetRefused as error:
        print(str(error), file=sys.stderr)
        return 3
    except ValueError as error:
        print(f"Không đo được: {error}", file=sys.stderr)
        return 5
    print(f"measurement_status={payload['measurement_status']} "
          f"qrels_status={payload['qrels_status']} "
          f"(pending={payload['n_qrels_pending']}, confirmed={payload['n_qrels_confirmed']}) "
          f"tokenizer={next(iter(payload['configurations'].values()))['tokenizer_status']}")
    for configuration, entry in payload["configurations"].items():
        latency = entry["latency"]
        print(f"  {configuration:14s} mean={latency['mean_ms']:8.3f} ms "
              f"p95={latency['p95_ms']:8.3f} ms "
              f"serving={entry['index_bytes']['serving_bytes']:>9d} B "
              f"reduction={entry['reduction_vs_baseline']['reduction'] * 100:6.2f}%")
        fallback = entry.get("fallback_measurement")
        if fallback:
            print(f"  {'C4_fallback':14s} mean={fallback['mean_ms']:8.3f} ms "
                  f"p95={fallback['p95_ms']:8.3f} ms "
                  f"fallback_used={fallback['n_fallback_used']}/{fallback['n_measurements']}")
    bundle = payload["deployment_bundles"].get("C4_with_fallback")
    if bundle:
        verdict = "đạt" if bundle["meets_30_percent_reduction_target"] else "KHÔNG đạt"
        print(f"  gói C4+C3 (C4 bật fallback): serving={bundle['serving_bytes']} B, "
              f"giảm {bundle['reduction_vs_baseline'] * 100:.2f}% so với "
              f"{payload['baseline_configuration']} -> mục tiêu 30%: {verdict}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUT",
    "DEFAULT_REPEAT",
    "DEFAULT_TOP_K",
    "DEFAULT_WARMUP",
    "OFFICIAL_RESULT_DIRECTORIES",
    "TARGET_INDEX_REDUCTION",
    "TARGET_MEAN_MS",
    "TARGET_P95_MS",
    "environment",
    "main",
    "run",
]
