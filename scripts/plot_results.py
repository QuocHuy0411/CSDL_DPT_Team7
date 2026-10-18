"""Vẽ biểu đồ kết quả thực nghiệm từ đúng những file đã đo.

Cách dùng (từ thư mục gốc repo)::

    python scripts/plot_results.py --ablation results/ablation --latency results/latency
    python scripts/plot_results.py --tdv data/indexes/C4 --out report/figures

Nguyên tắc bất di bất dịch của script này:

* **Không bịa số.** Mỗi biểu đồ chỉ được vẽ khi đã có file kết quả tương ứng.
  Thiếu dữ liệu thì in ra danh sách biểu đồ *không* tạo được và thoát với mã
  lỗi; không có hình nào "minh hoạ bằng số ảo".
* Nếu `run.json`/`latency.json` ghi `official = false` hoặc
  `metrics_status = unmeasured` thì hình đó được đóng dấu rõ là *tạm thời* trong
  tiêu đề và việc vẽ biểu đồ chỉ số chất lượng bị từ chối.
* Biểu đồ chỉ số chất lượng chỉ vẽ khi `metrics_status == "measured"`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ABLATION = ROOT / "results/ablation"
DEFAULT_LATENCY = ROOT / "results/latency"
DEFAULT_TDV = ROOT / "data/indexes/C4"
DEFAULT_OUT = ROOT / "report/figures"

CONFIGURATION_ORDER = ("C1", "C2", "C3", "C4", "C4_fallback")

#: Nhãn tiếng Việt cho bốn cấu hình lọc stopword.
CONFIGURATION_LABELS = {
    "C1": "C1 (không lọc)",
    "C2": "C2 (stoplist tĩnh)",
    "C3": "C3 (tầng 1)",
    "C4": "C4 (tầng 1 + TDV)",
    "C4_fallback": "C4 + fallback C3",
}


class MissingData(RuntimeError):
    """Không đủ dữ liệu thật để vẽ một biểu đồ."""


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise MissingData(f"Thiếu {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise MissingData(f"Thiếu {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
        "axes.spines.right": False, "figure.autolayout": True,
    })
    return plt


def _ordered(names: Sequence[str]) -> List[str]:
    known = [name for name in CONFIGURATION_ORDER if name in names]
    extra = sorted(name for name in names if name not in CONFIGURATION_ORDER)
    return known + extra


def _label(name: str) -> str:
    return CONFIGURATION_LABELS.get(name, name)


def plot_quality_metrics(summary_rows: Sequence[Dict[str, str]], out: Path, k: int,
                         provisional: bool) -> Path:
    """Cột nhóm: MAP, P@K, R@K, F1@K, NDCG@K cho từng cấu hình."""
    plt = _setup_matplotlib()
    names = _ordered([row["configuration"] for row in summary_rows])
    lookup = {row["configuration"]: row for row in summary_rows}
    fields = [("map", "MAP"), ("macro_precision_at_k", f"P@{k}"),
              ("macro_recall_at_k", f"R@{k}"), ("macro_f1_at_k", f"F@{k}"),
              ("macro_ndcg_at_k", f"NDCG@{k}")]
    positions = list(range(len(fields)))
    width = 0.8 / max(len(names), 1)
    figure, axes = plt.subplots(figsize=(1.9 * len(fields) + 2.2, 4.0))
    for index, name in enumerate(names):
        row = lookup[name]
        values = [float(row[field]) for field, _ in fields]
        offsets = [p - 0.4 + width / 2 + index * width for p in positions]
        axes.bar(offsets, values, width=width, label=_label(name))
    axes.set_xticks(positions)
    axes.set_xticklabels([label for _, label in fields])
    axes.set_ylim(0, 1)
    axes.set_ylabel("giá trị chỉ số")
    axes.set_title(f"Chỉ số truy xuất theo cấu hình lọc stopword (K={k})"
                   + ("\nTẠM THỜI: chưa đủ điều kiện đánh giá chính thức" if provisional else ""))
    axes.legend(fontsize=7, ncols=2)
    path = out / "fig_quality_metrics.png"
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_map_by_category(category_rows: Sequence[Dict[str, str]], out: Path,
                         provisional: bool) -> Path:
    """MAP theo từng lĩnh vực, một thanh cho mỗi cấu hình."""
    plt = _setup_matplotlib()
    categories = sorted({row["category"] for row in category_rows})
    names = _ordered({row["configuration"] for row in category_rows})
    lookup = {(row["configuration"], row["category"]): float(row["map"]) for row in category_rows}
    figure, axis = plt.subplots(figsize=(max(6.0, 1.25 * len(categories)), 4.2))
    width = 0.8 / max(len(names), 1)
    for index, name in enumerate(names):
        values = [lookup.get((name, category), 0.0) for category in categories]
        offsets = [p - 0.4 + width / 2 + index * width for p in range(len(categories))]
        axis.bar(offsets, values, width=width, label=_label(name))
    axis.set_xticks(range(len(categories)))
    axis.set_xticklabels(categories, rotation=20, ha="right", fontsize=7)
    axis.set_ylabel("MAP (trên toàn ranking)")
    axis.set_ylim(0, 1)
    axis.set_title("MAP theo lĩnh vực" + ("\nTẠM THỜI" if provisional else ""))
    axis.legend(fontsize=7, ncols=2)
    path = out / "fig_map_by_category.png"
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_latency(payload: dict, out: Path) -> Path:
    """Độ trễ mean/P95 trên thang số đo; mục tiêu xa trục được ghi bằng chữ."""
    plt = _setup_matplotlib()
    entries = payload["configurations"]
    names = _ordered(list(entries))
    means = [float(entries[name]["latency"]["mean_ms"]) for name in names]
    p95s = [float(entries[name]["latency"]["p95_ms"]) for name in names]
    labels = [_label(name) for name in names]
    positions = list(range(len(names)))
    figure, axis = plt.subplots(figsize=(1.5 * len(names) + 2.0, 4.0))
    mean_bars = axis.bar([p - 0.2 for p in positions], means, width=0.4, label="mean")
    p95_bars = axis.bar([p + 0.2 for p in positions], p95s, width=0.4, label="P95")
    for bar in (*mean_bars, *p95_bars):
        axis.annotate(f"{bar.get_height():.2f}",
                      (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    axis.set_ylim(0, max(p95s + means) * 1.27)
    mean_target = float(payload["targets"]["mean_latency_ms"])
    target_result = "đạt" if all(value < mean_target for value in means) else "chưa đạt"
    axis.text(0.98, 0.97, f"Mục tiêu mean < {mean_target:g} ms: {target_result}",
              transform=axis.transAxes, ha="right", va="top", fontsize=8)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axis.set_ylabel("thời gian (ms)")
    provisional = payload.get("measurement_status") != "official"
    axis.set_title("Độ trễ truy vấn"
                   + ("\nNhãn relevance còn pending; biểu đồ chỉ đo thời gian" if provisional else ""))
    axis.legend(fontsize=7)
    path = out / "fig_latency.png"
    figure.tight_layout()
    figure.subplots_adjust(left=0.13)
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_index_size(payload: dict, out: Path) -> Path:
    """Dung lượng: từng chỉ mục đơn và gói triển khai C4 + C3."""
    plt = _setup_matplotlib()
    bundles = payload["deployment_bundles"]
    names = _ordered([name for name in bundles if name != "C4_with_fallback"])
    serving = [bundles[name]["serving_bytes"] for name in names]
    analysis = [bundles[name]["tdv_analysis_bytes"] for name in names]
    labels = [_label(name) for name in names]
    positions = list(range(len(names)))
    figure, axis = plt.subplots(figsize=(1.5 * len(names) + 2.0, 4.2))
    axis.bar(positions, serving, width=0.5, label="tệp phục vụ (serving)")
    if any(analysis):
        axis.bar(positions, analysis, width=0.5, bottom=serving, label="tệp phân tích TDV")
    if "C4_with_fallback" in bundles:
        bundle = bundles["C4_with_fallback"]
        axis.axhline(bundle["serving_bytes"], color="tab:red", linestyle="--", linewidth=0.9,
                     label="gói C4 + C3 (C4 bật fallback)")
        axis.text(len(names) - 0.4, bundle["serving_bytes"], " gói C4+C3", color="tab:red",
                  va="bottom", ha="left", fontsize=7)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axis.set_ylabel("số byte trên đĩa")
    axis.set_title("Dung lượng chỉ mục trên đĩa")
    axis.legend(fontsize=7)
    path = out / "fig_index_size.png"
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_tdv_distribution(scores: Dict[str, float], metadata: dict, out: Path) -> Path:
    """Phân phối điểm TDV và đường tích lũy của phần bị loại."""
    plt = _setup_matplotlib()
    values = [float(value) for value in scores.values()]
    if not values:
        raise MissingData("tdv_scores.json không có điểm TDV nào")
    tolerance = float(metadata.get("tolerance", 1e-12))
    filtered = [value for value in values if value < -tolerance]
    ordered = sorted(values)
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    left.hist(values, bins=60, color="tab:blue", alpha=0.8)
    left.axvline(-tolerance, color="tab:red", linestyle="--", linewidth=0.9,
                 label=f"ngưỡng loại < -{tolerance:g}")
    left.set_yscale("log")
    left.set_xlabel("điểm TDV")
    left.set_ylabel("số từ (thang log)")
    left.set_title(f"Phân phối TDV trên {len(values)} từ")
    left.legend(fontsize=7)
    right.plot(ordered, [index / len(ordered) for index in range(1, len(ordered) + 1)],
               color="tab:green")
    right.axvline(-tolerance, color="tab:red", linestyle="--", linewidth=0.9)
    right.set_xlabel("điểm TDV (đã sắp xếp)")
    right.set_ylabel("phân phối tích lũy")
    right.set_title(f"{len(filtered)} từ ({len(filtered) / len(values) * 100:.2f}%) "
                    f"có điểm < -{tolerance:g}")
    path = out / "fig_tdv_distribution.png"
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ từ kết quả đã đo")
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--latency", type=Path, default=DEFAULT_LATENCY)
    parser.add_argument("--tdv", type=Path, default=DEFAULT_TDV,
                        help="Thư mục chỉ mục C4 chứa tdv_scores.json/tdv_metadata.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="+",
                        choices=["quality", "category", "latency", "size", "tdv"],
                        default=["quality", "category", "latency", "size", "tdv"])
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    skipped: List[str] = []

    if "quality" in args.only or "category" in args.only:
        try:
            run = _load_json(args.ablation / "run.json")
        except MissingData as error:
            run = None
            skipped.append(f"chỉ số chất lượng: {error}")
        if run is not None:
            if run.get("metrics_status") != "measured":
                skipped.append(
                    "chỉ số chất lượng: metrics_status="
                    f"{run.get('metrics_status')} ({run.get('metrics_reason')}). "
                    "Không vẽ biểu đồ chỉ số từ dữ liệu chưa đo.")
            else:
                k = int(run.get("k", 10))
                provisional = not run.get("official", False)
                summary_rows = _load_csv(args.ablation / "summary.csv")
                if "quality" in args.only:
                    written.append(plot_quality_metrics(summary_rows, out, k, provisional))
                if "category" in args.only:
                    category_rows = _load_csv(args.ablation / "summary_by_category.csv")
                    written.append(plot_map_by_category(category_rows, out, provisional))

    if "latency" in args.only or "size" in args.only:
        try:
            payload = _load_json(args.latency / "latency.json")
        except MissingData as error:
            payload = None
            skipped.append(f"độ trễ/kích thước: {error}")
        if payload is not None:
            if "latency" in args.only:
                written.append(plot_latency(payload, out))
            if "size" in args.only:
                written.append(plot_index_size(payload, out))

    if "tdv" in args.only:
        try:
            scores = _load_json(args.tdv / "tdv_scores.json")
            metadata = _load_json(args.tdv / "tdv_metadata.json")
            written.append(plot_tdv_distribution(scores, metadata, out))
        except MissingData as error:
            skipped.append(f"phân phối TDV: {error}")

    for path in written:
        print(f"đã vẽ {path}")
    for reason in skipped:
        print(f"BỎ QUA {reason}", file=sys.stderr)
    if not written:
        print("Không vẽ được biểu đồ nào: thiếu dữ liệu thực nghiệm đã đo.", file=sys.stderr)
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIGURATION_LABELS",
    "CONFIGURATION_ORDER",
    "DEFAULT_ABLATION",
    "DEFAULT_LATENCY",
    "DEFAULT_OUT",
    "DEFAULT_TDV",
    "MissingData",
    "main",
    "plot_index_size",
    "plot_latency",
    "plot_map_by_category",
    "plot_quality_metrics",
    "plot_tdv_distribution",
]
