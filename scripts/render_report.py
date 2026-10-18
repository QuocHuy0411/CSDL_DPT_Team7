"""Render M2 result tables from real benchmark artifacts, never from pending labels.

Run after rebuilding indexes and rerunning benchmark_latency. This generates the
tables included by report/chapter4.tex plus a compact Markdown handoff.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from engine.preprocessing import ROOT
from scripts.annotate import load_context


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tex(value):
    substitutions = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                     "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(substitutions.get(c, c) for c in str(value))


def table(headers, rows, spec, caption, *, long=False):
    env = "longtable" if long else "tabular"
    lines = ([rf"\begin{{{env}}}{{{spec}}}", r"\hline"] if long else
             [r"\begin{table}[htbp]", r"\centering", rf"\begin{{{env}}}{{{spec}}}", r"\hline"])
    lines += [" & ".join(headers) + r" \\", r"\hline"]
    if long:
        lines.append(r"\endhead")
    for row in rows:
        lines.append(" & ".join(tex(x) for x in row) + r" \\")
    lines += [r"\hline"]
    if long:
        lines += [rf"\caption{{{caption}}} \\", rf"\end{{{env}}}"]
    else:
        lines += [rf"\end{{{env}}}", rf"\caption{{{caption}}}", r"\end{table}"]
    return "\n".join(lines)


def render():
    benchmark = load(ROOT / "results/latency/latency.json")
    queries_path = ROOT / "data/queries.json"
    query_hash = hashlib.sha256(queries_path.read_bytes()).hexdigest()
    if benchmark["queries_sha256"] != query_hash:
        raise ValueError("Benchmark queries are stale; rerun benchmark first")
    queries = load(queries_path)["queries"]
    entries = benchmark["configurations"]
    for name in entries:
        current = load(ROOT / "data/indexes" / name / "build_metadata.json")
        if (current["corpus_sha256"] != benchmark["corpus_sha256"] or
                current["tokenizer"] != entries[name]["tokenizer"] or
                current["preprocessing"]["fingerprint"] != entries[name].get("preprocessing_fingerprint")):
            raise ValueError("Benchmark/index corpus or tokenizer mismatch")
    pool, qrels, index = load_context(ROOT / "data/ground_truth/mm/pool.json",
                                      ROOT / "data/ground_truth/mm/qrels.json")
    confirmed = sum(r["status"] == "confirmed" for r in index.values())
    sample_path = ROOT / "data/ground_truth/mm/m1_review_sample.json"
    sample = load(sample_path)
    review_path = ROOT / "results/label_review.json"
    review = load(review_path) if review_path.is_file() else {}
    review_open = review.get("status") == "needs_human_adjudication"
    gold_path = ROOT / "data/gold/segmentation_labels.jsonl"
    review_stale = bool(review_open and (
        review.get("qrels_sha256_at_review") !=
        hashlib.sha256((ROOT / "data/ground_truth/mm/qrels.json").read_bytes()).hexdigest()
        or review.get("gold_sha256_at_review") !=
        (hashlib.sha256(gold_path.read_bytes()).hexdigest() if gold_path.is_file() else None)))
    gold_rows = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()] if gold_path.is_file() else []
    gold_confirmed = sum(row.get("label_status") == "confirmed" for row in gold_rows)
    ablation_summary = ROOT / "results/ablation/summary.csv"
    ablation_run_path = ROOT / "results/ablation/run.json"
    ablation_run = load(ablation_run_path) if ablation_run_path.is_file() else {}
    qrels_path = ROOT / "data/ground_truth/mm/qrels.json"
    qrels_hash = hashlib.sha256(qrels_path.read_bytes()).hexdigest()
    resolution_path = ROOT / "results/label_review_resolution.json"
    resolution = load(resolution_path) if resolution_path.is_file() else {}
    source_path = ROOT / resolution["source"] if resolution.get("source") else None
    amendment_path = ROOT / resolution["amendment_source"] if resolution.get("amendment_source") else None
    resolution_current = bool(
        resolution.get("qrels_before_sha256") == review.get("qrels_sha256_at_review")
        and resolution.get("qrels_after_sha256") == qrels_hash
        and source_path and source_path.is_file()
        and resolution.get("source_sha256") == hashlib.sha256(source_path.read_bytes()).hexdigest()
        and (not amendment_path or (
            amendment_path.is_file() and resolution.get("amendment_source_sha256") ==
            hashlib.sha256(amendment_path.read_bytes()).hexdigest())))
    if resolution_current:
        unresolved = [row for row in resolution.get("disagreements", [])
                      if row.get("status") == "pending_user_confirmation"]
        review_notice = (
            f"Đã nhập {resolution['n_submitted_pairs']} quyết định do Team7 tự xác nhận; "
            + (f"{len(unresolved)} cặp Q16 còn cần chốt mức độ liên quan. "
               if unresolved else "Q16 đã được Team7 sửa mức 2 thành 1 ở bước bổ sung. ")
            + "Đây là trạng thái rà nhãn, không chứng minh có người thứ hai chấm độc lập."
        )
        review_stale = review.get("gold_sha256_at_review") != hashlib.sha256(gold_path.read_bytes()).hexdigest()
    else:
        review_notice = f"Có {len(review.get('qrels_flags', []))} cặp cần người gán duyệt lại."
    crosscheck_path = ROOT / "results/m1_crosscheck.json"
    crosscheck = load(crosscheck_path) if crosscheck_path.is_file() else {}
    crosscheck_matches = bool(
        crosscheck.get("qrels_sha256") == qrels_hash
        and crosscheck.get("sample_sha256") == hashlib.sha256(sample_path.read_bytes()).hexdigest()
        and crosscheck.get("sample_size") == sample["sample_size"]
        and crosscheck.get("status") in {"agreement_recorded", "requires_human_adjudication"}
        and crosscheck.get("m1_reviewer") and crosscheck.get("m1_labels_sha256"))
    if crosscheck_matches:
        disposition = ("Cần phân xử bất đồng và xác minh danh tính người chấm."
                       if crosscheck["disagreement_count"] else
                       "Không có bất đồng trong mẫu; vẫn cần xác minh danh tính người chấm.")
        crosscheck_text = (
            f"Phiếu M1 tự khai chấm độc lập đã được đối chiếu: "
            f"{crosscheck['agreements']}/{sample['sample_size']} cặp trùng nhãn; "
            f"{crosscheck['disagreement_count']} bất đồng. "
            f"{disposition} Công cụ chỉ kiểm tra cấu trúc."
        )
    else:
        crosscheck_text = (
            f"Có tệp đối chiếu M1 nhưng hash qrels đã đổi; cần đối chiếu lại và xác minh "
            f"người chấm trước khi tuyên bố kiểm tra chéo {sample['sample_size']} cặp."
            if crosscheck_path.is_file() else
            f"Chưa có tệp nhãn M1 chấm độc lập hợp lệ cho mẫu {sample['sample_size']} cặp; "
            "không được tuyên bố đã kiểm tra chéo."
        )
    legacy_run_matches_review_snapshot = bool(
        ablation_run_path.is_file() and ablation_run.get("qrels_sha256") is None
        and review.get("qrels_sha256_at_review") == qrels_hash
        and ablation_run_path.stat().st_mtime_ns >= qrels_path.stat().st_mtime_ns)
    metrics_available = bool(
        confirmed == len(index) and qrels.get("status") == "human_annotation_confirmed"
        and ablation_summary.is_file() and ablation_run_path.is_file()
        and (ablation_run.get("qrels_sha256") == qrels_hash or
             legacy_run_matches_review_snapshot)
        and ablation_run.get("corpus_sha256") == benchmark["corpus_sha256"]
        and ablation_run.get("queries_sha256") == query_hash
        and ablation_run.get("qrels", {}).get("n_confirmed") == confirmed)
    gold_manifest_path = ROOT / "data/gold/segmentation_manifest.json"
    gold_eval_path = ROOT / "results/segmentation/evaluation.json"
    gold_manifest = load(gold_manifest_path) if gold_manifest_path.is_file() else {}
    gold_eval = load(gold_eval_path) if gold_eval_path.is_file() else {}
    gold_hash = hashlib.sha256(gold_path.read_bytes()).hexdigest() if gold_path.is_file() else None
    legacy_gold_matches_review_snapshot = bool(
        gold_eval_path.is_file() and gold_eval.get("gold_labels_sha256") is None
        and review.get("gold_sha256_at_review") == gold_hash
        and gold_eval_path.stat().st_mtime_ns >= gold_path.stat().st_mtime_ns)
    gold_metrics_available = bool(
        gold_rows and gold_eval.get("official") is True
        and (gold_eval.get("gold_labels_sha256") == gold_hash or
             legacy_gold_matches_review_snapshot)
        and gold_eval.get("sample_sha256") == gold_manifest.get("sample_sha256")
        and gold_eval.get("corpus_sha256") == benchmark["corpus_sha256"])
    gold_lines = ["% Generated by python -m scripts.render_report; do not hand edit."]
    if gold_metrics_available:
        scores = gold_eval["metrics"]
        gold_lines.append(
            r"\textbf{Phân đoạn từ: số đo trên nhãn hiện có, tạm thời khi audit gold còn mở.} "
            r"Word-span micro trên 150 câu; chỉ công bố là kết quả cuối sau khi duyệt các cặp nghi vấn."
        )
        gold_lines.append(table(
            ["Thuật toán", "Precision", "Recall", "F1"],
            [[name.upper() if name != "bimm" else "BiMM",
              f"{scores[name]['precision']:.4f}", f"{scores[name]['recall']:.4f}",
              f"{scores[name]['f1']:.4f}"] for name in ("fmm", "bmm", "bimm")],
            "|l|r|r|r|", "Word-span micro trên 150 câu gold hiện có; cần rà soát nhãn.",
        ))
    else:
        gold_lines.append(
            r"\textbf{Chưa xuất F1 tách từ hợp lệ:} cần chạy lại "
            r"\texttt{evaluator.segmentation\_eval} sau khi cập nhật gold và xác minh SHA256."
        )
    (ROOT / "report/segmentation_results.tex").write_text("\n\n".join(gold_lines) + "\n", encoding="utf-8")
    tdv = load(ROOT / "data/indexes/C4/tdv_metadata.json")
    c4 = load(ROOT / "data/indexes/C4/build_metadata.json")
    env, parameters = benchmark["environment"], benchmark["parameters"]
    lines = ["% Generated by python -m scripts.render_report; do not hand edit.",
             r"\section{Trạng thái dữ liệu và phạm vi phép đo}",
             f"Corpus chính gồm 2.500 bài thuộc 10 lĩnh vực, 250 bài/lĩnh vực. "
             f"Bộ 20 query có hai query/lĩnh vực. Pool dùng BiMM có {len(index)} cặp, "
             f"{confirmed} nhãn confirmed; mẫu kiểm tra chéo gồm {sample['sample_size']} cặp.",
             crosscheck_text,
             "Ba baseline pooling: TF-IDF Cosine sublinear/smoothed/L2; binary presence "
             "Cosine không IDF; BM25 với $k_1=1{,}2$, $b=0{,}75$, IDF một lần. "
             "Cùng corpus và tokenizer, hợp Top-20; TF-IDF đã đối soát SearchEngine.",
             (r"\textbf{P@10, Recall@10, F1@10, MAP và NDCG@10 đã tính trên nhãn hiện có, "
              r"nhưng chỉ là số tạm thời vì rà soát ngữ nghĩa còn mở.} "
              f"{review_notice} "
              "Cổng kỹ thuật của evaluator không thay thế được bước này. "
              r"Xem \texttt{docs/label\_review.md}." +
              (" Bộ nhãn đã thay đổi sau lần audit, cần rà lại review." if review_stale else "")
              if metrics_available and review_open else
              (r"\textbf{P@10, Recall@10, F1@10, MAP và NDCG@10 đã được tính "
               r"trên qrels đủ điều kiện kỹ thuật.} "
               "Giới hạn pooling và độ tin cậy nhãn vẫn phải ghi rõ."
               if metrics_available else
               r"\textbf{P@10, Recall@10, F1@10, MAP và NDCG@10 chưa có kết quả "
               r"cập nhật hợp lệ.} Nhãn pending hoặc file đo cũ không được biến thành "
               "số đo chính thức.")),
             r"\section{Môi trường và thiết lập}",
             tex(f"Lần đo UTC: {benchmark['created_utc']}. Python {env['python']}; "
                 f"{env['platform']}; CPU {env['processor']}; {env['cpu_count']} logical CPU."),
             f"Mỗi query lặp {parameters['repeat_per_query']} lần, "
             f"{parameters['warmup_iterations']} lượt warm-up trên toàn query set, "
             f"Top-{parameters['top_k']}. Chỉ mục đã nạp; đo gồm chuẩn hóa, BiMM, lọc và "
             "truy xuất, không gồm khởi động Python/nạp từ điển/đĩa hay vẽ giao diện. "
             "C4 thuần và fallback C3 được đo độc lập.",
             r"\section{Kết quả độ trễ và dung lượng}"]
    latency_rows = []
    for name, entry in entries.items():
        values = entry["latency"]
        latency_rows.append([name, f"{values['mean_ms']:.3f}", f"{values['p95_ms']:.3f}",
                             values["n_empty_results"], values["n_measurements"]])
        if name == "C4" and entry.get("fallback_measurement"):
            values = entry["fallback_measurement"]
            latency_rows.append(["C4 + C3", f"{values['mean_ms']:.3f}", f"{values['p95_ms']:.3f}",
                                 values["n_empty_results"], values["n_measurements"]])
    lines.append(table(["Cấu hình", "Mean (ms)", "P95 (ms)", "Rỗng", "Số đo"],
                       latency_rows, "|l|r|r|r|r|", "Độ trễ trên corpus chính, BiMM thật."))
    size_rows = []
    for name, bundle in benchmark["deployment_bundles"].items():
        size_rows.append([name.replace("C4_with_fallback", "C4 + C3"), bundle["serving_bytes"],
                          bundle["tdv_analysis_bytes"],
                          f"{100 * bundle['reduction_vs_baseline']:.2f}%",
                          "đạt" if bundle["meets_30_percent_reduction_target"] else "không đạt"])
    lines.append(table(["Cấu hình", "Phục vụ (byte)", "TDV (byte)", "Giảm/C1", "Mục tiêu 30\\%"],
                       size_rows, "|l|r|r|r|l|", "Dung lượng chỉ mục đơn và gói triển khai fallback."))
    c4_empty = [qid for qid, values in entries["C4"]["per_query_latency"].items()
                if values["n_empty_results"]]
    if c4_empty:
        lines.append("C4 thuần trả kết quả rỗng ở " + ", ".join(map(tex, c4_empty)) +
                     "; phép đo C4 + C3 dùng chỉ mục C3 riêng để khôi phục truy vấn, "
                     "không giữ term đã bị xóa trong chỉ mục C4.")
    mean_pass = all(e["latency"]["mean_ms"] < 30 for e in entries.values())
    lines += ["Mục tiêu mean dưới 30 ms: " + ("đạt" if mean_pass else "không đạt ở mọi cấu hình") + ". "
              "Tỷ lệ giảm âm có nghĩa dung lượng tăng. Phải giữ C3 nếu triển khai fallback; "
              "không sử dụng riêng dung lượng C4 để kết luận cho cả gói. "
              "Tốc độ thấp hơn không chứng minh relevance tốt hơn.",
              r"\begin{figure}[htbp]\centering\includegraphics[width=.9\textwidth]{figures/fig_latency.png}"
              r"\caption{Mean/P95 truy vấn BiMM.}\end{figure}",
              r"\begin{figure}[htbp]\centering\includegraphics[width=.9\textwidth]{figures/fig_index_size.png}"
              r"\caption{Dung lượng phục vụ, phân tích và gói fallback.}\end{figure}",
              r"\section{Thống kê TDV trên corpus đa lĩnh vực}",
              f"Mật độ gốc {tdv['base_density']:.8f}; chấm {tdv['n_scored_terms']} term, "
              f"{tdv['n_filtered_terms']} điểm âm dưới dung sai. Bảo vệ "
              f"{len(c4['tdv_guarded_terms'])} term theo whitelist/policy hợp từ; "
              f"lọc thật {c4['stoplist_matched_vocabulary_count']} term. "
              f"Thời gian TDV {c4['tdv_seconds']:.3f} giây, one-shot trên đủ 2.500 bài. "
              "Đây là tiêu chí hình học, không phải nhãn không liên quan ngữ nghĩa.",
              r"\begin{figure}[htbp]\centering\includegraphics[width=.9\textwidth]{figures/fig_tdv_distribution.png}"
              r"\caption{Phân bố TDV thực, không ép hình chữ U.}\end{figure}",
              r"\clearpage\section{Toàn bộ query và phân tích theo lĩnh vực}"]
    qrows = []
    for q in queries:
        qid = q["query_id"]
        qrows.append([qid, q["text"], *[f"{entries[c]['per_query_latency'][qid]['mean_ms']:.3f}"
                                       for c in ("C1", "C2", "C3", "C4")]])
    lines.append(table(["Query", "Nội dung", "C1", "C2", "C3", "C4"], qrows,
                       "|l|p{.48\\textwidth}|r|r|r|r|", "Mean ms của toàn bộ 20 query.", long=True))
    groups = defaultdict(list)
    for q in queries:
        groups[q["category"]].append(q["query_id"])
    category_rows = [[category, len(ids), *[f"{statistics.mean(entries[c]['per_query_latency'][q]['mean_ms'] for q in ids):.3f}"
                                           for c in ("C1", "C2", "C3", "C4")]]
                     for category, ids in groups.items()]
    lines.append(table(["Lĩnh vực", "Query", "C1", "C2", "C3", "C4"], category_rows,
                       "|l|r|r|r|r|r|", "Mean ms theo 10 lĩnh vực; chỉ hai query/lĩnh vực."))
    if metrics_available:
        import csv
        with open(ablation_summary, encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        ir_rows = []
        for r in reader:
            cfg = r["configuration"].replace("C4_fallback", "C4 + C3")
            ir_rows.append([cfg, f"{float(r['map']):.4f}", f"{float(r['macro_precision_at_k']):.4f}",
                            f"{float(r['macro_recall_at_k']):.4f}", f"{float(r['macro_f1_at_k']):.4f}",
                            f"{float(r['macro_ndcg_at_k']):.4f}"])
        lines.append(r"\section{Kết quả chất lượng truy xuất trên nhãn hiện có}")
        lines.append(table(["Cấu hình", "MAP", "P@10", "R@10", "F1@10", "NDCG@10"],
                           ir_rows, "|l|r|r|r|r|r|",
                           "Chất lượng trên 20 query, Top-10; cần rà soát nhãn."
                           if review_open else "Chất lượng truy xuất trên 20 query, Top-10."))
        per_query_path = ROOT / "results/ablation/per_query.csv"
        by_category_path = ROOT / "results/ablation/summary_by_category.csv"
        if per_query_path.is_file() and by_category_path.is_file():
            with open(per_query_path, encoding="utf-8") as f:
                per_query = {(r["query_id"], r["configuration"]): r
                             for r in csv.DictReader(f)}
            with open(by_category_path, encoding="utf-8") as f:
                by_category = {(r["category"], r["configuration"]): r
                               for r in csv.DictReader(f)}
            configs = ("C1", "C2", "C3", "C4")
            query_quality_rows = [[q["query_id"],
                                   *[f"{float(per_query[q['query_id'], c]['average_precision']):.3f}"
                                     for c in configs]] for q in queries]
            lines.append(table(["Query", *configs], query_quality_rows,
                               "|l|r|r|r|r|", "AP theo từng query trên nhãn hiện có.",
                               long=True))
            category_quality_rows = [[category,
                                      *[f"{float(by_category[category, c]['map']):.3f}"
                                        for c in configs]] for category in groups]
            lines.append(table(["Lĩnh vực", *configs], category_quality_rows,
                               "|p{.32\\textwidth}|r|r|r|r|",
                               "MAP theo 10 lĩnh vực; chỉ hai query mỗi lĩnh vực."))
    lines.append("Mỗi lĩnh vực chỉ có hai query nên kết quả nhóm chưa đủ để suy rộng. "
                 "Các bảng chất lượng toàn query/nhóm do evaluator xuất khi qrels đủ điều kiện, "
                 "không thay dữ liệu thiếu bằng số 0.")
    (ROOT / "report/experiment_tables.tex").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    md = ["# Báo cáo M2 — bản tích hợp M1", "",
          f"Đo UTC {benchmark['created_utc']}; tokenizer BiMM, corpus 2.500 bài/10 lĩnh vực.",
          f"Pool {len(index)} cặp; confirmed {confirmed}; mẫu kiểm tra chéo {sample['sample_size']}.",
          crosscheck_text,
          (f"P/R/F1/MAP/NDCG đã tính trên {confirmed} nhãn confirmed nhưng chỉ là số tạm thời: "
           f"{review_notice}"
           if metrics_available and review_open else
           (f"P/R/F1/MAP/NDCG đã tính trên {confirmed} nhãn đủ điều kiện kỹ thuật."
            if metrics_available else "P/R/F1/MAP/NDCG chưa có kết quả cập nhật hợp lệ.")), "",
          "| Cấu hình | Mean ms | P95 ms | Rỗng | Phép đo |", "|---|---:|---:|---:|---:|"]
    md += ["| " + " | ".join(map(str, row)) + " |" for row in latency_rows]
    md += ["", "| Cấu hình | Byte phục vụ | Byte TDV | Giảm so C1 | Giảm30% |",
           "|---|---:|---:|---:|---|"]
    md += ["| " + " | ".join(map(str, row)) + " |" for row in size_rows]
    if metrics_available:
        md += ["", "### Chất lượng truy xuất trên nhãn hiện có (Top-10; tạm thời nếu review còn mở)",
               "| Cấu hình | MAP | P@10 | R@10 | F1@10 | NDCG@10 |",
               "|---|---:|---:|---:|---:|---:|"]
        for r in ir_rows:
            md.append("| " + " | ".join(map(str, r)) + " |")
    md += ["", "Fallback phải cộng C3+C4. Không tuyên bố C4 có relevance tốt nhất.",
           "C4 thuần trả rỗng ở " + ", ".join(c4_empty) + "; fallback C3 khôi phục các truy vấn này."
           if c4_empty else "C4 thuần không có truy vấn rỗng trong bộ 20 query.",
           f"Gold M1: {gold_confirmed}/{len(gold_rows)} câu có cờ confirmed; "
           "cần rà " + ", ".join(flag["sample_id"] for flag in review.get("gold_flags", [])) +
           " và kiểm tra chéo độc lập trước khi kết luận chất lượng."
           if review_open else f"Gold M1: {gold_confirmed}/{len(gold_rows)} câu có cờ confirmed.",
           "", "Chi tiết thiết lập, toàn bộ query và 10 lĩnh vực: report/experiment_tables.tex; "
           "CSV latency: results/latency/latency_per_query.csv; báo cáo đầy đủ: report/main.pdf.",
           "", "Tái tạo: `python -X utf8 -m scripts.build_indexes`, "
           "`python -X utf8 -m benchmarks.benchmark_latency`, "
           "`python -X utf8 -m scripts.plot_results --only latency size tdv --out report/figures`, "
           "`python -X utf8 -m scripts.render_report`."]
    (ROOT / "report/m2_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Rendered report/experiment_tables.tex and report/m2_report.md from measured artifacts")


if __name__ == "__main__":
    render()
