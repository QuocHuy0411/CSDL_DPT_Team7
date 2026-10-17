# Báo cáo M2 — bản tích hợp M1

Đo UTC 2026-10-16T13:42:48.188882+00:00; tokenizer BiMM, corpus 2.500 bài/10 lĩnh vực.
Pool 669 cặp; confirmed 0; mẫu kiểm tra chéo 134.
Chất lượng P/R/F1/MAP/NDCG chưa đo chính thức khi qrels/gold còn pending.

| Cấu hình | Mean ms | P95 ms | Rỗng | Phép đo |
|---|---:|---:|---:|---:|
| C1 | 0.994 | 1.992 | 0 | 2000 |
| C2 | 0.463 | 1.034 | 0 | 2000 |
| C3 | 0.772 | 1.737 | 0 | 2000 |
| C4 | 0.487 | 1.355 | 200 | 2000 |
| C4 + C3 | 0.503 | 1.226 | 0 | 2000 |

| Cấu hình | Byte phục vụ | Byte TDV | Giảm so C1 | Giảm30% |
|---|---:|---:|---:|---|
| C1 | 9621765 | 0 | 0.00% | không đạt |
| C2 | 6987156 | 0 | 27.38% | không đạt |
| C3 | 8748327 | 0 | 9.08% | không đạt |
| C4 | 6542944 | 1834419 | 32.00% | đạt |
| C4 + C3 | 15291271 | 1834419 | -58.92% | không đạt |

Fallback phải cộng C3+C4. Không tuyên bố C4 có relevance tốt nhất.
C4 thuần trả rỗng ở Q04, Q15; fallback C3 khôi phục các truy vấn này.
Word-span gold M1 và relevance đều cần người xác nhận, không thay bằng nhãn gợi ý.

Chi tiết thiết lập, toàn bộ query và 10 lĩnh vực: report/experiment_tables.tex; CSV latency: results/latency/latency_per_query.csv; báo cáo đầy đủ: report/main.pdf.

Tái tạo: `python -X utf8 -m scripts.build_indexes`, `python -X utf8 -m benchmarks.benchmark_latency`, `python -X utf8 -m scripts.plot_results --only latency size tdv --out report/figures`, `python -X utf8 -m scripts.render_report`.
