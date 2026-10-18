# Báo cáo M2 — bản tích hợp M1

Đo UTC 2026-10-16T13:42:48.188882+00:00; tokenizer BiMM, corpus 2.500 bài/10 lĩnh vực.
Pool 669 cặp; confirmed 669; mẫu kiểm tra chéo 134.
Có tệp đối chiếu M1 nhưng hash qrels đã đổi; cần đối chiếu lại và xác minh người chấm trước khi tuyên bố kiểm tra chéo 134 cặp.
P/R/F1/MAP/NDCG đã tính trên 669 nhãn confirmed nhưng chỉ là số tạm thời: Có 37 cặp cần người gán duyệt lại.

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

### Chất lượng truy xuất trên nhãn hiện có (Top-10; tạm thời nếu review còn mở)
| Cấu hình | MAP | P@10 | R@10 | F1@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| C1 | 0.7707 | 0.6250 | 0.5747 | 0.5127 | 0.7956 |
| C2 | 0.7607 | 0.6200 | 0.5699 | 0.5081 | 0.7930 |
| C3 | 0.7710 | 0.6250 | 0.5626 | 0.5098 | 0.7920 |
| C4 | 0.6359 | 0.5150 | 0.5150 | 0.4467 | 0.6820 |
| C4 + C3 | 0.7177 | 0.6050 | 0.5467 | 0.4931 | 0.7622 |

Fallback phải cộng C3+C4. Không tuyên bố C4 có relevance tốt nhất.
C4 thuần trả rỗng ở Q04, Q15; fallback C3 khôi phục các truy vấn này.
Gold M1: 150/150 câu có cờ confirmed; cần rà SG010, SG053, SG089 và kiểm tra chéo độc lập trước khi kết luận chất lượng.

Chi tiết thiết lập, toàn bộ query và 10 lĩnh vực: report/experiment_tables.tex; CSV latency: results/latency/latency_per_query.csv; báo cáo đầy đủ: report/main.pdf.

Tái tạo: `python -X utf8 -m scripts.build_indexes`, `python -X utf8 -m benchmarks.benchmark_latency`, `python -X utf8 -m scripts.plot_results --only latency size tdv --out report/figures`, `python -X utf8 -m scripts.render_report`.
