# Trạng thái tích hợp M1–M2–M3 (16/10/2026)

## Môi trường và đội triển khai

- Windows 11, Python 3.13.5; Herdr 0.9.3 với `HERDR_ENV=1` được kiểm tra thật. Lead ở pane `w2:p9`; đúng ba builder OpenCode của dự án ở `w2:pA` (data), `w2:pB` (retrieval), `w2:pC` (experiment). Không tạo builder/agent thứ tư, không dùng lại phiên hội thoại đang bị ứng dụng khác giữ.
- Ba phiên cũ hết hạn Space Bunny Free. Lead đã đổi trong chính các phiên đó sang DeepSeek V4.1 Flash Free và xác nhận bằng phản hồi/tác vụ thật. Gemini 2.5 Flash được thử nhưng khóa API Google hiện có bị từ chối (`API key not valid`); không đăng ký gói trả phí.
- Giữ nguyên worktree bẩn ban đầu; không `git reset/clean`, commit, push, merge hay công khai raw. Phân quyền giai đoạn cuối và API ở [api_contracts.md](api_contracts.md).

## Dữ liệu khóa và tài nguyên M1

- `data/raw/`: 33.759 TXT UTF-16 LE có BOM, đủ 10 thư mục lĩnh vực. Băm lại toàn bộ raw cuối đợt cho SHA256 tổng `76d541105d7ce1ad4c561d8cf7e875c4a3e5451ff6d139f283bf7d5eaa88701f`, không đổi. Không sửa raw.
- `data/corpus.jsonl`: lọc 1.011 bài dưới 100 đơn vị `len(content.split())`, 49 bản trùng toàn tập; chọn 2.500 bài, 250/lĩnh vực. SHA256 khóa `f42acebb1ac34084e859f378243145a3f8e981869350edb41a3dc27e86320fa7`. `scripts.verify_release --raw` đối chiếu cả 2.500 bài với raw, metadata/MD5/doc_id/phân bố và bốn chỉ mục: `technical_integrity: passed`.
- 20 query khóa SHA256 `a46100f23d937133a82d709b4b45ded5dcc0125520fe59ead647644e58be692e`, hai query/lĩnh vực. Pool BiMM hợp Top-20 TF-IDF/binary cosine/BM25 có 669 cặp; 134 cặp mẫu kiểm tra chéo. Qrels **0/669 confirmed**, không lấy category/evidence/gợi ý làm relevance.
- Từ điển Viet74K GPL-2.0 ghim snapshot/giấy phép; 69.858 mục, SHA256 đầu ra `175e8b1bdd77e81d79b12c3d815cd9caa8f5a48f734c7a59f6124bc8cdc5d8ab`, 97,4233% mục ≤4 âm tiết, dài nhất 17. `scripts.clean_dict` xác thực nguồn và tạo byte-identical. M1 có chuẩn hóa, FMM/BMM/BiMM trên Trie, trace, stopword tầng 1 và guard; mẫu gold 150 câu/10 lĩnh vực đã khóa, **0/150 nhãn người**. Xem [quy trình gold](segmentation.md).

## Tính toán và chỉ mục cuối

- C1–C4 đã dựng lại đồng thời sau khi chốt source M1; loader kiểm fingerprint source/tài nguyên. `scripts.verify_release --raw` nạp và kiểm search/explain của cả bốn. Sai số phân rã Cosine lớn nhất `5.55e-17`. TDV được đối soát bản tham chiếu trực tiếp trên tập nhỏ; tính one-shot trên toàn corpus 2.500 bài, không suy từ dấu TDV ra relevance.
- Dung lượng phục vụ C1/C2/C3/C4: **9.621.765 / 6.987.156 / 8.748.327 / 6.542.944 byte**. C4 có thêm 1.834.419 byte tệp phân tích TDV (tổng 8.377.363). C4 thuần giảm **32,00%** dung lượng phục vụ so với C1, đạt mục tiêu 30%; nhưng gói cần C3 để fallback là **15.291.271 byte**, tăng 58,92% so với C1, nên gói không đạt mục tiêu giảm dung lượng.
- Benchmark truy vấn: 20 query × 100 lần/cấu hình, warm-up 5 lượt, Top-10, chỉ mục nạp sẵn. Mean C1/C2/C3/C4 = **0,994 / 0,463 / 0,772 / 0,487 ms**; P95 = **1,992 / 1,034 / 1,737 / 1,355 ms**. Mean đều <30 ms trên máy đo, không phải cam kết cho máy khác. C4 thuần rỗng ở Q04 và Q15 (200/2.000 lần); fallback C3 trả kết quả ở cả hai, mean gộp 0,503 ms. Đây là số đo thời gian, **không** phải số đo chất lượng IR.
- Benchmark M1 Trie–set cùng thuật toán/từ điển/đầu vào: 750 đối soát FMM/BMM/BiMM, 0 mismatch; 15 unit tests riêng. Bản Trie đang dùng **chậm hơn** set khoảng 2–2,5 lần trên mẫu 150 câu và lát cắt 100 bài, vì tra lại từ gốc cho từng độ dài ứng viên. Xem [số liệu và giới hạn](trie_vs_set_benchmark.md); không đổi thuật toán trước khi nộp để tránh làm mất hiệu lực corpus/index/gold đã khóa.

## Kiểm thử và báo cáo

- `python -X utf8 -m pytest -q --tb=short`: **485 passed (39,08 s)** sau tất cả sửa đổi. `python -X utf8 -m compileall -q engine scripts evaluator app tests benchmarks` và `git diff --check` chạy cuối; cảnh báo LF/CRLF của Git không phải lỗi. AppTest tự động bao phủ Home/Tokenizer/Search/Analytics trên chỉ mục cuối; Search trả 10 kết quả và 10 bảng explain trong lượt smoke test.
- Bộ đánh giá official **từ chối** qrels pending; chế độ development ghi `metrics_status=unmeasured`, không xuất CSV P/R/F1/MAP/NDCG toàn số 0. `evaluator.segmentation_eval` ghi `official: false`, `metrics_status: unmeasured` khi 0/150 gold. Các cổng hash/schema/preprocessing và việc AP cần ranking đầy đủ được kiểm bằng test.
- Ba hình latency/index/TDV sinh từ benchmark mới. [Báo cáo LaTeX](../report/main.tex) biên dịch bằng Tectonic thành [PDF 28 trang](../report/main.pdf), có bảng toàn 20 query và 10 lĩnh vực, ghi rõ chưa đo chất lượng; chỉ còn cảnh báo dàn trang underfull và font hệ thống Windows, không có lỗi TeX/overfull. [Tóm tắt M2](../report/m2_report.md) cùng nguồn số liệu. Tectonic đã tải trong `.test-runs/` (bị ignore); README nêu lệnh cài/chạy/biên dịch.

## Đầu vào con người còn thiếu

1. Gán xác nhận relevance cho 669 cặp trong `data/ground_truth/mm/annotation_sheet.html`, dùng `scripts.annotate` để import/status/review/confirm. Mẫu 134 cặp để M1 kiểm tra chéo. Sau đó chạy `evaluator.evaluate --mode official` trên corpus/query/index đã khóa.
2. Gán ranh giới từ cho 150 câu trong `data/gold/annotation_sheet.html`, dùng `scripts.annotate_segmentation` để import. Sau đó chạy `evaluator.segmentation_eval` để có word-span P/R/F1 thật.

Không tuyên bố C4 có chất lượng tốt nhất hoặc F1/MAP khi hai bộ nhãn còn pending. Nếu source tiền xử lý, corpus, query hoặc từ điển đổi sau khi gán nhãn, phải xác minh lại fingerprint/hash và xây lại artefact tương ứng; không âm thầm ghi đè nhãn người.
