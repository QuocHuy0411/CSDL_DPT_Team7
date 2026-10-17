# Trạng thái tích hợp M1–M2–M3 (18/10/2026)

## Môi trường và đội triển khai

- Windows 11, Python 3.13.5; Herdr 0.9.3 với `HERDR_ENV=1` được kiểm tra thật. Lead ở pane `w2:p9`; đúng ba builder OpenCode của dự án ở `w2:pA` (data), `w2:pB` (retrieval), `w2:pC` (experiment). Không tạo builder/agent thứ tư, không dùng lại phiên hội thoại đang bị ứng dụng khác giữ.
- Ba phiên cũ hết hạn Space Bunny Free. Lead đã đổi trong chính các phiên đó sang DeepSeek V4.1 Flash Free và xác nhận bằng phản hồi/tác vụ thật. Gemini 2.5 Flash được thử nhưng khóa API Google hiện có bị từ chối (`API key not valid`); không đăng ký gói trả phí.
- Các nhãn, kết quả đo và chỉnh sửa giao diện mới nằm trong worktree chưa commit sau commit `d59baf7`; không `git reset/clean`, push, merge hay công khai raw. Phân quyền và API ở [api_contracts.md](api_contracts.md).

## Dữ liệu khóa và tài nguyên M1

- `data/raw/`: 33.759 TXT UTF-16 LE có BOM, đủ 10 thư mục lĩnh vực. Băm lại toàn bộ raw cuối đợt cho SHA256 tổng `76d541105d7ce1ad4c561d8cf7e875c4a3e5451ff6d139f283bf7d5eaa88701f`, không đổi. Không sửa raw.
- `data/corpus.jsonl`: lọc 1.011 bài dưới 100 đơn vị `len(content.split())`, 49 bản trùng toàn tập; chọn 2.500 bài, 250/lĩnh vực. SHA256 khóa `f42acebb1ac34084e859f378243145a3f8e981869350edb41a3dc27e86320fa7`. `scripts.verify_release --raw` đối chiếu cả 2.500 bài với raw, metadata/MD5/doc_id/phân bố và bốn chỉ mục: `technical_integrity: passed`.
- 20 query khóa SHA256 `a46100f23d937133a82d709b4b45ded5dcc0125520fe59ead647644e58be692e`, hai query/lĩnh vực. Pool BiMM hợp Top-20 TF-IDF/binary cosine/BM25 có 669 cặp; 134 cặp mẫu kiểm tra chéo. Qrels hiện **669/669 mang cờ confirmed** (385 nhãn 0, 141 nhãn 1, 143 nhãn 2), SHA256 `697fafc7a89e40a782428204a4b0df21cb0545f24cbb714477c9328cfc88893f`. Đã nhập [37 quyết định Team7 gửi và sửa thêm một cặp Q16 theo xác nhận của bạn](../results/label_review_resolution.json): nhãn cuối 36/37 trùng đề xuất AI, một khác có lý do. Tệp tự khai chấm M1 đã được [đối chiếu với qrels cuối](../results/m1_crosscheck.json): **126/134 trùng, 8 bất đồng**, nhưng chưa xác minh được người thứ hai độc lập hay phân xử tám cặp; vì vậy **chưa tuyên bố kiểm tra chéo hoàn tất**.
- Từ điển Viet74K GPL-2.0 ghim snapshot/giấy phép; 69.858 mục, SHA256 đầu ra `175e8b1bdd77e81d79b12c3d815cd9caa8f5a48f734c7a59f6124bc8cdc5d8ab`, 97,4233% mục ≤4 âm tiết, dài nhất 17. `scripts.clean_dict` xác thực nguồn và tạo byte-identical. M1 có chuẩn hóa, FMM/BMM/BiMM trên Trie, trace, stopword tầng 1 và guard; mẫu gold 150 câu/10 lĩnh vực đã khóa, **150/150 mang cờ confirmed**. Có 102 câu trùng hệt gợi ý BiMM; [rà gold](m1_gold_review.md) nêu SG010, SG053 là lỗi ranh giới rõ và SG089 bất nhất cần chốt quy ước.

## Tính toán và chỉ mục cuối

- C1–C4 đã dựng lại đồng thời sau khi chốt source M1; loader kiểm fingerprint source/tài nguyên. `scripts.verify_release --raw` nạp và kiểm search/explain của cả bốn. Sai số phân rã Cosine lớn nhất `5.55e-17`. TDV được đối soát bản tham chiếu trực tiếp trên tập nhỏ; tính one-shot trên toàn corpus 2.500 bài, không suy từ dấu TDV ra relevance.
- Dung lượng phục vụ C1/C2/C3/C4: **9.621.765 / 6.987.156 / 8.748.327 / 6.542.944 byte**. C4 có thêm 1.834.419 byte tệp phân tích TDV (tổng 8.377.363). C4 thuần giảm **32,00%** dung lượng phục vụ so với C1, đạt mục tiêu 30%; nhưng gói cần C3 để fallback là **15.291.271 byte**, tăng 58,92% so với C1, nên gói không đạt mục tiêu giảm dung lượng.
- Benchmark truy vấn: 20 query × 100 lần/cấu hình, warm-up 5 lượt, Top-10, chỉ mục nạp sẵn. Mean C1/C2/C3/C4 = **0,994 / 0,463 / 0,772 / 0,487 ms**; P95 = **1,992 / 1,034 / 1,737 / 1,355 ms**. Mean đều <30 ms trên máy đo, không phải cam kết cho máy khác. C4 thuần rỗng ở Q04 và Q15 (200/2.000 lần); fallback C3 trả kết quả ở cả hai, mean gộp 0,503 ms. Đây là số đo thời gian, **không** phải số đo chất lượng IR.
- Benchmark M1 Trie–set cùng thuật toán/từ điển/đầu vào: 750 đối soát FMM/BMM/BiMM, 0 mismatch; 15 unit tests riêng. Bản Trie đang dùng **chậm hơn** set khoảng 2–2,5 lần trên mẫu 150 câu và lát cắt 100 bài, vì tra lại từ gốc cho từng độ dài ứng viên. Xem [số liệu và giới hạn](trie_vs_set_benchmark.md); không đổi thuật toán trước khi nộp để tránh làm mất hiệu lực corpus/index/gold đã khóa.

## Kiểm thử và báo cáo

- Sau khi nhập 37 quyết định và bước sửa Q16, `python -X utf8 -m pytest -q --tb=short` cho **502 passed, 1 skipped**. AppTest bao phủ Home/Tokenizer/Search/Analytics trên chỉ mục cuối; `scripts.verify_release` trên corpus, 2.500 nguồn raw và bốn chỉ mục trả `technical_integrity: passed`.
- Evaluator đã chạy lại trên qrels cuối với cổng kỹ thuật thông qua: MAP C1/C2/C3/C4 = **0,7764 / 0,7667 / 0,7768 / 0,6431**, C4 fallback = **0,7248**. Gold chưa đổi; span F1 FMM/BMM/BiMM vẫn **0,9573 / 0,9637 / 0,9637**. Đây là số đo **tạm thời**, không phải bằng chứng mọi nhãn đúng ngữ nghĩa; ba câu gold SG010/SG053/SG089 và nguồn chấm M1 còn cần xác minh. Cổng hash/schema/preprocessing vẫn từ chối nhãn pending.
- Ba hình latency/index/TDV sinh từ benchmark. [Báo cáo LaTeX](../report/main.tex), bảng thí nghiệm và [tóm tắt M2](../report/m2_report.md) đã cập nhật theo qrels cuối; `report/main.pdf` **vẫn là bản cũ, không dùng làm kết quả cuối** vì môi trường hiện chưa có trình biên dịch LaTeX khả dụng. README nêu lệnh biên dịch khi có Tectonic/XeLaTeX.

## Đầu vào con người còn thiếu

1. Q16/`vn_f4f2b5ae9443c172` đã được Team7 xác nhận đổi thành mức 1. [37 quyết định và bước sửa đã nhập](../results/label_review_resolution.json); phiếu 37 cặp cũ gắn hash trước nhập, không nhập lại. Xác minh tệp M1 có thật do người thứ hai tự chấm hay không và phân xử [tám bất đồng](../results/m1_crosscheck.json); nếu không, ghi rõ giới hạn một người gán.
2. Duyệt SG010/SG053/SG089 bằng [phiếu ba câu riêng](../data/gold/review_SG010_SG053_SG089_sheet.html) và kiểm tra thêm gold tách từ, nhất là 102 câu trùng gợi ý. Chạy lại `evaluator.segmentation_eval` sau khi người gán xác nhận quyết định mới.

Không tuyên bố C4 có chất lượng tốt nhất: trên nhãn mới C3 có MAP cao nhất và C4 thuần thấp hơn. Nếu source tiền xử lý, corpus, query hoặc từ điển đổi, phải xác minh lại fingerprint/hash và xây lại artefact tương ứng; không âm thầm ghi đè nhãn người.
