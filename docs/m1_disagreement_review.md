# Đề xuất phân xử 8 bất đồng qrels–M1 (18/10/2026)

Đây là **ý kiến rà soát của lead, chưa phải quyết định gán nhãn**. Tôi đọc toàn văn tám bài trong corpus và so với đúng câu query, không dùng category làm relevance. Không sửa `qrels.json`, `m1_review_labels.json`, corpus, query hoặc raw. Bản [đối chiếu máy](../results/m1_crosscheck.json) gắn với qrels SHA256 `697fafc7a89e40a782428204a4b0df21cb0545f24cbb714477c9328cfc88893f` và tệp M1 SHA256 `36247b4b5f92ab15ccb3b4eccc4294b82247d11fc0d2ad6c6291454fa9d71f5d`: 126/134 trùng, 8 bất đồng. Tệp M1 tự khai người chấm “Hiep”; phần mềm không chứng minh được danh tính hoặc tính độc lập. Người gán phải xác nhận nguồn tệp và chốt từng bất đồng trước khi gọi là đã kiểm tra chéo.

Quy ước đề xuất: 0 = không trả lời chủ đề cụ thể; 1 = liên quan một phần/bối cảnh; 2 = trả lời trực tiếp trọng tâm query. Query dạng “và” gồm nhiều ý không tự động được mức 2 khi bài chỉ bàn một ý.

| Query / doc ID | Qrels / M1 | Lead đề xuất | Căn cứ đọc toàn bài |
|---|---:|---:|---|
| Q03 / `vn_706ae3e191d963a8` | 1 / 0 | **0 (M1)** | Query hỏi có nên tha thứ cho **người vợ** ngoại tình. Bài “Nếu là An…” khuyên một người vợ rời **người chồng** phản bội; đảo vai chủ thể và không bàn trường hợp vợ ngoại tình. |
| Q04 / `vn_10288e4d8ae0d55b` | 1 / 2 | **2 (M1)** | Query rộng “tâm sự gia đình”; toàn bài là thư tâm sự/khuyên Anh Khoa về hôn nhân, vợ, con và ly hôn, đáp ứng trực tiếp. |
| Q08 / `vn_a51c84de855dc95c` | 1 / 0 | **0 (M1)** | Query hỏi lãi suất cơ bản và chính sách tiền tệ **Fed** ảnh hưởng thị trường. Bài chỉ bàn đề nghị bỏ thuế trái phiếu Chính phủ **Việt Nam** và tác động lãi suất trong nước, không có Fed. |
| Q10 / `vn_a7ab1dbd1332334b` | 1 / 2 | **1 (giữ qrels)** | Bài cùng vụ PMU18, mô tả các vụ chém cán bộ và xô xát từng bị bỏ qua rồi điều tra lại, nhưng không xác định **đường dây chạy án** hoặc cơ chế/tác nhân chạy án. Hữu ích làm bối cảnh, chưa đủ mức 2. |
| Q16 / `vn_cbe182ef26813cda` | 1 / 2 | **1 (giữ qrels)** | Bài nêu sân Mỹ Đình tổ chức ba trận vòng bảng và TP HCM bán vé các trận ở sân Thống Nhất; không nói diễn biến hay kết quả **các trận chung kết** Tiger Cup 2004. Chỉ đáp ứng phần đăng cai. |
| Q18 / `vn_161f379098b8499f` | 1 / 2 | **2 (M1)** | Query rộng về ca khúc và chương trình biểu diễn âm nhạc của nghệ sĩ; bài tường thuật liveshow “Nắng lên” của Thanh Lam, liệt kê nhiều ca khúc và đánh giá phần biểu diễn. |
| Q19 / `vn_dfcce0e16a08fae9` | 1 / 0 | **1 (giữ qrels)** | Bài hỏi đèn ổ cứng nháy trên Windows; phần hồi âm nêu khả năng **virus**, khuyên cài/cập nhật antivirus và bản vá Windows. Không bàn đánh cắp mật khẩu, nhưng có phần “virus và cách phòng tránh”, nên không phải 0; cũng không đủ 2. |
| Q20 / `vn_2a683b464546fcd2` | 1 / 2 | **1 (giữ qrels)** | Bài phân tích sự cố dịch vụ di động VinaPhone và cách doanh nghiệp bù/không bù cho khách hàng; đáp ứng mảng **thông tin di động**, không bàn **dịch vụ phần mềm**. |

Tổng hợp đề xuất: 4 cặp theo M1 (Q03, Q04, Q08, Q18), 4 cặp giữ qrels (Q10, Q16, Q19, Q20). Đây không phải tỷ lệ đúng/sai của từng người trên toàn bộ 134 cặp. Người gán mở [phiếu qrels](../data/ground_truth/mm/annotation_sheet.html), tìm theo ID, đọc lại bài và quyết định; nếu sửa, dùng công cụ `scripts.annotate import` với tệp quyết định mới gắn hash qrels hiện tại, rồi chạy lại `evaluator.evaluate` và `scripts.render_report`. Không nhập bảng đề xuất của lead thành nhãn confirmed.
