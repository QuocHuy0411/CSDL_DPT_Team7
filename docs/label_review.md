# Rà soát chất lượng nhãn (17/10/2026)

Đây là **đề xuất kiểm tra bằng AI tại thời điểm trước khi Team7 gửi nhãn**, không phải phán quyết của người gán. Bản qrels trước rà đã lưu ở [`qrels.before_review_2026-10-18.json`](../data/ground_truth/mm/qrels.before_review_2026-10-18.json); gold chưa đổi. Tệp máy đọc được là [`results/label_review.json`](../results/label_review.json), gắn với SHA256 của qrels và gold tại thời điểm rà soát. Không nhập tệp đề xuất này bằng `scripts.annotate import` và không coi `suggested_grade` là `confirmed`.

## Cập nhật sau khi nhận nhãn Team7 (18/10/2026)

Đã nhập [37 quyết định do Team7 gửi](../data/ground_truth/mm/labels.json) (36 nhãn khác bản cũ); sau đó Team7 xác nhận [sửa thêm Q16/`vn_f4f2b5ae9443c172` từ 2 xuống 1](../data/ground_truth/mm/q16_amendment_2026-10-18.json). Tính theo nhãn cuối, 36/37 cặp trùng đề xuất AI, còn một cặp Q16/`vn_be3de050c8c6f373` được Team7 chấm 0 thay vì gợi ý 1; bài chỉ bàn vé bán kết, nên đây là lựa chọn hợp lý. Qrels hiện có SHA256 `697fafc7a89e40a782428204a4b0df21cb0545f24cbb714477c9328cfc88893f`; [biên bản đối chiếu](../results/label_review_resolution.json) lưu hash từng bước. Các bảng bên dưới là **ảnh chụp lịch sử trước khi nhập**, không phải danh sách 37 việc còn tồn.

Tệp tự khai chấm M1 134 cặp có trong workspace; [đối chiếu trên qrels cuối](../results/m1_crosscheck.json) cho 126 trùng, 8 bất đồng. Danh tính và tính độc lập của người chấm chưa được xác minh, tám cặp chưa phân xử; chưa được gọi là kiểm tra chéo hoàn tất. Ba câu gold SG010/SG053/SG089 vẫn chưa nhận quyết định mới; bảng F1 tách từ còn tạm thời.

## Quy tắc đọc

- Grade 2: bài trả lời trực tiếp ý định và các thực thể/điều kiện đặc trưng của query. Chỉ cùng lĩnh vực, cùng từ khóa hoặc cùng loại sự kiện không đủ.
- Grade 1: bài cung cấp một phần thông tin hữu ích nhưng thiếu điều kiện quan trọng.
- Grade 0: bài không giúp trả lời query cụ thể. Với câu hỏi về một vụ án, năm/giải đấu hoặc quốc gia nhất định, bài về vụ/năm/giải/quốc gia khác không được chấm 2.
- Đọc **toàn nội dung** trước khi chốt. Tiêu đề và khớp từ khóa chỉ dùng để ưu tiên rà soát; chú ý cả câu phủ định và nội dung ở cuối bài.

Tôi đã lướt tiêu đề của 177 cặp đang được chấm grade 2; lượt M2 tiếp theo lướt 134 cặp trong mẫu M1 và đọc toàn nội dung các cặp nghi vấn. Tổng cộng **37 cặp không trùng nhau** được đề nghị rà lại (17 ở bảng dưới, 20 ở [phiếu M2](m2_qrels_review.md)); đây là đề xuất AI, không phải 37 nhãn sai đã xác minh. Chưa kiểm chứng ngữ nghĩa toàn bộ 669 cặp và **chưa có điểm đồng thuận giữa hai người gán**. Mẫu 134 cặp M1 hiện là danh sách để kiểm tra chéo, không phải kết quả kiểm tra chéo.

## 17 cặp từ lượt rà grade 2 ban đầu

| Query | Doc ID | Nhãn đang có → gợi ý | Lý do chính |
| --- | --- | --- | --- |
| Q09 | `vn_7c91a99985430932`, `vn_7b7f48798157fbd4`, `vn_881ac2e2a3d5e02b`, `vn_ed62db66d49f897a`, `vn_ce64d3ce83c05594`, `vn_651d72b2e1c67a15` | 2 → 0 | Query hỏi vụ biển thủ 8,7 tỷ ở Hậu Giang; sáu bài nói về PMU18, Năm Cam hoặc mùa giải bóng đá khác. |
| Q13 | `vn_8bf8fea1831dae4f`, `vn_c06d3f31c0c6b2ce`, `vn_48e21771a3039483` | 2 → 0 | Lần lượt nói về cơ sở hạt nhân Iran, tình trạng khẩn cấp và chọn thủ tướng Iraq; không giải thích việc không tìm thấy vũ khí cấm ở Iraq. |
| Q14 | `vn_9f4e4f06d4e1315b` | 2 → 0 | Viện trợ lương thực Triều Tiên, không có nội dung xác nhận danh tính hài cốt. |
| Q16 | `vn_0e5283a87489dcf9`, `vn_5702926e97410dbc` | 2 → 0 | Chung kết Cúp Nhà vua Tây Ban Nha; xử lý đua xe trong mùa Tiger Cup. Không trả lời việc đăng cai vòng bảng và các trận chung kết Tiger Cup 2004. |
| Q17 | `vn_c2e14d8631c291cf` | 2 → 1 | Bài về trang phục trong phim Việt, nhưng không bàn cảnh hở hoặc tranh luận; chỉ liên quan một phần nếu người gán quyết định tính. |
| Q17 | `vn_5299564cd98633c6`, `vn_dbff556832e8f8ca` | 2 → 0 | Phỏng vấn đạo diễn không bàn cảnh nóng; bài còn lại giới thiệu phim Hàn Quốc. |
| Q19 | `vn_77dbfa9b0ba5764c`, `vn_9ce0946696a769b9` | 2 → 0 | Quy chế Internet và giới thiệu Windows Vista không giải thích virus đánh cắp mật khẩu Windows hay cách phòng tránh. |

Lượt rà mẫu M1 bổ sung [20 cặp khác và chứng cứ theo từng bài](m2_qrels_review.md), gồm cả grade 1 và grade 2 hiện có. [Phiếu tập trung 37 cặp](../data/ground_truth/mm/review_37_sheet.html) hợp hai lượt mà không lặp bốn cặp nằm trong cả hai phạm vi.

Gold M1: [SG010](../data/gold/segmentation_labels.jsonl) đang gộp `độc_đáo_để`; SG053 ghép `tra_của` sai ranh giới; SG089 tách hai lần xuất hiện của tên nước “Phần Lan” không nhất quán. Xem [rà gold chi tiết](m1_gold_review.md). Có 102/150 câu gold trùng toàn bộ gợi ý BiMM; điều đó có thể đúng sau khi người gán đọc lại, nhưng điểm F1 trên bộ này cần ghi rõ nguy cơ neo theo gợi ý. Không suy diễn rằng các nhãn còn lại sai chỉ vì giống gợi ý.

## Quy trình tham khảo và việc còn lại

Các bước nhắc “duyệt 37 cặp” dưới đây mô tả quy trình **đã thực hiện**; phiếu 37 cặp cũ mang hash qrels trước nhập, không dùng để nhập lại. Việc thực tế còn lại là xác minh người chấm M1, phân xử tám bất đồng và duyệt gold.

1. Người gán mở [phiếu riêng 37 cặp](../data/ground_truth/mm/review_37_sheet.html) và [phiếu ba câu gold](../data/gold/review_SG010_SG053_SG089_sheet.html): đều để trống câu trả lời cũ và không hiện gợi ý máy. Tìm theo ID ở bảng, đọc toàn bài/câu và tự quyết định nhãn cuối. Có thể dùng [phiếu qrels đầy đủ](../data/ground_truth/mm/annotation_sheet.html) hoặc [phiếu gold đầy đủ](../data/gold/annotation_sheet.html) để đọc rộng hơn. Phiếu 37 cặp chỉ xuất JSON khi điền đủ, ghi tên và xác nhận đã tự đọc; import còn kiểm tra hash bản qrels trước khi rà. JSON chỉ gồm 37 quyết định để nhập bằng `python -X utf8 -m scripts.annotate import --input <đường-dẫn-JSON>`; phiếu gold nhập bằng `python -X utf8 -m scripts.annotate_segmentation import --input <đường-dẫn-JSON>`. Không sửa corpus, query hay raw. Không nhập tệp đề xuất AI.
2. Phiếu [M1 chấm độc lập](../data/ground_truth/mm/m1_review_sheet.html) đã chuẩn bị đủ 134 cặp, hiển thị toàn bài nhưng **không hiển thị nhãn gốc**. Người M1 thật tự đọc, ghi tên, chấm 0/1/2 và xuất `m1_review_labels.json`. Không thể gọi việc cùng một AI chấm lại là hai người gán độc lập. Lưu tệp xuất riêng, không nhập nó trực tiếp vào qrels; chạy `python -X utf8 -m scripts.crosscheck_m1 compare --labels <đường-dẫn-JSON>` để tạo `results/m1_crosscheck.json`. Công cụ từ chối mẫu thiếu/trùng, hash cũ, nhãn sai hoặc tên người chấm trùng tên gốc; mọi bất đồng phải được người gán phân xử riêng. Nếu không có người M1 thật, ghi rõ giới hạn một người gán và **không** tuyên bố đã kiểm tra chéo.
3. Sau khi người gán nhập nhãn đã duyệt, chạy lại `python -X utf8 -m evaluator.evaluate --mode official --k 10 --out results/ablation`, `python -X utf8 -m evaluator.segmentation_eval` và `python -X utf8 -m scripts.render_report`. SHA256 của qrels/gold sẽ đổi; cập nhật hoặc đóng tệp review này và biên dịch lại PDF. Không dùng các CSV/PDF cũ làm kết quả cuối. Trước lúc nhận tệp người gán và quyết định bất đồng, **không chạy lại phép đo rồi gọi số cũ là đã duyệt**.

Số đo hiện có trong `results/ablation/` đã được tính lại trên qrels SHA256 `697fafc7a89e40a782428204a4b0df21cb0545f24cbb714477c9328cfc88893f`; `results/segmentation/` vẫn dùng gold chưa sửa. Cổng kỹ thuật của evaluator đã qua, nhưng kiểm tra chéo M1 và audit gold chưa đóng; vì vậy báo cáo phải gọi chúng là kết quả tạm thời.
