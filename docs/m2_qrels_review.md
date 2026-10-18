# Rà soát M2 trên mẫu M1 (17/10/2026)

> Ảnh chụp đề xuất AI trước khi nhập nhãn Team7. Đã nhập 37 quyết định và sửa riêng Q16/`vn_f4f2b5ae9443c172` thành mức 1 theo xác nhận của Team7 ngày 18/10/2026. Xem [biên bản cuối](../results/label_review_resolution.json); cột “Hiện tại → đề xuất” dưới đây chỉ mô tả trạng thái lúc audit, không phải nhãn hiện tại. Nguồn chấm độc lập của tệp M1 vẫn chưa được xác minh.

Đây là **đề xuất rà soát bằng AI**, không phải nhãn của người thứ hai, không phải kết quả kiểm tra chéo và **không được nhập tự động** vào `qrels.json`. Tôi giữ nguyên mọi nhãn hiện tại. Người gán phải đọc toàn bài trong phiếu và tự quyết định từng cặp. Việc một agent được giao vai M1 không tạo ra một người gán độc lập về mặt nghiên cứu.

Nguồn kiểm tra: `data/ground_truth/mm/m1_review_sample.json`, `pool.json`, `qrels.json`, `data/queries.json`. Mẫu có 134/669 cặp, với phân bố nhãn hiện tại: 60 mức 0, 29 mức 1, 45 mức 2. Tôi lướt tiêu đề của 134 cặp, rồi đọc toàn nội dung các cặp nghi vấn nêu dưới đây; **không** tuyên bố đã thẩm định ngữ nghĩa toàn bộ 134 hay 669 cặp. Bốn cặp trong mẫu đã có ở [phiếu 17 cặp](label_review.md): Q14/`vn_9f4e4f06d4e1315b`, Q09/`vn_ce64d3ce83c05594`, Q16/`vn_5702926e97410dbc`, Q09/`vn_ed62db66d49f897a`. Không tính lặp chúng ở bảng mới.

Quy ước đề xuất: 2 = bài trả lời trực tiếp nhu cầu của query; 1 = cung cấp một phần thông tin hữu ích nhưng thiếu thành tố chính; 0 = chỉ cùng chủ đề/từ khóa hoặc nói vụ, thời điểm, đối tượng khác. Các trường hợp 2→1 là đánh giá thận trọng: người gán có thể giữ 2 nếu chứng minh được phần nội dung trực tiếp trả lời cả query.

| Query | Doc ID | Hiện tại → đề xuất | Chứng cứ từ toàn bài và lý do cần kiểm tra |
| --- | --- | --- | --- |
| Q01 `lương tối thiểu` | `vn_873c8ffcdb012aec` | 1 → 0 | Bài về tài xế/công nhân đình công và đề xuất cộng 80.000 đồng **lương cơ bản**; không trình bày mức **lương tối thiểu**. |
| Q03 `có nên tha thứ cho người vợ ngoại tình` | `vn_bbca0beefc72a774` | 2 → 0 | “Chị Hồng hãy hỏi thẳng chồng” bàn nghi ngờ **người chồng** ngoại tình và bao cao su; khác vai giới tính và không trả lời có nên tha thứ cho **vợ**. |
| Q12 `mỡ máu cao ... biến chứng gì` | `vn_4fcc7d8950f09351` | 2 → 1 | “Chữa trị rối loạn trí nhớ” chỉ một đoạn nhắc cholesterol cao/xơ vữa cản tuần hoàn và nguy cơ tim, não; bài chính là cách chữa suy giảm trí nhớ, chưa trình bày hệ thống biến chứng mỡ máu cao. |
| Q13 `vì sao Mỹ vẫn chưa tìm thấy vũ khí cấm ở Iraq` | `vn_fdc2d967912b1682` | 2 → 0 | Bài lịch sử quan hệ Mỹ–Iraq 1980–1983, có đoạn Iraq dùng vũ khí hóa học trong chiến tranh với Iran; không giải thích cuộc **tìm kiếm sau chiến tranh Iraq** hoặc vì sao không tìm thấy. |
| Q13 | `vn_3db3a7acff470e96` | 2 → 0 | “Tổng thống Mỹ đe doạ Syria vì vấn đề Iraq” nói can thiệp của Syria trước bầu cử Iraq; không bàn vũ khí cấm hay cuộc tìm kiếm. |
| Q13 | `vn_b8abbd2e2108c902` | 1 → 0 | “Binh sĩ Mỹ bị doạ chặt đầu tại Iraq đã được thả” là tin bắt cóc con tin, không cung cấp phần nào về vũ khí cấm. |
| Q14 `xác nhận danh tính hài cốt lính Mỹ...` | `vn_4afe3144432aef0c` | 1 → 0 | Chuyến thăm Trung Quốc của Chủ tịch Quốc hội Bắc Triều Tiên bàn đàm phán hạt nhân; không nói hài cốt hoặc xác nhận danh tính. |
| Q16 `Việt Nam đăng cai một vòng bảng Tiger Cup 2004 và các trận chung kết diễn ra thế nào` | `vn_c19512de83b56291` | 2 → 0 | “Đội tuyển Việt Nam bỏ JVC Cup và Cup Độc lập” là lịch tập trung/giao hữu của đội tuyển; chỉ ghi sẽ dự Tiger Cup ngày 5/12, không bàn địa điểm tổ chức hoặc chung kết. |
| Q16 | `vn_3fd2aa9cd9e2ec02` | 2 → 1 | Chân dung đội Thái Lan trước Tiger Cup; đoạn cuối cho biết Thái Lan ở bảng B gặp **chủ nhà Malaysia**, nên chỉ có thể hỗ trợ bối cảnh phân bảng, không trả lời Việt Nam đăng cai hay trận chung kết. |
| Q16 | `vn_cbe182ef26813cda` | 2 → 1 | “Căng thẳng 'cuộc chiến' vé Tiger Cup” ghi Mỹ Đình tổ chức ba trận vòng bảng và TP HCM bán vé các ngày 7, 9/12; có phần hữu ích về đăng cai nhưng không diễn biến chung kết. |
| Q16 | `vn_0b9a17293edc527c` | 2 → 0 | Trận giao hữu Việt Nam–Hải Phòng 1–2 và lựa chọn đội hình trước Tiger Cup; không nói khâu đăng cai/chung kết. |
| Q16 | `vn_51d1a60e43e5afbb` | 2 → 0 | Tỷ số Malaysia–Đông Timor và Myanmar–Philippines ở bảng B, không nói Việt Nam đăng cai hoặc chung kết. |
| Q16 | `vn_f4f2b5ae9443c172` | 2 → 1 | “TP HCM đã sẵn sàng cho Tiger Cup” xác nhận lễ khai mạc bảng A tại sân Thống Nhất và trận Lào–Indonesia; không có kết quả/trình tự chung kết. |
| Q16 | `vn_be3de050c8c6f373` | 2 → 1 | Bài giá vé **bán kết** ở Indonesia, Malaysia, Myanmar, Singapore; giúp hiểu vòng knock-out nhưng không trả lời cụ thể các trận **chung kết** hoặc vai trò đăng cai của Việt Nam. |
| Q18 `ca khúc và chương trình biểu diễn âm nhạc của nghệ sĩ` | `vn_2eef9edaaa5cafcb` | 2 → 0 | Phỏng vấn đạo diễn về chương trình **hài** “Gặp nhau cuối tuần”, “Gala cười”, kịch bản và diễn viên; không nói ca khúc/chương trình âm nhạc. |
| Q19 `virus máy tính đánh cắp mật khẩu Windows và cách phòng tránh` | `vn_354a82dbd6c4f7a1` | 2 → 0 | Jitux.A qua MSN và PE_QUIS.A qua e-mail; bài không nêu đánh cắp mật khẩu Windows hay hướng phòng tránh tương ứng. |
| Q19 | `vn_58907f6a11f7fc42` | 2 → 0 | Hỏi đáp lỗi `illegal operation`/`KERNEL32.DLL`; có người dùng nghi virus nhưng hồi âm nói cấu hình, dọn ổ và cài lại Windows, không nói đánh cắp mật khẩu. |
| Q19 | `vn_dfcce0e16a08fae9` | 2 → 1 | Hỏi đáp đèn ổ cứng nháy và nghi virus, khuyên cập nhật chống virus/bản vá; có phần phòng vệ chung nhưng không có virus đánh cắp mật khẩu. |
| Q19 | `vn_5be1a9708bdbdeba` | 2 → 1 | Phân tích worm XRobots qua Yahoo Messenger và cách gỡ; **không có chứng cứ cụ thể** nó đã lấy mật khẩu Windows. Chỉ hữu ích một phần về phát hiện/phòng chống mã độc. |
| Q19 | `vn_1f0ca5a8d0e2dd44` | 2 → 1 | Trojan/bot khai thác Windows chưa cập nhật, có thể “ăn cắp thông tin” và khuyến cáo kiểm tra hệ thống; không nêu mật khẩu Windows cụ thể hay đường đánh cắp. |

Ngoài bảng trên, những cặp có grade 0 không tự động đúng chỉ vì tiêu đề không khớp; nội dung cuối bài có thể liên quan. Không thay thế đọc toàn bài bằng bảng này. Đặc biệt các query rộng Q07 (`cổ phiếu`), Q15 (`vô địch`), Q18 và Q20 cho phép nhiều dạng văn bản hợp lệ; không hạ nhãn chỉ vì khác lĩnh vực hẹp.

## Đối chiếu độc lập đúng nghĩa

Phiếu mù 134 cặp được tạo bằng `python -X utf8 -m scripts.crosscheck_m1 export` tại `data/ground_truth/mm/m1_review_sheet.html`. Nó chứa toàn văn nhưng không chứa nhãn gốc. Chỉ **người M1 thực** tự đọc, tự chấm và xuất tệp `m1_review_labels.json` với tên mình; một AI cùng phiên không thể tự nhận là người thứ hai. Sau đó dùng `python -X utf8 -m scripts.crosscheck_m1 compare --labels <tệp JSON>` để tạo ma trận grade và danh sách bất đồng. Công cụ kiểm tra mẫu ngẫu nhiên tái tạo được, schema/hash, đủ đúng 134 cặp, điểm 0/1/2, người chấm khác tên người đầu, và không ghi vào qrels. Checkbox/tên tự khai **không chứng minh** danh tính hoặc độc lập. Từng bất đồng phải được người gán phân xử; nếu chưa có tệp M1 thật, báo cáo phải ghi **chỉ một người gán, chưa kiểm tra chéo**.
