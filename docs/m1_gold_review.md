# Rà soát gold tách từ M1 (17/10/2026)

Đây là **đề xuất kiểm tra bằng AI, không phải nhãn của người gán độc lập**. Bản `data/gold/segmentation_labels.jsonl` vẫn giữ nguyên 150/150 nhãn `confirmed`; SHA256 tại lúc rà soát là `8f47eaa66342fe9531f556315bed0145c6bb5f22c71414984ae339ee5487b563`. Chưa có quyền coi các sửa gợi ý bên dưới là gold đã được người xác nhận.

Tôi đã đọc chuỗi nhãn của cả 150 câu, đối chiếu câu gốc ở các trường hợp đáng ngờ, và kiểm tra tập 78 token ghép có âm tiết đầu/cuối trùng nhóm hư từ thường gặp. Cách dò này không bảo đảm tìm mọi lỗi, cũng không thay thế vòng chấm độc lập. 102/150 câu gold trùng hoàn toàn gợi ý BiMM; sự trùng khớp không tự chứng minh lỗi hay thiếu duyệt, nhưng cần nêu nguy cơ neo vào gợi ý khi diễn giải F1.

| ID | Mức độ | Đoạn trong câu gốc | Gold hiện tại | Đề xuất để người gán quyết định |
| --- | --- | --- | --- | --- |
| SG010 | Lỗi ranh giới rõ | “các chất liệu độc đáo để trang trí” | `chất_liệu độc_đáo_để trang_trí` | `chất_liệu độc_đáo để trang_trí`; `để` là từ chỉ mục đích, không thuộc tính từ `độc đáo`. |
| SG053 | Lỗi ranh giới rõ | “Theo điều tra của cơ quan công an” | `theo điều tra_của cơ_quan công_an` | `theo điều_tra của cơ_quan công_an`; cụm động từ `điều tra` và giới từ `của` không tạo một từ `tra_của`. |
| SG089 | Bất nhất cần chốt quy ước | “Tổng thống Phần Lan ... Tổng thống Phần Lan” | `tổng_thống phần_lan ... tổng_thống phần lan` | Chọn một quy ước nhất quán cho cùng tên nước trong cùng câu; nếu coi tên nước là một từ, dùng `phần_lan` ở cả hai lần. |

Phiếu [rà riêng ba câu](../data/gold/review_SG010_SG053_SG089_sheet.html) để trống câu trả lời và ẩn cả nhãn cũ lẫn gợi ý máy. Có thể tạo lại bằng PowerShell từ thư mục dự án:

```powershell
python -X utf8 -m scripts.annotate_segmentation export --only SG010 SG053 SG089 --out data/gold/review_SG010_SG053_SG089_sheet.html
```

Người gán thật đọc câu và nguồn, tự quyết định từng ranh giới, nhập tên và xác nhận đã kiểm tra trong phiếu, rồi xuất JSON. Chỉ sau bước đó mới nhập bằng `python -X utf8 -m scripts.annotate_segmentation import --input <đường-dẫn-JSON>`; không nhập văn bản đề xuất này như nhãn. Nếu người gán không đồng ý, ghi quyết định và lý do, vẫn để bản gold như cũ. Sau bất kỳ thay đổi nhãn nào, chạy lại đánh giá M1 và cập nhật hash trong báo cáo/PDF; không tái sử dụng kết quả cũ.

Phiếu focused không hiển thị gợi ý, nhưng trình duyệt có thể khôi phục bản nháp người gán đã nhập trước đó trong `localStorage` của cùng phiếu; người gán cần kiểm tra lại toàn câu trước khi xác nhận. Không dùng tác nhân AI này như “người M1 thứ hai” để khai báo đồng thuận liên-người-gán.
