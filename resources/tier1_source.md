# Tầng 1: nguồn, chính sách và số mục thực tế

Danh sách do nhóm biên soạn theo chức năng ngữ pháp; không sao chép danh sách không rõ giấy phép. Không phải nhãn POS tự động hay gold đã được chuyên gia ngôn ngữ xác nhận. C2 dùng stoplist tĩnh riêng, không trộn với C3.

| Đại lượng | Số mục duy nhất |
|---|---:|
| Danh sách ngữ pháp | 181 |
| Whitelist bảo vệ | 110 |
| Giao trực tiếp hai danh sách | 49 |
| Hiệu hai tập trực tiếp | 132 |
| Danh sách thực sự bị loại, sau cả bảo vệ hợp từ | **124** |

Có dòng lặp trong whitelist; đếm tập canonical, không lấy số dòng. Canonical = NFC + casefold, khoảng trắng nối gạch dưới. Số mục và SHA256 lấy bằng `Tier1StopwordFilter.metadata()`; không nhúng fingerprint tự tham chiếu vào tài liệu.

## Quy tắc

Inventory gồm đại từ, từ chỉ định, giới từ, trợ từ/trợ động từ và một số từ nhấn mạnh/liên kết diễn ngôn. Đây là heuristic không gán POS theo ngữ cảnh: từ đa nghĩa có thể bị lọc cả ở cách dùng thực từ. Không khẳng định lọc sẽ tăng MAP khi chưa có relevance.

Guard gồm phủ định, nguyên nhân/điều kiện/mục đích, tương phản/lựa chọn, nghi vấn, phạm vi/lượng từ, so sánh, thứ tự thời gian và tình thái. Với G là whitelist, thành phần bảo vệ là C = {c thuộc g.split('_') với g thuộc G, đồng thời c thuộc G}. Token t được bảo vệ khi t thuộc G hoặc có thành phần thuộc C.

Ví dụ `không` bảo vệ `không_thể`; nhưng `thể` chỉ là thành phần trong guard `có_thể`, bản thân không thuộc G, nên không tự bảo vệ `thể_thao`. Chính sách này vẫn có thể bảo vệ từ đa nghĩa; không coi nó là suy luận ngữ nghĩa. M2 dùng đúng cùng policy để bảo vệ chiều TDV, không chỉ giữ term query sau khi chiều đã bị xóa.

Ví dụ bắt buộc: `['bệnh','không','lây','của','người']` → `['bệnh','không','lây','người']`. Guard thắng stopword, thứ tự và tần suất được giữ nguyên. Inventory181 mục nằm trong quy mô150–200 của kế hoạch; số thực sự loại chỉ124 do guard, không bù thực từ để tăng con số.

## Truy xuất

```powershell
python -X utf8 -c "from engine.tokenizer.stopwords import get_default_filter; print(get_default_filter().metadata())"
python -X utf8 -m pytest -q tests/test_tier1_filter.py
```

VietTreebank là nguồn khái niệm phân loại POS trong kế hoạch nhóm; danh sách cụ thể là quy tắc dự án, không được gắn nhãn là trích xuất nguyên bản VietTreebank. Cần rà soát theo lỗi thực tế sau khi có qrels. File inventory và whitelist có nhóm/chú thích; metadata kiểm tra hash trước khi nạp lại.
