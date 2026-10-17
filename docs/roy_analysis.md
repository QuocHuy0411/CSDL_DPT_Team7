# Phân tích kỹ thuật `roy-a/Roy_VnTokenizer` và quyết định từ điển

Tài liệu này ghi lại khảo sát công cụ tách từ tham chiếu `roy-a/Roy_VnTokenizer`
(Ngày 1–2 theo kế hoạch M1) và **quyết định giấy phép** quan trọng nhất: dự án
**không** trích xuất hay phân phối từ điển của kho này vì thiếu giấy phép.

> Trạng thái nguồn: kho Roy **không được clone vào workspace này**. Phân tích dưới
> đây dựa trên nội dung công khai của repository (README, cây thư mục) truy cập ngày
> **2026-10-16**, đối chiếu với kế hoạch M1 trong `_docx.txt` và với
> `resources/dictionary_source.json`. Mọi con số hiệu năng trong mục 3 là **số tác
> giả tự báo**, không phải số dự án tự đo lại.

## 1. Kho tham chiếu có gì

`roy-a/Roy_VnTokenizer` (Python) tự mô tả là "Vietnamese
tokenizer (Maximum Matching and CRF)". Cấu trúc cấp cao:

```
roy-a/Roy_VnTokenizer
├── data/         # từ điển + 10 file IOB2 (5 train, 5 test)
├── literature/   # các bài báo tham khảo (viet1..viet6)
├── scripts/      # vn_tokenizer.py, vn_tokens_evaluate.py, runExperiments.py, model.pkl
├── README
└── README.2
```

Tác giả (Anindya Roy, 2014) chọn **Maximum Matching (MM)** làm nền vì đây là cơ sở
của hai hệ chuẩn `vnTokenizer` và `PVnSeg`; `JVnSegmenter` (CRF/SVM) là hướng khác
và **không được dùng lại** — tác giả nêu rõ "All code was developed by me", nghĩa là
mã nguồn viết tay chứ không sao chép hệ khác.

## 2. Thuật toán của Roy (theo README)

* **MM cơ sở:** chọn cách phân đoạn cho ra **ít từ nhất**; tại mỗi vị trí lấy chuỗi
  âm tiết dài nhất có trong lexicon làm một từ, nhảy con trỏ qua chuỗi đó rồi lặp.
  Greedy, dựa trên giả định "từ dài thường đúng hơn từ ngắn".
* Hai loại nhập nhằng được nêu tên: **cross ambiguity** (a, b, a_b đều là từ) và
  **overlap ambiguity** (a, b, c với a_b, c, a, b_c đều là từ ⇒ cả (a,b_c) và (a_b,c)
  đều hợp lệ).
* **MM+ (enhanced Maximum Matching)** — phần mở rộng chính của tác giả:
  1. một bước **regex parsing** khởi đầu;
  2. **lookahead hai âm tiết** trước khi quyết định kết thúc một từ;
  3. **giải nhập nhằng bằng xác suất unigram** thay vì bigram (tác giả chọn unigram
     vì `viet5` cho thấy bigram thiếu dữ liệu để ước lượng).
* Đánh giá bằng P/R/F trên 5-fold CV, tác giả báo **92,9 / 92,6 / 92,7 %**;
  MM+ hơn MM khoảng **0,5 %**. Bảng `runExperiments.py` (theo README): MM thuần
  0,919 → thêm locations/person names 0,921 → thêm lexicon 0,922 → MM+ 0,927 (F-ratio).
* Script: `vn_tokenizer.py` cài MM ở khoảng dòng 264–302 và MM+ ở 306–409;
  `vn_tokens_evaluate.py` tính P/R/F bằng **dynamic time warping / LCS**;
  `model.pkl` là pickle chứa lexicon + xác suất unigram.

## 3. Dữ liệu của Roy (điểm mấu chốt về giấy phép)

Lexicon của Roy được **ghép** từ nhiều nguồn:

* một **từ điển tiếng Việt**,
* danh sách **địa danh**,
* danh sách **tên người**,

và theo README, ba tệp này lấy từ gói **JVnSegmenter** trên SourceForge
(`VNDic_UTF-8.txt`, `vnlocations.txt`, `vnpernames.txt`), được tác giả chép lại vào
`data/`. Ngoài ra còn 5+5 file IOB2 (`trainX.iob2`/`testX.iob2`) do con người gán
nhãn, trung bình mỗi fold test khoảng 1.561 câu / 25.498 từ / 35.352 âm tiết.

**Vấn đề:** bản thân kho `roy-a/Roy_VnTokenizer` **không có tệp LICENSE** và GitHub
API báo `license = null`. Khi chưa xác định được giấy phép, dự án **không có căn cứ
rõ ràng để phân phối lại danh sách từ**. Ghi nhận này đã được lưu trong `resources/dictionary_source.json`
(`why_this_source`) và `scripts/clean_dict.py`.

## 4. Quyết định và lý do

**Quyết định:** *không* trích xuất từ điển Roy; dùng **Viet74K** (kho
`duyet/vietnamese-wordlist`, giấy phép **GPL-2.0**, ghim commit
`688b0953…`, snapshot kèm LICENSE đã vendor) làm nguồn lexicon, làm sạch còn **69.858
mục** (`resources/dict.txt`). Chi tiết nguồn và hash ở
`resources/dictionary_source.json`.

Lý do:

1. **Giấy phép:** từ điển Roy không có LICENSE trong kho khảo sát ⇒ dự án không phân phối lại; gói
   JVnSegmenter cũng cần xem giấy phép riêng của nó. Viet74K có grant GPL-2.0 rõ ràng.
2. **Kỹ thuật:** Viet74K có nhiều mục đa âm tiết — đúng thứ Maximum Matching cần.
3. **Tái lập:** Viet74K được ghim commit và kiểm SHA256 mỗi lần dựng, chạy offline.

**Không dùng `model.pkl`** của Roy: pickle là định dạng thực thi, không nên nạp tệp
lạ, và mô hình gắn với dữ liệu/giấy phép không rõ. Dự án tự viết lõi Trie MM.

### Cái gì được kế thừa về mặt ý tưởng (không phải mã)

Kế thừa **ý tưởng thuật toán** (không bản quyền): MM greedy, biến thể FMM/BMM/BiMM,
xem xét nhập nhằng cross/overlap, xử lý dấu câu, âm tiết OOV tách riêng. **Không**
kế thừa: regex parsing và giải nhập nhằng unigram của MM+, thuật toán CRF, và tất
nhiên không dùng dữ liệu của họ.

Dự án cài lại độc lập trong `engine/tokenizer/`:

* Trie theo **âm tiết** (mỗi cạnh một âm tiết) thay vì theo ký tự;
* cửa sổ = **mục dài nhất trong từ điển** (hiện là 17) thay vì cố định `Lmax = 4`;
* BiMM hòa theo thứ tự *ít token → ít token đơn → ít OOV → BMM*;
* có API `trace()` để soi từng bước cửa sổ;
* chuẩn hóa bỏ URL/email/HTML/chữ số và coi **dấu câu là biên cứng**;
* **không dùng xác suất unigram/bigram** (không có dữ liệu gán nhãn để ước lượng ở
  thời điểm dựng). Điều này nhất quán với nhận xét trong README Roy/`viet5` rằng lợi
  ích của bigram chỉ ~0,2–0,3 %.

Ngoài ra, ở tầng sau dự án không chỉ phân đoạn mà còn có **bộ lọc hư từ tầng 1** với
`guard_whitelist` bảo vệ phủ định/logic (xem `resources/tier1_source.md`); Roy không
có bước này.

## 5. Hệ quả cho benchmark Trie-vs-Set

Vì không dùng MM+ của Roy, dự án chỉ cài MM thuần (FMM/BMM/BiMM) trên Trie. So sánh
hiệu năng Trie với tra cứu `set` cùng thuật toán được đo riêng, xem
[`docs/trie_vs_set_benchmark.md`](trie_vs_set_benchmark.md). Ở bản cài đặt hiện tại,
`set` nhanh hơn; tài liệu đó giải thích nguyên nhân nằm ở cách vòng lặp ứng viên
chưa dùng `Trie.longest_match` một lượt.

## 6. Nguồn tham chiếu

* Kho chính: `https://github.com/roy-a/Roy_VnTokenizer` (Anindya Roy, 2014).
* Trang giới thiệu: `https://vlsp.org.vn/node/172`.
* Dinh et al., "Word segmentation of Vietnamese texts: a comparison of approaches",
  LREC 2008 — `http://www.lrec-conf.org/proceedings/lrec2008/pdf/493_paper.pdf`.
* Nguyen et al., "Vietnamese Word Segmentation with CRFs and SVMs: An Investigation"
  — `https://aclanthology.org/anthology-files/pdf/Y/Y06/Y06-1028.pdf`.
* Từ điển dùng thật: `resources/dictionary_source.json` (Viet74K, GPL-2.0, ghim commit).

## 7. Kết luận

Kho Roy là một tham chiếu thuật toán tốt về MM/MM+ và về cách xử lý nhập nhằng khi
tách từ tiếng Việt, nhưng **không tìm thấy giấy phép cho từ điển của nó trong kho
khảo sát nên dự án không dùng lại**. Dự án chọn Viet74K (GPL-2.0) và tự viết lõi Trie Maximum Matching, giữ lại
đúng phần ý tưởng thuật toán cần thiết và ghi rõ mọi nguồn dữ liệu.
