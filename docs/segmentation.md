# Đo chất lượng tách từ M1 (pipeline gold 150 câu)

Tài liệu này mô tả **đúng quy trình đang có trong repo** để đo chất lượng tách từ
của M1, trạng thái dữ liệu thật tại thời điểm viết, các lệnh PowerShell, và **những
gì phép đo này không nói được**. Mọi con số dưới đây lấy từ output thật của
`scripts/prepare_segmentation.py`, `scripts/annotate_segmentation.py` và
`evaluator/segmentation_eval.py` (đều do lead sở hữu; tài liệu này chỉ đọc và ghi lại).

## 0. Trạng thái hiện tại — chưa có số đo chính thức

| Chỉ số | Giá trị thật |
| --- | --- |
| Số câu trong mẫu | 150 |
| Số nhãn người **xác nhận** | **0 / 150** |
| `label_status` trong `segmentation_labels.jsonl` | `pending` cho 150/150 |
| `segmentation_manifest.status` | `pending_human_annotation` |
| `official_evaluation_allowed` | `false` |
| `results/segmentation/evaluation.json` | `official: false`, `metrics_status: "unmeasured"` |
| Lý do | `Need all 150 human-confirmed labels (15/category)` |

**Không có F1 nào được công bố.** Cả ba phương pháp FMM/BMM/BiMM đều **chưa được
đo** trên bộ này, vì chưa có nhãn người. Gợi ý của máy (`suggestion`) chỉ để gán
nhanh và **không bao giờ** tự trở thành gold.

## 1. Vì sao cần một bộ gold riêng

Các fixture kiểm thử (ví dụ `tests/test_mm.py`, `tests/test_segmentation.py`) chứng
minh **luật hoạt động đúng**, không đo **chất lượng**. Chất lượng tách từ tiếng Việt
chỉ đo được khi có người đọc câu thật và chốt ranh giới từ. Bộ 150 câu này là mẫu
câu thật rút từ corpus đã khóa, do đó không thay thế được bằng fixture tự viết tay.

## 2. Kiến trúc ba tầng

```
data/corpus.jsonl (khóa)                         data/raw (chỉ đọc, không đụng)
        │
        ├─ scripts/prepare_segmentation.py   → data/gold/segmentation_sample.jsonl
        │                                       data/gold/segmentation_manifest.json
        │                                       data/gold/segmentation_labels.jsonl (rỗng)
        │
        ├─ scripts/annotate_segmentation.py  → data/gold/annotation_sheet.html
        │        ◄── người gán nhập nhãn rồi xuất JSON ──┐
        │                                                │  import
        └─ evaluator/segmentation_eval.py  ←─────────────┘
                 → results/segmentation/evaluation.json
```

### 2.1 `scripts/prepare_segmentation.py` — lấy mẫu

* Rút câu từ `data/corpus.jsonl`; **từ chối chạy** nếu SHA256 của corpus trên đĩa
  không khớp `data/corpus_manifest.json` (không dựng mẫu từ corpus chưa khóa).
* Cắt câu tại ranh giới `[.!?;:]` theo sau là khoảng trắng, hoặc `\n`. Câu **không**
  vắt qua dấu câu.
* Chỉ giữ câu đủ điều kiện: `12 ≤ số đơn vị khoảng trắng ≤ 60`, không chứa URL/
  email/`www.`/tên miền (`SKIP_PATTERN`), và tỉ lệ ký tự chữ cái ≥ 0,6.
* Lấy **15 câu/lĩnh vực × 10 lĩnh vực = 150**, seed **2026**, ưu tiên mỗi câu từ một
  tài liệu khác nhau. Trong mẫu hiện tại: 150/150 câu đến từ **150 tài liệu khác nhau**.
* Mỗi câu có `sample_id` (`SG001…SG150`), `sentence_sha256`, `normalized_text`
  (do `engine.tokenizer.preprocessor.normalize` sinh), `whitespace_units`, nguồn
  `doc_id` + `sentence_index`, và `suggestion` (token BiMM, `status: "pending"`).
* `label = null`, `label_status = "pending_human_annotation"`.
* `--force` **không bao giờ xóa nhãn**: nếu file nhãn đã có dòng `confirmed` thì dừng.

Tham số mẫu thật: `units min/max/avg = 12 / 60 / 26,6`, `sample_sha256 =
37112c076cc9e54e335b2f0930a42dc6cbd672908994ac3ac7189bcf1fd59fb3`.

### 2.2 `scripts/annotate_segmentation.py` — gán nhãn thủ công, offline

* `export` sinh `annotation_sheet.html` **tự chứa** (kiểm: 0 tham chiếu `http`,
  121.686 byte). Mỗi câu có ô nhập; gợi ý máy nằm trong `<details>` **tách biệt**
  với ô trả lời, nên người gán không bị "điền sẵn" gợi ý.
* Quy ước: các **từ** cách nhau bằng khoảng trắng, các **âm tiết** trong một từ nối
  bằng `_`. Không thêm/bớt/đổi thứ tự âm tiết, không nối từ qua dấu câu.
* Trình duyệt lưu nháp vào `localStorage`; chỉ khi bấm "Xuất JSON" kèm ô **xác nhận
  đã tự kiểm tra** và **tên người gán** thì tệp JSON mới `human_confirmed: true`.
* `import` **kiểm tra toàn bộ rồi mới ghi** (nguyên tử): từ chối nếu thiếu
  `human_confirmed`, thiếu người gán, lệch `corpus_sha256`/`sample_sha256`/
  `normalizer_sha256`, lệch `sentence_sha256`, sample_id lạ/trùng, hoặc token không
  khớp trọn dãy âm tiết. Lỗi thì file nhãn **không đổi một byte**.
* `status` in tiến độ: hiện `Human confirmed: 0/150`.

### 2.3 `evaluator/segmentation_eval.py` — đo (khi đủ nhãn)

* `load_dataset` xác minh chuỗi hash: sample khớp manifest, `corpus.jsonl` khớp
  manifest khóa, và `normalizer_sha256` khớp `engine/tokenizer/preprocessor.py`.
  Lệch bất kỳ hash nào → `SegmentationError`.
* Xác minh thêm: mỗi câu có trong `content` của đúng `doc_id` và đúng `category`;
  `normalize(text)` khớp `normalized_text`; nhãn `confirmed` phải có tên người gán.
* **Metric là micro word-span P/R/F1**, không phải độ chính xác theo âm tiết:
  * mỗi từ là một span `(start, end)` theo chỉ số âm tiết (end mở),
  * `correct = |Predicted ∩ Gold|`, rồi precision/recall/F1 trên tổng các câu,
  * một từ **không** được vắt qua ranh giới dấu câu/đoạn bị gỡ (bị từ chối nếu vượt).
* **Chỉ đo chính thức khi** `confirmed == 150` **và** đúng 10 lĩnh vực × 15 câu;
  nếu không, ghi `metrics_status: "unmeasured"` và **không** kèm `metrics`.
* Khi đủ, đo cả ba phương pháp `fmm`, `bmm`, `bimm` (dùng `engine.tokenizer`), kèm
  P/R/F1 theo từng lĩnh vực và tối đa 15 ví dụ lỗi. Ghi `results/segmentation/evaluation.json`;
  exit code 5 nếu chưa đủ điều kiện.

## 3. Lệnh PowerShell

```powershell
# 1) Dựng mẫu 150 câu (lần đầu). Đã tồn tại thì dừng; dùng --force để dựng lại.
python -X utf8 -m scripts.prepare_segmentation
python -X utf8 -m scripts.prepare_segmentation --check     # chỉ kiểm tra, không ghi
python -X utf8 -m scripts.prepare_segmentation --force     # từ chối ghi đè nếu đã có nhãn confirmed

# 2) Gán nhãn: xuất phiếu HTML, người điền, rồi nhập lại
python -X utf8 -m scripts.annotate_segmentation export
python -X utf8 -m scripts.annotate_segmentation status
python -X utf8 -m scripts.annotate_segmentation import --input segmentation_labels.json --assessor "<tên>"

# 3) Đo (chỉ ra số khi đủ 150 nhãn confirmed)
python -X utf8 -m evaluator.segmentation_eval
```

`--check` hiện trả `ok: true`, 150 mẫu, 10 × 15, corpus
`f42acebb…20fa7`, `status: pending_human_annotation`.

## 4. Ràng buộc toàn vẹn (hash pin)

| Hash | Ý nghĩa |
| --- | --- |
| `corpus_sha256` | Corpus phải là bản đã khóa; đổi corpus là vô hiệu mẫu. |
| `sample_sha256` | Nội dung 150 câu không được sửa tay. |
| `normalizer_sha256` | Đổi `engine/tokenizer/preprocessor.py` là phải xem lại căn chỉnh trước khi đo. |
| `sentence_sha256` | Từng câu bị khóa, nhãn cũ không áp nhầm sang câu khác. |

Vì `normalizer_sha256` là SHA256 của **tệp** preprocessor, bất kỳ chỉnh sửa nào lên
tệp đó — dù đổi hành vi hay không — đều làm nhãn đang có thành "stale"; lúc đó phải
dựng lại mẫu và gán lại. Pin hiện tại:
`normalizer_sha256 = 6a384d7628c66c999d8e857fa2721955f8856157ce5234839b636bc498d1f438`
(khớp `engine/tokenizer/preprocessor.py` trên đĩa tại thời điểm viết).

## 5. Giới hạn của phép đo (phải nói rõ trong báo cáo)

1. **Chưa có số nào để báo.** 0/150 nhãn người; mọi "kết quả" hiện chỉ là đề xuất máy.
2. **Mẫu nhỏ.** 150 câu, 15 câu/lĩnh vực. Không có khoảng tin cậy, không kiểm định
   ý nghĩa; F1 từng lĩnh vực rất dễ dao động. So sánh FMM/BMM/BiMM chỉ nên nói là
   xu hướng, không phải kết luận chắc.
3. **Chỉ một người gán.** Không đo đồng thuận giữa người gán (inter-annotator
   agreement); sai sót chủ quan không được phát hiện bằng số.
4. **Metric span khắt khe.** Khớp đúng cặp chỉ số `(start, end)`; lệch một âm tiết
   là sai hoàn toàn, không có credit một phần.
5. **Miền bị loại khỏi mẫu.** Câu chứa URL/email/tên miền và câu < 12 hoặc > 60 đơn
   vị **không** được lấy mẫu, nên phép đo không bao phủ các trường hợp đó (dù
   preprocessor có xử lý URL/email — xem giới hạn guard bên dưới).
6. **Phụ thuộc từ điển.** Chất lượng đo được gắn với `resources/dict.txt`
   (`sha256 = 175e8b1b…d8ab`, 69.858 mục). Đổi từ điển thì phải đo lại; con số cũ
   không còn nghĩa.
7. **Nhãn tay vẫn có thể sai.** Công cụ chỉ ép *định dạng* (khớp âm tiết, không vắt
   dấu câu), không ép *đúng ngữ nghĩa*; chất lượng gold phụ thuộc người gán.
8. **Ranh giới bảo vệ phủ định chưa có trong metric.** Việc tách `không`/`chẳng`/`chưa`
   thành từ riêng để bảo vệ phủ định là chính sách của `engine/tokenizer/stopwords.py`
   (builder-retrieval), **không** được đo bởi bộ span này; metric chỉ đo ranh giới từ
   của Maximum Matching.
9. **Không so với tokenizer ngoài.** Chỉ có ba phương pháp trong bộ này; không so
   với VnCoreNLP/underthesea vì không nằm trong phạm vi offline của repo.

## 6. Kiểm thử liên quan

`tests/test_segmentation.py` (fixture tạm, không đụng gold thật):

* metric span không phải độ chính xác âm tiết;
* từ chối token thiếu/đảo/không hợp lệ/vắt qua ranh giới;
* mẫu thật đủ 150 và 10 × 15, và **vẫn bị chặn** công bố chất lượng khi chưa có nhãn;
* `import` từ chối nhãn lệch hash / tự động / thiếu người gán / trùng, **nguyên tử**;
* một nhãn không mở khóa đo chính thức; chỉ khi đủ 150 (nhãn fixture) mới ra metric
  cho cả ba phương pháp và 10 lĩnh vực.

## 7. Việc còn lại

1. Người gán điền đủ **150 câu** (qua `annotation_sheet.html` hoặc JSON) và import.
2. Chạy `python -X utf8 -m evaluator.segmentation_eval` để có F1 chính thức.
3. Nếu từ điển hoặc preprocessor đổi: dựng lại mẫu (`--force`) và gán lại; nhãn cũ
   sẽ bị coi là stale.
4. Chỉ khi đó mới đưa số tách từ vào báo cáo, kèm các giới hạn ở mục 5.
