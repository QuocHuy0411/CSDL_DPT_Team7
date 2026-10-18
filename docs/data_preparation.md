# Chuẩn bị dữ liệu M2 (corpus, queries, pool, qrels)

Tài liệu này mô tả đúng những gì đã **thực sự chạy và đo** cho vai trò builder-data.
Mọi con số dưới đây lấy từ output thật của `scripts/build_corpus.py`,
`scripts/build_pool.py` và `scripts/annotate.py`.

## 1. Kiểm kê dữ liệu thô (`data/raw`, chỉ đọc)

`data/raw` là VNTC `10Topics/Ver1.1` đã giải nén. Không script nào trong repo ghi
vào thư mục này.

| Kiểm kê | Kết quả thật |
| --- | --- |
| Số tệp `.txt` | 33.759 |
| Encoding phát hiện | `utf-16-le-bom` cho **33.759/33.759** tệp |
| Lỗi giải mã | 0 |
| Kết thúc dòng | CRLF trong toàn bộ tệp |
| Trùng lặp toàn cục theo MD5 nội dung chuẩn hóa (toàn bộ 33.759 tệp) | 50 nhóm, 50 bản sao thừa |
| Trùng lặp trong tập đủ điều kiện (≥100 đơn vị) | **49 nhóm, 49 bản sao thừa** |
| Dưới ngưỡng 100 đơn vị khoảng trắng | 1.011 |
| Đơn vị khoảng trắng: min / trung vị / max | 49 / 361 / 16.988 |

Số tệp từng thư mục khớp đúng `Stats.txt` của VNTC:

| Thư mục thô | Tệp | Nhãn hiển thị |
| --- | ---: | --- |
| `Chinh tri Xa hoi` | 5.219 | Chính trị – Xã hội |
| `Doi song` | 3.159 | Đời sống |
| `Khoa hoc` | 1.820 | Khoa học |
| `Kinh doanh` | 2.552 | Kinh doanh |
| `Phap luat` | 3.868 | Pháp luật |
| `Suc khoe` | 3.384 | Sức khỏe |
| `The gioi` | 2.898 | Thế giới |
| `The thao` | 5.298 | Thể thao |
| `Van hoa` | 3.080 | Văn hóa |
| `Vi tinh` | 2.481 | Vi tính |

### 1.1 Quy tắc giải mã

* BOM quyết định codec: `FF FE` → `utf-16`, `FE FF` → `utf-16`, `EF BB BF` → `utf-8-sig`.
* Không có BOM thì thử `utf-16`, `utf-8-sig`, `utf-8`.
* **Không** dùng `errors="replace"` và **không** có codepage cũ (cp1258). cp1258 nhận
  mọi chuỗi byte nên sẽ âm thầm làm hỏng dấu tiếng Việt. Tệp không giải mã được
  bị loại khỏi corpus và được báo cáo trong `corpus_stats.json`
  (`undecodable_files`; lần chạy này bằng 0).

### 1.2 Chuẩn hóa nội dung

`content = " ".join(unicodedata.normalize("NFC", text).split())`

Hệ quả cần nói rõ:

* `whitespace_unit_count == len(content.split())` **luôn đúng** về mặt kỹ thuật.
* Ngưỡng 100 là **đơn vị khoảng trắng**, không phải số từ sau khi tách tiếng Việt.
  Hai con số này khác nhau và không được gọi lẫn lẫn.
* Tiêu đề nằm trong `content` (dòng đầu của tệp thô là một phần của bài), nên chỉ mục
  `content` không cộng thêm tiêu đề.

### 1.3 Quy tắc tiêu đề và tính trung thực

VNTC không có trường tiêu đề riêng. Quy tắc H1, áp dụng đồng nhất:

`title_origin = "source_line"` khi **cả bốn** điều kiện đúng:

1. tài liệu có ít nhất 2 dòng không rỗng;
2. dòng đầu thô **không** bắt đầu bằng khoảng trắng;
3. dòng đầu ≤ 20 đơn vị khoảng trắng;
4. dòng đầu ≤ 0,75 × trung vị số đơn vị của các dòng còn lại.

Nếu không, `title_origin = "inferred_excerpt"`: tiêu đề là **trích đoạn nguyên văn**
lấy từ chính `content` (câu đầu tiên trong 20 đơn vị khoảng trắng đầu, nếu không thì
20 đơn vị đầu, thêm `…` khi bị cắt). Không sinh ra chữ nào không có trong nguồn.

Vì sao cần điều kiện 2: 11.466 tệp của một nguồn báo ghi tiêu đề nối thẳng vào đầu
đoạn văn trên cùng dòng (ví dụ ` Thành lập dự án POLICY phòng chống HIV/AIDS ở VN
(NLĐ)- Quỹ hỗ trợ …`). Không tồn tại ranh giới tin cậy, nên đánh dấu là suy ra.

Kết quả trên corpus 2.500 bài: **1.360 `source_line`**, **1.140 `inferred_excerpt`**
(tất cả 1.140 đều bị cắt, thống kê ở `title_truncated_count`).

## 2. Xây corpus khóa

```powershell
python scripts/build_corpus.py            # ghi data/corpus.jsonl + manifest + stats
python scripts/build_corpus.py --audit-only   # chỉ kiểm kê, không ghi gì
```

| Thuộc tính | Giá trị |
| --- | --- |
| Số tài liệu | 2.500 |
| Số lĩnh vực | 10 × 250 |
| `corpus_sha256` | `f42acebb1ac34084e859f378243145a3f8e981869350edb41a3dc27e86320fa7` |
| Seed | 2026 |
| `doc_id` | `vn_` + 16 hex đầu của MD5 nội dung |
| Thứ tự bản ghi | tăng dần `source_file` tương đối |
| Tệp không thuộc 10 thư mục | bỏ qua, có báo cáo |

### 2.1 Thứ tự thực thi (đã sửa theo review)

1. Đọc và chuẩn hóa **toàn bộ** 33.759 tệp.
2. Lọc ngưỡng 100 đơn vị khoảng trắng.
3. **Khử trùng lặp MD5 toàn cục trên toàn bộ tập đủ điều kiện**, theo thứ tự đường
   dẫn tương đối tăng dần; bản sao đầu tiên giữ quyền sở hữu.
4. Mới lấy mẫu 250/lĩnh vực từ nhóm đã khử trùng lặp.

Bước 3 nằm trước bước 4 là bắt buộc. Nếu khử trùng lặp chạy trong lúc lấy mẫu,
báo cáo chỉ thấy trùng lặp rơi vào mẫu và bỏ sót phần còn lại của đầu vào.

Báo cáo đầy đủ nằm ở `manifest.deduplication` và `corpus_stats.json`:

```
eligible_documents               32.748
unique_documents                 32.699
duplicate_copies_removed         49
cross_category_duplicate_copies  49
within_category_duplicate_copies 0
```

Cả 49 bản sao đều là bài đăng ở **hai thư mục khác nhau**, phân bố theo 13 cặp thư mục
(lớn nhất: `Khoa hoc <-> Suc khoe` 17, `Doi song <-> Khoa hoc` 11,
`Chinh tri Xa hoi <-> Phap luat` 6). Nguyên nhân là cùng một bài báo được gắn nhãn
chủ đề khác nhau ở nhiều bài khác nhau của VNTC. Không có nhóm nào mất bài nào sau
khi khử trùng lặp vì mỗi thư mục vẫn còn dư hơn 250 bài.

**Sai lệch 49 so với 50 cần nói rõ.** Kiểm kê toàn bộ 33.759 tệp cho ra 50 nhóm
trùng lặp, nhưng một nhóm là `Doi song/DS_ VNE_ (2469).txt` và
`Doi song/DS_ VNE_ (2470).txt`, mỗi tệp chỉ có **54** đơn vị khoảng trắng. Cả hai bị
loại ở bước lọc ngưỡng, trước bước khử trùng lặp, nên không xuất hiện trong báo cáo
đầu ra. 50 là con số trên toàn bộ đầu vào, 49 là con số trên tập đủ điều kiện.

### 2.2 Chốt khóa (lock)

`data/corpus_manifest.json` khóa SHA256 của chính `data/corpus.jsonl` trước khi
bất kỳ nhãn nào được tạo. Có hai chặn:

* **Chặn 1** — nếu manifest đang khóa một hash mà tệp corpus trên đĩa không khớp,
  dừng ngay.
* **Chặn 2** — sau khi dựng xong corpus mới, so `corpus_sha256` **mới** với hash
  đang khóa; lệch thì từ chối ghi đè, trừ khi có cờ tường minh
  `--allow-overwrite-lock`.

Không ghi thời gian hệ thống vào manifest, nên dựng lại với cùng đầu vào cho ra
tệp **giống hệt từng byte** (đã kiểm tra 3 lần chạy liên tiếp).

## 3. Bộ 20 truy vấn (`data/queries.json`)

* 2 truy vấn/lĩnh vực, tổng 20.
* Phân bố kiểu: `short` 5, `long` 6, `negation` 3, `technical` 6.
* `corpus_sha256` khớp manifest; `frozen: true`.

| ID | Lĩnh vực | Kiểu | Truy vấn |
| --- | --- | --- | --- |
| Q01 | Chính trị – Xã hội | short | lương tối thiểu |
| Q02 | Chính trị – Xã hội | negation | biển cấm đỗ xe ô tô ở Hà Nội mất hiệu lực thì người dân đỗ xe ở đâu |
| Q03 | Đời sống | long | có nên tha thứ cho người vợ ngoại tình |
| Q04 | Đời sống | short | tâm sự gia đình |
| Q05 | Khoa học | technical | nghiên cứu lõi băng để khám phá quy luật biến đổi khí hậu |
| Q06 | Khoa học | long | Trung Quốc xây trạm không gian và Nga phóng tàu lên ISS khác nhau như thế nào |
| Q07 | Kinh doanh | short | cổ phiếu |
| Q08 | Kinh doanh | technical | lãi suất cơ bản và chính sách tiền tệ của Fed ảnh hưởng thị trường ra sao |
| Q09 | Pháp luật | long | điều tra vụ biển thủ hơn 8,7 tỷ đồng cá độ bóng đá đã được xử lý ra sao |
| Q10 | Pháp luật | technical | đường dây chạy án trong vụ PMU18 |
| Q11 | Sức khỏe | short | ung thư |
| Q12 | Sức khỏe | negation | mỡ máu cao nếu không điều trị có thể gây những biến chứng gì |
| Q13 | Thế giới | negation | vì sao Mỹ vẫn chưa tìm thấy vũ khí cấm ở Iraq |
| Q14 | Thế giới | technical | xác nhận danh tính hài cốt lính Mỹ mất tích trong Chiến tranh Triều Tiên |
| Q15 | Thể thao | short | vô địch |
| Q16 | Thể thao | long | Việt Nam đăng cai một vòng bảng Tiger Cup 2004 và các trận chung kết diễn ra thế nào |
| Q17 | Văn hóa | long | cảnh nóng và trang phục hở trong phim truyền hình Việt Nam gây tranh luận như thế nào |
| Q18 | Văn hóa | technical | ca khúc và chương trình biểu diễn âm nhạc của nghệ sĩ |
| Q19 | Vi tính | technical | virus máy tính đánh cắp mật khẩu Windows và cách phòng tránh |
| Q20 | Vi tính | long | dịch vụ phần mềm và thông tin di động được doanh nghiệp cung cấp cho khách hàng ra sao |

### 3.1 `evidence_doc_ids` không phải là qrels

`evidence_doc_ids` là bài viết đã **đọc** để soạn truy vấn. Mỗi ID đều được kiểm
tra: tồn tại trong corpus và thuộc đúng lĩnh vực khai báo. `evidence_doc_ids`
không bao giờ được dùng làm nhãn liên quan, và `category` cũng không thay được nhãn.

Ba lần soạn lại đã diễn ra vì chọn bằng đếm từ khóa cho ra bài không trả lời câu hỏi
(ví dụ Q03 ban đầu trỏ vào bài về ly dị hậu hôn thay vì bài "có nên tha thứ").
Bài chọn phải đọc tiêu đề và nội dung thật.

### 3.2 Ba kiểu phủ định thực tế

Q02 (mất hiệu lực), Q12 (không điều trị), Q13 (chưa tìm thấy) đều là tình huống có
thật trong bài được trích, không phải phủ định bịa thêm. Preprocessing của M1 giữ
một tập `GUARDS` gồm các từ phủ định; kết quả C1–C4 sau này phải cho thấy các từ này
còn trong truy vấn.

## 4. Gộp pool (`data/ground_truth/pool.json`)

```powershell
python scripts/build_pool.py
```

Union Top-20 của ba baseline trên **cùng corpus và cùng tokenizer**
(`engine.preprocessing:baseline_tokenize`, đang ở trạng thái `development_only`),
chỉ lập chỉ mục trường `content`, hòa điểm sắp theo `doc_id` tăng dần.

| Baseline | Công thức |
| --- | --- |
| `tfidf_cosine` | TF = 1 + ln f; IDF = ln((N+1)/(df+1))+1; chuẩn L2 cả tài liệu và truy vấn |
| `binary_cosine` | nhị phân có/xuất hiện, **không IDF**, chuẩn L2 cả hai phía |
| `bm25` | IDF = ln(1+(N−df+0,5)/(df+0,5)); k1 = 1,2; b = 0,75; không chuẩn hóa |

BM25 đặt IDF ở **phía tài liệu**; mỗi từ truy vấn duy nhất đóng góp trọng số 1,0.
Không nhân IDF hai lần và không dùng tần suất truy vấn.

Kết quả thật:

| Chỉ số | Giá trị |
| --- | --- |
| Số cặp trong pool | 704 |
| Kích thước pool / truy vấn | min 29, max 43, trung bình 35,2 |
| Kích thước tối đa lý thuyết | 60 |
| Số truy vấn đạt pool tối đa | 0 |
| Kiểm tra parity với `SearchEngine` | **passed**, 20/20 truy vấn, sai số tối đa 1,11e-16 |

`pool.json` lưu provenance cho từng cặp (`retrieved_by`, `ranks`, `scores`,
`best_rank`) và **toàn văn bài viết** để người gán đọc offline được.

## 5. Công cụ gán nhãn (`scripts/annotate.py`)

```powershell
python scripts/annotate.py init      # qrels.json (tất cả pending) + mẫu M1
python scripts/annotate.py status    # tiến độ
python scripts/annotate.py label --query Q07 --assessor "<tên>"
python scripts/annotate.py export --assessor "<tên>"   # phiếu HTML offline
python scripts/annotate.py import --input labels.json
python scripts/annotate.py review    # mẫu đối chiếu M1
python scripts/annotate.py confirm   # chỉ khi không còn pending
```

### 5.1 Trạng thái hiện tại

| Chỉ số | Giá trị |
| --- | --- |
| `qrels.status` | `pending_human_annotation` |
| Số phán đoán | 704 |
| Đã xác nhận | **0** |
| Mẫu M1 | 141 cặp = 20% × 704, seed 2026 |

### 5.2 Nguyên tắc

* Công cụ **không** tự tạo nhãn `confirmed`. Một phán đoán chỉ thành `confirmed` khi
  người gán nhập 0/1/2 **và** tên người gán.
* `relevance = null` nghĩa là chưa đánh giá; dòng đó không bao giờ được tính là
  relevant.
* `confirm` từ chối chạy khi còn dòng pending.
* Hồ sơ nhãn phải mang `corpus_sha256` và `queries_sha256`; `import` **từ chối**
  nếu thiếu hoặc lệch, để nhãn của phiếu cũ không áp nhầm lên bộ truy vấn đã đổi.
  `load_context` và `init` cũng so `queries_sha256` giữa pool và qrels.
* `data/ground_truth/*.html` đã nằm trong `.gitignore`; phiếu HTML là tự chứa,
  không có tài nguyên ngoài (đã kiểm: 0 tham chiếu `http`).

### 5.3 Mẫu M1

`random.Random(2026).sample` trên danh sách cặp đã sắp theo `(query_id, doc_id)`,
lấy `round(704 × 0,20) = 141` cặp. Cùng seed cho cùng pool thì cho cùng kết quả.

## 6. Kiểm thử

```powershell
python -m pytest -q tests/test_corpus.py tests/test_pool.py     # 69 passed
```

Bao gồm:

* `build_corpus`: chuẩn hóa NFC + gộp khoảng trắng, giải mã nghiêm ngặt (từ chối
  byte hỏng thay vì thay thế), quy tắc tiêu đề H1, khử trùng lặp toàn cục trước khi
  lấy mẫu (kể cả bản sao không bao giờ được lấy mẹo), đổi seed thì đổi mẫu nhưng
  không đổi số lượng, chặn relock, dựng lại cho tệp giống hệt từng byte.
* `build_pool`: khớp `TfidfVectorizer` của scikit-learn cho cả tài liệu lẫn truy
  vấn (sai số < 1e-12), BM25 và binary cosine khớp tham chiếu viết tay
  (< 1e-12), IDF không bị nhân hai lần, `bm25` không nhân qTF, hòa điểm theo
  `doc_id`, truy vấn rỗng/ngoài từ vốn trả `[]`, parity với `SearchEngine` là
  điều kiện bắt buộc.
* `annotate`: vòng init → export → import → confirm trên fixture tạm, mọi nhãn
  thật vẫn pending, mẫu M1 tất định 20%, từ chối hash lệch và từ chối nhãn thiếu
  người gán.

## 7. Đo chất lượng tách từ M1 (gold 150 câu)

Quy trình đầy đủ ở `docs/segmentation.md`. Phần liên quan trực tiếp tới dữ liệu M2:

* Mẫu **150 câu** (15 × 10 lĩnh vực) rút từ **chính `data/corpus.jsonl` đã khóa**,
  seed 2026, cắt câu tại `[.!?;:]`/xuống dòng, chỉ giữ câu 12–60 đơn vị và không
  chứa URL/email; corpus không bị sửa.
* Gợi ý token là đầu ra của `engine.tokenizer.max_matching:tokenize` (BiMM), trạng
  thái `pending`; **nhãn người hiện `0/150`**.
* `evaluator/segmentation_eval.py` **từ chối công bố chất lượng** cho tới khi đủ 150
  nhãn `confirmed`, đúng 10 × 15. Kết quả hiện tại ở `results/segmentation/evaluation.json`:
  `official: false`, `metrics_status: "unmeasured"`.
* Hash khóa: `corpus_sha256 = f42acebb…20fa7`,
  `sample_sha256 = 37112c07…59fb3`,
  `normalizer_sha256 = 6a384d76…d1f438`, từ điển `175e8b1b…c5d8ab` (69.858 mục).
* Lệnh PowerShell: `python -X utf8 -m scripts.prepare_segmentation [--check|--force]`,
  `python -X utf8 -m scripts.annotate_segmentation [export|import|status]`,
  `python -X utf8 -m evaluator.segmentation_eval`.
* Giới hạn: mẫu nhỏ (15/lĩnh vực, không khoảng tin cậy), **một** người gán (không đo
  đồng thuận), metric span khắt khe (không credit một phần), không bao phủ URL/email,
  và kết quả gắn với từ điển hiện tại. Xem mục 5 của `docs/segmentation.md`.

## 8. Việc còn lại

1. Người gán thật chạy `label` hoặc `export` → `import`; và điền đủ **150 câu** tách
   từ rồi import để mở khóa đo chất lượng M1.
2. Đối chiếu chéo 141 cặp của mẫu M1.
3. `confirm` khi 704 phán đoán đã có nhãn.
4. **Cho đến lúc đó, không được công bố số đo đánh giá chính thức** (cả retrieval lẫn
   tách từ). Bộ đánh giá phải từ chối khi `qrels.status`/`segmentation status` còn là
   `pending_human_annotation` hoặc hash không khớp.
5. Tokenizer đang là `development_only`; khi M1 bàn giao thật thì dựng lại pool và
   qrels (pool phụ thuộc tokenizer), đồng thời dựng lại mẫu tách từ nếu từ điển hoặc
   preprocessor đổi.