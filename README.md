# Truy xuất văn bản tiếng Việt bằng TF-IDF — Team7

Dự án giữa kỳ Hệ cơ sở dữ liệu đa phương tiện. M1: chuẩn hóa, Trie, FMM/BMM/BiMM, stopword tầng 1 và trace. M2: corpus, chỉ mục đảo, TF-IDF, Cosine/explain, TDV, đánh giá và benchmark. Giao diện Streamlit chạy local, không cần dịch vụ cloud.

Corpus chính đã khóa: **2.500 bài, 10 lĩnh vực, 250 bài/lĩnh vực**. Raw chỉ đọc. Đã tính chỉ số trên bộ nhãn hiện có, nhưng [rà soát ngữ nghĩa](docs/label_review.md) còn mở; không coi cờ `confirmed`, unit test hay tốc độ tìm kiếm là bằng chứng nhãn đúng nội dung.

## Hướng dẫn chạy nhanh khi Clone repo về máy

Yêu cầu: **Python 3.10+** (khuyến nghị Python 3.12 hoặc 3.13). Mở terminal tại thư mục gốc của dự án:

### Bước 1: Khởi tạo môi trường ảo & cài đặt thư viện
```powershell
# Tạo và kích hoạt môi trường ảo (PowerShell trên Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài đặt các thư viện cần thiết (bao gồm cả dev/test)
python -X utf8 -m pip install -r requirements-dev.txt
```
*(Nếu dùng Linux/macOS, kích hoạt bằng `source .venv/bin/activate`)*

### Bước 2: Dựng chỉ mục tìm kiếm (Bắt buộc cho lần đầu chạy)
Vì thư mục chỉ mục `data/indexes/` không đẩy lên Git để tối ưu dung lượng, bạn cần chạy lệnh sau để tự động tạo 4 chỉ mục C1–C4 (chỉ mất ~10–15 giây):
```powershell
python -X utf8 -m scripts.build_indexes
```

### Bước 3: Chạy ứng dụng và trải nghiệm
Hệ thống hỗ trợ 2 giao diện trực quan:
1. **Giao diện Web hiện đại (HTML/CSS/JS + Python HTTP Server - Khuyên dùng)**:
   ```powershell
   python -X utf8 app/server.py --port 8000
   ```
   Sau đó mở trình duyệt truy cập: `http://localhost:8000`

2. **Giao diện Streamlit (Đầy đủ chức năng phân tích chuyên sâu)**:
   ```powershell
   python -X utf8 -m streamlit run app/Home.py
   ```

### Bước 4: Kiểm thử hệ thống (Automated Testing)
Chạy toàn bộ bộ test kiểm thử tự động (Unit test, IR Math, Segmentation, IAA Crosscheck):
```powershell
python -X utf8 -m pytest -q
```
*(Hiện tại 502/502 bài test đều đạt chuẩn `passed`)*

## Tái tạo dữ liệu và tài nguyên

```powershell
python -X utf8 -m scripts.clean_dict
python -X utf8 -m scripts.build_corpus
python -X utf8 -m scripts.build_indexes
python -X utf8 -m scripts.verify_release --raw
```

Raw: 33.759 TXT UTF-16 LE có BOM, đủ 10 thư mục trực tiếp trong `data/raw/`. Chuẩn hóa NFC/whitespace, loại dưới 100 đơn vị `len(content.split())` (không phải số từ tiếng Việt), MD5 khử trùng trên toàn tập, seed 2026 và thứ tự nguồn ổn định. Sau lọc: 32.748 bài, bỏ 49 bản trùng còn 32.699; chọn 250/lĩnh vực. Tiêu đề suy ra được đánh dấu, không bịa URL. Không thay seed hoặc doc_id sau khi bắt đầu gán nhãn.

Corpus SHA256: `f42acebb1ac34084e859f378243145a3f8e981869350edb41a3dc27e86320fa7`. Script từ chối ghi corpus khác hash lên tập đã khóa. Để thử seed khác phải chỉ định các đường dẫn corpus/manifest/stats mới. Raw không được sửa hoặc đưa vào Git.

Từ điển: Viet74K được ghim commit, kèm nguồn và GPL-2.0 tại [dictionary_source](resources/dictionary_source.json), làm sạch thành 69.858 mục. Có 97,4233% mục dài tối đa 4 âm tiết, không đạt giả thiết 99,5% của kế hoạch cũ; cửa sổ mặc định theo mục dài nhất, không cố định 4. [Phân tích nguồn Roy và quyết định thay thế](docs/roy_analysis.md). Danh sách tầng 1 biên soạn theo chức năng ngữ pháp, có [quy tắc và số mục thực](resources/tier1_source.md); không chép stoplist tĩnh C2 sang C3 để đạt số lượng.

## Cấu hình và API

CLI build mặc định dùng **BiMM thật, chưa lọc tầng 1** cùng `engine.tokenizer.stopwords:filter_tier1`; tài nguyên và toàn bộ source phụ thuộc được fingerprint. Đánh dấu preprocessing official có nghĩa triển khai đã tích hợp, **không có nghĩa gold/relevance đã được người xác nhận**.

- C1: không lọc stopword.
- C2: danh sách tĩnh tiếng Việt độc lập, 1.942 mục chuẩn hóa; có [nguồn MIT](resources/static_stopwords_source.md). Số mục khớp vocabulary được ghi riêng.
- C3: chỉ tầng 1 và guard.
- C4: tầng 1 + TDV; bảo vệ guard cả ở tầng 1 lẫn khi loại chiều TDV.

TF = `1+ln(f)`; IDF = `ln((N+1)/(df+1))+1`; query dùng vocabulary/IDF corpus; L2/Cosine và sparse CSR/posting list. Hòa điểm theo doc_id. Query rỗng, stopword/OOV hoặc vector 0 không trả kết quả giả. Tổng contribution trong explain bằng Cosine.

```powershell
python -X utf8 -m scripts.search 'bệnh nhân không mắc bệnh' --configuration C4 --explain
# Chỉ dùng cho phát triển; ghi riêng, không thay chỉ mục M1:
python -X utf8 -m scripts.build_indexes --development-baseline --output data/indexes/development
```

C4 thuần và C4 fallback là hai phép đo riêng. Fallback truy xuất bằng **chỉ mục C3 riêng**, không giữ một term đã bị xóa rồi giả định tìm lại được. Gói fallback phải tính dung lượng C3+C4. Loader từ chối khi source/resources đổi; dựng lại cả bốn cấu hình sau khi sửa M1.

TDV: mật độ pairwise Cosine không gồm self trên toàn corpus đa lĩnh vực; `TDV=D_without_term-D_base`. Khi xóa chiều phải L2 lại và cập nhật tổng vector; IDF sơ bộ cố định, one-shot. Có bản tham chiếu dense nhỏ và test đối soát. Không ép đường cong chữ U hay mặc định C4 tốt nhất.

## Query, pool và relevance do người đọc xác nhận

20 query đã khóa, 2/lĩnh vực; có câu ngắn/dài/phủ định/thuật ngữ. Category và evidence không thay relevance. Pool mới dùng MM nằm tại **`data/ground_truth/mm/`**; pool baseline cũ và mọi nhãn cũ giữ ở đường dẫn gốc.

```powershell
python -X utf8 -m scripts.build_pool
python -X utf8 -m scripts.migrate_qrels
python -X utf8 -m scripts.annotate status
python -X utf8 -m scripts.annotate export
Invoke-Item .\data\ground_truth\mm\annotation_sheet.html
```

Pool hợp Top-20 của TF-IDF cosine, binary presence cosine (không IDF), BM25 (`k1=1.2,b=0.75`, IDF một lần). Cả ba cùng corpus/tokenizer; TF-IDF được đối soát với SearchEngine. Mẫu ngẫu nhiên 20% seed 2026 cho kiểm tra chéo. Script migrate chỉ mang nhãn confirmed sang cặp query/doc chung có cùng nội dung/hash, giữ nguyên nguồn và từ chối xung đột.

Hiện qrels có 669/669 cặp mang cờ confirmed, gold tách từ 150/150. Team7 đã gửi [37 quyết định qrels](data/ground_truth/mm/labels.json), rồi xác nhận sửa thêm [Q16/`vn_f4f2b5ae9443c172` thành mức 1](data/ground_truth/mm/q16_amendment_2026-10-18.json). Đánh giá đã chạy lại trên qrels SHA256 `697fafc7a89e40a782428204a4b0df21cb0545f24cbb714477c9328cfc88893f`. [Biên bản đối chiếu](results/label_review_resolution.json) ghi nhãn cuối 36/37 cặp trùng đề xuất AI, một khác có lý do. Gold chưa được sửa sau [audit ba câu](docs/label_review.md), nên kết quả trong `results/ablation/` và `results/segmentation/` vẫn phải đọc cùng cảnh báo ngữ nghĩa. [Tám bất đồng M1](docs/m1_disagreement_review.md) đã có phiếu đề xuất, chưa sửa qrels. Không nhập `results/label_review.json` như nhãn người.

Phiếu 37 cặp cũ gắn hash qrels **trước lần nhập này**, chỉ dùng để truy vết, không nhập lại. Để duyệt ba câu gold mà không thấy nhãn cũ, dùng [phiếu ba câu gold](data/gold/review_SG010_SG053_SG089_sheet.html). Có thể tạo lại phiếu gold bằng:

```powershell
python -X utf8 -m scripts.annotate_segmentation export --only SG010 SG053 SG089 --out data/gold/review_SG010_SG053_SG089_sheet.html
```

Các phiếu HTML chứa toàn văn, hiện có trong workspace nhưng được `.gitignore` bỏ qua; chạy lại lệnh `export` gold ở trên và `scripts.crosscheck_m1 export` sau khi lấy repo ở máy khác. Không công khai phiếu nếu quyền dữ liệu raw chưa được xác minh.

Phiếu riêng cho **người M1 thật** chấm độc lập 134 cặp không hiển thị nhãn gốc. Chấm đủ rồi xuất JSON từ trình duyệt; chỉ đối chiếu, không tự ghi đè qrels:

```powershell
python -X utf8 -m scripts.crosscheck_m1 export
Invoke-Item .\data\ground_truth\mm\m1_review_sheet.html
python -X utf8 -m scripts.crosscheck_m1 compare --labels 'DUONG_DAN\m1_review_labels.json'
```

Tệp `m1_review_labels.json` hiện có tự khai người M1 chấm 134 cặp; [đối chiếu trên qrels cuối](results/m1_crosscheck.json) cho **126 trùng, 8 bất đồng**. Chưa xác minh được danh tính/tính độc lập hoặc phân xử tám cặp, nên **chưa tuyên bố kiểm tra chéo hoàn tất**. Một AI làm thay nhiều vai trò không tạo ra các đánh giá viên độc lập.

Người đọc điền 0=không liên quan, 1=liên quan một phần, 2=đáp ứng trực tiếp; đọc cả nội dung và điều kiện phủ định. Xuất JSON từ phiếu rồi nhập:

```powershell
python -X utf8 -m scripts.annotate import --input 'DUONG_DAN\labels.json' --assessor 'TEN_NGUOI_GAN'
python -X utf8 -m scripts.annotate status
python -X utf8 -m scripts.annotate review
python -X utf8 -m scripts.annotate confirm
```

Không dùng `init --force` để chuyển phiên bản. Các tùy chọn đường dẫn của annotate đứng **trước** lệnh con. Script không tự xác nhận nhãn gợi ý; pending phải giữ null.

## Kiểm định tách từ M1

150 câu từ corpus khóa, 15/lĩnh vực, có nguồn và hash. Gợi ý của tokenizer chỉ là draft, không phải gold. Xem [workflow và lệnh chính xác](docs/segmentation.md) để chuẩn bị/điền/nhập nhãn, tính word-span Precision/Recall/F1 và phân tích lỗi khi đủ điều kiện. [Benchmark Trie/Set](docs/trie_vs_set_benchmark.md) kiểm tra output parity và báo thời gian/RAM thật, không giả định Trie nhanh hơn.

```powershell
python -X utf8 -m scripts.prepare_segmentation --check
python -X utf8 -m benchmarks.benchmark_lookup --repeat 10 --warmup 2 --corpus-docs 100
python -X utf8 -m scripts.annotate_segmentation status
# Sau khi người gán xác nhận đủ 150 câu:
python -X utf8 -m evaluator.segmentation_eval
```

## Đánh giá, benchmark, biểu đồ và báo cáo

```powershell
python -X utf8 -m benchmarks.benchmark_latency --repeat 100 --warmup 5 --top-k 10 --out results/latency
python -X utf8 -m scripts.plot_results --latency results/latency --tdv data/indexes/C4 --only latency size tdv --out report/figures
python -X utf8 -m scripts.render_report
python -X utf8 -m evaluator.evaluate --mode official --k 10 --out results/ablation
# Khi pending, lệnh official từ chối; khi đủ cờ confirmed nó kiểm kỹ thuật,
# không tự phát hiện nhãn sai ngữ nghĩa. Duyệt audit trước khi nộp số đo.
# Không ghi số 0 thành chất lượng đã đo.
python -X utf8 -m evaluator.evaluate --allow-development --out results/development_eval
```

Official evaluation bị chặn khi nhãn chưa đủ, hash/schema sai hoặc preprocessing còn baseline/fixture. K=10 cho P/R/F1/NDCG, AP/MAP dùng toàn ranking (depth mặc định N), relevant=grade>0. P@K mẫu số K, AP chia tổng relevant đã biết; query không relevant/không kết quả giữ 0 trong macro và đếm riêng. Cắt ranking phải cho phép rõ và ghi AP@depth, không gọi là MAP toàn ranking. Pooled qrels không bao phủ mọi relevance trong corpus; mỗi lĩnh vực chỉ 2 query nên phân tích từng lĩnh vực còn hạn chế.

Số liệu hiện hành: [latency.json](results/latency/latency.json), [toàn bộ query](results/latency/latency_per_query.csv), [audit tích hợp](results/release_audit.json). Snapshot số đo baseline cũ tại `results/baseline_development/`, không dùng làm kết quả MM. Mục tiêu giảm index 30%/mean dưới30ms phải đối chiếu số đo đúng phạm vi; không suy ra chất lượng relevance từ tốc độ.

### Báo cáo nghiệm thu hoàn chỉnh

1. **Báo cáo Microsoft Word (`.docx`) chính thức (Khuyên dùng khi nộp bài)**:
   - File: [`report/Bao_Cao_Nghiem_Thu_De_Tai_8_Team7.docx`](report/Bao_Cao_Nghiem_Thu_De_Tai_8_Team7.docx)
   - Đầy đủ mẫu bìa Học viện Công nghệ Bưu chính Viễn thông (PTIT), thông tin Nhóm 7, SV Đặng Văn Hiệp (B23DCCN155), 6 chương đầy đủ hình ảnh, bảng biểu và công thức toán học.
2. **Báo cáo nguồn LaTeX & Markdown**:
   - Báo cáo nguồn tại [report/main.tex](report/main.tex), Markdown M2 tại [m2_report.md](report/m2_report.md). Biên dịch Unicode với XeLaTeX hoặc Tectonic từ thư mục report:
```powershell
Push-Location report
tectonic main.tex --keep-logs
# Hoặc: xelatex main.tex (chạy hai lượt để cập nhật mục lục/tham chiếu)
Pop-Location
```

Kiểm tra cuối, số test, số đo thật và đầu vào con người còn thiếu: [integration_status.md](docs/integration_status.md). Không tự push/merge/phát hành dữ liệu; quyền tái phân phối raw chưa được xác minh.
