# Truy xuất văn bản tiếng Việt bằng TF-IDF — Team 7

Dự án đề tài 8, học phần Hệ cơ sở dữ liệu đa phương tiện. Hệ thống tách từ tiếng Việt, xây dựng chỉ mục TF-IDF, truy xuất theo Cosine và so sánh các chiến lược lọc từ dừng bằng Term Discrimination Value (TDV).

## Báo cáo chính thức

**Bản nộp chính thức: [Bao_Cao_De_Tai_8.docx](report/Bao_Cao_De_Tai_8.docx) và [Bao_Cao_De_Tai_8.pdf](report/Bao_Cao_De_Tai_8.pdf).** Báo cáo gồm đầy đủ 6 chương, 20 bảng biểu, các hình ảnh thực nghiệm, trích dẫn nguồn VNTC và 3 phụ lục. Các bản Word và PDF cũ đã được dọn dẹp để tránh nhầm lẫn; sử dụng các bản báo cáo được liên kết tại đây để nộp và đối chiếu kết quả.

Giảng viên: **Nguyễn Ngọc Duy**.

| Thành viên | MSSV | Phân công |
| --- | --- | --- |
| Lê Đình Bảo | N23DCCN141 | M1: xử lý ngôn ngữ; rà soát nhãn, tổng hợp kết quả và hoàn thiện báo cáo |
| Đặng Văn Hiệp | N23DCCN155 | M2: corpus, TDV, chỉ mục, truy xuất và đánh giá |
| Nguyễn Ngọc Quốc Huy | N23DCCN162 | M3: giao diện và tích hợp hệ thống |

## Chức năng

- Chuẩn hóa Unicode, từ điển Trie, tách từ FMM/BMM/BiMM và hiển thị từng bước xử lý.
- Lọc từ dừng tầng 1, bảo vệ từ phủ định và từ thuộc danh sách guard.
- Lập chỉ mục đảo, trọng số TF-IDF, xếp hạng Cosine và giải thích đóng góp của từng term.
- So sánh C1 (không lọc), C2 (stoplist tĩnh), C3 (tầng 1), C4 (tầng 1 + TDV).
- Tùy chọn chuyển sang chỉ mục C3 khi truy vấn C4 không có kết quả. C4 thuần và C4 fallback được đánh giá riêng.

Corpus cố định gồm **2.500 bài thuộc 10 lĩnh vực, 250 bài/lĩnh vực**. Bộ đánh giá có **20 truy vấn, 669 nhãn relevance và 150 câu gold tách từ**. Từ điển sau làm sạch có 69.858 mục. Chỉ mục dùng tệp JSON/NPZ và ma trận thưa, không yêu cầu máy chủ cơ sở dữ liệu riêng.

## Chạy trên máy nhóm

Môi trường đã kiểm tra: **Python 3.13**. Tại thư mục gốc repo, chạy PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -X utf8 -m pip install -r requirements-dev.txt
python -X utf8 -m scripts.build_indexes
python -X utf8 app/server.py --port 8000
```

Mở `http://localhost:8000`. Chạy backend để tìm kiếm trên dữ liệu thật; mở trực tiếp HTML chỉ cung cấp phần minh họa offline được đánh dấu rõ.

Giao diện Streamlit:

```powershell
python -X utf8 -m streamlit run app/Home.py
```

`data/indexes/` được dựng lại từ corpus đã lưu trong Git. Dữ liệu raw và môi trường ảo không được đưa vào repo. Không cần dữ liệu raw để dựng chỉ mục và chạy tìm kiếm từ corpus cố định. Trên Linux/macOS, kích hoạt môi trường bằng `source .venv/bin/activate`.

## Kết quả chốt

MAP tính trên ranking đầy đủ; P/R/F1/NDCG tại K=10, lấy trung bình trên 20 truy vấn. Nhãn liên quan có grade > 0; NDCG sử dụng mức relevance.

| Cấu hình | MAP | P@10 | R@10 | F1@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 0,7685 | 0,6200 | 0,5736 | 0,5105 | 0,7917 |
| C2 | 0,7585 | 0,6150 | 0,5687 | 0,5059 | 0,7891 |
| C3 | 0,7689 | 0,6200 | 0,5614 | 0,5076 | 0,7881 |
| C4 | 0,6385 | 0,5150 | 0,5152 | 0,4469 | 0,6809 |
| C4 fallback C3 | 0,7202 | 0,6050 | 0,5468 | 0,4934 | 0,7611 |

BiMM đạt Precision 0,9554, Recall 0,9684, F1 **0,9619** trên 150 câu gold. Chỉ mục phục vụ C4 giảm khoảng **32%** so với C1, nhưng MAP thấp hơn. Triển khai fallback cần lưu cả C3 và C4: **15.291.271 byte**, lớn hơn C1 khoảng 58,92%.

Benchmark warm gồm 20 truy vấn × 100 lượt, 5 lượt khởi động/truy vấn, Top-10. Độ trễ trung bình lần lượt C1/C2/C3/C4/fallback là **0,770/0,363/0,606/0,403/0,415 ms** trên máy đo. Số đo này không bao gồm toàn bộ thời gian HTTP, giao diện hay khởi động hệ thống và có thể thay đổi theo máy.

Chi tiết: [đánh giá truy xuất](results/ablation/summary.csv), [đánh giá theo lĩnh vực](results/ablation/summary_by_category.csv), [tách từ](results/segmentation/evaluation.json), [benchmark](results/latency/latency.json).

Bộ nhãn relevance được xây dựng theo pool, chưa bao phủ mọi tài liệu liên quan trong corpus. Mỗi lĩnh vực chỉ có hai truy vấn; gold chưa có vòng gán nhãn độc lập đầy đủ thứ hai. Vì vậy, kết quả phản ánh bộ dữ liệu và nhãn hiện có, chưa đủ để khái quát cho mọi truy vấn tiếng Việt.

## Kiểm tra và tái tạo kết quả

```powershell
python -X utf8 -m pytest -q
python -X utf8 -m evaluator.evaluate --mode official --k 10 --out results/ablation
python -X utf8 -m evaluator.segmentation_eval --out results/segmentation
python -X utf8 -m benchmarks.benchmark_latency --repeat 100 --warmup 5 --top-k 10 --out results/latency
```

Dựng chỉ mục trước khi chạy đánh giá. Không đổi corpus, truy vấn hoặc nguồn tiền xử lý giữa các lần so sánh. `.gitattributes` giữ nguyên byte của dữ liệu và nguồn tiền xử lý để các hash kiểm tra nhất quán khi clone; tránh để trình soạn thảo tự chuyển xuống dòng các tệp đã khóa.

## Cấu trúc chính

| Thư mục | Nội dung |
| --- | --- |
| `engine/` | Tiền xử lý, tokenizer, TF-IDF, TDV và công cụ tìm kiếm |
| `app/` | Giao diện web, API local và các trang Streamlit |
| `data/` | Corpus cố định, truy vấn, nhãn và manifest |
| `resources/` | Từ điển, stoplist, guard và thông tin nguồn/giấy phép |
| `evaluator/`, `benchmarks/` | Đánh giá chất lượng và hiệu năng |
| `results/` | Kết quả thực nghiệm |
| `report/` | Bản báo cáo chính thức, hình minh họa và các bản nguồn cũ |
| `tests/` | Kiểm thử thuật toán, dữ liệu và tích hợp |

Nguồn và giấy phép từ điển: [Viet74K](resources/dictionary_source.json); nguồn stoplist tĩnh: [static_stopwords_source.md](resources/static_stopwords_source.md).
