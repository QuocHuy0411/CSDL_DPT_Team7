# LỊCH SỬ TRAO ĐỔI – ĐỒ ÁN HỆ CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN (M2)

> **Bản ghi biên tập toàn bộ nội dung trao đổi theo trình tự thời gian**, tính đến yêu cầu xuất Markdown ngày **16/10/2026**. Tài liệu tổng hợp lời yêu cầu, thông tin đã cung cấp, các phản hồi/hướng dẫn, kết luận kỹ thuật, công thức, câu lệnh, liên kết và các việc đang chờ. Để dễ đọc, nội dung trả lời dài được tái cấu trúc thành các mục có tiêu đề; vì vậy đây là **bản viết lại đầy đủ về nội dung**, không phải bản xuất nguyên trạng từng thành phần giao diện cuộc trò chuyện. Các số liệu thực nghiệm minh họa không được coi là kết quả thực nghiệm cuối cùng của đồ án.

## MỤC LỤC

1. [Bối cảnh, đề tài và tài liệu đầu vào](#1-bối-cảnh-đề-tài-và-tài-liệu-đầu-vào)
2. [Lần trao đổi 1 – Giới thiệu đề tài và giáo trình](#2-lần-trao-đổi-1--giới-thiệu-đề-tài-và-giáo-trình)
3. [Lần trao đổi 2 – Tài liệu nhóm trưởng soạn](#3-lần-trao-đổi-2--tài-liệu-nhóm-trưởng-soạn)
4. [Lần trao đổi 3 – Phân công M2 và hướng dẫn chi tiết](#4-lần-trao-đổi-3--phân-công-m2-và-hướng-dẫn-chi-tiết)
5. [Lần trao đổi 4 – Kiểm tra repository và bộ TF-IDF khởi đầu](#5-lần-trao-đổi-4--kiểm-tra-repository-và-bộ-tf-idf-khởi-đầu)
6. [Lần trao đổi 5 – Kiểm tra nguồn dataset](#6-lần-trao-đổi-5--kiểm-tra-nguồn-dataset)
7. [Lần trao đổi 6 – Bắt đầu theo kế hoạch nhóm trưởng, từng bước](#7-lần-trao-đổi-6--bắt-đầu-theo-kế-hoạch-nhóm-trưởng-từng-bước)
8. [Lần trao đổi 7 – Yêu cầu xuất lịch sử thành Markdown](#8-lần-trao-đổi-7--yêu-cầu-xuất-lịch-sử-thành-markdown)
9. [Tổng hợp quyết định, tồn đọng và việc tiếp theo](#9-tổng-hợp-quyết-định-tồn-đọng-và-việc-tiếp-theo)
10. [Phụ lục: đường dẫn và sản phẩm](#10-phụ-lục-đường-dẫn-và-sản-phẩm)

---

## 1. Bối cảnh, đề tài và tài liệu đầu vào

### 1.1. Đề tài nhóm

**Tên đề tài:** Xây dựng hệ thống truy xuất văn bản tiếng Việt sử dụng TF-IDF.

Yêu cầu gốc của đề tài số 8 trong tài liệu danh sách đồ án:

1. Tách từ tiếng Việt.
2. Loại stopword.
3. Áp dụng TF-IDF.
4. Dùng Cosine similarity.
5. Đánh giá kết quả truy xuất.

Tài liệu giao đồ án cũng yêu cầu ba nội dung khi trình bày: **lý thuyết liên quan; cài đặt giải thuật và chương trình; thực nghiệm và đánh giá**.

Trong hình đề tài do người dùng cung cấp, nội dung đã được diễn giải theo hướng tách từ **Maximum Matching**, sau đó loại stopword, TF-IDF, Cosine similarity và đánh giá.

### 1.2. Những file đã gửi trong cuộc trao đổi

| STT | Tên file | Vai trò |
|---:|---|---|
| 1 | `image.png` | Hình ảnh đề tài nhóm được gửi ở đầu cuộc trò chuyện. |
| 2 | `Bai tap nhom - D23CQCNHT01.pdf` | Danh sách 20 đề tài và yêu cầu thực hiện. |
| 3 | `Chuong 0 - He CSDL MD - Gioi thieu.ppt` | Giới thiệu môn Hệ CSDL đa phương tiện. |
| 4 | `Chuong 1 - He CSDL MD - Tong quan - 1.pdf` | Tổng quan dữ liệu đa phương tiện, đặc trưng và truy xuất. |
| 5 | `Chuong 1 - He CSDL MD - Tong quan - 2.pdf` | Cấu trúc không gian đa chiều, cây k-d, Quadtree. |
| 6 | `Chuong 1 - He CSDL MD - Tong quan - 3.pdf` | Quadtree, MX-Quadtree, R-tree. |
| 7 | `Chuong 1 - He CSDL MD - Tong quan - 4.pdf` | Nén dữ liệu và các vấn đề giao diện/truy vấn đa phương tiện. |
| 8 | `Chuong 2 - He CSDL MD - Quản trị dữ liệu đa phương tiện - 1.pdf` | Khái niệm, kiến trúc, thành phần MMDB. |
| 9 | `Chuong 2 - He CSDL MD - Quản trị dữ liệu đa phương tiện - 2.pdf` | Mô hình hóa, metadata và biểu diễn đối tượng đa phương tiện. |
| 10 | `Chuong 3 - He thong tuy van thong tin van ban - 1.pdf` | Information Retrieval, document/query/index, inverted index và tiền xử lý văn bản. |
| 11 | `CSDL-ĐPT.docx` | Tài liệu kế hoạch học thuật, kỹ thuật và phân công của trưởng nhóm, 44 trang. |
| 12 | `CSDL_DPT_Team7.zip` | Repo hiện tại của nhóm, được gửi để kiểm tra triển khai. |
| 13 | `M2_Buoc1_TFIDF_Baseline.zip` | Bộ code baseline TF-IDF được trợ lý chuẩn bị trong một lượt trao đổi trước. |

Một số tên file trong bảng là tên file xuất hiện trong lịch sử, không có nghĩa toàn bộ đã được sử dụng làm đầu vào chạy chương trình chính thức.

### 1.3. Kiến thức giáo trình liên quan nhất

- Chương 3: **Hệ thống truy vấn thông tin văn bản** là cơ sở trực tiếp cho đề tài.
- Các khái niệm trọng tâm: **Information Retrieval (IR), Document, Query, Index, Relevance, Tokenization, Inverted Index, Vector Space Model, TF-IDF, Cosine Similarity, Ranking**.
- Chương 1 và 2 bổ trợ góc nhìn về trích đặc trưng, chỉ số hóa, cấu trúc và hệ CSDL đa phương tiện.

---

## 2. Lần trao đổi 1 – Giới thiệu đề tài và giáo trình

### Người dùng

> đây là đề tài của nhóm tôi, và tôi sẽ gửi bạn thêm giáo trình của môn hệ cơ sở dữ liệu đa phương tiện

Kèm hình ảnh đề tài, tài liệu bài tập nhóm, các chương giáo trình đã nêu trong mục 1.2.

### Trợ lý phản hồi – Nội dung chính

Trợ lý xác nhận đã nhận được đề tài và giáo trình môn **Hệ cơ sở dữ liệu đa phương tiện**.

#### 2.1. Xác định 5 yêu cầu của đề tài

1. **Tách từ tiếng Việt:** sử dụng Maximum Matching theo yêu cầu được thể hiện trong hình đề tài.
2. **Loại stopword:** loại các từ ít đóng góp vào việc phân biệt nội dung tài liệu.
3. **TF-IDF:** chuyển tài liệu và câu truy vấn thành các vector trọng số.
4. **Cosine similarity:** đo tương đồng, sắp xếp kết quả tìm kiếm.
5. **Đánh giá truy xuất:** kiểm tra chất lượng của hệ thống.

#### 2.2. Quy trình tổng quát đã đề xuất

```text
Tập tài liệu văn bản tiếng Việt
                ↓
Tiền xử lý văn bản
  • Chuẩn hóa
  • Maximum Matching
  • Loại stopword
                ↓
Biểu diễn vector TF-IDF
                ↓
Xử lý truy vấn + Cosine Similarity
                ↓
Xếp hạng và trả về Top-K tài liệu liên quan
```

Trợ lý nhấn mạnh nên bám giáo trình giảng viên, đồng thời đáp ứng cả **lý thuyết – cài đặt – thực nghiệm**. Trợ lý đề nghị người dùng tiếp tục gửi giáo trình để hỗ trợ lên kế hoạch, xây dựng hệ thống, triển khai thuật toán và kiểm tra kết quả.

---

## 3. Lần trao đổi 2 – Tài liệu nhóm trưởng soạn

### Người dùng

> đây là những nội dung mà nhóm trưởng tôi đã soạn

Đính kèm: `CSDL-ĐPT.docx` (44 trang).

### Trợ lý phản hồi – Nội dung chính

Trợ lý đọc tài liệu và xác định đây là kế hoạch chi tiết bao gồm: **cơ sở lý thuyết, kiến trúc, hướng dẫn triển khai thuật toán, phân chia công việc và lộ trình 4 tuần**.

#### 3.1. Các lựa chọn kỹ thuật được nhóm trưởng đề xuất

| Hạng mục | Phương án trong tài liệu |
|---|---|
| Tách từ tiếng Việt | Maximum Matching: FMM, BMM, BiMM trên Trie |
| Lọc stopword | Hai tầng: hư từ ngữ pháp + TDV Salton |
| Trọng số văn bản | Sublinear TF + Smoothed IDF |
| Tìm kiếm | Inverted Index + Cosine Similarity |
| Đánh giá | Precision@K, Recall@K, F1@K, MAP, NDCG |
| Giao diện | Python + Streamlit |
| Quy mô dự kiến | 2.000–3.000 bài viết; 20 truy vấn gán nhãn |

#### 3.2. Những điểm cần chú ý được trợ lý nêu

- **Phạm vi lớn hơn yêu cầu gốc:** ngoài TF-IDF và Cosine còn có TDV, BiMM, UI giải thích điểm số, biểu đồ và nhiều bộ kiểm thử.
- **Không mặc định mục tiêu là kết quả thực nghiệm:** các yêu cầu như BiMM F1 cao nhất, C4 tốt nhất hay chỉ mục giảm trên 30% cần đo rồi mới kết luận.
- **Phải nhất quán giữa lập chỉ mục và xử lý truy vấn:** đặc biệt khi bảo vệ từ phủ định hoặc lọc stopword tầng 2.
- **Bố cục báo cáo có chỗ chưa đồng nhất:** kế hoạch có khung 5 chương, trong khi ghi chú cuối file đề cập 6 chương.

#### 3.3. Vai trò của ba nguồn hướng dẫn

1. **Đề bài giảng viên:** các hạng mục bắt buộc.
2. **Giáo trình:** thuật ngữ và lý thuyết.
3. **Tài liệu nhóm trưởng:** giải pháp kỹ thuật, tiến độ và phân công.

Trợ lý hỏi người dùng được giao M1, M2 hay M3; người dùng về sau xác nhận là **M2**.

---

## 4. Lần trao đổi 3 – Phân công M2 và hướng dẫn chi tiết

### Người dùng

> Tôi được phân công làm M2. Hãy phân tích chi tiết nhiệm vụ của tôi theo tài liệu nhóm trưởng và hướng dẫn cách thực hiện.

### Trợ lý phản hồi – Xác định vai trò

**M2 – Information Retrieval Specialist (Kỹ sư Chỉ mục và Truy xuất thông tin)** chịu trách nhiệm lõi tìm kiếm. M1 chuẩn hóa/tách từ, M2 biến các từ đó thành dữ liệu có thể lập chỉ mục và tìm kiếm, M3 tích hợp giao diện.

#### 4.1. Sáu nhóm nhiệm vụ chính của M2

| STT | Công việc | Sản phẩm |
|---:|---|---|
| 1 | Thu thập corpus, xây 20 truy vấn và nhãn liên quan | `corpus.jsonl`, `qrels.json` |
| 2 | Phân tích Term Discrimination Value | `tdv_engine.py` |
| 3 | Tính TF-IDF và Inverted Index | `tfidf_engine.py`, `inverted_index.py` |
| 4 | Tìm kiếm và xếp hạng bằng Cosine | `search_engine.py` |
| 5 | Đánh giá chỉ số và tốc độ phản hồi | `evaluate.py`, benchmark, CSV và biểu đồ |
| 6 | Viết báo cáo học thuật | Nội dung IR Chương 3, thực nghiệm Chương 4 |

#### 4.2. Quan hệ giữa M1 – M2 – M3

```text
                 M1 – Xử lý tiếng Việt
    Chuẩn hóa → Maximum Matching → Stopword Tầng 1
                           ↓
              M2 – Information Retrieval
    TDV → TF-IDF → Inverted Index → Cosine → Ranking
                           ↓
                     M3 – UI/UX
      Nhận câu truy vấn → Top-K → giải thích kết quả
```

Ví dụ truy vấn: `Tìm kiếm tài liệu về trí tuệ nhân tạo trong giáo dục`.

Sau khi M1 xử lý, M2 có thể nhận các token dạng `trí_tuệ_nhân_tạo`, `giáo_dục` (đây là ví dụ, thực tế phụ thuộc từ điển và stopword). M2 cần tra chỉ mục, tính TF-IDF/Cosine và trả về các bản ghi có thứ hạng, ID, tiêu đề, điểm số, đoạn trích để M3 hiển thị.

### 4.3. Tuần 1 – Chuẩn bị corpus và Ground Truth

**Mục tiêu nhóm trưởng:** 2.000–3.000 bài viết đa chủ đề. Ví dụ phân bổ 2.500 bài thành 5 nhóm, mỗi nhóm 500:

| Chủ đề trong kế hoạch | Số bài minh họa |
|---|---:|
| Công nghệ | 500 |
| Kinh tế | 500 |
| Thể thao | 500 |
| Giáo dục | 500 |
| Y tế | 500 |

Các bước làm sạch: loại bài quá ngắn (dưới 100 từ), chuẩn hóa chuỗi, khử trùng lặp bằng MD5, tạo `doc_id` và gắn `category`.

Ví dụ bản ghi JSONL:

```json
{
  "doc_id": 1,
  "title": "Ứng dụng trí tuệ nhân tạo trong giáo dục",
  "content": "Trí tuệ nhân tạo đang được ứng dụng trong nhiều hoạt động giáo dục...",
  "category": "Giáo dục",
  "source_url": "https://example.org/article"
}
```

`source_url` từng được trợ lý đề xuất thêm để truy vết xuất xứ; không phải một trường bắt buộc trong tài liệu phân công gốc.

**Ground Truth / qrels:** lập 20 truy vấn gồm câu ngắn, câu dài, câu phủ định và câu chuyên ngành. Gán nhãn tài liệu thực sự liên quan bằng phương pháp pooling; M1 kiểm tra chéo khoảng 20% mẫu.

Ví dụ câu truy vấn:

| ID | Ví dụ |
|---|---|
| Q01 | trí tuệ nhân tạo |
| Q02 | ứng dụng trí tuệ nhân tạo trong giáo dục |
| Q03 | bệnh không truyền nhiễm |
| Q04 | chính sách lãi suất ngân hàng |
| Q05 | đội tuyển bóng đá Việt Nam |

Ví dụ bản ghi `qrels.json`:

```json
{
  "query_id": "Q02",
  "query_text": "ứng dụng trí tuệ nhân tạo trong giáo dục",
  "relevant_doc_ids": [15, 26, 31]
}
```

Danh sách ID này chỉ là ví dụ, chưa được gán nhãn thực tế.

### 4.4. Tuần 2 – TDV, Inverted Index và TF-IDF

#### A. TDV (Term Discrimination Value)

Trong phương án nhóm trưởng, TDV đánh giá độ phân biệt của từ dựa trên thay đổi độ tương đồng trung bình của không gian tài liệu khi loại từ đó.

\[
TDV(t)=Q_{-t}-Q
\]

- \(Q\): độ tương đồng/mật độ trung bình trước khi loại từ.
- \(Q_{-t}\): độ tương đồng/mật độ trung bình sau khi loại từ \(t\).
- Quy tắc dự kiến: \(TDV(t)<0\) thì đưa từ vào danh sách stopword miền.

Các ví dụ dấu `−0.08`, `0.00`, `+0.12` từng được dùng để minh họa, không phải số liệu chạy thật.

File:

- `engine/indexer/tdv_engine.py`
- `data/indexes/tdv_stopwords.json`
- `data/indexes/tdv_scores.json`

**Cảnh báo kỹ thuật đã được nêu:** đoạn code TDV mẫu trong tài liệu nhóm trưởng cập nhật điểm tương đồng khi bỏ một chiều nhưng chưa tái chuẩn hóa/tính lại đúng toàn bộ trọng tâm mới. Vì thế phải đối chiếu trên tập nhỏ bằng cách tính trực tiếp trước khi tối ưu. Dấu âm của TDV cũng không tự động chứng minh từ vô nghĩa; phải đánh giá trên kết quả truy xuất.

#### B. Inverted Index

Bản chất: ánh xạ **từ khóa → posting list tài liệu chứa từ**.

Ví dụ:

```text
D1: trí_tuệ_nhân_tạo, giáo_dục
D2: trí_tuệ_nhân_tạo, công_nghệ
D3: giáo_dục, học_sinh

Inverted Index:
trí_tuệ_nhân_tạo → [D1, D2]
giáo_dục        → [D1, D3]
công_nghệ       → [D2]
học_sinh        → [D3]
```

Khi có query `trí_tuệ_nhân_tạo`, không phải quét toàn bộ corpus. Có thể lấy D1, D2 từ posting list. Cấu trúc thật nên lưu `doc_id`, tần suất hoặc trọng số TF-IDF.

File: `engine/indexer/inverted_index.py`.

#### C. TF-IDF

Phương án nhóm trưởng dùng **Sublinear TF** và **Smoothed IDF**:

\[
TF(t,d)=
\begin{cases}
1+\ln(f_{t,d}),&f_{t,d}>0\\
0,&f_{t,d}=0
\end{cases}
\]

\[
IDF(t)=\ln\left(\frac{N+1}{df(t)+1}\right)+1
\]

\[
w_{t,d}=TF(t,d)\times IDF(t)
\]

Trong đó \(N\) là số tài liệu; \(df(t)\) là số tài liệu có từ \(t\); \(f_{t,d}\) là số lần từ xuất hiện trong tài liệu.

Ví dụ số học đã trình bày: với \(N=100\), từ xuất hiện ba lần trong D1 và có trong năm tài liệu:

\[
TF=1+\ln3\approx2.099
\]

\[
IDF=\ln\left(\frac{101}{6}\right)+1\approx3.823
\]

\[
TF\!\cdot\!IDF\approx8.02
\]

Chuẩn hóa/tính chuẩn vector:

\[
\|\vec d\|_2=\sqrt{\sum_{t\in V}w_{t,d}^2}
\]

Nên kiểm chứng kết quả với `TfidfVectorizer` của scikit-learn, đảm bảo cùng cấu hình TF, IDF, tokenizer và normalization khi đối soát.

Files:

- `engine/indexer/tfidf_engine.py`
- `data/indexes/idf_weights.json`
- `data/indexes/doc_metadata.json`
- `tests/test_ir_math.py`

### 4.5. Tuần 3 – Search Engine, explain() và latency

**Hàm `search(query, top_k)`** sẽ:

1. Nhận truy vấn, gọi tiền xử lý/tách từ của M1.
2. Lọc stopword tầng 2 theo đúng cấu hình lập chỉ mục.
3. Vector hóa query bằng TF-IDF, dùng bảng IDF tính ngoại tuyến.
4. Lấy tập ứng viên thông qua hợp các posting list trong Inverted Index.
5. Tính Cosine similarity trên ứng viên.
6. Sắp xếp điểm giảm dần, trả về Top-K với `doc_id`, `title`, `score`, `snippet`.

Công thức:

\[
\cos(q,d)=\frac{\sum_{t\in V}w_{t,q}w_{t,d}}{\|\vec q\|_2\|\vec d\|_2}
\]

Code minh họa đã được chia sẻ:

```python
import math

def cosine_similarity(query_vector, doc_vector):
    dot_product = sum(
        weight * doc_vector.get(term, 0.0)
        for term, weight in query_vector.items()
    )

    query_norm = math.sqrt(
        sum(w * w for w in query_vector.values())
    )
    doc_norm = math.sqrt(
        sum(w * w for w in doc_vector.values())
    )

    if query_norm == 0 or doc_norm == 0:
        return 0.0

    return dot_product / (query_norm * doc_norm)
```

Đây là phiên bản giải thích công thức; phiên bản triển khai nên dùng posting list để tránh quét mọi vector.

**Hàm `explain(query, doc_id)`** phân rã đóng góp của các từ trong query vào tích vô hướng, ví dụ bảng:

| Token | Trọng số query | Trọng số tài liệu | Tích đóng góp |
|---|---:|---:|---:|
| `trí_tuệ_nhân_tạo` | 2.3 | 3.1 | 7.13 |
| `giáo_dục` | 1.8 | 2.5 | 4.50 |
| `công_nghệ` | 1.2 | 0.0 | 0.00 |

Các số trên chỉ minh họa.

**Benchmark latency:** chạy lặp lại 20 truy vấn, mỗi truy vấn 100 lần; tính mean và P95. Trưởng nhóm đặt mục tiêu trung bình dưới 30 ms, P95 dưới 50 ms, cần đo thực tế và nêu cấu hình máy. File `benchmarks/benchmark_latency.py`.

### 4.6. Tuần 4 – Ablation Study và báo cáo

So sánh bốn cấu hình lọc stopword:

| Mã | Cấu hình |
|---|---|
| C1 | Không lọc stopword (baseline) |
| C2 | Danh sách stopword tĩnh (tài liệu dự kiến 1.942 từ) |
| C3 | Chỉ lọc hư từ ngữ pháp Tầng 1 |
| C4 | Tầng 1 + TDV Salton Tầng 2 |

Cùng một corpus, bộ query, qrels và công thức TF-IDF, tính:

- Precision@K.
- Recall@K.
- F1@K.
- MAP.
- NDCG@K.

Ví dụ nếu Top-5 có 4 kết quả liên quan và corpus có 10 tài liệu liên quan cho query:

\[
P@5=4/5=0.8,\quad R@5=4/10=0.4
\]

Phải đo kết quả thật chứ không được tuyên bố C4 tốt nhất trước khi chạy thí nghiệm.

Sản phẩm:

- `evaluator/evaluate.py`
- `results/ablation_results.csv`
- `scripts/plot_results.py`
- `report/chapter3_ir.tex`
- `report/chapter4.tex`

### 4.7. Danh sách file M2 được tổng hợp trong lượt trao đổi

```text
scripts/build_corpus.py
data/corpus.jsonl
data/ground_truth/qrels.json
engine/indexer/tdv_engine.py
engine/indexer/inverted_index.py
engine/indexer/tfidf_engine.py
engine/indexer/search_engine.py
evaluator/evaluate.py
benchmarks/benchmark_latency.py
tests/test_ir_math.py
tests/test_search_engine.py
results/ablation_results.csv
report/chapter3_ir.tex
report/chapter4.tex
```

### 4.8. Các điều chỉnh triển khai được đề xuất

1. **Tạo TF-IDF + Cosine baseline trước khi tối ưu TDV** để có bộ kiểm chứng toán học.
2. **Chỉ mục và query phải thống nhất vocabulary:** nếu TDV loại một từ khỏi chỉ mục thì cố giữ lại từ đó trong query cũng không đủ để tìm tài liệu; cần cơ chế dự phòng hoặc chỉ mục baseline.
3. **Kiểm chứng TDV cả tính đúng lẫn thời gian** trên tập nhỏ, sau đó mới tối ưu cho hàng nghìn bài.
4. **Không ép kết quả theo mục tiêu:** các nhận định về tăng MAP, giảm 30% index, BiMM F1 cao nhất là giả thuyết/tiêu chí mong muốn.

Trợ lý đề xuất quy trình ưu tiên thực hành: thiết lập Python → thống nhất API → corpus/qrels → TF-IDF baseline → kiểm thử công thức → TDV → search/explain/benchmark → đánh giá bốn cấu hình và báo cáo.

Cuối lượt trao đổi, trợ lý hỏi tình trạng repo hiện tại để hướng dẫn sát với tiến độ thực tế.

---

## 5. Lần trao đổi 4 – Kiểm tra repository và bộ TF-IDF khởi đầu

### Người dùng

> đây là repo hiện tại, bắt đâu hướng dẫn

Đính kèm: `CSDL_DPT_Team7.zip`.

### Trợ lý phản hồi – Nội dung chính

Trợ lý cho biết đã kiểm tra ZIP và kết luận repo lúc đó là **khung ban đầu**, M2 chưa có code thuật toán. Các trạng thái được báo lại:

| Thành phần | Trạng thái được báo |
|---|---|
| Git repository/nhánh `main` | Có |
| GitHub Actions `lint.yml` | Có |
| Streamlit `app/Home.py` | Có khung |
| Trang Tokenizer / Analytics / Search | Mới là placeholder |
| `report/main.tex` | Có khung |
| `tdv_engine.py`, `tfidf_engine.py`, `search_engine.py` | Chưa có |
| Inverted Index | Chưa có |
| Corpus / qrels | Chưa có |
| README / requirements / tests | Chưa có |

Lưu ý được nêu: ZIP tên `Team7`, nhưng một số nội dung báo cáo/giao diện ghi `Nhóm 8`, cần thống nhất tên chính thức. Workflow sử dụng `flake8 --exit-zero`, tức có lỗi lint vẫn không khiến công việc CI thất bại.

### 5.1. Trợ lý đã chuẩn bị bộ code TF-IDF mẫu

Sản phẩm chia sẻ trước đó: `M2_Buoc1_TFIDF_Baseline.zip`, chứa bộ ví dụ ba tài liệu và năm kiểm thử. Trợ lý hướng dẫn giữ riêng phần M2, không đè các file M1/M3.

Lệnh mở repo và kiểm tra:

```bash
git status
python --version
```

Nếu Git sạch, tạo nhánh:

```bash
git switch -c feature/m2-tfidf-baseline
```

Tạo môi trường ảo:

```bash
python -m venv .venv
```

Kích hoạt trên PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Trên macOS/Linux:

```bash
source .venv/bin/activate
```

Cài pytest:

```bash
python -m pip install pytest
```

Cấu trúc file được bộ baseline đề xuất thêm:

```text
CSDL_DPT_Team7/
├── app/                      # M3
├── engine/
│   ├── engine.md
│   ├── __init__.py
│   └── indexer/
│       ├── __init__.py
│       └── tfidf_engine.py
├── data/
│   └── demo_m2_tokens.jsonl
├── scripts/
│   ├── __init__.py
│   └── demo_m2.py
├── tests/
│   └── test_tfidf_engine.py
└── report/
```

Chạy demo:

```bash
python -m scripts.demo_m2
```

Kết quả ví dụ đã được chia sẻ trong lượt trao đổi:

| Kết quả tìm kiếm với query `trí_tuệ_nhân_tạo giáo_dục` | Cosine |
|---|---:|
| D1 — AI trong giáo dục | 0.968439 |
| D2 — AI và công nghệ | 0.428046 |
| D3 — Học sinh và giáo dục | 0.428046 |

Sau đó chạy kiểm thử:

```bash
python -m pytest -q tests/test_tfidf_engine.py
```

Kỳ vọng của bộ baseline: `5 passed`.

**Quan trọng về tiến độ:** đây chỉ là hướng dẫn và code ví dụ, chưa phải bước lựa chọn corpus thực, chưa triển khai TDV, Inverted Index hay 20 qrels. Sau đó người dùng đã phản hồi cần **ưu tiên nguồn dữ liệu đầu vào theo kế hoạch nhóm trưởng**, và trợ lý điều chỉnh kế hoạch cho phù hợp.

---

## 6. Lần trao đổi 5 – Kiểm tra nguồn dataset

### Người dùng – Nội dung đề xuất nguồn dữ liệu

Người dùng chỉ rõ vấn đề: **M2 cần có dữ liệu đầu vào trước khi viết thuật toán**, trong khi các hướng dẫn trước chưa chỉ ra cách chọn nguồn thực tế. Người dùng đưa năm gợi ý:

#### 6.1. VNTC – Vietnamese News Corpus

- Link: <https://github.com/duyvuleo/VNTC>.
- Là corpus tin tức tiếng Việt, có nhiều chủ đề và nhãn chủ đề.
- Dễ đổi sang JSONL; phù hợp truy xuất và phân tích stopword miền.
- Khuyến nghị lấy thử 200–500 tài liệu ở 3–5 chủ đề (40–100 tài liệu/chủ đề), sau khi pipeline chạy ổn định mới mở rộng 1.000–3.000.

#### 6.2. Vietnamese NLP Resources

- Link người dùng cung cấp: <https://github.com/undertheseanlp/NLP-Vietnamese-Progress>.
- Là trang tổng hợp tài nguyên NLP tiếng Việt: corpus, classification, QA, tách từ, nhận dạng thực thể và công cụ.
- Dùng như danh mục để tìm nguồn và kiểm tra giấy phép, không nhất thiết làm corpus chính.

#### 6.3. UIT-ViNewsQA

- Link người dùng cung cấp: <https://github.com/UIT-NLP/UIT-ViNewsQA>.
- Dataset hỏi đáp tin tức tiếng Việt.
- Hữu ích để tham khảo cách tổ chức tài liệu, câu hỏi, liên hệ query – nội dung và relevance judgment.
- Không đồng nhất với corpus truy xuất văn bản.

#### 6.4. UIT-ViQuAD

- Link người dùng cung cấp: <https://github.com/vanlongpham/UIT-ViQuAD>.
- Có đoạn văn và câu hỏi tiếng Việt.
- Có thể dùng để kiểm tra câu truy vấn dài, truy vấn cụ thể hoặc query–document mapping.
- Không nên dùng làm corpus tin tức đa chủ đề chính.

#### 6.5. VLSP datasets

- Link: <https://vlsp.org.vn/resources>.
- Có tài nguyên tách từ, POS, NER, phân loại, cảm xúc.
- Đặc biệt hữu ích cho M1 khi đánh giá Maximum Matching với dữ liệu gán nhãn chuẩn.
- Có thể phải đăng ký/có giới hạn phân phối.

### Trợ lý phản hồi – Kết quả kiểm tra nguồn

Trợ lý đã truy cập các liên kết, xác nhận **VNTC là lựa chọn khả thi nhất làm corpus chính của M2**, các nguồn còn lại chủ yếu hỗ trợ tham khảo hoặc đánh giá riêng.

| Nguồn | Đánh giá |
|---|---|
| VNTC | Lựa chọn chính |
| `undertheseanlp/NLP-Vietnamese-progress` | Danh mục tài nguyên, không phải corpus để nạp trực tiếp |
| UIT-ViNewsQA | Liên kết GitHub đã đưa trả 404 lúc kiểm tra; có mô tả dataset qua nguồn nghiên cứu UIT |
| UIT-ViQuAD | Liên kết GitHub đã đưa trả 404 lúc kiểm tra; có mô tả dataset qua nguồn nghiên cứu UIT |
| VLSP | Hữu ích cho bộ gold segmentation của M1, một số dữ liệu cần đăng ký |

### 6.6. VNTC – Thông tin đã được kiểm tra

Repository: <https://github.com/duyvuleo/VNTC>.

Cấu trúc GitHub có:

```text
Data/
├── 10Topics/
│   └── Ver1.1/
│       ├── Stats.txt
│       ├── Train_Full.rar
│       └── Test_Full.rar
└── 27Topics/
```

Theo `Stats.txt` thuộc bản 10Topics/Ver1.1:

- Train: **33.759** tệp.
- Test: **50.373** tệp.
- Tổng cộng theo file thống kê: **84.132** tệp.

Bảng Train theo nhãn:

| Mã | Nhãn gốc | Số tệp Train |
|---|---|---:|
| XH | Chính trị – Xã hội | 5.219 |
| DS | Đời sống | 3.159 |
| KH | Khoa học | 1.820 |
| KD | Kinh doanh | 2.552 |
| PL | Pháp luật | 3.868 |
| SK | Sức khỏe | 3.384 |
| TG | Thế giới | 2.898 |
| TT | Thể thao | 5.298 |
| VH | Văn hóa | 3.080 |
| VT | Vi tính | 2.481 |

Đường dẫn:

- Thư mục tải: <https://github.com/duyvuleo/VNTC/tree/master/Data/10Topics/Ver1.1>
- Train: <https://github.com/duyvuleo/VNTC/blob/master/Data/10Topics/Ver1.1/Train_Full.rar>
- Test: <https://github.com/duyvuleo/VNTC/blob/master/Data/10Topics/Ver1.1/Test_Full.rar>
- Thống kê: <https://raw.githubusercontent.com/duyvuleo/VNTC/master/Data/10Topics/Ver1.1/Stats.txt>

Kích thước tệp nén trên GitHub được kiểm tra khoảng 49,2 MB (Train) và 77,3 MB (Test).

#### 6.7. Khác biệt với 5 chủ đề của nhóm trưởng

Nhóm trưởng muốn **Công nghệ, Kinh tế, Thể thao, Giáo dục, Y tế**, nhưng VNTC 10 Topics **không có nhãn Giáo dục riêng**.

Phương án thay thế do trợ lý đề xuất (chưa phải quyết định chính thức của nhóm trưởng):

| Nhãn VNTC | Chủ đề sử dụng cho đồ án |
|---|---|
| VT – Vi tính | Công nghệ thông tin |
| KD – Kinh doanh | Kinh tế – Kinh doanh |
| TT – Thể thao | Thể thao |
| SK – Sức khỏe | Sức khỏe – Y tế |
| KH – Khoa học | Khoa học (thay cho Giáo dục) |

Nếu cần đúng Giáo dục, phải bổ sung dữ liệu khác và ghi rõ xuất xứ; không tự đổi nhãn Khoa học thành Giáo dục.

#### 6.8. Kế hoạch dùng VNTC

- Giai đoạn đầu: **500 tài liệu**, 100 tài liệu mỗi chủ đề.
- Sau tích hợp: **1.000 tài liệu**.
- Thực nghiệm chính: **2.500 tài liệu**, cân bằng năm chủ đề nếu nhóm trưởng đồng ý.
- Chỉ cần tải `Train_Full.rar` ban đầu; chưa cần Test.
- Giải nén vào `data/raw/VNTC/`.
- Giữ corpus gốc chưa tokenization/stopword; để M1 xử lý về sau.
- Tạo JSONL UTF-8 với các trường `doc_id`, `title`, `content`, `category`; nên bổ sung `source_dataset` và `source_file`.
- Nếu bản tin không có title rõ ràng, không bịa: cần quy tắc suy ra từ dữ liệu, đồng thời đánh dấu title là suy ra.
- Kiểm tra encoding thực tế trước khi chuyển đổi; có tài liệu ngoài nêu VNTC dùng UTF-16, nhưng không mặc định nếu chưa đọc file gốc.
- Random seed cố định khi lấy mẫu để có thể tái lập kết quả.

### 6.9. Vấn đề quyền sử dụng

Repo VNTC có giấy phép MIT trong GitHub, nhưng `Stats.txt` cho thấy nguồn tin đến từ nhiều báo (VnExpress, Tuổi Trẻ, Thanh Niên, Người Lao Động). Trợ lý lưu ý giấy phép repo không nhất thiết đồng nghĩa với quyền công khai phân phối lại mọi văn bản báo chí. Nên ghi nguồn, trích dẫn nghiên cứu gốc và xem điều kiện sử dụng corpus trước khi đưa lên GitHub công khai.

### 6.10. Qrels không được tự tạo từ nhãn chủ đề

Đây là điểm trọng yếu trong lượt trao đổi:

**VNTC là bộ dữ liệu phân loại văn bản, không phải tập qrels chuẩn của Information Retrieval.**

Ví dụ D1 và D2 cùng thuộc nhãn Khoa học, nhưng D1 nói về AI, D2 nói về thiên văn. Với truy vấn `ứng dụng trí tuệ nhân tạo`, D1 có thể liên quan, D2 có thể không. Vì vậy:

- `category` dùng quản lý corpus/phân tích chủ đề.
- `relevant_doc_ids` trong qrels phải được đánh giá **theo từng truy vấn**.
- Cần pooling + gán nhãn thủ công, không coi mọi bài cùng category là relevant.

### 6.11. Các nguồn phụ trợ khác

- Danh mục Underthesea: <https://github.com/undertheseanlp/NLP-Vietnamese-progress>.
- Công cụ Underthesea: <https://github.com/undertheseanlp/underthesea>.
- Tổng hợp mô tả bộ dữ liệu UIT: <https://github.com/kietnv/VietnameseDatasets>.
- VLSP: <https://vlsp.org.vn/resources>.

Theo tài liệu tham khảo, UIT-ViQuAD có trên 23.000 cặp hỏi–đáp; UIT-ViNewsQA có trên 22.000 cặp hỏi–đáp trong miền tin sức khỏe. Trợ lý lưu ý các corpus UIT có thể cần ký thỏa thuận để nhận dữ liệu. Hai bộ QA không được dùng thay qrels truy xuất nếu chưa chuyển đổi và đánh giá lại ở cấp tài liệu.

### 6.12. Quyết định tạm thời sau lượt kiểm tra dataset

| Thành phần | Phương án được đề xuất |
|---|---|
| Corpus chính | VNTC 10Topics/Ver1.1 |
| Bản đầu | Train_Full |
| Mẫu đầu tiên | 500 tài liệu, năm chủ đề |
| Mẫu chính thức | 2.500 tài liệu |
| Định dạng | JSONL UTF-8 |
| Query set | 20 truy vấn nhóm tạo |
| Qrels | Pooling + gán nhãn thủ công |
| Dữ liệu M1 nhận | Văn bản gốc chưa tách từ |
| Dữ liệu M2 nhận từ M1 | Token đã tiền xử lý |

Trợ lý yêu cầu tải VNTC, giải nén, rồi gửi vài tệp TXT mẫu để kiểm tra trước khi viết chính thức `build_corpus.py`.

---

## 7. Lần trao đổi 6 – Bắt đầu theo kế hoạch nhóm trưởng, từng bước

### Người dùng

> giờ hãy hướng dẫn từng bước để bám sát kế hoạch trưởng nhóm phân công

### Trợ lý phản hồi – Kế hoạch triển khai theo thứ tự

Trợ lý đồng ý thực hiện từng bước nhỏ, **xong và kiểm tra bước trước mới sang bước sau**, không làm phần thuật toán khi corpus chưa có.

#### 7.1. Lộ trình bốn tuần

| Giai đoạn | Nhiệm vụ | Sản phẩm |
|---|---|---|
| Tuần 1 | Corpus 2.000–3.000 bài | `corpus.jsonl` |
| Tuần 1 | 20 queries + gán nhãn | `qrels.json` |
| Tuần 2 | TDV Salton | `tdv_engine.py` |
| Tuần 2 | Inverted Index + TF-IDF | File chỉ mục/trọng số |
| Tuần 3 | Search + Cosine | `search_engine.py` |
| Tuần 3 | Benchmark latency | Báo cáo thời gian |
| Tuần 4 | Ablation bốn cấu hình | CSV và biểu đồ |
| Tuần 4 | Báo cáo | Chương 3 và 4 |

#### 7.2. Việc đang thực hiện: Tuần 1 / Nhiệm vụ 1 / Ngày 1–4

**Mục tiêu:** chuẩn bị VNTC gốc rồi mới chọn, làm sạch và xuất `data/corpus.jsonl`.

**Bước 1.1 – Mở repo trong VS Code**

Kiểm tra Git:

```powershell
git status
```

Nếu repo không có thay đổi cần lưu, tạo nhánh M2:

```powershell
git switch -c feature/m2-corpus
```

Mục đích là tách công việc của M2 khỏi file M1/M3. Không dùng `reset --hard` nếu có thay đổi chưa lưu.

**Bước 1.2 – Tải Train_Full.rar**

Mở <https://github.com/duyvuleo/VNTC/tree/master/Data/10Topics/Ver1.1> rồi tải `Train_Full.rar` (khoảng 49 MB). Chưa cần tải `Test_Full.rar`.

**Bước 1.3 – Tạo thư mục cho dữ liệu gốc**

```powershell
New-Item -ItemType Directory -Force -Path "data/raw/VNTC"
```

Giải nén RAR bằng WinRAR hoặc 7-Zip vào `CSDL_DPT_Team7/data/raw/VNTC/`. Giữ nguyên cấu trúc thư mục và tên file, chưa đổi tên/chọn dữ liệu.

Cấu trúc dự kiến:

```text
CSDL_DPT_Team7/
├── app/
├── engine/
├── report/
└── data/
    └── raw/
        └── VNTC/
            └── ... các file sau giải nén
```

**Bước 1.4 – Kiểm tra dữ liệu giải nén trong PowerShell**

Liệt kê thư mục:

```powershell
Get-ChildItem .\data\raw\VNTC -Directory -Recurse |
    Select-Object -First 20 FullName
```

Đếm số tệp `.txt`:

```powershell
(Get-ChildItem .\data\raw\VNTC -Recurse -File -Filter *.txt |
    Measure-Object).Count
```

Hiển thị 5 đường dẫn `.txt`:

```powershell
Get-ChildItem .\data\raw\VNTC -Recurse -File -Filter *.txt |
    Select-Object -First 5 FullName
```

Cần xác minh số file, cấu trúc nhãn và encoding thực, tránh lỗi dấu tiếng Việt khiến Maximum Matching xử lý sai.

**Bước 1.5 – Thống nhất năm chủ đề với trưởng nhóm**

- Công nghệ ↔ Vi tính.
- Kinh tế ↔ Kinh doanh.
- Thể thao ↔ Thể thao.
- Y tế ↔ Sức khỏe.
- Giáo dục: VNTC không có; trợ lý đề nghị dùng Khoa học thay thế, **phải chờ đồng ý của trưởng nhóm**.

#### 7.3. Checklist Bước 1

- [ ] Tạo nhánh Git riêng cho M2.
- [ ] Tải `Train_Full.rar`.
- [ ] Giải nén vào `data/raw/VNTC/`.
- [ ] Xác nhận có tệp `.txt` và các thư mục chủ đề.
- [ ] Có đường dẫn năm tệp `.txt` mẫu và kiểm tra sơ bộ mã hóa.

Lưu ý: không commit RAR/toàn bộ corpus thô lên GitHub trước khi xem quyền phân phối. Có thể thêm `data/raw/` vào `.gitignore`.

#### 7.4. Bước tiếp theo sau khi hoàn thành checklist

Sau khi người dùng thực hiện xong Bước 1 và gửi ảnh cây thư mục, kết quả lệnh, cùng 3–5 file TXT mẫu, trợ lý sẽ hướng dẫn viết:

`script/build_corpus.py` / **đường dẫn chính thống đã thống nhất:** `scripts/build_corpus.py`.

Các chức năng cần viết:

1. Đọc tệp văn bản VNTC đúng encoding.
2. Chuẩn hóa Unicode NFC, chuyển UTF-8 khi ghi JSONL.
3. Bỏ bài dưới 100 từ theo quy tắc nhóm trưởng.
4. Khử trùng lặp MD5.
5. Chọn mẫu cân bằng theo chủ đề, dùng seed cố định.
6. Tạo `doc_id`, `title`, `content`, `category` (và metadata nguồn nếu thống nhất).
7. Xuất `data/corpus.jsonl` và thống kê chất lượng.

Người dùng chưa gửi kết quả chạy các lệnh trên trong lịch sử được biên tập tại thời điểm này.

---

## 8. Lần trao đổi 7 – Yêu cầu xuất lịch sử thành Markdown

### Người dùng

> hãy viết toàn bộ lịch sử trao đổi thành lại file md

### Kết quả

Tài liệu Markdown này được tạo để lưu lại các yêu cầu, tệp đã gửi, phản hồi và các điểm kỹ thuật/phạm vi/tiến độ đã được trao đổi từ lúc giới thiệu đề tài đến hết hướng dẫn **Tuần 1 – Bước 1**.

---

## 9. Tổng hợp quyết định, tồn đọng và việc tiếp theo

### 9.1. Những điểm đã thống nhất trong trao đổi

- Người dùng thuộc **M2**.
- M2 chịu trách nhiệm **corpus, qrels, TDV, TF-IDF, Inverted Index, Search Engine, Cosine, evaluation, benchmarking và nội dung báo cáo tương ứng**.
- M2 phối hợp với M1 thông qua API token hóa/lọc tầng 1, phối hợp với M3 thông qua API tìm kiếm/giải thích kết quả.
- **Ưu tiên corpus thực trước thuật toán**, bám nhiệm vụ tuần 1 của trưởng nhóm.
- **VNTC 10Topics/Ver1.1** là lựa chọn chính được đề xuất cho corpus ban đầu.
- **Nhãn chủ đề VNTC không thay thế nhãn relevance trong qrels**.
- Tiến độ chia từng bước: thực hiện → kiểm tra → chuyển sang bước tiếp theo.

### 9.2. Những điều còn phải xác nhận

| Vấn đề | Trạng thái |
|---|---|
| Nhóm chính thức tên Team7 hay Nhóm 8 | Cần thống nhất với trưởng nhóm; repo trước có chỗ không đồng nhất |
| Chủ đề thứ 5 là Giáo dục hay thay bằng Khoa học | Chưa được trưởng nhóm xác nhận |
| Đã tải/giải nén được VNTC chưa | Chưa có kết quả người dùng gửi lại |
| Encoding thực tế của các tệp VNTC | Chưa kiểm tra trên tệp mẫu người dùng tải |
| Nguồn và giấy phép phân phối corpus | Cần ghi nguồn và kiểm tra điều khoản |
| API từ M1 bàn giao cho M2 | Chưa có xác nhận bản triển khai thực tế |
| Bộ 20 query/qrels và quy trình gán nhãn | Chưa triển khai |
| TDV hiện thực đúng công thức, có cải thiện IR không | Chưa kiểm chứng |
| Mốc thời gian hoàn thành từng tuần | Kế hoạch dự kiến, chưa có lịch thực tế |
| Bố cục báo cáo 5 hay 6 chương | Cần thống nhất |

### 9.3. Việc kế tiếp, theo đúng thứ tự phân công

**Ngay lúc này:** hoàn thành các lệnh kiểm tra repo và dữ liệu VNTC ở mục 7.2, gửi kết quả và một số tệp TXT mẫu.

**Sau đó:** viết và kiểm thử `scripts/build_corpus.py`; tạo corpus 500 bài làm mẫu, rồi mở rộng dần. Khi corpus có nội dung thật, làm 20 truy vấn, qrels. Tiếp theo xây TDV, Inverted Index, TF-IDF, Search Engine, benchmarks và báo cáo. Nếu làm TF-IDF baseline trước TDV để kiểm chứng toán học, cần ghi rõ đây là thay đổi nhỏ về thứ tự kỹ thuật so với tiến độ dự kiến trong tài liệu trưởng nhóm.

---

## 10. Phụ lục: đường dẫn và sản phẩm

### 10.1. Dataset và nguồn học thuật

- **VNTC:** <https://github.com/duyvuleo/VNTC>
- **VNTC 10Topics/Ver1.1:** <https://github.com/duyvuleo/VNTC/tree/master/Data/10Topics/Ver1.1>
- **Train_Full.rar:** <https://github.com/duyvuleo/VNTC/blob/master/Data/10Topics/Ver1.1/Train_Full.rar>
- **Stats.txt:** <https://raw.githubusercontent.com/duyvuleo/VNTC/master/Data/10Topics/Ver1.1/Stats.txt>
- **NLP-Vietnamese-progress:** <https://github.com/undertheseanlp/NLP-Vietnamese-progress>
- **Underthesea:** <https://github.com/undertheseanlp/underthesea>
- **VietnameseDatasets / UIT:** <https://github.com/kietnv/VietnameseDatasets>
- **Link UIT-ViNewsQA do người dùng đề xuất (đã báo không tìm thấy ở thời điểm tra cứu):** <https://github.com/UIT-NLP/UIT-ViNewsQA>
- **Link UIT-ViQuAD do người dùng đề xuất (đã báo không tìm thấy ở thời điểm tra cứu):** <https://github.com/vanlongpham/UIT-ViQuAD>
- **VLSP:** <https://vlsp.org.vn/resources>

### 10.2. Kế hoạch file trong repo M2

```text
CSDL_DPT_Team7/
├── data/
│   ├── raw/
│   │   └── VNTC/
│   ├── corpus.jsonl
│   ├── ground_truth/
│   │   └── qrels.json
│   └── indexes/
│       ├── inverted_index.pkl
│       ├── idf_weights.json
│       ├── doc_metadata.json
│       ├── tdv_scores.json
│       └── tdv_stopwords.json
├── engine/
│   └── indexer/
│       ├── tdv_engine.py
│       ├── inverted_index.py
│       ├── tfidf_engine.py
│       └── search_engine.py
├── scripts/
│   ├── build_corpus.py
│   └── plot_results.py
├── evaluator/
│   └── evaluate.py
├── benchmarks/
│   └── benchmark_latency.py
├── tests/
│   ├── test_ir_math.py
│   └── test_search_engine.py
├── results/
│   └── ablation_results.csv
└── report/
    ├── chapter3_ir.tex
    └── chapter4.tex
```

### 10.3. Những ghi chú để dùng tài liệu an toàn

1. Mục tiêu, benchmark kỳ vọng và con số minh họa không được trình bày như dữ liệu đã đo thực tế.
2. Quy mô, năm chủ đề và công thức chọn mẫu có thể được điều chỉnh sau khi trưởng nhóm thống nhất.
3. Dataset VNTC cung cấp nhãn **phân loại**, không phải các nhãn liên quan theo truy vấn của IR.
4. File Markdown này ghi nhận **lịch sử hướng dẫn và quyết định đến thời điểm hiện tại**; nó không khẳng định rằng người dùng đã tự chạy thành công các lệnh được đề xuất.
5. Khi báo cáo cuối cùng, cần trích dẫn nguồn dataset, phương pháp gán nhãn, công thức, điều kiện thí nghiệm, môi trường chạy và kết quả đúng theo thực nghiệm.

---

**Kết thúc bản ghi lịch sử trao đổi.**
