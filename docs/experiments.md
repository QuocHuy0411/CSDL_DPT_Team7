# Thực nghiệm M2: TDV, đánh giá ablation, đo độ trễ

Tài liệu này mô tả **cách cài đặt** và **trạng thái số đo thật** của phần thực
nghiệm M2. Nguyên tắc xuyên suốt: mọi con số trong đây phải truy được về một
lần chạy cụ thể, và khi chưa đủ điều kiện thì ghi *chưa đo*, không ghi số 0.

Tài liệu tham khảo thư viện: xem `docs/academic_sources.md`. Phần "TDV" dưới đây
ghi rõ chỗ nào là lựa chọn cài đặt của nhóm và chỗ nào bám công thức gốc.

---

## 1. Công thức TDV đã cài đặt

File: `engine/indexer/tdv_engine.py`.

### 1.1 Mật độ không gian tài liệu

Với $N$ tài liệu đã vector hoá theo TF-IDF và chuẩn L2, mật độ dùng trong TDV là
mật độ cosine pairwise **chính xác**, loại trừ self-pair:

$$
D(X)=\frac{1}{N(N-1)}\sum_{i\neq j}\cos(x_i,x_j)
$$

* Mẫu số cố định là $N(N-1)$, tức số cặp *có thứ tự*. Nhân đôi so với cặp vô thứ
  tự là cố ý, để $D$ là trung bình mỗi cặp đúng một lần.
* $N<2$ thì $D=0$.
* Vectơ tất cả bằng không thì $\cos=0$ với mọi tài liệu khác.

### 1.2 Trọng số sơ bộ

TDV được đo trên TF-IDF **sơ bộ**, tức trước khi áp bất kỳ bộ lọc TDV nào:

$$
w_{t,d}=\bigl(1+\ln f_{t,d}\bigr)\cdot\left(\ln\frac{N+1}{df(t)+1}+1\right),
\qquad \vec x_d=\frac{w_d}{\lVert w_d\rVert_2}
$$

**IDF sơ bộ được giữ nguyên khi loại một chiều.** Nếu tính lại IDF sau mỗi lần
loại thì hai hiệu ứng (đổi trọng số và đổi hình học) trộn vào nhau và con số
không còn là TDV của riêng một từ. Sau khi chọn được stoplist, TF-IDF *chính
thức* của C4 được dựng lại từ đầu (việc này do `scripts/build_indexes.py` thực
hiện), không tái dùng ma trận đo TDV.

### 1.3 Định nghĩa TDV và điểm cần cẩn thận

$$
TDV(t)=D_{-t}-D_{base}
$$

trong đó $D_{-t}$ là mật độ sau khi **xoá hẳn cột $t$**. Hai việc bắt buộc sau khi
xoá chiều, và là hai việc dễ sót:

1. **Tái chuẩn hóa L2 từng tài liệu.** Nếu chỉ đặt phần tử về 0 mà không chia lại
   chuẩn thì độ dài vectơ khác nhau làm sai cosin.
2. **Cập nhật tổng vector (centroid).** Mật độ phụ thuộc toàn bộ ma trận, không
   phụ thuộc riêng từng tài liệu.

Cài đặt làm đúng cả hai. Với $v_i$ là thành phần chiều $t$ của tài liệu $i$ và
$\lVert x_i\rVert=1$, chuẩn sau khi xoá là $n_i=\sqrt{\lVert x_i\rVert^2-v_i^2}$,
nên với $u_i=1/n_i$ (và $u_i=0$ khi vectơ sau khi xoá bằng không):

$$
\sum_{i\neq j} u_iu_j\,(S_{ij}-v_iv_j)
=\lVert X^{\mathsf T}u\rVert^2-\sum_i u_i^2S_{ii}-\Bigl(\langle u,v\rangle^2-\sum_i u_i^2v_i^2\Bigr)
$$

Nhờ vậy chỉ cần duyệt posting list của từ đang xét thay vì dựng lại ma trận cosin
đầy đủ cho từng từ. Hàm `reference_scores()` trong cùng file làm đúng theo cách
"chậm nhưng hiển nhiên": xoá cột, chuẩn hóa lại, nhân ma trận cosin đầy đủ.

> **Khác biệt với bài báo gốc.** Bài báo Salton định nghĩa TDV theo mật độ không
> gian; việc *chọn* mật độ là trung bình cosin trên các cặp phân biệt, giữ mẫu số
> $N(N-1)$ cố định, coi vectơ không là cosin 0, và dùng dung sai số $10^{-12}$ là
> **lựa chọn cài đặt của nhóm** để thống nhất và kiểm chứng được, không phải khẳng
> định rằng mọi chi tiết hiện thực là nguyên văn từ bài báo.

### 1.4 Một lần duy nhất, không loại nối tiếp

TDV của mọi từ được tính trên **cùng một ma trận gốc**. Không lấy điểm của từ đã bị
loại ở bước trước rồi mới xoá thêm từ sau. `tests/test_tdv.py` có một fixture cố
ý cho kết quả khác nhau giữa hai cách và khẳng định engine theo cách one-shot.

### 1.5 Ngưỡng lọc

```
giữ từ nếu  TDV(t) >= -tolerance      với  tolerance = 1e-12
```

$10^{-12}$ là **dung sai số học**, không phải siêu tham số IR; nó được ghi vào
`metadata_` của mọi lần chạy. Không ép đường cong hình chữ U: một từ có TDV
âm nghĩa là khi bỏ nó đi mật độ tăng lên, và việc bỏ nó có cải thiện truy xuất hay
không phải đo bằng evaluator mới biết.

### 1.6 Kết quả TDV trên corpus chính

Đo bằng `TDVEngine().fit(tokenized_docs)` trên toàn bộ 2.500 tài liệu, 10 lĩnh vực,
một lần:

| Đại lượng | Giá trị |
| --- | ---: |
| Số tài liệu | 2.500 |
| Số từ trong từ vựng | 17.727 |
| $D_{base}$ | 0,09666545 |
| $\min TDV$ | $-1{,}08918 \times 10^{-3}$ |
| $\max TDV$ | $5{,}97833 \times 10^{-5}$ |
| Số từ có $TDV<-10^{-12}$ | 498 |
| Số từ bị loại sau khi bỏ guard ngữ nghĩa | 494 |
| Từ bị guard giữ lại | `chưa`, `chẳng`, `không`, `vì` |
| Thời gian chạy TDV | 11,65 s |

Nguồn: `data/indexes/C4/tdv_metadata.json` và `data/indexes/C4/build_metadata.json`.

---

## 2. Giao diện đánh giá

### 2.1 Định nghĩa chỉ số

File `evaluator/metrics.py`.

| Chỉ số | Công thức | Mẫu số |
| --- | --- | --- |
| $P@K$ | số trúng trong Top-$K$ | $K$ cố định, kể cả khi trả về ít hơn $K$ |
| $Recall@K$ | số trúng trong Top-$K$ | tổng số relevant **đã confirmed** trong qrels |
| $F1@K$ | $2PR/(P+R)$ | trung bình điều hòa |
| $AP$ | $\frac{1}{R}\sum_{i=1}^{L}P@i\cdot rel_i$ | $R$ = tổng relevant trong qrels |
| $MAP$ | trung bình macro của $AP$ trên **mọi** query | query không có relevant vẫn tính 0 và vẫn giữ trong mẫu |
| $NDCG@K$ | gain $2^{rel}-1$, giảm giám $\log_2(rank+1)$ | IDCG lấy từ qrels |

Ba điều dễ sai đã được khoá bằng kiểm thử:

* **AP không cắt theo $K$.** $L$ là độ dài ranking thực tế. Bộ đánh giá từ chối
  chạy nếu `--ranking-depth` nhỏ hơn số tài liệu corpus, trừ khi người dùng cố ý
  `--allow-truncated-depth`; khi đó kết quả được ghi nhãn `AP@depth` chứ không
  gọi là MAP.
* **Không có khái niệm "MAP@K".** Cột `map` luôn kèm `map_scope` ghi rõ phạm vi.
* **Query rỗng relevant không bị loại.** Giá trị 0 của nó vẫn ở trong bình quân và
  được đếm riêng ở `n_zero_relevant_queries`.

### 2.2 Ranking rỗng là kết quả hợp lệ

Với C4 thuần, một truy vấn mà mọi token đều đã bị lọc sẽ trả về danh sách rỗng.
Đây là hệ quả đúng của thiết kế, được giữ nguyên trong mọi chỉ số (không loại bỏ,
vì loại bỏ sẽ làm đẹp số) và được đếm ở `n_queries_with_empty_result` cùng danh
sách `empty_result_query_ids`.

### 2.3 Cổng chặn đánh giá chính thức

`validate_official_run()` chặn kết quả chính thức nếu có bất kỳ mục nào sau:

| Mã | Điều kiện |
| --- | --- |
| `expected_corpus_hash_missing` | không xác định được SHA256 corpus từ manifest khóa hoặc từ file corpus |
| `qrels_corpus_hash_mismatch`, `queries_corpus_hash_mismatch`, `index_corpus_hash_mismatch` | lệch hash |
| `qrels_corpus_hash_missing`, `index_corpus_hash_missing` | thiếu hash |
| `insufficient_queries` | dưới 20 query |
| `qrels_status_not_final` | `status` của qrels chưa phải `human_annotation_confirmed` |
| `pending_judgments` | còn phán quyết `pending` trong phạm vi đánh giá |
| `confirmed_without_relevance` | nhãn confirmed nhưng `relevance` còn null |
| `confirmed_without_assessor` | nhãn confirmed mà không ghi người gán |
| `duplicate_judgments` | trùng cặp `(query_id, doc_id)` |
| `invalid_relevance_grade` | `relevance` ngoài $\{0,1,2\}$ |
| `unknown_query_id_in_qrels` | qrels chứa query không có trong `queries.json` |
| `query_without_confirmed_judgments` | query chưa có nhãn confirmed nào |
| `qrels_schema_version_unsupported`, `queries_schema_version_unsupported` | `schema_version` khác 1 |
| `qrels_queries_hash_missing` | qrels có phán quyết nhưng không ghi `queries_sha256` |
| `queries_hash_mismatch` | `queries.json` đã bị sửa nên hash không còn khớp hash mà nhãn được gán cho |
| `queries_hash_unavailable` | không tính được hash của `queries.json` để đối chiếu |
| `preprocessing_not_official` | tiền xử lý chưa official |
| `preprocessing_metadata_inconsistent` | `preprocessing_status` và `official_preprocessing`/`m1_verified` lệch nhau |
| `development_tokenizer` | đang dùng tokenizer baseline của M1 |
| `fixture_tier1_filter` | đang dùng fixture tầng 1 |

Hai dòng cuối là **chặn tuyệt đối**: metadata ghi nhầm `official` cũng không mở được.
`official` chỉ đúng khi `preprocessing_status == "official"` **và**
`official_preprocessing` đúng **và** `preprocessing.m1_verified` đúng.

SHA256 corpus **không bao giờ** lấy từ qrels. Nguồn được phép là manifest đã khóa,
hoặc hash tính trực tiếp từ `data/corpus.jsonl`; nếu có cả hai thì phải khớp nhau.

### 2.3.1 Khoá hash của bộ truy vấn

Cặp truy vấn–tài liệu trong qrels được người gán chấm theo *nội dung* câu truy vấn.
Nếu `queries.json` bị sửa mà giữ nguyên `query_id`, cùng một nhãn người sẽ bị áp
nhầm cho câu khác mà không có dấu hiệu gì. Vì vậy qrels phải ghi
`queries_sha256`, và bộ đánh giá đối chiếu với hash thật của `queries.json`.

Đây là lỗi toàn vẹn dữ liệu chứ không phải chuyện trạng thái chính thức, nên nó
chặn **mọi** lần chạy kể cả development, và bị từ chối trước khi ghi bất kỳ tệp
kết quả nào. Benchmark cũng kiểm tra cặp hash này và từ chối đo nếu lệch.

### 2.4 Khi chưa đủ nhãn thì không xuất số 0

Nếu bất kỳ query nào trong phạm vi đánh giá chưa có nhãn confirmed, `evaluate()`
trả `metrics_status = "unmeasured"` và **không ghi** `per_query.csv` /
`summary.csv`. Chỉ ghi `run.json` với hash, trạng thái tiền xử lý, số byte chỉ mục và
lý do. Lý do: dữ liệu chưa có nhãn thì mọi chỉ số đều bằng 0 một cách máy móc, và
những số 0 đó dễ bị đọc nhầm là "hệ thống không tìm được gì".

### 2.5 Kết quả fixture không được lẫn vào kết quả chính thức

`assert_result_target()` chặn ghi kết quả không chính thức vào `results/ablation`,
`results/latency`, `results/ablation_fallback`. Kết quả chạy thử phải ghi ra thư mục
khác, ví dụ `results/development_eval`.

---

## 3. Lệnh chạy

```powershell
# Xây chỉ mục (do lead chạy, không chạy song song)
python scripts/build_indexes.py --m1-verified --configurations C1 C2 C3 C4

# Đánh giá chính thức: chỉ chạy được khi qrels confirmed + preprocessing official
python -m evaluator.evaluate --k 10

# Chạy tận dụng / kiểm thử (kết quả ghi ra thư mục KHÔNG phải kết quả chính thức)
python -m evaluator.evaluate --allow-development --out results/development_eval

# Đo độ trễ và dung lượng
python -m benchmarks.benchmark_latency --repeat 100 --warmup 5

# Vẽ biểu đồ (tự bỏ qua biểu đồ thiếu dữ liệu)
python scripts/plot_results.py
```

Cờ chính:

| Lệnh | Cờ | Ý nghĩa |
| --- | --- | --- |
| `evaluator.evaluate` | `--k` | K cho P/R/F1/NDCG, mặc định 10 |
| | `--ranking-depth` | độ sâu ranking cho AP; mặc định = số tài liệu corpus |
| | `--allow-truncated-depth` | cho phép AP trên ranking cắt cụt, kết quả ghi nhãn `AP@depth` |
| | `--mode {auto,official,development}` | `auto` dùng kết luận của cổng chặn |
| | `--allow-development` | cho phép ghi ra thư mục không chính thức |
| | `--corpus-manifest`, `--corpus` | nguồn SHA256 corpus |
| | `--no-fallback-measurement` | bỏ lần đo fallback của C4 |
| `benchmarks.benchmark_latency` | `--repeat`, `--warmup`, `--top-k` | tham số đo |
| | `--configurations` | danh sách cấu hình cần đo |
| `scripts.plot_results.py` | `--ablation`, `--latency`, `--tdv`, `--out` | nguồn dữ liệu và thư mục ảnh |
| | `--only` | chỉ vẽ một nhóm biểu đồ |

Mã thoát của `evaluator.evaluate`: `0` đã đo, `2` bị từ chối chính thức, `3` sai
thư mục đích, `5` chỉ số chưa đo (qrels chưa đủ nhãn), `6` độ sâu ranking không đủ.

---

## 4. Định nghĩa dung lượng chỉ mục

`measure_index_bytes()` tách ba số vì chúng trả lời ba câu hỏi khác nhau:

| Tên | Ý nghĩa |
| --- | --- |
| `serving_bytes` | mọi tệp trong thư mục chỉ mục **trừ** `tdv_*.json`; đây là số dùng cho mục tiêu giảm kích thước của *một chỉ mục* |
| `tdv_analysis_bytes` | riêng `tdv_scores.json`, `tdv_stopwords.json`, `tdv_metadata.json`, `tdv_df.json` — chỉ có ở C4, phục vụ phân tích, **không cần khi phục vụ truy vấn** |
| `total_bytes` | toàn bộ thư mục, dùng để đối chiếu với `build_summary.json` |

**Gói triển khai.** C4 bật fallback thì hệ thống phải giữ **cả** chỉ mục C4 và một
chỉ mục C3 riêng. Vì vậy báo cáo tách hai mục tiêu:

* giảm kích thước của *chỉ mục C4 thuần*;
* dung lượng của *gói triển khai C4 + C3*.

Không được dùng số của chỉ mục thuần để tuyên bố giảm 30% cho cả hệ thống.

---

## 5. Cấu hình đo

* Máy: ghi trong `results/latency/latency.json` → `environment`
  (Python, nền tảng, máy, số CPU, độ phân giải bộ đếm).
* Bộ query: 20 truy vấn trong `data/queries.json` (SHA256 `a46100f2…`), `--repeat 100`
  mỗi truy vấn (2.000 phép đo mỗi cấu hình), `--warmup 5` lượt chạy toàn bộ bộ query
  trước khi đo.
* `top_k = 10` cho phép đo độ trễ.
* C4 thuần: `load_engine()` không bật fallback. Fallback: `SearchService(..., enable_fallback=True)`,
  đo thành mục riêng có nhãn `C4_fallback`.
* Bộ đếm: `time.perf_counter_ns`.

---

## 6. Số đo thật hiện có

> **Trạng thái: chưa đo được chỉ số chất lượng.**
> `data/ground_truth/qrels.json` đang ở trạng thái `pending_human_annotation`
> (704 cặp query/tài liệu, 0 confirmed), nên P@K, R@K, F1@K, MAP và NDCG@K
> **chưa được tính**. Không có bảng chỉ số nào trong báo cáo. Độ trễ và dung lượng
> không cần nhãn nên đã đo được, nhưng mang nhãn *tạm thời* vì tokenizer còn là
> baseline phát triển của M1.

### 6.1 Độ trễ và dung lượng

Nguồn: `results/latency/latency.json`, `results/latency/latency.csv`.

| Cấu hình | mean (ms) | P95 (ms) | serving (byte) | tdv\_analysis (byte) | total (byte) | Giảm so với C1 | Đạt mục tiêu 30% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C1 | 1,002 | 1,880 | 10.127.659 | 0 | 10.127.659 | — | baseline |
| C2 | 0,563 | 1,135 | 7.369.530 | 0 | 7.369.530 | 27,23% | **không đạt** |
| C3 | 0,907 | 1,746 | 9.716.238 | 0 | 9.716.238 | 4,06% | **không đạt** |
| C4 (thuần) | 0,238 | 0,650 | 5.221.953 | 935.717 | 6.157.670 | 48,44% | **đạt** |
| C4 + fallback C3 (gói) | 0,244 | 0,620 | 14.938.191 | 935.717 | 15.873.908 | $-47{,}50\%$ | **không đạt** |

Môi trường đo: Python 3.13.5, Windows 11 (10.0.26300), 16 luồng CPU, bộ đếm
`time.perf_counter_ns`. SHA256 corpus
`f42acebb1ac34084e859f378243145a3f8e981869350edb41a3dc27e86320fa7`, SHA256 tập
truy vấn `a46100f23d937133a82d709b4b45ded5dcc0125520fe59ead647644e58be692e`; hash
trong qrels khớp hash thực tế của `queries.json`.

Đối chiếu mục tiêu:

| Mục tiêu | Ngưỡng | Kết quả | Trạng thái |
| --- | ---: | ---: | --- |
| mean latency | $<30$ ms | tốt nhất 0,238 ms (C4 thuần) | **đạt** ở cả bốn cấu hình |
| P95 latency | $<50$ ms | tốt nhất 0,620 ms | **đạt** ở cả bốn cấu hình |
| giảm 30% chỉ mục (C4 thuần) | $\ge 30\%$ | 48,44% | **đạt** |
| giảm 30% chỉ mục (gói C4+C3) | $\ge 30\%$ | $-47{,}50\%$ | **không đạt** |
| giảm 30% chỉ mục (C2) | $\ge 30\%$ | 27,23% | **không đạt** |
| giảm 30% chỉ mục (C3) | $\ge 30\%$ | 4,06% | **không đạt** |

Đọc đúng các con số này như sau:

1. Mục tiêu độ trễ đạt ở **mọi** cấu hình, nên nó không phân biệt được Cấu hình nào
   tốt hơn; nó chỉ chứng minh thiết kế chỉ mục đảo đủ nhanh ở quy mô 2.500 tài liệu.
2. Chỉ **chỉ mục C4 thuần** đạt mục tiêu giảm 30%.
3. **Gói triển khai thật của C4 không đạt**: bật fallback thì phải giữ thêm C3, tổng
   dung lượng *tăng* 47,5% so với C1. Không được trích số 48,44% để nói rằng "hệ
   thống giảm 30% kích thước".
4. C4 thuần có 100/2.000 phép đo trả về ranking rỗng (5%): đây là hậu quả thật của
   việc lọc bằng TDV, không phải lỗi. Khi bật fallback, số phép đo rỗng về 0 và
   fallback được dùng 100/2.000 lần.
5. Chưa có kết luận nào về chất lượng truy xuất. C2, C3, C4 chưa được so sánh về
   MAP/NDCG; **không** được nói "C4 tốt nhất".

### 6.2 Chỉ số chất lượng

Chưa đo. Xem `results/ablation/` sau khi qrels được confirmed và chạy lại
`python -m evaluator.evaluate`.

---

## 7. Biểu đồ

`scripts/plot_results.py` chỉ vẽ khi đã có file kết quả; thiếu dữ liệu thì in lý
do bỏ qua và thoát với mã lỗi, không bao giờ vẽ hình minh hoạ bằng số ảo.

| Tệp | Nguồn | Trạng thái |
| --- | --- | --- |
| `report/figures/fig_latency.png` | `results/latency/latency.json` | đã vẽ (tạm thời) |
| `report/figures/fig_index_size.png` | `results/latency/latency.json` | đã vẽ |
| `report/figures/fig_tdv_distribution.png` | `data/indexes/C4/tdv_*.json` | đã vẽ |
| `report/figures/fig_quality_metrics.png` | `results/ablation/summary.csv` | **chưa vẽ**: thiếu `run.json` |
| `report/figures/fig_map_by_category.png` | `results/ablation/summary_by_category.csv` | **chưa vẽ**: thiếu `run.json` |

---

## 8. Kiểm thử

```powershell
python -m pytest tests/test_tdv.py tests/test_evaluator.py -q
```

* `tests/test_tdv.py`: 47 test. Trong đó có fixture số học viết tay, so khớp với
  `reference_scores` trên nhiều fixture ngẫu nhiên, chứng minh cách làm sai
  (không tái chuẩn hóa) cho kết quả khác, one-shot khác loại nối tiếp, trường hợp
  biên ($N<2$, tài liệu rỗng, vectơ sau khi xoá bằng không) và ngưỡng dung sai.
* `tests/test_evaluator.py`: 51 test. Gồm công thức chỉ số tay, toàn bộ mã chặn
  đánh giá chính thức (kể cả khoá hash tập truy vấn), chặn rò fixture, và một bài
  kiểm thử end-to-end dựng thật bốn chỉ mục bằng `scripts.build_indexes.py` rồi
  chạy `evaluator.evaluate` và `benchmark_latency` trên đó.

---

## 9. Còn tồn đọng

1. **Chỉ số chất lượng chưa đo** — chờ qrels confirmed. Đây là phần thiếu quan
   trọng nhất của Chương 4.
2. **Tiền xử lý chưa official** — tokenizer còn là baseline âm thầm của M1, nên
   mọi số đo hiện là *tạm thời* dù corpus, chỉ mục và bộ query đã thật.
3. **C3/C4 dùng fixture tầng 1** — theo hợp đồng, C3/C4 dùng fixture tầng 1 phải
   được ghi là provisional, không coi là Maximum Matching chính thức.
4. **Quy mô 20 truy vấn** — khoảng 2 truy vấn mỗi lĩnh vực là giới hạn đã biết;
   không suy rộng kết luận ngoài tập truy vấn này.
