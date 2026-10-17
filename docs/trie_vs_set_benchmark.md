# Benchmark: Trie Maximum Matching vs. tra cứu bằng `set`

Tài liệu này trả lời một câu hỏi hẹp: **với đúng cùng thuật toán, cùng từ điển và
cùng đầu vào, cấu trúc Trie có nhanh hơn tra cứu bằng hash set hay không?**

Kết luận ngắn: **không, với bản cài đặt Trie hiện có của dự án.** Trên cả mẫu phân
đoạn cố định và một lát cắt corpus cố định, bản `set` nhanh hơn khoảng **2,0–2,5
lần**. Đây là số đo thật trên máy chạy, không phải suy đoán; lý do và giới hạn được
nêu ở [§6](#6-vì-sao-set-nhanh-hơn-ở-bản-cài-đặt-này).

Công cụ: `benchmarks/benchmark_lookup.py` (builder-retrieval sở hữu). Kết quả thô:
`results/lookup/trie_vs_set.json` và `results/lookup/trie_vs_set.csv`.

## 1. Hai bên được so sánh

| Bên | Cài đặt | Cấu trúc tra cứu |
| --- | --- | --- |
| `trie` | `engine.tokenizer.max_matching.MaximumMatching` (bản đang dùng) | `Trie` âm tiết, tra cứu bằng `Trie.search` |
| `set` | `benchmarks.benchmark_lookup.SetMaximumMatching` (viết độc lập) | `frozenset` các tuple âm tiết |

Hai bên dùng **cùng** `engine.tokenizer.preprocessor.normalize_segments` (biên dấu
câu), **cùng** thứ tự duyệt (FMM/BMM/BiMM), **cùng** quy tắc hòa của BiMM, **cùng**
`max_syllables` và **cùng** danh sách mục từ điển. `SetMaximumMatching` không gọi
vào `engine.tokenizer.trie` (có test riêng chứng minh điều này).

## 2. Oracle: parity là cổng cứng

Trước khi ghi bất kỳ số đo nào, `parity_report()` chạy cả hai bên trên mọi câu, mọi
phương pháp, và so từng token:

* **750 phép đối soát** (250 câu × 3 phương pháp), **0 mismatch**.
* Nếu lệch dù một token, `run()` ném `LookupBenchmarkError` và **không ghi kết quả**
  (test `test_run_refuses_and_writes_nothing_on_parity_failure`).
* Phép thử oracle trên fixture nhỏ nằm ở
  `tests/test_benchmark_lookup.py::test_set_matcher_matches_trie_for_all_methods`.

Chỉ khi hai cài đặt cho ra output giống hệt nhau thì việc so thời gian mới có nghĩa.

## 3. Đầu vào cố định

| Workload | Nguồn | Kích thước | SHA256 (rút gọn) |
| --- | --- | --- | --- |
| `sample` | `data/gold/segmentation_sample.jsonl` (150 câu, 15/lĩnh vực) | 150 văn bản, 18.096 ký tự | `37112c076cc9…` |
| `corpus` | `data/corpus.jsonl`, 100 tài liệu đầu theo thứ tự file | 100 văn bản, 180.544 ký tự | `f42acebb1ac3…` |

`load_sample_texts` bắt buộc manifest mẫu có `sample_sha256`; `load_corpus_texts`
bắt buộc manifest corpus có `corpus_sha256`/`sha256`. Cả hai đều so SHA256 thật;
thiếu manifest, thiếu hash hoặc hash sai đều bị từ chối đo (xem
`test_load_corpus_rejects_*` trong bộ test). Từ điển:
`resources/dict.txt`, 69.858 mục, SHA256 `175e8b1bdd77…`, `max_syllables = 17`.

> **Nhãn vàng vẫn pending.** Benchmark chỉ đọc `sentence_text` làm đầu vào; nó
> không đọc, không ghi, không suy diễn nhãn, và không sửa `data/gold`. Có test
> `test_benchmark_does_not_mutate_gold_labels` kiểm SHA256 của cả file mẫu lẫn file
> nhãn trước/sau. Chất lượng phân đoạn chính thức vẫn phải chờ người gán nhãn.

## 4. Phương pháp đo

* **Xen kẽ trên từng câu:** mỗi câu được đo bên này rồi bên kia trong cùng vòng lặp,
  và **đảo thứ tự** theo từng vòng (vòng chẵn Trie trước, vòng lẻ Set trước) để giảm
  thiên lệch cache/thứ tự.
* **Warmup** trước khi đo; mỗi cấu hình đo `repeat = 10` vòng trên toàn bộ workload.
* Đồng hồ `time.perf_counter_ns()`; thống kê `mean`, `median`, `p95`, `p99`, `min`,
  `max`, `stdev` (p95 dùng `evaluator.metrics.percentile`).
* Số mẫu: `sample` = 150×10 = 1.500 đo/bên/phương pháp; `corpus` = 100×10 = 1.000
  đo/bên/phương pháp.

**Điều kiện chạy** (ghi trong payload): Python 3.13.5 (CPython), Windows 11
`10.0.26300`, AMD64, 16 CPU logic. Thời gian phụ thuộc máy; tỉ lệ giữa hai bên là
phần ổn định hơn.

## 5. Kết quả đo thật

### 5.1 Mẫu phân đoạn (150 câu ngắn, đơn vị: ms/văn bản)

| Phương pháp | Bên | mean | median | p95 | p99 | min | max | texts/giây |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FMM | trie | 0,2328 | 0,1912 | 0,5260 | 0,8826 | 0,0313 | 1,5774 | 4.296 |
| FMM | **set** | **0,1147** | 0,0950 | 0,2459 | 0,3644 | 0,0256 | 1,2784 | **8.720** |
| BMM | trie | 0,2276 | 0,1938 | 0,4987 | 0,7927 | 0,0284 | 1,6360 | 4.394 |
| BMM | **set** | **0,1120** | 0,0956 | 0,2431 | 0,3590 | 0,0225 | 1,0497 | **8.931** |
| BiMM | trie | 0,4557 | 0,3826 | 1,0436 | 1,6005 | 0,0571 | 4,1615 | 2.195 |
| BiMM | **set** | **0,1958** | 0,1674 | 0,4109 | 0,6813 | 0,0338 | 1,2109 | **5.107** |

Tỉ lệ `set/trie` theo mean: FMM **0,493**, BMM **0,492**, BiMM **0,430**.

### 5.2 Lát cắt corpus (100 tài liệu, đơn vị: ms/văn bản)

| Phương pháp | Bên | mean | median | p95 | p99 | min | max | texts/giây |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FMM | trie | 3,0767 | 1,7377 | 10,4293 | 13,1926 | 0,4116 | 26,3385 | 325,0 |
| FMM | **set** | **1,4625** | 0,8131 | 4,9081 | 6,4639 | 0,2118 | 13,2632 | **683,8** |
| BMM | trie | 3,1993 | 1,7948 | 10,5230 | 13,8441 | 0,4250 | 23,1769 | 312,6 |
| BMM | **set** | **1,4850** | 0,8427 | 4,7933 | 6,5292 | 0,2218 | 13,8060 | **673,4** |
| BiMM | trie | 6,5355 | 3,5796 | 22,4668 | 27,8042 | 0,7972 | 79,1085 | 153,0 |
| BiMM | **set** | **2,6611** | 1,4270 | 8,9432 | 12,0011 | 0,3421 | 36,2576 | **375,8** |

Tỉ lệ `set/trie` theo mean: FMM **0,475**, BMM **0,464**, BiMM **0,407**.

Tổng thời gian tokenize toàn bộ 100 tài liệu: FMM 3.077 ms (trie) so với 1.463 ms
(set); BMM 3.199 ms so với 1.485 ms; BiMM 6.536 ms so với 2.661 ms.

**Đọc p95 cho đúng:** trên corpus, `p95` lớn hơn `mean` nhiều lần vì độ dài tài
liệu rất lệch (từ vài trăm đến vài nghìn âm tiết). p95 phản ánh **phân bố độ dài**,
không phải nhiễu đồng hồ; vì vậy nên so `mean`, `median` và throughput giữa hai bên
hơn là so p95 tuyệt đối.

### 5.3 Bộ nhớ dựng cấu trúc (heap Python, `tracemalloc`)

| Cấu trúc | Đỉnh heap khi dựng | Ghi chú |
| --- | ---: | --- |
| Trie | 20.985.701 B (≈20,0 MiB) | 69.858 mục, 7.893 con ở gốc, `max_syllables = 17` |
| Set tuple | 15.368.109 B (≈14,7 MiB) | 69.858 mục |

Đây là **đỉnh heap Python khi dựng** (nhiều node/dict nhỏ cho Trie), **không phải
RSS của cả tiến trình** và không tính bộ nhớ của chính từ điển nguồn. Đừng dùng số
này để tuyên bố tiết kiệm bộ nhớ khi triển khai.

## 6. Vì sao `set` nhanh hơn ở bản cài đặt này

Đây là điểm dễ bị hiểu sai nhất, nên nói thẳng: **bản Trie hiện tại không khai thác
lợi thế lý thuyết của Trie.**

`MaximumMatching._fmm/_bmm` duyệt độ dài ứng viên từ dài xuống ngắn và gọi
`Trie.search(candidate)` cho **từng** độ dài. Mỗi lần `search` lại **đi từ gốc**
qua từng âm tiết, rồi còn tạo chuỗi `" ".join(...)`. Với một cửa sổ không khớp, chi
phí tiệm cận bậc hai theo độ dài cửa sổ và có hằng số Python cao (mỗi âm tiết là
một `dict.get`, mỗi ứng viên là một phép nối chuỗi).

Bản `SetMaximumMatching` cũng thử từng độ dài, nhưng mỗi lần chỉ là **một phép băm
tuple** do C thực hiện (`tuple(syllables[start:end]) in entries`), không nối chuỗi,
không đi lại tiền tố. Hằng số nhỏ hơn hẳn, nên thắng trên cả văn bản ngắn lẫn dài.

Lợi thế *thật* của Trie là **một lượt đi duy nhất** cho mỗi vị trí
(`Trie.longest_match`: đi tới khi hết cạnh, nhớ lần cuối gặp `is_end_of_word`), tức
chi phí bậc một theo độ dài cửa sổ thay vì bậc hai. **Bản `_fmm/_bmm` đang dùng
không gọi `longest_match`**, nên lợi thế đó chưa xuất hiện trong số đo này. Kết quả
ở đây *không* phủ định Trie nói chung; nó chỉ nói về **bản Trie đang chạy**.

Hệ quả thực dụng: nếu muốn Trie thắng, phải viết lại vòng lặp theo `longest_match`
(đi một lượt, nhớ terminal cuối) rồi đo lại — việc đó là thay đổi thuật toán trong
`engine.tokenizer`, thuộc quyền builder-data, **không** nằm trong phạm vi tài liệu
này. Cho tới lúc đó, kết luận trung thực là: *set lookup nhanh hơn, và lý do nằm ở
cách bản Trie hiện tại duyệt ứng viên.*

## 7. Giới hạn và điều không được suy ra

* **Không suy ra chất lượng phân đoạn.** Benchmark chỉ đo thời gian và bộ nhớ; độ
  chính xác (word-span P/R/F1) do `evaluator/segmentation_eval.py` quyết định và cần
  nhãn người, hiện vẫn **pending**.
* **Không suy ra thứ hạng trên mọi máy.** Số tuyệt đối phụ thuộc CPU/OS; tỉ lệ
  `set/trie` (~0,4–0,5) mới là phần nên trích dẫn, và vẫn nên đo lại trên máy đích.
* **Không phải benchmark truy xuất.** Đây là M1 (tra cứu từ điển khi tách từ), khác
  với `benchmarks/benchmark_latency.py` đo M2 (truy vấn chỉ mục).
* **`tracemalloc` không phải RSS.** Số bộ nhớ chỉ là heap Python.

## 8. Tái lập

```powershell
# kiểm thử oracle + cổng từ chối (nhanh)
python -X utf8 -m pytest tests/test_benchmark_lookup.py -q

# đo thật (mất khoảng một phút)
python -X utf8 -m benchmarks.benchmark_lookup --repeat 10 --warmup 2 --corpus-docs 100
```

`--methods fmm bmm bimm` (mặc định), `--sample`, `--corpus`, `--dictionary`, `--out`
đều chỉnh được. Payload JSON ghi rõ môi trường, tham số, hash đầu vào, parity,
bảng thời gian và bộ nhớ để tái lập và đối chiếu.
