# M2: quyết định, hợp đồng và quyền sở hữu (2026-10-16)

## Giai đoạn tiếp nối: hoàn thiện M1 + M2

Người dùng đã giao tiếp tục triển khai cả M1 và M2. Tái sử dụng đúng ba builder OpenCode hiện có, không tạo thêm agent. Bảng dưới ghi quyền sở hữu thực tế sau khi đổi model và tích hợp; các tệp không nêu cụ thể do lead rà soát trước khi sửa.

| Vai trò | File/phạm vi sở hữu cuối |
| --- | --- |
| builder-data (w2:pA) | `engine/tokenizer/{__init__,preprocessor,trie,max_matching}.py`; `scripts/clean_dict.py`; tài nguyên từ điển, kiểm thử M1; `docs/{tokenizer,data_preparation,segmentation}.md` |
| builder-retrieval (w2:pB) | `engine/tokenizer/stopwords.py`, tài nguyên tầng 1 và kiểm thử tương ứng; `benchmarks/benchmark_lookup.py`, `tests/test_benchmark_lookup.py`, `docs/{trie_vs_set_benchmark,roy_analysis}.md`, `results/lookup/` |
| builder-experiment (w2:pC) | `app/{Home.py,pages/1_Tokenizer.py}`, `tests/test_app.py`, `docs/demo.md`; `report/{chapter1,chapter2_nlp,chapter3_nlp,chapter5,references}.tex` |
| Lead | `engine/preprocessing.py`; script gold/evaluator segmentation; build index/search/pool/annotation, cấu hình chung, README; `report/{main,chapter3_ir,chapter4,experiment_tables}.tex`, `report/m2_report.md`; artefact/benchmark cuối, kiểm thử tích hợp và sửa lỗi giao diện/báo cáo |

Sau khi Space Bunny Free hết hạn, các việc còn lại được giao tiếp trong đúng ba phiên cũ. Lead tiếp nhận `scripts/prepare_segmentation.py`, `scripts/annotate_segmentation.py`, `evaluator/segmentation_eval.py`, `tests/test_segmentation.py`, `scripts/render_report.py`, `scripts/verify_release.py`, `tests/test_app_integration.py` và các lần build/benchmark cuối. Không hoàn tác thay đổi đã giao ở lượt trước chỉ vì phạm vi của tác vụ bổ sung hẹp hơn.

### API M1 chốt trước khi viết

Ghi chú vận hành: ba phiên OpenCode đã gặp hạn mức Space Bunny Free. Lead đổi ngay trong đúng ba phiên cũ sang DeepSeek V4.1 Flash Free, xác nhận có phản hồi và tiếp tục giao việc; không tạo agent thứ tư hay đăng ký trả phí. Gemini 2.5 Flash đã thử nhưng khóa API Google hiện có bị từ chối, nên không dùng để triển khai.

`engine.tokenizer.preprocessor.normalize(text: str) -> str`: NFC, lowercase, bỏ URL/email/HTML/chữ số và ký tự không phải chữ; whitespace ổn định. Bộ tách từ phải coi dấu câu/ngắt câu là ranh giới, không ghép từ xuyên ranh giới dấu câu dù normalize hiển thị gộp khoảng trắng.

`engine.tokenizer.max_matching.MaximumMatching(words=None, max_syllables=None)` nạp từ điển local mặc định. Phương thức `tokenize(text, method="bimm") -> list[str]`; module xuất `tokenize`, `tokenize_fmm`, `tokenize_bmm`, `tokenize_bimm`, `trace(text, method="bimm") -> list[dict]`. `tokenize` mặc định BiMM, **chưa lọc tầng 1**, từ ghép nối `_`. `trace` mỗi dòng có `step`, `direction` (fmm/bmm), `start`, `end` (chỉ số âm tiết chuẩn hóa, end exclusive), `candidate`, `size`, `matched`, `action` (MATCHED/SHRINK/SINGLE); có thể thêm trường. BiMM chọn ít token, rồi ít token đơn, rồi ít OOV, hòa chọn BMM. Ví dụ sai/khác trong tài liệu không được ép thuật toán cho khớp một đáp án không đúng quy tắc; phải giải thích bằng fixture từ điển rõ ràng.

`engine.tokenizer.stopwords.filter_tier1(tokens: list[str]) -> list[str]`, `Tier1StopwordFilter(...).filter_tokens(tokens)`; module này xuất `tokenize_and_filter_t1(text)` bằng import lazy tokenizer. Tất cả phủ định/logic được bảo vệ, gồm không/chưa/chẳng/vì/do_đó; không tự bù danh sách cho đạt số lượng bằng thực từ. Tài nguyên có nguồn/quy tắc biên soạn, số mục thực tế và SHA256. Từ điển không rõ giấy phép không được đóng gói lại; có thể thay nguồn được phép sử dụng, phải ghi rõ sai khác DoD cũ.

Gold segmentation: chuẩn bị 150 câu (15/lĩnh vực) từ corpus khóa; token gợi ý là **pending**, không phải gold do người xác nhận. Công cụ nhập nhãn phải kiểm tra nội dung, hash, word spans và assessor; chỉ đánh giá chính thức khi toàn bộ nhãn đủ điều kiện. Hand-authored unit fixtures không thay thế gold corpus. Benchmark Trie/Set phải dùng cùng thuật toán/đầu vào, xác nhận output parity; đo thật và không mặc định Trie nhanh hơn.

Không thay corpus/query/doc_id đang khóa, không ghi data/raw. Giữ nguyên mọi nhãn đã xác nhận. Chỉ mục/pool mới dùng M1 phải có provenance và không được gọi kết quả baseline cũ là kết quả M1. Báo cáo chất lượng IR vẫn chờ qrels do con người xác nhận; triển khai M1 không tự mở khóa relevance pending.

Quyết định hiện tại của người dùng thay thế kế hoạch 5 lĩnh vực cũ: 10 thư mục trực tiếp trong `data/raw`, 250 bài hợp lệ mỗi lĩnh vực, tổng 2.500. Raw chỉ đọc. Seed 2026, thứ tự đường dẫn tương đối ổn định. Không push/merge/công khai dữ liệu.

## Quyền sở hữu

| Vai trò | File được viết |
| --- | --- |
| Lead | docs/api_contracts.md, docs/integration_status.md, README.md, requirements*.txt, pyproject.toml, .gitignore, engine/__init__.py, engine/indexer/__init__.py, engine/preprocessing.py, scripts/build_indexes.py, scripts/search.py, tests/test_integration.py, cấu hình chung và tích hợp app/report hiện có |
| builder-data | scripts/build_corpus.py, scripts/build_pool.py, scripts/annotate.py, tests/test_corpus.py, tests/test_pool.py, data/corpus.jsonl, data/corpus_manifest.json, data/corpus_stats.json, data/queries.json, data/ground_truth/**, docs/data_preparation.md |
| builder-retrieval | engine/indexer/tfidf_engine.py, engine/indexer/inverted_index.py, engine/indexer/search_engine.py, tests/test_ir_math.py, tests/test_search_engine.py, docs/retrieval.md |
| builder-experiment | engine/indexer/tdv_engine.py, evaluator/**, benchmarks/**, scripts/plot_results.py, tests/test_tdv.py, tests/test_evaluator.py, report/chapter3_ir.tex, report/chapter4.tex, report/m2_report.md, docs/experiments.md, results/** |

Không sửa ngoài phạm vi nếu chưa trao đổi lead. Chia sẻ worktree hiện tại, không commit tự động. Mỗi builder là một hội thoại OpenCode riêng trong pane Herdr; không tạo subagent.

## Corpus và query

JSONL UTF-8, mỗi record: `doc_id: str`, `title: str`, `content: str`, `category: str` (nhãn hiển thị tiếng Việt), `metadata: dict`. Metadata gồm source_file tương đối với raw, source_dataset, source_encoding, content_md5, whitespace_unit_count, title_origin (`source_line` hoặc `inferred_excerpt`); không bịa URL. NFC + gộp whitespace, MD5 nội dung chuẩn hóa toàn tập; ngưỡng 100 = `len(content.split())`, không gọi đây là số từ đã tách tiếng Việt. ID ổn định theo hash nội dung; manifest khóa SHA256 corpus trước khi gán nhãn. Không ghi đè corpus khóa khác nội dung nếu không có cờ rõ ràng.

`data/queries.json`: object gồm schema_version, corpus_sha256, queries (20 object: query_id, text, category, types, evidence_doc_ids). Khoảng 2 query/lĩnh vực; đủ short/long/negation/technical trên toàn tập; evidence là hỗ trợ soạn query, không phải qrels.

`data/ground_truth/qrels.json`: object gồm schema_version, corpus_sha256, status (`pending_human_annotation` ban đầu), judgments (query_id, doc_id, relevance null/0/1/2, status pending/confirmed, assessor). Chỉ nhãn confirmed do người gán mới hợp lệ cho đánh giá; không lấy category hoặc evidence thay relevance. Pool union Top-20 từ TF-IDF, binary cosine và BM25; baseline cùng corpus/tokenizer, mô tả chính xác. Pool lưu provenance/rank và nội dung; mẫu M1 20% seed 2026. Công cụ gán nhãn cần dễ dùng offline, không tạo nhãn confirmed tự động.

## API retrieval (công khai)

`SearchEngine(tokenizer=None, stopwords=None, metadata=None)`. tokenizer callable `str -> list[str]`; khi None chỉ baseline tách whitespace. `fit(documents, tokenized_docs=None) -> self` với tokens aligned theo corpus; index toàn bộ `content` (không cộng lại title đã nằm trong content). `search(query, top_k=10) -> list[dict]`; query là str hoặc list[str]. Kết quả: doc_id, title, category, score, snippet; hòa điểm sort doc_id tăng; chỉ kết quả điểm dương. `explain(query, doc_id) -> dict` có score, contributions list (term, query_tf, doc_tf, idf, query_weight, doc_weight, contribution); tổng contribution = cosine. `save(directory)` và `SearchEngine.load(directory, tokenizer=None)` lưu/nạp toàn bộ state và metadata; không silently đổi tokenizer. `.metadata` là dict dùng lưu provenance. Có `.tfidf` để TDV truy cập matrix/vocabulary nếu cần, mô tả chi tiết trong docs/retrieval.md.

Sublinear TF `1+ln(f)`, IDF `ln((N+1)/(df+1))+1`, corpus IDF dùng cho query; chuẩn L2; sparse CSR và posting list. Query rỗng/stopword/OOV/zero trả []. Stopwords dùng đồng nhất corpus/query. Kiểm chứng sklearn fixture gồm token 1 ký tự, lặp term, zero vector, tie, save/load.

## API TDV và cấu hình

`TDVEngine().fit(tokenized_docs) -> self` cung cấp `.scores_` dict term->float, `.stopwords_` set, `.metadata_` dict. `reference_scores(tokenized_docs)` tính trực tiếp để test. TDV sơ bộ dùng sublinear TF, smoothed corpus IDF và L2 trước lọc TDV. Dùng mật độ pairwise cosine (không bao gồm self) chính xác; `TDV(t)=D_without_t-D_base`. Sau xóa chiều phải chuẩn hóa lại từng document và cập nhật centroid/tổng vector. Zero vector cosine=0, mẫu số cố định N(N-1); N<2 mật độ=0. Giữ IDF sơ bộ khi loại một chiều để đo TDV; sau chọn stoplist build lại TF-IDF chính thức. Lọc score < -1e-12 (dung sai số học, phải ghi rõ); không ép đường cong U hoặc hiệu quả tốt nhất. Tính trên toàn bộ corpus 10 lĩnh vực, one-shot (không loại nối tiếp).

C1: không stopword. C2: stoplist tĩnh độc lập, có nguồn và số mục thực tế. C3: tầng 1 từ M1. C4: tầng 1 + TDV. Mỗi thư mục `data/indexes/Cn` có metadata corpus SHA256/tokenizer/configuration/stoplist. Hiện repo chưa có M1: lead viết adapter `engine.preprocessing` với baseline được đánh dấu development_only; C3/C4 dùng fixture tầng 1 phải ghi provisional, không coi là Maximum Matching chính thức. Cho phép `module:function` M1 ngoài qua CLI và fingerprint đầu vào.

Fallback C4 khi không còn token thuộc vocabulary: truy xuất bằng chỉ mục C3 riêng và ghi rõ cấu hình fallback. Không khẳng định giữ term đã xóa khỏi index sẽ khôi phục truy xuất. Cần đo C4 thuần và fallback riêng, metadata chính sách.

## Đánh giá và thực nghiệm

K mặc định 10 và có thể cấu hình; P@K mẫu số K, Recall mẫu số tổng relevant đã xác nhận trong pooled qrels, relevance>0 là relevant; F1 harmonic; AP theo toàn bộ ranking có sẵn / tổng relevant, MAP macro trên toàn bộ query (zero relevant =0, ghi số query này); NDCG gain 2^rel-1/log2(rank+1), IDCG dựa qrels. Không gọi AP@K là MAP toàn ranking. Từ chối đánh giá chính thức nếu chưa đủ 20 query, corpus/query hash mismatch, nhãn pending hoặc metadata preprocessing chưa official. Có thể kiểm thử bằng fixture nhưng không ghi kết quả fixture vào kết quả chính.

Benchmark có thể chạy khi qrels pending, ghi development tokenizer: warm-up, mean/P95, số repeat, query, kích thước bytes index trên đĩa và môi trường. Báo cáo đủ bảng query và 10 lĩnh vực khi có qrels; khoảng 2 query/lĩnh vực là giới hạn. Chỉ dùng số đo thật; mục tiêu giảm index 30% và mean<30ms phải báo đạt/không đạt theo số đo.
