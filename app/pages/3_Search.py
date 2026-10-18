from pathlib import Path
import json
import hashlib
import sys
import time

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.search import SearchService

st.set_page_config(page_title="Cổng tìm kiếm", layout="wide")
st.title("3. Tìm kiếm văn bản & giải thích điểm Cosine")


@st.cache_resource
def get_service(directory, signature, fallback):
    return SearchService(directory, enable_fallback=fallback)


@st.cache_data
def get_corpus(signature):
    payload = (ROOT / "data/corpus.jsonl").read_bytes()
    return (hashlib.sha256(payload).hexdigest(),
            {d["doc_id"]: d for d in (json.loads(line) for line in payload.decode("utf-8").splitlines()
                                      if line.strip())})


indexes = ROOT / "data/indexes"
available = [name for name in ("C1", "C2", "C3", "C4")
             if (indexes / name / "build_metadata.json").is_file()]
if not available:
    st.info("Chưa có chỉ mục. Chạy: python -m scripts.build_indexes")
    st.stop()

configuration = st.selectbox("Cấu hình", available, index=len(available) - 1)
fallback = st.checkbox("Dùng C3 khi truy vấn C4 không còn term trong chỉ mục", value=True)
directory = indexes / configuration
signature = (directory / "build_metadata.json").stat().st_mtime_ns
try:
    service = get_service(str(directory), signature, fallback)
except (ValueError, FileNotFoundError, ImportError) as error:
    st.error(str(error))
    st.stop()

metadata = service.engine.metadata
if metadata["preprocessing_status"] != "official":
    st.warning("Bản phát triển dùng tokenizer âm tiết và fixture tầng 1. Chưa phải Maximum Matching của M1.")
st.caption(f"Corpus: {metadata['n_documents']:,} bài · {configuration} · "
           f"SHA256: {metadata['corpus_sha256'][:16]}…")
with st.form("search"):
    query = st.text_input("Truy vấn tiếng Việt", value="phòng chống dịch cúm gia cầm")
    top_k = st.slider("Số kết quả", min_value=1, max_value=20, value=10)
    submitted = st.form_submit_button("Tìm kiếm")

if submitted:
    started = time.perf_counter()
    response = service.search(query, top_k)
    elapsed = (time.perf_counter() - started) * 1000
    if response["fallback_used"]:
        st.info("Truy vấn được chạy bằng chỉ mục C3; điểm và giải thích bên dưới thuộc C3.")
    st.caption(f"Token trước lọc TDV: {service.engine.tokenizer(query)}")
    st.caption(f"Tìm kiếm: {elapsed:.3f} ms (một lần chạy giao diện, không phải benchmark). "
               "Cosine bag-of-words không tự suy luận phủ định; relevance cần nhãn người đọc.")
    documents = {}
    corpus_path = ROOT / "data/corpus.jsonl"
    if corpus_path.is_file():
        corpus_hash, corpus_documents = get_corpus(corpus_path.stat().st_mtime_ns)
        if corpus_hash == metadata["corpus_sha256"]:
            documents = corpus_documents
    if not response["results"]:
        st.info("Không có kết quả cho truy vấn này.")
    for rank, hit in enumerate(response["results"], 1):
        st.subheader(f"{rank}. {hit['title']}")
        st.caption(f"{hit['category']} · {hit['doc_id']} · Cosine {hit['score']:.6f}")
        st.write(hit["snippet"])
        if hit["doc_id"] in documents:
            with st.expander("Đọc toàn bộ bài"):
                document = documents[hit["doc_id"]]
                st.text(document["content"])
                source = document.get("metadata", {})
                st.caption(f"Nguồn local: {source.get('source_file', '')} · "
                           f"Tiêu đề: {source.get('title_origin', '')}")
        with st.expander("Giải thích đóng góp từng term"):
            explanation = service.explain(query, hit["doc_id"],
                                           used_configuration=response["used_configuration"])
            st.dataframe(explanation["contributions"], width="stretch")
            total = sum(row["contribution"] for row in explanation["contributions"])
            st.caption(f"Tổng đóng góp = {total:.9f}; Cosine = {explanation['score']:.9f}")
