"""Trang chủ hệ thống: điều hướng, siêu dữ liệu thật và trạng thái đo lường.

Trang này chỉ hiển thị những gì đọc được từ artifact thật trên đĩa. Mỗi tệp
được dò trước khi hiện; nếu thiếu thì ghi rõ là thiếu chứ không dựng số giả.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
CORPUS_MANIFEST = ROOT / "data/corpus_manifest.json"
CORPUS_STATS = ROOT / "data/corpus_stats.json"
QUERIES = ROOT / "data/queries.json"
QRELS = ROOT / "data/ground_truth/mm/qrels.json"
INDEX_ROOT = ROOT / "data/indexes"
C4_DIR = INDEX_ROOT / "C4"
CONFIGURATIONS = ("C1", "C2", "C3", "C4")

TEAM = "Nhóm 7 – Team7"
COURSE = "Cơ sở dữ liệu Đa phương tiện"
SYSTEM_NAME = "Hệ thống tìm kiếm văn bản và phân tích hình học TDV Salton"


def read_json(path: Path) -> Optional[dict]:
    """Đọc JSON, trả None nếu tệp không tồn tại hoặc hỏng."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def corpus_facts() -> Dict[str, object]:
    """Số liệu corpus đọc từ manifest đã khóa."""
    manifest = read_json(CORPUS_MANIFEST)
    if manifest is None:
        return {"available": False}
    facts = {
        "available": True,
        "documents": manifest.get("corpus_documents"),
        "sha256": manifest.get("corpus_sha256"),
        "locked": bool(manifest.get("locked", False)),
        "categories": None,
    }
    counts = manifest.get("counts") or {}
    by_category = counts.get("by_category") if isinstance(counts, dict) else None
    if isinstance(by_category, dict) and by_category:
        facts["categories"] = len(by_category)
        facts["category_counts"] = by_category
    facts["eligible_documents"] = (counts.get("eligible_documents")
                                   if isinstance(counts, dict) else None)
    facts["rejected_short"] = (counts.get("rejected_below_min_whitespace_units")
                               if isinstance(counts, dict) else None)
    facts["duplicates_removed"] = (counts.get("duplicate_copies_removed")
                                   if isinstance(counts, dict) else None)
    stats = read_json(CORPUS_STATS)
    if isinstance(stats, dict):
        facts["stats_available"] = True
        facts["stats_keys"] = len(stats)
    return facts


def query_facts() -> Dict[str, object]:
    """Số liệu bộ truy vấn và qrels, kèm trạng thái gán nhãn."""
    queries = read_json(QUERIES)
    qrels = read_json(QRELS)
    facts: Dict[str, object] = {
        "queries_available": queries is not None,
        "n_queries": len(queries.get("queries", [])) if queries else 0,
        "queries_sha256": queries.get("queries_sha256") if queries else None,
        "qrels_available": qrels is not None,
        "qrels_status": qrels.get("status") if qrels else None,
    }
    if qrels:
        progress = qrels.get("annotation_progress")
        if isinstance(progress, dict):
            facts["n_judgments"] = progress.get("total")
            facts["n_confirmed"] = progress.get("confirmed")
    return facts


def index_facts() -> Dict[str, Dict[str, object]]:
    """Siêu dữ liệu từng cấu hình C1–C4, chỉ gồm trường có thật trong tệp."""
    facts: Dict[str, Dict[str, object]] = {}
    for configuration in CONFIGURATIONS:
        metadata = read_json(INDEX_ROOT / configuration / "build_metadata.json")
        if metadata is None:
            facts[configuration] = {"available": False}
            continue
        facts[configuration] = {
            "available": True,
            "documents": metadata.get("n_documents"),
            "stoplist_count": metadata.get("stoplist_count"),
            "preprocessing_status": metadata.get("preprocessing_status"),
            "corpus_sha256": metadata.get("corpus_sha256"),
            "created_utc": metadata.get("created_utc"),
        }
    return facts


def tdv_facts() -> Dict[str, object]:
    """Thống kê TDV của C4 nếu artifact đã được sinh."""
    scores = read_json(C4_DIR / "tdv_scores.json")
    metadata = read_json(C4_DIR / "tdv_metadata.json")
    build = read_json(C4_DIR / "build_metadata.json")
    if scores is None or metadata is None:
        return {"available": False}
    values = [float(value) for value in scores.values()]
    return {
        "available": True,
        "n_terms": len(values),
        "n_filtered": metadata.get("n_filtered_terms"),
        "base_density": metadata.get("base_density"),
        "minimum": metadata.get("density_min"),
        "maximum": metadata.get("density_max"),
        "tolerance": metadata.get("tolerance"),
        "seconds": (build or {}).get("tdv_seconds"),
    }


def official_gate() -> Dict[str, object]:
    """Kết luận về việc số đo chất lượng đã đủ điều kiện chính thức chưa.

    Dùng đúng bộ quy tắc của ``evaluator.qrels.validate_official_run`` để trang
    chủ không tự chế ra tiêu chí riêng.
    """
    queries = read_json(QUERIES)
    qrels = read_json(QRELS)
    build = read_json(C4_DIR / "build_metadata.json")
    if queries is None or qrels is None:
        return {"evaluated": False,
                "reason": "thiếu data/queries.json hoặc data/ground_truth/qrels.json"}
    try:
        from evaluator.qrels import Qrels, QuerySet, Judgment, validate_official_run
    except ImportError as error:
        return {"evaluated": False, "reason": f"không nạp được evaluator: {error}"}
    judgments = tuple(
        Judgment(str(entry["query_id"]), str(entry["doc_id"]),
                 None if entry.get("relevance") is None else int(entry["relevance"]),
                 str(entry.get("status", "pending")), entry.get("assessor"))
        for entry in qrels.get("judgments", []) or [])
    from evaluator.qrels import file_sha256

    query_set = QuerySet(
        schema_version=queries.get("schema_version"),
        corpus_sha256=queries.get("corpus_sha256"),
        queries=tuple(queries.get("queries", []) or []),
        source_path=str(QUERIES),
        file_sha256=file_sha256(QUERIES) if QUERIES.is_file() else None,
        queries_sha256=queries.get("queries_sha256"))
    qrels_object = Qrels(
        schema_version=qrels.get("schema_version"),
        corpus_sha256=qrels.get("corpus_sha256"),
        status=str(qrels.get("status", "unknown")),
        judgments=judgments,
        source_path=str(QRELS),
        queries_sha256=qrels.get("queries_sha256"))
    report = validate_official_run(qrels_object, query_set,
                                   corpus_sha256=(build or {}).get("corpus_sha256"),
                                   index_metadata=build)
    return {"evaluated": True, "official_allowed": report.official_allowed,
            "issues": [str(issue) for issue in report.issues]}


def render_corpus_section(facts: Dict[str, object]) -> None:
    st.subheader("Kho ngữ liệu")
    if not facts.get("available"):
        st.info("Chưa có `data/corpus_manifest.json`. Chạy `python scripts/build_corpus.py`.")
        return
    columns = st.columns(4)
    columns[0].metric("Tài liệu", f"{facts['documents']:,}" if facts.get("documents") else "—")
    columns[1].metric("Lĩnh vực", facts.get("categories") if facts.get("categories") else "—")
    columns[2].metric("Manifest", "đã khóa" if facts.get("locked") else "chưa khóa")
    digest = facts.get("sha256")
    st.caption(f"SHA256 corpus: `{digest[:32]}…`" if digest else "Thiếu SHA256 trong manifest.")
    if facts.get("category_counts"):
        counts = facts["category_counts"]
        st.caption("Phân bổ theo lĩnh vực: " +
                   " · ".join(f"{name} {value}" for name, value in sorted(counts.items())))
    cleanup = [label for label, key in (("bài dưới ngưỡng", "rejected_short"),
                                        ("bản trùng đã loại", "duplicates_removed"),
                                        ("ứng viên trước lấy mẫu", "eligible_documents"))
               if facts.get(key) is not None]
    if cleanup:
        st.caption("Làm sạch: " + " · ".join(cleanup) + " — " +
                   " · ".join(f"{label} {facts[key]:,}"
                              for label, key in (("bài dưới ngưỡng", "rejected_short"),
                                                 ("bản trùng đã loại", "duplicates_removed"),
                                                 ("ứng viên trước lấy mẫu", "eligible_documents"))
                              if facts.get(key) is not None))


def render_query_section(facts: Dict[str, object]) -> None:
    st.subheader("Bộ truy vấn và nhãn liên quan")
    columns = st.columns(3)
    columns[0].metric("Truy vấn", facts.get("n_queries") or "—")
    columns[1].metric("Cặp trong pool", facts.get("n_judgments") or "—")
    columns[2].metric("Đã xác nhận", facts.get("n_confirmed") if facts.get("n_confirmed")
                      is not None else "—")
    st.caption(f"Trạng thái qrels: `{facts.get('qrels_status')}`")
    digest = facts.get("queries_sha256")
    if digest:
        st.caption(f"SHA256 tập truy vấn: `{digest[:32]}…`")


def render_index_section(indexes: Dict[str, Dict[str, object]]) -> None:
    st.subheader("Bốn cấu hình chỉ mục")
    rows = []
    for configuration, fact in indexes.items():
        rows.append({
            "Cấu hình": configuration,
            "Tài liệu": fact.get("documents") if fact.get("available") else "chưa build",
            "Stoplist": fact.get("stoplist_count") if fact.get("available") else "—",
            "Tiền xử lý": fact.get("preprocessing_status") if fact.get("available") else "—",
        })
    st.dataframe(rows, width="stretch", hide_index=True)
    statuses = {fact.get("preprocessing_status") for fact in indexes.values()
                if fact.get("available")}
    if "official" not in statuses and any(fact.get("available") for fact in indexes.values()):
        st.warning("Tiền xử lý chưa ở trạng thái `official`: kết quả truy xuất là bản "
                   "phát triển, chưa phải Maximum Matching chính thức của M1.")


def render_tdv_section(facts: Dict[str, object]) -> None:
    st.subheader("Phân tích TDV (C4)")
    if not facts.get("available"):
        st.info("Chưa có bảng điểm TDV của C4. "
                "Chạy `python scripts/build_indexes.py` để dựng chỉ mục.")
        return
    columns = st.columns(5)
    columns[0].metric("Từ trong từ vựng", f"{facts['n_terms']:,}")
    columns[1].metric("Bị lọc", facts.get("n_filtered") if facts.get("n_filtered")
                      is not None else "—")
    columns[2].metric("Mật độ nền", f"{facts['base_density']:.6f}"
                      if facts.get("base_density") is not None else "—")
    columns[3].metric("TDV nhỏ nhất", f"{facts['minimum']:.3e}"
                      if facts.get("minimum") is not None else "—")
    columns[4].metric("TDV lớn nhất", f"{facts['maximum']:.3e}"
                      if facts.get("maximum") is not None else "—")
    st.caption(f"Ngưỡng lọc: điểm TDV < -{facts['tolerance']:g} (dung sai số học). "
               f"Thời gian chạy TDV: {facts['seconds']:.2f} s."
               if facts.get("seconds") else
               f"Ngưỡng lọc: điểm TDV < -{facts['tolerance']:g} (dung sai số học)."
               if facts.get("tolerance") is not None else "")


def render_gate_section(gate: Dict[str, object]) -> None:
    st.subheader("Trạng thái đánh giá chất lượng")
    if not gate.get("evaluated"):
        st.info(gate.get("reason", "Không đánh giá được điều kiện chính thức."))
        return
    if gate.get("official_allowed"):
        st.success("Đủ điều kiện chạy đánh giá chính thức: "
                   "`python -m evaluator.evaluate --k 10`.")
        return
    st.warning("Chưa đủ điều kiện để coi kết quả là chính thức. Bộ đánh giá sẽ từ chối "
               "ghi kết quả chính thức và không xuất chỉ số nếu qrels chưa có nhãn "
               "đã xác nhận.")
    with st.expander("Các điều kiện chưa thoả"):
        for issue in gate.get("issues", [])[:12]:
            st.write(f"• {issue}")
        remaining = len(gate.get("issues", [])) - 12
        if remaining > 0:
            st.caption(f"... và {remaining} điều kiện khác.")


def main() -> None:
    st.set_page_config(page_title="Đồ án CSDL ĐPT — Nhóm 7", page_icon="📚", layout="wide")
    st.title(SYSTEM_NAME)
    st.subheader(f"Đồ án môn học: {COURSE} · {TEAM}")

    st.markdown(
        """
Hệ thống truy xuất văn bản tiếng Việt gồm hai tầng xử lý: **tách từ theo nguyên lý
so khớp cực đại** rồi **lọc từ dừng phân tầng** — tầng 1 theo ngữ pháp, tầng 2 theo
hình học không gian vector. Tầng 2 dùng Giá trị phân biệt từ (TDV) của Salton để tự
động phát hiện từ dừng đặc thù của từng miền dữ liệu mà không cần đặt ngưỡng tần suất
thủ công.

**Phân công thực hiện**

| Vị trí | Phạm vi |
| --- | --- |
| M1 | Tiền xử lý chuỗi, từ điển và cây Trie, tách từ FMM/BMM/BiMM, lọc từ dừng tầng 1 |
| M2 | TDV, TF-IDF, chỉ mục đảo, truy xuất Cosine, bộ đánh giá, đo độ trễ và biểu đồ |
| M3 | Quản trị mã nguồn, hợp đồng API, tích hợp ứng dụng và đóng gói |
""")

    st.markdown("---")
    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            """
### Các trang trong ứng dụng

1. **Tách từ & Lọc tầng 1** — mô phỏng từng bước FMM, BMM, BiMM trên cây Trie, xem
   danh sách từ bị lọc và các từ phủ định được bảo vệ.
2. **Phân tích TDV** — phân phối điểm TDV theo tần suất tài liệu và bảng tra cứu
   từ bị loại.
3. **Tìm kiếm & Giải thích điểm** — truy vấn, Top-K và bảng phân rã đóng góp từng
   từ vào điểm Cosine.
""")
    with right:
        render_corpus_section(corpus_facts())

    render_query_section(query_facts())
    render_index_section(index_facts())
    render_tdv_section(tdv_facts())
    render_gate_section(official_gate())

    st.markdown("---")
    st.caption("Số liệu trên trang này đọc trực tiếp từ artifact trên đĩa. "
               "Trang không hiển thị số liệu suy đoán.")


main()
