"""Trang 1 — Mô phỏng thuật toán tách từ và lọc từ dừng tầng 1.

Trang này **chỉ** hiển thị kết quả do mã lõi của M1 trả về:

* ``engine.tokenizer.max_matching.trace(text, method)`` cho từng bước FMM/BMM/BiMM;
* ``engine.tokenizer.preprocessor.normalize(text)`` cho chuỗi đã chuẩn hóa;
* ``engine.tokenizer.stopwords.filter_tier1(tokens)`` cho lọc tầng 1 và guard.

Khi mã lõi chưa sẵn sàng, trang dừng lại với thông báo cài đặt cụ thể. Trang không
tự dựng kết quả tách từ, không tự đoán từ dừng và không hiển thị ví dụ giả.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DICT_PATH = ROOT / "resources/dict.txt"
GRAMMAR_STOPWORDS = ROOT / "resources/grammar_stopwords.txt"
GUARD_WHITELIST = ROOT / "resources/guard_whitelist.txt"

METHODS = ("fmm", "bmm", "bimm")
METHOD_LABELS = {"fmm": "FMM — so khớp tiến cực đại",
                 "bmm": "BMM — so khớp lùi cực đại",
                 "bimm": "BiMM — so khớp hai chiều"}
DEFAULT_TEXT = "trí tuệ nhân tạo trong giáo dục"
ACTIONS = {"MATCHED": "Khớp từ điển", "SHRINK": "Thu cửa sổ",
           "SINGLE": "Âm tiết đơn (OOV)"}


def load_core() -> Tuple[Optional[object], Optional[object], Optional[object], Optional[str]]:
    """Nạp mã lõi M1.

    Trả về ``(matcher, normalize, filter_tier1, error)``. Vế ``error`` mang hai
    nghĩa tuần tự: nếu ``matcher`` là ``None`` thì đó là lỗi chặn toàn trang; nếu
    ``matcher`` có mặt mà ``filter_tier1`` là ``None`` thì đó là lỗi chỉ ảnh
    hưởng tab lọc tầng 1.
    """
    try:
        from engine.tokenizer import max_matching
        from engine.tokenizer import preprocessor
    except ImportError as error:
        return None, None, None, f"Chưa có `engine.tokenizer` ({error})."
    filter_tier1 = None
    tier1_error = None
    try:
        from engine.tokenizer import stopwords as stopword_module

        filter_tier1 = stopword_module.filter_tier1
    except (ImportError, AttributeError) as error:
        tier1_error = f"Chưa có `engine.tokenizer.stopwords` ({error})."
    try:
        matcher = max_matching.MaximumMatching()
    except Exception as error:  # noqa: BLE001 - báo lỗi nạp từ điển cho người dùng
        return None, None, filter_tier1, f"Không nạp được từ điển: {error}"
    return matcher, preprocessor.normalize, filter_tier1, tier1_error


def build_matcher(max_syllables: int) -> Tuple[Optional[object], Optional[str]]:
    """Dựng matcher với Lmax do người dùng chọn; trả ``(matcher, lỗi)``.

    Không tự thay Lmax bằng giá trị mặc định khi dựng lỗi, vì như vậy con số hiển
    thị sẽ không còn khớp với kết quả trả về.
    """
    from engine.tokenizer import max_matching

    try:
        return max_matching.MaximumMatching(max_syllables=int(max_syllables)), None
    except TypeError:
        try:
            return max_matching.MaximumMatching(words=None,
                                                max_syllables=int(max_syllables)), None
        except TypeError as error:
            return None, f"Bản cài đặt không nhận tham số Lmax: {error}"
    except Exception as error:  # noqa: BLE001
        return None, f"Không dựng được matcher với Lmax={max_syllables}: {error}"


def accepts_method(matcher) -> bool:
    """Kiểm tra chữ ký ``tokenize`` có nhận tham số ``method`` hay không."""
    import inspect

    try:
        return "method" in inspect.signature(matcher.tokenize).parameters
    except (TypeError, ValueError):
        return False


@st.cache_resource
def cached_matcher():
    """Nạp từ điển một lần cho mỗi phiên làm việc."""
    return load_core()


def syllables_of(normalize, text: str) -> List[str]:
    """Tách âm tiết đã chuẩn hóa để hiển thị, không phải kết quả tách từ."""
    normalized = normalize(text)
    return [piece for piece in normalized.replace("\n", " ").split(" ") if piece]


def token_view(tokens: Sequence[str]) -> str:
    return " · ".join(tokens) if tokens else "(rỗng)"


def tier1_filter_instance():
    """Lấy bộ lọc tầng 1 từ mã lõi; trả ``None`` nếu chưa có."""
    try:
        from engine.tokenizer import stopwords as stopword_module
    except ImportError:
        return None
    getter = getattr(stopword_module, "get_default_filter", None)
    if callable(getter):
        try:
            return getter()
        except Exception:  # noqa: BLE001 - thử cách khác nếu getter lỗi
            pass
    try:
        return stopword_module.Tier1StopwordFilter()
    except Exception:  # noqa: BLE001
        return None


def guard_report(tokens: Sequence[str], filtered: Sequence[str],
                 tier1_filter=None) -> List[Dict[str, object]]:
    """Gắn nhãn từng token: giữ, bị lọc tầng 1, hay được guard bảo vệ.

    Trạng thái guard hỏi đúng bộ lọc tầng 1 (``is_protected``) thay vì tự suy đoán
    từ danh sách, vì quy tắc guard áp dụng cả cho từ ghép có thành phần được bảo vệ.
    """
    kept = set(filtered)
    guards = set(getattr(tier1_filter, "guards", ()) or ())
    stopwords = set(getattr(tier1_filter, "stopwords", ()) or ())
    is_protected = getattr(tier1_filter, "is_protected", None)
    rows: List[Dict[str, object]] = []
    for position, token in enumerate(tokens, start=1):
        if callable(is_protected):
            protected = bool(is_protected(token))
        else:
            protected = token in guards
        removed = token not in kept
        if protected and token in stopwords:
            reason = "Có trong hư từ tầng 1 nhưng được guard giữ lại"
        elif protected:
            reason = "Guard bảo vệ (giữ dù có thể là từ chức năng)"
        elif removed:
            reason = "Bị lọc tầng 1 (hư từ ngữ pháp)"
        else:
            reason = "Thực từ — giữ nguyên"
        rows.append({"#": position, "Token": token, "Kết quả": "loại" if removed else "giữ",
                     "Lý do": reason, "Guard": "có" if protected else "không"})
    return rows


def tier1_resource_summary(tier1_filter) -> List[Dict[str, object]]:
    """Số mục thực tế của ba danh mục, đọc từ bộ lọc tầng 1."""
    if tier1_filter is None:
        return []
    rows = []
    for label, attribute in (("Hư từ ngữ pháp tầng 1", "stopwords"),
                             ("Guard bảo vệ", "guards"),
                             ("Danh sách loại hiệu lực", "effective_stopwords")):
        values = getattr(tier1_filter, attribute, None)
        rows.append({"Danh mục": label,
                     "Số mục": len(values) if values is not None else "—",
                     "Nguồn": str(getattr(tier1_filter, "resource_dir", "—"))})
    return rows


def resource_status() -> List[Dict[str, object]]:
    rows = []
    for label, path in (("Từ điển", DICT_PATH),
                        ("Hư từ ngữ pháp (tầng 1)", GRAMMAR_STOPWORDS),
                        ("Guard bảo vệ", GUARD_WHITELIST)):
        exists = path.is_file()
        count = None
        if exists:
            try:
                count = sum(1 for line in path.read_text(encoding="utf-8-sig").splitlines()
                            if line.strip() and not line.lstrip().startswith("#"))
            except OSError:
                exists = False
        rows.append({"Tài nguyên": label, "Tệp": str(path.relative_to(ROOT)),
                     "Tồn tại": "có" if exists else "chưa có",
                     "Số mục": count if count is not None else "—"})
    return rows


def render_trace_tab(matcher, normalize) -> None:
    st.subheader("Mô phỏng tách từ từng bước")
    text = st.text_area("Văn bản đầu vào", value=DEFAULT_TEXT, height=90)
    method = st.radio("Thuật toán", METHODS, index=2, horizontal=True,
                      format_func=lambda value: METHOD_LABELS[value])
    max_syllables = st.number_input("Cửa sổ tối đa (Lmax)", min_value=1, max_value=64,
                                    value=matcher.max_syllables, step=1)
    run = st.button("Chạy thuật toán", type="primary")
    if not run:
        st.caption("Nhấn “Chạy thuật toán” để mô phỏng. Trang không tự tính trước kết quả.")
        return
    if not text.strip():
        st.warning("Chuỗi rỗng: không có âm tiết nào để tách.")
        return

    normalized = normalize(text)
    syllables = syllables_of(normalize, text)

    chosen, build_error = build_matcher(max_syllables)
    if chosen is None:
        st.error(build_error)
        st.info("Dừng lại thay vì dùng Lmax khác để tránh hiển thị kết quả không khớp "
                "với tham số đã chọn.")
        return
    tokens = chosen.tokenize(text, method=method) if accepts_method(chosen) \
        else chosen.tokenize(text)

    st.markdown(f"**Chuỗi sau chuẩn hóa NFC + chữ thường + làm sạch:** `{normalized}`")
    st.caption(f"{len(syllables)} âm tiết · Lmax = {max_syllables} · "
               f"{METHOD_LABELS[method]}")

    st.markdown("**Kết quả tách từ**")
    st.code(token_view(tokens), language=None)
    st.caption(f"{len(tokens)} token" + ("" if tokens else " — chuỗi không còn token nào"))

    st.markdown("**Âm tiết đầu vào**")
    st.code(" ".join(syllables), language=None)

    st.markdown("**Vết thuật toán từng bước**")
    st.caption("Vết dùng cùng matcher, từ điển và Lmax với kết quả ở trên; "
               "start/end là chỉ số âm tiết, end không bao gồm.")
    trace_rows = _trace_rows(text, method, matcher=chosen)
    if not trace_rows:
        st.info("Mã lõi không trả về vết thuật toán cho lần chạy này.")
        return
    st.dataframe(trace_rows, width="stretch", hide_index=True)
    matched = sum(1 for row in trace_rows if row["Hành động"] == ACTIONS["MATCHED"])
    st.caption(f"{len(trace_rows)} bước · {matched} bước khớp từ điển · "
               f"{sum(1 for row in trace_rows if row['Hành động'] == ACTIONS['SINGLE'])} "
               f"âm tiết đơn do ngoài từ điển")


def _trace_rows(text: str, method: str, matcher=None) -> List[Dict[str, object]]:
    """Gọi ``trace`` của mã lõi rồi chuẩn hóa tên cột cho hiển thị."""
    try:
        from engine.tokenizer import max_matching
    except ImportError:
        return []
    try:
        raw = matcher.trace(text, method=method) if matcher is not None else max_matching.trace(text, method=method)
    except Exception as error:  # noqa: BLE001 - vết lỗi phải hiện ra
        st.warning(f"Không lấy được vết thuật toán: {error}")
        return []
    rows: List[Dict[str, object]] = []
    for step in raw:
        action = str(step.get("action", ""))
        rows.append({
            "Bước": step.get("step"),
            "Hướng": str(step.get("direction", "")).upper(),
            "Từ": step.get("start"),
            "Đến": step.get("end"),
            "Ứng viên": step.get("candidate"),
            "Độ dài": step.get("size"),
            "Khớp": "có" if step.get("matched") else "không",
            "Hành động": ACTIONS.get(action, action),
        })
    return rows


def render_tier1_tab(filter_tier1, tier1_error) -> None:
    st.subheader("Lọc từ dừng tầng 1 và guard phủ định")
    if filter_tier1 is None:
        st.info(tier1_error or "Chưa có `engine.tokenizer.stopwords`.")
        return
    text = st.text_area("Văn bản đầu vào", key="tier1_text", height=90,
                        value="nghiên cứu bệnh không truyền nhiễm ở người cao tuổi")
    if not st.button("Chạy lọc tầng 1", key="run_tier1"):
        st.caption("Nhấn “Chạy lọc tầng 1” để xem danh sách token trước và sau khi lọc.")
        return
    if not text.strip():
        st.warning("Chuỗi rỗng: không có token nào để lọc.")
        return
    matcher, normalize, _filter, _error = cached_matcher()
    if matcher is None:
        st.info("Chưa có mã lõi tách từ để sinh token đầu vào.")
        return
    tokens = matcher.tokenize(text)
    try:
        filtered = filter_tier1(tokens)
    except Exception as error:  # noqa: BLE001 - lỗi bộ lọc phải hiện ra
        st.error(f"Không lọc được: {error}")
        return
    st.markdown("**Token trước khi lọc**")
    st.code(token_view(tokens), language=None)
    st.markdown("**Token sau khi lọc tầng 1**")
    st.code(token_view(filtered), language=None)
    removed = [token for token in tokens if token not in set(filtered)]
    st.caption(f"{len(tokens)} token trước lọc · {len(filtered)} còn lại · "
               f"{len(removed)} bị loại")
    if removed:
        st.caption("Bị loại: " + " · ".join(sorted(set(removed))))
    if not filtered:
        st.warning("Sau khi lọc tầng 1 không còn token nào. Truy vấn dạng này không "
                   "tạo được vectơ; hệ thống truy xuất sẽ không có kết quả.")
    rows = guard_report(tokens, filtered, tier1_filter_instance())
    st.dataframe(rows, width="stretch", hide_index=True)
    protected = [row for row in rows if row["Guard"] == "có"]
    rescued = [row for row in rows
               if row["Guard"] == "có" and row["Kết quả"] == "giữ"]
    if protected:
        st.success(f"Guard giữ lại {len(rescued)}/{len(protected)} token thuộc danh mục "
                   f"bảo vệ: " + " · ".join(str(row["Token"]) for row in rescued))
    summary = tier1_resource_summary(tier1_filter_instance())
    if summary:
        st.markdown("**Số mục thực tế của các danh mục**")
        st.dataframe(summary, width="stretch", hide_index=True)


def render_resource_tab() -> None:
    st.subheader("Tài nguyên ngôn ngữ học")
    st.dataframe(resource_status(), width="stretch", hide_index=True)
    st.caption("Số mục đếm các dòng không rỗng và không phải chú thích. "
               "Nguồn và giấy phép của từng tài nguyên ghi trong `resources/*_source.md`.")
    st.markdown(
        """
Ba danh mục có vai trò khác nhau và không được trộn:

* **Từ điển** quyết định âm tiết nào ghép được thành từ khi chạy FMM, BMM, BiMM.
* **Hư từ ngữ pháp tầng 1** là danh sách từ chức năng bị loại sau khi tách từ.
* **Guard bảo vệ** là danh mục từ không bao giờ được loại ở tầng 1, vì phủ định và
  quan hệ nhân quả làm đảo chiều nghĩa của truy vấn.
""")


def main() -> None:
    st.set_page_config(page_title="Tách từ & Lọc tầng 1", layout="wide")
    st.title("1. Tách từ Maximum Matching & Lọc từ dừng tầng 1")

    matcher, normalize, filter_tier1, error = cached_matcher()
    if matcher is None or normalize is None:
        st.error(error or "Chưa có mã lõi `engine.tokenizer`.")
        st.info("Trang này cần các module do M1 phụ trách: "
                "`engine/tokenizer/max_matching.py`, `engine/tokenizer/preprocessor.py` "
                "và `engine/tokenizer/stopwords.py`.")
        render_resource_tab()
        st.stop()

    st.caption("Số liệu và vết thuật toán dưới đây do mã lõi M1 sinh ra trực tiếp.")
    trace_tab, tier1_tab, resource_tab = st.tabs(
        ["Tách từ FMM/BMM/BiMM", "Lọc tầng 1 & guard", "Tài nguyên"])
    with trace_tab:
        render_trace_tab(matcher, normalize)
    with tier1_tab:
        render_tier1_tab(filter_tier1, error)
    with resource_tab:
        render_resource_tab()


main()
