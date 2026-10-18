"""Kiểm thử giao diện Streamlit bằng AppTest (không cần trình duyệt).

Phạm vi: trang chủ và trang Tách từ/Lọc tầng 1. Các test chạy thật mã lõi M1 nếu
đã có; nếu chưa có thì bỏ qua kèm lý do, thay vì dựng kết quả giả để test "xanh".

Nguyên tắc: mọi khẳng định về nội dung hiển thị phải truy được về lời gọi hàm
thật của M1. Trang không được hiện số liệu suy đoán, nên test cũng kiểm tra trường
hợp thiếu mã lõi phải báo lỗi và dừng.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

HOME_PAGE = ROOT / "app/Home.py"
TOKENIZER_PAGE = ROOT / "app/pages/1_Tokenizer.py"
TIMEOUT = 180

DEFAULT_TEXT = "trí tuệ nhân tạo trong giáo dục"
CUSTOM_TEXT = "nghiên cứu bệnh không truyền nhiễm ở người cao tuổi"
REQUIRED_TRACE_KEYS = {"step", "direction", "start", "end", "candidate", "size",
                      "matched", "action"}


def core_available() -> bool:
    try:
        import engine.tokenizer.max_matching  # noqa: F401
        import engine.tokenizer.preprocessor  # noqa: F401
    except ImportError:
        return False
    return True


requires_core = pytest.mark.skipif(
    not core_available(),
    reason="mã lõi M1 (engine.tokenizer.max_matching/preprocessor) chưa có trong repo")

#: Từ điển nhỏ, tự định nghĩa, cho các khẳng định toán học. Không phụ thuộc
#: ``resources/dict.txt`` nên fixture không đổi khi từ điển đóng gói được thay.
SMALL_DICTIONARY = ["học sinh", "sinh học", "học", "sinh"]
SMALL_DICTIONARY_TEXT = "học sinh học sinh học"


def run_page(path: Path, **kwargs) -> AppTest:
    app = AppTest.from_file(str(path), default_timeout=TIMEOUT, **kwargs)
    app.run()
    assert not app.exception, [error.value for error in app.exception]
    return app


def table_records(table) -> list:
    """Trả về list dict từ một phần tử dataframe của AppTest."""
    value = table.value
    if hasattr(value, "to_dict"):
        return list(value.to_dict("records"))
    return list(value)


def click(app: AppTest, label: str):
    matches = [button for button in app.button if button.label == label]
    assert matches, f"không thấy nút {label!r}; hiện có {[b.label for b in app.button]}"
    matches[0].click().run()
    assert not app.exception, [error.value for error in app.exception]
    return app


def trace_tables(app: AppTest) -> list:
    """Các bảng vết thuật toán; bảng tài nguyên luôn có mặt nên phải lọc theo cột."""
    found = []
    for table in app.dataframe:
        records = table_records(table)
        if records and "Hành động" in records[0]:
            found.append(records)
    return found


def tables_with_column(app: AppTest, column: str) -> list:
    """Mọi hàng của các bảng có cột đã cho, gộp lại."""
    rows = []
    for table in app.dataframe:
        records = table_records(table)
        if records and column in records[0]:
            rows.extend(records)
    return rows


def token_tables(app: AppTest) -> list:
    """Các bảng gán nhãn token của tab lọc tầng 1."""
    found = []
    for table in app.dataframe:
        records = table_records(table)
        if records and "Kết quả" in records[0]:
            found.append(records)
    return found


# ------------------------------------------------------------------ trang chủ


def test_home_renders_without_exception():
    app = run_page(HOME_PAGE)
    assert app.title


def test_home_states_team_seven_identity():
    app = run_page(HOME_PAGE)
    rendered = " ".join([node.value for node in app.subheader])
    assert "Team7" in rendered
    assert "Nhóm 8" not in rendered


def test_home_reports_real_corpus_numbers_when_manifest_present():
    manifest = ROOT / "data/corpus_manifest.json"
    if not manifest.is_file():
        pytest.skip("chưa có data/corpus_manifest.json")
    import json

    documents = json.loads(manifest.read_text(encoding="utf-8"))["corpus_documents"]
    app = run_page(HOME_PAGE)
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Tài liệu"] == f"{documents:,}"
    captions = " ".join(item.value for item in app.caption)
    assert "SHA256 corpus" in captions


def test_home_never_claims_official_evaluation_while_qrels_pending():
    qrels = ROOT / "data/ground_truth/mm/qrels.json"
    if not qrels.is_file():
        qrels = ROOT / "data/ground_truth/qrels.json"
    if not qrels.is_file():
        pytest.skip("chưa có data/ground_truth/mm/qrels.json")
    import json

    payload = json.loads(qrels.read_text(encoding="utf-8"))
    if payload.get("status") == "human_annotation_confirmed":
        pytest.skip("qrels đã confirmed nên trang chủ có thể báo đủ điều kiện")
    app = run_page(HOME_PAGE)
    assert not app.success, "trang chủ không được báo đủ điều kiện khi qrels còn pending"
    warnings = [item.value for item in app.warning]
    assert any("chưa đủ điều kiện" in message.lower() for message in warnings)


def test_home_does_not_invent_index_numbers():
    app = run_page(HOME_PAGE)
    rows = table_records(app.dataframe[0])
    assert {row["Cấu hình"] for row in rows} == {"C1", "C2", "C3", "C4"}
    for row in rows:
        if row["Tài liệu"] == "chưa build":
            assert row["Stoplist"] == "—"


# ---------------------------------------------------- trang tách từ: khung nền


def test_tokenizer_page_has_three_tabs():
    app = run_page(TOKENIZER_PAGE)
    assert [tab.label for tab in app.tabs] == [
        "Tách từ FMM/BMM/BiMM", "Lọc tầng 1 & guard", "Tài nguyên"]


def test_tokenizer_page_does_not_compute_before_button_press():
    app = run_page(TOKENIZER_PAGE)
    assert not trace_tables(app), "trang không được tự dựng bảng vết trước khi người dùng chạy"
    assert any("Nhấn" in item.value for item in app.caption)


def test_tokenizer_page_lists_resources_even_without_core():
    app = run_page(TOKENIZER_PAGE)
    rows = table_records(app.dataframe[-1])
    assert {row["Tài nguyên"] for row in rows} >= {"Từ điển", "Hư từ ngữ pháp (tầng 1)",
                                                   "Guard bảo vệ"}


@requires_core
def test_tokenizer_page_runs_all_three_methods_on_default_text():
    """Chọn thật từng thuật toán trên giao diện và so với API của mã lõi.

    Không so với một danh sách token viết tay: giá trị hiển thị phải bằng đúng
    kết quả của mã lõi cho thuật toán đang chọn, vì từ điển đóng gói có thể đổi.
    """
    import engine.tokenizer.max_matching as maximum_matching

    for method in ("fmm", "bmm", "bimm"):
        app = run_page(TOKENIZER_PAGE)
        app.text_area[0].set_value(DEFAULT_TEXT)
        app.radio[0].set_value(method)
        app.run()
        assert not app.exception
        app = click(app, "Chạy thuật toán")

        expected_tokens = maximum_matching.tokenize(DEFAULT_TEXT, method=method)
        expected_trace = maximum_matching.trace(DEFAULT_TEXT, method=method)
        assert expected_tokens, f"{method} phải cho ít nhất một token"
        rendered = app.code[0].value
        assert rendered == " · ".join(expected_tokens), (
            f"{method}: giao diện hiển thị {rendered!r} nhưng mã lõi trả "
            f"{expected_tokens!r}")
        shown = trace_tables(app)[0]
        assert len(shown) == len(expected_trace)
        directions = {row["Hướng"] for row in shown}
        assert directions <= {"FMM", "BMM"}
        assert directions == ({"FMM", "BMM"} if method == "bimm" else {method.upper()})
        assert any(f"{len(shown)} bước" in item.value for item in app.caption)


@requires_core
def test_tokenizer_page_runs_custom_text():
    import engine.tokenizer.max_matching as maximum_matching

    expected = maximum_matching.tokenize(CUSTOM_TEXT, method="bimm")
    app = run_page(TOKENIZER_PAGE)
    app.text_area[0].set_value(CUSTOM_TEXT)
    app = click(app, "Chạy thuật toán")
    codes = [item.value for item in app.code]
    assert expected[0] in codes[0]
    assert "không" in codes[0]


@requires_core
def test_trace_table_exposes_required_fields():
    app = run_page(TOKENIZER_PAGE)
    app.text_area[0].set_value(DEFAULT_TEXT)
    app = click(app, "Chạy thuật toán")
    records = trace_tables(app)[0]
    assert records
    row = records[0]
    assert {"Bước", "Hướng", "Từ", "Đến", "Ứng viên", "Độ dài", "Khớp", "Hành động"} \
        <= set(row)
    assert row["Hướng"] in {"FMM", "BMM"}
    assert row["Đến"] > row["Từ"]
    assert row["Độ dài"] > 0
    assert row["Khớp"] in {"có", "không"}
    assert row["Hành động"] in {"Khớp từ điển", "Thu cửa sổ", "Âm tiết đơn (OOV)"}
    assert any(item["Hành động"] == "Khớp từ điển" for item in records)


@requires_core
def test_trace_has_no_row_where_matched_contradicts_action():
    import engine.tokenizer.max_matching as maximum_matching

    app = run_page(TOKENIZER_PAGE)
    app.text_area[0].set_value(DEFAULT_TEXT)
    app = click(app, "Chạy thuật toán")
    shown = trace_tables(app)[0]
    raw = maximum_matching.trace(DEFAULT_TEXT, method="bimm")
    assert len(shown) == len(raw)
    for shown_row, raw_row in zip(shown, raw):
        assert shown_row["Ứng viên"] == raw_row["candidate"]
        assert (shown_row["Khớp"] == "có") == bool(raw_row["matched"])
        if raw_row["action"] == "MATCHED":
            assert raw_row["matched"] is True
            assert shown_row["Hành động"] == "Khớp từ điển"
        else:
            assert raw_row["matched"] is False
        assert shown_row["Độ dài"] == raw_row["size"]
        assert REQUIRED_TRACE_KEYS <= set(raw_row)


@requires_core
def test_fmm_and_bmm_disagree_on_explicit_dictionary():
    """Fixture nhập nhằng kinh điển trên từ điển nhỏ tự định nghĩa.

    Dùng từ điển riêng thay vì ``resources/dict.txt`` để khẳng định này là về thuật
    toán, không phụ thuộc từ điển đóng gói có còn chứa ``học sinh`` hay không.
    """
    import engine.tokenizer.max_matching as maximum_matching

    matcher = maximum_matching.MaximumMatching(words=SMALL_DICTIONARY, max_syllables=4)
    forward = matcher.tokenize(SMALL_DICTIONARY_TEXT, method="fmm")
    backward = matcher.tokenize(SMALL_DICTIONARY_TEXT, method="bmm")
    assert forward != backward, (
        "fixture này phải cho FMM khác BMM; nếu bằng nhau thì fixture không còn "
        "phân biệt được và test không kiểm được gì")
    assert forward == ["học_sinh", "học_sinh", "học"]
    assert backward == ["học", "sinh_học", "sinh_học"]


@requires_core
def test_bimm_tie_break_prefers_bmm_on_explicit_dictionary():
    """Hai cách tách cùng số token và cùng số từ đơn thì BiMM chọn BMM."""
    import engine.tokenizer.max_matching as maximum_matching

    matcher = maximum_matching.MaximumMatching(words=SMALL_DICTIONARY, max_syllables=4)
    assert matcher.tokenize(SMALL_DICTIONARY_TEXT, method="bimm") == \
        matcher.tokenize(SMALL_DICTIONARY_TEXT, method="bmm")


@requires_core
def test_ui_token_output_changes_with_selected_method():
    """Đổi thuật toán trên giao diện thì kết quả hiển thị phải đổi theo."""
    import engine.tokenizer.max_matching as maximum_matching

    text = SMALL_DICTIONARY_TEXT
    rendered = {}
    for method in ("fmm", "bmm"):
        app = run_page(TOKENIZER_PAGE)
        app.text_area[0].set_value(text)
        app.radio[0].set_value(method)
        app.run()
        app = click(app, "Chạy thuật toán")
        rendered[method] = app.code[0].value
        assert rendered[method] == " · ".join(
            maximum_matching.tokenize(text, method=method))
    assert rendered["fmm"] != rendered["bmm"], (
        "giao diện phải đổi kết quả khi đổi thuật toán; nếu không thì bộ chọn "
        "thuật toán không có tác dụng")


@requires_core
def test_empty_input_warns_and_produces_no_table():
    app = run_page(TOKENIZER_PAGE)
    app.text_area[0].set_value("")
    app = click(app, "Chạy thuật toán")
    assert any("rỗng" in item.value for item in app.warning)
    assert not trace_tables(app)


@requires_core
def test_whitespace_and_punctuation_only_input_is_handled():
    import engine.tokenizer.max_matching as maximum_matching

    for text in ("   ", "!!! ??? ---", "\n\t"):
        assert maximum_matching.tokenize(text) == []
        assert maximum_matching.trace(text) == []
    app = run_page(TOKENIZER_PAGE)
    app.text_area[0].set_value("   ")
    app = click(app, "Chạy thuật toán")
    assert not app.exception
    assert any("rỗng" in item.value for item in app.warning)


@requires_core
def test_tier1_tab_keeps_negation_tokens():
    app = run_page(TOKENIZER_PAGE)
    app.text_area[1].set_value("chúng ta không nên chủ quan")
    app = click(app, "Chạy lọc tầng 1")
    codes = [item.value for item in app.code]
    before, after = codes[0], codes[1]
    assert "không" in before
    assert "không" in after, "guard phải giữ lại từ phủ định"
    assert any("Guard" in item.value for item in app.success)


@requires_core
def test_tier1_tab_marks_guard_rows():
    app = run_page(TOKENIZER_PAGE)
    app.text_area[1].set_value(CUSTOM_TEXT)
    app = click(app, "Chạy lọc tầng 1")
    records = token_tables(app)[0]
    assert {"#", "Token", "Kết quả", "Lý do", "Guard"} <= set(records[0])
    guarded = [row for row in records if row["Guard"] == "có"]
    assert guarded, "fixture này phải có ít nhất một token được guard giữ lại"
    for row in guarded:
        assert row["Kết quả"] == "giữ"


@requires_core
def test_tier1_tab_reports_real_resource_counts():
    from engine.tokenizer import stopwords as stopword_module

    tier1 = stopword_module.get_default_filter()
    app = run_page(TOKENIZER_PAGE)
    app.text_area[1].set_value(CUSTOM_TEXT)
    app = click(app, "Chạy lọc tầng 1")
    counts = {row["Danh mục"]: row["Số mục"] for row in tables_with_column(app, "Danh mục")}
    assert counts["Hư từ ngữ pháp tầng 1"] == len(tier1.stopwords)
    assert counts["Guard bảo vệ"] == len(tier1.guards)
    assert counts["Danh sách loại hiệu lực"] == len(tier1.effective_stopwords)


@requires_core
def test_tier1_tab_handles_empty_input():
    app = run_page(TOKENIZER_PAGE)
    app.text_area[1].set_value("")
    app = click(app, "Chạy lọc tầng 1")
    assert any("rỗng" in item.value for item in app.warning)


@requires_core
def test_token_view_and_resource_status_helpers():
    """Kiểm tra các hàm thuần của trang mà không phụ thuộc phiên Streamlit."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("tokenizer_page", TOKENIZER_PAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.token_view([]) == "(rỗng)"
    assert module.token_view(["a", "b"]) == "a · b"
    rows = module.resource_status()
    assert all(row["Tài nguyên"] for row in rows)
    report = module.guard_report(["và", "không", "bóng"], ["không", "bóng"], None)
    assert [row["Kết quả"] for row in report] == ["loại", "giữ", "giữ"]
    assert module.syllables_of(module.cached_matcher()[1], "Xin chào, Việt Nam!") == [
        "xin", "chào", "việt", "nam"]
