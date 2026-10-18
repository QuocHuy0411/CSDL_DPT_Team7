"""Static demo must not present unadjudicated label scores as final evidence."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGES = (ROOT / "app/index.html", ROOT / "web_demo.html")


def test_static_demos_are_synced_and_explain_evaluation_scope():
    pages = [path.read_text(encoding="utf-8") for path in PAGES]
    assert pages[0] == pages[1]
    page = pages[0]
    assert "Nhãn theo pool chưa bao phủ toàn corpus" in page
    assert "chưa có vòng gán nhãn độc lập đầy đủ thứ hai" in page
    assert "báo cáo Word chính thức" in page
    assert "Đo lường chính thức" not in page
    assert "C3 (Khuyên dùng)" not in page
