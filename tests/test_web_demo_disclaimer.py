"""Static demo must not present unadjudicated label scores as final evidence."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "app/index.html"


def test_static_demos_are_synced_and_explain_evaluation_scope():
    page = PAGE.read_text(encoding="utf-8")
    assert "Nhãn theo pool chưa bao phủ toàn corpus" in page
    assert "chưa có vòng gán nhãn độc lập đầy đủ thứ hai" in page
    assert "báo cáo Word chính thức" in page
    assert "Đo lường chính thức" not in page
    assert "C3 (Khuyên dùng)" not in page
