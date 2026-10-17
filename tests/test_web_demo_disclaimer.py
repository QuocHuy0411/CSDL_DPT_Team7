"""Static demo must not present unadjudicated label scores as final evidence."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGES = (ROOT / "app/index.html", ROOT / "web_demo.html")


def test_static_demos_are_synced_and_flag_provisional_scores():
    pages = [path.read_text(encoding="utf-8") for path in PAGES]
    assert pages[0] == pages[1]
    page = pages[0]
    assert "chưa phải kết luận cuối" in page
    assert "37 quyết định" in page
    assert "docs/label_review.md" in page
    assert "results/label_review_resolution.json" in page
    assert all(case in page for case in ("SG010", "SG053", "SG089"))
    assert "chưa xác minh được người thứ hai độc lập" in page
    assert "Đo lường chính thức" not in page
    assert "C3 (Khuyên dùng)" not in page
