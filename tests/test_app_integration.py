"""Smoke-test the integrated M2 pages against the built, locked indexes."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
SEARCH_PAGE = ROOT / "app/pages/3_Search.py"
ANALYTICS_PAGE = ROOT / "app/pages/2_Analytics.py"
INDEX_C4 = ROOT / "data/indexes/C4"


@pytest.mark.skipif(
    not (INDEX_C4 / "build_metadata.json").is_file(),
    reason="C4 index has not been built in this checkout",
)
def test_search_page_runs_query_and_renders_explanations():
    app = AppTest.from_file(str(SEARCH_PAGE), default_timeout=180).run()
    assert not app.exception
    assert not app.error
    assert len(app.selectbox) == 1
    assert app.selectbox[0].value == "C4"

    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(app.subheader) == 10
    assert len(app.dataframe) == 10


@pytest.mark.skipif(
    not (INDEX_C4 / "tdv_scores.json").is_file(),
    reason="C4 TDV scores have not been built in this checkout",
)
def test_analytics_page_renders_real_tdv_values():
    app = AppTest.from_file(str(ANALYTICS_PAGE), default_timeout=180).run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 1
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value.shape[0] > 0
