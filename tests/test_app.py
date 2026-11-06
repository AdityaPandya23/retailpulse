"""App smoke tests on the MOCK artifacts (env var RETAILPULSE_ARTIFACTS), never real data.

st.page_link only resolves when Home.py is the entrypoint, so every page is opened from Home
with at.switch_page (never AppTest.from_file on a page file).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from retailpulse import contracts as C
from retailpulse import demo
from retailpulse.viz import rules as VR

APP_DIR = Path(__file__).resolve().parents[1] / "app"

PAGES = [
    "pages/2_Data.py",
    "pages/3_Segmentation.py",
    "pages/4_Prediction.py",
    "pages/5_Recommendation.py",
    "pages/6_Insights.py",
]
CARD_TITLES = ("#### Who is this customer?", "#### Will they come back?", "#### What to offer")


def _markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def _mock_metrics() -> dict:
    return C.read_metrics(C.mock_path(C.METRICS_JSON))


def test_home_runs_on_mock(run_app_page):
    at = run_app_page("Home.py")
    assert not at.exception, at.exception
    assert any("Mock data" in w.value for w in at.sidebar.warning)
    assert len(at.metric) == 4  # clean rows, customers, revenue, date range
    text = _markdown(at)
    assert all(t in text for t in CARD_TITLES)
    assert "Mock segment" in text  # segment badge names come from the artifacts
    assert len(at.number_input) == 6
    names = _mock_metrics()["segment"]["segment_names"]
    assert [b.label for b in at.button] == [f"Typical {n}" for n in names.values()]


def test_home_then_every_page_runs_on_mock(run_app_page):
    at = run_app_page("Home.py")
    assert not at.exception, at.exception
    for page in PAGES:
        at.switch_page(page).run()
        assert not at.exception, f"{page}: {at.exception}"
        assert at.title[0].value != "RetailPulse", page  # really switched


def test_preset_button_fills_inputs_and_cards_render(run_app_page):

    at = run_app_page("Home.py")
    presets = pd.read_parquet(C.mock_path(C.SEGMENT_PRESETS_PARQUET))
    sid = int(presets["segment"].max())  # not the default (first) segment
    at.button(key=f"preset_{sid}").click().run()
    assert not at.exception, at.exception
    expected = demo.preset_values(presets, sid)
    for f, v in expected.items():
        assert at.number_input(key=f"demo_in_{f}").value == v
    text = _markdown(at)
    assert all(t in text for t in CARD_TITLES)
    assert "come back** within" in text  # verdict of the chosen model
    assert sum("best seller" in m.value or "because you bought" in m.value for m in at.markdown)
    assert len(at.get("plotly_chart")) == 2  # PCA map + P(return) bars


def test_home_inputs_survive_a_page_switch(run_app_page):
    at = run_app_page("Home.py")
    at.number_input(key="demo_in_recency").set_value(321).run()
    at.switch_page("pages/2_Data.py").run()
    at.switch_page("Home.py").run()
    assert not at.exception, at.exception
    assert at.number_input(key="demo_in_recency").value == 321


def test_prediction_reveal_and_knn_slider(run_app_page):
    at = run_app_page("Home.py")
    at.switch_page("pages/4_Prediction.py").run()
    assert not at.exception, at.exception
    assert not any(m.label == "Lift" for m in at.metric)  # outcome hidden before the reveal
    at.button(key="pred_reveal").click().run()
    assert not at.exception, at.exception
    assert any(m.label == "Lift" for m in at.metric)
    slider = at.select_slider(key="pred_knn_k")
    assert slider.value == _mock_metrics()["classify"]["knn_k"]  # starts at the best k
    assert len(slider.options) >= 2
    slider.set_value(1).run()
    assert not at.exception, at.exception
    assert any("overfitting" in m.value for m in at.markdown)
    assert {m.label for m in at.metric} >= {"Train accuracy", "Test accuracy", "Gap (train − test)"}


def test_segmentation_3d_toggle_and_recommendation_picker(run_app_page):
    at = run_app_page("Home.py")
    at.switch_page("pages/3_Segmentation.py").run()
    at.toggle(key="seg_pca_3d").set_value(True).run()
    assert not at.exception, at.exception
    at.switch_page("pages/5_Recommendation.py").run()
    assert not at.exception, at.exception
    rules = pd.read_parquet(C.mock_path(C.RULES_PARQUET))
    products = pd.read_parquet(C.mock_path(C.PRODUCTS_PARQUET))
    codes = VR.antecedent_products(rules, products)["stock_code"].tolist()
    at.selectbox(key="rec_product").set_value(codes[-1]).run()
    assert not at.exception, at.exception
    assert at.selectbox(key="rec_product").value == codes[-1]


def test_insights_shows_readme_limitations(run_app_page):
    from retailpulse.insights import limitations

    at = run_app_page("Home.py")
    at.switch_page("pages/6_Insights.py").run()
    assert not at.exception, at.exception
    expected = limitations(C.README_MD.read_text(encoding="utf-8"))
    assert len(expected) == 4
    assert all(item in _markdown(at) for item in expected)


def test_guided_tour_toggle_shows_callouts(run_app_page):
    at = run_app_page("Home.py")
    assert not at.info
    at.sidebar.toggle(key="tour_toggle").set_value(True).run()
    assert not at.exception
    assert any("Tour 1" in i.value for i in at.info)


# ---------------- data.py: per-file / per-section fallback to mock ----------------
def test_data_falls_back_per_file_and_section(mock_artifacts, tmp_path, monkeypatch):
    from components import data

    # a "real" root with only metrics.json holding one section
    (tmp_path / "metrics.json").write_text(json.dumps({"segment": {"k": 99}}), encoding="utf-8")
    monkeypatch.setenv(data.ENV_VAR, str(tmp_path))
    assert data.metrics("segment") == {"k": 99}  # real section wins
    assert data.metrics("classify")["mock"] is True  # missing section -> mock
    file, is_mock = data.resolve(C.SEGMENTS_PARQUET)
    assert is_mock and file.exists()
    listed = data.mock_files()
    assert "segments/segments.parquet" in listed and "metrics.json [classify]" in listed
    assert "metrics.json [segment]" not in listed
    assert data.table(C.SEGMENTS_PARQUET) is not None
    assert data.model(C.CLASSIFIER_JOBLIB)["model"] == data.metrics("classify")["best_model"]


def test_data_on_mock_root(mock_artifacts):
    from components import data

    assert data.missing_files() == []
    names = data.segment_names()
    assert names and all(isinstance(k, int) for k in names)
    assert set(data.segment_colours()) == set(names.values())
    assert len(data.limitations()) == 4


@pytest.mark.parametrize("page", ["Home.py", *PAGES])
def test_page_links_point_at_real_files(page):
    source = (APP_DIR / page).read_text(encoding="utf-8")
    targets = re.findall(r'st\.page_link\(\s*"([^"]+)"', source)
    assert targets, page
    assert all((APP_DIR / t).is_file() for t in targets), targets
