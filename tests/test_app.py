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

APP_DIR = Path(__file__).resolve().parents[1] / "app"

DEMO = "pages/2_Demo.py"
PAGES = [DEMO, "pages/3_Segmentation.py", "pages/4_Prediction.py"]
CARD_TITLES = ("#### Who is this customer?", "#### Will they come back?")


def _markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def _mock_metrics() -> dict:
    return C.read_metrics(C.mock_path(C.METRICS_JSON))


def _demo(run_app_page):
    at = run_app_page("Home.py")
    at.switch_page(DEMO).run()
    assert not at.exception, at.exception
    return at


def test_home_overview_runs_on_mock(run_app_page):
    at = run_app_page("Home.py")
    assert not at.exception, at.exception
    assert any("Mock data" in w.value for w in at.sidebar.warning)
    assert len(at.metric) == 8  # 4 headline numbers + 4 on the dataset card
    assert at.get("graphviz_chart")  # pipeline always visible (no expander)
    assert not at.main.expander
    assert not at.toggle  # no Guided Tour
    assert [b.key for b in at.sidebar.button] == ["theme_toggle"]


def test_demo_runs_on_mock(run_app_page):
    at = _demo(run_app_page)
    text = _markdown(at)
    assert all(t in text for t in CARD_TITLES)
    assert "What to offer" not in text and not at.multiselect  # recommendation hidden
    assert "Mock segment" in text  # segment badge names come from the artifacts
    assert "Suggested action for this segment" in text  # insights for this request
    assert len(at.number_input) == 6
    names = _mock_metrics()["segment"]["segment_names"]
    presets = [b.label for b in at.button if (b.key or "").startswith("preset_")]
    assert presets == [f"Typical {n}" for n in names.values()]


def test_demo_enter_applies_typed_numbers(run_app_page):
    at = _demo(run_app_page)
    at.number_input(key="demo_in_recency").set_value(321).run()
    assert not any("your own numbers" in c.value for c in at.caption)  # not applied yet
    at.number_input(key="demo_in_recency").set_value(321)
    at.button(key="demo_enter").click().run()
    assert not at.exception, at.exception
    assert any("your own numbers" in c.value for c in at.caption)


def test_theme_button_switches_to_dark_and_back(run_app_page):
    import streamlit as st

    at = run_app_page("Home.py")
    try:
        at.sidebar.button(key="theme_toggle").click().run()
        assert not at.exception, at.exception
        assert st.get_option("theme.base") == "dark"
        assert at.sidebar.button(key="theme_toggle").label == "Light mode"
        at.switch_page(DEMO).run()  # charts recoloured without errors
        assert not at.exception, at.exception
        at.sidebar.button(key="theme_toggle").click().run()
        assert st.get_option("theme.base") == "light"
    finally:
        st._config.set_option("theme.base", "light")


def test_home_then_every_page_runs_on_mock(run_app_page):
    at = run_app_page("Home.py")
    assert not at.exception, at.exception
    for page in PAGES:
        at.switch_page(page).run()
        assert not at.exception, f"{page}: {at.exception}"
        assert at.title[0].value != "RetailPulse", page  # really switched


def test_preset_button_fills_inputs_and_cards_render(run_app_page):
    at = _demo(run_app_page)
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
    assert len(at.get("plotly_chart")) == 2  # PCA map + P(return) bars


def test_demo_inputs_survive_a_page_switch(run_app_page):
    at = _demo(run_app_page)
    at.number_input(key="demo_in_recency").set_value(321)
    at.button(key="demo_enter").click().run()
    at.switch_page("Home.py").run()
    at.switch_page(DEMO).run()
    assert not at.exception, at.exception
    assert at.number_input(key="demo_in_recency").value == 321


def test_prediction_outcome_visible_and_knn_slider(run_app_page):
    at = run_app_page("Home.py")
    at.switch_page("pages/4_Prediction.py").run()
    assert not at.exception, at.exception
    assert any(m.label == "Lift" for m in at.metric)  # Time Machine outcome always shown
    slider = at.select_slider(key="pred_knn_k")
    assert slider.value == _mock_metrics()["classify"]["knn_k"]  # starts at the best k
    assert len(slider.options) >= 2
    slider.set_value(1).run()
    assert not at.exception, at.exception
    assert any("overfitting" in m.value for m in at.markdown)
    assert {m.label for m in at.metric} >= {"Train accuracy", "Test accuracy", "Gap (train − test)"}


def test_segmentation_has_2d_map_only(run_app_page):
    at = run_app_page("Home.py")
    at.switch_page("pages/3_Segmentation.py").run()
    assert not at.exception, at.exception
    assert not at.toggle


def test_segmentation_pc_lines_show_variance_share(run_app_page):
    at = run_app_page("Home.py")
    at.switch_page("pages/3_Segmentation.py").run()
    assert any("**PC1** (" in m.value and "of variance" in m.value for m in at.markdown)


def test_prediction_without_test_rows_for_best_model(mock_artifacts, tmp_path, monkeypatch):
    """Review fix: metrics name a best model with no test predictions -> info, no KeyError."""
    from streamlit.testing.v1 import AppTest

    (tmp_path / "metrics.json").write_text(
        json.dumps({"classify": {"best_model": "lda"}}), encoding="utf-8"
    )
    monkeypatch.setenv("RETAILPULSE_ARTIFACTS", str(tmp_path))  # every other file: mock
    at = AppTest.from_file(str(APP_DIR / "Home.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    at.switch_page("pages/4_Prediction.py").run()
    assert not at.exception, at.exception
    assert any("No test-window predictions" in i.value for i in at.info)


def test_horizon_days_falls_back_to_config(mock_artifacts, tmp_path, monkeypatch):
    from components import data

    from retailpulse.config import HORIZON_DAYS

    (tmp_path / "metrics.json").write_text(json.dumps({"features": {}}), encoding="utf-8")
    monkeypatch.setenv(data.ENV_VAR, str(tmp_path))
    assert data.horizon_days() == HORIZON_DAYS
    (tmp_path / "metrics.json").write_text(
        json.dumps({"features": {"horizon_days": 30}}), encoding="utf-8"
    )
    assert data.horizon_days() == 30


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


def _badge_and_verdict(at) -> tuple[str, str]:
    """(segment badge markdown, verdict markdown) of the demo cards."""
    badge = next(m.value for m in at.markdown if m.value.startswith("Segment: "))
    verdict = next(m.value for m in at.markdown if "come back** within" in m.value)
    return badge, verdict


def test_home_first_and_last_preset_change_badge_and_probability(run_app_page):
    """Typical <first segment> vs Typical <last segment>: badge + chosen P(return) change."""
    at = _demo(run_app_page)
    presets = pd.read_parquet(C.mock_path(C.SEGMENT_PRESETS_PARQUET))
    names = {int(k): v for k, v in _mock_metrics()["segment"]["segment_names"].items()}
    first, last = int(presets["segment"].min()), int(presets["segment"].max())
    assert first != last
    seen = {}
    for sid in (first, last):
        at.button(key=f"preset_{sid}").click().run()
        assert not at.exception, at.exception
        badge, verdict = _badge_and_verdict(at)
        assert names[sid] in badge and f"(segment {sid})" in badge
        p = re.search(r"\(([^)]+)\) gives (\d+)%", verdict)
        assert p, verdict
        seen[sid] = (badge, p.group(1), int(p.group(2)))
    assert seen[first][0] != seen[last][0]  # segment badge text changed
    assert seen[first][1] == seen[last][1]  # same chosen model ...
    assert seen[first][2] != seen[last][2]  # ... different P(return)
