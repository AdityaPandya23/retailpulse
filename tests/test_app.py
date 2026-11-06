"""App smoke tests on the MOCK artifacts (env var RETAILPULSE_ARTIFACTS), never real data."""

from __future__ import annotations

import json

import pytest

from retailpulse import contracts as C


def test_home_runs_on_mock(run_app_page):
    at = run_app_page("Home.py")
    assert not at.exception, at.exception
    assert any("Mock data" in w.value for w in at.sidebar.warning)
    assert len(at.metric) >= 7
    assert any("Mock segment" in m.value for m in at.markdown)  # names come from metrics


def test_guided_tour_toggle_shows_callouts(run_app_page):
    at = run_app_page("Home.py")
    assert not at.info
    at.sidebar.toggle(key="tour_toggle").set_value(True).run()
    assert not at.exception
    assert any("Tour 1" in i.value for i in at.info)


@pytest.mark.parametrize(
    "page",
    [
        "pages/1_Data_and_Cube.py",
        "pages/2_Segments_and_Stats.py",
        "pages/3_Predictions.py",
        "pages/4_Customer_360.py",
    ],
)
def test_every_page_runs_on_mock(run_app_page, page):
    at = run_app_page(page)
    assert not at.exception, at.exception


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
