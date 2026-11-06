"""AppTest smoke + interaction tests for pages 1-2 on the MOCK artifacts (never real data)."""

from __future__ import annotations

PAGE_1 = "pages/1_Data_and_Cube.py"
PAGE_2 = "pages/2_Segments_and_Stats.py"


def _texts(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [m.label + " " + str(m.value) for m in at.metric]
    return "\n".join(parts)


# ------------------------------------------------------------------ page 1
def test_data_and_cube_renders(run_app_page):
    at = run_app_page(PAGE_1)
    assert not at.exception, at.exception
    assert len(at.tabs) == 4
    text = _texts(at)
    assert "Gini of customer revenue" in text and "Verdict" in text
    assert "How CUBE works" in [e.label for e in at.expander][0]


def test_cube_slice_and_dice(run_app_page):
    at = run_app_page(PAGE_1)
    countries = at.multiselect(key="cube_countries")
    options = countries.options
    assert len(countries.value) >= 1

    # one country -> exact distinct customers from the country grouping set
    at.multiselect(key="cube_countries").set_value([options[0]]).run()
    assert not at.exception, at.exception
    cust = next(m for m in at.metric if m.label == "Distinct customers")
    assert not cust.value.startswith("≤")

    # several countries -> customers are not additive: shown as an upper bound
    at.multiselect(key="cube_countries").set_value(options[:3]).run()
    cust = next(m for m in at.metric if m.label == "Distinct customers")
    assert cust.value.startswith("≤")

    # slice a year, size by orders, then all countries (empty selection)
    years = [y for y in at.selectbox(key="cube_year").options if y != "All years"]
    at.selectbox(key="cube_year").set_value(int(years[-1])).run()
    at.radio(key="cube_measure").set_value("orders").run()
    assert not at.exception, at.exception
    at.multiselect(key="cube_countries").set_value([]).run()
    assert not at.exception, at.exception
    cust = next(m for m in at.metric if m.label == "Distinct customers")
    assert not cust.value.startswith("≤")
    assert any(m.label.startswith("Revenue · all countries") for m in at.metric)


def test_data_and_cube_tour_callouts(run_app_page):
    at = run_app_page(PAGE_1)
    at.sidebar.toggle(key="tour_toggle").set_value(True).run()
    assert not at.exception
    tours = [i.value for i in at.info if "Tour" in i.value]
    assert [t.split("**")[1].split(" ·")[0] for t in tours] == [
        "Tour 1",
        "Tour 2",
        "Tour 3",
        "Tour 4",
    ]


# ------------------------------------------------------------------ page 2
def test_segments_and_stats_renders(run_app_page):
    at = run_app_page(PAGE_2)
    assert not at.exception, at.exception
    assert len(at.tabs) == 5
    text = _texts(at)
    assert "Mock segment A" in text  # names from the data, never hard-coded
    assert "config.yaml" in text  # human override note
    assert "Chosen: k = 4" in text  # mock k


def test_segment_box_and_stats_selectors(run_app_page):
    at = run_app_page(PAGE_2)
    for feature in ("recency", "return_rate", "frequency"):
        at.selectbox(key="seg_box_feature").set_value(feature).run()
        assert not at.exception, at.exception
    for var in at.selectbox(key="qq_variable").options:
        at.selectbox(key="qq_variable").set_value(var).run()
        assert not at.exception, at.exception
    at.selectbox(key="hist_feature").set_value("tenure").run()
    assert not at.exception, at.exception


def test_live_test_picker(run_app_page):
    at = run_app_page(PAGE_2)
    assert at.selectbox(key="live_seg_a").value == 0
    assert at.selectbox(key="live_seg_b").value == 1
    assert any("Mann-Whitney" in m.value for m in at.markdown)

    at.selectbox(key="live_seg_b").set_value(3).run()
    at.selectbox(key="live_feature").set_value("frequency").run()
    assert not at.exception, at.exception
    assert any("order count" in m.value for m in at.markdown)
    assert len(at.dataframe) >= 1

    # identical segments: a friendly message, no test table, no crash
    at.selectbox(key="live_seg_b").set_value(0).run()
    assert not at.exception, at.exception
    assert any("two different segments" in m.value for m in at.markdown)


def test_segments_tour_callouts(run_app_page):
    at = run_app_page(PAGE_2)
    at.sidebar.toggle(key="tour_toggle").set_value(True).run()
    assert not at.exception
    assert sum("Tour" in i.value for i in at.info) == 4


def test_pages_survive_null_metrics(run_app_page, tmp_path, monkeypatch):
    """Regression (P4 review): NaN is stored as null in metrics.json; chi-square / Gini tiles
    must show "n/a" instead of crashing on format(None)."""
    import json

    from retailpulse import contracts as C

    mock = C.read_metrics(C.mock_path(C.METRICS_JSON))
    seg, eda = dict(mock["segment"]), dict(mock["eda"])
    seg["chi2_segment_uk"] = {
        **seg["chi2_segment_uk"],
        "chi2": None,
        "p_value": None,
        "cramers_v": None,
    }
    eda["gini"] = None
    (tmp_path / "metrics.json").write_text(json.dumps({"segment": seg, "eda": eda}))
    monkeypatch.setenv("RETAILPULSE_ARTIFACTS", str(tmp_path))  # files fall back to mock
    for page, label in ((PAGE_2, "Cramér's V"), (PAGE_1, "Gini of customer revenue")):
        at = run_app_page(page)
        assert not at.exception, at.exception
        assert next(m for m in at.metric if m.label == label).value == "n/a"
