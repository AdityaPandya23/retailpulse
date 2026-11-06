"""Shared viz helpers: fixed segment colours from metrics names, log axes, template."""

from __future__ import annotations

import plotly.io as pio

from retailpulse import contracts as C
from retailpulse.viz import common as VC
from retailpulse.viz import palette as P


def test_segment_names_from_json_string_keys():
    seg = {"segment_names": {"1": "New", "0": "Champions", "2": "Lost"}}
    assert VC.segment_names(seg) == {0: "Champions", 1: "New", 2: "Lost"}
    assert VC.segment_order(VC.segment_names(seg)) == ["Champions", "New", "Lost"]


def test_segment_colours_follow_the_id_not_the_rank():
    cmap = VC.segment_colour_map({2: "Lost", 0: "Champions"})
    assert cmap == {"Champions": "#0072B2", "Lost": "#009E73"}
    assert P.segment_colour(6) == "#F0E442" and P.segment_colour(7) == P.OTHER_COLOUR


def test_segment_names_fallback_to_profiles():
    import pandas as pd

    prof = pd.DataFrame({"segment": [1, 0], "segment_name": ["B", "A"]})
    assert VC.segment_names({}, prof) == {0: "A", 1: "B"}


def test_model_colours_and_dashes_are_unique():
    assert set(P.CLASSIFIER_COLOURS) == set(C.CLASSIFIERS)
    assert len(set(P.CLASSIFIER_COLOURS.values())) == len(C.CLASSIFIERS)
    assert len(set(P.CLASSIFIER_DASH.values())) == len(C.CLASSIFIERS)
    assert set(P.REGRESSOR_COLOURS) == set(C.REGRESSORS)


def test_is_skewed_and_log_axis():
    assert VC.is_skewed([1, 1, 2, 2, 3, 5, 100, 1000])
    assert not VC.is_skewed([0, 1, 2, 3])  # zeros cannot go on a log axis
    fig = VC.log_axis(VC.empty_figure("x"), "y", "Revenue (£)")
    assert fig.layout.yaxis.type == "log" and "log scale" in fig.layout.yaxis.title.text
    assert VC.log_axis(VC.empty_figure("x"), "y2").layout.yaxis2.type == "log"


def test_style_sets_the_template_on_the_figure():
    # Streamlit overrides plotly's global default, so every figure carries the template itself
    pio.templates.default = "plotly"
    fig = VC.style(VC.empty_figure("x"), "Title", "x (£)", "y (customers)")
    assert fig.layout.template.layout.font.family == P.FONT_FAMILY
    assert fig.layout.template.layout.plot_bgcolor == P.SURFACE
    assert fig.layout.title.text == "Title" and fig.layout.xaxis.title.text == "x (£)"
    assert VC.money(1234.5) == "£1,234" and VC.money(17_370_000) == "£17.4M"
    assert VC.pct(0.253, 1) == "25.3%"
