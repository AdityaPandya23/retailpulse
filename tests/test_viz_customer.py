"""Page 4 helpers / figures on the MOCK artifacts (artifacts/mock/) and tiny hand-made frames."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from retailpulse import contracts as C
from retailpulse.viz import customer as VCu
from retailpulse.viz import palette as P


@pytest.fixture(scope="module")
def mock():
    if not C.all_exist(list(C.MOCK_OUTPUTS)):
        pytest.skip("artifacts/mock/ not generated")

    def read(p):
        return pd.read_parquet(C.mock_path(p))

    return {
        "features": read(C.FEATURES_CUTOFF_PARQUET),
        "segments": read(C.SEGMENTS_PARQUET),
        "labels": read(C.LABELS_PARQUET),
        "classification": read(C.CLASSIFICATION_PARQUET),
        "recs": read(C.RECOMMENDATIONS_PARQUET),
        "reasons": read(C.REC_REASONS_PARQUET),
        "timeline": read(C.TIMELINE_PARQUET),
        "seg_model": joblib.load(C.mock_path(C.SEGMENT_MODEL_JOBLIB)),
        "clf": joblib.load(C.mock_path(C.CLASSIFIER_JOBLIB)),
        "reg": joblib.load(C.mock_path(C.REGRESSOR_JOBLIB)),
    }


# ---------------- lookup ----------------
def test_customer_ids_and_labels(mock):
    ids = VCu.customer_ids(mock["features"], mock["segments"])
    assert ids == sorted(mock["features"]["customer_id"].tolist())
    names = {0: "A", 1: "B", 2: "C", 3: "D"}
    labels = VCu.customer_labels(ids, mock["segments"], names)
    seg = mock["segments"].set_index("customer_id")["segment"]
    assert labels[ids[0]] == f"{ids[0]} · {names[int(seg[ids[0]])]}"
    assert VCu.customer_labels([1], None, names) == {1: "1"}
    assert VCu.customer_ids(None, None) == []


def test_default_customer_is_a_deterministic_segment_0_member(mock):
    cid = VCu.default_customer(mock["features"], mock["segments"])
    seg0 = set(mock["segments"].query("segment == 0")["customer_id"])
    assert cid in seg0
    assert VCu.default_customer(mock["features"], mock["segments"]) == cid
    # a preference that intersects the segment wins
    other = sorted(seg0 - {cid})[0]
    assert VCu.default_customer(mock["features"], mock["segments"], prefer_ids=[[other]]) == other
    # a preference with no segment-0 member is ignored
    assert VCu.default_customer(mock["features"], mock["segments"], prefer_ids=[[-1]]) == cid
    assert VCu.default_customer(None, None) is None
    assert VCu.default_customer(mock["features"], mock["segments"], segment_id=99) == min(
        mock["features"]["customer_id"]
    )


def test_unknown_customer_is_handled(mock):
    assert VCu.row_of(mock["features"], -5) is None
    assert VCu.row_of(None, 1) is None
    assert VCu.rows_of(mock["timeline"], -5).empty
    assert VCu.test_prediction(mock["classification"], -5, "logistic") is None
    assert VCu.test_prediction(None, 1, "logistic") is None
    assert VCu.recommendation_table(mock["recs"], mock["reasons"], -5).empty
    assert not VCu.timeline_figure(VCu.rows_of(mock["timeline"], -5), "2011-09-09", 90).data


def test_test_prediction_for_a_test_customer(mock):
    c = mock["classification"]
    row = c[(c["split"] == "test") & (c["model"] == "lda")].iloc[0]
    got = VCu.test_prediction(c, int(row["customer_id"]), "lda")
    assert got == {
        "p_return": pytest.approx(row["p_return"]),
        "pred_return": int(row["pred_return"]),
        "returned_flag": int(row["returned_flag"]),
    }


# ---------------- models / What-if ----------------
def test_apply_models_on_mock_payloads(mock):
    cols = VCu.model_features(mock["seg_model"], mock["clf"], mock["reg"])
    assert cols == [c for c in C.FEATURE_COLUMNS if c in cols]
    raw = mock["features"][cols]
    out = VCu.apply_models(raw, mock["seg_model"], mock["clf"], mock["reg"])
    assert len(out) == len(raw)
    assert set(out["segment"]) <= set(mock["seg_model"]["label_to_segment"].values())
    assert out["p_return"].between(0, 1).all()
    assert (out["spend_if_return"] >= 0).all()
    assert np.allclose(out["expected_spend"], out["p_return"] * out["spend_if_return"])


def test_expected_spend_by_hand():
    assert VCu.expected_spend(0.25, 400.0) == pytest.approx(100.0)
    assert np.allclose(VCu.expected_spend([0.5, 0.0], [10.0, 99.0]), [5.0, 0.0])


def test_spend_is_expm1_of_regressor_prediction(mock):
    raw = mock["features"][mock["reg"]["features"]].head(3)
    pred = mock["reg"]["pipeline"].predict(raw)
    assert np.allclose(
        VCu.predict_spend_if_return(mock["reg"], raw), np.clip(np.expm1(pred), 0, None)
    )


def test_missing_models_give_nan_not_errors(mock):
    raw = mock["features"].head(2)
    out = VCu.apply_models(raw, None, None, None)
    assert (out["segment"] == -1).all() and out["p_return"].isna().all()


def test_slider_specs_ranges_and_types(mock):
    f = mock["features"]
    row = f.iloc[0]
    specs = VCu.slider_specs(f, row, list(C.FEATURE_COLUMNS) + ["not_a_column"])
    assert [s["feature"] for s in specs] == list(C.FEATURE_COLUMNS)
    for s in specs:
        assert s["min"] <= s["value"] <= s["max"]
        if s["is_int"]:
            assert all(isinstance(s[k], int) for k in ("min", "max", "value", "step"))
        if s["feature"] in VCu.SHARE_FEATURES:
            assert (s["min"], s["max"]) == (0.0, 1.0)
    # a customer far above the 99th percentile still fits on the slider
    big = row.copy()
    big["monetary"] = f["monetary"].max() * 10
    spec = next(s for s in VCu.slider_specs(f, big, ["monetary"]))
    assert spec["max"] >= big["monetary"] and spec["value"] == pytest.approx(big["monetary"])


def test_raw_frame_overrides_only_given_values(mock):
    row = mock["features"].iloc[0]
    cols = ["recency", "frequency", "monetary"]
    raw = VCu.raw_frame({"recency": 3}, row, cols)
    assert list(raw.columns) == cols and raw.loc[0, "recency"] == 3
    assert raw.loc[0, "monetary"] == row["monetary"]


def test_what_if_changes_the_prediction(mock):
    cols = VCu.model_features(mock["seg_model"], mock["clf"], mock["reg"])
    row = mock["features"].iloc[0]
    base = VCu.apply_models(
        VCu.raw_frame({}, row, cols), mock["seg_model"], mock["clf"], mock["reg"]
    )
    new = VCu.apply_models(
        VCu.raw_frame({"recency": 0, "frequency": 50}, row, cols),
        mock["seg_model"],
        mock["clf"],
        mock["reg"],
    )
    assert 0 <= new["p_return"].iloc[0] <= 1
    assert new["p_return"].iloc[0] != pytest.approx(base["p_return"].iloc[0])


# ---------------- figures / display ----------------
def test_timeline_before_after_and_cutoff(mock):
    tl = mock["timeline"]
    cid = tl.loc[tl["period"] == "after", "customer_id"].iloc[0]
    fig = VCu.timeline_figure(VCu.rows_of(tl, cid), "2011-09-09", 90)
    names = [d.name for d in fig.data]
    assert len(fig.data) == 2 and names[0].startswith("before") and names[1].startswith("after")
    assert fig.data[0].marker.color == P.PERIOD_COLOURS["before"]
    assert fig.data[1].marker.color == P.PERIOD_COLOURS["after"]
    assert any("cutoff 2011-09-09" in a.text for a in fig.layout.annotations)
    assert len(fig.layout.shapes) == 2  # outcome window + cutoff line
    only_before = tl[(tl["customer_id"] == cid) & (tl["period"] == "before")]
    assert len(VCu.timeline_figure(only_before, None).data) == 1


def test_timeline_log_axis_when_skewed():
    tl = pd.DataFrame(
        {
            "customer_id": 1,
            "month": pd.date_range("2010-01-01", periods=8, freq="MS"),
            "revenue": [10.0, 12, 11, 9, 13, 10, 12, 5000],
            "orders": 1,
            "period": "before",
        }
    )
    assert VCu.timeline_figure(tl, "2011-01-01", 90).layout.yaxis.type == "log"
    tl["revenue"] = 10.0
    assert VCu.timeline_figure(tl, "2011-01-01", 90).layout.yaxis.type != "log"


def test_gauge_has_threshold(mock):
    fig = VCu.gauge_figure(0.42, 0.37, "knn")
    ind = fig.data[0]
    assert isinstance(ind, go.Indicator) and ind.value == pytest.approx(42)
    assert ind.gauge.threshold.value == pytest.approx(37)
    assert ind.gauge.bar.color == P.CLASSIFIER_COLOURS["knn"]
    assert VCu.gauge_figure(0.5, None, "lda").data[0].gauge.threshold.value is None
    assert not VCu.gauge_figure(None, 0.5, "lda").data


def test_segment_badge_colour_and_escaped_name():
    badge = VCu.segment_badge_html(0, "Champs <b>")
    assert P.segment_colour(0) in badge and "Champs &lt;b&gt;" in badge and "segment 0" in badge
    assert P.INK in VCu.segment_badge_html(6, "Yellow")  # light yellow gets an ink ring
    assert "unknown" in VCu.segment_badge_html(None, None)


def test_key_features_formatting(mock):
    items = VCu.key_features(mock["features"].iloc[0], ["monetary", "recency", "return_rate"])
    assert [i["label"] for i in items] == ["Monetary", "Recency", "Return rate"]
    assert items[0]["value"].startswith("£") and items[1]["value"].endswith("days")
    assert items[2]["value"].endswith("%")
    assert VCu.key_features(None) == []


# ---------------- recommendations ----------------
def test_recommendations_with_rule_reason(mock):
    reasons = mock["reasons"]
    cid = int(reasons["customer_id"].iloc[0])
    t = VCu.recommendation_table(mock["recs"], reasons, cid)
    assert 1 <= len(t) <= 5 and list(t["Rank"]) == sorted(t["Rank"])
    code = str(reasons["stock_code"].iloc[0])
    row = t[t["Stock code"] == code].iloc[0]
    assert row["Why"].startswith("because you bought") and "→" in row["Why"]
    assert row["Confidence"] == pytest.approx(reasons["confidence"].iloc[0])


def test_recommendations_fewer_than_five_none_and_no_reasons(mock):
    recs = mock["recs"]
    counts = recs.groupby("customer_id").size()
    few = int(counts[counts < 5].index[0])
    assert len(VCu.recommendation_table(recs, mock["reasons"], few)) == counts[few]
    none = sorted(set(mock["features"]["customer_id"]) - set(recs["customer_id"]))[0]
    assert VCu.recommendation_table(recs, mock["reasons"], none).empty
    # no reasons table and no "source" column (optional) -> still a table
    cid = int(counts.index[0])
    t = VCu.recommendation_table(recs.drop(columns="source"), None, cid)
    assert len(t) == counts[cid] and t["Confidence"].isna().all()


def test_sliders_need_fill_on_new_customer_or_lost_widget_state():
    """Regression (P4 review): refill after a page switch drops the slider keys."""
    keys = ["c360_whatif_recency", "c360_whatif_frequency"]
    full = dict.fromkeys(keys, 1)
    assert not VCu.sliders_need_fill(full, keys, 7, 7)
    assert VCu.sliders_need_fill(full, keys, 7, 8)  # other customer
    assert VCu.sliders_need_fill({"c360_whatif_recency": 1}, keys, 7, 7)  # state dropped


def test_figure_labels_follow_configured_horizon(monkeypatch):
    """Regression (P4 review): labels used a fixed '90 days' instead of config horizon_days."""
    from retailpulse.viz import predictions as VP

    monkeypatch.setattr(VCu, "HORIZON_DAYS", 60)
    monkeypatch.setattr(VP, "HORIZON_DAYS", 60)
    assert "within 60 days" in VCu.gauge_figure(0.4, 0.37, "knn").layout.title.text
    test = pd.DataFrame({"returned_flag": [0, 1, 1], "pred_return": [0, 1, 0]})
    fig = VP.confusion_figure(test, "knn", 0.37)
    assert fig.layout.yaxis.title.text == "Actual outcome in the 60 days"
