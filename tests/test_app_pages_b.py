"""AppTest smoke + interaction tests for pages 3 (Predictions) and 4 (Customer 360) on MOCK data."""

from __future__ import annotations

import pandas as pd

from retailpulse import contracts as C

PRED = "pages/3_Predictions.py"
C360 = "pages/4_Customer_360.py"


def _text(at) -> str:
    parts = [h.value for h in at.subheader] + [m.value for m in at.markdown]
    parts += [m.label for m in at.metric] + [c.value for c in at.caption]
    return "\n".join(str(p) for p in parts)


# ---------------- Predictions ----------------
def test_predictions_runs_and_hides_outcome_until_reveal(run_app_page):
    at = run_app_page(PRED)
    assert not at.exception, at.exception
    assert "We are at the cutoff" in _text(at)
    assert "actually bought again" not in _text(at)  # outcome hidden before the reveal
    at.button(key="tm_reveal").click().run()
    assert not at.exception, at.exception
    text = _text(at)
    assert "actually bought again" in text and "Base rate" in text and "Lift" in text
    at.button(key="tm_hide").click().run()
    assert not at.exception and "actually bought again" not in _text(at)


def test_predictions_model_selectbox_and_knn_slider(run_app_page):
    at = run_app_page(PRED)
    assert (
        at.selectbox(key="cm_model").value
        == C.read_metrics(C.mock_path(C.METRICS_JSON))["classify"]["best_model"]
    )
    at.selectbox(key="cm_model").set_value("qda").run()
    assert not at.exception, at.exception
    assert any("**qda** flags" in m.value for m in at.markdown)
    at.slider(key="knn_k").set_value(7).run()
    assert not at.exception, at.exception
    assert any("At **k = 7**" in m.value for m in at.markdown)
    at.selectbox(key="reg_model").set_value("simple_lr").run()
    assert not at.exception, at.exception


# ---------------- Customer 360 ----------------
def test_customer_360_default_is_segment_0(run_app_page):
    at = run_app_page(C360)
    assert not at.exception, at.exception
    seg = pd.read_parquet(C.mock_path(C.SEGMENTS_PARQUET))
    cid = at.selectbox(key="c360_customer").value
    assert int(seg.loc[seg["customer_id"] == cid, "segment"].iloc[0]) == 0
    assert any("Expected 90-day spend" in m.label for m in at.metric)
    assert any(m.label.startswith("P(buys again)") for m in at.metric)


def test_customer_360_change_customer_and_what_if(run_app_page):
    at = run_app_page(C360)
    feats = pd.read_parquet(C.mock_path(C.FEATURES_CUTOFF_PARQUET))
    current = at.selectbox(key="c360_customer").value
    other_id = int(next(c for c in sorted(feats["customer_id"]) if c != current))
    at.selectbox(key="c360_customer").set_value(other_id).run()
    assert not at.exception, at.exception
    assert at.selectbox(key="c360_customer").value == other_id
    # sliders are refilled with the new customer's values
    expected = int(feats.loc[feats["customer_id"] == other_id, "recency"].iloc[0])
    assert at.slider(key="c360_whatif_recency").value == expected

    at.slider(key="c360_whatif_recency").set_value(0).run()
    assert not at.exception, at.exception
    p = next(m for m in at.metric if m.label.startswith("P(buys again)"))
    assert p.value.endswith("%") and "vs real profile" in p.delta
    assert any("what-if" in m.value for m in at.markdown)  # old -> new segment badge
    at.button(key="c360_whatif_reset").click().run()
    assert not at.exception and at.slider(key="c360_whatif_recency").value == expected


def test_customer_360_without_recommendations(run_app_page):
    recs = pd.read_parquet(C.mock_path(C.RECOMMENDATIONS_PARQUET))
    feats = pd.read_parquet(C.mock_path(C.FEATURES_CUTOFF_PARQUET))
    none = sorted(set(feats["customer_id"]) - set(recs["customer_id"]))[0]
    at = run_app_page(C360)
    at.selectbox(key="c360_customer").set_value(int(none)).run()
    assert not at.exception, at.exception
    assert any("No recommendations" in i.value for i in at.info)
