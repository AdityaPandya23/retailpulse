"""Customer 360: one customer's story (timeline, segment, return backtest, expected spend,
recommendations) and What-if sliders that re-run the saved models live.

All logic lives in retailpulse.viz.customer; this page lays it out.
"""

from __future__ import annotations

import streamlit as st
from components import data, layout, tour

from retailpulse import contracts as C
from retailpulse.viz import customer as VCu
from retailpulse.viz import predictions as VP
from retailpulse.viz.common import money, pct

CUSTOMER_KEY = "c360_customer"
WHATIF_FOR = "c360_whatif_for"  # customer the What-if sliders are currently filled for


def slider_key(feature: str) -> str:
    return f"c360_whatif_{feature}"


layout.setup_page("Customer 360")
layout.page_header(
    "Customer 360",
    "C · One customer: what they did, what we predicted, what to offer them",
    "Pick a customer to see their purchase history around the cutoff, their segment, the "
    "return prediction made at the cutoff next to what really happened, and product "
    "recommendations. The What-if sliders re-run the saved models live.",
)

feats_m = data.metrics("features")
cls = data.metrics("classify")
bt = data.metrics("backtest")
reg = data.metrics("regress")
names = data.segment_names()

features = data.table(C.FEATURES_CUTOFF_PARQUET)
segments = data.table(C.SEGMENTS_PARQUET)
labels = data.table(C.LABELS_PARQUET)
classification = data.table(C.CLASSIFICATION_PARQUET)
recs = data.table(C.RECOMMENDATIONS_PARQUET)
reasons = data.table(C.REC_REASONS_PARQUET)
timeline = data.table(C.TIMELINE_PARQUET)
seg_model = data.model(C.SEGMENT_MODEL_JOBLIB)
clf = data.model(C.CLASSIFIER_JOBLIB)
regr = data.model(C.REGRESSOR_JOBLIB)

best = VP.best_model(cls, classification)
cutoff = feats_m.get("cutoff") or cls.get("test_cutoff")
horizon = int(feats_m.get("horizon_days") or 90)


@st.cache_data(show_spinner=False)
def _options(features, segments, names):
    ids = VCu.customer_ids(features, segments)
    return ids, VCu.customer_labels(ids, segments, names)


@st.cache_data(show_spinner=False)
def _default(features, segments, classification, reasons, model):
    prefer = []
    test = VP.test_predictions(classification, model)
    if len(test):
        prefer.append(test["customer_id"].tolist())
    if reasons is not None and len(reasons):
        prefer.append(reasons["customer_id"].tolist())
    return VCu.default_customer(features, segments, segment_id=0, prefer_ids=prefer)


@st.cache_data(show_spinner=False)
def _specs(features, cid, columns):
    return VCu.slider_specs(features, VCu.row_of(features, cid), list(columns))


def _fill_sliders(specs: list[dict], cid: int) -> None:
    for s in specs:
        st.session_state[slider_key(s["feature"])] = s["value"]
    st.session_state[WHATIF_FOR] = cid


ids, option_labels = _options(features, segments, names)
if not ids:
    st.info("No customers available: run `python -m retailpulse.pipeline` first.")
    st.stop()

default_id = _default(features, segments, classification, reasons, best)
tour.callout(
    1,
    "Search any customer id. The default is a typical customer of the most valuable segment "
    "(closest to that segment's median profile).",
    "Pick a customer",
)
cid = st.selectbox(
    "Customer (type to search)",
    ids,
    index=ids.index(default_id) if default_id in ids else 0,
    format_func=lambda i: option_labels.get(i, str(i)),
    key=CUSTOMER_KEY,
    help="Customers who bought before the cutoff (the modelled population).",
)

feat_row = VCu.row_of(features, cid)
seg_row = VCu.row_of(segments, cid)
label_row = VCu.row_of(labels, cid)
seg_id = int(seg_row["segment"]) if seg_row is not None else None
seg_name = names.get(seg_id) if seg_id is not None else None
if seg_name is None and seg_row is not None and "segment_name" in seg_row:
    seg_name = str(seg_row["segment_name"])

# ------------------------------------------------------------------ profile
st.markdown(
    "**Segment:** " + VCu.segment_badge_html(seg_id, seg_name),
    unsafe_allow_html=True,
)
if feat_row is None:
    st.info("This customer has no feature row (bought nothing before the cutoff).")
else:
    cols = ["recency", "frequency", "monetary", "avg_basket_value", "tenure", "n_distinct_products"]
    layout.kpi_row(VCu.key_features(feat_row, [c for c in cols if c in feat_row]))
    st.caption(
        f"Features as of the cutoff {cutoff}; only purchases before it are used. "
        f"Return rate {VCu.format_feature('return_rate', feat_row.get('return_rate'))}, "
        f"non-UK share {VCu.format_feature('non_uk_share', feat_row.get('non_uk_share'))}."
    )

layout.chart(VCu.timeline_figure(VCu.rows_of(timeline, cid), cutoff, horizon), key="c360_timeline")

# ------------------------------------------------------------------ backtest for one customer
st.markdown("#### Will they come back? Prediction at the cutoff vs what happened")
pred = VCu.test_prediction(classification, cid, best)
raw_actual = (
    VCu.raw_frame({}, feat_row, VCu.model_features(seg_model, clf, regr))
    if feat_row is not None
    else None
)
live_actual = (
    VCu.apply_models(raw_actual, seg_model, clf, regr).iloc[0] if raw_actual is not None else None
)
if pred is not None:
    p_model, p_value = best, pred["p_return"]
    threshold = VP.model_threshold(cls, best, bt)
    source_note = f"Backtest prediction of {best} (best model) on the test split."
elif live_actual is not None and clf:
    p_model, p_value = clf.get("model"), float(live_actual["p_return"])
    threshold = clf.get("threshold")
    source_note = (
        "Not in the backtest table: live estimate of the saved classifier on the cutoff features."
    )
else:
    p_model, p_value, threshold, source_note = best, None, None, ""
if threshold is None and clf:
    threshold = clf.get("threshold")

g1, g2 = st.columns([3, 2])
with g1:
    tour.callout(
        2,
        "The needle is the probability the model gave at the cutoff; the black tick is the "
        "threshold. On the right: what this customer actually did afterwards.",
        "One-customer backtest",
    )
    layout.chart(VCu.gauge_figure(p_value, threshold, p_model), key="c360_gauge")
    if source_note:
        st.caption(source_note)
with g2:
    if p_value is not None and threshold is not None:
        verdict = "will buy again" if p_value >= threshold else "will not buy again"
        st.markdown(f"**Prediction:** {verdict} ({pct(p_value)} vs threshold {threshold:.2f})")
    if label_row is not None:
        came_back = int(label_row["returned_flag"]) == 1
        spend = float(label_row["next_spend"])
        st.markdown(
            f"**What happened in the {horizon} days after the cutoff:** "
            + (f"bought again, spending {money(spend, 2)}." if came_back else "did not buy again.")
        )
        if p_value is not None and threshold is not None:
            right = (p_value >= threshold) == came_back
            st.markdown(
                ":material/check_circle: The prediction was right."
                if right
                else ":material/cancel: The prediction was wrong for this customer."
            )
    else:
        st.info("No outcome recorded for this customer.")
    if live_actual is not None and p_value is not None:
        spend_if = float(live_actual["spend_if_return"])
        exp = float(VCu.expected_spend(p_value, spend_if))
        st.metric(
            f"Expected {horizon}-day spend",
            money(exp),
            help=reg.get("expected_spend_note"),
            border=True,
        )
        st.caption(
            f"= {pct(p_value)} chance of returning × {money(spend_if)} typical spend if they "
            f"return ({regr.get('model') if regr else 'regressor'}). Conservative: "
            "a typical-customer estimate, not the average."
        )

# ------------------------------------------------------------------ recommendations
st.markdown("#### Top-5 recommendations")
tour.callout(
    3,
    "Each product comes from an association rule that fired on this customer's own basket "
    "history (segment rules first), or from the segment's best sellers when no rule fires.",
    "Why this product?",
)
rec_table = VCu.recommendation_table(recs, reasons, cid)
if rec_table.empty:
    st.info("No recommendations for this customer.")
else:
    if len(rec_table) < 5:
        st.caption(f"Only {len(rec_table)} recommendation(s) available for this customer.")
    st.dataframe(
        rec_table,
        hide_index=True,
        width="stretch",
        column_config={
            "Score": st.column_config.NumberColumn(format="%.2f"),
            "Confidence": st.column_config.NumberColumn(format="percent"),
            "Lift": st.column_config.NumberColumn(format="%.1f"),
            "Why": st.column_config.TextColumn(width="large"),
        },
        key="c360_recs",
    )
    st.caption(
        "Confidence = share of baskets with the 'bought' items that also contain the product; "
        "lift = how many times more likely than for a random basket. Score ranks the list "
        "within a source."
    )

# ------------------------------------------------------------------ What-if
st.markdown("#### What if this customer behaved differently?")
columns = VCu.model_features(seg_model, clf, regr)
if feat_row is None or not columns or live_actual is None:
    st.info("What-if needs the customer's features and the saved models.")
    st.stop()

specs = _specs(features, cid, tuple(columns))
if st.session_state.get(WHATIF_FOR) != cid:
    _fill_sliders(specs, cid)

tour.callout(
    4,
    "Move a slider (e.g. recency or frequency): the saved K-Means and classifier re-score the "
    "edited profile live. Segment and probability update; deltas compare with the real profile.",
    "What-if",
)
values = {}
grid = st.columns(3)
for i, s in enumerate(specs):
    with grid[i % 3]:
        values[s["feature"]] = st.slider(
            s["label"],
            min_value=s["min"],
            max_value=s["max"],
            step=s["step"],
            key=slider_key(s["feature"]),
        )
st.button(
    "Reset to the customer's real values",
    key="c360_whatif_reset",
    icon=":material/restart_alt:",
    on_click=_fill_sliders,
    args=(specs, cid),
)

what = VCu.apply_models(VCu.raw_frame(values, feat_row, columns), seg_model, clf, regr).iloc[0]
new_seg = int(what["segment"])
d_spend = float(what["expected_spend"] - live_actual["expected_spend"])
old_seg = int(live_actual["segment"])
w1, w2, w3 = st.columns(3)
with w1, st.container(border=True):
    st.markdown("**Segment (live K-Means)**")
    st.markdown(
        VCu.segment_badge_html(old_seg, names.get(old_seg), "real")
        + "<br>→ "
        + VCu.segment_badge_html(new_seg, names.get(new_seg), "what-if"),
        unsafe_allow_html=True,
    )
    if new_seg != old_seg:
        st.caption("The edited profile falls in a different segment.")
with w2:
    st.metric(
        f"P(buys again) · {clf.get('model') if clf else 'n/a'}",
        pct(what["p_return"]),
        delta=f"{100 * (what['p_return'] - live_actual['p_return']):+.1f} pp vs real profile",
        delta_color="normal",
        border=True,
    )
with w3:
    st.metric(
        f"Expected {horizon}-day spend",
        money(what["expected_spend"]),
        delta=f"{'-' if d_spend < 0 else '+'}{money(abs(d_spend))} vs real profile",
        delta_color="normal",
        border=True,
    )
st.caption(
    "Sliders change the raw features independently (e.g. monetary without frequency), so some "
    "combinations are unrealistic; the models still score them. Ranges: dataset minimum to the "
    "99th percentile (or the customer's own value if higher)."
)
