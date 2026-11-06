"""RetailPulse home: headline numbers and the three questions. Run: streamlit run app/Home.py"""

from __future__ import annotations

import streamlit as st
from components import data, layout, tour

from retailpulse import contracts as C
from retailpulse.viz.common import money, pct

layout.setup_page("Home")
layout.page_header(
    "RetailPulse",
    "Who are our customers, who will come back, and what do they buy together?",
    "Customer analytics on two years of a UK online gift retailer's sales "
    "(Dec 2009 – Dec 2011): segment → predict → recommend.",
)

wh = data.json_file(C.WAREHOUSE_SUMMARY_JSON)
feats = data.metrics("features")
seg = data.metrics("segment")
cls = data.metrics("classify")
bt = data.metrics("backtest")
rules = data.metrics("rules")
horizon = feats.get("horizon_days", 90)
names = data.segment_names()

best = cls.get("best_model")
best_auc = (cls.get("models", {}).get(best) or {}).get("test_auc") if best else None
test_aucs = [m.get("test_auc") for m in cls.get("models", {}).values() if m.get("test_auc")]
hit, pop_hit = rules.get("hit_rate_at_5"), rules.get("popularity_hit_rate_at_5")

tour.callout(
    1,
    "Each tile is one headline result; the pages behind the three questions below show how "
    "it was reached. Numbers come straight from the pipeline's metrics.json.",
    "Start here",
)
layout.kpi_row(
    [
        {
            "label": "Revenue (cleaned sales)",
            "value": money(wh.get("total_revenue", float("nan")), 2),
            "help": "Sum of quantity × price over valid sales lines of registered customers.",
        },
        {
            "label": "Customers",
            "value": f"{(wh.get('row_counts') or {}).get('dim_customer', 0):,}",
            "help": (
                f"{feats.get('n_customers_cutoff', 0):,} bought before the "
                f"{feats.get('cutoff', 'cutoff')} cutoff and are modelled."
            ),
        },
        {
            "label": "Segments (K-Means)",
            "value": f"k = {seg.get('k', len(names))}",
            "help": seg.get("k_choice_reason"),
        },
        {
            "label": "Best model · test ROC-AUC",
            "value": f"{best_auc:.3f}" if best_auc else "n/a",
            "help": (
                f"{best} was chosen on train cross-validation only; "
                f"all {len(test_aucs)} models score "
                f"{min(test_aucs):.2f}–{max(test_aucs):.2f} on the future window."
                if test_aucs
                else None
            ),
        },
    ]
)
layout.kpi_row(
    [
        {
            "label": "Backtest: predicted to return",
            "value": f"{bt.get('n_predicted_return', 0):,}",
            "help": bt.get("conclusion"),
        },
        {
            "label": "… of whom actually came back",
            "value": pct(bt.get("share_actually_returned", float("nan"))),
            "delta": f"vs {pct(bt.get('base_rate', float('nan')))} of all customers",
            "help": (
                f"Precision of the 'will return' prediction in the {horizon} days after the cutoff."
            ),
        },
        {
            "label": "Recommendations hit-rate @5",
            "value": pct(hit if hit is not None else float("nan"), 1),
            "delta": (f"vs {pct(pop_hit, 1)} for best sellers" if pop_hit is not None else None),
            "help": (
                "Share of customers who bought something new after the cutoff and found at "
                "least one of it among their 5 recommendations."
            ),
        },
    ]
)
if names:
    st.markdown(
        "**Segments:** "
        + " · ".join(f"{i} **{n}**" for i, n in names.items())
        + "  \n"
        + "Segment ids are ranked by revenue (0 = most valuable); names come from the pipeline."
    )

st.divider()
st.subheader("Three questions")
tour.callout(
    2,
    "The demo follows A → B → C. Turn on this tour on every page for numbered notes.",
    "Demo route",
)
qa, qb, qc = st.columns(3)
with qa, st.container(border=True):
    st.markdown("#### A · Who are our customers?")
    st.markdown(
        "Data quality and the OLAP cube, then K-Means segments vs an RFM-quartile baseline, "
        "validated with PCA, stability and hypothesis tests."
    )
    st.page_link("pages/1_Data_and_Cube.py", label="Data & Cube", icon=":material/database:")
    st.page_link(
        "pages/2_Segments_and_Stats.py", label="Segments & Stats", icon=":material/scatter_plot:"
    )
with qb, st.container(border=True):
    st.markdown("#### B · Who will come back?")
    st.markdown(
        f"{len(C.CLASSIFIERS)} classifiers predict 'buys again within {horizon} days' "
        "on an out-of-time split; "
        "a Time Machine backtest compares the predictions with what really happened."
    )
    st.page_link("pages/3_Predictions.py", label="Predictions", icon=":material/insights:")
with qc, st.container(border=True):
    st.markdown("#### C · What do they buy together?")
    st.markdown(
        "Association rules per segment drive 5 product recommendations per customer; "
        "Customer 360 shows one customer's story and what-if scenarios."
    )
    st.page_link("pages/4_Customer_360.py", label="Customer 360", icon=":material/person_search:")

st.divider()
layout.conclusion(
    [seg.get("conclusion"), cls.get("conclusion"), bt.get("conclusion"), rules.get("conclusion")],
    "Headline conclusions",
)
st.caption(layout.DATA_CREDIT + " Built with free, open-source tools only.")
