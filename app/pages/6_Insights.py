"""Insights: one row per segment (size, revenue, came back, predicted P(return), top recommended
products, suggested action) and the project's limitations (same wording as README.md)."""

from __future__ import annotations

import streamlit as st
from components import data, layout, tour

from retailpulse import contracts as C
from retailpulse import insights as I
from retailpulse.viz import predictions as VP

layout.setup_page("Insights")
horizon = data.metrics("features").get("horizon_days", 90)
layout.page_header(
    "Insights",
    "So what should the business do?",
    "The three answers combined per segment, with one suggested action each.",
)

classification = data.table(C.CLASSIFICATION_PARQUET)
best = VP.best_model(data.metrics("classify"), classification)
table = I.segment_insights(
    data.table(C.SEGMENT_PROFILES_PARQUET),
    data.segment_names(),
    data.table(C.SEGMENTS_PARQUET),
    classification,
    best,
    data.table(C.RECOMMENDATIONS_PARQUET),
)

st.subheader("One row per segment")
tour.callout(
    1,
    "Read a row left to right: how big the segment is, how much it earns, how many came back, "
    "what the model expected, what to offer, and the action.",
    "Per segment",
)
if table.empty:
    st.info("Segment profiles are not available yet.")
else:
    st.dataframe(
        table.drop(columns="segment"),
        hide_index=True,
        width="stretch",
        column_config={
            "Customers %": st.column_config.NumberColumn(format="percent"),
            "Revenue %": st.column_config.NumberColumn(format="percent"),
            "Came back %": st.column_config.NumberColumn(
                format="percent", help=f"Bought again in the {horizon} days after the cutoff."
            ),
            "Avg P(return)": st.column_config.NumberColumn(
                format="percent",
                help=f"Mean predicted probability of the chosen model ({best}) on the test split.",
            ),
            "Top 3 recommended products": st.column_config.TextColumn(width="large"),
        },
        key="insights_table",
    )
with st.expander("How the action is chosen"):
    st.markdown(
        "Checked in this order, first match wins (one action per segment):\n"
        f"1. highest mean recency → **{I.WIN_BACK}**\n"
        f"2. highest mean spend (monetary) → **{I.REWARD}**\n"
        f"3. lowest mean tenure → **{I.SECOND_ORDER}**\n"
        f"4. otherwise → **{I.KEEP_ENGAGED}**"
    )

st.subheader("Limitations")
limits = data.limitations()
st.markdown("\n".join(f"- {item}" for item in limits) if limits else "See README.md.")

st.divider()
st.page_link("Home.py", label="← Back to the live demo", icon=":material/home:")
