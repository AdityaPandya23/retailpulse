"""Recommendation (question C): support / confidence / lift on the strongest real rule, the top
rules, every rule as a bubble, a product picker and the hit-rate vs best sellers."""

from __future__ import annotations

import streamlit as st
from components import data, layout, tour

from retailpulse import contracts as C
from retailpulse.config import RULES
from retailpulse.viz import rules as VR

layout.setup_page("Recommendation")
layout.page_header(
    "Recommendation",
    "C · What should we offer them?",
    "Association rules find products that are bought together in the same basket. A customer "
    "who has the first product is offered the second.",
)

rm = data.metrics("rules")
rules = data.table(C.RULES_PARQUET)
products = data.table(C.PRODUCTS_PARQUET)
n_inv = rm.get("n_invoices")
n_inv = n_inv.get("overall") if isinstance(n_inv, dict) else n_inv
top = rm.get("top_rule") or {}
n_rec = int(rm.get("n_recommend") or RULES.n_recommend)

PERCENT = st.column_config.NumberColumn(format="percent")
RULE_COLUMNS = {
    "Support": st.column_config.NumberColumn(format="%.2f%%"),
    "Confidence": PERCENT,
    "Lift": st.column_config.NumberColumn(format="%.1f"),
}


def _rule_frame(df):
    """Support shown as a percentage with 2 decimals (support values are small)."""
    return df.assign(Support=100 * df["Support"])


# ------------------------------------------------------------------ the three measures
st.subheader("1 · Three numbers behind every rule")
if top:
    st.markdown(f"The strongest rule: **{top.get('antecedents')} → {top.get('consequents')}**")
tour.callout(
    1,
    "Read the three boxes left to right: how common the pair is, how often the rule is right, "
    "and how much stronger than chance it is.",
    "Support, confidence, lift",
)
explained = VR.rule_explainer(top, n_inv)
if explained:
    for col, key in zip(st.columns(3), ("support", "confidence", "lift"), strict=True):
        with col, st.container(border=True):
            st.markdown(explained[key])
else:
    st.info("No rule recorded.")

# ------------------------------------------------------------------ top rules
st.subheader("2 · The strongest rules")
col_table, col_bubble = st.columns([1, 1])
with col_table:
    st.dataframe(
        _rule_frame(VR.top_rules_table(rules, 10)),
        hide_index=True,
        width="stretch",
        column_config=RULE_COLUMNS,
        key="rec_top_rules",
    )
    st.caption(
        f"Rules need confidence ≥ {rm.get('min_confidence', 0):.0%} and lift > 1; sorted by lift."
    )
with col_bubble:
    tour.callout(
        2,
        "Each bubble is one rule. Rare pairs (left) can still be very reliable (top); big dark "
        "bubbles have the highest lift.",
        "Every rule",
    )
    layout.chart(VR.rules_bubble(rules), key="rec_bubble")

# ------------------------------------------------------------------ product picker
st.subheader("3 · Pick a product: what goes with it?")
starts = VR.antecedent_products(rules, products)
if starts.empty:
    st.info("No rules available.")
else:
    codes = starts["stock_code"].tolist()
    label = dict(zip(codes, starts["description"], strict=True))
    first = (top.get("antecedent_codes") or "").split(",")[0]
    code = st.selectbox(
        f"Products that start a rule ({len(codes)})",
        codes,
        index=codes.index(first) if first in codes else 0,
        format_func=lambda c: f"{label.get(c, c)} ({c})",
        key="rec_product",
    )
    st.dataframe(
        _rule_frame(VR.rules_for_product(rules, code)),
        hide_index=True,
        width="stretch",
        column_config=RULE_COLUMNS,
        key="rec_product_rules",
    )

# ------------------------------------------------------------------ hit-rate
st.subheader("4 · Do the suggestions work?")
col_hit, col_text = st.columns([3, 2])
with col_hit:
    tour.callout(
        3,
        f"We gave every customer {n_rec} suggestions from data before the cutoff, then checked "
        "what "
        "they actually bought next. Rules beat simply offering the best sellers.",
        "Hit-rate",
    )
    layout.chart(VR.hit_rate_bar(rm, n_rec), key="rec_hit")
with col_text:
    st.markdown(
        f"**Hit-rate@{n_rec}** = share of customers who bought something new after the cutoff and "
        f"found at least one of it among their {n_rec} suggestions."
    )
    layout.conclusion(rm.get("conclusion"))

st.divider()
st.page_link("Home.py", label="← Back to the live demo", icon=":material/home:")
