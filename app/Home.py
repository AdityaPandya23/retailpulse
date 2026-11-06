"""RetailPulse v2 home = the LIVE DEMO (hub). Run: streamlit run app/Home.py

Describe one customer (a "Typical <segment>" preset or 6 typed numbers) and a basket; the
saved models answer the three questions in three cards, each linking to its explainer page.
All logic is in retailpulse.demo / viz; this page only lays it out.
"""

from __future__ import annotations

import streamlit as st
from components import data, layout, tour

from retailpulse import contracts as C
from retailpulse import demo
from retailpulse.baskets.rules import recommend_from_basket
from retailpulse.viz import predictions as VP
from retailpulse.viz import segments as S
from retailpulse.viz.common import money
from retailpulse.viz.customer import segment_badge_html

FEATURES = demo.DEMO_FEATURES
VALUES_KEY = "demo_values"  # plain session state: survives page switches (widget keys do not)
BASKET_KEY = "demo_basket_codes"
BASKET_WIDGET = "demo_basket"
PCA_POINTS = 2000
HOW = "How does this work? →"


def _widget(feature: str) -> str:
    return f"demo_in_{feature}"


layout.setup_page("Live demo")
layout.page_header(
    "RetailPulse",
    "Who are our customers, will they come back, and what should we offer them?",
    f"Live demo: describe a customer with {len(FEATURES)} numbers and a basket, and the saved "
    "models answer "
    "all three questions. Every card links to a page that explains how it works.",
)

# ------------------------------------------------------------------ headline + pipeline
facts = demo.kpi_facts(
    data.json_file(C.WAREHOUSE_SUMMARY_JSON),
    data.json_file(C.CLEANING_STEPS_JSON),
    data.json_file(C.FORMAT_COMPARISON_JSON),
)
tour.callout(
    1,
    "A UK online gift shop's invoices, cleaned. Open the pipeline below to see "
    "how raw Excel becomes the three models behind the cards.",
    "The data",
)
layout.kpi_row(
    [
        {
            "label": "Clean sales lines",
            "value": f"{facts['clean_rows']:,}" if facts["clean_rows"] else "n/a",
            "help": f"From {facts['raw_rows']:,} raw rows." if facts["raw_rows"] else None,
        },
        {
            "label": "Customers",
            "value": f"{facts['customers']:,}" if facts["customers"] else "n/a",
            "help": "Registered customers (guest checkouts have no Customer ID).",
        },
        {"label": "Revenue", "value": money(facts["revenue"] or float("nan"), 2)},
        {
            "label": "Date range",
            "value": f"{facts['date_min']} → {facts['date_max']}" if facts["date_min"] else "n/a",
        },
    ]
)
with st.expander("How it works: the pipeline", icon=":material/account_tree:"):
    st.graphviz_chart(demo.pipeline_dot())
    st.caption(
        "An offline pipeline builds every model and table once; this app only reads the results."
    )

# ------------------------------------------------------------------ inputs
names = data.segment_names()
presets = data.table(C.SEGMENT_PRESETS_PARQUET)
products = data.table(C.PRODUCTS_PARQUET)
horizon = data.metrics("features").get("horizon_days", 90)


def _fill(segment_id: int) -> None:
    """Preset button: copy the segment's typical (median) values into the inputs."""
    values = demo.preset_values(presets, segment_id)
    if not values:
        return
    st.session_state[VALUES_KEY] = values
    for f, v in values.items():
        st.session_state[_widget(f)] = v


if VALUES_KEY not in st.session_state and names:
    _fill(next(iter(names)))
for f in FEATURES:  # restore inputs after a page switch (Streamlit drops widget state)
    if _widget(f) not in st.session_state:
        saved = st.session_state.get(VALUES_KEY, {}).get(f)
        st.session_state[_widget(f)] = demo.input_value(f, saved)

st.subheader("1 · Describe a customer")
tour.callout(
    2,
    "Click a typical customer of a segment (the median of each number), or type your own. "
    "Then add products to the basket.",
    "Try it",
)
if names:
    for col, (sid, name) in zip(st.columns(len(names)), names.items(), strict=True):
        col.button(
            f"Typical {name}",
            key=f"preset_{sid}",
            on_click=_fill,
            args=(sid,),
            width="stretch",
            icon=":material/person:",
        )
for col, f in zip(st.columns(len(FEATURES)), FEATURES, strict=True):
    spec = demo.input_spec(f)
    col.number_input(
        spec["label"],
        min_value=spec["min"],
        step=spec["step"],
        key=_widget(f),
        help=spec["help"],
        format=None if spec["integer"] else "%.2f",
    )
values = {f: st.session_state[_widget(f)] for f in FEATURES}
st.session_state[VALUES_KEY] = values
preset = demo.matching_preset(presets, values)
st.caption(
    f"Inputs = the typical **{names.get(preset, preset)}** customer (median of each number)."
    if preset is not None
    else "Inputs = your own numbers."
)
for note in demo.input_warnings(values):
    st.warning(note, icon=":material/warning:")

codes = [] if products is None else products["stock_code"].astype(str).tolist()
desc = (
    {} if products is None else dict(zip(codes, products["description"].astype(str), strict=True))
)
if BASKET_KEY not in st.session_state:
    st.session_state[BASKET_KEY] = demo.default_basket(data.metrics("rules"), codes)
if BASKET_WIDGET not in st.session_state:
    st.session_state[BASKET_WIDGET] = [c for c in st.session_state[BASKET_KEY] if c in desc]
basket = st.multiselect(
    "Basket: products this customer bought",
    codes,
    key=BASKET_WIDGET,
    format_func=lambda c: f"{desc.get(c, c)} ({c})",
    placeholder=f"Type to search the {len(codes):,} most-bought products",
)
st.session_state[BASKET_KEY] = basket

# ------------------------------------------------------------------ results
raw = demo.raw_frame(values)
place = demo.place_segment(data.model(C.SEGMENT_MODEL_JOBLIB), raw)
probs = demo.return_probabilities(data.model(C.CLASSIFIERS_JOBLIB), raw)
answer = demo.verdict(probs, horizon)
recs = (
    recommend_from_basket(basket, data.table(C.RULES_PARQUET), products)
    if products is not None
    else None
)

st.subheader("2 · What the models say")
tour.callout(
    3,
    "Each card answers one question for this customer. Change a number or the basket and "
    "watch all three update; the link under each card opens its explainer page.",
    "Three answers",
)
card_seg, card_back, card_offer = st.columns(3)
with card_seg, st.container(border=True):
    st.markdown("#### Who is this customer?")
    sid = place["segment"]
    st.markdown(
        "Segment: " + segment_badge_html(sid, names.get(sid) if sid is not None else None),
        unsafe_allow_html=True,
    )
    segments = data.table(C.SEGMENTS_PARQUET, columns=["customer_id", "segment", "pc1", "pc2"])
    layout.chart(
        S.pca_scatter_2d(
            segments,
            names,
            data.metrics("segment").get("pca_explained_variance"),
            n=PCA_POINTS,
            star=place["pc"],
            star_segment=sid,
            height=430,
        ),
        key="home_pca",
    )
    st.caption(
        f"★ = this customer; faded dots = about {PCA_POINTS:,} real customers "
        "(seeded sample), coloured by segment."
    )
    st.page_link("pages/3_Segmentation.py", label=HOW, icon=":material/scatter_plot:")
with card_back, st.container(border=True):
    st.markdown("#### Will they come back?")
    st.markdown(answer["text"])
    layout.chart(VP.model_probability_bars(probs, horizon), key="home_probs")
    st.caption("Black tick = each model's threshold, chosen on the training data.")
    st.page_link("pages/4_Prediction.py", label=HOW, icon=":material/insights:")
with card_offer, st.container(border=True):
    st.markdown("#### What to offer")
    lines = demo.recommendation_lines(recs)
    if not lines:
        st.info("No products available.")
    for line in lines:
        st.markdown(
            f"**{line['rank']}. {demo.md_escape(line['product'])}**  \n"
            + demo.md_escape(line["reason"])
        )
    st.caption("Never a product that is already in the basket.")
    st.page_link("pages/5_Recommendation.py", label=HOW, icon=":material/shopping_basket:")

st.divider()
st.markdown("**More:**")
c_data, c_ins = st.columns(2)
with c_data:
    st.page_link(
        "pages/2_Data.py", label="The data and how it was cleaned", icon=":material/database:"
    )
with c_ins:
    st.page_link(
        "pages/6_Insights.py",
        label="What should the business do? (Insights)",
        icon=":material/lightbulb:",
    )
st.caption(layout.DATA_CREDIT + " Built with free, open-source tools only.")
