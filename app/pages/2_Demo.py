"""Live demo: describe one customer (a "Typical <segment>" preset or 6 typed numbers + Enter);
the saved models answer who they are and whether they come back, followed by a few lines of
insight for this request. All logic is in retailpulse.demo / viz; this page only lays it out.
"""

from __future__ import annotations

import streamlit as st
from components import data, layout

from retailpulse import contracts as C
from retailpulse import demo
from retailpulse import insights as I
from retailpulse.viz import predictions as VP
from retailpulse.viz import segments as S
from retailpulse.viz.customer import segment_badge_html

FEATURES = demo.DEMO_FEATURES
VALUES_KEY = "demo_values"  # APPLIED values (after Enter / a preset); survives page switches
PCA_POINTS = 2000
HOW = "How does this work? →"


def _widget(feature: str) -> str:
    return f"demo_in_{feature}"


layout.setup_page("Live demo")
layout.page_header(
    "Live demo",
    "Who is this customer, and will they come back?",
    f"Describe a customer with {len(FEATURES)} numbers and press Enter: the saved models "
    "answer both questions. Every card links to a page that explains how it works.",
)

names = data.segment_names()
presets = data.table(C.SEGMENT_PRESETS_PARQUET)
profiles = data.table(C.SEGMENT_PROFILES_PARQUET)
horizon = data.horizon_days()


def _fill(segment_id: int) -> None:
    """Preset button: copy the segment's typical (median) values into the inputs and apply."""
    values = demo.preset_values(presets, segment_id)
    if not values:
        return
    st.session_state[VALUES_KEY] = values
    for f, v in values.items():
        st.session_state[_widget(f)] = v


def _apply() -> None:
    """Enter: the typed numbers become the customer the models score."""
    st.session_state[VALUES_KEY] = {f: st.session_state[_widget(f)] for f in FEATURES}


if VALUES_KEY not in st.session_state and names:
    _fill(next(iter(names)))
for f in FEATURES:  # restore inputs after a page switch (Streamlit drops widget state)
    if _widget(f) not in st.session_state:
        saved = st.session_state.get(VALUES_KEY, {}).get(f)
        st.session_state[_widget(f)] = demo.input_value(f, saved)
if VALUES_KEY not in st.session_state:  # no presets: start from the default inputs
    _apply()

# ------------------------------------------------------------------ inputs
st.subheader("1 · Describe a customer")
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
with st.form("demo_form"):
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
    st.form_submit_button(
        "Enter",
        key="demo_enter",
        type="primary",
        icon=":material/keyboard_return:",
        on_click=_apply,
    )
values = dict(st.session_state[VALUES_KEY])
preset = demo.matching_preset(presets, values)
st.caption(
    f"Results below = the typical **{names.get(preset, preset)}** customer (median of each number)."
    if preset is not None
    else "Results below = your own numbers. Change a number and press Enter to update them."
)
for note in demo.input_warnings(values):
    st.warning(note, icon=":material/warning:")

# ------------------------------------------------------------------ results
raw = demo.raw_frame(values)
place = demo.place_segment(data.model(C.SEGMENT_MODEL_JOBLIB), raw)
probs = demo.return_probabilities(data.model(C.CLASSIFIERS_JOBLIB), raw)
answer = demo.verdict(probs, horizon)
sid = place["segment"]

st.subheader("2 · What the models say")
card_seg, card_back = st.columns(2)
with card_seg, st.container(border=True):
    st.markdown("#### Who is this customer?")
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
            star=place["pc"][:2] if place["pc"] else None,
            star_segment=sid,
            height=430,
        ),
        key="demo_pca",
    )
    st.caption(
        f"★ = this customer; faded dots = about {PCA_POINTS:,} real customers "
        "(seeded sample), coloured by segment."
    )
    st.page_link("pages/3_Segmentation.py", label=HOW, icon=":material/scatter_plot:")
with card_back, st.container(border=True):
    st.markdown("#### Will they come back?")
    st.markdown(answer["text"])
    layout.chart(VP.model_probability_bars(probs, horizon), key="demo_probs")
    st.caption("Black tick = each model's threshold, chosen on the training data.")
    st.page_link("pages/4_Prediction.py", label=HOW, icon=":material/insights:")

# ------------------------------------------------------------------ insights
st.subheader("3 · Insights for this customer")
lines = demo.request_insights(
    sid,
    names.get(sid) if sid is not None else None,
    answer,
    len(probs),
    I.suggest_actions(profiles),
    I.came_back_share(profiles),
    horizon,
)
with st.container(border=True):
    st.markdown("\n".join(f"- {line}" for line in lines) if lines else "No insights available.")

st.divider()
st.page_link("Home.py", label="← Back to the overview", icon=":material/home:")
st.caption(layout.DATA_CREDIT + " Built with free, open-source tools only.")
