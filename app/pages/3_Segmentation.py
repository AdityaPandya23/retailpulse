"""Segmentation (question A): the features, choosing k, the PCA map and the segment profiles."""

from __future__ import annotations

import html

import streamlit as st
from components import data, layout, tour

from retailpulse import contracts as C
from retailpulse.config import SEGMENT
from retailpulse.viz import segments as S

layout.setup_page("Segmentation")
layout.page_header(
    "Segmentation",
    "A · Who are our customers?",
    "K-Means puts customers with similar numbers into the same group (segment). The number "
    "of segments k and their names are chosen automatically by a fixed rule.",
)

seg = data.metrics("segment")
names = data.segment_names()
chosen_k = seg.get("k", len(names) or None)
explained = seg.get("pca_explained_variance")
segments = data.table(C.SEGMENTS_PARQUET)
profiles = data.table(C.SEGMENT_PROFILES_PARQUET)
features_used = list(seg.get("features_used") or SEGMENT.features)

# ------------------------------------------------------------------ features
st.subheader(f"1 · The {len(features_used)} numbers that describe a customer")
st.markdown(
    "\n".join(
        f"- **{S.feature_label(f, False)}**: {S.FEATURE_MEANING.get(f, f)}" for f in features_used
    )
)
logged = [S.feature_label(f, False).lower() for f in seg.get("log1p_features") or []]
st.caption(
    "All are measured on the data before the cutoff. "
    + (f"Skewed ones ({', '.join(logged)}) get log(1 + x), then " if logged else "")
    + "every feature is standardised so no single one dominates the distances."
)

# ------------------------------------------------------------------ choosing k
st.subheader("2 · How many segments? (k)")
tour.callout(
    1,
    "Left: the error (WCSS) always falls as k grows; look for where it stops falling fast "
    "(the elbow). Right: silhouette, higher = better separated. The dashed line is the k in use.",
    "Choosing k",
)
layout.chart(
    S.k_selection_panels(data.table(C.K_SELECTION_PARQUET), chosen_k, ("wcss", "silhouette")),
    key="seg_k",
)
with st.container(border=True):
    st.markdown(f"**Chosen: k = {chosen_k}**")
    if seg.get("k_choice_reason"):
        st.markdown(seg["k_choice_reason"])

# ------------------------------------------------------------------ PCA
st.subheader("3 · The customer map (PCA)")
tour.callout(
    2,
    "PCA squeezes the features into 2 new axes that keep most of the spread, so customers "
    "can be drawn on a map. The segments sit in different regions.",
    "PCA map",
)
col_map, col_scree = st.columns([3, 2])
with col_map:
    layout.chart(S.pca_scatter_2d(segments, names, explained), key="seg_pca2d")
with col_scree:
    layout.chart(S.scree_plot(explained), key="seg_scree")
    st.markdown("**What each axis means**")
    st.markdown(
        "\n".join(
            f"- {line}" for line in S.pc_meanings(data.table(C.PCA_LOADINGS_PARQUET), explained)
        )
        or "Loadings not available."
    )
st.caption("PCA is only used to draw and check the segments; K-Means works on all features.")

# ------------------------------------------------------------------ profiles
st.subheader("4 · The segments")
tour.callout(
    3,
    "One card per segment, ranked by revenue (0 = most valuable). Its colour is the same on "
    "every chart of the app.",
    "Segment profiles",
)
cards = S.profile_cards(profiles, names)
if not cards:
    st.info("Segment profiles are not available yet (pipeline stage `segment`).")
per_row = min(len(cards), 4) or 1
for start in range(0, len(cards), per_row):
    for col, card in zip(st.columns(per_row), cards[start : start + per_row], strict=False):
        with col, st.container(border=True):
            st.markdown(
                f'<div style="border-left: 8px solid {card["colour"]}; padding-left: 10px;">'
                f"<b>{card['segment']} · {html.escape(card['name'])}</b></div>",
                unsafe_allow_html=True,
            )
            st.markdown("\n".join(f"- {k}: **{v}**" for k, v in card["lines"]))

col_share, col_box = st.columns(2)
with col_share:
    layout.chart(S.share_bars(profiles, names), key="seg_share")
    st.caption("Few customers can bring most of the revenue: compare the two bars of a segment.")
with col_box:
    joined = S.segment_feature_frame(data.table(C.FEATURES_CUTOFF_PARQUET), segments)
    options = [f for f in features_used if f in joined.columns]
    if options:
        feature = st.selectbox(
            "Feature",
            options,
            index=options.index("monetary") if "monetary" in options else 0,
            format_func=S.feature_label,
            key="seg_box_feature",
        )
        layout.chart(S.segment_boxplots(joined, feature, names), key="seg_box")
        st.caption("Box = middle half of customers, line = median, dashed = mean.")
    else:
        st.info("Customer features are not available.")
layout.conclusion(S.segment_conclusion(seg, profiles))

st.divider()
st.page_link("pages/2_Demo.py", label="← Back to the live demo", icon=":material/play_circle:")
