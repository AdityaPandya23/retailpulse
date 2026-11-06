"""Segments & Stats: K-Means segments, how k was chosen, validation, statistics, live tests."""

from __future__ import annotations

import html

import streamlit as st
from components import data, layout, tour

from retailpulse import contracts as C
from retailpulse.viz import segments as S

layout.setup_page("Segments & Stats")
layout.page_header(
    "Segments & Stats",
    "A · Who are our customers? K-Means segments, checked against a simple baseline.",
    "Customers are described by 7 features at the cutoff (log1p + standardised), grouped "
    "with K-Means and validated with PCA, stability, hypothesis tests and distribution fits.",
)

seg = data.metrics("segment")
stats = data.metrics("stats")
names = data.segment_names()
segments = data.table(C.SEGMENTS_PARQUET)
profiles = data.table(C.SEGMENT_PROFILES_PARQUET)
features = data.table(C.FEATURES_CUTOFF_PARQUET)
explained = seg.get("pca_explained_variance")
chosen_k = seg.get("k", len(names) or None)
joined = S.segment_feature_frame(features, segments)
feature_options = [f for f in C.FEATURE_COLUMNS if f in joined.columns]


def seg_label(sid: int) -> str:
    return f"{sid} · {names.get(sid, sid)}"


tab_seg, tab_k, tab_valid, tab_stats, tab_live = st.tabs(
    ["Segments", "Choosing k & PCA", "Validation", "Statistics", "Live test"]
)

# ------------------------------------------------------------------ segments
with tab_seg:
    ari = (seg.get("stability_ari") or {}).get("mean")
    layout.kpi_row(
        [
            {"label": "Segments (k)", "value": str(chosen_k), "help": seg.get("k_choice_reason")},
            {
                "label": "Silhouette",
                "value": f"{seg['silhouette']:.2f}" if seg.get("silhouette") is not None else "n/a",
                "help": "−1 … 1: how much closer customers are to their own segment than to "
                "the next one.",
            },
            {
                "label": "Stability (mean ARI)",
                "value": f"{ari:.2f}" if ari is not None else "n/a",
                "help": seg.get("stability_method"),
            },
            {"label": "Customers segmented", "value": f"{len(joined):,}"},
        ]
    )
    tour.callout(
        1,
        "One card per segment, ranked by revenue (0 = most valuable). The colour bar is the "
        "segment's colour on every chart of the app.",
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

    tour.callout(
        2,
        "Each dot is a customer on the first three principal components; drag to rotate. "
        "The segments occupy different regions, with some overlap at the borders.",
        "PCA view",
    )
    layout.chart(S.pca_scatter_3d(segments, names, explained), key="p2_pca3d")

    col_par, col_box = st.columns(2)
    with col_par:
        layout.chart(S.parallel_profiles(profiles, names), key="p2_parallel")
        st.caption("Hover a point for the segment's real mean.")
    with col_box:
        if feature_options:
            box_feature = st.selectbox(
                "Feature",
                feature_options,
                index=feature_options.index("monetary") if "monetary" in feature_options else 0,
                format_func=S.feature_label,
                key="seg_box_feature",
            )
            layout.chart(S.segment_boxplots(joined, box_feature, names), key="p2_box")
            st.caption("Box = quartiles, line = median, dashed = mean; log axis where skewed.")
        else:
            st.info("Customer features are not available.")
    layout.conclusion(seg.get("conclusion"))

# ------------------------------------------------------------------ choosing k + PCA
with tab_k:
    tour.callout(
        3,
        "K-Means needs k up front. Each panel scores every candidate k in a different way; "
        "the dashed line marks the k in use. The rule below picks it, a human can override it.",
        "Choosing k",
    )
    layout.chart(S.k_selection_panels(data.table(C.K_SELECTION_PARQUET), chosen_k), key="p2_ksel")
    with st.container(border=True):
        source = seg.get("k_source", "automatic rule")
        st.markdown(f"**Chosen: k = {chosen_k}** ({source})")
        if seg.get("k_choice_reason"):
            st.markdown(seg["k_choice_reason"])
        if seg.get("k_rule"):
            st.caption(f"Rule: {seg['k_rule']}")
        if seg.get("naming_rule"):
            with st.expander("How the segment names are chosen"):
                st.markdown(seg["naming_rule"])
        st.markdown(
            ":material/edit: **A human can override k and the names** with the evidence on "
            "this page: set `segment.k` and/or `segment.names` ({id: name}) in `config.yaml`, "
            "then rerun `python -m retailpulse.pipeline`."
        )
    col_scree, col_biplot = st.columns(2)
    with col_scree:
        layout.chart(S.scree_plot(explained), key="p2_scree")
    with col_biplot:
        layout.chart(
            S.pca_biplot(data.table(C.PCA_LOADINGS_PARQUET), segments, explained), key="p2_biplot"
        )
    st.caption(
        "Loadings: an arrow pointing right means customers high on that feature score high "
        "on PC1. PCA is used to visualise and check the segments, not to build them."
    )

# ------------------------------------------------------------------ validation
with tab_valid:
    baseline = seg.get("baseline") or {}
    st.subheader("K-Means vs an RFM-quartile baseline")
    table = S.baseline_table(baseline)
    if table.empty:
        st.info("Baseline comparison not recorded.")
    else:
        st.dataframe(table, hide_index=True, width="stretch", key="p2_baseline")
    st.caption(
        "Same scaled features for both. The RFM baseline groups customers by their recency, "
        "frequency and monetary quartiles; ROC-AUC (5-fold CV ± sd) asks how well the group "
        "alone predicts who buys again in the next 90 days."
    )
    layout.conclusion(baseline.get("conclusion"))

    st.subheader("Does segment depend on the country? (chi-square)")
    chi2 = seg.get("chi2_segment_uk") or {}
    col_chi, col_bar = st.columns([1, 2])
    with col_chi:
        if chi2:
            layout.kpi_row([{"label": "χ²", "value": f"{chi2.get('chi2', float('nan')):,.1f}"}])
            layout.kpi_row(
                [
                    {"label": "dof", "value": str(chi2.get("dof", "n/a"))},
                    {"label": "p-value", "value": f"{chi2.get('p_value', float('nan')):.2g}"},
                ]
            )
            layout.kpi_row(
                [
                    {
                        "label": "Cramér's V",
                        "value": f"{chi2.get('cramers_v', float('nan')):.2f}",
                        "help": "0 = no association, 1 = perfect; < 0.1 is negligible.",
                    }
                ]
            )
        else:
            st.info("Chi-square test not recorded.")
    with col_bar:
        layout.chart(S.uk_share_bar(chi2, names), key="p2_uk")
    layout.conclusion(chi2.get("conclusion"))

# ------------------------------------------------------------------ statistics
with tab_stats:
    st.caption(
        "Distribution fits and tests on all customers (features_full); every result has a "
        "one-line conclusion from the pipeline."
    )
    qq = data.table(C.QQ_PARQUET)
    qq_vars = list(dict.fromkeys(qq["variable"])) if qq is not None and len(qq) else []
    st.subheader("Which distribution fits? (Q-Q plots)")
    if qq_vars:
        qq_var = st.selectbox(
            "Variable",
            qq_vars,
            format_func=lambda v: S.QQ_LABELS.get(v, v),
            key="qq_variable",
        )
        layout.chart(S.qq_plot(qq, qq_var), key="p2_qq")
        st.caption("Points on the dashed line = the distribution fits; bends = it does not.")
    else:
        st.info("Q-Q points not available.")
    layout.section_conclusion(
        "stats", "order_value_lognormal", "gap_exponential", title="Distribution fits"
    )

    col_ff, col_hist = st.columns(2)
    with col_ff:
        layout.chart(S.frequency_fit(data.table(C.FREQUENCY_FIT_PARQUET)), key="p2_freqfit")
        layout.section_conclusion("stats", "frequency_poisson_vs_nbinom", title="Poisson or NB?")
    with col_hist:
        hist = data.table(C.EDA_HISTOGRAMS_PARQUET)
        hist_feats = [
            f for f in C.FEATURE_COLUMNS if hist is not None and f in set(hist["feature"])
        ]
        if hist_feats:
            hist_feature = st.selectbox(
                "Histogram feature",
                hist_feats,
                index=hist_feats.index("monetary") if "monetary" in hist_feats else 0,
                format_func=S.feature_label,
                key="hist_feature",
            )
            layout.chart(S.histogram_polygon(hist, hist_feature), key="p2_hist")
        else:
            st.info("Histograms not available.")

    st.subheader("Correlation: Pearson vs Spearman")
    layout.chart(S.correlation_heatmaps(data.table(C.CORRELATION_PARQUET)), key="p2_corr")
    layout.section_conclusion("stats", "correlation", title="Correlation")
    layout.section_conclusion(
        "stats", "basket_repeat_vs_onetime", "paired_spend_y1_y2", title="Hypothesis tests"
    )

# ------------------------------------------------------------------ live test
with tab_live:
    tour.callout(
        4,
        "Pick any two segments and a feature: the app runs Welch's t-test (means) and the "
        "Mann-Whitney U test (ranks) live and explains the result in plain English.",
        "Live hypothesis test",
    )
    ids = list(names)
    if len(ids) < 2 or not feature_options:
        st.info("Need at least two segments and customer features for a test.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            seg_a = st.selectbox("Segment A", ids, index=0, format_func=seg_label, key="live_seg_a")
        with c2:
            seg_b = st.selectbox("Segment B", ids, index=1, format_func=seg_label, key="live_seg_b")
        with c3:
            live_feature = st.selectbox(
                "Feature",
                feature_options,
                index=feature_options.index("monetary") if "monetary" in feature_options else 0,
                format_func=S.feature_label,
                key="live_feature",
            )
        result = S.compare_segments(joined, seg_a, seg_b, live_feature, names)
        layout.conclusion(result.get("conclusion"), "Result")
        res_table = S.result_table(result)
        if not res_table.empty:
            groups = result.get("groups") or {}
            layout.kpi_row(
                [
                    {
                        "label": f"{name}: n · median",
                        "value": f"{g.get('n', 0):,} · {g.get('median', float('nan')):,.1f}",
                    }
                    for name, g in groups.items()
                ]
            )
            st.dataframe(res_table, hide_index=True, width="stretch", key="live_table")
            agree = result.get("tests_agree")
            st.caption(
                "The two tests "
                + ("agree." if agree else "disagree — trust the rank test for skewed data.")
                + f" α = {result.get('alpha', 0.05)}. Mann-Whitney compares whole distributions "
                "(robust to outliers); Welch compares means without assuming equal variances."
            )
            layout.chart(
                S.two_segment_distribution(joined, seg_a, seg_b, live_feature, names),
                key="p2_live_dist",
            )
