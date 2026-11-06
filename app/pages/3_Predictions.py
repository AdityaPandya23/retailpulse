"""Predictions (question B): who will come back in the 90 days after the cutoff?

Time Machine backtest, ROC / PR overlays, model table, confusion matrix, KNN bias-variance and
the spend regression. All logic lives in retailpulse.viz.predictions; this page lays it out.
"""

from __future__ import annotations

import streamlit as st
from components import data, layout, tour

from retailpulse import contracts as C
from retailpulse.viz import predictions as VP
from retailpulse.viz.common import pct

REVEAL_KEY = "tm_revealed"

layout.setup_page("Predictions")
layout.page_header(
    "Predictions",
    "B · Who will come back in the 90 days after the cutoff?",
    "Five classifiers were trained on an earlier snapshot and judged on the future window "
    "(out-of-time split). The model is chosen on train cross-validation only.",
)

cls = data.metrics("classify")
reg = data.metrics("regress")
bt = data.metrics("backtest")
feats = data.metrics("features")
classification = data.table(C.CLASSIFICATION_PARQUET)

if classification is None or classification.empty:
    st.info("No classification predictions yet: run `python -m retailpulse.pipeline`.")
    st.stop()

best = VP.best_model(cls, classification)
cutoff = cls.get("test_cutoff") or feats.get("cutoff") or "the cutoff"
horizon = feats.get("horizon_days", 90)


# cached wrappers around pure src functions (args hashed once per rerun; frames are small)
@st.cache_data(show_spinner=False)
def _test(frame, model):
    return VP.test_predictions(frame, model)


@st.cache_data(show_spinner=False)
def _roc_pr(roc, pr, metrics, base_rate):
    return VP.roc_figure(roc, metrics), VP.pr_figure(pr, metrics, base_rate)


@st.cache_data(show_spinner=False)
def _regression_figs(frame, model):
    return VP.pred_vs_actual_figure(frame, model), VP.residual_figure(frame, model)


def _reveal(flag: bool) -> None:
    st.session_state[REVEAL_KEY] = flag


tab_tm, tab_models, tab_knn, tab_reg = st.tabs(
    ["Time Machine", "Model comparison", "KNN bias-variance", "Spend regression"]
)

# ------------------------------------------------------------------ Time Machine
with tab_tm:
    test_best = _test(classification, best)
    thr = VP.model_threshold(cls, best, bt)
    s = VP.time_machine_summary(test_best)
    st.subheader(f"We are at the cutoff: {cutoff}")
    st.markdown(
        f"Using **{best}** (best by train cross-validation) and its train-chosen threshold, "
        f"we score all {s['n_customers']:,} customers and predict who will buy again within "
        f"the next {horizon} days. Nobody has seen the future yet."
    )
    layout.kpi_row(
        [
            {
                "label": "Customers scored at the cutoff",
                "value": f"{s['n_customers']:,}",
            },
            {
                "label": "Predicted to buy again",
                "value": f"{s['n_predicted']:,}",
                "help": "Customers whose predicted probability is at or above the threshold.",
            },
            {
                "label": "Threshold (chosen on train)",
                "value": f"{thr:.2f}" if thr is not None else "n/a",
                "help": cls.get("threshold_rule"),
            },
        ]
    )
    tour.callout(
        1,
        "Every customer gets a probability of buying again. Customers right of the dashed "
        "threshold are our 'will return' list, written down before we look at the future.",
        "The prediction",
    )
    layout.chart(VP.probability_histogram(test_best, thr, best), key="tm_hist")

    revealed = bool(st.session_state.get(REVEAL_KEY, False))
    if not revealed:
        st.button(
            "Reveal what actually happened",
            key="tm_reveal",
            type="primary",
            icon=":material/visibility:",
            on_click=_reveal,
            args=(True,),
        )
    else:
        st.button("Hide the outcome again", key="tm_hide", on_click=_reveal, args=(False,))
        st.markdown(f"#### What actually happened in the {horizon} days after {cutoff}")
        layout.kpi_row(
            [
                {"label": "We predicted", "value": f"{s['n_predicted']:,} customers"},
                {
                    "label": "… of whom actually bought again",
                    "value": pct(s["share_actually_returned"]),
                    "help": (
                        f"{s['n_predicted_and_returned']:,} of {s['n_predicted']:,} (precision)."
                    ),
                },
                {
                    "label": "Base rate (all customers)",
                    "value": pct(s["base_rate"]),
                    "help": f"{s['n_actually_returned']:,} of {s['n_customers']:,} came back.",
                },
                {
                    "label": "Lift",
                    "value": f"{s['lift']:.2f}×" if s["lift"] == s["lift"] else "n/a",
                    "help": "Share who returned among the predicted / base rate.",
                },
                {
                    "label": "Recall",
                    "value": pct(s["recall"]),
                    "help": "Share of all returners who were on our list.",
                },
            ]
        )
        tour.callout(
            2,
            "Customers ranked by predicted probability into ten equal groups: the top group "
            "really came back far more often than the base rate, the bottom group rarely.",
            "The reveal",
        )
        layout.chart(
            VP.decile_figure(VP.deciles(test_best), s["base_rate"], best), key="tm_deciles"
        )
        layout.conclusion([bt.get("conclusion")], "Backtest")

# ------------------------------------------------------------------ Model comparison
with tab_models:
    base_rate = VP.test_base_rate(cls, classification)
    roc_fig, pr_fig = _roc_pr(data.table(C.ROC_PARQUET), data.table(C.PR_PARQUET), cls, base_rate)
    tour.callout(
        3,
        "All five curves lie close together, far above the chance diagonal: every model ranks "
        "customers much better than guessing, and they are nearly interchangeable.",
        "Five models, one story",
    )
    c1, c2 = st.columns(2)
    with c1:
        layout.chart(roc_fig, key="roc")
    with c2:
        layout.chart(pr_fig, key="pr")

    st.markdown("#### Model table")
    table = VP.model_table(cls)
    if table.empty:
        st.info("No model metrics available.")
    else:
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Test ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
                "Test PR-AUC (AP)": st.column_config.NumberColumn(format="%.3f"),
                "Accuracy": st.column_config.NumberColumn(format="percent"),
                "Threshold": st.column_config.NumberColumn(format="%.2f"),
            },
            key="model_table",
        )
        st.caption(
            "CV ROC-AUC: mean ± sd over the train folds (used to choose the model). Test "
            "columns: the future window, used only to report. Threshold: max F1 on train."
        )
    layout.conclusion(
        [
            VP.tie_note(cls),
            (cls.get("paired_t_auc") or {}).get("conclusion"),
            cls.get("conclusion"),
        ],
        "Are the models really different?",
    )

    st.markdown("#### Confusion matrix")
    models = VP.model_order(classification["model"].unique())
    model = st.selectbox(
        "Model",
        models,
        index=models.index(best) if best in models else 0,
        key="cm_model",
        help="Counts on the test split at the model's own train-chosen threshold.",
    )
    c1, c2 = st.columns([3, 2])
    with c1:
        layout.chart(
            VP.confusion_figure(
                _test(classification, model), model, VP.model_threshold(cls, model, bt)
            ),
            key="confusion",
        )
    with c2:
        cm = VP.time_machine_summary(_test(classification, model))
        if cm.get("n_customers"):
            st.markdown(
                f"**{model}** flags {cm['n_predicted']:,} of {cm['n_customers']:,} customers; "
                f"{pct(cm['share_actually_returned'])} of them came back (precision) and they "
                f"cover {pct(cm['recall'])} of everyone who came back (recall)."
            )
            st.caption(
                "Rows are the actual outcome; the percentage is the share of that row, so the "
                "diagonal shows how often each kind of customer is classified correctly."
            )

# ------------------------------------------------------------------ KNN bias-variance
with tab_knn:
    curve = data.table(C.KNN_CURVE_PARQUET)
    if curve is None or curve.empty:
        st.info("No KNN bias-variance curve available.")
    else:
        k_chosen = cls.get("knn_k") or ((cls.get("models") or {}).get("knn") or {}).get("k")
        k_min, k_max = int(curve["k"].min()), int(curve["k"].max())
        default_k = (
            int(k_chosen) if k_chosen and k_min <= int(k_chosen) <= k_max else min(15, k_max)
        )
        k = st.slider(
            "Number of neighbours k",
            k_min,
            k_max,
            default_k,
            key="knn_k",
            help="Move k and watch train and test error.",
        )
        tour.callout(
            4,
            "Small k memorises the training data (train error near 0, test error high: "
            "variance). Large k averages over many neighbours (smoother, more bias). The test "
            "error is lowest in between.",
            "Bias vs variance",
        )
        layout.chart(VP.knn_figure(curve, k, k_chosen), key="knn_curve")
        pt = VP.knn_point(curve, k)
        st.markdown(
            f"At **k = {int(pt['k'])}**: train error {pct(pt['train_error'], 1)}, test error "
            f"{pct(pt['test_error'], 1)}"
            + (f", test ROC-AUC {pt['test_auc']:.3f}" if "test_auc" in pt else "")
            + "."
        )
        st.caption(
            "Bias = error from a model too simple to follow the data (large k); variance = error "
            "from a model that follows noise in the training data (small k)."
            + (f" {cls['knn_k_note']}." if cls.get("knn_k_note") else "")
        )

# ------------------------------------------------------------------ Spend regression
with tab_reg:
    regression = data.table(C.REGRESSION_PARQUET)
    st.markdown(
        f"How much will a customer spend in the {horizon} days **if** they come back? Target: "
        f"`{reg.get('target', 'log1p(next_spend)')}` on returners; simple linear regression "
        "(frequency only) vs multiple linear regression vs KNN, and a mean-only baseline."
    )
    rtable = VP.regression_table(reg, regression)
    if rtable.empty:
        st.info("No regression results available.")
    else:
        st.dataframe(
            rtable,
            hide_index=True,
            width="stretch",
            column_config={
                c: st.column_config.NumberColumn(format="%.3f")
                for c in ("CV RMSE (log)", "Test RMSE (log)", "Test MAE (log)", "Test R²")
            }
            | {"Test RMSE (£)": st.column_config.NumberColumn(format="£%.0f")},
            key="regression_table",
        )
    if regression is not None and not regression.empty:
        regs = VP.model_order(regression["model"].unique(), C.REGRESSORS)
        rbest = reg.get("best_model")
        rmodel = st.selectbox(
            "Regressor",
            regs,
            index=regs.index(rbest) if rbest in regs else 0,
            key="reg_model",
        )
        fig_pa, fig_res = _regression_figs(regression, rmodel)
        c1, c2 = st.columns(2)
        with c1:
            layout.chart(fig_pa, key="reg_scatter")
        with c2:
            layout.chart(fig_res, key="reg_residuals")
        st.caption(
            "Points on the dashed line are predicted exactly. A residual of +1 means the actual "
            "spend was about e ≈ 2.7 times the prediction (log scale)."
        )
    note = reg.get("expected_spend_note")
    layout.conclusion(
        [
            reg.get("conclusion"),
            f"Customer 360 uses expected spend = P(return) × predicted spend; {note}."
            if note
            else None,
        ],
    )
    if reg.get("population"):
        st.caption(f"Population: {reg['population']}.")
