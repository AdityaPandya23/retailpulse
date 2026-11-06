"""Prediction (question B): the out-of-time split, the 3 models, the Time Machine backtest and
a simple KNN bias-variance chart. Logic in retailpulse.viz.predictions; this page lays it out."""

from __future__ import annotations

import streamlit as st
from components import data, layout, tour

from retailpulse import contracts as C
from retailpulse.config import PREDICT
from retailpulse.viz import predictions as VP
from retailpulse.viz.common import pct

REVEAL_KEY = "pred_revealed"

layout.setup_page("Prediction")
horizon = data.horizon_days()
layout.page_header(
    "Prediction",
    f"B · Will they buy again in the next {horizon} days?",
    f"{len(C.CLASSIFIERS)} classifiers learn from the past and are judged on a future they "
    "never saw. The model is chosen on training data only.",
)

cls = data.metrics("classify")
bt = data.metrics("backtest")
classification = data.table(C.CLASSIFICATION_PARQUET)
if classification is None or classification.empty:
    st.info("No predictions yet: run `python -m retailpulse.pipeline`.")
    st.stop()
best = VP.best_model(cls, classification)
best_test = VP.test_predictions(classification, best)
threshold = VP.model_threshold(cls, best, bt)
summary = VP.time_machine_summary(best_test)  # once per rerun: section 2 and section 3
cutoff = cls.get("test_cutoff") or data.metrics("features").get("cutoff") or "the cutoff"
train_cutoff = cls.get("train_cutoff") or "an earlier snapshot"

# ------------------------------------------------------------------ split
st.subheader("1 · A fair test: the out-of-time split")
st.markdown(
    f"The models learn from customers as they looked on **{train_cutoff}** and whether they "
    f"bought in the {horizon} days after. They are then judged on the real cutoff "
    f"**{cutoff}**: same features, a future they never saw. A random split would leak "
    "future behaviour into training."
)
tour.callout(
    1,
    f"Blue = history used to build the {len(PREDICT.features)} features; orange = the window "
    "where we check "
    "who bought again. The test outcome lies entirely after the training data.",
    "Out-of-time split",
)
layout.chart(
    VP.split_timeline(cls, data.json_file(C.FORMAT_COMPARISON_JSON).get("date_min"), horizon),
    key="pred_split",
)

# ------------------------------------------------------------------ models
st.subheader(f"2 · {len(C.CLASSIFIERS)} models compared")
tour.callout(
    2,
    "ROC-AUC = the chance that a random returner gets a higher score than a random "
    "non-returner (0.5 = guessing, 1 = perfect). The curves lie close together.",
    "Model comparison",
)
col_table, col_roc = st.columns([2, 3])
with col_table:
    table = VP.simple_model_table(cls)
    if table.empty:
        st.info("No model metrics available.")
    else:
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Test ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
                "Accuracy": st.column_config.NumberColumn(format="percent"),
                "Threshold": st.column_config.NumberColumn(format="%.2f"),
            },
            key="pred_models",
        )
        st.caption(
            f"CV ROC-AUC: mean ± sd over {cls.get('cv_folds', PREDICT.cv_folds)} folds of the "
            "training data (used to choose the "
            "model). Test: the future window (only reported). Threshold: best F1 on training."
        )
    layout.conclusion(
        [VP.tie_note(cls), (cls.get("paired_t_auc") or {}).get("conclusion")],
        "Are the models really different?",
    )
with col_roc:
    layout.chart(VP.roc_figure(data.table(C.ROC_PARQUET), cls), key="pred_roc")

col_cm, col_cm_text = st.columns([3, 2])
with col_cm:
    layout.chart(VP.confusion_figure(best_test, best, threshold), key="pred_confusion")
with col_cm_text:
    cm = summary
    if cm.get("n_customers"):
        st.markdown(
            f"**{best}** (the chosen model) flags {cm['n_predicted']:,} of "
            f"{cm['n_customers']:,} customers as 'will buy again'. "
            f"{pct(cm['share_actually_returned'])} of them did (precision), and they include "
            f"{pct(cm['recall'])} of everyone who came back (recall)."
        )
        st.caption("Each row shows how one kind of customer was classified.")

# ------------------------------------------------------------------ Time Machine
st.subheader("3 · Time Machine: were we right?")


def _reveal(flag: bool) -> None:
    st.session_state[REVEAL_KEY] = flag


s = summary
if not s.get("n_customers"):  # no test rows for the chosen model: nothing to reveal
    st.info(f"No test-window predictions for the chosen model ({best}).")
else:
    st.markdown(
        f"On **{cutoff}** the chosen model scored all {s['n_customers']:,} customers and "
        f"wrote down **{s['n_predicted']:,}** it expected to buy again within {horizon} days"
        + (f" (probability ≥ {threshold:.2f})." if threshold is not None else ".")
    )
    if not st.session_state.get(REVEAL_KEY, False):
        st.button(
            "Reveal what actually happened",
            key="pred_reveal",
            type="primary",
            icon=":material/visibility:",
            on_click=_reveal,
            args=(True,),
        )
    else:
        st.button("Hide the outcome again", key="pred_hide", on_click=_reveal, args=(False,))
        tour.callout(
            3,
            "Customers ranked by predicted probability into ten groups: the top group really came "
            "back far more often than average, the bottom group rarely.",
            "The reveal",
        )
        layout.kpi_row(
            [
                {"label": "We predicted", "value": f"{s['n_predicted']:,} customers"},
                {
                    "label": "… of whom bought again",
                    "value": pct(s["share_actually_returned"]),
                    "help": f"{s['n_predicted_and_returned']:,} of {s['n_predicted']:,}.",
                },
                {
                    "label": "Base rate (all customers)",
                    "value": pct(s["base_rate"]),
                    "help": f"{s['n_actually_returned']:,} of {s['n_customers']:,} came back.",
                },
                {
                    "label": "Lift",
                    "value": f"{s['lift']:.2f}×" if s["lift"] == s["lift"] else "n/a",
                    "help": "How much better than picking customers at random.",
                },
            ]
        )
        layout.chart(
            VP.decile_figure(VP.deciles(best_test), s["base_rate"], best), key="pred_deciles"
        )
        layout.conclusion(bt.get("conclusion"), "Backtest")

# ------------------------------------------------------------------ KNN
st.subheader("4 · KNN: how many neighbours?")
curve = data.table(C.KNN_CURVE_PARQUET)
best_k = VP.knn_best_k(cls)
options = VP.knn_k_options(curve, best_k)
if not options:
    st.info("No KNN curve available.")
else:
    st.markdown(
        "KNN predicts from the k most similar past customers. Move k and compare the error on "
        "the data it learned from (train) with the error on the future (test)."
    )
    k = st.select_slider(
        "Number of neighbours k",
        options=options,
        value=best_k if best_k in options else options[0],
        format_func=lambda v: f"{v} (best)" if v == best_k else str(v),
        key="pred_knn_k",
    )
    tour.callout(
        4,
        "Small k: train error near 0 but test error high (it memorised). Large k: both errors "
        "rise (too simple). The best k sits in between.",
        "Bias vs variance",
    )
    acc = VP.knn_accuracies(curve, k)
    layout.kpi_row(
        [
            {"label": "Train accuracy", "value": pct(acc["train_accuracy"], 1)},
            {"label": "Test accuracy", "value": pct(acc["test_accuracy"], 1)},
            {
                "label": "Gap (train − test)",
                "value": f"{100 * acc['gap']:.1f} pts",
                "help": "A big gap means the model memorised the training data.",
            },
        ]
    )
    st.markdown(VP.knn_regime(k, best_k))
    layout.chart(VP.knn_simple_figure(curve, k, best_k), key="pred_knn")

st.divider()
st.page_link("Home.py", label="← Back to the live demo", icon=":material/home:")
