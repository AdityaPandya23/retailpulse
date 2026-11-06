"""Page 3 figures on the MOCK artifacts (artifacts/mock/) and tiny hand-made frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from retailpulse import contracts as C
from retailpulse.viz import palette as P
from retailpulse.viz import predictions as VP


@pytest.fixture(scope="module")
def mock():
    if not C.all_exist(list(C.MOCK_OUTPUTS)):
        pytest.skip("artifacts/mock/ not generated")
    m = C.read_metrics(C.mock_path(C.METRICS_JSON))

    def read(p):
        return pd.read_parquet(C.mock_path(p))

    return {
        "cls": m["classify"],
        "reg": m["regress"],
        "bt": m["backtest"],
        "classification": read(C.CLASSIFICATION_PARQUET),
        "roc": read(C.ROC_PARQUET),
        "pr": read(C.PR_PARQUET),
        "knn": read(C.KNN_CURVE_PARQUET),
        "regression": read(C.REGRESSION_PARQUET),
    }


def _hand_test() -> pd.DataFrame:
    # 6 customers: 3 predicted (2 returned), 3 not predicted (1 returned)
    return pd.DataFrame(
        {
            "customer_id": range(1, 7),
            "model": "logistic",
            "split": "test",
            "p_return": [0.9, 0.8, 0.6, 0.3, 0.2, 0.1],
            "pred_return": [1, 1, 1, 0, 0, 0],
            "returned_flag": [1, 1, 0, 1, 0, 0],
        }
    )


# ---------------- lookup ----------------
def test_best_model_from_metrics_then_cv_then_frame(mock):
    assert VP.best_model(mock["cls"]) == mock["cls"]["best_model"]
    assert (
        VP.best_model({"models": {"lda": {"cv_auc_mean": 0.7}, "qda": {"cv_auc_mean": 0.8}}})
        == "qda"
    )
    assert VP.best_model({}, mock["classification"]) == C.CLASSIFIERS[0]
    assert VP.best_model(None) is None


def test_model_threshold_fallbacks():
    cls = {"models": {"knn": {"threshold": 0.4}, "lda": {}}, "best_model": "lda", "threshold": 0.3}
    assert VP.model_threshold(cls, "knn") == 0.4
    assert VP.model_threshold(cls, "lda") == 0.3
    assert VP.model_threshold({}, "qda", {"model": "qda", "threshold": 0.5}) == 0.5
    assert VP.model_threshold({}, "qda") is None


def test_test_predictions_filters_split_and_model(mock):
    t = VP.test_predictions(mock["classification"], "logistic")
    assert len(t) and set(t["split"]) == {"test"} and set(t["model"]) == {"logistic"}
    assert VP.test_predictions(None, "logistic").empty
    assert VP.test_predictions(mock["classification"], "unknown").empty


# ---------------- Time Machine ----------------
def test_time_machine_summary_by_hand():
    s = VP.time_machine_summary(_hand_test())
    assert s["n_customers"] == 6 and s["n_predicted"] == 3
    assert s["share_actually_returned"] == pytest.approx(2 / 3)
    assert s["base_rate"] == pytest.approx(0.5)
    assert s["lift"] == pytest.approx((2 / 3) / 0.5)
    assert s["recall"] == pytest.approx(2 / 3)
    assert VP.time_machine_summary(_hand_test().iloc[:0]) == {"n_customers": 0, "n_predicted": 0}


def test_probability_histogram_and_threshold(mock):
    t = VP.test_predictions(mock["classification"], mock["cls"]["best_model"])
    fig = VP.probability_histogram(t, 0.5, mock["cls"]["best_model"])
    assert isinstance(fig.data[0], go.Histogram) and len(fig.data[0].x) == len(t)
    assert any("threshold 0.50" in a.text for a in fig.layout.annotations)
    assert fig.layout.showlegend is False
    assert not VP.probability_histogram(t.iloc[:0], 0.5, "x").data  # empty -> note only


def test_decile_chart_has_ten_bars_and_base_rate(mock):
    t = VP.test_predictions(mock["classification"], mock["cls"]["best_model"])
    table = VP.deciles(t)
    fig = VP.decile_figure(table, float(t["returned_flag"].mean()), "logistic")
    bars = [d for d in fig.data if isinstance(d, go.Bar)]
    assert len(bars) == 1 and len(bars[0].x) == 10
    assert len(fig.data) == 2 and "base rate" in fig.data[1].name
    assert table["n"].sum() == len(t)
    assert not VP.decile_figure(VP.deciles(t.iloc[:0]), 0.5).data


# ---------------- ROC / PR ----------------
def test_roc_has_five_models_plus_chance(mock):
    fig = VP.roc_figure(mock["roc"], mock["cls"])
    assert len(fig.data) == 6
    names = [d.name for d in fig.data]
    assert names[-1].startswith("chance")
    for m, d in zip(C.CLASSIFIERS, fig.data[:5], strict=True):
        assert d.name.startswith(m) and "AUC" in d.name
        assert d.line.color == P.CLASSIFIER_COLOURS[m] and d.line.dash == P.CLASSIFIER_DASH[m]
    best = mock["cls"]["best_model"]
    assert sum("best" in n for n in names) == 1 and any(
        n.startswith(best) and "best" in n for n in names
    )
    assert fig.layout.xaxis.title.text and fig.layout.yaxis.title.text


def test_pr_drops_inf_thresholds_and_has_baseline(mock):
    pr = pd.concat(
        [
            mock["pr"],
            pd.DataFrame(
                {"model": ["logistic"], "recall": [0.0], "precision": [1.0], "threshold": [np.inf]}
            ),
        ]
    )
    fig = VP.pr_figure(pr, mock["cls"], 0.4)
    assert len(fig.data) == 6 and fig.data[-1].y == (0.4, 0.4)
    for d in fig.data[:5]:
        assert np.isfinite(np.asarray(d.customdata, dtype=float)).all()
    assert not VP.pr_figure(None, mock["cls"], 0.4).data


def test_thin_keeps_first_and_last():
    df = pd.DataFrame({"a": range(5000)})
    t = VP.thin(df, 100)
    assert len(t) <= 100 and t["a"].iloc[0] == 0 and t["a"].iloc[-1] == 4999


# ---------------- model table ----------------
def test_model_table_marks_best_once(mock):
    t = VP.model_table(mock["cls"])
    assert len(t) == len(mock["cls"]["models"]) and (t["Best (train CV)"] == "★").sum() == 1
    assert {"CV ROC-AUC", "Test ROC-AUC", "Test PR-AUC (AP)", "Accuracy", "Threshold"} <= set(t)
    assert VP.model_table({}).empty


def test_tie_note_from_numbers():
    tied = {
        "best_model": "a",
        "models": {
            "a": {"test_auc": 0.79, "cv_auc_std": 0.01},
            "b": {"test_auc": 0.80, "cv_auc_std": 0.01},
        },
    }
    assert "practically tied" in VP.tie_note(
        tied
    ) and "(a) is not the best on test (b)" in VP.tie_note(tied)
    apart = {
        "models": {
            "a": {"test_auc": 0.6, "cv_auc_std": 0.01},
            "b": {"test_auc": 0.8, "cv_auc_std": 0.01},
        }
    }
    assert "meaningful" in VP.tie_note(apart)
    assert VP.tie_note({"models": {"a": {"test_auc": 0.7}}}) is None


# ---------------- confusion ----------------
def test_confusion_counts_sum_to_n(mock):
    t = VP.test_predictions(mock["classification"], "lda")
    cm = VP.confusion_counts(t)
    assert cm.shape == (2, 2) and cm.to_numpy().sum() == len(t)
    hand = VP.confusion_counts(_hand_test())
    assert (
        hand.loc[1, 1] == 2 and hand.loc[0, 1] == 1 and hand.loc[1, 0] == 1 and hand.loc[0, 0] == 2
    )
    fig = VP.confusion_figure(t, "lda", 0.5)
    assert isinstance(fig.data[0], go.Heatmap) and len(fig.layout.annotations) == 4
    assert np.allclose(np.asarray(fig.data[0].z).sum(axis=1), 1.0)  # row %


def test_confusion_with_one_class_only():
    t = _hand_test().assign(pred_return=0)
    assert VP.confusion_counts(t).loc[:, 1].sum() == 0
    assert VP.confusion_figure(t, "logistic", None).data


# ---------------- KNN ----------------
def test_knn_figure_marks_selected_and_chosen_k(mock):
    fig = VP.knn_figure(mock["knn"], 10, 25)
    lines = [d for d in fig.data if d.mode == "lines"]
    marks = [d for d in fig.data if d.mode == "markers"]
    assert len(lines) == 2 and len(marks) == 2 and all(m.x == (10,) for m in marks)
    assert any("chosen k = 25" in a.text for a in fig.layout.annotations)
    pt = VP.knn_point(mock["knn"], 10)
    assert pt["k"] == 10 and 0 <= pt["test_error"] <= 1
    assert VP.knn_point(mock["knn"], 10_000)["k"] == mock["knn"]["k"].max()
    assert not VP.knn_figure(None, 5, None).data
    assert not VP.knn_figure(mock["knn"][["k"]], 5, None).data  # missing columns


# ---------------- regression ----------------
def test_regression_table_includes_mean_baseline(mock):
    t = VP.regression_table(mock["reg"], mock["regression"])
    assert list(t["Model"].str.split(" ").str[0])[:-1] == list(C.REGRESSORS)
    assert t["Model"].iloc[-1] == "mean-only baseline"  # mock has no reference_mean -> computed
    reg = mock["regression"]
    te = reg[(reg["model"] == "knn") & (reg["split"] == "test")]
    rmse = float(np.sqrt(((te["y_pred"] - te["y_true"]) ** 2).mean()))
    assert t.loc[t["Model"].str.startswith("knn"), "Test RMSE (log)"].iloc[0] == pytest.approx(rmse)
    ref = {"reference_mean": {"rmse": 9.0, "mae": 8.0, "r2": 0.0, "rmse_gbp": 7.0}}
    assert VP.regression_table(ref, mock["regression"])["Test RMSE (log)"].iloc[-1] == 9.0


def test_regression_figures(mock):
    fig = VP.pred_vs_actual_figure(mock["regression"], "multiple_lr")
    assert len(fig.data) == 2 and "45°" in fig.data[1].name
    assert fig.data[0].marker.color == P.REGRESSOR_COLOURS["multiple_lr"]
    assert fig.layout.xaxis.ticktext[0].startswith("£")
    res = VP.residual_figure(mock["regression"], "knn")
    te = mock["regression"].query("model == 'knn' and split == 'test'")
    assert np.allclose(res.data[0].y, te["y_true"] - te["y_pred"])
    assert not VP.pred_vs_actual_figure(mock["regression"], "nope").data
    assert not VP.residual_figure(None, "knn").data
