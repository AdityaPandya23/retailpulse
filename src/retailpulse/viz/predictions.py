"""Page 3 (Predictions): Time Machine backtest, ROC / PR overlays, model table, confusion matrix,
KNN bias-variance curve and the spend-regression comparison.

Every function takes the P3 artifacts (frames / metrics.json sections) and returns a DataFrame,
a dict or a go.Figure; missing data gives an empty figure / empty frame, never an exception.
The best model, k and thresholds are always read from metrics, never hard-coded.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from retailpulse import contracts as C
from retailpulse.config import HORIZON_DAYS
from retailpulse.predict.backtest import decile_table
from retailpulse.viz import common as VC
from retailpulse.viz import palette as P

MAX_CURVE_POINTS = 600  # per model: the real PR curves have ~3,000 points each
OUTCOME_LABELS = ("Did not return", "Returned")
GBP_TICKS = (1, 10, 100, 1_000, 10_000, 100_000)


# ------------------------------------------------------------------ model / threshold lookup
def model_order(names, order=C.CLASSIFIERS) -> list[str]:
    """Models in a fixed order (C.CLASSIFIERS or C.REGRESSORS), unknown ones appended."""
    names = [str(n) for n in dict.fromkeys(names)]
    fixed = [m for m in order if m in names]
    return fixed + [m for m in names if m not in fixed]


def best_model(metrics: Mapping | None, frame: pd.DataFrame | None = None) -> str | None:
    """metrics["best_model"]; else the highest CV AUC; else the first model in the frame."""
    metrics = metrics or {}
    best = metrics.get("best_model")
    if best:
        return str(best)
    models = metrics.get("models") or {}
    scored = {m: v.get("cv_auc_mean") for m, v in models.items() if v.get("cv_auc_mean")}
    if scored:
        return max(scored, key=scored.get)
    if frame is not None and "model" in frame and len(frame):
        return model_order(frame["model"].unique())[0]
    return None


def model_threshold(
    cls_metrics: Mapping | None, model: str | None, backtest: Mapping | None = None
) -> float | None:
    """Train-chosen threshold of a model: models[m].threshold -> classify.threshold (best) ->
    backtest.threshold (its model). None when unknown (the mock metrics have none)."""
    cls_metrics, backtest = cls_metrics or {}, backtest or {}
    t = ((cls_metrics.get("models") or {}).get(model) or {}).get("threshold")
    if t is None and model is not None and model == cls_metrics.get("best_model"):
        t = cls_metrics.get("threshold")
    if t is None and model is not None and model == backtest.get("model"):
        t = backtest.get("threshold")
    return float(t) if t is not None else None


def test_predictions(classification: pd.DataFrame | None, model: str | None) -> pd.DataFrame:
    """Rows of one model on the test split (the future window), or an empty frame."""
    cols = C.CLASSIFICATION_COLUMNS
    if classification is None or model is None or not set(cols) <= set(classification):
        return pd.DataFrame(columns=cols)
    df = classification[(classification["split"] == "test") & (classification["model"] == model)]
    return df[cols].reset_index(drop=True)


def test_base_rate(cls_metrics: Mapping | None, classification: pd.DataFrame | None) -> float:
    """Share of test customers who came back (metrics, else from the test split)."""
    rate = (cls_metrics or {}).get("base_rate_test")
    if rate is not None:
        return float(rate)
    if classification is None or classification.empty:
        return float("nan")
    test = classification[classification["split"] == "test"]
    test = test.drop_duplicates("customer_id")
    return float(test["returned_flag"].mean()) if len(test) else float("nan")


# ------------------------------------------------------------------ Time Machine
def time_machine_summary(test: pd.DataFrame) -> dict:
    """Predicted vs actual for one model's test rows (pred_return is at its train threshold)."""
    n = int(len(test))
    if not n:
        return {"n_customers": 0, "n_predicted": 0}
    y = test["returned_flag"].to_numpy().astype(bool)
    pred = test["pred_return"].to_numpy().astype(bool)
    n_pred, n_ret = int(pred.sum()), int(y.sum())
    hit = int((pred & y).sum())
    share = hit / n_pred if n_pred else float("nan")
    base = n_ret / n
    return {
        "n_customers": n,
        "n_predicted": n_pred,
        "share_predicted": n_pred / n,
        "n_predicted_and_returned": hit,
        "share_actually_returned": share,
        "n_actually_returned": n_ret,
        "base_rate": base,
        "lift": share / base if base > 0 and n_pred else float("nan"),
        "recall": hit / n_ret if n_ret else float("nan"),
    }


def deciles(test: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Customers ranked by P(return) into equal bins (1 = most likely): actual return rate."""
    if test is None or test.empty:
        return pd.DataFrame(columns=["decile", "n", "mean_p", "return_rate", "lift"])
    return decile_table(test["returned_flag"].to_numpy(), test["p_return"].to_numpy(), n_bins)


def probability_histogram(
    test: pd.DataFrame, threshold: float | None, model: str | None
) -> go.Figure:
    """Distribution of the predicted probabilities made at the cutoff (no outcomes shown)."""
    title = "Predicted chance of buying again (made at the cutoff)"
    if test is None or test.empty:
        return VC.empty_figure("No test predictions available.", title)
    colour = P.model_colour(model or "")
    fig = go.Figure(
        go.Histogram(
            x=test["p_return"],
            xbins={"start": 0.0, "end": 1.0, "size": 0.05},
            marker={"color": colour, "line": {"color": P.SURFACE, "width": 1}},
            name=str(model),
            hovertemplate="P(return) %{x}<br>%{y:,} customers<extra></extra>",
        )
    )
    if threshold is not None:
        n_pred = int((test["p_return"] >= threshold).sum())
        fig.add_vline(x=threshold, line={"color": P.INK, "width": 2, "dash": "dash"})
        fig.add_annotation(
            x=threshold,
            y=1,
            yref="paper",
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            text=f" threshold {threshold:.2f}: {n_pred:,} predicted to return →",
            font={"color": P.INK, "size": 12},
        )
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(showlegend=False, bargap=0.02)
    return VC.style(fig, title, "Predicted probability of buying again", "Customers", height=360)


def decile_figure(table: pd.DataFrame, base_rate: float, model: str | None = None) -> go.Figure:
    """Actual return rate per probability decile (1 = highest P) vs the base-rate line."""
    title = "What happened: actual return rate by predicted-probability decile"
    if table is None or table.empty:
        return VC.empty_figure("No decile table available.", title)
    t = table.sort_values("decile")
    colour = P.model_colour(model or "")
    lift = t["lift"] if "lift" in t else t["return_rate"] / base_rate
    fig = go.Figure(
        go.Bar(
            x=t["decile"].astype(int),
            y=t["return_rate"],
            name=f"actual return rate ({model})" if model else "actual return rate",
            marker={"color": colour, "line": {"color": P.SURFACE, "width": 1}},
            customdata=np.column_stack([t["n"], t["mean_p"], lift]),
            hovertemplate=(
                "Decile %{x}<br>%{customdata[0]:,} customers<br>"
                "mean predicted %{customdata[1]:.0%}<br>actually returned %{y:.0%}"
                "<br>lift %{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )
    if np.isfinite(base_rate):
        fig.add_trace(
            go.Scatter(
                x=[0.5, len(t) + 0.5],
                y=[base_rate, base_rate],
                mode="lines",
                name=f"base rate {base_rate:.0%} (all customers)",
                line={"color": P.REFERENCE_COLOUR, "width": 2, "dash": "dash"},
                hovertemplate=f"base rate {base_rate:.1%}<extra></extra>",
            )
        )
    fig.update_xaxes(tickmode="linear", dtick=1, range=[0.5, len(t) + 0.5])
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_layout(legend={"orientation": "h", "y": -0.25})
    return VC.style(
        fig,
        title,
        "Decile of predicted probability (1 = most likely to return)",
        "Customers who bought again (%)",
        height=380,
    )


# ------------------------------------------------------------------ ROC / PR
def thin(df: pd.DataFrame, max_points: int = MAX_CURVE_POINTS) -> pd.DataFrame:
    """Evenly spaced rows (first and last kept) so long curves stay light in the browser."""
    if len(df) <= max_points:
        return df
    idx = np.unique(np.linspace(0, len(df) - 1, max_points).round().astype(int))
    return df.iloc[idx]


def _legend_name(model: str, label: str, value, best: str | None) -> str:
    v = f"{value:.3f}" if value is not None else "n/a"
    return f"{model} · {label} {v}" + (" ★ best by CV" if model == best else "")


def roc_figure(roc: pd.DataFrame | None, cls_metrics: Mapping | None) -> go.Figure:
    """ROC curve per model (colour + dash) with test AUC in the legend, plus the chance line."""
    title = "ROC curves on the future window (test)"
    if roc is None or roc.empty:
        return VC.empty_figure("No ROC curves available.", title)
    models = (cls_metrics or {}).get("models") or {}
    best = best_model(cls_metrics, roc)
    fig = go.Figure()
    for m in model_order(roc["model"].unique()):
        d = roc[roc["model"] == m].sort_values(["fpr", "tpr"])
        thr = d["threshold"].where(np.isfinite(d["threshold"]))  # (0, 0) has no real cut-off
        d = thin(d.assign(threshold=thr))
        fig.add_trace(
            go.Scatter(
                x=d["fpr"],
                y=d["tpr"],
                mode="lines",
                name=_legend_name(m, "AUC", (models.get(m) or {}).get("test_auc"), best),
                line={
                    "color": P.CLASSIFIER_COLOURS.get(m, P.GREY),
                    "dash": P.CLASSIFIER_DASH.get(m, "solid"),
                    "width": 2,
                },
                customdata=d[["threshold"]],
                hovertemplate=(
                    f"{m}<br>FPR %{{x:.2f}} · TPR %{{y:.2f}}<br>"
                    "threshold %{customdata[0]:.2f}<extra></extra>"
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="chance · AUC 0.500",
            line={"color": P.REFERENCE_COLOUR, "dash": "dot", "width": 1.5},
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(range=[0, 1], constrain="domain")
    fig.update_yaxes(range=[0, 1.01], scaleanchor="x", scaleratio=1)
    # legend in the empty lower-right corner (high FPR, low TPR) keeps the plot wide
    fig.update_layout(legend={"x": 0.98, "y": 0.02, "xanchor": "right", "yanchor": "bottom"})
    return VC.style(
        fig,
        title,
        "False positive rate (share of non-returners flagged)",
        "True positive rate (share of returners found)",
        height=480,
        legend_title="Model · test ROC-AUC",
    )


def pr_figure(pr: pd.DataFrame | None, cls_metrics: Mapping | None, base_rate: float) -> go.Figure:
    """Precision-recall per model (inf thresholds dropped) with the no-skill baseline."""
    title = "Precision-recall curves on the future window (test)"
    if pr is None or pr.empty:
        return VC.empty_figure("No precision-recall curves available.", title)
    models = (cls_metrics or {}).get("models") or {}
    best = best_model(cls_metrics, pr)
    pr = pr[np.isfinite(pr["threshold"])]
    fig = go.Figure()
    for m in model_order(pr["model"].unique()):
        d = thin(pr[pr["model"] == m].sort_values(["recall", "precision"], ascending=[False, True]))
        fig.add_trace(
            go.Scatter(
                x=d["recall"],
                y=d["precision"],
                mode="lines",
                name=_legend_name(m, "AP", (models.get(m) or {}).get("test_ap"), best),
                line={
                    "color": P.CLASSIFIER_COLOURS.get(m, P.GREY),
                    "dash": P.CLASSIFIER_DASH.get(m, "solid"),
                    "width": 2,
                },
                customdata=d[["threshold"]],
                hovertemplate=(
                    f"{m}<br>recall %{{x:.2f}} · precision %{{y:.2f}}<br>"
                    "threshold %{customdata[0]:.2f}<extra></extra>"
                ),
            )
        )
    if np.isfinite(base_rate):
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[base_rate, base_rate],
                mode="lines",
                name=f"no skill · AP {base_rate:.3f} (base rate)",
                line={"color": P.REFERENCE_COLOUR, "dash": "dot", "width": 1.5},
                hoverinfo="skip",
            )
        )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.01])
    # legend in the empty lower-left corner (low recall never has low precision here)
    fig.update_layout(legend={"x": 0.02, "y": 0.02, "xanchor": "left", "yanchor": "bottom"})
    return VC.style(
        fig,
        title,
        "Recall (share of returners found)",
        "Precision (share of flagged who returned)",
        height=480,
        legend_title="Model · test PR-AUC (AP)",
    )


# ------------------------------------------------------------------ model table
def model_table(cls_metrics: Mapping | None) -> pd.DataFrame:
    """One row per classifier: CV AUC ± sd, test AUC, AP, accuracy, threshold, best marker."""
    cls_metrics = cls_metrics or {}
    models = cls_metrics.get("models") or {}
    best = best_model(cls_metrics)
    rows = []
    for m in model_order(models):
        v = models[m] or {}
        mean, sd = v.get("cv_auc_mean"), v.get("cv_auc_std")
        rows.append(
            {
                "Model": m + (f" (k = {v['k']})" if v.get("k") else ""),
                "Best (train CV)": "★" if m == best else "",
                "CV ROC-AUC": (
                    f"{mean:.3f} ± {sd:.3f}"
                    if mean is not None and sd is not None
                    else (f"{mean:.3f}" if mean is not None else "n/a")
                ),
                "Test ROC-AUC": v.get("test_auc"),
                "Test PR-AUC (AP)": v.get("test_ap"),
                "Accuracy": v.get("accuracy"),
                "Threshold": model_threshold(cls_metrics, m),
            }
        )
    return pd.DataFrame(rows)


def tie_note(cls_metrics: Mapping | None) -> str | None:
    """Plain-English verdict from the numbers: are the models practically tied?"""
    models = (cls_metrics or {}).get("models") or {}
    test = {m: v.get("test_auc") for m, v in models.items() if v.get("test_auc") is not None}
    sds = [v.get("cv_auc_std") for v in models.values() if v.get("cv_auc_std") is not None]
    if len(test) < 2:
        return None
    lo_m, hi_m = min(test, key=test.get), max(test, key=test.get)
    spread = test[hi_m] - test[lo_m]
    text = (
        f"Test ROC-AUC ranges from {test[lo_m]:.3f} ({lo_m}) to {test[hi_m]:.3f} ({hi_m}), "
        f"a spread of {spread:.3f}"
    )
    if sds:
        sd = float(np.mean(sds))
        verdict = (
            "about the size of the fold-to-fold noise, so the models are practically tied: "
            "any of them ranks customers about equally well."
            if spread <= 2 * sd
            else "larger than the fold-to-fold noise, so the ranking of models is meaningful."
        )
        text += f"; the CV AUC varies by ±{sd:.3f} between folds, " + verdict
    else:
        text += "."
    best = best_model(cls_metrics)
    if best and best in test and best != hi_m:
        text += f" The model chosen on train CV ({best}) is not the best on test ({hi_m})."
    return text


# ------------------------------------------------------------------ confusion matrix
def confusion_counts(test: pd.DataFrame) -> pd.DataFrame:
    """2x2 counts: rows = actual (0, 1), columns = predicted (0, 1)."""
    y = test["returned_flag"].astype(int) if len(test) else pd.Series(dtype=int)
    p = test["pred_return"].astype(int) if len(test) else pd.Series(dtype=int)
    tab = pd.crosstab(y, p).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    tab.index.name, tab.columns.name = "actual", "predicted"
    return tab.astype(int)


def confusion_figure(test: pd.DataFrame, model: str | None, threshold: float | None) -> go.Figure:
    """Counts + row % (share of each actual class), sequential colour by row %."""
    thr = f" at threshold {threshold:.2f}" if threshold is not None else ""
    title = f"Confusion matrix · {model}{thr} (test)"
    if test is None or test.empty:
        return VC.empty_figure("No test predictions for this model.", title)
    counts = confusion_counts(test)
    rows = counts.sum(axis=1).replace(0, np.nan)
    share = counts.div(rows, axis=0).fillna(0.0)
    cells = [["True negative", "False positive"], ["False negative", "True positive"]]
    pred_labels = [f"Predicted: {s.lower()}" for s in OUTCOME_LABELS]
    act_labels = [f"Actual: {s.lower()}" for s in OUTCOME_LABELS]
    text = [
        [f"{counts.iat[i, j]:,}<br>({share.iat[i, j]:.0%} of row)" for j in (0, 1)] for i in (0, 1)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=share.to_numpy(),
            x=pred_labels,
            y=act_labels,
            zmin=0,
            zmax=1,
            colorscale=P.SEQUENTIAL,
            showscale=False,
            customdata=np.dstack([counts.to_numpy(), np.array(cells, dtype=object)]),
            hovertemplate=(
                "%{customdata[1]}<br>%{y} · %{x}<br>%{customdata[0]:,} customers"
                "<br>%{z:.1%} of the actual row<extra></extra>"
            ),
            xgap=2,
            ygap=2,
        )
    )
    for i in (0, 1):
        for j in (0, 1):
            fig.add_annotation(
                x=pred_labels[j],
                y=act_labels[i],
                text=text[i][j],
                showarrow=False,
                font={"color": P.SURFACE if share.iat[i, j] > 0.55 else P.INK, "size": 14},
            )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="bottom")
    return VC.style(
        fig, title, "Prediction", f"Actual outcome in the {HORIZON_DAYS} days", height=380
    )


# ------------------------------------------------------------------ KNN bias-variance
def knn_point(curve: pd.DataFrame, k: int) -> dict:
    """Train / test error at one k (nearest available k)."""
    if curve is None or curve.empty:
        return {}
    row = curve.iloc[int((curve["k"] - int(k)).abs().argmin())]
    return {k_: float(v) for k_, v in row.items() if pd.notna(v)}


def knn_figure(curve: pd.DataFrame | None, k_selected: int, k_chosen: int | None) -> go.Figure:
    """Train (in-sample) vs test error over k; selected k marked, CV-chosen k as a line."""
    title = "KNN bias-variance: error vs number of neighbours k"
    if curve is None or curve.empty or not {"k", "train_error", "test_error"} <= set(curve):
        return VC.empty_figure("No KNN curve available.", title)
    c = curve.sort_values("k")
    fig = go.Figure()
    series = (
        (
            "train_error",
            "train error (in-sample, past snapshot)",
            P.PERIOD_COLOURS["before"],
            "solid",
        ),
        ("test_error", "test error (future window)", P.PERIOD_COLOURS["after"], "dash"),
    )
    for col, name, colour, dash in series:
        fig.add_trace(
            go.Scatter(
                x=c["k"],
                y=c[col],
                mode="lines",
                name=name,
                line={"color": colour, "dash": dash, "width": 2},
                hovertemplate=f"k %{{x}}<br>{name.split(' (')[0]} %{{y:.1%}}<extra></extra>",
            )
        )
    pt = knn_point(c, k_selected)
    if pt:
        for col, _, colour, _ in series:
            fig.add_trace(
                go.Scatter(
                    x=[pt["k"]],
                    y=[pt[col]],
                    mode="markers",
                    marker={"color": colour, "size": 11, "line": P.marker_line(colour)},
                    showlegend=False,
                    hovertemplate=f"selected k %{{x}}<br>{col.replace('_', ' ')} %{{y:.1%}}"
                    "<extra></extra>",
                )
            )
        fig.add_annotation(
            x=pt["k"],
            y=max(pt["train_error"], pt["test_error"]),
            text=f"k = {int(pt['k'])}: train {pt['train_error']:.1%} · test {pt['test_error']:.1%}",
            showarrow=True,
            arrowcolor=P.MUTED_INK,
            ay=-40,
            font={"color": P.INK, "size": 12},
            bgcolor="rgba(255,255,255,0.85)",
        )
    if k_chosen is not None:
        fig.add_vline(
            x=int(k_chosen), line={"color": P.REFERENCE_COLOUR, "dash": "dot", "width": 1.5}
        )
        fig.add_annotation(
            x=int(k_chosen),
            y=0,
            yref="paper",
            yanchor="bottom",
            xanchor="left",
            showarrow=False,
            text=f" chosen k = {int(k_chosen)} (best train CV AUC)",
            font={"color": P.MUTED_INK, "size": 12},
        )
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_layout(legend={"orientation": "h", "y": -0.22}, hovermode="x unified")
    return VC.style(
        fig,
        title,
        "Number of neighbours k (small = flexible, large = smooth)",
        "Error rate (1 − accuracy)",
        height=420,
    )


# ------------------------------------------------------------------ regression
def _errors(y, pred) -> dict:
    y, pred = np.asarray(y, dtype="float64"), np.asarray(pred, dtype="float64")
    if not len(y):
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "rmse_gbp": np.nan}
    err = pred - y
    ss_tot = ((y - y.mean()) ** 2).sum()
    gbp = np.expm1(pred) - np.expm1(y)
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2": float(1 - (err**2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "rmse_gbp": float(np.sqrt((gbp**2).mean())),
    }


def regression_table(reg_metrics: Mapping | None, regression: pd.DataFrame | None) -> pd.DataFrame:
    """Test RMSE / MAE / R² (log1p scale) + £ RMSE per regressor and the mean-only baseline.

    Test errors are computed from regression.parquet (identical to metrics on the real data);
    CV RMSE and the baseline come from metrics, the baseline falls back to the train mean.
    """
    reg_metrics = reg_metrics or {}
    models = reg_metrics.get("models") or {}
    best = reg_metrics.get("best_model")
    rows = []
    names = (
        model_order(regression["model"].unique(), C.REGRESSORS) if regression is not None else []
    )
    names = names or model_order(models, C.REGRESSORS)
    for m in names:
        v = models.get(m) or {}
        if regression is not None and len(regression):
            t = regression[(regression["model"] == m) & (regression["split"] == "test")]
            err = _errors(t["y_true"], t["y_pred"])
        else:
            err = {k: v.get(k, np.nan) for k in ("rmse", "mae", "r2", "rmse_gbp")}
        rows.append(
            {
                "Model": m + (f" (k = {v['k']})" if v.get("k") else ""),
                "Best (train CV)": "★" if m == best else "",
                "CV RMSE (log)": v.get("cv_rmse_mean"),
                "Test RMSE (log)": err["rmse"],
                "Test MAE (log)": err["mae"],
                "Test R²": err["r2"],
                "Test RMSE (£)": err["rmse_gbp"],
            }
        )
    ref = reg_metrics.get("reference_mean")
    if not ref and regression is not None and len(regression):
        first = regression[regression["model"] == regression["model"].iloc[0]]
        tr, te = first[first["split"] == "train"], first[first["split"] == "test"]
        if len(tr) and len(te):
            ref = _errors(te["y_true"], np.full(len(te), tr["y_true"].mean()))
    if ref:
        rows.append(
            {
                "Model": "mean-only baseline",
                "Best (train CV)": "",
                "CV RMSE (log)": None,
                "Test RMSE (log)": ref.get("rmse"),
                "Test MAE (log)": ref.get("mae"),
                "Test R²": ref.get("r2"),
                "Test RMSE (£)": ref.get("rmse_gbp"),
            }
        )
    return pd.DataFrame(rows)


def _gbp_axis(fig: go.Figure, axis: str, lo: float, hi: float) -> None:
    """log1p(£) axis labelled in pounds (ticks at £1, £10, ... inside the data range)."""
    vals = [v for v in GBP_TICKS if np.log1p(v) >= lo - 0.5 and np.log1p(v) <= hi + 0.5]
    upd = {"tickvals": [float(np.log1p(v)) for v in vals], "ticktext": [VC.money(v) for v in vals]}
    (fig.update_xaxes if axis == "x" else fig.update_yaxes)(**upd)


def _test_rows(regression: pd.DataFrame | None, model: str | None) -> pd.DataFrame:
    if regression is None or model is None:
        return pd.DataFrame(columns=C.REGRESSION_COLUMNS)
    return regression[(regression["model"] == model) & (regression["split"] == "test")]


def pred_vs_actual_figure(regression: pd.DataFrame | None, model: str | None) -> go.Figure:
    """Predicted vs actual 90-day spend of test returners (log1p scale, £ ticks) + 45° line."""
    title = f"Predicted vs actual spend · {model} (test returners)"
    d = _test_rows(regression, model)
    if d.empty:
        return VC.empty_figure("No regression predictions for this model.", title)
    colour = P.REGRESSOR_COLOURS.get(model, P.GREY)
    lo = float(min(d["y_true"].min(), d["y_pred"].min()))
    hi = float(max(d["y_true"].max(), d["y_pred"].max()))
    fig = go.Figure(
        go.Scatter(
            x=d["y_pred"],
            y=d["y_true"],
            mode="markers",
            name=f"{model}: one customer",
            marker={"color": colour, "size": 6, "opacity": 0.55, "line": P.marker_line(colour)},
            customdata=np.column_stack([np.expm1(d["y_pred"]), np.expm1(d["y_true"])]),
            hovertemplate=(
                "predicted £%{customdata[0]:,.0f}<br>actual £%{customdata[1]:,.0f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="perfect prediction (45°)",
            line={"color": P.REFERENCE_COLOUR, "dash": "dash", "width": 1.5},
            hoverinfo="skip",
        )
    )
    _gbp_axis(fig, "x", lo, hi)
    _gbp_axis(fig, "y", lo, hi)
    fig.update_layout(legend={"orientation": "h", "y": -0.22})
    return VC.style(
        fig,
        title,
        f"Predicted {HORIZON_DAYS}-day spend (£, log scale)",
        f"Actual {HORIZON_DAYS}-day spend (£, log scale)",
        height=440,
    )


def residual_figure(regression: pd.DataFrame | None, model: str | None) -> go.Figure:
    """Residual (actual − predicted, log1p units) vs prediction, with the zero line."""
    title = f"Residuals · {model} (test returners)"
    d = _test_rows(regression, model)
    if d.empty:
        return VC.empty_figure("No regression predictions for this model.", title)
    colour = P.REGRESSOR_COLOURS.get(model, P.GREY)
    resid = d["y_true"] - d["y_pred"]
    fig = go.Figure(
        go.Scatter(
            x=d["y_pred"],
            y=resid,
            mode="markers",
            name=f"{model}: one customer",
            marker={"color": colour, "size": 6, "opacity": 0.55, "line": P.marker_line(colour)},
            customdata=np.column_stack([np.expm1(d["y_pred"]), np.expm1(d["y_true"])]),
            hovertemplate=(
                "predicted £%{customdata[0]:,.0f} · actual £%{customdata[1]:,.0f}"
                "<br>residual %{y:+.2f}: actual ≈ %{text}× predicted<extra></extra>"
            ),
            text=[f"{np.exp(r):.2f}" for r in resid],
        )
    )
    fig.add_hline(y=0, line={"color": P.REFERENCE_COLOUR, "dash": "dash", "width": 1.5})
    _gbp_axis(fig, "x", float(d["y_pred"].min()), float(d["y_pred"].max()))
    fig.update_layout(showlegend=False)
    return VC.style(
        fig,
        title,
        f"Predicted {HORIZON_DAYS}-day spend (£, log scale)",
        "Residual: log1p(actual) − log1p(predicted)",
        height=440,
    )
