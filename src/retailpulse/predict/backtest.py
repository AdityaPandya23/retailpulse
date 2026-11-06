"""Backtest: predictions made at the cutoff vs what actually happened in the horizon window."""

from __future__ import annotations

import numpy as np
import pandas as pd


def decile_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Customers ranked by P(return), cut into equal-size bins (1 = highest probability)."""
    y, p = np.asarray(y), np.asarray(p, dtype="float64")
    n_bins = int(min(n_bins, len(y)))
    order = np.argsort(-p, kind="stable")
    bins = np.empty(len(y), dtype="int64")
    bins[order] = np.arange(len(y)) * n_bins // len(y) + 1
    base = y.mean()
    t = (
        pd.DataFrame({"decile": bins, "y": y, "p": p})
        .groupby("decile")
        .agg(n=("y", "size"), mean_p=("p", "mean"), return_rate=("y", "mean"))
        .reset_index()
    )
    t["lift"] = t["return_rate"] / base if base > 0 else np.nan
    return t


def backtest(y, p, threshold: float, model: str, cutoff=None, horizon_days=None) -> dict:
    """Every P3_METRICS_KEYS["backtest"] key + lift, recall, precision and a decile table."""
    y, p = np.asarray(y), np.asarray(p, dtype="float64")
    pred = p >= threshold
    n_pred = int(pred.sum())
    share = float(y[pred].mean()) if n_pred else 0.0
    base = float(y.mean())
    n_ret = int(y.sum())
    recall = float(y[pred].sum() / n_ret) if n_ret else 0.0
    lift = share / base if base > 0 else float("nan")
    when = ""
    if cutoff is not None and horizon_days is not None:
        when = f" within {horizon_days} days of {pd.Timestamp(cutoff).date()}"
    return {
        "model": model,
        "threshold": float(threshold),
        "n_customers": int(len(y)),
        "n_predicted_return": n_pred,
        "n_actually_returned": n_ret,
        "share_actually_returned": share,
        "base_rate": base,
        "lift": lift,
        "precision": share,
        "recall": recall,
        "deciles": decile_table(y, p).to_dict(orient="records"),
        "conclusion": (
            f"Using {model} (threshold {threshold:.2f}), we predicted {n_pred:,} of {len(y):,} "
            f"customers would buy again{when}; {share:.0%} of them did vs a base rate of "
            f"{base:.0%} (lift {lift:.2f}), and they include {recall:.0%} of all who came back."
        ),
    }
