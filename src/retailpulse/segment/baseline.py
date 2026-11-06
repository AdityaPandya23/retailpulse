"""Rule-based RFM-quartile baseline and the K-Means vs baseline comparison.

r/f/m scores are quartiles 1-4 (r_score 4 = most recent = lowest recency; f/m 4 = highest).
Ties are broken by row order (rank(method="first")) so every quartile has ~25% of customers.
Baseline groups = total RFM score (r + f + m) cut into the same k quantile groups as K-Means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from retailpulse.config import SEED


def quartile_score(values: pd.Series, reverse: bool = False) -> pd.Series:
    """Quartile 1-4 of `values` (4 = highest; reverse=True -> 4 = lowest)."""
    q = pd.qcut(values.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype("int64")
    return (5 - q) if reverse else q


def rfm_scores(features: pd.DataFrame) -> pd.DataFrame:
    """r_score, f_score, m_score (1-4 each) and rfm_score = their sum (3-12)."""
    out = pd.DataFrame(
        {
            "r_score": quartile_score(features["recency"], reverse=True),  # recent = 4
            "f_score": quartile_score(features["frequency"]),
            "m_score": quartile_score(features["monetary"]),
        },
        index=features.index,
    )
    out["rfm_score"] = out[["r_score", "f_score", "m_score"]].sum(axis=1).astype("int64")
    return out


def rfm_groups(rfm_score: pd.Series, k: int) -> np.ndarray:
    """Total RFM score cut into k quantile groups; group 0 = highest scores (like segment 0)."""
    q = pd.qcut(rfm_score.rank(method="first"), k, labels=False).astype("int64")
    return (k - 1 - q).to_numpy()


def cluster_quality(X: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette (higher better), Davies-Bouldin (lower better), Calinski-Harabasz (higher)."""
    if len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def groups_auc(groups, y, seed: int = SEED, folds: int = 5) -> tuple[float, float, list[float]]:
    """CV ROC-AUC of a logistic regression on the one-hot group -> (mean, std, per-fold)."""
    y = np.asarray(y).astype("int64")
    X = pd.DataFrame({"group": np.asarray(groups).astype(str)})
    counts = np.bincount(y, minlength=2)
    n_splits = min(folds, int(counts.min()))
    if n_splits < 2:
        return np.nan, np.nan, []
    model = Pipeline(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ("logit", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std(ddof=1)), [float(s) for s in scores]


def _compare_text(kq: dict, bq: dict) -> str:
    wins = sum(
        [
            kq["silhouette"] > bq["silhouette"],
            kq["davies_bouldin"] < bq["davies_bouldin"],
            kq["calinski_harabasz"] > bq["calinski_harabasz"],
        ]
    )
    sep = {3: "separates customers better than", 0: "separates customers worse than"}.get(
        wins, "is mixed against"
    )
    diff = kq["auc_returned"] - bq["auc_returned"]
    noise = max(kq["auc_returned_std"], bq["auc_returned_std"])
    if not np.isfinite(diff):
        pred = "the return AUC could not be computed"
    elif abs(diff) <= noise:
        pred = "predicts who buys again about as well (difference within one CV std)"
    else:
        pred = "predicts who buys again " + ("better" if diff > 0 else "worse")
    return (
        f"K-Means {sep} the RFM-quartile baseline on the same scaled features "
        f"(silhouette {kq['silhouette']:.2f} vs {bq['silhouette']:.2f}, "
        f"Davies-Bouldin {kq['davies_bouldin']:.2f} vs {bq['davies_bouldin']:.2f}, "
        f"Calinski-Harabasz {kq['calinski_harabasz']:.0f} vs {bq['calinski_harabasz']:.0f}) "
        f"and {pred}: CV ROC-AUC {kq['auc_returned']:.3f} vs {bq['auc_returned']:.3f}."
    )


def compare_baseline(
    X: np.ndarray,
    kmeans_labels,
    baseline_labels,
    returned_flag,
    seed: int = SEED,
    folds: int = 5,
) -> dict:
    """K-Means vs RFM-quartile groups: cluster quality in X + CV AUC on returned_flag."""
    out = {}
    for name, labels in (("kmeans", kmeans_labels), ("rfm_quartile", baseline_labels)):
        q = cluster_quality(X, np.asarray(labels))
        mean, std, folds_auc = groups_auc(labels, returned_flag, seed=seed, folds=folds)
        out[name] = {**q, "auc_returned": mean, "auc_returned_std": std, "auc_folds": folds_auc}
    out["conclusion"] = _compare_text(out["kmeans"], out["rfm_quartile"])
    return out
