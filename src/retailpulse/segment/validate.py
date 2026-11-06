"""Segment validation: subsample stability (ARI), PCA, chi-square segment x UK/non-UK."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from retailpulse.config import ALPHA, SEED


def stability_ari(
    X: np.ndarray,
    k: int,
    n_subsamples: int = 10,
    frac: float = 0.8,
    seed: int = SEED,
    reference: np.ndarray | None = None,
) -> list[float]:
    """ARI between the full-data fit and fits on random subsamples (each predicts all rows).

    The subsamples come from np.random.default_rng(seed), so every k sees the same subsamples
    and the result is deterministic.
    """
    X = np.asarray(X)
    if reference is None:
        reference = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X).labels_
    rng = np.random.default_rng(seed)
    size = int(round(frac * len(X)))
    values = []
    for _ in range(n_subsamples):
        idx = rng.choice(len(X), size=size, replace=False)
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X[idx])
        values.append(float(adjusted_rand_score(reference, km.predict(X))))
    return values


def ari_summary(values: list[float]) -> dict:
    arr = np.asarray(values, dtype="float64")
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "values": [float(v) for v in arr]}


def pca_summary(X: np.ndarray, features: list[str], seed: int = SEED, n_coords: int = 3):
    """Full PCA -> (coords n x 3, explained-variance ratio list, loadings frame).

    Loadings are the unit eigenvectors (pca.components_), one row per clustering feature.
    """
    pca = PCA(random_state=seed).fit(X)
    scores = pca.transform(X)
    coords = np.zeros((len(X), n_coords))
    m = min(n_coords, scores.shape[1])
    coords[:, :m] = scores[:, :m]
    comps = np.zeros((len(features), n_coords))
    comps[:, :m] = pca.components_[:m].T
    loadings = pd.DataFrame(comps, columns=[f"pc{i + 1}" for i in range(n_coords)])
    loadings.insert(0, "feature", list(features))
    return coords, [float(v) for v in pca.explained_variance_ratio_], loadings


def _cramer_label(v: float) -> str:
    # Cohen's thresholds for a 2-column table (df* = 1): 0.1 small, 0.3 medium, 0.5 large
    if v >= 0.5:
        return "strong"
    if v >= 0.3:
        return "moderate"
    if v >= 0.1:
        return "weak"
    return "negligible"


def chi2_segment_uk(segment, non_uk_share, alpha: float = ALPHA) -> dict:
    """Chi-square test of independence segment x UK customer (non_uk_share < 0.5)."""
    is_uk = np.where(np.asarray(non_uk_share) < 0.5, "UK", "non-UK")
    table = pd.crosstab(np.asarray(segment), is_uk)
    uk_share = (table["UK"] / table.sum(axis=1)) if "UK" in table else table.sum(axis=1) * 0.0
    base = {"uk_share_by_segment": {int(s): float(v) for s, v in uk_share.items()}}
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {
            "chi2": np.nan,
            "dof": 0,
            "p_value": np.nan,
            "cramers_v": np.nan,
            **base,
            "conclusion": "Not testable: only one segment or only one of UK / non-UK present.",
        }
    res = stats.chi2_contingency(table.to_numpy())
    chi2, p, dof, expected = res.statistic, res.pvalue, res.dof, res.expected_freq
    n = table.to_numpy().sum()
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    low_expected = float((expected < 5).mean())
    verdict = (
        f"segment membership and UK/non-UK are associated (p = {p:.3g} < {alpha})"
        if p < alpha
        else f"no evidence that segment membership depends on UK/non-UK (p = {p:.3g})"
    )
    caveat = (
        f"; {low_expected:.0%} of expected counts are below 5, so treat p with care"
        if low_expected > 0.2
        else ""
    )
    return {
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "cramers_v": v,
        "share_expected_below_5": low_expected,
        **base,
        "conclusion": (
            f"Chi-square {chi2:.1f} (dof {dof}): {verdict}; the association is "
            f"{_cramer_label(v)} (Cramer's V {v:.2f}){caveat}."
        ),
    }
