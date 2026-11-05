"""Hypothesis tests and correlation, each with a one-line plain-English conclusion.

  repeat vs one-time customers' average basket value -> Welch t-test vs Mann-Whitney U
  same customers' spend in Y1 vs Y2                  -> paired t-test (+ Wilcoxon signed-rank)
  customer features (features_full)                  -> Pearson vs Spearman + covariance

compare_groups() is also what the app's live test picker calls (two segments + one feature).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from retailpulse import contracts as C
from retailpulse.config import ALPHA, YEAR1, YEAR2


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def effect_label(value: float, kind: str = "r") -> str:
    """Conventional size labels: kind "r" (rank-biserial, Cramér's V) or "d" (Cohen's d)."""
    small, medium, large = (0.1, 0.3, 0.5) if kind == "r" else (0.2, 0.5, 0.8)
    v = abs(value)
    if v < small:
        return "negligible"
    if v < medium:
        return "small"
    if v < large:
        return "medium"
    return "large"


def _fmt(x: float, unit: str) -> str:
    if unit == "£":
        return f"£{x:,.0f}" if abs(x) >= 100 else f"£{x:,.2f}"
    return f"{x:,.1f}{(' ' + unit) if unit else ''}"


def _signed_money(x: float) -> str:
    return f"{'-' if x < 0 else '+'}£{abs(x):,.0f}"


def _groups_disagree(u_sig: bool, t_sig: bool) -> str:
    """Why Mann-Whitney (ranks) and Welch (means) reached different verdicts."""
    if u_sig and not t_sig:
        return "disagrees: the ranks differ, but a few extreme values make the means too noisy."
    if t_sig and not u_sig:
        return "disagrees: only the means differ, pulled by a few extreme values."
    return "disagrees on the direction: a few extreme values pull the means the other way."


def _paired_disagree(w_sig: bool, t_sig: bool, median_change: float) -> str:
    """Why Wilcoxon (typical change) and the paired t-test (mean change) disagree."""
    if w_sig and not t_sig:
        return (
            f"disagrees: the typical customer spent {'more' if median_change > 0 else 'less'}, "
            "but a few very large changes make the mean change too noisy for the t-test."
        )
    if t_sig and not w_sig:
        return "disagrees: the mean change comes from a few large customers, not the typical one."
    return "disagrees on the direction: a few large customers pull the mean the other way."


def _describe(x: np.ndarray) -> dict:
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
    }


def _clean(x) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    return x[np.isfinite(x)]


def compare_groups(
    a,
    b,
    name_a: str = "group A",
    name_b: str = "group B",
    what: str = "value",
    unit: str = "",
    alpha: float = ALPHA,
) -> dict:
    """Welch t-test (compares means) vs Mann-Whitney U (compares ranks) + effect sizes."""
    a, b = _clean(a), _clean(b)
    out: dict = {"groups": {name_a: _describe(a), name_b: _describe(b)}, "alpha": alpha}
    if len(a) < 2 or len(b) < 2:
        out["conclusion"] = f"Not enough customers to compare {what} ({len(a)} vs {len(b)})."
        return out

    t = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0
    r = float(2 * u.statistic / (n1 * n2) - 1)  # rank-biserial = P(a > b) - P(a < b)
    out["welch_t"] = {
        "t": float(t.statistic),
        "df": float(t.df),
        "p_value": float(t.pvalue),
        "cohens_d": d,
        "effect": effect_label(d, "d"),
    }
    out["mann_whitney"] = {
        "U": float(u.statistic),
        "p_value": float(u.pvalue),
        "rank_biserial": r,
        "effect": effect_label(r, "r"),
    }

    u_sig, t_sig = u.pvalue < alpha, t.pvalue < alpha
    med_a, med_b = np.median(a), np.median(b)
    if u_sig:
        head = (
            f"{name_a} have {'higher' if r > 0 else 'lower'} {what} than {name_b} (median "
            f"{_fmt(med_a, unit)} vs {_fmt(med_b, unit)}; Mann-Whitney {fmt_p(u.pvalue)}, "
            f"r = {r:+.2f}, {effect_label(r)})"
        )
    else:
        head = (
            f"no significant difference in {what} between {name_a} and {name_b} (median "
            f"{_fmt(med_a, unit)} vs {_fmt(med_b, unit)}; Mann-Whitney {fmt_p(u.pvalue)})"
        )
    agree = (u_sig == t_sig) and (not u_sig or np.sign(d) == np.sign(r))
    tail = (
        f"; Welch's t-test on the means ({_fmt(a.mean(), unit)} vs {_fmt(b.mean(), unit)}, "
        f"{fmt_p(t.pvalue)}, d = {d:+.2f}) "
        + ("agrees." if agree else _groups_disagree(u_sig, t_sig))
    )
    out["tests_agree"] = bool(agree)
    out["conclusion"] = head[0].upper() + head[1:] + tail
    return out


def repeat_vs_onetime(features: pd.DataFrame, alpha: float = ALPHA) -> dict:
    repeat = features["frequency"] > 1
    res = compare_groups(
        features.loc[repeat, "avg_basket_value"],
        features.loc[~repeat, "avg_basket_value"],
        name_a="repeat customers",
        name_b="one-time customers",
        what="average basket value",
        unit="£",
        alpha=alpha,
    )
    return {
        "title": "Average basket value: repeat vs one-time customers (Welch t vs Mann-Whitney U)",
        "data": "features_full (repeat = 2+ invoices)",
        **res,
    }


def spend_in(sales: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Total spend per customer in [start, end)."""
    m = (
        (sales["invoice_date"] >= start)
        & (sales["invoice_date"] < end)
        & sales["customer_id"].notna()
    )
    return sales.loc[m].groupby("customer_id")["line_value"].sum()


def paired_spend(
    sales: pd.DataFrame,
    year1: tuple[pd.Timestamp, pd.Timestamp] = YEAR1,
    year2: tuple[pd.Timestamp, pd.Timestamp] = YEAR2,
    alpha: float = ALPHA,
) -> dict:
    """Paired t-test on the same customers' spend in Y1 vs Y2 (+ Wilcoxon signed-rank)."""
    y1, y2 = spend_in(sales, *year1), spend_in(sales, *year2)
    both = y1.index.intersection(y2.index)
    a = y1.loc[both].to_numpy(dtype="float64")
    b = y2.loc[both].to_numpy(dtype="float64")
    out: dict = {
        "title": "Same customers' spend, year 1 vs year 2 (paired t-test)",
        "data": "sales of customers with purchases in both years",
        "year1": [str(year1[0].date()), str((year1[1] - pd.Timedelta(days=1)).date())],
        "year2": [str(year2[0].date()), str((year2[1] - pd.Timedelta(days=1)).date())],
        "n_customers": int(len(both)),
        "n_year1_only": int(len(y1.index.difference(both))),
        "n_year2_only": int(len(y2.index.difference(both))),
        "alpha": alpha,
    }
    if len(both) < 2:
        out["conclusion"] = "Not enough customers active in both years for a paired test."
        return out

    diff = b - a
    t = stats.ttest_rel(b, a)
    w = stats.wilcoxon(diff)
    sd = diff.std(ddof=1)
    dz = float(diff.mean() / sd) if sd > 0 else 0.0
    med_diff = float(np.median(diff))
    out.update(
        {
            "mean_year1": float(a.mean()),
            "mean_year2": float(b.mean()),
            "median_year1": float(np.median(a)),
            "median_year2": float(np.median(b)),
            "mean_change": float(diff.mean()),
            "median_change": med_diff,
            "share_increased": float((diff > 0).mean()),
            "paired_t": {
                "t": float(t.statistic),
                "df": int(len(both) - 1),
                "p_value": float(t.pvalue),
                "cohens_dz": dz,
                "effect": effect_label(dz, "d"),
            },
            "wilcoxon": {"W": float(w.statistic), "p_value": float(w.pvalue)},
        }
    )
    t_sig, w_sig = t.pvalue < alpha, w.pvalue < alpha
    n = len(both)
    change = f"£{a.mean():,.0f} -> £{b.mean():,.0f}, {_signed_money(diff.mean())}"
    if t_sig:
        head = (
            f"The {n:,} customers active in both years spent {'more' if dz > 0 else 'less'} in "
            f"year 2 on average ({change}; paired t {fmt_p(t.pvalue)}, d_z = {dz:+.2f}, "
            f"{effect_label(dz, 'd')})"
        )
    else:
        head = (
            f"No significant change in mean spend for the {n:,} customers active in both years "
            f"({change}; paired t {fmt_p(t.pvalue)})"
        )
    agree = (t_sig == w_sig) and (not t_sig or np.sign(med_diff) == np.sign(dz))
    tail = (
        f"; the Wilcoxon signed-rank test (median change {_signed_money(med_diff)}, "
        f"{(diff > 0).mean():.0%} spent more, {fmt_p(w.pvalue)}) "
        + ("agrees." if agree else _paired_disagree(w_sig, t_sig, med_diff))
    )
    out["tests_agree"] = bool(agree)
    out["conclusion"] = head + tail
    return out


def correlations(
    features: pd.DataFrame, columns: list[str] = C.FEATURE_COLUMNS
) -> tuple[dict, pd.DataFrame]:
    """Pearson vs Spearman correlation and the covariance matrix (long format for heatmaps)."""
    X = features[columns].astype("float64")
    mats = {
        "pearson": X.corr(method="pearson"),
        "spearman": X.corr(method="spearman"),
        "covariance": X.cov(),
    }
    long = pd.concat(
        [
            m.rename_axis(index="x", columns="y")
            .stack()
            .rename("value")
            .reset_index()
            .assign(method=name)
            for name, m in mats.items()
        ],
        ignore_index=True,
    )[C.CORRELATION_COLUMNS]

    iu = np.triu_indices(len(columns), k=1)
    names = np.asarray(columns)
    pairs = pd.DataFrame(
        {
            "x": names[iu[0]],
            "y": names[iu[1]],
            "pearson": mats["pearson"].to_numpy()[iu],
            "spearman": mats["spearman"].to_numpy()[iu],
        }
    )
    pairs["gap"] = pairs["spearman"] - pairs["pearson"]
    strong = pairs.loc[pairs["spearman"].abs().idxmax()]
    gaps = pairs.reindex(pairs["gap"].abs().sort_values(ascending=False).index)
    worst = gaps.iloc[0]
    variances = np.diag(mats["covariance"].to_numpy())
    entry = {
        "title": "Customer features: Pearson vs Spearman correlation, covariance",
        "data": "features_full",
        "n": int(len(X)),
        "features": list(columns),
        "strongest_spearman": strong.to_dict(),
        "largest_pearson_spearman_gaps": gaps.head(3).round(4).to_dict("records"),
        "variances": dict(zip(columns, variances.tolist(), strict=True)),
        "conclusion": (
            f"Strongest monotonic link: {strong['x']}-{strong['y']} (Spearman ρ = "
            f"{strong['spearman']:.2f}, Pearson r = {strong['pearson']:.2f}); the two disagree "
            f"most on {worst['x']}-{worst['y']} (r = {worst['pearson']:.2f} vs ρ = "
            f"{worst['spearman']:.2f}) because a few extreme customers dominate Pearson, and "
            "covariances depend on units (monetary in £²), so models use log1p + standardised "
            "features."
        ),
    }
    return entry, long


def run(features_full: pd.DataFrame, sales: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """All tests -> (metrics entries, correlation/covariance long table)."""
    corr_entry, corr_long = correlations(features_full)
    entries = {
        "basket_repeat_vs_onetime": repeat_vs_onetime(features_full),
        "paired_spend_y1_y2": paired_spend(sales),
        "correlation": corr_entry,
    }
    return entries, C.validate(corr_long, C.CORRELATION_PARQUET)
