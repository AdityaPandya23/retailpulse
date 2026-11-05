"""Small aggregate tables for the EDA charts (data, not images).

histograms / frequency polygons, box plots, monthly sales & orders, country revenue,
customer revenue concentration (Pareto / Lorenz) and the cohort retention matrix.
Descriptive only: built from all sales and features_full.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from retailpulse import contracts as C
from retailpulse.features import UK

N_BINS = 30
PARETO_POINTS = 201  # customer share 0, 0.5%, ..., 100%
BOX_GROUPS = ("all", "one-time", "repeat")


def histograms(
    features: pd.DataFrame, columns: list[str] = C.FEATURE_COLUMNS, n_bins: int = N_BINS
) -> pd.DataFrame:
    """Counts per bin; skewed features get log1p-spaced bins (draw them on a log axis)."""
    parts = []
    for col in columns:
        x = features[col].to_numpy(dtype="float64")
        x = x[np.isfinite(x)]
        lo, hi = float(x.min()), float(x.max())
        hi = hi if hi > lo else lo + 1.0
        skewed = col in C.SKEWED_FEATURES
        if skewed:
            edges = np.expm1(np.linspace(np.log1p(lo), np.log1p(hi), n_bins + 1))
        else:
            edges = np.linspace(lo, hi, n_bins + 1)
        edges[0], edges[-1] = lo, hi  # exact ends so every value is counted
        if skewed:
            mids = np.expm1((np.log1p(edges[:-1]) + np.log1p(edges[1:])) / 2)
        else:
            mids = (edges[:-1] + edges[1:]) / 2
        counts, _ = np.histogram(x, bins=edges)
        parts.append(
            pd.DataFrame(
                {
                    "feature": col,
                    "bin": np.arange(n_bins),
                    "left": edges[:-1],
                    "right": edges[1:],
                    "mid": mids,
                    "count": counts.astype("int64"),
                    "scale": "log1p" if skewed else "linear",
                }
            )
        )
    return C.validate(pd.concat(parts, ignore_index=True), C.EDA_HISTOGRAMS_PARQUET)


def _box(x: np.ndarray) -> dict:
    q1, median, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    inside = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
    return {
        "n": int(len(x)),
        "min": float(x.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "lower_whisker": float(inside.min()),
        "upper_whisker": float(inside.max()),
        "n_outliers": int(len(x) - len(inside)),
    }


def boxplots(features: pd.DataFrame, columns: list[str] = C.FEATURE_COLUMNS) -> pd.DataFrame:
    """Precomputed box-plot statistics (1.5 IQR whiskers) per feature, overall and by type."""
    repeat = features["frequency"] > 1
    everyone = pd.Series(True, index=features.index)
    masks = dict(zip(BOX_GROUPS, [everyone, ~repeat, repeat], strict=True))
    rows = [
        {"feature": col, "group": group, **_box(features.loc[m, col].to_numpy(dtype="float64"))}
        for col in columns
        for group, m in masks.items()
        if m.any()
    ]
    return C.validate(pd.DataFrame(rows), C.EDA_BOXPLOTS_PARQUET)


def monthly(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales[["invoice", "invoice_date", "customer_id", "line_value"]]
    s = s.assign(month=s["invoice_date"].dt.to_period("M").dt.to_timestamp())
    out = s.groupby("month").agg(
        revenue=("line_value", "sum"),
        orders=("invoice", "nunique"),
        customers=("customer_id", "nunique"),
    )
    first_month = s.groupby("customer_id")["month"].min()
    out["new_customers"] = (
        first_month.value_counts().reindex(out.index, fill_value=0).astype("int64")
    )
    out["avg_order_value"] = out["revenue"] / out["orders"]
    start = s["invoice_date"].min().normalize()
    end = s["invoice_date"].max().normalize()
    month_end = out.index + pd.offsets.MonthEnd(0)
    out["is_partial"] = (out.index < start) | (month_end > end)
    out = out.rename_axis("month").reset_index()
    return C.validate(out[C.MONTHLY_COLUMNS], C.EDA_MONTHLY_PARQUET)


def countries(sales: pd.DataFrame) -> pd.DataFrame:
    out = (
        sales.groupby("country")
        .agg(
            revenue=("line_value", "sum"),
            orders=("invoice", "nunique"),
            customers=("customer_id", "nunique"),
        )
        .reset_index()
    )
    out["is_uk"] = out["country"] == UK
    out["revenue_share"] = out["revenue"] / out["revenue"].sum()
    out = out.sort_values("revenue", ascending=False, ignore_index=True)
    return C.validate(out[C.COUNTRY_COLUMNS], C.EDA_COUNTRIES_PARQUET)


def pareto(monetary, n_points: int = PARETO_POINTS) -> tuple[pd.DataFrame, dict]:
    """Cumulative revenue share of the top x% of customers + Gini and headline shares.

    (The Lorenz curve is the mirror image: L(p) = 1 - pareto(1 - p).)
    """
    x = np.sort(np.asarray(monetary, dtype="float64"))[::-1]
    n = len(x)
    cum = np.concatenate([[0.0], np.cumsum(x) / x.sum()])
    grid = np.linspace(0.0, 1.0, n_points)
    df = pd.DataFrame(
        {"customer_share": grid, "revenue_share": np.interp(grid, np.arange(n + 1) / n, cum)}
    )
    asc = x[::-1]
    gini = 2 * np.sum(np.arange(1, n + 1) * asc) / (n * asc.sum()) - (n + 1) / n
    summary = {"n_customers": int(n), "gini": float(gini)}
    for p in (0.01, 0.10, 0.20):
        summary[f"top_{round(p * 100)}pct_revenue_share"] = float(cum[int(np.ceil(p * n))])
    return C.validate(df, C.EDA_PARETO_PARQUET), summary


def cohorts(sales: pd.DataFrame) -> pd.DataFrame:
    """Share of each first-purchase-month cohort that buys again k months later.

    Every observable (cohort, period) cell is present (0 if nobody bought). The first cohort
    (Dec 2009) also holds customers who bought before the data starts (left-censored).
    """
    s = sales.loc[sales["customer_id"].notna(), ["customer_id", "invoice_date"]]
    ym = s["invoice_date"].dt.year * 12 + s["invoice_date"].dt.month - 1
    s = s.assign(ym=ym).drop_duplicates(["customer_id", "ym"])
    first = s.groupby("customer_id")["ym"].transform("min")
    s = s.assign(cohort_ym=first, period=s["ym"] - first)
    active = s.groupby(["cohort_ym", "period"])["customer_id"].nunique()
    last = int(s["ym"].max())
    cohort_ids = sorted(int(c) for c in s["cohort_ym"].unique())
    grid = pd.MultiIndex.from_tuples(
        [(c, p) for c in cohort_ids for p in range(last - c + 1)], names=["cohort_ym", "period"]
    )
    out = active.reindex(grid, fill_value=0).rename("active").reset_index()
    size = out.loc[out["period"] == 0].set_index("cohort_ym")["active"]
    out["cohort_size"] = out["cohort_ym"].map(size).astype("int64")
    out["retention"] = out["active"] / out["cohort_size"]
    out["cohort"] = pd.to_datetime(
        pd.DataFrame({"year": out["cohort_ym"] // 12, "month": out["cohort_ym"] % 12 + 1, "day": 1})
    )
    out["period"] = out["period"].astype("int64")
    out["active"] = out["active"].astype("int64")
    return C.validate(out[C.COHORT_COLUMNS], C.EDA_COHORTS_PARQUET)


def run(features_full: pd.DataFrame, sales: pd.DataFrame) -> tuple[dict[Path, pd.DataFrame], dict]:
    """All EDA tables (keyed by artifact path) + a headline summary for metrics.json."""
    pareto_df, conc = pareto(features_full["monetary"])
    monthly_df, countries_df = monthly(sales), countries(sales)
    tables = {
        C.EDA_HISTOGRAMS_PARQUET: histograms(features_full),
        C.EDA_BOXPLOTS_PARQUET: boxplots(features_full),
        C.EDA_MONTHLY_PARQUET: monthly_df,
        C.EDA_COUNTRIES_PARQUET: countries_df,
        C.EDA_PARETO_PARQUET: pareto_df,
        C.EDA_COHORTS_PARQUET: cohorts(sales),
    }
    full_months = monthly_df.loc[~monthly_df["is_partial"]]
    peak = full_months.loc[full_months["revenue"].idxmax()]
    uk_share = float(countries_df.loc[countries_df["is_uk"], "revenue_share"].sum())
    summary = {
        **conc,
        "uk_revenue_share": uk_share,
        "n_countries": int(len(countries_df)),
        "peak_month": str(peak["month"].date()),
        "peak_month_revenue": float(peak["revenue"]),
        "skewness_full": {c: round(float(features_full[c].skew()), 2) for c in C.FEATURE_COLUMNS},
        "conclusion": (
            f"Spend is highly concentrated: the top 20% of customers bring "
            f"{conc['top_20pct_revenue_share']:.0%} of revenue (top 1%: "
            f"{conc['top_1pct_revenue_share']:.0%}, Gini {conc['gini']:.2f}); the UK brings "
            f"{uk_share:.0%} of revenue and the peak month is {peak['month']:%B %Y}."
        ),
    }
    return tables, summary
