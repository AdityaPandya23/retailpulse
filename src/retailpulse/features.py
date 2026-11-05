"""Customer features at a cutoff (ONLY rows strictly before it) + labels from the horizon after.

Snapshots (written by the pipeline):
  features_cutoff  cutoff = config.CUTOFF            -> modelling: segmentation AND prediction,
                                                        so segment labels are leakage-free features
  labels           [cutoff, cutoff + horizon_days)   -> returned_flag, next_spend
  features_full    cutoff = last sale day + 1 day    -> descriptive EDA / stats only
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retailpulse import contracts as C

UK = "United Kingdom"
# Product StockCodes start with 5 digits; POST, M (manual), D (discount), CRUK, ADJUST, BANK
# CHARGES, ... are fees or adjustments, so they are excluded from product counts and returns.
PRODUCT_CODE = r"^\d{5}"


def is_product(stock_code: pd.Series) -> pd.Series:
    return stock_code.astype(str).str.match(PRODUCT_CODE).fillna(False).astype(bool)


def _before(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Rows strictly before the cutoff that belong to a known customer."""
    return df[(df["invoice_date"] < cutoff) & df["customer_id"].notna()]


def build_features(sales: pd.DataFrame, returns: pd.DataFrame, cutoff) -> pd.DataFrame:
    """One row per customer with >= 1 purchase before `cutoff`; uses ONLY rows before it.

    Day counts (recency, tenure, gaps) are measured between calendar days.
    """
    C.require_columns(sales, C.SALES_COLUMNS, "sales")
    C.require_columns(returns, C.RETURNS_COLUMNS, "returns")
    cutoff = pd.Timestamp(cutoff)
    ref_day = cutoff.normalize()

    s = _before(sales, cutoff)
    product = is_product(s["stock_code"])
    s = s.assign(
        day=s["invoice_date"].dt.normalize(),
        product_code=s["stock_code"].where(product),
        product_value=s["line_value"].where(product, 0.0),
        non_uk_value=s["line_value"].where(s["country"] != UK, 0.0),
    )
    f = s.groupby("customer_id").agg(
        first_day=("day", "min"),
        last_day=("day", "max"),
        n_days=("day", "nunique"),
        frequency=("invoice", "nunique"),
        monetary=("line_value", "sum"),
        n_distinct_products=("product_code", "nunique"),
        product_value=("product_value", "sum"),
        non_uk_value=("non_uk_value", "sum"),
    )

    r = _before(returns, cutoff)
    r = r[is_product(r["stock_code"])]
    returned = (-r["line_value"]).groupby(r["customer_id"]).sum()
    returned = returned.reindex(f.index, fill_value=0.0)

    tenure = (ref_day - f["first_day"]).dt.days
    span = (f["last_day"] - f["first_day"]).dt.days
    # mean gap between consecutive purchase days = span / (n_days - 1); a customer seen on a
    # single day has no gap yet: use the days since that purchase (a lower bound on the gap)
    mean_gap = (span / (f["n_days"] - 1)).where(f["n_days"] > 1, tenure)
    return_rate = (returned / f["product_value"]).where(f["product_value"] > 0, 0.0)

    out = pd.DataFrame(
        {
            "recency": (ref_day - f["last_day"]).dt.days.astype("int64"),
            "frequency": f["frequency"].astype("int64"),
            "monetary": f["monetary"].astype("float64"),
            "tenure": tenure.astype("int64"),
            "avg_basket_value": (f["monetary"] / f["frequency"]).astype("float64"),
            "mean_interpurchase_days": mean_gap.astype("float64"),
            "n_distinct_products": f["n_distinct_products"].astype("int64"),
            "return_rate": return_rate.clip(0.0, 1.0).astype("float64"),
            "non_uk_share": (f["non_uk_value"] / f["monetary"]).clip(0.0, 1.0).astype("float64"),
        },
        index=f.index,
    )
    out = out.rename_axis("customer_id").reset_index()
    out["customer_id"] = out["customer_id"].astype("int64")
    return C.require_columns(out, C.FEATURES_COLUMNS, "features")


def build_labels(sales: pd.DataFrame, cutoff, horizon_days: int) -> pd.DataFrame:
    """returned_flag / next_spend from [cutoff, cutoff + horizon_days).

    Population = customers with >= 1 purchase before the cutoff (same rows as build_features).
    """
    C.require_columns(sales, C.SALES_COLUMNS, "sales")
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=horizon_days)
    population = pd.Index(_before(sales, cutoff)["customer_id"].unique()).sort_values()

    window = sales[
        (sales["invoice_date"] >= cutoff)
        & (sales["invoice_date"] < end)
        & sales["customer_id"].notna()
    ]
    g = window.groupby("customer_id")
    n_invoices = g["invoice"].nunique().reindex(population, fill_value=0)
    spend = g["line_value"].sum().reindex(population, fill_value=0.0)
    out = pd.DataFrame(
        {
            "customer_id": np.asarray(population, dtype="int64"),
            "returned_flag": (n_invoices.to_numpy() > 0).astype("int64"),
            "next_spend": spend.to_numpy(dtype="float64"),
        }
    )
    return C.require_columns(out, C.LABELS_COLUMNS, "labels")


def full_cutoff(sales: pd.DataFrame) -> pd.Timestamp:
    """A cutoff that keeps every sale: the day after the last invoice day."""
    return sales["invoice_date"].max().normalize() + pd.Timedelta(days=1)


def log1p_skewed(features: pd.DataFrame, columns: list[str] = C.SKEWED_FEATURES) -> pd.DataFrame:
    """log1p on the right-skewed features (apply before StandardScaler in P3 models)."""
    out = features.copy()
    out[columns] = np.log1p(out[columns].astype("float64"))
    return out


def summarize(
    features_cutoff: pd.DataFrame,
    labels: pd.DataFrame,
    features_full: pd.DataFrame,
    cutoff,
    horizon_days: int,
    data_end: pd.Timestamp,
) -> dict:
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=horizon_days)
    rate = float(labels["returned_flag"].mean())
    returners = labels.loc[labels["returned_flag"] == 1, "next_spend"]
    mean_spend = float(returners.mean()) if len(returners) else 0.0
    median_spend = float(returners.median()) if len(returners) else 0.0
    n, n_full = len(features_cutoff), len(features_full)
    return {
        "cutoff": str(cutoff.date()),
        "horizon_days": int(horizon_days),
        "label_window": [str(cutoff.date()), str(end.date())],
        "label_window_complete": bool(end <= data_end),
        "data_end": str(data_end),
        "n_customers_cutoff": n,
        "n_customers_full": n_full,
        "n_new_after_cutoff": n_full - n,
        "n_returned": int(labels["returned_flag"].sum()),
        "returned_rate": round(rate, 4),
        "mean_next_spend_returners": round(mean_spend, 2),
        "median_next_spend_returners": round(median_spend, 2),
        "skewness_cutoff": {
            c: round(float(features_cutoff[c].skew()), 2) for c in C.FEATURE_COLUMNS
        },
        # evidence for contracts.SKEWED_FEATURES: skewness after log1p (closer to 0 = helped)
        "skewness_log1p_cutoff": {
            c: round(float(np.log1p(features_cutoff[c].astype("float64")).skew()), 2)
            for c in C.FEATURE_COLUMNS
        },
        "conclusion": (
            f"{rate:.1%} of the {n:,} customers who bought before {cutoff.date()} bought again "
            f"in the next {horizon_days} days (median spend of those who did: £{median_spend:,.0f})"
            f"; {n_full - n:,} customers first appear after the cutoff and are outside the "
            "modelling set."
        ),
    }
