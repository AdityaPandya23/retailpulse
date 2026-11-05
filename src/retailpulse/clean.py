"""Cleaning steps (PLAN.md) with per-step row counts, plus the missing Customer ID analysis.

Order:
 1. remove exact duplicate rows
 2. split invoices starting with "C" into a returns table
 3. drop Quantity <= 0 or Price <= 0 from sales
 4. analyse rows with missing Customer ID (MCAR vs MAR), then drop them
 5. impute missing Description with the most common Description of that StockCode
 line_value = quantity * price
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd
from scipy import stats

from retailpulse import contracts as C

UNKNOWN_DESCRIPTION = "UNKNOWN"


@dataclass
class CleanResult:
    sales: pd.DataFrame
    returns: pd.DataFrame
    steps: list[dict] = field(default_factory=list)
    missing_id: dict = field(default_factory=dict)


def _step(name: str, rows_in: int, rows_out: int, removed_to: str | None, **extra) -> dict:
    return {
        "step": name,
        "rows_in": int(rows_in),
        "rows_out": int(rows_out),
        "removed": int(rows_in - rows_out),
        "removed_to": removed_to,
        **extra,
    }


def description_lookup(df: pd.DataFrame) -> pd.Series:
    """Most common description per stock_code (ties broken alphabetically for determinism)."""
    counts = (
        df.dropna(subset=["description"])
        .groupby(["stock_code", "description"], observed=True)
        .size()
        .reset_index(name="n")
        .sort_values(["stock_code", "n", "description"], ascending=[True, False, True])
        .drop_duplicates("stock_code")
    )
    return counts.set_index("stock_code")["description"]


def impute_description(df: pd.DataFrame, lookup: pd.Series) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    missing = df["description"].isna()
    filled = df.loc[missing, "stock_code"].map(lookup).fillna(UNKNOWN_DESCRIPTION)
    df.loc[missing, "description"] = filled.astype("string")
    return df, int(missing.sum())


def _effect_label(x: float) -> str:
    x = abs(x)
    if x < 0.1:
        return "negligible"
    if x < 0.3:
        return "small"
    if x < 0.5:
        return "medium"
    return "large"


def missing_id_analysis(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Compare rows with vs without Customer ID.

    Mann-Whitney U on quantity and price (rank-biserial effect size), chi-square on
    UK vs non-UK (Cramer's V). With ~10^6 rows p-values are ~0, so effect sizes matter.
    """
    miss = df["customer_id"].isna()
    n_miss, n_ok = int(miss.sum()), int((~miss).sum())
    out: dict = {
        "n_rows": int(len(df)),
        "n_missing": n_miss,
        "share_missing": round(n_miss / len(df), 4) if len(df) else 0.0,
        "alpha": alpha,
        "tests": {},
    }
    if n_miss == 0 or n_ok == 0:
        out["verdict"] = "No comparison possible (one group is empty)."
        return out

    # Is missingness per whole invoice (e.g. guest checkout) or scattered within invoices?
    per_inv = df.assign(_m=miss).groupby("invoice")["_m"].mean()
    inv_any = per_inv[per_inv > 0]
    out["invoices_with_missing"] = int(len(inv_any))
    out["share_of_those_fully_missing"] = round(float((inv_any == 1).mean()), 4)

    for col in ["quantity", "price"]:
        a, b = df.loc[miss, col], df.loc[~miss, col]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
        r = 2 * u / (len(a) * len(b)) - 1  # rank-biserial = P(a>b) - P(a<b)
        out["tests"][f"mannwhitney_{col}"] = {
            "median_missing": float(a.median()),
            "median_present": float(b.median()),
            "mean_missing": round(float(a.mean()), 3),
            "mean_present": round(float(b.mean()), 3),
            "U": float(u),
            "p_value": float(p),
            "rank_biserial": round(float(r), 4),
            "effect": _effect_label(r),
        }

    uk = df["country"].eq("United Kingdom")
    table = pd.crosstab(miss, uk)
    if table.shape == (2, 2):
        chi2, p, dof, _ = stats.chi2_contingency(table)
        v = math.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))
        out["tests"]["chi2_uk"] = {
            "share_uk_missing": round(float(uk[miss].mean()), 4),
            "share_uk_present": round(float(uk[~miss].mean()), 4),
            "chi2": float(chi2),
            "dof": int(dof),
            "p_value": float(p),
            "cramers_v": round(v, 4),
            "effect": _effect_label(v),
        }

    out["verdict"] = _verdict(out)
    return out


def _verdict(res: dict) -> str:
    t = res["tests"]
    sig = [k for k, v in t.items() if v["p_value"] < res["alpha"]]
    if not sig:
        return (
            "No significant difference between rows with and without Customer ID: "
            "consistent with MCAR, so dropping them should not bias customer-level results."
        )
    parts = []
    q, pr = t.get("mannwhitney_quantity"), t.get("mannwhitney_price")
    if q:
        parts.append(
            f"median quantity {q['median_missing']:g} vs {q['median_present']:g} "
            f"({q['effect']} effect)"
        )
    if pr:
        parts.append(
            f"median price {pr['median_missing']:g} vs {pr['median_present']:g} "
            f"({pr['effect']} effect)"
        )
    if "chi2_uk" in t:
        c = t["chi2_uk"]
        parts.append(
            f"UK share {c['share_uk_missing']:.1%} vs {c['share_uk_present']:.1%} "
            f"({c['effect']} effect)"
        )
    return (
        "Missingness depends on observed variables (missing vs present: "
        + "; ".join(parts)
        + "), so it is NOT MCAR. It is consistent with MAR (e.g. guest / unregistered "
        "checkouts); MNAR cannot be ruled out from the data alone. Rows without an ID cannot "
        "be attributed to a customer, so they are dropped from customer-level analysis; "
        "results therefore describe registered customers only."
    )


def clean(raw: pd.DataFrame) -> CleanResult:
    C.require_columns(raw, C.RAW_COLUMNS, "raw")
    steps: list[dict] = []

    # 1. exact duplicates
    df = raw.drop_duplicates(ignore_index=True)
    steps.append(_step("remove_duplicates", len(raw), len(df), "Duplicates"))

    # 2. returns split
    is_return = df["invoice"].str.upper().str.startswith("C")
    returns = df[is_return].reset_index(drop=True)
    sales = df[~is_return].reset_index(drop=True)
    steps.append(_step("split_returns", len(df), len(sales), "Returns table"))

    # 3. non-positive quantity / price (adjustments, bad-debt entries, free samples)
    valid = (sales["quantity"] > 0) & (sales["price"] > 0)
    n_in = len(sales)
    sales = sales[valid].reset_index(drop=True)
    steps.append(_step("drop_nonpositive_qty_price", n_in, len(sales), "Invalid qty/price"))

    # 4. missing customer id: analyse first, then drop
    missing_id = missing_id_analysis(sales)
    lookup = description_lookup(sales)  # from valid sales lines, before dropping
    n_in = len(sales)
    sales = sales[sales["customer_id"].notna()].reset_index(drop=True)
    steps.append(_step("drop_missing_customer_id", n_in, len(sales), "Missing Customer ID"))

    # 5. impute description (no rows removed)
    sales, n_imp = impute_description(sales, lookup)
    returns, n_imp_ret = impute_description(returns, lookup)
    steps.append(
        _step(
            "impute_description",
            len(sales),
            len(sales),
            None,
            imputed_sales=n_imp,
            imputed_returns=n_imp_ret,
        )
    )

    for frame in (sales, returns):
        frame["line_value"] = frame["quantity"] * frame["price"]

    C.require_columns(sales, C.SALES_COLUMNS, "sales")
    C.require_columns(returns, C.RETURNS_COLUMNS, "returns")
    return CleanResult(sales=sales, returns=returns, steps=steps, missing_id=missing_id)
