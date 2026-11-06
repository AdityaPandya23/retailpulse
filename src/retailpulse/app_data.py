"""P4 app data: small per-customer tables the deployed app needs (it never reads data/).

timeline     one row per customer x active month x period for every features_cutoff customer:
             "before" = invoice_date < cutoff, "after" = [cutoff, cutoff + horizon), i.e. exactly
             the label window the models predicted (later rows are dropped). The cutoff month can
             hold one row per period.
rec_reasons  for every recommendation with source "rule": the strongest rule that fired for it
             (all antecedents bought before the cutoff, consequent contains the item), the
             customer's segment scope first, then "overall" (same logic as baskets.rules).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retailpulse import contracts as C
from retailpulse.baskets import rules as R
from retailpulse.config import CUTOFF, HORIZON_DAYS


def build_timeline(
    sales: pd.DataFrame, customer_ids, cutoff=CUTOFF, horizon_days: int = HORIZON_DAYS
) -> pd.DataFrame:
    """Monthly revenue (gross, like `monetary`) and distinct invoices per customer and period."""
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=horizon_days)
    ids = pd.Index(pd.Series(customer_ids).astype("int64").unique())
    s = sales[sales["customer_id"].notna() & (sales["invoice_date"] < end)]
    s = s[s["customer_id"].astype("int64").isin(ids)]
    df = pd.DataFrame(
        {
            "customer_id": s["customer_id"].astype("int64").to_numpy(),
            "month": s["invoice_date"].dt.to_period("M").dt.to_timestamp().to_numpy(),
            "period": np.where(s["invoice_date"] < cutoff, "before", "after"),
            "invoice": s["invoice"].astype(str).to_numpy(),
            "line_value": s["line_value"].astype("float64").to_numpy(),
        }
    )
    out = (
        df.groupby(["customer_id", "month", "period"], sort=True)
        .agg(revenue=("line_value", "sum"), orders=("invoice", "nunique"))
        .reset_index()
    )
    out["revenue"] = out["revenue"].round(2)
    out["orders"] = out["orders"].astype("int32")
    out["month"] = pd.to_datetime(out["month"]).astype("datetime64[ns]")
    out = out.sort_values(["customer_id", "month", "period"], ascending=[True, True, False])
    return C.validate(out[C.TIMELINE_COLUMNS].reset_index(drop=True), C.TIMELINE_PARQUET)


def recommendation_reasons(
    sales: pd.DataFrame,
    rules: pd.DataFrame,
    recs: pd.DataFrame,
    segments: pd.DataFrame | None = None,
    cutoff=CUTOFF,
) -> pd.DataFrame:
    """customer_id, stock_code -> the strongest fired rule behind a "rule" recommendation."""
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in C.REC_REASONS_COLUMNS})
    empty = empty.astype({"customer_id": "int64", "confidence": "float64", "lift": "float64"})
    need = {"antecedent_codes", "consequent_codes"}
    if rules is None or rules.empty or not need <= set(rules.columns) or recs.empty:
        return empty
    todo = recs if "source" not in recs.columns else recs[recs["source"] == "rule"]
    if todo.empty:
        return empty

    before = R.product_lines(sales, end=cutoff)
    bought = before.groupby("customer_id")["stock_code"].agg(set).to_dict()
    seg_of = {}
    if segments is not None and not segments.empty:
        seg_of = dict(
            zip(
                segments["customer_id"].astype("int64"),
                segments["segment_name"].astype(str),
                strict=True,
            )
        )

    # item -> candidate rules (scope, antecedent set, weight, row)
    by_item: dict[str, list[tuple[str, frozenset, float, int]]] = {}
    for i, r in enumerate(rules.itertuples(index=False)):
        ante = frozenset(c for c in str(r.antecedent_codes).split(",") if c)
        w = float(r.confidence) * float(r.lift)
        for item in (c for c in str(r.consequent_codes).split(",") if c):
            by_item.setdefault(item, []).append((str(r.scope), ante, w, i))

    rows = []
    for cid, code in zip(
        todo["customer_id"].astype("int64"), todo["stock_code"].astype(str), strict=True
    ):
        have = bought.get(int(cid), set())
        own = R.segment_scope(seg_of[int(cid)]) if int(cid) in seg_of else None
        fired = [
            (scope != own, -w, i)
            for scope, ante, w, i in by_item.get(code, [])
            if scope in (own, R.OVERALL) and ante <= have
        ]
        if not fired:
            continue
        r = rules.iloc[min(fired)[2]]
        rows.append(
            (
                int(cid),
                code,
                str(r["scope"]),
                str(r["antecedents"]),
                str(r["consequents"]),
                float(r["confidence"]),
                float(r["lift"]),
            )
        )
    if not rows:
        return empty
    out = pd.DataFrame(rows, columns=C.REC_REASONS_COLUMNS)
    return C.validate(out, C.REC_REASONS_PARQUET)


def summarize(timeline: pd.DataFrame, reasons: pd.DataFrame, recs: pd.DataFrame) -> dict:
    n_rule = int((recs["source"] == "rule").sum()) if "source" in recs.columns else len(recs)
    return {
        "timeline_rows": int(len(timeline)),
        "timeline_customers": int(timeline["customer_id"].nunique()),
        "timeline_periods": {
            str(k): int(v) for k, v in timeline["period"].value_counts().sort_index().items()
        },
        "rec_reasons_rows": int(len(reasons)),
        "rule_recommendations": n_rule,
        "conclusion": (
            f"Customer timeline: {len(timeline):,} customer-months for "
            f"{timeline['customer_id'].nunique():,} customers; {len(reasons):,} of {n_rule:,} "
            "rule-based recommendations have their firing rule stored for the app."
        ),
    }
