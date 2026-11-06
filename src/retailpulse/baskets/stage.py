"""Basket stage: `python -m retailpulse.baskets.stage`.

Inputs : C.SALES_PARQUET, C.FEATURES_CUTOFF_PARQUET, C.SEGMENTS_PARQUET (used only when
         rules.per_segment is true; without it only the "overall" scope is mined),
         config.yaml (rules.*).
Outputs: C.RULES_PARQUET, C.RECOMMENDATIONS_PARQUET, C.PRODUCTS_PARQUET (top products =
         the overall rule universe); metrics.json section "rules".
`compute()` is pure (frames in, tables / metrics out); `run()` only does I/O.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from retailpulse import contracts as C
from retailpulse.baskets import rules as R
from retailpulse.config import CUTOFF, HORIZON_DAYS, RULES

log = logging.getLogger("retailpulse.baskets")

OUTPUTS = (C.RULES_PARQUET, C.RECOMMENDATIONS_PARQUET, C.PRODUCTS_PARQUET)
INPUTS = (C.SALES_PARQUET, C.FEATURES_CUTOFF_PARQUET, C.SEGMENTS_PARQUET)
METRICS_SECTIONS = ("rules",)
HIT_K = 5


def compute(
    sales: pd.DataFrame,
    customer_ids,
    segments: pd.DataFrame | None = None,
    cutoff=CUTOFF,
    horizon_days: int = HORIZON_DAYS,
    top_n_products: int = RULES.top_n_products,
    min_support: float = RULES.min_support,
    min_abs_invoices: int = RULES.min_abs_invoices,
    min_confidence: float = RULES.min_confidence,
    max_len: int = RULES.max_len,
    max_rules: int = RULES.max_rules_per_scope,
    n_recommend: int = RULES.n_recommend,
    per_segment: bool = RULES.per_segment,
) -> tuple[dict[Path, pd.DataFrame], dict]:
    """-> ({RULES_PARQUET, RECOMMENDATIONS_PARQUET, PRODUCTS_PARQUET}, metrics["rules"]).

    per_segment False: `segments` is ignored (overall rules + overall best sellers only).
    """
    cutoff = pd.Timestamp(cutoff)
    before = R.product_lines(sales, end=cutoff)
    window = R.product_lines(sales, start=cutoff, end=cutoff + pd.Timedelta(days=horizon_days))
    if not per_segment:
        segments = None
    if segments is not None:
        segments = segments[segments["customer_id"].isin(before["customer_id"].unique())]
    desc = R.descriptions(before)
    rules, infos = R.mine_all(
        before,
        segments,
        top_n_products=top_n_products,
        min_support=min_support,
        min_abs_invoices=min_abs_invoices,
        min_confidence=min_confidence,
        max_len=max_len,
        max_rules=max_rules,
    )
    recs = R.recommend_all(before, rules, customer_ids, segments, n_recommend, desc)
    baseline = R.recommend_all(before, rules.iloc[0:0], customer_ids, None, HIT_K, desc)
    hit = R.hit_rate(recs, before, window, HIT_K)
    pop_hit = R.hit_rate(baseline, before, window, HIT_K)

    chi = None
    overall = rules[rules["scope"] == R.OVERALL]
    if not overall.empty:
        top = overall.iloc[0]
        chi = R.rule_contingency(before, top["antecedent_codes"], top["consequent_codes"])
    metrics = R.summarize(rules, infos, chi, hit, pop_hit, min_confidence, HIT_K)
    metrics["n_recommend"] = int(n_recommend)
    metrics["hit_window"] = [
        str(cutoff.date()),
        str((cutoff + pd.Timedelta(days=horizon_days)).date()),
    ]

    metrics["per_segment"] = bool(per_segment)
    if not metrics["segments_used"]:
        metrics["recommendation_note"] = (
            "score = sum of confidence x lift of the fired overall rules; 'popular' fill items "
            "score their invoice support (overall best sellers)"
        )
    tables = {
        C.RULES_PARQUET: rules,
        C.RECOMMENDATIONS_PARQUET: recs,
        C.PRODUCTS_PARQUET: R.product_table(before, top_n_products, desc),
    }
    for path, df in tables.items():
        C.validate(df, path)
    missing = [k for k in C.P3_METRICS_KEYS["rules"] if k not in metrics]
    if missing:
        raise C.ContractError(f"metrics[rules] is missing keys: {missing}")
    return tables, metrics


def run() -> dict:
    t0 = time.perf_counter()
    sales = pd.read_parquet(
        C.SALES_PARQUET,
        columns=["invoice", "stock_code", "description", "invoice_date", "customer_id"],
    )
    customers = pd.read_parquet(C.FEATURES_CUTOFF_PARQUET, columns=["customer_id"])
    segments = None
    if not RULES.per_segment:
        log.info("rules.per_segment is false: overall rules and best sellers only")
    elif C.SEGMENTS_PARQUET.exists():
        segments = pd.read_parquet(
            C.SEGMENTS_PARQUET, columns=["customer_id", "segment", "segment_name"]
        )
    else:
        log.warning("no %s yet: mining the overall scope only", C.SEGMENTS_PARQUET.name)
    tables, metrics = compute(sales, customers["customer_id"], segments)
    for path, df in tables.items():
        C.validate(df, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    C.update_metrics("rules", metrics)
    return {
        "n_rules": metrics["n_rules"],
        "hit_rate_at_5": round(metrics["hit_rate_at_5"], 3),
        "popularity_hit_rate_at_5": round(metrics["popularity_hit_rate_at_5"], 3),
        "seconds": round(time.perf_counter() - t0, 1),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("baskets stage done: %s", run())
