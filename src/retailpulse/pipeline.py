"""Offline pipeline. Runs stages in order; skips a stage whose outputs all exist.

Stages: ingest -> clean -> warehouse -> features -> stats -> cube -> eda -> mock

Usage:
    python -m retailpulse.pipeline                 # run missing stages
    python -m retailpulse.pipeline --force         # rebuild everything
    python -m retailpulse.pipeline --only clean    # run selected stage(s) (forced)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from retailpulse import contracts as C
from retailpulse import cube as cube_mod
from retailpulse import eda as eda_mod
from retailpulse import features as features_mod
from retailpulse import ingest as ingest_mod
from retailpulse import mock as mock_mod
from retailpulse import warehouse as warehouse_mod
from retailpulse.clean import clean
from retailpulse.config import CUTOFF, HORIZON_DAYS
from retailpulse.stats import distributions
from retailpulse.stats import tests as stat_tests

log = logging.getLogger("retailpulse.pipeline")


@dataclass(frozen=True)
class Stage:
    name: str
    run: Callable[[], dict | None]
    outputs: tuple[Path, ...]
    metrics: tuple[str, ...] = ()  # metrics.json sections it writes (must exist to skip)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def _write_parquet(df: pd.DataFrame, path: Path, contract: Path | None = None) -> None:
    """Validate against the contract of `contract` (default: `path`), then write."""
    C.validate(df, contract or path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def run_ingest() -> dict:
    rec = ingest_mod.ingest()
    return {k: rec[k] for k in ("rows", "xlsx_mb", "parquet_mb", "xlsx_load_s", "parquet_load_s")}


def run_clean() -> dict:
    raw = pd.read_parquet(C.RAW_PARQUET)
    res = clean(raw)
    C.SALES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    res.sales.to_parquet(C.SALES_PARQUET, index=False)
    res.returns.to_parquet(C.RETURNS_PARQUET, index=False)
    _write_json(C.CLEANING_STEPS_JSON, {"raw_rows": len(raw), "steps": res.steps})
    _write_json(C.MISSING_ID_JSON, res.missing_id)
    return {"sales_rows": len(res.sales), "returns_rows": len(res.returns)}


def run_warehouse() -> dict:
    return warehouse_mod.build()["row_counts"]


def run_features() -> dict:
    sales = pd.read_parquet(C.SALES_PARQUET)
    returns = pd.read_parquet(C.RETURNS_PARQUET)
    at_cutoff = features_mod.build_features(sales, returns, CUTOFF)
    labels = features_mod.build_labels(sales, CUTOFF, HORIZON_DAYS)
    full = features_mod.build_features(sales, returns, features_mod.full_cutoff(sales))
    _write_parquet(at_cutoff, C.FEATURES_CUTOFF_PARQUET)
    _write_parquet(labels, C.LABELS_PARQUET)
    _write_parquet(full, C.FEATURES_FULL_PARQUET)
    summary = features_mod.summarize(
        at_cutoff, labels, full, CUTOFF, HORIZON_DAYS, sales["invoice_date"].max()
    )
    if not summary["label_window_complete"]:
        log.warning("label window %s runs past the data end", summary["label_window"])
    C.update_metrics("features", summary)
    return {k: summary[k] for k in ("n_customers_cutoff", "n_customers_full", "returned_rate")}


def run_stats() -> dict:
    full = pd.read_parquet(C.FEATURES_FULL_PARQUET)
    sales = pd.read_parquet(C.SALES_PARQUET, columns=["customer_id", "invoice_date", "line_value"])
    dist_entries, qq, freq_table = distributions.run(full, sales)
    test_entries, corr = stat_tests.run(full, sales)
    _write_parquet(qq, C.QQ_PARQUET)
    _write_parquet(freq_table, C.FREQUENCY_FIT_PARQUET)
    _write_parquet(corr, C.CORRELATION_PARQUET)
    entries = {**dist_entries, **test_entries}
    C.update_metrics("stats", entries)
    return {"entries": len(entries)}


def run_cube() -> dict:
    con = warehouse_mod.connect()
    try:
        cube = cube_mod.build_cube(con)
    finally:
        con.close()
    _write_parquet(cube, C.CUBE_PARQUET)
    return {"rows": len(cube), "levels": int(cube["level"].nunique())}


def run_eda() -> dict:
    full = pd.read_parquet(C.FEATURES_FULL_PARQUET)
    sales = pd.read_parquet(
        C.SALES_PARQUET, columns=["invoice", "invoice_date", "customer_id", "country", "line_value"]
    )
    tables, summary = eda_mod.run(full, sales)
    for path, df in tables.items():
        _write_parquet(df, path)
    C.update_metrics("eda", summary)
    return {"tables": len(tables), "gini": round(summary["gini"], 3)}


def run_mock() -> dict:
    tables, metrics = mock_mod.build_mock()
    for path, df in tables.items():
        _write_parquet(df, C.mock_path(path), contract=path)
    C.write_json(C.mock_path(C.METRICS_JSON), metrics)
    return {"files": len(tables) + 1}


STAGES: list[Stage] = [
    Stage("ingest", run_ingest, (C.RAW_PARQUET, C.FORMAT_COMPARISON_JSON)),
    Stage(
        "clean",
        run_clean,
        (C.SALES_PARQUET, C.RETURNS_PARQUET, C.CLEANING_STEPS_JSON, C.MISSING_ID_JSON),
    ),
    Stage("warehouse", run_warehouse, (C.WAREHOUSE_SUMMARY_JSON,)),
    Stage(
        "features",
        run_features,
        (C.FEATURES_CUTOFF_PARQUET, C.LABELS_PARQUET, C.FEATURES_FULL_PARQUET),
        ("features",),
    ),
    Stage(
        "stats",
        run_stats,
        (C.QQ_PARQUET, C.FREQUENCY_FIT_PARQUET, C.CORRELATION_PARQUET),
        ("stats",),
    ),
    Stage("cube", run_cube, (C.CUBE_PARQUET,)),
    Stage("eda", run_eda, C.EDA_OUTPUTS, ("eda",)),
    Stage("mock", run_mock, C.MOCK_OUTPUTS),
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="rebuild all stages")
    parser.add_argument("--only", nargs="+", choices=[s.name for s in STAGES], help="stages to run")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for stage in STAGES:
        if args.only and stage.name not in args.only:
            continue
        forced = args.force or bool(args.only)
        if not forced and C.all_exist(list(stage.outputs)) and C.has_metrics(stage.metrics):
            log.info("[skip] %s (outputs exist)", stage.name)
            continue
        t0 = time.perf_counter()
        log.info("[run ] %s", stage.name)
        info = stage.run()
        log.info("[done] %s in %.1fs %s", stage.name, time.perf_counter() - t0, info or "")


if __name__ == "__main__":
    main()
