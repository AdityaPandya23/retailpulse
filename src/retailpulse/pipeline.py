"""Offline pipeline. Runs stages in order; skips a stage whose outputs exist and are newer
than its inputs (so `--only features` makes the next plain run refresh stats and eda).

Stages: ingest -> clean -> warehouse -> features -> stats -> cube -> eda -> mock
        -> segment -> predict (classify, regress, backtest) -> rules -> app_data

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

import joblib
import pandas as pd

from retailpulse import app_data as app_data_mod
from retailpulse import contracts as C
from retailpulse import cube as cube_mod
from retailpulse import eda as eda_mod
from retailpulse import features as features_mod
from retailpulse import ingest as ingest_mod
from retailpulse import mock as mock_mod
from retailpulse import warehouse as warehouse_mod
from retailpulse.baskets import stage as rules_stage
from retailpulse.clean import clean
from retailpulse.config import CONFIG_PATH, CUTOFF, HORIZON_DAYS
from retailpulse.predict import stage as predict_stage
from retailpulse.segment import stage as segment_stage
from retailpulse.stats import distributions
from retailpulse.stats import tests as stat_tests

log = logging.getLogger("retailpulse.pipeline")


@dataclass(frozen=True)
class Stage:
    name: str
    run: Callable[[], dict | None]
    outputs: tuple[Path, ...]
    metrics: tuple[str, ...] = ()  # metrics.json sections it writes (must exist to skip)
    inputs: tuple[Path, ...] = ()  # rerun when any input is newer than the oldest output


def up_to_date(stage: Stage) -> bool:
    """Outputs + metrics sections exist and no (existing) input is newer than any output."""
    if not C.all_exist(list(stage.outputs)) or not C.has_metrics(stage.metrics):
        return False
    inputs = [Path(p) for p in stage.inputs if Path(p).exists()]
    if not inputs:
        return True
    oldest_output = min(Path(p).stat().st_mtime for p in stage.outputs)
    return oldest_output >= max(p.stat().st_mtime for p in inputs)


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
    jsons = mock_mod.mock_quality_jsons()
    for path, obj in jsons.items():
        C.write_json(C.mock_path(path), obj)
    models = mock_mod.build_mock_models(tables, metrics)
    for path, payload in models.items():
        C.mock_path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, C.mock_path(path))
    C.write_json(C.mock_path(C.METRICS_JSON), metrics)
    return {"files": len(tables) + len(jsons) + len(models) + 1}


APP_DATA_OUTPUTS = (C.TIMELINE_PARQUET, C.REC_REASONS_PARQUET)


def run_app_data() -> dict:
    sales = pd.read_parquet(
        C.SALES_PARQUET,
        columns=["invoice", "stock_code", "description", "invoice_date", "customer_id"]
        + ["line_value"],
    )
    customers = pd.read_parquet(C.FEATURES_CUTOFF_PARQUET, columns=["customer_id"])
    rules = pd.read_parquet(C.RULES_PARQUET)
    recs = pd.read_parquet(C.RECOMMENDATIONS_PARQUET)
    segments = pd.read_parquet(C.SEGMENTS_PARQUET, columns=["customer_id", "segment_name"])
    timeline = app_data_mod.build_timeline(sales, customers["customer_id"], CUTOFF, HORIZON_DAYS)
    reasons = app_data_mod.recommendation_reasons(sales, rules, recs, segments, CUTOFF)
    _write_parquet(timeline, C.TIMELINE_PARQUET)
    _write_parquet(reasons, C.REC_REASONS_PARQUET)
    summary = app_data_mod.summarize(timeline, reasons, recs)
    summary["size_kb"] = {p.name: round(p.stat().st_size / 1024, 1) for p in APP_DATA_OUTPUTS}
    C.update_metrics("app_data", summary)
    return {"rows": summary["timeline_rows"], "size_kb": summary["size_kb"]}


_CLEANED = (C.SALES_PARQUET, C.RETURNS_PARQUET)

STAGES: list[Stage] = [
    Stage("ingest", run_ingest, (C.RAW_PARQUET, C.FORMAT_COMPARISON_JSON)),
    Stage(
        "clean",
        run_clean,
        (*_CLEANED, C.CLEANING_STEPS_JSON, C.MISSING_ID_JSON),
        inputs=(C.RAW_PARQUET,),
    ),
    Stage("warehouse", run_warehouse, (C.WAREHOUSE_SUMMARY_JSON,), inputs=_CLEANED),
    Stage(
        "features",
        run_features,
        (C.FEATURES_CUTOFF_PARQUET, C.LABELS_PARQUET, C.FEATURES_FULL_PARQUET),
        ("features",),
        inputs=(*_CLEANED, CONFIG_PATH),  # cutoff / horizon live in config.yaml
    ),
    Stage(
        "stats",
        run_stats,
        (C.QQ_PARQUET, C.FREQUENCY_FIT_PARQUET, C.CORRELATION_PARQUET),
        ("stats",),
        inputs=(C.FEATURES_FULL_PARQUET, C.SALES_PARQUET, CONFIG_PATH),  # alpha, Y1/Y2
    ),
    Stage("cube", run_cube, (C.CUBE_PARQUET,), inputs=_CLEANED),
    Stage(
        "eda",
        run_eda,
        C.EDA_OUTPUTS,
        ("eda",),
        inputs=(C.FEATURES_FULL_PARQUET, C.SALES_PARQUET),
    ),
    # synthetic, but the mock timeline / recommendations / features use the cutoff + horizon
    Stage("mock", run_mock, C.MOCK_OUTPUTS, inputs=(CONFIG_PATH,)),
    # ---- P3 models (k / names / hyper-parameters live in config.yaml) ----
    Stage(
        "segment",
        segment_stage.run,
        segment_stage.OUTPUTS,
        segment_stage.METRICS_SECTIONS,
        inputs=(*segment_stage.INPUTS, CONFIG_PATH),
    ),
    Stage(
        "predict",
        predict_stage.run,
        predict_stage.OUTPUTS,
        predict_stage.METRICS_SECTIONS,
        inputs=(*predict_stage.INPUTS, CONFIG_PATH),
    ),
    Stage(
        "rules",  # per-segment scopes: reruns whenever segments.parquet changes
        rules_stage.run,
        rules_stage.OUTPUTS,
        rules_stage.METRICS_SECTIONS,
        inputs=(*rules_stage.INPUTS, CONFIG_PATH),
    ),
    # ---- P4: small per-customer tables for the app (timeline, rule behind each rec) ----
    Stage(
        "app_data",
        run_app_data,
        APP_DATA_OUTPUTS,
        ("app_data",),
        inputs=(
            C.SALES_PARQUET,
            C.FEATURES_CUTOFF_PARQUET,
            C.RULES_PARQUET,
            C.RECOMMENDATIONS_PARQUET,
            C.SEGMENTS_PARQUET,
            CONFIG_PATH,
        ),
    ),
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
        if not forced and up_to_date(stage):
            log.info("[skip] %s (outputs up to date)", stage.name)
            continue
        t0 = time.perf_counter()
        log.info("[run ] %s", stage.name)
        info = stage.run()
        log.info("[done] %s in %.1fs %s", stage.name, time.perf_counter() - t0, info or "")


if __name__ == "__main__":
    main()
