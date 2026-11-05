"""Offline pipeline. Runs stages in order; skips a stage whose outputs all exist.

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
from retailpulse import ingest as ingest_mod
from retailpulse import warehouse as warehouse_mod
from retailpulse.clean import clean

log = logging.getLogger("retailpulse.pipeline")


@dataclass(frozen=True)
class Stage:
    name: str
    run: Callable[[], dict | None]
    outputs: tuple[Path, ...]


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


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


STAGES: list[Stage] = [
    Stage("ingest", run_ingest, (C.RAW_PARQUET, C.FORMAT_COMPARISON_JSON)),
    Stage(
        "clean",
        run_clean,
        (C.SALES_PARQUET, C.RETURNS_PARQUET, C.CLEANING_STEPS_JSON, C.MISSING_ID_JSON),
    ),
    Stage("warehouse", run_warehouse, (C.WAREHOUSE_SUMMARY_JSON,)),
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
        if not forced and C.all_exist(list(stage.outputs)):
            log.info("[skip] %s (outputs exist)", stage.name)
            continue
        t0 = time.perf_counter()
        log.info("[run ] %s", stage.name)
        info = stage.run()
        log.info("[done] %s in %.1fs %s", stage.name, time.perf_counter() - t0, info or "")


if __name__ == "__main__":
    main()
