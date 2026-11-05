"""Artifact contracts: every path the pipeline writes / the app reads, plus required columns.

P1 defines the data-layer contracts; later phases append feature/model artifacts here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from retailpulse.config import ARTIFACTS_DIR, PROCESSED_DIR, RAW_DIR, XLSX_NAME

# ---- data layer (large, never deployed) ----
RAW_ZIP = RAW_DIR / "online_retail_ii.zip"
RAW_XLSX = RAW_DIR / XLSX_NAME
RAW_PARQUET = RAW_DIR / "online_retail_ii.parquet"
SALES_PARQUET = PROCESSED_DIR / "sales.parquet"
RETURNS_PARQUET = PROCESSED_DIR / "returns.parquet"

# ---- small artifacts (app reads these) ----
QUALITY_DIR = ARTIFACTS_DIR / "quality"
FORMAT_COMPARISON_JSON = QUALITY_DIR / "format_comparison.json"
CLEANING_STEPS_JSON = QUALITY_DIR / "cleaning_steps.json"
MISSING_ID_JSON = QUALITY_DIR / "missing_id.json"
WAREHOUSE_SUMMARY_JSON = QUALITY_DIR / "warehouse_summary.json"

# ---- required columns ----
RAW_COLUMNS = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]
SALES_COLUMNS = [*RAW_COLUMNS, "line_value"]
RETURNS_COLUMNS = [*RAW_COLUMNS, "line_value"]


class ContractError(ValueError):
    """Raised when a frame does not satisfy its contract."""


def require_columns(df: pd.DataFrame, columns: list[str], name: str = "frame") -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ContractError(f"{name} is missing required columns: {missing}")
    return df


def all_exist(paths: list[Path]) -> bool:
    return all(Path(p).exists() for p in paths)
