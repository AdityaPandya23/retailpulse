"""Download UCI Online Retail II, combine both xlsx sheets into one Parquet file.

Also records the xlsx-vs-parquet storage format comparison (file size + load time).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from retailpulse import contracts as C
from retailpulse.config import DATA_URL, XLSX_NAME

log = logging.getLogger(__name__)

COLUMN_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def download(url: str = DATA_URL, zip_path: Path = C.RAW_ZIP, xlsx_path: Path = C.RAW_XLSX) -> Path:
    """Download the dataset zip once and extract only the xlsx member (by basename)."""
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    if not xlsx_path.exists():
        if not zip_path.exists():
            log.info("Downloading %s", url)
            req = urllib.request.Request(url, headers={"User-Agent": "retailpulse/0.1"})
            with urllib.request.urlopen(req, timeout=300) as resp, open(zip_path, "wb") as fh:
                fh.write(resp.read())
        with zipfile.ZipFile(zip_path) as zf:
            member = next(n for n in zf.namelist() if Path(n).name == XLSX_NAME)
            with zf.open(member) as src, open(xlsx_path, "wb") as dst:
                dst.write(src.read())
    return xlsx_path


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to snake_case and enforce stable dtypes (xlsx gives mixed int/str codes)."""
    df = df.rename(columns=COLUMN_MAP)
    C.require_columns(df, C.RAW_COLUMNS, "raw")
    out = pd.DataFrame(
        {
            "invoice": df["invoice"].astype(str).str.strip(),
            "stock_code": df["stock_code"].astype(str).str.strip().str.upper(),
            "description": df["description"].astype("string").str.strip(),
            "quantity": pd.to_numeric(df["quantity"]).astype("int64"),
            "invoice_date": pd.to_datetime(df["invoice_date"]),
            "price": pd.to_numeric(df["price"]).astype("float64"),
            "customer_id": pd.to_numeric(df["customer_id"]).astype("Int64"),
            "country": df["country"].astype(str).str.strip(),
        }
    )
    out["description"] = out["description"].replace("", pd.NA)
    return out


def _mb(path: Path) -> float:
    return round(path.stat().st_size / 1e6, 2)


def ingest(xlsx_path: Path | None = None, out_path: Path = C.RAW_PARQUET) -> dict:
    """xlsx (both sheets) -> one parquet. Returns the format-comparison record."""
    xlsx_path = xlsx_path or download()

    t0 = time.perf_counter()
    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    xlsx_load_s = time.perf_counter() - t0
    sheet_rows = {name: len(s) for name, s in sheets.items()}
    log.info("Read xlsx in %.1fs: %s", xlsx_load_s, sheet_rows)

    df = standardize(pd.concat(sheets.values(), ignore_index=True))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # best of 3: the first read also pays pyarrow's one-off import/warm-up cost
    timings = []
    for _ in range(3):
        t0 = time.perf_counter()
        reread = pd.read_parquet(out_path)
        timings.append(time.perf_counter() - t0)
    parquet_load_s = min(timings)

    record = {
        "rows": len(reread),
        "sheets": sheet_rows,
        "date_min": str(reread["invoice_date"].min()),
        "date_max": str(reread["invoice_date"].max()),
        "xlsx_mb": _mb(xlsx_path),
        "parquet_mb": _mb(out_path),
        "xlsx_load_s": round(xlsx_load_s, 2),
        "parquet_load_s": round(parquet_load_s, 3),
    }
    record["size_ratio"] = round(record["xlsx_mb"] / record["parquet_mb"], 1)
    record["speedup"] = round(xlsx_load_s / parquet_load_s, 1)

    C.FORMAT_COMPARISON_JSON.parent.mkdir(parents=True, exist_ok=True)
    C.FORMAT_COMPARISON_JSON.write_text(json.dumps(record, indent=2))
    return record
