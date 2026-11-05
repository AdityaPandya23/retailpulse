"""Synthetic fixture mimicking UCI Online Retail II quirks. Never uses the real dataset.

Raw rows (18):
  8 clean sales lines
  1 exact duplicate of the first line
  2 returns (invoice starts with "C"), one with missing description
  3 invalid lines: negative qty (non-C), negative price ("A" bad-debt), zero price
  2 valid lines with missing Customer ID
  2 valid lines with missing Description (one known StockCode, one never described)
Expected: dedup -1 -> 17, returns -2 -> 15, invalid -3 -> 12, missing ID -2 -> 10 sales.
"""

from __future__ import annotations

import pandas as pd
import pytest

from retailpulse.ingest import standardize

NA = None
UK = "United Kingdom"

ROWS = [
    # Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
    ("100001", "85123A", "WHITE HEART", 6, "2010-01-04 10:00", 2.55, 1, UK),
    ("100001", "22423", "CAKESTAND", 2, "2010-01-04 10:00", 12.75, 1, UK),
    ("100002", "85123A", "WHITE HEART", 12, "2010-02-10 11:00", 2.55, 2, UK),
    ("100003", "85123A", "HEART", 4, "2010-03-01 09:30", 2.95, 3, "France"),
    ("100003", "84879", "BIRD ORNAMENT", 8, "2010-03-01 09:30", 1.69, 3, "France"),
    ("100004", "22423", "CAKESTAND", 1, "2010-05-20 14:00", 12.75, 4, "Germany"),
    ("100005", "84879", "BIRD ORNAMENT", 24, "2010-06-11 12:00", 1.45, 5, UK),
    ("100006", "22423", "CAKESTAND", 3, "2011-01-15 15:00", 12.75, 1, UK),
    # exact duplicate
    ("100001", "85123A", "WHITE HEART", 6, "2010-01-04 10:00", 2.55, 1, UK),
    # returns
    ("C100007", "85123A", "WHITE HEART", -2, "2010-02-01 08:00", 2.55, 1, UK),
    ("C100008", "22423", NA, -1, "2010-06-01 08:00", 12.75, 4, "Germany"),
    # invalid quantity / price
    ("100009", "84879", "BIRD ORNAMENT", -3, "2010-07-01 10:00", 1.69, 5, UK),
    ("A100010", "B", "Adjust bad debt", 1, "2011-08-12 14:50", -11062.06, NA, UK),
    ("100011", "22423", "?", 5, "2010-08-01 10:00", 0.0, NA, UK),
    # missing customer id
    ("100012", "85123A", "WHITE HEART", 6, "2010-09-01 10:00", 2.95, NA, UK),
    ("100012", "84879", "BIRD ORNAMENT", 12, "2010-09-01 10:00", 1.69, NA, UK),
    # missing description
    ("100013", "85123A", NA, 2, "2010-10-01 10:00", 2.55, 2, UK),
    ("100013", "99999", NA, 1, "2010-10-01 10:00", 5.00, 2, UK),
]

RAW_COLS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


@pytest.fixture
def raw_xlsx_like() -> pd.DataFrame:
    """As pandas would read it from the xlsx (original column names, mixed types)."""
    return pd.DataFrame(ROWS, columns=RAW_COLS)


@pytest.fixture
def raw(raw_xlsx_like: pd.DataFrame) -> pd.DataFrame:
    """Standardized raw frame, as stored in data/raw/*.parquet."""
    return standardize(raw_xlsx_like)
