"""Synthetic fixture mimicking UCI Online Retail II quirks. Never uses the real dataset.

Raw rows (18):
  8 clean sales lines
  1 exact duplicate of the first line
  2 returns (invoice starts with "C"), one with missing description
  3 invalid lines: negative qty (non-C), negative price ("A" bad-debt), zero price
  2 valid lines with missing Customer ID
  2 valid lines with missing Description (one known StockCode, one never described)
Expected: dedup -1 -> 17, returns -2 -> 15, invalid -3 -> 12, missing ID -2 -> 10 sales.

P2 adds two cleaned-schema fixtures: `feature_case` (hand-computed features / labels around a
cutoff) and `synthetic_shop` (seeded random customers for stats, EDA and cube tests).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def _cleaned(rows: list[tuple]) -> pd.DataFrame:
    """Rows -> the cleaned sales/returns schema (same dtypes as data/processed/*.parquet)."""
    df = standardize(pd.DataFrame(rows, columns=RAW_COLS))
    df["line_value"] = df["quantity"] * df["price"]
    return df


# ---------------- P2: hand-computed feature case (cutoff 2011-01-01, 90-day horizon) ----------
# label window = [2011-01-01, 2011-04-01).  Expected features / labels: tests/test_features.py
FEATURE_SALES = [
    # customer 1 (UK): 4 invoices on 3 days before the cutoff, 110 GBP of which 10 is postage
    ("500001", "10001", "P1", 2, "2010-01-10 10:00", 5.0, 1, UK),
    ("500001", "10002", "P2", 1, "2010-01-10 10:00", 20.0, 1, UK),
    ("500002", "10001", "P1", 2, "2010-01-10 15:00", 5.0, 1, UK),  # same day, 2nd invoice
    ("500003", "10003", "P3", 4, "2010-06-09 09:00", 5.0, 1, UK),
    ("500004", "10002", "P2", 2, "2010-12-01 12:00", 20.0, 1, UK),
    ("500004", "POST", "POSTAGE", 1, "2010-12-01 12:00", 10.0, 1, UK),  # not a product
    ("500005", "10001", "P1", 10, "2011-02-01 10:00", 5.0, 1, UK),  # in the label window
    ("500006", "10002", "P2", 1, "2011-05-01 10:00", 20.0, 1, UK),  # after the window
    # customer 2 (France): one-time buyer, never comes back
    ("500010", "10003", "P3", 10, "2010-11-01 10:00", 5.0, 2, "France"),
    # customer 3: UK + Germany (non-UK share 60/80), returns more than it bought
    ("500020", "10001", "P1", 4, "2010-03-01 10:00", 5.0, 3, UK),
    ("500021", "10002", "P2", 3, "2010-09-01 10:00", 20.0, 3, "Germany"),
    ("500022", "10001", "P1", 1, "2011-03-31 23:00", 5.0, 3, UK),  # last hours of the window
    # customer 4: first purchase after the cutoff -> not in features_cutoff / labels
    ("500030", "10001", "P1", 1, "2011-01-15 10:00", 5.0, 4, UK),
    # customer 5: bought the minute before the cutoff and exactly at the cutoff
    ("500040", "10003", "P3", 1, "2010-12-31 23:59", 5.0, 5, UK),
    ("500041", "10003", "P3", 2, "2011-01-01 00:00", 5.0, 5, UK),  # label window, not features
    # customer 6: comes back exactly when the window closes (excluded: window end is open)
    ("500050", "10001", "P1", 1, "2010-05-05 10:00", 5.0, 6, UK),
    ("500051", "10001", "P1", 1, "2011-04-01 00:00", 5.0, 6, UK),
]
FEATURE_RETURNS = [
    ("C500101", "10002", "P2", -1, "2010-02-01 10:00", 20.0, 1, UK),  # product return
    ("C500102", "M", "Manual", -1, "2010-07-01 10:00", 50.0, 1, UK),  # adjustment: ignored
    ("C500103", "10001", "P1", -2, "2011-02-15 10:00", 5.0, 1, UK),  # after the cutoff
    ("C500120", "10002", "P2", -5, "2010-10-01 10:00", 20.0, 3, "Germany"),  # 100 > 80 bought
    ("C500200", "10001", "P1", -1, "2010-07-01 10:00", 5.0, NA, UK),  # no customer id
    ("C500300", "10003", "P3", -1, "2010-08-01 10:00", 5.0, 7, UK),  # customer with no sales
]


@pytest.fixture
def feature_case() -> SimpleNamespace:
    return SimpleNamespace(
        sales=_cleaned(FEATURE_SALES),
        returns=_cleaned(FEATURE_RETURNS),
        cutoff=pd.Timestamp("2011-01-01"),
        horizon=90,
    )


# ---------------- P2: seeded synthetic shop for stats / EDA / cube tests ----------------
@pytest.fixture(scope="session")
def synthetic_shop() -> SimpleNamespace:
    """~300 synthetic customers over the real date range (cleaned schema). Never real data."""
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2009-12-01")
    n_days = (pd.Timestamp("2011-12-09") - start).days
    codes = [str(20000 + i) for i in range(40)]
    prices = np.round(rng.lognormal(1.0, 0.7, len(codes)), 2) + 0.01
    sales_rows, return_rows = [], []
    invoice = 600000
    for cid in range(1, 301):
        country = UK if rng.random() < 0.85 else str(rng.choice(["France", "Germany", "EIRE"]))
        first = int(rng.integers(0, n_days))
        days = np.sort(rng.integers(first, n_days + 1, 1 + int(rng.negative_binomial(1, 0.25))))
        days[0] = first
        for d in days:
            invoice += 1
            ts = start + pd.Timedelta(days=int(d), hours=int(rng.integers(8, 18)))
            for j in rng.choice(len(codes), int(rng.integers(1, 5)), replace=False):
                row = (str(invoice), codes[j], f"ITEM {j}", int(rng.integers(1, 25)), ts)
                sales_rows.append((*row, float(prices[j]), cid, country))
                if rng.random() < 0.05:
                    ret_ts = ts + pd.Timedelta(days=7)
                    ret = (f"C{invoice}", codes[j], f"ITEM {j}", -1, ret_ts, float(prices[j]))
                    return_rows.append((*ret, cid, country))
    return SimpleNamespace(sales=_cleaned(sales_rows), returns=_cleaned(return_rows))


# ---------------- P4 app: run pages on the MOCK artifacts (never the real data) ----------------
APP_DIR = Path(__file__).resolve().parents[1] / "app"


@pytest.fixture
def mock_artifacts(monkeypatch):
    """Point the app at artifacts/mock/ and make `components` importable."""
    from retailpulse import contracts as C

    if not C.all_exist(list(C.MOCK_OUTPUTS)):
        pytest.skip("artifacts/mock/ not generated (python -m retailpulse.pipeline --only mock)")
    monkeypatch.setenv("RETAILPULSE_ARTIFACTS", str(C.MOCK_DIR))
    monkeypatch.syspath_prepend(str(APP_DIR))
    return C.MOCK_DIR


@pytest.fixture
def run_app_page(mock_artifacts):
    """run_app_page("pages/3_Predictions.py") -> AppTest after its first run (mock data)."""
    from streamlit.testing.v1 import AppTest

    def _run(rel_path: str, timeout: float = 120):
        at = AppTest.from_file(str(APP_DIR / rel_path), default_timeout=timeout)
        at.run()
        return at

    return _run
