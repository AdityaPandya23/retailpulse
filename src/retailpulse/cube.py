"""OLAP cube: GROUP BY CUBE(country, year, month) over the warehouse views.

All 2^3 = 8 grouping sets are precomputed because distinct counts (orders, customers) are not
additive: customers per country cannot be summed into a total, so roll-ups must come from SQL.
NULL in a dimension means "all"; grouping_id = SQL GROUPING() bitmask (bit set = rolled up).
"""

from __future__ import annotations

import duckdb
import pandas as pd

from retailpulse import contracts as C

CUBE_SQL = """
SELECT country, year, month,
       GROUPING(country, year, month) AS grouping_id,
       round(sum(line_value), 2)      AS revenue,
       count(DISTINCT invoice)        AS orders,
       count(DISTINCT customer_id)    AS customers
FROM fact_sales
GROUP BY CUBE (country, year, month)
"""


def level_name(grouping_id: int, dims: list[str] = C.CUBE_DIMENSIONS) -> str:
    """Grouping bitmask -> readable level, e.g. 3 -> "country", 7 -> "total"."""
    n = len(dims)
    kept = [d for i, d in enumerate(dims) if not (grouping_id >> (n - 1 - i)) & 1]
    return " × ".join(kept) if kept else "total"


def build_cube(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(CUBE_SQL).df()
    df["grouping_id"] = df["grouping_id"].astype("int64")
    df["level"] = df["grouping_id"].map(level_name)
    df["year"] = df["year"].astype("Int64")
    df["month"] = df["month"].astype("Int64")
    for col in ("orders", "customers"):
        df[col] = df[col].astype("int64")
    df = df.sort_values(["grouping_id", "country", "year", "month"], na_position="first")
    return C.validate(df[C.CUBE_COLUMNS].reset_index(drop=True), C.CUBE_PARQUET)
