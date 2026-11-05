"""DuckDB star schema as views over the cleaned Parquet files (no data copied).

fact_sales, fact_returns, dim_customer, dim_product, dim_date, dim_country.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from retailpulse import contracts as C

VIEWS = ["fact_sales", "fact_returns", "dim_customer", "dim_product", "dim_date", "dim_country"]


def _sql_path(p: Path) -> str:
    return str(Path(p).resolve()).replace("\\", "/").replace("'", "''")


def connect(
    sales_path: Path = C.SALES_PARQUET,
    returns_path: Path = C.RETURNS_PARQUET,
    database: str = ":memory:",
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with all warehouse views registered."""
    con = duckdb.connect(database)
    fact_cols = """
        invoice, stock_code, description, quantity, invoice_date, price,
        customer_id, country, line_value,
        CAST(invoice_date AS DATE) AS date_key,
        year(invoice_date) AS year, month(invoice_date) AS month
    """
    con.execute(
        f"CREATE OR REPLACE VIEW fact_sales AS SELECT {fact_cols} "
        f"FROM read_parquet('{_sql_path(sales_path)}')"
    )
    con.execute(
        f"CREATE OR REPLACE VIEW fact_returns AS SELECT {fact_cols} "
        f"FROM read_parquet('{_sql_path(returns_path)}')"
    )
    con.execute("""
        CREATE OR REPLACE VIEW dim_customer AS
        SELECT customer_id,
               mode(country)               AS country,
               min(invoice_date)           AS first_purchase,
               max(invoice_date)           AS last_purchase,
               count(DISTINCT invoice)     AS n_invoices,
               sum(line_value)             AS total_spend
        FROM fact_sales GROUP BY customer_id
    """)
    con.execute("""
        CREATE OR REPLACE VIEW dim_product AS
        SELECT stock_code,
               mode(description)           AS description,
               median(price)               AS median_price,
               count(*)                    AS n_lines,
               count(DISTINCT customer_id) AS n_customers
        FROM fact_sales GROUP BY stock_code
    """)
    con.execute("""
        CREATE OR REPLACE VIEW dim_date AS
        WITH bounds AS (
            SELECT CAST(min(invoice_date) AS DATE) AS lo, CAST(max(invoice_date) AS DATE) AS hi
            FROM fact_sales
        ), days AS (
            SELECT CAST(unnest(generate_series(CAST(lo AS TIMESTAMP), CAST(hi AS TIMESTAMP),
                                               INTERVAL 1 DAY)) AS DATE) AS date_key
            FROM bounds
        )
        SELECT date_key,
               year(date_key)      AS year,
               quarter(date_key)   AS quarter,
               month(date_key)     AS month,
               monthname(date_key) AS month_name,
               weekofyear(date_key) AS week,
               dayofweek(date_key) AS day_of_week,
               dayname(date_key)   AS day_name
        FROM days
    """)
    con.execute("""
        CREATE OR REPLACE VIEW dim_country AS
        SELECT country,
               country = 'United Kingdom'  AS is_uk,
               count(DISTINCT customer_id) AS n_customers,
               count(DISTINCT invoice)     AS n_invoices
        FROM fact_sales GROUP BY country
    """)
    return con


def summarize(con: duckdb.DuckDBPyConnection) -> dict:
    """Small aggregates proving the schema works (written to artifacts/quality)."""
    counts = {v: con.execute(f"SELECT count(*) FROM {v}").fetchone()[0] for v in VIEWS}
    by_year = con.execute(
        "SELECT year, round(sum(line_value), 2) AS revenue, count(DISTINCT invoice) AS invoices "
        "FROM fact_sales GROUP BY year ORDER BY year"
    ).fetchall()
    top_countries = con.execute(
        "SELECT country, round(sum(line_value), 2) AS revenue FROM fact_sales "
        "GROUP BY country ORDER BY revenue DESC LIMIT 5"
    ).fetchall()
    total = con.execute("SELECT round(sum(line_value), 2) FROM fact_sales").fetchone()[0]
    returns_value = con.execute("SELECT round(sum(line_value), 2) FROM fact_returns").fetchone()[0]
    return {
        "row_counts": counts,
        "total_revenue": total,
        "returns_value": returns_value,
        "revenue_by_year": [{"year": y, "revenue": r, "invoices": n} for y, r, n in by_year],
        "top_countries": [{"country": c, "revenue": r} for c, r in top_countries],
    }


def build(out_path: Path = C.WAREHOUSE_SUMMARY_JSON) -> dict:
    con = connect()
    try:
        summary = summarize(con)
    finally:
        con.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    return summary
