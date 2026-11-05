from retailpulse.clean import clean
from retailpulse.warehouse import VIEWS, connect, summarize


def test_views_over_parquet(raw, tmp_path):
    res = clean(raw)
    sales_p, returns_p = tmp_path / "sales.parquet", tmp_path / "returns.parquet"
    res.sales.to_parquet(sales_p, index=False)
    res.returns.to_parquet(returns_p, index=False)

    con = connect(sales_p, returns_p)
    s = summarize(con)
    counts = s["row_counts"]
    assert set(counts) == set(VIEWS)
    assert counts["fact_sales"] == len(res.sales)
    assert counts["fact_returns"] == len(res.returns)
    assert counts["dim_customer"] == res.sales["customer_id"].nunique()
    assert counts["dim_product"] == res.sales["stock_code"].nunique()
    assert counts["dim_country"] == res.sales["country"].nunique()
    # dim_date covers every day between first and last sale
    dates = res.sales["invoice_date"].dt.normalize()
    span = dates.max() - dates.min()
    assert counts["dim_date"] == span.days + 1
    assert abs(s["total_revenue"] - res.sales["line_value"].sum()) < 0.01
    con.close()
