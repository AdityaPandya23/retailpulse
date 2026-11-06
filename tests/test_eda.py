import numpy as np
import pandas as pd
import pytest

from retailpulse import contracts as C
from retailpulse import eda
from retailpulse.features import build_features, full_cutoff


@pytest.fixture(scope="module")
def eda_out(synthetic_shop):
    s = synthetic_shop
    full = build_features(s.sales, s.returns, full_cutoff(s.sales))
    tables, summary = eda.run(full, s.sales)
    return full, tables, summary


def test_tables_match_contracts(eda_out):
    _, tables, _ = eda_out
    assert set(tables) == set(C.EDA_OUTPUTS)
    for path, df in tables.items():
        C.validate(df, path)
        assert len(df) > 0


def test_histograms_count_every_customer(eda_out):
    full, tables, _ = eda_out
    hist = tables[C.EDA_HISTOGRAMS_PARQUET]
    counts = hist.groupby("feature")["count"].sum()
    assert (counts == len(full)).all() and set(counts.index) == set(C.FEATURE_COLUMNS)
    assert set(hist.loc[hist["feature"] == "monetary", "scale"]) == {"log1p"}
    assert set(hist.loc[hist["feature"] == "non_uk_share", "scale"]) == {"linear"}


def test_boxplot_order(eda_out):
    full, tables, _ = eda_out
    box = tables[C.EDA_BOXPLOTS_PARQUET]
    cols = ["min", "lower_whisker", "q1", "median", "q3", "upper_whisker", "max"]
    assert (np.diff(box[cols].to_numpy(), axis=1) >= -1e-9).all()
    assert set(box["group"]) == set(eda.BOX_GROUPS)
    assert (box.loc[box["group"] == "all", "n"] == len(full)).all()


def test_monthly_and_countries(eda_out, synthetic_shop):
    sales = synthetic_shop.sales
    _, tables, summary = eda_out
    monthly = tables[C.EDA_MONTHLY_PARQUET]
    assert monthly["revenue"].sum() == pytest.approx(sales["line_value"].sum())
    assert monthly["orders"].sum() == sales["invoice"].nunique()
    assert monthly["new_customers"].sum() == sales["customer_id"].nunique()
    assert monthly["month"].is_monotonic_increasing
    assert not monthly["is_partial"].iloc[1:-1].any()  # only the first/last month can be cut
    assert bool(monthly["is_partial"].iloc[-1]) is True  # the shop data stops on 9 Dec 2011
    countries = tables[C.EDA_COUNTRIES_PARQUET]
    assert countries["revenue_share"].sum() == pytest.approx(1.0)
    assert countries["revenue"].is_monotonic_decreasing
    assert summary["uk_revenue_share"] == pytest.approx(
        countries.loc[countries["is_uk"], "revenue_share"].sum()
    )


def test_pareto_curve(eda_out):
    full, tables, summary = eda_out
    p = tables[C.EDA_PARETO_PARQUET]
    assert p.iloc[0].tolist() == [0.0, 0.0]
    assert p.iloc[-1].tolist() == pytest.approx([1.0, 1.0])
    assert p["revenue_share"].is_monotonic_increasing
    assert (p["revenue_share"] >= p["customer_share"] - 1e-12).all()  # top customers first
    assert 0 <= summary["gini"] <= 1
    top20 = full["monetary"].nlargest(int(np.ceil(0.2 * len(full)))).sum() / full["monetary"].sum()
    assert summary["top_20pct_revenue_share"] == pytest.approx(top20)


def test_run_on_data_without_a_full_month(synthetic_shop):
    s = synthetic_shop.sales
    short = s[(s["invoice_date"] >= "2011-03-05") & (s["invoice_date"] < "2011-03-20")]
    full = build_features(short, synthetic_shop.returns, full_cutoff(short))
    _, summary = eda.run(full, short)  # every month is partial -> falls back to all months
    assert summary["peak_month"] == "2011-03-01"


def test_sample_rows_are_seeded_real_lines(eda_out, synthetic_shop):
    _, tables, _ = eda_out
    rows = tables[C.EDA_SAMPLE_ROWS_PARQUET]
    assert list(rows.columns) == C.SALES_COLUMNS and len(rows) == eda.N_SAMPLE_ROWS == 5
    assert rows["invoice_date"].is_monotonic_increasing
    sales = synthetic_shop.sales
    assert len(rows.merge(sales, on=C.SALES_COLUMNS)) >= 5  # unchanged lines of the input
    pd.testing.assert_frame_equal(rows, eda.sample_rows(sales))  # same rows every run


def test_gini_extremes():
    _, equal = eda.pareto(np.full(100, 5.0))
    assert equal["gini"] == pytest.approx(0.0, abs=1e-9)
    _, one = eda.pareto(np.r_[np.zeros(99), 1.0])
    assert one["gini"] == pytest.approx(0.99)


def test_cohorts(eda_out, synthetic_shop):
    _, tables, _ = eda_out
    c = tables[C.EDA_COHORTS_PARQUET]
    assert (c.loc[c["period"] == 0, "retention"] == 1.0).all()
    assert c["retention"].between(0, 1).all()
    assert (
        c.loc[c["period"] == 0, "cohort_size"].sum()
        == synthetic_shop.sales["customer_id"].nunique()
    )
    # every observable (cohort, period) cell exists, up to the last month of data
    last = synthetic_shop.sales["invoice_date"].max()
    cohort_ym = c["cohort"].dt.year * 12 + c["cohort"].dt.month
    per = c.groupby(cohort_ym)["period"].agg(["max", "count"])
    expected_max = (last.year * 12 + last.month) - per.index
    assert (per["max"] == expected_max).all() and (per["count"] == expected_max + 1).all()
