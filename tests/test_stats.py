import json

import numpy as np
import pandas as pd
import pytest

from retailpulse import contracts as C
from retailpulse.features import build_features, full_cutoff, log1p_skewed
from retailpulse.stats import distributions as D
from retailpulse.stats import tests as T


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def shop_full(synthetic_shop):
    s = synthetic_shop
    return build_features(s.sales, s.returns, full_cutoff(s.sales))


# ---------------- distributions ----------------
def test_lognormal_sample_prefers_lognormal(rng):
    entry, qq = D.fit_order_value(rng.lognormal(3.0, 0.8, 3000))
    assert entry["best_fit"] == "lognormal"
    params = entry["fits"]["lognormal"]["params"]
    assert params["mu_log"] == pytest.approx(3.0, abs=0.05)
    assert params["sigma_log"] == pytest.approx(0.8, abs=0.05)
    assert len(qq) == 2 * D.N_QQ and set(qq["dist"]) == {"lognormal", "normal"}
    for _, g in qq.groupby("dist"):
        assert g["prob"].is_monotonic_increasing and g["sample"].is_monotonic_increasing


def test_nbinom_mle_recovers_parameters(rng):
    r, p = D.nbinom_mle(rng.negative_binomial(2.0, 0.2, 5000))
    assert r == pytest.approx(2.0, rel=0.15)
    assert p == pytest.approx(0.2, abs=0.03)


def test_overdispersed_counts_prefer_nbinom(rng):
    frequency = 1 + rng.negative_binomial(0.7, 0.1, 3000)
    entry, qq, table = D.fit_frequency(frequency)
    assert entry["best_fit"] == "nbinom"
    assert entry["dispersion_ratio"] > 2
    assert table["k"].tolist() == list(range(D.MAX_K + 1))
    assert table["observed"].sum() + entry["tail_above_max_k"]["observed"] == len(frequency)
    assert set(qq["dist"]) == {"poisson", "nbinom"}


def test_poisson_counts_have_dispersion_near_one(rng):
    entry, _, _ = D.fit_frequency(1 + rng.poisson(3.0, 3000))
    assert entry["dispersion_ratio"] == pytest.approx(1.0, abs=0.15)
    # the NB adds nothing: AICs within the 2-point parameter penalty
    assert abs(entry["fits"]["poisson"]["aic"] - entry["fits"]["nbinom"]["aic"]) < 4


def test_exponential_gaps_give_weibull_shape_one(rng):
    entry, qq = D.fit_gaps(rng.exponential(30.0, 4000))
    assert entry["fits"]["weibull"]["params"]["shape"] == pytest.approx(1.0, abs=0.05)
    assert entry["fits"]["exponential"]["params"]["scale_days"] == pytest.approx(30, rel=0.05)
    assert entry["cv"] == pytest.approx(1.0, abs=0.1)
    assert set(qq["dist"]) == {"exponential", "weibull"}


def test_interpurchase_gaps_use_distinct_days():
    sales = pd.DataFrame(
        {
            "customer_id": pd.array([1, 1, 1, 1, 2, 3], dtype="Int64"),
            "invoice_date": pd.to_datetime(
                [
                    "2010-01-01 09:00",
                    "2010-01-01 17:00",  # same day -> no gap
                    "2010-01-11 10:00",
                    "2010-01-31 10:00",
                    "2010-05-05 10:00",  # single-day customer -> no gap
                    "2010-06-01 10:00",
                ]
            ),
        }
    )
    sales.loc[5, "customer_id"] = pd.NA  # unknown customer is ignored
    assert sorted(D.interpurchase_gaps(sales).tolist()) == [10.0, 20.0]


# ---------------- hypothesis tests ----------------
def test_compare_groups_detects_shift(rng):
    a, b = rng.lognormal(3.5, 0.5, 400), rng.lognormal(3.0, 0.5, 400)
    res = T.compare_groups(a, b, "big spenders", "small spenders", "spend", unit="£")
    assert res["mann_whitney"]["p_value"] < 0.001
    assert res["mann_whitney"]["rank_biserial"] > 0.3
    assert res["welch_t"]["cohens_d"] > 0.5
    assert res["tests_agree"] is True
    assert res["conclusion"].startswith("Big spenders have higher spend")


def test_compare_groups_same_distribution(rng):
    res = T.compare_groups(rng.normal(10, 2, 300), rng.normal(10, 2, 300), "A", "B", "x")
    assert res["mann_whitney"]["p_value"] > 0.05
    assert res["conclusion"].startswith("No significant difference")


def test_compare_groups_too_small():
    res = T.compare_groups([1.0], [2.0, 3.0], "A", "B", "x")
    assert "Not enough" in res["conclusion"] and "welch_t" not in res


def test_effect_labels():
    assert T.effect_label(0.05) == "negligible" and T.effect_label(-0.35) == "medium"
    assert T.effect_label(0.6, "d") == "medium" and T.effect_label(0.9, "d") == "large"


def _yearly_sales(rows):
    df = pd.DataFrame(rows, columns=["customer_id", "invoice_date", "line_value"])
    df["customer_id"] = df["customer_id"].astype("Int64")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def test_paired_spend_detects_increase(rng):
    y1 = (pd.Timestamp("2009-12-01"), pd.Timestamp("2010-12-01"))
    y2 = (pd.Timestamp("2010-12-01"), pd.Timestamp("2011-12-01"))
    rows = []
    for cid in range(1, 51):
        base = float(rng.uniform(50, 500))
        rows.append((cid, "2010-03-01", base))
        rows.append((cid, "2011-03-01", base + 20 + float(rng.normal(0, 5))))
    rows += [(51, "2010-05-01", 100.0), (52, "2011-05-01", 100.0)]  # one year only
    rows += [(1, "2011-12-05", 1e6)]  # after Y2: must be ignored
    res = T.paired_spend(_yearly_sales(rows), y1, y2)
    assert res["n_customers"] == 50
    assert res["n_year1_only"] == 1 and res["n_year2_only"] == 1
    assert res["mean_change"] == pytest.approx(20, abs=3)
    assert res["paired_t"]["p_value"] < 0.001 and res["wilcoxon"]["p_value"] < 0.001
    assert "spent more in year 2" in res["conclusion"] and res["tests_agree"] is True


def test_paired_spend_typical_drop_hidden_by_outliers():
    # 55 customers spend ~20 less; 5 swing wildly -> mean change ~0, median change clearly < 0
    y1 = (pd.Timestamp("2009-12-01"), pd.Timestamp("2010-12-01"))
    y2 = (pd.Timestamp("2010-12-01"), pd.Timestamp("2011-12-01"))
    swings = [3000.0, -1500.0, 2500.0, -2000.0, 1200.0]
    rows = []
    for cid in range(60):
        change = swings[cid - 55] if cid >= 55 else -20.0 - (cid % 5)
        rows += [(cid, "2010-06-01", 3000.0), (cid, "2011-06-01", 3000.0 + change)]
    res = T.paired_spend(_yearly_sales(rows), y1, y2)
    assert res["paired_t"]["p_value"] > 0.05 and res["wilcoxon"]["p_value"] < 0.001
    assert res["tests_agree"] is False
    assert res["conclusion"].startswith("No significant change in mean spend")
    assert "the typical customer spent less" in res["conclusion"]
    assert "median change -£22" in res["conclusion"]


# ---------------- correlation ----------------
def test_correlations_matrices(shop_full):
    entry, long = T.correlations(shop_full)
    k = len(C.FEATURE_COLUMNS)
    assert list(long.columns) == C.CORRELATION_COLUMNS and len(long) == 3 * k * k
    pearson = long[long["method"] == "pearson"].pivot(index="x", columns="y", values="value")
    assert np.allclose(np.diag(pearson.to_numpy()), 1.0)
    assert np.allclose(pearson.to_numpy(), pearson.to_numpy().T)
    # Spearman only uses ranks, so log1p does not change it; Pearson does change
    _, long_log = T.correlations(log1p_skewed(shop_full))
    sp = long[long["method"] == "spearman"]["value"].to_numpy()
    sp_log = long_log[long_log["method"] == "spearman"]["value"].to_numpy()
    assert np.allclose(sp, sp_log)
    assert entry["strongest_spearman"]["x"] in C.FEATURE_COLUMNS


# ---------------- whole stage on the synthetic shop ----------------
def test_run_entries_have_one_line_conclusions(synthetic_shop, shop_full):
    dist_entries, qq, table = D.run(shop_full, synthetic_shop.sales)
    test_entries, corr = T.run(shop_full, synthetic_shop.sales)
    entries = {**dist_entries, **test_entries}
    assert set(entries) == {
        "order_value_lognormal",
        "frequency_poisson_vs_nbinom",
        "gap_exponential",
        "basket_repeat_vs_onetime",
        "paired_spend_y1_y2",
        "correlation",
    }
    for name, e in entries.items():
        assert isinstance(e["conclusion"], str) and len(e["conclusion"]) > 20, name
        assert "\n" not in e["conclusion"], name
    assert entries["paired_spend_y1_y2"]["n_customers"] > 10
    C.validate(qq, C.QQ_PARQUET)
    C.validate(table, C.FREQUENCY_FIT_PARQUET)
    C.validate(corr, C.CORRELATION_PARQUET)
    json.dumps(C.to_jsonable(entries), allow_nan=False)  # metrics.json-safe
