import numpy as np
import pandas as pd
import pytest

from retailpulse import contracts as C
from retailpulse.clean import clean
from retailpulse.features import (
    build_features,
    build_labels,
    full_cutoff,
    is_product,
    log1p_skewed,
    summarize,
)

# hand-computed from tests/conftest.py FEATURE_SALES / FEATURE_RETURNS at cutoff 2011-01-01
EXPECTED = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 5, 6],
        "recency": [31, 61, 122, 1, 241],
        "frequency": [4, 1, 2, 1, 1],
        "monetary": [110.0, 50.0, 80.0, 5.0, 5.0],
        "tenure": [356, 61, 306, 1, 241],
        "avg_basket_value": [27.5, 50.0, 40.0, 5.0, 5.0],
        "mean_interpurchase_days": [162.5, 61.0, 184.0, 1.0, 241.0],  # 325 days / 2 gaps
        "n_distinct_products": [3, 1, 2, 1, 1],  # POST is not a product
        "return_rate": [0.2, 0.0, 1.0, 0.0, 0.0],  # 20 / 100 of products; 100/80 capped
        "non_uk_share": [0.0, 1.0, 0.75, 0.0, 0.0],
    }
)


def test_feature_values(feature_case):
    f = build_features(feature_case.sales, feature_case.returns, feature_case.cutoff)
    assert list(f.columns) == C.FEATURES_COLUMNS
    pd.testing.assert_frame_equal(f, EXPECTED, check_dtype=False)
    for col in ("customer_id", "recency", "frequency", "tenure", "n_distinct_products"):
        assert f[col].dtype == "int64"
    assert not f.isna().any().any()


def test_labels(feature_case):
    lab = build_labels(feature_case.sales, feature_case.cutoff, feature_case.horizon)
    assert list(lab.columns) == C.LABELS_COLUMNS
    assert lab["customer_id"].tolist() == [1, 2, 3, 5, 6]
    # 5 bought exactly at the cutoff (in), 6 exactly at the window end (out)
    assert lab["returned_flag"].tolist() == [1, 0, 1, 1, 0]
    assert lab["next_spend"].tolist() == [50.0, 0.0, 5.0, 10.0, 0.0]


def test_labels_align_with_features(feature_case):
    f = build_features(feature_case.sales, feature_case.returns, feature_case.cutoff)
    lab = build_labels(feature_case.sales, feature_case.cutoff, feature_case.horizon)
    assert lab["customer_id"].tolist() == f["customer_id"].tolist()


def _perturb_after(df: pd.DataFrame, cutoff: pd.Timestamp, rng) -> pd.DataFrame:
    df = df.copy()
    after = df["invoice_date"] >= cutoff
    df.loc[after, "quantity"] = df.loc[after, "quantity"] * rng.integers(2, 9, after.sum())
    df.loc[after, "price"] = df.loc[after, "price"] + rng.uniform(0.5, 50, after.sum())
    df.loc[after, "line_value"] = df.loc[after, "quantity"] * df.loc[after, "price"]
    df.loc[after, "country"] = "France"
    df.loc[after, "stock_code"] = "77777"
    df.loc[after, "invoice_date"] = df.loc[after, "invoice_date"] + pd.Timedelta(days=3)
    return df


def test_no_leakage_rows_after_cutoff_do_not_change_features(feature_case):
    cutoff = feature_case.cutoff
    base = build_features(feature_case.sales, feature_case.returns, cutoff)
    rng = np.random.default_rng(42)
    sales = _perturb_after(feature_case.sales, cutoff, rng)
    returns = _perturb_after(feature_case.returns, cutoff, rng)
    # brand-new rows after the cutoff: existing customer 2, unseen customer 99, a return
    extra = feature_case.sales.iloc[[0, 0]].copy()
    extra["invoice"] = ["900001", "900002"]
    extra["customer_id"] = pd.array([2, 99], dtype="Int64")
    extra["invoice_date"] = [cutoff, cutoff + pd.Timedelta(days=200)]
    extra_ret = feature_case.returns.iloc[[0]].copy()
    extra_ret["customer_id"] = pd.array([2], dtype="Int64")
    extra_ret["invoice_date"] = [cutoff + pd.Timedelta(hours=1)]
    sales = pd.concat([sales, extra], ignore_index=True)
    returns = pd.concat([returns, extra_ret], ignore_index=True)

    pd.testing.assert_frame_equal(build_features(sales, returns, cutoff), base)
    # and dropping everything from the cutoff on changes nothing either
    before_s = feature_case.sales[feature_case.sales["invoice_date"] < cutoff]
    before_r = feature_case.returns[feature_case.returns["invoice_date"] < cutoff]
    pd.testing.assert_frame_equal(build_features(before_s, before_r, cutoff), base)


def test_rows_before_cutoff_do_change_features(feature_case):
    # guards the leakage test against passing trivially
    cutoff = feature_case.cutoff
    base = build_features(feature_case.sales, feature_case.returns, cutoff)
    sales = feature_case.sales.copy()
    sales.loc[0, "line_value"] = sales.loc[0, "line_value"] + 100
    changed = build_features(sales, feature_case.returns, cutoff)
    assert changed.loc[0, "monetary"] == base.loc[0, "monetary"] + 100


def test_row_order_does_not_matter(feature_case):
    base = build_features(feature_case.sales, feature_case.returns, feature_case.cutoff)
    shuffled = feature_case.sales.sample(frac=1, random_state=42)
    out = build_features(shuffled, feature_case.returns, feature_case.cutoff)
    pd.testing.assert_frame_equal(out, base)


def test_full_snapshot(feature_case):
    cut = full_cutoff(feature_case.sales)
    assert cut == pd.Timestamp("2011-05-02")
    full = build_features(feature_case.sales, feature_case.returns, cut)
    assert full["customer_id"].tolist() == [1, 2, 3, 4, 5, 6]
    c1 = full.set_index("customer_id").loc[1]
    assert c1["frequency"] == 6 and c1["recency"] == 1
    # the C500103 return after the old cutoff now counts: (20 + 10) / (100 + 50 + 20)
    assert c1["return_rate"] == pytest.approx(30 / 170)


def test_is_product():
    codes = pd.Series(["85123A", "22423", "POST", "M", "C2", "BANK CHARGES", "ADJUST", "D"])
    assert is_product(codes).tolist() == [True, True, False, False, False, False, False, False]


def test_log1p_skewed(feature_case):
    f = build_features(feature_case.sales, feature_case.returns, feature_case.cutoff)
    before = f.copy()
    out = log1p_skewed(f)
    pd.testing.assert_frame_equal(f, before)  # input untouched
    assert out["monetary"].tolist() == pytest.approx(np.log1p(f["monetary"]).tolist())
    pd.testing.assert_series_equal(out["return_rate"], f["return_rate"])


def test_summary(feature_case):
    s, r, cut = feature_case.sales, feature_case.returns, feature_case.cutoff
    f, lab = build_features(s, r, cut), build_labels(s, cut, feature_case.horizon)
    full = build_features(s, r, full_cutoff(s))
    out = summarize(f, lab, full, cut, feature_case.horizon, s["invoice_date"].max())
    assert out["n_customers_cutoff"] == 5 and out["n_new_after_cutoff"] == 1
    assert out["returned_rate"] == pytest.approx(3 / 5)
    assert out["label_window"] == ["2011-01-01", "2011-04-01"]
    assert out["label_window_complete"] is True
    assert "60.0%" in out["conclusion"]


def test_on_p1_cleaned_fixture(raw):
    res = clean(raw)
    f = build_features(res.sales, res.returns, "2011-01-01").set_index("customer_id")
    assert f.index.tolist() == [1, 2, 3, 4, 5]
    # customer 1 returned 2 x 2.55 of 40.80 bought; customer 4 returned all of 12.75
    assert f.loc[1, "return_rate"] == pytest.approx(5.10 / 40.80)
    assert f.loc[4, "return_rate"] == pytest.approx(1.0)
    assert f.loc[3, "non_uk_share"] == 1.0


def test_missing_columns_raise(feature_case):
    with pytest.raises(C.ContractError):
        build_features(feature_case.sales.drop(columns="line_value"), feature_case.returns, "2011")
