"""P3c basket tests: hand-made baskets (rule metrics by hand) + the synthetic shop fixture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from retailpulse import contracts as C
from retailpulse.baskets import rules as R
from retailpulse.baskets.stage import compute
from retailpulse.config import CUTOFF

A, B, Cc, D = "10001", "10002", "10003", "10004"
NAMES = {A: "ALPHA MUG", B: "BETA PLATE", Cc: "GAMMA CUP", D: "DELTA BOWL"}
# invoice -> (customer, items). 10 product invoices + 1 postage-only invoice (not a basket).
# A in 6, B in 5, A&B in 4 -> support 0.4, conf(A->B) 4/6, conf(B->A) 4/5, lift 0.4/(0.6*0.5)
BASKETS = {
    "1": (1, [A, B]),
    "2": (2, [A, B]),
    "3": (3, [A, B, Cc]),
    "4": (1, [A]),
    "5": (2, [B]),
    "6": (3, [Cc]),
    "7": (4, [Cc]),
    "8": (4, [A, B]),
    "9": (5, [D]),
    "10": (6, [A]),
    "11": (6, ["POST"]),
}
PARAMS = dict(
    top_n_products=10,
    min_support=0.1,
    min_abs_invoices=2,
    min_confidence=0.3,
    max_len=3,
    max_rules=50,
)
SEGMENTS = pd.DataFrame(
    {
        "customer_id": [1, 2, 6, 3, 4, 5],
        "segment": [0, 0, 0, 1, 1, 1],
        "segment_name": ["Alpha"] * 3 + ["Beta"] * 3,
    }
)


def _sales(baskets: dict, when: str = "2011-03-01") -> pd.DataFrame:
    rows = []
    for inv, (cid, items) in baskets.items():
        for code in items:
            rows.append((inv, code, NAMES.get(code, "POSTAGE"), 1, pd.Timestamp(when), 1.0, cid))
    df = pd.DataFrame(
        rows,
        columns=[
            "invoice",
            "stock_code",
            "description",
            "quantity",
            "invoice_date",
            "price",
            "cid",
        ],
    )
    df["customer_id"] = df.pop("cid").astype("Int64")
    df["country"] = "United Kingdom"
    df["line_value"] = df["quantity"] * df["price"]
    return df


@pytest.fixture
def lines() -> pd.DataFrame:
    return R.product_lines(_sales(BASKETS))


def test_basket_matrix(lines):
    m = R.basket_matrix(lines, [A, B, Cc, D])
    assert m.shape == (10, 4)  # postage-only invoice is not a basket
    assert isinstance(m[A].dtype, pd.SparseDtype)
    assert m.sum().tolist() == [6, 5, 3, 1]


def test_rule_metrics_by_hand(lines):
    rules, info = R.mine_rules(lines, **PARAMS)
    assert info["n_invoices"] == 10
    assert info["min_support"] == pytest.approx(0.2)  # max(0.1, 2 / 10)
    assert len(rules) == 2  # A->B and B->A; C, D and pairs with C are too rare
    ab = rules[rules["antecedent_codes"] == A].iloc[0]
    assert ab["consequent_codes"] == B
    assert ab["antecedents"] == "ALPHA MUG" and ab["consequents"] == "BETA PLATE"
    assert ab["support"] == pytest.approx(0.4)
    assert ab["confidence"] == pytest.approx(4 / 6)
    assert ab["lift"] == pytest.approx(0.4 / (0.6 * 0.5))
    assert ab["antecedent_support"] == pytest.approx(0.6)
    ba = rules[rules["antecedent_codes"] == B].iloc[0]
    assert ba["confidence"] == pytest.approx(0.8)
    assert (rules["lift"] > 1).all() and (rules["scope"] == "overall").all()


def test_chi_square_top_rule(lines):
    chi = R.rule_contingency(lines, A, B)
    assert chi["counts"] == {"a_and_b": 4, "a_not_b": 2, "not_a_b": 1, "neither": 3}
    ref = stats.chi2_contingency(np.array([[4, 2], [1, 3]]), correction=False)
    assert chi["chi2"] == pytest.approx(ref[0]) and chi["p_value"] == pytest.approx(ref[1])
    assert chi["p_b_given_a"] == pytest.approx(4 / 6)


def test_multi_item_rule_strings():
    assert R._join({B, A}) == f"{A},{B}"
    assert R._join({B, A}, NAMES) == "ALPHA MUG + BETA PLATE"


def test_per_segment_scopes(lines):
    rules, infos = R.mine_all(lines, SEGMENTS, **PARAMS)
    assert list(infos) == ["overall", "segment:Alpha", "segment:Beta"]
    # Alpha = invoices 1, 2, 4, 5, 10; Beta = 3, 6, 7, 8, 9
    assert infos["segment:Alpha"]["n_invoices"] == 5
    assert infos["segment:Beta"]["n_invoices"] == 5
    assert infos["segment:Beta"]["min_support"] == pytest.approx(0.4)  # 2 / 5
    # Alpha: A&B in 2 of 5, lift 0.4 / (0.8 * 0.6) < 1 -> no rule
    assert infos["segment:Alpha"]["n_rules"] == 0
    beta = rules[rules["scope"] == "segment:Beta"]
    ab = beta[beta["antecedent_codes"] == A].iloc[0]
    assert ab["support"] == pytest.approx(0.4) and ab["confidence"] == pytest.approx(1.0)
    assert ab["lift"] == pytest.approx(2.5)


def test_recommend_by_hand(lines):
    rules, _ = R.mine_rules(lines, **PARAMS)
    rec = R.recommend(6, lines, rules, n=3)  # customer 6 bought A only
    assert rec["stock_code"].tolist() == [B, Cc, D]
    assert rec["source"].tolist() == ["rule", "popular", "popular"]
    assert rec.loc[0, "score"] == pytest.approx((4 / 6) * (0.4 / 0.3))  # confidence x lift
    assert rec.loc[1, "score"] == pytest.approx(0.3)  # support of C
    assert rec["rank"].tolist() == [1, 2, 3]
    assert R.recommend(5, lines, rules, n=3)["stock_code"].tolist() == [A, B, Cc]


def test_segment_rules_rank_first(lines):
    base = {"support": 0.2, "antecedent_support": 0.6, "consequent_support": 0.3}
    rules = pd.DataFrame(
        [
            {"scope": "overall", "antecedent_codes": A, "consequent_codes": Cc,
             "confidence": 0.9, "lift": 5.0, **base},
            {"scope": "segment:Alpha", "antecedent_codes": A, "consequent_codes": D,
             "confidence": 0.4, "lift": 1.1, **base},
        ]
    )  # fmt: skip
    rec = R.recommend(6, lines, rules, SEGMENTS, n=3)
    assert rec["stock_code"].tolist()[:2] == [D, Cc]
    # popular fill comes from the customer's segment (Alpha best seller not bought: B)
    assert rec["stock_code"].tolist()[2] == B and rec["source"].tolist()[2] == "popular"


def test_hit_rate_by_hand():
    recs = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3],
            "rank": [1, 2, 1, 2, 1],
            "stock_code": [Cc, D, Cc, D, A],
        }
    )
    before = pd.DataFrame({"customer_id": [1, 2, 3], "stock_code": [A, A, A]})
    # 1 buys C (new, recommended: hit); 2 buys B (new, missed); 3 re-buys A only (not evaluated)
    window = pd.DataFrame({"customer_id": [1, 2, 3], "stock_code": [Cc, B, A]})
    h = R.hit_rate(recs, before, window, k=5)
    assert h == {"n_eval_customers": 2, "hits": 1, "hit_rate": 0.5}


# ------------------------------------------------------------------ synthetic shop
@pytest.fixture(scope="module")
def shop_segments(synthetic_shop):
    ids = np.sort(synthetic_shop.sales["customer_id"].dropna().unique().astype("int64"))
    seg = (ids % 3).astype("int64")
    return pd.DataFrame(
        {"customer_id": ids, "segment": seg, "segment_name": [f"Seg {s}" for s in seg]}
    )


@pytest.fixture(scope="module")
def shop_result(synthetic_shop, shop_segments):
    sales = synthetic_shop.sales
    customers = R.product_lines(sales, end=CUTOFF)["customer_id"].unique()
    params = dict(min_support=0.01, min_abs_invoices=3, min_confidence=0.1)
    return customers, compute(sales, customers, shop_segments, **params)


def test_recommend_exactly_n_and_never_bought(synthetic_shop, shop_result):
    customers, (tables, _) = shop_result
    recs = tables[C.RECOMMENDATIONS_PARQUET]
    assert set(recs["customer_id"]) == set(customers)
    assert (recs.groupby("customer_id").size() == 5).all()
    assert recs.groupby("customer_id")["rank"].apply(lambda r: list(r) == [1, 2, 3, 4, 5]).all()
    assert not recs.duplicated(["customer_id", "stock_code"]).any()
    before = R.product_lines(synthetic_shop.sales, end=CUTOFF)
    bought = set(zip(before["customer_id"], before["stock_code"], strict=True))
    assert not any(
        pair in bought for pair in zip(recs["customer_id"], recs["stock_code"], strict=True)
    )
    assert set(recs["source"]) <= {"rule", "popular"}


def test_shop_contracts_and_metrics(shop_result):
    _, (tables, metrics) = shop_result
    for path, df in tables.items():
        C.validate(df, path)
    assert set(C.P3_METRICS_KEYS["rules"]) <= set(metrics)
    assert set(metrics["n_rules"]) == {"overall", "segment:Seg 0", "segment:Seg 1", "segment:Seg 2"}
    assert metrics["segments_used"] is True
    assert {"hit_rate_at_5", "popularity_hit_rate_at_5", "n_eval_customers"} <= set(metrics)
    rules = tables[C.RULES_PARQUET]
    assert (rules["lift"] > 1).all() and (rules["confidence"] >= 0.1).all()
    for scope, grp in rules.groupby("scope"):
        assert (grp["support"] >= metrics["min_support_by_scope"][scope] - 1e-12).all()
    C.to_jsonable(metrics)


def test_rules_ignore_rows_from_cutoff(synthetic_shop, shop_segments):
    sales = synthetic_shop.sales
    customers = R.product_lines(sales, end=CUTOFF)["customer_id"].unique()
    params = dict(min_support=0.01, min_abs_invoices=3, min_confidence=0.1)
    late = sales[sales["invoice_date"] >= CUTOFF]
    noisy = pd.concat([sales[sales["invoice_date"] < CUTOFF], late, late.assign(invoice="X")])
    a, _ = compute(sales, customers, shop_segments, horizon_days=1, **params)
    b, _ = compute(noisy, customers, shop_segments, horizon_days=1, **params)
    for path in a:
        pd.testing.assert_frame_equal(a[path], b[path])


def test_overall_only_without_segments(lines):
    sales = _sales(BASKETS)
    tables, metrics = compute(sales, [1, 2, 3], None, cutoff="2011-06-01", **PARAMS)
    assert metrics["n_rules"] == {"overall": 2}
    assert metrics["segments_used"] is False
    assert metrics["top_rule"]["lift"] == pytest.approx(0.4 / 0.3)
    assert metrics["chi2_top_rule"]["counts"]["a_and_b"] == 4
    assert (tables[C.RECOMMENDATIONS_PARQUET].groupby("customer_id").size() <= 5).all()
