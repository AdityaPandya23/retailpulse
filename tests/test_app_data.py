"""P4 app_data stage: customer timeline + the rule behind each recommendation (fixtures only)."""

from __future__ import annotations

import pandas as pd
import pytest

from retailpulse import app_data as AD
from retailpulse import contracts as C
from retailpulse.features import build_features


@pytest.fixture
def timeline(feature_case):
    fc = feature_case
    feats = build_features(fc.sales, fc.returns, fc.cutoff)
    tl = AD.build_timeline(fc.sales, feats["customer_id"], fc.cutoff, fc.horizon)
    return feats, tl


def _row(tl, cid, month, period):
    r = tl[(tl.customer_id == cid) & (tl.month == pd.Timestamp(month)) & (tl.period == period)]
    return r.iloc[0] if len(r) else None


def test_timeline_contract_and_population(timeline):
    feats, tl = timeline
    C.validate(tl, C.TIMELINE_PARQUET)
    assert set(tl["customer_id"]) == set(feats["customer_id"])  # customer 4 (after only) absent
    assert 4 not in set(tl["customer_id"])
    assert set(tl["period"]) <= {"before", "after"}
    assert not tl.duplicated(["customer_id", "month", "period"]).any()


def test_timeline_periods_follow_the_label_window(timeline):
    _, tl = timeline
    # customer 1: 2011-02 is in the window (10 x 5.0); 2011-05 is after it -> dropped
    r = _row(tl, 1, "2011-02-01", "after")
    assert r is not None and r.revenue == pytest.approx(50.0) and r.orders == 1
    assert _row(tl, 1, "2011-05-01", "after") is None
    # customer 5: the minute before the cutoff is "before", the cutoff itself is "after"
    assert _row(tl, 5, "2010-12-01", "before").revenue == pytest.approx(5.0)
    assert _row(tl, 5, "2011-01-01", "after").revenue == pytest.approx(10.0)
    # customer 6: 2011-04-01 00:00 is the (open) window end -> not in the timeline
    assert _row(tl, 6, "2011-04-01", "after") is None
    # customer 3: last hours of the window count
    assert _row(tl, 3, "2011-03-01", "after").revenue == pytest.approx(5.0)


def test_before_revenue_equals_monetary(timeline):
    feats, tl = timeline
    before = tl[tl.period == "before"].groupby("customer_id")["revenue"].sum()
    m = feats.set_index("customer_id")["monetary"]
    pd.testing.assert_series_equal(before.sort_index(), m.sort_index(), check_names=False)


# ---------------- rec_reasons ----------------
A, B, Cc, D = "10001", "10002", "10003", "10004"


def _lines(rows):
    df = pd.DataFrame(rows, columns=["invoice", "stock_code", "customer_id", "invoice_date"])
    df["description"] = "P" + df["stock_code"].str[-1]
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


SALES = _lines(
    [
        ("1", A, 1, "2011-01-01"),
        ("1", B, 1, "2011-01-01"),
        ("2", A, 2, "2011-02-01"),
        ("3", B, 2, "2011-10-01"),  # after the cutoff: must not make a rule fire
    ]
)
RULES = pd.DataFrame(
    [
        ("overall", "PA", "PC", A, Cc, 0.5, 2.0),
        ("overall", "PA + PB", "PD", f"{A},{B}", D, 0.6, 3.0),
        ("segment:S1", "PA", "PD", A, D, 0.4, 2.0),
    ],
    columns=[
        "scope",
        "antecedents",
        "consequents",
        "antecedent_codes",
        "consequent_codes",
        "confidence",
        "lift",
    ],
)
RECS = pd.DataFrame(
    [(1, 1, D, "rule"), (1, 2, Cc, "rule"), (2, 1, D, "rule"), (2, 2, Cc, "popular")],
    columns=["customer_id", "rank", "stock_code", "source"],
)
SEGS = pd.DataFrame({"customer_id": [1, 2], "segment_name": ["S1", "S2"]})


def test_rec_reasons_by_hand():
    out = AD.recommendation_reasons(SALES, RULES, RECS, SEGS, cutoff="2011-09-09")
    C.validate(out, C.REC_REASONS_PARQUET)
    got = {(r.customer_id, r.stock_code): r.scope for r in out.itertuples()}
    # own segment rule wins even with a lower confidence x lift than an overall rule
    assert got[(1, D)] == "segment:S1"
    assert got[(1, Cc)] == "overall"
    # customer 2 bought B only after the cutoff -> A+B -> D did not fire; popular recs skipped
    assert (2, D) not in got and (2, Cc) not in got
    assert len(out) == 2


def test_rec_reasons_empty_inputs():
    out = AD.recommendation_reasons(SALES, RULES.iloc[0:0], RECS, SEGS)
    assert out.empty and list(out.columns) == C.REC_REASONS_COLUMNS


def test_summary_counts(timeline):
    _, tl = timeline
    s = AD.summarize(tl, AD.recommendation_reasons(SALES, RULES, RECS, SEGS), RECS)
    assert s["timeline_rows"] == len(tl) and s["rule_recommendations"] == 3
    assert s["rec_reasons_rows"] == 2 and "conclusion" in s
