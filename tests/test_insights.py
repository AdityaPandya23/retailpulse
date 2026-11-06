"""Insights logic (retailpulse.insights) on tiny hand-made frames (never real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retailpulse import contracts as C
from retailpulse import insights as I


def _profiles(recency, monetary, tenure, extra=None) -> pd.DataFrame:
    n = len(recency)
    df = pd.DataFrame(
        {
            "segment": range(n),
            "segment_name": [f"S{i}" for i in range(n)],
            "n_customers": [10] * n,
            "customer_share": [1 / n] * n,
            "revenue_share": [1 / n] * n,
            "recency": recency,
            "monetary": monetary,
            "tenure": tenure,
        }
    )
    return df.assign(**(extra or {}))


def test_actions_follow_the_rule_order():
    # S0 best spender, S1 lapsed, S2 newest, S3 nothing extreme
    p = _profiles([20, 300, 50, 60], [5000, 300, 400, 900], [600, 500, 90, 400])
    acts = {s: a for s, (a, _) in I.suggest_actions(p).items()}
    assert acts == {0: I.REWARD, 1: I.WIN_BACK, 2: I.SECOND_ORDER, 3: I.KEEP_ENGAGED}
    reasons = I.suggest_actions(p)
    assert "300 days" in reasons[1][1] and "£5,000" in reasons[0][1]


def test_first_match_wins_one_action_per_segment():
    # S0 has BOTH the highest recency and the highest spend -> only "Win-back offer";
    # nobody else gets "Reward and keep" (the rule is checked per segment, in order)
    p = _profiles([300, 20, 50], [5000, 300, 400], [600, 90, 400])
    acts = {s: a for s, (a, _) in I.suggest_actions(p).items()}
    assert acts == {0: I.WIN_BACK, 1: I.SECOND_ORDER, 2: I.KEEP_ENGAGED}
    assert all(a in I.ACTIONS for a in acts.values())
    assert I.suggest_actions(None) == {} and I.suggest_actions(p.iloc[0:0]) == {}


def test_ties_go_to_the_lowest_segment_id():
    p = _profiles([100, 100], [10, 10], [5, 5])
    acts = {s: a for s, (a, _) in I.suggest_actions(p).items()}
    assert acts == {0: I.WIN_BACK, 1: I.KEEP_ENGAGED}


def _frames():
    segments = pd.DataFrame({"customer_id": [1, 2, 3, 4], "segment": [0, 0, 1, 1]})
    cls = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 1, 2, 3, 4, 1],
            "model": ["knn"] * 4 + ["logistic"] * 4 + ["knn"],
            "split": ["test"] * 8 + ["train"],
            "p_return": [0.8, 0.6, 0.2, 0.0, 0.1, 0.1, 0.1, 0.1, 0.99],
            "pred_return": [1, 1, 0, 0, 0, 0, 0, 0, 1],
            "returned_flag": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        }
    )
    recs = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3, 4],
            "rank": [1, 2, 1, 2, 1, 1],
            "stock_code": ["a", "b", "a", "c", "z", "z"],
            "description": ["MUG", "BOWL", "MUG", "CUP", "TIN", "TIN"],
            "score": [1.0] * 6,
        }
    )
    return segments, cls, recs


def test_mean_p_return_uses_chosen_model_on_test_split_only():
    segments, cls, _ = _frames()
    assert I.mean_p_return(cls, segments, "knn") == pytest.approx({0: 0.7, 1: 0.1})
    assert I.mean_p_return(cls, segments, None) == {}
    assert I.mean_p_return(None, segments, "knn") == {}


def test_came_back_share_profiles_first_then_test_outcome():
    segments, cls, _ = _frames()
    p = _profiles([1, 2], [1, 2], [1, 2], {"bought_again_share": [0.65, 0.19]})
    assert I.came_back_share(p) == {0: 0.65, 1: 0.19}
    p_no = _profiles([1, 2], [1, 2], [1, 2])
    assert I.came_back_share(p_no, cls, segments, "knn") == {0: 0.5, 1: 0.0}


def test_top_recommended_counts_per_segment_ties_by_name():
    segments, _, recs = _frames()
    top = I.top_recommended(recs, segments, n=2)
    assert top == {0: ["MUG", "BOWL"], 1: ["TIN"]}
    assert I.top_recommended(None, segments) == {}


def test_segment_insights_one_row_per_segment():
    segments, cls, recs = _frames()
    p = _profiles([300, 20], [100, 5000], [500, 400], {"bought_again_share": [0.2, 0.6]})
    t = I.segment_insights(p, {0: "Lost", 1: "Champions"}, segments, cls, "knn", recs)
    assert list(t.columns) == I.INSIGHT_COLUMNS
    assert list(t["Segment"]) == ["0 · Lost", "1 · Champions"]
    assert list(t["Suggested action"]) == [I.WIN_BACK, I.REWARD]
    assert t.loc[0, "Avg P(return)"] == pytest.approx(0.7) and t.loc[1, "Came back %"] == 0.6
    assert t.loc[0, "Top 3 recommended products"] == "MUG, BOWL, CUP"
    # missing predictions / recommendations -> NaN / "n/a", never a crash
    t2 = I.segment_insights(p, None, None, None, None, None)
    assert np.isnan(t2.loc[0, "Avg P(return)"]) and t2.loc[0, "Top 3 recommended products"] == "n/a"
    assert t2.loc[0, "Segment"] == "0 · S0"
    assert I.segment_insights(None, {}, None, None, None, None).empty


def test_limitations_parse_bullets_and_continuations():
    text = (
        "# T\n\n## Limitations\n\n- **One:** first line\n  goes on.\n- **Two:** short.\n"
        "\n## Next\n- not this\n"
    )
    assert I.limitations(text) == ["**One:** first line goes on.", "**Two:** short."]
    assert I.limitations("no such heading") == []


def test_readme_has_the_four_v2_limitations():
    items = I.limitations(C.README_MD.read_text(encoding="utf-8"))
    assert len(items) == 4
    assert items[0].startswith("**Registered customers only:**")
