"""P3a segmentation tests (synthetic data only)."""

from __future__ import annotations

import dataclasses
import io

import joblib
import numpy as np
import pandas as pd
import pytest

from retailpulse import contracts as C
from retailpulse.config import SEGMENT
from retailpulse.features import build_features, build_labels
from retailpulse.segment import baseline as B
from retailpulse.segment import cluster as CL
from retailpulse.segment import validate as V
from retailpulse.segment.stage import check_metrics


def _cands(rows):
    return pd.DataFrame(rows, columns=["k", "silhouette", "min_segment_share", "ari_mean"])


# ---------------- k choice rule ----------------
def test_choose_k_max_silhouette_when_alone():
    t = _cands([(3, 0.40, 0.10, 0.80), (4, 0.30, 0.10, 0.99), (5, 0.25, 0.10, 0.99)])
    k, reason, table = CL.choose_k(t)
    assert k == 3
    assert table.loc[table["k"] == 3, "chosen"].item()
    assert "No other eligible k" in reason


def test_choose_k_window_prefers_higher_ari():
    t = _cands([(3, 0.40, 0.10, 0.80), (4, 0.385, 0.10, 0.95), (5, 0.37, 0.10, 0.99)])
    k, reason, table = CL.choose_k(t)
    assert k == 4  # k=5 is 0.03 below the max: outside the 0.02 window
    assert table["in_window"].tolist() == [True, True, False]
    assert "highest mean stability ARI" in reason


def test_choose_k_eligibility_3_percent():
    t = _cands([(3, 0.50, 0.02, 0.99), (4, 0.30, 0.031, 0.90), (5, 0.29, 0.05, 0.80)])
    k, _, table = CL.choose_k(t)
    assert table["eligible"].tolist() == [False, True, True]
    assert k == 4  # k=3 has the best silhouette but a 2% segment; k=4 wins on ARI in window


def test_choose_k_ari_tie_goes_to_smaller_k():
    t = _cands([(3, 0.40, 0.10, 0.90), (4, 0.41, 0.10, 0.90), (5, 0.30, 0.10, 0.99)])
    k, reason, _ = CL.choose_k(t)
    assert k == 3
    assert "smaller k" in reason


def test_choose_k_fallback_when_none_eligible():
    t = _cands([(3, 0.40, 0.01, 0.90), (4, 0.45, 0.02, 0.90)])
    k, reason, table = CL.choose_k(t)
    assert k == 4
    assert "falls back" in reason
    assert not table["eligible"].any()


# ---------------- naming ----------------
def _z(**kw):
    row = dict.fromkeys(SEGMENT.features, 0.0)
    row.update(kw)
    return row


def test_naming_rule_outcomes():
    z = pd.DataFrame(
        [
            _z(recency=-1.0, frequency=1.2, monetary=1.5),
            _z(recency=1.0, monetary=1.0, frequency=0.2),
            _z(recency=-1.0, tenure=-1.0, frequency=-0.3, monetary=-0.3),
            _z(recency=1.2, frequency=-0.8, monetary=-0.9),
            _z(avg_basket_value=1.0, frequency=-0.4, monetary=0.2),
            _z(),
        ]
    )
    names = CL.name_segments(z)
    assert names == {
        0: "Champions",
        1: "At Risk High-Value",
        2: "New Customers",
        3: "Lost / Hibernating",
        4: "Big-Basket Occasionals",
        5: "Regulars",
    }


def test_naming_unique_and_overrides():
    z = pd.DataFrame([_z(), _z(), _z(recency=-1, frequency=1, monetary=1), _z()])
    names = CL.name_segments(z)
    assert len(set(names.values())) == 4
    assert names[0] == "Regulars" and names[1] == "Regulars 2" and names[3] == "Regulars 3"
    names = CL.name_segments(z, overrides={1: "Core", 9: "ignored"})
    assert names[1] == "Core" and 9 not in names
    assert CL.unique_names({0: "A", 1: "A 2", 2: "A"}) == {0: "A", 1: "A 2", 2: "A 3"}


# ---------------- ids, baseline, stability ----------------
def test_segment_ids_ordered_by_revenue():
    labels = np.array([0, 0, 1, 1, 2, 2])
    monetary = np.array([1.0, 1.0, 50.0, 50.0, 10.0, 10.0])
    assert CL.order_by_revenue(labels, monetary) == {1: 0, 2: 1, 0: 2}


def test_rfm_score_direction():
    f = pd.DataFrame(
        {
            "recency": [1, 50, 100, 200, 300, 400, 500, 600],
            "frequency": [10, 8, 6, 5, 4, 3, 2, 1],
            "monetary": [5.0, 900.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0],
        }
    )
    s = B.rfm_scores(f)
    assert s.loc[0, "r_score"] == 4 and s.loc[7, "r_score"] == 1  # most recent = 4
    assert s.loc[0, "f_score"] == 4 and s.loc[7, "f_score"] == 1
    assert s.loc[1, "m_score"] == 4 and s.loc[0, "m_score"] == 1  # biggest spender = 4
    assert s[["r_score", "f_score", "m_score"]].isin([1, 2, 3, 4]).all().all()
    groups = B.rfm_groups(s["rfm_score"], 3)
    assert set(groups) == {0, 1, 2}
    assert groups[s["rfm_score"].idxmax()] == 0  # highest total score -> group 0


def test_rfm_quartiles_with_ties_are_balanced():
    s = B.quartile_score(pd.Series([1] * 6 + [2, 3]))
    assert s.value_counts().sort_index().tolist() == [2, 2, 2, 2]


def test_stability_ari_deterministic():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.3, (40, 2)) for c in (0, 3, 6)])
    a = V.stability_ari(X, 3, n_subsamples=4, seed=42)
    b = V.stability_ari(X, 3, n_subsamples=4, seed=42)
    assert a == b and len(a) == 4
    assert min(a) > 0.9  # well-separated blobs are stable


def test_chi2_segment_uk_and_degenerate_case():
    seg = np.repeat([0, 1], 50)
    non_uk = np.r_[np.zeros(45), np.ones(5), np.zeros(10), np.ones(40)]
    res = V.chi2_segment_uk(seg, non_uk)
    assert res["p_value"] < 0.001 and res["dof"] == 1 and 0 < res["cramers_v"] <= 1
    assert "associated" in res["conclusion"]
    flat = V.chi2_segment_uk(seg, np.zeros(100))
    assert np.isnan(flat["chi2"]) and "Not testable" in flat["conclusion"]


def test_log1p_frame_accepts_array_and_frame():
    feats = ["recency", "monetary"]
    df = pd.DataFrame({"monetary": [0.0, np.e - 1], "recency": [5, 6], "x": [1, 2]})
    out = CL.log1p_frame(df, feats, CL.log_columns(feats))
    assert list(out.columns) == feats
    assert out["monetary"].tolist() == pytest.approx([0.0, 1.0])
    assert out["recency"].tolist() == [5.0, 6.0]
    arr = CL.log1p_frame(df[feats].to_numpy(), feats, ["monetary"])
    pd.testing.assert_frame_equal(arr, out)


# ---------------- end-to-end contracts on the synthetic shop ----------------
@pytest.fixture(scope="module")
def seg_result(synthetic_shop):
    cutoff = pd.Timestamp("2011-06-01")
    f = build_features(synthetic_shop.sales, synthetic_shop.returns, cutoff)
    lab = build_labels(synthetic_shop.sales, cutoff, 90)
    cfg = dataclasses.replace(SEGMENT, k_range=(2, 5), k_choice_range=(3, 4), k=None, names={})
    res = CL.segment_customers(f, lab, cfg=cfg, n_subsamples=3, cv_folds=3)
    return f, lab, cfg, res


def test_end_to_end_contracts(seg_result):
    f, _, cfg, res = seg_result
    assert set(res.tables) == {
        C.SEGMENTS_PARQUET,
        C.SEGMENT_PROFILES_PARQUET,
        C.K_SELECTION_PARQUET,
        C.PCA_LOADINGS_PARQUET,
        C.SEGMENT_PRESETS_PARQUET,
    }
    for path, df in res.tables.items():
        C.validate(df, path)
    check_metrics(res.metrics)
    m = res.metrics
    k = m["k"]
    seg = res.tables[C.SEGMENTS_PARQUET]
    prof = res.tables[C.SEGMENT_PROFILES_PARQUET]
    ksel = res.tables[C.K_SELECTION_PARQUET]
    load = res.tables[C.PCA_LOADINGS_PARQUET]
    assert len(seg) == len(f) and seg["customer_id"].is_unique
    assert set(seg["segment"]) == set(range(k)) and 3 <= k <= 4
    assert ksel["k"].tolist() == [2, 3, 4, 5]
    assert ksel.loc[ksel["k"].isin([3, 4]), "ari_mean"].notna().all()
    assert ksel.loc[~ksel["k"].isin([3, 4]), "ari_mean"].isna().all()
    assert load["feature"].tolist() == list(cfg.features)
    assert len(m["pca_explained_variance"]) == len(cfg.features)
    assert sum(m["pca_explained_variance"]) == pytest.approx(1.0)
    assert prof["n_customers"].sum() == len(f)
    assert prof["customer_share"].sum() == pytest.approx(1.0)
    assert prof["revenue_share"].sum() == pytest.approx(1.0)
    assert prof["revenue_share"].is_monotonic_decreasing  # segment 0 = most valuable
    assert prof["segment_name"].is_unique
    assert len(m["stability_ari"]["values"]) == 3
    for side in ("kmeans", "rfm_quartile"):
        assert {"silhouette", "davies_bouldin", "calinski_harabasz", "auc_returned"} <= set(
            m["baseline"][side]
        )
    assert {"chi2", "dof", "p_value", "cramers_v", "conclusion"} <= set(m["chi2_segment_uk"])
    assert isinstance(m["k_choice_reason"], str) and m["k_candidates"]
    # JSON-safe after contracts.to_jsonable
    import json

    json.dumps(C.to_jsonable(m), allow_nan=False)


def test_model_round_trips_on_raw_features(seg_result):
    f, _, _, res = seg_result
    buf = io.BytesIO()
    joblib.dump(res.model, buf)
    buf.seek(0)
    model = joblib.load(buf)
    assert set(model) == {"pipeline", "features", "label_to_segment", "segment_names", "pca"}
    pred = CL.predict_segment(model, f)  # raw, un-logged features (extra columns ignored)
    seg = res.tables[C.SEGMENTS_PARQUET]
    np.testing.assert_array_equal(pred, seg["segment"].to_numpy())
    one = f[model["features"]].iloc[[0]]
    assert CL.predict_segment(model, one)[0] == seg["segment"].iloc[0]


def test_model_pca_places_a_new_customer_on_the_scatter(seg_result):
    f, _, _, res = seg_result
    model, seg = res.model, res.tables[C.SEGMENTS_PARQUET]
    assert model["pca"].n_components_ == 3
    pcs = seg[["pc1", "pc2", "pc3"]].to_numpy()
    np.testing.assert_allclose(CL.pca_coords(model, f), pcs, atol=1e-8)  # raw features in
    one = f[model["features"]].iloc[[3]]  # a single "new" customer
    np.testing.assert_allclose(CL.pca_coords(model, one)[0], pcs[3], atol=1e-8)


def test_presets_are_segment_medians(seg_result):
    f, _, _, res = seg_result
    presets = C.validate(res.tables[C.SEGMENT_PRESETS_PARQUET], C.SEGMENT_PRESETS_PARQUET)
    k, names = res.metrics["k"], res.metrics["segment_names"]
    assert presets["segment"].tolist() == list(range(k))
    assert presets["segment_name"].tolist() == [names[s] for s in range(k)]
    seg = res.tables[C.SEGMENTS_PARQUET][["customer_id", "segment"]]
    medians = f.merge(seg, on="customer_id").groupby("segment")[C.PRESET_FEATURES].median()
    np.testing.assert_allclose(presets[C.PRESET_FEATURES].to_numpy(), medians.to_numpy())
    # a preset is a valid raw input for the saved model
    assert len(CL.predict_segment(res.model, presets)) == k


def test_end_to_end_deterministic_and_config_k(synthetic_shop, seg_result):
    f, lab, cfg, res = seg_result
    again = CL.segment_customers(f, lab, cfg=cfg, n_subsamples=3, cv_folds=3)
    pd.testing.assert_frame_equal(again.tables[C.SEGMENTS_PARQUET], res.tables[C.SEGMENTS_PARQUET])
    assert again.metrics["stability_ari"] == res.metrics["stability_ari"]
    fixed = dataclasses.replace(cfg, k=2, names={0: "Best"})
    out = CL.segment_customers(f, lab, cfg=fixed, n_subsamples=2, cv_folds=3)
    assert out.metrics["k"] == 2 and "config.yaml" in out.metrics["k_choice_reason"]
    assert out.metrics["segment_names"][0] == "Best"
    assert out.tables[C.SEGMENTS_PARQUET]["segment"].nunique() == 2


def test_override_name_wins_over_matching_automatic_name():
    # segment 0 is automatically "Champions"; the human names segment 1 "Champions" too
    z = pd.DataFrame([_z(recency=-1, frequency=1, monetary=1), _z()])
    assert CL.name_segments(z)[0] == "Champions"
    names = CL.name_segments(z, overrides={1: "Champions"})
    assert names == {0: "Champions 2", 1: "Champions"}
    assert list(names) == [0, 1]


def test_config_k_marks_the_used_k_as_chosen(seg_result):
    f, lab, cfg, res = seg_result
    auto = res.metrics["k"]
    other = next(k for k in range(cfg.k_choice_range[0], cfg.k_choice_range[1] + 1) if k != auto)
    out = CL.segment_customers(
        f, lab, cfg=dataclasses.replace(cfg, k=other), n_subsamples=2, cv_folds=3
    )
    cand = pd.DataFrame(out.metrics["k_candidates"])
    assert cand.loc[cand["chosen"], "k"].tolist() == [other]
    assert cand.loc[cand["rule_choice"], "k"].tolist() == [out.metrics["auto_k"]]
