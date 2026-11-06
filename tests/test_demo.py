"""Live-demo logic (retailpulse.demo) on tiny hand-made frames and the MOCK models."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from retailpulse import contracts as C
from retailpulse import demo
from retailpulse.config import PREDICT, SEGMENT


def _mock(path):
    file = C.mock_path(path)
    if not file.exists():
        pytest.skip("artifacts/mock/ not generated")
    return joblib.load(file) if file.suffix == ".joblib" else pd.read_parquet(file)


def test_demo_features_follow_config_order():
    assert list(dict.fromkeys([*SEGMENT.features, *PREDICT.features])) == demo.DEMO_FEATURES
    assert len(demo.DEMO_FEATURES) == 6


def test_input_value_casts_and_clamps():
    assert demo.input_value("recency", 12.6) == 13 and isinstance(
        demo.input_value("recency", 1), int
    )
    assert demo.input_value("frequency", 0) == 1  # min 1 order
    assert demo.input_value("monetary", 123.456) == 123.46
    assert isinstance(demo.input_value("monetary", 5), float)
    assert demo.input_value("monetary", float("nan")) == 0.0
    assert demo.input_value("tenure", None) == 0
    spec = demo.input_spec("brand_new_feature")  # config-added feature: float spec
    assert spec["integer"] is False and spec["label"] == "Brand new feature"


def test_preset_values_and_matching_preset():
    presets = pd.DataFrame(
        {
            "segment": [0, 1],
            "segment_name": ["A", "B"],
            **{f: [10.4, 200.0] for f in demo.DEMO_FEATURES},
        }
    )
    vals = demo.preset_values(presets, 0)
    assert list(vals) == demo.DEMO_FEATURES
    assert vals["recency"] == 10 and vals["monetary"] == 10.4
    assert demo.preset_values(presets, 7) == {} and demo.preset_values(None, 0) == {}
    assert demo.matching_preset(presets, vals) == 0
    assert demo.matching_preset(presets, {**vals, "recency": 11}) is None
    assert demo.matching_preset(None, vals) is None


def test_input_warnings():
    assert demo.input_warnings({"recency": 30, "tenure": 300}) == []
    assert "Tenure" in demo.input_warnings({"recency": 300, "tenure": 30})[0]


def test_kpi_facts_from_quality_json():
    wh = {"row_counts": {"fact_sales": 100, "dim_customer": 7}, "total_revenue": 1234.5}
    cleaning = {"raw_rows": 150, "steps": [{"rows_out": 120}, {"rows_out": 100}]}
    fmt = {"rows": 150, "date_min": "2009-12-01 07:45:00", "date_max": "2011-12-09 12:50:00"}
    facts = demo.kpi_facts(wh, cleaning, fmt)
    assert facts == {
        "clean_rows": 100,
        "raw_rows": 150,
        "customers": 7,
        "revenue": 1234.5,
        "date_min": "2009-12-01",
        "date_max": "2011-12-09",
    }
    # fallback: clean rows from the last cleaning step; nothing -> None
    assert demo.kpi_facts({}, cleaning, {})["clean_rows"] == 100
    assert demo.kpi_facts(None, None, None)["date_min"] is None


def test_pipeline_dot_names_every_stage_and_model():
    dot = demo.pipeline_dot(["logistic", "knn"], 6)
    assert dot.startswith("digraph") and dot.rstrip().endswith("}")
    for word in ("Raw Excel", "Clean", "Features", "K-Means", "2 classifiers", "Dashboard"):
        assert word in dot
    assert "logistic · knn" in dot and "clean -> rules" in dot


def test_score_customer_with_mock_models():
    seg_model = _mock(C.SEGMENT_MODEL_JOBLIB)
    classifiers = _mock(C.CLASSIFIERS_JOBLIB)
    presets = _mock(C.SEGMENT_PRESETS_PARQUET)
    sid = int(presets["segment"].iloc[0])
    raw = demo.raw_frame(demo.preset_values(presets, sid))
    place = demo.place_segment(seg_model, raw)
    assert isinstance(place["segment"], int) and len(place["pc"]) == 3
    assert demo.place_segment(None, raw) == {"segment": None, "pc": None}

    probs = demo.return_probabilities(classifiers, raw)
    assert list(probs["model"]) == [m for m in C.ALL_CLASSIFIERS if m in classifiers["pipelines"]]
    assert probs["p_return"].between(0, 1).all()
    assert probs["chosen"].sum() == 1
    assert probs.loc[probs["chosen"], "model"].iloc[0] == classifiers["best_model"]
    assert demo.return_probabilities(None, raw).empty


def test_verdict_uses_chosen_model_threshold_and_counts_agreement():
    probs = pd.DataFrame(
        {
            "model": ["logistic", "knn", "naive_bayes"],
            "p_return": [0.30, 0.40, 0.70],
            "threshold": [0.35, 0.35, 0.50],
            "chosen": [False, True, False],
        }
    )
    v = demo.verdict(probs, 90)
    assert v["model"] == "knn" and v["will_return"] is True and v["n_agree"] == 2
    assert "Likely to come back" in v["text"] and "40%" in v["text"] and "35%" in v["text"]
    v2 = demo.verdict(probs.assign(threshold=[0.35, 0.45, 0.5]), 90)
    assert v2["will_return"] is False and "Unlikely" in v2["text"] and v2["n_agree"] == 2
    assert demo.verdict(probs.iloc[0:0])["will_return"] is None


def test_default_basket_and_recommendation_lines():
    metrics = {"top_rule": {"antecedent_codes": "111,222"}}
    assert demo.default_basket(metrics, ["222", "333"]) == ["222"]
    assert demo.default_basket({}, ["222"]) == []
    recs = pd.DataFrame(
        {
            "rank": [2, 1],
            "stock_code": ["9", "8"],
            "description": ["MUG", "PLATE"],
            "score": [np.nan, 3.2],
            "source": ["popular", "rule"],
            "antecedents": [None, "CUP"],
            "confidence": [np.nan, 0.71],
            "lift": [np.nan, 54.93],
        }
    )
    lines = demo.recommendation_lines(recs)
    assert [x["rank"] for x in lines] == [1, 2]
    assert lines[0]["reason"] == "because you bought CUP (confidence 71%, lift 54.9)"
    assert lines[1]["reason"] == "best seller"
    assert demo.recommendation_lines(None) == []


def test_md_escape():
    assert demo.md_escape("SET OF 3 *CAKE* TINS $5") == "SET OF 3 \\*CAKE\\* TINS \\$5"
    assert demo.md_escape("PLAIN") == "PLAIN"
