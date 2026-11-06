import os

import numpy as np

from retailpulse import contracts as C
from retailpulse.pipeline import STAGES, Stage, up_to_date


def test_stage_reruns_when_an_input_is_newer(tmp_path):
    inp, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    inp.write_text("x")
    out.write_text("y")
    stage = Stage("s", lambda: None, (out,), inputs=(inp, tmp_path / "missing.parquet"))
    os.utime(inp, (1_000, 1_000))
    os.utime(out, (2_000, 2_000))
    assert up_to_date(stage)  # output newer; a missing input is ignored
    os.utime(inp, (3_000, 3_000))
    assert not up_to_date(stage)  # e.g. config.yaml or features edited after the run
    out.unlink()
    assert not up_to_date(stage)


def test_downstream_stages_declare_their_inputs():
    by_name = {s.name: s for s in STAGES}
    assert C.FEATURES_FULL_PARQUET in by_name["stats"].inputs
    assert C.FEATURES_FULL_PARQUET in by_name["eda"].inputs
    assert any(p.name == "config.yaml" for p in by_name["features"].inputs)


def test_stage_order():
    names = [s.name for s in STAGES]
    assert names == [
        *("ingest", "clean", "warehouse", "features", "stats", "cube", "eda", "mock"),
        *("segment", "predict", "rules", "app_data"),
    ]


def test_p3_stages_cover_contracts_and_dependencies():
    by_name = {s.name: s for s in STAGES}
    p3_outputs = {p for n in ("segment", "predict", "rules") for p in by_name[n].outputs}
    p3_dirs = (C.SEGMENTS_DIR, C.PREDICTIONS_DIR, C.RULES_DIR)
    p3_parquets = {p for p in C.MOCKED_PARQUETS if p.parent in p3_dirs}
    assert p3_parquets <= p3_outputs
    assert set(C.MODEL_JOBLIBS) <= p3_outputs
    metrics = {m for n in ("segment", "predict", "rules") for m in by_name[n].metrics}
    assert metrics == set(C.P3_METRICS_KEYS)
    assert C.SEGMENTS_PARQUET in by_name["rules"].inputs  # per-segment scopes stay fresh
    for n in ("segment", "predict", "rules"):
        assert C.FEATURES_CUTOFF_PARQUET in by_name[n].inputs


def test_p2_parquet_outputs_are_registered_contracts():
    for stage in STAGES:
        if stage.name in ("features", "stats", "cube", "eda"):
            for path in stage.outputs:
                assert path in C.PARQUET_CONTRACTS, path
    mock = next(s for s in STAGES if s.name == "mock")
    assert set(mock.outputs) == set(C.MOCK_OUTPUTS)


def test_mock_path():
    assert C.mock_path(C.SEGMENTS_PARQUET) == C.MOCK_DIR / "segments" / "segments.parquet"
    assert C.mock_path(C.METRICS_JSON) == C.MOCK_DIR / "metrics.json"


def test_update_metrics_keeps_other_sections(tmp_path):
    path = tmp_path / "metrics.json"
    C.update_metrics("a", {"x": np.float64(1.5), "n": np.int64(3), "bad": float("nan")}, path)
    C.update_metrics("b", {"y": np.array([1, 2])}, path)
    C.update_metrics("a", {"x": 2.0}, path)  # replaces only section a
    assert C.read_metrics(path) == {"a": {"x": 2.0}, "b": {"y": [1, 2]}}
    assert C.has_metrics(("a", "b"), path) and C.has_metrics((), path)
    assert not C.has_metrics(("c",), path)
    assert not list(tmp_path.glob("*.tmp"))


def test_to_jsonable_cleans_nan_and_numpy():
    out = C.to_jsonable({"a": [np.float32(0.5), float("inf")], "b": np.bool_(True)})
    assert out == {"a": [0.5, None], "b": True}


def test_app_data_stage_wiring():
    by_name = {s.name: s for s in STAGES}
    st = by_name["app_data"]
    assert set(st.outputs) == {C.TIMELINE_PARQUET, C.REC_REASONS_PARQUET}
    for p in (C.SALES_PARQUET, C.FEATURES_CUTOFF_PARQUET, C.RULES_PARQUET, C.SEGMENTS_PARQUET):
        assert p in st.inputs
    assert st.metrics == ("app_data",)
    # every file the app reads is produced by a stage (or is a P1 quality JSON / metrics)
    produced = {p for s in STAGES for p in s.outputs}
    assert set(C.APP_PARQUETS) | set(C.MODEL_JOBLIBS) <= produced
