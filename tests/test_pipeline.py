import numpy as np

from retailpulse import contracts as C
from retailpulse.pipeline import STAGES


def test_stage_order():
    names = [s.name for s in STAGES]
    assert names == ["ingest", "clean", "warehouse", "features", "stats", "cube", "eda", "mock"]


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
