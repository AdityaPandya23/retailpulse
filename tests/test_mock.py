import json

import pandas as pd
import pytest

from retailpulse import contracts as C
from retailpulse.mock import FIRST_ID, build_mock


@pytest.fixture(scope="module")
def mock():
    return build_mock()


def test_mock_matches_every_contract(mock):
    tables, metrics = mock
    assert set(tables) == set(C.MOCKED_PARQUETS)
    for path, df in tables.items():
        C.validate(df, path)
        assert 0 < len(df) <= 1000, path.name  # tiny: tracked in git
    for section, keys in C.P3_METRICS_KEYS.items():
        assert set(keys) <= set(metrics[section]) and metrics[section]["mock"] is True
    json.dumps(C.to_jsonable(metrics), allow_nan=False)


def test_mock_is_self_consistent(mock):
    tables, _ = mock
    ids = set(tables[C.FEATURES_CUTOFF_PARQUET]["customer_id"])
    assert min(ids) >= FIRST_ID  # never collides with real customer IDs
    for path in (
        C.LABELS_PARQUET,
        C.SEGMENTS_PARQUET,
        C.CLASSIFICATION_PARQUET,
        C.REGRESSION_PARQUET,
        C.RECOMMENDATIONS_PARQUET,
    ):
        assert set(tables[path]["customer_id"]) == ids, path.name
    assert set(tables[C.CLASSIFICATION_PARQUET]["model"]) == set(C.CLASSIFIERS)
    assert set(tables[C.REGRESSION_PARQUET]["model"]) == set(C.REGRESSORS)
    profiles = tables[C.SEGMENT_PROFILES_PARQUET]
    assert profiles["n_customers"].sum() == len(ids)
    assert profiles["revenue_share"].sum() == pytest.approx(1.0)


def test_mock_is_deterministic(mock):
    again, _ = build_mock()
    for path, df in mock[0].items():
        pd.testing.assert_frame_equal(df, again[path])


def test_committed_mock_files_match_contracts():
    if not C.all_exist(list(C.MOCK_OUTPUTS)):
        pytest.skip("artifacts/mock/ not generated yet (python -m retailpulse.pipeline)")
    for path in C.MOCKED_PARQUETS:
        C.validate(pd.read_parquet(C.mock_path(path)), path)
    metrics = C.read_metrics(C.mock_path(C.METRICS_JSON))
    for section, keys in C.P3_METRICS_KEYS.items():
        assert set(keys) <= set(metrics[section])
