"""The committed app artifacts agree with each other: the saved pipelines reproduce the stored
test-window predictions.

Explicit exception to "tests use the synthetic fixture": this test reads ONLY committed app
artifacts (contracts paths under artifacts/, never data/), prints nothing, and skips when a file
is missing. It cannot run on artifacts/mock/: the mock predictions are synthetic, not model output.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from retailpulse import contracts as C
from retailpulse.config import SEED

N_CUSTOMERS = 20
TOL = 1e-9
FILES = (C.CLASSIFICATION_PARQUET, C.CLASSIFIERS_JOBLIB, C.FEATURES_CUTOFF_PARQUET)


@pytest.fixture(scope="module")
def committed():
    missing = [p.name for p in FILES if not p.exists()]
    if missing:
        pytest.skip(f"committed app artifacts missing: {missing}")
    bundle = joblib.load(C.CLASSIFIERS_JOBLIB)
    features = list(bundle["features"])
    preds = pd.read_parquet(
        C.CLASSIFICATION_PARQUET, columns=["customer_id", "model", "split", "p_return"]
    )
    feats = pd.read_parquet(C.FEATURES_CUTOFF_PARQUET, columns=["customer_id", *features])
    return bundle, features, preds, feats


def test_saved_pipelines_reproduce_stored_test_predictions(committed):
    bundle, features, preds, feats = committed
    test = preds[preds["split"] == "test"]
    ids = pd.Series(test["customer_id"].unique()).sample(N_CUSTOMERS, random_state=SEED).tolist()
    rows = feats.assign(customer_id=feats["customer_id"].astype(test["customer_id"].dtype))
    rows = rows.set_index("customer_id").loc[ids, features]
    assert set(bundle["pipelines"]) == set(C.CLASSIFIERS)
    for model, pipe in bundle["pipelines"].items():
        stored = (
            test[test["model"] == model].set_index("customer_id").loc[ids, "p_return"].to_numpy()
        )
        live = pipe.predict_proba(rows)[:, 1]
        np.testing.assert_allclose(live, stored, rtol=0, atol=TOL, err_msg=model)
