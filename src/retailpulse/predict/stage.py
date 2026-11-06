"""Prediction stage: `python -m retailpulse.predict.stage`.

Inputs : C.SALES_PARQUET, C.RETURNS_PARQUET (train snapshot), C.FEATURES_CUTOFF_PARQUET,
         C.LABELS_PARQUET (test snapshot), config.yaml (cutoff, horizon, predict.*).
Outputs: C.CLASSIFICATION_PARQUET, C.ROC_PARQUET, C.PR_PARQUET, C.KNN_CURVE_PARQUET,
         C.REGRESSION_PARQUET, C.CLASSIFIER_JOBLIB, C.REGRESSOR_JOBLIB;
         metrics.json sections "classify", "regress", "backtest".
`compute()` is pure (frames in, tables / metrics / models out); `run()` only does I/O.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import pandas as pd

from retailpulse import contracts as C
from retailpulse.config import CUTOFF, HORIZON_DAYS, PREDICT, SEED
from retailpulse.predict.backtest import backtest
from retailpulse.predict.classify import run_classification
from retailpulse.predict.dataset import FEATURES, build_test, build_train, train_cutoff
from retailpulse.predict.regress import EXPECTED_SPEND_NOTE, TARGET_NAME, run_regression

log = logging.getLogger("retailpulse.predict")

OUTPUTS = (
    C.CLASSIFICATION_PARQUET,
    C.ROC_PARQUET,
    C.PR_PARQUET,
    C.KNN_CURVE_PARQUET,
    C.REGRESSION_PARQUET,
    C.CLASSIFIER_JOBLIB,
    C.REGRESSOR_JOBLIB,
)
INPUTS = (C.SALES_PARQUET, C.RETURNS_PARQUET, C.FEATURES_CUTOFF_PARQUET, C.LABELS_PARQUET)
METRICS_SECTIONS = ("classify", "regress", "backtest")


def compute(
    sales: pd.DataFrame,
    returns: pd.DataFrame,
    features_cutoff: pd.DataFrame,
    labels: pd.DataFrame,
    cutoff=CUTOFF,
    horizon_days: int = HORIZON_DAYS,
    train_offset_days: int = PREDICT.train_offset_days,
    cv_folds: int = PREDICT.cv_folds,
    knn_k_max: int = PREDICT.knn_k_max,
    qda_reg_param: float = PREDICT.qda_reg_param,
    seed: int = SEED,
) -> tuple[dict[Path, pd.DataFrame], dict[str, dict], dict[Path, dict]]:
    """-> (tables keyed by artifact path, metrics sections, joblib payloads keyed by path)."""
    train = build_train(sales, returns, cutoff, train_offset_days, horizon_days)
    test = build_test(features_cutoff, labels)
    t_cut = train_cutoff(cutoff, train_offset_days)

    test_months = pd.date_range(cutoff, periods=horizon_days, freq="D").month
    train_months = pd.date_range(t_cut, periods=horizon_days, freq="D").month
    season_note = ""
    if 11 in test_months and 11 not in train_months:
        season_note = " (only the test window contains the November peak season)"
    cls = run_classification(
        train, test, cv_folds, knn_k_max, qda_reg_param, seed, season_note=season_note
    )
    reg = run_regression(train, test, cv_folds, knn_k_max, seed)
    best_test = cls.predictions[
        (cls.predictions["model"] == cls.best_model) & (cls.predictions["split"] == "test")
    ]
    bt = backtest(
        best_test["returned_flag"],
        best_test["p_return"],
        cls.threshold,
        cls.best_model,
        cutoff,
        horizon_days,
    )

    windows = {
        "train_cutoff": str(t_cut.date()),
        "train_label_window": [
            str(t_cut.date()),
            str((t_cut + pd.Timedelta(days=horizon_days)).date()),
        ],
        "test_cutoff": str(pd.Timestamp(cutoff).date()),
        "test_label_window": [
            str(pd.Timestamp(cutoff).date()),
            str((pd.Timestamp(cutoff) + pd.Timedelta(days=horizon_days)).date()),
        ],
    }
    metrics = {
        "classify": {**windows, **cls.metrics},
        "regress": {**windows, **reg.metrics},
        "backtest": bt,
    }
    tables = {
        C.CLASSIFICATION_PARQUET: cls.predictions,
        C.ROC_PARQUET: cls.roc,
        C.PR_PARQUET: cls.pr,
        C.KNN_CURVE_PARQUET: cls.knn_curve,
        C.REGRESSION_PARQUET: reg.predictions,
    }
    models = {
        C.CLASSIFIER_JOBLIB: {
            "pipeline": cls.pipeline,
            "features": list(FEATURES),
            "model": cls.best_model,
            "threshold": cls.threshold,
        },
        C.REGRESSOR_JOBLIB: {
            "pipeline": reg.pipeline,
            "features": reg.features,
            "model": reg.best_model,
            "target": TARGET_NAME,
            "note": EXPECTED_SPEND_NOTE,
        },
    }
    for path, df in tables.items():
        C.validate(df, path)
    for section, payload in metrics.items():
        missing = [k for k in C.P3_METRICS_KEYS[section] if k not in payload]
        if missing:
            raise C.ContractError(f"metrics[{section}] is missing keys: {missing}")
    return tables, metrics, models


def run() -> dict:
    t0 = time.perf_counter()
    sales = pd.read_parquet(C.SALES_PARQUET)
    returns = pd.read_parquet(C.RETURNS_PARQUET)
    features_cutoff = pd.read_parquet(C.FEATURES_CUTOFF_PARQUET)
    labels = pd.read_parquet(C.LABELS_PARQUET)
    tables, metrics, models = compute(sales, returns, features_cutoff, labels)
    for path, df in tables.items():
        C.validate(df, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    for path, payload in models.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)
    for section, payload in metrics.items():
        C.update_metrics(section, payload)
    c, r, b = metrics["classify"], metrics["regress"], metrics["backtest"]
    return {
        "n_train": c["n_train"],
        "n_test": c["n_test"],
        "best_classifier": c["best_model"],
        "test_auc": round(c["models"][c["best_model"]]["test_auc"], 3),
        "best_regressor": r["best_model"],
        "backtest_share": round(b["share_actually_returned"], 3),
        "seconds": round(time.perf_counter() - t0, 1),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("predict stage done: %s", run())
