"""How much will a returning customer spend? Target = log1p(next_spend) on returners only.

Fit on train returners, evaluate once on test returners. Model choice = lowest mean CV RMSE on
train (KFold(cv_folds, shuffle, SEED)); test RMSE is reported, never used to choose.
App: expected spend = p_return x expm1(prediction). expm1 of a log-scale prediction is closer
to the median than to the mean spend, so this is a conservative (typical-customer) estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline

from retailpulse import contracts as C
from retailpulse.config import PREDICT, SEED
from retailpulse.predict.dataset import FEATURES, preprocessor

TARGET_NAME = "log1p(next_spend)"
EXPECTED_SPEND_NOTE = "expected spend = p_return x expm1(prediction)"
MODEL_FEATURES: dict[str, list[str]] = {
    "simple_lr": ["frequency"],  # log1p(frequency) after the preprocessor
    "multiple_lr": FEATURES,
    "knn": FEATURES,
}


def make_regressor(name: str, k: int = 15) -> Pipeline:
    models = {
        "simple_lr": LinearRegression,
        "multiple_lr": LinearRegression,
        "knn": lambda: KNeighborsRegressor(n_neighbors=k),
    }
    if name not in models:
        raise ValueError(f"unknown regressor {name!r}")
    return Pipeline([*preprocessor().steps, ("model", models[name]())])


def returners(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["next_spend"] > 0].reset_index(drop=True)


def _errors(y, pred) -> dict:
    err = pred - y
    ss_tot = ((y - y.mean()) ** 2).sum()
    gbp = np.expm1(pred) - np.expm1(y)
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2": float(1 - (err**2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "rmse_gbp": float(np.sqrt((gbp**2).mean())),
        "mae_gbp": float(np.abs(gbp).mean()),
    }


def knn_k_scan_reg(X: pd.DataFrame, y, folds, k_max: int) -> pd.DataFrame:
    """Mean CV RMSE for k = 1..k_max (uniform KNN = running mean of the k nearest targets)."""
    y = np.asarray(y, dtype="float64")
    k_max = int(min(k_max, min(len(tr) for tr, _ in folds)))
    rmse = np.empty((len(folds), k_max))
    for f, (tr, va) in enumerate(folds):
        prep = preprocessor().fit(X.iloc[tr])
        nn = NearestNeighbors(n_neighbors=k_max).fit(prep.transform(X.iloc[tr]))
        idx = nn.kneighbors(prep.transform(X.iloc[va]), return_distance=False)
        pred = np.cumsum(y[tr][idx], axis=1) / np.arange(1, k_max + 1)
        rmse[f] = np.sqrt(((pred - y[va][:, None]) ** 2).mean(0))
    return pd.DataFrame({"k": np.arange(1, k_max + 1), "cv_rmse": rmse.mean(0)})


@dataclass
class RegressResult:
    metrics: dict
    predictions: pd.DataFrame  # C.REGRESSION_COLUMNS (+ y_true/y_pred on the log1p scale)
    best_model: str
    pipeline: Pipeline
    features: list[str]


def run_regression(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cv_folds: int = PREDICT.cv_folds,
    knn_k_max: int = PREDICT.knn_k_max,
    seed: int = SEED,
) -> RegressResult:
    tr, te = returners(train), returners(test)
    ytr = np.log1p(tr["next_spend"].to_numpy())
    yte = np.log1p(te["next_spend"].to_numpy())
    folds = list(KFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(tr))

    scan = knn_k_scan_reg(tr[FEATURES], ytr, folds, knn_k_max)
    best_k = int(scan.loc[scan["cv_rmse"].idxmin(), "k"])

    models, fitted, rows = {}, {}, []
    for name in C.REGRESSORS:
        cols = MODEL_FEATURES[name]
        pipe = make_regressor(name, k=best_k)
        oof = np.empty(len(tr))
        fold_rmse = []
        for f_tr, f_va in folds:
            m = clone(pipe).fit(tr[cols].iloc[f_tr], ytr[f_tr])
            oof[f_va] = m.predict(tr[cols].iloc[f_va])
            fold_rmse.append(float(np.sqrt(((oof[f_va] - ytr[f_va]) ** 2).mean())))
        model = clone(pipe).fit(tr[cols], ytr)
        pred = model.predict(te[cols])
        fitted[name] = model
        models[name] = {
            "features": cols,
            "cv_rmse_mean": float(np.mean(fold_rmse)),
            "cv_rmse_std": float(np.std(fold_rmse, ddof=1)),
            **_errors(yte, pred),
        }
        if name == "knn":
            models[name]["k"] = best_k
        if name != "knn":
            lr = model.named_steps["model"]
            models[name]["coefficients_scaled"] = dict(zip(cols, lr.coef_.tolist(), strict=True))
        for split, frame, y, p in (("train", tr, ytr, oof), ("test", te, yte, pred)):
            rows.append(
                pd.DataFrame(
                    {
                        "customer_id": frame["customer_id"].to_numpy(),
                        "model": name,
                        "split": split,
                        "y_true": y,
                        "y_pred": p,
                    }
                )
            )

    best = min(C.REGRESSORS, key=lambda m: (models[m]["cv_rmse_mean"], C.REGRESSORS.index(m)))
    mean_ref = _errors(yte, np.full(len(yte), ytr.mean()))
    bm = models[best]
    lowest_test = min(C.REGRESSORS, key=lambda m: models[m]["rmse"])
    metrics = {
        "target": TARGET_NAME,
        "population": "customers with next_spend > 0 (returners)",
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "selection": "min mean CV RMSE on train (test never used for choices)",
        "train_predictions": "out-of-fold",
        "knn_k": best_k,
        "models": models,
        "best_model": best,
        "reference_mean": {**mean_ref, "note": "predict the train mean of log1p(next_spend)"},
        "expected_spend": EXPECTED_SPEND_NOTE,
        "expected_spend_note": (
            "expm1 of a log-scale prediction estimates a typical (median-like) spend, not the "
            "mean, so expected spend is conservative"
        ),
        "conclusion": (
            f"{best} has the lowest CV RMSE on train ({bm['cv_rmse_mean']:.3f} on the log scale); "
            f"on test returners it reaches RMSE {bm['rmse']:.3f}, R² {bm['r2']:.2f} "
            f"(predicting the mean: RMSE {mean_ref['rmse']:.3f})"
            + (
                f"; lowest test RMSE: {lowest_test}."
                if lowest_test != best
                else ", also the lowest test RMSE."
            )
            + f" In pounds the RMSE is £{bm['rmse_gbp']:,.0f}: spend is hard to predict exactly."
        ),
    }
    return RegressResult(
        metrics=metrics,
        predictions=pd.concat(rows, ignore_index=True),
        best_model=best,
        pipeline=fitted[best],
        features=list(MODEL_FEATURES[best]),
    )
