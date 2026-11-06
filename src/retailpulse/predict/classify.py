"""Return classifiers: will a customer buy again in the next horizon_days?

Protocol (no test information is used for any choice):
  * the same StratifiedKFold(cv_folds, shuffle, SEED) folds on TRAIN for every model;
  * KNN k = argmax mean CV ROC-AUC over 1..knn_k_max (fast scan: one neighbour search per fold);
  * best model = max mean CV ROC-AUC on train;
  * each model's threshold = max F1 on its train out-of-fold probabilities;
  * the test window is scored once with pipelines refit on the full train set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline

from retailpulse import contracts as C
from retailpulse.config import ALPHA, PREDICT, SEED
from retailpulse.predict.dataset import FEATURES, TARGET, preprocessor


def make_classifier(name: str, k: int = 15, qda_reg_param: float = PREDICT.qda_reg_param):
    """Raw feature DataFrame -> log1p(skewed) -> StandardScaler -> model."""
    models = {
        "logistic": lambda: LogisticRegression(max_iter=2000, random_state=SEED),
        "lda": LinearDiscriminantAnalysis,
        "qda": lambda: QuadraticDiscriminantAnalysis(reg_param=qda_reg_param),
        "naive_bayes": GaussianNB,
        "knn": lambda: KNeighborsClassifier(n_neighbors=k),
    }
    if name not in models:
        raise ValueError(f"unknown classifier {name!r}")
    return Pipeline([*preprocessor().steps, ("model", models[name]())])


def make_folds(y, n_folds: int, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = np.asarray(y)
    return list(skf.split(np.zeros(len(y)), y))


def oof_predict(pipe, X: pd.DataFrame, y, folds) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold P(return) and the per-fold ROC-AUCs (= cross_val_predict / cross_val_score)."""
    y = np.asarray(y)
    oof = np.full(len(y), np.nan)
    aucs = []
    for tr, va in folds:
        model = clone(pipe).fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y[va], oof[va]))
    return oof, np.asarray(aucs)


def _knn_votes(X_fit, y_fit, X_query, k_max: int) -> np.ndarray:
    """(n_query, k_max) running share of positive neighbours: column k-1 = P(return) for k."""
    nn = NearestNeighbors(n_neighbors=k_max).fit(X_fit)
    idx = nn.kneighbors(X_query, return_distance=False)
    return np.cumsum(np.asarray(y_fit)[idx], axis=1) / np.arange(1, k_max + 1)


def knn_k_scan(X: pd.DataFrame, y, folds, k_max: int) -> pd.DataFrame:
    """Mean / sd CV ROC-AUC for every k in 1..k_max (one neighbour search per fold)."""
    y = np.asarray(y)
    k_max = int(min(k_max, min(len(tr) for tr, _ in folds)))
    aucs = np.empty((len(folds), k_max))
    for f, (tr, va) in enumerate(folds):
        prep = preprocessor().fit(X.iloc[tr])
        votes = _knn_votes(prep.transform(X.iloc[tr]), y[tr], prep.transform(X.iloc[va]), k_max)
        aucs[f] = [roc_auc_score(y[va], votes[:, j]) for j in range(k_max)]
    return pd.DataFrame(
        {"k": np.arange(1, k_max + 1), "cv_auc": aucs.mean(0), "cv_auc_sd": aucs.std(0, ddof=1)}
    )


def knn_bias_variance(
    train: pd.DataFrame, test: pd.DataFrame, k_max: int, k_scan: pd.DataFrame | None = None
) -> pd.DataFrame:
    """KNN fitted on the full train set: in-sample train error vs test error for k = 1..k_max.

    Error = 1 - accuracy of the majority vote (p > 0.5; a tie predicts "no return", as
    sklearn's argmax does). Train error is in-sample (each point is its own nearest neighbour),
    so it is ~0 at k = 1 and grows with k (less variance, more bias).
    """
    k_max = int(min(k_max, len(train)))
    prep = preprocessor().fit(train[FEATURES])
    Xtr, Xte = prep.transform(train[FEATURES]), prep.transform(test[FEATURES])
    ytr, yte = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    v_tr = _knn_votes(Xtr, ytr, Xtr, k_max)
    v_te = _knn_votes(Xtr, ytr, Xte, k_max)
    out = pd.DataFrame(
        {
            "k": np.arange(1, k_max + 1),
            "train_error": ((v_tr > 0.5) != ytr[:, None]).mean(0),
            "test_error": ((v_te > 0.5) != yte[:, None]).mean(0),
            "train_auc": [roc_auc_score(ytr, v_tr[:, j]) for j in range(k_max)],
            "test_auc": [roc_auc_score(yte, v_te[:, j]) for j in range(k_max)],
        }
    )
    if k_scan is not None:
        out = out.merge(k_scan, on="k", how="left")
    return out


def max_f1_threshold(y, p) -> float:
    """Probability cut-off that maximises F1 (predict return when p >= threshold)."""
    precision, recall, thr = precision_recall_curve(np.asarray(y), np.asarray(p))
    p_, r_ = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p_ * r_, p_ + r_, out=np.zeros_like(p_), where=(p_ + r_) > 0)
    return float(thr[int(np.argmax(f1))])


def _paired_t(a: np.ndarray, b: np.ndarray, n_train: int, n_val: int) -> dict:
    """Paired t on per-fold AUCs + Nadeau-Bengio corrected t (folds share training data)."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    t, p = stats.ttest_rel(a, b)
    var = d.var(ddof=1)
    if var > 0:
        t_c = d.mean() / np.sqrt((1 / k + n_val / n_train) * var)
        p_c = 2 * stats.t.sf(abs(t_c), df=k - 1)
    else:
        t_c, p_c = np.nan, np.nan
    return {"t": float(t), "p_value": float(p), "t_corrected": t_c, "p_value_corrected": p_c}


@dataclass
class ClassifyResult:
    metrics: dict
    predictions: pd.DataFrame  # C.CLASSIFICATION_COLUMNS
    roc: pd.DataFrame
    pr: pd.DataFrame
    knn_curve: pd.DataFrame
    best_model: str
    threshold: float
    pipeline: Pipeline  # best model refit on the full train set
    pipelines: dict = field(default_factory=dict)


def run_classification(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cv_folds: int = PREDICT.cv_folds,
    knn_k_max: int = PREDICT.knn_k_max,
    qda_reg_param: float = PREDICT.qda_reg_param,
    seed: int = SEED,
    season_note: str = "",
) -> ClassifyResult:
    """season_note: optional text appended to the base-rate sentence of the conclusion."""
    Xtr, ytr = train[FEATURES], train[TARGET].to_numpy()
    Xte, yte = test[FEATURES], test[TARGET].to_numpy()
    folds = make_folds(ytr, cv_folds, seed)

    k_scan = knn_k_scan(Xtr, ytr, folds, knn_k_max)
    best_k = int(k_scan.loc[k_scan["cv_auc"].idxmax(), "k"])

    models, pipelines, pred_rows, roc_rows, pr_rows, cv = {}, {}, [], [], [], {}
    for name in C.CLASSIFIERS:
        pipe = make_classifier(name, k=best_k, qda_reg_param=qda_reg_param)
        oof, aucs = oof_predict(pipe, Xtr, ytr, folds)
        thr = max_f1_threshold(ytr, oof)
        fitted = clone(pipe).fit(Xtr, ytr)
        p_te = fitted.predict_proba(Xte)[:, 1]
        pred_te = (p_te >= thr).astype("int64")
        tn, fp, fn, tp = confusion_matrix(yte, pred_te, labels=[0, 1]).ravel()
        cv[name] = aucs
        pipelines[name] = fitted
        models[name] = {
            "cv_auc_mean": float(aucs.mean()),
            "cv_auc_std": float(aucs.std(ddof=1)),
            "cv_aucs": aucs.tolist(),
            "train_oof_auc": float(roc_auc_score(ytr, oof)),
            "threshold": thr,
            "test_auc": float(roc_auc_score(yte, p_te)),
            "test_ap": float(average_precision_score(yte, p_te)),
            "accuracy": float((pred_te == yte).mean()),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
        if name == "knn":
            models[name]["k"] = best_k
        for split, frame, p in (("train", train, oof), ("test", test, p_te)):
            pred_rows.append(
                pd.DataFrame(
                    {
                        "customer_id": frame["customer_id"].to_numpy(),
                        "model": name,
                        "split": split,
                        "p_return": p,
                        "pred_return": (p >= thr).astype("int64"),
                        "returned_flag": frame[TARGET].to_numpy(),
                    }
                )
            )
        fpr, tpr, roc_thr = roc_curve(yte, p_te, drop_intermediate=True)
        roc_rows.append(pd.DataFrame({"model": name, "fpr": fpr, "tpr": tpr, "threshold": roc_thr}))
        prec, rec, pr_thr = precision_recall_curve(yte, p_te, drop_intermediate=True)
        pr_rows.append(
            pd.DataFrame(
                {
                    "model": name,
                    "recall": rec,
                    "precision": prec,
                    "threshold": [*pr_thr, np.inf],  # last point (recall 0) has no cut-off
                }
            )
        )

    # max mean CV AUC on train; ties -> earlier (simpler) model in C.CLASSIFIERS
    best = max(C.CLASSIFIERS, key=lambda m: (models[m]["cv_auc_mean"], -C.CLASSIFIERS.index(m)))
    other = "logistic"
    if best == "logistic":
        rest = [m for m in C.CLASSIFIERS if m != "logistic"]
        other = max(rest, key=lambda m: (models[m]["cv_auc_mean"], -C.CLASSIFIERS.index(m)))
    n_val = int(np.mean([len(va) for _, va in folds]))
    pt = _paired_t(cv[best], cv[other], len(train) - n_val, n_val)
    diff = float(np.mean(cv[best] - cv[other]))
    sig = pt["p_value"] < ALPHA
    pt = {
        "model_a": best,
        "model_b": other,
        "mean_auc_diff": diff,
        **pt,
        "conclusion": (
            f"Across the {cv_folds} train folds {best} beats {other} by {diff:+.4f} ROC-AUC on "
            f"average (paired t = {pt['t']:.2f}, p = {pt['p_value']:.3g}): "
            + (
                "a significant difference"
                if sig
                else f"not significant at alpha {ALPHA}: they rank customers about equally well"
            )
            + (
                f"; the corrected t (folds share training rows) gives p = "
                f"{pt['p_value_corrected']:.3g}."
                if np.isfinite(pt["p_value_corrected"])
                else "."
            )
        ),
    }

    base_tr, base_te = float(ytr.mean()), float(yte.mean())
    majority = int(base_te >= 0.5)
    ref_acc = float((yte == majority).mean())
    bm = models[best]
    metrics = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "base_rate_train": base_tr,
        "base_rate_test": base_te,
        "cv_folds": int(cv_folds),
        "selection": "max mean CV ROC-AUC on train (test never used for choices)",
        "threshold_rule": "max F1 on train out-of-fold probabilities, per model",
        "knn_k": best_k,
        "knn_k_note": "k chosen by the same CV folds, so the KNN CV AUC is slightly optimistic",
        "models": models,
        "best_model": best,
        "threshold": bm["threshold"],
        "paired_t_auc": pt,
        "reference": {
            "auc": 0.5,
            "majority_class": majority,
            "accuracy": ref_acc,
            "ap": base_te,
            "note": "always predicting the majority class of the test window / random ranking",
        },
        "conclusion": (
            f"{best} has the best {cv_folds}-fold CV ROC-AUC on the train snapshot "
            f"({bm['cv_auc_mean']:.3f} ± {bm['cv_auc_std']:.3f}); on the future window it scores "
            f"ROC-AUC {bm['test_auc']:.3f} (vs 0.5 for guessing) and accuracy "
            f"{bm['accuracy']:.1%} at its train-chosen threshold {bm['threshold']:.2f} "
            f"(always guessing the majority class: {ref_acc:.1%}). Base rate differs: "
            f"{base_tr:.1%} returned in the train window vs {base_te:.1%} in the test window"
            f"{season_note}."
        ),
    }
    return ClassifyResult(
        metrics=metrics,
        predictions=pd.concat(pred_rows, ignore_index=True),
        roc=pd.concat(roc_rows, ignore_index=True),
        pr=pd.concat(pr_rows, ignore_index=True),
        knn_curve=knn_bias_variance(train, test, knn_k_max, k_scan),
        best_model=best,
        threshold=bm["threshold"],
        pipeline=pipelines[best],
        pipelines=pipelines,
    )
