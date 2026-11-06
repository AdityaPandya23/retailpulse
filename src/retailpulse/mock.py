"""Tiny synthetic artifacts matching every P3 contract (+ the feature/label files they join to).

Written to artifacts/mock/ (tracked in git) so the app (P4) can be built before P3 finishes.
Nothing comes from the real data: customer IDs start at 900001, products are "MOCK PRODUCT nn",
segment names are placeholders (real names come from the P3 rules or config.yaml), and every metrics
section has "mock": true so the app can show a banner.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from retailpulse import contracts as C
from retailpulse.config import SEED
from retailpulse.features import log1p_skewed

N_CUSTOMERS = 40
# Placeholder for the fake data only: the real k and segment names are chosen in
# P3 from the k-selection evidence (silhouette / elbow / DB / CH in k_selection.parquet).
K = 4
FIRST_ID = 900001
SEGMENT_NAMES = [f"Mock segment {c}" for c in "ABCD"]
N_PRODUCTS = 12


def _z(x) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    return (x - x.mean()) / x.std()


def _sigmoid(x) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def mock_features(rng: np.random.Generator, n: int = N_CUSTOMERS) -> pd.DataFrame:
    frequency = 1 + rng.negative_binomial(1, 0.2, n)
    basket = np.round(rng.lognormal(5.8, 0.6, n), 2)
    tenure = rng.integers(60, 700, n)
    recency = np.minimum(rng.integers(1, 365, n), tenure)
    gap = np.where(frequency > 1, (tenure - recency) / np.maximum(frequency - 1, 1), tenure)
    return pd.DataFrame(
        {
            "customer_id": np.arange(FIRST_ID, FIRST_ID + n, dtype="int64"),
            "recency": recency.astype("int64"),
            "frequency": frequency.astype("int64"),
            "monetary": np.round(frequency * basket, 2),
            "tenure": tenure.astype("int64"),
            "avg_basket_value": basket,
            "mean_interpurchase_days": np.round(gap, 1).astype("float64"),
            "n_distinct_products": np.minimum(rng.integers(1, 15, n) * frequency, 400),
            "return_rate": np.round(np.where(rng.random(n) < 0.6, 0.0, rng.uniform(0, 0.3, n)), 3),
            "non_uk_share": np.where(rng.random(n) < 0.8, 0.0, 1.0),
        }
    )


def _true_logit(f: pd.DataFrame) -> np.ndarray:
    return -0.3 - 1.0 * _z(np.log1p(f["recency"])) + 1.0 * _z(np.log1p(f["frequency"]))


def mock_labels(rng: np.random.Generator, f: pd.DataFrame) -> pd.DataFrame:
    flag = (rng.random(len(f)) < _sigmoid(_true_logit(f))).astype("int64")
    spend = np.where(flag == 1, np.round(rng.lognormal(6.0, 0.8, len(f)), 2), 0.0)
    return pd.DataFrame(
        {"customer_id": f["customer_id"], "returned_flag": flag, "next_spend": spend}
    )


def _segment_tables(rng, f: pd.DataFrame) -> tuple[dict[Path, pd.DataFrame], dict]:
    X = StandardScaler().fit_transform(log1p_skewed(f)[C.FEATURE_COLUMNS])
    k_rows, labels_by_k = [], {}
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        labels_by_k[k] = km.labels_
        k_rows.append(
            {
                "k": k,
                "wcss": float(km.inertia_),
                "silhouette": float(silhouette_score(X, km.labels_)),
                "davies_bouldin": float(davies_bouldin_score(X, km.labels_)),
                "calinski_harabasz": float(calinski_harabasz_score(X, km.labels_)),
            }
        )
    k_sel = pd.DataFrame(k_rows)
    seg = labels_by_k[K]
    pca = PCA(n_components=3, random_state=SEED).fit(X)
    coords = pca.transform(X)

    def quartile(s: pd.Series, reverse: bool = False) -> np.ndarray:
        q = pd.qcut(s.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype("int64").to_numpy()
        return 5 - q if reverse else q

    segments = pd.DataFrame(
        {
            "customer_id": f["customer_id"],
            "segment": seg.astype("int64"),
            "segment_name": [SEGMENT_NAMES[s] for s in seg],
            "r_score": quartile(f["recency"], reverse=True),  # recent = 4
            "f_score": quartile(f["frequency"]),
            "m_score": quartile(f["monetary"]),
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "pc3": coords[:, 2],
        }
    )
    joined = f.assign(segment=seg)
    profiles = joined.groupby("segment")[C.FEATURE_COLUMNS].mean().reset_index()
    sizes = joined.groupby("segment").agg(n=("customer_id", "size"), rev=("monetary", "sum"))
    profiles.insert(1, "segment_name", [SEGMENT_NAMES[s] for s in profiles["segment"]])
    profiles.insert(2, "n_customers", sizes["n"].to_numpy())
    profiles.insert(3, "customer_share", (sizes["n"] / sizes["n"].sum()).to_numpy())
    profiles.insert(4, "revenue_share", (sizes["rev"] / sizes["rev"].sum()).to_numpy())
    loadings = pd.DataFrame(pca.components_.T, columns=["pc1", "pc2", "pc3"])
    loadings.insert(0, "feature", C.FEATURE_COLUMNS)

    chosen = k_sel.loc[k_sel["k"] == K].iloc[0]
    table = pd.crosstab(seg, f["non_uk_share"] > 0.5)
    chi2, p, dof, _ = stats.chi2_contingency(table)
    cramers_v = float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))
    ari = np.round(rng.uniform(0.82, 0.95, 10), 3)
    metrics = {
        "mock": True,
        "k": K,
        "silhouette": float(chosen["silhouette"]),
        "davies_bouldin": float(chosen["davies_bouldin"]),
        "calinski_harabasz": float(chosen["calinski_harabasz"]),
        "stability_ari": {"mean": float(ari.mean()), "std": float(ari.std()), "values": ari},
        "pca_explained_variance": pca.explained_variance_ratio_,
        "baseline": {
            "kmeans": {"silhouette": float(chosen["silhouette"]), "auc_returned": 0.71},
            "rfm_quartile": {"silhouette": 0.18, "auc_returned": 0.68},
        },
        "chi2_segment_uk": {
            "chi2": float(chi2),
            "dof": int(dof),
            "p_value": float(p),
            "cramers_v": cramers_v,
            "conclusion": "MOCK: segment membership vs UK/non-UK (placeholder result).",
        },
        "conclusion": f"MOCK: k = {K} segments (placeholder; real k and names come from P3).",
    }
    tables = {
        C.SEGMENTS_PARQUET: segments,
        C.SEGMENT_PROFILES_PARQUET: profiles,
        C.K_SELECTION_PARQUET: k_sel,
        C.PCA_LOADINGS_PARQUET: loadings,
    }
    return tables, metrics


def _prediction_tables(rng, f: pd.DataFrame, lab: pd.DataFrame) -> tuple[dict, dict]:
    y = lab["returned_flag"].to_numpy()
    _, test_idx = train_test_split(np.arange(len(f)), test_size=0.3, stratify=y, random_state=SEED)
    split = np.where(np.isin(np.arange(len(f)), test_idx), "test", "train")
    test = split == "test"
    logit = _true_logit(f)

    cls_rows, roc_rows, pr_rows, models = [], [], [], {}
    for i, model in enumerate(C.CLASSIFIERS):
        p = _sigmoid(logit * (0.6 + 0.1 * i) + rng.normal(0, 0.6, len(f)))
        pred = (p >= 0.5).astype("int64")
        cls_rows.append(
            pd.DataFrame(
                {
                    "customer_id": f["customer_id"],
                    "model": model,
                    "split": split,
                    "p_return": p,
                    "pred_return": pred,
                    "returned_flag": y,
                }
            )
        )
        fpr, tpr, thr = roc_curve(y[test], p[test])
        roc_rows.append(pd.DataFrame({"model": model, "fpr": fpr, "tpr": tpr, "threshold": thr}))
        prec, rec, thr_pr = precision_recall_curve(y[test], p[test])
        pr_rows.append(
            pd.DataFrame(
                {"model": model, "recall": rec, "precision": prec, "threshold": [*thr_pr, 1.0]}
            )
        )
        tn, fp, fn, tp = confusion_matrix(y[test], pred[test], labels=[0, 1]).ravel()
        auc = float(roc_auc_score(y[test], p[test]))
        cv = np.clip(auc + rng.normal(0, 0.03, 5), 0, 1)
        models[model] = {
            "cv_auc_mean": float(cv.mean()),
            "cv_auc_std": float(cv.std(ddof=1)),
            "cv_aucs": cv,
            "test_auc": auc,
            "test_ap": float(average_precision_score(y[test], p[test])),
            "accuracy": float((pred[test] == y[test]).mean()),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
    classification = pd.concat(cls_rows, ignore_index=True)
    best = max(models, key=lambda m: models[m]["test_auc"])

    k = np.arange(1, 101)
    knn = pd.DataFrame(
        {
            "k": k,
            "train_error": 0.22 * (1 - np.exp(-k / 10)),
            "test_error": 0.22 + 0.12 / np.sqrt(k) + 0.0007 * k,
        }
    )

    y_true = np.log1p(lab["next_spend"].to_numpy())
    reg_rows, reg_models = [], {}
    for i, model in enumerate(C.REGRESSORS):
        w = 0.4 + 0.2 * i
        y_pred = w * y_true + (1 - w) * y_true.mean() + rng.normal(0, 0.8, len(f))
        reg_rows.append(
            pd.DataFrame(
                {
                    "customer_id": f["customer_id"],
                    "model": model,
                    "split": split,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
        )
        err = y_pred[test] - y_true[test]
        ss_tot = ((y_true[test] - y_true[test].mean()) ** 2).sum()
        reg_models[model] = {
            "rmse": float(np.sqrt((err**2).mean())),
            "mae": float(np.abs(err).mean()),
            "r2": float(1 - (err**2).sum() / ss_tot),
        }
    best_reg = min(reg_models, key=lambda m: reg_models[m]["rmse"])

    best_test = classification[
        (classification["model"] == best) & (classification["split"] == "test")
    ]
    predicted = best_test[best_test["pred_return"] == 1]
    share = float(predicted["returned_flag"].mean()) if len(predicted) else 0.0
    base = float(best_test["returned_flag"].mean())
    metrics = {
        "classify": {
            "mock": True,
            "models": models,
            "best_model": best,
            "paired_t_auc": {
                "model_a": best,
                "model_b": "logistic",
                "t": 1.2,
                "p_value": 0.3,
                "conclusion": "MOCK: per-fold AUCs of the best model vs logistic (placeholder).",
            },
            "conclusion": f"MOCK: best classifier is {best} (placeholder).",
        },
        "regress": {
            "mock": True,
            "target": "log1p(next_spend)",
            "models": reg_models,
            "best_model": best_reg,
            "conclusion": f"MOCK: best regressor is {best_reg} (placeholder).",
        },
        "backtest": {
            "mock": True,
            "model": best,
            "threshold": 0.5,
            "n_customers": int(len(best_test)),
            "n_predicted_return": int(len(predicted)),
            "share_actually_returned": share,
            "base_rate": base,
            "conclusion": (
                f"MOCK: we predicted {len(predicted)} customers would return; {share:.0%} did "
                f"(base rate {base:.0%})."
            ),
        },
    }
    tables = {
        C.CLASSIFICATION_PARQUET: classification,
        C.ROC_PARQUET: pd.concat(roc_rows, ignore_index=True),
        C.PR_PARQUET: pd.concat(pr_rows, ignore_index=True),
        C.KNN_CURVE_PARQUET: knn,
        C.REGRESSION_PARQUET: pd.concat(reg_rows, ignore_index=True),
    }
    return tables, metrics


def _rule_tables(rng, f: pd.DataFrame) -> tuple[dict, dict]:
    codes = [f"MOCK{i:02d}" for i in range(1, N_PRODUCTS + 1)]
    names = [f"MOCK PRODUCT {i:02d}" for i in range(1, N_PRODUCTS + 1)]
    rows = []
    for scope in ["overall", *(f"segment:{n}" for n in SEGMENT_NAMES)]:
        for _ in range(3):
            a, b = rng.choice(N_PRODUCTS, 2, replace=False)
            sa, sb = rng.uniform(0.03, 0.10, 2)
            sab = min(sa, sb) * rng.uniform(0.3, 0.7)
            rows.append(
                {
                    "scope": scope,
                    "antecedents": names[a],
                    "consequents": names[b],
                    "support": sab,
                    "confidence": sab / sa,
                    "lift": sab / (sa * sb),
                }
            )
    rules = pd.DataFrame(rows)
    rec_rows = []
    for cid in f["customer_id"]:
        picks = rng.choice(N_PRODUCTS, 5, replace=False)
        scores = np.sort(rng.uniform(0.1, 0.9, 5))[::-1]
        for rank, (j, s) in enumerate(zip(picks, scores, strict=True), start=1):
            rec_rows.append(
                {
                    "customer_id": int(cid),
                    "rank": rank,
                    "stock_code": codes[j],
                    "description": names[j],
                    "score": float(s),
                }
            )
    top = rules.loc[rules["lift"].idxmax()]
    metrics = {
        "mock": True,
        "min_support": 0.02,
        "min_confidence": 0.3,
        "n_rules": rules.groupby("scope").size().to_dict(),
        "top_rule": top.to_dict(),
        "conclusion": "MOCK: association rules per segment (placeholder).",
    }
    tables = {C.RULES_PARQUET: rules, C.RECOMMENDATIONS_PARQUET: pd.DataFrame(rec_rows)}
    return tables, metrics


def build_mock(seed: int = SEED, n: int = N_CUSTOMERS) -> tuple[dict[Path, pd.DataFrame], dict]:
    """Mock frames keyed by their REAL artifact path (write them to C.mock_path(path))."""
    rng = np.random.default_rng(seed)
    f = mock_features(rng, n)
    lab = mock_labels(rng, f)
    seg_tables, seg_metrics = _segment_tables(rng, f)
    pred_tables, pred_metrics = _prediction_tables(rng, f, lab)
    rule_tables, rule_metrics = _rule_tables(rng, f)
    tables = {
        C.FEATURES_CUTOFF_PARQUET: f,
        C.LABELS_PARQUET: lab,
        **seg_tables,
        **pred_tables,
        **rule_tables,
    }
    metrics = {"segment": seg_metrics, **pred_metrics, "rules": rule_metrics}
    if set(tables) != set(C.MOCKED_PARQUETS):
        raise C.ContractError("mock tables do not match contracts.MOCKED_PARQUETS")
    for path, df in tables.items():
        C.validate(df, path)
    for section, keys in C.P3_METRICS_KEYS.items():
        missing = [k for k in keys if k not in metrics[section]]
        if missing:
            raise C.ContractError(f"mock metrics[{section}] is missing keys: {missing}")
    return tables, metrics
