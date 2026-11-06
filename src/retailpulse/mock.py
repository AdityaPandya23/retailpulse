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
from retailpulse.config import CUTOFF, HORIZON_DAYS, RULES, SEED, SEGMENT
from retailpulse.features import log1p_skewed

N_CUSTOMERS = 40
# Placeholder for the fake data only: the real k and segment names are chosen in
# P3 from the k-selection evidence (silhouette / elbow / DB / CH in k_selection.parquet).
K = 4
FIRST_ID = 900001
SEGMENT_NAMES = [f"Mock segment {c}" for c in "ABCD"]
N_PRODUCTS = 12


def mock_codes() -> list[str]:
    """5-digit stock codes so features.is_product / baskets treat them as products."""
    return [f"{99000 + i}" for i in range(1, N_PRODUCTS + 1)]


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
    feats = list(SEGMENT.features)  # same clustering inputs as the real segment stage
    X = StandardScaler().fit_transform(log1p_skewed(f)[feats])
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
    loadings.insert(0, "feature", feats)
    seg_ids = seg.astype("int64")
    presets = f[C.PRESET_FEATURES].groupby(seg_ids).median().rename_axis("segment").reset_index()
    presets.insert(1, "segment_name", [SEGMENT_NAMES[s] for s in presets["segment"]])

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
            "uk_share_by_segment": {
                int(s): float(v) for s, v in (table[False] / table.sum(axis=1)).items()
            }
            if False in table
            else {},
            "conclusion": "MOCK: segment membership vs UK/non-UK (placeholder result).",
        },
        "segment_names": dict(enumerate(SEGMENT_NAMES)),
        "k_choice_reason": f"MOCK: k = {K} is a placeholder for the fake data.",
        "conclusion": f"MOCK: k = {K} segments (placeholder; real k and names come from P3).",
    }
    tables = {
        C.SEGMENTS_PARQUET: segments,
        C.SEGMENT_PROFILES_PARQUET: profiles,
        C.K_SELECTION_PARQUET: k_sel,
        C.PCA_LOADINGS_PARQUET: loadings,
        C.SEGMENT_PRESETS_PARQUET: presets,
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
            "threshold": 0.5,
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
            "threshold": 0.5,
            "knn_k": 20,
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
    codes = mock_codes()
    names = [f"MOCK PRODUCT {i:02d}" for i in range(1, N_PRODUCTS + 1)]
    rows = []
    seg_scopes = [f"segment:{n}" for n in SEGMENT_NAMES] if RULES.per_segment else []
    for scope in ["overall", *seg_scopes]:  # same scopes as the real rules stage
        for _ in range(3):
            a, b = rng.choice(N_PRODUCTS, 2, replace=False)
            sa, sb = rng.uniform(0.03, 0.10, 2)
            sab = min(sa, sb) * rng.uniform(0.3, 0.7)
            rows.append(
                {
                    "scope": scope,
                    "antecedents": names[a],
                    "consequents": names[b],
                    "antecedent_codes": codes[a],
                    "consequent_codes": codes[b],
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
                    "source": "rule" if rank <= 2 else "popular",
                }
            )
    top = rules.loc[rules["lift"].idxmax()]
    metrics = {
        "mock": True,
        "min_support": 0.02,
        "min_confidence": 0.3,
        "n_rules": rules.groupby("scope").size().to_dict(),
        "top_rule": top.to_dict(),
        "hit_rate_at_5": 0.25,
        "popularity_hit_rate_at_5": 0.15,
        "n_eval_customers": 20,
        "per_segment": RULES.per_segment,
        "conclusion": "MOCK: association rules (placeholder).",
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
    app_tables, app_metrics = _app_tables(rng, f, lab, seg_tables, rule_tables)
    tables = {
        C.FEATURES_CUTOFF_PARQUET: f,
        C.LABELS_PARQUET: lab,
        **seg_tables,
        **pred_tables,
        **rule_tables,
        **app_tables,
    }
    metrics = {"segment": seg_metrics, **pred_metrics, "rules": rule_metrics, **app_metrics}
    if set(tables) != set(C.MOCKED_PARQUETS):
        raise C.ContractError("mock tables do not match contracts.MOCKED_PARQUETS")
    for path, df in tables.items():
        C.validate(df, path)
    for section, keys in C.P3_METRICS_KEYS.items():
        missing = [k for k in keys if k not in metrics[section]]
        if missing:
            raise C.ContractError(f"mock metrics[{section}] is missing keys: {missing}")
    return tables, metrics


# ======================= P4: everything else the app reads =======================
MOCK_COUNTRIES = ["France", "Germany", "Netherlands", "EIRE"]
DATA_START = pd.Timestamp("2009-12-01")


def mock_sales(rng: np.random.Generator, f: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """Cleaned-schema sales lines for the mock customers (MOCK products, 2009-12 -> 2011-12).

    Every customer buys at least once before the cutoff; returners (labels) also buy in the
    label window, so the mock timeline agrees with the mock labels.
    """
    codes = mock_codes()
    names = dict(
        zip(codes, [f"MOCK PRODUCT {i:02d}" for i in range(1, N_PRODUCTS + 1)], strict=True)
    )
    before_days = (CUTOFF - DATA_START).days
    rows, inv = [], 500000
    returned = dict(
        zip(f["customer_id"].astype(int), lab["returned_flag"].astype(int), strict=True)
    )
    for cid, freq, non_uk in zip(f["customer_id"], f["frequency"], f["non_uk_share"], strict=True):
        country = "United Kingdom" if non_uk < 0.5 else str(rng.choice(MOCK_COUNTRIES))
        days = rng.integers(0, before_days, int(min(freq, 12)))
        dates = [DATA_START + pd.Timedelta(days=int(d), hours=10) for d in days]
        if returned[int(cid)]:
            dates.append(CUTOFF + pd.Timedelta(days=int(rng.integers(0, HORIZON_DAYS)), hours=11))
        for date in dates:
            inv += 1
            for code in rng.choice(codes, int(rng.integers(1, 5)), replace=False):
                qty = int(rng.integers(1, 25))
                price = float(np.round(rng.uniform(0.5, 15.0), 2))
                rows.append((str(inv), str(code), names[code], qty, date, price, int(cid), country))
    sales = pd.DataFrame(rows, columns=C.RAW_COLUMNS)
    sales["invoice_date"] = pd.to_datetime(sales["invoice_date"]).astype("datetime64[ns]")
    sales["line_value"] = np.round(sales["quantity"] * sales["price"], 2)
    return sales.sort_values(["invoice_date", "invoice"]).reset_index(drop=True)


def mock_cube(sales: pd.DataFrame) -> pd.DataFrame:
    import duckdb

    from retailpulse.cube import build_cube

    con = duckdb.connect()
    try:
        con.register("mock_sales", sales)
        con.execute(
            "CREATE VIEW fact_sales AS SELECT *, year(invoice_date) AS year, "
            "month(invoice_date) AS month FROM mock_sales"
        )
        return build_cube(con)
    finally:
        con.close()


def _app_tables(rng, f, lab, seg_tables, rule_tables) -> tuple[dict, dict]:
    """Timeline, rec reasons, cube, EDA and stats tables (+ their metrics) on mock sales."""
    from retailpulse import app_data, eda
    from retailpulse import features as features_mod
    from retailpulse.baskets import rules as R
    from retailpulse.stats import distributions
    from retailpulse.stats import tests as stat_tests

    sales = mock_sales(rng, f, lab)
    full = features_mod.build_features(sales, sales.iloc[0:0], features_mod.full_cutoff(sales))
    timeline = app_data.build_timeline(sales, f["customer_id"])
    # recommendations from the real recommender on the mock sales + mock rules, so "rule"
    # recommendations have a rule that fired (overrides the random ones of _rule_tables)
    segments = seg_tables[C.SEGMENTS_PARQUET]
    before = R.product_lines(sales, end=CUTOFF)
    recs = R.recommend_all(
        before,
        rule_tables[C.RULES_PARQUET],
        f["customer_id"],
        segments if RULES.per_segment else None,
    )
    reasons = app_data.recommendation_reasons(sales, rule_tables[C.RULES_PARQUET], recs, segments)
    eda_tables, eda_summary = eda.run(full, sales)
    dist_entries, qq, freq_table = distributions.run(full, sales)
    test_entries, corr = stat_tests.run(full, sales)
    feats_summary = features_mod.summarize(
        f, lab, full, CUTOFF, HORIZON_DAYS, sales["invoice_date"].max()
    )
    tables = {
        C.RECOMMENDATIONS_PARQUET: recs,
        C.PRODUCTS_PARQUET: R.product_table(before),
        C.TIMELINE_PARQUET: timeline,
        C.REC_REASONS_PARQUET: reasons,
        C.CUBE_PARQUET: mock_cube(sales),
        **eda_tables,
        C.QQ_PARQUET: qq,
        C.FREQUENCY_FIT_PARQUET: freq_table,
        C.CORRELATION_PARQUET: corr,
    }
    app_summary = app_data.summarize(timeline, reasons, recs)
    metrics = {
        "features": {**feats_summary, "mock": True},
        "eda": {**eda_summary, "mock": True},
        "stats": {**dist_entries, **test_entries, "mock": True},
        "app_data": {**app_summary, "mock": True},
    }
    return tables, metrics


def mock_quality_jsons() -> dict[Path, dict]:
    """P1 quality JSONs with the real structure and made-up numbers ("mock": true)."""
    steps = [
        ("remove exact duplicates", 1000, 970, "dropped"),
        ("split returns (invoice starts with C)", 970, 950, "returns table"),
        ("drop quantity <= 0 or price <= 0", 950, 940, "dropped"),
        ("drop missing customer id", 940, 720, "dropped (guest checkouts)"),
        ("impute missing description", 720, 720, "imputed"),
    ]
    mw = {
        "median_missing": 1.0,
        "median_present": 6.0,
        "mean_missing": 3.0,
        "mean_present": 11.0,
        "U": 1000.0,
        "p_value": 1e-6,
        "rank_biserial": -0.5,
        "effect": "large",
    }
    return {
        C.FORMAT_COMPARISON_JSON: {
            "mock": True,
            "rows": 1000,
            "sheets": {"Year 2009-2010": 480, "Year 2010-2011": 520},
            "date_min": "2009-12-01 07:45:00",
            "date_max": "2011-12-09 12:50:00",
            "xlsx_mb": 45.0,
            "parquet_mb": 7.0,
            "xlsx_load_s": 55.0,
            "parquet_load_s": 0.15,
            "size_ratio": 6.4,
            "speedup": 366.0,
        },
        C.CLEANING_STEPS_JSON: {
            "mock": True,
            "raw_rows": 1000,
            "steps": [
                {"step": s, "rows_in": a, "rows_out": b, "removed": a - b, "removed_to": t}
                for s, a, b, t in steps
            ],
        },
        C.MISSING_ID_JSON: {
            "mock": True,
            "n_rows": 950,
            "n_missing": 220,
            "share_missing": 0.23,
            "alpha": 0.05,
            "tests": {
                "mannwhitney_quantity": mw,
                "mannwhitney_price": {**mw, "rank_biserial": 0.3, "effect": "medium"},
                "chi2_uk": {
                    "share_uk_missing": 0.99,
                    "share_uk_present": 0.9,
                    "chi2": 50.0,
                    "dof": 1,
                    "p_value": 1e-6,
                    "cramers_v": 0.14,
                    "effect": "small",
                },
            },
            "invoices_with_missing": 30,
            "share_of_those_fully_missing": 1.0,
            "verdict": "MOCK: not MCAR; consistent with MAR (placeholder).",
        },
        C.WAREHOUSE_SUMMARY_JSON: {
            "mock": True,
            "row_counts": {
                "fact_sales": 720,
                "fact_returns": 20,
                "dim_customer": N_CUSTOMERS,
                "dim_product": N_PRODUCTS,
                "dim_date": 739,
                "dim_country": 5,
            },
            "total_revenue": 50000.0,
            "returns_value": -2000.0,
            "revenue_by_year": [
                {"year": 2009, "revenue": 2000.0, "invoices": 10},
                {"year": 2010, "revenue": 24000.0, "invoices": 120},
                {"year": 2011, "revenue": 24000.0, "invoices": 120},
            ],
            "top_countries": [{"country": "United Kingdom", "revenue": 40000.0}],
        },
    }


def build_mock_models(
    tables: dict[Path, pd.DataFrame], metrics: dict | None = None
) -> dict[Path, dict]:
    """Tiny fitted models with the real joblib payload layout (raw features in)."""
    from retailpulse.predict.classify import make_classifier
    from retailpulse.predict.dataset import FEATURES
    from retailpulse.predict.regress import EXPECTED_SPEND_NOTE, TARGET_NAME, make_regressor
    from retailpulse.segment.cluster import segment_pipeline

    f = tables[C.FEATURES_CUTOFF_PARQUET]
    lab = tables[C.LABELS_PARQUET]
    seg = tables[C.SEGMENTS_PARQUET]
    # same preprocessing + KMeans settings as _segment_tables -> the same labels (= segment ids)
    feats = list(SEGMENT.features)
    seg_pipe = segment_pipeline(feats, K, SEED).fit(f[feats])
    pca = PCA(n_components=3, random_state=SEED).fit(seg_pipe[:-1].transform(f[feats]))
    names = dict(zip(seg["segment"].astype(int), seg["segment_name"].astype(str), strict=True))
    df = f.merge(lab, on="customer_id")
    best = ((metrics or {}).get("classify") or {}).get("best_model", C.CLASSIFIERS[0])
    X, y = df[FEATURES], df["returned_flag"]
    pipes = {m: make_classifier(m, k=5).fit(X, y) for m in C.CLASSIFIERS}
    clf = pipes[best] if best in pipes else make_classifier(best, k=5).fit(X, y)
    ret = df[df["next_spend"] > 0]
    reg = make_regressor("multiple_lr").fit(ret[FEATURES], np.log1p(ret["next_spend"]))
    return {
        C.SEGMENT_MODEL_JOBLIB: {
            "pipeline": seg_pipe,
            "features": feats,
            "label_to_segment": {k: k for k in range(K)},
            "segment_names": {k: names.get(k, SEGMENT_NAMES[k]) for k in range(K)},
            "pca": pca,
            "mock": True,
        },
        C.CLASSIFIER_JOBLIB: {
            "pipeline": clf,
            "features": list(FEATURES),
            "model": best,
            "threshold": 0.5,
            "mock": True,
        },
        C.CLASSIFIERS_JOBLIB: {
            "pipelines": pipes,
            "thresholds": dict.fromkeys(pipes, 0.5),  # = the mock classify thresholds
            "features": list(FEATURES),
            "best_model": best,
            "mock": True,
        },
        C.REGRESSOR_JOBLIB: {
            "pipeline": reg,
            "features": list(FEATURES),
            "model": "multiple_lr",
            "target": TARGET_NAME,
            "note": EXPECTED_SPEND_NOTE,
            "mock": True,
        },
    }
