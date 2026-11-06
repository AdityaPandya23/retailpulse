"""Artifact contracts: every path the pipeline writes / the app reads, plus required columns.

P1: data layer + quality JSON.  P2: features, stats, cube, EDA tables, metrics.json.
P3 (mocked in artifacts/mock/ until P3 lands): segments, predictions, rules, model metrics.
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from retailpulse.config import ARTIFACTS_DIR, PREDICT, PROCESSED_DIR, RAW_DIR, SEGMENT, XLSX_NAME

# ---- data layer (large, never deployed) ----
RAW_ZIP = RAW_DIR / "online_retail_ii.zip"
RAW_XLSX = RAW_DIR / XLSX_NAME
RAW_PARQUET = RAW_DIR / "online_retail_ii.parquet"
SALES_PARQUET = PROCESSED_DIR / "sales.parquet"
RETURNS_PARQUET = PROCESSED_DIR / "returns.parquet"

# ---- small artifacts (app reads these) ----
QUALITY_DIR = ARTIFACTS_DIR / "quality"
FORMAT_COMPARISON_JSON = QUALITY_DIR / "format_comparison.json"
CLEANING_STEPS_JSON = QUALITY_DIR / "cleaning_steps.json"
MISSING_ID_JSON = QUALITY_DIR / "missing_id.json"
WAREHOUSE_SUMMARY_JSON = QUALITY_DIR / "warehouse_summary.json"

# ---- required columns ----
RAW_COLUMNS = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]
SALES_COLUMNS = [*RAW_COLUMNS, "line_value"]
RETURNS_COLUMNS = [*RAW_COLUMNS, "line_value"]

# ---- metrics: {section: {...}}; every statistical result carries a one-line "conclusion" ----
METRICS_JSON = ARTIFACTS_DIR / "metrics.json"

# ======================= P2: features (one row per customer) =======================
FEATURES_DIR = ARTIFACTS_DIR / "features"
FEATURES_CUTOFF_PARQUET = FEATURES_DIR / "features_cutoff.parquet"  # modelling: P3a AND P3b
LABELS_PARQUET = FEATURES_DIR / "labels.parquet"  # outcomes in [cutoff, cutoff + horizon)
FEATURES_FULL_PARQUET = FEATURES_DIR / "features_full.parquet"  # descriptive EDA / stats only

FEATURE_COLUMNS = [
    "recency",  # days from last purchase day to the cutoff
    "frequency",  # distinct invoices
    "monetary",  # total sales value (GBP)
    "tenure",  # days from first purchase day to the cutoff
    "avg_basket_value",  # monetary / frequency
    "mean_interpurchase_days",  # mean gap between purchase days (single-day customers: tenure)
    "n_distinct_products",  # distinct product StockCodes
    "return_rate",  # returned product value / purchased product value, capped at 1
    "non_uk_share",  # share of spend shipped outside the UK
]
# log1p before distance-based models where it moves skewness towards 0 (metrics.json
# features.skewness_log1p_cutoff). Not recency (0.67 -> -0.97) or tenure (-0.63 -> -2.58):
# log1p over-corrects them; return_rate / non_uk_share are [0, 1] shares and stay as they are.
SKEWED_FEATURES = [
    "frequency",
    "monetary",
    "avg_basket_value",
    "mean_interpurchase_days",
    "n_distinct_products",
]
FEATURES_COLUMNS = ["customer_id", *FEATURE_COLUMNS]
LABELS_COLUMNS = ["customer_id", "returned_flag", "next_spend"]

# ======================= P2: statistics =======================
STATS_DIR = ARTIFACTS_DIR / "stats"
QQ_PARQUET = STATS_DIR / "qq_points.parquet"
FREQUENCY_FIT_PARQUET = STATS_DIR / "frequency_fit.parquet"
CORRELATION_PARQUET = STATS_DIR / "correlation.parquet"
QQ_COLUMNS = ["variable", "dist", "prob", "theoretical", "sample"]
FREQUENCY_FIT_COLUMNS = ["k", "observed", "poisson", "nbinom"]  # counts of customers
CORRELATION_COLUMNS = ["method", "x", "y", "value"]  # pearson | spearman | covariance

# ======================= P2: OLAP cube =======================
CUBE_DIR = ARTIFACTS_DIR / "cube"
CUBE_PARQUET = CUBE_DIR / "sales_cube.parquet"
CUBE_DIMENSIONS = ["country", "year", "month"]
CUBE_COLUMNS = [*CUBE_DIMENSIONS, "grouping_id", "level", "revenue", "orders", "customers"]

# ======================= P2: EDA aggregate tables =======================
EDA_DIR = ARTIFACTS_DIR / "eda"
EDA_HISTOGRAMS_PARQUET = EDA_DIR / "histograms.parquet"
EDA_BOXPLOTS_PARQUET = EDA_DIR / "boxplots.parquet"
EDA_MONTHLY_PARQUET = EDA_DIR / "monthly.parquet"
EDA_COUNTRIES_PARQUET = EDA_DIR / "countries.parquet"
EDA_PARETO_PARQUET = EDA_DIR / "pareto.parquet"
EDA_COHORTS_PARQUET = EDA_DIR / "cohorts.parquet"
EDA_SAMPLE_ROWS_PARQUET = EDA_DIR / "sample_rows.parquet"  # 5 cleaned sales lines (seeded)
SAMPLE_ROWS_COLUMNS = SALES_COLUMNS
HISTOGRAM_COLUMNS = ["feature", "bin", "left", "right", "mid", "count", "scale"]
BOXPLOT_COLUMNS = [
    "feature",
    "group",
    "n",
    "min",
    "q1",
    "median",
    "q3",
    "max",
    "mean",
    "lower_whisker",
    "upper_whisker",
    "n_outliers",
]
MONTHLY_COLUMNS = [
    "month",
    "revenue",
    "orders",
    "customers",
    "new_customers",
    "avg_order_value",
    "is_partial",
]
COUNTRY_COLUMNS = ["country", "is_uk", "revenue", "orders", "customers", "revenue_share"]
PARETO_COLUMNS = ["customer_share", "revenue_share"]  # top x% of customers -> y% of revenue
COHORT_COLUMNS = ["cohort", "period", "cohort_size", "active", "retention"]

# ======================= P3 contracts (mocked until P3 lands) =======================
SEGMENTS_DIR = ARTIFACTS_DIR / "segments"
SEGMENTS_PARQUET = SEGMENTS_DIR / "segments.parquet"  # one row per customer
SEGMENT_PROFILES_PARQUET = SEGMENTS_DIR / "profiles.parquet"  # one row per segment
K_SELECTION_PARQUET = SEGMENTS_DIR / "k_selection.parquet"  # one row per k
PCA_LOADINGS_PARQUET = SEGMENTS_DIR / "pca_loadings.parquet"  # one row per feature
SEGMENT_PRESETS_PARQUET = SEGMENTS_DIR / "presets.parquet"  # one row per segment: median customer
# raw model inputs a preset fills (segment model + classifiers), in FEATURE_COLUMNS order
PRESET_FEATURES = [c for c in FEATURE_COLUMNS if c in SEGMENT.features or c in PREDICT.features]
SEGMENT_PRESETS_COLUMNS = ["segment", "segment_name", *PRESET_FEATURES]
SEGMENTS_COLUMNS = [
    "customer_id",
    "segment",
    "segment_name",
    "r_score",  # RFM-quartile baseline, 1-4 each
    "f_score",
    "m_score",
    "pc1",
    "pc2",
    "pc3",
]
SEGMENT_PROFILE_COLUMNS = [
    "segment",
    "segment_name",
    "n_customers",
    "customer_share",
    "revenue_share",
    *FEATURE_COLUMNS,  # segment means
]
K_SELECTION_COLUMNS = ["k", "wcss", "silhouette", "davies_bouldin", "calinski_harabasz"]
PCA_LOADINGS_COLUMNS = ["feature", "pc1", "pc2", "pc3"]

PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
CLASSIFICATION_PARQUET = PREDICTIONS_DIR / "classification.parquet"  # customer x model
ROC_PARQUET = PREDICTIONS_DIR / "roc_curves.parquet"
PR_PARQUET = PREDICTIONS_DIR / "pr_curves.parquet"
KNN_CURVE_PARQUET = PREDICTIONS_DIR / "knn_bias_variance.parquet"
REGRESSION_PARQUET = PREDICTIONS_DIR / "regression.parquet"  # customer x model
ALL_CLASSIFIERS = ["logistic", "lda", "qda", "naive_bayes", "knn"]  # what classify.py can fit
_UNKNOWN_MODELS = sorted(set(PREDICT.models) - set(ALL_CLASSIFIERS))
if _UNKNOWN_MODELS:
    raise ValueError(f"config.yaml predict.models has unknown classifiers: {_UNKNOWN_MODELS}")
# trained + compared (config.yaml predict.models), always in ALL_CLASSIFIERS order
CLASSIFIERS = [m for m in ALL_CLASSIFIERS if m in PREDICT.models]
REGRESSORS = ["simple_lr", "multiple_lr", "knn"]
CLASSIFICATION_COLUMNS = [
    "customer_id",
    "model",
    "split",  # train | test
    "p_return",
    "pred_return",
    "returned_flag",
]
ROC_COLUMNS = ["model", "fpr", "tpr", "threshold"]
PR_COLUMNS = ["model", "recall", "precision", "threshold"]
KNN_CURVE_COLUMNS = ["k", "train_error", "test_error"]
REGRESSION_COLUMNS = ["customer_id", "model", "split", "y_true", "y_pred"]  # log1p(next_spend)

# ---- fitted sklearn pipelines (joblib) for the P4 What-if sliders; not mocked ----
MODELS_DIR = ARTIFACTS_DIR / "models"
SEGMENT_MODEL_JOBLIB = MODELS_DIR / "segment_model.joblib"  # log1p -> scaler -> kmeans
CLASSIFIER_JOBLIB = MODELS_DIR / "classifier.joblib"  # best classifier: p_return
# every CLASSIFIERS pipeline: {pipelines: {m: ...}, thresholds: {m: ...}, features, best_model}
CLASSIFIERS_JOBLIB = MODELS_DIR / "classifiers.joblib"
REGRESSOR_JOBLIB = MODELS_DIR / "regressor.joblib"  # best regressor: log1p(next_spend | return)
MODEL_JOBLIBS = (SEGMENT_MODEL_JOBLIB, CLASSIFIER_JOBLIB, CLASSIFIERS_JOBLIB, REGRESSOR_JOBLIB)

RULES_DIR = ARTIFACTS_DIR / "rules"
RULES_PARQUET = RULES_DIR / "rules.parquet"
RECOMMENDATIONS_PARQUET = RULES_DIR / "recommendations.parquet"
PRODUCTS_PARQUET = RULES_DIR / "products.parquet"  # top products by invoices (= rule universe)
RULES_COLUMNS = ["scope", "antecedents", "consequents", "support", "confidence", "lift"]
RECOMMENDATIONS_COLUMNS = ["customer_id", "rank", "stock_code", "description", "score"]
PRODUCTS_COLUMNS = ["stock_code", "description", "n_invoices"]  # best sellers first

# ======================= P4: app data (the deployed app never reads data/) =======================
CUSTOMERS_DIR = ARTIFACTS_DIR / "customers"
TIMELINE_PARQUET = CUSTOMERS_DIR / "timeline.parquet"  # customer x month x period
REC_REASONS_PARQUET = CUSTOMERS_DIR / "rec_reasons.parquet"  # rule behind each "rule" rec
TIMELINE_COLUMNS = ["customer_id", "month", "revenue", "orders", "period"]  # before | after
REC_REASONS_COLUMNS = [
    "customer_id",
    "stock_code",
    "scope",
    "antecedents",
    "consequents",
    "confidence",
    "lift",
]

# metrics.json sections written by P3, with the keys the app relies on
P3_METRICS_KEYS: dict[str, list[str]] = {
    "segment": [
        "k",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "stability_ari",
        "pca_explained_variance",
        "baseline",
        "chi2_segment_uk",
        "conclusion",
    ],
    "classify": ["models", "best_model", "paired_t_auc", "conclusion"],
    "regress": ["target", "models", "best_model", "conclusion"],
    "backtest": [
        "model",
        "threshold",
        "n_customers",
        "n_predicted_return",
        "share_actually_returned",
        "base_rate",
        "conclusion",
    ],
    "rules": ["min_support", "min_confidence", "n_rules", "top_rule", "conclusion"],
}

# ---- registry: parquet artifact -> required columns ----
PARQUET_CONTRACTS: dict[Path, list[str]] = {
    FEATURES_CUTOFF_PARQUET: FEATURES_COLUMNS,
    LABELS_PARQUET: LABELS_COLUMNS,
    FEATURES_FULL_PARQUET: FEATURES_COLUMNS,
    QQ_PARQUET: QQ_COLUMNS,
    FREQUENCY_FIT_PARQUET: FREQUENCY_FIT_COLUMNS,
    CORRELATION_PARQUET: CORRELATION_COLUMNS,
    CUBE_PARQUET: CUBE_COLUMNS,
    EDA_HISTOGRAMS_PARQUET: HISTOGRAM_COLUMNS,
    EDA_BOXPLOTS_PARQUET: BOXPLOT_COLUMNS,
    EDA_MONTHLY_PARQUET: MONTHLY_COLUMNS,
    EDA_COUNTRIES_PARQUET: COUNTRY_COLUMNS,
    EDA_PARETO_PARQUET: PARETO_COLUMNS,
    EDA_COHORTS_PARQUET: COHORT_COLUMNS,
    EDA_SAMPLE_ROWS_PARQUET: SAMPLE_ROWS_COLUMNS,
    SEGMENTS_PARQUET: SEGMENTS_COLUMNS,
    SEGMENT_PROFILES_PARQUET: SEGMENT_PROFILE_COLUMNS,
    K_SELECTION_PARQUET: K_SELECTION_COLUMNS,
    PCA_LOADINGS_PARQUET: PCA_LOADINGS_COLUMNS,
    SEGMENT_PRESETS_PARQUET: SEGMENT_PRESETS_COLUMNS,
    CLASSIFICATION_PARQUET: CLASSIFICATION_COLUMNS,
    ROC_PARQUET: ROC_COLUMNS,
    PR_PARQUET: PR_COLUMNS,
    KNN_CURVE_PARQUET: KNN_CURVE_COLUMNS,
    REGRESSION_PARQUET: REGRESSION_COLUMNS,
    RULES_PARQUET: RULES_COLUMNS,
    RECOMMENDATIONS_PARQUET: RECOMMENDATIONS_COLUMNS,
    PRODUCTS_PARQUET: PRODUCTS_COLUMNS,
    TIMELINE_PARQUET: TIMELINE_COLUMNS,
    REC_REASONS_PARQUET: REC_REASONS_COLUMNS,
}
EDA_OUTPUTS = (
    EDA_HISTOGRAMS_PARQUET,
    EDA_BOXPLOTS_PARQUET,
    EDA_MONTHLY_PARQUET,
    EDA_COUNTRIES_PARQUET,
    EDA_PARETO_PARQUET,
    EDA_COHORTS_PARQUET,
    EDA_SAMPLE_ROWS_PARQUET,
)

# ---- mock: same relative paths under artifacts/mock/ (tracked in git, tiny, synthetic) ----
MOCK_DIR = ARTIFACTS_DIR / "mock"
# every P3 parquet, plus the feature/label files they join to, so the mock set is self-consistent
MOCKED_PARQUETS = (
    FEATURES_CUTOFF_PARQUET,
    LABELS_PARQUET,
    SEGMENTS_PARQUET,
    SEGMENT_PROFILES_PARQUET,
    K_SELECTION_PARQUET,
    PCA_LOADINGS_PARQUET,
    SEGMENT_PRESETS_PARQUET,
    CLASSIFICATION_PARQUET,
    ROC_PARQUET,
    PR_PARQUET,
    KNN_CURVE_PARQUET,
    REGRESSION_PARQUET,
    RULES_PARQUET,
    RECOMMENDATIONS_PARQUET,
    PRODUCTS_PARQUET,
    # P4: everything else the app reads, so the whole app runs on mock data (tests, CI)
    TIMELINE_PARQUET,
    REC_REASONS_PARQUET,
    CUBE_PARQUET,
    *EDA_OUTPUTS,
    QQ_PARQUET,
    FREQUENCY_FIT_PARQUET,
    CORRELATION_PARQUET,
)
QUALITY_JSONS = (
    FORMAT_COMPARISON_JSON,
    CLEANING_STEPS_JSON,
    MISSING_ID_JSON,
    WAREHOUSE_SUMMARY_JSON,
)

# ---- what the app reads (P5 deploy: track exactly these) ----
APP_PARQUETS = tuple(p for p in MOCKED_PARQUETS)
APP_JSONS = (*QUALITY_JSONS, METRICS_JSON)
APP_FILES = (*APP_PARQUETS, *APP_JSONS, *MODEL_JOBLIBS)
# metrics.json sections the app reads
APP_METRICS_SECTIONS = ("features", "stats", "eda", *P3_METRICS_KEYS, "app_data")


def mock_path(path: Path, mock_dir: Path = MOCK_DIR) -> Path:
    """artifacts/<x> -> artifacts/mock/<x>."""
    return Path(mock_dir) / Path(path).relative_to(ARTIFACTS_DIR)


MOCK_OUTPUTS = (
    *(mock_path(p) for p in MOCKED_PARQUETS),
    *(mock_path(p) for p in QUALITY_JSONS),
    *(mock_path(p) for p in MODEL_JOBLIBS),
    mock_path(METRICS_JSON),
)


class ContractError(ValueError):
    """Raised when a frame does not satisfy its contract."""


def require_columns(df: pd.DataFrame, columns: list[str], name: str = "frame") -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ContractError(f"{name} is missing required columns: {missing}")
    return df


def validate(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Check a frame against the registered contract of `path` (an artifact path)."""
    return require_columns(df, PARQUET_CONTRACTS[Path(path)], Path(path).name)


def all_exist(paths: list[Path]) -> bool:
    return all(Path(p).exists() for p in paths)


# ---- JSON / metrics helpers ----
def to_jsonable(obj):
    """numpy scalars -> Python, NaN/inf -> None, Timestamps/Paths -> str (recursively)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | np.ndarray):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, pd.Timestamp | Path):
        return str(obj)
    return obj


def write_json(path: Path, obj) -> None:
    """Atomic write (temp file + replace) so readers never see a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(obj), indent=2, allow_nan=False), encoding="utf-8")
    # Windows refuses to replace a file another process (e.g. the app) has open: retry briefly
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.2 * (attempt + 1))


def read_metrics(path: Path = METRICS_JSON) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class MetricsLockTimeout(TimeoutError):
    """Another process held the metrics.json lock for longer than the timeout."""


@contextmanager
def file_lock(path: Path, timeout: float = 60.0, poll: float = 0.05, stale_after: float = 300.0):
    """Cross-process lock: `<path>.lock` created with O_CREAT | O_EXCL, removed in finally.

    A lock file older than `stale_after` seconds (a killed process) is broken.
    """
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale_after:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue  # released between open and stat: retry at once
            if time.monotonic() > deadline:
                raise MetricsLockTimeout(f"could not lock {path} within {timeout:.0f}s") from None
            time.sleep(poll)
        except PermissionError:  # Windows: lock file is being deleted right now
            if time.monotonic() > deadline:
                raise MetricsLockTimeout(f"could not lock {path} within {timeout:.0f}s") from None
            time.sleep(poll)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        for attempt in range(5):
            try:
                lock.unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))


def update_metrics(
    section: str, payload: dict, path: Path = METRICS_JSON, timeout: float = 60.0
) -> None:
    """Replace one section of metrics.json, keeping the sections other stages wrote.

    Read-modify-write runs under a lock file, so parallel stages never lose a section.
    """
    with file_lock(Path(path), timeout=timeout):
        metrics = read_metrics(path)
        metrics[section] = payload
        write_json(path, metrics)


def has_metrics(sections: tuple[str, ...] | list[str], path: Path = METRICS_JSON) -> bool:
    if not sections:
        return True
    metrics = read_metrics(path)
    return all(s in metrics for s in sections)
