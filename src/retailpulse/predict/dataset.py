"""Out-of-time datasets and the shared preprocessing step.

TRAIN = features at cutoff - train_offset_days, labels from the next horizon_days, built ONLY
        from rows strictly before the real cutoff (so nothing at or after it can leak in).
TEST  = features_cutoff + labels (the real P2 artifacts): features before the cutoff, outcomes
        in [cutoff, cutoff + horizon).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from retailpulse import contracts as C
from retailpulse.config import PREDICT
from retailpulse.features import build_features, build_labels, log1p_skewed

FEATURES: list[str] = list(PREDICT.features)  # config.yaml predict.features (raw values in)
if not set(FEATURES) <= set(C.FEATURE_COLUMNS):
    raise ValueError(f"predict.features must be feature columns: {FEATURES}")
TARGET = "returned_flag"


def log1p_frame(X):
    """log1p on the skewed feature columns present in X (module level so pipelines pickle).

    Accepts a DataFrame with any subset of the feature columns (raw, un-logged values), or an
    array whose columns are FEATURES in order.
    """
    if not isinstance(X, pd.DataFrame):
        X = np.asarray(X, dtype="float64")
        if X.ndim != 2 or X.shape[1] != len(FEATURES):
            raise ValueError("array input must have the columns of FEATURES, in order")
        X = pd.DataFrame(X, columns=FEATURES)
    cols = [c for c in C.SKEWED_FEATURES if c in X.columns]
    return log1p_skewed(X.astype("float64"), cols)


def preprocessor() -> Pipeline:
    """raw features -> log1p(skewed) -> StandardScaler."""
    return Pipeline(
        [
            ("log1p", FunctionTransformer(log1p_frame, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )


def train_cutoff(cutoff, offset_days: int) -> pd.Timestamp:
    return pd.Timestamp(cutoff) - pd.Timedelta(days=offset_days)


def join_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Left-join labels onto features; customers without a label row get 0 / 0.0."""
    C.require_columns(features, C.FEATURES_COLUMNS, "features")
    C.require_columns(labels, C.LABELS_COLUMNS, "labels")
    if labels["customer_id"].duplicated().any():
        raise ValueError("labels have duplicate customer_id")
    out = features.merge(labels[C.LABELS_COLUMNS], on="customer_id", how="left", validate="1:1")
    out["returned_flag"] = out["returned_flag"].fillna(0).astype("int64")
    out["next_spend"] = out["next_spend"].fillna(0.0).astype("float64")
    if len(out) != len(features):
        raise AssertionError("join changed the number of customers")
    if out[[*C.FEATURE_COLUMNS, "returned_flag", "next_spend"]].isna().any().any():
        raise AssertionError("NaN after joining labels")
    return out.reset_index(drop=True)


def build_train(
    sales: pd.DataFrame, returns: pd.DataFrame, cutoff, offset_days: int, horizon_days: int
) -> pd.DataFrame:
    """Train snapshot at cutoff - offset_days; uses ONLY rows with invoice_date < cutoff."""
    if offset_days < horizon_days:
        raise ValueError("train_offset_days < horizon_days: train labels would be truncated")
    cutoff = pd.Timestamp(cutoff)
    s = sales[sales["invoice_date"] < cutoff]
    r = returns[returns["invoice_date"] < cutoff]
    t_cut = train_cutoff(cutoff, offset_days)
    return join_labels(build_features(s, r, t_cut), build_labels(s, t_cut, horizon_days))


def build_test(features_cutoff: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    return join_labels(features_cutoff, labels)
