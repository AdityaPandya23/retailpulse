"""Segmentation stage (I/O only). Run: python -m retailpulse.segment.stage

Inputs:  features_cutoff.parquet, labels.parquet, config.yaml (segment section, seed)
Outputs: segments / profiles / k_selection / pca_loadings parquet, segment_model.joblib,
         metrics.json section "segment"
"""

from __future__ import annotations

import joblib
import pandas as pd

from retailpulse import contracts as C
from retailpulse.segment.cluster import segment_customers

OUTPUTS = (
    C.SEGMENTS_PARQUET,
    C.SEGMENT_PROFILES_PARQUET,
    C.K_SELECTION_PARQUET,
    C.PCA_LOADINGS_PARQUET,
    C.SEGMENT_MODEL_JOBLIB,
)
INPUTS = (C.FEATURES_CUTOFF_PARQUET, C.LABELS_PARQUET)
METRICS_SECTIONS = ("segment",)


def check_metrics(metrics: dict) -> dict:
    missing = [k for k in C.P3_METRICS_KEYS["segment"] if k not in metrics]
    if missing:
        raise C.ContractError(f"metrics['segment'] is missing keys: {missing}")
    return metrics


def run() -> dict:
    features = C.validate(pd.read_parquet(C.FEATURES_CUTOFF_PARQUET), C.FEATURES_CUTOFF_PARQUET)
    labels = C.validate(pd.read_parquet(C.LABELS_PARQUET), C.LABELS_PARQUET)
    res = segment_customers(features, labels)
    for path, df in res.tables.items():
        C.validate(df, path)
    check_metrics(res.metrics)
    for path, df in res.tables.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    C.SEGMENT_MODEL_JOBLIB.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(res.model, C.SEGMENT_MODEL_JOBLIB)
    C.update_metrics("segment", res.metrics)
    m = res.metrics
    return {
        "k": m["k"],
        "silhouette": round(m["silhouette"], 3),
        "ari_mean": round(m["stability_ari"]["mean"], 3),
        "sizes": m["segment_sizes"],
    }


if __name__ == "__main__":
    summary = run()
    print(summary)
    print(C.read_metrics()["segment"]["conclusion"])
