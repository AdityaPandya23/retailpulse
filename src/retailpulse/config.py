"""Project configuration loaded from config.yaml at the repo root."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


_CFG = load_config()

SEED: int = int(_CFG["seed"])
CUTOFF: pd.Timestamp = pd.Timestamp(_CFG["cutoff"])
HORIZON_DAYS: int = int(_CFG["horizon_days"])
HORIZON_END: pd.Timestamp = CUTOFF + timedelta(days=HORIZON_DAYS)

ALPHA: float = float(_CFG["stats"]["alpha"])
YEAR1: tuple[pd.Timestamp, pd.Timestamp] = tuple(pd.Timestamp(d) for d in _CFG["stats"]["year1"])
YEAR2: tuple[pd.Timestamp, pd.Timestamp] = tuple(pd.Timestamp(d) for d in _CFG["stats"]["year2"])


@dataclass(frozen=True)
class SegmentConfig:
    features: tuple[str, ...]  # K-Means inputs
    k_range: tuple[int, int]  # inclusive scan for the k-selection table
    k_choice_range: tuple[int, int]  # inclusive range for the automatic choice + stability
    k: int | None  # None = automatic rule
    names: dict[int, str]  # {} = automatic naming rule


@dataclass(frozen=True)
class PredictConfig:
    train_offset_days: int  # train snapshot cutoff = CUTOFF - train_offset_days
    cv_folds: int
    knn_k_max: int
    qda_reg_param: float


@dataclass(frozen=True)
class RulesConfig:
    top_n_products: int
    min_support: float
    min_abs_invoices: int  # support floor = max(min_support, min_abs_invoices / n_invoices)
    min_confidence: float
    max_len: int
    max_rules_per_scope: int
    n_recommend: int


def _segment_config(d: dict) -> SegmentConfig:
    return SegmentConfig(
        features=tuple(d["features"]),
        k_range=(int(d["k_range"][0]), int(d["k_range"][1])),
        k_choice_range=(int(d["k_choice_range"][0]), int(d["k_choice_range"][1])),
        k=None if d.get("k") is None else int(d["k"]),
        names={int(k): str(v) for k, v in (d.get("names") or {}).items()},
    )


SEGMENT: SegmentConfig = _segment_config(_CFG["segment"])
PREDICT: PredictConfig = PredictConfig(
    train_offset_days=int(_CFG["predict"]["train_offset_days"]),
    cv_folds=int(_CFG["predict"]["cv_folds"]),
    knn_k_max=int(_CFG["predict"]["knn_k_max"]),
    qda_reg_param=float(_CFG["predict"]["qda_reg_param"]),
)
_R = _CFG["rules"]
RULES: RulesConfig = RulesConfig(
    top_n_products=int(_R["top_n_products"]),
    min_support=float(_R["min_support"]),
    min_abs_invoices=int(_R["min_abs_invoices"]),
    min_confidence=float(_R["min_confidence"]),
    max_len=int(_R["max_len"]),
    max_rules_per_scope=int(_R["max_rules_per_scope"]),
    n_recommend=int(_R["n_recommend"]),
)

DATA_URL: str = _CFG["data"]["url"]
XLSX_NAME: str = _CFG["data"]["xlsx_name"]

# Directories
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
