"""Project configuration loaded from config.yaml at the repo root."""

from __future__ import annotations

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

DATA_URL: str = _CFG["data"]["url"]
XLSX_NAME: str = _CFG["data"]["xlsx_name"]

# Directories
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
