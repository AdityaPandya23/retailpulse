"""The ONLY place the app reads artifacts. Paths come from retailpulse.contracts.

* Artifact root = env var RETAILPULSE_ARTIFACTS (tests point it at artifacts/mock) or
  contracts.ARTIFACTS_DIR.
* Per file fallback: a file missing under the root is read from artifacts/mock/ instead, and
  mock_files() lists it so every page shows a visible "Mock data" badge.
* metrics.json falls back per SECTION to the mock metrics (a stage that has not run yet).
* st.cache_data / st.cache_resource are keyed on the file's mtime, so a pipeline rerun is
  picked up on the next page load without restarting the app.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from retailpulse import contracts as C
from retailpulse.config import HORIZON_DAYS
from retailpulse.viz import common as VC

ENV_VAR = "RETAILPULSE_ARTIFACTS"


# ------------------------------------------------------------------ paths
def artifact_root() -> Path:
    env = os.environ.get(ENV_VAR, "").strip()
    return Path(env) if env else C.ARTIFACTS_DIR


def _is_under(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def resolve(path: Path) -> tuple[Path, bool]:
    """Contract path -> (file to read, is_mock). Missing everywhere -> (root path, False)."""
    rel = Path(path).relative_to(C.ARTIFACTS_DIR)
    primary = artifact_root() / rel
    if primary.exists():
        return primary, _is_under(primary, C.MOCK_DIR)
    fallback = C.MOCK_DIR / rel
    if fallback.exists():
        return fallback, True
    return primary, False


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else -1.0


# ------------------------------------------------------------------ cached readers
@st.cache_data(show_spinner=False, max_entries=128)
def _read_parquet(path: str, mtime: float, columns: tuple[str, ...] | None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns) if columns else None)


@st.cache_data(show_spinner=False, max_entries=32)
def _read_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False, max_entries=16)
def _read_joblib(path: str, mtime: float):
    return joblib.load(path)


# ------------------------------------------------------------------ public API
def table(path: Path, columns: list[str] | None = None) -> pd.DataFrame | None:
    """A parquet artifact (validated against its contract) or None if it does not exist."""
    file, _ = resolve(path)
    if not file.exists():
        return None
    df = _read_parquet(str(file), _mtime(file), tuple(columns) if columns else None)
    if columns:
        return C.require_columns(df, list(columns), Path(path).name)
    if Path(path) in C.PARQUET_CONTRACTS:
        C.validate(df, path)
    return df


def json_file(path: Path) -> dict:
    """A JSON artifact (quality/*.json) or {} if it does not exist."""
    file, _ = resolve(path)
    return _read_json(str(file), _mtime(file)) if file.exists() else {}


def _metrics_files() -> tuple[dict, dict]:
    """(real metrics, mock metrics); when the root itself is the mock folder, real = {}."""
    file, is_mock = resolve(C.METRICS_JSON)
    mfile = file if is_mock else C.mock_path(C.METRICS_JSON)
    real = _read_json(str(file), _mtime(file)) if file.exists() and not is_mock else {}
    mock = _read_json(str(mfile), _mtime(mfile)) if mfile.exists() else {}
    return real, mock


def metrics(section: str | None = None) -> dict:
    """metrics.json (one section or all); a section missing in the real file comes from mock."""
    real, mock = _metrics_files()
    if section is None:
        return {**mock, **real}
    if section in real:
        return real[section]
    return mock.get(section, {})


def model(path: Path) -> dict | None:
    """A joblib model payload (shared across sessions) or None if it does not exist."""
    file, _ = resolve(path)
    return _read_joblib(str(file), _mtime(file)) if file.exists() else None


def mock_files() -> list[str]:
    """Every app input currently served from artifacts/mock/ (empty list = all real data)."""
    out = []
    for path in C.APP_FILES:
        if path == C.METRICS_JSON:
            continue
        file, is_mock = resolve(path)
        if file.exists() and is_mock:
            out.append(str(Path(path).relative_to(C.ARTIFACTS_DIR)).replace("\\", "/"))
    real, mock = _metrics_files()
    out += [f"metrics.json [{s}]" for s in C.APP_METRICS_SECTIONS if s not in real and s in mock]
    return out


def missing_files() -> list[str]:
    """App inputs found neither under the root nor in artifacts/mock/."""
    return [
        str(Path(p).relative_to(C.ARTIFACTS_DIR)).replace("\\", "/")
        for p in C.APP_FILES
        if not resolve(p)[0].exists()
    ]


@st.cache_data(show_spinner=False, max_entries=4)
def _read_text(path: str, mtime: float) -> str:
    return Path(path).read_text(encoding="utf-8")


def limitations() -> list[str]:
    """The bullets under "## Limitations" in README.md (same wording as the README)."""
    from retailpulse.insights import limitations as parse

    file = Path(C.README_MD)
    return parse(_read_text(str(file), _mtime(file))) if file.exists() else []


def horizon_days() -> int:
    """Label horizon in days: metrics.json features.horizon_days, else config.yaml."""
    return int(metrics("features").get("horizon_days") or HORIZON_DAYS)


# ------------------------------------------------------------------ segment helpers
def segment_names() -> dict[int, str]:
    """{segment id: name} from metrics.json (k and names are never hard-coded)."""
    return VC.segment_names(metrics("segment"), table(C.SEGMENT_PROFILES_PARQUET))


def segment_colours() -> dict[str, str]:
    """{segment name: fixed colour} for plotly color_discrete_map."""
    return VC.segment_colour_map(segment_names())
