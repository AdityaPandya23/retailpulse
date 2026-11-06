"""Deploy guard: git tracks exactly the app artifacts, never data/ (names + sizes, no reads)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from retailpulse import contracts as C
from retailpulse import deploy
from retailpulse.config import ROOT

MAX_TRACKED_ARTIFACT_MB = 10

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout"
)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def _ignored(rel: str) -> bool:
    """True if .gitignore rules ignore the path (whether or not it exists / is tracked)."""
    return _git("check-ignore", "-q", "--no-index", rel).returncode == 0


def _tracked(prefix: str) -> list[str]:
    out = _git("ls-files", "-z", "--", prefix).stdout
    return [p for p in out.split("\0") if p]


def test_gitignore_block_matches_contracts():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert deploy.render_block() in text, "run: python -m retailpulse.deploy"


def test_every_app_file_is_trackable():
    for rel in deploy.app_paths():
        assert not _ignored(rel), rel


@pytest.mark.parametrize(
    "rel",
    [
        "data/raw/online_retail_II.xlsx",
        "data/raw/online_retail_II.parquet",
        "data/processed/sales.parquet",
        "data/processed/returns.parquet",
        "artifacts/features/features_full.parquet",
        "artifacts/features/features_cutoff.parquet.tmp",
        "artifacts/models/old_model.joblib",
        "artifacts/figures/anything.png",
        "artifacts/mock/segments/segments.parquet.tmp",
    ],
)
def test_data_and_non_app_artifacts_stay_ignored(rel):
    assert _ignored(rel), rel


def test_nothing_under_data_is_tracked():
    assert _tracked("data") == []


def test_tracked_artifacts_are_app_files_or_mock_and_small():
    tracked = _tracked("artifacts")
    allowed = set(deploy.app_paths())
    mock = Path(C.MOCK_DIR).relative_to(ROOT).as_posix() + "/"
    extra = [p for p in tracked if p not in allowed and not p.startswith(mock)]
    assert extra == []
    total = sum((ROOT / p).stat().st_size for p in tracked if (ROOT / p).exists())
    assert total < MAX_TRACKED_ARTIFACT_MB * 1e6, f"{total / 1e6:.1f} MB tracked"


def test_every_app_file_is_tracked():
    """The deployed app must find every input in the clone (no mock fallback on the cloud)."""
    missing = sorted(set(deploy.app_paths()) - set(_tracked("artifacts")))
    assert missing == [], missing
