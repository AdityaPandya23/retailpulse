"""metrics.json lock: parallel P3 stages must never lose each other's sections."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from retailpulse import contracts as C

N_UPDATES = 15
WRITER = """
import sys
from retailpulse import contracts as C
path, name, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
for i in range(n):
    C.update_metrics(f"{name}_{i}", {"i": i, "conclusion": name}, path=path)
"""


def test_update_metrics_two_processes_keep_every_section(tmp_path):
    path = tmp_path / "metrics.json"
    C.update_metrics("existing", {"x": 1}, path=path)
    procs = [
        subprocess.Popen([sys.executable, "-c", WRITER, str(path), name, str(N_UPDATES)])
        for name in ("a", "b")
    ]
    assert [p.wait(timeout=120) for p in procs] == [0, 0]
    metrics = C.read_metrics(path)
    expected = {"existing"} | {f"{n}_{i}" for n in ("a", "b") for i in range(N_UPDATES)}
    assert set(metrics) == expected
    assert not (tmp_path / "metrics.json.lock").exists()


def test_lock_times_out_when_held_and_is_released_after_errors(tmp_path):
    path = tmp_path / "metrics.json"
    lock = tmp_path / "metrics.json.lock"
    with C.file_lock(path):
        assert lock.exists()
        with pytest.raises(C.MetricsLockTimeout):
            C.update_metrics("s", {}, path=path, timeout=0.3)
    assert not lock.exists()

    with pytest.raises(RuntimeError), C.file_lock(path):
        raise RuntimeError("boom")
    assert not lock.exists()  # released in finally
    C.update_metrics("s", {"ok": True}, path=path, timeout=1)
    assert C.read_metrics(path) == {"s": {"ok": True}}


def test_stale_lock_is_broken(tmp_path):
    path = tmp_path / "metrics.json"
    lock = tmp_path / "metrics.json.lock"
    lock.write_text("12345")
    old = time.time() - 1000
    os.utime(lock, (old, old))
    C.update_metrics("s", {"ok": True}, path=path, timeout=1)
    assert C.read_metrics(path)["s"] == {"ok": True}


def test_model_paths_live_under_artifacts_models():
    assert {p.parent for p in C.MODEL_JOBLIBS} == {C.MODELS_DIR}
    assert C.MODELS_DIR.parent == C.ARTIFACTS_DIR
