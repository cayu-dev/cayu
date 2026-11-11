"""Exercise the real CLI supervisor and child worker entrypoints."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT = """\
[tool.cayu]
factory = "recovery_project:build_app"
[tool.cayu.workers]
recover = "recovery_project:run_worker"
"""
_MODULE = """\
import os
from pathlib import Path
from cayu import CayuApp
from cayu.tasks import complete_managed_task, run_task_worker, run_task_worker_until_idle
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore

def build_app():
    return CayuApp(enable_logging=False)

async def run_worker(app, stop):
    store = InMemoryTaskStore()
    await store.create_task(TaskCreate(type="job", restartable=True))
    async def handle(app, task, worker_id):
        Path(f"handled-{os.getpid()}.txt").write_text("handled")
        await complete_managed_task(store, task, worker_id, {"ok": True})
    run = run_task_worker_until_idle if os.environ.get("TEST_UNTIL_IDLE") else run_task_worker
    await run(app, store, handle, worker_id=f"worker-{os.getpid()}",
              max_tasks=1, reconcile_abandoned_after_s=0,
              restart_abandoned=os.environ.get("TEST_RESTART") == "1", stop=stop)
"""


@pytest.mark.skipif(os.name != "posix", reason="multi-process supervision requires POSIX")
@pytest.mark.parametrize("until_idle", [False, True])
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("processes", [1, 2])
def test_cli_code_declared_recovery_requires_one_real_worker_process(
    tmp_path: Path,
    until_idle: bool,
    restart: bool,
    processes: int,
) -> None:
    (tmp_path / "pyproject.toml").write_text(_PROJECT)
    (tmp_path / "recovery_project.py").write_text(_MODULE)
    env = {key: value for key, value in os.environ.items() if not key.startswith("CAYU_WORKER_")}
    env.pop("CAYU_SUPERVISOR_PID", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    env["TEST_UNTIL_IDLE"] = "1" if until_idle else ""
    env["TEST_RESTART"] = "1" if restart else "0"
    result = subprocess.run(
        [sys.executable, "-m", "cayu", "worker", "recover", "--processes", str(processes)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if processes == 1:
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(list(tmp_path.glob("handled-*.txt"))) == 1
    else:
        assert result.returncode != 0, result.stdout + result.stderr
        assert "single worker process" in result.stderr
        assert "--processes 1" in result.stderr
        assert not list(tmp_path.glob("handled-*.txt"))


@pytest.mark.parametrize(
    "metadata,allowed",
    [
        ({"CAYU_WORKER_COUNT": "2", "CAYU_WORKER_INDEX": "0"}, False),
        ({"CAYU_WORKER_INDEX": "0"}, False),
        ({"CAYU_WORKER_COUNT": "true"}, False),
        ({"CAYU_WORKER_COUNT": "1", "CAYU_WORKER_INDEX": "1"}, False),
        ({"CAYU_WORKER_COUNT": "1", "CAYU_WORKER_INDEX": "0"}, True),
    ],
    ids=["group-of-two", "missing-count", "boolean-count", "conflicting-index", "single"],
)
def test_cli_single_process_flag_preserves_inherited_group_metadata(
    tmp_path: Path, metadata: dict[str, str], allowed: bool
) -> None:
    # Forward through the real store methods while keeping their class-owned
    # quiescence declarations. An instance monkeypatch would invalidate those.
    observe_store = """\
    for name in ("claim_task", "reclaim_expired", "list_tasks"):
        original = getattr(InMemoryTaskStore, name)
        async def observed(*args, _original=original, **kwargs):
            Path("store-access.txt").write_text("accessed")
            return await _original(*args, **kwargs)
        setattr(InMemoryTaskStore, name, observed)
"""
    module = _MODULE.replace("    async def handle", observe_store + "    async def handle")
    (tmp_path / "pyproject.toml").write_text(_PROJECT)
    (tmp_path / "recovery_project.py").write_text(module)
    env = {key: value for key, value in os.environ.items() if not key.startswith("CAYU_WORKER_")}
    env.pop("CAYU_SUPERVISOR_PID", None)
    env.update(metadata)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    env["TEST_UNTIL_IDLE"] = ""
    env["TEST_RESTART"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "cayu", "worker", "recover", "--processes", "1"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if allowed:
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(list(tmp_path.glob("handled-*.txt"))) == 1
        assert (tmp_path / "store-access.txt").exists()
    else:
        assert result.returncode != 0, result.stdout + result.stderr
        assert "single worker process" in result.stderr
        assert "--processes 1" in result.stderr
        assert "CAYU_WORKER_COUNT=1" in result.stderr
        assert not list(tmp_path.glob("handled-*.txt"))
        assert not (tmp_path / "store-access.txt").exists()
