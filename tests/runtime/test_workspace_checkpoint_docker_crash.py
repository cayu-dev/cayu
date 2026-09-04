from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from tests.runners.test_docker_live import _docker_path_or_skip

pytestmark = pytest.mark.process


@pytest.mark.parametrize(
    "phase", ["intent", "checkpointing", "uploaded", "pinned", "durable", "model"]
)
def test_sigkill_and_deleted_tmpfs_container_at_checkpoint_boundary(tmp_path, phase):
    docker = _docker_path_or_skip()
    worker = Path(__file__).with_name("_workspace_checkpoint_crash_worker.py")
    name = "cayu-checkpoint-" + uuid4().hex[:12]
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")
        + os.pathsep
        + str(Path(__file__).resolve().parents[2]),
    }

    def create():
        subprocess.run(
            [
                docker,
                "run",
                "-d",
                "--name",
                name,
                "--network",
                "none",
                "--tmpfs",
                "/workspace:rw,size=32m",
                "python:3.14-slim",
                "sleep",
                "600",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )

    try:
        create()
        produced = subprocess.run(
            [sys.executable, str(worker), str(tmp_path), name, phase, "produce"],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert produced.returncode == -signal.SIGKILL, produced.stdout + produced.stderr
        subprocess.run([docker, "rm", "-f", name], check=True, capture_output=True, timeout=30)
        create()
        restored = subprocess.run(
            [sys.executable, str(worker), str(tmp_path), name, phase, "restore"],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert restored.returncode == 0, restored.stdout + restored.stderr
    finally:
        subprocess.run([docker, "rm", "-f", name], capture_output=True, timeout=30)
