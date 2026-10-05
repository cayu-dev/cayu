"""Generated application process-loss/HTTP receipt replay; no Docker claim."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.qualification.repository_maintenance_application import maintenance_project_files
from tests.qualification.repository_maintenance_case import materialize_seed_repository


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_generated_worker_loss_http_settlement_across_fresh_processes(tmp_path, request, backend):
    project = tmp_path / "maintenance-app"
    for relative, content in maintenance_project_files(database="sqlite").items():
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    (project / "uv.lock").write_text("version = 1\n")
    materialize_seed_repository(tmp_path / "source")
    repo = Path(__file__).resolve().parents[2]
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CAYU_", "OPENAI_", "ANTHROPIC_"))
    }
    environment.update(
        PYTHONPATH=os.pathsep.join((str(repo / "src"), str(repo))), CAYU_MODEL="maintenance-fixture"
    )
    if backend == "postgres":
        environment["CAYU_DATABASE_URL"] = request.getfixturevalue("postgres_url")
    script = Path(__file__).with_name("maintenance_cancellation_process.py")
    with (tmp_path / "worker.log").open("w+") as log:
        worker = subprocess.Popen(
            [sys.executable, str(script), str(tmp_path), "start", "0"],
            cwd=repo,
            env=environment,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 60
            while not (tmp_path / "ready").exists():
                if worker.poll() is not None or time.monotonic() >= deadline:
                    log.seek(0)
                    pytest.fail("Worker did not reach release barrier:\n" + log.read())
                time.sleep(0.05)
            worker.kill()
            assert worker.wait(timeout=10) < 0
            for action in ("ack-loss", "replay"):
                result = subprocess.run(
                    [sys.executable, str(script), str(tmp_path), action, str(worker.pid)],
                    cwd=repo,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                assert result.returncode == 0, result.stdout + result.stderr
            assert (tmp_path / "settled.json").exists()
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=10)
