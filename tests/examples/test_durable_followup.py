from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path


def test_followup_survives_worker_process_loss_before_due(tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(root / "examples/durable_followup.py")]
    database = ["--database", str(tmp_path / "followups.sqlite")]
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}

    def invoke(action, *extra):
        result = subprocess.run(
            [*command, action, *database, *extra],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return [json.loads(line) for line in result.stdout.splitlines()]

    due = datetime.now(UTC) + timedelta(seconds=15)
    assert invoke("schedule", "--due", due.isoformat())[0]["status"] == "pending"
    # A distinct producer replay converges without adding another occurrence.
    assert invoke("schedule", "--due", due.isoformat())[0]["status"] == "pending"
    worker = subprocess.Popen(
        [*command, "work", *database],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        readiness_deadline = time.monotonic() + 8
        while True:
            try:
                worker.communicate(timeout=0.25)
            except subprocess.TimeoutExpired as pending:
                if b'"worker_ready": true' in (pending.output or b""):
                    break
                assert time.monotonic() < readiness_deadline, "Worker did not become ready"
            else:
                raise AssertionError("Future work finished before the worker-loss boundary")
        assert datetime.now(UTC) < due
        worker.kill()
        stdout, stderr = worker.communicate(timeout=10)
        assert worker.returncode != 0
        assert '"worker_ready": true' in stdout
        assert "handled" not in stdout
        assert not stderr
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.communicate(timeout=10)
    before = invoke("inspect")[0]
    assert before["status"] == "pending"
    assert before["events"] == ["task.scheduled"]
    assert invoke("work")[-1] == {"handled": 1}
    after = invoke("inspect")[0]
    assert after["status"] == "completed"
    assert datetime.fromisoformat(after["result"]["observed_at"]) >= due
    assert after["events"].count("task.scheduled") == 1
    assert after["events"].count("task.schedule_claimed") == 1
    assert after["events"][-1] == "task.schedule_completed"
    assert invoke("probe")[0] == {"claimed": None}
