from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def phase(state: Path, name: str, *options: str, exit_code: int = 0):
    result = subprocess.run(
        [sys.executable, "-m", "examples.human_attention.app", name, str(state), *options],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])},
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    return json.loads(result.stdout) if result.stdout.strip() else None


@pytest.mark.parametrize(
    ("kind", "resolution", "settlement"),
    [
        ("user_input", "answer", "resolved"),
        ("tool_approval", "deny", "cancelled"),
    ],
)
def test_attention_pause_restart_durable_acceptance_resolution(
    tmp_path, kind, resolution, settlement
):
    state = tmp_path / kind
    assert (
        phase(state, "pause", "--kind", kind, "--fail-after-accept")["last_event"]
        == "session.interrupted"
    )
    # A failed sink acknowledgement leaves the exact pause available. The
    # destination has durably accepted its hint before producer process exit.
    connection = sqlite3.connect(state / "runtime.sqlite")
    try:
        assert (
            connection.execute(
                "SELECT count(*) FROM cayu_persisted_event_side_effects WHERE status = 'failed'"
            ).fetchone()[0]
            >= 1
        )
        assert (
            connection.execute(
                "SELECT status FROM cayu_sessions WHERE id = 'attention-example'"
            ).fetchone()[0]
            == "interrupted"
        )
    finally:
        connection.close()
    phase(state, "consume", "--crash-after-accept", exit_code=18)
    accepted = phase(state, "inspect")["notifications"]
    assert len(accepted) == 1 and accepted[0]["state"] == "active"
    # Concurrent fresh consumers repeat an ambiguous acceptance safely.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
    processes = [
        subprocess.Popen(
            [sys.executable, "-m", "examples.human_attention.app", "consume", str(state)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    for process in processes:
        stdout, stderr = process.communicate(timeout=120)
        assert process.returncode == 0, stdout + stderr
    assert phase(state, resolution)["last_event"] == "session.completed"
    settled = phase(state, "consume")["notifications"]
    assert len(settled) == 1 and settled[0]["state"] == settlement
    late = phase(state, "late-hint")["notifications"]
    assert late == settled
    assert "Which environment?" not in json.dumps(late)


def test_attention_crash_after_pause_commit_before_sink_delivery(tmp_path):
    state = tmp_path / "crashed-producer"
    phase(state, "pause", "--crash-before-delivery", exit_code=17)
    # Consumer startup repairs from authoritative pending actions, even while
    # the producer's last delivery claim has not expired yet.
    result = phase(state, "consume")
    assert result["complete"]
    assert len(result["notifications"]) == 1
    assert result["notifications"][0]["state"] == "active"
    assert phase(state, "answer")["last_event"] == "session.completed"
    assert phase(state, "consume")["notifications"][0]["state"] == "resolved"
