"""Grants survive real process restarts and settle from durable evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _phase(state: Path, name: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "tests.runtime.authorization_grant_process", name, str(state)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])},
        text=True,
        capture_output=True,
        # Each phase starts a fresh interpreter; leave room for loaded CI runners.
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_grants_are_reused_and_settled_after_process_restarts(tmp_path) -> None:
    state = tmp_path / "state"

    first = _phase(state, "approve-and-remember")
    assert first["status"] == "active"

    interrupted = _phase(state, "approve-without-settling")
    assert interrupted["status"] == "staged"

    later = _phase(state, "settle-and-reuse")
    assert later["settled"] == {interrupted["grant_id"]: "active"}
    assert later["paused"] is False
    assert later["allowed_uses"] == [first["grant_id"]]
    assert later["active"] == sorted([first["grant_id"], interrupted["grant_id"]])
