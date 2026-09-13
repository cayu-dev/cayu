from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from examples.durable_service_tools.app import HandbookReader

if TYPE_CHECKING:
    from tests.sqlite_resources import SQLiteResourceScope

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "durable_service_tools" / "app.py"


def run_phase(state: Path, phase: str, *options: str, succeeds: bool = True) -> str:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]))
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), phase, str(state), *options],
        cwd=state.parent,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    output = result.stdout + result.stderr
    assert (result.returncode == 0) == succeeds, output
    return output


def observations(state: Path, phase: str) -> dict:
    return json.loads((state / f"{phase}-observations.json").read_text())


def receipt_count(state: Path, resources: SQLiteResourceScope) -> int:
    database = state / "effects.sqlite"
    if not database.exists():
        return 0
    connection = resources.own(sqlite3.connect(database), kind="connection")
    cursor = resources.own(connection.execute("SELECT count(*) FROM receipts"), kind="cursor")
    return cursor.fetchone()[0]


@pytest.mark.parametrize("wiring", ["injected", "environment"])
def test_same_build_resume_uses_supplied_scoped_store(tmp_path: Path, wiring: str) -> None:
    state = tmp_path / "state"
    run_phase(state, "start", "--wiring", wiring)
    run_phase(state, "resume", "--wiring", wiring)
    assert observations(state, "resume") == {"searches": 1, "provider_requests": 2}


@pytest.mark.parametrize("option", ["--version", "--policy-version", "--environment-version"])
def test_changed_identity_rejected_before_provider_or_effect(tmp_path: Path, option: str) -> None:
    state = tmp_path / "state"
    run_phase(state, "pause", "--wiring", "environment")
    output = run_phase(state, "approve", "--wiring", "environment", option, "2", succeeds=False)
    assert "ExecutionProfileMismatchError" in output
    assert observations(state, "approve") == {"searches": 0, "provider_requests": 0}
    assert not (state / "effects.sqlite").exists()


def test_undeclared_service_identity_remains_fail_closed(tmp_path: Path) -> None:
    state = tmp_path / "state"
    run_phase(state, "start", "--opaque")
    output = run_phase(state, "resume", "--opaque", succeeds=False)
    assert "ExecutionProfileMismatchError" in output
    assert observations(state, "resume") == {"searches": 0, "provider_requests": 0}


def test_required_store_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="supplied, bound knowledge store"):
        HandbookReader(None)


@pytest.mark.parametrize("decision", ["approve", "deny"])
def test_fresh_process_approval_and_completed_resolution(
    tmp_path: Path,
    decision: str,
    sqlite_resources: SQLiteResourceScope,
) -> None:
    async def scenario() -> None:
        async with sqlite_resources as resources:
            state = tmp_path / "state"
            run_phase(state, "pause", "--wiring", "environment")
            assert receipt_count(state, resources) == 0
            run_phase(state, decision, "--wiring", "environment")
            expected = 1 if decision == "approve" else 0
            assert receipt_count(state, resources) == expected
            run_phase(state, "repeat", "--wiring", "environment")
            assert receipt_count(state, resources) == expected
            assert observations(state, "repeat") == {"searches": 0, "provider_requests": 0}

    asyncio.run(scenario())
