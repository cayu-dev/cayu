"""Public control crosses processes; continuation starts in a fresh runtime."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from tests.core.test_session_store_shared_conformance import (
    _reset_postgres_data,
)
from tests.core.test_session_store_shared_conformance import (
    conformance_postgres_dsn as conformance_postgres_dsn,
)

from cayu import EventType, PostgresSessionStore, SessionStatus, SQLiteSessionStore
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.runtime._tool_round_recovery import pending_tool_round_from_checkpoint


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("process_loss", ["none", "acceptance", "promotion", "unknown_tool"])
def test_process_control_settles_current_round_then_fresh_resume(
    tmp_path, backend, request, process_loss
):
    dsn = None if backend == "sqlite" else request.getfixturevalue("conformance_postgres_dsn")
    if dsn is not None:
        asyncio.run(_reset_postgres_data(dsn))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(Path.cwd() / "src"), str(Path.cwd())))}
    if dsn is not None:
        env["CAYU_STEERING_TEST_DSN"] = dsn

    def command(mode):
        return [
            sys.executable,
            "-m",
            "tests.core._session_steering_process_worker",
            mode,
            backend,
            str(tmp_path),
        ]

    def wait_marker(process, name):
        deadline = time.monotonic() + 40
        while not (tmp_path / name).exists():
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=5)
                pytest.fail(f"Worker exited before {name}: {stdout!r} {stderr!r}")
            assert time.monotonic() < deadline, f"Worker did not reach {name}."
            time.sleep(0.02)

    owner = subprocess.Popen(
        command("run-crash" if process_loss == "promotion" else "run"),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        wait_marker(owner, "tool-started")
        if process_loss == "acceptance":
            accepting = subprocess.Popen(
                command("accept-crash"), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            try:
                wait_marker(accepting, "acceptance-committed")
            finally:
                if accepting.poll() is None:
                    accepting.kill()
                accepting.communicate(timeout=10)
            assert accepting.returncode != 0
        else:
            accepted = subprocess.run(command("accept"), env=env, capture_output=True, timeout=30)
            assert accepted.returncode == 0, accepted.stderr.decode()
        assert owner.poll() is None
        assert (tmp_path / "provider-calls").read_text().splitlines() == ["dispatch"]
        assert not (tmp_path / "tool-effects").exists()
        if process_loss == "unknown_tool":
            # The durable started event proves dispatch, not its outcome. Kill
            # the actual owner before a terminal tool result exists.
            owner.kill()
            owner.communicate(timeout=10)
            assert owner.returncode != 0
            recovered = subprocess.run(
                command("recover-interruption"), env=env, capture_output=True, timeout=40
            )
            assert recovered.returncode == 0, recovered.stderr.decode()

            async def inspect_unknown():
                store = (
                    SQLiteSessionStore(tmp_path / "sessions.db")
                    if backend == "sqlite"
                    else PostgresSessionStore(dsn, min_size=1, max_size=2)
                )
                try:
                    events = await store.load_events("process-safe-steering")
                    assert sum(e.type is EventType.TOOL_CALL_STARTED for e in events) == 1
                    unknown = [e for e in events if e.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN]
                    assert len(unknown) == 1
                    session = await store.load("process-safe-steering")
                    assert session is not None and session.status is SessionStatus.INTERRUPTED
                    checkpoint_store = runtime_checkpoint_session_store(store)
                    pending = pending_tool_round_from_checkpoint(
                        await checkpoint_store.load_checkpoint(session.id)
                    )
                    assert pending is not None and not pending.staged_terminals
                    record = await ToolEffectStateOwner(checkpoint_store).resolve_call(
                        session,
                        tool_round_id=pending.tool_round_id,
                        tool_call_id="current-tool",
                    )
                    assert record is not None and record.state == "outcome_unknown"
                    assert record.terminal is None
                    assert unknown[0].payload["dispatch_id"] == record.dispatch_id
                    assert not any(
                        e.type
                        in {
                            EventType.TOOL_CALL_COMPLETED,
                            EventType.TOOL_CALL_FAILED,
                            EventType.SESSION_MESSAGE_DELIVERED,
                        }
                        for e in events
                    )
                    return record
                finally:
                    await store.close()

            retained = asyncio.run(inspect_unknown())
            resumed = subprocess.run(command("resume"), env=env, capture_output=True, timeout=40)
            assert resumed.returncode == 0, resumed.stderr.decode()
            assert asyncio.run(inspect_unknown()) == retained
            assert (tmp_path / "provider-calls").read_text().splitlines() == ["dispatch"]
            assert not (tmp_path / "tool-effects").exists()
            return
        (tmp_path / "release-tool").touch()
        if process_loss == "promotion":
            wait_marker(owner, "promotion-committed")
            owner.kill()
            owner.communicate(timeout=10)
            assert owner.returncode != 0
            recovered = subprocess.run(
                command("recover-interruption"), env=env, capture_output=True, timeout=40
            )
            assert recovered.returncode == 0, recovered.stderr.decode()
        stdout, stderr = owner.communicate(timeout=40)
        if process_loss != "promotion":
            assert owner.returncode == 0, stderr.decode()
            assert b'"interrupted"' in stdout
        assert (tmp_path / "tool-effects").read_text().splitlines() == ["effect"]
        assert (tmp_path / "provider-calls").read_text().splitlines() == ["dispatch"]

        async def inspect(expected_status):
            store = (
                SQLiteSessionStore(tmp_path / "sessions.db")
                if backend == "sqlite"
                else PostgresSessionStore(dsn, min_size=1, max_size=2)
            )
            try:
                session = await store.load("process-safe-steering")
                assert session is not None and session.status is expected_status
                return await store.load_events(session.id)
            finally:
                await store.close()

        events = asyncio.run(inspect(SessionStatus.INTERRUPTED))
        terminals = [event for event in events if event.type is EventType.SESSION_INTERRUPTED]
        assert len(terminals) == 1
        settled = next(event for event in events if event.type is EventType.TOOL_CALL_COMPLETED)
        assert events.index(settled) < events.index(terminals[0])
        assert not any(event.type is EventType.SESSION_MESSAGE_DELIVERED for event in events)
        resumed = subprocess.run(command("resume"), env=env, capture_output=True, timeout=40)
        assert resumed.returncode == 0, resumed.stderr.decode()
        events = asyncio.run(inspect(SessionStatus.COMPLETED))
        assert sum(event.type is EventType.SESSION_MESSAGE_DELIVERED for event in events) == 1
        assert (tmp_path / "provider-calls").read_text().splitlines() == ["dispatch", "dispatch"]
        assert (tmp_path / "tool-effects").read_text().splitlines() == ["effect"]
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.communicate(timeout=10)
