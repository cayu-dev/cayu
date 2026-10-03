"""Non-success binding cleanup survives process loss without changing the outcome."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.faults.test_workspace_sync_failure import FailOnceWriteWorkspace

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    EventType,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    RecoveryDecision,
    RecoveryExecutionRequest,
    RecoveryPlanAction,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    SyncBinding,
    Tool,
    ToolResult,
    ToolSpec,
)


async def _child(
    root: Path, mode: str, kind: str, outcome: str, partial: bool, control: str = "normal"
) -> None:
    source, target = root / "source", root / "target"
    fresh = mode in {"seed", "ordinary"}
    if fresh:
        source.mkdir()
        target.mkdir()
        (source / "result.txt").write_bytes(b"original")
        (source / "a-first.txt").write_bytes(b"first original")

    class FailingBinding(SyncBinding):
        async def finalize(self, bound, *, outcome=None, metadata=None):
            raise OSError("injected terminal copy-back failure")

    class Mutate(Tool):
        spec = ToolSpec(name="mutate", input_schema={"type": "object", "properties": {}})

        async def run(self, ctx, args):
            assert ctx.workspace is not None
            await ctx.workspace.write_bytes("a-first.txt", b"first retained")
            await ctx.workspace.write_bytes("result.txt", b"retained output")
            return ToolResult(content="written")

    batches = [
        [
            ModelStreamEvent.tool_call(id="write", name="mutate", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ],
        [
            ModelStreamEvent.error("deliberate provider failure"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
    ]
    provider = ScriptedModelProvider(batches if fresh else [], name="fixture")
    clock_offset = timedelta()

    def clock():
        return datetime.now(UTC) + clock_offset

    store = SQLiteSessionStore(root / "sessions.sqlite", ownership_clock=clock)
    app = CayuApp(session_store=store, enable_logging=False, clock=clock)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="agent", model="fixture"), tools=[Mutate()])
    environment = Environment(
        EnvironmentSpec(name="coding"),
        workspace=(
            FailOnceWriteWorkspace(source, workspace_id="source", fail_path="result.txt")
            if mode == "seed" and partial
            else LocalWorkspace(source, workspace_id="source")
        ),
        binding=(FailingBinding if mode == "seed" and not partial else SyncBinding)(
            target_workspace=LocalWorkspace(target, workspace_id="target"),
            sync_back="always",
            source_conflict_policy="require_revision",
            max_file_bytes=1024,
            max_total_bytes=4096,
        ),
    )
    reconnects = []
    if kind == "factory":

        class Factory(EnvironmentFactory):
            async def create(self, request):
                if not fresh:
                    assert request.operation is EnvironmentFactoryOperation.RECONNECT
                    assert request.reconnect_metadata == {"target": "target"}
                    assert not request.execution_requirements.tool_requirements
                    reconnects.append(request)
                return EnvironmentFactoryResult(
                    environment=environment,
                    reconnect_metadata={"target": "target"},
                )

        app.register_environment_factory(EnvironmentSpec(name="coding"), Factory(), default=True)
    else:
        app.register_environment(environment, default=True)
    if fresh:
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="terminal-finalization",
                    messages=[Message.text("user", "write once")],
                    max_steps=1 if outcome == "interrupted" else 2,
                )
            )
        ]
        session = await store.load("terminal-finalization")
        assert session is not None and session.status.value == outcome
        checkpoint = await store.load_checkpoint(session.id)
        if mode == "ordinary":
            assert "pending_completion_finalization" not in checkpoint
            assert not any(
                event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED for event in events
            )
            assert (source / "result.txt").read_bytes() == b"retained output"
            assert (source / "a-first.txt").read_bytes() == b"first retained"
            assert await app.drain_environment_cleanups(timeout_s=10)
            assert not app._environment_lifecycle._active_environment_setups
            await store.close()
            return
        assert any(event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED for event in events)
        assert checkpoint["pending_completion_finalization"]["outcome"] == outcome
        assert app._environment_lifecycle._active_environment_setups
        assert (target / "result.txt").read_bytes() == b"retained output"
        if partial:
            assert (source / "a-first.txt").read_bytes() == b"first retained"
            assert (source / "result.txt").read_bytes() == b"original"
            (root / "first-mtime").write_text(str((source / "a-first.txt").stat().st_mtime_ns))
        os.kill(os.getpid(), signal.SIGKILL)
        raise AssertionError("SIGKILL returned")

    if control == "source_conflict" and mode == "recover":
        (source / "a-first.txt").write_bytes(b"external owner edit")
    before = await store.load_checkpoint("terminal-finalization")
    plan = await app.plan_recovery(
        RecoveryPlanRequest(
            selection=RecoveryPlanSelection(
                session_ids=("terminal-finalization",),
                inactive_for_seconds=0,
            )
        )
    )
    assert await store.load_checkpoint("terminal-finalization") == before
    item = plan.items[0]
    assert item.environment_recovery.completion_finalization_pending
    assert item.environment_recovery.finalization_outcome == outcome
    assert RecoveryPlanAction.AUTOMATIC_REPAIR in item.allowed_actions, plan.model_dump_json()
    request = RecoveryExecutionRequest(
        plan=plan,
        decisions=(
            RecoveryDecision(
                item_id=item.item_id,
                action=RecoveryPlanAction.AUTOMATIC_REPAIR,
            ),
        ),
        execution_id="terminal-cleanup",
    )
    clear = app._environment_lifecycle.clear_completion_finalization
    clear_started, allow_clear = asyncio.Event(), asyncio.Event()
    clear_mtimes = []

    async def controlled_clear(**kwargs):
        clear_started.set()
        if control == "cancel":
            await allow_clear.wait()
        await clear(**kwargs)
        clear_mtimes.append((source / "result.txt").stat().st_mtime_ns)
        if control == "lost_clear_ack" and len(clear_mtimes) == 1:
            raise OSError("lost finalization clear acknowledgement")

    if control in {"cancel", "lost_clear_ack"}:
        app._environment_lifecycle.clear_completion_finalization = controlled_clear
    operation = None
    try:
        if control == "cancel":
            operation = asyncio.create_task(app.execute_recovery(request))
            await asyncio.wait_for(clear_started.wait(), timeout=10)
            assert (source / "result.txt").read_bytes() == b"retained output"
            assert "pending_completion_finalization" in await store.load_checkpoint(
                "terminal-finalization"
            )
            operation.cancel("cancel finalization recovery observer")
            assert operation.cancelling() == 1
            await asyncio.sleep(0)
            assert not operation.done()
            allow_clear.set()
            with pytest.raises(asyncio.CancelledError):
                await operation
            assert operation.cancelled() and operation.cancelling() == 1
            fenced = await app.execute_recovery(request)
            assert fenced.items[0].error_code == "RecoveryPlanExecutionFenced"
            # Cancellation did not publish an execution receipt. The original
            # plan lease still fences retry even though binding cleanup settled.
            clock_offset += timedelta(seconds=901)
            result = await app.execute_recovery(request)
            assert result.items[0].error_code == "recovery_plan_outcome_unknown"
            assert not result.items[0].replayed
        else:
            result = await app.execute_recovery(request)
    finally:
        allow_clear.set()
        if operation is not None:
            await asyncio.gather(operation, return_exceptions=True)
        app._environment_lifecycle.clear_completion_finalization = clear
    if control == "source_conflict":
        assert result.items[0].error_code == "incomplete_session_recovery_failed", (
            result.model_dump_json()
        )
        assert result.items[0].final_session_status.value == outcome
        assert (source / "a-first.txt").read_bytes() == b"external owner edit"
        assert (source / "result.txt").read_bytes() == b"original"
        assert (target / "a-first.txt").read_bytes() == b"first retained"
        assert (target / "result.txt").read_bytes() == b"retained output"
        checkpoint = await store.load_checkpoint("terminal-finalization")
        assert (
            checkpoint["pending_completion_finalization"]
            == before["pending_completion_finalization"]
        )
        replay = await app.execute_recovery(request)
        assert replay.items[0] == result.items[0].model_copy(update={"replayed": True})
        assert len(reconnects) == (1 if kind == "factory" else 0)
        assert provider.requests == []
        events = await store.load_events("terminal-finalization")
        assert not any(event.type is EventType.SESSION_COMPLETED for event in events)
        await store.close()
        return
    if control == "lost_clear_ack":
        assert result.items[0].error_code == "OSError", result.model_dump_json()
    elif control == "normal":
        assert result.items[0].error_code is None, result.model_dump_json()
    assert result.items[0].final_session_status.value == outcome
    assert (source / "result.txt").read_bytes() == b"retained output"
    assert (source / "a-first.txt").read_bytes() == b"first retained"
    if partial:
        assert (
            str((source / "a-first.txt").stat().st_mtime_ns) == (root / "first-mtime").read_text()
        )
    assert len(reconnects) == (1 if kind == "factory" else 0)
    if clear_mtimes:
        assert (source / "result.txt").stat().st_mtime_ns == clear_mtimes[0]
    checkpoint = await store.load_checkpoint("terminal-finalization")
    assert "pending_completion_finalization" not in checkpoint
    replay = await app.execute_recovery(request)
    assert replay.items[0].replayed
    assert replay.items[0].receipt_event_id == result.items[0].receipt_event_id
    assert len(reconnects) == (1 if kind == "factory" else 0)
    assert provider.requests == []
    events = await store.load_events("terminal-finalization")
    assert not any(event.type is EventType.SESSION_COMPLETED for event in events)
    assert await app.drain_environment_cleanups(timeout_s=10)
    await store.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Requires POSIX SIGKILL")
@pytest.mark.parametrize("kind", ["static", "factory"])
@pytest.mark.parametrize("outcome", ["interrupted", "failed"])
@pytest.mark.parametrize("partial", [False, True])
def test_terminal_binding_finalization_survives_process_loss(tmp_path, kind, outcome, partial):
    script = (
        "import asyncio,sys; from pathlib import Path; "
        "from tests.faults.test_terminal_binding_finalization_restart import _child; "
        "asyncio.run(_child(Path(sys.argv[1]),sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]=='True'))"
    )
    for mode in ("seed", "recover"):
        process = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), mode, kind, outcome, str(partial)],
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert process.returncode == (-signal.SIGKILL if mode == "seed" else 0), (
            process.stdout,
            process.stderr,
        )


@pytest.mark.parametrize("kind", ["static", "factory"])
@pytest.mark.parametrize("outcome", ["interrupted", "failed"])
def test_terminal_binding_ordinary_cleanup_clears_its_marker(tmp_path, kind, outcome):
    asyncio.run(_child(tmp_path, "ordinary", kind, outcome, False))


@pytest.mark.skipif(sys.platform == "win32", reason="Requires POSIX SIGKILL")
@pytest.mark.parametrize("outcome", ["interrupted", "failed"])
@pytest.mark.parametrize("control", ["cancel", "lost_clear_ack"])
def test_terminal_binding_recovery_retains_clear_settlement(tmp_path, outcome, control):
    script = (
        "import asyncio,sys; from pathlib import Path; "
        "from tests.faults.test_terminal_binding_finalization_restart import _child; "
        "asyncio.run(_child(Path(sys.argv[1]),sys.argv[2],'factory',sys.argv[3],False,sys.argv[4]))"
    )
    for mode in ("seed", "recover"):
        process = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), mode, outcome, control],
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert process.returncode == (-signal.SIGKILL if mode == "seed" else 0), (
            process.stdout,
            process.stderr,
        )


@pytest.mark.skipif(sys.platform == "win32", reason="Requires POSIX SIGKILL")
@pytest.mark.parametrize("outcome", ["interrupted", "failed"])
@pytest.mark.parametrize("kind", ["static", "factory"])
def test_terminal_binding_recovery_preserves_conflicting_source(tmp_path, outcome, kind):
    script = (
        "import asyncio,sys; from pathlib import Path; "
        "from tests.faults.test_terminal_binding_finalization_restart import _child; "
        "asyncio.run(_child(Path(sys.argv[1]),sys.argv[2],sys.argv[3],sys.argv[4],"
        "False,'source_conflict'))"
    )
    for mode in ("seed", "recover", "recheck"):
        process = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), mode, kind, outcome],
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert process.returncode == (-signal.SIGKILL if mode == "seed" else 0), (
            process.stdout,
            process.stderr,
        )
