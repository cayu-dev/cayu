"""Completion cleanup after real process loss does not execute current tools."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    EventType,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    SyncBinding,
    Tool,
    ToolExecutableRequirement,
    ToolExecutionRequirement,
    ToolSpec,
)
from cayu.runtime._environment_exposure import require_environment_exposed


async def _child(root: Path, mode: str, kind: str, changed: bool) -> None:
    source, target = root / "source", root / "target"
    if mode == "seed":
        source.mkdir()
        target.mkdir()
        (source / "result.txt").write_text("original", encoding="utf-8")

    class InterruptedBinding(SyncBinding):
        async def finalize(self, bound, *, outcome=None, metadata=None):
            (target / "result.txt").write_text("recovered", encoding="utf-8")
            raise asyncio.CancelledError("retain completion for process-loss recovery")

    class UnusedTool(Tool):
        async def run(self, ctx, args):
            raise AssertionError("Completion cleanup must not execute tools")

    store = SQLiteSessionStore(root / "sessions.sqlite")
    app = CayuApp(session_store=store, enable_logging=False)
    provider = ScriptedModelProvider(
        [[ModelStreamEvent.completed({"finish_reason": "stop"})]] if mode == "seed" else [],
        name="completion-provider",
    )
    app.register_provider(provider, default=True)
    environment = Environment(
        EnvironmentSpec(name="completion"),
        workspace=LocalWorkspace(source, workspace_id="completion-source"),
        binding=(InterruptedBinding if mode == "seed" else SyncBinding)(
            target_workspace=LocalWorkspace(target, workspace_id="completion-target"),
            source_conflict_policy="require_revision",
            max_file_bytes=1024,
        ),
    )
    reconnects = []
    if kind == "factory":

        class Factory(EnvironmentFactory):
            async def create(self, request):
                if mode == "recover":
                    assert request.operation is EnvironmentFactoryOperation.RECONNECT
                    assert request.reconnect_metadata == {"target": "completion-target"}
                    assert not request.execution_requirements.tool_requirements
                    reconnects.append(request)
                return EnvironmentFactoryResult(
                    environment=environment,
                    reconnect_metadata={"target": "completion-target"},
                )

        app.register_environment_factory(
            EnvironmentSpec(name="completion"), Factory(), default=True
        )
    else:
        app.register_environment(environment, default=True)
    requirements = (
        (
            ToolExecutionRequirement(
                name="new_dependency",
                alternatives=(ToolExecutableRequirement(executable="rg"),),
            ),
        )
        if mode == "recover" and changed
        else ()
    )
    app.register_agent(
        AgentSpec(name="agent", model="scripted-model", provider_name="completion-provider"),
        tools=[UnusedTool(ToolSpec(name="unused", execution_requirements=requirements))],
    )
    if mode == "seed":
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="completion",
                    messages=[Message.text("user", "finish")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_FAILED
        checkpoint = await store.load_checkpoint("completion")
        assert checkpoint is not None and "pending_completion_finalization" in checkpoint
        assert app._environment_lifecycle._active_environment_setups
        # The durable marker exists and the process-local owner is still live.
        # Kill the process, not merely its coroutine, before any cleanup retry.
        os.kill(os.getpid(), signal.SIGKILL)
        raise AssertionError("SIGKILL returned")
    bind = app._environment_lifecycle.bind
    recovered_bindings = []

    async def check_cleanup_binding(**kwargs):
        result = await bind(**kwargs)
        if kwargs.get("completion_recovery") is not None and result.error is None:
            recovered = result.registered_environment
            assert recovered is not None and recovered.environment_exposure is None
            context = kwargs["invocation_context"].with_registered_environment(
                recovered, validated_profile=kwargs["execution_profile"]
            )
            with pytest.raises(RuntimeError, match="runtime-admitted exposure authority"):
                require_environment_exposed(
                    recovered,
                    session=kwargs["session"],
                    invocation_context=context,
                    registered_agent=kwargs["registered_agent"],
                    execution_profile=kwargs["execution_profile"],
                )
            recovered_bindings.append(recovered)
        return result

    app._environment_lifecycle.bind = check_cleanup_binding
    try:
        result = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="completion", reason="process loss")
        )
        assert result.actions == (IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_FINALIZATION,)
        assert len(reconnects) == (1 if kind == "factory" else 0)
        assert len(recovered_bindings) == 1
        assert (source / "result.txt").read_text(encoding="utf-8") == "recovered"
        checkpoint = await store.load_checkpoint("completion")
        assert checkpoint is not None and "pending_completion_finalization" not in checkpoint
        assert not any(
            event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
            and event.payload.get("phase") == "exposure"
            for event in result.events
        )
        assert provider.requests == []
        if changed:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="ordinary-new-requirement",
                        messages=[Message.text("user", "run normally")],
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_FAILED
            assert not any(event.type is EventType.MODEL_STARTED for event in events)
            assert provider.requests == []
        assert await app.drain_environment_cleanups(timeout_s=10)
        assert not app._environment_lifecycle._active_environment_setups
    finally:
        await store.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Requires POSIX SIGKILL")
@pytest.mark.parametrize("kind", ["static", "factory"])
@pytest.mark.parametrize("changed", [False, True])
def test_completion_recovery_after_process_loss(tmp_path, kind, changed):
    script = (
        "import asyncio,sys; from pathlib import Path; "
        "from tests.faults.test_completion_tool_requirement_restart import _child; "
        "asyncio.run(_child(Path(sys.argv[1]),sys.argv[2],sys.argv[3],sys.argv[4]=='True'))"
    )
    for mode in ("seed", "recover"):
        process = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), mode, kind, str(changed)],
            capture_output=True,
            text=True,
            timeout=45,
        )
        expected = -signal.SIGKILL if mode == "seed" else 0
        assert process.returncode == expected, (process.stdout, process.stderr)
