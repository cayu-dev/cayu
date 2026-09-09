"""A terminal workspace publication must retain its recovery owner."""

import asyncio

import pytest
from tests.core.test_workspace_mutation_receipts import (
    DeterministicWorkspaceBinding,
    Environment,
    ExecCommandTool,
    LocalRunner,
    LocalWorkspace,
    _portable_environment_spec,
    _ScriptedProvider,
)

import cayu.runtime._tool_round_executor as executor
from cayu.core import AgentSpec, EventType, Message
from cayu.runtime import (
    CayuApp,
    EventQuery,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    RunRequest,
)
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_cancel_before_workspace_terminal(tmp_path, monkeypatch, cancel, backend):
    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "probe.sqlite")
        )

        def registered_app(provider):
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_environment(
                Environment(
                    _portable_environment_spec("local"),
                    workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
                    runner=LocalRunner(tmp_path),
                    binding=DeterministicWorkspaceBinding(),
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="scripted-model"), tools=[ExecCommandTool()]
            )
            return app

        provider = _ScriptedProvider()
        app = registered_app(provider)
        reached = asyncio.Event()
        original = executor.publish_workspace_observation_transition
        paused = False

        async def intercept(**kwargs):
            nonlocal paused
            if cancel and kwargs.get("phase") == "terminal" and not paused:
                paused = True
                reached.set()
                await asyncio.Event().wait()
            return await original(**kwargs)

        monkeypatch.setattr(executor, "publish_workspace_observation_transition", intercept)

        async def consume():
            return [
                e
                async for e in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="settlement-probe",
                        messages=[Message.text("user", "create a file")],
                    )
                )
            ]

        task = asyncio.create_task(consume())
        try:
            if cancel:
                await asyncio.wait_for(reached.wait(), 15)
                task.cancel("cancel before workspace terminal")
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await task
            checkpoint = await store.load_checkpoint("settlement-probe")
            assert checkpoint.get("pending_tool_round") or not checkpoint.get(
                "workspace_observations"
            )
            assert (tmp_path / "shell.txt").read_text() == "created"
            # Reexecution would recreate the file and change this sentinel.
            (tmp_path / "shell.txt").write_text("settled sentinel")
            requests = provider.requests
            if backend == "sqlite":
                await store.close()
                store = SQLiteSessionStore(tmp_path / "probe.sqlite")
            recovery_provider = _ScriptedProvider()
            recovery_app = registered_app(recovery_provider)
            for _ in range(2):
                await recovery_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id="settlement-probe",
                    )
                )
            checkpoint = await store.load_checkpoint("settlement-probe")
            assert not checkpoint.get("workspace_observations")
            assert not checkpoint.get("pending_tool_round")
            assert provider.requests == requests
            assert recovery_provider.requests == 0
            assert (tmp_path / "shell.txt").read_text() == "settled sentinel"
            events = await store.query_events(EventQuery(session_id="settlement-probe"))
            assert sum(e.event.type == EventType.TOOL_CALL_COMPLETED for e in events) == 1
            assert (
                len(
                    [e for e in events if e.event.type == EventType.WORKSPACE_OBSERVATION_FINALIZED]
                )
                == 1
            )
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if backend == "sqlite":
                await store.close()

    asyncio.run(run())
