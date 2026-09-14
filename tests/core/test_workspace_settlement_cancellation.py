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
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.deadlines import ExecutionDeadline
from cayu.events import EventType
from cayu.messages import Message
from cayu.sessions.base import (
    EventQuery,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    RunRequest,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.workflows.base import WorkflowSpec
from cayu.workflows.models import StepError
from cayu.workflows.workflow import StepRunOptions, WorkflowBase, step


class SettlementWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="settlement-control")

    async def run(self, session_id):
        yield await self.context(session_id).start()


@pytest.mark.parametrize("cancel_phase", ["workspace_terminal", "runner_terminal"])
@pytest.mark.parametrize("native_child", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_cancel_before_workspace_terminal(
    tmp_path,
    monkeypatch,
    cancel,
    backend,
    native_child,
    cancel_phase,
    runner_factory=LocalRunner,
    provider_factory=_ScriptedProvider,
    child_deadline=False,
    binding_factory=DeterministicWorkspaceBinding,
    child_deadline_seconds=5,
):
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
                    runner=runner_factory(tmp_path),
                    binding=binding_factory(),
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="scripted-model"), tools=[ExecCommandTool()]
            )
            return app

        provider = provider_factory()
        app = registered_app(provider)
        reached = asyncio.Event()
        release_runner_publication = asyncio.Event()
        original = executor.publish_workspace_observation_transition
        paused = False
        session_id = "settlement-probe"

        async def intercept(**kwargs):
            nonlocal paused, session_id
            session_id = kwargs["session"].id
            if (
                cancel
                and cancel_phase == "workspace_terminal"
                and kwargs.get("phase") == "terminal"
                and not paused
            ):
                paused = True
                reached.set()
                await asyncio.Event().wait()
            return await original(**kwargs)

        monkeypatch.setattr(executor, "publish_workspace_observation_transition", intercept)

        original_emit = executor.RuntimeEventWriter.emit

        async def emit(writer, event, *args, **kwargs):
            nonlocal session_id, paused
            saved = await original_emit(writer, event, *args, **kwargs)
            if (
                cancel
                and cancel_phase == "runner_terminal"
                and event.type == EventType.RUNNER_EXEC_COMPLETED
                and not paused
            ):
                session_id = event.session_id
                paused = True
                reached.set()
                await release_runner_publication.wait()
            return saved

        monkeypatch.setattr(executor.RuntimeEventWriter, "emit", emit)

        async def consume():
            if native_child:
                ctx = SettlementWorkflow(app).context("settlement-parent")
                await ctx.start()
                return await step(
                    ctx,
                    agent="assistant",
                    step_id="write",
                    prompt="create a file",
                    run_options=StepRunOptions(
                        execution_deadline=ExecutionDeadline.after(
                            child_deadline_seconds if child_deadline else None
                        )
                    ),
                )
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
                await asyncio.wait_for(reached.wait(), max(15, child_deadline_seconds + 5))
                if child_deadline:
                    with pytest.raises(StepError) as caught:
                        await asyncio.wait_for(task, max(15, child_deadline_seconds + 5))
                    assert caught.value.evidence.classification == "deadline"
                    assert not caught.value.evidence.secondary_failures
                else:
                    task.cancel("cancel before terminal staging")
                if cancel_phase == "runner_terminal" and not child_deadline:
                    await asyncio.sleep(0.01)
                    task.cancel("repeated cancellation during runner publication")
                    await asyncio.sleep(0.01)
                    release_runner_publication.set()
                if not child_deadline:
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    if cancel_phase == "runner_terminal":
                        assert task.cancelling() == 2
            else:
                await task
            checkpoint = await store.load_checkpoint(session_id)
            if child_deadline:
                assert not checkpoint.get("workspace_observations")
                assert not checkpoint.get("pending_tool_round")
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
            recovery_provider = provider_factory()
            recovery_app = registered_app(recovery_provider)
            for _ in range(2):
                await recovery_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id=session_id,
                    )
                )
            checkpoint = await store.load_checkpoint(session_id)
            assert not checkpoint.get("workspace_observations")
            assert not checkpoint.get("pending_tool_round")
            assert provider.requests == requests
            assert recovery_provider.requests == 0
            assert (tmp_path / "shell.txt").read_text() == "settled sentinel"
            events = await store.query_events(EventQuery(session_id=session_id))
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


@pytest.mark.parametrize("configured_binding", [False, True])
def test_native_deadline_settles_workspace_publication(tmp_path, monkeypatch, configured_binding):
    test_cancel_before_workspace_terminal(
        tmp_path,
        monkeypatch,
        cancel=True,
        backend="sqlite",
        native_child=True,
        cancel_phase="workspace_terminal",
        child_deadline=True,
        binding_factory=DeterministicWorkspaceBinding if configured_binding else lambda: None,
    )
