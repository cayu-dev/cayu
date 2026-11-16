"""Cancellation around completion-critical copy-back must settle the session once."""

import asyncio
import contextlib

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus
from cayu.environments import Environment, EnvironmentSpec, SyncBinding
from cayu.environments.bindings import SyncBindingStagingCapacity
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
from cayu.sessions.requests import InterruptSessionRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus
from cayu.workspaces import LocalWorkspace


class _CancelAtCompletionTasks(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.attempts = 0

    async def complete_task(self, task_id, result, **kwargs):
        # Model an abortable database wait before dispatching the mutation.
        # Cancellation here leaves no task mutation running in the background.
        self.attempts += 1
        if self.attempts == 1:
            self.entered.set()
            await self.release.wait()
        return await super().complete_task(task_id, result, **kwargs)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("critical", [False, True], ids=["ordinary", "copy-back"])
@pytest.mark.parametrize("cancel", [False, True], ids=["finish", "cancel"])
def test_cancel_after_workspace_commit_finishes_completion(tmp_path, backend, critical, cancel):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "completion.sqlite")
        )
        tasks = _CancelAtCompletionTasks()
        await tasks.create_task(TaskCreate(task_id="task", type="respond"))
        app = CayuApp(session_store=store, task_store=tasks, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        if critical:
            source, target = tmp_path / "source", tmp_path / "target"
            source.mkdir()
            target.mkdir()
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="sync"),
                    workspace=LocalWorkspace(source, workspace_id="source"),
                    binding=SyncBinding(
                        target_workspace=LocalWorkspace(target, workspace_id="target")
                    ),
                ),
                default=True,
            )

        async def consume():
            async with contextlib.aclosing(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="completion",
                        task_id="task",
                        messages=[Message.text("user", "answer")],
                    )
                )
            ) as stream:
                return [event async for event in stream]

        owner = asyncio.create_task(consume())
        entered = asyncio.create_task(tasks.entered.wait())
        try:
            done, _ = await asyncio.wait(
                {owner, entered}, timeout=10, return_when=asyncio.FIRST_COMPLETED
            )
            if owner in done:
                pytest.fail(f"Run ended before task completion: {owner.result()!r}")
            assert entered in done
            await entered
            # Copy-back has already succeeded and the marker has been cleared.
            assert (await store.load("completion")).status is SessionStatus.COMPLETED
            assert "pending_completion_finalization" not in (
                await store.load_checkpoint("completion")
            )
            if cancel:
                owner.cancel("caller disconnected after commit")
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(owner, 10)
            else:
                tasks.release.set()
                await asyncio.wait_for(owner, 10)
            assert await app.drain_recovery_cleanups(timeout_s=5)
            assert await app.drain_environment_cleanups(timeout_s=5)
            # The normal sweep should also be able to finish a committed outcome.
            await app.recover_abandoned_sessions(inactive_for_seconds=0)
            task = await tasks.load_task("task")
            types = [event.type for event in await store.load_events("completion")]
            assert (task.status, types.count(EventType.SESSION_COMPLETED)) == (
                TaskStatus.COMPLETED,
                1,
            )
        finally:
            tasks.release.set()
            if not owner.done():
                owner.cancel()
            entered.cancel()
            await asyncio.gather(owner, entered, return_exceptions=True)
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("interrupt", [False, True], ids=["finish", "interrupt"])
def test_operator_interrupt_during_copyback_settles(tmp_path, monkeypatch, backend, interrupt):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "interrupted-copyback.sqlite")
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        source, target = tmp_path / "source", tmp_path / "target"
        source.mkdir()
        target.mkdir()
        capacity = SyncBindingStagingCapacity(max_concurrency=1)
        competing_transfer = None
        interrupt_dispatched = asyncio.Event()

        app.register_environment(
            Environment(
                EnvironmentSpec(name="sync"),
                workspace=LocalWorkspace(source, workspace_id="source"),
                binding=SyncBinding(
                    target_workspace=LocalWorkspace(target, workspace_id="target"),
                    staging_capacity=capacity,
                ),
            ),
            default=True,
        )
        native_cancel = app._session_control.cancel_active_runs

        def observe_interrupt_dispatch(session_id):
            result = native_cancel(session_id)
            interrupt_dispatched.set()
            return result

        monkeypatch.setattr(app._session_control, "cancel_active_runs", observe_interrupt_dispatch)

        async def consume():
            nonlocal competing_transfer
            async with contextlib.aclosing(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="copyback-interrupt",
                        messages=[Message.text("user", "answer")],
                    )
                )
            ) as stream:
                async for event in stream:
                    if event.type is EventType.MODEL_STARTED:
                        (target / "answer.txt").write_text("saved answer")
                        # Simulate another transfer occupying the shared governor
                        # after copy-in but before this run starts copy-back.
                        competing_transfer = await capacity._acquire_transfer()

        async def wait_for_copyback():
            while capacity.snapshot().waiting_transfers != 1:
                await asyncio.sleep(0.01)

        async def request_interrupt():
            return [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(session_id="copyback-interrupt", reason="stop")
                )
            ]

        owner = asyncio.create_task(consume())
        controller = None
        try:
            await asyncio.wait_for(wait_for_copyback(), 10)
            assert (await store.load("copyback-interrupt")).status is SessionStatus.RUNNING
            checkpoint = await store.load_checkpoint("copyback-interrupt")
            assert "pending_completion_finalization" in checkpoint
            assert any(
                event.type is EventType.INTERACTION_COMPLETED
                for event in await store.load_events("copyback-interrupt")
            )
            interruption_error = None
            interruption = []
            if interrupt:
                controller = asyncio.create_task(request_interrupt())
                await asyncio.wait_for(interrupt_dispatched.wait(), 10)
                assert (await store.load("copyback-interrupt")).status is SessionStatus.INTERRUPTING
            assert competing_transfer is not None
            competing_transfer.release()
            if controller is not None:
                try:
                    interruption = await asyncio.wait_for(controller, 10)
                except TimeoutError as error:
                    interruption_error = error
            owner_outcome = await asyncio.wait_for(
                asyncio.gather(owner, return_exceptions=True), 10
            )
            recovery_drained = await app.drain_recovery_cleanups(timeout_s=5)
            environment_drained = await app.drain_environment_cleanups(timeout_s=5)
            session = await store.load("copyback-interrupt")
            types = [event.type for event in await store.load_events(session.id)]
            expected_status = SessionStatus.INTERRUPTED if interrupt else SessionStatus.COMPLETED
            expected_event = (
                EventType.SESSION_INTERRUPTED if interrupt else EventType.SESSION_COMPLETED
            )
            assert (session.status, types.count(expected_event)) == (expected_status, 1), (
                f"status={session.status}, interruption_error={interruption_error!r}, "
                f"owner_outcome={owner_outcome!r}, cleanup_drained="
                f"{(recovery_drained, environment_drained)}, events={[str(t) for t in types]}"
            )
            assert recovery_drained and environment_drained
            assert EventType.SESSION_FAILED not in types
            # The owning run settles its own turn, interrupted or completed.
            assert types.count(EventType.TURN_COMPLETED) == 1
            if interrupt:
                assert interruption_error is None
                assert any(event.type is EventType.SESSION_INTERRUPTED for event in interruption)
            else:
                assert (source / "answer.txt").read_text() == "saved answer"
        finally:
            if competing_transfer is not None:
                competing_transfer.release()
            tasks = [task for task in (owner, controller) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_copyback_owed_by_interruption_survives_worker_loss(tmp_path, backend):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "interrupted-copyback-recovery.sqlite")
        )
        source, target = tmp_path / "source", tmp_path / "target"
        source.mkdir()
        target.mkdir()
        capacity = SyncBindingStagingCapacity(max_concurrency=1)

        def make_app():
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(
                ScriptedModelProvider(
                    [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]]
                ),
                default=True,
            )
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="sync"),
                    workspace=LocalWorkspace(source, workspace_id="source"),
                    binding=SyncBinding(
                        target_workspace=LocalWorkspace(target, workspace_id="target"),
                        staging_capacity=capacity,
                    ),
                ),
                default=True,
            )
            return app

        app = make_app()
        competing_transfer = None
        recovering_app = None
        worker_closed = False

        async def consume():
            nonlocal competing_transfer
            async with contextlib.aclosing(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="copyback-recovery",
                        messages=[Message.text("user", "answer")],
                    )
                )
            ) as stream:
                async for event in stream:
                    if event.type is EventType.MODEL_STARTED:
                        (target / "answer.txt").write_text("saved answer")
                        competing_transfer = await capacity._acquire_transfer()

        async def wait_for_copyback():
            while capacity.snapshot().waiting_transfers != 1:
                await asyncio.sleep(0.01)

        owner = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(wait_for_copyback(), 10)
            interruption = await asyncio.wait_for(
                _collect(
                    app.interrupt_session(
                        InterruptSessionRequest(session_id="copyback-recovery", reason="stop")
                    )
                ),
                10,
            )
            assert any(event.type is EventType.SESSION_INTERRUPTED for event in interruption)
            await asyncio.wait_for(asyncio.gather(owner, return_exceptions=True), 10)
            assert (await store.load("copyback-recovery")).status is SessionStatus.INTERRUPTED
            # The interruption now owes the retained copy-back; recovery after a
            # worker loss accepts that pairing instead of rejecting it.
            marker = (await store.load_checkpoint("copyback-recovery"))[
                "pending_completion_finalization"
            ]
            assert marker["outcome"] == "interrupted"
            # The worker shuts down while staging is still saturated, so its
            # process-local retry cannot finish the copy-back.
            await app.aclose(timeout_s=1)
            worker_closed = True
            assert competing_transfer is not None
            competing_transfer.release()
            competing_transfer = None
            recovering_app = make_app()
            result = await recovering_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="copyback-recovery")
            )
            assert result.status is SessionStatus.INTERRUPTED, result
            assert "pending_completion_finalization" not in (
                await store.load_checkpoint("copyback-recovery") or {}
            )
            assert (source / "answer.txt").read_text() == "saved answer"
            types = [event.type for event in await store.load_events("copyback-recovery")]
            assert types.count(EventType.SESSION_INTERRUPTED) == 1
            assert EventType.SESSION_COMPLETED not in types
            assert EventType.SESSION_FAILED not in types
        finally:
            if competing_transfer is not None:
                competing_transfer.release()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            if not worker_closed:
                await app.aclose()
            if recovering_app is not None:
                await recovering_app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


async def _collect(stream):
    return [event async for event in stream]
