"""Cancellation after copy-back must still finish the committed completion."""

import asyncio
import contextlib

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus
from cayu.environments import Environment, EnvironmentSpec, SyncBinding
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore
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
