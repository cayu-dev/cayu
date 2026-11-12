"""TaskGroup wrapping must not turn continuation authority loss into task failure."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from tests.core.test_task_worker import _seed_receipt_backed_continuation

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import ResumeRequest
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import TaskQuery, TaskStatus, run_task_worker
from cayu.tasks.memory import InMemoryTaskStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("grouped", [False, True])
def test_runtime_resume_fence_in_task_group_preserves_the_continuation(backend, grouped, tmp_path):
    current = _run_fenced_continuation(backend, tmp_path, grouped=grouped, cleanup_error=None)
    assert (current.status, current.worker_id, current.error) == (
        TaskStatus.RUNNING,
        None,
        None,
    ), "A runtime resume fence wrapped by TaskGroup destroyed the continuation"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_cleanup_failure_beside_a_resume_fence_fails_the_task(backend, tmp_path):
    current = _run_fenced_continuation(
        backend, tmp_path, grouped=True, cleanup_error=RuntimeError("cleanup broke")
    )
    assert current.status is TaskStatus.FAILED
    assert current.error is not None and current.error["error"] == "ExceptionGroup"


def _run_fenced_continuation(backend, tmp_path, *, grouped, cleanup_error):
    async def scenario():
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider([]), default=True)
        app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))
        observed = []

        async def fresh(*_args):
            raise AssertionError("Only a receipt-backed continuation is queued")

        try:
            await _seed_receipt_backed_continuation(app, store, task_id="job", session_id="session")
            selected = await app.session_store.load("session")
            assert selected is not None
            # A controller inspected the interrupted epoch, then another turn
            # entered and stopped before the inspected resume was dispatched.
            # Native status transitions advance the epoch; no store method or
            # worker behavior is mocked, and resume produces the fence itself.
            await app.session_store.update_status("session", SessionStatus.RUNNING)
            current = await app.session_store.update_status("session", SessionStatus.INTERRUPTED)
            assert current.run_epoch > selected.run_epoch

            async def continuation(owned_app, task, worker_id):
                async def resume():
                    try:
                        async for _ in owned_app.resume(
                            ResumeRequest(
                                session_id=task.session_id,
                                expected_run_epoch=selected.run_epoch,
                                messages=[Message.text("user", "continue")],
                                task_worker_id=worker_id,
                                task_handoff_id=task.interrupted_handoff_id,
                            )
                        ):
                            pass
                    except SessionRunFenced as error:
                        observed.append(error)
                        if cleanup_error is not None:
                            raise ExceptionGroup(
                                "continuation failed", [error, cleanup_error]
                            ) from None
                        raise

                if grouped:
                    # TaskGroup wraps even one failed child in ExceptionGroup.
                    async with asyncio.TaskGroup() as group:
                        group.create_task(resume())
                else:
                    await resume()

            assert (
                await run_task_worker(
                    app,
                    store,
                    fresh,
                    worker_id="owner",
                    max_tasks=1,
                    query=TaskQuery(type="job"),
                    recovered_interrupted_task_handler=continuation,
                    reclaim=False,
                    recover_interrupted_handoffs=False,
                )
                == 1
            )
            assert len(observed) == 1, (await store.load_task("job")).error
            assert "Resume expected run epoch" in str(observed[0])
            current = await store.load_task("job")
            assert current is not None
            if cleanup_error is None:
                reclaimed = await store.claim_interrupted_task_continuation(
                    "retry-owner", TaskQuery(type="job"), handoff_id=str(uuid4())
                )
                assert reclaimed.task is not None and reclaimed.task.id == "job"
            return current
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    return asyncio.run(scenario())
