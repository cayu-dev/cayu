"""A worker's pending runtime-failure handoff must honor an operator cancellation."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_failed_session_cancelled_task import _app, _FailingStopPolicy

from cayu import Message, RunRequest
from cayu.runtime._continuation_task_failure import (
    runtime_task_failure_terminalization_pending_from_error,
)
from cayu.sessions import InMemorySessionStore
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus, run_task_worker
from cayu.tasks.terminalization import TaskTerminalKind


async def _scenario(backend, operator_cancels, tmp_path, monkeypatch):
    tasks = (
        InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.sqlite")
    )
    app = _app(InMemorySessionStore(), tasks)
    created = await tasks.create_task(TaskCreate(type="answer"))
    failure_returned = asyncio.Event()
    finish_cleanup = asyncio.Event()
    original_terminalize = tasks.terminalize_task
    failures = 0

    async def temporarily_unavailable(self, request):
        nonlocal failures
        if request.kind is TaskTerminalKind.FAILED and failures < 3:
            failures += 1
            # Exhaust the runtime's bounded write retries before the storage
            # outage ends. Its elected failure is then handed to the worker.
            raise ConnectionError("task write temporarily unavailable")
        return await original_terminalize(request)

    monkeypatch.setattr(type(tasks), "terminalize_task", temporarily_unavailable)
    policy = _FailingStopPolicy()
    policy.release.set()

    async def handler(owned_app, task, worker_id):
        try:
            async for _ in owned_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="pending-failure",
                    task_id=task.id,
                    task_worker_id=worker_id,
                    task_lease_expires_at=task.lease_expires_at,
                    messages=[Message.text("user", "answer")],
                    loop_policies=(policy,),
                )
            ):
                pass
        except Exception as error:
            assert runtime_task_failure_terminalization_pending_from_error(error) is not None
            # Model normal handler cleanup while the operator acts concurrently.
            failure_returned.set()
            await finish_cleanup.wait()
            raise

    worker = asyncio.create_task(
        run_task_worker(
            app,
            tasks,
            handler,
            worker_id="owner",
            max_tasks=1,
            reclaim=False,
            recover_interrupted_handoffs=False,
        )
    )
    try:
        await asyncio.wait_for(failure_returned.wait(), 10)
        assert failures == 3
        if operator_cancels:
            await tasks.cancel_task(created.id, {"code": "operator_stop"})
        finish_cleanup.set()
        outcome = (await asyncio.gather(worker, return_exceptions=True))[0]
        current = await tasks.load_task(created.id)
        expected = TaskStatus.CANCELLED if operator_cancels else TaskStatus.FAILED
        assert current.status is expected, (
            "The worker abandoned the pending runtime failure's cancellation",
            current.status,
            current.status_reason,
            repr(outcome),
        )
        assert outcome == 1
        if operator_cancels:
            assert current.error == {"code": "operator_stop"}
    finally:
        finish_cleanup.set()
        if not worker.done():
            worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await app.aclose()
        if backend == "sqlite":
            await tasks.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("operator_cancels", [False, True])
def test_pending_runtime_failure_honors_cancellation(
    backend, operator_cancels, tmp_path, monkeypatch
):
    asyncio.run(_scenario(backend, operator_cancels, tmp_path, monkeypatch))
