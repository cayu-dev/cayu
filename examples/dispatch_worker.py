"""Queue-backed dispatch worker example.

Usage:
    uv sync --extra dev
    PYTHONPATH=src .venv/bin/python examples/dispatch_worker.py

API-key-free. Shows the producer/consumer split of ``TaskStoreDispatcher``: ``app.dispatch()``
ENQUEUES dispatched work as a claimable task instead of running it inline, and a separate
worker claims it (atomically — ``PostgresTaskStore`` uses ``FOR UPDATE SKIP LOCKED``) and runs
it through the resume path. Backed here by ``InMemoryTaskStore`` (single process); inject a
``PostgresTaskStore`` for a distributed worker pool. ``dispatcher.run_worker(app, ...)`` is the
long-running loop form of the single ``process_next`` call shown below.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cayu import (
    AgentSpec,
    CayuApp,
    DispatchRequest,
    InMemoryTaskStore,
    Message,
    RunRequest,
    TaskStoreDispatcher,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent


class FakeProvider(ModelProvider):
    name = "fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.text_delta("dispatched work done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


async def main() -> None:
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)
    app = CayuApp(
        task_store=tasks,
        dispatcher=dispatcher,
        enable_logging=False,
    )
    app.register_provider(FakeProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    # A dispatch resumes an existing session, so create one first by running it once.
    async for _ in app.run(
        RunRequest(
            agent_name="assistant",
            session_id="sess_demo",
            messages=[Message.text("user", "start the session")],
        )
    ):
        pass

    # PRODUCER: enqueue dispatched work — returns a handle immediately without running it.
    handle = await app.dispatch(
        DispatchRequest(
            session_id="sess_demo",
            messages=[Message.text("user", "do the queued follow-up")],
        )
    )
    queue_task_id = handle.metadata["queue_task_id"]
    pending = await tasks.load_task(queue_task_id)
    assert pending is not None
    print("submitted", handle.dispatch_id, handle.status, "queued_task=", pending.status)

    # CONSUMER: a worker claims the next queued dispatch and runs it to completion.
    result = await dispatcher.process_next(app, worker_id="worker_a")
    assert result is not None
    done = await tasks.load_task(queue_task_id)
    assert done is not None
    print("processed", result.dispatch_id, result.status, "queued_task=", done.status)

    # The queue is now empty.
    print("drained", await dispatcher.process_next(app, worker_id="worker_a"))


if __name__ == "__main__":
    asyncio.run(main())
