"""Queue-backed dispatch worker example.

Usage:
    uv sync --extra dev
    PYTHONPATH=src .venv/bin/python examples/dispatch_worker.py

API-key-free. Shows the producer/consumer split of ``TaskStoreDispatcher``: ``app.dispatch()``
ENQUEUES dispatched work as a claimable, execution-profile-bound task instead of running it
inline. A separately constructed worker application claims it (atomically —
``PostgresTaskStore`` uses ``FOR UPDATE SKIP LOCKED``), resolves the recorded profile, and runs
it through the resume path. Backed here by in-memory stores (single process); inject Postgres
stores for a distributed worker pool. ``dispatcher.run_worker(app, ...)`` is the long-running
loop form of the single ``process_next`` call shown below.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cayu import (
    AgentSpec,
    CayuApp,
    DispatchRequest,
    ExecutionProfileBehaviorIdentity,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    RunRequest,
    TaskStoreDispatcher,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent


class FakeProvider(ModelProvider):
    name = "fake"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        """Bind the producer and worker to this example's exact adapter behavior."""

        return ExecutionProfileBehaviorIdentity(
            name="examples:dispatch-worker-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.text_delta("dispatched work done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


async def main() -> None:
    sessions = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)
    producer = CayuApp(
        session_store=sessions,
        task_store=tasks,
        dispatcher=dispatcher,
        enable_logging=False,
    )
    producer.register_provider(FakeProvider(), default=True)
    producer.register_agent(AgentSpec(name="assistant", model="fake-model"))

    # A dispatch resumes an existing session, so create one first by running it once.
    async for _ in producer.run(
        RunRequest(
            agent_name="assistant",
            session_id="sess_demo",
            messages=[Message.text("user", "start the session")],
        )
    ):
        pass

    # PRODUCER: enqueue dispatched work — returns a handle immediately without running it.
    handle = await producer.dispatch(
        DispatchRequest(
            session_id="sess_demo",
            messages=[Message.text("user", "do the queued follow-up")],
        )
    )
    queue_task_id = handle.metadata["queue_task_id"]
    pending = await tasks.load_task(queue_task_id)
    assert pending is not None
    print(
        "submitted",
        handle.dispatch_id,
        handle.status,
        "queued_task=",
        pending.status,
        "profile=",
        handle.metadata["required_execution_profile_fingerprint"],
    )

    # CONSUMER: a separately constructed worker must declare the same executable profile.
    worker = CayuApp(
        session_store=sessions,
        task_store=tasks,
        dispatcher=dispatcher,
        enable_logging=False,
    )
    worker.register_provider(FakeProvider(), default=True)
    worker.register_agent(AgentSpec(name="assistant", model="fake-model"))
    result = await dispatcher.process_next(worker, worker_id="worker_a")
    assert result is not None
    done = await tasks.load_task(queue_task_id)
    assert done is not None
    print("processed", result.dispatch_id, result.status, "queued_task=", done.status)

    # The queue is now empty.
    print("drained", await dispatcher.process_next(worker, worker_id="worker_a"))


if __name__ == "__main__":
    asyncio.run(main())
