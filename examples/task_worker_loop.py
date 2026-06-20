"""Task worker loop example.

Usage:
    uv sync --extra dev
    PYTHONPATH=src .venv/bin/python examples/task_worker_loop.py

This example is intentionally API-key-free. It shows how app-owned worker code
can claim durable tasks, run a Cayu agent with the claimed task id, heartbeat the
lease while the run is active, fail a task before model execution, and reclaim
expired leases.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cayu import (
    AgentSpec,
    CayuApp,
    Event,
    InMemoryTaskStore,
    Message,
    RunRequest,
    Task,
    TaskCreate,
    TaskOrder,
    TaskQuery,
    TaskStatus,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent


class FakeProvider(ModelProvider):
    """Slow deterministic provider so the heartbeat loop has time to run."""

    name = "fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        await asyncio.sleep(0.2)
        yield ModelStreamEvent.text_delta("task processed")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


async def main() -> None:
    task_store = InMemoryTaskStore()
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(FakeProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    await app.create_task(
        TaskCreate(
            task_id="task_success",
            type="summarize",
            title="Summarize customer note",
            assigned_agent_name="assistant",
            input={"prompt": "Summarize this customer note in one sentence."},
        )
    )
    await app.create_task(
        TaskCreate(
            task_id="task_fail",
            type="summarize",
            title="Fail before model execution",
            assigned_agent_name="assistant",
        )
    )

    completed = await run_one_worker_task(app, task_store, worker_id="worker_a")
    print("completed", completed.id, completed.status, completed.session_id)

    failed = await fail_one_worker_task(task_store, worker_id="worker_a")
    print("failed", failed.id, failed.status, failed.error)

    await demonstrate_reclaim(task_store)


async def run_one_worker_task(
    app: CayuApp,
    task_store: InMemoryTaskStore,
    *,
    worker_id: str,
) -> Task:
    task = await task_store.claim_task(
        worker_id,
        TaskQuery(
            type="summarize",
            assigned_agent_name="assistant",
            order_by=TaskOrder.CREATED_AT_ASC,
        ),
        lease_seconds=30,
    )
    if task is None:
        raise RuntimeError("No task available.")
    print("claimed", task.id, task.status, task.worker_id)

    stop_heartbeat = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        heartbeat_until_done(task_store, task.id, worker_id, stop_heartbeat)
    )
    try:
        request = RunRequest(
            agent_name=task.assigned_agent_name or "assistant",
            session_id=f"session_{task.id}",
            task_id=task.id,
            task_worker_id=worker_id,
            messages=[
                Message.text(
                    "user",
                    str(task.input.get("prompt", "Process the claimed task.")),
                )
            ],
        )
        async for event in app.run(request):
            print_event(event)
    finally:
        stop_heartbeat.set()

    heartbeat_count = await heartbeat_task
    print("heartbeats", heartbeat_count)
    completed = await task_store.load_task(task.id)
    if completed is None:
        raise RuntimeError(f"Task disappeared: {task.id}")
    return completed


async def heartbeat_until_done(
    task_store: InMemoryTaskStore,
    task_id: str,
    worker_id: str,
    stop: asyncio.Event,
) -> int:
    count = 0
    while not stop.is_set():
        await asyncio.sleep(0.05)
        if stop.is_set():
            break
        try:
            await task_store.heartbeat(task_id, worker_id, extend_seconds=30)
        except ValueError:
            break
        count += 1
    return count


async def fail_one_worker_task(
    task_store: InMemoryTaskStore,
    *,
    worker_id: str,
) -> Task:
    task = await task_store.claim_task(
        worker_id,
        TaskQuery(
            status=TaskStatus.PENDING,
            type="summarize",
            assigned_agent_name="assistant",
            order_by=TaskOrder.CREATED_AT_ASC,
        ),
        lease_seconds=30,
    )
    if task is None:
        raise RuntimeError("No task available for failure path.")
    return await task_store.fail_task(
        task.id,
        {"message": "Worker setup failed before model execution."},
    )


async def demonstrate_reclaim(task_store: InMemoryTaskStore) -> None:
    await task_store.create_task(
        TaskCreate(
            task_id="task_expired",
            type="summarize",
            assigned_agent_name="assistant",
        )
    )
    expired = await task_store.claim_task(
        "stale_worker",
        TaskQuery(type="summarize", assigned_agent_name="assistant"),
        lease_seconds=1,
    )
    if expired is None:
        raise RuntimeError("No task available for reclaim path.")
    await asyncio.sleep(1.05)
    reclaimed = await task_store.reclaim_expired(
        query=TaskQuery(type="summarize", assigned_agent_name="assistant"),
        max_reclaims=10,
    )
    print("reclaimed", [task.id for task in reclaimed])


def print_event(event: Event) -> None:
    if event.type.startswith("model.text"):
        return
    print(event.type, event.payload)


if __name__ == "__main__":
    asyncio.run(main())
