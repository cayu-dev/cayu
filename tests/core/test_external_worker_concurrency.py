"""External shutdown retains keyed task leases until adapter cleanup finishes."""

import asyncio

import pytest

from cayu import CayuApp, RunRequest
from cayu.deadlines import ExecutionDeadline
from cayu.external_agents import ExternalAgentEvent, run_external_agent
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskQuery, TaskStatus, run_task_worker


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_concurrent_external_restart_shutdown(backend, tmp_path):
    async def scenario():
        app = CayuApp(
            task_store=InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.db"),
            session_store=None
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.db"),
            enable_logging=False,
        )
        for i in range(4):
            await app.create_task(
                TaskCreate(type="vendor", restartable=True, concurrency_key=f"repo-{i % 2}")
            )
        entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        old, closing = [], []

        class Adapter:
            name, provider_name, model = "test", "test", "test"

            async def stream(self, ctx):
                old.append(ctx.session_id)
                if len(old) == 2:
                    entered.set()
                try:
                    await asyncio.Event().wait()
                    yield
                finally:
                    closing.append(ctx.session_id)
                    if len(closing) == 2:
                        cleaning.set()
                    await release.wait()

        adapter = Adapter()

        async def handler(app, task, worker):
            await run_external_agent(
                app,
                RunRequest(
                    agent_name="vendor",
                    messages=[],
                    task_id=task.id,
                    task_worker_id=worker,
                    task_lease_expires_at=task.lease_expires_at,
                    execution_deadline=ExecutionDeadline.after(30),
                ),
                adapter,
                prompt="review",
            )

        worker = asyncio.create_task(
            run_task_worker(
                app, app.task_store, handler, worker_id="first", concurrency=4, max_tasks=4
            )
        )
        await asyncio.wait_for(entered.wait(), 10)
        worker.cancel()
        await asyncio.wait_for(cleaning.wait(), 10)
        assert not worker.done()
        assert await app.task_store.claim_task("competitor") is None
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(worker, 10)
        pending = await app.task_store.list_tasks(TaskQuery())
        assert len(pending) == 4 and all(t.status is TaskStatus.PENDING for t in pending)

        assert sum(t.restart_count for t in pending) == 2

        class Fresh(Adapter):
            async def stream(self, ctx):
                assert ctx.session_id not in old
                yield ExternalAgentEvent("result", {"output": "done"}, "result")

        adapter = Fresh()
        assert (
            await run_task_worker(
                app, app.task_store, handler, worker_id="next", concurrency=4, max_tasks=4
            )
            == 4
        )
        assert all(
            t.status is TaskStatus.COMPLETED for t in await app.task_store.list_tasks(TaskQuery())
        )
        await app.aclose()

    asyncio.run(scenario())
