"""Subprocess worker used to verify durable shutdown across process lifetimes."""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from pathlib import Path

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.evals.testing import ScriptedModelProvider
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import complete_managed_task, run_task_worker


async def main() -> None:
    database, ready_path, mode = sys.argv[1:]
    store = SQLiteTaskStore(database)
    app = CayuApp(task_store=store, enable_logging=False)

    class BlockingProvider(ScriptedModelProvider):
        async def stream(self, request):
            Path(ready_path).write_text("ready")
            await asyncio.Event().wait()
            async for event in super().stream(request):
                yield event

    app.register_provider(BlockingProvider([]), default=True)
    app.register_agent(AgentSpec(name="worker", model="fake"))

    async def handler(_app, task, worker_id):
        if mode == "complete":
            assert task.restart_count == 1
            assert task.session_id is None
            await complete_managed_task(store, task, worker_id, {"restarted": True})
            return
        async for _ in app.run(
            RunRequest(
                agent_name="worker",
                messages=[Message.text("user", "wait")],
                task_id=task.id,
                task_worker_id=worker_id,
                task_lease_expires_at=task.lease_expires_at,
            )
        ):
            pass

    worker = asyncio.create_task(
        run_task_worker(
            app,
            store,
            handler,
            worker_id=mode,
            max_tasks=1,
            poll_interval_s=0.01,
        )
    )
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, worker.cancel)
    with contextlib.suppress(asyncio.CancelledError):
        await worker
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
