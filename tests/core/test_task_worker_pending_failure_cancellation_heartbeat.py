"""Pending runtime-failure cancellation must survive its own claim's renewal."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_task_worker_pending_failure_cancellation import _scenario

from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore
from cayu.tasks.terminalization import TaskTerminalKind


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_pending_runtime_failure_cancellation_survives_heartbeat(backend, tmp_path, monkeypatch):
    async def scenario():
        store_type = InMemoryTaskStore if backend == "memory" else SQLiteTaskStore
        original_terminalize = store_type.terminalize_task
        original_heartbeat = store_type.heartbeat
        cancellation_started = asyncio.Event()
        renewed = asyncio.Event()
        cancellation_committed = False

        async def heartbeat(self, *args, **kwargs):
            task = await original_heartbeat(self, *args, **kwargs)
            if cancellation_started.is_set():
                renewed.set()
            return task

        async def terminalize(self, request):
            nonlocal cancellation_committed
            if request.kind is TaskTerminalKind.CANCELLED:
                before = await self.load_task(request.task_id)
                cancellation_started.set()
                # Hold the write until the worker's real heartbeat renews the
                # default lease; no lease expiry or reclaim is involved.
                await asyncio.wait_for(renewed.wait(), 5)
                after = await self.load_task(request.task_id)
                assert after.lease_expires_at > before.lease_expires_at
                assert after.claim_generation == before.claim_generation
                task = await original_terminalize(self, request)
                cancellation_committed = True
                return task
            # The elected failure request must retain its original identity.
            assert request.claim_generation is None
            return await original_terminalize(self, request)

        monkeypatch.setattr(store_type, "heartbeat", heartbeat)
        monkeypatch.setattr(store_type, "terminalize_task", terminalize)
        await _scenario(backend, True, tmp_path, monkeypatch)
        assert renewed.is_set()
        assert cancellation_committed

    asyncio.run(scenario())
