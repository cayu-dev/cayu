from __future__ import annotations

import asyncio

import pytest

from cayu.tasks._group_maintenance import TaskGroupMaintenance
from cayu.tasks.base import InMemoryTaskStore
from cayu.vaults.redaction import SecretRedactor


def test_worker_entrances_share_one_inflight_scan_and_idle_cadence(monkeypatch):
    async def run():
        store = InMemoryTaskStore()
        entered = asyncio.Event()
        release = asyncio.Event()
        scans = []

        async def candidates(*, after_group_id, limit):
            scans.append((after_group_id, limit))
            entered.set()
            await release.wait()
            return []

        monkeypatch.setattr(store, "list_task_group_reconciliation_candidates", candidates)
        entrances = [TaskGroupMaintenance.for_store(store) for _ in range(100)]
        owner = asyncio.create_task(entrances[0].step(store, SecretRedactor(), now=10))
        await entered.wait()
        await asyncio.gather(*(m.step(store, SecretRedactor(), now=10) for m in entrances[1:]))
        assert scans == [(None, 32)]
        release.set()
        await owner
        await asyncio.gather(*(m.advance(store, now=10.5) for m in entrances))
        assert scans == [(None, 32)]
        await entrances[-1].advance(store, now=11)
        assert scans == [(None, 32), (None, 32)]
        assert TaskGroupMaintenance.for_store(InMemoryTaskStore()) is not entrances[0]

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_failed_maintenance_releases_owner_and_retries_unacknowledged_identity(monkeypatch, cancel):
    async def run():
        store = InMemoryTaskStore()
        cursors = []
        reconciled = []
        fail = True

        async def candidates(*, after_group_id, limit):
            cursors.append(after_group_id)
            return ["group-b"] if after_group_id else ["group-a", "group-b"]

        async def reconcile(identity):
            reconciled.append(identity)
            if identity == "group-b" and fail:
                if cancel:
                    raise asyncio.CancelledError
                raise RuntimeError("lost acknowledgement")

        monkeypatch.setattr(store, "list_task_group_reconciliation_candidates", candidates)
        monkeypatch.setattr(store, "reconcile_task_group", reconcile)
        first = TaskGroupMaintenance.for_store(store)
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await first.advance(store, now=10)
        fail = False
        await TaskGroupMaintenance.for_store(store).advance(store, now=10)
        assert cursors == [None, "group-a"]
        assert reconciled == ["group-a", "group-b", "group-b"]
        assert first.next_scan_at == 11

    asyncio.run(run())
