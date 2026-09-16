"""Actual owner loss around decisive group publication, followed by fresh recovery."""

from __future__ import annotations

import asyncio
import sys

import pytest
from tests.core.test_task_groups import group_request

from cayu import CayuApp
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore

pytestmark = pytest.mark.process

_CHILD = r"""
import asyncio
import sys
from cayu import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.storage.postgres import PostgresTaskStore
from cayu.tasks.groups import TaskGroupEventType, TaskGroupStatus

async def main():
    backend, address, group_id, task_id, outcome, mode = sys.argv[1:]
    store = SQLiteTaskStore(address) if backend == "sqlite" else PostgresTaskStore(address)
    try:
        app = CayuApp(task_store=store, enable_logging=False)
        operation = store.complete_task if outcome == "success" else store.fail_task
        if mode == "before":
            print("boundary", flush=True)
            await asyncio.Future()
        elif mode == "after":
            await operation(task_id, {"code": "test"})
            print("boundary", flush=True)
            await asyncio.Future()
        else:
            snapshot = await app.load_task_group(group_id)
            if snapshot.status is TaskGroupStatus.PENDING:
                await operation(task_id, {"code": "test"})
            snapshot = await app.load_task_group(group_id)
            expected = TaskGroupStatus.SUCCEEDED if outcome == "success" else TaskGroupStatus.FAILED
            assert snapshot.status is expected
            events = await app.list_task_group_events(group_id)
            kind = TaskGroupEventType.SUCCEEDED if outcome == "success" else TaskGroupEventType.FAILED
            assert sum(e.type is kind for e in events) == 1
            assert sum(e.type is TaskGroupEventType.MEMBER_TERMINAL and e.task_id == task_id for e in events) == 1
            assert [e.sequence for e in events] == list(range(1, len(events) + 1))
            print("recovered", flush=True)
    finally:
        await store.close()

asyncio.run(main())
"""


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("policy", ["all", "first_success", "quorum"])
@pytest.mark.parametrize("outcome", ["success", "failure"])
@pytest.mark.parametrize("phase", ["before", "after"])
def test_group_recovers_decisive_outcome_from_fresh_process(
    backend, policy, outcome, phase, tmp_path, request
):
    address = (
        str(tmp_path / "group.sqlite")
        if backend == "sqlite"
        else request.getfixturevalue("postgres_dsn")
    )
    group = group_request(policy)

    async def spawn(mode):
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            _CHILD,
            backend,
            address,
            group.group_id,
            decisive,
            outcome,
            mode,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def run():
        nonlocal decisive
        store = (
            SQLiteTaskStore(address)
            if backend == "sqlite"
            else PostgresTaskStore(address, schema_mode=SchemaMode.CREATE)
        )
        try:
            await CayuApp(task_store=store, enable_logging=False).create_task_group(group)
            required = group.policy.required_successes(len(group.member_task_ids))
            prior_count = (
                required - 1 if outcome == "success" else len(group.member_task_ids) - required
            )
            operation = store.complete_task if outcome == "success" else store.fail_task
            for identity in group.member_task_ids[:prior_count]:
                await operation(identity, {"code": "test"})
            decisive = group.member_task_ids[prior_count]
        finally:
            await store.close()
        child = await spawn(phase)
        try:
            assert child.stdout is not None
            assert await asyncio.wait_for(child.stdout.readline(), 30) == b"boundary\n"
            child.kill()
            await asyncio.wait_for(child.wait(), 10)
            assert child.returncode < 0
        finally:
            if child.returncode is None:
                child.kill()
                await asyncio.wait_for(child.wait(), 10)
        recovery = await spawn("recover")
        try:
            stdout, stderr = await asyncio.wait_for(recovery.communicate(), 30)
            assert recovery.returncode == 0, stderr.decode()
            assert stdout == b"recovered\n"
        finally:
            if recovery.returncode is None:
                recovery.kill()
                await asyncio.wait_for(recovery.wait(), 10)

    decisive = ""
    asyncio.run(run())
