"""Persistent fence lookup failures must not prevent joining a cancelled dispatch."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from cayu.tasks.creation import TaskCreate
from cayu.tasks.dispatch import (
    TaskStoreDispatcher,
    _dispatch_exception_graph_contains_identity,
    _DispatchLeaseAuthority,
)
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskClaimLost


@pytest.mark.parametrize(
    ("failure_type", "expected_attempts"),
    [
        (ConnectionError, 3),
        (PermissionError, 3),
        (ValueError, 3),
        (sqlite3.ProgrammingError, 3),
        (TaskClaimLost, 1),
        (asyncio.CancelledError, 1),
        (lambda: BaseExceptionGroup("control", [SystemExit()]), 1),
    ],
)
def test_fence_lookup_failure_still_joins_operation(monkeypatch, failure_type, expected_attempts):
    async def scenario():
        store = InMemoryTaskStore()
        await store.create_task(TaskCreate(task_id="draining", type="cayu.dispatch"))
        claimed = await store.claim_task("owner", lease_seconds=60)
        assert claimed is not None
        dispatcher = TaskStoreDispatcher(store)
        started = asyncio.Event()
        release = asyncio.Event()
        operation_failure = RuntimeError("operation failed after natural drain")
        failures = []

        async def operation():
            started.set()
            await release.wait()
            raise operation_failure

        async def fail_lookup(_identity):
            failure = failure_type()
            failures.append(failure)
            raise failure

        monkeypatch.setattr(store, "load_task", fail_lookup)
        operation_task = asyncio.create_task(operation())
        drain = asyncio.create_task(
            dispatcher._drain_dispatch_operation_under_cancellation_fence(
                operation_task,
                task=claimed,
                worker_id="owner",
                lease_authority=_DispatchLeaseAuthority(
                    claimed.lease_expires_at, claimed.claim_generation
                ),
                error={"code": "shutdown"},
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 5)
            done, _ = await asyncio.wait({drain}, timeout=0.4)
            assert not done  # Exhausting publication retries is not proof of quiescence.
            assert len(failures) == expected_attempts
            assert not operation_task.done()
            release.set()
            done, _ = await asyncio.wait({drain}, timeout=5)
            assert drain in done
            outcome = drain.result()
            assert operation_task.done()
            assert outcome.operation_failure is operation_failure
            assert outcome.settlement_failure is None
            assert outcome.fence_failure is not None
            for failure in failures:
                assert _dispatch_exception_graph_contains_identity(outcome.fence_failure, failure)
        finally:
            release.set()
            drain.cancel()
            await asyncio.gather(drain, operation_task, return_exceptions=True)

    asyncio.run(scenario())


def test_cancellation_during_fence_retry_delay_still_joins_operation(monkeypatch):
    async def scenario():
        store = InMemoryTaskStore()
        await store.create_task(TaskCreate(task_id="draining", type="cayu.dispatch"))
        claimed = await store.claim_task("owner", lease_seconds=60)
        assert claimed is not None
        dispatcher = TaskStoreDispatcher(store)
        release = asyncio.Event()
        delaying = asyncio.Event()
        lookup_failure = ConnectionError("unavailable")
        attempts = 0
        original_sleep = asyncio.sleep

        async def operation():
            await release.wait()
            raise ValueError("natural operation failure")

        async def fail_lookup(_identity):
            nonlocal attempts
            attempts += 1
            raise lookup_failure

        async def delay(seconds):
            if seconds == 0:
                return await original_sleep(0)
            delaying.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(store, "load_task", fail_lookup)
        monkeypatch.setattr("cayu.tasks.dispatch.asyncio.sleep", delay)
        operation_task = asyncio.create_task(operation())
        drain = asyncio.create_task(
            dispatcher._drain_dispatch_operation_under_cancellation_fence(
                operation_task,
                task=claimed,
                worker_id="owner",
                lease_authority=_DispatchLeaseAuthority(
                    claimed.lease_expires_at, claimed.claim_generation
                ),
                error={"code": "shutdown"},
            )
        )
        try:
            await asyncio.wait_for(delaying.wait(), 5)
            drain.cancel("cancelled during retry delay")
            release.set()
            done, _ = await asyncio.wait({drain}, timeout=5)
            assert drain in done
            outcome = drain.result()
            assert attempts == 1
            assert operation_task.done()
            assert isinstance(outcome.operation_failure, ValueError)
            assert isinstance(outcome.fence_failure, RuntimeError)
            assert _dispatch_exception_graph_contains_identity(
                outcome.fence_failure, lookup_failure
            )
        finally:
            release.set()
            drain.cancel()
            await asyncio.gather(drain, operation_task, return_exceptions=True)

    asyncio.run(scenario())
