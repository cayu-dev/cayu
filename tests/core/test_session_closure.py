from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from cayu.runtime.session_closure import (
    RetainedSessionClosureStore,
    SessionClosureCoordinator,
    SessionClosureDisposition,
    SessionClosurePolicy,
    TaskSessionClosureStore,
)
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.base import InMemoryTaskStore, TaskCreate


class _SessionStore:
    supports_session_closure_receipts = True

    def __init__(self, *, status: str = "completed") -> None:
        self.session = SimpleNamespace(status=SimpleNamespace(value=status))
        self.deleted = False
        self.receipts = {}

    async def load(self, _session_id: str):
        return None if self.deleted else self.session

    async def delete_session(self, _session_id: str, *, closure_receipt=None) -> None:
        self.deleted = True
        if closure_receipt is not None:
            self.receipts[(_session_id, closure_receipt["plan_id"])] = closure_receipt

    async def load_session_closure_receipt(self, session_id: str, plan_id: str):
        return self.receipts.get((session_id, plan_id))

    async def list_sessions(self, query):
        return []


class _DependentStore:
    store_id = "dependent"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.erased = False

    async def inspect_session_closure(self, _session_id: str, *, policy):
        return {
            "store_id": self.store_id,
            "record_class": "records",
            "disposition": SessionClosureDisposition.OWNED_ELIGIBLE,
            "count": 1,
        }

    async def erase_session_closure(self, _session_id: str, *, policy, plan_id: str):
        if self.fail:
            raise RuntimeError("dependent store unavailable")
        self.erased = True
        return {
            "store_id": self.store_id,
            "record_class": "records",
            "disposition": SessionClosureDisposition.ERASED,
            "count": 1,
        }


def test_closure_inspection_reports_unsupported_capability() -> None:
    class Unsupported:
        store_id = "unsupported"

    async def run() -> None:
        manifest = await SessionClosureCoordinator(
            _SessionStore(), dependent_stores=(Unsupported(),)
        ).inspect("session")
        record = manifest.records[-1]
        assert record.disposition is SessionClosureDisposition.UNSUPPORTED
        assert manifest.complete is False

    asyncio.run(run())


def test_closure_erases_dependents_before_session_store() -> None:
    async def run() -> None:
        session_store = _SessionStore()
        dependent = _DependentStore()
        report = await SessionClosureCoordinator(
            session_store, dependent_stores=(dependent,)
        ).erase("session")
        assert report.complete is True
        assert dependent.erased is True
        assert session_store.deleted is True
        assert all(
            record.disposition is SessionClosureDisposition.ERASED
            for record in report.manifest.records
        )

    asyncio.run(run())


def test_closure_failure_is_partial_and_does_not_delete_session() -> None:
    async def run() -> None:
        session_store = _SessionStore()
        report = await SessionClosureCoordinator(
            session_store, dependent_stores=(_DependentStore(fail=True),)
        ).erase("session")
        assert report.complete is False
        assert session_store.deleted is False
        assert report.error is not None

    asyncio.run(run())


def test_closure_rejects_nonterminal_sessions() -> None:
    async def run() -> None:
        with pytest.raises(ValueError, match="terminal"):
            await SessionClosureCoordinator(_SessionStore(status="running")).erase("session")

    asyncio.run(run())


def test_closure_export_is_bounded_and_marks_export_operation() -> None:
    async def run() -> None:
        export = await SessionClosureCoordinator(_SessionStore()).export("session")
        assert export.manifest.operation.value == "export"
        payload = export.to_bytes()
        assert b"session" in payload
        assert b"content_redacted" in payload

    asyncio.run(run())


def test_closure_rejects_plan_identity_conflict_before_deletion() -> None:
    async def run() -> None:
        session_store = _SessionStore()
        with pytest.raises(ValueError, match="plan identity"):
            await SessionClosureCoordinator(session_store).erase(
                "session", expected_plan_id="wrong"
            )
        assert session_store.deleted is False

    asyncio.run(run())


def test_closure_does_not_relabel_retained_records_as_erased() -> None:
    async def run() -> None:
        report = await SessionClosureCoordinator(
            _SessionStore(),
            dependent_stores=(RetainedSessionClosureStore("budget-store"),),
        ).erase("session")
        assert report.complete is True
        assert report.manifest.records[-1].disposition is SessionClosureDisposition.RETAINED

    asyncio.run(run())


def test_closure_retries_same_plan_after_session_deletion() -> None:
    async def run() -> None:
        coordinator = SessionClosureCoordinator(_SessionStore())
        first = await coordinator.erase("session")
        second = await coordinator.erase("session", expected_plan_id=first.plan_id)
        assert first.complete is True
        assert second.complete is True
        assert second.already_absent is True

    asyncio.run(run())


def test_in_memory_task_closure_removes_terminal_session_tasks() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        await store.create_task(TaskCreate(task_id="task", type="job", session_id="session"))
        await store.complete_task("task", {})
        adapter = TaskSessionClosureStore(store)
        policy = SessionClosurePolicy()
        record = await adapter.inspect_session_closure("session", policy=policy)
        assert record.disposition is SessionClosureDisposition.OWNED_ELIGIBLE
        await adapter.erase_session_closure("session", policy=policy, plan_id="plan")
        assert await store.load_task("task") is None

    asyncio.run(run())


def test_sqlite_task_closure_removes_terminal_session_tasks(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteTaskStore(tmp_path / "closure.sqlite")
        try:
            await store.create_task(
                TaskCreate(task_id="sqlite-task", type="job", session_id="session")
            )
            await store.complete_task("sqlite-task", {})
            adapter = TaskSessionClosureStore(store)
            policy = SessionClosurePolicy()
            record = await adapter.inspect_session_closure("session", policy=policy)
            assert record.disposition is SessionClosureDisposition.OWNED_ELIGIBLE
            await adapter.erase_session_closure("session", policy=policy, plan_id="plan")
            assert await store.load_task("sqlite-task") is None
        finally:
            await store.close()

    asyncio.run(run())
