"""Memory evidence operations compose with domain state and one caller-owned lock."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests._session_provenance import fixture_session_invocation
from tests.core.session_store_conformance.support import _recall_receipt

import cayu
from cayu.memory.evidence import RecallEvidenceConflict, RecallEvidenceQuery
from cayu.sessions import _memory_evidence as evidence
from cayu.sessions.records import Session


def _recall_state():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    session = Session(
        id="session",
        agent_name="assistant",
        provider_name="fake",
        model="fake",
        causal_budget_id="session",
        invocation=fixture_session_invocation("session"),
        created_at=now,
        updated_at=now,
        last_activity_at=now,
    )
    return evidence.MemoryRecallState(
        recall_receipt_ids_by_session={},
        recall_receipt_page_keys_by_interaction={},
        recall_receipt_page_keys_by_interaction_step={},
        recall_receipt_page_keys_by_session={},
        recall_receipt_page_keys_by_step={},
        recall_receipts={},
        session_closure_progress={},
        sessions={session.id: session},
    )


def test_memory_evidence_imports_without_store_implementations():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import sys
class RejectStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"cayu.sessions.base", "cayu.sessions.store", "cayu.sessions.memory",
                        "cayu.storage.sqlite", "cayu.storage.postgres"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectStores())
owner = importlib.import_module("cayu.sessions._memory_evidence")
assert callable(owner.create_recall_receipt)
assert callable(owner.list_recall_receipts)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_recall_receipts_compose_independently_with_atomic_indexes():
    async def scenario():
        state = _recall_state()
        lock = asyncio.Lock()
        entrances = 0

        @asynccontextmanager
        async def locked():
            nonlocal entrances
            async with lock:
                entrances += 1
                yield state

        first = _recall_receipt("a", session_id="session")
        second = _recall_receipt("b", session_id="session")
        returned = await evidence.create_recall_receipt(locked, first)
        assert entrances == 1 and not lock.locked()
        assert returned == first and returned is not first
        assert (
            returned.items[0].identity
            is not state.recall_receipts[first.receipt_id].items[0].identity
        )
        object.__setattr__(returned.items[0].identity, "record_id", "changed-output")
        assert await evidence.load_recall_receipt(locked, "session", first.receipt_id) == first
        await evidence.create_recall_receipt(locked, second)
        for filters in (
            {},
            {"interaction_id": first.interaction_id},
            {"model_step_id": first.model_step_id},
            {"interaction_id": first.interaction_id, "model_step_id": first.model_step_id},
        ):
            query = RecallEvidenceQuery(session_id="session", limit=1, **filters)
            page = await evidence.list_recall_receipts(locked, query)
            assert page.items == (first,) and page.truncated and page.next_cursor
            last = await evidence.list_recall_receipts(
                locked, query.model_copy(update={"cursor": page.next_cursor})
            )
            assert last.items == (second,) and not last.truncated and last.next_cursor is None
        assert await evidence.load_recall_receipt(locked, "other", first.receipt_id) is None
        state.session_closure_progress[("session", "closure")] = {
            "root_session_id": "session",
            "plan_id": "closure",
            "descendants": [],
        }
        before = deepcopy(state)
        assert await evidence.create_recall_receipt(locked, first) == first
        with pytest.raises(RecallEvidenceConflict):
            await evidence.create_recall_receipt(
                locked, first.model_copy(update={"engine_version": "different"})
            )
        with pytest.raises(ValueError, match="unfinished recursive closure"):
            await evidence.create_recall_receipt(
                locked, _recall_receipt("blocked", session_id="session")
            )
        assert state == before and not lock.locked()

    asyncio.run(scenario())


@pytest.mark.parametrize("write", [False, True])
def test_recall_state_resolves_after_lock_acquisition(write):
    from cayu.sessions.memory import InMemorySessionStore

    async def scenario():
        store = InMemorySessionStore()
        state = _recall_state()
        receipt = _recall_receipt("current", session_id="session")
        old_receipts = store._recall_receipts
        await store._lock.acquire()
        operation = (
            store.create_recall_receipt(receipt)
            if write
            else store.load_recall_receipt("session", receipt.receipt_id)
        )
        pending = asyncio.create_task(operation)
        try:
            await asyncio.sleep(0)
            assert not pending.done()
            if not write:
                state.recall_receipts[receipt.receipt_id] = receipt
            for name, value in zip(state._fields, state, strict=True):
                setattr(store, "_" + name, value)
        finally:
            store._lock.release()
        assert await pending == receipt
        assert old_receipts == {} and state.recall_receipts[receipt.receipt_id] == receipt
        if write:
            page = await store.list_recall_receipts(RecallEvidenceQuery(session_id="session"))
            assert page.items == (receipt,)

    asyncio.run(scenario())


@pytest.mark.parametrize("write", [False, True])
def test_cancelled_recall_waiter_leaves_state_and_lock_usable(write):
    async def scenario():
        state = _recall_state()
        before = deepcopy(state)
        lock = asyncio.Lock()
        started = asyncio.Event()

        @asynccontextmanager
        async def locked():
            started.set()
            async with lock:
                yield state

        receipt = _recall_receipt("cancelled", session_id="session")
        await lock.acquire()
        operation = (
            evidence.create_recall_receipt(locked, receipt)
            if write
            else evidence.load_recall_receipt(locked, "session", receipt.receipt_id)
        )
        pending = asyncio.create_task(operation)
        try:
            await started.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert lock.locked() and state == before
        finally:
            lock.release()
        assert await evidence.create_recall_receipt(locked, receipt) == receipt
        assert not lock.locked()

    asyncio.run(scenario())
