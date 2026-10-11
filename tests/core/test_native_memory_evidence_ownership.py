"""Native memory evidence composes independently and rolls back partial writes."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from tests.core.session_store_conformance.support import (
    _planned_context_exposure,
    _recall_receipt,
)

import cayu
from cayu.memory.evidence import ContextExposureTransitionRequest, RecallEvidenceQuery
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest

WRITES = {"create_recall_receipt", "create_context_exposure", "transition_context_exposure"}
READS = {
    "load_recall_receipt",
    "list_recall_receipts",
    "load_context_exposure",
    "load_recall_item_exposures",
    "list_context_exposures",
}
TABLES = ("cayu_recall_receipts", "cayu_context_exposures", "cayu_recall_item_exposures")


def assert_owner_imports(backend):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import sys
class RejectAdapters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"cayu.storage.sqlite", "cayu.storage.postgres", "psycopg", "psycopg_pool"}:
            raise AssertionError(f"Unexpected dependency: {fullname}")
sys.meta_path.insert(0, RejectAdapters())
owner = importlib.import_module(f"cayu.storage._{sys.argv[1]}_memory_evidence")
for name in sys.argv[2:]:
    assert callable(getattr(owner, name))
""",
            backend,
            *sorted(WRITES | READS),
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sqlite_memory_evidence_imports_without_adapters_or_driver():
    assert_owner_imports("sqlite")


async def seed(store):
    session_id = f"evidence-{uuid4().hex}"
    await store.create(
        RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    return session_id


async def exercise(ops, session_id, snapshot, fault):
    async def rollback_then_retry(operation, statement):
        before = await snapshot()
        fault.statement = statement
        try:
            with pytest.raises(fault.error):
                await operation()
        finally:
            fault.statement = None
        assert await snapshot() == before
        result = await operation()
        committed = await snapshot()
        assert await operation() == result
        assert await snapshot() == committed
        return result

    receipts = tuple(_recall_receipt(f"{session_id}-{i}", session_id=session_id) for i in range(2))
    for receipt in receipts:
        assert (
            await rollback_then_retry(
                partial(ops.create_recall_receipt, receipt), "INSERT INTO cayu_recall_receipts "
            )
            == receipt
        )
        assert await ops.load_recall_receipt(session_id, receipt.receipt_id) == receipt
    assert await ops.load_recall_receipt("another-session", receipts[0].receipt_id) is None

    expected_exposures = []
    for index, table in enumerate(("cayu_context_exposures", "cayu_recall_item_exposures")):
        exposure, items = _planned_context_exposure(f"{session_id}-{index}", receipts[index])
        assert (
            await rollback_then_retry(
                partial(ops.create_context_exposure, exposure, items), f"INSERT INTO {table} "
            )
            == exposure
        )
        assert await ops.load_recall_item_exposures(session_id, exposure.exposure_id) == items
        transition = ContextExposureTransitionRequest(
            transition_id=f"prepared-{exposure.exposure_id}",
            expected_state="planned",
            expected_revision=0,
            state="prepared",
            occurred_at=exposure.created_at + timedelta(seconds=1),
            evidence_kind="request_prepared",
            evidence_ref="prepared-request",
        )
        updated = await rollback_then_retry(
            partial(ops.transition_context_exposure, session_id, exposure.exposure_id, transition),
            "UPDATE cayu_context_exposures ",
        )
        assert updated.state.value == "prepared" and updated.state_revision == 1
        assert await ops.load_context_exposure(session_id, exposure.exposure_id) == updated
        expected_exposures.append((updated, items))

    for listing, expected in (
        (ops.list_recall_receipts, receipts),
        (ops.list_context_exposures, tuple(pair[0] for pair in expected_exposures)),
    ):
        first = await listing(RecallEvidenceQuery(session_id=session_id, limit=1))
        assert first.items == (expected[0],) and first.truncated and first.next_cursor
        second = await listing(
            RecallEvidenceQuery(session_id=session_id, limit=1, cursor=first.next_cursor)
        )
        assert second.items == (expected[1],) and not second.truncated
        assert second.next_cursor is None
    return receipts, expected_exposures


async def assert_persisted(store, session_id, receipts, exposures):
    for receipt in receipts:
        assert await store.load_recall_receipt(session_id, receipt.receipt_id) == receipt
    for exposure, items in exposures:
        assert await store.load_context_exposure(session_id, exposure.exposure_id) == exposure
        assert await store.load_recall_item_exposures(session_id, exposure.exposure_id) == items
    snapshot = await store.load_session_closure_records(
        session_id, max_records=100, max_bytes=100_000
    )
    assert snapshot["counts"]["recall_receipts"] == 2
    assert snapshot["counts"]["context_exposures"] == 2


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_sqlite_memory_evidence_composes_with_native_runners(tmp_path, error):
    from cayu.storage import _sqlite_memory_evidence as owner
    from cayu.storage._sqlite_functions import _register_sqlite_functions
    from cayu.storage.sqlite import SQLiteSessionStore

    async def run():
        path = tmp_path / "evidence.sqlite"
        store = SQLiteSessionStore(path)
        try:
            session_id = await seed(store)
        finally:
            await store.close()
        del store
        lock = asyncio.Lock()
        fault = SimpleNamespace(statement=None, error=error)

        class Connection(sqlite3.Connection):
            def execute(self, query, parameters=()):
                result = super().execute(query, parameters)
                self.fail_after_write(query)
                return result

            def executemany(self, query, parameters):
                result = super().executemany(query, parameters)
                self.fail_after_write(query)
                return result

            def fail_after_write(self, query):
                if fault.statement and " ".join(query.split()).startswith(fault.statement):
                    assert lock.locked() and self.in_transaction
                    raise fault.error("injected evidence write failure")

        connection = sqlite3.connect(path, factory=Connection)
        connection.row_factory = sqlite3.Row
        _register_sqlite_functions(connection)
        connection.execute("PRAGMA foreign_keys = ON")

        async def execute(operation):
            async with lock:
                assert not connection.in_transaction
                try:
                    return operation(connection)
                finally:
                    assert not connection.in_transaction

        def closure_owners(targets, *, connection):
            assert lock.locked() and connection.in_transaction
            return SQLiteSessionStore._closure_lineage_owners_unlocked(
                None, targets, connection=connection
            )

        ops = SimpleNamespace(
            **{name: partial(getattr(owner, name), execute) for name in READS},
            **{
                name: partial(getattr(owner, name), execute, closure_owners=closure_owners)
                for name in WRITES
            },
        )

        async def snapshot():
            assert not lock.locked() and not connection.in_transaction
            return tuple(
                tuple(
                    tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1, 2")
                )
                for table in TABLES
            )

        try:
            receipts, exposures = await exercise(ops, session_id, snapshot, fault)
        finally:
            connection.close()
        reopened = SQLiteSessionStore(path)
        try:
            await assert_persisted(reopened, session_id, receipts, exposures)
        finally:
            await reopened.close()

    asyncio.run(run())
