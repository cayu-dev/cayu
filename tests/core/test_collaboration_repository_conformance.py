"""Direct repository conformance for every native collaboration store.

Each case drives the transaction repository the runtime itself uses and asserts
the same observable results on memory, SQLite and PostgreSQL. Records come from
the public collaboration API or the existing typed record fixtures; no model
steps or runtime scenarios run here.
"""

import asyncio
import json
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from tests.core.test_clarification_contracts import fixture as clarification_fixture
from tests.core.test_clarification_deliveries import delivery_record, settled_record
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_participant_identity import app, create, registration
from tests.core.test_request_planning_records import _records as planning_records

from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import CollaborationContractError
from cayu.collaboration._planning_records import (
    RequestPlanningCursor,
    RequestPlanningRecord,
    RequestPlanningStageIntent,
    RequestPlanningStageRecord,
)
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration._request_receipts import record_operation
from cayu.collaboration.clarifications import ClarificationDueCursor
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.collaboration.requests import RequestAdmissionCommand, RequestControl
from cayu.collaboration.waits import CollaborationWait
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode
from cayu.vaults.redaction import SecretRedactor

pytestmark = [pytest.mark.anyio, pytest.mark.store_semantics]
REDACTOR = SecretRedactor()
EVERYTHING_DUE = 2**53 - 1


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _truncate_postgres(dsn):
    import psycopg
    from psycopg import sql

    # postgres_dsn is module-scoped; fixed-scope record fixtures need a clean
    # database per case.
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as connection:
        cursor = await connection.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = current_schema() "
            "AND tablename LIKE 'cayu\\_collaboration\\_%'"
        )
        tables = [row[0] for row in await cursor.fetchall()]
        if tables:
            await connection.execute(
                sql.SQL("TRUNCATE {}").format(sql.SQL(", ").join(map(sql.Identifier, tables)))
            )


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def stores(request, tmp_path):
    backend = request.param
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None
    opened = []

    def factory():
        if backend == "memory":
            if opened:
                return opened[0]
            value = InMemoryCollaborationStore()
        elif backend == "sqlite":
            value = SQLiteCollaborationStore(tmp_path / "collaboration.sqlite")
        else:
            value = PostgresCollaborationStore(dsn, schema_mode=SchemaMode.CREATE)
        opened.append(value)
        return value

    factory.backend = backend
    try:
        if dsn is not None:
            await factory()._ensure_ready()
            await _truncate_postgres(dsn)
        yield factory
    finally:
        for value in opened:
            await value.close()


def canonical(value):
    # Distinguishes True from 1 and 1.0 from 1, which == does not.
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def nested(depth):
    value = {"leaf": ["ä", 1, None]}
    for level in range(depth):
        value = {f"level-{level}": [value, level]}
    return value


DOCUMENT = {
    "text": 'Gr\u00fc\u00dfe, \u4e16\u754c \u2014 \u05e2\u05d1\u05e8\u05d9\u05ea e\u0301 "quoted" back\\slash\ttab\nline \u0001 \u2028 \uffff',
    "astral": "\U0001f600\U00010348\U0001f469\U0001f3fd\u200d\U0001f4bb",
    "integers": [0, -1, 1, 2**31, 2**32 + 1, 2**53 - 1, 2**53 + 1, 2**63 - 1, -(2**63)],
    "floats": [0.1, -2.5, 1e-7, 1.5e-300, -0.0, 1.0],
    "flags": [True, False, None, 1, 0],
    "empty": {"object": {}, "array": [], "string": ""},
    "keys": {"\u043a\u043b\u044e\u0447": 1, "\U0001f511": 2, "": 3, "a.b": 4},
    "nested": nested(20),
}
# Near the 64 KiB envelope: substr() bounds characters, the decoder bounds bytes.
LARGE_DOCUMENT = {"text": "\u754c" * 21_000}
GENERIC_FAMILIES = (
    ("anchors", ()),
    ("aliases", ("alias-\u00fc",)),
    ("configurations", ("participant", 2**53 - 1)),
    ("namespaces", ("namespace", 1)),
)


async def test_generic_documents_round_trip_exactly(stores):
    store = stores()
    scope = uuid4().hex
    async with store._transaction(scope, write=True) as tx:
        for family, key in GENERIC_FAMILIES:
            await tx.put(family, key, DOCUMENT, insert=True)
            assert canonical(await tx.get(family, key)) == canonical(DOCUMENT)
        await tx.put("aliases", ("large",), LARGE_DOCUMENT, insert=True)
    for reader in (store, stores()):
        async with reader._transaction(scope, write=False) as tx:
            for family, key in GENERIC_FAMILIES:
                assert canonical(await tx.get(family, key)) == canonical(DOCUMENT)
            assert await tx.get("aliases", ("large",)) == LARGE_DOCUMENT


@pytest.mark.parametrize(
    "value",
    [
        {"text": "nul\x00byte"},
        {"text": "lone \ud800 surrogate"},
        {"number": 2**63},
        {"number": -(2**63) - 1},
        {"number": float("nan")},
        {"number": float("inf")},
        {"entries": {str(index): index for index in range(65)}},
        {"entries": list(range(65))},
        nested(70),
        {"text": "\u754c" * 22_000},
        {1: "integer key"},
        {"value": {1, 2}},
    ],
    ids=[
        "nul",
        "surrogate",
        "integer-above",
        "integer-below",
        "nan",
        "infinity",
        "object-entries",
        "array-entries",
        "depth",
        "envelope",
        "key-type",
        "set",
    ],
)
async def test_unrepresentable_documents_reject_before_any_write(stores, value):
    store = stores()
    scope = uuid4().hex
    with pytest.raises(ValueError):
        async with store._transaction(scope, write=True) as tx:
            await tx.put("aliases", ("kept",), {"kept": True}, insert=True)
            await tx.put("aliases", ("rejected",), value, insert=True)
    async with store._transaction(scope, write=False) as tx:
        assert await tx.get("aliases", ("kept",)) is None
        assert await tx.get("aliases", ("rejected",)) is None


async def test_insert_update_delete_and_scope_isolation(stores):
    store = stores()
    scope, other = uuid4().hex, uuid4().hex
    async with store._transaction(scope, write=True) as tx:
        assert await tx.get("aliases", ("a",)) is None
        await tx.put("aliases", ("a",), {"version": 1}, insert=True)
        await tx.put("aliases", ("a",), {"version": 2}, insert=False)
        await tx.put("aliases", ("b",), {"version": 1}, insert=False)
        await tx.put("configurations", ("p", 1), {"revision": 1}, insert=True)
        await tx.put("configurations", ("p", 2), {"revision": 2}, insert=True)
        await tx.put("configurations", ("q", 1), {"revision": 3}, insert=True)
        assert await tx.get("aliases", ("a",)) == {"version": 2}
    async with store._transaction(other, write=True) as tx:
        await tx.put("aliases", ("a",), {"version": "other"}, insert=True)
    async with stores()._transaction(scope, write=True) as tx:
        assert await tx.get("aliases", ("a",)) == {"version": 2}
        assert await tx.get("aliases", ("b",)) == {"version": 1}
        assert [
            await tx.get("configurations", key) for key in (("p", 1), ("p", 2), ("q", 1), ("q", 2))
        ] == [{"revision": 1}, {"revision": 2}, {"revision": 3}, None]
        await tx.delete("aliases", ("a",))
        await tx.delete("aliases", ("missing",))
        await tx.delete("configurations", ("p", 1))
    async with store._transaction(scope, write=False) as tx:
        assert await tx.get("aliases", ("a",)) is None
        assert await tx.get("aliases", ("b",)) == {"version": 1}
        assert await tx.get("configurations", ("p", 1)) is None
        assert await tx.get("configurations", ("p", 2)) == {"revision": 2}
    async with store._transaction(other, write=False) as tx:
        assert await tx.get("aliases", ("a",)) == {"version": "other"}
        assert await tx.get("aliases", ("b",)) is None


async def test_duplicate_insert_fails_and_rolls_back_whole_transaction(stores):
    store = stores()
    scope = uuid4().hex
    async with store._transaction(scope, write=True) as tx:
        await tx.put("aliases", ("a",), {"version": 1}, insert=True)
    # Each backend reports its own conflict type; the runtime only relies on
    # failure plus complete rollback.
    with pytest.raises(Exception) as caught:
        async with store._transaction(scope, write=True) as tx:
            await tx.put("aliases", ("b",), {"version": 1}, insert=True)
            await tx.put("aliases", ("a",), {"version": 2}, insert=True)
    assert not isinstance(caught.value, CollaborationContractError)
    async with store._transaction(scope, write=True) as tx:
        assert await tx.get("aliases", ("a",)) == {"version": 1}
        assert await tx.get("aliases", ("b",)) is None
        await tx.put("aliases", ("c",), {"version": 1}, insert=True)
    async with stores()._transaction(scope, write=False) as tx:
        assert await tx.get("aliases", ("c",)) == {"version": 1}


async def test_failed_and_cancelled_transactions_roll_back(stores):
    store = stores()
    scope = uuid4().hex
    async with store._transaction(scope, write=True) as tx:
        await tx.put("aliases", ("a",), {"version": 1}, insert=True)
        await tx.put("aliases", ("b",), {"version": 1}, insert=True)

    async def mutate(tx):
        await tx.put("aliases", ("a",), {"version": 2}, insert=False)
        await tx.delete("aliases", ("b",))
        await tx.put("aliases", ("c",), {"version": 1}, insert=True)
        assert await tx.get("aliases", ("c",)) == {"version": 1}

    with pytest.raises(RuntimeError, match="abort"):
        async with store._transaction(scope, write=True) as tx:
            await mutate(tx)
            raise RuntimeError("abort")
    mutated = asyncio.Event()

    async def cancelled():
        async with store._transaction(scope, write=True) as tx:
            await mutate(tx)
            mutated.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(cancelled())
    await asyncio.wait_for(mutated.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for reader in (store, stores()):
        async with reader._transaction(scope, write=False) as tx:
            assert await tx.get("aliases", ("a",)) == {"version": 1}
            assert await tx.get("aliases", ("b",)) == {"version": 1}
            assert await tx.get("aliases", ("c",)) is None


async def test_same_scope_transactions_serialize_behind_a_writer(stores):
    store = stores()
    scope = uuid4().hex
    held, release = asyncio.Event(), asyncio.Event()
    entered = []

    async def holder():
        async with store._transaction(scope, write=True) as tx:
            await tx.put("aliases", ("lock",), {"owner": "first"}, insert=True)
            held.set()
            await release.wait()

    async def contender(write):
        async with store._transaction(scope, write=write) as tx:
            entered.append(write)
            return await tx.get("aliases", ("lock",))

    first = asyncio.create_task(holder())
    await asyncio.wait_for(held.wait(), timeout=5)
    writer = asyncio.create_task(contender(True))
    reader = asyncio.create_task(contender(False))
    try:
        await asyncio.sleep(0.2)
        assert entered == []
    finally:
        release.set()
    await asyncio.wait_for(first, timeout=5)
    assert await asyncio.wait_for(writer, timeout=5) == {"owner": "first"}
    assert await asyncio.wait_for(reader, timeout=5) == {"owner": "first"}
    assert sorted(entered) == [False, True]


async def test_owner_time_is_integer_wall_time_and_non_decreasing(stores):
    store = stores()
    scope = uuid4().hex
    # PostgreSQL samples its own clock; allow for host/database skew.
    skew = 60_000 if stores.backend == "postgres" else 0
    before = time.time_ns() // 1_000_000
    observed = []
    for write in (False, True, False):
        async with store._transaction(scope, write=write) as tx:
            observed.append(await tx.now_ms())
            observed.append(await tx.now_ms())
        await asyncio.sleep(0.002)
    after = time.time_ns() // 1_000_000
    assert all(type(value) is int for value in observed)
    assert observed == sorted(observed)
    assert observed[-1] > observed[0]
    assert before - skew <= observed[0] and observed[-1] <= after + skew


def _page_all(results):
    return [item for page in results for item in page]


async def test_participant_and_event_scans_order_and_paginate(stores):
    store = stores()
    scope = uuid4().hex
    # Same-length lowercase hex sorts identically under every database collation.
    ids = sorted(uuid4().hex for _ in range(5))
    sequences = [1, 2, 9, 10, 11, 100]
    async with store._transaction(scope, write=False) as tx:
        assert await tx.scan("participants", after="", limit=64, allowed=None) == []
        assert await tx.scan("events", after=0, limit=64, allowed=None) == []
        assert await tx.scan_request_events(after=0, limit=64) == []
    async with store._transaction(scope, write=True) as tx:
        for identity in (ids[3], ids[0], ids[4], ids[2], ids[1]):
            await tx.put(
                "participants",
                (identity,),
                {"reference": {"participant_id": identity}, "label": f"\u00fc-{identity}"},
                insert=True,
            )
        for sequence in (10, 1, 100, 9, 2, 11):
            await tx.put("events", (sequence,), {"sequence": sequence}, insert=True)
            await tx.put("request_events", (sequence,), {"sequence": sequence}, insert=True)
    async with stores()._transaction(scope, write=False) as tx:

        async def participants(after, limit, allowed=None):
            rows = await tx.scan("participants", after=after, limit=limit, allowed=allowed)
            return [row["reference"]["participant_id"] for row in rows]

        async def events(after, limit):
            rows = await tx.scan("events", after=after, limit=limit, allowed=None)
            return [row["sequence"] for row in rows]

        async def request_events(after, limit):
            return [
                row["sequence"] for row in await tx.scan_request_events(after=after, limit=limit)
            ]

        assert await participants("", 64) == ids
        assert await participants("", 5) == ids
        assert await participants("", 1) == ids[:1]
        assert await participants(ids[1], 2) == ids[2:4]
        assert await participants(ids[1][:-1], 64) == ids[1:]
        assert await participants(ids[-1], 64) == []
        assert await participants("g", 64) == []
        pages, after = [], ""
        while page := await participants(after, 2):
            pages.append(page)
            after = page[-1]
        assert pages == [ids[:2], ids[2:4], ids[4:]]
        assert await participants("", 64, allowed=()) == []
        assert await participants("", 64, allowed=(ids[3], ids[1], "absent")) == [ids[1], ids[3]]
        assert await participants(ids[1], 64, allowed=(ids[3], ids[1])) == [ids[3]]
        rows = await tx.scan("participants", after="", limit=1, allowed=None)
        assert rows == [{"reference": {"participant_id": ids[0]}, "label": f"\u00fc-{ids[0]}"}]

        for scan in (events, request_events):
            assert await scan(0, 64) == sequences
            assert await scan(0, 6) == sequences
            assert await scan(2, 2) == [9, 10]
            assert await scan(3, 64) == [9, 10, 11, 100]
            assert await scan(9, 1) == [10]
            assert await scan(100, 64) == []
            assert await scan(-1, 1) == [1]
            pages, after = [], 0
            while page := await scan(after, 4):
                pages.append(page)
                after = page[-1]
            assert pages == [sequences[:4], sequences[4:]]


async def _collaboration(store, count):
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    created = []
    for index in range(count):
        _, receipt = await create(application, initialized, f"create-{index}")
        created.append(receipt.participants[0])
    return application, initialized, created


async def test_real_participant_events_operations_and_history_pins(stores):
    store = stores()
    _, initialized, created = await _collaboration(store, 3)
    scope = initialized.binding.application_scope
    ids = [snapshot.reference.participant_id for snapshot in created]
    async with stores()._transaction(scope, write=False) as tx:
        rows = await tx.scan("participants", after="", limit=64, allowed=None)
        assert [row["reference"]["participant_id"] for row in rows] == sorted(ids)
        events = await tx.scan("events", after=0, limit=64, allowed=None)
        sequences = [row["sequence"] for row in events]
        assert sequences == sorted(sequences) and len(sequences) == len(set(sequences)) >= 4
        # Restricted pages contain only events whose every participant is allowed.
        for identity in ids:
            restricted = await tx.scan("events", after=0, limit=64, allowed=(identity,))
            assert restricted
            assert all(
                [ref["participant_id"] for ref in row["participants"]] == [identity]
                for row in restricted
            )
        both = await tx.scan("events", after=0, limit=64, allowed=tuple(ids[:2]))
        assert {row["participants"][0]["participant_id"] for row in both} == set(ids[:2])
        assert await tx.scan("events", after=0, limit=64, allowed=("absent",)) == []
        assert await tx.scan("events", after=0, limit=64, allowed=()) == []
        operations = await tx.scan_operations(initialized.namespace_incarnation, 1, limit=64)
        keys = [record_operation(row, redactor=REDACTOR).caller_key for row in operations]
        assert keys == ["create-0", "create-1", "create-2"]
        limited = await tx.scan_operations(initialized.namespace_incarnation, 1, limit=2)
        assert limited == operations[:2]
        assert await tx.scan_operations(initialized.namespace_incarnation, 2, limit=64) == []
        assert await tx.scan_operations("absent", 1, limit=64) == []
        for snapshot in created:
            identity = snapshot.reference.participant_id
            revision = snapshot.configuration_revision
            assert await tx.history_in_use("configurations", identity, revision)
            assert await tx.history_in_use(
                "lifecycle_history", identity, snapshot.lifecycle_revision
            )
            assert not await tx.history_in_use("configurations", identity, revision + 1)
        assert not await tx.history_in_use("configurations", "absent", 1)
    operation_key = (initialized.namespace_incarnation, 1, "create-0")
    original = None
    with pytest.raises(RuntimeError, match="rollback"):
        async with store._transaction(scope, write=True) as tx:
            original = await tx.get("operations", operation_key)
            await tx.delete("operations", operation_key)
            assert await tx.get("operations", operation_key) is None
            assert not await tx.history_in_use(
                "configurations", ids[0], created[0].configuration_revision
            )
            raise RuntimeError("rollback")
    async with store._transaction(scope, write=False) as tx:
        assert await tx.get("operations", operation_key) == original
        assert await tx.history_in_use("configurations", ids[0], created[0].configuration_revision)


def _roomy_registration():
    reg = registration()
    limits = reg.bootstrap.limits.model_copy(
        update={"retained_bytes": 16 * 1024 * 1024, "operations": 512, "events": 1024}
    )
    return replace(reg, bootstrap=reg.bootstrap.model_copy(update={"limits": limits}))


async def _requests(store, count):
    application, resolver, values = await public_setup(store, reg=_roomy_registration())
    initialized, recipient, request = values[1], values[3], values[4]
    receipts = []
    for index in range(count):
        receipts.append(
            await application.accept_collaboration_request(
                request.model_copy(update={"operation": initialized.operation(f"request-{index}")}),
                context=resolver.context,
            )
        )
    return application, resolver, initialized, recipient, receipts


async def test_request_and_permit_scans_order_filter_and_paginate(stores):
    store = stores()
    application, resolver, initialized, recipient, receipts = await _requests(store, 4)
    scope = initialized.binding.application_scope
    recipient_id = recipient.reference.participant_id
    async with stores()._transaction(scope, write=False) as tx:
        due = await tx.scan_due_requests(after=0, now_ms=EVERYTHING_DUE, limit=64)
        positions = [row["receipt"]["event"]["sequence"] for row in due]
        assert len(positions) == 4 and positions == sorted(set(positions))
        assert [row["receipt"]["expected"]["operation"]["caller_key"] for row in due] == [
            f"request-{index}" for index in range(4)
        ]
        assert all(row["state"] == "open" for row in due)
        assert await tx.scan_due_requests(after=0, now_ms=EVERYTHING_DUE, limit=2) == due[:2]
        assert (
            await tx.scan_due_requests(after=positions[1], now_ms=EVERYTHING_DUE, limit=64)
            == due[2:]
        )
        assert (
            await tx.scan_due_requests(after=positions[-1], now_ms=EVERYTHING_DUE, limit=64) == []
        )
        assert await tx.scan_due_requests(after=0, now_ms=0, limit=64) == []
        first_due = min(row["next_due_at_ms"] for row in due)
        at_boundary = await tx.scan_due_requests(after=0, now_ms=first_due, limit=64)
        assert at_boundary and all(row["next_due_at_ms"] == first_due for row in at_boundary)
        assert await tx.scan_due_requests(after=0, now_ms=first_due - 1, limit=64) == []
        assert (
            await tx.get("requests", (initialized.namespace_incarnation, 1, "request-0")) == due[0]
        )

        permits = await tx.scan_permits(recipient_id, after=0, limit=64, pending_only=False)
        permit_positions = [row["position"] for row in permits]
        assert len(permits) == 4 and permit_positions == sorted(set(permit_positions))
        assert all(row["state"] == "pending" for row in permits)
        assert await tx.scan_permits(recipient_id, after=0, limit=64, pending_only=True) == permits
        assert (
            await tx.scan_permits(recipient_id, after=0, limit=1, pending_only=False) == permits[:1]
        )
        assert (
            await tx.scan_permits(
                recipient_id, after=permit_positions[0], limit=2, pending_only=False
            )
            == permits[1:3]
        )
        assert (
            await tx.scan_permits(
                recipient_id, after=permit_positions[-1], limit=64, pending_only=False
            )
            == []
        )
        assert await tx.scan_permits("absent", after=0, limit=64, pending_only=False) == []
    await application.control_collaboration_request(
        RequestControl(
            operation=initialized.operation("cancel-request-1"),
            expected=receipts[1].expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    async with stores()._transaction(scope, write=False) as tx:
        open_rows = await tx.scan_due_requests(after=0, now_ms=EVERYTHING_DUE, limit=64)
        assert [row["receipt"]["expected"]["operation"]["caller_key"] for row in open_rows] == [
            "request-0",
            "request-2",
            "request-3",
        ]
        cancelled = await tx.get("requests", (initialized.namespace_incarnation, 1, "request-1"))
        assert cancelled["state"] == "cancelled"
        everything = await tx.scan_permits(recipient_id, after=0, limit=64, pending_only=False)
        pending = await tx.scan_permits(recipient_id, after=0, limit=64, pending_only=True)
        assert [row["position"] for row in everything] == permit_positions
        assert len(pending) == 3 and all(row["state"] == "pending" for row in pending)
        assert [row for row in everything if row["state"] != "pending"] == [
            row for row in everything if row not in pending
        ]


def _wait(receipt, initialized, key):
    return CollaborationWait(
        operation=initialized.operation(key),
        source_owner=initialized.owner,
        targets=(receipt.expected,),
        predicate="ALL_SUCCESS",
        predicate_version=1,
        threshold=None,
        deadline=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        failure_policy="settle",
        service_policy="external_observer",
        projection=None,
        wait_edge_revision=1,
        initiator=receipt.expected.initiator,
    )


async def test_wait_scan_uses_codepoint_order_and_tracks_state(stores):
    store = stores()
    application, resolver, initialized, _, receipts = await _requests(store, 1)
    scope = initialized.binding.application_scope
    namespace = initialized.namespace_incarnation
    # Mixed case: byte order differs from common database locales.
    keys = ["wait-\u00e9", "wait-a", "wait-B"]
    waits = {key: _wait(receipts[0], initialized, key) for key in keys}
    for wait in waits.values():
        await application.register_collaboration_wait(wait, context=resolver.context)
    expected = sorted(keys)
    assert expected == ["wait-B", "wait-a", "wait-\u00e9"]

    def caller_keys(rows):
        return [row["registration"]["wait"]["operation"]["caller_key"] for row in rows]

    async with stores()._transaction(scope, write=False) as tx:
        rows = await tx.scan_waits(namespace=namespace, after=None, limit=64)
        assert caller_keys(rows) == expected
        assert all(row["state"] == "pending" for row in rows)
        assert await tx.scan_waits(namespace=namespace, after=None, limit=2) == rows[:2]
        assert (
            caller_keys(await tx.scan_waits(namespace=namespace, after=(1, "wait-Z"), limit=64))
            == expected[1:]
        )
        assert (
            caller_keys(await tx.scan_waits(namespace=namespace, after=(1, "wait-["), limit=64))
            == expected[1:]
        )
        assert await tx.scan_waits(namespace=namespace, after=(1, expected[-1]), limit=64) == []
        assert await tx.scan_waits(namespace=namespace, after=(2, ""), limit=64) == []
        assert caller_keys(await tx.scan_waits(namespace=namespace, after=(0, "~"), limit=64)) == (
            expected
        )
        assert await tx.scan_waits(namespace="absent", after=None, limit=64) == []
        pages, after = [], None
        while page := await tx.scan_waits(namespace=namespace, after=after, limit=2):
            pages.append(caller_keys(page))
            after = (1, pages[-1][-1])
        assert pages == [expected[:2], expected[2:]]
    await application.cancel_collaboration_wait(waits["wait-a"], context=resolver.context)
    async with stores()._transaction(scope, write=False) as tx:
        rows = await tx.scan_waits(namespace=namespace, after=None, limit=64)
        assert caller_keys(rows) == expected
        assert {key: row["state"] for key, row in zip(expected, rows, strict=True)} == {
            key: "cancelled" if key == "wait-a" else "pending" for key in expected
        }


def _question(state, key, deadline):
    operation = state.question.operation.model_copy(update={"caller_key": key})
    record = state.model_copy(
        update={
            "question": state.question.model_copy(
                update={"operation": operation, "deadline_at_ms": deadline}
            )
        }
    )
    return (operation.namespace_incarnation, operation.generation, key), record


async def test_clarification_question_and_delivery_scans(stores):
    store = stores()
    state = clarification_fixture()[0]
    questions = [
        _question(state, key, deadline)
        for key, deadline in (
            ("question-c", 200),
            ("question-a", 300),
            ("question-b", 200),
            ("question-d", 250),
        )
    ]
    closed_key, closed = _question(state, "question-e", 150)
    closed = closed.model_copy(
        update={
            "state": "cancelled",
            "closed_at_ms": 120,
            "closure_operation": closed.question.operation.model_copy(
                update={"caller_key": "close-e"}
            ),
        }
    )
    deliveries = [
        delivery_record(f"delivery-{index}", deadline=deadline)
        for index, deadline in (
            (2, 180),
            (0, 190),
            (1, 180),
        )
    ]
    delivery_keys = [("namespace", 1, row.intent.operation.caller_key) for row in deliveries]

    def question_keys(rows):
        return [row["question"]["operation"]["caller_key"] for row in rows]

    def delivery_keys_of(rows):
        return [row["intent"]["operation"]["caller_key"] for row in rows]

    async with store._transaction("app", write=False) as tx:
        assert await tx.scan_due_clarifications(after=None, now_ms=EVERYTHING_DUE, limit=64) == []
        assert await tx.scan_pending_clarification_deliveries(after=None, limit=64) == []
    async with store._transaction("app", write=True) as tx:
        for key, record in questions:
            await tx.put("clarification_questions", key, record, insert=True)
        await tx.put("clarification_questions", closed_key, closed, insert=True)
        for key, record in zip(delivery_keys, deliveries, strict=True):
            await tx.put("clarification_deliveries", key, record, insert=True)
    async with stores()._transaction("app", write=False) as tx:
        by_request = await tx.scan_clarification_questions(state.question.request, limit=32)
        assert question_keys(by_request) == [f"question-{key}" for key in "abcde"]
        assert (
            await tx.scan_clarification_questions(state.question.request, limit=2)
            == (by_request[:2])
        )
        lineage = await tx.scan_clarification_lineage_questions(state.question.lineage, limit=32)
        assert lineage == by_request
        assert (
            await tx.scan_clarification_lineage_questions(
                state.question.lineage.model_copy(update={"caller_key": "other"}), limit=32
            )
            == []
        )
        due = await tx.scan_due_clarifications(after=None, now_ms=EVERYTHING_DUE, limit=64)
        assert question_keys(due) == ["question-b", "question-c", "question-d", "question-a"]
        assert await tx.scan_due_clarifications(after=None, now_ms=199, limit=64) == []
        assert question_keys(
            await tx.scan_due_clarifications(after=None, now_ms=200, limit=64)
        ) == [
            "question-b",
            "question-c",
        ]
        cursor = ClarificationDueCursor(
            deadline_at_ms=200,
            operation=state.question.operation.model_copy(update={"caller_key": "question-b"}),
        )
        assert question_keys(
            await tx.scan_due_clarifications(after=cursor, now_ms=EVERYTHING_DUE, limit=2)
        ) == ["question-c", "question-d"]
        cursor = cursor.model_copy(
            update={"deadline_at_ms": 300, "operation": questions[1][1].question.operation}
        )
        assert await tx.scan_due_clarifications(after=cursor, now_ms=EVERYTHING_DUE, limit=64) == []

        pending = await tx.scan_pending_clarification_deliveries(after=None, limit=64)
        assert delivery_keys_of(pending) == ["delivery-1", "delivery-2", "delivery-0"]
        assert await tx.scan_pending_clarification_deliveries(after=None, limit=1) == pending[:1]
        cursor = ClarificationDueCursor(
            deadline_at_ms=180, operation=deliveries[2].intent.operation
        )
        assert await tx.scan_pending_clarification_deliveries(after=cursor, limit=64) == pending[1:]
        request = deliveries[0].intent.question.request
        handoffs = await tx.scan_clarification_request_handoffs(
            request, family="clarification_deliveries", limit=64
        )
        assert delivery_keys_of(handoffs) == ["delivery-0", "delivery-1", "delivery-2"]
        assert (
            await tx.scan_clarification_request_handoffs(
                request, family="clarification_deliveries", limit=1
            )
            == handoffs[:1]
        )
    async with store._transaction("app", write=True) as tx:
        await tx.put(
            "clarification_deliveries",
            delivery_keys[2],
            settled_record(deliveries[2]),
            insert=False,
        )
    async with stores()._transaction("app", write=False) as tx:
        assert delivery_keys_of(
            await tx.scan_pending_clarification_deliveries(after=None, limit=64)
        ) == [
            "delivery-2",
            "delivery-0",
        ]
        request = deliveries[0].intent.question.request
        assert delivery_keys_of(
            await tx.scan_clarification_request_handoffs(
                request, family="clarification_deliveries", limit=64, pending_only=True
            )
        ) == ["delivery-0", "delivery-2"]
        assert (
            len(
                await tx.scan_clarification_request_handoffs(
                    request, family="clarification_deliveries", limit=64
                )
            )
            == 3
        )


def _stage(record, initial, ordinal, name=None):
    name = ordinal if name is None else name
    command = record.receipt.command
    admission = RequestAdmissionCommand(
        operation=initial.operation(f"plan-admission-{name}"),
        expected=command.expected,
        expected_revision=command.expected_revision,
        expected_input_revision=command.expected_input_revision,
        expected_input_sha256=command.expected_input_sha256,
        generation=command.admission_generation,
        decision="defer",
        evidence=(),
        initiator=command.initiator,
    )
    intent = RequestPlanningStageIntent(
        operation=initial.operation(f"plan-stage-{name}"),
        plan=command.operation,
        plan_sha256=clarification_commitment(command, REDACTOR),
        ordinal=ordinal,
        command=admission,
    )
    event = record.receipt.event.model_copy(
        update={
            "id": f"stage-retained-{ordinal}",
            "sequence": 1 + ordinal,
            "operation": intent.operation,
            "type": "plan_stage_retained",
            "commitment": clarification_commitment(intent, REDACTOR),
        }
    )
    stage = RequestPlanningStageRecord(
        intent=intent,
        state="pending",
        registered_at_ms=record.receipt.retained_at_ms,
        settled_at_ms=None,
        receipt=None,
        registration_event=event,
        settlement_event=None,
        reserved_bytes=65536,
    )
    operation = intent.operation
    return (operation.namespace_incarnation, operation.generation, operation.caller_key), stage


async def test_planning_scans_unique_indexes_and_pending_transitions(stores):
    store = stores()
    initial, record = await planning_records(store)
    scope = initial.binding.application_scope
    command = record.receipt.command
    operation = command.operation
    key = (operation.namespace_incarnation, operation.generation, operation.caller_key)
    request = command.expected.intent.selection.reference
    stages = [_stage(record, initial, ordinal) for ordinal in (2, 1, 3)]
    async with store._transaction(scope, write=False) as tx:
        assert await tx.scan_pending_request_plans(after=None, limit=32) == []
        assert await tx.scan_request_plans(request, limit=32) == []
        assert await tx.scan_request_plan_stages(operation, limit=32) == []
    async with store._transaction(scope, write=True) as tx:
        await tx.put("request_plans", key, record, insert=True)
        for stage_key, stage in stages:
            await tx.put("request_plan_stages", stage_key, stage, insert=True)
    # A second stage at an existing ordinal violates the plan's unique ordinal.
    duplicate_key, duplicate = _stage(record, initial, 1, name="duplicate")
    with pytest.raises(Exception) as caught:
        async with store._transaction(scope, write=True) as tx:
            await tx.put("request_plan_stages", duplicate_key, duplicate, insert=True)
    assert not isinstance(caught.value, (AssertionError, CollaborationContractError))
    async with stores()._transaction(scope, write=False) as tx:
        assert await tx.get("request_plan_stages", duplicate_key) is None
        assert (
            prepare_contract(
                RequestPlanningRecord, await tx.get("request_plans", key), redactor=REDACTOR
            )
            == record
        )
        pending = await tx.scan_pending_request_plans(after=None, limit=32)
        assert pending == await tx.scan_request_plans(request, limit=32)
        assert [
            prepare_contract(RequestPlanningRecord, row, redactor=REDACTOR) for row in pending
        ] == [record]
        cursor = RequestPlanningCursor(next_due_at_ms=record.next_due_at_ms, operation=operation)
        assert await tx.scan_pending_request_plans(after=cursor, limit=32) == []
        earlier = cursor.model_copy(update={"next_due_at_ms": record.next_due_at_ms - 1})
        assert await tx.scan_pending_request_plans(after=earlier, limit=32) == pending
        ordered = await tx.scan_request_plan_stages(operation, limit=32)
        assert [row["intent"]["ordinal"] for row in ordered] == [1, 2, 3]
        assert await tx.scan_request_plan_stages(operation, limit=2) == ordered[:2]
        for stage_key, stage in stages:
            native = await tx.find_request_plan_stage(stage.intent.command.operation)
            assert prepare_contract(RequestPlanningStageRecord, native, redactor=REDACTOR) == stage
            assert (
                prepare_contract(
                    RequestPlanningStageRecord,
                    await tx.get("request_plan_stages", stage_key),
                    redactor=REDACTOR,
                )
                == stage
            )
        assert await tx.find_request_plan_stage(initial.operation("absent-stage")) is None
    cancelled = record.model_copy(
        update={
            "state": "cancelled",
            "decision_commitment": clarification_commitment(
                record.receipt.policy.rules[0].proposal, REDACTOR
            ),
            "stage_count": 3,
            "pending_stages": 3,
            "revision": 2,
            "event_sequences": (1, 5),
        }
    )
    async with store._transaction(scope, write=True) as tx:
        await tx.put("request_plans", key, cancelled, insert=False)
    async with stores()._transaction(scope, write=False) as tx:
        pending = await tx.scan_pending_request_plans(after=None, limit=32)
        assert [
            prepare_contract(RequestPlanningRecord, row, redactor=REDACTOR) for row in pending
        ] == [cancelled]
    async with store._transaction(scope, write=True) as tx:
        await tx.put(
            "request_plans", key, cancelled.model_copy(update={"pending_stages": 0}), insert=False
        )
    async with stores()._transaction(scope, write=False) as tx:
        assert await tx.scan_pending_request_plans(after=None, limit=32) == []
        assert len(await tx.scan_request_plans(request, limit=32)) == 1
