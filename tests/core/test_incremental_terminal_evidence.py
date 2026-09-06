from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from datetime import UTC, datetime

import pytest
from benchmarks.incremental_evidence import (
    complete_session,
    insert_sqlite_records,
    prepare_session,
)
from tests.core.test_terminal_session_evidence import _reset_postgres

from cayu import (
    Event,
    PostgresSessionStore,
    RunRequest,
    SessionIdentity,
    SQLiteSessionStore,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceLimits,
)
from cayu.runtime.evidence_spool import (
    EvidenceSpool,
    IncrementalEvidenceError,
    IncrementalEvidenceLimits,
    _incremental_evidence_sha256,
)
from cayu.storage.migrations import SchemaMode


@pytest.fixture(params=("sqlite", "postgres"))
def source(request, tmp_path):
    if request.param == "sqlite":
        location = tmp_path / "source.sqlite3"

        async def create():
            return SQLiteSessionStore(location)
    else:
        location = request.getfixturevalue("postgres_dsn")

        async def create():
            await _reset_postgres(location)
            return PostgresSessionStore(
                location, min_size=1, max_size=4, schema_mode=SchemaMode.CREATE
            )

    return request.param, location, create


async def _insert(store, backend, count, payload_bytes=128):
    if backend == "sqlite":
        await insert_sqlite_records(store, count, payload_bytes)
        return
    stamp = datetime.now(UTC).isoformat()
    async with store._connection() as connection, connection.cursor() as cursor:
        await cursor.execute(
            "UPDATE cayu_sessions SET event_seq=event_seq+%s WHERE id='benchmark' RETURNING event_seq-%s",
            (count, count),
        )
        first_order = (await cursor.fetchone())[0]
        async with cursor.copy(
            "COPY cayu_events(session_id, session_order, event_id, event_type, timestamp, payload, event) FROM STDIN"
        ) as copy:
            for index in range(count):
                event = {
                    "id": f"synthetic-{index}",
                    "type": "custom.synthetic.evidence",
                    "session_id": "benchmark",
                    "timestamp": stamp,
                    "payload": {"text": "x" * payload_bytes},
                }
                await copy.write_row(
                    (
                        "benchmark",
                        first_order + index + 1,
                        event["id"],
                        event["type"],
                        stamp,
                        json.dumps(event["payload"]),
                        json.dumps(event),
                    )
                )


async def _seed(source, count=10, payload_bytes=128):
    backend, _location, create = source
    store = await create()
    await prepare_session(store)
    await _insert(store, backend, count, payload_bytes)
    await complete_session(store)
    return store


def _mutate(source, *, delete=False):
    backend, location, _ = source
    if backend == "sqlite":
        with sqlite3.connect(location) as connection:
            if delete:
                connection.execute("DELETE FROM cayu_events WHERE event_id='synthetic-7'")
            else:
                connection.execute(
                    "UPDATE cayu_events SET payload_json='{}' WHERE event_id='synthetic-7'"
                )
    else:
        import psycopg

        with psycopg.connect(location) as connection:
            if delete:
                connection.execute("DELETE FROM cayu_events WHERE event_id='synthetic-7'")
            else:
                connection.execute(
                    "UPDATE cayu_events SET payload='{}'::jsonb, event=jsonb_set(event, '{payload}', '{}'::jsonb) WHERE event_id='synthetic-7'"
                )


def test_spool_equals_eager_and_closes_partial_consumer(source):
    async def exercise():
        store = await _seed(source)
        try:
            eager = await store.load_terminal_session_evidence("benchmark")
            with EvidenceSpool(IncrementalEvidenceLimits(batch_records=1)) as spool:
                path = spool.path
                await store.export_terminal_session_evidence("benchmark", spool=spool)
                assert spool.session == eager.session
                assert tuple(spool.events) == eager.events
                assert tuple(spool.transcript) == eager.transcript
                assert spool.boundary == eager.boundary
                assert spool.evidence_sha256 == _incremental_evidence_sha256(eager)
                assert path.exists()
                reader = iter(spool.events)
                next(reader)
                # This independent append succeeds before the consumer finishes.
                await store.create(
                    RunRequest(session_id="unrelated", agent_name="synthetic", messages=[]),
                    identity=SessionIdentity(provider_name="synthetic", model="synthetic"),
                )
                await store.append_event(
                    "unrelated", Event(type="custom.synthetic.unrelated", session_id="unrelated")
                )
            assert not path.exists()
            reader.close()
            with EvidenceSpool(IncrementalEvidenceLimits()) as fresh:
                await store.export_terminal_session_evidence("benchmark", spool=fresh)
                assert fresh.evidence_sha256 == spool.evidence_sha256
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("delete", [False, True])
def test_source_snapshot_is_stable_across_pages_and_fresh_seal_detects_change(source, delete):
    async def exercise():
        store = await _seed(source)
        try:
            with EvidenceSpool(IncrementalEvidenceLimits(batch_records=1)) as initial:
                await store.export_terminal_session_evidence("benchmark", spool=initial)
                original = initial.evidence_sha256
            with EvidenceSpool(IncrementalEvidenceLimits(batch_records=1)) as spool:
                append = spool.append
                changed = False

                def mutate_after_first_page(kind, record):
                    nonlocal changed
                    append(kind, record)
                    if not changed:
                        changed = True
                        _mutate(source, delete=delete)

                spool.append = mutate_after_first_page
                await store.export_terminal_session_evidence("benchmark", spool=spool)
                assert spool.evidence_sha256 == original
            with EvidenceSpool(IncrementalEvidenceLimits(batch_records=1)) as fresh:
                await store.export_terminal_session_evidence("benchmark", spool=fresh)
                assert fresh.evidence_sha256 != original
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("damage", ["missing", "duplicate", "out_of_order"])
def test_missing_duplicate_and_out_of_order_delivery_fail_closed(source, damage):
    async def exercise():
        store = await _seed(source)
        try:
            with EvidenceSpool(IncrementalEvidenceLimits(batch_records=1)) as spool:
                append = spool.append
                held = []

                def corrupt(kind, record):
                    if kind == "event" and record.event.id in {"synthetic-3", "synthetic-4"}:
                        if damage == "missing":
                            return
                        if damage == "duplicate":
                            append(kind, record)
                        if damage == "out_of_order":
                            held.append(record)
                            if len(held) == 2:
                                append(kind, held[1])
                                append(kind, held[0])
                            return
                    append(kind, record)

                spool.append = corrupt
                with pytest.raises((IncrementalEvidenceError, TerminalSessionEvidenceError)):
                    await store.export_terminal_session_evidence("benchmark", spool=spool)
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "limits",
    [
        {"max_events": 1},
        {"max_transcript_records": 0},
        {"max_record_bytes": 64},
        {"max_total_bytes": 100},
        {"max_lifecycle_records": 1},
        {"max_lifecycle_bytes": 1},
        {"max_spill_bytes": 12288},
    ],
)
def test_independent_limits_remain_enforced(source, limits):
    async def exercise():
        store = await _seed(source)
        try:
            with EvidenceSpool(IncrementalEvidenceLimits(**limits)) as spool:
                path = spool.path
                with pytest.raises((IncrementalEvidenceError, TerminalSessionEvidenceError)):
                    await store.export_terminal_session_evidence("benchmark", spool=spool)
            assert not path.exists()
        finally:
            await store.close()

    asyncio.run(exercise())


def test_cancel_validation_settles_worker_before_spill_disposal(source):
    async def exercise():
        store = await _seed(source)
        entered = threading.Event()
        stopped = threading.Event()
        try:
            with EvidenceSpool(IncrementalEvidenceLimits()) as spool:
                path = spool.path

                def blocked_validation():
                    entered.set()
                    try:
                        while not spool.should_interrupt():
                            time.sleep(0.001)
                        spool.check()
                    finally:
                        stopped.set()

                spool.seal = blocked_validation
                task = asyncio.create_task(
                    store.export_terminal_session_evidence("benchmark", spool=spool)
                )
                assert await asyncio.to_thread(entered.wait, 5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=2)
                assert stopped.is_set()
            assert not path.exists()
            assert (await store.load_bounded("benchmark", max_bytes=1_048_576)).id == "benchmark"
            assert not getattr(store, "_detached_read_tasks", set())
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.stress
def test_large_trace_exceeds_eager_hard_ceiling(source):
    async def exercise():
        store = await _seed(source, count=250_000, payload_bytes=8)
        try:
            with pytest.raises(TerminalSessionEvidenceError) as rejected:
                await store.load_terminal_session_evidence(
                    "benchmark", limits=TerminalSessionEvidenceLimits(max_events=100_000)
                )
            assert rejected.value.code == "event_limit_exceeded"
            with EvidenceSpool(
                IncrementalEvidenceLimits(max_seconds=600, batch_records=64, max_record_bytes=4096)
            ) as spool:
                await store.export_terminal_session_evidence("benchmark", spool=spool)
                assert len(spool.events) > 250_000
                assert (
                    sum(record.event.type == "custom.synthetic.evidence" for record in spool.events)
                    == 250_000
                )
                assert spool.peak_record_bytes < 4096
                assert spool.peak_page_records <= spool.limits.batch_records
                assert spool.peak_page_transport_bytes < 4096 * spool.limits.batch_records
                assert spool.path.stat().st_size <= spool.limits.max_spill_bytes
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("damage", ["payload", "position", "oversize"])
def test_private_backing_integrity_is_checked_before_decode(source, damage, monkeypatch):
    import cayu.runtime.evidence_spool as backing

    async def exercise():
        store = await _seed(source)
        try:
            with EvidenceSpool(IncrementalEvidenceLimits()) as spool:
                await store.export_terminal_session_evidence("benchmark", spool=spool)
                if damage == "payload":
                    spool._db.execute(
                        "UPDATE records SET payload=? WHERE kind='event' AND position=0", (b"{}",)
                    )
                elif damage == "position":
                    spool._db.execute(
                        "UPDATE records SET payload=(SELECT payload FROM records WHERE kind='event' AND position=1), tag=(SELECT tag FROM records WHERE kind='event' AND position=1) WHERE kind='event' AND position=0"
                    )
                else:
                    spool._db.execute(
                        "UPDATE records SET payload=zeroblob(?) WHERE kind='event' AND position=0",
                        (spool._max_encoded_record_bytes + 1,),
                    )

                def forbidden(*args, **kwargs):
                    pytest.fail("Unauthenticated bytes reached the decoder")

                monkeypatch.setattr(backing.json, "loads", forbidden)
                with pytest.raises(IncrementalEvidenceError, match="spill_integrity_rejected"):
                    spool.events[0]
        finally:
            monkeypatch.undo()
            await store.close()

    asyncio.run(exercise())
