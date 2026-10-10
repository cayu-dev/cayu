"""Native event publication composes with explicit transaction capabilities."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

import cayu
from cayu.events import Event, EventType
from cayu.sessions.base import BudgetReservationIdentityConflict
from cayu.sessions.mcp_manifest_history import (
    McpManifestBaseline,
    _mcp_authoritative_manifest_hash,
    _mcp_manifest_session_ref,
)
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest


def _assert_import_without_adapters_or_postgres_driver(backend):
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
            raise AssertionError(f"Unexpected adapter/driver dependency: {fullname}")

sys.meta_path.insert(0, RejectAdapters())
owner = importlib.import_module(f"cayu.storage._{sys.argv[1]}_event_publication")
assert callable(owner.append_events)
assert callable(owner.compare_and_publish_mcp_manifest_checks)
""",
            backend,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sqlite_event_publication_imports_without_adapters_or_postgres_driver():
    _assert_import_without_adapters_or_postgres_driver("sqlite")


async def _seed(store):
    session = await store.create(
        RunRequest(session_id="event-owner", agent_name="assistant", messages=[]),
        identity=SessionIdentity(provider_name="fake", model="fake"),
    )
    await store.append_event(
        session.id,
        Event(
            id="attempt-marker",
            type="custom.cayu.workflow.attempt",
            session_id=session.id,
            workflow_name="maintenance",
            payload={"attempt_id": "attempt"},
        ),
    )
    return session.id


def _publication(session_id):
    history_key = "sha256:" + "1" * 64
    event = Event(
        id="manifest-event",
        type=EventType.MCP_MANIFEST_CHECKED,
        session_id=session_id,
        payload={
            "history_key": history_key,
            "manifest_identity": "sha256:" + "2" * 64,
            "source_manifest_hash": "sha256:" + "3" * 64,
            "server_hash": "sha256:" + "4" * 64,
            "manifest_hash": _mcp_authoritative_manifest_hash(
                source_manifest_hash="sha256:" + "3" * 64,
                server_hash="sha256:" + "4" * 64,
                tools=(),
                exposed_tools=(),
            ),
            "status": "first_seen",
            "outcome": "accepted",
        },
    )
    baseline = McpManifestBaseline(
        history_key=history_key,
        generation=1,
        manifest_identity=event.payload["manifest_identity"],
        manifest_hash=event.payload["manifest_hash"],
        source_manifest_hash=event.payload["source_manifest_hash"],
        server_hash=event.payload["server_hash"],
        tools=(),
        exposed_tools=(),
        accepted_session_ref=_mcp_manifest_session_ref(session_id),
        accepted_event_id=event.id,
        accepted_at=event.timestamp,
    )
    return history_key, dict(
        expected_generations={history_key: None},
        baseline_updates={history_key: baseline},
        events=[event],
    )


async def _exercise_admission(ops, session_id):
    claim = dict(
        reservation_id="reservation", publication_session_id=session_id, publication_id="budget"
    )
    await ops.claim_budget_reservation_identity(**claim)
    await ops.claim_budget_reservation_identity(**claim)
    with pytest.raises(BudgetReservationIdentityConflict):
        await ops.claim_budget_reservation_identity(**{**claim, "publication_id": "other"})
    event = Event(
        id="workflow-step",
        type=EventType.WORKFLOW_STEP_STARTED,
        session_id=session_id,
        workflow_name="maintenance",
        payload={"attempt_id": "attempt", "step_id": "step"},
    )
    publish = partial(
        ops.append_workflow_step_started,
        session_id,
        workflow_name="maintenance",
        attempt_id="attempt",
    )
    assert await publish(event)
    assert not await publish(event)
    stale = event.model_copy(
        update={"id": "stale-step", "payload": {"attempt_id": "stale", "step_id": "step"}}
    )
    assert not await ops.append_workflow_step_started(
        session_id, stale, workflow_name="maintenance", attempt_id="stale"
    )


def test_sqlite_event_publication_composes_with_direct_connection(tmp_path):
    import sqlite3

    from cayu.storage import _sqlite_connection
    from cayu.storage import _sqlite_event_publication as owner
    from cayu.storage import sqlite as adapter

    async def run():
        path = tmp_path / "events.sqlite"
        store = adapter.SQLiteSessionStore(path)
        try:
            sid = await _seed(store)
        finally:
            await store.close()
        connection = _sqlite_connection.connect(path)
        # This independently composed fixture has no public-authority codec.
        for name, arity, value in (
            ("cayu_public_authority_alias", 3, None),
            ("cayu_public_authority_aliases", 3, "[]"),
            ("cayu_public_authority_active_key_id", 0, None),
            ("cayu_public_authority_keyring_fingerprint", 0, None),
        ):
            connection.create_function(
                name, arity, lambda *_, value=value: value, deterministic=True
            )
        active = False

        async def execute(operation):
            nonlocal active
            assert not active
            active = True
            try:
                return operation(connection)
            finally:
                active = False
                assert not connection.in_transaction

        def closure_owners(targets, *, connection=connection):
            assert active and connection.in_transaction and tuple(targets) == (sid,)
            return ()

        common = dict(
            store_now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            closure_owners=closure_owners,
            first_existing_event_id=adapter._first_existing_event_id,
        )
        ops = SimpleNamespace(
            claim_budget_reservation_identity=partial(
                owner.claim_budget_reservation_identity,
                execute,
                closure_owners=closure_owners,
                raise_write_conflict=adapter._raise_session_write_conflict,
                claim_identity=adapter._claim_budget_reservation_identity,
            ),
            append_workflow_step_started=partial(
                owner.append_workflow_step_started,
                execute,
                **common,
                touch_activity=adapter._touch_session_activity,
            ),
        )
        read = partial(owner.load_mcp_manifest_baselines, execute)
        publish = partial(
            owner.compare_and_publish_mcp_manifest_checks,
            execute,
            sid,
            **common,
            touch_activity=adapter._touch_session_activity,
        )

        def snapshot():
            return tuple(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "cayu_events",
                    "cayu_persisted_event_side_effects",
                    "cayu_mcp_manifest_baselines",
                )
            ) + tuple(connection.execute("SELECT last_activity_at FROM cayu_sessions").fetchone())

        try:
            await _exercise_admission(ops, sid)
            event = Event(id="batch", type="custom.batch", session_id=sid)
            await owner.append_events(
                execute, sid, [event], **common, append_events=adapter._append_events_in_transaction
            )
            with pytest.raises(ValueError, match="Event already exists"):
                await owner.append_events(
                    execute,
                    sid,
                    [event],
                    **common,
                    append_events=adapter._append_events_in_transaction,
                )
            key, publication = _publication(sid)
            before = snapshot()
            connection.execute("""CREATE TEMP TRIGGER fail_manifest_baseline
                BEFORE INSERT ON cayu_mcp_manifest_baselines
                BEGIN SELECT RAISE(ABORT, 'baseline write failed'); END""")
            with pytest.raises(sqlite3.IntegrityError, match="baseline write failed"):
                await publish(**publication)
            assert snapshot() == before
            assert (await read((key,))).baselines == {}
            connection.execute("DROP TRIGGER fail_manifest_baseline")
            assert (await publish(**publication)).published
            assert not (await publish(**publication)).published
            assert (await read((key,))).baselines == publication["baseline_updates"]
            assert snapshot()[:3] == tuple(count + 1 for count in before[:3])
        finally:
            connection.close()

    asyncio.run(run())


def test_postgres_event_publication_imports_without_adapters_or_postgres_driver():
    _assert_import_without_adapters_or_postgres_driver("postgres")


def test_postgres_event_publication_composes_with_direct_connection(postgres_dsn):
    from contextlib import asynccontextmanager

    import psycopg

    from cayu.storage import _postgres_event_publication as owner
    from cayu.storage import _postgres_support as support
    from cayu.storage import postgres as adapter
    from cayu.storage.migrations import SchemaMode

    async def run():
        store = adapter.PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            sid = await _seed(store)
        finally:
            await store.close()
        active = False
        ready = False

        async def ensure_ready():
            nonlocal ready
            assert not active
            ready = True

        @asynccontextmanager
        async def connect():
            nonlocal active, ready
            assert ready and not active
            ready, active = False, True
            try:
                async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
                    yield connection
            finally:
                active = False

        async def lock_closure(cur):
            assert active
            await cur.execute("SELECT pg_advisory_xact_lock(42)")

        async def load_session(cur, session_id):
            assert active and session_id == sid
            await cur.execute(
                f"SELECT {support.SESSION_COLUMNS} FROM cayu_sessions WHERE id = %s FOR UPDATE",
                (session_id,),
            )
            row = await cur.fetchone()
            return None if row is None else support.session_from_row(row, labels={})

        async def closure_owners(cur, targets):
            assert active and tuple(targets) == (sid,)
            assert cur.connection.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS
            return ()

        async def register_events(cur, session_id, events):
            # No authority codec is configured for this independently composed fixture.
            assert active and session_id == sid and all(event.session_id == sid for event in events)
            assert cur.connection.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS

        async def first_existing_event_id(session_id, ids):
            assert active and session_id == sid
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
                for event_id in ids:
                    cur = await connection.execute(
                        "SELECT 1 FROM cayu_events WHERE session_id = %s AND event_id = %s",
                        (session_id, event_id),
                    )
                    if await cur.fetchone() is not None:
                        return event_id
            return None

        common = dict(
            ensure_ready=ensure_ready,
            load_session=load_session,
            store_now=adapter.PostgresSessionStore._session_store_now,
            closure_owners=closure_owners,
            lock_closure=lock_closure,
            register_event_authorities=register_events,
            first_existing_event_id=first_existing_event_id,
            raise_write_conflict=adapter._raise_session_write_conflict,
            unique_violation=psycopg.errors.UniqueViolation,
        )
        ops = SimpleNamespace(
            claim_budget_reservation_identity=partial(
                owner.claim_budget_reservation_identity,
                connect,
                ensure_ready=ensure_ready,
                closure_owners=closure_owners,
                raise_write_conflict=adapter._raise_session_write_conflict,
            ),
            append_workflow_step_started=partial(
                owner.append_workflow_step_started, connect, **common
            ),
        )
        read = partial(owner.load_mcp_manifest_baselines, connect, ensure_ready=ensure_ready)
        publish = partial(owner.compare_and_publish_mcp_manifest_checks, connect, sid, **common)

        async def snapshot():
            assert not active
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
                values = []
                for table in (
                    "cayu_events",
                    "cayu_persisted_event_side_effects",
                    "cayu_mcp_manifest_baselines",
                ):
                    cur = await connection.execute(f"SELECT COUNT(*) FROM {table}")
                    values.append((await cur.fetchone())[0])
                cur = await connection.execute(
                    "SELECT event_seq, last_activity_at FROM cayu_sessions"
                )
                return tuple(values) + tuple(await cur.fetchone())

        await _exercise_admission(ops, sid)
        key, publication = _publication(sid)
        before = await snapshot()
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            await connection.execute("""CREATE FUNCTION fail_manifest_baseline() RETURNS trigger
                LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'baseline write failed'; END $$""")
            await connection.execute("""CREATE TRIGGER fail_manifest_baseline BEFORE INSERT
                ON cayu_mcp_manifest_baselines FOR EACH ROW EXECUTE FUNCTION fail_manifest_baseline()""")
        try:
            with pytest.raises(psycopg.errors.RaiseException, match="baseline write failed"):
                await publish(**publication)
            assert await snapshot() == before
            assert (await read((key,))).baselines == {}
        finally:
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
                await connection.execute(
                    "DROP TRIGGER fail_manifest_baseline ON cayu_mcp_manifest_baselines"
                )
                await connection.execute("DROP FUNCTION fail_manifest_baseline()")
        assert (await publish(**publication)).published
        assert not (await publish(**publication)).published
        assert (await read((key,))).baselines == publication["baseline_updates"]
        assert (await snapshot())[:3] == tuple(count + 1 for count in before[:3])

    asyncio.run(run())
