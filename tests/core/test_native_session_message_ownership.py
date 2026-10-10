"""Native queue operations compose without retaining a session-store instance."""

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
from cayu.sessions.messaging import (
    EnqueueSessionMessageRequest,
    SessionMessageActionRequest,
    SessionMessageConditions,
    SessionMessageConflict,
    SessionMessageQuery,
)
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest


def _assert_import_without_adapters(backend):
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
        if fullname in {"cayu.storage.sqlite", "cayu.storage.postgres"}:
            raise AssertionError(f"Queue owner imported a store adapter: {fullname}")

sys.meta_path.insert(0, RejectAdapters())
owner = importlib.import_module(f"cayu.storage._{sys.argv[1]}_session_messages")
assert callable(owner.enqueue_session_message)
assert callable(owner.deliver_queued_session_messages)
""",
            backend,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


async def _seed(store):
    session = await store.create(
        RunRequest(session_id="queue-owner", agent_name="assistant", messages=[]),
        identity=SessionIdentity(provider_name="fake", model="fake"),
    )
    await store.checkpoint(session.id, {"queue_test": True})
    await store.transition_status(
        session.id, from_statuses={SessionStatus.PENDING}, to_status=SessionStatus.RUNNING
    )
    return session.id


async def _exercise(ops, sid, assert_idle):
    source = await ops.snapshot_session_message_source(
        sid, include_transcript_digest=True, include_checkpoint_digest=True
    )
    assert source.transcript_sha256 and source.checkpoint_sha256
    request = EnqueueSessionMessageRequest(
        session_id=sid,
        idempotency_key="withdraw",
        delivery_mode="next_turn",
        content="withdraw this message",
        conditions=SessionMessageConditions(source=source),
    )
    accepted = await ops.enqueue_session_message(request)
    replay = await ops.enqueue_session_message(request)
    assert replay.replayed and replay.message == accepted.message
    with pytest.raises(SessionMessageConflict):
        await ops.inspect_session_messages(
            SessionMessageQuery(session_id=sid),
            expected_authorized_session_instance_id="different-incarnation",
        )
    page = await ops.inspect_session_messages(SessionMessageQuery(session_id=sid))
    (record,) = page.records
    action = SessionMessageActionRequest(
        session_id=sid,
        session_instance_id=page.session_instance_id,
        queue_id=record.queue_id,
        expected_revision=record.revision,
        idempotency_key="withdraw-action",
        action="withdraw",
    )
    withdrawn = await ops.apply_session_message_action(action)
    assert withdrawn.record.status == "withdrawn"
    assert (await ops.apply_session_message_action(action)).replayed
    await ops.enqueue_session_message(
        EnqueueSessionMessageRequest(
            session_id=sid,
            idempotency_key="deliver",
            content="deliver this message",
            delivery_mode="next_turn",
        )
    )
    first = await ops.inspect_session_messages(SessionMessageQuery(session_id=sid, limit=1))
    assert first.next_cursor is not None
    second = await ops.inspect_session_messages(
        SessionMessageQuery(session_id=sid, limit=1, cursor=first.next_cursor)
    )
    assert second.next_cursor is None
    assert first.records[0].queue_id != second.records[0].queue_id
    batch = await ops.deliver_queued_session_messages(
        sid, include_on_idle=True, delivery_id="delivery"
    )
    assert [message.content for message in batch.messages] == ["deliver this message"]
    replay = await ops.deliver_queued_session_messages(
        sid, include_on_idle=True, delivery_id="delivery"
    )
    assert replay.replayed and replay.messages == batch.messages and replay.events == batch.events
    assert_idle()


def test_sqlite_queue_imports_without_store_adapters():
    _assert_import_without_adapters("sqlite")


def test_sqlite_queue_operations_compose_with_direct_connection(tmp_path):
    from cayu.storage import _sqlite_connection, _sqlite_records, _sqlite_session_messages
    from cayu.storage import sqlite as sqlite_adapter

    async def run():
        path = tmp_path / "queue.sqlite"
        store = sqlite_adapter.SQLiteSessionStore(path)
        try:
            sid = await _seed(store)
        finally:
            await store.close()
        connection = _sqlite_connection.connect(path)
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

        def assert_idle():
            assert not active and not connection.in_transaction

        def load_session(session_id):
            assert active and connection.in_transaction
            return _sqlite_records.load_session(connection, session_id)

        def closure_owners(targets, *, connection=connection):
            assert active and connection.in_transaction and tuple(targets) == (sid,)
            return ()

        common = dict(load_session=load_session, store_now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
        ops = SimpleNamespace(
            snapshot_session_message_source=partial(
                _sqlite_session_messages.snapshot_session_message_source,
                execute,
                load_checkpoint=sqlite_adapter._load_checkpoint_state,
            ),
            inspect_session_messages=partial(
                _sqlite_session_messages.inspect_session_messages, execute
            ),
            apply_session_message_action=partial(
                _sqlite_session_messages.apply_session_message_action,
                execute,
                **common,
                closure_owners=closure_owners,
                append_events=sqlite_adapter._append_events_in_transaction,
            ),
            enqueue_session_message=partial(
                _sqlite_session_messages.enqueue_session_message,
                execute,
                **common,
                closure_owners=closure_owners,
                load_checkpoint=sqlite_adapter._load_checkpoint_state,
                touch_activity=sqlite_adapter._touch_session_activity,
            ),
            deliver_queued_session_messages=partial(
                _sqlite_session_messages.deliver_queued_session_messages,
                execute,
                **common,
                load_checkpoint=sqlite_adapter._load_checkpoint_state,
                touch_activity=sqlite_adapter._touch_session_activity,
                reject_steering=sqlite_adapter._reject_new_work_after_steering,
                decode_stage_record=sqlite_adapter._decode_model_completion_stage_record,
            ),
        )
        try:
            await _exercise(ops, sid, assert_idle)
            await ops.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=sid,
                    idempotency_key="fault",
                    content="retry",
                    delivery_mode="next_turn",
                )
            )
            before = await ops.inspect_session_messages(SessionMessageQuery(session_id=sid))
            connection.execute("""CREATE TEMP TRIGGER fail_queue_receipt
                BEFORE INSERT ON cayu_session_message_deliveries
                BEGIN SELECT RAISE(ABORT, 'receipt write failed'); END""")
            import sqlite3

            with pytest.raises(sqlite3.IntegrityError, match="receipt write failed"):
                await ops.deliver_queued_session_messages(
                    sid, include_on_idle=True, delivery_id="retry-delivery"
                )
            assert_idle()
            assert await ops.inspect_session_messages(SessionMessageQuery(session_id=sid)) == before
            assert (
                connection.execute("SELECT COUNT(*) FROM cayu_transcript_messages").fetchone()[0]
                == 1
            )
            connection.execute("DROP TRIGGER fail_queue_receipt")
            batch = await ops.deliver_queued_session_messages(
                sid, include_on_idle=True, delivery_id="retry-delivery"
            )
            assert len(batch.messages) == 1
        finally:
            connection.close()

    asyncio.run(run())


@pytest.fixture
def queue_postgres_dsn(postgres_dsn):
    from tests.core.postgres_contention_support import drop_cayu_tables

    asyncio.run(drop_cayu_tables(postgres_dsn))
    try:
        yield postgres_dsn
    finally:
        asyncio.run(drop_cayu_tables(postgres_dsn))


def test_postgres_queue_imports_without_store_adapters():
    _assert_import_without_adapters("postgres")


def test_postgres_queue_operations_compose_with_direct_connection(queue_postgres_dsn):
    from contextlib import asynccontextmanager

    import psycopg

    from cayu.storage import _postgres_session_messages as owner
    from cayu.storage import _postgres_support as support
    from cayu.storage import postgres as adapter
    from cayu.storage.migrations import SchemaMode

    async def run():
        store = adapter.PostgresSessionStore(queue_postgres_dsn, schema_mode=SchemaMode.CREATE)
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
            ready = False
            active = True
            try:
                async with await psycopg.AsyncConnection.connect(queue_postgres_dsn) as connection:
                    yield connection
            finally:
                active = False

        def assert_transaction(cur):
            assert active
            assert cur.connection.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS

        async def load_labels(cur, session_id):
            assert_transaction(cur)
            assert session_id == sid
            return {}

        async def load_session(cur, session_id):
            assert active
            await cur.execute(
                f"SELECT {support.SESSION_COLUMNS} FROM cayu_sessions WHERE id = %s FOR UPDATE",
                (session_id,),
            )
            row = await cur.fetchone()
            return None if row is None else support.session_from_row(row, labels={})

        async def load_checkpoint(cur, session_id):
            assert_transaction(cur)
            await cur.execute(
                "SELECT state FROM cayu_checkpoints WHERE session_id = %s", (session_id,)
            )
            row = await cur.fetchone()
            return None if row is None else support._json_obj(row[0])

        async def closure_owners(cur, targets):
            assert_transaction(cur)
            assert tuple(targets) == (sid,)
            return ()

        async def register_events(cur, session_id, events):
            # No authority codec is configured for this independently composed store.
            assert_transaction(cur)
            assert session_id == sid and all(event.session_id == sid for event in events)

        async def register_authorities(cur, session_id, *, interaction_ids=()):
            assert_transaction(cur)
            assert session_id == sid and not interaction_ids

        async def reject_steering(cur, session, *, allow_completed_interaction=False):
            assert_transaction(cur)
            assert session.id == sid and session.status == SessionStatus.RUNNING

        async def upsert_checkpoint(*args):
            raise AssertionError("This delivery has no profile handoff to publish")

        common = dict(ensure_ready=ensure_ready)
        writes = dict(
            **common,
            load_session=load_session,
            store_now=adapter.PostgresSessionStore._session_store_now,
            load_checkpoint=load_checkpoint,
        )
        snapshot = partial(
            owner.snapshot_session_message_source,
            connect,
            **common,
            load_labels=load_labels,
            load_checkpoint=load_checkpoint,
        )
        inspect = partial(
            owner.inspect_session_messages, connect, **common, load_labels=load_labels
        )
        enqueue = partial(
            owner.enqueue_session_message,
            connect,
            **writes,
            closure_owners=closure_owners,
            register_event_authorities=register_events,
        )
        deliver = partial(
            owner.deliver_queued_session_messages,
            connect,
            **writes,
            register_event_authorities=register_events,
            register_authorities=register_authorities,
            reject_steering=reject_steering,
            upsert_checkpoint=upsert_checkpoint,
            decode_stage_record=adapter._decode_model_completion_stage_record,
        )
        source = await snapshot(sid, include_transcript_digest=True, include_checkpoint_digest=True)
        assert source.transcript_sha256 and source.checkpoint_sha256
        request = EnqueueSessionMessageRequest(
            session_id=sid,
            idempotency_key="direct",
            content="direct queue message",
            delivery_mode="next_turn",
            conditions=SessionMessageConditions(source=source),
        )
        accepted = await enqueue(request)
        assert (await enqueue(request)).replayed
        query = SessionMessageQuery(session_id=sid)
        before = await inspect(query)
        assert before.records[0].queue_id == accepted.message.queue_id
        with pytest.raises(SessionMessageConflict):
            await inspect(query, expected_authorized_session_instance_id="different-incarnation")
        async with await psycopg.AsyncConnection.connect(queue_postgres_dsn) as conn:
            await conn.execute("""CREATE FUNCTION fail_queue_receipt() RETURNS trigger
                LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'receipt write failed'; END $$""")
            await conn.execute("""CREATE TRIGGER fail_queue_receipt BEFORE INSERT
                ON cayu_session_message_deliveries FOR EACH ROW EXECUTE FUNCTION fail_queue_receipt()""")
        try:
            with pytest.raises(psycopg.errors.RaiseException, match="receipt write failed"):
                await deliver(sid, include_on_idle=True, delivery_id="direct-delivery")
            assert not active
            assert await inspect(query) == before
            async with await psycopg.AsyncConnection.connect(queue_postgres_dsn) as conn:
                cur = await conn.execute("SELECT COUNT(*) FROM cayu_transcript_messages")
                assert (await cur.fetchone())[0] == 0
        finally:
            async with await psycopg.AsyncConnection.connect(queue_postgres_dsn) as conn:
                await conn.execute(
                    "DROP TRIGGER fail_queue_receipt ON cayu_session_message_deliveries"
                )
                await conn.execute("DROP FUNCTION fail_queue_receipt()")
        batch = await deliver(sid, include_on_idle=True, delivery_id="direct-delivery")
        assert len(batch.messages) == 1 and batch.messages[0].queue_id == accepted.message.queue_id
        replay = await deliver(sid, include_on_idle=True, delivery_id="direct-delivery")
        assert (
            replay.replayed and replay.messages == batch.messages and replay.events == batch.events
        )
        assert not active

    asyncio.run(run())
