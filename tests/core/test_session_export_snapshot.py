from __future__ import annotations

import asyncio
import io
import json
from uuid import uuid4

import pytest

from cayu.core import Event, EventType, Message
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.runtime.exports import SessionExportBuilder, SessionExportLimits, SessionExportTooLarge
from cayu.runtime.sessions import (
    EnqueueSessionMessageRequest,
    InMemorySessionStore,
    RunRequest,
    SessionIdentity,
    SessionMessageDeliveryMode,
    SessionStatus,
)
from cayu.storage import PostgresSessionStore, SQLiteSessionStore
from cayu.storage.jsonl_export import export_sessions, import_sessions
from cayu.storage.migrations import SchemaMode


def _store(backend, tmp_path, request):
    if backend == "memory":
        return InMemorySessionStore()
    if backend == "sqlite":
        return SQLiteSessionStore(tmp_path / "export.sqlite")
    from tests.core.test_targeted_tool_grants import _codec

    return PostgresSessionStore(
        request.getfixturevalue("postgres_dsn"),
        schema_mode=SchemaMode.CREATE,
        public_authority_alias_codec=_codec(),
    )


async def _create(store, session_id):
    await store.create(
        RunRequest(
            session_id=session_id, agent_name="assistant", messages=[Message.text("user", "hi")]
        ),
        identity=SessionIdentity(provider_name="fake", model="fake"),
    )
    await store.append_transcript_messages(session_id, [Message.text("assistant", "before")])
    await store.checkpoint(
        session_id,
        {
            CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
            "compacted_transcript_cursor": 1,
        },
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_export_snapshot_does_not_compose_independent_reads(
    backend, tmp_path, request, monkeypatch
):
    async def run():
        store = _store(backend, tmp_path, request)
        session_id = f"snapshot-{uuid4()}"
        try:
            await _create(store, session_id)
            event = Event(session_id=session_id, type=EventType.TOOL_CALL_STARTED)
            await store.append_event(session_id, event)

            async def forbidden(*args, **kwargs):
                raise AssertionError("Export must use one native snapshot.")

            for method in [
                "load_events",
                "query_transcript",
                "load_checkpoint",
                "load_deferred_interaction_input",
                "load_targeted_tool_grant_state",
            ]:
                monkeypatch.setattr(store, method, forbidden)
            snapshot = await store.load_session_export_snapshot(session_id)
            assert snapshot is not None
            assert [record.event.id for record in snapshot.events] == [event.id]
            assert snapshot.boundary.transcript_cursor == 1
            assert snapshot.boundary.event_sequences == (snapshot.events[0].sequence,)
            [restored] = import_sessions([json.dumps(snapshot.document())])
            assert restored.boundary == snapshot.boundary
            assert restored.events[0].id == event.id
            assert restored.transcript_records[0].index == 0
            assert await store.load_session_export_snapshot("absent") is None
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_export_snapshot_preserves_one_boundary_during_concurrent_mutation(
    backend, tmp_path, request, monkeypatch
):
    async def run():
        store = _store(backend, tmp_path, request)
        session_id = f"race-{uuid4()}"
        try:
            await _create(store, session_id)
            initial = [
                Event(session_id=session_id, type=EventType.TOOL_CALL_STARTED) for _ in range(513)
            ]
            await store.append_events(session_id, initial)
            tail = Event(session_id=session_id, type=EventType.SESSION_COMPLETED)
            mutated = False
            mutation_task = None

            async def mutate():
                await store.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id=session_id,
                        idempotency_key="raced-input",
                        content="queued while exporting",
                        delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                    )
                )
                await store.append_transcript_messages_and_transform_checkpoint(
                    session_id,
                    [Message.text("assistant", "after")],
                    lambda session, checkpoint: {**checkpoint, "compacted_transcript_cursor": 2},
                )
                await store.append_event(session_id, tail)
                await store.update_status(session_id, SessionStatus.COMPLETED)

            if backend == "postgres":
                from psycopg import AsyncServerCursor

                original_fetch = AsyncServerCursor.fetchmany

                async def interleave(cursor, size=0):
                    nonlocal mutated
                    page = await original_fetch(cursor, size)
                    if cursor.name.startswith("session_export_events_") and not mutated:
                        mutated = True
                        await mutate()
                    return page

                monkeypatch.setattr(AsyncServerCursor, "fetchmany", interleave)
            else:
                original_event = SessionExportBuilder.event
                loop = asyncio.get_running_loop()

                def interleave(builder, record):
                    nonlocal mutated, mutation_task
                    if not mutated:
                        mutated = True
                        if backend == "sqlite":
                            asyncio.run_coroutine_threadsafe(mutate(), loop).result(timeout=10)
                        else:
                            mutation_task = asyncio.create_task(mutate())
                    return original_event(builder, record)

                monkeypatch.setattr(SessionExportBuilder, "event", interleave)
            old = await store.load_session_export_snapshot(session_id)
            assert mutated
            if mutation_task is not None:
                await mutation_task
            assert old.session.status == SessionStatus.PENDING
            assert len(old.events) == 513
            assert old.boundary.transcript_cursor == 1
            assert old.checkpoint["compacted_transcript_cursor"] == 1
            assert [record.message.content[0].text for record in old.transcript_records] == [
                "before"
            ]
            current = await store.load_session_export_snapshot(session_id)
            assert current.session.status == SessionStatus.COMPLETED
            assert current.boundary.transcript_cursor == 2
            assert current.checkpoint["compacted_transcript_cursor"] == 2
            assert len(current.events) == 515
            assert (
                sum(
                    record.event.type == EventType.SESSION_MESSAGE_QUEUED
                    for record in current.events
                )
                == 1
            )
            assert current.events[-1].event.id == tail.id
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_oversized_export_emits_no_partial_session_line(backend, tmp_path, request):
    async def run():
        store = _store(backend, tmp_path, request)
        try:
            await _create(store, f"bounded-{uuid4()}")
            stream = io.StringIO()
            with pytest.raises(SessionExportTooLarge):
                await export_sessions(
                    store, stream=stream, limits=SessionExportLimits(max_bytes=100)
                )
            assert stream.getvalue() == ""
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "change",
    [
        "cursor",
        "event_watermark",
        "event_positions",
        "event_session",
        "model_pointer",
        "version",
        "deferred",
    ],
)
def test_import_rejects_inconsistent_snapshot_before_yielding(change):
    async def make():
        store = InMemorySessionStore()
        await _create(store, "validation")
        await store.append_event(
            "validation", Event(session_id="validation", type=EventType.MODEL_COMPLETED)
        )
        snapshot = await store.load_session_export_snapshot("validation")
        return snapshot.document()

    document = asyncio.run(make())
    if change == "cursor":
        document["checkpoint"]["compacted_transcript_cursor"] = 2
    elif change == "event_watermark":
        document["snapshot"]["through_sequence"] += 1
    elif change == "event_positions":
        document["snapshot"]["event_sequences"] = []
    elif change == "event_session":
        document["events"][0]["session_id"] = "foreign"
    elif change == "deferred":
        document["deferred_interaction_input"] = {
            "interaction_id": "absent",
            "source_messages": [Message.text("user", "input").model_dump(mode="json")],
        }
    elif change == "version":
        document["format_version"] = 3
    else:
        document["checkpoint"]["last_model_step_publication"] = {
            "logical_step_id": "step",
            "stage_id": "stage",
            "source_transcript_cursor": 0,
            "transcript_end_cursor": 1,
            "completion_event_id": "absent",
            "classification": {"type": "final"},
            "assistant_message_published": True,
        }
    parsed = import_sessions([json.dumps(document)])
    with pytest.raises(ValueError):
        next(parsed)


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_export_size_preflight_rejects_before_hydrating_events(
    backend, tmp_path, request, monkeypatch
):
    async def run():
        store = _store(backend, tmp_path, request)
        session_id = f"oversized-{uuid4()}"
        try:
            await _create(store, session_id)
            await store.append_event(
                session_id,
                Event(
                    session_id=session_id,
                    type=EventType.MODEL_TEXT_DELTA,
                    payload={"delta": "x" * 8192},
                ),
            )

            def forbidden(*args, **kwargs):
                raise AssertionError(
                    "Oversized SQL source must be rejected before event hydration."
                )

            monkeypatch.setattr(SessionExportBuilder, "event", forbidden)
            with pytest.raises(SessionExportTooLarge, match="source exceeds"):
                await store.load_session_export_snapshot(
                    session_id, limits=SessionExportLimits(max_record_bytes=4096)
                )
        finally:
            await store.close()

    asyncio.run(run())


def test_sqlite_retention_during_export_preserves_old_transcript_and_permanent_cursor(
    tmp_path, monkeypatch
):
    async def run():
        store = SQLiteSessionStore(tmp_path / "retention.sqlite")
        session_id = "retention"
        try:
            await _create(store, session_id)
            await store.append_event(
                session_id, Event(session_id=session_id, type=EventType.MODEL_COMPLETED)
            )
            loop = asyncio.get_running_loop()
            original = SessionExportBuilder.event
            changed = False

            def interleave(builder, record):
                nonlocal changed
                if not changed:
                    changed = True
                    assert (
                        asyncio.run_coroutine_threadsafe(
                            store.compact_transcript(session_id, keep_last=0), loop
                        ).result(timeout=10)
                        == 1
                    )
                return original(builder, record)

            monkeypatch.setattr(SessionExportBuilder, "event", interleave)
            snapshot = await store.load_session_export_snapshot(session_id)
            assert snapshot.boundary.transcript_cursor == 1
            assert len(snapshot.transcript_records) == 1
            current = await store.load_session_export_snapshot(session_id)
            assert current.boundary.transcript_cursor == 1
            assert current.transcript_records == ()
            [restored] = import_sessions([json.dumps(current.document())])
            assert restored.boundary.transcript_cursor == 1
        finally:
            await store.close()

    asyncio.run(run())


def test_legacy_session_shape_remains_readable_without_inventing_snapshot_proof():
    async def make():
        store = InMemorySessionStore()
        await _create(store, "legacy")
        snapshot = await store.load_session_export_snapshot("legacy")
        return snapshot.document()

    record = asyncio.run(make())
    del record["format_version"]
    del record["snapshot"]
    [restored] = import_sessions([json.dumps(record)])
    assert restored.boundary is None
    assert restored.transcript_records[0].index == 0


def test_postgres_snapshot_keeps_targeted_grants_and_uses_in_the_same_state(postgres_dsn):
    from datetime import UTC, datetime

    from tests.core.test_targeted_tool_grants import _codec, _open_targeted_grant

    from cayu.runtime.tool_grants import TargetedToolUseRequest

    async def run():
        store = PostgresSessionStore(
            postgres_dsn, schema_mode=SchemaMode.CREATE, public_authority_alias_codec=_codec()
        )
        stream = None
        try:
            session_id = f"export-grants-{uuid4()}"
            _, _, stream, _, grant, session = await _open_targeted_grant(
                store, session_id=session_id
            )
            await store.bind_targeted_tool_grant_use(
                TargetedToolUseRequest(
                    **{
                        field: getattr(grant, field)
                        for field in (
                            "tool_ref",
                            "interaction_id",
                            "generation_id",
                            "agent_name",
                            "task_id",
                            "environment_name",
                            "principal",
                            "tenant",
                            "catalogue_revision",
                            "descriptor_version",
                            "schema_fingerprint",
                            "tool_id",
                            "tool_name",
                        )
                    },
                    session_id=session_id,
                    model_step_id="step",
                    outer_tool_call_id="call",
                    arguments_sha256="sha256:" + "1" * 64,
                    invocation_id="invocation",
                    expected_run_epoch=session.run_epoch,
                ),
                observed_at=datetime.now(UTC),
            )
            async for _ in stream:
                pass
            snapshot = await store.load_session_export_snapshot(session_id)
            expected = await store.load_targeted_tool_grant_state(session_id)
            assert len(expected.records) == len(expected.uses) == 1
            assert snapshot.targeted_tool_grant_state == expected
            [restored] = import_sessions([json.dumps(snapshot.document())])
            assert restored.targeted_tool_grant_state == expected
        finally:
            if stream is not None:
                await stream.aclose()
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_deferred_input_materialization_cannot_disappear_between_export_components(
    backend, tmp_path, request, monkeypatch
):
    async def run():
        store = _store(backend, tmp_path, request)
        session_id = f"deferred-{uuid4()}"
        interaction_id = f"interaction-{uuid4()}"
        source = [Message.text("user", "admitted private input")]
        mutation_task = None
        changed = False
        try:
            await store.create(
                RunRequest(session_id=session_id, agent_name="assistant", messages=source),
                identity=SessionIdentity(provider_name="fake", model="fake"),
                interaction_started_event=Event(
                    session_id=session_id,
                    interaction_id=interaction_id,
                    type=EventType.INTERACTION_STARTED,
                ),
                interaction_source_messages=source,
            )

            async def materialize():
                assert await store.materialize_deferred_interaction_input(
                    session_id, interaction_id=interaction_id
                )

            if backend == "postgres":
                from psycopg import AsyncServerCursor

                original_fetch = AsyncServerCursor.fetchmany

                async def interleave(cursor, size=0):
                    nonlocal changed
                    page = await original_fetch(cursor, size)
                    if cursor.name.startswith("session_export_events_") and not changed:
                        changed = True
                        await materialize()
                    return page

                monkeypatch.setattr(AsyncServerCursor, "fetchmany", interleave)
            else:
                original_event = SessionExportBuilder.event
                loop = asyncio.get_running_loop()

                def interleave(builder, record):
                    nonlocal changed, mutation_task
                    if not changed:
                        changed = True
                        if backend == "sqlite":
                            asyncio.run_coroutine_threadsafe(materialize(), loop).result(timeout=10)
                        else:
                            mutation_task = asyncio.create_task(materialize())
                    return original_event(builder, record)

                monkeypatch.setattr(SessionExportBuilder, "event", interleave)
            old = await store.load_session_export_snapshot(session_id)
            if mutation_task is not None:
                await mutation_task
            assert changed
            assert old.boundary.transcript_cursor == 0
            assert old.transcript_records == ()
            assert old.deferred_interaction_input.source_messages == source
            current = await store.load_session_export_snapshot(session_id)
            assert current.boundary.transcript_cursor == 1
            assert current.deferred_interaction_input is None
            assert current.transcript_records[0].message == source[0]
        finally:
            await store.release_run_fence(session_id)
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


def test_custom_store_without_snapshot_support_fails_closed():
    from cayu.runtime.sessions import SessionStore

    class LegacyStore(InMemorySessionStore):
        load_session_export_snapshot = SessionStore.load_session_export_snapshot

        async def load_events(self, session_id):
            raise AssertionError("Legacy reads cannot implement an export snapshot.")

    async def run():
        store = LegacyStore()
        await _create(store, "legacy-store")
        stream = io.StringIO()
        with pytest.raises(NotImplementedError, match="consistent exports"):
            await export_sessions(store, stream=stream)
        assert stream.getvalue() == ""

    asyncio.run(run())
