"""Lifecycle acceptance through real store entrances, without provider dispatch."""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import UTC, datetime

import pytest
from tests.core.test_session_store_shared_conformance import (
    _close_store,
    _identity,
    _open_store,
    _reopen_store,
)
from tests.core.test_session_store_shared_conformance import (
    conformance_postgres_dsn as conformance_postgres_dsn,
)

from cayu.core.events import Event, EventType
from cayu.core.messages import Message
from cayu.runtime.session_message_lifecycle import (
    SessionMessageActionRequest,
    SessionMessageConditions,
    SessionMessageConflict,
    SessionMessageCursor,
    SessionMessageQuery,
    SessionMessageTarget,
)
from cayu.runtime.sessions import (
    EnqueueSessionMessageRequest,
    RunRequest,
    SessionMessageDeliveryMode,
    SessionQueuedMessagesPending,
    SessionStatus,
    TranscriptRecord,
    TranscriptSnapshot,
    fork_source_transcript_sha256,
)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def lifecycle_case(request, tmp_path):
    if request.param == "postgres":
        # Ordinary focused local runs do not allocate a database implicitly.
        # Required CI uses the canonical fixture and must not silently skip it.
        if not os.environ.get("CAYU_TEST_POSTGRES_DSN") and os.environ.get(
            "CAYU_REQUIRE_POSTGRES", ""
        ).strip().lower() not in {"1", "true", "yes", "on"}:
            pytest.skip("Set CAYU_TEST_POSTGRES_DSN to run PostgreSQL lifecycle tests.")
        return request.param, tmp_path, request.getfixturevalue("conformance_postgres_dsn")
    return request.param, tmp_path, None


async def _session(store, sid="queue-target"):
    return await store.create(
        RunRequest(agent_name="assistant", session_id=sid, messages=[]), identity=_identity()
    )


def test_required_postgres_lane_uses_canonical_fixture_without_explicit_dsn(monkeypatch, tmp_path):
    monkeypatch.delenv("CAYU_TEST_POSTGRES_DSN", raising=False)
    monkeypatch.setenv("CAYU_REQUIRE_POSTGRES", "1")
    calls = []

    class Request:
        param = "postgres"

        def getfixturevalue(self, name):
            calls.append(name)
            return "canonical-test-dsn"

    case = lifecycle_case.__wrapped__(Request(), tmp_path)
    assert case == ("postgres", tmp_path, "canonical-test-dsn")
    assert calls == ["conformance_postgres_dsn"]


def _request(sid, key, conditions=None):
    return EnqueueSessionMessageRequest(
        session_id=sid,
        idempotency_key=key,
        content="private steering",
        delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
        conditions=conditions or SessionMessageConditions(),
    )


def _action(page, record, action="withdraw"):
    return SessionMessageActionRequest(
        session_id=page.session_id,
        session_instance_id=page.session_instance_id,
        queue_id=record.queue_id,
        expected_revision=record.revision,
        idempotency_key="action-" + record.queue_id,
        action=action,
    )


@pytest.mark.parametrize("operation", ["inspect", "snapshot"])
def test_authorized_read_incarnation_fences_before_content_lookup(
    lifecycle_case, operation, monkeypatch
):
    async def run():
        store = await _open_store(lifecycle_case)
        try:
            original = await _session(store)
            await store.delete_session(original.id)
            target = await _session(store, original.id)
            await store.enqueue_session_message(_request(target.id, "replacement-content"))
            reads = []

            async def read(**kwargs):
                if operation == "inspect":
                    # The first page has no incarnation-bound cursor to protect it.
                    return await store.inspect_session_messages(
                        SessionMessageQuery(session_id=target.id), **kwargs
                    )
                return await store.snapshot_session_message_source(
                    target.id,
                    include_transcript_digest=True,
                    include_checkpoint_digest=True,
                    **kwargs,
                )

            def observe_sql(statement):
                if any(
                    table in str(statement).lower()
                    for table in (
                        "cayu_session_message_queue",
                        "cayu_transcript",
                        "cayu_checkpoints",
                    )
                ):
                    reads.append("content SQL")

            with monkeypatch.context() as patch:
                if lifecycle_case[0] == "memory":

                    class TrackedContent(dict):
                        def get(self, *args):
                            reads.append("content mapping")
                            return super().get(*args)

                    for name in (
                        "_queued_session_messages_by_idempotency",
                        "_transcripts",
                        "_checkpoints",
                    ):
                        patch.setattr(store, name, TrackedContent(getattr(store, name)))
                elif lifecycle_case[0] == "sqlite":
                    store._read_connection.set_trace_callback(observe_sql)
                else:
                    async with store._connection() as connection, connection.cursor() as cursor:
                        cursor_type = type(cursor)
                    execute = cursor_type.execute

                    async def observed_execute(cursor, query, *args, **kwargs):
                        observe_sql(query)
                        return await execute(cursor, query, *args, **kwargs)

                    patch.setattr(cursor_type, "execute", observed_execute)
                try:
                    for expected in (original.instance_id, True, ""):
                        reads.clear()
                        with pytest.raises(SessionMessageConflict):
                            await read(expected_authorized_session_instance_id=expected)
                        assert reads == []
                    scoped = await read(expected_authorized_session_instance_id=target.instance_id)
                    assert reads, "Positive control must exercise the content lookup spy."
                    reads.clear()
                    assert await read() == scoped
                    assert reads
                finally:
                    if lifecycle_case[0] == "sqlite":
                        store._read_connection.set_trace_callback(None)
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_enqueue_authorized_target_incarnation_fences_admission_and_replay(lifecycle_case):
    async def run():
        store = await _open_store(lifecycle_case)
        try:
            original = await _session(store)
            await store.delete_session(original.id)
            target = await _session(store, original.id)
            assert target.instance_id != original.instance_id
            request = _request(target.id, "authorized-admission")

            async def state():
                return (
                    await store.load(target.id),
                    await store.load_checkpoint(target.id),
                    await store.load_transcript(target.id),
                    await store.load_events(target.id),
                    await store.inspect_session_messages(SessionMessageQuery(session_id=target.id)),
                )

            before = await state()
            for expected in (original.instance_id, True, ""):
                with pytest.raises(SessionMessageConflict):
                    await store.enqueue_session_message(
                        request, expected_authorized_target_instance_id=expected
                    )
                assert await state() == before
            accepted = await store.enqueue_session_message(
                request, expected_authorized_target_instance_id=target.instance_id
            )
            assert not accepted.replayed
            assert accepted.message.conditions == request.conditions
            store = await _reopen_store(lifecycle_case, store)
            before = await state()
            with pytest.raises(SessionMessageConflict):
                await store.enqueue_session_message(
                    request, expected_authorized_target_instance_id=original.instance_id
                )
            assert await state() == before
            replay = await store.enqueue_session_message(
                request, expected_authorized_target_instance_id=target.instance_id
            )
            assert replay.replayed and replay.event == accepted.event
            assert replay.message == accepted.message
            assert await state() == before
            # Unscoped callers retain the existing entrance, without a new keyword.
            assert (await store.enqueue_session_message(request)).replayed
            assert await state() == before
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("page_limit", [1, 2, 100])
def test_inspection_delivery_priority_cursor_includes_terminal_unreadable_and_fixed_boundary(
    lifecycle_case,
    page_limit,
):
    async def run():
        store = await _open_store(lifecycle_case)
        try:
            session = await _session(store)
            empty = await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
            assert empty.records == () and empty.next_cursor is None
            modes = ["on_idle", "next_turn", "on_idle", "next_turn", "on_idle", "next_turn"]
            accepted = [
                await store.enqueue_session_message(
                    _request(
                        session.id,
                        str(index),
                    ).model_copy(update={"delivery_mode": SessionMessageDeliveryMode(mode)})
                )
                for index, mode in enumerate(modes)
            ]
            for index in (1, 4):
                page = await store.inspect_session_messages(
                    SessionMessageQuery(session_id=session.id)
                )
                record = next(
                    record
                    for record in page.records
                    if record.queue_id == accepted[index].message.queue_id
                )
                await store.apply_session_message_action(_action(page, record))
            if lifecycle_case[0] == "memory":
                messages = store._queued_session_messages_by_idempotency[session.id]
                object.__setattr__(messages["2"], "delivery_mode", "future-mode")
                object.__setattr__(messages["3"], "conditions", 7)
            elif lifecycle_case[0] == "sqlite":
                store._connection.execute(
                    "UPDATE cayu_session_message_queue SET delivery_mode = 'future-mode' WHERE queue_id = ?",
                    (accepted[2].message.queue_id,),
                )
                store._connection.execute(
                    "UPDATE cayu_session_message_queue SET conditions_json = '7' WHERE queue_id = ?",
                    (accepted[3].message.queue_id,),
                )
                store._connection.commit()
            else:
                async with store._connection() as connection:
                    async with connection.cursor() as cursor:
                        await cursor.execute(
                            "UPDATE cayu_session_message_queue SET delivery_mode = 'future-mode' WHERE queue_id = %s",
                            (accepted[2].message.queue_id,),
                        )
                        await cursor.execute(
                            "UPDATE cayu_session_message_queue SET conditions_json = '7'::jsonb WHERE queue_id = %s",
                            (accepted[3].message.queue_id,),
                        )
                    await connection.commit()
            page = await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
            unknown = next(
                record for record in page.records if record.queue_id == accepted[2].message.queue_id
            )
            await store.apply_session_message_action(_action(page, unknown, "quarantine"))
            first = await store.inspect_session_messages(
                SessionMessageQuery(session_id=session.id, limit=page_limit)
            )
            saved_cursor = first.next_cursor
            late_next = await store.enqueue_session_message(_request(session.id, "late-next"))
            late_idle = await store.enqueue_session_message(
                _request(session.id, "late-idle").model_copy(
                    update={"delivery_mode": SessionMessageDeliveryMode.ON_IDLE},
                )
            )
            expected_events = await store.load_events(session.id)
            records = list(first.records)
            cursor = first.next_cursor
            while cursor is not None:
                assert cursor.session_instance_id == session.instance_id
                assert cursor.through_ordering_key == accepted[-1].message.ordering_key
                store = await _reopen_store(lifecycle_case, store)
                query = SessionMessageQuery(session_id=session.id, cursor=cursor, limit=page_limit)
                query = SessionMessageQuery.model_validate_json(query.model_dump_json())
                page = await store.inspect_session_messages(query.model_copy(deep=True))
                records.extend(page.records)
                cursor = page.next_cursor
            assert [record.queue_id for record in records] == [
                accepted[index].message.queue_id for index in (1, 3, 5, 0, 4, 2)
            ]
            assert [record.status for record in records] == [
                "withdrawn",
                "queued",
                "queued",
                "queued",
                "withdrawn",
                "quarantined",
            ]
            assert records[1].validity == records[-1].validity == "unreadable"
            assert await store.load_events(session.id) == expected_events
            fresh = await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
            assert [record.queue_id for record in fresh.records] == [
                accepted[1].message.queue_id,
                accepted[3].message.queue_id,
                accepted[5].message.queue_id,
                late_next.message.queue_id,
                accepted[0].message.queue_id,
                accepted[4].message.queue_id,
                late_idle.message.queue_id,
                accepted[2].message.queue_id,
            ]
            if saved_cursor is not None:
                await store.delete_session(session.id)
                replacement = await _session(store)
                assert replacement.instance_id != session.instance_id
                with pytest.raises(SessionMessageConflict):
                    await store.inspect_session_messages(
                        SessionMessageQuery(
                            session_id=session.id,
                            cursor=saved_cursor,
                        )
                    )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_inspection_revalidates_model_copy_cursor_before_storage(lifecycle_case):
    async def run():
        store = await _open_store(lifecycle_case)
        try:
            session = await _session(store)
            original = SessionMessageCursor(
                session_instance_id=session.instance_id,
                through_ordering_key=0,
                after_priority=0,
                after_ordering_key=0,
            )
            query = SessionMessageQuery(session_id=session.id, cursor=original)
            for field, value in (
                ("after_priority", True),
                ("after_priority", 3),
                ("through_ordering_key", False),
                ("after_ordering_key", 1),
            ):
                bad = original.model_copy(update={field: value})
                with pytest.raises((TypeError, ValueError)):
                    await store.inspect_session_messages(query.model_copy(update={"cursor": bad}))
            with pytest.raises(TypeError):
                await store.inspect_session_messages(query.model_copy(update={"cursor": {}}))
            other = original.model_copy(update={"session_instance_id": "wrong-instance"})
            with pytest.raises(SessionMessageConflict):
                await store.inspect_session_messages(query.model_copy(update={"cursor": other}))
            result = await store.inspect_session_messages(query.model_copy(deep=True))
            assert result.records == () and result.next_cursor is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_ordered_inspection_withdrawal_exact_replay_and_restart(lifecycle_case):
    async def run():
        store = await _open_store(lifecycle_case)
        try:
            session = await _session(store)
            requests = [_request(session.id, str(i)) for i in range(3)]
            accepted = [await store.enqueue_session_message(request) for request in requests]
            page = await store.inspect_session_messages(
                SessionMessageQuery(session_id=session.id, limit=2)
            )
            assert [record.queue_id for record in page.records] == [
                result.message.queue_id for result in accepted[:2]
            ]
            assert page.next_cursor.after_ordering_key == page.records[-1].ordering_key
            assert page.next_cursor.after_priority == 0
            action = _action(page, page.records[0])
            result = await store.apply_session_message_action(action)
            assert result.record.status == "withdrawn"
            assert result.event.type == EventType.SESSION_MESSAGE_WITHDRAWN
            assert "private steering" not in result.event.model_dump_json()
            store = await _reopen_store(lifecycle_case, store)
            replay = await store.apply_session_message_action(action)
            assert replay.replayed and replay.event == result.event
            assert replay.record == result.record
            with pytest.raises(SessionMessageConflict):
                await store.apply_session_message_action(
                    action.model_copy(update={"action": "quarantine"})
                )
            assert (await store.enqueue_session_message(requests[0])).message.status == "withdrawn"
            await store.transition_status(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            batch = await store.deliver_queued_session_messages(session.id, include_on_idle=False)
            assert action.queue_id not in {message.queue_id for message in batch.messages}
            assert len(await store.load_transcript(session.id)) == 2
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("limit", [1, 2, 100])
def test_freshness_per_message_is_batch_size_independent(lifecycle_case, limit):
    async def run():
        store = await _open_store(lifecycle_case)
        try:
            session = await _session(store)
            session = await store.transition_status(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            cursor = await store.load_transcript_cursor(session.id)
            conditions = SessionMessageConditions(
                target=SessionMessageTarget(
                    session_instance_id=session.instance_id,
                    run_epoch=session.run_epoch,
                    transcript_cursor=cursor,
                )
            )
            first = await store.enqueue_session_message(_request(session.id, "first", conditions))
            second = await store.enqueue_session_message(_request(session.id, "second", conditions))
            expired = await store.enqueue_session_message(
                _request(
                    session.id,
                    "expired",
                    SessionMessageConditions(expires_at=datetime(2000, 1, 1, tzinfo=UTC)),
                )
            )
            events, messages = [], []
            cutoff = None
            for i in range(5):
                batch = await store.deliver_queued_session_messages(
                    session.id,
                    include_on_idle=True,
                    limit=limit,
                    delivery_id=f"drain-{i}",
                    eligible_through=cutoff,
                )
                replay = await store.deliver_queued_session_messages(
                    session.id,
                    include_on_idle=True,
                    limit=limit,
                    delivery_id=f"drain-{i}",
                    eligible_through=cutoff,
                )
                assert replay == batch.model_copy(update={"replayed": True})
                cutoff = batch.eligible_through
                events.extend(batch.events)
                messages.extend(batch.messages)
                if not batch.has_more:
                    break
            assert [message.queue_id for message in messages] == [first.message.queue_id]
            page = await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
            assert [(record.queue_id, record.status) for record in page.records] == [
                (first.message.queue_id, "delivered"),
                (second.message.queue_id, "stale"),
                (expired.message.queue_id, "expired"),
            ]
            assert [event.type for event in events] == [
                EventType.SESSION_MESSAGE_DELIVERED,
                EventType.SESSION_MESSAGE_STALE,
                EventType.SESSION_MESSAGE_EXPIRED,
            ]
            with pytest.raises(SessionMessageConflict):
                await store.apply_session_message_action(_action(page, page.records[0]))
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_terminal_only_batch_does_not_start_interaction(lifecycle_case):
    async def run():
        store = await _open_store(lifecycle_case)
        try:
            session = await _session(store)
            await store.transition_status(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.enqueue_session_message(
                _request(
                    session.id,
                    "expired",
                    SessionMessageConditions(expires_at=datetime(2000, 1, 1, tzinfo=UTC)),
                )
            )
            event = Event(
                type=EventType.INTERACTION_STARTED, session_id=session.id, interaction_id="new"
            )
            batch = await store.deliver_queued_session_messages(
                session.id,
                include_on_idle=True,
                interaction_id="new",
                interaction_started_event=event,
                delivery_id="terminal-only",
            )
            assert batch.messages == ()
            assert batch.active_invocation_profile is None
            assert [item.type for item in batch.events] == [EventType.SESSION_MESSAGE_EXPIRED]
            assert batch.has_more is False
            assert await store.load_transcript(session.id) == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_source_snapshot_checked_once_but_conditions_always_bind_replay(lifecycle_case):
    async def run():
        store = await _open_store(lifecycle_case)
        try:
            source = await _session(store, "source")
            target = await _session(store)
            snapshot = await store.snapshot_session_message_source(
                source.id,
                include_transcript_digest=True,
                include_checkpoint_digest=True,
            )
            request = _request(target.id, "derived", SessionMessageConditions(source=snapshot))
            accepted = await store.enqueue_session_message(request)
            await store.checkpoint(source.id, {"changed_after_acceptance": True})
            await store.append_transcript_messages(
                source.id, [Message.text("user", "later source input")]
            )
            await store.transition_status(
                source.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            assert (await store.enqueue_session_message(request)).message == accepted.message
            with pytest.raises(SessionMessageConflict):
                await store.enqueue_session_message(
                    request.model_copy(update={"idempotency_key": "stale-source"})
                )
            with pytest.raises(ValueError):
                await store.enqueue_session_message(
                    request.model_copy(update={"conditions": SessionMessageConditions()})
                )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_source_incarnation_is_checked_before_replay_without_refreshing_snapshot(lifecycle_case):
    async def run():
        store = await _open_store(lifecycle_case)
        try:
            source = await _session(store, "source")
            target = await _session(store)
            snapshot = await store.snapshot_session_message_source(source.id)
            request = _request(target.id, "derived", SessionMessageConditions(source=snapshot))
            accepted = await store.enqueue_session_message(request)
            await store.delete_session(source.id)
            for recreate in (False, True):
                if recreate:
                    replacement = await _session(store, source.id)
                    assert replacement.instance_id != snapshot.session_instance_id
                store = await _reopen_store(lifecycle_case, store)
                events = await store.load_events(target.id)
                page = await store.inspect_session_messages(
                    SessionMessageQuery(session_id=target.id)
                )
                for retry in (
                    request,
                    request.model_copy(update={"idempotency_key": "new-derived"}),
                ):
                    with pytest.raises(SessionMessageConflict):
                        await store.enqueue_session_message(retry)
                assert await store.load_events(target.id) == events
                assert (
                    await store.inspect_session_messages(SessionMessageQuery(session_id=target.id))
                    == page
                )
                assert page.records[0].message == accepted.message
        finally:
            await _close_store(store)

    asyncio.run(run())


async def _replace_queue_conditions(case, store, accepted, encoded):
    import json

    message = accepted.message
    if case[0] == "memory":
        stored = store._queued_session_messages_by_idempotency[message.session_id][
            message.idempotency_key
        ]
        object.__setattr__(stored, "conditions", json.loads(encoded))
    elif case[0] == "sqlite":
        store._connection.execute(
            "UPDATE cayu_session_message_queue SET conditions_json = ? WHERE queue_id = ?",
            (encoded, message.queue_id),
        )
        store._connection.commit()
    else:
        async with store._connection() as connection, connection.cursor() as cursor:
            await cursor.execute(
                "UPDATE cayu_session_message_queue SET conditions_json = %s::jsonb WHERE queue_id = %s",
                (encoded, message.queue_id),
            )
            await connection.commit()


@pytest.mark.parametrize("reject_only", [False, True])
@pytest.mark.parametrize("malformed_first", [False, True])
def test_non_user_queue_record_rejected_atomically_then_quarantined(
    lifecycle_case, reject_only, malformed_first
):
    async def run():
        store = await _open_store(lifecycle_case)
        try:
            session = await _session(store)
            await store.transition_status(
                session.id, from_statuses={SessionStatus.PENDING}, to_status=SessionStatus.RUNNING
            )
            conditions = (
                SessionMessageConditions(expires_at=datetime(2000, 1, 1, tzinfo=UTC))
                if reject_only
                else SessionMessageConditions()
            )
            accepted = {}
            for key in ("bad", "good") if malformed_first else ("good", "bad"):
                request = _request(session.id, key, conditions).model_copy(
                    update={"message": Message.text("user", "private steering")}
                )
                accepted[key] = await store.enqueue_session_message(request)
            bad = accepted["bad"].message
            corrupted = Message.text("system", "private steering")
            if lifecycle_case[0] == "memory":
                stored = store._queued_session_messages_by_idempotency[session.id][
                    bad.idempotency_key
                ]
                object.__setattr__(stored, "message", corrupted)
            elif lifecycle_case[0] == "sqlite":
                store._connection.execute(
                    "UPDATE cayu_session_message_queue SET message_json = ? WHERE queue_id = ?",
                    (corrupted.model_dump_json(), bad.queue_id),
                )
                store._connection.commit()
            else:
                async with store._connection() as connection, connection.cursor() as cursor:
                    await cursor.execute(
                        "UPDATE cayu_session_message_queue SET message_json = %s::jsonb "
                        "WHERE queue_id = %s",
                        (corrupted.model_dump_json(), bad.queue_id),
                    )
                    await connection.commit()
            before = await store.inspect_session_messages(
                SessionMessageQuery(session_id=session.id)
            )
            unreadable = next(
                record for record in before.records if record.queue_id == bad.queue_id
            )
            assert unreadable.validity == "unreadable"
            events = await store.load_events(session.id)
            prior_session = await store.load(session.id)
            for _ in range(2):
                with pytest.raises((TypeError, ValueError)):
                    await store.deliver_queued_session_messages(
                        session.id,
                        include_on_idle=True,
                        reject_only=reject_only,
                        delivery_id="corrupt-role",
                    )
                assert await store.load_events(session.id) == events
                assert await store.load(session.id) == prior_session
                assert await store.load_transcript(session.id) == []
                assert (
                    await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
                    == before
                )
            await store.apply_session_message_action(_action(before, unreadable, "quarantine"))
            store = await _reopen_store(lifecycle_case, store)
            batch = await store.deliver_queued_session_messages(
                session.id,
                include_on_idle=True,
                reject_only=reject_only,
                delivery_id="corrupt-role",
            )
            assert not batch.replayed
            assert [event.type for event in batch.events] == [
                EventType.SESSION_MESSAGE_EXPIRED
                if reject_only
                else EventType.SESSION_MESSAGE_DELIVERED
            ]
            assert len(await store.load_transcript(session.id)) == (0 if reject_only else 1)
            replay = await store.deliver_queued_session_messages(
                session.id,
                include_on_idle=True,
                reject_only=reject_only,
                delivery_id="corrupt-role",
            )
            assert replay.replayed and replay.events == batch.events
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("malformed_first", [False, True])
def test_reject_only_malformed_lookahead_is_atomic_and_retryable(lifecycle_case, malformed_first):
    async def run():
        store = await _open_store(lifecycle_case)
        try:
            session = await _session(store)
            await store.transition_status(
                session.id, from_statuses={SessionStatus.PENDING}, to_status=SessionStatus.RUNNING
            )
            accepted = {}
            for key in ("bad", "expired") if malformed_first else ("expired", "bad"):
                accepted[key] = await store.enqueue_session_message(
                    _request(
                        session.id,
                        key,
                        SessionMessageConditions(expires_at=datetime(2000, 1, 1, tzinfo=UTC)),
                    )
                )
            await _replace_queue_conditions(lifecycle_case, store, accepted["bad"], "7")
            before = await store.inspect_session_messages(
                SessionMessageQuery(session_id=session.id)
            )
            events = await store.load_events(session.id)
            prior_session = await store.load(session.id)
            for _ in range(2):
                with pytest.raises((TypeError, ValueError)):
                    await store.deliver_queued_session_messages(
                        session.id,
                        include_on_idle=True,
                        reject_only=True,
                        limit=1,
                        delivery_id="failed-lookahead",
                    )
                assert await store.load_events(session.id) == events
                assert await store.load(session.id) == prior_session
                assert await store.load_transcript(session.id) == []
                assert (
                    await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
                    == before
                )
            bad = next(
                record
                for record in before.records
                if record.queue_id == accepted["bad"].message.queue_id
            )
            await store.apply_session_message_action(_action(before, bad, "quarantine"))
            store = await _reopen_store(lifecycle_case, store)
            batch = await store.deliver_queued_session_messages(
                session.id,
                include_on_idle=True,
                reject_only=True,
                limit=1,
                delivery_id="failed-lookahead",
            )
            assert not batch.replayed and batch.messages == () and not batch.has_more
            assert [event.type for event in batch.events] == [EventType.SESSION_MESSAGE_EXPIRED]
            replay = await store.deliver_queued_session_messages(
                session.id,
                include_on_idle=True,
                reject_only=True,
                limit=1,
                delivery_id="failed-lookahead",
            )
            assert replay.replayed and replay.events == batch.events
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_raw_revision_oversized_witness_cannot_alias_inline_value(lifecycle_case):
    import hashlib
    import json

    async def run():
        store = await _open_store(lifecycle_case)
        try:
            session = await _session(store)
            accepted = await store.enqueue_session_message(_request(session.id, "large-conditions"))
            encoded = json.dumps({"large": "x" * 140000})
            await _replace_queue_conditions(lifecycle_case, store, accepted, encoded)
            page = await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
            action = _action(page, page.records[0], "quarantine")
            if lifecycle_case[0] == "postgres":
                async with store._connection() as connection, connection.cursor() as cursor:
                    await cursor.execute(
                        "SELECT encode(sha256(convert_to(conditions_json::text, 'UTF8')), 'hex') "
                        "FROM cayu_session_message_queue WHERE queue_id = %s",
                        (accepted.message.queue_id,),
                    )
                    digest = (await cursor.fetchone())[0]
            else:
                digest = hashlib.sha256(encoded.encode()).hexdigest()
            replacement = {"oversized_sha256": digest}
            if lifecycle_case[0] == "sqlite":
                replacement.update(bytes=len(encoded.encode()), storage_type="text")
            await _replace_queue_conditions(
                lifecycle_case, store, accepted, json.dumps(replacement)
            )
            changed = await store.inspect_session_messages(
                SessionMessageQuery(session_id=session.id)
            )
            assert changed.records[0].revision != page.records[0].revision
            events = await store.load_events(session.id)
            with pytest.raises(SessionMessageConflict):
                await store.apply_session_message_action(action)
            assert await store.load_events(session.id) == events
            current_action = _action(changed, changed.records[0], "quarantine")
            terminal = await store.apply_session_message_action(current_action)
            store = await _reopen_store(lifecycle_case, store)
            replay = await store.apply_session_message_action(current_action)
            assert replay.replayed and replay.event == terminal.event
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_raw_revision_domain_separates_witnesses_and_inline_typed_values():
    from cayu.runtime._session_message_queue import OversizedStorageValue, raw_revision

    digest = "a" * 64
    witness = OversizedStorageValue(digest)
    values = [
        witness,
        {"oversized_sha256": digest},
        ["oversized", digest, None, None],
        datetime(2000, 1, 1, tzinfo=UTC),
        {"datetime": "2000-01-01T00:00:00+00:00"},
        b"a",
        {"bytes": "61"},
        ["bytes", "61"],
    ]
    assert len({raw_revision({"cell": value}) for value in values}) == len(values)


def test_sqlite_malformed_raw_row_quarantine_keeps_original_and_checks_revision(tmp_path):
    import json
    from copy import deepcopy

    from cayu.storage.sqlite import SQLiteSessionStore

    async def run():
        store = SQLiteSessionStore(tmp_path / "raw.sqlite")
        try:
            session = await _session(store)
            accepted = await store.enqueue_session_message(_request(session.id, "malformed"))
            page = await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
            stale_action = _action(page, page.records[0], "quarantine")
            raw = '{"secret_canary":'
            store._connection.execute(
                "UPDATE cayu_session_message_queue SET conditions_json = ? WHERE queue_id = ?",
                (raw, accepted.message.queue_id),
            )
            store._connection.commit()
            with pytest.raises(SessionMessageConflict):
                await store.apply_session_message_action(stale_action)
            page = await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
            assert page.records[0].validity == "unreadable"
            action = _action(page, page.records[0], "quarantine")
            result = await store.apply_session_message_action(action)
            assert result.record.status == "quarantined"
            assert result.record.validity == "unreadable"
            assert result.record.terminal_event_id == result.event.id
            assert "secret_canary" not in result.model_dump_json()
            assert (
                store._connection.execute(
                    "SELECT conditions_json FROM cayu_session_message_queue WHERE queue_id = ?",
                    (accepted.message.queue_id,),
                ).fetchone()[0]
                == raw
            )
            replay = await store.apply_session_message_action(action)
            assert replay.replayed and replay.event == result.event
            receipt_json = store._connection.execute(
                "SELECT terminal_json FROM cayu_session_message_queue WHERE queue_id = ?",
                (accepted.message.queue_id,),
            ).fetchone()[0]
            receipt = json.loads(receipt_json)
            events = await store.load_events(session.id)
            for field in ("version", "ordering_key", "actor", "status"):
                malformed = deepcopy(receipt)
                if field == "version":
                    malformed["version"] = True
                else:
                    malformed["event"]["payload"][field] = {
                        "ordering_key": True,
                        "actor": {"subject": "wrong-actor"},
                        "status": "withdrawn",
                    }[field]
                store._connection.execute(
                    "UPDATE cayu_session_message_queue SET terminal_json = ? WHERE queue_id = ?",
                    (json.dumps(malformed), accepted.message.queue_id),
                )
                store._connection.commit()
                await store.close()
                store = SQLiteSessionStore(tmp_path / "raw.sqlite")
                before = await store.inspect_session_messages(
                    SessionMessageQuery(session_id=session.id)
                )
                if field == "version":
                    assert before.records[0].terminal_event_id is None
                with pytest.raises(SessionMessageConflict):
                    await store.apply_session_message_action(action)
                assert (
                    await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
                    == before
                )
                assert await store.load_events(session.id) == events
            store._connection.execute(
                "UPDATE cayu_session_message_queue SET terminal_json = ? WHERE queue_id = ?",
                (receipt_json, accepted.message.queue_id),
            )
            store._connection.commit()
            replay = await store.apply_session_message_action(action)
            assert replay.replayed and replay.event == result.event
        finally:
            await store.close()

    asyncio.run(run())


def test_sqlite_missing_acceptance_event_never_falls_back_to_conditions(tmp_path):
    from cayu.storage.sqlite import SQLiteSessionStore

    async def run():
        store = SQLiteSessionStore(tmp_path / "audit-proof.sqlite")
        try:
            source = await _session(store, "audit-source")
            snapshot = await store.snapshot_session_message_source(source.id)
            target = await _session(store)
            accepted = await store.enqueue_session_message(
                _request(
                    target.id,
                    "audit",
                    SessionMessageConditions(source=snapshot),
                )
            )
            claim = await store.claim_persisted_event_side_effect(
                session_id=target.id,
                event_id=accepted.event.id,
            )
            assert claim is not None
            await store.mark_persisted_event_side_effect_delivered(claim)
            deleted = store._connection.execute(
                "DELETE FROM cayu_events WHERE session_id = ? AND event_id = ?",
                (target.id, accepted.event.id),
            )
            assert deleted.rowcount == 1
            store._connection.commit()
            await store.close()
            store = SQLiteSessionStore(tmp_path / "audit-proof.sqlite")
            assert await store.load_events(target.id) == []
            page = await store.inspect_session_messages(SessionMessageQuery(session_id=target.id))
            with pytest.raises(SessionMessageConflict):
                await store.apply_session_message_action(
                    _action(page, page.records[0], "quarantine")
                )
            assert await store.load_events(target.id) == []
            assert (
                await store.inspect_session_messages(SessionMessageQuery(session_id=target.id))
                == page
            )
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("ambiguous", [False, True])
def test_quarantine_corrupt_acceptance_pointer_uses_unique_event_owned_identity(
    lifecycle_case,
    ambiguous,
):
    from uuid import uuid4

    async def run():
        store = await _open_store(lifecycle_case)
        try:
            source = await _session(store, "pointer-source")
            snapshot = await store.snapshot_session_message_source(source.id)
            target = await _session(store)
            accepted = await store.enqueue_session_message(
                _request(
                    target.id,
                    "pointer",
                    SessionMessageConditions(source=snapshot),
                )
            )
            page = await store.inspect_session_messages(SessionMessageQuery(session_id=target.id))
            stale = _action(page, page.records[0], "quarantine")
            broken = "broken-acceptance-pointer"
            if lifecycle_case[0] == "memory":
                message = store._queued_session_messages_by_idempotency[target.id]["pointer"]
                object.__setattr__(message, "accepted_event_id", broken)
            elif lifecycle_case[0] == "sqlite":
                plan = store._connection.execute(
                    "EXPLAIN QUERY PLAN SELECT event_id FROM cayu_events "
                    "WHERE session_id = ? AND event_type = 'session.message.queued' "
                    "AND json_extract(payload_json, '$.queue_id') = ? LIMIT 2",
                    (target.id, accepted.message.queue_id),
                ).fetchall()
                assert any("idx_cayu_events_queue_acceptance" in row[3] for row in plan)
                store._connection.execute(
                    "UPDATE cayu_session_message_queue SET accepted_event_id = ? WHERE queue_id = ?",
                    (broken, accepted.message.queue_id),
                )
                store._connection.commit()
            else:
                async with store._connection() as connection:
                    async with connection.cursor() as cursor:
                        await cursor.execute(
                            "UPDATE cayu_session_message_queue SET accepted_event_id = %s "
                            "WHERE queue_id = %s",
                            (broken, accepted.message.queue_id),
                        )
                    await connection.commit()
            if ambiguous:
                await store.append_events(
                    target.id,
                    [
                        accepted.event.model_copy(
                            update={"id": str(uuid4())},
                            deep=True,
                        )
                    ],
                )
            await _acknowledge_and_prune_queue_events(store, target.id)
            store = await _reopen_store(lifecycle_case, store)
            with pytest.raises(SessionMessageConflict):
                await store.apply_session_message_action(stale)
            page = await store.inspect_session_messages(SessionMessageQuery(session_id=target.id))
            action = _action(page, page.records[0], "quarantine")
            before_events = await store.load_events(target.id)
            if ambiguous:
                with pytest.raises(SessionMessageConflict):
                    await store.apply_session_message_action(action)
                assert await store.load_events(target.id) == before_events
                assert (
                    await store.inspect_session_messages(SessionMessageQuery(session_id=target.id))
                    == page
                )
            else:
                result = await store.apply_session_message_action(action)
                assert result.event.payload["source"] == snapshot.model_dump(mode="json")
                assert result.record.message.accepted_event_id == broken
                await _acknowledge_and_prune_queue_events(store, target.id)
                store = await _reopen_store(lifecycle_case, store)
                replay = await store.apply_session_message_action(action)
                assert replay.replayed and replay.event == result.event
                assert replay.record.message.accepted_event_id == broken
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_source_digest_streaming_matches_existing_exact_snapshot():
    from cayu.runtime._session_message_queue import SourceTranscriptHasher

    records = [
        TranscriptRecord(
            index=4, interaction_id="a", message=Message.text("user", "unicode: café")
        ),
        TranscriptRecord(index=9, interaction_id=None, message=Message.text("assistant", "next")),
    ]
    for selected in ([], records):
        hasher = SourceTranscriptHasher(12)
        for record in selected:
            hasher.add(record.index, record.message)
        assert hasher.hexdigest() == fork_source_transcript_sha256(
            TranscriptSnapshot(cursor=12, records=selected)
        )


async def _acknowledge_and_prune_queue_events(store, session_id):
    """Exercise both real retention entrances after side effects cease protecting events."""
    from datetime import timedelta

    from cayu.storage.sqlite import SQLiteSessionStore

    if not isinstance(store, SQLiteSessionStore):
        return
    events = await store.load_events(session_id)
    before = {}
    for event in events:
        claim = await store.claim_persisted_event_side_effect(
            session_id=session_id,
            event_id=event.id,
        )
        if claim is not None:
            await store.mark_persisted_event_side_effect_delivered(claim)
        delivery = await store.get_persisted_event_side_effect_delivery(
            session_id=session_id,
            event_id=event.id,
        )
        assert delivery is not None and delivery.status == "delivered"
        before[event.id] = delivery
    for scope in (session_id, None):
        # A new unpinned event for each branch proves pruning really executes.
        unrelated = Event(type="custom.test.unrelated", session_id=session_id)
        await store.append_events(session_id, [unrelated])
        claim = await store.claim_persisted_event_side_effect(
            session_id=session_id,
            event_id=unrelated.id,
        )
        assert claim is not None
        await store.mark_persisted_event_side_effect_delivered(claim)
        assert (
            await store.prune_events(
                before=datetime.now(UTC) + timedelta(days=1),
                session_id=scope,
            )
            == 1
        )
        assert await store.load_events(session_id) == events
        for event_id, delivery in before.items():
            assert (
                await store.get_persisted_event_side_effect_delivery(
                    session_id=session_id,
                    event_id=event_id,
                )
                == delivery
            )


@pytest.mark.parametrize(
    "outcome,reject_only",
    [
        ("delivered", False),
        ("withdrawn", False),
        ("quarantined", False),
        ("stale", False),
        ("expired", False),
        ("stale", True),
        ("expired", True),
    ],
)
def test_source_audit_survives_conditions_mutation_retention_and_restart(
    lifecycle_case,
    outcome,
    reject_only,
):
    import json
    from datetime import timedelta

    async def run():
        store = await _open_store(lifecycle_case)
        try:
            source = await _session(store, "audit-source")
            await store.append_transcript_messages(
                source.id, [Message.text("user", "source-content-canary")]
            )
            snapshot = await store.snapshot_session_message_source(
                source.id,
                include_transcript_digest=True,
                include_checkpoint_digest=True,
            )
            target = await _session(store)
            target = await store.transition_status(
                target.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            conditions = SessionMessageConditions(
                source=snapshot,
                expires_at=(datetime.now(UTC) - timedelta(seconds=1))
                if outcome == "expired"
                else None,
                target=SessionMessageTarget(
                    session_instance_id=target.instance_id,
                    run_epoch=target.run_epoch,
                    transcript_cursor=99,
                )
                if outcome == "stale"
                else None,
            )
            request = _request(target.id, "audited", conditions)
            accepted = await store.enqueue_session_message(request)
            expected = snapshot.model_dump(mode="json")
            assert accepted.event.payload["source"] == expected
            replay = await store.enqueue_session_message(request)
            assert replay.replayed and replay.event == accepted.event
            await _acknowledge_and_prune_queue_events(store, target.id)
            store = await _reopen_store(lifecycle_case, store)
            replay = await store.enqueue_session_message(request)
            assert replay.replayed and replay.event == accepted.event

            changed = conditions.model_dump(mode="json")
            changed["source"]["session_id"] = "forged-source-canary"
            malformed = 7 if outcome == "quarantined" else changed
            kind = lifecycle_case[0]
            if kind == "memory":
                message = store._queued_session_messages_by_idempotency[target.id]["audited"]
                object.__setattr__(
                    message,
                    "conditions",
                    (
                        malformed
                        if outcome == "quarantined"
                        else SessionMessageConditions.model_validate(changed)
                    ),
                )
            elif kind == "sqlite":
                store._connection.execute(
                    "UPDATE cayu_session_message_queue SET conditions_json = ? WHERE queue_id = ?",
                    (json.dumps(malformed), accepted.message.queue_id),
                )
                store._connection.commit()
            else:
                async with store._connection() as connection:
                    async with connection.cursor() as cursor:
                        await cursor.execute(
                            "UPDATE cayu_session_message_queue SET conditions_json = %s::jsonb "
                            "WHERE queue_id = %s",
                            (json.dumps(malformed), accepted.message.queue_id),
                        )
                    await connection.commit()
            await _acknowledge_and_prune_queue_events(store, target.id)
            store = await _reopen_store(lifecycle_case, store)
            if outcome in {"withdrawn", "quarantined"}:
                page = await store.inspect_session_messages(
                    SessionMessageQuery(session_id=target.id)
                )
                if outcome == "quarantined":
                    assert page.records[0].validity == "unreadable"
                action = _action(
                    page, page.records[0], ("withdraw" if outcome == "withdrawn" else "quarantine")
                )
                result = await store.apply_session_message_action(action)
                event = result.event
                await _acknowledge_and_prune_queue_events(store, target.id)
                store = await _reopen_store(lifecycle_case, store)
                replay = await store.apply_session_message_action(action)
                assert replay.replayed and replay.event == event
            else:
                result = await store.deliver_queued_session_messages(
                    target.id,
                    include_on_idle=True,
                    delivery_id="audited-delivery",
                    reject_only=reject_only,
                )
                event = next(
                    event for event in result.events if event.type == f"session.message.{outcome}"
                )
                await _acknowledge_and_prune_queue_events(store, target.id)
                store = await _reopen_store(lifecycle_case, store)
                replay = await store.deliver_queued_session_messages(
                    target.id,
                    include_on_idle=True,
                    delivery_id="audited-delivery",
                    reject_only=reject_only,
                )
                assert replay.replayed and replay.events == result.events
            assert event.payload["source"] == expected
            assert "source-content-canary" not in event.model_dump_json()
            assert "forged-source-canary" not in event.model_dump_json()
            assert (
                next(
                    stored for stored in await store.load_events(target.id) if stored.id == event.id
                ).payload["source"]
                == expected
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_withdrawal_and_delivery_elect_one_durable_winner(lifecycle_case):
    async def run():
        store = await _open_store(lifecycle_case)
        try:
            session = await _session(store)
            await store.transition_status(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.enqueue_session_message(_request(session.id, "racing"))
            page = await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
            action = _action(page, page.records[0])
            gate = asyncio.Event()

            async def withdraw():
                await gate.wait()
                return await store.apply_session_message_action(action)

            async def deliver():
                await gate.wait()
                return await store.deliver_queued_session_messages(
                    session.id,
                    include_on_idle=False,
                    delivery_id="racing-delivery",
                )

            pending = [asyncio.create_task(withdraw()), asyncio.create_task(deliver())]
            gate.set()
            withdrawal, delivery = await asyncio.gather(*pending, return_exceptions=True)
            assert not isinstance(delivery, BaseException)
            final = await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
            if final.records[0].status == "withdrawn":
                assert not isinstance(withdrawal, BaseException)
                assert delivery.messages == ()
                assert await store.load_transcript(session.id) == []
            else:
                assert final.records[0].status == "delivered"
                assert isinstance(withdrawal, SessionMessageConflict)
                assert len(delivery.messages) == 1
                assert len(await store.load_transcript(session.id)) == 1
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_empty_queue_checkpoint_publication_retains_epoch_and_rejects_pending(
    lifecycle_case, monkeypatch
):
    from copy import deepcopy
    from datetime import timedelta

    from tests.core.test_queued_session_messages import BlockingTwoTurnProvider

    from cayu.core import AgentSpec
    from cayu.environments import Environment, EnvironmentSpec, SyncBinding
    from cayu.runtime import CayuApp
    from cayu.runtime.checkpoints import ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY
    from cayu.runtime.sessions import (
        PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY,
        SessionRunFenced,
        SessionRuntimePublicationConflict,
        runtime_publication_checkpoint_value_digest,
    )
    from cayu.workspaces import LocalWorkspace

    async def run():
        store = await _open_store(lifecycle_case)
        provider = BlockingTwoTurnProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        source = lifecycle_case[1] / "completion-source"
        target = lifecycle_case[1] / "completion-target"
        source.mkdir()
        target.mkdir()
        app.register_environment(
            Environment(
                EnvironmentSpec(name="sync"),
                workspace=LocalWorkspace(source, workspace_id="completion-source"),
                binding=SyncBinding(
                    target_workspace=LocalWorkspace(target, workspace_id="completion-target")
                ),
            ),
            default=True,
        )
        original = store.transition_status_if_no_queued_messages
        checked = False

        async def guarded(session_id, *, from_statuses, to_status, checkpoint_mutation=None):
            nonlocal checked
            # Capture the real runtime owner's prepared marker, not a synthetic fixture.
            assert checkpoint_mutation is not None
            before = await store.load(session_id)
            checkpoint = await store.load_checkpoint(session_id)
            events = await store.load_events(session_id)
            assert checkpoint is not None
            marker_key = PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY
            active_key = ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY
            assert marker_key not in checkpoint and active_key in checkpoint
            bad_mutations = [{"operations": []}]
            for kind in (
                "wrong_root",
                "extra_root",
                "delete",
                "replace",
                "malformed",
                "wrong_profile",
                "wrong_environment",
            ):
                bad = deepcopy(checkpoint_mutation)
                operation = bad["operations"][0]
                if kind in {"wrong_root", "extra_root"}:
                    overwrite = {
                        "key": active_key,
                        "action": "set",
                        "expected_value_digest": runtime_publication_checkpoint_value_digest(
                            checkpoint[active_key]
                        ),
                        "value": {"forged": True},
                    }
                    if kind == "wrong_root":
                        bad["operations"] = [overwrite]
                    else:
                        bad["operations"].append(overwrite)
                elif kind == "delete":
                    operation["key"] = active_key
                    operation["action"] = "delete"
                    operation["expected_value_digest"] = (
                        runtime_publication_checkpoint_value_digest(checkpoint[active_key])
                    )
                    operation["value"] = None
                elif kind == "replace":
                    operation["expected_value_digest"] = "a" * 64
                elif kind == "malformed":
                    operation["value"] = {"ready": True}
                elif kind == "wrong_profile":
                    operation["value"]["execution_profile_fingerprint"] = "a" * 64
                else:
                    operation["value"]["environment_name"] = "not-the-owner"
                bad_mutations.append(bad)
            for bad in bad_mutations:
                with pytest.raises((ValueError, SessionRunFenced)):
                    await original(
                        session_id,
                        from_statuses=from_statuses,
                        to_status=to_status,
                        checkpoint_mutation=bad,
                    )
                assert await store.load(session_id) == before
                assert await store.load_checkpoint(session_id) == checkpoint
                assert await store.load_events(session_id) == events

            await store.enqueue_session_message(_request(session_id, "pending-race"))
            with pytest.raises(SessionQueuedMessagesPending):
                await original(
                    session_id,
                    from_statuses=from_statuses,
                    to_status=to_status,
                    checkpoint_mutation=checkpoint_mutation,
                )
            assert await store.load_checkpoint(session_id) == checkpoint
            page = await store.inspect_session_messages(SessionMessageQuery(session_id=session_id))
            pending = next(record for record in page.records if record.status == "queued")
            await store.apply_session_message_action(_action(page, pending))
            updated = await original(
                session_id,
                from_statuses=from_statuses,
                to_status=to_status,
                checkpoint_mutation=checkpoint_mutation,
            )
            installed = await store.load_checkpoint(session_id)
            assert installed == {
                **checkpoint,
                marker_key: checkpoint_mutation["operations"][0]["value"],
            }
            assert updated.run_epoch == before.run_epoch
            with pytest.raises(SessionRuntimePublicationConflict):
                await original(
                    session_id,
                    from_statuses=from_statuses,
                    to_status=to_status,
                    checkpoint_mutation=checkpoint_mutation,
                )
            assert await store.load_checkpoint(session_id) == installed
            checked = True
            return updated

        monkeypatch.setattr(store, "transition_status_if_no_queued_messages", guarded)

        async def execute():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="queue-target",
                        messages=[Message.text("user", "initial")],
                        max_steps=1,
                    )
                )
            ]

        task = asyncio.create_task(execute())
        try:
            await asyncio.wait_for(provider.first_started.wait(), 10)
            await store.enqueue_session_message(
                _request(
                    "queue-target",
                    "expired",
                    SessionMessageConditions(expires_at=datetime.now(UTC) - timedelta(seconds=1)),
                )
            )
            provider.release_first.set()
            events = await asyncio.wait_for(task, 20)
            assert checked, [
                event.payload for event in events if event.type == EventType.SESSION_FAILED
            ]
            assert (await store.load("queue-target")).status == SessionStatus.COMPLETED
            assert len(provider.requests) == 1
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await _close_store(store)

    asyncio.run(run())


def test_sqlite_cancelled_action_keeps_dispatched_worker_owned(tmp_path):
    from cayu.storage.sqlite import SQLiteSessionStore

    async def run():
        store = SQLiteSessionStore(tmp_path / "cancel.sqlite")
        dispatched, release = threading.Event(), threading.Event()
        try:
            session = await _session(store)
            await store.enqueue_session_message(_request(session.id, "cancel-action"))
            page = await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
            action = _action(page, page.records[0])

            def barrier():
                dispatched.set()
                if not release.wait(5):
                    raise RuntimeError("Test barrier timed out.")
                return 1

            store._connection.create_function("queue_test_barrier", 0, barrier)
            store._connection.execute(
                "CREATE TEMP TRIGGER queue_action_barrier BEFORE UPDATE OF terminal_json "
                "ON cayu_session_message_queue BEGIN SELECT queue_test_barrier(); END"
            )
            owner = asyncio.create_task(store.apply_session_message_action(action))
            while not dispatched.is_set():
                await asyncio.sleep(0.001)
            owner.cancel()
            assert owner.cancelling() == 1
            competitor = asyncio.create_task(store.apply_session_message_action(action))
            await asyncio.sleep(0.02)
            assert not owner.done() and not competitor.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await owner
            assert owner.cancelled()
            replay = await competitor
            assert replay.replayed and replay.record.status == "withdrawn"
        finally:
            release.set()
            await store.close()

    asyncio.run(run())


def test_sqlite_oversized_raw_quarantine_and_post_terminal_mutation(tmp_path):
    from cayu.storage.sqlite import SQLiteSessionStore

    async def run():
        store = SQLiteSessionStore(tmp_path / "large.sqlite")
        try:
            session = await _session(store)
            await store.enqueue_session_message(_request(session.id, "large"))
            original = "secret-canary" * 30000
            store._connection.execute(
                "UPDATE cayu_session_message_queue SET content = ?",
                (original,),
            )
            store._connection.commit()
            page = await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
            assert page.records[0].validity == "unreadable"
            action = _action(page, page.records[0], "quarantine")
            result = await store.apply_session_message_action(action)
            assert "secret-canary" not in result.model_dump_json()
            assert (
                store._connection.execute(
                    "SELECT content FROM cayu_session_message_queue",
                ).fetchone()[0]
                == original
            )
            assert (await store.apply_session_message_action(action)).replayed
            store._connection.execute(
                "UPDATE cayu_session_message_queue SET content = 'replacement'"
            )
            store._connection.commit()
            with pytest.raises(SessionMessageConflict):
                await store.apply_session_message_action(action)
        finally:
            await store.close()

    asyncio.run(run())


def test_reject_only_skips_live_prefix_and_receipt_binds_mode(lifecycle_case):
    async def run():
        store = await _open_store(lifecycle_case)
        try:
            session = await _session(store)
            await store.transition_status(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            for index in range(103):
                await store.enqueue_session_message(_request(session.id, f"live-{index}"))
            for index in range(2):
                await store.enqueue_session_message(
                    _request(
                        session.id,
                        f"expired-{index}",
                        SessionMessageConditions(expires_at=datetime(2000, 1, 1, tzinfo=UTC)),
                    )
                )
            cutoff = None
            for index in range(2):
                batch = await store.deliver_queued_session_messages(
                    session.id,
                    include_on_idle=True,
                    reject_only=True,
                    limit=1,
                    delivery_id=f"reject-{index}",
                    eligible_through=cutoff,
                )
                assert batch.messages == ()
                assert len(batch.events) == 1
                assert batch.events[0].type == EventType.SESSION_MESSAGE_EXPIRED
                assert batch.has_more is (index == 0)
                with pytest.raises(ValueError):
                    await store.deliver_queued_session_messages(
                        session.id,
                        include_on_idle=True,
                        reject_only=False,
                        limit=1,
                        delivery_id=f"reject-{index}",
                        eligible_through=cutoff,
                    )
                replay = await store.deliver_queued_session_messages(
                    session.id,
                    include_on_idle=True,
                    reject_only=True,
                    limit=1,
                    delivery_id=f"reject-{index}",
                    eligible_through=cutoff,
                )
                assert replay == batch.model_copy(update={"replayed": True})
                cutoff = batch.eligible_through
            empty = await store.deliver_queued_session_messages(
                session.id,
                include_on_idle=True,
                reject_only=True,
                delivery_id="reject-empty",
            )
            assert empty.messages == empty.events == ()
            assert not empty.has_more
            with pytest.raises(ValueError):
                await store.deliver_queued_session_messages(
                    session.id,
                    include_on_idle=True,
                    delivery_id="reject-empty",
                )
            assert await store.load_transcript(session.id) == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_sqlite_revision_83_migrates_and_validates_lifecycle_schema(tmp_path):
    import sqlite3

    from cayu.storage import migrations
    from cayu.storage.sqlite import SQLiteSessionStore

    async def run():
        path = tmp_path / "migration.sqlite"
        store = SQLiteSessionStore(path)
        await store.close()
        with sqlite3.connect(path) as connection:
            connection.execute("DROP INDEX idx_cayu_events_queue_acceptance")
            connection.execute("ALTER TABLE cayu_session_message_queue DROP COLUMN conditions_json")
            connection.execute("ALTER TABLE cayu_session_message_queue DROP COLUMN terminal_json")
            connection.execute(
                "ALTER TABLE cayu_session_message_deliveries DROP COLUMN reject_only"
            )
            connection.execute("DELETE FROM cayu_schema_migrations WHERE revision = 83")
            connection.execute("PRAGMA user_version = 82")
        with pytest.raises(RuntimeError):
            SQLiteSessionStore(path, schema_mode=migrations.SchemaMode.VALIDATE)
        store = SQLiteSessionStore(path, schema_mode=migrations.SchemaMode.MIGRATE)
        try:
            session = await _session(store)
            await store.enqueue_session_message(_request(session.id, "new-schema"))
            assert (
                await store.inspect_session_messages(
                    SessionMessageQuery(session_id=session.id),
                )
            ).records[0].validity == "valid"
        finally:
            await store.close()
        with sqlite3.connect(path) as connection:
            assert connection.execute(
                "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 83"
            ).fetchone() == ("breaking", 83)
            connection.execute("ALTER TABLE cayu_session_message_queue DROP COLUMN conditions_json")
            connection.execute(
                "ALTER TABLE cayu_session_message_queue ADD COLUMN conditions_json INTEGER"
            )
        with pytest.raises(RuntimeError, match="lifecycle columns"):
            SQLiteSessionStore(path, schema_mode=migrations.SchemaMode.VALIDATE)

    asyncio.run(run())


@pytest.mark.parametrize("operation", ["withdraw", "expire"])
def test_sqlite_terminal_event_failure_rolls_back_row_and_receipt(tmp_path, operation):
    import sqlite3

    from cayu.storage.sqlite import SQLiteSessionStore

    async def run():
        store = SQLiteSessionStore(tmp_path / "rollback.sqlite")
        try:
            session = await _session(store)
            await store.transition_status(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
            await store.enqueue_session_message(
                _request(
                    session.id,
                    "atomic-terminal",
                    SessionMessageConditions(expires_at=datetime(2000, 1, 1, tzinfo=UTC)),
                )
            )
            before = await store.inspect_session_messages(
                SessionMessageQuery(session_id=session.id)
            )
            action = _action(before, before.records[0])
            store._connection.execute(
                "CREATE TEMP TRIGGER reject_terminal_event BEFORE INSERT ON cayu_events "
                "WHEN NEW.event_type IN ('session.message.withdrawn', 'session.message.expired') "
                "BEGIN SELECT RAISE(ABORT, 'injected terminal publication failure'); END"
            )

            async def mutate():
                if operation == "withdraw":
                    return await store.apply_session_message_action(action)
                return await store.deliver_queued_session_messages(
                    session.id,
                    include_on_idle=True,
                    delivery_id="failed-publication",
                )

            with pytest.raises(sqlite3.IntegrityError, match="injected terminal"):
                await mutate()
            assert (
                await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
                == before
            )
            assert await store.load_transcript(session.id) == []
            assert (
                store._connection.execute(
                    "SELECT COUNT(*) FROM cayu_session_message_deliveries",
                ).fetchone()[0]
                == 0
            )
            store._connection.execute("DROP TRIGGER reject_terminal_event")
            await mutate()
            final = await store.inspect_session_messages(SessionMessageQuery(session_id=session.id))
            assert final.records[0].status == (
                "withdrawn" if operation == "withdraw" else "expired"
            )
        finally:
            await store.close()

    asyncio.run(run())
