from __future__ import annotations

import asyncio
import warnings
from datetime import UTC, datetime, timedelta

import pytest

from cayu.core import Message
from cayu.runtime.approvals import ResolutionActor, ResolutionActorSource
from cayu.runtime.session_message_lifecycle import (
    SessionMessageActionRequest,
    SessionMessageConditions,
    SessionMessageConflict,
    SessionMessageCursor,
    SessionMessageQuery,
    SessionMessageQueueStatus,
    SessionMessageTarget,
    copy_session_message_conditions,
    session_message_rejection,
)
from cayu.runtime.sessions import (
    EnqueueSessionMessageRequest,
    InMemorySessionStore,
    RunRequest,
    SessionIdentity,
    SessionMessageInspection,
    SessionStatus,
)


@pytest.mark.parametrize(
    "field,value",
    [
        ("after_priority", True),
        ("after_priority", -1),
        ("after_priority", 3),
        ("after_ordering_key", True),
        ("after_ordering_key", 11),
        ("through_ordering_key", True),
        ("through_ordering_key", 2**63),
        ("session_instance_id", "bad\0identity"),
    ],
)
def test_cursor_rejects_invalid_authority(field, value):
    values = {
        "session_instance_id": "instance",
        "through_ordering_key": 10,
        "after_priority": 0,
        "after_ordering_key": 2,
    }
    with pytest.raises(ValueError):
        SessionMessageCursor(**{**values, field: value})


def test_query_revalidates_mutated_cursor_without_serializer_diagnostics(capsys, caplog):
    canary = "private-cursor-canary"

    class Invalid:
        def __repr__(self):
            return canary

    cursor = SessionMessageCursor(
        session_instance_id="instance",
        through_ordering_key=10,
        after_priority=0,
        after_ordering_key=2,
    )
    object.__setattr__(cursor, "after_priority", Invalid())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as failure:
            SessionMessageQuery(session_id="target", cursor=cursor)
    output = capsys.readouterr()
    assert canary not in str(failure.value) + repr(failure.value)
    assert canary not in output.out + output.err + caplog.text
    assert not caught


def test_empty_inspection_cannot_advertise_continuation_cursor():
    with pytest.raises(ValueError):
        SessionMessageInspection(
            session_id="target",
            session_instance_id="instance",
            next_cursor=SessionMessageCursor(
                session_instance_id="instance",
                through_ordering_key=10,
                after_priority=0,
                after_ordering_key=2,
            ),
        )


@pytest.mark.parametrize("delta,expected", [(-1, "expired"), (0, "expired"), (1, None)])
def test_message_expiry_uses_inclusive_store_time(delta, expected):
    now = datetime(2026, 9, 9, tzinfo=UTC)
    conditions = SessionMessageConditions(expires_at=now + timedelta(seconds=delta))
    assert (
        session_message_rejection(
            conditions, session_instance_id="instance", run_epoch=1, transcript_cursor=2, now=now
        )
        == expected
    )


@pytest.mark.parametrize("field", ["session_instance_id", "run_epoch", "transcript_cursor"])
def test_message_freshness_checks_every_target_field(field):
    values = {"session_instance_id": "instance", "run_epoch": 1, "transcript_cursor": 2}
    conditions = SessionMessageConditions(target=SessionMessageTarget(**values))
    changed = {**values, field: "different" if field == "session_instance_id" else 3}
    assert (
        session_message_rejection(conditions, **changed, now=datetime(2026, 9, 9, tzinfo=UTC))
        is SessionMessageQueueStatus.STALE
    )


def test_conditions_defensive_copy_does_not_serialize_mutated_values(capsys, caplog):
    canary = "private-mutation-canary"

    class Invalid:
        def __repr__(self):
            return canary

    target = SessionMessageTarget(session_instance_id="instance", run_epoch=1, transcript_cursor=2)
    conditions = SessionMessageConditions(target=target)
    object.__setattr__(conditions.target, "run_epoch", Invalid())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as failure:
            copy_session_message_conditions(conditions)
    output = capsys.readouterr()
    assert canary not in str(failure.value)
    assert canary not in repr(failure.value)
    assert canary not in output.out + output.err + caplog.text
    assert not caught


@pytest.mark.parametrize("field", ["run_epoch", "transcript_cursor"])
def test_target_rejects_boolean_integer_authority(field):
    values = {"session_instance_id": "instance", "run_epoch": 1, "transcript_cursor": 2}
    with pytest.raises(ValueError):
        SessionMessageTarget(**{**values, field: True})


def test_source_is_verified_at_admission_but_replay_retains_historical_observation():
    async def run():
        store = InMemorySessionStore()
        for session_id in ("source", "target"):
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
        source = await store.snapshot_session_message_source(
            "source", include_checkpoint_digest=True, include_transcript_digest=True
        )
        request = EnqueueSessionMessageRequest(
            session_id="target",
            idempotency_key="once",
            content="finding",
            delivery_mode="next_turn",
            conditions=SessionMessageConditions(source=source),
        )
        accepted = await store.enqueue_session_message(request)
        assert accepted.message.conditions.source == source
        await store.append_transcript_messages("source", [Message.text("user", "new source state")])
        replay = await store.enqueue_session_message(request)
        assert replay.replayed
        assert replay.message.conditions.source == source
        with pytest.raises(SessionMessageConflict):
            await store.enqueue_session_message(
                request.model_copy(update={"idempotency_key": "new"})
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    "action,expected", [("withdraw", "withdrawn"), ("quarantine", "quarantined")]
)
def test_memory_queue_terminal_action_is_replayable_and_prevents_delivery(action, expected):
    async def run():
        store = InMemorySessionStore()
        await store.create(
            RunRequest(agent_name="assistant", session_id="target", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        accepted = await store.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="target",
                idempotency_key="message",
                content="private message",
                delivery_mode="next_turn",
            )
        )
        page = await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
        action_request = SessionMessageActionRequest(
            session_id="target",
            session_instance_id=page.session_instance_id,
            queue_id=page.records[0].queue_id,
            expected_revision=page.records[0].revision,
            idempotency_key="action",
            action=action,
        )
        settled = await store.apply_session_message_action(action_request)
        assert settled.record.status == expected
        assert "private message" not in str(settled.event.payload)
        replay = await store.apply_session_message_action(action_request)
        assert replay.replayed and replay.event == settled.event
        assert accepted.message.status == "queued"  # caller snapshot stays detached
        await store.update_status("target", SessionStatus.RUNNING)
        delivered = await store.deliver_queued_session_messages("target", include_on_idle=True)
        assert not delivered.messages
        assert not await store.load_transcript("target")

    asyncio.run(run())


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_instance_id", "different-instance"),
        ("queue_id", "different-queue"),
        ("idempotency_key", "different-action"),
        ("expected_revision", "f" * 64),
        ("action", "quarantine"),
        (
            "requested_by",
            ResolutionActor(subject="different-actor", source=ResolutionActorSource.REQUEST),
        ),
    ],
)
def test_memory_terminal_replay_rejects_changed_action_authority(field, value):
    async def run():
        store = InMemorySessionStore()
        await store.create(
            RunRequest(agent_name="assistant", session_id="target", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        await store.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="target",
                idempotency_key="message",
                content="private message",
                delivery_mode="next_turn",
            )
        )
        before = await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
        request = SessionMessageActionRequest(
            session_id="target",
            session_instance_id=before.session_instance_id,
            queue_id=before.records[0].queue_id,
            expected_revision=before.records[0].revision,
            idempotency_key="action",
            action="withdraw",
        )
        settled = await store.apply_session_message_action(request)
        events = await store.load_events("target")
        with pytest.raises(SessionMessageConflict):
            await store.apply_session_message_action(request.model_copy(update={field: value}))
        after = await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
        assert after.records == (settled.record,)
        assert await store.load_events("target") == events
        assert not await store.load_transcript("target")

    asyncio.run(run())


def test_memory_reject_only_retains_valid_input_and_binds_replay_mode():
    async def run():
        store = InMemorySessionStore()
        await store.create(
            RunRequest(agent_name="assistant", session_id="target", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        for key, conditions in (
            ("valid", SessionMessageConditions()),
            ("expired", SessionMessageConditions(expires_at=datetime(2000, 1, 1, tzinfo=UTC))),
        ):
            await store.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id="target",
                    idempotency_key=key,
                    content=key,
                    delivery_mode="on_idle",
                    conditions=conditions,
                )
            )
        await store.update_status("target", SessionStatus.RUNNING)
        batch = await store.deliver_queued_session_messages(
            "target", include_on_idle=True, reject_only=True, delivery_id="rejection", limit=1
        )
        assert not batch.messages and not batch.has_more
        assert len(batch.events) == 1
        assert not await store.load_transcript("target")
        replay = await store.deliver_queued_session_messages(
            "target", include_on_idle=True, reject_only=True, delivery_id="rejection", limit=1
        )
        assert replay.replayed and replay.events == batch.events
        with pytest.raises(ValueError, match="different queue delivery"):
            await store.deliver_queued_session_messages(
                "target", include_on_idle=True, delivery_id="rejection", limit=1
            )
        delivered = await store.deliver_queued_session_messages("target", include_on_idle=True)
        assert tuple(message.content for message in delivered.messages) == ("valid",)

    asyncio.run(run())


def test_memory_expired_only_delivery_has_events_but_no_new_interaction():
    async def run():
        now = datetime(2026, 9, 9, tzinfo=UTC)
        store = InMemorySessionStore()
        await store.create(
            RunRequest(agent_name="assistant", session_id="target", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        await store.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="target",
                idempotency_key="message",
                content="expired message",
                delivery_mode="next_turn",
                conditions=SessionMessageConditions(expires_at=now - timedelta(days=365)),
            )
        )
        await store.update_status("target", SessionStatus.RUNNING)
        batch = await store.deliver_queued_session_messages(
            "target", include_on_idle=True, delivery_id="delivery"
        )
        assert not batch.messages and not batch.has_more
        assert [str(event.type) for event in batch.events] == ["session.message.expired"]
        assert not await store.load_transcript("target")
        replay = await store.deliver_queued_session_messages(
            "target", include_on_idle=True, delivery_id="delivery"
        )
        assert replay.replayed and replay.events == batch.events

    asyncio.run(run())


def test_memory_unreadable_message_can_be_quarantined_without_repairing_content():
    async def run():
        store = InMemorySessionStore()
        await store.create(
            RunRequest(agent_name="assistant", session_id="target", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        await store.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="target",
                idempotency_key="message",
                content="original",
                delivery_mode="next_turn",
            )
        )
        stored = store._queued_session_messages_by_idempotency["target"]["message"]
        stored.content = ""
        page = await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
        record = page.records[0]
        assert record.validity == "unreadable" and record.message is None
        request = SessionMessageActionRequest(
            session_id="target",
            session_instance_id=page.session_instance_id,
            queue_id=record.queue_id,
            expected_revision=record.revision,
            idempotency_key="quarantine",
            action="quarantine",
        )
        result = await store.apply_session_message_action(request)
        assert result.record.status == "quarantined" and result.record.validity == "unreadable"
        assert store._queued_session_messages_by_idempotency["target"]["message"].content == ""
        assert not store._pending_session_messages
        replay = await store.apply_session_message_action(request)
        assert replay.replayed and replay.event == result.event

    asyncio.run(run())


def test_memory_withdrawal_races_with_delivery_to_one_terminal_outcome():
    async def run():
        store = InMemorySessionStore()
        await store.create(
            RunRequest(agent_name="assistant", session_id="target", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        await store.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id="target",
                idempotency_key="message",
                content="once",
                delivery_mode="next_turn",
            )
        )
        page = await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
        record = page.records[0]
        request = SessionMessageActionRequest(
            session_id="target",
            session_instance_id=page.session_instance_id,
            queue_id=record.queue_id,
            expected_revision=record.revision,
            idempotency_key="withdraw",
            action="withdraw",
        )
        await store.update_status("target", SessionStatus.RUNNING)
        barrier = asyncio.Event()

        async def deliver():
            await barrier.wait()
            return await store.deliver_queued_session_messages("target", include_on_idle=True)

        async def withdraw():
            await barrier.wait()
            return await store.apply_session_message_action(request)

        tasks = [asyncio.create_task(deliver()), asyncio.create_task(withdraw())]
        barrier.set()
        delivery, action = await asyncio.gather(*tasks, return_exceptions=True)
        page = await store.inspect_session_messages(SessionMessageQuery(session_id="target"))
        if page.records[0].status == "delivered":
            assert isinstance(action, SessionMessageConflict)
            assert len(delivery.messages) == 1
            assert len(await store.load_transcript("target")) == 1
        else:
            assert page.records[0].status == "withdrawn"
            assert not delivery.messages
            assert not await store.load_transcript("target")

    asyncio.run(run())
