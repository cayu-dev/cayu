"""Public aggregate admission, independent workers and durable capacity."""

import asyncio

import pytest
from tests.core.test_participant_identity import CONTEXT, app, create, registration
from tests.core.test_peer_content import (
    QualificationPeerExposurePolicy,
    QualifiedPeerProvider,
    _delivery_request,
)
from tests.core.test_session_creation_fence import _collaboration_factory, _store_factory

from cayu.agents import AgentSpec
from cayu.collaboration.exports import ExportLimits, SessionExportRegistration
from cayu.collaboration.peer_content import (
    PEER_CONTENT_MAX_OUTSTANDING_PER_CONSUMER,
    PeerContentConflict,
    PeerContentUnavailable,
)
from cayu.messages import Message
from cayu.runtime.session_message_lifecycle import (
    SessionMessageAccessContext,
    SessionMessageAccessPolicy,
    SessionMessageActionRequest,
    SessionMessageQuery,
)
from cayu.sessions.base import (
    EnqueueSessionMessageRequest,
    InterruptSessionRequest,
    RunRequest,
    SessionMessageDeliveryMode,
    SessionStatus,
    SessionStatusConflict,
)
from cayu.sessions.context_views import ParticipantSessionCreationRequest


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_public_peer_capacity_combines_pending_and_queue(
    backend, tmp_path, request, monkeypatch
):
    factory = _store_factory(backend, tmp_path, request)
    collaboration_factory = _collaboration_factory(backend, tmp_path, request)
    store, other_store = factory(), factory()
    collaboration = collaboration_factory()
    policy = QualificationPeerExposurePolicy()
    config = registration()

    class QueuePolicy(SessionMessageAccessPolicy):
        def authorize(self, context, **kwargs):
            return context.subject == "queue-operator"

    def application(sessions):
        result = app(
            collaboration,
            config,
            session_store=sessions,
            session_message_access_policy=QueuePolicy(),
            session_exports=SessionExportRegistration(
                owner=policy.ref.owner,
                policy=policy,
                projectors=(),
                limits=ExportLimits(max_exports=8, max_pending=4, max_retained_bytes=65536),
            ),
        )
        result.register_provider(QualifiedPeerProvider([]), default=True)
        result.register_agent(AgentSpec(name="reviewer", model="model"))
        return result

    first, second = application(store), application(other_store)
    initialized = await first.initialize_collaboration()
    await second.initialize_collaboration()
    _, source_record = await create(first, initialized, key="source")
    _, target_record = await create(first, initialized, key="target")
    sender, consumer = (
        source_record.participants[0].reference,
        target_record.participants[0].reference,
    )

    async def session(participant, key):
        value, _ = await first.create_participant_session(
            ParticipantSessionCreationRequest(
                request=RunRequest(agent_name="reviewer", messages=[Message.text("user", key)]),
                creation_key=key,
            ),
            participant=participant,
            context=CONTEXT,
        )
        return value

    source, target = await session(sender, "source"), await session(consumer, "target")

    def delivery(index, *, pending=False):
        value = _delivery_request(
            source=source,
            target=target,
            sender=sender,
            consumer=consumer,
            suffix=f"capacity-{index}",
        )
        policy.allowed_receipts.add(value.occurrence.producer_receipt_id)
        if pending:
            value = value.model_copy(
                update={
                    "attempt_key": value.attempt_key.model_copy(
                        update={"target_transcript_cursor": 1}
                    ),
                }
            )
        return value

    try:
        queue_context = SessionMessageAccessContext(subject="queue-operator")
        collision = delivery("collision")
        steering = EnqueueSessionMessageRequest(
            session_id=target.id,
            idempotency_key=collision.operation_key,
            content="ordinary steering",
            delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
        )
        original = await first.enqueue_session_message(steering, context=queue_context)
        with pytest.raises(PeerContentConflict):
            await second.append_peer_content(collision, context=CONTEXT)
        assert await store.read_peer_content(collision.append_key) is None
        replay = await first.enqueue_session_message(steering, context=queue_context)
        assert replay.message.queue_id == original.message.queue_id
        assert replay.message.content == "ordinary steering"

        # The receiving transaction must honor the same closed-admission states
        # as ordinary steering, without retaining a pending peer responsibility.
        checkpoint = await store.load_checkpoint(target.id)
        await store.checkpoint(
            target.id, {**(checkpoint or {}), "pending_completion_finalization": {}}
        )
        finalizing = delivery("finalizing")
        with pytest.raises(SessionStatusConflict, match="finalization"):
            await second.append_peer_content(finalizing, context=CONTEXT)
        assert await store.read_peer_content(finalizing.append_key) is None
        await store.checkpoint(target.id, checkpoint or {})
        await store.update_status(target.id, SessionStatus.INTERRUPTING)
        interrupting = delivery("interrupting")
        with pytest.raises(SessionStatusConflict, match="pending or running"):
            await second.append_peer_content(interrupting, context=CONTEXT)
        assert await store.read_peer_content(interrupting.append_key) is None
        await store.update_status(target.id, SessionStatus.PENDING)

        admitted = []
        for index in range(PEER_CONTENT_MAX_OUTSTANDING_PER_CONSUMER - 1):
            value = delivery(index, pending=bool(index % 2))
            result = await first.append_peer_content(value, context=CONTEXT)
            assert result.status == ("pending" if index % 2 else "appended")
            admitted.append(value)

        # Each representation is below the ceiling, but their sum has one slot.
        candidates = [delivery("a"), delivery("b", pending=True)]
        results = await asyncio.gather(
            first.append_peer_content(candidates[0], context=CONTEXT),
            second.append_peer_content(candidates[1], context=CONTEXT),
            return_exceptions=True,
        )
        assert sum(isinstance(value, PeerContentUnavailable) for value in results) == 1
        loser = candidates[
            next(i for i, value in enumerate(results) if isinstance(value, PeerContentUnavailable))
        ]
        assert await store.read_peer_content(loser.append_key) is None
        assert (
            await first.read_peer_content(
                admitted[0].append_key, expected=admitted[0], context=CONTEXT
            )
            is not None
        )
        assert (await first.append_peer_content(admitted[0], context=CONTEXT)).replayed
        # Pending retry neither takes another slot nor escapes the ceiling.
        assert (await first.append_peer_content(admitted[1], context=CONTEXT)).status == "pending"

        if backend != "memory":
            await other_store.close()
            other_store = factory()
            second = application(other_store)
            await second.initialize_collaboration()
        with pytest.raises(PeerContentUnavailable, match="capacity"):
            await second.append_peer_content(loser, context=CONTEXT)
        assert (
            await first.exclude_peer_content(admitted[1], reason="withdrawn", context=CONTEXT)
        ).status == "excluded"
        assert (await second.append_peer_content(loser, context=CONTEXT)).status in {
            "pending",
            "appended",
        }
        policy.revoked = True
        queue_context = SessionMessageAccessContext(subject="queue-operator")
        page = await second.inspect_session_messages(
            SessionMessageQuery(session_id=target.id), context=queue_context
        )
        row = next(item for item in page.records if item.message is None)
        assert row.message is None and row.status == "queued"
        action = SessionMessageActionRequest(
            session_id=page.session_id,
            session_instance_id=page.session_instance_id,
            queue_id=row.queue_id,
            expected_revision=row.revision,
            idempotency_key="capacity-withdraw",
            action="withdraw",
        )
        withdrawn = await second.apply_session_message_action(action, context=queue_context)
        assert withdrawn.record.status == "withdrawn"
        assert withdrawn.record.message is None
        assert "finding" not in withdrawn.model_dump_json()
        assert (await second.apply_session_message_action(action, context=queue_context)).replayed
        policy.revoked = False
        assert (
            await first.append_peer_content(delivery("after-withdraw"), context=CONTEXT)
        ).status == "appended"

        # Deletion removes queue rows, not exact peer receipts. Positive deletion
        # evidence must prevent those missing rows from resurrecting capacity.
        from cayu.collaboration.exports import SessionExportDenied

        policy.revoked = True
        policy.allow_cleanup = False
        with pytest.raises(SessionExportDenied):
            await first.exclude_peer_content(admitted[3], reason="withdrawn", context=CONTEXT)
        assert (await store.read_peer_content(admitted[3].append_key)).status == "pending"
        policy.allow_cleanup = True
        excluded = await second.exclude_peer_content(
            admitted[3], reason="withdrawn", context=CONTEXT
        )
        assert excluded.status == "excluded" and excluded.occurrence is None
        assert (
            await first.exclude_peer_content(admitted[3], reason="withdrawn", context=CONTEXT)
        ).replayed
        # Cleanup cannot renew append permission or leak a winning append.
        with pytest.raises(SessionExportDenied):
            await first.append_peer_content(delivery("revoked-new"), context=CONTEXT)
        historical = await first.exclude_peer_content(
            admitted[0], reason="withdrawn", context=CONTEXT
        )
        assert historical.status == "appended"
        assert historical.occurrence is None and historical.disclosure == "withheld"
        policy.revoked = False
        assert (
            await first.append_peer_content(delivery("after-cleanup"), context=CONTEXT)
        ).status == "appended"
        old_target = target
        old_receipt = await store.read_peer_content(admitted[0].append_key)
        await store.delete_session(target.id)
        if backend != "memory":
            await store.close()
            store = factory()
            first = application(store)
            await first.initialize_collaboration()
        target = await session(consumer, "replacement-target")
        assert target.instance_id != old_target.instance_id
        assert (
            await first.append_peer_content(delivery("after-delete"), context=CONTEXT)
        ).status == "appended"
        assert await store.read_peer_content(admitted[0].append_key) == old_receipt

        # Real operator interruption: pause after the durable transition, while
        # another app instance attempts admission before terminalization.
        target = await session(consumer, "operator-interruption-target")
        entered, release = asyncio.Event(), asyncio.Event()
        transition = store.transition_status_and_checkpoint

        async def pause_interrupt(session_id, **kwargs):
            result = await transition(session_id, **kwargs)
            if session_id == target.id and kwargs["to_status"] == SessionStatus.INTERRUPTING:
                entered.set()
                await release.wait()
            return result

        monkeypatch.setattr(store, "transition_status_and_checkpoint", pause_interrupt)

        async def interrupt():
            return [
                event
                async for event in first.interrupt_session(
                    InterruptSessionRequest(session_id=target.id, reason="operator stop")
                )
            ]

        task = asyncio.create_task(interrupt())
        try:
            await asyncio.wait_for(entered.wait(), timeout=10)
            # Remove the instrumentation before export capability attestation;
            # Memory shares the same native instance between both applications.
            monkeypatch.undo()
            delattr(store, "transition_status_and_checkpoint")
            rejected = delivery("operator-interrupt")
            with pytest.raises(SessionStatusConflict):
                await second.append_peer_content(rejected, context=CONTEXT)
            assert await store.read_peer_content(rejected.append_key) is None
        finally:
            release.set()
            await asyncio.wait_for(task, timeout=20)
        assert (await store.load(target.id)).status == SessionStatus.INTERRUPTED
    finally:
        if backend != "memory":
            await other_store.close()
            await store.close()
            await collaboration.close()
