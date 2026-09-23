"""Recipient creation and peer delivery share one receiving transaction."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from uuid import uuid4

import pytest
from tests.core.test_participant_identity import CONTEXT, app, create, registration
from tests.core.test_peer_content import (
    QualificationPeerExposurePolicy,
    QualifiedPeerProvider,
    _delivery_request,
)
from tests.core.test_session_creation_fence import (
    _collaboration_factory,
    _fork_selection,
    _store_factory,
)

from cayu.agents import AgentSpec
from cayu.collaboration._contracts import ExactMatch
from cayu.collaboration.exports import ExportLimits, SessionExportRegistration
from cayu.collaboration.peer_content import PeerContentAppendRequest, PeerContentConflict
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import RunRequest
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    RecipientSessionCreationRequest,
)
from cayu.sessions.creation_fence import SessionCreationConflict


def future_delivery(source, sender, consumer, target, suffix):
    request = _delivery_request(
        source=source, target=source, sender=sender, consumer=consumer, suffix=suffix
    )
    key = request.append_key.model_copy(
        update={
            "target_session_id": None,
            "target_session_instance_id": None,
            "creation_target": target,
        }
    )
    return PeerContentAppendRequest.model_validate(
        {
            **request.model_dump(),
            "append_key": key.model_dump(),
            "attempt_key": {
                **request.attempt_key.model_dump(),
                "append_key": key.model_dump(),
                "target_run_epoch": 0,
            },
        }
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("mode", ["fresh", "fork"])
@pytest.mark.parametrize("winner", ["exclusion", "creation"])
@pytest.mark.parametrize("fault", ["none", "lost_ack", "cancelled"])
async def test_public_peer_creation_race(
    backend, mode, winner, fault, tmp_path, request, monkeypatch
):
    unique = uuid4().hex
    factory = _store_factory(backend, tmp_path, request)
    collaboration_factory = _collaboration_factory(backend, tmp_path, request)
    sessions, competitor = factory(), factory()
    collaboration = collaboration_factory()
    policy = QualificationPeerExposurePolicy()
    config = registration()
    providers = {}

    def application(store, collaboration_store):
        result = app(
            collaboration_store,
            config,
            session_store=store,
            session_exports=SessionExportRegistration(
                owner=policy.ref.owner,
                policy=policy,
                projectors=(),
                limits=ExportLimits(max_exports=8, max_pending=4, max_retained_bytes=65536),
            ),
        )
        provider = QualifiedPeerProvider(
            [
                [
                    ModelStreamEvent.text_delta("history"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        providers[id(result)] = provider
        result.register_provider(provider, default=True)
        result.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
        return result

    sender_app = application(sessions, collaboration)
    initialized = await sender_app.initialize_collaboration()
    _, sender_receipt = await create(sender_app, initialized, key="sender")
    _, consumer_receipt = await create(sender_app, initialized, key="consumer")
    sender, consumer = (
        sender_receipt.participants[0].reference,
        consumer_receipt.participants[0].reference,
    )
    source, _ = await sender_app.create_participant_session(
        ParticipantSessionCreationRequest(
            request=RunRequest(agent_name="reviewer", messages=[]), creation_key="source-" + unique
        ),
        participant=sender,
        context=CONTEXT,
    )
    selection = await _fork_selection(sender_app, sessions, consumer) if mode == "fork" else None
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "child")]),
        recipient=consumer,
        creation_key="child-" + uuid4().hex,
        mode=mode,
        selected_view=selection,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    targets = []
    original = sessions.create_participant_owned_session

    async def pause(*args, **kwargs):
        targets.append(kwargs["creation_target"])
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(sessions, "create_participant_owned_session", pause)
    task = asyncio.create_task(sender_app.create_recipient_session(creation, context=CONTEXT))
    try:
        await asyncio.wait_for(entered.wait(), 15)
        target = targets[0]
        delivery = future_delivery(source, sender, consumer, target, "race-" + unique)
        policy.allowed_receipts.add(delivery.occurrence.producer_receipt_id)
        other = application(competitor, collaboration)
        await other.initialize_collaboration()
        real_append = competitor.append_peer_content
        committed = asyncio.Event()

        async def interrupted_append(value, **kwargs):
            result = await real_append(value, **kwargs)
            committed.set()
            if fault == "lost_ack":
                raise ConnectionError("peer acknowledgement lost")
            if fault == "cancelled":
                await asyncio.Future()
            return result

        with monkeypatch.context() as patch:
            patch.setattr(competitor, "append_peer_content", interrupted_append)
            attempt = asyncio.create_task(other.append_peer_content(delivery, context=CONTEXT))
            await asyncio.wait_for(committed.wait(), 10)
            if fault == "cancelled":
                attempt.cancel()
                attempt.cancel()
                assert attempt.cancelling() == 2
                with pytest.raises(asyncio.CancelledError):
                    await attempt
                assert attempt.cancelled()
            elif fault == "lost_ack":
                with pytest.raises(ConnectionError, match="acknowledgement lost"):
                    await attempt
            else:
                await attempt
        if backend != "memory":
            await competitor.close()
            competitor = factory()
            other = application(competitor, collaboration)
            await other.initialize_collaboration()
        pending = await other.read_peer_content(delivery.append_key, context=CONTEXT)
        assert pending.status == "pending"
        if backend != "memory":
            location = (
                str(tmp_path / "creation-fence.sqlite")
                if backend == "sqlite"
                else request.getfixturevalue("postgres_dsn")
            )
            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-m", "tests.recovery.peer_creation_readback_worker"],
                input=json.dumps(
                    {
                        "backend": backend,
                        "location": location,
                        "request": delivery.model_dump(mode="json"),
                    }
                ),
                text=True,
                capture_output=True,
                timeout=20,
                check=True,
            )
            assert json.loads(result.stdout) == {
                "status": "pending",
                "pending": True,
                "same_target": True,
            }
        assert (await competitor.list_pending_peer_content())[0] == delivery
        for invalid_limit in (0, 65, True):
            with pytest.raises(ValueError):
                await competitor.list_pending_peer_content(limit=invalid_limit)
        assert await competitor.list_pending_peer_content(limit=1) == (delivery,)
        assert (
            await competitor.list_pending_peer_content(after_operation_key=delivery.operation_key)
            == ()
        )
        if winner == "exclusion":
            excluded = await other.exclude_peer_content(
                delivery, reason="withdrawn", context=CONTEXT
            )
            assert excluded.status == "excluded"
        release.set()
        child, _ = await asyncio.wait_for(task, 15)
        found = await competitor.read_session_creation_decision(target)
        assert isinstance(found, ExactMatch) and found.receipt.state == "created"
        assert found.receipt.settlement_acknowledged
        if winner == "creation":
            assert not (
                await competitor.list_pending_session_creations(target.receiving_owner)
            ).decisions
            assert (await competitor.list_pending_peer_content())[0] == delivery
            providers[id(other)].accept_peers = False
            with pytest.raises(ValueError, match="peer-content support"):
                await other.service_pending_peer_content(child.id, context=CONTEXT)
            assert (await competitor.read_peer_content(delivery.append_key)).status == "pending"
            providers[id(other)].accept_peers = True
            recovered = await other.service_pending_peer_content(child.id, context=CONTEXT)
            assert len(recovered) == 1 and recovered[0].status == "appended"
            result = await other.exclude_peer_content(delivery, reason="withdrawn", context=CONTEXT)
            assert result.status == "appended"
            assert result.target_session_instance_id == child.instance_id
        else:
            assert (await other.append_peer_content(delivery, context=CONTEXT)).status == "excluded"
        for field, value in (
            ("wake_policy", "ordinary_continuation"),
            ("operation_key", "conflicting"),
        ):
            with pytest.raises(PeerContentConflict):
                await other.append_peer_content(
                    delivery.model_copy(update={field: value}), context=CONTEXT
                )
        conflicting_target = target.model_copy(update={"material_commitment": "changed"})
        key = delivery.append_key.model_copy(update={"creation_target": conflicting_target})
        changed = delivery.model_copy(
            update={
                "append_key": key,
                "attempt_key": delivery.attempt_key.model_copy(update={"append_key": key}),
            }
        )
        with pytest.raises(SessionCreationConflict):
            await other.append_peer_content(changed, context=CONTEXT)
        assert await competitor.list_pending_peer_content() == ()
        results = await asyncio.gather(
            sender_app.append_peer_content(delivery, context=CONTEXT),
            other.append_peer_content(delivery, context=CONTEXT),
        )
        assert all(item.replayed for item in results)
        competing = future_delivery(source, sender, consumer, target, "competition-" + unique)
        policy.allowed_receipts.add(competing.occurrence.producer_receipt_id)
        cursor = len(await competitor.load_transcript(child.id))
        competing = competing.model_copy(
            update={
                "attempt_key": competing.attempt_key.model_copy(
                    update={"target_transcript_cursor": cursor}
                )
            }
        )
        raced = await asyncio.gather(
            sender_app.append_peer_content(competing, context=CONTEXT),
            other.exclude_peer_content(competing, reason="withdrawn", context=CONTEXT),
        )
        assert raced[0].status == raced[1].status
        assert raced[0].status in {"appended", "excluded"}
        assert sum(not item.replayed for item in raced) == 1
        if raced[0].status == "appended":
            # Addressing the same created recipient with its resolved IDs is
            # not a second opportunity to append the same successful identity.
            alias = competing.append_key.model_copy(
                update={
                    "creation_target": None,
                    "target_session_id": child.id,
                    "target_session_instance_id": child.instance_id,
                }
            )
            with pytest.raises(PeerContentConflict):
                await other.append_peer_content(
                    competing.model_copy(
                        update={
                            "operation_key": "alias-" + unique,
                            "append_key": alias,
                            "attempt_key": competing.attempt_key.model_copy(
                                update={"append_key": alias}
                            ),
                        }
                    ),
                    context=CONTEXT,
                )
        projection = competing.append_key.model_copy(update={"projection_id": "second-projection"})
        projected = await other.append_peer_content(
            competing.model_copy(
                update={
                    "operation_key": "projection-" + unique,
                    "append_key": projection,
                    "attempt_key": competing.attempt_key.model_copy(
                        update={"append_key": projection}
                    ),
                }
            ),
            context=CONTEXT,
        )
        assert projected.status == "appended"
        assert projected.queue_id != raced[0].queue_id
        if backend != "memory":
            await competitor.close()
            competitor = factory()
        replay = await competitor.read_peer_content(delivery.append_key)
        assert replay.status == ("excluded" if winner == "exclusion" else "appended")
        assert replay.append_key.creation_target == target
        if mode == "fresh":
            await competitor.delete_session(child.id)
            other = application(competitor, collaboration)
            await other.initialize_collaboration()
            replacement, _ = await other.create_recipient_session(
                RecipientSessionCreationRequest(
                    request=RunRequest(agent_name="reviewer", session_id=child.id, messages=[]),
                    recipient=consumer,
                    creation_key="unrelated-" + uuid4().hex,
                ),
                context=CONTEXT,
            )
            assert replacement.instance_id != child.instance_id
            stale = future_delivery(source, sender, consumer, target, "stale-" + unique)
            policy.allowed_receipts.add(stale.occurrence.producer_receipt_id)
            rejected = await other.append_peer_content(stale, context=CONTEXT)
            assert rejected.status == "excluded"
            assert await competitor.load_transcript(replacement.id) == []
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if backend != "memory":
            await sessions.close()
            await competitor.close()
            await collaboration.close()
