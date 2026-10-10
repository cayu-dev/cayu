"""Public host-driven reply delivery; no private admission or transcript injection."""

import json
from hashlib import sha256

from tests.core.test_peer_content import _delivery_request

from cayu.collaboration._clarification_deliveries import ClarificationDeliveryIntent
from cayu.collaboration._contracts import ExactMatch
from cayu.collaboration._request_coordinator import _initiator
from cayu.collaboration.peer_content import PeerContentExposureRequest, PeerContentPayload
from cayu.events import EventType
from cayu.messages import Message
from cayu.sessions.requests import ResumeRequest


async def continue_with_reply(
    app,
    *,
    opening,
    question,
    reply_export,
    reply_exported,
    service_context,
    actor_a,
    actor_b,
    reply_resource,
    audiences,
    initialized,
    responder,
    questioner,
    source,
    destination,
    export_policy,
    reply_text,
    payloads,
    context,
):
    decision = await app.inspect_clarification(opening, context=actor_a.context)
    assert isinstance(decision, ExactMatch)
    reply = decision.receipt.reply
    assert reply is not None
    payload = PeerContentPayload(
        text=reply_text,
        content_sha256=sha256(
            json.dumps(
                {"text": reply_text, "artifact_commitments": []},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    )
    # The receiving participant needs an explicit current grant to the producing
    # row. The accepted reply or earlier question export cannot confer that grant.
    resolution = actor_b.resolution
    entry = resolution.chain.entries[-1]
    actor_b.resolution = resolution.model_copy(
        update={
            "principal": resolution.principal.model_copy(update={"audiences": audiences}),
            "chain": resolution.chain.model_copy(
                update={
                    "entries": (
                        entry.model_copy(
                            update={
                                "audiences": audiences,
                                "resources": (*entry.resources, reply_resource),
                            }
                        ),
                    )
                }
            ),
        }
    )
    export_policy.register_export(
        reply_exported,
        payload_sha256=payload.content_sha256,
        consumer_id=questioner.participant_id,
    )
    count = len(payloads)
    stream = app.resume(
        ResumeRequest(
            session_id=destination.id,
            messages=[Message.text("user", "Continue with the authorized reply.")],
        ),
        context=context,
    )
    events = []
    delivered = None
    try:
        async for event in stream:
            events.append(event)
            if event.type != EventType.INTERACTION_STARTED:
                continue
            assert delivered is None and len(payloads) == count
            current = await app.session_store.load(destination.id)
            peer = _delivery_request(
                source=source,
                target=current,
                sender=responder,
                consumer=questioner,
                suffix="clarification-reply-" + initialized.owner.application_scope,
                source_export_receipt_id=reply_exported.event_id,
                payload=payload,
            )
            key = peer.append_key.model_copy(
                update={
                    "collaboration_namespace": initialized.namespace_incarnation,
                    "collaboration_generation": question.operation.generation,
                }
            )
            peer = peer.model_copy(
                update={
                    "append_key": key,
                    "attempt_key": peer.attempt_key.model_copy(
                        update={
                            "append_key": key,
                            "target_run_epoch": current.run_epoch,
                            "target_transcript_cursor": len(
                                await app.session_store.load_transcript(destination.id)
                            ),
                            "deadline_at_ms": question.deadline_at_ms,
                        }
                    ),
                }
            )
            export_policy.allowed_receipts.add(peer.occurrence.producer_receipt_id)
            delivery = ClarificationDeliveryIntent(
                operation=initialized.operation("reply-delivery"),
                initiator=_initiator(actor_a.context),
                question=question,
                reply=reply,
                sender=responder,
                recipient=questioner,
                export=reply_export,
                append=peer,
            )
            prepared = await app.prepare_clarification_delivery(delivery, context=service_context)
            assert prepared.status == "pending"
            delivered = await app.deliver_clarification(delivery, context=service_context)
            assert delivered.status == "appended"
            assert await app.deliver_clarification(delivery, context=service_context) == delivered
    finally:
        await stream.aclose()
    assert delivered is not None
    assert any(event.type == EventType.SESSION_COMPLETED for event in events)
    assert len(payloads) == count + 1
    assert reply_text.encode() in payloads[-1]
    # The questioner's own private state may recur, but the responder's private
    # state must never arrive through its approved visible-text reply.
    assert b"recipient-state" not in payloads[-1]
    assert b"recipient-thinking" not in payloads[-1]
    exposure_call = export_policy.calls[-1]
    exposure_request = PeerContentExposureRequest.for_model_attempt(
        append_key=key,
        append_operation_key=peer.operation_key,
        model_attempt_id=exposure_call["model_attempt_id"],
        provider_name=exposure_call["provider_name"],
        capability_version=exposure_call["capability_version"],
    )
    exposure = await app.session_store.read_peer_content_exposure(key, exposure_request.exposure_id)
    assert exposure is not None and exposure.outcome == "exposed"
    return delivery, delivered
