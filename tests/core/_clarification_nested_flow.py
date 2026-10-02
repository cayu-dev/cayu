"""Nested public service after real parent writer release, before its return."""

import json
from hashlib import sha256

from tests.core._clarification_service_observation import await_service_return
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_peer_content import _delivery_request

from cayu.collaboration._clarification_commands import ClarificationOpenCommand
from cayu.collaboration._clarification_deliveries import ClarificationDeliveryIntent
from cayu.collaboration._clarification_service_api import ClarificationServiceRequest
from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import ObjectRef, OwnerRef
from cayu.collaboration._request_coordinator import _initiator
from cayu.collaboration.exports import SessionExportAccessContext, SessionExportRef
from cayu.collaboration.mandates import ResourceSelector
from cayu.collaboration.peer_content import PeerContentPayload
from cayu.collaboration.requests import RequestAdmissionCommand
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._session_continuation import require_ticket_identity
from cayu.sessions._model_completion_publication import model_step_publication_from_checkpoint
from cayu.vaults.redaction import SecretRedactor


async def nested_service(
    app,
    initialized,
    original,
    parent,
    export_template,
    actor_a,
    actor_b,
    source,
    target,
    export_policy,
    payloads,
):
    """All records below come from public owners; the hook only orders observation."""
    sessions = app.session_store
    root = await sessions.load_continuation_ticket(
        parent.ticket.session_id,
        session_instance_id=parent.ticket.session_instance_id,
        registration_key=parent.ticket.registration_key,
    )
    assert root.ticket.state == "SERVICING" and root.latch is None
    assert len(root.services) == 1 and root.services[0].state == "admitted"
    require_ticket_identity(parent.ticket, root.ticket)
    reverse = original.model_copy(
        update={
            "operation": initialized.operation("nested-request"),
            "sender": parent.delivery.sender,
            "target": parent.delivery.recipient,
        }
    )
    accepted = await app.accept_collaboration_request(reverse, context=actor_b.context)
    snapshot = await app.inspect_collaboration_request(accepted.expected, context=actor_a.context)
    admission = RequestAdmissionCommand(
        operation=initialized.operation("nested-clarify"),
        expected=accepted.expected,
        expected_revision=snapshot.revision,
        expected_input_revision=0,
        expected_input_sha256=clarification_commitment(accepted.expected, SecretRedactor()),
        generation=1,
        decision="clarify",
        evidence=(),
        initiator=_initiator(actor_a.context),
    )
    await app.admit_collaboration_request(admission, context=actor_a.context)
    checkpoint = await runtime_checkpoint_session_store(sessions).load_checkpoint(target.id)
    pointer = model_step_publication_from_checkpoint(checkpoint)
    assert pointer is not None
    resource = ResourceSelector(
        resource=ObjectRef(
            owner=initialized.owner,
            kind="session_transcript_row",
            object_id=target.id,
            incarnation=target.instance_id,
            revision=pointer.source_transcript_cursor + 1,
        )
    )
    audience = OwnerRef(
        application_scope=initialized.owner.application_scope,
        owner_id=parent.delivery.sender.participant_id,
        incarnation=parent.delivery.sender.incarnation,
    )
    for actor in (actor_a, actor_b):
        resolution = actor.resolution
        entry = resolution.chain.entries[-1]
        actions = tuple(dict.fromkeys((*entry.actions, "source", "publish", "readback", "expose")))
        audiences = tuple(dict.fromkeys((*entry.audiences, audience)))
        actor.resolution = resolution.model_copy(
            update={
                "principal": resolution.principal.model_copy(
                    update={"actions": actions, "audiences": audiences}
                ),
                "chain": resolution.chain.model_copy(
                    update={
                        "entries": (
                            entry.model_copy(
                                update={
                                    "actions": actions,
                                    "audiences": audiences,
                                    "resources": (*entry.resources, resource),
                                }
                            ),
                        )
                    }
                ),
            }
        )
    source_context = SessionExportAccessContext(principal="operator", mandate=actor_a.context)
    receiver_context = SessionExportAccessContext(principal="operator", mandate=actor_b.context)
    namespace = await app.initialize_session_exports(target.id, context=source_context)
    export = export_template.model_copy(
        update={
            "ref": SessionExportRef(
                session_id=target.id,
                session_instance_id=target.instance_id,
                operation=initialized.operation("nested-question-export").model_copy(
                    update={
                        "namespace_incarnation": namespace.namespace_incarnation,
                        "generation": namespace.generation,
                    }
                ),
            ),
            "source_indices": (pointer.source_transcript_cursor,),
            "audience": audience,
        }
    )
    exported = await app.export_session(export, context=source_context)
    projection = await app.inspect_clarification_source(
        export,
        sender=parent.delivery.recipient,
        audience=parent.delivery.sender,
        context=source_context,
    )
    snapshot = await app.inspect_collaboration_request(accepted.expected, context=actor_a.context)
    question = parent.delivery.question.model_copy(
        update={
            "operation": initialized.operation("nested-question"),
            "request": accepted.expected.intent.selection.reference,
            "request_sha256": clarification_commitment(accepted.expected, SecretRedactor()),
            "admission": admission.operation,
            "initiator": _initiator(actor_a.context),
            "responder": parent.delivery.sender,
            "generation": 1,
            "input_revision": 0,
            "input_sha256": clarification_commitment(accepted.expected, SecretRedactor()),
            "parent_question": parent.delivery.question.operation,
            "depth": 2,
            "source": projection,
        }
    )
    opening = ClarificationOpenCommand(
        operation=question.operation,
        question=question,
        expected=accepted.expected,
        expected_revision=snapshot.revision,
    )
    await app.open_clarification(opening, source=export, context=source_context)
    current_target = await sessions.load(source.id)
    text = "Use API v2."
    payload = PeerContentPayload(
        text=text,
        content_sha256=sha256(
            json.dumps(
                {"text": text, "artifact_commitments": []}, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    )
    peer = _delivery_request(
        suffix="nested-question-" + initialized.owner.application_scope,
        source=await sessions.load(target.id),
        target=current_target,
        sender=parent.delivery.recipient,
        consumer=parent.delivery.sender,
        source_export_receipt_id=exported.event_id,
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
                    "deadline_at_ms": question.deadline_at_ms,
                    "target_run_epoch": current_target.run_epoch + 1,
                    "target_transcript_cursor": len(await sessions.load_transcript(source.id)) + 1,
                }
            ),
        }
    )
    export_policy.register_export(
        exported,
        payload_sha256=payload.content_sha256,
        consumer_id=parent.delivery.sender.participant_id,
    )
    export_policy.allowed_receipts.add(peer.occurrence.producer_receipt_id)
    delivery = ClarificationDeliveryIntent(
        operation=initialized.operation("nested-question-delivery"),
        initiator=question.initiator,
        question=question,
        sender=parent.delivery.recipient,
        recipient=parent.delivery.sender,
        export=export,
        append=peer,
    )
    assert (
        await app.prepare_clarification_delivery(delivery, context=source_context)
    ).status == "pending"
    service = ClarificationServiceRequest(
        operation=initialized.operation("nested-service"),
        initiator=_initiator(actor_b.context),
        delivery=delivery,
        ticket=root.ticket,
        service_generation=2,
        parent_service=parent.operation,
        instruction="Answer the nested authorized question.",
    )
    count = len(payloads)
    try:
        # The admitted child has a 300-second execution budget, followed by
        # native return/foreign acknowledgement. Observe that complete scope.
        result = await await_service_return(
            app,
            service,
            context=receiver_context,
            delivery_context=source_context,
            recovery_context=CONTEXT,
            timeout=360,
            nested_errors=[],
        )
    except TimeoutError as error:
        retained = await sessions.load_continuation_ticket(
            root.ticket.session_id,
            session_instance_id=root.ticket.session_instance_id,
            registration_key=root.ticket.registration_key,
        )
        failures = [
            (event.type, event.payload)
            for event in await sessions.load_events(source.id)
            if "failed" in event.type or "error" in event.type
        ]
        error.add_note(
            f"Native stack: {[item.state for item in retained.services]}; provider requests: {len(payloads) - count}; failures: {failures!r}"
        )
        raise
    assert result.released_session_status == "completed"
    assert len(payloads) == count + 1
    assert b"Use API v2." in payloads[-1]
    assert b"recipient-state" not in payloads[-1]
    assert b"recipient-thinking" not in payloads[-1]
    assert (
        await app.service_clarification(
            service, context=receiver_context, delivery_context=source_context
        )
        == result
    )
    assert len(payloads) == count + 1
    retained = await sessions.load_continuation_ticket(
        root.ticket.session_id,
        session_instance_id=root.ticket.session_instance_id,
        registration_key=root.ticket.registration_key,
    )
    assert [item.state for item in retained.services] == ["admitted", "returned"]
    assert retained.ticket.state == "SERVICING" and retained.latch is None
    from tests.core._clarification_nested_reply import accept_nested_reply

    await accept_nested_reply(
        app,
        initialized,
        accepted,
        question,
        service,
        source,
        actor_b,
        receiver_context,
        export_template,
        payloads,
    )
    return result
