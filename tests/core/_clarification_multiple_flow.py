"""A second host-selected question through the existing public runtime owners."""

import pytest
from tests.core.test_peer_content import _delivery_request

from cayu import ClarificationReplyRequest
from cayu.collaboration._clarification_commands import ClarificationOpenCommand
from cayu.collaboration._clarification_deliveries import ClarificationDeliveryIntent
from cayu.collaboration._clarification_service_api import ClarificationServiceRequest
from cayu.collaboration._contracts import CollaborationConflict, ObjectRef
from cayu.collaboration._request_coordinator import _initiator
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.mandates import ResourceSelector
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._model_completion_publication import model_step_publication_from_checkpoint
from cayu.runtime._session_continuation import require_ticket_identity


async def second_question(
    application,
    initialized,
    original,
    first_service,
    source_export,
    reply_export,
    actor_a,
    actor_b,
    source_context,
    service_context,
    export_policy,
    payloads,
    service_driver=None,
):
    expected = original.expected
    before = await application.inspect_collaboration_request(expected, context=actor_a.context)
    assert before.clarification.generation == before.clarification.input_revision == 1
    question = original.question.model_copy(
        update={
            "operation": initialized.operation("second-question"),
            "generation": 2,
            "input_revision": before.clarification.input_revision,
            "input_sha256": before.clarification.input_sha256,
        }
    )
    opening = ClarificationOpenCommand(
        operation=question.operation,
        question=question,
        expected=expected,
        expected_revision=before.revision,
    )
    await application.open_clarification(opening, source=source_export, context=source_context)
    sessions = application.session_store
    retained = await sessions.load_continuation_ticket(
        first_service.ticket.session_id,
        session_instance_id=first_service.ticket.session_instance_id,
        registration_key=first_service.ticket.registration_key,
    )
    assert retained.ticket.state == "WAITING" and retained.latch is None
    require_ticket_identity(first_service.ticket, retained.ticket)
    assert len(retained.services) == 1 and retained.services[0].state == "returned"
    prior_service = retained.services[0]
    source = await sessions.load(source_export.ref.session_id)
    target = await sessions.load(first_service.delivery.append.append_key.target_session_id)
    peer = _delivery_request(
        suffix="second-question-" + initialized.owner.application_scope,
        source=source,
        target=target,
        sender=first_service.delivery.sender,
        consumer=first_service.delivery.recipient,
        source_export_receipt_id=first_service.delivery.append.occurrence.source_export_receipt_id,
        payload=first_service.delivery.append.occurrence.payload,
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
                    "target_run_epoch": target.run_epoch + 1,
                    "target_transcript_cursor": len(await sessions.load_transcript(target.id)) + 1,
                }
            ),
        }
    )
    export_policy.allowed_receipts.add(peer.occurrence.producer_receipt_id)
    delivery = ClarificationDeliveryIntent(
        operation=initialized.operation("second-question-delivery"),
        initiator=question.initiator,
        question=question,
        sender=first_service.delivery.sender,
        recipient=first_service.delivery.recipient,
        export=source_export,
        append=peer,
    )
    assert (
        await application.prepare_clarification_delivery(delivery, context=source_context)
    ).status == "pending"
    service = ClarificationServiceRequest(
        operation=initialized.operation("second-question-service"),
        initiator=_initiator(actor_a.context),
        delivery=delivery,
        ticket=retained.ticket,
        service_generation=2,
        parent_service=None,
        instruction="Answer the second authorized clarification.",
    )
    count = len(payloads)
    with pytest.raises(CollaborationConflict):
        await application.service_clarification(
            service.model_copy(update={"service_generation": 1}),
            context=service_context,
            delivery_context=source_context,
        )
    assert len(payloads) == count
    assert (
        await sessions.load_continuation_ticket(
            retained.ticket.session_id,
            session_instance_id=retained.ticket.session_instance_id,
            registration_key=retained.ticket.registration_key,
        )
        == retained
    )
    if service_driver is not None:
        from tests.core.test_participant_identity import CONTEXT

        result = await service_driver(
            application,
            service,
            context=service_context,
            delivery_context=source_context,
            recovery_context=CONTEXT,
            timeout=360,
        )
    else:
        from tests.core._clarification_service_observation import await_service_return
        from tests.core.test_participant_identity import CONTEXT

        result = await await_service_return(
            application,
            service,
            context=service_context,
            delivery_context=source_context,
            recovery_context=CONTEXT,
            timeout=120,
            nested_errors=[],
        )
    assert result.released_session_status == "completed", [
        (event.type, event.payload)
        for event in await sessions.load_events(target.id)
        if "error" in event.type or "failed" in event.type
    ]
    assert len(payloads) == count + 1
    assert (
        await application.service_clarification(
            service, context=service_context, delivery_context=source_context
        )
        == result
    )
    assert len(payloads) == count + 1
    after_service = await sessions.load_continuation_ticket(
        retained.ticket.session_id,
        session_instance_id=retained.ticket.session_instance_id,
        registration_key=retained.ticket.registration_key,
    )
    require_ticket_identity(first_service.ticket, after_service.ticket)
    assert after_service.ticket.state == "WAITING" and after_service.latch is None
    assert after_service.services[0] == prior_service
    assert [item.generation for item in after_service.services] == [1, 2]
    assert all(item.state == "returned" for item in after_service.services)

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
    resolution = actor_a.resolution
    entry = resolution.chain.entries[-1]
    actor_a.resolution = resolution.model_copy(
        update={
            "chain": resolution.chain.model_copy(
                update={
                    "entries": (entry.model_copy(update={"resources": (resource,)}),),
                }
            ),
        }
    )
    exported_reply = reply_export.model_copy(
        update={
            "ref": reply_export.ref.model_copy(
                update={
                    "operation": reply_export.ref.operation.model_copy(
                        update={"caller_key": "second-reply-export"}
                    ),
                }
            ),
            "source_indices": (pointer.source_transcript_cursor,),
        }
    )
    # The real export owner sees only this exact producing row; old source
    # grants are restored for historical peer exposure after publication.
    try:
        reply_receipt = await application.export_session(exported_reply, context=service_context)
        acceptance = await application.reply_to_clarification(
            ClarificationReplyRequest(
                operation=initialized.operation("second-reply"),
                initiator=_initiator(actor_a.context),
                expected=expected,
                service=service,
                source=exported_reply,
                production_stage_id=pointer.stage_id,
                expected_input_revision=1,
                expected_input_sha256=before.clarification.input_sha256,
            ),
            context=service_context,
        )
    finally:
        actor_a.resolution = resolution
    # Subsequent delivery must obtain a fresh current grant to this new row;
    # neither the export receipt nor the accepted reply supplies that authority.
    actor_a.resolution = resolution.model_copy(
        update={
            "chain": resolution.chain.model_copy(
                update={
                    "entries": (
                        entry.model_copy(update={"resources": (*entry.resources, resource)}),
                    ),
                }
            ),
        }
    )
    assert acceptance.input_revision == 2
    after = await application.inspect_collaboration_request(expected, context=actor_b.context)
    assert (
        after.state == "open"
        and after.clarification.generation == after.clarification.input_revision == 2
    )
    collaboration, _ = application._participant_coordinator._ready()
    async with collaboration._transaction(initialized.owner.application_scope, write=False) as tx:
        lineage = await tx.get("clarification_lineages", operation_key(question.lineage))
        assert lineage["usage"]["questions"] == lineage["usage"]["service_turns"] == 2
        assert lineage["usage"]["pending"] == 0
    assert question.policy.max_questions == question.policy.max_service_turns == 2
    third = question.model_copy(
        update={
            "operation": initialized.operation("third-question-over-limit"),
            "generation": 3,
            "input_revision": after.clarification.input_revision,
            "input_sha256": after.clarification.input_sha256,
        }
    )
    with pytest.raises(CollaborationUnavailable):
        await application.open_clarification(
            ClarificationOpenCommand(
                operation=third.operation,
                expected=expected,
                expected_revision=after.revision,
                question=third,
            ),
            source=source_export,
            context=source_context,
        )
    assert (
        await application.inspect_collaboration_request(expected, context=actor_b.context) == after
    )
    async with collaboration._transaction(initialized.owner.application_scope, write=False) as tx:
        assert await tx.get("clarification_lineages", operation_key(question.lineage)) == lineage
        assert await tx.get("operations", operation_key(third.operation)) is None
        assert await tx.get("clarification_questions", operation_key(third.operation)) is None
    assert len(payloads) == count + 1
    return opening, exported_reply, reply_receipt, resource
