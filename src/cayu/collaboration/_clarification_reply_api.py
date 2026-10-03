"""Source-authenticated reply election through the existing request transaction."""

from __future__ import annotations

from cayu.collaboration._clarification_commands import ClarificationReplyCommand
from cayu.collaboration._clarification_export import acquire_clarification_source
from cayu.collaboration._clarification_reply_production import authenticate_reply_production
from cayu.collaboration._clarification_service_api import ClarificationServiceRequest
from cayu.collaboration._clarification_services import ClarificationServiceRecord
from cayu.collaboration._clarification_store import reply_in_transaction
from cayu.collaboration._contracts import (
    CollaborationConflict,
    ContractValue,
    Identifier,
    InitiatorBinding,
    ObjectRef,
    OperationRef,
)
from cayu.collaboration._preparation import prepare_contract, require_exact_contract
from cayu.collaboration._request_coordinator import _initiator
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.clarifications import ClarificationReply, Commitment, InputRevision
from cayu.collaboration.exports import SessionExportAccessContext, SessionExportRequest
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestCommand
from cayu.runtime._temporary_continuation_permits import TemporaryServicePermitAuthority
from cayu.sessions._session_continuation import continuation_digest
from cayu.sessions._temporary_continuation import (
    TemporaryServiceRecord,
    temporary_service_key,
)


class ClarificationReplyRequest(ContractValue):
    operation: OperationRef
    initiator: InitiatorBinding
    expected: RequestCommand
    service: ClarificationServiceRequest
    source: SessionExportRequest
    production_stage_id: Identifier
    expected_input_revision: InputRevision
    expected_input_sha256: Commitment


class ClarificationReplyAcceptance(ContractValue):
    """Public election evidence, without executable admission/permit material."""

    operation: OperationRef
    question: OperationRef
    input_revision: InputRevision
    input_sha256: Commitment
    event_id: Identifier


async def accept_reply(coordinator, app, request, *, context):
    requests = coordinator.requests
    redactor = requests._redactor
    request = prepare_contract(ClarificationReplyRequest, request, redactor=redactor)
    context = prepare_contract(SessionExportAccessContext, context, redactor=redactor)
    question = request.service.delivery.question
    selected = request.expected.intent.selection
    if (
        context.mandate is None
        or context.mandate.participant != question.responder
        or requests._resolver_ref is None
        or coordinator.exports.mandate_ref != requests._resolver_ref
        or question.request != selected.reference
        or question.responder != selected.sender.reference
        or request.source.source_selection != "assistant_visible_text_v1"
    ):
        raise CollaborationAccessDenied("Reply requires current responder and source authority.")
    require_exact_contract(request.initiator, _initiator(context.mandate), redactor=redactor)
    participants = requests._participants
    store, initialized = participants._ready()
    participants._capability(store, initialized, mutation=True, family=REQUEST_FAMILY)
    _, grant = participants._authorize(
        CollaborationAccessContext(principal=context.principal), "request_accept"
    )
    participants._require_refs(grant, (selected.sender.reference, selected.recipient.reference))
    requests._require_clarification_policy(question.policy)
    async with acquire_clarification_source(
        coordinator.exports,
        request.source,
        context=context,
        sender=question.responder,
        audience=selected.recipient.reference,
    ) as projection:
        await coordinator._validate_source_use(
            projection, context, request.expected, store, initialized
        )
        async with store._transaction(initialized.binding.application_scope, write=False) as tx:
            raw = await tx.get("clarification_services", operation_key(request.service.operation))
        if raw is None:
            raise CollaborationUnavailable("Reply service registration is unavailable.")
        registered = prepare_contract(ClarificationServiceRecord, raw, redactor=redactor)
        intent = registered.dispatch.intent
        if (
            intent.selection_sha256 != continuation_digest(request.service)
            or intent.question != question
            or request.source.ref.session_id != intent.target.object_id
            or request.source.ref.session_instance_id != intent.target.incarnation
        ):
            raise CollaborationConflict("Reply source or service selection conflicts.")
        authority = TemporaryServicePermitAuthority(
            store, initialized, redactor=redactor, owners=app._request_coordinator.owners
        )
        if await authority.lookup(intent) != registered:
            raise CollaborationUnavailable("Reply service responsibility changed.")
        raw_native = await app.session_store.load_session_operation(
            intent.ticket.session_id, temporary_service_key(intent.operation)
        )
        if raw_native is None:
            raise CollaborationUnavailable("Reply service receiving evidence is unavailable.")
        native = prepare_contract(TemporaryServiceRecord, raw_native, redactor=redactor)
        if native.intent != intent or native.admission.dispatch != registered.dispatch:
            raise CollaborationConflict("Reply native service has different authority.")
        await authority.authenticate(native.acknowledged_admission)
        production = await authenticate_reply_production(
            app.session_store,
            native,
            stage_id=request.production_stage_id,
            source_indices=request.source.source_indices,
        )
        if production.source_commitment != projection.receipt.expected.intent.source_commitment:
            raise CollaborationConflict("Reply export was not produced by the selected service.")
        assert native.execution is not None
        reply = ClarificationReply(
            operation=request.operation,
            question=question.operation,
            question_sha256=continuation_digest(question),
            question_generation=question.generation,
            question_input_revision=question.input_revision,
            expected_input_revision=request.expected_input_revision,
            expected_input_sha256=request.expected_input_sha256,
            initiator=request.initiator,
            responder=question.responder,
            service=intent.operation,
            service_generation=intent.service_generation,
            service_session=intent.target,
            invocation=ObjectRef(
                owner=intent.target.owner,
                kind="invocation",
                object_id=intent.invocation_id,
                incarnation=intent.target.incarnation,
                revision=native.execution.run_epoch,
            ),
            admission_sha256=continuation_digest(native.acknowledged_admission),
            production_stage_id=production.stage_id,
            production_sha256=production.publication_sha256,
            source=projection.source,
        )
        command = ClarificationReplyCommand(
            operation=request.operation, expected=request.expected, reply=reply
        )
        async with store._transaction(initialized.binding.application_scope, write=True) as tx:
            if await tx.now_ms() >= projection.expires_at_ms:
                raise CollaborationAccessDenied("Reply disclosure authority expired.")
            # Disablement/replacement orders against election in this owner.
            # Already admitted execution does not renew disclosure permission.
            for participant in (selected.sender, selected.recipient):
                current = await store._participant(
                    tx, participant.reference, initialized.owner, redactor
                )
                if current.lifecycle != "active" or (
                    current.lifecycle_revision,
                    current.configuration_revision,
                    current.admission_generation,
                ) != (
                    participant.lifecycle_revision,
                    participant.configuration_revision,
                    participant.admission_generation,
                ):
                    raise CollaborationAccessDenied("Reply participant authority changed.")
            receipt = await reply_in_transaction(store, tx, initialized, command, redactor=redactor)
        assert receipt.decision.input is not None
        return ClarificationReplyAcceptance(
            operation=request.operation,
            question=question.operation,
            input_revision=receipt.decision.input.revision,
            input_sha256=continuation_digest(receipt.decision.input),
            event_id=receipt.event.id,
        )
    raise CollaborationUnavailable("Reply authorization produced no decision.")
