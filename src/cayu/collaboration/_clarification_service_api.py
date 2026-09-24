"""Explicit host-selected clarification service through registered runtime owners."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from cayu.collaboration._clarification_deliveries import ClarificationDeliveryIntent
from cayu.collaboration._contracts import (
    ContractValue,
    Identifier,
    InitiatorBinding,
    ObjectRef,
    OperationRef,
)
from cayu.collaboration._preparation import prepare_contract, require_exact_contract
from cayu.collaboration.clarifications import MAX_CLARIFICATION_TEXT_BYTES, MAX_CLARIFICATION_TURNS
from cayu.collaboration.exports import SessionExportAccessContext
from cayu.messages import Message
from cayu.runtime._session_continuation import (
    ContinuationConflict,
    ContinuationTicket,
    ContinuationUnavailable,
    continuation_digest,
    require_ticket_identity,
)
from cayu.runtime._temporary_continuation import (
    ServiceReleasedSessionStatus,
    TemporaryServiceIntent,
    TemporaryServiceRecord,
    temporary_service_invocation_id,
    temporary_service_key,
)

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.collaboration._clarification_coordinator import ClarificationCoordinator


class ClarificationServiceRequest(ContractValue):
    """Exact host selection, not a permit, invocation or source disclosure grant.

    The delivery fixes the existing target and its incarnation. Service never
    creates a target, injects raw peer input, or retries an unresolved dispatch.
    """

    operation: OperationRef
    initiator: InitiatorBinding
    delivery: ClarificationDeliveryIntent
    ticket: ContinuationTicket
    service_generation: StrictInt = Field(ge=1, le=MAX_CLARIFICATION_TURNS)
    parent_service: OperationRef | None
    # Explicit host input uses ordinary resume validation. It is not a copy of
    # the question or an alternate peer-provenance channel.
    instruction: StrictStr = Field(min_length=1, max_length=MAX_CLARIFICATION_TEXT_BYTES)

    @field_validator("instruction")
    @classmethod
    def bounded_instruction(cls, value):
        if not value.strip() or len(value.encode("utf-8")) > MAX_CLARIFICATION_TEXT_BYTES:
            raise ValueError("Service instruction must be nonblank bounded text.")
        return value

    @model_validator(mode="after")
    def exact_selection(self):
        question = self.delivery.question
        key = self.delivery.append.append_key
        if (
            self.delivery.reply is not None
            or self.operation in (self.delivery.operation, question.operation, question.admission)
            or self.operation.application_scope != self.ticket.owner.application_scope
            or self.operation.application_scope != self.delivery.operation.application_scope
            or self.initiator.issuer != self.ticket.owner
            or self.delivery.recipient != question.responder
            or key.target_session_id is None
            or key.target_session_instance_id is None
            or key.creation_target is not None
            or self.service_generation > question.policy.max_service_turns
            or (self.parent_service is None) != (question.depth == 1)
        ):
            raise ValueError("Clarification service selection conflicts with its question.")
        return self


class ClarificationServiceReceipt(ContractValue):
    """Bounded status projection; private permit/admission material stays internal."""

    operation: OperationRef
    question: OperationRef
    state: Literal["prepared", "reserved", "admitted", "returned", "excluded"]
    session_id: Identifier
    session_instance_id: Identifier
    invocation_id: Identifier
    released_session_status: ServiceReleasedSessionStatus | None = None

    @model_validator(mode="after")
    def exact_return_report(self):
        if (self.state == "returned") != (self.released_session_status is not None):
            raise ValueError("Service report requires exact receiving return evidence.")
        return self

    @classmethod
    def from_record(cls, record: TemporaryServiceRecord):
        return cls(
            operation=record.intent.operation,
            question=record.intent.question.operation,
            state=record.state,
            session_id=record.intent.target.object_id,
            session_instance_id=record.intent.target.incarnation,
            invocation_id=record.intent.invocation_id,
            released_session_status=record.released_session_status,
        )


async def service_clarification(
    coordinator: ClarificationCoordinator,
    app: CayuApp,
    request: ClarificationServiceRequest,
    *,
    context: SessionExportAccessContext,
    delivery_context: SessionExportAccessContext | None = None,
) -> ClarificationServiceReceipt:
    """Acquire fresh disclosure per observer; retain registration in its owner."""
    from cayu.collaboration._capabilities import CapabilityDescriptor
    from cayu.collaboration._clarification_commands import ClarificationOpenReceipt
    from cayu.collaboration._clarification_delivery_store import load_delivery_in_transaction
    from cayu.collaboration._clarification_export import acquire_clarification_source
    from cayu.collaboration._request_coordinator import _initiator
    from cayu.collaboration._request_store import operation_key
    from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied
    from cayu.collaboration.base import REQUEST_FAMILY
    from cayu.runtime._checkpoint_store import (
        load_runtime_session_checkpoint_snapshot,
        runtime_checkpoint_session_store,
    )
    from cayu.runtime._session_continuation_owner import LATCH_FAMILY, SessionContinuationOwner
    from cayu.runtime._temporary_continuation_permits import TemporaryServicePermitAuthority
    from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
    from cayu.sessions.base import ResumeRequest

    requests = coordinator.requests
    redactor = requests._redactor
    request = prepare_contract(ClarificationServiceRequest, request, redactor=redactor)
    context = prepare_contract(SessionExportAccessContext, context, redactor=redactor)
    if delivery_context is not None:
        delivery_context = prepare_contract(
            SessionExportAccessContext, delivery_context, redactor=redactor
        )
        if (
            delivery_context.mandate is None
            or delivery_context.mandate.participant != request.delivery.sender
        ):
            raise CollaborationAccessDenied("Deferred delivery requires the actual sender mandate.")
        require_exact_contract(
            request.delivery.initiator, _initiator(delivery_context.mandate), redactor=redactor
        )
    if (
        context.mandate is None
        or context.mandate.participant != request.delivery.recipient
        or coordinator.common_root_enabled is not True
        or requests._resolver_ref is None
        or coordinator.exports.mandate_ref != requests._resolver_ref
    ):
        raise CollaborationAccessDenied("Service requires current responder mandate authority.")
    require_exact_contract(request.initiator, _initiator(context.mandate), redactor=redactor)
    participants = requests._participants
    store, initialized = participants._ready()
    participants._capability(store, initialized, mutation=True, family=REQUEST_FAMILY)
    participant_context = CollaborationAccessContext(principal=context.principal)
    _, grant = participants._authorize(participant_context, "request_accept")
    participants._require_refs(grant, (request.delivery.recipient, request.delivery.sender))
    requests._require_clarification_policy(request.delivery.question.policy)

    @asynccontextmanager
    async def authority_guard(_dispatch=None):
        if _dispatch is not None and (
            _dispatch.intent.selection_sha256 != continuation_digest(request)
            or _dispatch.intent.operation != request.operation
            or _dispatch.intent.question != request.delivery.question
            or _dispatch.intent.initiator != request.initiator
        ):
            raise ContinuationConflict("Service authorization received a different dispatch.")
        async with acquire_clarification_source(
            coordinator.exports,
            request.delivery.export,
            context=context,
            sender=request.delivery.sender,
            audience=request.delivery.recipient,
            expected=request.delivery.question.source,
        ) as projection:
            async with store._transaction(initialized.binding.application_scope, write=False) as tx:
                opening = prepare_contract(
                    ClarificationOpenReceipt,
                    await tx.get("operations", operation_key(request.delivery.question.operation)),
                    redactor=redactor,
                )
                require_exact_contract(
                    request.delivery.question, opening.command.question, redactor=redactor
                )
                delivery = await load_delivery_in_transaction(
                    store, tx, initialized, request.delivery, redactor=redactor
                )
                if delivery is None or not (
                    (
                        delivery.state == "settled"
                        and delivery.receipt is not None
                        and delivery.receipt.status == "appended"
                    )
                    or (delivery.state == "pending" and delivery_context is not None)
                ):
                    raise ContinuationUnavailable(
                        "Service requires its authenticated delivery responsibility."
                    )
            await coordinator._validate_source_use(
                projection, context, opening.command.expected, store, initialized
            )
            yield projection.expires_at_ms

    # Fresh read/disclosure authority is required even for public replay. The
    # registration guard is separately acquired inside the owned permit task.
    authorized = False
    async with authority_guard():
        authorized = True
    if not authorized:
        raise ContinuationUnavailable("Service authorization produced no decision.")
    owner = SessionContinuationOwner(
        store=app.session_store,
        owner=initialized.owner,
        receiver=app.collaboration_wait_latch_receiver(),
        receiver_capability=CapabilityDescriptor(
            owner=initialized.owner, mutations=(), readbacks=(LATCH_FAMILY,)
        ),
        redactor=redactor,
        temporary_permits=TemporaryServicePermitAuthority(
            store, initialized, redactor=redactor, admission_guard=authority_guard
        ),
    )
    parent = await app.session_store.load_continuation_ticket(
        request.ticket.session_id,
        session_instance_id=request.ticket.session_instance_id,
        registration_key=request.ticket.registration_key,
    )
    if parent is None:
        raise ContinuationUnavailable("Service requires its retained original wait.")
    require_ticket_identity(request.ticket, parent.ticket)
    selected = continuation_digest(request)
    raw = await app.session_store.load_session_operation(
        request.ticket.session_id, temporary_service_key(request.operation)
    )
    if raw is not None:
        record = prepare_contract(TemporaryServiceRecord, raw, redactor=redactor)
        if record.intent.selection_sha256 != selected:
            raise ContinuationConflict("Service selection conflicts with its original operation.")
        authenticated = await app.session_store._load_temporary_continuation_service(
            record.admission
        )
        if authenticated != record:
            raise ContinuationUnavailable("Service selection changed during reconstruction.")
        intent = record.intent
    else:
        key = request.delivery.append.append_key
        assert key.target_session_id is not None
        session, checkpoint = await load_runtime_session_checkpoint_snapshot(
            runtime_checkpoint_session_store(app.session_store), key.target_session_id
        )
        binding = await app.session_store.load_participant_session_binding(session.id)
        profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if (
            session.instance_id != key.target_session_instance_id
            or binding is None
            or binding.participant != request.delivery.recipient
            or binding.session_instance_id != session.instance_id
            or profile is None
        ):
            raise CollaborationAccessDenied("Service target authority is unavailable.")
        async with store._transaction(initialized.binding.application_scope, write=False) as tx:
            prepared_at_ms = await tx.now_ms()
        resume = ResumeRequest(
            session_id=session.id, messages=[Message.text("user", request.instruction)]
        )
        question = request.delivery.question
        intent = TemporaryServiceIntent(
            operation=request.operation,
            initiator=request.initiator,
            ticket=request.ticket,
            question=question,
            service_generation=request.service_generation,
            parent_service=request.parent_service,
            depth=question.depth,
            mode="same_session" if session.id == request.ticket.session_id else "side_session",
            target=ObjectRef(
                owner=initialized.owner,
                kind="session",
                object_id=session.id,
                incarnation=session.instance_id,
            ),
            participant_binding_sha256=continuation_digest(binding),
            execution_profile_sha256=profile.profile.fingerprint,
            resume_sha256=app._session_engine.work_attempt_source_request_sha256(
                resume, kind="continuation"
            ),
            invocation_id=temporary_service_invocation_id(request.operation),
            prepared_at_ms=prepared_at_ms,
            budget_binding=question.budget_binding,
            budget_authority_sha256=question.budget_authority_sha256,
            selection_sha256=selected,
            required_peer_append=request.delivery.append,
        )

    async def deliver_after_admission(_invocation):
        assert delivery_context is not None
        # The admitted runtime's owner-time deadline bounds this dependency.
        # A public observer timing out is not a negative delivery receipt.
        delivered = await coordinator.deliver(
            request.delivery, context=delivery_context, wait_for_settlement=True
        )
        if delivered.status != "appended":
            raise ContinuationUnavailable("Admitted service delivery has not been appended.")

    try:
        result = await owner.service_temporary(
            app,
            ResumeRequest(
                session_id=intent.target.object_id,
                messages=[Message.text("user", request.instruction)],
            ),
            intent,
            participant_context=participant_context,
            delivery=None if delivery_context is None else deliver_after_admission,
        )
    except (ContinuationConflict, ContinuationUnavailable) as error:
        # A competing exact preparation or lost acknowledgement may have won.
        # Observe its authenticated native record; do not retry the dispatch or
        # interpret an exception/absence as exclusion. Caller cancellation is
        # deliberately not caught here.
        retained = await app.session_store.load_session_operation(
            request.ticket.session_id, temporary_service_key(request.operation)
        )
        if retained is None:
            raise
        result = prepare_contract(TemporaryServiceRecord, retained, redactor=redactor)
        if result.intent.selection_sha256 != selected:
            raise ContinuationConflict(
                "Service operation has a conflicting retained selection."
            ) from error
        if await app.session_store._load_temporary_continuation_service(result.admission) != result:
            raise ContinuationUnavailable(
                "Service responsibility changed during reconciliation."
            ) from error
    return ClarificationServiceReceipt.from_record(result)
