"""Administrative service reconciliation, never renewed disclosure or dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu.collaboration._capabilities import CapabilityDescriptor
from cayu.collaboration._clarification_recovery_types import (
    ClarificationPendingService,
    ClarificationPendingServicePage,
    ClarificationPendingServiceQuery,
    ClarificationServiceInspection,
    ClarificationServiceInspectionPage,
    ClarificationServiceRecovery,
)
from cayu.collaboration._clarification_service_api import (
    ClarificationServiceReceipt,
    ClarificationServiceRequest,
)
from cayu.collaboration._contracts import CollaborationConflict, ContractValue
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.clarifications import ClarificationDueCursor
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.runtime._session_continuation import (
    ContinuationConflict,
    ContinuationTicket,
    ContinuationUnavailable,
    continuation_digest,
    require_ticket_identity,
)
from cayu.runtime._session_continuation_owner import LATCH_FAMILY, SessionContinuationOwner
from cayu.runtime._temporary_continuation import (
    TemporaryServiceAdmission,
    TemporaryServiceRecord,
    reference_for_service,
    temporary_service_key,
)
from cayu.runtime._temporary_continuation_permits import TemporaryServicePermitAuthority

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.collaboration._clarification_coordinator import ClarificationCoordinator


class ServiceRecoveryInput(ContractValue):
    request: ClarificationServiceRequest | ClarificationServiceRecovery


async def inspect_services(coordinator, app, ticket, *, context, cursor=None, limit=32):
    participants = coordinator.requests._participants
    redactor = coordinator.requests._redactor
    ticket = prepare_contract(ContinuationTicket, ticket, redactor=redactor)
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    query = prepare_contract(
        ClarificationPendingServiceQuery, {"cursor": cursor, "limit": limit}, redactor=redactor
    )
    store, initialized = participants._ready()
    participants._capability(store, initialized, mutation=False, family=REQUEST_FAMILY)
    _, grant = participants._authorize(context, "request_readback")
    participants._require_refs(grant, (), create=True)
    if ticket.owner != initialized.owner:
        raise CollaborationAccessDenied("Service inspection belongs to another owner.")
    if not app.session_store._supports_session_continuation_protocol():
        raise CollaborationUnavailable("Session store does not qualify native service inspection.")
    parent = await app.session_store.load_continuation_ticket(
        ticket.session_id,
        session_instance_id=ticket.session_instance_id,
        registration_key=ticket.registration_key,
    )
    if parent is None:
        raise CollaborationUnavailable("Original wait receiving evidence is unavailable.")
    require_ticket_identity(ticket, parent.ticket)
    from cayu.collaboration._clarification_records import due_cursor_key

    if query.cursor is not None and (
        query.cursor.operation.application_scope != initialized.binding.application_scope
        or query.cursor.operation.namespace_incarnation != initialized.namespace_incarnation
    ):
        raise CollaborationConflict("Service inspection cursor belongs to another owner.")
    items = []
    for reference in parent.services:
        raw = await app.session_store.load_session_operation(ticket.session_id, reference.key)
        if raw is None:
            raise CollaborationUnavailable("Native service index lost its retained record.")
        record = prepare_contract(TemporaryServiceRecord, raw, redactor=redactor)
        require_ticket_identity(ticket, record.intent.ticket)
        if (
            reference_for_service(record, parent.services) != reference
            or await app.session_store._load_temporary_continuation_service(record.admission)
            != record
            or record.intent.selection_sha256 is None
        ):
            raise CollaborationUnavailable("Native service inspection has conflicting evidence.")
        item = ClarificationServiceInspection(
            recovery=ClarificationServiceRecovery(
                operation=record.intent.operation,
                session_id=ticket.session_id,
                session_instance_id=ticket.session_instance_id,
                selection_sha256=record.intent.selection_sha256,
                dispatch_sha256=continuation_digest(record.admission.dispatch),
            ),
            question=record.intent.question.operation,
            deadline_at_ms=record.intent.question.deadline_at_ms,
            state=record.state,
        )
        position = ClarificationDueCursor(
            deadline_at_ms=item.deadline_at_ms, operation=item.recovery.operation
        )
        if query.cursor is None or due_cursor_key(position) > due_cursor_key(query.cursor):
            items.append((due_cursor_key(position), item))
    items.sort(key=lambda entry: entry[0])
    return participants._page(
        ClarificationServiceInspectionPage,
        "items",
        tuple(item for _, item in items[: query.limit]),
        lambda item: {
            "deadline_at_ms": item.deadline_at_ms,
            "operation": item.recovery.operation.model_dump(mode="json"),
        },
        query.limit,
    )


async def pending_services(
    coordinator: ClarificationCoordinator,
    *,
    context: CollaborationAccessContext,
    cursor: ClarificationDueCursor | None = None,
    limit: int = 32,
) -> ClarificationPendingServicePage:
    participants = coordinator.requests._participants
    redactor = coordinator.requests._redactor
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    query = prepare_contract(
        ClarificationPendingServiceQuery, {"cursor": cursor, "limit": limit}, redactor=redactor
    )
    store, initialized = participants._ready()
    participants._capability(store, initialized, mutation=False, family=REQUEST_FAMILY)
    _, grant = participants._authorize(context, "request_readback")
    participants._require_refs(grant, (), create=True)
    if query.cursor is not None and (
        query.cursor.operation.application_scope != initialized.binding.application_scope
        or query.cursor.operation.namespace_incarnation != initialized.namespace_incarnation
    ):
        raise CollaborationConflict("Pending service cursor belongs to another owner.")
    from cayu.collaboration._clarification_service_store import discover_services_in_transaction

    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        records = await discover_services_in_transaction(
            store, tx, initialized, after=query.cursor, limit=query.limit, redactor=redactor
        )
    items = []
    for record in records:
        selection = record.dispatch.intent.selection_sha256
        if selection is None:
            raise CollaborationUnavailable("Pending service lacks its public selection commitment.")
        items.append(
            ClarificationPendingService(
                recovery=ClarificationServiceRecovery(
                    operation=record.dispatch.intent.operation,
                    session_id=record.dispatch.intent.ticket.session_id,
                    session_instance_id=record.dispatch.intent.ticket.session_instance_id,
                    selection_sha256=selection,
                    dispatch_sha256=continuation_digest(record.dispatch),
                ),
                question=record.dispatch.intent.question.operation,
                deadline_at_ms=record.dispatch.intent.question.deadline_at_ms,
            )
        )
    return participants._page(
        ClarificationPendingServicePage,
        "items",
        tuple(items),
        lambda item: {
            "deadline_at_ms": item.deadline_at_ms,
            "operation": item.recovery.operation.model_dump(mode="json"),
        },
        query.limit,
    )


async def _read_service_record(
    coordinator: ClarificationCoordinator,
    app: CayuApp,
    request: ClarificationServiceRequest | ClarificationServiceRecovery,
    *,
    context: CollaborationAccessContext,
    mutation: bool,
):
    participants = coordinator.requests._participants
    redactor = coordinator.requests._redactor
    request = prepare_contract(
        ServiceRecoveryInput, {"request": request}, redactor=redactor
    ).request
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    store, initialized = participants._ready()
    participants._capability(store, initialized, mutation=mutation, family=REQUEST_FAMILY)
    _, grant = participants._authorize(
        context, "request_control" if mutation else "request_readback"
    )
    # Public reconciliation is scope-wide maintenance. Read-only host settlement
    # instead authenticates the exact participating references below.
    if mutation:
        participants._require_refs(grant, (), create=True)
    if request.operation.application_scope != initialized.binding.application_scope:
        raise CollaborationAccessDenied("Service belongs to another receiving owner.")
    if isinstance(request, ClarificationServiceRequest):
        participants._require_refs(grant, (request.delivery.sender, request.delivery.recipient))
        if request.ticket.owner != initialized.owner:
            raise CollaborationAccessDenied("Service belongs to another receiving owner.")
        session_id = request.ticket.session_id
        session_instance_id = request.ticket.session_instance_id
        selected = continuation_digest(request)
    else:
        session_id = request.session_id
        session_instance_id = request.session_instance_id
        selected = request.selection_sha256
    if not app.session_store._supports_session_continuation_protocol():
        raise CollaborationUnavailable("Session store does not qualify service readback.")
    raw = await app.session_store.load_session_operation(
        session_id, temporary_service_key(request.operation)
    )
    if raw is None:
        raise ContinuationUnavailable("Service responsibility is unavailable, not excluded.")
    record = prepare_contract(TemporaryServiceRecord, raw, redactor=redactor)
    if record.intent.ticket.owner != initialized.owner:
        raise CollaborationAccessDenied("Service belongs to another receiving owner.")
    # Authorize retained participants before reporting a conflicting caller tuple.
    participants._require_refs(grant, (record.intent.question.responder,))
    if (
        record.intent.operation != request.operation
        or record.intent.ticket.session_id != session_id
        or record.intent.ticket.session_instance_id != session_instance_id
        or record.intent.selection_sha256 != selected
        or (
            isinstance(request, ClarificationServiceRecovery)
            and continuation_digest(record.admission.dispatch) != request.dispatch_sha256
        )
    ):
        raise ContinuationConflict("Service reconciliation requires the exact original selection.")
    if await app.session_store._load_temporary_continuation_service(record.admission) != record:
        raise ContinuationUnavailable("Service responsibility changed during reconstruction.")
    return store, initialized, request, record


async def inspect_settled_service(coordinator, app, request, *, context):
    """Positive native return readback only; no permit settlement or dispatch."""
    _, _, _, record = await _read_service_record(
        coordinator, app, request, context=context, mutation=False
    )
    if record.state not in {"returned", "excluded"} or not record.settlement_acknowledged:
        return None
    return ClarificationServiceReceipt.from_record(record)


async def reconcile_service(
    coordinator: ClarificationCoordinator,
    app: CayuApp,
    request: ClarificationServiceRequest | ClarificationServiceRecovery,
    *,
    context: CollaborationAccessContext,
    exclude: bool = False,
) -> ClarificationServiceReceipt:
    store, initialized, request, record = await _read_service_record(
        coordinator, app, request, context=context, mutation=True
    )
    redactor = coordinator.requests._redactor
    owner = SessionContinuationOwner(
        store=app.session_store,
        owner=initialized.owner,
        receiver=app.collaboration_wait_latch_receiver(),
        receiver_capability=CapabilityDescriptor(
            owner=initialized.owner, mutations=(), readbacks=(LATCH_FAMILY,)
        ),
        redactor=redactor,
        temporary_permits=TemporaryServicePermitAuthority(store, initialized, redactor=redactor),
    )
    if isinstance(request, ClarificationServiceRecovery):
        assert owner.temporary_permits is not None
        registered = await owner.temporary_permits.lookup(record.intent)
        if (
            registered is None
            and not exclude
            and record.settlement is None
            and (
                isinstance(record.admission, TemporaryServiceAdmission)
                or record.state not in {"prepared", "excluded"}
            )
        ) or (
            registered is not None
            and (
                registered.dispatch != record.admission.dispatch
                or registered.permit.expected != record.admission.permit
            )
        ):
            raise ContinuationUnavailable("Service recovery lost its exact source responsibility.")
    if exclude:
        preparation = (
            record.admission.preparation
            if isinstance(record.admission, TemporaryServiceAdmission)
            else record.admission
        )
        result = await owner.exclude_temporary(preparation)
        return ClarificationServiceReceipt.from_record(result)
    # Preparation alone does not prove failed admission. Preserve it for exact
    # exclusion/recovery rather than fabricating a return or starting execution.
    if not isinstance(record.admission, TemporaryServiceAdmission):
        if record.state == "excluded":
            record = await owner.exclude_temporary(record.admission)
        return ClarificationServiceReceipt.from_record(record)
    result = await owner.reconcile_temporary(record.admission)
    return ClarificationServiceReceipt.from_record(result)
