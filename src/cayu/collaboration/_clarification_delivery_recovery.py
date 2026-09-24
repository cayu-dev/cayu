"""Content-free maintenance of existing delivery responsibility, never dispatch."""

from cayu.collaboration._clarification_deliveries import (
    ClarificationDeliveryReceipt,
    ClarificationDeliveryRecord,
)
from cayu.collaboration._clarification_delivery_store import (
    discover_deliveries_in_transaction,
    load_delivery_in_transaction,
    reconcile_delivery,
)
from cayu.collaboration._clarification_recovery_types import (
    ClarificationDeliveryRecovery,
    ClarificationPendingDelivery,
    ClarificationPendingDeliveryPage,
    ClarificationPendingServiceQuery,
)
from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.exports import SessionExportDenied
from cayu.collaboration.participants import CollaborationUnavailable


def _authorize(coordinator, context, *, mutation):
    participants = coordinator.requests._participants
    redactor = coordinator.requests._redactor
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    store, initialized = participants._ready()
    participants._capability(store, initialized, mutation=mutation, family=REQUEST_FAMILY)
    _, grant = participants._authorize(
        context, "request_control" if mutation else "request_readback"
    )
    participants._require_refs(grant, (), create=True)
    return participants, store, initialized, redactor


async def pending_deliveries(coordinator, *, context, cursor=None, limit=32):
    participants, store, initialized, redactor = _authorize(coordinator, context, mutation=False)
    query = prepare_contract(
        ClarificationPendingServiceQuery, {"cursor": cursor, "limit": limit}, redactor=redactor
    )
    if query.cursor is not None and (
        query.cursor.operation.application_scope != initialized.binding.application_scope
        or query.cursor.operation.namespace_incarnation != initialized.namespace_incarnation
    ):
        raise CollaborationConflict("Delivery recovery cursor belongs to another owner.")
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        records = await discover_deliveries_in_transaction(
            store, tx, initialized, after=query.cursor, limit=query.limit, redactor=redactor
        )
    items = tuple(
        ClarificationPendingDelivery(
            recovery=ClarificationDeliveryRecovery(
                operation=record.intent.operation,
                intent_sha256=clarification_commitment(record.intent, redactor),
            ),
            question=record.intent.question.operation,
            deadline_at_ms=record.intent.append.attempt_key.deadline_at_ms,
        )
        for record in records
    )
    return participants._page(
        ClarificationPendingDeliveryPage,
        "items",
        items,
        lambda item: {
            "deadline_at_ms": item.deadline_at_ms,
            "operation": item.recovery.operation.model_dump(mode="json"),
        },
        query.limit,
    )


async def reconcile_pending_delivery(coordinator, recovery, *, context, exclude=False):
    _, store, initialized, redactor = _authorize(coordinator, context, mutation=True)
    recovery = prepare_contract(ClarificationDeliveryRecovery, recovery, redactor=redactor)
    if (
        recovery.operation.application_scope != initialized.binding.application_scope
        or recovery.operation.namespace_incarnation != initialized.namespace_incarnation
    ):
        raise CollaborationConflict("Delivery recovery belongs to another owner.")
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        raw = await tx.get("clarification_deliveries", operation_key(recovery.operation))
        if raw is None:
            raise CollaborationUnavailable("Delivery responsibility is unavailable, not excluded.")
        record = prepare_contract(ClarificationDeliveryRecord, raw, redactor=redactor)
        if (
            record.intent.operation != recovery.operation
            or clarification_commitment(record.intent, redactor) != recovery.intent_sha256
        ):
            raise CollaborationConflict("Delivery recovery requires the exact original intent.")
        exact = await load_delivery_in_transaction(
            store, tx, initialized, record.intent, redactor=redactor
        )
        if exact != record:
            raise CollaborationUnavailable("Delivery recovery contradicts its native registration.")
    if exclude:
        receiving = coordinator.exports.store
        if receiving.peer_content_version != 1:
            raise CollaborationUnavailable("Receiving store cannot fence peer delivery.")
        # The registered export policy independently authorizes cleanup of this
        # owner-read preparation. Never fabricate a native receipt or reacquire
        # disclosure to discharge a permanently revoked source obligation.
        try:
            async with coordinator.exports.acquire_peer_exclusion(
                context, request=record.intent.append, receipt=record, reason="withdrawn"
            ):
                if record.state != "settled":
                    await receiving.exclude_peer_content(record.intent.append, reason="withdrawn")
        except SessionExportDenied:
            raise CollaborationAccessDenied("Delivery cleanup is not authorized.") from None
    # Receiving readback proves only append/exclusion, not exposure or quiescence.
    # Absence stays pending. Current export authorization is still required by the
    # separate delivery/exposure entrances; this path returns no source content.
    result = await reconcile_delivery(
        store, initialized, record.intent, coordinator.exports.store, redactor=redactor
    )
    return ClarificationDeliveryReceipt.from_record(result)
