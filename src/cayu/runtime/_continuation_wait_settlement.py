"""Receiving evidence retained until the registered wait owner confirms exclusion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu.runtime._session_continuation import (
    ContinuationConflict,
    ContinuationRecord,
    continuation_digest,
    continuation_operation_key,
)
from cayu.runtime._session_continuation_store import require_history
from cayu.sessions.base import SessionOperationPublication

if TYPE_CHECKING:
    from cayu.collaboration.waits import CollaborationWait
    from cayu.runtime._session_continuation_owner import SessionContinuationOwner


def retirement_receipt(record: ContinuationRecord) -> ContinuationRecord:
    """Recover the immutable receipt sent before the acknowledgement handshake."""
    require_history(record)
    if not record.retirement_acknowledged:
        return record
    if not record.events or record.events[-1].kind != "retirement_acknowledged":
        raise ContinuationConflict("Retirement acknowledgement history is unavailable.")
    original = record.model_copy(
        update={"retirement_acknowledged": False, "events": record.events[:-1]}
    )
    require_history(original)
    return original


async def acknowledge_retirement(
    owner: SessionContinuationOwner,
    wait: CollaborationWait,
    candidate: ContinuationRecord,
) -> None:
    """Only configured receiving code may turn foreign readback into native release."""
    from cayu.collaboration._preparation import prepare_contract
    from cayu.collaboration.waits import CollaborationWait

    record = prepare_contract(ContinuationRecord, candidate, redactor=owner.redactor)
    wait = prepare_contract(CollaborationWait, wait, redactor=owner.redactor)
    original = retirement_receipt(record)
    if (
        record.ticket.owner != owner.owner
        or record.released_retirement is None
        or record.preparation.registration.child.destination != owner.receiver_capability.owner
    ):
        raise PermissionError("Retirement settlement requires this registered receiving owner.")
    authenticate = getattr(owner.receiver, "authenticate_continuation_retirement", None)
    if not callable(authenticate):
        raise PermissionError("Registered receiver does not qualify retirement settlement.")
    expected = continuation_digest(original)
    # No store publication scope crosses the foreign receiver callback.
    if await authenticate(wait, original) != expected:
        raise PermissionError("Wait owner did not authenticate the exact retirement receipt.")
    key = continuation_operation_key(record.ticket)

    def transform(session, checkpoint, current):
        if session.instance_id != record.ticket.session_instance_id or current is None:
            raise ContinuationConflict("Retirement receiving incarnation is unavailable.")
        retained = ContinuationRecord.model_validate(current)
        if retirement_receipt(retained) != original:
            raise ContinuationConflict("Retirement changed before settlement acknowledgement.")
        updated = retained.model_copy(update={"retirement_acknowledged": True})
        return SessionOperationPublication(
            checkpoint={} if checkpoint is None else checkpoint,
            operation_records={key: updated.model_dump(mode="json")},
        )

    await owner.store._publish_continuation_operation(
        record.ticket.session_id,
        idempotency_key=key,
        operation_transform=transform,
        events=[],
    )
