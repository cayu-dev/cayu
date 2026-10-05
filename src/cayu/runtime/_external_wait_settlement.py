"""Exact native settlement readback; absence never discharges responsibility."""

from contextlib import contextmanager
from contextvars import ContextVar

from cayu.sessions._external_wait_transition import ExternalWaitMutation
from cayu.sessions._session_continuation import (
    ContinuationRecord,
    continuation_digest,
    require_latch_identity,
)
from cayu.sessions.external_waits import (
    ExternalWaitConflict,
    ExternalWaitRecord,
    external_wait_digest,
)

_SETTLEMENT: ContextVar[str | None] = ContextVar("external_wait_settlement", default=None)


@contextmanager
def settlement_scope(command: ExternalWaitMutation):
    token = _SETTLEMENT.set(external_wait_digest(command))
    try:
        yield
    finally:
        _SETTLEMENT.reset(token)


def require_settlement_scope(command: ExternalWaitMutation) -> None:
    if (
        command.kind
        not in {
            "settle",
            "prepare_service",
            "complete_retirement",
            "reconcile_binding",
            "prepare_retirement",
        }
        or command.continuation is None
        or command.registration is None
        or (
            command.kind in {"settle", "complete_retirement"}
            and (command.handoff_disposition is None or command.handoff_receipt_sha256 is None)
        )
        or (
            command.kind == "prepare_service"
            and (command.service is None or command.service_stage_id is None)
        )
        or _SETTLEMENT.get() != external_wait_digest(command)
    ):
        raise PermissionError("External settlement requires its registered receiving owner.")


def require_native_settlement(
    command: ExternalWaitMutation, current: ExternalWaitRecord | None, native: object
) -> None:
    require_settlement_scope(command)
    if command.kind == "reconcile_binding":
        from cayu.sessions._external_wait_records import require_binding_identity

        if (
            current is None
            or current.registration != command.registration
            or current.execution is None
            or native is None
        ):
            raise ExternalWaitConflict("External binding recovery has no native preparation.")
        record = (
            ContinuationRecord.model_validate_json(native)
            if isinstance(native, str)
            else ContinuationRecord.model_validate(native)
        )
        assert command.registration is not None
        if record.preparation != command.continuation or record.ticket.state not in {
            "ARMING",
            "WAITING",
            "RETIRED",
        }:
            raise ExternalWaitConflict("External binding recovery native preparation conflicts.")
        require_binding_identity(command.registration, record.preparation)
        return
    if current is None or current.continuation != command.continuation:
        raise ExternalWaitConflict("External settlement binding is unavailable or changed.")
    if command.kind == "prepare_retirement" and current.execution_retirement is not None:
        # The transition still compares the exact control key. Committed replay
        # must survive native retirement acknowledgement and source deletion.
        return
    if command.kind == "complete_retirement" and current.retirement_complete:
        return
    if command.kind == "prepare_service" and current.service is not None:
        if (
            current.service != command.service
            or current.service_stage_id != command.service_stage_id
        ):
            raise ExternalWaitConflict("External service preparation conflicts.")
        return
    if command.kind == "settle" and current.handoff in {"settled", "excluded"}:
        # Exact committed replay no longer depends on retaining a deleted session.
        return
    if native is None:
        raise ExternalWaitConflict("External settlement has no native receiving evidence.")
    record = (
        ContinuationRecord.model_validate_json(native)
        if isinstance(native, str)
        else ContinuationRecord.model_validate(native)
    )
    if command.kind == "prepare_retirement":
        if (
            record.preparation != command.continuation
            or record.ticket.state not in {"ARMING", "WAITING"}
            or record.consumption is not None
        ):
            raise ExternalWaitConflict("External retirement requires an unconsumed native wait.")
        return
    if command.kind == "complete_retirement":
        from cayu.runtime._continuation_wait_settlement import retirement_receipt

        if (
            current.handoff != "excluded"
            or command.handoff_disposition != "excluded"
            or record.preparation != command.continuation
            or record.ticket.state != "RETIRED"
            or record.retirement is None
            or (record.released_retirement is not None and not record.retirement_acknowledged)
            or continuation_digest(retirement_receipt(record)) != current.handoff_receipt_sha256
        ):
            raise ExternalWaitConflict("External retirement acknowledgement is not durable.")
        return
    if command.kind == "prepare_service":
        from cayu.sessions._external_wait_records import elected_external_latch

        if (
            command.service is None
            or record.preparation != command.continuation
            or record.ticket.state != "WAITING"
            or record.ticket != command.service.ticket
            or record.consumption is not None
        ):
            raise ExternalWaitConflict(
                "External service requires its exact unconsumed parked ticket."
            )
        require_latch_identity(elected_external_latch(current), command.service.latch)
        return
    if (
        record.preparation != command.continuation
        or continuation_digest(record) != command.handoff_receipt_sha256
    ):
        raise ExternalWaitConflict("External settlement native evidence conflicts.")
    if command.handoff_disposition == "settled":
        if (
            record.ticket.state != "CONSUMED"
            or record.consumption is None
            or record.consumption.receipt_stage != "admitted"
            or record.latch is None
        ):
            raise ExternalWaitConflict("External continuation admission is not durable.")
        from cayu.sessions._external_wait_records import elected_external_latch

        require_latch_identity(elected_external_latch(current), record.latch)
    elif record.ticket.state != "RETIRED" or record.retirement is None:
        raise ExternalWaitConflict("External continuation exclusion is not durable.")
