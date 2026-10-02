"""Durable acceptance and lookup for the existing session execution owner."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from cayu._validation import canonical_durable_json_bytes
from cayu.events import Event, EventType
from cayu.runtime._session_control import SessionInterruptedByRequest
from cayu.runtime.session_steering import (
    SessionSteeringConflict,
    SessionSteeringReceipt,
    StopAfterCurrentToolRoundRequest,
    copy_stop_after_current_tool_round_request,
)
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._invocation_terminal_decision import invocation_terminal_decision_from_checkpoint
from cayu.sessions.base import (
    Session,
    SessionOperationPublication,
    SessionRunFenced,
    SessionStatus,
    SessionStore,
    _invocation_lifecycle_authority_read_scope,
)
from cayu.vaults.redaction import SecretRedactor


class SessionSteeringBoundaryReached(SessionInterruptedByRequest):
    """New work or successful completion lost to an already accepted safe stop.

    The execution owner promotes the receipt at its settled safe boundary.
    Rejection admits no new provider operation and commits no completion.
    """


_COMPLETION_RECORD_TYPE = "cayu.session-steering.completion.v1"


class _InteractionCompletion(BaseModel):
    """Content-free rejection evidence at an incarnation/interaction scoped key."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.session-steering.completion.v1"] = _COMPLETION_RECORD_TYPE
    interaction_event_id: Annotated[StrictStr, Field(min_length=1, max_length=512)]


def steering_receipt_from_record(record: dict[str, Any]) -> SessionSteeringReceipt | None:
    """Distinguish an accepted stop from a transactionally completed interaction."""

    if record.get("record_type") == _COMPLETION_RECORD_TYPE:
        _InteractionCompletion.model_validate(record)
        return None
    return SessionSteeringReceipt.model_validate(record)


def require_steering_receipt(record: dict[str, Any]) -> SessionSteeringReceipt:
    receipt = steering_receipt_from_record(record)
    if receipt is None:
        raise SessionSteeringConflict()
    return receipt


def interaction_completion_steering_key(
    session: Session, checkpoint: dict[str, Any] | None, event: Event
) -> str | None:
    """Name the active completion after the store has authenticated its authority."""

    if event.type != EventType.INTERACTION_COMPLETED or session.status is not SessionStatus.RUNNING:
        return None
    profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if profile is None or profile.interaction_id != event.interaction_id:
        return None
    if profile.session_id != session.id or profile.run_epoch != session.run_epoch:
        raise SessionSteeringConflict()
    return steering_operation_key(session.instance_id, profile.interaction_id)


def prepare_interaction_completion_steering_record(
    session: Session,
    checkpoint: dict[str, Any] | None,
    event: Event,
    record: dict[str, Any] | None,
    *,
    keeps_running: bool,
) -> dict[str, Any] | None:
    """Resolve stop versus completion inside the existing settlement transaction.

    Receipt replay must precede this check. When queued work or environment
    finalization keeps the session running, reserve this interaction's key so
    stop acceptance cannot mistake its closed predecessor for a live owner.
    """

    if record is not None and steering_receipt_from_record(record) is None:
        raise SessionRunFenced("The interaction has already completed.")
    reject_new_work_after_steering(session, checkpoint, record)
    if keeps_running:
        return _InteractionCompletion(interaction_event_id=event.id).model_dump(mode="json")
    return None


def steering_operation_key_from_checkpoint(
    session: Session, checkpoint: dict[str, Any] | None
) -> str | None:
    profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if profile is None:
        return None
    if profile.session_id != session.id or profile.run_epoch != session.run_epoch:
        raise SessionSteeringConflict()
    return steering_operation_key(session.instance_id, profile.interaction_id)


def reject_new_work_after_steering(
    session: Session,
    checkpoint: dict[str, Any] | None,
    record: dict[str, Any] | None,
    *,
    allow_completed_interaction: bool = False,
) -> None:
    """Check a receipt read under the same transaction as new-work admission."""

    if record is None:
        return
    receipt = steering_receipt_from_record(record)
    if receipt is None:
        # Only a queue handoff may use a completed predecessor's checkpoint.
        # Its existing settlement receipt authenticates the new interaction.
        if allow_completed_interaction:
            return
        raise SessionRunFenced("The interaction has already completed.")
    request = receipt.request
    profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if (
        profile is None
        or request.session_id != session.id
        or request.session_instance_id != session.instance_id
        or request.interaction_id != profile.interaction_id
        or receipt.execution_profile_fingerprint != profile.profile.fingerprint
        or request.expected_run_epoch > session.run_epoch
    ):
        raise SessionSteeringConflict()
    raise SessionSteeringBoundaryReached(session.id)


def steering_operation_key(session_instance_id: str, interaction_id: str) -> str:
    identity = canonical_durable_json_bytes(
        {"session_instance_id": session_instance_id, "interaction_id": interaction_id},
        "session steering identity",
    )
    return "cayu.session-steering.v1:" + sha256(identity).hexdigest()


async def accept_session_steering(
    request: StopAfterCurrentToolRoundRequest,
    *,
    session_store: SessionStore,
    redactor: SecretRedactor,
    invocation_guard: Callable[[Session, dict[str, Any] | None], None] | None = None,
) -> SessionSteeringReceipt:
    """Persist acceptance without cancelling work or changing session status.

    One immutable operation record belongs to one interaction. Exact replay is
    possible after terminalization; a second, different request cannot replace
    that authority. Store records, unlike process-local signals, survive worker
    replacement and checkpoint reconstruction.
    """

    owned = copy_stop_after_current_tool_round_request(request)
    for value in (
        owned.session_id,
        owned.session_instance_id,
        owned.interaction_id,
        owned.idempotency_key,
    ):
        if redactor.redact_text(value) != value:
            raise ValueError("Session steering identity cannot contain workload secrets.")
    key = steering_operation_key(owned.session_instance_id, owned.interaction_id)
    existing = await session_store.load_session_operation(owned.session_id, key)
    if existing is not None:
        receipt = require_steering_receipt(existing)
        if receipt.request != owned:
            raise SessionSteeringConflict()
        return receipt
    if not session_store._supports_session_steering_protocol():
        raise RuntimeError("Session store does not support atomic safe steering.")

    def prepare(
        session: Session,
        checkpoint: dict[str, Any] | None,
        existing: dict[str, Any] | None,
    ) -> SessionOperationPublication:
        if session.instance_id != owned.session_instance_id:
            raise SessionSteeringConflict()
        if invocation_guard is not None:
            invocation_guard(session, checkpoint)
        if existing is not None:
            receipt = require_steering_receipt(existing)
            if receipt.request != owned:
                raise SessionSteeringConflict()
            return SessionOperationPublication(checkpoint=checkpoint or {})
        profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if (
            session.status is not SessionStatus.RUNNING
            or session.run_epoch != owned.expected_run_epoch
            or profile is None
            or profile.session_id != session.id
            or profile.run_epoch != session.run_epoch
            or profile.interaction_id != owned.interaction_id
            or invocation_terminal_decision_from_checkpoint(checkpoint) is not None
        ):
            raise SessionSteeringConflict()
        receipt = SessionSteeringReceipt(
            request=owned,
            execution_profile_fingerprint=profile.profile.fingerprint,
        )
        return SessionOperationPublication(
            checkpoint=checkpoint or {},
            operation_records={key: receipt.model_dump(mode="json")},
        )

    with _invocation_lifecycle_authority_read_scope():
        await session_store.publish_session_operation(
            owned.session_id,
            idempotency_key=key,
            operation_transform=prepare,
            events=[],
        )
    stored = await session_store.load_session_operation(owned.session_id, key)
    if stored is None:
        raise SessionSteeringConflict()
    receipt = require_steering_receipt(stored)
    if receipt.request != owned:
        raise SessionSteeringConflict()
    return receipt
