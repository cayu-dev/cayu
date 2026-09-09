"""Durable acceptance and lookup for the existing session execution owner."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from cayu._validation import canonical_durable_json_bytes
from cayu.runtime._invocation_terminal_decision import invocation_terminal_decision_from_checkpoint
from cayu.runtime._session_control import SessionInterruptedByRequest
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
from cayu.runtime.session_steering import (
    SessionSteeringConflict,
    SessionSteeringReceipt,
    StopAfterCurrentToolRoundRequest,
    copy_stop_after_current_tool_round_request,
)
from cayu.runtime.sessions import (
    Session,
    SessionOperationPublication,
    SessionStatus,
    SessionStore,
    _invocation_lifecycle_authority_read_scope,
)
from cayu.vaults.redaction import SecretRedactor


class SessionSteeringBoundaryReached(SessionInterruptedByRequest):
    """A new model stage lost admission to an already accepted safe stop.

    The execution owner must promote the receipt to durable interruption after
    unwinding pre-dispatch preparation. No provider operation was admitted.
    """


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
    session: Session, checkpoint: dict[str, Any] | None, record: dict[str, Any] | None
) -> None:
    """Check a receipt read under the same transaction as new-work admission."""

    if record is None:
        return
    receipt = SessionSteeringReceipt.model_validate(record)
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
        receipt = SessionSteeringReceipt.model_validate(existing)
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
        if existing is not None:
            receipt = SessionSteeringReceipt.model_validate(existing)
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
    receipt = SessionSteeringReceipt.model_validate(stored)
    if receipt.request != owned:
        raise SessionSteeringConflict()
    return receipt
