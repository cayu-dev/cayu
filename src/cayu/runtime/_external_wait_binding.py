"""Private binding scope, nested beneath the existing invocation preparation owner."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from cayu.runtime._session_continuation_scope import preparation_scope
from cayu.sessions._external_wait_records import (
    require_binding_identity,
)
from cayu.sessions._session_continuation import (
    ContinuationPreparation,
    ContinuationRecord,
)
from cayu.sessions._session_continuation_scope import (
    require_preparation,
    require_preparation_writer,
)
from cayu.sessions.external_waits import (
    ExternalWaitExecution,
    external_wait_digest,
)

if TYPE_CHECKING:
    from cayu.runtime._invocation_lifecycle import InvocationContext
    from cayu.sessions._external_wait_transition import ExternalWaitMutation
    from cayu.sessions.base import SessionStore
    from cayu.sessions.records import Session

_BINDING: ContextVar[str | None] = ContextVar("external_wait_binding", default=None)


async def load_prepared_continuation(
    store: SessionStore, preparation: ContinuationPreparation
) -> ContinuationRecord | None:
    """Read the retained destination; callers still validate state and authority."""
    ticket = preparation.intent
    return await store.load_continuation_ticket(
        ticket.session_id,
        registration_key=ticket.registration_key,
        session_instance_id=ticket.session_instance_id,
    )


@contextmanager
def binding_scope(command: ExternalWaitMutation, invocation: InvocationContext):
    if command.kind != "bind" or command.continuation is None:
        raise PermissionError("External binding requires exact continuation preparation.")
    with preparation_scope(command.continuation, invocation):
        token = _BINDING.set(external_wait_digest(command))
        try:
            yield
        finally:
            _BINDING.reset(token)


def require_binding_scope(command: ExternalWaitMutation) -> None:
    if command.continuation is None or _BINDING.get() != external_wait_digest(command):
        raise PermissionError("External binding requires runtime-owned invocation authority.")
    require_preparation(command.continuation)
    if command.registration is None:
        raise PermissionError("External binding lacks its complete registration.")
    require_binding_identity(command.registration, command.continuation)


def require_binding_writer(
    command: ExternalWaitMutation,
    session: Session | None,
    checkpoint: dict | None,
    native: object,
    *,
    execution: ExternalWaitExecution | None = None,
) -> None:
    require_binding_scope(command)
    if session is None:
        raise PermissionError("External continuation session is unavailable.")
    require_preparation_writer(session, checkpoint)
    if execution is not None:
        from cayu.sessions._execution_profile_checkpoint import (
            active_invocation_execution_profile_from_checkpoint,
        )

        active = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if active is None or active.profile.fingerprint != execution.intent.profile_sha256:
            raise PermissionError("External execution profile changed before binding.")
    if native is None:
        raise PermissionError("External binding has no native preparation record.")
    record = (
        ContinuationRecord.model_validate_json(native)
        if isinstance(native, str)
        else ContinuationRecord.model_validate(native)
    )
    if record.preparation != command.continuation or record.ticket.state != "ARMING":
        raise PermissionError("External binding lacks its exact native preparation record.")
