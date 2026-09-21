"""Private continuation publication scopes; never pass these to receiver callbacks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from cayu.collaboration._preparation import contract_bytes
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    from cayu.runtime._invocation_lifecycle import InvocationContext
    from cayu.runtime._session_continuation import (
        ContinuationConsumption,
        ContinuationLatch,
        ContinuationNamespace,
        ContinuationPreparation,
        ContinuationRetirement,
        ContinuationTicket,
    )
    from cayu.sessions.base import Session

_PUBLICATION: ContextVar[str | None] = ContextVar("continuation_publication", default=None)
_LATCH: ContextVar[bytes | None] = ContextVar("continuation_authenticated_latch", default=None)
_RETIREMENT: ContextVar[tuple[bytes, bool] | None] = ContextVar(
    "continuation_retirement", default=None
)
_PARK: ContextVar[bytes | None] = ContextVar("continuation_park", default=None)
_CONSUMPTION: ContextVar[bytes | None] = ContextVar("continuation_consumption", default=None)
_ADMISSION_CLAIM: ContextVar[ContinuationConsumption | None] = ContextVar(
    "continuation_admission_claim", default=None
)
_PREPARATION: ContextVar[tuple[bytes, ContinuationPreparation, InvocationContext] | None] = (
    ContextVar("continuation_preparation", default=None)
)


def require_ticket_invocation(ticket: ContinuationTicket, invocation: InvocationContext) -> None:
    from cayu.runtime._invocation_lifecycle import AdmittedInvocationBinding, InvocationContext

    if (
        type(invocation) is not InvocationContext
        or type(invocation.binding) is not AdmittedInvocationBinding
    ):
        raise PermissionError("Continuation requires an admitted runtime invocation.")
    invocation.require_runtime_authority()
    binding = invocation.binding
    if (
        binding.session_id != ticket.session_id
        or binding.session_instance_id != ticket.session_instance_id
        or binding.run_epoch != ticket.writer_generation
        or binding.interaction_id != ticket.interaction_id
    ):
        raise PermissionError("Continuation conflicts with its originating invocation.")


@contextmanager
def preparation_scope(
    command: ContinuationPreparation, invocation: InvocationContext
) -> Iterator[None]:
    require_ticket_invocation(command.intent, invocation)
    token = _PREPARATION.set(
        (contract_bytes(command, redactor=SecretRedactor()), command, invocation)
    )
    try:
        yield
    finally:
        _PREPARATION.reset(token)


def require_preparation(command: ContinuationPreparation) -> None:
    authority = _PREPARATION.get()
    if authority is None or authority[0] != contract_bytes(command, redactor=SecretRedactor()):
        raise PermissionError("Continuation preparation requires runtime ownership.")


def require_namespace_preparation(namespace: ContinuationNamespace) -> None:
    authority = _PREPARATION.get()
    if authority is None or authority[1].intent.namespace != namespace:
        raise PermissionError("Continuation namespace requires runtime ownership.")


def require_preparation_writer(session: Session, checkpoint: dict | None) -> None:
    from cayu.runtime._invocation_lifecycle import require_invocation_command_authority
    from cayu.sessions.base import SessionRunFenced, SessionStatus

    authority = _PREPARATION.get()
    if authority is None:
        raise PermissionError("Continuation preparation requires runtime ownership.")
    _, command, invocation = authority
    invocation.require_runtime_authority()
    binding = invocation.binding
    require_invocation_command_authority(
        session,
        checkpoint,
        session_id=binding.session_id,
        session_instance_id=binding.session_instance_id,
        run_epochs=frozenset({binding.run_epoch}),
        active_profile=invocation.active_profile,
    )
    if (
        session.status is not SessionStatus.RUNNING
        or command.initiator.invocation_id != session.invocation.root_invocation_id
        or command.initiator.principal != command.source.owner_id
    ):
        raise SessionRunFenced("Continuation preparation lacks current invocation authority.")


@contextmanager
def publication_scope(key: str) -> Iterator[None]:
    token = _PUBLICATION.set(key)
    try:
        yield
    finally:
        _PUBLICATION.reset(token)


def require_publication(key: str) -> None:
    if _PUBLICATION.get() != key:
        raise PermissionError("Continuation evidence requires its session owner.")


def current_publication_key() -> str | None:
    return _PUBLICATION.get()


def continuation_authority_visible() -> bool:
    return any(
        value is not None
        for value in (
            _PUBLICATION.get(),
            _PREPARATION.get(),
            _PARK.get(),
            _CONSUMPTION.get(),
            _RETIREMENT.get(),
        )
    )


def require_operation_key_access(key: str, *, read: bool) -> None:
    from cayu.runtime._session_continuation import CONTINUATION_OPERATION_PREFIX

    if not read and key.startswith(CONTINUATION_OPERATION_PREFIX):
        require_publication(key)


@contextmanager
def authenticated_latch_scope(latch: ContinuationLatch) -> Iterator[None]:
    token = _LATCH.set(contract_bytes(latch, redactor=SecretRedactor()))
    try:
        yield
    finally:
        _LATCH.reset(token)


def require_authenticated_latch(latch: ContinuationLatch) -> None:
    if _LATCH.get() != contract_bytes(latch, redactor=SecretRedactor()):
        raise PermissionError("Continuation latch requires authenticated receiving authority.")


@contextmanager
def retirement_scope(
    retirement: ContinuationRetirement, invocation: InvocationContext | None
) -> Iterator[None]:
    if invocation is None:
        if retirement.reason != "superseded":
            raise PermissionError("Only supersession cleanup can omit the original invocation.")
    else:
        require_ticket_invocation(retirement.ticket, invocation)
    token = _RETIREMENT.set(
        (contract_bytes(retirement, redactor=SecretRedactor()), invocation is None)
    )
    try:
        yield
    finally:
        _RETIREMENT.reset(token)


def require_retirement(retirement: ContinuationRetirement) -> bool:
    authority = _RETIREMENT.get()
    if authority is None or authority[0] != contract_bytes(retirement, redactor=SecretRedactor()):
        raise PermissionError("Continuation retirement requires its registered runtime owner.")
    return authority[1]


@contextmanager
def park_scope(ticket: ContinuationTicket, invocation: InvocationContext) -> Iterator[None]:
    require_ticket_invocation(ticket, invocation)
    token = _PARK.set(contract_bytes(ticket, redactor=SecretRedactor()))
    try:
        yield
    finally:
        _PARK.reset(token)


def require_park(ticket: ContinuationTicket) -> None:
    if _PARK.get() != contract_bytes(ticket, redactor=SecretRedactor()):
        raise PermissionError("Continuation parking requires its registered runtime owner.")


@contextmanager
def consumption_scope(consumption: ContinuationConsumption) -> Iterator[None]:
    """Enclose only the validated admission handoff, not foreign callbacks."""
    token = _CONSUMPTION.set(
        contract_bytes(
            consumption.model_copy(
                update={
                    "receipt_stage": "prepared",
                    "admission_claimed": False,
                    "admission_claim_id": None,
                }
            ),
            redactor=SecretRedactor(),
        )
    )
    try:
        yield
    finally:
        _CONSUMPTION.reset(token)


def require_consumption(consumption: ContinuationConsumption) -> None:
    if _CONSUMPTION.get() != contract_bytes(
        consumption.model_copy(
            update={
                "receipt_stage": "prepared",
                "admission_claimed": False,
                "admission_claim_id": None,
            }
        ),
        redactor=SecretRedactor(),
    ):
        raise PermissionError("Continuation consumption requires runtime admission ownership.")


@contextmanager
def admission_claim_scope(consumption: ContinuationConsumption) -> Iterator[None]:
    require_consumption(consumption)
    if not consumption.admission_claimed or consumption.admission_claim_id is None:
        raise PermissionError("Continuation admission requires a retained claim.")
    token = _ADMISSION_CLAIM.set(consumption)
    try:
        yield
    finally:
        _ADMISSION_CLAIM.reset(token)


def current_admission_claim() -> ContinuationConsumption | None:
    return _ADMISSION_CLAIM.get()
