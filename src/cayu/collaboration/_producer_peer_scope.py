"""Live native-producer provenance for inert delivery to its waiting recipient.

This is not disclosure or execution authority. The ordinary peer entrance still
authenticates both participants and the current export policy. Only the producer
owner enters this scope, after exact durable delivery preparation; SessionStore
compares it with its own continuation record inside the append transaction.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from cayu.collaboration._contracts import ObjectRef
from cayu.collaboration._preparation import contract_bytes
from cayu.collaboration.waits import request_object_ref
from cayu.vaults.redaction import SecretRedactor


@dataclass
class _Delivery:
    append: bytes
    target: ObjectRef
    redactor: SecretRedactor
    active: bool = True


_current: ContextVar[_Delivery | None] = ContextVar("native_producer_peer_delivery", default=None)


@contextmanager
def producer_delivery_scope(command, append, *, redactor):
    """Called only with the owner's authenticated, retained preparation."""
    value = _Delivery(
        contract_bytes(append, redactor=redactor),
        request_object_ref(command.admission.expected.intent.selection.reference),
        redactor,
    )
    token = _current.set(value)
    try:
        yield
    finally:
        # Inherited task/thread contexts cannot keep authorizing later appends.
        value.active = False
        _current.reset(token)


def permits_producer_wait_delivery(request, record):
    value = _current.get()
    if value is None or not value.active:
        return False
    if contract_bytes(request, redactor=value.redactor) != value.append:
        return False
    ticket = record.ticket
    return (
        ticket.purpose == "explicit_execution_wait"
        and ticket.execution_admission_sha256 is not None
        and ticket.collaboration_wait_sha256 is not None
        and value.target in ticket.targets
        and (record.latch is None or value.target in record.latch.selected_manifest)
    )
