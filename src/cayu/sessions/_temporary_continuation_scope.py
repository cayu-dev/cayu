"""Shared temporary-service admission identity and exact transition checks."""

from __future__ import annotations

from contextvars import ContextVar

from cayu.collaboration._preparation import prepare_contract
from cayu.sessions._temporary_continuation import (
    TemporaryServiceAdmission,
    require_temporary_service_command,
)
from cayu.vaults.redaction import SecretRedactor

_ADMISSION: ContextVar[TemporaryServiceAdmission | None] = ContextVar(
    "temporary_continuation_admission", default=None
)


def require_temporary_admission(command) -> TemporaryServiceAdmission | None:
    admission = _ADMISSION.get()
    if command.temporary_service_operation_key is None:
        if admission is not None:
            raise PermissionError("Temporary service cannot omit its native command identity.")
        return None
    if admission is None:
        raise PermissionError("Temporary service requires its registered receiving owner.")
    require_temporary_service_command(admission, command)
    return admission


def require_temporary_transition(admission: TemporaryServiceAdmission) -> None:
    if _ADMISSION.get() != admission:
        raise PermissionError(
            "Temporary service transition lacks exact native admission authority."
        )


def prepare_temporary_transition(value: object | None) -> TemporaryServiceAdmission | None:
    if value is None:
        if _ADMISSION.get() is not None:
            raise PermissionError(
                "Temporary service transition cannot drop its receiving identity."
            )
        return None
    admission = prepare_contract(TemporaryServiceAdmission, value, redactor=SecretRedactor())
    require_temporary_transition(admission)
    return admission
