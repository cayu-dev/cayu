"""Exact private receiving scope; registered foreign authentication occurs outside."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from cayu.collaboration._preparation import prepare_contract
from cayu.runtime._session_continuation import continuation_operation_key
from cayu.runtime._session_continuation_scope import service_publication_scope
from cayu.runtime._temporary_continuation import (
    TemporaryServiceAdmission,
    require_temporary_service_command,
    temporary_service_key,
)
from cayu.vaults.redaction import SecretRedactor

_ADMISSION: ContextVar[TemporaryServiceAdmission | None] = ContextVar(
    "temporary_continuation_admission", default=None
)


@contextmanager
def temporary_admission_scope(
    admission: TemporaryServiceAdmission, command: object
) -> Iterator[None]:
    admission = prepare_contract(TemporaryServiceAdmission, admission, redactor=SecretRedactor())
    require_temporary_service_command(admission, command)
    token = _ADMISSION.set(admission)
    try:
        from cayu.runtime._session_continuation import CONTINUATION_NAMESPACE_KEY
        from cayu.runtime._temporary_service_target import target_service_key

        if admission.dispatch.intent.mode == "side_session":
            with service_publication_scope(
                target_service_key(admission.dispatch.intent.operation), CONTINUATION_NAMESPACE_KEY
            ):
                yield
            return
        with service_publication_scope(
            continuation_operation_key(admission.dispatch.intent.ticket),
            temporary_service_key(admission.dispatch.intent.operation),
        ):
            yield
    finally:
        _ADMISSION.reset(token)


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
