"""Runtime temporary-admission scope producer and compatible imports."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from cayu.collaboration._preparation import prepare_contract
from cayu.sessions._session_continuation import continuation_operation_key
from cayu.sessions._session_continuation_scope import service_publication_scope
from cayu.sessions._temporary_continuation import (
    TemporaryServiceAdmission,
    require_temporary_service_command,
    temporary_service_key,
)

# Runtime producers and store validators share the exact same authority objects.
from cayu.sessions._temporary_continuation_scope import _ADMISSION as _ADMISSION
from cayu.sessions._temporary_continuation_scope import (
    prepare_temporary_transition as prepare_temporary_transition,
)
from cayu.sessions._temporary_continuation_scope import (
    require_temporary_admission as require_temporary_admission,
)
from cayu.sessions._temporary_continuation_scope import (
    require_temporary_transition as require_temporary_transition,
)
from cayu.vaults.redaction import SecretRedactor


@contextmanager
def temporary_admission_scope(
    admission: TemporaryServiceAdmission, command: object
) -> Iterator[None]:
    admission = prepare_contract(TemporaryServiceAdmission, admission, redactor=SecretRedactor())
    require_temporary_service_command(admission, command)
    token = _ADMISSION.set(admission)
    try:
        from cayu.runtime._temporary_service_target import target_service_key
        from cayu.sessions._session_continuation import CONTINUATION_NAMESPACE_KEY

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
