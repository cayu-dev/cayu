"""Runtime-authenticated continuation scope producers and compatible imports."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from cayu.collaboration._preparation import contract_bytes
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    from cayu.runtime._invocation_lifecycle import InvocationContext
    from cayu.sessions._session_continuation import (
        ContinuationPreparation,
        ContinuationRetirement,
        ContinuationTicket,
    )

# Runtime producers and store validators share the exact same authority objects.
from cayu.sessions._session_continuation_scope import _ADMISSION_CLAIM as _ADMISSION_CLAIM
from cayu.sessions._session_continuation_scope import _CONSUMPTION as _CONSUMPTION
from cayu.sessions._session_continuation_scope import _LATCH as _LATCH
from cayu.sessions._session_continuation_scope import _PARK as _PARK
from cayu.sessions._session_continuation_scope import _PREPARATION as _PREPARATION
from cayu.sessions._session_continuation_scope import _PUBLICATION as _PUBLICATION
from cayu.sessions._session_continuation_scope import _RETIREMENT as _RETIREMENT
from cayu.sessions._session_continuation_scope import _SERVICE_PUBLICATION as _SERVICE_PUBLICATION
from cayu.sessions._session_continuation_scope import admission_claim_scope as admission_claim_scope
from cayu.sessions._session_continuation_scope import (
    authenticated_latch_scope as authenticated_latch_scope,
)
from cayu.sessions._session_continuation_scope import consumption_scope as consumption_scope
from cayu.sessions._session_continuation_scope import (
    continuation_authority_visible as continuation_authority_visible,
)
from cayu.sessions._session_continuation_scope import (
    current_admission_claim as current_admission_claim,
)
from cayu.sessions._session_continuation_scope import (
    current_publication_key as current_publication_key,
)
from cayu.sessions._session_continuation_scope import publication_scope as publication_scope
from cayu.sessions._session_continuation_scope import (
    released_retirement_scope as released_retirement_scope,
)
from cayu.sessions._session_continuation_scope import (
    require_authenticated_latch as require_authenticated_latch,
)
from cayu.sessions._session_continuation_scope import require_consumption as require_consumption
from cayu.sessions._session_continuation_scope import (
    require_namespace_preparation as require_namespace_preparation,
)
from cayu.sessions._session_continuation_scope import (
    require_operation_key_access as require_operation_key_access,
)
from cayu.sessions._session_continuation_scope import require_park as require_park
from cayu.sessions._session_continuation_scope import require_preparation as require_preparation
from cayu.sessions._session_continuation_scope import (
    require_preparation_writer as require_preparation_writer,
)
from cayu.sessions._session_continuation_scope import require_publication as require_publication
from cayu.sessions._session_continuation_scope import require_retirement as require_retirement
from cayu.sessions._session_continuation_scope import (
    service_publication_scope as service_publication_scope,
)


def require_ticket_invocation(ticket: ContinuationTicket, invocation: InvocationContext) -> None:
    from cayu.runtime._invocation_lifecycle import (
        InvocationContext,
    )
    from cayu.sessions._invocation_lifecycle import (
        AdmittedInvocationBinding,
    )

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


@contextmanager
def park_scope(ticket: ContinuationTicket, invocation: InvocationContext) -> Iterator[None]:
    require_ticket_invocation(ticket, invocation)
    token = _PARK.set(contract_bytes(ticket, redactor=SecretRedactor()))
    try:
        yield
    finally:
        _PARK.reset(token)
