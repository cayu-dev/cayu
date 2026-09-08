"""Application permission owner; callers supply authenticated/store-owned inputs.

This is not a public bearer-token verification entrance. Server authentication
and durable record loading must precede this owner. Its result must be compared
against the same record at the eventual mutation or channel-admission boundary.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from hashlib import sha256

from cayu._validation import require_durable_clean_nonblank
from cayu.runtime.browser_control import (
    BrowserControlAction,
    BrowserControlConflict,
    BrowserControlPolicy,
    BrowserControlPolicyRequest,
    BrowserControlPolicyResult,
    BrowserControlPrincipal,
    BrowserControlRecord,
    BrowserOperatorIdentity,
)


class BrowserControlPermissionDenied(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Browser control permission is unavailable for this action.")


class BrowserControlRevisionChanged(BrowserControlConflict):
    """A validated record positively proves that a read request is stale."""


class BrowserControlInputRejected(BrowserControlConflict):
    """Validated live authority refuses input before any admission publication."""


@dataclass(frozen=True, slots=True, repr=False)
class AuthorizedBrowserControl:
    """Runtime-owned exact permission, never serialized as a client capability."""

    record: BrowserControlRecord
    operator: BrowserOperatorIdentity
    action: BrowserControlAction


def _policy_identity(policy: BrowserControlPolicy) -> str:
    value = policy.identity
    if type(value) is not str or len(value) > 256:
        raise BrowserControlPermissionDenied()
    return require_durable_clean_nonblank(value, "browser control policy identity")


async def authorize_browser_control(
    *,
    policy: BrowserControlPolicy | None,
    principal: BrowserControlPrincipal,
    record: BrowserControlRecord,
    operator_session_id: str,
    action: BrowserControlAction,
) -> AuthorizedBrowserControl:
    if not isinstance(policy, BrowserControlPolicy):
        raise BrowserControlPermissionDenied()
    owned_record = BrowserControlRecord.model_validate(record)
    owned_principal = BrowserControlPrincipal.model_validate(principal)
    policy_identity = _policy_identity(policy)
    request = BrowserControlPolicyRequest(
        principal=owned_principal,
        identity=owned_record.identity,
        operator_session_id=operator_session_id,
        action=action,
        record_revision=owned_record.revision,
        control_epoch=owned_record.control_epoch,
        state=owned_record.state,
    )
    # Preserve runtime-owned values across arbitrary async application code. A
    # frozen Pydantic object is not protection against deliberate object mutation.
    task = asyncio.current_task()
    cancellation_baseline = 0 if task is None else task.cancelling()
    try:
        result = await policy.decide(request.model_copy(deep=True))
    except asyncio.CancelledError:
        if task is not None and task.cancelling() > cancellation_baseline:
            raise
        # An extension's child cancellation is a failed permission check, not
        # evidence that the authenticated request owner was cancelled.
        raise BrowserControlPermissionDenied() from None
    if task is not None and task.cancelling() > cancellation_baseline:
        # A callback swallowing the delivered signal cannot authorize a request
        # whose owner has cancelled it. No browser effect has been admitted here.
        raise asyncio.CancelledError()
    if type(result) is not BrowserControlPolicyResult:
        raise BrowserControlPermissionDenied()
    decision = BrowserControlPolicyResult.model_validate(result)
    if not decision.allowed or _policy_identity(policy) != policy_identity:
        raise BrowserControlPermissionDenied()
    return AuthorizedBrowserControl(
        record=owned_record,
        operator=BrowserOperatorIdentity(
            subject=owned_principal.subject,
            tenant=owned_principal.tenant,
            operator_session_id=request.operator_session_id,
            authorization_policy_fingerprint=sha256(policy_identity.encode("utf-8")).hexdigest(),
        ),
        action=request.action,
    )
