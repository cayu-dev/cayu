"""Private, single-use allocation capabilities for the guest transport.

Only the admitted browser runtime may issue these and deliver them through private
runner I/O. HTTP operator credentials and guest-supplied identity are not inputs to
issuance. Restart invalidates pending bootstrap capabilities, never restoring model
control or replaying operator input.
"""

from __future__ import annotations

import hashlib
import math
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from cayu._validation import canonical_durable_json_bytes
from cayu.core.tools import ToolContext, _runtime_tool_invocation_authority
from cayu.runtime._browser_control_authorization import BrowserControlPermissionDenied
from cayu.runtime.browser_control import BrowserControlAllocation, BrowserOperatorPurpose


@dataclass(frozen=True, slots=True, repr=False)
class _PendingGuest:
    allocation: BrowserControlAllocation
    expires: float


class BrowserGuestBootstrap:
    """Bounded transient capability owner; not an application permission policy."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._pending: dict[bytes, _PendingGuest] = {}
        self._closed = False

    def _now(self) -> float:
        now = self._clock()
        if type(now) not in {float, int} or not math.isfinite(now) or now < 0:
            raise BrowserControlPermissionDenied()
        for digest, pending in tuple(self._pending.items()):
            if now >= pending.expires:
                del self._pending[digest]
        return now

    def issue(self, allocation: BrowserControlAllocation) -> str:
        allocation = BrowserControlAllocation.model_validate(allocation)
        now = self._now()
        if self._closed or len(self._pending) >= 32:
            raise BrowserControlPermissionDenied()
        token = secrets.token_hex(32)
        digest = hashlib.sha256(token.encode("ascii")).digest()
        self._pending[digest] = _PendingGuest(allocation, now + 60)
        return token

    def issue_for_invocation(
        self,
        context: ToolContext,
        *,
        browser_session_id: str,
        arguments: dict,
        purpose: BrowserOperatorPurpose,
    ) -> str:
        return self.issue(
            self.allocation_for_invocation(
                context, browser_session_id=browser_session_id, arguments=arguments, purpose=purpose
            )
        )

    @staticmethod
    def allocation_for_invocation(
        context: ToolContext,
        *,
        browser_session_id: str,
        arguments: dict,
        purpose: BrowserOperatorPurpose,
    ) -> BrowserControlAllocation:
        """Accept only the exact built-in's captured in-process runtime authority."""
        authority = _runtime_tool_invocation_authority(context)
        if authority is None or authority.browser_allocation is None:
            raise BrowserControlPermissionDenied()
        bound = authority.browser_allocation
        if (
            authority.tool_name != "browser_session"
            or context.session_id != bound.session_id
            or context.environment_name != bound.environment_name
            or context.idempotency_key != authority.idempotency_key
            or hashlib.sha256(
                canonical_durable_json_bytes(arguments, "browser bootstrap arguments")
            ).hexdigest()
            != authority.effective_arguments_sha256
        ):
            raise BrowserControlPermissionDenied()
        return BrowserControlAllocation(
            session_id=bound.session_id,
            session_instance_id=bound.session_instance_id,
            run_epoch=bound.run_epoch,
            interaction_id=bound.interaction_id,
            execution_profile_fingerprint=bound.execution_profile_fingerprint,
            environment_name=bound.environment_name,
            allocation_fingerprint=bound.allocation_fingerprint,
            browser_session_id=browser_session_id,
            profile_checkpoint_policy=bound.profile_checkpoint_policy,
            operator_purpose=purpose,
        )

    def consume(self, token: str) -> BrowserControlAllocation:
        """Authenticate once, before accepting the WebSocket or reading its hello."""
        self._now()
        if (
            self._closed
            or type(token) is not str
            or len(token) != 64
            or any(character not in "0123456789abcdef" for character in token)
        ):
            raise BrowserControlPermissionDenied()
        digest = hashlib.sha256(token.encode("ascii")).digest()
        pending = self._pending.pop(digest, None)
        if pending is None:
            raise BrowserControlPermissionDenied()
        return BrowserControlAllocation.model_validate(pending.allocation)

    def revoke(self, token: str) -> None:
        """Retire an undispatched bootstrap without retaining its bearer value."""
        if type(token) is str and len(token) == 64 and token.isascii():
            self._pending.pop(hashlib.sha256(token.encode("ascii")).digest(), None)

    def close(self) -> None:
        self._closed = True
        self._pending.clear()
