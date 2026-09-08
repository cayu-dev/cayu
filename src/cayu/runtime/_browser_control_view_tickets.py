"""Single-use, short-lived operator transport handoff; not durable permission."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256

from cayu.runtime._browser_control_authorization import BrowserControlPermissionDenied
from cayu.runtime._browser_control_coordinator import BrowserControlCoordinator
from cayu.runtime.browser_control import BrowserControlPrincipal, BrowserViewIntent


@dataclass(frozen=True, slots=True, repr=False)
class BrowserViewerIdentity:
    principal: BrowserControlPrincipal
    operator_session_id: str
    intent: BrowserViewIntent


class BrowserViewTickets:
    def __init__(
        self, coordinator: BrowserControlCoordinator, *, clock: Callable[[], float] = time.monotonic
    ):
        self._coordinator = coordinator
        self._clock = clock
        self._pending: dict[str, tuple[float, BrowserViewerIdentity]] = {}
        self._closed = False

    def _expire(self) -> None:
        now = self._clock()
        self._pending = {key: entry for key, entry in self._pending.items() if entry[0] > now}

    async def issue(
        self,
        *,
        principal: BrowserControlPrincipal,
        operator_session_id: str,
        intent: BrowserViewIntent,
    ) -> str:
        self._expire()
        if self._closed or len(self._pending) >= 32:
            raise BrowserControlPermissionDenied()
        intent = BrowserViewIntent.model_validate(intent)
        principal = BrowserControlPrincipal.model_validate(principal)
        authorized = await self._coordinator.authorize_view(
            principal=principal,
            operator_session_id=operator_session_id,
            identity=intent.identity,
            expected_record_revision=intent.expected_record_revision,
        )
        _, current = await self._coordinator._load(intent.identity)
        self._expire()
        if (
            self._closed
            or len(self._pending) >= 32
            or current != authorized.record
            or current.sensitive_entry
            or current.sensitive_entry_pending
            or current.state
            not in {"agent_controlled", "takeover_requested", "operator_controlled"}
        ):
            raise BrowserControlPermissionDenied()
        token = secrets.token_hex(32)
        self._pending[sha256(token.encode("ascii")).hexdigest()] = (
            self._clock() + 30,
            BrowserViewerIdentity(principal, authorized.operator.operator_session_id, intent),
        )
        return token

    def consume(self, token: str) -> BrowserViewerIdentity:
        self._expire()
        if (
            self._closed
            or type(token) is not str
            or len(token) != 64
            or any(char not in "0123456789abcdef" for char in token)
        ):
            raise BrowserControlPermissionDenied()
        entry = self._pending.pop(sha256(token.encode("ascii")).hexdigest(), None)
        if entry is None:
            raise BrowserControlPermissionDenied()
        return entry[1]

    def close(self) -> None:
        self._closed = True
        self._pending.clear()
