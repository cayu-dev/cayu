"""Single-use input-channel handoff, structurally separate from viewer tickets."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256

from cayu.runtime._browser_control_authorization import BrowserControlPermissionDenied
from cayu.runtime._browser_control_coordinator import BrowserControlCoordinator
from cayu.runtime.browser_control import BrowserControlPrincipal, BrowserTextInputIntent


@dataclass(frozen=True, slots=True, repr=False)
class BrowserInputIdentity:
    principal: BrowserControlPrincipal
    operator_session_id: str
    intent: BrowserTextInputIntent


class BrowserInputTickets:
    def __init__(
        self, coordinator: BrowserControlCoordinator, *, clock: Callable[[], float] = time.monotonic
    ):
        self._coordinator = coordinator
        self._clock = clock
        self._pending: dict[str, tuple[float, BrowserInputIdentity]] = {}
        self._closed = False

    def _expire(self) -> None:
        now = self._clock()
        self._pending = {key: entry for key, entry in self._pending.items() if entry[0] > now}

    async def issue(
        self,
        *,
        principal: BrowserControlPrincipal,
        operator_session_id: str,
        intent: BrowserTextInputIntent,
    ) -> str:
        self._expire()
        if self._closed or len(self._pending) >= 32:
            raise BrowserControlPermissionDenied()
        intent = BrowserTextInputIntent.model_validate(intent)
        principal = BrowserControlPrincipal.model_validate(principal)
        _, record, authorized = await self._coordinator._prepare_text_input(
            principal=principal, operator_session_id=operator_session_id, intent=intent
        )
        _, current = await self._coordinator._load(intent.identity)
        self._expire()
        if self._closed or len(self._pending) >= 32 or current != record:
            raise BrowserControlPermissionDenied()
        token = secrets.token_hex(32)
        self._pending[sha256(token.encode("ascii")).hexdigest()] = (
            self._clock() + 10,
            BrowserInputIdentity(principal, authorized.operator.operator_session_id, intent),
        )
        return token

    def consume(self, token: str) -> BrowserInputIdentity:
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
