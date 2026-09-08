"""Exact viewer purge acknowledgement; no pixel or credential storage."""

from __future__ import annotations

import asyncio
import secrets

from cayu.runtime.browser_control import BrowserControlConflict, BrowserControlIdentity


class BrowserViewerDelivery:
    def __init__(self, identity: BrowserControlIdentity) -> None:
        self.identity = BrowserControlIdentity.model_validate(identity)
        self.purge_token: str | None = None
        self._purged = asyncio.Event()
        self._sending = False
        self._may_hold_frames = False
        self.closed = False

    def request_purge(self) -> str:
        if self.purge_token is None:
            self.purge_token = secrets.token_hex(16)
        # Issuing the purge closes send admission synchronously. If no send has
        # ever been admitted, there are no client pixels to acknowledge. A
        # failed/cancelled send deliberately keeps may_hold_frames true.
        if not self._may_hold_frames and not self._sending:
            self._purged.set()
        return self.purge_token

    def begin_send(self) -> None:
        if self.closed or self.purge_token is not None or self._sending:
            raise BrowserControlConflict("Browser viewer delivery is suspended.")
        self._sending = True
        # Admission precedes the transport await: a failed send can still have
        # delivered bytes. Socket closure is not a client purge acknowledgement.
        self._may_hold_frames = True

    def finish_send(self) -> None:
        self._sending = False

    def acknowledge_purge(self, token: str) -> None:
        if self.closed or self._sending or self.purge_token is None or token != self.purge_token:
            raise BrowserControlConflict("Browser viewer purge acknowledgement differs.")
        self._purged.set()
        self._may_hold_frames = False

    @property
    def may_hold_frames(self) -> bool:
        return self._may_hold_frames

    async def wait_purged(self) -> None:
        async with asyncio.timeout(5):
            await self._purged.wait()

    @property
    def purge_settled(self) -> bool:
        return self._purged.is_set() and not self._sending
