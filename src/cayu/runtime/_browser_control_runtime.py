"""Application composition for existing browser control owners, not another driver."""

import asyncio
from collections.abc import Callable
from datetime import datetime

from cayu.runtime._browser_control_coordinator import BrowserControlCoordinator
from cayu.runtime._browser_control_service import BrowserControlService
from cayu.sessions.base import SessionStore
from cayu.tools.browser_control_config import BrowserControlConfig
from cayu.vaults.redaction import SecretRedactor

_MAX_PHASE_SECONDS = 30.0


class BrowserControlRuntime:
    def __init__(
        self,
        *,
        config: BrowserControlConfig,
        store: SessionStore,
        redactor: SecretRedactor,
        clock: Callable[[], datetime],
    ) -> None:
        if type(config) is not BrowserControlConfig:
            raise TypeError("Browser control requires resolved application configuration.")
        # Revalidate rather than trusting a mutated frozen configuration.
        owned = BrowserControlConfig(
            policy=config.policy, guest_endpoint=config.guest_endpoint, purpose=config.purpose
        )
        self.service = BrowserControlService(
            guest_endpoint=owned.guest_endpoint, purpose=owned.purpose
        )
        self.coordinator = BrowserControlCoordinator(
            store=store,
            policy=owned.policy,
            purpose=owned.purpose,
            redactor=redactor,
            clock=clock,
            origin_redactor=self.service.invocation_origin_redactor,
        )
        # A Cayu server drains browser control itself, before the application,
        # because its routes and tickets close first. Resolves to whether that
        # drain settled; the application's shutdown reports it once.
        self.host_drain: asyncio.Future[bool] | None = None

    async def drain(self, *, timeout_s: float) -> bool:
        """Stop issuance and settle guest channels, then seal publication.

        The publisher stays open while the service is unsettled, because guest
        cleanup may still need to publish its fence.
        """

        # The service waits in three phases, each up to its own timeout, and no
        # owner waits longer than its bound.
        phase = min(timeout_s / 4, _MAX_PHASE_SECONDS)
        if not await self.service.drain(timeout_s=phase):
            return False
        return await self.coordinator.drain(timeout_s=phase)
