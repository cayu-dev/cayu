"""Application composition for existing browser control owners, not another driver."""

from collections.abc import Callable
from datetime import datetime

from cayu.runtime._browser_control_coordinator import BrowserControlCoordinator
from cayu.runtime._browser_control_service import BrowserControlService
from cayu.runtime.browser_control_config import BrowserControlConfig
from cayu.runtime.sessions import SessionStore
from cayu.vaults.redaction import SecretRedactor


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
