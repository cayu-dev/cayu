"""Explicit application-owned browser operator configuration."""

from dataclasses import dataclass, field

from cayu.runtime.browser_control import BrowserControlPolicy, BrowserOperatorPurpose
from cayu.tools._browser_control_transport import validate_control_endpoint


@dataclass(frozen=True, slots=True)
class BrowserControlConfig:
    """Opt-in policy and browser-reachable TLS control endpoint.

    Authentication and transport configuration alone never grant browser access.
    Operator permissions are evaluated by this policy for each exact operation.
    """

    policy: BrowserControlPolicy = field(repr=False)
    guest_endpoint: str = field(repr=False)
    purpose: BrowserOperatorPurpose = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, BrowserControlPolicy):
            raise TypeError("Browser control requires an application policy.")
        validate_control_endpoint(self.guest_endpoint)
        object.__setattr__(self, "purpose", BrowserOperatorPurpose.model_validate(self.purpose))
