from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cayu.egress.broker import TransparentEgressBroker
from cayu.egress.errors import UnsupportedEgressError
from cayu.egress.grants import VirtualCredentialGrant
from cayu.egress.proxy_exposure import HttpProxyEndpoint
from cayu.runners.base import Runner


@dataclass
class EgressBinding:
    """The result of configuring enforced egress for one sandbox.

    ``env`` is the overlay the runner must apply to the sandbox process
    (proxy vars + CA trust). ``ca_cert_pem`` is the per-session CA the sandbox
    must trust. ``close`` tears everything down (removes networks/sidecars and
    revokes grants) and is idempotent.
    """

    env: dict[str, str] = field(default_factory=dict)
    ca_cert_pem: bytes | None = None
    runner_kind: str | None = None
    network: str | None = None
    sidecar: str | None = None
    guest_ca_path: str | None = None
    proxy_url: str | None = None
    proxy_port: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    teardown: Callable[[], Awaitable[None]] | None = None
    _closed: bool = field(default=False, init=False, repr=False)
    _proxy_endpoint: HttpProxyEndpoint | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for field_name in ("runner_kind", "network", "sidecar", "guest_ca_path"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be nonblank when set.")
        if self.proxy_port is not None and self.proxy_port <= 0:
            raise ValueError("proxy_port must be positive when set.")
        if self.proxy_url is not None:
            try:
                self._proxy_endpoint = HttpProxyEndpoint.parse(self.proxy_url)
            except ValueError as exc:
                raise ValueError(f"proxy_url is invalid: {exc}") from exc

    @property
    def proxy_endpoint(self) -> HttpProxyEndpoint | None:
        return self._proxy_endpoint

    async def close(self) -> None:
        if self._closed:
            return
        if self.teardown is not None:
            await self.teardown()
        self._closed = True


@dataclass(frozen=True)
class VirtualEgressRunnerRequest:
    """Inputs an egress adapter needs to start its enforced sandbox."""

    name: str
    runner_kind: str
    image: str
    binding: EgressBinding
    env_overlay: Mapping[str, str]
    ca_cert_host_path: str
    guest_ca_path: str
    setup_commands: tuple[str, ...]
    egress_destinations: tuple[str, ...]


class SandboxEgressAdapter(ABC):
    """Configures egress and creates the matching enforced runner.

    An adapter must either return a binding that provably routes provider
    traffic through the broker (and blocks direct egress), or raise
    ``UnsupportedEgressError``. It must never return a binding that leaves
    direct egress open — that would silently downgrade the security boundary.
    Runner creation lives on the same interface so a prepared binding cannot be
    paired with an unrelated factory that ignores its network policy.
    """

    #: Identifier of the runner family this adapter enforces.
    runner_kind: str

    @abstractmethod
    async def prepare(
        self,
        *,
        session_id: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
    ) -> EgressBinding:
        """Configure enforced egress for the session or raise."""

    @abstractmethod
    async def create_runner(self, request: VirtualEgressRunnerRequest) -> Runner:
        """Create a runner that applies this adapter's binding without downgrade."""


class UnsupportedEgressAdapter(SandboxEgressAdapter):
    """Fail-closed adapter for runners that cannot enforce egress.

    ``prepare`` always raises ``UnsupportedEgressError``. This is what makes the
    absence of a real adapter safe: virtual egress can never proceed without
    enforcement.
    """

    def __init__(self, runner_kind: str, *, reason: str | None = None) -> None:
        self.runner_kind = runner_kind
        self._reason = reason or "no enforcing egress adapter is registered"

    async def prepare(
        self,
        *,
        session_id: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
    ) -> EgressBinding:
        raise UnsupportedEgressError(
            f"Runner {self.runner_kind!r} cannot enforce virtual egress: {self._reason}. "
            "Virtual credentials refuse to downgrade to raw secret injection."
        )

    async def create_runner(self, request: VirtualEgressRunnerRequest) -> Runner:
        raise UnsupportedEgressError(
            f"Runner {self.runner_kind!r} cannot enforce virtual egress: {self._reason}. "
            "Virtual credentials refuse to downgrade to raw secret injection."
        )


class EgressAdapterRegistry:
    """Resolves a runner kind to its egress adapter, failing closed by default.

    ``resolve`` never returns ``None``: an unregistered runner kind yields an
    ``UnsupportedEgressAdapter`` whose ``prepare`` raises, so callers cannot
    accidentally skip enforcement.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, SandboxEgressAdapter] = {}

    def register(self, adapter: SandboxEgressAdapter) -> None:
        if not isinstance(adapter, SandboxEgressAdapter):
            raise TypeError("Egress adapters must be SandboxEgressAdapter instances.")
        runner_kind = adapter.runner_kind.strip()
        if not runner_kind:
            raise ValueError("Egress adapter runner_kind must be nonblank.")
        self._adapters[runner_kind] = adapter

    def resolve(self, runner_kind: str) -> SandboxEgressAdapter:
        adapter = self._adapters.get(runner_kind)
        if adapter is not None:
            return adapter
        return UnsupportedEgressAdapter(runner_kind)
