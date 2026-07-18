from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any, Literal

from cayu._task_wait import await_shielded_task_outcome, consume_pending_task_cancellation
from cayu.egress.broker import TransparentEgressBroker
from cayu.egress.capabilities import EgressCapabilityEvidence
from cayu.egress.errors import UnsupportedEgressError, UnsupportedEgressReconnectError
from cayu.egress.grants import VirtualCredentialGrant
from cayu.egress.proxy_exposure import HttpProxyEndpoint
from cayu.environments.admission import (
    EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
)
from cayu.runners.base import Runner

DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS = 15.0
_ExecutionCapabilityPosture = Literal[
    "available",
    "live_verified",
    "unverified",
    "unsupported",
]


def _virtual_egress_execution_capability_evidence(
    *,
    runner_kind: str,
    runner_ready: bool,
    preflight_observed_at: datetime | None,
    untrusted_isolation: bool,
    credential_non_possession_posture: _ExecutionCapabilityPosture,
    guest_privilege: _ExecutionCapabilityPosture,
    unprivileged_guest: _ExecutionCapabilityPosture,
    host_filesystem_isolation: bool,
    reconnect: bool,
    network_unverified: bool = False,
    cancellation_confirmed: bool = True,
) -> ExecutionCapabilityEvidence:
    """Build the shared admission claims for one enforced virtual-egress path."""

    if preflight_observed_at is not None:
        if not runner_ready:
            raise ValueError("Preflight evidence requires a ready runner.")
        if preflight_observed_at.tzinfo is None or preflight_observed_at.utcoffset() is None:
            raise ValueError("Preflight evidence timestamps must include a timezone.")
        observed_at = preflight_observed_at.astimezone(UTC)
        valid_until = observed_at + timedelta(seconds=EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS)
    else:
        observed_at = None
        valid_until = None
    claims: list[ExecutionCapabilityClaim] = [
        (
            _declared_or_available("untrusted_code_isolation", runner_ready=runner_ready)
            if untrusted_isolation
            else ExecutionCapabilityClaim.unsupported(
                "untrusted_code_isolation",
                reason_code="isolation_boundary_unsupported",
                remediation_code="select_isolated_execution",
            )
        ),
        _execution_posture_claim(
            "real_credential_non_possession",
            posture=credential_non_possession_posture,
            runner_ready=runner_ready,
            observed_at=observed_at,
            valid_until=valid_until,
            unverified_reason_code="credential_boundary_unverified",
            unverified_remediation_code="verify_guest_credential_boundary",
        ),
        (
            ExecutionCapabilityClaim.unverified(
                "deny_by_default_network",
                reason_code="network_boundary_unverified",
                remediation_code="enable_network_preflight",
            )
            if network_unverified
            else (
                _preflight_claim(
                    "deny_by_default_network",
                    observation="denied",
                    runner_ready=runner_ready,
                    observed_at=observed_at,
                    valid_until=valid_until,
                )
            )
        ),
        _preflight_claim(
            "brokered_egress",
            observation="reachable",
            runner_ready=runner_ready,
            observed_at=observed_at,
            valid_until=valid_until,
        ),
        _execution_posture_claim(
            "guest_privilege_containment",
            posture=guest_privilege,
            runner_ready=runner_ready,
            observed_at=observed_at,
            valid_until=valid_until,
        ),
        _execution_posture_claim(
            "unprivileged_guest",
            posture=unprivileged_guest,
            runner_ready=runner_ready,
            observed_at=observed_at,
            valid_until=valid_until,
        ),
        (
            _declared_or_available("host_filesystem_isolation", runner_ready=runner_ready)
            if host_filesystem_isolation
            else ExecutionCapabilityClaim.unsupported(
                "host_filesystem_isolation",
                reason_code="host_filesystem_boundary_unsupported",
                remediation_code="select_isolated_execution",
            )
        ),
        (
            _declared_or_available("confirmed_cancellation", runner_ready=runner_ready)
            if cancellation_confirmed
            else ExecutionCapabilityClaim.unsupported(
                "confirmed_cancellation",
                reason_code="cancellation_cleanup_disabled",
                remediation_code="enable_cancellation_cleanup",
            )
        ),
        _declared_or_available("confirmed_cleanup", runner_ready=runner_ready),
        (
            _declared_or_available("reconnect", runner_ready=runner_ready)
            if reconnect
            else ExecutionCapabilityClaim.unsupported(
                "reconnect",
                reason_code="reconnect_unsupported",
                remediation_code="select_reconnectable_execution",
            )
        ),
    ]
    return ExecutionCapabilityEvidence(subject=runner_kind, claims=tuple(claims))


def _execution_posture_claim(
    capability: str,
    *,
    posture: _ExecutionCapabilityPosture,
    runner_ready: bool,
    observed_at: datetime | None,
    valid_until: datetime | None,
    unverified_reason_code: str = "guest_boundary_unverified",
    unverified_remediation_code: str = "enable_guest_boundary_preflight",
) -> ExecutionCapabilityClaim:
    if posture == "live_verified":
        if not runner_ready or observed_at is None or valid_until is None:
            return ExecutionCapabilityClaim.declared(capability)
        return ExecutionCapabilityClaim.live_verified(
            capability,
            observation="supported",
            observed_at=observed_at,
            valid_until=valid_until,
        )
    if posture == "available":
        return _declared_or_available(capability, runner_ready=runner_ready)
    if posture == "unverified":
        return ExecutionCapabilityClaim.unverified(
            capability,
            reason_code=unverified_reason_code,
            remediation_code=unverified_remediation_code,
        )
    return ExecutionCapabilityClaim.unsupported(
        capability,
        reason_code="guest_boundary_unsupported",
        remediation_code="select_hardened_execution",
    )


def _preflight_claim(
    capability: str,
    *,
    observation: Literal["denied", "reachable", "supported"],
    runner_ready: bool,
    observed_at: datetime | None,
    valid_until: datetime | None,
) -> ExecutionCapabilityClaim:
    if not runner_ready or observed_at is None or valid_until is None:
        return ExecutionCapabilityClaim.declared(capability)
    return ExecutionCapabilityClaim.live_verified(
        capability,
        observation=observation,
        observed_at=observed_at,
        valid_until=valid_until,
    )


def _declared_or_available(
    capability: str,
    *,
    runner_ready: bool,
) -> ExecutionCapabilityClaim:
    if runner_ready:
        return ExecutionCapabilityClaim.available(capability)
    return ExecutionCapabilityClaim.declared(capability)


async def _await_bounded_cleanup_task(
    task: asyncio.Task[None],
    *,
    timeout_s: float,
    timeout_message: str,
) -> bool:
    """Finish one cleanup task despite cancellation, or report a bounded timeout."""

    outcome = await await_shielded_task_outcome(task, timeout_s=timeout_s)

    def timeout_failure(
        cancellation: asyncio.CancelledError | None,
    ) -> TimeoutError | BaseExceptionGroup:
        timeout_error = TimeoutError(timeout_message)
        if cancellation is None:
            return timeout_error
        return BaseExceptionGroup(
            "Cleanup timed out after caller cancellation.",
            [cancellation, timeout_error],
        )

    if outcome.timed_out:
        raise timeout_failure(outcome.cancellation)
    if outcome.error is not None:
        if isinstance(outcome.error, asyncio.CancelledError):
            if outcome.cancellation is not None:
                raise outcome.cancellation from outcome.error
            raise outcome.error
        if outcome.cancellation is not None:
            raise BaseExceptionGroup(
                "Cleanup failed after caller cancellation.",
                [outcome.cancellation, outcome.error],
            ) from outcome.error
        raise outcome.error
    return outcome.cancellation is not None


def _explicit_cleanup_cancellation(
    error: BaseException,
) -> asyncio.CancelledError | None:
    """Find cancellation carried by cleanup without following stale causes."""

    if isinstance(error, asyncio.CancelledError):
        return error
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            cancellation = _explicit_cleanup_cancellation(child)
            if cancellation is not None:
                return cancellation
    return None


def _consume_accounted_task_cancellation(error: BaseException) -> None:
    """Normalize task cancellation already represented by a primary error."""

    if _explicit_cleanup_cancellation(error) is not None:
        consume_pending_task_cancellation()


def _raise_primary_with_cleanup_cancellation(
    primary_error: BaseException,
    cleanup_error: BaseException,
    *,
    message: str,
) -> None:
    """Retain a primary failure when its rollback also carries cancellation."""

    cancellation = _explicit_cleanup_cancellation(primary_error)
    if cancellation is None:
        cancellation = _explicit_cleanup_cancellation(cleanup_error)
    if cancellation is None:
        return
    raise BaseExceptionGroup(message, [primary_error, cleanup_error]) from cancellation


def validate_grant_scope(
    *,
    session_id: str,
    grants: Sequence[VirtualCredentialGrant],
) -> None:
    """Reject grants minted for a different session before allocating resources."""

    if any(grant.session_id != session_id for grant in grants):
        raise UnsupportedEgressError(
            "Virtual-egress grants do not belong to the requested session."
        )


@dataclass
class EgressBinding:
    """The result of configuring enforced egress for one runner workload.

    ``env`` is the overlay the runner must apply to the workload process
    (proxy vars + CA trust). ``ca_cert_pem`` is the per-session CA the workload
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
    teardown_timeout_s: float = DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS
    _closed: bool = field(default=False, init=False, repr=False)
    _proxy_endpoint: HttpProxyEndpoint | None = field(default=None, init=False, repr=False)
    _teardown_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

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
        if type(self.teardown_timeout_s) not in {int, float}:
            raise TypeError("teardown_timeout_s must be numeric.")
        if not isfinite(self.teardown_timeout_s) or self.teardown_timeout_s <= 0:
            raise ValueError("teardown_timeout_s must be finite and greater than zero.")
        self.teardown_timeout_s = float(self.teardown_timeout_s)

    @property
    def proxy_endpoint(self) -> HttpProxyEndpoint | None:
        return self._proxy_endpoint

    async def close(self) -> None:
        if self._closed:
            return
        if self.teardown is None:
            self._closed = True
            return
        if self._teardown_task is None:
            teardown = self.teardown

            async def run_teardown() -> None:
                await teardown()

            self._teardown_task = asyncio.create_task(run_teardown())
        task = self._teardown_task
        try:
            cancelled = await _await_bounded_cleanup_task(
                task,
                timeout_s=self.teardown_timeout_s,
                timeout_message=(
                    f"Egress teardown did not complete within {self.teardown_timeout_s:g} seconds."
                ),
            )
        except BaseException:
            if task.done() and self._teardown_task is task:
                self._teardown_task = None
            raise
        self._closed = True
        if cancelled:
            raise asyncio.CancelledError()


@dataclass(frozen=True)
class VirtualEgressRunnerRequest:
    """Inputs an egress adapter needs to start its enforced runner."""

    name: str
    runner_kind: str
    image: str
    binding: EgressBinding
    env_overlay: Mapping[str, str]
    ca_cert_host_path: str
    guest_ca_path: str
    setup_commands: tuple[str, ...]
    egress_destinations: tuple[str, ...]
    session_id: str | None = None
    parent_session_id: str | None = None
    reconnect_metadata: Mapping[str, Any] = field(default_factory=dict)
    environment_name: str | None = None


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
    #: True only when same-sandbox reconnect has durable single-owner semantics.
    supports_reconnect: bool = False

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

    async def prepare_reconnect(
        self,
        *,
        session_id: str,
        environment_name: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
        reconnect_metadata: Mapping[str, Any],
    ) -> EgressBinding:
        """Re-establish enforcement for an existing sandbox or fail closed."""
        del session_id, environment_name, grants, broker, reconnect_metadata
        raise UnsupportedEgressReconnectError(
            f"Runner {self.runner_kind!r} does not support virtual-egress reconnect. "
            "The application must explicitly rebuild the environment."
        )

    def reconnect_metadata(self, runner: Runner) -> dict[str, Any]:
        """Return durable identity required to reattach to ``runner``."""
        return {}

    def capability_evidence(self, runner: Runner) -> EgressCapabilityEvidence:
        """Return typed runtime evidence for capabilities proven by ``runner``."""
        return EgressCapabilityEvidence.unclaimed(self.runner_kind)

    def execution_capability_evidence(
        self,
        runner: Runner | None = None,
    ) -> ExecutionCapabilityEvidence:
        """Return admission evidence before creation or before runner exposure.

        Implementations must remain side-effect free when ``runner`` is ``None`` so
        the first admission gate cannot create provider resources while gathering
        evidence.
        """

        del runner
        return ExecutionCapabilityEvidence.unclaimed(self.runner_kind)

    def configuration_metadata(self) -> dict[str, Any]:
        """Return JSON-safe configured intent without claiming runtime proof."""
        return {}

    def validate_reconnect_metadata(
        self,
        reconnect_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Allowlist and normalize this adapter's non-secret durable identity."""
        del reconnect_metadata
        raise UnsupportedEgressReconnectError(
            f"Runner {self.runner_kind!r} does not support virtual-egress reconnect. "
            "The application must explicitly rebuild the environment."
        )

    async def finalize_runner(self, runner: Runner, *, outcome: str | None) -> None:
        """Map a session outcome to the runner's lifecycle action."""
        await runner.close()


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
