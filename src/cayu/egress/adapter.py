from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal, TypeVar

from cayu._exception_groups import iter_exception_tree
from cayu._task_wait import await_shielded_task_outcome, consume_pending_task_cancellation
from cayu.egress.authority import (
    EgressAuthorityCutoverReceipt,
    EgressAuthorityCutoverStrategy,
    EgressAuthorityIdentity,
)
from cayu.egress.broker import TransparentEgressBroker
from cayu.egress.capabilities import EgressCapabilityEvidence
from cayu.egress.errors import (
    UnsupportedEgressAuthorityCutoverError,
    UnsupportedEgressError,
    UnsupportedEgressReconnectError,
)
from cayu.egress.grants import VirtualCredentialGrant
from cayu.egress.proxy_exposure import HttpProxyEndpoint
from cayu.environments.admission import (
    EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
)
from cayu.runners.base import Runner

DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS = 15.0
_CleanupResultT = TypeVar("_CleanupResultT")
_ExecutionCapabilityPosture = Literal[
    "available",
    "live_verified",
    "unverified",
    "unsupported",
]


@dataclass(frozen=True)
class RunnerFinalizationResult:
    """Positive lifecycle evidence after runner finalization."""

    workspace_mutations_quiescent: bool
    allocation_preserved: bool = False

    def __post_init__(self) -> None:
        if type(self.workspace_mutations_quiescent) is not bool:
            raise TypeError(
                "RunnerFinalizationResult workspace_mutations_quiescent must be a bool."
            )
        if type(self.allocation_preserved) is not bool:
            raise TypeError("RunnerFinalizationResult allocation_preserved must be a bool.")


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
    task: asyncio.Task[_CleanupResultT],
    *,
    timeout_s: float,
    timeout_message: str,
    cancellation: asyncio.CancelledError | None = None,
) -> bool:
    """Finish one cleanup task despite cancellation, or report a bounded timeout."""

    outcome = await await_shielded_task_outcome(
        task,
        cancellation=cancellation,
        timeout_s=timeout_s,
    )

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

    return next(
        (
            candidate
            for candidate in iter_exception_tree(error)
            if isinstance(candidate, asyncio.CancelledError)
        ),
        None,
    )


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
    certificate_authority: Any = field(default=None, repr=False)
    adopt_certificate_authority: Callable[[Any], None] | None = field(
        default=None,
        repr=False,
    )
    relinquish_certificate_authority: Callable[[Any], None] | None = field(
        default=None,
        repr=False,
    )
    teardown_timeout_s: float = DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS
    authority_fingerprint: str | None = None
    authority_generation: int | None = None
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
        if (self.authority_fingerprint is None) != (self.authority_generation is None):
            raise ValueError("Egress binding authority identity must be complete or absent.")
        if self.authority_fingerprint is not None and (
            len(self.authority_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.authority_fingerprint)
        ):
            raise ValueError("Egress binding authority fingerprint must be a SHA-256 digest.")
        if self.authority_generation is not None and (
            type(self.authority_generation) is not int or self.authority_generation < 1
        ):
            raise ValueError("Egress binding authority generation must be a positive integer.")

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

    def transfer_certificate_authority_to(self, replacement: EgressBinding) -> None:
        """Move teardown ownership to a staged binding using the exact same CA."""

        authority = self.certificate_authority
        if authority is None or replacement.certificate_authority is not authority:
            raise RuntimeError("Fresh egress paths must retain the exact trusted session CA.")
        if (
            self.relinquish_certificate_authority is None
            or replacement.adopt_certificate_authority is None
        ):
            raise RuntimeError("Egress binding does not support certificate-authority transfer.")
        replacement.adopt_certificate_authority(authority)
        try:
            self.relinquish_certificate_authority(authority)
        except BaseException:
            if replacement.relinquish_certificate_authority is not None:
                replacement.relinquish_certificate_authority(authority)
            raise

    def bind_authority(self, authority: EgressAuthorityIdentity) -> None:
        """Freeze the exact authority generation enforced by this live path."""

        if type(authority) is not EgressAuthorityIdentity:
            raise TypeError("Egress binding authority must be EgressAuthorityIdentity.")
        if self.runner_kind != authority.runner_kind:
            raise ValueError("Egress binding runner kind does not match its authority.")
        if self.authority_fingerprint is not None:
            if (
                self.authority_fingerprint == authority.fingerprint
                and self.authority_generation == authority.generation
            ):
                return
            raise RuntimeError("Egress binding authority is immutable once assigned.")
        self.authority_fingerprint = authority.fingerprint
        self.authority_generation = authority.generation


def retain_predecessor_binding_cleanup(
    replacement: EgressBinding,
    predecessor: EgressBinding,
) -> None:
    """Make one replacement retry both cleanups after old-path retirement."""

    if type(replacement) is not EgressBinding or type(predecessor) is not EgressBinding:
        raise TypeError("Retained egress cleanup requires exact EgressBinding values.")
    replacement_teardown = replacement.teardown

    async def teardown() -> None:
        async def close_replacement() -> None:
            if replacement_teardown is not None:
                await replacement_teardown()

        results = await asyncio.gather(
            predecessor.close(),
            close_replacement(),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise BaseExceptionGroup(
                "Retained egress predecessor cleanup failed.",
                errors,
            )

    replacement.teardown = teardown


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
    env_overlay_secret_values_present: bool | None = None
    allocation_id: str | None = None
    host_workspace_path: str | None = None


@dataclass(frozen=True)
class EgressAuthorityCutoverRequest:
    """Trusted live inputs for one quiescent same-allocation authority rotation."""

    session_id: str
    environment_name: str
    owner_fingerprint: str
    environment_fingerprint: str
    runner: Runner = field(repr=False)
    current_binding: EgressBinding = field(repr=False)
    expected_authority: EgressAuthorityIdentity
    target_authority: EgressAuthorityIdentity
    target_broker: TransparentEgressBroker = field(repr=False)
    target_grants: tuple[VirtualCredentialGrant, ...] = field(repr=False)
    target_env_overlay: Mapping[str, str] = field(repr=False)
    target_egress_destinations: tuple[str, ...]
    revoke_current_authority: Callable[[], Awaitable[bool]] = field(repr=False)
    ca_cert_host_path: str
    guest_ca_path: str
    invocation_quiescent: bool

    def __post_init__(self) -> None:
        if not self.session_id.strip() or not self.environment_name.strip():
            raise ValueError("Egress authority cutover session/environment must be nonblank.")
        if len(self.owner_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.owner_fingerprint
        ):
            raise ValueError("Egress authority cutover owner fingerprint is invalid.")
        if len(self.environment_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.environment_fingerprint
        ):
            raise ValueError("Egress authority cutover environment fingerprint is invalid.")
        if not isinstance(self.runner, Runner):
            raise TypeError("Egress authority cutover runner must be a Runner.")
        if type(self.current_binding) is not EgressBinding:
            raise TypeError("Egress authority cutover requires an exact EgressBinding.")
        if (
            type(self.expected_authority) is not EgressAuthorityIdentity
            or type(self.target_authority) is not EgressAuthorityIdentity
        ):
            raise TypeError("Egress authority cutover identities must be typed.")
        if not isinstance(self.target_broker, TransparentEgressBroker):
            raise TypeError("Egress authority cutover target_broker is invalid.")
        if type(self.target_grants) is not tuple:
            raise TypeError("Egress authority cutover target_grants must be a tuple.")
        validate_grant_scope(session_id=self.session_id, grants=self.target_grants)
        if not callable(self.revoke_current_authority):
            raise TypeError("Egress authority cutover revoker must be callable.")
        if type(self.target_egress_destinations) is not tuple or not (
            self.target_egress_destinations
        ):
            raise ValueError("Egress authority cutover requires target destinations.")
        if not self.ca_cert_host_path.strip() or not self.guest_ca_path.strip():
            raise ValueError("Egress authority cutover CA paths must be nonblank.")
        object.__setattr__(
            self,
            "target_env_overlay",
            MappingProxyType(dict(self.target_env_overlay)),
        )
        if type(self.invocation_quiescent) is not bool or not self.invocation_quiescent:
            raise ValueError("Egress authority cutover requires a proven quiescent invocation.")
        if self.target_authority.generation <= self.expected_authority.generation:
            raise ValueError("Egress authority target generation must increase.")
        if self.target_authority.fingerprint == self.expected_authority.fingerprint:
            raise ValueError("Egress authority target identity must be distinct.")
        if self.expected_authority.runner_kind != self.target_authority.runner_kind:
            raise ValueError("Same-allocation cutover cannot change runner kind.")


@dataclass(frozen=True)
class EgressAuthorityCutoverResult:
    """A live replacement binding paired with its durable-safe activation receipt."""

    binding: EgressBinding = field(repr=False)
    receipt: EgressAuthorityCutoverReceipt
    cancellation: asyncio.CancelledError | None = field(default=None, repr=False)
    cancellation_requests_consumed: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.cancellation is not None and not isinstance(
            self.cancellation,
            asyncio.CancelledError,
        ):
            raise TypeError("Cutover result cancellation must be CancelledError or None.")
        if (
            type(self.cancellation_requests_consumed) is not int
            or self.cancellation_requests_consumed < 0
        ):
            raise ValueError("Cutover cancellation request count must be non-negative.")


@dataclass(frozen=True)
class EgressAuthorityRenewalRequest:
    """Trusted inputs for renewing an unchanged authority on one parked allocation."""

    session_id: str
    environment_name: str
    environment_fingerprint: str
    runner: Runner = field(repr=False)
    current_binding: EgressBinding = field(repr=False)
    authority: EgressAuthorityIdentity
    renewed_grants: tuple[VirtualCredentialGrant, ...] = field(repr=False)
    renewed_env_overlay: Mapping[str, str] = field(repr=False)
    egress_destinations: tuple[str, ...]
    ca_cert_host_path: str
    guest_ca_path: str
    invocation_quiescent: bool

    def __post_init__(self) -> None:
        if not self.session_id.strip() or not self.environment_name.strip():
            raise ValueError("Egress authority renewal session/environment must be nonblank.")
        if len(self.environment_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.environment_fingerprint
        ):
            raise ValueError("Egress authority renewal environment fingerprint is invalid.")
        if not isinstance(self.runner, Runner):
            raise TypeError("Egress authority renewal runner must be a Runner.")
        if type(self.current_binding) is not EgressBinding:
            raise TypeError("Egress authority renewal requires an exact EgressBinding.")
        if type(self.authority) is not EgressAuthorityIdentity:
            raise TypeError("Egress authority renewal identity must be typed.")
        if type(self.renewed_grants) is not tuple:
            raise TypeError("Egress authority renewal grants must be a tuple.")
        validate_grant_scope(session_id=self.session_id, grants=self.renewed_grants)
        if type(self.egress_destinations) is not tuple or not self.egress_destinations:
            raise ValueError("Egress authority renewal requires destinations.")
        if not self.ca_cert_host_path.strip() or not self.guest_ca_path.strip():
            raise ValueError("Egress authority renewal CA paths must be nonblank.")
        object.__setattr__(
            self,
            "renewed_env_overlay",
            MappingProxyType(dict(self.renewed_env_overlay)),
        )
        if type(self.invocation_quiescent) is not bool or not self.invocation_quiescent:
            raise ValueError("Egress authority renewal requires a proven quiescent invocation.")
        if (
            self.current_binding.authority_fingerprint != self.authority.fingerprint
            or self.current_binding.authority_generation != self.authority.generation
            or self.current_binding.runner_kind != self.authority.runner_kind
        ):
            raise ValueError("Egress authority renewal must preserve the exact active identity.")


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
    #: Whether runner creation mutates a process-external provider.
    #:
    #: Concrete adapters must classify this explicitly. ``None`` fails closed
    #: before CREATE so an older or custom remote adapter cannot silently retain
    #: an orphan-allocation crash window.
    process_external_allocation: bool | None = None
    #: Stable durable-allocation scope for process-external creation.  Both
    #: fields must be supplied together by adapters implementing
    #: ``create_or_recover_runner``.
    allocation_provider: str | None = None
    allocation_adapter_generation: str | None = None
    #: True only when same-sandbox reconnect has durable single-owner semantics.
    supports_reconnect: bool = False
    #: How this adapter can adopt a new egress authority at a quiescent boundary.
    #: Missing declarations remain fail-closed rather than implying a hot update.
    egress_authority_cutover_strategy: EgressAuthorityCutoverStrategy = (
        EgressAuthorityCutoverStrategy.UNSUPPORTED
    )

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

    async def create_or_recover_runner(
        self,
        request: VirtualEgressRunnerRequest,
        *,
        allow_create: bool,
    ) -> Runner:
        """Create once or recover the intent-owned external allocation.

        ``allow_create`` is true only for the worker that durably advanced the
        allocation intent from PREPARED to DISPATCHED.  Recovery workers must
        perform positive lookup and must never infer that absence authorizes a
        second provider submission.
        """

        del request, allow_create
        raise UnsupportedEgressError(
            f"Runner {self.runner_kind!r} does not implement durable create-or-lookup."
        )

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

    async def cutover_authority(
        self,
        request: EgressAuthorityCutoverRequest,
    ) -> EgressAuthorityCutoverResult:
        """Replace active enforcement at a quiescent boundary or fail closed."""

        del request
        raise UnsupportedEgressAuthorityCutoverError(
            f"Runner {self.runner_kind!r} does not support governed egress cutover."
        )

    async def renew_authority(self, request: EgressAuthorityRenewalRequest) -> str:
        """Verify renewed grants and routes on an unchanged parked allocation."""

        del request
        raise UnsupportedEgressAuthorityCutoverError(
            f"Runner {self.runner_kind!r} does not support governed egress renewal."
        )

    async def egress_environment_fingerprint(self, runner: Runner) -> str:
        """Return the exact backend allocation identity used by cutover receipts."""

        del runner
        raise UnsupportedEgressAuthorityCutoverError(
            f"Runner {self.runner_kind!r} cannot prove a stable egress environment identity."
        )

    async def reconcile_authority_cutover(
        self,
        request: EgressAuthorityCutoverRequest,
    ) -> EgressAuthorityCutoverReceipt | None:
        """Read back one exact cutover without trusting caller-produced proof."""

        del request
        raise UnsupportedEgressAuthorityCutoverError(
            f"Runner {self.runner_kind!r} cannot reconcile governed egress cutover."
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

    async def finalize_runner(
        self,
        runner: Runner,
        *,
        outcome: str | None,
    ) -> RunnerFinalizationResult:
        """Map an outcome to lifecycle cleanup without inferring remote quiescence."""
        await runner.close()
        return RunnerFinalizationResult(workspace_mutations_quiescent=False)

    async def finalize_runner_for_binding(
        self,
        runner: Runner,
        *,
        outcome: str | None,
    ) -> RunnerFinalizationResult:
        """Finalize a runner while positively quiescing workspace mutation.

        Reconnectable adapters should override this when they can stop or
        suspend an allocation without destroying its durable identity. The
        default fails closed by using terminal cleanup.
        """

        return await self.finalize_runner(
            runner,
            outcome=None if outcome == "interrupted" else outcome,
        )

    async def park_runner_for_authority_adoption(
        self,
        runner: Runner,
    ) -> RunnerFinalizationResult:
        """Fence workload execution while retaining the exact live allocation."""

        del runner
        raise UnsupportedEgressAuthorityCutoverError(
            f"Runner {self.runner_kind!r} cannot park an allocation for governed adoption."
        )


class UnsupportedEgressAdapter(SandboxEgressAdapter):
    """Fail-closed adapter for runners that cannot enforce egress.

    ``prepare`` always raises ``UnsupportedEgressError``. This is what makes the
    absence of a real adapter safe: virtual egress can never proceed without
    enforcement.
    """

    process_external_allocation = False

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
