from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from cayu.egress import SandboxEgressAdapter, UnsupportedEgressError, VirtualCredentialError
from cayu.egress.broker import TransparentEgressBroker
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.egress.e2b_adapter import E2BEgressAdapter
from cayu.egress.grants import VirtualCredentialRegistry
from cayu.egress.microsandbox_adapter import MicrosandboxEgressAdapter
from cayu.egress.proxy_exposure import ExposedProxy

VerificationStatus = Literal["verified", "smoke", "skipped", "failed", "unclaimed"]
ProofSource = Literal["deterministic", "live", "nightly"]
CapabilityProof = Literal["deterministic", "live_required"]
EvidenceObservation = Literal[
    "brokered-call-succeeded",
    "session-ca-trusted",
    "public-ip-denied",
    "metadata-service-denied",
    "virtual-credential-only",
    "raw-secret-absent",
    "broker-secret-absent",
    "enforced-proxy-only",
    "trusted-host-env-absent",
]
CleanupOutcome = Literal[
    "complete",
    "complete-and-grant-revoked",
    "incomplete",
    "not-applicable",
    "unknown",
]

_SAFE_EVIDENCE_TOKEN = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_SAFE_EVIDENCE_OBSERVATIONS = frozenset(get_args(EvidenceObservation))
_SAFE_CLEANUP_OUTCOMES = frozenset(get_args(CleanupOutcome))
_SAFE_VERIFICATION_STATUSES = frozenset(get_args(VerificationStatus))
_SAFE_PROOF_SOURCES = frozenset(get_args(ProofSource))
_SAFE_REASONS = frozenset((None, "check-failed", "contract-satisfied", "prerequisites-unavailable"))


class _UnconfiguredE2BExposure:
    async def expose(self, *, local_host: str, local_port: int):  # type: ignore[no-untyped-def]
        raise UnsupportedEgressError(
            "E2B live conformance requires an explicit bounded proxy exposure."
        )


def _docker_adapter_factory(**options: Any) -> SandboxEgressAdapter:
    return DockerEgressAdapter(**options)


def _e2b_adapter_factory(**options: Any) -> SandboxEgressAdapter:
    options.setdefault("exposure", _UnconfiguredE2BExposure())
    return E2BEgressAdapter(**options)


def _microsandbox_adapter_factory(**options: Any) -> SandboxEgressAdapter:
    return MicrosandboxEgressAdapter(**options)


@dataclass
class ResourceReleaseProbe:
    registry: VirtualCredentialRegistry | None = None
    presented_value: str | None = None
    observations: list[bool] = field(default_factory=list)

    def arm(self, registry: VirtualCredentialRegistry, presented_value: str) -> None:
        self.registry = registry
        self.presented_value = presented_value

    def observe_release(self) -> None:
        if self.registry is None or self.presented_value is None:
            raise AssertionError("Deterministic release probe was not armed.")
        try:
            self.registry.lookup(self.presented_value)
        except VirtualCredentialError:
            self.observations.append(True)
        else:
            self.observations.append(False)


@dataclass(frozen=True)
class DeterministicEgressFixture:
    adapter: SandboxEgressAdapter
    release_probe: ResourceReleaseProbe


class _DeterministicDockerExec:
    def __init__(self, release_probe: ResourceReleaseProbe) -> None:
        self._release_probe = release_probe

    async def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        if argv[0] == "rm" or argv[:2] == ["network", "rm"]:
            self._release_probe.observe_release()
        return 0, ""


class _DeterministicAuthority:
    @staticmethod
    def ca_cert_pem() -> bytes:
        return b"-----BEGIN CERTIFICATE-----\n"


class _DeterministicProxyServer:
    def __init__(
        self,
        _broker: TransparentEgressBroker,
        *,
        release_probe: ResourceReleaseProbe,
        **_options: Any,
    ) -> None:
        self.authority = _DeterministicAuthority()
        self._release_probe = release_probe

    async def start(self) -> int:
        return 8123

    async def close(self) -> None:
        self._release_probe.observe_release()


class _DeterministicExposure:
    def __init__(self, release_probe: ResourceReleaseProbe) -> None:
        self._release_probe = release_probe

    async def expose(self, *, local_host: str, local_port: int) -> ExposedProxy:
        async def teardown() -> None:
            self._release_probe.observe_release()

        return ExposedProxy(
            proxy_url="http://203.0.113.10:8443",
            teardown=teardown,
            credentialless_isolated=True,
        )


def _deterministic_docker_fixture() -> DeterministicEgressFixture:
    release_probe = ResourceReleaseProbe()
    adapter = DockerEgressAdapter(
        docker_exec=_DeterministicDockerExec(release_probe),
        proxy_host="127.0.0.1",
    )
    return DeterministicEgressFixture(adapter=adapter, release_probe=release_probe)


def _deterministic_remote_fixture(
    adapter_type: type[E2BEgressAdapter] | type[MicrosandboxEgressAdapter],
) -> DeterministicEgressFixture:
    release_probe = ResourceReleaseProbe()

    def proxy_server_factory(*args: Any, **kwargs: Any) -> _DeterministicProxyServer:
        return _DeterministicProxyServer(
            *args,
            release_probe=release_probe,
            **kwargs,
        )

    adapter = adapter_type(
        exposure=_DeterministicExposure(release_probe),
        proxy_server_factory=proxy_server_factory,
    )
    return DeterministicEgressFixture(adapter=adapter, release_probe=release_probe)


def _deterministic_e2b_fixture() -> DeterministicEgressFixture:
    return _deterministic_remote_fixture(E2BEgressAdapter)


def _deterministic_microsandbox_fixture() -> DeterministicEgressFixture:
    return _deterministic_remote_fixture(MicrosandboxEgressAdapter)


@dataclass(frozen=True)
class EgressCapabilities:
    fail_closed_orchestration: CapabilityProof
    brokered_ca_trust: CapabilityProof
    public_ip_denial: CapabilityProof
    metadata_denial: CapabilityProof
    guest_non_possession: CapabilityProof
    bounded_retryable_teardown: CapabilityProof


@dataclass(frozen=True)
class LivePrerequisites:
    required_commands: tuple[str, ...] = ()
    required_modules: tuple[str, ...] = ()
    required_env: tuple[str, ...] = ()
    required_env_values: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()


AdapterFactory = Callable[..., SandboxEgressAdapter]
DeterministicFixtureFactory = Callable[[], DeterministicEgressFixture]


@dataclass(frozen=True)
class EgressConformanceRegistration:
    name: str
    runner_kind: str
    adapter_type: type[SandboxEgressAdapter]
    adapter_factory: AdapterFactory
    deterministic_fixture_factory: DeterministicFixtureFactory
    live_prerequisites: LivePrerequisites
    capabilities: EgressCapabilities
    bounded_destinations: tuple[str, ...]
    teardown_timeout_s: float
    live_proof_source: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.runner_kind.strip():
            raise ValueError("Egress conformance registration names must be nonblank.")
        if not issubclass(self.adapter_type, SandboxEgressAdapter):
            raise TypeError("Egress registration adapter_type must implement SandboxEgressAdapter.")
        if self.teardown_timeout_s <= 0:
            raise ValueError("Egress registration teardown_timeout_s must be positive.")
        if not self.bounded_destinations:
            raise ValueError("Egress registrations require bounded destinations.")

    def create_adapter(self, **options: Any) -> SandboxEgressAdapter:
        adapter = self.adapter_factory(**options)
        if type(adapter) is not self.adapter_type:
            raise TypeError(
                f"Egress registration {self.name!r} created {type(adapter).__name__}, "
                f"expected {self.adapter_type.__name__}."
            )
        return adapter

    def create_deterministic_fixture(self) -> DeterministicEgressFixture:
        fixture = self.deterministic_fixture_factory()
        if type(fixture.adapter) is not self.adapter_type:
            raise TypeError(
                f"Egress registration {self.name!r} deterministic fixture created "
                f"{type(fixture.adapter).__name__}, expected {self.adapter_type.__name__}."
            )
        return fixture


@dataclass(frozen=True)
class EgressScenarioEvidence:
    adapter: str
    scenario: str
    status: VerificationStatus
    proof_source: ProofSource
    observations: tuple[EvidenceObservation, ...]
    cleanup_outcome: CleanupOutcome
    duration_ms: int
    reason: Literal["contract-satisfied"] | None = None

    def __post_init__(self) -> None:
        for field_name in ("adapter", "scenario"):
            value = getattr(self, field_name)
            if type(value) is not str:
                raise TypeError(f"Egress evidence {field_name} must be a string.")
            if len(value) > 80 or _SAFE_EVIDENCE_TOKEN.fullmatch(value) is None:
                raise ValueError(
                    f"Egress evidence {field_name} must be a safe 1-80 character token."
                )
        observations = tuple(self.observations)
        if len(observations) > 8:
            raise ValueError("Egress evidence accepts at most eight bounded observations.")
        if any(
            type(value) is not str or value not in _SAFE_EVIDENCE_OBSERVATIONS
            for value in observations
        ):
            raise ValueError("Egress evidence contains an unsupported observation.")
        object.__setattr__(self, "observations", observations)
        if type(self.status) is not str or self.status not in _SAFE_VERIFICATION_STATUSES:
            raise ValueError("Egress evidence contains an unsupported status.")
        if type(self.proof_source) is not str or self.proof_source not in _SAFE_PROOF_SOURCES:
            raise ValueError("Egress evidence contains an unsupported proof source.")
        if (
            self.reason is not None and type(self.reason) is not str
        ) or self.reason not in _SAFE_REASONS:
            raise ValueError("Egress evidence contains an unsupported reason.")
        if (
            type(self.cleanup_outcome) is not str
            or self.cleanup_outcome not in _SAFE_CLEANUP_OUTCOMES
        ):
            raise ValueError("Egress evidence contains an unsupported cleanup outcome.")
        if type(self.duration_ms) is not int:
            raise TypeError("Egress evidence duration_ms must be an integer.")
        if self.duration_ms < 0 or self.duration_ms > 600_000:
            raise ValueError("Egress evidence duration_ms must be between 0 and 600000.")


def emit_egress_nightly_evidence(evidence: Sequence[EgressScenarioEvidence]) -> None:
    """Emit one bounded, secret-safe evidence envelope for nightly ingestion."""

    records = tuple(evidence)
    if not 1 <= len(records) <= 8:
        raise ValueError("Egress nightly evidence requires between one and eight records.")
    if any(not isinstance(record, EgressScenarioEvidence) for record in records):
        raise TypeError("Egress nightly evidence records must be EgressScenarioEvidence values.")
    if len({record.adapter for record in records}) != 1:
        raise ValueError("Egress nightly evidence records must describe one adapter.")
    payload = {
        "schema": "cayu.egress_conformance.v1",
        "records": [
            {
                "adapter": record.adapter,
                "scenario": record.scenario,
                "status": record.status,
                "proof_source": record.proof_source,
                "observations": list(record.observations),
                "cleanup_outcome": record.cleanup_outcome,
                "duration_ms": record.duration_ms,
                "reason": record.reason,
            }
            for record in records
        ],
    }
    print("CAYU_NIGHTLY_EVIDENCE=" + json.dumps(payload, sort_keys=True, separators=(",", ":")))


@contextmanager
def egress_nightly_failure_boundary(adapter: str) -> Iterator[None]:
    """Emit a conservative typed live-failure record before propagating an error."""

    try:
        yield
    except BaseException:
        emit_egress_nightly_evidence(
            (
                EgressScenarioEvidence(
                    adapter=adapter,
                    scenario="live-security-conformance",
                    status="failed",
                    proof_source="live",
                    observations=(),
                    cleanup_outcome="unknown",
                    duration_ms=0,
                    reason="check-failed",
                ),
            )
        )
        raise


_LIVE_SECURITY = EgressCapabilities(
    fail_closed_orchestration="deterministic",
    brokered_ca_trust="live_required",
    public_ip_denial="live_required",
    metadata_denial="live_required",
    guest_non_possession="live_required",
    bounded_retryable_teardown="deterministic",
)

EGRESS_CONFORMANCE_REGISTRATIONS = (
    EgressConformanceRegistration(
        name="docker",
        runner_kind="docker",
        adapter_type=DockerEgressAdapter,
        adapter_factory=_docker_adapter_factory,
        deterministic_fixture_factory=_deterministic_docker_fixture,
        live_prerequisites=LivePrerequisites(
            required_commands=("docker",),
            required_modules=("cryptography",),
            notes=("Runs in the existing uncredentialed Docker live CI tier.",),
        ),
        capabilities=_LIVE_SECURITY,
        bounded_destinations=("api.stripe.com:443", "1.1.1.1:443", "169.254.169.254:80"),
        teardown_timeout_s=15,
        live_proof_source="tests/egress/test_docker_egress_e2e.py",
    ),
    EgressConformanceRegistration(
        name="e2b",
        runner_kind="e2b",
        adapter_type=E2BEgressAdapter,
        adapter_factory=_e2b_adapter_factory,
        deterministic_fixture_factory=_deterministic_e2b_fixture,
        live_prerequisites=LivePrerequisites(
            required_modules=("cryptography", "e2b"),
            required_env=(
                "E2B_API_KEY",
                "CAYU_E2B_PROXY_EXPOSURE_COMMAND",
                "CAYU_E2B_PROXY_URL",
            ),
            required_env_values=(("CAYU_RUN_E2B_EGRESS_E2E", "1"),),
        ),
        capabilities=_LIVE_SECURITY,
        bounded_destinations=(
            "api.stripe.com:443",
            "1.1.1.1:443",
            "8.8.8.8:443",
            "169.254.169.254:80",
        ),
        teardown_timeout_s=15,
        live_proof_source="tests/egress/test_e2b_egress_e2e.py",
    ),
    EgressConformanceRegistration(
        name="microsandbox",
        runner_kind="microsandbox",
        adapter_type=MicrosandboxEgressAdapter,
        adapter_factory=_microsandbox_adapter_factory,
        deterministic_fixture_factory=_deterministic_microsandbox_fixture,
        live_prerequisites=LivePrerequisites(
            required_modules=("cryptography", "microsandbox"),
            required_env_values=(("CAYU_RUN_MICROSANDBOX_EGRESS_E2E", "1"),),
            notes=("Requires an intentionally available local Microsandbox runtime.",),
        ),
        capabilities=_LIVE_SECURITY,
        bounded_destinations=(
            "api.stripe.com:443",
            "1.1.1.1:443",
            "8.8.8.8:443",
            "169.254.169.254:80",
        ),
        teardown_timeout_s=15,
        live_proof_source="tests/egress/test_microsandbox_egress_e2e.py",
    ),
)


def registration_for(name: str) -> EgressConformanceRegistration:
    for registration in EGRESS_CONFORMANCE_REGISTRATIONS:
        if registration.name == name:
            return registration
    raise KeyError(name)
