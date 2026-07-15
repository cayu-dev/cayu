from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest
from tests.egress_conformance import (
    EGRESS_CONFORMANCE_REGISTRATIONS,
    EgressScenarioEvidence,
    egress_nightly_failure_boundary,
    emit_egress_nightly_evidence,
)

from cayu.egress import (
    EgressAdapterRegistry,
    EgressBinding,
    HttpEgressPolicy,
    SandboxEgressAdapter,
    TransparentEgressBroker,
    UnsupportedEgressAdapter,
    UnsupportedEgressError,
    VirtualCredentialError,
    VirtualCredentialGrant,
    VirtualCredentialRegistry,
    VirtualEgressRunnerRequest,
)
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.egress.e2b_adapter import E2BEgressAdapter
from cayu.egress.microsandbox_adapter import MicrosandboxEgressAdapter
from cayu.runners import ExecCommand, ExecResult, Runner
from cayu.vaults import SecretRef, StaticVault

REAL_SECRET = "sk_test_conformance_real_secret"


def _broker_and_grant() -> tuple[
    TransparentEgressBroker,
    VirtualCredentialRegistry,
    VirtualCredentialGrant,
]:
    registry = VirtualCredentialRegistry()
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=StaticVault({"stripe": REAL_SECRET}),
        policies={
            "stripe": HttpEgressPolicy(
                name="stripe",
                allowed_hosts=["api.stripe.com"],
                allowed_endpoints=[("POST", "/v1/customers")],
            )
        },
    )
    grant = registry.mint(
        session_id="session-a",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
        policy_name="stripe",
    )
    return broker, registry, grant


def _runner_request(*, runner_kind: str, binding: EgressBinding) -> VirtualEgressRunnerRequest:
    return VirtualEgressRunnerRequest(
        name="conformance-runner",
        runner_kind=runner_kind,
        image="conformance-image",
        binding=binding,
        env_overlay=dict(binding.env),
        ca_cert_host_path="/tmp/conformance-ca.pem",
        guest_ca_path="/etc/cayu/ca.pem",
        setup_commands=(),
        egress_destinations=("api.stripe.com",),
    )


def test_registration_covers_every_enforcing_builtin_adapter() -> None:
    assert {registration.adapter_type for registration in EGRESS_CONFORMANCE_REGISTRATIONS} == {
        DockerEgressAdapter,
        E2BEgressAdapter,
        MicrosandboxEgressAdapter,
    }
    assert {registration.runner_kind for registration in EGRESS_CONFORMANCE_REGISTRATIONS} == {
        "docker",
        "e2b",
        "microsandbox",
    }


@pytest.mark.parametrize(
    "registration",
    EGRESS_CONFORMANCE_REGISTRATIONS,
    ids=lambda registration: registration.name,
)
def test_registration_factory_and_runner_pairing_fail_closed(registration) -> None:  # type: ignore[no-untyped-def]
    adapter = registration.create_adapter()
    assert type(adapter) is registration.adapter_type
    assert adapter.runner_kind == registration.runner_kind
    binding = EgressBinding(
        network="conformance-network",
        proxy_url="http://203.0.113.10:8443",
        guest_ca_path="/etc/cayu/ca.pem",
    )
    request = _runner_request(runner_kind="different-runner", binding=binding)

    with pytest.raises(UnsupportedEgressError, match="runner kind"):
        asyncio.run(adapter.create_runner(request))


def test_unregistered_adapter_resolution_refuses_raw_secret_downgrade() -> None:
    adapter = EgressAdapterRegistry().resolve("unregistered")
    assert isinstance(adapter, UnsupportedEgressAdapter)
    broker, _registry, grant = _broker_and_grant()

    with pytest.raises(UnsupportedEgressError, match="refuse to downgrade") as exc_info:
        asyncio.run(adapter.prepare(session_id="session-a", grants=[grant], broker=broker))
    assert REAL_SECRET not in str(exc_info.value)
    assert grant.presented_value not in str(exc_info.value)


@pytest.mark.parametrize(
    "registration",
    EGRESS_CONFORMANCE_REGISTRATIONS,
    ids=lambda registration: registration.name,
)
def test_registration_rejects_cross_session_grants_before_allocating_resources(
    registration,
) -> None:  # type: ignore[no-untyped-def]
    adapter = registration.create_adapter()
    broker, _registry, grant = _broker_and_grant()

    with pytest.raises(UnsupportedEgressError, match="requested session"):
        asyncio.run(adapter.prepare(session_id="different-session", grants=[grant], broker=broker))


@pytest.mark.parametrize(
    "registration",
    EGRESS_CONFORMANCE_REGISTRATIONS,
    ids=lambda registration: registration.name,
)
def test_registered_builtin_prepare_lifecycle_is_cancellation_safe_and_revoke_first(
    registration,
) -> None:  # type: ignore[no-untyped-def]
    async def run() -> tuple[bool, tuple[bool, ...]]:
        fixture = registration.create_deterministic_fixture()
        broker, registry, grant = _broker_and_grant()
        fixture.release_probe.arm(registry, grant.presented_value)
        binding = await fixture.adapter.prepare(
            session_id="session-a",
            grants=[grant],
            broker=broker,
        )
        assert binding.runner_kind == registration.runner_kind
        assert REAL_SECRET not in repr(binding)
        lease = registry.acquire(grant.presented_value)

        close_task = asyncio.create_task(binding.close())
        for _ in range(20):
            try:
                registry.lookup(grant.presented_value)
            except VirtualCredentialError:
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("Registered adapter did not revoke before teardown wait.")
        assert fixture.release_probe.observations == []

        close_task.cancel()
        await asyncio.sleep(0)
        assert close_task.done() is False
        lease.close()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        await binding.close()
        return binding._closed, tuple(fixture.release_probe.observations)

    closed, release_observations = asyncio.run(run())

    assert closed is True
    assert release_observations
    assert all(release_observations)


def test_live_prerequisites_keep_paid_lanes_explicit() -> None:
    registrations = {item.name: item for item in EGRESS_CONFORMANCE_REGISTRATIONS}
    assert registrations["docker"].live_prerequisites.required_env == ()
    assert registrations["docker"].live_prerequisites.required_env_values == ()
    assert ("CAYU_RUN_E2B_EGRESS_E2E", "1") in (
        registrations["e2b"].live_prerequisites.required_env_values
    )
    assert "E2B_API_KEY" in registrations["e2b"].live_prerequisites.required_env
    assert ("CAYU_RUN_MICROSANDBOX_EGRESS_E2E", "1") in (
        registrations["microsandbox"].live_prerequisites.required_env_values
    )
    for registration in registrations.values():
        assert Path(registration.live_proof_source).is_file()
        assert registration.teardown_timeout_s <= 30
        assert registration.bounded_destinations == (
            "api.stripe.com:443",
            "1.1.1.1:443",
            "169.254.169.254:80",
        )


def test_evidence_is_typed_bounded_and_secret_free_by_construction() -> None:
    evidence = EgressScenarioEvidence(
        adapter="docker",
        scenario="guest-network-bypass-denial",
        status="verified",
        proof_source="live",
        observations=("public-ip-denied", "metadata-service-denied"),
        cleanup_outcome="complete-and-grant-revoked",
        duration_ms=123,
        reason="contract-satisfied",
    )
    assert REAL_SECRET not in repr(evidence)
    assert "headers" not in repr(evidence)
    assert "response_body" not in repr(evidence)

    with pytest.raises(ValueError, match="safe 1-80 character token"):
        EgressScenarioEvidence(
            adapter="docker",
            scenario="x" * 81,
            status="verified",
            proof_source="live",
            observations=(),
            cleanup_outcome="complete",
            duration_ms=123,
        )

    with pytest.raises(ValueError, match="unsupported observation"):
        EgressScenarioEvidence(
            adapter="docker",
            scenario="guest-network-bypass-denial",
            status="verified",
            proof_source="live",
            observations=(REAL_SECRET,),  # type: ignore[arg-type]
            cleanup_outcome="complete",
            duration_ms=123,
        )


def test_nightly_evidence_emits_one_bounded_typed_record(
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence = (
        EgressScenarioEvidence(
            adapter="docker",
            scenario="guest-network-bypass-denial",
            status="verified",
            proof_source="live",
            observations=("public-ip-denied", "metadata-service-denied"),
            cleanup_outcome="complete-and-grant-revoked",
            duration_ms=123,
            reason="contract-satisfied",
        ),
        EgressScenarioEvidence(
            adapter="docker",
            scenario="cleanup-failure",
            status="failed",
            proof_source="live",
            observations=(),
            cleanup_outcome="unknown",
            duration_ms=124,
            reason="check-failed",
        ),
        EgressScenarioEvidence(
            adapter="docker",
            scenario="prerequisite-check",
            status="skipped",
            proof_source="nightly",
            observations=(),
            cleanup_outcome="not-applicable",
            duration_ms=0,
            reason="prerequisites-unavailable",
        ),
    )

    emit_egress_nightly_evidence(evidence)

    output = capsys.readouterr().out.splitlines()
    assert len(output) == 1
    prefix = "CAYU_NIGHTLY_EVIDENCE="
    assert output[0].startswith(prefix)
    assert json.loads(output[0].removeprefix(prefix)) == {
        "records": [
            {
                "adapter": "docker",
                "cleanup_outcome": "complete-and-grant-revoked",
                "duration_ms": 123,
                "observations": ["public-ip-denied", "metadata-service-denied"],
                "proof_source": "live",
                "reason": "contract-satisfied",
                "scenario": "guest-network-bypass-denial",
                "status": "verified",
            },
            {
                "adapter": "docker",
                "cleanup_outcome": "unknown",
                "duration_ms": 124,
                "observations": [],
                "proof_source": "live",
                "reason": "check-failed",
                "scenario": "cleanup-failure",
                "status": "failed",
            },
            {
                "adapter": "docker",
                "cleanup_outcome": "not-applicable",
                "duration_ms": 0,
                "observations": [],
                "proof_source": "nightly",
                "reason": "prerequisites-unavailable",
                "scenario": "prerequisite-check",
                "status": "skipped",
            },
        ],
        "schema": "cayu.egress_conformance.v1",
    }


def test_nightly_failure_boundary_emits_typed_failure_before_reraising(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        pytest.raises(RuntimeError, match="live probe failed"),
        egress_nightly_failure_boundary("docker"),
    ):
        raise RuntimeError("live probe failed")

    output = capsys.readouterr().out.splitlines()
    assert len(output) == 1
    payload = json.loads(output[0].removeprefix("CAYU_NIGHTLY_EVIDENCE="))
    assert payload["records"] == [
        {
            "adapter": "docker",
            "cleanup_outcome": "unknown",
            "duration_ms": 0,
            "observations": [],
            "proof_source": "live",
            "reason": "check-failed",
            "scenario": "live-security-conformance",
            "status": "failed",
        }
    ]


def test_nightly_evidence_rejects_unbounded_or_mixed_adapter_records() -> None:
    evidence = EgressScenarioEvidence(
        adapter="docker",
        scenario="guest-network-bypass-denial",
        status="verified",
        proof_source="live",
        observations=(),
        cleanup_outcome="complete",
        duration_ms=123,
    )

    with pytest.raises(ValueError, match="between one and eight"):
        emit_egress_nightly_evidence(())
    with pytest.raises(ValueError, match="between one and eight"):
        emit_egress_nightly_evidence((evidence,) * 9)
    with pytest.raises(ValueError, match="one adapter"):
        emit_egress_nightly_evidence(
            (
                evidence,
                EgressScenarioEvidence(
                    adapter="e2b",
                    scenario="guest-network-bypass-denial",
                    status="verified",
                    proof_source="live",
                    observations=(),
                    cleanup_outcome="complete",
                    duration_ms=123,
                ),
            )
        )


@pytest.mark.parametrize(
    ("field_name", "message"),
    (
        ("status", "unsupported status"),
        ("proof_source", "unsupported proof source"),
        ("reason", "unsupported reason"),
    ),
)
def test_evidence_rejects_secret_bearing_values_in_every_typed_text_field(
    field_name: str,
    message: str,
) -> None:
    values: dict[str, Any] = {
        "adapter": "docker",
        "scenario": "guest-network-bypass-denial",
        "status": "verified",
        "proof_source": "live",
        "observations": (),
        "cleanup_outcome": "complete",
        "duration_ms": 123,
        "reason": None,
    }
    values[field_name] = REAL_SECRET

    with pytest.raises(ValueError, match=message):
        EgressScenarioEvidence(**values)  # type: ignore[arg-type]


def test_evidence_copies_observations_and_rejects_non_integer_duration() -> None:
    observations = ["public-ip-denied"]
    evidence = EgressScenarioEvidence(
        adapter="docker",
        scenario="guest-network-bypass-denial",
        status="verified",
        proof_source="live",
        observations=observations,  # type: ignore[arg-type]
        cleanup_outcome="complete",
        duration_ms=123,
    )
    observations.append(REAL_SECRET)

    assert evidence.observations == ("public-ip-denied",)
    assert REAL_SECRET not in repr(evidence)

    with pytest.raises(TypeError, match="duration_ms must be an integer"):
        EgressScenarioEvidence(
            adapter="docker",
            scenario="guest-network-bypass-denial",
            status="verified",
            proof_source="live",
            observations=(),
            cleanup_outcome="complete",
            duration_ms=float("nan"),  # type: ignore[arg-type]
        )


class _ObservedRunner(Runner):
    isolation = "conformance"

    def __init__(self, binding: EgressBinding) -> None:
        self.binding = binding

    async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        raise NotImplementedError

    async def close(self) -> None:
        self._closed = True


Defect = Literal[
    "none",
    "raw-secret-downgrade",
    "false-cleanup-completion",
    "mismatched-runner",
    "skipped-revocation",
]


@dataclass
class _ScriptedAdapter(SandboxEgressAdapter):
    defect: Defect
    registry: VirtualCredentialRegistry
    grant: VirtualCredentialGrant
    runner_kind: str = "scripted"
    released: bool = False

    async def prepare(
        self,
        *,
        session_id: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
    ) -> EgressBinding:
        assert session_id == self.grant.session_id
        assert tuple(grants) == (self.grant,)

        async def teardown() -> None:
            if self.defect != "skipped-revocation":
                self.registry.revoke(self.grant.presented_value)
            if self.defect == "false-cleanup-completion":
                return
            self.released = True

        env_value = (
            REAL_SECRET if self.defect == "raw-secret-downgrade" else self.grant.presented_value
        )
        return EgressBinding(
            env={"STRIPE_SECRET_KEY": env_value},
            runner_kind=self.runner_kind,
            guest_ca_path="/etc/cayu/ca.pem",
            teardown=teardown,
        )

    async def create_runner(self, request: VirtualEgressRunnerRequest) -> Runner:
        binding = EgressBinding() if self.defect == "mismatched-runner" else request.binding
        return _ObservedRunner(binding)


async def _verify_scripted_adapter(
    adapter: _ScriptedAdapter,
    broker: TransparentEgressBroker,
) -> None:
    registry = adapter.registry
    grant = adapter.grant
    binding = await adapter.prepare(session_id="session-a", grants=[grant], broker=broker)
    assert REAL_SECRET not in repr(binding.env)
    runner = await adapter.create_runner(_runner_request(runner_kind="scripted", binding=binding))
    assert isinstance(runner, _ObservedRunner)
    assert runner.binding is binding
    assert grant.session_id == "session-a"
    await binding.close()
    assert adapter.released is True
    try:
        registry.lookup(grant.presented_value)
    except VirtualCredentialError:
        pass
    else:
        raise AssertionError("Grant remained active after resource release.")


def test_scripted_reference_adapter_satisfies_orchestration_contract() -> None:
    broker, registry, grant = _broker_and_grant()
    asyncio.run(_verify_scripted_adapter(_ScriptedAdapter("none", registry, grant), broker))


@pytest.mark.parametrize(
    "defect",
    (
        "raw-secret-downgrade",
        "false-cleanup-completion",
        "mismatched-runner",
        "skipped-revocation",
    ),
)
def test_seeded_broken_adapter_is_rejected(defect: Defect) -> None:
    broker, registry, grant = _broker_and_grant()
    adapter = _ScriptedAdapter(defect, registry, grant)
    with pytest.raises(AssertionError):
        asyncio.run(_verify_scripted_adapter(adapter, broker))
