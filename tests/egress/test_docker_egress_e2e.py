"""Adversarial end-to-end tests for Docker virtual egress.

These prove *non-possession*: the sandbox never receives the real secret, cannot
reach the provider directly, and the credential dies with the session. They spin
real containers, so they are gated on a responsive Docker daemon.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess

import pytest
from tests.egress_conformance import (
    EgressScenarioEvidence,
    egress_nightly_failure_boundary,
    emit_egress_nightly_evidence,
    registration_for,
)
from tests.egress_e2e_support import CapturingEgressAdapter, drive_adversarial_egress_contract

from cayu.egress import (
    CapturedRequest,
    CapturedResponse,
    HttpEgressPolicy,
)
from cayu.runners.base import ExecCommand
from cayu.vaults import SecretRef, StaticVault

pytest.importorskip("cryptography")

from cayu.egress.docker_adapter import DockerEgressAdapter

REAL_SECRET = "sk_test_51E2ERealSecretNeverInSandbox"
SANDBOX_IMAGE = "python:3.12-slim"


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


_DOCKER_AVAILABLE = _docker_available()
if os.environ.get("CAYU_REQUIRE_DOCKER_EGRESS") == "1" and not _DOCKER_AVAILABLE:
    raise RuntimeError("CAYU_REQUIRE_DOCKER_EGRESS=1 but the Docker daemon is unavailable.")

pytestmark = pytest.mark.skipif(
    not _DOCKER_AVAILABLE,
    reason="Docker daemon not available for virtual-egress E2E.",
)


class _FakeStripe:
    """Stands in for api.stripe.com so the E2E needs no real Stripe key."""

    def __init__(self) -> None:
        self.upstream_authorization: str | None = None

    async def send(self, request: CapturedRequest) -> CapturedResponse:
        self.upstream_authorization = request.headers.get("Authorization")
        return CapturedResponse(
            status_code=200,
            headers={"Request-Id": "req_e2e"},
            body=b'{"id":"cus_fake123","object":"customer"}',
        )


def _stripe_example_policy() -> HttpEgressPolicy:
    return HttpEgressPolicy(
        name="stripe-example",
        allowed_hosts=["api.stripe.com"],
        allowed_endpoints=[("POST", "/v1/customers")],
    )


async def _drive() -> tuple[EgressScenarioEvidence, ...]:
    return await drive_adversarial_egress_contract(
        registration=registration_for("docker"),
        adapter=CapturingEgressAdapter(DockerEgressAdapter(loop=asyncio.get_running_loop())),
        real_secret=REAL_SECRET,
        image=SANDBOX_IMAGE,
        search_roots=("/workspace", "/tmp", "/etc/cayu", "/root"),
        response_id="cus_fake123",
    )


@pytest.fixture(scope="module")
def e2e_results() -> tuple[EgressScenarioEvidence, ...]:
    with egress_nightly_failure_boundary("docker"):
        return asyncio.run(_drive())


def test_shared_real_boundary_security_contract(
    e2e_results: tuple[EgressScenarioEvidence, ...],
) -> None:
    with egress_nightly_failure_boundary("docker"):
        assert {item.scenario for item in e2e_results} == {
            "brokered-provider-and-session-ca",
            "guest-network-bypass-denial",
            "guest-secret-non-possession",
        }
        assert all(item.adapter == "docker" for item in e2e_results)
        assert all(item.status == "verified" for item in e2e_results)
        assert REAL_SECRET not in repr(e2e_results)
        emit_egress_nightly_evidence(e2e_results)


# --- Full runtime wiring: the VirtualEgressEnvironmentFactory lifecycle. ---


async def _drive_factory() -> dict[str, object]:
    from cayu.core.events import Event, EventType
    from cayu.environments import EnvironmentFactoryRequest
    from cayu.runtime.egress import VirtualCredentialSpec, VirtualEgressEnvironmentFactory

    events: list[Event] = []
    authorized_seen = asyncio.Event()

    async def emitter(event: Event) -> Event:
        events.append(event)
        if event.type == EventType.EGRESS_REQUEST_AUTHORIZED:
            authorized_seen.set()
        return event

    upstream = _FakeStripe()
    factory = VirtualEgressEnvironmentFactory(
        resolver=StaticVault({"stripe_test_key": REAL_SECRET}),
        policies={"stripe-example": _stripe_example_policy()},
        credentials=[
            VirtualCredentialSpec(
                env_name="STRIPE_SECRET_KEY",
                secret=SecretRef(name="stripe_test_key"),
                destination="api.stripe.com",
                policy_name="stripe-example",
            )
        ],
        image=SANDBOX_IMAGE,
        event_emitter=emitter,
        upstream=upstream,
    )
    request = EnvironmentFactoryRequest(
        session_id="e2e-factory", agent_name="agent", environment_name="egress-env"
    )
    result = await factory.create(request)
    runner = result.environment.runner
    assert runner is not None

    call_script = (
        "import os, urllib.request as u\n"
        "req = u.Request('https://api.stripe.com/v1/customers', data=b'email=a@b.co',\n"
        "    headers={'Authorization': 'Bearer ' + os.environ['STRIPE_SECRET_KEY']})\n"
        "print(u.urlopen(req, timeout=25).read().decode())\n"
    )
    try:
        call = await runner.exec(ExecCommand.process("python3", "-c", call_script), timeout_s=60)
        await asyncio.wait_for(authorized_seen.wait(), timeout=2)
    finally:
        bound = await result.environment.binding.bind(None, runner, session_id="e2e-factory")
        await result.environment.binding.finalize(bound, outcome="completed")

    return {
        "call_stdout": call.stdout,
        "upstream_authorization": upstream.upstream_authorization,
        "event_types": [str(e.type) for e in events],
        "event_payloads": " ".join(str(e.payload) for e in events),
        "authorized": EventType.EGRESS_REQUEST_AUTHORIZED,
        "minted": EventType.EGRESS_GRANT_MINTED,
        "revoked": EventType.EGRESS_GRANT_REVOKED,
    }


@pytest.fixture(scope="module")
def factory_results() -> dict[str, object]:
    return asyncio.run(_drive_factory())


def test_factory_allowed_call_succeeds_with_swap(factory_results: dict[str, object]) -> None:
    assert "cus_fake123" in str(factory_results["call_stdout"])
    assert REAL_SECRET not in str(factory_results["call_stdout"])
    assert factory_results["upstream_authorization"] == f"Bearer {REAL_SECRET}"


def test_factory_emits_audit_events_without_secret(factory_results: dict[str, object]) -> None:
    types = factory_results["event_types"]
    assert str(factory_results["minted"]) in types
    assert str(factory_results["authorized"]) in types
    assert str(factory_results["revoked"]) in types
    assert REAL_SECRET not in str(factory_results["event_payloads"])
