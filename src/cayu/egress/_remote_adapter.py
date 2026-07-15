from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

from cayu.egress.adapter import (
    DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
    EgressBinding,
    VirtualEgressRunnerRequest,
    _await_bounded_cleanup_task,
    validate_grant_scope,
)
from cayu.egress.broker import TransparentEgressBroker
from cayu.egress.errors import UnsupportedEgressError
from cayu.egress.grants import VirtualCredentialGrant
from cayu.egress.proxy_exposure import ExposedProxy, HttpProxyEndpoint, ProxyExposure
from cayu.egress.proxy_server import TransparentEgressProxyServer
from cayu.runners.base import ExecCommand, Runner

GUEST_CA_PATH = "/etc/cayu/ca.pem"

ProxyServerFactory = Callable[..., TransparentEgressProxyServer]


async def prepare_exposed_proxy_binding(
    *,
    runner_kind: str,
    session_id: str,
    broker: TransparentEgressBroker,
    grants: Sequence[VirtualCredentialGrant],
    exposure: ProxyExposure,
    bind_host: str,
    loop: asyncio.AbstractEventLoop | None,
    proxy_server_factory: ProxyServerFactory,
) -> EgressBinding:
    validate_grant_scope(session_id=session_id, grants=grants)
    resolved_loop = loop or asyncio.get_running_loop()
    server = proxy_server_factory(broker, loop=resolved_loop, host=bind_host)
    exposed: ExposedProxy | None = None

    async def cleanup() -> None:
        # Grant revocation is the security transition and must complete before
        # the proxy exposure or listener is released.
        await broker.registry.revoke_values_and_wait(
            tuple(grant.presented_value for grant in grants)
        )
        errors: list[str] = []
        if exposed is not None:
            try:
                await exposed.close()
            except Exception as exc:
                errors.append(f"proxy exposure: {type(exc).__name__}")
        try:
            await server.close()
        except Exception as exc:
            errors.append(f"proxy listener: {type(exc).__name__}")
        if errors:
            raise RuntimeError(f"{runner_kind} egress teardown incomplete: {'; '.join(errors)}")

    try:
        proxy_port = await server.start()
        exposed = await exposure.expose(local_host=bind_host, local_port=proxy_port)
        try:
            endpoint = HttpProxyEndpoint.parse(exposed.proxy_url)
        except ValueError as exc:
            raise UnsupportedEgressError(
                f"{runner_kind} proxy exposure returned an invalid HTTP proxy URL: {exc}"
            ) from exc
        proxy_url = endpoint.url
    except BaseException as original:
        cleanup_task = asyncio.create_task(cleanup())
        try:
            await _await_bounded_cleanup_task(
                cleanup_task,
                timeout_s=DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
                timeout_message=f"{runner_kind} egress prepare rollback timed out.",
            )
        except BaseException as cleanup_error:
            original.add_note(
                f"{runner_kind} egress prepare rollback incomplete: {type(cleanup_error).__name__}."
            )
        raise

    env = {
        "HTTPS_PROXY": proxy_url,
        "https_proxy": proxy_url,
        "SSL_CERT_FILE": GUEST_CA_PATH,
        "REQUESTS_CA_BUNDLE": GUEST_CA_PATH,
        "CURL_CA_BUNDLE": GUEST_CA_PATH,
        "NODE_EXTRA_CA_CERTS": GUEST_CA_PATH,
    }

    async def teardown() -> None:
        await cleanup()

    return EgressBinding(
        env=env,
        ca_cert_pem=server.authority.ca_cert_pem(),
        runner_kind=runner_kind,
        guest_ca_path=GUEST_CA_PATH,
        proxy_url=proxy_url,
        proxy_port=proxy_port,
        metadata={
            "runner_kind": runner_kind,
            "proxy_url": proxy_url,
            "proxy_bind_host": bind_host,
            "proxy_port": proxy_port,
            "guest_ca_path": GUEST_CA_PATH,
        },
        teardown=teardown,
    )


async def run_enforcement_preflight(
    runner: Runner,
    request: VirtualEgressRunnerRequest,
    *,
    timeout_s: int,
    probe_metadata: bool = True,
) -> None:
    if not request.egress_destinations:
        raise UnsupportedEgressError(
            f"Runner {request.runner_kind!r} has no provider destination to preflight."
        )
    endpoint = request.binding.proxy_endpoint
    if endpoint is None:
        raise UnsupportedEgressError(
            f"Runner {request.runner_kind!r} egress binding did not provide proxy_url."
        )
    # Proxy reachability and CA trust are session-wide, so one grant destination
    # samples the positive TLS path; raw-IP probes prove runtime-wide denial.
    destination = request.egress_destinations[0]
    script = _preflight_script(
        proxy_host=endpoint.host,
        proxy_port=endpoint.port,
        destination=destination,
        guest_ca_path=request.guest_ca_path,
        probe_metadata=probe_metadata,
    )
    result = await runner.exec(
        ExecCommand.process("python3", "-c", script),
        timeout_s=timeout_s,
    )
    if result.exit_code != 0 or result.timed_out:
        raise UnsupportedEgressError(
            f"Runner {request.runner_kind!r} failed virtual-egress preflight "
            f"(exit_code={result.exit_code}, timed_out={result.timed_out})."
        )


async def run_setup_commands(runner: Runner, request: VirtualEgressRunnerRequest) -> None:
    for command in request.setup_commands:
        result = await runner.exec(ExecCommand.bash(command), timeout_s=300)
        if result.exit_code != 0 or result.timed_out:
            raise RuntimeError(
                f"{request.runner_kind} setup command failed "
                f"(exit_code={result.exit_code}, timed_out={result.timed_out})."
            )


def _preflight_script(
    *,
    proxy_host: str,
    proxy_port: int,
    destination: str,
    guest_ca_path: str,
    probe_metadata: bool,
) -> str:
    return f"""
import socket
import ssl

proxy_host = {proxy_host!r}
proxy_port = {proxy_port}
destination = {destination!r}

sock = socket.create_connection((proxy_host, proxy_port), timeout=5)
print("preflight: proxy TCP connected", flush=True)
sock.sendall(f"CONNECT {{destination}}:443 HTTP/1.1\\r\\nHost: {{destination}}:443\\r\\n\\r\\n".encode())
head = b""
while b"\\r\\n\\r\\n" not in head and len(head) < 8192:
    chunk = sock.recv(1024)
    if not chunk:
        break
    head += chunk
if not head.startswith(b"HTTP/1.1 200"):
    raise RuntimeError(f"proxy CONNECT failed: {{head[:200]!r}}")
print("preflight: proxy CONNECT accepted", flush=True)
context = ssl.create_default_context(cafile={guest_ca_path!r})
tls = context.wrap_socket(sock, server_hostname=destination)
tls.close()
print("preflight: proxy TLS completed", flush=True)

def tls_reachable(host, port):
    try:
        direct = socket.create_connection((host, port), timeout=2)
        direct_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        direct_context.check_hostname = False
        direct_context.verify_mode = ssl.CERT_NONE
        direct_tls = direct_context.wrap_socket(direct)
        direct_tls.close()
    except OSError:
        return False
    return True

def http_request_succeeds(host, port, request):
    try:
        direct = socket.create_connection((host, port), timeout=2)
        direct.sendall(request)
        head = b""
        while b"\\r\\n\\r\\n" not in head and len(head) < 8192:
            chunk = direct.recv(1024)
            if not chunk:
                break
            head += chunk
        direct.close()
    except OSError:
        return False
    try:
        status = int(head.split(b" ", 2)[1])
    except (IndexError, ValueError):
        return False
    print(f"preflight: metadata HTTP status {{status}}", flush=True)
    return 200 <= status < 300

def metadata_reachable(host, port):
    get_request = b"GET /latest/meta-data/ HTTP/1.1\\r\\nHost: 169.254.169.254\\r\\nConnection: close\\r\\n\\r\\n"
    token_request = b"PUT /latest/api/token HTTP/1.1\\r\\nHost: 169.254.169.254\\r\\nX-aws-ec2-metadata-token-ttl-seconds: 60\\r\\nConnection: close\\r\\n\\r\\n"
    return http_request_succeeds(host, port, get_request) or http_request_succeeds(
        host, port, token_request
    )

probes = [("1.1.1.1", 443, tls_reachable)]
if {probe_metadata!r}:
    probes.append(("169.254.169.254", 80, metadata_reachable))
for host, port, probe in probes:
    print(f"preflight: probing direct {{host}}:{{port}}", flush=True)
    if probe(host, port):
        raise RuntimeError(f"direct egress unexpectedly reached {{host}}:{{port}}")
print("preflight: direct egress blocked", flush=True)
""".strip()


DEFAULT_PROXY_SERVER_FACTORY: ProxyServerFactory = TransparentEgressProxyServer
