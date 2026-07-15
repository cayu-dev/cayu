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
from cayu.egress.errors import UnsupportedEgressCapabilityError, UnsupportedEgressError
from cayu.egress.grants import VirtualCredentialGrant
from cayu.egress.proxy_exposure import ExposedProxy, HttpProxyEndpoint, ProxyExposure
from cayu.egress.proxy_server import TransparentEgressProxyServer
from cayu.runners.base import ExecCommand, Runner

GUEST_CA_PATH = "/etc/cayu/ca.pem"
_METADATA_ISOLATION_UNSUPPORTED_EXIT_CODE = 42

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
        await broker.revoke_authority_and_wait(tuple(grant.presented_value for grant in grants))
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
        if broker.has_credentialless_destinations and not exposed.credentialless_isolated:
            raise UnsupportedEgressError(
                f"{runner_kind} credentialless egress requires a session-isolated "
                "proxy exposure; shared or public proxy endpoints would be usable "
                "without a virtual credential."
            )
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
    metadata_isolation_reason: str | None = None,
    metadata_isolation_remediation: str | None = None,
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
    if result.timed_out:
        raise UnsupportedEgressError(
            f"Runner {request.runner_kind!r} failed virtual-egress preflight "
            f"(exit_code={result.exit_code}, timed_out=True)."
        )
    if result.exit_code == _METADATA_ISOLATION_UNSUPPORTED_EXIT_CODE:
        raise UnsupportedEgressCapabilityError(
            runner_kind=request.runner_kind,
            capability="metadata_isolation",
            reason=metadata_isolation_reason
            or (
                "guest-initiated requests reached the link-local metadata endpoint; "
                "the runner's deny-by-default network boundary was absent or ineffective"
            ),
            remediation=metadata_isolation_remediation
            or (
                f"restore {request.runner_kind!r} network enforcement so guest commands cannot "
                "reach link-local metadata, then retry the virtual-egress preflight"
            ),
        )
    if result.exit_code != 0:
        raise UnsupportedEgressError(
            f"Runner {request.runner_kind!r} failed virtual-egress preflight "
            f"(exit_code={result.exit_code}, timed_out={result.timed_out})."
        )


async def run_setup_commands(runner: Runner, request: VirtualEgressRunnerRequest) -> None:
    for command in request.setup_commands:
        result = await runner.exec_system(ExecCommand.bash(command), timeout_s=300)
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
import os
from urllib.parse import urlsplit

proxy_host = {proxy_host!r}
proxy_port = {proxy_port}
destination = {destination!r}

proxy_url = (
    os.environ.get("HTTPS_PROXY")
    or os.environ.get("https_proxy")
    or os.environ.get("HTTP_PROXY")
    or os.environ.get("http_proxy")
)
if proxy_url:
    parsed_proxy = urlsplit(proxy_url)
    if parsed_proxy.scheme != "http" or parsed_proxy.hostname is None:
        raise RuntimeError("preflight proxy environment must be an HTTP URL")
    proxy_host = parsed_proxy.hostname
    proxy_port = parsed_proxy.port or 80

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

def tcp_reachable(host, port):
    try:
        direct = socket.create_connection((host, port), timeout=2)
    except OSError:
        return False
    direct.close()
    return True

def http_request_reaches(host, port, request):
    try:
        direct = socket.create_connection((host, port), timeout=2)
    except OSError:
        return False
    head = b""
    try:
        direct.sendall(request)
        while b"\\r\\n\\r\\n" not in head and len(head) < 8192:
            chunk = direct.recv(1024)
            if not chunk:
                break
            head += chunk
    except OSError:
        pass
    finally:
        direct.close()
    try:
        status = int(head.split(b" ", 2)[1])
    except (IndexError, ValueError):
        status = None
    if status is not None:
        print(f"preflight: metadata HTTP status {{status}}", flush=True)
    return True

def metadata_reachable(host, port):
    get_request = b"GET /latest/meta-data/ HTTP/1.1\\r\\nHost: 169.254.169.254\\r\\nConnection: close\\r\\n\\r\\n"
    token_request = b"PUT /latest/api/token HTTP/1.1\\r\\nHost: 169.254.169.254\\r\\nX-aws-ec2-metadata-token-ttl-seconds: 60\\r\\nConnection: close\\r\\n\\r\\n"
    return http_request_reaches(host, port, get_request) or http_request_reaches(
        host, port, token_request
    )

print("preflight: probing direct 1.1.1.1:443", flush=True)
if tcp_reachable("1.1.1.1", 443):
    raise RuntimeError("direct egress unexpectedly reached 1.1.1.1:443")
if {probe_metadata!r}:
    print("preflight: probing metadata 169.254.169.254:80", flush=True)
    if metadata_reachable("169.254.169.254", 80):
        print("preflight: metadata endpoint reachable", flush=True)
        raise SystemExit({_METADATA_ISOLATION_UNSUPPORTED_EXIT_CODE})
print("preflight: direct egress blocked", flush=True)
""".strip()


DEFAULT_PROXY_SERVER_FACTORY: ProxyServerFactory = TransparentEgressProxyServer
