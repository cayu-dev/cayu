from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from cayu.credentials import CredentialMode
from cayu.egress.adapter import (
    DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
    EgressBinding,
    SandboxEgressAdapter,
    VirtualEgressRunnerRequest,
    _await_bounded_cleanup_task,
    validate_grant_scope,
)
from cayu.egress.broker import TransparentEgressBroker
from cayu.egress.errors import UnsupportedEgressError
from cayu.egress.grants import VirtualCredentialGrant
from cayu.egress.proxy_server import TransparentEgressProxyServer
from cayu.runners.base import Runner
from cayu.runners.docker import DockerRunner

_logger = logging.getLogger(__name__)

#: Where the per-session CA is mounted inside the sandbox and trusted from.
GUEST_CA_PATH = "/etc/cayu/ca.pem"
_SIDECAR_LISTEN_PORT = 8080
_DEFAULT_SIDECAR_IMAGE = "alpine/socat"
_SESSION_LABEL = "cayu.egress.session"
_SIDECAR_AUTH_PATH = "/run/cayu/broker.auth"
_SIDECAR_CONNECTOR_PATH = "/run/cayu/connect-broker"
_SIDECAR_READY_SCRIPT = (
    "attempts=0; "
    'while [ "$(cat /proc/1/comm 2>/dev/null)" != socat ] && '
    '[ "$attempts" -lt 100 ]; do '
    "attempts=$((attempts + 1)); sleep 0.05; "
    "done; "
    'test "$(cat /proc/1/comm 2>/dev/null)" = socat'
)

# A docker executor returns (exit_code, stderr) so orchestration can be faked in
# tests without a real Docker daemon.
DockerExec = Callable[[Sequence[str]], Awaitable[tuple[int, str]]]
# A docker STDOUT runner used only for host-interface discovery.
DockerRun = Callable[[Sequence[str]], Awaitable[tuple[int, str]]]


@dataclass(frozen=True)
class _SidecarTransportAuthorization:
    directory: str
    auth_path: str
    connector_path: str
    token: bytes = field(repr=False)

    def close(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)


def _create_sidecar_transport_authorization() -> _SidecarTransportAuthorization:
    directory = tempfile.mkdtemp(prefix="cayu-egress-sidecar-")
    auth_path = os.path.join(directory, "broker.auth")
    connector_path = os.path.join(directory, "connect-broker")
    token = secrets.token_urlsafe(32).encode("ascii")
    try:
        _write_private(auth_path, b"cayu:" + token, mode=0o600)
        _write_private(
            connector_path,
            b"#!/bin/sh\n"
            b"set -eu\n"
            b'if [ "${1:-}" = "listen" ]; then\n'
            b"  attempts=0\n"
            b"  bind_ip=\n"
            b'  while [ "$attempts" -lt 100 ]; do\n'
            b'    default_if="$(ip route show default | '
            b"awk 'NR == 1 { print $5 }')\"\n"
            b'    bind_ip="$(ip -o -4 addr show | '
            b'awk -v default_if="$default_if" '
            b'\'$2 != "lo" && $2 != default_if { split($4, address, "/"); '
            b"print address[1]; exit }')\"\n"
            b'    [ -n "$bind_ip" ] && break\n'
            b"    attempts=$((attempts + 1))\n"
            b"    sleep 0.05\n"
            b"  done\n"
            b'  [ -n "$bind_ip" ] || exit 70\n'
            b'  exec socat "TCP-LISTEN:8080,bind=${bind_ip},fork,reuseaddr" '
            b'"PROXY:host.docker.internal:cayu-transport.invalid:443,'
            b'proxyport=${CAYU_BROKER_PORT},proxyauthfile=/run/cayu/broker.auth"\n'
            b"fi\n"
            b"exit 64\n",
            mode=0o700,
        )
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return _SidecarTransportAuthorization(
        directory=directory,
        auth_path=auth_path,
        connector_path=connector_path,
        token=token,
    )


def _write_private(path: str, data: bytes, *, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


async def _default_docker_exec(argv: Sequence[str]) -> tuple[int, str]:
    docker = shutil.which("docker")
    if not docker:
        raise UnsupportedEgressError("docker CLI not found; cannot enforce virtual egress.")
    process = await asyncio.create_subprocess_exec(
        docker,
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    return process.returncode or 0, stderr.decode("utf-8", "replace")


async def _run_docker_stdout(argv: Sequence[str]) -> tuple[int, str]:
    docker = shutil.which("docker")
    if not docker:
        raise UnsupportedEgressError("docker CLI not found; cannot enforce virtual egress.")
    process = await asyncio.create_subprocess_exec(
        docker,
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    return process.returncode or 0, stdout.decode("utf-8", "replace")


async def resolve_proxy_bind_host(run: DockerRun = _run_docker_stdout) -> str:
    """Pick the narrowest host interface the sidecar can still reach.

    - Docker Desktop routes ``host.docker.internal`` to the host loopback, so
      ``127.0.0.1`` is both reachable and tightest.
    - Native Linux: bind to the default-bridge gateway (e.g. ``172.17.0.1``),
      reachable from containers via host-gateway but not on the host's LAN.
    - If neither can be determined, fall back to ``0.0.0.0`` and warn loudly.
    """
    with contextlib.suppress(Exception):
        code, out = await run(["info", "--format", "{{.OperatingSystem}}"])
        if code == 0 and "docker desktop" in out.strip().lower():
            return "127.0.0.1"
    with contextlib.suppress(Exception):
        code, out = await run(
            [
                "network",
                "inspect",
                "bridge",
                "--format",
                "{{range .IPAM.Config}}{{.Gateway}} {{end}}",
            ]
        )
        if code == 0:
            gateways = out.strip().split()
            if gateways and gateways[0]:
                return gateways[0]
    _logger.warning(
        "Cayu egress proxy is binding to 0.0.0.0 (ALL host interfaces) because the "
        "Docker host interface could not be determined — this is LAN-reachable. The "
        "listener is credential-gated (no secret leaks), but pass an explicit "
        "proxy_host=... to DockerEgressAdapter to avoid exposure."
    )
    return "0.0.0.0"


class DockerEgressAdapter(SandboxEgressAdapter):
    """Enforced egress for the Docker runner.

    Fail-closed by construction: the sandbox joins an ``--internal`` Docker
    network with no route to the internet, so the *only* reachable egress is a
    dual-homed sidecar that forwards to the in-process broker. Direct provider
    calls cannot leave the sandbox. Returns the network/env the runner must use;
    ``teardown`` removes the sidecar and network and revokes the grants.
    """

    runner_kind = "docker"

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        docker_exec: DockerExec | None = None,
        sidecar_image: str = _DEFAULT_SIDECAR_IMAGE,
        proxy_host: str | None = None,
        proxy_bind_host_resolver: Callable[[], Awaitable[str]] | None = None,
    ) -> None:
        self._loop = loop
        self._docker_exec = docker_exec or _default_docker_exec
        self._sidecar_image = sidecar_image
        # None => auto-resolve the narrowest reachable interface at prepare time
        # (loopback on Docker Desktop, bridge gateway on Linux). An explicit value
        # is used verbatim. The broker still requires a valid unguessable virtual
        # credential + destination/policy, so the listener is not usable on its own.
        self._proxy_host = proxy_host
        self._proxy_bind_host_resolver = proxy_bind_host_resolver or resolve_proxy_bind_host

    async def prepare(
        self,
        *,
        session_id: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
    ) -> EgressBinding:
        validate_grant_scope(session_id=session_id, grants=grants)
        loop = self._loop or asyncio.get_running_loop()
        bind_host = (
            self._proxy_host
            if self._proxy_host is not None
            else await self._proxy_bind_host_resolver()
        )
        # Resource names use a random token, not the session_id, so distinct
        # sessions can never collide (which would let one teardown remove
        # another live session's network). The session_id rides in a label.
        token = secrets.token_hex(6)
        network = f"cayu-egress-net-{token}"
        sidecar = f"cayu-egress-{token}"
        label = f"{_SESSION_LABEL}={session_id}"
        transport_authorization = _create_sidecar_transport_authorization()
        try:
            server = TransparentEgressProxyServer(
                broker,
                loop=loop,
                host=bind_host,
                transport_auth_token=transport_authorization.token,
            )
        except BaseException:
            transport_authorization.close()
            raise

        try:
            proxy_port = await server.start()
            await self._run(["network", "create", "--internal", "--label", label, network])
            # Sidecar starts on the default bridge (with host-gateway) so it can
            # reach the host broker, then also joins the internal sandbox network.
            await self._run(
                [
                    "run",
                    "-d",
                    "--name",
                    sidecar,
                    "--label",
                    label,
                    "--add-host",
                    "host.docker.internal:host-gateway",
                    "--mount",
                    (
                        f"type=bind,src={transport_authorization.auth_path},"
                        f"dst={_SIDECAR_AUTH_PATH},readonly"
                    ),
                    "--mount",
                    (
                        f"type=bind,src={transport_authorization.connector_path},"
                        f"dst={_SIDECAR_CONNECTOR_PATH},readonly"
                    ),
                    "--env",
                    f"CAYU_BROKER_PORT={proxy_port}",
                    "--entrypoint",
                    _SIDECAR_CONNECTOR_PATH,
                    self._sidecar_image,
                    "listen",
                ]
            )
            await self._run(["network", "connect", network, sidecar])
            await self._run(["exec", sidecar, "sh", "-c", _SIDECAR_READY_SCRIPT])
        except BaseException as original:
            cleanup_task = asyncio.create_task(
                self._teardown(
                    server,
                    network,
                    sidecar,
                    broker,
                    grants,
                    transport_authorization,
                )
            )
            try:
                await _await_bounded_cleanup_task(
                    cleanup_task,
                    timeout_s=DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
                    timeout_message="Docker egress prepare rollback timed out.",
                )
            except BaseException as cleanup_error:
                original.add_note(
                    f"Docker egress prepare rollback incomplete: {type(cleanup_error).__name__}."
                )
            raise

        proxy_url = f"http://{sidecar}:{_SIDECAR_LISTEN_PORT}"
        env = {
            "HTTPS_PROXY": proxy_url,
            "https_proxy": proxy_url,
            "SSL_CERT_FILE": GUEST_CA_PATH,
            "REQUESTS_CA_BUNDLE": GUEST_CA_PATH,
            "CURL_CA_BUNDLE": GUEST_CA_PATH,
            "NODE_EXTRA_CA_CERTS": GUEST_CA_PATH,
        }

        async def teardown() -> None:
            await self._teardown(
                server,
                network,
                sidecar,
                broker,
                grants,
                transport_authorization,
            )

        return EgressBinding(
            env=env,
            ca_cert_pem=server.authority.ca_cert_pem(),
            runner_kind=self.runner_kind,
            network=network,
            sidecar=sidecar,
            guest_ca_path=GUEST_CA_PATH,
            proxy_url=proxy_url,
            proxy_port=proxy_port,
            metadata={
                "runner_kind": self.runner_kind,
                "network": network,
                "sidecar": sidecar,
                "guest_ca_path": GUEST_CA_PATH,
                "proxy_bind_host": bind_host,
                "proxy_port": proxy_port,
            },
            teardown=teardown,
        )

    async def create_runner(self, request: VirtualEgressRunnerRequest) -> Runner:
        if request.runner_kind != self.runner_kind:
            raise UnsupportedEgressError(
                f"Docker egress adapter cannot create runner kind {request.runner_kind!r}."
            )
        network = request.binding.network
        if network is None:
            raise UnsupportedEgressError(
                "Docker egress adapter did not return a network; refusing to start "
                "a virtual-egress sandbox without enforced routing."
            )
        return await DockerRunner.create(
            request.name,
            image=request.image,
            close_action="remove",
            credential_mode=CredentialMode.VIRTUAL_EGRESS,
            network=network,
            env_overlay=dict(request.env_overlay),
            ca_mount=(request.ca_cert_host_path, request.guest_ca_path),
            setup_commands=request.setup_commands,
        )

    async def _run(self, argv: Sequence[str]) -> None:
        exit_code, _stderr = await self._docker_exec(argv)
        if exit_code != 0:
            raise UnsupportedEgressError(
                f"docker {argv[0]} failed while preparing egress (exit_code={exit_code})."
            )

    async def _teardown(
        self,
        server: TransparentEgressProxyServer,
        network: str,
        sidecar: str,
        broker: TransparentEgressBroker,
        grants: Sequence[VirtualCredentialGrant],
        transport_authorization: _SidecarTransportAuthorization,
    ) -> None:
        # Revoke before releasing any resource that enforced the grant boundary.
        # EgressBinding.close keeps failures retryable and never marks an
        # incomplete teardown closed.
        await broker.revoke_authority_and_wait(tuple(grant.presented_value for grant in grants))
        errors: list[str] = []
        for argv in (["rm", "-f", sidecar], ["network", "rm", network]):
            try:
                exit_code, stderr = await self._docker_exec(argv)
                if exit_code != 0 and not _docker_resource_is_absent(stderr):
                    errors.append(f"docker {argv[0]}: exit code {exit_code}")
            except Exception as exc:
                errors.append(f"docker {argv[0]}: {type(exc).__name__}")
        try:
            await server.close()
        except Exception as exc:
            errors.append(f"proxy listener: {type(exc).__name__}")
        try:
            transport_authorization.close()
        except Exception as exc:
            errors.append(f"sidecar transport authorization: {type(exc).__name__}")
        if errors:
            raise RuntimeError(f"Docker egress teardown incomplete: {'; '.join(errors)}")


def _docker_resource_is_absent(stderr: str) -> bool:
    """Treat an already-removed teardown target as successful convergence."""

    normalized = stderr.lower()
    return "no such container" in normalized or "not found" in normalized
