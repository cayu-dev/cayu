from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import shutil
from collections.abc import Awaitable, Callable, Sequence

from cayu.credentials import CredentialMode
from cayu.egress.adapter import (
    EgressBinding,
    SandboxEgressAdapter,
    VirtualEgressRunnerRequest,
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

# A docker executor returns (exit_code, stderr) so orchestration can be faked in
# tests without a real Docker daemon.
DockerExec = Callable[[Sequence[str]], Awaitable[tuple[int, str]]]
# A docker STDOUT runner used only for host-interface discovery.
DockerRun = Callable[[Sequence[str]], Awaitable[tuple[int, str]]]


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
        loop = self._loop or asyncio.get_running_loop()
        bind_host = (
            self._proxy_host
            if self._proxy_host is not None
            else await self._proxy_bind_host_resolver()
        )
        server = TransparentEgressProxyServer(broker, loop=loop, host=bind_host)
        proxy_port = await server.start()

        # Resource names use a random token, not the session_id, so distinct
        # sessions can never collide (which would let one teardown remove
        # another live session's network). The session_id rides in a label.
        token = secrets.token_hex(6)
        network = f"cayu-egress-net-{token}"
        sidecar = f"cayu-egress-{token}"
        label = f"{_SESSION_LABEL}={session_id}"

        try:
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
                    self._sidecar_image,
                    f"TCP-LISTEN:{_SIDECAR_LISTEN_PORT},fork,reuseaddr",
                    f"TCP:host.docker.internal:{proxy_port}",
                ]
            )
            await self._run(["network", "connect", network, sidecar])
        except BaseException:
            await self._teardown(server, network, sidecar, broker, grants)
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
            await self._teardown(server, network, sidecar, broker, grants)

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
        exit_code, stderr = await self._docker_exec(argv)
        if exit_code != 0:
            raise UnsupportedEgressError(
                f"docker {argv[0]} failed while preparing egress: {stderr[:300]}"
            )

    async def _teardown(
        self,
        server: TransparentEgressProxyServer,
        network: str,
        sidecar: str,
        broker: TransparentEgressBroker,
        grants: Sequence[VirtualCredentialGrant],
    ) -> None:
        # Best-effort cleanup: never raise from teardown (it runs in prepare's
        # failure path, where a teardown error would mask the original cause).
        with contextlib.suppress(Exception):
            await broker.registry.revoke_values_and_wait(
                tuple(grant.presented_value for grant in grants)
            )
        for argv in (["rm", "-f", sidecar], ["network", "rm", network]):
            with contextlib.suppress(Exception):
                await self._docker_exec(argv)
        with contextlib.suppress(Exception):
            await server.close()
