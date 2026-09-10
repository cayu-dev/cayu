from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any

from cayu._exception_groups import add_exception_note_safely
from cayu._task_wait import await_shielded_task_outcome
from cayu.credentials import CredentialMode
from cayu.egress._docker_reconnect import (
    OWNER_LABEL,
    PROXY_ALIAS,
    DockerReconnect,
    _OwnedDockerRunner,
    validate_identity,
)
from cayu.egress._remote_adapter import run_enforcement_preflight
from cayu.egress.adapter import (
    DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
    EgressAuthorityCutoverRequest,
    EgressAuthorityCutoverResult,
    EgressAuthorityRenewalRequest,
    EgressBinding,
    RunnerFinalizationResult,
    SandboxEgressAdapter,
    VirtualEgressRunnerRequest,
    _await_bounded_cleanup_task,
    _consume_accounted_task_cancellation,
    _raise_primary_with_cleanup_cancellation,
    retain_predecessor_binding_cleanup,
    validate_grant_scope,
)
from cayu.egress.authority import (
    EgressAuthorityCutoverReceipt,
    EgressAuthorityCutoverStrategy,
    _build_adapter_verified_egress_authority_cutover_receipt,
)
from cayu.egress.broker import TransparentEgressBroker
from cayu.egress.errors import (
    DockerEgressReconnectError,
    EgressAuthorityCutoverNeedsAttention,
    UnsupportedEgressError,
)
from cayu.egress.grants import VirtualCredentialGrant
from cayu.egress.proxy_server import SessionCertificateAuthority, TransparentEgressProxyServer
from cayu.environments.admission import (
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
)
from cayu.runners._docker_cli import docker_cli_env, normalize_docker_cli_env_allowlist
from cayu.runners.base import Runner
from cayu.runners.docker import DockerRunner, validate_docker_seccomp_profile

_logger = logging.getLogger(__name__)

#: Where the per-session CA is mounted inside the container and trusted from.
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


@dataclass
class _PreparationCleanup:
    teardown: Callable[[], Awaitable[None]]
    task: asyncio.Task[None] | None = None


@dataclass(frozen=True)
class _SidecarTransportAuthorization:
    directory: str
    auth_path: str
    connector_path: str
    token: bytes = field(repr=False)

    def close(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)


def _create_sidecar_transport_authorization(
    *,
    colocated_control_server: bool = False,
) -> _SidecarTransportAuthorization:
    directory = tempfile.mkdtemp(prefix="cayu-egress-sidecar-")
    auth_path = os.path.join(directory, "broker.auth")
    connector_path = os.path.join(directory, "connect-broker")
    token = secrets.token_urlsafe(32).encode("ascii")
    broker_host = b"cayu-control" if colocated_control_server else b"host.docker.internal"
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
            b'"PROXY:' + broker_host + b":cayu-transport.invalid:443,"
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


async def _default_docker_exec(
    argv: Sequence[str],
    *,
    docker_cli_env_allowlist: Sequence[str] = (),
) -> tuple[int, str]:
    docker = shutil.which("docker")
    if not docker:
        raise UnsupportedEgressError("docker CLI not found; cannot enforce virtual egress.")
    process = await asyncio.create_subprocess_exec(
        docker,
        *argv,
        env=docker_cli_env(docker_cli_env_allowlist),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    return process.returncode or 0, stderr.decode("utf-8", "replace")


async def _run_docker_stdout(
    argv: Sequence[str],
    *,
    docker_cli_env_allowlist: Sequence[str] = (),
) -> tuple[int, str]:
    docker = shutil.which("docker")
    if not docker:
        raise UnsupportedEgressError("docker CLI not found; cannot enforce virtual egress.")
    process = await asyncio.create_subprocess_exec(
        docker,
        *argv,
        env=docker_cli_env(docker_cli_env_allowlist),
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

    Egress enforcement is fail-closed by construction: the container joins an
    ``--internal`` Docker network with no route to the internet, so the *only*
    reachable egress is a dual-homed sidecar that forwards to the in-process
    broker. Direct provider calls cannot leave the container. This egress
    topology does not make the container a secure sandbox boundary. Returns the
    network/env the runner must use; ``teardown`` removes the sidecar and network
    and revokes the grants.
    """

    runner_kind = "docker"

    def execution_admission_evidence_for(self, requirements):
        return self._execution_admission_executable_declaration(requirements)

    process_external_allocation = False
    supports_allocation_fingerprint = True
    egress_authority_cutover_strategy = EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH

    def execution_capability_evidence(
        self,
        runner: Runner | None = None,
    ) -> ExecutionCapabilityEvidence:
        """Declare Docker egress without representing containers as sandboxes."""

        if runner is not None and not isinstance(runner, DockerRunner):
            raise TypeError("Docker adapter received a different runner type.")
        available = (
            "real_credential_non_possession",
            "deny_by_default_network",
            "brokered_egress",
            "confirmed_cancellation",
            "confirmed_cleanup",
        )
        unsupported = {
            "untrusted_code_isolation": "container_isolation_unsupported",
            "guest_privilege_containment": "container_privilege_boundary_unsupported",
            "unprivileged_guest": "container_guest_user_unverified",
            "host_filesystem_isolation": "container_host_boundary_unsupported",
            "read_only_host_inputs": "container_host_boundary_unsupported",
            **({} if self.supports_reconnect else {"reconnect": "reconnect_unsupported"}),
        }
        if self.supports_reconnect:
            available = (*available, "reconnect")
            if runner is not None:
                self.reconnect_metadata(runner)
        return ExecutionCapabilityEvidence(
            subject=self.runner_kind,
            claims=(
                *(
                    ExecutionCapabilityClaim(
                        capability=capability,
                        state="available" if runner is not None else "declared",
                        proof_source=(
                            "integration_validation"
                            if runner is not None
                            else "integration_declaration"
                        ),
                        observation="available" if runner is not None else "supported",
                    )
                    for capability in available
                ),
                *(
                    ExecutionCapabilityClaim(
                        capability=capability,
                        state="unsupported",
                        proof_source="integration_declaration",
                        observation="unavailable",
                        reason_code=reason_code,
                        remediation_code=(
                            "select_isolated_execution"
                            if capability != "reconnect"
                            else "select_reconnectable_execution"
                        ),
                    )
                    for capability, reason_code in unsupported.items()
                ),
            ),
        )

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        docker_exec: DockerExec | None = None,
        docker_run: DockerRun | None = None,
        sidecar_image: str = _DEFAULT_SIDECAR_IMAGE,
        proxy_host: str | None = None,
        proxy_bind_host_resolver: Callable[[], Awaitable[str]] | None = None,
        seccomp_profile: str | None = None,
        docker_cli_env_allowlist: Sequence[str] = (),
        control_server_container_id: str | None = None,
        reconnect_state_dir: str | Path | None = None,
        reconnect_timeout_s: float = 20.0,
    ) -> None:
        if type(reconnect_timeout_s) not in {int, float} or not 0 < reconnect_timeout_s <= 60:
            raise ValueError("reconnect_timeout_s must be positive and at most 60 seconds.")
        self._reconnect = (
            None
            if reconnect_state_dir is None
            else DockerReconnect(self, Path(reconnect_state_dir), reconnect_timeout_s)
        )
        if self._reconnect is not None:
            self.egress_authority_cutover_strategy = EgressAuthorityCutoverStrategy.UNSUPPORTED
        if control_server_container_id is not None and (
            type(control_server_container_id) is not str
            or len(control_server_container_id) != 64
            or any(char not in "0123456789abcdef" for char in control_server_container_id)
        ):
            raise ValueError("Control server requires an exact full Docker container ID.")
        if control_server_container_id is not None and proxy_host not in (None, "0.0.0.0"):
            raise ValueError(
                "A colocated control server requires a container-reachable broker listener."
            )
        self._control_server_container_id = control_server_container_id
        self._preparation_cleanups: dict[str, _PreparationCleanup] = {}
        self._docker_cli_env_allowlist = normalize_docker_cli_env_allowlist(
            docker_cli_env_allowlist
        )
        self._loop = loop
        self._docker_exec = docker_exec or partial(
            _default_docker_exec,
            docker_cli_env_allowlist=self._docker_cli_env_allowlist,
        )
        self._docker_run = docker_run or partial(
            _run_docker_stdout,
            docker_cli_env_allowlist=self._docker_cli_env_allowlist,
        )
        self._sidecar_image = sidecar_image
        self._seccomp_profile = validate_docker_seccomp_profile(seccomp_profile)
        # None => auto-resolve the narrowest reachable interface at prepare time
        # (loopback on Docker Desktop, bridge gateway on Linux). An explicit value
        # is used verbatim. The broker still requires a valid unguessable virtual
        # credential + destination/policy, so the listener is not usable on its own.
        self._proxy_host = "0.0.0.0" if control_server_container_id is not None else proxy_host
        self._proxy_bind_host_resolver = proxy_bind_host_resolver or partial(
            resolve_proxy_bind_host,
            run=partial(
                _run_docker_stdout,
                docker_cli_env_allowlist=self._docker_cli_env_allowlist,
            ),
        )

    @property
    def supports_reconnect(self) -> bool:
        return self._reconnect is not None

    def configuration_metadata(self) -> dict[str, Any]:
        return (
            {} if self._reconnect is None else {"docker_reconnect": self._reconnect.configuration}
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
        if self._reconnect is None:
            return await super().prepare_reconnect(
                session_id=session_id,
                environment_name=environment_name,
                grants=grants,
                broker=broker,
                reconnect_metadata=reconnect_metadata,
            )
        return await self._reconnect.prepare(
            session_id=session_id,
            environment_name=environment_name,
            grants=grants,
            broker=broker,
            reconnect_metadata=reconnect_metadata,
        )

    def complete_runner_admission(self, runner: Runner) -> None:
        if self._reconnect is not None:
            if not isinstance(runner, _OwnedDockerRunner) or runner._owner is None:
                raise UnsupportedEgressError("Docker reconnect admission has no owned runner.")
            runner._owner.require_owned()
            runner._admission_complete = True

    async def is_allocation_disposed(self, reconnect_metadata: Mapping[str, Any]) -> bool:
        if self._reconnect is None:
            return False
        return await self._reconnect.is_allocation_disposed(reconnect_metadata)

    def reconnect_metadata(self, runner: Runner) -> dict[str, Any]:
        if self._reconnect is None or not isinstance(runner, DockerRunner):
            return super().reconnect_metadata(runner)
        return self._reconnect.metadata(runner)

    def validate_reconnect_metadata(self, reconnect_metadata: Mapping[str, Any]) -> dict[str, Any]:
        if self._reconnect is None:
            return super().validate_reconnect_metadata(reconnect_metadata)
        return validate_identity(reconnect_metadata)

    async def drain_preparation_cleanup(self) -> None:
        """Retry failed preparation rollback, or join cleanup still in flight.

        Keep this adapter alive until the drain succeeds. A failed or timed-out
        drain retains its exact resource owner and can be called again.
        """
        for network, cleanup in tuple(self._preparation_cleanups.items()):
            if await self._settle_preparation_cleanup(network, cleanup):
                raise asyncio.CancelledError()

    async def _settle_preparation_cleanup(self, network: str, cleanup: _PreparationCleanup) -> bool:
        if self._preparation_cleanups.get(network) is not cleanup:
            return False
        if (
            cleanup.task is not None
            and cleanup.task.done()
            and not cleanup.task.cancelled()
            and cleanup.task.exception() is None
        ):
            del self._preparation_cleanups[network]
            return False
        if cleanup.task is None or cleanup.task.done():

            async def teardown() -> None:
                await cleanup.teardown()

            cleanup.task = asyncio.create_task(teardown())

            def retire(task: asyncio.Task[None]) -> None:
                # Observe failures, but retain the callable for a later retry.
                # Successful late completion needs no further operator sweep.
                if task.cancelled() or task.exception() is not None:
                    return
                if cleanup.task is task and self._preparation_cleanups.get(network) is cleanup:
                    del self._preparation_cleanups[network]

            cleanup.task.add_done_callback(retire)
        return await _await_bounded_cleanup_task(
            cleanup.task,
            timeout_s=DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
            timeout_message="Docker egress prepare rollback timed out.",
        )

    async def prepare(
        self,
        *,
        session_id: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
    ) -> EgressBinding:
        if self._reconnect is not None:
            return await self._reconnect.prepare(
                session_id=session_id, grants=grants, broker=broker
            )
        return await self._prepare(
            session_id=session_id,
            grants=grants,
            broker=broker,
        )

    async def _prepare(
        self,
        *,
        session_id: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
        certificate_authority: SessionCertificateAuthority | None = None,
        owns_certificate_authority: bool = True,
        reconnect_network: str | None = None,
        reconnect_token: str | None = None,
        reconnect_sidecar: str | None = None,
    ) -> EgressBinding:
        validate_grant_scope(session_id=session_id, grants=grants)
        await self.drain_preparation_cleanup()
        control_server_container_id = self._control_server_container_id
        if (
            self._reconnect is None
            and control_server_container_id is not None
            and (
                await self._container_id(control_server_container_id) != control_server_container_id
            )
        ):
            raise UnsupportedEgressError("The exact control server container is unavailable.")
        loop = self._loop or asyncio.get_running_loop()
        if self._proxy_host is not None:
            bind_host = self._proxy_host
        elif self._reconnect is not None:
            try:
                async with asyncio.timeout(self._reconnect.timeout_s):
                    bind_host = await self._proxy_bind_host_resolver()
            except TimeoutError:
                raise DockerEgressReconnectError("daemon_unavailable") from None
        else:
            bind_host = await self._proxy_bind_host_resolver()
        # Resource names use a random token, not the session_id, so distinct
        # sessions can never collide (which would let one teardown remove
        # another live session's network). The session_id rides in a label.
        token = secrets.token_hex(6)
        network = reconnect_network or f"cayu-egress-net-{reconnect_token or token}"
        sidecar = reconnect_sidecar or f"cayu-egress-{token}"
        label = f"{_SESSION_LABEL}={session_id}"
        transport_authorization = (
            _create_sidecar_transport_authorization(colocated_control_server=True)
            if control_server_container_id is not None
            else _create_sidecar_transport_authorization()
        )
        try:
            if certificate_authority is not None or not owns_certificate_authority:
                server = TransparentEgressProxyServer(
                    broker,
                    loop=loop,
                    host=bind_host,
                    transport_auth_token=transport_authorization.token,
                    authority=certificate_authority,
                    owns_authority=owns_certificate_authority,
                )
            else:
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
            ownership_label = (
                [] if reconnect_token is None else ["--label", f"{OWNER_LABEL}={reconnect_token}"]
            )
            if reconnect_network is None:
                await self._run(
                    ["network", "create", "--internal", "--label", label, *ownership_label, network]
                )
            if control_server_container_id is not None and self._reconnect is not None:
                assert reconnect_token is not None
                await self._reconnect.attach_control_server(reconnect_token, network)
            elif control_server_container_id is not None:
                attachment = asyncio.create_task(
                    self._run(
                        [
                            "network",
                            "connect",
                            "--alias",
                            "cayu-control",
                            network,
                            control_server_container_id,
                        ]
                    )
                )
                # Cancelling a Docker CLI waiter cannot prove that the daemon
                # aborted network attachment. Join its exact outcome before
                # allowing rollback to disconnect/remove the private network.
                outcome = await await_shielded_task_outcome(attachment)
                if outcome.error is not None:
                    if outcome.cancellation is not None:
                        raise BaseExceptionGroup(
                            "Control server attachment failed during cancellation.",
                            [outcome.cancellation, outcome.error],
                        )
                    raise outcome.error
                if outcome.cancellation is not None:
                    raise outcome.cancellation
            # Sidecar starts on the default bridge (with host-gateway) so it can
            # reach the host broker, then also joins the internal container network.
            await self._run(
                [
                    "run",
                    "-d",
                    "--name",
                    sidecar,
                    "--label",
                    label,
                    *ownership_label,
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
            await self._run(
                [
                    "network",
                    "connect",
                    *([] if reconnect_token is None else ["--alias", PROXY_ALIAS]),
                    network,
                    sidecar,
                ]
            )
            await self._run(["exec", sidecar, "sh", "-c", _SIDECAR_READY_SCRIPT])
        except BaseException as original:
            _consume_accounted_task_cancellation(original)
            cleanup = _PreparationCleanup(
                partial(
                    self._teardown,
                    server,
                    network,
                    sidecar,
                    broker,
                    grants,
                    transport_authorization,
                    skip_docker=reconnect_token is not None,
                    control_server_container_id=control_server_container_id,
                )
            )
            # Publish the retry owner before starting rollback. The factory
            # cannot own these resources because prepare never returned them.
            self._preparation_cleanups[network] = cleanup
            try:
                rollback_cancelled = await self._settle_preparation_cleanup(network, cleanup)
            except BaseException as cleanup_error:
                add_exception_note_safely(
                    original,
                    f"Docker egress prepare rollback incomplete: {type(cleanup_error).__name__}.",
                )
                _raise_primary_with_cleanup_cancellation(
                    original,
                    cleanup_error,
                    message="Docker egress prepare rollback failed after cancellation.",
                )
            else:
                if rollback_cancelled:
                    raise BaseExceptionGroup(
                        "Docker egress prepare rollback completed after cancellation.",
                        [original, asyncio.CancelledError()],
                    )
            raise

        proxy_url = f"http://{PROXY_ALIAS if reconnect_token is not None else sidecar}:{_SIDECAR_LISTEN_PORT}"
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
                skip_docker=reconnect_token is not None,
                control_server_container_id=control_server_container_id,
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
            certificate_authority=server.authority,
            adopt_certificate_authority=getattr(server, "adopt_authority_ownership", None),
            relinquish_certificate_authority=getattr(
                server,
                "relinquish_authority_ownership",
                None,
            ),
        )

    async def create_runner(self, request: VirtualEgressRunnerRequest) -> Runner:
        if self._reconnect is not None:
            return await self._reconnect.create_runner(request)
        if request.runner_kind != self.runner_kind:
            raise UnsupportedEgressError(
                f"Docker egress adapter cannot create runner kind {request.runner_kind!r}."
            )
        network = request.binding.network
        if network is None:
            raise UnsupportedEgressError(
                "Docker egress adapter did not return a network; refusing to start "
                "a virtual-egress container without enforced routing."
            )
        return await DockerRunner.create(
            request.name,
            image=request.image,
            close_action="remove",
            mount_path=request.host_workspace_path,
            credential_mode=CredentialMode.VIRTUAL_EGRESS,
            network=network,
            env_overlay=dict(request.env_overlay),
            _env_overlay_secret_values_present=(request.env_overlay_secret_values_present),
            ca_mount=(request.ca_cert_host_path, request.guest_ca_path),
            seccomp_profile=self._seccomp_profile,
            setup_commands=request.setup_commands,
            docker_cli_env_allowlist=self._docker_cli_env_allowlist,
        )

    async def egress_environment_fingerprint(self, runner: Runner) -> str:
        if not isinstance(runner, DockerRunner):
            raise TypeError("Docker egress identity requires a DockerRunner.")
        if self._reconnect is not None:
            identity = self._reconnect.metadata(runner)
            if not isinstance(runner, _OwnedDockerRunner) or runner._owner is None:
                raise UnsupportedEgressError("Docker reconnect identity has no owned runner.")
            await self._reconnect.validate_allocation(runner._owner, identity)
            return _docker_environment_fingerprint(identity["container_id"])
        return _docker_environment_fingerprint(await self._container_id(runner.name))

    async def reconcile_authority_cutover(
        self,
        request: EgressAuthorityCutoverRequest,
    ) -> EgressAuthorityCutoverReceipt | None:
        if self._reconnect is not None:
            raise UnsupportedEgressError("Reconnect ownership does not support authority adoption.")
        if type(request) is not EgressAuthorityCutoverRequest:
            raise TypeError("Docker egress reconciliation requires EgressAuthorityCutoverRequest.")
        if not isinstance(request.runner, DockerRunner):
            raise TypeError("Docker egress reconciliation requires a DockerRunner.")
        observed_fingerprint = await self.egress_environment_fingerprint(request.runner)
        if observed_fingerprint != request.environment_fingerprint:
            return None
        if (
            request.current_binding.authority_fingerprint != request.target_authority.fingerprint
            or request.current_binding.authority_generation != request.target_authority.generation
        ):
            return None
        if request.runner.image is None:
            return None
        await run_enforcement_preflight(
            request.runner,
            VirtualEgressRunnerRequest(
                name=request.runner.name,
                runner_kind=self.runner_kind,
                image=request.runner.image,
                binding=request.current_binding,
                env_overlay=dict(request.target_env_overlay),
                env_overlay_secret_values_present=bool(request.target_grants),
                ca_cert_host_path=request.ca_cert_host_path,
                guest_ca_path=request.guest_ca_path,
                setup_commands=(),
                egress_destinations=request.target_egress_destinations,
                session_id=request.session_id,
                environment_name=request.environment_name,
            ),
            timeout_s=20,
            probe_metadata=False,
        )
        return _build_adapter_verified_egress_authority_cutover_receipt(
            expected=request.expected_authority,
            target=request.target_authority,
            environment_fingerprint=observed_fingerprint,
        )

    async def cutover_authority(
        self,
        request: EgressAuthorityCutoverRequest,
    ) -> EgressAuthorityCutoverResult:
        """Rotate broker, sidecar, and network while retaining one exact container."""

        if self._reconnect is not None:
            raise UnsupportedEgressError("Reconnect ownership does not support authority adoption.")
        if type(request) is not EgressAuthorityCutoverRequest:
            raise TypeError("Docker egress cutover requires EgressAuthorityCutoverRequest.")
        if not isinstance(request.runner, DockerRunner):
            raise TypeError("Docker egress cutover requires a DockerRunner.")
        if request.target_authority.runner_kind != self.runner_kind:
            raise UnsupportedEgressError("Docker egress cutover target has the wrong runner kind.")
        if (
            request.current_binding.authority_fingerprint != request.expected_authority.fingerprint
            or request.current_binding.authority_generation != request.expected_authority.generation
        ):
            raise UnsupportedEgressError(
                "Docker egress cutover expected authority is not the active binding."
            )
        current_network = request.current_binding.network
        authority = request.current_binding.certificate_authority
        if current_network is None or not isinstance(authority, SessionCertificateAuthority):
            raise UnsupportedEgressError(
                "Docker egress cutover requires the active network and trusted session CA."
            )
        container_id = await self._container_id(request.runner.name)
        environment_fingerprint = _docker_environment_fingerprint(container_id)
        if environment_fingerprint != request.environment_fingerprint:
            raise UnsupportedEgressError(
                "Docker egress cutover belongs to a different container allocation."
            )
        paused = False
        environment_mutation_dispatched = False
        replacement: EgressBinding | None = None
        replacement_connected = False
        old_network_disconnected = False
        authority_transferred = False
        revocation_cancellation: asyncio.CancelledError | None = None
        try:
            environment_mutation_dispatched = True
            await request.runner.fence_guest_processes_for_egress_cutover()
            await self._run(["pause", request.runner.name])
            paused = True
            replacement = await self._prepare(
                session_id=request.session_id,
                grants=request.target_grants,
                broker=request.target_broker,
                certificate_authority=authority,
                owns_certificate_authority=False,
            )
            if replacement.network is None:
                raise UnsupportedEgressError("Docker replacement binding omitted its network.")
            replacement.bind_authority(request.target_authority)
            await self._run(["network", "connect", replacement.network, request.runner.name])
            replacement_connected = True
            if await request.revoke_current_authority():
                revocation_cancellation = asyncio.CancelledError()
            await self._run(["network", "disconnect", current_network, request.runner.name])
            old_network_disconnected = True
            request.current_binding.transfer_certificate_authority_to(replacement)
            authority_transferred = True
            await request.current_binding.close()
            target_env_overlay = {
                **replacement.env,
                **dict(request.target_env_overlay),
            }
            request.runner.env_overlay = target_env_overlay
            request.runner._env_overlay_secret_values_present = bool(request.target_grants)
            await self._run(["unpause", request.runner.name])
            paused = False
            if request.runner.image is None:
                raise RuntimeError("Docker runner image identity is unavailable at cutover.")
            target_runner_request = VirtualEgressRunnerRequest(
                name=request.runner.name,
                runner_kind=self.runner_kind,
                image=request.runner.image,
                binding=replacement,
                env_overlay=target_env_overlay,
                env_overlay_secret_values_present=bool(request.target_grants),
                ca_cert_host_path=request.ca_cert_host_path,
                guest_ca_path=request.guest_ca_path,
                setup_commands=(),
                egress_destinations=request.target_egress_destinations,
                session_id=request.session_id,
                environment_name=request.environment_name,
            )
            observed_at = await run_enforcement_preflight(
                request.runner,
                target_runner_request,
                timeout_s=20,
                probe_metadata=False,
            )
            del observed_at
            if await self._container_id(request.runner.name) != container_id:
                raise RuntimeError("Docker container identity changed during egress cutover.")
            receipt = _build_adapter_verified_egress_authority_cutover_receipt(
                expected=request.expected_authority,
                target=request.target_authority,
                environment_fingerprint=environment_fingerprint,
            )
            return EgressAuthorityCutoverResult(
                binding=replacement,
                receipt=receipt,
                cancellation=revocation_cancellation,
                cancellation_requests_consumed=(1 if revocation_cancellation is not None else 0),
            )
        except BaseException as original:
            if environment_mutation_dispatched:
                cleanup_errors: list[BaseException] = []
                if not paused:
                    try:
                        await self._run(["pause", request.runner.name])
                        paused = True
                    except BaseException as cleanup_error:
                        cleanup_errors.append(cleanup_error)
                target_authority_installed = old_network_disconnected and authority_transferred
                retained_binding = replacement
                if not target_authority_installed:
                    if (
                        replacement_connected
                        and replacement is not None
                        and replacement.network is not None
                    ):
                        try:
                            await self._run(
                                [
                                    "network",
                                    "disconnect",
                                    replacement.network,
                                    request.runner.name,
                                ]
                            )
                        except BaseException as cleanup_error:
                            cleanup_errors.append(cleanup_error)
                    if replacement is not None:
                        try:
                            await replacement.close()
                        except BaseException as cleanup_error:
                            cleanup_errors.append(cleanup_error)
                    retained_binding = request.current_binding
                elif replacement is not None:
                    retain_predecessor_binding_cleanup(
                        replacement,
                        request.current_binding,
                    )
                attention = EgressAuthorityCutoverNeedsAttention(
                    "Docker egress cutover dispatched a backend mutation but exact target "
                    "activation could not be proven; the container must remain fenced.",
                    replacement_binding=retained_binding,
                    environment_fingerprint=environment_fingerprint,
                    target_authority_installed=target_authority_installed,
                    cancellation=revocation_cancellation,
                    cancellation_requests_consumed=(
                        1 if revocation_cancellation is not None else 0
                    ),
                )
                failures = [original, *cleanup_errors]
                cause: BaseException = (
                    failures[0]
                    if len(failures) == 1
                    else BaseExceptionGroup(
                        "Docker egress cutover and fail-closed settlement both failed.",
                        failures,
                    )
                )
                raise attention from cause
            cleanup_errors = []
            if (
                replacement_connected
                and replacement is not None
                and replacement.network is not None
            ):
                try:
                    await self._run(
                        ["network", "disconnect", replacement.network, request.runner.name]
                    )
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if replacement is not None:
                try:
                    await replacement.close()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if paused:
                try:
                    await self._run(["unpause", request.runner.name])
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "Docker egress cutover failed before dispatch and cleanup also failed.",
                    [original, *cleanup_errors],
                ) from original
            raise

    async def renew_authority(self, request: EgressAuthorityRenewalRequest) -> str:
        """Verify fresh grants on the unchanged container and enforcement path."""

        if self._reconnect is not None:
            raise UnsupportedEgressError("Reconnect ownership does not support authority adoption.")
        if type(request) is not EgressAuthorityRenewalRequest:
            raise TypeError("Docker egress renewal requires EgressAuthorityRenewalRequest.")
        if not isinstance(request.runner, DockerRunner):
            raise TypeError("Docker egress renewal requires a DockerRunner.")
        container_id = await self._container_id(request.runner.name)
        environment_fingerprint = _docker_environment_fingerprint(container_id)
        if environment_fingerprint != request.environment_fingerprint:
            raise UnsupportedEgressError(
                "Docker egress renewal belongs to a different container allocation."
            )
        target_env_overlay = {
            **request.current_binding.env,
            **dict(request.renewed_env_overlay),
        }
        request.runner.env_overlay = target_env_overlay
        request.runner._env_overlay_secret_values_present = bool(request.renewed_grants)
        if request.runner.image is None:
            raise RuntimeError("Docker runner image identity is unavailable at renewal.")
        await run_enforcement_preflight(
            request.runner,
            VirtualEgressRunnerRequest(
                name=request.runner.name,
                runner_kind=self.runner_kind,
                image=request.runner.image,
                binding=request.current_binding,
                env_overlay=target_env_overlay,
                env_overlay_secret_values_present=bool(request.renewed_grants),
                ca_cert_host_path=request.ca_cert_host_path,
                guest_ca_path=request.guest_ca_path,
                setup_commands=(),
                egress_destinations=request.egress_destinations,
                session_id=request.session_id,
                environment_name=request.environment_name,
            ),
            timeout_s=20,
            probe_metadata=False,
        )
        if await self._container_id(request.runner.name) != container_id:
            raise RuntimeError("Docker container identity changed during egress renewal.")
        return environment_fingerprint

    async def finalize_runner(
        self,
        runner: Runner,
        *,
        outcome: str | None,
    ) -> RunnerFinalizationResult:
        if self._reconnect is not None and isinstance(runner, DockerRunner):
            return await self._reconnect.finalize(runner, outcome=outcome)
        if not isinstance(runner, DockerRunner):
            raise TypeError("Docker adapter received a different runner type.")
        await runner._finalize_browser_recordings(normal=outcome == "completed")
        await runner.close()
        return RunnerFinalizationResult(workspace_mutations_quiescent=True)

    async def finalize_runner_for_binding(
        self,
        runner: Runner,
        *,
        outcome: str | None,
    ) -> RunnerFinalizationResult:
        if self._reconnect is not None:
            return await self.finalize_runner(runner, outcome=outcome)
        return await super().finalize_runner_for_binding(runner, outcome=outcome)

    async def park_runner_for_authority_adoption(
        self,
        runner: Runner,
    ) -> RunnerFinalizationResult:
        if self._reconnect is not None:
            raise UnsupportedEgressError("Reconnect ownership does not support authority adoption.")
        if not isinstance(runner, DockerRunner):
            raise TypeError("Docker adapter received a different runner type.")
        await runner.fence_guest_processes_for_egress_cutover()
        return RunnerFinalizationResult(
            workspace_mutations_quiescent=True,
            allocation_preserved=True,
        )

    async def _run(self, argv: Sequence[str]) -> None:
        if self._reconnect is not None:
            await self._reconnect.run(argv)
            return
        exit_code, _stderr = await self._docker_exec(argv)
        if exit_code != 0:
            raise UnsupportedEgressError(
                f"docker {argv[0]} failed while preparing egress (exit_code={exit_code})."
            )

    async def _container_id(self, name: str) -> str:
        exit_code, stdout = await self._docker_run(
            ["inspect", "--type", "container", "--format", "{{.Id}}", name]
        )
        container_id = stdout.strip().lower()
        if (
            exit_code != 0
            or len(container_id) != 64
            or any(character not in "0123456789abcdef" for character in container_id)
        ):
            raise UnsupportedEgressError("Docker container identity could not be verified.")
        return container_id

    async def _teardown(
        self,
        server: TransparentEgressProxyServer,
        network: str,
        sidecar: str,
        broker: TransparentEgressBroker,
        grants: Sequence[VirtualCredentialGrant],
        transport_authorization: _SidecarTransportAuthorization,
        *,
        skip_docker: bool = False,
        control_server_container_id: str | None = None,
    ) -> None:
        # Revoke before releasing any resource that enforced the grant boundary.
        # EgressBinding.close keeps failures retryable and never marks an
        # incomplete teardown closed.
        await broker.revoke_authority_and_wait(tuple(grant.presented_value for grant in grants))
        errors: list[str] = []
        if control_server_container_id is not None and not skip_docker:
            try:
                await self._disconnect_control_server(network, control_server_container_id)
            except Exception as exc:
                errors.append(f"control server detach: {type(exc).__name__}")
        for argv in () if skip_docker else (["rm", "-f", sidecar], ["network", "rm", network]):
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

    async def _disconnect_control_server(self, network: str, container_id: str) -> None:
        code, _ = await self._docker_exec(
            ["network", "disconnect", "--force", network, container_id]
        )
        if code == 0:
            return
        # A failed attach or a lost detach acknowledgement may already have
        # converged. Only positive network readback proves that condition.
        code, output = await self._docker_run(
            ["network", "inspect", "--format", "{{json .Containers}}", network]
        )
        if code == 0:
            try:
                containers = json.loads(output)
            except (ValueError, TypeError):
                containers = None
            if type(containers) is dict and container_id not in containers:
                return
        # A previous close may have removed the whole network. Inspecting the
        # container's networks also proves absence without trusting error prose.
        code, output = await self._docker_run(
            [
                "inspect",
                "--type",
                "container",
                "--format",
                "{{json .NetworkSettings.Networks}}",
                container_id,
            ]
        )
        if code == 0:
            try:
                networks = json.loads(output)
            except (ValueError, TypeError):
                networks = None
            if type(networks) is dict and network not in networks:
                return
        raise UnsupportedEgressError("Control server network detachment is unconfirmed.")


def _docker_resource_is_absent(stderr: str) -> bool:
    """Treat an already-removed teardown target as successful convergence."""

    normalized = stderr.lower()
    return "no such container" in normalized or "not found" in normalized


def _docker_environment_fingerprint(container_id: str) -> str:
    return sha256(f"docker\0{container_id}".encode("ascii")).hexdigest()
