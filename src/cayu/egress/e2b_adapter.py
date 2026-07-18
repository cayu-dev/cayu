from __future__ import annotations

import asyncio
import importlib
import weakref
from collections.abc import Mapping, Sequence
from datetime import datetime
from ipaddress import ip_address
from pathlib import Path
from types import ModuleType
from typing import Any

from cayu.egress._remote_adapter import (
    DEFAULT_PROXY_SERVER_FACTORY,
    DEFAULT_REMOTE_SETUP_COMMAND_TIMEOUT_SECONDS,
    ProxyServerFactory,
    prepare_exposed_proxy_binding,
    run_enforcement_preflight,
    run_setup_commands,
)
from cayu.egress.adapter import (
    DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
    EgressBinding,
    SandboxEgressAdapter,
    VirtualEgressRunnerRequest,
    _virtual_egress_execution_capability_evidence,
)
from cayu.egress.broker import TransparentEgressBroker
from cayu.egress.errors import UnsupportedEgressCapabilityError, UnsupportedEgressError
from cayu.egress.grants import VirtualCredentialGrant
from cayu.egress.proxy_exposure import ProxyExposure
from cayu.environments.admission import ExecutionCapabilityEvidence
from cayu.runners.base import Runner
from cayu.runners.e2b import (
    DEFAULT_E2B_HANDOFF_TIMEOUT_SECONDS,
    E2BGuestHandoffError,
    E2BGuestProvisioner,
    E2BRunner,
)

_RESERVED_E2B_OPTIONS = {
    "allow_internet_access",
    "bootstrap",
    "close_action",
    "cleanup_timeout_s",
    "e2b_module",
    "env_overlay",
    "envs",
    "ensure_default_cwd",
    "exec_user",
    "guest_probe",
    "guest_setup",
    "guest_user",
    "handoff_timeout_s",
    "network",
    "sandbox_timeout_s",
    "secure",
    "template",
}


class E2BEgressAdapter(SandboxEgressAdapter):
    """Enforced virtual egress for E2B cloud sandboxes.

    The caller supplies a ``ProxyExposure`` because E2B cannot reach a process
    listening only on the Cayu host. The exposure must provide a dedicated raw
    TCP endpoint with an IPv4-literal URL that forwards to the local HTTP CONNECT
    proxy.
    """

    runner_kind = "e2b"

    def execution_capability_evidence(
        self,
        runner: Runner | None = None,
    ) -> ExecutionCapabilityEvidence:
        if runner is not None and not isinstance(runner, E2BRunner):
            raise TypeError("E2B adapter received a different runner type.")
        return _virtual_egress_execution_capability_evidence(
            runner_kind=self.runner_kind,
            runner_ready=runner is not None,
            preflight_observed_at=(
                self._runner_preflight_observations.get(runner) if runner is not None else None
            ),
            untrusted_isolation=True,
            credential_non_possession_posture="available",
            guest_privilege="live_verified",
            unprivileged_guest="live_verified",
            host_filesystem_isolation=True,
            reconnect=self.supports_reconnect,
            cancellation_confirmed=(
                getattr(runner, "cancellation_cleanup", None)
                if runner is not None
                else self._options.get("cancellation_cleanup", "command")
            )
            != "none",
        )

    def __init__(
        self,
        *,
        exposure: ProxyExposure,
        bind_host: str = "127.0.0.1",
        loop: asyncio.AbstractEventLoop | None = None,
        e2b_module: ModuleType | Any | None = None,
        e2b_options: Mapping[str, Any] | None = None,
        sandbox_timeout_s: int | None = None,
        proxy_server_factory: ProxyServerFactory = DEFAULT_PROXY_SERVER_FACTORY,
        preflight_timeout_s: int = 20,
    ) -> None:
        if not bind_host.strip():
            raise ValueError("E2B proxy bind_host must be nonblank.")
        if type(preflight_timeout_s) is not int or preflight_timeout_s <= 0:
            raise ValueError("preflight_timeout_s must be a positive integer.")
        options = dict(e2b_options or {})
        reserved = sorted(_RESERVED_E2B_OPTIONS.intersection(options))
        if reserved:
            raise ValueError(
                "E2B virtual-egress security options are adapter-owned: " + ", ".join(reserved)
            )
        self._exposure = exposure
        self._bind_host = bind_host
        self._loop = loop
        self._module = e2b_module
        self._options = options
        self._sandbox_timeout_s = sandbox_timeout_s
        self._proxy_server_factory = proxy_server_factory
        self._preflight_timeout_s = preflight_timeout_s
        self._runner_preflight_observations: weakref.WeakKeyDictionary[
            Runner,
            datetime,
        ] = weakref.WeakKeyDictionary()

    async def prepare(
        self,
        *,
        session_id: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
    ) -> EgressBinding:
        return await prepare_exposed_proxy_binding(
            runner_kind=self.runner_kind,
            session_id=session_id,
            broker=broker,
            grants=grants,
            exposure=self._exposure,
            bind_host=self._bind_host,
            loop=self._loop,
            proxy_server_factory=self._proxy_server_factory,
        )

    async def create_runner(self, request: VirtualEgressRunnerRequest) -> Runner:
        if request.runner_kind != self.runner_kind:
            raise UnsupportedEgressError(
                f"E2B adapter cannot create runner kind {request.runner_kind!r}."
            )
        proxy_endpoint = request.binding.proxy_endpoint
        if proxy_endpoint is None:
            raise UnsupportedEgressError("E2B egress binding did not provide proxy_url.")
        try:
            proxy_address = ip_address(proxy_endpoint.host)
        except ValueError as exc:
            raise UnsupportedEgressError(
                "E2B virtual egress requires an IPv4-literal proxy exposure; "
                "hostname allowlists inspect the tunneled CONNECT destination."
            ) from exc
        if proxy_address.version != 4:
            raise UnsupportedEgressError(
                "E2B virtual egress requires an IPv4-literal proxy exposure; "
                "IPv6 deny-by-default enforcement is not yet verified."
            )
        proxy_ip = str(proxy_address)
        ca_cert_pem = Path(request.ca_cert_host_path).read_bytes()

        async def bootstrap(provisioner: E2BGuestProvisioner) -> None:
            await provisioner.install_file(
                request.guest_ca_path,
                ca_cert_pem,
                mode=0o444,
            )

        async def guest_setup(runner: E2BRunner) -> None:
            preflight_observed_at = await run_enforcement_preflight(
                runner,
                request,
                timeout_s=self._preflight_timeout_s,
            )
            await run_setup_commands(runner, request)
            if request.setup_commands:
                preflight_observed_at = await run_enforcement_preflight(
                    runner,
                    request,
                    timeout_s=self._preflight_timeout_s,
                )
            self._runner_preflight_observations[runner] = preflight_observed_at

        handoff_timeout_s = (
            DEFAULT_E2B_HANDOFF_TIMEOUT_SECONDS
            + ((1 + bool(request.setup_commands)) * self._preflight_timeout_s)
            + (len(request.setup_commands) * DEFAULT_REMOTE_SETUP_COMMAND_TIMEOUT_SECONDS)
        )
        try:
            return await E2BRunner.create_hardened(
                template=request.image,
                sandbox_timeout_s=self._sandbox_timeout_s,
                close_action="kill",
                network={
                    "allow_out": [proxy_ip],
                    "deny_out": ["0.0.0.0/0"],
                },
                env_overlay=request.env_overlay,
                guest_user="user",
                handoff_timeout_s=handoff_timeout_s,
                cleanup_timeout_s=DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
                bootstrap=bootstrap,
                guest_setup=guest_setup,
                e2b_module=self._e2b_module(),
                **self._options,
            )
        except E2BGuestHandoffError as exc:
            raise UnsupportedEgressError(str(exc)) from exc
        except ExceptionGroup as exc:
            authoritative = _find_authoritative_egress_failure(exc)
            if authoritative is None:
                raise
            raise _copy_public_egress_failure(authoritative) from exc

    def _e2b_module(self) -> ModuleType | Any:
        if self._module is not None:
            return self._module
        try:
            return importlib.import_module("e2b")
        except ModuleNotFoundError as exc:
            if exc.name != "e2b":
                raise
            raise UnsupportedEgressError(
                "E2B virtual egress requires the optional e2b package."
            ) from exc


def _find_authoritative_egress_failure(
    error: BaseExceptionGroup,
) -> UnsupportedEgressError | E2BGuestHandoffError | None:
    """Find the public fail-closed error without hiding rollback diagnostics."""

    for item in error.exceptions:
        if isinstance(item, UnsupportedEgressError | E2BGuestHandoffError):
            return item
        if isinstance(item, ExceptionGroup):
            nested = _find_authoritative_egress_failure(item)
            if nested is not None:
                return nested
    return None


def _copy_public_egress_failure(
    error: UnsupportedEgressError | E2BGuestHandoffError,
) -> UnsupportedEgressError:
    """Preserve structured public failures without creating an exception cycle."""

    if isinstance(error, UnsupportedEgressCapabilityError):
        return UnsupportedEgressCapabilityError(
            runner_kind=error.runner_kind,
            capability=error.capability,
            reason=error.reason,
            remediation=error.remediation,
        )
    return UnsupportedEgressError(str(error))
