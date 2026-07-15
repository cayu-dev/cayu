from __future__ import annotations

import asyncio
import importlib
import posixpath
from collections.abc import Sequence
from types import ModuleType
from typing import Any

from cayu.egress._remote_adapter import (
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
    _await_bounded_cleanup_task,
)
from cayu.egress.broker import TransparentEgressBroker
from cayu.egress.errors import UnsupportedEgressError
from cayu.egress.grants import VirtualCredentialGrant
from cayu.egress.proxy_exposure import (
    MICROSANDBOX_HOST,
    MicrosandboxHostProxyExposure,
    ProxyExposure,
)
from cayu.egress.proxy_server import DualStackLoopbackEgressProxyServer
from cayu.runners.base import Runner
from cayu.runners.microsandbox import MicrosandboxRunner


class MicrosandboxEgressAdapter(SandboxEgressAdapter):
    """Enforced virtual egress for local Microsandbox microVMs."""

    runner_kind = "microsandbox"

    def __init__(
        self,
        *,
        exposure: ProxyExposure | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        microsandbox_module: ModuleType | Any | None = None,
        proxy_server_factory: ProxyServerFactory = DualStackLoopbackEgressProxyServer,
        preflight_timeout_s: int = 15,
    ) -> None:
        if type(preflight_timeout_s) is not int or preflight_timeout_s <= 0:
            raise ValueError("preflight_timeout_s must be a positive integer.")
        self._exposure = exposure or MicrosandboxHostProxyExposure()
        self._loop = loop
        self._module = microsandbox_module
        self._proxy_server_factory = proxy_server_factory
        self._preflight_timeout_s = preflight_timeout_s

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
            bind_host="127.0.0.1",
            loop=self._loop,
            proxy_server_factory=self._proxy_server_factory,
        )

    async def create_runner(self, request: VirtualEgressRunnerRequest) -> Runner:
        if request.runner_kind != self.runner_kind:
            raise UnsupportedEgressError(
                f"Microsandbox adapter cannot create runner kind {request.runner_kind!r}."
            )
        proxy_endpoint = request.binding.proxy_endpoint
        if proxy_endpoint is None:
            raise UnsupportedEgressError("Microsandbox egress binding did not provide proxy_url.")
        if proxy_endpoint.host != MICROSANDBOX_HOST:
            raise UnsupportedEgressError(
                "Microsandbox virtual egress requires host.microsandbox.internal exposure."
            )

        module = self._microsandbox_module()
        network = module.Network(
            policy=module.NetworkPolicy(
                default_egress=module.Action.DENY,
                rules=(
                    *module.Rule.allow_dns(),
                    module.Rule.allow(
                        destination=module.Destination.group(module.DestGroup.HOST),
                        protocol=module.Protocol.TCP,
                        port=proxy_endpoint.port,
                    ),
                ),
            )
        )
        guest_ca_dir = posixpath.dirname(request.guest_ca_path)
        patches = [
            module.Patch.mkdir(guest_ca_dir, mode=0o755),
            module.Patch.copy_file(
                request.ca_cert_host_path,
                request.guest_ca_path,
                mode=0o644,
                replace=True,
            ),
        ]

        runner: MicrosandboxRunner | None = None
        try:
            runner = await MicrosandboxRunner.create(
                request.name,
                image=request.image,
                close_action="remove",
                env_overlay=request.env_overlay,
                network=network,
                patches=patches,
                sandbox_module=module,
            )
            await run_enforcement_preflight(
                runner,
                request,
                timeout_s=self._preflight_timeout_s,
            )
            await run_setup_commands(runner, request)
            return runner
        except BaseException as original:
            if runner is not None:
                cleanup_task = asyncio.create_task(runner.close())
                try:
                    await _await_bounded_cleanup_task(
                        cleanup_task,
                        timeout_s=DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
                        timeout_message="Microsandbox egress runner rollback timed out.",
                    )
                except BaseException as cleanup_error:
                    original.add_note(
                        "Microsandbox egress runner rollback incomplete: "
                        f"{type(cleanup_error).__name__}."
                    )
            raise

    def _microsandbox_module(self) -> ModuleType | Any:
        if self._module is not None:
            return self._module
        try:
            return importlib.import_module("microsandbox")
        except ModuleNotFoundError as exc:
            if exc.name != "microsandbox":
                raise
            raise UnsupportedEgressError(
                "Microsandbox virtual egress requires the optional microsandbox package."
            ) from exc
