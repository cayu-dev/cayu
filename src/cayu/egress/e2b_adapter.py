from __future__ import annotations

import asyncio
import contextlib
import importlib
from collections.abc import Mapping, Sequence
from ipaddress import ip_address
from pathlib import Path
from types import ModuleType
from typing import Any

from cayu.egress._remote_adapter import (
    DEFAULT_PROXY_SERVER_FACTORY,
    ProxyServerFactory,
    prepare_exposed_proxy_binding,
    run_enforcement_preflight,
    run_setup_commands,
)
from cayu.egress.adapter import (
    EgressBinding,
    SandboxEgressAdapter,
    VirtualEgressRunnerRequest,
)
from cayu.egress.broker import TransparentEgressBroker
from cayu.egress.errors import UnsupportedEgressError
from cayu.egress.grants import VirtualCredentialGrant
from cayu.egress.proxy_exposure import ProxyExposure
from cayu.runners.base import Runner
from cayu.runners.e2b import E2BRunner

_RESERVED_E2B_OPTIONS = {
    "allow_internet_access",
    "close_action",
    "e2b_module",
    "env_overlay",
    "envs",
    "exec_user",
    "network",
    "sandbox_timeout_s",
    "secure",
    "template",
}

_HARDEN_METADATA_SCRIPT = r"""
set -eu
test "$(id -u)" -eq 0
test -x /usr/sbin/iptables
/usr/sbin/iptables -I OUTPUT 1 -d 169.254.169.254/32 -j REJECT
if id -nG user | tr ' ' '\n' | grep -qx sudo; then
  gpasswd -d user sudo
fi
for path in /usr/bin/sudo /bin/su /usr/bin/su; do
  if [ -e "$path" ]; then
    chmod 0700 "$path"
  fi
done
""".strip()

_VERIFY_GUEST_HARDENING_SCRIPT = r"""
if [ -x /usr/bin/sudo ]; then
  echo "sudo remains executable" >&2
  exit 41
fi
if /usr/sbin/iptables -D OUTPUT -d 169.254.169.254/32 -j REJECT 2>/dev/null; then
  echo "guest can remove metadata firewall rule" >&2
  exit 42
fi
""".strip()


class E2BEgressAdapter(SandboxEgressAdapter):
    """Enforced virtual egress for E2B cloud sandboxes.

    The caller supplies a ``ProxyExposure`` because E2B cannot reach a process
    listening only on the Cayu host. The exposure must provide a dedicated raw
    TCP endpoint with an IPv4-literal URL that forwards to the local HTTP CONNECT
    proxy.
    """

    runner_kind = "e2b"

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

    async def prepare(
        self,
        *,
        session_id: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
    ) -> EgressBinding:
        return await prepare_exposed_proxy_binding(
            runner_kind=self.runner_kind,
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

        runner: E2BRunner | None = None
        try:
            runner = await E2BRunner.create(
                template=request.image,
                sandbox_timeout_s=self._sandbox_timeout_s,
                close_action="kill",
                secure=True,
                allow_internet_access=False,
                network={
                    "allow_out": [proxy_ip],
                    "deny_out": ["0.0.0.0/0"],
                },
                env_overlay=request.env_overlay,
                exec_user="user",
                e2b_module=self._e2b_module(),
                **self._options,
            )
            await self._harden_guest(runner)
            ca_cert_pem = Path(request.ca_cert_host_path).read_bytes()
            await runner.filesystem().write(request.guest_ca_path, ca_cert_pem)
            await run_enforcement_preflight(
                runner,
                request,
                timeout_s=self._preflight_timeout_s,
            )
            await run_setup_commands(runner, request)
            return runner
        except BaseException:
            if runner is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await runner.close()
            raise

    async def _harden_guest(self, runner: E2BRunner) -> None:
        result = await runner._exec_admin(_HARDEN_METADATA_SCRIPT)
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout).strip()[:500]
            raise UnsupportedEgressError(
                f"E2B metadata hardening failed during root bootstrap: {detail}"
            )
        verification = await runner._exec_guest_check(_VERIFY_GUEST_HARDENING_SCRIPT)
        if verification.exit_code != 0:
            detail = (verification.stderr or verification.stdout).strip()[:500]
            raise UnsupportedEgressError(f"E2B guest could bypass metadata hardening: {detail}")

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
