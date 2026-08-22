from __future__ import annotations

import asyncio
import importlib
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from hashlib import sha256
from ipaddress import ip_address
from pathlib import Path
from types import ModuleType
from typing import Any

from cayu._exception_groups import iter_exception_tree
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
    EgressAuthorityCutoverRequest,
    EgressAuthorityCutoverResult,
    EgressAuthorityRenewalRequest,
    EgressBinding,
    RunnerFinalizationResult,
    SandboxEgressAdapter,
    VirtualEgressRunnerRequest,
    _virtual_egress_execution_capability_evidence,
    retain_predecessor_binding_cleanup,
)
from cayu.egress.authority import (
    EgressAuthorityCutoverReceipt,
    EgressAuthorityCutoverStrategy,
    _build_adapter_verified_egress_authority_cutover_receipt,
)
from cayu.egress.broker import TransparentEgressBroker
from cayu.egress.errors import (
    EgressAuthorityCutoverNeedsAttention,
    UnsupportedEgressCapabilityError,
    UnsupportedEgressError,
)
from cayu.egress.grants import VirtualCredentialGrant
from cayu.egress.proxy_exposure import ProxyExposure
from cayu.egress.proxy_server import SessionCertificateAuthority
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
    "handoff_id",
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
    process_external_allocation = True
    allocation_provider = "e2b"
    allocation_adapter_generation = "virtual-egress-hardened-v1"
    egress_authority_cutover_strategy = EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH

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
        protected_bootstrap: (Callable[[E2BGuestProvisioner], Awaitable[None]] | None) = None,
    ) -> None:
        if not bind_host.strip():
            raise ValueError("E2B proxy bind_host must be nonblank.")
        if type(preflight_timeout_s) is not int or preflight_timeout_s <= 0:
            raise ValueError("preflight_timeout_s must be a positive integer.")
        if protected_bootstrap is not None and not callable(protected_bootstrap):
            raise TypeError("E2B protected_bootstrap must be an async callable.")
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
        self._protected_bootstrap = protected_bootstrap
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
            certificate_authority=certificate_authority,
            owns_certificate_authority=owns_certificate_authority,
        )

    async def create_runner(self, request: VirtualEgressRunnerRequest) -> Runner:
        return await self._materialize_runner(
            request,
            allow_create=True,
            handoff_id=None,
        )

    async def create_or_recover_runner(
        self,
        request: VirtualEgressRunnerRequest,
        *,
        allow_create: bool,
    ) -> Runner:
        if type(allow_create) is not bool:
            raise TypeError("E2B recoverable creation requires an exact create decision.")
        handoff_id = request.allocation_id
        if type(handoff_id) is not str:
            raise TypeError("E2B recoverable creation requires a durable allocation id.")
        return await self._materialize_runner(
            request,
            allow_create=allow_create,
            handoff_id=handoff_id,
        )

    async def _materialize_runner(
        self,
        request: VirtualEgressRunnerRequest,
        *,
        allow_create: bool,
        handoff_id: str | None,
    ) -> Runner:
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
            if self._protected_bootstrap is not None:
                await self._protected_bootstrap(provisioner)

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
            if allow_create:
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
                    handoff_id=handoff_id,
                    e2b_module=self._e2b_module(),
                    **self._options,
                )
            if handoff_id is None:
                raise TypeError("E2B recovery requires a durable handoff identity.")
            return await E2BRunner.recover_hardened(
                handoff_id,
                sandbox_timeout_s=self._sandbox_timeout_s,
                close_action="kill",
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

    async def finalize_runner(
        self,
        runner: Runner,
        *,
        outcome: str | None,
    ) -> RunnerFinalizationResult:
        if not isinstance(runner, E2BRunner):
            raise TypeError("E2B adapter received a different runner type.")
        if outcome == "interrupted":
            await runner.fence_guest_processes_for_egress_cutover()
            await runner.detach_preserving_allocation()
            return RunnerFinalizationResult(
                workspace_mutations_quiescent=True,
                allocation_preserved=True,
            )
        await runner.close()
        return RunnerFinalizationResult(workspace_mutations_quiescent=True)

    async def park_runner_for_authority_adoption(
        self,
        runner: Runner,
    ) -> RunnerFinalizationResult:
        if not isinstance(runner, E2BRunner):
            raise TypeError("E2B adapter received a different runner type.")
        await runner.fence_guest_processes_for_egress_cutover()
        return RunnerFinalizationResult(
            workspace_mutations_quiescent=True,
            allocation_preserved=True,
        )

    async def finalize_runner_for_binding(
        self,
        runner: Runner,
        *,
        outcome: str | None,
    ) -> RunnerFinalizationResult:
        """Preserve interrupted E2B bindings after positively fencing guest work."""

        return await self.finalize_runner(runner, outcome=outcome)

    async def egress_environment_fingerprint(self, runner: Runner) -> str:
        if not isinstance(runner, E2BRunner):
            raise TypeError("E2B egress identity requires an E2BRunner.")
        return _e2b_environment_fingerprint(runner.sandbox_id)

    async def reconcile_authority_cutover(
        self,
        request: EgressAuthorityCutoverRequest,
    ) -> EgressAuthorityCutoverReceipt | None:
        if type(request) is not EgressAuthorityCutoverRequest:
            raise TypeError("E2B egress reconciliation requires EgressAuthorityCutoverRequest.")
        if not isinstance(request.runner, E2BRunner):
            raise TypeError("E2B egress reconciliation requires an E2BRunner.")
        observed_fingerprint = await self.egress_environment_fingerprint(request.runner)
        if observed_fingerprint != request.environment_fingerprint:
            return None
        if (
            request.current_binding.authority_fingerprint != request.target_authority.fingerprint
            or request.current_binding.authority_generation != request.target_authority.generation
        ):
            return None
        observed_at = await run_enforcement_preflight(
            request.runner,
            VirtualEgressRunnerRequest(
                name=f"cayu-e2b-reconcile-{request.runner.sandbox_id}",
                runner_kind=self.runner_kind,
                image="e2b-existing-sandbox",
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
            timeout_s=self._preflight_timeout_s,
        )
        self._runner_preflight_observations[request.runner] = observed_at
        return _build_adapter_verified_egress_authority_cutover_receipt(
            expected=request.expected_authority,
            target=request.target_authority,
            environment_fingerprint=observed_fingerprint,
        )

    async def cutover_authority(
        self,
        request: EgressAuthorityCutoverRequest,
    ) -> EgressAuthorityCutoverResult:
        """Rotate the Cayu exposure and broker while retaining one E2B sandbox."""

        if type(request) is not EgressAuthorityCutoverRequest:
            raise TypeError("E2B egress cutover requires EgressAuthorityCutoverRequest.")
        if not isinstance(request.runner, E2BRunner):
            raise TypeError("E2B egress cutover requires an E2BRunner.")
        if request.target_authority.runner_kind != self.runner_kind:
            raise UnsupportedEgressError("E2B egress cutover target has the wrong runner kind.")
        if (
            request.current_binding.authority_fingerprint != request.expected_authority.fingerprint
            or request.current_binding.authority_generation != request.expected_authority.generation
        ):
            raise UnsupportedEgressError(
                "E2B egress cutover expected authority is not the active binding."
            )
        authority = request.current_binding.certificate_authority
        if not isinstance(authority, SessionCertificateAuthority):
            raise UnsupportedEgressError(
                "E2B egress cutover requires the active trusted session CA."
            )
        current_proxy_ip = _e2b_proxy_ip(request.current_binding)
        sandbox_id = request.runner.sandbox_id
        environment_fingerprint = _e2b_environment_fingerprint(sandbox_id)
        if environment_fingerprint != request.environment_fingerprint:
            raise UnsupportedEgressError(
                "E2B egress cutover belongs to a different sandbox allocation."
            )
        replacement: EgressBinding | None = None
        network_fenced = False
        environment_mutation_dispatched = False
        authority_transferred = False
        revocation_cancellation: asyncio.CancelledError | None = None
        try:
            replacement = await self._prepare(
                session_id=request.session_id,
                grants=request.target_grants,
                broker=request.target_broker,
                certificate_authority=authority,
                owns_certificate_authority=False,
            )
            replacement.bind_authority(request.target_authority)
            target_proxy_ip = _e2b_proxy_ip(replacement)
            environment_mutation_dispatched = True
            await request.runner.update_egress_network({"allow_out": [], "deny_out": ["0.0.0.0/0"]})
            network_fenced = True
            await request.runner.fence_guest_processes_for_egress_cutover()
            if await request.revoke_current_authority():
                revocation_cancellation = asyncio.CancelledError()
            request.current_binding.transfer_certificate_authority_to(replacement)
            authority_transferred = True
            await request.current_binding.close()
            await request.runner.update_egress_network(
                {
                    "allow_out": [target_proxy_ip],
                    "deny_out": ["0.0.0.0/0"],
                }
            )
            target_env_overlay = {
                **replacement.env,
                **dict(request.target_env_overlay),
            }
            request.runner.env_overlay = target_env_overlay
            target_runner_request = VirtualEgressRunnerRequest(
                name=f"cayu-e2b-cutover-{sandbox_id}",
                runner_kind=self.runner_kind,
                image="e2b-existing-sandbox",
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
                timeout_s=self._preflight_timeout_s,
            )
            self._runner_preflight_observations[request.runner] = observed_at
            if request.runner.sandbox_id != sandbox_id:
                raise RuntimeError("E2B sandbox identity changed during egress cutover.")
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
                try:
                    await request.runner.update_egress_network(
                        {"allow_out": [], "deny_out": ["0.0.0.0/0"]}
                    )
                    network_fenced = True
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                retained_binding = replacement
                if not authority_transferred:
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
                    "E2B egress cutover dispatched a backend mutation but exact target "
                    "activation could not be proven; the sandbox must remain network-fenced.",
                    replacement_binding=retained_binding,
                    environment_fingerprint=environment_fingerprint,
                    target_authority_installed=authority_transferred,
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
                        "E2B egress cutover and fail-closed settlement both failed.",
                        failures,
                    )
                )
                raise attention from cause
            if replacement is not None:
                cleanup_errors = []
                try:
                    await replacement.close()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if network_fenced:
                try:
                    await request.runner.update_egress_network(
                        {
                            "allow_out": [current_proxy_ip],
                            "deny_out": ["0.0.0.0/0"],
                        }
                    )
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "E2B egress cutover failed before dispatch and cleanup also failed.",
                    [original, *cleanup_errors],
                ) from original
            raise

    async def renew_authority(self, request: EgressAuthorityRenewalRequest) -> str:
        """Verify fresh grants on the unchanged sandbox and proxy route."""

        if type(request) is not EgressAuthorityRenewalRequest:
            raise TypeError("E2B egress renewal requires EgressAuthorityRenewalRequest.")
        if not isinstance(request.runner, E2BRunner):
            raise TypeError("E2B egress renewal requires an E2BRunner.")
        environment_fingerprint = _e2b_environment_fingerprint(request.runner.sandbox_id)
        if environment_fingerprint != request.environment_fingerprint:
            raise UnsupportedEgressError(
                "E2B egress renewal belongs to a different sandbox allocation."
            )
        target_env_overlay = {
            **request.current_binding.env,
            **dict(request.renewed_env_overlay),
        }
        request.runner.env_overlay = target_env_overlay
        observed_at = await run_enforcement_preflight(
            request.runner,
            VirtualEgressRunnerRequest(
                name=f"cayu-e2b-renew-{request.runner.sandbox_id}",
                runner_kind=self.runner_kind,
                image="e2b-existing-sandbox",
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
            timeout_s=self._preflight_timeout_s,
        )
        self._runner_preflight_observations[request.runner] = observed_at
        if _e2b_environment_fingerprint(request.runner.sandbox_id) != environment_fingerprint:
            raise RuntimeError("E2B sandbox identity changed during egress renewal.")
        return environment_fingerprint

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

    for item in iter_exception_tree(error):
        if isinstance(item, UnsupportedEgressError | E2BGuestHandoffError):
            return item
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


def _e2b_proxy_ip(binding: EgressBinding) -> str:
    endpoint = binding.proxy_endpoint
    if endpoint is None:
        raise UnsupportedEgressError("E2B egress binding omitted its proxy endpoint.")
    try:
        address = ip_address(endpoint.host)
    except ValueError as exc:
        raise UnsupportedEgressError(
            "E2B egress cutover requires an IPv4-literal proxy exposure."
        ) from exc
    if address.version != 4:
        raise UnsupportedEgressError("E2B egress cutover requires an IPv4-literal proxy exposure.")
    return str(address)


def _e2b_environment_fingerprint(sandbox_id: str) -> str:
    return sha256(f"e2b\0{sandbox_id}".encode()).hexdigest()
