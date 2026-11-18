"""Virtual egress for sandbox pods on Kubernetes, behind the host's pod allocator."""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from hashlib import sha256
from typing import Any, ClassVar, Literal

from cayu._validation import require_durable_clean_nonblank
from cayu.egress._remote_adapter import (
    DEFAULT_PROXY_SERVER_FACTORY,
    ProxyServerFactory,
    prepare_exposed_proxy_binding,
    run_enforcement_preflight,
    run_setup_commands,
    verify_agent_privilege_boundary,
)
from cayu.egress.adapter import (
    EgressBinding,
    RunnerFinalizationResult,
    SandboxEgressAdapter,
    VirtualEgressAllocationPreparation,
    VirtualEgressAllocationReap,
    VirtualEgressRunnerRequest,
    _virtual_egress_execution_capability_evidence,
)
from cayu.egress.broker import TransparentEgressBroker
from cayu.egress.capabilities import EgressCapabilityClaim, EgressCapabilityEvidence
from cayu.egress.errors import UnsupportedEgressError
from cayu.egress.grants import VirtualCredentialGrant
from cayu.egress.proxy_exposure import PodNetworkProxyExposure, ProxyExposure
from cayu.environments.admission import ExecutionCapabilityEvidence
from cayu.runners._sidecar_runner import _positive_float
from cayu.runners._sidecar_transport import (
    SidecarEndpointTransientError,
    SidecarEndpointTransport,
    SidecarRunnerError,
)
from cayu.runners.base import ExecCommand, Runner
from cayu.runners.kubernetes_sandbox import (
    KubernetesSandboxAllocation,
    KubernetesSandboxAllocator,
    KubernetesSandboxRunner,
    KubernetesSandboxSpec,
)

DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 30
DEFAULT_READY_TIMEOUT_SECONDS = 120.0
DEFAULT_DELETE_TIMEOUT_SECONDS = 120.0
_POLL_INTERVAL_SECONDS = 1.0
_POD_NAME_PREFIX = "cayu-sandbox-"
_ALLOCATION_METADATA_KEYS = frozenset({"namespace", "name", "image"})
_RESERVED_RUNNER_OPTIONS = frozenset(
    {
        "endpoint",
        "token",
        "sandbox",
        "endpoint_transport",
        "verify",
        "identity_observer",
        "default_cwd",
        "env_overlay",
        "owner_epoch",
    }
)
_PRIVILEGE_REMEDIATION = (
    "run the agent container as a non-root user with allowPrivilegeEscalation: false, "
    "all capabilities dropped, no cloud credentials, and the sidecar in Kubernetes mode, "
    "which keeps the transport token in the supervisor container"
)
_monotonic: Callable[[], float] = time.monotonic


def kubernetes_sandbox_pod_name(allocation_id: str) -> str:
    """The deterministic pod name for one allocation: a DNS label of 53 characters."""

    digest = sha256(require_durable_clean_nonblank(allocation_id, "allocation_id").encode())
    return _POD_NAME_PREFIX + digest.hexdigest()[:40]


class _EgressSandboxRunner(KubernetesSandboxRunner):
    """A sandbox runner whose sidecar must also relay egress from the pod's loopback."""

    _required_capabilities: ClassVar[tuple[str, ...]] = ("bearer_auth", "boot_id", "egress_relay")


class KubernetesSandboxEgressAdapter(SandboxEgressAdapter):
    """Run virtual-egress sandboxes as Kubernetes pods that the host's allocator owns.

    Each session gets its own proxy listener in this process, advertised through
    ``exposure`` (normally :class:`PodNetworkProxyExposure` with this pod's IP)
    and usable only with a per-session transport token. The ``allocator``
    creates the sandbox pod in ``namespace`` under a name derived from the
    durable allocation id, with the session CA mounted for the agent. The pod's
    sidecar must run in Kubernetes mode and advertise ``egress_relay``: it holds
    the token in its supervisor container and relays agent commands' proxy
    connections, so agent commands never see the token. Once the pod is up the
    listener is restricted to the pod's IP, and a preflight proves the broker is
    reachable while direct public and link-local metadata egress are denied,
    and that agent commands run unprivileged.

    The platform supplies the network boundary (a NetworkPolicy that lets the
    sandbox pod reach only this listener) and the pod's runtime: declare
    ``untrusted_isolation=True`` only when the allocator runs pods under a
    sandboxing runtime such as gVisor. Setup commands run with ``/bin/sh`` as
    the agent, since the sandbox has no trusted lane and its image may lack
    bash. The pod lives as long as the environment: finalization deletes it, and
    neither recovery after a worker stops nor reconnect is supported, since the
    pod trusts only its creating worker's session CA.
    """

    runner_kind = "kubernetes-sandbox"
    process_external_allocation = True
    allocation_provider = "kubernetes"
    allocation_adapter_generation = "kubernetes-sandbox/1"
    supports_reconnect = False
    supports_allocation_fingerprint = True

    def __init__(
        self,
        *,
        allocator: KubernetesSandboxAllocator,
        exposure: ProxyExposure,
        namespace: str,
        bind_host: str = "0.0.0.0",
        loop: asyncio.AbstractEventLoop | None = None,
        proxy_server_factory: ProxyServerFactory = DEFAULT_PROXY_SERVER_FACTORY,
        preflight_timeout_s: int = DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
        ready_timeout_s: float = DEFAULT_READY_TIMEOUT_SECONDS,
        delete_timeout_s: float = DEFAULT_DELETE_TIMEOUT_SECONDS,
        untrusted_isolation: bool = False,
        runner_options: Mapping[str, Any] | None = None,
        endpoint_transport_factory: Callable[[], SidecarEndpointTransport] | None = None,
    ) -> None:
        for method in ("create", "observe", "delete"):
            if not callable(getattr(allocator, method, None)):
                raise TypeError(f"allocator must implement KubernetesSandboxAllocator.{method}.")
        if type(preflight_timeout_s) is not int or preflight_timeout_s <= 0:
            raise ValueError("preflight_timeout_s must be a positive integer.")
        if type(untrusted_isolation) is not bool:
            raise TypeError("untrusted_isolation must be a boolean.")
        self.allocator = allocator
        self.exposure = exposure
        self.namespace = require_durable_clean_nonblank(namespace, "namespace")
        self.bind_host = bind_host
        self.loop = loop
        self.proxy_server_factory = proxy_server_factory
        self.preflight_timeout_s = preflight_timeout_s
        self.ready_timeout_s = _positive_float(ready_timeout_s, "ready_timeout_s")
        self.delete_timeout_s = _positive_float(delete_timeout_s, "delete_timeout_s")
        self.untrusted_isolation = untrusted_isolation
        self.runner_options = dict(runner_options or {})
        overlap = _RESERVED_RUNNER_OPTIONS.intersection(self.runner_options)
        if overlap:
            raise ValueError(
                "runner_options cannot override adapter-owned options: "
                + ", ".join(sorted(overlap))
            )
        self.endpoint_transport_factory = endpoint_transport_factory
        # Each prepared listener, by the session CA its binding carries, so the
        # runner can restrict it to the pod once the pod's IP is known. The
        # listener is held weakly: the binding owns it, and an entry whose
        # binding is never dispatched goes with that binding.
        self._proxy_servers: weakref.WeakKeyDictionary[Any, weakref.ReferenceType[Any]] = (
            weakref.WeakKeyDictionary()
        )
        self._runner_pods: weakref.WeakKeyDictionary[Runner, KubernetesSandboxAllocation] = (
            weakref.WeakKeyDictionary()
        )
        self._runner_capabilities: weakref.WeakKeyDictionary[Runner, EgressCapabilityEvidence] = (
            weakref.WeakKeyDictionary()
        )
        self._runner_preflight_observations: weakref.WeakKeyDictionary[Runner, datetime] = (
            weakref.WeakKeyDictionary()
        )

    def execution_admission_evidence_for(self, requirements):
        return self._execution_admission_executable_declaration(requirements)

    def execution_capability_evidence(
        self,
        runner: Runner | None = None,
    ) -> ExecutionCapabilityEvidence:
        if runner is not None and not isinstance(runner, _EgressSandboxRunner):
            raise TypeError("Kubernetes sandbox adapter received a different runner type.")
        observed_at = (
            self._runner_preflight_observations.get(runner) if runner is not None else None
        )
        privilege: Literal["available", "live_verified"] = (
            "live_verified" if observed_at is not None else "available"
        )
        return _virtual_egress_execution_capability_evidence(
            runner_kind=self.runner_kind,
            runner_ready=runner is not None,
            preflight_observed_at=observed_at,
            untrusted_isolation=self.untrusted_isolation,
            credential_non_possession_posture="available",
            guest_privilege=privilege,
            unprivileged_guest=privilege,
            host_filesystem_isolation=True,
            reconnect=False,
            cancellation_confirmed=(
                getattr(runner, "cancellation_cleanup", None)
                if runner is not None
                else self.runner_options.get("cancellation_cleanup", "command")
            )
            != "none",
        )

    async def prepare(
        self,
        *,
        session_id: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
    ) -> EgressBinding:
        # The pod network is shared by every sandbox pod the NetworkPolicy admits,
        # so the listener is usable only with this session's transport token.
        servers: list[Any] = []

        def capture_server(*args: Any, **kwargs: Any) -> Any:
            server = self.proxy_server_factory(*args, **kwargs)
            servers.append(server)
            return server

        binding = await prepare_exposed_proxy_binding(
            runner_kind=self.runner_kind,
            session_id=session_id,
            broker=broker,
            grants=grants,
            exposure=self.exposure,
            bind_host=self.bind_host,
            loop=self.loop,
            proxy_server_factory=capture_server,
            transport_auth_token=secrets.token_hex(32).encode("ascii"),
        )
        try:
            self._proxy_servers[binding.certificate_authority] = weakref.ref(servers[0])
        except BaseException:
            await binding.close()
            raise
        return binding

    async def create_runner(self, request: VirtualEgressRunnerRequest) -> Runner:
        del request
        raise UnsupportedEgressError(
            "Kubernetes sandbox pods are created only through recoverable allocation."
        )

    async def prepare_allocation_metadata(
        self,
        request: VirtualEgressAllocationPreparation,
    ) -> dict[str, Any]:
        """Pin the namespace, pod name and image before the pod is created."""

        return {
            "namespace": self.namespace,
            "name": kubernetes_sandbox_pod_name(request.allocation_id),
            "image": require_durable_clean_nonblank(request.image, "image"),
        }

    async def create_or_recover_runner(
        self,
        request: VirtualEgressRunnerRequest,
        *,
        allow_create: bool,
    ) -> Runner:
        """Create the intent's pod once; recovery deletes it and fails.

        Only the dispatching worker creates. A pod of the pinned name is admitted
        only when it records the same allocation id. A pod is never recovered:
        it mounts the session CA of the worker that created it, which no later
        worker's listener serves, so recovery deletes the intent's pod (never
        another allocation's) and raises without running any command in it.
        """

        if type(allow_create) is not bool:
            raise TypeError("Kubernetes sandbox recoverable creation requires an exact decision.")
        if request.runner_kind != self.runner_kind:
            raise UnsupportedEgressError(
                f"Kubernetes sandbox adapter cannot create runner kind {request.runner_kind!r}."
            )
        allocation_id = request.allocation_id
        if type(allocation_id) is not str:
            raise TypeError("Kubernetes sandbox creation requires a durable allocation id.")
        namespace, name, image = _validated_allocation_metadata(
            request.allocation_metadata, allocation_id
        )
        if not allow_create:
            await self._delete_pod(namespace, name, allocation_id)
            raise UnsupportedEgressError(
                "Kubernetes sandbox pods are not recovered after their creating worker stops: "
                "the pod trusts that worker's session CA. Its pod was deleted; once session "
                "recovery reaps the abandoned allocation, a new one can be created."
            )
        endpoint = request.binding.proxy_endpoint
        if endpoint is None:
            raise UnsupportedEgressError("Kubernetes sandbox egress requires a proxy endpoint.")
        try:
            PodNetworkProxyExposure(endpoint.host)
        except ValueError as exc:
            raise UnsupportedEgressError(
                "Kubernetes sandbox egress requires a non-public IPv4 proxy endpoint."
            ) from exc
        server_ref = self._proxy_servers.pop(request.binding.certificate_authority, None)
        server = server_ref() if server_ref is not None else None
        if server is None or not callable(getattr(server, "restrict_peer", None)):
            raise UnsupportedEgressError(
                "Kubernetes sandbox egress requires a prepared listener that can restrict its peer."
            )
        certificate = request.binding.ca_cert_pem
        if not certificate:
            raise UnsupportedEgressError(
                "Kubernetes sandbox egress binding did not provide a session CA certificate."
            )
        allocation = await self.allocator.create(
            KubernetesSandboxSpec(
                namespace=namespace,
                name=name,
                allocation_id=allocation_id,
                image=image,
                ca_certificate_pem=certificate,
                ca_path=request.guest_ca_path,
            )
        )
        if type(allocation) is not KubernetesSandboxAllocation:
            raise TypeError("Kubernetes sandbox allocator must return KubernetesSandboxAllocation.")
        if (allocation.namespace, allocation.name) != (namespace, name):
            raise UnsupportedEgressError("Kubernetes sandbox allocator returned a different pod.")
        if allocation.allocation_id != allocation_id:
            # Another owner's pod: it is never adopted, reused or deleted here.
            raise UnsupportedEgressError(
                "Kubernetes sandbox pod of this name belongs to another allocation."
            )
        return await self._admit_runner(request, allocation, server)

    async def _admit_runner(
        self,
        request: VirtualEgressRunnerRequest,
        allocation: KubernetesSandboxAllocation,
        server: Any,
    ) -> Runner:
        options: dict[str, Any] = dict(self.runner_options)
        if self.endpoint_transport_factory is not None:
            options["endpoint_transport"] = self.endpoint_transport_factory()
        else:
            options["verify"] = allocation.verify
        runner = _EgressSandboxRunner(
            endpoint=allocation.endpoint,
            token=allocation.token,
            sandbox=(allocation.namespace, allocation.name),
            identity_observer=self._identity_observer(allocation),
            default_cwd=allocation.workspace_root,
            env_overlay=dict(request.env_overlay),
            owner_epoch=request.ownership_epoch,
            **options,
        )
        try:
            await self._await_sidecar(runner)
            server.restrict_peer(allocation.pod_ip)
            await run_setup_commands(
                runner,
                request,
                shell=lambda command: ExecCommand.process("/bin/sh", "-c", command),
            )
            preflight_observed_at = await run_enforcement_preflight(
                runner,
                request,
                timeout_s=self.preflight_timeout_s,
                metadata_isolation_reason=(
                    "sandbox pod commands reached the link-local metadata endpoint; "
                    "the pod's network policy or sandbox runtime does not deny it"
                ),
                metadata_isolation_remediation=(
                    "deny the sandbox pods every egress except the Cayu proxy listener "
                    "(for example with a NetworkPolicy) and run them under a runtime that "
                    "blocks node metadata, then retry the virtual-egress preflight"
                ),
            )
            await verify_agent_privilege_boundary(
                runner,
                runner_kind=self.runner_kind,
                timeout_s=self.preflight_timeout_s,
                remediation=_PRIVILEGE_REMEDIATION,
            )
        except BaseException:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await runner.close()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self._delete_pod(
                    allocation.namespace, allocation.name, allocation.allocation_id
                )
            raise
        # The probe proved agent commands cannot see the transport token, so only
        # virtual grants can reach command output.
        if request.env_overlay_secret_values_present is not None:
            runner._env_overlay_secret_values_present = request.env_overlay_secret_values_present
        self._runner_pods[runner] = allocation
        self._runner_preflight_observations[runner] = preflight_observed_at
        self._runner_capabilities[runner] = EgressCapabilityEvidence(
            adapter=self.runner_kind,
            claims=(
                EgressCapabilityClaim(
                    capability="proxy_reachability",
                    state="verified",
                    proof_source="agent_preflight",
                    observation="reachable",
                ),
                EgressCapabilityClaim(
                    capability="direct_public_egress",
                    state="verified",
                    proof_source="agent_preflight",
                    observation="denied",
                ),
                EgressCapabilityClaim(
                    capability="metadata_isolation",
                    state="verified",
                    proof_source="agent_preflight",
                    observation="denied",
                ),
            ),
        )
        return runner

    async def _await_sidecar(self, runner: _EgressSandboxRunner) -> None:
        """Wait until the pod's sidecar answers its authenticated health check.

        ``ready_timeout_s`` bounds the whole wait, a health request in flight
        included, and an answer that arrives after it is refused.
        """

        deadline = _monotonic() + self.ready_timeout_s
        last_error: Exception | None = None
        reason = "its health check did not answer"
        while (remaining := deadline - _monotonic()) > 0:
            request_deadline = asyncio.timeout(remaining)
            try:
                async with request_deadline:
                    await runner.observe_boot_id()
            except SidecarEndpointTransientError as exc:
                last_error = exc
            except TimeoutError:
                if not request_deadline.expired():
                    raise
                last_error = None
                break
            else:
                if _monotonic() <= deadline:
                    return
                last_error, reason = None, "its health check answered too late"
                break
            await asyncio.sleep(min(_POLL_INTERVAL_SECONDS, max(deadline - _monotonic(), 0)))
        detail = f": {last_error}" if last_error is not None else f"; {reason}."
        raise SidecarRunnerError(
            "Kubernetes sandbox sidecar did not become ready within "
            f"{self.ready_timeout_s:g} seconds{detail}"
        ) from last_error

    def _identity_observer(self, allocation: KubernetesSandboxAllocation):
        namespace, name, uid = allocation.namespace, allocation.name, allocation.uid

        async def observe() -> dict[str, str]:
            current = await self.allocator.observe(namespace, name)
            if type(current) is not KubernetesSandboxAllocation or current.uid != uid:
                raise SidecarRunnerError("Kubernetes sandbox pod was replaced or removed.")
            return {**current.identity, "pod_uid": uid}

        return observe

    async def _delete_pod(self, namespace: str, name: str, allocation_id: str) -> None:
        """Delete the allocation's pod and return only once a read confirms it is gone."""

        deadline = _monotonic() + self.delete_timeout_s
        while not await self.allocator.delete(namespace, name, allocation_id=allocation_id):
            if _monotonic() >= deadline:
                raise UnsupportedEgressError(
                    "Kubernetes sandbox pod deletion was not confirmed in time."
                )
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    async def reap_allocation(self, request: VirtualEgressAllocationReap) -> None:
        """Delete the intent's pod by its pinned name, or raise while that is unconfirmed.

        The allocator deletes the pod only when it records this allocation id, in
        any phase, so a pod that never started is reaped and a pod of another
        allocation is left alone.
        """

        if type(request) is not VirtualEgressAllocationReap:
            raise TypeError("Kubernetes sandbox cleanup requires VirtualEgressAllocationReap.")
        namespace, name, _image = _validated_allocation_metadata(
            request.allocation_metadata, request.allocation_id
        )
        await self._delete_pod(namespace, name, request.allocation_id)

    async def finalize_runner(
        self,
        runner: Runner,
        *,
        outcome: str | None,
    ) -> RunnerFinalizationResult:
        """Close the runner and delete its pod; every outcome ends the sandbox."""

        del outcome
        if not isinstance(runner, _EgressSandboxRunner):
            raise TypeError("Kubernetes sandbox adapter received a different runner type.")
        allocation = self._runner_pod(runner)
        await runner.close()
        await self._delete_pod(allocation.namespace, allocation.name, allocation.allocation_id)
        return RunnerFinalizationResult(workspace_mutations_quiescent=True)

    async def egress_environment_fingerprint(self, runner: Runner) -> str:
        if not isinstance(runner, _EgressSandboxRunner):
            raise TypeError("Kubernetes sandbox adapter received a different runner type.")
        pod = self._runner_pod(runner)
        return sha256(
            f"kubernetes-sandbox\0{pod.namespace}\0{pod.name}\0{pod.uid}".encode()
        ).hexdigest()

    def capability_evidence(self, runner: Runner) -> EgressCapabilityEvidence:
        if not isinstance(runner, _EgressSandboxRunner):
            raise TypeError("Kubernetes sandbox adapter received a different runner type.")
        capabilities = self._runner_capabilities.get(runner)
        if capabilities is None:
            raise ValueError("Kubernetes sandbox runner has not passed its egress preflight.")
        return capabilities

    def _runner_pod(self, runner: Runner) -> KubernetesSandboxAllocation:
        pod = self._runner_pods.get(runner)
        if pod is None:
            raise ValueError("Kubernetes sandbox runner was not admitted by this adapter.")
        return pod


def _validated_allocation_metadata(
    metadata: Mapping[str, Any],
    allocation_id: str,
) -> tuple[str, str, str]:
    if not isinstance(metadata, Mapping) or set(metadata) != _ALLOCATION_METADATA_KEYS:
        raise UnsupportedEgressError(
            "Kubernetes sandbox allocation metadata has an invalid schema."
        )
    values = []
    for key in ("namespace", "name", "image"):
        value = metadata[key]
        if type(value) is not str:
            raise UnsupportedEgressError(f"Kubernetes sandbox allocation {key} must be a string.")
        values.append(require_durable_clean_nonblank(value, key))
    namespace, name, image = values
    if name != kubernetes_sandbox_pod_name(allocation_id):
        raise UnsupportedEgressError(
            "Kubernetes sandbox allocation name does not belong to this allocation."
        )
    return namespace, name, image


__all__ = [
    "KubernetesSandboxEgressAdapter",
    "kubernetes_sandbox_pod_name",
]
