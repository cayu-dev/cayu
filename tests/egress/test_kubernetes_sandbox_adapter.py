"""Virtual egress for Kubernetes sandbox pods behind the host's allocator.

These tests drive the real durable allocation coordinator, virtual-egress
factory and adapter against an in-memory pod model and a Kubernetes-mode
sidecar whose commands run through an in-process executor. The network
preflight is replaced because it needs a real pod network; the live lane and
the local Docker check cover it.
"""

from __future__ import annotations

import asyncio
import gc
import itertools
import uuid
import weakref
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from tests.core.test_environment_allocation_recovery import (
    _create_session,
    _pending_record,
    _resolve,
    _SimulatedProcessDeath,
)
from tests.egress.test_aws_lambda_microvm_recoverable_allocation import _reap
from tests.runners.lambda_microvm_harness import (
    KUBERNETES_SANDBOX_TOKEN,
    KubernetesSupervisorTransport,
)

import cayu.egress.kubernetes_sandbox_adapter as adapter_module
from cayu.egress import (
    HttpEgressPolicy,
    KubernetesSandboxEgressAdapter,
    PodNetworkProxyExposure,
    UnsupportedEgressError,
    VirtualCredentialSpec,
    VirtualEgressAllocationPreparation,
    VirtualEgressEnvironmentFactory,
)
from cayu.egress.kubernetes_sandbox_adapter import kubernetes_sandbox_pod_name
from cayu.egress.proxy_exposure import ExposedProxy
from cayu.environments import EnvironmentAllocationState, EnvironmentFactoryOperation
from cayu.runners import (
    KubernetesSandboxAllocation,
    KubernetesSandboxSpec,
    SidecarProtocolError,
    SidecarRunnerError,
)
from cayu.sessions import InMemorySessionStore
from cayu.vaults import SecretRef, StaticVault

_IMAGE = "registry.example/agent@sha256:" + "a" * 64
_NAMESPACE = "cayu-sandboxes"
_POD_IP = "100.64.3.4"
_WORKER_IP = "100.64.0.9"
_CA_SERIALS = itertools.count()
# The CA each pod mounts, as its agent container would trust it.
_MOUNTED_CAS: dict[tuple[str, str], bytes] = {}


class _FakeAuthority:
    """A session CA that differs per prepared binding, as real ones do."""

    def __init__(self) -> None:
        self.pem = f"session-ca-{next(_CA_SERIALS)}".encode()

    def ca_cert_pem(self) -> bytes:
        return self.pem


class _FakeProxyServer:
    instances: list[_FakeProxyServer] = []

    def __init__(self, broker: Any, *, loop: Any, host: str, **kwargs: Any) -> None:
        del broker, loop, host
        self.transport_auth_token = kwargs.get("transport_auth_token")
        self.authority = _FakeAuthority()
        self.peers: list[str] = []
        _FakeProxyServer.instances.append(self)

    def restrict_peer(self, address: str) -> None:
        self.peers.append(address)

    async def start(self) -> int:
        return 40123

    async def close(self) -> None:
        return None


@dataclass
class _Pod:
    uid: str
    allocation_id: str
    phase: str = "Running"


class _PodModel:
    """An API server's view of sandbox pods, with crash points around creation."""

    def __init__(
        self, *, crash: str = "", terminating_reads: int = 0, workspace_root: str = "/workspace"
    ) -> None:
        self.workspace_root = workspace_root
        self.identity_override: dict[str, str] = {}
        self.pods: dict[tuple[str, str], _Pod] = {}
        self.crash = crash
        self.terminating_reads = terminating_reads
        self.specs: list[KubernetesSandboxSpec] = []
        self.observed: list[tuple[str, str]] = []
        self.deleted: list[tuple[str, str]] = []

    def _allocation(self, namespace: str, name: str) -> KubernetesSandboxAllocation:
        pod = self.pods[(namespace, name)]
        return KubernetesSandboxAllocation(
            namespace=namespace,
            name=name,
            uid=pod.uid,
            allocation_id=pod.allocation_id,
            pod_ip=_POD_IP,
            endpoint="https://sandbox.pod.invalid:8080",
            token=KUBERNETES_SANDBOX_TOKEN,
            workspace_root=self.workspace_root,
            identity={"agent_image": _IMAGE, **self.identity_override},
        )

    async def create(self, spec: KubernetesSandboxSpec) -> KubernetesSandboxAllocation:
        self.specs.append(spec)
        if self.crash == "before_provider":
            self.crash = ""
            raise _SimulatedProcessDeath("dispatched but the pod was never created")
        key = (spec.namespace, spec.name)
        if key not in self.pods:
            self.pods[key] = _Pod(uid=str(uuid.uuid4()), allocation_id=spec.allocation_id)
            _MOUNTED_CAS[key] = spec.ca_certificate_pem
        if self.crash == "after_provider":
            self.crash = ""
            raise _SimulatedProcessDeath("pod created; acknowledgement lost")
        return self._allocation(*key)

    async def observe(self, namespace: str, name: str) -> KubernetesSandboxAllocation | None:
        self.observed.append((namespace, name))
        pod = self.pods.get((namespace, name))
        if pod is None:
            return None
        if pod.phase != "Running":
            raise RuntimeError(f"Kubernetes sandbox pod is {pod.phase}.")
        return self._allocation(namespace, name)

    async def delete(self, namespace: str, name: str, *, allocation_id: str) -> bool:
        self.deleted.append((namespace, name))
        pod = self.pods.get((namespace, name))
        if pod is None or pod.allocation_id != allocation_id:
            return True
        if self.terminating_reads:
            self.terminating_reads -= 1
            pod.phase = "Terminating"
            return False
        del self.pods[(namespace, name)]
        return True


class _EgressRelaySidecar(KubernetesSupervisorTransport):
    """A Kubernetes-mode sidecar that relays agent egress, as the real one advertises."""

    def __init__(self, root: Path, *, egress_relay: bool = True) -> None:
        super().__init__(root)
        self.egress_relay = egress_relay

    async def health(self, *, token: str, **kwargs: Any) -> dict[str, Any]:
        response = await super().health(token=token, **kwargs)
        if self.egress_relay:
            response["capabilities"] = [*response["capabilities"], "egress_relay"]
        return response


@pytest.fixture(autouse=True)
def _no_network_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    async def preflight(runner: Any, request: Any, **_kwargs: Any) -> datetime:
        # The agent's TLS to the proxy works only if its pod trusts this binding's CA.
        if _MOUNTED_CAS.get((runner.namespace, runner.name)) != request.binding.ca_cert_pem:
            raise UnsupportedEgressError("the sandbox pod trusts another session CA")
        return datetime.now(UTC)

    async def privilege(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(adapter_module, "run_enforcement_preflight", preflight)
    monkeypatch.setattr(adapter_module, "verify_agent_privilege_boundary", privilege)
    monkeypatch.setattr(adapter_module, "_POLL_INTERVAL_SECONDS", 0)
    _FakeProxyServer.instances.clear()
    _MOUNTED_CAS.clear()


def _adapter(model: _PodModel, sidecar: _EgressRelaySidecar, **options: Any):
    return KubernetesSandboxEgressAdapter(
        allocator=model,
        exposure=options.pop("exposure", PodNetworkProxyExposure(_WORKER_IP)),
        namespace=_NAMESPACE,
        proxy_server_factory=_FakeProxyServer,
        runner_options={"poll_interval_s": 0},
        endpoint_transport_factory=lambda: sidecar,
        **options,
    )


def _factory(adapter: KubernetesSandboxEgressAdapter, **options: Any):
    return VirtualEgressEnvironmentFactory(
        policies={
            "model": HttpEgressPolicy(
                name="model",
                allowed_hosts=["api.model.example"],
                allowed_endpoints=[("POST", "/v1/responses")],
            )
        },
        credentials=[
            VirtualCredentialSpec(
                env_name="MODEL_API_KEY",
                secret=SecretRef(name="model"),
                destination="api.model.example",
                policy_name="model",
            )
        ],
        resolver=StaticVault({"model": "model-test-secret"}),
        adapter=adapter,
        image=_IMAGE,
        **options,
    )


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    return root


def test_create_admits_one_pod_with_the_session_ca_and_restricts_the_listener(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = _PodModel(workspace_root=str(_workspace(tmp_path)))
        sidecar = _EgressRelaySidecar(_workspace(tmp_path))
        adapter = _adapter(model, sidecar)
        factory = _factory(adapter, setup_commands=["printf ok > setup-ran"])

        resolution = await _resolve(
            store, session, factory, operation=EnvironmentFactoryOperation.CREATE
        )

        assert resolution.error is None
        [spec] = model.specs
        assert spec.namespace == _NAMESPACE
        assert spec.name == kubernetes_sandbox_pod_name(spec.allocation_id)
        assert len(spec.name) <= 63
        assert spec.image == _IMAGE
        [server] = _FakeProxyServer.instances
        assert spec.ca_certificate_pem == server.authority.ca_cert_pem()
        assert spec.ca_path == "/etc/cayu/ca.pem"
        assert len(server.transport_auth_token) == 64
        assert server.peers == [_POD_IP]
        # Setup ran through /bin/sh in the pod, as the agent.
        assert (_workspace(tmp_path) / "setup-ran").read_text() == "ok"
        assert set(sidecar.tokens_seen) == {KUBERNETES_SANDBOX_TOKEN}
        [runner] = list(adapter._runner_pods)
        assert len(adapter._proxy_servers) == 0  # the listener is not retained past admission
        evidence = adapter.capability_evidence(runner)
        assert {claim.capability for claim in evidence.claims} == {
            "proxy_reachability",
            "direct_public_egress",
            "metadata_isolation",
        }
        checkpoint = await store.load_checkpoint(session.id) or {}
        assert _pending_record(checkpoint) is None
        fingerprint = await adapter.egress_environment_fingerprint(runner)
        assert len(fingerprint) == 64

        result = await adapter.finalize_runner(runner, outcome="completed")
        assert result.workspace_mutations_quiescent is True
        assert model.pods == {}

    asyncio.run(run())


def test_recovery_deletes_the_pod_the_crashed_dispatch_created_and_fails(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = _PodModel(crash="after_provider", workspace_root=str(_workspace(tmp_path)))
        sidecar = _EgressRelaySidecar(_workspace(tmp_path))
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store,
                session,
                _factory(_adapter(model, sidecar)),
                operation=EnvironmentFactoryOperation.CREATE,
            )
        pending = _pending_record(await store.load_checkpoint(session.id) or {})
        assert pending is not None and pending["state"] == "dispatched"
        [name] = {name for _namespace, name in model.pods}

        resolution = await _resolve(
            store,
            session,
            _factory(_adapter(model, sidecar), setup_commands=["printf ran > setup-ran"]),
            operation=EnvironmentFactoryOperation.CREATE,
        )

        # The pod trusts the crashed worker's session CA, which this worker's
        # listener does not serve: it is deleted and nothing runs in it.
        assert isinstance(resolution.error, UnsupportedEgressError)
        assert "not recovered" in str(resolution.error)
        assert len(model.specs) == 1
        assert model.pods == {}
        assert model.deleted == [(_NAMESPACE, name)]
        assert sidecar.tokens_seen == []
        assert not (_workspace(tmp_path) / "setup-ran").exists()

        allocation, reaped = await _reap(store, session, _factory(_adapter(model, sidecar)))
        assert allocation.state is EnvironmentAllocationState.REAPED
        assert reaped is not None and reaped["state"] == "reaped"

    asyncio.run(run())


def test_recovery_never_creates_a_pod_the_crashed_dispatch_did_not(tmp_path: Path) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = _PodModel(crash="before_provider")
        sidecar = _EgressRelaySidecar(_workspace(tmp_path))
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store,
                session,
                _factory(_adapter(model, sidecar)),
                operation=EnvironmentFactoryOperation.CREATE,
            )

        resolution = await _resolve(
            store,
            session,
            _factory(_adapter(model, sidecar)),
            operation=EnvironmentFactoryOperation.CREATE,
        )

        assert resolution.error is not None
        assert len(model.specs) == 1
        assert model.pods == {}
        pending = _pending_record(await store.load_checkpoint(session.id) or {})
        assert pending is not None and pending["state"] == "dispatched"

        allocation, reaped = await _reap(store, session, _factory(_adapter(model, sidecar)))
        assert allocation.state is EnvironmentAllocationState.REAPED
        assert reaped is not None and reaped["state"] == "reaped"
        # Recovery and the reap each confirmed the absent pod gone; neither created one.
        assert model.deleted == [(_NAMESPACE, model.specs[0].name)] * 2
        assert len(model.specs) == 1

    asyncio.run(run())


def test_a_pod_of_another_allocation_is_neither_adopted_nor_deleted(tmp_path: Path) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = _PodModel()
        original_create = model.create

        async def create(spec: KubernetesSandboxSpec) -> KubernetesSandboxAllocation:
            model.pods[(spec.namespace, spec.name)] = _Pod(uid="other", allocation_id="another")
            return await original_create(spec)

        model.create = create  # type: ignore[method-assign]
        sidecar = _EgressRelaySidecar(_workspace(tmp_path))

        resolution = await _resolve(
            store,
            session,
            _factory(_adapter(model, sidecar)),
            operation=EnvironmentFactoryOperation.CREATE,
        )

        assert isinstance(resolution.error, UnsupportedEgressError)
        assert "another allocation" in str(resolution.error)
        assert [pod.uid for pod in model.pods.values()] == ["other"]
        assert model.deleted == []
        assert sidecar.tokens_seen == []

        # Cleanup of this intent leaves the other allocation's pod alone too,
        # whatever its phase.
        [other] = model.pods.values()
        other.phase = "Pending"
        allocation, reaped = await _reap(store, session, _factory(_adapter(model, sidecar)))
        assert allocation.state is EnvironmentAllocationState.REAPED
        assert reaped is not None and reaped["state"] == "reaped"
        assert [pod.uid for pod in model.pods.values()] == ["other"]

    asyncio.run(run())


def test_a_sidecar_without_the_egress_relay_is_refused_and_its_pod_deleted(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = _PodModel()
        sidecar = _EgressRelaySidecar(_workspace(tmp_path), egress_relay=False)

        resolution = await _resolve(
            store,
            session,
            _factory(_adapter(model, sidecar)),
            operation=EnvironmentFactoryOperation.CREATE,
        )

        assert isinstance(resolution.error, SidecarProtocolError)
        assert "egress_relay" in str(resolution.error)
        assert model.pods == {}
        assert _FakeProxyServer.instances[0].peers == []

    asyncio.run(run())


class _StalledHealthSidecar(_EgressRelaySidecar):
    """A sidecar whose health check never answers."""

    async def health(self, *, token: str, **kwargs: Any) -> dict[str, Any]:
        del token, kwargs
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def test_readiness_bounds_a_health_check_in_flight(tmp_path: Path) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = _PodModel()
        sidecar = _StalledHealthSidecar(_workspace(tmp_path))

        resolution = await asyncio.wait_for(
            _resolve(
                store,
                session,
                _factory(_adapter(model, sidecar, ready_timeout_s=0.2)),
                operation=EnvironmentFactoryOperation.CREATE,
            ),
            timeout=10,
        )

        assert isinstance(resolution.error, SidecarRunnerError)
        assert str(resolution.error).endswith(
            "within 0.2 seconds; its health check did not answer."
        )
        assert model.pods == {}
        assert _FakeProxyServer.instances[0].peers == []

    asyncio.run(run())


def test_readiness_refuses_a_health_answer_after_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [0.0]
    monkeypatch.setattr(adapter_module, "_monotonic", lambda: clock[0])

    class LateSidecar(_EgressRelaySidecar):
        async def health(self, *, token: str, **kwargs: Any) -> dict[str, Any]:
            clock[0] += 60
            return await super().health(token=token, **kwargs)

    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = _PodModel()

        resolution = await _resolve(
            store,
            session,
            _factory(_adapter(model, LateSidecar(_workspace(tmp_path)), ready_timeout_s=30)),
            operation=EnvironmentFactoryOperation.CREATE,
        )

        assert isinstance(resolution.error, SidecarRunnerError)
        assert str(resolution.error).endswith(
            "within 30 seconds; its health check answered too late."
        )
        assert model.pods == {}
        assert _FakeProxyServer.instances[0].peers == []

    asyncio.run(run())


def test_a_public_proxy_endpoint_is_refused_before_any_pod_exists(tmp_path: Path) -> None:
    class PublicExposure:
        async def expose(self, *, local_host: str, local_port: int) -> ExposedProxy:
            del local_host
            return ExposedProxy(proxy_url=f"http://8.8.8.8:{local_port}")

    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = _PodModel()
        adapter = _adapter(
            model, _EgressRelaySidecar(_workspace(tmp_path)), exposure=PublicExposure()
        )

        resolution = await _resolve(
            store, session, _factory(adapter), operation=EnvironmentFactoryOperation.CREATE
        )

        assert isinstance(resolution.error, UnsupportedEgressError)
        assert "non-public IPv4" in str(resolution.error)
        assert model.specs == []

    asyncio.run(run())


def test_reap_waits_for_the_pod_to_be_gone_and_stays_pending_while_it_is_not(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = _PodModel(crash="after_provider", workspace_root=str(_workspace(tmp_path)))
        sidecar = _EgressRelaySidecar(_workspace(tmp_path))
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store,
                session,
                _factory(_adapter(model, sidecar)),
                operation=EnvironmentFactoryOperation.CREATE,
            )

        model.terminating_reads = 10**9
        with pytest.raises(UnsupportedEgressError, match="not confirmed"):
            await _reap(store, session, _factory(_adapter(model, sidecar, delete_timeout_s=0.05)))
        pending = _pending_record(await store.load_checkpoint(session.id) or {})
        assert pending is not None and pending["state"] == "reaping"
        assert len(model.pods) == 1

        model.terminating_reads = 2
        allocation, reaped = await _reap(store, session, _factory(_adapter(model, sidecar)))
        assert allocation.state is EnvironmentAllocationState.REAPED
        assert reaped is not None and reaped["state"] == "reaped"
        assert model.pods == {}

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["Pending", "Failed", "Terminating"])
def test_reap_deletes_a_pod_that_is_not_running(tmp_path: Path, phase: str) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        model = _PodModel(crash="after_provider", workspace_root=str(_workspace(tmp_path)))
        sidecar = _EgressRelaySidecar(_workspace(tmp_path))
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store,
                session,
                _factory(_adapter(model, sidecar)),
                operation=EnvironmentFactoryOperation.CREATE,
            )
        # A pod stuck pulling its image, failed or already terminating cannot be
        # observed as running, and is still this allocation's to delete.
        [pod] = model.pods.values()
        pod.phase = phase
        model.terminating_reads = 1 if phase == "Terminating" else 0

        allocation, reaped = await _reap(store, session, _factory(_adapter(model, sidecar)))

        assert allocation.state is EnvironmentAllocationState.REAPED
        assert reaped is not None and reaped["state"] == "reaped"
        assert model.pods == {}
        assert model.observed == []

    asyncio.run(run())


def test_a_binding_never_dispatched_releases_its_listener(tmp_path: Path) -> None:
    class Broker:
        has_credentialless_destinations = False

        async def revoke_authority_and_wait(self, values: tuple[str, ...]) -> None:
            del values

    async def run() -> None:
        adapter = _adapter(_PodModel(), _EgressRelaySidecar(_workspace(tmp_path)))
        binding = await adapter.prepare(
            session_id="session",
            grants=(),
            broker=Broker(),  # type: ignore[arg-type]
        )
        [server] = _FakeProxyServer.instances
        _FakeProxyServer.instances.clear()
        listener = weakref.ref(server)
        del server
        assert len(adapter._proxy_servers) == 1

        # Cancelled before dispatch: the binding is closed and dropped unused.
        await binding.close()
        del binding
        gc.collect()

        assert listener() is None
        assert len(adapter._proxy_servers) == 0

    asyncio.run(run())


def test_allocation_metadata_is_pinned_and_checked_before_any_pod_call(tmp_path: Path) -> None:
    async def run() -> None:
        model = _PodModel()
        adapter = _adapter(model, _EgressRelaySidecar(_workspace(tmp_path)))
        metadata = await adapter.prepare_allocation_metadata(
            VirtualEgressAllocationPreparation(
                allocation_id="alloc-1",
                session_id="session",
                environment_name="remote",
                image=_IMAGE,
            )
        )
        assert metadata == {
            "namespace": _NAMESPACE,
            "name": kubernetes_sandbox_pod_name("alloc-1"),
            "image": _IMAGE,
        }
        assert kubernetes_sandbox_pod_name("alloc-1") != kubernetes_sandbox_pod_name("alloc-2")

        for malformed in (
            {**metadata, "name": kubernetes_sandbox_pod_name("alloc-2")},
            {**metadata, "extra": "x"},
            {**metadata, "image": 3},
        ):
            with pytest.raises(UnsupportedEgressError):
                adapter_module._validated_allocation_metadata(malformed, "alloc-1")
        assert model.specs == model.observed == model.deleted == []

    asyncio.run(run())


def test_the_identity_observer_fails_once_the_pod_is_replaced(tmp_path: Path) -> None:
    async def run() -> None:
        model = _PodModel()
        adapter = _adapter(model, _EgressRelaySidecar(_workspace(tmp_path)))
        spec = KubernetesSandboxSpec(
            namespace=_NAMESPACE,
            name=kubernetes_sandbox_pod_name("alloc-1"),
            allocation_id="alloc-1",
            image=_IMAGE,
            ca_certificate_pem=b"ca",
            ca_path="/etc/cayu/ca.pem",
        )
        allocation = await model.create(spec)
        observe = adapter._identity_observer(allocation)

        assert await observe() == {"pod_uid": allocation.uid, "agent_image": _IMAGE}
        model.identity_override = {"pod_uid": "forged"}
        assert (await observe())["pod_uid"] == allocation.uid
        model.identity_override = {}
        model.pods[(spec.namespace, spec.name)] = _Pod(uid="recreated", allocation_id="alloc-1")
        with pytest.raises(Exception, match="replaced or removed"):
            await observe()

    asyncio.run(run())


def test_allocations_and_specs_validate_what_the_host_reports() -> None:
    base = {
        "namespace": _NAMESPACE,
        "name": "pod",
        "uid": "uid",
        "allocation_id": "alloc",
        "pod_ip": _POD_IP,
        "endpoint": "https://sandbox.pod.invalid:8080",
        "token": "t" * 40,
    }
    KubernetesSandboxAllocation(**base)
    for field_name, value in (
        ("pod_ip", "pod.example"),
        ("endpoint", "http://sandbox.pod.invalid:8080"),
        ("uid", " "),
        ("workspace_root", "workspace"),
    ):
        with pytest.raises((ValueError, TypeError)):
            KubernetesSandboxAllocation(**{**base, field_name: value})
    with pytest.raises(ValueError, match="identity"):
        KubernetesSandboxAllocation(**base, identity={str(i): "x" for i in range(32)})
    with pytest.raises(ValueError, match="ca_certificate_pem"):
        KubernetesSandboxSpec(
            namespace=_NAMESPACE,
            name="pod",
            allocation_id="alloc",
            image=_IMAGE,
            ca_certificate_pem=b"",
            ca_path="/etc/cayu/ca.pem",
        )


def test_runner_options_cannot_override_what_the_adapter_owns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="adapter-owned"):
        KubernetesSandboxEgressAdapter(
            allocator=_PodModel(),
            exposure=PodNetworkProxyExposure(_WORKER_IP),
            namespace=_NAMESPACE,
            runner_options={"owner_epoch": 3},
        )
    with pytest.raises(TypeError, match="KubernetesSandboxAllocator.create"):
        KubernetesSandboxEgressAdapter(
            allocator=object(),  # type: ignore[arg-type]
            exposure=PodNetworkProxyExposure(_WORKER_IP),
            namespace=_NAMESPACE,
        )
