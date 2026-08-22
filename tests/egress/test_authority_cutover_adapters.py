from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from cayu.egress import (
    EgressAuthorityBindingIdentity,
    EgressAuthorityCutoverRequest,
    EgressAuthorityCutoverStrategy,
    EgressBinding,
    HttpEgressPolicy,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
    build_egress_authority_identity,
)
from cayu.egress.docker_adapter import DockerEgressAdapter, _docker_environment_fingerprint
from cayu.egress.e2b_adapter import E2BEgressAdapter, _e2b_environment_fingerprint
from cayu.egress.errors import EgressAuthorityCutoverNeedsAttention
from cayu.egress.proxy_exposure import ExposedProxy
from cayu.egress.proxy_server import SessionCertificateAuthority
from cayu.runners.base import ExecCommand, ExecResult, Runner
from cayu.vaults import SecretRef, StaticVault


def _authority(runner_kind: str, generation: int, *, allow_post: bool):
    endpoints = [("GET", "/v1/items")]
    if allow_post:
        endpoints.append(("POST", "/v1/items"))
    policy = HttpEgressPolicy(
        name="provider",
        allowed_hosts=("api.example.com",),
        allowed_endpoints=endpoints,
    )
    return build_egress_authority_identity(
        policies={policy.name: policy},
        bindings=(
            EgressAuthorityBindingIdentity(
                destination="api.example.com",
                policy_name=policy.name,
                credential_kind="opaque_bearer",
                credential_authority_fingerprint="1" * 64,
            ),
        ),
        generation=generation,
        authority_source="trusted-app",
        authority_scope="session",
        policy_version=f"v{generation}",
        runner_kind=runner_kind,
        cutover_strategy=EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH,
    )


def _broker_and_grant(session_id: str):
    registry = VirtualCredentialRegistry()
    policy = HttpEgressPolicy(
        name="provider",
        allowed_hosts=("api.example.com",),
        allowed_endpoints=(("GET", "/v1/items"), ("POST", "/v1/items")),
    )
    grant = registry.mint(
        session_id=session_id,
        env_name="PROVIDER_KEY",
        secret=SecretRef(name="provider"),
        destination="api.example.com",
        credential_kind="opaque_bearer",
        policy_name=policy.name,
    )
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=StaticVault({"provider": "provider-secret"}),
        policies={policy.name: policy},
        require_test_mode_credentials=False,
    )
    return broker, grant


class _AuthorityOwnership:
    def __init__(self) -> None:
        self.old_owns = True
        self.new_owns = False

    def old_relinquish(self, _authority) -> None:
        assert self.old_owns
        self.old_owns = False

    def new_adopt(self, _authority) -> None:
        assert not self.new_owns
        self.new_owns = True

    def new_relinquish(self, _authority) -> None:
        assert self.new_owns
        self.new_owns = False


class _FakeDockerRunner(Runner):
    def __init__(self, calls: list[str] | None = None) -> None:
        self.name = "exact-container"
        self.image = "python:3.12-slim"
        self.env_overlay = {"HTTPS_PROXY": "http://old-sidecar:8080"}
        self._env_overlay_secret_values_present = True
        self.calls = calls

    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        del command, kwargs
        return ExecResult()

    async def fence_guest_processes_for_egress_cutover(self) -> None:
        if self.calls is not None:
            self.calls.append("guest-fenced")


class _FakeE2BRunner(Runner):
    def __init__(self, calls: list[str]) -> None:
        self.sandbox_id = "exact-sandbox"
        self.env_overlay = {"HTTPS_PROXY": "http://203.0.113.10:8443"}
        self.calls = calls

    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        del command, kwargs
        return ExecResult()

    async def update_egress_network(self, network) -> None:
        self.calls.append(f"network:{network['allow_out']}")

    async def fence_guest_processes_for_egress_cutover(self) -> None:
        self.calls.append("guest-fenced")


class _UnusedExposure:
    async def expose(self, *, local_host: str, local_port: int) -> ExposedProxy:
        del local_host, local_port
        raise AssertionError("The deterministic cutover test injects staged bindings.")


def _bindings(runner_kind: str, calls: list[str]):
    authority = SessionCertificateAuthority()
    ownership = _AuthorityOwnership()

    async def close_old() -> None:
        calls.append("old-path-closed")

    async def close_new() -> None:
        calls.append("new-path-closed")

    old = EgressBinding(
        runner_kind=runner_kind,
        network="old-network" if runner_kind == "docker" else None,
        proxy_url=(
            "http://old-sidecar:8080" if runner_kind == "docker" else "http://203.0.113.10:8443"
        ),
        guest_ca_path="/etc/cayu/ca.pem",
        ca_cert_pem=authority.ca_cert_pem(),
        certificate_authority=authority,
        relinquish_certificate_authority=ownership.old_relinquish,
        teardown=close_old,
    )
    new = EgressBinding(
        runner_kind=runner_kind,
        network="new-network" if runner_kind == "docker" else None,
        proxy_url=(
            "http://new-sidecar:8080" if runner_kind == "docker" else "http://203.0.113.20:9443"
        ),
        guest_ca_path="/etc/cayu/ca.pem",
        ca_cert_pem=authority.ca_cert_pem(),
        certificate_authority=authority,
        adopt_certificate_authority=ownership.new_adopt,
        relinquish_certificate_authority=ownership.new_relinquish,
        teardown=close_new,
    )
    return old, new, ownership


def test_docker_cutover_keeps_container_and_closes_old_path_before_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        calls: list[str] = []

        async def docker_exec(argv):
            calls.append("docker:" + " ".join(argv))
            return 0, ""

        async def docker_run(argv):
            calls.append("docker-read:" + " ".join(argv))
            return 0, "a" * 64

        expected = _authority("docker", 1, allow_post=False)
        target = _authority("docker", 2, allow_post=True)
        old, new, ownership = _bindings("docker", calls)
        old.bind_authority(expected)
        broker, grant = _broker_and_grant("session-1")
        adapter = DockerEgressAdapter(
            docker_exec=docker_exec,
            docker_run=docker_run,
            proxy_host="127.0.0.1",
        )

        async def prepare(**kwargs):
            del kwargs
            calls.append("new-path-staged")
            return new

        async def preflight(*args, **kwargs):
            del args, kwargs
            calls.append("preflight")
            return datetime.now(UTC)

        async def revoke_current_authority() -> bool:
            calls.append("old-authority-revoked")
            return True

        monkeypatch.setattr(adapter, "_prepare", prepare)
        monkeypatch.setattr("cayu.egress.docker_adapter.DockerRunner", _FakeDockerRunner)
        monkeypatch.setattr("cayu.egress.docker_adapter.run_enforcement_preflight", preflight)
        runner = _FakeDockerRunner(calls)
        result = await adapter.cutover_authority(
            EgressAuthorityCutoverRequest(
                session_id="session-1",
                environment_name="egress",
                owner_fingerprint="b" * 64,
                environment_fingerprint=_docker_environment_fingerprint("a" * 64),
                runner=runner,
                current_binding=old,
                expected_authority=expected,
                target_authority=target,
                target_broker=broker,
                target_grants=(grant,),
                target_env_overlay={"HTTPS_PROXY": "http://new-sidecar:8080"},
                target_egress_destinations=("api.example.com",),
                revoke_current_authority=revoke_current_authority,
                ca_cert_host_path="/tmp/ca.pem",
                guest_ca_path="/etc/cayu/ca.pem",
                invocation_quiescent=True,
            )
        )

        assert result.binding is new
        assert result.receipt.same_allocation is True
        assert result.receipt.old_path_closed is True
        assert isinstance(result.cancellation, asyncio.CancelledError)
        assert result.cancellation_requests_consumed == 1
        assert runner.name == "exact-container"
        assert runner.env_overlay["HTTPS_PROXY"] == "http://new-sidecar:8080"
        assert ownership.old_owns is False and ownership.new_owns is True
        assert calls.index("guest-fenced") < calls.index("docker:pause exact-container")
        assert calls.index("old-authority-revoked") < calls.index("old-path-closed")
        assert calls.index("old-path-closed") < calls.index("preflight")
        assert calls.index("docker:pause exact-container") < calls.index(
            "docker:unpause exact-container"
        )

    asyncio.run(exercise())


def test_e2b_cutover_keeps_sandbox_and_fences_before_retiring_old_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        calls: list[str] = []
        expected = _authority("e2b", 1, allow_post=False)
        target = _authority("e2b", 2, allow_post=True)
        old, new, ownership = _bindings("e2b", calls)
        old.bind_authority(expected)
        broker, grant = _broker_and_grant("session-1")
        adapter = E2BEgressAdapter(exposure=_UnusedExposure())

        async def prepare(**kwargs):
            del kwargs
            calls.append("new-path-staged")
            return new

        async def preflight(*args, **kwargs):
            del args, kwargs
            calls.append("preflight")
            return datetime.now(UTC)

        async def revoke_current_authority() -> bool:
            calls.append("old-authority-revoked")
            return True

        monkeypatch.setattr(adapter, "_prepare", prepare)
        monkeypatch.setattr("cayu.egress.e2b_adapter.E2BRunner", _FakeE2BRunner)
        monkeypatch.setattr("cayu.egress.e2b_adapter.run_enforcement_preflight", preflight)
        runner = _FakeE2BRunner(calls)
        result = await adapter.cutover_authority(
            EgressAuthorityCutoverRequest(
                session_id="session-1",
                environment_name="egress",
                owner_fingerprint="b" * 64,
                environment_fingerprint=_e2b_environment_fingerprint("exact-sandbox"),
                runner=runner,
                current_binding=old,
                expected_authority=expected,
                target_authority=target,
                target_broker=broker,
                target_grants=(grant,),
                target_env_overlay={"HTTPS_PROXY": "http://203.0.113.20:9443"},
                target_egress_destinations=("api.example.com",),
                revoke_current_authority=revoke_current_authority,
                ca_cert_host_path="/tmp/ca.pem",
                guest_ca_path="/etc/cayu/ca.pem",
                invocation_quiescent=True,
            )
        )

        assert result.binding is new
        assert result.receipt.same_allocation is True
        assert isinstance(result.cancellation, asyncio.CancelledError)
        assert result.cancellation_requests_consumed == 1
        assert runner.sandbox_id == "exact-sandbox"
        assert ownership.old_owns is False and ownership.new_owns is True
        assert calls.index("old-authority-revoked") < calls.index("old-path-closed")
        assert calls.index("guest-fenced") < calls.index("old-path-closed")
        assert calls.index("old-path-closed") < calls.index("preflight")
        assert calls[1] == "network:[]"

    asyncio.run(exercise())


def test_docker_cutover_retains_current_binding_when_revocation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        calls: list[str] = []

        async def docker_exec(argv):
            calls.append("docker:" + " ".join(argv))
            return 0, ""

        async def docker_run(argv):
            calls.append("docker-read:" + " ".join(argv))
            return 0, "a" * 64

        expected = _authority("docker", 1, allow_post=False)
        target = _authority("docker", 2, allow_post=True)
        old, new, ownership = _bindings("docker", calls)
        old.bind_authority(expected)
        broker, grant = _broker_and_grant("session-1")
        adapter = DockerEgressAdapter(
            docker_exec=docker_exec,
            docker_run=docker_run,
            proxy_host="127.0.0.1",
        )

        async def prepare(**kwargs):
            del kwargs
            calls.append("new-path-staged")
            return new

        async def revoke_current_authority() -> bool:
            calls.append("old-authority-revoke-failed")
            raise RuntimeError("injected revocation failure")

        monkeypatch.setattr(adapter, "_prepare", prepare)
        monkeypatch.setattr("cayu.egress.docker_adapter.DockerRunner", _FakeDockerRunner)
        runner = _FakeDockerRunner(calls)
        with pytest.raises(EgressAuthorityCutoverNeedsAttention) as raised:
            await adapter.cutover_authority(
                EgressAuthorityCutoverRequest(
                    session_id="session-1",
                    environment_name="egress",
                    owner_fingerprint="b" * 64,
                    environment_fingerprint=_docker_environment_fingerprint("a" * 64),
                    runner=runner,
                    current_binding=old,
                    expected_authority=expected,
                    target_authority=target,
                    target_broker=broker,
                    target_grants=(grant,),
                    target_env_overlay={"HTTPS_PROXY": "http://new-sidecar:8080"},
                    target_egress_destinations=("api.example.com",),
                    revoke_current_authority=revoke_current_authority,
                    ca_cert_host_path="/tmp/ca.pem",
                    guest_ca_path="/etc/cayu/ca.pem",
                    invocation_quiescent=True,
                )
            )

        assert raised.value.target_authority_installed is False
        assert raised.value.replacement_binding is old
        assert ownership.old_owns is True and ownership.new_owns is False
        assert "new-path-closed" in calls
        assert "old-path-closed" not in calls
        assert "docker:pause exact-container" in calls
        assert "docker:unpause exact-container" not in calls

    asyncio.run(exercise())


def test_e2b_cutover_retains_current_binding_when_revocation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        calls: list[str] = []
        expected = _authority("e2b", 1, allow_post=False)
        target = _authority("e2b", 2, allow_post=True)
        old, new, ownership = _bindings("e2b", calls)
        old.bind_authority(expected)
        broker, grant = _broker_and_grant("session-1")
        adapter = E2BEgressAdapter(exposure=_UnusedExposure())

        async def prepare(**kwargs):
            del kwargs
            calls.append("new-path-staged")
            return new

        async def revoke_current_authority() -> bool:
            calls.append("old-authority-revoke-failed")
            raise RuntimeError("injected revocation failure")

        monkeypatch.setattr(adapter, "_prepare", prepare)
        monkeypatch.setattr("cayu.egress.e2b_adapter.E2BRunner", _FakeE2BRunner)
        runner = _FakeE2BRunner(calls)
        with pytest.raises(EgressAuthorityCutoverNeedsAttention) as raised:
            await adapter.cutover_authority(
                EgressAuthorityCutoverRequest(
                    session_id="session-1",
                    environment_name="egress",
                    owner_fingerprint="b" * 64,
                    environment_fingerprint=_e2b_environment_fingerprint("exact-sandbox"),
                    runner=runner,
                    current_binding=old,
                    expected_authority=expected,
                    target_authority=target,
                    target_broker=broker,
                    target_grants=(grant,),
                    target_env_overlay={"HTTPS_PROXY": "http://203.0.113.20:9443"},
                    target_egress_destinations=("api.example.com",),
                    revoke_current_authority=revoke_current_authority,
                    ca_cert_host_path="/tmp/ca.pem",
                    guest_ca_path="/etc/cayu/ca.pem",
                    invocation_quiescent=True,
                )
            )

        assert raised.value.target_authority_installed is False
        assert raised.value.replacement_binding is old
        assert ownership.old_owns is True and ownership.new_owns is False
        assert "new-path-closed" in calls
        assert "old-path-closed" not in calls
        assert calls.count("network:[]") == 2

    asyncio.run(exercise())


def test_docker_cutover_fences_after_pause_acknowledgement_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        calls: list[str] = []
        pause_calls = 0

        async def docker_exec(argv):
            nonlocal pause_calls
            calls.append("docker:" + " ".join(argv))
            if argv[:2] == ["pause", "exact-container"]:
                pause_calls += 1
                if pause_calls == 1:
                    raise RuntimeError("pause acknowledgement was lost")
            return 0, ""

        async def docker_run(argv):
            calls.append("docker-read:" + " ".join(argv))
            return 0, "a" * 64

        expected = _authority("docker", 1, allow_post=False)
        target = _authority("docker", 2, allow_post=True)
        old, new, ownership = _bindings("docker", calls)
        old.bind_authority(expected)
        broker, grant = _broker_and_grant("session-1")
        adapter = DockerEgressAdapter(
            docker_exec=docker_exec,
            docker_run=docker_run,
            proxy_host="127.0.0.1",
        )

        async def prepare(**kwargs):
            del kwargs
            calls.append("new-path-staged")
            return new

        monkeypatch.setattr(adapter, "_prepare", prepare)
        monkeypatch.setattr("cayu.egress.docker_adapter.DockerRunner", _FakeDockerRunner)
        runner = _FakeDockerRunner(calls)

        with pytest.raises(EgressAuthorityCutoverNeedsAttention) as raised:
            await adapter.cutover_authority(
                EgressAuthorityCutoverRequest(
                    session_id="session-1",
                    environment_name="egress",
                    owner_fingerprint="b" * 64,
                    environment_fingerprint=_docker_environment_fingerprint("a" * 64),
                    runner=runner,
                    current_binding=old,
                    expected_authority=expected,
                    target_authority=target,
                    target_broker=broker,
                    target_grants=(grant,),
                    target_env_overlay={"HTTPS_PROXY": "http://new-sidecar:8080"},
                    target_egress_destinations=("api.example.com",),
                    revoke_current_authority=lambda: asyncio.sleep(0, result=False),
                    ca_cert_host_path="/tmp/ca.pem",
                    guest_ca_path="/etc/cayu/ca.pem",
                    invocation_quiescent=True,
                )
            )

        assert raised.value.target_authority_installed is False
        assert raised.value.replacement_binding is old
        assert pause_calls == 2
        assert "docker:unpause exact-container" not in calls
        assert "new-path-staged" not in calls
        assert "old-path-closed" not in calls
        assert ownership.old_owns is True and ownership.new_owns is False

    asyncio.run(exercise())


def test_e2b_cutover_fences_after_network_update_acknowledgement_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        calls: list[str] = []
        expected = _authority("e2b", 1, allow_post=False)
        target = _authority("e2b", 2, allow_post=True)
        old, new, ownership = _bindings("e2b", calls)
        old.bind_authority(expected)
        broker, grant = _broker_and_grant("session-1")
        adapter = E2BEgressAdapter(exposure=_UnusedExposure())

        async def prepare(**kwargs):
            del kwargs
            calls.append("new-path-staged")
            return new

        monkeypatch.setattr(adapter, "_prepare", prepare)
        monkeypatch.setattr("cayu.egress.e2b_adapter.E2BRunner", _FakeE2BRunner)
        runner = _FakeE2BRunner(calls)
        network_updates = 0

        async def update_network(network) -> None:
            nonlocal network_updates
            network_updates += 1
            calls.append(f"network:{network['allow_out']}")
            if network_updates == 1:
                raise RuntimeError("network acknowledgement was lost")

        monkeypatch.setattr(runner, "update_egress_network", update_network)

        with pytest.raises(EgressAuthorityCutoverNeedsAttention) as raised:
            await adapter.cutover_authority(
                EgressAuthorityCutoverRequest(
                    session_id="session-1",
                    environment_name="egress",
                    owner_fingerprint="b" * 64,
                    environment_fingerprint=_e2b_environment_fingerprint("exact-sandbox"),
                    runner=runner,
                    current_binding=old,
                    expected_authority=expected,
                    target_authority=target,
                    target_broker=broker,
                    target_grants=(grant,),
                    target_env_overlay={"HTTPS_PROXY": "http://203.0.113.20:9443"},
                    target_egress_destinations=("api.example.com",),
                    revoke_current_authority=lambda: asyncio.sleep(0, result=False),
                    ca_cert_host_path="/tmp/ca.pem",
                    guest_ca_path="/etc/cayu/ca.pem",
                    invocation_quiescent=True,
                )
            )

        assert raised.value.target_authority_installed is False
        assert raised.value.replacement_binding is old
        assert network_updates == 2
        assert "new-path-closed" in calls
        assert "old-path-closed" not in calls
        assert ownership.old_owns is True and ownership.new_owns is False

    asyncio.run(exercise())
