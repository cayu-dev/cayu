from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from cayu.egress import (
    HttpEgressPolicy,
    TransparentEgressBroker,
    VirtualCredentialError,
    VirtualCredentialGrant,
    VirtualCredentialRegistry,
)
from cayu.vaults import SecretRef, StaticVault

pytest.importorskip("cryptography")

# Imported after importorskip: docker_adapter -> proxy_server -> cryptography.
from cayu.egress.docker_adapter import (
    GUEST_CA_PATH,
    DockerEgressAdapter,
    resolve_proxy_bind_host,
)


class _FakeDocker:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        self.calls.append(list(argv))
        return 0, ""


def _broker_with_grant() -> tuple[
    TransparentEgressBroker,
    VirtualCredentialRegistry,
    VirtualCredentialGrant,
]:
    registry = VirtualCredentialRegistry()
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=StaticVault({"stripe_test_key": "sk_test_real"}),
        policies={
            "provider-example": HttpEgressPolicy(
                name="provider-example",
                allowed_hosts=["api.stripe.com"],
                allowed_endpoints=[("POST", "/v1/customers")],
            )
        },
    )
    grant = registry.mint(
        session_id="sess_1",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
        policy_name="provider-example",
    )
    return broker, registry, grant


def test_prepare_builds_internal_network_and_sidecar() -> None:
    docker = _FakeDocker()

    async def run():
        broker, _registry, grant = _broker_with_grant()
        adapter = DockerEgressAdapter(docker_exec=docker, proxy_host="127.0.0.1")
        binding = await adapter.prepare(session_id="sess_1", grants=[grant], broker=broker)
        await binding.close()
        return binding

    binding = asyncio.run(run())

    network = binding.network
    sidecar = binding.sidecar
    assert network is not None
    assert sidecar is not None
    # Resource names are random, decoupled from session_id, but carry a label.
    assert network.startswith("cayu-egress-net-") and network != "cayu-egress-net-sess_1"
    label = "cayu.egress.session=sess_1"
    # Internal network (no internet route) is what makes egress fail-closed.
    assert ["network", "create", "--internal", "--label", label, network] in docker.calls
    # Sidecar attaches to the internal network after starting.
    assert ["network", "connect", network, sidecar] in docker.calls
    # Env overlay routes the sandbox through the sidecar and trusts the CA.
    assert binding.env["HTTPS_PROXY"] == f"http://{sidecar}:8080"
    assert binding.env["https_proxy"] == f"http://{sidecar}:8080"
    assert "HTTP_PROXY" not in binding.env
    assert "http_proxy" not in binding.env
    assert binding.env["SSL_CERT_FILE"] == GUEST_CA_PATH
    assert binding.ca_cert_pem is not None and binding.ca_cert_pem.startswith(b"-----BEGIN")


def test_prepare_names_are_unique_across_sessions() -> None:
    docker = _FakeDocker()

    async def run():
        broker, _registry, grant = _broker_with_grant()
        adapter = DockerEgressAdapter(docker_exec=docker, proxy_host="127.0.0.1")
        first = await adapter.prepare(session_id="same-id", grants=[grant], broker=broker)
        await first.close()
        second = await adapter.prepare(session_id="same-id", grants=[grant], broker=broker)
        await second.close()
        return first.network, second.network

    first_net, second_net = asyncio.run(run())

    assert first_net != second_net  # same session_id must not collide


def test_prepare_uses_injected_proxy_bind_host_resolver() -> None:
    docker = _FakeDocker()
    calls = {"resolver": 0}

    async def resolver() -> str:
        calls["resolver"] += 1
        return "127.0.0.1"

    async def run() -> None:
        broker, _registry, grant = _broker_with_grant()
        adapter = DockerEgressAdapter(
            docker_exec=docker,
            proxy_bind_host_resolver=resolver,
        )
        binding = await adapter.prepare(session_id="sess_1", grants=[grant], broker=broker)
        await binding.close()

    asyncio.run(run())

    assert calls["resolver"] == 1


def test_teardown_revokes_grants_and_removes_resources() -> None:
    docker = _FakeDocker()

    async def run():
        broker, registry, grant = _broker_with_grant()
        adapter = DockerEgressAdapter(docker_exec=docker, proxy_host="127.0.0.1")
        binding = await adapter.prepare(session_id="sess_1", grants=[grant], broker=broker)
        network = binding.network
        sidecar = binding.sidecar
        assert network is not None
        assert sidecar is not None
        await binding.close()
        return registry, grant, network, sidecar

    registry, grant, network, sidecar = asyncio.run(run())

    assert ["rm", "-f", sidecar] in docker.calls
    assert ["network", "rm", network] in docker.calls
    with pytest.raises(VirtualCredentialError):
        registry.lookup(grant.presented_value)


def test_teardown_revokes_all_grants_before_waiting_on_active_lease() -> None:
    docker = _FakeDocker()

    async def run() -> tuple[
        VirtualCredentialRegistry, VirtualCredentialGrant, VirtualCredentialGrant
    ]:
        broker, registry, first = _broker_with_grant()
        second = registry.mint(
            session_id="sess_1",
            env_name="OTHER_KEY",
            secret=SecretRef(name="other_key"),
            destination="api.example.com",
            credential_kind="opaque_bearer",
        )
        first_lease = registry.acquire(first.presented_value)
        adapter = DockerEgressAdapter(docker_exec=docker, proxy_host="127.0.0.1")
        binding = await adapter.prepare(session_id="sess_1", grants=[first, second], broker=broker)

        close_task = asyncio.create_task(binding.close())
        await asyncio.sleep(0)

        assert close_task.done() is False
        with pytest.raises(VirtualCredentialError, match="revoked"):
            registry.lookup(second.presented_value)

        first_lease.close()
        await close_task
        return registry, first, second

    registry, first, second = asyncio.run(run())

    assert registry.was_revoked(first.grant_id)
    assert registry.was_revoked(second.grant_id)


def test_resolve_bind_host_docker_desktop_uses_loopback() -> None:
    async def run(argv: Sequence[str]) -> tuple[int, str]:
        if argv[0] == "info":
            return 0, "Docker Desktop\n"
        return 0, ""

    assert asyncio.run(resolve_proxy_bind_host(run)) == "127.0.0.1"


def test_resolve_bind_host_linux_uses_bridge_gateway() -> None:
    async def run(argv: Sequence[str]) -> tuple[int, str]:
        if argv[0] == "info":
            return 0, "Ubuntu 22.04\n"
        if argv[0] == "network":
            return 0, "172.17.0.1 \n"
        return 1, ""

    assert asyncio.run(resolve_proxy_bind_host(run)) == "172.17.0.1"


def test_resolve_bind_host_falls_back_to_all_interfaces_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def run(argv: Sequence[str]) -> tuple[int, str]:
        return 1, ""  # nothing discoverable

    with caplog.at_level("WARNING"):
        host = asyncio.run(resolve_proxy_bind_host(run))

    assert host == "0.0.0.0"
    assert any("0.0.0.0" in record.message for record in caplog.records)
    assert any("proxy_host" in record.message for record in caplog.records)


def test_prepare_fails_closed_when_docker_errors() -> None:
    class _FailingDocker:
        async def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
            return 1, "network create: permission denied"

    async def run() -> None:
        broker, _registry, grant = _broker_with_grant()
        adapter = DockerEgressAdapter(docker_exec=_FailingDocker(), proxy_host="127.0.0.1")
        await adapter.prepare(session_id="sess_1", grants=[grant], broker=broker)

    from cayu.egress import UnsupportedEgressError

    with pytest.raises(UnsupportedEgressError):
        asyncio.run(run())
