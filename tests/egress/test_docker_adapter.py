from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from cayu.egress import (
    ApprovedEgressDestination,
    HttpEgressPolicy,
    TransparentEgressBroker,
    UnsupportedEgressError,
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


class _FlakyCleanupDocker(_FakeDocker):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_failures = 0
        self.network_removed = False
        self.network_absent_seen = False

    async def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        self.calls.append(list(argv))
        if argv[0] == "rm" and self.cleanup_failures == 0:
            self.cleanup_failures += 1
            return 1, "sidecar still stopping"
        if argv[:2] == ["network", "rm"]:
            if self.network_removed:
                self.network_absent_seen = True
                return 1, "Error response from daemon: network not found"
            self.network_removed = True
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


def _credentialless_broker() -> TransparentEgressBroker:
    return TransparentEgressBroker(
        registry=VirtualCredentialRegistry(),
        policies={
            "public-docs": HttpEgressPolicy(
                name="public-docs",
                allowed_hosts=["docs.example.com"],
                allowed_endpoints=[("GET", "/sdk/index.json")],
            )
        },
        approved_destinations=[
            ApprovedEgressDestination(
                destination="docs.example.com",
                policy_name="public-docs",
            )
        ],
    )


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
    readiness = next(argv for argv in docker.calls if argv[:2] == ["exec", sidecar])
    assert "/proc/1/comm" in readiness[-1]
    # Env overlay routes the sandbox through the sidecar and trusts the CA.
    assert binding.env["HTTPS_PROXY"] == f"http://{sidecar}:8080"
    assert binding.proxy_url == binding.env["HTTPS_PROXY"]
    assert binding.env["https_proxy"] == f"http://{sidecar}:8080"
    assert "HTTP_PROXY" not in binding.env
    assert "http_proxy" not in binding.env
    assert binding.env["SSL_CERT_FILE"] == GUEST_CA_PATH
    assert binding.ca_cert_pem is not None and binding.ca_cert_pem.startswith(b"-----BEGIN")


def test_prepare_authenticates_sidecar_without_putting_broker_credentials_in_guest() -> None:
    docker = _FakeDocker()

    async def run() -> tuple[dict[str, str], dict[str, object], list[list[str]], str]:
        adapter = DockerEgressAdapter(docker_exec=docker, proxy_host="0.0.0.0")
        binding = await adapter.prepare(
            session_id="sess_public_docs",
            grants=[],
            broker=_credentialless_broker(),
        )
        env = dict(binding.env)
        metadata = dict(binding.metadata)
        sidecar_run = next(argv for argv in docker.calls if argv[0] == "run")
        connector_mount = next(
            value
            for value in sidecar_run
            if value.startswith("type=bind,") and "dst=/run/cayu/connect-broker" in value
        )
        connector_source = next(
            part.removeprefix("src=")
            for part in connector_mount.split(",")
            if part.startswith("src=")
        )
        connector_script = Path(connector_source).read_text()
        await binding.close()
        return env, metadata, docker.calls, connector_script

    env, metadata, calls, connector_script = asyncio.run(run())
    sidecar_run = next(argv for argv in calls if argv[0] == "run")
    mounts = [value for value in sidecar_run if value.startswith("type=bind,")]

    assert len(mounts) == 2
    assert all("readonly" in mount for mount in mounts)
    assert any("dst=/run/cayu/broker.auth" in mount for mount in mounts)
    assert any("dst=/run/cayu/connect-broker" in mount for mount in mounts)
    assert sidecar_run[-4:] == [
        "--entrypoint",
        "/run/cayu/connect-broker",
        "alpine/socat",
        "listen",
    ]
    assert any(value.startswith("CAYU_BROKER_PORT=") for value in sidecar_run)
    assert "ip route show default" in connector_script
    assert '$2 != "lo" && $2 != default_if' in connector_script
    assert "TCP-LISTEN:8080,bind=${bind_ip},fork,reuseaddr" in connector_script
    assert "PROXY:host.docker.internal:cayu-transport.invalid:443" in connector_script
    assert "proxyport=${CAYU_BROKER_PORT}" in connector_script
    assert "proxyauthfile=/run/cayu/broker.auth" in connector_script
    assert not any("broker.auth" in value for value in env.values())
    assert not any("connect-broker" in value for value in env.values())
    assert not any("broker.auth" in str(value) for value in metadata.values())
    assert not any("connect-broker" in str(value) for value in metadata.values())
    for mount in mounts:
        source = next(
            part.removeprefix("src=") for part in mount.split(",") if part.startswith("src=")
        )
        assert Path(source).exists() is False


def test_transport_authorization_repr_omits_the_raw_token() -> None:
    import cayu.egress.docker_adapter as docker_adapter

    authorization = docker_adapter._create_sidecar_transport_authorization()
    try:
        assert authorization.token.decode("ascii") not in repr(authorization)
    finally:
        authorization.close()


def test_prepare_removes_transport_authorization_when_proxy_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.egress.docker_adapter as docker_adapter

    created: list[docker_adapter._SidecarTransportAuthorization] = []
    original_create = docker_adapter._create_sidecar_transport_authorization

    def record_authorization() -> docker_adapter._SidecarTransportAuthorization:
        authorization = original_create()
        created.append(authorization)
        return authorization

    monkeypatch.setattr(
        docker_adapter,
        "_create_sidecar_transport_authorization",
        record_authorization,
    )
    adapter = DockerEgressAdapter(docker_exec=_FakeDocker(), proxy_host="")

    with pytest.raises(ValueError, match="listen hosts must be nonblank"):
        asyncio.run(
            adapter.prepare(
                session_id="sess_constructor_failure",
                grants=[],
                broker=_credentialless_broker(),
            )
        )

    assert len(created) == 1
    assert Path(created[0].directory).exists() is False


def test_prepare_names_are_unique_across_sessions() -> None:
    docker = _FakeDocker()

    async def run():
        broker, registry, _grant = _broker_with_grant()
        grants = [
            registry.mint(
                session_id="same-id",
                env_name="STRIPE_SECRET_KEY",
                secret=SecretRef(name="stripe_test_key"),
                destination="api.stripe.com",
                credential_kind="stripe_bearer",
                policy_name="stripe-example",
            )
            for _ in range(2)
        ]
        adapter = DockerEgressAdapter(docker_exec=docker, proxy_host="127.0.0.1")
        first = await adapter.prepare(session_id="same-id", grants=[grants[0]], broker=broker)
        await first.close()
        second = await adapter.prepare(session_id="same-id", grants=[grants[1]], broker=broker)
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


def test_teardown_failure_is_truthful_and_retryable_after_revocation() -> None:
    docker = _FlakyCleanupDocker()

    async def run() -> tuple[VirtualCredentialRegistry, VirtualCredentialGrant, bool]:
        broker, registry, grant = _broker_with_grant()
        adapter = DockerEgressAdapter(docker_exec=docker, proxy_host="127.0.0.1")
        binding = await adapter.prepare(session_id="sess_1", grants=[grant], broker=broker)
        with pytest.raises(RuntimeError, match="docker rm: exit code 1"):
            await binding.close()
        assert binding._closed is False
        with pytest.raises(VirtualCredentialError, match="revoked"):
            registry.lookup(grant.presented_value)
        await binding.close()
        return registry, grant, binding._closed

    registry, grant, closed = asyncio.run(run())
    assert registry.was_revoked(grant.grant_id)
    assert docker.cleanup_failures == 1
    assert docker.network_absent_seen is True
    assert closed is True


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
        for _ in range(10):
            try:
                registry.lookup(second.presented_value)
            except VirtualCredentialError:
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("Second grant was not revoked before teardown wait.")
        assert close_task.done() is False

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

    with pytest.raises(UnsupportedEgressError):
        asyncio.run(run())


def test_prepare_rolls_back_when_authenticated_sidecar_never_becomes_ready() -> None:
    class _UnreadyDocker(_FakeDocker):
        async def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
            self.calls.append(list(argv))
            if argv[0] == "exec":
                return 1, "sidecar exited before readiness"
            return 0, ""

    docker = _UnreadyDocker()

    async def run() -> None:
        adapter = DockerEgressAdapter(docker_exec=docker, proxy_host="127.0.0.1")
        with pytest.raises(UnsupportedEgressError, match="docker exec"):
            await adapter.prepare(
                session_id="sess_unready",
                grants=[],
                broker=_credentialless_broker(),
            )

    asyncio.run(run())

    sidecar_run = next(argv for argv in docker.calls if argv[0] == "run")
    mounted_sources = [
        next(part.removeprefix("src=") for part in value.split(",") if part.startswith("src="))
        for value in sidecar_run
        if value.startswith("type=bind,")
    ]
    assert all(not Path(source).exists() for source in mounted_sources)
    assert any(argv[0:2] == ["rm", "-f"] for argv in docker.calls)
    assert any(argv[0:2] == ["network", "rm"] for argv in docker.calls)
