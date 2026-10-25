from __future__ import annotations

import asyncio
import json
import re
import sys
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
from cayu.egress.adapter import EgressBinding, VirtualEgressRunnerRequest
from cayu.egress.docker_adapter import (
    DEFAULT_SIDECAR_IMAGE,
    GUEST_CA_PATH,
    DockerEgressAdapter,
    _default_docker_exec,
    _run_docker_stdout,
    resolve_proxy_bind_host,
)


class _FakeDocker:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        self.calls.append(list(argv))
        return 0, ""


def test_docker_exec_cancellation_reaps_the_cli_process(monkeypatch) -> None:
    created = asyncio.Event()
    processes = []
    create_process = asyncio.create_subprocess_exec

    async def capture_process(*args, **kwargs):
        process = await create_process(*args, **kwargs)
        processes.append(process)
        created.set()
        return process

    monkeypatch.setattr("cayu.egress.docker_adapter.shutil.which", lambda _name: sys.executable)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_process)

    async def run():
        task = asyncio.create_task(_default_docker_exec(["-c", "import time; time.sleep(60)"]))
        await asyncio.wait_for(created.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert processes[0].returncode is not None

    asyncio.run(run())


@pytest.mark.parametrize("caller_cancellations", [0, 1, 2])
def test_prepare_preserves_caller_cancellation_during_probe_timeout_cleanup(
    monkeypatch, caller_cancellations
) -> None:
    import cayu.egress.docker_adapter as docker_adapter

    # Expire only once the fake CLI has been created and starts waiting.
    monkeypatch.setattr(docker_adapter, "_SIDECAR_PROBE_TIMEOUT_S", 0)
    monkeypatch.setattr(docker_adapter.shutil, "which", lambda _name: "/docker")

    async def run():
        reaping = asyncio.Event()
        release_reaping = asyncio.Event()

        class Process:
            returncode = None
            killed = False

            def kill(self):
                self.killed = True

            async def communicate(self):
                if not self.killed:
                    await asyncio.Event().wait()
                reaping.set()
                await release_reaping.wait()
                self.returncode = -9
                return b"", b""

        process = Process()

        async def create_process(*_args, **_kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

        class ProbeDocker(_FakeDocker):
            async def __call__(self, argv):
                result = await super().__call__(argv)
                if argv[-1] == "probe":
                    return await _default_docker_exec(argv)
                return result

        docker = ProbeDocker()
        broker, registry, grant = _broker_with_grant()
        decisions = []
        broker._audit = decisions.append
        adapter = DockerEgressAdapter(docker_exec=docker, proxy_host="127.0.0.1")
        task = asyncio.create_task(
            adapter.prepare(session_id="sess_1", grants=[grant], broker=broker)
        )
        try:
            await asyncio.wait_for(reaping.wait(), timeout=5)
            for _ in range(caller_cancellations):
                task.cancel("caller cancelled during CLI reaping")
                await asyncio.sleep(0)
            assert not task.done()
            assert process.returncode is None
        finally:
            release_reaping.set()
            expected_error = (
                asyncio.CancelledError if caller_cancellations else UnsupportedEgressError
            )
            with pytest.raises(expected_error):
                await asyncio.wait_for(task, timeout=5)
            await adapter.drain_preparation_cleanup()

        assert task.cancelled() is bool(caller_cancellations)
        assert process.returncode == -9
        assert registry.was_revoked(grant.grant_id)
        assert not adapter._preparation_cleanups
        assert any(argv[:2] == ["rm", "-f"] for argv in docker.calls)
        assert any(argv[:2] == ["network", "rm"] for argv in docker.calls)
        assert [decision.error_code for decision in decisions] == (
            [] if caller_cancellations else ["egress_sidecar_unreachable"]
        )

    asyncio.run(run())


@pytest.mark.parametrize("helper", [_default_docker_exec, _run_docker_stdout])
def test_docker_egress_helpers_use_bounded_operational_environment(
    monkeypatch: pytest.MonkeyPatch,
    helper,
) -> None:
    observed: dict[str, object] = {}
    provider_canaries = {
        "OPENAI_API_KEY": "provider-openai-canary-0123456789",
        "ANTHROPIC_API_KEY": "provider-anthropic-canary-0123456789",
        "CAYU_HOME": "/tmp/provider-auth-store-canary-0123456789",
    }
    for name, value in provider_canaries.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/docker.sock")
    monkeypatch.setattr("cayu.egress.docker_adapter.shutil.which", lambda _name: "/docker")

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"stdout", b"stderr"

    async def fake_create_subprocess_exec(*argv, **kwargs):
        observed.update(argv=argv, kwargs=kwargs)
        return FakeProcess()

    monkeypatch.setattr(
        "cayu.egress.docker_adapter.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    asyncio.run(helper(["info"]))

    environment = observed["kwargs"]["env"]
    assert environment["PATH"] == "/usr/local/bin:/usr/bin"
    assert environment["DOCKER_HOST"] == "unix:///tmp/docker.sock"
    assert set(provider_canaries).isdisjoint(environment)
    assert all(value not in repr(observed) for value in provider_canaries.values())


@pytest.mark.parametrize("helper", [_default_docker_exec, _run_docker_stdout])
def test_docker_egress_helpers_accept_explicit_trusted_grants(
    monkeypatch: pytest.MonkeyPatch,
    helper,
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setenv("AWS_PROFILE", "private-registry")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/run/registry/google.json")
    monkeypatch.setattr("cayu.egress.docker_adapter.shutil.which", lambda _name: "/docker")

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"stdout", b"stderr"

    async def fake_create_subprocess_exec(*argv, **kwargs):
        observed.update(argv=argv, kwargs=kwargs)
        return FakeProcess()

    monkeypatch.setattr(
        "cayu.egress.docker_adapter.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    asyncio.run(helper(["info"], docker_cli_env_allowlist=("AWS_PROFILE",)))

    environment = observed["kwargs"]["env"]
    assert environment["AWS_PROFILE"] == "private-registry"
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in environment


def test_create_runner_forwards_explicit_docker_cli_grants(monkeypatch) -> None:
    observed: dict[str, object] = {}
    expected_runner = object()

    class FakeDockerRunner:
        @classmethod
        async def create(cls, name: str, **kwargs):
            observed.update(name=name, kwargs=kwargs)
            return expected_runner

    monkeypatch.setattr("cayu.egress.docker_adapter.DockerRunner", FakeDockerRunner)
    adapter = DockerEgressAdapter(
        docker_exec=_FakeDocker(),
        proxy_host="127.0.0.1",
        docker_cli_env_allowlist=("AWS_PROFILE",),
    )
    request = VirtualEgressRunnerRequest(
        name="credential-helper",
        runner_kind="docker",
        image="private.example/cayu:latest",
        binding=EgressBinding(runner_kind="docker", network="internal"),
        env_overlay={},
        ca_cert_host_path="/tmp/ca.pem",
        guest_ca_path=GUEST_CA_PATH,
        setup_commands=(),
        egress_destinations=(),
        host_workspace_path="/workspace/candidate-one",
    )

    runner = asyncio.run(adapter.create_runner(request))

    assert runner is expected_runner
    assert observed["kwargs"]["docker_cli_env_allowlist"] == ("AWS_PROFILE",)
    assert observed["kwargs"]["mount_path"] == "/workspace/candidate-one"


@pytest.mark.parametrize("output_secret_values_present", (False, True))
def test_create_runner_forwards_runtime_owned_overlay_secret_authority(
    monkeypatch,
    output_secret_values_present: bool,
) -> None:
    observed: dict[str, object] = {}
    expected_runner = object()

    class FakeDockerRunner:
        @classmethod
        async def create(cls, name: str, **kwargs):
            observed.update(name=name, kwargs=kwargs)
            return expected_runner

    monkeypatch.setattr("cayu.egress.docker_adapter.DockerRunner", FakeDockerRunner)
    adapter = DockerEgressAdapter(
        docker_exec=_FakeDocker(),
        proxy_host="127.0.0.1",
    )
    request = VirtualEgressRunnerRequest(
        name="browser-worker",
        runner_kind="docker",
        image="cayu-browser-fetch:2",
        binding=EgressBinding(runner_kind="docker", network="internal"),
        env_overlay={"HTTPS_PROXY": "http://proxy:8080"},
        env_overlay_secret_values_present=output_secret_values_present,
        ca_cert_host_path="/tmp/ca.pem",
        guest_ca_path=GUEST_CA_PATH,
        setup_commands=(),
        egress_destinations=(),
    )

    runner = asyncio.run(adapter.create_runner(request))

    assert runner is expected_runner
    assert observed["kwargs"]["_env_overlay_secret_values_present"] is (
        output_secret_values_present
    )


@pytest.mark.parametrize("pinned", [False, True])
def test_create_runner_forwards_explicit_seccomp_profile(monkeypatch, tmp_path, pinned) -> None:
    from cayu.runners.workloads import PINNED_BROWSER_SESSION_IMAGE

    observed: dict[str, object] = {}
    expected_runner = object()
    profile = tmp_path / "chromium-seccomp.json"
    profile.write_text("{}")

    class FakeDockerRunner:
        @classmethod
        async def create(cls, name: str, **kwargs):
            observed.update(name=name, kwargs=kwargs)
            return expected_runner

    monkeypatch.setattr("cayu.egress.docker_adapter.DockerRunner", FakeDockerRunner)
    adapter = DockerEgressAdapter(
        docker_exec=_FakeDocker(),
        proxy_host="127.0.0.1",
        seccomp_profile=str(profile),
    )
    request = VirtualEgressRunnerRequest(
        name="browser-worker",
        runner_kind="docker",
        image=PINNED_BROWSER_SESSION_IMAGE if pinned else "cayu-browser-fetch:2",
        binding=EgressBinding(runner_kind="docker", network="internal"),
        env_overlay={},
        ca_cert_host_path="/tmp/ca.pem",
        guest_ca_path=GUEST_CA_PATH,
        setup_commands=(),
        egress_destinations=(),
    )

    runner = asyncio.run(adapter.create_runner(request))

    assert runner is expected_runner
    assert observed["kwargs"]["seccomp_profile"] == str(profile)


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
    # The host gets no address on it, so the guest cannot reach host services.
    assert [
        "network",
        "create",
        "--internal",
        "--opt",
        "com.docker.network.bridge.inhibit_ipv4=true",
        "--ipv6=false",
        "--label",
        label,
        network,
    ] in docker.calls
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
        DEFAULT_SIDECAR_IMAGE,
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


def test_default_sidecar_image_is_pinned_by_digest() -> None:
    assert re.fullmatch(r"alpine/socat:[0-9.]+@sha256:[0-9a-f]{64}", DEFAULT_SIDECAR_IMAGE)


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


@pytest.mark.parametrize("gateway", [None, "", "<no value>", "0.0.0.0", "secret-canary"])
def test_resolve_bind_host_requires_configuration_without_bridge_gateway(gateway) -> None:
    async def run(argv: Sequence[str]) -> tuple[int, str]:
        if argv[0] == "network" and gateway is not None:
            return 0, gateway
        return 1, ""

    with pytest.raises(UnsupportedEgressError, match="egress_proxy_host_unresolved") as error:
        asyncio.run(resolve_proxy_bind_host(run))

    assert "network=bridge" in str(error.value)
    assert "--bridge=none" in str(error.value)
    assert "proxy_host" in str(error.value)
    assert "secret-canary" not in str(error.value)


def test_prepare_with_missing_bridge_fails_before_allocating_resources() -> None:
    docker = _FakeDocker()

    async def inspect(_argv):
        return 1, "No such network: bridge"

    async def run():
        adapter = DockerEgressAdapter(docker_exec=docker, docker_run=inspect)
        with pytest.raises(UnsupportedEgressError, match="egress_proxy_host_unresolved"):
            await adapter.prepare(
                session_id="missing_bridge", grants=[], broker=_credentialless_broker()
            )

    asyncio.run(run())
    assert docker.calls == []


@pytest.mark.parametrize("failure", ["no_route", "timeout", "cancelled"])
def test_prepare_probes_broker_and_rolls_back_unreachable_sidecar(monkeypatch, failure) -> None:
    import cayu.egress.docker_adapter as docker_adapter

    # A zero deadline deterministically cancels the hanging probe at its first await.
    if failure == "timeout":
        monkeypatch.setattr(docker_adapter, "_SIDECAR_PROBE_TIMEOUT_S", 0)
    decisions = []

    class UnreachableDocker(_FakeDocker):
        async def __call__(self, argv):
            self.calls.append(list(argv))
            if argv[-1] == "probe":
                if failure == "timeout":
                    await asyncio.Event().wait()
                if failure == "cancelled":
                    raise asyncio.CancelledError()
                return 1, "cayu-broker-gateway=172.30.0.1\nNetwork unreachable TOKEN=secret-canary"
            return 0, ""

    docker = UnreachableDocker()

    async def run():
        broker, registry, grant = _broker_with_grant()
        broker._audit = decisions.append
        adapter = DockerEgressAdapter(docker_exec=docker, proxy_host="127.0.0.1")
        if failure == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await adapter.prepare(session_id="sess_1", grants=[grant], broker=broker)
        else:
            with pytest.raises(UnsupportedEgressError, match="egress_sidecar_unreachable") as error:
                await adapter.prepare(session_id="sess_1", grants=[grant], broker=broker)
            assert "network=bridge" in str(error.value)
            assert "host.docker.internal" in str(error.value)
            if failure == "no_route":
                assert "172.30.0.1" in str(error.value)
                assert "Network unreachable" in str(error.value)
            assert "secret-canary" not in str(error.value)
        assert registry.was_revoked(grant.grant_id)
        assert not adapter._preparation_cleanups

    asyncio.run(run())
    sidecar_run = next(argv for argv in docker.calls if argv[0] == "run")
    for value in sidecar_run:
        if value.startswith("type=bind,"):
            source = next(part[4:] for part in value.split(",") if part.startswith("src="))
            assert not Path(source).exists()
    assert any(argv[:2] == ["rm", "-f"] for argv in docker.calls)
    assert any(argv[:2] == ["network", "rm"] for argv in docker.calls)
    if failure == "cancelled":
        assert decisions == []
    else:
        assert len(decisions) == 1
        assert decisions[0].error_code == "egress_sidecar_unreachable"
        assert decisions[0].authorization_kind == "transport"
        assert "secret-canary" not in repr(decisions)


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


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_docker_prepare_real_task_cancellation_is_not_duplicated(
    monkeypatch: pytest.MonkeyPatch,
    cleanup_fails: bool,
) -> None:
    prepare_started = asyncio.Event()
    cleanup_error = RuntimeError("docker rollback failed")

    class _FakeAuthority:
        def ca_cert_pem(self) -> bytes:
            return b"session-ca"

    class _FakeProxyServer:
        authority = _FakeAuthority()

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def start(self) -> int:
            return 9123

        async def close(self) -> None:
            pass

    class _CancelledPrepareDocker(_FakeDocker):
        async def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
            self.calls.append(list(argv))
            if argv[:2] == ["network", "create"]:
                prepare_started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            if cleanup_fails and argv[:2] == ["rm", "-f"]:
                raise cleanup_error
            return 0, ""

    async def run() -> tuple[BaseException, bool, int]:
        import cayu.egress.docker_adapter as docker_adapter_module

        monkeypatch.setattr(
            docker_adapter_module,
            "TransparentEgressProxyServer",
            _FakeProxyServer,
        )
        adapter = DockerEgressAdapter(
            docker_exec=_CancelledPrepareDocker(),
            proxy_host="127.0.0.1",
        )
        task = asyncio.create_task(
            adapter.prepare(
                session_id="sess_cancelled_docker_prepare",
                grants=[],
                broker=_credentialless_broker(),
            )
        )
        await prepare_started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError as exc:
            return exc, task.cancelled(), task.cancelling()
        except BaseException as exc:
            return exc, task.cancelled(), task.cancelling()
        raise AssertionError("cancelled preparation unexpectedly succeeded")

    failure, task_cancelled, pending_cancellations = asyncio.run(run())

    assert pending_cancellations == 0
    if cleanup_fails:
        assert isinstance(failure, BaseExceptionGroup)
        assert isinstance(failure.exceptions[1], RuntimeError)
        assert "Docker egress teardown incomplete" in str(failure.exceptions[1])
        assert sum(isinstance(exc, asyncio.CancelledError) for exc in failure.exceptions) == 1
        assert task_cancelled is False
    else:
        assert isinstance(failure, asyncio.CancelledError)
        assert task_cancelled is True


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


@pytest.mark.parametrize("browser", [True, False])
def test_default_seccomp_is_scoped_to_pinned_browser(monkeypatch, browser):
    from cayu.runners.browser_sandbox import browser_seccomp_profile
    from cayu.runners.workloads import PINNED_BROWSER_SESSION_IMAGE

    observed = {}

    class FakeDockerRunner:
        @classmethod
        async def create(cls, name, **kwargs):
            observed.update(kwargs)
            return object()

    monkeypatch.setattr("cayu.egress.docker_adapter.DockerRunner", FakeDockerRunner)
    adapter = DockerEgressAdapter(docker_exec=_FakeDocker(), proxy_host="127.0.0.1")
    request = VirtualEgressRunnerRequest(
        name="worker",
        runner_kind="docker",
        image=PINNED_BROWSER_SESSION_IMAGE if browser else "python:3.12-slim",
        binding=EgressBinding(runner_kind="docker", network="internal"),
        env_overlay={},
        ca_cert_host_path="/tmp/ca.pem",
        guest_ca_path=GUEST_CA_PATH,
        setup_commands=(),
        egress_destinations=(),
    )
    asyncio.run(adapter.create_runner(request))
    assert observed["seccomp_profile"] == (browser_seccomp_profile() if browser else None)


def test_packaged_browser_seccomp_matches_maintained_profile():
    from pathlib import Path

    from cayu.runners.browser_sandbox import browser_seccomp_profile

    packaged = Path(browser_seccomp_profile()).read_bytes()
    assert (
        packaged
        == (
            Path(__file__).resolve().parents[2] / "examples/browser_fetch/seccomp_profile.json"
        ).read_bytes()
    )


_GUEST_ID = "d" * 64
_SIDECAR_ID = "e" * 64


def _isolated_network(**overrides: object) -> dict[str, object]:
    network: dict[str, object] = {
        "Name": "internal",
        "Internal": True,
        "EnableIPv6": False,
        "Options": {"com.docker.network.bridge.inhibit_ipv4": "true"},
        "IPAM": {"Config": [{"Subnet": "10.213.42.0/24"}]},
        "Containers": {
            _SIDECAR_ID: {"IPv4Address": "10.213.42.1/24"},
            _GUEST_ID: {"IPv4Address": "10.213.42.2/24"},
        },
    }
    network.update(overrides)
    return network


class _HostIsolationDocker:
    """Answers the network inspection and the in-namespace host probe."""

    def __init__(self, network: dict[str, object], probe: tuple[int, str]) -> None:
        self.network = network
        self.probe = probe
        self.calls: list[list[str]] = []

    async def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        self.calls.append(list(argv))
        if argv[:2] == ["network", "inspect"]:
            return 0, json.dumps(self.network)
        if argv[:1] == ["run"]:
            return self.probe
        raise AssertionError(argv)


# `docker network inspect` on Docker Engine 27.5.1 (API 1.47) for a network the
# adapter created. The engine honors the inhibit option, so the host holds no
# address on the bridge, yet IPAM still lists the gateway it reserved.
def _docker_27_network() -> dict[str, object]:
    return {
        "Internal": True,
        "EnableIPv6": False,
        "Options": {"com.docker.network.bridge.inhibit_ipv4": "true"},
        "IPAM": {"Config": [{"Subnet": "172.18.0.0/16", "Gateway": "172.18.0.1"}]},
        "Containers": {
            _SIDECAR_ID: {"IPv4Address": "172.18.0.2/16"},
            _GUEST_ID: {"IPv4Address": "172.18.0.3/16"},
        },
    }


def _hardened_runner_request() -> VirtualEgressRunnerRequest:
    return VirtualEgressRunnerRequest(
        name="hardened-tests",
        runner_kind="docker",
        image="python:3.12-slim",
        binding=EgressBinding(
            runner_kind="docker",
            network="internal",
            proxy_port=41000,
            metadata={"proxy_bind_host": "172.17.0.1"},
        ),
        env_overlay={},
        ca_cert_host_path="/tmp/ca.pem",
        guest_ca_path=GUEST_CA_PATH,
        setup_commands=(),
        egress_destinations=(),
    )


def _fake_hardened_runner_type(observed: dict[str, object], closed: list[str]):
    from cayu.runners.docker import DockerRunner

    class FakeDockerRunner(DockerRunner):
        @classmethod
        async def create(cls, name: str, **kwargs):
            observed.update(kwargs)
            return cls(
                name,
                docker_path="/usr/bin/docker",
                _container_id=_GUEST_ID,
                _workload_restrictions=kwargs["workload_restrictions"],
            )

        async def close(self) -> None:
            closed.append(self.container_reference)

    return FakeDockerRunner


def test_create_runner_hardens_the_egress_container_when_configured(monkeypatch) -> None:
    from cayu.runners.docker_workload import DockerWorkloadRestrictions

    restrictions = DockerWorkloadRestrictions(pids_limit=256)
    observed: dict[str, object] = {}
    closed: list[str] = []
    monkeypatch.setattr(
        "cayu.egress.docker_adapter.DockerRunner", _fake_hardened_runner_type(observed, closed)
    )
    docker_run = _HostIsolationDocker(_isolated_network(), (0, "isolated\n"))
    adapter = DockerEgressAdapter(
        docker_exec=_FakeDocker(),
        docker_run=docker_run,
        proxy_host="127.0.0.1",
        workload_restrictions=restrictions,
    )
    request = _hardened_runner_request()

    declared = adapter.execution_capability_evidence()
    runner = asyncio.run(adapter.create_runner(request))
    evidence = adapter.execution_capability_evidence(runner)

    assert observed["workload_restrictions"] == restrictions
    assert observed["network"] == "internal"
    assert closed == []
    # The probe runs in the guest's own network namespace, restricted, and
    # targets the broker's host address; the first address is the sidecar's.
    probe = docker_run.calls[-1]
    assert probe[probe.index("--network") + 1] == f"container:{_GUEST_ID}"
    assert probe[probe.index("--user") + 1] == "65534:65534"
    assert probe[probe.index("--cap-drop") + 1] == "ALL"
    assert probe[-1] == "172.17.0.1:41000"
    assert "10.213.42.1:41000" not in probe
    assert declared.state_for("unprivileged_guest") == "declared"
    for capability in ("guest_privilege_containment", "unprivileged_guest"):
        assert evidence.state_for(capability) == "available"
    for capability in ("deny_by_default_network", "brokered_egress"):
        assert evidence.state_for(capability) == "available"
    # A hardened container still shares the host kernel.
    assert evidence.state_for("untrusted_code_isolation") == "unsupported"
    assert evidence.state_for("host_filesystem_isolation") == "unsupported"

    # Without hardening the adapter keeps refusing privilege claims, and a
    # runner whose verified controls differ from the adapter's is not trusted.
    plain = DockerEgressAdapter(docker_exec=_FakeDocker(), proxy_host="127.0.0.1")
    assert plain.execution_capability_evidence(runner).state_for("unprivileged_guest") == (
        "unsupported"
    )
    other = DockerEgressAdapter(
        docker_exec=_FakeDocker(),
        proxy_host="127.0.0.1",
        workload_restrictions=DockerWorkloadRestrictions(),
    )
    assert other.execution_capability_evidence(runner).state_for("unprivileged_guest") == (
        "unsupported"
    )


@pytest.mark.parametrize(
    ("network", "probe", "reason"),
    [
        (
            _isolated_network(Options={}),
            (0, "isolated\n"),
            "not set up to keep the Docker host off it: it does not set "
            "com.docker.network.bridge.inhibit_ipv4=true.",
        ),
        (
            _isolated_network(Internal=False),
            (0, "isolated\n"),
            "not set up to keep the Docker host off it: it is not internal.",
        ),
        (
            _isolated_network(EnableIPv6=True),
            (0, "isolated\n"),
            "not set up to keep the Docker host off it: it has IPv6 enabled.",
        ),
        (
            _isolated_network(IPAM={"Config": ["10.213.42.0/24"]}),
            (0, "isolated\n"),
            "its IPAM config is unreadable",
        ),
        (
            _isolated_network(Containers={_SIDECAR_ID: {"IPv4Address": "10.213.42.1/24"}}),
            (0, "isolated\n"),
            "not set up to keep the Docker host off it: the guest is not attached to it.",
        ),
        (_isolated_network(), (3, "host_reachable\n"), r"\(host_reachable\)"),
        # An engine that ignores the inhibit option inspects like Docker 27 but
        # gives the host the gateway address, so the probe finds it.
        (_docker_27_network(), (3, "host_reachable\n"), r"\(host_reachable\)"),
        (_docker_27_network(), (3, "host_address_present\n"), r"\(host_address_present\)"),
        (_isolated_network(), (3, "routed_path_present\n"), r"\(routed_path_present\)"),
        (_isolated_network(), (3, "ipv6_address_present\n"), r"\(ipv6_address_present\)"),
        (_isolated_network(), (125, "unexpected\n"), r"\(probe_failed\)"),
        (_isolated_network(), (0, ""), r"\(probe_failed\)"),
        # A sidecar image without a probe tool says so instead of reading as
        # an unverified host.
        (
            _isolated_network(),
            (4, "probe_tool_missing:timeout\n"),
            r"^egress_probe_tool_missing: the sidecar image .* has no `timeout`",
        ),
        (_isolated_network(), (4, "probe_tool_missing:curl\n"), r"\(probe_failed\)"),
    ],
)
def test_create_runner_refuses_a_hardened_guest_that_may_reach_the_host(
    monkeypatch, network, probe, reason
) -> None:
    from cayu.runners.docker_workload import DockerWorkloadRestrictions

    closed: list[str] = []
    monkeypatch.setattr(
        "cayu.egress.docker_adapter.DockerRunner", _fake_hardened_runner_type({}, closed)
    )
    adapter = DockerEgressAdapter(
        docker_exec=_FakeDocker(),
        docker_run=_HostIsolationDocker(network, probe),
        proxy_host="127.0.0.1",
        workload_restrictions=DockerWorkloadRestrictions(),
    )

    with pytest.raises(UnsupportedEgressError, match=reason):
        asyncio.run(adapter.create_runner(_hardened_runner_request()))
    # The runner is never handed out; its container is removed.
    assert closed == [_GUEST_ID]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
@pytest.mark.parametrize("missing", ["timeout", "socat", "awk"])
def test_host_probe_names_a_missing_tool_before_connecting(tmp_path, missing) -> None:
    import subprocess

    from cayu.egress.docker_adapter import _HOST_ISOLATION_PROBE_SCRIPT

    tools = tmp_path / "bin"
    tools.mkdir()
    for tool in {"timeout", "socat", "awk"} - {missing}:
        # Any connect attempt would be recorded and then look unverified.
        stub = tools / tool
        stub.write_text(f"#!/bin/sh\necho {tool} ran >> {tmp_path}/ran\nexit 1\n")
        stub.chmod(0o755)

    completed = subprocess.run(
        ["/bin/sh", "-c", _HOST_ISOLATION_PROBE_SCRIPT, "probe", "10.213.42.1:41000"],
        env={"PATH": str(tools)},
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert (completed.returncode, completed.stdout) == (4, f"probe_tool_missing:{missing}\n")
    assert not (tmp_path / "ran").exists()


def test_host_probe_targets_a_free_gateway_address(monkeypatch) -> None:
    from cayu.runners.docker_workload import DockerWorkloadRestrictions

    monkeypatch.setattr(
        "cayu.egress.docker_adapter.DockerRunner", _fake_hardened_runner_type({}, [])
    )
    docker_run = _HostIsolationDocker(
        _isolated_network(Containers={_GUEST_ID: {"IPv4Address": "10.213.42.2/24"}}),
        (0, "isolated\n"),
    )
    adapter = DockerEgressAdapter(
        docker_exec=_FakeDocker(),
        docker_run=docker_run,
        proxy_host="127.0.0.1",
        workload_restrictions=DockerWorkloadRestrictions(),
    )
    asyncio.run(adapter.create_runner(_hardened_runner_request()))
    probe = docker_run.calls[-1]
    assert probe[-2:] == ["10.213.42.1:41000", "172.17.0.1:41000"]


def test_docker_27_network_with_a_listed_gateway_is_left_to_the_probe(monkeypatch) -> None:
    from cayu.runners.docker_workload import DockerWorkloadRestrictions

    closed: list[str] = []
    monkeypatch.setattr(
        "cayu.egress.docker_adapter.DockerRunner", _fake_hardened_runner_type({}, closed)
    )
    docker_run = _HostIsolationDocker(_docker_27_network(), (0, "isolated\n"))
    adapter = DockerEgressAdapter(
        docker_exec=_FakeDocker(),
        docker_run=docker_run,
        proxy_host="127.0.0.1",
        workload_restrictions=DockerWorkloadRestrictions(),
    )

    runner = asyncio.run(adapter.create_runner(_hardened_runner_request()))

    assert runner is not None
    assert closed == []
    # The listed gateway is where the host would be; the probe checks it.
    probe = docker_run.calls[-1]
    assert probe[-2:] == ["172.18.0.1:41000", "172.17.0.1:41000"]


def test_host_probe_also_targets_a_gateway_other_than_the_first_address(monkeypatch) -> None:
    from cayu.runners.docker_workload import DockerWorkloadRestrictions

    monkeypatch.setattr(
        "cayu.egress.docker_adapter.DockerRunner", _fake_hardened_runner_type({}, [])
    )
    docker_run = _HostIsolationDocker(
        _isolated_network(
            IPAM={"Config": [{"Subnet": "10.213.42.0/24", "Gateway": "10.213.42.254"}]}
        ),
        (0, "isolated\n"),
    )
    adapter = DockerEgressAdapter(
        docker_exec=_FakeDocker(),
        docker_run=docker_run,
        proxy_host="127.0.0.1",
        workload_restrictions=DockerWorkloadRestrictions(),
    )
    asyncio.run(adapter.create_runner(_hardened_runner_request()))
    probe = docker_run.calls[-1]
    # 10.213.42.1 is the sidecar's, so only the listed gateway and the broker.
    assert probe[-2:] == ["10.213.42.254:41000", "172.17.0.1:41000"]


def _run_host_probe(tmp_path: Path, socat_output: str, arp_flags: str) -> tuple[int, str]:
    """Run the probe script with stub connect tools and a fake guest /proc/net."""

    import shutil
    import subprocess

    from cayu.egress.docker_adapter import _HOST_ISOLATION_PROBE_SCRIPT

    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "timeout").write_text('#!/bin/sh\nshift\nexec "$@"\n')
    (tools / "socat").write_text(f"#!/bin/sh\necho '{socat_output}' >&2\nexit 1\n")
    for stub in ("timeout", "socat"):
        (tools / stub).chmod(0o755)
    awk = shutil.which("awk")
    assert awk is not None
    (tools / "awk").symlink_to(awk)
    # A guest on 172.18.0.0/16 with only its link route, as on Docker 27.5.1.
    proc_net = tmp_path / "net"
    proc_net.mkdir()
    (proc_net / "route").write_text(
        "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n"
        "eth0\t000012AC\t00000000\t0001\t0\t0\t0\t0000FFFF\t0\t0\t0\n"
    )
    (proc_net / "arp").write_text(
        "IP address       HW type     Flags       HW address            Mask     Device\n"
        "172.18.0.2       0x1         0x2         02:42:ac:12:00:02     *        eth0\n"
        f"172.18.0.1       0x1         {arp_flags}         00:00:00:00:00:00     *        eth0\n"
    )
    script = _HOST_ISOLATION_PROBE_SCRIPT.replace("/proc/net/", f"{proc_net}/")
    assert "/proc/" not in script
    completed = subprocess.run(
        ["/bin/sh", "-c", script, "probe", "172.18.0.1:41000"],
        env={"PATH": str(tools)},
        capture_output=True,
        text=True,
        timeout=20,
    )
    return completed.returncode, completed.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
@pytest.mark.parametrize(
    ("socat_output", "arp_flags", "expected"),
    [
        # An engine that ignores the option: the host holds the gateway address
        # and a closed port answers with a reset.
        (
            "socat[1] E connect(5, AF=2 172.18.0.1:41000, 16): Connection refused",
            "0x2",
            (3, "host_reachable\n"),
        ),
        # The same host behind a firewall that drops the probe: the connect
        # times out but the address answered ARP.
        (
            "socat[1] E connect(5, AF=2 172.18.0.1:41000, 16): Connection timed out",
            "0x2",
            (3, "host_address_present\n"),
        ),
        # Docker 27.5.1 honoring the option: nothing answers at the gateway.
        (
            "socat[1] E connect(5, AF=2 172.18.0.1:41000, 16): Connection timed out",
            "0x0",
            (0, "isolated\n"),
        ),
        (
            "socat[1] E connect(5, AF=2 172.18.0.1:41000, 16): No route to host",
            "0x0",
            (0, "isolated\n"),
        ),
    ],
)
def test_host_probe_tells_a_host_that_holds_the_gateway_from_one_that_does_not(
    tmp_path, socat_output, arp_flags, expected
) -> None:
    assert _run_host_probe(tmp_path, socat_output, arp_flags) == expected


def test_plain_egress_runner_skips_the_hardened_host_probe(monkeypatch) -> None:
    closed: list[str] = []
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        "cayu.egress.docker_adapter.DockerRunner", _fake_hardened_runner_type(observed, closed)
    )
    docker_run = _HostIsolationDocker(_isolated_network(), (3, "host_reachable\n"))
    adapter = DockerEgressAdapter(
        docker_exec=_FakeDocker(), docker_run=docker_run, proxy_host="127.0.0.1"
    )
    asyncio.run(adapter.create_runner(_hardened_runner_request()))
    assert observed["workload_restrictions"] is None
    assert docker_run.calls == []
    assert closed == []


def test_hardened_egress_refuses_reconnect_and_untyped_restrictions(tmp_path) -> None:
    from cayu.runners.docker_workload import DockerWorkloadRestrictions

    with pytest.raises(ValueError, match="reconnect does not support workload_restrictions"):
        DockerEgressAdapter(
            reconnect_state_dir=tmp_path,
            workload_restrictions=DockerWorkloadRestrictions(),
        )
    with pytest.raises(TypeError, match="DockerWorkloadRestrictions"):
        DockerEgressAdapter(workload_restrictions={"read_only_root": True})  # type: ignore[arg-type]
