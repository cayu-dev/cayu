from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

import cayu.egress.aws_lambda_microvm_adapter as adapter_module
from cayu import ExecCommand, ExecResult
from cayu.egress import (
    HttpEgressPolicy,
    TransparentEgressBroker,
    UnsupportedEgressError,
    VirtualCredentialRegistry,
    VirtualEgressRunnerRequest,
)
from cayu.egress.aws_lambda_microvm_adapter import LambdaMicroVMEgressAdapter
from cayu.egress.proxy_exposure import ExposedProxy, VpcTaskProxyExposure
from cayu.runners import Runner
from cayu.vaults import SecretRef, StaticVault


class _FakeAuthority:
    def ca_cert_pem(self) -> bytes:
        return b"session-ca"


class _FakeProxyServer:
    instances: list[_FakeProxyServer] = []

    def __init__(self, broker: Any, *, loop: Any, host: str) -> None:
        self.host = host
        self.authority = _FakeAuthority()
        self.closed = False
        self.instances.append(self)

    async def start(self) -> int:
        return 9443

    async def close(self) -> None:
        self.closed = True


class _PrivateExposure:
    def __init__(self, proxy_url: str = "http://10.0.1.20:9443") -> None:
        self.proxy_url = proxy_url
        self.calls: list[tuple[str, int]] = []
        self.closed = False

    async def expose(self, *, local_host: str, local_port: int) -> ExposedProxy:
        self.calls.append((local_host, local_port))

        async def teardown() -> None:
            self.closed = True

        return ExposedProxy(proxy_url=self.proxy_url, teardown=teardown)


class _FakeLambdaRunner(Runner):
    isolation = "lambda-microvm"
    default_cwd = "/workspace"
    created: list[dict[str, Any]] = []
    attached: list[dict[str, Any]] = []
    fail_exec = False
    fail_suspend = False
    last_instance: _FakeLambdaRunner | None = None

    def __init__(self) -> None:
        self.microvm_id = "mvm-123"
        self.endpoint = "mvm.internal"
        self.image_identifier = "image-arn"
        self.image_version = "7"
        self.region_name = "us-east-1"
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.suspended = False
        self.terminated = False

    @classmethod
    async def create(cls, image: str, **kwargs: Any) -> _FakeLambdaRunner:
        cls.created.append({"image": image, **kwargs})
        cls.last_instance = cls()
        return cls.last_instance

    @classmethod
    async def from_existing(cls, microvm_id: str, **kwargs: Any) -> _FakeLambdaRunner:
        cls.attached.append({"microvm_id": microvm_id, **kwargs})
        cls.last_instance = cls()
        return cls.last_instance

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = None,
    ) -> ExecResult:
        self.calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout_s": timeout_s,
                "stdin": stdin,
            }
        )
        if self.fail_exec:
            return ExecResult(exit_code=1, stderr="setup failed")
        return ExecResult()

    async def suspend(self) -> None:
        if self.fail_suspend:
            raise RuntimeError("suspend failed")
        self.suspended = True

    async def terminate(self) -> None:
        self.terminated = True

    async def close(self) -> None:
        self.closed = True


def _broker_and_grant() -> tuple[TransparentEgressBroker, Any]:
    registry = VirtualCredentialRegistry()
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=StaticVault({"token": "real-token"}),
        policies={
            "internal": HttpEgressPolicy(
                name="internal",
                allowed_hosts=["receiver.internal"],
                allowed_endpoints=[("POST", "/v1/actions")],
            )
        },
        require_test_mode_credentials=False,
    )
    grant = registry.mint(
        session_id="session-1",
        env_name="INTERNAL_TOKEN",
        secret=SecretRef(name="token"),
        destination="receiver.internal",
        credential_kind="opaque_bearer",
        policy_name="internal",
    )
    return broker, grant


def test_vpc_task_proxy_exposure_advertises_only_private_ipv4() -> None:
    exposure = VpcTaskProxyExposure("10.0.1.20")

    exposed = asyncio.run(exposure.expose(local_host="0.0.0.0", local_port=9443))

    assert exposed.proxy_url == "http://10.0.1.20:9443"
    with pytest.raises(ValueError, match="private IPv4"):
        VpcTaskProxyExposure("8.8.8.8")


def test_lambda_microvm_adapter_creates_private_vpc_enforced_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(adapter_module, "LambdaMicroVMRunner", _FakeLambdaRunner)
    _FakeLambdaRunner.created = []
    _FakeProxyServer.instances = []
    exposure = _PrivateExposure()
    broker, grant = _broker_and_grant()
    adapter = LambdaMicroVMEgressAdapter(
        region_name="us-east-1",
        egress_network_connector_arn="arn:aws:lambda:us-east-1:123:network-connector:nc-1",
        exposure=exposure,
        client=object(),
        ingress_network_connectors=["managed-ingress"],
        execution_role_arn="arn:aws:iam::123:role/microvm",
        proxy_server_factory=_FakeProxyServer,
        runner_options={"poll_interval_s": 0},
    )

    async def run() -> tuple[Any, _FakeLambdaRunner]:
        binding = await adapter.prepare(session_id="session-1", grants=[grant], broker=broker)
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(binding.ca_cert_pem or b"")
        runner = await adapter.create_runner(
            VirtualEgressRunnerRequest(
                name="sandbox-1",
                runner_kind="lambda-microvm",
                image="image-arn",
                binding=binding,
                env_overlay={
                    **binding.env,
                    "INTERNAL_TOKEN": grant.presented_value,
                },
                ca_cert_host_path=str(ca_path),
                guest_ca_path="/etc/cayu/ca.pem",
                setup_commands=("python3 -V",),
                egress_destinations=("receiver.internal",),
            )
        )
        return binding, runner  # type: ignore[return-value]

    binding, runner = asyncio.run(run())

    assert exposure.calls == [("0.0.0.0", 9443)]
    assert binding.proxy_url == "http://10.0.1.20:9443"
    assert _FakeLambdaRunner.created == [
        {
            "image": "image-arn",
            "region_name": "us-east-1",
            "client": adapter.client,
            "ingress_network_connectors": ["managed-ingress"],
            "egress_network_connectors": ["arn:aws:lambda:us-east-1:123:network-connector:nc-1"],
            "execution_role_arn": "arn:aws:iam::123:role/microvm",
            "close_action": "none",
            "env_overlay": {
                **binding.env,
                "INTERNAL_TOKEN": grant.presented_value,
            },
            "poll_interval_s": 0,
        }
    ]
    ca_install = runner.calls[0]
    assert ca_install["command"].argv[-1] == "/etc/cayu/ca.pem"
    assert ca_install["stdin"] == "session-ca"
    preflight = runner.calls[1]["command"].argv[-1]
    assert "receiver.internal" in preflight
    assert "1.1.1.1" in preflight
    assert "169.254.169.254" in preflight
    assert runner.calls[2]["command"] == ExecCommand.bash("python3 -V")

    asyncio.run(binding.close())
    assert exposure.closed is True
    assert _FakeProxyServer.instances[0].closed is True


def test_lambda_microvm_adapter_rejects_public_proxy_exposure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(adapter_module, "LambdaMicroVMRunner", _FakeLambdaRunner)
    broker, grant = _broker_and_grant()
    adapter = LambdaMicroVMEgressAdapter(
        region_name="us-east-1",
        egress_network_connector_arn="connector-arn",
        exposure=_PrivateExposure("http://203.0.113.10:9443"),
        client=object(),
        proxy_server_factory=_FakeProxyServer,
    )

    async def run() -> None:
        binding = await adapter.prepare(session_id="session-1", grants=[grant], broker=broker)
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(binding.ca_cert_pem or b"")
        with pytest.raises(UnsupportedEgressError, match="private IPv4"):
            await adapter.create_runner(
                VirtualEgressRunnerRequest(
                    name="sandbox-1",
                    runner_kind="lambda-microvm",
                    image="image-arn",
                    binding=binding,
                    env_overlay=binding.env,
                    ca_cert_host_path=str(ca_path),
                    guest_ca_path="/etc/cayu/ca.pem",
                    setup_commands=(),
                    egress_destinations=("receiver.internal",),
                )
            )
        await binding.close()

    asyncio.run(run())


def test_lambda_microvm_adapter_reattaches_and_maps_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(adapter_module, "LambdaMicroVMRunner", _FakeLambdaRunner)
    _FakeLambdaRunner.created = []
    _FakeLambdaRunner.attached = []
    _FakeLambdaRunner.fail_exec = False
    broker, grant = _broker_and_grant()
    adapter = LambdaMicroVMEgressAdapter(
        region_name="us-east-1",
        egress_network_connector_arn="connector-arn",
        exposure=_PrivateExposure(),
        client=object(),
        proxy_server_factory=_FakeProxyServer,
        runner_options={"poll_interval_s": 0},
    )

    async def run() -> tuple[_FakeLambdaRunner, dict[str, Any]]:
        binding = await adapter.prepare(session_id="session-1", grants=[grant], broker=broker)
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(binding.ca_cert_pem or b"")
        runner = await adapter.create_runner(
            VirtualEgressRunnerRequest(
                name="sandbox-1",
                runner_kind="lambda-microvm",
                image="image-arn",
                binding=binding,
                env_overlay=binding.env,
                ca_cert_host_path=str(ca_path),
                guest_ca_path="/etc/cayu/ca.pem",
                setup_commands=(),
                egress_destinations=("receiver.internal",),
                session_id="session-1",
                parent_session_id="parent-session",
                reconnect_metadata={
                    "microvm_id": "mvm-123",
                    "endpoint": "mvm.internal",
                    "region": "us-east-1",
                    "image_identifier": "image-arn",
                    "image_version": "7",
                    "session_id": "session-1",
                },
            )
        )
        metadata = adapter.reconnect_metadata(runner)
        await adapter.finalize_runner(runner, outcome="interrupted")
        await binding.close()
        return runner, metadata  # type: ignore[return-value]

    runner, metadata = asyncio.run(run())

    assert _FakeLambdaRunner.created == []
    assert len(_FakeLambdaRunner.attached) == 1
    attached = _FakeLambdaRunner.attached[0]
    assert attached["microvm_id"] == "mvm-123"
    assert attached["region_name"] == "us-east-1"
    assert attached["client"] is adapter.client
    assert attached["close_action"] == "none"
    assert attached["env_overlay"]["HTTPS_PROXY"] == "http://10.0.1.20:9443"
    assert attached["poll_interval_s"] == 0
    assert metadata == {
        "microvm_id": "mvm-123",
        "endpoint": "mvm.internal",
        "region": "us-east-1",
        "image_identifier": "image-arn",
        "image_version": "7",
        "session_id": "session-1",
    }
    assert runner.suspended is True
    assert runner.terminated is False
    assert runner.closed is True


def test_lambda_microvm_adapter_fork_ignores_parent_owned_reconnect_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(adapter_module, "LambdaMicroVMRunner", _FakeLambdaRunner)
    _FakeLambdaRunner.created = []
    _FakeLambdaRunner.attached = []
    broker, grant = _broker_and_grant()
    adapter = LambdaMicroVMEgressAdapter(
        region_name="us-east-1",
        egress_network_connector_arn="connector-arn",
        exposure=_PrivateExposure(),
        client=object(),
        proxy_server_factory=_FakeProxyServer,
    )

    async def run() -> _FakeLambdaRunner:
        binding = await adapter.prepare(session_id="child-session", grants=[grant], broker=broker)
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(binding.ca_cert_pem or b"")
        runner = await adapter.create_runner(
            VirtualEgressRunnerRequest(
                name="child-sandbox",
                runner_kind="lambda-microvm",
                image="image-arn",
                binding=binding,
                env_overlay=binding.env,
                ca_cert_host_path=str(ca_path),
                guest_ca_path="/etc/cayu/ca.pem",
                setup_commands=(),
                egress_destinations=("receiver.internal",),
                session_id="child-session",
                parent_session_id="parent-session",
                reconnect_metadata={
                    "microvm_id": "parent-mvm",
                    "endpoint": "parent.internal",
                    "region": "us-east-1",
                    "image_identifier": "image-arn",
                    "image_version": "7",
                    "session_id": "parent-session",
                },
            )
        )
        await binding.close()
        return runner  # type: ignore[return-value]

    runner = asyncio.run(run())

    assert len(_FakeLambdaRunner.created) == 1
    assert _FakeLambdaRunner.attached == []
    assert adapter.reconnect_metadata(runner)["session_id"] == "child-session"


def test_lambda_microvm_adapter_does_not_terminate_existing_vm_on_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(adapter_module, "LambdaMicroVMRunner", _FakeLambdaRunner)
    _FakeLambdaRunner.fail_exec = True
    broker, grant = _broker_and_grant()
    adapter = LambdaMicroVMEgressAdapter(
        region_name="us-east-1",
        egress_network_connector_arn="connector-arn",
        exposure=_PrivateExposure(),
        client=object(),
        proxy_server_factory=_FakeProxyServer,
    )

    async def run() -> _FakeLambdaRunner:
        binding = await adapter.prepare(session_id="session-1", grants=[grant], broker=broker)
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(binding.ca_cert_pem or b"")
        with pytest.raises(UnsupportedEgressError, match="install"):
            await adapter.create_runner(
                VirtualEgressRunnerRequest(
                    name="sandbox-1",
                    runner_kind="lambda-microvm",
                    image="image-arn",
                    binding=binding,
                    env_overlay=binding.env,
                    ca_cert_host_path=str(ca_path),
                    guest_ca_path="/etc/cayu/ca.pem",
                    setup_commands=(),
                    egress_destinations=("receiver.internal",),
                    reconnect_metadata={
                        "microvm_id": "mvm-123",
                        "endpoint": "mvm.internal",
                        "region": "us-east-1",
                        "image_identifier": "image-arn",
                        "image_version": "7",
                    },
                )
            )
        runner = adapter_module.LambdaMicroVMRunner.last_instance
        assert runner is not None
        await binding.close()
        return runner

    runner = asyncio.run(run())
    _FakeLambdaRunner.fail_exec = False

    assert runner.closed is True
    assert runner.terminated is False


def test_lambda_microvm_adapter_keeps_transport_open_when_suspend_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter_module, "LambdaMicroVMRunner", _FakeLambdaRunner)
    adapter = LambdaMicroVMEgressAdapter(
        region_name="us-east-1",
        egress_network_connector_arn="connector-arn",
        exposure=_PrivateExposure(),
        client=object(),
        proxy_server_factory=_FakeProxyServer,
    )
    runner = _FakeLambdaRunner()
    runner.fail_suspend = True

    with pytest.raises(RuntimeError, match="suspend failed"):
        asyncio.run(adapter.finalize_runner(runner, outcome="interrupted"))

    assert runner.closed is False
    assert runner.terminated is False
