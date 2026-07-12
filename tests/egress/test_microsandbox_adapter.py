from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from cayu.egress import (
    HttpEgressPolicy,
    TransparentEgressBroker,
    UnsupportedEgressError,
    VirtualCredentialRegistry,
    VirtualEgressRunnerRequest,
)
from cayu.egress.microsandbox_adapter import MicrosandboxEgressAdapter
from cayu.runners import MicrosandboxRunner
from cayu.vaults import SecretRef, StaticVault


class _FakeAuthority:
    def ca_cert_pem(self) -> bytes:
        return b"session-ca"


class _FakeProxyServer:
    instances: list[_FakeProxyServer] = []

    def __init__(self, broker: Any, *, loop: Any, host: str) -> None:
        self.broker = broker
        self.loop = loop
        self.host = host
        self.authority = _FakeAuthority()
        self.closed = False
        self.instances.append(self)

    async def start(self) -> int:
        return 8123

    async def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class _FakeNetworkPolicy:
    default_egress: Any
    rules: tuple[Any, ...]


@dataclass(frozen=True)
class _FakeNetwork:
    policy: _FakeNetworkPolicy


class _FakeDestination:
    @staticmethod
    def group(group: Any) -> tuple[str, Any]:
        return ("group", group)


class _FakeRule:
    @staticmethod
    def allow_dns() -> tuple[str, str]:
        return ("dns-udp", "dns-tcp")

    @staticmethod
    def allow(*, destination: Any, protocol: Any, port: int) -> tuple[Any, ...]:
        return ("allow", destination, protocol, port)


class _FakePatch:
    @staticmethod
    def mkdir(path: str, *, mode: int) -> tuple[Any, ...]:
        return ("mkdir", path, mode)

    @staticmethod
    def copy_file(
        source: str,
        destination: str,
        *,
        mode: int,
        replace: bool,
    ) -> tuple[Any, ...]:
        return ("copy_file", source, destination, mode, replace)


@dataclass
class _FakeEvent:
    event_type: str
    code: int | None = None
    data: bytes | None = None


class _FakeExecOutput:
    exit_code = 0
    stdout_bytes = b""
    stderr_bytes = b""


class _FakeExecHandle:
    def __init__(self) -> None:
        self._events = [_FakeEvent("exited", code=0)]

    def __aiter__(self):
        return self

    async def __anext__(self) -> _FakeEvent:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def collect(self) -> _FakeExecOutput:
        return _FakeExecOutput()


class _FakeSandbox:
    def __init__(self, name: str) -> None:
        self.name = name
        self.exec_calls: list[dict[str, Any]] = []
        self.shell_calls: list[dict[str, Any]] = []
        self.stopped = False

    async def exec(self, cmd: str, args: list[str], **kwargs: Any) -> _FakeExecOutput:
        return _FakeExecOutput()

    async def exec_stream(self, cmd: str, args: list[str], **kwargs: Any) -> _FakeExecHandle:
        self.exec_calls.append({"cmd": cmd, "args": args, **kwargs})
        return _FakeExecHandle()

    async def shell_stream(self, script: str, **kwargs: Any) -> _FakeExecHandle:
        self.shell_calls.append({"script": script, **kwargs})
        return _FakeExecHandle()

    async def stop_and_wait(self) -> None:
        self.stopped = True


class _FakeSandboxApi:
    created: list[dict[str, Any]] = []
    removed: list[str] = []
    sandbox: _FakeSandbox | None = None

    @classmethod
    async def create(cls, name: str, **kwargs: Any) -> _FakeSandbox:
        cls.created.append({"name": name, **kwargs})
        cls.sandbox = _FakeSandbox(name)
        return cls.sandbox

    @classmethod
    async def remove(cls, name: str) -> None:
        cls.removed.append(name)


class _FakeMicrosandboxModule:
    Action = type("Action", (), {"DENY": "deny"})
    DestGroup = type("DestGroup", (), {"HOST": "host"})
    Protocol = type("Protocol", (), {"TCP": "tcp"})
    Destination = _FakeDestination
    Network = _FakeNetwork
    NetworkPolicy = _FakeNetworkPolicy
    Patch = _FakePatch
    Rule = _FakeRule
    Sandbox = _FakeSandboxApi


def _broker_and_grant() -> tuple[TransparentEgressBroker, Any]:
    registry = VirtualCredentialRegistry()
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=StaticVault({"stripe": "sk_test_real"}),
        policies={
            "stripe": HttpEgressPolicy(
                name="stripe",
                allowed_hosts=["api.stripe.com"],
                allowed_endpoints=[("GET", "/")],
            )
        },
    )
    grant = registry.mint(
        session_id="session-1",
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe"),
        destination="api.stripe.com",
        credential_kind="stripe_bearer",
        policy_name="stripe",
    )
    return broker, grant


def test_microsandbox_adapter_creates_only_a_proxy_reachable_runner(tmp_path: Path) -> None:
    async def run() -> tuple[Any, MicrosandboxRunner, str]:
        _FakeProxyServer.instances = []
        _FakeSandboxApi.created = []
        _FakeSandboxApi.removed = []
        broker, grant = _broker_and_grant()
        adapter = MicrosandboxEgressAdapter(
            bind_host="0.0.0.0",
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
        )
        binding = await adapter.prepare(
            session_id="session-1",
            grants=[grant],
            broker=broker,
        )
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(binding.ca_cert_pem or b"")
        request = VirtualEgressRunnerRequest(
            name="sandbox-1",
            runner_kind="microsandbox",
            image="python:3.13",
            binding=binding,
            env_overlay={
                **binding.env,
                "STRIPE_SECRET_KEY": grant.presented_value,
            },
            ca_cert_host_path=str(ca_path),
            guest_ca_path="/etc/cayu/ca.pem",
            setup_commands=("python3 -V",),
            egress_destinations=("api.stripe.com",),
        )
        runner = await adapter.create_runner(request)
        return binding, runner, str(ca_path)

    binding, runner, ca_path = asyncio.run(run())

    assert binding.proxy_url == "http://host.microsandbox.internal:8123"
    assert binding.env["HTTPS_PROXY"] == binding.proxy_url
    created = _FakeSandboxApi.created[0]
    network = created["network"]
    assert network.policy.default_egress == "deny"
    assert network.policy.rules == (
        "dns-udp",
        "dns-tcp",
        ("allow", ("group", "host"), "tcp", 8123),
    )
    assert created["patches"] == [
        ("mkdir", "/etc/cayu", 0o755),
        ("copy_file", ca_path, "/etc/cayu/ca.pem", 0o644, True),
    ]
    assert _FakeSandboxApi.sandbox is not None
    preflight = _FakeSandboxApi.sandbox.exec_calls[0]
    assert preflight["env"]["HTTPS_PROXY"] == binding.proxy_url
    assert preflight["env"]["STRIPE_SECRET_KEY"].startswith("sk_test_cayu_vc_")
    assert "api.stripe.com" in preflight["args"][1]
    assert "1.1.1.1" in preflight["args"][1]
    assert "169.254.169.254" in preflight["args"][1]
    assert "SSLContext(ssl.PROTOCOL_TLS_CLIENT)" in preflight["args"][1]
    assert "X-aws-ec2-metadata-token-ttl-seconds" in preflight["args"][1]
    setup = _FakeSandboxApi.sandbox.shell_calls[0]
    assert setup["script"] == "python3 -V"
    assert setup["env"]["HTTPS_PROXY"] == binding.proxy_url

    asyncio.run(runner.close())
    asyncio.run(binding.close())
    assert _FakeSandboxApi.removed == ["sandbox-1"]
    assert _FakeProxyServer.instances[0].closed is True


def test_microsandbox_adapter_rejects_mismatched_runner_kind(tmp_path: Path) -> None:
    async def run() -> None:
        broker, grant = _broker_and_grant()
        adapter = MicrosandboxEgressAdapter(
            bind_host="0.0.0.0",
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
        )
        binding = await adapter.prepare(
            session_id="session-1",
            grants=[grant],
            broker=broker,
        )
        with pytest.raises(UnsupportedEgressError, match="runner kind"):
            await adapter.create_runner(
                VirtualEgressRunnerRequest(
                    name="sandbox-1",
                    runner_kind="e2b",
                    image="python:3.13",
                    binding=binding,
                    env_overlay=binding.env,
                    ca_cert_host_path=str(tmp_path / "ca.pem"),
                    guest_ca_path="/etc/cayu/ca.pem",
                    setup_commands=(),
                    egress_destinations=("api.stripe.com",),
                )
            )
        await binding.close()

    asyncio.run(run())
