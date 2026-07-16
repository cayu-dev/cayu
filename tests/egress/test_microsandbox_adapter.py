from __future__ import annotations

import asyncio
import errno
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import cayu.egress.microsandbox_adapter as adapter_module
from cayu.egress import (
    ApprovedEgressDestination,
    CapturedRequest,
    CapturedResponse,
    EgressReconnectConflictError,
    EgressReconnectError,
    EgressReconnectNotFoundError,
    HttpEgressPolicy,
    InvalidEgressReconnectMetadataError,
    TransparentEgressBroker,
    UnsupportedEgressError,
    VirtualCredentialRegistry,
    VirtualEgressRunnerRequest,
)
from cayu.egress.microsandbox_adapter import MicrosandboxEgressAdapter
from cayu.egress.proxy_exposure import MICROSANDBOX_HOST, ExposedProxy
from cayu.runners import MicrosandboxRunner
from cayu.vaults import SecretRef, StaticVault


class _FakeAuthority:
    def ca_cert_pem(self) -> bytes:
        return b"session-ca"


class _FakeProxyServer:
    instances: list[_FakeProxyServer] = []
    claimed_ports: set[int] = set()

    def __init__(self, broker: Any, *, loop: Any, host: str, port: int = 0) -> None:
        self.broker = broker
        self.loop = loop
        self.host = host
        self.requested_port = port
        self.port = port or 8123
        self.authority = _FakeAuthority()
        self.closed = False
        self.started = False
        self.instances.append(self)

    async def start(self) -> int:
        if self.port in self.claimed_ports:
            raise OSError(errno.EADDRINUSE, "address already in use")
        self.claimed_ports.add(self.port)
        self.started = True
        return self.port

    async def close(self) -> None:
        if self.started:
            self.claimed_ports.discard(self.port)
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
    def __init__(self, name: str, *, created_at: float) -> None:
        self.name = name
        self.created_at = created_at
        self.exec_calls: list[dict[str, Any]] = []
        self.shell_calls: list[dict[str, Any]] = []
        self.stopped = False
        self.detached = False
        self.fail_stop = False
        self.sentinel: bytes | None = None

    async def exec(self, cmd: str, args: list[str], **kwargs: Any) -> _FakeExecOutput:
        return _FakeExecOutput()

    async def exec_stream(self, cmd: str, args: list[str], **kwargs: Any) -> _FakeExecHandle:
        self.exec_calls.append({"cmd": cmd, "args": args, **kwargs})
        return _FakeExecHandle()

    async def shell_stream(self, script: str, **kwargs: Any) -> _FakeExecHandle:
        self.shell_calls.append({"script": script, **kwargs})
        return _FakeExecHandle()

    async def stop_and_wait(self) -> None:
        if self.fail_stop:
            raise RuntimeError("stop failed")
        self.stopped = True

    async def detach(self) -> None:
        self.detached = True


class _FakeSandboxHandle:
    def __init__(self, sandbox: _FakeSandbox) -> None:
        self.sandbox = sandbox
        self.created_at = sandbox.created_at

    async def connect(self) -> _FakeSandbox:
        return self.sandbox


class _FakeSandboxNotFoundError(RuntimeError):
    pass


class _FakeSandboxApi:
    created: list[dict[str, Any]] = []
    removed: list[str] = []
    sandbox: _FakeSandbox | None = None
    sandboxes: dict[str, _FakeSandbox] = {}
    next_created_at = 1_000.0

    @classmethod
    async def create(cls, name: str, **kwargs: Any) -> _FakeSandbox:
        cls.created.append({"name": name, **kwargs})
        cls.sandbox = _FakeSandbox(name, created_at=cls.next_created_at)
        cls.next_created_at += 1.0
        cls.sandboxes[name] = cls.sandbox
        return cls.sandbox

    @classmethod
    async def get(cls, name: str) -> _FakeSandboxHandle:
        try:
            sandbox = cls.sandboxes[name]
        except KeyError as exc:
            raise _FakeSandboxNotFoundError(name) from exc
        return _FakeSandboxHandle(sandbox)

    @classmethod
    async def remove(cls, name: str) -> None:
        cls.removed.append(name)
        cls.sandboxes.pop(name, None)


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
    SandboxNotFoundError = _FakeSandboxNotFoundError


class _FakeUpstream:
    requests: list[CapturedRequest] = []

    async def send(self, request: CapturedRequest) -> CapturedResponse:
        self.requests.append(request)
        return CapturedResponse(status_code=200, body=b'{"ok":true}')


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
        upstream=_FakeUpstream(),
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


def test_default_microsandbox_exposure_fails_closed_for_credentialless_routes() -> None:
    async def run() -> _FakeProxyServer:
        _FakeProxyServer.instances = []
        adapter = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
        )
        with pytest.raises(UnsupportedEgressError, match="session-isolated"):
            await adapter.prepare(
                session_id="session-public-docs",
                grants=[],
                broker=_credentialless_broker(),
            )
        return _FakeProxyServer.instances[0]

    server = asyncio.run(run())

    assert server.closed is True


def _reset_fakes() -> None:
    _FakeProxyServer.instances = []
    _FakeProxyServer.claimed_ports = set()
    _FakeSandboxApi.created = []
    _FakeSandboxApi.removed = []
    _FakeSandboxApi.sandbox = None
    _FakeSandboxApi.sandboxes = {}
    _FakeSandboxApi.next_created_at = 1_000.0
    _FakeUpstream.requests = []


def _request(
    *,
    binding: Any,
    grant: Any,
    ca_path: Path,
    reconnect_metadata: dict[str, Any] | None = None,
) -> VirtualEgressRunnerRequest:
    ca_path.write_bytes(binding.ca_cert_pem or b"")
    return VirtualEgressRunnerRequest(
        name="sandbox-generated-name",
        runner_kind="microsandbox",
        image="python:3.13",
        binding=binding,
        env_overlay={
            **binding.env,
            "STRIPE_SECRET_KEY": grant.presented_value,
        },
        ca_cert_host_path=str(ca_path),
        guest_ca_path="/etc/cayu/ca.pem",
        setup_commands=(),
        egress_destinations=("api.stripe.com",),
        session_id="session-1",
        environment_name="egress-env",
        reconnect_metadata=reconnect_metadata or {},
    )


def test_microsandbox_adapter_creates_only_a_proxy_reachable_runner(tmp_path: Path) -> None:
    async def run() -> tuple[Any, MicrosandboxRunner, str, MicrosandboxEgressAdapter]:
        _reset_fakes()
        broker, grant = _broker_and_grant()
        adapter = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
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
            session_id="session-1",
            environment_name="egress-env",
        )
        runner = await adapter.create_runner(request)
        return binding, runner, str(ca_path), adapter

    binding, runner, ca_path, adapter = asyncio.run(run())

    assert binding.proxy_url == "http://host.microsandbox.internal:8123"
    assert binding.env["HTTPS_PROXY"] == binding.proxy_url
    assert _FakeProxyServer.instances[0].host == "127.0.0.1"
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
    assert 'tcp_reachable("1.1.1.1", 443)' in preflight["args"][1]
    assert "X-aws-ec2-metadata-token-ttl-seconds" in preflight["args"][1]
    setup = _FakeSandboxApi.sandbox.shell_calls[0]
    assert setup["script"] == "python3 -V"
    assert setup["env"]["HTTPS_PROXY"] == binding.proxy_url

    asyncio.run(adapter.finalize_runner(runner, outcome="completed"))
    asyncio.run(binding.close())
    assert _FakeSandboxApi.removed == ["sandbox-1"]
    assert _FakeProxyServer.instances[0].closed is True


def test_microsandbox_process_restart_reuses_sandbox_with_fresh_enforcement(
    tmp_path: Path,
) -> None:
    async def run() -> tuple[dict[str, Any], str, str, _FakeSandbox]:
        _reset_fakes()
        first_broker, first_grant = _broker_and_grant()
        first_adapter = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        first_binding = await first_adapter.prepare(
            session_id="session-1",
            grants=[first_grant],
            broker=first_broker,
        )
        first_runner = await first_adapter.create_runner(
            _request(
                binding=first_binding,
                grant=first_grant,
                ca_path=tmp_path / "first-ca.pem",
            )
        )
        identity = json.loads(json.dumps(first_adapter.reconnect_metadata(first_runner)))
        old_presented_value = first_grant.presented_value
        sandbox = _FakeSandboxApi.sandboxes[identity["sandbox_name"]]
        sandbox.sentinel = b"survives-restart"

        await first_adapter.finalize_runner(first_runner, outcome="interrupted")
        await first_binding.close()
        assert sandbox.detached is True
        assert identity["sandbox_name"] in _FakeSandboxApi.sandboxes
        del first_adapter, first_binding, first_broker, first_grant, first_runner

        second_broker, second_grant = _broker_and_grant()
        second_adapter = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        second_binding = await second_adapter.prepare_reconnect(
            session_id="session-1",
            environment_name="egress-env",
            grants=[second_grant],
            broker=second_broker,
            reconnect_metadata=identity,
        )
        second_runner = await second_adapter.create_runner(
            _request(
                binding=second_binding,
                grant=second_grant,
                ca_path=tmp_path / "second-ca.pem",
                reconnect_metadata=identity,
            )
        )

        assert len(_FakeSandboxApi.created) == 1
        assert second_runner.name == identity["sandbox_name"]
        assert sandbox.sentinel == b"survives-restart"
        assert second_binding.proxy_port == identity["proxy_listener_port"]
        assert second_grant.presented_value != old_presented_value
        ca_install = sandbox.exec_calls[-2]
        assert ca_install["args"][-1] == "/etc/cayu/ca.pem"
        assert ca_install["stdin"] == b"session-ca"
        reconnect_preflight = sandbox.exec_calls[-1]
        assert reconnect_preflight["env"]["STRIPE_SECRET_KEY"] == second_grant.presented_value
        assert "1.1.1.1" in reconnect_preflight["args"][1]

        fresh_response = await second_broker.handle_request(
            CapturedRequest(
                method="GET",
                host="api.stripe.com",
                path="/",
                headers={"Authorization": f"Bearer {second_grant.presented_value}"},
            )
        )
        assert fresh_response.status_code == 200
        assert _FakeUpstream.requests[-1].headers["Authorization"] == "Bearer sk_test_real"

        stale_response = await second_broker.handle_request(
            CapturedRequest(
                method="GET",
                host="api.stripe.com",
                path="/",
                headers={"Authorization": f"Bearer {old_presented_value}"},
            )
        )
        assert stale_response.status_code == 403
        assert old_presented_value.encode() not in stale_response.body

        await second_adapter.finalize_runner(second_runner, outcome="completed")
        await second_binding.close()
        return (
            identity,
            old_presented_value,
            second_grant.presented_value,
            sandbox,
        )

    identity, old_value, new_value, sandbox = asyncio.run(run())

    assert identity["sandbox_name"] == "sandbox-generated-name"
    assert identity["proxy_listener_port"] == 8123
    assert identity["proxy_endpoint_port"] == 8123
    assert identity["sandbox_created_at"] == 1_000.0
    assert identity["owner_session_id"] == "session-1"
    assert identity["owner_environment_name"] == "egress-env"
    assert len(identity["ownership_id"]) == 32
    int(identity["ownership_id"], 16)
    assert set(identity) == {
        "sandbox_name",
        "sandbox_created_at",
        "proxy_listener_port",
        "proxy_endpoint_port",
        "ownership_id",
        "owner_session_id",
        "owner_environment_name",
    }
    assert old_value not in json.dumps(identity)
    assert new_value not in json.dumps(identity)
    assert sandbox.stopped is True
    assert _FakeSandboxApi.removed == ["sandbox-generated-name"]
    assert _FakeProxyServer.instances[-1].closed is True
    assert list((tmp_path / "claims").glob("*.json")) == []


def test_microsandbox_terminal_retry_escalates_detach_after_claim_release(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        _reset_fakes()
        broker, grant = _broker_and_grant()
        adapter = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        binding = await adapter.prepare(session_id="session-1", grants=[grant], broker=broker)
        runner = await adapter.create_runner(
            _request(binding=binding, grant=grant, ca_path=tmp_path / "ca.pem")
        )
        identity = adapter.reconnect_metadata(runner)

        await adapter.finalize_runner(runner, outcome="interrupted")
        await binding.close()
        assert identity["sandbox_name"] in _FakeSandboxApi.sandboxes

        await adapter.finalize_runner(runner, outcome="failed")
        assert identity["sandbox_name"] not in _FakeSandboxApi.sandboxes
        assert list((tmp_path / "claims").glob("*.json")) == []

    asyncio.run(run())


def test_microsandbox_failed_terminal_retry_retains_reacquired_claim(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        _reset_fakes()
        broker, grant = _broker_and_grant()
        adapter = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        binding = await adapter.prepare(session_id="session-1", grants=[grant], broker=broker)
        runner = await adapter.create_runner(
            _request(binding=binding, grant=grant, ca_path=tmp_path / "ca.pem")
        )
        identity = adapter.reconnect_metadata(runner)
        sandbox = _FakeSandboxApi.sandboxes[identity["sandbox_name"]]
        await adapter.finalize_runner(runner, outcome="interrupted")
        await binding.close()

        sandbox.fail_stop = True
        with pytest.raises(RuntimeError, match="stop failed"):
            await adapter.finalize_runner(runner, outcome="failed")

        contender = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        with pytest.raises(EgressReconnectConflictError, match="active reconnect owner"):
            await contender.prepare_reconnect(
                session_id="session-1",
                environment_name="egress-env",
                grants=[grant],
                broker=broker,
                reconnect_metadata=identity,
            )

        sandbox.fail_stop = False
        await adapter.finalize_runner(runner, outcome="failed")
        assert identity["sandbox_name"] not in _FakeSandboxApi.sandboxes
        assert list((tmp_path / "claims").glob("*.json")) == []

    asyncio.run(run())


@pytest.mark.parametrize(
    ("session_id", "environment_name", "match"),
    [
        ("other-session", "egress-env", "different session"),
        ("session-1", "other-env", "different environment"),
    ],
)
def test_microsandbox_reconnect_rejects_rewrapped_scope_before_proxy_start(
    tmp_path: Path,
    session_id: str,
    environment_name: str,
    match: str,
) -> None:
    async def run() -> None:
        _reset_fakes()
        broker, grant = _broker_and_grant()
        first = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        binding = await first.prepare(session_id="session-1", grants=[grant], broker=broker)
        runner = await first.create_runner(
            _request(binding=binding, grant=grant, ca_path=tmp_path / "ca.pem")
        )
        identity = first.reconnect_metadata(runner)
        await first.finalize_runner(runner, outcome="interrupted")
        await binding.close()
        proxy_count = len(_FakeProxyServer.instances)

        other = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        with pytest.raises(InvalidEgressReconnectMetadataError, match=match):
            await other.prepare_reconnect(
                session_id=session_id,
                environment_name=environment_name,
                grants=[grant],
                broker=broker,
                reconnect_metadata=identity,
            )
        assert len(_FakeProxyServer.instances) == proxy_count

    asyncio.run(run())


def test_microsandbox_reconnect_rejects_recreated_same_name_before_guest_mutation(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        _reset_fakes()
        broker, grant = _broker_and_grant()
        first = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        binding = await first.prepare(session_id="session-1", grants=[grant], broker=broker)
        runner = await first.create_runner(
            _request(binding=binding, grant=grant, ca_path=tmp_path / "first-ca.pem")
        )
        identity = first.reconnect_metadata(runner)
        await first.finalize_runner(runner, outcome="interrupted")
        await binding.close()

        _FakeSandboxApi.sandboxes.pop(identity["sandbox_name"])
        replacement = await _FakeSandboxApi.create(identity["sandbox_name"])
        other = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        resumed_binding = await other.prepare_reconnect(
            session_id="session-1",
            environment_name="egress-env",
            grants=[grant],
            broker=broker,
            reconnect_metadata=identity,
        )
        with pytest.raises(InvalidEgressReconnectMetadataError, match="incarnation"):
            await other.create_runner(
                _request(
                    binding=resumed_binding,
                    grant=grant,
                    ca_path=tmp_path / "second-ca.pem",
                    reconnect_metadata=identity,
                )
            )
        assert replacement.exec_calls == []
        await resumed_binding.close()

    asyncio.run(run())


def test_microsandbox_adapter_reports_bounded_reattach_timeout(tmp_path: Path) -> None:
    async def run() -> None:
        _reset_fakes()
        broker, grant = _broker_and_grant()
        first = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        binding = await first.prepare(session_id="session-1", grants=[grant], broker=broker)
        runner = await first.create_runner(
            _request(binding=binding, grant=grant, ca_path=tmp_path / "first-ca.pem")
        )
        identity = first.reconnect_metadata(runner)
        await first.finalize_runner(runner, outcome="interrupted")
        await binding.close()

        class HangingSandboxApi(_FakeSandboxApi):
            @classmethod
            async def get(cls, name: str) -> _FakeSandboxHandle:
                del name
                await asyncio.Event().wait()
                raise AssertionError

        class HangingModule(_FakeMicrosandboxModule):
            Sandbox = HangingSandboxApi

        other = MicrosandboxEgressAdapter(
            microsandbox_module=HangingModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
            reconnect_timeout_s=0.01,
        )
        resumed_binding = await other.prepare_reconnect(
            session_id="session-1",
            environment_name="egress-env",
            grants=[grant],
            broker=broker,
            reconnect_metadata=identity,
        )
        with pytest.raises(EgressReconnectError, match="timed out"):
            await other.create_runner(
                _request(
                    binding=resumed_binding,
                    grant=grant,
                    ca_path=tmp_path / "second-ca.pem",
                    reconnect_metadata=identity,
                )
            )
        await resumed_binding.close()

    asyncio.run(run())


def test_microsandbox_reconnect_rejects_missing_provider_incarnation_as_typed_error(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        _reset_fakes()
        broker, grant = _broker_and_grant()
        first = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        binding = await first.prepare(session_id="session-1", grants=[grant], broker=broker)
        runner = await first.create_runner(
            _request(binding=binding, grant=grant, ca_path=tmp_path / "first-ca.pem")
        )
        identity = first.reconnect_metadata(runner)
        sandbox = _FakeSandboxApi.sandboxes[identity["sandbox_name"]]
        await first.finalize_runner(runner, outcome="interrupted")
        await binding.close()
        exec_count = len(sandbox.exec_calls)

        class MissingIncarnationHandle(_FakeSandboxHandle):
            created_at = None

            def __init__(self, existing: _FakeSandbox) -> None:
                self.sandbox = existing

            async def connect(self) -> _FakeSandbox:
                raise AssertionError("invalid incarnation must fail before connect")

        class MissingIncarnationApi(_FakeSandboxApi):
            @classmethod
            async def get(cls, name: str) -> MissingIncarnationHandle:
                return MissingIncarnationHandle(cls.sandboxes[name])

        class MissingIncarnationModule(_FakeMicrosandboxModule):
            Sandbox = MissingIncarnationApi

        other = MicrosandboxEgressAdapter(
            microsandbox_module=MissingIncarnationModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        resumed_binding = await other.prepare_reconnect(
            session_id="session-1",
            environment_name="egress-env",
            grants=[grant],
            broker=broker,
            reconnect_metadata=identity,
        )
        with pytest.raises(InvalidEgressReconnectMetadataError, match="incarnation"):
            await other.create_runner(
                _request(
                    binding=resumed_binding,
                    grant=grant,
                    ca_path=tmp_path / "second-ca.pem",
                    reconnect_metadata=identity,
                )
            )
        assert len(sandbox.exec_calls) == exec_count
        await resumed_binding.close()

    asyncio.run(run())


def test_microsandbox_reconnect_preserves_mapped_credentialless_exposure(
    tmp_path: Path,
) -> None:
    class _MappedSessionExposure:
        async def expose(self, *, local_host: str, local_port: int) -> ExposedProxy:
            assert local_host == "127.0.0.1"
            return ExposedProxy(
                proxy_url=f"http://{MICROSANDBOX_HOST}:{local_port + 10_000}",
                credentialless_isolated=True,
            )

    def request(
        *,
        binding: Any,
        ca_path: Path,
        reconnect_metadata: dict[str, Any] | None = None,
    ) -> VirtualEgressRunnerRequest:
        ca_path.write_bytes(binding.ca_cert_pem or b"")
        return VirtualEgressRunnerRequest(
            name="credentialless-sandbox",
            runner_kind="microsandbox",
            image="python:3.13",
            binding=binding,
            env_overlay=dict(binding.env),
            ca_cert_host_path=str(ca_path),
            guest_ca_path="/etc/cayu/ca.pem",
            setup_commands=(),
            egress_destinations=("docs.example.com",),
            session_id="session-public-docs",
            environment_name="egress-env",
            reconnect_metadata=reconnect_metadata or {},
        )

    async def run() -> tuple[dict[str, Any], Any]:
        _reset_fakes()
        first_adapter = MicrosandboxEgressAdapter(
            exposure=_MappedSessionExposure(),
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        first_binding = await first_adapter.prepare(
            session_id="session-public-docs",
            grants=[],
            broker=_credentialless_broker(),
        )
        first_runner = await first_adapter.create_runner(
            request(binding=first_binding, ca_path=tmp_path / "first-ca.pem")
        )
        identity = first_adapter.reconnect_metadata(first_runner)
        sandbox = _FakeSandboxApi.sandboxes[identity["sandbox_name"]]
        await first_adapter.finalize_runner(first_runner, outcome="interrupted")
        await first_binding.close()

        second_adapter = MicrosandboxEgressAdapter(
            exposure=_MappedSessionExposure(),
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        second_binding = await second_adapter.prepare_reconnect(
            session_id="session-public-docs",
            environment_name="egress-env",
            grants=[],
            broker=_credentialless_broker(),
            reconnect_metadata=identity,
        )
        second_runner = await second_adapter.create_runner(
            request(
                binding=second_binding,
                ca_path=tmp_path / "second-ca.pem",
                reconnect_metadata=identity,
            )
        )

        assert len(_FakeSandboxApi.created) == 1
        assert second_runner.name == identity["sandbox_name"]
        assert second_binding.proxy_port == identity["proxy_listener_port"]
        assert second_binding.proxy_endpoint is not None
        assert second_binding.proxy_endpoint.port == identity["proxy_endpoint_port"]
        assert _FakeProxyServer.instances[-1].requested_port == identity["proxy_listener_port"]
        assert sandbox.exec_calls[-1]["env"]["HTTPS_PROXY"] == second_binding.proxy_url

        await second_adapter.finalize_runner(second_runner, outcome="completed")
        await second_binding.close()
        return identity, sandbox

    identity, sandbox = asyncio.run(run())

    assert identity["proxy_listener_port"] == 8123
    assert identity["proxy_endpoint_port"] == 18123
    assert set(identity) == {
        "sandbox_name",
        "sandbox_created_at",
        "proxy_listener_port",
        "proxy_endpoint_port",
        "ownership_id",
        "owner_session_id",
        "owner_environment_name",
    }
    assert sandbox.stopped is True
    assert _FakeSandboxApi.removed == ["credentialless-sandbox"]


def test_microsandbox_reconnect_rejects_concurrent_proxy_owner(tmp_path: Path) -> None:
    async def run() -> None:
        _reset_fakes()
        broker, grant = _broker_and_grant()
        first_adapter = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        binding = await first_adapter.prepare(
            session_id="session-1",
            grants=[grant],
            broker=broker,
        )
        runner = await first_adapter.create_runner(
            _request(binding=binding, grant=grant, ca_path=tmp_path / "ca.pem")
        )
        identity = first_adapter.reconnect_metadata(runner)

        child_script = """
import asyncio
import json
import sys

from cayu.egress import EgressReconnectConflictError
from cayu.egress.microsandbox_adapter import MicrosandboxEgressAdapter
from tests.egress.test_microsandbox_adapter import (
    _FakeMicrosandboxModule,
    _FakeProxyServer,
    _broker_and_grant,
)

async def run():
    broker, grant = _broker_and_grant()
    adapter = MicrosandboxEgressAdapter(
        microsandbox_module=_FakeMicrosandboxModule,
        proxy_server_factory=_FakeProxyServer,
        reconnect_state_dir=sys.argv[1],
    )
    try:
        await adapter.prepare_reconnect(
            session_id="session-1",
            environment_name="egress-env",
            grants=[grant],
            broker=broker,
            reconnect_metadata=json.loads(sys.argv[2]),
        )
    except EgressReconnectConflictError:
        return
    raise AssertionError("child process acquired a claim already held by its parent")

asyncio.run(run())
"""
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                child_script,
                str(tmp_path / "claims"),
                json.dumps(identity),
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert child.returncode == 0, child.stdout + child.stderr

        other_broker, other_grant = _broker_and_grant()
        other_adapter = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        with pytest.raises(EgressReconnectConflictError, match="active reconnect owner"):
            await other_adapter.prepare_reconnect(
                session_id="session-1",
                environment_name="egress-env",
                grants=[other_grant],
                broker=other_broker,
                reconnect_metadata=identity,
            )

        await first_adapter.finalize_runner(runner, outcome="interrupted")
        await binding.close()

    asyncio.run(run())


def test_microsandbox_reconnect_claim_is_keyed_by_attested_sandbox_identity(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        _reset_fakes()
        first_broker, first_grant = _broker_and_grant()
        first_adapter = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        first_binding = await first_adapter.prepare(
            session_id="session-1",
            grants=[first_grant],
            broker=first_broker,
        )
        first_runner = await first_adapter.create_runner(
            _request(
                binding=first_binding,
                grant=first_grant,
                ca_path=tmp_path / "first-ca.pem",
            )
        )
        identity = first_adapter.reconnect_metadata(first_runner)
        tampered = {
            **identity,
            "proxy_listener_port": identity["proxy_listener_port"] + 1,
        }
        sandbox = _FakeSandboxApi.sandboxes[identity["sandbox_name"]]
        proxy_count = len(_FakeProxyServer.instances)
        exec_count = len(sandbox.exec_calls)

        other_broker, other_grant = _broker_and_grant()
        other_adapter = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        with pytest.raises(EgressReconnectConflictError, match="active reconnect owner"):
            await other_adapter.prepare_reconnect(
                session_id="session-1",
                environment_name="egress-env",
                grants=[other_grant],
                broker=other_broker,
                reconnect_metadata=tampered,
            )
        assert len(_FakeProxyServer.instances) == proxy_count
        assert len(sandbox.exec_calls) == exec_count

        await first_adapter.finalize_runner(first_runner, outcome="interrupted")
        await first_binding.close()

        with pytest.raises(
            InvalidEgressReconnectMetadataError,
            match="durable attestation",
        ):
            await other_adapter.prepare_reconnect(
                session_id="session-1",
                environment_name="egress-env",
                grants=[other_grant],
                broker=other_broker,
                reconnect_metadata=tampered,
            )
        assert len(_FakeProxyServer.instances) == proxy_count
        assert len(sandbox.exec_calls) == exec_count

        resumed_binding = await other_adapter.prepare_reconnect(
            session_id="session-1",
            environment_name="egress-env",
            grants=[other_grant],
            broker=other_broker,
            reconnect_metadata=identity,
        )
        resumed_runner = await other_adapter.create_runner(
            _request(
                binding=resumed_binding,
                grant=other_grant,
                ca_path=tmp_path / "resumed-ca.pem",
                reconnect_metadata=identity,
            )
        )
        await other_adapter.finalize_runner(resumed_runner, outcome="completed")
        await resumed_binding.close()

    asyncio.run(run())


def test_microsandbox_reconnect_rejects_missing_sandbox_and_bad_identity(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        _reset_fakes()
        broker, grant = _broker_and_grant()
        adapter = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        with pytest.raises(InvalidEgressReconnectMetadataError, match="invalid schema"):
            await adapter.prepare_reconnect(
                session_id="session-1",
                environment_name="egress-env",
                grants=[grant],
                broker=broker,
                reconnect_metadata={"sandbox_name": "missing"},
            )

        first_binding = await adapter.prepare(
            session_id="session-1",
            grants=[grant],
            broker=broker,
        )
        first_runner = await adapter.create_runner(
            _request(
                binding=first_binding,
                grant=grant,
                ca_path=tmp_path / "initial-ca.pem",
            )
        )
        identity = adapter.reconnect_metadata(first_runner)
        await adapter.finalize_runner(first_runner, outcome="interrupted")
        await first_binding.close()
        _FakeSandboxApi.sandboxes.pop(identity["sandbox_name"])

        reconnect_adapter = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        binding = await reconnect_adapter.prepare_reconnect(
            session_id="session-1",
            environment_name="egress-env",
            grants=[grant],
            broker=broker,
            reconnect_metadata=identity,
        )
        with pytest.raises(EgressReconnectNotFoundError, match="no longer exists"):
            await reconnect_adapter.create_runner(
                _request(
                    binding=binding,
                    grant=grant,
                    ca_path=tmp_path / "missing-ca.pem",
                    reconnect_metadata=identity,
                )
            )
        await binding.close()

    asyncio.run(run())


@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
def test_microsandbox_fresh_runner_rollback_removes_attestation(
    failure_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fail_preflight(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise failure_type("preflight failed")

    monkeypatch.setattr(adapter_module, "run_enforcement_preflight", fail_preflight)

    async def run() -> None:
        _reset_fakes()
        broker, grant = _broker_and_grant()
        adapter = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        binding = await adapter.prepare(
            session_id="session-1",
            grants=[grant],
            broker=broker,
        )
        with pytest.raises(failure_type):
            await adapter.create_runner(
                _request(
                    binding=binding,
                    grant=grant,
                    ca_path=tmp_path / "ca.pem",
                )
            )
        assert list((tmp_path / "claims").glob("*.json")) != []
        await binding.close()

    asyncio.run(run())

    assert _FakeSandboxApi.removed == ["sandbox-generated-name"]
    assert list((tmp_path / "claims").glob("*.json")) == []


def test_microsandbox_partial_attestation_write_is_removed_after_runner_rollback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_write(claim: Any, identity: Any) -> None:
        del identity
        claim.attestation_path.write_bytes(b"{")
        raise OSError("disk write failed")

    monkeypatch.setattr(adapter_module._ReconnectClaim, "write_identity", fail_write)

    async def run() -> None:
        _reset_fakes()
        broker, grant = _broker_and_grant()
        adapter = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
        )
        binding = await adapter.prepare(
            session_id="session-1",
            grants=[grant],
            broker=broker,
        )
        with pytest.raises(OSError, match="disk write failed"):
            await adapter.create_runner(
                _request(
                    binding=binding,
                    grant=grant,
                    ca_path=tmp_path / "ca.pem",
                )
            )
        await binding.close()

    asyncio.run(run())

    assert _FakeSandboxApi.removed == ["sandbox-generated-name"]
    assert list((tmp_path / "claims").glob("*.json")) == []


def test_microsandbox_adapter_rejects_mismatched_runner_kind(tmp_path: Path) -> None:
    async def run() -> None:
        broker, grant = _broker_and_grant()
        adapter = MicrosandboxEgressAdapter(
            microsandbox_module=_FakeMicrosandboxModule,
            proxy_server_factory=_FakeProxyServer,
            reconnect_state_dir=tmp_path / "claims",
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


def test_microsandbox_adapter_has_no_proxy_bind_override() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'bind_host'"):
        MicrosandboxEgressAdapter(bind_host="0.0.0.0")
