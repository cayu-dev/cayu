from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

import cayu.egress.e2b_adapter as e2b_adapter_module
from cayu.egress import (
    ApprovedEgressDestination,
    EgressBinding,
    HttpEgressPolicy,
    TransparentEgressBroker,
    UnsupportedEgressCapabilityError,
    UnsupportedEgressError,
    VirtualCredentialError,
    VirtualCredentialRegistry,
    VirtualEgressRunnerRequest,
)
from cayu.egress.e2b_adapter import E2BEgressAdapter
from cayu.egress.proxy_exposure import ExposedProxy
from cayu.runners import E2BRunner
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
        return 9123

    async def close(self) -> None:
        self.closed = True


class _FakeExposure:
    def __init__(self, *, credentialless_isolated: bool = False) -> None:
        self.calls: list[tuple[str, int]] = []
        self.closed = False
        self.credentialless_isolated = credentialless_isolated

    async def expose(self, *, local_host: str, local_port: int) -> ExposedProxy:
        self.calls.append((local_host, local_port))

        async def teardown() -> None:
            self.closed = True

        return ExposedProxy(
            proxy_url="http://203.0.113.10:8443",
            teardown=teardown,
            credentialless_isolated=self.credentialless_isolated,
        )


class _FlakyCleanupExposure(_FakeExposure):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_calls = 0
        self.registry: Any = None
        self.presented_value: str | None = None
        self.revoked_before_release = False

    async def expose(self, *, local_host: str, local_port: int) -> ExposedProxy:
        self.calls.append((local_host, local_port))

        async def teardown() -> None:
            self.cleanup_calls += 1
            if self.registry is not None and self.presented_value is not None:
                try:
                    self.registry.lookup(self.presented_value)
                except VirtualCredentialError:
                    self.revoked_before_release = True
            if self.cleanup_calls == 1:
                raise RuntimeError("tunnel still stopping")
            self.closed = True

        return ExposedProxy(
            proxy_url="http://203.0.113.10:8443",
            teardown=teardown,
        )


class _FailingExposure:
    async def expose(self, *, local_host: str, local_port: int) -> ExposedProxy:
        raise RuntimeError("tunnel failed")


class _InvalidExposure:
    def __init__(self) -> None:
        self.closed = False

    async def expose(self, *, local_host: str, local_port: int) -> ExposedProxy:
        async def teardown() -> None:
            self.closed = True

        return ExposedProxy(
            proxy_url="http://invalid-proxy.example/unexpected-path",
            teardown=teardown,
        )


class _FakeCommandResult:
    def __init__(self, *, exit_code: int = 0, stderr: str = "") -> None:
        self.exit_code = exit_code
        self.stdout = ""
        self.stderr = stderr


class _FakeHandle:
    def __init__(self, result: _FakeCommandResult | None = None) -> None:
        self.result = result or _FakeCommandResult()

    async def wait(self) -> _FakeCommandResult:
        return self.result


class _FakeCommands:
    background_result = _FakeCommandResult()
    foreground_results: list[_FakeCommandResult] = []

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run(self, command: str, **kwargs: Any) -> Any:
        self.calls.append({"command": command, **kwargs})
        if kwargs.get("background"):
            return _FakeHandle(self.background_result)
        if self.foreground_results:
            return self.foreground_results.pop(0)
        return _FakeCommandResult()


class _FakeFiles:
    def __init__(self) -> None:
        self.writes: list[dict[str, Any]] = []

    async def write(self, path: str, data: bytes, **kwargs: Any) -> None:
        self.writes.append({"path": path, "data": data, **kwargs})


class _FakeSandbox:
    def __init__(self) -> None:
        self.sandbox_id = "e2b-sandbox-1"
        self.commands = _FakeCommands()
        self.files = _FakeFiles()
        self.killed = False

    async def kill(self) -> bool:
        self.killed = True
        return True


class _FakeAsyncSandbox:
    created: list[dict[str, Any]] = []
    sandbox: _FakeSandbox | None = None

    @classmethod
    async def create(cls, **kwargs: Any) -> _FakeSandbox:
        cls.created.append(dict(kwargs))
        cls.sandbox = _FakeSandbox()
        return cls.sandbox


class _FakeE2BModule:
    AsyncSandbox = _FakeAsyncSandbox


class _FailingKillSandbox(_FakeSandbox):
    async def kill(self) -> bool:
        raise RuntimeError("sandbox kill failed with provider diagnostics")


class _FailingKillAsyncSandbox(_FakeAsyncSandbox):
    @classmethod
    async def create(cls, **kwargs: Any) -> _FakeSandbox:
        cls.created.append(dict(kwargs))
        cls.sandbox = _FailingKillSandbox()
        return cls.sandbox


class _FailingKillE2BModule:
    AsyncSandbox = _FailingKillAsyncSandbox


class _SelfCancellingKillSandbox(_FakeSandbox):
    async def kill(self) -> bool:
        raise asyncio.CancelledError


class _SelfCancellingKillAsyncSandbox(_FakeAsyncSandbox):
    @classmethod
    async def create(cls, **kwargs: Any) -> _FakeSandbox:
        cls.created.append(dict(kwargs))
        cls.sandbox = _SelfCancellingKillSandbox()
        return cls.sandbox


class _SelfCancellingKillE2BModule:
    AsyncSandbox = _SelfCancellingKillAsyncSandbox


class _HangingKillSandbox(_FakeSandbox):
    async def kill(self) -> bool:
        await asyncio.Event().wait()
        return True


class _HangingKillAsyncSandbox(_FakeAsyncSandbox):
    @classmethod
    async def create(cls, **kwargs: Any) -> _FakeSandbox:
        cls.created.append(dict(kwargs))
        cls.sandbox = _HangingKillSandbox()
        return cls.sandbox


class _HangingKillE2BModule:
    AsyncSandbox = _HangingKillAsyncSandbox


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


def test_e2b_credentialless_prepare_requires_session_isolated_exposure() -> None:
    async def run() -> tuple[_FakeExposure, _FakeProxyServer]:
        _FakeProxyServer.instances = []
        exposure = _FakeExposure()
        adapter = E2BEgressAdapter(
            exposure=exposure,
            e2b_module=_FakeE2BModule,
            proxy_server_factory=_FakeProxyServer,
        )
        with pytest.raises(UnsupportedEgressError, match="session-isolated"):
            await adapter.prepare(
                session_id="session-public-docs",
                grants=[],
                broker=_credentialless_broker(),
            )
        return exposure, _FakeProxyServer.instances[0]

    exposure, server = asyncio.run(run())

    assert exposure.closed is True
    assert server.closed is True


def test_e2b_credentialless_prepare_accepts_declared_session_isolation() -> None:
    async def run() -> tuple[Any, _FakeExposure]:
        _FakeProxyServer.instances = []
        exposure = _FakeExposure(credentialless_isolated=True)
        adapter = E2BEgressAdapter(
            exposure=exposure,
            e2b_module=_FakeE2BModule,
            proxy_server_factory=_FakeProxyServer,
        )
        binding = await adapter.prepare(
            session_id="session-public-docs",
            grants=[],
            broker=_credentialless_broker(),
        )
        await binding.close()
        return binding, exposure

    binding, exposure = asyncio.run(run())

    assert binding.proxy_url == "http://203.0.113.10:8443"
    assert exposure.closed is True


def test_e2b_adapter_allows_only_the_exposed_cayu_proxy(tmp_path: Path) -> None:
    async def run() -> tuple[Any, E2BRunner, _FakeExposure]:
        _FakeProxyServer.instances = []
        _FakeAsyncSandbox.created = []
        _FakeCommands.background_result = _FakeCommandResult()
        _FakeCommands.foreground_results = []
        exposure = _FakeExposure()
        broker, grant = _broker_and_grant()
        adapter = E2BEgressAdapter(
            exposure=exposure,
            e2b_module=_FakeE2BModule,
            proxy_server_factory=_FakeProxyServer,
        )
        binding = await adapter.prepare(
            session_id="session-1",
            grants=[grant],
            broker=broker,
        )
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(binding.ca_cert_pem or b"")
        runner = await adapter.create_runner(
            VirtualEgressRunnerRequest(
                name="sandbox-1",
                runner_kind="e2b",
                image="base-template",
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
        )
        return binding, runner, exposure

    binding, runner, exposure = asyncio.run(run())

    assert exposure.calls == [("127.0.0.1", 9123)]
    assert binding.proxy_url == "http://203.0.113.10:8443"
    assert len(_FakeAsyncSandbox.created) == 1
    create_options = _FakeAsyncSandbox.created[0]
    assert create_options["secure"] is True
    assert create_options["allow_internet_access"] is False
    assert create_options["template"] == "base-template"
    assert create_options["network"] == {
        "allow_out": ["203.0.113.10"],
        "deny_out": ["0.0.0.0/0"],
    }
    assert len(create_options["metadata"]["cayu_guest_handoff_id"]) == 32
    assert _FakeAsyncSandbox.sandbox is not None
    assert len(_FakeAsyncSandbox.sandbox.files.writes) == 1
    staged_ca = _FakeAsyncSandbox.sandbox.files.writes[0]
    assert staged_ca["path"].startswith("/root/.cayu-handoff-")
    assert staged_ca["data"] == b"session-ca"
    assert staged_ca["user"] == "root"
    command_calls = _FakeAsyncSandbox.sandbox.commands.calls
    hardening = next(call for call in command_calls if "iptables -I OUTPUT" in call["command"])
    protected_install = next(
        call
        for call in command_calls
        if call.get("envs", {}).get("CAYU_PROTECTED_PATH") == "/etc/cayu/ca.pem"
        and call.get("user") == "root"
    )
    setup_index = next(
        index for index, call in enumerate(command_calls) if "python3 -V" in call["command"]
    )
    preflight_index = next(
        index
        for index, call in enumerate(command_calls)
        if call.get("background") and "api.stripe.com" in call["command"]
    )
    verification_indices = [
        index for index, call in enumerate(command_calls) if "sudo -n true" in call["command"]
    ]
    assert hardening["user"] == "root"
    assert protected_install["envs"]["CAYU_PROTECTED_MODE"] == "444"
    assert command_calls[preflight_index]["envs"]["HTTPS_PROXY"] == binding.proxy_url
    assert command_calls[preflight_index]["user"] == "user"
    assert '("1.1.1.1", 443, "one.one.one.one")' in command_calls[preflight_index]["command"]
    assert '("8.8.8.8", 443, "dns.google")' in command_calls[preflight_index]["command"]
    assert (
        "for host, port, server_hostname in direct_tls_probes:"
        in (command_calls[preflight_index]["command"])
    )
    assert command_calls[preflight_index]["envs"]["STRIPE_SECRET_KEY"].startswith(
        "sk_test_cayu_vc_"
    )
    assert command_calls[setup_index]["timeout"] == 300
    assert len(verification_indices) == 2
    assert verification_indices[0] < preflight_index < setup_index < verification_indices[1]
    with pytest.raises(RuntimeError, match="Raw E2B filesystem"):
        runner.filesystem()

    asyncio.run(runner.close())
    asyncio.run(binding.close())
    assert exposure.closed is True
    assert _FakeProxyServer.instances[0].closed is True


def test_e2b_teardown_failure_is_truthful_and_retryable_after_revocation() -> None:
    async def run() -> tuple[Any, Any, _FlakyCleanupExposure, bool]:
        _FakeProxyServer.instances = []
        exposure = _FlakyCleanupExposure()
        broker, grant = _broker_and_grant()
        exposure.registry = broker.registry
        exposure.presented_value = grant.presented_value
        adapter = E2BEgressAdapter(
            exposure=exposure,
            e2b_module=_FakeE2BModule,
            proxy_server_factory=_FakeProxyServer,
        )
        binding = await adapter.prepare(
            session_id="session-1",
            grants=[grant],
            broker=broker,
        )
        with pytest.raises(RuntimeError, match="proxy exposure: RuntimeError"):
            await binding.close()
        assert binding._closed is False
        with pytest.raises(VirtualCredentialError, match="revoked"):
            broker.registry.lookup(grant.presented_value)
        await binding.close()
        return broker.registry, grant, exposure, binding._closed

    registry, grant, exposure, closed = asyncio.run(run())
    assert registry.was_revoked(grant.grant_id)
    assert exposure.cleanup_calls == 2
    assert exposure.revoked_before_release is True
    assert exposure.closed is True
    assert closed is True


def test_e2b_adapter_rejects_security_options_that_could_bypass_enforcement() -> None:
    for options in (
        {"allow_internet_access": False},
        {"network": {"allow_out": ["0.0.0.0/0"]}},
        {"cleanup_timeout_s": 0.001},
        {"envs": {"REAL_SECRET": "must-not-enter-sandbox"}},
        {"ensure_default_cwd": False},
        {"exec_user": "root"},
        {"guest_user": "root"},
        {"handoff_timeout_s": 600},
        {"bootstrap": object()},
        {"guest_setup": object()},
        {"guest_probe": object()},
    ):
        with pytest.raises(ValueError, match="adapter-owned"):
            E2BEgressAdapter(
                exposure=_FakeExposure(),
                e2b_module=_FakeE2BModule,
                e2b_options=options,
            )


def test_e2b_adapter_rejects_hostname_proxy_exposure(tmp_path: Path) -> None:
    adapter = E2BEgressAdapter(
        exposure=_FakeExposure(),
        e2b_module=_FakeE2BModule,
        proxy_server_factory=_FakeProxyServer,
    )

    async def run() -> None:
        with pytest.raises(UnsupportedEgressError, match="IPv4-literal"):
            await adapter.create_runner(
                VirtualEgressRunnerRequest(
                    name="sandbox-1",
                    runner_kind="e2b",
                    image="base-template",
                    binding=EgressBinding(
                        proxy_url="http://proxy.example:8443",
                        guest_ca_path="/etc/cayu/ca.pem",
                    ),
                    env_overlay={},
                    ca_cert_host_path=str(tmp_path / "ca.pem"),
                    guest_ca_path="/etc/cayu/ca.pem",
                    setup_commands=(),
                    egress_destinations=("api.stripe.com",),
                )
            )

    asyncio.run(run())


def test_e2b_adapter_rejects_ipv6_proxy_exposure(tmp_path: Path) -> None:
    adapter = E2BEgressAdapter(
        exposure=_FakeExposure(),
        e2b_module=_FakeE2BModule,
        proxy_server_factory=_FakeProxyServer,
    )

    async def run() -> None:
        with pytest.raises(UnsupportedEgressError, match="IPv4-literal"):
            await adapter.create_runner(
                VirtualEgressRunnerRequest(
                    name="sandbox-1",
                    runner_kind="e2b",
                    image="base-template",
                    binding=EgressBinding(
                        proxy_url="http://[2001:db8::1]:8443",
                        guest_ca_path="/etc/cayu/ca.pem",
                    ),
                    env_overlay={},
                    ca_cert_host_path=str(tmp_path / "ca.pem"),
                    guest_ca_path="/etc/cayu/ca.pem",
                    setup_commands=(),
                    egress_destinations=("api.stripe.com",),
                )
            )

    asyncio.run(run())


def test_e2b_adapter_revokes_grant_when_proxy_exposure_fails() -> None:
    async def run() -> tuple[VirtualCredentialRegistry, Any]:
        _FakeProxyServer.instances = []
        broker, grant = _broker_and_grant()
        adapter = E2BEgressAdapter(
            exposure=_FailingExposure(),
            e2b_module=_FakeE2BModule,
            proxy_server_factory=_FakeProxyServer,
        )
        with pytest.raises(RuntimeError, match="tunnel failed"):
            await adapter.prepare(
                session_id="session-1",
                grants=[grant],
                broker=broker,
            )
        return broker.registry, grant

    registry, grant = asyncio.run(run())

    with pytest.raises(VirtualCredentialError, match="revoked"):
        registry.lookup(grant.presented_value)
    assert _FakeProxyServer.instances[0].closed is True


def test_e2b_prepare_rollback_is_bounded_and_reports_incomplete_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cayu.egress._remote_adapter as remote_adapter

    finish = asyncio.Event()

    class _HangingCloseProxyServer(_FakeProxyServer):
        async def close(self) -> None:
            await finish.wait()
            await super().close()

    async def run() -> BaseException:
        _FakeProxyServer.instances = []
        broker, grant = _broker_and_grant()
        adapter = E2BEgressAdapter(
            exposure=_FailingExposure(),
            e2b_module=_FakeE2BModule,
            proxy_server_factory=_HangingCloseProxyServer,
        )
        with pytest.raises(RuntimeError, match="tunnel failed") as exc_info:
            await adapter.prepare(
                session_id="session-1",
                grants=[grant],
                broker=broker,
            )
        return exc_info.value

    monkeypatch.setattr(remote_adapter, "DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS", 0.01)
    error = asyncio.run(run())

    assert error.__notes__ == ["e2b egress prepare rollback incomplete: TimeoutError."]


def test_e2b_adapter_closes_exposure_that_returns_invalid_url() -> None:
    async def run() -> _InvalidExposure:
        exposure = _InvalidExposure()
        broker, grant = _broker_and_grant()
        adapter = E2BEgressAdapter(
            exposure=exposure,
            e2b_module=_FakeE2BModule,
            proxy_server_factory=_FakeProxyServer,
        )
        with pytest.raises(UnsupportedEgressError, match="invalid HTTP proxy URL"):
            await adapter.prepare(
                session_id="session-1",
                grants=[grant],
                broker=broker,
            )
        return exposure

    exposure = asyncio.run(run())

    assert exposure.closed is True


def test_e2b_adapter_closes_sandbox_when_preflight_fails(tmp_path: Path) -> None:
    async def run() -> _FakeSandbox:
        _FakeCommands.background_result = _FakeCommandResult(
            exit_code=17,
            stderr="direct egress unexpectedly succeeded",
        )
        broker, grant = _broker_and_grant()
        adapter = E2BEgressAdapter(
            exposure=_FakeExposure(),
            e2b_module=_FakeE2BModule,
            proxy_server_factory=_FakeProxyServer,
        )
        binding = await adapter.prepare(
            session_id="session-1",
            grants=[grant],
            broker=broker,
        )
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(binding.ca_cert_pem or b"")
        with pytest.raises(UnsupportedEgressError, match="preflight"):
            await adapter.create_runner(
                VirtualEgressRunnerRequest(
                    name="sandbox-1",
                    runner_kind="e2b",
                    image="base-template",
                    binding=binding,
                    env_overlay=binding.env,
                    ca_cert_host_path=str(ca_path),
                    guest_ca_path="/etc/cayu/ca.pem",
                    setup_commands=(),
                    egress_destinations=("api.stripe.com",),
                )
            )
        await binding.close()
        assert _FakeAsyncSandbox.sandbox is not None
        return _FakeAsyncSandbox.sandbox

    sandbox = asyncio.run(run())

    assert sandbox.killed is True


def test_e2b_adapter_preserves_public_error_when_hardening_and_rollback_fail(
    tmp_path: Path,
) -> None:
    async def run() -> UnsupportedEgressError:
        _FakeCommands.background_result = _FakeCommandResult()
        _FakeCommands.foreground_results = [
            _FakeCommandResult(),
            _FakeCommandResult(exit_code=20, stderr="must not be surfaced"),
        ]
        adapter = E2BEgressAdapter(
            exposure=_FakeExposure(),
            e2b_module=_FailingKillE2BModule,
            proxy_server_factory=_FakeProxyServer,
        )
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(b"session-ca")
        with pytest.raises(UnsupportedEgressError) as exc_info:
            await adapter.create_runner(
                VirtualEgressRunnerRequest(
                    name="sandbox-1",
                    runner_kind="e2b",
                    image="base-template",
                    binding=EgressBinding(
                        proxy_url="http://203.0.113.10:8443",
                        guest_ca_path="/etc/cayu/ca.pem",
                    ),
                    env_overlay={},
                    ca_cert_host_path=str(ca_path),
                    guest_ca_path="/etc/cayu/ca.pem",
                    setup_commands=(),
                    egress_destinations=("api.stripe.com",),
                )
            )
        return exc_info.value

    error = asyncio.run(run())

    assert "guest handoff hardening" in str(error)
    assert "must not be surfaced" not in str(error)
    assert "provider diagnostics" not in str(error)
    assert isinstance(error.__cause__, ExceptionGroup)
    assert isinstance(error.__cause__.exceptions[0], Exception)
    assert isinstance(error.__cause__.exceptions[1], RuntimeError)


def test_e2b_adapter_preserves_public_error_when_rollback_self_cancels(
    tmp_path: Path,
) -> None:
    async def run() -> UnsupportedEgressError:
        _FakeCommands.background_result = _FakeCommandResult()
        _FakeCommands.foreground_results = [
            _FakeCommandResult(),
            _FakeCommandResult(exit_code=20, stderr="must not be surfaced"),
        ]
        adapter = E2BEgressAdapter(
            exposure=_FakeExposure(),
            e2b_module=_SelfCancellingKillE2BModule,
            proxy_server_factory=_FakeProxyServer,
        )
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(b"session-ca")
        with pytest.raises(UnsupportedEgressError) as exc_info:
            await adapter.create_runner(
                VirtualEgressRunnerRequest(
                    name="sandbox-1",
                    runner_kind="e2b",
                    image="base-template",
                    binding=EgressBinding(
                        proxy_url="http://203.0.113.10:8443",
                        guest_ca_path="/etc/cayu/ca.pem",
                    ),
                    env_overlay={},
                    ca_cert_host_path=str(ca_path),
                    guest_ca_path="/etc/cayu/ca.pem",
                    setup_commands=(),
                    egress_destinations=("api.stripe.com",),
                )
            )
        return exc_info.value

    error = asyncio.run(run())

    assert "guest handoff hardening" in str(error)
    assert "must not be surfaced" not in str(error)
    assert isinstance(error.__cause__, ExceptionGroup)
    assert isinstance(error.__cause__.exceptions[0], Exception)
    assert isinstance(error.__cause__.exceptions[1], RuntimeError)
    assert str(error.__cause__.exceptions[1]) == (
        "E2B guest handoff rollback cancelled without caller cancellation."
    )


def test_e2b_adapter_preserves_pending_handoff_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = UnsupportedEgressError("preflight denied")

    async def cancel_then_fail(*_args: Any, **_kwargs: Any) -> None:
        current_task = asyncio.current_task()
        assert current_task is not None
        current_task.cancel()
        raise primary

    async def run() -> tuple[BaseExceptionGroup, _FakeSandbox]:
        _FakeCommands.background_result = _FakeCommandResult()
        _FakeCommands.foreground_results = []
        monkeypatch.setattr(
            e2b_adapter_module,
            "run_enforcement_preflight",
            cancel_then_fail,
        )
        adapter = E2BEgressAdapter(
            exposure=_FakeExposure(),
            e2b_module=_FakeE2BModule,
            proxy_server_factory=_FakeProxyServer,
        )
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(b"session-ca")
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await adapter.create_runner(
                VirtualEgressRunnerRequest(
                    name="sandbox-1",
                    runner_kind="e2b",
                    image="base-template",
                    binding=EgressBinding(
                        proxy_url="http://203.0.113.10:8443",
                        guest_ca_path="/etc/cayu/ca.pem",
                    ),
                    env_overlay={},
                    ca_cert_host_path=str(ca_path),
                    guest_ca_path="/etc/cayu/ca.pem",
                    setup_commands=(),
                    egress_destinations=("api.stripe.com",),
                )
            )
        assert _FakeAsyncSandbox.sandbox is not None
        return exc_info.value, _FakeAsyncSandbox.sandbox

    error, sandbox = asyncio.run(run())

    assert error.exceptions[0] is primary
    assert isinstance(error.exceptions[1], asyncio.CancelledError)
    assert sandbox.killed is True


def test_e2b_adapter_preserves_capability_error_when_rollback_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> UnsupportedEgressCapabilityError:
        _FakeCommands.background_result = _FakeCommandResult(exit_code=42)
        _FakeCommands.foreground_results = []
        monkeypatch.setattr(
            e2b_adapter_module,
            "DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS",
            0.01,
        )
        adapter = E2BEgressAdapter(
            exposure=_FakeExposure(),
            e2b_module=_HangingKillE2BModule,
            proxy_server_factory=_FakeProxyServer,
        )
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(b"session-ca")
        with pytest.raises(UnsupportedEgressCapabilityError) as exc_info:
            await adapter.create_runner(
                VirtualEgressRunnerRequest(
                    name="sandbox-1",
                    runner_kind="e2b",
                    image="base-template",
                    binding=EgressBinding(
                        proxy_url="http://203.0.113.10:8443",
                        guest_ca_path="/etc/cayu/ca.pem",
                    ),
                    env_overlay={},
                    ca_cert_host_path=str(ca_path),
                    guest_ca_path="/etc/cayu/ca.pem",
                    setup_commands=(),
                    egress_destinations=("api.stripe.com",),
                )
            )
        return exc_info.value

    error = asyncio.run(run())

    assert error.runner_kind == "e2b"
    assert error.capability == "metadata_isolation"
    assert "link-local metadata endpoint" in error.reason
    assert "restore 'e2b' network enforcement" in error.remediation
    assert isinstance(error.__cause__, ExceptionGroup)
    assert isinstance(
        error.__cause__.exceptions[0],
        UnsupportedEgressCapabilityError,
    )
    assert isinstance(error.__cause__.exceptions[1], TimeoutError)


def test_e2b_adapter_handoff_budget_covers_every_setup_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    returned_runner = E2BRunner(_FakeSandbox(), e2b_module=_FakeE2BModule)

    async def create_hardened(**options: Any) -> E2BRunner:
        captured.update(options)
        return returned_runner

    monkeypatch.setattr(E2BRunner, "create_hardened", create_hardened)
    adapter = E2BEgressAdapter(
        exposure=_FakeExposure(),
        e2b_module=_FakeE2BModule,
        proxy_server_factory=_FakeProxyServer,
        preflight_timeout_s=180,
    )
    ca_path = tmp_path / "ca.pem"
    ca_path.write_bytes(b"session-ca")
    request = VirtualEgressRunnerRequest(
        name="sandbox-1",
        runner_kind="e2b",
        image="base-template",
        binding=EgressBinding(
            proxy_url="http://203.0.113.10:8443",
            guest_ca_path="/etc/cayu/ca.pem",
        ),
        env_overlay={},
        ca_cert_host_path=str(ca_path),
        guest_ca_path="/etc/cayu/ca.pem",
        setup_commands=("first", "second"),
        egress_destinations=("api.stripe.com",),
    )

    runner = asyncio.run(adapter.create_runner(request))

    assert runner is returned_runner
    assert captured["handoff_timeout_s"] == 900


def test_e2b_adapter_closes_sandbox_when_guest_hardening_fails(
    tmp_path: Path,
) -> None:
    async def run() -> _FakeSandbox:
        _FakeCommands.background_result = _FakeCommandResult()
        _FakeCommands.foreground_results = [
            _FakeCommandResult(),
            _FakeCommandResult(exit_code=20, stderr="must not be surfaced"),
        ]
        broker, grant = _broker_and_grant()
        adapter = E2BEgressAdapter(
            exposure=_FakeExposure(),
            e2b_module=_FakeE2BModule,
            proxy_server_factory=_FakeProxyServer,
        )
        binding = await adapter.prepare(
            session_id="session-1",
            grants=[grant],
            broker=broker,
        )
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(binding.ca_cert_pem or b"")
        with pytest.raises(UnsupportedEgressError, match="guest handoff hardening"):
            await adapter.create_runner(
                VirtualEgressRunnerRequest(
                    name="sandbox-1",
                    runner_kind="e2b",
                    image="base-template",
                    binding=binding,
                    env_overlay=binding.env,
                    ca_cert_host_path=str(ca_path),
                    guest_ca_path="/etc/cayu/ca.pem",
                    setup_commands=(),
                    egress_destinations=("api.stripe.com",),
                )
            )
        await binding.close()
        assert _FakeAsyncSandbox.sandbox is not None
        return _FakeAsyncSandbox.sandbox

    sandbox = asyncio.run(run())

    assert sandbox.killed is True
