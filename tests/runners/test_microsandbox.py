from __future__ import annotations

import asyncio
from dataclasses import dataclass
from math import inf, nan
from typing import Any

import pytest

from cayu.runners import (
    DEFAULT_MICROSANDBOX_CWD,
    DEFAULT_MICROSANDBOX_REMOVE_TIMEOUT_SECONDS,
    ExecCommand,
    MicrosandboxCleanupError,
    MicrosandboxRunner,
    MicrosandboxUnavailableError,
)


@dataclass
class FakeEvent:
    event_type: str
    data: bytes | str | None = None
    code: int | None = None


@dataclass
class FakeStdoutEvent:
    data: bytes


@dataclass
class FakeStderrEvent:
    data: bytes


@dataclass
class FakeExitedEvent:
    code: int


class FakeHandle:
    def __init__(
        self,
        events: list[Any],
        *,
        wait_result: tuple[int, bool] = (0, False),
        collect_output: Any | None = None,
        collect_error: Exception | None = None,
    ) -> None:
        self.events = list(events)
        self.wait_result = wait_result
        self.collect_output = collect_output
        self.collect_error = collect_error
        self.killed = False
        self.fail_kill = False
        self.hang_kill = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> Any:
        await asyncio.sleep(0)
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)

    async def wait(self) -> tuple[int, bool]:
        return self.wait_result

    async def collect(self) -> Any:
        if self.collect_error is not None:
            raise self.collect_error
        return self.collect_output or FakeExecOutput(exit_code=self.wait_result[0])

    async def kill(self) -> None:
        if self.hang_kill:
            await asyncio.sleep(30)
        if self.fail_kill:
            raise RuntimeError("kill failed")
        self.killed = True


@dataclass
class FakeExecOutput:
    exit_code: int = 0
    stdout_bytes: bytes = b""
    stderr_bytes: bytes = b""


@dataclass
class FakePingResult:
    latency_ms: float = 1.5


class BlockingHandle(FakeHandle):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()

    async def __anext__(self) -> FakeEvent:
        self.started.set()
        await asyncio.sleep(30)
        raise StopAsyncIteration


class FakeSandbox:
    def __init__(self, name: str) -> None:
        self.name = name
        self.exec_sync_calls: list[dict[str, Any]] = []
        self.exec_calls: list[dict[str, Any]] = []
        self.shell_calls: list[dict[str, Any]] = []
        self.stop_calls = 0
        self.stop_and_wait_calls = 0
        self.detach_calls = 0
        self.kill_calls = 0
        self.fail_next_exec = False
        self.cancel_next_exec = False
        self.cancel_next_stream = False
        self.timeout_next_stream = False
        self.stream_error: Exception | None = None
        self.fail_kill = False
        self.hang_kill = False
        self.fail_next_stop = False
        self.fail_repeated_stop = False
        self.already_stopped = False
        self.stop_failure: Exception | None = None
        self.ping_calls = 0
        self.ping_error: Exception | None = None
        self.hang_ping = False
        self.next_handle = FakeHandle(
            [
                FakeEvent("stdout", b"hello "),
                FakeEvent("stdout", "world"),
                FakeEvent("stderr", b"warn"),
                FakeEvent("exited", code=7),
            ],
            wait_result=(7, False),
        )

    async def exec(self, cmd: str, args: list[str], **kwargs: Any) -> FakeExecOutput:
        self.exec_sync_calls.append({"cmd": cmd, "args": args, **kwargs})
        if self.cancel_next_exec:
            self.cancel_next_exec = False
            raise asyncio.CancelledError
        if self.fail_next_exec:
            self.fail_next_exec = False
            raise RuntimeError("exec failed")
        return FakeExecOutput()

    async def exec_stream(self, cmd: str, args: list[str], **kwargs: Any) -> FakeHandle:
        self.exec_calls.append({"cmd": cmd, "args": args, **kwargs})
        if self.stream_error is not None:
            raise self.stream_error
        if self.cancel_next_stream:
            self.cancel_next_stream = False
            raise asyncio.CancelledError
        if self.timeout_next_stream:
            self.timeout_next_stream = False
            raise TimeoutError
        return self.next_handle

    async def shell_stream(self, script: str, **kwargs: Any) -> FakeHandle:
        self.shell_calls.append({"script": script, **kwargs})
        if self.cancel_next_stream:
            self.cancel_next_stream = False
            raise asyncio.CancelledError
        if self.timeout_next_stream:
            self.timeout_next_stream = False
            raise TimeoutError
        return self.next_handle

    async def stop(self) -> None:
        self.stop_calls += 1

    async def stop_and_wait(self) -> None:
        self.stop_and_wait_calls += 1
        if self.stop_failure is not None:
            error = self.stop_failure
            self.stop_failure = None
            raise error
        if self.already_stopped:
            raise FakeSandboxNotRunningError("sandbox is not running")
        if self.fail_next_stop:
            self.fail_next_stop = False
            raise RuntimeError("stop failed")
        if self.fail_repeated_stop and self.stop_and_wait_calls > 1:
            raise FakeSandboxNotRunningError("sandbox is not running")

    async def ping(self) -> FakePingResult:
        self.ping_calls += 1
        if self.hang_ping:
            await asyncio.sleep(30)
        if self.ping_error is not None:
            raise self.ping_error
        return FakePingResult()

    async def kill(self) -> None:
        if self.hang_kill:
            await asyncio.sleep(30)
        if self.fail_kill:
            raise RuntimeError("sandbox kill failed")
        self.kill_calls += 1

    async def detach(self) -> None:
        self.detach_calls += 1


class FakeHandleRecord:
    def __init__(
        self,
        sandbox: FakeSandbox,
        *,
        status: str = "stopped",
        hang_refresh: bool = False,
    ) -> None:
        self.sandbox = sandbox
        self.status = status
        self.hang_refresh = hang_refresh
        self.refresh_calls = 0

    async def connect(self) -> FakeSandbox:
        return self.sandbox

    async def refresh(self) -> FakeHandleRecord:
        self.refresh_calls += 1
        if self.hang_refresh:
            await asyncio.sleep(30)
        return self


class FakeSandboxStillRunningError(RuntimeError):
    pass


class FakeSandboxNotFoundError(RuntimeError):
    pass


class FakeSandboxNotRunningError(RuntimeError):
    pass


class FakeSandboxApi:
    created: list[dict[str, Any]] = []
    removed: list[str] = []
    existing: FakeSandbox | None = None
    fail_next_remove = False
    remove_failures: list[Exception] = []
    always_still_running = False
    hang_get = False
    hang_refresh = False
    hang_remove = False
    remove_calls: list[str] = []
    statuses: list[str] = []
    fail_created_setup = False
    cancel_created_setup = False
    created_stop_failure: Exception | None = None
    registry_status = "running"
    get_error: Exception | None = None

    @classmethod
    async def create(cls, name: str, **kwargs: Any) -> FakeSandbox:
        cls.created.append({"name": name, **kwargs})
        sandbox = FakeSandbox(name)
        sandbox.fail_next_exec = cls.fail_created_setup
        sandbox.cancel_next_exec = cls.cancel_created_setup
        sandbox.stop_failure = cls.created_stop_failure
        cls.existing = sandbox
        return sandbox

    @classmethod
    async def get(cls, name: str) -> FakeHandleRecord:
        if cls.hang_get:
            await asyncio.sleep(30)
        if cls.get_error is not None:
            raise cls.get_error
        sandbox = cls.existing or FakeSandbox(name)
        cls.existing = sandbox
        status = cls.statuses.pop(0) if cls.statuses else cls.registry_status
        return FakeHandleRecord(sandbox, status=status, hang_refresh=cls.hang_refresh)

    @classmethod
    async def remove(cls, name: str) -> None:
        cls.remove_calls.append(name)
        if cls.hang_remove:
            await asyncio.sleep(30)
        if cls.fail_next_remove:
            cls.fail_next_remove = False
            raise RuntimeError("remove failed")
        if cls.remove_failures:
            raise cls.remove_failures.pop(0)
        if cls.always_still_running:
            raise FakeSandboxStillRunningError("sandbox status has not settled")
        cls.removed.append(name)


class FakeMicrosandboxError(RuntimeError):
    pass


@dataclass(frozen=True)
class FakeNetwork:
    policy: str

    @classmethod
    def none(cls) -> FakeNetwork:
        return cls(policy="none")


class FakeMicrosandboxModule:
    Network = FakeNetwork
    Sandbox = FakeSandboxApi
    SandboxNotFoundError = FakeSandboxNotFoundError
    SandboxNotRunningError = FakeSandboxNotRunningError
    SandboxStillRunningError = FakeSandboxStillRunningError
    MicrosandboxError = FakeMicrosandboxError


class FailingNetwork:
    @classmethod
    def none(cls) -> FakeNetwork:
        raise RuntimeError("default network policy unavailable")


class FailingNetworkMicrosandboxModule(FakeMicrosandboxModule):
    Network = FailingNetwork


class MalformedNetwork:
    @classmethod
    def none(cls) -> None:
        return None


class MalformedNetworkMicrosandboxModule(FakeMicrosandboxModule):
    Network = MalformedNetwork


class UnsupportedNetwork:
    pass


class UnsupportedNetworkMicrosandboxModule(FakeMicrosandboxModule):
    Network = UnsupportedNetwork


def reset_fake_module() -> None:
    FakeSandboxApi.created = []
    FakeSandboxApi.removed = []
    FakeSandboxApi.existing = None
    FakeSandboxApi.fail_next_remove = False
    FakeSandboxApi.remove_failures = []
    FakeSandboxApi.always_still_running = False
    FakeSandboxApi.hang_get = False
    FakeSandboxApi.hang_refresh = False
    FakeSandboxApi.hang_remove = False
    FakeSandboxApi.remove_calls = []
    FakeSandboxApi.statuses = []
    FakeSandboxApi.fail_created_setup = False
    FakeSandboxApi.cancel_created_setup = False
    FakeSandboxApi.created_stop_failure = None
    FakeSandboxApi.registry_status = "running"
    FakeSandboxApi.get_error = None


def cleanup_diagnostic(error: BaseException) -> dict[str, Any]:
    diagnostic = error.__dict__.get("diagnostic")
    assert isinstance(diagnostic, dict)
    return diagnostic


def test_microsandbox_runner_create_passes_lifecycle_options() -> None:
    network = {"policy": "none"}

    async def run() -> MicrosandboxRunner:
        reset_fake_module()
        runner = await MicrosandboxRunner.create(
            "agent-session",
            image="python:3.13",
            liveness_timeout_s=2.5,
            replace=True,
            cpus=2,
            network=network,
            sandbox_module=FakeMicrosandboxModule,
        )
        return runner

    runner = asyncio.run(run())

    assert runner.name == "agent-session"
    assert runner.default_cwd == DEFAULT_MICROSANDBOX_CWD
    assert runner.close_action == "remove"
    assert runner.liveness_timeout_s == 2.5
    assert runner._sandbox.exec_sync_calls == [
        {"cmd": "mkdir", "args": ["-p", "/workspace"], "cwd": "/"}
    ]
    assert FakeSandboxApi.created == [
        {
            "name": "agent-session",
            "image": "python:3.13",
            "replace": True,
            "cpus": 2,
            "network": {"policy": "none"},
        }
    ]
    assert FakeSandboxApi.created[0]["network"] is network


def test_microsandbox_runner_create_denies_network_by_default() -> None:
    async def run() -> MicrosandboxRunner:
        reset_fake_module()
        return await MicrosandboxRunner.create(
            "agent-session",
            sandbox_module=FakeMicrosandboxModule,
        )

    runner = asyncio.run(run())

    assert FakeSandboxApi.created == [
        {
            "name": "agent-session",
            "image": "python:3.13",
            "network": FakeNetwork(policy="none"),
        }
    ]
    assert runner._sandbox.exec_sync_calls == [
        {"cmd": "mkdir", "args": ["-p", "/workspace"], "cwd": "/"}
    ]


@pytest.mark.parametrize(
    ("module", "error_type", "match"),
    [
        (
            FailingNetworkMicrosandboxModule,
            RuntimeError,
            "default network policy unavailable",
        ),
        (
            MalformedNetworkMicrosandboxModule,
            TypeError,
            "returned an invalid network policy",
        ),
        (
            UnsupportedNetworkMicrosandboxModule,
            RuntimeError,
            "does not provide Network.none",
        ),
    ],
    ids=("construction-failure", "malformed-policy", "unsupported-sdk"),
)
def test_microsandbox_runner_create_fails_closed_for_invalid_default_network(
    module: type[FakeMicrosandboxModule],
    error_type: type[Exception],
    match: str,
) -> None:
    async def run() -> None:
        reset_fake_module()
        with pytest.raises(error_type, match=match):
            await MicrosandboxRunner.create(
                "agent-session",
                sandbox_module=module,
            )

    asyncio.run(run())

    assert FakeSandboxApi.created == []


def test_microsandbox_runner_executes_process_with_explicit_env_and_bounds_output(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CAYU_SECRET_HOST_ENV", "hidden")
    sandbox = FakeSandbox("runner")
    sandbox.next_handle = FakeHandle(
        [
            FakeEvent("stdout", b"abcdef"),
            FakeEvent("stderr", b"uvwxyz"),
            FakeEvent("exited", code=3),
        ],
        wait_result=(3, False),
    )
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        default_cwd="/workspace",
        sandbox_module=FakeMicrosandboxModule,
    )

    result = asyncio.run(
        runner.exec(
            ExecCommand.process("python", "-c", "print(1)"),
            cwd="/workspace/src",
            env={"VISIBLE": "1"},
            timeout_s=5,
            stdin="input",
            output_limit_bytes=3,
        )
    )

    assert result.exit_code == 3
    assert result.stdout == "abc"
    assert result.stderr == "uvw"
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True
    assert result.stdout_bytes == 6
    assert result.stderr_bytes == 6
    assert sandbox.exec_calls == [
        {
            "cmd": "python",
            "args": ["-c", "print(1)"],
            "cwd": "/workspace/src",
            "env": {"VISIBLE": "1"},
            "timeout": 5.0,
            "stdin": b"input",
        }
    ]
    assert "CAYU_SECRET_HOST_ENV" not in sandbox.exec_calls[0]["env"]
    assert sandbox.ping_calls == 0


def test_microsandbox_runner_returns_signal_nine_when_agent_ping_succeeds() -> None:
    sandbox = FakeSandbox("runner")
    sandbox.next_handle = FakeHandle(
        [FakeEvent("exited", code=-9)],
        wait_result=(-9, False),
    )
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        sandbox_module=FakeMicrosandboxModule,
    )

    result = asyncio.run(runner.exec(ExecCommand.process("memory-heavy-command")))

    assert result.exit_code == -9
    assert result.timed_out is False
    assert result.artifacts == []
    assert sandbox.ping_calls == 1


def test_microsandbox_runner_latches_typed_unavailable_state_after_failed_ping() -> None:
    async def run() -> tuple[MicrosandboxUnavailableError, MicrosandboxUnavailableError]:
        reset_fake_module()
        sandbox = FakeSandbox("dead-agent")
        sandbox.next_handle = FakeHandle(
            [FakeEvent("exited", code=-9)],
            wait_result=(-9, False),
        )
        sandbox.ping_error = ConnectionResetError("agent connection reset")
        runner = MicrosandboxRunner(
            sandbox,
            name="dead-agent",
            sandbox_module=FakeMicrosandboxModule,
        )
        with pytest.raises(MicrosandboxUnavailableError) as first_info:
            await runner.exec(ExecCommand.process("memory-heavy-command"))
        with pytest.raises(MicrosandboxUnavailableError) as second_info:
            await runner.exec(ExecCommand.process("must-not-launch"))
        assert len(sandbox.exec_calls) == 1
        assert sandbox.ping_calls == 1
        return first_info.value, second_info.value

    first, second = asyncio.run(run())

    expected_diagnostic = {
        "type": "cayu.runner_unavailable.v1",
        "adapter": "microsandbox",
        "sandbox_name": "dead-agent",
        "status": "unavailable",
        "reason": "guest_agent_unavailable_after_signal_9",
        "last_command": {
            "exit_code": -9,
            "timed_out": False,
            "cancelled": False,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
            "error_type": None,
            "error": None,
        },
        "probe": {
            "method": "Sandbox.ping",
            "status": "failed",
            "timeout_s": 1.0,
            "registry_status": "running",
            "error_type": "ConnectionResetError",
            "error": "agent connection reset",
            "status_error_type": None,
            "status_error": None,
        },
        "remediation": "Reconnect to or replace the Microsandbox before executing more commands.",
    }
    assert first.diagnostic == expected_diagnostic
    assert first.artifacts == [expected_diagnostic]
    assert isinstance(first.__cause__, ConnectionResetError)
    assert second.diagnostic == expected_diagnostic
    assert second.__cause__ is None


def test_microsandbox_runner_blocks_concurrent_launch_during_unavailability_probe() -> None:
    async def run() -> None:
        reset_fake_module()
        sandbox = FakeSandbox("dead-agent")
        sandbox.next_handle = FakeHandle(
            [FakeEvent("exited", code=-9)],
            wait_result=(-9, False),
        )
        ping_started = asyncio.Event()
        release_ping = asyncio.Event()

        async def fail_ping_after_release() -> FakePingResult:
            sandbox.ping_calls += 1
            ping_started.set()
            await release_ping.wait()
            raise ConnectionResetError("agent connection reset")

        sandbox.ping = fail_ping_after_release  # type: ignore[method-assign]
        runner = MicrosandboxRunner(
            sandbox,
            name="dead-agent",
            sandbox_module=FakeMicrosandboxModule,
        )

        first = asyncio.create_task(runner.exec(ExecCommand.process("first-command")))
        await asyncio.wait_for(ping_started.wait(), timeout=1.0)
        second = asyncio.create_task(runner.exec(ExecCommand.process("must-not-launch")))
        await asyncio.sleep(0)

        assert len(sandbox.exec_calls) == 1
        assert second.done() is False

        release_ping.set()
        with pytest.raises(MicrosandboxUnavailableError):
            await first
        with pytest.raises(MicrosandboxUnavailableError):
            await second
        assert len(sandbox.exec_calls) == 1

    asyncio.run(run())


def test_microsandbox_runner_bounds_guest_agent_ping() -> None:
    async def run() -> MicrosandboxUnavailableError:
        reset_fake_module()
        sandbox = FakeSandbox("hung-agent")
        sandbox.next_handle = FakeHandle(
            [FakeEvent("exited", code=-9)],
            wait_result=(-9, False),
        )
        sandbox.hang_ping = True
        runner = MicrosandboxRunner(
            sandbox,
            name="hung-agent",
            liveness_timeout_s=0.01,
            sandbox_module=FakeMicrosandboxModule,
        )
        with pytest.raises(MicrosandboxUnavailableError) as exc_info:
            await runner.exec(ExecCommand.process("memory-heavy-command"))
        return exc_info.value

    error = asyncio.run(run())

    assert error.probe_status == "timed_out"
    assert error.diagnostic["probe"] == {
        "method": "Sandbox.ping",
        "status": "timed_out",
        "timeout_s": 0.01,
        "registry_status": None,
        "error_type": "TimeoutError",
        "error": "Microsandbox guest-agent liveness probe timed out.",
        "status_error_type": None,
        "status_error": None,
    }
    assert isinstance(error.__cause__, TimeoutError)


def test_microsandbox_runner_classifies_no_exit_event_when_agent_ping_fails() -> None:
    async def run() -> MicrosandboxUnavailableError:
        reset_fake_module()
        sandbox = FakeSandbox("dead-agent")
        command_error = FakeMicrosandboxError(
            "runtime error: exec session ended without exit event"
        )
        sandbox.next_handle = FakeHandle([], collect_error=command_error)
        sandbox.ping_error = ConnectionResetError("agent connection reset")
        FakeSandboxApi.registry_status = "stopped"
        runner = MicrosandboxRunner(
            sandbox,
            name="dead-agent",
            sandbox_module=FakeMicrosandboxModule,
        )
        with pytest.raises(MicrosandboxUnavailableError) as exc_info:
            await runner.exec(ExecCommand.process("sleep", "30"))
        return exc_info.value

    error = asyncio.run(run())

    assert error.last_command == {
        "exit_code": None,
        "timed_out": False,
        "cancelled": False,
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "error_type": "FakeMicrosandboxError",
        "error": "runtime error: exec session ended without exit event",
    }
    assert error.probe["registry_status"] == "stopped"
    assert error.probe["status"] == "failed"
    assert isinstance(error.__cause__, ConnectionResetError)


def test_microsandbox_runner_preserves_no_exit_event_when_agent_ping_succeeds() -> None:
    sandbox = FakeSandbox("live-agent")
    command_error = FakeMicrosandboxError("runtime error: exec session ended without exit event")
    sandbox.next_handle = FakeHandle([], collect_error=command_error)
    runner = MicrosandboxRunner(
        sandbox,
        name="live-agent",
        sandbox_module=FakeMicrosandboxModule,
    )

    with pytest.raises(FakeMicrosandboxError, match="ended without exit event") as exc_info:
        asyncio.run(runner.exec(ExecCommand.process("sleep", "30")))

    assert exc_info.value is command_error
    assert sandbox.ping_calls == 1
    assert runner._exec_closed is False


def test_microsandbox_runner_preserves_other_exact_base_error_without_liveness_probe() -> None:
    sandbox = FakeSandbox("runner")
    command_error = FakeMicrosandboxError("protocol error: response ended mid-frame")
    sandbox.next_handle = FakeHandle([], collect_error=command_error)
    sandbox.ping_error = ConnectionResetError("agent connection reset")
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        sandbox_module=FakeMicrosandboxModule,
    )

    with pytest.raises(FakeMicrosandboxError, match="response ended mid-frame") as exc_info:
        asyncio.run(runner.exec(ExecCommand.process("pwd")))

    assert exc_info.value is command_error
    assert sandbox.ping_calls == 0
    assert runner._exec_closed is False


@pytest.mark.parametrize(
    "stream_error",
    [RuntimeError("sandbox is stopped"), ConnectionError("transport failed")],
    ids=["sandbox-stopped", "transport-failed"],
)
def test_microsandbox_runner_does_not_reclassify_other_sdk_failures(
    stream_error: Exception,
) -> None:
    sandbox = FakeSandbox("runner")
    sandbox.stream_error = stream_error
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        sandbox_module=FakeMicrosandboxModule,
    )

    with pytest.raises(type(stream_error), match=str(stream_error)):
        asyncio.run(runner.exec(ExecCommand.process("pwd")))

    assert sandbox.ping_calls == 0
    assert runner._exec_closed is False


def test_microsandbox_runner_applies_trusted_env_overlay_after_command_env() -> None:
    sandbox = FakeSandbox("runner")
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        env_overlay={
            "HTTPS_PROXY": "http://host.microsandbox.internal:8443",
            "STRIPE_SECRET_KEY": "sk_test_cayu_virtual",
        },
        sandbox_module=FakeMicrosandboxModule,
    )

    asyncio.run(
        runner.exec(
            ExecCommand.process("env"),
            env={
                "HTTPS_PROXY": "http://attacker.example:8080",
                "STRIPE_SECRET_KEY": "attacker-value",
                "VISIBLE": "1",
            },
        )
    )

    assert sandbox.exec_calls[0]["env"] == {
        "HTTPS_PROXY": "http://host.microsandbox.internal:8443",
        "STRIPE_SECRET_KEY": "sk_test_cayu_virtual",
        "VISIBLE": "1",
    }


def test_microsandbox_runner_accepts_sdk_dataclass_events_without_wait() -> None:
    class WaitRaisesHandle(FakeHandle):
        async def wait(self) -> tuple[int, bool]:
            raise RuntimeError("wait should not be called after exit event")

    sandbox = FakeSandbox("runner")
    sandbox.next_handle = WaitRaisesHandle(
        [
            FakeStdoutEvent(b"ok"),
            FakeStderrEvent(b"warn"),
            FakeExitedEvent(0),
        ]
    )
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        sandbox_module=FakeMicrosandboxModule,
    )

    result = asyncio.run(runner.exec(ExecCommand.process("pwd")))

    assert result.exit_code == 0
    assert result.stdout == "ok"
    assert result.stderr == "warn"


def test_microsandbox_runner_uses_collect_when_stream_has_no_exit_event() -> None:
    sandbox = FakeSandbox("runner")
    sandbox.next_handle = FakeHandle(
        [FakeStdoutEvent(b"ok")],
        collect_output=FakeExecOutput(exit_code=4, stderr_bytes=b"late warn"),
    )
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        sandbox_module=FakeMicrosandboxModule,
    )

    result = asyncio.run(runner.exec(ExecCommand.process("pwd")))

    assert result.exit_code == 4
    assert result.stdout == "ok"
    assert result.stderr == "late warn"


def test_microsandbox_runner_prefers_collected_output_after_incomplete_stream() -> None:
    sandbox = FakeSandbox("runner")
    sandbox.next_handle = FakeHandle(
        [FakeStdoutEvent(b"partial"), FakeStderrEvent(b"partial err")],
        collect_output=FakeExecOutput(
            exit_code=0,
            stdout_bytes=b"complete stdout",
            stderr_bytes=b"complete stderr",
        ),
    )
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        sandbox_module=FakeMicrosandboxModule,
    )

    result = asyncio.run(
        runner.exec(
            ExecCommand.process("pwd"),
            output_limit_bytes=8,
        )
    )

    assert result.exit_code == 0
    assert result.stdout == "complete"
    assert result.stderr == "complete"
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True


def test_microsandbox_runner_uses_collected_output_when_stream_has_no_data() -> None:
    sandbox = FakeSandbox("runner")
    sandbox.next_handle = FakeHandle(
        [],
        collect_output=FakeExecOutput(
            exit_code=0,
            stdout_bytes=b"collected stdout",
            stderr_bytes=b"collected stderr",
        ),
    )
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        sandbox_module=FakeMicrosandboxModule,
    )

    result = asyncio.run(runner.exec(ExecCommand.process("pwd")))

    assert result.exit_code == 0
    assert result.stdout == "collected stdout"
    assert result.stderr == "collected stderr"


def test_microsandbox_runner_executes_shell() -> None:
    sandbox = FakeSandbox("runner")
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        sandbox_module=FakeMicrosandboxModule,
    )

    result = asyncio.run(runner.exec(ExecCommand.bash("echo ok")))

    assert result.stdout == "hello world"
    assert result.stderr == "warn"
    assert result.exit_code == 7
    assert sandbox.shell_calls == [
        {
            "script": "echo ok",
            "cwd": "/workspace",
            "env": {},
            "timeout": None,
            "stdin": None,
        }
    ]


def test_microsandbox_runner_restricts_cwd_to_guest_root() -> None:
    runner = MicrosandboxRunner(
        FakeSandbox("runner"),
        name="runner",
        default_cwd="/repo",
        sandbox_module=FakeMicrosandboxModule,
    )

    assert runner.resolve_cwd(None) == "/repo"
    assert runner.resolve_cwd("src") == "/repo/src"
    assert runner.resolve_cwd("src/../tests") == "/repo/tests"
    assert runner.resolve_cwd("/repo") == "/repo"
    assert runner.resolve_cwd("/repo/src/../tests") == "/repo/tests"
    with pytest.raises(ValueError, match="outside the runner root"):
        runner.resolve_cwd("/etc")
    with pytest.raises(ValueError, match="escapes"):
        runner.resolve_cwd("../etc")


def test_microsandbox_runner_close_actions_are_explicit() -> None:
    async def run() -> None:
        reset_fake_module()
        removable = await MicrosandboxRunner.create(
            "remove-me",
            sandbox_module=FakeMicrosandboxModule,
        )
        removable_sandbox = removable._sandbox
        await removable.close()
        await removable.close()
        assert removable_sandbox.stop_calls == 0
        assert removable_sandbox.stop_and_wait_calls == 1
        assert FakeSandboxApi.removed == ["remove-me"]
        assert removable.last_cleanup_diagnostic is not None
        assert removable.last_cleanup_diagnostic["status"] == "removed"

        detachable_sandbox = FakeSandbox("detach-me")
        detachable = MicrosandboxRunner(
            detachable_sandbox,
            name="detach-me",
            close_action="detach",
            sandbox_module=FakeMicrosandboxModule,
        )
        await detachable.close()
        assert detachable_sandbox.detach_calls == 1
        assert detachable_sandbox.stop_calls == 0
        assert detachable_sandbox.stop_and_wait_calls == 0
        assert detachable.last_cleanup_diagnostic is not None
        assert detachable.last_cleanup_diagnostic["status"] == "detached"

        no_op_sandbox = FakeSandbox("keep-me")
        no_op = MicrosandboxRunner(
            no_op_sandbox,
            name="keep-me",
            close_action="none",
            sandbox_module=FakeMicrosandboxModule,
        )
        await no_op.close()
        assert no_op_sandbox.stop_calls == 0
        assert no_op_sandbox.stop_and_wait_calls == 0
        assert no_op.last_cleanup_diagnostic is not None
        assert no_op.last_cleanup_diagnostic["status"] == "skipped"

    asyncio.run(run())


def test_microsandbox_runner_remove_records_immediate_cleanup_without_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_sleep(_delay: float) -> None:
        raise AssertionError("immediate removal must not sleep")

    monkeypatch.setattr("cayu.runners.microsandbox.asyncio.sleep", unexpected_sleep)

    async def run() -> dict[str, Any] | None:
        reset_fake_module()
        runner = await MicrosandboxRunner.create(
            "remove-immediately",
            sandbox_module=FakeMicrosandboxModule,
        )
        assert runner.remove_timeout_s == DEFAULT_MICROSANDBOX_REMOVE_TIMEOUT_SECONDS
        await runner.close()
        return runner.last_cleanup_diagnostic

    diagnostic = asyncio.run(run())

    assert FakeSandboxApi.remove_calls == ["remove-immediately"]
    assert diagnostic is not None
    assert diagnostic["status"] == "removed"
    assert diagnostic["attempts"] == [{"attempt": 1, "status": "removed"}]


def test_microsandbox_runner_retries_only_still_running_removal() -> None:
    async def run() -> dict[str, Any] | None:
        reset_fake_module()
        FakeSandboxApi.remove_failures = [
            FakeSandboxStillRunningError("sandbox status has not settled")
        ]
        FakeSandboxApi.statuses = ["draining"]
        runner = await MicrosandboxRunner.create(
            "settles-after-retry",
            sandbox_module=FakeMicrosandboxModule,
        )
        await runner.close()
        return runner.last_cleanup_diagnostic

    diagnostic = asyncio.run(run())

    assert FakeSandboxApi.remove_calls == ["settles-after-retry", "settles-after-retry"]
    assert diagnostic is not None
    assert diagnostic["status"] == "removed"
    assert diagnostic["attempts"] == [
        {"attempt": 1, "status": "deferred", "sandbox_status": "draining"},
        {"attempt": 2, "status": "removed"},
    ]
    assert diagnostic["observed_statuses"] == ["draining"]


def test_microsandbox_runner_bounds_unsettled_removal() -> None:
    async def run() -> tuple[MicrosandboxRunner, MicrosandboxCleanupError]:
        reset_fake_module()
        FakeSandboxApi.always_still_running = True
        FakeSandboxApi.statuses = ["running"] * 10
        runner = await MicrosandboxRunner.create(
            "never-settles",
            remove_timeout_s=0.01,
            sandbox_module=FakeMicrosandboxModule,
        )
        with pytest.raises(MicrosandboxCleanupError) as exc_info:
            await runner.close()
        return runner, exc_info.value

    runner, error = asyncio.run(run())

    assert len(FakeSandboxApi.remove_calls) >= 2
    assert error.diagnostic["status"] == "timed_out"
    assert error.diagnostic["error_type"] == "FakeSandboxStillRunningError"
    assert runner.last_cleanup_diagnostic == error.diagnostic
    assert runner._closed is False


@pytest.mark.parametrize("stalled_operation", ["remove", "get", "refresh"])
def test_microsandbox_runner_deadline_bounds_sdk_operations(stalled_operation: str) -> None:
    async def run() -> tuple[MicrosandboxRunner, MicrosandboxCleanupError, float]:
        reset_fake_module()
        if stalled_operation == "remove":
            FakeSandboxApi.hang_remove = True
        else:
            FakeSandboxApi.remove_failures = [
                FakeSandboxStillRunningError("sandbox status has not settled")
            ]
            setattr(FakeSandboxApi, f"hang_{stalled_operation}", True)
        runner = await MicrosandboxRunner.create(
            f"stalled-{stalled_operation}",
            remove_timeout_s=0.02,
            sandbox_module=FakeMicrosandboxModule,
        )
        started = asyncio.get_running_loop().time()
        with pytest.raises(MicrosandboxCleanupError) as exc_info:
            await runner.close()
        return runner, exc_info.value, asyncio.get_running_loop().time() - started

    runner, error, elapsed_s = asyncio.run(run())

    assert elapsed_s < 1
    assert error.diagnostic["status"] == "timed_out"
    assert error.diagnostic["attempts"] == [
        {
            "attempt": 1,
            "status": "timed_out",
            "operation": "remove" if stalled_operation == "remove" else "status_refresh",
        }
    ]
    assert runner.last_cleanup_diagnostic == error.diagnostic
    assert runner._closed is False


@pytest.mark.parametrize("cancelled_operation", ["remove", "get", "refresh", "backoff"])
def test_microsandbox_runner_records_removal_cancellation(
    cancelled_operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[MicrosandboxRunner, asyncio.CancelledError]:
        reset_fake_module()
        cancellation_started = asyncio.Event()

        if cancelled_operation == "remove":

            async def hanging_remove(cls: type[FakeSandboxApi], name: str) -> None:
                cls.remove_calls.append(name)
                cancellation_started.set()
                await asyncio.Future()

            monkeypatch.setattr(FakeSandboxApi, "remove", classmethod(hanging_remove))
        else:
            FakeSandboxApi.remove_failures = [
                FakeSandboxStillRunningError("sandbox status has not settled")
            ]
            if cancelled_operation == "get":

                async def hanging_get(cls: type[FakeSandboxApi], name: str) -> FakeHandleRecord:
                    del cls, name
                    cancellation_started.set()
                    await asyncio.Future()
                    raise AssertionError("unreachable")

                monkeypatch.setattr(FakeSandboxApi, "get", classmethod(hanging_get))
            elif cancelled_operation == "refresh":

                async def hanging_refresh(self: FakeHandleRecord) -> FakeHandleRecord:
                    del self
                    cancellation_started.set()
                    await asyncio.Future()
                    raise AssertionError("unreachable")

                monkeypatch.setattr(FakeHandleRecord, "refresh", hanging_refresh)
            else:

                async def hanging_backoff(_delay_s: float) -> None:
                    cancellation_started.set()
                    await asyncio.Future()

                monkeypatch.setattr(
                    "cayu.runners.microsandbox._sleep_before_microsandbox_retry",
                    hanging_backoff,
                )

        runner = await MicrosandboxRunner.create(
            f"cancel-removal-{cancelled_operation}",
            sandbox_module=FakeMicrosandboxModule,
        )
        close_task = asyncio.create_task(runner.close())
        await asyncio.wait_for(cancellation_started.wait(), timeout=1)
        close_task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await close_task
        return runner, exc_info.value

    runner, error = asyncio.run(run())

    diagnostic = cleanup_diagnostic(error)
    assert runner.last_cleanup_diagnostic == diagnostic
    assert diagnostic["action"] == "remove"
    assert diagnostic["status"] == "failed"
    assert diagnostic["error_type"] == "CancelledError"
    assert diagnostic["attempts"][-1] == {
        "attempt": 1,
        "status": "cancelled",
        "operation": (
            "status_refresh" if cancelled_operation in {"get", "refresh"} else cancelled_operation
        ),
    }
    assert runner._closed is False


def test_microsandbox_runner_surfaces_nonretryable_remove_error_immediately() -> None:
    async def run() -> MicrosandboxRunner:
        reset_fake_module()
        FakeSandboxApi.remove_failures = [PermissionError("remove denied")]
        runner = await MicrosandboxRunner.create(
            "remove-denied",
            sandbox_module=FakeMicrosandboxModule,
        )
        with pytest.raises(PermissionError, match="remove denied"):
            await runner.close()
        return runner

    runner = asyncio.run(run())

    assert FakeSandboxApi.remove_calls == ["remove-denied"]
    assert runner.last_cleanup_diagnostic is not None
    assert runner.last_cleanup_diagnostic["status"] == "failed"
    assert runner.last_cleanup_diagnostic["error_type"] == "PermissionError"
    assert runner.last_cleanup_diagnostic["attempts"] == [
        {"attempt": 1, "status": "failed", "operation": "remove"}
    ]


def test_microsandbox_runner_preserves_history_before_terminal_remove_error() -> None:
    async def run() -> MicrosandboxRunner:
        reset_fake_module()
        FakeSandboxApi.remove_failures = [
            FakeSandboxStillRunningError("sandbox status has not settled"),
            PermissionError("remove denied"),
        ]
        FakeSandboxApi.statuses = ["draining"]
        runner = await MicrosandboxRunner.create(
            "remove-denied-after-retry",
            sandbox_module=FakeMicrosandboxModule,
        )
        with pytest.raises(PermissionError, match="remove denied"):
            await runner.close()
        return runner

    runner = asyncio.run(run())

    assert runner.last_cleanup_diagnostic is not None
    assert runner.last_cleanup_diagnostic["attempts"] == [
        {"attempt": 1, "status": "deferred", "sandbox_status": "draining"},
        {"attempt": 2, "status": "failed", "operation": "remove"},
    ]
    assert runner.last_cleanup_diagnostic["observed_statuses"] == ["draining"]


def test_microsandbox_runner_treats_already_removed_as_success() -> None:
    async def run() -> dict[str, Any] | None:
        reset_fake_module()
        FakeSandboxApi.remove_failures = [FakeSandboxNotFoundError("sandbox not found")]
        runner = await MicrosandboxRunner.create(
            "already-removed",
            sandbox_module=FakeMicrosandboxModule,
        )
        await runner.close()
        return runner.last_cleanup_diagnostic

    diagnostic = asyncio.run(run())

    assert diagnostic is not None
    assert diagnostic["status"] == "removed"
    assert diagnostic["attempts"] == [{"attempt": 1, "status": "already_removed"}]


def test_microsandbox_runner_treats_not_found_during_stop_as_already_removed() -> None:
    async def run() -> tuple[FakeSandbox, MicrosandboxRunner]:
        reset_fake_module()
        runner = await MicrosandboxRunner.create(
            "removed-before-stop",
            sandbox_module=FakeMicrosandboxModule,
        )
        sandbox = runner._sandbox
        sandbox.stop_failure = FakeSandboxNotFoundError("sandbox not found")
        await runner.close()
        return sandbox, runner

    sandbox, runner = asyncio.run(run())

    assert sandbox.stop_and_wait_calls == 1
    assert FakeSandboxApi.remove_calls == []
    assert runner.last_cleanup_diagnostic is not None
    assert runner.last_cleanup_diagnostic["status"] == "removed"
    assert runner.last_cleanup_diagnostic["attempts"] == [
        {"attempt": 1, "status": "already_removed", "operation": "stop"}
    ]
    assert runner._closed is True


def test_microsandbox_runner_uses_06_stop_and_wait_contract() -> None:
    class CurrentSandbox(FakeSandbox):
        stop_and_wait = None

        def __init__(self, name: str) -> None:
            super().__init__(name)
            self.wait_until_stopped_calls = 0

        async def wait_until_stopped(self) -> Any:
            self.wait_until_stopped_calls += 1
            return type("StopResult", (), {"status": "stopped"})()

    async def run() -> tuple[CurrentSandbox, dict[str, Any] | None]:
        sandbox = CurrentSandbox("current-sdk")
        runner = MicrosandboxRunner(
            sandbox,
            name="current-sdk",
            close_action="stop",
            sandbox_module=FakeMicrosandboxModule,
        )
        await runner.close()
        return sandbox, runner.last_cleanup_diagnostic

    sandbox, diagnostic = asyncio.run(run())

    assert sandbox.stop_calls == 1
    assert sandbox.wait_until_stopped_calls == 1
    assert diagnostic is not None
    assert diagnostic["observed_statuses"] == ["stopped"]


def test_microsandbox_runner_does_not_create_sandbox_for_invalid_lifecycle_config() -> None:
    async def run() -> None:
        reset_fake_module()
        bad_action: Any = "delete"
        with pytest.raises(ValueError, match="close_action"):
            await MicrosandboxRunner.create(
                "bad-action",
                close_action=bad_action,
                sandbox_module=FakeMicrosandboxModule,
            )
        with pytest.raises(ValueError, match="absolute"):
            await MicrosandboxRunner.create(
                "bad-cwd",
                default_cwd="workspace",
                sandbox_module=FakeMicrosandboxModule,
            )
        bad_ensure_default_cwd: Any = "yes"
        with pytest.raises(TypeError, match="ensure_default_cwd"):
            await MicrosandboxRunner.create(
                "bad-ensure",
                ensure_default_cwd=bad_ensure_default_cwd,
                sandbox_module=FakeMicrosandboxModule,
            )
        with pytest.raises(ValueError, match="remove_timeout_s"):
            await MicrosandboxRunner.create(
                "bad-remove-timeout",
                remove_timeout_s=0,
                sandbox_module=FakeMicrosandboxModule,
            )
        with pytest.raises(ValueError, match="liveness_timeout_s"):
            await MicrosandboxRunner.create(
                "bad-liveness-timeout",
                liveness_timeout_s=0,
                sandbox_module=FakeMicrosandboxModule,
            )

    asyncio.run(run())

    assert FakeSandboxApi.created == []


def test_microsandbox_runner_cleans_up_created_sandbox_when_setup_fails() -> None:
    async def run() -> FakeSandbox:
        reset_fake_module()
        FakeSandboxApi.fail_created_setup = True
        with pytest.raises(RuntimeError, match="exec failed"):
            await MicrosandboxRunner.create(
                "setup-fails",
                sandbox_module=FakeMicrosandboxModule,
            )
        assert FakeSandboxApi.existing is not None
        return FakeSandboxApi.existing

    sandbox = asyncio.run(run())

    assert sandbox.stop_calls == 0
    assert sandbox.stop_and_wait_calls == 1
    assert FakeSandboxApi.removed == ["setup-fails"]


def test_microsandbox_runner_setup_failure_retries_transient_removal_lag() -> None:
    async def run() -> FakeSandbox:
        reset_fake_module()
        FakeSandboxApi.fail_created_setup = True
        FakeSandboxApi.remove_failures = [
            FakeSandboxStillRunningError("sandbox status has not settled")
        ]
        FakeSandboxApi.statuses = ["draining"]
        with pytest.raises(RuntimeError, match="exec failed"):
            await MicrosandboxRunner.create(
                "setup-fails-remove-lags",
                sandbox_module=FakeMicrosandboxModule,
            )
        assert FakeSandboxApi.existing is not None
        return FakeSandboxApi.existing

    sandbox = asyncio.run(run())

    assert sandbox.stop_and_wait_calls == 1
    assert FakeSandboxApi.remove_calls == [
        "setup-fails-remove-lags",
        "setup-fails-remove-lags",
    ]
    assert FakeSandboxApi.removed == ["setup-fails-remove-lags"]


def test_microsandbox_runner_setup_failure_accepts_sandbox_removed_before_stop() -> None:
    async def run() -> FakeSandbox:
        reset_fake_module()
        FakeSandboxApi.fail_created_setup = True
        FakeSandboxApi.created_stop_failure = FakeSandboxNotFoundError("sandbox not found")
        with pytest.raises(RuntimeError, match="exec failed"):
            await MicrosandboxRunner.create(
                "setup-fails-after-removal",
                sandbox_module=FakeMicrosandboxModule,
            )
        assert FakeSandboxApi.existing is not None
        return FakeSandboxApi.existing

    sandbox = asyncio.run(run())

    assert sandbox.stop_and_wait_calls == 1
    assert FakeSandboxApi.remove_calls == []


def test_microsandbox_runner_reports_setup_and_cleanup_failures_together() -> None:
    async def run() -> FakeSandbox:
        reset_fake_module()
        FakeSandboxApi.fail_created_setup = True
        FakeSandboxApi.remove_failures = [PermissionError("remove denied")]
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await MicrosandboxRunner.create(
                "setup-and-cleanup-fail",
                sandbox_module=FakeMicrosandboxModule,
            )
        assert "setup failed and cleanup failed" in str(exc_info.value)
        assert len(exc_info.value.exceptions) == 2
        assert isinstance(exc_info.value.exceptions[0], RuntimeError)
        cleanup_error = exc_info.value.exceptions[1]
        assert isinstance(cleanup_error, PermissionError)
        diagnostic = cleanup_diagnostic(cleanup_error)
        assert diagnostic["type"] == "cayu.microsandbox_cleanup.v1"
        assert diagnostic["status"] == "failed"
        assert diagnostic["error_type"] == "PermissionError"
        assert diagnostic["attempts"] == [{"attempt": 1, "status": "failed", "operation": "remove"}]
        assert FakeSandboxApi.existing is not None
        return FakeSandboxApi.existing

    sandbox = asyncio.run(run())

    assert sandbox.stop_calls == 0
    assert sandbox.stop_and_wait_calls == 1
    assert FakeSandboxApi.removed == []


def test_microsandbox_runner_reports_setup_and_stop_failure_after_successful_removal() -> None:
    async def run() -> FakeSandbox:
        reset_fake_module()
        FakeSandboxApi.fail_created_setup = True
        FakeSandboxApi.created_stop_failure = PermissionError("stop denied")
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await MicrosandboxRunner.create(
                "setup-and-stop-fail",
                sandbox_module=FakeMicrosandboxModule,
            )
        errors = exc_info.value.exceptions
        assert len(errors) == 2
        assert isinstance(errors[0], RuntimeError)
        stop_error = errors[1]
        assert isinstance(stop_error, PermissionError)
        diagnostic = cleanup_diagnostic(stop_error)
        assert diagnostic["action"] == "stop"
        assert diagnostic["status"] == "failed"
        assert diagnostic["error_type"] == "PermissionError"
        assert diagnostic["attempts"] == []
        assert FakeSandboxApi.existing is not None
        return FakeSandboxApi.existing

    sandbox = asyncio.run(run())

    assert sandbox.stop_and_wait_calls == 1
    assert FakeSandboxApi.remove_calls == ["setup-and-stop-fail"]
    assert FakeSandboxApi.removed == ["setup-and-stop-fail"]


def test_microsandbox_runner_preserves_setup_stop_and_removal_failures() -> None:
    async def run() -> FakeSandbox:
        reset_fake_module()
        FakeSandboxApi.fail_created_setup = True
        FakeSandboxApi.created_stop_failure = PermissionError("stop denied")
        FakeSandboxApi.remove_failures = [ConnectionError("remove transport failed")]
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await MicrosandboxRunner.create(
                "setup-stop-and-remove-fail",
                sandbox_module=FakeMicrosandboxModule,
            )
        errors = exc_info.value.exceptions
        assert len(errors) == 3
        assert isinstance(errors[0], RuntimeError)
        stop_error = errors[1]
        removal_error = errors[2]
        assert isinstance(stop_error, PermissionError)
        assert isinstance(removal_error, ConnectionError)
        stop_diagnostic = cleanup_diagnostic(stop_error)
        removal_diagnostic = cleanup_diagnostic(removal_error)
        assert stop_diagnostic["action"] == "stop"
        assert stop_diagnostic["error_type"] == "PermissionError"
        assert removal_diagnostic["action"] == "remove"
        assert removal_diagnostic["error_type"] == "ConnectionError"
        assert removal_diagnostic["attempts"] == [
            {"attempt": 1, "status": "failed", "operation": "remove"}
        ]
        assert FakeSandboxApi.existing is not None
        return FakeSandboxApi.existing

    sandbox = asyncio.run(run())

    assert sandbox.stop_and_wait_calls == 1
    assert FakeSandboxApi.remove_calls == ["setup-stop-and-remove-fail"]
    assert FakeSandboxApi.removed == []


def test_microsandbox_runner_preserves_setup_cancellation_before_stop_failure() -> None:
    async def run() -> FakeSandbox:
        reset_fake_module()
        FakeSandboxApi.cancel_created_setup = True
        FakeSandboxApi.created_stop_failure = PermissionError("stop denied")
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await MicrosandboxRunner.create(
                "setup-cancelled-stop-fails",
                sandbox_module=FakeMicrosandboxModule,
            )
        errors = exc_info.value.exceptions
        assert len(errors) == 2
        assert isinstance(errors[0], asyncio.CancelledError)
        stop_error = errors[1]
        assert isinstance(stop_error, PermissionError)
        assert cleanup_diagnostic(stop_error)["action"] == "stop"
        assert FakeSandboxApi.existing is not None
        return FakeSandboxApi.existing

    sandbox = asyncio.run(run())

    assert sandbox.stop_and_wait_calls == 1
    assert FakeSandboxApi.remove_calls == ["setup-cancelled-stop-fails"]
    assert FakeSandboxApi.removed == ["setup-cancelled-stop-fails"]


def test_microsandbox_runner_cleans_up_created_sandbox_when_setup_is_cancelled() -> None:
    async def run() -> FakeSandbox:
        reset_fake_module()
        FakeSandboxApi.cancel_created_setup = True
        with pytest.raises(asyncio.CancelledError):
            await MicrosandboxRunner.create(
                "setup-cancelled",
                sandbox_module=FakeMicrosandboxModule,
            )
        assert FakeSandboxApi.existing is not None
        return FakeSandboxApi.existing

    sandbox = asyncio.run(run())

    assert sandbox.stop_calls == 0
    assert sandbox.stop_and_wait_calls == 1
    assert FakeSandboxApi.removed == ["setup-cancelled"]


def test_microsandbox_runner_close_can_retry_after_cleanup_failure() -> None:
    async def run() -> FakeSandbox:
        reset_fake_module()
        runner = await MicrosandboxRunner.create(
            "retry-cleanup",
            sandbox_module=FakeMicrosandboxModule,
        )
        sandbox = runner._sandbox
        sandbox.fail_repeated_stop = True
        FakeSandboxApi.fail_next_remove = True
        with pytest.raises(RuntimeError, match="remove failed"):
            await runner.close()
        await runner.close()
        return sandbox

    sandbox = asyncio.run(run())

    assert sandbox.stop_calls == 0
    assert sandbox.stop_and_wait_calls == 1
    assert FakeSandboxApi.removed == ["retry-cleanup"]


def test_microsandbox_runner_close_retry_accepts_late_removal_completion() -> None:
    async def run() -> tuple[FakeSandbox, MicrosandboxRunner]:
        reset_fake_module()
        runner = await MicrosandboxRunner.create(
            "late-removal",
            sandbox_module=FakeMicrosandboxModule,
        )
        sandbox = runner._sandbox
        sandbox.fail_repeated_stop = True
        FakeSandboxApi.remove_failures = [
            RuntimeError("remove acknowledgement lost"),
            FakeSandboxNotFoundError("sandbox not found"),
        ]
        with pytest.raises(RuntimeError, match="acknowledgement lost"):
            await runner.close()
        await runner.close()
        return sandbox, runner

    sandbox, runner = asyncio.run(run())

    assert sandbox.stop_and_wait_calls == 1
    assert FakeSandboxApi.remove_calls == ["late-removal", "late-removal"]
    assert runner.last_cleanup_diagnostic is not None
    assert runner.last_cleanup_diagnostic["status"] == "removed"
    assert runner.last_cleanup_diagnostic["attempts"] == [
        {"attempt": 1, "status": "already_removed"}
    ]
    assert runner._closed is True


def test_microsandbox_runner_close_retries_failed_stop() -> None:
    async def run() -> tuple[FakeSandbox, MicrosandboxRunner]:
        reset_fake_module()
        runner = await MicrosandboxRunner.create(
            "retry-stop",
            sandbox_module=FakeMicrosandboxModule,
        )
        sandbox = runner._sandbox
        sandbox.fail_next_stop = True
        with pytest.raises(RuntimeError, match="stop failed"):
            await runner.close()
        await runner.close()
        return sandbox, runner

    sandbox, runner = asyncio.run(run())

    assert sandbox.stop_and_wait_calls == 2
    assert FakeSandboxApi.removed == ["retry-stop"]
    assert runner._closed is True


def test_microsandbox_runner_close_recovers_after_lost_stop_acknowledgement() -> None:
    async def run() -> tuple[FakeSandbox, MicrosandboxRunner]:
        reset_fake_module()
        runner = await MicrosandboxRunner.create(
            "lost-stop-acknowledgement",
            sandbox_module=FakeMicrosandboxModule,
        )
        sandbox = runner._sandbox
        sandbox.fail_next_stop = True
        sandbox.fail_repeated_stop = True
        with pytest.raises(RuntimeError, match="stop failed"):
            await runner.close()
        await runner.close()
        return sandbox, runner

    sandbox, runner = asyncio.run(run())

    assert sandbox.stop_and_wait_calls == 2
    assert FakeSandboxApi.removed == ["lost-stop-acknowledgement"]
    assert runner.last_cleanup_diagnostic is not None
    assert runner.last_cleanup_diagnostic["status"] == "removed"
    assert runner.last_cleanup_diagnostic["observed_statuses"] == ["stopped"]
    assert runner._closed is True


def test_microsandbox_runner_stop_accepts_already_stopped_sandbox() -> None:
    async def run() -> tuple[FakeSandbox, MicrosandboxRunner]:
        reset_fake_module()
        runner = await MicrosandboxRunner.create(
            "already-stopped",
            close_action="stop",
            sandbox_module=FakeMicrosandboxModule,
        )
        sandbox = runner._sandbox
        sandbox.already_stopped = True
        await runner.close()
        return sandbox, runner

    sandbox, runner = asyncio.run(run())

    assert sandbox.stop_and_wait_calls == 1
    assert FakeSandboxApi.removed == []
    assert runner.last_cleanup_diagnostic is not None
    assert runner.last_cleanup_diagnostic["status"] == "stopped"
    assert runner.last_cleanup_diagnostic["observed_statuses"] == ["stopped"]
    assert runner._closed is True


def test_microsandbox_runner_stop_does_not_accept_missing_sandbox() -> None:
    async def run() -> tuple[MicrosandboxRunner, FakeSandboxNotFoundError]:
        reset_fake_module()
        runner = await MicrosandboxRunner.create(
            "missing-before-stop-only",
            close_action="stop",
            sandbox_module=FakeMicrosandboxModule,
        )
        runner._sandbox.stop_failure = FakeSandboxNotFoundError("sandbox not found")
        with pytest.raises(FakeSandboxNotFoundError, match="sandbox not found") as exc_info:
            await runner.close()
        return runner, exc_info.value

    runner, error = asyncio.run(run())

    assert runner.last_cleanup_diagnostic is not None
    assert runner.last_cleanup_diagnostic["action"] == "stop"
    assert runner.last_cleanup_diagnostic["status"] == "failed"
    assert cleanup_diagnostic(error) == runner.last_cleanup_diagnostic
    assert runner._closed is False


def test_microsandbox_runner_from_existing_does_not_own_lifecycle_by_default() -> None:
    async def run() -> FakeSandbox:
        reset_fake_module()
        FakeSandboxApi.existing = FakeSandbox("existing")
        runner = await MicrosandboxRunner.from_existing(
            "existing",
            liveness_timeout_s=2.5,
            sandbox_module=FakeMicrosandboxModule,
        )
        sandbox = runner._sandbox
        assert runner.liveness_timeout_s == 2.5
        await runner.close()
        return sandbox

    sandbox = asyncio.run(run())

    assert sandbox.stop_calls == 0
    assert sandbox.stop_and_wait_calls == 0
    assert FakeSandboxApi.removed == []


def test_microsandbox_runner_from_existing_can_remove_with_bounded_retry() -> None:
    async def run() -> dict[str, Any] | None:
        reset_fake_module()
        FakeSandboxApi.existing = FakeSandbox("existing-remove")
        FakeSandboxApi.remove_failures = [
            FakeSandboxStillRunningError("sandbox status has not settled")
        ]
        runner = await MicrosandboxRunner.from_existing(
            "existing-remove",
            close_action="remove",
            sandbox_module=FakeMicrosandboxModule,
        )
        await runner.close()
        return runner.last_cleanup_diagnostic

    diagnostic = asyncio.run(run())

    assert FakeSandboxApi.remove_calls == ["existing-remove", "existing-remove"]
    assert diagnostic is not None
    assert diagnostic["status"] == "removed"


@pytest.mark.parametrize("phase", ["get", "connect"])
def test_microsandbox_runner_bounds_existing_sandbox_reattach(phase: str) -> None:
    class HangingHandle:
        async def connect(self) -> FakeSandbox:
            await asyncio.Event().wait()
            raise AssertionError

    class HangingSandboxApi:
        @classmethod
        async def get(cls, name: str) -> HangingHandle:
            del name
            if phase == "get":
                await asyncio.Event().wait()
            return HangingHandle()

    class HangingModule(FakeMicrosandboxModule):
        Sandbox = HangingSandboxApi

    async def run() -> None:
        with pytest.raises(TimeoutError, match="did not attach"):
            await MicrosandboxRunner.from_existing(
                "existing",
                reconnect_timeout_s=0.01,
                sandbox_module=HangingModule,
            )

    asyncio.run(run())


def test_microsandbox_runner_kills_command_on_cancellation_by_default() -> None:
    async def run() -> tuple[FakeSandbox, BlockingHandle, int]:
        sandbox = FakeSandbox("runner")
        handle = BlockingHandle()
        sandbox.next_handle = handle
        runner = MicrosandboxRunner(
            sandbox,
            name="runner",
            close_action="remove",
            sandbox_module=FakeMicrosandboxModule,
        )
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await handle.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        sandbox.next_handle = FakeHandle([FakeExitedEvent(code=0)])
        after = await runner.exec(ExecCommand.process("pwd"))
        return sandbox, handle, exc_info.value, after.exit_code

    sandbox, handle, exc, after = asyncio.run(run())

    assert handle.killed is True
    assert sandbox.kill_calls == 0
    assert sandbox.ping_calls == 0
    assert after == 0
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "kill_command",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]


def test_microsandbox_runner_can_kill_sandbox_on_cancellation_explicitly() -> None:
    async def run() -> tuple[FakeSandbox, BlockingHandle]:
        sandbox = FakeSandbox("runner")
        handle = BlockingHandle()
        sandbox.next_handle = handle
        runner = MicrosandboxRunner(
            sandbox,
            name="runner",
            cancellation_cleanup="sandbox",
            sandbox_module=FakeMicrosandboxModule,
        )
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await handle.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        with pytest.raises(RuntimeError, match="closed"):
            await runner.exec(ExecCommand.process("pwd"))
        return sandbox, handle, exc_info.value

    sandbox, handle, exc = asyncio.run(run())

    assert handle.killed is False
    assert sandbox.kill_calls == 1
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "kill_sandbox",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]


def test_microsandbox_runner_can_skip_cancellation_cleanup_explicitly() -> None:
    async def run() -> tuple[FakeSandbox, BlockingHandle, int]:
        sandbox = FakeSandbox("runner")
        handle = BlockingHandle()
        sandbox.next_handle = handle
        runner = MicrosandboxRunner(
            sandbox,
            name="runner",
            cancellation_cleanup="none",
            sandbox_module=FakeMicrosandboxModule,
        )
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await handle.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        sandbox.next_handle = FakeHandle([FakeExitedEvent(code=0)])
        after = await runner.exec(ExecCommand.process("pwd"))
        return sandbox, handle, exc_info.value, after.exit_code

    sandbox, handle, exc, after = asyncio.run(run())

    assert handle.killed is False
    assert sandbox.kill_calls == 0
    assert after == 0
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "none",
            "status": "skipped",
            "timeout_s": 5.0,
        }
    ]


def test_microsandbox_runner_latches_when_cancelled_before_handle_is_returned() -> None:
    async def run() -> tuple[FakeSandbox, Any]:
        reset_fake_module()
        sandbox = FakeSandbox("runner")
        sandbox.cancel_next_stream = True
        runner = MicrosandboxRunner(
            sandbox,
            name="runner",
            close_action="remove",
            sandbox_module=FakeMicrosandboxModule,
        )
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await runner.exec(ExecCommand.process("sleep", "30"))
        with pytest.raises(RuntimeError, match="command state is unknown"):
            await runner.exec(ExecCommand.process("pwd"))
        await runner.close()
        return sandbox, exc_info.value

    sandbox, exc = asyncio.run(run())

    assert sandbox.kill_calls == 0
    assert sandbox.stop_and_wait_calls == 1
    assert FakeSandboxApi.removed == ["runner"]
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "kill_command",
            "status": "unsupported",
            "timeout_s": 5.0,
            "error": "command handle is not available",
        }
    ]


def test_microsandbox_runner_reports_explicit_sandbox_cleanup_failure() -> None:
    async def run() -> tuple[FakeSandbox, BlockingHandle]:
        sandbox = FakeSandbox("runner")
        handle = BlockingHandle()
        sandbox.next_handle = handle
        sandbox.fail_kill = True
        runner = MicrosandboxRunner(
            sandbox,
            name="runner",
            cancellation_cleanup="sandbox",
            sandbox_module=FakeMicrosandboxModule,
        )
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await handle.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        with pytest.raises(RuntimeError, match="closed"):
            await runner.exec(ExecCommand.process("pwd"))
        return sandbox, handle, exc_info.value

    sandbox, handle, exc = asyncio.run(run())

    assert handle.killed is False
    assert sandbox.kill_calls == 0
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "kill_sandbox",
            "status": "failed",
            "timeout_s": 5.0,
            "error_type": "RuntimeError",
            "error": "sandbox kill failed",
        }
    ]


def test_microsandbox_runner_bounds_hanging_command_kill_on_cancellation() -> None:
    async def run() -> tuple[FakeSandbox, BlockingHandle, int]:
        sandbox = FakeSandbox("runner")
        handle = BlockingHandle()
        sandbox.next_handle = handle
        handle.hang_kill = True
        runner = MicrosandboxRunner(
            sandbox,
            name="runner",
            cancel_timeout_s=0.01,
            sandbox_module=FakeMicrosandboxModule,
        )
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await handle.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.wait_for(task, timeout=1)
        sandbox.next_handle = FakeHandle([FakeExitedEvent(code=0)])
        after = await runner.exec(ExecCommand.process("pwd"))
        return sandbox, handle, exc_info.value, after.exit_code

    sandbox, handle, exc, after = asyncio.run(run())

    assert handle.killed is False
    assert sandbox.kill_calls == 0
    assert after == 0
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "kill_command",
            "status": "timeout",
            "timeout_s": 0.01,
        }
    ]


def test_microsandbox_runner_stays_reusable_when_command_kill_fails() -> None:
    async def run() -> tuple[BlockingHandle, int]:
        sandbox = FakeSandbox("runner")
        handle = BlockingHandle()
        handle.fail_kill = True
        sandbox.next_handle = handle
        runner = MicrosandboxRunner(
            sandbox,
            name="runner",
            sandbox_module=FakeMicrosandboxModule,
        )
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await handle.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        sandbox.next_handle = FakeHandle([FakeExitedEvent(code=0)])
        after = await runner.exec(ExecCommand.process("pwd"))
        return handle, exc_info.value, after.exit_code

    handle, exc, after = asyncio.run(run())

    assert handle.killed is False
    assert after == 0
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "kill_command",
            "status": "failed",
            "timeout_s": 5.0,
            "error_type": "RuntimeError",
            "error": "kill failed",
        }
    ]


def test_microsandbox_runner_latches_when_timeout_happens_before_handle_is_returned() -> None:
    async def run() -> tuple[FakeSandbox, Any]:
        sandbox = FakeSandbox("runner")
        sandbox.timeout_next_stream = True
        runner = MicrosandboxRunner(
            sandbox,
            name="runner",
            sandbox_module=FakeMicrosandboxModule,
        )
        result = await runner.exec(ExecCommand.process("sleep", "30"))
        with pytest.raises(RuntimeError, match="command state is unknown"):
            await runner.exec(ExecCommand.process("pwd"))
        return sandbox, result

    sandbox, result = asyncio.run(run())

    assert result.timed_out is True
    assert result.exit_code == -9
    assert sandbox.kill_calls == 0
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "kill_command",
            "status": "unsupported",
            "timeout_s": 5.0,
            "error": "command handle is not available",
        }
    ]


def test_microsandbox_runner_enforces_timeout_and_kills_command_by_default() -> None:
    async def run() -> tuple[FakeSandbox, BlockingHandle, Any]:
        sandbox = FakeSandbox("runner")
        handle = BlockingHandle()
        sandbox.next_handle = handle
        runner = MicrosandboxRunner(
            sandbox,
            name="runner",
            sandbox_module=FakeMicrosandboxModule,
        )
        result = await runner.exec(
            ExecCommand.process("sleep", "30"),
            timeout_s=1,
        )
        sandbox.next_handle = FakeHandle([FakeExitedEvent(code=0)])
        after = await runner.exec(ExecCommand.process("pwd"))
        return sandbox, handle, result, after.exit_code

    sandbox, handle, result, after = asyncio.run(run())

    assert result.timed_out is True
    assert result.exit_code == -9
    assert handle.killed is True
    assert sandbox.kill_calls == 0
    assert sandbox.ping_calls == 0
    assert after == 0
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "kill_command",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]


def test_microsandbox_runner_can_kill_sandbox_on_timeout_explicitly() -> None:
    async def run() -> tuple[FakeSandbox, Any]:
        sandbox = FakeSandbox("runner")
        handle = BlockingHandle()
        sandbox.next_handle = handle
        runner = MicrosandboxRunner(
            sandbox,
            name="runner",
            timeout_cleanup="sandbox",
            sandbox_module=FakeMicrosandboxModule,
        )
        result = await runner.exec(
            ExecCommand.process("sleep", "30"),
            timeout_s=1,
        )
        with pytest.raises(RuntimeError, match="closed"):
            await runner.exec(ExecCommand.process("pwd"))
        return sandbox, result

    sandbox, result = asyncio.run(run())

    assert result.timed_out is True
    assert result.exit_code == -9
    assert sandbox.kill_calls == 1
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "kill_sandbox",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]


def test_microsandbox_runner_preserves_timeout_when_command_kill_fails() -> None:
    async def run() -> tuple[FakeSandbox, Any, int]:
        sandbox = FakeSandbox("runner")
        handle = BlockingHandle()
        handle.fail_kill = True
        sandbox.next_handle = handle
        runner = MicrosandboxRunner(
            sandbox,
            name="runner",
            sandbox_module=FakeMicrosandboxModule,
        )
        result = await runner.exec(
            ExecCommand.process("sleep", "30"),
            timeout_s=1,
        )
        sandbox.next_handle = FakeHandle([FakeExitedEvent(code=0)])
        after = await runner.exec(ExecCommand.process("pwd"))
        return sandbox, result, after.exit_code

    sandbox, result, after = asyncio.run(run())

    assert result.timed_out is True
    assert result.exit_code == -9
    assert sandbox.kill_calls == 0
    assert after == 0
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "kill_command",
            "status": "failed",
            "timeout_s": 5.0,
            "error_type": "RuntimeError",
            "error": "kill failed",
        }
    ]


def test_microsandbox_runner_validates_inputs() -> None:
    bad_close_action: Any = "delete"
    bad_command: Any = "echo bad"
    bad_cleanup: Any = "delete_process"
    bad_env: Any = []

    with pytest.raises(ValueError, match="whitespace"):
        MicrosandboxRunner(FakeSandbox("runner"), name=" runner")
    with pytest.raises(ValueError, match="128"):
        MicrosandboxRunner(FakeSandbox("runner"), name="x" * 129)
    with pytest.raises(ValueError, match="absolute"):
        MicrosandboxRunner(
            FakeSandbox("runner"),
            name="runner",
            default_cwd="workspace",
        )
    with pytest.raises(ValueError, match="close_action"):
        MicrosandboxRunner(
            FakeSandbox("runner"),
            name="runner",
            close_action=bad_close_action,
        )
    with pytest.raises(ValueError, match="cancel_timeout_s"):
        MicrosandboxRunner(
            FakeSandbox("runner"),
            name="runner",
            cancel_timeout_s=0,
        )
    with pytest.raises(ValueError, match="cancel_timeout_s"):
        MicrosandboxRunner(
            FakeSandbox("runner"),
            name="runner",
            cancel_timeout_s=inf,
        )
    with pytest.raises(ValueError, match="cancel_timeout_s"):
        MicrosandboxRunner(
            FakeSandbox("runner"),
            name="runner",
            cancel_timeout_s=nan,
        )
    with pytest.raises(ValueError, match="cancellation_cleanup"):
        MicrosandboxRunner(
            FakeSandbox("runner"),
            name="runner",
            cancellation_cleanup=bad_cleanup,
        )
    with pytest.raises(ValueError, match="timeout_cleanup"):
        MicrosandboxRunner(
            FakeSandbox("runner"),
            name="runner",
            timeout_cleanup=bad_cleanup,
        )

    runner = MicrosandboxRunner(
        FakeSandbox("runner"),
        name="runner",
        sandbox_module=FakeMicrosandboxModule,
    )
    with pytest.raises(TypeError, match="ExecCommand"):
        asyncio.run(runner.exec(bad_command))
    with pytest.raises(TypeError, match="dictionary"):
        asyncio.run(runner.exec(ExecCommand.process("env"), env=bad_env))
    with pytest.raises(ValueError, match="greater than zero"):
        asyncio.run(runner.exec(ExecCommand.process("pwd"), timeout_s=0))
