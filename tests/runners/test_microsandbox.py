from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from enum import StrEnum
from math import inf, nan
from typing import Any

import pytest

from cayu.runners import (
    DEFAULT_MICROSANDBOX_CWD,
    DEFAULT_MICROSANDBOX_REMOVE_TIMEOUT_SECONDS,
    ExecCommand,
    ExecResult,
    MicrosandboxCleanupError,
    MicrosandboxRunner,
    MicrosandboxUnavailableError,
    RunnerExecutionError,
)
from cayu.runners._cleanup import runner_cancellation_failure, sanitize_runner_artifacts
from cayu.runners.microsandbox import (
    _defer_reconnect_restoration,
    microsandbox_reconnect_settlement_task,
)
from cayu.testing import verify_provider_credential_isolation
from cayu.vaults import REDACTED_SECRET, SecretRedactor


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


@pytest.mark.anyio
@pytest.mark.parametrize("acknowledged", [False, True])
@pytest.mark.parametrize("policy", ["command", "none", "sandbox"])
async def test_transport_failure_settles_or_fences_command(acknowledged, policy):
    entered = asyncio.Event()
    release = asyncio.Event()

    class Handle(FakeHandle):
        async def __anext__(self):
            entered.set()
            await release.wait()
            raise ConnectionResetError("private-transport-canary")

    handle = Handle([])
    sandbox = FakeSandbox("runner")

    async def dispatch(*args, **kwargs):
        if acknowledged:
            return handle
        entered.set()
        await release.wait()
        raise ConnectionResetError("private-transport-canary")

    sandbox.exec_stream = dispatch
    runner = MicrosandboxRunner(
        sandbox, name="runner", cancellation_cleanup=policy, sandbox_module=FakeMicrosandboxModule
    )
    task = asyncio.create_task(runner.exec(ExecCommand.process("true")))
    await asyncio.wait_for(entered.wait(), 10)
    release.set()
    with pytest.raises(RunnerExecutionError) as caught:
        await task
    assert "private-transport-canary" not in str(caught.value)
    assert caught.value.artifacts[-1]["type"] == "cayu.runner_cleanup.v1"
    if policy == "sandbox":
        assert runner.lifecycle_state == "closed"
    elif policy == "command" and acknowledged:
        assert handle.killed and runner.lifecycle_state == "reusable"
    else:
        assert runner.lifecycle_state == "poisoned"
        with pytest.raises(RuntimeError):
            runner.reopen_exec()


@pytest.mark.anyio
async def test_constructor_rollback_retains_stop_across_repeated_cancellation(monkeypatch):
    reset_fake_module()
    setup = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def blocked_setup(self, *args, **kwargs):
        setup.set()
        await asyncio.Event().wait()

    async def blocked_stop(self):
        calls.append("stop")
        cleaning.set()
        await release.wait()

    monkeypatch.setattr(FakeSandbox, "exec", blocked_setup)
    monkeypatch.setattr(FakeSandbox, "stop_and_wait", blocked_stop)
    task = asyncio.create_task(
        MicrosandboxRunner.create(
            "owned-rollback",
            sandbox_module=FakeMicrosandboxModule,
            cancel_timeout_s=0.03,
            remove_timeout_s=0.03,
        )
    )
    try:
        await asyncio.wait_for(setup.wait(), 1)
        # The first constructor owns the name before failure creates rollback.
        with pytest.raises(RuntimeError, match="acquisition or rollback is still pending"):
            await MicrosandboxRunner.create(
                "owned-rollback", replace=True, sandbox_module=FakeMicrosandboxModule
            )
        with pytest.raises(RuntimeError, match="acquisition or rollback is still pending"):
            await MicrosandboxRunner.from_existing(
                "owned-rollback", sandbox_module=FakeMicrosandboxModule
            )
        assert len(FakeSandboxApi.created) == 1
        task.cancel("setup")
        await asyncio.wait_for(cleaning.wait(), 1)
        task.cancel("cleanup")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 2
        assert await MicrosandboxRunner.drain_failed_creations(timeout_s=0.01) == 1
        with pytest.raises(RuntimeError, match="rollback is still pending"):
            await MicrosandboxRunner.create(
                "owned-rollback", replace=True, sandbox_module=FakeMicrosandboxModule
            )
        with pytest.raises(RuntimeError, match="rollback is still pending"):
            await MicrosandboxRunner.from_existing(
                "owned-rollback", sandbox_module=FakeMicrosandboxModule
            )
        assert len(FakeSandboxApi.created) == 1
        assert calls == ["stop"] and not FakeSandboxApi.removed
        release.set()
        assert await MicrosandboxRunner.drain_failed_creations(timeout_s=1) == 0
        assert calls == ["stop"] and FakeSandboxApi.removed == ["owned-rollback"]
        replacement = await MicrosandboxRunner.create(
            "owned-rollback", ensure_default_cwd=False, sandbox_module=FakeMicrosandboxModule
        )
        assert await MicrosandboxRunner.drain_failed_creations(timeout_s=1) == 0
        assert calls == ["stop"]
        await replacement.close()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await MicrosandboxRunner.drain_failed_creations(timeout_s=1)


@pytest.mark.anyio
@pytest.mark.parametrize("first_entrance", ["create", "attach"])
@pytest.mark.parametrize("fail_ack", [False, True])
async def test_same_name_acquisition_is_exclusive_before_provider_ack(
    monkeypatch, first_entrance, fail_ack
):
    reset_fake_module()
    entered = asyncio.Event()
    release = asyncio.Event()
    method = "create" if first_entrance == "create" else "get"
    original = getattr(FakeSandboxApi, method)
    calls = []

    async def blocked(cls, name, **kwargs):
        calls.append(name)
        if name == "acquisition":
            entered.set()
            await release.wait()
            if fail_ack and calls.count(name) == 1:
                raise ConnectionError("provider lookup failed before allocation")
        return await original(name, **kwargs)

    monkeypatch.setattr(FakeSandboxApi, method, classmethod(blocked))
    entrance = (
        MicrosandboxRunner.create
        if first_entrance == "create"
        else MicrosandboxRunner.from_existing
    )
    task = asyncio.create_task(
        entrance("acquisition", close_action="none", sandbox_module=FakeMicrosandboxModule)
    )
    try:
        await asyncio.wait_for(entered.wait(), 10)
        with pytest.raises(RuntimeError, match="acquisition or rollback is still pending"):
            await MicrosandboxRunner.create(
                "acquisition", replace=True, sandbox_module=FakeMicrosandboxModule
            )
        with pytest.raises(RuntimeError, match="acquisition or rollback is still pending"):
            await MicrosandboxRunner.from_existing(
                "acquisition", sandbox_module=FakeMicrosandboxModule
            )
        assert calls == ["acquisition"]
        independent = await MicrosandboxRunner.create(
            "independent", close_action="none", sandbox_module=FakeMicrosandboxModule
        )
        await independent.close()
        if first_entrance == "attach":
            FakeSandboxApi.existing = FakeSandbox("acquisition")
        release.set()
        runner = None
        if fail_ack:
            with pytest.raises(ConnectionError):
                await asyncio.wait_for(task, 1)
        else:
            runner = await asyncio.wait_for(task, 1)
            assert runner.lifecycle_state == "reusable"
        attached = await MicrosandboxRunner.from_existing(
            "acquisition", close_action="none", sandbox_module=FakeMicrosandboxModule
        )
        assert attached.lifecycle_state == "reusable"
        await attached.close()
        if runner is not None:
            await runner.close()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
@pytest.mark.parametrize("constructor_rollback", [False, True])
async def test_dispatched_removal_retains_owner_past_deadline(monkeypatch, constructor_rollback):
    reset_fake_module()
    accepted = asyncio.Event()
    release = asyncio.Event()
    calls = []
    cancelled = []
    original_remove = FakeSandboxApi.remove

    async def pending_remove(cls, name):
        calls.append(name)
        accepted.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.append(name)
            raise
        await original_remove(name)

    monkeypatch.setattr(FakeSandboxApi, "remove", classmethod(pending_remove))
    name = f"pending-removal-{constructor_rollback}"
    runner = None
    if constructor_rollback:
        FakeSandboxApi.fail_created_setup = True
        owner = asyncio.create_task(
            MicrosandboxRunner.create(
                name,
                sandbox_module=FakeMicrosandboxModule,
                cancel_timeout_s=0.02,
                remove_timeout_s=0.02,
            )
        )
    else:
        runner = await MicrosandboxRunner.create(
            name,
            sandbox_module=FakeMicrosandboxModule,
            cancel_timeout_s=0.02,
            remove_timeout_s=0.02,
        )
        owner = asyncio.create_task(runner.close())
    try:
        async with asyncio.timeout(1):
            await accepted.wait()
        if constructor_rollback:
            with pytest.raises(ExceptionGroup) as caught:
                await owner
            assert isinstance(caught.value.exceptions[-1], TimeoutError)
            assert await MicrosandboxRunner.drain_failed_creations(timeout_s=0.03) == 1
            with pytest.raises(RuntimeError, match="acquisition or rollback"):
                await MicrosandboxRunner.create(
                    name, replace=True, sandbox_module=FakeMicrosandboxModule
                )
            with pytest.raises(RuntimeError, match="acquisition or rollback"):
                await MicrosandboxRunner.from_existing(name, sandbox_module=FakeMicrosandboxModule)
        else:
            with pytest.raises(TimeoutError):
                await owner
            assert runner is not None
            with pytest.raises(TimeoutError):
                await runner.close()
            with pytest.raises(RuntimeError):
                runner.reopen_exec()
        assert calls == [name] and cancelled == []
        assert name not in FakeSandboxApi.removed
        release.set()
        if constructor_rollback:
            assert await MicrosandboxRunner.drain_failed_creations(timeout_s=1) == 0
            replacement = await MicrosandboxRunner.create(
                name,
                ensure_default_cwd=False,
                close_action="none",
                sandbox_module=FakeMicrosandboxModule,
            )
            assert replacement.lifecycle_state == "reusable"
            await replacement.close()
        else:
            assert runner is not None
            await runner.close()
            assert runner.is_closed
        assert calls == [name] and cancelled == []
        assert FakeSandboxApi.removed == [name]
    finally:
        release.set()
        await asyncio.gather(owner, return_exceptions=True)
        if constructor_rollback:
            await MicrosandboxRunner.drain_failed_creations(timeout_s=1)
        elif runner is not None:
            await runner.close()


@pytest.mark.anyio
async def test_confirmed_constructor_removal_does_not_reclaim_replacement():
    reset_fake_module()
    FakeSandboxApi.fail_created_setup = True
    FakeSandboxApi.created_stop_failure = PermissionError("stop denied")
    with pytest.raises(ExceptionGroup):
        await MicrosandboxRunner.create("removed-rollback", sandbox_module=FakeMicrosandboxModule)
    assert FakeSandboxApi.removed == ["removed-rollback"]
    FakeSandboxApi.created_stop_failure = None
    replacement = await MicrosandboxRunner.create(
        "removed-rollback", ensure_default_cwd=False, sandbox_module=FakeMicrosandboxModule
    )
    assert await MicrosandboxRunner.drain_failed_creations(timeout_s=1) == 0
    assert FakeSandboxApi.removed == ["removed-rollback"]
    assert replacement.lifecycle_state == "reusable"
    await replacement.close()


@pytest.mark.anyio
@pytest.mark.parametrize("stall_provider", [False, True])
@pytest.mark.parametrize("cancel_owner", [False, True])
async def test_terminal_close_preserves_transport_failure_before_stall(
    stall_provider, cancel_owner
):
    from cayu.runners._cleanup import runner_cancellation_failure

    reset_fake_module()
    entered = asyncio.Event()
    release = asyncio.Event()
    primary = RuntimeError("SFTP close failed")

    class Transport:
        async def close(self):
            raise primary

    class PendingTransport:
        async def close(self):
            entered.set()
            await release.wait()

    sandbox = FakeSandbox("terminal-progress")
    runner = MicrosandboxRunner(
        sandbox,
        name="terminal-progress",
        close_action="stop",
        cancel_timeout_s=0.03,
        remove_timeout_s=0.03,
        sandbox_module=FakeMicrosandboxModule,
    )
    runner._sftp = Transport()
    if stall_provider:

        async def stop():
            entered.set()
            await release.wait()

        sandbox.stop_and_wait = stop
    else:
        runner._sftp_client = PendingTransport()
    task = asyncio.create_task(runner.close())
    try:
        await asyncio.wait_for(entered.wait(), 10)
        if cancel_owner:
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert task.cancelled() and task.cancelling() == 1
            failure = runner_cancellation_failure(caught.value)
        else:
            with pytest.raises(ExceptionGroup) as caught:
                await task
            failure = caught.value
        assert isinstance(failure, ExceptionGroup)
        assert failure.exceptions[0] is primary and len(failure.exceptions) == 2
        assert isinstance(failure.exceptions[1], TimeoutError)
        with pytest.raises(RuntimeError):
            runner.reopen_exec()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if runner._terminal_lifecycle_task is not None:
            await asyncio.wait_for(asyncio.shield(runner._terminal_lifecycle_task), 1)


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

    async def stop_and_wait(self) -> None:
        await self.sandbox.stop_and_wait()

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
    start_calls: list[dict[str, Any]] = []

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
    async def start(cls, name: str, **kwargs: Any) -> FakeSandbox:
        cls.start_calls.append({"name": name, **kwargs})
        cls.registry_status = "running"
        if cls.existing is None:
            cls.existing = FakeSandbox(name)
        return cls.existing

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


@pytest.mark.anyio
@pytest.mark.parametrize("provider_fails", [False, True])
async def test_terminal_close_retains_failed_transports_and_attempts_provider(provider_fails):
    channel_error = RuntimeError("channel close failed")
    client_error = RuntimeError("client close failed")
    provider_error = RuntimeError("provider stop failed")

    class Resource:
        def __init__(self, failure):
            self.failure = failure
            self.calls = 0

        async def close(self):
            self.calls += 1
            if self.calls == 1:
                raise self.failure

    channel = Resource(channel_error)
    client = Resource(client_error)
    sandbox = FakeSandbox("terminal-transports")
    if provider_fails:
        sandbox.stop_failure = provider_error
    runner = MicrosandboxRunner(
        sandbox,
        name=sandbox.name,
        close_action="stop",
        sandbox_module=FakeMicrosandboxModule,
    )
    runner._sftp = channel
    runner._sftp_client = client
    with pytest.raises(BaseExceptionGroup) as captured:
        await runner.close()
    error = captured.value
    transport_error = error.exceptions[0] if provider_fails else error
    assert isinstance(transport_error, BaseExceptionGroup)
    assert transport_error.exceptions == (channel_error, client_error)
    if provider_fails:
        assert error.exceptions[1] is provider_error
    assert sandbox.stop_and_wait_calls == 1
    assert runner._sftp is channel and runner._sftp_client is client
    assert runner.lifecycle_state == "poisoned"
    await runner.close()
    assert channel.calls == client.calls == 2
    assert runner._sftp is runner._sftp_client is None
    assert runner.lifecycle_state == "closed"


@pytest.mark.anyio
async def test_terminal_sftp_cleanup_prevents_concurrent_reopening():
    entered = asyncio.Event()
    release = asyncio.Event()

    class Channel:
        async def close(self):
            entered.set()
            await release.wait()

        async def real_path(self, path):
            raise AssertionError("closing channel must not be reused")

    sandbox = FakeSandbox("terminal-sftp-race")
    runner = MicrosandboxRunner(
        sandbox, name=sandbox.name, close_action="none", sandbox_module=FakeMicrosandboxModule
    )
    runner._sftp = Channel()
    closing = asyncio.create_task(runner.close())
    await entered.wait()
    resolving = asyncio.create_task(runner.real_path("/workspace"))
    try:
        await asyncio.sleep(0)
        assert not resolving.done()
        release.set()
        await closing
        with pytest.raises(RuntimeError, match="closed"):
            await resolving
        assert runner._sftp is None
        assert runner.lifecycle_state == "closed"
    finally:
        release.set()
        await asyncio.gather(closing, resolving, return_exceptions=True)


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
    FakeSandboxApi.start_calls = []


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


@pytest.mark.parametrize("invalid_text", ("/workspace\x00bad", "/workspace\ud800bad"))
def test_microsandbox_runner_rejects_nonportable_default_cwd(invalid_text: str) -> None:
    with pytest.raises(ValueError, match="default_cwd"):
        MicrosandboxRunner(object(), name="sandbox", default_cwd=invalid_text)

    runner = MicrosandboxRunner(object(), name="sandbox")
    runner.default_cwd = invalid_text
    with pytest.raises(ValueError, match="default_cwd"):
        runner.resolve_cwd()


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


@pytest.mark.parametrize("public_method", ("exec", "exec_redacted"))
def test_microsandbox_owns_environment_before_waiting_for_exec_lock(
    public_method: str,
) -> None:
    sandbox = FakeSandbox("runner")
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        sandbox_module=FakeMicrosandboxModule,
    )
    environment = {"ORIGINAL": "value"}

    async def run() -> None:
        await runner._exec_lock.acquire()
        kwargs: dict[str, Any] = {"env": environment}
        if public_method == "exec_redacted":
            kwargs["redactor"] = SecretRedactor()
        task = asyncio.create_task(
            getattr(runner, public_method)(
                ExecCommand.process("python", "-c", "pass"),
                **kwargs,
            )
        )
        try:
            await asyncio.sleep(0)
            assert not task.done()
            environment.clear()
            environment["INVALID\x00NAME"] = "changed"
        finally:
            runner._exec_lock.release()
        await task

    asyncio.run(run())

    assert sandbox.exec_calls[0]["env"] == {"ORIGINAL": "value"}
    assert environment == {"INVALID\x00NAME": "changed"}


@pytest.mark.parametrize("public_method", ("exec", "exec_redacted"))
def test_microsandbox_owns_command_before_waiting_for_exec_lock(
    public_method: str,
) -> None:
    sandbox = FakeSandbox("runner")
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        sandbox_module=FakeMicrosandboxModule,
    )
    command = ExecCommand.process("python", "-c", "print('owned-command')")

    async def run() -> None:
        await runner._exec_lock.acquire()
        kwargs: dict[str, Any] = {}
        if public_method == "exec_redacted":
            kwargs["redactor"] = SecretRedactor()
        task = asyncio.create_task(getattr(runner, public_method)(command, **kwargs))
        try:
            await asyncio.sleep(0)
            assert not task.done()
            assert command.argv is not None
            command.argv[:] = ["mutated-command"]
        finally:
            runner._exec_lock.release()
        await task

    asyncio.run(run())

    assert sandbox.exec_calls[0]["cmd"] == "python"
    assert sandbox.exec_calls[0]["args"] == ["-c", "print('owned-command')"]


@pytest.mark.parametrize("public_method", ("exec", "exec_redacted"))
def test_microsandbox_owns_overlay_before_waiting_for_exec_lock(
    public_method: str,
) -> None:
    sandbox = FakeSandbox("runner")
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        env_overlay={"OVERLAY": "owned"},
        sandbox_module=FakeMicrosandboxModule,
    )

    async def run() -> None:
        await runner._exec_lock.acquire()
        kwargs: dict[str, Any] = {}
        if public_method == "exec_redacted":
            kwargs["redactor"] = SecretRedactor()
        task = asyncio.create_task(
            getattr(runner, public_method)(
                ExecCommand.process("python", "-c", "pass"),
                **kwargs,
            )
        )
        try:
            await asyncio.sleep(0)
            assert not task.done()
            runner.env_overlay.clear()
            runner.env_overlay["OVERLAY"] = "mutated"
            runner.env_overlay["INVALID\x00NAME"] = "changed"
        finally:
            runner._exec_lock.release()
        await task

    asyncio.run(run())

    assert sandbox.exec_calls[0]["env"] == {"OVERLAY": "owned"}
    assert runner.env_overlay == {
        "OVERLAY": "mutated",
        "INVALID\x00NAME": "changed",
    }


def test_microsandbox_runner_redacts_stream_events_before_bounding() -> None:
    secret = "microsandbox-stream-boundary-secret"
    sandbox = FakeSandbox("runner")
    sandbox.next_handle = FakeHandle(
        [
            FakeEvent("stdout", f"prefix:{secret[:11]}".encode()),
            FakeEvent("stdout", f"{secret[11:]}:suffix".encode()),
            FakeEvent("stderr", secret[:5].encode()),
            FakeEvent("stderr", secret[5:].encode()),
            FakeEvent("exited", code=0),
        ]
    )
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        sandbox_module=FakeMicrosandboxModule,
    )

    result = asyncio.run(
        runner.exec_redacted(
            ExecCommand.process("echo", "ignored"),
            redactor=SecretRedactor(secret),
            output_limit_bytes=128,
        )
    )

    assert result.stdout == f"prefix:{REDACTED_SECRET}:suffix"
    assert result.stderr == REDACTED_SECRET
    assert result.stdout_bytes == len(f"prefix:{secret}:suffix".encode())
    assert result.stderr_bytes == len(secret.encode())


def test_microsandbox_reconnect_passes_provider_credential_isolation_probe(
    provider_credential_canaries,
) -> None:
    class ProbeSandbox(FakeSandbox):
        async def exec_stream(self, cmd: str, args: list[str], **kwargs: Any) -> FakeHandle:
            self.exec_calls.append({"cmd": cmd, "args": args, **kwargs})
            payload = json.dumps(
                {
                    "environment": dict(kwargs.get("env", {})),
                    "auth_paths": {},
                    "auth_scan_complete": True,
                    "provider_canary_matches": [],
                    "detector_control_match": True,
                },
                sort_keys=True,
            ).encode()
            return FakeHandle(
                [FakeEvent("stdout", payload), FakeEvent("exited", code=0)],
                wait_result=(0, False),
            )

    async def run():
        sandbox = ProbeSandbox("credential-probe")
        FakeSandboxApi.existing = sandbox
        runner = await MicrosandboxRunner.from_existing(
            "credential-probe",
            sandbox_module=FakeMicrosandboxModule,
        )
        evidence = await verify_provider_credential_isolation(
            runner,
            adapter="microsandbox",
            scope="isolated_guest",
            provider_canaries=provider_credential_canaries.values,
            operational_env={
                "CAYU_PROBE_VISIBLE": provider_credential_canaries.positive_env[
                    "CAYU_PROBE_VISIBLE"
                ]
            },
            workload_env={
                "CAYU_WORKLOAD_TOKEN": provider_credential_canaries.positive_env[
                    "CAYU_WORKLOAD_TOKEN"
                ]
            },
            guest_cwd="/workspace",
            guest_auth_search_paths={"mounted_workspace": "/workspace"},
        )
        return evidence, sandbox

    evidence, sandbox = asyncio.run(run())

    assert evidence.status == "verified"
    assert "os.walk(root" in repr(sandbox.exec_calls[-1]["args"])
    assert provider_credential_canaries.positive_env.items() <= (
        sandbox.exec_calls[-1]["env"].items()
    )
    assert all(
        value not in repr(sandbox.exec_calls)
        for value in provider_credential_canaries.values.values()
    )
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
        },
        "probe": {
            "method": "Sandbox.ping",
            "status": "failed",
            "timeout_s": 1.0,
            "registry_status": "running",
            "error_type": "ConnectionResetError",
            "status_error_type": None,
        },
        "remediation": "Reconnect to or replace the Microsandbox before executing more commands.",
    }
    assert first.diagnostic == expected_diagnostic
    assert first.artifacts == [expected_diagnostic]
    assert first.__cause__ is None
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
        await asyncio.wait_for(ping_started.wait(), timeout=10.0)
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
        "status_error_type": None,
    }
    assert error.__cause__ is None


@pytest.mark.parametrize("policy", ["command", "sandbox"])
def test_microsandbox_runner_classifies_no_exit_event_when_agent_ping_fails(policy) -> None:
    async def run() -> RunnerExecutionError | MicrosandboxUnavailableError:
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
            cancellation_cleanup=policy,
            sandbox_module=FakeMicrosandboxModule,
        )
        expected = RunnerExecutionError if policy == "sandbox" else MicrosandboxUnavailableError
        with pytest.raises(expected) as exc_info:
            await runner.exec(ExecCommand.process("sleep", "30"))
        from cayu.runtime.egress import _workspace_dispatch_settlement_kind

        assert _workspace_dispatch_settlement_kind(result=None, error=exc_info.value) == (
            "runner_quiescent" if policy == "sandbox" else "complete"
        )
        assert sandbox.ping_calls == (0 if policy == "sandbox" else 1)
        assert (
            sum(item.get("type") == "cayu.runner_cleanup.v1" for item in exc_info.value.artifacts)
            == 1
        )
        assert (
            sum(
                item.get("type") == "cayu.runner_execution_error.v1"
                for item in exc_info.value.artifacts
            )
            == 1
        )
        return exc_info.value

    error = asyncio.run(run())

    if policy == "sandbox":
        assert error.diagnostic["type"] == "cayu.runner_execution_error.v1"
        return

    assert error.last_command == {
        "exit_code": None,
        "timed_out": False,
        "cancelled": False,
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "error_type": "FakeMicrosandboxError",
    }
    assert error.probe["registry_status"] == "stopped"
    assert error.probe["status"] == "failed"
    assert error.__cause__ is None


@pytest.mark.anyio
async def test_health_probe_cancellation_preserves_completed_command_cleanup():
    from cayu.runners._cleanup import runner_cancellation_failure
    from cayu.runtime.egress import _workspace_dispatch_settlement_kind

    reset_fake_module()
    sandbox = FakeSandbox("cancelled-probe")
    sandbox.next_handle = FakeHandle(
        [],
        collect_error=FakeMicrosandboxError("runtime error: exec session ended without exit event"),
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_ping():
        entered.set()
        await release.wait()

    sandbox.ping = blocked_ping
    runner = MicrosandboxRunner(
        sandbox, name="cancelled-probe", sandbox_module=FakeMicrosandboxModule
    )
    task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert task.cancelled() and task.cancelling() == 1
        failure = runner_cancellation_failure(caught.value)
        assert isinstance(failure, RunnerExecutionError)
        assert sum(item.get("type") == "cayu.runner_cleanup.v1" for item in failure.artifacts) == 1
        assert _workspace_dispatch_settlement_kind(result=None, error=caught.value) == "complete"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await runner.close()


def test_microsandbox_runner_sanitizes_no_exit_event_when_agent_ping_succeeds() -> None:
    sandbox = FakeSandbox("live-agent")
    command_error = FakeMicrosandboxError("runtime error: exec session ended without exit event")
    sandbox.next_handle = FakeHandle([], collect_error=command_error)
    runner = MicrosandboxRunner(
        sandbox,
        name="live-agent",
        sandbox_module=FakeMicrosandboxModule,
    )

    with pytest.raises(RunnerExecutionError) as exc_info:
        asyncio.run(runner.exec(ExecCommand.process("sleep", "30")))

    assert exc_info.value.diagnostic == {
        "type": "cayu.runner_execution_error.v1",
        "errno": None,
        "errno_code": None,
        "execution_phase": "unknown",
        "adapter": "microsandbox",
        "status": "failed",
        "error_type": "Exception",
        "timed_out": False,
        "cancelled": False,
        "stdout_bytes": 0,
        "stderr_bytes": 0,
    }
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sandbox.ping_calls == 1
    assert runner._exec_closed is False


def test_microsandbox_runner_sanitizes_other_exact_base_error_without_liveness_probe() -> None:
    sandbox = FakeSandbox("runner")
    command_error = FakeMicrosandboxError("protocol error: response ended mid-frame")
    sandbox.next_handle = FakeHandle([], collect_error=command_error)
    sandbox.ping_error = ConnectionResetError("agent connection reset")
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        sandbox_module=FakeMicrosandboxModule,
    )

    with pytest.raises(RunnerExecutionError) as exc_info:
        asyncio.run(runner.exec(ExecCommand.process("pwd")))

    assert exc_info.value.diagnostic["error_type"] == "Exception"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sandbox.ping_calls == 0
    assert runner._exec_closed is False


@pytest.mark.parametrize(
    "failure_message",
    [
        "workload-secret-canary-ABCDEFGHIJKLMNOP",
        "workload-secret-",
    ],
    ids=["complete-secret", "recoverable-prefix"],
)
def test_microsandbox_runner_detaches_opaque_sdk_failure_text(
    failure_message: str,
) -> None:
    sandbox = FakeSandbox("runner")
    sandbox.stream_error = FakeMicrosandboxError(failure_message)
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        sandbox_module=FakeMicrosandboxModule,
    )

    with pytest.raises(RunnerExecutionError) as exc_info:
        asyncio.run(runner.exec(ExecCommand.process("pwd")))

    assert str(exc_info.value) == "Runner command execution failed."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert exc_info.value.diagnostic["error_type"] == "Exception"
    assert failure_message not in repr(exc_info.value.diagnostic)


@pytest.mark.parametrize(
    "stream_error",
    [RuntimeError("sandbox is stopped"), ConnectionError("transport failed")],
    ids=["sandbox-stopped", "transport-failed"],
)
def test_microsandbox_runner_sanitizes_other_sdk_failures(
    stream_error: Exception,
) -> None:
    sandbox = FakeSandbox("runner")
    sandbox.stream_error = stream_error
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        sandbox_module=FakeMicrosandboxModule,
    )

    with pytest.raises(RunnerExecutionError) as exc_info:
        asyncio.run(runner.exec(ExecCommand.process("pwd")))

    assert exc_info.value.diagnostic["error_type"] == type(stream_error).__name__
    assert exc_info.value.__cause__ is None
    assert sandbox.ping_calls == 0
    assert runner.lifecycle_state == "poisoned"
    with pytest.raises(RuntimeError):
        runner.reopen_exec()


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
    original_sleep = asyncio.sleep

    async def unexpected_sleep(_delay: float) -> None:
        assert _delay == 0, "immediate removal must not back off"
        await original_sleep(0)

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

    # The first backoff exhausts this deadline; do not dispatch another
    # destructive operation merely to discover that time already expired.
    assert FakeSandboxApi.remove_calls == ["never-settles"]
    assert error.diagnostic["status"] == "timed_out"
    assert error.diagnostic["error_type"] == "FakeSandboxStillRunningError"
    assert runner.last_cleanup_diagnostic == error.diagnostic
    assert runner._closed is False


@pytest.mark.parametrize("stalled_operation", ["get", "refresh"])
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
        release_cleanup = asyncio.Event()

        if cancelled_operation == "remove":

            async def hanging_remove(cls: type[FakeSandboxApi], name: str) -> None:
                cls.remove_calls.append(name)
                cancellation_started.set()
                await release_cleanup.wait()

            monkeypatch.setattr(FakeSandboxApi, "remove", classmethod(hanging_remove))
        else:
            FakeSandboxApi.remove_failures = [
                FakeSandboxStillRunningError("sandbox status has not settled")
            ]
            if cancelled_operation == "get":

                async def hanging_get(cls: type[FakeSandboxApi], name: str) -> FakeHandleRecord:
                    cancellation_started.set()
                    await release_cleanup.wait()
                    return FakeHandleRecord(cls.existing or FakeSandbox(name), status="stopped")

                monkeypatch.setattr(FakeSandboxApi, "get", classmethod(hanging_get))
            elif cancelled_operation == "refresh":

                async def hanging_refresh(self: FakeHandleRecord) -> FakeHandleRecord:
                    cancellation_started.set()
                    await release_cleanup.wait()
                    return self

                monkeypatch.setattr(FakeHandleRecord, "refresh", hanging_refresh)
            else:

                async def hanging_backoff(_delay_s: float) -> None:
                    cancellation_started.set()
                    await release_cleanup.wait()

                monkeypatch.setattr(
                    "cayu.runners.microsandbox._sleep_before_microsandbox_retry",
                    hanging_backoff,
                )

        runner = await MicrosandboxRunner.create(
            f"cancel-removal-{cancelled_operation}",
            sandbox_module=FakeMicrosandboxModule,
        )
        close_task = asyncio.create_task(runner.close())
        await asyncio.wait_for(cancellation_started.wait(), timeout=10)
        close_task.cancel("first")
        await asyncio.sleep(0)
        close_task.cancel("second")
        assert not close_task.done()
        assert runner.lifecycle_state == "closing"
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await close_task
        assert close_task.cancelled()
        assert close_task.cancelling() == 2
        assert exc_info.value.args == ("first",)
        return runner, exc_info.value

    runner, error = asyncio.run(run())

    diagnostic = cleanup_diagnostic(error)
    assert runner.last_cleanup_diagnostic == diagnostic
    assert diagnostic["action"] == "remove"
    assert diagnostic["status"] == "removed"
    assert diagnostic["attempts"][-1]["status"] == "removed"
    assert runner.is_closed


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


@pytest.mark.parametrize(
    "options",
    [
        {"cancel_timeout_s": 0},
        {"cancel_timeout_s": -1},
        {"cancel_timeout_s": inf},
        {"cancel_timeout_s": nan},
        {"cancel_timeout_s": True},
        {"env_overlay": object()},
    ],
)
@pytest.mark.parametrize("factory", ["create", "from_existing"])
def test_microsandbox_create_validates_constructor_inputs_before_allocation(
    options: dict[str, Any],
    factory: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_get(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Invalid local configuration reached provider lookup.")

    monkeypatch.setattr(FakeSandboxApi, "get", unexpected_get)

    async def run() -> None:
        reset_fake_module()
        with pytest.raises((TypeError, ValueError)):
            await getattr(MicrosandboxRunner, factory)(
                "invalid", sandbox_module=FakeMicrosandboxModule, **options
            )
        assert FakeSandboxApi.created == []

    asyncio.run(run())


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
        assert len(errors) == 2
        assert isinstance(errors[0], RuntimeError)
        assert isinstance(errors[1], BaseExceptionGroup)
        stop_error, removal_error = errors[1].exceptions
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
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await MicrosandboxRunner.create(
                "setup-cancelled-stop-fails",
                sandbox_module=FakeMicrosandboxModule,
            )
        from cayu.runners._cleanup import runner_cancellation_failure

        stop_error = runner_cancellation_failure(exc_info.value)
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


def test_microsandbox_runner_restarts_stopped_sandbox_before_reconnect() -> None:
    class SandboxStatus(StrEnum):
        STOPPED = "stopped"

    async def run() -> MicrosandboxRunner:
        reset_fake_module()
        FakeSandboxApi.existing = FakeSandbox("stopped-existing")
        FakeSandboxApi.registry_status = SandboxStatus.STOPPED
        return await MicrosandboxRunner.from_existing(
            "stopped-existing",
            sandbox_module=FakeMicrosandboxModule,
        )

    runner = asyncio.run(run())

    assert runner.name == "stopped-existing"
    assert FakeSandboxApi.start_calls == [{"name": "stopped-existing", "detached": True}]
    assert runner.restarted_from_stopped is True


def test_microsandbox_runner_removal_attachment_remains_executable() -> None:
    async def run() -> int:
        reset_fake_module()
        sandbox = FakeSandbox("stopped-removal")
        FakeSandboxApi.existing = sandbox
        FakeSandboxApi.registry_status = "stopped"
        runner = await MicrosandboxRunner.from_existing(
            "stopped-removal",
            close_action="remove",
            sandbox_module=FakeMicrosandboxModule,
        )
        result = await runner.exec(ExecCommand.process("true"))
        await runner.close()
        return result.exit_code

    exit_code = asyncio.run(run())

    assert exit_code == 7
    assert FakeSandboxApi.start_calls == [{"name": "stopped-removal", "detached": True}]
    assert FakeSandboxApi.remove_calls == ["stopped-removal"]


def test_microsandbox_runner_restops_allocation_when_restart_reattest_fails() -> None:
    class FailingSecondGetSandboxApi(FakeSandboxApi):
        get_calls = 0

        @classmethod
        async def get(cls, name: str) -> FakeHandleRecord:
            cls.get_calls += 1
            if cls.get_calls == 2:
                raise RuntimeError("restarted identity lookup failed")
            return await super().get(name)

    class FailingSecondGetModule(FakeMicrosandboxModule):
        Sandbox = FailingSecondGetSandboxApi

    async def run() -> FakeSandbox:
        reset_fake_module()
        FailingSecondGetSandboxApi.get_calls = 0
        sandbox = FakeSandbox("failed-restart")
        FakeSandboxApi.existing = sandbox
        FakeSandboxApi.registry_status = "stopped"
        with pytest.raises(RuntimeError, match="identity lookup failed"):
            await MicrosandboxRunner.from_existing(
                "failed-restart",
                sandbox_module=FailingSecondGetModule,
            )
        return sandbox

    sandbox = asyncio.run(run())

    assert FakeSandboxApi.start_calls == [{"name": "failed-restart", "detached": True}]
    assert sandbox.stop_and_wait_calls == 1


@pytest.mark.parametrize(
    "first_restoration_outcome",
    ["failed", "child_cancelled", "grouped_failure"],
)
def test_microsandbox_runner_retries_failed_restart_restoration_owner(
    first_restoration_outcome: str,
) -> None:
    retry_started = asyncio.Event()
    allow_retry = asyncio.Event()

    class RecoverableStopSandbox(FakeSandbox):
        async def stop_and_wait(self) -> None:
            self.stop_and_wait_calls += 1
            if self.stop_and_wait_calls == 1:
                if first_restoration_outcome == "child_cancelled":
                    raise asyncio.CancelledError("provider stop cancelled itself")
                if first_restoration_outcome == "grouped_failure":
                    raise BaseExceptionGroup(
                        "provider stop failed as a group",
                        [
                            asyncio.CancelledError("provider child cancelled itself"),
                            ConnectionError("transient grouped stop failure"),
                        ],
                    )
                raise ConnectionError("restoration stop failed")
            retry_started.set()
            await allow_retry.wait()

    class FailingSecondGetSandboxApi(FakeSandboxApi):
        get_calls = 0

        @classmethod
        async def get(cls, name: str) -> FakeHandleRecord:
            cls.get_calls += 1
            if cls.get_calls == 2:
                raise RuntimeError("restarted identity lookup failed")
            return await super().get(name)

    class FailingSecondGetModule(FakeMicrosandboxModule):
        Sandbox = FailingSecondGetSandboxApi

    async def run() -> tuple[FakeSandbox, BaseException, asyncio.Task[None]]:
        reset_fake_module()
        FailingSecondGetSandboxApi.get_calls = 0
        sandbox = RecoverableStopSandbox("failed-restart-restoration")
        FakeSandboxApi.existing = sandbox
        FakeSandboxApi.registry_status = "stopped"
        with pytest.raises(BaseExceptionGroup, match="restoring") as exc_info:
            await MicrosandboxRunner.from_existing(
                "failed-restart-restoration",
                sandbox_module=FailingSecondGetModule,
            )
        settlement_task = microsandbox_reconnect_settlement_task(exc_info.value)
        assert settlement_task is not None
        assert not settlement_task.done()
        await retry_started.wait()
        assert not settlement_task.done()
        allow_retry.set()
        async with asyncio.timeout(0.2):
            await asyncio.shield(settlement_task)
        return sandbox, exc_info.value, settlement_task

    sandbox, error, settlement_task = asyncio.run(run())

    assert settlement_task.done()
    assert not settlement_task.cancelled()
    if first_restoration_outcome == "child_cancelled":
        assert any(isinstance(child, asyncio.CancelledError) for child in error.exceptions)
    elif first_restoration_outcome == "grouped_failure":
        restoration_group = next(
            child for child in error.exceptions if isinstance(child, BaseExceptionGroup)
        )
        assert any(
            isinstance(child, asyncio.CancelledError) for child in restoration_group.exceptions
        )
        assert any(
            isinstance(child, ConnectionError) and str(child) == "transient grouped stop failure"
            for child in restoration_group.exceptions
        )
    else:
        assert any(
            isinstance(child, ConnectionError) and str(child) == "restoration stop failed"
            for child in error.exceptions
        )
    assert sandbox.stop_and_wait_calls == 2


def test_microsandbox_restoration_does_not_loop_on_permanent_failure() -> None:
    async def run() -> tuple[int, int]:
        class PermissionDeniedStopSandbox:
            def __init__(self) -> None:
                self.stop_calls = 0
                self.allow_stop = False

            async def stop_and_wait(self) -> None:
                self.stop_calls += 1
                if not self.allow_stop:
                    raise PermissionError("provider credentials rejected")

        sandbox = PermissionDeniedStopSandbox()
        failed_settlement = _defer_reconnect_restoration(
            FakeMicrosandboxModule,
            sandbox,
            "permanent-restoration-failure",
        )
        with pytest.raises(PermissionError, match="credentials rejected"):
            await failed_settlement
        calls_after_failure = sandbox.stop_calls
        await asyncio.sleep(0.12)
        assert sandbox.stop_calls == calls_after_failure

        sandbox.allow_stop = True
        recovered_settlement = _defer_reconnect_restoration(
            FakeMicrosandboxModule,
            sandbox,
            "explicit-restoration-recovery",
        )
        await recovered_settlement
        return calls_after_failure, sandbox.stop_calls

    calls_after_failure, calls_after_recovery = asyncio.run(run())

    assert calls_after_failure == 1
    assert calls_after_recovery == 2


@pytest.mark.parametrize("failure_kind", ["forged_retryable", "ambiguous_io"])
def test_microsandbox_restoration_rejects_ambiguous_retry_evidence(
    failure_kind: str,
) -> None:
    class FakeIoError(RuntimeError):
        pass

    class ModuleWithIoError(FakeMicrosandboxModule):
        IoError = FakeIoError

    async def run() -> int:
        class AmbiguousStopSandbox:
            def __init__(self) -> None:
                self.stop_calls = 0

            async def stop_and_wait(self) -> None:
                self.stop_calls += 1
                if failure_kind == "ambiguous_io":
                    raise FakeIoError("provider reported generic I/O failure")
                error = RuntimeError("extension supplied retryable attribute")
                error.retryable = True
                raise error

        sandbox = AmbiguousStopSandbox()
        settlement = _defer_reconnect_restoration(
            ModuleWithIoError,
            sandbox,
            f"ambiguous-{failure_kind}",
        )
        expected = FakeIoError if failure_kind == "ambiguous_io" else RuntimeError
        with pytest.raises(expected):
            await settlement
        await asyncio.sleep(0.12)
        return sandbox.stop_calls

    assert asyncio.run(run()) == 1


def test_microsandbox_restoration_owner_cancellation_stops_retries() -> None:
    async def run() -> tuple[asyncio.Task[None], int]:
        retry_started = asyncio.Event()

        class BlockingStopSandbox:
            def __init__(self) -> None:
                self.stop_calls = 0

            async def stop_and_wait(self) -> None:
                self.stop_calls += 1
                retry_started.set()
                await asyncio.Event().wait()

        sandbox = BlockingStopSandbox()
        settlement_task = _defer_reconnect_restoration(
            FakeMicrosandboxModule,
            sandbox,
            "cancelled-restoration-owner",
        )
        await retry_started.wait()
        settlement_task.cancel("operator stopped restoration")
        with pytest.raises(asyncio.CancelledError, match="operator stopped restoration"):
            await settlement_task
        await asyncio.sleep(0.1)
        return settlement_task, sandbox.stop_calls

    settlement_task, stop_calls = asyncio.run(run())

    assert settlement_task.cancelled()
    assert settlement_task.cancelling() == 1
    assert stop_calls == 1


def test_microsandbox_reconnect_settlement_handoff_ignores_exception_descriptors_and_forgery() -> (
    None
):
    attribute_name = "_cayu_microsandbox_reconnect_settlement_task"

    class DescriptorControlledError(RuntimeError):
        descriptor_reads = 0

        @property
        def _cayu_microsandbox_reconnect_settlement_task(self) -> object:
            type(self).descriptor_reads += 1
            raise RuntimeError("workload-secret-from-reconnect-descriptor")

    async def scenario() -> None:
        task = asyncio.create_task(asyncio.sleep(0))
        unattached = DescriptorControlledError("unattached provider failure")
        assert microsandbox_reconnect_settlement_task(unattached) is None

        forged = DescriptorControlledError("forged provider failure")
        forged.__dict__[attribute_name] = task
        assert microsandbox_reconnect_settlement_task(forged) is None
        await task

    asyncio.run(scenario())
    assert DescriptorControlledError.descriptor_reads == 0


def test_microsandbox_runner_bounds_failed_restart_restoration() -> None:
    stop_started = asyncio.Event()
    allow_stop = asyncio.Event()

    class HangingStopSandbox(FakeSandbox):
        async def stop_and_wait(self) -> None:
            self.stop_and_wait_calls += 1
            stop_started.set()
            await allow_stop.wait()

    class FailingSecondGetSandboxApi(FakeSandboxApi):
        get_calls = 0

        @classmethod
        async def get(cls, name: str) -> FakeHandleRecord:
            cls.get_calls += 1
            if cls.get_calls == 2:
                raise RuntimeError("restarted identity lookup failed")
            return await super().get(name)

    class FailingSecondGetModule(FakeMicrosandboxModule):
        Sandbox = FailingSecondGetSandboxApi

    async def run() -> tuple[BaseException, asyncio.Task[None]]:
        reset_fake_module()
        FailingSecondGetSandboxApi.get_calls = 0
        sandbox = HangingStopSandbox("hanging-restart-restoration")
        FakeSandboxApi.existing = sandbox
        FakeSandboxApi.registry_status = "stopped"
        with pytest.raises(BaseExceptionGroup, match="restoring") as exc_info:
            async with asyncio.timeout(0.2):
                await MicrosandboxRunner.from_existing(
                    "hanging-restart-restoration",
                    reconnect_timeout_s=0.01,
                    sandbox_module=FailingSecondGetModule,
                )
        settlement_task = microsandbox_reconnect_settlement_task(exc_info.value)
        assert settlement_task is not None
        assert stop_started.is_set()
        assert not settlement_task.done()
        with pytest.raises(RuntimeError, match="acquisition or rollback is still pending"):
            await MicrosandboxRunner.create(
                "hanging-restart-restoration", replace=True, sandbox_module=FailingSecondGetModule
            )
        with pytest.raises(RuntimeError, match="acquisition or rollback is still pending"):
            await MicrosandboxRunner.from_existing(
                "hanging-restart-restoration", sandbox_module=FailingSecondGetModule
            )
        allow_stop.set()
        async with asyncio.timeout(0.2):
            await asyncio.shield(settlement_task)
        replacement = await MicrosandboxRunner.create(
            "hanging-restart-restoration",
            ensure_default_cwd=False,
            close_action="none",
            sandbox_module=FailingSecondGetModule,
        )
        assert replacement.lifecycle_state == "reusable"
        await replacement.close()
        return exc_info.value, settlement_task

    error, settlement_task = asyncio.run(run())

    assert any(isinstance(child, TimeoutError) for child in getattr(error, "exceptions", ()))
    assert settlement_task.done()
    assert not settlement_task.cancelled()


@pytest.mark.parametrize("late_start", [False, True])
def test_direct_attachment_restoration_can_be_drained_after_provider_repair(
    late_start: bool,
) -> None:
    async def run() -> None:
        reset_fake_module()
        allow_start = asyncio.Event()
        allow_initial_stop = asyncio.Event()
        retry_started = asyncio.Event()
        allow_retry = asyncio.Event()
        repaired = False

        class RestorationSandbox(FakeSandbox):
            async def stop_and_wait(self) -> None:
                self.stop_and_wait_calls += 1
                if not repaired:
                    await allow_initial_stop.wait()
                    raise PermissionError("restoration denied")
                retry_started.set()
                await allow_retry.wait()

        sandbox = RestorationSandbox(f"direct-restoration-{late_start}")

        class RestorationApi(FakeSandboxApi):
            get_calls = 0

            @classmethod
            async def get(cls, name: str) -> FakeHandleRecord:
                cls.get_calls += 1
                if not late_start and cls.get_calls == 2:
                    raise RuntimeError("post-start identity lookup failed")
                return await super().get(name)

            @classmethod
            async def start(cls, name: str, **kwargs: Any) -> FakeSandbox:
                result = await super().start(name, **kwargs)
                if late_start:
                    await allow_start.wait()
                return result

        class RestorationModule(FakeMicrosandboxModule):
            Sandbox = RestorationApi

        FakeSandboxApi.existing = sandbox
        FakeSandboxApi.registry_status = "stopped"
        try:
            with pytest.raises(BaseException) as failure:
                async with asyncio.timeout(1):
                    await MicrosandboxRunner.from_existing(
                        sandbox.name,
                        reconnect_timeout_s=0.01,
                        sandbox_module=RestorationModule,
                    )
            settlement = microsandbox_reconnect_settlement_task(failure.value)
            assert settlement is not None
            allow_start.set()
            allow_initial_stop.set()
            with pytest.raises(PermissionError, match="restoration denied"):
                async with asyncio.timeout(1):
                    await asyncio.shield(settlement)
            assert await MicrosandboxRunner.drain_failed_attachments(timeout_s=0.1) == 1
            with pytest.raises(RuntimeError, match="acquisition or rollback"):
                await MicrosandboxRunner.from_existing(
                    sandbox.name, sandbox_module=RestorationModule
                )
            repaired = True
            drain = asyncio.create_task(MicrosandboxRunner.drain_failed_attachments(timeout_s=1))
            async with asyncio.timeout(1):
                await retry_started.wait()
            calls = sandbox.stop_and_wait_calls
            assert await MicrosandboxRunner.drain_failed_attachments(timeout_s=0.01) == 1
            assert sandbox.stop_and_wait_calls == calls
            drain.cancel("stop waiting for restoration")
            with pytest.raises(asyncio.CancelledError, match="stop waiting"):
                await drain
            assert drain.cancelled() and drain.cancelling() == 1
            with pytest.raises(RuntimeError, match="acquisition or rollback"):
                await MicrosandboxRunner.create(
                    sandbox.name, replace=True, sandbox_module=RestorationModule
                )
            assert len(FakeSandboxApi.start_calls) == 1
            allow_retry.set()
            assert await MicrosandboxRunner.drain_failed_attachments(timeout_s=1) == 0
            assert sandbox.stop_and_wait_calls == calls
            runner = await MicrosandboxRunner.create(
                sandbox.name,
                ensure_default_cwd=False,
                close_action="none",
                sandbox_module=RestorationModule,
            )
            assert runner.lifecycle_state == "reusable"
            await runner.close()
        finally:
            repaired = True
            allow_start.set()
            allow_initial_stop.set()
            allow_retry.set()
            await MicrosandboxRunner.drain_failed_attachments(timeout_s=1)

    asyncio.run(run())


def test_microsandbox_runner_cancelled_restart_waits_for_start_then_restops() -> None:
    start_accepted = asyncio.Event()
    allow_start_return = asyncio.Event()

    class BlockingStartSandboxApi(FakeSandboxApi):
        @classmethod
        async def start(cls, name: str, **kwargs: Any) -> FakeSandbox:
            cls.start_calls.append({"name": name, **kwargs})
            cls.registry_status = "running"
            if cls.existing is None:
                cls.existing = FakeSandbox(name)
            start_accepted.set()
            await allow_start_return.wait()
            return cls.existing

    class BlockingStartModule(FakeMicrosandboxModule):
        Sandbox = BlockingStartSandboxApi

    async def run() -> tuple[FakeSandbox, asyncio.Task[MicrosandboxRunner]]:
        reset_fake_module()
        sandbox = FakeSandbox("cancelled-restart")
        FakeSandboxApi.existing = sandbox
        FakeSandboxApi.registry_status = "stopped"
        task = asyncio.create_task(
            MicrosandboxRunner.from_existing(
                "cancelled-restart",
                sandbox_module=BlockingStartModule,
            )
        )
        await start_accepted.wait()
        task.cancel("caller stopped waiting")
        await asyncio.sleep(0)
        assert not task.done()
        allow_start_return.set()
        with pytest.raises(asyncio.CancelledError, match="caller stopped waiting"):
            await task
        return sandbox, task

    sandbox, task = asyncio.run(run())

    assert sandbox.stop_and_wait_calls == 1
    assert task.cancelled()
    assert task.cancelling() == 1


@pytest.mark.parametrize("restoration_fails", [False, True])
def test_microsandbox_runner_preserves_restart_failure_after_cancellation(
    restoration_fails,
) -> None:
    from cayu.runners._cleanup import runner_cancellation_failure

    start_accepted = asyncio.Event()
    allow_start_failure = asyncio.Event()
    stop_started = asyncio.Event()
    allow_stop = asyncio.Event()

    class FailingCancelledStartSandboxApi(FakeSandboxApi):
        @classmethod
        async def start(cls, name: str, **kwargs: Any) -> FakeSandbox:
            cls.start_calls.append({"name": name, **kwargs})
            cls.registry_status = "running"
            if cls.existing is None:
                cls.existing = FakeSandbox(name)
            start_accepted.set()
            await allow_start_failure.wait()
            raise RuntimeError("provider restart failed")

    class FailingCancelledStartModule(FakeMicrosandboxModule):
        Sandbox = FailingCancelledStartSandboxApi

    async def run() -> tuple[asyncio.CancelledError, asyncio.Task[MicrosandboxRunner]]:
        reset_fake_module()
        sandbox = FakeSandbox("cancelled-failed-restart")
        original_stop = sandbox.stop_and_wait

        async def failed_stop():
            sandbox.stop_and_wait_calls += 1
            stop_started.set()
            await allow_stop.wait()
            raise PermissionError("restoration denied")

        if restoration_fails:
            sandbox.stop_and_wait = failed_stop
        FakeSandboxApi.existing = sandbox
        FakeSandboxApi.registry_status = "stopped"
        task = asyncio.create_task(
            MicrosandboxRunner.from_existing(
                "cancelled-failed-restart",
                sandbox_module=FailingCancelledStartModule,
            )
        )
        await start_accepted.wait()
        task.cancel("caller stopped waiting")
        await asyncio.sleep(0)
        allow_start_failure.set()
        if restoration_fails:
            await stop_started.wait()
            task.cancel("second cancellation during restoration")
            await asyncio.sleep(0)
            allow_stop.set()
        with pytest.raises(asyncio.CancelledError, match="caller stopped waiting") as exc_info:
            await task
        assert sandbox.stop_and_wait_calls == 1
        if restoration_fails:
            settlement = microsandbox_reconnect_settlement_task(exc_info.value)
            assert settlement is not None
            with pytest.raises(PermissionError):
                await settlement
            sandbox.stop_and_wait = original_stop
            assert await MicrosandboxRunner.drain_failed_attachments(timeout_s=1) == 0
        return exc_info.value, task

    error, task = asyncio.run(run())

    evidence = runner_cancellation_failure(error)
    if restoration_fails:
        assert isinstance(evidence, ExceptionGroup)
        assert len(evidence.exceptions) == 2
        assert isinstance(evidence.exceptions[1], PermissionError)
        evidence = evidence.exceptions[0]
    assert isinstance(evidence, RuntimeError)
    assert str(evidence) == "provider restart failed"
    assert task.cancelled()
    assert task.cancelling() == (2 if restoration_fails else 1)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("restoration_fails", [False, True])
def test_reconnect_fatal_failure_survives_cancellation_during_restoration(
    grouped: bool, restoration_fails: bool
) -> None:
    class FatalProviderSignal(BaseException):
        pass

    original: BaseException = (
        BaseExceptionGroup("fatal restart", [SystemExit(7), KeyboardInterrupt()])
        if grouped
        else FatalProviderSignal("fatal restart")
    )
    cleanup_error = PermissionError("restoration denied")

    class FatalStartApi(FakeSandboxApi):
        @classmethod
        async def start(cls, name: str, **kwargs: Any) -> FakeSandbox:
            cls.registry_status = "running"
            raise original

    class FatalStartModule(FakeMicrosandboxModule):
        Sandbox = FatalStartApi

    async def run() -> None:
        reset_fake_module()
        sandbox = FakeSandbox("fatal-restart")
        original_stop = sandbox.stop_and_wait
        stop_started = asyncio.Event()
        allow_stop = asyncio.Event()

        async def blocked_stop() -> None:
            stop_started.set()
            await allow_stop.wait()
            if restoration_fails:
                raise cleanup_error
            await original_stop()

        sandbox.stop_and_wait = blocked_stop
        FakeSandboxApi.existing = sandbox
        FakeSandboxApi.registry_status = "stopped"
        task = asyncio.create_task(
            MicrosandboxRunner.from_existing("fatal-restart", sandbox_module=FatalStartModule)
        )
        try:
            await stop_started.wait()
            task.cancel("cancel during restoration")
            await asyncio.sleep(0)
            assert not task.done()
            allow_stop.set()
            with pytest.raises(BaseExceptionGroup) as exc_info:
                await task
            failures = exc_info.value.exceptions
            assert failures[0] is original
            assert len(failures) == (3 if restoration_fails else 2)
            if restoration_fails:
                assert failures[1] is cleanup_error
            assert isinstance(failures[-1], asyncio.CancelledError)
            assert str(failures[-1]) == "cancel during restoration"
            assert not task.cancelled()
            assert task.cancelling() == 1
            settlement = microsandbox_reconnect_settlement_task(exc_info.value)
            assert settlement is not None
            if restoration_fails:
                with pytest.raises(PermissionError) as settled:
                    await settlement
                assert settled.value is cleanup_error
                with pytest.raises(RuntimeError, match="pending"):
                    await MicrosandboxRunner.from_existing(
                        "fatal-restart", sandbox_module=FatalStartModule
                    )
            else:
                await settlement
        finally:
            allow_stop.set()
            await asyncio.gather(task, return_exceptions=True)
            sandbox.stop_and_wait = original_stop
            assert await MicrosandboxRunner.drain_failed_attachments(timeout_s=1) == 0

    asyncio.run(run())


def test_microsandbox_runner_restart_timeout_returns_while_retaining_stop_owner() -> None:
    start_accepted = asyncio.Event()
    allow_start_return = asyncio.Event()

    class BlockingStartSandboxApi(FakeSandboxApi):
        @classmethod
        async def start(cls, name: str, **kwargs: Any) -> FakeSandbox:
            cls.start_calls.append({"name": name, **kwargs})
            cls.registry_status = "running"
            if cls.existing is None:
                cls.existing = FakeSandbox(name)
            start_accepted.set()
            await allow_start_return.wait()
            return cls.existing

    class BlockingStartModule(FakeMicrosandboxModule):
        Sandbox = BlockingStartSandboxApi

    async def run() -> FakeSandbox:
        reset_fake_module()
        sandbox = FakeSandbox("timed-out-restart")
        FakeSandboxApi.existing = sandbox
        FakeSandboxApi.registry_status = "stopped"

        with pytest.raises(TimeoutError, match="did not attach") as exc_info:
            async with asyncio.timeout(0.2):
                await MicrosandboxRunner.from_existing(
                    "timed-out-restart",
                    reconnect_timeout_s=0.01,
                    sandbox_module=BlockingStartModule,
                )

        settlement_task = microsandbox_reconnect_settlement_task(exc_info.value)
        assert settlement_task is not None
        assert start_accepted.is_set()
        assert not settlement_task.done()
        assert sandbox.stop_and_wait_calls == 0

        allow_start_return.set()
        async with asyncio.timeout(0.2):
            await asyncio.shield(settlement_task)
        return sandbox

    sandbox = asyncio.run(run())

    assert sandbox.stop_and_wait_calls == 1
    assert FakeSandboxApi.start_calls == [{"name": "timed-out-restart", "detached": True}]


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


@pytest.mark.anyio
@pytest.mark.parametrize("timeout", [False, True])
@pytest.mark.parametrize("close_fails", [False, True])
@pytest.mark.parametrize("kill_fails", [False, True])
@pytest.mark.parametrize("close_stalls", [False, True, "after_channel_failure"])
async def test_sandbox_command_cleanup_finalizes_open_transports(
    timeout, close_fails, kill_fails, close_stalls
):
    from cayu.runners.base import runner_workspace_mutation_settlement
    from cayu.runtime.egress import _workspace_dispatch_settlement_kind

    closed = []
    errors = []
    release_close = asyncio.Event()
    client_closed = asyncio.Event()

    class Transport:
        def __init__(self, name):
            self.name = name

        async def close(self):
            closed.append(self.name)
            if close_stalls and (close_stalls != "after_channel_failure" or self.name == "client"):
                await release_close.wait()
            if self.name == "client":
                client_closed.set()
            if close_fails or (close_stalls == "after_channel_failure" and self.name == "channel"):
                failure = RuntimeError("transport close failed")
                errors.append(failure)
                raise failure

        async def real_path(self, path):
            return path

        async def sftp(self):
            return channel

        async def open_client(self, **kwargs):
            return client

    sandbox = FakeSandbox("runner")
    sandbox.fail_kill = kill_fails
    handle = BlockingHandle()
    sandbox.next_handle = handle
    runner = MicrosandboxRunner(
        sandbox,
        name="runner",
        cancellation_cleanup="sandbox",
        timeout_cleanup="sandbox",
        cancel_timeout_s=0.05 if close_stalls else 5,
        sandbox_module=FakeMicrosandboxModule,
    )
    channel = Transport("channel")
    client = Transport("client")
    sandbox.ssh = lambda: client
    assert await runner.real_path("/workspace") == "/workspace"
    task = asyncio.create_task(
        runner.exec(ExecCommand.process("sleep", "30"), timeout_s=1 if timeout else None)
    )
    await asyncio.wait_for(handle.started.wait(), 10)
    if timeout:
        result = await task
        assert result.timed_out
        artifacts = result.artifacts
    else:
        task.cancel("owner")
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert task.cancelled() and task.cancelling() == 1
        artifacts = caught.value.artifacts
    for classify in (runner_workspace_mutation_settlement, _workspace_dispatch_settlement_kind):
        assert classify(
            result=result if timeout else None, error=None if timeout else caught.value
        ) == ("uncertain" if kill_fails else "runner_quiescent")
        assert (
            classify(
                result=ExecResult(
                    timed_out=True,
                    artifacts=[
                        item for item in artifacts if item.get("action") == "close_transports"
                    ],
                ),
                error=None,
            )
            == "uncertain"
        )
    if close_stalls:
        try:
            partial_failure = close_stalls == "after_channel_failure"
            assert closed == (["channel", "client"] if partial_failure else ["channel"])
            assert runner.lifecycle_state == "poisoned"
            assert [item["action"] for item in artifacts] == ["kill_sandbox"] + [
                "close_transports"
            ] * (2 if partial_failure else 1)
            assert artifacts[0]["status"] == ("failed" if kill_fails else "completed")
            assert artifacts[-1]["status"] == "timeout"
            if partial_failure:
                assert len(errors) == 1
                assert artifacts[1]["status"] == "failed"
                assert artifacts[1]["error_type"] == "RuntimeError"
                assert sanitize_runner_artifacts(artifacts) == artifacts
                if not timeout:
                    assert runner_cancellation_failure(caught.value) is errors[0]
            with pytest.raises(RuntimeError):
                runner.reopen_exec()
        finally:
            release_close.set()
            await asyncio.wait_for(client_closed.wait(), 1)
        return
    assert closed == ["channel", "client"]
    assert sandbox.kill_calls == (0 if kill_fails else 1)
    assert runner.lifecycle_state == ("poisoned" if close_fails or kill_fails else "closed")
    assert artifacts[0]["action"] == "kill_sandbox"
    assert artifacts[0]["status"] == ("failed" if kill_fails else "completed")
    if kill_fails:
        assert artifacts[0]["error_type"] == "RuntimeError"
    if close_fails:
        assert len(artifacts) == 2
        assert artifacts[1]["action"] == "close_transports"
        assert artifacts[1]["status"] == "failed"
        assert sanitize_runner_artifacts(artifacts) == artifacts
        if not timeout:
            failure = runner_cancellation_failure(caught.value)
            assert isinstance(failure, BaseExceptionGroup)
            assert failure.exceptions == tuple(errors)
    if not close_fails and not kill_fails:
        await runner.close()
        assert closed == ["channel", "client"]


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
        }
    ]


def test_microsandbox_runner_bounds_hanging_command_kill_on_cancellation() -> None:
    async def run() -> tuple[FakeSandbox, BlockingHandle, BaseException]:
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
        with pytest.raises(RuntimeError, match="closed"):
            await runner.exec(ExecCommand.process("pwd"))
        assert runner.lifecycle_state == "poisoned"
        return sandbox, handle, exc_info.value

    sandbox, handle, exc = asyncio.run(run())

    assert handle.killed is False
    assert sandbox.kill_calls == 0
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "kill_command",
            "status": "timeout",
            "timeout_s": 0.01,
        }
    ]


def test_microsandbox_runner_is_poisoned_when_command_kill_fails() -> None:
    async def run() -> tuple[BlockingHandle, BaseException]:
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
        with pytest.raises(RuntimeError, match="closed"):
            await runner.exec(ExecCommand.process("pwd"))
        with pytest.raises(RuntimeError, match="permanently poisoned"):
            runner.reopen_exec()
        return handle, exc_info.value

    handle, exc = asyncio.run(run())

    assert handle.killed is False
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "kill_command",
            "status": "failed",
            "timeout_s": 5.0,
            "error_type": "RuntimeError",
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
    async def run() -> tuple[FakeSandbox, Any]:
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
        with pytest.raises(RuntimeError, match="closed"):
            await runner.exec(ExecCommand.process("pwd"))
        assert runner.lifecycle_state == "poisoned"
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
            "status": "failed",
            "timeout_s": 5.0,
            "error_type": "RuntimeError",
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
