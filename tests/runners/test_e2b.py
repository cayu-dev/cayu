from __future__ import annotations

import asyncio
from dataclasses import dataclass
from math import inf, nan
from typing import Any

import pytest

from cayu.runners import DEFAULT_E2B_CWD, E2BRunner, ExecCommand, ExecResult


@dataclass
class FakeCommandResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error: str | None = None


class FakeCommandExit(Exception):
    def __init__(self, *, stdout: str, stderr: str, exit_code: int) -> None:
        super().__init__("command exited")
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class FakeHandle:
    def __init__(
        self,
        *,
        result: FakeCommandResult | None = None,
        raise_exit: FakeCommandExit | None = None,
    ) -> None:
        self.result = result or FakeCommandResult(stdout="ok", exit_code=0)
        self.raise_exit = raise_exit
        self.stdin: list[str] = []
        self.stdin_closed = False
        self.killed = False
        self.fail_kill = False
        self.hang_kill = False
        self.wait_started = asyncio.Event()

    async def wait(self) -> FakeCommandResult:
        self.wait_started.set()
        if self.raise_exit is not None:
            raise self.raise_exit
        return self.result

    async def send_stdin(self, data: str) -> None:
        self.stdin.append(data)

    async def close_stdin(self) -> None:
        self.stdin_closed = True

    async def kill(self) -> bool:
        if self.hang_kill:
            await asyncio.sleep(30)
        if self.fail_kill:
            raise RuntimeError("kill failed")
        self.killed = True
        return True


class BlockingHandle(FakeHandle):
    async def wait(self) -> FakeCommandResult:
        self.wait_started.set()
        await asyncio.sleep(30)
        return FakeCommandResult()


class FakeCommands:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.next_handle: FakeHandle = FakeHandle()
        self.fail_next_setup = False
        self.cancel_next_background = False
        self.timeout_next_background = False
        self.background_delay_s = 0.0
        self.start_cancelled = False
        self.hang_after_start_cancel = False
        self.return_after_start_cancel_delay_s: float | None = None

    async def run(self, cmd: str, **kwargs: Any) -> Any:
        self.calls.append({"cmd": cmd, **kwargs})
        if kwargs.get("background"):
            if self.background_delay_s:
                try:
                    await asyncio.sleep(self.background_delay_s)
                except asyncio.CancelledError:
                    self.start_cancelled = True
                    if self.hang_after_start_cancel:
                        await asyncio.sleep(30)
                    if self.return_after_start_cancel_delay_s is not None:
                        await asyncio.sleep(self.return_after_start_cancel_delay_s)
                        return self.next_handle
                    raise
            if self.cancel_next_background:
                self.cancel_next_background = False
                raise asyncio.CancelledError
            if self.timeout_next_background:
                self.timeout_next_background = False
                raise TimeoutError
            on_stdout = kwargs.get("on_stdout")
            on_stderr = kwargs.get("on_stderr")
            if on_stdout is not None:
                maybe = on_stdout("abcdef")
                if hasattr(maybe, "__await__"):
                    await maybe
            if on_stderr is not None:
                maybe = on_stderr("uvwxyz")
                if hasattr(maybe, "__await__"):
                    await maybe
            return self.next_handle
        if self.fail_next_setup:
            self.fail_next_setup = False
            raise RuntimeError("setup failed")
        return FakeCommandResult()


class FakeSandbox:
    def __init__(self, sandbox_id: str = "e2b_123") -> None:
        self.sandbox_id = sandbox_id
        self.commands = FakeCommands()
        self.kill_calls = 0
        self.fail_kill = False
        self.hang_kill = False

    async def kill(self) -> bool:
        if self.hang_kill:
            await asyncio.sleep(30)
        if self.fail_kill:
            raise RuntimeError("sandbox kill failed")
        self.kill_calls += 1
        return True


class FakeAsyncSandbox:
    created: list[dict[str, Any]] = []
    connected: list[dict[str, Any]] = []
    next_sandbox: FakeSandbox | None = None

    @classmethod
    async def create(cls, **kwargs: Any) -> FakeSandbox:
        cls.created.append(dict(kwargs))
        sandbox = cls.next_sandbox or FakeSandbox()
        cls.next_sandbox = sandbox
        return sandbox

    @classmethod
    async def connect(cls, sandbox_id: str, **kwargs: Any) -> FakeSandbox:
        cls.connected.append({"sandbox_id": sandbox_id, **kwargs})
        sandbox = cls.next_sandbox or FakeSandbox(sandbox_id)
        cls.next_sandbox = sandbox
        return sandbox


class FakeE2BModule:
    AsyncSandbox = FakeAsyncSandbox


def reset_fake_e2b() -> None:
    FakeAsyncSandbox.created = []
    FakeAsyncSandbox.connected = []
    FakeAsyncSandbox.next_sandbox = None


def test_e2b_runner_create_passes_e2b_lifecycle_options() -> None:
    async def run() -> E2BRunner:
        reset_fake_e2b()
        return await E2BRunner.create(
            template="base",
            sandbox_timeout_s=600,
            metadata={"session": "s1"},
            envs={"BOOT": "1"},
            allow_internet_access=False,
            network={"deny_out": ["0.0.0.0/0"]},
            lifecycle={"on_timeout": "kill"},
            e2b_module=FakeE2BModule,
            api_key="test",
        )

    runner = asyncio.run(run())

    assert runner.sandbox_id == "e2b_123"
    assert runner.default_cwd == DEFAULT_E2B_CWD
    assert runner.close_action == "kill"
    assert FakeAsyncSandbox.created == [
        {
            "secure": True,
            "allow_internet_access": False,
            "template": "base",
            "timeout": 600,
            "metadata": {"session": "s1"},
            "envs": {"BOOT": "1"},
            "network": {"deny_out": ["0.0.0.0/0"]},
            "lifecycle": {"on_timeout": "kill"},
            "api_key": "test",
        }
    ]
    assert runner._sandbox.commands.calls[0] == {
        "cmd": "mkdir -p /home/user/workspace",
        "cwd": "/",
        "timeout": 60,
    }


def test_e2b_runner_from_existing_connects_without_claiming_lifecycle() -> None:
    async def run() -> E2BRunner:
        reset_fake_e2b()
        return await E2BRunner.from_existing(
            "e2b_existing",
            sandbox_timeout_s=300,
            e2b_module=FakeE2BModule,
            api_key="test",
        )

    runner = asyncio.run(run())

    assert runner.sandbox_id == "e2b_existing"
    assert runner.close_action == "none"
    assert FakeAsyncSandbox.connected == [
        {"sandbox_id": "e2b_existing", "timeout": 300, "api_key": "test"}
    ]
    assert runner._sandbox.commands.calls[0] == {
        "cmd": "mkdir -p /home/user/workspace",
        "cwd": "/",
        "timeout": 60,
    }


def test_e2b_runner_from_existing_can_skip_default_cwd_setup() -> None:
    async def run() -> E2BRunner:
        reset_fake_e2b()
        return await E2BRunner.from_existing(
            "e2b_existing",
            ensure_default_cwd=False,
            e2b_module=FakeE2BModule,
        )

    runner = asyncio.run(run())

    assert runner.sandbox_id == "e2b_existing"
    assert runner._sandbox.commands.calls == []


def test_e2b_runner_executes_process_with_shell_quoting_and_isolated_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CAYU_HOST_SECRET_SHOULD_NOT_LEAK", "hidden")
    sandbox = FakeSandbox()
    sandbox.commands.next_handle = FakeHandle(
        result=FakeCommandResult(stdout="", stderr="", exit_code=3)
    )
    runner = E2BRunner(sandbox, e2b_module=FakeE2BModule)

    result = asyncio.run(
        runner.exec(
            ExecCommand.process("python", "-c", "print('hello world')"),
            cwd="src",
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
    assert sandbox.commands.calls == [
        {
            "cmd": "python -c 'print('\"'\"'hello world'\"'\"')'",
            "background": True,
            "envs": {"VISIBLE": "1"},
            "cwd": "/home/user/workspace/src",
            "on_stdout": sandbox.commands.calls[0]["on_stdout"],
            "on_stderr": sandbox.commands.calls[0]["on_stderr"],
            "stdin": True,
            "timeout": 5.0,
        }
    ]
    assert "CAYU_HOST_SECRET_SHOULD_NOT_LEAK" not in sandbox.commands.calls[0]["envs"]
    assert sandbox.commands.next_handle.stdin == ["input"]
    assert sandbox.commands.next_handle.stdin_closed is True


def test_e2b_runner_pins_commands_to_configured_exec_user() -> None:
    sandbox = FakeSandbox()
    runner = E2BRunner(sandbox, exec_user="sandbox-user", e2b_module=FakeE2BModule)

    asyncio.run(runner.exec(ExecCommand.process("whoami")))

    assert sandbox.commands.calls[0]["user"] == "sandbox-user"


def test_e2b_runner_applies_trusted_env_overlay_after_command_env() -> None:
    sandbox = FakeSandbox()
    runner = E2BRunner(
        sandbox,
        env_overlay={
            "HTTPS_PROXY": "http://cayu-egress.example:8443",
            "STRIPE_SECRET_KEY": "sk_test_cayu_virtual",
        },
        e2b_module=FakeE2BModule,
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

    assert sandbox.commands.calls[0]["envs"] == {
        "HTTPS_PROXY": "http://cayu-egress.example:8443",
        "STRIPE_SECRET_KEY": "sk_test_cayu_virtual",
        "VISIBLE": "1",
    }


def test_e2b_runner_returns_nonzero_exit_as_exec_result() -> None:
    sandbox = FakeSandbox()
    sandbox.commands.next_handle = FakeHandle(
        raise_exit=FakeCommandExit(stdout="full out", stderr="full err", exit_code=42)
    )
    runner = E2BRunner(sandbox, e2b_module=FakeE2BModule)

    result = asyncio.run(runner.exec(ExecCommand.bash("exit 42"), output_limit_bytes=20))

    assert result.exit_code == 42
    assert result.stdout == "abcdef"
    assert result.stderr == "uvwxyz"


def test_e2b_runner_kills_command_on_timeout_by_default() -> None:
    sandbox = FakeSandbox()
    sandbox.commands.next_handle = BlockingHandle()
    runner = E2BRunner(sandbox, e2b_module=FakeE2BModule)

    result = asyncio.run(
        runner.exec(
            ExecCommand.process("sleep", "30"),
            timeout_s=1,
            output_limit_bytes=10,
        )
    )

    assert result.timed_out is True
    assert result.exit_code == -9
    assert sandbox.commands.next_handle.killed is True
    assert sandbox.kill_calls == 0
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "kill_command",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]
    sandbox.commands.next_handle = FakeHandle()
    after = asyncio.run(runner.exec(ExecCommand.process("pwd")))
    assert after.exit_code == 0


def test_e2b_runner_shares_one_deadline_across_start_and_wait_phases() -> None:
    # A slow command start must consume the exec timeout budget, so the wait
    # phase gets only the *remaining* time rather than a fresh full timeout.
    # Otherwise total wall-clock could reach ~2x the requested timeout.
    sandbox = FakeSandbox()
    sandbox.commands.next_handle = BlockingHandle()
    sandbox.commands.background_delay_s = 0.9
    runner = E2BRunner(sandbox, e2b_module=FakeE2BModule)

    async def run() -> tuple[ExecResult, float]:
        loop = asyncio.get_running_loop()
        started = loop.time()
        result = await runner.exec(
            ExecCommand.process("sleep", "30"),
            timeout_s=1,
            output_limit_bytes=10,
        )
        return result, loop.time() - started

    result, elapsed = asyncio.run(run())

    assert result.timed_out is True
    assert result.exit_code == -9
    # Bug behaviour would be ~0.9 (start) + 1.0 (fresh wait) ~= 1.9s.
    assert elapsed < 1.4, f"exec ran {elapsed:.3f}s, expected a single ~1s deadline"


def test_e2b_runner_can_kill_sandbox_on_timeout_explicitly() -> None:
    sandbox = FakeSandbox()
    handle = BlockingHandle()
    sandbox.commands.next_handle = handle
    runner = E2BRunner(sandbox, timeout_cleanup="sandbox", e2b_module=FakeE2BModule)

    result = asyncio.run(
        runner.exec(
            ExecCommand.process("sleep", "30"),
            timeout_s=1,
            output_limit_bytes=10,
        )
    )

    assert result.timed_out is True
    assert result.exit_code == -9
    assert handle.killed is False
    assert sandbox.kill_calls == 1
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "kill_sandbox",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(runner.exec(ExecCommand.process("pwd")))


def test_e2b_runner_reports_timeout_cleanup_failure() -> None:
    sandbox = FakeSandbox()
    handle = BlockingHandle()
    handle.fail_kill = True
    sandbox.commands.next_handle = handle
    runner = E2BRunner(sandbox, e2b_module=FakeE2BModule)

    result = asyncio.run(
        runner.exec(
            ExecCommand.process("sleep", "30"),
            timeout_s=1,
            output_limit_bytes=10,
        )
    )

    assert result.timed_out is True
    assert result.exit_code == -9
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "kill_command",
            "status": "failed",
            "timeout_s": 5.0,
            "error_type": "RuntimeError",
            "error": "kill failed",
        }
    ]
    sandbox.commands.next_handle = FakeHandle()
    after = asyncio.run(runner.exec(ExecCommand.process("pwd")))
    assert after.exit_code == 0


def test_e2b_runner_kills_command_on_cancellation_by_default() -> None:
    async def run() -> tuple[FakeSandbox, FakeHandle, int]:
        sandbox = FakeSandbox()
        handle = BlockingHandle()
        sandbox.commands.next_handle = handle
        runner = E2BRunner(sandbox, e2b_module=FakeE2BModule)
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await handle.wait_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        sandbox.commands.next_handle = FakeHandle()
        after = await runner.exec(ExecCommand.process("pwd"))
        return sandbox, handle, exc_info.value, after.exit_code

    sandbox, handle, exc, after = asyncio.run(run())

    assert handle.killed is True
    assert sandbox.kill_calls == 0
    assert after == 0
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "kill_command",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]


def test_e2b_runner_can_kill_sandbox_on_cancellation_explicitly() -> None:
    async def run() -> tuple[FakeSandbox, FakeHandle]:
        sandbox = FakeSandbox()
        handle = BlockingHandle()
        sandbox.commands.next_handle = handle
        runner = E2BRunner(
            sandbox,
            cancellation_cleanup="sandbox",
            e2b_module=FakeE2BModule,
        )
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await handle.wait_started.wait()
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
            "adapter": "e2b",
            "action": "kill_sandbox",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]


def test_e2b_runner_can_skip_cancellation_cleanup_explicitly() -> None:
    async def run() -> tuple[FakeSandbox, FakeHandle, int]:
        sandbox = FakeSandbox()
        handle = BlockingHandle()
        sandbox.commands.next_handle = handle
        runner = E2BRunner(
            sandbox,
            cancellation_cleanup="none",
            e2b_module=FakeE2BModule,
        )
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await handle.wait_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        sandbox.commands.next_handle = FakeHandle()
        after = await runner.exec(ExecCommand.process("pwd"))
        return sandbox, handle, exc_info.value, after.exit_code

    sandbox, handle, exc, after = asyncio.run(run())

    assert handle.killed is False
    assert sandbox.kill_calls == 0
    assert after == 0
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "none",
            "status": "skipped",
            "timeout_s": 5.0,
        }
    ]


def test_e2b_runner_waits_for_start_handle_on_cancellation() -> None:
    async def run() -> tuple[FakeSandbox, FakeHandle, int]:
        sandbox = FakeSandbox()
        handle = BlockingHandle()
        sandbox.commands.next_handle = handle
        sandbox.commands.background_delay_s = 0.01
        runner = E2BRunner(sandbox, e2b_module=FakeE2BModule)
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        sandbox.commands.next_handle = FakeHandle()
        after = await runner.exec(ExecCommand.process("pwd"))
        return sandbox, handle, exc_info.value, after.exit_code

    sandbox, handle, exc, after = asyncio.run(run())

    assert handle.killed is True
    assert sandbox.kill_calls == 0
    assert after == 0
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "kill_command",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]


def test_e2b_runner_stays_reusable_when_cancelled_before_handle_is_returned() -> None:
    async def run() -> tuple[FakeSandbox, int]:
        sandbox = FakeSandbox()
        sandbox.commands.cancel_next_background = True
        runner = E2BRunner(sandbox, close_action="kill", e2b_module=FakeE2BModule)

        with pytest.raises(asyncio.CancelledError) as exc_info:
            await runner.exec(ExecCommand.process("sleep", "30"))
        sandbox.commands.next_handle = FakeHandle()
        after = await runner.exec(ExecCommand.process("pwd"))
        await runner.close()
        return sandbox, exc_info.value, after.exit_code

    sandbox, exc, after = asyncio.run(run())

    assert sandbox.kill_calls == 1
    assert after == 0
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "kill_command",
            "status": "unsupported",
            "timeout_s": 5.0,
            "error": "command handle is not available",
        }
    ]


def test_e2b_runner_cancels_delayed_start_task_when_handle_wait_times_out() -> None:
    async def run() -> tuple[FakeSandbox, int]:
        sandbox = FakeSandbox()
        sandbox.commands.background_delay_s = 1
        runner = E2BRunner(
            sandbox,
            cancel_timeout_s=0.01,
            e2b_module=FakeE2BModule,
        )
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.wait_for(task, timeout=1)
        await asyncio.sleep(0)
        sandbox.commands.next_handle = FakeHandle()
        after = await runner.exec(ExecCommand.process("pwd"))
        return sandbox, exc_info.value, after.exit_code

    sandbox, exc, after = asyncio.run(run())

    assert sandbox.commands.start_cancelled is True
    assert sandbox.kill_calls == 0
    assert after == 0
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "kill_command",
            "status": "unsupported",
            "timeout_s": 0.01,
            "error": "command handle is not available",
        }
    ]


def test_e2b_runner_bounds_delayed_start_task_drain_when_sdk_ignores_cancellation() -> None:
    async def run() -> tuple[FakeSandbox, bool, int]:
        sandbox = FakeSandbox()
        sandbox.commands.background_delay_s = 1
        sandbox.commands.hang_after_start_cancel = True
        runner = E2BRunner(
            sandbox,
            cancel_timeout_s=0.01,
            e2b_module=FakeE2BModule,
        )
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.wait_for(task, timeout=1)
        await asyncio.sleep(0.05)
        with pytest.raises(RuntimeError, match="command state is unknown"):
            await runner.exec(ExecCommand.process("pwd"))
        return sandbox, exc_info.value, runner._exec_closed, len(runner._late_start_cleanup_tasks)

    sandbox, exc, exec_closed, late_cleanup_tasks = asyncio.run(run())

    assert sandbox.commands.start_cancelled is True
    assert sandbox.kill_calls == 0
    assert exec_closed is True
    assert late_cleanup_tasks == 0
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "kill_command",
            "status": "deferred",
            "timeout_s": 0.01,
            "late_start_cleanup_timeout_s": 0.04,
            "reason": "command handle is not available yet; cleanup will continue in background",
        }
    ]


def test_e2b_runner_kills_late_handle_after_start_cancellation_timeout() -> None:
    async def run() -> tuple[FakeSandbox, FakeHandle, bool, int]:
        sandbox = FakeSandbox()
        late_handle = FakeHandle()
        sandbox.commands.next_handle = late_handle
        sandbox.commands.background_delay_s = 1
        sandbox.commands.return_after_start_cancel_delay_s = 0.02
        runner = E2BRunner(
            sandbox,
            cancel_timeout_s=0.01,
            e2b_module=FakeE2BModule,
        )
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.wait_for(task, timeout=1)

        with pytest.raises(RuntimeError, match="cleanup is pending"):
            await runner.exec(ExecCommand.process("pwd"))

        for _ in range(20):
            if late_handle.killed:
                break
            await asyncio.sleep(0.01)

        sandbox.commands.next_handle = FakeHandle()
        after = await runner.exec(ExecCommand.process("pwd"))
        return sandbox, late_handle, exc_info.value, runner._exec_closed, after.exit_code

    sandbox, late_handle, exc, exec_closed, after = asyncio.run(run())

    assert sandbox.commands.start_cancelled is True
    assert late_handle.killed is True
    assert sandbox.kill_calls == 0
    assert exec_closed is False
    assert after == 0
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "kill_command",
            "status": "deferred",
            "timeout_s": 0.01,
            "late_start_cleanup_timeout_s": 0.04,
            "reason": "command handle is not available yet; cleanup will continue in background",
        }
    ]


def test_e2b_runner_late_start_sandbox_cleanup_keeps_runner_closed() -> None:
    async def run() -> tuple[FakeSandbox, FakeHandle, bool, bool]:
        sandbox = FakeSandbox()
        late_handle = FakeHandle()
        sandbox.commands.next_handle = late_handle
        sandbox.commands.background_delay_s = 1
        sandbox.commands.return_after_start_cancel_delay_s = 0.02
        runner = E2BRunner(
            sandbox,
            cancel_timeout_s=0.01,
            cancellation_cleanup="sandbox",
            e2b_module=FakeE2BModule,
        )
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.wait_for(task, timeout=1)

        for _ in range(20):
            if sandbox.kill_calls:
                break
            await asyncio.sleep(0.01)

        with pytest.raises(RuntimeError, match="closed"):
            await runner.exec(ExecCommand.process("pwd"))

        return sandbox, late_handle, exc_info.value, runner._closed, runner._exec_closed

    sandbox, late_handle, exc, closed, exec_closed = asyncio.run(run())

    assert late_handle.killed is False
    assert sandbox.kill_calls == 1
    assert closed is True
    assert exec_closed is True
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "kill_sandbox",
            "status": "completed",
            "timeout_s": 0.01,
        }
    ]


def test_e2b_runner_late_start_skip_cleanup_keeps_exec_closed() -> None:
    async def run() -> tuple[FakeSandbox, FakeHandle, bool]:
        sandbox = FakeSandbox()
        late_handle = FakeHandle()
        sandbox.commands.next_handle = late_handle
        sandbox.commands.background_delay_s = 1
        sandbox.commands.return_after_start_cancel_delay_s = 0.02
        runner = E2BRunner(
            sandbox,
            cancel_timeout_s=0.01,
            cancellation_cleanup="none",
            e2b_module=FakeE2BModule,
        )
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.wait_for(task, timeout=1)

        await asyncio.sleep(0.05)
        with pytest.raises(RuntimeError, match="command state is unknown"):
            await runner.exec(ExecCommand.process("pwd"))

        return sandbox, late_handle, exc_info.value, runner._exec_closed

    sandbox, late_handle, exc, exec_closed = asyncio.run(run())

    assert late_handle.killed is False
    assert sandbox.kill_calls == 0
    assert exec_closed is True
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "none",
            "status": "skipped",
            "timeout_s": 0.01,
        }
    ]


def test_e2b_runner_stays_reusable_when_timeout_happens_before_handle_is_returned() -> None:
    sandbox = FakeSandbox()
    sandbox.commands.timeout_next_background = True
    runner = E2BRunner(sandbox, e2b_module=FakeE2BModule)

    result = asyncio.run(runner.exec(ExecCommand.process("sleep", "30")))

    assert result.timed_out is True
    assert result.exit_code == -9
    assert sandbox.kill_calls == 0
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "kill_command",
            "status": "unsupported",
            "timeout_s": 5.0,
            "error": "command handle is not available",
        }
    ]
    sandbox.commands.next_handle = FakeHandle()
    after = asyncio.run(runner.exec(ExecCommand.process("pwd")))
    assert after.exit_code == 0


def test_e2b_runner_times_out_delayed_start_without_hanging() -> None:
    async def run() -> tuple[FakeSandbox, ExecResult, int]:
        sandbox = FakeSandbox()
        sandbox.commands.background_delay_s = 30
        runner = E2BRunner(
            sandbox,
            cancel_timeout_s=0.01,
            e2b_module=FakeE2BModule,
        )

        result = await asyncio.wait_for(
            runner.exec(ExecCommand.process("sleep", "30"), timeout_s=1),
            timeout=2,
        )
        sandbox.commands.background_delay_s = 0
        sandbox.commands.next_handle = FakeHandle()
        after = await runner.exec(ExecCommand.process("pwd"))
        return sandbox, result, after.exit_code

    sandbox, result, after = asyncio.run(run())

    assert sandbox.commands.start_cancelled is True
    assert result.timed_out is True
    assert result.exit_code == -9
    assert sandbox.kill_calls == 0
    assert after == 0
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "kill_command",
            "status": "unsupported",
            "timeout_s": 0.01,
            "error": "command handle is not available",
        }
    ]


def test_e2b_runner_closes_exec_when_delayed_start_timeout_cannot_be_resolved() -> None:
    async def run() -> tuple[FakeSandbox, ExecResult, bool, int]:
        sandbox = FakeSandbox()
        sandbox.commands.background_delay_s = 30
        sandbox.commands.hang_after_start_cancel = True
        runner = E2BRunner(
            sandbox,
            cancel_timeout_s=0.01,
            e2b_module=FakeE2BModule,
        )

        result = await asyncio.wait_for(
            runner.exec(ExecCommand.process("sleep", "30"), timeout_s=1),
            timeout=2,
        )
        await asyncio.sleep(0.05)
        with pytest.raises(RuntimeError, match="command state is unknown"):
            await runner.exec(ExecCommand.process("pwd"))
        return sandbox, result, runner._exec_closed, len(runner._late_start_cleanup_tasks)

    sandbox, result, exec_closed, late_cleanup_tasks = asyncio.run(run())

    assert sandbox.commands.start_cancelled is True
    assert result.timed_out is True
    assert result.exit_code == -9
    assert sandbox.kill_calls == 0
    assert exec_closed is True
    assert late_cleanup_tasks == 0
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "kill_command",
            "status": "deferred",
            "timeout_s": 0.01,
            "late_start_cleanup_timeout_s": 0.04,
            "reason": "command handle is not available yet; cleanup will continue in background",
        }
    ]


def test_e2b_runner_bounds_hanging_command_kill_on_cancellation() -> None:
    async def run() -> tuple[FakeHandle, int]:
        sandbox = FakeSandbox()
        handle = BlockingHandle()
        sandbox.commands.next_handle = handle
        handle.hang_kill = True
        runner = E2BRunner(
            sandbox,
            cancel_timeout_s=0.01,
            e2b_module=FakeE2BModule,
        )
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await handle.wait_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.wait_for(task, timeout=1)
        sandbox.commands.next_handle = FakeHandle()
        after = await runner.exec(ExecCommand.process("pwd"))
        return handle, exc_info.value, after.exit_code

    handle, exc, after = asyncio.run(run())

    assert handle.killed is False
    assert after == 0
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "kill_command",
            "status": "timeout",
            "timeout_s": 0.01,
        }
    ]


def test_e2b_runner_stays_reusable_when_command_kill_fails() -> None:
    async def run() -> tuple[FakeHandle, int]:
        sandbox = FakeSandbox()
        handle = BlockingHandle()
        handle.fail_kill = True
        sandbox.commands.next_handle = handle
        runner = E2BRunner(
            sandbox,
            e2b_module=FakeE2BModule,
        )
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await handle.wait_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        sandbox.commands.next_handle = FakeHandle()
        after = await runner.exec(ExecCommand.process("pwd"))
        return handle, exc_info.value, after.exit_code

    handle, exc, after = asyncio.run(run())

    assert handle.killed is False
    assert after == 0
    assert exc.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "e2b",
            "action": "kill_command",
            "status": "failed",
            "timeout_s": 5.0,
            "error_type": "RuntimeError",
            "error": "kill failed",
        }
    ]


def test_e2b_runner_cleans_up_created_sandbox_when_setup_fails() -> None:
    async def run() -> FakeSandbox:
        reset_fake_e2b()
        sandbox = FakeSandbox()
        sandbox.commands.fail_next_setup = True
        FakeAsyncSandbox.next_sandbox = sandbox
        with pytest.raises(RuntimeError, match="setup failed"):
            await E2BRunner.create(e2b_module=FakeE2BModule)
        return sandbox

    sandbox = asyncio.run(run())

    assert sandbox.kill_calls == 1


def test_e2b_runner_validates_boundary_inputs() -> None:
    sandbox = FakeSandbox()
    runner = E2BRunner(sandbox, e2b_module=FakeE2BModule)
    bad_cleanup: Any = "delete_process"

    assert runner.resolve_cwd("/home/user/workspace") == "/home/user/workspace"
    with pytest.raises(ValueError, match="outside the runner root"):
        asyncio.run(runner.exec(ExecCommand.process("pwd"), cwd="/etc"))

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(runner.exec(ExecCommand.process("pwd"), cwd="../"))

    with pytest.raises(ValueError, match="sandbox_timeout_s"):
        asyncio.run(E2BRunner.create(sandbox_timeout_s=0, e2b_module=FakeE2BModule))

    with pytest.raises(ValueError, match="cancel_timeout_s"):
        E2BRunner(sandbox, cancel_timeout_s=0, e2b_module=FakeE2BModule)

    with pytest.raises(ValueError, match="cancel_timeout_s"):
        E2BRunner(sandbox, cancel_timeout_s=inf, e2b_module=FakeE2BModule)

    with pytest.raises(ValueError, match="cancel_timeout_s"):
        E2BRunner(sandbox, cancel_timeout_s=nan, e2b_module=FakeE2BModule)

    with pytest.raises(ValueError, match="cancellation_cleanup"):
        E2BRunner(
            sandbox,
            cancellation_cleanup=bad_cleanup,
            e2b_module=FakeE2BModule,
        )

    with pytest.raises(ValueError, match="timeout_cleanup"):
        E2BRunner(
            sandbox,
            timeout_cleanup=bad_cleanup,
            e2b_module=FakeE2BModule,
        )
