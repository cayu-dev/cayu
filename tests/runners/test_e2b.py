from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from cayu.runners import DEFAULT_E2B_CWD, E2BRunner, ExecCommand


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

    async def run(self, cmd: str, **kwargs: Any) -> Any:
        self.calls.append({"cmd": cmd, **kwargs})
        if kwargs.get("background"):
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
    def __init__(self, sandbox_id: str = "sbx_123") -> None:
        self.sandbox_id = sandbox_id
        self.commands = FakeCommands()
        self.kill_calls = 0

    async def kill(self) -> bool:
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

    assert runner.sandbox_id == "sbx_123"
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
            "sbx_existing",
            sandbox_timeout_s=300,
            e2b_module=FakeE2BModule,
            api_key="test",
        )

    runner = asyncio.run(run())

    assert runner.sandbox_id == "sbx_existing"
    assert runner.close_action == "none"
    assert FakeAsyncSandbox.connected == [
        {"sandbox_id": "sbx_existing", "timeout": 300, "api_key": "test"}
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
            "sbx_existing",
            ensure_default_cwd=False,
            e2b_module=FakeE2BModule,
        )

    runner = asyncio.run(run())

    assert runner.sandbox_id == "sbx_existing"
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


def test_e2b_runner_kills_command_on_timeout() -> None:
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


def test_e2b_runner_kills_command_on_cancellation() -> None:
    async def run() -> FakeHandle:
        sandbox = FakeSandbox()
        handle = BlockingHandle()
        sandbox.commands.next_handle = handle
        runner = E2BRunner(sandbox, e2b_module=FakeE2BModule)
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await handle.wait_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return handle

    handle = asyncio.run(run())

    assert handle.killed is True


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

    with pytest.raises(ValueError, match="relative"):
        asyncio.run(runner.exec(ExecCommand.process("pwd"), cwd="/home/user/workspace"))

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(runner.exec(ExecCommand.process("pwd"), cwd="../"))

    with pytest.raises(ValueError, match="sandbox_timeout_s"):
        asyncio.run(E2BRunner.create(sandbox_timeout_s=0, e2b_module=FakeE2BModule))
