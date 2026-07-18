from __future__ import annotations

import asyncio
from dataclasses import dataclass
from math import inf, nan
from typing import Any

import pytest

from cayu.runners import (
    DEFAULT_E2B_CWD,
    E2BGuestHandoffError,
    E2BGuestProvisioner,
    E2BRunner,
    E2BWorkspaceCapability,
    ExecCommand,
    ExecResult,
)


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
        self.foreground_results: list[FakeCommandResult] = []
        self.foreground_exceptions: dict[int, Exception] = {}
        self.foreground_call_count = 0

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
        self.foreground_call_count += 1
        if error := self.foreground_exceptions.get(self.foreground_call_count):
            raise error
        if self.foreground_results:
            return self.foreground_results.pop(0)
        return FakeCommandResult()


class FakeFiles:
    def __init__(self) -> None:
        self.writes: list[dict[str, Any]] = []
        self.write_error: Exception | None = None

    async def write(self, path: str, data: bytes, **kwargs: Any) -> None:
        if self.write_error is not None:
            raise self.write_error
        self.writes.append({"path": path, "data": data, **kwargs})


class FakeSandbox:
    def __init__(self, sandbox_id: str = "e2b_123") -> None:
        self.sandbox_id = sandbox_id
        self.commands = FakeCommands()
        self.files = FakeFiles()
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


class CoordinatedKillSandbox(FakeSandbox):
    def __init__(self) -> None:
        super().__init__()
        self.kill_started = asyncio.Event()
        self.release_kill = asyncio.Event()

    async def kill(self) -> bool:
        self.kill_started.set()
        await self.release_kill.wait()
        self.kill_calls += 1
        return True


class SelfCancellingKillSandbox(FakeSandbox):
    async def kill(self) -> bool:
        self.kill_calls += 1
        raise asyncio.CancelledError


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


@dataclass(frozen=True)
class FakeSandboxQuery:
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class FakeSandboxInfo:
    sandbox_id: str
    metadata: dict[str, str]


class FakeSandboxPaginator:
    def __init__(self, entries: list[FakeSandboxInfo]) -> None:
        self._entries = entries
        self.has_next = True

    async def next_items(self) -> list[FakeSandboxInfo]:
        self.has_next = False
        return self._entries


class AmbiguousCreateAsyncSandbox:
    active: dict[str, FakeSandboxInfo] = {}
    started: asyncio.Event | None = None
    allocate = True
    publish_after_cancel = False
    publish_delay_s = 0.05
    list_calls: list[dict[str, Any]] = []
    kill_calls: list[dict[str, Any]] = []
    late_tasks: set[asyncio.Task[None]] = set()

    @classmethod
    async def create(cls, **kwargs: Any) -> FakeSandbox:
        metadata = dict(kwargs["metadata"])
        sandbox_id = "e2b_ambiguous"
        info = FakeSandboxInfo(
            sandbox_id=sandbox_id,
            metadata=metadata,
        )
        if cls.allocate and not cls.publish_after_cancel:
            cls.active[sandbox_id] = info
        assert cls.started is not None
        cls.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if cls.allocate and cls.publish_after_cancel:

                async def publish_late() -> None:
                    await asyncio.sleep(cls.publish_delay_s)
                    cls.active[sandbox_id] = info

                task = asyncio.create_task(publish_late())
                cls.late_tasks.add(task)
                task.add_done_callback(cls.late_tasks.discard)
            raise
        raise AssertionError("ambiguous create should have been cancelled")

    @classmethod
    def list(
        cls,
        *,
        query: FakeSandboxQuery,
        limit: int,
        **kwargs: Any,
    ) -> FakeSandboxPaginator:
        cls.list_calls.append({"query": query, "limit": limit, **kwargs})
        matching = [
            info
            for info in cls.active.values()
            if all(info.metadata.get(key) == value for key, value in (query.metadata or {}).items())
        ]
        return FakeSandboxPaginator(matching)

    @classmethod
    async def kill(cls, sandbox_id: str, **kwargs: Any) -> bool:
        cls.kill_calls.append({"sandbox_id": sandbox_id, **kwargs})
        return cls.active.pop(sandbox_id, None) is not None


class AmbiguousCreateE2BModule:
    AsyncSandbox = AmbiguousCreateAsyncSandbox
    SandboxQuery = FakeSandboxQuery


class LateReturningCreateAsyncSandbox:
    sandbox = FakeSandbox("e2b_late_return")
    started: asyncio.Event | None = None

    @classmethod
    async def create(cls, **_kwargs: Any) -> FakeSandbox:
        assert cls.started is not None
        cls.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
            return cls.sandbox
        raise AssertionError("late create should have been cancelled")

    @classmethod
    def list(cls, **_kwargs: Any) -> FakeSandboxPaginator:
        return FakeSandboxPaginator([])

    @classmethod
    async def kill(cls, _sandbox_id: str, **_kwargs: Any) -> bool:
        return False


class LateReturningCreateE2BModule:
    AsyncSandbox = LateReturningCreateAsyncSandbox
    SandboxQuery = FakeSandboxQuery


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


def test_e2b_runner_hardened_handoff_seals_root_capability() -> None:
    retained: list[E2BGuestProvisioner] = []
    phases: list[str] = []

    async def bootstrap(provisioner: E2BGuestProvisioner) -> None:
        retained.append(provisioner)
        phases.append("bootstrap")
        await provisioner.install_directory("/opt/cayu/verification", mode=0o700)
        await provisioner.install_file(
            "/opt/cayu/verification/pristine.txt",
            b"trusted",
            mode=0o444,
        )

    async def guest_setup(runner: E2BRunner) -> None:
        assert retained[0].is_sealed is True
        phases.append("guest_setup")
        await runner.exec(ExecCommand.process("guest-setup-marker"))

    async def guest_probe(runner: E2BRunner) -> None:
        phases.append("guest_probe")
        await runner.exec(ExecCommand.process("guest-probe-marker"))

    async def run() -> tuple[E2BRunner, FakeSandbox]:
        reset_fake_e2b()
        runner = await E2BRunner.create_hardened(
            template="base",
            metadata={"session": "s1"},
            env_overlay={
                "BASH_ENV": "/home/user/workspace/guest-bootstrap.sh",
                "CAYU_GUEST_USER": "root",
                "LD_PRELOAD": "/home/user/workspace/guest.so",
                "PATH": "/home/user/workspace/bin",
                "PYTHONPATH": "/home/user/workspace",
            },
            bootstrap=bootstrap,
            guest_setup=guest_setup,
            guest_probe=guest_probe,
            e2b_module=FakeE2BModule,
        )
        sandbox = FakeAsyncSandbox.next_sandbox
        assert sandbox is not None
        return runner, sandbox

    runner, sandbox = asyncio.run(run())

    assert phases == ["bootstrap", "guest_setup", "guest_probe"]
    assert runner.exec_user == "user"
    assert retained[0].is_sealed is True
    with pytest.raises(RuntimeError, match="sealed"):
        asyncio.run(retained[0].install_file("/opt/cayu/late.txt", b"late"))
    with pytest.raises(RuntimeError, match="Raw E2B filesystem"):
        runner.filesystem()
    assert FakeAsyncSandbox.created[0]["secure"] is True
    assert FakeAsyncSandbox.created[0]["allow_internet_access"] is False
    assert FakeAsyncSandbox.created[0]["metadata"]["session"] == "s1"
    assert len(FakeAsyncSandbox.created[0]["metadata"]["cayu_guest_handoff_id"]) == 32
    assert sandbox.files.writes[0]["data"] == b"trusted"
    assert sandbox.files.writes[0]["user"] == "root"
    assert b"trusted" not in repr(sandbox.commands.calls).encode()
    root_calls = [call for call in sandbox.commands.calls if call.get("user") == "root"]
    assert any("iptables -I OUTPUT" in call["cmd"] for call in root_calls)
    assert any(
        call.get("envs", {}).get("CAYU_PROTECTED_PATH") == "/opt/cayu/verification/pristine.txt"
        for call in root_calls
    )
    protected_file_root_checks = [
        call
        for call in root_calls
        if call.get("envs", {}).get("CAYU_PROTECTED_PATH") == "/opt/cayu/verification/pristine.txt"
        and "stat -c %u" in call["cmd"]
    ]
    assert len(protected_file_root_checks) == 2
    guest_calls = [call for call in sandbox.commands.calls if call.get("user") == "user"]
    assert any("sudo -n true" in call["cmd"] for call in guest_calls)
    assert any("CAYU_PROTECTED_PATH" in call.get("envs", {}) for call in guest_calls)
    verification_calls = [
        (index, call)
        for index, call in enumerate(sandbox.commands.calls)
        if call.get("user") == "user" and "sudo -n true" in call["cmd"]
    ]
    assert len(verification_calls) == 2
    guest_setup_index = next(
        index
        for index, call in enumerate(sandbox.commands.calls)
        if call["cmd"] == "guest-setup-marker"
    )
    guest_probe_index = next(
        index
        for index, call in enumerate(sandbox.commands.calls)
        if call["cmd"] == "guest-probe-marker"
    )
    assert (
        verification_calls[0][0] < guest_setup_index < guest_probe_index < verification_calls[1][0]
    )
    verification = verification_calls[0][1]
    assert verification["envs"]["CAYU_GUEST_USER"] == "user"
    assert verification["envs"]["PATH"] == "/usr/sbin:/usr/bin:/sbin:/bin"
    assert verification["envs"]["BASH_ENV"] == "/dev/null"
    assert verification["envs"]["LD_PRELOAD"] == ""
    assert verification["envs"]["PYTHONPATH"] == ""
    runner.exec_user = "root"
    asyncio.run(runner.exec(ExecCommand.process("whoami")))
    assert sandbox.commands.calls[-1]["user"] == "user"
    workspace = runner.workspace_capability(E2BWorkspaceCapability)
    assert workspace is not None
    with pytest.raises(ValueError, match="pinned"):
        asyncio.run(workspace.get_info(".", user="root", request_timeout_s=None))


@pytest.mark.parametrize(
    ("path", "error"),
    [
        ("relative/path", "absolute"),
        ("/home/user/workspace/protected", "guest-writable"),
        ("/home/user/protected", "guest-writable"),
        ("/tmp/protected", "guest-writable"),
        ("/etc/../tmp/protected", "normalized"),
        ("/", "cannot be root"),
    ],
)
def test_e2b_guest_provisioner_rejects_unsafe_paths(path: str, error: str) -> None:
    provisioner = E2BGuestProvisioner._create(
        FakeSandbox(),
        default_cwd=DEFAULT_E2B_CWD,
        guest_user="user",
        operation_timeout_s=5,
        max_file_bytes=1024,
    )

    with pytest.raises(ValueError, match=error):
        asyncio.run(provisioner.install_file(path, b"trusted"))


@pytest.mark.parametrize(
    ("mode", "error_type"),
    [
        (True, TypeError),
        (0o666, ValueError),
        (0o646, ValueError),
        (0o777, ValueError),
        (0o1755, ValueError),
    ],
)
def test_e2b_guest_provisioner_rejects_unsafe_modes(
    mode: Any,
    error_type: type[Exception],
) -> None:
    provisioner = E2BGuestProvisioner._create(
        FakeSandbox(),
        default_cwd=DEFAULT_E2B_CWD,
        guest_user="user",
        operation_timeout_s=5,
        max_file_bytes=1024,
    )

    with pytest.raises(error_type, match="mode"):
        asyncio.run(provisioner.install_file("/opt/cayu/trusted", b"trusted", mode=mode))


def test_e2b_guest_provisioner_enforces_file_size_and_content_type() -> None:
    provisioner = E2BGuestProvisioner._create(
        FakeSandbox(),
        default_cwd=DEFAULT_E2B_CWD,
        guest_user="user",
        operation_timeout_s=5,
        max_file_bytes=4,
    )

    asyncio.run(provisioner.install_file("/opt/cayu/exact", b"1234"))
    with pytest.raises(ValueError, match="exceeds 4 bytes"):
        asyncio.run(provisioner.install_file("/opt/cayu/large", b"12345"))
    with pytest.raises(TypeError, match="str or bytes"):
        asyncio.run(provisioner.install_file("/opt/cayu/type", bytearray(b"1234")))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="owner-traversable"):
        asyncio.run(provisioner.install_directory("/opt/cayu/no-traverse", mode=0o644))


def test_e2b_guest_provisioner_rejects_duplicate_asset_paths() -> None:
    provisioner = E2BGuestProvisioner._create(
        FakeSandbox(),
        default_cwd=DEFAULT_E2B_CWD,
        guest_user="user",
        operation_timeout_s=5,
        max_file_bytes=1024,
    )

    asyncio.run(provisioner.install_directory("/opt/cayu/trusted"))
    with pytest.raises(ValueError, match="already registered"):
        asyncio.run(provisioner.install_directory("/opt/cayu/trusted", mode=0o700))


def test_e2b_guest_provisioner_cannot_be_constructed_outside_handoff() -> None:
    with pytest.raises(TypeError, match="issued only"):
        E2BGuestProvisioner(
            FakeSandbox(),
            default_cwd=DEFAULT_E2B_CWD,
            guest_user="user",
            operation_timeout_s=5,
            max_file_bytes=1024,
        )


def test_e2b_handoff_failure_kills_sandbox_and_preserves_primary_error() -> None:
    primary = RuntimeError("guest setup failed")
    retained: list[E2BGuestProvisioner] = []

    async def bootstrap(provisioner: E2BGuestProvisioner) -> None:
        retained.append(provisioner)

    async def fail_setup(_runner: E2BRunner) -> None:
        raise primary

    async def run() -> FakeSandbox:
        reset_fake_e2b()
        with pytest.raises(RuntimeError) as exc_info:
            await E2BRunner.create_hardened(
                bootstrap=bootstrap,
                guest_setup=fail_setup,
                e2b_module=FakeE2BModule,
            )
        assert exc_info.value is primary
        sandbox = FakeAsyncSandbox.next_sandbox
        assert sandbox is not None
        return sandbox

    sandbox = asyncio.run(run())

    assert sandbox.kill_calls == 1
    assert retained[0].is_sealed is True


def test_e2b_handoff_cleanup_failure_preserves_both_errors() -> None:
    primary = RuntimeError("guest probe failed")

    async def fail_probe(_runner: E2BRunner) -> None:
        raise primary

    async def run() -> BaseExceptionGroup:
        reset_fake_e2b()
        sandbox = FakeSandbox()
        sandbox.fail_kill = True
        FakeAsyncSandbox.next_sandbox = sandbox
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await E2BRunner.create_hardened(
                guest_probe=fail_probe,
                e2b_module=FakeE2BModule,
            )
        return exc_info.value

    error = asyncio.run(run())

    assert error.exceptions[0] is primary
    assert isinstance(error.exceptions[1], RuntimeError)
    assert "sandbox kill failed" in str(error.exceptions[1])


def test_e2b_handoff_cleanup_timeout_preserves_primary_failure() -> None:
    primary = RuntimeError("guest probe failed")

    async def fail_probe(_runner: E2BRunner) -> None:
        raise primary

    async def run() -> BaseExceptionGroup:
        reset_fake_e2b()
        sandbox = FakeSandbox()
        sandbox.hang_kill = True
        FakeAsyncSandbox.next_sandbox = sandbox
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await E2BRunner.create_hardened(
                guest_probe=fail_probe,
                cleanup_timeout_s=0.01,
                e2b_module=FakeE2BModule,
            )
        return exc_info.value

    error = asyncio.run(run())

    assert error.exceptions[0] is primary
    assert isinstance(error.exceptions[1], TimeoutError)
    assert "rollback timed out" in str(error.exceptions[1])


def test_e2b_handoff_cleanup_self_cancellation_preserves_primary_failure() -> None:
    primary = RuntimeError("guest probe failed")

    async def fail_probe(_runner: E2BRunner) -> None:
        raise primary

    async def run() -> tuple[BaseExceptionGroup, SelfCancellingKillSandbox]:
        reset_fake_e2b()
        sandbox = SelfCancellingKillSandbox()
        FakeAsyncSandbox.next_sandbox = sandbox
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await E2BRunner.create_hardened(
                guest_probe=fail_probe,
                e2b_module=FakeE2BModule,
            )
        return exc_info.value, sandbox

    error, sandbox = asyncio.run(run())

    assert error.exceptions[0] is primary
    assert isinstance(error.exceptions[1], RuntimeError)
    assert str(error.exceptions[1]) == (
        "E2B guest handoff rollback cancelled without caller cancellation."
    )
    assert sandbox.kill_calls == 1


def test_e2b_handoff_builtin_failure_is_secret_safe() -> None:
    secret = b"secret-verification-archive"

    async def bootstrap(provisioner: E2BGuestProvisioner) -> None:
        await provisioner.install_file("/opt/cayu/pristine.tar", secret)

    async def run() -> E2BGuestHandoffError:
        reset_fake_e2b()
        sandbox = FakeSandbox()
        sandbox.commands.foreground_results = [
            FakeCommandResult(),
            FakeCommandResult(),
            FakeCommandResult(exit_code=73),
        ]
        FakeAsyncSandbox.next_sandbox = sandbox
        with pytest.raises(E2BGuestHandoffError) as exc_info:
            await E2BRunner.create_hardened(
                bootstrap=bootstrap,
                e2b_module=FakeE2BModule,
            )
        return exc_info.value

    error = asyncio.run(run())

    assert error.phase == "bootstrap"
    assert error.exit_code == 73
    assert secret.decode() not in str(error)


def test_e2b_handoff_guest_verification_failure_kills_sandbox() -> None:
    callbacks: list[str] = []

    async def guest_setup(_runner: E2BRunner) -> None:
        callbacks.append("guest_setup")

    async def guest_probe(_runner: E2BRunner) -> None:
        callbacks.append("guest_probe")

    async def run() -> tuple[E2BGuestHandoffError, FakeSandbox]:
        reset_fake_e2b()
        sandbox = FakeSandbox()
        sandbox.commands.foreground_results = [
            FakeCommandResult(),
            FakeCommandResult(),
            FakeCommandResult(exit_code=46),
        ]
        FakeAsyncSandbox.next_sandbox = sandbox
        with pytest.raises(E2BGuestHandoffError) as exc_info:
            await E2BRunner.create_hardened(
                guest_setup=guest_setup,
                guest_probe=guest_probe,
                e2b_module=FakeE2BModule,
            )
        return exc_info.value, sandbox

    error, sandbox = asyncio.run(run())

    assert error.phase == "verification"
    assert error.exit_code == 46
    assert callbacks == []
    assert sandbox.kill_calls == 1


def test_e2b_handoff_command_exit_exception_is_secret_safe() -> None:
    secret = "provider-stderr-secret"

    async def run() -> tuple[E2BGuestHandoffError, FakeSandbox]:
        reset_fake_e2b()
        sandbox = FakeSandbox()
        sandbox.commands.foreground_exceptions[3] = FakeCommandExit(
            stdout=f"stdout {secret}",
            stderr=f"stderr {secret}",
            exit_code=46,
        )
        FakeAsyncSandbox.next_sandbox = sandbox
        with pytest.raises(E2BGuestHandoffError) as exc_info:
            await E2BRunner.create_hardened(e2b_module=FakeE2BModule)
        return exc_info.value, sandbox

    error, sandbox = asyncio.run(run())

    assert error.phase == "verification"
    assert error.exit_code == 46
    assert secret not in str(error)
    assert error.__cause__ is None
    assert sandbox.kill_calls == 1


def test_e2b_handoff_file_transfer_failure_is_secret_safe() -> None:
    secret = "secret-file-payload"

    async def bootstrap(provisioner: E2BGuestProvisioner) -> None:
        await provisioner.install_file("/opt/cayu/pristine.tar", secret)

    async def run() -> tuple[E2BGuestHandoffError, FakeSandbox]:
        reset_fake_e2b()
        sandbox = FakeSandbox()
        sandbox.files.write_error = RuntimeError(f"upload rejected: {secret}")
        FakeAsyncSandbox.next_sandbox = sandbox
        with pytest.raises(E2BGuestHandoffError) as exc_info:
            await E2BRunner.create_hardened(
                bootstrap=bootstrap,
                e2b_module=FakeE2BModule,
            )
        return exc_info.value, sandbox

    error, sandbox = asyncio.run(run())

    assert error.phase == "bootstrap"
    assert str(error) == "E2B guest handoff bootstrap failed: protected file transfer."
    assert secret not in str(error)
    assert sandbox.kill_calls == 1


def test_e2b_hardened_create_rejects_reserved_metadata_and_root_guest() -> None:
    reset_fake_e2b()
    with pytest.raises(ValueError, match="Cayu-owned"):
        asyncio.run(
            E2BRunner.create_hardened(
                metadata={"cayu_guest_handoff_id": "caller"},
                e2b_module=FakeE2BModule,
            )
        )
    with pytest.raises(ValueError, match="must not be root"):
        asyncio.run(E2BRunner.create_hardened(guest_user="root", e2b_module=FakeE2BModule))
    with pytest.raises(ValueError, match="must not be root"):
        asyncio.run(E2BRunner.create_hardened(guest_user="00", e2b_module=FakeE2BModule))
    with pytest.raises(TypeError, match="guest_probe"):
        asyncio.run(E2BRunner.create_hardened(guest_probe=object(), e2b_module=FakeE2BModule))
    with pytest.raises(ValueError, match="owns provider options"):
        asyncio.run(E2BRunner.create_hardened(secure=False, e2b_module=FakeE2BModule))
    with pytest.raises(ValueError, match="owns provider options"):
        asyncio.run(
            E2BRunner.create_hardened(
                envs={"PATH": "/home/user/workspace/bin"},
                e2b_module=FakeE2BModule,
            )
        )
    with pytest.raises(ValueError, match="absolute guest path"):
        asyncio.run(
            E2BRunner.create_hardened(
                default_cwd="relative",
                e2b_module=FakeE2BModule,
            )
        )
    assert FakeAsyncSandbox.created == []


def test_e2b_handoff_timeout_kills_created_sandbox() -> None:
    async def block_probe(_runner: E2BRunner) -> None:
        await asyncio.Event().wait()

    async def run() -> FakeSandbox:
        reset_fake_e2b()
        with pytest.raises(TimeoutError):
            await E2BRunner.create_hardened(
                guest_probe=block_probe,
                handoff_timeout_s=0.01,
                cleanup_timeout_s=0.2,
                e2b_module=FakeE2BModule,
            )
        sandbox = FakeAsyncSandbox.next_sandbox
        assert sandbox is not None
        return sandbox

    sandbox = asyncio.run(run())

    assert sandbox.kill_calls == 1


@pytest.mark.parametrize("callback_name", ["bootstrap", "guest_setup", "guest_probe"])
def test_e2b_handoff_timeout_does_not_depend_on_callback_cancellation(
    callback_name: str,
) -> None:
    async def run() -> FakeSandbox:
        reset_fake_e2b()
        cancellation_seen = asyncio.Event()
        release_callback = asyncio.Event()

        async def suppress_cancellation(_capability: Any) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_seen.set()
                await release_callback.wait()

        callback_options = {callback_name: suppress_cancellation}
        try:
            with pytest.raises(TimeoutError, match="handoff timed out"):
                await asyncio.wait_for(
                    E2BRunner.create_hardened(
                        handoff_timeout_s=0.01,
                        cleanup_timeout_s=0.2,
                        e2b_module=FakeE2BModule,
                        **callback_options,
                    ),
                    timeout=0.5,
                )
            assert cancellation_seen.is_set()
            sandbox = FakeAsyncSandbox.next_sandbox
            assert sandbox is not None
            assert sandbox.kill_calls == 1
            return sandbox
        finally:
            release_callback.set()
            # Let the detached handoff task observe revocation and terminate so
            # the test also guards against leaving local task work behind.
            await asyncio.sleep(0)
            await asyncio.sleep(0)

    sandbox = asyncio.run(run())

    assert sandbox.kill_calls == 1


def test_e2b_handoff_revocation_stops_shielded_provisioning_between_awaits() -> None:
    async def run() -> FakeSandbox:
        reset_fake_e2b()
        sandbox = FakeSandbox()
        staging_started = asyncio.Event()
        release_staging = asyncio.Event()
        original_run = sandbox.commands.run

        async def block_staging(cmd: str, **kwargs: Any) -> Any:
            if "/bin/mkdir -p /root/.cayu-handoff-" not in cmd:
                return await original_run(cmd, **kwargs)
            sandbox.commands.calls.append({"cmd": cmd, **kwargs})
            staging_started.set()
            await release_staging.wait()
            return FakeCommandResult()

        sandbox.commands.run = block_staging  # type: ignore[method-assign]
        FakeAsyncSandbox.next_sandbox = sandbox
        provisioning_tasks: list[asyncio.Task[None]] = []

        async def shield_provisioning(provisioner: E2BGuestProvisioner) -> None:
            task = asyncio.create_task(
                provisioner.install_file("/opt/cayu/pristine.tar", b"trusted")
            )
            provisioning_tasks.append(task)
            await asyncio.shield(task)

        handoff = asyncio.create_task(
            E2BRunner.create_hardened(
                bootstrap=shield_provisioning,
                handoff_timeout_s=0.01,
                cleanup_timeout_s=0.2,
                e2b_module=FakeE2BModule,
            )
        )
        await staging_started.wait()
        with pytest.raises(TimeoutError, match="handoff timed out"):
            await handoff

        release_staging.set()
        with pytest.raises(RuntimeError, match="sealed"):
            await provisioning_tasks[0]
        return sandbox

    sandbox = asyncio.run(run())

    assert sandbox.kill_calls == 1
    assert sandbox.files.writes == []
    assert not any(
        "CAYU_PROTECTED_STAGE" in call.get("envs", {}) for call in sandbox.commands.calls
    )


def test_e2b_handoff_cancellation_kills_created_sandbox() -> None:
    async def run() -> FakeSandbox:
        reset_fake_e2b()
        probe_started = asyncio.Event()

        async def block_probe(_runner: E2BRunner) -> None:
            probe_started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(
            E2BRunner.create_hardened(
                guest_probe=block_probe,
                cleanup_timeout_s=0.2,
                e2b_module=FakeE2BModule,
            )
        )
        await probe_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        sandbox = FakeAsyncSandbox.next_sandbox
        assert sandbox is not None
        return sandbox

    sandbox = asyncio.run(run())

    assert sandbox.kill_calls == 1


def test_e2b_handoff_preserves_cancellation_pending_beside_primary_failure() -> None:
    primary = RuntimeError("guest probe failed")

    async def fail_with_pending_cancellation(_runner: E2BRunner) -> None:
        current_task = asyncio.current_task()
        assert current_task is not None
        current_task.cancel()
        raise primary

    async def run() -> tuple[BaseExceptionGroup, FakeSandbox]:
        reset_fake_e2b()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await E2BRunner.create_hardened(
                guest_probe=fail_with_pending_cancellation,
                cleanup_timeout_s=0.2,
                e2b_module=FakeE2BModule,
            )
        sandbox = FakeAsyncSandbox.next_sandbox
        assert sandbox is not None
        return exc_info.value, sandbox

    error, sandbox = asyncio.run(run())

    assert error.exceptions[0] is primary
    assert isinstance(error.exceptions[1], asyncio.CancelledError)
    assert sandbox.kill_calls == 1


def test_e2b_handoff_preserves_repeated_cancellation_during_cleanup() -> None:
    async def run() -> tuple[
        asyncio.Task[E2BRunner], asyncio.CancelledError, CoordinatedKillSandbox
    ]:
        reset_fake_e2b()
        sandbox = CoordinatedKillSandbox()
        FakeAsyncSandbox.next_sandbox = sandbox
        probe_started = asyncio.Event()

        async def block_probe(_runner: E2BRunner) -> None:
            probe_started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(
            E2BRunner.create_hardened(
                guest_probe=block_probe,
                cleanup_timeout_s=0.5,
                e2b_module=FakeE2BModule,
            )
        )
        await probe_started.wait()
        task.cancel()
        await sandbox.kill_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        sandbox.release_kill.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return task, exc_info.value, sandbox

    task, error, sandbox = asyncio.run(run())

    assert isinstance(error, asyncio.CancelledError)
    assert task.cancelled() is True
    assert task.cancelling() == 1
    assert sandbox.kill_calls == 1


def test_e2b_handoff_cancellation_remains_authoritative_when_rollback_fails() -> None:
    async def run() -> tuple[asyncio.Task[E2BRunner], asyncio.CancelledError]:
        reset_fake_e2b()
        sandbox = FakeSandbox()
        sandbox.fail_kill = True
        FakeAsyncSandbox.next_sandbox = sandbox
        probe_started = asyncio.Event()

        async def block_probe(_runner: E2BRunner) -> None:
            probe_started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(
            E2BRunner.create_hardened(
                guest_probe=block_probe,
                cleanup_timeout_s=0.2,
                e2b_module=FakeE2BModule,
            )
        )
        await probe_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return task, exc_info.value

    task, error = asyncio.run(run())

    assert task.cancelled() is True
    assert task.cancelling() == 0
    assert isinstance(error.__cause__, RuntimeError)
    assert "sandbox kill failed" in str(error.__cause__)
    assert error.__notes__ == ["E2B guest handoff rollback incomplete: RuntimeError."]


def test_e2b_handoff_cancellation_reconciles_ambiguous_create_by_metadata() -> None:
    async def run() -> None:
        AmbiguousCreateAsyncSandbox.active = {}
        AmbiguousCreateAsyncSandbox.started = asyncio.Event()
        AmbiguousCreateAsyncSandbox.allocate = True
        AmbiguousCreateAsyncSandbox.publish_after_cancel = True
        AmbiguousCreateAsyncSandbox.publish_delay_s = 0.05
        AmbiguousCreateAsyncSandbox.list_calls = []
        AmbiguousCreateAsyncSandbox.kill_calls = []
        AmbiguousCreateAsyncSandbox.late_tasks = set()
        task = asyncio.create_task(
            E2BRunner.create_hardened(
                cleanup_timeout_s=0.3,
                e2b_module=AmbiguousCreateE2BModule,
                api_key="secret-api-key",
            )
        )
        await AmbiguousCreateAsyncSandbox.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

    assert AmbiguousCreateAsyncSandbox.active == {}
    assert AmbiguousCreateAsyncSandbox.kill_calls[0]["sandbox_id"] == "e2b_ambiguous"
    queries = [call["query"].metadata for call in AmbiguousCreateAsyncSandbox.list_calls]
    assert all(query is not None and len(query["cayu_guest_handoff_id"]) == 32 for query in queries)
    assert all(
        call["api_key"] == "secret-api-key" for call in AmbiguousCreateAsyncSandbox.list_calls
    )


def test_e2b_handoff_watches_full_cleanup_window_for_late_allocation() -> None:
    async def run() -> None:
        AmbiguousCreateAsyncSandbox.active = {}
        AmbiguousCreateAsyncSandbox.started = asyncio.Event()
        AmbiguousCreateAsyncSandbox.allocate = True
        AmbiguousCreateAsyncSandbox.publish_after_cancel = True
        AmbiguousCreateAsyncSandbox.publish_delay_s = 1.1
        AmbiguousCreateAsyncSandbox.list_calls = []
        AmbiguousCreateAsyncSandbox.kill_calls = []
        AmbiguousCreateAsyncSandbox.late_tasks = set()
        task = asyncio.create_task(
            E2BRunner.create_hardened(
                cleanup_timeout_s=2.0,
                e2b_module=AmbiguousCreateE2BModule,
            )
        )
        await AmbiguousCreateAsyncSandbox.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

    assert AmbiguousCreateAsyncSandbox.active == {}
    assert AmbiguousCreateAsyncSandbox.kill_calls[0]["sandbox_id"] == "e2b_ambiguous"
    assert len(AmbiguousCreateAsyncSandbox.list_calls) >= 5


def test_e2b_handoff_reports_unresolved_ambiguous_create_cleanup() -> None:
    async def run() -> tuple[asyncio.Task[E2BRunner], asyncio.CancelledError]:
        AmbiguousCreateAsyncSandbox.active = {}
        AmbiguousCreateAsyncSandbox.started = asyncio.Event()
        AmbiguousCreateAsyncSandbox.allocate = False
        AmbiguousCreateAsyncSandbox.publish_after_cancel = False
        AmbiguousCreateAsyncSandbox.publish_delay_s = 0.05
        AmbiguousCreateAsyncSandbox.list_calls = []
        AmbiguousCreateAsyncSandbox.kill_calls = []
        AmbiguousCreateAsyncSandbox.late_tasks = set()
        task = asyncio.create_task(
            E2BRunner.create_hardened(
                cleanup_timeout_s=0.2,
                e2b_module=AmbiguousCreateE2BModule,
            )
        )
        await AmbiguousCreateAsyncSandbox.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return task, exc_info.value

    task, error = asyncio.run(run())

    assert task.cancelled() is True
    assert task.cancelling() == 0
    assert isinstance(error.__cause__, TimeoutError)
    assert "reconciliation remained ambiguous" in str(error.__cause__)
    assert error.__notes__ == ["E2B guest handoff rollback incomplete: TimeoutError."]
    assert AmbiguousCreateAsyncSandbox.kill_calls == []


def test_e2b_handoff_kills_create_result_that_arrives_after_cleanup_window() -> None:
    async def run() -> BaseExceptionGroup:
        LateReturningCreateAsyncSandbox.sandbox = FakeSandbox("e2b_late_return")
        LateReturningCreateAsyncSandbox.started = asyncio.Event()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await E2BRunner.create_hardened(
                handoff_timeout_s=0.01,
                cleanup_timeout_s=0.02,
                e2b_module=LateReturningCreateE2BModule,
            )
        await asyncio.sleep(0.08)
        return exc_info.value

    error = asyncio.run(run())

    assert isinstance(error.exceptions[0], TimeoutError)
    assert isinstance(error.exceptions[1], TimeoutError)
    assert LateReturningCreateAsyncSandbox.sandbox.kill_calls == 1
    assert LateReturningCreateAsyncSandbox.sandbox.commands.calls == []


def test_e2b_handoff_timeout_reconciles_late_ambiguous_create() -> None:
    async def run() -> None:
        AmbiguousCreateAsyncSandbox.active = {}
        AmbiguousCreateAsyncSandbox.started = asyncio.Event()
        AmbiguousCreateAsyncSandbox.allocate = True
        AmbiguousCreateAsyncSandbox.publish_after_cancel = True
        AmbiguousCreateAsyncSandbox.publish_delay_s = 0.05
        AmbiguousCreateAsyncSandbox.list_calls = []
        AmbiguousCreateAsyncSandbox.kill_calls = []
        AmbiguousCreateAsyncSandbox.late_tasks = set()
        with pytest.raises(TimeoutError):
            await E2BRunner.create_hardened(
                handoff_timeout_s=0.01,
                cleanup_timeout_s=0.3,
                e2b_module=AmbiguousCreateE2BModule,
            )

    asyncio.run(run())

    assert AmbiguousCreateAsyncSandbox.active == {}
    assert AmbiguousCreateAsyncSandbox.kill_calls[0]["sandbox_id"] == "e2b_ambiguous"


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
