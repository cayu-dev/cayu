"""Tests for the worked `examples/modal_runner.py` (loaded via importlib).

The example is not importable from `cayu`, so we load it from disk and drive it
with an injected fake Modal SDK — no `modal` install or account needed.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from cayu import ExecCommand, RunnerCancelledError

_EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "modal_runner.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("modal_runner_example", _EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_example()


class _Aio:
    """Mimics Modal's `.aio()` async accessor over an async function."""

    def __init__(self, fn):
        self._fn = fn

    async def aio(self, *a, **k):
        return await self._fn(*a, **k)


class _Reader:
    def __init__(self, data, *, cancel=False):
        async def read():
            if cancel:
                raise asyncio.CancelledError
            return data

        self.read = _Aio(read)


class _Stdin:
    def __init__(self):
        self.written = b""
        self.eof = False
        self.drain = _Aio(lambda: asyncio.sleep(0))

    def write(self, data):
        self.written += data

    def write_eof(self):
        self.eof = True


class _Process:
    def __init__(self, out="", err="", code=0, *, cancel=False):
        self.stdout = _Reader(out, cancel=cancel)
        self.stderr = _Reader(err)
        self.stdin = _Stdin()

        async def wait():
            return code

        self.wait = _Aio(wait)


class FakeSandbox:
    def __init__(self, out="", err="", code=0, *, cancel=False, mkdir_code=0):
        self.exec_calls: list[dict[str, Any]] = []
        self.last_process: _Process | None = None
        self.terminated = False
        self._out, self._err, self._code = out, err, code
        self._cancel = cancel
        self._mkdir_code = mkdir_code

        async def _exec(*argv, workdir=None, env=None, timeout=None):
            self.exec_calls.append(
                {"argv": list(argv), "workdir": workdir, "env": env, "timeout": timeout}
            )
            if argv and argv[0] == "mkdir":  # guest-root creation in create()
                return _Process(code=self._mkdir_code)
            self.last_process = _Process(self._out, self._err, self._code, cancel=self._cancel)
            return self.last_process

        self.exec = _Aio(_exec)

        async def _terminate():
            self.terminated = True

        self.terminate = _Aio(_terminate)


class _FakeImage:
    def __init__(self):
        self.debian_slim_calls = 0

    def debian_slim(self):
        self.debian_slim_calls += 1
        return {"image": "debian_slim"}


class FakeModalModule:
    def __init__(self, sandbox: FakeSandbox):
        self.Image = _FakeImage()
        self._sandbox = sandbox
        self.create_kwargs: dict[str, Any] | None = None

        async def _create(**kwargs):
            self.create_kwargs = kwargs
            return sandbox

        # module.Sandbox.create.aio(...)
        self.Sandbox = type("Sandbox", (), {"create": _Aio(_create)})


# ---- create(): image default (#1) + guest-root creation (#5) ----


def test_create_builds_default_image_and_creates_guest_root() -> None:
    sandbox = FakeSandbox(out="ok")
    module = FakeModalModule(sandbox)

    async def run():
        return await mod.ModalRunner.create(app="app", modal_module=module)

    runner = asyncio.run(run())
    assert module.Image.debian_slim_calls == 1
    assert module.create_kwargs is not None
    assert module.create_kwargs["image"] == {"image": "debian_slim"}
    assert sandbox.exec_calls[0]["argv"] == ["mkdir", "-p", "/workspace"]
    assert runner.default_cwd == "/workspace"


def test_create_keeps_an_explicit_image() -> None:
    sandbox = FakeSandbox()
    module = FakeModalModule(sandbox)
    sentinel = {"image": "custom"}

    async def run():
        await mod.ModalRunner.create(app="app", image=sentinel, modal_module=module)

    asyncio.run(run())
    assert module.Image.debian_slim_calls == 0
    assert module.create_kwargs is not None
    assert module.create_kwargs["image"] is sentinel


def test_create_terminates_sandbox_when_guest_root_fails() -> None:
    # A provisioned (billable) sandbox must not leak if guest-root setup fails.
    sandbox = FakeSandbox(mkdir_code=1)
    module = FakeModalModule(sandbox)

    async def run():
        await mod.ModalRunner.create(app="app", modal_module=module)

    with pytest.raises(RuntimeError, match="guest root"):
        asyncio.run(run())
    assert sandbox.terminated is True


# ---- exec(): translation, env, truncation, exit, timeout, cancellation ----


def test_exec_translates_argv_env_cwd_and_passes_native_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = FakeSandbox(out="abcdef")
    monkeypatch.setenv("HOSTSECRET", "leak")

    async def run():
        runner = mod.ModalRunner(sandbox)
        return await runner.exec(
            ExecCommand.process("echo", "abcdef"),
            cwd="src",
            env={"VISIBLE": "1"},
            timeout_s=7,
            output_limit_bytes=3,
        )

    result = asyncio.run(run())
    call = sandbox.exec_calls[0]
    assert call["argv"] == ["echo", "abcdef"]
    assert call["workdir"] == "/workspace/src"
    assert call["env"] == {"VISIBLE": "1"}  # no host env leaked in
    assert "HOSTSECRET" not in call["env"]  # host env is not inherited
    assert call["timeout"] == 7  # native exec timeout is passed through
    assert result.stdout == "abc" and result.stdout_truncated is True


def test_exec_shell_runs_under_bash() -> None:
    sandbox = FakeSandbox()

    async def run():
        await mod.ModalRunner(sandbox).exec(ExecCommand.bash("ls | wc -l"))

    asyncio.run(run())
    assert sandbox.exec_calls[0]["argv"] == ["bash", "-c", "ls | wc -l"]


def test_exec_truncates_stderr() -> None:
    sandbox = FakeSandbox(err="erroroutput")

    async def run():
        return await mod.ModalRunner(sandbox).exec(ExecCommand.process("x"), output_limit_bytes=3)

    result = asyncio.run(run())
    assert result.stderr == "err" and result.stderr_truncated is True


def test_exec_forwards_stdin_to_the_process() -> None:
    sandbox = FakeSandbox(out="hi\n")

    async def run():
        await mod.ModalRunner(sandbox).exec(ExecCommand.process("cat"), stdin="hi\n")

    asyncio.run(run())
    assert sandbox.last_process is not None
    assert sandbox.last_process.stdin.written == b"hi\n"
    assert sandbox.last_process.stdin.eof is True


def test_exec_coerces_non_int_exit_code() -> None:
    sandbox = FakeSandbox(code=None)

    async def run():
        return await mod.ModalRunner(sandbox).exec(ExecCommand.process("x"))

    assert asyncio.run(run()).exit_code == -1


def test_exec_timeout_preserves_partial_output_and_maps_to_minus_nine() -> None:
    # Model Modal's native timeout: the command is killed server-side -> exit_code -1,
    # with the output produced before the kill still readable.
    sandbox = FakeSandbox(out="PARTIAL\n", code=-1)

    async def run():
        return await mod.ModalRunner(sandbox).exec(
            ExecCommand.bash("echo PARTIAL; sleep 9"), timeout_s=1
        )

    result = asyncio.run(run())
    assert result.timed_out is True
    assert result.exit_code == -9  # runner timeout contract, matching built-in runners
    assert result.stdout == "PARTIAL\n"  # partial output preserved, not discarded
    assert sandbox.terminated is False  # a single command timeout must not kill the sandbox
    assert sandbox.exec_calls[-1]["timeout"] == 1  # native exec timeout passed through


def test_exec_minus_one_without_timeout_is_not_treated_as_timeout() -> None:
    # exit_code -1 only means "timed out" when a timeout was requested; otherwise it is
    # just an unknown/killed exit and must not be mislabeled timed_out.
    sandbox = FakeSandbox(code=-1)

    async def run():
        return await mod.ModalRunner(sandbox).exec(ExecCommand.process("x"))

    result = asyncio.run(run())
    assert result.timed_out is False and result.exit_code == -1


def test_exec_cancellation_terminates_sandbox_with_diagnostic() -> None:
    sandbox = FakeSandbox(cancel=True)

    async def run():
        await mod.ModalRunner(sandbox).exec(ExecCommand.process("x"))

    with pytest.raises(RunnerCancelledError) as excinfo:
        asyncio.run(run())
    assert sandbox.terminated is True
    artifact = excinfo.value.artifacts[0]
    assert artifact["type"] == "cayu.runner_cleanup.v1"
    assert artifact["adapter"] == "modal"
    assert artifact["action"] == "kill_sandbox"
    assert artifact["status"] == "ok"


def test_exec_rejects_cwd_escape() -> None:
    async def run():
        await mod.ModalRunner(FakeSandbox()).exec(ExecCommand.process("x"), cwd="../etc")

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(run())


def test_resolve_cwd_is_idempotent_for_contained_canonical_paths() -> None:
    runner = mod.ModalRunner(FakeSandbox())

    assert runner.resolve_cwd("/workspace/src/../tests") == "/workspace/tests"
    with pytest.raises(ValueError, match="outside the runner root"):
        runner.resolve_cwd("/etc")


# ---- _truncate handles both bytes and str (#8) ----


def test_truncate_handles_bytes_and_str() -> None:
    assert mod._truncate("abcdef", 3) == ("abc", True)
    assert mod._truncate(b"abcdef", 3) == ("abc", True)
    assert mod._truncate("hi", None) == ("hi", False)
    assert mod._truncate(b"hi", None) == ("hi", False)
