from __future__ import annotations

import asyncio
import errno
import io
import shlex
import subprocess
import sys

import pytest

from cayu.runners._diagnostics import runner_failure_fields
from cayu.runners._subprocess import SubprocessCommand, SubprocessLaunchRefused, run_subprocess
from cayu.runners.base import ExecCommand, ExecResult
from cayu.runners.docker import DockerRunner, _build_docker_exec_argv


def _argv(command, *, direct=False):
    return _build_docker_exec_argv(
        "docker",
        "test",
        command,
        cwd="/tmp",
        env_file=None,
        has_stdin=True,
        pid_file="/tmp/synthetic.pid",
        direct_process_supervisor=direct,
    )


@pytest.mark.parametrize("direct, shell", [(False, False), (True, False), (False, True)])
def test_quote_heavy_command_transport_preserves_execution(tmp_path, direct, shell):
    if direct and sys.platform != "linux":
        pytest.skip("Python process supervisor requires Linux prctl")
    raw = "import sys; print(" + repr(["alpha'beta"] * 1000) + "); print(sys.stdin.read())"
    command = ExecCommand.process("python3", "-c", raw)
    if shell:
        command = ExecCommand.bash(shlex.join(command.argv))
    argv = _argv(command, direct=direct)
    guest = argv[argv.index("test") + 1 :]
    # Use a per-test state path, leaving stdin exclusively for the guest command.
    guest = [arg.replace("/tmp/synthetic.pid", str(tmp_path / "command.pid")) for arg in guest]
    assert max(len(arg.encode()) for arg in argv) < 64 * 1024
    result = subprocess.run(guest, input="stdin-canary", text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(["alpha'beta"] * 1000) + "\nstdin-canary\n"
    assert not (tmp_path / "command.pid").exists()


@pytest.mark.parametrize("direct, shell", [(False, False), (True, False), (False, True)])
def test_oversized_transport_is_refused_without_dispatch(monkeypatch, direct, shell):
    command = ExecCommand.process("echo", "secret-canary" * 12000)
    if shell:
        command = ExecCommand.bash(shlex.join(command.argv))
    with pytest.raises(ValueError, match="local admission limit") as caught:
        _argv(command, direct=direct)
    assert "secret-canary" not in str(caught.value)

    async def unexpected(*args, **kwargs):
        pytest.fail("Admission refusal attempted dispatch or guest cleanup")

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", unexpected)
    runner = DockerRunner("test", docker_path="docker")
    with pytest.raises(ValueError, match="local admission limit"):
        asyncio.run(runner.exec(command))
    assert not runner._exec_closed


def test_aggregate_transport_limit():
    with pytest.raises(ValueError, match="local admission limit"):
        _argv(ExecCommand.process("echo", *(["a" * 1000] * 140)), direct=True)


def test_spawn_e2big_is_proven_refusal_without_guest_cleanup(monkeypatch):
    calls = 0

    async def reject(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise OSError(errno.E2BIG, "secret-canary")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", reject)
    runner = DockerRunner("test", docker_path="docker")
    with pytest.raises(SubprocessLaunchRefused) as caught:
        asyncio.run(runner.exec(ExecCommand.process("true")))
    assert runner_failure_fields(caught.value) == {
        "errno": errno.E2BIG,
        "errno_code": "E2BIG",
        "execution_phase": "launch",
    }
    assert caught.value.errno == errno.E2BIG
    assert "secret-canary" not in str(caught.value)
    assert not getattr(caught.value, "artifacts", None)
    assert calls == 1
    assert not runner._exec_closed


@pytest.mark.parametrize("error", [OSError(errno.E2BIG, "ambiguous"), OSError(errno.EIO, "io")])
def test_unproven_oserror_still_requires_guest_cleanup(monkeypatch, error):
    calls = []

    async def fail(command, **kwargs):
        calls.append(command.argv)
        if "kill -TERM" in command.argv[-1]:
            return ExecResult()
        raise error

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fail)
    runner = DockerRunner("test", docker_path="docker")
    with pytest.raises(OSError) as caught:
        asyncio.run(runner.exec(ExecCommand.process("true")))
    assert caught.value is error
    assert len(calls) == 2
    assert getattr(error, "artifacts", [])[0]["status"] == "completed"


def test_spawn_other_oserror_is_not_proven_refusal(monkeypatch):
    async def reject(*args, **kwargs):
        raise OSError(errno.EIO, "ambiguous spawn failure")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", reject)
    with pytest.raises(OSError) as caught:
        asyncio.run(run_subprocess(SubprocessCommand(argv=["true"])))
    assert not isinstance(caught.value, SubprocessLaunchRefused)


def test_post_launch_e2big_from_stream_is_not_a_spawn_refusal():
    class FailingSource(io.BytesIO):
        def read(self, size=-1):
            raise OSError(errno.E2BIG, "stream failure")

    with pytest.raises(OSError) as caught:
        asyncio.run(
            run_subprocess(
                SubprocessCommand(argv=[sys.executable, "-c", "import sys; sys.stdin.read()"]),
                stdin_stream=FailingSource(),
            )
        )
    assert not isinstance(caught.value, SubprocessLaunchRefused)
    assert runner_failure_fields(caught.value)["execution_phase"] == "stream_handling"
