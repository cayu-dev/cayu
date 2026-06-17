from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cayu.runners.base import ExecCommand, ExecResult, RunnerCancelledError
from cayu.runners.docker import (
    DEFAULT_DOCKER_CWD,
    DOCKER_COMMAND_STATE_DIR,
    DockerRunner,
    _build_docker_exec_argv,
    _require_docker,
)


def test_require_docker_uses_explicit_path():
    assert _require_docker("/usr/bin/docker") == "/usr/bin/docker"


def test_require_docker_missing_raises(monkeypatch):
    monkeypatch.setattr("cayu.runners.docker.shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="docker CLI not found"):
        _require_docker(None)


def test_runner_init_and_resolve_cwd():
    r = DockerRunner("agent1", docker_path="/usr/bin/docker")
    assert r.name == "agent1"
    assert r.default_cwd == DEFAULT_DOCKER_CWD
    assert r.resolve_cwd() == DEFAULT_DOCKER_CWD
    assert r.resolve_cwd("subdir") == "/workspace/subdir"
    assert r.close_action == "none"
    assert r.isolation == "docker"


def test_runner_rejects_bad_close_action():
    bad_action: Any = "bogus"
    with pytest.raises(ValueError, match="close_action"):
        DockerRunner("a", docker_path="/usr/bin/docker", close_action=bad_action)


def test_runner_rejects_relative_default_cwd():
    with pytest.raises(ValueError, match="absolute"):
        DockerRunner("a", docker_path="/usr/bin/docker", default_cwd="relative/dir")


def test_runner_normalizes_default_cwd():
    r = DockerRunner("a", docker_path="/usr/bin/docker", default_cwd="/workspace/../work")
    assert r.default_cwd == "/work"


def test_build_exec_argv_process():
    argv = _build_docker_exec_argv(
        "/usr/bin/docker",
        "a1",
        ExecCommand.process("whois", "x.ai"),
        cwd="/workspace",
        env=None,
        has_stdin=False,
        pid_file="/tmp/cayu-docker-commands/cmd.pid",
    )
    assert argv[:6] == ["/usr/bin/docker", "exec", "-w", "/workspace", "a1", "sh"]
    assert argv[6] == "-c"
    assert "setsid" in argv[7]
    assert "/tmp/cayu-docker-commands/cmd.pid" in argv[7]
    assert "whois x.ai" in argv[7]
    assert " & " not in argv[7]
    assert "> /tmp/cayu-docker-commands/cmd.pid || exit 1" in argv[7]


def test_build_exec_argv_shell_env_stdin():
    argv = _build_docker_exec_argv(
        "/usr/bin/docker",
        "a1",
        ExecCommand.bash("echo hi"),
        cwd="/workspace",
        env={"K": "v"},
        has_stdin=True,
        pid_file="/tmp/cayu-docker-commands/cmd.pid",
    )
    assert argv[:9] == [
        "/usr/bin/docker",
        "exec",
        "-i",
        "-w",
        "/workspace",
        "-e",
        "K=v",
        "a1",
        "sh",
    ]
    assert argv[9] == "-c"
    assert "setsid" in argv[10]
    assert "/tmp/cayu-docker-commands/cmd.pid" in argv[10]
    assert "echo hi" in argv[10]
    assert " & " not in argv[10]
    assert "> /tmp/cayu-docker-commands/cmd.pid || exit 1" in argv[10]


def test_exec_forwards_to_run_subprocess(monkeypatch):
    calls = {}

    async def fake_run_subprocess(command, **kwargs):
        calls["argv"] = command.argv
        calls["kwargs"] = kwargs
        return ExecResult(stdout="ok")

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    r = DockerRunner("a1", docker_path="/usr/bin/docker")
    result = asyncio.run(
        r.exec(ExecCommand.process("whoami"), timeout_s=12, output_limit_bytes=999)
    )
    assert result.stdout == "ok"
    assert calls["argv"][:6] == ["/usr/bin/docker", "exec", "-w", "/workspace", "a1", "sh"]
    assert calls["argv"][6] == "-c"
    assert "setsid" in calls["argv"][7]
    assert DOCKER_COMMAND_STATE_DIR in calls["argv"][7]
    assert "whoami" in calls["argv"][7]
    assert calls["kwargs"]["timeout_s"] == 12
    assert calls["kwargs"]["output_limit_bytes"] == 999
    # env boundary: host docker process inherits host env (PATH present).
    assert "PATH" in calls["kwargs"]["env"]


def test_exec_keeps_stdin_attached_to_supervised_command(monkeypatch):
    calls = {}

    async def fake_run_subprocess(command, **kwargs):
        calls["argv"] = command.argv
        calls["kwargs"] = kwargs
        return ExecResult(stdout="hello")

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    r = DockerRunner("a1", docker_path="/usr/bin/docker")

    result = asyncio.run(r.exec(ExecCommand.process("cat"), stdin="hello", timeout_s=12))

    assert result.stdout == "hello"
    assert calls["argv"][2] == "-i"
    assert "cat" in calls["argv"][-1]
    assert " & " not in calls["argv"][-1]
    assert calls["kwargs"]["stdin"] == "hello"


def test_exec_on_closed_runner_raises(monkeypatch):
    monkeypatch.setattr(
        "cayu.runners.docker.run_subprocess",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    async def run():
        r = DockerRunner("a1", docker_path="/usr/bin/docker")
        r._closed = True
        await r.exec(ExecCommand.process("whoami"))

    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(run())


def test_close_remove_and_idempotent(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    async def run():
        r = DockerRunner("a1", docker_path="/usr/bin/docker", close_action="remove")
        await r.close()
        await r.close()  # idempotent

    asyncio.run(run())
    assert issued.count(["/usr/bin/docker", "rm", "-f", "a1"]) == 1


def test_close_stop_and_none(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    asyncio.run(DockerRunner("s", docker_path="/usr/bin/docker", close_action="stop").close())
    assert issued == [["/usr/bin/docker", "stop", "s"]]

    issued.clear()
    asyncio.run(DockerRunner("n", docker_path="/usr/bin/docker", close_action="none").close())
    assert issued == []


def test_close_remove_failure_keeps_runner_open(monkeypatch):
    async def fake_run_subprocess(command, **kwargs):
        return ExecResult(stderr="nope", exit_code=1)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    r = DockerRunner("a1", docker_path="/usr/bin/docker", close_action="remove")

    with pytest.raises(RuntimeError, match="docker rm failed"):
        asyncio.run(r.close())
    assert r._closed is False


def test_close_stop_failure_keeps_runner_open(monkeypatch):
    async def fake_run_subprocess(command, **kwargs):
        return ExecResult(stderr="nope", exit_code=1)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    r = DockerRunner("a1", docker_path="/usr/bin/docker", close_action="stop")

    with pytest.raises(RuntimeError, match="docker stop failed"):
        asyncio.run(r.close())
    assert r._closed is False


def test_create_bind_mount_mode(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    runner = asyncio.run(
        DockerRunner.create(
            "a1",
            docker_path="/usr/bin/docker",
            image="debian:stable-slim",
            mount_path="/host/ws",
            setup_commands=("apt-get install -y whois",),
        )
    )
    # replace removes any stale container first
    assert ["/usr/bin/docker", "rm", "-f", "a1"] in issued
    # bind-mount at same absolute path, no --runtime, keep-alive command
    assert [
        "/usr/bin/docker",
        "run",
        "-d",
        "--name",
        "a1",
        "--mount",
        "type=bind,source=/host/ws,target=/host/ws",
        "debian:stable-slim",
        "sleep",
        "infinity",
    ] in issued
    # bind mode: default_cwd defaults to the mount, and NO mkdir exec is issued
    assert runner.default_cwd == "/host/ws"
    assert not any(
        a[:4] == ["/usr/bin/docker", "exec", "-u", "root"] and "mkdir -p /host/ws" in a
        for a in issued
    )
    # setup command runs as root
    assert [
        "/usr/bin/docker",
        "exec",
        "-u",
        "root",
        "a1",
        "sh",
        "-c",
        "apt-get install -y whois",
    ] in issued
    assert runner.close_action == "remove"


def test_create_isolated_mode_with_runtime(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    runner = asyncio.run(
        DockerRunner.create(
            "a1",
            docker_path="/usr/bin/docker",
            runtime="runsc",
            setup_commands=("apt-get install -y whois python3",),
        )
    )
    # --runtime present, no -v bind mount
    assert [
        "/usr/bin/docker",
        "run",
        "-d",
        "--runtime",
        "runsc",
        "--name",
        "a1",
        "debian:stable-slim",
        "sleep",
        "infinity",
    ] in issued
    # isolated mode: default_cwd is /workspace and mkdir runs as root
    assert runner.default_cwd == "/workspace"
    assert [
        "/usr/bin/docker",
        "exec",
        "-u",
        "root",
        "a1",
        "sh",
        "-c",
        "mkdir -p /workspace",
    ] in issued


def test_create_run_failure_cleans_up(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        # fail the `run` step
        if command.argv[1] == "run":
            return ExecResult(exit_code=125, stderr="bad runtime")
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    with pytest.raises(RuntimeError, match="docker run failed"):
        asyncio.run(DockerRunner.create("a1", docker_path="/usr/bin/docker"))
    # cleanup rm -f issued after the failure (in addition to the replace rm)
    assert issued.count(["/usr/bin/docker", "rm", "-f", "a1"]) >= 1


def test_resolve_cwd_relative_path():
    r = DockerRunner("a1", docker_path="/usr/bin/docker")
    assert r.resolve_cwd("sub/dir") == "/workspace/sub/dir"


def test_resolve_cwd_rejects_absolute():
    r = DockerRunner("a1", docker_path="/usr/bin/docker")
    with pytest.raises(ValueError, match="relative"):
        r.resolve_cwd("/etc")


def test_resolve_cwd_rejects_escape():
    r = DockerRunner("a1", docker_path="/usr/bin/docker")
    with pytest.raises(ValueError, match="escapes"):
        r.resolve_cwd("../../etc")


def test_exec_timeout_records_cleanup_diagnostic(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        if "kill -TERM" in command.argv[-1]:
            return ExecResult()
        return ExecResult(timed_out=True, exit_code=-9)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    r = DockerRunner("a1", docker_path="/usr/bin/docker", close_action="none")
    result = asyncio.run(r.exec(ExecCommand.process("sleep", "999"), timeout_s=1))

    assert result.timed_out is True
    assert r._closed is False
    assert r._exec_closed is False
    assert len(issued) == 2
    assert "setsid" in issued[0][-1]
    assert "kill -TERM" in issued[1][-1]
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "docker",
            "action": "kill_command",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]


def test_exec_timeout_can_remove_container_when_configured(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        if command.argv[:2] == ["/usr/bin/docker", "exec"]:
            return ExecResult(stdout="before", timed_out=True, exit_code=-9)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    r = DockerRunner(
        "a1",
        docker_path="/usr/bin/docker",
        timeout_cleanup="sandbox",
    )

    result = asyncio.run(r.exec(ExecCommand.process("sleep", "999"), timeout_s=1))

    assert result.timed_out is True
    assert r._closed is True
    assert ["/usr/bin/docker", "rm", "-f", "a1"] in issued
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "docker",
            "action": "kill_sandbox",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]


def test_exec_cancellation_raises_runner_cancelled_error(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        if "kill -TERM" in command.argv[-1]:
            return ExecResult()
        raise asyncio.CancelledError

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    r = DockerRunner("a1", docker_path="/usr/bin/docker")

    async def run():
        with pytest.raises(RunnerCancelledError) as exc_info:
            await r.exec(ExecCommand.process("sleep", "999"))
        return exc_info.value

    error = asyncio.run(run())

    assert r._exec_closed is False
    assert len(issued) == 2
    assert "setsid" in issued[0][-1]
    assert "kill -TERM" in issued[1][-1]
    assert error.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "docker",
            "action": "kill_command",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]


def test_exec_marks_exec_closed_when_command_cleanup_fails(monkeypatch):
    async def fake_run_subprocess(command, **kwargs):
        if "kill -TERM" in command.argv[-1]:
            return ExecResult(stderr="cleanup failed", exit_code=1)
        return ExecResult(timed_out=True, exit_code=-9)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    r = DockerRunner("a1", docker_path="/usr/bin/docker")

    result = asyncio.run(r.exec(ExecCommand.process("sleep", "999"), timeout_s=1))

    assert result.timed_out is True
    assert r._closed is False
    assert r._exec_closed is True
    assert (
        r._exec_closed_reason == "docker command cleanup did not complete; command state is unknown"
    )
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "docker",
            "action": "kill_command",
            "status": "failed",
            "timeout_s": 5.0,
            "error": "kill returned false",
        }
    ]


def test_exec_validates_env_before_building_docker_env(monkeypatch):
    async def fake_run_subprocess(command, **kwargs):
        raise AssertionError("run_subprocess should not be called")

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    r = DockerRunner("a1", docker_path="/usr/bin/docker")
    bad_env: Any = {"BAD": 1}

    with pytest.raises(ValueError, match="Runner env values must be strings"):
        asyncio.run(r.exec(ExecCommand.process("env"), env=bad_env))


def test_create_quotes_default_cwd(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    runner = asyncio.run(
        DockerRunner.create(
            "a1",
            docker_path="/usr/bin/docker",
            default_cwd="/work space",
        )
    )
    assert runner.default_cwd == "/work space"
    # The mkdir command should contain the shlex-quoted path
    mkdir_cmds = [
        a for a in issued if a[:4] == ["/usr/bin/docker", "exec", "-u", "root"] and "mkdir" in a[-1]
    ]
    assert len(mkdir_cmds) == 1
    assert mkdir_cmds[0][-1] == "mkdir -p '/work space'"
