from __future__ import annotations

import asyncio

import pytest

from cayu.runners.base import ExecCommand, ExecResult
from cayu.runners.sbx import (
    DEFAULT_SBX_CWD,
    SBX_COMMAND_STATE_DIR,
    SbxRunner,
    _build_sbx_exec_argv,
    _require_sbx,
)
from cayu.vaults import REDACTED_SECRET, SecretEnv, SecretRef, StaticVault


def test_require_sbx_uses_explicit_path():
    assert _require_sbx("/usr/bin/sbx") == "/usr/bin/sbx"


def test_require_sbx_missing_raises(monkeypatch):
    monkeypatch.setattr("cayu.runners.sbx.shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="sbx CLI not found"):
        _require_sbx(None)


def test_runner_init_and_resolve_cwd():
    r = SbxRunner("agent1", mount_path="/tmp/m", sbx_path="/usr/bin/sbx")
    assert r.name == "agent1"
    assert r.default_cwd == DEFAULT_SBX_CWD
    assert r.resolve_cwd() == DEFAULT_SBX_CWD
    assert r.resolve_cwd("subdir") == "/workspace/subdir"
    assert r.resolve_cwd("/workspace/subdir") == "/workspace/subdir"
    with pytest.raises(ValueError, match="outside the runner root"):
        r.resolve_cwd("/x")
    with pytest.raises(ValueError, match="escapes"):
        r.resolve_cwd("../../etc")
    assert r.close_action == "none"
    assert r.isolation == "sbx"


def test_runner_rejects_bad_close_action():
    with pytest.raises(ValueError, match="close_action"):
        SbxRunner("a", mount_path="/tmp/m", sbx_path="/usr/bin/sbx", close_action="bogus")


def test_build_exec_argv_process():
    argv = _build_sbx_exec_argv(
        "/usr/bin/sbx",
        "a1",
        ExecCommand.process("whois", "x.ai"),
        cwd="/workspace",
        env_file=None,
        has_stdin=False,
        pid_file="/tmp/cayu-sbx-commands/cmd.pid",
    )
    assert argv[:6] == ["/usr/bin/sbx", "exec", "-w", "/workspace", "a1", "sh"]
    assert argv[6] == "-c"
    assert "setsid" in argv[7]
    assert "/tmp/cayu-sbx-commands/cmd.pid" in argv[7]
    assert "whois x.ai" in argv[7]
    assert " & " not in argv[7]
    assert "> /tmp/cayu-sbx-commands/cmd.pid || exit 1" in argv[7]


def test_build_exec_argv_shell_env_stdin():
    argv = _build_sbx_exec_argv(
        "/usr/bin/sbx",
        "a1",
        ExecCommand.bash("echo hi"),
        cwd="/workspace",
        env_file="/tmp/cayu-runner-env-abc",
        has_stdin=True,
        pid_file="/tmp/cayu-sbx-commands/cmd.pid",
    )
    # Env is passed via --env-file; values never appear in argv.
    assert argv[:9] == [
        "/usr/bin/sbx",
        "exec",
        "-i",
        "-w",
        "/workspace",
        "--env-file",
        "/tmp/cayu-runner-env-abc",
        "a1",
        "sh",
    ]
    assert not any("K=v" in item for item in argv)
    assert argv[9] == "-c"
    assert "setsid" in argv[10]
    assert "/tmp/cayu-sbx-commands/cmd.pid" in argv[10]
    assert "echo hi" in argv[10]
    assert " & " not in argv[10]
    assert "> /tmp/cayu-sbx-commands/cmd.pid || exit 1" in argv[10]


def test_exec_forwards_to_run_subprocess(monkeypatch):
    calls = {}

    async def fake_run_subprocess(command, **kwargs):
        calls["argv"] = command.argv
        calls["kwargs"] = kwargs
        return ExecResult(stdout="ok")

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    r = SbxRunner("a1", mount_path="/tmp/m", sbx_path="/usr/bin/sbx")
    result = asyncio.run(
        r.exec(ExecCommand.process("whoami"), timeout_s=12, output_limit_bytes=999)
    )
    assert result.stdout == "ok"
    assert calls["argv"][:6] == ["/usr/bin/sbx", "exec", "-w", "/workspace", "a1", "sh"]
    assert calls["argv"][6] == "-c"
    assert "setsid" in calls["argv"][7]
    assert SBX_COMMAND_STATE_DIR in calls["argv"][7]
    assert "whoami" in calls["argv"][7]
    assert " & " not in calls["argv"][7]
    assert calls["kwargs"]["timeout_s"] == 12
    assert calls["kwargs"]["output_limit_bytes"] == 999
    assert "PATH" in calls["kwargs"]["env"]


def test_exec_keeps_stdin_attached_to_supervised_command(monkeypatch):
    calls = {}

    async def fake_run_subprocess(command, **kwargs):
        calls["argv"] = command.argv
        calls["kwargs"] = kwargs
        return ExecResult(stdout=kwargs["stdin"])

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    r = SbxRunner("a1", mount_path="/tmp/m", sbx_path="/usr/bin/sbx")

    result = asyncio.run(r.exec(ExecCommand.process("cat"), stdin="hello", timeout_s=12))

    assert result.stdout == "hello"
    assert "-i" in calls["argv"]
    assert " & " not in calls["argv"][-1]
    assert calls["kwargs"]["stdin"] == "hello"


def test_create_issues_expected_sbx_calls(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    monkeypatch.setattr("cayu.runners.sbx.tempfile.mkdtemp", lambda prefix: "/tmp/cayu-sbx-x")

    runner = asyncio.run(
        SbxRunner.create("a1", sbx_path="/usr/bin/sbx", setup_commands=["apt-get install -y whois"])
    )
    assert ["/usr/bin/sbx", "rm", "--force", "a1"] in issued
    assert ["/usr/bin/sbx", "create", "--name", "a1", "shell", "/tmp/cayu-sbx-x"] in issued
    assert [
        "/usr/bin/sbx",
        "exec",
        "-u",
        "root",
        "a1",
        "sh",
        "-c",
        "mkdir -p /workspace && chmod 0777 /workspace",
    ] in issued
    assert [
        "/usr/bin/sbx",
        "exec",
        "-u",
        "root",
        "a1",
        "sh",
        "-c",
        "apt-get install -y whois",
    ] in issued
    assert runner.mount_path == "/tmp/cayu-sbx-x"
    assert runner._owns_mount is True
    assert runner.close_action == "remove"


def test_close_remove_and_idempotent(monkeypatch):
    issued = []
    removed_dirs = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(
        "cayu.runners.sbx.shutil.rmtree",
        lambda p, ignore_errors=False: removed_dirs.append(p),
    )

    async def run():
        r = SbxRunner(
            "a1",
            mount_path="/tmp/cayu-sbx-x",
            sbx_path="/usr/bin/sbx",
            close_action="remove",
            owns_mount=True,
        )
        await r.close()
        await r.close()  # idempotent

    asyncio.run(run())
    assert issued.count(["/usr/bin/sbx", "rm", "--force", "a1"]) == 1
    assert removed_dirs == ["/tmp/cayu-sbx-x"]


def test_close_stop_and_none(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)

    asyncio.run(
        SbxRunner("s", mount_path="/m", sbx_path="/usr/bin/sbx", close_action="stop").close()
    )
    assert issued == [["/usr/bin/sbx", "stop", "s"]]

    issued.clear()
    asyncio.run(
        SbxRunner("n", mount_path="/m", sbx_path="/usr/bin/sbx", close_action="none").close()
    )
    assert issued == []


def test_resolve_cwd_relative_path():
    r = SbxRunner("a1", mount_path="/tmp/m", sbx_path="/usr/bin/sbx")
    assert r.resolve_cwd("sub/dir") == "/workspace/sub/dir"


def test_resolve_cwd_accepts_contained_absolute_and_rejects_outside():
    r = SbxRunner("a1", mount_path="/tmp/m", sbx_path="/usr/bin/sbx")
    assert r.resolve_cwd("/workspace/sub/../tests") == "/workspace/tests"
    with pytest.raises(ValueError, match="outside the runner root"):
        r.resolve_cwd("/etc")


def test_resolve_cwd_rejects_escape():
    r = SbxRunner("a1", mount_path="/tmp/m", sbx_path="/usr/bin/sbx")
    with pytest.raises(ValueError, match="escapes"):
        r.resolve_cwd("../../etc")


def test_exec_timeout_records_cleanup_diagnostic(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        if "kill -TERM" in command.argv[-1]:
            return ExecResult()
        return ExecResult(stdout="", timed_out=True, exit_code=-9)

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    r = SbxRunner("a1", mount_path="/tmp/m", sbx_path="/usr/bin/sbx")

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
            "adapter": "sbx",
            "action": "kill_command",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]


def test_exec_timeout_can_remove_sandbox_when_configured(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        if command.argv[:2] == ["/usr/bin/sbx", "exec"]:
            return ExecResult(stdout="before", timed_out=True, exit_code=-9)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    r = SbxRunner(
        "a1",
        mount_path="/tmp/m",
        sbx_path="/usr/bin/sbx",
        timeout_cleanup="sandbox",
    )

    result = asyncio.run(r.exec(ExecCommand.process("sleep", "999"), timeout_s=1))

    assert result.timed_out is True
    assert r._closed is True
    assert ["/usr/bin/sbx", "rm", "--force", "a1"] in issued
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "sbx",
            "action": "kill_sandbox",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]


def test_exec_cancellation_reraises_plain_cancelled_error_with_artifacts(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        if "kill -TERM" in command.argv[-1]:
            return ExecResult()
        raise asyncio.CancelledError

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    r = SbxRunner("a1", mount_path="/tmp/m", sbx_path="/usr/bin/sbx")

    async def run():
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await r.exec(ExecCommand.process("sleep", "999"))
        return exc_info.value

    error = asyncio.run(run())

    # The original cancellation propagates unchanged; diagnostics ride out-of-band.
    assert type(error) is asyncio.CancelledError
    assert r._exec_closed is False
    assert len(issued) == 2
    assert "setsid" in issued[0][-1]
    assert "kill -TERM" in issued[1][-1]
    assert error.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "sbx",
            "action": "kill_command",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]


def test_exec_marks_exec_closed_when_command_cleanup_fails(monkeypatch):
    async def fake_run_subprocess(command, **kwargs):
        if "kill -TERM" in command.argv[-1]:
            return ExecResult(stderr="cleanup failed", exit_code=1)
        return ExecResult(stdout="", timed_out=True, exit_code=-9)

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    r = SbxRunner("a1", mount_path="/tmp/m", sbx_path="/usr/bin/sbx")

    result = asyncio.run(r.exec(ExecCommand.process("sleep", "999"), timeout_s=1))

    assert result.timed_out is True
    assert r._closed is False
    assert r._exec_closed is True
    assert r._exec_closed_reason == "sbx command cleanup did not complete; command state is unknown"
    assert result.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "sbx",
            "action": "kill_command",
            "status": "failed",
            "timeout_s": 5.0,
            "error": "kill returned false",
        }
    ]


def test_command_kill_verifies_missing_pid_file_as_stopped(monkeypatch):
    kill_attempts = []

    async def fake_run_subprocess(command, **kwargs):
        if "kill -TERM" in command.argv[-1]:
            kill_attempts.append(command.argv)
            return ExecResult(exit_code=1)
        if command.argv[-1].startswith("test -f"):
            # pid file absent: the supervised command is not running.
            return ExecResult(exit_code=1)
        return ExecResult(timed_out=True, exit_code=-9)

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    r = SbxRunner("a1", mount_path="/tmp/m", sbx_path="/usr/bin/sbx")

    result = asyncio.run(r.exec(ExecCommand.process("sleep", "999"), timeout_s=1))

    assert result.timed_out is True
    assert len(kill_attempts) == 2
    assert r._exec_closed is False
    assert r._exec_closed_reason is None
    assert result.artifacts[0]["status"] == "completed"


def test_reopen_exec_recovers_latched_runner(monkeypatch):
    async def fake_run_subprocess(command, **kwargs):
        if "kill -TERM" in command.argv[-1]:
            return ExecResult(exit_code=1)
        if command.argv[-1].startswith("test -f"):
            # pid file still present: the command state stays unknown.
            return ExecResult(exit_code=0)
        return ExecResult(timed_out=True, exit_code=-9)

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    r = SbxRunner("a1", mount_path="/tmp/m", sbx_path="/usr/bin/sbx")

    result = asyncio.run(r.exec(ExecCommand.process("sleep", "999"), timeout_s=1))
    assert result.timed_out is True
    assert r._exec_closed is True

    with pytest.raises(RuntimeError, match="SbxRunner is closed: sbx command cleanup"):
        asyncio.run(r.exec(ExecCommand.process("true")))

    r.reopen_exec()
    assert r._exec_closed is False


def test_exec_validates_env_before_building_sbx_env(monkeypatch):
    async def fake_run_subprocess(command, **kwargs):
        raise AssertionError("run_subprocess should not be called")

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    r = SbxRunner("a1", mount_path="/tmp/m", sbx_path="/usr/bin/sbx")

    with pytest.raises(ValueError, match="Runner env values must be strings"):
        asyncio.run(r.exec(ExecCommand.process("env"), env={"BAD": 1}))


def test_close_remove_failure_keeps_runner_open(monkeypatch):
    async def fake_run_subprocess(command, **kwargs):
        return ExecResult(stderr="nope", exit_code=1)

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    r = SbxRunner("a1", mount_path="/tmp/m", sbx_path="/usr/bin/sbx", close_action="remove")

    with pytest.raises(RuntimeError, match="sbx rm failed"):
        asyncio.run(r.close())
    assert r._closed is False


def test_close_stop_failure_keeps_runner_open(monkeypatch):
    async def fake_run_subprocess(command, **kwargs):
        return ExecResult(stderr="nope", exit_code=1)

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    r = SbxRunner("a1", mount_path="/tmp/m", sbx_path="/usr/bin/sbx", close_action="stop")

    with pytest.raises(RuntimeError, match="sbx stop failed"):
        asyncio.run(r.close())
    assert r._closed is False


def test_create_quotes_default_cwd(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    monkeypatch.setattr("cayu.runners.sbx.tempfile.mkdtemp", lambda prefix: "/tmp/cayu-sbx-x")

    runner = asyncio.run(SbxRunner.create("a1", sbx_path="/usr/bin/sbx", default_cwd="/work space"))
    mkdir_cmds = [c for c in issued if len(c) > 5 and c[5] == "sh" and "mkdir" in c[-1]]
    assert len(mkdir_cmds) == 1
    assert "'/work space'" in mkdir_cmds[0][-1]
    assert runner.default_cwd == "/work space"


def test_sbx_runner_exported_from_top_level():
    import cayu
    from cayu import SbxRunner as TopLevel
    from cayu.runners import SbxRunner as RunnersLevel

    assert TopLevel is RunnersLevel
    assert "SbxRunner" in cayu.__all__


def test_exec_injects_declared_secret_env_without_argv_exposure(monkeypatch):
    calls = {}

    async def fake_run_subprocess(command, **kwargs):
        calls["argv"] = command.argv
        calls["kwargs"] = kwargs
        return ExecResult(stdout="token is sk-super-secret-token", stderr="sk-super-secret-token")

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    r = SbxRunner(
        "a1",
        mount_path="/tmp/m",
        sbx_path="/usr/bin/sbx",
        secret_env=[SecretEnv(name="API_TOKEN", ref=SecretRef(name="api_token"))],
        secret_resolver=StaticVault({"api_token": "sk-super-secret-token"}),
    )

    result = asyncio.run(r.exec(ExecCommand.process("env"), env={"PLAIN": "x"}))

    # Container env is passed via --env-file; neither values nor names appear in argv.
    assert "--env-file" in calls["argv"]
    assert not any("sk-super-secret-token" in item for item in calls["argv"])
    assert not any("API_TOKEN" in item for item in calls["argv"])
    assert not any("PLAIN" in item for item in calls["argv"])
    # SECURITY (V4): the sbx CLI's OWN process env is pristine — model-controlled env and
    # secrets are NOT merged into it, so a prompt-injected agent cannot hijack the CLI.
    assert "API_TOKEN" not in calls["kwargs"]["env"]
    assert "PLAIN" not in calls["kwargs"]["env"]
    # Captured output is scrubbed before reaching model-visible context.
    assert result.stdout == f"token is {REDACTED_SECRET}"
    assert result.stderr == REDACTED_SECRET


def test_exec_rejects_env_key_colliding_with_secret_env(monkeypatch):
    async def fake_run_subprocess(command, **kwargs):
        raise AssertionError("run_subprocess should not be called")

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    r = SbxRunner(
        "a1",
        mount_path="/tmp/m",
        sbx_path="/usr/bin/sbx",
        secret_env={"API_TOKEN": SecretRef(name="api_token")},
        secret_resolver=StaticVault({"api_token": "sk-super-secret-token"}),
    )

    with pytest.raises(ValueError, match="collides with declared secret_env"):
        asyncio.run(r.exec(ExecCommand.process("env"), env={"API_TOKEN": "override"}))


def test_runner_secret_env_requires_resolver():
    with pytest.raises(ValueError, match="secret_resolver"):
        SbxRunner(
            "a1",
            mount_path="/tmp/m",
            sbx_path="/usr/bin/sbx",
            secret_env={"API_TOKEN": SecretRef(name="api_token")},
        )


def test_create_prevalidates_secret_env_mode_before_sbx_resources(monkeypatch):
    async def fake_run_subprocess(command, **kwargs):
        raise AssertionError("sbx should not be called after local validation fails")

    def fake_mkdtemp(prefix):
        raise AssertionError("temp mount should not be created after local validation fails")

    monkeypatch.setattr("cayu.runners.sbx.run_subprocess", fake_run_subprocess)
    monkeypatch.setattr("cayu.runners.sbx.tempfile.mkdtemp", fake_mkdtemp)

    with pytest.raises(ValueError, match="virtual_egress"):
        asyncio.run(
            SbxRunner.create(
                "a1",
                sbx_path="/usr/bin/sbx",
                secret_env={"API_TOKEN": SecretRef(name="api_token")},
                secret_resolver=StaticVault({"api_token": "sk-super-secret-token"}),
                credential_mode="virtual_egress",
            )
        )
