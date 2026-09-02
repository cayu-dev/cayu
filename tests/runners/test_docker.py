from __future__ import annotations

import asyncio
import io
import json
import os
from typing import Any

import pytest

from cayu.runners import BROWSER_FETCH_WORKLOAD_NAME, PINNED_BROWSER_FETCH_WORKLOAD
from cayu.runners._docker_cli import docker_cli_env
from cayu.runners.base import ExecCommand, ExecResult
from cayu.runners.docker import (
    _PYTHON_PROCESS_SUPERVISOR,
    DEFAULT_DOCKER_CWD,
    DOCKER_COMMAND_STATE_DIR,
    DockerRunner,
    _build_docker_exec_argv,
    _kill_supervised_command_script,
    _require_docker,
    _validate_mount_path,
)
from cayu.testing import verify_provider_credential_isolation
from cayu.vaults import REDACTED_SECRET, SecretEnv, SecretRedactor, SecretRef, StaticVault


def test_require_docker_uses_explicit_path():
    assert _require_docker("/usr/bin/docker") == "/usr/bin/docker"


def test_require_docker_missing_raises(monkeypatch):
    monkeypatch.setattr("cayu.runners.docker.shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="docker CLI not found"):
        _require_docker(None)


def test_docker_runner_declares_browser_workload_only_for_exact_image() -> None:
    pinned = DockerRunner(
        "browser",
        image=PINNED_BROWSER_FETCH_WORKLOAD.image,
        docker_path="/usr/bin/docker",
    )
    other = DockerRunner(
        "other",
        image="browser:latest",
        docker_path="/usr/bin/docker",
    )
    credentialed = DockerRunner(
        "credentialed-browser",
        image=PINNED_BROWSER_FETCH_WORKLOAD.image,
        docker_path="/usr/bin/docker",
        secret_env={"AUTH_TOKEN": SecretRef(name="browser_auth")},
        secret_resolver=StaticVault({"browser_auth": "browser-secret-canary"}),
    )
    overlaid = DockerRunner(
        "overlaid-browser",
        image=PINNED_BROWSER_FETCH_WORKLOAD.image,
        docker_path="/usr/bin/docker",
        env_overlay={"WORKLOAD_TOKEN": "overlay-secret-canary"},
    )

    assert pinned.workload_authority(BROWSER_FETCH_WORKLOAD_NAME) == (PINNED_BROWSER_FETCH_WORKLOAD)
    assert other.workload_authority(BROWSER_FETCH_WORKLOAD_NAME) is None
    assert pinned.output_secret_values_present() is False
    assert credentialed.output_secret_values_present() is True
    assert overlaid.output_secret_values_present() is True


def test_docker_egress_cutover_fence_terminates_detached_guest_work(monkeypatch) -> None:
    issued: list[list[str]] = []

    async def fake_run_subprocess(command, **kwargs):
        del kwargs
        issued.append(command.argv)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    runner = DockerRunner("agent", docker_path="/usr/bin/docker")

    asyncio.run(runner.fence_guest_processes_for_egress_cutover())

    assert issued[0][:7] == [
        "/usr/bin/docker",
        "exec",
        "-u",
        "root",
        "agent",
        "python3",
        "-c",
    ]
    assert 'os.listdir("/proc")' in issued[0][7]


def test_docker_egress_cutover_fence_fails_closed(monkeypatch) -> None:
    async def fake_run_subprocess(command, **kwargs):
        del command, kwargs
        return ExecResult(exit_code=70)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    runner = DockerRunner("agent", docker_path="/usr/bin/docker")

    with pytest.raises(RuntimeError, match="could not be fenced"):
        asyncio.run(runner.fence_guest_processes_for_egress_cutover())


def test_docker_cli_helper_environment_is_operationally_allowlisted(monkeypatch):
    provider_canaries = {
        "OPENAI_API_KEY": "provider-openai-canary-0123456789",
        "ANTHROPIC_API_KEY": "provider-anthropic-canary-0123456789",
        "GEMINI_API_KEY": "provider-gemini-canary-0123456789",
        "CAYU_HOME": "/tmp/provider-auth-store-canary-0123456789",
        "OPENAI_AUTHORIZATION": "Bearer provider-header-canary-0123456789",
    }
    for name, value in provider_canaries.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/docker.sock")
    monkeypatch.setenv("DOCKER_CONFIG", "/tmp/docker-config")
    monkeypatch.setenv("SSL_CERT_FILE", "/tmp/ca.pem")
    monkeypatch.setenv("HTTPS_PROXY", "http://trusted-proxy.test:8443")

    helper_env = docker_cli_env()

    assert helper_env["PATH"] == "/usr/local/bin:/usr/bin"
    assert helper_env["DOCKER_HOST"] == "unix:///tmp/docker.sock"
    assert helper_env["DOCKER_CONFIG"] == "/tmp/docker-config"
    assert helper_env["SSL_CERT_FILE"] == "/tmp/ca.pem"
    assert helper_env["HTTPS_PROXY"] == "http://trusted-proxy.test:8443"
    assert set(provider_canaries).isdisjoint(helper_env)
    assert all(value not in repr(helper_env) for value in provider_canaries.values())


def test_docker_cli_helper_environment_accepts_explicit_trusted_grants(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "private-registry")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/run/registry/google.json")

    helper_env = docker_cli_env(("AWS_PROFILE", "GOOGLE_APPLICATION_CREDENTIALS"))

    assert helper_env["AWS_PROFILE"] == "private-registry"
    assert helper_env["GOOGLE_APPLICATION_CREDENTIALS"] == "/run/registry/google.json"


def test_docker_runner_forwards_only_explicit_docker_cli_grants(monkeypatch):
    calls: list[dict[str, Any]] = []

    async def fake_run_subprocess(command, **kwargs):
        calls.append({"argv": command.argv, **kwargs})
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    monkeypatch.setenv("AWS_PROFILE", "private-registry")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/run/registry/google.json")
    runner = DockerRunner(
        "credential-helper",
        docker_path="/usr/bin/docker",
        docker_cli_env_allowlist=("AWS_PROFILE",),
    )

    asyncio.run(runner.exec(ExecCommand.process("true")))

    assert calls[0]["env"]["AWS_PROFILE"] == "private-registry"
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in calls[0]["env"]


def test_docker_create_forwards_explicit_cli_grants_through_lifecycle(monkeypatch):
    calls: list[dict[str, Any]] = []

    async def fake_run_subprocess(command, **kwargs):
        calls.append({"argv": command.argv, **kwargs})
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    monkeypatch.setenv("AWS_PROFILE", "private-registry")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/run/registry/google.json")

    async def run() -> DockerRunner:
        runner = await DockerRunner.create(
            "credential-helper",
            docker_path="/usr/bin/docker",
            setup_commands=("true",),
            docker_cli_env_allowlist=("AWS_PROFILE",),
        )
        await runner.close()
        return runner

    runner = asyncio.run(run())

    assert runner.docker_cli_env_allowlist == ("AWS_PROFILE",)
    assert any(call["argv"][1] == "run" for call in calls)
    assert any(call["argv"][1:3] == ["exec", "-u"] for call in calls)
    assert calls[-1]["argv"][1:3] == ["rm", "-f"]
    assert all(call["env"]["AWS_PROFILE"] == "private-registry" for call in calls)
    assert all("GOOGLE_APPLICATION_CREDENTIALS" not in call["env"] for call in calls)


def test_docker_runner_passes_provider_credential_isolation_probe(
    monkeypatch,
    provider_credential_canaries,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_run_subprocess(command, **kwargs):
        calls.append({"argv": command.argv, **kwargs})
        env_file = command.argv[command.argv.index("--env-file") + 1]
        environment = {}
        with open(env_file, encoding="utf-8") as handle:
            for line in handle:
                name, value = line.rstrip("\n").split("=", 1)
                environment[name] = value
        calls[-1]["guest_env"] = environment
        return ExecResult(
            stdout=json.dumps(
                {
                    "environment": environment,
                    "auth_paths": {},
                    "auth_scan_complete": True,
                    "provider_canary_matches": [],
                    "detector_control_match": True,
                },
                sort_keys=True,
            )
        )

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    workload_secret = "declared-workload-secret-0123456789"
    runner = DockerRunner(
        "credential-probe",
        docker_path="/usr/bin/docker",
        secret_env={"DECLARED_WORKLOAD_SECRET": SecretRef(name="workload_secret")},
        secret_resolver=StaticVault({"workload_secret": workload_secret}),
    )

    evidence = asyncio.run(
        verify_provider_credential_isolation(
            runner,
            adapter="docker",
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
    )

    assert evidence.status == "verified"
    # RAW_ENV remains an explicit workload grant: the guest sees that declared
    # value while none of the model-provider authority crosses the boundary.
    assert calls[0]["guest_env"]["DECLARED_WORKLOAD_SECRET"] == workload_secret
    assert "os.walk(root" in repr(calls[0]["argv"])
    assert set(calls[0]["env"]).isdisjoint(provider_credential_canaries.host_env)
    assert all(value not in repr(calls) for value in provider_credential_canaries.values.values())


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


@pytest.mark.parametrize("invalid_text", ("/workspace\x00bad", "/workspace\ud800bad"))
def test_runner_rejects_nonportable_default_cwd(invalid_text: str) -> None:
    with pytest.raises(ValueError, match="default_cwd"):
        DockerRunner("a", docker_path="/usr/bin/docker", default_cwd=invalid_text)


@pytest.mark.parametrize("invalid_text", ("/workspace\x00bad", "/workspace\ud800bad"))
def test_runner_revalidates_mutated_default_cwd_before_secret_resolution(
    invalid_text: str,
) -> None:
    class _CountingVault(StaticVault):
        def __init__(self) -> None:
            super().__init__({"token": "secret-value"})
            self.resolve_calls = 0

        async def resolve(self, ref, *, scope=None):  # type: ignore[no-untyped-def]
            self.resolve_calls += 1
            return await super().resolve(ref, scope=scope)

    vault = _CountingVault()
    runner = DockerRunner(
        "a",
        docker_path="/usr/bin/docker",
        secret_env={"TOKEN": SecretRef(name="token")},
        secret_resolver=vault,
    )
    runner.default_cwd = invalid_text

    async def run() -> None:
        with pytest.raises(ValueError, match="default_cwd"):
            await runner.exec(ExecCommand.process("true"))

    asyncio.run(run())

    assert vault.resolve_calls == 0


def test_runner_normalizes_default_cwd():
    r = DockerRunner("a", docker_path="/usr/bin/docker", default_cwd="/workspace/../work")
    assert r.default_cwd == "/work"


def test_build_exec_argv_process():
    argv = _build_docker_exec_argv(
        "/usr/bin/docker",
        "a1",
        ExecCommand.process("whois", "x.ai"),
        cwd="/workspace",
        env_file=None,
        has_stdin=False,
        pid_file="/tmp/cayu-docker-commands/cmd.pid",
    )
    assert argv[:6] == ["/usr/bin/docker", "exec", "-w", "/workspace", "a1", "sh"]
    assert argv[6] == "-c"
    assert "setsid -w true" in argv[7]
    assert "exec setsid -w sh -c" in argv[7]
    assert "else exec sh -c" in argv[7]
    assert "/tmp/cayu-docker-commands/cmd.pid" in argv[7]
    assert "whois x.ai" in argv[7]
    assert " & " not in argv[7]
    assert "> /tmp/cayu-docker-commands/cmd.pid || exit 1" in argv[7]
    assert argv[7].count('"$$" "1"') == 1
    assert 'test "$observed_group" = "$$"' in argv[7]
    assert '"$$" "$process_group"' in argv[7]


def test_build_exec_argv_strict_process_uses_fixed_python_supervisor() -> None:
    argv = _build_docker_exec_argv(
        "/usr/bin/docker",
        "a1",
        ExecCommand.process("/opt/tools/pytest", "-q", "tests/test_unit.py;echo"),
        cwd="/workspace",
        env_file=None,
        has_stdin=False,
        pid_file="/tmp/cayu-docker-commands/cmd.pid",
        direct_process_supervisor=True,
    )

    assert argv[:6] == [
        "/usr/bin/docker",
        "exec",
        "-w",
        "/workspace",
        "a1",
        "python3",
    ]
    assert argv[6] == "-c"
    assert "os.execvp(command[0], command)" in argv[7]
    assert "PR_SET_CHILD_SUBREAPER = 36" in argv[7]
    assert "prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)" in argv[7]
    assert "os.waitpid(-1, 0)" in argv[7]
    assert 'f"{os.getpid()} 2' in argv[7]
    assert argv[-3:] == ["/opt/tools/pytest", "-q", "tests/test_unit.py;echo"]
    assert "sh" not in argv[:6]


def test_python_supervisor_cleanup_requires_descendant_settlement() -> None:
    script = _kill_supervised_command_script("/tmp/cayu-docker-commands/cmd.pid")

    assert 'process_group" = 2' in script
    assert "while test -f" in script
    assert "kill -KILL" in script
    assert "exit 1" in script
    assert "PR_SET_CHILD_SUBREAPER" in _PYTHON_PROCESS_SUPERVISOR


def test_build_exec_argv_shell_env_stdin():
    argv = _build_docker_exec_argv(
        "/usr/bin/docker",
        "a1",
        ExecCommand.bash("echo hi"),
        cwd="/workspace",
        env_file="/tmp/cayu-runner-env-abc",
        has_stdin=True,
        pid_file="/tmp/cayu-docker-commands/cmd.pid",
    )
    # Env is passed via --env-file; values never appear in argv.
    assert argv[:9] == [
        "/usr/bin/docker",
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
    assert "setsid -w true" in argv[10]
    assert "exec setsid -w sh -c" in argv[10]
    assert "else exec sh -c" in argv[10]
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


def test_exec_stream_forwards_binary_streams_without_host_paths(monkeypatch) -> None:
    calls: dict[str, Any] = {}
    source = io.BytesIO(b"tar-input")
    target = io.BytesIO()

    async def fake_run_subprocess(command, **kwargs):
        calls["argv"] = command.argv
        calls["kwargs"] = kwargs
        kwargs["stdout_stream"].write(b"tar-output")
        return ExecResult(stdout_bytes=10)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    runner = DockerRunner("a1", docker_path="/usr/bin/docker")

    result = asyncio.run(
        runner.exec_stream(
            ExecCommand.process("python3", "-c", "pass"),
            stdin=source,
            stdout=target,
            stdout_limit_bytes=1024,
        )
    )

    assert result.stdout_bytes == 10
    assert target.getvalue() == b"tar-output"
    assert calls["argv"][2] == "-i"
    assert calls["kwargs"]["stdin_stream"] is source
    assert calls["kwargs"]["stdout_stream"] is target
    assert calls["kwargs"]["stdout_limit_bytes"] == 1024


def test_exec_stream_failure_settles_remote_command_and_preserves_primary_error(
    monkeypatch,
) -> None:
    issued: list[list[str]] = []
    stream_error = OSError("test binary sink failed")

    async def fake_run_subprocess(command, **kwargs):
        del kwargs
        issued.append(command.argv)
        if "kill -TERM" in command.argv[-1]:
            return ExecResult()
        raise stream_error

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    runner = DockerRunner("a1", docker_path="/usr/bin/docker")

    with pytest.raises(OSError) as exc_info:
        asyncio.run(
            runner.exec_stream(
                ExecCommand.process("sh", "-c", "printf output; sleep 999"),
                stdout=io.BytesIO(),
            )
        )

    assert exc_info.value is stream_error
    assert runner._exec_closed is False
    assert len(issued) == 2
    assert "setsid" in issued[0][-1]
    assert "kill -TERM" in issued[1][-1]
    assert getattr(stream_error, "artifacts", None) == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "docker",
            "action": "kill_command",
            "status": "completed",
            "timeout_s": 5.0,
        }
    ]


def test_exec_stream_failure_latches_runner_when_remote_settlement_is_unproven(
    monkeypatch,
) -> None:
    issued: list[list[str]] = []
    stream_error = OSError("test binary source failed")

    async def fake_run_subprocess(command, **kwargs):
        del kwargs
        issued.append(command.argv)
        if "kill -TERM" in command.argv[-1]:
            return ExecResult(exit_code=1)
        if command.argv[-1].startswith("test -f"):
            return ExecResult(exit_code=0)
        raise stream_error

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    runner = DockerRunner("a1", docker_path="/usr/bin/docker")

    with pytest.raises(OSError) as exc_info:
        asyncio.run(
            runner.exec_stream(
                ExecCommand.process("sh", "-c", "printf output; sleep 999"),
                stdout=io.BytesIO(),
            )
        )

    assert exc_info.value is stream_error
    assert runner._exec_closed is True
    assert runner._exec_closed_reason == (
        "docker command cleanup did not complete; command state is unknown"
    )
    assert len(issued) == 4
    assert getattr(stream_error, "artifacts", None) == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "docker",
            "action": "kill_command",
            "status": "failed",
            "timeout_s": 5.0,
            "error": "kill returned false",
        }
    ]
    with pytest.raises(RuntimeError, match="DockerRunner is closed"):
        asyncio.run(runner.exec(ExecCommand.process("true")))


def test_exec_redacted_forwards_invocation_redactor_to_subprocess_capture(monkeypatch) -> None:
    secret = "docker-exec-boundary-secret"
    calls: dict[str, Any] = {}

    async def fake_run_subprocess(command, **kwargs):
        calls["redactor"] = kwargs["output_redactor"]
        return ExecResult(stdout=kwargs["output_redactor"].redact_text(secret))

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    runner = DockerRunner("a1", docker_path="/usr/bin/docker")

    result = asyncio.run(
        runner.exec_redacted(
            ExecCommand.process("echo", "ignored"),
            redactor=SecretRedactor(secret),
        )
    )

    assert isinstance(calls["redactor"], SecretRedactor)
    assert result.stdout == REDACTED_SECRET


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


@pytest.mark.parametrize("close_action", ["remove", "stop"])
def test_docker_lifecycle_redacts_complete_stderr_before_bounding(
    monkeypatch,
    close_action,
) -> None:
    secret = "docker-lifecycle-boundary-secret"
    stderr = "x" * 290 + secret + "tail"

    async def fake_run_subprocess(command, **kwargs):
        return ExecResult(stderr=stderr, exit_code=1)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    runner = DockerRunner(
        "a1",
        docker_path="/usr/bin/docker",
        close_action=close_action,
        env_overlay={"WORKLOAD_TOKEN": secret},
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(runner.close())

    message = str(exc_info.value)
    assert secret not in message
    assert secret[:12] not in message
    assert "[truncated]" in message
    assert len(message.encode()) < 400


@pytest.mark.parametrize("close_action", ["remove", "stop"])
def test_docker_lifecycle_refreshes_rotated_allowlisted_credentials(
    monkeypatch,
    close_action,
) -> None:
    env_name = "CAYU_TEST_ROTATED_DOCKER_CREDENTIAL"
    old_secret = "old-docker-lifecycle-secret"
    current_secret = "current-docker-lifecycle-secret"
    monkeypatch.setenv(env_name, old_secret)
    runner = DockerRunner(
        "a1",
        docker_path="/usr/bin/docker",
        close_action=close_action,
        docker_cli_env_allowlist=(env_name,),
    )
    monkeypatch.setenv(env_name, current_secret)

    async def fake_run_subprocess(command, **kwargs):
        return ExecResult(stderr=f"failure:{current_secret}", exit_code=1)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(runner.close())

    message = str(exc_info.value)
    assert current_secret not in message
    assert current_secret[:12] not in message
    assert REDACTED_SECRET in message


def test_close_stop_failure_keeps_runner_open(monkeypatch):
    async def fake_run_subprocess(command, **kwargs):
        return ExecResult(stderr="nope", exit_code=1)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    r = DockerRunner("a1", docker_path="/usr/bin/docker", close_action="stop")

    with pytest.raises(RuntimeError, match="docker stop failed"):
        asyncio.run(r.close())
    assert r._closed is False


def test_create_bind_mount_mode(monkeypatch, tmp_path):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    host_ws = str(tmp_path)
    runner = asyncio.run(
        DockerRunner.create(
            "a1",
            docker_path="/usr/bin/docker",
            image="debian:stable-slim",
            mount_path=host_ws,
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
        "--init",
        "--name",
        "a1",
        "--mount",
        f"type=bind,source={host_ws},target={host_ws}",
        "debian:stable-slim",
        "sleep",
        "infinity",
    ] in issued
    # bind mode: default_cwd defaults to the mount, and NO mkdir exec is issued
    assert runner.default_cwd == host_ws
    assert not any(
        a[:4] == ["/usr/bin/docker", "exec", "-u", "root"] and f"mkdir -p {host_ws}" in a
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


def test_create_setup_env_overlay_uses_env_file_not_argv(monkeypatch):
    issued = []
    env_file_contents: list[str] = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        if "--env-file" in command.argv:
            env_file = command.argv[command.argv.index("--env-file") + 1]
            assert os.path.exists(env_file)
            with open(env_file) as handle:
                env_file_contents.append(handle.read())
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    asyncio.run(
        DockerRunner.create(
            "a1",
            docker_path="/usr/bin/docker",
            image="debian:stable-slim",
            setup_commands=("python -V",),
            env_overlay={
                "STRIPE_SECRET_KEY": "sk_test_cayu_vc_setupsecret",
                "HTTPS_PROXY": "http://cayu-egress:8080",
            },
        )
    )

    setup_exec = next(
        argv
        for argv in issued
        if argv[:4] == ["/usr/bin/docker", "exec", "-u", "root"] and argv[-1] == "python -V"
    )
    assert "--env-file" in setup_exec
    assert not any("sk_test_cayu_vc_setupsecret" in item for item in setup_exec)
    assert not any("STRIPE_SECRET_KEY=" in item for item in setup_exec)
    assert any(
        "STRIPE_SECRET_KEY=sk_test_cayu_vc_setupsecret\n" in data for data in env_file_contents
    )


def test_validate_mount_path_normalizes_existing_dir(tmp_path):
    messy = f"{tmp_path}/sub/.."
    assert _validate_mount_path(messy) == str(tmp_path)


def test_validate_mount_path_rejects_relative():
    with pytest.raises(ValueError, match="absolute host path"):
        _validate_mount_path("relative/ws")


def test_validate_mount_path_rejects_comma(tmp_path):
    with pytest.raises(ValueError, match="must not contain commas"):
        _validate_mount_path(f"{tmp_path},readonly")


def test_validate_mount_path_rejects_missing(tmp_path):
    with pytest.raises(ValueError, match="existing directory"):
        _validate_mount_path(str(tmp_path / "does-not-exist"))


def test_validate_mount_path_rejects_file(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("x")
    with pytest.raises(ValueError, match="existing directory"):
        _validate_mount_path(str(target))


def test_create_applies_explicit_seccomp_profile(monkeypatch, tmp_path):
    issued = []
    profile = tmp_path / "chromium-seccomp.json"
    profile.write_text("{}")

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    asyncio.run(
        DockerRunner.create(
            "a1",
            docker_path="/usr/bin/docker",
            seccomp_profile=str(profile),
        )
    )

    run = next(argv for argv in issued if argv[1] == "run")
    assert run[2:5] == ["-d", "--init", "--security-opt"]
    assert run[5] == f"seccomp={profile}"


@pytest.mark.parametrize("profile", ["relative.json", "/missing/seccomp.json"])
def test_create_rejects_invalid_seccomp_profile(monkeypatch, profile):
    async def fake_run_subprocess(command, **kwargs):
        raise AssertionError("docker should not run for an invalid seccomp profile")

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    with pytest.raises(ValueError, match="seccomp_profile"):
        asyncio.run(
            DockerRunner.create(
                "a1",
                docker_path="/usr/bin/docker",
                seccomp_profile=profile,
            )
        )


def test_create_rejects_bad_mount_path(monkeypatch, tmp_path):
    async def fake_run_subprocess(command, **kwargs):
        raise AssertionError("docker should not be invoked when mount_path is invalid")

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    with pytest.raises(ValueError, match="must not contain commas"):
        asyncio.run(
            DockerRunner.create(
                "a1",
                docker_path="/usr/bin/docker",
                mount_path=f"{tmp_path},z",
            )
        )


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
        "--init",
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


@pytest.mark.parametrize("failure_stage", ["run", "mkdir", "setup"])
def test_docker_create_lifecycle_redacts_before_bounding(
    monkeypatch,
    failure_stage: str,
) -> None:
    secret = "docker-create-lifecycle-boundary-secret"
    stderr = "x" * 290 + secret + "tail"

    async def fake_run_subprocess(command, **kwargs):
        argv = command.argv
        stage = None
        if argv[1] == "run":
            stage = "run"
        elif argv[-1] == "mkdir -p /workspace":
            stage = "mkdir"
        elif argv[-1] == "setup-command":
            stage = "setup"
        if stage == failure_stage:
            return ExecResult(exit_code=1, stderr=stderr)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(
            DockerRunner.create(
                "a1",
                docker_path="/usr/bin/docker",
                setup_commands=("setup-command",),
                env_overlay={"WORKLOAD_TOKEN": secret},
            )
        )

    message = str(exc_info.value)
    assert secret not in message
    assert secret[:12] not in message
    assert "[truncated]" in message


def test_resolve_cwd_relative_path():
    r = DockerRunner("a1", docker_path="/usr/bin/docker")
    assert r.resolve_cwd("sub/dir") == "/workspace/sub/dir"


def test_resolve_cwd_accepts_contained_absolute_and_rejects_outside():
    r = DockerRunner("a1", docker_path="/usr/bin/docker")
    assert r.resolve_cwd("/workspace/sub/../tests") == "/workspace/tests"
    with pytest.raises(ValueError, match="outside the runner root"):
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


def test_exec_cancellation_reraises_plain_cancelled_error_with_artifacts(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        if "kill -TERM" in command.argv[-1]:
            return ExecResult()
        raise asyncio.CancelledError

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    r = DockerRunner("a1", docker_path="/usr/bin/docker")

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


def test_command_kill_retries_before_reporting_failure(monkeypatch):
    kill_attempts = []

    async def fake_run_subprocess(command, **kwargs):
        if "kill -TERM" in command.argv[-1]:
            kill_attempts.append(command.argv)
            # First attempt flakes (pid file not visible yet), second succeeds.
            if len(kill_attempts) == 1:
                return ExecResult(exit_code=1)
            return ExecResult()
        return ExecResult(timed_out=True, exit_code=-9)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    r = DockerRunner("a1", docker_path="/usr/bin/docker")

    result = asyncio.run(r.exec(ExecCommand.process("sleep", "999"), timeout_s=1))

    assert result.timed_out is True
    assert len(kill_attempts) == 2
    assert r._exec_closed is False
    assert result.artifacts[0]["status"] == "completed"


def test_command_kill_verifies_missing_pid_file_as_stopped(monkeypatch):
    issued = []

    async def fake_run_subprocess(command, **kwargs):
        issued.append(command.argv)
        if "kill -TERM" in command.argv[-1]:
            return ExecResult(exit_code=1)
        if command.argv[-1].startswith("test -f"):
            # pid file absent: the supervised command is not running.
            return ExecResult(exit_code=1)
        return ExecResult(timed_out=True, exit_code=-9)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    r = DockerRunner("a1", docker_path="/usr/bin/docker")

    result = asyncio.run(r.exec(ExecCommand.process("sleep", "999"), timeout_s=1))

    assert result.timed_out is True
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

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    r = DockerRunner("a1", docker_path="/usr/bin/docker")

    result = asyncio.run(r.exec(ExecCommand.process("sleep", "999"), timeout_s=1))
    assert result.timed_out is True
    assert r._exec_closed is True

    with pytest.raises(RuntimeError, match="DockerRunner is closed: docker command cleanup"):
        asyncio.run(r.exec(ExecCommand.process("true")))

    r.reopen_exec()

    after = asyncio.run(r.exec(ExecCommand.process("true"), timeout_s=None))
    assert after.timed_out is True  # fake still reports timeouts; exec path is open again


def test_reopen_exec_rejects_closed_runner(monkeypatch):
    async def fake_run_subprocess(command, **kwargs):
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    r = DockerRunner("a1", docker_path="/usr/bin/docker", close_action="none")
    asyncio.run(r.close())

    with pytest.raises(RuntimeError, match="DockerRunner is closed."):
        r.reopen_exec()


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


def test_exec_injects_declared_secret_env_without_argv_exposure(monkeypatch):
    calls = {}

    async def fake_run_subprocess(command, **kwargs):
        calls["argv"] = command.argv
        calls["kwargs"] = kwargs
        return ExecResult(stdout="token is sk-super-secret-token", stderr="sk-super-secret-token")

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    vault = StaticVault({"api_token": "sk-super-secret-token"})
    r = DockerRunner(
        "a1",
        docker_path="/usr/bin/docker",
        secret_env=[SecretEnv(name="API_TOKEN", ref=SecretRef(name="api_token"))],
        secret_resolver=vault,
    )

    result = asyncio.run(r.exec(ExecCommand.process("env"), env={"PLAIN": "x"}))

    # Container env is passed via --env-file; neither values nor names appear in argv.
    assert "--env-file" in calls["argv"]
    assert not any("sk-super-secret-token" in item for item in calls["argv"])
    assert not any("API_TOKEN" in item for item in calls["argv"])
    assert not any("PLAIN" in item for item in calls["argv"])
    # SECURITY (V4): the docker CLI's OWN process env is pristine — model-controlled env
    # (PLAIN) and secrets (API_TOKEN) are NOT merged into it, so a prompt-injected agent
    # cannot hijack the host CLI (e.g. by setting DOCKER_HOST to an attacker daemon).
    assert "API_TOKEN" not in calls["kwargs"]["env"]
    assert "PLAIN" not in calls["kwargs"]["env"]
    # Captured output is scrubbed before reaching model-visible context.
    assert result.stdout == f"token is {REDACTED_SECRET}"
    assert result.stderr == REDACTED_SECRET


def test_exec_rejects_env_key_colliding_with_secret_env(monkeypatch):
    async def fake_run_subprocess(command, **kwargs):
        raise AssertionError("run_subprocess should not be called")

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    r = DockerRunner(
        "a1",
        docker_path="/usr/bin/docker",
        secret_env={"API_TOKEN": SecretRef(name="api_token")},
        secret_resolver=StaticVault({"api_token": "sk-super-secret-token"}),
    )

    with pytest.raises(ValueError, match="collides with declared secret_env"):
        asyncio.run(r.exec(ExecCommand.process("env"), env={"API_TOKEN": "override"}))


def test_runner_secret_env_requires_resolver():
    with pytest.raises(ValueError, match="secret_resolver"):
        DockerRunner(
            "a1",
            docker_path="/usr/bin/docker",
            secret_env={"API_TOKEN": SecretRef(name="api_token")},
        )


def test_create_prevalidates_secret_env_mode_before_docker_calls(monkeypatch):
    async def fake_run_subprocess(command, **kwargs):
        raise AssertionError("docker should not be called after local validation fails")

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    with pytest.raises(ValueError, match="virtual_egress"):
        asyncio.run(
            DockerRunner.create(
                "a1",
                docker_path="/usr/bin/docker",
                secret_env={"API_TOKEN": SecretRef(name="api_token")},
                secret_resolver=StaticVault({"api_token": "sk-super-secret-token"}),
                credential_mode="virtual_egress",
            )
        )


def test_create_owns_validated_secret_env_across_docker_awaits(monkeypatch):
    first_call_started = asyncio.Event()
    release_first_call = asyncio.Event()
    calls = 0

    async def fake_run_subprocess(command, **kwargs):
        nonlocal calls
        del command, kwargs
        calls += 1
        if calls == 1:
            first_call_started.set()
            await release_first_call.wait()
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    secret_env = {"API_TOKEN": SecretRef(name="api_token")}
    resolver = StaticVault({"api_token": "sk-super-secret-token"})

    async def run() -> DockerRunner:
        create_task = asyncio.create_task(
            DockerRunner.create(
                "a1",
                docker_path="/usr/bin/docker",
                replace=False,
                close_action="none",
                secret_env=secret_env,
                secret_resolver=resolver,
            )
        )
        await first_call_started.wait()
        secret_env.clear()
        secret_env["INVALID\nNAME"] = SecretRef(name="api_token")
        release_first_call.set()
        return await create_task

    runner = asyncio.run(run())

    assert tuple(runner.secret_env) == ("API_TOKEN",)
    assert runner.secret_resolver is resolver
    assert calls == 2


def test_exec_owns_command_before_resolving_secrets(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    dispatched_argv: list[list[str] | None] = []

    class BlockingVault(StaticVault):
        async def resolve(self, ref, *, scope=None):
            started.set()
            await release.wait()
            return await super().resolve(ref, scope=scope)

    async def fake_run_subprocess(command, **kwargs):
        del kwargs
        dispatched_argv.append(command.argv)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    runner = DockerRunner(
        "validation-probe",
        docker_path="/unreachable/docker",
        secret_env={"API_TOKEN": SecretRef(name="api_token")},
        secret_resolver=BlockingVault({"api_token": "secret-value"}),
    )
    command = ExecCommand.process("original-command", "original-argument")

    async def run() -> None:
        task = asyncio.create_task(runner.exec(command))
        await started.wait()
        assert command.argv is not None
        command.argv[:] = ["mutated-command"]
        release.set()
        await task

    asyncio.run(run())

    rendered = repr(dispatched_argv)
    assert "original-command" in rendered
    assert "original-argument" in rendered
    assert "mutated-command" not in rendered
