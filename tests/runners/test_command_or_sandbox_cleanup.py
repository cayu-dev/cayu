from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from cayu.environments.docker_toolchains import (
    DockerCodingCommandAuthority,
    DockerCodingToolchainProfile,
)
from cayu.runners._cleanup import (
    cleanup_runner_command_with_diagnostic,
    runner_cleanup_observation_timeout,
    validate_runner_cleanup_policy,
)
from cayu.runners._docker_command_receipt import guest_program
from cayu.runners.base import ExecCommand, ExecResult, runner_workspace_mutation_settlement
from cayu.runners.docker import (
    _COMMAND_TREE_OBSERVER,
    _COMMAND_TREE_OBSERVER_LAUNCHER,
    DockerRunner,
    _DockerRuntimeEvidence,
    _kill_supervised_command_script,
)
from cayu.runners.docker_workload import (
    DockerImageIdentity,
    DockerTmpfsMount,
    DockerWorkloadRestrictions,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._runner import invocation_runner_handle
from cayu.tools.base import ToolContext, _bind_runtime_tool_invocation_authority
from cayu.tools.command_policy import ProcessCommandPolicy
from cayu.tools.commands import ExecCommandTool
from cayu.tools.named_checks import NamedCheck, RunCheckTool
from cayu.tools.structured_commands import RunCommandTool
from cayu.vaults import SecretRedactor
from cayu.workspaces import LocalWorkspace


def _artifact(action: str, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "cayu.runner_cleanup.v1",
        "adapter": "test",
        "action": action,
        "status": status,
        "timeout_s": 5.0,
        **extra,
    }


class _Handle:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls = 0

    async def kill(self) -> bool:
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome  # type: ignore[return-value]


class _Sandbox:
    def __init__(self) -> None:
        self.calls = 0

    async def kill(self) -> bool:
        self.calls += 1
        return True


def _cleanup(handle: _Handle | None, sandbox: _Sandbox):
    return asyncio.run(
        cleanup_runner_command_with_diagnostic(
            sandbox,
            handle=handle,
            adapter="test",
            timeout_s=5.0,
            policy="command_or_sandbox",
        )
    )


def test_confirmed_command_kill_keeps_the_sandbox() -> None:
    handle, sandbox = _Handle(True), _Sandbox()

    result = _cleanup(handle, sandbox)

    assert result.artifacts == [_artifact("kill_command", "completed")]
    assert result.close_runner is False
    assert (handle.calls, sandbox.calls) == (1, 0)


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (False, _artifact("kill_command", "failed", error="kill returned false")),
        (RuntimeError("docker stream broke"), _artifact("kill_command", "failed")),
        (None, _artifact("kill_command", "unsupported", error="command handle is not available")),
    ],
)
def test_unconfirmed_command_kill_falls_back_to_the_sandbox(
    outcome: object, expected: dict[str, Any]
) -> None:
    handle = None if outcome is None else _Handle(outcome)
    sandbox = _Sandbox()

    result = _cleanup(handle, sandbox)

    command_artifact = dict(result.artifacts[0])
    command_artifact.pop("error_type", None)
    assert command_artifact == expected
    assert result.artifacts[1] == _artifact("kill_sandbox", "completed")
    assert result.artifact == _artifact("kill_sandbox", "completed")
    assert result.close_runner is True
    assert sandbox.calls == 1


def test_sandbox_fallback_policy_is_opt_in_per_runner() -> None:
    with pytest.raises(ValueError, match="command, sandbox, none"):
        validate_runner_cleanup_policy("command_or_sandbox", "timeout_cleanup")
    assert (
        validate_runner_cleanup_policy(
            "command_or_sandbox", "timeout_cleanup", allow_sandbox_fallback=True
        )
        == "command_or_sandbox"
    )
    assert runner_cleanup_observation_timeout("command_or_sandbox", 5.0) == 10.0
    assert runner_cleanup_observation_timeout("sandbox", 5.0) == 5.0


def test_completed_sandbox_fallback_is_quiescent_after_unconfirmed_command_kill() -> None:
    result = ExecResult(
        timed_out=True,
        exit_code=-9,
        artifacts=[_artifact("kill_command", "failed"), _artifact("kill_sandbox", "completed")],
    )

    assert runner_workspace_mutation_settlement(result=result, error=None) == "runner_quiescent"
    failed_fallback = result.model_copy(
        update={
            "artifacts": [
                _artifact("kill_command", "failed"),
                _artifact("kill_sandbox", "timeout"),
            ]
        }
    )
    assert runner_workspace_mutation_settlement(result=failed_fallback, error=None) == "uncertain"


def _docker_runner(**kwargs: Any) -> DockerRunner:
    return DockerRunner(
        "a1",
        docker_path="/usr/bin/docker",
        close_action="none",
        timeout_cleanup="command_or_sandbox",
        cancellation_cleanup="command_or_sandbox",
        **kwargs,
    )


_CONTAINER = "c" * 64


def _evidence_runner(
    restrictions: DockerWorkloadRestrictions | None = None,
    required_executables: tuple[str, ...] = ("python3",),
    docker_cli_env_allowlist: tuple[str, ...] = (),
) -> DockerRunner:
    now = datetime.now(UTC)
    evidence = _DockerRuntimeEvidence(
        container_id=_CONTAINER,
        image_id="sha256:" + "a" * 64,
        image_reference="test@sha256:" + "a" * 64,
        network_mode="none",
        default_cwd="/workspace",
        runtime=None,
        seccomp_profile_sha256=None,
        restrictions=restrictions or DockerWorkloadRestrictions(),
        image_identity=DockerImageIdentity(reference="test@sha256:" + "a" * 64),
        toolchain_profile_fingerprint=None,
        required_executables=required_executables,
        executable_availability=tuple((name, True) for name in required_executables),
        immutable_input_mounts=(),
        observed_at=now,
        valid_until=now + timedelta(hours=1),
    )
    return DockerRunner(
        _CONTAINER,
        _container_id=_CONTAINER,
        docker_path="/usr/bin/docker",
        default_cwd="/workspace",
        cancellation_cleanup="command_or_sandbox",
        timeout_cleanup="command_or_sandbox",
        docker_cli_env_allowlist=docker_cli_env_allowlist,
        _runtime_evidence=evidence,
    )


def _observer_args(argv: list[str]) -> list[str] | None:
    """Return the observer's mode and arguments, after its writable paths."""

    if _COMMAND_TREE_OBSERVER not in argv:
        return None
    start = argv.index(_COMMAND_TREE_OBSERVER) + 1
    return argv[start + 1 + int(argv[start]) :]


def _is_observer(argv: list[str], mode: str) -> bool:
    args = _observer_args(argv)
    return args is not None and args[0] == mode


class _GuestDocker:
    """Fake Docker CLI: the command never finishes; cleanup execs answer."""

    def __init__(
        self,
        *,
        found: ExecResult | None = None,
        verified: ExecResult | None = None,
        cancel: bool = False,
    ) -> None:
        self.found = found or ExecResult(stdout="4321 9876\n")
        self.verified = verified or ExecResult()
        self.cancel = cancel
        self.issued: list[list[str]] = []

    async def __call__(self, command, **kwargs):
        argv = command.argv
        self.issued.append(argv)
        if _is_observer(argv, "find"):
            redactor = kwargs.get("output_redactor")
            if redactor is None:
                return self.found
            return self.found.model_copy(update={"stdout": redactor.redact_text(self.found.stdout)})
        if _is_observer(argv, "verify"):
            return self.verified
        if argv[1:3] == ["rm", "-f"] or "kill -TERM" in argv[-1]:
            return ExecResult()
        if self.cancel:
            raise asyncio.CancelledError
        return ExecResult(timed_out=True, exit_code=-9)

    @property
    def removed(self) -> bool:
        return any(argv[1:3] == ["rm", "-f"] for argv in self.issued)


def _cleanup_steps(artifacts: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(item["action"], item["status"]) for item in artifacts]


@pytest.mark.parametrize("cancel", [False, True])
def test_docker_verified_tree_stop_keeps_container(monkeypatch, cancel: bool) -> None:
    docker = _GuestDocker(cancel=cancel)
    monkeypatch.setattr("cayu.runners.docker.run_subprocess", docker)
    runner = _evidence_runner()

    if cancel:
        with pytest.raises(asyncio.CancelledError) as cancelled:
            asyncio.run(runner.exec(ExecCommand.process("sleep", "999")))
        artifacts = cancelled.value.artifacts  # type: ignore[attr-defined]
    else:
        artifacts = asyncio.run(runner.exec(ExecCommand.bash("sleep 999"), timeout_s=1)).artifacts

    assert _cleanup_steps(artifacts) == [("kill_command", "completed")]
    assert runner.lifecycle_state == "reusable"
    assert not docker.removed
    launch, find, kill, verify = docker.issued
    pid_file = launch[launch.index("-c") + 2]
    assert pid_file.startswith("/tmp/cayu-docker-commands/")
    # The observer runs as a user the command cannot ptrace, with loader
    # settings cleared, and finds the supervisor from the host-recorded launch
    # arguments. Its launcher refuses an interpreter on guest-writable storage.
    assert find[:6] == ["/usr/bin/docker", "exec", "-u", "0:0", "-w", "/"]
    assert find[6:12] == ["-e", "LD_PRELOAD=", "-e", "LD_LIBRARY_PATH=", "-e", "LD_AUDIT="]
    assert find[12:16] == [_CONTAINER, "/bin/sh", "-c", _COMMAND_TREE_OBSERVER_LAUNCHER]
    assert find[16:23] == [
        _COMMAND_TREE_OBSERVER,
        "5",
        "/tmp",
        "/workspace",
        "/dev",
        "/proc",
        "/sys",
    ]
    assert _observer_args(find) == ["find", pid_file, "1.0"]
    # The command's user signals exactly the observed supervisor, never a PID
    # read from the guest-writable state file.
    assert "-u" not in kill
    assert kill[-1].startswith("kill -TERM 4321 ")
    assert "read pid" not in kill[-1]
    assert verify[:4] == ["/usr/bin/docker", "exec", "-u", "0:0"]
    assert _observer_args(verify) == ["verify", "4321", "9876", "2.0"]


def test_docker_verified_tree_stop_reads_the_identity_before_redaction(monkeypatch) -> None:
    # An allowlisted host value that happens to equal the supervisor's PID
    # must not erase it from the observer's answer.
    monkeypatch.setenv("CAYU_TEST_PROFILE", "4321")
    docker = _GuestDocker()
    monkeypatch.setattr("cayu.runners.docker.run_subprocess", docker)
    runner = _evidence_runner(docker_cli_env_allowlist=("CAYU_TEST_PROFILE",))

    result = asyncio.run(runner.exec(ExecCommand.process("sleep", "999"), timeout_s=1))

    assert _cleanup_steps(result.artifacts) == [("kill_command", "completed")]
    assert runner.lifecycle_state == "reusable"
    assert not docker.removed


@pytest.mark.parametrize(
    ("found", "verified"),
    [
        # The supervisor is gone (normal exit, or the command killed it).
        (ExecResult(exit_code=1), ExecResult()),
        (ExecResult(stdout="not a pid\n"), ExecResult()),
        (ExecResult(stdout="1 9876\n"), ExecResult()),
        (ExecResult(timed_out=True, exit_code=-9), ExecResult()),
        # A forged state file or acknowledgement cannot help: the observer
        # still sees the supervisor, or an orphan it left under PID 1.
        (ExecResult(stdout="4321 9876\n"), ExecResult(exit_code=1)),
        (ExecResult(stdout="4321 9876\n"), ExecResult(timed_out=True, exit_code=-9)),
    ],
)
def test_docker_unverified_tree_stop_removes_container(
    monkeypatch, found: ExecResult, verified: ExecResult
) -> None:
    docker = _GuestDocker(found=found, verified=verified)
    monkeypatch.setattr("cayu.runners.docker.run_subprocess", docker)
    runner = _evidence_runner()

    result = asyncio.run(runner.exec(ExecCommand.process("sleep", "999"), timeout_s=1))

    assert _cleanup_steps(result.artifacts) == [
        ("kill_command", "failed"),
        ("kill_sandbox", "completed"),
    ]
    assert runner.lifecycle_state == "closed"
    assert docker.removed
    assert runner_workspace_mutation_settlement(result=result, error=None) == "runner_quiescent"


@pytest.mark.parametrize(
    "runner_factory",
    [
        # No runtime evidence: shell supervisors, no observer.
        lambda: _docker_runner(),
        # No admitted python3: shell supervisors.
        lambda: _evidence_runner(required_executables=("sh",)),
        # Controls that let the command replace image files, gain UID 0
        # through a set-UID program, or use added capabilities.
        lambda: _evidence_runner(DockerWorkloadRestrictions(read_only_root=False)),
        lambda: _evidence_runner(DockerWorkloadRestrictions(no_new_privileges=False)),
        lambda: _evidence_runner(DockerWorkloadRestrictions(capability_add=("SYS_PTRACE",))),
        # A tmpfs over, or anywhere inside, the trees the observer's shell,
        # loader, loader configuration, interpreter and standard library come
        # from lets the command plant code the observer loads as UID 0.
        *(
            lambda target=target: _evidence_runner(
                DockerWorkloadRestrictions(
                    tmpfs=(
                        *DockerWorkloadRestrictions().tmpfs,
                        DockerTmpfsMount(target=target, size_bytes=1024 * 1024, mode=0o755),
                    )
                )
            )
            for target in (
                "/usr",
                "/etc",
                "/usr/lib64",
                "/usr/lib/python3.12/encodings/__pycache__",
                "/usr/local/lib/python3.12",
                "/opt/python",
            )
        ),
    ],
)
def test_docker_without_a_trusted_observer_always_removes_container(
    monkeypatch, runner_factory
) -> None:
    # The guest kill helper would report success, as a command that forged
    # its state file and acknowledgement could make it do.
    docker = _GuestDocker()
    monkeypatch.setattr("cayu.runners.docker.run_subprocess", docker)
    runner = runner_factory()

    result = asyncio.run(runner.exec(ExecCommand.bash("sleep 999"), timeout_s=1))

    assert _cleanup_steps(result.artifacts) == [
        ("kill_command", "failed"),
        ("kill_sandbox", "completed"),
    ]
    assert runner.lifecycle_state == "closed"
    assert docker.removed
    # Nothing guest-writable was consulted before removal.
    assert len(docker.issued) == 2


def test_docker_durable_receipt_cleanup_uses_container_removal(monkeypatch) -> None:
    issued: list[list[str]] = []

    async def fake_run_subprocess(command, **kwargs):
        del kwargs
        issued.append(command.argv)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    runner = _docker_runner()

    async def cleanup():
        from cayu.runners.docker import _DockerCommandHandle

        handle = _DockerCommandHandle(
            docker_path="/usr/bin/docker",
            name="a1",
            pid_file="/tmp/cayu-docker-commands/unused.pid",
            docker_cli_env_allowlist=(),
        )
        return await runner._cleanup_command(
            handle=handle, policy="command_or_sandbox", durable=True
        )

    result = asyncio.run(cleanup())

    assert [(item["action"], item["status"]) for item in result.artifacts] == [
        ("kill_sandbox", "completed")
    ]
    assert issued == [["/usr/bin/docker", "rm", "-f", "a1"]]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell process groups")
def test_shell_supervisor_cleanup_signals_its_process_group(tmp_path) -> None:
    process = subprocess.Popen(["sleep", "30"], start_new_session=True)
    pid_file = tmp_path / "command.pid"
    pid_file.write_text(f"{process.pid} 1\n", encoding="ascii")
    try:
        completed = subprocess.run(
            [
                "sh",
                "-c",
                _kill_supervised_command_script(str(pid_file)),
            ],
            capture_output=True,
            timeout=10,
        )
        assert process.wait(timeout=5) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert completed.returncode == 0
    assert not pid_file.exists()


_FAKE_PYTHON_SUPERVISOR = """\
import os, signal, sys, time
state, acknowledge = sys.argv[1], sys.argv[2] == "1"
def stop(_signum, _frame):
    if acknowledge:
        os.close(os.open(state + ".stopped", os.O_WRONLY | os.O_CREAT | os.O_EXCL))
    os.unlink(state)
    os._exit(0)
signal.signal(signal.SIGTERM, stop)
with open(state + ".tmp", "w") as handle:
    handle.write(f"{os.getpid()} 2\\n")
os.rename(state + ".tmp", state)
time.sleep(30)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("acknowledged", [False, True])
def test_python_supervisor_stop_needs_its_acknowledgement_not_just_a_removed_state_file(
    tmp_path, acknowledged: bool
) -> None:
    # A normal exit also removes the state file, and in direct mode it can
    # leave detached descendants running. Only the handler's acknowledgement,
    # written after it saw no remaining child, confirms the stop.
    pid_file = tmp_path / "command.pid"
    supervisor = subprocess.Popen(
        [sys.executable, "-c", _FAKE_PYTHON_SUPERVISOR, str(pid_file), str(int(acknowledged))]
    )
    try:
        completed = subprocess.run(
            [
                "sh",
                "-c",
                _kill_supervised_command_script(str(pid_file)),
            ],
            capture_output=True,
            timeout=20,
        )
        supervisor.wait(timeout=5)
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()

    assert completed.returncode == (0 if acknowledged else 1)
    assert not pid_file.exists()
    assert not (tmp_path / "command.pid.stopped").exists()


class _TimedOutRunner(DockerRunner):
    def __init__(self, artifacts: list[dict[str, Any]]) -> None:
        super().__init__("a1", docker_path="/usr/bin/docker")
        self._artifacts = artifacts

    async def exec(self, command, **kwargs):  # type: ignore[override]
        del command, kwargs
        return ExecResult(timed_out=True, exit_code=-9, artifacts=self._artifacts)


@pytest.mark.parametrize(
    ("artifacts", "expected"),
    [
        (
            [_artifact("kill_command", "completed")],
            "Command timed out after 3 seconds. The command was stopped.",
        ),
        (
            [_artifact("kill_command", "failed"), _artifact("kill_sandbox", "completed")],
            "Command timed out after 3 seconds. The command was stopped by removing its "
            "sandbox, so this environment cannot run more commands.",
        ),
        ([_artifact("kill_command", "failed")], "Command timed out after 3 seconds."),
    ],
)
def test_exec_command_result_reports_only_a_confirmed_stop(
    artifacts: list[dict[str, Any]], expected: str
) -> None:
    result = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="sess_1", runner=_TimedOutRunner(artifacts)),
            {"argv": ["sleep", "30"], "timeout_s": 3},
        )
    )

    assert result.is_error is True
    assert result.content == expected


_STATE = "/tmp/cayu-docker-commands/0123456789abcdef.pid"
_SUPERVISOR_ARGV = ("/usr/bin/python3", "-I", "-S", "-c", "supervisor", _STATE, "direct", "sh")


def _fake_proc(root: Path, processes: dict[int, tuple[Any, ...]]) -> Path:
    """Write ``pid: (state, ppid, starttime, argv[, threads])`` as /proc entries."""

    for pid, (state, parent, started, argv, *threads) in processes.items():
        entry = root / str(pid)
        entry.mkdir(parents=True)
        # Field 22 is starttime; the name may contain spaces and parentheses.
        fields = [state, str(parent), *["0"] * 17, str(started), "0"]
        (entry / "stat").write_text(f"{pid} (odd) name) {' '.join(fields)}\n", encoding="ascii")
        (entry / "cmdline").write_bytes(b"".join(arg.encode() + b"\0" for arg in argv))
        for thread in range(threads[0] if threads else 1):
            (entry / "task" / str(pid + thread)).mkdir(parents=True)
    kernel = root / "sys" / "kernel"
    kernel.mkdir(parents=True)
    (kernel / "ns_last_pid").write_text(f"{max(processes, default=1)}\n", encoding="ascii")
    return root


def _observe(
    proc: Path, *args: str, writable: tuple[str, ...] = (), prelude: str = ""
) -> subprocess.CompletedProcess[str]:
    program = _COMMAND_TREE_OBSERVER.replace('PROC = "/proc"', f"PROC = {str(proc)!r}\n{prelude}")
    assert program != _COMMAND_TREE_OBSERVER
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", program, str(len(writable)), *writable, *args],
        capture_output=True,
        text=True,
        timeout=20,
    )


_INIT = ("S", 0, 100, ("/sbin/docker-init", "--", "sleep", "infinity"))
_KEEPALIVE = ("S", 1, 101, ("sleep", "infinity"))


def test_tree_observer_finds_only_the_supervisor_docker_exec_started(tmp_path) -> None:
    # A decoy with the supervisor's exact arguments is the command's own
    # child, never a process with parent PID 0, so it is not the supervisor.
    proc = _fake_proc(
        tmp_path,
        {
            1: _INIT,
            7: _KEEPALIVE,
            40: ("S", 0, 500, _SUPERVISOR_ARGV),
            41: ("S", 40, 501, ("sh", "-c", "sleep 300")),
            42: ("S", 41, 502, _SUPERVISOR_ARGV),
            43: ("S", 0, 503, ("sh", "-c", f"cat {_STATE}")),
        },
    )

    found = _observe(proc, "find", _STATE, "0.1")

    assert (found.returncode, found.stdout) == (0, "40 500\n")
    assert _observe(proc, "find", _STATE + ".other", "0.1").returncode == 1


def test_tree_observer_refuses_an_ambiguous_supervisor(tmp_path) -> None:
    proc = _fake_proc(
        tmp_path,
        {1: _INIT, 40: ("S", 0, 500, _SUPERVISOR_ARGV), 44: ("S", 0, 504, _SUPERVISOR_ARGV)},
    )

    assert _observe(proc, "find", _STATE, "0.1").returncode == 1


@pytest.mark.parametrize(
    ("processes", "stopped"),
    [
        # The supervisor and its tree are gone; Docker's keepalive predates it.
        ({1: _INIT, 7: _KEEPALIVE}, True),
        # An unreaped orphan is not running.
        ({1: _INIT, 7: _KEEPALIVE, 60: ("Z", 1, 600, ())}, True),
        # A main thread that exited alone is a zombie while its other
        # threads keep running, so that process is still running.
        ({1: _INIT, 7: _KEEPALIVE, 60: ("Z", 1, 600, (), 2)}, False),
        # Work another command left behind before this one started is not its tree.
        ({1: _INIT, 7: _KEEPALIVE, 30: ("S", 1, 400, ("server",))}, True),
        # The supervisor still runs, for example because the command sent the
        # stop to a decoy named in its rewritten state file.
        ({1: _INIT, 40: ("S", 0, 500, _SUPERVISOR_ARGV)}, False),
        # A descendant escaped under PID 1 after the supervisor started.
        ({1: _INIT, 7: _KEEPALIVE, 60: ("S", 1, 600, ("sleep", "300"))}, False),
        ({1: _INIT, 7: _KEEPALIVE, 60: ("S", 1, 500, ("sleep", "300"))}, False),
        # Without PID 1 in view, /proc may hide the command's processes.
        ({7: _KEEPALIVE}, False),
    ],
)
def test_tree_observer_verifies_the_stop_from_proc(tmp_path, processes, stopped: bool) -> None:
    proc = _fake_proc(tmp_path, processes)

    verified = _observe(proc, "verify", "40", "500", "0.1")

    assert verified.returncode == (0 if stopped else 1)


def test_tree_observer_does_not_mistake_a_reused_pid_for_the_supervisor(tmp_path) -> None:
    proc = _fake_proc(tmp_path, {1: _INIT, 40: ("S", 0, 300, ("sh",))})

    assert _observe(proc, "verify", "40", "500", "0.1").returncode == 0


_CHURN = """
import shutil
_listdir, _churned = os.listdir, []
def _churn(path):
    names = _listdir(path)
    if path == PROC and not _churned:
        _churned.append(True)
        shutil.rmtree(PROC + "/60")
        os.makedirs(PROC + "/61/task/61")
        with open(PROC + "/61/stat", "w") as handle:
            handle.write("61 (sh) S 1 " + "0 " * 17 + "601 0\\n")
        with open(PROC + "/61/cmdline", "wb") as handle:
            handle.write(b"sh\\0")
        with open(PROC + "/sys/kernel/ns_last_pid", "w") as handle:
            handle.write("61\\n")
    return names
os.listdir = _churn
"""


def test_tree_observer_retries_a_scan_that_processes_started_during(tmp_path) -> None:
    # An orphan forks and exits after /proc is listed: its child is missing
    # from the listing and its own entry is gone when read. The namespace's
    # last allocated PID moved, so that scan proves nothing; the next sees
    # the child.
    proc = _fake_proc(tmp_path, {1: _INIT, 7: _KEEPALIVE, 60: ("S", 1, 600, ("sh",))})

    verified = _observe(proc, "verify", "40", "500", "0.3", prelude=_CHURN)

    assert verified.returncode == 1
    assert (proc / "61").exists()


def test_tree_observer_refuses_an_interpreter_in_writable_storage(tmp_path) -> None:
    proc = _fake_proc(tmp_path / "proc", {1: _INIT, 7: _KEEPALIVE})
    interpreter = str(Path(sys.executable).resolve().parent)
    library = next(path for path in sys.path if path and Path(path, "encodings").is_dir())

    assert _observe(proc, "verify", "40", "500", "0.1").returncode == 0
    assert _observe(proc, "verify", "40", "500", "0.1", writable=(interpreter,)).returncode == 1
    # A writable directory inside the standard library counts too.
    cache = library + "/encodings/__pycache__"
    assert _observe(proc, "verify", "40", "500", "0.1", writable=(cache,)).returncode == 1


_ORPHAN_EXIT = """
_open, _exited = open, []
def _exit_mid_scan(path, *args, **kwargs):
    if path == PROC + "/60/stat" and not _exited:
        _exited.append(True)
        import shutil
        shutil.rmtree(PROC + "/60")
        # Its child is reparented to PID 1; no PID was allocated.
        with _open(PROC + "/50/stat", "w") as handle:
            handle.write("50 (sleep) S 1 " + "0 " * 17 + "601 0\\n")
    return _open(path, *args, **kwargs)
open = _exit_mid_scan
_listdir = os.listdir
def _pid_order(path):
    names = _listdir(path)
    return sorted(names, key=lambda name: int(name) if name.isdigit() else -1)
os.listdir = _pid_order
"""


def test_tree_observer_retries_a_scan_that_processes_exited_during(tmp_path) -> None:
    # After PIDs wrap, an orphan's child can have the lower PID. The scan
    # reads the child first, still under its parent, and the parent exits
    # before its own entry is read. No PID was allocated, but the child's
    # parent is missing from the scan, so the next scan must decide.
    proc = _fake_proc(
        tmp_path,
        {
            1: _INIT,
            7: _KEEPALIVE,
            50: ("S", 60, 601, ("sleep", "300")),
            60: ("S", 1, 600, ("sh",)),
        },
    )

    verified = _observe(proc, "verify", "40", "500", "0.3", prelude=_ORPHAN_EXIT)

    assert verified.returncode == 1
    assert not (proc / "60").exists()


def _launch(
    path: str, *writable: str, system: str | None = None
) -> subprocess.CompletedProcess[str]:
    launcher = _COMMAND_TREE_OBSERVER_LAUNCHER
    if system is not None:
        # Stand in a test directory for the system interpreter directories.
        launcher = launcher.replace("/usr/bin|/usr/local/bin", system)
        assert launcher != _COMMAND_TREE_OBSERVER_LAUNCHER
    # "probe" is not a mode, so an observer that starts exits 125 at once.
    return subprocess.run(
        [
            "/bin/sh",
            "-c",
            launcher,
            _COMMAND_TREE_OBSERVER,
            str(len(writable)),
            *writable,
            "probe",
            "0.1",
        ],
        env={"PATH": path},
        capture_output=True,
        text=True,
        timeout=20,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
def test_tree_observer_launcher_runs_only_a_system_interpreter(tmp_path) -> None:
    real = Path(sys.executable).resolve()
    system = str(real.parent)
    if not (real.parent / "python3").exists():
        pytest.skip("interpreter directory has no python3")
    root = tmp_path.resolve()
    planted = root / "writable"
    planted.mkdir()
    (planted / "python3").symlink_to(real)
    (root / "bin").symlink_to(real.parent)
    other = str(root / "elsewhere")

    # The interpreter in a system directory, also through a directory symlink.
    assert _launch(system, other, system=system).returncode == 125
    assert _launch(str(root / "bin"), other, system=system).returncode == 125
    # Any other directory on the image's PATH is refused, writable or not.
    assert _launch(str(planted), other, system=system).returncode == 1
    assert _launch(str(planted), other).returncode == 1
    # So is a python3 in a system directory that resolves out of it, as a
    # symlink into writable storage would.
    assert _launch(str(planted), other, system=str(planted)).returncode == 1
    # And a system directory that is guest-writable.
    assert _launch(system, system, system=system).returncode == 1
    assert _launch(str(root / "missing"), other, system=system).returncode == 1


def _bind_app_tool_call(ctx: ToolContext, *, tool_name: str, arguments: dict[str, Any]) -> None:
    """Bind the durable authority CayuApp gives every tool call."""

    records: dict[str, dict[str, Any]] = {}

    async def load_operation(storage_key: str):
        return records.get(storage_key)

    async def compare_and_set_operation(storage_key, expected, desired, secondary):
        assert records.get(storage_key) == expected
        records[storage_key] = desired
        return desired

    _bind_runtime_tool_invocation_authority(
        ctx,
        parent_task_id="task",
        parent_run_epoch=1,
        model_step_id="mstep_" + "1" * 32,
        model_attempt_id="mattempt_" + "1" * 32,
        tool_round_id="tround_" + "1" * 32,
        tool_call_id="call",
        tool_name=tool_name,
        idempotency_key="invocation",
        effective_arguments=arguments,
        execution_profile_fingerprint="profile",
        environment_allocation_fingerprint="allocation",
        load_durable_operation=load_operation,
        compare_and_set_durable_operation=compare_and_set_operation,
        seal_durable_output=lambda record: record,
        secret_publication_sealer=lambda: None,
    )


@pytest.mark.parametrize("tool_name", ["run_command", "run_check"])
@pytest.mark.parametrize("app_tool_call", [False, True])
def test_receipt_backed_coding_commands_remove_the_container_on_timeout(
    monkeypatch, tmp_path, tool_name: str, app_tool_call: bool
) -> None:
    # Inside an app's tool call, run_command and run_check get a durable
    # receipt, and a receipt-backed command's timeout always removes the
    # container, even under command_or_sandbox. Only a call without that
    # authority (a bare ToolContext) gets the verified command-level stop.
    import cayu.tools.named_checks as named_checks
    import cayu.tools.structured_commands as structured_commands

    async def admitted(*args, **kwargs):
        return None

    monkeypatch.setattr(structured_commands, "ensure_coding_toolchain_runner_admission", admitted)
    monkeypatch.setattr(named_checks, "ensure_coding_toolchain_runner_admission", admitted)
    docker = _GuestDocker()
    monkeypatch.setattr("cayu.runners.docker.run_subprocess", docker)
    authority = DockerCodingCommandAuthority(
        selector="tests",
        revision="1",
        description="Run the tests.",
        exposure="named_check" if tool_name == "run_check" else "structured_command",
        executable="/usr/bin/python3",
        fixed_arguments=("-m", "pytest"),
        max_arguments=0,
        timeout_seconds=2,
        max_output_bytes=1024,
    )
    profile = DockerCodingToolchainProfile(
        profile_id="cleanup",
        revision="1",
        image_identity=DockerImageIdentity(reference="test@sha256:" + "a" * 64),
        platform_architecture="amd64",
        command_authorities=(authority,),
    )
    if tool_name == "run_check":
        tool: Any = RunCheckTool(
            checks=(
                NamedCheck(
                    name="tests",
                    description="Run the tests.",
                    command=ExecCommand.process(*authority.command_argv()),
                    timeout_s=2,
                    max_output_bytes=1024,
                    execution_profile_identity=ExecutionProfileBehaviorIdentity(
                        name="tests.cleanup-check", behavior_version="1", implementation_version="1"
                    ),
                ),
            ),
            toolchain_profile=profile,
            command_policy=ProcessCommandPolicy(
                allowed_executables=("/usr/bin/python3",), allowed_cwds=("/workspace",)
            ),
        )
        arguments: dict[str, Any] = {"check": "tests"}
    else:
        tool = RunCommandTool(toolchain_profile=profile)
        arguments = {"selector": "tests", "args": []}
    runner = _evidence_runner()
    # The capability says so: the command-level stop is reported for commands
    # without a receipt only.
    claim = runner.execution_capability_evidence().claim_for("confirmed_cancellation")
    assert claim is not None
    details = {detail.name: detail.value for detail in claim.adapter_details}
    assert details["command_tree_cleanup"] is True
    assert details["receipt_command_tree_cleanup"] is False
    ctx = ToolContext(
        session_id="session",
        agent_name="agent",
        environment_name="coding",
        workspace_id="workspace",
        idempotency_key="invocation",
        workspace=LocalWorkspace(tmp_path),
        runner=invocation_runner_handle(
            runner,
            redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
                revision=0, redactor=SecretRedactor()
            ),
        ),
    )
    if app_tool_call:
        _bind_app_tool_call(ctx, tool_name=tool_name, arguments=arguments)

    result = asyncio.run(tool.run(ctx, arguments))

    assert result.is_error is True
    assert result.structured["timed_out"] is True
    receipt_launched = any(guest_program() in argv for argv in docker.issued)
    if app_tool_call:
        assert receipt_launched
        assert _cleanup_steps(result.structured["artifacts"]) == [("kill_sandbox", "completed")]
        assert runner.lifecycle_state == "closed"
        assert docker.removed
        assert not any(_is_observer(argv, "find") for argv in docker.issued)
    else:
        assert not receipt_launched
        assert _cleanup_steps(result.structured["artifacts"]) == [("kill_command", "completed")]
        assert runner.lifecycle_state == "reusable"
        assert not docker.removed
