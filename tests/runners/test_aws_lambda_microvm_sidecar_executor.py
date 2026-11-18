"""The Kubernetes-mode sidecar's command executor and the supervisor's remote processes.

Most tests serve the executor from this process. Abstract sockets, peer
credentials, the executor's entry point and the agent namespace need Linux;
the agent namespace also needs unprivileged user namespaces. Those tests skip
elsewhere.
"""

from __future__ import annotations

import base64
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from examples.aws.lambda_microvm_sidecar import executor as executor_module
from examples.aws.lambda_microvm_sidecar import executor_protocol
from examples.aws.lambda_microvm_sidecar.executor import Executor, ExecutorServer
from examples.aws.lambda_microvm_sidecar.executor_protocol import (
    MAX_FRAME_BYTES,
    ExecutorClient,
    ExecutorError,
    ExecutorUnavailableError,
    FrameError,
    listen,
    recv_frame,
    send_frame,
)
from examples.aws.lambda_microvm_sidecar.remote_processes import RemoteProcess, RemoteProcesses
from examples.aws.lambda_microvm_sidecar.supervisor import (
    DEFAULT_COMMAND_PATH,
    CommandRequestError,
    CommandSupervisor,
    OwnerFence,
)
from tests.runners.sidecar_executor_harness import (
    EXECUTOR_PEER_CREDENTIALS,
    InProcessExecutor,
    agent_namespace_unavailable,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX executor")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_linux_only = pytest.mark.skipif(
    not EXECUTOR_PEER_CREDENTIALS, reason="abstract sockets and SO_PEERCRED are Linux-only"
)


_NAMESPACE_UNAVAILABLE = agent_namespace_unavailable()
_needs_agent_namespace = pytest.mark.skipif(
    _NAMESPACE_UNAVAILABLE is not None, reason=str(_NAMESPACE_UNAVAILABLE)
)


@pytest.fixture
def served(tmp_path: Path) -> Iterator[InProcessExecutor]:
    executor = InProcessExecutor(tmp_path)
    try:
        yield executor
    finally:
        executor.close()


def _supervisor(served: InProcessExecutor, root: Path, **options: Any) -> CommandSupervisor:
    return CommandSupervisor(
        root=root, processes=RemoteProcesses(served.client()), cancel_timeout_s=2, **options
    )


def _terminal(supervisor: CommandSupervisor, command_id: str, timeout_s: float = 15) -> dict:
    deadline = time.monotonic() + timeout_s
    while (result := supervisor.get(command_id))["state"] in {"accepted", "running"}:
        assert time.monotonic() < deadline, result
        time.sleep(0.01)
    return result


def _spawn(client: ExecutorClient, command_id: str, argv: list[str], root: Path, **fields: Any):
    return client.call(
        "spawn",
        command_id=command_id,
        argv=argv,
        env=fields.pop("env", {"PATH": os.environ["PATH"]}),
        cwd=str(root),
        process_tree=fields.pop("process_tree", "process_group"),
        **fields,
    )


def _read_all(client: ExecutorClient, command_id: str, stream: str = "stdout") -> bytes:
    data, offset = bytearray(), 0
    while True:
        response = client.call(
            "read", command_id=command_id, stream=stream, offset=offset, max_bytes=65536, wait_s=5
        )
        chunk = base64.b64decode(response["data_base64"])
        data.extend(chunk)
        offset += len(chunk)
        if response["eof"]:
            return bytes(data)


def test_a_command_gets_only_the_environment_it_was_given(
    served: InProcessExecutor, tmp_path: Path
) -> None:
    supervisor = _supervisor(served, tmp_path)
    supervisor.start(
        "env",
        {"kind": "process", "argv": ["/usr/bin/env"], "cwd": str(tmp_path), "env": {"ONLY": "1"}},
    )

    result = _terminal(supervisor, "env")
    assert result["exit_code"] == 0
    listed = base64.b64decode(result["stdout_base64"]).decode().splitlines()
    assert sorted(listed) == ["ONLY=1", f"PATH={DEFAULT_COMMAND_PATH}"]


def test_output_reaches_the_supervisor_within_its_limits(
    served: InProcessExecutor, tmp_path: Path
) -> None:
    supervisor = _supervisor(served, tmp_path)
    payload = b"x" * (3 * 1024 * 1024 + 17)
    supervisor.start(
        "big",
        {
            "kind": "process",
            "argv": [
                sys.executable,
                "-c",
                "import sys; data = sys.stdin.buffer.read(); sys.stdout.buffer.write(data); "
                "sys.stderr.write('done'); sys.exit(5)",
            ],
            "cwd": str(tmp_path),
            "stdin_base64": base64.b64encode(payload).decode("ascii"),
            "output_limit_bytes": 1000,
        },
    )

    result = _terminal(supervisor, "big")
    assert result["exit_code"] == 5
    assert base64.b64decode(result["stdout_base64"]) == payload[:1000]
    assert result["stdout_bytes"] == len(payload)
    assert result["stdout_truncated"] is True
    assert base64.b64decode(result["stderr_base64"]) == b"done"


def test_unread_output_stalls_the_command_until_the_supervisor_reads_it(tmp_path: Path) -> None:
    served = InProcessExecutor(tmp_path)
    served.executor.stream_window_bytes = 4096
    client = served.client()
    try:
        _spawn(
            client,
            "stalled",
            [sys.executable, "-c", "import os; os.write(1, b'y' * 1_000_000)"],
            tmp_path,
        )
        time.sleep(0.5)
        # The executor holds one window and stops draining the pipe: the writer blocks.
        assert client.call("wait", command_id="stalled", wait_s=0.2)["returncode"] is None

        assert _read_all(client, "stalled") == b"y" * 1_000_000
        assert client.call("wait", command_id="stalled", wait_s=5)["returncode"] == 0
    finally:
        served.close()


def test_cancel_stops_the_command_and_its_process_group(
    served: InProcessExecutor, tmp_path: Path
) -> None:
    supervisor = _supervisor(served, tmp_path)
    marker = tmp_path / "child.pid"
    supervisor.start(
        "sleeper",
        {
            "kind": "process",
            "argv": [
                sys.executable,
                "-c",
                "import pathlib, subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                f"pathlib.Path({str(marker)!r}).write_text(str(child.pid))\n"
                "time.sleep(60)\n",
            ],
            "cwd": str(tmp_path),
        },
    )
    deadline = time.monotonic() + 10
    while not marker.exists() or not marker.read_text():
        assert time.monotonic() < deadline
        time.sleep(0.01)

    result = supervisor.cancel("sleeper", reason="test")
    assert result["state"] == "cancelled"
    assert result["cancel_reason"] == "test"
    child = int(marker.read_text())
    deadline = time.monotonic() + 5
    while _alive(child):
        assert time.monotonic() < deadline, "the command's child outlived its process group"
        time.sleep(0.01)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_a_timed_out_command_is_stopped(served: InProcessExecutor, tmp_path: Path) -> None:
    supervisor = _supervisor(served, tmp_path)
    supervisor.start(
        "slow",
        {
            "kind": "process",
            "argv": [sys.executable, "-c", "import time; time.sleep(60)"],
            "cwd": str(tmp_path),
            "timeout_s": 0.3,
        },
    )

    result = _terminal(supervisor, "slow")
    assert result["timed_out"] is True
    assert result["exit_code"] < 0


def test_a_missing_program_fails_as_a_local_spawn_does(
    served: InProcessExecutor, tmp_path: Path
) -> None:
    supervisor = _supervisor(served, tmp_path)
    supervisor.start(
        "missing", {"kind": "process", "argv": ["/definitely/missing-cayu"], "cwd": str(tmp_path)}
    )

    result = _terminal(supervisor, "missing")
    assert (result["state"], result["error_type"]) == ("failed", "FileNotFoundError")


def test_the_executor_keeps_commands_inside_its_workspace(
    served: InProcessExecutor, tmp_path: Path
) -> None:
    supervisor = _supervisor(served, tmp_path)
    with pytest.raises(CommandRequestError, match="escapes the workspace root"):
        supervisor.start("outside", {"kind": "process", "argv": ["true"], "cwd": "/"})
    with pytest.raises(CommandRequestError, match="not a directory"):
        supervisor.start(
            "absent", {"kind": "process", "argv": ["true"], "cwd": str(tmp_path / "absent")}
        )
    # The executor checks a spawn's cwd itself too.
    with pytest.raises(ExecutorError, match="escapes the workspace root") as refused:
        _spawn(served.client(), "direct", ["true"], Path("/"))
    assert refused.value.kind == "invalid_request"


def test_interactive_stdio_moves_by_offset(served: InProcessExecutor, tmp_path: Path) -> None:
    supervisor = _supervisor(served, tmp_path)
    accepted = supervisor.start(
        "echo",
        {
            "kind": "process",
            "argv": [
                sys.executable,
                "-u",
                "-c",
                "import sys\nfor line in sys.stdin:\n    print(line.upper(), end='')\n",
            ],
            "cwd": str(tmp_path),
            "interactive": True,
        },
    )
    # Without an agent namespace the executor offers no PID namespace, and says so.
    assert accepted["process_tree"] == "process_group"

    def write(offset: int, data: bytes, eof: bool = False) -> dict:
        encoded = base64.b64encode(data).decode("ascii")
        return supervisor.write_stdin("echo", offset=offset, data_base64=encoded, eof=eof)

    assert write(0, b"one\n")["offset"] == 4
    # A replayed chunk is skipped, not written twice.
    assert write(0, b"one\ntwo\n")["offset"] == 8
    read = supervisor.read_stdout("echo", offset=0, max_bytes=100, wait_s=5)
    output = base64.b64decode(read["data_base64"])
    while len(output) < 8:
        more = supervisor.read_stdout("echo", offset=len(output), max_bytes=100, wait_s=5)
        output += base64.b64decode(more["data_base64"])
    assert output == b"ONE\nTWO\n"
    write(8, b"", eof=True)
    assert supervisor.read_stdout("echo", offset=8, max_bytes=100, wait_s=5)["eof"] is True
    assert _terminal(supervisor, "echo")["exit_code"] == 0


def test_spawn_and_reads_are_idempotent(served: InProcessExecutor, tmp_path: Path) -> None:
    client = served.client()
    argv = [sys.executable, "-c", "print('once')"]
    first = _spawn(client, "same", argv, tmp_path)
    # A connection reset after the executor spawned it only causes a retry.
    assert _spawn(client, "same", argv, tmp_path)["pid"] == first["pid"]
    with pytest.raises(ExecutorError, match="already spawned with another command") as conflict:
        _spawn(client, "same", [*argv, "x"], tmp_path)
    assert conflict.value.kind == "conflict"

    assert client.call("wait", command_id="same", wait_s=5)["returncode"] == 0
    request = {"command_id": "same", "stream": "stdout", "offset": 0, "max_bytes": 100}
    first_read = client.call("read", wait_s=5, **request)
    # A lost answer is recovered by reading the same offset again.
    assert client.call("read", wait_s=5, **request) == first_read
    assert base64.b64decode(first_read["data_base64"]) == b"once\n"


def test_a_new_supervisor_session_stops_the_earlier_sessions_commands(
    served: InProcessExecutor, tmp_path: Path
) -> None:
    earlier = served.client(session="earlier")
    _spawn(earlier, "orphan", [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path)
    process = served.executor._commands["orphan"].process

    later = served.client(session="later")
    later.call("hello")
    assert process.wait(timeout=5) < 0
    with pytest.raises(ExecutorError) as unknown:
        later.call("wait", command_id="orphan")
    assert unknown.value.kind == "unknown_command"


def test_a_stdin_write_waits_for_a_slow_reader_instead_of_failing(
    served: InProcessExecutor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reader takes far longer than a request timeout and the reconnect window.
    monkeypatch.setattr(executor_protocol, "_REQUEST_TIMEOUT_SECONDS", 0.2)
    client = served.client(reconnect_s=0.3)
    size = 2 * executor_protocol.MAX_CHUNK_BYTES
    reader = "import sys, time\ntime.sleep(1.5)\nprint(len(sys.stdin.buffer.read()))\n"
    _spawn(client, "slow", [sys.executable, "-c", reader], tmp_path)
    process = RemoteProcess(client, "slow")

    assert process.stdin.write(b"x" * size) == size
    process.stdin.close()
    output = bytearray()
    while chunk := process.stdout.read():
        output.extend(chunk)
    assert output.strip() == str(size).encode()
    assert process.wait(timeout=5) == 0


def test_release_forgets_a_command(served: InProcessExecutor, tmp_path: Path) -> None:
    client = served.client()
    _spawn(client, "done", ["true"], tmp_path)
    assert client.call("wait", command_id="done", wait_s=5)["returncode"] == 0

    client.call("release", command_id="done")
    with pytest.raises(ExecutorError, match="no command 'done'"):
        client.call("wait", command_id="done")


def test_every_answer_carries_the_boot_id_and_a_change_reports_a_restart(
    served: InProcessExecutor,
) -> None:
    restarted = threading.Event()
    client = served.client()
    client.on_restart = restarted.set
    assert client.call("hello")["boot_id"] == served.executor.boot_id
    with pytest.raises(ExecutorError):
        client.call("no-such-operation")
    assert client.boot_id == served.executor.boot_id
    assert not restarted.is_set()

    served.executor.boot_id = str(uuid.uuid4())
    client.call("hello")
    assert restarted.wait(5)


def test_a_restarted_executor_drops_the_owner_and_cancels_commands(
    served: InProcessExecutor, tmp_path: Path
) -> None:
    owner = OwnerFence()
    claim = "o" * 40
    owner.claim(claim)
    supervisor = _supervisor(served, tmp_path)
    processes = supervisor.processes
    assert isinstance(processes, RemoteProcesses)
    stopped = threading.Event()

    def restarted() -> None:
        owner.reset()
        supervisor.cancel_all()
        stopped.set()

    processes.client.on_restart = restarted
    with owner.admitted(claim) as generation:
        supervisor.start(
            "lost",
            {"kind": "process", "argv": ["sleep", "60"], "cwd": str(tmp_path)},
            owner_generation=generation,
        )
    served.executor.boot_id = str(uuid.uuid4())
    processes.hello()

    assert stopped.wait(10)
    assert not owner.is_current(claim)
    assert _terminal(supervisor, "lost")["state"] == "cancelled"
    # A new host claims again and is admitted.
    assert owner.claim("n" * 40) == (2, False)


def test_unknown_operations_and_malformed_requests_are_refused(
    served: InProcessExecutor,
) -> None:
    client = served.client()
    for op, fields, kind in (
        ("format-disk", {}, "invalid_request"),
        ("wait", {"command_id": "absent"}, "unknown_command"),
        ("read", {"command_id": "x/y"}, "invalid_request"),
    ):
        with pytest.raises(ExecutorError) as refused:
            client.call(op, **fields)
        assert refused.value.kind == kind


def test_a_caller_with_another_uid_is_refused_before_any_request(tmp_path: Path) -> None:
    executor = Executor(root=tmp_path, isolate=False)
    directory = Path(_short_directory())
    try:
        address = str(directory / "executor.sock")
        server = ExecutorServer(
            executor,
            listen(address),
            supervisor_uid=10001,
            peer_uid=lambda _connection: 1000,
        )
        server.start()
        client = ExecutorClient(address, session="s", reconnect_s=0)
        with pytest.raises(ExecutorUnavailableError, match="unreachable"):
            client.call("hello")
        server.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _short_directory() -> str:
    return tempfile.mkdtemp(prefix="cayu-executor-", dir="/tmp")


def test_the_client_reconnects_until_the_executor_listens(tmp_path: Path) -> None:
    directory = Path(_short_directory())
    address = str(directory / "executor.sock")
    started: list[ExecutorServer] = []

    def serve_later() -> None:
        time.sleep(0.5)
        server = ExecutorServer(
            Executor(root=tmp_path, isolate=False),
            listen(address),
            supervisor_uid=os.getuid(),
            peer_uid=lambda _connection: os.getuid(),
        )
        server.start()
        started.append(server)

    threading.Thread(target=serve_later, daemon=True).start()
    try:
        client = ExecutorClient(address, session="s", reconnect_s=10)
        assert type(client.call("hello")["boot_id"]) is str
        with pytest.raises(ExecutorUnavailableError):
            ExecutorClient(str(directory / "absent.sock"), session="s", reconnect_s=0).call("hello")
    finally:
        for server in started:
            server.close()
        shutil.rmtree(directory, ignore_errors=True)


def test_frames_are_bounded_json_objects() -> None:
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with left, right:
        send_frame(left, {"op": "hello"})
        assert recv_frame(right) == {"op": "hello"}
        with pytest.raises(FrameError, match="exceeds"):
            send_frame(left, {"data": "x" * MAX_FRAME_BYTES})
        left.sendall((MAX_FRAME_BYTES + 1).to_bytes(4, "big"))
        with pytest.raises(FrameError, match="exceeds"):
            recv_frame(right)
        left.sendall((2).to_bytes(4, "big") + b"[]")
        with pytest.raises(FrameError, match="not a JSON object"):
            recv_frame(right)


@_linux_only
def test_peer_credentials_name_the_calling_uid() -> None:
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with left, right:
        assert executor_module.peer_uid(right) == os.getuid()


@_linux_only
def test_supervisor_and_executor_run_as_two_processes(tmp_path: Path) -> None:
    address = f"@cayu-test-executor-{uuid.uuid4().hex}"
    serve = (
        "import os, sys\n"
        "from examples.aws.lambda_microvm_sidecar.executor import Executor, ExecutorServer\n"
        "from examples.aws.lambda_microvm_sidecar.executor_protocol import listen\n"
        "listener = listen(sys.argv[1])\n"
        "print('listening', flush=True)\n"
        "ExecutorServer(Executor(root=sys.argv[2], isolate=False), listener,\n"
        "               supervisor_uid=int(sys.argv[3])).serve_forever()\n"
    )

    def start(supervisor_uid: int) -> subprocess.Popen[str]:
        process = subprocess.Popen(
            [sys.executable, "-c", serve, address, str(tmp_path), str(supervisor_uid)],
            cwd=_PROJECT_ROOT,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "listening"
        return process

    executor = start(os.getuid())
    try:
        supervisor = CommandSupervisor(
            root=tmp_path,
            processes=RemoteProcesses(ExecutorClient(address, session="s", reconnect_s=5)),
        )
        supervisor.start(
            "two", {"kind": "shell", "shell": "printf from-executor", "cwd": str(tmp_path)}
        )
        result = _terminal(supervisor, "two")
        assert base64.b64decode(result["stdout_base64"]) == b"from-executor"
    finally:
        executor.terminate()
        executor.wait(timeout=5)

    # The executor's peer check uses the kernel's credentials: another UID is refused.
    executor = start(os.getuid() + 1)
    try:
        with pytest.raises(ExecutorUnavailableError):
            ExecutorClient(address, session="s", reconnect_s=0).call("hello")
    finally:
        executor.terminate()
        executor.wait(timeout=5)


@_linux_only
@pytest.mark.parametrize(
    ("supervisor_uid", "message"),
    [
        ("", "must name the supervisor container's UID"),
        ("self", "must differ from the executor's own UID"),
    ],
)
def test_the_executor_entry_point_refuses_an_unsafe_configuration(
    supervisor_uid: str, message: str
) -> None:
    if os.geteuid() == 0:
        message = "must not run as root" if supervisor_uid == "self" else message
    environment = {
        "PATH": os.environ["PATH"],
        "CAYU_EXECUTOR_SOCKET": f"@cayu-test-executor-{uuid.uuid4().hex}",
        "CAYU_EXECUTOR_SUPERVISOR_UID": str(os.geteuid()) if supervisor_uid == "self" else "",
    }
    refused = subprocess.run(
        [sys.executable, "-m", "examples.aws.lambda_microvm_sidecar.executor"],
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert refused.returncode != 0
    assert message in refused.stderr


def test_commands_enter_the_agent_namespace_with_only_the_executor_interpreter() -> None:
    namespace = object.__new__(executor_module.AgentNamespace)
    namespace.interpreter = "/usr/local/bin/python3"
    namespace.init_pid = 42

    argv = namespace.command(
        ["git", "status"], {"PATH": "/usr/bin", "A": "x=y"}, cwd="/workspace/repo"
    )

    # Isolated mode, and no nsenter, env or shell from the agent's image.
    assert argv == [
        "/usr/local/bin/python3",
        "-I",
        "-S",
        "-c",
        executor_module.NAMESPACE_ENTER,
        "42",
        "/workspace/repo",
        "2",
        "PATH=/usr/bin",
        "A=x=y",
        "git",
        "status",
    ]
    with pytest.raises(CommandRequestError, match="env key"):
        namespace.command(["true"], {"A=B": "1"}, cwd="/")


def test_the_agent_namespace_holder_runs_without_the_interpreter_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[list[str]] = []

    class Stop(Exception):
        pass

    def popen(argv: list[str], **kwargs: Any) -> None:
        del kwargs
        started.append(argv)
        raise Stop

    monkeypatch.setattr(executor_module.subprocess, "Popen", popen)
    with pytest.raises(Stop):
        executor_module.AgentNamespace("/usr/local/bin/python3")

    # No .pth file of the base interpreter runs outside the agent namespace.
    assert started == [
        ["/usr/local/bin/python3", "-I", "-S", "-c", executor_module.NAMESPACE_HOLDER]
    ]


@_needs_agent_namespace
def test_agent_commands_run_in_their_directory_with_exactly_their_environment(
    tmp_path: Path,
) -> None:
    served = InProcessExecutor(tmp_path, isolate=True)
    supervisor = _supervisor(served, tmp_path)
    (tmp_path / "repo").mkdir()
    script = "import os, sys; print(os.getcwd()); print(os.getuid()); sys.exit(7)"
    try:
        supervisor.start(
            "cwd",
            {
                "kind": "process",
                "argv": [sys.executable, "-I", "-c", script],
                "cwd": str(tmp_path / "repo"),
            },
        )
        result = _terminal(supervisor, "cwd")
        lines = base64.b64decode(result["stdout_base64"]).decode().splitlines()
        assert result["exit_code"] == 7
        assert lines == [str((tmp_path / "repo").resolve()), str(os.getuid())]

        supervisor.start(
            "env",
            {
                "kind": "process",
                "argv": ["env"],
                "cwd": str(tmp_path),
                "env": {"ONLY": "a=b", "EMPTY": ""},
            },
        )
        listed = base64.b64decode(_terminal(supervisor, "env")["stdout_base64"]).decode()
        assert sorted(listed.splitlines()) == ["EMPTY=", "ONLY=a=b", f"PATH={DEFAULT_COMMAND_PATH}"]

        supervisor.start("pwd", {"kind": "process", "argv": ["pwd"], "cwd": str(tmp_path)})
        pwd = _terminal(supervisor, "pwd")
        assert base64.b64decode(pwd["stdout_base64"]).decode().strip() == str(tmp_path.resolve())

        supervisor.start(
            "missing", {"kind": "process", "argv": ["missing-cayu-program"], "cwd": str(tmp_path)}
        )
        missing = _terminal(supervisor, "missing")
        assert missing["exit_code"] == 127
        assert b"missing-cayu-program" in base64.b64decode(missing["stderr_base64"])

        supervisor.start(
            "signal",
            {"kind": "process", "argv": ["sh", "-c", "kill -TERM $$"], "cwd": str(tmp_path)},
        )
        assert _terminal(supervisor, "signal")["exit_code"] == -signal.SIGTERM
        supervisor.start(
            "killed",
            {"kind": "process", "argv": ["sh", "-c", "kill -KILL $$"], "cwd": str(tmp_path)},
        )
        killed = _terminal(supervisor, "killed")
        assert killed["exit_code"] == -signal.SIGKILL
        assert base64.b64decode(killed["stderr_base64"]) == b""
        # Commands start with SIGPIPE at its default action, so a pipeline's writer dies of it.
        supervisor.start(
            "pipe",
            {
                "kind": "process",
                "argv": ["sh", "-c", "(yes; echo $? >&2) | head -n 1 >/dev/null"],
                "cwd": str(tmp_path),
            },
        )
        piped = _terminal(supervisor, "pipe")
        assert base64.b64decode(piped["stderr_base64"]).strip() == b"141"
    finally:
        served.close()


@_needs_agent_namespace
def test_agent_commands_run_scripts_without_a_shebang_with_the_shell(tmp_path: Path) -> None:
    served = InProcessExecutor(tmp_path, isolate=True)
    supervisor = _supervisor(served, tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "cayu-script"
    script.write_text('echo "$0|$*|$MARK"\nexit 5\n')
    script.chmod(0o755)
    unexecutable = bin_dir / "cayu-unexecutable"
    unexecutable.write_text("exit 0\n")
    denied_dir = tmp_path / "denied"
    denied_dir.mkdir()
    (denied_dir / "cayu-script").write_text("exit 9\n")
    try:
        # As execvp(3): the file is run with /bin/sh, its path as $0, the arguments and env kept.
        supervisor.start(
            "direct",
            {
                "kind": "process",
                "argv": [str(script), "a b", "c"],
                "cwd": str(tmp_path),
                "env": {"MARK": "direct"},
            },
        )
        direct = _terminal(supervisor, "direct")
        assert direct["exit_code"] == 5
        assert base64.b64decode(direct["stdout_base64"]).decode() == f"{script}|a b c|direct\n"

        supervisor.start(
            "path",
            {
                "kind": "process",
                "argv": ["cayu-script", "x"],
                "cwd": str(tmp_path),
                "env": {"PATH": f"{tmp_path / 'missing'}:{denied_dir}:{bin_dir}", "MARK": "path"},
            },
        )
        found = _terminal(supervisor, "path")
        assert found["exit_code"] == 5
        assert base64.b64decode(found["stdout_base64"]).decode() == f"{script}|x|path\n"

        supervisor.start(
            "unexecutable",
            {"kind": "process", "argv": [str(unexecutable)], "cwd": str(tmp_path)},
        )
        refused = _terminal(supervisor, "unexecutable")
        assert refused["exit_code"] == 126
        assert b"Permission denied" in base64.b64decode(refused["stderr_base64"])

        # A file found but not executable is reported over the later entries that lack it.
        supervisor.start(
            "denied",
            {
                "kind": "process",
                "argv": ["cayu-script"],
                "cwd": str(tmp_path),
                "env": {"PATH": f"{denied_dir}:{tmp_path / 'missing'}"},
            },
        )
        denied = _terminal(supervisor, "denied")
        assert denied["exit_code"] == 126
        assert b"Permission denied" in base64.b64decode(denied["stderr_base64"])
    finally:
        served.close()


@_needs_agent_namespace
def test_agent_commands_cannot_signal_the_executor(tmp_path: Path) -> None:
    served = InProcessExecutor(tmp_path, isolate=True)
    supervisor = _supervisor(served, tmp_path)
    try:
        supervisor.start(
            "kill",
            {
                "kind": "shell",
                "shell": "cat /proc/1/cmdline | tr '\\0' ' '; echo; kill -9 -1; kill -9 1; true",
                "cwd": str(tmp_path),
            },
        )
        result = _terminal(supervisor, "kill")
        # PID 1 in the command's view is the agent namespace's init.
        assert b"libc.unshare" in base64.b64decode(result["stdout_base64"])
        # The executor (this process) survived, and kept serving.
        assert type(served.client().call("hello")["boot_id"]) is str
    finally:
        served.close()


@_needs_agent_namespace
def test_the_agent_namespace_is_renewed_with_a_new_boot_id(tmp_path: Path) -> None:
    served = InProcessExecutor(tmp_path, isolate=True)
    try:
        client = served.client()
        before = client.call("hello")["boot_id"]
        namespace = served.executor._namespace
        assert namespace is not None
        pid_namespace = served.executor.pid_namespace
        # The probe's answer is what it prints; an exit status the orphan reaper
        # collected first would read as success.
        assert not namespace._prints_marker(["true"])
        os.kill(namespace.init_pid, signal.SIGKILL)

        deadline = time.monotonic() + 15
        while client.call("hello")["boot_id"] == before:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert served.executor._namespace is not namespace
        assert served.executor.pid_namespace is pid_namespace
        supervisor = _supervisor(served, tmp_path)
        supervisor.start("after", {"kind": "process", "argv": ["true"], "cwd": str(tmp_path)})
        assert _terminal(supervisor, "after")["exit_code"] == 0
    finally:
        served.close()


@_needs_agent_namespace
def test_a_setsid_server_outlives_its_command_until_a_new_supervisor_session(
    tmp_path: Path,
) -> None:
    served = InProcessExecutor(tmp_path, isolate=True)
    supervisor = _supervisor(served, tmp_path)
    marker = tmp_path / "server.alive"
    try:
        supervisor.start(
            "server",
            {
                "kind": "shell",
                "shell": (
                    "id -u; setsid sh -c 'i=0; while true; do i=$((i+1)); echo $i > server.alive; sleep 0.1; done' "
                    "</dev/null >/dev/null 2>&1 &"
                ),
                "cwd": str(tmp_path),
            },
        )
        result = _terminal(supervisor, "server")
        assert base64.b64decode(result["stdout_base64"]).strip() == str(os.getuid()).encode()
        time.sleep(0.5)
        first = marker.read_text()
        time.sleep(0.5)
        assert marker.read_text() != first

        # A restarted supervisor knows nothing of it, so the executor ends it.
        before = served.executor.boot_id
        later = served.client(session="later")
        assert later.call("hello")["boot_id"] != before
        last = marker.read_text()
        time.sleep(0.5)
        assert marker.read_text() == last
        later_supervisor = CommandSupervisor(
            root=tmp_path, processes=RemoteProcesses(later), cancel_timeout_s=2
        )
        later_supervisor.start("after", {"kind": "process", "argv": ["true"], "cwd": str(tmp_path)})
        assert _terminal(later_supervisor, "after")["exit_code"] == 0
    finally:
        served.close()


@_needs_agent_namespace
def test_interactive_commands_get_a_pid_namespace_whose_teardown_is_confirmed(
    tmp_path: Path,
) -> None:
    served = InProcessExecutor(tmp_path, isolate=True)
    supervisor = _supervisor(served, tmp_path)
    try:
        if not served.executor.pid_namespace:
            pytest.skip("this host does not allow nested PID namespaces")
        accepted = supervisor.start(
            "tree",
            {
                "kind": "shell",
                "shell": "setsid sleep 60 & sleep 60",
                "cwd": str(tmp_path),
                "interactive": True,
            },
        )
        assert accepted["process_tree"] == "pid_namespace"
        time.sleep(0.5)
        result = supervisor.cancel("tree")
        assert result["state"] == "cancelled"
        assert result["process_tree_stopped"] is True
    finally:
        served.close()
