"""Sidecar protocol 6: interactive commands with streamed stdio and confirmed teardown."""

from __future__ import annotations

import asyncio
import base64
import importlib
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from examples.aws.lambda_microvm_sidecar import supervisor as sidecar
from examples.aws.lambda_microvm_sidecar.supervisor import (
    CommandExecutionBoundary,
    CommandNotFoundError,
    CommandRequestError,
    CommandSupervisor,
)

_ECHO_UPPER = (
    "import sys\n"
    "for line in sys.stdin:\n"
    "    sys.stdout.write(line.upper())\n"
    "    sys.stdout.flush()\n"
    "sys.exit(7)\n"
)


def _start(supervisor: CommandSupervisor, command_id: str, root: Path, script: str) -> dict:
    return supervisor.start(
        command_id,
        {
            "kind": "process",
            "argv": [sys.executable, "-c", script],
            "cwd": str(root),
            "env": {},
            "interactive": True,
            "timeout_s": 30,
        },
    )


def _write(supervisor: CommandSupervisor, command_id: str, offset: int, data: bytes, eof=False):
    return supervisor.write_stdin(
        command_id,
        offset=offset,
        data_base64=base64.b64encode(data).decode("ascii"),
        eof=eof,
    )


def _read(supervisor: CommandSupervisor, command_id: str, offset: int, wait_s: float = 5):
    response = supervisor.read_stdout(command_id, offset=offset, max_bytes=1024, wait_s=wait_s)
    return response, base64.b64decode(response["data_base64"])


def _terminal(supervisor: CommandSupervisor, command_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while True:
        result = supervisor.get(command_id)
        if result["state"] in {"completed", "cancelled", "failed"}:
            return result
        assert time.monotonic() < deadline, result
        time.sleep(0.01)


def test_interactive_command_streams_stdio_until_exit(tmp_path: Path) -> None:
    supervisor = CommandSupervisor(root=tmp_path)

    accepted = _start(supervisor, "chat", tmp_path, _ECHO_UPPER)
    assert accepted == {
        "command_id": "chat",
        "state": "accepted",
        "interactive": True,
        "process_tree": "process_group",
    }

    assert _write(supervisor, "chat", 0, b"hello\n") == {
        "command_id": "chat",
        "offset": 6,
        "closed": False,
    }
    response, data = _read(supervisor, "chat", 0)
    assert (response["offset"], data, response["eof"]) == (0, b"HELLO\n", False)
    # A second line arrives while the first is still unacknowledged.
    _write(supervisor, "chat", 6, b"world\n", eof=True)
    output = b""
    offset = 6
    while True:
        response, data = _read(supervisor, "chat", offset)
        output += data
        offset += len(data)
        if response["eof"]:
            break
    assert output == b"WORLD\n"
    assert response["state"] in {"running", "completed"}

    result = _terminal(supervisor, "chat")
    assert result["exit_code"] == 7
    assert result["interactive"] is True
    assert result["stdout_base64"] == ""
    assert result["stdout_bytes"] == 12
    # A process group alone cannot prove that no descendant escaped it.
    assert result["process_tree_stopped"] is False


def test_interactive_stdio_retries_are_idempotent(tmp_path: Path) -> None:
    supervisor = CommandSupervisor(root=tmp_path)
    _start(supervisor, "retry", tmp_path, _ECHO_UPPER)

    _write(supervisor, "retry", 0, b"abc\n")
    # A replayed chunk is skipped, and its unseen suffix is written once.
    assert _write(supervisor, "retry", 0, b"abc\ndef\n")["offset"] == 8
    with pytest.raises(CommandRequestError, match="skips bytes"):
        _write(supervisor, "retry", 20, b"x")

    first, data = _read(supervisor, "retry", 0)
    # Reading the same offset again (a lost response) returns the same bytes.
    again, replay = _read(supervisor, "retry", 0, wait_s=0)
    assert data and replay.startswith(data)
    assert first["offset"] == again["offset"] == 0
    with pytest.raises(CommandRequestError, match="outside the retained window"):
        _read(supervisor, "retry", len(replay) + 100, wait_s=0)
    supervisor.cancel("retry")


def test_cancel_abandons_unread_output_and_settles(tmp_path: Path) -> None:
    supervisor = CommandSupervisor(root=tmp_path)
    _start(supervisor, "stop", tmp_path, "import time\nprint('x', flush=True)\ntime.sleep(60)\n")
    _read(supervisor, "stop", 0)

    result = supervisor.cancel("stop")

    assert result["state"] == "cancelled"
    assert result["cancelled"] is True
    response, data = _read(supervisor, "stop", 0, wait_s=0)
    assert (data, response["eof"]) == (b"", True)
    assert _write(supervisor, "stop", 0, b"late")["closed"] is True


def test_interactive_validation(tmp_path: Path) -> None:
    supervisor = CommandSupervisor(root=tmp_path)
    base = {"kind": "process", "argv": ["true"], "cwd": str(tmp_path), "interactive": True}

    with pytest.raises(CommandRequestError, match="write_stdin"):
        supervisor.start("stdin", {**base, "stdin_base64": "eA=="})
    with pytest.raises(CommandRequestError, match="agent profile"):
        supervisor.start("trusted", {**base, "execution_profile": "trusted"})
    with pytest.raises(CommandNotFoundError):
        supervisor.read_stdout("missing", offset=0, max_bytes=1, wait_s=0)
    supervisor.start("batch", {**base, "interactive": False})
    with pytest.raises(CommandRequestError, match="not interactive"):
        supervisor.read_stdout("batch", offset=0, max_bytes=1, wait_s=0)
    _start(supervisor, "limits", tmp_path, _ECHO_UPPER)
    with pytest.raises(CommandRequestError, match="max_bytes"):
        supervisor.read_stdout("limits", offset=0, max_bytes=0, wait_s=0)
    with pytest.raises(CommandRequestError, match="wait_s"):
        supervisor.read_stdout("limits", offset=0, max_bytes=1, wait_s=60)
    supervisor.cancel("limits")


def test_stdout_window_applies_backpressure_until_acknowledged() -> None:
    streams = sidecar._InteractiveStreams(window_bytes=4)
    streams.add_stdout(b"abcd")
    added = threading.Event()

    def produce() -> None:
        streams.add_stdout(b"efgh")
        added.set()

    threading.Thread(target=produce, daemon=True).start()
    assert not added.wait(0.1)
    assert streams.read(0, max_bytes=8, wait_s=0) == (b"abcd", False)
    assert not added.wait(0.1)
    # Reading at offset 4 acknowledges the first four bytes, freeing the window.
    assert streams.read(4, max_bytes=8, wait_s=1)[0] == b"efgh"
    assert added.wait(1)
    streams.finish_stdout()
    assert streams.read(8, max_bytes=8, wait_s=0) == (b"", True)


def test_agent_interactive_command_runs_in_its_own_pid_namespace() -> None:
    boundary = CommandExecutionBoundary(
        agent_uid=1000,
        agent_gid=1000,
        agent_netns="cayu-agent",
        init_interpreter="/usr/bin/python3.11",
    )

    argv, environment = boundary.command_for(
        ["claude", "--print"], {"HOME": "/tmp/h"}, execution_profile="agent", interactive=True
    )

    assert argv[:9] == [
        "/usr/sbin/ip",
        "netns",
        "exec",
        "cayu-agent",
        "/usr/bin/unshare",
        "--pid",
        "--fork",
        "--kill-child",
        "--",
    ]
    assert argv[9] == "/usr/bin/setpriv"
    # The init runs after the identity drop, with the command's environment.
    tail = argv[argv.index("/usr/bin/env") :]
    assert tail == [
        "/usr/bin/env",
        "-i",
        "--",
        "HOME=/tmp/h",
        "/usr/bin/python3.11",
        "-I",
        "-c",
        sidecar.PID_NAMESPACE_INIT,
        "claude",
        "--print",
    ]
    assert environment == sidecar.LAUNCHER_ENVIRONMENT
    assert boundary.isolates_process_tree(execution_profile="agent")
    assert not boundary.isolates_process_tree(execution_profile="trusted")
    assert not CommandExecutionBoundary().isolates_process_tree(execution_profile="agent")


def test_isolated_validation_reports_pid_namespace(tmp_path: Path) -> None:
    boundary = CommandExecutionBoundary(agent_uid=1000, agent_gid=1000)

    payload = sidecar._validated_payload(
        {"kind": "process", "argv": ["cat"], "cwd": str(tmp_path), "interactive": True},
        root=tmp_path.resolve(),
        execution_boundary=boundary,
    )

    assert payload["process_tree"] == "pid_namespace"
    assert "/usr/bin/unshare" in payload["argv"]


def test_namespace_init_runs_reaps_and_returns_the_command_status() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            sidecar.PID_NAMESPACE_INIT,
            "/bin/sh",
            "-c",
            "(sleep 0 &) ; echo started; exit 3",
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.stdout == b"started\n"
    assert completed.returncode == 3


def _write_stat(root: Path, pid: int, *, state: str, ppid: int, start: str) -> None:
    fields = [state, str(ppid), *["0"] * 17, start, "0"]
    (root / str(pid)).mkdir(parents=True, exist_ok=True)
    (root / str(pid) / "stat").write_text(f"{pid} (odd) name) " + " ".join(fields))


def test_proc_stat_tolerates_parentheses_in_command_names(tmp_path: Path) -> None:
    _write_stat(tmp_path, 41, state="S", ppid=7, start="123")

    assert sidecar._proc_stat(41, proc_root=str(tmp_path)) == ("S", 7, "123")
    assert sidecar._proc_stat(42, proc_root=str(tmp_path)) is None
    assert sidecar._child_pids(7, proc_root=str(tmp_path)) == [41]


@pytest.mark.parametrize(
    ("state", "start", "expected"),
    [
        (None, "100", True),  # gone: the kernel reaped the namespace
        ("Z", "100", True),  # zombie: exit, including namespace teardown, completed
        ("S", "999", True),  # the PID now belongs to an unrelated process
        ("S", "100", False),  # the init is still running
    ],
)
def test_process_tree_confirmation_tracks_the_namespace_init(
    tmp_path: Path, state: str | None, start: str, expected: bool
) -> None:
    if state is not None:
        _write_stat(tmp_path, 41, state=state, ppid=7, start=start)
    record = sidecar._CommandRecord(command_id="c", payload_fingerprint=None)
    record.process_tree = "pid_namespace"
    record.namespace_init = (41, "100")

    assert (
        sidecar._process_tree_stopped(record, timeout_s=0.05, proc_root=str(tmp_path)) is expected
    )


def test_process_group_commands_are_never_confirmed() -> None:
    record = sidecar._CommandRecord(command_id="c", payload_fingerprint=None)

    assert sidecar._process_tree_stopped(record, timeout_s=0) is False


def test_unconfirmed_interactive_records_are_never_pruned(tmp_path: Path) -> None:
    supervisor = CommandSupervisor(root=tmp_path, result_ttl_s=0.01)
    _start(supervisor, "kept", tmp_path, "pass\n")
    assert _terminal(supervisor, "kept")["process_tree_stopped"] is False
    time.sleep(0.05)

    # The record survives, so a cancel reports it rather than a fresh tombstone.
    assert supervisor.cancel("kept")["process_tree_stopped"] is False
    assert supervisor.cancel("never-started")["process_tree_stopped"] is True


def test_sidecar_routes_require_the_current_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    monkeypatch.setenv("CAYU_MICROVM_WORKSPACE_ROOT", str(tmp_path))
    sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)
    app = importlib.import_module("examples.aws.lambda_microvm_sidecar.app")
    # The routes' owner checks are under test, not the agent launchers.
    monkeypatch.setattr(app, "SUPERVISOR", CommandSupervisor(root=tmp_path))
    claim = "c" * 40
    try:
        asyncio.run(app.claim_owner({"claim_id": claim}))
        app.SUPERVISOR.start(
            "routed",
            {
                "kind": "process",
                "argv": [sys.executable, "-c", _ECHO_UPPER],
                "cwd": str(tmp_path),
                "interactive": True,
            },
        )
        written = asyncio.run(
            app.write_command_stdin(
                "routed",
                {"owner_claim": claim, "offset": 0, "data_base64": "aGkK", "eof": True},
            )
        )
        assert written["offset"] == 3
        read = asyncio.run(
            app.read_command_stdout(
                "routed", {"owner_claim": claim, "offset": 0, "max_bytes": 16, "wait_s": 5}
            )
        )
        assert base64.b64decode(read["data_base64"]) == b"HI\n"
        assert _terminal(app.SUPERVISOR, "routed")["exit_code"] == 7
        with pytest.raises(HTTPException) as stale:
            asyncio.run(
                app.read_command_stdout(
                    "routed", {"owner_claim": "d" * 40, "offset": 0, "max_bytes": 1}
                )
            )
        assert stale.value.status_code == 412
        with pytest.raises(HTTPException) as missing:
            asyncio.run(
                app.read_command_stdout(
                    "missing", {"owner_claim": claim, "offset": 0, "max_bytes": 1}
                )
            )
        assert missing.value.status_code == 404
    finally:
        app.SUPERVISOR.cancel_all()
        sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)


def test_exited_command_backpressure_obeys_its_timeout(tmp_path):
    supervisor = CommandSupervisor(root=tmp_path, cancel_timeout_s=1)
    supervisor.start(
        "backpressure",
        {
            "kind": "process",
            "argv": [
                sys.executable,
                "-c",
                "import os; os.write(1, b'x' * (8 * 1024 * 1024 + 4096))",
            ],
            "cwd": str(tmp_path),
            "interactive": True,
            "timeout_s": 0.5,
        },
    )
    record = supervisor._records["backpressure"]
    try:
        assert record.finished.wait(10)
        assert supervisor.get("backpressure")["timed_out"] is True
        assert record.interactive.abandoned
        assert record.finished_at is not None
    finally:
        supervisor.cancel("backpressure")


def test_cancel_refreshes_unconfirmed_terminal_process_evidence(tmp_path, monkeypatch):
    supervisor = CommandSupervisor(root=tmp_path)
    _start(supervisor, "retry-stop", tmp_path, "pass")
    assert _terminal(supervisor, "retry-stop")["process_tree_stopped"] is False
    assert supervisor.cancel("retry-stop")["process_tree_stopped"] is False
    # The namespace owner may settle after the first bounded observation.
    monkeypatch.setattr(sidecar, "_process_tree_stopped", lambda *args, **kwargs: True)
    assert supervisor.cancel("retry-stop")["process_tree_stopped"] is True
