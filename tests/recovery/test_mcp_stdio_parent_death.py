from __future__ import annotations

import asyncio
import fcntl
import gc
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

import cayu.mcp.stdio as stdio_module
from cayu import (
    McpProtocolError,
    McpServerSpec,
    ResolvedSecret,
    SecretRef,
    StdioMcpClient,
)
from cayu.mcp._stdio_containment import _rendezvous_address
from cayu.mcp._stdio_process import (
    ContainedStdioMcpProcess,
    _create_sealed_server_environment_fd,
    create_contained_stdio_mcp_process,
    preflight_stdio_mcp_parent_death_containment,
    stdio_mcp_parent_death_containment_platform_candidate,
)

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        not stdio_mcp_parent_death_containment_platform_candidate()
        or not hasattr(signal, "SIGKILL"),
        reason="stdio MCP parent-death containment requires supported Linux enforcement",
    ),
]

_OWNER = Path(__file__).parents[1] / "fixtures" / "stdio_mcp_containment_owner.py"
_TREE = Path(__file__).parents[1] / "fixtures" / "fake_mcp_process_tree.py"
_FAKE_SERVER = Path(__file__).parents[1] / "fixtures" / "fake_mcp_server.py"
_ROOT = Path(__file__).parents[2]


def _wait_for_json(path: Path, *, timeout_s: float = 8.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        else:
            if type(value) is dict:
                return value
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path.name}")


def _wait_for_text(path: Path, expected: str, *, timeout_s: float = 8.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if path.read_text(encoding="utf-8") == expected:
                return
        except FileNotFoundError:
            pass
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path.name}")


def _pid_exists(pid: int) -> bool:
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            stat = proc_stat.read_text(encoding="ascii")
            suffix = stat[stat.rfind(")") + 2 :].split()
        except (FileNotFoundError, PermissionError, OSError):
            return False
        if suffix and suffix[0] == "Z":
            # Minimal containers may run a non-reaping PID 1. A zombie has no
            # executable process or retained resource and is quiescent for the
            # containment contract even until that external parent reaps it.
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _evidence_pids(evidence: dict[str, object]) -> set[int]:
    return {
        value for name, value in evidence.items() if name.endswith("_pid") and type(value) is int
    }


def _wait_for_processes_gone(pids: set[int], *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not any(_pid_exists(pid) for pid in pids):
            return
        time.sleep(0.02)
    remaining = sorted(pid for pid in pids if _pid_exists(pid))
    raise AssertionError(f"contained MCP processes remained alive: {remaining}")


def _linux_child_pids(parent_pid: int) -> set[int]:
    children_path = Path(f"/proc/{parent_pid}/task/{parent_pid}/children")
    try:
        children = children_path.read_text(encoding="ascii").split()
    except FileNotFoundError:
        return set()
    return {int(value) for value in children}


def _linux_process_command(pid: int) -> tuple[str, ...]:
    try:
        payload = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return ()
    return tuple(part.decode("utf-8") for part in payload.split(b"\0") if part)


def test_server_loader_environment_never_reaches_trusted_containment_helpers(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the loader-boundary regression")
    source = tmp_path / "loader_probe.c"
    library = tmp_path / "loader_probe.so"
    marker = tmp_path / "loader_probe.log"
    source.write_text(
        r"""
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

__attribute__((constructor)) static void record_loaded_process(void) {
    const char *marker = getenv("CAYU_MCP_PRELOAD_MARKER");
    if (marker == NULL) return;
    char command[4096];
    int command_fd = open("/proc/self/cmdline", O_RDONLY);
    ssize_t length = command_fd < 0 ? -1 : read(command_fd, command, sizeof(command) - 1);
    if (command_fd >= 0) close(command_fd);
    if (length < 0) length = 0;
    for (ssize_t index = 0; index < length; ++index) {
        if (command[index] == '\0') command[index] = ' ';
    }
    command[length] = '\0';
    const char *label = strstr(command, "_stdio_containment.py") == NULL
        ? "server\n"
        : "helper\n";
    int marker_fd = open(marker, O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (marker_fd >= 0) {
        write(marker_fd, label, strlen(label));
        close(marker_fd);
    }
}
""".strip(),
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-shared", "-fPIC", "-o", str(library), str(source)],
        check=True,
        capture_output=True,
    )

    async def run() -> None:
        session = await StdioMcpClient().connect(
            McpServerSpec(
                name="loader-boundary",
                command=[sys.executable, str(_FAKE_SERVER)],
                env={
                    "CAYU_MCP_PRELOAD_MARKER": str(marker),
                    "LD_PRELOAD": str(library),
                },
            )
        )
        try:
            await session.list_tools()
        finally:
            await session.close()

    asyncio.run(run())

    loaded_processes = marker.read_text(encoding="utf-8").splitlines()
    assert loaded_processes
    assert set(loaded_processes) == {"server"}


def _launch_owner(
    tmp_path: Path,
    *,
    generation: str,
    server_role: str,
    lock_path: Path,
    process_lifetime: str = "parent_death_containment",
    close_standard_fds: bool = False,
    high_descriptors: bool = False,
    graceful_close_trigger_path: Path | None = None,
    eof_marker_path: Path | None = None,
    eof_release_path: Path | None = None,
    close_inherited_protocol_stdin: bool = False,
    attack_rendezvous: bool = False,
    connection_id: str | None = None,
    server_state_path: Path | None = None,
    startup_timeout_s: float = 3.0,
    term_timeout_s: float = 0.2,
    kill_timeout_s: float = 1.0,
) -> tuple[subprocess.Popen[bytes], Path, Path]:
    owner_state = tmp_path / f"owner-{generation}.json"
    server_state = (
        tmp_path / f"server-{generation}.json" if server_state_path is None else server_state_path
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT / "src")
    process = subprocess.Popen(
        [
            sys.executable,
            str(_OWNER),
            "--owner-state-path",
            str(owner_state),
            "--server-state-path",
            str(server_state),
            "--lock-path",
            str(lock_path),
            "--server-role",
            server_role,
            "--process-lifetime",
            process_lifetime,
            "--startup-timeout-s",
            str(startup_timeout_s),
            "--term-timeout-s",
            str(term_timeout_s),
            "--kill-timeout-s",
            str(kill_timeout_s),
            *(["--attack-rendezvous"] if attack_rendezvous else []),
            *(["--connection-id", connection_id] if connection_id is not None else []),
            *(["--close-standard-fds"] if close_standard_fds else []),
            *(["--high-descriptors"] if high_descriptors else []),
            *(
                ["--graceful-close-trigger-path", str(graceful_close_trigger_path)]
                if graceful_close_trigger_path is not None
                else []
            ),
            *(["--eof-marker-path", str(eof_marker_path)] if eof_marker_path is not None else []),
            *(
                ["--eof-release-path", str(eof_release_path)]
                if eof_release_path is not None
                else []
            ),
            *(["--close-inherited-protocol-stdin"] if close_inherited_protocol_stdin else []),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        close_fds=True,
    )
    return process, owner_state, server_state


@pytest.mark.parametrize("descriptor_mode", ["closed_standard", "high_numbered"])
def test_parent_death_observation_handles_real_descriptor_boundaries(
    tmp_path: Path,
    descriptor_mode: str,
) -> None:
    lock_path = tmp_path / f"{descriptor_mode}.lock"
    owner, owner_state_path, server_state_path = _launch_owner(
        tmp_path,
        generation=descriptor_mode,
        server_role="server",
        lock_path=lock_path,
        close_standard_fds=descriptor_mode == "closed_standard",
        high_descriptors=descriptor_mode == "high_numbered",
    )
    evidence: dict[str, object] = {}
    server_evidence: dict[str, object] = {}
    try:
        evidence = _wait_for_json(owner_state_path)
        server_evidence = _wait_for_json(server_state_path)
        control_fd = evidence["owner_control_fd"]
        liveness_fd = evidence["owner_liveness_fd"]
        assert type(control_fd) is int
        assert type(liveness_fd) is int
        minimum = 1024 if descriptor_mode == "high_numbered" else 3
        assert control_fd >= minimum
        assert liveness_fd >= minimum

        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout=3)
        inherited_writer_pid = evidence["inherited_writer_pid"]
        assert type(inherited_writer_pid) is int
        process_ids = _evidence_pids(evidence) | _evidence_pids(server_evidence)
        _wait_for_processes_gone(process_ids - {owner.pid, inherited_writer_pid})
    finally:
        if owner.poll() is None:
            with suppress(ProcessLookupError):
                os.kill(owner.pid, signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                owner.wait(timeout=2)
        for pid in _evidence_pids(evidence) | _evidence_pids(server_evidence):
            if pid != os.getpid():
                with suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)


def test_graceful_close_settles_descendants_left_by_the_stdio_server(tmp_path: Path) -> None:
    lock_path = tmp_path / "graceful-resource.lock"
    server_state_path = tmp_path / "graceful-server.json"
    eof_marker_path = tmp_path / "graceful-eof.txt"

    async def run() -> set[int]:
        session = await StdioMcpClient(
            containment_term_timeout_s=0.2,
            containment_kill_timeout_s=1.0,
        ).connect(
            McpServerSpec(
                name="graceful-tree",
                command=[
                    sys.executable,
                    str(_TREE),
                    "--role",
                    "server",
                    "--lock-path",
                    str(lock_path),
                    "--state-path",
                    str(server_state_path),
                    "--eof-marker-path",
                    str(eof_marker_path),
                ],
            )
        )
        evidence = _wait_for_json(server_state_path)
        process = session.process
        assert isinstance(process, ContainedStdioMcpProcess)
        pids = _evidence_pids(evidence)
        pids.update(
            pid
            for pid in (
                process.pid,
                process.anchor_pid,
                process.server_pid,
            )
            if pid is not None
        )
        await session.close()
        return pids

    process_ids = asyncio.run(run())
    assert eof_marker_path.read_text(encoding="utf-8") == "graceful"
    _wait_for_processes_gone(process_ids)
    probe = subprocess.run(
        [
            sys.executable,
            str(_TREE),
            "--role",
            "probe",
            "--lock-path",
            str(lock_path),
        ],
        check=False,
        timeout=3,
    )
    assert probe.returncode == 0


def test_graceful_close_racing_owner_death_converges_on_complete_tree_cleanup(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "close-owner-death-race.lock"
    close_trigger_path = tmp_path / "start-graceful-close"
    eof_marker_path = tmp_path / "server-observed-eof"
    eof_release_path = tmp_path / "release-server-after-eof"
    sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    owner, owner_state_path, server_state_path = _launch_owner(
        tmp_path,
        generation="close-owner-death-race",
        server_role="server",
        lock_path=lock_path,
        graceful_close_trigger_path=close_trigger_path,
        eof_marker_path=eof_marker_path,
        eof_release_path=eof_release_path,
        close_inherited_protocol_stdin=True,
    )
    owner_evidence: dict[str, object] = {}
    server_evidence: dict[str, object] = {}
    try:
        owner_evidence = _wait_for_json(owner_state_path)
        server_evidence = _wait_for_json(server_state_path)
        process_ids = _evidence_pids(owner_evidence) | _evidence_pids(server_evidence)
        inherited_writer_pid = owner_evidence["inherited_writer_pid"]
        assert type(inherited_writer_pid) is int

        close_trigger_path.write_text("close", encoding="utf-8")
        _wait_for_text(eof_marker_path, "graceful")
        assert not eof_release_path.exists()
        assert owner.poll() is None

        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout=3)
        _wait_for_processes_gone(process_ids - {owner.pid, inherited_writer_pid})

        assert _pid_exists(inherited_writer_pid)
        assert sibling.poll() is None
        probe = subprocess.run(
            [
                sys.executable,
                str(_TREE),
                "--role",
                "probe",
                "--lock-path",
                str(lock_path),
            ],
            check=False,
            timeout=3,
        )
        assert probe.returncode == 0
    finally:
        if owner.poll() is None:
            with suppress(ProcessLookupError):
                os.kill(owner.pid, signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                owner.wait(timeout=2)
        for pid in _evidence_pids(owner_evidence) | _evidence_pids(server_evidence):
            if pid != os.getpid():
                with suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
        sibling.terminate()
        with suppress(subprocess.TimeoutExpired):
            sibling.wait(timeout=2)


def test_containment_term_reaches_a_server_with_default_signal_disposition() -> None:
    async def run() -> tuple[float, int, int]:
        process = await create_contained_stdio_mcp_process(
            "/bin/sleep",
            "30",
            env=dict(os.environ),
            limit=65_536,
            startup_timeout_s=2.0,
            term_timeout_s=1.0,
            kill_timeout_s=1.0,
        )
        assert process.server_pid is not None
        server_pid = process.server_pid
        started_at = time.monotonic()
        process.terminate()
        returncode = await process.wait()
        return time.monotonic() - started_at, returncode, server_pid

    elapsed, returncode, server_pid = asyncio.run(run())
    assert elapsed < 0.8
    assert returncode == -signal.SIGTERM
    _wait_for_processes_gone({server_pid})


def test_supervisor_exit_does_not_publish_false_tree_settlement(tmp_path: Path) -> None:
    lock_path = tmp_path / "supervisor-exit-resource.lock"
    server_state_path = tmp_path / "supervisor-exit-server.json"

    async def run() -> set[int]:
        session = await StdioMcpClient(
            containment_term_timeout_s=0.2,
            containment_kill_timeout_s=1.0,
        ).connect(
            McpServerSpec(
                name="supervisor-exit-tree",
                command=[
                    sys.executable,
                    str(_TREE),
                    "--role",
                    "server",
                    "--lock-path",
                    str(lock_path),
                    "--state-path",
                    str(server_state_path),
                ],
            )
        )
        evidence = _wait_for_json(server_state_path)
        process = session.process
        assert isinstance(process, ContainedStdioMcpProcess)
        pids = _evidence_pids(evidence)
        pids.update(pid for pid in (process.anchor_pid, process.server_pid) if pid is not None)
        os.kill(process.pid, signal.SIGKILL)
        while process.returncode is None:
            await asyncio.sleep(0.005)
        with pytest.raises(McpProtocolError, match="without settlement"):
            await session.close()
        return pids

    process_ids = asyncio.run(run())
    _wait_for_processes_gone(process_ids)


def test_replacement_waits_for_anchor_settlement_after_supervisor_exit(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "supervisor-replacement-resource.lock"
    server_state_path = tmp_path / "supervisor-replacement-server.json"
    server = McpServerSpec(
        name="supervisor-replacement-tree",
        connection_id="supervisor-replacement-tree",
        command=[
            sys.executable,
            str(_TREE),
            "--role",
            "server",
            "--lock-path",
            str(lock_path),
            "--state-path",
            str(server_state_path),
        ],
    )

    async def run() -> set[int]:
        client = StdioMcpClient(
            containment_startup_timeout_s=5.0,
            containment_term_timeout_s=2.0,
            containment_kill_timeout_s=1.0,
        )
        first = await client.connect(server)
        evidence = _wait_for_json(server_state_path)
        first_process = first.process
        assert isinstance(first_process, ContainedStdioMcpProcess)
        old_pids = _evidence_pids(evidence)
        old_pids.update(
            pid for pid in (first_process.anchor_pid, first_process.server_pid) if pid is not None
        )
        os.kill(first_process.pid, signal.SIGKILL)
        server_state_path.unlink()
        server_state_path.with_suffix(f"{server_state_path.suffix}.grandchild").unlink()

        replacement_task = asyncio.create_task(client.connect(server))
        try:
            await asyncio.sleep(0.25)
            assert not replacement_task.done()
            assert any(_pid_exists(pid) for pid in old_pids)
            replacement = await asyncio.wait_for(replacement_task, timeout=5.0)
            try:
                _wait_for_processes_gone(old_pids)
                await replacement.list_tools()
            finally:
                await replacement.close()
        finally:
            if not replacement_task.done():
                replacement_task.cancel()
                await asyncio.gather(replacement_task, return_exceptions=True)
            with pytest.raises(McpProtocolError, match="without settlement"):
                await first.close()
        return old_pids

    old_process_ids = asyncio.run(run())
    _wait_for_processes_gone(old_process_ids)


def test_resolver_fork_cannot_retain_the_containment_rendezvous() -> None:
    release_read_fd, release_write_fd = os.pipe()

    class ForkingSecretResolver:
        child_pid: int | None = None

        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict[str, Any] | None = None,
        ) -> ResolvedSecret:
            del scope
            child_pid = os.fork()
            if child_pid == 0:
                os.close(release_write_fd)
                try:
                    os.read(release_read_fd, 1)
                finally:
                    os._exit(0)
            self.child_pid = child_pid
            os.close(release_read_fd)
            return ResolvedSecret(name=ref.name, value=SecretStr("forked-resolver-secret"))

    resolver = ForkingSecretResolver()
    command = [sys.executable, str(_FAKE_SERVER)]
    server = McpServerSpec(
        name="forking-resolver-rendezvous",
        connection_id="forking-resolver-rendezvous",
        command=command,
        secret_env={"TOKEN": SecretRef(name="token")},
    )

    async def run() -> None:
        first = await StdioMcpClient(
            secret_resolver=resolver,
            containment_startup_timeout_s=3.0,
        ).connect(server)
        try:
            await first.list_tools()
        finally:
            await first.close()

        assert resolver.child_pid is not None
        assert _pid_exists(resolver.child_pid)
        replacement = await asyncio.wait_for(
            StdioMcpClient(containment_startup_timeout_s=1.0).connect(
                server.model_copy(update={"secret_env": {}})
            ),
            timeout=3.0,
        )
        try:
            assert _pid_exists(resolver.child_pid)
            await replacement.list_tools()
        finally:
            await replacement.close()

    try:
        asyncio.run(run())
    finally:
        with suppress(OSError):
            os.close(release_read_fd)
        with suppress(OSError):
            os.write(release_write_fd, b"x")
        with suppress(OSError):
            os.close(release_write_fd)
        if resolver.child_pid is not None:
            with suppress(ChildProcessError):
                os.waitpid(resolver.child_pid, 0)


def test_supervisor_takes_over_cleanup_after_anchor_is_killed(tmp_path: Path) -> None:
    lock_path = tmp_path / "anchor-crash-resource.lock"
    server_state_path = tmp_path / "anchor-crash-server.json"

    async def run() -> tuple[int, set[int]]:
        process = await create_contained_stdio_mcp_process(
            sys.executable,
            str(_TREE),
            "--role",
            "server",
            "--lock-path",
            str(lock_path),
            "--state-path",
            str(server_state_path),
            "--churn-descendants",
            env=dict(os.environ),
            limit=65_536,
            startup_timeout_s=2.0,
            term_timeout_s=0.2,
            kill_timeout_s=1.0,
        )
        evidence = _wait_for_json(server_state_path)
        assert evidence["grandchild_churns_descendants"] is True
        assert process.anchor_pid is not None
        assert process.server_pid is not None
        pids = _evidence_pids(evidence) | {process.anchor_pid, process.server_pid}
        os.kill(process.anchor_pid, signal.SIGKILL)
        return await process.wait(), pids

    returncode, process_ids = asyncio.run(run())

    assert returncode == 0
    _wait_for_processes_gone(process_ids)
    probe = subprocess.run(
        [
            sys.executable,
            str(_TREE),
            "--role",
            "probe",
            "--lock-path",
            str(lock_path),
        ],
        check=False,
        timeout=3,
    )
    assert probe.returncode == 0


def test_anchor_settles_dispatched_tree_when_ready_publication_loses_its_reader(
    tmp_path: Path,
) -> None:
    """Exercise external success followed by loss of the supervisor receipt pipe."""

    lock_path = tmp_path / "ready-publication-loss.lock"
    server_state_path = tmp_path / "ready-publication-loss.json"
    owner_read_fd, owner_write_fd = os.pipe()
    event_read_fd, event_write_fd = os.pipe()
    environment_fd = _create_sealed_server_environment_fd(dict(os.environ))
    rendezvous_identity = "a" * 64
    rendezvous = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    rendezvous.bind(_rendezvous_address(rendezvous_identity))
    event_flags = fcntl.fcntl(event_write_fd, fcntl.F_GETFL)
    fcntl.fcntl(event_write_fd, fcntl.F_SETFL, event_flags | os.O_NONBLOCK)
    try:
        while True:
            os.write(event_write_fd, b"x" * 4096)
    except BlockingIOError:
        pass
    fcntl.fcntl(event_write_fd, fcntl.F_SETFL, event_flags)

    helper = Path(__file__).parents[2] / "src" / "cayu" / "mcp" / "_stdio_containment.py"
    anchor = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            str(helper),
            "--role",
            "anchor",
            "--nonce",
            "ready-publication-loss",
            "--expected-parent-pid",
            str(os.getpid()),
            "--owner-fd",
            str(owner_read_fd),
            "--anchor-event-fd",
            str(event_write_fd),
            "--server-env-fd",
            str(environment_fd),
            "--rendezvous-fd",
            str(rendezvous.fileno()),
            "--rendezvous-identity",
            rendezvous_identity,
            "--term-timeout-s",
            "0.2",
            "--kill-timeout-s",
            "1.0",
            "--",
            sys.executable,
            str(_TREE),
            "--role",
            "server",
            "--lock-path",
            str(lock_path),
            "--state-path",
            str(server_state_path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={},
        pass_fds=(
            owner_read_fd,
            event_write_fd,
            environment_fd,
            rendezvous.fileno(),
        ),
        start_new_session=True,
    )
    rendezvous.close()
    os.close(owner_read_fd)
    os.close(event_write_fd)
    os.close(environment_fd)
    evidence: dict[str, object] = {}
    try:
        evidence = _wait_for_json(server_state_path)
        assert anchor.poll() is None
        os.close(event_read_fd)
        event_read_fd = -1
        anchor.wait(timeout=5)
        _wait_for_processes_gone(_evidence_pids(evidence) | {anchor.pid})
        probe = subprocess.run(
            [
                sys.executable,
                str(_TREE),
                "--role",
                "probe",
                "--lock-path",
                str(lock_path),
            ],
            check=False,
            timeout=3,
        )
        assert probe.returncode == 0
    finally:
        os.close(owner_write_fd)
        if event_read_fd >= 0:
            os.close(event_read_fd)
        if anchor.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(anchor.pid, signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                anchor.wait(timeout=2)
        for pid in _evidence_pids(evidence):
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize("exit_code", [73, 200])
def test_server_exit_code_is_preserved_without_becoming_cleanup_failure(exit_code: int) -> None:
    async def run() -> int:
        process = await create_contained_stdio_mcp_process(
            sys.executable,
            "-c",
            f"raise SystemExit({exit_code})",
            env=dict(os.environ),
            limit=65_536,
            startup_timeout_s=2.0,
            term_timeout_s=0.2,
            kill_timeout_s=1.0,
        )
        return await process.wait()

    assert asyncio.run(run()) == exit_code


def test_server_signal_exit_is_preserved() -> None:
    async def run() -> int:
        process = await create_contained_stdio_mcp_process(
            sys.executable,
            "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGKILL)",
            env=dict(os.environ),
            limit=65_536,
            startup_timeout_s=2.0,
            term_timeout_s=0.2,
            kill_timeout_s=1.0,
        )
        return await process.wait()

    assert asyncio.run(run()) == -signal.SIGKILL


def test_contained_descendants_cannot_create_a_detached_session(tmp_path: Path) -> None:
    lock_path = tmp_path / "detach-resource.lock"
    server_state_path = tmp_path / "detach-server.json"

    async def run() -> tuple[dict[str, object], set[int]]:
        session = await StdioMcpClient(
            containment_term_timeout_s=0.2,
            containment_kill_timeout_s=1.0,
        ).connect(
            McpServerSpec(
                name="detachment-resistant-tree",
                command=[
                    sys.executable,
                    str(_TREE),
                    "--role",
                    "server",
                    "--lock-path",
                    str(lock_path),
                    "--state-path",
                    str(server_state_path),
                    "--detach-grandchild",
                ],
            )
        )
        evidence = json.loads(server_state_path.read_text(encoding="utf-8"))
        pids = _evidence_pids(evidence)
        await session.close()
        return evidence, pids

    evidence, process_ids = asyncio.run(run())
    assert evidence["grandchild_detachment_denied"] is True
    _wait_for_processes_gone(process_ids)
    probe = subprocess.run(
        [
            sys.executable,
            str(_TREE),
            "--role",
            "probe",
            "--lock-path",
            str(lock_path),
        ],
        check=False,
        timeout=3,
    )
    assert probe.returncode == 0


def test_contained_server_cannot_terminate_its_cleanup_parent(tmp_path: Path) -> None:
    lock_path = tmp_path / "parent-signal-resource.lock"
    server_state_path = tmp_path / "parent-signal-server.json"

    async def run() -> tuple[dict[str, object], set[int]]:
        session = await StdioMcpClient(
            containment_term_timeout_s=0.2,
            containment_kill_timeout_s=1.0,
        ).connect(
            McpServerSpec(
                name="parent-signal-resistant-tree",
                command=[
                    sys.executable,
                    str(_TREE),
                    "--role",
                    "server",
                    "--lock-path",
                    str(lock_path),
                    "--state-path",
                    str(server_state_path),
                    "--attack-parent",
                ],
            )
        )
        evidence = _wait_for_json(server_state_path)
        pids = _evidence_pids(evidence)
        await session.close()
        return evidence, pids

    evidence, process_ids = asyncio.run(run())

    assert evidence["server_capabilities_dropped"] is True
    assert evidence["anchor_memory_denied"] is True
    assert evidence["anchor_prlimit_denied"] is True
    assert evidence["anchor_signal_denied"] is True
    assert evidence["supervisor_memory_denied"] is True
    assert evidence["supervisor_signal_denied"] is True
    _wait_for_processes_gone(process_ids)


def test_persistent_detached_is_an_explicit_parent_death_opt_out(tmp_path: Path) -> None:
    lock_path = tmp_path / "persistent-resource.lock"
    owner, owner_state_path, server_state_path = _launch_owner(
        tmp_path,
        generation="persistent",
        server_role="server",
        lock_path=lock_path,
        process_lifetime="persistent_detached",
    )
    owner_evidence: dict[str, object] = {}
    server_evidence: dict[str, object] = {}
    try:
        owner_evidence = _wait_for_json(owner_state_path)
        server_evidence = _wait_for_json(server_state_path)
        assert "direct_process_pid" in owner_evidence
        assert owner_evidence["parent_death_containment"] == "unsupported"
        assert owner_evidence["persistent_detached"] == "available"
        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout=3)

        grandchild_pid = server_evidence["grandchild_pid"]
        assert type(grandchild_pid) is int
        assert _pid_exists(grandchild_pid)
        probe = subprocess.run(
            [
                sys.executable,
                str(_TREE),
                "--role",
                "probe",
                "--lock-path",
                str(lock_path),
            ],
            check=False,
            timeout=3,
        )
        assert probe.returncode == 25
    finally:
        if owner.poll() is None:
            with suppress(ProcessLookupError):
                os.kill(owner.pid, signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                owner.wait(timeout=2)
        server_pgid = server_evidence.get("server_pgid")
        if type(server_pgid) is int:
            with suppress(ProcessLookupError):
                os.killpg(server_pgid, signal.SIGKILL)
        inherited_writer_pid = owner_evidence.get("inherited_writer_pid")
        if type(inherited_writer_pid) is int:
            with suppress(ProcessLookupError):
                os.kill(inherited_writer_pid, signal.SIGKILL)


@pytest.mark.parametrize("server_role", ["server", "launcher"])
def test_sigkill_owner_contains_complete_stdio_mcp_tree_and_allows_replacement(
    tmp_path: Path,
    server_role: str,
) -> None:
    lock_path = tmp_path / "exclusive-resource.lock"
    shared_server_state = tmp_path / "replacement-server.json"
    connection_id = "contained-replacement-connector"
    sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    owners: list[subprocess.Popen[bytes]] = []
    known_pids: set[int] = set()
    try:
        first, first_owner_state, first_server_state = _launch_owner(
            tmp_path,
            generation="one",
            server_role=server_role,
            lock_path=lock_path,
            attack_rendezvous=True,
            connection_id=connection_id,
            server_state_path=shared_server_state,
            startup_timeout_s=5.0,
            term_timeout_s=2.0,
        )
        owners.append(first)
        owner_evidence = _wait_for_json(first_owner_state)
        server_evidence = _wait_for_json(first_server_state)
        assert server_evidence["server_rendezvous_rebind_denied"] is True
        known_pids.update(_evidence_pids(owner_evidence))
        known_pids.update(_evidence_pids(server_evidence))
        assert all(_pid_exists(pid) for pid in known_pids)
        inherited_writer_pid = owner_evidence["inherited_writer_pid"]
        assert type(inherited_writer_pid) is int

        os.kill(first.pid, signal.SIGKILL)
        first.wait(timeout=3)
        first_tree = known_pids - {first.pid, inherited_writer_pid}
        assert any(_pid_exists(pid) for pid in first_tree)
        assert _pid_exists(inherited_writer_pid)
        assert sibling.poll() is None
        shared_server_state.unlink()
        shared_server_state.with_suffix(f"{shared_server_state.suffix}.grandchild").unlink()

        second, second_owner_state, second_server_state = _launch_owner(
            tmp_path,
            generation="two",
            server_role=server_role,
            lock_path=lock_path,
            attack_rendezvous=True,
            connection_id=connection_id,
            server_state_path=shared_server_state,
            startup_timeout_s=5.0,
            term_timeout_s=2.0,
        )
        owners.append(second)
        time.sleep(0.25)
        assert second.poll() is None
        assert any(_pid_exists(pid) for pid in first_tree)
        second_owner = _wait_for_json(second_owner_state)
        second_server = _wait_for_json(second_server_state)
        assert second_server["server_rendezvous_rebind_denied"] is True
        _wait_for_processes_gone(first_tree)
        second_pids = _evidence_pids(second_owner) | _evidence_pids(second_server)
        known_pids.update(second_pids)
        assert all(_pid_exists(pid) for pid in second_pids)
        second_inherited_writer = second_owner["inherited_writer_pid"]
        assert type(second_inherited_writer) is int
        os.kill(second.pid, signal.SIGKILL)
        second.wait(timeout=3)
        _wait_for_processes_gone(second_pids - {second.pid, second_inherited_writer})
        assert _pid_exists(second_inherited_writer)
        assert sibling.poll() is None
    finally:
        for owner in owners:
            if owner.poll() is None:
                with suppress(ProcessLookupError):
                    os.kill(owner.pid, signal.SIGKILL)
                with suppress(subprocess.TimeoutExpired):
                    owner.wait(timeout=2)
        for pid in known_pids:
            if pid != os.getpid():
                with suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
        sibling.terminate()
        with suppress(subprocess.TimeoutExpired):
            sibling.wait(timeout=2)


def test_cancelling_a_waiting_replacement_never_dispatches_its_server() -> None:
    async def run() -> None:
        client = StdioMcpClient(
            containment_startup_timeout_s=5.0,
            containment_term_timeout_s=0.2,
            containment_kill_timeout_s=1.0,
        )
        server = McpServerSpec(
            name="replacement-cancellation",
            connection_id="replacement-cancellation",
            command=[sys.executable, str(_FAKE_SERVER)],
        )
        first = await client.connect(server)
        second_task: asyncio.Task[Any] | None = None
        try:
            assert isinstance(first.process, ContainedStdioMcpProcess)
            first_supervisor_pid = first.process.pid
            second_task = asyncio.create_task(client.connect(server))
            deadline = time.monotonic() + 3.0
            second_supervisor_pid = 0
            while time.monotonic() < deadline:
                for candidate in _linux_child_pids(os.getpid()) - {first_supervisor_pid}:
                    command = _linux_process_command(candidate)
                    if "--role" in command and "supervisor" in command:
                        second_supervisor_pid = candidate
                        break
                if second_supervisor_pid:
                    break
                await asyncio.sleep(0.01)
            assert second_supervisor_pid > 0
            assert not _linux_child_pids(second_supervisor_pid)

            second_task.cancel()
            assert second_task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await second_task
            assert second_task.cancelled()
            _wait_for_processes_gone({second_supervisor_pid})

            await first.list_tools()
        finally:
            if second_task is not None and not second_task.done():
                second_task.cancel()
                await asyncio.gather(second_task, return_exceptions=True)
            await first.close()

    asyncio.run(run())


def test_timed_out_replacement_settlement_consumes_late_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> list[dict[str, object]]:
        loop = asyncio.get_running_loop()
        diagnostics: list[dict[str, object]] = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
        server = McpServerSpec(
            name="replacement-timeout",
            connection_id="replacement-timeout",
            command=[sys.executable, str(_FAKE_SERVER)],
        )
        first = await StdioMcpClient(
            containment_startup_timeout_s=5.0,
            containment_term_timeout_s=0.2,
            containment_kill_timeout_s=1.0,
        ).connect(server)
        try:
            preflight_proof = await preflight_stdio_mcp_parent_death_containment(5.0)
            observed_preflight_timeouts: list[float] = []

            async def reuse_preflight_proof(timeout_s: float) -> Any:
                observed_preflight_timeouts.append(timeout_s)
                return preflight_proof

            monkeypatch.setattr(
                stdio_module,
                "preflight_stdio_mcp_parent_death_containment",
                reuse_preflight_proof,
            )
            with pytest.raises(
                McpProtocolError,
                match="containment supervisor did not become ready",
            ):
                await StdioMcpClient(
                    containment_startup_timeout_s=0.05,
                    containment_term_timeout_s=0.2,
                    containment_kill_timeout_s=1.0,
                ).connect(server)
            assert observed_preflight_timeouts == [0.05]
            gc.collect()
            await asyncio.sleep(0)
            assert not [
                context
                for context in diagnostics
                if context.get("message") == "Future exception was never retrieved"
            ]
            await first.list_tools()
            return diagnostics
        finally:
            await first.close()
            loop.set_exception_handler(previous_handler)

    assert asyncio.run(run()) == []


def test_distinct_connection_identities_do_not_share_replacement_ownership() -> None:
    async def run() -> None:
        client = StdioMcpClient(
            containment_startup_timeout_s=5.0,
            containment_term_timeout_s=0.2,
            containment_kill_timeout_s=1.0,
        )
        command = [sys.executable, str(_FAKE_SERVER)]
        first = await client.connect(
            McpServerSpec(
                name="independent-first",
                connection_id="independent-first",
                command=command,
            )
        )
        second = None
        try:
            second = await asyncio.wait_for(
                client.connect(
                    McpServerSpec(
                        name="independent-second",
                        connection_id="independent-second",
                        command=command,
                    )
                ),
                timeout=3.0,
            )
            await first.list_tools()
            await second.list_tools()
        finally:
            if second is not None:
                await second.close()
            await first.close()

    asyncio.run(run())
