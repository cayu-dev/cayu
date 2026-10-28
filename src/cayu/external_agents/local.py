"""Run a vendor CLI as an ordinary local process. This is not a sandbox."""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from tempfile import TemporaryDirectory

from cayu.external_agents._cleanup import close_external_guest
from cayu.external_agents.base import ExternalAgentLaunch
from cayu.external_agents.environment import _guest_secret_values, _validated_environment
from cayu.runners.local import _safe_host_env

# A live supervisor anchors each group until cleanup. Only that supervisor
# signals its own group; saved process numbers never authorize a host signal.
_STAND_IN = """\
import os, selectors, signal, socket, sys, threading
state, cli, blocked = {state!r}, {cli!r}, frozenset({blocked!r})
environment = {{key: value for key, value in os.environ.items() if key not in blocked}}
status_read, status_write = os.pipe()
control_read, control_write = os.pipe()
child = os.fork()
if child == 0:
    try:
        os.close(status_read)
        os.close(control_write)
        os.setsid()
        connection = socket.socket(socket.AF_UNIX)
        connection.connect(os.path.join(state, "control"))
        connection.sendall((str(os.getpid()) + "\\n").encode())
        if connection.recv(1) != b"R":
            os._exit(127)
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(number, signal.SIG_IGN)
        cli_child = os.fork()
        if cli_child == 0:
            connection.close()
            os.close(status_write)
            os.close(control_read)
            for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(number, signal.SIG_DFL)
            os.execve(cli, [cli, *sys.argv[1:]], environment)
        for descriptor in (0, 1, 2):
            os.close(descriptor)
        def wait_cli():
            _, status = os.waitpid(cli_child, 0)
            code = os.waitstatus_to_exitcode(status)
            os.write(status_write, str(code if code >= 0 else 128 - code).encode())
            os.close(status_write)
        threading.Thread(target=wait_cli, daemon=True).start()
        selector = selectors.DefaultSelector()
        selector.register(connection, selectors.EVENT_READ)
        selector.register(control_read, selectors.EVENT_READ)
        while True:
            for key, _ in selector.select():
                if key.fileobj is connection:
                    command = connection.recv(1)
                    if command == b"K":
                        connection.sendall(b"S")
                    os.killpg(os.getpgrp(), signal.SIGKILL)
                else:
                    command = os.read(control_read, 1)
                    if command:
                        os.killpg(os.getpgrp(), command[0])
                    else:
                        selector.unregister(control_read)
                        os.close(control_read)
    except BaseException:
        os._exit(127)
os.close(status_write)
os.close(control_read)
os.close(0)
os.close(1)
os.close(2)
def forward(number, frame):
    try:
        os.write(control_write, bytes([number]))
    except OSError:
        pass
for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(number, forward)
code = os.read(status_read, 32)
if code:
    os._exit(int(code))
_, status = os.waitpid(child, 0)
code = os.waitstatus_to_exitcode(status)
os._exit(code if code >= 0 else 128 - code)
"""


def _group_alive(group: int) -> bool:
    # PID 1 and subreapers adopt orphaned CLI descendants. Reap only children
    # in this launch's recorded group; zombies cannot be stopped with signals.
    while True:
        try:
            pid, _status = os.waitpid(-group, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if sys.platform == "linux":
        # A container's PID 1 may never reap orphaned zombies. They cannot
        # execute, hold files or receive signals, so do not report them as a
        # running guest. Any unavailable observation remains unconfirmed.
        try:
            for entry in Path("/proc").iterdir():
                if not entry.name.isdecimal():
                    continue
                try:
                    fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
                except FileNotFoundError:
                    continue
                # A leader may exit while its other threads still execute.
                # Only a single-thread zombie proves this process is stopped.
                if int(fields[2]) == group and (fields[0] not in {"Z", "X"} or int(fields[17]) > 1):
                    return True
            return False
        except (OSError, ValueError, IndexError):
            return True
    return True


async def _request_group_stop(writer: asyncio.StreamWriter) -> bool:
    writer.write(b"K")
    await writer.drain()
    return True


@asynccontextmanager
async def local_agent_launch(
    *,
    cli: str,
    cwd: str | os.PathLike[str],
    environment: Mapping[str, str] | None = None,
    close_timeout_s: float = 10.0,
) -> AsyncIterator[ExternalAgentLaunch]:
    """Run the vendor CLI on this host, as the current user. This is not a sandbox.

    Use it for trusted code, local development and tests. The CLI and the tools
    it runs can read and change anything the current user can, and reach any
    network the host can. They are not confined to ``cwd``.

    ``cli`` is the CLI's path, or a name found on ``PATH``. ``cwd`` is the
    checkout the CLI starts in. The CLI gets ``SAFE_LOCAL_ENV_KEYS`` from this
    process (``PATH``, ``HOME`` and locale settings), so a CLI that is already
    logged in keeps using its own login, plus ``environment``. No other host
    variable reaches it. Values whose names contain ``KEY``, ``TOKEN`` or
    ``AUTHORIZATION`` are scrubbed from what Cayu stores.

    Each CLI process starts in its own process group. ``close`` kills those
    groups and returns once they are empty, or raises after
    ``close_timeout_s`` so the run's cleanup is retried. A descendant that
    starts its own session leaves the group and is not stopped. POSIX only.

    See ``cayu guide external-agents``.
    """
    if os.name != "posix" or not hasattr(os, "fork"):
        raise RuntimeError("local_agent_launch needs a POSIX host.")
    if not close_timeout_s > 0:
        raise ValueError("close_timeout_s must be positive.")
    explicit = _validated_environment(environment or {})
    executable = shutil.which(cli) if not os.path.isabs(cli) else cli
    if executable is not None:
        executable = os.path.abspath(executable)
    if executable is None or not os.access(executable, os.X_OK):
        raise ValueError(f"The CLI {cli!r} was not found or is not executable.")
    root = Path(cwd).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("The CLI working directory must be a directory.")
    guest_environment = {**_safe_host_env(), **explicit}
    blocked = sorted(set(os.environ) - set(guest_environment))

    with TemporaryDirectory(prefix="cayu-local-agent-") as directory:
        state = Path(directory)
        groups = state / "groups"
        groups.mkdir()
        # Connections survive directory removal so failed cleanup can be retried.
        pending: dict[int, tuple[asyncio.StreamWriter, asyncio.Task[None]]] = {}
        handshakes: set[asyncio.StreamWriter] = set()
        admissions: set[asyncio.Task[None]] = set()
        closing = False
        stop_requested: set[int] = set()

        async def observe(reader: asyncio.StreamReader) -> None:
            with suppress(ConnectionError):
                await reader.read()

        async def admit(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                identity = await reader.readline()
                if closing:
                    return
                group = int(identity)
                if group <= 0 or group in pending:
                    return
                pending[group] = (writer, asyncio.create_task(observe(reader)))
                (groups / str(group)).touch()
                # Registration and positive admission have no intervening await.
                writer.write(b"R")
                handshakes.discard(writer)
            except (ValueError, OSError):
                pass
            finally:
                if writer in handshakes:
                    handshakes.discard(writer)
                    writer.close()

        def connected(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            if closing:
                writer.close()
                return
            handshakes.add(writer)
            task = asyncio.create_task(admit(reader, writer))
            admissions.add(task)
            task.add_done_callback(admissions.discard)

        server = await asyncio.start_unix_server(connected, path=str(state / "control"))

        async def close() -> None:
            nonlocal closing
            closing = True
            server.close()
            for writer in tuple(handshakes):
                writer.close()
            deadline = asyncio.get_running_loop().time() + close_timeout_s
            while pending:
                for group, (writer, observed) in tuple(pending.items()):
                    if not observed.done():
                        if group not in stop_requested:
                            with suppress(ConnectionError, OSError):
                                if await _request_group_stop(writer):
                                    stop_requested.add(group)
                    elif not _group_alive(group):
                        # EOF/reset plus an empty group proves cleanup even if
                        # the owner died before its acknowledgement arrived.
                        pending.pop(group)
                        stop_requested.discard(group)
                        writer.close()
                        (groups / str(group)).unlink(missing_ok=True)
                if not pending:
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError("Local CLI processes were not confirmed stopped.")
                await asyncio.sleep(0.02)
            await asyncio.wait_for(
                server.wait_closed(),
                max(0.001, deadline - asyncio.get_running_loop().time()),
            )

        stand_in = state / "cli"
        script = state / "cli.py"
        script.write_text(
            _STAND_IN.format(state=str(state), cli=str(executable), blocked=blocked),
            encoding="utf-8",
        )
        stand_in.write_text(
            f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -I {shlex.quote(str(script))} "$@"\n',
            encoding="utf-8",
        )
        stand_in.chmod(0o700)
        try:
            yield ExternalAgentLaunch(
                str(stand_in),
                str(root),
                close,
                guest_environment,
                tuple(value for value in _guest_secret_values(guest_environment) if value),
            )
        finally:
            await close_external_guest(close)
