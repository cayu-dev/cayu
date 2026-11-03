"""Run bulk workspace work outside the process whose event loop runs sessions.

Reading, hashing and packing thousands of files is mostly Python code. In a
thread it still takes turns with the event loop for the interpreter lock, so
one large copy into or out of a sandbox slows model streaming and tool calls
for every other session in the process. This module runs such work in a few
helper processes instead, which also bounds how many bulk transfers a process
runs at once: the rest wait for a free helper.

``CAYU_WORKSPACE_TRANSFER_PROCESSES`` sets how many helpers a process may
start (default 2); ``0`` runs the work in a thread of this process instead, as
it does where a process can't hand an open file to another (Windows). Helpers
start when first needed and are reused. Each is a fresh interpreter that
imports only what the work needs (never the program's ``__main__``), ignores
SIGINT and SIGTERM so a worker's graceful shutdown can finish the copies it
still runs, and exits when the process that started it does.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import pickle
import signal
import socket
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

TRANSFER_PROCESSES_ENV = "CAYU_WORKSPACE_TRANSFER_PROCESSES"
DEFAULT_TRANSFER_PROCESSES = 2
MAX_TRANSFER_PROCESSES = 64

_ResultT = TypeVar("_ResultT")
_LENGTH = struct.Struct("!Q")
# A request starts with how many descriptors follow it, so the helper takes them
# off the socket even when it can't unpickle the rest.
_DESCRIPTORS = struct.Struct("!I")
# The helper runs isolated (-I) with this process's import path, so it imports
# the same Cayu as its parent and nothing from its working directory.
_BOOTSTRAP = (
    "import json, sys; sys.path[:] = json.loads(sys.argv[1]); "
    "from cayu.workspaces._transfer_processes import _serve; _serve(int(sys.argv[2]))"
)
_HELPERS_SUPPORTED = hasattr(socket, "AF_UNIX") and hasattr(socket, "send_fds")


@dataclass(frozen=True, slots=True)
class OpenFile:
    """An argument that hands the work this open file descriptor.

    The work receives a descriptor number for the same open file, in whatever
    process it runs, and must not close it.
    """

    descriptor: int


def transfer_process_count() -> int:
    """How many helper processes may run bulk workspace work; 0 runs it in a thread."""

    value = os.environ.get(TRANSFER_PROCESSES_ENV, "").strip()
    if not value:
        return DEFAULT_TRANSFER_PROCESSES
    if not value.isdecimal() or int(value) > MAX_TRANSFER_PROCESSES:
        raise ValueError(
            f"{TRANSFER_PROCESSES_ENV} must be a whole number from 0 to {MAX_TRANSFER_PROCESSES}."
        )
    return int(value)


async def run_workspace_transfer(function: Callable[..., _ResultT], /, *args: object) -> _ResultT:
    """Return ``function(*args)``, run in a helper process or, when they're off, a thread.

    ``function`` must be a module-level function, and its arguments, result
    and exceptions must pickle; an :class:`OpenFile` argument reaches it as a
    descriptor number. It runs in a fresh interpreter, so it must not depend
    on this process's state. Cancelling the call stops waiting for it, but
    work that already started runs to its end in the helper, so a caller must
    not reuse a file it handed over until the work is done (``SyncBinding``
    waits for it).
    """

    if transfers_use_helpers():
        return await run_in_helper(function, *args)
    return await run_in_thread(function, *args)


def transfers_use_helpers() -> bool:
    """Whether :func:`run_workspace_transfer` would run work in a helper process now."""

    return transfer_process_count() > 0 and _HELPERS_SUPPORTED and bool(sys.executable)


async def run_in_helper(function: Callable[..., _ResultT], /, *args: object) -> _ResultT:
    """:func:`run_workspace_transfer` in a helper process, whatever the configuration says."""

    # Calls queue for one of the configured threads, each driving one helper,
    # so waiting transfers don't hold the loop's default threads.
    return await asyncio.get_running_loop().run_in_executor(
        _callers(max(1, transfer_process_count())), _call_in_helper, function, args
    )


async def run_in_thread(function: Callable[..., _ResultT], /, *args: object) -> _ResultT:
    """:func:`run_workspace_transfer` in a thread of this process."""

    local = tuple(arg.descriptor if isinstance(arg, OpenFile) else arg for arg in args)
    return await asyncio.to_thread(function, *local)


class _HelperGoneError(Exception):
    pass


class _Helper:
    def __init__(self) -> None:
        self._socket, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    _BOOTSTRAP,
                    json.dumps(sys.path),
                    str(theirs.fileno()),
                ],
                stdin=subprocess.DEVNULL,
                pass_fds=(theirs.fileno(),),
            )
        except BaseException:
            self._socket.close()
            raise
        finally:
            theirs.close()

    def call(self, request: bytes, descriptors: list[int]) -> bytes:
        """Send one pickled request and its descriptors; return the pickled answer."""

        try:
            _send_frame(self._socket, request)
            if descriptors:
                socket.send_fds(self._socket, [b"\0"], descriptors)
            return _receive_frame(self._socket)
        except (OSError, EOFError) as exc:
            raise _HelperGoneError from exc

    def close(self) -> None:
        # Closing its socket ends an idle helper; one still working is stopped.
        self._socket.close()
        try:
            self._process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()


_lock = threading.Lock()
_idle: list[_Helper] = []
_executor: ThreadPoolExecutor | None = None


def _callers(count: int) -> ThreadPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=count, thread_name_prefix="cayu-workspace-transfer"
            )
        return _executor


@dataclass(frozen=True, slots=True)
class _Descriptor:
    index: int


def _call_in_helper(function: Callable[..., _ResultT], args: tuple[object, ...]) -> _ResultT:
    descriptors: list[int] = []
    sent: list[object] = []
    for arg in args:
        if isinstance(arg, OpenFile):
            sent.append(_Descriptor(len(descriptors)))
            descriptors.append(arg.descriptor)
        else:
            sent.append(arg)
    # Pickle first: a request that doesn't pickle never reaches a helper.
    request = _DESCRIPTORS.pack(len(descriptors)) + pickle.dumps(
        (function, tuple(sent)), pickle.HIGHEST_PROTOCOL
    )
    with _lock:
        helper = _idle.pop() if _idle else None
    if helper is None:
        helper = _Helper()
    try:
        answer = helper.call(request, descriptors)
    except _HelperGoneError as exc:
        helper.close()
        raise RuntimeError(
            "A workspace transfer process exited before it finished, for example "
            f"because it ran out of memory. Set {TRANSFER_PROCESSES_ENV}=0 to run "
            "workspace transfers in this process."
        ) from exc.__cause__
    except BaseException:
        helper.close()
        raise
    # The answer was read whole, so the helper is ready for another request
    # whatever unpickling it raises.
    with _lock:
        _idle.append(helper)
    outcome, value = cast("tuple[str, Any]", pickle.loads(answer))
    if outcome == "error":
        raise cast("BaseException", value)
    return cast("_ResultT", value)


def _send_frame(connection: socket.socket, payload: bytes) -> None:
    connection.sendall(_LENGTH.pack(len(payload)) + payload)


def _receive_frame(connection: socket.socket) -> bytes:
    (length,) = _LENGTH.unpack(_receive_exactly(connection, _LENGTH.size))
    return _receive_exactly(connection, length)


def _receive_exactly(connection: socket.socket, size: int) -> bytes:
    # Never read past the frame: descriptors ride on the byte that follows it.
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(min(remaining, 1 << 20))
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _serve(descriptor: int) -> None:
    """A helper's loop: run each request from its parent and send back the answer."""

    # Its parent decides when to stop; its parent's exit closes the socket and ends it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    # What the work prints goes to stderr.
    os.dup2(2, 1)
    connection = socket.socket(fileno=descriptor)
    while True:
        try:
            request = _receive_frame(connection)
        except EOFError:
            return
        (count,) = _DESCRIPTORS.unpack_from(request)
        received: list[int] = []
        if count:
            _, received, _, _ = socket.recv_fds(connection, 1, count)
        try:
            if len(received) != count:
                raise RuntimeError("A workspace transfer lost an open file on its way.")
            function, args = pickle.loads(memoryview(request)[_DESCRIPTORS.size :])
            local = tuple(
                received[arg.index] if isinstance(arg, _Descriptor) else arg for arg in args
            )
            answer: tuple[str, object] = ("ok", function(*local))
        except Exception as exc:
            answer = ("error", exc)
        finally:
            for received_descriptor in received:
                os.close(received_descriptor)
        try:
            payload = pickle.dumps(answer, pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            # The result or the error didn't pickle; say what it was.
            payload = pickle.dumps(
                ("error", RuntimeError(f"Workspace transfer answer could not be returned: {exc!r}"))
            )
        _send_frame(connection, payload)


def _close_idle_helpers() -> None:
    with _lock:
        helpers = list(_idle)
        _idle.clear()
    for helper in helpers:
        helper.close()


def _forget_helpers_after_fork() -> None:
    # A forked child doesn't own its parent's helpers or their threads.
    global _lock, _executor
    _lock = threading.Lock()
    _idle.clear()
    _executor = None


atexit.register(_close_idle_helpers)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_helpers_after_fork)
