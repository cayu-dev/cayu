"""Linux guest command owner, independent of the worker/attached Docker client.

The launcher sends the private JSON launch through stdin, never argv or env.
The command receives /dev/null as stdin. This narrowly supports named checks;
interactive commands require a different transport and are deliberately refused.
Do not run this module directly until the runner has durably saved launch
authority and acquired the exact allocation. A terminal receipt is not authority
to redispatch, publish a workspace, release an allocation, or resume a model.
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

if __package__:
    from cayu.runners._docker_command_receipt import (
        MAX_OUTPUT_BYTES,
        SCHEMA,
        seal,
        validate_identity,
    )

MAX_LAUNCH_BYTES = 128 * 1024
CLEANUP_SECONDS = 2.0


def _children() -> tuple[int, ...]:
    # Subreaping brings orphaned grandchildren back under this owner. Success
    # requires ECHILD, not merely exit of the command's original process group.
    value = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children").read_text()
    return tuple(int(item) for item in value.split())


def _publish(directory: int, document: bytes) -> None:
    fd = os.open(
        "receipt.pending",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory,
    )
    with os.fdopen(fd, "wb") as stream:
        stream.write(document)
        stream.flush()
        os.fsync(stream.fileno())
    # Never replace an existing receipt. A failed/lost publication retains
    # the exclusive operation directory and cannot authorize another launch.
    os.link(
        "receipt.pending",
        "receipt.json",
        src_dir_fd=directory,
        dst_dir_fd=directory,
        follow_symlinks=False,
    )
    os.fsync(directory)


def _abort(process: subprocess.Popen[bytes]) -> None:
    """Best-effort bounded containment on supervisor failure, never a receipt."""

    deadline = time.monotonic() + CLEANUP_SECONDS
    while time.monotonic() < deadline:
        try:
            children = _children()
        except OSError:
            return
        for pid in children:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                break
            if pid == process.pid:
                process.returncode = os.waitstatus_to_exitcode(status)
        time.sleep(0.01)


def main() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    # Protect the private key before reading it, including against same-UID
    # command access to /proc/<supervisor>/fd and /proc/<supervisor>/mem.
    # Admitted guests must also drop CAP_SYS_PTRACE (and all other capabilities).
    if libc.prctl(4, 0, 0, 0, 0) != 0 or libc.prctl(36, 1, 0, 0, 0) != 0:
        return 125
    raw = sys.stdin.buffer.readline(MAX_LAUNCH_BYTES + 1)
    if len(raw) > MAX_LAUNCH_BYTES or not raw.endswith(b"\n"):
        return 125
    launch = json.loads(raw)
    del raw
    if type(launch) is not dict or set(launch) != {
        "identity",
        "key",
        "argv",
        "directory",
        "timeout_seconds",
        "output_limit",
        "request_sha256",
    }:
        return 125
    identity = validate_identity(launch["identity"])
    key = bytes.fromhex(launch["key"])
    if len(key) != 32:
        return 125
    argv, directory = launch["argv"], launch["directory"]
    timeout, limit = launch["timeout_seconds"], launch["output_limit"]
    request_sha256 = launch["request_sha256"]
    if (
        type(argv) is not list
        or not argv
        or any(type(item) is not str or "\0" in item for item in argv)
        or type(directory) is not str
        or not os.path.isabs(directory)
        or type(timeout) is not int
        or not 1 <= timeout <= 86_400
        or type(limit) is not int
        or not 0 <= limit <= MAX_OUTPUT_BYTES
        or type(request_sha256) is not str
        or len(request_sha256) != 64
        or any(char not in "0123456789abcdef" for char in request_sha256)
    ):
        return 125
    del launch
    os.makedirs(os.path.dirname(directory), mode=0o700, exist_ok=True)
    os.mkdir(directory, 0o700)  # Exclusive operation reservation before dispatch.
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    deadline = time.monotonic() + timeout
    interrupted = False

    def interrupt(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # Attached client loss is not command loss.
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        close_fds=True,
    )
    retained = {"stdout": bytearray(), "stderr": bytearray()}
    counts = {"stdout": 0, "stderr": 0}
    direct_status = None
    timed_out = False
    cleanup_deadline = None
    settled = False
    try:
        with selectors.DefaultSelector() as selector:
            for name in retained:
                stream = getattr(process, name)
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            while True:
                now = time.monotonic()
                if cleanup_deadline is None and (interrupted or now >= deadline):
                    timed_out = not interrupted
                    cleanup_deadline = now + CLEANUP_SECONDS
                if cleanup_deadline is not None:
                    for pid in _children():
                        with suppress(ProcessLookupError):
                            os.kill(pid, signal.SIGKILL)
                no_children = False
                while True:
                    try:
                        pid, status = os.waitpid(-1, os.WNOHANG)
                    except ChildProcessError:
                        no_children = True
                        break
                    if pid == 0:
                        break
                    if pid == process.pid:
                        direct_status = os.waitstatus_to_exitcode(status)
                        process.returncode = direct_status
                for entry, _mask in selector.select(0.01):
                    data = os.read(entry.fd, 65_536)
                    if not data:
                        selector.unregister(entry.fileobj)
                    else:
                        name = entry.data
                        counts[name] += len(data)
                        retained[name].extend(data[: max(0, limit - len(retained[name]))])
                if no_children and not selector.get_map():
                    settled = True
                    break
                if cleanup_deadline is not None and time.monotonic() >= cleanup_deadline:
                    return 125  # No positive settlement; never publish terminal evidence.
        if direct_status is None or interrupted:
            return 125
        payload = {
            "schema": SCHEMA,
            "identity": identity,
            "request_sha256": request_sha256,
            "timeout_seconds": timeout,
            "output_limit": limit,
            "exit_code": direct_status,
            "timed_out": timed_out,
            **{name: base64.b64encode(value).decode("ascii") for name, value in retained.items()},
            **{name + "_bytes": value for name, value in counts.items()},
        }
        _publish(directory_fd, seal(payload, key))
        return 0
    finally:
        if not settled:
            _abort(process)
        os.close(directory_fd)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        # Private launch material must not enter a traceback/guest log.
        exit_code = 125
    raise SystemExit(exit_code)
