"""Guest-side atomic resolve-and-open guard for remote workspaces.

Remote workspaces used to enforce symlink containment host-side by inspecting
guest metadata (``get_info``/``realpath``) and then issuing a separate
open/read/write/delete API call. That check-then-use sequence is racy: a
co-resident process inside the sandbox can swap a checked component for a
symlink between the check and the use (TOCTOU) and redirect the operation
outside the workspace root.

This module instead ships a small Python program into the guest via
``runner.exec`` and performs resolve-and-open *atomically inside the guest*:
every path component below the workspace root is opened with ``O_NOFOLLOW``
relative to the previous component's file descriptor (``openat`` semantics),
so no symlink below the root is ever followed regardless of how the tree
mutates concurrently.

Residual trust
--------------

- The guest must provide a ``python3`` on ``PATH``; guarded operations fail
  closed with ``RuntimeError`` when it is missing.
- The workspace *root* is operator configuration and is trusted: symlinks in
  root components are resolved normally. Containment is enforced strictly
  below the root.
- A co-resident guest process with sufficient privileges can still read or
  modify workspace files directly, bind-mount over the root, or replace the
  ``python3`` interpreter. The sandbox boundary — not this guard — remains
  the security boundary between guest and host; the guard only keeps
  workspace API operations from being redirected outside the root by
  guest-controlled symlinks.
- Guarded operations run as the runner's default exec user, not as any
  workspace-level filesystem API user override.
"""

from __future__ import annotations

import base64
import binascii
from typing import TYPE_CHECKING

from cayu.runners import ExecCommand

if TYPE_CHECKING:
    from cayu.runners import ExecResult, Runner

GUEST_PYTHON = "python3"

_STATUS_OK = "ok"
_STATUS_ENOENT = "enoent"
_STATUS_ESCAPE = "escape"
_STATUS_NOTFILE = "notfile"
_STATUS_ISDIR = "isdir"

_READ_OUTPUT_HEADROOM_BYTES = 4096

# The program below runs inside the guest. It communicates over a tiny
# protocol: exit code 0 with a first stdout line of "ok[ <size>]", "enoent",
# "escape", "notfile", or "isdir"; any non-zero exit is an operational error
# whose detail is on stderr. Read payloads are base64 on stdout after the
# status line; write payloads are base64 on stdin.
GUEST_GUARD_PROGRAM = """
import base64
import errno
import os
import stat
import sys

ESCAPE_ERRNOS = (errno.ELOOP, errno.EMLINK)
MISSING_ERRNOS = (errno.ENOENT, errno.ENOTDIR)
OPEN_BASE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
DIR_FLAGS = OPEN_BASE_FLAGS | os.O_NOFOLLOW | os.O_DIRECTORY


def finish(status):
    print(status)
    sys.stdout.flush()
    sys.exit(0)


def classify_missing(name, dir_fd):
    # The failed O_NOFOLLOW open never followed anything, so this lstat only
    # refines the error report (symlink vs missing), never containment.
    try:
        info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except OSError:
        return "enoent"
    if stat.S_ISLNK(info.st_mode):
        return "escape"
    return "enoent"


def open_component(name, dir_fd, create):
    try:
        return os.open(name, DIR_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in ESCAPE_ERRNOS:
            finish("escape")
        if exc.errno == errno.ENOENT and create:
            try:
                os.mkdir(name, mode=0o755, dir_fd=dir_fd)
            except FileExistsError:
                pass
            return open_component(name, dir_fd, False)
        if exc.errno in MISSING_ERRNOS:
            # Some kernels report ENOTDIR (not ELOOP) for O_NOFOLLOW|O_DIRECTORY
            # on a symlink component; distinguish it from a truly missing path.
            finish(classify_missing(name, dir_fd))
        raise


def read_leaf(name, dir_fd, limit):
    try:
        fd = os.open(name, OPEN_BASE_FLAGS | os.O_NOFOLLOW, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in ESCAPE_ERRNOS:
            finish("escape")
        if exc.errno in MISSING_ERRNOS:
            finish("enoent")
        raise
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        finish("notfile")
    chunks = []
    remaining = limit
    while remaining > 0:
        chunk = os.read(fd, min(remaining, 1 << 16))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    print("ok " + str(info.st_size))
    sys.stdout.write(base64.b64encode(b"".join(chunks)).decode("ascii"))
    sys.stdout.flush()


def write_leaf(name, dir_fd):
    payload = base64.b64decode(sys.stdin.read(), validate=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(name, flags, mode=0o644, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in ESCAPE_ERRNOS:
            finish("escape")
        if exc.errno in MISSING_ERRNOS:
            finish("enoent")
        if exc.errno == errno.EISDIR:
            finish("isdir")
        raise
    view = memoryview(payload)
    while view:
        view = view[os.write(fd, view):]
    finish("ok")


def delete_leaf(name, dir_fd):
    try:
        info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except OSError as exc:
        if exc.errno in MISSING_ERRNOS:
            finish("enoent")
        raise
    if stat.S_ISLNK(info.st_mode):
        finish("escape")
    if not stat.S_ISREG(info.st_mode):
        finish("isdir")
    try:
        os.unlink(name, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in MISSING_ERRNOS:
            finish("enoent")
        raise
    finish("ok")


def main():
    mode = sys.argv[1]
    root = sys.argv[2]
    rel_path = sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    parts = [part for part in rel_path.split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        finish("escape")
    try:
        dir_fd = os.open(root, OPEN_BASE_FLAGS | os.O_DIRECTORY)
    except OSError as exc:
        if exc.errno in MISSING_ERRNOS:
            finish("enoent")
        raise
    for name in parts[:-1]:
        next_fd = open_component(name, dir_fd, mode == "write")
        os.close(dir_fd)
        dir_fd = next_fd
    if mode == "read":
        read_leaf(parts[-1], dir_fd, limit)
    elif mode == "write":
        write_leaf(parts[-1], dir_fd)
    elif mode == "delete":
        delete_leaf(parts[-1], dir_fd)
    else:
        raise SystemExit("unknown guard mode: " + mode)


main()
"""


async def guard_read(
    runner: Runner,
    *,
    root: str,
    rel_path: str,
    limit: int,
    original_path: str,
    backend: str,
    timeout_s: int | None = None,
) -> tuple[bytes, int]:
    """Atomically resolve-and-read a contained file; return (content, total size)."""

    output_limit = 4 * ((limit + 2) // 3) + _READ_OUTPUT_HEADROOM_BYTES
    result = await _exec_guard(
        runner,
        "read",
        root,
        rel_path,
        str(limit),
        timeout_s=timeout_s,
        output_limit_bytes=output_limit,
    )
    status, payload = _guard_status(
        result, mode="read", backend=backend, original_path=original_path
    )
    if status in {_STATUS_ENOENT, _STATUS_NOTFILE}:
        raise FileNotFoundError(f"Workspace file not found: {original_path}")
    _raise_common_status(status, mode="read", backend=backend, original_path=original_path)
    total_bytes = _parse_ok_size(status, backend=backend, original_path=original_path)
    try:
        content = base64.b64decode(payload.strip(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(
            f"{backend} workspace guard returned an invalid read payload: {original_path}"
        ) from exc
    return content, total_bytes


async def guard_write(
    runner: Runner,
    *,
    root: str,
    rel_path: str,
    content: bytes,
    original_path: str,
    backend: str,
    timeout_s: int | None = None,
) -> None:
    """Atomically resolve-and-write a contained file, creating parent directories."""

    result = await _exec_guard(
        runner,
        "write",
        root,
        rel_path,
        stdin=base64.b64encode(content).decode("ascii"),
        timeout_s=timeout_s,
    )
    status, _ = _guard_status(result, mode="write", backend=backend, original_path=original_path)
    if status == _STATUS_OK:
        return
    if status == _STATUS_ENOENT:
        raise FileNotFoundError(f"Workspace path not found: {original_path}")
    if status == _STATUS_ISDIR:
        raise IsADirectoryError(f"Workspace path is not a file: {original_path}")
    _raise_common_status(status, mode="write", backend=backend, original_path=original_path)
    raise AssertionError("unreachable")


async def guard_delete(
    runner: Runner,
    *,
    root: str,
    rel_path: str,
    original_path: str,
    backend: str,
    timeout_s: int | None = None,
) -> None:
    """Atomically resolve-and-unlink a contained file; missing paths are a no-op."""

    result = await _exec_guard(runner, "delete", root, rel_path, timeout_s=timeout_s)
    status, _ = _guard_status(result, mode="delete", backend=backend, original_path=original_path)
    if status in {_STATUS_OK, _STATUS_ENOENT}:
        return
    if status == _STATUS_ISDIR:
        raise IsADirectoryError(f"Workspace path is not a file: {original_path}")
    _raise_common_status(status, mode="delete", backend=backend, original_path=original_path)
    raise AssertionError("unreachable")


async def _exec_guard(
    runner: Runner,
    mode: str,
    root: str,
    rel_path: str,
    *extra_args: str,
    stdin: str | None = None,
    timeout_s: int | None = None,
    output_limit_bytes: int | None = None,
) -> ExecResult:
    command = ExecCommand.process(
        GUEST_PYTHON, "-c", GUEST_GUARD_PROGRAM, mode, root, rel_path, *extra_args
    )
    kwargs: dict[str, object] = {"stdin": stdin, "timeout_s": timeout_s}
    if output_limit_bytes is not None:
        kwargs["output_limit_bytes"] = output_limit_bytes
    return await runner.exec(command, **kwargs)  # type: ignore


def _guard_status(
    result: ExecResult,
    *,
    mode: str,
    backend: str,
    original_path: str,
) -> tuple[str, str]:
    if result.exit_code != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.exit_code}"
        hint = ""
        if result.exit_code == 127:
            hint = f" ({GUEST_PYTHON} is required inside the guest for guarded workspace access)"
        if result.timed_out:
            hint = " (guard command timed out)"
        raise RuntimeError(
            f"Failed to {mode} {backend} workspace file: {original_path}: {detail}{hint}"
        )
    if result.stdout_truncated:
        raise RuntimeError(f"{backend} workspace guard output was truncated: {original_path}")
    head, _, tail = result.stdout.partition("\n")
    return head.strip(), tail


def _raise_common_status(status: str, *, mode: str, backend: str, original_path: str) -> None:
    if status == _STATUS_ESCAPE:
        raise ValueError("Workspace path escapes the workspace root.")
    if status == _STATUS_OK or status.startswith(f"{_STATUS_OK} "):
        return
    raise RuntimeError(
        f"Failed to {mode} {backend} workspace file: "
        f"{original_path}: unexpected guard status {status!r}"
    )


def _parse_ok_size(status: str, *, backend: str, original_path: str) -> int:
    parts = status.split()
    if len(parts) == 2 and parts[0] == _STATUS_OK and parts[1].isdigit():
        return int(parts[1])
    raise RuntimeError(
        f"Failed to read {backend} workspace file: "
        f"{original_path}: unexpected guard status {status!r}"
    )
