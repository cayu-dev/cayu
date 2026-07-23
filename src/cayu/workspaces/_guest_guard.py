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
mutates concurrently. Within one traversal, an opened descriptor authorizes
that inode, not its current pathname: moving the inode later cannot redirect
that traversal through a replacement symlink. Operations with multiple passes
start each later traversal from the pinned root and reject a replacement
symlink there.

Residual trust
--------------

- The guest must provide a ``python3`` on ``PATH``; guarded operations fail
  closed with ``RuntimeError`` when it is missing.
- The workspace *root* is operator configuration and is trusted: symlinks in
  root components are resolved normally. Containment is enforced strictly
  below the root.
- A co-resident guest process with sufficient privileges can still read or
  modify or relocate workspace files directly, bind-mount over the root, or
  replace the ``python3`` interpreter. The sandbox boundary — not this guard
  — remains the security boundary between guest and host; the guard only
  keeps workspace API operations from being redirected through
  guest-controlled symlinks. It does not enforce continuous pathname
  membership for an inode after that inode has been safely opened.
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
_STATUS_NOTDIR = "notdir"
_STATUS_ISDIR = "isdir"
_STATUS_HARDLINK = "hardlink"
_STATUS_UNSUPPORTED = "unsupported"

_READ_OUTPUT_HEADROOM_BYTES = 4096


# The program below runs inside the guest. It communicates over a tiny
# protocol: exit code 0 with a first stdout line of "ok[ <size>]", "enoent",
# "escape", "notfile", "notdir", "isdir", "hardlink", or "unsupported"; any
# non-zero exit is an operational error whose detail is on stderr. Read
# payloads are base64 on stdout after the status line; write payloads are
# base64 on stdin.
GUEST_DESCRIPTOR_GUARD_SOURCE = r"""
import errno
import os
import stat
import sys


class GuardPathError(Exception):
    def __init__(self, status):
        self.status = status
        super().__init__(status)


ESCAPE_ERRNOS = (errno.ELOOP, errno.EMLINK)
MISSING_ERRNOS = (errno.ENOENT, errno.ENOTDIR)
OPEN_BASE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
NONBLOCK_FLAG = getattr(os, "O_NONBLOCK", 0)
# Preserve the established remote-adapter creation policy by default. Programs
# with different native semantics, such as RunnerWorkspace, may override these
# globals before invoking the shared guard functions.
GUARDED_DIRECTORY_CREATE_MODE = 0o755
GUARDED_FILE_CREATE_MODE = 0o644
if hasattr(os, "O_PATH"):
    SEARCH_BASE_FLAGS = os.O_PATH | getattr(os, "O_CLOEXEC", 0)
elif hasattr(os, "O_SEARCH"):
    SEARCH_BASE_FLAGS = os.O_SEARCH | getattr(os, "O_CLOEXEC", 0)
elif sys.platform == "darwin":
    # CPython does not expose Darwin's O_SEARCH even though the kernel supports it.
    SEARCH_BASE_FLAGS = 0x40000000 | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
else:
    SEARCH_BASE_FLAGS = OPEN_BASE_FLAGS


def require_descriptor_guard_support():
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise GuardPathError("unsupported")
    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    supports_follow_symlinks = getattr(os, "supports_follow_symlinks", ())
    required_dir_fd = (os.open, os.mkdir, os.stat, os.unlink)
    if any(operation not in supports_dir_fd for operation in required_dir_fd):
        raise GuardPathError("unsupported")
    if os.stat not in supports_follow_symlinks or not hasattr(os, "ftruncate"):
        raise GuardPathError("unsupported")


def guarded_parts(rel_path):
    if not isinstance(rel_path, str) or not rel_path or "\x00" in rel_path:
        raise GuardPathError("escape")
    if rel_path.startswith("/"):
        raise GuardPathError("escape")
    raw_parts = rel_path.split("/")
    if ".." in raw_parts:
        raise GuardPathError("escape")
    parts = [part for part in raw_parts if part not in ("", ".")]
    if not parts:
        raise GuardPathError("escape")
    return parts


def open_guard_root(root, readable=False):
    require_descriptor_guard_support()
    try:
        base_flags = OPEN_BASE_FLAGS if readable else SEARCH_BASE_FLAGS
        return os.open(root, base_flags | os.O_DIRECTORY)
    except OSError as exc:
        if exc.errno in MISSING_ERRNOS:
            raise GuardPathError("enoent") from exc
        raise


def classify_missing(name, dir_fd):
    # The failed O_NOFOLLOW open never followed anything, so this lstat only
    # refines the error report (symlink vs missing), never containment.
    try:
        info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except OSError:
        return "enoent"
    if stat.S_ISLNK(info.st_mode):
        return "escape"
    return "notdir"


def open_guarded_directory(name, dir_fd, create, readable=False):
    base_flags = OPEN_BASE_FLAGS if readable else SEARCH_BASE_FLAGS
    flags = base_flags | os.O_NOFOLLOW | os.O_DIRECTORY
    try:
        opened_fd = os.open(name, flags, dir_fd=dir_fd)
        # CAYU_TEST_BARRIER_AFTER_DIRECTORY_OPEN
        return opened_fd
    except OSError as exc:
        if exc.errno in ESCAPE_ERRNOS:
            raise GuardPathError("escape") from exc
        if exc.errno == errno.ENOENT and create:
            try:
                os.mkdir(
                    name,
                    mode=GUARDED_DIRECTORY_CREATE_MODE,
                    dir_fd=dir_fd,
                )
            except FileExistsError:
                pass
            return open_guarded_directory(name, dir_fd, False, readable)
        if exc.errno in MISSING_ERRNOS:
            # Some kernels report ENOTDIR (not ELOOP) for O_NOFOLLOW|O_DIRECTORY
            # on a symlink component; distinguish it from a truly missing path.
            raise GuardPathError(classify_missing(name, dir_fd)) from exc
        raise


def open_guarded_parent(root_fd, parts, create):
    current_fd = os.dup(root_fd)
    try:
        for name in parts[:-1]:
            next_fd = open_guarded_directory(name, current_fd, create)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd, parts[-1]
    except BaseException:
        os.close(current_fd)
        raise


def guarded_lstat(name, dir_fd):
    try:
        return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except OSError as exc:
        if exc.errno in MISSING_ERRNOS:
            raise GuardPathError("enoent") from exc
        raise


def open_guarded_regular(name, dir_fd):
    try:
        before = guarded_lstat(name, dir_fd)
    except GuardPathError:
        raise
    if stat.S_ISLNK(before.st_mode):
        raise GuardPathError("escape")
    if not stat.S_ISREG(before.st_mode):
        raise GuardPathError("notfile")
    try:
        fd = os.open(
            name,
            OPEN_BASE_FLAGS | NONBLOCK_FLAG | os.O_NOFOLLOW,
            dir_fd=dir_fd,
        )
    except OSError as exc:
        if exc.errno in ESCAPE_ERRNOS:
            raise GuardPathError("escape") from exc
        if exc.errno in MISSING_ERRNOS:
            raise GuardPathError(classify_missing(name, dir_fd)) from exc
        raise
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise GuardPathError("notfile")
    return fd, info


def inspect_guarded_write_target(name, dir_fd):
    try:
        before = guarded_lstat(name, dir_fd)
    except GuardPathError as exc:
        if exc.status != "enoent":
            raise
        return None
    if stat.S_ISLNK(before.st_mode):
        raise GuardPathError("escape")
    if not stat.S_ISREG(before.st_mode):
        raise GuardPathError("isdir")
    flags = os.O_WRONLY | NONBLOCK_FLAG | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in ESCAPE_ERRNOS:
            raise GuardPathError("escape") from exc
        if exc.errno in MISSING_ERRNOS:
            raise GuardPathError(classify_missing(name, dir_fd)) from exc
        if exc.errno == errno.EISDIR:
            raise GuardPathError("isdir") from exc
        raise
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise GuardPathError("isdir")
    if info.st_nlink != 1:
        os.close(fd)
        raise GuardPathError("hardlink")
    os.close(fd)
    return stat.S_IMODE(info.st_mode) & 0o777


def open_guarded_regular_for_write(name, dir_fd):
    # CAYU_TEST_BARRIER_BEFORE_WRITE_TARGET_OPEN
    flags = (
        os.O_WRONLY
        | NONBLOCK_FLAG
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _ in range(128):
        try:
            before = guarded_lstat(name, dir_fd)
        except GuardPathError as exc:
            if exc.status != "enoent":
                raise
            try:
                # O_EXCL makes the absence decision and creation one kernel
                # operation. The selected mode remains filtered by the guest
                # process's umask.
                fd = os.open(
                    name,
                    flags | os.O_CREAT | os.O_EXCL,
                    GUARDED_FILE_CREATE_MODE,
                    dir_fd=dir_fd,
                )
            except FileExistsError:
                continue
            except OSError as open_exc:
                if open_exc.errno in ESCAPE_ERRNOS:
                    raise GuardPathError("escape") from open_exc
                if open_exc.errno in (errno.EISDIR, errno.ENXIO):
                    raise GuardPathError("isdir") from open_exc
                raise
        else:
            if stat.S_ISLNK(before.st_mode):
                raise GuardPathError("escape")
            if not stat.S_ISREG(before.st_mode):
                raise GuardPathError("isdir")
            try:
                fd = os.open(name, flags, dir_fd=dir_fd)
            except OSError as open_exc:
                if open_exc.errno in MISSING_ERRNOS:
                    continue
                if open_exc.errno in ESCAPE_ERRNOS:
                    raise GuardPathError("escape") from open_exc
                if open_exc.errno in (errno.EISDIR, errno.ENXIO):
                    raise GuardPathError("isdir") from open_exc
                raise

        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise GuardPathError("isdir")
        if info.st_nlink != 1:
            os.close(fd)
            raise GuardPathError("hardlink")
        # CAYU_TEST_BARRIER_AFTER_WRITE_TARGET_OPEN
        return fd
    raise OSError("workspace write target changed too often")


def write_guarded_regular(name, dir_fd, chunks):
    fd = open_guarded_regular_for_write(name, dir_fd)
    try:
        os.ftruncate(fd, 0)
        # CAYU_TEST_AFTER_WRITE_TRUNCATE
        for payload in chunks:
            write_all(fd, payload)
    finally:
        os.close(fd)


def write_all(fd, payload):
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("guarded file write made no progress")
        view = view[written:]


def delete_guarded_regular(name, dir_fd):
    try:
        before = guarded_lstat(name, dir_fd)
    except GuardPathError as exc:
        if exc.status == "enoent":
            return False
        raise
    if stat.S_ISLNK(before.st_mode):
        raise GuardPathError("escape")
    if not stat.S_ISREG(before.st_mode):
        raise GuardPathError("isdir")
    try:
        os.unlink(name, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in MISSING_ERRNOS:
            return False
        raise
    return True
"""


GUEST_GUARD_PROGRAM = (
    GUEST_DESCRIPTOR_GUARD_SOURCE
    + r"""
import base64
import sys


def finish(status):
    print(status)
    sys.stdout.flush()
    sys.exit(0)


def main():
    mode = sys.argv[1]
    root = sys.argv[2]
    rel_path = sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    root_fd = None
    parent_fd = None
    leaf_fd = None
    try:
        parts = guarded_parts(rel_path)
        root_fd = open_guard_root(root)
        parent_fd, leaf_name = open_guarded_parent(root_fd, parts, mode == "write")
        if mode == "read":
            leaf_fd, info = open_guarded_regular(leaf_name, parent_fd)
            chunks = []
            remaining = limit
            while remaining > 0:
                chunk = os.read(leaf_fd, min(remaining, 1 << 16))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            print("ok " + str(info.st_size))
            sys.stdout.write(base64.b64encode(b"".join(chunks)).decode("ascii"))
            sys.stdout.flush()
            return
        if mode == "write":
            payload = base64.b64decode(sys.stdin.read(), validate=True)
            write_guarded_regular(leaf_name, parent_fd, (payload,))
            finish("ok")
        if mode == "delete":
            finish("ok" if delete_guarded_regular(leaf_name, parent_fd) else "enoent")
        raise SystemExit("unknown guard mode: " + mode)
    except GuardPathError as exc:
        finish(exc.status)
    finally:
        for fd in (leaf_fd, parent_fd, root_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass


main()
"""
)


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
    if status in {_STATUS_ENOENT, _STATUS_NOTFILE, _STATUS_NOTDIR}:
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
    """Resolve a contained file and write through its guarded descriptor."""

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
    if status in {_STATUS_ENOENT, _STATUS_NOTDIR}:
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
    if status in {_STATUS_OK, _STATUS_ENOENT, _STATUS_NOTDIR}:
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
    if status == _STATUS_HARDLINK:
        raise ValueError(f"Workspace file has multiple hard links: {original_path}")
    if status == _STATUS_UNSUPPORTED:
        raise RuntimeError(
            f"{backend} workspace requires POSIX descriptor-relative filesystem primitives."
        )
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
