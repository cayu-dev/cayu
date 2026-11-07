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
import json
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast

from cayu.runners import ExecCommand
from cayu.workspaces.base import (
    WorkspaceFileReadOutcome,
    WorkspaceFileReadRequest,
    WorkspaceGitMode,
    WorkspaceLineRead,
    WorkspaceLineSpan,
    WorkspaceMoveAmbiguousError,
    WorkspaceMoveResult,
    WorkspaceMoveUnsupportedError,
    WorkspaceMutationResult,
    WorkspacePreconditionUnsupportedError,
    WorkspaceReadOffsetError,
    WorkspaceReadResult,
    WorkspaceRevisionMismatchError,
    WorkspaceTextEditResult,
    WorkspaceTextReplacement,
)

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
_STATUS_EXISTS = "exists"
_STATUS_OFFSET = "offset"
_STATUS_STALE = "stale"
_STATUS_AMBIGUOUS = "ambiguous"

_READ_OUTPUT_HEADROOM_BYTES = 4096
# Transports cap command input (a brokered sandbox takes 1 MiB), and the
# separate replace sends only the result, so a combined edit carries at most
# this much edit text.
_COMBINED_EDIT_STDIN_LIMIT_BYTES = 64 * 1024


# The program below runs inside the guest. It communicates over a tiny
# protocol: exit code 0 with a first stdout line of "ok[ <size>]", "enoent",
# "escape", "notfile", "notdir", "isdir", "hardlink", "offset <size>", or
# "unsupported"; any
# non-zero exit is an operational error whose detail is on stderr. Read
# payloads are base64 on stdout after the status line; write payloads are
# base64 on stdin. Line lookups answer only
# "ok <size> <start offset> <end offset> <lines>" and return no file content.
# A line read ("read-lines") and a text edit ("edit-text") answer "ok", then
# JSON metadata lines, then each read's bytes as one base64 line.
GUEST_DESCRIPTOR_GUARD_SOURCE = r"""
import contextlib
import errno
import fcntl
import hashlib
import os
import stat
import sys
import tempfile
import unicodedata


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
    required_dir_fd = (os.open, os.mkdir, os.stat, os.unlink, os.link, os.rename)
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


def open_guarded_parent(root_fd, parts, create, readable=False):
    current_fd = os.dup(root_fd)
    try:
        for name in parts[:-1]:
            next_fd = open_guarded_directory(
                name,
                current_fd,
                create,
                readable,
            )
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


def write_guarded_regular(name, dir_fd, chunks, mode=None):
    fd = open_guarded_regular_for_write(name, dir_fd)
    try:
        os.ftruncate(fd, 0)
        # CAYU_TEST_AFTER_WRITE_TRUNCATE
        for payload in chunks:
            write_all(fd, payload)
        if mode is not None:
            os.fchmod(fd, mode)
    finally:
        os.close(fd)


def write_all(fd, payload):
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("guarded file write made no progress")
        view = view[written:]


def content_identity(content):
    digest = hashlib.sha256(content).hexdigest()
    return "sha256:" + digest, digest


def read_all(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(fd, 1 << 16)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


@contextlib.contextmanager
def workspace_path_lock(root_fd, rel_path):
    root_info = os.fstat(root_fd)
    normalized_path = unicodedata.normalize("NFC", rel_path.replace("\\", "/")).casefold()
    identity = f"{root_info.st_dev}:{root_info.st_ino}\0{normalized_path}".encode()
    lock_name = hashlib.sha256(identity).hexdigest()
    lock_root = os.path.join(tempfile.gettempdir(), "cayu-workspace-locks")
    os.makedirs(lock_root, mode=0o700, exist_ok=True)
    lock_fd = os.open(
        os.path.join(lock_root, lock_name),
        os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


@contextlib.contextmanager
def workspace_path_locks(root_fd, *rel_paths):
    ordered = sorted(
        rel_paths,
        key=lambda value: unicodedata.normalize("NFC", value.replace("\\", "/")).casefold(),
    )
    identities = [
        unicodedata.normalize("NFC", value.replace("\\", "/")).casefold()
        for value in ordered
    ]
    if len(set(identities)) != len(identities):
        raise GuardPathError("escape")
    with contextlib.ExitStack() as stack:
        for path in ordered:
            stack.enter_context(workspace_path_lock(root_fd, path))
        yield


@contextlib.contextmanager
def workspace_source_lock(root_fd, exclusive):
    # Cooperate with guest-side multi-path branch capture/publication.
    root_info = os.fstat(root_fd)
    identity = f"{root_info.st_dev}:{root_info.st_ino}".encode()
    lock_name = hashlib.sha256(identity).hexdigest()
    lock_root = os.path.join(tempfile.gettempdir(), "cayu-workspace-source-locks")
    os.makedirs(lock_root, mode=0o700, exist_ok=True)
    lock_fd = os.open(
        os.path.join(lock_root, lock_name),
        os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _temporary_name(name):
    return "." + name + ".cayu-" + os.urandom(12).hex()


def create_guarded_regular_atomic(name, dir_fd, content):
    try:
        guarded_lstat(name, dir_fd)
    except GuardPathError as exc:
        if exc.status != "enoent":
            raise
    else:
        raise GuardPathError("exists")
    temp_name = _temporary_name(name)
    temp_fd = None
    try:
        temp_fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            GUARDED_FILE_CREATE_MODE,
            dir_fd=dir_fd,
        )
        write_all(temp_fd, content)
        os.close(temp_fd)
        temp_fd = None
        try:
            os.link(
                temp_name,
                name,
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise GuardPathError("exists") from exc
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        try:
            os.unlink(temp_name, dir_fd=dir_fd)
        except FileNotFoundError:
            pass


def _conditional_snapshot(name, dir_fd, require_single_link=True):
    fd, info = open_guarded_regular(name, dir_fd)
    try:
        if require_single_link and info.st_nlink != 1:
            raise GuardPathError("hardlink")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    finally:
        os.close(fd)
    hexdigest = digest.hexdigest()
    return (
        "sha256:" + hexdigest,
        hexdigest,
        size,
        stat.S_IMODE(info.st_mode) & 0o777,
    )


def replace_guarded_regular_if_revision(name, dir_fd, content, expected_revision):
    revision, digest, size, mode = _conditional_snapshot(name, dir_fd)
    if revision != expected_revision:
        return None, revision
    temp_name = _temporary_name(name)
    temp_fd = None
    try:
        temp_fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            mode,
            dir_fd=dir_fd,
        )
        write_all(temp_fd, content)
        os.close(temp_fd)
        temp_fd = None
        os.rename(temp_name, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        try:
            os.unlink(temp_name, dir_fd=dir_fd)
        except FileNotFoundError:
            pass
    return (revision, digest, size), None


def delete_guarded_regular_if_revision(name, dir_fd, expected_revision):
    revision, digest, size, _ = _conditional_snapshot(name, dir_fd)
    if revision != expected_revision:
        return None, revision
    os.unlink(name, dir_fd=dir_fd)
    return (revision, digest, size), None


def require_guarded_absent(name, dir_fd):
    try:
        guarded_lstat(name, dir_fd)
    except GuardPathError as exc:
        if exc.status == "enoent":
            return
        raise
    raise GuardPathError("exists")


def move_guarded_regular_if_revision(
    root_fd,
    source_parts,
    destination_parts,
    expected_revision,
):
    source_parent_fd = None
    destination_parent_fd = None
    try:
        source_parent_fd, source_name = open_guarded_parent(
            root_fd, source_parts, False
        )
        before = _conditional_snapshot(source_name, source_parent_fd)
        if before[0] != expected_revision:
            return None, before[0], False
        destination_parent_fd, destination_name = open_guarded_parent(
            root_fd, destination_parts, True
        )
        require_guarded_absent(destination_name, destination_parent_fd)
        try:
            os.link(
                source_name,
                destination_name,
                src_dir_fd=source_parent_fd,
                dst_dir_fd=destination_parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise GuardPathError("exists") from exc
        except OSError as exc:
            if exc.errno == errno.EXDEV:
                raise GuardPathError("unsupported") from exc
            raise
        destination = _conditional_snapshot(
            destination_name,
            destination_parent_fd,
            require_single_link=False,
        )
        identity = before[:3]
        if destination[:3] != identity:
            return identity, None, True
        try:
            os.unlink(source_name, dir_fd=source_parent_fd)
        except OSError:
            return identity, None, True
        return identity, None, False
    finally:
        for fd in (destination_parent_fd, source_parent_fd):
            if fd is not None:
                os.close(fd)


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
import json
import sys


def finish(status):
    print(status)
    sys.stdout.flush()
    sys.exit(0)


def locate_lines(fd, start_line, end_line):
    # Line N starts after the (N-1)th newline and ends after the Nth; an
    # end_line of 0 runs to the end of the file. Offsets not found are EOF.
    newlines = 0
    offset = 0
    start_offset = 0 if start_line == 1 else -1
    end_offset = -1
    ends_with_newline = True
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            break
        pos = 0
        while start_offset < 0 or (end_line and end_offset < 0):
            wanted = start_line - 1 if start_offset < 0 else end_line
            needed = wanted - newlines
            if chunk.count(b"\n", pos) < needed:
                break
            for _ in range(needed):
                pos = chunk.index(b"\n", pos) + 1
            newlines = wanted
            if start_offset < 0:
                start_offset = offset + pos
            else:
                end_offset = offset + pos
        newlines += chunk.count(b"\n", pos)
        offset += len(chunk)
        ends_with_newline = chunk.endswith(b"\n")
    total_lines = newlines if ends_with_newline else newlines + 1
    if start_offset < 0:
        start_offset = offset
    if end_offset < 0:
        end_offset = offset
    return offset, start_offset, end_offset, total_lines


def mutation_payload(operation, before, after):
    before_revision, before_sha256 = (
        content_identity(before) if before is not None else (None, None)
    )
    after_revision, after_sha256 = (
        content_identity(after) if after is not None else (None, None)
    )
    return {
        "operation": operation,
        "before_revision": before_revision,
        "after_revision": after_revision,
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
        "before_bytes": len(before) if before is not None else None,
        "after_bytes": len(after) if after is not None else None,
    }


def mutation_payload_from_identity(operation, before, after):
    after_revision, after_sha256 = (
        content_identity(after) if after is not None else (None, None)
    )
    return {
        "operation": operation,
        "before_revision": before[0] if before is not None else None,
        "after_revision": after_revision,
        "before_sha256": before[1] if before is not None else None,
        "after_sha256": after_sha256,
        "before_bytes": before[2] if before is not None else None,
        "after_bytes": len(after) if after is not None else None,
    }


def move_payload_from_identity(before):
    return {
        "source_before_revision": before[0],
        "source_after_revision": None,
        "destination_before_revision": None,
        "destination_after_revision": before[0],
        "source_before_sha256": before[1],
        "source_after_sha256": None,
        "destination_before_sha256": None,
        "destination_after_sha256": before[1],
        "source_before_bytes": before[2],
        "source_after_bytes": None,
        "destination_before_bytes": None,
        "destination_after_bytes": before[2],
        "fidelity": "link_unlink",
    }


def main():
    mode = sys.argv[1]
    root = sys.argv[2]
    rel_path = sys.argv[3]
    root_fd = None
    parent_fd = None
    leaf_fd = None
    try:
        parts = guarded_parts(rel_path)
        root_fd = open_guard_root(root)
        if mode == "require-absent":
            with workspace_source_lock(root_fd, False), workspace_path_lock(
                root_fd, rel_path
            ):
                parent_fd, leaf_name = open_guarded_parent(root_fd, parts, False)
                require_guarded_absent(leaf_name, parent_fd)
                finish("ok")
        if mode == "move-if-revision":
            destination_path = sys.argv[4]
            expected_revision = sys.argv[5]
            destination_parts = guarded_parts(destination_path)
            with workspace_source_lock(root_fd, False), workspace_path_locks(
                root_fd, rel_path, destination_path
            ):
                before_data, stale_revision, ambiguous = (
                    move_guarded_regular_if_revision(
                        root_fd,
                        parts,
                        destination_parts,
                        expected_revision,
                    )
                )
                if stale_revision is not None:
                    finish("stale " + stale_revision)
                assert before_data is not None
                print("ambiguous" if ambiguous else "ok")
                print(json.dumps(move_payload_from_identity(before_data)))
                return
        with workspace_source_lock(root_fd, False), workspace_path_lock(root_fd, rel_path):
            parent_fd, leaf_name = open_guarded_parent(
                root_fd, parts, mode in ("write", "create")
            )
            if mode == "read":
                offset = int(sys.argv[4])
                limit = int(sys.argv[5])
                leaf_fd, info = open_guarded_regular(leaf_name, parent_fd)
                if offset > info.st_size:
                    finish(f"offset {info.st_size}")
                os.lseek(leaf_fd, offset, os.SEEK_SET)
                chunks = []
                remaining = limit
                while remaining > 0:
                    chunk = os.read(leaf_fd, min(remaining, 1 << 16))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                content = b"".join(chunks)
                revision, digest = (
                    content_identity(content)
                    if offset == 0 and len(content) == info.st_size
                    else ("-", "-")
                )
                git_mode = (
                    "100755"
                    if revision != "-" and stat.S_IMODE(os.fstat(leaf_fd).st_mode) & 0o111
                    else "100644"
                    if revision != "-"
                    else "-"
                )
                print(f"ok {info.st_size} {revision} {digest} {git_mode}")
                sys.stdout.write(base64.b64encode(content).decode("ascii"))
                sys.stdout.flush()
                return
            if mode == "lines":
                leaf_fd, _ = open_guarded_regular(leaf_name, parent_fd)
                located = locate_lines(leaf_fd, int(sys.argv[4]), int(sys.argv[5]))
                finish("ok " + " ".join(str(value) for value in located))
            if mode == "write":
                payload = base64.b64decode(sys.stdin.read(), validate=True)
                write_guarded_regular(leaf_name, parent_fd, (payload,))
                finish("ok")
            if mode == "delete":
                finish("ok" if delete_guarded_regular(leaf_name, parent_fd) else "enoent")
            if mode == "create":
                payload = base64.b64decode(sys.stdin.read(), validate=True)
                create_guarded_regular_atomic(leaf_name, parent_fd, payload)
                print("ok")
                print(json.dumps(mutation_payload("create", None, payload)))
                return
            if mode == "replace":
                expected_revision = sys.argv[4]
                payload = base64.b64decode(sys.stdin.read(), validate=True)
                before_data, stale_revision = replace_guarded_regular_if_revision(
                    leaf_name, parent_fd, payload, expected_revision
                )
                if stale_revision is not None:
                    finish("stale " + stale_revision)
                print("ok")
                print(json.dumps(mutation_payload_from_identity("replace", before_data, payload)))
                return
            if mode == "delete-if-revision":
                expected_revision = sys.argv[4]
                before_data, stale_revision = delete_guarded_regular_if_revision(
                    leaf_name, parent_fd, expected_revision
                )
                if stale_revision is not None:
                    finish("stale " + stale_revision)
                print("ok")
                print(json.dumps(mutation_payload_from_identity("delete", before_data, None)))
                return
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


# Combined modes, which run in place of main() and are sent only with the
# commands that use them, so every other guarded command stays as it was.
# combined_main() applies main()'s containment and locks. "read-lines" locates
# a line range, then reads a window around it and, when the range starts after
# offset 0, the file's head, as one observation; revisions, digests and Git
# modes describe only complete reads. "edit-text" reads the file and, when the
# edit applies, replaces it under the same path lock.
GUEST_COMBINED_MODES_SOURCE = r"""
def read_range(fd, offset, limit):
    os.lseek(fd, offset, os.SEEK_SET)
    chunks = []
    remaining = limit
    while remaining > 0:
        chunk = os.read(fd, min(remaining, 1 << 16))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_identity(fd, offset, content, size):
    if offset or len(content) != size:
        return None, None, None
    revision, digest = content_identity(content)
    executable = stat.S_IMODE(os.fstat(fd).st_mode) & 0o111
    return revision, digest, "100755" if executable else "100644"


def apply_text_edits(text, edits, max_bytes):
    # edit_file's rules; the size is checked before the result is built.
    replacements = []
    size = len(text.encode("utf-8"))
    for old_text, new_text, expected in edits:
        found = text.find(old_text)
        starts = []
        while found >= 0 and len(starts) <= expected:
            starts.append(found)
            found = text.find(old_text, found + len(old_text))
        if len(starts) != expected:
            return None
        size += expected * (len(new_text.encode("utf-8")) - len(old_text.encode("utf-8")))
        replacements.extend((start, start + len(old_text), new_text) for start in starts)
    if size > max_bytes:
        return None
    replacements.sort()
    parts = []
    cursor = 0
    for start, end, new_text in replacements:
        if start < cursor:
            return None
        parts.extend((text[cursor:start], new_text))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def read_located_lines(leaf_fd, start_line, end_line, before, after, limit, head_limit):
    total, start, end, lines = locate_lines(leaf_fd, start_line, end_line)
    # Reads stop at the size the scan saw, so all three agree.
    offset = start - min(start, before)
    window = read_range(leaf_fd, offset, min(limit, min(total, end + after) - offset))
    reads = [(offset, window)]
    if head_limit and start:
        reads.append((0, read_range(leaf_fd, 0, min(head_limit, total))))
    metadata = {
        "span": [start, end, total, lines],
        "reads": [[at, read_identity(leaf_fd, at, content, total)] for at, content in reads],
    }
    return metadata, [content for _, content in reads]


def read_lines_mode(leaf_name, parent_fd):
    leaf_fd, _ = open_guarded_regular(leaf_name, parent_fd)
    try:
        metadata, contents = read_located_lines(
            leaf_fd, *(int(value) for value in sys.argv[4:10])
        )
    finally:
        os.close(leaf_fd)
    print("ok")
    print(json.dumps(metadata))
    for content in contents:
        print(base64.b64encode(content).decode("ascii"))


def edit_text_mode(leaf_name, parent_fd):
    expected_revision = sys.argv[4]
    limit = int(sys.argv[5])
    max_bytes = int(sys.argv[6])
    edits = json.loads(sys.stdin.read())
    leaf_fd, info = open_guarded_regular(leaf_name, parent_fd)
    try:
        content = read_range(leaf_fd, 0, limit)
        identity = read_identity(leaf_fd, 0, content, info.st_size)
    finally:
        os.close(leaf_fd)
    edited = None
    if identity[0] == expected_revision:
        try:
            edited = apply_text_edits(content.decode("utf-8"), edits, max_bytes)
        except UnicodeDecodeError:
            pass
    mutation = None
    if edited is not None:
        edited = edited.encode("utf-8")
        before_data, stale_revision = replace_guarded_regular_if_revision(
            leaf_name, parent_fd, edited, expected_revision
        )
        if stale_revision is not None:
            finish("stale " + stale_revision)
        mutation = mutation_payload_from_identity("replace", before_data, edited)
    print("ok")
    print(json.dumps({"total_bytes": info.st_size, "identity": identity}))
    print(json.dumps(mutation))
    print(base64.b64encode(content).decode("ascii"))


def combined_main():
    mode, root, rel_path = sys.argv[1:4]
    root_fd = None
    parent_fd = None
    try:
        parts = guarded_parts(rel_path)
        root_fd = open_guard_root(root)
        with workspace_source_lock(root_fd, False), workspace_path_lock(root_fd, rel_path):
            parent_fd, leaf_name = open_guarded_parent(root_fd, parts, False)
            if mode == "read-lines":
                read_lines_mode(leaf_name, parent_fd)
            elif mode == "edit-text":
                edit_text_mode(leaf_name, parent_fd)
            else:
                raise SystemExit("unknown guard mode: " + mode)
    except GuardPathError as exc:
        finish(exc.status)
    finally:
        for fd in (parent_fd, root_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
"""


# "read-files" reads several files in one command, each under its own path
# lock and with main()'s containment, as "read" or "read-lines" would read it
# alone. The requests come as JSON on stdin. When their bytes could exceed the
# output the runner returns ("total"), a first pass sizes each file and the
# total is shared out, smallest first, so a large file can't starve the rest;
# a file given less than it asked for comes back as a shorter read. The answer
# is "ok", one JSON line with an outcome per request, then each read's bytes as
# one base64 line. An outcome's "s" is "ok", a guard status, or "error" with
# the OSError's text.
GUEST_READ_FILES_SOURCE = r"""
def share_out(wants, total):
    shares = list(wants)
    order = sorted(range(len(wants)), key=lambda index: wants[index])
    for done, index in enumerate(order):
        shares[index] = min(wants[index], total // (len(order) - done))
        total -= shares[index]
    return shares


def under_path_lock(root_fd, rel_path, action):
    try:
        with workspace_path_lock(root_fd, rel_path):
            parent_fd, leaf_name = open_guarded_parent(root_fd, guarded_parts(rel_path), False)
            try:
                leaf_fd, info = open_guarded_regular(leaf_name, parent_fd)
            finally:
                os.close(parent_fd)
            try:
                return action(leaf_fd, info)
            finally:
                os.close(leaf_fd)
    except GuardPathError as exc:
        return {"s": exc.status}, []
    except OSError as exc:
        return {"s": "error", "detail": str(exc)[:300]}, []


def file_want(request, info):
    if request[0] == "bytes":
        return max(0, min(request[3], info.st_size - request[2]))
    return min(request[6], info.st_size) + min(request[7], info.st_size)


def read_request(request, share, leaf_fd, info):
    if request[0] == "bytes":
        offset = request[2]
        if offset > info.st_size:
            return {"s": "offset", "total": info.st_size}, []
        content = read_range(leaf_fd, offset, min(request[3], share))
        identity = read_identity(leaf_fd, offset, content, info.st_size)
        return {"s": "ok", "total": info.st_size, "reads": [[offset, identity]]}, [content]
    start_line, end_line, before, after, limit, head_limit = request[2:8]
    # The head takes no more than the file holds, and the window gets the rest.
    head_limit = min(head_limit, info.st_size, share)
    metadata, contents = read_located_lines(
        leaf_fd, start_line, end_line, before, after, min(limit, share - head_limit), head_limit
    )
    metadata["s"] = "ok"
    return metadata, contents


def read_files_main():
    batch = json.loads(sys.stdin.read())
    requests = batch["requests"]
    total = batch["total"]
    outcomes = [None] * len(requests)
    blobs = []
    root_fd = None
    try:
        root_fd = open_guard_root(sys.argv[2])
        with workspace_source_lock(root_fd, False):
            shares = [float("inf")] * len(requests)
            if total is not None:
                wanted = [0] * len(requests)
                for index, request in enumerate(requests):
                    outcome, want = under_path_lock(
                        root_fd, request[1], lambda fd, info: (None, file_want(request, info))
                    )
                    if outcome is None:
                        wanted[index] = want
                    else:
                        outcomes[index] = outcome
                shares = share_out(wanted, total)
            for index, request in enumerate(requests):
                if outcomes[index] is None:
                    share = shares[index]
                    outcomes[index], contents = under_path_lock(
                        root_fd,
                        request[1],
                        lambda fd, info: read_request(request, share, fd, info),
                    )
                    blobs.extend(contents)
    except GuardPathError as exc:
        finish(exc.status)
    finally:
        if root_fd is not None:
            os.close(root_fd)
    print("ok")
    print(json.dumps(outcomes))
    for content in blobs:
        print(base64.b64encode(content).decode("ascii"))
"""


_COMBINED_MODES = frozenset({"read-lines", "edit-text"})
_READ_FILES_MODE = "read-files"


def _guard_program(mode: str) -> str:
    """The guard program, with the combined modes only when ``mode`` is one."""

    if mode not in _COMBINED_MODES and mode != _READ_FILES_MODE:
        return GUEST_GUARD_PROGRAM
    program = GUEST_GUARD_PROGRAM.removesuffix("\nmain()\n")
    if program == GUEST_GUARD_PROGRAM:
        raise RuntimeError("The guest guard program must end by calling main().")
    program += "\n" + GUEST_COMBINED_MODES_SOURCE
    if mode == _READ_FILES_MODE:
        return program + "\n" + GUEST_READ_FILES_SOURCE + "\nread_files_main()\n"
    return program + "\ncombined_main()\n"


def guard_read_limit(runner: Runner) -> int | None:
    """The most file bytes one guarded read can return through ``runner``.

    Reads come back base64-encoded after a status line, so a runner that caps
    command output caps reads at about three quarters of that. ``None`` means
    the runner declares no cap.
    """

    output_cap = runner.max_exec_output_bytes
    if output_cap is None:
        return None
    return max(1, (output_cap - _READ_OUTPUT_HEADROOM_BYTES) // 4 * 3)


async def guard_read(
    runner: Runner,
    *,
    root: str,
    rel_path: str,
    offset: int,
    limit: int,
    original_path: str,
    backend: str,
    timeout_s: int | None = None,
    python_executable: str = GUEST_PYTHON,
) -> tuple[bytes, int, str | None, str | None, WorkspaceGitMode | None]:
    """Atomically resolve-and-read a contained file and complete-snapshot identity.

    A ``limit`` larger than the runner can return is lowered to fit, so a
    large request comes back as a shorter page rather than cut-off output.
    """

    runner_limit = guard_read_limit(runner)
    if runner_limit is not None:
        limit = min(limit, runner_limit)
    output_limit = 4 * ((limit + 2) // 3) + _READ_OUTPUT_HEADROOM_BYTES
    result = await _exec_guard(
        runner,
        "read",
        root,
        rel_path,
        str(offset),
        str(limit),
        timeout_s=timeout_s,
        output_limit_bytes=output_limit,
        python_executable=python_executable,
    )
    status, payload = _guard_status(
        result, mode="read", backend=backend, original_path=original_path
    )
    if status in {_STATUS_ENOENT, _STATUS_NOTFILE, _STATUS_NOTDIR}:
        raise FileNotFoundError(f"Workspace file not found: {original_path}")
    if status.startswith(f"{_STATUS_OFFSET} "):
        total_bytes = status.partition(" ")[2]
        if total_bytes.isdigit():
            raise WorkspaceReadOffsetError(offset, int(total_bytes))
    _raise_common_status(status, mode="read", backend=backend, original_path=original_path)
    total_bytes, revision, digest, git_mode = _parse_ok_read(
        status, backend=backend, original_path=original_path
    )
    try:
        content = base64.b64decode(payload.strip(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(
            f"{backend} workspace guard returned an invalid read payload: {original_path}"
        ) from exc
    return content, total_bytes, revision, digest, git_mode


async def guard_locate_lines(
    runner: Runner,
    *,
    root: str,
    rel_path: str,
    start_line: int,
    end_line: int | None,
    original_path: str,
    backend: str,
    timeout_s: int | None = None,
    python_executable: str = GUEST_PYTHON,
) -> WorkspaceLineSpan:
    """Find a line range's byte offsets inside the guest, returning no file content."""

    _validate_line_range(start_line, end_line)
    result = await _exec_guard(
        runner,
        "lines",
        root,
        rel_path,
        str(start_line),
        str(end_line or 0),
        timeout_s=timeout_s,
        output_limit_bytes=_READ_OUTPUT_HEADROOM_BYTES,
        python_executable=python_executable,
    )
    status, _ = _guard_status(result, mode="read", backend=backend, original_path=original_path)
    if status in {_STATUS_ENOENT, _STATUS_NOTFILE, _STATUS_NOTDIR}:
        raise FileNotFoundError(f"Workspace file not found: {original_path}")
    _raise_common_status(status, mode="read", backend=backend, original_path=original_path)
    parts = status.split()
    if len(parts) == 5 and parts[0] == _STATUS_OK and all(part.isdigit() for part in parts[1:]):
        total_bytes, start_offset, end_offset, total_lines = (int(part) for part in parts[1:])
        try:
            return WorkspaceLineSpan(
                start_offset=start_offset,
                end_offset=end_offset,
                total_bytes=total_bytes,
                total_lines=total_lines,
            )
        except ValueError:
            pass
    raise RuntimeError(
        f"Failed to read {backend} workspace file: "
        f"{original_path}: unexpected guard status {status!r}"
    )


def _validate_line_range(start_line: int, end_line: int | None) -> None:
    if type(start_line) is not int or start_line < 1:
        raise ValueError("start_line must be a positive integer.")
    if end_line is not None and (type(end_line) is not int or end_line < start_line):
        raise ValueError("end_line must be an integer no less than start_line.")


async def guard_read_lines(
    runner: Runner,
    *,
    root: str,
    rel_path: str,
    start_line: int,
    end_line: int | None,
    before_bytes: int,
    after_bytes: int,
    limit: int,
    head_limit: int,
    original_path: str,
    backend: str,
    timeout_s: int | None = None,
    python_executable: str = GUEST_PYTHON,
) -> WorkspaceLineRead:
    """Locate a line range, read a window around it and the file's head, in one command.

    Like ``guard_read``, the window and head are lowered to what the runner
    can return; together they fit one command's output.
    """

    _validate_line_range(start_line, end_line)
    runner_limit = guard_read_limit(runner)
    if runner_limit is not None:
        # The head comes first; the window gets what output is left.
        head_limit = min(head_limit, runner_limit)
        limit = min(limit, max(1, runner_limit - head_limit))
    output_limit = (
        4 * ((limit + 2) // 3) + 4 * ((head_limit + 2) // 3) + _READ_OUTPUT_HEADROOM_BYTES
    )
    result = await _exec_guard(
        runner,
        "read-lines",
        root,
        rel_path,
        str(start_line),
        str(end_line or 0),
        str(before_bytes),
        str(after_bytes),
        str(limit),
        str(head_limit),
        timeout_s=timeout_s,
        output_limit_bytes=output_limit,
        python_executable=python_executable,
    )
    status, payload = _guard_status(
        result, mode="read", backend=backend, original_path=original_path
    )
    if status in {_STATUS_ENOENT, _STATUS_NOTFILE, _STATUS_NOTDIR}:
        raise FileNotFoundError(f"Workspace file not found: {original_path}")
    _raise_common_status(status, mode="read", backend=backend, original_path=original_path)
    try:
        metadata_line, *encoded = payload.splitlines()
        metadata = json.loads(metadata_line)
        start_offset, end_offset, total_bytes, total_lines = metadata["span"]
        span = WorkspaceLineSpan(
            start_offset=start_offset,
            end_offset=end_offset,
            total_bytes=total_bytes,
            total_lines=total_lines,
        )
        reads = [
            _guest_read_result(offset, identity, data, total_bytes)
            for (offset, identity), data in zip(metadata["reads"], encoded, strict=True)
        ]
        return WorkspaceLineRead(span=span, window=reads[0], head=reads[1] if reads[1:] else None)
    except (TypeError, ValueError, KeyError, binascii.Error) as exc:
        raise RuntimeError(
            f"{backend} workspace guard returned an invalid line read: {original_path}"
        ) from exc


async def guard_edit_text(
    runner: Runner,
    *,
    root: str,
    rel_path: str,
    replacements: Sequence[WorkspaceTextReplacement],
    expected_revision: str,
    max_bytes: int,
    original_path: str,
    backend: str,
    timeout_s: int | None = None,
    python_executable: str = GUEST_PYTHON,
) -> WorkspaceTextEditResult | None:
    """Read a contained file and, when the edit applies, replace it, in one command.

    The read is lowered to what the runner can return, like ``guard_read``.
    Returns ``None``, running nothing, when the edit text is too large to send
    with the command; the caller then reads and replaces in separate steps.
    """

    limit = max_bytes
    runner_limit = guard_read_limit(runner)
    if runner_limit is not None:
        limit = min(limit, runner_limit)
    edits = json.dumps(
        [
            [replacement.old_text, replacement.new_text, replacement.expected_replacements]
            for replacement in replacements
        ],
        ensure_ascii=False,
    )
    if len(edits.encode("utf-8")) > _COMBINED_EDIT_STDIN_LIMIT_BYTES:
        return None
    result = await _exec_guard(
        runner,
        "edit-text",
        root,
        rel_path,
        expected_revision,
        str(limit),
        str(max_bytes),
        stdin=edits,
        timeout_s=timeout_s,
        output_limit_bytes=4 * ((limit + 2) // 3) + _READ_OUTPUT_HEADROOM_BYTES,
        python_executable=python_executable,
    )
    status, payload = _guard_status(
        result, mode="edit", backend=backend, original_path=original_path
    )
    if status.startswith(f"{_STATUS_STALE} "):
        raise WorkspaceRevisionMismatchError(expected_revision, status.partition(" ")[2])
    if status in {_STATUS_ENOENT, _STATUS_NOTFILE, _STATUS_NOTDIR}:
        raise FileNotFoundError(f"Workspace file not found: {original_path}")
    _raise_common_status(status, mode="edit", backend=backend, original_path=original_path)
    try:
        read_line, mutation_line, encoded = payload.splitlines()
        metadata = json.loads(read_line)
        mutation = json.loads(mutation_line)
        return WorkspaceTextEditResult(
            read=_guest_read_result(0, metadata["identity"], encoded, metadata["total_bytes"]),
            mutation=None if mutation is None else WorkspaceMutationResult(**mutation),
        )
    except (TypeError, ValueError, KeyError, binascii.Error) as exc:
        raise RuntimeError(
            f"{backend} workspace guard returned an invalid edit result: {original_path}"
        ) from exc


# Output beyond the files' bytes, per file: its outcome's JSON (two reads'
# identities, or an error's text) and its base64 lines' padding and newlines.
_READ_FILES_OUTPUT_HEADROOM_PER_FILE_BYTES = 768


async def guard_read_files(
    runner: Runner,
    *,
    root: str,
    requests: Sequence[WorkspaceFileReadRequest],
    backend: str,
    timeout_s: int | None = None,
    python_executable: str = GUEST_PYTHON,
) -> tuple[WorkspaceFileReadOutcome, ...]:
    """Read several contained files in one command, each as its single read would.

    Each outcome is the request's result or the exception its ``guard_read``
    or ``guard_read_lines`` would raise. When the runner caps command output,
    the files share what fits, and a file given less than it asked for comes
    back as a shorter read; when even the outcomes would not fit, every
    outcome is ``None`` and nothing runs. Raises when the command fails as a
    whole.
    """

    requests = tuple(requests)
    if not requests:
        return ()
    headroom = _READ_OUTPUT_HEADROOM_BYTES + len(requests) * (
        _READ_FILES_OUTPUT_HEADROOM_PER_FILE_BYTES
    )
    asked = sum(request.max_bytes + request.head_bytes for request in requests)
    total: int | None = None
    output_cap = runner.max_exec_output_bytes
    if output_cap is not None:
        if output_cap <= headroom:
            return (None,) * len(requests)
        fits = (output_cap - headroom) // 4 * 3
        if asked > fits:
            total = fits
    output_limit = 4 * (((asked if total is None else total) + 2) // 3) + headroom
    batch = {
        "requests": [
            (
                [
                    "lines",
                    request.path,
                    request.start_line,
                    request.end_line or 0,
                    request.before_bytes,
                    request.after_bytes,
                    request.max_bytes,
                    request.head_bytes,
                ]
                if request.reads_lines
                else ["bytes", request.path, request.offset, request.max_bytes]
            )
            for request in requests
        ],
        "total": total,
    }
    label = f"{len(requests)} files"
    result = await _exec_guard(
        runner,
        _READ_FILES_MODE,
        root,
        "-",
        stdin=json.dumps(batch, ensure_ascii=False),
        timeout_s=timeout_s,
        output_limit_bytes=output_limit,
        python_executable=python_executable,
    )
    status, payload = _guard_status(result, mode="read", backend=backend, original_path=label)
    if status != _STATUS_OK:
        _raise_common_status(status, mode="read", backend=backend, original_path=label)
        raise RuntimeError(
            f"Failed to read {backend} workspace files: unexpected guard status {status!r}"
        )
    try:
        outcomes_line, *encoded = payload.splitlines()
        outcomes = json.loads(outcomes_line)
        if type(outcomes) is not list or len(outcomes) != len(requests):
            raise ValueError("one outcome per request")
        blobs = iter(encoded)
        parsed = tuple(
            _guest_file_outcome(request, outcome, blobs, backend=backend)
            for request, outcome in zip(requests, outcomes, strict=True)
        )
        if next(blobs, None) is not None:
            raise ValueError("unclaimed read payload")
        return parsed
    except (TypeError, ValueError, KeyError, StopIteration, binascii.Error) as exc:
        raise RuntimeError(f"{backend} workspace guard returned an invalid batch read") from exc


def _guest_file_outcome(
    request: WorkspaceFileReadRequest,
    outcome: Any,
    blobs: Iterator[str],
    *,
    backend: str,
) -> WorkspaceFileReadOutcome:
    """One file of a batch read: its result, or the exception its own read raises.

    ``None`` when its share of the output left a read empty before the end of
    the file, so the caller reads it on its own.
    """

    status = outcome["s"]
    path = request.path
    if status in {_STATUS_ENOENT, _STATUS_NOTFILE, _STATUS_NOTDIR}:
        return FileNotFoundError(f"Workspace file not found: {path}")
    if status == _STATUS_OFFSET and not request.reads_lines:
        return WorkspaceReadOffsetError(request.offset, _guest_int(outcome["total"]))
    if status == "error":
        detail = outcome["detail"]
        return RuntimeError(f"Failed to read {backend} workspace file: {path}: {detail}")
    if status != _STATUS_OK:
        try:
            _raise_common_status(str(status), mode="read", backend=backend, original_path=path)
        except (ValueError, RuntimeError) as exc:
            return exc
        raise ValueError("unexpected file status")
    reads = outcome["reads"]
    if not request.reads_lines:
        ((offset, identity),) = reads
        return _guest_batch_read(offset, identity, next(blobs), _guest_int(outcome["total"]))
    start_offset, end_offset, total_bytes, total_lines = (
        _guest_int(value) for value in outcome["span"]
    )
    span = WorkspaceLineSpan(
        start_offset=start_offset,
        end_offset=end_offset,
        total_bytes=total_bytes,
        total_lines=total_lines,
    )
    if not 1 <= len(reads) <= 2:
        raise ValueError("a line read has a window and at most a head")
    results = [
        _guest_batch_read(offset, identity, next(blobs), total_bytes) for offset, identity in reads
    ]
    if any(result is None for result in results):
        return None
    window, *head = cast("list[WorkspaceReadResult]", results)
    return WorkspaceLineRead(span=span, window=window, head=head[0] if head else None)


def _guest_int(value: object) -> int:
    if type(value) is not int:
        raise TypeError("expected an integer")
    return value


def _guest_read_result(
    offset: int, identity: Sequence[str | None], encoded: str, total_bytes: int
) -> WorkspaceReadResult:
    content = base64.b64decode(encoded.strip(), validate=True)
    return _guest_read(offset, identity, content, total_bytes)


def _guest_batch_read(
    offset: int, identity: Sequence[str | None], encoded: str, total_bytes: int
) -> WorkspaceReadResult | None:
    """A batch read, or ``None`` when its share left it empty before the end of the file."""

    content = base64.b64decode(encoded.strip(), validate=True)
    if not content and offset < total_bytes:
        return None
    return _guest_read(offset, identity, content, total_bytes)


def _guest_read(
    offset: int, identity: Sequence[str | None], content: bytes, total_bytes: int
) -> WorkspaceReadResult:
    revision, digest, git_mode = identity
    if git_mode not in {None, "100644", "100755"}:
        raise ValueError("invalid Git mode")
    return WorkspaceReadResult(
        content=content,
        total_bytes=max(total_bytes, offset + len(content)),
        truncated=total_bytes > offset + len(content),
        offset=offset,
        revision=revision,
        sha256=digest,
        git_mode=cast("WorkspaceGitMode | None", git_mode),
    )


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


async def guard_create(
    runner: Runner,
    *,
    root: str,
    rel_path: str,
    content: bytes,
    original_path: str,
    backend: str,
    timeout_s: int | None = None,
    python_executable: str = GUEST_PYTHON,
) -> WorkspaceMutationResult:
    return await _guard_conditional_mutation(
        runner,
        "create",
        root=root,
        rel_path=rel_path,
        content=content,
        expected_revision=None,
        original_path=original_path,
        backend=backend,
        timeout_s=timeout_s,
        python_executable=python_executable,
    )


async def guard_replace(
    runner: Runner,
    *,
    root: str,
    rel_path: str,
    content: bytes,
    expected_revision: str,
    original_path: str,
    backend: str,
    timeout_s: int | None = None,
    python_executable: str = GUEST_PYTHON,
) -> WorkspaceMutationResult:
    return await _guard_conditional_mutation(
        runner,
        "replace",
        root=root,
        rel_path=rel_path,
        content=content,
        expected_revision=expected_revision,
        original_path=original_path,
        backend=backend,
        timeout_s=timeout_s,
        python_executable=python_executable,
    )


async def guard_delete_if_revision(
    runner: Runner,
    *,
    root: str,
    rel_path: str,
    expected_revision: str,
    original_path: str,
    backend: str,
    timeout_s: int | None = None,
    python_executable: str = GUEST_PYTHON,
) -> WorkspaceMutationResult:
    return await _guard_conditional_mutation(
        runner,
        "delete-if-revision",
        root=root,
        rel_path=rel_path,
        content=None,
        expected_revision=expected_revision,
        original_path=original_path,
        backend=backend,
        timeout_s=timeout_s,
        python_executable=python_executable,
    )


async def guard_require_absent(
    runner: Runner,
    *,
    root: str,
    rel_path: str,
    original_path: str,
    backend: str,
    timeout_s: int | None = None,
    python_executable: str = GUEST_PYTHON,
) -> None:
    result = await _exec_guard(
        runner,
        "require-absent",
        root,
        rel_path,
        timeout_s=timeout_s,
        python_executable=python_executable,
    )
    status, _ = _guard_status(
        result,
        mode="require absent",
        backend=backend,
        original_path=original_path,
    )
    if status in {_STATUS_OK, _STATUS_ENOENT}:
        return
    if status == _STATUS_EXISTS:
        raise FileExistsError(f"Workspace file already exists: {original_path}")
    if status == _STATUS_UNSUPPORTED:
        raise WorkspacePreconditionUnsupportedError(
            f"{backend} workspace cannot authoritatively prove target absence."
        )
    if status in {_STATUS_NOTDIR, _STATUS_NOTFILE}:
        raise FileNotFoundError(f"Workspace path is not creatable: {original_path}")
    _raise_common_status(
        status,
        mode="require absent",
        backend=backend,
        original_path=original_path,
    )


async def guard_move_if_revision(
    runner: Runner,
    *,
    root: str,
    source_rel_path: str,
    destination_rel_path: str,
    expected_revision: str,
    original_source_path: str,
    original_destination_path: str,
    backend: str,
    timeout_s: int | None = None,
    python_executable: str = GUEST_PYTHON,
) -> WorkspaceMoveResult:
    result = await _exec_guard(
        runner,
        "move-if-revision",
        root,
        source_rel_path,
        destination_rel_path,
        expected_revision,
        timeout_s=timeout_s,
        python_executable=python_executable,
    )
    status, payload = _guard_status(
        result,
        mode="move",
        backend=backend,
        original_path=original_source_path,
    )
    if status == _STATUS_EXISTS:
        raise FileExistsError(f"Workspace file already exists: {original_destination_path}")
    if status.startswith(f"{_STATUS_STALE} "):
        raise WorkspaceRevisionMismatchError(
            expected_revision,
            status.partition(" ")[2],
        )
    if status in {_STATUS_ENOENT, _STATUS_NOTFILE, _STATUS_NOTDIR}:
        raise FileNotFoundError(f"Workspace file not found: {original_source_path}")
    if status == _STATUS_HARDLINK:
        raise WorkspaceMoveUnsupportedError(
            f"{backend} workspace move refuses a hard-linked source."
        )
    if status == _STATUS_UNSUPPORTED:
        raise WorkspaceMoveUnsupportedError(
            f"{backend} workspace cannot provide a conditional same-workspace move."
        )
    if status not in {_STATUS_OK, _STATUS_AMBIGUOUS}:
        _raise_common_status(
            status,
            mode="move",
            backend=backend,
            original_path=original_source_path,
        )
    move = _parse_move_payload(
        payload,
        backend=backend,
        original_path=original_source_path,
    )
    if status == _STATUS_AMBIGUOUS:
        raise WorkspaceMoveAmbiguousError(move)
    return move


def _parse_move_payload(
    payload: str,
    *,
    backend: str,
    original_path: str,
) -> WorkspaceMoveResult:
    try:
        decoded = json.loads(payload)
        return WorkspaceMoveResult(**decoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"{backend} workspace guard returned invalid move metadata: {original_path}"
        ) from exc


async def _guard_conditional_mutation(
    runner: Runner,
    mode: str,
    *,
    root: str,
    rel_path: str,
    content: bytes | None,
    expected_revision: str | None,
    original_path: str,
    backend: str,
    timeout_s: int | None,
    python_executable: str,
) -> WorkspaceMutationResult:
    extra_args = (expected_revision,) if expected_revision is not None else ()
    stdin = base64.b64encode(content).decode("ascii") if content is not None else None
    result = await _exec_guard(
        runner,
        mode,
        root,
        rel_path,
        *extra_args,
        stdin=stdin,
        timeout_s=timeout_s,
        python_executable=python_executable,
    )
    status, payload = _guard_status(result, mode=mode, backend=backend, original_path=original_path)
    if status == _STATUS_EXISTS:
        raise FileExistsError(f"Workspace file already exists: {original_path}")
    if status.startswith(f"{_STATUS_STALE} "):
        actual_revision = status.partition(" ")[2]
        assert expected_revision is not None
        raise WorkspaceRevisionMismatchError(expected_revision, actual_revision)
    if status in {_STATUS_ENOENT, _STATUS_NOTFILE, _STATUS_NOTDIR}:
        raise FileNotFoundError(f"Workspace file not found: {original_path}")
    _raise_common_status(status, mode=mode, backend=backend, original_path=original_path)
    try:
        decoded = json.loads(payload)
        return WorkspaceMutationResult(**decoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"{backend} workspace guard returned invalid mutation metadata: {original_path}"
        ) from exc


async def _exec_guard(
    runner: Runner,
    mode: str,
    root: str,
    rel_path: str,
    *extra_args: str,
    stdin: str | None = None,
    timeout_s: int | None = None,
    output_limit_bytes: int | None = None,
    python_executable: str = GUEST_PYTHON,
) -> ExecResult:
    command = ExecCommand.process(
        python_executable, "-c", _guard_program(mode), mode, root, rel_path, *extra_args
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


def _parse_ok_read(
    status: str, *, backend: str, original_path: str
) -> tuple[int, str | None, str | None, WorkspaceGitMode | None]:
    parts = status.split()
    if len(parts) == 5 and parts[0] == _STATUS_OK and parts[1].isdigit():
        revision = None if parts[2] == "-" else parts[2]
        digest = None if parts[3] == "-" else parts[3]
        git_mode = None if parts[4] == "-" else parts[4]
        if git_mode not in {None, "100644", "100755"}:
            raise RuntimeError(
                f"Failed to read {backend} workspace file: "
                f"{original_path}: unexpected guard status {status!r}"
            )
        return int(parts[1]), revision, digest, cast("WorkspaceGitMode | None", git_mode)
    raise RuntimeError(
        f"Failed to read {backend} workspace file: "
        f"{original_path}: unexpected guard status {status!r}"
    )
