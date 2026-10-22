"""Recoverable publication of directories owned by the filesystem snapshot CAS."""

from __future__ import annotations

import errno
import json
import os
import stat
import unicodedata
import uuid
from contextlib import suppress
from pathlib import Path

PENDING_FILE = ".cayu-snapshot-directory.pending"
LOCK_FILE = ".cayu-snapshot-directory.lock"
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)


def identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def require_named_directory(parent_fd: int, name: str, descriptor: int) -> None:
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(named.st_mode) or identity(named) != identity(os.fstat(descriptor)):
        raise OSError(errno.ESTALE, "Snapshot storage directory changed during publication.")


def _has_pending(descriptor: int) -> bool:
    if os.fstat(descriptor).st_uid != os.geteuid():
        return False
    try:
        marker = os.stat(PENDING_FILE, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return marker.st_uid == os.geteuid()


def _pending_bytes(parent_fd: int, name: str, descriptor: int) -> bytes:
    return json.dumps(
        {
            "version": 2,
            "parent": os.fstat(parent_fd).st_ino,
            "name": unicodedata.normalize("NFC", name).casefold(),
            "directory": os.fstat(descriptor).st_ino,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_owned_file(descriptor: int) -> None:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid():
        raise OSError(errno.EINVAL, "Invalid snapshot directory publication metadata.")


def _recover_child(parent_fd: int, name: str, descriptor: int) -> None:
    if not _has_pending(descriptor):
        return
    # This module is called only behind the existing POSIX CAS capability guard.
    # Keep the optional module out of import-time paths on Windows.
    import fcntl

    lock = os.open(LOCK_FILE, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
    try:
        _require_owned_file(lock)
        fcntl.flock(lock, fcntl.LOCK_EX)
        require_named_directory(parent_fd, name, descriptor)
        # A child staged in this parent cannot cross a filesystem boundary.
        # The live device relationship matters; a reboot may renumber it.
        if os.fstat(parent_fd).st_dev != os.fstat(descriptor).st_dev:
            raise OSError(errno.EINVAL, "Snapshot directory publication crossed filesystems.")
        try:
            marker = os.open(PENDING_FILE, _FILE_FLAGS, dir_fd=descriptor)
        except FileNotFoundError:
            return
        try:
            _require_owned_file(marker)
            expected = _pending_bytes(parent_fd, name, descriptor)
            if os.read(marker, len(expected) + 1) != expected:
                raise OSError(errno.EINVAL, "Invalid snapshot directory publication marker.")
            os.fsync(marker)
        finally:
            os.close(marker)
        os.fsync(parent_fd)
        require_named_directory(parent_fd, name, descriptor)
        os.unlink(PENDING_FILE, dir_fd=descriptor)
        os.fsync(descriptor)
        require_named_directory(parent_fd, name, descriptor)
    finally:
        os.close(lock)


def _create_metadata(descriptor: int, name: str, content: bytes) -> None:
    output = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=descriptor)
    try:
        remaining = memoryview(content)
        while remaining:
            written = os.write(output, remaining)
            if written <= 0:
                raise OSError("Snapshot directory marker write made no progress.")
            remaining = remaining[written:]
        os.fsync(output)
    finally:
        os.close(output)


def prepare_child(parent_fd: int, name: str, *, mode: int = 0o777) -> int:
    """Return an owned descriptor after this child entry is durably published."""
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        descriptor = None
    if descriptor is not None:
        try:
            _recover_child(parent_fd, name, descriptor)
            require_named_directory(parent_fd, name, descriptor)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    temporary = ".cayu-snapshot-directory-" + uuid.uuid4().hex
    os.mkdir(temporary, mode=mode, dir_fd=parent_fd)
    staged = os.open(temporary, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    descriptor = None
    try:
        _create_metadata(staged, LOCK_FILE, b"")
        _create_metadata(staged, PENDING_FILE, _pending_bytes(parent_fd, name, staged))
        os.fsync(staged)
        require_named_directory(parent_fd, temporary, staged)
        published = False
        try:
            # A cooperative winner stays nonempty because its lock is permanent.
            os.rename(temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            published = True
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        if published and identity(os.fstat(descriptor)) != identity(os.fstat(staged)):
            raise OSError(errno.ESTALE, "Snapshot staging directory changed during publication.")
        _recover_child(parent_fd, name, descriptor)
        require_named_directory(parent_fd, name, descriptor)
        return descriptor
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise
    finally:
        # Only the unique, still-unpublished staging entry belongs to this call.
        try:
            try:
                cleanup = os.open(temporary, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            else:
                try:
                    if identity(os.fstat(cleanup)) != identity(os.fstat(staged)):
                        raise OSError(
                            errno.ESTALE, "Snapshot staging directory changed during cleanup."
                        )
                    for filename in (PENDING_FILE, LOCK_FILE):
                        with suppress(FileNotFoundError):
                            os.unlink(filename, dir_fd=cleanup)
                finally:
                    os.close(cleanup)
                os.rmdir(temporary, dir_fd=parent_fd)
        finally:
            os.close(staged)


def prepare_directory(path: Path) -> tuple[int, int]:
    """Prepare an owned chain beneath an existing durable external anchor.

    Internal resolution preserves existing aliases without changing the caller's
    advertised path. Only our pending components require their parent to reopen;
    unmarked existing anchors do not require access to unrelated ancestors.
    """
    missing: list[str] = []
    anchor = path
    while not anchor.exists():
        if anchor.is_symlink():
            raise FileExistsError(errno.EEXIST, "Snapshot storage alias has no existing target.")
        missing.append(anchor.name)
        anchor = anchor.parent
    anchor = anchor.resolve(strict=True)
    descriptor = os.open(anchor, _DIRECTORY_FLAGS)
    try:
        if _has_pending(descriptor):
            parent = os.open(anchor.parent, _DIRECTORY_FLAGS)
            try:
                _recover_child(parent, anchor.name, descriptor)
            finally:
                os.close(parent)
        for name in reversed(missing):
            child = prepare_child(descriptor, name)
            os.close(descriptor)
            descriptor = child
        os.fsync(descriptor)
        result = identity(os.fstat(descriptor))
        if identity(path.stat()) != result:
            raise OSError(errno.ESTALE, "Snapshot storage root changed during initialization.")
        return result
    finally:
        os.close(descriptor)
