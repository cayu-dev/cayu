"""Descriptor-relative containment for path-addressed ``LocalWorkspace`` operations.

The configured root is trusted and opened once per operation. Every component below it is
opened with ``O_NOFOLLOW`` relative to the preceding directory descriptor, and final reads or
mutations stay relative to the pinned parent. A hostile same-user process may still access host
paths directly; this guard only prevents it from redirecting a workspace API operation through
concurrent symlink replacement.
"""

from __future__ import annotations

import errno
import hashlib
import os
import secrets
import stat
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import BinaryIO, NoReturn

from cayu.workspaces._mutations import move_result_from_identity
from cayu.workspaces.base import (
    WorkspaceGitModeMismatchError,
    WorkspaceMoveAmbiguousError,
    WorkspaceMoveResult,
    WorkspaceMoveUnsupportedError,
    WorkspaceRevisionMismatchError,
)

_ESCAPE_ERRNOS = (errno.ELOOP, errno.EMLINK)
_MISSING_ERRNOS = (errno.ENOENT, errno.ENOTDIR)
_OPEN_BASE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
_NONBLOCK_FLAG = getattr(os, "O_NONBLOCK", 0)
_TEMP_OPEN_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
if hasattr(os, "O_PATH"):
    _SEARCH_BASE_FLAGS = os.O_PATH | getattr(os, "O_CLOEXEC", 0)
elif hasattr(os, "O_SEARCH"):
    _SEARCH_BASE_FLAGS = os.O_SEARCH | getattr(os, "O_CLOEXEC", 0)
elif sys.platform == "darwin":
    # CPython does not expose Darwin's O_SEARCH even though the kernel supports it.
    _SEARCH_BASE_FLAGS = 0x40000000 | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
else:
    _SEARCH_BASE_FLAGS = _OPEN_BASE_FLAGS

_SUPPORTS_DIR_FD = all(
    operation in os.supports_dir_fd
    for operation in (os.open, os.stat, os.mkdir, os.link, os.unlink, os.rename)
)
_SUPPORTS_NOFOLLOW_STAT = os.stat in os.supports_follow_symlinks
_SUPPORTS_NOFOLLOW_LINK = os.link in os.supports_follow_symlinks


class _LocalGuardPathError(Exception):
    def __init__(self, status: str) -> None:
        self.status = status
        super().__init__(status)


class _LocalGuardStagingCleanupError(RuntimeError):
    """Retain the exact directory owner when atomic staging cannot be removed."""

    def __init__(
        self,
        parent_fd: int,
        temp_name: str,
        failures: tuple[BaseException, ...],
    ) -> None:
        self._lock = threading.Lock()
        self._temp_name = temp_name
        try:
            self._parent_fd: int | None = os.dup(parent_fd)
        except BaseException as owner_error:
            self._parent_fd = None
            failures = (*failures, owner_error)
        self.failures = failures
        super().__init__("Workspace atomic staging cleanup did not complete.")

    def retry_cleanup(self) -> bool:
        """Retry against the pinned parent and release it after positive absence."""

        with self._lock:
            parent_fd = self._parent_fd
            if parent_fd is None:
                return False
            absent, _failures = _unlink_staging_and_inspect(parent_fd, self._temp_name)
            if not absent:
                return False
            self._parent_fd = None
            with suppress(OSError):
                os.close(parent_fd)
            return True

    @property
    def cleanup_owned(self) -> bool:
        with self._lock:
            return self._parent_fd is not None

    @property
    def staging_name(self) -> str:
        """Return the exact private staging name owned by this cleanup."""

        return self._temp_name

    def release_cleanup_owner(self) -> None:
        """Release the descriptor after a wider owner removed the containing tree."""

        with self._lock:
            parent_fd = self._parent_fd
            self._parent_fd = None
        if parent_fd is not None:
            with suppress(OSError):
                os.close(parent_fd)

    def __del__(self) -> None:
        self.release_cleanup_owner()


class _LocalGuardStagingConflictError(RuntimeError):
    """A durable staging name exists without the expected owned content."""


def _require_descriptor_guard_support() -> None:
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "fchmod")
        or not _SUPPORTS_DIR_FD
        or not _SUPPORTS_NOFOLLOW_STAT
        or not _SUPPORTS_NOFOLLOW_LINK
    ):
        raise RuntimeError(
            "LocalWorkspace requires POSIX descriptor-relative filesystem primitives."
        )


def _classify_missing(name: str, dir_fd: int) -> str:
    try:
        info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except OSError:
        return "missing"
    if stat.S_ISLNK(info.st_mode):
        return "escape"
    return "notdir"


def _open_directory(name: str, dir_fd: int, *, create: bool) -> int:
    flags = _SEARCH_BASE_FLAGS | os.O_NOFOLLOW | os.O_DIRECTORY
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in _ESCAPE_ERRNOS:
            raise _LocalGuardPathError("escape") from exc
        if exc.errno == errno.ENOENT and create:
            with suppress(FileExistsError):
                os.mkdir(name, mode=0o777, dir_fd=dir_fd)
            return _open_directory(name, dir_fd, create=False)
        if exc.errno in _MISSING_ERRNOS:
            raise _LocalGuardPathError(_classify_missing(name, dir_fd)) from exc
        raise


@contextmanager
def _open_parent(
    root: Path,
    relative_path: str,
    *,
    create: bool = False,
) -> Iterator[tuple[int, str]]:
    _require_descriptor_guard_support()
    parts = relative_path.split("/")
    try:
        root_fd = os.open(
            root,
            _SEARCH_BASE_FLAGS | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
    except OSError as exc:
        if exc.errno in _ESCAPE_ERRNOS:
            raise _LocalGuardPathError("escape") from exc
        if exc.errno in _MISSING_ERRNOS:
            raise _LocalGuardPathError("missing") from exc
        raise
    current_fd = root_fd
    try:
        for name in parts[:-1]:
            next_fd = _open_directory(name, current_fd, create=create)
            os.close(current_fd)
            current_fd = next_fd
        yield current_fd, parts[-1]
    finally:
        os.close(current_fd)


def _open_regular(parent_fd: int, name: str) -> tuple[int, os.stat_result]:
    try:
        descriptor = os.open(
            name,
            _OPEN_BASE_FLAGS | _NONBLOCK_FLAG | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        if exc.errno in _ESCAPE_ERRNOS:
            raise _LocalGuardPathError("escape") from exc
        if exc.errno in _MISSING_ERRNOS:
            raise _LocalGuardPathError(_classify_missing(name, parent_fd)) from exc
        raise
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise _LocalGuardPathError("notfile")
    return descriptor, info


def _identity_at(
    parent_fd: int,
    name: str,
    *,
    require_single_link: bool = False,
) -> tuple[tuple[str, str, int], int]:
    descriptor, info = _open_regular(parent_fd, name)
    if require_single_link and info.st_nlink != 1:
        os.close(descriptor)
        raise WorkspaceMoveUnsupportedError(
            "Workspace move refuses a hard-linked source whose alias metadata is unrepresentable."
        )
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(descriptor, "rb") as source:
        while chunk := source.read(1 << 16):
            digest.update(chunk)
            size += len(chunk)
    hexdigest = digest.hexdigest()
    return (f"sha256:{hexdigest}", hexdigest, size), stat.S_IMODE(info.st_mode)


def _inspect_regular_target_mode(parent_fd: int, name: str) -> int | None:
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode):
        raise _LocalGuardPathError("escape")
    if not stat.S_ISREG(info.st_mode):
        raise _LocalGuardPathError("notfile")
    return stat.S_IMODE(info.st_mode)


def _write_temp(
    parent_fd: int,
    name: str,
    content: bytes,
    *,
    mode: int | None,
    staging_name: str | None = None,
) -> str:
    descriptor: int | None = None
    temp_name: str | None = None
    for _attempt in range(1 if staging_name is not None else 100):
        candidate = (
            staging_name if staging_name is not None else f".{name}.cayu-{secrets.token_hex(12)}"
        )
        try:
            descriptor = os.open(candidate, _TEMP_OPEN_FLAGS, 0o666, dir_fd=parent_fd)
        except FileExistsError:
            if staging_name is not None:
                raise
            continue
        temp_name = candidate
        break
    if descriptor is None or temp_name is None:
        raise OSError("Could not allocate an atomic workspace temporary file.")
    try:
        with os.fdopen(descriptor, "wb") as temp:
            temp.write(content)
            if mode is not None:
                os.fchmod(temp.fileno(), mode)
    except BaseException as primary:
        absent, cleanup_failures = _unlink_staging_and_inspect(parent_fd, temp_name)
        if not absent:
            _raise_staging_cleanup_error(
                parent_fd,
                temp_name,
                (primary, *cleanup_failures),
            )
        if _contains_fatal_cleanup_failure(cleanup_failures):
            _raise_operation_and_cleanup_failures(primary, cleanup_failures)
        raise
    return temp_name


def _unlink_staging_and_inspect(
    parent_fd: int,
    temp_name: str,
) -> tuple[bool, tuple[BaseException, ...]]:
    """Try twice, then prove whether one private staging name still exists."""

    failures: list[BaseException] = []
    for _attempt in range(2):
        try:
            os.unlink(temp_name, dir_fd=parent_fd)
        except FileNotFoundError:
            return True, tuple(failures)
        except BaseException as error:
            failures.append(error)
        else:
            return True, tuple(failures)
    try:
        os.stat(temp_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return True, tuple(failures)
    except BaseException as error:
        failures.append(error)
    return False, tuple(failures)


def _contains_fatal_cleanup_failure(failures: tuple[BaseException, ...]) -> bool:
    return any(not isinstance(error, Exception) for error in failures)


def _raise_operation_and_cleanup_failures(
    primary: BaseException | None,
    cleanup_failures: tuple[BaseException, ...],
) -> NoReturn:
    failures = cleanup_failures if primary is None else (primary, *cleanup_failures)
    if len(failures) == 1:
        raise failures[0]
    raise BaseExceptionGroup(
        "Workspace staging operation and cleanup failures.",
        list(failures),
    )


def _raise_staging_cleanup_error(
    parent_fd: int,
    temp_name: str,
    failures: tuple[BaseException, ...],
) -> NoReturn:
    error = _LocalGuardStagingCleanupError(parent_fd, temp_name, failures)
    retained_failures = error.failures
    cause: BaseException
    if len(retained_failures) == 1:
        cause = retained_failures[0]
    else:
        cause = BaseExceptionGroup(
            "Workspace staging operation and cleanup failures.",
            list(retained_failures),
        )
    raise error from cause


def _create_at(
    parent_fd: int,
    name: str,
    content: bytes,
    *,
    mode: int | None = None,
    staging_name: str | None = None,
) -> None:
    temp_name = _write_temp(
        parent_fd,
        name,
        content,
        mode=mode,
        staging_name=staging_name,
    )
    primary: BaseException | None = None
    try:
        os.link(
            temp_name,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        try:
            info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except BaseException as classification_error:
            primary = classification_error
        else:
            if stat.S_ISLNK(info.st_mode):
                primary = _LocalGuardPathError("escape")
                primary.__cause__ = exc
            else:
                primary = exc
    except BaseException as error:
        primary = error
    absent, cleanup_failures = _unlink_staging_and_inspect(parent_fd, temp_name)
    if not absent:
        failures = cleanup_failures if primary is None else (primary, *cleanup_failures)
        _raise_staging_cleanup_error(parent_fd, temp_name, failures)
    if _contains_fatal_cleanup_failure(cleanup_failures):
        _raise_operation_and_cleanup_failures(primary, cleanup_failures)
    if primary is not None:
        raise primary


def _replace_at(
    parent_fd: int,
    name: str,
    content: bytes,
    *,
    mode: int,
    staging_name: str | None = None,
) -> None:
    temp_name = _write_temp(
        parent_fd,
        name,
        content,
        mode=mode,
        staging_name=staging_name,
    )
    primary: BaseException | None = None
    try:
        os.rename(
            temp_name,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
    except BaseException as error:
        primary = error
    absent, cleanup_failures = _unlink_staging_and_inspect(parent_fd, temp_name)
    if not absent:
        failures = cleanup_failures if primary is None else (primary, *cleanup_failures)
        _raise_staging_cleanup_error(parent_fd, temp_name, failures)
    if _contains_fatal_cleanup_failure(cleanup_failures):
        _raise_operation_and_cleanup_failures(primary, cleanup_failures)
    if primary is not None:
        raise primary


def _raise_workspace_path_error(
    error: _LocalGuardPathError,
    relative_path: str,
) -> NoReturn:
    if error.status == "escape":
        raise ValueError("Workspace path escapes the workspace root.") from error
    if error.status == "notfile":
        raise IsADirectoryError(f"Workspace path is not a file: {relative_path}") from error
    raise FileNotFoundError(f"Workspace file not found: {relative_path}") from error


@contextmanager
def open_regular_for_read(root: Path, relative_path: str) -> Iterator[tuple[BinaryIO, int]]:
    try:
        with _open_parent(root, relative_path) as (parent_fd, name):
            descriptor, info = _open_regular(parent_fd, name)
            with os.fdopen(descriptor, "rb") as file:
                yield file, info.st_size
    except _LocalGuardPathError as exc:
        if exc.status == "notfile":
            raise FileNotFoundError(f"Workspace file not found: {relative_path}") from exc
        _raise_workspace_path_error(exc, relative_path)


def create_regular(
    root: Path,
    relative_path: str,
    content: bytes,
    *,
    mode: int | None = None,
    staging_name: str | None = None,
) -> None:
    try:
        with _open_parent(root, relative_path, create=True) as (parent_fd, name):
            _create_at(parent_fd, name, content, mode=mode, staging_name=staging_name)
    except _LocalGuardPathError as exc:
        _raise_workspace_path_error(exc, relative_path)


def require_absent_regular(root: Path, relative_path: str) -> None:
    """Require an absent final path without following any component."""

    try:
        with _open_parent(root, relative_path) as (parent_fd, name):
            try:
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return
            raise FileExistsError(f"Workspace file already exists: {relative_path}")
    except _LocalGuardPathError as exc:
        if exc.status == "missing":
            return
        _raise_workspace_path_error(exc, relative_path)


def move_regular_if_revision(
    root: Path,
    source_path: str,
    destination_path: str,
    expected_source_revision: str,
) -> WorkspaceMoveResult:
    """Conditionally move by no-overwrite link then unlink.

    The link creation is the authoritative destination-absence precondition.
    Its two-name window is reported as ``link_unlink`` rather than as an atomic
    rename. If unlink settlement fails after the link, the caller receives an
    ambiguity carrying the known destination identity.
    """

    try:
        with (
            _open_parent(root, source_path) as (source_parent_fd, source_name),
            _open_parent(root, destination_path, create=True) as (
                destination_parent_fd,
                destination_name,
            ),
        ):
            before, _mode = _identity_at(
                source_parent_fd,
                source_name,
                require_single_link=True,
            )
            if before[0] != expected_source_revision:
                raise WorkspaceRevisionMismatchError(expected_source_revision, before[0])
            try:
                os.link(
                    source_name,
                    destination_name,
                    src_dir_fd=source_parent_fd,
                    dst_dir_fd=destination_parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise FileExistsError(f"Workspace file already exists: {destination_path}") from exc
            except OSError as exc:
                if exc.errno == errno.EXDEV:
                    raise WorkspaceMoveUnsupportedError(
                        "Workspace move crosses filesystem devices and cannot preserve "
                        "the conditional move contract."
                    ) from exc
                raise
            result = move_result_from_identity(before, fidelity="link_unlink")
            destination_identity, _destination_mode = _identity_at(
                destination_parent_fd,
                destination_name,
            )
            if destination_identity != before:
                raise WorkspaceMoveAmbiguousError(None)
            try:
                os.unlink(source_name, dir_fd=source_parent_fd)
            except BaseException as exc:
                raise WorkspaceMoveAmbiguousError(result) from exc
            return result
    except _LocalGuardPathError as exc:
        _raise_workspace_path_error(exc, source_path)


def write_regular(root: Path, relative_path: str, content: bytes) -> None:
    try:
        with _open_parent(root, relative_path, create=True) as (parent_fd, name):
            mode = _inspect_regular_target_mode(parent_fd, name)
            if mode is None:
                _create_at(parent_fd, name, content)
            else:
                _replace_at(parent_fd, name, content, mode=mode)
    except _LocalGuardPathError as exc:
        _raise_workspace_path_error(exc, relative_path)


def restore_regular(
    root: Path,
    relative_path: str,
    content: bytes,
    *,
    mode: int,
) -> None:
    """Restore exact bytes and mode without following any path component."""

    try:
        with _open_parent(root, relative_path, create=True) as (parent_fd, name):
            current_mode = _inspect_regular_target_mode(parent_fd, name)
            if current_mode is None:
                _create_at(parent_fd, name, content, mode=mode)
            else:
                _replace_at(parent_fd, name, content, mode=mode)
    except _LocalGuardPathError as exc:
        _raise_workspace_path_error(exc, relative_path)


def replace_regular_if_revision(
    root: Path,
    relative_path: str,
    content: bytes,
    expected_revision: str,
    *,
    expected_git_mode: int | None = None,
    replacement_mode: int | None = None,
    staging_name: str | None = None,
) -> tuple[str, str, int]:
    try:
        with _open_parent(root, relative_path) as (parent_fd, name):
            before, mode = _identity_at(parent_fd, name)
            if before[0] != expected_revision:
                raise WorkspaceRevisionMismatchError(expected_revision, before[0])
            if expected_git_mode is not None and bool(mode & 0o111) != bool(
                expected_git_mode & 0o111
            ):
                raise WorkspaceGitModeMismatchError(
                    "100755" if expected_git_mode & 0o111 else "100644",
                    "100755" if mode & 0o111 else "100644",
                )
            _replace_at(
                parent_fd,
                name,
                content,
                mode=mode if replacement_mode is None else replacement_mode,
                staging_name=staging_name,
            )
            return before
    except _LocalGuardPathError as exc:
        _raise_workspace_path_error(exc, relative_path)


def settle_regular_staging(
    root: Path,
    relative_path: str,
    staging_name: str,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    """Remove one positively identified durable staging file before replay."""

    if not staging_name or staging_name in {".", ".."} or "/" in staging_name:
        raise ValueError("Workspace staging identity is invalid.")
    try:
        with _open_parent(root, relative_path) as (parent_fd, _name):
            try:
                descriptor, info = _open_regular(parent_fd, staging_name)
            except _LocalGuardPathError as error:
                if error.status == "missing":
                    return
                raise _LocalGuardStagingConflictError(
                    "Workspace durable staging identity is not a regular file."
                ) from error
            if info.st_size != expected_bytes:
                os.close(descriptor)
                raise _LocalGuardStagingConflictError(
                    "Workspace durable staging content does not match its publication."
                )
            digest = hashlib.sha256()
            actual_bytes = 0
            with os.fdopen(descriptor, "rb") as staging:
                remaining = expected_bytes + 1
                while remaining and (chunk := staging.read(min(1 << 16, remaining))):
                    digest.update(chunk)
                    actual_bytes += len(chunk)
                    remaining -= len(chunk)
            if digest.hexdigest() != expected_sha256 or actual_bytes != expected_bytes:
                raise _LocalGuardStagingConflictError(
                    "Workspace durable staging content does not match its publication."
                )
            absent, cleanup_failures = _unlink_staging_and_inspect(parent_fd, staging_name)
            if not absent:
                _raise_staging_cleanup_error(parent_fd, staging_name, cleanup_failures)
            if _contains_fatal_cleanup_failure(cleanup_failures):
                _raise_operation_and_cleanup_failures(None, cleanup_failures)
    except _LocalGuardPathError as error:
        if error.status == "missing":
            return
        raise _LocalGuardStagingConflictError(
            "Workspace durable staging parent identity changed."
        ) from error


def delete_regular(root: Path, relative_path: str) -> None:
    try:
        with _open_parent(root, relative_path) as (parent_fd, name):
            mode = _inspect_regular_target_mode(parent_fd, name)
            if mode is not None:
                os.unlink(name, dir_fd=parent_fd)
    except _LocalGuardPathError as exc:
        if exc.status not in {"missing", "notdir"}:
            _raise_workspace_path_error(exc, relative_path)


def delete_empty_directory(root: Path, relative_path: str) -> None:
    """Remove one empty directory without following any path component."""

    try:
        with _open_parent(root, relative_path) as (parent_fd, name):
            info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise _LocalGuardPathError("escape")
            if not stat.S_ISDIR(info.st_mode):
                raise _LocalGuardPathError("notdir")
            os.rmdir(name, dir_fd=parent_fd)
    except _LocalGuardPathError as exc:
        _raise_workspace_path_error(exc, relative_path)


def delete_regular_if_revision(
    root: Path,
    relative_path: str,
    expected_revision: str,
) -> tuple[str, str, int]:
    try:
        with _open_parent(root, relative_path) as (parent_fd, name):
            before, _mode = _identity_at(parent_fd, name)
            if before[0] != expected_revision:
                raise WorkspaceRevisionMismatchError(expected_revision, before[0])
            os.unlink(name, dir_fd=parent_fd)
            return before
    except _LocalGuardPathError as exc:
        _raise_workspace_path_error(exc, relative_path)
