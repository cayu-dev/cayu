from __future__ import annotations

import asyncio
import ctypes
import errno
import hashlib
import importlib
import json
import mimetypes
import os
import re
import shutil
import stat
import sys
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Any
from uuid import uuid4

from cayu._exception_groups import exception_cause, exception_context, set_exception_context
from cayu._filesystem_lock import cooperative_path_lock
from cayu._validation import (
    copy_durable_json_object,
    require_clean_nonblank,
    require_nonblank,
    require_unicode_scalar_text,
)
from cayu.artifacts._settlement import (
    _absent_artifact_write,
    _ArtifactWritePhaseReporter,
    _ArtifactWriteRegistry,
    _await_owned_sync_call,
    _committed_artifact_write,
    _settle_artifact_write,
    _unsettled_artifact_write,
)
from cayu.artifacts.base import (
    ArtifactListResult,
    ArtifactMetadata,
    ArtifactReadResult,
    ArtifactScope,
    ArtifactStore,
    ArtifactStoreUnavailableError,
    InvalidArtifactIdError,
    _require_matching_artifact,
)
from cayu.artifacts.settlement import (
    ArtifactWriteSettlementFailureCode,
    ArtifactWriteSettlementPhase,
)

_CONTENT_FILE = "content"
_METADATA_FILE = "metadata.json"
_ARTIFACT_ID_PREFIX = "art_"
_ARTIFACT_ID_PATTERN = re.compile(r"\Aart_[0-9a-f]{32}\Z")
_OPEN_READ_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
_OPEN_NOFOLLOW_FLAG = getattr(os, "O_NOFOLLOW", 0)
_OPEN_DIRECTORY_FLAG = getattr(os, "O_DIRECTORY", 0)
_WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 4
_ARTIFACT_LOCK_DIRECTORY_NAME = "cayu-artifact-locks"
_ARTIFACT_LOCK_SHARD_COUNT = 256
_ARTIFACT_ROOT_LOCK_DIRECTORY_NAME = "cayu-artifact-root-locks"
_ARTIFACT_ROOT_PENDING_PREFIX = ".cayu-artifact-root-pending-"
_SUPPORTS_DURABLE_DIRECTORY_SYNC = os.name != "nt" and hasattr(os, "O_DIRECTORY")
try:
    _FCNTL_MODULE = importlib.import_module("fcntl")
except ImportError:  # pragma: no cover - Windows
    _FCNTL_MODULE = None
_DARWIN_FULL_SYNC_COMMAND = (
    getattr(_FCNTL_MODULE, "F_FULLFSYNC", None) if sys.platform == "darwin" else None
)
_SUPPORTS_DIRECTORY_FD = (
    os.open in os.supports_dir_fd
    and os.mkdir in os.supports_dir_fd
    and os.rename in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.stat in os.supports_follow_symlinks
)


class _PublishedLocalArtifactError(Exception):
    """Carry a failure that happened after the final artifact became visible."""

    def __init__(self, error: BaseException) -> None:
        super().__init__("Local artifact publication failed after becoming visible.")
        self.error = error


class _AbsentLocalArtifactError(Exception):
    """Carry positive evidence that this invocation left no artifact state."""

    def __init__(self, error: BaseException) -> None:
        super().__init__("Local artifact publication failed before publication.")
        self.error = error


def _detach_redundant_cleanup_context(
    cleanup_error: BaseException,
    primary_error: BaseException,
) -> None:
    if exception_cause(cleanup_error) is None and exception_context(cleanup_error) is primary_error:
        set_exception_context(cleanup_error, None)


class LocalArtifactStore(ArtifactStore):
    """Local filesystem implementation of ArtifactStore."""

    def __init__(self, root: str | Path, *, store_id: str | None = None) -> None:
        if not isinstance(root, str | PathLike):
            raise TypeError("LocalArtifactStore root must be a string or Path.")
        root_path = Path(root).expanduser().resolve()
        initialized_identity: tuple[int, int] | None = None
        durable_publication_supported = _supports_durable_publication()
        durable_initialization_required = durable_publication_supported and not root_path.exists()
        if durable_publication_supported and not durable_initialization_required:
            try:
                durable_initialization_required = _root_pending_marker(root_path).exists()
            except OSError as exc:
                raise ArtifactStoreUnavailableError(
                    "Local artifact store root could not be made durable."
                ) from exc
        if durable_initialization_required:
            try:
                initialized_identity = _initialize_durable_store_root(root_path)
            except OSError as exc:
                raise ArtifactStoreUnavailableError(
                    "Local artifact store root could not be made durable."
                ) from exc
        elif not root_path.exists():
            root_path.mkdir(parents=True, exist_ok=True)
        root_stat = os.stat(root_path, follow_symlinks=False)
        if _is_windows_reparse_point(root_stat) or not stat.S_ISDIR(root_stat.st_mode):
            raise NotADirectoryError(f"Artifact store root is not a directory: {root_path}")
        if initialized_identity is not None and _stat_identity(root_stat) != initialized_identity:
            raise ArtifactStoreUnavailableError(
                "Local artifact store root changed during durable initialization."
            )

        if store_id is None:
            self.id = str(root_path)
        else:
            clean_store_id = require_clean_nonblank(store_id, "store_id")
            self.id = require_unicode_scalar_text(clean_store_id, "store_id")
        self.root = root_path
        self._root_identity = _stat_identity(root_stat)
        self._write_registry = _ArtifactWriteRegistry()

    async def put_bytes(
        self,
        content: bytes,
        *,
        artifact_id: str | None = None,
        filename: str,
        content_type: str | None = None,
        scope: ArtifactScope = ArtifactScope.SESSION,
        session_id: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactMetadata:
        if type(content) is not bytes:
            raise TypeError("Artifact content must be bytes.")
        filename = require_nonblank(filename, "filename")
        if content_type is None:
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        content_type = require_clean_nonblank(content_type, "content_type")
        scope = _validate_scope(scope)
        session_id = _validate_optional_id(session_id, "session_id")
        agent_name = _validate_optional_id(agent_name, "agent_name")
        environment_name = _validate_optional_id(environment_name, "environment_name")
        _validate_scope_owner(scope, session_id=session_id, environment_name=environment_name)
        copied_metadata = copy_durable_json_object(metadata or {}, "metadata")

        resolved_artifact_id = (
            _new_artifact_id() if artifact_id is None else _validate_artifact_id(artifact_id)
        )
        artifact = ArtifactMetadata(
            id=resolved_artifact_id,
            filename=filename,
            content_type=content_type,
            size_bytes=len(content),
            scope=scope,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=copied_metadata,
        )
        _require_durable_publication_support()
        callback = _write_generated_artifact if artifact_id is None else _put_deterministic_artifact
        return await _settle_artifact_write(
            registry=self._write_registry,
            store_id=self.id,
            artifact_id=artifact.id,
            operation_name="Local artifact publication",
            operation=lambda reporter: _run_local_artifact_write(
                reporter,
                callback=callback,
                root=self.root,
                root_identity=self._root_identity,
                artifact=artifact,
                content=content,
            ),
        )

    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        limit = _validate_limit(max_bytes, "max_bytes")
        try:
            return await asyncio.to_thread(
                _read_artifact,
                self.root,
                self._root_identity,
                artifact_id,
                limit,
            )
        except (
            ArtifactStoreUnavailableError,
            FileNotFoundError,
            InvalidArtifactIdError,
            TypeError,
            ValueError,
        ):
            raise
        except OSError as exc:
            raise ArtifactStoreUnavailableError(
                "Local artifact store could not read artifact content."
            ) from exc

    async def list(
        self,
        *,
        scope: ArtifactScope | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        limit: int | None = None,
    ) -> ArtifactListResult:
        validated_scope = _validate_scope(scope) if scope is not None else None
        session_id = _validate_optional_id(session_id, "session_id")
        agent_name = _validate_optional_id(agent_name, "agent_name")
        environment_name = _validate_optional_id(environment_name, "environment_name")
        validated_limit = _validate_limit(limit, "limit")
        try:
            return await asyncio.to_thread(
                _list_artifacts,
                self.root,
                self._root_identity,
                validated_scope,
                session_id,
                agent_name,
                environment_name,
                validated_limit,
            )
        except ArtifactStoreUnavailableError:
            raise
        except OSError as exc:
            raise ArtifactStoreUnavailableError(
                "Local artifact store could not list artifacts."
            ) from exc

    async def delete(self, artifact_id: str) -> None:
        try:
            await asyncio.to_thread(
                _delete_artifact,
                self.root,
                self._root_identity,
                artifact_id,
            )
        except (ArtifactStoreUnavailableError, InvalidArtifactIdError, TypeError, ValueError):
            raise
        except OSError as exc:
            raise ArtifactStoreUnavailableError(
                "Local artifact store could not delete artifact content."
            ) from exc


async def _run_local_artifact_write(
    reporter: _ArtifactWritePhaseReporter,
    *,
    callback: Callable[..., object],
    root: Path,
    root_identity: tuple[int, int],
    artifact: ArtifactMetadata,
    content: bytes,
):
    reporter.set(ArtifactWriteSettlementPhase.CONTENT)
    try:
        result = await _await_owned_sync_call(
            reporter,
            callback,
            root,
            root_identity,
            artifact,
            content,
            reporter.set,
        )
    except _AbsentLocalArtifactError as failure:
        return _absent_artifact_write(
            _local_publication_error(failure.error),
            phase=ArtifactWriteSettlementPhase.CLEANUP,
            failure_codes=(ArtifactWriteSettlementFailureCode.MUTATION_FAILED,),
            cancellation_error=failure.error,
        )
    except _PublishedLocalArtifactError as failure:
        return _unsettled_artifact_write(
            _local_publication_error(failure.error),
            phase=ArtifactWriteSettlementPhase.COMMIT,
            failure_codes=(ArtifactWriteSettlementFailureCode.COMMIT_FAILED,),
            cancellation_error=failure.error,
        )
    except BaseException as error:
        return _unsettled_artifact_write(
            _local_publication_error(error),
            phase=reporter.phase,
            failure_codes=(
                ArtifactWriteSettlementFailureCode.MUTATION_FAILED,
                ArtifactWriteSettlementFailureCode.CLEANUP_FAILED,
            ),
            cancellation_error=error,
        )
    committed = result if type(result) is ArtifactMetadata else artifact
    return _committed_artifact_write(committed)


def _local_publication_error(error: BaseException) -> BaseException:
    if issubclass(type(error), (ArtifactStoreUnavailableError, TypeError, ValueError)):
        return error
    if issubclass(type(error), OSError):
        unavailable = ArtifactStoreUnavailableError(
            "Local artifact store could not write artifact content."
        )
        unavailable.__cause__ = error
        return unavailable
    return error


def _new_artifact_id() -> str:
    return f"{_ARTIFACT_ID_PREFIX}{uuid4().hex}"


def _artifact_lock_key(artifact_id: str) -> str:
    digest = hashlib.sha256(artifact_id.encode("ascii")).digest()
    shard = int.from_bytes(digest[:8], "big") % _ARTIFACT_LOCK_SHARD_COUNT
    return f"artifact-shard-{shard:03d}"


@contextmanager
def _artifact_ownership_lock(root: Path, artifact_id: str) -> Iterator[None]:
    """Serialize one artifact through a bounded cross-process lock namespace."""

    with cooperative_path_lock(
        root,
        _artifact_lock_key(artifact_id),
        lock_directory_name=_ARTIFACT_LOCK_DIRECTORY_NAME,
    ):
        yield


def _validate_artifact_id(value: str) -> str:
    try:
        value = require_clean_nonblank(value, "artifact_id")
        value = require_unicode_scalar_text(value, "artifact_id")
    except (TypeError, ValueError) as exc:
        raise InvalidArtifactIdError("Invalid local artifact id.") from exc
    if _ARTIFACT_ID_PATTERN.fullmatch(value) is None:
        raise InvalidArtifactIdError("Invalid local artifact id.")
    return value


def _validate_scope(value: ArtifactScope | str) -> ArtifactScope:
    if isinstance(value, ArtifactScope):
        return value
    if type(value) is str:
        try:
            return ArtifactScope(value)
        except ValueError as exc:
            raise ValueError(f"Unsupported artifact scope: {value!r}") from exc
    raise TypeError("Artifact scope must be an ArtifactScope.")


def _validate_optional_id(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    value = require_clean_nonblank(value, field_name)
    return require_unicode_scalar_text(value, field_name)


def _validate_scope_owner(
    scope: ArtifactScope,
    *,
    session_id: str | None,
    environment_name: str | None,
) -> None:
    if scope == ArtifactScope.SESSION and session_id is None:
        raise ValueError("Session-scoped artifacts require session_id.")
    if scope == ArtifactScope.ENVIRONMENT and environment_name is None:
        raise ValueError("Environment-scoped artifacts require environment_name.")


def _validate_limit(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"Artifact {field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"Artifact {field_name} must be greater than zero.")
    return value


def _require_durable_publication_support() -> None:
    if not _supports_durable_publication():
        raise ArtifactStoreUnavailableError(
            "Local artifact publication requires durable directory synchronization, "
            "which is unavailable on this platform."
        )


def _supports_durable_publication() -> bool:
    if not _SUPPORTS_DURABLE_DIRECTORY_SYNC:
        return False
    return sys.platform != "darwin" or _DARWIN_FULL_SYNC_COMMAND is not None


def _create_durable_directory_ancestry(root: Path) -> None:
    """Create and durably publish each missing root ancestor in order."""

    missing_ancestors: list[Path] = []
    candidate = root
    while not candidate.exists():
        missing_ancestors.append(candidate)
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent

    _sync_directory_path(candidate)
    if candidate.parent != candidate:
        _sync_directory_path(candidate.parent)

    for directory in reversed(missing_ancestors):
        directory.mkdir(exist_ok=True)
        current = os.stat(directory, follow_symlinks=False)
        if _is_windows_reparse_point(current) or not stat.S_ISDIR(current.st_mode):
            raise NotADirectoryError(f"Artifact store root is not a directory: {directory}")
        _sync_directory_path(directory, expected_identity=_stat_identity(current))
        _sync_directory_path(directory.parent)


def _root_pending_marker_identity(root: Path) -> bytes:
    parent = os.stat(root.parent, follow_symlinks=False)
    normalized_name = unicodedata.normalize("NFC", root.name).casefold()
    parent_identity = f"{parent.st_dev}:{parent.st_ino}\0".encode("ascii")
    return parent_identity + os.fsencode(normalized_name)


def _root_pending_marker(root: Path) -> Path:
    digest = hashlib.sha256(_root_pending_marker_identity(root)).hexdigest()
    return root.parent / f"{_ARTIFACT_ROOT_PENDING_PREFIX}{digest}"


def _root_pending_marker_payload(root: Path) -> bytes:
    return b"cayu.local-artifact-root.pending.v1\n" + _root_pending_marker_identity(root)


def _initialize_durable_store_root(root: Path) -> tuple[int, int]:
    """Finish a recoverable root creation without burdening pre-existing roots."""

    _create_durable_directory_ancestry(root.parent)
    lock_anchor = Path(root.anchor)
    with cooperative_path_lock(
        lock_anchor,
        str(root),
        lock_directory_name=_ARTIFACT_ROOT_LOCK_DIRECTORY_NAME,
    ):
        marker = _root_pending_marker(root)
        marker_exists = marker.exists()
        with _open_artifact_directory(root.parent) as (parent_fd, parent_identity):
            marker_name = marker.name
            if marker_exists:
                _validate_pending_root_marker(
                    parent_fd,
                    root.parent,
                    parent_identity,
                    marker_name,
                    root,
                )
            else:
                _create_pending_root_marker(
                    parent_fd,
                    root.parent,
                    parent_identity,
                    marker_name,
                    root,
                )
            _sync_open_directory(parent_fd, root.parent, parent_identity)

            try:
                root_stat = _stat_directory_entry(root, parent_fd=parent_fd)
            except FileNotFoundError:
                if parent_fd is None:
                    root.mkdir()
                else:
                    os.mkdir(root.name, dir_fd=parent_fd)
                root_stat = _stat_directory_entry(root, parent_fd=parent_fd)
            if _is_windows_reparse_point(root_stat) or not stat.S_ISDIR(root_stat.st_mode):
                raise NotADirectoryError(f"Artifact store root is not a directory: {root}")
            root_identity = _stat_identity(root_stat)
            with _open_artifact_directory(root, parent_fd=parent_fd) as (
                root_fd,
                opened_root_identity,
            ):
                if opened_root_identity != root_identity:
                    raise ArtifactStoreUnavailableError(
                        "Local artifact store root changed during initialization."
                    )
                _sync_open_directory(root_fd, root, opened_root_identity)
            _sync_open_directory(parent_fd, root.parent, parent_identity)
            if parent_fd is None:
                marker.unlink()
            else:
                os.unlink(marker_name, dir_fd=parent_fd)
            _sync_open_directory(parent_fd, root.parent, parent_identity)
            return root_identity


def _create_pending_root_marker(
    parent_fd: int | None,
    parent: Path,
    parent_identity: tuple[int, int],
    marker_name: str,
    root: Path,
) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
        | _OPEN_NOFOLLOW_FLAG
    )
    marker_fd = (
        os.open(parent / marker_name, flags, 0o600)
        if parent_fd is None
        else os.open(marker_name, flags, 0o600, dir_fd=parent_fd)
    )
    try:
        _require_directory_identity(parent_fd, parent, parent_identity)
        payload = _root_pending_marker_payload(root)
        written = 0
        while written < len(payload):
            chunk_size = os.write(marker_fd, payload[written:])
            if chunk_size <= 0:
                raise OSError("Local artifact root pending marker write made no progress.")
            written += chunk_size
        _sync_descriptor(marker_fd)
    finally:
        os.close(marker_fd)


def _validate_pending_root_marker(
    parent_fd: int | None,
    parent: Path,
    parent_identity: tuple[int, int],
    marker_name: str,
    root: Path,
) -> None:
    marker_fd = _open_artifact_file(
        parent_fd,
        parent,
        parent_identity,
        marker_name,
        missing_message="Local artifact root pending marker disappeared.",
    )
    try:
        with os.fdopen(marker_fd, "rb", closefd=False) as file:
            payload = file.read()
        expected = _root_pending_marker_payload(root)
        if not expected.startswith(payload):
            raise ArtifactStoreUnavailableError("Local artifact root pending marker is invalid.")
        _sync_descriptor(marker_fd)
    finally:
        os.close(marker_fd)


def _sync_descriptor(descriptor: int) -> None:
    if sys.platform != "darwin":
        os.fsync(descriptor)
        return
    if _FCNTL_MODULE is None or _DARWIN_FULL_SYNC_COMMAND is None:
        raise ArtifactStoreUnavailableError(
            "Local artifact publication requires F_FULLFSYNC on macOS."
        )
    _FCNTL_MODULE.fcntl(descriptor, _DARWIN_FULL_SYNC_COMMAND)


def _sync_directory_path(path: Path, *, expected_identity: tuple[int, int] | None = None) -> None:
    """Open and synchronize one directory without following a replacement link."""

    flags = _OPEN_READ_FLAGS | _OPEN_DIRECTORY_FLAG | _OPEN_NOFOLLOW_FLAG
    descriptor = os.open(path, flags)
    try:
        current = os.fstat(descriptor)
        if _is_windows_reparse_point(current) or not stat.S_ISDIR(current.st_mode):
            raise ArtifactStoreUnavailableError(
                f"Local artifact directory could not be synchronized safely: {path}"
            )
        if expected_identity is not None and _stat_identity(current) != expected_identity:
            raise ArtifactStoreUnavailableError(
                f"Local artifact directory changed before synchronization: {path}"
            )
        _sync_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _sync_open_directory(
    directory_fd: int | None,
    path: Path,
    expected_identity: tuple[int, int],
) -> None:
    if directory_fd is None:
        _sync_directory_path(path, expected_identity=expected_identity)
        return
    _require_directory_identity(directory_fd, path, expected_identity)
    _sync_descriptor(directory_fd)


@contextmanager
def _open_store_root(
    path: Path,
    expected_identity: tuple[int, int],
) -> Iterator[int | None]:
    before = _require_store_root_identity(path, expected_identity)
    if not _SUPPORTS_DIRECTORY_FD:
        try:
            yield None
        finally:
            _require_store_root_identity(path, expected_identity)
        return

    flags = _OPEN_READ_FLAGS | _OPEN_DIRECTORY_FLAG | _OPEN_NOFOLLOW_FLAG
    try:
        root_fd = os.open(path, flags)
    except OSError as exc:
        raise ArtifactStoreUnavailableError(
            "Local artifact store root could not be opened safely."
        ) from exc
    try:
        after = os.fstat(root_fd)
        if (
            _is_windows_reparse_point(after)
            or not stat.S_ISDIR(after.st_mode)
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != expected_identity
        ):
            raise ArtifactStoreUnavailableError(
                "Local artifact store root changed while it was being opened."
            )
        try:
            yield root_fd
        finally:
            _require_store_root_identity(path, expected_identity)
    finally:
        os.close(root_fd)


def _require_store_root_identity(
    path: Path,
    expected_identity: tuple[int, int],
) -> os.stat_result:
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ArtifactStoreUnavailableError("Local artifact store root is unavailable.") from exc
    if (
        _is_windows_reparse_point(current)
        or not stat.S_ISDIR(current.st_mode)
        or _stat_identity(current) != expected_identity
    ):
        raise ArtifactStoreUnavailableError(
            "Local artifact store root changed after initialization."
        )
    return current


def _write_artifact(
    root: Path,
    root_identity: tuple[int, int],
    artifact: ArtifactMetadata,
    content: bytes,
    report_phase: Callable[[ArtifactWriteSettlementPhase], None] | None = None,
) -> tuple[int, int]:
    target = _artifact_dir(root, artifact.id)
    staging_name = f"{artifact.id}.staging-{uuid4().hex}"
    staging = root / staging_name
    published = False
    staging_created = False
    created_identity: tuple[int, int] | None = None
    try:
        with _open_store_root(root, root_identity) as root_fd:
            try:
                if root_fd is None:
                    staging.mkdir(mode=0o700, parents=False)
                else:
                    os.mkdir(staging_name, mode=0o700, dir_fd=root_fd)
                staging_created = True
            except FileExistsError as exc:  # pragma: no cover - UUID collision
                raise ArtifactStoreUnavailableError(
                    "Local artifact staging directory already exists."
                ) from exc

            created_identity = _stat_identity(_stat_directory_entry(staging, parent_fd=root_fd))
            try:
                with _open_artifact_directory(staging, parent_fd=root_fd) as (
                    directory_fd,
                    directory_identity,
                ):
                    _write_artifact_file(
                        directory_fd,
                        staging,
                        directory_identity,
                        _CONTENT_FILE,
                        content,
                    )
                    metadata_bytes = json.dumps(
                        artifact.model_dump(mode="json"),
                        sort_keys=True,
                        indent=2,
                    ).encode("utf-8")
                    _write_artifact_file(
                        directory_fd,
                        staging,
                        directory_identity,
                        _METADATA_FILE,
                        metadata_bytes,
                    )
                    _sync_open_directory(directory_fd, staging, directory_identity)
                if report_phase is not None:
                    report_phase(ArtifactWriteSettlementPhase.COMMIT)
                try:
                    _rename_directory_no_replace(
                        staging,
                        target,
                        parent_fd=root_fd,
                    )
                    published = True
                    _sync_open_directory(root_fd, root, root_identity)
                except OSError as exc:
                    if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                        raise FileExistsError(f"Artifact already exists: {artifact.id}") from exc
                    raise
            except BaseException as primary_error:
                if report_phase is not None:
                    report_phase(ArtifactWriteSettlementPhase.CLEANUP)
                try:
                    if not published:
                        _remove_artifact_directory_if_unchanged(
                            staging,
                            created_identity,
                            parent_fd=root_fd,
                            ignore_errors=False,
                        )
                        try:
                            _stat_directory_entry(staging, parent_fd=root_fd)
                        except FileNotFoundError:
                            _sync_open_directory(root_fd, root, root_identity)
                        else:
                            raise ArtifactStoreUnavailableError(
                                "Local artifact staging cleanup could not prove absence."
                            )
                except BaseException as cleanup_error:
                    _detach_redundant_cleanup_context(cleanup_error, primary_error)
                    raise ArtifactStoreUnavailableError(
                        "Local artifact publication failed and staging cleanup also failed."
                    ) from BaseExceptionGroup(
                        "Local artifact publication and staging cleanup failures.",
                        [primary_error, cleanup_error],
                    )
                raise _AbsentLocalArtifactError(primary_error) from primary_error
            return created_identity
    except BaseException as error:
        if published:
            published_error = error.error if isinstance(error, _AbsentLocalArtifactError) else error
            raise _PublishedLocalArtifactError(published_error) from published_error
        if isinstance(error, _AbsentLocalArtifactError):
            raise
        if not staging_created:
            raise _AbsentLocalArtifactError(error) from error
        raise


def _write_generated_artifact(
    root: Path,
    root_identity: tuple[int, int],
    artifact: ArtifactMetadata,
    content: bytes,
    report_phase: Callable[[ArtifactWriteSettlementPhase], None],
) -> tuple[int, int]:
    published_identity: tuple[int, int] | None = None
    write_error: BaseException | None = None
    try:
        with _artifact_ownership_lock(root, artifact.id):
            try:
                published_identity = _write_artifact(
                    root,
                    root_identity,
                    artifact,
                    content,
                    report_phase,
                )
            except BaseException as error:
                # Keep the body outcome until the lock has physically released so
                # teardown cannot replace publication state or failure evidence.
                write_error = error
    except BaseException as lock_error:
        if isinstance(write_error, _PublishedLocalArtifactError):
            combined_error = ArtifactStoreUnavailableError(
                "Local artifact publication and ownership-lock cleanup both failed."
            )
            combined_error.__cause__ = BaseExceptionGroup(
                "Local artifact publication and ownership-lock cleanup failures.",
                [write_error.error, lock_error],
            )
            raise _PublishedLocalArtifactError(combined_error) from combined_error
        if write_error is not None:
            combined_error = ArtifactStoreUnavailableError(
                "Local artifact publication and ownership-lock cleanup both failed."
            )
            combined_error.__cause__ = BaseExceptionGroup(
                "Local artifact publication and ownership-lock cleanup failures.",
                [write_error, lock_error],
            )
            if isinstance(write_error, _AbsentLocalArtifactError):
                raise _AbsentLocalArtifactError(combined_error) from combined_error
            raise combined_error from combined_error.__cause__
        if published_identity is not None:
            raise _PublishedLocalArtifactError(lock_error) from lock_error
        raise _AbsentLocalArtifactError(lock_error) from lock_error
    if write_error is not None:
        raise write_error
    if published_identity is None:  # pragma: no cover - write returned without an outcome
        raise AssertionError("Generated artifact publication produced no result.")
    return published_identity


def _put_deterministic_artifact(
    root: Path,
    root_identity: tuple[int, int],
    artifact: ArtifactMetadata,
    content: bytes,
    report_phase: Callable[[ArtifactWriteSettlementPhase], None],
) -> ArtifactMetadata:
    with _artifact_ownership_lock(root, artifact.id):
        for attempt in range(3):
            try:
                _write_artifact(
                    root,
                    root_identity,
                    artifact,
                    content,
                    report_phase,
                )
            except _AbsentLocalArtifactError as write_failure:
                if not isinstance(write_failure.error, FileExistsError):
                    raise
                try:
                    existing = _read_artifact(root, root_identity, artifact.id, None)
                except (FileNotFoundError, ValueError) as read_error:
                    if attempt >= 2:
                        raise _AbsentLocalArtifactError(read_error) from read_error
                    _remove_matching_incomplete_artifact(
                        root,
                        root_identity,
                        artifact,
                        content,
                    )
                    continue
                try:
                    _require_matching_artifact(existing, expected=artifact, content=content)
                except ValueError as conflict:
                    raise _AbsentLocalArtifactError(conflict) from conflict
                try:
                    _sync_existing_artifact(root, root_identity, artifact.id)
                except BaseException as sync_error:
                    raise _PublishedLocalArtifactError(sync_error) from sync_error
                return existing.metadata
            else:
                return artifact
        raise AssertionError("Artifact write retry loop did not terminate.")


def _remove_matching_incomplete_artifact(
    root: Path,
    root_identity: tuple[int, int],
    artifact: ArtifactMetadata,
    content: bytes,
) -> bool:
    """Remove only a legacy partial directory for the same deterministic write."""

    target = _artifact_dir(root, artifact.id)
    with _open_store_root(root, root_identity) as root_fd:
        try:
            with _open_artifact_directory(target, parent_fd=root_fd) as (
                directory_fd,
                directory_identity,
            ):
                names = set(
                    os.listdir(directory_fd) if directory_fd is not None else os.listdir(target)
                )
                if not names <= {_CONTENT_FILE, _METADATA_FILE}:
                    raise ValueError("Incomplete artifact directory contains unexpected entries.")
                if _CONTENT_FILE in names:
                    content_fd = _open_artifact_file(
                        directory_fd,
                        target,
                        directory_identity,
                        _CONTENT_FILE,
                        missing_message=f"Artifact content not found: {artifact.id}",
                    )
                    with os.fdopen(content_fd, "rb") as file:
                        existing_content = file.read()
                    if existing_content != content:
                        raise ValueError(
                            "Incomplete artifact content conflicts with deterministic retry."
                        )
                if _METADATA_FILE in names:
                    try:
                        existing_metadata = _load_metadata_from_directory(
                            target,
                            directory_fd,
                            directory_identity,
                        )
                    except FileNotFoundError:
                        pass
                    else:
                        comparable_existing = existing_metadata.model_dump(
                            mode="json",
                            exclude={"created_at"},
                        )
                        comparable_expected = artifact.model_dump(
                            mode="json",
                            exclude={"created_at"},
                        )
                        if comparable_existing != comparable_expected:
                            raise ValueError(
                                "Incomplete artifact metadata conflicts with deterministic retry."
                            )
                        if _CONTENT_FILE in names:
                            return False
                incomplete_identity = directory_identity
        except FileNotFoundError:
            return True
        quarantine_name = f"{artifact.id}.partial-{uuid4().hex}"
        quarantine = root / quarantine_name
        try:
            if root_fd is None:
                target.rename(quarantine)
            else:
                os.rename(
                    artifact.id,
                    quarantine_name,
                    src_dir_fd=root_fd,
                    dst_dir_fd=root_fd,
                )
        except FileNotFoundError:
            return True
        claimed_identity = _stat_identity(_stat_directory_entry(quarantine, parent_fd=root_fd))
        if claimed_identity != incomplete_identity:
            try:
                if root_fd is None:
                    quarantine.rename(target)
                else:
                    os.rename(
                        quarantine_name,
                        artifact.id,
                        src_dir_fd=root_fd,
                        dst_dir_fd=root_fd,
                    )
            except OSError as exc:
                raise ArtifactStoreUnavailableError(
                    "A completed artifact raced legacy partial-write recovery."
                ) from exc
            return False
        _remove_artifact_directory_if_unchanged(
            quarantine,
            incomplete_identity,
            parent_fd=root_fd,
            ignore_errors=False,
        )
        try:
            _stat_directory_entry(quarantine, parent_fd=root_fd)
        except FileNotFoundError:
            return True
        return False


def _rename_directory_no_replace(
    source: Path,
    target: Path,
    *,
    parent_fd: int | None,
) -> None:
    """Atomically publish a staged directory without replacing any target."""

    if os.name == "nt":
        os.rename(source, target)
        return
    if sys.platform == "darwin":
        if parent_fd is None:
            _call_native_rename(
                "renamex_np",
                (os.fsencode(source), os.fsencode(target), _RENAME_EXCL),
            )
        else:
            _call_native_rename(
                "renameatx_np",
                (
                    parent_fd,
                    os.fsencode(source.name),
                    parent_fd,
                    os.fsencode(target.name),
                    _RENAME_EXCL,
                ),
            )
        return
    if sys.platform.startswith("linux"):
        source_fd = _AT_FDCWD if parent_fd is None else parent_fd
        target_fd = _AT_FDCWD if parent_fd is None else parent_fd
        source_path = source if parent_fd is None else Path(source.name)
        target_path = target if parent_fd is None else Path(target.name)
        _call_native_rename(
            "renameat2",
            (
                source_fd,
                os.fsencode(source_path),
                target_fd,
                os.fsencode(target_path),
                _RENAME_NOREPLACE,
            ),
        )
        return
    raise ArtifactStoreUnavailableError(
        "Local artifact publication requires atomic no-replace rename support."
    )


def _call_native_rename(function_name: str, arguments: tuple[object, ...]) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = getattr(libc, function_name)
    except AttributeError as exc:
        raise ArtifactStoreUnavailableError(
            "Atomic no-replace rename is unavailable on this platform."
        ) from exc
    function.restype = ctypes.c_int
    if function(*arguments) == 0:
        return
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number))


def _read_artifact(
    root: Path,
    root_identity: tuple[int, int],
    artifact_id: str,
    max_bytes: int | None,
) -> ArtifactReadResult:
    target = _artifact_dir(root, artifact_id)
    with (
        _open_store_root(root, root_identity) as root_fd,
        _open_artifact_directory(target, parent_fd=root_fd) as (
            directory_fd,
            directory_identity,
        ),
    ):
        metadata = _load_metadata_from_directory(
            target,
            directory_fd,
            directory_identity,
        )
        content_fd = _open_artifact_file(
            directory_fd,
            target,
            directory_identity,
            _CONTENT_FILE,
            missing_message=f"Artifact content not found: {artifact_id}",
        )
        with os.fdopen(content_fd, "rb") as file:
            content = file.read() if max_bytes is None else file.read(max_bytes)
            total_bytes = os.fstat(file.fileno()).st_size
    total_bytes = max(total_bytes, len(content))
    return ArtifactReadResult(
        metadata=metadata,
        content=content,
        total_bytes=total_bytes,
        truncated=total_bytes > len(content),
    )


def _sync_existing_artifact(
    root: Path,
    root_identity: tuple[int, int],
    artifact_id: str,
) -> None:
    """Re-establish durability before acknowledging an exact deterministic retry."""

    target = _artifact_dir(root, artifact_id)
    with (
        _open_store_root(root, root_identity) as root_fd,
        _open_artifact_directory(target, parent_fd=root_fd) as (
            directory_fd,
            directory_identity,
        ),
    ):
        for filename, missing_message in (
            (_CONTENT_FILE, f"Artifact content not found: {artifact_id}"),
            (_METADATA_FILE, f"Artifact metadata not found: {artifact_id}"),
        ):
            file_fd = _open_artifact_file(
                directory_fd,
                target,
                directory_identity,
                filename,
                missing_message=missing_message,
            )
            try:
                _sync_descriptor(file_fd)
            finally:
                os.close(file_fd)
        _sync_open_directory(directory_fd, target, directory_identity)
        _sync_open_directory(root_fd, root, root_identity)


def _list_artifacts(
    root: Path,
    root_identity: tuple[int, int],
    scope: ArtifactScope | None,
    session_id: str | None,
    agent_name: str | None,
    environment_name: str | None,
    limit: int | None,
) -> ArtifactListResult:
    artifacts: list[ArtifactMetadata] = []
    with _open_store_root(root, root_identity) as root_fd:
        names = os.listdir(root_fd) if root_fd is not None else os.listdir(root)
        for name in names:
            if _ARTIFACT_ID_PATTERN.fullmatch(name) is None:
                continue
            try:
                artifact = _load_metadata(root / name, parent_fd=root_fd)
            except (FileNotFoundError, ValueError):
                continue
            if scope is not None and artifact.scope != scope:
                continue
            if session_id is not None and artifact.session_id != session_id:
                continue
            if agent_name is not None and artifact.agent_name != agent_name:
                continue
            if environment_name is not None and artifact.environment_name != environment_name:
                continue
            artifacts.append(artifact)

    artifacts.sort(key=lambda artifact: artifact.created_at, reverse=True)
    total_count = len(artifacts)
    truncated = limit is not None and total_count > limit
    if limit is not None:
        artifacts = artifacts[:limit]
    return ArtifactListResult(
        artifacts=tuple(artifacts),
        total_count=total_count,
        truncated=truncated,
    )


def _delete_artifact(
    root: Path,
    root_identity: tuple[int, int],
    artifact_id: str,
) -> None:
    target = _artifact_dir(root, artifact_id)
    with (
        _artifact_ownership_lock(root, target.name),
        _open_store_root(root, root_identity) as root_fd,
    ):
        try:
            with _open_artifact_directory(target, parent_fd=root_fd) as (
                _,
                directory_identity,
            ):
                pass
        except FileNotFoundError:
            return
        _remove_artifact_directory_if_unchanged(
            target,
            directory_identity,
            parent_fd=root_fd,
            ignore_errors=False,
        )


def _artifact_dir(root: Path, artifact_id: str) -> Path:
    try:
        artifact_id = require_clean_nonblank(artifact_id, "artifact_id")
        artifact_id = require_unicode_scalar_text(artifact_id, "artifact_id")
    except ValueError as exc:
        raise InvalidArtifactIdError(str(exc)) from exc
    if _ARTIFACT_ID_PATTERN.fullmatch(artifact_id) is None:
        raise InvalidArtifactIdError(
            "Artifact id must match the local artifact id format `art_` plus 32 lowercase "
            "hexadecimal characters."
        )
    return root / artifact_id


def _load_metadata(
    artifact_dir: Path,
    *,
    parent_fd: int | None = None,
) -> ArtifactMetadata:
    with _open_artifact_directory(artifact_dir, parent_fd=parent_fd) as (
        directory_fd,
        directory_identity,
    ):
        artifact = _load_metadata_from_directory(
            artifact_dir,
            directory_fd,
            directory_identity,
        )
        content_fd = _open_artifact_file(
            directory_fd,
            artifact_dir,
            directory_identity,
            _CONTENT_FILE,
            missing_message=f"Artifact content not found: {artifact_dir.name}",
        )
        try:
            content_size = os.fstat(content_fd).st_size
        finally:
            os.close(content_fd)
        if content_size != artifact.size_bytes:
            raise ValueError(f"Artifact content size does not match metadata: {artifact_dir.name}")
        return artifact


def _load_metadata_from_directory(
    artifact_dir: Path,
    directory_fd: int | None,
    directory_identity: tuple[int, int],
) -> ArtifactMetadata:
    metadata_fd = _open_artifact_file(
        directory_fd,
        artifact_dir,
        directory_identity,
        _METADATA_FILE,
        missing_message=f"Artifact metadata not found: {artifact_dir.name}",
    )
    try:
        with os.fdopen(metadata_fd, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Artifact metadata is not valid JSON: {artifact_dir.name}") from exc
    artifact = ArtifactMetadata.model_validate(payload)
    if artifact.id != artifact_dir.name:
        raise ValueError("Artifact metadata id does not match directory name.")
    return artifact


@contextmanager
def _open_artifact_directory(
    path: Path,
    *,
    parent_fd: int | None = None,
) -> Iterator[tuple[int | None, tuple[int, int]]]:
    before = _stat_directory_entry(path, parent_fd=parent_fd)
    if _is_windows_reparse_point(before) or not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"Artifact path is not a regular directory: {path.name}")
    directory_identity = _stat_identity(before)
    if not _SUPPORTS_DIRECTORY_FD:
        try:
            yield None, directory_identity
        except Exception:
            raise
        else:
            _require_unchanged_directory(
                path,
                directory_identity,
                parent_fd=parent_fd,
            )
        return
    flags = _OPEN_READ_FLAGS | _OPEN_DIRECTORY_FLAG | _OPEN_NOFOLLOW_FLAG
    try:
        if parent_fd is None:
            directory_fd = os.open(path, flags)
        else:
            directory_fd = os.open(path.name, flags, dir_fd=parent_fd)
    except OSError as exc:
        if _is_unsafe_open_error(exc):
            raise ValueError(f"Artifact directory could not be opened safely: {path.name}") from exc
        raise ArtifactStoreUnavailableError(
            f"Artifact directory could not be opened: {path.name}"
        ) from exc
    try:
        after = os.fstat(directory_fd)
        if (
            _is_windows_reparse_point(after)
            or not stat.S_ISDIR(after.st_mode)
            or _stat_identity(before) != _stat_identity(after)
        ):
            raise ValueError(f"Artifact directory changed while it was being opened: {path.name}")
        opened_identity = _stat_identity(after)
        try:
            yield directory_fd, opened_identity
        except Exception:
            raise
        else:
            _require_unchanged_directory(
                path,
                opened_identity,
                parent_fd=parent_fd,
            )
    finally:
        os.close(directory_fd)


def _open_artifact_file(
    directory_fd: int | None,
    directory_path: Path,
    directory_identity: tuple[int, int],
    filename: str,
    *,
    missing_message: str,
) -> int:
    before = _stat_artifact_file(
        directory_fd,
        directory_path,
        directory_identity,
        filename,
        missing_message=missing_message,
    )
    if _is_windows_reparse_point(before) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Artifact file is not a regular file: {filename}")
    flags = _OPEN_READ_FLAGS | _OPEN_NOFOLLOW_FLAG
    try:
        if directory_fd is not None:
            file_fd = os.open(filename, flags, dir_fd=directory_fd)
        else:
            file_fd = os.open(directory_path / filename, flags)
    except FileNotFoundError as exc:
        raise FileNotFoundError(missing_message) from exc
    except OSError as exc:
        if _is_unsafe_open_error(exc):
            raise ValueError(f"Artifact file could not be opened safely: {filename}") from exc
        raise ArtifactStoreUnavailableError(
            f"Artifact file could not be opened: {filename}"
        ) from exc
    try:
        after = os.fstat(file_fd)
        if (
            _is_windows_reparse_point(after)
            or not stat.S_ISREG(after.st_mode)
            or _stat_identity(before) != _stat_identity(after)
        ):
            raise ValueError(f"Artifact file changed while it was being opened: {filename}")
        _require_directory_identity(
            directory_fd,
            directory_path,
            directory_identity,
        )
    except Exception:
        os.close(file_fd)
        raise
    return file_fd


def _stat_artifact_file(
    directory_fd: int | None,
    directory_path: Path,
    directory_identity: tuple[int, int],
    filename: str,
    *,
    missing_message: str,
) -> os.stat_result:
    try:
        if directory_fd is not None:
            result = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        else:
            result = os.stat(directory_path / filename, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(missing_message) from exc
    _require_directory_identity(
        directory_fd,
        directory_path,
        directory_identity,
    )
    return result


def _write_artifact_file(
    directory_fd: int | None,
    directory_path: Path,
    directory_identity: tuple[int, int],
    filename: str,
    content: bytes,
) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
        | _OPEN_NOFOLLOW_FLAG
    )
    if directory_fd is not None:
        file_fd = os.open(filename, flags, 0o600, dir_fd=directory_fd)
    else:
        file_fd = os.open(directory_path / filename, flags, 0o600)
    try:
        _require_directory_identity(
            directory_fd,
            directory_path,
            directory_identity,
        )
        with os.fdopen(file_fd, "wb", closefd=False) as file:
            file.write(content)
            file.flush()
            _sync_descriptor(file_fd)
    finally:
        os.close(file_fd)


def _stat_directory_entry(
    path: Path,
    *,
    parent_fd: int | None,
) -> os.stat_result:
    if parent_fd is None:
        return os.stat(path, follow_symlinks=False)
    return os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)


def _require_directory_identity(
    directory_fd: int | None,
    path: Path,
    expected_identity: tuple[int, int],
) -> None:
    if directory_fd is None:
        _require_unchanged_directory(path, expected_identity)
        return
    current = os.fstat(directory_fd)
    if (
        _is_windows_reparse_point(current)
        or not stat.S_ISDIR(current.st_mode)
        or _stat_identity(current) != expected_identity
    ):
        raise ValueError(f"Artifact directory changed while in use: {path.name}")


def _require_unchanged_directory(
    path: Path,
    expected_identity: tuple[int, int],
    *,
    parent_fd: int | None = None,
) -> None:
    try:
        current = _stat_directory_entry(path, parent_fd=parent_fd)
    except FileNotFoundError as exc:
        raise ValueError(f"Artifact directory disappeared while in use: {path.name}") from exc
    if (
        _is_windows_reparse_point(current)
        or not stat.S_ISDIR(current.st_mode)
        or _stat_identity(current) != expected_identity
    ):
        raise ValueError(f"Artifact directory changed while in use: {path.name}")


def _remove_artifact_directory_if_unchanged(
    path: Path,
    expected_identity: tuple[int, int] | None,
    *,
    parent_fd: int | None = None,
    ignore_errors: bool = True,
) -> None:
    if expected_identity is None:
        return
    try:
        current = _stat_directory_entry(path, parent_fd=parent_fd)
    except FileNotFoundError:
        return
    if (
        not _is_windows_reparse_point(current)
        and stat.S_ISDIR(current.st_mode)
        and _stat_identity(current) == expected_identity
    ):
        if parent_fd is None:
            shutil.rmtree(path, ignore_errors=ignore_errors)
        else:
            shutil.rmtree(path.name, ignore_errors=ignore_errors, dir_fd=parent_fd)


def _stat_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _is_windows_reparse_point(value: os.stat_result) -> bool:
    file_attributes = getattr(value, "st_file_attributes", 0)
    reparse_tag = getattr(value, "st_reparse_tag", 0)
    return bool(file_attributes & _WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT) or bool(reparse_tag)


def _is_unsafe_open_error(exc: OSError) -> bool:
    return exc.errno in {errno.ELOOP, errno.ENOTDIR}
