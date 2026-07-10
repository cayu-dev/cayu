from __future__ import annotations

import asyncio
import errno
import json
import mimetypes
import os
import re
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Any
from uuid import uuid4

from cayu._validation import (
    copy_json_value,
    require_clean_nonblank,
    require_nonblank,
    require_unicode_scalar_text,
)
from cayu.artifacts.base import (
    ArtifactListResult,
    ArtifactMetadata,
    ArtifactReadResult,
    ArtifactScope,
    ArtifactStore,
    ArtifactStoreUnavailableError,
    InvalidArtifactIdError,
)

_CONTENT_FILE = "content"
_METADATA_FILE = "metadata.json"
_ARTIFACT_ID_PREFIX = "art_"
_ARTIFACT_ID_PATTERN = re.compile(r"\Aart_[0-9a-f]{32}\Z")
_OPEN_READ_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
_OPEN_NOFOLLOW_FLAG = getattr(os, "O_NOFOLLOW", 0)
_OPEN_DIRECTORY_FLAG = getattr(os, "O_DIRECTORY", 0)
_WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_SUPPORTS_DIRECTORY_FD = (
    os.open in os.supports_dir_fd
    and os.mkdir in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.stat in os.supports_follow_symlinks
)


class LocalArtifactStore(ArtifactStore):
    """Local filesystem implementation of ArtifactStore."""

    def __init__(self, root: str | Path, *, store_id: str | None = None) -> None:
        if not isinstance(root, str | PathLike):
            raise TypeError("LocalArtifactStore root must be a string or Path.")
        root_path = Path(root).expanduser().resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        root_stat = os.stat(root_path, follow_symlinks=False)
        if _is_windows_reparse_point(root_stat) or not stat.S_ISDIR(root_stat.st_mode):
            raise NotADirectoryError(f"Artifact store root is not a directory: {root_path}")

        if store_id is None:
            self.id = str(root_path)
        else:
            clean_store_id = require_clean_nonblank(store_id, "store_id")
            self.id = require_unicode_scalar_text(clean_store_id, "store_id")
        self.root = root_path
        self._root_identity = _stat_identity(root_stat)

    async def put_bytes(
        self,
        content: bytes,
        *,
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
        copied_metadata = copy_json_value(metadata or {}, "metadata")

        artifact = ArtifactMetadata(
            id=_new_artifact_id(),
            filename=filename,
            content_type=content_type,
            size_bytes=len(content),
            scope=scope,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=copied_metadata,
        )
        try:
            await asyncio.to_thread(
                _write_artifact,
                self.root,
                self._root_identity,
                artifact,
                content,
            )
        except (ArtifactStoreUnavailableError, FileExistsError, TypeError, ValueError):
            raise
        except OSError as exc:
            raise ArtifactStoreUnavailableError(
                "Local artifact store could not write artifact content."
            ) from exc
        return artifact

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


def _new_artifact_id() -> str:
    return f"{_ARTIFACT_ID_PREFIX}{uuid4().hex}"


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
) -> None:
    target = _artifact_dir(root, artifact.id)
    with _open_store_root(root, root_identity) as root_fd:
        try:
            if root_fd is None:
                target.mkdir(mode=0o700, parents=False)
            else:
                os.mkdir(artifact.id, mode=0o700, dir_fd=root_fd)
        except FileExistsError as exc:
            raise FileExistsError(f"Artifact already exists: {artifact.id}") from exc

        created_identity = _stat_identity(_stat_directory_entry(target, parent_fd=root_fd))
        try:
            with _open_artifact_directory(target, parent_fd=root_fd) as (
                directory_fd,
                directory_identity,
            ):
                _write_artifact_file(
                    directory_fd,
                    target,
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
                    target,
                    directory_identity,
                    _METADATA_FILE,
                    metadata_bytes,
                )
        except Exception:
            _remove_artifact_directory_if_unchanged(
                target,
                created_identity,
                parent_fd=root_fd,
            )
            raise


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
    with _open_store_root(root, root_identity) as root_fd:
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
        return _load_metadata_from_directory(
            artifact_dir,
            directory_fd,
            directory_identity,
        )


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
