from __future__ import annotations

import asyncio
from os import PathLike
from pathlib import Path

from cayu._validation import require_clean_nonblank
from cayu.workspaces._local_guard import (
    create_regular,
    delete_regular,
    delete_regular_if_revision,
    open_regular_for_read,
    replace_regular_if_revision,
    write_regular,
)
from cayu.workspaces._mutations import (
    content_identity,
    mutation_result,
    mutation_result_from_identities,
    workspace_path_lock,
)
from cayu.workspaces.base import (
    Workspace,
    WorkspaceListResult,
    WorkspaceMutationResult,
    WorkspaceReadOffsetError,
    WorkspaceReadResult,
    _local_resource_key,
    _validate_workspace_relative_path,
    _WorkspaceListCollector,
    matches_list_pattern,
    validate_list_pattern,
)


class LocalWorkspace(Workspace):
    """Filesystem workspace rooted at one local directory."""

    def __init__(self, root: str | Path, *, workspace_id: str | None = None) -> None:
        if not isinstance(root, str | PathLike):
            raise TypeError("LocalWorkspace root must be a string or Path.")
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            raise FileNotFoundError(f"Workspace root does not exist: {root_path}")
        if not root_path.is_dir():
            raise NotADirectoryError(f"Workspace root is not a directory: {root_path}")

        if workspace_id is None:
            self.id = str(root_path)
        else:
            self.id = require_clean_nonblank(workspace_id, "workspace_id")
        self.root = root_path

    @property
    def resource_key(self) -> tuple[object, ...]:
        return _local_resource_key(self.root)

    def bounded_read_limit(self, max_bytes: int) -> int:
        validated = _validate_limit(max_bytes, "max_bytes")
        if validated is None:
            raise TypeError("Workspace max_bytes must be an integer.")
        return validated

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        relative_path = _validate_workspace_relative_path(path)
        validated_offset = _validate_offset(offset)
        limit = _validate_limit(max_bytes, "max_bytes")
        return await asyncio.to_thread(
            _read_file_locked,
            self.root,
            relative_path,
            validated_offset,
            limit,
        )

    async def write_bytes(self, path: str, content: bytes) -> None:
        if type(content) is not bytes:
            raise TypeError("Workspace write content must be bytes.")
        relative_path = _validate_workspace_relative_path(path)
        await asyncio.to_thread(_write_file, self.root, relative_path, content)

    async def delete(self, path: str) -> None:
        relative_path = _validate_workspace_relative_path(path)
        await asyncio.to_thread(_delete_file, self.root, relative_path)

    async def create_bytes(self, path: str, content: bytes) -> WorkspaceMutationResult:
        if type(content) is not bytes:
            raise TypeError("Workspace create content must be bytes.")
        relative_path = _validate_workspace_relative_path(path)
        return await asyncio.to_thread(_create_file, self.root, relative_path, content)

    async def replace_bytes(
        self,
        path: str,
        content: bytes,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        if type(content) is not bytes:
            raise TypeError("Workspace replace content must be bytes.")
        expected_revision = _validate_revision(expected_revision)
        relative_path = _validate_workspace_relative_path(path)
        return await asyncio.to_thread(
            _replace_file,
            self.root,
            relative_path,
            content,
            expected_revision,
        )

    async def delete_if_revision(
        self,
        path: str,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        expected_revision = _validate_revision(expected_revision)
        relative_path = _validate_workspace_relative_path(path)
        return await asyncio.to_thread(
            _delete_file_if_revision,
            self.root,
            relative_path,
            expected_revision,
        )

    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        pattern = validate_list_pattern(pattern)
        validated_limit = _validate_limit(limit, "limit")

        return await asyncio.to_thread(
            _list_files,
            self.root,
            pattern,
            validated_limit,
        )

    def resolve(self, path: str) -> Path:
        candidate = Path(_validate_workspace_relative_path(path))
        resolved = (self.root / candidate).resolve()
        self._ensure_inside_root(resolved)
        if resolved == self.root:
            raise ValueError("Workspace paths must reference a file.")
        return resolved

    def _ensure_inside_root(self, path: Path) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Workspace path escapes the workspace root.") from exc

    def resolve_no_symlinks(self, path: str) -> Path:
        candidate = Path(_validate_workspace_relative_path(path))
        target = self._resolve_without_symlinks(candidate)
        resolved = target.resolve(strict=False)
        self._ensure_inside_root(resolved)
        if resolved == self.root:
            raise ValueError("Workspace paths must reference a file.")
        return target

    def _resolve_without_symlinks(self, candidate: Path) -> Path:
        current = self.root
        for part in candidate.parts:
            if part in {"", "."}:
                continue
            if part == "..":
                current = (current / part).resolve(strict=False)
                self._ensure_inside_root(current)
                continue
            current = current / part
            if current.is_symlink():
                raise ValueError("Workspace path escapes the workspace root.")
        return current


def _write_file(root: Path, relative_path: str, content: bytes) -> None:
    with workspace_path_lock(root, relative_path):
        write_regular(root, relative_path, content)


def _delete_file(root: Path, relative_path: str) -> None:
    with workspace_path_lock(root, relative_path):
        delete_regular(root, relative_path)


def _read_file_locked(
    root: Path,
    relative_path: str,
    offset: int,
    max_bytes: int | None,
) -> WorkspaceReadResult:
    with workspace_path_lock(root, relative_path):
        return _read_file(root, relative_path, offset, max_bytes)


def _read_file(
    root: Path,
    relative_path: str,
    offset: int,
    max_bytes: int | None,
) -> WorkspaceReadResult:
    with open_regular_for_read(root, relative_path) as (file, total_bytes):
        if offset > total_bytes:
            raise WorkspaceReadOffsetError(offset, total_bytes)
        file.seek(offset)
        content = file.read() if max_bytes is None else file.read(max_bytes)
    complete = offset == 0 and len(content) == total_bytes
    revision, digest = content_identity(content) if complete else (None, None)
    return WorkspaceReadResult(
        content=content,
        total_bytes=total_bytes,
        truncated=offset + len(content) < total_bytes,
        offset=offset,
        revision=revision,
        sha256=digest,
    )


def _create_file(root: Path, relative_path: str, content: bytes) -> WorkspaceMutationResult:
    with workspace_path_lock(root, relative_path):
        create_regular(root, relative_path, content)
        return mutation_result("create", before=None, after=content)


def _replace_file(
    root: Path,
    relative_path: str,
    content: bytes,
    expected_revision: str,
) -> WorkspaceMutationResult:
    with workspace_path_lock(root, relative_path):
        before = replace_regular_if_revision(
            root,
            relative_path,
            content,
            expected_revision,
        )
        return mutation_result_from_identities("replace", before=before, after=content)


def _delete_file_if_revision(
    root: Path,
    relative_path: str,
    expected_revision: str,
) -> WorkspaceMutationResult:
    with workspace_path_lock(root, relative_path):
        before = delete_regular_if_revision(root, relative_path, expected_revision)
        return mutation_result_from_identities("delete", before=before, after=None)


def _list_files(
    root: Path,
    pattern: str,
    limit: int | None,
) -> WorkspaceListResult:
    collector = _WorkspaceListCollector(limit)
    for path in root.rglob("*"):
        if _has_symlink_component(root, path):
            continue
        resolved = path.resolve()
        _ensure_inside_root(root, resolved)
        if resolved == root or not resolved.is_file():
            continue
        if not matches_list_pattern(resolved.relative_to(root).as_posix(), pattern):
            continue
        collector.add(resolved.relative_to(root).as_posix())
    return collector.result(exact_total_when_truncated=False)


def _has_symlink_component(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _ensure_inside_root(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Workspace path escapes the workspace root.") from exc


def _validate_limit(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"Workspace {field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"Workspace {field_name} must be greater than zero.")
    return value


def _validate_offset(value: int) -> int:
    if type(value) is not int:
        raise TypeError("Workspace offset must be an integer.")
    if value < 0:
        raise ValueError("Workspace offset must be non-negative.")
    return value


def _validate_revision(value: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("Workspace expected_revision must be a nonblank string.")
    return value
