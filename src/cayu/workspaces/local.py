from __future__ import annotations

import asyncio
import os
from os import PathLike
from pathlib import Path

from cayu._validation import require_clean_nonblank
from cayu.workspaces.base import (
    Workspace,
    WorkspaceListResult,
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
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        target = self.resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"Workspace file not found: {path}")
        limit = _validate_limit(max_bytes, "max_bytes")
        return await asyncio.to_thread(_read_file, target, limit)

    async def write_bytes(self, path: str, content: bytes) -> None:
        if type(content) is not bytes:
            raise TypeError("Workspace write content must be bytes.")
        target = self.resolve_no_symlinks(path)
        await asyncio.to_thread(_write_file, target, content)

    async def delete(self, path: str) -> None:
        target = self.resolve_no_symlinks(path)
        await asyncio.to_thread(_delete_file, target)

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


def _write_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _delete_file(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_file():
        raise IsADirectoryError(f"Workspace path is not a file: {path}")
    path.unlink()


def _read_file(path: Path, max_bytes: int | None) -> WorkspaceReadResult:
    if max_bytes is None:
        content = path.read_bytes()
        return WorkspaceReadResult(
            content=content,
            total_bytes=len(content),
            truncated=False,
        )
    with path.open("rb") as file:
        chunk = file.read(max_bytes + 1)
        total_bytes = os.fstat(file.fileno()).st_size
    content = chunk[:max_bytes]
    total_bytes = max(total_bytes, len(chunk))
    return WorkspaceReadResult(
        content=content,
        total_bytes=total_bytes,
        truncated=total_bytes > len(content),
    )


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
