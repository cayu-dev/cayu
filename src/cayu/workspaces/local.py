from __future__ import annotations

import asyncio
import os
from os import PathLike
from pathlib import Path

from cayu._validation import require_clean_nonblank, require_nonblank
from cayu.workspaces.base import Workspace, WorkspaceListResult, WorkspaceReadResult


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
        target = self.resolve(path)
        await asyncio.to_thread(_write_file, target, content)

    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        pattern = require_nonblank(pattern, "pattern")
        if Path(pattern).is_absolute() or _has_parent_reference(pattern):
            raise ValueError("Workspace list pattern must stay inside the workspace.")
        validated_limit = _validate_limit(limit, "limit")

        return await asyncio.to_thread(
            _list_files,
            self.root,
            pattern,
            validated_limit,
        )

    def resolve(self, path: str) -> Path:
        path = require_nonblank(path, "path")
        candidate = Path(path)
        if candidate.is_absolute():
            raise ValueError("Workspace paths must be relative.")
        resolved = (self.root / candidate).resolve()
        self._ensure_inside_root(resolved)
        return resolved

    def _ensure_inside_root(self, path: Path) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Workspace path escapes the workspace root.") from exc


def _has_parent_reference(pattern: str) -> bool:
    return ".." in Path(pattern).parts


def _write_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


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
    paths: list[str] = []
    total_count = 0
    for path in root.glob(pattern):
        resolved = path.resolve()
        _ensure_inside_root(root, resolved)
        if resolved == root or not resolved.is_file():
            continue
        total_count += 1
        if limit is None or len(paths) < limit:
            paths.append(resolved.relative_to(root).as_posix())
        elif limit is not None:
            return WorkspaceListResult(
                paths=tuple(sorted(paths)),
                total_count=None,
                truncated=True,
            )
    return WorkspaceListResult(
        paths=tuple(sorted(paths)),
        total_count=total_count,
        truncated=False,
    )


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
