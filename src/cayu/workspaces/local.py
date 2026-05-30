from __future__ import annotations

import asyncio
from os import PathLike
from pathlib import Path

from cayu._validation import require_nonblank
from cayu.workspaces.base import Workspace


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
            self.id = require_nonblank(workspace_id, "workspace_id")
        self.root = root_path

    async def read_bytes(self, path: str) -> bytes:
        target = self.resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"Workspace file not found: {path}")
        return await asyncio.to_thread(target.read_bytes)

    async def write_bytes(self, path: str, content: bytes) -> None:
        if type(content) is not bytes:
            raise TypeError("Workspace write content must be bytes.")
        target = self.resolve(path)
        await asyncio.to_thread(_write_file, target, content)

    async def list(self, pattern: str = "**/*") -> list[str]:
        pattern = require_nonblank(pattern, "pattern")
        if Path(pattern).is_absolute() or _has_parent_reference(pattern):
            raise ValueError("Workspace list pattern must stay inside the workspace.")

        return await asyncio.to_thread(_list_files, self.root, pattern)

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


def _list_files(root: Path, pattern: str) -> list[str]:
    paths: list[str] = []
    for path in root.glob(pattern):
        resolved = path.resolve()
        _ensure_inside_root(root, resolved)
        if resolved == root or not resolved.is_file():
            continue
        paths.append(resolved.relative_to(root).as_posix())
    return sorted(paths)


def _ensure_inside_root(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Workspace path escapes the workspace root.") from exc
