from __future__ import annotations

import fnmatch
import posixpath
from pathlib import PurePosixPath
from typing import Any

from cayu._validation import require_clean_nonblank, require_nonblank
from cayu.runners import DEFAULT_MICROSANDBOX_CWD, MicrosandboxRunner
from cayu.workspaces.base import Workspace, WorkspaceListResult, WorkspaceReadResult

DEFAULT_MICROSANDBOX_WORKSPACE_READ_LIMIT_BYTES = 256 * 1024
DEFAULT_MICROSANDBOX_WORKSPACE_LIST_LIMIT = 500


class MicrosandboxWorkspace(Workspace):
    """Workspace implementation backed by Microsandbox's native filesystem API."""

    def __init__(
        self,
        runner: MicrosandboxRunner,
        *,
        root: str = DEFAULT_MICROSANDBOX_CWD,
        workspace_id: str | None = None,
        default_read_limit_bytes: int = DEFAULT_MICROSANDBOX_WORKSPACE_READ_LIMIT_BYTES,
        default_list_limit: int = DEFAULT_MICROSANDBOX_WORKSPACE_LIST_LIMIT,
    ) -> None:
        if not isinstance(runner, MicrosandboxRunner):
            raise TypeError("MicrosandboxWorkspace runner must be a MicrosandboxRunner.")
        self.runner = runner
        self.root = _validate_guest_root(root)
        self.default_read_limit_bytes = _validate_required_limit(
            default_read_limit_bytes,
            "default_read_limit_bytes",
        )
        self.default_list_limit = _validate_required_limit(default_list_limit, "default_list_limit")
        if workspace_id is None:
            self.id = f"microsandbox:{runner.name}:{self.root}"
        else:
            self.id = require_clean_nonblank(workspace_id, "workspace_id")

    async def read_bytes(
        self,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        guest_path = self.resolve(path)
        checked_path = await self._require_contained_real_path(guest_path)
        limit = (
            self.default_read_limit_bytes
            if max_bytes is None
            else _validate_required_limit(max_bytes, "max_bytes")
        )
        fs = self._filesystem()
        metadata = await _stat_file(fs, checked_path, original_path=path)
        content = await _read_limited(fs, checked_path, limit)
        total_bytes = _metadata_size(metadata)
        return WorkspaceReadResult(
            content=content,
            total_bytes=max(total_bytes, len(content)),
            truncated=total_bytes > len(content),
        )

    async def write_bytes(self, path: str, content: bytes) -> None:
        guest_path = self.resolve(path)
        if type(content) is not bytes:
            raise TypeError("Workspace write content must be bytes.")
        fs = self._filesystem()
        await _mkdir_parents(fs, posixpath.dirname(guest_path), stop_at=self.root)
        parent_path = posixpath.dirname(guest_path)
        await self._require_contained_real_path(parent_path)
        existing_path = await self._optional_contained_real_path(guest_path)
        await fs.write(existing_path or guest_path, content)

    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        pattern = _validate_list_pattern(pattern)
        effective_limit = (
            self.default_list_limit if limit is None else _validate_required_limit(limit, "limit")
        )
        fs = self._filesystem()
        await self._require_contained_real_path(self.root)
        paths: list[str] = []
        total_count = 0
        async for rel_path in _iter_files(self, fs, self.root):
            if _matches_pattern(rel_path, pattern):
                total_count += 1
                if len(paths) < effective_limit:
                    paths.append(rel_path)
        return WorkspaceListResult(
            paths=tuple(sorted(paths)),
            total_count=total_count,
            truncated=total_count > len(paths),
        )

    def resolve(self, path: str) -> str:
        rel_path = _validate_relative_path(path)
        resolved = posixpath.normpath(posixpath.join(self.root, rel_path))
        if not _is_same_or_child(resolved, self.root):
            raise ValueError("Workspace path escapes the workspace root.")
        return resolved

    def _filesystem(self) -> Any:
        return self.runner.filesystem()

    async def _require_contained_real_path(self, guest_path: str) -> str:
        try:
            real_path = await self.runner.real_path(guest_path)
        except Exception as exc:
            if _is_path_not_found_error(exc):
                raise FileNotFoundError(f"Workspace path not found: {guest_path}") from exc
            raise RuntimeError(
                f"Failed to resolve Microsandbox workspace path: {guest_path}: {exc}"
            ) from exc
        _ensure_real_path_inside_root(real_path, self.root)
        return real_path

    async def _optional_contained_real_path(self, guest_path: str) -> str | None:
        try:
            real_path = await self.runner.real_path(guest_path)
        except Exception as exc:
            if _is_path_not_found_error(exc):
                return None
            raise RuntimeError(
                f"Failed to resolve Microsandbox workspace path: {guest_path}: {exc}"
            ) from exc
        _ensure_real_path_inside_root(real_path, self.root)
        return real_path


async def _stat_file(fs: Any, guest_path: str, *, original_path: str) -> Any:
    try:
        metadata = await fs.stat(guest_path)
    except Exception as exc:
        if _is_path_not_found_error(exc):
            raise FileNotFoundError(f"Workspace file not found: {original_path}") from exc
        raise RuntimeError(
            f"Failed to stat Microsandbox workspace file: {original_path}: {exc}"
        ) from exc
    kind = getattr(metadata, "kind", None)
    if kind not in {"file", "regular"}:
        raise FileNotFoundError(f"Workspace file not found: {original_path}")
    return metadata


async def _read_limited(fs: Any, guest_path: str, limit: int) -> bytes:
    read_stream = getattr(fs, "read_stream", None)
    if read_stream is not None:
        chunks = bytearray()
        stream = await read_stream(guest_path)
        async for chunk in stream:
            if type(chunk) is not bytes:
                raise TypeError("Microsandbox filesystem read_stream yielded non-bytes data.")
            remaining = limit - len(chunks)
            if remaining <= 0:
                break
            chunks.extend(chunk[:remaining])
            if len(chunk) > remaining:
                break
        return bytes(chunks)
    content = await fs.read(guest_path)
    if type(content) is not bytes:
        raise TypeError("Microsandbox filesystem read returned non-bytes data.")
    return content[:limit]


async def _mkdir_parents(fs: Any, guest_path: str, *, stop_at: str) -> None:
    if guest_path in {"", "/", stop_at}:
        return
    if not _is_same_or_child(guest_path, stop_at):
        raise ValueError("Workspace path escapes the workspace root.")
    current = stop_at
    rel_path = posixpath.relpath(guest_path, stop_at)
    for part in rel_path.split("/"):
        if not part or part == ".":
            continue
        current = posixpath.join(current, part)
        try:
            await fs.mkdir(current)
        except Exception as exc:
            if await _is_existing_directory(fs, current):
                continue
            raise RuntimeError(
                f"Failed to create Microsandbox workspace directory: {current}: {exc}"
            ) from exc


async def _is_existing_directory(fs: Any, guest_path: str) -> bool:
    try:
        metadata = await fs.stat(guest_path)
    except Exception:
        return False
    kind = getattr(metadata, "kind", None)
    return kind in {"dir", "directory"}


async def _iter_files(workspace: MicrosandboxWorkspace, fs: Any, root: str):
    pending = [root]
    seen_dirs: set[str] = set()
    seen_files: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen_dirs:
            continue
        seen_dirs.add(current)
        try:
            entries = await fs.list(current)
        except Exception as exc:
            raise RuntimeError(f"Failed to list Microsandbox workspace path: {current}") from exc
        for entry in entries:
            guest_path = _entry_guest_path(current, getattr(entry, "path", None))
            if guest_path is None or not _is_same_or_child(guest_path, root):
                continue
            real_path = await workspace._require_contained_real_path(guest_path)
            kind = getattr(entry, "kind", None)
            if kind in {"dir", "directory"}:
                pending.append(real_path)
            elif kind in {"file", "regular"}:
                rel_path = posixpath.relpath(real_path, root)
                if rel_path not in seen_files:
                    seen_files.add(rel_path)
                    yield rel_path


def _entry_guest_path(current_dir: str, entry_path: Any) -> str | None:
    if type(entry_path) is not str or not entry_path:
        return None
    if posixpath.isabs(entry_path):
        return posixpath.normpath(entry_path)
    return posixpath.normpath(posixpath.join(current_dir, entry_path))


def _metadata_size(metadata: Any) -> int:
    size = getattr(metadata, "size", None)
    if type(size) is not int:
        raise TypeError("Microsandbox filesystem metadata missing integer size.")
    if size < 0:
        raise ValueError("Microsandbox filesystem metadata size must be non-negative.")
    return size


def _validate_guest_root(path: str) -> str:
    root = require_clean_nonblank(path, "root")
    if not posixpath.isabs(root):
        raise ValueError("MicrosandboxWorkspace root must be an absolute guest path.")
    return posixpath.normpath(root)


def _validate_relative_path(path: str) -> str:
    value = require_nonblank(path, "path")
    if posixpath.isabs(value):
        raise ValueError("Workspace paths must be relative.")
    normalized = posixpath.normpath(value)
    if normalized in {"", "."}:
        raise ValueError("Workspace paths must reference a file.")
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError("Workspace path escapes the workspace root.")
    return normalized


def _validate_list_pattern(pattern: str) -> str:
    value = require_nonblank(pattern, "pattern")
    if posixpath.isabs(value):
        raise ValueError("Workspace list pattern must stay inside the workspace.")
    parts = tuple(part for part in value.split("/") if part)
    if ".." in parts:
        raise ValueError("Workspace list pattern must stay inside the workspace.")
    return value


def _validate_required_limit(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"MicrosandboxWorkspace {field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"MicrosandboxWorkspace {field_name} must be greater than zero.")
    return value


def _is_same_or_child(path: str, root: str) -> bool:
    if root == "/":
        return posixpath.isabs(path)
    return path == root or path.startswith(f"{root.rstrip('/')}/")


def _ensure_real_path_inside_root(real_path: str, root: str) -> None:
    normalized = posixpath.normpath(real_path)
    if not _is_same_or_child(normalized, root):
        raise ValueError("Workspace path escapes the workspace root.")


def _is_path_not_found_error(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    code = getattr(exc, "code", None)
    if code == "path-not-found":
        return True
    if type(exc).__name__ == "PathNotFoundError":
        return True
    # microsandbox 0.5.x SFTP real_path raises a generic error whose message
    # carries the ENOENT text rather than a typed not-found error.
    return "no such file" in str(exc).lower()


def _matches_pattern(path: str, pattern: str) -> bool:
    return (
        PurePosixPath(path).match(pattern)
        or fnmatch.fnmatchcase(path, pattern)
        or (pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:]))
    )
