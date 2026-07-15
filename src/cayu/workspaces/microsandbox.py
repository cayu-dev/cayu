from __future__ import annotations

import posixpath
from typing import Any

from cayu._validation import require_clean_nonblank
from cayu.runners import DEFAULT_MICROSANDBOX_CWD, MicrosandboxRunner
from cayu.workspaces._guest_guard import guard_delete, guard_read, guard_write
from cayu.workspaces.base import (
    Workspace,
    WorkspaceListResult,
    WorkspaceReadResult,
    _runner_workspace_resource_key,
    _validate_absolute_guest_root,
    _validate_workspace_positive_limit,
    _validate_workspace_relative_path,
    _WorkspaceListCollector,
    matches_list_pattern,
    validate_list_pattern,
)

DEFAULT_MICROSANDBOX_WORKSPACE_READ_LIMIT_BYTES = 256 * 1024
DEFAULT_MICROSANDBOX_WORKSPACE_LIST_LIMIT = 500


class MicrosandboxWorkspace(Workspace):
    """Workspace backed by a Microsandbox sandbox.

    ``read_bytes``/``write_bytes``/``delete`` run through a guest-side guard
    program (see :mod:`cayu.workspaces._guest_guard`) that resolves and opens
    every path component atomically with ``O_NOFOLLOW`` inside the sandbox, so
    a co-resident guest process cannot race a host-side ``realpath`` check
    (TOCTOU). Those operations require ``python3`` inside the guest. ``list``
    uses Microsandbox's native filesystem API with best-effort ``realpath``
    containment; any subsequent read is re-checked by the guard.
    """

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

    @property
    def resource_key(self) -> tuple[object, ...] | None:
        return _runner_workspace_resource_key(self.runner, str(self.root))

    def bounded_read_limit(self, max_bytes: int) -> int:
        return min(
            self.default_read_limit_bytes,
            _validate_required_limit(max_bytes, "max_bytes"),
        )

    async def read_bytes(
        self,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        rel_path = self._contained_rel_path(path)
        limit = (
            self.default_read_limit_bytes
            if max_bytes is None
            else _validate_required_limit(max_bytes, "max_bytes")
        )
        content, total_bytes = await guard_read(
            self.runner,
            root=self.root,
            rel_path=rel_path,
            limit=limit,
            original_path=path,
            backend="Microsandbox",
        )
        return WorkspaceReadResult(
            content=content,
            total_bytes=max(total_bytes, len(content)),
            truncated=total_bytes > len(content),
        )

    async def write_bytes(self, path: str, content: bytes) -> None:
        rel_path = self._contained_rel_path(path)
        if type(content) is not bytes:
            raise TypeError("Workspace write content must be bytes.")
        await guard_write(
            self.runner,
            root=self.root,
            rel_path=rel_path,
            content=content,
            original_path=path,
            backend="Microsandbox",
        )

    async def delete(self, path: str) -> None:
        rel_path = self._contained_rel_path(path)
        await guard_delete(
            self.runner,
            root=self.root,
            rel_path=rel_path,
            original_path=path,
            backend="Microsandbox",
        )

    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        pattern = validate_list_pattern(pattern)
        effective_limit = (
            self.default_list_limit if limit is None else _validate_required_limit(limit, "limit")
        )
        fs = self._filesystem()
        await self._require_contained_real_path(self.root)
        collector = _WorkspaceListCollector(effective_limit)
        async for rel_path in _iter_files(self, fs, self.root):
            if matches_list_pattern(rel_path, pattern):
                collector.add(rel_path)
        return collector.result()

    def resolve(self, path: str) -> str:
        rel_path = _validate_relative_path(path)
        resolved = posixpath.normpath(posixpath.join(self.root, rel_path))
        if not _is_same_or_child(resolved, self.root):
            raise ValueError("Workspace path escapes the workspace root.")
        return resolved

    def _filesystem(self) -> Any:
        return self.runner.filesystem()

    def _contained_rel_path(self, path: str) -> str:
        return posixpath.relpath(self.resolve(path), self.root)

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


def _validate_guest_root(path: str) -> str:
    return _validate_absolute_guest_root(path, owner="MicrosandboxWorkspace")


def _validate_relative_path(path: str) -> str:
    return _validate_workspace_relative_path(path)


def _validate_required_limit(value: int, field_name: str) -> int:
    return _validate_workspace_positive_limit(
        value,
        field_name,
        owner="MicrosandboxWorkspace",
    )


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
    # Microsandbox SFTP real_path raises a generic error whose message carries
    # the ENOENT text rather than a typed not-found error.
    return "no such file" in str(exc).lower()
