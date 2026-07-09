from __future__ import annotations

import math
import posixpath
from typing import Any

from cayu._validation import require_clean_nonblank, require_nonblank
from cayu.runners import DEFAULT_E2B_CWD, E2BRunner
from cayu.workspaces._guest_guard import guard_delete, guard_read, guard_write
from cayu.workspaces.base import (
    Workspace,
    WorkspaceListResult,
    WorkspaceReadResult,
    _runner_workspace_resource_key,
    matches_list_pattern,
    validate_list_pattern,
)

DEFAULT_E2B_WORKSPACE_READ_LIMIT_BYTES = 256 * 1024
DEFAULT_E2B_WORKSPACE_LIST_LIMIT = 500
DEFAULT_E2B_WORKSPACE_LIST_DEPTH = 64


class E2BWorkspace(Workspace):
    """Workspace backed by an E2B sandbox.

    ``read_bytes``/``write_bytes``/``delete`` run through a guest-side guard
    program (see :mod:`cayu.workspaces._guest_guard`) that resolves and opens
    every path component atomically with ``O_NOFOLLOW`` inside the sandbox, so
    a co-resident guest process cannot race a host-side symlink check
    (TOCTOU). Those operations require ``python3`` inside the guest and run as
    the runner's default exec user (the ``user`` override applies only to
    ``list``). ``list`` uses E2B's native filesystem API and is advisory:
    symlinked entries are skipped best-effort, and any subsequent read is
    re-checked by the guard.
    """

    def __init__(
        self,
        runner: E2BRunner,
        *,
        root: str = DEFAULT_E2B_CWD,
        workspace_id: str | None = None,
        default_read_limit_bytes: int = DEFAULT_E2B_WORKSPACE_READ_LIMIT_BYTES,
        default_list_limit: int = DEFAULT_E2B_WORKSPACE_LIST_LIMIT,
        default_list_depth: int = DEFAULT_E2B_WORKSPACE_LIST_DEPTH,
        user: str | None = None,
        request_timeout_s: float | None = None,
    ) -> None:
        if not isinstance(runner, E2BRunner):
            raise TypeError("E2BWorkspace runner must be an E2BRunner.")
        self.runner = runner
        self.root = _validate_guest_root(root)
        self.default_read_limit_bytes = _validate_required_limit(
            default_read_limit_bytes,
            "default_read_limit_bytes",
        )
        self.default_list_limit = _validate_required_limit(default_list_limit, "default_list_limit")
        self.default_list_depth = _validate_required_limit(default_list_depth, "default_list_depth")
        self.user = _validate_optional_user(user)
        self.request_timeout_s = _validate_optional_timeout(request_timeout_s)
        if workspace_id is None:
            self.id = f"e2b:{runner.sandbox_id}:{self.root}"
        else:
            self.id = require_clean_nonblank(workspace_id, "workspace_id")

    @property
    def resource_key(self) -> tuple[object, ...] | None:
        return _runner_workspace_resource_key(self.runner, str(self.root))

    async def read_bytes(
        self,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        rel_path = self._contained_rel_path(path)
        content, total_bytes = await guard_read(
            self.runner,
            root=self.root,
            rel_path=rel_path,
            limit=self._effective_read_limit(max_bytes),
            original_path=path,
            backend="E2B",
            timeout_s=self._guard_timeout_s(),
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
            backend="E2B",
            timeout_s=self._guard_timeout_s(),
        )

    async def delete(self, path: str) -> None:
        rel_path = self._contained_rel_path(path)
        await guard_delete(
            self.runner,
            root=self.root,
            rel_path=rel_path,
            original_path=path,
            backend="E2B",
            timeout_s=self._guard_timeout_s(),
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
        await self._reject_symlink_path(self.root, allow_missing_suffix=False)
        try:
            entries = await self._filesystem().list(
                self.root,
                depth=self.default_list_depth,
                user=self.user,
                request_timeout=self.request_timeout_s,
            )
        except Exception as exc:
            if _is_path_not_found_error(exc):
                raise FileNotFoundError(f"Workspace path not found: {self.root}") from exc
            raise RuntimeError(f"Failed to list E2B workspace path: {self.root}") from exc

        paths: list[str] = []
        total_count = 0
        for entry in entries:
            if _entry_type(entry) != "file" or _is_symlink(entry):
                continue
            guest_path = _entry_guest_path(getattr(entry, "path", None))
            if guest_path is None or not _is_same_or_child(guest_path, self.root):
                continue
            rel_path = posixpath.relpath(guest_path, self.root)
            if matches_list_pattern(rel_path, pattern):
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

    def _contained_rel_path(self, path: str) -> str:
        return posixpath.relpath(self.resolve(path), self.root)

    def _guard_timeout_s(self) -> int | None:
        if self.request_timeout_s is None:
            return None
        return max(1, math.ceil(self.request_timeout_s))

    def _effective_read_limit(self, max_bytes: int | None) -> int:
        if max_bytes is None:
            return self.default_read_limit_bytes
        return _validate_required_limit(max_bytes, "max_bytes")

    async def _reject_symlink_path(self, guest_path: str, *, allow_missing_suffix: bool) -> None:
        current = "/"
        parts = [part for part in guest_path.split("/") if part]
        for part in parts:
            current = posixpath.join(current, part)
            fs = self._filesystem()
            try:
                info = await fs.get_info(
                    current,
                    user=self.user,
                    request_timeout=self.request_timeout_s,
                )
            except Exception as exc:
                if allow_missing_suffix and _is_path_not_found_error(exc):
                    return
                if _is_path_not_found_error(exc):
                    raise FileNotFoundError(f"Workspace path not found: {guest_path}") from exc
                raise RuntimeError(f"Failed to inspect E2B workspace path: {current}") from exc
            if _is_symlink(info):
                raise ValueError("Workspace path escapes the workspace root.")


def _entry_guest_path(entry_path: Any) -> str | None:
    if type(entry_path) is not str or not entry_path:
        return None
    if not posixpath.isabs(entry_path):
        return None
    return posixpath.normpath(entry_path)


def _entry_type(entry: Any) -> str | None:
    value = getattr(entry, "type", None)
    if type(value) is str:
        return value
    enum_value = getattr(value, "value", None)
    if type(enum_value) is str:
        return enum_value
    return None


def _is_symlink(entry: Any) -> bool:
    symlink_target = getattr(entry, "symlink_target", None)
    return type(symlink_target) is str and bool(symlink_target)


def _validate_guest_root(path: str) -> str:
    root = require_clean_nonblank(path, "root")
    if not posixpath.isabs(root):
        raise ValueError("E2BWorkspace root must be an absolute guest path.")
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


def _validate_required_limit(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"E2BWorkspace {field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"E2BWorkspace {field_name} must be greater than zero.")
    return value


def _validate_optional_user(user: str | None) -> str | None:
    if user is None:
        return None
    return require_clean_nonblank(user, "user")


def _validate_optional_timeout(timeout_s: float | None) -> float | None:
    if timeout_s is None:
        return None
    if type(timeout_s) not in {int, float}:
        raise TypeError("E2BWorkspace request_timeout_s must be numeric.")
    if timeout_s <= 0:
        raise ValueError("E2BWorkspace request_timeout_s must be greater than zero.")
    return float(timeout_s)


def _is_same_or_child(path: str, root: str) -> bool:
    if root == "/":
        return posixpath.isabs(path)
    return path == root or path.startswith(f"{root.rstrip('/')}/")


def _is_path_not_found_error(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    return type(exc).__name__ in {"FileNotFoundException", "PathNotFoundError"}
