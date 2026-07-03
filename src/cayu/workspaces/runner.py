from __future__ import annotations

import base64
import json
import posixpath
from collections.abc import Sequence
from typing import Any

from cayu._validation import require_clean_nonblank, require_nonblank
from cayu.runners import ExecCommand, Runner
from cayu.workspaces.base import (
    Workspace,
    WorkspaceListResult,
    WorkspaceReadResult,
    translate_list_pattern,
    validate_list_pattern,
)

DEFAULT_RUNNER_WORKSPACE_READ_LIMIT_BYTES = 256 * 1024
DEFAULT_RUNNER_WORKSPACE_LIST_LIMIT = 500
RUNNER_WORKSPACE_SCRIPT_OUTPUT_OVERHEAD_BYTES = 4096
# Per-file tar framing bound: header blocks (including pax long-name records)
# plus up to 512 bytes of content padding per member; the archive itself adds
# end-of-archive blocks.
RUNNER_WORKSPACE_TAR_MEMBER_OVERHEAD_BYTES = 3072
RUNNER_WORKSPACE_TAR_ARCHIVE_OVERHEAD_BYTES = 1024

_READ_SCRIPT = r"""
import base64
import json
import os
import pathlib
import sys


def fail(error_type, message):
    print(json.dumps({"ok": False, "error_type": error_type, "message": message}))
    sys.exit(1)


try:
    rel_path = sys.argv[1]
    max_bytes_raw = sys.argv[2]
    max_bytes = None if max_bytes_raw == "" else int(max_bytes_raw)
    root = pathlib.Path.cwd().resolve()
    candidate = pathlib.Path(rel_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        fail("invalid_path", "Workspace paths must stay inside the workspace.")
    target = (root / candidate).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        fail("invalid_path", "Workspace path escapes the workspace root.")
    if not target.is_file():
        fail("not_found", f"Workspace file not found: {rel_path}")
    total_bytes = target.stat().st_size
    if max_bytes is None:
        content = target.read_bytes()
    else:
        with target.open("rb") as file:
            content = file.read(max_bytes)
    print(json.dumps({
        "ok": True,
        "content_base64": base64.b64encode(content).decode("ascii"),
        "total_bytes": total_bytes,
    }))
except Exception as exc:
    fail("workspace_error", str(exc))
"""

_WRITE_SCRIPT = r"""
import base64
import json
import pathlib
import sys


def fail(error_type, message):
    print(json.dumps({"ok": False, "error_type": error_type, "message": message}))
    sys.exit(1)


try:
    payload = json.loads(sys.stdin.read())
    rel_path = payload["path"]
    content = base64.b64decode(payload["content_base64"], validate=True)
    root = pathlib.Path.cwd().resolve()
    candidate = pathlib.Path(rel_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        fail("invalid_path", "Workspace paths must stay inside the workspace.")
    target = root
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        target = target / part
        if target.is_symlink():
            fail("invalid_path", "Workspace path escapes the workspace root.")
    resolved = target.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        fail("invalid_path", "Workspace path escapes the workspace root.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    print(json.dumps({"ok": True, "bytes": len(content)}))
except Exception as exc:
    fail("workspace_error", str(exc))
"""

_DELETE_SCRIPT = r"""
import json
import pathlib
import sys


def fail(error_type, message):
    print(json.dumps({"ok": False, "error_type": error_type, "message": message}))
    sys.exit(1)


try:
    rel_path = sys.argv[1]
    root = pathlib.Path.cwd().resolve()
    candidate = pathlib.Path(rel_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        fail("invalid_path", "Workspace paths must stay inside the workspace.")
    target = root
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        target = target / part
        if target.is_symlink():
            fail("invalid_path", "Workspace path escapes the workspace root.")
    resolved = target.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        fail("invalid_path", "Workspace path escapes the workspace root.")
    if not target.exists():
        print(json.dumps({"ok": True, "deleted": False}))
    elif not target.is_file():
        fail("not_file", f"Workspace path is not a file: {rel_path}")
    else:
        target.unlink()
        print(json.dumps({"ok": True, "deleted": True}))
except Exception as exc:
    fail("workspace_error", str(exc))
"""

# The pattern regex is translated host-side by cayu.workspaces.base
# translate_list_pattern so every backend shares one normative matcher
# regardless of the guest Python version.
_LIST_SCRIPT = r"""
import json
import pathlib
import re
import sys


def fail(error_type, message):
    print(json.dumps({"ok": False, "error_type": error_type, "message": message}))
    sys.exit(1)


try:
    pattern_regex = re.compile(sys.argv[1])
    limit_raw = sys.argv[2]
    limit = None if limit_raw == "" else int(limit_raw)
    root = pathlib.Path.cwd().resolve()
    matches = []
    for path in root.rglob("*"):
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            continue
        current = root
        has_symlink = False
        for part in rel_parts:
            current = current / part
            if current.is_symlink():
                has_symlink = True
                break
        if has_symlink:
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if not resolved.is_file():
            continue
        rel_path = resolved.relative_to(root).as_posix()
        if pattern_regex.fullmatch(rel_path):
            matches.append(rel_path)
    sorted_matches = sorted(matches)
    paths = sorted_matches if limit is None else sorted_matches[:limit]
    print(json.dumps({"ok": True, "paths": paths, "total_count": len(matches)}))
except Exception as exc:
    fail("workspace_error", str(exc))
"""


_READ_TAR_SCRIPT = r"""
import base64
import io
import json
import pathlib
import sys
import tarfile


def fail(error_type, message):
    print(json.dumps({"ok": False, "error_type": error_type, "message": message}))
    sys.exit(1)


try:
    payload = json.loads(sys.stdin.read())
    rel_paths = payload["paths"]
    max_file_bytes = payload["max_file_bytes"]
    root = pathlib.Path.cwd().resolve()
    buffer = io.BytesIO()
    total_bytes = 0
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for rel_path in rel_paths:
            candidate = pathlib.Path(rel_path)
            if candidate.is_absolute() or ".." in candidate.parts:
                fail("invalid_path", "Workspace paths must stay inside the workspace.")
            target = (root / candidate).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                fail("invalid_path", "Workspace path escapes the workspace root.")
            if not target.is_file():
                fail("not_found", f"Workspace file not found: {rel_path}")
            size = target.stat().st_size
            if max_file_bytes is not None and size > max_file_bytes:
                fail(
                    "workspace_error",
                    f"Workspace file exceeds max_file_bytes={max_file_bytes}: {rel_path}",
                )
            info = tarfile.TarInfo(name=rel_path)
            info.size = size
            with target.open("rb") as file:
                archive.addfile(info, file)
            total_bytes += size
    print(json.dumps({
        "ok": True,
        "tar_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "file_count": len(rel_paths),
        "total_bytes": total_bytes,
    }))
except Exception as exc:
    fail("workspace_error", str(exc))
"""

_WRITE_TAR_SCRIPT = r"""
import base64
import io
import json
import pathlib
import sys
import tarfile


def fail(error_type, message):
    print(json.dumps({"ok": False, "error_type": error_type, "message": message}))
    sys.exit(1)


try:
    payload = json.loads(sys.stdin.read())
    data = base64.b64decode(payload["tar_base64"], validate=True)
    root = pathlib.Path.cwd().resolve()
    written_files = 0
    written_bytes = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as archive:
        for member in archive.getmembers():
            if not member.isreg():
                fail(
                    "invalid_path",
                    f"Workspace tar member must be a regular file: {member.name}",
                )
            candidate = pathlib.Path(member.name)
            if candidate.is_absolute() or ".." in candidate.parts:
                fail("invalid_path", "Workspace paths must stay inside the workspace.")
            target = root
            for part in candidate.parts:
                if part in {"", "."}:
                    continue
                target = target / part
                if target.is_symlink():
                    fail("invalid_path", "Workspace path escapes the workspace root.")
            resolved = target.resolve(strict=False)
            try:
                resolved.relative_to(root)
            except ValueError:
                fail("invalid_path", "Workspace path escapes the workspace root.")
            extracted = archive.extractfile(member)
            if extracted is None:
                fail("workspace_error", f"Workspace tar member could not be read: {member.name}")
            content = extracted.read()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            written_files += 1
            written_bytes += len(content)
    print(json.dumps({"ok": True, "files": written_files, "bytes": written_bytes}))
except Exception as exc:
    fail("workspace_error", str(exc))
"""


class RunnerWorkspace(Workspace):
    """Workspace implementation whose file operations execute through a runner."""

    def __init__(
        self,
        runner: Runner,
        *,
        cwd: str | None = None,
        workspace_id: str | None = None,
        python_executable: str = "python3",
        default_read_limit_bytes: int = DEFAULT_RUNNER_WORKSPACE_READ_LIMIT_BYTES,
        default_list_limit: int = DEFAULT_RUNNER_WORKSPACE_LIST_LIMIT,
    ) -> None:
        if not isinstance(runner, Runner):
            raise TypeError("RunnerWorkspace runner must be a Runner.")
        self.runner = runner
        self.cwd = _validate_optional_cwd(cwd)
        self.python_executable = require_clean_nonblank(python_executable, "python_executable")
        self.default_read_limit_bytes = _validate_required_limit(
            default_read_limit_bytes,
            "default_read_limit_bytes",
        )
        self.default_list_limit = _validate_required_limit(default_list_limit, "default_list_limit")
        if workspace_id is None:
            self.id = f"runner:{getattr(runner, 'isolation', 'unknown')}:{self.cwd or '.'}"
        else:
            self.id = require_clean_nonblank(workspace_id, "workspace_id")

    async def read_bytes(
        self,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        path = _validate_relative_path(path)
        limit = (
            self.default_read_limit_bytes
            if max_bytes is None
            else _validate_required_limit(max_bytes, "max_bytes")
        )
        result = await self._run_json_script(
            _READ_SCRIPT,
            path,
            str(limit),
            output_limit_bytes=_json_read_output_limit(limit),
        )
        content = _decode_base64(result["content_base64"], "content_base64")
        total_bytes = result["total_bytes"]
        if type(total_bytes) is not int:
            raise TypeError("Runner workspace read returned invalid total_bytes.")
        return WorkspaceReadResult(
            content=content,
            total_bytes=max(total_bytes, len(content)),
            truncated=total_bytes > len(content),
        )

    async def write_bytes(self, path: str, content: bytes) -> None:
        path = _validate_relative_path(path)
        if type(content) is not bytes:
            raise TypeError("Workspace write content must be bytes.")
        payload = {
            "path": path,
            "content_base64": base64.b64encode(content).decode("ascii"),
        }
        await self._run_json_script(
            _WRITE_SCRIPT,
            stdin=json.dumps(payload),
            output_limit_bytes=RUNNER_WORKSPACE_SCRIPT_OUTPUT_OVERHEAD_BYTES,
        )

    async def delete(self, path: str) -> None:
        path = _validate_relative_path(path)
        await self._run_json_script(
            _DELETE_SCRIPT,
            path,
            output_limit_bytes=RUNNER_WORKSPACE_SCRIPT_OUTPUT_OVERHEAD_BYTES,
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
        result = await self._run_json_script(
            _LIST_SCRIPT,
            translate_list_pattern(pattern),
            str(effective_limit),
            output_limit_bytes=_json_list_output_limit(effective_limit),
        )
        paths = result["paths"]
        total_count = result["total_count"]
        if not isinstance(paths, list):
            raise TypeError("Runner workspace list returned invalid paths.")
        if type(total_count) is not int:
            raise TypeError("Runner workspace list returned invalid total_count.")
        return WorkspaceListResult(
            paths=tuple(paths),
            total_count=total_count,
            truncated=total_count > len(paths),
        )

    async def read_tar_bytes(
        self,
        paths: Sequence[str],
        *,
        max_file_bytes: int | None = None,
    ) -> bytes:
        """Read many workspace files in one runner exec as an uncompressed tar.

        This is the bulk-transfer fast path used by SyncBinding: one guest
        process archives every requested file instead of one exec per file.
        Each file is capped at ``max_file_bytes`` (``default_read_limit_bytes``
        when omitted); an oversized file fails the whole transfer.
        """

        validated_paths = _validate_tar_paths(paths)
        per_file_limit = (
            self.default_read_limit_bytes
            if max_file_bytes is None
            else _validate_required_limit(max_file_bytes, "max_file_bytes")
        )
        payload = {"paths": list(validated_paths), "max_file_bytes": per_file_limit}
        result = await self._run_json_script(
            _READ_TAR_SCRIPT,
            stdin=json.dumps(payload),
            output_limit_bytes=_json_read_output_limit(
                _tar_size_bound(per_file_limit, len(validated_paths))
            ),
        )
        return _decode_base64(result["tar_base64"], "tar_base64")

    async def write_tar_bytes(self, data: bytes) -> None:
        """Write many workspace files in one runner exec from an uncompressed tar.

        Members must be regular files with workspace-relative paths; symlink,
        absolute, and ``..`` members are rejected inside the guest before any
        file is written through them.
        """

        if type(data) is not bytes:
            raise TypeError("Workspace tar content must be bytes.")
        payload = {"tar_base64": base64.b64encode(data).decode("ascii")}
        await self._run_json_script(
            _WRITE_TAR_SCRIPT,
            stdin=json.dumps(payload),
            output_limit_bytes=RUNNER_WORKSPACE_SCRIPT_OUTPUT_OVERHEAD_BYTES,
        )

    async def _run_json_script(
        self,
        script: str,
        *args: str,
        stdin: str | None = None,
        output_limit_bytes: int,
    ) -> dict[str, Any]:
        exec_result = await self.runner.exec(
            ExecCommand.process(self.python_executable, "-c", script, *args),
            cwd=self.cwd,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )
        if exec_result.stdout_truncated:
            raise RuntimeError("Runner workspace operation output exceeded its transfer limit.")
        try:
            payload = _parse_json_object(exec_result.stdout)
        except RuntimeError:
            if exec_result.exit_code != 0:
                raise RuntimeError(
                    f"Runner workspace operation failed with exit code {exec_result.exit_code}: "
                    f"{exec_result.stderr.strip() or exec_result.stdout.strip()}"
                ) from None
            raise
        if payload.get("ok") is not True:
            _raise_workspace_error(payload)
        if exec_result.exit_code != 0:
            raise RuntimeError(
                f"Runner workspace operation failed with exit code {exec_result.exit_code}: "
                f"{exec_result.stderr.strip() or exec_result.stdout.strip()}"
            )
        return payload


def _validate_optional_cwd(cwd: str | None) -> str | None:
    if cwd is None:
        return None
    value = require_nonblank(cwd, "cwd")
    if posixpath.isabs(value):
        raise ValueError("RunnerWorkspace cwd must be relative to the runner root.")
    normalized = posixpath.normpath(value)
    if normalized == ".":
        return None
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError("RunnerWorkspace cwd escapes the runner root.")
    return normalized


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
        raise TypeError(f"RunnerWorkspace {field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"RunnerWorkspace {field_name} must be greater than zero.")
    return value


def _validate_tar_paths(paths: Sequence[str]) -> tuple[str, ...]:
    if isinstance(paths, str) or not isinstance(paths, Sequence):
        raise TypeError("RunnerWorkspace read_tar_bytes paths must be a sequence of strings.")
    if not paths:
        raise ValueError("RunnerWorkspace read_tar_bytes requires at least one path.")
    return tuple(_validate_relative_path(path) for path in paths)


def _tar_size_bound(per_file_limit: int, file_count: int) -> int:
    return (
        per_file_limit + RUNNER_WORKSPACE_TAR_MEMBER_OVERHEAD_BYTES
    ) * file_count + RUNNER_WORKSPACE_TAR_ARCHIVE_OVERHEAD_BYTES


def _json_read_output_limit(max_bytes: int) -> int:
    return int(max_bytes * 4 / 3) + RUNNER_WORKSPACE_SCRIPT_OUTPUT_OVERHEAD_BYTES


def _json_list_output_limit(limit: int) -> int:
    return (limit * 512) + RUNNER_WORKSPACE_SCRIPT_OUTPUT_OVERHEAD_BYTES


def _decode_base64(value: Any, field_name: str) -> bytes:
    if type(value) is not str:
        raise TypeError(f"Runner workspace returned invalid {field_name}.")
    return base64.b64decode(value.encode("ascii"), validate=True)


def _parse_json_object(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Runner workspace operation returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise TypeError("Runner workspace operation returned invalid JSON object.")
    return payload


def _raise_workspace_error(payload: dict[str, Any]) -> None:
    error_type = payload.get("error_type")
    message = payload.get("message")
    if type(message) is not str or not message:
        message = "Runner workspace operation failed."
    if error_type == "not_found":
        raise FileNotFoundError(message)
    if error_type in {"invalid_path", "invalid_pattern"}:
        raise ValueError(message)
    if error_type == "not_file":
        raise IsADirectoryError(message)
    raise RuntimeError(message)
