from __future__ import annotations

import json
import math
import ntpath
import posixpath
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Any

from cayu._validation import (
    json_utf8_size_within_limit,
    require_nonblank,
    require_unicode_scalar_text,
)
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.runners import ExecCommand, ExecResult, RunnerUnavailableError
from cayu.tools._errors import (
    reject_unknown_tool_arguments,
    structured_invalid_arguments,
    tool_argument_validation,
)
from cayu.vaults import SecretRedactor

DEFAULT_SEARCH_LIMIT = 100
MAX_SEARCH_LIMIT = 500
DEFAULT_SEARCH_PREVIEW_LIMIT_BYTES = 500
MAX_SEARCH_PREVIEW_LIMIT_BYTES = 4 * 1024
DEFAULT_SEARCH_RESULT_LIMIT_BYTES = 20_000
MAX_SEARCH_RESULT_LIMIT_BYTES = 128 * 1024
MAX_SEARCH_ARGUMENT_BYTES = 4 * 1024
MAX_SEARCH_OFFSET = 10_000_000
DEFAULT_SEARCH_TIMEOUT_SECONDS = 30
MAX_SEARCH_TIMEOUT_SECONDS = 120
DEFAULT_SEARCH_CAPTURE_LIMIT_BYTES = 1024 * 1024
MAX_SEARCH_CAPTURE_LIMIT_BYTES = 4 * 1024 * 1024
DEFAULT_SEARCH_MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024
MAX_SEARCH_MAX_FILE_SIZE_BYTES = 64 * 1024 * 1024
DEFAULT_SEARCH_EXCLUDE_DIRECTORIES = (
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".next",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "vendor",
    "venv",
)
_MATCH_SEPARATOR = "\x1f"
_MATCH_SEPARATOR_ARG = r"\x1f"
_PREVIEW_TRUNCATION_MARKER = " [match preview truncated]"
_SEARCH_TEXT_ARGUMENTS = frozenset(
    {"pattern", "path", "glob", "mode", "limit", "offset", "case_sensitive"}
)


class _SearchEntryTooLargeError(ValueError):
    pass


@dataclass(frozen=True)
class _ParsedSearchMatch:
    value: dict[str, Any]
    line_truncated: bool = False


@dataclass(frozen=True)
class _ParsedSearchPage:
    matches: list[_ParsedSearchMatch]
    captured_entries: int


@dataclass(frozen=True)
class _BoundedSearchPage:
    matches: list[dict[str, Any]]
    content: str
    truncation_reasons: tuple[str, ...]
    next_offset: int | None
    projected_matches_bytes: int

    @property
    def truncated(self) -> bool:
        return bool(self.truncation_reasons)


def _search_text_tool_spec(*, default_limit: int, max_limit: int) -> ToolSpec:
    return ToolSpec(
        name="search_text",
        effect=ToolEffect.NONE,
        description=(
            "Search text in the active runner workspace with bounded, pageable results. "
            "Use mode='files' to locate matching files before requesting content."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "pattern": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_SEARCH_ARGUMENT_BYTES,
                    "pattern": r"\S",
                },
                "path": {
                    "type": "string",
                    "maxLength": MAX_SEARCH_ARGUMENT_BYTES,
                    "default": ".",
                },
                "glob": {"type": "string", "maxLength": MAX_SEARCH_ARGUMENT_BYTES},
                "mode": {
                    "type": "string",
                    "enum": ["files", "content", "count"],
                    "default": "files",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_limit,
                    "default": default_limit,
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": MAX_SEARCH_OFFSET,
                    "default": 0,
                },
                "case_sensitive": {"type": "boolean", "default": True},
            },
            "required": ["pattern"],
        },
    )


class SearchTextTool(Tool):
    """Bounded runner-backed workspace text search."""

    spec = _search_text_tool_spec(
        default_limit=DEFAULT_SEARCH_LIMIT,
        max_limit=MAX_SEARCH_LIMIT,
    )

    def _execution_profile_material(self) -> dict[str, Any]:
        """Return the bounded configuration that governs search behavior."""

        return {
            "default_limit": self.default_limit,
            "max_limit": self.max_limit,
            "max_preview_bytes": self.max_preview_bytes,
            "max_result_bytes": self.max_result_bytes,
            "timeout_s": self.timeout_s,
            "capture_limit_bytes": self.capture_limit_bytes,
            "max_file_size_bytes": self.max_file_size_bytes,
            "exclude_directories": list(self.exclude_directories),
        }

    def __init__(
        self,
        *,
        default_limit: int = DEFAULT_SEARCH_LIMIT,
        max_limit: int = MAX_SEARCH_LIMIT,
        max_preview_bytes: int = DEFAULT_SEARCH_PREVIEW_LIMIT_BYTES,
        max_result_bytes: int = DEFAULT_SEARCH_RESULT_LIMIT_BYTES,
        timeout_s: int = DEFAULT_SEARCH_TIMEOUT_SECONDS,
        capture_limit_bytes: int = DEFAULT_SEARCH_CAPTURE_LIMIT_BYTES,
        max_file_size_bytes: int = DEFAULT_SEARCH_MAX_FILE_SIZE_BYTES,
        exclude_directories: Iterable[str] = DEFAULT_SEARCH_EXCLUDE_DIRECTORIES,
        spec: ToolSpec | None = None,
    ) -> None:
        self.default_limit = _configuration_int(
            default_limit,
            "default_limit",
            maximum=MAX_SEARCH_LIMIT,
        )
        self.max_limit = _configuration_int(
            max_limit,
            "max_limit",
            maximum=MAX_SEARCH_LIMIT,
        )
        if self.default_limit > self.max_limit:
            raise ValueError("default_limit must be less than or equal to max_limit.")
        self.max_preview_bytes = _configuration_int(
            max_preview_bytes,
            "max_preview_bytes",
            minimum=len(_PREVIEW_TRUNCATION_MARKER.encode("utf-8")) + 1,
            maximum=MAX_SEARCH_PREVIEW_LIMIT_BYTES,
        )
        self.max_result_bytes = _configuration_int(
            max_result_bytes,
            "max_result_bytes",
            minimum=128,
            maximum=MAX_SEARCH_RESULT_LIMIT_BYTES,
        )
        if self.max_result_bytes < self.max_preview_bytes + 64:
            raise ValueError("max_result_bytes must be at least max_preview_bytes + 64.")
        self.timeout_s = _configuration_int(
            timeout_s,
            "timeout_s",
            maximum=MAX_SEARCH_TIMEOUT_SECONDS,
        )
        self.capture_limit_bytes = _configuration_int(
            capture_limit_bytes,
            "capture_limit_bytes",
            minimum=1024,
            maximum=MAX_SEARCH_CAPTURE_LIMIT_BYTES,
        )
        self.max_file_size_bytes = _configuration_int(
            max_file_size_bytes,
            "max_file_size_bytes",
            maximum=MAX_SEARCH_MAX_FILE_SIZE_BYTES,
        )
        self.exclude_directories = _validate_exclude_directories(exclude_directories)
        super().__init__(
            spec
            or _search_text_tool_spec(
                default_limit=self.default_limit,
                max_limit=self.max_limit,
            )
        )

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        with tool_argument_validation():
            reject_unknown_tool_arguments(args, allowed=_SEARCH_TEXT_ARGUMENTS)
        runner = ctx.runner
        if runner is None:
            return ToolResult(
                content="No runner configured for this tool call.",
                structured={"error": "runner_unavailable"},
                is_error=True,
            )
        with tool_argument_validation():
            pattern = _required_text(args, "pattern")
            path = _validate_search_path(_optional_text(args, "path", default="."))
            glob = _optional_nullable_text(args, "glob")
            if glob is not None:
                glob = _validate_search_glob(glob)
            mode = _optional_text(args, "mode", default="files")
            if mode not in {"files", "content", "count"}:
                raise ValueError("Tool argument `mode` must be one of: files, content, count.")
            limit = _optional_int(args, "limit", default=self.default_limit, minimum=1)
            if limit > self.max_limit:
                raise ValueError(f"Tool argument `limit` must be at most {self.max_limit}.")
            offset = _optional_int(args, "offset", default=0, minimum=0)
            if offset > MAX_SEARCH_OFFSET:
                raise ValueError(f"Tool argument `offset` must be at most {MAX_SEARCH_OFFSET}.")
            case_sensitive = _optional_bool(args, "case_sensitive", default=True)

        excluded_path = _matching_excluded_directory(
            path,
            self.exclude_directories,
        )
        if excluded_path is not None:
            return ToolResult(
                content=_bounded_message(
                    f"Search path is inside the excluded `{excluded_path}` directory. "
                    "Narrow the search elsewhere or change the tool registration.",
                    self.max_result_bytes,
                ),
                structured={
                    "error": "search_path_excluded",
                    "path": path,
                    "excluded_directory": excluded_path,
                },
                is_error=True,
            )

        scope_globs = _search_scope_globs(path, glob)
        visible_paths: set[str] | None = None
        started_at = time.monotonic()
        deadline = started_at + self.timeout_s
        if scope_globs:
            visible_result = await _run_search_command(
                ctx,
                runner,
                _file_listing_argv(
                    exclude_directories=self.exclude_directories,
                    max_file_size_bytes=self.max_file_size_bytes,
                ),
                timeout_s=self.timeout_s,
                output_limit_bytes=self.capture_limit_bytes,
                max_result_bytes=self.max_result_bytes,
                started_at=started_at,
                deadline=deadline,
                use_full_timeout=True,
            )
            if isinstance(visible_result, ToolResult):
                return visible_result
            visible_failure = _preflight_search_failure(
                visible_result,
                max_result_bytes=self.max_result_bytes,
                timeout_s=self.timeout_s,
                duration_ms=_elapsed_ms(started_at),
            )
            if visible_failure is not None:
                return visible_failure
            if visible_result.stdout_truncated:
                return _search_scope_exhausted_result(
                    phase="visibility",
                    result=visible_result,
                    duration_ms=_elapsed_ms(started_at),
                    max_result_bytes=self.max_result_bytes,
                )
            visible_paths = set(_iter_file_paths(visible_result.stdout))

        argv = [
            "rg",
            "--no-config",
            "--hidden",
            "--no-require-git",
            "--sort",
            "path",
            "--color",
            "never",
            "--max-filesize",
            str(self.max_file_size_bytes),
        ]
        if mode == "files":
            argv.extend(["--files-with-matches", "--null"])
        elif mode == "content":
            argv.extend(
                [
                    "--with-filename",
                    "--line-number",
                    "--null",
                    "--field-match-separator",
                    _MATCH_SEPARATOR_ARG,
                    "--max-columns",
                    str(self.max_preview_bytes),
                    "--max-columns-preview",
                ]
            )
        else:
            argv.extend(["--with-filename", "--count-matches", "--null"])
        if not case_sensitive:
            argv.append("--ignore-case")
        for scope_glob in scope_globs:
            argv.extend(["--glob", scope_glob])
        for directory in self.exclude_directories:
            argv.extend(["--glob", f"!{directory}/**"])
            argv.extend(["--glob", f"!**/{directory}/**"])
        argv.extend(["--", pattern, "."])
        raw = await _run_search_command(
            ctx,
            runner,
            argv,
            timeout_s=self.timeout_s,
            output_limit_bytes=self.capture_limit_bytes,
            max_result_bytes=self.max_result_bytes,
            started_at=started_at,
            deadline=deadline,
            use_full_timeout=not scope_globs,
        )
        if isinstance(raw, ToolResult):
            return raw
        duration_ms = _elapsed_ms(started_at)
        result = _require_exec_result(raw)
        artifacts, artifacts_truncated = _bounded_runner_artifacts(
            result.artifacts,
            max_bytes=self.max_result_bytes,
        )
        if result.timed_out or result.cancelled or result.exit_code not in {0, 1}:
            return _search_failure_result(
                result,
                max_result_bytes=self.max_result_bytes,
                timeout_s=self.timeout_s,
                duration_ms=duration_ms,
                artifacts=artifacts,
                artifacts_truncated=artifacts_truncated,
            )

        parsed_page = _parse_search_page(
            mode,
            result.stdout,
            offset=offset,
            limit=limit,
            max_preview_bytes=self.max_preview_bytes,
            allowed_paths=visible_paths,
        )
        stdout_bytes = (
            result.stdout_bytes
            if result.stdout_bytes is not None
            else len(result.stdout.encode("utf-8"))
        )
        if result.stdout_truncated and offset >= parsed_page.captured_entries:
            return ToolResult(
                content=_bounded_message(
                    "Search output reached its capture limit before the requested offset. "
                    "Use a narrower path, glob, or pattern.",
                    self.max_result_bytes,
                ),
                structured={
                    "error": "search_capture_exhausted",
                    "mode": mode,
                    "offset": offset,
                    "captured_entries": parsed_page.captured_entries,
                    "stdout_bytes": stdout_bytes,
                    "duration_ms": duration_ms,
                    **({"artifacts_truncated": True} if artifacts_truncated else {}),
                },
                artifacts=artifacts,
                is_error=True,
            )
        try:
            bounded_page = _bounded_page(
                mode,
                parsed_page,
                offset=offset,
                limit=limit,
                max_result_bytes=self.max_result_bytes,
                runner_truncated=result.stdout_truncated,
            )
        except _SearchEntryTooLargeError:
            return ToolResult(
                content=_bounded_message(
                    "The next search entry exceeds the configured result byte limit. "
                    "Use a narrower path or glob, or raise the registration-time limit.",
                    self.max_result_bytes,
                ),
                structured={
                    "error": "search_entry_too_large",
                    "mode": mode,
                    "offset": offset,
                    "limit": limit,
                    "stdout_bytes": stdout_bytes,
                    "duration_ms": duration_ms,
                    **({"artifacts_truncated": True} if artifacts_truncated else {}),
                },
                artifacts=artifacts,
                is_error=True,
            )
        return ToolResult(
            content=bounded_page.content,
            structured={
                "mode": mode,
                "pattern": pattern,
                "path": path,
                "glob": glob,
                "matches": bounded_page.matches,
                "returned": len(bounded_page.matches),
                "offset": offset,
                "limit": limit,
                "truncated": bounded_page.truncated,
                "truncation_reasons": list(bounded_page.truncation_reasons),
                "next_offset": bounded_page.next_offset,
                "stdout_bytes": stdout_bytes,
                "projected_content_bytes": len(bounded_page.content.encode("utf-8")),
                "projected_matches_bytes": bounded_page.projected_matches_bytes,
                "duration_ms": duration_ms,
                **({"artifacts_truncated": True} if artifacts_truncated else {}),
            },
            artifacts=artifacts,
        )


def _required_text(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if type(value) is not str:
        raise ValueError(f"Tool argument `{key}` must be a string.")
    return _validate_no_nul(
        require_unicode_scalar_text(require_nonblank(value, key), key),
        key,
    )


def _optional_text(args: dict[str, Any], key: str, *, default: str) -> str:
    value = args.get(key, default)
    if type(value) is not str:
        raise ValueError(f"Tool argument `{key}` must be a string.")
    return _validate_no_nul(
        require_unicode_scalar_text(require_nonblank(value, key), key),
        key,
    )


def _optional_nullable_text(args: dict[str, Any], key: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"Tool argument `{key}` must be a string.")
    return _validate_no_nul(
        require_unicode_scalar_text(require_nonblank(value, key), key),
        key,
    )


def _optional_int(
    args: dict[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
) -> int:
    value = args.get(key, default)
    if type(value) is not int:
        raise ValueError(f"Tool argument `{key}` must be an integer.")
    if value < minimum:
        raise ValueError(f"Tool argument `{key}` must be at least {minimum}.")
    return value


def _optional_bool(args: dict[str, Any], key: str, *, default: bool) -> bool:
    value = args.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"Tool argument `{key}` must be a boolean.")
    return value


def _require_exec_result(value: object) -> ExecResult:
    if type(value) is not ExecResult:
        raise TypeError("Runner.exec must return an ExecResult.")
    return value


async def _run_search_command(
    ctx: ToolContext,
    runner: Any,
    argv: list[str],
    *,
    timeout_s: int,
    output_limit_bytes: int,
    max_result_bytes: int,
    started_at: float,
    deadline: float,
    use_full_timeout: bool,
) -> ExecResult | ToolResult:
    remaining_s = deadline - time.monotonic()
    command_timeout_s = timeout_s if use_full_timeout else math.floor(remaining_s)
    if command_timeout_s <= 0:
        return _search_failure_result(
            ExecResult(exit_code=-1, timed_out=True),
            max_result_bytes=max_result_bytes,
            timeout_s=timeout_s,
            duration_ms=_elapsed_ms(started_at),
            artifacts=[],
            artifacts_truncated=False,
        )
    try:
        raw = await runner.exec(
            ExecCommand.process(*argv),
            timeout_s=command_timeout_s,
            output_limit_bytes=output_limit_bytes,
        )
    except RunnerUnavailableError as exc:
        diagnostic, diagnostic_truncated = _bounded_runner_diagnostic(
            exc.diagnostic,
            max_bytes=max_result_bytes,
        )
        return ToolResult(
            content="Runner is unavailable for text search.",
            structured={
                "error": "runner_unavailable",
                "diagnostic": diagnostic,
                "duration_ms": _elapsed_ms(started_at),
                **({"diagnostic_truncated": True} if diagnostic_truncated else {}),
            },
            artifacts=[diagnostic],
            is_error=True,
        )
    return _require_exec_result(raw)


def _file_listing_argv(
    *,
    exclude_directories: tuple[str, ...],
    max_file_size_bytes: int,
) -> list[str]:
    argv = [
        "rg",
        "--no-config",
        "--hidden",
        "--no-require-git",
        "--sort",
        "path",
        "--files",
        "--null",
        "--max-filesize",
        str(max_file_size_bytes),
    ]
    for directory in exclude_directories:
        argv.extend(["--glob", f"!{directory}/**"])
        argv.extend(["--glob", f"!**/{directory}/**"])
    argv.extend(["--", "."])
    return argv


def _preflight_search_failure(
    result: ExecResult,
    *,
    max_result_bytes: int,
    timeout_s: int,
    duration_ms: int,
) -> ToolResult | None:
    if not (result.timed_out or result.cancelled or result.exit_code not in {0, 1}):
        return None
    artifacts, artifacts_truncated = _bounded_runner_artifacts(
        result.artifacts,
        max_bytes=max_result_bytes,
    )
    return _search_failure_result(
        result,
        max_result_bytes=max_result_bytes,
        timeout_s=timeout_s,
        duration_ms=duration_ms,
        artifacts=artifacts,
        artifacts_truncated=artifacts_truncated,
    )


def _search_scope_exhausted_result(
    *,
    phase: str,
    result: ExecResult,
    duration_ms: int,
    max_result_bytes: int,
) -> ToolResult:
    artifacts, artifacts_truncated = _bounded_runner_artifacts(
        result.artifacts,
        max_bytes=max_result_bytes,
    )
    return ToolResult(
        content=_bounded_message(
            "Search scope contains too many paths for bounded resolution. "
            "Use a narrower path or glob.",
            max_result_bytes,
        ),
        structured={
            "error": "search_scope_exhausted",
            "phase": phase,
            "stdout_bytes": (
                result.stdout_bytes
                if result.stdout_bytes is not None
                else len(result.stdout.encode("utf-8"))
            ),
            "duration_ms": duration_ms,
            **({"artifacts_truncated": True} if artifacts_truncated else {}),
        },
        artifacts=artifacts,
        is_error=True,
    )


def _iter_file_paths(output: str) -> Iterable[str]:
    cursor = 0
    while cursor < len(output):
        path_end = output.find("\0", cursor)
        if path_end < 0:
            break
        if path_end > cursor:
            yield _workspace_relative_path(output[cursor:path_end])
        cursor = path_end + 1


def _parse_search_page(
    mode: str,
    output: str,
    *,
    offset: int,
    limit: int,
    max_preview_bytes: int,
    allowed_paths: set[str] | None = None,
) -> _ParsedSearchPage:
    if mode == "files":
        parsed_matches = _iter_file_matches(output)
    elif mode == "content":
        parsed_matches = _iter_content_matches(
            output,
            max_preview_bytes=max_preview_bytes,
        )
    else:
        parsed_matches = _iter_count_matches(output)
    page: list[_ParsedSearchMatch] = []
    captured_entries = 0
    for match in parsed_matches:
        if allowed_paths is not None and match.value["path"] not in allowed_paths:
            continue
        if captured_entries >= offset and len(page) < limit:
            page.append(match)
        captured_entries += 1
        if captured_entries > offset + limit:
            break
    return _ParsedSearchPage(matches=page, captured_entries=captured_entries)


def _iter_content_matches(
    output: str,
    *,
    max_preview_bytes: int,
) -> Iterable[_ParsedSearchMatch]:
    cursor = 0
    while cursor < len(output):
        path_end = output.find("\0", cursor)
        if path_end < 0:
            break
        record_end = output.find("\n", path_end + 1)
        if record_end < 0:
            break
        record = output[path_end + 1 : record_end]
        line_text, separator, preview = record.partition(_MATCH_SEPARATOR)
        if not separator:
            cursor = record_end + 1
            continue
        try:
            line = int(line_text)
        except ValueError:
            cursor = record_end + 1
            continue
        preview, line_truncated = _truncate_utf8(
            preview,
            max_preview_bytes,
            marker=_PREVIEW_TRUNCATION_MARKER,
        )
        yield _ParsedSearchMatch(
            value={
                "path": _workspace_relative_path(output[cursor:path_end]),
                "line": line,
                "preview": preview,
            },
            line_truncated=line_truncated,
        )
        cursor = record_end + 1


def _iter_file_matches(output: str) -> Iterable[_ParsedSearchMatch]:
    for path in _iter_file_paths(output):
        yield _ParsedSearchMatch(value={"path": path})


def _iter_count_matches(output: str) -> Iterable[_ParsedSearchMatch]:
    cursor = 0
    while cursor < len(output):
        path_end = output.find("\0", cursor)
        if path_end < 0:
            break
        record_end = output.find("\n", path_end + 1)
        if record_end < 0:
            break
        try:
            count = int(output[path_end + 1 : record_end])
        except ValueError:
            cursor = record_end + 1
            continue
        yield _ParsedSearchMatch(
            value={"path": _workspace_relative_path(output[cursor:path_end]), "count": count}
        )
        cursor = record_end + 1


def _render_matches(mode: str, matches: list[dict[str, Any]]) -> str:
    if mode == "files":
        return "\n".join(match["path"] for match in matches)
    if mode == "content":
        return "\n".join(f"{match['path']}:{match['line']}:{match['preview']}" for match in matches)
    return "\n".join(f"{match['path']}:{match['count']}" for match in matches)


def _bounded_page(
    mode: str,
    parsed_page: _ParsedSearchPage,
    *,
    offset: int,
    limit: int,
    max_result_bytes: int,
    runner_truncated: bool,
) -> _BoundedSearchPage:
    candidates = parsed_page.matches
    for returned in range(len(candidates), -1, -1):
        if candidates and returned == 0:
            continue
        parsed_matches = candidates[:returned]
        page = [match.value for match in parsed_matches]
        output_truncated = returned < len(candidates)
        has_more = parsed_page.captured_entries > offset + returned
        line_truncated = any(match.line_truncated for match in parsed_matches)
        reasons: list[str] = ["line"] if line_truncated else []
        if (
            not output_truncated
            and returned == limit
            and parsed_page.captured_entries > offset + returned
        ):
            reasons.append("limit")
        if output_truncated:
            reasons.append("output")
        if runner_truncated:
            reasons.append("runner_output")
        next_offset = offset + returned if has_more and page else None
        content = _render_matches(mode, page) if page else "No matches."
        if reasons:
            content += "\n\n" + _render_search_metadata(
                returned=returned,
                reasons=reasons,
                next_offset=next_offset,
            )
        projected_matches_bytes = len(
            json.dumps(page, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        if (
            len(content.encode("utf-8")) <= max_result_bytes
            and projected_matches_bytes <= max_result_bytes
        ):
            return _BoundedSearchPage(
                matches=page,
                content=content,
                truncation_reasons=tuple(reasons),
                next_offset=next_offset,
                projected_matches_bytes=projected_matches_bytes,
            )
    raise _SearchEntryTooLargeError


def _render_search_metadata(
    *,
    returned: int,
    reasons: list[str],
    next_offset: int | None,
) -> str:
    rendered_next_offset = "none" if next_offset is None else str(next_offset)
    return (
        f"[search metadata: returned={returned}; truncated=true; "
        f"reasons={','.join(reasons)}; next_offset={rendered_next_offset}]"
    )


def _truncate_utf8(value: str, maximum: int, *, marker: str) -> tuple[str, bool]:
    return SecretRedactor().redact_text_bounded_with_marker(
        value,
        max_bytes=maximum,
        truncation_marker=marker,
    )


def _bounded_message(value: str, maximum: int) -> str:
    content, _ = _truncate_utf8(
        value,
        maximum,
        marker="\n[search message truncated]",
    )
    return content


def _configuration_int(
    value: object,
    field_name: str,
    *,
    minimum: int = 1,
    maximum: int,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    if value > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}.")
    return value


def _workspace_relative_path(path: str) -> str:
    path = path.encode("utf-8", errors="replace").decode("utf-8")
    while path.startswith("./"):
        path = path[2:]
    return path


def _validate_search_path(path: str) -> str:
    windows_path = PureWindowsPath(path)
    if (
        posixpath.isabs(path)
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
    ):
        raise ValueError("Search path must be relative to the runner workspace.")
    normalized = posixpath.normpath(path)
    windows_normalized = ntpath.normpath(path)
    if (
        normalized == ".."
        or normalized.startswith("../")
        or windows_normalized == ".."
        or windows_normalized.startswith("..\\")
    ):
        raise ValueError("Search path escapes the runner workspace.")
    return normalized


def _validate_search_glob(value: str) -> str:
    if value.startswith("!"):
        raise ValueError("Tool argument `glob` must be an inclusion glob.")
    windows_path = PureWindowsPath(value)
    if posixpath.isabs(value) or windows_path.is_absolute() or windows_path.drive:
        raise ValueError("Tool argument `glob` must be workspace-relative.")
    if any(part == ".." for part in value.replace("\\", "/").split("/")):
        raise ValueError("Tool argument `glob` must not contain parent traversal.")
    _validate_balanced_glob_operators(value)
    _validate_glob_character_ranges(value)
    return value


def _search_scope_globs(path: str, glob: str | None) -> tuple[str, ...]:
    if path == ".":
        return () if glob is None else (glob,)
    escaped_path = _escape_glob_literal(path.rstrip("/"))
    if glob is None:
        return (escaped_path, f"{escaped_path}/**")
    relative_glob = glob[3:] if glob.startswith("**/") else glob
    return (f"{escaped_path}/**/{relative_glob}",)


def _escape_glob_literal(value: str) -> str:
    return "".join(
        f"\\{character}" if character in "\\*?[]{{}}!" else character for character in value
    )


def _matching_excluded_directory(
    path: str,
    excluded_directories: tuple[str, ...],
) -> str | None:
    path_parts = tuple(part for part in path.strip("/").split("/") if part not in {"", "."})
    for directory in excluded_directories:
        directory_parts = tuple(directory.strip("/").split("/"))
        width = len(directory_parts)
        if any(
            path_parts[index : index + width] == directory_parts
            for index in range(len(path_parts) - width + 1)
        ):
            return directory
    return None


def _validate_balanced_glob_operators(value: str) -> None:
    stack: list[str] = []
    escaped = False
    pairs = {"]": "[", "}": "{"}
    for character in value:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
        elif character in "[{":
            stack.append(character)
        elif character in pairs and (not stack or stack.pop() != pairs[character]):
            raise ValueError("Tool argument `glob` has unbalanced glob operators.")
    if escaped or stack:
        raise ValueError("Tool argument `glob` has unbalanced glob operators.")


def _validate_glob_character_ranges(value: str) -> None:
    cursor = 0
    escaped_outside_class = False
    while cursor < len(value):
        if escaped_outside_class:
            escaped_outside_class = False
            cursor += 1
            continue
        if value[cursor] == "\\":
            escaped_outside_class = True
            cursor += 1
            continue
        if value[cursor] != "[":
            cursor += 1
            continue
        end = cursor + 1
        escaped = False
        tokens: list[tuple[str, bool]] = []
        while end < len(value) and (value[end] != "]" or escaped):
            character = value[end]
            if escaped:
                tokens.append((character, True))
                escaped = False
            elif character == "\\":
                escaped = True
            else:
                tokens.append((character, False))
            end += 1
        first = 1 if tokens and tokens[0][0] in {"!", "^"} else 0
        if len(tokens) == first:
            raise ValueError("Tool argument `glob` has an empty character class.")
        for index in range(first + 1, len(tokens) - 1):
            character, was_escaped = tokens[index]
            if character != "-" or was_escaped:
                continue
            range_start = tokens[index - 1][0]
            range_end = tokens[index + 1][0]
            if ord(range_start) > ord(range_end):
                raise ValueError("Tool argument `glob` has an invalid character range.")
        cursor = end + 1


def _validate_no_nul(value: str, field_name: str) -> str:
    if "\0" in value:
        raise ValueError(f"Tool argument `{field_name}` must not contain NUL characters.")
    if len(value.encode("utf-8")) > MAX_SEARCH_ARGUMENT_BYTES:
        raise ValueError(
            f"Tool argument `{field_name}` must be at most {MAX_SEARCH_ARGUMENT_BYTES} bytes."
        )
    return value


def _search_failure_result(
    result: ExecResult,
    *,
    max_result_bytes: int,
    timeout_s: int,
    duration_ms: int,
    artifacts: list[dict[str, Any]],
    artifacts_truncated: bool,
) -> ToolResult:
    if result.timed_out:
        error = "search_timed_out"
        message = f"Search timed out after {timeout_s} seconds."
    elif result.cancelled:
        error = "search_cancelled"
        message = "Search was cancelled."
    elif result.exit_code in {126, 127}:
        error = "search_unavailable"
        message = "Text search is unavailable because ripgrep could not be started."
    else:
        error = (
            "invalid_pattern"
            if result.exit_code == 2 and "regex parse error" in result.stderr.lower()
            else "search_failed"
        )
        message = (
            "Search pattern is invalid." if error == "invalid_pattern" else "Text search failed."
        )
    content = _bounded_message(message, max_result_bytes)
    return ToolResult(
        content=content,
        structured={
            "error": error,
            "exit_code": result.exit_code,
            "stderr_bytes": (
                result.stderr_bytes
                if result.stderr_bytes is not None
                else len(result.stderr.encode("utf-8"))
            ),
            "duration_ms": duration_ms,
            **({"artifacts_truncated": True} if artifacts_truncated else {}),
        },
        artifacts=artifacts,
        is_error=True,
    )


def _elapsed_ms(started_at: float) -> int:
    return max(0, int((time.monotonic() - started_at) * 1000))


def _bounded_runner_diagnostic(
    diagnostic: dict[str, Any],
    *,
    max_bytes: int,
) -> tuple[dict[str, Any], bool]:
    sanitized = _omit_raw_stream_fields(diagnostic)
    if json_utf8_size_within_limit(sanitized, max_bytes):
        return sanitized, False
    return {"truncated": True}, True


def _bounded_runner_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    max_bytes: int,
) -> tuple[list[dict[str, Any]], bool]:
    sanitized = _omit_raw_stream_fields(artifacts)
    if json_utf8_size_within_limit(sanitized, max_bytes):
        return sanitized, False
    return [{"type": "cayu.search_runner_artifacts.v1", "truncated": True}], True


def _omit_raw_stream_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [_omit_raw_stream_fields(item) for item in value]
    if not isinstance(value, dict):
        return value
    sanitized: dict[str, Any] = {}
    omitted = False
    for key, item in value.items():
        if "stdout" in key.lower() or "stderr" in key.lower():
            omitted = True
            continue
        sanitized[key] = _omit_raw_stream_fields(item)
    if omitted:
        sanitized["raw_streams_omitted"] = True
    return sanitized


def _validate_exclude_directories(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str | bytes):
        raise TypeError("exclude_directories must be an iterable of strings.")
    try:
        copied = tuple(values)
    except TypeError as exc:
        raise TypeError("exclude_directories must be an iterable of strings.") from exc
    validated: list[str] = []
    for value in copied:
        if type(value) is not str:
            raise TypeError("exclude_directories entries must be strings.")
        value = _validate_no_nul(
            require_unicode_scalar_text(
                require_nonblank(value, "exclude directory"),
                "exclude directory",
            ),
            "exclude directory",
        )
        normalized = posixpath.normpath(value)
        if posixpath.isabs(value) or normalized in {".", ".."} or normalized.startswith("../"):
            raise ValueError("exclude_directories entries must be relative directory paths.")
        if any(character in value for character in "*?[]{}!"):
            raise ValueError("exclude_directories entries must not contain glob operators.")
        validated.append(value.rstrip("/"))
    return tuple(validated)
