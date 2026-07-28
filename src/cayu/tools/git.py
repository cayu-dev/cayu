from __future__ import annotations

import json
import ntpath
import posixpath
from dataclasses import dataclass
from typing import Any, cast

from cayu._validation import require_nonblank, require_unicode_scalar_text
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.runners import ExecCommand, ExecResult, LocalRunner, Runner, RunnerUnavailableError
from cayu.tools._errors import structured_invalid_arguments, tool_argument_validation
from cayu.workspaces import LocalWorkspace, RunnerBoundWorkspace

DEFAULT_GIT_CHANGES_LIMIT = 50
MAX_GIT_CHANGES_LIMIT = 200
DEFAULT_GIT_CHANGES_RESULT_BYTES = 32 * 1024
MAX_GIT_CHANGES_RESULT_BYTES = 200 * 1024
GIT_STATUS_CAPTURE_BYTES = 1024 * 1024
GIT_CHANGES_TIMEOUT_SECONDS = 30
MAX_GIT_DIFF_OFFSET_BYTES = 4 * 1024 * 1024
_TRUNCATION_MARKER = "\n[git changes truncated]"
_GIT_ENV_REMOVE = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_SYSTEM",
    "GIT_DIR",
    "GIT_EXTERNAL_DIFF",
    "GIT_INDEX_FILE",
    "GIT_NAMESPACE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_WORK_TREE",
)


@dataclass(frozen=True)
class _StatusPage:
    entries: list[dict[str, Any]]
    capture_truncated: bool


class GitChangesTool(Tool):
    """Inspect bounded Git changes without exposing an unrestricted shell."""

    spec = ToolSpec(
        name="git_changes",
        parallel_safe=True,
        effect=ToolEffect.NONE,
        description=(
            "Inspect bounded Git status, numstat summaries, or diffs in the active runner "
            "workspace. Results are pageable and report explicit truncation metadata."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["status", "summary", "diff"],
                    "default": "status",
                },
                "scope": {
                    "type": "string",
                    "enum": ["all", "staged", "unstaged"],
                    "default": "all",
                },
                "paths": {
                    "type": "array",
                    "maxItems": 100,
                    "items": {"type": "string", "minLength": 1},
                },
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "diff_offset": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": MAX_GIT_DIFF_OFFSET_BYTES,
                    "default": 0,
                    "description": "Byte offset into diff output for deterministic continuation.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_GIT_CHANGES_LIMIT,
                    "default": DEFAULT_GIT_CHANGES_LIMIT,
                },
                "max_result_bytes": {
                    "type": "integer",
                    "minimum": 1024,
                    "maximum": MAX_GIT_CHANGES_RESULT_BYTES,
                    "default": DEFAULT_GIT_CHANGES_RESULT_BYTES,
                },
            },
        },
    )

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        runner = ctx.runner
        if runner is None:
            return ToolResult(
                content="No runner configured for this tool call.",
                structured={"error": "runner_unavailable"},
                is_error=True,
            )
        with tool_argument_validation():
            mode = _enum_arg(args, "mode", {"status", "summary", "diff"}, "status")
            scope = _enum_arg(args, "scope", {"all", "staged", "unstaged"}, "all")
            paths = _paths_arg(args)
            offset = _integer_arg(args, "offset", default=0, minimum=0)
            diff_offset = _integer_arg(
                args,
                "diff_offset",
                default=0,
                minimum=0,
                maximum=MAX_GIT_DIFF_OFFSET_BYTES,
            )
            if mode != "diff" and diff_offset:
                raise ValueError("Tool argument `diff_offset` is only valid in diff mode.")
            limit = _integer_arg(
                args,
                "limit",
                default=DEFAULT_GIT_CHANGES_LIMIT,
                minimum=1,
                maximum=MAX_GIT_CHANGES_LIMIT,
            )
            max_result_bytes = _integer_arg(
                args,
                "max_result_bytes",
                default=DEFAULT_GIT_CHANGES_RESULT_BYTES,
                minimum=1024,
                maximum=MAX_GIT_CHANGES_RESULT_BYTES,
            )

        cwd_or_error = _workspace_cwd(ctx)
        if isinstance(cwd_or_error, ToolResult):
            return cwd_or_error
        cwd = cwd_or_error
        root_result = await _exec_git(
            runner,
            ["rev-parse", "--show-toplevel"],
            cwd=cwd,
            output_limit_bytes=GIT_STATUS_CAPTURE_BYTES,
        )
        if isinstance(root_result, ToolResult):
            return root_result
        if root_result.exit_code != 0:
            return _git_failure(
                root_result,
                operation="repository discovery",
                max_result_bytes=max_result_bytes,
            )
        repository_root = root_result.stdout.strip()
        if not _same_runner_path(repository_root, cwd):
            return ToolResult(
                content="Git repository root is outside the active workspace.",
                structured={
                    "error": "git_repository_outside_workspace",
                    "workspace_cwd": cwd,
                },
                is_error=True,
            )
        filter_names_result = await _configured_filter_names(runner, cwd=cwd)
        if isinstance(filter_names_result, ToolResult):
            return filter_names_result
        filter_names = filter_names_result
        status_result = await _exec_git(
            runner,
            _status_argv(paths),
            cwd=cwd,
            filter_names=filter_names,
            output_limit_bytes=GIT_STATUS_CAPTURE_BYTES,
        )
        if isinstance(status_result, ToolResult):
            return status_result
        if status_result.exit_code != 0:
            return _git_failure(
                status_result,
                operation="status",
                max_result_bytes=max_result_bytes,
            )
        page = _parse_status(status_result.stdout, truncated=status_result.stdout_truncated)
        filtered = [entry for entry in page.entries if _in_scope(entry, scope)]
        if page.capture_truncated and offset >= len(filtered):
            return ToolResult(
                content=(
                    "Git status reached its capture limit before the requested offset. "
                    "Use narrower paths."
                ),
                structured={
                    "error": "git_status_capture_exhausted",
                    "offset": offset,
                    "captured_changes": len(filtered),
                },
                is_error=True,
            )
        selected = filtered[offset : offset + limit]
        more_by_page = offset + len(selected) < len(filtered)
        capture_truncated = page.capture_truncated
        next_offset = (
            offset + len(selected) if selected and (more_by_page or capture_truncated) else None
        )
        truncation_reasons = [
            reason
            for condition, reason in (
                (more_by_page, "limit"),
                (capture_truncated, "capture_limit"),
            )
            if condition
        ]
        selected, next_offset, truncation_reasons = _fit_entries(
            mode=mode,
            scope=scope,
            entries=selected,
            offset=offset,
            limit=limit,
            next_offset=next_offset,
            truncation_reasons=truncation_reasons,
            max_result_bytes=max_result_bytes,
            structured_extra=(
                {
                    "binary_omitted": False,
                    "diff_offset": diff_offset,
                    "next_diff_offset": MAX_GIT_DIFF_OFFSET_BYTES,
                }
                if mode == "diff"
                else None
            ),
        )

        if mode == "status":
            content = "\n".join(_format_status(entry) for entry in selected)
            if not content:
                content = "No matching Git changes."
            content, result_truncated = _bounded_text(content, max_result_bytes)
            if result_truncated:
                truncation_reasons.append("result_bytes")
            return ToolResult(
                content=content,
                structured=_base_structured(
                    mode=mode,
                    scope=scope,
                    entries=selected,
                    offset=offset,
                    limit=limit,
                    next_offset=next_offset,
                    truncation_reasons=truncation_reasons,
                ),
            )

        selected_paths = [entry["path"] for entry in selected]
        if mode == "summary":
            return await _summary_result(
                runner,
                cwd=cwd,
                filter_names=filter_names,
                scope=scope,
                entries=selected,
                paths=selected_paths,
                offset=offset,
                limit=limit,
                next_offset=next_offset,
                truncation_reasons=truncation_reasons,
                max_result_bytes=max_result_bytes,
            )
        return await _diff_result(
            runner,
            cwd=cwd,
            filter_names=filter_names,
            scope=scope,
            entries=selected,
            paths=selected_paths,
            offset=offset,
            limit=limit,
            next_offset=next_offset,
            truncation_reasons=truncation_reasons,
            max_result_bytes=max_result_bytes,
            diff_offset=diff_offset,
        )


async def _summary_result(
    runner: Any,
    *,
    cwd: str,
    filter_names: tuple[str, ...],
    scope: str,
    entries: list[dict[str, Any]],
    paths: list[str],
    offset: int,
    limit: int,
    next_offset: int | None,
    truncation_reasons: list[str],
    max_result_bytes: int,
) -> ToolResult:
    counts: dict[str, dict[str, Any]] = {}
    command_truncated = False
    for staged in _diff_variants(scope):
        result = await _exec_git(
            runner,
            _diff_argv(paths, staged=staged, numstat=True),
            cwd=cwd,
            filter_names=filter_names,
            output_limit_bytes=max_result_bytes,
        )
        if isinstance(result, ToolResult):
            return result
        if result.exit_code != 0:
            return _git_failure(
                result,
                operation="summary",
                max_result_bytes=max_result_bytes,
            )
        command_truncated = command_truncated or result.stdout_truncated
        for path, additions, deletions, count_kind in _parse_numstat(
            result.stdout,
            truncated=result.stdout_truncated,
        ):
            current = counts.setdefault(
                path,
                {"additions": 0, "deletions": 0, "count_kind": "text"},
            )
            current["additions"] = _sum_count(current["additions"], additions)
            current["deletions"] = _sum_count(current["deletions"], deletions)
            if count_kind != "text":
                current["count_kind"] = count_kind
    summaries = [
        {
            **entry,
            "additions": counts.get(entry["path"], {}).get("additions"),
            "deletions": counts.get(entry["path"], {}).get("deletions"),
            "count_kind": (
                "untracked"
                if entry["index"] == "?"
                else counts.get(entry["path"], {}).get("count_kind", "unknown")
            ),
        }
        for entry in entries
    ]
    summaries, next_offset, truncation_reasons = _fit_entries(
        mode="summary",
        scope=scope,
        entries=summaries,
        offset=offset,
        limit=limit,
        next_offset=next_offset,
        truncation_reasons=truncation_reasons,
        max_result_bytes=max_result_bytes,
    )
    content = "\n".join(_format_summary(item) for item in summaries)
    if not content:
        content = "No matching Git changes."
    content, bounded = _bounded_text(content, max_result_bytes)
    reasons = [*truncation_reasons]
    if command_truncated:
        reasons.append("capture_limit")
    if bounded:
        reasons.append("result_bytes")
    return ToolResult(
        content=content,
        structured=_base_structured(
            mode="summary",
            scope=scope,
            entries=summaries,
            offset=offset,
            limit=limit,
            next_offset=next_offset,
            truncation_reasons=reasons,
        ),
    )


async def _diff_result(
    runner: Any,
    *,
    cwd: str,
    filter_names: tuple[str, ...],
    scope: str,
    entries: list[dict[str, Any]],
    paths: list[str],
    offset: int,
    limit: int,
    next_offset: int | None,
    truncation_reasons: list[str],
    max_result_bytes: int,
    diff_offset: int,
) -> ToolResult:
    chunks: list[str] = []
    command_truncated = False
    tracked_paths = [entry["path"] for entry in entries if entry["index"] != "?"]
    if tracked_paths:
        for staged in _diff_variants(scope):
            result = await _exec_git(
                runner,
                _diff_argv(tracked_paths, staged=staged, numstat=False),
                cwd=cwd,
                filter_names=filter_names,
                output_limit_bytes=diff_offset + max_result_bytes + 1,
            )
            if isinstance(result, ToolResult):
                return result
            if result.exit_code != 0:
                return _git_failure(
                    result,
                    operation="diff",
                    max_result_bytes=max_result_bytes,
                )
            command_truncated = command_truncated or result.stdout_truncated
            if result.stdout:
                chunks.append(result.stdout if result.stdout_truncated else result.stdout.rstrip())
            if result.stdout_truncated:
                break
    untracked = [entry["path"] for entry in entries if entry["index"] == "?"]
    if untracked:
        chunks.append("Untracked content omitted:\n" + "\n".join(untracked))
    complete_content = "\n\n".join(chunks) or "No textual diff for the selected changes."
    binary_omitted = _diff_contains_binary(complete_content)
    if binary_omitted:
        complete_content = "Binary diff omitted for the selected changes."
    content, bounded, next_diff_offset = _page_diff_text(
        complete_content,
        offset=diff_offset,
        maximum=max_result_bytes,
        capture_truncated=command_truncated,
    )
    reasons = [*truncation_reasons]
    if command_truncated:
        reasons.append("capture_limit")
    if bounded:
        reasons.append("result_bytes")
    structured = _base_structured(
        mode="diff",
        scope=scope,
        entries=entries,
        offset=offset,
        limit=limit,
        next_offset=next_offset,
        truncation_reasons=reasons,
    )
    structured.update(
        {
            "diff_offset": diff_offset,
            "next_diff_offset": next_diff_offset,
            "binary_omitted": binary_omitted,
        }
    )
    return ToolResult(
        content=content,
        structured=structured,
    )


async def _exec_git(
    runner: Any,
    argv: list[str],
    *,
    cwd: str,
    filter_names: tuple[str, ...] = (),
    output_limit_bytes: int,
) -> ExecResult | ToolResult:
    try:
        result = await runner.exec(
            ExecCommand.process(*_safe_git_argv(argv, filter_names=filter_names)),
            cwd=cwd,
            env={
                "GIT_ATTR_NOSYSTEM": "1",
                "GIT_CEILING_DIRECTORIES": cwd,
                "GIT_CONFIG_GLOBAL": "NUL" if ntpath.splitdrive(cwd)[0] else "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_DISCOVERY_ACROSS_FILESYSTEM": "0",
                "GIT_OPTIONAL_LOCKS": "0",
                "LC_ALL": "C",
            },
            env_remove=_GIT_ENV_REMOVE,
            timeout_s=GIT_CHANGES_TIMEOUT_SECONDS,
            output_limit_bytes=output_limit_bytes,
        )
    except RunnerUnavailableError as exc:
        del exc
        return ToolResult(
            content="Runner is unavailable for Git inspection.",
            structured={"error": "runner_unavailable"},
            is_error=True,
        )
    if not isinstance(result, ExecResult):
        raise TypeError("Runner exec must return ExecResult.")
    return result


def _workspace_cwd(ctx: ToolContext) -> str | ToolResult:
    runner = ctx.runner
    if runner is None:
        raise AssertionError("runner checked before resolving Git workspace")
    resolve_cwd = getattr(runner, "resolve_cwd", None)
    if not callable(resolve_cwd):
        return ToolResult(
            content="Git inspection requires a runner with bounded cwd resolution.",
            structured={"error": "runner_unavailable"},
            is_error=True,
        )
    workspace = ctx.workspace
    if workspace is None:
        return resolve_cwd(None)
    if isinstance(workspace, RunnerBoundWorkspace):
        if not workspace.is_bound_to_runner(cast("Runner", runner)):
            return ToolResult(
                content="Git inspection requires a workspace bound to the active runner.",
                structured={"error": "workspace_runner_mismatch"},
                is_error=True,
            )
        return workspace.runner_cwd
    if isinstance(workspace, LocalWorkspace) and isinstance(runner, LocalRunner):
        runner_root = runner.resolve_cwd(None)
        workspace_root = str(workspace.root)
        if _runner_path_is_within(workspace_root, runner_root):
            return workspace_root
    return ToolResult(
        content="Git inspection cannot prove the active workspace directory.",
        structured={"error": "workspace_runner_mismatch"},
        is_error=True,
    )


async def _configured_filter_names(
    runner: Any,
    *,
    cwd: str,
) -> tuple[str, ...] | ToolResult:
    result = await _exec_git(
        runner,
        [
            "config",
            "--local",
            "--null",
            "--name-only",
            "--get-regexp",
            r"^filter\..*\.(clean|smudge|process)$",
        ],
        cwd=cwd,
        output_limit_bytes=64 * 1024,
    )
    if isinstance(result, ToolResult):
        return result
    if result.exit_code not in {0, 1}:
        return _git_failure(
            result,
            operation="filter configuration inspection",
            max_result_bytes=DEFAULT_GIT_CHANGES_RESULT_BYTES,
        )
    if result.stdout_truncated:
        return ToolResult(
            content="Git filter configuration exceeds the safe inspection limit.",
            structured={"error": "git_filter_config_capture_exhausted"},
            is_error=True,
        )
    names: set[str] = set()
    for key in result.stdout.split("\0"):
        if not key.startswith("filter."):
            continue
        driver, separator, field = key.removeprefix("filter.").rpartition(".")
        if separator and driver and field in {"clean", "smudge", "process"}:
            names.add(driver)
    return tuple(sorted(names))


def _safe_git_argv(argv: list[str], *, filter_names: tuple[str, ...]) -> list[str]:
    command = ["git", "--no-pager", "-c", "core.fsmonitor=false"]
    for driver in filter_names:
        command.extend(
            (
                "-c",
                f"filter.{driver}.clean=",
                "-c",
                f"filter.{driver}.smudge=",
                "-c",
                f"filter.{driver}.process=",
                "-c",
                f"filter.{driver}.required=false",
            )
        )
    command.extend(argv)
    return command


def _same_runner_path(left: str, right: str) -> bool:
    return _normalized_runner_path(left) == _normalized_runner_path(right)


def _runner_path_is_within(path: str, root: str) -> bool:
    normalized_path = _normalized_runner_path(path)
    normalized_root = _normalized_runner_path(root)
    module = ntpath if ntpath.splitdrive(normalized_root)[0] else posixpath
    try:
        return module.commonpath((normalized_path, normalized_root)) == normalized_root
    except ValueError:
        return False


def _normalized_runner_path(path: str) -> str:
    module = ntpath if ntpath.splitdrive(path)[0] else posixpath
    normalized = module.normpath(path)
    return module.normcase(normalized) if module is ntpath else normalized


def _status_argv(paths: tuple[str, ...]) -> list[str]:
    return [
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=normal",
        "--",
        *paths,
    ]


def _diff_argv(paths: list[str], *, staged: bool, numstat: bool) -> list[str]:
    argv = [
        "diff",
        "--no-color",
        "--no-ext-diff",
        "--no-textconv",
    ]
    if staged:
        argv.append("--cached")
    if numstat:
        argv.extend(("--numstat", "-z"))
    else:
        argv.append("--unified=3")
    return [*argv, "--", *paths]


def _diff_variants(scope: str) -> tuple[bool, ...]:
    if scope == "staged":
        return (True,)
    if scope == "unstaged":
        return (False,)
    return (True, False)


def _parse_status(stdout: str, *, truncated: bool) -> _StatusPage:
    records = stdout.split("\0")
    if records and (records[-1] == "" or truncated):
        records.pop()
    entries: list[dict[str, Any]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if len(record) < 4 or record[2] != " ":
            continue
        x, y, path = record[0], record[1], record[3:]
        original_path = None
        if (x in "RC" or y in "RC") and index < len(records):
            original_path = records[index]
            index += 1
        entries.append(
            {
                "path": path,
                "original_path": original_path,
                "index": x,
                "worktree": y,
            }
        )
    entries.sort(key=lambda entry: entry["path"])
    return _StatusPage(entries, truncated)


def _parse_numstat(
    stdout: str,
    *,
    truncated: bool,
) -> list[tuple[str, int | None, int | None, str]]:
    values: list[tuple[str, int | None, int | None, str]] = []
    records = stdout.split("\0")
    if records and (records[-1] == "" or truncated):
        records.pop()
    index = 0
    while index < len(records):
        parts = records[index].split("\t", 2)
        index += 1
        if len(parts) != 3:
            continue
        path = parts[2]
        if not path:
            if index + 1 >= len(records):
                continue
            index += 1
            path = records[index]
            index += 1
        additions = int(parts[0]) if parts[0].isdigit() else None
        deletions = int(parts[1]) if parts[1].isdigit() else None
        count_kind = (
            "text"
            if additions is not None and deletions is not None
            else "binary"
            if parts[0] == "-" and parts[1] == "-"
            else "unknown"
        )
        values.append((path, additions, deletions, count_kind))
    return values


def _in_scope(entry: dict[str, Any], scope: str) -> bool:
    if scope == "staged":
        return entry["index"] not in {" ", "?"}
    if scope == "unstaged":
        return entry["worktree"] != " " or entry["index"] == "?"
    return True


def _format_status(entry: dict[str, Any]) -> str:
    rename = f" <- {entry['original_path']}" if entry["original_path"] is not None else ""
    return f"{entry['index']}{entry['worktree']} {entry['path']}{rename}"


def _format_summary(item: dict[str, Any]) -> str:
    kind = item["count_kind"]
    counts = f"+{item['additions']} -{item['deletions']}" if kind == "text" else kind
    return f"{item['path']}: {counts} ({item['index']}{item['worktree']})"


def _base_structured(
    *,
    mode: str,
    scope: str,
    entries: list[dict[str, Any]],
    offset: int,
    limit: int,
    next_offset: int | None,
    truncation_reasons: list[str],
) -> dict[str, Any]:
    reasons = list(dict.fromkeys(truncation_reasons))
    return {
        "mode": mode,
        "scope": scope,
        "changes": entries,
        "returned": len(entries),
        "offset": offset,
        "limit": limit,
        "truncated": bool(reasons),
        "truncation_reasons": reasons,
        "next_offset": next_offset,
    }


def _fit_entries(
    *,
    mode: str,
    scope: str,
    entries: list[dict[str, Any]],
    offset: int,
    limit: int,
    next_offset: int | None,
    truncation_reasons: list[str],
    max_result_bytes: int,
    structured_extra: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int | None, list[str]]:
    fitted = list(entries)
    reasons = list(truncation_reasons)
    while fitted:
        candidate = _base_structured(
            mode=mode,
            scope=scope,
            entries=fitted,
            offset=offset,
            limit=limit,
            next_offset=next_offset,
            truncation_reasons=[*reasons, "structured_result_bytes"],
        )
        if structured_extra:
            candidate.update(structured_extra)
        if len(json.dumps(candidate, separators=(",", ":")).encode()) <= max_result_bytes:
            break
        fitted.pop()
    if len(fitted) != len(entries):
        reasons.append("structured_result_bytes")
        next_offset = offset + len(fitted) if fitted else offset + 1
    return fitted, next_offset, reasons


def _bounded_text(content: str, maximum: int) -> tuple[str, bool]:
    encoded = content.encode("utf-8")
    if len(encoded) <= maximum:
        return content, False
    marker = _TRUNCATION_MARKER.encode()
    prefix = encoded[: maximum - len(marker)]
    return prefix.decode("utf-8", errors="ignore").rstrip() + _TRUNCATION_MARKER, True


def _page_diff_text(
    content: str,
    *,
    offset: int,
    maximum: int,
    capture_truncated: bool,
) -> tuple[str, bool, int | None]:
    encoded = content.encode("utf-8")
    if offset < len(encoded) and encoded[offset] & 0xC0 == 0x80:
        raise ValueError(
            "Tool argument `diff_offset` splits a UTF-8 character; use the previous "
            "`next_diff_offset` value."
        )
    if offset > len(encoded) and not capture_truncated:
        return "No textual diff at the requested diff offset.", False, None
    remaining = encoded[offset:]
    has_more = capture_truncated or len(remaining) > maximum
    if not has_more:
        return remaining.decode("utf-8"), False, None
    marker = _TRUNCATION_MARKER.encode()
    payload_limit = max(0, maximum - len(marker))
    prefix = remaining[:payload_limit]
    page = prefix.decode("utf-8", errors="ignore")
    consumed = len(page.encode("utf-8"))
    return page.rstrip() + _TRUNCATION_MARKER, True, offset + consumed


def _diff_contains_binary(content: str) -> bool:
    return "\ufffd" in content or any(
        ord(char) < 32 and char not in {"\t", "\n", "\r"} for char in content
    )


def _git_failure(
    result: ExecResult,
    *,
    operation: str,
    max_result_bytes: int,
) -> ToolResult:
    detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.exit_code}"
    content, _ = _bounded_text(f"Git {operation} failed: {detail}", max_result_bytes)
    return ToolResult(
        content=content,
        structured={
            "error": "git_command_failed",
            "operation": operation,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "cancelled": result.cancelled,
        },
        is_error=True,
    )


def _enum_arg(args: dict[str, Any], key: str, choices: set[str], default: str) -> str:
    value = args.get(key, default)
    if type(value) is not str or value not in choices:
        raise ValueError(f"Tool argument `{key}` must be one of: {', '.join(sorted(choices))}.")
    return value


def _integer_arg(
    args: dict[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = args.get(key, default)
    if type(value) is not int:
        raise ValueError(f"Tool argument `{key}` must be an integer.")
    if value < minimum:
        raise ValueError(f"Tool argument `{key}` must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"Tool argument `{key}` must be at most {maximum}.")
    return value


def _paths_arg(args: dict[str, Any]) -> tuple[str, ...]:
    value = args.get("paths", [])
    if not isinstance(value, list):
        raise ValueError("Tool argument `paths` must be an array.")
    if len(value) > 100:
        raise ValueError("Tool argument `paths` must contain at most 100 items.")
    return tuple(_validate_path(path) for path in value)


def _validate_path(value: object) -> str:
    if type(value) is not str:
        raise ValueError("Tool argument `paths` entries must be strings.")
    value = require_unicode_scalar_text(require_nonblank(value, "paths"), "paths")
    if (
        "\0" in value
        or posixpath.isabs(value)
        or ntpath.isabs(value)
        or ntpath.splitdrive(value)[0]
    ):
        raise ValueError("Git change paths must be workspace-relative.")
    normalized = posixpath.normpath(value.replace("\\", "/"))
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError("Git change paths must stay inside the workspace.")
    return normalized


def _sum_count(current: int | None, value: int | None) -> int | None:
    return current + value if current is not None and value is not None else None


def _display_count(value: int | None) -> str:
    return str(value) if value is not None else "binary"
