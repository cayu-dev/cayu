from __future__ import annotations

from cayu._validation import require_nonblank
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workspaces import Workspace

DEFAULT_READ_LIMIT_BYTES = 256 * 1024
MAX_READ_LIMIT_BYTES = 4 * 1024 * 1024
DEFAULT_WRITE_LIMIT_BYTES = 256 * 1024
MAX_WRITE_LIMIT_BYTES = 4 * 1024 * 1024
DEFAULT_LIST_LIMIT = 500
MAX_LIST_LIMIT = 10_000


class ReadFileTool(Tool):
    spec = ToolSpec(
        name="read_file",
        description="Read a UTF-8 text file from the active workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_READ_LIMIT_BYTES,
                    "default": DEFAULT_READ_LIMIT_BYTES,
                },
            },
            "required": ["path"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        workspace = _require_workspace(ctx)
        if workspace is None:
            return _missing_workspace_result()
        path = _require_arg_string(args, "path")
        max_bytes = _optional_int(
            args,
            "max_bytes",
            default=DEFAULT_READ_LIMIT_BYTES,
            maximum=MAX_READ_LIMIT_BYTES,
        )
        result = await workspace.read_bytes(path, max_bytes=max_bytes)
        text = result.content.decode("utf-8", errors="replace")
        return ToolResult(
            content=f"{text}\n\n[file truncated]" if result.truncated else text,
            structured={
                "path": path,
                "bytes": len(result.content),
                "total_bytes": result.total_bytes,
                "encoding": "utf-8",
                "truncated": result.truncated,
            },
        )


class WriteFileTool(Tool):
    spec = ToolSpec(
        name="write_file",
        description="Write UTF-8 text to a file in the active workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_WRITE_LIMIT_BYTES,
                    "default": DEFAULT_WRITE_LIMIT_BYTES,
                },
            },
            "required": ["path", "content"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        workspace = _require_workspace(ctx)
        if workspace is None:
            return _missing_workspace_result()
        path = _require_arg_string(args, "path")
        content = _require_arg_string(args, "content", allow_blank=True)
        max_bytes = _optional_int(
            args,
            "max_bytes",
            default=DEFAULT_WRITE_LIMIT_BYTES,
            maximum=MAX_WRITE_LIMIT_BYTES,
        )
        encoded = content.encode("utf-8")
        if len(encoded) > max_bytes:
            return ToolResult(
                content=(
                    f"Write refused: content is {len(encoded)} bytes, "
                    f"which exceeds max_bytes={max_bytes}."
                ),
                structured={
                    "path": path,
                    "bytes": len(encoded),
                    "max_bytes": max_bytes,
                    "encoding": "utf-8",
                },
                is_error=True,
            )
        await workspace.write_bytes(path, encoded)
        return ToolResult(
            content=f"Wrote {len(encoded)} bytes to {path}.",
            structured={
                "path": path,
                "bytes": len(encoded),
                "encoding": "utf-8",
            },
        )


class ListFilesTool(Tool):
    spec = ToolSpec(
        name="list_files",
        description="List files in the active workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "default": "**/*",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_LIST_LIMIT,
                    "default": DEFAULT_LIST_LIMIT,
                },
            },
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        workspace = _require_workspace(ctx)
        if workspace is None:
            return _missing_workspace_result()
        pattern = args.get("pattern", "**/*")
        if type(pattern) is not str:
            raise ValueError("Tool argument `pattern` must be a string.")
        pattern = require_nonblank(pattern, "pattern")
        limit = _optional_int(
            args,
            "limit",
            default=DEFAULT_LIST_LIMIT,
            maximum=MAX_LIST_LIMIT,
        )
        result = await workspace.list(pattern, limit=limit)
        result_content = "\n".join(result.paths) if result.paths else "No files matched."
        if result.truncated:
            result_content = f"{result_content}\n\n[file list truncated]"
        return ToolResult(
            content=result_content,
            structured={
                "pattern": pattern,
                "files": list(result.paths),
                "total_files": result.total_count,
                "truncated": result.truncated,
            },
        )


def _require_workspace(ctx: ToolContext) -> Workspace | None:
    if ctx.workspace is None:
        return None
    if not isinstance(ctx.workspace, Workspace):
        raise TypeError("Tool context workspace must implement Workspace.")
    return ctx.workspace


def _missing_workspace_result() -> ToolResult:
    return ToolResult(
        content="No workspace configured for this tool call.",
        is_error=True,
    )


def _require_arg_string(
    args: dict,
    key: str,
    *,
    allow_blank: bool = False,
) -> str:
    value = args.get(key)
    if type(value) is not str:
        raise ValueError(f"Tool argument `{key}` must be a string.")
    if allow_blank:
        return value
    return require_nonblank(value, key)


def _optional_int(
    args: dict,
    key: str,
    *,
    default: int,
    maximum: int,
) -> int:
    value = args.get(key, default)
    if type(value) is not int:
        raise ValueError(f"Tool argument `{key}` must be an integer.")
    if value <= 0:
        raise ValueError(f"Tool argument `{key}` must be greater than zero.")
    if value > maximum:
        raise ValueError(f"Tool argument `{key}` must be at most {maximum}.")
    return value
