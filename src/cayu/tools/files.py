from __future__ import annotations

import asyncio
import difflib
import hashlib
import io
import json
import mimetypes
import ntpath
import posixpath
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from itertools import pairwise
from pathlib import PurePosixPath, PureWindowsPath
from typing import Protocol

from cayu._validation import require_nonblank, require_unicode_scalar_text
from cayu.artifacts import (
    DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
    FILE_ATTACHMENT_IMAGE_CONTENT_TYPES,
    ArtifactMetadata,
    ArtifactReadResult,
    ArtifactScope,
    ArtifactStore,
    FileAttachmentKind,
    InvalidArtifactIdError,
    copy_artifact_read_result,
    file_attachment,
)
from cayu.artifacts._images import decode_verified_image_format
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools._errors import (
    invalid_tool_arguments_result,
    structured_invalid_arguments,
    tool_argument_validation,
)
from cayu.tools._redaction import (
    active_secret_redactor,
    active_secret_redactor_snapshot,
    await_revision_stable_secret_output,
    record_ambiguous_secret_output,
    unstable_secret_redaction_result,
)
from cayu.workspaces import (
    Workspace,
    WorkspaceReadResult,
    WorkspaceRevisionMismatchError,
    validate_list_pattern,
)

DEFAULT_READ_LIMIT_BYTES = 256 * 1024
MAX_READ_LIMIT_BYTES = 4 * 1024 * 1024
DEFAULT_ATTACHMENT_LIMIT_BYTES = DEFAULT_MAX_FILE_ATTACHMENT_BYTES
DEFAULT_MAX_ATTACHMENT_LIMIT_BYTES = DEFAULT_MAX_FILE_ATTACHMENT_BYTES
MAX_IMAGE_SOURCE_BYTES = 20 * 1024 * 1024
MAX_PDF_SOURCE_BYTES = 32 * 1024 * 1024
MAX_PDF_PAGES_PER_READ = 10
DEFAULT_WRITE_LIMIT_BYTES = 256 * 1024
MAX_WRITE_LIMIT_BYTES = 4 * 1024 * 1024
DEFAULT_EDIT_DIFF_LIMIT_BYTES = 32 * 1024
MAX_EDIT_DIFF_LIMIT_BYTES = 200 * 1024
MAX_EDIT_OPERATIONS = 100
MAX_FILE_REVISION_CHARS = 4096
DEFAULT_LIST_LIMIT = 500
MAX_LIST_LIMIT = 10_000

IMAGE_CONTENT_TYPES = FILE_ATTACHMENT_IMAGE_CONTENT_TYPES
PDF_CONTENT_TYPE = "application/pdf"
_PIL_IMAGE_FORMAT_CONTENT_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}
_CONTROL_TEXT_BYTES = {9, 10, 12, 13, 27}
_TEXT_CONTENT_TYPES = {
    "application/javascript",
    "application/json",
    "application/ld+json",
    "application/toml",
    "application/x-httpd-php",
    "application/x-ndjson",
    "application/x-sh",
    "application/x-yaml",
    "application/xml",
    "image/svg+xml",
}
_TEXT_CONTENT_TYPE_SUFFIXES = ("+json", "+xml", "+yaml")
_BINARY_CONTENT_TYPE_PREFIXES = ("image/",)
_BINARY_CONTENT_TYPES = {
    "application/gzip",
    "application/java-archive",
    "application/pdf",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/x-7z-compressed",
    "application/x-bzip2",
    "application/x-dosexec",
    "application/x-executable",
    "application/x-rar-compressed",
    "application/x-tar",
    "application/zip",
}


class _InvalidPdfPageRange(ValueError):
    """A model-supplied PDF page selection that cannot be satisfied."""


@dataclass(frozen=True)
class ReadFileOptions:
    max_bytes: int = DEFAULT_READ_LIMIT_BYTES
    max_attachment_bytes: int = DEFAULT_ATTACHMENT_LIMIT_BYTES
    pages: str | None = None


@dataclass(frozen=True)
class _TextEdit:
    old_text: str
    new_text: str
    expected_replacements: int


@dataclass(frozen=True)
class ArtifactReadRequest:
    ctx: ToolContext
    artifact_store: ArtifactStore
    artifact: ArtifactMetadata
    initial_read: ArtifactReadResult
    options: ReadFileOptions
    structured: dict


class ArtifactReader(Protocol):
    """Extension point for artifact-specific read behavior."""

    def can_read(self, artifact: ArtifactMetadata) -> bool:
        """Return true when this reader handles the artifact."""

    async def read(self, request: ArtifactReadRequest) -> ToolResult:
        """Read the artifact and return a model-facing tool result."""


def _read_file_tool_spec(
    *,
    default_attachment_limit_bytes: int,
    max_attachment_limit_bytes: int,
) -> ToolSpec:
    return ToolSpec(
        name="read_file",
        # Workspace image/PDF reads can create artifact snapshots; keep this conservative until
        # snapshot writes are idempotent/content-addressed.
        effect=ToolEffect.EXTERNAL,
        description=(
            "Read a file from the active workspace or read an artifact by id. "
            "Use `path` for workspace files and `artifact_id` for uploaded/generated artifacts. "
            "Workspace text results begin with model-visible JSON metadata. A complete offset-zero "
            "read includes the opaque revision required by edit_file, write_file overwrite mode, "
            "and delete_file; paged reads include next_offset. Workspace image/PDF files are "
            "captured as artifact snapshots when an artifact store is configured. Image and PDF "
            "artifacts return provider-neutral file attachments that capable providers can inspect "
            "natively."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path to read.",
                },
                "artifact_id": {
                    "type": "string",
                    "description": "Artifact id to read, such as an uploaded file reference.",
                },
                "max_bytes": {
                    "type": "integer",
                    "description": "Maximum text bytes to read for workspace and text artifact reads.",
                    "minimum": 1,
                    "maximum": MAX_READ_LIMIT_BYTES,
                    "default": DEFAULT_READ_LIMIT_BYTES,
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                    "description": "Byte offset for pageable workspace text reads.",
                },
                "max_attachment_bytes": {
                    "type": "integer",
                    "description": "Maximum bytes for a native image/PDF attachment sent to a provider.",
                    "minimum": 1,
                    "maximum": max_attachment_limit_bytes,
                    "default": default_attachment_limit_bytes,
                },
                "pages": {
                    "type": "string",
                    "description": (
                        "PDF page range, such as '1-5' or '3'. At most 10 pages per read."
                    ),
                },
            },
        },
    )


class ReadFileTool(Tool):
    spec = _read_file_tool_spec(
        default_attachment_limit_bytes=DEFAULT_ATTACHMENT_LIMIT_BYTES,
        max_attachment_limit_bytes=DEFAULT_MAX_ATTACHMENT_LIMIT_BYTES,
    )

    def __init__(
        self,
        *,
        extra_artifact_readers: Iterable[ArtifactReader] | None = None,
        artifact_readers: Iterable[ArtifactReader] | None = None,
        default_attachment_limit_bytes: int = DEFAULT_ATTACHMENT_LIMIT_BYTES,
        max_attachment_limit_bytes: int = DEFAULT_MAX_ATTACHMENT_LIMIT_BYTES,
        spec: ToolSpec | None = None,
    ) -> None:
        default_attachment_limit_bytes = _validate_positive_int(
            default_attachment_limit_bytes,
            "default_attachment_limit_bytes",
        )
        max_attachment_limit_bytes = _validate_positive_int(
            max_attachment_limit_bytes,
            "max_attachment_limit_bytes",
        )
        if default_attachment_limit_bytes > max_attachment_limit_bytes:
            raise ValueError(
                "default_attachment_limit_bytes must be less than or equal to "
                "max_attachment_limit_bytes."
            )
        if spec is None:
            spec = _read_file_tool_spec(
                default_attachment_limit_bytes=default_attachment_limit_bytes,
                max_attachment_limit_bytes=max_attachment_limit_bytes,
            )
        super().__init__(spec=spec)
        self.default_attachment_limit_bytes = default_attachment_limit_bytes
        self.max_attachment_limit_bytes = max_attachment_limit_bytes
        if artifact_readers is not None and extra_artifact_readers is not None:
            raise ValueError(
                "Use either artifact_readers for full replacement or "
                "extra_artifact_readers to extend defaults, not both."
            )
        if artifact_readers is not None:
            self.artifact_readers = _validate_artifact_readers(
                artifact_readers,
                field_name="artifact_readers",
            )
        else:
            self.artifact_readers = [
                *_validate_optional_artifact_readers(extra_artifact_readers),
                *default_artifact_readers(),
            ]

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        with tool_argument_validation():
            path = _optional_arg_string(args, "path")
            artifact_id = _optional_arg_string(args, "artifact_id")
            if (path is None) == (artifact_id is None):
                raise ValueError(
                    "Tool arguments must include exactly one of `path` or `artifact_id`."
                )
            max_bytes = _optional_int(
                args,
                "max_bytes",
                default=DEFAULT_READ_LIMIT_BYTES,
                maximum=MAX_READ_LIMIT_BYTES,
            )
            offset = _optional_nonnegative_int(args, "offset", default=0)
            max_attachment_bytes = _optional_int(
                args,
                "max_attachment_bytes",
                default=self.default_attachment_limit_bytes,
                maximum=self.max_attachment_limit_bytes,
            )
            pages = _optional_arg_string(args, "pages")
            if path is not None:
                path = _validate_workspace_path_argument(path)
        if path is not None:
            return await _read_workspace_file(
                ctx,
                path=path,
                artifact_readers=self.artifact_readers,
                max_bytes=max_bytes,
                offset=offset,
                max_attachment_bytes=max_attachment_bytes,
                pages=pages,
            )
        if artifact_id is not None:
            if offset != 0:
                return invalid_tool_arguments_result(
                    ValueError("Tool argument `offset` is only valid for workspace text files.")
                )
            return await _read_artifact(
                ctx,
                artifact_id=artifact_id,
                artifact_readers=self.artifact_readers,
                max_bytes=max_bytes,
                max_attachment_bytes=max_attachment_bytes,
                pages=pages,
            )
        raise AssertionError("unreachable")


async def _read_workspace_file(
    ctx: ToolContext,
    *,
    path: str,
    artifact_readers: list[ArtifactReader],
    max_bytes: int,
    offset: int,
    max_attachment_bytes: int,
    pages: str | None,
) -> ToolResult:
    workspace = _require_workspace(ctx)
    if workspace is None:
        return _missing_workspace_result()
    content_type = _guess_workspace_content_type(path)
    if _is_workspace_file_attachment_content_type(content_type):
        result = await workspace.read_bytes(path, offset=offset, max_bytes=max_bytes)
        if offset != 0:
            return invalid_tool_arguments_result(
                ValueError("Tool argument `offset` is only valid for workspace text files.")
            )
        return await _read_workspace_file_attachment(
            ctx,
            path=path,
            content_type=content_type,
            artifact_readers=artifact_readers,
            max_bytes=max_bytes,
            max_attachment_bytes=max_attachment_bytes,
            pages=pages,
            initial_result=result,
        )

    async def read_window(redactor):
        redaction_overlap = redactor.pagination_overlap_utf8_bytes
        fetch_offset = max(0, offset - redaction_overlap - 3)
        prefix_bytes = offset - fetch_offset
        result = await workspace.read_bytes(
            path,
            offset=fetch_offset,
            max_bytes=prefix_bytes + max_bytes + redaction_overlap + 4,
        )
        return result, redaction_overlap, fetch_offset

    captured = await await_revision_stable_secret_output(ctx, read_window)
    if captured is None:
        return unstable_secret_redaction_result()
    (result, redaction_overlap, fetch_offset), capture_snapshot = captured
    redactor = capture_snapshot.redactor
    if result.offset != fetch_offset:
        return _secret_redaction_window_unavailable_result(
            offset=offset,
            required_window_start=fetch_offset,
            observed_window_start=result.offset,
        )
    page_or_error = _utf8_workspace_page(
        result,
        path=path,
        content_type=content_type,
        requested_offset=offset,
        max_bytes=max_bytes,
    )
    if isinstance(page_or_error, ToolResult):
        return page_or_error
    page, next_offset = page_or_error
    if _is_binary_workspace_file(content_type=content_type, content=page):
        if offset != 0:
            return invalid_tool_arguments_result(
                ValueError("Tool argument `offset` is only valid for workspace text files.")
            )
        return _binary_workspace_file_result(
            path=path,
            content_type=content_type,
            bytes_read=len(page),
            total_bytes=result.total_bytes,
            truncated=next_offset is not None,
            inspectable=False,
        )
    if pages is not None:
        return invalid_tool_arguments_result(
            ValueError("Tool argument `pages` is only valid for PDF files.")
        )
    page_end = offset + len(page)
    required_window_end = min(
        result.total_bytes,
        page_end + redaction_overlap,
    )
    observed_window_end = result.offset + len(result.content)
    if observed_window_end < required_window_end:
        return _secret_redaction_window_unavailable_result(
            offset=offset,
            required_window_end=required_window_end,
            observed_window_end=observed_window_end,
        )
    if redactor.has_values:
        text, redaction_truncated = redactor.redact_utf8_page(
            result.content,
            window_offset=result.offset,
            page_offset=offset,
            page_end=page_end,
            max_bytes=max_bytes,
            source_complete=next_offset is None,
        )
    else:
        # With no registered values there is no redaction boundary to enforce.
        # In particular, a literal public marker is ordinary workspace content
        # and must remain byte-for-byte pageable instead of becoming an atomic
        # replacement token.
        text = page.decode("utf-8")
        redaction_truncated = False
    truncated = next_offset is not None or redaction_truncated
    complete = offset == 0 and not truncated
    if not complete:
        record_ambiguous_secret_output(ctx, capture_snapshot)
    structured = {
        "source": "workspace",
        "path": path,
        "bytes": len(page),
        "total_bytes": result.total_bytes,
        "offset": offset,
        "next_offset": next_offset,
        "revision": result.revision if complete else None,
        "sha256": result.sha256 if complete else None,
        "encoding": "utf-8",
        "truncated": truncated,
    }
    return ToolResult(
        content=_model_visible_workspace_text(text, structured),
        structured=structured,
    )


def _secret_redaction_window_unavailable_result(
    *,
    offset: int,
    required_window_start: int | None = None,
    observed_window_start: int | None = None,
    required_window_end: int | None = None,
    observed_window_end: int | None = None,
) -> ToolResult:
    structured = {
        "error": "secret_redaction_window_unavailable",
        "offset": offset,
    }
    if required_window_start is not None:
        structured["required_window_start"] = required_window_start
    if observed_window_start is not None:
        structured["observed_window_start"] = observed_window_start
    if required_window_end is not None:
        structured["required_window_end"] = required_window_end
    if observed_window_end is not None:
        structured["observed_window_end"] = observed_window_end
    return ToolResult(
        content=(
            "Workspace text could not be safely paginated with the active secret-redaction scope."
        ),
        structured=structured,
        is_error=True,
    )


def _model_visible_workspace_text(text: str, structured: dict) -> str:
    metadata = {
        key: structured[key]
        for key in (
            "path",
            "bytes",
            "total_bytes",
            "offset",
            "next_offset",
            "revision",
            "sha256",
            "truncated",
        )
    }
    encoded = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    return f"[read_file metadata]\n{encoded}\n[/read_file metadata]\n{text}"


def _utf8_workspace_page(
    result: WorkspaceReadResult,
    *,
    path: str,
    content_type: str,
    requested_offset: int,
    max_bytes: int,
) -> tuple[bytes, int | None] | ToolResult:
    start = requested_offset - result.offset
    available = result.content[start:]
    if available and available[0] & 0xC0 == 0x80:
        return invalid_tool_arguments_result(
            ValueError(
                "Tool argument `offset` splits a UTF-8 character; use the previous "
                "`next_offset` value."
            )
        )
    page = available[:max_bytes]
    while page:
        try:
            page.decode("utf-8")
            break
        except UnicodeDecodeError as exc:
            if exc.reason != "unexpected end of data":
                return _binary_workspace_file_result(
                    path=path,
                    content_type=content_type,
                    bytes_read=len(page),
                    total_bytes=result.total_bytes,
                    truncated=True,
                    inspectable=False,
                )
            page = page[: exc.start]
    if not page and requested_offset < result.total_bytes:
        return ToolResult(
            content=(
                "Text page is too small to contain the next complete UTF-8 character; "
                "retry with max_bytes of at least 4."
            ),
            structured={
                "error": "text_page_too_small",
                "offset": requested_offset,
                "minimum_max_bytes": 4,
            },
            is_error=True,
        )
    end = requested_offset + len(page)
    return page, end if end < result.total_bytes else None


def _guess_workspace_content_type(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


async def _read_workspace_file_attachment(
    ctx: ToolContext,
    *,
    path: str,
    content_type: str,
    artifact_readers: list[ArtifactReader],
    max_bytes: int,
    max_attachment_bytes: int,
    pages: str | None,
    initial_result: WorkspaceReadResult,
) -> ToolResult:
    artifact_store = _require_artifact_store(ctx)
    if artifact_store is None:
        return _binary_workspace_file_result(
            path=path,
            content_type=content_type,
            bytes_read=len(initial_result.content),
            total_bytes=initial_result.total_bytes,
            truncated=initial_result.truncated,
            inspectable=True,
            reason="Native inspection of this workspace file requires an artifact store.",
        )

    source_cap = _workspace_attachment_source_cap(content_type)
    try:
        result = await _read_full_workspace_attachment(
            ctx,
            path=path,
            content_type=content_type,
            initial_result=initial_result,
            max_source_bytes=source_cap,
        )
    except WorkspaceFileChangedError as exc:
        return _binary_workspace_file_result(
            path=path,
            content_type=content_type,
            bytes_read=len(initial_result.content),
            total_bytes=initial_result.total_bytes,
            truncated=initial_result.truncated,
            inspectable=True,
            reason=f"{exc} Retry read_file after the workspace file is stable.",
        )
    if result.truncated:
        return _binary_workspace_file_result(
            path=path,
            content_type=content_type,
            bytes_read=len(result.content),
            total_bytes=result.total_bytes,
            truncated=True,
            inspectable=True,
            reason=(f"Workspace file is too large to inspect natively (max {source_cap} bytes)."),
        )

    snapshot = await artifact_store.put_bytes(
        result.content,
        filename=_workspace_snapshot_filename(path),
        content_type=content_type,
        scope=ArtifactScope.SESSION,
        session_id=ctx.session_id,
        agent_name=ctx.agent_name,
        environment_name=ctx.environment_name,
        metadata={
            "source": "workspace",
            "source_workspace_id": ctx.workspace_id,
            "source_workspace_path": path,
            "operation": "read_file_workspace_snapshot",
        },
    )
    artifact_result = await _read_artifact(
        ctx,
        artifact_id=snapshot.id,
        artifact_readers=artifact_readers,
        max_bytes=max_bytes,
        max_attachment_bytes=max_attachment_bytes,
        pages=pages,
    )
    return _workspace_snapshot_result(
        path=path,
        content_type=content_type,
        workspace_result=result,
        snapshot=snapshot,
        artifact_result=artifact_result,
    )


async def _read_full_workspace_attachment(
    ctx: ToolContext,
    *,
    path: str,
    content_type: str,
    initial_result: WorkspaceReadResult,
    max_source_bytes: int,
) -> WorkspaceReadResult:
    workspace = _require_workspace(ctx)
    if workspace is None:
        raise RuntimeError("Workspace disappeared while reading workspace file attachment.")
    result = await workspace.read_bytes(path, max_bytes=max_source_bytes)
    if (
        not initial_result.truncated
        and not result.truncated
        and (
            result.total_bytes != initial_result.total_bytes
            or result.content != initial_result.content
        )
    ):
        raise WorkspaceFileChangedError(
            "Workspace file content changed while it was being captured as an artifact snapshot."
        )
    if result.content and not _is_binary_workspace_file(
        content_type=content_type,
        content=result.content,
    ):
        raise WorkspaceFileChangedError(
            "Workspace file content changed while it was being captured as an artifact snapshot."
        )
    return result


class WorkspaceFileChangedError(RuntimeError):
    pass


def _workspace_snapshot_filename(path: str) -> str:
    return PurePosixPath(path).name or "workspace-file"


def _is_binary_workspace_file(*, content_type: str, content: bytes) -> bool:
    if not content:
        return False
    if b"\x00" in content:
        return True
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError:
        return True
    if not decoded:
        return False
    control_count = sum(
        1 for char in decoded if ord(char) < 32 and ord(char) not in _CONTROL_TEXT_BYTES
    )
    if control_count / len(decoded) > 0.05:
        return True
    if _is_text_content_type(content_type):
        return False
    return _is_binary_content_type(content_type)


def _is_binary_content_type(content_type: str) -> bool:
    normalized = content_type.lower()
    if _is_text_content_type(normalized):
        return False
    return normalized in _BINARY_CONTENT_TYPES or normalized.startswith(
        _BINARY_CONTENT_TYPE_PREFIXES
    )


def _is_workspace_file_attachment_content_type(content_type: str) -> bool:
    return content_type == PDF_CONTENT_TYPE or content_type in IMAGE_CONTENT_TYPES


def _workspace_attachment_source_cap(content_type: str) -> int:
    if content_type == PDF_CONTENT_TYPE:
        return MAX_PDF_SOURCE_BYTES
    if content_type in IMAGE_CONTENT_TYPES:
        return MAX_IMAGE_SOURCE_BYTES
    return DEFAULT_ATTACHMENT_LIMIT_BYTES


def _binary_workspace_file_result(
    *,
    path: str,
    content_type: str,
    bytes_read: int,
    total_bytes: int,
    truncated: bool,
    inspectable: bool,
    reason: str | None = None,
) -> ToolResult:
    if reason is None:
        reason = "Use an artifact or custom parser/tool to inspect this file."
    return ToolResult(
        content=(
            f"Workspace file '{path}' appears to be binary "
            f"({content_type}, {total_bytes} bytes). "
            "read_file does not decode unsupported binary workspace files as text. "
            f"{reason}"
        ),
        structured={
            "source": "workspace",
            "path": path,
            "bytes": bytes_read,
            "total_bytes": total_bytes,
            "content_type": content_type,
            "encoding": None,
            "binary": True,
            "inspectable": inspectable,
            "truncated": truncated,
        },
        is_error=True,
    )


def _workspace_snapshot_result(
    *,
    path: str,
    content_type: str,
    workspace_result: WorkspaceReadResult,
    snapshot: ArtifactMetadata,
    artifact_result: ToolResult,
) -> ToolResult:
    structured = {
        **(artifact_result.structured or {}),
        "source": "workspace",
        "path": path,
        "workspace_id": snapshot.metadata.get("source_workspace_id"),
        "content_type": content_type,
        "bytes": len(workspace_result.content),
        "total_bytes": workspace_result.total_bytes,
        "binary": True,
        "inspectable": True,
        "truncated": workspace_result.truncated,
        "snapshot_artifact_id": snapshot.id,
        "snapshot_artifact_bytes": snapshot.size_bytes,
        "snapshot_artifact_scope": snapshot.scope.value,
    }
    return ToolResult(
        content=(
            f"Captured workspace file '{path}' as artifact snapshot {snapshot.id}. "
            f"{artifact_result.content}"
        ),
        structured=structured,
        artifacts=artifact_result.artifacts,
        is_error=artifact_result.is_error,
    )


async def _read_artifact(
    ctx: ToolContext,
    *,
    artifact_id: str,
    artifact_readers: list[ArtifactReader],
    max_bytes: int,
    max_attachment_bytes: int,
    pages: str | None,
) -> ToolResult:
    artifact_store = _require_artifact_store(ctx)
    if artifact_store is None:
        return _missing_artifact_store_result()
    try:
        result = await _read_artifact_store(
            artifact_store,
            artifact_id,
            max_bytes=max_bytes,
        )
    except InvalidArtifactIdError as exc:
        return invalid_tool_arguments_result(exc)
    artifact = result.metadata
    access_error = _artifact_access_error(ctx, artifact)
    if access_error is not None:
        return access_error
    structured = {
        "source": "artifact",
        "artifact_id": artifact.id,
        "filename": artifact.filename,
        "content_type": artifact.content_type,
        "bytes": len(result.content),
        "total_bytes": result.total_bytes,
        "size_bytes": artifact.size_bytes,
        "scope": artifact.scope.value,
        "session_id": artifact.session_id,
        "agent_name": artifact.agent_name,
        "environment_name": artifact.environment_name,
        "truncated": result.truncated,
    }
    request = ArtifactReadRequest(
        ctx=ctx,
        artifact_store=artifact_store,
        artifact=artifact,
        initial_read=result,
        options=ReadFileOptions(
            max_bytes=max_bytes,
            max_attachment_bytes=max_attachment_bytes,
            pages=pages,
        ),
        structured=structured,
    )
    for reader in artifact_readers:
        if reader.can_read(artifact):
            return await reader.read(request)
    return ToolResult(
        content=(
            f"Artifact {artifact.id} ({artifact.filename}) is {artifact.content_type} "
            f"and is {artifact.size_bytes} bytes. No built-in reader is available for this "
            "content type. Register a custom tool if this format should be inspectable."
        ),
        structured=structured,
        is_error=True,
    )


class TextArtifactReader:
    def can_read(self, artifact: ArtifactMetadata) -> bool:
        return _is_text_content_type(artifact.content_type)

    async def read(self, request: ArtifactReadRequest) -> ToolResult:
        if request.options.pages is not None:
            return invalid_tool_arguments_result(
                ValueError("Tool argument `pages` is only valid for PDF artifacts.")
            )
        text, redaction_truncated = active_secret_redactor(request.ctx).redact_utf8_head(
            request.initial_read.content,
            max_bytes=request.options.max_bytes,
            source_complete=not request.initial_read.truncated,
        )
        truncated = request.initial_read.truncated or redaction_truncated
        return ToolResult(
            content=f"{text}\n\n[file truncated]" if truncated else text,
            structured={
                **request.structured,
                "encoding": "utf-8",
                "truncated": truncated,
            },
        )


class ImageArtifactReader:
    def can_read(self, artifact: ArtifactMetadata) -> bool:
        return artifact.content_type in IMAGE_CONTENT_TYPES

    async def read(self, request: ArtifactReadRequest) -> ToolResult:
        if request.options.pages is not None:
            return invalid_tool_arguments_result(
                ValueError("Tool argument `pages` is only valid for PDF artifacts.")
            )
        artifact_store = request.artifact_store
        artifact = request.artifact
        if artifact.size_bytes == 0:
            return ToolResult(
                content=f"Image artifact '{artifact.filename}' is empty and cannot be inspected.",
                structured=request.structured,
                is_error=True,
            )
        result = await _read_artifact_store(
            artifact_store,
            artifact.id,
            max_bytes=request.options.max_attachment_bytes,
        )
        attachment_artifact = artifact
        if result.truncated:
            source = await _read_artifact_store(
                artifact_store,
                artifact.id,
                max_bytes=MAX_IMAGE_SOURCE_BYTES,
            )
            if source.truncated:
                return ToolResult(
                    content=(
                        f"Image '{artifact.filename}' is too large to inspect "
                        f"({artifact.size_bytes} bytes)."
                    ),
                    structured=request.structured,
                    is_error=True,
                )
            detected_content_type, validation_error = await asyncio.to_thread(
                _detect_image_content_type,
                source.content,
            )
            if validation_error is not None:
                return ToolResult(
                    content=f"Image '{artifact.filename}' could not be inspected: {validation_error}",
                    structured=request.structured,
                    is_error=True,
                )
            if detected_content_type is None:
                return ToolResult(
                    content=f"Image '{artifact.filename}' could not be inspected: unknown image type.",
                    structured=request.structured,
                    is_error=True,
                )
            if detected_content_type != artifact.content_type:
                return ToolResult(
                    content=(
                        f"Image '{artifact.filename}' content type mismatch: metadata says "
                        f"{artifact.content_type}, but bytes are {detected_content_type}."
                    ),
                    structured=request.structured,
                    is_error=True,
                )
            derivation_key = _derivation_key(
                source_hash=_content_hash(source.content),
                operation="resize_image",
                params=str(request.options.max_attachment_bytes),
            )
            reused = await _find_derived_artifact(artifact_store, request.ctx, derivation_key)
            if reused is not None:
                attachment_artifact = reused
            else:
                try:
                    resized = _resize_image_bytes(
                        source.content,
                        content_type=detected_content_type,
                        max_bytes=request.options.max_attachment_bytes,
                    )
                except Exception as exc:
                    return ToolResult(
                        content=f"Image '{artifact.filename}' could not be inspected: {exc}",
                        structured=request.structured,
                        is_error=True,
                    )
                if resized is None:
                    return ToolResult(
                        content=(
                            f"Image '{artifact.filename}' exceeds max_attachment_bytes="
                            f"{request.options.max_attachment_bytes}. Install cayu[files] or "
                            "register a custom reader to resize this image."
                        ),
                        structured=request.structured,
                        is_error=True,
                    )
                image_bytes, content_type = resized
                attachment_artifact = await artifact_store.put_bytes(
                    image_bytes,
                    filename=artifact.filename,
                    content_type=content_type,
                    scope=ArtifactScope.SESSION,
                    session_id=request.ctx.session_id,
                    agent_name=request.ctx.agent_name,
                    environment_name=request.ctx.environment_name,
                    metadata={
                        "derived_from_artifact_id": artifact.id,
                        "reader": type(self).__name__,
                        "operation": "resize_image",
                        "cayu_derivation_key": derivation_key,
                        "content_hash": _content_hash(image_bytes),
                    },
                )
        else:
            detected_content_type, validation_error = await asyncio.to_thread(
                _detect_image_content_type,
                result.content,
            )
            if validation_error is not None:
                return ToolResult(
                    content=f"Image '{artifact.filename}' could not be inspected: {validation_error}",
                    structured=request.structured,
                    is_error=True,
                )
            if detected_content_type is None:
                return ToolResult(
                    content=f"Image '{artifact.filename}' could not be inspected: unknown image type.",
                    structured=request.structured,
                    is_error=True,
                )
            if detected_content_type != artifact.content_type:
                return ToolResult(
                    content=(
                        f"Image '{artifact.filename}' content type mismatch: metadata says "
                        f"{artifact.content_type}, but bytes are {detected_content_type}."
                    ),
                    structured=request.structured,
                    is_error=True,
                )
        attachment = file_attachment(
            artifact_id=attachment_artifact.id,
            kind=FileAttachmentKind.IMAGE,
            filename=attachment_artifact.filename,
            content_type=attachment_artifact.content_type,
            size_bytes=attachment_artifact.size_bytes,
            metadata={"source_artifact_id": artifact.id},
        )
        return ToolResult(
            content=(
                f"Attached image artifact {attachment_artifact.id}: "
                f"{attachment_artifact.filename} ({attachment_artifact.content_type}, "
                f"{attachment_artifact.size_bytes} bytes)."
            ),
            structured={
                **request.structured,
                "attachment_artifact_id": attachment_artifact.id,
                "attachment_content_type": attachment_artifact.content_type,
                "attachment_bytes": attachment_artifact.size_bytes,
            },
            artifacts=[attachment],
        )


class PdfArtifactReader:
    def can_read(self, artifact: ArtifactMetadata) -> bool:
        return artifact.content_type == PDF_CONTENT_TYPE

    async def read(self, request: ArtifactReadRequest) -> ToolResult:
        artifact_store = request.artifact_store
        artifact = request.artifact
        if artifact.size_bytes == 0:
            return ToolResult(
                content=f"PDF artifact '{artifact.filename}' is empty and cannot be inspected.",
                structured=request.structured,
                is_error=True,
            )
        source_content: bytes | None = None
        if (
            request.options.pages is None
            and artifact.size_bytes <= request.options.max_attachment_bytes
        ):
            result = await _read_artifact_store(
                artifact_store,
                artifact.id,
                max_bytes=request.options.max_attachment_bytes,
            )
            validation_error = _validate_pdf_bytes(result.content)
            if validation_error is not None:
                return ToolResult(
                    content=f"PDF '{artifact.filename}' could not be inspected: {validation_error}",
                    structured=request.structured,
                    is_error=True,
                )
            total_pages = _count_pdf_pages(result.content)
            if total_pages is not None and total_pages > MAX_PDF_PAGES_PER_READ:
                # A short-but-many-page PDF fits under the byte cap yet still needs
                # the 10-page limit enforced; fall through to page extraction reusing
                # the bytes we already read (they are complete, not truncated).
                source_content = result.content
                attachment_artifact = None
            else:
                attachment_artifact = artifact
                page_note = ""
        else:
            attachment_artifact = None
        if attachment_artifact is None:
            if source_content is None:
                source = await _read_artifact_store(
                    artifact_store,
                    artifact.id,
                    max_bytes=MAX_PDF_SOURCE_BYTES,
                )
                if source.truncated:
                    return ToolResult(
                        content=(
                            f"PDF '{artifact.filename}' is too large to inspect "
                            f"({artifact.size_bytes} bytes)."
                        ),
                        structured=request.structured,
                        is_error=True,
                    )
                source_content = source.content
            derivation_key = _derivation_key(
                source_hash=_content_hash(source_content),
                operation="extract_pdf_pages",
                params=request.options.pages or "",
            )
            reused = await _find_derived_artifact(artifact_store, request.ctx, derivation_key)
            if reused is not None:
                attachment_artifact = reused
                page_note = reused.metadata.get("page_note", "")
            else:
                try:
                    extracted = _extract_pdf_pages(source_content, request.options.pages)
                except _InvalidPdfPageRange as exc:
                    return invalid_tool_arguments_result(exc)
                except Exception as exc:
                    return ToolResult(
                        content=f"PDF '{artifact.filename}' could not be inspected: {exc}",
                        structured=request.structured,
                        is_error=True,
                    )
                if extracted is None:
                    return ToolResult(
                        content=(
                            "PDF page extraction requires the optional files dependencies. "
                            "Install cayu[files] or register a custom PDF reader."
                        ),
                        structured=request.structured,
                        is_error=True,
                    )
                pdf_bytes, page_note = extracted
                if len(pdf_bytes) > request.options.max_attachment_bytes:
                    return ToolResult(
                        content=(
                            f"PDF selection for '{artifact.filename}' is {len(pdf_bytes)} bytes, "
                            f"which exceeds "
                            f"max_attachment_bytes={request.options.max_attachment_bytes}."
                        ),
                        structured=request.structured,
                        is_error=True,
                    )
                attachment_artifact = await artifact_store.put_bytes(
                    pdf_bytes,
                    filename=artifact.filename,
                    content_type=PDF_CONTENT_TYPE,
                    scope=ArtifactScope.SESSION,
                    session_id=request.ctx.session_id,
                    agent_name=request.ctx.agent_name,
                    environment_name=request.ctx.environment_name,
                    metadata={
                        "derived_from_artifact_id": artifact.id,
                        "reader": type(self).__name__,
                        "operation": "extract_pdf_pages",
                        "pages": request.options.pages,
                        "cayu_derivation_key": derivation_key,
                        "content_hash": _content_hash(pdf_bytes),
                        "page_note": page_note,
                    },
                )
        attachment = file_attachment(
            artifact_id=attachment_artifact.id,
            kind=FileAttachmentKind.DOCUMENT,
            filename=attachment_artifact.filename,
            content_type=attachment_artifact.content_type,
            size_bytes=attachment_artifact.size_bytes,
            metadata={"source_artifact_id": artifact.id, "pages": request.options.pages},
        )
        return ToolResult(
            content=(
                f"Attached PDF artifact {attachment_artifact.id}: {attachment_artifact.filename}"
                f"{page_note} ({attachment_artifact.size_bytes} bytes)."
            ),
            structured={
                **request.structured,
                "attachment_artifact_id": attachment_artifact.id,
                "attachment_content_type": attachment_artifact.content_type,
                "attachment_bytes": attachment_artifact.size_bytes,
                "pages": request.options.pages,
            },
            artifacts=[attachment],
        )


def default_artifact_readers() -> tuple[ArtifactReader, ...]:
    return (
        TextArtifactReader(),
        ImageArtifactReader(),
        PdfArtifactReader(),
    )


def _validate_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


def _validate_optional_artifact_readers(
    readers: Iterable[ArtifactReader] | None,
) -> list[ArtifactReader]:
    if readers is None:
        return []
    return _validate_artifact_readers(
        readers,
        field_name="extra_artifact_readers",
        allow_empty=True,
    )


def _validate_artifact_readers(
    readers: Iterable[ArtifactReader],
    *,
    field_name: str,
    allow_empty: bool = False,
) -> list[ArtifactReader]:
    if isinstance(readers, str | bytes):
        raise TypeError(f"{field_name} must be an iterable of ArtifactReader objects.")
    try:
        copied = list(readers)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of ArtifactReader objects.") from exc
    if not copied and not allow_empty:
        raise ValueError(f"{field_name} cannot be empty.")
    for reader in copied:
        if not callable(getattr(reader, "can_read", None)) or not callable(
            getattr(reader, "read", None)
        ):
            raise TypeError(f"{field_name} entries must implement can_read and read.")
    return copied


class EditFileTool(Tool):
    """Apply exact text replacements to one existing workspace file."""

    spec = ToolSpec(
        name="edit_file",
        parallel_safe=False,
        effect=ToolEffect.EXTERNAL,
        description=(
            "Atomically edit an existing UTF-8 workspace file using one or more exact "
            "text replacements. Requires the opaque revision returned by read_file and "
            "refuses stale content, ambiguous matches, partial edits, and oversized files."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path of the existing UTF-8 file.",
                },
                "expected_revision": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_FILE_REVISION_CHARS,
                    "description": "Opaque revision from a complete read_file result.",
                },
                "edits": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_EDIT_OPERATIONS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_text": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "new_text": {"type": "string"},
                            "expected_replacements": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 1000,
                                "default": 1,
                            },
                        },
                        "required": ["old_text", "new_text"],
                        "additionalProperties": False,
                    },
                },
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_WRITE_LIMIT_BYTES,
                    "default": DEFAULT_WRITE_LIMIT_BYTES,
                },
                "max_diff_bytes": {
                    "type": "integer",
                    "minimum": 256,
                    "maximum": MAX_EDIT_DIFF_LIMIT_BYTES,
                    "default": DEFAULT_EDIT_DIFF_LIMIT_BYTES,
                },
            },
            "required": ["path", "expected_revision", "edits"],
            "additionalProperties": False,
        },
    )

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        workspace = _require_workspace(ctx)
        if workspace is None:
            return _missing_workspace_result()
        with tool_argument_validation():
            path = _validate_workspace_path_argument(_require_arg_string(args, "path"))
            expected_revision = _require_revision(args, "expected_revision")
            edits = _validate_text_edits(args.get("edits"))
            max_bytes = _optional_int(
                args,
                "max_bytes",
                default=DEFAULT_WRITE_LIMIT_BYTES,
                maximum=MAX_WRITE_LIMIT_BYTES,
            )
            max_diff_bytes = _optional_int(
                args,
                "max_diff_bytes",
                default=DEFAULT_EDIT_DIFF_LIMIT_BYTES,
                maximum=MAX_EDIT_DIFF_LIMIT_BYTES,
            )
            if max_diff_bytes < 256:
                raise ValueError("Tool argument `max_diff_bytes` must be at least 256.")

        try:
            read = await workspace.read_bytes(path, max_bytes=max_bytes)
        except FileNotFoundError:
            return ToolResult(
                content=f"Edit refused: workspace file not found: {path}.",
                structured={"path": path, "reason": "not_found"},
                is_error=True,
            )
        if read.truncated:
            return ToolResult(
                content=(
                    f"Edit refused: {path} is {read.total_bytes} bytes and exceeds "
                    f"max_bytes={max_bytes}."
                ),
                structured={
                    "path": path,
                    "total_bytes": read.total_bytes,
                    "max_bytes": max_bytes,
                    "reason": "file_truncated",
                },
                is_error=True,
            )
        before_sha256 = _content_hash(read.content)
        if read.revision != expected_revision:
            return ToolResult(
                content=(
                    f"Edit refused: {path} changed since it was read; expected revision "
                    f"{expected_revision}, found {read.revision}."
                ),
                structured={
                    "path": path,
                    "expected_revision": expected_revision,
                    "actual_revision": read.revision,
                    "reason": "stale_content",
                },
                is_error=True,
            )
        try:
            original = read.content.decode("utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                content=f"Edit refused: {path} is not valid UTF-8 text.",
                structured={"path": path, "encoding": "utf-8", "reason": "invalid_encoding"},
                is_error=True,
            )

        replacements: list[tuple[int, int, str, int]] = []
        replacement_count = 0
        for index, edit in enumerate(edits):
            starts = _exact_match_offsets(original, edit.old_text)
            actual_replacements = len(starts)
            if actual_replacements != edit.expected_replacements:
                return ToolResult(
                    content=(
                        f"Edit refused: edit {index} expected "
                        f"{edit.expected_replacements} exact occurrence(s), "
                        f"found {actual_replacements}. No changes were written."
                    ),
                    structured={
                        "path": path,
                        "edit_index": index,
                        "expected_replacements": edit.expected_replacements,
                        "actual_replacements": actual_replacements,
                    },
                    is_error=True,
                )
            replacements.extend(
                (start, start + len(edit.old_text), edit.new_text, index) for start in starts
            )
            replacement_count += actual_replacements
        replacements.sort()
        for previous, current in pairwise(replacements):
            if current[0] < previous[1]:
                return ToolResult(
                    content=(
                        f"Edit refused: edits {previous[3]} and {current[3]} overlap in "
                        f"{path}. No changes were written."
                    ),
                    structured={
                        "path": path,
                        "first_edit_index": previous[3],
                        "second_edit_index": current[3],
                        "reason": "overlapping_edits",
                    },
                    is_error=True,
                )
        edit_byte_deltas = [
            _utf8_text_size(edit.new_text) - _utf8_text_size(edit.old_text) for edit in edits
        ]
        projected_bytes = len(read.content) + sum(
            edit_byte_deltas[edit_index] for _start, _end, _replacement, edit_index in replacements
        )
        if projected_bytes > max_bytes:
            return ToolResult(
                content=(
                    f"Edit refused: result is {projected_bytes} bytes, "
                    f"which exceeds max_bytes={max_bytes}."
                ),
                structured={
                    "path": path,
                    "after_bytes": projected_bytes,
                    "max_bytes": max_bytes,
                    "reason": "result_too_large",
                },
                is_error=True,
            )
        updated_parts: list[str] = []
        cursor = 0
        for start, end, replacement, _ in replacements:
            updated_parts.extend((original[cursor:start], replacement))
            cursor = end
        updated_parts.append(original[cursor:])
        updated = "".join(updated_parts)

        encoded = updated.encode("utf-8")
        if len(encoded) != projected_bytes:
            raise AssertionError("Edit result byte-size projection drifted.")
        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        diff_snapshot = active_secret_redactor_snapshot(ctx)
        diff_preview, diff_truncated = diff_snapshot.redactor.redact_text_bounded_with_marker(
            diff,
            max_bytes=max_diff_bytes,
            truncation_marker="\n[edit diff truncated]",
        )
        if diff_truncated:
            record_ambiguous_secret_output(ctx, diff_snapshot)
        try:
            mutation = await workspace.replace_bytes(
                path,
                encoded,
                expected_revision=expected_revision,
            )
        except WorkspaceRevisionMismatchError as exc:
            return _stale_mutation_result("Edit", path, exc)
        except FileNotFoundError:
            return _missing_mutation_target_result("Edit", path)
        after_sha256 = mutation.after_sha256
        summary = (
            f"Edited {path}: {len(edits)} edit(s), {replacement_count} replacement(s), "
            f"{len(read.content)} -> {len(encoded)} bytes."
        )
        return ToolResult(
            content=f"{summary}\n\n{diff_preview}" if diff_preview else summary,
            structured={
                "path": path,
                "edit_count": len(edits),
                "replacement_count": replacement_count,
                "before_sha256": before_sha256,
                "after_sha256": after_sha256,
                "before_revision": mutation.before_revision,
                "after_revision": mutation.after_revision,
                "before_bytes": len(read.content),
                "after_bytes": len(encoded),
                "diff": diff_preview,
                "diff_truncated": diff_truncated,
                "max_diff_bytes": max_diff_bytes,
            },
        )


class DeleteFileTool(Tool):
    """Delete one workspace file only when its content precondition matches."""

    spec = ToolSpec(
        name="delete_file",
        parallel_safe=False,
        effect=ToolEffect.EXTERNAL,
        description=(
            "Delete one workspace-relative file after verifying the opaque revision "
            "returned by read_file. Refuses missing, stale, directory, and oversized paths."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path of the file to delete.",
                },
                "expected_revision": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_FILE_REVISION_CHARS,
                    "description": "Opaque revision from a complete read_file result.",
                },
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_WRITE_LIMIT_BYTES,
                    "default": DEFAULT_WRITE_LIMIT_BYTES,
                },
            },
            "required": ["path", "expected_revision"],
            "additionalProperties": False,
        },
    )

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        workspace = _require_workspace(ctx)
        if workspace is None:
            return _missing_workspace_result()
        with tool_argument_validation():
            path = _validate_workspace_path_argument(_require_arg_string(args, "path"))
            expected_revision = _require_revision(args, "expected_revision")
            max_bytes = _optional_int(
                args,
                "max_bytes",
                default=DEFAULT_WRITE_LIMIT_BYTES,
                maximum=MAX_WRITE_LIMIT_BYTES,
            )
        try:
            read = await workspace.read_bytes(path, max_bytes=max_bytes)
        except FileNotFoundError:
            return ToolResult(
                content=f"Delete refused: workspace file not found: {path}.",
                structured={"path": path, "reason": "not_found"},
                is_error=True,
            )
        if read.truncated:
            return ToolResult(
                content=(
                    f"Delete refused: {path} is {read.total_bytes} bytes and exceeds "
                    f"max_bytes={max_bytes}."
                ),
                structured={
                    "path": path,
                    "total_bytes": read.total_bytes,
                    "max_bytes": max_bytes,
                    "reason": "file_truncated",
                },
                is_error=True,
            )
        actual_sha256 = _content_hash(read.content)
        if read.revision != expected_revision:
            return ToolResult(
                content=(
                    f"Delete refused: {path} changed since it was read; expected revision "
                    f"{expected_revision}, found {read.revision}."
                ),
                structured={
                    "path": path,
                    "expected_revision": expected_revision,
                    "actual_revision": read.revision,
                    "reason": "stale_content",
                },
                is_error=True,
            )
        try:
            mutation = await workspace.delete_if_revision(
                path,
                expected_revision=expected_revision,
            )
        except WorkspaceRevisionMismatchError as exc:
            return _stale_mutation_result("Delete", path, exc)
        except FileNotFoundError:
            return _missing_mutation_target_result("Delete", path)
        return ToolResult(
            content=f"Deleted {path} ({len(read.content)} bytes).",
            structured={
                "path": path,
                "deleted_bytes": len(read.content),
                "deleted_sha256": actual_sha256,
                "deleted_revision": mutation.before_revision,
            },
        )


class WriteFileTool(Tool):
    spec = ToolSpec(
        name="write_file",
        # Mutates the workspace; never overlaps other tools in a round.
        parallel_safe=False,
        effect=ToolEffect.EXTERNAL,
        description=(
            "Create a missing UTF-8 file or conditionally overwrite an existing file. "
            "Overwrite requires the opaque revision returned by a complete read_file."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["create", "overwrite"],
                    "description": "Use create for missing paths or overwrite for existing files.",
                },
                "expected_revision": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_FILE_REVISION_CHARS,
                    "description": "Required when mode is overwrite; omit for create.",
                },
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_WRITE_LIMIT_BYTES,
                    "default": DEFAULT_WRITE_LIMIT_BYTES,
                },
            },
            "required": ["path", "content", "mode"],
            "additionalProperties": False,
        },
    )

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        workspace = _require_workspace(ctx)
        if workspace is None:
            return _missing_workspace_result()
        with tool_argument_validation():
            path = _validate_workspace_path_argument(_require_arg_string(args, "path"))
            content = _require_arg_string(args, "content", allow_blank=True)
            mode = _require_arg_string(args, "mode")
            if mode not in {"create", "overwrite"}:
                raise ValueError("Tool argument `mode` must be `create` or `overwrite`.")
            expected_revision = (
                _require_revision(args, "expected_revision")
                if args.get("expected_revision") is not None
                else None
            )
            if mode == "create" and expected_revision is not None:
                raise ValueError("Tool argument `expected_revision` is invalid for create mode.")
            if mode == "overwrite" and expected_revision is None:
                raise ValueError(
                    "Tool argument `expected_revision` is required for overwrite mode."
                )
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
        try:
            if mode == "create":
                mutation = await workspace.create_bytes(path, encoded)
            else:
                assert expected_revision is not None
                mutation = await workspace.replace_bytes(
                    path,
                    encoded,
                    expected_revision=expected_revision,
                )
        except FileExistsError:
            return ToolResult(
                content=f"Write refused: workspace file already exists: {path}.",
                structured={"path": path, "mode": mode, "reason": "already_exists"},
                is_error=True,
            )
        except WorkspaceRevisionMismatchError as exc:
            return _stale_mutation_result("Write", path, exc)
        except FileNotFoundError:
            return _missing_mutation_target_result("Write", path)
        return ToolResult(
            content=f"Wrote {len(encoded)} bytes to {path}.",
            structured={
                "path": path,
                "bytes": len(encoded),
                "encoding": "utf-8",
                "mode": mode,
                "revision": mutation.after_revision,
                "sha256": mutation.after_sha256,
            },
        )


class ListFilesTool(Tool):
    spec = ToolSpec(
        name="list_files",
        effect=ToolEffect.NONE,
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

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        workspace = _require_workspace(ctx)
        if workspace is None:
            return _missing_workspace_result()
        with tool_argument_validation():
            pattern = args.get("pattern", "**/*")
            if type(pattern) is not str:
                raise ValueError("Tool argument `pattern` must be a string.")
            pattern = require_unicode_scalar_text(pattern, "pattern")
            if "\0" in pattern:
                raise ValueError("Workspace list patterns must not contain NUL characters.")
            pattern = validate_list_pattern(pattern)
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


class ListArtifactsTool(Tool):
    spec = ToolSpec(
        name="list_artifacts",
        effect=ToolEffect.NONE,
        description=(
            "List uploaded/generated artifacts available in the active artifact store. "
            "Use read_file with artifact_id to inspect a listed artifact."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": [ArtifactScope.SESSION.value, ArtifactScope.ENVIRONMENT.value],
                    "default": ArtifactScope.SESSION.value,
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

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        artifact_store = _require_artifact_store(ctx)
        if artifact_store is None:
            return _missing_artifact_store_result()
        with tool_argument_validation():
            scope = _optional_scope(args, "scope", default=ArtifactScope.SESSION)
            limit = _optional_int(
                args,
                "limit",
                default=DEFAULT_LIST_LIMIT,
                maximum=MAX_LIST_LIMIT,
            )
        session_id = ctx.session_id if scope == ArtifactScope.SESSION else None
        environment_name = ctx.environment_name if scope == ArtifactScope.ENVIRONMENT else None
        if scope == ArtifactScope.ENVIRONMENT and environment_name is None:
            return ToolResult(
                content="No environment configured for environment-scoped artifact listing.",
                is_error=True,
            )
        result = await artifact_store.list(
            scope=scope,
            session_id=session_id,
            environment_name=environment_name,
            limit=limit,
        )
        artifacts = [
            {
                "artifact_id": artifact.id,
                "filename": artifact.filename,
                "content_type": artifact.content_type,
                "size_bytes": artifact.size_bytes,
                "scope": artifact.scope.value,
                "session_id": artifact.session_id,
                "agent_name": artifact.agent_name,
                "environment_name": artifact.environment_name,
                "created_at": artifact.created_at.isoformat(),
                "metadata": artifact.metadata,
            }
            for artifact in result.artifacts
        ]
        lines = [
            (
                f"{artifact['artifact_id']} {artifact['filename']} "
                f"{artifact['content_type']} {artifact['size_bytes']} bytes"
            )
            for artifact in artifacts
        ]
        result_content = "\n".join(lines) if lines else "No artifacts matched."
        if result.truncated:
            result_content = f"{result_content}\n\n[artifact list truncated]"
        return ToolResult(
            content=result_content,
            structured={
                "scope": scope.value,
                "artifacts": artifacts,
                "total_artifacts": result.total_count,
                "truncated": result.truncated,
            },
        )


def _require_workspace(ctx: ToolContext) -> Workspace | None:
    authority = ctx._authoritative_workspace_for_builtin()
    workspace = authority if authority is not None else ctx.workspace
    if workspace is None:
        return None
    if not isinstance(workspace, Workspace):
        raise TypeError("Tool context workspace must implement Workspace.")
    return workspace


def _require_artifact_store(ctx: ToolContext) -> ArtifactStore | None:
    authority = ctx._authoritative_artifact_store_for_builtin()
    artifact_store = authority if authority is not None else ctx.artifact_store
    if artifact_store is None:
        return None
    if not isinstance(artifact_store, ArtifactStore):
        raise TypeError("Tool context artifact_store must implement ArtifactStore.")
    return artifact_store


async def _read_artifact_store(
    artifact_store: ArtifactStore,
    artifact_id: str,
    *,
    max_bytes: int,
) -> ArtifactReadResult:
    return copy_artifact_read_result(
        await artifact_store.read_bytes(artifact_id, max_bytes=max_bytes),
        expected_artifact_id=artifact_id,
        max_content_bytes=max_bytes,
    )


def _missing_workspace_result() -> ToolResult:
    return ToolResult(
        content="No workspace configured for this tool call.",
        is_error=True,
    )


def _missing_artifact_store_result() -> ToolResult:
    return ToolResult(
        content="No artifact store configured for this tool call.",
        is_error=True,
    )


def _artifact_access_error(
    ctx: ToolContext,
    artifact: ArtifactMetadata,
) -> ToolResult | None:
    if artifact.scope == ArtifactScope.SESSION and artifact.session_id != ctx.session_id:
        return ToolResult(
            content="Artifact is not available in this session.",
            structured={
                "artifact_id": artifact.id,
                "scope": artifact.scope.value,
            },
            is_error=True,
        )
    if (
        artifact.scope == ArtifactScope.ENVIRONMENT
        and artifact.environment_name != ctx.environment_name
    ):
        return ToolResult(
            content="Artifact is not available in this environment.",
            structured={
                "artifact_id": artifact.id,
                "scope": artifact.scope.value,
            },
            is_error=True,
        )
    return None


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


def _exact_match_offsets(text: str, needle: str) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    while True:
        offset = text.find(needle, cursor)
        if offset < 0:
            return offsets
        offsets.append(offset)
        cursor = offset + len(needle)


def _require_revision(args: dict, key: str) -> str:
    value = require_unicode_scalar_text(_require_arg_string(args, key), key)
    if "\0" in value:
        raise ValueError(f"Tool argument `{key}` must not contain NUL characters.")
    if len(value) > MAX_FILE_REVISION_CHARS:
        raise ValueError(
            f"Tool argument `{key}` must be at most {MAX_FILE_REVISION_CHARS} characters."
        )
    return value


def _stale_mutation_result(
    operation: str,
    path: str,
    error: WorkspaceRevisionMismatchError,
) -> ToolResult:
    return ToolResult(
        content=(
            f"{operation} refused: {path} changed during the operation; expected revision "
            f"{error.expected_revision}, found {error.actual_revision}."
        ),
        structured={
            "path": path,
            "expected_revision": error.expected_revision,
            "actual_revision": error.actual_revision,
            "reason": "stale_content",
        },
        is_error=True,
    )


def _missing_mutation_target_result(operation: str, path: str) -> ToolResult:
    return ToolResult(
        content=f"{operation} refused: workspace file not found: {path}.",
        structured={"path": path, "reason": "not_found"},
        is_error=True,
    )


def _validate_workspace_path_argument(path: str) -> str:
    # This is intentionally a tool-level, backend-independent validation boundary:
    # private backend validators run too late to classify model input, while rejecting
    # Windows roots/drives here keeps paths portable. Preserve the caller's spelling so
    # custom Workspace implementations continue to receive the original relative path.
    value = require_unicode_scalar_text(require_nonblank(path, "path"), "path")
    if "\0" in value:
        raise ValueError("Workspace paths must not contain NUL characters.")
    windows_path = PureWindowsPath(value)
    if (
        posixpath.isabs(value)
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
    ):
        raise ValueError("Workspace paths must be relative.")
    normalized = posixpath.normpath(value)
    windows_normalized = ntpath.normpath(value)
    if normalized in {"", "."}:
        raise ValueError("Workspace paths must reference a file.")
    if (
        normalized == ".."
        or normalized.startswith("../")
        or windows_normalized == ".."
        or windows_normalized.startswith("..\\")
    ):
        raise ValueError("Workspace path escapes the workspace root.")
    return value


def _optional_arg_string(args: dict, key: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"Tool argument `{key}` must be a string.")
    return require_unicode_scalar_text(require_nonblank(value, key), key)


def _require_sha256(args: dict, key: str) -> str:
    value = _require_arg_string(args, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Tool argument `{key}` must be a lowercase SHA-256 digest.")
    return value


def _validate_text_edits(value: object) -> tuple[_TextEdit, ...]:
    if not isinstance(value, list):
        raise ValueError("Tool argument `edits` must be an array.")
    if not value:
        raise ValueError("Tool argument `edits` cannot be empty.")
    if len(value) > MAX_EDIT_OPERATIONS:
        raise ValueError(f"Tool argument `edits` must contain at most {MAX_EDIT_OPERATIONS} items.")
    edits: list[_TextEdit] = []
    for index, item in enumerate(value):
        if type(item) is not dict:
            raise ValueError(f"Tool argument `edits[{index}]` must be an object.")
        unknown = set(item) - {"old_text", "new_text", "expected_replacements"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Tool argument `edits[{index}]` has unknown fields: {names}.")
        old_text = _require_arg_string(item, "old_text")
        new_text = _require_arg_string(item, "new_text", allow_blank=True)
        old_text = require_unicode_scalar_text(old_text, f"edits[{index}].old_text")
        new_text = require_unicode_scalar_text(new_text, f"edits[{index}].new_text")
        if old_text == new_text:
            raise ValueError(f"Tool argument `edits[{index}]` must change the matched text.")
        expected_replacements = _optional_int(
            item,
            "expected_replacements",
            default=1,
            maximum=1000,
        )
        edits.append(
            _TextEdit(
                old_text=old_text,
                new_text=new_text,
                expected_replacements=expected_replacements,
            )
        )
    return tuple(edits)


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


def _optional_nonnegative_int(args: dict, key: str, *, default: int) -> int:
    value = args.get(key, default)
    if type(value) is not int:
        raise ValueError(f"Tool argument `{key}` must be an integer.")
    if value < 0:
        raise ValueError(f"Tool argument `{key}` must be non-negative.")
    return value


def _truncate_utf8_text(value: str, maximum: int, *, marker: str) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum:
        return value, False
    marker_bytes = marker.encode("utf-8")
    prefix = encoded[: maximum - len(marker_bytes)]
    return prefix.decode("utf-8", errors="ignore").rstrip() + marker, True


def _optional_scope(
    args: dict,
    key: str,
    *,
    default: ArtifactScope,
) -> ArtifactScope:
    value = args.get(key, default.value)
    if isinstance(value, ArtifactScope):
        return value
    if type(value) is not str:
        raise ValueError(f"Tool argument `{key}` must be a string.")
    try:
        return ArtifactScope(value)
    except ValueError as exc:
        raise ValueError(f"Tool argument `{key}` has unsupported scope: {value!r}.") from exc


def _is_text_content_type(content_type: str) -> bool:
    normalized = content_type.lower()
    return (
        normalized.startswith("text/")
        or normalized in _TEXT_CONTENT_TYPES
        or normalized.endswith(_TEXT_CONTENT_TYPE_SUFFIXES)
    )


def _resize_image_bytes(
    content: bytes,
    *,
    content_type: str,
    max_bytes: int,
) -> tuple[bytes, str] | None:
    """Resize bytes already accepted by the shared bounded image decoder."""
    try:
        image_module = import_module("PIL.Image")
    except ImportError:
        return None

    preferred_format = "PNG" if content_type == "image/png" else "JPEG"
    with image_module.open(io.BytesIO(content)) as image:
        source = image.copy()

    for max_side in (2048, 1600, 1280, 1024):
        resized = source.copy()
        resized.thumbnail((max_side, max_side))
        encoded = _encode_image(resized, preferred_format)
        if len(encoded) <= max_bytes:
            return encoded, f"image/{preferred_format.lower()}"

    jpeg_source = _flatten_image_on_white(source)
    for max_side in (2048, 1600, 1280, 1024):
        resized = jpeg_source.copy()
        resized.thumbnail((max_side, max_side))
        for quality in (85, 75, 65, 55, 45, 35):
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=quality, optimize=True)
            encoded = buffer.getvalue()
            if len(encoded) <= max_bytes:
                return encoded, "image/jpeg"
    return None


def _detect_image_content_type(content: bytes) -> tuple[str | None, str | None]:
    try:
        image_module = import_module("PIL.Image")
    except ImportError:
        return None, "Install cayu[files] or register a custom image reader."

    try:
        detected_format = decode_verified_image_format(image_module, content)
    except Exception as exc:
        return None, str(exc)

    detected_content_type = _PIL_IMAGE_FORMAT_CONTENT_TYPES.get(str(detected_format).upper())
    if detected_content_type is None:
        return None, f"Unsupported image format: {detected_format}."
    return detected_content_type, None


def _encode_image(image, preferred_format: str) -> bytes:
    buffer = io.BytesIO()
    if preferred_format == "JPEG":
        image = _flatten_image_on_white(image)
        image.save(buffer, format="JPEG", quality=85, optimize=True)
    else:
        image.save(buffer, format=preferred_format, optimize=True)
    return buffer.getvalue()


def _flatten_image_on_white(image):
    image_module = import_module("PIL.Image")

    if image.mode == "RGB":
        return image
    rgba = image.convert("RGBA")
    white = image_module.new("RGBA", rgba.size, (255, 255, 255, 255))
    white.alpha_composite(rgba)
    return white.convert("RGB")


def _extract_pdf_pages(content: bytes, pages: str | None) -> tuple[bytes, str] | None:
    try:
        pypdf = import_module("pypdf")
    except ImportError:
        return None

    reader = pypdf.PdfReader(io.BytesIO(content))
    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError("PDF has no pages.")
    start, end = _parse_pdf_page_range(pages, total_pages)
    writer = pypdf.PdfWriter()
    for page_index in range(start, end):
        writer.add_page(reader.pages[page_index])
    buffer = io.BytesIO()
    writer.write(buffer)
    if pages is None and total_pages > MAX_PDF_PAGES_PER_READ:
        page_note = f" (showing pages 1-{end} of {total_pages})"
    elif pages is None:
        page_note = f" ({total_pages} pages)"
    else:
        page_note = f" (showing pages {start + 1}-{end} of {total_pages})"
    return buffer.getvalue(), page_note


def _count_pdf_pages(content: bytes) -> int | None:
    try:
        pypdf = import_module("pypdf")
    except ImportError:
        return None
    reader = pypdf.PdfReader(io.BytesIO(content))
    return len(reader.pages)


def _content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _utf8_text_size(value: str) -> int:
    size = 0
    for char in value:
        codepoint = ord(char)
        size += (
            1 if codepoint < 0x80 else 2 if codepoint < 0x800 else 3 if codepoint < 0x10000 else 4
        )
    return size


def _derivation_key(*, source_hash: str, operation: str, params: str) -> str:
    payload = f"{source_hash}\x00{operation}\x00{params}".encode()
    return hashlib.sha256(payload).hexdigest()


async def _find_derived_artifact(
    artifact_store: ArtifactStore,
    ctx: ToolContext,
    derivation_key: str,
) -> ArtifactMetadata | None:
    """Return a previously derived session artifact for this derivation, if any.

    Re-reading the same image/PDF each turn re-runs the same resize/page-extract
    and would otherwise store an identical (potentially multi-MB) copy every time.
    Derivations are deterministic in (source bytes, operation, params), so a prior
    result carrying the same derivation key can be reused verbatim.
    """
    if ctx.session_id is None:
        return None
    try:
        listing = await artifact_store.list(
            scope=ArtifactScope.SESSION,
            session_id=ctx.session_id,
        )
    except Exception:
        return None
    for meta in listing.artifacts:
        if meta.metadata.get("cayu_derivation_key") == derivation_key:
            return meta
    return None


def _validate_pdf_bytes(content: bytes) -> str | None:
    try:
        pypdf = import_module("pypdf")
    except ImportError:
        return "Install cayu[files] or register a custom PDF reader."

    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        total_pages = len(reader.pages)
    except Exception as exc:
        return str(exc)
    if total_pages == 0:
        return "PDF has no pages."
    return None


def _parse_pdf_page_range(pages: str | None, total_pages: int) -> tuple[int, int]:
    if pages is None:
        return 0, min(total_pages, MAX_PDF_PAGES_PER_READ)
    value = pages.strip()
    if not value:
        raise _InvalidPdfPageRange("Tool argument `pages` cannot be blank.")
    if "-" in value:
        start_text, end_text = value.split("-", 1)
    else:
        start_text = value
        end_text = value
    if not start_text.isdecimal() or not end_text.isdecimal():
        raise _InvalidPdfPageRange("Tool argument `pages` must be a page number or range.")
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError as exc:
        raise _InvalidPdfPageRange("Tool argument `pages` must be a page number or range.") from exc
    if start <= 0 or end <= 0 or end < start:
        raise _InvalidPdfPageRange("Tool argument `pages` must be a valid 1-based page range.")
    if start > total_pages:
        raise _InvalidPdfPageRange("Tool argument `pages` starts after the end of the PDF.")
    end = min(end, total_pages)
    if end - start + 1 > MAX_PDF_PAGES_PER_READ:
        raise _InvalidPdfPageRange(
            f"Tool argument `pages` may include at most {MAX_PDF_PAGES_PER_READ} pages."
        )
    return start - 1, end
