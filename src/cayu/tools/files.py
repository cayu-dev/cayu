from __future__ import annotations

import hashlib
import io
import mimetypes
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import PurePosixPath
from typing import Protocol

from cayu._validation import require_nonblank
from cayu.artifacts import (
    DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
    FILE_ATTACHMENT_IMAGE_CONTENT_TYPES,
    ArtifactMetadata,
    ArtifactReadResult,
    ArtifactScope,
    ArtifactStore,
    FileAttachmentKind,
    file_attachment,
)
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools._errors import structured_invalid_arguments
from cayu.workspaces import Workspace, WorkspaceReadResult

DEFAULT_READ_LIMIT_BYTES = 256 * 1024
MAX_READ_LIMIT_BYTES = 4 * 1024 * 1024
DEFAULT_ATTACHMENT_LIMIT_BYTES = DEFAULT_MAX_FILE_ATTACHMENT_BYTES
DEFAULT_MAX_ATTACHMENT_LIMIT_BYTES = DEFAULT_MAX_FILE_ATTACHMENT_BYTES
MAX_IMAGE_SOURCE_BYTES = 20 * 1024 * 1024
MAX_PDF_SOURCE_BYTES = 32 * 1024 * 1024
MAX_PDF_PAGES_PER_READ = 10
DEFAULT_WRITE_LIMIT_BYTES = 256 * 1024
MAX_WRITE_LIMIT_BYTES = 4 * 1024 * 1024
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


@dataclass(frozen=True)
class ReadFileOptions:
    max_bytes: int = DEFAULT_READ_LIMIT_BYTES
    max_attachment_bytes: int = DEFAULT_ATTACHMENT_LIMIT_BYTES
    pages: str | None = None


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
        description=(
            "Read a file from the active workspace or read an artifact by id. "
            "Use `path` for workspace files and `artifact_id` for uploaded/generated artifacts. "
            "Text files return text. Workspace image/PDF files are captured as artifact "
            "snapshots when an artifact store is configured. Image and PDF artifacts return "
            "provider-neutral file attachments that capable providers can inspect natively."
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
        path = _optional_arg_string(args, "path")
        artifact_id = _optional_arg_string(args, "artifact_id")
        if (path is None) == (artifact_id is None):
            raise ValueError("Tool arguments must include exactly one of `path` or `artifact_id`.")
        max_bytes = _optional_int(
            args,
            "max_bytes",
            default=DEFAULT_READ_LIMIT_BYTES,
            maximum=MAX_READ_LIMIT_BYTES,
        )
        max_attachment_bytes = _optional_int(
            args,
            "max_attachment_bytes",
            default=self.default_attachment_limit_bytes,
            maximum=self.max_attachment_limit_bytes,
        )
        pages = _optional_arg_string(args, "pages")
        if path is not None:
            return await _read_workspace_file(
                ctx,
                path=path,
                artifact_readers=self.artifact_readers,
                max_bytes=max_bytes,
                max_attachment_bytes=max_attachment_bytes,
                pages=pages,
            )
        if artifact_id is not None:
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
    max_attachment_bytes: int,
    pages: str | None,
) -> ToolResult:
    workspace = _require_workspace(ctx)
    if workspace is None:
        return _missing_workspace_result()
    result = await workspace.read_bytes(path, max_bytes=max_bytes)
    content_type = _guess_workspace_content_type(path)
    if _is_workspace_file_attachment_content_type(content_type):
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
    if _is_binary_workspace_file(content_type=content_type, content=result.content):
        return _binary_workspace_file_result(
            path=path,
            content_type=content_type,
            bytes_read=len(result.content),
            total_bytes=result.total_bytes,
            truncated=result.truncated,
            inspectable=False,
        )
    if pages is not None:
        raise ValueError("Tool argument `pages` is only valid for PDF files.")
    text = result.content.decode("utf-8", errors="replace")
    return ToolResult(
        content=f"{text}\n\n[file truncated]" if result.truncated else text,
        structured={
            "source": "workspace",
            "path": path,
            "bytes": len(result.content),
            "total_bytes": result.total_bytes,
            "encoding": "utf-8",
            "truncated": result.truncated,
        },
    )


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
    result = await artifact_store.read_bytes(artifact_id, max_bytes=max_bytes)
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
            raise ValueError("Tool argument `pages` is only valid for PDF artifacts.")
        text = request.initial_read.content.decode("utf-8", errors="replace")
        return ToolResult(
            content=f"{text}\n\n[file truncated]" if request.initial_read.truncated else text,
            structured={
                **request.structured,
                "encoding": "utf-8",
            },
        )


class ImageArtifactReader:
    def can_read(self, artifact: ArtifactMetadata) -> bool:
        return artifact.content_type in IMAGE_CONTENT_TYPES

    async def read(self, request: ArtifactReadRequest) -> ToolResult:
        if request.options.pages is not None:
            raise ValueError("Tool argument `pages` is only valid for PDF artifacts.")
        artifact_store = request.artifact_store
        artifact = request.artifact
        if artifact.size_bytes == 0:
            return ToolResult(
                content=f"Image artifact '{artifact.filename}' is empty and cannot be inspected.",
                structured=request.structured,
                is_error=True,
            )
        result = await artifact_store.read_bytes(
            artifact.id,
            max_bytes=request.options.max_attachment_bytes,
        )
        attachment_artifact = artifact
        if result.truncated:
            source = await artifact_store.read_bytes(
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
            detected_content_type, validation_error = _detect_image_content_type(source.content)
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
            detected_content_type, validation_error = _detect_image_content_type(result.content)
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
            result = await artifact_store.read_bytes(
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
                source = await artifact_store.read_bytes(
                    artifact.id, max_bytes=MAX_PDF_SOURCE_BYTES
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

    @structured_invalid_arguments
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

    @structured_invalid_arguments
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


class ListArtifactsTool(Tool):
    spec = ToolSpec(
        name="list_artifacts",
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
    if ctx.workspace is None:
        return None
    if not isinstance(ctx.workspace, Workspace):
        raise TypeError("Tool context workspace must implement Workspace.")
    return ctx.workspace


def _require_artifact_store(ctx: ToolContext) -> ArtifactStore | None:
    if ctx.artifact_store is None:
        return None
    if not isinstance(ctx.artifact_store, ArtifactStore):
        raise TypeError("Tool context artifact_store must implement ArtifactStore.")
    return ctx.artifact_store


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


def _optional_arg_string(args: dict, key: str) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"Tool argument `{key}` must be a string.")
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
        with image_module.open(io.BytesIO(content)) as image:
            detected_format = image.format
            image.verify()
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
        raise ValueError("Tool argument `pages` cannot be blank.")
    if "-" in value:
        start_text, end_text = value.split("-", 1)
    else:
        start_text = value
        end_text = value
    if not start_text.isdecimal() or not end_text.isdecimal():
        raise ValueError("Tool argument `pages` must be a page number or range.")
    start = int(start_text)
    end = int(end_text)
    if start <= 0 or end <= 0 or end < start:
        raise ValueError("Tool argument `pages` must be a valid 1-based page range.")
    if start > total_pages:
        raise ValueError("Tool argument `pages` starts after the end of the PDF.")
    end = min(end, total_pages)
    if end - start + 1 > MAX_PDF_PAGES_PER_READ:
        raise ValueError(
            f"Tool argument `pages` may include at most {MAX_PDF_PAGES_PER_READ} pages."
        )
    return start - 1, end
