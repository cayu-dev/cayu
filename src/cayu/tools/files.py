from __future__ import annotations

import asyncio
import difflib
import hashlib
import io
import json
import mimetypes
import ntpath
import posixpath
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from importlib import import_module
from itertools import pairwise
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Protocol

from cayu._validation import require_nonblank, require_unicode_scalar_text
from cayu.artifacts._images import (
    ImageDecodePolicy,
    decode_verified_image_format,
    effective_image_decode_policy,
)
from cayu.artifacts.attachments import (
    DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
    FILE_ATTACHMENT_IMAGE_CONTENT_TYPES,
    FileAttachmentKind,
    file_attachment,
    file_attachment_from_payload,
)
from cayu.artifacts.base import (
    ArtifactIdentityConflictError,
    ArtifactMetadata,
    ArtifactReadResult,
    ArtifactScope,
    ArtifactStore,
    InvalidArtifactIdError,
    copy_artifact_read_result,
)
from cayu.tools._errors import (
    invalid_tool_arguments_result,
    reject_unknown_tool_arguments,
    structured_invalid_arguments,
    tool_argument_validation,
)
from cayu.tools._redaction import (
    InvocationRedactorSnapshot,
    active_secret_redactor,
    active_secret_redactor_snapshot,
    await_revision_stable_secret_output,
    record_ambiguous_secret_output,
    unstable_secret_redaction_result,
)
from cayu.tools._terminal_bounds import (
    DEFAULT_RECORD_TOOL_TERMINAL_PAYLOAD_BYTES,
    DEFAULT_TEXT_TOOL_TERMINAL_PAYLOAD_BYTES,
    MAX_TOOL_TERMINAL_PAYLOAD_BYTES,
    TERMINAL_PAYLOAD_FIT_MARGIN_BYTES,
    bounded_terminal_result,
    spec_with_terminal_payload_bound,
    tool_result_fits,
)
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.vaults import SecretRedactor
from cayu.workspaces.base import (
    Workspace,
    WorkspaceReadOffsetError,
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
# Providers accept PDFs of up to 100 pages natively; attachment byte limits
# still apply to the extracted pages.
MAX_PDF_PAGES_PER_READ = 100
DEFAULT_WRITE_LIMIT_BYTES = 256 * 1024
MAX_WRITE_LIMIT_BYTES = 4 * 1024 * 1024
DEFAULT_EDIT_DIFF_LIMIT_BYTES = 32 * 1024
MAX_EDIT_DIFF_LIMIT_BYTES = 200 * 1024
MAX_EDIT_OPERATIONS = 100
MAX_FILE_REVISION_CHARS = 4096
DEFAULT_LIST_LIMIT = 500
MAX_LIST_LIMIT = 10_000
DEFAULT_LIST_RESULT_LIMIT_BYTES = 50_000
MAX_LIST_RESULT_LIMIT_BYTES = 200_000
MAX_LIST_OFFSET = 1_000_000
DEFAULT_LIST_SCAN_LIMIT = 100_000
MAX_LIST_SCAN_LIMIT = 1_000_000
MAX_READ_FILES = 20
# Room in a read_file result for its metadata header and structured fields.
_READ_RESULT_FRAMING_BYTES = 64 * 1024
_MIN_REFITTED_READ_BYTES = 4 * 1024
_LINE_CUT_RETRY_HINT = "Retry next_start_line with a larger max_bytes to read the complete line."
_LINE_CUT_PAGE_HINT = (
    "This line is longer than read_file can return in one result. Read it in byte "
    "pages instead: omit start_line and follow offset=next_offset."
)
_READ_FILE_TRUNCATION_MARKER = "\n\n[file truncated]"
# Everything read_file appends to a text artifact page must fit in this, so a
# readback sized by ArtifactExternalizingToolResultPolicy stays inline.
_ARTIFACT_PAGE_METADATA_MAX_BYTES = 192
# Part of that allowance kept for the ignored-arguments note. The paging
# footer gets the rest, which fits offsets and sizes of up to 19 digits.
_ARTIFACT_PAGE_NOTE_MAX_BYTES = 64

_READ_FILE_ARGUMENTS = frozenset(
    {
        "path",
        "files",
        "artifact_id",
        "max_bytes",
        "offset",
        "start_line",
        "end_line",
        "max_attachment_bytes",
        "pages",
        "image_region",
    }
)
_READ_FILE_ENTRY_ARGUMENTS = frozenset({"path", "offset", "start_line", "end_line"})
_EDIT_FILE_ARGUMENTS = frozenset(
    {"path", "expected_revision", "edits", "max_bytes", "max_diff_bytes"}
)
_DELETE_FILE_ARGUMENTS = frozenset({"path", "expected_revision", "max_bytes"})
_WRITE_FILE_ARGUMENTS = frozenset({"path", "content", "mode", "expected_revision", "max_bytes"})
_LIST_FILES_ARGUMENTS = frozenset({"pattern", "limit", "offset", "max_result_bytes"})
_LIST_ARTIFACTS_ARGUMENTS = frozenset({"scope", "limit"})

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
    max_attachment_validation_bytes: int = DEFAULT_MAX_ATTACHMENT_LIMIT_BYTES
    pages: str | None = None
    image_region: tuple[int, int, int, int] | None = None
    offset: int = 0


@dataclass(frozen=True)
class _LineRange:
    start: int
    end: int | None


@dataclass(frozen=True)
class _ReadFileEntry:
    path: str
    offset: int
    line_range: _LineRange | None
    path_error: str | None
    has_explicit_range: bool


@dataclass(frozen=True)
class _WorkspaceText:
    text: str
    structured: dict
    capture_snapshot: InvocationRedactorSnapshot


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
    _missing_as_result: bool = False


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
        # Attachments are references to stored artifacts, not inline bytes. A
        # text page that would not fit is read again with a smaller max_bytes.
        # The default is the largest storable result, so any file Cayu can
        # publish whole can still be read whole for its revision.
        max_terminal_payload_bytes=MAX_TOOL_TERMINAL_PAYLOAD_BYTES,
        # Workspace image/PDF snapshots and derived attachments use content-addressed
        # artifact ids, so replaying a read returns the same artifacts.
        effect=ToolEffect.IDEMPOTENT,
        description=(
            "Read a file from the active workspace or read an artifact by id. "
            "Use `path` for workspace files and `artifact_id` for uploaded/generated artifacts. "
            "To read several workspace text files, pass `files` instead of `path`; request every "
            "file you already know you need in one call. `start_line` and `end_line` select whole "
            "lines of a workspace text file. "
            "Workspace text results begin with model-visible JSON metadata. A complete offset-zero "
            "read includes the opaque revision required by edit_file, write_file overwrite mode, "
            "and delete_file. Text artifact pages append JSON continuation metadata; repeat the "
            "same artifact_id and max_bytes with offset=next_offset until next_offset is null. "
            "Initial complete text reads have no footer. "
            "Workspace image/PDF files are "
            "captured as artifact snapshots when an artifact store is configured. Image and PDF "
            "artifacts return provider-neutral file attachments that capable providers can inspect "
            "natively."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Workspace-relative file path to read. Leave it out when you use `files`."
                    ),
                },
                # Provider schemas can't say "one of", so empty `files` and a
                # `path` repeated in it reach the tool, which accepts them.
                "files": {
                    "type": "array",
                    "maxItems": MAX_READ_FILES,
                    "description": (
                        "Several workspace text files to read in one call, instead of `path`. "
                        "Put each file's range in its own entry. "
                        "max_bytes bounds the whole result: files are read in order and each "
                        "may use an equal share of what earlier files left unused. A file that "
                        "can't be read reports its own error without failing the others. "
                        "Images and PDFs are read one at a time with `path`."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Workspace-relative file path to read.",
                            },
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                            "offset": {
                                "type": "integer",
                                "minimum": 0,
                                "description": "Byte offset to continue from; use next_offset.",
                            },
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
                "start_line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "First line to return (1-based) from a workspace text file. Line "
                        "results include total_lines and next_start_line instead of byte offsets."
                    ),
                },
                "end_line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Last line to return, inclusive. Defaults to the end of the file.",
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
                    "description": "Stored byte offset for workspace or artifact text reads; use next_offset.",
                },
                "max_attachment_bytes": {
                    "type": "integer",
                    "description": "Maximum bytes for a native image/PDF attachment sent to a provider.",
                    "minimum": 1,
                    "maximum": max_attachment_limit_bytes,
                    "default": default_attachment_limit_bytes,
                },
                "image_region": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "integer", "minimum": 0},
                    "description": (
                        "Image-only crop [left, top, right, bottom] in original source pixels; "
                        "right/bottom are exclusive. Returns frame 0 at original resolution. "
                        "Repeat with other coordinates on the source artifact to inspect more. "
                        "Source decoding must fit the host image policy. Leave it out to read "
                        "the whole image; a region of one pixel or less counts as left out. "
                        "Text reads ignore it."
                    ),
                },
                "pages": {
                    "type": "string",
                    "description": (
                        "PDF page range, such as '1-5' or '3'. At most 10 pages per read. "
                        "Text reads ignore it."
                    ),
                },
            },
            "additionalProperties": False,
        },
    )


class ReadFileTool(Tool):
    spec = _read_file_tool_spec(
        default_attachment_limit_bytes=DEFAULT_ATTACHMENT_LIMIT_BYTES,
        max_attachment_limit_bytes=DEFAULT_MAX_ATTACHMENT_LIMIT_BYTES,
    )

    def _policy_argument_sets(self, arguments: dict) -> tuple[dict, ...]:
        files = arguments.get("files")
        if not isinstance(files, list) or not files:
            return (arguments,)
        # Schema validation precedes authorization. Keep the whole batch visible
        # while checking every path, even with a renamed tool or custom policy.
        return tuple({**arguments, **entry} for entry in files)

    def _execution_profile_material(self) -> dict[str, object] | None:
        """Return bounded reader configuration when every reader is Cayu-owned."""

        reader_components: dict[type[ArtifactReader], str] = {
            ImageArtifactReader: "cayu.tools.files:ImageArtifactReader",
            PdfArtifactReader: "cayu.tools.files:PdfArtifactReader",
            TextArtifactReader: "cayu.tools.files:TextArtifactReader",
        }
        try:
            artifact_readers = [reader_components[type(reader)] for reader in self.artifact_readers]
        except KeyError:
            return None
        return {
            "default_attachment_limit_bytes": self.default_attachment_limit_bytes,
            "max_attachment_limit_bytes": self.max_attachment_limit_bytes,
            "artifact_readers": artifact_readers,
            "batch_policy_projection_version": 1,
            "line_redaction_version": 1,
        }

    def __init__(
        self,
        *,
        extra_artifact_readers: Iterable[ArtifactReader] | None = None,
        artifact_readers: Iterable[ArtifactReader] | None = None,
        default_attachment_limit_bytes: int = DEFAULT_ATTACHMENT_LIMIT_BYTES,
        max_attachment_limit_bytes: int = DEFAULT_MAX_ATTACHMENT_LIMIT_BYTES,
        spec: ToolSpec | None = None,
        max_terminal_payload_bytes: int | None = None,
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
        super().__init__(
            spec=spec_with_terminal_payload_bound(
                spec,
                max_terminal_payload_bytes,
                default=MAX_TOOL_TERMINAL_PAYLOAD_BYTES,
            )
        )
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

    @bounded_terminal_result
    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        with tool_argument_validation():
            reject_unknown_tool_arguments(args, allowed=_READ_FILE_ARGUMENTS)
            # Models often fill in every field, so an empty string or an empty
            # `files` counts as left out rather than as a second target.
            path = _given_arg_string(args, "path")
            artifact_id = _given_arg_string(args, "artifact_id")
            files = args.get("files")
            entries = None
            if files is not None and files != []:
                entries = _read_file_entries(files)
            elif path is None and artifact_id is None:
                # A blank path is the only target here; say so.
                _optional_arg_string(args, "path")
            if entries is None and (path is None) == (artifact_id is None):
                raise ValueError(
                    "Tool arguments must include exactly one of `path`, `files` or `artifact_id`."
                )
            max_bytes = _optional_int(
                args,
                "max_bytes",
                default=DEFAULT_READ_LIMIT_BYTES,
                maximum=MAX_READ_LIMIT_BYTES,
            )
            offset = _optional_nonnegative_int(args, "offset", default=0)
            line_range = _optional_line_range(args)
            if line_range is not None:
                if artifact_id is not None and entries is None:
                    raise ValueError(
                        "Tool arguments `start_line` and `end_line` are only valid with `path`."
                    )
                if offset != 0:
                    raise ValueError(
                        "Tool argument `offset` cannot be combined with `start_line` or `end_line`."
                    )
            if entries is not None:
                entries = _batch_with_single_read_fields(
                    entries,
                    path=path,
                    artifact_id=artifact_id,
                    offset=offset,
                    line_range=line_range,
                )
                path = None
            max_attachment_bytes = _optional_int(
                args,
                "max_attachment_bytes",
                default=self.default_attachment_limit_bytes,
                maximum=self.max_attachment_limit_bytes,
            )
            # Batches read text only, so the image and PDF options don't apply.
            # A single read only learns the file type later, so a malformed
            # region fails only if the file turns out to be an image or PDF.
            pages = None if entries is not None else _given_arg_string(args, "pages")
            region, region_error = (
                (None, None)
                if entries is not None
                else _image_region_argument(args.get("image_region"))
            )

            if path is not None:
                path = _validate_workspace_path_argument(path)

        requested_max_bytes = max_bytes

        async def read(max_bytes: int) -> ToolResult:
            options = ReadFileOptions(
                max_bytes=max_bytes,
                max_attachment_bytes=max_attachment_bytes,
                max_attachment_validation_bytes=self.max_attachment_limit_bytes,
                pages=pages,
                image_region=region,
                offset=offset,
            )
            if entries is not None:
                return await _read_workspace_files(ctx, entries=entries, max_bytes=max_bytes)
            if path is not None:
                return await _read_workspace_file(
                    ctx,
                    path=path,
                    artifact_readers=self.artifact_readers,
                    offset=offset,
                    options=options,
                    image_region_error=region_error,
                    line_range=line_range,
                    # A larger max_bytes cannot help once the bound limits it.
                    line_cut_hint=(
                        _LINE_CUT_PAGE_HINT
                        if max_bytes < requested_max_bytes
                        else _LINE_CUT_RETRY_HINT
                    ),
                )
            if artifact_id is not None:
                return await _read_artifact(
                    ctx,
                    artifact_id=artifact_id,
                    artifact_readers=self.artifact_readers,
                    options=options,
                    missing_as_result=True,
                    image_region_error=region_error,
                )
            raise AssertionError("unreachable")

        # A page never asks for more text than the result may hold. Escaped
        # text can still outgrow it, so read a smaller page until the result
        # fits; its next_offset and truncated fields then stay accurate.
        terminal_bound = self.spec.max_terminal_payload_bytes
        if terminal_bound is not None:
            framing = min(_READ_RESULT_FRAMING_BYTES, terminal_bound // 4)
            max_bytes = min(max_bytes, terminal_bound - framing)
        result = await read(max_bytes)
        while (
            not tool_result_fits(result, terminal_bound)
            and max_bytes > _MIN_REFITTED_READ_BYTES
            and not result.is_error
        ):
            max_bytes = max(_MIN_REFITTED_READ_BYTES, max_bytes // 2)
            result = await read(max_bytes)
        return result


class _WorkspaceFileNotFoundError(RuntimeError):
    pass


def _bounded_workspace_read_limit(workspace: Workspace, max_bytes: int) -> int:
    value = workspace.bounded_read_limit(max_bytes)
    if type(value) is not int or value <= 0 or value > max_bytes:
        raise RuntimeError(
            f"{type(workspace).__name__}.bounded_read_limit() must return a positive integer "
            f"no greater than max_bytes={max_bytes}."
        )
    return value


async def _read_workspace_bytes(
    workspace: Workspace,
    path: str,
    *,
    offset: int = 0,
    max_bytes: int,
) -> tuple[WorkspaceReadResult, int]:
    read_limit = _bounded_workspace_read_limit(workspace, max_bytes)
    try:
        result = await workspace.read_bytes(path, offset=offset, max_bytes=read_limit)
    except FileNotFoundError as exc:
        raise _WorkspaceFileNotFoundError from exc
    return result, read_limit


async def _read_workspace_file(
    ctx: ToolContext,
    *,
    path: str,
    artifact_readers: list[ArtifactReader],
    offset: int,
    options: ReadFileOptions,
    image_region_error: str | None = None,
    line_range: _LineRange | None = None,
    line_cut_hint: str = _LINE_CUT_RETRY_HINT,
) -> ToolResult:
    workspace = _require_workspace(ctx)
    if workspace is None:
        return _missing_workspace_result()
    content_type = _guess_workspace_content_type(path)
    if _is_workspace_file_attachment_content_type(content_type):
        if image_region_error is not None:
            return invalid_tool_arguments_result(ValueError(image_region_error))
        if offset != 0:
            return invalid_tool_arguments_result(
                ValueError("Tool argument `offset` is only valid for workspace text files.")
            )
        if line_range is not None:
            return invalid_tool_arguments_result(
                ValueError(
                    "Tool arguments `start_line` and `end_line` are only valid for "
                    "workspace text files."
                )
            )
        try:
            result, _ = await _read_workspace_bytes(
                workspace,
                path,
                offset=offset,
                max_bytes=options.max_bytes,
            )
        except _WorkspaceFileNotFoundError:
            return _missing_workspace_file_result("Read", path)
        return await _read_workspace_file_attachment(
            ctx,
            path=path,
            content_type=content_type,
            artifact_readers=artifact_readers,
            options=options,
            initial_result=result,
        )
    # Models fill in every field, so the image and PDF options reach text
    # reads too. Text reads don't use them, so say they were ignored.
    ignored = _ignored_attachment_arguments(options, image_region_error)
    read = await _read_workspace_window(
        ctx,
        workspace,
        path=path,
        content_type=content_type,
        offset=offset,
        line_range=line_range,
        options=options,
        line_cut_hint=line_cut_hint,
    )
    if isinstance(read, ToolResult):
        return read
    structured = {**read.structured, **_ignored_arguments_fields(ignored)}
    return ToolResult(
        content=_model_visible_workspace_text(read.text, structured),
        structured=structured,
    )


async def _read_workspace_text(
    ctx: ToolContext,
    workspace: Workspace,
    *,
    path: str,
    content_type: str,
    offset: int,
    options: ReadFileOptions,
    preserve_line_breaks: bool = False,
) -> _WorkspaceText | ToolResult:
    async def read_window(redactor):
        if redactor.has_values:
            redaction_overlap = redactor.pagination_overlap_utf8_bytes
            fetch_offset = max(0, offset - redaction_overlap - 3)
            trailing_window_bytes = redaction_overlap + 4
        else:
            redaction_overlap = 0
            fetch_offset = offset
            trailing_window_bytes = 0
        prefix_bytes = offset - fetch_offset
        requested_read_limit = prefix_bytes + options.max_bytes + trailing_window_bytes
        result, effective_read_limit = await _read_workspace_bytes(
            workspace,
            path,
            offset=fetch_offset,
            max_bytes=requested_read_limit,
        )
        return (
            result,
            redaction_overlap,
            fetch_offset,
            effective_read_limit,
            effective_read_limit < requested_read_limit,
        )

    try:
        captured = await await_revision_stable_secret_output(ctx, read_window)
    except _WorkspaceFileNotFoundError:
        return _missing_workspace_file_result("Read", path)
    except WorkspaceReadOffsetError as exc:
        return _invalid_workspace_read_offset_result(
            offset=offset,
            total_bytes=exc.total_bytes,
        )
    if captured is None:
        return unstable_secret_redaction_result()
    (
        (
            result,
            redaction_overlap,
            fetch_offset,
            effective_read_limit,
            workspace_limit_constrained,
        ),
        capture_snapshot,
    ) = captured
    redactor = capture_snapshot.redactor
    if result.offset != fetch_offset:
        return _secret_redaction_window_unavailable_result(
            offset=offset,
            required_window_start=fetch_offset,
            observed_window_start=result.offset,
        )
    if offset > result.total_bytes:
        return _invalid_workspace_read_offset_result(
            offset=offset,
            total_bytes=result.total_bytes,
        )
    requested_page_start = offset - result.offset
    observed_requested_page = result.content[requested_page_start:]
    if observed_requested_page and observed_requested_page[0] & 0xC0 == 0x80:
        return invalid_tool_arguments_result(
            ValueError(
                "Tool argument `offset` splits a UTF-8 character; use the previous "
                "`next_offset` value."
            )
        )
    raw_read_limit_exhausted = (
        result.truncated
        and result.source_bytes_read == effective_read_limit
        and len(result.content) == effective_read_limit
        and not result.redaction_truncated
    )
    workspace_limit_exhausted = workspace_limit_constrained and raw_read_limit_exhausted
    exact_eof = offset == result.total_bytes
    observed_window_end = result.offset + len(result.content)
    workspace_page_limit_bytes = max(0, observed_window_end - offset)
    if redactor.has_values and observed_window_end < result.total_bytes:
        safe_page_end = max(offset, observed_window_end - redaction_overlap)
        workspace_page_limit_bytes = min(
            workspace_page_limit_bytes,
            safe_page_end - offset,
        )
        if workspace_page_limit_bytes == 0 and offset < result.total_bytes:
            return _secret_redaction_window_unavailable_result(
                offset=offset,
                required_window_end=min(
                    result.total_bytes,
                    offset + redaction_overlap + 1,
                ),
                observed_window_end=observed_window_end,
            )
    effective_page_limit_bytes = min(options.max_bytes, workspace_page_limit_bytes)
    if exact_eof:
        page, next_offset = b"", None
    else:
        page_or_error = _utf8_workspace_page(
            result,
            path=path,
            content_type=content_type,
            requested_offset=offset,
            max_bytes=effective_page_limit_bytes,
            caller_max_bytes=options.max_bytes,
            effective_read_limit_bytes=effective_read_limit,
            workspace_page_limit_bytes=workspace_page_limit_bytes,
            raw_read_limit_exhausted=raw_read_limit_exhausted,
            workspace_limit_exhausted=workspace_limit_exhausted,
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
    page_end = offset + len(page)
    required_window_end = min(
        result.total_bytes,
        page_end + redaction_overlap,
    )
    if not exact_eof and observed_window_end < required_window_end:
        return _secret_redaction_window_unavailable_result(
            offset=offset,
            required_window_end=required_window_end,
            observed_window_end=observed_window_end,
        )
    if exact_eof:
        text, redaction_truncated = "", False
    elif redactor.has_values:
        try:
            text, redaction_truncated = redactor.redact_utf8_page(
                result.content,
                window_offset=result.offset,
                page_offset=offset,
                page_end=page_end,
                max_bytes=options.max_bytes,
                source_complete=next_offset is None,
                preserve_line_breaks=preserve_line_breaks,
            )
        except ValueError:
            if not preserve_line_breaks:
                raise
            return ToolResult(
                content="The active secret scope cannot preserve source line boundaries. Use byte-offset reads.",
                structured={"error": "line_range_redaction_unavailable"},
                is_error=True,
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
    return _WorkspaceText(text=text, structured=structured, capture_snapshot=capture_snapshot)


async def _read_workspace_lines(
    ctx: ToolContext,
    workspace: Workspace,
    *,
    path: str,
    content_type: str,
    line_range: _LineRange,
    options: ReadFileOptions,
    line_cut_hint: str = _LINE_CUT_RETRY_HINT,
) -> _WorkspaceText | ToolResult:
    """Select whole lines from the redacted text of the file's first read window."""

    read = await _read_workspace_text(
        ctx,
        workspace,
        path=path,
        content_type=content_type,
        offset=0,
        options=replace(options, max_bytes=MAX_READ_LIMIT_BYTES),
        preserve_line_breaks=True,
    )
    if isinstance(read, ToolResult):
        return read
    source = read.structured
    complete = not source["truncated"]
    lines = _text_lines(read.text)
    if not complete and lines and not lines[-1].endswith("\n"):
        lines.pop()
    start = line_range.start
    if start > len(lines) and (start > 1 or not complete):
        if complete:
            return invalid_tool_arguments_result(
                ValueError(
                    f"Tool argument `start_line` ({start}) is past the end of the file "
                    f"({len(lines)} lines)."
                )
            )
        return ToolResult(
            content=(
                f"Line ranges cover the first {MAX_READ_LIMIT_BYTES} bytes of a file, which end "
                f"at line {len(lines)}. Use `offset` to read further."
            ),
            structured={
                "error": "line_range_beyond_read_window",
                "path": path,
                "start_line": start,
                "readable_lines": len(lines),
            },
            is_error=True,
        )
    stop = len(lines) if line_range.end is None else min(line_range.end, len(lines))
    selected: list[str] = []
    used_bytes = 0
    line_cut = False
    for line in lines[start - 1 : stop]:
        line_bytes = len(line.encode("utf-8"))
        if used_bytes + line_bytes > options.max_bytes:
            if not selected:
                selected.append(
                    line.encode("utf-8")[: options.max_bytes].decode("utf-8", errors="ignore")
                )
                line_cut = True
            break
        selected.append(line)
        used_bytes += line_bytes
    end_line = start - 1 + len(selected)
    next_start_line = end_line if line_cut else end_line + 1 if end_line < stop else None
    window_exhausted = not complete and (line_range.end is None or line_range.end > end_line)
    whole_file = complete and start == 1 and end_line == len(lines) and not line_cut
    text = "".join(selected)
    structured = {
        "source": "workspace",
        "path": path,
        "bytes": len(text.encode("utf-8")),
        "total_bytes": source["total_bytes"],
        "start_line": start,
        "end_line": end_line,
        "total_lines": len(lines) if complete else None,
        "next_start_line": next_start_line,
        "revision": source["revision"] if whole_file else None,
        "sha256": source["sha256"] if whole_file else None,
        "encoding": "utf-8",
        "truncated": next_start_line is not None or line_cut or window_exhausted,
    }
    if line_cut:
        structured["line_truncated"] = True
        structured["continuation_hint"] = line_cut_hint
    if not whole_file:
        record_ambiguous_secret_output(ctx, read.capture_snapshot)
    return _WorkspaceText(text=text, structured=structured, capture_snapshot=read.capture_snapshot)


def _text_lines(text: str) -> list[str]:
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


async def _read_workspace_files(
    ctx: ToolContext,
    *,
    entries: tuple[_ReadFileEntry, ...],
    max_bytes: int,
) -> ToolResult:
    workspace = _require_workspace(ctx)
    if workspace is None:
        return _missing_workspace_result()
    remaining = max_bytes
    sections: list[str] = []
    files: list[dict] = []
    for index, entry in enumerate(entries):
        # Each file may use an equal share of what earlier files left unused.
        budget = -(-remaining // (len(entries) - index))
        read = await _read_workspace_entry(ctx, workspace, entry, max_bytes=budget)
        if isinstance(read, _WorkspaceText):
            remaining -= len(read.text.encode("utf-8"))
            files.append(read.structured)
            sections.append(_model_visible_workspace_text(read.text, read.structured))
            continue
        structured = read.structured or {}
        failure = {
            "path": entry.path,
            "error": (
                structured.get("error")
                or structured.get("reason")
                or ("binary_file" if structured.get("binary") else "read_failed")
            ),
            "message": read.content,
        }
        files.append(failure)
        sections.append(_model_visible_workspace_text(read.content, failure))
    return ToolResult(
        content="\n".join(sections),
        structured={
            "source": "workspace",
            "files": files,
            "max_bytes": max_bytes,
            "bytes": max_bytes - remaining,
            "truncated": any(item.get("truncated") is True for item in files),
            "failed": sum(1 for item in files if "error" in item),
        },
        is_error=all("error" in item for item in files),
    )


async def _read_workspace_entry(
    ctx: ToolContext,
    workspace: Workspace,
    entry: _ReadFileEntry,
    *,
    max_bytes: int,
) -> _WorkspaceText | ToolResult:
    if entry.path_error is not None:
        return invalid_tool_arguments_result(ValueError(entry.path_error))
    if max_bytes == 0:
        return ToolResult(
            content="Not read: earlier files used the whole max_bytes budget.",
            structured={"error": "max_bytes_exhausted"},
            is_error=True,
        )
    content_type = _guess_workspace_content_type(entry.path)
    if _is_workspace_file_attachment_content_type(content_type):
        return ToolResult(
            content="Images and PDFs are read one at a time with `path`.",
            structured={"error": "attachment_requires_single_read"},
            is_error=True,
        )
    return await _read_workspace_window(
        ctx,
        workspace,
        path=entry.path,
        content_type=content_type,
        offset=entry.offset,
        line_range=entry.line_range,
        options=ReadFileOptions(max_bytes=max_bytes),
    )


async def _read_workspace_window(
    ctx: ToolContext,
    workspace: Workspace,
    *,
    path: str,
    content_type: str,
    offset: int,
    line_range: _LineRange | None,
    options: ReadFileOptions,
    line_cut_hint: str = _LINE_CUT_RETRY_HINT,
) -> _WorkspaceText | ToolResult:
    if line_range is None:
        return await _read_workspace_text(
            ctx,
            workspace,
            path=path,
            content_type=content_type,
            offset=offset,
            options=options,
        )
    return await _read_workspace_lines(
        ctx,
        workspace,
        path=path,
        content_type=content_type,
        line_range=line_range,
        options=options,
        line_cut_hint=line_cut_hint,
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
    metadata = {key: structured[key] for key in _MODEL_VISIBLE_READ_METADATA if key in structured}
    encoded = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    return f"[read_file metadata]\n{encoded}\n[/read_file metadata]\n{text}"


_MODEL_VISIBLE_READ_METADATA = (
    "path",
    "error",
    "bytes",
    "total_bytes",
    "offset",
    "next_offset",
    "start_line",
    "end_line",
    "total_lines",
    "next_start_line",
    "line_truncated",
    "continuation_hint",
    "revision",
    "sha256",
    "truncated",
    "note",
)


def _utf8_workspace_page(
    result: WorkspaceReadResult,
    *,
    path: str,
    content_type: str,
    requested_offset: int,
    max_bytes: int,
    caller_max_bytes: int,
    effective_read_limit_bytes: int,
    workspace_page_limit_bytes: int,
    raw_read_limit_exhausted: bool,
    workspace_limit_exhausted: bool,
) -> tuple[bytes, int | None] | ToolResult:
    if requested_offset > result.total_bytes:
        return _invalid_workspace_read_offset_result(
            offset=requested_offset,
            total_bytes=result.total_bytes,
        )
    start = requested_offset - result.offset
    available = result.content[start:]
    page = available[:max_bytes]
    caller_page_limit_exhausted = max_bytes == caller_max_bytes and (
        len(available) > caller_max_bytes or raw_read_limit_exhausted
    )
    next_scalar_width = _utf8_scalar_width(available[0]) if available else None
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
        if (
            workspace_limit_exhausted
            and next_scalar_width is not None
            and workspace_page_limit_bytes < next_scalar_width
        ):
            return _workspace_utf8_page_limit_result(
                requested_offset=requested_offset,
                effective_read_limit_bytes=effective_read_limit_bytes,
                effective_page_limit_bytes=workspace_page_limit_bytes,
            )
        if caller_max_bytes < 4 and caller_page_limit_exhausted:
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
        if workspace_limit_exhausted:
            return _workspace_utf8_page_limit_result(
                requested_offset=requested_offset,
                effective_read_limit_bytes=effective_read_limit_bytes,
                effective_page_limit_bytes=workspace_page_limit_bytes,
            )
        return ToolResult(
            content="The workspace returned an incomplete UTF-8 character.",
            structured={
                "error": "text_page_too_small",
                "offset": requested_offset,
            },
            is_error=True,
        )
    end = requested_offset + len(page)
    return page, end if end < result.total_bytes else None


def _utf8_scalar_width(first_byte: int) -> int | None:
    if first_byte <= 0x7F:
        return 1
    if 0xC2 <= first_byte <= 0xDF:
        return 2
    if 0xE0 <= first_byte <= 0xEF:
        return 3
    if 0xF0 <= first_byte <= 0xF4:
        return 4
    return None


def _workspace_utf8_page_limit_result(
    *,
    requested_offset: int,
    effective_read_limit_bytes: int,
    effective_page_limit_bytes: int,
) -> ToolResult:
    return ToolResult(
        content=(
            "The effective workspace read window is too small to contain the next "
            "safely framed UTF-8 character."
        ),
        structured={
            "error": "text_page_too_small",
            "offset": requested_offset,
            "effective_read_limit_bytes": effective_read_limit_bytes,
            "effective_page_limit_bytes": effective_page_limit_bytes,
        },
        is_error=True,
    )


def _invalid_workspace_read_offset_result(*, offset: int, total_bytes: int) -> ToolResult:
    return invalid_tool_arguments_result(
        ValueError(
            f"Tool argument `offset` ({offset}) cannot exceed workspace file size "
            f"({total_bytes} bytes)."
        )
    )


def _guess_workspace_content_type(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


async def _read_workspace_file_attachment(
    ctx: ToolContext,
    *,
    path: str,
    content_type: str,
    artifact_readers: list[ArtifactReader],
    options: ReadFileOptions,
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
        result, effective_source_limit = await _read_full_workspace_attachment(
            ctx,
            path=path,
            content_type=content_type,
            initial_result=initial_result,
            max_source_bytes=source_cap,
        )
    except _WorkspaceFileNotFoundError:
        return _missing_workspace_file_result("Read", path)
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
            reason=(
                "Workspace read was incomplete under the effective native-inspection "
                f"limit (max {effective_source_limit} bytes)."
            ),
            effective_read_limit_bytes=effective_source_limit,
        )

    snapshot = await _put_content_addressed_artifact(
        artifact_store,
        result.content,
        artifact_id=_content_addressed_artifact_id(
            ctx,
            "read_file_workspace_snapshot",
            ctx.workspace_id,
            path,
            content_type,
            _content_hash(result.content),
        ),
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
        options=options,
        missing_as_result=False,
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
) -> tuple[WorkspaceReadResult, int]:
    workspace = _require_workspace(ctx)
    if workspace is None:
        raise RuntimeError("Workspace disappeared while reading workspace file attachment.")
    result, effective_read_limit = await _read_workspace_bytes(
        workspace,
        path,
        max_bytes=max_source_bytes,
    )
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
    return result, effective_read_limit


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
    effective_read_limit_bytes: int | None = None,
) -> ToolResult:
    if reason is None:
        reason = "Use an artifact or custom parser/tool to inspect this file."
    structured = {
        "source": "workspace",
        "path": path,
        "bytes": bytes_read,
        "total_bytes": total_bytes,
        "content_type": content_type,
        "encoding": None,
        "binary": True,
        "inspectable": inspectable,
        "truncated": truncated,
    }
    if effective_read_limit_bytes is not None:
        structured["effective_read_limit_bytes"] = effective_read_limit_bytes
    return ToolResult(
        content=(
            f"Workspace file '{path}' appears to be binary "
            f"({content_type}, {total_bytes} bytes). "
            "read_file does not decode unsupported binary workspace files as text. "
            f"{reason}"
        ),
        structured=structured,
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
    options: ReadFileOptions,
    missing_as_result: bool,
    image_region_error: str | None = None,
) -> ToolResult:
    artifact_store = _require_artifact_store(ctx)
    if artifact_store is None:
        return _missing_artifact_store_result()
    try:
        result = await _read_artifact_store(
            artifact_store,
            artifact_id,
            max_bytes=1 if options.offset else options.max_bytes,
        )
    except InvalidArtifactIdError as exc:
        return invalid_tool_arguments_result(exc)
    except FileNotFoundError:
        if missing_as_result:
            return _missing_artifact_result(artifact_id)
        raise
    artifact = result.metadata
    access_error = _artifact_access_error(ctx, artifact)
    if access_error is not None:
        return access_error
    if options.offset and not _is_text_content_type(artifact.content_type):
        return invalid_tool_arguments_result(
            ValueError("Tool argument `offset` is only valid for text files and artifacts.")
        )
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
        options=options,
        structured=structured,
        _missing_as_result=missing_as_result,
    )
    for reader in artifact_readers:
        if reader.can_read(artifact):
            ignored: tuple[str, ...] = ()
            if isinstance(reader, TextArtifactReader):
                # The image and PDF options can't apply to text; drop them.
                ignored = _ignored_attachment_arguments(options, image_region_error)
                request = replace(request, options=replace(options, image_region=None, pages=None))
            elif image_region_error is not None:
                return invalid_tool_arguments_result(ValueError(image_region_error))
            if options.offset and not isinstance(reader, TextArtifactReader):
                return _artifact_page_error(
                    "artifact_reader_range_unsupported",
                    "This custom artifact reader does not support text offsets. Use a bounded custom inspection tool.",
                )
            try:
                reader_result = await reader.read(request)
                admitted = await _admit_artifact_reader_attachments(request, reader_result)
            except _ArtifactReadNotFound:
                return _missing_artifact_result(artifact_id)
            return _with_ignored_arguments_note(admitted, ignored)
    return ToolResult(
        content=(
            f"Artifact {artifact.id} ({artifact.filename}) is {artifact.content_type} "
            f"and is {artifact.size_bytes} bytes. No built-in reader is available for this "
            "content type. Register a custom tool if this format should be inspectable."
        ),
        structured=structured,
        is_error=True,
    )


async def _admit_artifact_reader_attachments(
    request: ArtifactReadRequest,
    result: ToolResult,
) -> ToolResult:
    attachments = []
    for payload in result.artifacts:
        try:
            attachment = file_attachment_from_payload(payload)
        except (TypeError, ValueError):
            return _attachment_admission_error(
                request,
                "Artifact reader returned an invalid file attachment reference.",
            )
        if attachment is not None:
            attachments.append(attachment)
    for attachment in attachments:
        if attachment.size_bytes > request.options.max_attachment_bytes:
            return _attachment_size_error(
                request,
                size_bytes=attachment.size_bytes,
            )
        try:
            stored = await _read_artifact_store(
                request.artifact_store,
                attachment.artifact_id,
                max_bytes=request.options.max_attachment_bytes,
                missing_as_result=request._missing_as_result,
            )
        except _ArtifactReadNotFound:
            return _attachment_admission_error(
                request,
                "Artifact reader attachment is no longer available.",
            )
        except (InvalidArtifactIdError, TypeError, ValueError):
            return _attachment_admission_error(
                request,
                "Artifact reader attachment could not be validated.",
            )
        if stored.truncated:
            return _attachment_size_error(
                request,
                size_bytes=stored.total_bytes,
            )
        access_error = _artifact_access_error(request.ctx, stored.metadata)
        if access_error is not None:
            return _attachment_admission_error(
                request,
                "Artifact reader attachment is not available in this context.",
            )
        if (
            stored.metadata.size_bytes != attachment.size_bytes
            or stored.metadata.content_type != attachment.content_type
        ):
            return _attachment_admission_error(
                request,
                "Artifact reader attachment changed before admission.",
            )
    return result


def _attachment_size_error(
    request: ArtifactReadRequest,
    *,
    size_bytes: int,
) -> ToolResult:
    return _attachment_admission_error(
        request,
        f"File attachment is {size_bytes} bytes, which exceeds "
        f"max_attachment_bytes={request.options.max_attachment_bytes}.",
    )


def _attachment_admission_error(
    request: ArtifactReadRequest,
    message: str,
) -> ToolResult:
    return ToolResult(
        content=message,
        structured=request.structured,
        is_error=True,
    )


class TextArtifactReader:
    def can_read(self, artifact: ArtifactMetadata) -> bool:
        return _is_text_content_type(artifact.content_type)

    async def read(self, request: ArtifactReadRequest) -> ToolResult:
        # `image_region` and `pages` don't apply to text. read_file drops
        # them before this point and notes it in the result.
        return await _read_text_artifact_page(request)


async def _read_text_artifact_page(request: ArtifactReadRequest) -> ToolResult:
    offset = request.options.offset
    limit = request.options.max_bytes
    if offset > request.artifact.size_bytes:
        return invalid_tool_arguments_result(ValueError("offset exceeds artifact size."))

    async def read_window(redactor):
        overlap = redactor.pagination_overlap_utf8_bytes if redactor.has_values else 0
        start = max(0, offset - overlap - 3) if overlap else offset
        bound = offset - start + limit + overlap + 4
        result = copy_artifact_read_result(
            await request.artifact_store.read_range(
                request.artifact.id, offset=start, max_bytes=bound
            ),
            expected_artifact_id=request.artifact.id,
            max_content_bytes=bound,
            expected_offset=start,
        )
        if result.metadata != request.artifact or result.offset != start:
            raise ValueError("Artifact range identity or offset changed.")
        if result.redaction_truncated or result.source_bytes_read != len(result.content):
            raise NotImplementedError("Artifact ranges must preserve stored byte coordinates.")
        expected = min(bound, result.total_bytes - start)
        if len(result.content) != expected:
            raise ValueError("Artifact store returned an incomplete range.")
        return result

    try:
        captured = await await_revision_stable_secret_output(request.ctx, read_window)
    except NotImplementedError:
        # Existing prefix-only adapters remain usable for complete initial reads.
        if (
            offset == 0
            and not request.initial_read.truncated
            and not request.initial_read.redaction_truncated
        ):
            text, truncated = active_secret_redactor(request.ctx).redact_utf8_head(
                request.initial_read.content,
                max_bytes=limit,
                source_complete=True,
            )
            if not truncated:
                return ToolResult(
                    content=text,
                    structured={
                        **request.structured,
                        "encoding": "utf-8",
                        "offset": 0,
                        "next_offset": None,
                        "truncated": False,
                    },
                )
        return _artifact_page_error(
            "artifact_range_unsupported",
            "This store cannot page artifact text. Configure a range-capable artifact adapter or a bounded custom inspection tool.",
        )
    except FileNotFoundError:
        if request._missing_as_result:
            return _missing_artifact_result(request.artifact.id)
        raise
    if captured is None:
        return unstable_secret_redaction_result()
    result, snapshot = captured
    redactor = snapshot.redactor
    relative_start = offset - result.offset
    available = result.content[relative_start:]
    page = available[:limit]
    if offset and page and page[0] & 0xC0 == 0x80:
        return invalid_tool_arguments_result(
            ValueError("offset splits a UTF-8 character; use the previous next_offset.")
        )
    while page:
        try:
            page.decode("utf-8")
            break
        except UnicodeDecodeError as exc:
            if exc.reason != "unexpected end of data":
                return _artifact_page_error(
                    "artifact_text_encoding_unsupported",
                    "Artifact text is not valid UTF-8. Use a custom reader for this encoding.",
                )
            page = page[: exc.start]
    if not page and offset < result.total_bytes:
        return _artifact_page_error(
            "text_page_too_small",
            "Text page cannot fit the next UTF-8 character; retry this offset with max_bytes of at least 4.",
        )
    end = offset + len(page)
    next_offset = end if end < result.total_bytes else None
    if redactor.has_values and page:
        try:
            text, end = redactor.redact_utf8_page_with_progress(
                result.content,
                window_offset=result.offset,
                page_offset=offset,
                page_end=end,
                total_bytes=result.total_bytes,
                max_bytes=limit,
            )
        except ValueError:
            return _artifact_page_error(
                "artifact_redaction_page_unavailable",
                "Active redaction cannot preserve this text page. Use a bounded custom inspection tool for this secret scope.",
            )
        next_offset = end if end < result.total_bytes else None
    else:
        text = page.decode("utf-8")
    if next_offset is not None or offset:
        record_ambiguous_secret_output(request.ctx, snapshot)
        footer = (
            "\n[read_file "
            + json.dumps(
                {
                    "offset": offset,
                    "next_offset": next_offset,
                    "total_bytes": result.total_bytes,
                    "truncated": next_offset is not None,
                },
                separators=(",", ":"),
            )
            + "]"
        )
        if len(footer.encode()) > _ARTIFACT_PAGE_METADATA_MAX_BYTES - _ARTIFACT_PAGE_NOTE_MAX_BYTES:
            return _artifact_page_error(
                "artifact_size_unsupported", "Artifact size exceeds paging metadata capacity."
            )
        content = text + footer
    else:
        content = text
    return ToolResult(
        content=content,
        structured={
            **request.structured,
            "bytes": end - offset,
            "offset": offset,
            "next_offset": next_offset,
            "encoding": "utf-8",
            "truncated": next_offset is not None,
        },
    )


def _artifact_page_error(reason: str, message: str) -> ToolResult:
    return ToolResult(content=message, structured={"error": reason}, is_error=True)


class ImageArtifactReader:
    def can_read(self, artifact: ArtifactMetadata) -> bool:
        return artifact.content_type in IMAGE_CONTENT_TYPES

    async def read(self, request: ArtifactReadRequest) -> ToolResult:
        if request.options.pages is not None:
            return invalid_tool_arguments_result(
                ValueError("Tool argument `pages` is only valid for PDF artifacts.")
            )
        policy = effective_image_decode_policy(
            ImageDecodePolicy(**request.ctx.image_decode_limits)
            if request.ctx.image_decode_limits is not None
            else None
        )
        if request.options.image_region is not None:
            from cayu.tools._image_regions import read_image_region

            return await read_image_region(request, policy)
        structured = {
            **request.structured,
            "image_decode_limits": policy.as_dict(),
            "semantic_inspection": "not_established_by_tool",
        }
        artifact_store = request.artifact_store
        artifact = request.artifact
        if artifact.size_bytes == 0:
            return ToolResult(
                content=f"Image artifact '{artifact.filename}' is empty and cannot be inspected.",
                structured=structured,
                is_error=True,
            )
        result = await _read_artifact_store(
            artifact_store,
            artifact.id,
            max_bytes=request.options.max_attachment_bytes,
            missing_as_result=request._missing_as_result,
        )
        attachment_artifact = artifact
        if result.truncated:
            source = await _read_artifact_store(
                artifact_store,
                artifact.id,
                max_bytes=MAX_IMAGE_SOURCE_BYTES,
                missing_as_result=request._missing_as_result,
            )
            if source.truncated:
                return ToolResult(
                    content=(
                        f"Image '{artifact.filename}' is too large to inspect "
                        f"({artifact.size_bytes} bytes)."
                    ),
                    structured=structured,
                    is_error=True,
                )
            structured["source_sha256"] = _content_hash(source.content)
            detected_content_type, validation_error = await asyncio.to_thread(
                _detect_image_content_type,
                source.content,
                policy,
            )
            if validation_error is not None:
                return ToolResult(
                    content=(
                        f"Image '{artifact.filename}' could not be inspected: {validation_error}. "
                        "Host configuration: ToolExecutionConfig.image_max_frame_bytes, "
                        "image_max_total_bytes, image_max_frames. After source admission, "
                        "use image_region for readable portions; attachment bytes do not "
                        "change source decoding policy."
                    ),
                    structured=structured,
                    is_error=True,
                )
            if detected_content_type is None:
                return ToolResult(
                    content=f"Image '{artifact.filename}' could not be inspected: unknown image type.",
                    structured=structured,
                    is_error=True,
                )
            if detected_content_type != artifact.content_type:
                return ToolResult(
                    content=(
                        f"Image '{artifact.filename}' content type mismatch: metadata says "
                        f"{artifact.content_type}, but bytes are {detected_content_type}."
                    ),
                    structured=structured,
                    is_error=True,
                )
            derivation_key = _derivation_key(
                source_hash=_content_hash(source.content),
                operation="resize_image",
                params=str(request.options.max_attachment_bytes),
            )
            reused = await _find_derived_artifact(
                artifact_store,
                request.ctx,
                derivation_key,
                max_attachment_validation_bytes=request.options.max_attachment_validation_bytes,
            )
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
                        structured=structured,
                        is_error=True,
                    )
                if resized is None:
                    return ToolResult(
                        content=(
                            f"Image '{artifact.filename}' exceeds max_attachment_bytes="
                            f"{request.options.max_attachment_bytes}. Install cayu[files] or "
                            "register a custom reader to resize this image."
                        ),
                        structured=structured,
                        is_error=True,
                    )
                image_bytes, content_type = resized
                attachment_artifact = await _put_content_addressed_artifact(
                    artifact_store,
                    image_bytes,
                    artifact_id=_content_addressed_artifact_id(
                        request.ctx,
                        "derived_attachment",
                        type(self).__name__,
                        artifact.id,
                        derivation_key,
                        _content_hash(image_bytes),
                    ),
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
            structured["source_sha256"] = _content_hash(result.content)
            detected_content_type, validation_error = await asyncio.to_thread(
                _detect_image_content_type,
                result.content,
                policy,
            )
            if validation_error is not None:
                return ToolResult(
                    content=(
                        f"Image '{artifact.filename}' could not be inspected: {validation_error}. "
                        "Host configuration: ToolExecutionConfig.image_max_frame_bytes, "
                        "image_max_total_bytes, image_max_frames. After source admission, "
                        "use image_region for readable portions; attachment bytes do not "
                        "change source decoding policy."
                    ),
                    structured=structured,
                    is_error=True,
                )
            if detected_content_type is None:
                return ToolResult(
                    content=f"Image '{artifact.filename}' could not be inspected: unknown image type.",
                    structured=structured,
                    is_error=True,
                )
            if detected_content_type != artifact.content_type:
                return ToolResult(
                    content=(
                        f"Image '{artifact.filename}' content type mismatch: metadata says "
                        f"{artifact.content_type}, but bytes are {detected_content_type}."
                    ),
                    structured=structured,
                    is_error=True,
                )
        source_bytes = source.content if result.truncated else result.content
        from cayu.tools._image_regions import image_information

        dimensions, frames = await asyncio.to_thread(image_information, source_bytes)
        structured.update(
            {
                "source_sha256": _content_hash(source_bytes),
                "source_dimensions": dimensions,
                "source_frame_count": frames,
                "source_artifact_id": artifact.id,
                "representation": "original"
                if attachment_artifact.id == artifact.id
                else "resized_overview",
                "region_operation": "read_file(artifact_id=source_artifact_id, image_region=[left, top, right, bottom])",
            }
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
                f"{attachment_artifact.size_bytes} bytes). "
                f"Source artifact {artifact.id}, SHA-256 {structured['source_sha256']}, "
                f"dimensions {dimensions['width']} x {dimensions['height']}, "
                f"representation {structured['representation']}. "
                "For original-resolution portions, read the source artifact with "
                "image_region=[left, top, right, bottom]."
            ),
            structured={
                **structured,
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
        if request.options.image_region is not None:
            return invalid_tool_arguments_result(
                ValueError("image_region is only valid for images.")
            )
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
                missing_as_result=request._missing_as_result,
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
                    missing_as_result=request._missing_as_result,
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
            reused = await _find_derived_artifact(
                artifact_store,
                request.ctx,
                derivation_key,
                max_attachment_validation_bytes=request.options.max_attachment_validation_bytes,
            )
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
                attachment_artifact = await _put_content_addressed_artifact(
                    artifact_store,
                    pdf_bytes,
                    artifact_id=_content_addressed_artifact_id(
                        request.ctx,
                        "derived_attachment",
                        type(self).__name__,
                        artifact.id,
                        derivation_key,
                        _content_hash(pdf_bytes),
                    ),
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
        workspace_mutation=True,
        description=(
            "Atomically edit an existing UTF-8 workspace file using one or more exact "
            "text replacements. Requires the opaque revision from a complete read_file "
            "result or after_revision from a successful edit_file result. Returns the "
            "new after_revision for subsequent edits and refuses stale content, "
            "ambiguous matches, partial edits, and oversized files."
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
                    "description": (
                        "Opaque revision from a complete read_file result or after_revision "
                        "from the latest successful edit_file result for this file."
                    ),
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

    def _execution_profile_material(self) -> dict[str, object]:
        return {}

    def __init__(
        self,
        spec: ToolSpec | None = None,
        *,
        max_terminal_payload_bytes: int | None = None,
    ) -> None:
        super().__init__(
            spec_with_terminal_payload_bound(
                type(self).spec if spec is None else spec,
                max_terminal_payload_bytes,
                default=DEFAULT_TEXT_TOOL_TERMINAL_PAYLOAD_BYTES,
            )
        )

    @bounded_terminal_result
    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        with tool_argument_validation():
            reject_unknown_tool_arguments(args, allowed=_EDIT_FILE_ARGUMENTS)
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
            return _missing_workspace_file_result("Edit", path)
        after_sha256 = mutation.after_sha256
        summary = (
            f"Edited {path}: {len(edits)} edit(s), {replacement_count} replacement(s), "
            f"{len(read.content)} -> {len(encoded)} bytes."
        )
        metadata = json.dumps(
            {"after_revision": mutation.after_revision},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        content = f"[edit_file metadata]\n{metadata}\n[/edit_file metadata]\n{summary}"
        return ToolResult(
            content=f"{content}\n\n{diff_preview}" if diff_preview else content,
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
        workspace_mutation=True,
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

    def _execution_profile_material(self) -> dict[str, object]:
        return {}

    def __init__(
        self,
        spec: ToolSpec | None = None,
        *,
        max_terminal_payload_bytes: int | None = None,
    ) -> None:
        super().__init__(
            spec_with_terminal_payload_bound(
                type(self).spec if spec is None else spec,
                max_terminal_payload_bytes,
                default=DEFAULT_RECORD_TOOL_TERMINAL_PAYLOAD_BYTES,
            )
        )

    @bounded_terminal_result
    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        with tool_argument_validation():
            reject_unknown_tool_arguments(args, allowed=_DELETE_FILE_ARGUMENTS)
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
            return _missing_workspace_file_result("Delete", path)
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
        workspace_mutation=True,
        description=(
            "Create a missing UTF-8 file or conditionally overwrite an existing file. "
            "Overwrite requires the opaque revision returned by a complete read_file "
            "or a successful write_file receipt."
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

    def _execution_profile_material(self) -> dict[str, object]:
        return {}

    def __init__(
        self,
        spec: ToolSpec | None = None,
        *,
        max_terminal_payload_bytes: int | None = None,
    ) -> None:
        super().__init__(
            spec_with_terminal_payload_bound(
                type(self).spec if spec is None else spec,
                max_terminal_payload_bytes,
                default=DEFAULT_RECORD_TOOL_TERMINAL_PAYLOAD_BYTES,
            )
        )

    @bounded_terminal_result
    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        with tool_argument_validation():
            reject_unknown_tool_arguments(args, allowed=_WRITE_FILE_ARGUMENTS)
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
            return _missing_workspace_file_result("Write", path)
        return ToolResult(
            content=(f"Wrote {len(encoded)} bytes to {path}.\nRevision: {mutation.after_revision}"),
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
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": MAX_LIST_OFFSET,
                    "default": 0,
                    "description": "File offset for deterministic continuation.",
                },
                "max_result_bytes": {
                    "type": "integer",
                    "minimum": 4_096,
                    "maximum": MAX_LIST_RESULT_LIMIT_BYTES,
                    "default": DEFAULT_LIST_RESULT_LIMIT_BYTES,
                    "description": "Maximum projected model-facing result size in bytes.",
                },
            },
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        *,
        exclude_directories: Iterable[str] = (),
        max_scan_entries: int = DEFAULT_LIST_SCAN_LIMIT,
        max_terminal_payload_bytes: int | None = None,
    ) -> None:
        super().__init__(
            spec_with_terminal_payload_bound(
                type(self).spec,
                max_terminal_payload_bytes,
                default=DEFAULT_RECORD_TOOL_TERMINAL_PAYLOAD_BYTES,
            )
        )
        self.exclude_directories = _validate_list_exclude_directories(exclude_directories)
        if type(max_scan_entries) is not int or not 1 <= max_scan_entries <= MAX_LIST_SCAN_LIMIT:
            raise ValueError(f"max_scan_entries must be from 1 through {MAX_LIST_SCAN_LIMIT}.")
        self.max_scan_entries = max_scan_entries

    def _execution_profile_material(self) -> dict[str, object]:
        return {
            "exclude_directories": list(self.exclude_directories),
            "max_scan_entries": self.max_scan_entries,
        }

    @bounded_terminal_result
    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        with tool_argument_validation():
            reject_unknown_tool_arguments(
                args,
                allowed=_LIST_FILES_ARGUMENTS,
                hint="Use `pattern`, a workspace-relative glob (for example `src/**`). "
                "`path` is not accepted; choose a glob explicitly.",
            )
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
            offset = _optional_nonnegative_int(args, "offset", default=0)
            if offset > MAX_LIST_OFFSET:
                raise ValueError(f"Tool argument `offset` must be at most {MAX_LIST_OFFSET}.")
            max_result_bytes = _optional_int(
                args,
                "max_result_bytes",
                default=DEFAULT_LIST_RESULT_LIMIT_BYTES,
                maximum=MAX_LIST_RESULT_LIMIT_BYTES,
            )
            if max_result_bytes < 4_096:
                raise ValueError("Tool argument `max_result_bytes` must be at least 4096.")
            # Pages are fitted before next_offset is computed, so the declared
            # terminal bound limits the page rather than cutting it afterwards.
            terminal_bound = self.spec.max_terminal_payload_bytes
            if terminal_bound is not None:
                max_result_bytes = min(
                    max_result_bytes, terminal_bound - TERMINAL_PAYLOAD_FIT_MARGIN_BYTES - 1024
                )
            if offset >= self.max_scan_entries:
                raise ValueError(
                    "Tool argument `offset` must be smaller than the configured scan limit."
                )
            if (
                _minimum_list_file_result_bytes(
                    pattern=pattern,
                    offset=offset,
                    max_result_bytes=max_result_bytes,
                )
                > max_result_bytes
            ):
                raise ValueError("Tool argument `pattern` is too large for `max_result_bytes`.")
        requested_end = min(self.max_scan_entries, offset + limit)
        scan_limit = max(1, requested_end)
        while True:
            result = await workspace.list(pattern, limit=scan_limit)
            visible = tuple(
                path
                for path in result.paths
                if not _path_has_list_excluded_directory(path, self.exclude_directories)
            )
            if (
                len(visible) >= requested_end
                or not result.truncated
                or scan_limit >= self.max_scan_entries
            ):
                break
            scan_limit = min(self.max_scan_entries, max(scan_limit * 2, requested_end))

        candidates = visible[offset:requested_end]
        total_files = len(visible) if not result.truncated else None
        source_has_more = result.truncated or requested_end < len(visible)
        selected = _fit_list_file_result(
            pattern=pattern,
            candidates=candidates,
            offset=offset,
            total_files=total_files,
            source_has_more=source_has_more,
            max_result_bytes=max_result_bytes,
            # Measured as published: a redaction placeholder can be longer
            # than the secret it replaces in a path.
            redactor=active_secret_redactor(ctx),
        )
        next_offset = offset + len(selected)
        truncated_by_limit = len(candidates) == limit and source_has_more
        truncated_by_bytes = len(selected) < len(candidates)
        scan_limit_exhausted = (
            result.truncated and scan_limit >= self.max_scan_entries and next_offset >= len(visible)
        )
        reasons = tuple(
            name
            for name, active in (
                ("limit", truncated_by_limit),
                ("response_bytes", truncated_by_bytes),
                ("scan_limit", scan_limit_exhausted),
            )
            if active
        )
        truncated = bool(reasons)
        continuation = (
            next_offset
            if truncated and not scan_limit_exhausted and (selected or not truncated_by_bytes)
            else None
        )
        result_content = "\n".join(selected) if selected else "No files matched."
        if truncated:
            suffix = (
                "continuation unavailable: scan limit reached"
                if scan_limit_exhausted
                else (
                    "continuation unavailable: response byte limit reached"
                    if continuation is None
                    else f"continue with offset={continuation}"
                )
            )
            result_content = f"{result_content}\n\n[file list truncated; {suffix}]"
        structured = {
            "pattern": pattern,
            "files": list(selected),
            "offset": offset,
            "next_offset": continuation,
            "total_files": total_files,
            "truncated": truncated,
            "truncation_reasons": list(reasons),
            "projected_response_bytes": 0,
            "max_result_bytes": max_result_bytes,
        }
        while True:
            projected_response_bytes = _projected_list_result_bytes(result_content, structured)
            if projected_response_bytes == structured["projected_response_bytes"]:
                break
            structured["projected_response_bytes"] = projected_response_bytes
        return ToolResult(content=result_content, structured=structured)


def _validate_list_exclude_directories(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str | bytes):
        raise TypeError("exclude_directories must be an iterable of directory names.")
    copied = tuple(values)
    validated: list[str] = []
    seen: set[str] = set()
    for value in copied:
        if type(value) is not str:
            raise TypeError("exclude_directories entries must be strings.")
        name = require_unicode_scalar_text(
            require_nonblank(value, "exclude directory"), "exclude directory"
        )
        if name in {".", ".."} or "/" in name or "\\" in name or "\0" in name:
            raise ValueError("exclude_directories entries must be plain directory names.")
        if name not in seen:
            seen.add(name)
            validated.append(name)
    return tuple(sorted(validated))


def _path_has_list_excluded_directory(path: str, excluded: tuple[str, ...]) -> bool:
    return bool(excluded) and any(part in excluded for part in PurePosixPath(path).parts[:-1])


def _fit_list_file_result(
    *,
    pattern: str,
    candidates: tuple[str, ...],
    offset: int,
    total_files: int | None,
    source_has_more: bool,
    max_result_bytes: int,
    redactor: SecretRedactor | None = None,
) -> tuple[str, ...]:
    if redactor is not None and redactor.has_values:
        measured_pattern = redactor.redact_text(pattern)
        measured_candidates = tuple(redactor.redact_text(path) for path in candidates)
    else:
        measured_pattern, measured_candidates = pattern, candidates
    selected: list[str] = []
    for path, measured_path in zip(candidates, measured_candidates, strict=True):
        proposed = (*measured_candidates[: len(selected)], measured_path)
        has_more = source_has_more or len(proposed) < len(candidates)
        worst_case = {
            "pattern": measured_pattern,
            "files": list(proposed),
            "offset": offset,
            "next_offset": offset + len(proposed) if has_more else None,
            "total_files": total_files,
            "truncated": has_more,
            "truncation_reasons": ["limit", "response_bytes", "scan_limit"],
            "projected_response_bytes": MAX_LIST_RESULT_LIMIT_BYTES,
            "max_result_bytes": max_result_bytes,
        }
        content = "\n".join(proposed) or "No files matched."
        if has_more:
            # Reserve the longest terminal suffix so switching from a normal
            # continuation to an exhausted scan ceiling cannot exceed the
            # caller's result-byte bound after selection.
            content += (
                "\n\n[file list truncated; continuation unavailable: response byte limit reached]"
            )
        projected = _projected_list_result_bytes(content, worst_case)
        if projected > max_result_bytes:
            break
        selected.append(path)
    return tuple(selected)


def _minimum_list_file_result_bytes(
    *,
    pattern: str,
    offset: int,
    max_result_bytes: int,
) -> int:
    """Return a conservative projection floor before entering the workspace."""

    structured = {
        "pattern": pattern,
        "files": [],
        "offset": offset,
        "next_offset": None,
        "total_files": None,
        "truncated": True,
        "truncation_reasons": ["limit", "response_bytes", "scan_limit"],
        "projected_response_bytes": MAX_LIST_RESULT_LIMIT_BYTES,
        "max_result_bytes": max_result_bytes,
    }
    return _projected_list_result_bytes(
        "No files matched.\n\n[file list truncated; "
        "continuation unavailable: response byte limit reached]",
        structured,
    )


def _projected_list_result_bytes(content: str, structured: Mapping[str, object]) -> int:
    return len(
        json.dumps(
            {"content": content, "structured": structured},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
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
            "additionalProperties": False,
        },
    )

    def _execution_profile_material(self) -> dict[str, object]:
        return {}

    def __init__(
        self,
        spec: ToolSpec | None = None,
        *,
        max_terminal_payload_bytes: int | None = None,
    ) -> None:
        super().__init__(
            spec_with_terminal_payload_bound(
                type(self).spec if spec is None else spec,
                max_terminal_payload_bytes,
                default=DEFAULT_TEXT_TOOL_TERMINAL_PAYLOAD_BYTES,
            )
        )

    @bounded_terminal_result
    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        with tool_argument_validation():
            reject_unknown_tool_arguments(args, allowed=_LIST_ARTIFACTS_ARGUMENTS)
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
    missing_as_result: bool = False,
) -> ArtifactReadResult:
    try:
        return copy_artifact_read_result(
            await artifact_store.read_bytes(artifact_id, max_bytes=max_bytes),
            expected_artifact_id=artifact_id,
            max_content_bytes=max_bytes,
        )
    except FileNotFoundError as exc:
        if missing_as_result:
            raise _ArtifactReadNotFound from exc
        raise


class _ArtifactReadNotFound(Exception):
    pass


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


def _missing_artifact_result(artifact_id: str) -> ToolResult:
    return ToolResult(
        content="Artifact was not found.",
        structured={"artifact_id": artifact_id, "reason": "not_found"},
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


def _missing_workspace_file_result(operation: str, path: str) -> ToolResult:
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


def _given_arg_string(args: dict, key: str) -> str | None:
    """Return an optional string argument, treating a blank string as left out."""

    value = args.get(key)
    if type(value) is str and not value.strip():
        return None
    return _optional_arg_string(args, key)


_IMAGE_REGION_ERROR = (
    "image_region must be four non-negative integers [left, top, right, bottom] with "
    "right greater than left and bottom greater than top. Leave it out to read the whole image."
)


def _image_region_argument(
    value: object,
) -> tuple[tuple[int, int, int, int] | None, str | None]:
    """Return a usable region, or the error to report if the file is an image or PDF.

    Models that fill in every field send fillers such as ``[0, 0, 0, 0]`` or
    ``[0, 0, 1, 1]``. A region that selects one pixel or none can't help
    anyone inspect an image, so it counts as left out.
    """

    if value is None:
        return None, None
    if not isinstance(value, list):
        return None, _IMAGE_REGION_ERROR
    coordinates = [item for item in value if type(item) is int and item >= 0]
    if len(value) != 4 or len(coordinates) != 4:
        return None, _IMAGE_REGION_ERROR
    left, top, right, bottom = coordinates
    if right < left or bottom < top:
        return None, _IMAGE_REGION_ERROR
    if (right - left) * (bottom - top) <= 1:
        return None, None
    return (left, top, right, bottom), None


def _ignored_attachment_arguments(
    options: ReadFileOptions, image_region_error: str | None
) -> tuple[str, ...]:
    """Name the image and PDF options a text read drops."""

    ignored = []
    if options.image_region is not None or image_region_error is not None:
        ignored.append("image_region")
    if options.pages is not None:
        ignored.append("pages")
    return tuple(ignored)


def _ignored_arguments_fields(ignored: tuple[str, ...]) -> dict[str, object]:
    if not ignored:
        return {}
    # Kept short: on a text artifact page the note shares the paging footer's
    # allowance (_ARTIFACT_PAGE_NOTE_MAX_BYTES).
    return {
        "ignored_arguments": list(ignored),
        "note": f"Text file: ignored {' and '.join(ignored)}.",
    }


def _with_ignored_arguments_note(result: ToolResult, ignored: tuple[str, ...]) -> ToolResult:
    fields = _ignored_arguments_fields(ignored)
    if not fields or result.is_error:
        return result
    return result.model_copy(
        update={
            "content": f"{result.content}{_artifact_page_note(fields['note'])}",
            "structured": {**(result.structured or {}), **fields},
        }
    )


def _artifact_page_note(note: object) -> str:
    return f"\n[read_file note] {note}"


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


def _optional_line_range(args: dict) -> _LineRange | None:
    start = args.get("start_line")
    end = args.get("end_line")
    if start is None and end is None:
        return None
    for key, value in (("start_line", start), ("end_line", end)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"Tool argument `{key}` must be a positive integer.")
    start = 1 if start is None else start
    if end is not None and end < start:
        raise ValueError("Tool argument `end_line` must not be less than `start_line`.")
    return _LineRange(start=start, end=end)


def _read_file_entries(value: object) -> tuple[_ReadFileEntry, ...]:
    if not isinstance(value, list):
        raise ValueError("Tool argument `files` must be an array.")
    if len(value) > MAX_READ_FILES:
        raise ValueError(f"Tool argument `files` must contain at most {MAX_READ_FILES} items.")
    entries: list[_ReadFileEntry] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Tool argument `files` must contain objects.")
        reject_unknown_tool_arguments(
            item,
            allowed=_READ_FILE_ENTRY_ARGUMENTS,
            required=frozenset({"path"}),
        )
        path = _require_arg_string(item, "path")
        offset = _optional_nonnegative_int(item, "offset", default=0)
        line_range = _optional_line_range(item)
        if line_range is not None and offset != 0:
            raise ValueError(
                "Tool argument `offset` cannot be combined with `start_line` or `end_line`."
            )
        # A rejected path fails only its own entry, like a missing file.
        try:
            path = _validate_workspace_path_argument(path)
            path_error = None
        except ValueError as exc:
            path_error = str(exc)
        entries.append(
            _ReadFileEntry(
                path=path,
                offset=offset,
                line_range=line_range,
                path_error=path_error,
                has_explicit_range=any(key in item for key in ("offset", "start_line", "end_line")),
            )
        )
    return tuple(entries)


def _batch_with_single_read_fields(
    entries: tuple[_ReadFileEntry, ...],
    *,
    path: str | None,
    artifact_id: str | None,
    offset: int,
    line_range: _LineRange | None,
) -> tuple[_ReadFileEntry, ...]:
    """Apply single-read fields sent next to `files`, rejecting only real conflicts.

    Models that fill in every field repeat a `files` entry in `path` and copy
    its range to the top level. Policy saw each entry's own path, so `path` is
    accepted only when it names one of them. A top-level range applies to
    entries without their own and must match those that have one.
    """

    if artifact_id is not None:
        raise ValueError(
            "Tool argument `artifact_id` cannot be combined with `files`. Read the artifact "
            "in a separate call."
        )
    if path is not None and not any(_same_workspace_path(path, entry.path) for entry in entries):
        raise ValueError(
            "Tool argument `path` is not one of the `files` entries. List every file to read "
            "in `files` and leave `path` out."
        )
    if offset == 0 and line_range is None:
        return entries
    resolved: list[_ReadFileEntry] = []
    for entry in entries:
        # Authorization sees explicit entry fields override top-level fields,
        # including offset zero. Never reinterpret them as absent after policy.
        if not entry.has_explicit_range:
            entry = replace(entry, offset=offset, line_range=line_range)
        elif (entry.offset, entry.line_range) != (offset, line_range):
            raise ValueError(
                "Top-level `offset`, `start_line` or `end_line` differs from a `files` entry's "
                "own range. Set the range inside each `files` entry and leave the top-level "
                "fields out."
            )
        resolved.append(entry)
    return tuple(resolved)


def _same_workspace_path(left: str, right: str) -> bool:
    return posixpath.normpath(left) == posixpath.normpath(right)


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


def _detect_image_content_type(
    content: bytes, policy: ImageDecodePolicy | None = None
) -> tuple[str | None, str | None]:
    try:
        image_module = import_module("PIL.Image")
    except ImportError:
        return None, "Install cayu[files] or register a custom image reader."

    try:
        detected_format = decode_verified_image_format(image_module, content, policy=policy)
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


def _content_addressed_artifact_id(ctx: ToolContext, *parts: str | None) -> str:
    """Return a stable artifact id for one deterministic tool-produced artifact.

    The id covers the session, agent and environment the artifact is stored
    under plus every input that determines its bytes and immutable metadata, so
    a replayed or repeated call returns the existing artifact instead of a copy.
    """

    material = json.dumps(
        [
            "cayu.read_file.artifact.v1",
            ctx.session_id,
            ctx.agent_name,
            ctx.environment_name,
            *parts,
        ],
        separators=(",", ":"),
    )
    return "art_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


async def _put_content_addressed_artifact(
    artifact_store: ArtifactStore,
    content: bytes,
    *,
    artifact_id: str,
    filename: str,
    content_type: str | None = None,
    scope: ArtifactScope = ArtifactScope.SESSION,
    session_id: str | None = None,
    agent_name: str | None = None,
    environment_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactMetadata:
    """Store at the content-addressed id, or at a fresh id if that id holds other bytes.

    An artifact already at the id with other bytes (corrupted or replaced out of
    band) makes the idempotent write conflict; a fresh copy keeps the read
    working, as before content addressing. Any other store failure propagates,
    so a transient error never writes a second copy.
    """

    async def put(identity: str | None) -> ArtifactMetadata:
        return await artifact_store.put_bytes(
            content,
            artifact_id=identity,
            filename=filename,
            content_type=content_type,
            scope=scope,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata,
        )

    try:
        return await put(artifact_id)
    except ArtifactIdentityConflictError:
        return await put(None)


def _derivation_key(*, source_hash: str, operation: str, params: str) -> str:
    payload = f"{source_hash}\x00{operation}\x00{params}".encode()
    return hashlib.sha256(payload).hexdigest()


async def _find_derived_artifact(
    artifact_store: ArtifactStore,
    ctx: ToolContext,
    derivation_key: str,
    *,
    max_attachment_validation_bytes: int,
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
        if meta.metadata.get("cayu_derivation_key") != derivation_key:
            continue
        if meta.size_bytes > max_attachment_validation_bytes:
            continue
        try:
            stored = await _read_artifact_store(
                artifact_store,
                meta.id,
                max_bytes=max_attachment_validation_bytes,
            )
        except Exception:
            continue
        content_hash = meta.metadata.get("content_hash")
        if (
            not stored.truncated
            and stored.metadata.size_bytes == meta.size_bytes
            and type(content_hash) is str
            and re.fullmatch(r"[0-9a-f]{64}", content_hash)
            and _content_hash(stored.content) == content_hash
        ):
            return stored.metadata
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
