from __future__ import annotations

from base64 import standard_b64encode
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.artifacts._images import decode_verified_image_format
from cayu.artifacts.base import ArtifactReadResult

FILE_ATTACHMENT_TYPE = "cayu.file_attachment.v1"
DEFAULT_MAX_FILE_ATTACHMENT_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_FILE_ATTACHMENTS_PER_REQUEST = 20
DEFAULT_MAX_TOTAL_FILE_ATTACHMENT_BYTES = 32 * 1024 * 1024
RESOLVED_FILE_ATTACHMENTS_OPTION = "cayu_file_attachments"
FILE_ATTACHMENT_IMAGE_CONTENT_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
    }
)
FILE_ATTACHMENT_DOCUMENT_CONTENT_TYPES = frozenset({"application/pdf"})
_IMAGE_FORMAT_CONTENT_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}


class FileAttachmentKind(StrEnum):
    IMAGE = "image"
    DOCUMENT = "document"


class FileAttachment(BaseModel):
    """Provider-neutral model-facing artifact reference.

    This record is JSON-safe and safe to persist in tool results. It does not
    contain file bytes. The runtime resolves it from the active ArtifactStore
    immediately before a provider request.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = FILE_ATTACHMENT_TYPE
    artifact_id: str
    kind: FileAttachmentKind
    filename: str
    content_type: str
    size_bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        if value != FILE_ATTACHMENT_TYPE:
            raise ValueError(f"FileAttachment type must be {FILE_ATTACHMENT_TYPE!r}.")
        return value

    @field_validator("artifact_id", "content_type")
    @classmethod
    def validate_clean_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)

    @field_validator("size_bytes")
    @classmethod
    def validate_size_bytes(cls, value: int, info) -> int:
        if type(value) is not int:
            raise ValueError(f"`{info.field_name}` must be an integer.")
        if value <= 0:
            raise ValueError(f"`{info.field_name}` must be greater than zero.")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @model_validator(mode="after")
    def validate_kind_content_type(self) -> FileAttachment:
        validate_file_attachment_content_type(
            kind=self.kind,
            content_type=self.content_type,
        )
        return self


class ResolvedFileAttachment(BaseModel):
    """Provider-request-only attachment bytes encoded for JSON transport."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    kind: FileAttachmentKind
    filename: str
    content_type: str
    data_base64: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("artifact_id", "content_type")
    @classmethod
    def validate_clean_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("filename", "data_base64")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @model_validator(mode="after")
    def validate_kind_content_type(self) -> ResolvedFileAttachment:
        validate_file_attachment_content_type(
            kind=self.kind,
            content_type=self.content_type,
        )
        return self


def file_attachment(
    *,
    artifact_id: str,
    kind: FileAttachmentKind | str,
    filename: str,
    content_type: str,
    size_bytes: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return FileAttachment(
        artifact_id=artifact_id,
        kind=FileAttachmentKind(kind),
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        metadata={} if metadata is None else metadata,
    ).model_dump(mode="json")


def file_attachment_from_payload(payload: object) -> FileAttachment | None:
    if not isinstance(payload, Mapping):
        return None
    raw_payload: Mapping[Any, Any] = payload
    if raw_payload.get("type") != FILE_ATTACHMENT_TYPE:
        return None
    return FileAttachment.model_validate(dict(raw_payload))


def resolved_file_attachment(
    attachment: FileAttachment,
    result: ArtifactReadResult,
) -> dict[str, Any]:
    if result.truncated:
        raise ValueError(f"Artifact attachment was truncated: {attachment.artifact_id}")
    return ResolvedFileAttachment(
        artifact_id=attachment.artifact_id,
        kind=attachment.kind,
        filename=attachment.filename,
        content_type=attachment.content_type,
        data_base64=standard_b64encode(result.content).decode("ascii"),
        metadata=attachment.metadata,
    ).model_dump(mode="json")


def resolved_file_attachments_from_options(options: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = options.get(RESOLVED_FILE_ATTACHMENTS_OPTION, {})
    if raw is None:
        return {}
    if type(raw) is not dict:
        raise ValueError(f"{RESOLVED_FILE_ATTACHMENTS_OPTION} must be an object.")
    resolved: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        artifact_id = require_clean_nonblank(key, "attachment artifact id")
        if type(value) is not dict:
            raise ValueError("Resolved file attachment entries must be objects.")
        attachment = ResolvedFileAttachment.model_validate(value)
        if attachment.artifact_id != artifact_id:
            raise ValueError("Resolved file attachment id must match its map key.")
        resolved[artifact_id] = attachment.model_dump(mode="json")
    return resolved


def validate_file_attachment_content_type(
    *,
    kind: FileAttachmentKind,
    content_type: str,
) -> None:
    if kind == FileAttachmentKind.IMAGE:
        if content_type not in FILE_ATTACHMENT_IMAGE_CONTENT_TYPES:
            raise ValueError(
                "Image file attachments require one of these content types: "
                f"{', '.join(sorted(FILE_ATTACHMENT_IMAGE_CONTENT_TYPES))}."
            )
        return
    if kind == FileAttachmentKind.DOCUMENT:
        if content_type not in FILE_ATTACHMENT_DOCUMENT_CONTENT_TYPES:
            raise ValueError(
                "Document file attachments require one of these content types: "
                f"{', '.join(sorted(FILE_ATTACHMENT_DOCUMENT_CONTENT_TYPES))}."
            )
        return
    raise ValueError(f"Unsupported file attachment kind: {kind!r}.")


def validate_file_attachment_bytes(
    *,
    kind: FileAttachmentKind,
    content: bytes,
    content_type: str | None = None,
) -> None:
    """Validate that `content` is a parseable image/PDF for `kind`, raising `ValueError` if not.

    Parses the bytes with the optional file dependencies (`cayu[files]`): Pillow for images, pypdf
    for documents. Raises a clear `ValueError` when the bytes are not a valid image/PDF, or when the
    dependencies are not installed — so a caller cannot store a prompt attachment whose bytes a
    provider would later fail to serialize. When `content_type` is given for an image, the format
    detected from the bytes must match it (rejecting e.g. JPEG bytes labeled `image/png`).
    """
    from importlib import import_module
    from io import BytesIO

    if kind == FileAttachmentKind.IMAGE:
        try:
            image_module = import_module("PIL.Image")
        except ImportError as exc:
            raise ValueError(
                "Validating image file attachment bytes requires the optional file dependencies. "
                "Install cayu[files]."
            ) from exc
        try:
            detected_format = decode_verified_image_format(image_module, content)
        except Exception as exc:
            raise ValueError(f"File attachment bytes are not a valid image: {exc}") from exc
        detected_content_type = _IMAGE_FORMAT_CONTENT_TYPES.get(str(detected_format).upper())
        if detected_content_type is None:
            raise ValueError(f"Unsupported image file attachment format: {detected_format}.")
        if content_type is not None and detected_content_type != content_type:
            raise ValueError(
                f"File attachment bytes are {detected_content_type} but the declared content type "
                f"is {content_type}."
            )
        return
    if kind == FileAttachmentKind.DOCUMENT:
        try:
            pypdf = import_module("pypdf")
        except ImportError as exc:
            raise ValueError(
                "Validating document file attachment bytes requires the optional file dependencies. "
                "Install cayu[files]."
            ) from exc
        try:
            reader = pypdf.PdfReader(BytesIO(content))
            page_count = len(reader.pages)
        except Exception as exc:
            raise ValueError(f"File attachment bytes are not a valid PDF: {exc}") from exc
        if page_count == 0:
            raise ValueError("File attachment PDF has no pages.")
        return
    raise ValueError(f"Unsupported file attachment kind: {kind!r}.")
