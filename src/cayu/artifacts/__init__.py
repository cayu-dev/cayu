"""Artifact storage contracts."""

from cayu.artifacts.attachments import (
    DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
    DEFAULT_MAX_FILE_ATTACHMENTS_PER_REQUEST,
    DEFAULT_MAX_TOTAL_FILE_ATTACHMENT_BYTES,
    FILE_ATTACHMENT_DOCUMENT_CONTENT_TYPES,
    FILE_ATTACHMENT_IMAGE_CONTENT_TYPES,
    FILE_ATTACHMENT_TYPE,
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    FileAttachment,
    FileAttachmentKind,
    ResolvedFileAttachment,
    file_attachment,
    file_attachment_from_payload,
    resolved_file_attachment,
    resolved_file_attachments_from_options,
)
from cayu.artifacts.base import (
    ArtifactListResult,
    ArtifactMetadata,
    ArtifactReadResult,
    ArtifactScope,
    ArtifactStore,
)
from cayu.artifacts.local import LocalArtifactStore
from cayu.artifacts.workspace import (
    DEFAULT_ARTIFACT_WORKSPACE_COPY_LIMIT_BYTES,
    ArtifactToWorkspaceResult,
    WorkspaceToArtifactResult,
    copy_artifact_to_workspace,
    copy_workspace_file_to_artifact,
)

__all__ = [
    "DEFAULT_ARTIFACT_WORKSPACE_COPY_LIMIT_BYTES",
    "DEFAULT_MAX_FILE_ATTACHMENTS_PER_REQUEST",
    "DEFAULT_MAX_FILE_ATTACHMENT_BYTES",
    "DEFAULT_MAX_TOTAL_FILE_ATTACHMENT_BYTES",
    "FILE_ATTACHMENT_DOCUMENT_CONTENT_TYPES",
    "FILE_ATTACHMENT_IMAGE_CONTENT_TYPES",
    "FILE_ATTACHMENT_TYPE",
    "RESOLVED_FILE_ATTACHMENTS_OPTION",
    "ArtifactListResult",
    "ArtifactMetadata",
    "ArtifactReadResult",
    "ArtifactScope",
    "ArtifactStore",
    "ArtifactToWorkspaceResult",
    "FileAttachment",
    "FileAttachmentKind",
    "LocalArtifactStore",
    "ResolvedFileAttachment",
    "WorkspaceToArtifactResult",
    "copy_artifact_to_workspace",
    "copy_workspace_file_to_artifact",
    "file_attachment",
    "file_attachment_from_payload",
    "resolved_file_attachment",
    "resolved_file_attachments_from_options",
]
