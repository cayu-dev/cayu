from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from cayu._validation import copy_json_value, require_nonblank
from cayu.artifacts.base import (
    ArtifactMetadata,
    ArtifactScope,
    ArtifactStore,
    copy_artifact_read_result,
)
from cayu.workspaces import Workspace

DEFAULT_ARTIFACT_WORKSPACE_COPY_LIMIT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ArtifactToWorkspaceResult:
    artifact: ArtifactMetadata
    workspace_path: str
    bytes_written: int
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.artifact) is not ArtifactMetadata:
            raise TypeError("ArtifactToWorkspaceResult artifact must be ArtifactMetadata.")
        if type(self.workspace_path) is not str:
            raise TypeError("ArtifactToWorkspaceResult workspace_path must be a string.")
        if type(self.bytes_written) is not int:
            raise TypeError("ArtifactToWorkspaceResult bytes_written must be an integer.")
        if self.bytes_written < 0:
            raise ValueError("ArtifactToWorkspaceResult bytes_written must be non-negative.")
        if type(self.truncated) is not bool:
            raise TypeError("ArtifactToWorkspaceResult truncated must be a bool.")


@dataclass(frozen=True)
class WorkspaceToArtifactResult:
    artifact: ArtifactMetadata
    workspace_path: str
    bytes_read: int
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.artifact) is not ArtifactMetadata:
            raise TypeError("WorkspaceToArtifactResult artifact must be ArtifactMetadata.")
        if type(self.workspace_path) is not str:
            raise TypeError("WorkspaceToArtifactResult workspace_path must be a string.")
        if type(self.bytes_read) is not int:
            raise TypeError("WorkspaceToArtifactResult bytes_read must be an integer.")
        if self.bytes_read < 0:
            raise ValueError("WorkspaceToArtifactResult bytes_read must be non-negative.")
        if type(self.truncated) is not bool:
            raise TypeError("WorkspaceToArtifactResult truncated must be a bool.")


async def copy_artifact_to_workspace(
    artifact_store: ArtifactStore,
    workspace: Workspace,
    artifact_id: str,
    workspace_path: str,
    *,
    max_bytes: int | None = DEFAULT_ARTIFACT_WORKSPACE_COPY_LIMIT_BYTES,
    allow_truncated: bool = False,
) -> ArtifactToWorkspaceResult:
    """Copy one durable artifact into the mutable workspace.

    The copy is explicit and one-way. The artifact remains unchanged, and later
    workspace edits do not modify the original artifact.
    """

    _validate_artifact_store(artifact_store)
    _validate_workspace(workspace)
    workspace_path = require_nonblank(workspace_path, "workspace_path")
    max_bytes = _validate_optional_positive_int(max_bytes, "max_bytes")
    allow_truncated = _validate_bool(allow_truncated, "allow_truncated")

    result = copy_artifact_read_result(
        await artifact_store.read_bytes(artifact_id, max_bytes=max_bytes),
        expected_artifact_id=artifact_id,
        max_content_bytes=max_bytes,
    )
    if result.truncated and not allow_truncated:
        raise ValueError(
            "Artifact exceeds max_bytes; refusing to write a partial workspace copy. "
            "Increase max_bytes or pass allow_truncated=True."
        )
    await workspace.write_bytes(workspace_path, result.content)
    return ArtifactToWorkspaceResult(
        artifact=result.metadata,
        workspace_path=workspace_path,
        bytes_written=len(result.content),
        truncated=result.truncated,
    )


async def copy_workspace_file_to_artifact(
    workspace: Workspace,
    artifact_store: ArtifactStore,
    workspace_path: str,
    *,
    filename: str | None = None,
    content_type: str | None = None,
    scope: ArtifactScope = ArtifactScope.SESSION,
    session_id: str | None = None,
    agent_name: str | None = None,
    environment_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    max_bytes: int | None = DEFAULT_ARTIFACT_WORKSPACE_COPY_LIMIT_BYTES,
    allow_truncated: bool = False,
) -> WorkspaceToArtifactResult:
    """Store the current workspace file bytes as a new durable artifact."""

    _validate_workspace(workspace)
    _validate_artifact_store(artifact_store)
    workspace_path = require_nonblank(workspace_path, "workspace_path")
    max_bytes = _validate_optional_positive_int(max_bytes, "max_bytes")
    allow_truncated = _validate_bool(allow_truncated, "allow_truncated")
    artifact_filename = _artifact_filename(filename, workspace_path)
    if content_type is None:
        content_type = mimetypes.guess_type(artifact_filename)[0]

    result = await workspace.read_bytes(workspace_path, max_bytes=max_bytes)
    if result.truncated and not allow_truncated:
        raise ValueError(
            "Workspace file exceeds max_bytes; refusing to store a partial artifact. "
            "Increase max_bytes or pass allow_truncated=True."
        )

    artifact_metadata = _workspace_artifact_metadata(
        metadata,
        workspace_id=workspace.id,
        workspace_path=workspace_path,
        truncated=result.truncated,
        total_bytes=result.total_bytes,
    )
    artifact = await artifact_store.put_bytes(
        result.content,
        filename=artifact_filename,
        content_type=content_type,
        scope=scope,
        session_id=session_id,
        agent_name=agent_name,
        environment_name=environment_name,
        metadata=artifact_metadata,
    )
    return WorkspaceToArtifactResult(
        artifact=artifact,
        workspace_path=workspace_path,
        bytes_read=len(result.content),
        truncated=result.truncated,
    )


def _workspace_artifact_metadata(
    metadata: dict[str, Any] | None,
    *,
    workspace_id: str,
    workspace_path: str,
    truncated: bool,
    total_bytes: int,
) -> dict[str, Any]:
    copied = copy_json_value(metadata or {}, "metadata")
    copied.setdefault("source", "workspace")
    copied["source_workspace_id"] = workspace_id
    copied["source_workspace_path"] = workspace_path
    copied["source_workspace_total_bytes"] = total_bytes
    copied["source_workspace_truncated"] = truncated
    copied["operation"] = "copy_workspace_file_to_artifact"
    return copied


def _artifact_filename(filename: str | None, workspace_path: str) -> str:
    if filename is not None:
        return require_nonblank(filename, "filename")
    name = PurePosixPath(workspace_path).name
    if not name:
        raise ValueError("filename is required when workspace_path has no filename.")
    return name


def _validate_artifact_store(value: ArtifactStore) -> None:
    if not isinstance(value, ArtifactStore):
        raise TypeError("artifact_store must implement ArtifactStore.")


def _validate_workspace(value: Workspace) -> None:
    if not isinstance(value, Workspace):
        raise TypeError("workspace must implement Workspace.")


def _validate_optional_positive_int(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


def _validate_bool(value: bool, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a bool.")
    return value
