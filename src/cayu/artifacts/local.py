from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import shutil
from os import PathLike
from pathlib import Path
from typing import Any
from uuid import uuid4

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.artifacts.base import (
    ArtifactListResult,
    ArtifactMetadata,
    ArtifactReadResult,
    ArtifactScope,
    ArtifactStore,
)

_CONTENT_FILE = "content"
_METADATA_FILE = "metadata.json"
_ARTIFACT_ID_PREFIX = "art_"


class LocalArtifactStore(ArtifactStore):
    """Local filesystem implementation of ArtifactStore."""

    def __init__(self, root: str | Path, *, store_id: str | None = None) -> None:
        if not isinstance(root, str | PathLike):
            raise TypeError("LocalArtifactStore root must be a string or Path.")
        root_path = Path(root).expanduser().resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        if not root_path.is_dir():
            raise NotADirectoryError(f"Artifact store root is not a directory: {root_path}")

        if store_id is None:
            self.id = str(root_path)
        else:
            self.id = require_clean_nonblank(store_id, "store_id")
        self.root = root_path

    async def put_bytes(
        self,
        content: bytes,
        *,
        filename: str,
        content_type: str | None = None,
        scope: ArtifactScope = ArtifactScope.SESSION,
        session_id: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactMetadata:
        if type(content) is not bytes:
            raise TypeError("Artifact content must be bytes.")
        filename = require_nonblank(filename, "filename")
        if content_type is None:
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        content_type = require_clean_nonblank(content_type, "content_type")
        scope = _validate_scope(scope)
        session_id = _validate_optional_id(session_id, "session_id")
        agent_name = _validate_optional_id(agent_name, "agent_name")
        environment_name = _validate_optional_id(environment_name, "environment_name")
        _validate_scope_owner(scope, session_id=session_id, environment_name=environment_name)
        copied_metadata = copy_json_value(metadata or {}, "metadata")

        artifact = ArtifactMetadata(
            id=_new_artifact_id(),
            filename=filename,
            content_type=content_type,
            size_bytes=len(content),
            scope=scope,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=copied_metadata,
        )
        await asyncio.to_thread(_write_artifact, self.root, artifact, content)
        return artifact

    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        artifact_id = require_clean_nonblank(artifact_id, "artifact_id")
        limit = _validate_limit(max_bytes, "max_bytes")
        return await asyncio.to_thread(_read_artifact, self.root, artifact_id, limit)

    async def list(
        self,
        *,
        scope: ArtifactScope | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        limit: int | None = None,
    ) -> ArtifactListResult:
        validated_scope = _validate_scope(scope) if scope is not None else None
        session_id = _validate_optional_id(session_id, "session_id")
        agent_name = _validate_optional_id(agent_name, "agent_name")
        environment_name = _validate_optional_id(environment_name, "environment_name")
        validated_limit = _validate_limit(limit, "limit")
        return await asyncio.to_thread(
            _list_artifacts,
            self.root,
            validated_scope,
            session_id,
            agent_name,
            environment_name,
            validated_limit,
        )

    async def delete(self, artifact_id: str) -> None:
        artifact_id = require_clean_nonblank(artifact_id, "artifact_id")
        await asyncio.to_thread(_delete_artifact, self.root, artifact_id)


def _new_artifact_id() -> str:
    return f"{_ARTIFACT_ID_PREFIX}{uuid4().hex}"


def _validate_scope(value: ArtifactScope | str) -> ArtifactScope:
    if isinstance(value, ArtifactScope):
        return value
    if type(value) is str:
        try:
            return ArtifactScope(value)
        except ValueError as exc:
            raise ValueError(f"Unsupported artifact scope: {value!r}") from exc
    raise TypeError("Artifact scope must be an ArtifactScope.")


def _validate_optional_id(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return require_clean_nonblank(value, field_name)


def _validate_scope_owner(
    scope: ArtifactScope,
    *,
    session_id: str | None,
    environment_name: str | None,
) -> None:
    if scope == ArtifactScope.SESSION and session_id is None:
        raise ValueError("Session-scoped artifacts require session_id.")
    if scope == ArtifactScope.ENVIRONMENT and environment_name is None:
        raise ValueError("Environment-scoped artifacts require environment_name.")


def _validate_limit(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"Artifact {field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"Artifact {field_name} must be greater than zero.")
    return value


def _write_artifact(root: Path, artifact: ArtifactMetadata, content: bytes) -> None:
    target = _artifact_dir(root, artifact.id)
    if target.exists():
        raise FileExistsError(f"Artifact already exists: {artifact.id}")
    target.mkdir(parents=False)
    try:
        (target / _CONTENT_FILE).write_bytes(content)
        (target / _METADATA_FILE).write_text(
            json.dumps(artifact.model_dump(mode="json"), sort_keys=True, indent=2),
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def _read_artifact(root: Path, artifact_id: str, max_bytes: int | None) -> ArtifactReadResult:
    target = _artifact_dir(root, artifact_id)
    metadata = _load_metadata(target)
    content_path = target / _CONTENT_FILE
    if not content_path.is_file():
        raise FileNotFoundError(f"Artifact content not found: {artifact_id}")
    if max_bytes is None:
        content = content_path.read_bytes()
        return ArtifactReadResult(
            metadata=metadata,
            content=content,
            total_bytes=len(content),
            truncated=False,
        )
    with content_path.open("rb") as file:
        chunk = file.read(max_bytes + 1)
        total_bytes = os.fstat(file.fileno()).st_size
    content = chunk[:max_bytes]
    total_bytes = max(total_bytes, len(chunk))
    return ArtifactReadResult(
        metadata=metadata,
        content=content,
        total_bytes=total_bytes,
        truncated=total_bytes > len(content),
    )


def _list_artifacts(
    root: Path,
    scope: ArtifactScope | None,
    session_id: str | None,
    agent_name: str | None,
    environment_name: str | None,
    limit: int | None,
) -> ArtifactListResult:
    artifacts: list[ArtifactMetadata] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith(_ARTIFACT_ID_PREFIX):
            continue
        try:
            artifact = _load_metadata(child)
        except (OSError, ValueError):
            continue
        if scope is not None and artifact.scope != scope:
            continue
        if session_id is not None and artifact.session_id != session_id:
            continue
        if agent_name is not None and artifact.agent_name != agent_name:
            continue
        if environment_name is not None and artifact.environment_name != environment_name:
            continue
        artifacts.append(artifact)

    artifacts.sort(key=lambda artifact: artifact.created_at, reverse=True)
    total_count = len(artifacts)
    truncated = limit is not None and total_count > limit
    if limit is not None:
        artifacts = artifacts[:limit]
    return ArtifactListResult(
        artifacts=tuple(artifacts),
        total_count=total_count,
        truncated=truncated,
    )


def _delete_artifact(root: Path, artifact_id: str) -> None:
    target = _artifact_dir(root, artifact_id)
    if not target.exists():
        return
    shutil.rmtree(target)


def _artifact_dir(root: Path, artifact_id: str) -> Path:
    artifact_id = require_clean_nonblank(artifact_id, "artifact_id")
    if not artifact_id.startswith(_ARTIFACT_ID_PREFIX):
        raise ValueError("Artifact id must be a local artifact id.")
    candidate = root / artifact_id
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Artifact id escapes the artifact store root.") from exc
    if resolved.name != artifact_id:
        raise ValueError("Artifact id must identify one artifact directory.")
    return resolved


def _load_metadata(artifact_dir: Path) -> ArtifactMetadata:
    metadata_path = artifact_dir / _METADATA_FILE
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Artifact metadata not found: {artifact_dir.name}")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Artifact metadata is not valid JSON: {artifact_dir.name}") from exc
    artifact = ArtifactMetadata.model_validate(payload)
    if artifact.id != artifact_dir.name:
        raise ValueError("Artifact metadata id does not match directory name.")
    return artifact
