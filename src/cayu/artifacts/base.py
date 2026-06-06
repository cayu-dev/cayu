from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank


class ArtifactScope(StrEnum):
    SESSION = "session"
    ENVIRONMENT = "environment"


class ArtifactMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    filename: str
    content_type: str = "application/octet-stream"
    size_bytes: int
    scope: ArtifactScope = ArtifactScope.SESSION
    session_id: str | None = None
    agent_name: str | None = None
    environment_name: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("id", "content_type")
    @classmethod
    def validate_clean_nonblank(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)

    @field_validator("session_id", "agent_name", "environment_name")
    @classmethod
    def validate_optional_clean_nonblank(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("size_bytes")
    @classmethod
    def validate_size_bytes(cls, value: int, info) -> int:
        if type(value) is not int:
            raise ValueError(f"`{info.field_name}` must be an integer.")
        if value < 0:
            raise ValueError(f"`{info.field_name}` must be non-negative.")
        return value

    @model_validator(mode="after")
    def validate_scope_owner(self) -> ArtifactMetadata:
        if self.scope == ArtifactScope.SESSION and self.session_id is None:
            raise ValueError("Session-scoped artifacts require session_id.")
        if self.scope == ArtifactScope.ENVIRONMENT and self.environment_name is None:
            raise ValueError("Environment-scoped artifacts require environment_name.")
        return self


@dataclass(frozen=True)
class ArtifactReadResult:
    metadata: ArtifactMetadata
    content: bytes
    total_bytes: int
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.metadata) is not ArtifactMetadata:
            raise TypeError("ArtifactReadResult metadata must be ArtifactMetadata.")
        if type(self.content) is not bytes:
            raise TypeError("ArtifactReadResult content must be bytes.")
        if type(self.total_bytes) is not int:
            raise TypeError("ArtifactReadResult total_bytes must be an integer.")
        if self.total_bytes < 0:
            raise ValueError("ArtifactReadResult total_bytes must be non-negative.")
        if type(self.truncated) is not bool:
            raise TypeError("ArtifactReadResult truncated must be a bool.")
        if self.total_bytes < len(self.content):
            raise ValueError("ArtifactReadResult total_bytes cannot be smaller than content.")
        expected_truncated = len(self.content) < self.total_bytes
        if self.truncated != expected_truncated:
            raise ValueError("ArtifactReadResult truncated must match content and total_bytes.")


@dataclass(frozen=True)
class ArtifactListResult:
    artifacts: tuple[ArtifactMetadata, ...]
    total_count: int | None
    truncated: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.artifacts, ArtifactMetadata):
            artifacts = (self.artifacts,)
        elif isinstance(self.artifacts, str | bytes):
            raise TypeError("ArtifactListResult artifacts must be an iterable of ArtifactMetadata.")
        else:
            try:
                artifacts = tuple(self.artifacts)
            except TypeError as exc:
                raise TypeError(
                    "ArtifactListResult artifacts must be an iterable of ArtifactMetadata."
                ) from exc
        for artifact in artifacts:
            if type(artifact) is not ArtifactMetadata:
                raise TypeError("ArtifactListResult artifact entries must be ArtifactMetadata.")
        if self.total_count is not None:
            if type(self.total_count) is not int:
                raise TypeError("ArtifactListResult total_count must be an integer.")
            if self.total_count < 0:
                raise ValueError("ArtifactListResult total_count must be non-negative.")
            if self.total_count < len(artifacts):
                raise ValueError("ArtifactListResult total_count cannot be smaller than artifacts.")
        if type(self.truncated) is not bool:
            raise TypeError("ArtifactListResult truncated must be a bool.")
        if not self.truncated and self.total_count is None:
            raise ValueError("ArtifactListResult total_count is required when not truncated.")
        if not self.truncated and self.total_count != len(artifacts):
            raise ValueError(
                "ArtifactListResult total_count must equal artifacts when not truncated."
            )
        object.__setattr__(self, "artifacts", artifacts)


class ArtifactStore(ABC):
    """Durable uploaded/generated file storage for an environment."""

    id: str

    @abstractmethod
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
        """Store bytes and return a durable artifact reference."""

    @abstractmethod
    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        """Read an artifact by id."""

    @abstractmethod
    async def list(
        self,
        *,
        scope: ArtifactScope | None = None,
        session_id: str | None = None,
        environment_name: str | None = None,
        limit: int | None = None,
    ) -> ArtifactListResult:
        """List artifact metadata."""

    @abstractmethod
    async def delete(self, artifact_id: str) -> None:
        """Delete an artifact if it exists."""
