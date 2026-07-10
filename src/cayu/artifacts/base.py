from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from cayu._validation import (
    copy_json_value,
    freeze_json_value,
    require_clean_nonblank,
    require_nonblank,
    require_unicode_scalar_json,
    require_unicode_scalar_text,
    thaw_json_value,
)

_ARTIFACT_CONTENT_TYPE_MAX_LENGTH = 1024


class ArtifactScope(StrEnum):
    SESSION = "session"
    ENVIRONMENT = "environment"


class InvalidArtifactIdError(ValueError):
    """An artifact identifier is not valid for the selected store."""


class ArtifactStoreUnavailableError(RuntimeError):
    """An artifact store cannot currently complete an operation."""


class ArtifactMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    id: str
    filename: str
    content_type: str = "application/octet-stream"
    size_bytes: int
    scope: ArtifactScope = ArtifactScope.SESSION
    session_id: str | None = None
    agent_name: str | None = None
    environment_name: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        copied = copy_json_value(value, "metadata")
        return require_unicode_scalar_json(copied, "metadata")

    @field_validator("metadata")
    @classmethod
    def freeze_metadata(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_json_value(value)

    @field_serializer("metadata")
    def serialize_metadata(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return thaw_json_value(value)

    @field_validator("id")
    @classmethod
    def validate_clean_nonblank(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        return require_unicode_scalar_text(value, info.field_name)

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
            raise ValueError(f"`{info.field_name}` must not contain control characters.")
        if any(ord(char) > 0x7E for char in value):
            raise ValueError(f"`{info.field_name}` must contain printable ASCII characters only.")
        if len(value) > _ARTIFACT_CONTENT_TYPE_MAX_LENGTH:
            raise ValueError(
                f"`{info.field_name}` must be at most "
                f"{_ARTIFACT_CONTENT_TYPE_MAX_LENGTH} characters."
            )
        return value

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str, info) -> str:
        value = require_nonblank(value, info.field_name)
        return require_unicode_scalar_text(value, info.field_name)

    @field_validator("session_id", "agent_name", "environment_name")
    @classmethod
    def validate_optional_clean_nonblank(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        value = require_clean_nonblank(value, info.field_name)
        return require_unicode_scalar_text(value, info.field_name)

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

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Return an independently validated immutable metadata record."""

        payload = self.model_dump(round_trip=True)
        if update is not None:
            payload.update(update)
        copied = type(self).model_validate(payload)
        fields_set = set(self.model_fields_set)
        if update is not None:
            fields_set.update(update)
        object.__setattr__(copied, "__pydantic_fields_set__", fields_set)
        return copied


@dataclass(frozen=True)
class ArtifactReadResult:
    metadata: ArtifactMetadata
    content: bytes
    total_bytes: int
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.metadata) is not ArtifactMetadata:
            raise TypeError("ArtifactReadResult metadata must be ArtifactMetadata.")
        metadata = ArtifactMetadata.model_validate(self.metadata.model_dump())
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
        if metadata.size_bytes != self.total_bytes:
            raise ValueError("ArtifactReadResult metadata size_bytes must equal total_bytes.")
        object.__setattr__(self, "metadata", metadata)


def copy_artifact_read_result(
    value: ArtifactReadResult,
    *,
    expected_artifact_id: str | None = None,
    max_content_bytes: int | None = None,
) -> ArtifactReadResult:
    """Copy and revalidate a store read at a consumer boundary."""

    if type(value) is not ArtifactReadResult:
        raise TypeError("Artifact store reads must return ArtifactReadResult.")
    if max_content_bytes is not None:
        if type(max_content_bytes) is not int:
            raise TypeError("max_content_bytes must be an integer or None.")
        if max_content_bytes < 0:
            raise ValueError("max_content_bytes must be non-negative.")
    copied = ArtifactReadResult(
        metadata=value.metadata,
        content=value.content,
        total_bytes=value.total_bytes,
        truncated=value.truncated,
    )
    if expected_artifact_id is not None and copied.metadata.id != expected_artifact_id:
        raise ValueError("Artifact store returned metadata for a different artifact id.")
    if max_content_bytes is not None and len(copied.content) > max_content_bytes:
        raise ValueError("Artifact store returned content beyond the requested byte limit.")
    return copied


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
        validated_artifacts = []
        for artifact in artifacts:
            if type(artifact) is not ArtifactMetadata:
                raise TypeError("ArtifactListResult artifact entries must be ArtifactMetadata.")
            validated_artifacts.append(ArtifactMetadata.model_validate(artifact.model_dump()))
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
        object.__setattr__(self, "artifacts", tuple(validated_artifacts))


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
        """Read an artifact by id.

        ``max_bytes`` is a hard materialization boundary. When it is not
        ``None``, implementations must not read into application memory or
        return more than ``max_bytes`` content bytes. They must report the
        artifact's full size in ``total_bytes`` when known and set
        ``truncated=True`` whenever additional bytes exist.

        This guarantee is load-bearing for server and tool callers that use
        ``ArtifactStore`` implementations supplied by applications. A store
        that ignores ``max_bytes`` violates the public contract.

        Implementations should raise ``InvalidArtifactIdError`` when the id is
        syntactically invalid for the store, and ``FileNotFoundError`` when a
        valid id does not exist. They should raise
        ``ArtifactStoreUnavailableError`` for operational backend failures.
        Other validation failures indicate invalid store data rather than a
        malformed caller request.
        """

    @abstractmethod
    async def list(
        self,
        *,
        scope: ArtifactScope | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        limit: int | None = None,
    ) -> ArtifactListResult:
        """List artifact metadata.

        Implementations should raise ``ArtifactStoreUnavailableError`` for
        operational backend failures.
        """

    @abstractmethod
    async def delete(self, artifact_id: str) -> None:
        """Delete an artifact if it exists."""
