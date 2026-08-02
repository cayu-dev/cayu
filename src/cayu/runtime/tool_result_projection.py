from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, TypeGuard, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import (
    copy_durable_json_value,
    require_clean_nonblank,
    require_durable_text,
    require_unicode_scalar_text,
)
from cayu.artifacts import ArtifactMetadata, ArtifactScope, ArtifactStore
from cayu.core.tools import ToolResult
from cayu.tools.files import _READ_FILE_TRUNCATION_MARKER

TOOL_RESULT_ARTIFACT_TYPE = "cayu.tool_result_artifact.v1"
TOOL_RESULT_PROJECTION_AUTHORITY = "cayu.tool_result_projection.v1"
_TOOL_RESULT_PROJECTION_AUTHORITY_FIELD = "projection_authority"
_TOOL_RESULT_PROJECTION_PROVENANCE_PATH = ("tool_result_projection", "policy_id")
ARTIFACT_EXTERNALIZING_TOOL_RESULT_POLICY_ID = "cayu.artifact_externalizing_tool_result.v1"
TOOL_RESULT_TOKEN_ESTIMATION_METHOD = "unicode_codepoints_divided_by_4_ceiling_v1"
DEFAULT_TOOL_RESULT_MAX_INLINE_BYTES = 32 * 1024
DEFAULT_TOOL_RESULT_MAX_INLINE_TOKEN_ESTIMATE = 8 * 1024
DEFAULT_TOOL_RESULT_PREVIEW_BYTES = 2 * 1024
DEFAULT_TOOL_RESULT_ESTIMATE_CHARS_PER_TOKEN = 4
MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES = 64 * 1024
MAX_TOOL_RESULT_PREVIEW_BYTES = MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES - (4 * 1024)
MAX_TOOL_RESULT_ARTIFACT_REFERENCE_BYTES = 4 * 1024
_PROJECTION_FAILURE_CONTENT_MAX_BYTES = 1024
_ARTIFACT_STORE_ID_MAX_BYTES = 256
_TOOL_RESULT_PROJECTION_REFERENCE_MARKER = "\n[cayu tool-result projection reference]\n"
_MAX_GENERATED_READBACK_BYTES = DEFAULT_TOOL_RESULT_MAX_INLINE_BYTES
_UNAVAILABLE_BOUNDARY_TOKEN_ESTIMATION_METHOD = "unavailable_after_boundary_redaction_v1"
_BUILTIN_TOOL_RESULT_ARTIFACT_REFERENCE_FIELDS = frozenset(
    {
        "type",
        "artifact_id",
        "store_id",
        "filename",
        "content_type",
        "size_bytes",
        "sha256",
        "scope",
        "session_id_sha256",
        "readback_max_bytes",
        _TOOL_RESULT_PROJECTION_AUTHORITY_FIELD,
    }
)
_TOOL_RESULT_PROJECTION_PUBLIC_STRUCTURE_KEYS = _BUILTIN_TOOL_RESULT_ARTIFACT_REFERENCE_FIELDS | {
    "artifact_sha256",
    "failure_type",
    "logical_identity_sha256",
    "original_bytes",
    "original_token_estimate",
    "policy_id",
    "projected_bytes",
    "projected_token_estimate",
    "schema_version",
    "status",
    "token_estimation_method",
    "tool_call_id_sha256",
    "tool_result_projection",
}


class ToolResultProjectionStatus(StrEnum):
    UNCHANGED = "unchanged"
    EXTERNALIZED = "externalized"
    FAILED = "failed"


_TOOL_RESULT_PROJECTION_CONTENT_SUPPRESSION_STATUSES = frozenset(
    {
        ToolResultProjectionStatus.EXTERNALIZED,
        ToolResultProjectionStatus.FAILED,
    }
)


def tool_result_projection_suppresses_result_content(
    projection: object,
) -> TypeGuard[dict[str, Any]]:
    """Return whether observers should prefer bounded projection evidence over result text."""

    if type(projection) is not dict:
        return False
    projection_record = cast("dict[str, Any]", projection)
    return projection_record.get("status") in (_TOOL_RESULT_PROJECTION_CONTENT_SUPPRESSION_STATUSES)


class ToolResultProjectionRecord(BaseModel):
    """Bounded, content-free evidence for one configured projection decision."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    schema_version: Literal[1] = 1
    status: ToolResultProjectionStatus
    policy_id: str
    original_bytes: StrictInt = Field(ge=0)
    projected_bytes: StrictInt = Field(ge=0)
    original_token_estimate: StrictInt = Field(ge=0)
    projected_token_estimate: StrictInt = Field(ge=0)
    token_estimation_method: str
    artifact_id: str | None = None
    artifact_sha256: str | None = None
    logical_identity_sha256: str | None = None
    tool_call_id_sha256: str | None = None
    failure_type: str | None = None

    @field_validator("policy_id", "token_estimation_method")
    @classmethod
    def validate_identity_fields(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        value = require_unicode_scalar_text(value, info.field_name)
        if len(value.encode("utf-8")) > 256:
            raise ValueError(f"`{info.field_name}` must be at most 256 UTF-8 bytes.")
        return value

    @field_validator(
        "artifact_id",
        "artifact_sha256",
        "logical_identity_sha256",
        "tool_call_id_sha256",
        "failure_type",
    )
    @classmethod
    def validate_optional_identity_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        value = require_clean_nonblank(value, info.field_name)
        value = require_unicode_scalar_text(value, info.field_name)
        if len(value.encode("utf-8")) > 256:
            raise ValueError(f"`{info.field_name}` must be at most 256 UTF-8 bytes.")
        return value

    @model_validator(mode="after")
    def validate_status_fields(self) -> ToolResultProjectionRecord:
        artifact_fields = (
            self.artifact_id,
            self.artifact_sha256,
            self.logical_identity_sha256,
            self.tool_call_id_sha256,
        )
        if self.status is ToolResultProjectionStatus.EXTERNALIZED:
            if any(value is None for value in artifact_fields):
                raise ValueError("Externalized projection records require artifact identity.")
            if self.failure_type is not None:
                raise ValueError("Externalized projection records cannot include failure_type.")
        elif any(value is not None for value in artifact_fields):
            raise ValueError("Only externalized projection records can include artifact identity.")
        if self.status is ToolResultProjectionStatus.FAILED:
            if self.failure_type is None:
                raise ValueError("Failed projection records require failure_type.")
        elif self.failure_type is not None:
            raise ValueError("Only failed projection records can include failure_type.")
        return self


@dataclass(frozen=True, slots=True)
class ToolResultProjection:
    result: ToolResult
    record: ToolResultProjectionRecord

    def __post_init__(self) -> None:
        if type(self.result) is not ToolResult:
            raise TypeError("ToolResultProjection result must be a ToolResult.")
        if type(self.record) is not ToolResultProjectionRecord:
            raise TypeError("ToolResultProjection record must be a ToolResultProjectionRecord.")
        copied_result = ToolResult(**self.result.model_dump(mode="python"))
        copied_record = ToolResultProjectionRecord.model_validate(
            self.record.model_dump(mode="json")
        )
        projected_bytes = len(copied_result.content.encode("utf-8"))
        if copied_record.projected_bytes != projected_bytes:
            raise ValueError(
                "ToolResultProjection projected_bytes must match the projected content."
            )
        if projected_bytes > MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES:
            raise ValueError("Projected tool-result content exceeds the runtime safety bound.")
        object.__setattr__(self, "result", copied_result)
        object.__setattr__(self, "record", copied_record)


@dataclass(frozen=True, slots=True)
class ToolResultProjectionRequest:
    """Final redacted tool result plus the durable identity needed for projection."""

    result: ToolResult
    session_id: str
    agent_name: str
    environment_name: str | None
    tool_call_id: str
    artifact_store: ArtifactStore | None

    def __post_init__(self) -> None:
        if type(self.result) is not ToolResult:
            raise TypeError("ToolResultProjectionRequest result must be a ToolResult.")
        copied_result = ToolResult(**self.result.model_dump(mode="python"))
        object.__setattr__(self, "result", copied_result)
        for field_name in ("session_id", "agent_name", "tool_call_id"):
            value = require_clean_nonblank(getattr(self, field_name), field_name)
            object.__setattr__(
                self,
                field_name,
                require_unicode_scalar_text(value, field_name),
            )
        if self.environment_name is not None:
            environment_name = require_clean_nonblank(
                self.environment_name,
                "environment_name",
            )
            object.__setattr__(
                self,
                "environment_name",
                require_unicode_scalar_text(environment_name, "environment_name"),
            )
        if self.artifact_store is not None and not isinstance(
            self.artifact_store,
            ArtifactStore,
        ):
            raise TypeError("artifact_store must be an ArtifactStore or None.")


class ToolResultProjectionPolicy(ABC):
    """Projects final redacted tool results before durable terminal publication."""

    @property
    @abstractmethod
    def identity(self) -> str:
        """Stable, version-identifiable policy name used in durable evidence."""

    @abstractmethod
    async def project(
        self,
        request: ToolResultProjectionRequest,
    ) -> ToolResultProjection:
        """Return a bounded model-facing result and content-free projection record."""


@dataclass(frozen=True, slots=True, init=False)
class ArtifactExternalizingToolResultPolicy(ToolResultProjectionPolicy):
    """Externalize oversized redacted text through the configured ArtifactStore."""

    max_inline_bytes: int | None
    max_inline_token_estimate: int | None
    preview_bytes: int
    chars_per_token: int

    def __init__(
        self,
        *,
        max_inline_bytes: int | None = DEFAULT_TOOL_RESULT_MAX_INLINE_BYTES,
        max_inline_token_estimate: int | None = (DEFAULT_TOOL_RESULT_MAX_INLINE_TOKEN_ESTIMATE),
        preview_bytes: int = DEFAULT_TOOL_RESULT_PREVIEW_BYTES,
        chars_per_token: int = DEFAULT_TOOL_RESULT_ESTIMATE_CHARS_PER_TOKEN,
    ) -> None:
        resolved_max_inline_bytes = _optional_nonnegative_int(max_inline_bytes, "max_inline_bytes")
        resolved_max_inline_token_estimate = _optional_nonnegative_int(
            max_inline_token_estimate, "max_inline_token_estimate"
        )
        if resolved_max_inline_bytes is None and resolved_max_inline_token_estimate is None:
            raise ValueError(
                "At least one tool-result byte or token-estimate threshold is required."
            )
        resolved_preview_bytes = _bounded_nonnegative_int(
            preview_bytes,
            "preview_bytes",
            maximum=MAX_TOOL_RESULT_PREVIEW_BYTES,
        )
        resolved_chars_per_token = _positive_int(chars_per_token, "chars_per_token")
        marker_bytes = len(_READ_FILE_TRUNCATION_MARKER.encode("utf-8"))
        marker_characters = len(_READ_FILE_TRUNCATION_MARKER)
        if resolved_max_inline_bytes is not None and resolved_max_inline_bytes <= marker_bytes:
            raise ValueError("max_inline_bytes must leave room for a bounded read_file result.")
        if (
            resolved_max_inline_token_estimate is not None
            and resolved_max_inline_token_estimate * resolved_chars_per_token <= marker_characters
        ):
            raise ValueError(
                "max_inline_token_estimate must leave room for a bounded read_file result."
            )
        object.__setattr__(self, "max_inline_bytes", resolved_max_inline_bytes)
        object.__setattr__(
            self,
            "max_inline_token_estimate",
            resolved_max_inline_token_estimate,
        )
        object.__setattr__(self, "preview_bytes", resolved_preview_bytes)
        object.__setattr__(self, "chars_per_token", resolved_chars_per_token)

    @property
    def identity(self) -> str:
        return ARTIFACT_EXTERNALIZING_TOOL_RESULT_POLICY_ID

    @property
    def token_estimation_method(self) -> str:
        if self.chars_per_token == DEFAULT_TOOL_RESULT_ESTIMATE_CHARS_PER_TOKEN:
            return TOOL_RESULT_TOKEN_ESTIMATION_METHOD
        return f"unicode_codepoints_divided_by_{self.chars_per_token}_ceiling_v1"

    @property
    def readback_max_bytes(self) -> int:
        """Largest generated text read guaranteed to stay inline under this policy."""

        capacities = [_MAX_GENERATED_READBACK_BYTES]
        if self.max_inline_bytes is not None:
            capacities.append(
                self.max_inline_bytes - len(_READ_FILE_TRUNCATION_MARKER.encode("utf-8"))
            )
        if self.max_inline_token_estimate is not None:
            capacities.append(
                self.max_inline_token_estimate * self.chars_per_token
                - len(_READ_FILE_TRUNCATION_MARKER)
            )
        return min(capacities)

    async def project(
        self,
        request: ToolResultProjectionRequest,
    ) -> ToolResultProjection:
        if type(request) is not ToolResultProjectionRequest:
            raise TypeError(
                "ArtifactExternalizingToolResultPolicy requires a ToolResultProjectionRequest."
            )
        result = request.result
        original_bytes = len(result.content.encode("utf-8"))
        original_tokens = self.estimate_tokens(result.content)
        if not result.content or not self._exceeds_threshold(
            byte_count=original_bytes,
            token_estimate=original_tokens,
        ):
            return ToolResultProjection(
                result=result,
                record=self._record(
                    status=ToolResultProjectionStatus.UNCHANGED,
                    original_bytes=original_bytes,
                    projected_result=result,
                    original_token_estimate=original_tokens,
                ),
            )

        content = result.content.encode("utf-8")
        content_sha256 = hashlib.sha256(content).hexdigest()
        logical_identity_sha256 = _logical_identity_sha256(
            session_id=request.session_id,
            tool_call_id=request.tool_call_id,
            content_sha256=content_sha256,
        )
        tool_call_id_sha256 = hashlib.sha256(request.tool_call_id.encode("utf-8")).hexdigest()
        session_id_sha256 = hashlib.sha256(request.session_id.encode("utf-8")).hexdigest()
        artifact_id = f"art_{logical_identity_sha256[:32]}"
        filename = f"tool-result-{artifact_id}.txt"
        projected_content = _externalized_content(
            result.content,
            preview_bytes=self.preview_bytes,
            artifact_id=artifact_id,
            readback_max_bytes=self.readback_max_bytes,
            original_bytes=original_bytes,
            original_token_estimate=original_tokens,
            token_estimation_method=self.token_estimation_method,
        )
        artifact_store = request.artifact_store
        if artifact_store is None:
            return self._failed_projection(
                request=request,
                original_bytes=original_bytes,
                original_token_estimate=original_tokens,
                failure_type="artifact_store_missing",
            )

        metadata_payload = {
            "type": TOOL_RESULT_ARTIFACT_TYPE,
            "logical_identity_sha256": logical_identity_sha256,
            "sha256": content_sha256,
            "policy_id": self.identity,
            "tool_call_id_sha256": tool_call_id_sha256,
        }
        try:
            store_id = require_unicode_scalar_text(
                require_clean_nonblank(artifact_store.id, "artifact_store.id"),
                "artifact_store.id",
            )
            if len(store_id.encode("utf-8")) > _ARTIFACT_STORE_ID_MAX_BYTES:
                raise ValueError(
                    "artifact_store.id exceeds the tool-result reference safety bound."
                )
            artifact = await artifact_store.put_bytes(
                content,
                artifact_id=artifact_id,
                filename=filename,
                content_type="text/plain; charset=utf-8",
                scope=ArtifactScope.SESSION,
                session_id=request.session_id,
                agent_name=request.agent_name,
                environment_name=request.environment_name,
                metadata=metadata_payload,
            )
            artifact = _validate_persisted_artifact(
                artifact,
                artifact_id=artifact_id,
                filename=filename,
                content_size=original_bytes,
                session_id=request.session_id,
                agent_name=request.agent_name,
                environment_name=request.environment_name,
                metadata=metadata_payload,
            )
        except Exception as exc:
            return self._failed_projection(
                request=request,
                original_bytes=original_bytes,
                original_token_estimate=original_tokens,
                failure_type=safe_projection_failure_type(
                    exc,
                    fallback="artifact_store_failure",
                ),
            )

        reference = {
            "type": TOOL_RESULT_ARTIFACT_TYPE,
            "artifact_id": artifact.id,
            "store_id": store_id,
            "filename": artifact.filename,
            "content_type": artifact.content_type,
            "size_bytes": artifact.size_bytes,
            "sha256": content_sha256,
            "scope": artifact.scope.value,
            "session_id_sha256": session_id_sha256,
            "readback_max_bytes": self.readback_max_bytes,
            _TOOL_RESULT_PROJECTION_AUTHORITY_FIELD: TOOL_RESULT_PROJECTION_AUTHORITY,
        }
        reference = _validate_builtin_tool_result_artifact_reference(reference)
        projected_result = ToolResult(
            content=projected_content,
            structured=result.structured,
            artifacts=[*result.artifacts, reference],
            is_error=result.is_error,
        )
        return ToolResultProjection(
            result=projected_result,
            record=self._record(
                status=ToolResultProjectionStatus.EXTERNALIZED,
                original_bytes=original_bytes,
                projected_result=projected_result,
                original_token_estimate=original_tokens,
                artifact_id=artifact.id,
                artifact_sha256=content_sha256,
                logical_identity_sha256=logical_identity_sha256,
                tool_call_id_sha256=tool_call_id_sha256,
            ),
        )

    def estimate_tokens(self, text: str) -> int:
        text = require_durable_text(text, "text")
        return (len(text) + self.chars_per_token - 1) // self.chars_per_token

    def _exceeds_threshold(
        self,
        *,
        byte_count: int,
        token_estimate: int,
    ) -> bool:
        return (self.max_inline_bytes is not None and byte_count > self.max_inline_bytes) or (
            self.max_inline_token_estimate is not None
            and token_estimate > self.max_inline_token_estimate
        )

    def _failed_projection(
        self,
        *,
        request: ToolResultProjectionRequest,
        original_bytes: int,
        original_token_estimate: int,
        failure_type: str,
    ) -> ToolResultProjection:
        projected_result = ToolResult(
            content=_projection_failure_content(self.identity),
            structured=request.result.structured,
            artifacts=request.result.artifacts,
            is_error=request.result.is_error,
        )
        return ToolResultProjection(
            result=projected_result,
            record=self._record(
                status=ToolResultProjectionStatus.FAILED,
                original_bytes=original_bytes,
                projected_result=projected_result,
                original_token_estimate=original_token_estimate,
                failure_type=failure_type,
            ),
        )

    def _record(
        self,
        *,
        status: ToolResultProjectionStatus,
        original_bytes: int,
        projected_result: ToolResult,
        original_token_estimate: int,
        artifact_id: str | None = None,
        artifact_sha256: str | None = None,
        logical_identity_sha256: str | None = None,
        tool_call_id_sha256: str | None = None,
        failure_type: str | None = None,
    ) -> ToolResultProjectionRecord:
        return ToolResultProjectionRecord(
            status=status,
            policy_id=self.identity,
            original_bytes=original_bytes,
            projected_bytes=len(projected_result.content.encode("utf-8")),
            original_token_estimate=original_token_estimate,
            projected_token_estimate=self.estimate_tokens(projected_result.content),
            token_estimation_method=self.token_estimation_method,
            artifact_id=artifact_id,
            artifact_sha256=artifact_sha256,
            logical_identity_sha256=logical_identity_sha256,
            tool_call_id_sha256=tool_call_id_sha256,
            failure_type=failure_type,
        )


def copy_tool_result_projection_policy(
    policy: ToolResultProjectionPolicy | None,
) -> ToolResultProjectionPolicy | None:
    """Validate a policy collaborator; the built-in policy is immutable."""

    if policy is None:
        return None
    if not isinstance(policy, ToolResultProjectionPolicy):
        raise TypeError("tool_result_projection_policy must be a ToolResultProjectionPolicy.")
    identity = require_clean_nonblank(
        policy.identity,
        "tool_result_projection_policy.identity",
    )
    require_unicode_scalar_text(identity, "tool_result_projection_policy.identity")
    if len(identity.encode("utf-8")) > 256:
        raise ValueError("tool_result_projection_policy.identity must be at most 256 UTF-8 bytes.")
    return policy


def projection_failure(
    *,
    policy: ToolResultProjectionPolicy,
    request: ToolResultProjectionRequest,
    failure_type: str,
) -> ToolResultProjection:
    """Build the runtime fallback when a custom policy raises or returns invalid data."""

    validated_policy = copy_tool_result_projection_policy(policy)
    if validated_policy is None:
        raise AssertionError("Projection failure requires a policy.")
    original_bytes = len(request.result.content.encode("utf-8"))
    projected_result = ToolResult(
        content=_projection_failure_content(validated_policy.identity),
        structured=request.result.structured,
        artifacts=request.result.artifacts,
        is_error=request.result.is_error,
    )
    return ToolResultProjection(
        result=projected_result,
        record=ToolResultProjectionRecord(
            status=ToolResultProjectionStatus.FAILED,
            policy_id=validated_policy.identity,
            original_bytes=original_bytes,
            projected_bytes=len(projected_result.content.encode("utf-8")),
            original_token_estimate=0,
            projected_token_estimate=0,
            token_estimation_method="unavailable_after_policy_failure_v1",
            failure_type=require_clean_nonblank(failure_type, "failure_type"),
        ),
    )


def validate_tool_result_projection(
    projection: ToolResultProjection,
    *,
    request: ToolResultProjectionRequest,
    policy: ToolResultProjectionPolicy,
) -> ToolResultProjection:
    """Copy and validate a policy result at the runtime trust boundary."""

    if type(projection) is not ToolResultProjection:
        raise TypeError("Tool-result projection policies must return ToolResultProjection.")
    copied = ToolResultProjection(result=projection.result, record=projection.record)
    if copied.record.policy_id != policy.identity:
        raise ValueError("Tool-result projection record has a conflicting policy identity.")
    if copied.record.original_bytes != len(request.result.content.encode("utf-8")):
        raise ValueError("Tool-result projection record has a conflicting original byte count.")
    if copied.result.structured != request.result.structured:
        raise ValueError("Tool-result projection cannot replace structured result data.")
    if copied.result.is_error is not request.result.is_error:
        raise ValueError("Tool-result projection cannot replace the error disposition.")
    if copied.result.artifacts[: len(request.result.artifacts)] != request.result.artifacts:
        raise ValueError("Tool-result projection cannot replace existing artifact references.")
    if copied.record.status is ToolResultProjectionStatus.UNCHANGED:
        if copied.result != request.result:
            raise ValueError("Unchanged tool-result projection must preserve the complete result.")
    elif (
        copied.record.status is ToolResultProjectionStatus.FAILED
        and copied.result.artifacts != request.result.artifacts
    ):
        raise ValueError("Failed tool-result projection cannot publish a new artifact reference.")
    elif copied.record.status is ToolResultProjectionStatus.EXTERNALIZED:
        _validate_externalized_projection_reference(
            copied,
            artifact_start=len(request.result.artifacts),
        )
    return copied


def _validate_externalized_projection_reference(
    projection: ToolResultProjection,
    *,
    artifact_start: int = 0,
) -> dict[str, Any]:
    if projection.record.status is not ToolResultProjectionStatus.EXTERNALIZED:
        raise ValueError("Projection is not externalized.")
    references = [
        item
        for item in projection.result.artifacts[artifact_start:]
        if item.get("type") == TOOL_RESULT_ARTIFACT_TYPE
        and item.get("artifact_id") == projection.record.artifact_id
        and item.get("sha256") == projection.record.artifact_sha256
    ]
    if len(references) != 1:
        raise ValueError(
            "Externalized tool-result projection must retain its typed artifact reference."
        )
    return _validate_builtin_tool_result_artifact_reference(references[0])


def _logical_identity_sha256(
    *,
    session_id: str,
    tool_call_id: str,
    content_sha256: str,
) -> str:
    payload = json.dumps(
        {
            "schema_version": 1,
            "session_id": session_id,
            "tool_call_id": tool_call_id,
            "content_sha256": content_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_persisted_artifact(
    artifact: ArtifactMetadata,
    *,
    artifact_id: str,
    filename: str,
    content_size: int,
    session_id: str,
    agent_name: str,
    environment_name: str | None,
    metadata: dict[str, Any],
) -> ArtifactMetadata:
    if type(artifact) is not ArtifactMetadata:
        raise TypeError("ArtifactStore.put_bytes must return ArtifactMetadata.")
    copied = ArtifactMetadata.model_validate(artifact.model_dump(mode="json"))
    if (
        copied.id != artifact_id
        or copied.filename != filename
        or copied.content_type != "text/plain; charset=utf-8"
        or copied.size_bytes != content_size
        or copied.scope is not ArtifactScope.SESSION
        or copied.session_id != session_id
        or copied.agent_name != agent_name
        or copied.environment_name != environment_name
        or copy_durable_json_value(copied.metadata, "artifact.metadata") != metadata
    ):
        raise ValueError("Artifact store returned conflicting tool-result metadata.")
    return copied


def _externalized_content(
    text: str,
    *,
    preview_bytes: int,
    artifact_id: str,
    readback_max_bytes: int,
    original_bytes: int,
    original_token_estimate: int,
    token_estimation_method: str,
) -> str:
    preview = _utf8_prefix(text, preview_bytes)
    preview_section = f"\nPreview:\n{preview}\n" if preview else "\n"
    content = (
        "[tool result externalized]"
        f"{preview_section}"
        "The complete redacted result is stored as the referenced artifact "
        f"({original_bytes} bytes; estimated {original_token_estimate} tokens via "
        f"{token_estimation_method})."
        f"{_tool_result_projection_reference_instruction(artifact_id, readback_max_bytes)}"
    )
    if len(content.encode("utf-8")) > MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES:
        raise ValueError("Configured tool-result preview exceeds the projection safety bound.")
    return content


def redact_tool_result_projection_content(
    content: str,
    *,
    artifact_id: str,
    readback_max_bytes: int,
    redact_text: Callable[[str], str],
) -> str:
    """Re-redact projected text while rebuilding only its runtime-owned reference."""

    content = require_durable_text(content, "content")
    artifact_id = _bounded_reference_text(artifact_id, "artifact_id")
    if not callable(redact_text):
        raise TypeError("redact_text must be callable.")
    prefix, marker, suffix = content.rpartition(_TOOL_RESULT_PROJECTION_REFERENCE_MARKER)
    canonical_instruction = _tool_result_projection_reference_instruction(
        artifact_id,
        readback_max_bytes,
    )
    if not marker or f"{marker}{suffix}" != canonical_instruction:
        prefix = content
    redacted_prefix = redact_text(prefix)
    if type(redacted_prefix) is not str:
        raise TypeError("redact_text must return a string.")
    instruction_bytes = len(canonical_instruction.encode("utf-8"))
    if instruction_bytes > MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES:
        raise AssertionError("Runtime projection reference instruction exceeds its bound.")
    redacted_prefix = _utf8_prefix(
        redacted_prefix,
        MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES - instruction_bytes,
    )
    projected = f"{redacted_prefix}{canonical_instruction}"
    return projected


def reestimate_tool_result_projection_tokens(
    content: str,
    *,
    token_estimation_method: object,
) -> tuple[int, str]:
    """Recompute built-in token evidence after boundary redaction changes content."""

    content = require_durable_text(content, "content")
    prefix = "unicode_codepoints_divided_by_"
    suffix = "_ceiling_v1"
    if type(token_estimation_method) is str and (
        token_estimation_method.startswith(prefix) and token_estimation_method.endswith(suffix)
    ):
        divisor_text = token_estimation_method[len(prefix) : -len(suffix)]
        if divisor_text.isascii() and divisor_text.isdigit():
            divisor = int(divisor_text)
            if divisor > 0:
                return (
                    (len(content) + divisor - 1) // divisor,
                    token_estimation_method,
                )
    return 0, _UNAVAILABLE_BOUNDARY_TOKEN_ESTIMATION_METHOD


def _tool_result_projection_reference_instruction(
    artifact_id: str,
    readback_max_bytes: int,
) -> str:
    readback_max_bytes = _positive_int(readback_max_bytes, "readback_max_bytes")
    return (
        f"{_TOOL_RESULT_PROJECTION_REFERENCE_MARKER}"
        "Use read_file with "
        f'{{"artifact_id":"{artifact_id}","max_bytes":{readback_max_bytes}}} '
        "to inspect the "
        "complete redacted result through the bounded artifact-reading workflow."
    )


def _require_bounded_artifact_reference(reference: dict[str, Any]) -> None:
    encoded = json.dumps(
        reference,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if len(encoded) > MAX_TOOL_RESULT_ARTIFACT_REFERENCE_BYTES:
        raise ValueError("Tool-result artifact reference exceeds the runtime safety bound.")


def _validate_builtin_tool_result_artifact_reference(
    reference: dict[str, Any],
) -> dict[str, Any]:
    copied = copy_durable_json_value(reference, "tool_result_artifact_reference")
    if type(copied) is not dict or set(copied) != _BUILTIN_TOOL_RESULT_ARTIFACT_REFERENCE_FIELDS:
        raise ValueError("Tool-result artifact reference has an invalid shape.")
    if copied["type"] != TOOL_RESULT_ARTIFACT_TYPE:
        raise ValueError("Tool-result artifact reference has an invalid type.")

    artifact_id = _bounded_reference_text(copied["artifact_id"], "artifact_id")
    if (
        not artifact_id.startswith("art_")
        or len(artifact_id) != 36
        or not _is_lower_hex(artifact_id[4:], length=32)
    ):
        raise ValueError("Tool-result artifact reference has an invalid artifact_id.")
    store_id = _bounded_reference_text(copied["store_id"], "store_id")
    if len(store_id.encode("utf-8")) > _ARTIFACT_STORE_ID_MAX_BYTES:
        raise ValueError("Tool-result artifact reference has an oversized store_id.")
    if copied["filename"] != f"tool-result-{artifact_id}.txt":
        raise ValueError("Tool-result artifact reference has an invalid filename.")
    if copied["content_type"] != "text/plain; charset=utf-8":
        raise ValueError("Tool-result artifact reference has an invalid content_type.")
    if type(copied["size_bytes"]) is not int or copied["size_bytes"] < 0:
        raise ValueError("Tool-result artifact reference has an invalid size_bytes.")
    if not _is_lower_hex(copied["sha256"], length=64):
        raise ValueError("Tool-result artifact reference has an invalid sha256.")
    if copied["scope"] != ArtifactScope.SESSION.value:
        raise ValueError("Tool-result artifact reference has an invalid scope.")
    if not _is_lower_hex(copied["session_id_sha256"], length=64):
        raise ValueError("Tool-result artifact reference has an invalid session_id_sha256.")
    if (
        type(copied["readback_max_bytes"]) is not int
        or not 1 <= copied["readback_max_bytes"] <= _MAX_GENERATED_READBACK_BYTES
    ):
        raise ValueError("Tool-result artifact reference has an invalid readback_max_bytes.")
    if copied[_TOOL_RESULT_PROJECTION_AUTHORITY_FIELD] != TOOL_RESULT_PROJECTION_AUTHORITY:
        raise ValueError("Tool-result artifact reference has an invalid projection_authority.")
    _require_bounded_artifact_reference(copied)
    return copied


def _bounded_reference_text(value: Any, field_name: str) -> str:
    value = require_clean_nonblank(value, field_name)
    return require_unicode_scalar_text(value, field_name)


def _is_lower_hex(value: Any, *, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _projection_failure_content(policy_id: str) -> str:
    content = (
        "Cayu could not externalize this oversized tool result through "
        f"{policy_id}. The original content was not added to the durable transcript "
        "or model request. Inspect the configured artifact store before retrying."
    )
    encoded = content.encode("utf-8")
    if len(encoded) <= _PROJECTION_FAILURE_CONTENT_MAX_BYTES:
        return content
    return encoded[:_PROJECTION_FAILURE_CONTENT_MAX_BYTES].decode(
        "utf-8",
        errors="ignore",
    )


def _utf8_prefix(text: str, max_bytes: int) -> str:
    if max_bytes == 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def safe_projection_failure_type(
    error: Exception,
    *,
    fallback: str,
) -> str:
    """Return bounded exception-type evidence without invoking exception hooks."""

    try:
        name = type.__getattribute__(type(error), "__name__")
    except Exception:
        return fallback
    if type(name) is not str or not name.isidentifier():
        return fallback
    return name[:128]


def _optional_nonnegative_int(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, field_name)


def _nonnegative_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to zero.")
    return value


def _bounded_nonnegative_int(
    value: int,
    field_name: str,
    *,
    maximum: int,
) -> int:
    value = _nonnegative_int(value, field_name)
    if value > maximum:
        raise ValueError(f"{field_name} must be less than or equal to {maximum}.")
    return value


def _positive_int(value: int, field_name: str) -> int:
    value = _nonnegative_int(value, field_name)
    if value == 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value
