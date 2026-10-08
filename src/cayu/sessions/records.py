"""Session, event and transcript records independent of storage backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_durable_json_value,
    copy_label_map,
    copy_session_metadata,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.build_provenance import (
    RuntimeBuildProvenance,
    legacy_runtime_build_provenance,
    runtime_build_provenance_identity,
)
from cayu.deadlines import ExecutionDeadline, deadline_from_metadata
from cayu.events import Event, EventType, copy_event
from cayu.messages import Message, copy_message
from cayu.sessions._execution_profile_checkpoint import (
    EXECUTION_PROFILE_METADATA_KEY,
    execution_profile_from_session_metadata,
)
from cayu.sessions.invocation import (
    SessionInvocation,
    SessionInvocationBinding,
    copy_session_invocation,
)
from cayu.tools.exposure import ToolCapabilityCeiling, tool_capability_ceiling_from_session_metadata


class SessionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTING = "interrupting"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


RUNTIME_BUILD_PROVENANCE_METADATA_KEY = "cayu:runtime_build_provenance"


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    # SessionStore implementations set this from RunRequest.session_id or mint it
    # before constructing the record so invocation.root_session_id is exact.
    id: str = Field(default_factory=lambda: str(uuid4()))
    # One store-owned incarnation of ``id``. Deleting and recreating the same
    # public session ID must mint a different value so durable work cannot
    # publish into the replacement session.
    instance_id: str = Field(default_factory=lambda: str(uuid4()), frozen=True)
    agent_name: str
    provider_name: str
    model: str
    parent_session_id: str | None = None
    causal_budget_id: str
    runtime_name: str = "cayu"
    runtime_version: str | None = None
    environment_name: str | None = None
    status: SessionStatus = SessionStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_activity_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_epoch: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    invocation: SessionInvocation = Field(frozen=True)
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def execution_deadline(self) -> ExecutionDeadline:
        return deadline_from_metadata(self.metadata)

    @model_validator(mode="before")
    @classmethod
    def default_causal_budget_id(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            session_id = value.get("id")
            if session_id is None:
                session_id = str(uuid4())
                value["id"] = session_id
            if value.get("causal_budget_id") is None and isinstance(session_id, str):
                value["causal_budget_id"] = session_id
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_session_metadata(value)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator(
        "agent_name",
        "provider_name",
        "model",
        "causal_budget_id",
        "runtime_name",
    )
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        # The byte ceiling is a creation-boundary contract. Session records may
        # predate it or come from an external store, and must remain loadable so
        # operators can inspect and migrate them.
        return require_clean_nonblank(value, info.field_name)

    @field_validator("instance_id")
    @classmethod
    def validate_instance_id(cls, value: str) -> str:
        return SessionInvocationBinding.validate_session_instance_id(value)

    @field_validator("parent_session_id", "environment_name", "runtime_version")
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_profile_build_provenance(self) -> Session:
        if EXECUTION_PROFILE_METADATA_KEY not in self.metadata:
            return self
        profile = execution_profile_from_session_metadata(self.metadata)
        if profile.schema_version >= 6 and (
            profile.runtime_build_provenance
            != runtime_build_provenance_identity(self.runtime_build_provenance)
        ):
            raise ValueError(
                "Session runtime build provenance conflicts with its execution profile."
            )
        return self

    @property
    def tool_capability_ceiling(self) -> ToolCapabilityCeiling:
        """Return the session's required durable application-tool authority."""

        return tool_capability_ceiling_from_session_metadata(self.metadata)

    @property
    def runtime_build_provenance(self) -> RuntimeBuildProvenance:
        """Load exact build provenance without attributing legacy work to this process."""

        return runtime_build_provenance_from_session_metadata(self.metadata)

    @property
    def runtime_build_fingerprint(self) -> str | None:
        return self.runtime_build_provenance.fingerprint

    @property
    def runtime_source_revision(self) -> str | None:
        return self.runtime_build_provenance.source_revision


class EventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    event: Event

    @field_validator("event")
    @classmethod
    def copy_event(cls, value: Event) -> Event:
        return copy_event(value)


@dataclass(frozen=True, slots=True)
class RunnerObservedEventIdentity:
    """Lightweight identity for one event drained by a runner-owned execution.

    Sequence and type are the stable public/durable identity boundary; public
    streams intentionally replace private durable event IDs with authority-safe
    aliases. The eval runner retains this projection instead of full payloads
    while streaming a fresh run, then the store compares it with the complete
    append-only event log inside the bounded snapshot used for evidence.
    """

    session_id: str
    sequence: int | None
    event_type: EventType | str


class TranscriptRecord(BaseModel):
    """One retained message with its zero-based absolute transcript index.

    Physical retention never renumbers this index; it is not a page offset.
    """

    model_config = ConfigDict(extra="forbid")

    index: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    interaction_id: str | None = None
    message: Message

    @field_validator("interaction_id")
    @classmethod
    def validate_interaction_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "interaction_id")

    @field_validator("message")
    @classmethod
    def copy_message(cls, value: Message) -> Message:
        return copy_message(value)


def copy_session(session: Session) -> Session:
    if type(session) is not Session:
        raise TypeError("Session copy requires a Session.")
    return Session(
        id=session.id,
        instance_id=session.instance_id,
        agent_name=session.agent_name,
        provider_name=session.provider_name,
        model=session.model,
        parent_session_id=session.parent_session_id,
        causal_budget_id=session.causal_budget_id,
        runtime_name=session.runtime_name,
        runtime_version=session.runtime_version,
        environment_name=session.environment_name,
        status=session.status,
        created_at=session.created_at,
        updated_at=session.updated_at,
        last_activity_at=session.last_activity_at,
        run_epoch=session.run_epoch,
        invocation=copy_session_invocation(session.invocation),
        labels=copy_label_map(session.labels, "labels"),
        metadata=copy_session_metadata(session.metadata),
    )


def runtime_build_provenance_from_session_metadata(
    metadata: Mapping[str, Any],
) -> RuntimeBuildProvenance:
    """Load bounded build provenance, mapping old rows to explicit legacy state."""

    if not isinstance(metadata, Mapping):
        raise TypeError("Session metadata must be an object.")
    raw = metadata.get(RUNTIME_BUILD_PROVENANCE_METADATA_KEY)
    if raw is None:
        return legacy_runtime_build_provenance()
    try:
        return RuntimeBuildProvenance.model_validate(
            copy_durable_json_value(raw, "runtime_build_provenance")
        )
    except Exception as exc:
        raise ValueError("Session runtime build-provenance metadata is malformed.") from exc


MAX_SESSION_ID_BYTES = 2048


def _require_bounded_session_id(value: str, field_name: str) -> str:
    value = require_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > MAX_SESSION_ID_BYTES:
        raise ValueError(f"`{field_name}` must not exceed {MAX_SESSION_ID_BYTES} UTF-8 bytes.")
    return value


class PendingActionSession(BaseModel):
    """Bounded session identity embedded in pending-action query results."""

    model_config = ConfigDict(extra="forbid")

    id: str
    instance_id: str | None = None
    agent_name: str
    provider_name: str
    model: str
    parent_session_id: str | None = None
    causal_budget_id: str
    runtime_name: str
    runtime_version: str | None = None
    runtime_build_provenance: RuntimeBuildProvenance = Field(
        default_factory=legacy_runtime_build_provenance
    )
    environment_name: str | None = None
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    labels: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_session(cls, session: Session) -> PendingActionSession:
        return cls(
            id=session.id,
            instance_id=session.instance_id,
            agent_name=session.agent_name,
            provider_name=session.provider_name,
            model=session.model,
            parent_session_id=session.parent_session_id,
            causal_budget_id=session.causal_budget_id,
            runtime_name=session.runtime_name,
            runtime_version=session.runtime_version,
            runtime_build_provenance=session.runtime_build_provenance,
            environment_name=session.environment_name,
            status=session.status,
            created_at=session.created_at,
            updated_at=session.updated_at,
            labels=session.labels,
        )

    @field_validator(
        "id",
        "agent_name",
        "provider_name",
        "model",
        "causal_budget_id",
        "runtime_name",
    )
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("instance_id", "parent_session_id", "environment_name", "runtime_version")
    @classmethod
    def validate_optional_nonblank_fields(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("created_at", "updated_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value: dict[str, str]) -> dict[str, str]:
        return copy_label_map(value, "labels")


class PendingActionKind(StrEnum):
    TOOL_APPROVAL = "tool_approval"
    USER_INPUT = "user_input"
    MANUAL_RECOVERY = "manual_recovery"
    DELEGATED_ACTION = "delegated_action"
