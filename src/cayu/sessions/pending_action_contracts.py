"""Bounded pending-action queries, discovery records and result limits."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    computed_field,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_durable_json_object,
    json_utf8_size_within_limit,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu._validation import require_durable_nonblank as require_nonblank
from cayu.approvals.tools import ToolPolicyEvidence
from cayu.sessions import records as session_record_rules
from cayu.sessions.records import (
    EventRecord,
    PendingActionKind,
    PendingActionSession,
    SessionStatus,
)


class PendingActionIssueCode(StrEnum):
    """Why one pending-action candidate could not be projected safely."""

    SOURCE_TOO_LARGE = "source_too_large"
    SOURCE_TOO_COMPLEX = "source_too_complex"
    SOURCE_INVALID = "source_invalid"


DEFAULT_PENDING_ACTION_RESULT_MAX_BYTES = 2 * 1024 * 1024
MAX_PENDING_ACTION_RESULT_BYTES = 16 * 1024 * 1024
# A model/runtime should never produce hundreds of calls in one tool round. This
# cap is also a storage-safety boundary: SQL stores inspect the count before
# expanding checkpoint call identifiers into rows.
MAX_PENDING_ACTION_TOOL_CALLS = 256
# A normal call contributes one start and one terminal event. Keep enough room
# for repeated crashes and explicit reconciliation while bounding corrupted or
# adversarial evidence independently for every call in the pending round.
MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL = 16


class PendingActionResultTooLarge(RuntimeError):
    """A pending-action page exceeded its caller-selected serialized byte ceiling."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        super().__init__(
            f"Pending-action response exceeds the {max_bytes}-byte result limit. "
            "Request a smaller page or inspect the session directly."
        )


PENDING_ACTION_EVENT_TYPE_VALUES = frozenset(
    {
        "tool.call.approval_requested",
        "session.awaiting_user_input",
        "session.interrupted",
        "session.delegated_action.updated",
        "session.resumed",
        "session.completed",
        "session.failed",
        "tool.call.started",
        "tool.call.completed",
        "tool.call.failed",
        "tool.call.blocked",
        "tool.call.approval_denied",
    }
)


PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES = frozenset(
    {"session.resumed", "session.completed", "session.failed"}
)


class PendingActionQuery(BaseModel):
    """Bounded query for durable control-plane actions blocking a session."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str | None = None
    statuses: frozenset[SessionStatus] = frozenset(
        {
            SessionStatus.INTERRUPTED,
            SessionStatus.FAILED,
            SessionStatus.COMPLETED,
        }
    )
    kind: PendingActionKind | None = None
    agent_name: str | None = None
    environment_name: str | None = None
    q: str | None = None
    cursor: str | None = None
    limit: StrictInt = Field(default=50, ge=1, le=200)
    max_result_bytes: StrictInt = Field(
        default=DEFAULT_PENDING_ACTION_RESULT_MAX_BYTES,
        ge=1024,
        le=MAX_PENDING_ACTION_RESULT_BYTES,
    )

    @field_validator("statuses", mode="before")
    @classmethod
    def validate_statuses(cls, value: object) -> frozenset[SessionStatus]:
        if not isinstance(value, (set, frozenset, list, tuple)):
            raise TypeError("statuses must be a collection of SessionStatus values.")
        statuses = frozenset(
            status if isinstance(status, SessionStatus) else SessionStatus(status)
            for status in value
        )
        if not statuses:
            raise ValueError("statuses must not be empty.")
        return statuses

    @field_validator(
        "session_id",
        "agent_name",
        "environment_name",
        "q",
        "cursor",
    )
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class DelegatedActionReference(BaseModel):
    """Bounded discovery of a child-owned action; not resolution authority."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    child_session_id: StrictStr
    action_kind: Literal["tool_approval", "user_input", "delegated_action"]
    action_id: StrictStr
    status: Literal["waiting_on_child_action"] = "waiting_on_child_action"

    @field_validator("child_session_id", "action_id")
    @classmethod
    def bounded_identity(cls, value: str, info) -> str:
        value = session_record_rules._require_bounded_session_id(value, info.field_name)
        if info.field_name == "action_id" and len(value.encode("utf-8")) > 256:
            raise ValueError("Delegated action identity exceeds its byte bound.")
        return value


class PendingActionRecord(BaseModel):
    """One current action derived from a session checkpoint and its source event."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: PendingActionKind
    session: PendingActionSession
    event: EventRecord
    title: str
    detail: str | None = None
    tool_name: str | None = None
    approval_id: str | None = None
    input_id: str | None = None
    round_id: str | None = None
    tool_call_id: str | None = None
    source_linkage: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)
    policy_evidence: ToolPolicyEvidence | None = None
    question: str | None = None
    options: list[str] = Field(default_factory=list)
    arguments: dict[str, Any] | None = None
    delegated_action: DelegatedActionReference | None = None

    @model_validator(mode="before")
    @classmethod
    def ignore_cached_attention_identity(cls, value: Any) -> Any:
        # A serialized read projection can be reconstructed normally. Recompute
        # this read-only field from canonical identities, never from a cached ID.
        if isinstance(value, dict) and "attention_id" in value:
            return {key: item for key, item in value.items() if key != "attention_id"}
        return value

    @computed_field
    @property
    def attention_id(self) -> str | None:
        """Stable notification identity; never execution or resolution authority.

        Delegated rows are navigation only. Query the child-owned action.
        Legacy/custom projections without an incarnation remain unavailable.
        """
        if self.session.instance_id is None or self.kind is PendingActionKind.DELEGATED_ACTION:
            return None
        if self.kind is PendingActionKind.USER_INPUT:
            discriminator = self.input_id
        elif self.kind is PendingActionKind.TOOL_APPROVAL:
            discriminator = self.approval_id
        else:
            discriminator = self.round_id or self.id
        if discriminator is None:
            return None
        identity = [
            self.session.id,
            self.session.instance_id,
            self.kind.value,
            discriminator,
            self.tool_call_id if self.kind is PendingActionKind.MANUAL_RECOVERY else None,
        ]
        return (
            "attention_"
            + hashlib.sha256(
                json.dumps(identity, ensure_ascii=True, separators=(",", ":")).encode()
            ).hexdigest()
        )

    @model_validator(mode="after")
    def delegated_action_is_discovery_only(self) -> Self:
        if (self.kind is PendingActionKind.DELEGATED_ACTION) != (self.delegated_action is not None):
            raise ValueError("Delegated pending actions require exactly one child reference.")
        if self.delegated_action is not None and (
            self.approval_id is not None
            or self.input_id is not None
            or self.question is not None
            or self.arguments is not None
            or self.options
        ):
            raise ValueError("Delegated pending actions cannot duplicate child action content.")
        return self

    @field_validator("id", "title")
    @classmethod
    def validate_required_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "detail",
        "tool_name",
        "approval_id",
        "input_id",
        "round_id",
        "tool_call_id",
        "question",
    )
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("session")
    @classmethod
    def copy_session(cls, value: PendingActionSession) -> PendingActionSession:
        return value.model_copy(deep=True)

    @field_validator("event")
    @classmethod
    def copy_event_record(cls, value: EventRecord) -> EventRecord:
        return value.model_copy(deep=True)

    @field_validator("source_linkage", mode="before")
    @classmethod
    def copy_source_linkage(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if type(value) is not dict:
            raise TypeError("source_linkage must be a mapping of event fields to strings.")
        allowed = {"approval_id", "input_id", "tool_round_id", "tool_call_id"}
        if not set(value) <= allowed:
            raise ValueError("source_linkage contains an unsupported event field.")
        return {
            field_name: require_nonblank(field_value, f"source_linkage.{field_name}")
            for field_name, field_value in value.items()
        }

    @field_validator("options", mode="before")
    @classmethod
    def copy_options(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if type(value) is not list:
            raise TypeError("options must be a list of strings.")
        return [require_nonblank(item, "options") for item in value]

    @field_validator("arguments", mode="before")
    @classmethod
    def copy_arguments(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return copy_durable_json_object(value, "arguments")


class PendingActionIssue(BaseModel):
    """Bounded visibility for a candidate that could not be materialized."""

    model_config = ConfigDict(extra="forbid")

    code: PendingActionIssueCode
    session_id: str
    agent_name: str
    status: SessionStatus
    updated_at: datetime
    detail: str

    @field_validator("session_id", "agent_name", "detail")
    @classmethod
    def validate_required_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("updated_at")
    @classmethod
    def normalize_updated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware.")
        return value.astimezone(UTC)

    @classmethod
    def source_too_large(
        cls,
        session: PendingActionSession,
        *,
        max_bytes: int,
    ) -> PendingActionIssue:
        return cls(
            code=PendingActionIssueCode.SOURCE_TOO_LARGE,
            session_id=session.id,
            agent_name=session.agent_name,
            status=session.status,
            updated_at=session.updated_at,
            detail=(
                "Pending-action source data for this session exceeds the "
                f"{max_bytes}-byte inspection limit. Open the session directly "
                "to inspect or resolve it."
            ),
        )

    @classmethod
    def source_too_complex(
        cls,
        session: PendingActionSession,
        *,
        max_tool_calls: int,
    ) -> PendingActionIssue:
        return cls(
            code=PendingActionIssueCode.SOURCE_TOO_COMPLEX,
            session_id=session.id,
            agent_name=session.agent_name,
            status=session.status,
            updated_at=session.updated_at,
            detail=(
                "Pending tool-round state for this session exceeds the "
                f"{max_tool_calls}-call inspection limit. Open the session directly "
                "to inspect or resolve it."
            ),
        )

    @classmethod
    def ledger_too_complex(
        cls,
        session: PendingActionSession,
        *,
        max_events_per_call: int,
    ) -> PendingActionIssue:
        return cls(
            code=PendingActionIssueCode.SOURCE_TOO_COMPLEX,
            session_id=session.id,
            agent_name=session.agent_name,
            status=session.status,
            updated_at=session.updated_at,
            detail=(
                "Pending tool-round evidence for this session exceeds the "
                f"{max_events_per_call}-event per-call inspection limit. "
                "Open the session directly to inspect or resolve it."
            ),
        )

    @classmethod
    def source_invalid(cls, session: PendingActionSession) -> PendingActionIssue:
        return cls(
            code=PendingActionIssueCode.SOURCE_INVALID,
            session_id=session.id,
            agent_name=session.agent_name,
            status=session.status,
            updated_at=session.updated_at,
            detail=(
                "Pending-action state for this session is incomplete or inconsistent. "
                "Inspect the session directly before attempting to resume it."
            ),
        )


class PendingActionListResult(BaseModel):
    """One stable page of pending actions and its candidate continuation cursor."""

    model_config = ConfigDict(extra="forbid")

    actions: list[PendingActionRecord] = Field(default_factory=list)
    issues: list[PendingActionIssue] = Field(default_factory=list)
    next_cursor: str | None = None
    has_more: StrictBool = False
    total_count: StrictInt | None = Field(default=None, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    inspected_candidate_count: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)


def enforce_pending_action_result_size(
    result: PendingActionListResult,
    *,
    max_bytes: int,
) -> PendingActionListResult:
    """Return ``result`` or fail before an oversized API body is serialized."""
    if not json_utf8_size_within_limit(result, max_bytes):
        raise PendingActionResultTooLarge(max_bytes)
    return result
