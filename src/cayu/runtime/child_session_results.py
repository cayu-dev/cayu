from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import require_durable_clean_nonblank
from cayu.core.events import EventType
from cayu.core.messages import MessageRole
from cayu.runtime._child_session_notifications import (
    ChildSessionLifecycleOccurrence,
    ChildSessionLifecycleOccurrenceSource,
    ChildSessionLifecycleState,
    child_session_notification_occurrence_id,
)
from cayu.runtime.child_session_context import (
    CHILD_SESSION_PUBLIC_ALIAS_MAX_CHARS,
    CHILD_SESSION_PUBLIC_OCCURRENCE_ID_MAX_CHARS,
    ChildSessionResultReference,
    _require_child_occurrence_id,
    _require_public_session_alias,
)
from cayu.runtime.sessions import (
    LATEST_TRANSCRIPT_TEXT_MAX_CHARS,
    Session,
    SessionStatus,
    SessionStore,
    TranscriptTextReadLimitExceeded,
)

CHILD_SESSION_RESULT_PROJECTION_VERSION = "cayu.child-session-result.v1"
DEFAULT_CHILD_SESSION_RESULT_MAX_CHARS = 12_000
MAX_CHILD_SESSION_RESULT_MAX_CHARS = LATEST_TRANSCRIPT_TEXT_MAX_CHARS

_TERMINAL_STATE_BY_STATUS = {
    SessionStatus.COMPLETED: ChildSessionLifecycleState.COMPLETED,
    SessionStatus.FAILED: ChildSessionLifecycleState.FAILED,
    SessionStatus.INTERRUPTED: ChildSessionLifecycleState.INTERRUPTED,
}
_TERMINAL_EVENT_BY_STATUS = {
    SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
    SessionStatus.FAILED: EventType.SESSION_FAILED,
    SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
}


class ChildSessionResultUnavailable(LookupError):
    """The requested child result is absent, stale, or outside parent authority."""


class ChildSessionResultProjection(BaseModel):
    """A separately authorized, bounded terminal child-result read."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal["cayu.child-session-result.v1"] = (
        CHILD_SESSION_RESULT_PROJECTION_VERSION
    )
    parent_session_id: str = Field(max_length=CHILD_SESSION_PUBLIC_ALIAS_MAX_CHARS)
    child_session_id: str = Field(max_length=CHILD_SESSION_PUBLIC_ALIAS_MAX_CHARS)
    terminal_occurrence_id: str = Field(max_length=CHILD_SESSION_PUBLIC_OCCURRENCE_ID_MAX_CHARS)
    state: ChildSessionLifecycleState
    result_text: str = Field(max_length=MAX_CHILD_SESSION_RESULT_MAX_CHARS)
    result_truncated: StrictBool
    result_chars: StrictInt = Field(ge=0, le=MAX_CHILD_SESSION_RESULT_MAX_CHARS)
    max_chars: StrictInt = Field(ge=1, le=MAX_CHILD_SESSION_RESULT_MAX_CHARS)

    @field_validator("parent_session_id", "child_session_id")
    @classmethod
    def validate_identity(cls, value: str, info: Any) -> str:
        return _require_public_session_alias(value, info.field_name)

    @field_validator("terminal_occurrence_id")
    @classmethod
    def validate_terminal_occurrence_id(cls, value: str) -> str:
        return _require_child_occurrence_id(value, "terminal_occurrence_id")

    @model_validator(mode="after")
    def validate_terminal_bound(self) -> ChildSessionResultProjection:
        if not self.state.terminal:
            raise ValueError("A child-session result projection requires a terminal state.")
        if self.result_chars != len(self.result_text):
            raise ValueError("result_chars must equal the projected result-text length.")
        if self.result_chars > self.max_chars:
            raise ValueError("The projected result exceeds max_chars.")
        if self.result_truncated and self.result_chars != self.max_chars:
            raise ValueError("A truncated result must fill the requested character bound.")
        return self


@dataclass(frozen=True, slots=True)
class _CurrentTerminalAuthority:
    parent: Session
    child: Session
    state: ChildSessionLifecycleState
    occurrence: ChildSessionLifecycleOccurrence
    occurrence_id: str


async def project_terminal_child_session_result(
    session_store: SessionStore,
    *,
    parent_session_id: str,
    reference: ChildSessionResultReference,
    max_chars: int = DEFAULT_CHILD_SESSION_RESULT_MAX_CHARS,
) -> ChildSessionResultProjection:
    """Resolve one exact terminal occurrence under direct-parent authority.

    ``reference`` is a locator, not a capability. The caller must supply the
    private parent session authority, and this function revalidates the current
    direct-parent relationship and exact canonical terminal event before it
    reads any child content.
    """

    if not isinstance(session_store, SessionStore):
        raise TypeError("session_store must be a SessionStore.")
    parent_session_id = require_durable_clean_nonblank(
        parent_session_id,
        "parent_session_id",
    )
    reference = ChildSessionResultReference.model_validate(reference)
    if type(max_chars) is not int:
        raise TypeError("max_chars must be an integer.")
    if not 1 <= max_chars <= MAX_CHILD_SESSION_RESULT_MAX_CHARS:
        raise ValueError(f"max_chars must be between 1 and {MAX_CHILD_SESSION_RESULT_MAX_CHARS}.")
    if (
        session_store.child_session_notification_version != 1
        or not session_store.supports_public_authority_aliases
        or session_store.public_authority_alias_codec is None
    ):
        raise ChildSessionResultUnavailable("Child-session result resolution is unavailable.")

    try:
        child_session_id = await session_store.resolve_public_authority_alias(
            reference.child_session_id,
            field_name="session_id",
        )
    except (TypeError, ValueError):
        raise ChildSessionResultUnavailable(
            "Child-session result is unavailable to this parent session."
        ) from None
    if child_session_id is None:
        raise ChildSessionResultUnavailable(
            "Child-session result is unavailable to this parent session."
        )
    authority = await _load_current_terminal_authority(
        session_store,
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
    )
    if not hmac.compare_digest(
        reference.terminal_occurrence_id,
        authority.occurrence_id,
    ):
        raise ChildSessionResultUnavailable(
            "Child-session result reference no longer matches its terminal occurrence."
        )

    result_text, result_truncated = await _load_last_assistant_text(
        session_store,
        authority.child.id,
        max_chars=max_chars,
    )
    confirmed = await _load_current_terminal_authority(
        session_store,
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
    )
    if (
        confirmed.parent.instance_id != authority.parent.instance_id
        or confirmed.child.instance_id != authority.child.instance_id
        or confirmed.state is not authority.state
        or confirmed.occurrence != authority.occurrence
        or not hmac.compare_digest(confirmed.occurrence_id, authority.occurrence_id)
    ):
        raise ChildSessionResultUnavailable(
            "Child-session result authority changed while the result was read."
        )
    codec = session_store.public_authority_alias_codec
    assert codec is not None
    return ChildSessionResultProjection(
        parent_session_id=codec.encode(confirmed.parent.id, field_name="session_id"),
        child_session_id=reference.child_session_id,
        terminal_occurrence_id=reference.terminal_occurrence_id,
        state=confirmed.state,
        result_text=result_text,
        result_truncated=result_truncated,
        result_chars=len(result_text),
        max_chars=max_chars,
    )


async def _load_current_terminal_authority(
    session_store: SessionStore,
    *,
    parent_session_id: str,
    child_session_id: str,
) -> _CurrentTerminalAuthority:
    parent_before = await session_store.load(parent_session_id)
    child_before = await session_store.load(child_session_id)
    if (
        parent_before is None
        or child_before is None
        or child_before.parent_session_id != parent_before.id
    ):
        raise ChildSessionResultUnavailable(
            "Child-session result is unavailable to this parent session."
        )
    try:
        outcome_before = await session_store.summarize_outcome(child_session_id)
        parent_after = await session_store.load(parent_session_id)
        child_after = await session_store.load(child_session_id)
        outcome_after = await session_store.summarize_outcome(child_session_id)
    except KeyError:
        raise ChildSessionResultUnavailable(
            "Child-session result is unavailable to this parent session."
        ) from None
    if (
        parent_after is None
        or child_after is None
        or parent_after.instance_id != parent_before.instance_id
        or child_after.instance_id != child_before.instance_id
        or child_after.parent_session_id != parent_after.id
        or child_after.status is not child_before.status
        or child_after.run_epoch != child_before.run_epoch
    ):
        raise ChildSessionResultUnavailable(
            "Child-session result authority changed while it was validated."
        )

    state = _TERMINAL_STATE_BY_STATUS.get(child_after.status)
    expected_event_type = _TERMINAL_EVENT_BY_STATUS.get(child_after.status)
    terminal_before = outcome_before.terminal_event
    terminal_after = outcome_after.terminal_event
    if (
        state is None
        or expected_event_type is None
        or terminal_before is None
        or terminal_after is None
        or terminal_before != terminal_after
        or terminal_after.event.type is not expected_event_type
    ):
        raise ChildSessionResultUnavailable(
            "Child-session terminal result is not currently available."
        )
    occurrence = ChildSessionLifecycleOccurrence(
        source=ChildSessionLifecycleOccurrenceSource.EVENT,
        source_id=terminal_after.event.id,
        source_sequence=terminal_after.sequence,
        source_type=str(terminal_after.event.type),
        occurred_at=terminal_after.event.timestamp,
    )
    return _CurrentTerminalAuthority(
        parent=parent_after,
        child=child_after,
        state=state,
        occurrence=occurrence,
        occurrence_id=child_session_notification_occurrence_id(
            parent_session_instance_id=parent_after.instance_id,
            child_session_instance_id=child_after.instance_id,
            occurrence=occurrence,
        ),
    )


async def _load_last_assistant_text(
    session_store: SessionStore,
    session_id: str,
    *,
    max_chars: int,
) -> tuple[str, bool]:
    try:
        projection = await session_store.load_latest_transcript_text(
            session_id,
            role=MessageRole.ASSISTANT,
            max_chars=max_chars,
        )
    except (NotImplementedError, TranscriptTextReadLimitExceeded):
        raise ChildSessionResultUnavailable(
            "Child-session result exceeds the store's bounded-read contract."
        ) from None
    if projection is None:
        return "", False
    return projection


__all__ = [
    "CHILD_SESSION_RESULT_PROJECTION_VERSION",
    "DEFAULT_CHILD_SESSION_RESULT_MAX_CHARS",
    "MAX_CHILD_SESSION_RESULT_MAX_CHARS",
    "ChildSessionResultProjection",
    "ChildSessionResultUnavailable",
    "project_terminal_child_session_result",
]
