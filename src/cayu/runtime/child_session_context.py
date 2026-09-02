from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
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

from cayu._validation import (
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
)
from cayu.core.events import EventType
from cayu.core.messages import Message, MessageRole
from cayu.runtime._child_session_notifications import (
    CHILD_SESSION_ADMISSION_OCCURRENCE_TYPE,
    CHILD_SESSION_NOTIFICATION_DEFAULT_CHILDREN_INSPECTED,
    CHILD_SESSION_NOTIFICATION_DEFAULT_ENTRIES,
    CHILD_SESSION_NOTIFICATION_DEFAULT_PROJECTION_BYTES,
    CHILD_SESSION_NOTIFICATION_MAX_CHILDREN_INSPECTED,
    CHILD_SESSION_NOTIFICATION_MAX_ENTRIES,
    CHILD_SESSION_NOTIFICATION_MAX_PROJECTION_BYTES,
    CHILD_SESSION_NOTIFICATION_MAX_RETRIEVAL_REFERENCES,
    ChildSessionLifecycleEntry,
    ChildSessionLifecycleOccurrenceSource,
    ChildSessionLifecycleQuery,
    ChildSessionLifecycleState,
    ChildSessionNotificationClaim,
    ChildSessionNotificationFreshness,
    ChildSessionNotificationStageBinding,
    ChildSessionRelationship,
    child_session_notification_occurrence_id,
)
from cayu.runtime.public_authority import parse_public_authority_alias
from cayu.runtime.sessions import Session, SessionStore

CHILD_SESSION_CONTEXT_PROJECTION_VERSION = "cayu.child-session-context.v1"
CHILD_SESSION_RESULT_REFERENCE_VERSION = "cayu.child-session-result-reference.v1"
CHILD_SESSION_PUBLIC_ALIAS_MAX_CHARS = 256
CHILD_SESSION_PUBLIC_OCCURRENCE_ID_MAX_CHARS = 128
_MIN_PROJECTION_BYTES = 1024
_CONTEXT_OPEN_MARKER = '<cayu_child_session_notifications version="1">'
_CONTEXT_CLOSE_MARKER = "</cayu_child_session_notifications>"
_PUBLIC_SOURCE_TYPE_MAX_CHARS = 128
_UNAVAILABLE_PARENT_ALIAS = "cayu_authority_unavailable"
_CHILD_SESSION_OCCURRENCE_ID_PATTERN = re.compile(
    r"cayu_child_occurrence_v1_[0-9a-f]{64}\Z",
    flags=re.ASCII,
)


def _require_public_session_alias(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    parsed = parse_public_authority_alias(value)
    if parsed is None or parsed.field_name != "session_id":
        raise ValueError(f"{field_name} must be a canonical public session alias.")
    return value


def _require_child_occurrence_id(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if _CHILD_SESSION_OCCURRENCE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical child occurrence ID.")
    return value


class ChildSessionContextCoverageState(StrEnum):
    COMPLETE = "complete"
    TRUNCATED = "truncated"
    UNAVAILABLE = "unavailable"


class ChildSessionContextTruncationReason(StrEnum):
    CHILD_INSPECTION_LIMIT = "child_inspection_limit"
    ENTRY_LIMIT = "entry_limit"
    PROJECTION_BYTE_LIMIT = "projection_byte_limit"
    SOURCE_OCCURRENCE_UNAVAILABLE = "source_occurrence_unavailable"


class ChildSessionResultReference(BaseModel):
    """Public, non-authorizing locator for a separately authorized bounded read."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal["cayu.child-session-result-reference.v1"] = (
        CHILD_SESSION_RESULT_REFERENCE_VERSION
    )
    resolver: Literal["child_session_result"] = "child_session_result"
    child_session_id: str = Field(max_length=CHILD_SESSION_PUBLIC_ALIAS_MAX_CHARS)
    terminal_occurrence_id: str = Field(max_length=CHILD_SESSION_PUBLIC_OCCURRENCE_ID_MAX_CHARS)

    @field_validator("child_session_id")
    @classmethod
    def validate_child_session_id(cls, value: str) -> str:
        return _require_public_session_alias(value, "child_session_id")

    @field_validator("terminal_occurrence_id")
    @classmethod
    def validate_terminal_occurrence_id(cls, value: str) -> str:
        return _require_child_occurrence_id(value, "terminal_occurrence_id")


class ChildSessionContextOccurrence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str = Field(max_length=CHILD_SESSION_PUBLIC_OCCURRENCE_ID_MAX_CHARS)
    source: ChildSessionLifecycleOccurrenceSource
    source_type: str = Field(max_length=_PUBLIC_SOURCE_TYPE_MAX_CHARS)
    source_sequence: StrictInt | None = Field(default=None, ge=1)

    @field_validator("id")
    @classmethod
    def validate_occurrence_id(cls, value: str) -> str:
        return _require_child_occurrence_id(value, "id")

    @field_validator("source_type")
    @classmethod
    def validate_identity(cls, value: str, info: Any) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


class ChildSessionContextEntry(BaseModel):
    """Secret-free lifecycle state rendered to the parent model."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    parent_session_id: str = Field(max_length=CHILD_SESSION_PUBLIC_ALIAS_MAX_CHARS)
    child_session_id: str = Field(max_length=CHILD_SESSION_PUBLIC_ALIAS_MAX_CHARS)
    relationship: ChildSessionRelationship
    state: ChildSessionLifecycleState
    occurrence: ChildSessionContextOccurrence
    freshness: Literal["current", "fresh"]
    terminal_result_available: StrictBool
    result_reference: ChildSessionResultReference | None = None

    @field_validator("parent_session_id", "child_session_id")
    @classmethod
    def validate_identity(cls, value: str, info: Any) -> str:
        return _require_public_session_alias(value, info.field_name)

    @model_validator(mode="after")
    def validate_terminal_result(self) -> ChildSessionContextEntry:
        if self.state is ChildSessionLifecycleState.ADMITTED:
            if (
                self.occurrence.source is not ChildSessionLifecycleOccurrenceSource.SESSION
                or self.occurrence.source_sequence is not None
                or self.occurrence.source_type != CHILD_SESSION_ADMISSION_OCCURRENCE_TYPE
            ):
                raise ValueError("An admitted child entry must identify its admission record.")
        else:
            expected_event_types = {
                ChildSessionLifecycleState.RUNNING: {
                    str(EventType.SESSION_STARTED),
                    str(EventType.SESSION_RESUMED),
                    str(EventType.SESSION_FORKED),
                },
                ChildSessionLifecycleState.COMPLETED: {str(EventType.SESSION_COMPLETED)},
                ChildSessionLifecycleState.FAILED: {str(EventType.SESSION_FAILED)},
                ChildSessionLifecycleState.INTERRUPTED: {str(EventType.SESSION_INTERRUPTED)},
            }[self.state]
            if (
                self.occurrence.source is not ChildSessionLifecycleOccurrenceSource.EVENT
                or self.occurrence.source_sequence is None
                or self.occurrence.source_type not in expected_event_types
            ):
                raise ValueError("A child entry state must match its lifecycle event.")
        if self.state.terminal:
            if not self.terminal_result_available or self.result_reference is None:
                raise ValueError("A terminal child entry requires bounded result authority.")
            if self.freshness != "fresh":
                raise ValueError("A rendered terminal child entry must be fresh.")
            assert self.result_reference is not None
            if (
                self.result_reference.child_session_id != self.child_session_id
                or self.result_reference.terminal_occurrence_id != self.occurrence.id
            ):
                raise ValueError(
                    "A terminal result reference must identify its containing child occurrence."
                )
        else:
            if self.terminal_result_available or self.result_reference is not None:
                raise ValueError("A nonterminal child entry cannot expose result authority.")
            if self.freshness != "current":
                raise ValueError("A rendered nonterminal child entry must be current.")
        return self


class ChildSessionContextCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    state: ChildSessionContextCoverageState
    inspected_child_count: StrictInt = Field(
        ge=0,
        le=CHILD_SESSION_NOTIFICATION_MAX_CHILDREN_INSPECTED,
    )
    rendered_entry_count: StrictInt = Field(
        ge=0,
        le=min(
            CHILD_SESSION_NOTIFICATION_MAX_ENTRIES,
            CHILD_SESSION_NOTIFICATION_MAX_RETRIEVAL_REFERENCES,
        ),
    )
    unavailable_child_count: StrictInt = Field(
        ge=0,
        le=CHILD_SESSION_NOTIFICATION_MAX_CHILDREN_INSPECTED,
    )
    more_children: StrictBool = False
    reasons: tuple[ChildSessionContextTruncationReason, ...] = ()

    @model_validator(mode="after")
    def validate_coverage(self) -> ChildSessionContextCoverage:
        if len(self.reasons) != len(set(self.reasons)):
            raise ValueError("Child-session coverage reasons cannot repeat.")
        incomplete = self.more_children or self.unavailable_child_count > 0 or bool(self.reasons)
        if (self.state is ChildSessionContextCoverageState.COMPLETE) == incomplete:
            raise ValueError("Child-session context coverage state conflicts with its evidence.")
        if self.rendered_entry_count + self.unavailable_child_count > self.inspected_child_count:
            raise ValueError(
                "Rendered and unavailable children cannot exceed inspected children."
            )
        if self.state is ChildSessionContextCoverageState.UNAVAILABLE and (
            self.inspected_child_count != 0
            or self.rendered_entry_count != 0
            or self.unavailable_child_count != 0
            or self.more_children
        ):
            raise ValueError("Unavailable coverage cannot report inspected child state.")
        if self.state is ChildSessionContextCoverageState.UNAVAILABLE and self.reasons != (
            ChildSessionContextTruncationReason.SOURCE_OCCURRENCE_UNAVAILABLE,
        ):
            raise ValueError("Unavailable coverage requires its exact source evidence.")
        return self


class ChildSessionContextProjection(BaseModel):
    """One versioned, bounded, secret-free model-facing projection."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal["cayu.child-session-context.v1"] = (
        CHILD_SESSION_CONTEXT_PROJECTION_VERSION
    )
    parent_session_id: str = Field(max_length=CHILD_SESSION_PUBLIC_ALIAS_MAX_CHARS)
    consumption_scope: Literal["parent_session_instance"] = "parent_session_instance"
    entries: tuple[ChildSessionContextEntry, ...] = ()
    coverage: ChildSessionContextCoverage
    authority_notice: Literal[
        "notification and retrieval references do not grant tools, workspace, credentials, "
        "budgets, or external effects"
    ] = (
        "notification and retrieval references do not grant tools, workspace, credentials, "
        "budgets, or external effects"
    )

    @field_validator("parent_session_id")
    @classmethod
    def validate_parent_session_id(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "parent_session_id")
        if value == _UNAVAILABLE_PARENT_ALIAS:
            return value
        return _require_public_session_alias(value, "parent_session_id")

    @field_validator("entries", mode="before")
    @classmethod
    def copy_entries(cls, value: Any) -> tuple[ChildSessionContextEntry, ...]:
        if type(value) not in {list, tuple}:
            raise TypeError("entries must be a list or tuple.")
        return tuple(ChildSessionContextEntry.model_validate(entry) for entry in value)

    @model_validator(mode="after")
    def validate_projection(self) -> ChildSessionContextProjection:
        if self.coverage.rendered_entry_count != len(self.entries):
            raise ValueError("Coverage rendered_entry_count must match projection entries.")
        if any(entry.parent_session_id != self.parent_session_id for entry in self.entries):
            raise ValueError("Child-session context entries changed their parent authority.")
        if (
            self.parent_session_id == _UNAVAILABLE_PARENT_ALIAS
            and self.coverage.state is not ChildSessionContextCoverageState.UNAVAILABLE
        ):
            raise ValueError("Unavailable parent authority must match unavailable coverage.")
        child_ids = tuple(entry.child_session_id for entry in self.entries)
        if len(child_ids) != len(set(child_ids)):
            raise ValueError("A child-session context projection cannot repeat a child.")
        return self


class ChildSessionContextContribution(BaseModel):
    """Internal composition result kept separate from durable transcript messages."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    projection: ChildSessionContextProjection | None = None
    message: Message | None = None
    stage_binding: ChildSessionNotificationStageBinding | None = None

    @model_validator(mode="after")
    def validate_contribution(self) -> ChildSessionContextContribution:
        if (self.projection is None) != (self.message is None):
            raise ValueError("Projection and model message must be present together.")
        terminal_count = (
            0
            if self.projection is None
            else sum(entry.state.terminal for entry in self.projection.entries)
        )
        if (terminal_count > 0) != (self.stage_binding is not None):
            raise ValueError("Terminal entries must exactly bind their dispatch claims.")
        if self.stage_binding is not None and len(self.stage_binding.claims) != terminal_count:
            raise ValueError("Terminal entries and dispatch claims have different cardinality.")
        if self.projection is not None:
            assert self.message is not None
            if self.message != Message.text(MessageRole.USER, _render_projection(self.projection)):
                raise ValueError("The model message must exactly render its context projection.")
        if self.stage_binding is not None:
            assert self.projection is not None
            terminal_occurrence_ids = tuple(
                entry.occurrence.id for entry in self.projection.entries if entry.state.terminal
            )
            claimed_occurrence_ids = tuple(
                child_session_notification_occurrence_id(
                    parent_session_instance_id=self.stage_binding.parent_session_instance_id,
                    child_session_instance_id=claim.child_session_instance_id,
                    occurrence=claim.occurrence,
                )
                for claim in self.stage_binding.claims
            )
            if terminal_occurrence_ids != claimed_occurrence_ids:
                raise ValueError(
                    "Terminal entries must exactly match their ordered dispatch claims."
                )
        return self


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildSessionContextContributor:
    """Project current direct-child state after transcript context selection.

    The contributor stores no child lifecycle truth. Terminal freshness is
    consumed only when the owning model stage crosses Cayu's durable dispatch
    fence; nonterminal state remains a current level-triggered snapshot.
    """

    max_children_inspected: int = CHILD_SESSION_NOTIFICATION_DEFAULT_CHILDREN_INSPECTED
    max_entries: int = CHILD_SESSION_NOTIFICATION_DEFAULT_ENTRIES
    max_projection_bytes: int = CHILD_SESSION_NOTIFICATION_DEFAULT_PROJECTION_BYTES

    def __post_init__(self) -> None:
        if type(self.max_children_inspected) is not int or not (
            1 <= self.max_children_inspected <= CHILD_SESSION_NOTIFICATION_MAX_CHILDREN_INSPECTED
        ):
            raise ValueError(
                "max_children_inspected is outside the child-notification inspection bounds."
            )
        if type(self.max_entries) is not int or not (
            1
            <= self.max_entries
            <= min(
                CHILD_SESSION_NOTIFICATION_MAX_ENTRIES,
                CHILD_SESSION_NOTIFICATION_MAX_RETRIEVAL_REFERENCES,
            )
        ):
            raise ValueError("max_entries is outside the child-notification entry bounds.")
        if type(self.max_projection_bytes) is not int or not (
            _MIN_PROJECTION_BYTES
            <= self.max_projection_bytes
            <= CHILD_SESSION_NOTIFICATION_MAX_PROJECTION_BYTES
        ):
            raise ValueError("max_projection_bytes is outside the child-notification byte bounds.")

    def configuration_material(self) -> dict[str, Any]:
        return {
            "kind": "child_session_context",
            "version": 1,
            "max_children_inspected": self.max_children_inspected,
            "max_entries": self.max_entries,
            "max_projection_bytes": self.max_projection_bytes,
            "terminal_consumption_scope": "parent_session_instance",
        }

    async def build(
        self,
        *,
        session_store: SessionStore,
        session: Session,
    ) -> ChildSessionContextContribution:
        if not isinstance(session, Session):
            raise TypeError("session must be a Session.")
        if session_store.child_session_notification_version != 1:
            return self._unavailable(session_store=session_store, session=session)
        codec = session_store.public_authority_alias_codec
        if codec is None or not session_store.supports_public_authority_aliases:
            return self._unavailable(session_store=session_store, session=session)

        page = await session_store.query_child_session_lifecycle(
            ChildSessionLifecycleQuery(
                parent_session_id=session.id,
                max_children_inspected=self.max_children_inspected,
            )
        )
        if (
            page.parent_session_id != session.id
            or page.parent_session_instance_id != session.instance_id
        ):
            raise RuntimeError("Child-session lifecycle projection changed parent authority.")
        parent_public_id = codec.encode(session.id, field_name="session_id")
        candidates = tuple(
            entry
            for entry in page.entries
            if entry.freshness is not ChildSessionNotificationFreshness.CONSUMED
        )
        if not candidates and not page.has_more and page.unavailable_child_count == 0:
            return ChildSessionContextContribution()

        reasons: list[ChildSessionContextTruncationReason] = []
        if page.has_more:
            reasons.append(ChildSessionContextTruncationReason.CHILD_INSPECTION_LIMIT)
        if page.unavailable_child_count:
            reasons.append(ChildSessionContextTruncationReason.SOURCE_OCCURRENCE_UNAVAILABLE)
        retained_candidates = candidates[: self.max_entries]
        if len(candidates) > len(retained_candidates):
            reasons.append(ChildSessionContextTruncationReason.ENTRY_LIMIT)

        public_entries: list[ChildSessionContextEntry] = []
        claims: list[ChildSessionNotificationClaim] = []
        for private_entry in retained_candidates:
            public_entry = _public_child_entry(
                private_entry,
                parent_public_id=parent_public_id,
                session_store=session_store,
            )
            public_entries.append(public_entry)
            if private_entry.state.terminal:
                claims.append(
                    ChildSessionNotificationClaim(
                        child_session_id=private_entry.child_session_id,
                        child_session_instance_id=private_entry.child_session_instance_id,
                        occurrence=private_entry.occurrence,
                    )
                )

        while True:
            projection = _projection(
                parent_public_id=parent_public_id,
                entries=public_entries,
                inspected_child_count=page.inspected_child_count,
                unavailable_child_count=page.unavailable_child_count,
                more_children=page.has_more,
                reasons=reasons,
            )
            content = _render_projection(projection)
            if len(content.encode("utf-8")) <= self.max_projection_bytes:
                break
            if not public_entries:
                raise RuntimeError(
                    "The minimum child-session projection exceeds its configured byte bound."
                )
            removed = public_entries.pop()
            if removed.state.terminal:
                claims.pop()
            if ChildSessionContextTruncationReason.PROJECTION_BYTE_LIMIT not in reasons:
                reasons.append(ChildSessionContextTruncationReason.PROJECTION_BYTE_LIMIT)

        stage_binding = (
            None
            if not claims
            else ChildSessionNotificationStageBinding(
                parent_session_id=session.id,
                parent_session_instance_id=session.instance_id,
                claims=tuple(claims),
            )
        )
        return ChildSessionContextContribution(
            projection=projection,
            message=Message.text(MessageRole.USER, content),
            stage_binding=stage_binding,
        )

    def _unavailable(
        self,
        *,
        session_store: SessionStore,
        session: Session,
    ) -> ChildSessionContextContribution:
        codec = session_store.public_authority_alias_codec
        parent_public_id = (
            _UNAVAILABLE_PARENT_ALIAS
            if codec is None
            else codec.encode(session.id, field_name="session_id")
        )
        projection = ChildSessionContextProjection(
            parent_session_id=parent_public_id,
            coverage=ChildSessionContextCoverage(
                state=ChildSessionContextCoverageState.UNAVAILABLE,
                inspected_child_count=0,
                rendered_entry_count=0,
                unavailable_child_count=0,
                reasons=(ChildSessionContextTruncationReason.SOURCE_OCCURRENCE_UNAVAILABLE,),
            ),
        )
        content = _render_projection(projection)
        if len(content.encode("utf-8")) > self.max_projection_bytes:
            raise RuntimeError("Unavailable child-session projection exceeds its byte bound.")
        return ChildSessionContextContribution(
            projection=projection,
            message=Message.text(MessageRole.USER, content),
        )


def _public_child_entry(
    entry: ChildSessionLifecycleEntry,
    *,
    parent_public_id: str,
    session_store: SessionStore,
) -> ChildSessionContextEntry:
    codec = session_store.public_authority_alias_codec
    if codec is None:
        raise RuntimeError("Child-session public authority aliases are unavailable.")
    child_public_id = codec.encode(entry.child_session_id, field_name="session_id")
    occurrence_id = child_session_notification_occurrence_id(
        parent_session_instance_id=entry.parent_session_instance_id,
        child_session_instance_id=entry.child_session_instance_id,
        occurrence=entry.occurrence,
    )
    result_reference = (
        ChildSessionResultReference(
            child_session_id=child_public_id,
            terminal_occurrence_id=occurrence_id,
        )
        if entry.state.terminal
        else None
    )
    return ChildSessionContextEntry(
        parent_session_id=parent_public_id,
        child_session_id=child_public_id,
        relationship=entry.relationship,
        state=entry.state,
        occurrence=ChildSessionContextOccurrence(
            id=occurrence_id,
            source=entry.occurrence.source,
            source_type=entry.occurrence.source_type,
            source_sequence=entry.occurrence.source_sequence,
        ),
        freshness="fresh" if entry.state.terminal else "current",
        terminal_result_available=entry.state.terminal,
        result_reference=result_reference,
    )


def _projection(
    *,
    parent_public_id: str,
    entries: list[ChildSessionContextEntry],
    inspected_child_count: int,
    unavailable_child_count: int,
    more_children: bool,
    reasons: list[ChildSessionContextTruncationReason],
) -> ChildSessionContextProjection:
    unique_reasons = tuple(dict.fromkeys(reasons))
    if more_children or unavailable_child_count or unique_reasons:
        state = ChildSessionContextCoverageState.TRUNCATED
    else:
        state = ChildSessionContextCoverageState.COMPLETE
    return ChildSessionContextProjection(
        parent_session_id=parent_public_id,
        entries=tuple(entries),
        coverage=ChildSessionContextCoverage(
            state=state,
            inspected_child_count=inspected_child_count,
            rendered_entry_count=len(entries),
            unavailable_child_count=unavailable_child_count,
            more_children=more_children,
            reasons=unique_reasons,
        ),
    )


def _render_projection(projection: ChildSessionContextProjection) -> str:
    material = projection.model_dump(mode="json")
    canonical_durable_json_bytes(material, "child_session_context_projection")
    payload = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{_CONTEXT_OPEN_MARKER}\n{payload}\n{_CONTEXT_CLOSE_MARKER}"


__all__ = [
    "CHILD_SESSION_CONTEXT_PROJECTION_VERSION",
    "CHILD_SESSION_PUBLIC_ALIAS_MAX_CHARS",
    "CHILD_SESSION_PUBLIC_OCCURRENCE_ID_MAX_CHARS",
    "CHILD_SESSION_RESULT_REFERENCE_VERSION",
    "ChildSessionContextContribution",
    "ChildSessionContextContributor",
    "ChildSessionContextCoverage",
    "ChildSessionContextCoverageState",
    "ChildSessionContextEntry",
    "ChildSessionContextOccurrence",
    "ChildSessionContextProjection",
    "ChildSessionContextTruncationReason",
    "ChildSessionResultReference",
]
