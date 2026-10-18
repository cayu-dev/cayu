"""Bounded cross-session query over durable human-supervision events.

A supervision event is one settled human or operator touch on a session:
a tool-approval decision or expiry, a user-input answer, a manual recovery,
an operator interruption, or a settled steering message. The query reads only
durable runtime events; it never infers a touch from missing evidence.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import secrets
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER, canonical_durable_json_bytes
from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.approvals.revisions import ApprovalArgumentsRevision, ApprovalRevision
from cayu.budgets.usage import UsageMetrics
from cayu.releases import validate_release_id
from cayu.sessions.queries import SessionAggregateFilter

if TYPE_CHECKING:
    from cayu.budgets.pricing import ModelStepCostEstimate, PriceBook

SUPERVISION_EVENT_MAX_WINDOW = timedelta(days=366)
SUPERVISION_EVENT_DEFAULT_LIMIT = 100
SUPERVISION_EVENT_MAX_LIMIT = 200
SUPERVISION_EVENT_DEFAULT_RESULT_BYTES = 1_048_576
SUPERVISION_EVENT_MAX_RESULT_BYTES = 4_194_304
SUPERVISION_EVENT_MAX_CURSOR_BYTES = 4_096
_SUPERVISION_EVENT_CURSOR_VERSION = 1
_SUPERVISION_EVENT_CURSOR_DOMAIN = "cayu.supervision-events.cursor.v1"
_SUPERVISION_EVENT_CURSOR_KEYS = frozenset({"version", "query_sha256", "after_sequence"})
_SUPERVISION_EVENT_PAGE_FIELDS = frozenset({"cursor", "limit", "max_bytes"})
# Filters added after cursors were first issued.
_SUPERVISION_EVENT_FILTERS_DIGESTED_WHEN_SET = frozenset({"release_id"})
_LOWER_HEX_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_ASCII_PATTERN = r"^[\x21-\x7e]+$"
_LOWER_HEX_SHA256 = re.compile(_LOWER_HEX_SHA256_PATTERN)


class SupervisionEventKind(StrEnum):
    """Closed set of supervision touches the runtime records durably."""

    APPROVE = "approve"
    EDIT = "edit"
    DENY = "deny"
    EXPIRE = "expire"
    ANSWER = "answer"
    RECOVER = "recover"
    INTERRUPT = "interrupt"
    STEER = "steer"


class SupervisionEventOutcome(StrEnum):
    """Settled result of one supervision touch."""

    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    ANSWERED = "answered"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    DELIVERED = "delivered"
    WITHDRAWN = "withdrawn"
    QUARANTINED = "quarantined"
    STALE = "stale"


class SupervisionMessageOrigin(StrEnum):
    """Who submitted a steering message, from durable message provenance."""

    ACTOR = "actor"
    SESSION = "session"
    PEER = "peer"
    UNKNOWN = "unknown"


# Settled human-attention states a touch can report; the same vocabulary as
# ``cayu.runtime.human_attention.AttentionState``.
SupervisionAttentionState = Literal["resolved", "cancelled", "expired"]


class SupervisionGrantRole(StrEnum):
    """How a tool authorization grant took part in one approval."""

    SAMPLED_REVIEW = "sampled_review"
    CREATED = "created"


class SupervisionEventQuery(BaseModel):
    """Half-open ``[start_at, end_at)`` window over settled supervision touches.

    ``kinds`` and ``actor_sources`` are empty for "any". A non-empty
    ``actor_sources`` excludes touches without a recorded actor. ``release_id``
    keeps touches whose work ran on that Release.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    start_at: datetime
    end_at: datetime
    sessions: SessionAggregateFilter = Field(default_factory=SessionAggregateFilter)
    exclude_forked_sessions: StrictBool = False
    execution_profile_fingerprint: StrictStr | None = Field(
        default=None, pattern=_LOWER_HEX_SHA256_PATTERN
    )
    release_id: StrictStr | None = None
    kinds: tuple[SupervisionEventKind, ...] = Field(
        default_factory=tuple, max_length=len(SupervisionEventKind)
    )
    actor_sources: tuple[ResolutionActorSource, ...] = Field(
        default_factory=tuple, max_length=len(ResolutionActorSource)
    )
    limit: StrictInt = Field(
        default=SUPERVISION_EVENT_DEFAULT_LIMIT, ge=1, le=SUPERVISION_EVENT_MAX_LIMIT
    )
    max_bytes: StrictInt = Field(
        default=SUPERVISION_EVENT_DEFAULT_RESULT_BYTES,
        ge=1,
        le=SUPERVISION_EVENT_MAX_RESULT_BYTES,
    )
    cursor: StrictStr | None = Field(
        default=None,
        min_length=1,
        max_length=SUPERVISION_EVENT_MAX_CURSOR_BYTES,
        pattern=_ASCII_PATTERN,
    )

    @field_validator("start_at", "end_at")
    @classmethod
    def normalize_window_timestamp(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("release_id")
    @classmethod
    def require_valid_release_id(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return validate_release_id(value, info.field_name)

    @field_validator("kinds", "actor_sources")
    @classmethod
    def require_distinct_filter_values(cls, value: tuple[Any, ...], info) -> tuple[Any, ...]:
        if len(set(value)) != len(value):
            raise ValueError(f"{info.field_name} cannot repeat a value.")
        return value

    @model_validator(mode="after")
    def validate_window(self) -> SupervisionEventQuery:
        if self.start_at >= self.end_at:
            raise ValueError("Supervision event start_at must be before end_at.")
        if self.end_at - self.start_at > SUPERVISION_EVENT_MAX_WINDOW:
            raise ValueError(
                f"Supervision event window cannot exceed {SUPERVISION_EVENT_MAX_WINDOW.days} days."
            )
        return self


class SupervisionEventContent(BaseModel):
    """Runtime-published content of one touch, returned only on explicit request.

    Each field holds only what the runtime already published durably; content
    the runtime withheld at publication stays ``None``. ``tool_arguments`` maps
    a tool-call id to the arguments its terminal event published, which are the
    model's; ``approved_arguments`` maps the call a person changed to the
    arguments it ran with, under the same publication rule.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    tool_arguments: dict[StrictStr, Any] | None = None
    approved_arguments: dict[StrictStr, Any] | None = None
    question: StrictStr | None = None
    options: tuple[StrictStr, ...] | None = None
    answer: Any = None
    message: Any = None


class SupervisionEvent(BaseModel):
    """One settled supervision touch, keyed by a stable runtime identity.

    ``attention_id`` names the human-attention request the touch settled, with
    the identity pending-action reads and ``get_human_attention_state`` use.
    ``attention_state`` is that request's settled state when the shared
    settlement rule decides it, and ``None`` otherwise. ``approval_revision`` is
    the application's content-free declaration of the approved version.
    ``arguments_revision`` is the runtime's own content-free evidence that a
    person approved the call with changed arguments; either one makes the touch
    an ``edit``.
    ``authorization_grant_id`` names the tool authorization grant that sent the
    call to sampled review or that this approval created, per
    ``authorization_grant_role``. ``release_id`` and ``release_snapshot_root``
    name the Release of the work the touch belongs to: the invocation that
    proposed the call it settles, or the work in force at an interrupt or
    steering message. Both are ``None`` when that work recorded no Release.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    supervision_id: StrictStr = Field(pattern=_LOWER_HEX_SHA256_PATTERN)
    kind: SupervisionEventKind
    outcome: SupervisionEventOutcome
    sequence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    session_id: StrictStr
    parent_session_id: StrictStr | None = None
    forked: StrictBool
    agent_name: StrictStr | None = None
    environment_name: StrictStr | None = None
    tool_name: StrictStr | None = None
    execution_profile_fingerprint: StrictStr | None = Field(
        default=None, pattern=_LOWER_HEX_SHA256_PATTERN
    )
    release_id: StrictStr | None = None
    release_snapshot_root: StrictStr | None = Field(default=None, pattern=_LOWER_HEX_SHA256_PATTERN)
    model_step_id: StrictStr | None = None
    model_attempt_id: StrictStr | None = None
    tool_round_id: StrictStr | None = None
    tool_call_ids: tuple[StrictStr, ...] = ()
    approval_id: StrictStr | None = None
    input_id: StrictStr | None = None
    queue_id: StrictStr | None = None
    interruption_request_id: StrictStr | None = None
    message_origin: SupervisionMessageOrigin | None = None
    actor: ResolutionActor | None = None
    reason: StrictStr | None = None
    attention_id: StrictStr | None = Field(default=None, pattern=r"^attention_[0-9a-f]{64}$")
    attention_state: SupervisionAttentionState | None = None
    approval_revision: ApprovalRevision | None = None
    arguments_revision: ApprovalArgumentsRevision | None = None
    authorization_grant_id: StrictStr | None = None
    authorization_grant_role: SupervisionGrantRole | None = None
    requested_at: datetime | None = None
    resolved_at: datetime
    usage: UsageMetrics | None = None
    content: SupervisionEventContent | None = None

    @field_validator("actor")
    @classmethod
    def reject_actor_claims(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        if value is not None and value.claims:
            raise ValueError("Supervision event actors never carry claims.")
        return value


class SupervisionEventCursorInvalid(ValueError):
    """A cursor is malformed or was issued for another query or reader."""


class SupervisionEventResultTooLarge(ValueError):
    """A supervision page or one of its bounded evidence reads exceeded its limit."""


class SupervisionEventPage(BaseModel):
    """One bounded page in sequence order.

    ``next_cursor`` always resumes after the last durable position this page
    inspected, so a caller can continue later and receive touches settled
    since. ``has_more`` reports whether more settled touches were already
    available. A page can hold fewer than ``limit`` events when filters that
    are applied after classification drop candidates.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    events: tuple[SupervisionEvent, ...]
    next_cursor: StrictStr = Field(min_length=1, max_length=SUPERVISION_EVENT_MAX_CURSOR_BYTES)
    has_more: StrictBool


def supervision_event_query_fingerprint(
    query: SupervisionEventQuery,
    *,
    include_content: bool = False,
    audience: str | None = None,
) -> str:
    """Digest every decision-bearing query field except cursor and page bounds.

    ``audience`` names the reader the page was produced for, such as an access
    scope or a verified tenant, so a cursor cannot move between readers. A filter
    added after cursors were first issued is digested only when set, so a query
    that leaves it unset keeps the digest earlier Cayu releases issued cursors
    for and those cursors keep paging across an upgrade or rollback.
    """

    if type(query) is not SupervisionEventQuery:
        raise TypeError("query must be a SupervisionEventQuery.")
    if type(include_content) is not bool:
        raise TypeError("include_content must be a bool.")
    if audience is not None and type(audience) is not str:
        raise TypeError("audience must be a string or None.")
    undigested = {
        *_SUPERVISION_EVENT_PAGE_FIELDS,
        *(
            field_name
            for field_name in _SUPERVISION_EVENT_FILTERS_DIGESTED_WHEN_SET
            if getattr(query, field_name) is None
        ),
    }
    material = {
        "domain": _SUPERVISION_EVENT_CURSOR_DOMAIN,
        "query": query.model_dump(mode="json", exclude=undigested),
        "include_content": include_content,
        "audience": audience,
    }
    return sha256(canonical_durable_json_bytes(material, "supervision event query")).hexdigest()


def encode_supervision_event_cursor(
    query: SupervisionEventQuery,
    *,
    after_sequence: int,
    include_content: bool = False,
    audience: str | None = None,
) -> str:
    """Return an opaque cursor that resumes ``query`` after ``after_sequence``."""

    if type(after_sequence) is not int:
        raise TypeError("after_sequence must be an int.")
    if not 0 <= after_sequence <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError("Supervision event cursor sequence is invalid.")
    material = {
        "version": _SUPERVISION_EVENT_CURSOR_VERSION,
        "query_sha256": supervision_event_query_fingerprint(
            query, include_content=include_content, audience=audience
        ),
        "after_sequence": after_sequence,
    }
    encoded = base64.urlsafe_b64encode(
        canonical_durable_json_bytes(material, "supervision event cursor")
    ).decode("ascii")
    if len(encoded) > SUPERVISION_EVENT_MAX_CURSOR_BYTES:
        raise ValueError("Supervision event cursor exceeds its byte limit.")
    return encoded


def decode_supervision_event_cursor(
    query: SupervisionEventQuery,
    *,
    include_content: bool = False,
    audience: str | None = None,
) -> int | None:
    """Return the resume sequence of ``query.cursor``, or ``None`` without one.

    Raises ``SupervisionEventCursorInvalid`` for a malformed cursor or one issued
    for another query or reader; neither message echoes the cursor.
    """

    if type(query) is not SupervisionEventQuery:
        raise TypeError("query must be a SupervisionEventQuery.")
    if query.cursor is None:
        return None
    expected = supervision_event_query_fingerprint(
        query, include_content=include_content, audience=audience
    )
    try:
        encoded = query.cursor.encode("ascii")
        raw = base64.b64decode(encoded, altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw) != encoded:
            raise ValueError("Non-canonical supervision event cursor.")
        decoded = json.loads(raw.decode("utf-8"))
        if (
            type(decoded) is not dict
            or set(decoded) != _SUPERVISION_EVENT_CURSOR_KEYS
            or type(decoded["version"]) is not int
            or decoded["version"] != _SUPERVISION_EVENT_CURSOR_VERSION
            or type(decoded["query_sha256"]) is not str
            or _LOWER_HEX_SHA256.fullmatch(decoded["query_sha256"]) is None
            or type(decoded["after_sequence"]) is not int
            or not 0 <= decoded["after_sequence"] <= MAX_DURABLE_JSON_INTEGER
        ):
            raise ValueError("Invalid supervision event cursor material.")
    except (binascii.Error, UnicodeError, ValueError, TypeError) as exc:
        raise SupervisionEventCursorInvalid("Invalid supervision event cursor.") from exc
    if not secrets.compare_digest(decoded["query_sha256"], expected):
        raise SupervisionEventCursorInvalid("Supervision event cursor belongs to another query.")
    return decoded["after_sequence"]


def price_supervision_events(
    page: SupervisionEventPage, pricing: PriceBook
) -> dict[str, ModelStepCostEstimate]:
    """Estimate each event's joined model-step usage with an application price book.

    The result is keyed by ``supervision_id``; events without usage are absent.
    Prices are taken on the request date, since the model step ends just before
    its request is published, or on the touch date when no request was found.
    """

    from cayu.budgets.pricing import PriceBook, estimate_model_step_cost

    if type(page) is not SupervisionEventPage:
        raise TypeError("page must be a SupervisionEventPage.")
    if type(pricing) is not PriceBook:
        raise TypeError("pricing must be a PriceBook.")
    return {
        event.supervision_id: estimate_model_step_cost(
            metrics=event.usage,
            pricing=pricing,
            effective_on=(event.requested_at or event.resolved_at).astimezone(UTC).date(),
        )
        for event in page.events
        if event.usage is not None
    }
