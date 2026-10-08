"""Session listing, filtering and keyset cursor contracts and shared rules."""

from __future__ import annotations

import base64
import binascii
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER, canonical_durable_json_bytes, copy_label_map
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.budgets.aggregates import AggregateCount
from cayu.sessions.records import (
    PendingActionSession,
    Session,
    SessionStatus,
    _require_bounded_session_id,
)

# A maximum-size session ID fits after base64 encoding in this cursor budget.
MAX_SESSION_LIST_CURSOR_BYTES = 4096
_SESSION_CURSOR_VERSION = 1


def _require_bounded_session_list_cursor(value: str, field_name: str) -> str:
    value = require_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > MAX_SESSION_LIST_CURSOR_BYTES:
        raise ValueError(
            f"`{field_name}` must not exceed {MAX_SESSION_LIST_CURSOR_BYTES} UTF-8 bytes."
        )
    return value


class SessionDebugState(StrEnum):
    NEEDS_ATTENTION = "needs_attention"
    SESSION_FAILURE = "session_failure"
    TOOL_ISSUE = "tool_issue"
    INTERRUPTION = "interruption"


class SessionOrder(StrEnum):
    CREATED_AT_ASC = "created_at_asc"
    CREATED_AT_DESC = "created_at_desc"
    UPDATED_AT_ASC = "updated_at_asc"
    UPDATED_AT_DESC = "updated_at_desc"
    LAST_ACTIVITY_AT_ASC = "last_activity_at_asc"
    LAST_ACTIVITY_AT_DESC = "last_activity_at_desc"


class LabelSelectorOperator(StrEnum):
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    IN = "in"
    NOT_IN = "not_in"


class LabelSelectorRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    operator: LabelSelectorOperator
    values: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return next(iter(copy_label_map({value: "_"}, "label selector").keys()))

    @field_validator("values", mode="before")
    @classmethod
    def copy_values(cls, value) -> tuple[str, ...]:
        if value is None:
            return ()
        if type(value) is str:
            raise ValueError("`values` must be a sequence of strings.")
        values = tuple(value)
        copied: list[str] = []
        for index, item in enumerate(values):
            if type(item) is not str:
                raise ValueError("`values` must contain only strings.")
            copied_value = next(
                iter(copy_label_map({f"value_{index}": item}, "label selector value").values())
            )
            if copied_value in copied:
                raise ValueError("`values` must not contain duplicates.")
            copied.append(copied_value)
        return tuple(copied)

    @model_validator(mode="after")
    def validate_operator_values(self) -> LabelSelectorRequirement:
        if self.operator in {LabelSelectorOperator.EXISTS, LabelSelectorOperator.NOT_EXISTS}:
            if self.values:
                raise ValueError(f"`{self.operator}` label selector must not include values.")
        elif not self.values:
            raise ValueError(f"`{self.operator}` label selector requires at least one value.")
        return self


def copy_label_selector_requirements(
    value: Any,
    field_name: str = "label_selectors",
) -> tuple[LabelSelectorRequirement, ...]:
    if value is None:
        return ()
    if type(value) is LabelSelectorRequirement:
        return (value.model_copy(deep=True),)
    if type(value) in {str, dict}:
        raise ValueError(f"`{field_name}` must be a sequence of label selector requirements.")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(
            f"`{field_name}` must be a sequence of label selector requirements."
        ) from exc
    return tuple(
        item.model_copy(deep=True)
        if type(item) is LabelSelectorRequirement
        else LabelSelectorRequirement.model_validate(item)
        for item in values
    )


class SessionQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    q: str | None = None
    status: SessionStatus | None = None
    debug_state: SessionDebugState | None = None
    agent_name: str | None = None
    provider_name: str | None = None
    model: str | None = None
    environment_name: str | None = None
    parent_session_id: str | None = None
    causal_budget_id: str | None = None
    last_activity_before: datetime | None = None
    inactive_for_seconds: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    labels: dict[str, str] = Field(default_factory=dict)
    label_selectors: tuple[LabelSelectorRequirement, ...] = Field(default_factory=tuple)
    limit: StrictInt = Field(default=100, ge=1, le=1000)
    offset: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    cursor: str | None = None
    include_total_count: StrictBool = False
    order_by: SessionOrder = SessionOrder.UPDATED_AT_DESC

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_bounded_session_list_cursor(value, "cursor")

    @model_validator(mode="after")
    def reject_cursor_with_offset(self) -> SessionQuery:
        # A keyset cursor and offset are two different paging schemes; combining them
        # would silently ignore the offset, so reject it explicitly.
        if self.cursor is not None and self.offset:
            raise ValueError("cursor and a non-zero offset cannot be combined.")
        if self.last_activity_before is not None and self.inactive_for_seconds is not None:
            raise ValueError("last_activity_before and inactive_for_seconds cannot be combined.")
        return self

    @field_validator(
        "q",
        "agent_name",
        "provider_name",
        "model",
        "environment_name",
        "parent_session_id",
        "causal_budget_id",
    )
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("last_activity_before")
    @classmethod
    def validate_last_activity_before(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("last_activity_before must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_query_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator("label_selectors", mode="before")
    @classmethod
    def copy_query_label_selectors(cls, value) -> tuple[LabelSelectorRequirement, ...]:
        return copy_label_selector_requirements(value)


MAX_AGGREGATE_LABEL_FILTERS = 50
MAX_AGGREGATE_LABEL_SELECTORS = 25
MAX_AGGREGATE_LABEL_SELECTOR_VALUES = 100


class SessionAggregateFilter(BaseModel):
    """Current session attributes that may scope a store-native aggregate."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    agent_name: str | None = None
    provider_name: str | None = None
    model: str | None = None
    environment_name: str | None = None
    parent_session_id: str | None = None
    causal_budget_id: str | None = None
    labels: dict[str, str] = Field(
        default_factory=dict,
        max_length=MAX_AGGREGATE_LABEL_FILTERS,
    )
    label_selectors: tuple[LabelSelectorRequirement, ...] = Field(
        default_factory=tuple,
        max_length=MAX_AGGREGATE_LABEL_SELECTORS,
    )

    @field_validator(
        "agent_name",
        "provider_name",
        "model",
        "environment_name",
        "parent_session_id",
        "causal_budget_id",
    )
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_filter_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator("label_selectors", mode="before")
    @classmethod
    def copy_filter_label_selectors(cls, value) -> tuple[LabelSelectorRequirement, ...]:
        return copy_label_selector_requirements(value)

    @model_validator(mode="after")
    def validate_selector_value_count(self) -> SessionAggregateFilter:
        value_count = sum(len(selector.values) for selector in self.label_selectors)
        if value_count > MAX_AGGREGATE_LABEL_SELECTOR_VALUES:
            raise ValueError(
                "Aggregate label selectors cannot contain more than "
                f"{MAX_AGGREGATE_LABEL_SELECTOR_VALUES} total values."
            )
        return self


class SessionStatusCounts(BaseModel):
    """Complete current-session counts for every lifecycle status."""

    model_config = ConfigDict(extra="forbid")

    pending: AggregateCount = Field(ge=0)
    running: AggregateCount = Field(ge=0)
    interrupting: AggregateCount = Field(ge=0)
    completed: AggregateCount = Field(ge=0)
    failed: AggregateCount = Field(ge=0)
    interrupted: AggregateCount = Field(ge=0)


class SessionListResult(BaseModel):
    """One page of a session listing plus its keyset cursor and (optional) total count."""

    model_config = ConfigDict(extra="forbid")

    sessions: list[Session] = Field(default_factory=list)
    next_cursor: str | None = None
    # None unless the query opted in via include_total_count (COUNT is expensive at scale).
    total_count: StrictInt | None = Field(default=None, ge=0, le=MAX_DURABLE_JSON_INTEGER)

    @field_validator("next_cursor")
    @classmethod
    def validate_next_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_bounded_session_list_cursor(value, "next_cursor")


def copy_session_query(query: SessionQuery | None) -> SessionQuery:
    if query is None:
        return SessionQuery()
    if type(query) is not SessionQuery:
        raise TypeError("Session queries must be SessionQuery instances.")
    return SessionQuery(
        q=query.q,
        status=query.status,
        debug_state=query.debug_state,
        agent_name=query.agent_name,
        provider_name=query.provider_name,
        model=query.model,
        environment_name=query.environment_name,
        parent_session_id=query.parent_session_id,
        causal_budget_id=query.causal_budget_id,
        last_activity_before=query.last_activity_before,
        inactive_for_seconds=query.inactive_for_seconds,
        labels=copy_label_map(query.labels, "labels"),
        label_selectors=copy_label_selector_requirements(query.label_selectors),
        limit=query.limit,
        offset=query.offset,
        cursor=query.cursor,
        include_total_count=query.include_total_count,
        order_by=query.order_by,
    )


def copy_session_aggregate_filter(
    filters: SessionAggregateFilter | None,
) -> SessionAggregateFilter:
    if filters is None:
        return SessionAggregateFilter()
    if type(filters) is not SessionAggregateFilter:
        raise TypeError("Session aggregate filters must be SessionAggregateFilter instances.")
    return SessionAggregateFilter.model_validate(filters.model_dump(mode="python"))


def session_query_from_aggregate_filter(filters: SessionAggregateFilter) -> SessionQuery:
    filters = copy_session_aggregate_filter(filters)
    return SessionQuery(
        agent_name=filters.agent_name,
        provider_name=filters.provider_name,
        model=filters.model,
        environment_name=filters.environment_name,
        parent_session_id=filters.parent_session_id,
        causal_budget_id=filters.causal_budget_id,
        labels=filters.labels,
        label_selectors=filters.label_selectors,
    )


def _session_matches(session: Session, query: SessionQuery) -> bool:
    if query.q is not None and not _session_query_text_matches(session, query.q):
        return False
    if query.status is not None and session.status != query.status:
        return False
    if query.debug_state is not None:
        raise ValueError("SessionQuery debug_state requires event-aware store filtering.")
    if query.agent_name is not None and session.agent_name != query.agent_name:
        return False
    if query.provider_name is not None and session.provider_name != query.provider_name:
        return False
    if query.model is not None and session.model != query.model:
        return False
    if query.parent_session_id is not None and session.parent_session_id != query.parent_session_id:
        return False
    if query.causal_budget_id is not None and session.causal_budget_id != query.causal_budget_id:
        return False
    if (
        query.last_activity_before is not None
        and session.last_activity_at > query.last_activity_before
    ):
        return False
    if query.inactive_for_seconds is not None:
        raise ValueError("SessionQuery inactive_for_seconds must be resolved by the SessionStore.")
    for key, value in query.labels.items():
        if session.labels.get(key) != value:
            return False
    for selector in query.label_selectors:
        if not _label_selector_matches(session.labels, selector):
            return False
    return not (
        query.environment_name is not None and session.environment_name != query.environment_name
    )


def _session_query_text_matches(session: Session, query: str) -> bool:
    needle = query.casefold()
    haystacks = [
        session.id,
        session.agent_name,
        session.provider_name,
        session.model,
        session.environment_name,
        session.parent_session_id,
        session.causal_budget_id,
        *session.labels.keys(),
        *session.labels.values(),
    ]
    return any(needle in value.casefold() for value in haystacks if type(value) is str and value)


def _label_selector_matches(
    labels: dict[str, str],
    selector: LabelSelectorRequirement,
) -> bool:
    value = labels.get(selector.key)
    if selector.operator == LabelSelectorOperator.EXISTS:
        return value is not None
    if selector.operator == LabelSelectorOperator.NOT_EXISTS:
        return value is None
    if selector.operator == LabelSelectorOperator.IN:
        return value in selector.values
    if selector.operator == LabelSelectorOperator.NOT_IN:
        return value is None or value not in selector.values
    raise ValueError(f"Unsupported label selector operator: {selector.operator}")


def _sort_sessions(sessions: list[Session], order_by: SessionOrder) -> list[Session]:
    if order_by == SessionOrder.CREATED_AT_ASC:
        return sorted(sessions, key=lambda session: (session.created_at, session.id))
    if order_by == SessionOrder.CREATED_AT_DESC:
        return sorted(
            sorted(sessions, key=lambda session: session.id),
            key=lambda session: session.created_at,
            reverse=True,
        )
    if order_by == SessionOrder.UPDATED_AT_ASC:
        return sorted(sessions, key=lambda session: (session.updated_at, session.id))
    if order_by == SessionOrder.UPDATED_AT_DESC:
        return sorted(
            sorted(sessions, key=lambda session: session.id),
            key=lambda session: session.updated_at,
            reverse=True,
        )
    if order_by == SessionOrder.LAST_ACTIVITY_AT_ASC:
        return sorted(sessions, key=lambda session: (session.last_activity_at, session.id))
    return sorted(
        sorted(sessions, key=lambda session: session.id),
        key=lambda session: session.last_activity_at,
        reverse=True,
    )


_DESCENDING_SESSION_ORDERS = frozenset(
    {
        SessionOrder.CREATED_AT_DESC,
        SessionOrder.UPDATED_AT_DESC,
        SessionOrder.LAST_ACTIVITY_AT_DESC,
    }
)
_CREATED_AT_ORDERS = frozenset({SessionOrder.CREATED_AT_ASC, SessionOrder.CREATED_AT_DESC})
_LAST_ACTIVITY_AT_ORDERS = frozenset(
    {SessionOrder.LAST_ACTIVITY_AT_ASC, SessionOrder.LAST_ACTIVITY_AT_DESC}
)


def session_order_is_descending(order_by: SessionOrder) -> bool:
    return order_by in _DESCENDING_SESSION_ORDERS


def session_sort_column(order_by: SessionOrder) -> str:
    """The session column an order sorts by — the keyset cursor's primary key."""
    if order_by in _CREATED_AT_ORDERS:
        return "created_at"
    if order_by in _LAST_ACTIVITY_AT_ORDERS:
        return "last_activity_at"
    return "updated_at"


def _session_sort_value(
    session: Session | PendingActionSession,
    order_by: SessionOrder,
) -> datetime:
    if order_by in _CREATED_AT_ORDERS:
        return session.created_at
    if order_by in _LAST_ACTIVITY_AT_ORDERS:
        return session.last_activity_at if isinstance(session, Session) else session.updated_at
    return session.updated_at


def encode_session_cursor(
    session: Session | PendingActionSession,
    order_by: SessionOrder,
) -> str:
    """Opaque keyset cursor for the last row of a page: (sort value, session id)."""
    sort_value = _session_sort_value(session, order_by).astimezone(UTC).isoformat()
    session_id = _require_bounded_session_id(session.id, "session.id")
    material = {
        "version": _SESSION_CURSOR_VERSION,
        "sort_value": sort_value,
        "session_id_b64": base64.urlsafe_b64encode(session_id.encode("utf-8")).decode("ascii"),
    }
    encoded = base64.urlsafe_b64encode(
        canonical_durable_json_bytes(material, "session cursor")
    ).decode("ascii")
    return _require_bounded_session_list_cursor(encoded, "session cursor")


def decode_session_cursor(cursor: str) -> tuple[datetime, str]:
    """Decode a cursor to (sort value, session id). Raises ValueError if malformed."""
    try:
        cursor = _require_bounded_session_list_cursor(cursor, "cursor")
        encoded = cursor.encode("ascii")
        raw = base64.b64decode(encoded, altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw) != encoded:
            raise ValueError("Non-canonical session cursor.")
        decoded = json.loads(raw.decode("utf-8"))
        if (
            type(decoded) is not dict
            or set(decoded) != {"version", "sort_value", "session_id_b64"}
            or type(decoded["version"]) is not int
            or decoded["version"] != _SESSION_CURSOR_VERSION
            or type(decoded["sort_value"]) is not str
            or type(decoded["session_id_b64"]) is not str
        ):
            raise ValueError("Invalid session cursor material.")
        encoded_session_id = decoded["session_id_b64"].encode("ascii")
        session_id_bytes = base64.b64decode(
            encoded_session_id,
            altchars=b"-_",
            validate=True,
        )
        if base64.urlsafe_b64encode(session_id_bytes) != encoded_session_id:
            raise ValueError("Non-canonical session identifier.")
        session_id = _require_bounded_session_id(
            session_id_bytes.decode("utf-8"),
            "session cursor id",
        )
    except (
        binascii.Error,
        UnicodeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError("Invalid session cursor.") from exc
    try:
        sort_value = datetime.fromisoformat(decoded["sort_value"])
    except ValueError as exc:
        raise ValueError("Invalid session cursor.") from exc
    # Sort values are always encoded as UTC-aware timestamps; a naive datetime is
    # a malformed/forged cursor that would raise TypeError when later compared
    # against the timezone-aware session timestamps.
    if sort_value.tzinfo is None:
        raise ValueError("Invalid session cursor.")
    return sort_value, session_id


def session_next_cursor(page: list[Session], has_more: bool, order_by: SessionOrder) -> str | None:
    """The keyset cursor for the next page: the last row's cursor, or None if no more."""
    return encode_session_cursor(page[-1], order_by) if has_more and page else None


def _session_after_cursor(
    session: Session,
    order_by: SessionOrder,
    cursor_value: datetime,
    cursor_id: str,
) -> bool:
    """Whether ``session`` falls strictly after the cursor under ``order_by`` (id tiebreak ASC)."""
    value = _session_sort_value(session, order_by)
    if value != cursor_value:
        if session_order_is_descending(order_by):
            return value < cursor_value
        return value > cursor_value
    return session.id > cursor_id
