"""Read-only evaluation session observations through the SessionStore contract."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cayu.core.events import Event, EventType
from cayu.runtime.sessions import EventOrder, EventQuery, SessionLineageQuery, SessionStore


class EvalDiagnosticV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    event_id: str
    sequence: int
    timestamp: datetime
    event_type: str
    agent: str | None = None
    tool: str | None = None
    tool_call_id: str | None = None
    code: str | None = None
    message: str | None = Field(default=None, max_length=4096)
    retry_scheduled: bool = False
    manual_settlement_reported: bool = False


class EvalSessionObservationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    parent_session_id: str | None
    agent: str
    status: str
    last_activity_at: datetime
    activity: Literal["recent", "stale", "terminal", "clock_skew"]
    model_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    event_count: int = Field(ge=0)
    pending_action_count: int = Field(ge=0)
    pending_action_kinds: tuple[str, ...] = ()
    latest_model_event: str | None = None
    manual_settlement_reported: bool = False


class EvalSessionInspectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    root_session_id: str
    observed_at: datetime
    sessions: tuple[EvalSessionObservationV1, ...] = ()
    diagnostics: tuple[EvalDiagnosticV1, ...] = ()
    limitations: tuple[str, ...] = ()
    owner_liveness: Literal["not_checked"] = "not_checked"
    observation_note: str = (
        "Session state and recent activity do not prove worker liveness. "
        "Session completion is not a scored case result. "
        "Settlement observations do not authorize retry or recovery."
    )


_DIAGNOSTIC_TYPES = (
    EventType.MODEL_ERROR,
    EventType.MODEL_RETRY,
    EventType.TOOL_CALL_FAILED,
    EventType.SESSION_FAILED,
    EventType.SESSION_INTERRUPTED,
)
_MODEL_TYPES = (
    EventType.MODEL_STARTED,
    EventType.MODEL_COMPLETED,
    EventType.MODEL_ERROR,
    EventType.MODEL_RETRY,
)
_EVENT_BYTES = 1024 * 1024


def _bounded_text(value: object, maximum: int = 4096) -> str | None:
    return value[:maximum] if type(value) is str else None


def _diagnostic(event: Event, sequence: int) -> EvalDiagnosticV1:
    payload = event.payload
    result = payload.get("result")
    result = result if type(result) is dict else {}
    structured = result.get("structured")
    structured = structured if type(structured) is dict else {}
    return EvalDiagnosticV1(
        session_id=event.session_id,
        event_id=event.id,
        sequence=sequence,
        timestamp=event.timestamp,
        event_type=str(event.type),
        agent=event.agent_name,
        tool=event.tool_name,
        tool_call_id=_bounded_text(payload.get("tool_call_id"), 512),
        code=_bounded_text(payload.get("provider_error_code") or structured.get("error"), 256),
        message=_bounded_text(
            payload.get("error") or result.get("content") or payload.get("reason")
        ),
        retry_scheduled=payload.get("retry") is True
        or payload.get("retry_disposition") == "retry_scheduled"
        or event.type is EventType.MODEL_RETRY,
        manual_settlement_reported=payload.get("provider_recovery_disposition")
        == "manual_settlement_required",
    )


async def inspect_eval_sessions(
    store: SessionStore,
    root_session_id: str,
    *,
    max_sessions: int = 100,
    max_diagnostics: int = 20,
    recent_seconds: int = 120,
    now: datetime | None = None,
) -> EvalSessionInspectionV1:
    """Inspect one root and a bounded descendant set without schema-specific SQL.

    The caller owns the store and should open it read-only for operator use.
    Live records are independent observations, not a transaction-wide snapshot.
    Unsupported, missing, oversized, or truncated evidence is explicit.
    """
    if not isinstance(store, SessionStore):
        raise TypeError("store must implement SessionStore.")
    if (
        type(root_session_id) is not str
        or not root_session_id.strip()
        or len(root_session_id) > 512
    ):
        raise ValueError("root_session_id must be a nonblank identifier of at most 512 characters.")
    for value, name, maximum in (
        (max_sessions, "max_sessions", 1000),
        (max_diagnostics, "max_diagnostics", 1000),
        (recent_seconds, "recent_seconds", 86400),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be between 1 and {maximum}.")
    observed_at = now or datetime.now(UTC)
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("now must be timezone-aware.")
    pending = deque([root_session_id])
    discovered = {root_session_id}
    observations: list[EvalSessionObservationV1] = []
    diagnostics: list[EvalDiagnosticV1] = []
    limitations: set[str] = set()
    while pending:
        session_id = pending.popleft()
        try:
            summary = await store.inspect_summary(session_id)
        except Exception as exc:
            limitations.add(f"session_summary_unavailable:{type(exc).__name__}")
            continue
        identity = summary.session
        if identity.id != session_id:
            raise ValueError("Session inspection returned a different session identity.")
        age = (observed_at - identity.last_activity_at).total_seconds()
        terminal = identity.status.value in {"completed", "failed", "interrupted"}
        latest_model: Event | None = None
        try:
            records = await store.query_events_bounded(
                EventQuery(
                    session_id=session_id,
                    event_types=_MODEL_TYPES,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                ),
                max_bytes=_EVENT_BYTES,
            )
            latest_model = records[0].event if records else None
            if latest_model is not None and latest_model.session_id != session_id:
                raise ValueError("Model observation belongs to a different session.")
            records = await store.query_events_bounded(
                EventQuery(
                    session_id=session_id,
                    event_types=_DIAGNOSTIC_TYPES,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=max_diagnostics + 1,
                ),
                max_bytes=_EVENT_BYTES,
            )
            if len(records) > max_diagnostics:
                limitations.add("diagnostics_truncated")
            if any(row.event.session_id != session_id for row in records):
                raise ValueError("Diagnostics belong to a different session.")
            diagnostics.extend(
                _diagnostic(row.event, row.sequence) for row in records[:max_diagnostics]
            )
        except Exception as exc:
            limitations.add(f"event_diagnostics_unavailable:{type(exc).__name__}")
        manual = (
            not terminal
            and latest_model is not None
            and latest_model.type is EventType.MODEL_ERROR
            and latest_model.payload.get("provider_recovery_disposition")
            == "manual_settlement_required"
        )
        observations.append(
            EvalSessionObservationV1(
                session_id=identity.id,
                parent_session_id=identity.parent_session_id,
                agent=identity.agent_name,
                status=identity.status.value,
                last_activity_at=identity.last_activity_at,
                activity="terminal"
                if terminal
                else "clock_skew"
                if age < 0
                else "recent"
                if age <= recent_seconds
                else "stale",
                model_calls=summary.model_calls,
                tool_calls=summary.tool_calls,
                event_count=summary.events.record_count,
                pending_action_count=summary.pending_action_count,
                pending_action_kinds=tuple(kind.value for kind in summary.pending_action_kinds),
                latest_model_event=None if latest_model is None else str(latest_model.type),
                manual_settlement_reported=manual,
            )
        )
        cursor: str | None = None
        cursors: set[str] = set()
        while True:
            try:
                page = await store.query_session_lineage(
                    SessionLineageQuery(
                        parent_session_id=session_id,
                        cursor=cursor,
                        limit=min(100, max_sessions - len(discovered) + 1),
                    )
                )
            except Exception as exc:
                limitations.add(f"session_lineage_unavailable:{type(exc).__name__}")
                break
            if page.parent_session_id != session_id:
                raise ValueError("Session lineage returned a different parent identity.")
            for child in page.children:
                if child.parent_session_id != session_id or child.id in discovered:
                    limitations.add("session_lineage_conflict")
                    continue
                if len(discovered) >= max_sessions:
                    limitations.add("sessions_truncated")
                    break
                discovered.add(child.id)
                pending.append(child.id)
            if not page.has_more:
                break
            if len(discovered) >= max_sessions:
                limitations.add("sessions_truncated")
                break
            if not page.children or page.next_cursor is None or page.next_cursor in cursors:
                limitations.add("session_lineage_no_progress")
                break
            cursor = page.next_cursor
            cursors.add(cursor)
    diagnostics.sort(
        key=lambda item: (item.timestamp, item.session_id, item.sequence), reverse=True
    )
    if len(diagnostics) > max_diagnostics:
        limitations.add("diagnostics_truncated")
    return EvalSessionInspectionV1(
        root_session_id=root_session_id,
        observed_at=observed_at,
        sessions=tuple(observations),
        diagnostics=tuple(diagnostics[:max_diagnostics]),
        limitations=tuple(sorted(limitations)),
    )
