"""Shared classification for bounded terminal-event evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cayu._validation import copy_durable_json_object, require_clean_nonblank
from cayu.core.events import Event, EventType

TERMINAL_EVIDENCE_QUERY_LIMIT = 2
SESSION_RUN_OPERATION_ID_PAYLOAD_KEY = "session_run_operation_id"
TERMINAL_EVENT_TYPES = (
    EventType.SESSION_COMPLETED,
    EventType.SESSION_FAILED,
    EventType.SESSION_INTERRUPTED,
)
TERMINAL_LIFECYCLE_EVENT_TYPES = (
    EventType.SESSION_STARTED,
    EventType.SESSION_RESUMED,
    EventType.SESSION_FORKED,
)
TERMINAL_EVIDENCE_EVENT_TYPES = (
    *TERMINAL_LIFECYCLE_EVENT_TYPES,
    *TERMINAL_EVENT_TYPES,
)


@dataclass(frozen=True, slots=True)
class CurrentTerminalEvidence:
    """The bounded events belonging to the latest durable terminal boundary."""

    events: tuple[Event, ...]
    latest_lifecycle_event_type: EventType | str | None
    run_operation_conflict: bool = False


def interruption_request_id_from_payload(payload: dict[str, Any]) -> str | None:
    request_id = payload.get("interruption_request_id")
    if request_id is None:
        return None
    if type(request_id) is not str or not request_id.strip():
        raise ValueError("Interruption request ID must be a non-blank string.")
    return request_id


def require_interruption_event_matches_pending_marker(
    event: Event,
    marker: dict[str, Any],
) -> None:
    """Require every marker-owned field to survive terminal publication exactly.

    Terminal publication may add runtime-owned fields, so the event payload can
    be a strict superset. The pending marker is the durable authority for every
    field it does own, however, and acknowledgement readback must not clear it
    after accepting a conflicting sibling value.
    """

    if event.type is not EventType.SESSION_INTERRUPTED:
        raise RuntimeError("Published interruption evidence does not match the pending marker.")
    expected = copy_durable_json_object(marker, "pending session interrupt marker")
    actual = copy_durable_json_object(event.payload, "published interruption payload")
    if any(key not in actual or actual[key] != value for key, value in expected.items()):
        raise RuntimeError("Published interruption evidence conflicts with the pending marker.")


def classify_current_terminal_evidence(
    *,
    evidence_events: Sequence[Event],
    expected_event_type: EventType,
    run_operation_id: str | None,
    interruption_request_id: str | None,
) -> CurrentTerminalEvidence:
    """Classify newest-first evidence without conflating adjacent interruptions."""

    if len(evidence_events) > TERMINAL_EVIDENCE_QUERY_LIMIT:
        raise ValueError("Terminal evidence exceeds its bounded query.")

    latest_lifecycle_event_type: EventType | str | None = None
    events_after_lifecycle: list[Event] = []
    for event in evidence_events:
        if event.type in TERMINAL_LIFECYCLE_EVENT_TYPES:
            latest_lifecycle_event_type = event.type
            break
        events_after_lifecycle.append(event)

    candidate_events: list[Event] = []
    for event in events_after_lifecycle:
        if event.type != expected_event_type:
            break
        if event.interaction_id is not None:
            continue
        candidate_events.append(event)

    terminal_events: list[Event] = []
    if interruption_request_id is not None:
        for event in candidate_events:
            if interruption_request_id_from_payload(event.payload) != interruption_request_id:
                break
            terminal_events.append(event)
        return CurrentTerminalEvidence(
            events=tuple(terminal_events),
            latest_lifecycle_event_type=latest_lifecycle_event_type,
            run_operation_conflict=(
                run_operation_id is not None
                and any(
                    event.payload.get(SESSION_RUN_OPERATION_ID_PAYLOAD_KEY) != run_operation_id
                    for event in terminal_events
                )
            ),
        )

    if run_operation_id is not None:
        return CurrentTerminalEvidence(
            events=tuple(
                event
                for event in events_after_lifecycle
                if event.type in TERMINAL_EVENT_TYPES
                and event.interaction_id is None
                and event.payload.get(SESSION_RUN_OPERATION_ID_PAYLOAD_KEY) == run_operation_id
            ),
            latest_lifecycle_event_type=latest_lifecycle_event_type,
        )

    current_operation_id: str | None = None
    current_interrupt_request_id: str | None = None
    for event in candidate_events:
        event_operation_id = event.payload.get(SESSION_RUN_OPERATION_ID_PAYLOAD_KEY)
        event_interrupt_request_id = interruption_request_id_from_payload(event.payload)
        if not terminal_events:
            if event_operation_id is not None:
                current_operation_id = require_clean_nonblank(
                    event_operation_id,
                    "terminal event session_run_operation_id",
                )
            current_interrupt_request_id = event_interrupt_request_id
            terminal_events.append(event)
            continue
        if current_operation_id is not None:
            if event_operation_id == current_operation_id:
                terminal_events.append(event)
        elif current_interrupt_request_id is not None:
            if event_interrupt_request_id == current_interrupt_request_id:
                terminal_events.append(event)
        elif event_operation_id is None and event_interrupt_request_id is None:
            terminal_events.append(event)

    return CurrentTerminalEvidence(
        events=tuple(terminal_events),
        latest_lifecycle_event_type=latest_lifecycle_event_type,
    )
