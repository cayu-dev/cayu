"""Shared child-session lifecycle observations and exact notification consumption."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from cayu.events import EventType
from cayu.runtime._child_session_notifications import (
    CHILD_SESSION_ADMISSION_OCCURRENCE_TYPE,
    ChildSessionLifecycleEntry,
    ChildSessionLifecycleOccurrence,
    ChildSessionLifecycleOccurrenceSource,
    ChildSessionLifecycleState,
    ChildSessionNotificationConsumption,
    ChildSessionNotificationFreshness,
    ChildSessionRelationship,
    child_session_notification_stage_binding,
)
from cayu.sessions._model_stage import ModelCompletionStage, SessionModelCompletionStageConflict
from cayu.sessions.records import EventRecord, Session, SessionStatus

_CHILD_LIFECYCLE_EVENT_TYPE_BY_TERMINAL_STATUS = {
    SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
    SessionStatus.FAILED: EventType.SESSION_FAILED,
    SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
}


def _child_session_lifecycle_state(status: SessionStatus) -> ChildSessionLifecycleState:
    if status is SessionStatus.PENDING:
        return ChildSessionLifecycleState.ADMITTED
    if status in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
        return ChildSessionLifecycleState.RUNNING
    return {
        SessionStatus.COMPLETED: ChildSessionLifecycleState.COMPLETED,
        SessionStatus.FAILED: ChildSessionLifecycleState.FAILED,
        SessionStatus.INTERRUPTED: ChildSessionLifecycleState.INTERRUPTED,
    }[status]


def _child_session_lifecycle_occurrence(
    child: Session,
    records_by_type: Mapping[EventType, EventRecord],
) -> tuple[ChildSessionRelationship, ChildSessionLifecycleOccurrence] | None:
    fork_record = records_by_type.get(EventType.SESSION_FORKED)
    relationship = (
        ChildSessionRelationship.SESSION_FORK
        if fork_record is not None
        else ChildSessionRelationship.DIRECT_CHILD
    )
    started_record = records_by_type.get(EventType.SESSION_STARTED)
    resumed_record = records_by_type.get(EventType.SESSION_RESUMED)
    activation_record = max(
        (record for record in (started_record, resumed_record, fork_record) if record is not None),
        key=lambda record: record.sequence,
        default=None,
    )
    terminal_record = max(
        (
            record
            for event_type in _CHILD_LIFECYCLE_EVENT_TYPE_BY_TERMINAL_STATUS.values()
            if (record := records_by_type.get(event_type)) is not None
        ),
        key=lambda record: record.sequence,
        default=None,
    )
    latest_lifecycle_record = max(
        (record for record in (activation_record, terminal_record) if record is not None),
        key=lambda record: record.sequence,
        default=None,
    )
    terminal_type = _CHILD_LIFECYCLE_EVENT_TYPE_BY_TERMINAL_STATUS.get(child.status)
    if terminal_type is not None:
        record = latest_lifecycle_record
        if record is None or record.event.type is not terminal_type:
            return None
    elif child.status in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
        if latest_lifecycle_record is None or latest_lifecycle_record is not activation_record:
            return None
        record = latest_lifecycle_record
    else:
        if started_record is not None or resumed_record is not None or terminal_record is not None:
            return None
        record = fork_record
    if record is None:
        if child.status is SessionStatus.PENDING:
            return relationship, ChildSessionLifecycleOccurrence(
                source=ChildSessionLifecycleOccurrenceSource.SESSION,
                source_id=child.instance_id,
                source_type=CHILD_SESSION_ADMISSION_OCCURRENCE_TYPE,
                occurred_at=child.created_at,
            )
        return None
    return relationship, ChildSessionLifecycleOccurrence(
        source=ChildSessionLifecycleOccurrenceSource.EVENT,
        source_id=record.event.id,
        source_sequence=record.sequence,
        source_type=str(record.event.type),
        occurred_at=record.event.timestamp,
    )


def _child_session_notification_consumption(
    record: Mapping[str, Any] | None,
    *,
    parent: Session,
    child: Session,
    occurrence: ChildSessionLifecycleOccurrence,
) -> ChildSessionNotificationConsumption | None:
    if record is None:
        return None
    try:
        consumption = ChildSessionNotificationConsumption.model_validate(record)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Child-session notification consumption is malformed.") from exc
    if (
        consumption.parent_session_id != parent.id
        or consumption.parent_session_instance_id != parent.instance_id
        or consumption.child_session_id != child.id
        or consumption.child_session_instance_id != child.instance_id
        or consumption.occurrence != occurrence
    ):
        raise RuntimeError("Child-session notification consumption changed its authority.")
    return consumption


def _child_session_lifecycle_entry(
    *,
    parent: Session,
    child: Session,
    records_by_type: Mapping[EventType, EventRecord],
    consumption_record: Mapping[str, Any] | None,
) -> ChildSessionLifecycleEntry | None:
    if child.parent_session_id != parent.id:
        raise RuntimeError("Child-session lifecycle candidate changed its parent authority.")
    source = _child_session_lifecycle_occurrence(child, records_by_type)
    if source is None:
        return None
    relationship, occurrence = source
    state = _child_session_lifecycle_state(child.status)
    consumption = (
        _child_session_notification_consumption(
            consumption_record,
            parent=parent,
            child=child,
            occurrence=occurrence,
        )
        if state.terminal
        else None
    )
    return ChildSessionLifecycleEntry(
        parent_session_id=parent.id,
        parent_session_instance_id=parent.instance_id,
        child_session_id=child.id,
        child_session_instance_id=child.instance_id,
        relationship=relationship,
        state=state,
        occurrence=occurrence,
        freshness=(
            ChildSessionNotificationFreshness.CURRENT
            if not state.terminal
            else ChildSessionNotificationFreshness.FRESH
            if consumption is None
            else ChildSessionNotificationFreshness.CONSUMED
        ),
        consumed_by_stage_id=None if consumption is None else consumption.stage_id,
        created_at=child.created_at,
        updated_at=child.updated_at,
    )


def _child_session_lifecycle_entry_sort_key(
    entry: ChildSessionLifecycleEntry,
) -> tuple[int, datetime, str]:
    priority = (
        0
        if entry.freshness is ChildSessionNotificationFreshness.FRESH
        else 1
        if entry.freshness is ChildSessionNotificationFreshness.CURRENT
        else 2
    )
    return priority, entry.occurrence.occurred_at, entry.child_session_id


def _child_session_notification_consumption_record(
    *,
    parent: Session,
    child: Session,
    occurrence: ChildSessionLifecycleOccurrence,
    stage: ModelCompletionStage,
    consumed_at: datetime,
) -> ChildSessionNotificationConsumption:
    binding = child_session_notification_stage_binding(stage.intent)
    if binding is None:
        raise ValueError("Model stage has no child-session notification binding.")
    if (
        binding.parent_session_id != parent.id
        or binding.parent_session_instance_id != parent.instance_id
        or stage.session_id != parent.id
    ):
        raise SessionModelCompletionStageConflict(
            "Child-session notification binding changed its parent authority."
        )
    claim = next(
        (
            candidate
            for candidate in binding.claims
            if candidate.child_session_id == child.id
            and candidate.occurrence.source_id == occurrence.source_id
        ),
        None,
    )
    if (
        claim is None
        or claim.child_session_instance_id != child.instance_id
        or claim.occurrence != occurrence
        or child.parent_session_id != parent.id
        or not _child_session_lifecycle_state(child.status).terminal
        or occurrence.source is not ChildSessionLifecycleOccurrenceSource.EVENT
        or occurrence.source_type
        != str(_CHILD_LIFECYCLE_EVENT_TYPE_BY_TERMINAL_STATUS.get(child.status))
    ):
        raise SessionModelCompletionStageConflict(
            "Child-session notification claim no longer matches canonical terminal state."
        )
    model_step_id = stage.intent.get("model_step_id")
    model_attempt_id = stage.intent.get("model_attempt_id")
    if type(model_step_id) is not str or type(model_attempt_id) is not str:
        raise SessionModelCompletionStageConflict(
            "Child-session notification binding lost its model-attempt identity."
        )
    return ChildSessionNotificationConsumption(
        parent_session_id=parent.id,
        parent_session_instance_id=parent.instance_id,
        child_session_id=child.id,
        child_session_instance_id=child.instance_id,
        occurrence=occurrence,
        stage_id=stage.stage_id,
        model_step_id=model_step_id,
        model_attempt_id=model_attempt_id,
        source_run_epoch=stage.source_run_epoch,
        consumed_at=consumed_at,
    )


def _child_session_notification_consumption_replays(
    existing: Mapping[str, Any],
    requested: ChildSessionNotificationConsumption,
) -> bool:
    """Allow transport retries only within the same logical model step."""

    try:
        durable = ChildSessionNotificationConsumption.model_validate(existing)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Child-session notification consumption is malformed.") from exc
    return (
        durable.parent_session_id == requested.parent_session_id
        and durable.parent_session_instance_id == requested.parent_session_instance_id
        and durable.child_session_id == requested.child_session_id
        and durable.child_session_instance_id == requested.child_session_instance_id
        and durable.occurrence == requested.occurrence
        and durable.model_step_id == requested.model_step_id
        and durable.source_run_epoch == requested.source_run_epoch
    )
