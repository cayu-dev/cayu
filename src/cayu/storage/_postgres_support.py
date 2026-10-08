from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from cayu._validation import copy_label_map
from cayu.sessions.base import SessionTopologyNode
from cayu.sessions.invocation import SessionInvocation, TaskInvocation
from cayu.sessions.queries import SessionOrder
from cayu.sessions.records import (
    RUNTIME_BUILD_PROVENANCE_METADATA_KEY,
    PendingActionSession,
    Session,
    SessionStatus,
    runtime_build_provenance_from_session_metadata,
)
from cayu.storage import _session_store_sql as session_store_sql
from cayu.tasks.contracts import WorkContractRef
from cayu.tasks.queries import TaskOrder
from cayu.tasks.records import Task, TaskRetrySeriesSnapshot, TaskStatus
from cayu.tasks.scheduling import TaskScheduleState
from cayu.tasks.topology import (
    TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES,
    TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    TaskTopologyInconsistent,
    TaskTopologyNode,
)


def to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def to_utc_optional(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return to_utc(value)


def session_insert_values(session: Session) -> tuple[object, ...]:
    return (
        session.id,
        session.instance_id,
        session.agent_name,
        session.provider_name,
        session.model,
        session.parent_session_id,
        session.causal_budget_id,
        session.runtime_name,
        session.runtime_version,
        session.environment_name,
        str(session.status),
        to_utc(session.created_at),
        to_utc(session.updated_at),
        to_utc(session.last_activity_at),
        session.run_epoch,
        _dumps(session.invocation.model_dump(mode="json")),
        _dumps(session.metadata),
    )


def session_label_insert_values(session: Session) -> list[tuple[str, str, str]]:
    return [(session.id, key, value) for key, value in sorted(session.labels.items())]


def session_from_row(row: tuple[Any, ...], labels: dict[str, str] | None = None) -> Session:
    return Session(
        id=row[0],
        instance_id=row[1],
        agent_name=row[2],
        provider_name=row[3],
        model=row[4],
        parent_session_id=row[5],
        causal_budget_id=row[6],
        runtime_name=row[7],
        runtime_version=row[8],
        environment_name=row[9],
        status=SessionStatus(row[10]),
        created_at=to_utc(row[11]),
        updated_at=to_utc(row[12]),
        last_activity_at=to_utc(row[13]),
        run_epoch=row[14],
        invocation=SessionInvocation.model_validate(_loads(row[15])),
        metadata=_loads(row[16]),
        labels=copy_label_map(labels, "labels"),
    )


SESSION_COLUMNS = (
    "id, instance_id, agent_name, provider_name, model, parent_session_id, causal_budget_id, "
    "runtime_name, runtime_version, environment_name, status, created_at, updated_at, "
    "last_activity_at, run_epoch, invocation, metadata"
)

SESSION_TOPOLOGY_COLUMNS = (
    "id, agent_name, provider_name, model, parent_session_id, causal_budget_id, "
    "runtime_name, runtime_version, environment_name, status, created_at, updated_at, "
    "last_activity_at, metadata -> 'cayu:runtime_build_provenance' "
    "AS runtime_build_provenance"
)


def session_topology_node_from_row(row: tuple[Any, ...]) -> SessionTopologyNode:
    return SessionTopologyNode(
        id=row[0],
        agent_name=row[1],
        provider_name=row[2],
        model=row[3],
        parent_session_id=row[4],
        causal_budget_id=row[5],
        runtime_name=row[6],
        runtime_version=row[7],
        runtime_build_provenance=runtime_build_provenance_from_session_metadata(
            {} if row[13] is None else {RUNTIME_BUILD_PROVENANCE_METADATA_KEY: row[13]}
        ),
        environment_name=row[8],
        status=SessionStatus(row[9]),
        created_at=to_utc(row[10]),
        updated_at=to_utc(row[11]),
        last_activity_at=to_utc(row[12]),
    )


PENDING_ACTION_SESSION_COLUMNS = (
    "id, agent_name, provider_name, model, parent_session_id, causal_budget_id, "
    "runtime_name, runtime_version, environment_name, status, created_at, updated_at"
)


def pending_action_session_from_row(
    row: tuple[Any, ...],
    labels: dict[str, str] | None = None,
) -> PendingActionSession:
    return PendingActionSession(
        id=row[0],
        instance_id=row[13],
        agent_name=row[1],
        provider_name=row[2],
        model=row[3],
        parent_session_id=row[4],
        causal_budget_id=row[5],
        runtime_name=row[6],
        runtime_version=row[7],
        runtime_build_provenance=runtime_build_provenance_from_session_metadata(
            {} if row[12] is None else {RUNTIME_BUILD_PROVENANCE_METADATA_KEY: row[12]}
        ),
        environment_name=row[8],
        status=SessionStatus(row[9]),
        created_at=to_utc(row[10]),
        updated_at=to_utc(row[11]),
        labels=copy_label_map(labels, "labels"),
    )


def task_insert_values(task: Task) -> tuple[object, ...]:
    return (
        task.id,
        task.type,
        task.title,
        task.description,
        str(task.status),
        task.session_id,
        task.session_instance_id,
        task.parent_task_id,
        task.assigned_agent_name,
        to_utc_optional(task.available_at),
        task.worker_id,
        to_utc_optional(task.lease_expires_at),
        task.interrupted_handoff_id,
        task.status_reason,
        None if task.status_payload is None else _dumps(task.status_payload),
        _dumps(task.input),
        None if task.result is None else _dumps(task.result),
        None if task.error is None else _dumps(task.error),
        _dumps(task.metadata),
        to_utc(task.created_at),
        to_utc(task.updated_at),
        to_utc_optional(task.started_at),
        to_utc_optional(task.completed_at),
        _dumps(task.invocation.model_dump(mode="json")),
        (None if task.retry_series is None else _dumps(task.retry_series.model_dump(mode="json"))),
        (
            None
            if task.work_contract is None
            else _dumps(task.work_contract.model_dump(mode="json", warnings=False))
        ),
        None if task.schedule is None else _dumps(task.schedule.model_dump(mode="json")),
        task.graph_id,
        _dumps(list(task.prerequisite_task_ids)),
    )


TASK_COLUMNS = (
    "id, type, title, description, status, session_id, session_instance_id, parent_task_id, "
    "assigned_agent_name, available_at, worker_id, lease_expires_at, interrupted_handoff_id, "
    "status_reason, status_payload, input, result, error, metadata, created_at, updated_at, "
    "started_at, completed_at, invocation, retry_series, work_contract, schedule, "
    "graph_id, prerequisite_task_ids"
)


def task_from_row(row: tuple[Any, ...]) -> Task:
    return Task(
        id=row[0],
        type=row[1],
        title=row[2],
        description=row[3],
        status=TaskStatus(row[4]),
        session_id=row[5],
        session_instance_id=row[6],
        parent_task_id=row[7],
        assigned_agent_name=row[8],
        available_at=to_utc_optional(row[9]),
        worker_id=row[10],
        lease_expires_at=to_utc_optional(row[11]),
        interrupted_handoff_id=row[12],
        status_reason=row[13],
        status_payload=None if row[14] is None else _loads(row[14]),
        input=_loads(row[15]),
        result=None if row[16] is None else _loads(row[16]),
        error=None if row[17] is None else _loads(row[17]),
        metadata=_loads(row[18]),
        created_at=to_utc(row[19]),
        updated_at=to_utc(row[20]),
        started_at=to_utc_optional(row[21]),
        completed_at=to_utc_optional(row[22]),
        invocation=TaskInvocation.model_validate(_loads(row[23])),
        retry_series=(
            None if row[24] is None else TaskRetrySeriesSnapshot.model_validate(_loads(row[24]))
        ),
        work_contract=(
            None if row[25] is None else WorkContractRef.model_validate(_loads(row[25]))
        ),
        schedule=None if row[26] is None else TaskScheduleState.model_validate(_loads(row[26])),
        graph_id=row[27],
        prerequisite_task_ids=tuple(_loads(row[28])),
    )


TASK_TOPOLOGY_COLUMNS = f"""
    CASE
        WHEN octet_length(id) <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        THEN id
    END AS topology_id,
    octet_length(id) > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        AS topology_id_oversized,
    CASE
        WHEN octet_length(type) <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN type
    END AS topology_type,
    octet_length(type) > {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        AS topology_type_truncated,
    CASE
        WHEN title IS NULL
          OR octet_length(title) <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN title
    END AS topology_title,
    title IS NOT NULL
      AND octet_length(title) > {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        AS topology_title_truncated,
    CASE
        WHEN octet_length(status) <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN status
    END AS topology_status,
    CASE
        WHEN status_reason IS NULL
          OR octet_length(status_reason) <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN status_reason
    END AS topology_status_reason,
    status_reason IS NOT NULL
      AND octet_length(status_reason) > {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        AS topology_status_reason_truncated,
    CASE
        WHEN session_id IS NULL
          OR octet_length(session_id) <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        THEN session_id
    END AS topology_session_id,
    session_id IS NOT NULL
      AND octet_length(session_id) > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        AS topology_session_id_oversized,
    CASE
        WHEN parent_task_id IS NULL
          OR octet_length(parent_task_id) <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        THEN parent_task_id
    END AS topology_parent_task_id,
    parent_task_id IS NOT NULL
      AND octet_length(parent_task_id) > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        AS topology_parent_task_id_oversized,
    CASE
        WHEN assigned_agent_name IS NULL
          OR octet_length(assigned_agent_name)
             <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN assigned_agent_name
    END AS topology_assigned_agent_name,
    assigned_agent_name IS NOT NULL
      AND octet_length(assigned_agent_name)
          > {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        AS topology_assigned_agent_name_truncated,
    created_at AS topology_created_at,
    updated_at AS topology_updated_at
"""


def task_topology_node_from_row(row: tuple[Any, ...]) -> TaskTopologyNode:
    if row[1] or row[10] or row[12]:
        raise TaskTopologyInconsistent(
            "A task topology record contains an oversized structural identifier."
        )
    truncated_fields = tuple(
        field_name
        for field_name, truncated in (
            ("type", row[3]),
            ("title", row[5]),
            ("assigned_agent_name", row[14]),
            ("status_reason", row[8]),
        )
        if truncated
    )
    try:
        return TaskTopologyNode(
            id=row[0],
            type=row[2],
            title=row[4],
            status=TaskStatus(row[6]),
            status_reason=row[7],
            session_id=row[9],
            parent_task_id=row[11],
            assigned_agent_name=row[13],
            created_at=to_utc(row[15]),
            updated_at=to_utc(row[16]),
            truncated_fields=truncated_fields,
        )
    except (TypeError, ValueError) as exc:
        raise TaskTopologyInconsistent(
            "A task record cannot be represented by the bounded topology contract."
        ) from exc


def session_order_sql(order_by: SessionOrder) -> str:
    return session_store_sql.session_order_sql(order_by)


def task_order_sql(order_by: TaskOrder) -> str:
    if order_by == TaskOrder.CREATED_AT_ASC:
        return "created_at ASC"
    if order_by == TaskOrder.CREATED_AT_DESC:
        return "created_at DESC"
    if order_by == TaskOrder.UPDATED_AT_ASC:
        return "updated_at ASC"
    return "updated_at DESC"


def _dumps(value: Any) -> str:
    # JSONB columns accept a JSON-text string; we serialize explicitly so the
    # same json round-trip semantics as the SQLite store are preserved.
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _loads(value: Any) -> Any:
    # psycopg returns JSONB as already-decoded Python objects, but we accept a
    # JSON string too for robustness across configurations.
    if isinstance(value, str):
        return json.loads(value)
    return value
