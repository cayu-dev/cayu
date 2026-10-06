"""SQLite record representations shared by stores and schema migrations."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from cayu._validation import copy_durable_json_object, copy_label_map
from cayu.sessions.base import (
    RUNTIME_BUILD_PROVENANCE_METADATA_KEY,
    PendingActionSession,
    RunRequest,
    Session,
    SessionIdentity,
    SessionOrder,
    SessionStatus,
    runtime_build_provenance_from_session_metadata,
    session_instance_id_for_run_request,
    session_invocation_for_run_request,
    session_metadata_for_creation,
)
from cayu.sessions.invocation import SessionInvocation, TaskInvocation
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage._validated_cache import validated_row_cache
from cayu.tasks.base import TaskOrder
from cayu.tasks.contracts import WorkContractRef
from cayu.tasks.records import Task, TaskRetrySeriesSnapshot, TaskStatus
from cayu.tasks.scheduling import TaskScheduleState
from cayu.tasks.topology import (
    TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES,
    TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    TaskTopologyInconsistent,
    TaskTopologyNode,
)


def session_from_request(
    request: RunRequest,
    *,
    identity: SessionIdentity,
    parent_session: Session | None,
    created_at: datetime,
) -> Session:
    session_id = request.session_id if request.session_id is not None else str(uuid4())
    return Session(
        id=session_id,
        instance_id=session_instance_id_for_run_request(
            request,
            session_id=session_id,
        ),
        agent_name=request.agent_name,
        provider_name=identity.provider_name,
        model=identity.model,
        parent_session_id=request.parent_session_id,
        causal_budget_id=request.causal_budget_id or request.task_id or session_id,
        runtime_name=identity.runtime_name,
        runtime_version=identity.runtime_version,
        environment_name=request.environment_name,
        status=SessionStatus.PENDING,
        created_at=created_at,
        updated_at=created_at,
        last_activity_at=created_at,
        invocation=session_invocation_for_run_request(
            request,
            session_id=session_id,
            parent_session=parent_session,
        ),
        metadata=session_metadata_for_creation(
            request.metadata,
            identity=identity,
            tool_capability_ceiling=request.tool_capability_ceiling,
            execution_deadline=request.execution_deadline,
            parent_session=parent_session,
            prepared_request=request,
        ),
        labels=copy_label_map(request.labels, "labels"),
    )


def session_to_row_values(session: Session) -> tuple[object, ...]:
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
        format_datetime(session.created_at),
        format_datetime(session.updated_at),
        format_datetime(session.last_activity_at),
        session.run_epoch,
        json_dumps(session.invocation.model_dump(mode="json")),
        json_dumps(session.metadata),
    )


def session_label_row_values(session: Session) -> list[tuple[str, str, str]]:
    return [(session.id, key, value) for key, value in sorted(session.labels.items())]


def task_to_row_values(task: Task) -> tuple[object, ...]:
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
        format_optional_datetime(task.available_at),
        task.worker_id,
        format_optional_datetime(task.lease_expires_at),
        task.interrupted_handoff_id,
        task.status_reason,
        None if task.status_payload is None else json_dumps(task.status_payload),
        json_dumps(task.input),
        None if task.result is None else json_dumps(task.result),
        None if task.error is None else json_dumps(task.error),
        json_dumps(task.metadata),
        format_datetime(task.created_at),
        format_datetime(task.updated_at),
        format_optional_datetime(task.started_at),
        format_optional_datetime(task.completed_at),
        json_dumps(task.invocation.model_dump(mode="json")),
        (
            None
            if task.retry_series is None
            else json_dumps(task.retry_series.model_dump(mode="json"))
        ),
        (
            None
            if task.work_contract is None
            else json_dumps(task.work_contract.model_dump(mode="json", warnings=False))
        ),
        None if task.schedule is None else json_dumps(task.schedule.model_dump(mode="json")),
    )


@validated_row_cache
def task_from_row(row: sqlite3.Row) -> Task:
    status_payload_json = row["status_payload_json"]
    result_json = row["result_json"]
    error_json = row["error_json"]
    return Task(
        graph_id=row["graph_id"],
        prerequisite_task_ids=tuple(json.loads(row["prerequisite_task_ids_json"])),
        id=row["id"],
        type=row["type"],
        title=row["title"],
        description=row["description"],
        status=TaskStatus(row["status"]),
        session_id=row["session_id"],
        session_instance_id=row["session_instance_id"],
        parent_task_id=row["parent_task_id"],
        assigned_agent_name=row["assigned_agent_name"],
        available_at=parse_optional_datetime(row["available_at"]),
        worker_id=row["worker_id"],
        lease_expires_at=parse_optional_datetime(row["lease_expires_at"]),
        interrupted_handoff_id=row["interrupted_handoff_id"],
        status_reason=row["status_reason"],
        status_payload=(None if status_payload_json is None else json.loads(status_payload_json)),
        input=json.loads(row["input_json"]),
        result=None if result_json is None else json.loads(result_json),
        error=None if error_json is None else json.loads(error_json),
        metadata=json.loads(row["metadata_json"]),
        created_at=parse_datetime(row["created_at"]),
        updated_at=parse_datetime(row["updated_at"]),
        started_at=parse_optional_datetime(row["started_at"]),
        completed_at=parse_optional_datetime(row["completed_at"]),
        invocation=TaskInvocation.model_validate(json.loads(row["invocation_json"])),
        retry_series=(
            None
            if row["retry_series_json"] is None
            else TaskRetrySeriesSnapshot.model_validate(json.loads(row["retry_series_json"]))
        ),
        work_contract=(
            None
            if row["work_contract_json"] is None
            else WorkContractRef.model_validate(json.loads(row["work_contract_json"]))
        ),
        schedule=(
            None
            if row["schedule_json"] is None
            else TaskScheduleState.model_validate(json.loads(row["schedule_json"]))
        ),
    )


_TASK_TOPOLOGY_MAX_TIMESTAMP_BYTES = 128

TASK_TOPOLOGY_COLUMNS = f"""
    CASE
        WHEN length(CAST(id AS BLOB)) <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        THEN id
    END AS topology_id,
    length(CAST(id AS BLOB)) > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        AS topology_id_oversized,
    CASE
        WHEN length(CAST(type AS BLOB)) <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN type
    END AS topology_type,
    length(CAST(type AS BLOB)) > {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        AS topology_type_truncated,
    CASE
        WHEN title IS NULL
          OR length(CAST(title AS BLOB)) <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN title
    END AS topology_title,
    title IS NOT NULL
      AND length(CAST(title AS BLOB)) > {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        AS topology_title_truncated,
    CASE
        WHEN length(CAST(status AS BLOB)) <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN status
    END AS topology_status,
    CASE
        WHEN status_reason IS NULL
          OR length(CAST(status_reason AS BLOB)) <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN status_reason
    END AS topology_status_reason,
    status_reason IS NOT NULL
      AND length(CAST(status_reason AS BLOB)) > {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        AS topology_status_reason_truncated,
    CASE
        WHEN session_id IS NULL
          OR length(CAST(session_id AS BLOB)) <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        THEN session_id
    END AS topology_session_id,
    session_id IS NOT NULL
      AND length(CAST(session_id AS BLOB)) > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        AS topology_session_id_oversized,
    CASE
        WHEN parent_task_id IS NULL
          OR length(CAST(parent_task_id AS BLOB)) <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        THEN parent_task_id
    END AS topology_parent_task_id,
    parent_task_id IS NOT NULL
      AND length(CAST(parent_task_id AS BLOB)) > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        AS topology_parent_task_id_oversized,
    CASE
        WHEN assigned_agent_name IS NULL
          OR length(CAST(assigned_agent_name AS BLOB))
             <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN assigned_agent_name
    END AS topology_assigned_agent_name,
    assigned_agent_name IS NOT NULL
      AND length(CAST(assigned_agent_name AS BLOB))
          > {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        AS topology_assigned_agent_name_truncated,
    CASE
        WHEN length(CAST(created_at AS BLOB)) <= {_TASK_TOPOLOGY_MAX_TIMESTAMP_BYTES}
        THEN created_at
    END AS topology_created_at,
    CASE
        WHEN length(CAST(updated_at AS BLOB)) <= {_TASK_TOPOLOGY_MAX_TIMESTAMP_BYTES}
        THEN updated_at
    END AS topology_updated_at
"""


def task_topology_node_from_row(row: sqlite3.Row) -> TaskTopologyNode:
    if (
        row["topology_id_oversized"]
        or row["topology_session_id_oversized"]
        or row["topology_parent_task_id_oversized"]
    ):
        raise TaskTopologyInconsistent(
            "A task topology record contains an oversized structural identifier."
        )
    truncated_fields = tuple(
        field_name
        for field_name, column_name in (
            ("type", "topology_type_truncated"),
            ("title", "topology_title_truncated"),
            ("assigned_agent_name", "topology_assigned_agent_name_truncated"),
            ("status_reason", "topology_status_reason_truncated"),
        )
        if row[column_name]
    )
    try:
        return TaskTopologyNode(
            id=row["topology_id"],
            type=row["topology_type"],
            title=row["topology_title"],
            status=TaskStatus(row["topology_status"]),
            status_reason=row["topology_status_reason"],
            session_id=row["topology_session_id"],
            parent_task_id=row["topology_parent_task_id"],
            assigned_agent_name=row["topology_assigned_agent_name"],
            created_at=parse_datetime(row["topology_created_at"]),
            updated_at=parse_datetime(row["topology_updated_at"]),
            truncated_fields=truncated_fields,
        )
    except (TypeError, ValueError) as exc:
        raise TaskTopologyInconsistent(
            "A task record cannot be represented by the bounded topology contract."
        ) from exc


@validated_row_cache
def session_from_row(row: sqlite3.Row, labels: dict[str, str] | None = None) -> Session:
    return Session(
        id=row["id"],
        instance_id=row["instance_id"],
        agent_name=row["agent_name"],
        provider_name=row["provider_name"],
        model=row["model"],
        parent_session_id=row["parent_session_id"],
        causal_budget_id=row["causal_budget_id"],
        runtime_name=row["runtime_name"],
        runtime_version=row["runtime_version"],
        environment_name=row["environment_name"],
        status=SessionStatus(row["status"]),
        created_at=parse_datetime(row["created_at"]),
        updated_at=parse_datetime(row["updated_at"]),
        last_activity_at=parse_datetime(row["last_activity_at"]),
        run_epoch=row["run_epoch"],
        invocation=SessionInvocation.model_validate(json.loads(row["invocation_json"])),
        metadata=json.loads(row["metadata_json"]),
        labels=copy_label_map(labels, "labels"),
    )


def pending_action_session_from_row(
    row: sqlite3.Row,
    labels: dict[str, str] | None = None,
) -> PendingActionSession:
    return PendingActionSession(
        id=row["id"],
        instance_id=row["instance_id"],
        agent_name=row["agent_name"],
        provider_name=row["provider_name"],
        model=row["model"],
        parent_session_id=row["parent_session_id"],
        causal_budget_id=row["causal_budget_id"],
        runtime_name=row["runtime_name"],
        runtime_version=row["runtime_version"],
        runtime_build_provenance=runtime_build_provenance_from_session_metadata(
            {}
            if row["runtime_build_provenance_json"] is None
            else {
                RUNTIME_BUILD_PROVENANCE_METADATA_KEY: json.loads(
                    row["runtime_build_provenance_json"]
                )
            }
        ),
        environment_name=row["environment_name"],
        status=SessionStatus(row["status"]),
        created_at=parse_datetime(row["created_at"]),
        updated_at=parse_datetime(row["updated_at"]),
        labels=copy_label_map(labels, "labels"),
    )


def format_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def format_optional_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return format_datetime(value)


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_optional_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return parse_datetime(value)


def json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def checkpoint_row_values(
    session_id: str,
    checkpoint: dict[str, Any],
    updated_at: datetime,
) -> tuple[object, ...]:
    from cayu.sessions.pending_actions import pending_action_checkpoint_metrics

    checkpoint = copy_durable_json_object(checkpoint, "checkpoint")
    source_bytes, tool_call_count, flags = pending_action_checkpoint_metrics(checkpoint)
    return (
        session_id,
        json_dumps(checkpoint),
        format_datetime(updated_at),
        source_bytes,
        tool_call_count,
        flags,
        1,
    )


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
