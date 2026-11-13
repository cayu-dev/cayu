"""SQLite record codecs and cursor-bound row readers shared by native operations."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from cayu._validation import copy_durable_json_object, copy_label_map
from cayu.deadlines import (
    EXECUTION_DEADLINE_METADATA_KEY,
    EXECUTION_DEADLINE_ORIGIN_METADATA_KEY,
    TASK_EVENT_DEADLINE_METADATA_KEY,
    ExecutionDeadlineReport,
)
from cayu.events import Event, event_from_trusted_store_record
from cayu.messages import Message
from cayu.sessions.creation import session_invocation_for_run_request, session_metadata_for_creation
from cayu.sessions.event_delivery import (
    persisted_event_authority_candidates,
    restore_persisted_event_authority,
)
from cayu.sessions.invocation import SessionInvocation, TaskInvocation
from cayu.sessions.queries import SessionOrder
from cayu.sessions.records import (
    RUNTIME_BUILD_PROVENANCE_METADATA_KEY,
    EventRecord,
    PendingActionSession,
    Session,
    SessionIdentity,
    SessionStatus,
    runtime_build_provenance_from_session_metadata,
    stored_event_record,
)
from cayu.sessions.requests import RunRequest, session_instance_id_for_run_request
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage._validated_cache import validated_row_cache
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
        int(task.restartable),
        task.restart_count,
        task.concurrency_key,
        task.max_restarts,
        restart_history_json(task),
        task.claim_generation,
    )


def restart_history_json(task: Task) -> str:
    return json_dumps([record.model_dump(mode="json") for record in task.restart_history])


@validated_row_cache
def task_from_row(row: sqlite3.Row) -> Task:
    status_payload_json = row["status_payload_json"]
    result_json = row["result_json"]
    error_json = row["error_json"]
    return Task(
        restartable=bool(row["restartable"]),
        restart_count=row["restart_count"],
        concurrency_key=row["concurrency_key"],
        max_restarts=row["max_restarts"],
        restart_history=tuple(json.loads(row["restart_history_json"])),
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
        claim_generation=row["claim_generation"],
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
        execution_deadline=ExecutionDeadlineReport.from_metadata(
            {
                **(
                    {}
                    if row["execution_deadline_json"] is None
                    else {
                        EXECUTION_DEADLINE_METADATA_KEY: json.loads(row["execution_deadline_json"])
                    }
                ),
                EXECUTION_DEADLINE_ORIGIN_METADATA_KEY: row["execution_deadline_origin"],
                **(
                    {}
                    if row["task_event_deadline_json"] is None
                    else {
                        TASK_EVENT_DEADLINE_METADATA_KEY: json.loads(
                            row["task_event_deadline_json"]
                        )
                    }
                ),
            },
            top_level=row["parent_session_id"] is None,
        ),
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


def transcript_row(
    session_id: str,
    interaction_id: Any,
    message: Message,
    payload: dict[str, Any] | None = None,
    *,
    stored_payload: dict[str, Any] | None = None,
) -> tuple[str, str, Any, str, str, int]:
    """Values for one ``cayu_transcript_messages`` insert, including its stored size.

    ``stored_payload`` is the form written when tool results reference their
    tool event. The recorded size stays that of the full message, so bounded
    reads budget for what a read returns.
    """

    from cayu.sessions.transcript_queries import transcript_search_document

    message_json = json_dumps(message.model_dump(mode="json") if payload is None else payload)
    message_bytes = len(message_json.encode("utf-8"))
    if stored_payload is not None and stored_payload is not payload:
        message_json = json_dumps(stored_payload)
    return (
        session_id,
        str(message.role),
        interaction_id,
        message_json,
        transcript_search_document(message),
        message_bytes,
    )


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
    *,
    owned: bool = False,
) -> tuple[object, ...]:
    """Serialize one checkpoint row.

    ``owned`` is only for a checkpoint the calling statement just produced with
    ``copy_durable_json_object`` and has not shared, so copying it again would
    repeat the same validation on the same private object. A row that refers
    to its receipt ledger is checked as the whole logical checkpoint.
    """

    from cayu.sessions.pending_actions import pending_action_checkpoint_metrics
    from cayu.storage._receipt_ledger_rows import require_logical_checkpoint_limits

    if not owned:
        checkpoint = copy_durable_json_object(checkpoint, "checkpoint")
    serialized = json_dumps(checkpoint)
    require_logical_checkpoint_limits(checkpoint, serialized)
    source_bytes, tool_call_count, flags = pending_action_checkpoint_metrics(checkpoint)
    return (
        session_id,
        serialized,
        format_datetime(updated_at),
        source_bytes,
        tool_call_count,
        flags,
        1,
    )


@validated_row_cache
def checkpoint_from_json(value: str) -> dict[str, Any]:
    return copy_durable_json_object(json.loads(value), "checkpoint")


CheckpointRow = tuple[str, str | None, str | None]


def load_checkpoint_row(connection: sqlite3.Connection, session_id: str) -> CheckpointRow | None:
    """The checkpoint row and the receipt ledger row it refers to, in one read."""

    row = connection.execute(
        "SELECT checkpoint.state_json, ledger.record_sha256, ledger.ledger_json "
        "FROM cayu_checkpoints AS checkpoint "
        "LEFT JOIN cayu_invocation_receipt_ledgers AS ledger "
        "ON ledger.session_id = checkpoint.session_id AND ledger.record_sha256 = "
        "json_extract(checkpoint.state_json, '$.invocation_lifecycle_receipt.record_sha256') "
        "WHERE checkpoint.session_id = ?",
        (session_id,),
    ).fetchone()
    return None if row is None else (row[0], row[1], row[2])


def checkpoint_from_row(row: CheckpointRow) -> dict[str, Any]:
    """The logical checkpoint: the stored row with its receipt ledger joined back."""

    from cayu.storage._receipt_ledger_rows import join_receipt_ledger, ledger_reference_digest

    state_json, ledger_digest, ledger_json = row
    checkpoint = checkpoint_from_json(state_json)
    if ledger_reference_digest(checkpoint) is None:
        return checkpoint
    return join_receipt_ledger(
        checkpoint,
        ledger_digest,
        None if ledger_json is None else checkpoint_from_json(ledger_json),
    )


def load_checkpoint(connection: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    """The session's logical checkpoint, its receipt ledger joined from its own row."""

    row = load_checkpoint_row(connection, session_id)
    return None if row is None else checkpoint_from_row(row)


def checkpoint_row(
    connection: sqlite3.Connection,
    session_id: str,
    checkpoint: dict[str, Any],
    updated_at: datetime,
    *,
    owned: bool = False,
) -> tuple[object, ...]:
    """Serialize one checkpoint row, its receipt ledger stored in its own row.

    The ledger row is written in the caller's transaction, before the
    checkpoint row it is serialized for.
    """

    row = store_receipt_ledger(connection, session_id, checkpoint)
    return checkpoint_row_values(session_id, row, updated_at, owned=owned)


def store_receipt_ledger(
    connection: sqlite3.Connection, session_id: str, checkpoint: dict[str, Any]
) -> dict[str, Any]:
    """Keep a checkpoint's receipt ledger in its own row; return the row to store.

    Ledger rows are keyed by their content digest, so storing one never
    changes what the session's current checkpoint row refers to. A new
    version replaces every other one except the version that row refers to,
    so a session keeps at most two.
    """

    from cayu.storage._receipt_ledger_rows import split_receipt_ledger

    row, ledger = split_receipt_ledger(checkpoint)
    if ledger is None:
        return checkpoint
    digest = ledger["record_sha256"]
    inserted = connection.execute(
        "INSERT INTO cayu_invocation_receipt_ledgers (session_id, record_sha256, ledger_json) "
        "VALUES (?, ?, ?) ON CONFLICT(session_id, record_sha256) DO NOTHING",
        (session_id, digest, json_dumps(ledger)),
    ).rowcount
    if inserted:
        connection.execute(
            "DELETE FROM cayu_invocation_receipt_ledgers "
            "WHERE session_id = ? AND record_sha256 <> ? AND record_sha256 IS NOT ("
            "SELECT json_extract(state_json, '$.invocation_lifecycle_receipt.record_sha256') "
            "FROM cayu_checkpoints WHERE session_id = ?)",
            (session_id, digest, session_id),
        )
    return row


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


def load_session(connection: sqlite3.Connection, session_id: str) -> Session | None:
    row = connection.execute(
        """
        SELECT id, instance_id, agent_name, provider_name, model, parent_session_id,
               causal_budget_id, runtime_name, runtime_version, environment_name,
               status, created_at, updated_at, last_activity_at, run_epoch,
               invocation_json, metadata_json
        FROM cayu_sessions
        WHERE id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return session_from_row(
        row,
        labels=load_session_labels(connection, session_id),
    )


def load_session_labels(connection: sqlite3.Connection, session_id: str) -> dict[str, str]:
    rows = connection.execute(
        """
        SELECT key, value
        FROM cayu_session_labels
        WHERE session_id = ?
        ORDER BY key ASC
        """,
        (session_id,),
    ).fetchall()
    return {row["key"]: row["value"] for row in rows}


def session_exists(connection: sqlite3.Connection, session_id: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM cayu_sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    return row is not None


# Columns needed to reconstruct an Event, in a stable order. The formerly-stored
# event_json blob duplicated exactly these (plus payload_json), so the store now
# rebuilds Events from the individual columns instead of parsing a redundant copy.
EVENT_COLUMN_NAMES: tuple[str, ...] = (
    "session_id",
    "event_id",
    "interaction_id",
    "event_type",
    "timestamp",
    "agent_name",
    "environment_name",
    "workflow_name",
    "tool_name",
    "payload_json",
    "input_contract_runtime_owned",
    "file_attachment_attestations_runtime_owned",
)


def _stored_authority_flags(row: sqlite3.Row) -> tuple[bool, bool]:
    input_contract_runtime_owned = row["input_contract_runtime_owned"]
    if type(input_contract_runtime_owned) is not int or input_contract_runtime_owned not in {
        0,
        1,
    }:
        raise ValueError("Stored input-contract authority proof is malformed.")
    file_attachment_attestations_runtime_owned = row["file_attachment_attestations_runtime_owned"]
    if type(
        file_attachment_attestations_runtime_owned
    ) is not int or file_attachment_attestations_runtime_owned not in {0, 1}:
        raise ValueError("Stored file-attachment attestation proof is malformed.")
    return input_contract_runtime_owned == 1, file_attachment_attestations_runtime_owned == 1


def event_from_row(row: sqlite3.Row) -> Event:
    """Rebuild a stored event this store's ingestion validated, without re-walking it.

    ``event_from_trusted_store_record`` lists the checks that still run. The
    authority flags keep their exact checks, so a tampered flag is still
    rejected, and the store's private authority is restored without a copy.
    """

    input_contract_runtime_owned, file_attachment_attestations_runtime_owned = (
        _stored_authority_flags(row)
    )
    return event_from_trusted_store_record(
        event_type=row["event_type"],
        session_id=row["session_id"],
        interaction_id=row["interaction_id"],
        event_id=row["event_id"],
        timestamp=row["timestamp"],
        agent_name=row["agent_name"],
        environment_name=row["environment_name"],
        workflow_name=row["workflow_name"],
        tool_name=row["tool_name"],
        payload_json=row["payload_json"],
        runtime_payload_authority_candidates=persisted_event_authority_candidates(
            row["event_type"],
            input_contract_runtime_owned=input_contract_runtime_owned,
            file_attachment_attestations_runtime_owned=file_attachment_attestations_runtime_owned,
        ),
    )


def validating_event_from_row(row: sqlite3.Row) -> Event:
    """Rebuild a stored event through full durable validation.

    The full-validation reference for :func:`event_from_row`, used by
    equivalence tests and store verification.
    """

    input_contract_runtime_owned, file_attachment_attestations_runtime_owned = (
        _stored_authority_flags(row)
    )
    return restore_persisted_event_authority(
        Event(
            type=row["event_type"],
            session_id=row["session_id"],
            interaction_id=row["interaction_id"],
            id=row["event_id"],
            timestamp=row["timestamp"],
            agent_name=row["agent_name"],
            environment_name=row["environment_name"],
            workflow_name=row["workflow_name"],
            tool_name=row["tool_name"],
            payload=json.loads(row["payload_json"]),
        ),
        input_contract_runtime_owned=input_contract_runtime_owned,
        file_attachment_attestations_runtime_owned=file_attachment_attestations_runtime_owned,
    )


def event_record_from_row(row: sqlite3.Row | None) -> EventRecord | None:
    if row is None:
        return None
    return stored_event_record_from_row(row)


def stored_event_record_from_row(row: sqlite3.Row) -> EventRecord:
    """The record of one stored event row, without the defensive event copy."""

    return stored_event_record(row["sequence"], event_from_row(row))


def load_session_labels_batch(
    connection: sqlite3.Connection, session_ids: list[str]
) -> dict[str, dict[str, str]]:
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = connection.execute(
        f"""
        SELECT session_id, key, value
        FROM cayu_session_labels
        WHERE session_id IN ({placeholders})
        ORDER BY session_id ASC, key ASC
        """,
        session_ids,
    ).fetchall()
    labels_by_session_id: dict[str, dict[str, str]] = {session_id: {} for session_id in session_ids}
    for row in rows:
        labels_by_session_id[row["session_id"]][row["key"]] = row["value"]
    return labels_by_session_id
