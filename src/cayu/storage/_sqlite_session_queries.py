"""Complete SQLite session query operations with explicit native execution capabilities."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from cayu._clock import utc_duration_cutoff
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu.budgets.aggregates import UsageRollupStoreResult
from cayu.budgets.pricing import PriceBook
from cayu.events import (
    EVENT_ID_MAX_CHARS,
    Event,
    EventType,
)
from cayu.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.runtime._child_session_notifications import (
    ChildSessionLifecyclePage,
    ChildSessionLifecycleQuery,
    child_session_notification_storage_key,
)
from cayu.runtime._cost_accounting import CostAccountingSnapshot
from cayu.runtime._usage_accounting import UsageAccountingSnapshot
from cayu.sessions.access import (
    runtime_session_query,
)
from cayu.sessions.base import (
    _TOOL_ROUND_LIFECYCLE_EVENT_TYPES,
    _child_session_lifecycle_entry,
    _child_session_lifecycle_entry_sort_key,
    _child_session_lifecycle_occurrence,
    _tool_round_lifecycle_event_limit,
    _validate_interaction_page,
    _validate_tool_round_call_ids,
)
from cayu.sessions.event_queries import EventQuery, EventQueryResultTooLarge, copy_event_query
from cayu.sessions.inspection import SESSION_INSPECTION_LABEL_LIMIT, SessionInspectionIdentity
from cayu.sessions.lineage import (
    SESSION_LINEAGE_MAX_EVENT_ID_BYTES,
    SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
    SESSION_LINEAGE_MAX_ORIGIN_EVENTS,
    SESSION_LINEAGE_MAX_TIMESTAMP_BYTES,
    SessionLineageNode,
    SessionLineageOrigin,
    SessionLineageQuery,
    SessionLineageResult,
    copy_session_lineage_query,
    decode_session_lineage_cursor,
    encode_session_lineage_cursor,
)
from cayu.sessions.queries import (
    SessionListResult,
    SessionQuery,
    copy_session_query,
    session_next_cursor,
    session_query_from_aggregate_filter,
)
from cayu.sessions.records import (
    RUNTIME_BUILD_PROVENANCE_METADATA_KEY,
    EventRecord,
    Session,
    SessionStatus,
    runtime_build_provenance_from_session_metadata,
)
from cayu.sessions.summaries import (
    EventSummary,
)
from cayu.sessions.topology import (
    SessionTopologyCycle,
    SessionTopologyDepthExceeded,
    SessionTopologyNode,
    SessionTopologyQuery,
    SessionTopologyStoreResult,
    build_session_topology_result,
    decode_session_topology_cursor,
)
from cayu.sessions.usage import UsageRollupQuery, copy_usage_rollup_query
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage import _sqlite_aggregates as sqlite_aggregates
from cayu.storage import _sqlite_records as sqlite_records

if TYPE_CHECKING:
    from cayu.runtime._cost_accounting_refresh import CostAccountingAuthority
    from cayu.runtime._usage_accounting import SessionUsageCache
    from cayu.sessions.access import _SessionAccessBounds


_ReadResult = TypeVar("_ReadResult")


class SQLiteReadRunner(Protocol):
    """Execute one read while retaining native connection and cancellation ownership."""

    def __call__(
        self, query: Callable[[sqlite3.Connection], _ReadResult], /
    ) -> Awaitable[_ReadResult]: ...


_EVENT_QUERY_SESSION_IDS_BATCH_SIZE = 500


SQL_DIALECT = session_store_sql.SessionStoreSqlDialect(
    placeholder="?",
    contains_style="sqlite_nocase_like",
    datetime_param=sqlite_records.format_datetime,
)


# Keep this predicate text aligned with the revision-17 partial index. SQLite
# can prove a parameterized lifecycle subset is covered by that index only when
# the query also carries the index's literal predicate.
PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL = """
    event_type IN (
        'tool.call.approval_requested',
        'session.awaiting_user_input',
        'session.interrupted',
        'session.delegated_action.updated',
        'tool.call.started',
        'tool.call.completed',
        'tool.call.failed',
        'tool.call.blocked',
        'tool.call.approval_denied'
    )
    AND pending_action_lookup_key IS NOT NULL
"""


def _event_query_session_id_batches(
    session_ids: tuple[str, ...],
) -> list[tuple[str, ...]]:
    return [
        session_ids[index : index + _EVENT_QUERY_SESSION_IDS_BATCH_SIZE]
        for index in range(0, len(session_ids), _EVENT_QUERY_SESSION_IDS_BATCH_SIZE)
    ]


_SESSION_TOPOLOGY_COLUMNS = """
    id, agent_name, provider_name, model, parent_session_id,
    causal_budget_id, runtime_name, runtime_version, environment_name,
    status, created_at, updated_at, last_activity_at,
    json_extract(metadata_json, '$."cayu:runtime_build_provenance"')
        AS runtime_build_provenance_json
"""


_SESSION_TOPOLOGY_PROJECTED_COLUMNS = """
    id, agent_name, provider_name, model, parent_session_id,
    causal_budget_id, runtime_name, runtime_version, environment_name,
    status, created_at, updated_at, last_activity_at,
    runtime_build_provenance_json
"""


def _session_topology_node_from_sqlite_row(row: sqlite3.Row) -> SessionTopologyNode:
    return SessionTopologyNode(
        id=row["id"],
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
        created_at=sqlite_records.parse_datetime(row["created_at"]),
        updated_at=sqlite_records.parse_datetime(row["updated_at"]),
        last_activity_at=sqlite_records.parse_datetime(row["last_activity_at"]),
    )


async def inspect_identity(
    run_read: SQLiteReadRunner, session_id: str
) -> SessionInspectionIdentity:
    session_id = require_clean_nonblank(session_id, "session_id")

    def query(connection: sqlite3.Connection) -> SessionInspectionIdentity:
        row = connection.execute(
            """
            SELECT id, agent_name, provider_name, model, parent_session_id,
                   causal_budget_id, runtime_name, runtime_version, environment_name,
                   status, created_at, updated_at, last_activity_at, run_epoch,
                   json_extract(
                       metadata_json,
                       '$."cayu:runtime_build_provenance"'
                   ) AS runtime_build_provenance_json
            FROM cayu_sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        label_rows = connection.execute(
            """
            SELECT key, value,
                   (SELECT COUNT(*)
                    FROM cayu_session_labels
                    WHERE session_id = ?) AS label_count
            FROM cayu_session_labels
            WHERE session_id = ?
            ORDER BY key ASC
            LIMIT ?
            """,
            (session_id, session_id, SESSION_INSPECTION_LABEL_LIMIT),
        ).fetchall()
        label_count = 0 if not label_rows else label_rows[0]["label_count"]
        return SessionInspectionIdentity(
            id=row["id"],
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
            created_at=sqlite_records.parse_datetime(row["created_at"]),
            updated_at=sqlite_records.parse_datetime(row["updated_at"]),
            last_activity_at=sqlite_records.parse_datetime(row["last_activity_at"]),
            run_epoch=row["run_epoch"],
            labels={label_row["key"]: label_row["value"] for label_row in label_rows},
            label_count=label_count,
            labels_truncated=label_count > len(label_rows),
        )

    return await run_read(query)


async def load_events(run_read: SQLiteReadRunner, session_id: str) -> list[Event]:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")

    def query(connection: sqlite3.Connection) -> list[sqlite3.Row]:
        if not sqlite_records.session_exists(connection, session_id):
            raise KeyError(f"Session not found: {session_id}")
        rows = connection.execute(
            f"""
            SELECT {", ".join(sqlite_records.EVENT_COLUMN_NAMES)}
            FROM cayu_events
            WHERE session_id = ?
            ORDER BY sequence ASC
            """,
            (session_id,),
        ).fetchall()
        return rows

    from cayu.storage._session_access_records import sqlite_owner_read

    rows = await run_read(
        lambda connection: sqlite_owner_read(connection, access_bounds, session_id, query)
    )
    return await asyncio.to_thread(lambda: [sqlite_records.event_from_row(row) for row in rows])


async def load_user_input_supersession_events(
    run_read: SQLiteReadRunner, session_id: str, input_id: str
) -> list[Event]:
    from cayu.sessions.pending_actions import pending_action_lookup_key

    session_id = require_clean_nonblank(session_id, "session_id")
    input_id = require_clean_nonblank(input_id, "input_id")
    lookup_key = pending_action_lookup_key(input_id)

    def query(connection: sqlite3.Connection) -> list[Event]:
        if not sqlite_records.session_exists(connection, session_id):
            raise KeyError(f"Session not found: {session_id}")
        rows = connection.execute(
            f"SELECT {', '.join(sqlite_records.EVENT_COLUMN_NAMES)} FROM cayu_events "
            "INDEXED BY idx_cayu_events_pending_action_lookup "
            "WHERE session_id = ? AND pending_action_lookup_key = ? "
            "AND event_type = ? AND "
            f"({PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
            "AND json_extract(payload_json, "
            "'$.user_input_supersession_intent.input_id') = ? "
            "ORDER BY sequence ASC LIMIT 2",
            (
                session_id,
                lookup_key,
                str(EventType.SESSION_INTERRUPTED),
                input_id,
            ),
        ).fetchall()
        return [sqlite_records.event_from_row(row) for row in rows]

    return await run_read(query)


async def load_tool_round_lifecycle_events(
    run_read: SQLiteReadRunner, session_id: str, tool_call_ids: list[str] | tuple[str, ...]
) -> list[Event]:
    from cayu.sessions.pending_actions import pending_action_lookup_key

    session_id = require_clean_nonblank(session_id, "session_id")
    copied_ids = _validate_tool_round_call_ids(tool_call_ids, "tool_call_ids")
    lookup_keys = tuple(pending_action_lookup_key(call_id) for call_id in copied_ids)
    lifecycle_event_types = tuple(
        sorted(str(event_type) for event_type in _TOOL_ROUND_LIFECYCLE_EVENT_TYPES)
    )

    def query(connection: sqlite3.Connection) -> list[Event]:
        if not sqlite_records.session_exists(connection, session_id):
            raise KeyError(f"Session not found: {session_id}")
        event_type_placeholders = ", ".join("?" for _ in lifecycle_event_types)
        rows = connection.execute(
            f"SELECT {', '.join(sqlite_records.EVENT_COLUMN_NAMES)} FROM cayu_events "
            "INDEXED BY idx_cayu_events_pending_action_lookup "
            f"WHERE session_id = ? AND pending_action_lookup_key IN "
            "(SELECT value FROM json_each(?)) AND event_type IN "
            f"({event_type_placeholders}) AND "
            f"({PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
            "ORDER BY sequence ASC LIMIT ?",
            (
                session_id,
                json.dumps(lookup_keys),
                *lifecycle_event_types,
                _tool_round_lifecycle_event_limit(copied_ids) + 1,
            ),
        ).fetchall()
        if len(rows) > _tool_round_lifecycle_event_limit(copied_ids):
            raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
        return [sqlite_records.event_from_row(row) for row in rows]

    return await run_read(query)


async def load_tool_round_lifecycle_events_for_round(
    run_read: SQLiteReadRunner,
    session_id: str,
    tool_call_ids: list[str] | tuple[str, ...],
    *,
    tool_round_identity: ToolRoundIdentity,
) -> list[Event]:
    from cayu.sessions.pending_actions import pending_action_lookup_key

    session_id = require_clean_nonblank(session_id, "session_id")
    copied_ids = _validate_tool_round_call_ids(tool_call_ids, "tool_call_ids")
    tool_round_identity = copy_tool_round_identity(tool_round_identity)
    lookup_keys = tuple(pending_action_lookup_key(call_id) for call_id in copied_ids)
    lifecycle_event_types = tuple(
        sorted(str(event_type) for event_type in _TOOL_ROUND_LIFECYCLE_EVENT_TYPES)
    )

    def query(connection: sqlite3.Connection) -> list[Event]:
        if not sqlite_records.session_exists(connection, session_id):
            raise KeyError(f"Session not found: {session_id}")
        event_type_placeholders = ", ".join("?" for _ in lifecycle_event_types)
        rows = connection.execute(
            f"SELECT {', '.join(sqlite_records.EVENT_COLUMN_NAMES)} FROM cayu_events "
            "INDEXED BY idx_cayu_events_pending_action_lookup "
            f"WHERE session_id = ? AND pending_action_lookup_key IN "
            "(SELECT value FROM json_each(?)) AND event_type IN "
            f"({event_type_placeholders}) AND "
            f"({PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
            "AND (json_extract(payload_json, '$.tool_round_id') = ? "
            "OR (json_extract(payload_json, '$.model_step_id') = ? "
            "AND json_extract(payload_json, '$.model_attempt_id') = ?) "
            "OR cayu_is_execution_unit_id("
            "json_extract(payload_json, '$.tool_round_id'), 'tool_round_id') = 0 "
            "OR cayu_is_execution_unit_id("
            "json_extract(payload_json, '$.model_step_id'), 'model_step_id') = 0 "
            "OR cayu_is_execution_unit_id("
            "json_extract(payload_json, '$.model_attempt_id'), 'model_attempt_id') = 0) "
            "ORDER BY sequence ASC LIMIT ?",
            (
                session_id,
                json.dumps(lookup_keys),
                *lifecycle_event_types,
                tool_round_identity.tool_round_id,
                tool_round_identity.model_step_id,
                tool_round_identity.model_attempt_id,
                _tool_round_lifecycle_event_limit(copied_ids) + 1,
            ),
        ).fetchall()
        if len(rows) > _tool_round_lifecycle_event_limit(copied_ids):
            raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
        return [sqlite_records.event_from_row(row) for row in rows]

    return await run_read(query)


@runtime_session_query
async def query_events(
    run_read: SQLiteReadRunner, query: EventQuery | None = None
) -> list[EventRecord]:
    query = copy_event_query(query)
    if len(query.session_ids) > _EVENT_QUERY_SESSION_IDS_BATCH_SIZE:
        return await _query_events_by_session_id_batches(run_read, query)

    plan = session_store_sql.build_event_query_sql(query, dialect=SQL_DIALECT)
    params = [*plan.params, query.limit]

    def run_query(connection: sqlite3.Connection) -> list[EventRecord]:
        event_columns = ", ".join(
            f"cayu_events.{name}" for name in sqlite_records.EVENT_COLUMN_NAMES
        )
        rows = connection.execute(
            f"""
            SELECT cayu_events.sequence, {event_columns}
            FROM cayu_events
            JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
            {plan.where_sql}
            ORDER BY cayu_events.sequence {plan.order_direction}
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [
            EventRecord(sequence=row["sequence"], event=sqlite_records.event_from_row(row))
            for row in rows
        ]

    return await run_read(run_query)


@runtime_session_query
async def event_exists(run_read: SQLiteReadRunner, query: EventQuery) -> bool:
    plan = session_store_sql.build_accounting_event_query_sql(query, dialect=SQL_DIALECT)

    def read(connection: sqlite3.Connection) -> bool:
        return (
            connection.execute(
                "SELECT EXISTS(SELECT 1 FROM cayu_events "
                "JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id "
                f"{plan.where_sql})",
                plan.params,
            ).fetchone()[0]
            == 1
        )

    return await run_read(read)


@runtime_session_query
async def read_usage_accounting(
    run_read: SQLiteReadRunner,
    query: EventQuery,
    *,
    by_session: bool = False,
    by_identity: bool = False,
    usage_cache: SessionUsageCache,
) -> UsageAccountingSnapshot:
    from cayu.runtime._usage_accounting import (
        USAGE_ACCOUNTING_PAGE_SIZE,
        SessionUsageCache,
        UsageAccountingReducer,
        usage_accounting_query,
    )

    query = usage_accounting_query(query)
    plan = session_store_sql.build_accounting_event_query_sql(query, dialect=SQL_DIALECT)
    cached_session_id = SessionUsageCache.session_scope(
        query, by_session=by_session, by_identity=by_identity
    )

    def read(connection: sqlite3.Connection) -> UsageAccountingSnapshot:
        with connection:
            connection.execute("BEGIN")
            generation_row = connection.execute(
                "SELECT generation FROM cayu_accounting_state WHERE singleton = 1"
            ).fetchone()
            if generation_row is None:
                raise RuntimeError("Accounting deletion revision is missing.")
            generation = generation_row[0]
            read_query = query
            read_plan = plan
            prior = None
            boundary = 0
            if cached_session_id is not None:
                prior = usage_cache.resume_after(cached_session_id, generation)
                if prior is not None:
                    # A cached scope carries no access bounds, so this plan
                    # adds only the sequence cursor to the one built above.
                    read_query = copy_event_query(
                        query, update={"after_sequence": prior.scanned_through}
                    )
                    read_plan = session_store_sql.build_accounting_event_query_sql(
                        read_query, dialect=SQL_DIALECT
                    )
                # SQLite serializes writers, so later commits land above this.
                boundary = (
                    connection.execute(
                        "SELECT MAX(sequence) FROM cayu_events WHERE session_id = ?",
                        (cached_session_id,),
                    ).fetchone()[0]
                    or 0
                )
            reducer = UsageAccountingReducer(
                read_query, by_session=by_session, by_identity=by_identity
            )
            event_columns = ", ".join(
                f"cayu_events.{name}" for name in sqlite_records.EVENT_COLUMN_NAMES
            )
            # One statement holds one read snapshot. fetchmany bounds hydration;
            # query.limit is a page size, never a cap on authoritative history.
            cursor = connection.execute(
                f"SELECT cayu_events.sequence, {event_columns} FROM cayu_events "
                "JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id "
                f"{read_plan.where_sql} ORDER BY cayu_events.sequence ASC",
                read_plan.params,
            )
            try:
                while rows := cursor.fetchmany(USAGE_ACCOUNTING_PAGE_SIZE):
                    reducer.add_page(
                        [
                            EventRecord(
                                sequence=row["sequence"], event=sqlite_records.event_from_row(row)
                            )
                            for row in rows
                        ]
                    )
            finally:
                cursor.close()
            if cached_session_id is not None:
                return usage_cache.settle(
                    cached_session_id,
                    generation,
                    prior,
                    reducer.snapshot(),
                    boundary=boundary,
                )
            return reducer.snapshot().model_copy(update={"generation": generation})

    return await run_read(read)


@runtime_session_query
async def read_cost_accounting(
    run_read: SQLiteReadRunner,
    query: EventQuery,
    pricing: PriceBook,
    *,
    currency: str = "USD",
    details: bool = False,
    by_session: bool = False,
    additional_events: tuple[Event, ...] = (),
    max_detail_bytes: int | None = None,
    previous: CostAccountingSnapshot | None = None,
    cost_authority: CostAccountingAuthority,
) -> CostAccountingSnapshot:
    from cayu.runtime._cost_accounting import (
        COST_ACCOUNTING_PAGE_SIZE,
        cost_accounting_query,
        cost_pending_events,
    )
    from cayu.runtime._cost_accounting_refresh import CostAccountingRead
    from cayu.storage._cost_accounting_sql import (
        changed_cost_groups_statement,
        cost_boundary_statement,
        cost_group_lookup_statement,
        cost_group_statement,
    )

    query = cost_accounting_query(query)
    pending = cost_pending_events(query, additional_events)
    plan = session_store_sql.build_accounting_event_query_sql(query, dialect=SQL_DIALECT)

    def read(connection: sqlite3.Connection) -> CostAccountingSnapshot:
        with connection:
            connection.execute("BEGIN")
            generation_row = connection.execute(
                "SELECT generation FROM cayu_accounting_state WHERE singleton = 1"
            ).fetchone()
            if generation_row is None:
                raise RuntimeError("Accounting deletion revision is missing.")
            generation = generation_row[0]
            scoped_pending = pending
            if query.causal_budget_id is not None and pending:
                allowed_sessions = {
                    row[0]
                    for row in connection.execute(
                        "SELECT id FROM cayu_sessions WHERE causal_budget_id = ? "
                        "AND id IN (SELECT value FROM json_each(?))",
                        (
                            query.causal_budget_id,
                            json.dumps([event.session_id for event in pending]),
                        ),
                    )
                }
                scoped_pending = tuple(
                    event for event in pending if event.session_id in allowed_sessions
                )
            scoped_pending = tuple(
                event
                for event in scoped_pending
                if connection.execute(
                    "SELECT 1 FROM cayu_events "
                    "JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id "
                    f"{plan.where_sql} AND cayu_events.session_id = ? AND cayu_events.event_id = ? LIMIT 1",
                    (*plan.params, event.session_id, event.id),
                ).fetchone()
                is None
            )
            boundary_sql, boundary_params = cost_boundary_statement(query, dialect=SQL_DIALECT)
            boundary = connection.execute(boundary_sql, boundary_params).fetchone()[0] or 0
            reducer = CostAccountingRead(
                query,
                pricing,
                currency=currency,
                details=details,
                max_detail_bytes=max_detail_bytes,
                previous=previous,
                generation=generation,
                through_sequence=boundary,
                authority=cost_authority,
                by_session=by_session,
                additional_events=scoped_pending,
            )
            source_plan = session_store_sql.build_accounting_event_query_sql(
                reducer.source_query, dialect=SQL_DIALECT
            )
            event_columns = ", ".join(
                f"cayu_events.{name}" for name in sqlite_records.EVENT_COLUMN_NAMES
            )
            columns = f"cayu_events.sequence, {event_columns}"

            def add_group(key: tuple[str, bool, str]) -> None:
                statement, params = cost_group_lookup_statement(
                    columns=columns, plan=source_plan, key=key, postgres=False
                )
                cursor = connection.execute(statement, params)
                try:
                    while rows := cursor.fetchmany(COST_ACCOUNTING_PAGE_SIZE):
                        for row in rows:
                            reducer.add(row["sequence"], sqlite_records.event_from_row(row))
                finally:
                    cursor.close()

            if reducer.incremental:
                statement, params = changed_cost_groups_statement(reducer, dialect=SQL_DIALECT)
                cursor = connection.execute(statement, params)
                try:
                    while groups := cursor.fetchmany(COST_ACCOUNTING_PAGE_SIZE):
                        for group in groups:
                            add_group(
                                (
                                    group[0],
                                    group[1] is not None,
                                    group[1] if group[1] is not None else group[2],
                                )
                            )
                finally:
                    cursor.close()
                for key in reducer.remaining_pending_keys:
                    add_group(key)
            else:
                statement, group_params = cost_group_statement(
                    columns=columns, where_sql=source_plan.where_sql, postgres=False
                )
                cursor = connection.execute(statement, (*group_params, *source_plan.params))
                try:
                    while rows := cursor.fetchmany(COST_ACCOUNTING_PAGE_SIZE):
                        for row in rows:
                            reducer.add(row["sequence"], sqlite_records.event_from_row(row))
                finally:
                    cursor.close()
            return reducer.snapshot()

    return await run_read(read)


@runtime_session_query
async def query_events_bounded(
    run_read: SQLiteReadRunner, query: EventQuery, *, max_bytes: int
) -> list[EventRecord]:
    query = copy_event_query(query)
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer.")
    if len(query.session_ids) > _EVENT_QUERY_SESSION_IDS_BATCH_SIZE:
        raise ValueError("Byte-bounded event queries require one bounded SQL batch.")
    plan = session_store_sql.build_event_query_sql(query, dialect=SQL_DIALECT)
    params = [*plan.params, query.limit]

    def run_query(connection: sqlite3.Connection) -> list[EventRecord]:
        event_columns = ", ".join(
            f"cayu_events.{name}" for name in sqlite_records.EVENT_COLUMN_NAMES
        )
        serialized_bytes = " + ".join(
            [
                "256",
                *(
                    f"COALESCE(length(CAST(cayu_events.{name} AS BLOB)), 0)"
                    for name in sqlite_records.EVENT_COLUMN_NAMES
                ),
            ]
        )
        connection.execute("BEGIN")
        try:
            size_row = connection.execute(
                f"""
                WITH bounded_candidates AS (
                    SELECT {serialized_bytes} AS serialized_bytes
                    FROM cayu_events
                    JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                    {plan.where_sql}
                    ORDER BY cayu_events.sequence {plan.order_direction}
                    LIMIT ?
                )
                SELECT COALESCE(SUM(serialized_bytes), 0)
                FROM bounded_candidates
                """,
                params,
            ).fetchone()
            if size_row is None or int(size_row[0]) > max_bytes:
                raise EventQueryResultTooLarge(max_bytes)
            rows = connection.execute(
                f"""
                SELECT cayu_events.sequence, {event_columns}
                FROM cayu_events
                JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                {plan.where_sql}
                ORDER BY cayu_events.sequence {plan.order_direction}
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [
                EventRecord(sequence=row["sequence"], event=sqlite_records.event_from_row(row))
                for row in rows
            ]
        finally:
            connection.rollback()

    return await run_read(run_query)


async def query_latest_interaction_events(
    run_read: SQLiteReadRunner,
    session_id: str,
    *,
    before_sequence: int | None = None,
    limit: int = 100,
) -> list[EventRecord]:
    session_id = require_clean_nonblank(session_id, "session_id")
    before_sequence, limit = _validate_interaction_page(before_sequence, limit)
    cursor_clause = "" if before_sequence is None else "AND latest.latest_event_sequence < ?"
    params: list[object] = [session_id]
    if before_sequence is not None:
        params.append(before_sequence)
    params.append(limit)

    def run_query(connection: sqlite3.Connection) -> list[EventRecord]:
        if not sqlite_records.session_exists(connection, session_id):
            raise KeyError(f"Session not found: {session_id}")
        event_columns = ", ".join(f"event.{name}" for name in sqlite_records.EVENT_COLUMN_NAMES)
        rows = connection.execute(
            f"""
            SELECT event.sequence, {event_columns}
            FROM cayu_interaction_latest_events AS latest
            JOIN cayu_events AS event
              ON event.sequence = latest.latest_event_sequence
            WHERE latest.session_id = ? {cursor_clause}
            ORDER BY latest.latest_event_sequence DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [
            EventRecord(sequence=row["sequence"], event=sqlite_records.event_from_row(row))
            for row in rows
        ]

    return await run_read(run_query)


async def summarize_events(run_read: SQLiteReadRunner, session_id: str) -> EventSummary:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")

    def query(connection: sqlite3.Connection) -> EventSummary:
        if not sqlite_records.session_exists(connection, session_id):
            raise KeyError(f"Session not found: {session_id}")

        total_row = connection.execute(
            """
            SELECT COUNT(*) AS total_events
            FROM cayu_events
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        count_rows = connection.execute(
            """
            SELECT event_type, COUNT(*) AS count
            FROM cayu_events
            WHERE session_id = ?
            GROUP BY event_type
            ORDER BY event_type ASC
            """,
            (session_id,),
        ).fetchall()
        latest_row = connection.execute(
            f"""
            SELECT sequence, {", ".join(sqlite_records.EVENT_COLUMN_NAMES)}
            FROM cayu_events
            WHERE session_id = ?
            ORDER BY sequence DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()

        return EventSummary(
            session_id=session_id,
            total_events=int(total_row["total_events"]),
            counts_by_type={row["event_type"]: int(row["count"]) for row in count_rows},
            latest_event=sqlite_records.event_record_from_row(latest_row),
        )

    from cayu.storage._session_access_records import sqlite_owner_read

    return await run_read(
        lambda connection: sqlite_owner_read(connection, access_bounds, session_id, query)
    )


async def query_session_topology(
    run_read: SQLiteReadRunner, query: SessionTopologyQuery
) -> SessionTopologyStoreResult:
    if type(query) is not SessionTopologyQuery:
        raise TypeError("Session topology queries must be SessionTopologyQuery instances.")
    query = query.model_copy(deep=True)

    def read_topology_snapshot(
        connection: sqlite3.Connection,
    ) -> SessionTopologyStoreResult:
        focus_row = connection.execute(
            f"""
            SELECT {_SESSION_TOPOLOGY_COLUMNS}
            FROM cayu_sessions
            WHERE id = ?
            """,
            (query.focus_session_id,),
        ).fetchone()
        if focus_row is None:
            raise KeyError(f"Session not found: {query.focus_session_id}")
        focus = _session_topology_node_from_sqlite_row(focus_row)

        ancestors: list[SessionTopologyNode] = []
        seen_ids = {focus.id}
        parent_session_id = focus.parent_session_id
        while parent_session_id is not None:
            if parent_session_id in seen_ids:
                raise SessionTopologyCycle(
                    f"Session topology contains a parent cycle at {parent_session_id}."
                )
            if len(ancestors) >= query.ancestor_depth_limit:
                raise SessionTopologyDepthExceeded(
                    f"Session topology exceeds the {query.ancestor_depth_limit}-ancestor limit."
                )
            parent_row = connection.execute(
                f"""
                SELECT {_SESSION_TOPOLOGY_COLUMNS}
                FROM cayu_sessions
                WHERE id = ?
                """,
                (parent_session_id,),
            ).fetchone()
            if parent_row is None:
                raise ValueError(f"Session topology references missing parent {parent_session_id}.")
            parent = _session_topology_node_from_sqlite_row(parent_row)
            ancestors.append(parent)
            seen_ids.add(parent.id)
            parent_session_id = parent.parent_session_id
        ancestors.reverse()

        expanded_parents: list[SessionTopologyNode] = []
        if query.expanded_parent_ids:
            placeholders = ", ".join("?" for _ in query.expanded_parent_ids)
            parent_rows = connection.execute(
                f"""
                SELECT {_SESSION_TOPOLOGY_COLUMNS}
                FROM cayu_sessions
                WHERE id IN ({placeholders})
                """,
                query.expanded_parent_ids,
            ).fetchall()
            parents_by_id = {
                row["id"]: _session_topology_node_from_sqlite_row(row) for row in parent_rows
            }
            for parent_id in query.expanded_parent_ids:
                parent = parents_by_id.get(parent_id)
                if parent is None:
                    raise KeyError(f"Session not found: {parent_id}")
                expanded_parents.append(parent)

        candidates_by_parent: dict[str, list[SessionTopologyNode]] = {
            parent.id: [] for parent in expanded_parents
        }
        if expanded_parents:
            branch_queries: list[str] = []
            branch_params: list[object] = []
            for branch_order, parent in enumerate(expanded_parents):
                cursor = query.child_cursors.get(parent.id)
                if cursor is None:
                    cursor_clause = ""
                    cursor_params: list[object] = []
                else:
                    cursor_created_at, cursor_id = decode_session_topology_cursor(
                        cursor,
                        parent_session_id=parent.id,
                    )
                    cursor_clause = "AND (created_at > ? OR (created_at = ? AND id > ?))"
                    formatted_cursor = sqlite_records.format_datetime(cursor_created_at)
                    cursor_params = [formatted_cursor, formatted_cursor, cursor_id]
                branch_queries.append(
                    f"""
                    SELECT branch_order, {_SESSION_TOPOLOGY_PROJECTED_COLUMNS}
                    FROM (
                        SELECT ? AS branch_order, {_SESSION_TOPOLOGY_COLUMNS}
                        FROM cayu_sessions
                        WHERE parent_session_id = ?
                          {cursor_clause}
                        ORDER BY created_at ASC, id ASC
                        LIMIT ?
                    )
                    """
                )
                branch_params.extend(
                    [
                        branch_order,
                        parent.id,
                        *cursor_params,
                        query.child_limit + 1,
                    ]
                )
            candidate_rows = connection.execute(
                f"""
                {" UNION ALL ".join(branch_queries)}
                ORDER BY branch_order ASC, created_at ASC, id ASC
                """,
                branch_params,
            ).fetchall()
            for row in candidate_rows:
                candidates_by_parent[row["parent_session_id"]].append(
                    _session_topology_node_from_sqlite_row(row)
                )

        result = build_session_topology_result(
            focus=focus,
            ancestors=ancestors,
            expanded_parents=expanded_parents,
            branch_candidates=(candidates_by_parent[parent.id] for parent in expanded_parents),
            child_limit=query.child_limit,
        )
        return result

    def read_topology(connection: sqlite3.Connection) -> SessionTopologyStoreResult:
        # Multiple point reads plus the batched child query must describe one
        # SQLite snapshot. A plain sequence of SELECT statements in Python's
        # legacy transaction mode would otherwise observe commits made
        # between statements.
        connection.execute("BEGIN")
        try:
            return read_topology_snapshot(connection)
        finally:
            connection.rollback()

    return await run_read(read_topology)


async def query_session_lineage(
    run_read: SQLiteReadRunner, query: SessionLineageQuery
) -> SessionLineageResult:
    query = copy_session_lineage_query(query)

    def read_lineage_snapshot(connection: sqlite3.Connection) -> SessionLineageResult:
        parent_exists = connection.execute(
            "SELECT 1 FROM cayu_sessions WHERE id = ?",
            (query.parent_session_id,),
        ).fetchone()
        if parent_exists is None:
            raise KeyError(f"Session not found: {query.parent_session_id}")

        cursor_clause = ""
        params: list[object] = [
            SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
            SESSION_LINEAGE_MAX_TIMESTAMP_BYTES,
            query.parent_session_id,
        ]
        if query.cursor is not None:
            cursor_created_at, cursor_id = decode_session_lineage_cursor(
                query.cursor,
                parent_session_id=query.parent_session_id,
            )
            formatted_cursor = sqlite_records.format_datetime(cursor_created_at)
            cursor_clause = "AND (created_at > ? OR (created_at = ? AND id > ?))"
            params.extend((formatted_cursor, formatted_cursor, cursor_id))
        params.append(query.limit + 1)
        rows = connection.execute(
            f"""
            SELECT CASE
                       WHEN length(CAST(id AS BLOB)) <= ? THEN id
                   END AS id,
                   CASE
                       WHEN length(CAST(created_at AS BLOB)) <= ? THEN created_at
                   END AS created_at
            FROM cayu_sessions
            WHERE parent_session_id = ?
              {cursor_clause}
            ORDER BY created_at ASC, id ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
        retained_rows = rows[: query.limit]
        children: list[SessionLineageNode] = []
        for row in retained_rows:
            base = SessionLineageNode(
                id=row["id"],
                parent_session_id=query.parent_session_id,
                created_at=sqlite_records.parse_datetime(row["created_at"]),
            )
            origin_rows = connection.execute(
                """
                SELECT sequence,
                       CASE
                           WHEN length(event_id) <= ?
                            AND length(CAST(event_id AS BLOB)) <= ?
                           THEN event_id
                       END AS event_id,
                       event_type
                FROM cayu_events
                WHERE session_id = ?
                  AND event_type IN (?, ?)
                ORDER BY sequence ASC
                LIMIT ?
                """,
                (
                    EVENT_ID_MAX_CHARS,
                    SESSION_LINEAGE_MAX_EVENT_ID_BYTES,
                    base.id,
                    str(EventType.SESSION_STARTED),
                    str(EventType.SESSION_FORKED),
                    SESSION_LINEAGE_MAX_ORIGIN_EVENTS,
                ),
            ).fetchall()
            children.append(
                SessionLineageNode(
                    id=base.id,
                    parent_session_id=base.parent_session_id,
                    created_at=base.created_at,
                    origin_events=tuple(
                        SessionLineageOrigin(
                            sequence=origin_row["sequence"],
                            event_id=origin_row["event_id"],
                            event_type=EventType(origin_row["event_type"]),
                        )
                        for origin_row in origin_rows
                    ),
                )
            )
        has_more = len(rows) > len(retained_rows)
        return SessionLineageResult(
            parent_session_id=query.parent_session_id,
            children=tuple(children),
            next_cursor=(
                encode_session_lineage_cursor(query.parent_session_id, children[-1])
                if has_more and children
                else None
            ),
            has_more=has_more,
        )

    def read_lineage(connection: sqlite3.Connection) -> SessionLineageResult:
        connection.execute("BEGIN")
        try:
            return read_lineage_snapshot(connection)
        finally:
            connection.rollback()

    return await run_read(read_lineage)


async def query_child_session_lifecycle(
    run_read: SQLiteReadRunner, query: ChildSessionLifecycleQuery
) -> ChildSessionLifecyclePage:
    query = ChildSessionLifecycleQuery.model_validate(query)

    def read_snapshot(connection: sqlite3.Connection) -> ChildSessionLifecyclePage:
        parent = sqlite_records.load_session(connection, query.parent_session_id)
        if parent is None:
            raise KeyError(f"Session not found: {query.parent_session_id}")
        rows = connection.execute(
            "SELECT child_session_id FROM cayu_child_session_lifecycle_candidates "
            "WHERE parent_session_id = ? "
            "ORDER BY priority, sort_at, child_session_id "
            "LIMIT ?",
            (parent.id, query.max_children_inspected + 1),
        ).fetchall()
        retained_rows = rows[: query.max_children_inspected]
        retained_ids = [str(row["child_session_id"]) for row in retained_rows]
        entries = []
        unavailable_count = 0
        lifecycle_types = (
            str(EventType.SESSION_STARTED),
            str(EventType.SESSION_RESUMED),
            str(EventType.SESSION_FORKED),
            str(EventType.SESSION_COMPLETED),
            str(EventType.SESSION_FAILED),
            str(EventType.SESSION_INTERRUPTED),
        )
        children_by_id: dict[str, Session] = {}
        records_by_child: dict[str, dict[EventType, EventRecord]] = {
            child_id: {} for child_id in retained_ids
        }
        if retained_ids:
            placeholders = ", ".join("?" for _child_id in retained_ids)
            child_rows = connection.execute(
                "SELECT id, instance_id, agent_name, provider_name, model, "
                "parent_session_id, causal_budget_id, runtime_name, runtime_version, "
                "environment_name, status, created_at, updated_at, last_activity_at, "
                "run_epoch, invocation_json, metadata_json FROM cayu_sessions "
                f"WHERE id IN ({placeholders})",
                retained_ids,
            ).fetchall()
            children_by_id = {
                str(child_row["id"]): sqlite_records.session_from_row(
                    child_row,
                    labels={},
                )
                for child_row in child_rows
            }
            event_rows = connection.execute(
                """
                SELECT event.*
                FROM cayu_events AS event
                JOIN (
                    SELECT session_id, event_type, MAX(sequence) AS sequence
                    FROM cayu_events
                    WHERE session_id IN ("""
                + placeholders
                + """)
                      AND event_type IN (?, ?, ?, ?, ?, ?)
                    GROUP BY session_id, event_type
                ) AS latest ON latest.sequence = event.sequence
                ORDER BY event.session_id, event.sequence ASC
                """,
                (*retained_ids, *lifecycle_types),
            ).fetchall()
            for event_row in event_rows:
                event_record = sqlite_records.event_record_from_row(event_row)
                if event_record is None:  # pragma: no cover - row is present
                    raise RuntimeError("SQLite lifecycle event row disappeared.")
                records_by_child[str(event_row["session_id"])][
                    EventType(event_row["event_type"])
                ] = event_record

        consumption_key_by_child: dict[str, str] = {}
        for child_id in retained_ids:
            child = children_by_id.get(child_id)
            if child is None or child.parent_session_id != parent.id:
                raise RuntimeError("SQLite child-session lifecycle index is inconsistent.")
            occurrence_source = _child_session_lifecycle_occurrence(
                child,
                records_by_child[child_id],
            )
            if occurrence_source is not None:
                _relationship, occurrence = occurrence_source
                consumption_key_by_child[child_id] = child_session_notification_storage_key(
                    child.instance_id,
                    occurrence.source_id,
                )
        consumption_by_key: dict[str, dict[str, Any]] = {}
        if consumption_key_by_child:
            consumption_keys = tuple(consumption_key_by_child.values())
            placeholders = ", ".join("?" for _key in consumption_keys)
            operation_rows = connection.execute(
                "SELECT idempotency_key, record_json FROM cayu_session_operations "
                f"WHERE session_id = ? AND idempotency_key IN ({placeholders})",
                (parent.id, *consumption_keys),
            ).fetchall()
            consumption_by_key = {
                str(operation_row["idempotency_key"]): json.loads(operation_row["record_json"])
                for operation_row in operation_rows
            }

        for child_id in retained_ids:
            child = children_by_id[child_id]
            consumption_key = consumption_key_by_child.get(child_id)
            entry = _child_session_lifecycle_entry(
                parent=parent,
                child=child,
                records_by_type=records_by_child[child_id],
                consumption_record=(
                    None if consumption_key is None else consumption_by_key.get(consumption_key)
                ),
            )
            if entry is None:
                unavailable_count += 1
            else:
                entries.append(entry)
        entries.sort(key=_child_session_lifecycle_entry_sort_key)
        return ChildSessionLifecyclePage(
            parent_session_id=parent.id,
            parent_session_instance_id=parent.instance_id,
            entries=tuple(entries),
            inspected_child_count=len(retained_rows),
            unavailable_child_count=unavailable_count,
            has_more=len(rows) > len(retained_rows),
        )

    def read(connection: sqlite3.Connection) -> ChildSessionLifecyclePage:
        connection.execute("BEGIN")
        try:
            return read_snapshot(connection)
        finally:
            connection.rollback()

    return await run_read(read)


async def aggregate_usage(
    run_read: SQLiteReadRunner, query: UsageRollupQuery
) -> UsageRollupStoreResult:
    query = copy_usage_rollup_query(query)
    plan = session_store_sql.build_session_query_sql(
        session_query_from_aggregate_filter(query.sessions),
        dialect=SQL_DIALECT,
    )

    def query_aggregate(connection: sqlite3.Connection) -> UsageRollupStoreResult:
        return sqlite_aggregates.aggregate_session_usage(
            connection,
            session_plan=plan,
            query=query,
        )

    return await run_read(query_aggregate)


async def _query_events_by_session_id_batches(
    run_read: SQLiteReadRunner, query: EventQuery
) -> list[EventRecord]:
    records: list[EventRecord] = []
    for batch in _event_query_session_id_batches(query.session_ids):
        records.extend(
            await query_events(
                run_read,
                session_store_sql.event_query_with_session_ids(
                    query,
                    session_ids=batch,
                ),
            )
        )
    records.sort(
        key=lambda record: record.sequence,
        reverse=query.order_by.value == "sequence_desc",
    )
    return records[: query.limit]


async def list_sessions(
    run_read: SQLiteReadRunner,
    query: SessionQuery | None,
    *,
    pending_interruption_cascade_only: bool,
    access_bounds: _SessionAccessBounds | None = None,
    ownership_clock: Callable[[], datetime],
) -> SessionListResult:
    query = copy_session_query(query)
    session_source_sql = (
        """
        (
            SELECT session_id
            FROM cayu_checkpoints
                INDEXED BY idx_cayu_checkpoints_pending_interruption_cascade
            WHERE json_type(
                state_json,
                '$.pending_interruption_cascade'
            ) IS NOT NULL
        ) AS pending_interruption_cascades
        CROSS JOIN cayu_sessions
            ON cayu_sessions.id = pending_interruption_cascades.session_id
        """
        if pending_interruption_cascade_only
        else "cayu_sessions"
    )

    def run_query(connection: sqlite3.Connection) -> SessionListResult:
        inactive_before = (
            query.last_activity_before
            if query.inactive_for_seconds is None
            else utc_duration_cutoff(
                ownership_clock(),
                query.inactive_for_seconds,
            )
        )
        if query.inactive_for_seconds is not None and inactive_before is None:
            return SessionListResult(
                sessions=[],
                next_cursor=None,
                total_count=0 if query.include_total_count else None,
            )
        resolved_query = query.model_copy(
            update={
                "last_activity_before": inactive_before,
                "inactive_for_seconds": None,
            }
        )
        plan = session_store_sql.build_session_query_sql(
            resolved_query,
            dialect=SQL_DIALECT,
            access_clause=(
                None
                if access_bounds is None
                else session_store_sql.session_access_clause(access_bounds, dialect=SQL_DIALECT)
            ),
        )
        total_count: int | None = None
        if query.include_total_count:
            total_count = connection.execute(
                f"SELECT COUNT(*) FROM {session_source_sql} {plan.filter_where_sql}",
                plan.filter_params,
            ).fetchone()[0]
        rows = connection.execute(
            f"""
            SELECT id, instance_id, agent_name, provider_name, model, parent_session_id,
                   causal_budget_id, runtime_name, runtime_version, environment_name,
                   status, created_at, updated_at, last_activity_at, run_epoch,
                   invocation_json, metadata_json
            FROM {session_source_sql}
            {plan.page_where_sql}
            ORDER BY {plan.order_sql}
            {plan.pagination_sql}
            """,
            plan.page_params,
        ).fetchall()
        has_more = len(rows) > query.limit
        rows = rows[: query.limit]
        labels_by_session_id = sqlite_records.load_session_labels_batch(
            connection, [row["id"] for row in rows]
        )
        sessions = [
            sqlite_records.session_from_row(
                row,
                labels=labels_by_session_id.get(row["id"], {}),
            )
            for row in rows
        ]
        next_cursor = session_next_cursor(sessions, has_more, query.order_by)
        return SessionListResult(
            sessions=sessions,
            next_cursor=next_cursor,
            total_count=total_count,
        )

    def snapshot(connection):
        if access_bounds is None:
            return run_query(connection)
        connection.execute("BEGIN")
        try:
            return run_query(connection)
        finally:
            connection.rollback()

    return await run_read(snapshot)
