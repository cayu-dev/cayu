"""Complete PostgreSQL session query operations with explicit native execution capabilities."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import TYPE_CHECKING, Any, LiteralString, cast
from uuid import uuid4

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
from cayu.execution_units import (
    ToolRoundIdentity,
    copy_tool_round_identity,
)
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
from cayu.sessions.event_delivery import restore_persisted_event_authority
from cayu.sessions.event_queries import EventQuery, EventQueryResultTooLarge, copy_event_query
from cayu.sessions.inspection import SESSION_INSPECTION_LABEL_LIMIT, SessionInspectionIdentity
from cayu.sessions.lineage import (
    SESSION_LINEAGE_MAX_EVENT_ID_BYTES,
    SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
    SESSION_LINEAGE_MAX_ORIGIN_EVENTS,
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
    EventRecord,
    Session,
    SessionStatus,
)
from cayu.sessions.summaries import (
    EventSummary,
)
from cayu.sessions.topology import (
    SessionTopologyCycle,
    SessionTopologyDepthExceeded,
    SessionTopologyQuery,
    SessionTopologyStoreResult,
    build_session_topology_result,
    decode_session_topology_cursor,
)
from cayu.sessions.usage import UsageRollupQuery, copy_usage_rollup_query
from cayu.storage import _postgres_aggregates as postgres_aggregates
from cayu.storage import _postgres_support as pg_support
from cayu.storage import _session_store_sql as session_store_sql

if TYPE_CHECKING:
    from cayu.runtime._cost_accounting_refresh import CostAccountingAuthority
    from cayu.runtime._usage_accounting import SessionUsageCache
    from cayu.sessions.access import _SessionAccessBounds


PostgresConnection = Callable[[], AbstractAsyncContextManager[Any]]
SessionLoader = Callable[[Any, str], Awaitable[Session | None]]


_EVENT_QUERY_SESSION_IDS_BATCH_SIZE = 500


SQL_DIALECT = session_store_sql.SessionStoreSqlDialect(
    placeholder="%s",
    contains_style="postgres_ilike",
    datetime_param=pg_support.to_utc,
)


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


def _event_query_is_single_session(query: EventQuery) -> bool:
    return query.session_id is not None or len(query.session_ids) == 1


def _event_query_needs_snapshot_cutoff(query: EventQuery) -> bool:
    return (
        query.after_sequence is not None or query.before_sequence is not None
    ) and not _event_query_is_single_session(query)


def event_record_from_row(row: tuple[Any, Any] | None) -> EventRecord | None:
    """Build an EventRecord from a ``(sequence, event)`` row, or None for a missing row."""
    if row is None:
        return None
    return EventRecord(sequence=row[0], event=Event(**pg_support._json_obj(row[1])))


async def inspect_identity(
    connect: PostgresConnection, session_id: str
) -> SessionInspectionIdentity:
    session_id = require_clean_nonblank(session_id, "session_id")
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            SELECT id, agent_name, provider_name, model, parent_session_id,
                   causal_budget_id, runtime_name, runtime_version, environment_name,
                   status, created_at, updated_at, last_activity_at, run_epoch,
                   metadata -> 'cayu:runtime_build_provenance'
                       AS runtime_build_provenance
            FROM cayu_sessions
            WHERE id = %s
            """,
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(session_id)
        await cur.execute(
            """
            SELECT key, value,
                   (SELECT COUNT(*)
                    FROM cayu_session_labels
                    WHERE session_id = %s) AS label_count
            FROM cayu_session_labels
            WHERE session_id = %s
            ORDER BY key COLLATE "C" ASC
            LIMIT %s
            """,
            (session_id, session_id, SESSION_INSPECTION_LABEL_LIMIT),
        )
        label_rows = await cur.fetchall()
        label_count = 0 if not label_rows else label_rows[0][2]
        return SessionInspectionIdentity(
            id=row[0],
            agent_name=row[1],
            provider_name=row[2],
            model=row[3],
            parent_session_id=row[4],
            causal_budget_id=row[5],
            runtime_name=row[6],
            runtime_version=row[7],
            runtime_build_provenance=(
                pg_support.runtime_build_provenance_from_session_metadata(
                    {} if row[14] is None else {"cayu:runtime_build_provenance": row[14]}
                )
            ),
            environment_name=row[8],
            status=SessionStatus(row[9]),
            created_at=pg_support.to_utc(row[10]),
            updated_at=pg_support.to_utc(row[11]),
            last_activity_at=pg_support.to_utc(row[12]),
            run_epoch=row[13],
            labels={label_row[0]: label_row[1] for label_row in label_rows},
            label_count=label_count,
            labels_truncated=label_count > len(label_rows),
        )


async def load_events(
    connect: PostgresConnection, session_id: str, *, load_session: SessionLoader
) -> list[Event]:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")
    async with connect() as conn, conn.cursor() as cur:
        if access_bounds is not None:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            access_bounds.require_read(await load_session(cur, session_id))
        await cur.execute(
            "SELECT 1 FROM cayu_sessions WHERE id = %s",
            (session_id,),
        )
        if await cur.fetchone() is None:
            raise KeyError(f"Session not found: {session_id}")
        await cur.execute(
            """
            SELECT event
            FROM cayu_events
            WHERE session_id = %s
            ORDER BY session_order ASC
            """,
            (session_id,),
        )
        rows = await cur.fetchall()
        return [Event(**pg_support._json_obj(row[0])) for row in rows]


async def load_user_input_supersession_events(
    connect: PostgresConnection, session_id: str, input_id: str
) -> list[Event]:
    from cayu.sessions.pending_actions import pending_action_lookup_key

    session_id = require_clean_nonblank(session_id, "session_id")
    input_id = require_clean_nonblank(input_id, "input_id")
    lookup_key = pending_action_lookup_key(input_id)
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT 1 FROM cayu_sessions WHERE id = %s",
            (session_id,),
        )
        if await cur.fetchone() is None:
            raise KeyError(f"Session not found: {session_id}")
        await cur.execute(
            "SELECT event FROM cayu_events "
            "WHERE session_id = %s AND pending_action_lookup_key = %s "
            "AND event_type = %s "
            f"AND ({PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
            "AND event -> 'payload' -> 'user_input_supersession_intent' "
            "->> 'input_id' = %s "
            "ORDER BY session_order ASC LIMIT 2",
            (
                session_id,
                lookup_key,
                str(EventType.SESSION_INTERRUPTED),
                input_id,
            ),
        )
        rows = await cur.fetchall()
        return [Event(**pg_support._json_obj(row[0])) for row in rows]


async def load_tool_round_lifecycle_events(
    connect: PostgresConnection, session_id: str, tool_call_ids: list[str] | tuple[str, ...]
) -> list[Event]:
    from cayu.sessions.pending_actions import pending_action_lookup_key

    session_id = require_clean_nonblank(session_id, "session_id")
    copied_ids = _validate_tool_round_call_ids(tool_call_ids, "tool_call_ids")
    lookup_keys = [pending_action_lookup_key(call_id) for call_id in copied_ids]
    lifecycle_event_types = [
        str(event_type) for event_type in sorted(_TOOL_ROUND_LIFECYCLE_EVENT_TYPES, key=str)
    ]
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT 1 FROM cayu_sessions WHERE id = %s",
            (session_id,),
        )
        if await cur.fetchone() is None:
            raise KeyError(f"Session not found: {session_id}")
        await cur.execute(
            "SELECT event FROM cayu_events "
            "WHERE session_id = %s "
            "AND pending_action_lookup_key = ANY(%s) "
            "AND event_type = ANY(%s) "
            f"AND ({PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
            "ORDER BY session_order ASC LIMIT %s",
            (
                session_id,
                lookup_keys,
                lifecycle_event_types,
                _tool_round_lifecycle_event_limit(copied_ids) + 1,
            ),
        )
        rows = await cur.fetchall()
        if len(rows) > _tool_round_lifecycle_event_limit(copied_ids):
            raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
        return [Event(**pg_support._json_obj(row[0])) for row in rows]


async def load_tool_round_lifecycle_events_for_round(
    connect: PostgresConnection,
    session_id: str,
    tool_call_ids: list[str] | tuple[str, ...],
    *,
    tool_round_identity: ToolRoundIdentity,
) -> list[Event]:
    from cayu.sessions.pending_actions import pending_action_lookup_key

    session_id = require_clean_nonblank(session_id, "session_id")
    copied_ids = _validate_tool_round_call_ids(tool_call_ids, "tool_call_ids")
    tool_round_identity = copy_tool_round_identity(tool_round_identity)
    lookup_keys = [pending_action_lookup_key(call_id) for call_id in copied_ids]
    lifecycle_event_types = [
        str(event_type) for event_type in sorted(_TOOL_ROUND_LIFECYCLE_EVENT_TYPES, key=str)
    ]
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT 1 FROM cayu_sessions WHERE id = %s",
            (session_id,),
        )
        if await cur.fetchone() is None:
            raise KeyError(f"Session not found: {session_id}")
        await cur.execute(
            "SELECT event FROM cayu_events "
            "WHERE session_id = %s "
            "AND pending_action_lookup_key = ANY(%s) "
            "AND event_type = ANY(%s) "
            f"AND ({PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
            "AND (event -> 'payload' ->> 'tool_round_id' = %s "
            "OR (event -> 'payload' ->> 'model_step_id' = %s "
            "AND event -> 'payload' ->> 'model_attempt_id' = %s) "
            "OR ((event -> 'payload' ->> 'tool_round_id') "
            "~ '^tround_[0-9a-f]{32}$') IS NOT TRUE "
            "OR ((event -> 'payload' ->> 'model_step_id') "
            "~ '^mstep_[0-9a-f]{32}$') IS NOT TRUE "
            "OR ((event -> 'payload' ->> 'model_attempt_id') "
            "~ '^matt_[0-9a-f]{32}$') IS NOT TRUE) "
            "ORDER BY session_order ASC LIMIT %s",
            (
                session_id,
                lookup_keys,
                lifecycle_event_types,
                tool_round_identity.tool_round_id,
                tool_round_identity.model_step_id,
                tool_round_identity.model_attempt_id,
                _tool_round_lifecycle_event_limit(copied_ids) + 1,
            ),
        )
        rows = await cur.fetchall()
        if len(rows) > _tool_round_lifecycle_event_limit(copied_ids):
            raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
        return [Event(**pg_support._json_obj(row[0])) for row in rows]


@runtime_session_query
async def query_events(
    connect: PostgresConnection, query: EventQuery | None = None
) -> list[EventRecord]:
    query = copy_event_query(query)
    if len(query.session_ids) > _EVENT_QUERY_SESSION_IDS_BATCH_SIZE:
        return await _query_events_by_session_id_batches(connect, query)
    async with connect() as conn, conn.cursor() as cur:
        return await _query_events(cur, query, safe_insert_xid=None)


@runtime_session_query
async def event_exists(connect: PostgresConnection, query: EventQuery) -> bool:
    plan = session_store_sql.build_accounting_event_query_sql(query, dialect=SQL_DIALECT)
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            cast(
                "LiteralString",
                "SELECT EXISTS(SELECT 1 FROM cayu_events "
                "JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id "
                f"{plan.where_sql})",
            ),
            plan.params,
        )
        row = await cur.fetchone()
        if row is None:
            raise RuntimeError("Failed to read event existence result.")
        return bool(row[0])


@runtime_session_query
async def read_usage_accounting(
    connect: PostgresConnection,
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
    cached_session_id = SessionUsageCache.session_scope(
        query, by_session=by_session, by_identity=by_identity
    )
    prior = None
    boundary = 0
    async with connect() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            await cur.execute("SELECT generation FROM cayu_accounting_state WHERE singleton = 1")
            generation_row = await cur.fetchone()
            if generation_row is None:
                raise RuntimeError("Accounting deletion revision is missing.")
            generation = generation_row[0]
            if cached_session_id is not None:
                prior = usage_cache.resume_after(cached_session_id, generation)
                if prior is not None:
                    query = copy_event_query(
                        query, update={"after_sequence": prior.scanned_through}
                    )
                # The session's event counter row lock serializes its appends,
                # so a later commit for this session cannot land below this.
                await cur.execute(
                    "SELECT MAX(sequence) FROM cayu_events WHERE session_id = %s",
                    (cached_session_id,),
                )
                boundary_row = await cur.fetchone()
                boundary = 0 if boundary_row is None else boundary_row[0] or 0

        reducer = UsageAccountingReducer(query, by_session=by_session, by_identity=by_identity)
        plan = session_store_sql.build_accounting_event_query_sql(query, dialect=SQL_DIALECT)
        # A named server cursor prevents libpq from buffering all rows in the
        # client, while REPEATABLE READ pins one snapshot across fetches.
        async with conn.cursor(name=f"usage_{uuid4().hex}") as cur:
            await cur.execute(
                cast(
                    "LiteralString",
                    "SELECT cayu_events.sequence, cayu_events.event FROM cayu_events "
                    "JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id "
                    f"{plan.where_sql} ORDER BY cayu_events.sequence ASC",
                ),
                plan.params,
            )
            while rows := await cur.fetchmany(USAGE_ACCOUNTING_PAGE_SIZE):
                reducer.add_page(
                    [
                        EventRecord(sequence=row[0], event=Event(**pg_support._json_obj(row[1])))
                        for row in rows
                    ]
                )
        if cached_session_id is not None:
            return usage_cache.settle(
                cached_session_id, generation, prior, reducer.snapshot(), boundary=boundary
            )
        return reducer.snapshot().model_copy(update={"generation": generation})


@runtime_session_query
async def read_cost_accounting(
    connect: PostgresConnection,
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

    if previous is not None and type(previous) is not CostAccountingSnapshot:
        raise TypeError("previous must be a CostAccountingSnapshot.")
    query = cost_accounting_query(query)
    pending = cost_pending_events(query, additional_events)
    async with connect() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            await cur.execute("SELECT generation FROM cayu_accounting_state WHERE singleton = 1")
            generation_row = await cur.fetchone()
            if generation_row is None:
                raise RuntimeError("Accounting deletion revision is missing.")
            generation = generation_row[0]
            if query.causal_budget_id is not None and pending:
                await cur.execute(
                    "SELECT id FROM cayu_sessions WHERE causal_budget_id = %s AND id = ANY(%s)",
                    (query.causal_budget_id, [event.session_id for event in pending]),
                )
                allowed_sessions = {row[0] for row in await cur.fetchall()}
                pending = tuple(event for event in pending if event.session_id in allowed_sessions)

        plan = session_store_sql.build_accounting_event_query_sql(query, dialect=SQL_DIALECT)
        unique_pending: list[Event] = []
        async with conn.cursor() as cur:
            for event in pending:
                await cur.execute(
                    cast(
                        "LiteralString",
                        "SELECT 1 FROM cayu_events "
                        "JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id "
                        f"{plan.where_sql} AND cayu_events.session_id = %s AND cayu_events.event_id = %s LIMIT 1",
                    ),
                    (*plan.params, event.session_id, event.id),
                )
                if await cur.fetchone() is None:
                    unique_pending.append(event)
        boundary_sql, boundary_params = cost_boundary_statement(query, dialect=SQL_DIALECT)
        async with conn.cursor() as cur:
            await cur.execute(cast("LiteralString", boundary_sql), boundary_params)
            boundary_row = await cur.fetchone()
            if boundary_row is None:
                raise RuntimeError("Accounting event boundary is missing.")
            boundary = boundary_row[0] or 0
        reducer = CostAccountingRead(
            query,
            pricing,
            currency=currency,
            details=details,
            max_detail_bytes=max_detail_bytes,
            previous=previous if _event_query_is_single_session(query) else None,
            generation=generation,
            through_sequence=boundary,
            authority=cost_authority,
            by_session=by_session,
            additional_events=tuple(unique_pending),
        )
        source_plan = session_store_sql.build_accounting_event_query_sql(
            reducer.source_query,
            dialect=SQL_DIALECT,
        )
        columns = "cayu_events.sequence, cayu_events.event, cayu_events.session_id, cayu_events.event_id, cayu_events.event_type"

        async def add_group(key: tuple[str, bool, str]) -> None:
            statement, params = cost_group_lookup_statement(
                columns=columns, plan=source_plan, key=key, postgres=True
            )
            async with conn.cursor(name=f"cost_group_{uuid4().hex}") as group_cursor:
                await group_cursor.execute(cast("LiteralString", statement), params)
                while rows := await group_cursor.fetchmany(COST_ACCOUNTING_PAGE_SIZE):
                    for row in rows:
                        reducer.add(row[0], Event(**pg_support._json_obj(row[1])))

        if reducer.incremental:
            statement, params = changed_cost_groups_statement(reducer, dialect=SQL_DIALECT)
            async with conn.cursor(name=f"changed_cost_{uuid4().hex}") as cur:
                await cur.execute(cast("LiteralString", statement), params)
                while groups := await cur.fetchmany(COST_ACCOUNTING_PAGE_SIZE):
                    for group in groups:
                        await add_group(
                            (
                                group[0],
                                group[1] is not None,
                                group[1] if group[1] is not None else group[2],
                            )
                        )
            for key in reducer.remaining_pending_keys:
                await add_group(key)
        else:
            statement, group_params = cost_group_statement(
                columns=columns, where_sql=source_plan.where_sql, postgres=True
            )
            async with conn.cursor(name=f"cost_{uuid4().hex}") as cur:
                await cur.execute(
                    cast("LiteralString", statement), (*group_params, *source_plan.params)
                )
                while rows := await cur.fetchmany(COST_ACCOUNTING_PAGE_SIZE):
                    for row in rows:
                        reducer.add(row[0], Event(**pg_support._json_obj(row[1])))
        result = reducer.snapshot()
        # Global sequence allocation is not commit order. A later commit can
        # become visible below this snapshot's maximum sequence, so it is not
        # a safe incremental baseline across independently written sessions.
        if not _event_query_is_single_session(query):
            result = result.model_copy(update={"cursor": None})
        return result


@runtime_session_query
async def query_events_bounded(
    connect: PostgresConnection, query: EventQuery, *, max_bytes: int
) -> list[EventRecord]:
    query = copy_event_query(query)
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer.")
    if len(query.session_ids) > _EVENT_QUERY_SESSION_IDS_BATCH_SIZE:
        raise ValueError("Byte-bounded event queries require one bounded SQL batch.")
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        needs_snapshot_cutoff = _event_query_needs_snapshot_cutoff(query)
        safe_insert_xid = None
        extra_clauses: tuple[session_store_sql.SqlClause, ...] = ()
        if needs_snapshot_cutoff:
            await cur.execute("SELECT pg_snapshot_xmin(pg_current_snapshot())")
            snapshot_row = await cur.fetchone()
            if snapshot_row is None:
                raise RuntimeError("Failed to read Postgres event visibility snapshot.")
            safe_insert_xid = snapshot_row[0]
            extra_clauses = (
                session_store_sql.SqlClause(
                    "cayu_events.insert_xid < %s",
                    (safe_insert_xid,),
                ),
            )
        plan = session_store_sql.build_event_query_sql(
            query,
            dialect=SQL_DIALECT,
            extra_after_sequence_clauses=extra_clauses,
        )
        await cur.execute(
            cast(
                "LiteralString",
                f"""
                WITH bounded_candidates AS (
                    SELECT octet_length(cayu_events.event::text) + 256
                               AS serialized_bytes
                    FROM cayu_events
                    JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                    {plan.where_sql}
                    ORDER BY cayu_events.sequence {plan.order_direction}
                    LIMIT %s
                )
                SELECT COALESCE(SUM(serialized_bytes), 0)
                FROM bounded_candidates
                """,
            ),
            (*plan.params, query.limit),
        )
        size_row = await cur.fetchone()
        if size_row is None or int(size_row[0]) > max_bytes:
            raise EventQueryResultTooLarge(max_bytes)
        return await _query_events(
            cur,
            query,
            safe_insert_xid=safe_insert_xid,
            force_snapshot_cutoff=needs_snapshot_cutoff,
        )


async def query_latest_interaction_events(
    connect: PostgresConnection,
    session_id: str,
    *,
    before_sequence: int | None = None,
    limit: int = 100,
) -> list[EventRecord]:
    session_id = require_clean_nonblank(session_id, "session_id")
    before_sequence, limit = _validate_interaction_page(before_sequence, limit)
    cursor_clause = "" if before_sequence is None else "AND latest.latest_event_sequence < %s"
    params: list[object] = [session_id]
    if before_sequence is not None:
        params.append(before_sequence)
    params.append(limit)
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
        if await cur.fetchone() is None:
            raise KeyError(f"Session not found: {session_id}")
        await cur.execute(
            f"""
                SELECT event.sequence, event.event
                FROM cayu_interaction_latest_events AS latest
                JOIN cayu_events AS event
                  ON event.sequence = latest.latest_event_sequence
                WHERE latest.session_id = %s {cursor_clause}
                ORDER BY latest.latest_event_sequence DESC
                LIMIT %s
                """,
            params,
        )
        rows = await cur.fetchall()
        return [
            EventRecord(sequence=row[0], event=Event(**pg_support._json_obj(row[1])))
            for row in rows
        ]


async def summarize_events(
    connect: PostgresConnection, session_id: str, *, load_session: SessionLoader
) -> EventSummary:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")
    async with connect() as conn, conn.cursor() as cur:
        if access_bounds is not None:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            access_bounds.require_read(await load_session(cur, session_id))
        await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
        if await cur.fetchone() is None:
            raise KeyError(f"Session not found: {session_id}")

        await cur.execute(
            "SELECT COUNT(*) FROM cayu_events WHERE session_id = %s",
            (session_id,),
        )
        total_row = await cur.fetchone()
        total_events = int(total_row[0]) if total_row is not None else 0

        await cur.execute(
            """
            SELECT event_type, COUNT(*)
            FROM cayu_events
            WHERE session_id = %s
            GROUP BY event_type
            ORDER BY event_type ASC
            """,
            (session_id,),
        )
        counts_by_type = {row[0]: int(row[1]) for row in await cur.fetchall()}

        await cur.execute(
            """
            SELECT sequence, event
            FROM cayu_events
            WHERE session_id = %s
            ORDER BY sequence DESC
            LIMIT 1
            """,
            (session_id,),
        )
        latest_row = await cur.fetchone()

        return EventSummary(
            session_id=session_id,
            total_events=total_events,
            counts_by_type=counts_by_type,
            latest_event=event_record_from_row(latest_row),
        )


async def query_session_topology(
    connect: PostgresConnection, query: SessionTopologyQuery
) -> SessionTopologyStoreResult:
    if type(query) is not SessionTopologyQuery:
        raise TypeError("Session topology queries must be SessionTopologyQuery instances.")
    query = query.model_copy(deep=True)
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                await cur.execute(
                    cast(
                        "LiteralString",
                        f"""
                        SELECT {pg_support.SESSION_TOPOLOGY_COLUMNS}
                        FROM cayu_sessions
                        WHERE id = %s
                        """,
                    ),
                    (query.focus_session_id,),
                )
                focus_row = await cur.fetchone()
                if focus_row is None:
                    raise KeyError(f"Session not found: {query.focus_session_id}")
                focus = pg_support.session_topology_node_from_row(focus_row)

                ancestors = []
                seen_ids = {focus.id}
                parent_session_id = focus.parent_session_id
                while parent_session_id is not None:
                    if parent_session_id in seen_ids:
                        raise SessionTopologyCycle(
                            f"Session topology contains a parent cycle at {parent_session_id}."
                        )
                    if len(ancestors) >= query.ancestor_depth_limit:
                        raise SessionTopologyDepthExceeded(
                            "Session topology exceeds the "
                            f"{query.ancestor_depth_limit}-ancestor limit."
                        )
                    await cur.execute(
                        cast(
                            "LiteralString",
                            f"""
                            SELECT {pg_support.SESSION_TOPOLOGY_COLUMNS}
                            FROM cayu_sessions
                            WHERE id = %s
                            """,
                        ),
                        (parent_session_id,),
                    )
                    parent_row = await cur.fetchone()
                    if parent_row is None:
                        raise ValueError(
                            f"Session topology references missing parent {parent_session_id}."
                        )
                    parent = pg_support.session_topology_node_from_row(parent_row)
                    ancestors.append(parent)
                    seen_ids.add(parent.id)
                    parent_session_id = parent.parent_session_id
                ancestors.reverse()

                expanded_parents = []
                if query.expanded_parent_ids:
                    await cur.execute(
                        cast(
                            "LiteralString",
                            f"""
                            SELECT {pg_support.SESSION_TOPOLOGY_COLUMNS}
                            FROM cayu_sessions
                            WHERE id = ANY(%s)
                            """,
                        ),
                        (list(query.expanded_parent_ids),),
                    )
                    parents_by_id = {
                        row[0]: pg_support.session_topology_node_from_row(row)
                        for row in await cur.fetchall()
                    }
                    for parent_id in query.expanded_parent_ids:
                        parent = parents_by_id.get(parent_id)
                        if parent is None:
                            raise KeyError(f"Session not found: {parent_id}")
                        expanded_parents.append(parent)

                candidates_by_parent = {parent.id: [] for parent in expanded_parents}
                if expanded_parents:
                    requested_parent_ids: list[str] = []
                    cursor_created_ats: list[datetime | None] = []
                    cursor_ids: list[str | None] = []
                    for parent in expanded_parents:
                        requested_parent_ids.append(parent.id)
                        cursor = query.child_cursors.get(parent.id)
                        if cursor is None:
                            cursor_created_ats.append(None)
                            cursor_ids.append(None)
                            continue
                        cursor_created_at, cursor_id = decode_session_topology_cursor(
                            cursor,
                            parent_session_id=parent.id,
                        )
                        cursor_created_ats.append(cursor_created_at)
                        cursor_ids.append(cursor_id)
                    await cur.execute(
                        cast(
                            "LiteralString",
                            f"""
                            WITH requested_branches AS (
                                SELECT parent_session_id, cursor_created_at, cursor_id,
                                       branch_order
                                FROM unnest(
                                    %s::text[],
                                    %s::timestamptz[],
                                    %s::text[]
                                ) WITH ORDINALITY AS requested(
                                    parent_session_id,
                                    cursor_created_at,
                                    cursor_id,
                                    branch_order
                                )
                            )
                            SELECT child.*
                            FROM requested_branches AS requested
                            CROSS JOIN LATERAL (
                                SELECT {pg_support.SESSION_TOPOLOGY_COLUMNS}
                                FROM cayu_sessions
                                WHERE cayu_sessions.parent_session_id =
                                      requested.parent_session_id
                                  AND (
                                      requested.cursor_created_at IS NULL
                                      OR cayu_sessions.created_at >
                                         requested.cursor_created_at
                                      OR (
                                          cayu_sessions.created_at =
                                              requested.cursor_created_at
                                          AND cayu_sessions.id COLLATE "C" >
                                              requested.cursor_id COLLATE "C"
                                      )
                                  )
                                ORDER BY cayu_sessions.created_at ASC,
                                         cayu_sessions.id COLLATE "C" ASC
                                LIMIT %s
                            ) AS child
                            ORDER BY requested.branch_order ASC,
                                     child.created_at ASC,
                                     child.id COLLATE "C" ASC
                            """,
                        ),
                        (
                            requested_parent_ids,
                            cursor_created_ats,
                            cursor_ids,
                            query.child_limit + 1,
                        ),
                    )
                    for row in await cur.fetchall():
                        candidates_by_parent[row[4]].append(
                            pg_support.session_topology_node_from_row(row)
                        )
                result = build_session_topology_result(
                    focus=focus,
                    ancestors=ancestors,
                    expanded_parents=expanded_parents,
                    branch_candidates=(
                        candidates_by_parent[parent.id] for parent in expanded_parents
                    ),
                    child_limit=query.child_limit,
                )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
    return result


async def query_session_lineage(
    connect: PostgresConnection, query: SessionLineageQuery
) -> SessionLineageResult:
    query = copy_session_lineage_query(query)
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                await cur.execute(
                    "SELECT 1 FROM cayu_sessions WHERE id = %s",
                    (query.parent_session_id,),
                )
                if await cur.fetchone() is None:
                    raise KeyError(f"Session not found: {query.parent_session_id}")

                cursor_clause = ""
                params: list[object] = [
                    SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
                    query.parent_session_id,
                ]
                if query.cursor is not None:
                    cursor_created_at, cursor_id = decode_session_lineage_cursor(
                        query.cursor,
                        parent_session_id=query.parent_session_id,
                    )
                    cursor_clause = (
                        'AND (created_at > %s OR (created_at = %s AND id COLLATE "C" '
                        '> %s COLLATE "C"))'
                    )
                    params.extend((cursor_created_at, cursor_created_at, cursor_id))
                params.append(query.limit + 1)
                await cur.execute(
                    f"""
                    SELECT CASE
                               WHEN octet_length(id) <= %s THEN id
                           END AS id,
                           created_at
                    FROM cayu_sessions
                    WHERE parent_session_id = %s
                      {cursor_clause}
                    ORDER BY created_at ASC, id COLLATE "C" ASC
                    LIMIT %s
                    """,
                    params,
                )
                rows = await cur.fetchall()
                retained_rows = rows[: query.limit]
                bases = tuple(
                    SessionLineageNode(
                        id=row[0],
                        parent_session_id=query.parent_session_id,
                        created_at=pg_support.to_utc(row[1]),
                    )
                    for row in retained_rows
                )
                grouped_origins: dict[str, list[SessionLineageOrigin]] = {
                    base.id: [] for base in bases
                }
                if bases:
                    await cur.execute(
                        """
                        SELECT requested.session_id, origin.sequence,
                               origin.event_id, origin.event_type
                        FROM unnest(%s::text[]) WITH ORDINALITY AS requested(
                            session_id,
                            session_order
                        )
                        LEFT JOIN LATERAL (
                            SELECT sequence,
                                   CASE
                                       WHEN length(event_id) <= %s
                                        AND octet_length(event_id) <= %s
                                       THEN event_id
                                   END AS event_id,
                                   event_type
                            FROM cayu_events
                            WHERE cayu_events.session_id = requested.session_id
                              AND event_type = ANY(%s)
                            ORDER BY sequence ASC
                            LIMIT %s
                        ) AS origin ON TRUE
                        ORDER BY requested.session_order ASC, origin.sequence ASC
                        """,
                        (
                            [base.id for base in bases],
                            EVENT_ID_MAX_CHARS,
                            SESSION_LINEAGE_MAX_EVENT_ID_BYTES,
                            [
                                str(EventType.SESSION_STARTED),
                                str(EventType.SESSION_FORKED),
                            ],
                            SESSION_LINEAGE_MAX_ORIGIN_EVENTS,
                        ),
                    )
                    for row in await cur.fetchall():
                        if row[1] is None:
                            continue
                        grouped_origins[row[0]].append(
                            SessionLineageOrigin(
                                sequence=row[1],
                                event_id=row[2],
                                event_type=EventType(row[3]),
                            )
                        )
                children = tuple(
                    SessionLineageNode(
                        id=base.id,
                        parent_session_id=base.parent_session_id,
                        created_at=base.created_at,
                        origin_events=tuple(grouped_origins[base.id]),
                    )
                    for base in bases
                )
                has_more = len(rows) > len(retained_rows)
                result = SessionLineageResult(
                    parent_session_id=query.parent_session_id,
                    children=children,
                    next_cursor=(
                        encode_session_lineage_cursor(
                            query.parent_session_id,
                            children[-1],
                        )
                        if has_more and children
                        else None
                    ),
                    has_more=has_more,
                )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
    return result


async def query_child_session_lifecycle(
    connect: PostgresConnection, query: ChildSessionLifecycleQuery, *, load_session: SessionLoader
) -> ChildSessionLifecyclePage:
    query = ChildSessionLifecycleQuery.model_validate(query)
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                parent = await load_session(cur, query.parent_session_id)
                if parent is None:
                    raise KeyError(f"Session not found: {query.parent_session_id}")
                await cur.execute(
                    """
                    SELECT child_session_id
                    FROM cayu_child_session_lifecycle_candidates
                    WHERE parent_session_id = %s
                    ORDER BY priority, sort_at, child_session_id COLLATE "C"
                    LIMIT %s
                    """,
                    (
                        parent.id,
                        query.max_children_inspected + 1,
                    ),
                )
                rows = await cur.fetchall()
                retained_rows = rows[: query.max_children_inspected]
                retained_ids = [str(row[0]) for row in retained_rows]
                entries = []
                unavailable_count = 0
                lifecycle_types = [
                    str(EventType.SESSION_STARTED),
                    str(EventType.SESSION_RESUMED),
                    str(EventType.SESSION_FORKED),
                    str(EventType.SESSION_COMPLETED),
                    str(EventType.SESSION_FAILED),
                    str(EventType.SESSION_INTERRUPTED),
                ]
                children_by_id: dict[str, Session] = {}
                records_by_child: dict[str, dict[EventType, EventRecord]] = {
                    child_id: {} for child_id in retained_ids
                }
                if retained_ids:
                    await cur.execute(
                        f"SELECT {pg_support.SESSION_COLUMNS} FROM cayu_sessions "
                        "WHERE id = ANY(%s)",
                        (retained_ids,),
                    )
                    children_by_id = {
                        str(child_row[0]): pg_support.session_from_row(
                            child_row,
                            labels={},
                        )
                        for child_row in await cur.fetchall()
                    }
                    await cur.execute(
                        """
                        SELECT latest.session_id, latest.sequence, latest.event
                        FROM (
                            SELECT DISTINCT ON (session_id, event_type)
                                   session_id, event_type, sequence, event
                            FROM cayu_events
                            WHERE session_id = ANY(%s)
                              AND event_type = ANY(%s)
                            ORDER BY session_id, event_type, sequence DESC
                        ) AS latest
                        ORDER BY latest.session_id, latest.sequence ASC
                        """,
                        (retained_ids, lifecycle_types),
                    )
                    for event_row in await cur.fetchall():
                        event = Event(**pg_support._json_obj(event_row[2]))
                        records_by_child[str(event_row[0])][EventType(event.type)] = EventRecord(
                            sequence=event_row[1], event=event
                        )

                consumption_key_by_child: dict[str, str] = {}
                for child_id in retained_ids:
                    child = children_by_id.get(child_id)
                    if child is None or child.parent_session_id != parent.id:
                        raise RuntimeError(
                            "Postgres child-session lifecycle index is inconsistent."
                        )
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
                    await cur.execute(
                        "SELECT idempotency_key, record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                        (parent.id, list(consumption_key_by_child.values())),
                    )
                    consumption_by_key = {
                        str(operation_row[0]): pg_support._json_obj(operation_row[1])
                        for operation_row in await cur.fetchall()
                    }

                for child_id in retained_ids:
                    child = children_by_id[child_id]
                    consumption_key = consumption_key_by_child.get(child_id)
                    entry = _child_session_lifecycle_entry(
                        parent=parent,
                        child=child,
                        records_by_type=records_by_child[child_id],
                        consumption_record=(
                            None
                            if consumption_key is None
                            else consumption_by_key.get(consumption_key)
                        ),
                    )
                    if entry is None:
                        unavailable_count += 1
                    else:
                        entries.append(entry)
                entries.sort(key=_child_session_lifecycle_entry_sort_key)
                result = ChildSessionLifecyclePage(
                    parent_session_id=parent.id,
                    parent_session_instance_id=parent.instance_id,
                    entries=tuple(entries),
                    inspected_child_count=len(retained_rows),
                    unavailable_child_count=unavailable_count,
                    has_more=len(rows) > len(retained_rows),
                )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
    return result


async def aggregate_usage(
    connect: PostgresConnection, query: UsageRollupQuery
) -> UsageRollupStoreResult:
    query = copy_usage_rollup_query(query)
    plan = session_store_sql.build_session_query_sql(
        session_query_from_aggregate_filter(query.sessions),
        dialect=SQL_DIALECT,
    )
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                await cur.execute("SELECT transaction_timestamp()")
                as_of_row = await cur.fetchone()
                if as_of_row is None:
                    raise RuntimeError("Postgres did not return a snapshot timestamp.")
            result = await postgres_aggregates.aggregate_session_usage(
                conn,
                session_plan=plan,
                query=query,
                as_of=as_of_row[0],
            )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
    return result


async def _query_events_by_session_id_batches(
    connect: PostgresConnection, query: EventQuery
) -> list[EventRecord]:
    records: list[EventRecord] = []
    async with connect() as conn, conn.cursor() as cur:
        safe_insert_xid = None
        needs_snapshot_cutoff = query.after_sequence is not None
        if needs_snapshot_cutoff:
            await cur.execute("SELECT pg_snapshot_xmin(pg_current_snapshot())")
            row = await cur.fetchone()
            if row is None:
                raise RuntimeError("Failed to read Postgres event visibility snapshot.")
            safe_insert_xid = row[0]
        for batch in _event_query_session_id_batches(query.session_ids):
            records.extend(
                await _query_events(
                    cur,
                    session_store_sql.event_query_with_session_ids(
                        query,
                        session_ids=batch,
                    ),
                    safe_insert_xid=safe_insert_xid,
                    force_snapshot_cutoff=needs_snapshot_cutoff,
                )
            )
    records.sort(
        key=lambda record: record.sequence,
        reverse=query.order_by.value == "sequence_desc",
    )
    return records[: query.limit]


async def list_sessions(
    connect: PostgresConnection,
    query: SessionQuery | None,
    *,
    pending_interruption_cascade_only: bool,
    access_bounds: _SessionAccessBounds | None = None,
) -> SessionListResult:
    query = copy_session_query(query)
    session_source_sql = (
        """
        (
            SELECT session_id
            FROM cayu_checkpoints
            WHERE state ? 'pending_interruption_cascade'
        ) AS pending_interruption_cascades
        INNER JOIN cayu_sessions
            ON cayu_sessions.id = pending_interruption_cascades.session_id
        """
        if pending_interruption_cascade_only
        else "cayu_sessions"
    )
    async with connect() as conn, conn.cursor() as cur:
        if access_bounds is not None:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        inactive_before = query.last_activity_before
        if query.inactive_for_seconds is not None:
            await cur.execute("SELECT clock_timestamp()")
            now_row = await cur.fetchone()
            if now_row is None or type(now_row[0]) is not datetime:
                raise RuntimeError("Postgres did not return authoritative store time.")
            inactive_before = utc_duration_cutoff(
                now_row[0],
                query.inactive_for_seconds,
            )
            if inactive_before is None:
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
        # Interpolations are trusted: SESSION_COLUMNS is a constant, order_sql is
        # an enum-derived literal, the clauses are hard-coded; values bind via %s.
        total_count: int | None = None
        if query.include_total_count:
            await cur.execute(
                cast(
                    "LiteralString",
                    f"SELECT COUNT(*) FROM {session_source_sql} {plan.filter_where_sql}",
                ),
                plan.filter_params,
            )
            count_row = await cur.fetchone()
            total_count = count_row[0] if count_row is not None else 0
        await cur.execute(
            cast(
                "LiteralString",
                f"""
                SELECT {pg_support.SESSION_COLUMNS}
                FROM {session_source_sql}
                {plan.page_where_sql}
                ORDER BY {plan.order_sql}
                {plan.pagination_sql}
                """,
            ),
            plan.page_params,
        )
        rows = await cur.fetchall()
        has_more = len(rows) > query.limit
        rows = rows[: query.limit]
        labels_by_session_id = await load_session_labels_batch(
            cur,
            [row[0] for row in rows],
        )
        sessions = [
            pg_support.session_from_row(
                row,
                labels=labels_by_session_id.get(row[0], {}),
            )
            for row in rows
        ]
    next_cursor = session_next_cursor(sessions, has_more, query.order_by)
    return SessionListResult(sessions=sessions, next_cursor=next_cursor, total_count=total_count)


async def _query_events(
    cur: Any, query: EventQuery, *, safe_insert_xid: Any, force_snapshot_cutoff: bool = False
) -> list[EventRecord]:
    needs_snapshot_cutoff = force_snapshot_cutoff or _event_query_needs_snapshot_cutoff(query)
    if needs_snapshot_cutoff and safe_insert_xid is None:
        await cur.execute("SELECT pg_snapshot_xmin(pg_current_snapshot())")
        row = await cur.fetchone()
        if row is None:
            raise RuntimeError("Failed to read Postgres event visibility snapshot.")
        safe_insert_xid = row[0]
    extra_clauses: tuple[session_store_sql.SqlClause, ...] = ()
    if needs_snapshot_cutoff:
        # Postgres identity values are allocated at INSERT but published at COMMIT.
        # Cross-session event consumers must not advance an after_sequence cursor
        # past an event inserted by a still-open transaction with a lower identity.
        extra_clauses = (
            session_store_sql.SqlClause(
                "cayu_events.insert_xid < %s",
                (safe_insert_xid,),
            ),
        )
    plan = session_store_sql.build_event_query_sql(
        query,
        dialect=SQL_DIALECT,
        extra_after_sequence_clauses=extra_clauses,
    )
    params = [*plan.params, query.limit]

    # where_sql is built only from hard-coded clause literals; all values
    # are bound via %s params, so the assembled text carries no user input.
    await cur.execute(
        cast(
            "LiteralString",
            f"""
            SELECT cayu_events.sequence,
                   cayu_events.event,
                   cayu_events.input_contract_runtime_owned,
                   cayu_events.file_attachment_attestations_runtime_owned
            FROM cayu_events
            JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
            {plan.where_sql}
            ORDER BY cayu_events.sequence {plan.order_direction}
            LIMIT %s
            """,
        ),
        params,
    )
    rows = await cur.fetchall()
    return [
        EventRecord(
            sequence=row[0],
            event=restore_persisted_event_authority(
                Event(**pg_support._json_obj(row[1])),
                input_contract_runtime_owned=row[2],
                file_attachment_attestations_runtime_owned=row[3],
            ),
        )
        for row in rows
    ]


async def load_session_labels_batch(cur: Any, session_ids: list[str]) -> dict[str, dict[str, str]]:
    if not session_ids:
        return {}
    await cur.execute(
        """
        SELECT session_id, key, value
        FROM cayu_session_labels
        WHERE session_id = ANY(%s)
        ORDER BY session_id ASC, key ASC
        """,
        (session_ids,),
    )
    labels_by_session_id: dict[str, dict[str, str]] = {session_id: {} for session_id in session_ids}
    for row in await cur.fetchall():
        labels_by_session_id[row[0]][row[1]] = row[2]
    return labels_by_session_id
