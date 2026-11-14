"""Complete PostgreSQL session query operations with explicit native execution capabilities."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Collection, Sequence
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
from cayu.sessions._access_rules import runtime_session_query
from cayu.sessions._child_lifecycle_rules import (
    _child_session_lifecycle_entry,
    _child_session_lifecycle_entry_sort_key,
    _child_session_lifecycle_occurrence,
)
from cayu.sessions._event_snapshot import (
    EventSnapshotConsumer,
    EventSnapshotFold,
    fold_snapshot_async,
)
from cayu.sessions._runtime_publication import _tool_round_lifecycle_event_limit
from cayu.sessions._tool_round_publication_rules import (
    _TOOL_ROUND_LIFECYCLE_EVENT_TYPES,
    _validate_tool_round_call_ids,
)
from cayu.sessions.display import (
    SessionTitleQuery,
    SessionTitleSnapshot,
    copy_title_query,
    title_snapshot,
)
from cayu.sessions.environment_activity import (
    EnvironmentActivityQuery,
    EnvironmentActivitySnapshot,
    copy_environment_activity_query,
    environment_activity_snapshot,
)
from cayu.sessions.event_queries import (
    EventOrder,
    EventQuery,
    EventQueryResultTooLarge,
    copy_event_query,
)
from cayu.sessions.inspection import SESSION_INSPECTION_LABEL_LIMIT, SessionInspectionIdentity
from cayu.sessions.interactions import _validate_interaction_page
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
    EventHead,
    EventRecord,
    Session,
    SessionStatus,
    stored_event_head,
    stored_event_record,
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
from cayu.sessions.trace_events import TraceEvent, stored_trace_event, trace_events_of
from cayu.sessions.usage import UsageRollupQuery, copy_usage_rollup_query
from cayu.storage import _postgres_aggregates as postgres_aggregates
from cayu.storage import _postgres_stored_events as pg_stored_events
from cayu.storage import _postgres_support as pg_support
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage import _tool_usage_sql as tool_usage_sql

if TYPE_CHECKING:
    from cayu.budgets.usage import SessionUsageSummary
    from cayu.knowledge.usage import KnowledgeLookupTotal
    from cayu.runtime._cost_accounting_refresh import CostAccountingAuthority
    from cayu.runtime._usage_accounting import SessionUsageCache
    from cayu.sessions._access_rules import SessionReadBounds
    from cayu.sessions.tool_usage import ToolCallHeads, ToolUsageTally


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


# The highest sequence no open transaction can still commit below: the event
# sequence high-water, lowered to the smallest range an open event writer has
# reserved (revision 147). Transaction ids don't order sequences, so the
# snapshot's xmin is no such bound. The bound comes from live state, so the rows
# it filters must be read under a snapshot taken after it: a later statement
# under READ COMMITTED, or a REPEATABLE READ transaction begun after it.
_VISIBLE_SEQUENCE_SQL = "SELECT cayu_event_visible_sequence('cayu_events'::regclass)"


async def _read_visible_sequence(cur: Any) -> int:
    # The first statement of its transaction. Set the level explicitly: a role or
    # database may default to a stricter one, whose snapshot would predate the
    # bound's live reads.
    await cur.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
    await cur.execute(_VISIBLE_SEQUENCE_SQL)
    row = await cur.fetchone()
    if row is None:
        raise RuntimeError("Failed to read the Postgres event visibility bound.")
    return row[0]


async def _visible_sequence_before_snapshot(
    conn: Any, cur: Any, queries: Sequence[EventQuery]
) -> int | None:
    """Read the bound in its own transaction, before a REPEATABLE READ one begins."""

    if not any(_event_query_needs_snapshot_cutoff(query) for query in queries):
        return None
    visible_sequence = await _read_visible_sequence(cur)
    await conn.commit()
    return visible_sequence


def _event_query_needs_snapshot_cutoff(query: EventQuery) -> bool:
    return (
        query.after_sequence is not None or query.before_sequence is not None
    ) and not _event_query_is_single_session(query)


def event_record_from_row(row: tuple[Any, Any] | None) -> EventRecord | None:
    """Build an EventRecord from a ``(sequence, event)`` row, or None for a missing row."""
    if row is None:
        return None
    return pg_stored_events.event_record_from_document(row[0], row[1])


_INSPECTION_IDENTITY_COLUMNS = """
    id, agent_name, provider_name, model, parent_session_id,
    causal_budget_id, runtime_name, runtime_version, environment_name,
    status, created_at, updated_at, last_activity_at, run_epoch,
    metadata -> 'cayu:runtime_build_provenance' AS runtime_build_provenance
"""


def _inspection_identity_from_rows(
    row: Any, label_rows: list[Any], label_count: int
) -> SessionInspectionIdentity:
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


async def inspect_identity(
    connect: PostgresConnection, session_id: str
) -> SessionInspectionIdentity:
    session_id = require_clean_nonblank(session_id, "session_id")
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            f"SELECT {_INSPECTION_IDENTITY_COLUMNS} FROM cayu_sessions WHERE id = %s",
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
        return _inspection_identity_from_rows(row, label_rows, label_count)


async def inspect_identities(
    connect: PostgresConnection, session_ids: Collection[str]
) -> dict[str, SessionInspectionIdentity]:
    """``inspect_identity`` for several sessions in two reads; missing ones are left out."""

    wanted = list(
        dict.fromkeys(
            require_clean_nonblank(session_id, "session_id") for session_id in session_ids
        )
    )
    if not wanted:
        return {}
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            f"SELECT {_INSPECTION_IDENTITY_COLUMNS} FROM cayu_sessions WHERE id = ANY(%s::text[])",
            (wanted,),
        )
        rows = {row[0]: row for row in await cur.fetchall()}
        if not rows:
            return {}
        await cur.execute(
            """
            SELECT session_id, key, value, label_count
            FROM (
                SELECT session_id, key, value,
                       COUNT(*) OVER (PARTITION BY session_id) AS label_count,
                       ROW_NUMBER() OVER (
                           PARTITION BY session_id ORDER BY key COLLATE "C" ASC
                       ) AS position
                FROM cayu_session_labels
                WHERE session_id = ANY(%s::text[])
            ) AS ranked
            WHERE position <= %s
            ORDER BY session_id, position
            """,
            (list(rows), SESSION_INSPECTION_LABEL_LIMIT),
        )
        labels: dict[str, list[Any]] = {}
        label_counts: dict[str, int] = {}
        for session_id, key, value, label_count in await cur.fetchall():
            labels.setdefault(session_id, []).append((key, value))
            label_counts[session_id] = label_count
    return {
        session_id: _inspection_identity_from_rows(
            rows[session_id], labels.get(session_id, []), label_counts.get(session_id, 0)
        )
        for session_id in wanted
        if session_id in rows
    }


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
        return [pg_stored_events.event_from_document(row[0]) for row in rows]


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
        return [pg_stored_events.event_from_document(row[0]) for row in rows]


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
        return [pg_stored_events.event_from_document(row[0]) for row in rows]


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
        return [pg_stored_events.event_from_document(row[0]) for row in rows]


@runtime_session_query
async def query_events(
    connect: PostgresConnection, query: EventQuery | None = None
) -> list[EventRecord]:
    query = copy_event_query(query)
    if len(query.session_ids) > _EVENT_QUERY_SESSION_IDS_BATCH_SIZE:
        return await _query_events_by_session_id_batches(connect, query)
    async with connect() as conn, conn.cursor() as cur:
        return await _query_events(cur, query, visible_sequence=None)


@runtime_session_query
async def query_event_heads(connect: PostgresConnection, query: EventQuery) -> list[EventHead]:
    query = copy_event_query(query)
    if len(query.session_ids) > _EVENT_QUERY_SESSION_IDS_BATCH_SIZE:
        return [EventHead.of(record) for record in await query_events(connect, query)]
    async with connect() as conn, conn.cursor() as cur:
        plan, params = await _event_query_plan(cur, query, visible_sequence=None)
        # The timestamp comes from the stored event, as Event reads it.
        await cur.execute(
            cast(
                "LiteralString",
                f"""
                SELECT cayu_events.sequence,
                       cayu_events.session_id,
                       cayu_events.event_id,
                       cayu_events.event_type,
                       cayu_events.event ->> 'timestamp'
                FROM cayu_events
                JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                {plan.where_sql}
                ORDER BY cayu_events.sequence {plan.order_direction}
                LIMIT %s
                """,
            ),
            params,
        )
        return [stored_event_head(*row) for row in await cur.fetchall()]


@runtime_session_query
async def query_trace_events(connect: PostgresConnection, query: EventQuery) -> list[TraceEvent]:
    query = copy_event_query(query)
    if len(query.session_ids) > _EVENT_QUERY_SESSION_IDS_BATCH_SIZE:
        return trace_events_of(await query_events(connect, query))
    async with connect() as conn, conn.cursor() as cur:
        plan, params = await _event_query_plan(cur, query, visible_sequence=None)
        # The timestamp comes from the stored event, as Event reads it.
        await cur.execute(
            cast(
                "LiteralString",
                f"""
                SELECT cayu_events.sequence,
                       cayu_events.session_id,
                       cayu_events.event_id,
                       cayu_events.event_type,
                       cayu_events.event ->> 'timestamp',
                       cayu_events.interaction_id,
                       cayu_events.agent_name,
                       cayu_events.environment_name,
                       cayu_events.workflow_name,
                       cayu_events.tool_name,
                       trace.payload_bytes,
                       trace.payload
                FROM cayu_events
                JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                JOIN cayu_trace_events AS trace ON trace.sequence = cayu_events.sequence
                {plan.where_sql}
                ORDER BY cayu_events.sequence {plan.order_direction}
                LIMIT %s
                """,
            ),
            params,
        )
        return [
            stored_trace_event(
                stored_event_head(*row[:5]),
                interaction_id=row[5],
                agent_name=row[6],
                environment_name=row[7],
                workflow_name=row[8],
                tool_name=row[9],
                payload_bytes=row[10],
                payload=row[11],
            )
            for row in await cur.fetchall()
        ]


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
        USAGE_ACCOUNTING_FULL_PAYLOAD_TYPES,
        USAGE_ACCOUNTING_PAGE_SIZE,
        SessionUsageCache,
        UsageAccountingReducer,
        UsageAccountingRow,
        usage_accounting_query,
    )

    query = usage_accounting_query(query)
    full_payload_types = ", ".join(
        f"'{event_type}'" for event_type in USAGE_ACCOUNTING_FULL_PAYLOAD_TYPES
    )
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
                    # The narrow copy holds every payload member the reducer
                    # reads, except for an event without ``usage_metrics``,
                    # whose raw-usage normalization reads its whole payload.
                    "SELECT cayu_events.sequence, cayu_events.session_id, "
                    "cayu_events.event_type, "
                    "CASE WHEN narrow.sequence IS NULL OR ("
                    f"cayu_events.event_type IN ({full_payload_types}) "
                    "AND narrow.payload -> 'usage_metrics' IS NULL"
                    ") THEN cayu_events.event -> 'payload' ELSE narrow.payload END "
                    "FROM cayu_events "
                    "JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id "
                    "LEFT JOIN cayu_usage_events AS narrow "
                    "ON narrow.sequence = cayu_events.sequence "
                    f"{plan.where_sql} ORDER BY cayu_events.sequence ASC",
                ),
                plan.params,
            )
            while rows := await cur.fetchmany(USAGE_ACCOUNTING_PAGE_SIZE):
                reducer.add_rows(
                    [
                        UsageAccountingRow(
                            sequence=row[0],
                            session_id=row[1],
                            type=row[2],
                            payload=pg_support._json_obj(row[3]),
                        )
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
    fork_lineage_owner: str | None = None,
    without_reservation_for: str | None = None,
) -> CostAccountingSnapshot:
    from cayu.runtime._cost_accounting import (
        COST_ACCOUNTING_PAGE_SIZE,
        cost_accounting_query,
        cost_pending_events,
        require_no_pending_for_reservation_filter,
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
    filters, lineage = session_store_sql.cost_read_filter_clauses(
        fork_lineage_owner=fork_lineage_owner,
        without_reservation_for=without_reservation_for,
        dialect=SQL_DIALECT,
    )
    require_no_pending_for_reservation_filter(without_reservation_for, additional_events)
    if filters:
        # A filtered read never refreshes another read's snapshot.
        previous = None
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
            if lineage is not None and pending:
                await cur.execute(
                    cast(
                        "LiteralString",
                        f"SELECT id FROM cayu_sessions WHERE id = ANY(%s) AND {lineage.sql}",
                    ),
                    ([event.session_id for event in pending], *lineage.params),
                )
                owned_sessions = {row[0] for row in await cur.fetchall()}
                pending = tuple(event for event in pending if event.session_id in owned_sessions)

        plan = session_store_sql.build_accounting_event_query_sql(
            query, dialect=SQL_DIALECT, extra_after_sequence_clauses=filters
        )
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
            extra_after_sequence_clauses=filters,
        )
        columns = "cayu_events.sequence, cayu_events.event, cayu_events.session_id, cayu_events.event_id, cayu_events.event_type"

        def add_rows(rows: list[Any]) -> None:
            for row in rows:
                reducer.add(row[0], pg_stored_events.event_from_document(row[1]))

        async def add_group(key: tuple[str, bool, str]) -> None:
            statement, params = cost_group_lookup_statement(
                columns=columns, plan=source_plan, key=key, postgres=True
            )
            async with conn.cursor(name=f"cost_group_{uuid4().hex}") as group_cursor:
                await group_cursor.execute(cast("LiteralString", statement), params)
                while rows := await group_cursor.fetchmany(COST_ACCOUNTING_PAGE_SIZE):
                    await asyncio.to_thread(add_rows, rows)

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
                    # Each page of events is decoded and priced off the event loop.
                    await asyncio.to_thread(add_rows, rows)
        result = reducer.snapshot()
        # Global sequence allocation is not commit order. A later commit can
        # become visible below this snapshot's maximum sequence, so it is not
        # a safe incremental baseline across independently written sessions.
        # A filtered read's cursor would carry only the EventQuery's scope, so
        # an ordinary read could adopt it as a baseline it never counted.
        if filters or not _event_query_is_single_session(query):
            result = result.model_copy(update={"cursor": None})
        return result


@runtime_session_query
async def query_events_bounded(
    connect: PostgresConnection, query: EventQuery, *, max_bytes: int
) -> list[EventRecord]:
    fold = EventSnapshotFold(query, max_bytes=max_bytes, max_pages=1)
    async with connect() as conn, conn.cursor() as cur:
        visible_sequence = await _visible_sequence_before_snapshot(conn, cur, (fold.query,))
        await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        return await _query_events_bounded_snapshot(
            cur,
            fold.query,
            max_bytes=max_bytes,
            exact_snapshot=False,
            visible_sequence=visible_sequence,
        )


@runtime_session_query
async def query_events_bounded_many(
    connect: PostgresConnection, queries: Sequence[EventQuery], *, max_bytes: int
) -> list[list[EventRecord] | EventQueryResultTooLarge]:
    bounded = [
        EventSnapshotFold(query, max_bytes=max_bytes, max_pages=1).query for query in queries
    ]
    results: list[list[EventRecord] | EventQueryResultTooLarge] = []
    async with connect() as conn, conn.cursor() as cur:
        visible_sequence = await _visible_sequence_before_snapshot(conn, cur, bounded)
        await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        for query in bounded:
            try:
                results.append(
                    await _query_events_bounded_snapshot(
                        cur,
                        query,
                        max_bytes=max_bytes,
                        exact_snapshot=False,
                        visible_sequence=visible_sequence,
                    )
                )
            except EventQueryResultTooLarge as error:
                results.append(error)
    return results


@runtime_session_query
async def fold_event_snapshot(
    connect: PostgresConnection,
    query: EventQuery,
    consume: EventSnapshotConsumer,
    *,
    max_bytes: int,
    max_pages: int | None = None,
) -> int:
    fold = EventSnapshotFold(query, max_bytes=max_bytes, max_pages=max_pages)
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")

        async def read(page: EventQuery) -> list[EventRecord]:
            return await _query_events_bounded_snapshot(
                cur, page, max_bytes=max_bytes, exact_snapshot=True
            )

        return await fold_snapshot_async(fold, read, consume)


@runtime_session_query
async def stored_knowledge_lookup_totals(
    connect: PostgresConnection,
) -> dict[tuple[str, str], KnowledgeLookupTotal]:
    """Every row of ``cayu_knowledge_lookup_totals``, for the recompute check."""

    from cayu.knowledge.usage import KnowledgeLookupTotal

    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT namespace, entry_id, lookups, sessions, last_lookup_at "
            "FROM cayu_knowledge_lookup_totals"
        )
        rows = await cur.fetchall()
    return {
        (namespace, entry_id): KnowledgeLookupTotal(
            int(lookups), int(sessions), pg_support.to_utc(last)
        )
        for namespace, entry_id, lookups, sessions, last in rows
    }


@runtime_session_query
async def knowledge_lookup_totals(
    connect: PostgresConnection, identities: Collection[tuple[str, str]]
) -> tuple[dict[tuple[str, str], tuple[int, datetime | None]], int]:
    """Read lookup counts for the requested pairs from ``cayu_knowledge_lookup_totals``.

    Matches ``fold_knowledge_lookup_totals``; the totals are trigger-maintained
    and folded at commit, so a snapshot sees each committed lookup exactly once.
    The sequence is the snapshot's newest event of any type.
    """

    wanted = list(dict.fromkeys(identities))
    totals: dict[tuple[str, str], tuple[int, datetime | None]] = dict.fromkeys(wanted, (0, None))
    newest_plan = session_store_sql.build_event_query_sql(
        EventQuery(limit=1, order_by=EventOrder.SEQUENCE_DESC), dialect=SQL_DIALECT
    )
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        await cur.execute(
            cast(
                "LiteralString",
                f"""
                SELECT cayu_events.sequence
                FROM cayu_events
                JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                {newest_plan.where_sql}
                ORDER BY cayu_events.sequence DESC
                LIMIT 1
                """,
            ),
            newest_plan.params,
        )
        newest = await cur.fetchone()
        if wanted:
            await cur.execute(
                """
                SELECT totals.namespace, totals.entry_id, totals.lookups, totals.last_lookup_at
                FROM unnest(%s::text[], %s::text[]) AS wanted(namespace, entry_id)
                JOIN cayu_knowledge_lookup_totals AS totals
                  ON totals.namespace = wanted.namespace AND totals.entry_id = wanted.entry_id
                """,
                (
                    [namespace for namespace, _ in wanted],
                    [entry_id for _, entry_id in wanted],
                ),
            )
            for namespace, entry_id, count, last in await cur.fetchall():
                totals[(namespace, entry_id)] = int(count), pg_support.to_utc(last)
    return totals, 0 if newest is None else newest[0]


@runtime_session_query
async def compact_usage_projection(connect: PostgresConnection) -> None:
    """Fold pending hourly and tool deltas into their shared rows, if no one else is.

    Event writers append deltas instead of updating shared rows, so they never
    wait on each other; reads add pending deltas either way, so this only keeps
    the pending tables short.
    """

    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await cur.execute("SELECT cayu_compact_usage_projection()")
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise


async def tool_usage_totals(
    connect: PostgresConnection,
    *,
    from_events: bool = False,
) -> tuple[dict[str, ToolUsageTally], int]:
    """Tally retained tool usage in one grouped statement inside one snapshot.

    Sums the trigger-kept ``cayu_tool_usage`` rows. ``from_events`` counts the
    narrow ``cayu_usage_events`` copy instead, the reference the stored totals
    match; both match ``fold_tool_usage_totals`` without loading event documents.
    """

    from cayu.sessions.tool_usage import ToolUsageTally

    if not from_events:
        await compact_usage_projection(connect)
    statements = tool_usage_sql.tool_usage_totals_sql(
        dialect=SQL_DIALECT,
        payload_text=lambda key: f"(cayu_events.payload ->> '{key}')",
        payload_is_string=lambda key: f"jsonb_typeof(cayu_events.payload -> '{key}') = 'string'",
    )
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        await cur.execute(
            cast(
                "LiteralString",
                f"""
                SELECT cayu_events.sequence
                FROM cayu_events
                JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                {statements.newest.where_sql}
                ORDER BY cayu_events.sequence DESC
                LIMIT 1
                """,
            ),
            statements.newest.params,
        )
        newest = await cur.fetchone()
        if not from_events:
            await cur.execute(cast("LiteralString", tool_usage_sql.POSTGRES_TOOL_USAGE_RECHECK_SQL))
            recheck = await cur.fetchone()
            # A call removed after the last compaction leaves a newest-call time
            # only events can give.
            from_events = bool(recheck and recheck[0])
        if from_events:
            await cur.execute(
                cast("LiteralString", statements.totals_sql), statements.totals_params
            )
        else:
            await cur.execute(
                cast("LiteralString", tool_usage_sql.POSTGRES_STORED_TOOL_USAGE_TOTALS_SQL)
            )
        rows = await cur.fetchall()
    return (
        {
            str(tool): ToolUsageTally(
                calls=int(calls),
                failures=int(failures),
                approved=int(approved),
                rejected=int(rejected),
                last_used_at=None if last is None else pg_support.to_utc(last),
            )
            for tool, calls, failures, approved, rejected, last in rows
        },
        0 if newest is None else int(newest[0]),
    )


async def read_session_usage_totals(
    connect: PostgresConnection,
    session_ids: tuple[str, ...],
    *,
    since: datetime | None,
    until: datetime | None,
) -> tuple[dict[str, SessionUsageSummary], set[str]]:
    """Stored per-session usage, and the sessions to read from events instead."""

    from cayu.storage import _usage_projection_reads as projection_reads

    rows: list[dict[str, Any]] = []
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        for offset in range(0, len(session_ids), 1000):
            sql, params = projection_reads.session_totals_statement(
                "postgres",
                session_ids=session_ids[offset : offset + 1000],
                since=since,
                until=until,
            )
            if not sql:
                continue
            await cur.execute(cast("LiteralString", sql), params)
            names = [column.name for column in cur.description]
            rows.extend(dict(zip(names, row, strict=True)) for row in await cur.fetchall())
    return projection_reads.session_summaries_from_rows(rows)


async def tool_call_heads(
    connect: PostgresConnection,
    raw_names: tuple[str, ...] | None,
    *,
    before_sequence: int | None,
    limit: int,
) -> ToolCallHeads:
    """Newest terminal call heads by raw tool name, from indexed ``cayu_usage_events``."""

    from cayu.sessions.tool_usage import ToolCallHead, ToolCallHeads

    sql, params = tool_usage_sql.tool_call_heads_sql(
        dialect=SQL_DIALECT,
        raw_names=raw_names,
        before_sequence=before_sequence,
        limit=limit,
        payload_text=lambda key: f"(payload ->> '{key}')",
        payload_is_string=lambda key: f"jsonb_typeof(payload -> '{key}') = 'string'",
    )
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        await cur.execute("SELECT MAX(sequence) FROM cayu_events")
        newest_row = await cur.fetchone()
        rows: list[Any] = []
        if params:
            await cur.execute(cast("LiteralString", sql), params)
            rows = list(await cur.fetchall())
    return ToolCallHeads(
        heads=tuple(
            ToolCallHead(
                sequence=int(sequence),
                session_id=session_id,
                event_id=event_id,
                tool_name=tool_name,
            )
            for sequence, session_id, event_id, tool_name in rows
        ),
        newest_sequence=0 if newest_row is None or newest_row[0] is None else int(newest_row[0]),
    )


async def _query_events_bounded_snapshot(
    cur: Any,
    query: EventQuery,
    *,
    max_bytes: int,
    exact_snapshot: bool,
    visible_sequence: int | None = None,
) -> list[EventRecord]:
    if len(query.session_ids) > _EVENT_QUERY_SESSION_IDS_BATCH_SIZE:
        raise ValueError("Byte-bounded event queries require one bounded SQL batch.")
    needs_snapshot_cutoff = not exact_snapshot and _event_query_needs_snapshot_cutoff(query)
    extra_clauses: tuple[session_store_sql.SqlClause, ...] = ()
    if needs_snapshot_cutoff:
        # This runs under REPEATABLE READ, whose snapshot is already taken, so
        # the caller reads the bound before that transaction begins.
        if visible_sequence is None:
            raise RuntimeError("Read the event visibility bound before the snapshot.")
        extra_clauses = (
            session_store_sql.SqlClause(
                "cayu_events.sequence <= %s",
                (visible_sequence,),
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
        visible_sequence=visible_sequence,
        force_snapshot_cutoff=needs_snapshot_cutoff,
        exact_snapshot=exact_snapshot,
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
        return [pg_stored_events.event_record_from_document(row[0], row[1]) for row in rows]


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
                        event = pg_stored_events.event_from_document(event_row[2])
                        records_by_child[str(event_row[0])][EventType(event.type)] = (
                            stored_event_record(event_row[1], event)
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


async def read_session_titles(
    connect: PostgresConnection, query: SessionTitleQuery
) -> SessionTitleSnapshot:
    query = copy_title_query(query)
    async with connect() as conn:
        try:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """SELECT id, CASE
                    WHEN jsonb_typeof(metadata -> 'cayu:title') = 'string'
                    AND length(metadata ->> 'cayu:title') <= 256
                    THEN metadata ->> 'cayu:title' END AS title
                    FROM cayu_sessions WHERE id = ANY(%s::text[])""",
                    (list(query.session_ids),),
                )
                rows = await cursor.fetchall()
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
    return title_snapshot(query, {row[0]: row[1] for row in rows})


async def aggregate_environment_activity(
    connect: PostgresConnection, query: EnvironmentActivityQuery
) -> EnvironmentActivitySnapshot:
    query = copy_environment_activity_query(query)
    async with connect() as conn:
        try:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    WITH snapshot AS (SELECT statement_timestamp() AS as_of),
                    activity AS (
                        SELECT environment_name, COUNT(*) AS session_count, MAX(last_activity_at) AS last_activity_at
                        FROM cayu_sessions WHERE environment_name = ANY(%s::text[])
                        GROUP BY environment_name
                    )
                    SELECT snapshot.as_of, activity.* FROM snapshot LEFT JOIN activity ON TRUE
                """,
                    (list(query.environment_names),),
                )
                rows = await cursor.fetchall()
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
    values = {row[1]: (row[2], row[3]) for row in rows if row[1] is not None}
    return environment_activity_snapshot(query, rows[0][0], values)


async def aggregate_usage(
    connect: PostgresConnection, query: UsageRollupQuery, *, from_events: bool = False
) -> UsageRollupStoreResult:
    query = copy_usage_rollup_query(query)
    plan = session_store_sql.build_session_query_sql(
        session_query_from_aggregate_filter(query.sessions),
        dialect=SQL_DIALECT,
    )
    if not from_events:
        await compact_usage_projection(connect)
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
                from_events=from_events,
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
        visible_sequence = None
        needs_snapshot_cutoff = query.after_sequence is not None
        if needs_snapshot_cutoff:
            visible_sequence = await _read_visible_sequence(cur)
        for batch in _event_query_session_id_batches(query.session_ids):
            records.extend(
                await _query_events(
                    cur,
                    session_store_sql.event_query_with_session_ids(
                        query,
                        session_ids=batch,
                    ),
                    visible_sequence=visible_sequence,
                    force_snapshot_cutoff=needs_snapshot_cutoff,
                )
            )
    records.sort(
        key=lambda record: record.sequence,
        reverse=query.order_by.value == "sequence_desc",
    )
    return records[: query.limit]


_CHECKPOINT_MARKER_SESSION_SOURCES = {
    marker: f"""
        (
            SELECT session_id
            FROM cayu_checkpoints
            WHERE state ? '{marker}'
        ) AS checkpoint_markers
        INNER JOIN cayu_sessions
            ON cayu_sessions.id = checkpoint_markers.session_id
        """
    for marker in ("pending_interruption_cascade", "invocation_terminal_decision")
}


async def list_sessions(
    connect: PostgresConnection,
    query: SessionQuery | None,
    *,
    checkpoint_marker: str | None,
    access_bounds: SessionReadBounds | None = None,
) -> SessionListResult:
    if checkpoint_marker is None:
        session_source_sql = "cayu_sessions"
    elif checkpoint_marker in _CHECKPOINT_MARKER_SESSION_SOURCES:
        session_source_sql = _CHECKPOINT_MARKER_SESSION_SOURCES[checkpoint_marker]
    else:
        raise ValueError(f"Unsupported session checkpoint marker: {checkpoint_marker!r}.")
    query = copy_session_query(query)
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


async def _event_query_plan(
    cur: Any,
    query: EventQuery,
    *,
    visible_sequence: Any,
    force_snapshot_cutoff: bool = False,
    exact_snapshot: bool = False,
) -> tuple[session_store_sql.EventQuerySqlPlan, list[object]]:
    """The WHERE clause and parameters of an event query, its LIMIT value last."""

    needs_snapshot_cutoff = not exact_snapshot and (
        force_snapshot_cutoff or _event_query_needs_snapshot_cutoff(query)
    )
    if needs_snapshot_cutoff and visible_sequence is None:
        visible_sequence = await _read_visible_sequence(cur)
    extra_clauses: tuple[session_store_sql.SqlClause, ...] = ()
    if needs_snapshot_cutoff:
        # Postgres identity values are allocated at INSERT but published at COMMIT.
        # Cross-session event consumers must not advance an after_sequence cursor
        # past an event a still-open transaction can commit with a lower sequence,
        # so they read only up to the bound cayu_event_visible_sequence() reports.
        extra_clauses = (
            session_store_sql.SqlClause(
                "cayu_events.sequence <= %s",
                (visible_sequence,),
            ),
        )
    plan = session_store_sql.build_event_query_sql(
        query,
        dialect=SQL_DIALECT,
        extra_after_sequence_clauses=extra_clauses,
    )
    return plan, [*plan.params, query.limit]


async def _query_events(
    cur: Any,
    query: EventQuery,
    *,
    visible_sequence: Any,
    force_snapshot_cutoff: bool = False,
    exact_snapshot: bool = False,
) -> list[EventRecord]:
    plan, params = await _event_query_plan(
        cur,
        query,
        visible_sequence=visible_sequence,
        force_snapshot_cutoff=force_snapshot_cutoff,
        exact_snapshot=exact_snapshot,
    )
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
        pg_stored_events.event_record_from_document(
            row[0],
            row[1],
            restore_authority=True,
            input_contract_runtime_owned=row[2],
            file_attachment_attestations_runtime_owned=row[3],
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
