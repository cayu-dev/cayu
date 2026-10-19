from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from cayu.events import EventType
from cayu.sessions.event_queries import EventOrder, EventQuery, copy_event_query
from cayu.sessions.queries import (
    LabelSelectorOperator,
    LabelSelectorRequirement,
    SessionDebugState,
    SessionOrder,
    SessionQuery,
    copy_session_query,
    decode_session_cursor,
    session_order_is_descending,
    session_sort_column,
)
from cayu.sessions.records import SessionStatus
from cayu.workflows.base import WORKFLOW_ATTEMPT_EVENT_TYPE

if TYPE_CHECKING:
    from cayu.sessions._access_rules import SessionReadBounds
    from cayu.sessions._fork_controller_bounds import _ForkControllerReadBounds


@dataclass(frozen=True)
class SessionStoreSqlDialect:
    placeholder: str
    contains_style: Literal["sqlite_nocase_like", "postgres_ilike"]
    datetime_param: Callable[[datetime], object]

    def placeholders(self, count: int) -> str:
        return ", ".join(self.placeholder for _ in range(count))

    def contains_clause(self, column: str) -> str:
        if self.contains_style == "sqlite_nocase_like":
            return f"{column} COLLATE NOCASE LIKE {self.placeholder} ESCAPE '\\'"
        return f"{column} ILIKE {self.placeholder} ESCAPE '\\'"

    def event_payload_text(
        self,
        key: Literal[
            "attempt_id", "step_id", "budget_limit_id", "model_step_id", "parent_model_step_id"
        ],
    ) -> str:
        if self.contains_style == "sqlite_nocase_like":
            return f"json_extract(cayu_events.payload_json, '$.{key}')"
        return f"(cayu_events.event -> 'payload' ->> '{key}')"

    def workflow_attempt_fence_clause(self) -> str:
        if self.contains_style == "sqlite_nocase_like":
            attempt_payload = "json_extract(workflow_attempt.payload_json, '$.attempt_id')"
        else:
            attempt_payload = "(workflow_attempt.event -> 'payload' ->> 'attempt_id')"
        return (
            f"{self.event_payload_text('attempt_id')} = ("
            f"SELECT {attempt_payload} FROM cayu_events AS workflow_attempt "
            "WHERE workflow_attempt.session_id = cayu_events.session_id "
            "AND workflow_attempt.workflow_name = cayu_events.workflow_name "
            f"AND workflow_attempt.event_type = '{WORKFLOW_ATTEMPT_EVENT_TYPE}' "
            "AND workflow_attempt.sequence <= cayu_events.sequence "
            "ORDER BY workflow_attempt.sequence DESC LIMIT 1)"
        )

    def workflow_step_payload_valid_clause(self) -> str | None:
        if self.contains_style == "sqlite_nocase_like":
            return "json_valid(cayu_events.payload_json)"
        return None


@dataclass(frozen=True)
class SqlClause:
    sql: str
    params: tuple[object, ...] = ()


@dataclass(frozen=True)
class EventQuerySqlPlan:
    where_sql: str
    params: tuple[object, ...]
    order_direction: str


@dataclass(frozen=True)
class SessionQuerySqlPlan:
    filter_where_sql: str
    filter_params: tuple[object, ...]
    page_where_sql: str
    page_params: tuple[object, ...]
    order_sql: str
    pagination_sql: str


def event_query_with_session_ids(
    query: EventQuery,
    *,
    session_ids: tuple[str, ...],
) -> EventQuery:
    return copy_event_query(query, update={"session_ids": session_ids})


def session_order_sql(order_by: SessionOrder) -> str:
    direction = "DESC" if session_order_is_descending(order_by) else "ASC"
    return f"{session_sort_column(order_by)} {direction}, id ASC"


def build_accounting_event_query_sql(
    query: EventQuery,
    *,
    dialect: SessionStoreSqlDialect,
    extra_after_sequence_clauses: Sequence[SqlClause] = (),
) -> EventQuerySqlPlan:
    """Keep an arbitrarily large caller identity set inside one read snapshot."""
    query = copy_event_query(query)
    clauses = list(extra_after_sequence_clauses)
    if query.session_ids:
        if dialect.contains_style == "sqlite_nocase_like":
            clauses.append(
                SqlClause(
                    "cayu_events.session_id IN (SELECT value FROM json_each(?))",
                    (json.dumps(query.session_ids),),
                )
            )
        else:
            clauses.append(
                SqlClause("cayu_events.session_id = ANY(%s)", (list(query.session_ids),))
            )
        query = copy_event_query(query, update={"session_ids": ()})
    return build_event_query_sql(query, dialect=dialect, extra_after_sequence_clauses=clauses)


def build_event_query_sql(
    query: EventQuery | None,
    *,
    dialect: SessionStoreSqlDialect,
    extra_after_sequence_clauses: Sequence[SqlClause] = (),
) -> EventQuerySqlPlan:
    query = copy_event_query(query)
    clauses: list[str] = []
    params: list[object] = []
    from cayu.sessions._access_rules import _query_bounds
    from cayu.sessions._fork_controller_bounds import _ForkControllerReadBounds

    access_bounds = _query_bounds.get()
    if access_bounds is not None:
        clause = session_access_clause(access_bounds, dialect=dialect)
        clauses.append(
            "EXISTS (SELECT 1 FROM cayu_sessions WHERE cayu_sessions.id = cayu_events.session_id AND ("
            + clause.sql
            + "))"
        )
        params.extend(clause.params)
        if type(access_bounds) is _ForkControllerReadBounds and access_bounds.exclude_forked:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM cayu_events AS fork_marker "
                "WHERE fork_marker.session_id = cayu_events.session_id "
                f"AND fork_marker.event_type = {dialect.placeholder})"
            )
            params.append(str(EventType.SESSION_FORKED))
            clauses.append(
                "EXISTS (SELECT 1 FROM cayu_sessions WHERE "
                "cayu_sessions.id = cayu_events.session_id AND "
                + fork_lineage_unforked_sql(dialect)
                + ")"
            )

    if query.after_sequence is not None:
        clauses.append(f"cayu_events.sequence > {dialect.placeholder}")
        params.append(query.after_sequence)
    if query.before_sequence is not None:
        clauses.append(f"cayu_events.sequence < {dialect.placeholder}")
        params.append(query.before_sequence)
    _append_clauses(clauses, params, extra_after_sequence_clauses)
    if query.event_id is not None:
        clauses.append(f"cayu_events.event_id = {dialect.placeholder}")
        params.append(query.event_id)
    if query.interaction_id is not None:
        clauses.append(f"cayu_events.interaction_id = {dialect.placeholder}")
        params.append(query.interaction_id)
    if query.session_id is not None:
        clauses.append(f"cayu_events.session_id = {dialect.placeholder}")
        params.append(query.session_id)
    if query.session_ids:
        clauses.append(
            f"cayu_events.session_id IN ({dialect.placeholders(len(query.session_ids))})"
        )
        params.extend(query.session_ids)
    if query.causal_budget_id is not None:
        clauses.append(f"cayu_sessions.causal_budget_id = {dialect.placeholder}")
        params.append(query.causal_budget_id)
    if query.since is not None:
        clauses.append(f"cayu_events.timestamp >= {dialect.placeholder}")
        params.append(dialect.datetime_param(query.since))
    if query.until is not None:
        clauses.append(f"cayu_events.timestamp < {dialect.placeholder}")
        params.append(dialect.datetime_param(query.until))
    if query.event_type is not None:
        clauses.append(f"cayu_events.event_type = {dialect.placeholder}")
        params.append(str(query.event_type))
    if query.budget_limit_id is not None:
        if dialect.contains_style == "sqlite_nocase_like":
            clauses.append("json_type(cayu_events.payload_json, '$.budget_limit_id') = 'text'")
        else:
            clauses.append(
                "jsonb_typeof(cayu_events.event -> 'payload' -> 'budget_limit_id') = 'string'"
            )
        clauses.append(f"{dialect.event_payload_text('budget_limit_id')} = {dialect.placeholder}")
        params.append(query.budget_limit_id)
    if query.event_types:
        clauses.append(
            f"cayu_events.event_type IN ({dialect.placeholders(len(query.event_types))})"
        )
        params.extend(str(event_type) for event_type in query.event_types)
    if query.exclude_event_types:
        clauses.append(
            "cayu_events.event_type NOT IN "
            f"({dialect.placeholders(len(query.exclude_event_types))})"
        )
        params.extend(str(event_type) for event_type in query.exclude_event_types)
    if query.agent_name is not None:
        clauses.append(f"cayu_events.agent_name = {dialect.placeholder}")
        params.append(query.agent_name)
    if query.environment_name is not None:
        clauses.append(f"cayu_events.environment_name = {dialect.placeholder}")
        params.append(query.environment_name)
    if query.workflow_name is not None:
        clauses.append(f"cayu_events.workflow_name = {dialect.placeholder}")
        params.append(query.workflow_name)
    for value, key in (
        (query.model_step_id, "model_step_id"),
        (query.parent_model_step_id, "parent_model_step_id"),
        (query.operation_attempt_id, "attempt_id"),
    ):
        if value is not None:
            if dialect.contains_style == "sqlite_nocase_like":
                clauses.append(f"json_type(cayu_events.payload_json, '$.{key}') = 'text'")
            else:
                clauses.append(
                    f"jsonb_typeof(cayu_events.event -> 'payload' -> '{key}') = 'string'"
                )
            clauses.append(f"{dialect.event_payload_text(key)} = {dialect.placeholder}")
            params.append(value)
    if query.workflow_attempt_id is not None:
        clauses.append(f"{dialect.event_payload_text('attempt_id')} = {dialect.placeholder}")
        params.append(query.workflow_attempt_id)
    if query.workflow_step_id is not None:
        payload_valid_clause = dialect.workflow_step_payload_valid_clause()
        if payload_valid_clause is not None:
            clauses.append(payload_valid_clause)
        clauses.append(f"{dialect.event_payload_text('step_id')} = {dialect.placeholder}")
        params.append(query.workflow_step_id)
    if query.workflow_attempt_fenced:
        clauses.append(dialect.workflow_attempt_fence_clause())
    if query.tool_name is not None:
        clauses.append(f"cayu_events.tool_name = {dialect.placeholder}")
        params.append(query.tool_name)

    return EventQuerySqlPlan(
        where_sql=_where_sql(clauses),
        params=tuple(params),
        order_direction="DESC" if query.order_by == EventOrder.SEQUENCE_DESC else "ASC",
    )


def build_session_query_sql(
    query: SessionQuery | None,
    *,
    dialect: SessionStoreSqlDialect,
    access_clause: SqlClause | None = None,
) -> SessionQuerySqlPlan:
    query = copy_session_query(query)
    clauses: list[str] = []
    params: list[object] = []

    if access_clause is not None:
        clauses.append(access_clause.sql)
        params.extend(access_clause.params)

    if query.q is not None:
        like = _contains_pattern(query.q)
        clauses.append(_session_search_clause(dialect))
        params.extend([like, like, like, like, like, like, like, like, like])
    if query.status is not None:
        clauses.append(f"status = {dialect.placeholder}")
        params.append(str(query.status))
    if query.debug_state is not None:
        _append_debug_state_clause(query.debug_state, clauses, params, dialect)
    _append_session_field_filter(
        clauses,
        params,
        dialect=dialect,
        column="agent_name",
        value=query.agent_name,
    )
    _append_session_field_filter(
        clauses,
        params,
        dialect=dialect,
        column="provider_name",
        value=query.provider_name,
    )
    _append_session_field_filter(
        clauses,
        params,
        dialect=dialect,
        column="model",
        value=query.model,
    )
    _append_session_field_filter(
        clauses,
        params,
        dialect=dialect,
        column="environment_name",
        value=query.environment_name,
    )
    _append_session_field_filter(
        clauses,
        params,
        dialect=dialect,
        column="parent_session_id",
        value=query.parent_session_id,
    )
    _append_session_field_filter(
        clauses,
        params,
        dialect=dialect,
        column="causal_budget_id",
        value=query.causal_budget_id,
    )
    if query.created_after is not None:
        clauses.append(f"created_at > {dialect.placeholder}")
        params.append(dialect.datetime_param(query.created_after))
    if query.last_activity_before is not None:
        clauses.append(f"last_activity_at <= {dialect.placeholder}")
        params.append(dialect.datetime_param(query.last_activity_before))
    for key, value in query.labels.items():
        clauses.append(_label_exact_clause(dialect))
        params.extend([key, value])
    for selector in query.label_selectors:
        if selector.operator == LabelSelectorOperator.EXISTS:
            clauses.append(_label_exists_clause(dialect, negated=False))
            params.append(selector.key)
        elif selector.operator == LabelSelectorOperator.NOT_EXISTS:
            clauses.append(_label_exists_clause(dialect, negated=True))
            params.append(selector.key)
        else:
            exists_sql = _label_in_clause(dialect, value_count=len(selector.values))
            if selector.operator == LabelSelectorOperator.IN:
                clauses.append(exists_sql)
            elif selector.operator == LabelSelectorOperator.NOT_IN:
                clauses.append(f"NOT {exists_sql}")
            else:
                raise ValueError(f"Unsupported label selector operator: {selector.operator}")
            params.extend([selector.key, *selector.values])

    page_clauses = list(clauses)
    page_params = list(params)
    sort_column = session_sort_column(query.order_by)
    if query.cursor is not None:
        cursor_dt, cursor_id = decode_session_cursor(query.cursor)
        cursor_value = dialect.datetime_param(cursor_dt)
        comparison = "<" if session_order_is_descending(query.order_by) else ">"
        page_clauses.append(
            f"(({sort_column} {comparison} {dialect.placeholder}) "
            f"OR ({sort_column} = {dialect.placeholder} AND id > {dialect.placeholder}))"
        )
        page_params.extend([cursor_value, cursor_value, cursor_id, query.limit + 1])
        pagination_sql = f"LIMIT {dialect.placeholder}"
    else:
        page_params.extend([query.limit + 1, query.offset])
        pagination_sql = f"LIMIT {dialect.placeholder} OFFSET {dialect.placeholder}"

    return SessionQuerySqlPlan(
        filter_where_sql=_where_sql(clauses),
        filter_params=tuple(params),
        page_where_sql=_where_sql(page_clauses),
        page_params=tuple(page_params),
        order_sql=session_order_sql(query.order_by),
        pagination_sql=pagination_sql,
    )


def _append_clauses(
    clauses: list[str],
    params: list[object],
    extra_clauses: Sequence[SqlClause],
) -> None:
    for clause in extra_clauses:
        clauses.append(clause.sql)
        params.extend(clause.params)


def _contains_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _where_sql(clauses: list[str]) -> str:
    return f"WHERE {' AND '.join(clauses)}" if clauses else ""


def _session_search_clause(dialect: SessionStoreSqlDialect) -> str:
    fields = (
        "id",
        "agent_name",
        "provider_name",
        "model",
        "environment_name",
        "parent_session_id",
        "causal_budget_id",
    )
    field_clauses = [dialect.contains_clause(field) for field in fields]
    label_key_clause = dialect.contains_clause("cayu_session_labels.key")
    label_value_clause = dialect.contains_clause("cayu_session_labels.value")
    return f"""
        (
            {" OR ".join(field_clauses)}
            OR EXISTS (
                SELECT 1
                FROM cayu_session_labels
                WHERE cayu_session_labels.session_id = cayu_sessions.id
                  AND (
                    {label_key_clause}
                    OR {label_value_clause}
                  )
            )
        )
        """


def _append_debug_state_clause(
    debug_state: SessionDebugState,
    clauses: list[str],
    params: list[object],
    dialect: SessionStoreSqlDialect,
) -> None:
    tool_debug_sql = f"""
        EXISTS (
            SELECT 1
            FROM cayu_events
            WHERE cayu_events.session_id = cayu_sessions.id
              AND cayu_events.event_type IN ({dialect.placeholder}, {dialect.placeholder})
        )
        """
    if debug_state == SessionDebugState.TOOL_ISSUE:
        clauses.append(tool_debug_sql)
        params.extend([str(EventType.TOOL_CALL_FAILED), str(EventType.TOOL_CALL_BLOCKED)])
    elif debug_state == SessionDebugState.SESSION_FAILURE:
        clauses.append(f"status = {dialect.placeholder}")
        params.append(str(SessionStatus.FAILED))
    elif debug_state == SessionDebugState.INTERRUPTION:
        clauses.append(f"status = {dialect.placeholder}")
        params.append(str(SessionStatus.INTERRUPTED))
    elif debug_state == SessionDebugState.NEEDS_ATTENTION:
        clauses.append(
            f"(status IN ({dialect.placeholder}, {dialect.placeholder}) OR {tool_debug_sql})"
        )
        params.extend(
            [
                str(SessionStatus.FAILED),
                str(SessionStatus.INTERRUPTED),
                str(EventType.TOOL_CALL_FAILED),
                str(EventType.TOOL_CALL_BLOCKED),
            ]
        )
    else:
        raise ValueError(f"Unsupported session debug_state: {debug_state}")


def _append_session_field_filter(
    clauses: list[str],
    params: list[object],
    *,
    dialect: SessionStoreSqlDialect,
    column: str,
    value: str | None,
) -> None:
    if value is not None:
        clauses.append(f"{column} = {dialect.placeholder}")
        params.append(value)


def _label_exact_clause(dialect: SessionStoreSqlDialect) -> str:
    return f"""
        EXISTS (
            SELECT 1
            FROM cayu_session_labels
            WHERE cayu_session_labels.session_id = cayu_sessions.id
              AND cayu_session_labels.key = {dialect.placeholder}
              AND cayu_session_labels.value = {dialect.placeholder}
        )
        """


def _label_exists_clause(dialect: SessionStoreSqlDialect, *, negated: bool) -> str:
    prefix = "NOT " if negated else ""
    return f"""
        {prefix}EXISTS (
            SELECT 1
            FROM cayu_session_labels
            WHERE cayu_session_labels.session_id = cayu_sessions.id
              AND cayu_session_labels.key = {dialect.placeholder}
        )
        """


def _label_in_clause(dialect: SessionStoreSqlDialect, *, value_count: int) -> str:
    return f"""
        EXISTS (
            SELECT 1
            FROM cayu_session_labels
            WHERE cayu_session_labels.session_id = cayu_sessions.id
              AND cayu_session_labels.key = {dialect.placeholder}
              AND cayu_session_labels.value IN ({dialect.placeholders(value_count)})
        )
        """


def _json_type(dialect: SessionStoreSqlDialect, column: str, *keys: str) -> str:
    """SQL naming the JSON type at ``keys`` inside ``column``; NULL when absent."""

    if dialect.contains_style == "sqlite_nocase_like":
        return f"json_type({column}, '$." + ".".join(f'"{key}"' for key in keys) + "')"
    return f"jsonb_typeof({column}" + "".join(f" -> '{key}'" for key in keys) + ")"


def _json_text(dialect: SessionStoreSqlDialect, column: str, *keys: str) -> str:
    """SQL reading the scalar at ``keys`` inside ``column`` as text."""

    if dialect.contains_style == "sqlite_nocase_like":
        return f"json_extract({column}, '$." + ".".join(f'"{key}"' for key in keys) + "')"
    return f"({column} #>> '{{" + ",".join(keys) + "}')"


def _present(kind: str) -> str:
    # Mirrors the Python ``metadata.get(key) is not None``: JSON null is absent.
    return f"({kind} IS NOT NULL AND {kind} <> 'null')"


def _carries_fork_lineage(dialect: SessionStoreSqlDialect, metadata: str) -> str:
    """True for a row carrying an overlay record or a lineage marker.

    Such a row settles the parent walk, even when its marker says unforked.
    """

    from cayu.sessions.fork_overlays import FORK_LINEAGE_METADATA_KEY, FORK_OVERLAY_METADATA_KEY

    return (
        "("
        + " OR ".join(
            _present(_json_type(dialect, metadata, key))
            for key in (FORK_OVERLAY_METADATA_KEY, FORK_LINEAGE_METADATA_KEY)
        )
        + ")"
    )


def _fork_lineage_entries_admit(dialect: SessionStoreSqlDialect, metadata: str) -> str:
    """Compile ``fork_controller_lineage_admits`` for one row; binds the controller twice."""

    from cayu.sessions.fork_overlays import FORK_LINEAGE_METADATA_KEY, FORK_OVERLAY_METADATA_KEY

    sqlite = dialect.contains_style == "sqlite_nocase_like"
    string = "'text'" if sqlite else "'string'"
    record = _json_type(dialect, metadata, FORK_OVERLAY_METADATA_KEY)
    named = (FORK_OVERLAY_METADATA_KEY, "overlay", "controller")
    marker = _json_type(dialect, metadata, FORK_LINEAGE_METADATA_KEY)
    forked = (FORK_LINEAGE_METADATA_KEY, "forked")
    owner = (FORK_LINEAGE_METADATA_KEY, "controller")
    forked_true = (
        f"{_json_type(dialect, metadata, *forked)} = 'true'"
        if sqlite
        else f"{_json_type(dialect, metadata, *forked)} = 'boolean' AND "
        f"{_json_text(dialect, metadata, *forked)} = 'true'"
    )
    forked_boolean = (
        f"{_json_type(dialect, metadata, *forked)} IN ('true', 'false')"
        if sqlite
        else f"{_json_type(dialect, metadata, *forked)} = 'boolean'"
    )
    record_admits = (
        f"({record} IS NULL OR {record} = 'null' OR "
        f"({_json_type(dialect, metadata, *named)} = {string} AND "
        f"{_json_text(dialect, metadata, *named)} = {dialect.placeholder}))"
    )
    marker_admits = (
        f"({marker} IS NULL OR {marker} = 'null' OR "
        f"({marker} = 'object' AND {forked_boolean} AND "
        f"({_json_type(dialect, metadata, *owner)} = 'null' OR "
        f"({forked_true} AND {_json_type(dialect, metadata, *owner)} = {string} AND "
        f"{_json_text(dialect, metadata, *owner)} = {dialect.placeholder}))))"
    )
    return f"{record_admits} AND {marker_admits}"


def _fork_chain(dialect: SessionStoreSqlDialect, select: str) -> str:
    """Prefix ``select`` with ``fork_chain``: the row and its parents up to the one settling it.

    The walk starts at the enclosing query's ``cayu_sessions`` row and climbs
    ``parent_session_id`` while the current row carries neither an overlay
    record nor a lineage marker, as ``resolve_fork_lineage`` does. A deleted
    parent or a chain deeper than ``FORK_LINEAGE_MAX_DEPTH`` ends the walk on a
    row that settles nothing.
    """

    from cayu.sessions.fork_overlays import FORK_LINEAGE_MAX_DEPTH

    sqlite = dialect.contains_style == "sqlite_nocase_like"
    metadata, invocation = (
        ("metadata_json", "invocation_json") if sqlite else ("metadata", "invocation")
    )
    return (
        "WITH RECURSIVE fork_chain(id, parent_session_id, metadata, invocation, depth) AS ("
        "SELECT cayu_sessions.id, cayu_sessions.parent_session_id, "
        f"cayu_sessions.{metadata}, cayu_sessions.{invocation}, 0 "
        "UNION ALL SELECT fork_parent.id, fork_parent.parent_session_id, "
        f"fork_parent.{metadata}, fork_parent.{invocation}, fork_chain.depth + 1 "
        "FROM fork_chain JOIN cayu_sessions AS fork_parent "
        "ON fork_parent.id = fork_chain.parent_session_id "
        f"WHERE fork_chain.depth < {int(FORK_LINEAGE_MAX_DEPTH)} AND NOT "
        + _carries_fork_lineage(dialect, "fork_chain.metadata")
        + ") "
        + select
    )


def _row_metadata(dialect: SessionStoreSqlDialect) -> str:
    return (
        "cayu_sessions.metadata_json"
        if dialect.contains_style == "sqlite_nocase_like"
        else "cayu_sessions.metadata"
    )


def _unforked_terminal(
    dialect: SessionStoreSqlDialect, row: str, *, metadata: str | None = None
) -> str:
    """A known unforked marker or root that settles the walk."""

    from cayu.sessions.fork_overlays import FORK_LINEAGE_METADATA_KEY, FORK_OVERLAY_METADATA_KEY

    if metadata is None:
        metadata, invocation = f"{row}.metadata", f"{row}.invocation"
    else:
        invocation = (
            f"{row}.invocation_json"
            if dialect.contains_style == "sqlite_nocase_like"
            else f"{row}.invocation"
        )
    root = (
        f"({row}.parent_session_id IS NULL AND "
        + _json_text(dialect, invocation, "root_session_id")
        + f" = {row}.id AND NOT "
        + _carries_fork_lineage(dialect, metadata)
        + ")"
    )
    forked = (FORK_LINEAGE_METADATA_KEY, "forked")
    forked_false = (
        f"{_json_type(dialect, metadata, *forked)} = 'false'"
        if dialect.contains_style == "sqlite_nocase_like"
        else f"{_json_type(dialect, metadata, *forked)} = 'boolean' AND "
        f"{_json_text(dialect, metadata, *forked)} = 'false'"
    )
    marker = (
        f"({_json_type(dialect, metadata, FORK_LINEAGE_METADATA_KEY)} = 'object' AND "
        f"{forked_false} AND "
        f"{_json_type(dialect, metadata, FORK_LINEAGE_METADATA_KEY, 'controller')} = 'null' "
        f"AND NOT {_present(_json_type(dialect, metadata, FORK_OVERLAY_METADATA_KEY))})"
    )
    return f"({root} OR {marker})"


def fork_lineage_unforked_sql(dialect: SessionStoreSqlDialect) -> str:
    """True for a ``cayu_sessions`` row outside every fork lineage.

    Mirrors ``ForkLineage.forked`` being false: the walk settles on an unforked
    marker or root and passes no fork on the way. A row that settles itself skips the walk.
    """

    from cayu.sessions.invocation import SessionExecutionSource

    is_fork = (
        _json_text(dialect, "passed.invocation", "source")
        + f" = '{SessionExecutionSource.FORK.value}'"
    )
    walk = (
        "EXISTS ("
        + _fork_chain(
            dialect,
            "SELECT 1 FROM fork_chain WHERE "
            + _unforked_terminal(dialect, "fork_chain")
            + f" AND NOT EXISTS (SELECT 1 FROM fork_chain AS passed WHERE {is_fork})",
        )
        + ")"
    )
    return (
        "("
        + _unforked_terminal(dialect, "cayu_sessions", metadata=_row_metadata(dialect))
        + " OR (NOT "
        + _carries_fork_lineage(dialect, _row_metadata(dialect))
        + f" AND cayu_sessions.parent_session_id IS NOT NULL AND {walk}))"
    )


def _fork_lineage_entries_name(dialect: SessionStoreSqlDialect, metadata: str) -> str:
    """Compile ``fork_lineage_entries_name_owner`` for one row; binds the controller twice."""

    from cayu.sessions.fork_overlays import FORK_LINEAGE_METADATA_KEY, FORK_OVERLAY_METADATA_KEY

    string = "'text'" if dialect.contains_style == "sqlite_nocase_like" else "'string'"
    record = _present(_json_type(dialect, metadata, FORK_OVERLAY_METADATA_KEY))
    named = (FORK_OVERLAY_METADATA_KEY, "overlay", "controller")
    owner = (FORK_LINEAGE_METADATA_KEY, "controller")
    return (
        f"(({record} AND {_json_type(dialect, metadata, *named)} = {string} AND "
        f"{_json_text(dialect, metadata, *named)} = {dialect.placeholder}) OR "
        f"(NOT {record} AND {_json_type(dialect, metadata, *owner)} = {string} AND "
        f"{_json_text(dialect, metadata, *owner)} = {dialect.placeholder}))"
    )


def fork_lineage_owner_clause(controller: str, *, dialect: SessionStoreSqlDialect) -> SqlClause:
    """True for a ``cayu_sessions`` row whose fork lineage ``controller`` owns.

    Mirrors ``fork_lineage_owned_by``: a row with its own overlay record or
    lineage marker is judged by it, and any other row by the nearest parent
    that carries one, so a session written before lineage markers existed
    counts with the fork it descends from.
    """

    def settles_and_names(metadata: str) -> str:
        return (
            "("
            + _carries_fork_lineage(dialect, metadata)
            + " AND "
            + _fork_lineage_entries_name(dialect, metadata)
            + ")"
        )

    metadata = _row_metadata(dialect)
    walk = (
        "EXISTS ("
        + _fork_chain(
            dialect, "SELECT 1 FROM fork_chain WHERE " + settles_and_names("fork_chain.metadata")
        )
        + ")"
    )
    sql = (
        "("
        + settles_and_names(metadata)
        + " OR (NOT "
        + _carries_fork_lineage(dialect, metadata)
        + f" AND cayu_sessions.parent_session_id IS NOT NULL AND {walk}))"
    )
    return SqlClause(sql, (controller, controller, controller, controller))


def without_reservation_clause(
    budget_limit_id: str, *, dialect: SessionStoreSqlDialect
) -> SqlClause:
    """True for a cost event whose model attempt has no accepted reservation for the limit.

    Mirrors ``reserved_attempts``: the reservation is a ``budget.reserved``
    event in the same session naming the limit and the event's model attempt.
    A cost event without a model attempt never has one.
    """

    if dialect.contains_style == "sqlite_nocase_like":
        reserved = "json_extract(reserved.payload_json, '$.{}')"
        accepted = "json_type(reserved.payload_json, '$.accepted') = 'true'"
        cost_attempt = "json_extract(cayu_events.payload_json, '$.model_attempt_id')"
    else:
        reserved = "(reserved.event -> 'payload' ->> '{}')"
        accepted = "jsonb_typeof(reserved.event -> 'payload' -> 'accepted') = 'boolean' AND " + (
            f"{reserved.format('accepted')} = 'true'"
        )
        cost_attempt = "(cayu_events.event -> 'payload' ->> 'model_attempt_id')"
    sql = (
        "NOT EXISTS (SELECT 1 FROM cayu_events AS reserved "
        "WHERE reserved.session_id = cayu_events.session_id "
        f"AND reserved.event_type = {dialect.placeholder} "
        f"AND {reserved.format('budget_limit_id')} = {dialect.placeholder} "
        f"AND {reserved.format('model_attempt_id')} = {cost_attempt} "
        f"AND {accepted})"
    )
    return SqlClause(sql, (str(EventType.BUDGET_RESERVED), budget_limit_id))


def cost_read_filter_clauses(
    *,
    fork_lineage_owner: str | None,
    without_reservation_for: str | None,
    dialect: SessionStoreSqlDialect,
) -> tuple[tuple[SqlClause, ...], SqlClause | None]:
    """Validate a cost read's optional filters and compile them.

    Returns every filter clause for the event queries, and the lineage clause
    alone, which stores also apply to in-flight events' sessions.
    """

    from cayu.runtime._cost_accounting import require_cost_read_filters

    fork_lineage_owner, without_reservation_for = require_cost_read_filters(
        fork_lineage_owner, without_reservation_for
    )
    lineage = (
        None
        if fork_lineage_owner is None
        else fork_lineage_owner_clause(fork_lineage_owner, dialect=dialect)
    )
    clauses = tuple(
        clause
        for clause in (
            lineage,
            None
            if without_reservation_for is None
            else without_reservation_clause(without_reservation_for, dialect=dialect),
        )
        if clause is not None
    )
    return clauses, lineage


def fork_controller_read_clause(
    bounds: _ForkControllerReadBounds, *, dialect: SessionStoreSqlDialect
) -> SqlClause:
    """Compile one controller's read limit exactly as ``_ForkControllerReadBounds.admits``.

    The lineage part walks the row's parents to the session that settles it,
    so a session written before lineage markers existed is judged by the fork
    it descends from.
    """

    def settles_and_admits(metadata: str, root: str) -> str:
        return (
            "(("
            + _carries_fork_lineage(dialect, metadata)
            + " AND "
            + _fork_lineage_entries_admit(dialect, metadata)
            + f") OR {root})"
        )

    metadata = _row_metadata(dialect)
    walk = (
        "EXISTS ("
        + _fork_chain(
            dialect,
            "SELECT 1 FROM fork_chain WHERE "
            + settles_and_admits("fork_chain.metadata", _unforked_terminal(dialect, "fork_chain")),
        )
        + ")"
    )
    # A row with its own record or marker, or a true root, settles itself;
    # only the other rows need the walk, whose answer for them is the same.
    sql = (
        f"cayu_sessions.agent_name IN ({dialect.placeholders(len(bounds.agent_names))}) AND ("
        + settles_and_admits(
            metadata, _unforked_terminal(dialect, "cayu_sessions", metadata=metadata)
        )
        + " OR (NOT "
        + _carries_fork_lineage(dialect, metadata)
        + f" AND cayu_sessions.parent_session_id IS NOT NULL AND {walk}))"
    )
    controller = bounds.controller
    return SqlClause(sql, (*bounds.agent_names, controller, controller, controller, controller))


def session_access_clause(
    bounds: SessionReadBounds, *, dialect: SessionStoreSqlDialect
) -> SqlClause:
    """Compile admitted AND current authority; alternatives stay inside each scope."""
    from cayu.sessions._access_rules import _SessionAccessBounds
    from cayu.sessions._fork_controller_bounds import _ForkControllerReadBounds

    if type(bounds) is _ForkControllerReadBounds:
        return fork_controller_read_clause(bounds, dialect=dialect)
    if type(bounds) is not _SessionAccessBounds:
        raise TypeError("Unknown session read bound.")
    scopes = []
    params: list[object] = []
    for scope in (bounds.admitted, bounds.current):
        alternatives = []
        for rule in scope.read:
            if rule.allow_all:
                alternatives.append("1 = 1")
                continue
            query = SessionQuery(
                label_selectors=tuple(
                    LabelSelectorRequirement(
                        key=item.key, operator=item.operator, values=item.values
                    )
                    for item in rule.selectors
                )
            )
            plan = build_session_query_sql(query, dialect=dialect)
            alternatives.append(plan.filter_where_sql.removeprefix("WHERE "))
            params.extend(plan.filter_params)
        scopes.append(
            "(" + " OR ".join("(" + item + ")" for item in alternatives) + ")"
            if alternatives
            else "(1 = 0)"
        )
    return SqlClause(" AND ".join(scopes), tuple(params))
