from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from cayu.core.events import EventType
from cayu.runtime.sessions import (
    EventOrder,
    EventQuery,
    LabelSelectorOperator,
    SessionDebugState,
    SessionOrder,
    SessionQuery,
    SessionStatus,
    copy_event_query,
    copy_session_query,
    decode_session_cursor,
    session_order_is_descending,
    session_sort_column,
)


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


def session_order_sql(order_by: SessionOrder) -> str:
    direction = "DESC" if session_order_is_descending(order_by) else "ASC"
    return f"{session_sort_column(order_by)} {direction}"


def build_event_query_sql(
    query: EventQuery | None,
    *,
    dialect: SessionStoreSqlDialect,
    extra_after_sequence_clauses: Sequence[SqlClause] = (),
) -> EventQuerySqlPlan:
    query = copy_event_query(query)
    clauses: list[str] = []
    params: list[object] = []

    if query.after_sequence is not None:
        clauses.append(f"cayu_events.sequence > {dialect.placeholder}")
        params.append(query.after_sequence)
    _append_clauses(clauses, params, extra_after_sequence_clauses)
    if query.event_id is not None:
        clauses.append(f"cayu_events.event_id = {dialect.placeholder}")
        params.append(query.event_id)
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
    if query.event_types:
        clauses.append(
            f"cayu_events.event_type IN ({dialect.placeholders(len(query.event_types))})"
        )
        params.extend(str(event_type) for event_type in query.event_types)
    if query.agent_name is not None:
        clauses.append(f"cayu_events.agent_name = {dialect.placeholder}")
        params.append(query.agent_name)
    if query.environment_name is not None:
        clauses.append(f"cayu_events.environment_name = {dialect.placeholder}")
        params.append(query.environment_name)
    if query.workflow_name is not None:
        clauses.append(f"cayu_events.workflow_name = {dialect.placeholder}")
        params.append(query.workflow_name)
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
) -> SessionQuerySqlPlan:
    query = copy_session_query(query)
    clauses: list[str] = []
    params: list[object] = []

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
