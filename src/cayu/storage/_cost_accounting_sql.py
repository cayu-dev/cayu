"""Bounded group streams and indexed lookups for native exact cost accounting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.runtime.aggregates import AGGREGATE_IDENTITY_TRIM_CHARACTERS
from cayu.storage import _session_store_sql as session_sql

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cayu.runtime._cost_accounting import CostGroupKey
    from cayu.runtime._cost_accounting_refresh import CostAccountingRead

COST_EVENT_PREDICATE = "cayu_events.event_type IN ('model.completed', 'model.hosted_tool_call')"
COST_ATTEMPT_INDEX_PREFIX = 128


def _attempt_expression(postgres: bool) -> tuple[str, tuple[object, ...]]:
    if postgres:
        attempt = "cayu_events.event -> 'payload' ->> 'model_attempt_id'"
        valid = (
            "jsonb_typeof(cayu_events.event -> 'payload' -> 'model_attempt_id') = 'string' "
            f"AND {attempt} <> '' AND btrim({attempt}, %s) = {attempt}"
        )
        params: tuple[object, ...] = (AGGREGATE_IDENTITY_TRIM_CHARACTERS,)
    else:
        attempt = "json_extract(cayu_events.payload_json, '$.model_attempt_id')"
        valid = f"cayu_is_clean_nonblank_text({attempt}) = 1"
        params = ()
    return f"CASE WHEN {valid} THEN {attempt} ELSE NULL END", params


def cost_group_statement(
    *,
    columns: str,
    where_sql: str,
    postgres: bool,
) -> tuple[str, tuple[object, ...]]:
    attempt, params = _attempt_expression(postgres)
    collation = '"C"' if postgres else "BINARY"
    return (
        f"SELECT * FROM (SELECT {columns}, {attempt} AS cost_attempt FROM cayu_events "
        "JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id "
        f"{where_sql} AND {COST_EVENT_PREDICATE}) AS cost_rows ORDER BY session_id COLLATE {collation}, "
        f"(cost_attempt IS NOT NULL), COALESCE(cost_attempt, event_id) COLLATE {collation}, "
        "event_type DESC, sequence",
        params,
    )


def cost_group_lookup_statement(
    *,
    columns: str,
    plan: session_sql.EventQuerySqlPlan,
    key: CostGroupKey,
    postgres: bool,
) -> tuple[str, tuple[object, ...]]:
    placeholder = "%s" if postgres else "?"
    params = (*plan.params, key[0])
    collation = '"C"' if postgres else "BINARY"
    predicate = f"cayu_events.session_id COLLATE {collation} = {placeholder}"
    if key[1]:
        if postgres:
            attempt = "cayu_events.event -> 'payload' ->> 'model_attempt_id'"
            prefix = f'left({attempt}, {COST_ATTEMPT_INDEX_PREFIX}) COLLATE "C"'
            attempt = f'({attempt}) COLLATE "C"'
            type_predicate = (
                "jsonb_typeof(cayu_events.event -> 'payload' -> 'model_attempt_id') = 'string'"
            )
        else:
            attempt = "json_extract(cayu_events.payload_json, '$.model_attempt_id')"
            prefix = f"substr({attempt}, 1, {COST_ATTEMPT_INDEX_PREFIX})"
            type_predicate = "json_type(cayu_events.payload_json, '$.model_attempt_id') = 'text'"
        predicate += (
            f" AND {prefix} = {placeholder} AND {attempt} = {placeholder} AND {type_predicate}"
        )
        params = (*params, key[2][:COST_ATTEMPT_INDEX_PREFIX], key[2])
    else:
        predicate += f" AND cayu_events.event_id = {placeholder}"
        params = (*params, key[2])
    source = (
        "cayu_events INDEXED BY idx_cayu_events_cost_attempt"
        if key[1] and not postgres
        else "cayu_events"
    )
    return (
        f"SELECT {columns} FROM {source} "
        "JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id "
        f"{plan.where_sql} AND {COST_EVENT_PREDICATE} AND {predicate} "
        "ORDER BY cayu_events.event_type DESC, cayu_events.sequence",
        params,
    )


def changed_cost_groups_statement(
    read: CostAccountingRead,
    *,
    dialect: session_sql.SessionStoreSqlDialect,
    extra_clauses: Sequence[session_sql.SqlClause] = (),
) -> tuple[str, tuple[object, ...]]:
    from cayu.runtime.sessions import copy_event_query

    assert read.previous is not None
    previous = read.previous.through_sequence
    postgres = dialect.placeholder == "%s"
    if read.query.since == read.old_query.since and read.query.until == read.old_query.until:
        query = copy_event_query(
            read.query, update={"after_sequence": max(read.query.after_sequence or 0, previous)}
        )
        plan = session_sql.build_accounting_event_query_sql(
            query, dialect=dialect, extra_after_sequence_clauses=extra_clauses
        )
        where, params = plan.where_sql, plan.params
    else:
        before = None if previous == MAX_DURABLE_JSON_INTEGER else previous + 1
        if read.old_query.before_sequence is not None:
            before = (
                min(before, read.old_query.before_sequence)
                if before is not None
                else read.old_query.before_sequence
            )
        old = session_sql.build_accounting_event_query_sql(
            copy_event_query(read.old_query, update={"before_sequence": before}),
            dialect=dialect,
            extra_after_sequence_clauses=extra_clauses,
        )
        new = session_sql.build_accounting_event_query_sql(
            read.query, dialect=dialect, extra_after_sequence_clauses=extra_clauses
        )
        old_condition, new_condition = (
            old.where_sql.removeprefix("WHERE "),
            new.where_sql.removeprefix("WHERE "),
        )
        where = (
            f"WHERE (({new_condition}) AND cayu_events.sequence > {dialect.placeholder}) "
            f"OR (({old_condition}) AND NOT ({new_condition})) "
            f"OR (({new_condition}) AND NOT ({old_condition}))"
        )
        params = (*new.params, previous, *old.params, *new.params, *new.params, *old.params)
    source = "cayu_events"
    if (
        not postgres
        and read.query.since == read.old_query.since
        and read.query.until == read.old_query.until
    ):
        source += " INDEXED BY idx_cayu_events_cost_sequence"
    where = f"WHERE ({where.removeprefix('WHERE ')}) AND {COST_EVENT_PREDICATE}"
    attempt, attempt_params = _attempt_expression(postgres)
    # DISTINCT executes in the database; the client receives only a bounded page
    # of identities even when every historical attempt expires in one refresh.
    collation = '"C"' if postgres else "BINARY"
    return (
        f"SELECT DISTINCT session_id COLLATE {collation}, cost_attempt COLLATE {collation}, "
        f"(CASE WHEN cost_attempt IS NULL THEN event_id ELSE NULL END) COLLATE {collation} AS standalone_id FROM ("
        f"SELECT cayu_events.session_id, cayu_events.event_id, {attempt} AS cost_attempt "
        f"FROM {source} JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id "
        f"{where}) AS changed_cost_events",
        (*attempt_params, *params),
    )


def cost_boundary_statement(query, *, dialect, extra_clauses=()):
    from cayu.runtime.sessions import EventQuery

    session_id = query.session_id or (query.session_ids[0] if len(query.session_ids) == 1 else None)
    plan = session_sql.build_accounting_event_query_sql(
        EventQuery(
            session_id=session_id,
            after_sequence=query.after_sequence,
            before_sequence=query.before_sequence,
        ),
        dialect=dialect,
        extra_after_sequence_clauses=extra_clauses,
    )
    return f"SELECT MAX(cayu_events.sequence) FROM cayu_events {plan.where_sql}", plan.params
