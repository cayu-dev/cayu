"""SQL for native task index reads, shared by the SQLite and PostgreSQL stores.

A native read filters, orders and pages in SQL and counts statuses from
``cayu_task_counts`` when its filters are ones the counts are kept by (internal
flag, type, assigned agent, with no resource access bounds), or with one indexed
``GROUP BY`` otherwise. A text search scans only the narrow search columns of the
tasks whose stored text matches, lets the owner's ``TaskSearchMatch`` decide each,
and decodes only the page's tasks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cayu.storage._task_index_schema import POSTGRES_SEARCH_TEXT, SQLITE_TASK_SEARCH_INDEX
from cayu.tasks.index import (
    TASK_SEARCH_FIELDS,
    TaskIndexPage,
    TaskIndexQuery,
    TaskSearchMatch,
)
from cayu.tasks.queries import TaskOrder, TaskQuery
from cayu.tasks.records import TaskStatus


@dataclass(frozen=True)
class TaskIndexDialect:
    postgres: bool

    @property
    def mark(self) -> str:
        return "%s" if self.postgres else "?"

    @property
    def identity(self) -> str:
        return 'id COLLATE "C"' if self.postgres else "id"

    @property
    def external(self) -> str:
        return "is_internal = false" if self.postgres else "is_internal = 0"

    def marks(self, count: int) -> str:
        return ", ".join([self.mark] * count)


def eligibility_filters(query: TaskIndexQuery) -> TaskQuery:
    """The query's filters a store's list clauses apply before counting."""

    return query.filters.model_copy(update={"status": None, "q": None, "offset": 0})


def _like_contains(needle: str) -> str:
    escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


@dataclass
class TaskIndexPlan:
    """The statements one native read runs, and how to assemble their rows."""

    query: TaskIndexQuery
    dialect: TaskIndexDialect
    where: list[str]
    params: list[Any]
    uses_counts_table: bool
    counts_where: list[str] = field(default_factory=list)
    counts_params: list[Any] = field(default_factory=list)

    @property
    def order_column(self) -> str:
        order = self.query.filters.order_by
        return (
            "created_at"
            if order in {TaskOrder.CREATED_AT_ASC, TaskOrder.CREATED_AT_DESC}
            else "updated_at"
        )

    @property
    def ascending(self) -> bool:
        return self.query.filters.order_by in {TaskOrder.CREATED_AT_ASC, TaskOrder.UPDATED_AT_ASC}

    @property
    def order_sql(self) -> str:
        direction = "ASC" if self.ascending else "DESC"
        return f"{self.order_column} {direction}, {self.dialect.identity} ASC"

    def _where(self, extra: Sequence[str] = ()) -> str:
        clauses = [*self.where, *extra]
        return "WHERE " + " AND ".join(clauses) if clauses else ""

    def counts(self) -> tuple[str, list[Any]]:
        """Status counts over every eligible task."""

        if not self.uses_counts_table:
            return (
                f"SELECT status, COUNT(*) FROM cayu_tasks {self._where()} GROUP BY status",
                list(self.params),
            )
        where = "WHERE " + " AND ".join(self.counts_where) if self.counts_where else ""
        if not self.dialect.postgres:
            return (
                f"SELECT status, SUM(task_count) FROM cayu_task_counts {where} GROUP BY status",
                list(self.counts_params),
            )
        return (
            "SELECT status, SUM(n) FROM ("
            f"SELECT status, task_count AS n FROM cayu_task_counts {where} "
            f"UNION ALL SELECT status, delta AS n FROM cayu_task_count_changes {where}"
            ") AS counted GROUP BY status",
            [*self.counts_params, *self.counts_params],
        )

    def _page_filters(self) -> tuple[list[str], list[Any]]:
        mark = self.dialect.mark
        clauses: list[str] = []
        params: list[Any] = []
        filters = self.query.filters
        if filters.status is not None:
            clauses.append(f"status = {mark}")
            params.append(str(filters.status))
        if self.query.statuses:
            statuses = sorted({str(status) for status in self.query.statuses})
            clauses.append(f"status IN ({self.dialect.marks(len(statuses))})")
            params.extend(statuses)
        return clauses, params

    def page(
        self, columns: str, format_timestamp: Callable[[datetime], Any]
    ) -> tuple[str, list[Any]]:
        """One page of eligible tasks in the requested statuses, plus one to show more."""

        clauses, params = self._page_filters()
        mark = self.dialect.mark
        if self.query.after is not None:
            timestamp, identity = self.query.after
            column = self.order_column
            bound, strict = (">=", ">") if self.ascending else ("<=", "<")
            clauses.append(
                f"{column} {bound} {mark} AND ({column} {strict} {mark} "
                f"OR {self.dialect.identity} > {mark})"
            )
            value = format_timestamp(timestamp)
            params.extend([value, value, identity])
        offset = 0 if self.query.after is not None else self.query.filters.offset
        return (
            f"SELECT {columns} FROM cayu_tasks {self._where(clauses)} "
            f"ORDER BY {self.order_sql} LIMIT {mark} OFFSET {mark}",
            [*self.params, *params, self.query.filters.limit + 1, offset],
        )

    def search_scan(self) -> tuple[str, list[Any]]:
        """The narrow search columns of every eligible task, in query order."""

        columns = ", ".join(("status", "created_at", "updated_at", *TASK_SEARCH_FIELDS))
        # SQLite's planner would rather walk a filter index and read each task's
        # row; the search index holds every column a scan reads.
        source = (
            "cayu_tasks"
            if self.dialect.postgres
            else f"cayu_tasks INDEXED BY {SQLITE_TASK_SEARCH_INDEX}"
        )
        return (
            f"SELECT {columns} FROM {source} {self._where()} ORDER BY {self.order_sql}",
            list(self.params),
        )

    def by_identity(self, columns: str, count: int) -> str:
        if self.dialect.postgres:
            return f"SELECT {columns} FROM cayu_tasks WHERE id = ANY(%s)"
        return f"SELECT {columns} FROM cayu_tasks WHERE id IN ({self.dialect.marks(count)})"


#: Joins the search fields in PostgreSQL's one-pass text prefilter. A needle
#: holding it is matched field by field instead, so no match spans two fields.
_FIELD_SEPARATOR = "\x1f"


def _text_prefilter(
    needle: str, dialect: TaskIndexDialect, params: list[Any], *, search: bool
) -> str:
    mark = dialect.mark
    pattern = _like_contains(needle)
    if dialect.postgres and _FIELD_SEPARATOR not in needle:
        # One pass over the joined fields costs PostgreSQL much less than one per field.
        params.append(pattern)
        prefilter = f"{POSTGRES_SEARCH_TEXT} ILIKE {mark} ESCAPE '\\'"
        return f"({prefilter} OR search_folds)" if search else prefilter
    like = "ILIKE" if dialect.postgres else "LIKE"
    matches = [f"{column} {like} {mark} ESCAPE '\\'" for column in TASK_SEARCH_FIELDS]
    params.extend([pattern] * len(TASK_SEARCH_FIELDS))
    if search:
        # Computed when the task is written.
        matches.append("search_folds" if dialect.postgres else "search_folds = 1")
    return "(" + " OR ".join(matches) + ")"


def plan_native_read(
    query: TaskIndexQuery,
    list_clauses: tuple[list[str], list[Any]],
    *,
    dialect: TaskIndexDialect,
    bounded: bool,
    search: bool,
) -> TaskIndexPlan:
    """Plan a native read.

    ``list_clauses`` are the store's list clauses for ``eligibility_filters(query)``
    and the caller's access bounds. With ``search``, ``filters.q`` keeps every task
    the owner's match could accept: its stored text contains the needle ignoring
    ASCII case, or holds a character that casefolds to ASCII.
    """

    mark = dialect.mark
    where, params = list(list_clauses[0]), list(list_clauses[1])
    filters = query.filters
    if not query.include_internal:
        where.append(dialect.external)
    if query.prerequisite_task_id is not None:
        where.append(
            "id IN (SELECT task_id FROM cayu_task_prerequisites "
            f"WHERE prerequisite_task_id = {mark})"
        )
        params.append(query.prerequisite_task_id)
    if filters.q is not None:
        where.append(_text_prefilter(filters.q, dialect, params, search=search))
    countable = not bounded and all(
        value is None
        for value in (
            filters.q,
            filters.session_id,
            filters.parent_task_id,
            filters.has_work_contract,
            query.prerequisite_task_id,
        )
    )
    plan = TaskIndexPlan(query, dialect, where, params, uses_counts_table=countable)
    if countable:
        if not query.include_internal:
            plan.counts_where.append(dialect.external)
        types = filters.types or ((filters.type,) if filters.type is not None else None)
        if types is not None:
            plan.counts_where.append(f"type IN ({dialect.marks(len(types))})")
            plan.counts_params.extend(types)
        if filters.assigned_agent_name is not None:
            plan.counts_where.append(f"assigned_agent_name = {mark}")
            plan.counts_params.append(filters.assigned_agent_name)
    return plan


def count_page(query: TaskIndexQuery, rows: Iterable[Sequence[Any]]) -> TaskIndexPage:
    """Fill a page's counts from ``(status, count)`` rows."""

    page = TaskIndexPage(query)
    for status, count in rows:
        page.counts[TaskStatus(status)] += int(count)
    filters = query.filters
    page.total = sum(
        count
        for status, count in page.counts.items()
        if (filters.status is None or status == filters.status)
        and (not query.statuses or status in query.statuses)
    )
    return page


_STATUSES = {status.value: status for status in TaskStatus}


def search_page(
    query: TaskIndexQuery,
    rows: Iterable[Sequence[Any]],
    matches: TaskSearchMatch | None,
    *,
    parse_timestamp: Callable[[Any], datetime],
    format_timestamp: Callable[[datetime], Any],
) -> TaskIndexPage:
    """Count and page the scanned tasks that ``matches`` accepts.

    Keys stay in the store's timestamp representation, which orders as time does,
    and only the page's keys are parsed.
    """

    after = None if query.after is None else (format_timestamp(query.after[0]), query.after[1])
    page = TaskIndexPage(query, after=after)
    by_created = query.filters.order_by in {TaskOrder.CREATED_AT_ASC, TaskOrder.CREATED_AT_DESC}
    for scanned in rows:
        row = tuple(scanned)
        if matches is not None and not matches(row[3:]):
            continue
        page.visit(_STATUSES[row[0]], (row[1] if by_created else row[2], row[3]))
    page.keys = [(parse_timestamp(timestamp), identity) for timestamp, identity in page.keys]
    return page
