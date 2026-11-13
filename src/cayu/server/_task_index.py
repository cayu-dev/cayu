"""Public task matching and filter-bound continuation for native index reads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from cayu.tasks.index import TaskIndexQuery, decode_task_index_cursor
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import TaskStatus

_SEARCH_FIELDS = (
    "id",
    "type",
    "title",
    "description",
    "status",
    "session_id",
    "parent_task_id",
    "assigned_agent_name",
    "worker_id",
    "status_reason",
)


# At most this many `input` and `metadata` filters on one task list read.
_VALUE_FILTERS_MAX = 8


@dataclass(frozen=True, slots=True)
class TaskValueFilters:
    """Repeated `key=value` filters on a task's top-level input and metadata values."""

    input: tuple[tuple[str, str], ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    @classmethod
    def parse(cls, input: list[str] | None, metadata: list[str] | None) -> TaskValueFilters:
        return cls(_parse_pairs(input, "input"), _parse_pairs(metadata, "metadata"))

    def __bool__(self) -> bool:
        return bool(self.input or self.metadata)

    def binding(self) -> dict[str, Any]:
        return {"input": self.input, "metadata": self.metadata}

    def matches(self, task: Any, redact: Callable[[Any, str], Any]) -> bool:
        """Whether each named value matches it as the task detail shows it.

        The whole input or metadata is redacted first, as the detail does, since
        redaction can rename keys too.
        """
        for source, pairs, field_name in (
            (task.input, self.input, "task.input"),
            (task.metadata, self.metadata, "task.metadata"),
        ):
            if not pairs:
                continue
            projected = redact(source, field_name)
            if not isinstance(projected, dict):
                return False
            for key, expected in pairs:
                if key not in projected or not _value_matches(projected[key], expected):
                    return False
        return True


def _parse_pairs(values: list[str] | None, name: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for raw in values or ():
        key, separator, value = raw.partition("=")
        if not separator or not key or key != key.strip():
            raise HTTPException(422, f"Each {name} filter is key=value, such as {name}=key=value.")
        pairs.append((key, value))
    if len(pairs) > _VALUE_FILTERS_MAX:
        raise HTTPException(422, f"At most {_VALUE_FILTERS_MAX} {name} filters are allowed.")
    return tuple(pairs)


def _value_matches(value: Any, expected: str) -> bool:
    """Strings compare as text, booleans as true/false, numbers by value (`2` is `2.0`)."""
    if type(value) is str:
        return value == expected
    if type(value) is bool:
        return expected == ("true" if value else "false")
    if type(value) in (int, float):
        try:
            if type(value) is int and expected.lstrip("-").isdigit():
                return value == int(expected)
            return value == float(expected)
        except (ValueError, OverflowError):
            return False
    return False


async def read_task_index(
    app,
    query: TaskQuery,
    *,
    include_internal: bool,
    cursor: str | None,
    prerequisite_task_id: str | None,
    serialize: Callable[[Any, Any], dict[str, Any]],
    seal,
    actor=None,
    statuses: tuple[TaskStatus, ...] = (),
    allow_offset: bool = False,
    values: TaskValueFilters | None = None,
    redact: Callable[[Any, str], Any] | None = None,
) -> dict[str, Any]:
    """Read one page of the public task index.

    ``values`` filter on input and metadata values after ``redact``, and a cursor stays
    bound to them.
    """
    if app.task_store is None:
        raise HTTPException(501, "Task index is unavailable.")
    identity = query.model_dump(mode="json", exclude={"limit", "offset"})
    binding_parts: list[Any] = [
        identity,
        include_internal,
        prerequisite_task_id,
        sorted(set(statuses)),
    ]
    if values:
        binding_parts.append(values.binding())
    binding = hashlib.sha256(
        json.dumps(
            binding_parts,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    identity_binding = [binding, None if actor is None else [actor.subject, actor.tenant]]
    after = (
        None
        if cursor is None
        else decode_task_index_cursor(seal.open("tasks", cursor, identity_binding))
    )
    if query.offset and not allow_offset:
        raise HTTPException(422, "Task index pages use cursors, not offset.")
    needle = query.q.casefold() if query.q else None

    filtered_fields = ("type", "session_id", "parent_task_id", "assigned_agent_name")

    def matches(task):
        if values and not values.matches(task, redact or (lambda value, _field: value)):
            return False
        if needle is None and not any(
            getattr(query, field) is not None for field in filtered_fields
        ):
            return True
        projected = serialize(app, task)
        for field in filtered_fields:
            expected = getattr(query, field)
            if expected is not None and projected.get(field) != expected:
                return False
        if needle is None:
            return True
        return any(
            isinstance(projected.get(field), str) and needle in projected[field].casefold()
            for field in _SEARCH_FIELDS
        )

    try:
        result = await app.task_store.query_task_index(
            TaskIndexQuery(
                # These selectors refer to the public projection, not raw store
                # values. Preserve native scope/order while matching them only
                # through the predicate above. Native SDK queries stay raw.
                filters=query.model_copy(update={field: None for field in filtered_fields}),
                include_internal=include_internal,
                prerequisite_task_id=prerequisite_task_id,
                after=after,
                statuses=statuses,
            ),
            matches=matches,
        )
    except NotImplementedError:
        if allow_offset:
            return {
                "items": await _read_bare_task_items(
                    app,
                    query,
                    include_internal=include_internal,
                    statuses=statuses,
                    matches=matches,
                    serialize=serialize,
                )
            }
        raise HTTPException(501, "Task index is unavailable.") from None
    except Exception:
        raise HTTPException(503, "Task index could not complete its read snapshot.") from None
    next_cursor = (
        None
        if result.next_position is None
        else seal.seal("tasks", result.next_position, identity_binding)
    )
    if next_cursor is not None and len(next_cursor) > 6144:
        raise HTTPException(503, "Task index continuation exceeds its public byte bound.")
    return {
        "items": [serialize(app, task) for task in result.items],
        "total_count": result.total_count,
        "counts_by_status": result.counts_by_status.model_dump(),
        "next_cursor": next_cursor,
        "as_of": result.as_of,
    }


async def _read_bare_task_items(
    app,
    query: TaskQuery,
    *,
    include_internal: bool,
    statuses: tuple[TaskStatus, ...],
    matches,
    serialize,
) -> list[dict[str, Any]]:
    """Filter bounded native list pages without claiming a snapshot or counts."""
    scan = query.model_copy(
        update={
            "q": None,
            "type": None,
            "session_id": None,
            "parent_task_id": None,
            "assigned_agent_name": None,
            "limit": 256,
            "offset": 0,
        }
    )
    items = []
    skipped = 0
    seen: set[str] = set()
    try:
        while True:
            page = await app.task_store.list_tasks(scan)
            identities = tuple(task.id for task in page)
            if (
                len(page) > scan.limit
                or len(set(identities)) != len(identities)
                or seen.intersection(identities)
            ):
                raise ValueError("Task list source did not advance its page.")
            seen.update(identities)
            for task in page:
                if (not include_internal and task.type.startswith("cayu.")) or (
                    statuses and task.status not in statuses
                ):
                    continue
                if not matches(task):
                    continue
                if skipped < query.offset:
                    skipped += 1
                    continue
                items.append(serialize(app, task))
                if len(items) == query.limit:
                    return items
            if len(page) < scan.limit:
                return items
            if len(seen) >= 4096:
                raise ValueError("Task list exceeded its bounded fallback scan.")
            scan = scan.model_copy(update={"offset": scan.offset + len(page)})
    except Exception:
        raise HTTPException(503, "Task list could not complete its public read.") from None
