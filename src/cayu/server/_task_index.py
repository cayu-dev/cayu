"""Public task matching and filter-bound continuation for native index reads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException

from cayu.tasks._value_filters import TaskValueFilters
from cayu.tasks.index import TASK_SEARCH_FIELDS, TaskIndexQuery, decode_task_index_cursor
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import TaskStatus
from cayu.vaults.redaction import REDACTED_SECRET


def parse_task_value_filters(
    input: list[str] | None, metadata: list[str] | None
) -> TaskValueFilters:
    """``TaskValueFilters.parse``, refusing a malformed filter, or too many, with 422."""

    try:
        return TaskValueFilters.parse(input, metadata)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


_MARKER = REDACTED_SECRET.casefold()
#: The most distinct field values one search read remembers decisions for.
_SEARCH_DECISIONS_MAX = 16_384


def _public_text(app) -> Callable[[str], Any]:
    """How the task list item shows a string field (status is shown as stored)."""

    redactor = getattr(app, "redact_json", None)
    return redactor if callable(redactor) else (lambda value: value)


def _shown_as_stored(value: str, public: Callable[[str], Any]) -> bool:
    """Whether a stored value matches ``value`` exactly when its public form does.

    Redaction only ever replaces secrets with ``REDACTED_SECRET``, so a public
    value without the marker is the stored value. When ``value`` has no marker
    and is shown as itself, matching stored values matches public ones.
    """

    return REDACTED_SECRET not in value and public(value) == value


def _needle_stays_apart(needle: str) -> bool:
    """Whether a match of ``needle`` in a public value can never touch a marker.

    Then a public value contains the needle only where its stored value does, and
    an ASCII needle is matched by SQL ``LIKE`` as ``casefold`` matches it, apart
    from the few characters ``CASEFOLD_TO_ASCII`` names.
    """

    if not needle.isascii() or needle in _MARKER or _MARKER in needle:
        return False
    return not any(
        needle.startswith(_MARKER[-size:]) or needle.endswith(_MARKER[:size])
        for size in range(1, min(len(needle), len(_MARKER)))
    )


def _search_match(needle: str, public: Callable[[str], Any]):
    """Decide a task from its stored search fields as the public projection shows them."""

    status = TASK_SEARCH_FIELDS.index("status")
    # A broad search matches the same type or agent name on many tasks; each
    # distinct value is redacted once per read.
    decided: dict[str, bool] = {}

    def shown_matches(value: str) -> bool:
        known = decided.get(value)
        if known is None:
            shown = public(value)
            known = isinstance(shown, str) and needle in shown.casefold()
            if len(decided) < _SEARCH_DECISIONS_MAX:
                decided[value] = known
        return known

    def matches(fields) -> bool:
        for position, value in enumerate(fields):
            if type(value) is not str or needle not in value.casefold():
                continue
            if position == status or shown_matches(value):
                return True
        return False

    return matches


def _native_read(app, query: TaskQuery, values: TaskValueFilters | None) -> bool:
    if values or not getattr(app.task_store, "supports_native_task_index", False):
        return False
    public = _public_text(app)
    exact = [
        value
        for value in (query.type, query.session_id, query.parent_task_id, query.assigned_agent_name)
        if value is not None
    ]
    exact.extend(query.types or ())
    if not all(_shown_as_stored(value, public) for value in exact):
        return False
    # The SQL prefilter matches the stored text against ``q`` itself, ignoring ASCII
    # case. A needle that only casefolds to ASCII (``ß`` is ``ss``) would miss tasks.
    return query.q is None or (query.q.isascii() and _needle_stays_apart(query.q.casefold()))


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
            getattr(query, field) is not None for field in (*filtered_fields, "types")
        ):
            return True
        projected = serialize(app, task)
        for field in filtered_fields:
            expected = getattr(query, field)
            if expected is not None and projected.get(field) != expected:
                return False
        if query.types is not None and projected.get("type") not in query.types:
            return False
        if needle is None:
            return True
        return any(
            isinstance(projected.get(field), str) and needle in projected[field].casefold()
            for field in TASK_SEARCH_FIELDS
        )

    try:
        if _native_read(app, query, values):
            # Every filter matches stored values exactly as it matches public ones,
            # so the store filters, pages and counts them in SQL.
            result = await app.task_store.query_task_index(
                TaskIndexQuery(
                    filters=query,
                    include_internal=include_internal,
                    prerequisite_task_id=prerequisite_task_id,
                    after=after,
                    statuses=statuses,
                ),
                search=None if needle is None else _search_match(needle, _public_text(app)),
            )
        else:
            result = await app.task_store.query_task_index(
                TaskIndexQuery(
                    # These selectors refer to the public projection, not raw store
                    # values. Preserve native scope/order while matching them only
                    # through the predicate above. Native SDK queries stay raw.
                    filters=query.model_copy(
                        update={field: None for field in (*filtered_fields, "types")}
                    ),
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
            "types": None,
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


async def read_tasks_by_ids(app, request, response, ids: list[str]):
    """``GET /tasks?ids=…``: each listed task's detail, in order, leaving out missing ones."""
    from cayu.server._etag import tagged_json
    from cayu.server.routes import _serialize_task_detail

    if set(request.query_params.keys()) != {"ids"}:
        raise HTTPException(422, "ids takes no other task list parameter.")
    if app.task_store is None:
        raise HTTPException(501, "Task store is not configured.")
    try:
        found = await app.task_store.load_tasks(ids)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from None
    details = [_serialize_task_detail(app, task) for task in found]
    return tagged_json(
        request, response, details, tagged={"tasks": details}, cache_control="private, no-store"
    )
