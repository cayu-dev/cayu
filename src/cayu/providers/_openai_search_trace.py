"""Bounded structural evidence at the native Responses adapter boundary."""

import json
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_TRACE_LIMIT = 16
_COUNTER_LIMIT = 1_000_000
_EVENT_TYPES = frozenset(
    {
        "response.created",
        "response.output_item.added",
        "response.output_item.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
        "response.web_search_call.completed",
        "response.completed",
        "response.incomplete",
        "response.failed",
        "error",
        "other",
    }
)
_RELATIONS = frozenset({"missing", "invalid", "unregistered", "matches", "differs"})
_ITEM_TYPES = frozenset(
    {
        "message",
        "function_call",
        "reasoning",
        "web_search_call",
        "tool_search_call",
        "missing",
        "other",
    }
)
_STATES = frozenset({"pending", "completed", "absent"})


@dataclass(frozen=True)
class SearchStreamDiagnostic:
    # Entries: ordinal, type, output index (-1 = omitted), registration state,
    # item identity relation, response identity relation. Never raw identities.
    entries: tuple[tuple[int, str, int, str, str, str], ...]
    truncated: bool
    item_types: tuple[tuple[int, str, str], ...] = ()


def search_stream_diagnostic_fields(diagnostic: object) -> dict[str, str | int]:
    """Revalidate exception attributes before public/durable projection."""
    if type(diagnostic) is not SearchStreamDiagnostic:
        return {}
    if type(diagnostic.entries) is not tuple or not 1 <= len(diagnostic.entries) <= _TRACE_LIMIT:
        return {}
    if type(diagnostic.truncated) is not bool:
        return {}
    for row in diagnostic.entries:
        if type(row) is not tuple or len(row) != 6:
            return {}
        ordinal, event_type, index, state, item, response = row
        if type(ordinal) is not int or not 1 <= ordinal <= _COUNTER_LIMIT:
            return {}
        if type(index) is not int or not -1 <= index <= _COUNTER_LIMIT:
            return {}
        for value, vocabulary in (
            (event_type, _EVENT_TYPES),
            (state, _STATES),
            (item, _RELATIONS),
            (response, _RELATIONS),
        ):
            if type(value) is not str or len(value) > 64 or value not in vocabulary:
                return {}
    if type(diagnostic.item_types) is not tuple or len(diagnostic.item_types) > _TRACE_LIMIT:
        return {}
    if diagnostic.item_types and (len(diagnostic.item_types) != len(diagnostic.entries)):
        return {}
    for position, row in enumerate(diagnostic.item_types):
        if type(row) is not tuple or len(row) != 3:
            return {}
        ordinal, incoming, registered = row
        if (
            type(ordinal) is not int
            or not 1 <= ordinal <= _COUNTER_LIMIT
            or ordinal != diagnostic.entries[position][0]
        ):
            return {}
        if any(
            type(value) is not str or len(value) > 64 or value not in _ITEM_TYPES
            for value in (incoming, registered)
        ):
            return {}
    fields = {
        "provider_protocol_stream_boundary": "native_adapter",
        "provider_protocol_stream_trace": json.dumps(diagnostic.entries, separators=(",", ":")),
        "provider_protocol_stream_trace_truncated": int(diagnostic.truncated),
    }
    if diagnostic.item_types:
        fields["provider_protocol_stream_item_types"] = json.dumps(
            diagnostic.item_types, separators=(",", ":")
        )
    return fields


def _item_type(value: object) -> str:
    if value is None:
        return "missing"
    return value if type(value) is str and len(value) <= 64 and value in _ITEM_TYPES else "other"


def _relation(value: object, expected: object, *, strip: bool = False) -> str:
    if value is None:
        return "missing"
    if type(value) is not str or (not value or value.isspace()):
        return "invalid"
    if type(expected) is not str:
        return "unregistered"
    compared = value.strip() if strip else value
    return "matches" if compared == expected else "differs"


class SearchStreamTrace:
    """Constant storage, no source/query/body/credential/identity retention."""

    def __init__(self) -> None:
        self._entries: deque[tuple[int, str, int, str, str, str]] = deque(maxlen=_TRACE_LIMIT)
        self._ordinal = 0
        self._truncated = False

    def record(
        self,
        event: Mapping[str, Any],
        pending: Mapping[int, tuple[str, str]],
        completed: Mapping[int, dict[str, Any]],
        response_id: str | None,
    ) -> None:
        self._truncated |= len(self._entries) == _TRACE_LIMIT
        self._ordinal = min(self._ordinal + 1, _COUNTER_LIMIT)
        kind = event.get("type")
        kind = kind if type(kind) is str and len(kind) <= 64 and kind in _EVENT_TYPES else "other"
        index = event.get("output_index")
        valid_index = type(index) is int and index >= 0
        registered = pending.get(index) if valid_index else None
        finished = completed.get(index) if valid_index else None
        state = "absent"
        expected = None
        if registered is not None:
            state, expected = "pending", registered[0]
        elif finished is not None and finished.get("type") == "web_search_call":
            state, expected = "completed", finished.get("id")
        self._record(event, state, expected, response_id, kind, index, valid_index)

    def _record(
        self,
        event: Mapping[str, Any],
        state: str,
        expected: object,
        response_id: str | None,
        kind: str,
        index: Any,
        valid_index: bool,
    ) -> None:
        item = event.get("item")
        item_id = item.get("id") if isinstance(item, Mapping) else event.get("item_id")
        response = event.get("response")
        incoming_response_id = (
            response.get("id") if isinstance(response, Mapping) else event.get("response_id")
        )
        self._entries.append(
            (
                self._ordinal,
                kind,
                index if valid_index and index <= _COUNTER_LIMIT else -1,
                state,
                _relation(item_id, expected, strip=True),
                _relation(incoming_response_id, response_id),
            )
        )

    def snapshot(self) -> SearchStreamDiagnostic:
        return SearchStreamDiagnostic(tuple(self._entries), self._truncated)


class FunctionStreamTrace(SearchStreamTrace):
    """Same bounded structural format, using function registration state."""

    def __init__(self) -> None:
        super().__init__()
        self.has_function = False
        self._item_types: deque[tuple[int, str, str]] = deque(maxlen=_TRACE_LIMIT)

    def record(
        self,
        event: Mapping[str, Any],
        pending: Mapping[int, Any],
        completed: Mapping[int, dict[str, Any]],
        response_id: str | None,
    ) -> None:
        self._truncated |= len(self._entries) == _TRACE_LIMIT
        self._ordinal = min(self._ordinal + 1, _COUNTER_LIMIT)
        kind = event.get("type")
        kind = kind if type(kind) is str and len(kind) <= 64 and kind in _EVENT_TYPES else "other"
        index = event.get("output_index")
        valid_index = type(index) is int and index >= 0
        registered = pending.get(index) if valid_index else None
        finished = completed.get(index) if valid_index else None
        state, expected = "absent", None
        if registered is not None:
            state, expected = "pending", registered.item_id
        elif finished is not None:
            state, expected = "completed", finished.get("id")
        item = event.get("item")
        self._item_types.append(
            (
                self._ordinal,
                _item_type(item.get("type") if isinstance(item, Mapping) else None),
                "function_call"
                if registered is not None
                else _item_type(finished.get("type") if finished is not None else None),
            )
        )
        self.has_function |= kind in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        } or (isinstance(item, Mapping) and item.get("type") == "function_call")
        self._record(event, state, expected, response_id, kind, index, valid_index)

    def snapshot(self) -> SearchStreamDiagnostic:
        return SearchStreamDiagnostic(
            tuple(self._entries), self._truncated, tuple(self._item_types)
        )
