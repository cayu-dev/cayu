"""Bounded structural evidence at decoded HTTP and native Responses boundaries."""

import hashlib
import json
import os
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
    identities: tuple[tuple[int, str, int], ...] = ()
    structure: "ResponseStructureDiagnostic | None" = None
    transport: "ResponseStructureDiagnostic | None" = None


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
    if type(diagnostic.identities) is not tuple or len(diagnostic.identities) > _TRACE_LIMIT:
        return {}
    if diagnostic.identities and len(diagnostic.identities) != len(diagnostic.entries):
        return {}
    for position, row in enumerate(diagnostic.identities):
        if type(row) is not tuple or len(row) != 3:
            return {}
        ordinal, relation, index = row
        if (
            type(ordinal) is not int
            or ordinal != diagnostic.entries[position][0]
            or type(relation) is not str
            or relation not in _CROSS_RELATIONS
            or type(index) is not int
            or not -1 <= index <= _COUNTER_LIMIT
        ):
            return {}
    for value in (diagnostic.structure, diagnostic.transport):
        if value is not None and not _valid_structure(value):
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
    if diagnostic.identities:
        fields["provider_protocol_stream_identities"] = json.dumps(
            diagnostic.identities, separators=(",", ":")
        )
    for boundary, value in (("native", diagnostic.structure), ("transport", diagnostic.transport)):
        if value is not None:
            fields[f"provider_protocol_{boundary}_structure"] = json.dumps(
                value.entries, separators=(",", ":")
            )
            fields[f"provider_protocol_{boundary}_structure_truncated"] = int(value.truncated)
            fields[f"provider_protocol_{boundary}_aliases_exhausted"] = int(value.exhausted)
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
        self.has_search = False
        self._ordinal = 0
        self._truncated = False
        self._item_types: deque[tuple[int, str, str]] = deque(maxlen=_TRACE_LIMIT)
        self._identities: deque[tuple[int, str, int]] = deque(maxlen=_TRACE_LIMIT)
        self._structure = ResponseStructureTrace()
        self._transport: ResponseStructureDiagnostic | None = None

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
        item = event.get("item")
        item_id = item.get("id") if isinstance(item, Mapping) else event.get("item_id")
        self.has_search |= kind.startswith("response.web_search_call.") or (
            isinstance(item, Mapping) and _item_type(item.get("type")) == "web_search_call"
        )
        self._identities.append(
            (self._ordinal, *_cross_relation(item_id, index, pending, completed))
        )
        self._item_types.append(
            (
                self._ordinal,
                _item_type(item.get("type") if isinstance(item, Mapping) else None),
                "web_search_call"
                if registered is not None
                else _item_type(finished.get("type") if finished is not None else None),
            )
        )
        self._structure.record(event)
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
        return SearchStreamDiagnostic(
            tuple(self._entries),
            self._truncated,
            tuple(self._item_types),
            tuple(self._identities),
            self._structure.snapshot(),
            self._transport,
        )

    def transport_snapshot(self, value: object) -> None:
        # Exact internal envelope is checked by the caller; validate again on projection.
        self._transport = value if type(value) is ResponseStructureDiagnostic else None


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


_CROSS_RELATIONS = frozenset(
    {
        "missing",
        "invalid",
        "unknown",
        "pending_here",
        "pending_elsewhere",
        "completed_here",
        "completed_elsewhere",
        "ambiguous",
    }
)
_MISSING = object()
_ALIAS_LIMIT = 128
_ID_LENGTH_LIMIT = 1024
_NUMBER_STATES = frozenset({"missing", "invalid", "out_of_range", "present"})
_ALIAS_STATES = frozenset({"missing", "invalid", "oversized", "exhausted", "present"})


def _cross_relation(
    value: object,
    index: object,
    pending: Mapping[int, tuple[str, str]],
    completed: Mapping[int, dict[str, Any]],
) -> tuple[str, int]:
    if value is None:
        return "missing", -1
    if type(value) is not str or not value.strip():
        return "invalid", -1
    value = value.strip()
    match = None
    for key, registered in pending.items():
        if registered[0] == value:
            if match is not None:
                return "ambiguous", -1
            match = ("pending", key)
    for key, item in completed.items():
        if item.get("type") == "web_search_call" and item.get("id") == value:
            if match is not None:
                return "ambiguous", -1
            match = ("completed", key)
    if match is None:
        return "unknown", -1
    state, key = match
    return state + (
        "_here" if type(index) is int and index == key else "_elsewhere"
    ), key if 0 <= key <= _COUNTER_LIMIT else -1


def _number(value: object) -> tuple[str, int]:
    if value is _MISSING:
        return "missing", -1
    if type(value) is not int or value < 0:
        return "invalid", -1
    if value > _COUNTER_LIMIT:
        return "out_of_range", -1
    return "present", value


@dataclass(frozen=True)
class ResponseStructureDiagnostic:
    # ordinal, event type, index status/value, incoming item type,
    # upstream sequence status/value, identity status/local alias (0 = none).
    entries: tuple[tuple[int, str, str, int, str, str, int, str, int], ...]
    truncated: bool
    exhausted: bool


def _valid_structure(value: object) -> bool:
    if type(value) is not ResponseStructureDiagnostic:
        return False
    if type(value.truncated) is not bool or type(value.exhausted) is not bool:
        return False
    if type(value.entries) is not tuple or not 1 <= len(value.entries) <= _TRACE_LIMIT:
        return False
    for row in value.entries:
        if type(row) is not tuple or len(row) != 9:
            return False
        ordinal, kind, ix_state, index, item_type, seq_state, sequence, id_state, alias = row
        if type(ordinal) is not int or not 1 <= ordinal <= _COUNTER_LIMIT:
            return False
        for label, vocabulary in (
            (kind, _EVENT_TYPES),
            (ix_state, _NUMBER_STATES),
            (item_type, _ITEM_TYPES),
            (seq_state, _NUMBER_STATES),
            (id_state, _ALIAS_STATES),
        ):
            if type(label) is not str or len(label) > 64 or label not in vocabulary:
                return False
        for state, number in ((ix_state, index), (seq_state, sequence)):
            if (
                type(number) is not int
                or not -1 <= number <= _COUNTER_LIMIT
                or ((state == "present") != (number >= 0))
            ):
                return False
        if (
            type(alias) is not int
            or not 0 <= alias <= _ALIAS_LIMIT
            or ((id_state == "present") != (alias > 0))
        ):
            return False
    return True


class ResponseStructureTrace:
    """Attempt-local aliases; only salted digests are kept internally, never exported.

    First 128 distinct identities (at most 1024 characters each) get stable aliases.
    No eviction/reuse: unseen identities after exhaustion are explicitly uncorrelated.
    Both boundaries assign aliases in observation order, independently of acceptance.
    """

    def __init__(self) -> None:
        self._entries = deque(maxlen=_TRACE_LIMIT)
        self._ordinal = 0
        self._salt = os.urandom(32)
        self._aliases: dict[bytes, int] = {}
        self._exhausted = False
        self._truncated = False

    def record(self, event: Mapping[str, Any]) -> None:
        self._truncated |= len(self._entries) == _TRACE_LIMIT
        self._ordinal = min(self._ordinal + 1, _COUNTER_LIMIT)
        kind = event.get("type")
        kind = kind if type(kind) is str and len(kind) <= 64 and kind in _EVENT_TYPES else "other"
        item = event.get("item")
        identity = item.get("id") if isinstance(item, Mapping) else event.get("item_id")
        alias = 0
        if identity is None:
            state = "missing"
        elif type(identity) is not str:
            state = "invalid"
        elif len(identity) > _ID_LENGTH_LIMIT:
            state = "oversized"
        elif not identity.strip():
            state = "invalid"
        else:
            digest = hashlib.blake2b(
                identity.strip().encode("utf-8", errors="surrogatepass"),
                key=self._salt,
                digest_size=32,
            ).digest()
            if digest not in self._aliases and len(self._aliases) < _ALIAS_LIMIT:
                self._aliases[digest] = len(self._aliases) + 1
            alias = self._aliases.get(digest, 0)
            state = "present" if alias else "exhausted"
            self._exhausted |= not alias
        self._entries.append(
            (
                self._ordinal,
                kind,
                *_number(event.get("output_index", _MISSING)),
                _item_type(item.get("type") if isinstance(item, Mapping) else None),
                *_number(event.get("sequence_number", _MISSING)),
                state,
                alias,
            )
        )

    def snapshot(self) -> ResponseStructureDiagnostic:
        return ResponseStructureDiagnostic(tuple(self._entries), self._truncated, self._exhausted)
