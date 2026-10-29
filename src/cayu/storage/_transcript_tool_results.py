"""Transcript tool results stored once, in their tool event.

A tool round's runtime publication writes the round's transcript message and
binds the ``tool.call.completed`` / ``tool.call.failed`` events that carry the
same result text, either appending them or referencing them by digest.
Session stores keep that text once, in the event payload's
``result.content``. The transcript part is stored with an empty
``content`` and a reference naming the event and pinning the text's UTF-8 size
and SHA-256 digest.

Every transcript read resolves references before it builds a ``Message``, so
readers see the same messages as before. Rows written before this format
carry their text inline and read unchanged. A reference whose event is missing
or whose text no longer matches its digest fails the read instead of
returning different content.

The event is the copy that stays: events a runtime-publication receipt appends
or references are never deleted while their session exists, and retention
compaction rewrites a referenced event only together with the transcript part
that references it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cayu.events import Event

#: Key of the reference object inside a stored ``tool_result`` part.
CONTENT_REFERENCE_KEY = "cayu_event_content"
#: Version of the reference object. Readers reject versions they do not know.
CONTENT_REFERENCE_FORMAT = 1
#: Results smaller than this stay inline; the reference would not save bytes.
MIN_REFERENCED_CONTENT_BYTES = 1024
#: Event types whose payload ``result.content`` a transcript part may reference.
TOOL_RESULT_EVENT_TYPES = ("tool.call.completed", "tool.call.failed")

_MARKER = '"' + CONTENT_REFERENCE_KEY + '"'


def _content_digest(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _event_result_content(event: Mapping[str, Any]) -> tuple[str, str, str] | None:
    """Return ``(tool_call_id, event_id, content)`` for one tool terminal event dump."""

    if event.get("type") not in TOOL_RESULT_EVENT_TYPES:
        return None
    payload = event.get("payload")
    if type(payload) is not dict:
        return None
    result = payload.get("result")
    if type(result) is not dict:
        return None
    content = result.get("content")
    tool_call_id = payload.get("tool_call_id")
    event_id = event.get("id")
    if type(content) is not str or type(tool_call_id) is not str or type(event_id) is not str:
        return None
    return tool_call_id, event_id, content


def reference_event_contents(
    transcript_payloads: Sequence[dict[str, Any]],
    event_payloads: Sequence[Mapping[str, Any]],
    durable_events: Iterable[Event] = (),
) -> list[dict[str, Any]]:
    """Return the stored form of transcript messages published with these events.

    ``event_payloads`` are the event documents the publication appends and
    ``durable_events`` the already stored events it references; both stay
    stored for as long as the session exists. A ``tool_result`` part is
    replaced by a reference only when exactly one of the events is a tool
    terminal event for the same tool call whose ``result.content`` is the
    identical string, and that string is at least
    :data:`MIN_REFERENCED_CONTENT_BYTES` long. Everything else is returned as is.
    """

    candidates: dict[str, list[tuple[str, str]]] = {}
    for event in (
        *event_payloads,
        *(
            {"type": str(event.type), "id": event.id, "payload": event.payload}
            for event in durable_events
        ),
    ):
        found = _event_result_content(event)
        if found is not None:
            tool_call_id, event_id, content = found
            candidates.setdefault(tool_call_id, []).append((event_id, content))
    if not candidates:
        return list(transcript_payloads)
    stored: list[dict[str, Any]] = []
    for document in transcript_payloads:
        content = document.get("content")
        if document.get("role") != "tool" or type(content) is not list:
            stored.append(document)
            continue
        parts: list[Any] = []
        changed = False
        for part in content:
            reference = None
            if (
                type(part) is dict
                and part.get("type") == "tool_result"
                and CONTENT_REFERENCE_KEY not in part
                and type(part.get("content")) is str
            ):
                tool_call_id = part.get("tool_call_id")
                options = candidates.get(tool_call_id, []) if type(tool_call_id) is str else []
                if len(options) == 1 and options[0][1] == part["content"]:
                    event_id = options[0][0]
                    encoded = part["content"].encode("utf-8")
                    if len(encoded) >= MIN_REFERENCED_CONTENT_BYTES:
                        reference = {
                            "format": CONTENT_REFERENCE_FORMAT,
                            "event_id": event_id,
                            "bytes": len(encoded),
                            "sha256": _content_digest(encoded),
                        }
            if reference is None:
                parts.append(part)
            else:
                parts.append({**part, "content": "", CONTENT_REFERENCE_KEY: reference})
                changed = True
        stored.append({**document, "content": parts} if changed else document)
    return stored


def may_reference(raw: Any) -> bool:
    """Cheap pre-check on a stored message before looking for references.

    Serialized JSON is searched for the reference key; an already parsed
    document always needs the full check.
    """

    if isinstance(raw, bytes):
        return _MARKER.encode("utf-8") in raw
    if isinstance(raw, str):
        return _MARKER in raw
    return True


def _reference(part: Any) -> dict[str, Any] | None:
    if type(part) is not dict or CONTENT_REFERENCE_KEY not in part:
        return None
    reference = part[CONTENT_REFERENCE_KEY]
    if (
        type(reference) is not dict
        or reference.get("format") != CONTENT_REFERENCE_FORMAT
        or set(reference) != {"format", "event_id", "bytes", "sha256"}
        or type(reference["event_id"]) is not str
        or type(reference["bytes"]) is not int
        or type(reference["sha256"]) is not str
        or part.get("type") != "tool_result"
        or part.get("content") != ""
    ):
        raise ValueError("Stored transcript tool result reference is malformed or unsupported.")
    return reference


def referenced_event_ids(document: Any) -> list[str]:
    """Event ids the stored message document references, in part order."""

    if type(document) is not dict or type(document.get("content")) is not list:
        return []
    ids: list[str] = []
    for part in document["content"]:
        reference = _reference(part)
        if reference is not None:
            ids.append(reference["event_id"])
    return ids


def resolve_message_document(document: Any, contents: Mapping[str, Any]) -> Any:
    """Return the logical message document with referenced contents restored.

    ``contents`` maps event id to the event payload's ``result.content``. A
    missing event or text that does not match the pinned size and digest
    raises ``ValueError``.
    """

    if not referenced_event_ids(document):
        return document
    parts: list[Any] = []
    for part in document["content"]:
        reference = _reference(part)
        if reference is None:
            parts.append(part)
            continue
        content = contents.get(reference["event_id"])
        if type(content) is not str:
            raise ValueError("Stored transcript tool result references a missing tool event.")
        encoded = content.encode("utf-8")
        if len(encoded) != reference["bytes"] or _content_digest(encoded) != reference["sha256"]:
            raise ValueError("Stored transcript tool result no longer matches its tool event.")
        restored = {key: value for key, value in part.items() if key != CONTENT_REFERENCE_KEY}
        restored["content"] = content
        parts.append(restored)
    return {**document, "content": parts}


def without_references(document: Any) -> Any:
    """Drop references and keep the empty content, for readers of narrative text only."""

    if not referenced_event_ids(document):
        return document
    return {
        **document,
        "content": [
            {key: value for key, value in part.items() if key != CONTENT_REFERENCE_KEY}
            if type(part) is dict
            else part
            for part in document["content"]
        ],
    }


def compact_referenced_parts(
    document: Any, compacted_event_ids: Iterable[str], marker: Any
) -> tuple[Any, bool]:
    """Replace references to compacted events with an inline retention marker.

    ``marker(byte_count, sha256)`` returns the marker text. The reference pins
    both values, so the event's text is not needed.
    """

    compacted = set(compacted_event_ids)
    if type(document) is not dict or type(document.get("content")) is not list:
        return document, False
    parts: list[Any] = []
    changed = False
    for part in document["content"]:
        reference = _reference(part)
        if reference is None or reference["event_id"] not in compacted:
            parts.append(part)
            continue
        restored = {key: value for key, value in part.items() if key != CONTENT_REFERENCE_KEY}
        restored["content"] = marker(reference["bytes"], reference["sha256"])
        parts.append(restored)
        changed = True
    return ({**document, "content": parts} if changed else document), changed


__all__ = [
    "CONTENT_REFERENCE_FORMAT",
    "CONTENT_REFERENCE_KEY",
    "MIN_REFERENCED_CONTENT_BYTES",
    "TOOL_RESULT_EVENT_TYPES",
    "compact_referenced_parts",
    "may_reference",
    "reference_event_contents",
    "referenced_event_ids",
    "resolve_message_document",
    "without_references",
]
