"""The session page's light reads: event heads, a trace without events, cut transcript parts.

Each light read must give what the full read gives, minus payloads: heads match
``/events`` for every filter and cursor, a trace without events has the same
spans, the trace over native trace rows equals the trace over whole events, and
a cut transcript part is the start of the whole one.
"""

from __future__ import annotations

import json
import sqlite3
from functools import partial

import pytest
from fastapi.testclient import TestClient
from tests.core.test_tool_completion import FinalTool, app_for, call
from tests.core.test_tool_round_publication_failure_matrix import _TwoCallProvider

from cayu import Message, RunRequest
from cayu.providers import ModelStreamEvent
from cayu.server import ServerConfig, create_server
from cayu.sessions import InMemorySessionStore
from cayu.sessions.store import SessionStore
from cayu.storage import SQLiteSessionStore
from cayu.storage._sqlite_functions import _register_sqlite_functions

_SESSION = "page-session"
_LONG_PROMPT = "Review this change. " + "ü context line\n" * 4000


def _store(tmp_path, backend):
    return (
        InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(tmp_path / "page.db")
    )


def _client(store) -> TestClient:
    provider = _TwoCallProvider(
        [call(), call(), [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()]]
    )
    app = app_for(store, provider, FinalTool())

    async def run():
        return [
            event
            async for event in app.run(
                RunRequest(
                    session_id=_SESSION,
                    agent_name="support",
                    messages=[Message.text("user", _LONG_PROMPT)],
                )
            )
        ]

    client = TestClient(create_server(app, config=ServerConfig.local_development()))
    client.__enter__()
    assert client.portal is not None
    events = client.portal.call(run)
    assert events[-1].type == "session.completed"
    return client


_HEAD_FIELDS = (
    "sequence",
    "id",
    "type",
    "session_id",
    "interaction_id",
    "agent_name",
    "environment_name",
    "workflow_name",
    "tool_name",
    "timestamp",
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_event_heads_match_events_for_every_filter_and_cursor(tmp_path, backend):
    store = _store(tmp_path, backend)
    client = _client(store)
    try:
        everything = client.get(f"/api/sessions/{_SESSION}/events", params={"limit": 1000}).json()
        middle = everything["events"][len(everything["events"]) // 2]["sequence"]
        for params in (
            {"limit": 1000},
            {"limit": 3},
            {"exclude_event_type": "model.text.delta", "order_by": "sequence_desc", "limit": 5},
            {"order_by": "sequence_desc", "before_sequence": middle, "limit": 4},
            {"after_sequence": middle, "limit": 2},
            {"event_type": "tool.call.completed"},
            {"tool_name": "final"},
            {"event_id": everything["events"][2]["id"], "limit": 1},
        ):
            events = client.get(f"/api/sessions/{_SESSION}/events", params=params)
            heads = client.get(f"/api/sessions/{_SESSION}/event-heads", params=params)
            assert events.status_code == heads.status_code == 200, heads.text
            full, light = events.json(), heads.json()
            assert {key: value for key, value in full.items() if key != "events"} == {
                key: value for key, value in light.items() if key != "events"
            }, params
            assert [
                {field: event[field] for field in _HEAD_FIELDS} for event in full["events"]
            ] == [{field: head[field] for field in _HEAD_FIELDS} for head in light["events"]]
            assert all(head["payload_bytes"] > 0 for head in light["events"])
            assert all("payload" not in head for head in light["events"])
        missing = client.get("/api/sessions/no-such-session/event-heads")
        assert missing.status_code == 404
    finally:
        client.__exit__(None, None, None)


def _use_reference_reads(store: SQLiteSessionStore) -> None:
    """Make the store's trace reads use the SessionStore defaults over whole events."""

    store.query_trace_events = partial(SessionStore.query_trace_events, store)
    store.query_transcript_tool_calls = partial(SessionStore.query_transcript_tool_calls, store)


def _trace_pages(client: TestClient, **params) -> list[dict]:
    pages, after = [], 0
    while True:
        page = client.get(
            f"/api/sessions/{_SESSION}/trace", params={**params, "after_sequence": after}
        ).json()
        pages.append(page)
        if not page["has_more"]:
            return pages
        after = page["next_sequence"]


@pytest.mark.parametrize("limit", [1, 3, 1000])
def test_trace_without_events_has_the_same_spans_as_the_reference(tmp_path, limit):
    store = SQLiteSessionStore(tmp_path / "native.db")
    client = _client(store)
    try:
        native_pages = _trace_pages(client, limit=limit)
        light_pages = _trace_pages(client, limit=limit, include_events="false")
        _use_reference_reads(store)
        reference_pages = _trace_pages(client, limit=limit)
        assert len(native_pages) == len(reference_pages) == len(light_pages)
        for native_page, reference_page, light_page in zip(
            native_pages, reference_pages, light_pages, strict=True
        ):
            assert native_page["spans"] == reference_page["spans"]
            assert [event["id"] for event in native_page["events"]] == [
                event["id"] for event in reference_page["events"]
            ]
            assert light_page["events"] == []
            assert {key: value for key, value in light_page.items() if key != "events"} == {
                key: value for key, value in native_page.items() if key != "events"
            }
        # Only tool calls carry an argument preview; these calls take no arguments.
        spans = [span for page in native_pages for span in page["spans"]]
        assert any(span["kind"] == "tool" for span in spans)
        assert all(span["detail"] is None for span in spans)
        # Every span event ID names an event of the page, as with events included.
        for page in native_pages:
            ids = {event["id"] for event in page["events"]}
            assert {event_id for span in page["spans"] for event_id in span["event_ids"]} <= ids
    finally:
        client.__exit__(None, None, None)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_transcript_cuts_long_parts_and_serves_each_message_whole(tmp_path, backend):
    store = _store(tmp_path, backend)
    client = _client(store)
    try:
        whole = client.get(f"/api/sessions/{_SESSION}/transcript").json()["messages"]
        cut = client.get(
            f"/api/sessions/{_SESSION}/transcript", params={"max_part_bytes": 1000}
        ).json()["messages"]
        assert [message["index"] for message in cut] == [message["index"] for message in whole]
        first = cut[0]["content"][0]
        assert first["truncated"] is True
        assert first["full_bytes"] == len(
            json.dumps(whole[0]["content"][0], ensure_ascii=False, separators=(",", ":")).encode()
        )
        assert len(first["text"].encode()) <= 1000
        assert whole[0]["content"][0]["text"].startswith(first["text"])
        for full_message, cut_message in zip(whole[1:], cut[1:], strict=True):
            assert full_message == cut_message
        for message in whole:
            single = client.get(f"/api/sessions/{_SESSION}/transcript/{message['index']}")
            assert single.status_code == 200, single.text
            assert single.json() == message
        assert client.get(f"/api/sessions/{_SESSION}/transcript/{len(whole)}").status_code == 404
        too_small = client.get(
            f"/api/sessions/{_SESSION}/transcript", params={"max_part_bytes": 10}
        )
        assert too_small.status_code == 422
    finally:
        client.__exit__(None, None, None)


def test_tool_span_detail_previews_arguments_like_the_dashboard():
    from cayu.sessions.trace import DETAIL_MAX_CHARS, _argument_preview

    assert _argument_preview({}) is None
    assert _argument_preview(None) is None
    assert _argument_preview({"count": 2, "path": "src/app.py", "mode": "r"}) == "src/app.py"
    assert _argument_preview({"count": 2, "flags": [""]}) == '{"count":2,"flags":[""]}'
    assert _argument_preview({"text": "ü" * 500}) == "ü" * DETAIL_MAX_CHARS


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_span_events_are_the_spans_own_and_on_request_its_childrens(tmp_path, backend):
    store = _store(tmp_path, backend)
    client = _client(store)
    try:
        whole = client.get(f"/api/sessions/{_SESSION}/trace", params={"limit": 1000}).json()
        by_id = {event["id"]: event for event in whole["events"]}
        for span, children in (
            (span, children) for span in whole["spans"] for children in (False, True)
        ):
            wanted = set(span["event_ids"]) | {
                event_id
                for child in whole["spans"]
                if children and child["parent_id"] == span["id"]
                for event_id in child["event_ids"]
            }
            response = client.get(
                f"/api/sessions/{_SESSION}/trace/spans/{span['id']}/events",
                params={
                    "session_instance_id": whole["session_instance_id"],
                    "include_children": str(children).lower(),
                },
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["span_id"] == span["id"]
            assert body["through_sequence"] == whole["through_sequence"]
            assert body["events"] == sorted(
                (by_id[event_id] for event_id in wanted), key=lambda event: event["sequence"]
            ), span["id"]
        assert (
            client.get(f"/api/sessions/{_SESSION}/trace/spans/tool:999999/events").status_code
            == 404
        )
        assert (
            client.get(
                f"/api/sessions/{_SESSION}/trace/spans/session/events",
                params={"session_instance_id": "another-incarnation"},
            ).status_code
            == 409
        )
    finally:
        client.__exit__(None, None, None)


def test_cut_tool_results_leave_out_long_structured_data_and_artifacts():
    from cayu.server._transcript_parts import truncate_message_parts

    rows = [{"row": index, "text": "ü" * 40} for index in range(100)]
    message = {
        "index": 3,
        "role": "tool",
        "content": [
            {"type": "tool_result", "content": "short", "structured": {"rows": rows}},
            {"type": "tool_result", "content": "x" * 5000, "artifacts": [{"name": "a"}]},
            {"type": "tool_call", "arguments": {"text": "y" * 5000}},
            {"type": "text", "text": "fits"},
        ],
    }
    cut = truncate_message_parts(message, 1000)["content"]
    assert cut[0]["structured"] is None and cut[0]["content"] == "short"
    assert cut[0]["truncated"] is True
    assert cut[0]["full_bytes"] > 1000
    assert cut[1]["content"] == "x" * 1000 and cut[1]["artifacts"] == [{"name": "a"}]
    assert cut[1]["truncated"] is True
    assert cut[2:] == message["content"][2:]


def test_whole_message_read_uses_the_absolute_index_after_retention(tmp_path):
    store = _store(tmp_path, "sqlite")
    client = _client(store)
    try:
        whole = client.get(f"/api/sessions/{_SESSION}/transcript").json()["messages"]
        assert len(whole) >= 3
        # Retention removes rows without renumbering the rest.
        connection = sqlite3.connect(tmp_path / "page.db")
        _register_sqlite_functions(connection)
        with connection:
            connection.execute(
                "DELETE FROM cayu_transcript_messages WHERE session_id = ? AND session_order = 1",
                (_SESSION,),
            )
        connection.close()
        assert client.get(f"/api/sessions/{_SESSION}/transcript/0").status_code == 404
        for message in whole[1:]:
            single = client.get(f"/api/sessions/{_SESSION}/transcript/{message['index']}")
            assert single.status_code == 200, single.text
            assert single.json() == message
    finally:
        client.__exit__(None, None, None)
