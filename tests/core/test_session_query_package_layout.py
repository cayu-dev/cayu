"""Session read contracts compose independently of store implementations."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu


def _assert_store_independent(code: str, public_module: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import pickle
import sys
from typing import get_type_hints

class RejectStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("cayu.storage") or fullname in {
            "cayu.sessions.base", "cayu.tasks.memory",
        }:
            raise AssertionError(f"Query contracts imported {fullname}")
sys.meta_path.insert(0, RejectStores())
public = importlib.import_module(sys.argv[1])
"""
            + code,
            public_module,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_session_query_public_imports_and_historical_pickle_globals():
    from cayu.sessions import base, queries, records

    surfaces = [importlib.import_module(name) for name in ("cayu", "cayu.sessions", "cayu.runtime")]
    for owner, names in (
        (records, ("PendingActionSession", "MAX_SESSION_ID_BYTES")),
        (
            queries,
            (
                "SessionOrder",
                "SessionDebugState",
                "SessionQuery",
                "SessionAggregateFilter",
                "SessionStatusCounts",
                "SessionListResult",
                "LabelSelectorOperator",
                "LabelSelectorRequirement",
                "MAX_SESSION_LIST_CURSOR_BYTES",
            ),
        ),
    ):
        for name in names:
            canonical = getattr(owner, name)
            assert getattr(base, name) is canonical
            exported_surfaces = surfaces[1:] if name == "SessionDebugState" else surfaces
            assert all(getattr(surface, name) is canonical for surface in exported_surfaces)
            if isinstance(canonical, type):
                assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical


@pytest.mark.parametrize("public_module", ("cayu", "cayu.sessions", "cayu.runtime"))
def test_session_queries_filter_and_page_without_loading_stores(public_module):
    _assert_store_independent(
        """
from datetime import UTC, datetime
from cayu.sessions.invocation import InvocationOrigin, SessionInvocation
from cayu.sessions.queries import (
    _session_after_cursor, _session_matches, _sort_sessions, copy_session_query,
    decode_session_cursor, encode_session_cursor, session_next_cursor,
    session_query_from_aggregate_filter,
)

stamp = datetime(2026, 1, 1, tzinfo=UTC)
def session(name):
    return public.Session(
        id=name, agent_name="agent", provider_name="provider", model="model",
        created_at=stamp, updated_at=stamp, last_activity_at=stamp,
        labels={"team": "support", "stage": "active"},
        invocation=SessionInvocation(
            origin=InvocationOrigin(trust="unattributed"),
            root_invocation_id="12345678-1234-4234-8234-123456789abc",
            root_session_id=name, source="sdk_run",
        ),
    )

rows = [session("b"), session("a")]
selector = public.LabelSelectorRequirement(key="stage", operator="in", values=("active",))
filters = public.SessionAggregateFilter(labels={"team": "support"}, label_selectors=(selector,))
query = session_query_from_aggregate_filter(filters)
assert all(_session_matches(row, query) for row in rows)
copied = copy_session_query(query)
copied.labels["team"] = "changed"
copied.label_selectors[0].key = "changed"
assert query.labels == filters.labels == {"team": "support"}
assert query.label_selectors[0].key == filters.label_selectors[0].key == "stage"
assert not _session_matches(rows[0], copied)
for order in public.SessionOrder:
    ordered = _sort_sessions(rows, order)
    assert [row.id for row in ordered] == ["a", "b"]
    cursor = session_next_cursor(ordered[:1], True, order)
    value, name = decode_session_cursor(cursor)
    assert (value, name) == (stamp, "a")
    assert [row.id for row in ordered if _session_after_cursor(row, order, value, name)] == ["b"]
    projection = public.PendingActionSession.from_session(ordered[0])
    assert encode_session_cursor(projection, order) == cursor
    assert session_next_cursor(ordered, False, order) is None
result = public.SessionListResult(sessions=rows)
for value in (query, filters, result, projection):
    assert pickle.loads(pickle.dumps(value)) == value
    assert get_type_hints(type(value))
    assert type(value).model_json_schema()
try:
    decode_session_cursor("invalid")
except ValueError as exc:
    assert str(exc) == "Invalid session cursor."
else:
    raise AssertionError("Malformed cursor was accepted")
""",
        public_module,
    )


def test_event_query_public_imports_and_historical_pickle_globals():
    from cayu.sessions import base, event_queries

    surfaces = [importlib.import_module(name) for name in ("cayu", "cayu.sessions", "cayu.runtime")]
    for name in ("EventOrder", "EventQuery", "EventQueryResultTooLarge"):
        canonical = getattr(event_queries, name)
        assert getattr(base, name) is canonical
        assert all(getattr(surface, name) is canonical for surface in surfaces)
        assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical


@pytest.mark.parametrize("public_module", ("cayu", "cayu.sessions", "cayu.runtime"))
def test_event_queries_select_records_without_loading_stores(public_module):
    _assert_store_independent(
        """
from datetime import UTC, datetime, timedelta
from cayu.events import Event, EventType
from cayu.sessions.event_queries import _event_record_matches, copy_event_query

stamp = datetime(2026, 1, 1, tzinfo=UTC)
query = public.EventQuery(
    session_id="session", event_types=[EventType.SESSION_COMPLETED],
    after_sequence=1, before_sequence=3, since=stamp, until=stamp + timedelta(seconds=1),
)
record = public.EventRecord(sequence=2, event=Event(
    type=EventType.SESSION_COMPLETED, session_id="session", timestamp=stamp,
))
allowed = frozenset(str(kind) for kind in query.event_types)
assert _event_record_matches(record, query, allowed, frozenset())
assert not _event_record_matches(record, query, allowed, allowed)
for changes in ({"after_sequence": 2}, {"before_sequence": 2}, {"session_id": "other"},
                {"until": stamp, "since": None}):
    copied = copy_event_query(query, update=changes)
    assert not _event_record_matches(record, copied, allowed, frozenset())
assert query.session_id == "session" and query.after_sequence == 1
assert pickle.loads(pickle.dumps(query)) == query
assert get_type_hints(public.EventQuery)
assert public.EventQuery.model_json_schema()
try:
    copy_event_query(query, update={"after_sequence": 3})
except ValueError:
    pass
else:
    raise AssertionError("Query copy bypassed sequence-bound validation")
error = public.EventQueryResultTooLarge(64)
assert error.max_bytes == 64 and str(error) == "Event query exceeds the 64-byte safety limit."
""",
        public_module,
    )


def test_transcript_query_public_imports_and_historical_pickle_globals():
    from cayu.sessions import base, transcript_queries

    surfaces = [importlib.import_module(name) for name in ("cayu", "cayu.sessions", "cayu.runtime")]
    for name in (
        "TranscriptQuery",
        "TranscriptPage",
        "TranscriptSearchQuery",
        "TranscriptSearchHit",
        "TranscriptSearchResult",
    ):
        canonical = getattr(transcript_queries, name)
        assert getattr(base, name) is canonical
        assert all(getattr(surface, name) is canonical for surface in surfaces)
        assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical


@pytest.mark.parametrize("public_module", ("cayu", "cayu.sessions", "cayu.runtime"))
def test_transcript_queries_filter_and_search_without_loading_stores(public_module):
    _assert_store_independent(
        """
from cayu.messages import Message, TextPart, ThinkingPart
from cayu.sessions.transcript_queries import (
    copy_transcript_search_query, decode_transcript_search_cursor,
    encode_transcript_search_cursor, filter_transcript_records,
    transcript_search_document, transcript_search_document_from_text,
    transcript_search_hit_from_message, transcript_search_query_tokens,
    transcript_search_score,
)

message = Message(role="assistant", content=(
    ThinkingPart(text="private reasoning"), TextPart(text="Straße café"),
))
records = [
    public.TranscriptRecord(index=2, message=Message(
        role="assistant", content=(ThinkingPart(text="hidden"),),
    )),
    public.TranscriptRecord(index=7, interaction_id="interaction", message=message),
]
filtered = filter_transcript_records(records, include_thinking=False)
assert [record.index for record in filtered] == [7]
assert filtered[0].interaction_id == "interaction"
assert filtered[0].message.content == (TextPart(text="Straße café"),)
assert len(records[1].message.content) == 2
assert filter_transcript_records(records, include_thinking=True) is records
assert transcript_search_query_tokens("STRASSE Straße café") == ("strasse", "café")
assert transcript_search_document(message) == transcript_search_document_from_text("Straße café")
query = public.TranscriptSearchQuery(text="STRASSE", session_ids=("session",))
score = transcript_search_score("Straße café", query)
assert score > 0
hit = transcript_search_hit_from_message(
    session_id="session", transcript_index=7, interaction_id="interaction",
    message=message, max_text_bytes=100, raw_score=score,
)
assert hit.text == "Straße café" and hit.text_part_indexes == (1,)
cursor = encode_transcript_search_cursor(
    query, raw_score=int(score), session_id="session", transcript_index=7,
)
continued = copy_transcript_search_query(query, cursor=cursor)
assert query.cursor is None
assert decode_transcript_search_cursor(continued) == (int(score), "session", 7)
try:
    decode_transcript_search_cursor(public.TranscriptSearchQuery(
        text="different", session_ids=("session",), cursor=cursor,
    ))
except ValueError as exc:
    assert str(exc) == "Invalid transcript search cursor."
else:
    raise AssertionError("Search cursor escaped its query scope")
result = public.TranscriptSearchResult(query=query, hits=(hit,), matched_records_examined=1)
page = public.TranscriptPage(records=filtered, total_records=2)
for value in (public.TranscriptQuery(session_id="session"), page, query, hit, result):
    assert pickle.loads(pickle.dumps(value)) == value
    assert get_type_hints(type(value))
    assert type(value).model_json_schema()
""",
        public_module,
    )
