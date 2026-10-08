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
