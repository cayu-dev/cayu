"""Session discovery composes over current, consistently locked memory indexes."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests._session_provenance import fixture_session_invocation

import cayu
from cayu.sessions import _memory_session_queries as queries
from cayu.sessions._access_rules import (
    SessionAccessRule,
    SessionAccessScope,
    SessionAccessSelector,
    _SessionAccessBounds,
)
from cayu.sessions.lineage import SessionLineageQuery
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import Session
from cayu.sessions.topology import SessionTopologyQuery


def _state() -> queries.MemorySessionQueryState:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    sessions = {
        name: Session(
            id=name,
            agent_name="assistant",
            provider_name="fake",
            model="fake",
            causal_budget_id="root",
            invocation=fixture_session_invocation(name),
            parent_session_id=None if name == "root" else "root",
            created_at=now,
            updated_at=now,
            last_activity_at=now,
            labels={"team": "one" if name == "root" else "two"},
        )
        for name in ("root", "child-a", "child-b")
    }
    return queries.MemorySessionQueryState(
        checkpoints={"child-a": {"pending_interruption_cascade": {}}},
        child_lifecycle_candidate_keys_by_parent={},
        child_lifecycle_event_records={},
        child_session_keys_by_parent={"root": [(now, "child-a"), (now, "child-b")]},
        ownership_clock=lambda: now,
        pending_action_event_records={},
        pending_action_latest_barrier_records={},
        pending_action_session_ids=set(),
        queued_dispatch_terminal_receipt_keys=[],
        queued_dispatch_terminal_receipts={},
        session_event_records={},
        session_operation_records={},
        sessions=sessions,
    )


def test_session_queries_import_without_store_implementations() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import sys
class RejectStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"cayu.sessions.base", "cayu.sessions.memory", "cayu.sessions.store",
                        "cayu.storage.sqlite", "cayu.storage.postgres"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectStores())
owner = importlib.import_module("cayu.sessions._memory_session_queries")
assert callable(owner.list_sessions)
assert callable(owner.query_pending_actions)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_session_listing_composes_with_independent_state_and_access_bounds() -> None:
    async def scenario():
        state = _state()
        lock = asyncio.Lock()

        @asynccontextmanager
        async def read():
            async with lock:
                yield state

        scope = SessionAccessScope(
            read=(
                SessionAccessRule(selectors=(SessionAccessSelector(key="team", values=("two",)),)),
            )
        )
        bounds = _SessionAccessBounds(scope, scope)
        page = await queries.list_sessions(
            read,
            SessionQuery(limit=1, include_total_count=True),
            pending_interruption_cascade_only=False,
            access_bounds=bounds,
        )
        assert page.total_count == 2
        assert page.next_cursor is not None
        second = await queries.list_sessions(
            read,
            SessionQuery(limit=1, cursor=page.next_cursor),
            pending_interruption_cascade_only=False,
            access_bounds=bounds,
        )
        assert {page.sessions[0].id, second.sessions[0].id} == {"child-a", "child-b"}
        page.sessions[0].labels.clear()
        assert state.sessions[page.sessions[0].id].labels
        pending = await queries.list_sessions(
            read,
            None,
            pending_interruption_cascade_only=True,
        )
        assert [session.id for session in pending.sessions] == ["child-a"]
        aggregate = await queries.aggregate_operational_snapshot(read)
        assert aggregate.total_count == 3
        assert aggregate.as_of == state.ownership_clock()
        assert not lock.locked()

    asyncio.run(scenario())


def test_session_tree_queries_use_independent_indexes_and_reject_stale_entries() -> None:
    async def scenario():
        state = _state()
        lock = asyncio.Lock()

        @asynccontextmanager
        async def read():
            async with lock:
                yield state

        first = await queries.query_session_lineage(
            read, SessionLineageQuery(parent_session_id="root", limit=1)
        )
        assert [child.id for child in first.children] == ["child-a"]
        assert first.next_cursor is not None
        second = await queries.query_session_lineage(
            read, SessionLineageQuery(parent_session_id="root", limit=1, cursor=first.next_cursor)
        )
        assert [child.id for child in second.children] == ["child-b"]
        assert not second.has_more
        topology = await queries.query_session_topology(
            read, SessionTopologyQuery(focus_session_id="child-a", expanded_parent_ids=("root",))
        )
        assert [ancestor.id for ancestor in topology.ancestors] == ["root"]
        del state.sessions["child-a"]
        with pytest.raises(RuntimeError, match="Inconsistent in-memory session lineage index"):
            await queries.query_session_lineage(read, SessionLineageQuery(parent_session_id="root"))
        assert not lock.locked()

    asyncio.run(scenario())


def test_store_discovery_view_reads_replaced_indexes_after_acquiring_lock() -> None:
    from cayu.sessions.memory import InMemorySessionStore

    async def scenario():
        store = InMemorySessionStore()
        state = _state()
        await store._lock.acquire()
        task = asyncio.create_task(
            store.query_session_lineage(SessionLineageQuery(parent_session_id="root"))
        )
        try:
            await asyncio.sleep(0)
            assert not task.done()
            store._sessions = state.sessions
            store._child_session_keys_by_parent = state.child_session_keys_by_parent
        finally:
            store._lock.release()
        assert [child.id for child in (await task).children] == ["child-a", "child-b"]
        store._child_session_keys_by_parent = {}
        assert not (
            await store.query_session_lineage(SessionLineageQuery(parent_session_id="root"))
        ).children

    asyncio.run(scenario())
