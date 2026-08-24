from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from tests.core.session_topology_conformance import (
    assert_session_topology_store_conformance,
)

from cayu import SQLiteSessionStore
from cayu.core import Event, EventType, Message
from cayu.runtime import InMemorySessionStore, RunRequest, SessionIdentity, SessionStore
from cayu.runtime.sessions import (
    SESSION_TOPOLOGY_MAX_NODES,
    EventQuery,
    EventQueryResultTooLarge,
    SessionLineageNode,
    SessionLineageResult,
    SessionStatus,
    SessionTopologyBranch,
    SessionTopologyCycle,
    SessionTopologyNode,
    SessionTopologyQuery,
    SessionTopologyStoreResult,
    build_session_topology_result,
    decode_session_topology_cursor,
)


def test_session_lineage_page_requires_portable_identifier_order() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)

    with pytest.raises(ValidationError, match="stable creation ordering"):
        SessionLineageResult(
            parent_session_id="parent",
            children=(
                SessionLineageNode(
                    id="ä",
                    parent_session_id="parent",
                    created_at=created_at,
                ),
                SessionLineageNode(
                    id="Z",
                    parent_session_id="parent",
                    created_at=created_at,
                ),
            ),
        )


def test_session_lineage_page_rejects_descending_creation_time() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)

    with pytest.raises(ValidationError, match="stable creation ordering"):
        SessionLineageResult(
            parent_session_id="parent",
            children=(
                SessionLineageNode(
                    id="later",
                    parent_session_id="parent",
                    created_at=created_at + timedelta(seconds=1),
                ),
                SessionLineageNode(
                    id="earlier",
                    parent_session_id="parent",
                    created_at=created_at,
                ),
            ),
        )


def _make_store(
    store_factory: type[InMemorySessionStore] | type[SQLiteSessionStore],
    tmp_path,
) -> SessionStore:
    if store_factory is SQLiteSessionStore:
        return SQLiteSessionStore(tmp_path / "topology.sqlite")
    return InMemorySessionStore()


async def _close_store(store: SessionStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_session_topology_store_conformance(store_factory, tmp_path) -> None:
    async def exercise() -> None:
        store = _make_store(store_factory, tmp_path)
        try:
            await assert_session_topology_store_conformance(store)
        finally:
            await _close_store(store)

    asyncio.run(exercise())


@pytest.mark.parametrize("store_factory", [InMemorySessionStore, SQLiteSessionStore])
def test_bounded_event_query_rejects_payload_bytes_before_return(
    store_factory,
    tmp_path,
) -> None:
    async def exercise() -> None:
        store = _make_store(store_factory, tmp_path)
        try:
            await store.create(
                RunRequest(
                    agent_name="agent",
                    session_id="bounded-event-session",
                    messages=[Message.text("user", "test")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await store.append_events(
                "bounded-event-session",
                [
                    Event(
                        id="large-event",
                        type=EventType.SESSION_STARTED,
                        session_id="bounded-event-session",
                        payload={"irrelevant": "x" * 4096},
                    )
                ],
            )
            query = EventQuery(
                session_id="bounded-event-session",
                limit=1,
            )

            with pytest.raises(EventQueryResultTooLarge):
                await store.query_events_bounded(query, max_bytes=1024)

            records = await store.query_events_bounded(query, max_bytes=8192)
            assert [record.event.id for record in records] == ["large-event"]
        finally:
            await _close_store(store)

    asyncio.run(exercise())


def _node(
    node_id: str,
    *,
    parent_session_id: str | None = None,
    created_offset: int = 0,
) -> SessionTopologyNode:
    created_at = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=created_offset)
    return SessionTopologyNode(
        id=node_id,
        agent_name="agent",
        provider_name="provider",
        model="model",
        parent_session_id=parent_session_id,
        causal_budget_id="budget",
        runtime_name="cayu",
        runtime_version="test",
        environment_name="test",
        status=SessionStatus.RUNNING,
        created_at=created_at,
        updated_at=created_at,
        last_activity_at=created_at,
    )


def test_topology_node_ceiling_preserves_continuation_for_every_nonempty_branch() -> None:
    focus = _node("focus")
    parents = tuple(_node(f"parent-{index}") for index in range(50))
    candidates = tuple(
        tuple(
            _node(
                f"child-{parent_index}-{child_index}",
                parent_session_id=parent.id,
                created_offset=child_index,
            )
            for child_index in range(11)
        )
        for parent_index, parent in enumerate(parents)
    )

    result = build_session_topology_result(
        focus=focus,
        ancestors=(),
        expanded_parents=parents,
        branch_candidates=candidates,
        child_limit=100,
    )

    retained_ids = {
        result.focus.id,
        *(parent.id for parent in result.expanded_parents),
        *(child.id for branch in result.branches for child in branch.children),
    }
    assert len(retained_ids) == SESSION_TOPOLOGY_MAX_NODES
    assert all(branch.children for branch in result.branches)
    for branch in result.branches:
        if branch.has_more:
            assert branch.next_cursor is not None
            _, cursor_id = decode_session_topology_cursor(
                branch.next_cursor,
                parent_session_id=branch.parent_session_id,
            )
            assert cursor_id == branch.children[-1].id


def test_topology_query_rejects_unbounded_or_mismatched_branch_inputs() -> None:
    with pytest.raises(ValidationError, match="expanded_parent_ids"):
        SessionTopologyQuery(
            focus_session_id="focus",
            expanded_parent_ids=tuple(f"parent-{index}" for index in range(51)),
        )

    with pytest.raises(ValidationError, match="child_cursors"):
        SessionTopologyQuery(
            focus_session_id="focus",
            expanded_parent_ids=("parent",),
            child_cursors={"different-parent": "cursor"},
        )


def test_topology_result_rejects_contradictory_lineage_and_branch_edges() -> None:
    root = _node("root")
    focus = _node("focus", parent_session_id="root")
    wrong_child = _node("child", parent_session_id="different-parent")
    contradictory_focus = focus.model_copy(update={"status": SessionStatus.COMPLETED})

    with pytest.raises(ValidationError, match="omitted"):
        SessionTopologyStoreResult(focus=focus)

    with pytest.raises(ValidationError, match="contradictory parent edge"):
        SessionTopologyStoreResult(
            focus=focus,
            ancestors=(root,),
            expanded_parents=(root,),
            branches=(
                SessionTopologyBranch(
                    parent_session_id="root",
                    children=(wrong_child,),
                ),
            ),
        )

    with pytest.raises(ValidationError, match="contradictory representations"):
        SessionTopologyStoreResult(
            focus=focus,
            ancestors=(root,),
            expanded_parents=(contradictory_focus,),
            branches=(SessionTopologyBranch(parent_session_id="focus"),),
        )

    loop = _node("loop", parent_session_id="loop")
    with pytest.raises(ValidationError, match="cycle"):
        SessionTopologyStoreResult(
            focus=root,
            expanded_parents=(loop,),
            branches=(SessionTopologyBranch(parent_session_id="loop"),),
        )


def test_topology_result_builder_consumes_only_one_bounded_sentinel_page() -> None:
    focus = _node("focus")

    def candidates():
        yield _node("child-a", parent_session_id="focus")
        yield _node("child-b", parent_session_id="focus")
        yield _node("child-c", parent_session_id="focus")
        raise AssertionError("The topology builder consumed beyond its bounded sentinel.")

    result = build_session_topology_result(
        focus=focus,
        ancestors=(),
        expanded_parents=(focus,),
        branch_candidates=(candidates(),),
        child_limit=2,
    )

    assert [child.id for child in result.branches[0].children] == [
        "child-a",
        "child-b",
    ]
    assert result.branches[0].has_more is True


def test_in_memory_topology_rejects_durable_parent_cycle() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        await assert_session_topology_store_conformance(store)
        store._sessions["topology-root"].parent_session_id = "topology-focus"

        with pytest.raises(SessionTopologyCycle):
            await store.query_session_topology(
                SessionTopologyQuery(focus_session_id="topology-focus")
            )

    asyncio.run(exercise())


def test_in_memory_topology_rejects_cycle_in_expanded_branch() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        await assert_session_topology_store_conformance(store)
        store._sessions["topology-root-sibling"].parent_session_id = "topology-root-sibling"

        with pytest.raises(SessionTopologyCycle):
            await store.query_session_topology(
                SessionTopologyQuery(
                    focus_session_id="topology-focus",
                    expanded_parent_ids=("topology-root-sibling",),
                )
            )

    asyncio.run(exercise())


def test_in_memory_topology_child_page_does_not_scan_session_registry() -> None:
    class NoValuesScanDict(dict):
        def values(self):
            raise AssertionError("Topology child paging scanned the session registry.")

    async def exercise() -> None:
        store = InMemorySessionStore()
        await assert_session_topology_store_conformance(store)
        store._sessions = NoValuesScanDict(store._sessions)

        result = await store.query_session_topology(
            SessionTopologyQuery(
                focus_session_id="topology-focus",
                expanded_parent_ids=("topology-focus",),
                child_limit=2,
            )
        )

        assert [node.id for node in result.branches[0].children] == [
            "topology-child-a",
            "topology-child-b",
        ]

    asyncio.run(exercise())


def test_sqlite_topology_rejects_durable_parent_cycle(tmp_path) -> None:
    db_path = tmp_path / "topology-cycle.sqlite"

    async def exercise() -> None:
        store = SQLiteSessionStore(db_path)
        try:
            await assert_session_topology_store_conformance(store)
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    "UPDATE cayu_sessions SET parent_session_id = ? WHERE id = ?",
                    ("topology-focus", "topology-root"),
                )

            with pytest.raises(SessionTopologyCycle):
                await store.query_session_topology(
                    SessionTopologyQuery(focus_session_id="topology-focus")
                )
        finally:
            await store.close()

    asyncio.run(exercise())


def test_sqlite_topology_rejects_cycle_in_expanded_branch(tmp_path) -> None:
    db_path = tmp_path / "topology-expanded-cycle.sqlite"

    async def exercise() -> None:
        store = SQLiteSessionStore(db_path)
        try:
            await assert_session_topology_store_conformance(store)
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    "UPDATE cayu_sessions SET parent_session_id = id WHERE id = ?",
                    ("topology-root-sibling",),
                )

            with pytest.raises(SessionTopologyCycle):
                await store.query_session_topology(
                    SessionTopologyQuery(
                        focus_session_id="topology-focus",
                        expanded_parent_ids=("topology-root-sibling",),
                    )
                )
        finally:
            await store.close()

    asyncio.run(exercise())


def test_sqlite_topology_child_query_uses_composite_index(tmp_path) -> None:
    db_path = tmp_path / "topology-plan.sqlite"
    store = SQLiteSessionStore(db_path)
    asyncio.run(store.close())

    with sqlite3.connect(db_path) as connection:
        plan = " ".join(
            str(row[3])
            for row in connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT id
                FROM cayu_sessions
                WHERE parent_session_id = ?
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                ("parent", 101),
            ).fetchall()
        )

    assert "idx_cayu_sessions_parent_created_id" in plan
    assert "USE TEMP B-TREE" not in plan


def test_sqlite_topology_child_page_work_is_independent_of_branch_size(tmp_path) -> None:
    db_path = tmp_path / "topology-bounded-work.sqlite"

    async def exercise() -> int:
        store = SQLiteSessionStore(db_path)
        try:
            await store.create(
                RunRequest(
                    agent_name="parent",
                    session_id="parent",
                    messages=[Message.text("user", "parent")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            timestamp = "2026-01-01T00:00:00+00:00"
            with store._connection:
                store._connection.execute(
                    """
                    WITH RECURSIVE numbers(value) AS (
                        VALUES(0)
                        UNION ALL
                        SELECT value + 1 FROM numbers WHERE value < 99999
                    )
                    INSERT INTO cayu_sessions (
                        id, instance_id, agent_name, provider_name, model, parent_session_id,
                        causal_budget_id, runtime_name, runtime_version,
                        environment_name, status, created_at, updated_at,
                        last_activity_at, run_epoch, invocation_json, metadata_json
                    )
                    SELECT
                        printf('child-%06d', value),
                        printf('00000000-0000-4000-a000-%012d', value),
                        'agent', 'fake', 'fake-model',
                        'parent', 'budget', 'cayu', NULL, NULL, 'pending', ?, ?, ?,
                        0,
                        json_object(
                            'schema_version', 1,
                            'origin', json_object('trust', 'unattributed'),
                            'root_invocation_id',
                            'f055bedc-62cf-4fa4-979a-d0378ca93131',
                            'root_session_id', 'parent',
                            'source', 'subagent'
                        ),
                        '{}'
                    FROM numbers
                    """,
                    (timestamp, timestamp, timestamp),
                )

            progress_calls = 0

            def count_progress() -> int:
                nonlocal progress_calls
                progress_calls += 1
                return 0

            store._read_connection.set_progress_handler(count_progress, 1)
            result = await store.query_session_topology(
                SessionTopologyQuery(
                    focus_session_id="parent",
                    expanded_parent_ids=("parent",),
                    child_limit=25,
                )
            )
            store._read_connection.set_progress_handler(None, 0)
            assert len(result.branches[0].children) == 25
            assert result.branches[0].has_more is True
            return progress_calls
        finally:
            await store.close()

    progress_calls = asyncio.run(exercise())
    assert progress_calls < 5000
