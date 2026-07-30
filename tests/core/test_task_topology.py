from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.core.task_topology_conformance import (
    assert_task_topology_bounded_projection_conformance,
    assert_task_topology_store_conformance,
)

from cayu import InMemoryTaskStore, SQLiteTaskStore, TaskCreate, TaskStore
from cayu.runtime.tasks import (
    TASK_TOPOLOGY_MAX_ANCESTOR_DEPTH,
    TASK_TOPOLOGY_MAX_EXPANDED_PARENTS,
    TASK_TOPOLOGY_MAX_NODES,
    TaskTopologyCycle,
    TaskTopologyInconsistent,
    TaskTopologyQuery,
    TaskTopologyTraversalLimitExceeded,
    _allocate_task_topology_branch_limits,
    decode_task_topology_cursor,
    encode_task_topology_cursor,
)


def _store(store_factory, tmp_path: Path) -> TaskStore:
    if store_factory is SQLiteTaskStore:
        return SQLiteTaskStore(tmp_path / "task-topology.sqlite")
    return store_factory()


async def _close(store: TaskStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_topology_store_conformance(store_factory, tmp_path: Path) -> None:
    async def run() -> None:
        store = _store(store_factory, tmp_path)
        try:
            await assert_task_topology_store_conformance(store)
        finally:
            await _close(store)

    asyncio.run(run())


def test_task_topology_query_rejects_duplicate_or_misauthorized_cursors() -> None:
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        TaskTopologyQuery(linked_session_ids=("session-a", "session-a"))

    with pytest.raises(
        ValidationError,
        match="session_cursors keys must also appear",
    ):
        TaskTopologyQuery(session_cursors={"session-a": "cursor"})

    with pytest.raises(
        ValidationError,
        match="child_cursors keys must also appear",
    ):
        TaskTopologyQuery(child_cursors={"task-a": "cursor"})

    with pytest.raises(ValidationError):
        TaskTopologyQuery(
            expanded_parent_ids=tuple(
                f"task-{index}" for index in range(TASK_TOPOLOGY_MAX_EXPANDED_PARENTS + 1)
            )
        )


def test_task_topology_allocates_the_global_cap_before_candidate_hydration() -> None:
    query = TaskTopologyQuery(
        linked_session_ids=tuple(f"session-{index}" for index in range(50)),
        expanded_parent_ids=tuple(f"parent-{index}" for index in range(50)),
        session_task_limit=100,
        child_limit=100,
    )

    session_limits, child_limits = _allocate_task_topology_branch_limits(query)

    assert len(session_limits) == 50
    assert len(child_limits) == 50
    assert all(limit >= 1 for limit in (*session_limits, *child_limits))
    assert sum(session_limits) + sum(child_limits) + len(query.expanded_parent_ids) == (
        TASK_TOPOLOGY_MAX_NODES
    )
    # One sentinel per branch proves has_more without letting the stores hydrate
    # the original 100 * 101-row cross product.
    assert (
        sum(limit + 1 for limit in (*session_limits, *child_limits))
        + len(query.expanded_parent_ids)
        == 600
    )


def test_task_topology_cursor_is_canonical_and_scope_bound() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        task = await store.create_task(
            TaskCreate(task_id="task-a", type="step", session_id="session-a")
        )
        result = await store.query_task_topology(
            TaskTopologyQuery(linked_session_ids=("session-a",))
        )
        node = result.session_branches[0].tasks[0]
        cursor = encode_task_topology_cursor("session", "session-a", node)

        assert decode_task_topology_cursor(
            cursor,
            scope_kind="session",
            scope_id="session-a",
        ) == (task.created_at, task.id)
        with pytest.raises(ValueError, match="Invalid task topology cursor"):
            decode_task_topology_cursor(
                cursor,
                scope_kind="session",
                scope_id="session-b",
            )
        with pytest.raises(ValueError, match="Invalid task topology cursor"):
            decode_task_topology_cursor(
                cursor.rstrip("="),
                scope_kind="session",
                scope_id="session-a",
            )

    asyncio.run(run())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_topology_detects_loaded_cycles(store_factory, tmp_path: Path) -> None:
    async def run() -> None:
        store = _store(store_factory, tmp_path)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="task-loop",
                    type="step",
                    session_id="session-a",
                    parent_task_id="task-loop",
                )
            )
            with pytest.raises(TaskTopologyCycle):
                await store.query_task_topology(
                    TaskTopologyQuery(
                        linked_session_ids=("session-a",),
                        expanded_parent_ids=("task-loop",),
                    )
                )
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_topology_detects_cycles_split_across_pages(store_factory, tmp_path: Path) -> None:
    async def run() -> None:
        store = _store(store_factory, tmp_path)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="task-a",
                    type="step",
                    session_id="session-a",
                    parent_task_id="task-b",
                )
            )
            await store.create_task(
                TaskCreate(
                    task_id="task-b",
                    type="step",
                    session_id="session-a",
                    parent_task_id="task-a",
                )
            )
            with pytest.raises(TaskTopologyCycle):
                await store.query_task_topology(
                    TaskTopologyQuery(
                        linked_session_ids=("session-a",),
                        session_task_limit=1,
                    )
                )
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_topology_fails_clearly_when_ancestry_exceeds_its_bound(
    store_factory,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = _store(store_factory, tmp_path)
        try:
            parent_id: str | None = None
            for index in range(TASK_TOPOLOGY_MAX_ANCESTOR_DEPTH + 2):
                task_id = f"task-depth-{index:03d}"
                await store.create_task(
                    TaskCreate(
                        task_id=task_id,
                        type="step",
                        session_id=(
                            "session-depth"
                            if index == TASK_TOPOLOGY_MAX_ANCESTOR_DEPTH + 1
                            else None
                        ),
                        parent_task_id=parent_id,
                    )
                )
                parent_id = task_id
            with pytest.raises(TaskTopologyTraversalLimitExceeded):
                await store.query_task_topology(
                    TaskTopologyQuery(linked_session_ids=("session-depth",))
                )
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_topology_rejects_orphaned_parent_links(store_factory, tmp_path: Path) -> None:
    async def run() -> None:
        store = _store(store_factory, tmp_path)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="orphan",
                    type="step",
                    session_id="session-a",
                    parent_task_id="missing-parent",
                )
            )
            with pytest.raises(TaskTopologyInconsistent, match="missing durable parent"):
                await store.query_task_topology(
                    TaskTopologyQuery(linked_session_ids=("session-a",))
                )
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_task_topology_omits_oversized_display_text_but_rejects_oversized_identity(
    store_factory,
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = _store(store_factory, tmp_path)
        try:
            await assert_task_topology_bounded_projection_conformance(store)
        finally:
            await _close(store)

    asyncio.run(run())


def test_task_topology_global_node_cap_keeps_every_branch_pageable() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        parent_ids = tuple(f"parent-{index}" for index in range(5))
        for parent_id in parent_ids:
            await store.create_task(TaskCreate(task_id=parent_id, type="workflow"))
            for child_index in range(101):
                await store.create_task(
                    TaskCreate(
                        task_id=f"{parent_id}-child-{child_index:03d}",
                        type="step",
                        parent_task_id=parent_id,
                    )
                )

        result = await store.query_task_topology(
            TaskTopologyQuery(
                expanded_parent_ids=parent_ids,
                child_limit=100,
            )
        )
        unique_ids = {
            *(task.id for task in result.expanded_parents),
            *(task.id for branch in result.child_branches for task in branch.children),
        }
        assert len(unique_ids) == TASK_TOPOLOGY_MAX_NODES
        assert all(branch.children for branch in result.child_branches)
        assert all(branch.has_more for branch in result.child_branches)
        assert all(branch.next_cursor is not None for branch in result.child_branches)

        last_branch = result.child_branches[-1]
        continued = await store.query_task_topology(
            TaskTopologyQuery(
                expanded_parent_ids=(last_branch.parent_task_id,),
                child_cursors={
                    last_branch.parent_task_id: last_branch.next_cursor,
                },
                child_limit=100,
            )
        )
        assert continued.child_branches[0].children
        assert continued.child_branches[0].children[0].id not in unique_ids

    asyncio.run(run())


def test_in_memory_task_topology_uses_secondary_indexes_without_registry_scan() -> None:
    class NoValuesDict(dict):
        def values(self):
            raise AssertionError("Task topology scanned the task registry.")

    class BoundedKeyList(list):
        def __iter__(self):
            raise AssertionError("Task topology iterated a complete branch index.")

        def __getitem__(self, key):
            if isinstance(key, slice):
                start = 0 if key.start is None else key.start
                stop = len(self) if key.stop is None else key.stop
                assert stop - start <= 26
            return super().__getitem__(key)

    async def run() -> None:
        store = InMemoryTaskStore()
        await store.create_task(
            TaskCreate(
                task_id="parent",
                type="workflow",
                session_id="session-a",
            )
        )
        await store.create_task(
            TaskCreate(
                task_id="child",
                type="step",
                session_id="session-a",
                parent_task_id="parent",
            )
        )
        store._tasks = NoValuesDict(store._tasks)
        store._task_keys_by_session["session-a"] = BoundedKeyList(
            store._task_keys_by_session["session-a"]
        )
        await store.block_task("child", reason="Waiting")

        result = await store.query_task_topology(
            TaskTopologyQuery(
                linked_session_ids=("session-a",),
                expanded_parent_ids=("parent",),
            )
        )
        assert [task.id for task in result.session_branches[0].tasks] == [
            "parent",
            "child",
        ]
        assert [task.id for task in result.child_branches[0].children] == ["child"]
        assert result.child_branches[0].children[0].status_reason == "Waiting"

    asyncio.run(run())


def test_sqlite_task_topology_uses_composite_branch_indexes(tmp_path: Path) -> None:
    async def run() -> int:
        store = SQLiteTaskStore(tmp_path / "task-plan.sqlite")
        try:
            await store.create_task(
                TaskCreate(
                    task_id="task-parent",
                    type="workflow",
                    session_id="session-a",
                )
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
                    INSERT INTO cayu_tasks (
                        id, type, status, session_id, parent_task_id,
                        input_json, metadata_json, created_at, updated_at
                    )
                    SELECT
                        printf('task-child-%06d', value), 'step', 'pending',
                        'session-a', 'task-parent', '{}', '{}', ?, ?
                    FROM numbers
                    """,
                    (timestamp, timestamp),
                )

            session_plan = store._connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT id
                FROM cayu_tasks
                WHERE session_id = ?
                ORDER BY created_at ASC, id ASC
                LIMIT 26
                """,
                ("session-a",),
            ).fetchall()
            parent_plan = store._connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT id
                FROM cayu_tasks
                WHERE parent_task_id = ?
                ORDER BY created_at ASC, id ASC
                LIMIT 26
                """,
                ("task-parent",),
            ).fetchall()
            assert "idx_cayu_tasks_session_created_id" in " ".join(
                str(row["detail"]) for row in session_plan
            )
            assert "idx_cayu_tasks_parent_created_id" in " ".join(
                str(row["detail"]) for row in parent_plan
            )

            progress_calls = 0

            def count_progress() -> int:
                nonlocal progress_calls
                progress_calls += 1
                return 0

            store._connection.set_progress_handler(count_progress, 1)
            try:
                result = await store.query_task_topology(
                    TaskTopologyQuery(
                        linked_session_ids=("session-a",),
                        expanded_parent_ids=("task-parent",),
                        session_task_limit=25,
                        child_limit=25,
                    )
                )
            finally:
                store._connection.set_progress_handler(None, 0)
            assert len(result.session_branches[0].tasks) == 25
            assert result.session_branches[0].has_more is True
            assert len(result.child_branches[0].children) == 25
            assert result.child_branches[0].has_more is True
            return progress_calls
        finally:
            await store.close()

    progress_calls = asyncio.run(run())
    assert progress_calls < 10_000
