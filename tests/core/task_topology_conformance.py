from __future__ import annotations

import pytest

from cayu import TaskCreate, TaskStore
from cayu.runtime.tasks import (
    TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES,
    TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    TaskTopologyCycle,
    TaskTopologyInconsistent,
    TaskTopologyQuery,
    decode_task_topology_cursor,
)


async def assert_task_topology_store_conformance(store: TaskStore) -> None:
    parent = await store.create_task(
        TaskCreate(
            task_id="topology-parent",
            type="workflow",
            title="Parent",
            session_id="session-a",
            assigned_agent_name="coordinator",
            input={"secret": "must-not-project"},
            metadata={"private": True},
        )
    )
    first_child = await store.create_task(
        TaskCreate(
            task_id="topology-child-a",
            type="step",
            title="First child",
            session_id="session-a",
            parent_task_id=parent.id,
            assigned_agent_name="worker",
            input={"payload": "not topology"},
        )
    )
    second_child = await store.create_task(
        TaskCreate(
            task_id="topology-child-b",
            type="step",
            title="Second child",
            session_id="session-b",
            parent_task_id=parent.id,
        )
    )
    await store.create_task(
        TaskCreate(
            task_id="topology-unrelated",
            type="step",
            session_id="session-unrelated",
            parent_task_id="unrelated-parent",
        )
    )

    query = TaskTopologyQuery(
        linked_session_ids=("session-a", "session-b"),
        expanded_parent_ids=(parent.id,),
        session_task_limit=1,
        child_limit=1,
    )
    first = await store.query_task_topology(query)
    repeated = await store.query_task_topology(query)

    assert first.model_dump(exclude={"observed_at"}) == repeated.model_dump(exclude={"observed_at"})
    assert [branch.session_id for branch in first.session_branches] == [
        "session-a",
        "session-b",
    ]
    assert [task.id for task in first.session_branches[0].tasks] == [parent.id]
    assert first.session_branches[0].has_more is True
    assert first.session_branches[0].next_cursor is not None
    assert [task.id for task in first.session_branches[1].tasks] == [second_child.id]
    assert first.session_branches[1].has_more is False
    assert [task.id for task in first.expanded_parents] == [parent.id]
    assert [task.id for task in first.child_branches[0].children] == [first_child.id]
    assert first.child_branches[0].has_more is True
    assert first.child_branches[0].next_cursor is not None

    cursor_created_at, cursor_task_id = decode_task_topology_cursor(
        first.child_branches[0].next_cursor,
        scope_kind="parent_task",
        scope_id=parent.id,
    )
    assert cursor_created_at == first_child.created_at
    assert cursor_task_id == first_child.id

    inserted_after_first_page = await store.create_task(
        TaskCreate(
            task_id="topology-child-c",
            type="step",
            title="Inserted after first page",
            session_id="session-a",
            parent_task_id=parent.id,
        )
    )
    continuation = await store.query_task_topology(
        TaskTopologyQuery(
            linked_session_ids=("session-a",),
            session_cursors={
                "session-a": first.session_branches[0].next_cursor,
            },
            expanded_parent_ids=(parent.id,),
            child_cursors={
                parent.id: first.child_branches[0].next_cursor,
            },
            session_task_limit=1,
            child_limit=1,
        )
    )
    assert [task.id for task in continuation.session_branches[0].tasks] == [first_child.id]
    assert [task.id for task in continuation.child_branches[0].children] == [second_child.id]
    assert continuation.child_branches[0].has_more is True
    assert continuation.child_branches[0].next_cursor is not None

    final_page = await store.query_task_topology(
        TaskTopologyQuery(
            expanded_parent_ids=(parent.id,),
            child_cursors={
                parent.id: continuation.child_branches[0].next_cursor,
            },
            child_limit=1,
        )
    )
    assert [task.id for task in final_page.child_branches[0].children] == [
        inserted_after_first_page.id
    ]
    assert final_page.child_branches[0].has_more is False

    projected = first.expanded_parents[0]
    assert set(projected.model_fields_set) == {
        "id",
        "type",
        "title",
        "status",
        "status_reason",
        "session_id",
        "parent_task_id",
        "assigned_agent_name",
        "created_at",
        "updated_at",
        "truncated_fields",
    }

    attachable = await store.create_task(TaskCreate(task_id="topology-attach", type="step"))
    await store.start_task(attachable.id, session_id="session-c")
    attached = await store.query_task_topology(TaskTopologyQuery(linked_session_ids=("session-c",)))
    assert [task.id for task in attached.session_branches[0].tasks] == [attachable.id]

    await store.create_task(
        TaskCreate(
            task_id="topology-orphan",
            type="step",
            session_id="session-orphan",
            parent_task_id="topology-missing-parent",
        )
    )
    with pytest.raises(TaskTopologyInconsistent, match="missing durable parent"):
        await store.query_task_topology(TaskTopologyQuery(linked_session_ids=("session-orphan",)))

    await store.create_task(
        TaskCreate(
            task_id="topology-cycle-a",
            type="step",
            session_id="session-cycle",
            parent_task_id="topology-cycle-b",
        )
    )
    await store.create_task(
        TaskCreate(
            task_id="topology-cycle-b",
            type="step",
            session_id="session-cycle",
            parent_task_id="topology-cycle-a",
        )
    )
    with pytest.raises(TaskTopologyCycle):
        await store.query_task_topology(
            TaskTopologyQuery(
                linked_session_ids=("session-cycle",),
                session_task_limit=1,
            )
        )


async def assert_task_topology_bounded_projection_conformance(store: TaskStore) -> None:
    display = "x" * (TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES + 1)
    task = await store.create_task(
        TaskCreate(
            task_id="task-display",
            type=display,
            title=display,
            session_id="session-display",
            assigned_agent_name=display,
        )
    )
    await store.block_task(task.id, reason=display)
    result = await store.query_task_topology(
        TaskTopologyQuery(linked_session_ids=("session-display",))
    )
    node = result.session_branches[0].tasks[0]
    assert node.type is None
    assert node.title is None
    assert node.assigned_agent_name is None
    assert node.status_reason is None
    assert node.truncated_fields == (
        "type",
        "title",
        "assigned_agent_name",
        "status_reason",
    )

    await store.create_task(
        TaskCreate(
            task_id="i" * (TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES + 1),
            type="step",
            session_id="session-identity",
        )
    )
    with pytest.raises(TaskTopologyInconsistent):
        await store.query_task_topology(TaskTopologyQuery(linked_session_ids=("session-identity",)))
