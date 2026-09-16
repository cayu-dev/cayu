"""Graph non-execution outcomes retain atomic task-owned scheduling evidence."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from cayu import CayuApp
from cayu.storage import postgres, sqlite
from cayu.storage.migrations import SchemaMode
from cayu.tasks import base
from cayu.tasks.base import InMemoryTaskStore, TaskCreate, TaskQuery, TaskStatus
from cayu.tasks.graphs import TaskGraphCreate, TaskGraphEventType, TaskGraphNode
from cayu.tasks.scheduling import TaskMisfirePolicy, TaskScheduleEventType, TaskSchedulePolicy


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store_factory(request, tmp_path):
    backend = request.param
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None
    now = datetime.now(UTC)

    def open_store():
        if backend == "memory":
            return InMemoryTaskStore(clock=lambda: now, ownership_clock=lambda: now)
        if backend == "sqlite":
            return sqlite.SQLiteTaskStore(
                tmp_path / "graph-schedules.sqlite",
                clock=lambda: now,
                ownership_clock=lambda: now,
            )
        return postgres.PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE, clock=lambda: now)

    return open_store, now


def graph_request(available_at: datetime) -> TaskGraphCreate:
    identity = uuid4().hex
    root, child, tail = (f"{identity}-{name}" for name in ("root", "a", "b"))
    return TaskGraphCreate(
        graph_id=identity,
        nodes=(
            TaskGraphNode(task=TaskCreate(task_id=root, type=identity)),
            *(
                TaskGraphNode(
                    task=TaskCreate(
                        task_id=task_id,
                        type=identity,
                        available_at=available_at,
                        schedule_policy=TaskSchedulePolicy(
                            misfire_policy=TaskMisfirePolicy.SKIP, misfire_grace_seconds=0
                        ),
                    ),
                    prerequisite_task_ids=(prerequisite,),
                )
                for task_id, prerequisite in ((child, root), (tail, child))
            ),
        ),
    )


async def close_store(store):
    if not isinstance(store, InMemoryTaskStore):
        await store.close()


async def finish_root(store, request, outcome):
    root = f"{request.graph_id}-root"
    if outcome == "failed":
        return await store.fail_task(root, {"code": "prerequisite_failed"})
    return await store.cancel_task(root, {"code": "prerequisite_cancelled"})


async def observe(store, api, request):
    """Compare live tasks, graph projection and both independent journals."""
    return (
        [await store.load_task(node.task.task_id) for node in request.nodes],
        await api.load_task_graph(request.graph_id),
        await api.list_task_graph_events(request.graph_id),
        {
            node.task.task_id: await api.list_task_schedule_events(node.task.task_id)
            for node in request.nodes
        },
    )


async def assert_skipped_history(store, api, request, initial, outcome):
    state = await observe(store, api, request)
    tasks, graph, graph_events, histories = state
    initial_tasks = {task.id: task for task in initial[0]}
    root = f"{request.graph_id}-root"
    skipped_ids = {task.id for task in tasks if task.id != root}
    assert {member.task_id: member.status for member in graph.members} == {
        root: TaskStatus(outcome),
        **dict.fromkeys(skipped_ids, TaskStatus.DEPENDENCY_SKIPPED),
    }
    assert histories[root] == []  # An unmanaged member gets no schedule journal.
    skips = [event for event in graph_events if event.type is TaskGraphEventType.SKIPPED]
    assert len(skips) == len(skipped_ids)
    assert {event.task_id for event in skips} == skipped_ids
    assert [event.sequence for event in graph_events] == list(range(1, len(graph_events) + 1))
    assert graph.last_sequence == len(graph_events)
    for task in tasks:
        if task.id == root:
            assert task.status is TaskStatus(outcome)
            continue
        assert task.status is TaskStatus.DEPENDENCY_SKIPPED
        assert task.status_reason == "dependency_failed"
        assert task.started_at is None and task.worker_id is None and task.session_id is None
        assert task.schedule == initial_tasks[task.id].schedule
        assert task.schedule.admitted_at is None
        events = histories[task.id]
        assert [event.type for event in events] == [
            TaskScheduleEventType.SCHEDULED,
            TaskScheduleEventType.DEPENDENCY_SKIPPED,
        ]
        assert events[-1].type.value == "task.schedule_dependency_skipped"
        assert [event.sequence for event in events] == [1, 2]
        terminal = events[-1]
        assert terminal.revision == task.schedule.revision
        assert terminal.available_at == task.available_at
        assert terminal.policy == task.schedule.policy
        assert terminal.invocation_id == task.invocation.root_invocation_id
        assert terminal.operation_id is None
        assert terminal.occurred_at == task.updated_at == task.completed_at
        graph_event = next(event for event in skips if event.task_id == task.id)
        assert graph_event.status is task.status
        assert graph_event.occurred_at == terminal.occurred_at
        assert graph_event.prerequisite_task_ids == task.prerequisite_task_ids
        assert await api.list_task_schedule_events(task.id, after_sequence=1, limit=1) == [terminal]
        assert await api.list_task_schedule_events(task.id, after_sequence=2) == []
    return state


@pytest.mark.parametrize("entrance", ["sdk", "native"])
@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
@pytest.mark.parametrize("due", ["future", "overdue"])
def test_dependency_skip_history_survives_replay_and_reopen(store_factory, entrance, outcome, due):
    open_store, now = store_factory
    request = graph_request(now + timedelta(days=1 if due == "future" else -1))

    async def run():
        store = open_store()
        try:
            api = CayuApp(task_store=store, enable_logging=False) if entrance == "sdk" else store
            receipt = await api.create_task_graph(request)
            initial = await observe(store, api, request)
            await finish_root(store, request, outcome)
            terminal = await assert_skipped_history(store, api, request, initial, outcome)
            assert await api.create_task_graph(request) == receipt
            assert await store.claim_task("worker", TaskQuery(type=request.graph_id)) is None
            assert await observe(store, api, request) == terminal
            if not isinstance(store, InMemoryTaskStore):
                await store.close()
                store = open_store()
                api = (
                    CayuApp(task_store=store, enable_logging=False) if entrance == "sdk" else store
                )
            assert await api.create_task_graph(request) == receipt
            assert await store.claim_task("replacement", TaskQuery(type=request.graph_id)) is None
            assert await observe(store, api, request) == terminal
        finally:
            await close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_dependency_skip_schedule_publication_failure_rolls_back(
    store_factory, outcome, monkeypatch
):
    open_store, now = store_factory
    request = graph_request(now + timedelta(days=1))

    async def run():
        store = open_store()
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            receipt = await app.create_task_graph(request)
            initial = await observe(store, app, request)
            module = (
                base
                if isinstance(store, InMemoryTaskStore)
                else sqlite
                if isinstance(store, sqlite.SQLiteTaskStore)
                else postgres
            )
            original = module.schedule_transition_events
            prepared = []

            def fail_later(prior, current, **kwargs):
                events = original(prior, current, **kwargs)
                if current.status is TaskStatus.DEPENDENCY_SKIPPED:
                    assert [event.type for event in events] == [
                        TaskScheduleEventType.DEPENDENCY_SKIPPED
                    ]
                    prepared.append(current.id)
                    if current.id == f"{request.graph_id}-b":
                        raise RuntimeError("later dependency schedule publication failed")
                return events

            with monkeypatch.context() as patch:
                patch.setattr(module, "schedule_transition_events", fail_later)
                with pytest.raises(RuntimeError, match="later dependency schedule publication"):
                    await finish_root(store, request, outcome)
            assert prepared == [f"{request.graph_id}-a", f"{request.graph_id}-b"]
            assert await observe(store, app, request) == initial
            if not isinstance(store, InMemoryTaskStore):
                await store.close()
                store = open_store()
                app = CayuApp(task_store=store, enable_logging=False)
            assert await observe(store, app, request) == initial
            assert await app.create_task_graph(request) == receipt
            await finish_root(store, request, outcome)
            terminal = await assert_skipped_history(store, app, request, initial, outcome)
            assert await app.create_task_graph(request) == receipt
            assert await observe(store, app, request) == terminal
        finally:
            await close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "held_status", [TaskStatus.PAUSED, TaskStatus.BLOCKED, TaskStatus.NEEDS_ATTENTION]
)
def test_scheduled_dependent_resume_history_survives_replay_and_reopen(store_factory, held_status):
    open_store, now = store_factory
    identity = uuid4().hex
    root, child = f"{identity}-root", f"{identity}-child"
    request = TaskGraphCreate(
        graph_id=identity,
        nodes=(
            TaskGraphNode(task=TaskCreate(task_id=root, type=root)),
            TaskGraphNode(
                task=TaskCreate(
                    task_id=child,
                    type=child,
                    available_at=now,
                    schedule_policy=TaskSchedulePolicy(),
                ),
                prerequisite_task_ids=(root,),
            ),
        ),
    )

    async def run():
        store = open_store()
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            receipt = await app.create_task_graph(request)
            hold = {
                TaskStatus.PAUSED: app.pause_task,
                TaskStatus.BLOCKED: app.block_task,
                TaskStatus.NEEDS_ATTENTION: app.mark_task_needs_attention,
            }[held_status]
            held = await hold(child, reason="operator")
            assert held.status is held_status
            resumed = await app.resume_task(child)
            assert resumed.status is TaskStatus.WAITING_DEPENDENCIES
            assert resumed.status_reason is None
            assert resumed.worker_id is None and resumed.lease_expires_at is None
            assert resumed.schedule == held.schedule
            assert resumed.schedule.admitted_at is None
            history = await app.list_task_schedule_events(child)
            assert [event.type for event in history] == [
                TaskScheduleEventType.SCHEDULED,
                TaskScheduleEventType.HELD,
                TaskScheduleEventType.RESUMED,
            ]
            assert [event.sequence for event in history] == [1, 2, 3]
            event = history[-1]
            assert event.occurred_at == resumed.updated_at
            assert event.revision == resumed.schedule.revision
            assert event.available_at == resumed.available_at
            assert event.policy == resumed.schedule.policy
            assert event.invocation_id == resumed.invocation.root_invocation_id
            assert event.operation_id is None
            assert await store.claim_task("worker", TaskQuery(type=child)) is None
            before = await observe(store, app, request)
            assert await app.create_task_graph(request) == receipt
            assert await observe(store, app, request) == before
            # Repeated resume is not an idempotent operation; rejection must not
            # append a second history event or bypass the dependency gate.
            with pytest.raises(ValueError):
                await app.resume_task(child)
            assert await observe(store, app, request) == before
            if not isinstance(store, InMemoryTaskStore):
                await store.close()
                store = open_store()
                app = CayuApp(task_store=store, enable_logging=False)
            assert await app.create_task_graph(request) == receipt
            assert await store.claim_task("replacement", TaskQuery(type=child)) is None
            assert await observe(store, app, request) == before
            # Readiness still becomes executable normally after exact success.
            await store.complete_task(root, {})
            claimed = await store.claim_task("replacement", TaskQuery(type=child))
            assert claimed is not None and claimed.id == child
        finally:
            await close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "held_status", [TaskStatus.PAUSED, TaskStatus.BLOCKED, TaskStatus.NEEDS_ATTENTION]
)
@pytest.mark.parametrize(
    "scenario", ["held-before-success", "held-after-success", "publication-failure"]
)
def test_held_graph_member_publishes_readiness_once(
    store_factory, held_status, scenario, monkeypatch
):
    from cayu.storage import _postgres_task_graphs, _sqlite_task_graphs
    from cayu.tasks import _graphs

    open_store, now = store_factory
    identity = uuid4().hex
    root, child = f"{identity}-root", f"{identity}-child"
    request = TaskGraphCreate(
        graph_id=identity,
        nodes=(
            TaskGraphNode(task=TaskCreate(task_id=root, type=root)),
            TaskGraphNode(
                task=TaskCreate(
                    task_id=child,
                    type=child,
                    available_at=now,
                    schedule_policy=TaskSchedulePolicy(),
                ),
                prerequisite_task_ids=(root,),
            ),
        ),
    )

    async def run():
        store = open_store()
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            receipt = await app.create_task_graph(request)
            if scenario == "held-after-success":
                await store.complete_task(root, {})

            async def hold():
                return await {
                    TaskStatus.PAUSED: app.pause_task,
                    TaskStatus.BLOCKED: app.block_task,
                    TaskStatus.NEEDS_ATTENTION: app.mark_task_needs_attention,
                }[held_status](child, reason="operator")

            await hold()
            if scenario != "held-after-success":
                await store.complete_task(root, {})
            assert (await store.load_task(child)).status is held_status
            assert await store.claim_task("early", TaskQuery(type=child)) is None
            held = await observe(store, app, request)
            initial_ready = [
                event
                for event in held[2]
                if event.task_id == child and event.type is TaskGraphEventType.READY
            ]
            assert len(initial_ready) == (1 if scenario == "held-after-success" else 0)

            if scenario == "publication-failure":
                observed = []

                def reject(events):
                    for event in events:
                        if event.task_id == child and event.type is TaskGraphEventType.READY:
                            observed.append(event)
                            raise RuntimeError("readiness publication failed")

                with monkeypatch.context() as patch:
                    if isinstance(store, InMemoryTaskStore):
                        original = _graphs.TaskGraphEvent

                        def fail_preparation(**kwargs):
                            event = original(**kwargs)
                            reject((event,))
                            return event

                        patch.setattr(_graphs, "TaskGraphEvent", fail_preparation)
                    elif isinstance(store, sqlite.SQLiteTaskStore):
                        original = _sqlite_task_graphs.insert_events

                        def fail_sqlite(owner, events):
                            original(owner, events)
                            reject(events)

                        patch.setattr(_sqlite_task_graphs, "insert_events", fail_sqlite)
                    else:
                        original = _postgres_task_graphs.insert_events

                        async def fail_postgres(cur, events):
                            await original(cur, events)
                            reject(events)

                        patch.setattr(_postgres_task_graphs, "insert_events", fail_postgres)
                    with pytest.raises(RuntimeError, match="readiness publication failed"):
                        await app.resume_task(child)
                assert len(observed) == 1
                assert await observe(store, app, request) == held
                if not isinstance(store, InMemoryTaskStore):
                    await store.close()
                    store = open_store()
                    app = CayuApp(task_store=store, enable_logging=False)
                assert await observe(store, app, request) == held

            resumed = await app.resume_task(child)
            assert resumed.status is TaskStatus.PENDING
            events = await app.list_task_graph_events(identity)
            readiness = [
                event
                for event in events
                if event.task_id == child and event.type is TaskGraphEventType.READY
            ]
            assert len(readiness) == 1
            assert readiness[0].status is TaskStatus.PENDING
            assert readiness[0].prerequisite_task_ids == (root,)
            if initial_ready:
                assert readiness == initial_ready
            else:
                assert readiness[0].sequence == held[1].last_sequence + 1
            assert [event.sequence for event in events] == list(range(1, len(events) + 1))
            state = await observe(store, app, request)
            assert await app.create_task_graph(request) == receipt
            with pytest.raises(ValueError):
                await app.resume_task(child)
            assert await observe(store, app, request) == state
            if not isinstance(store, InMemoryTaskStore):
                await store.close()
                store = open_store()
                app = CayuApp(task_store=store, enable_logging=False)
            assert await observe(store, app, request) == state
            assert await app.create_task_graph(request) == receipt
            await hold()
            assert await store.claim_task("held", TaskQuery(type=child)) is None
            await app.resume_task(child)
            assert await app.list_task_graph_events(identity) == events
            claimed = await store.claim_task("worker", TaskQuery(type=child))
            assert claimed is not None and claimed.id == child
            assert await store.claim_task("competitor", TaskQuery(type=child)) is None
        finally:
            await close_store(store)

    asyncio.run(run())
