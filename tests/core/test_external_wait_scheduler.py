"""Managed one-shot hints compose with, but never replace, wait-store election."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu import CayuApp
from cayu.external_wait_scheduler import EXTERNAL_WAIT_TASK_TYPE, TaskStoreWaitScheduler
from cayu.external_waits import ExternalEventWaits
from cayu.sessions.external_waits import ExternalEventDelivery, ExternalWaitConflict
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("failure", ["none", "lost_ack", "cancel"])
def test_native_timer_reconstruction_and_exact_publication(
    backend, failure, tmp_path, request, monkeypatch
):
    async def scenario():
        clock = [datetime.now(UTC)]
        async with stores(backend, tmp_path, request, clock) as (store, reopen):
            tasks = []
            memory = InMemoryTaskStore(clock=lambda: clock[0], ownership_clock=lambda: clock[0])

            def open_tasks():
                if backend == "memory":
                    return memory
                if backend == "sqlite":
                    value = SQLiteTaskStore(
                        tmp_path / "tasks.sqlite",
                        clock=lambda: clock[0],
                        ownership_clock=lambda: clock[0],
                    )
                else:
                    value = PostgresTaskStore(
                        request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
                    )
                tasks.append(value)
                return value

            task_store = open_tasks()
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(
                reservation(deadline=clock[0] - timedelta(seconds=1)), context=CONTEXT
            )
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            scheduler = TaskStoreWaitScheduler(
                waits=waits, task_store=task_store, scheduler_id="host-scheduler"
            )
            observe = waits._observe_operation
            committed = asyncio.Event()

            async def lose_creation_ack(operation, *, key, expectation):
                result = await observe(operation, key=key, expectation=expectation)
                if key[0] == "external-timer":
                    if failure == "cancel":
                        committed.set()
                        await asyncio.Event().wait()
                    raise RuntimeError("task creation acknowledgement lost")
                return result

            try:
                if failure != "none":
                    with monkeypatch.context() as patch:
                        patch.setattr(waits, "_observe_operation", lose_creation_ack)
                        if failure == "cancel":
                            scheduling = asyncio.create_task(
                                scheduler.schedule(registered, context=CONTEXT)
                            )
                            await asyncio.wait_for(committed.wait(), 10)
                            scheduling.cancel()
                            with pytest.raises(asyncio.CancelledError):
                                await scheduling
                            assert scheduling.cancelled() and scheduling.cancelling() == 1
                        else:
                            with pytest.raises(RuntimeError, match="acknowledgement lost"):
                                await scheduler.schedule(registered, context=CONTEXT)
                    pending = await waits.inspect(correlation, context=CONTEXT)
                    assert pending.pending_timer and not pending.timer_published
                else:
                    await scheduler.schedule(registered, context=CONTEXT)
                durable = await store._read_external_wait(
                    correlation.request.scope, correlation.request.correlation_key
                )
                assert durable.timer is not None
                before = await task_store.load_task(durable.timer.task_id)
                assert before is not None
                reopened_tasks = open_tasks()
                restored_waits = ExternalEventWaits(store=reopen(), access_policy=Policy())
                restored = TaskStoreWaitScheduler(
                    waits=restored_waits, task_store=reopened_tasks, scheduler_id="host-scheduler"
                )
                repaired = await restored.reconcile(
                    scope=correlation.request.scope, source="renderer", context=CONTEXT
                )
                assert len(repaired) == int(failure != "none")
                timer = await restored.schedule(registered, context=CONTEXT)
                assert timer == durable.timer
                assert await reopened_tasks.load_task(timer.task_id) == before
                assert (await restored_waits.inspect(correlation, context=CONTEXT)).timer_published
                with pytest.raises(ExternalWaitConflict):
                    await TaskStoreWaitScheduler(
                        waits=restored_waits, task_store=reopened_tasks, scheduler_id="different"
                    ).schedule(registered, context=CONTEXT)
                app = CayuApp(task_store=reopened_tasks, enable_logging=False)
                task = await reopened_tasks.claim_task(
                    "timer-worker", TaskQuery(type=EXTERNAL_WAIT_TASK_TYPE)
                )
                assert task is not None and task.id == timer.task_id
                await restored.worker_handler(context=CONTEXT)(app, task, "timer-worker")
                final = await reopened_tasks.load_task(timer.task_id)
                assert (
                    final.status.value == "completed" and final.result["outcome_kind"] == "timeout"
                )
                assert (
                    await reopened_tasks.claim_task(
                        "timer-worker", TaskQuery(type=EXTERNAL_WAIT_TASK_TYPE)
                    )
                    is None
                )
                assert await restored.schedule(registered, context=CONTEXT) == timer
                assert await reopened_tasks.load_task(timer.task_id) == final
                await restored_waits.aclose()
                await waits.aclose()
            finally:
                for task_owner in tasks:
                    await task_owner.close()

    asyncio.run(scenario())


def test_event_only_wait_needs_no_timer_and_early_hint_does_not_elect():
    from cayu.sessions.base import InMemorySessionStore

    async def scenario():
        now = datetime.now(UTC)
        store = InMemorySessionStore(ownership_clock=lambda: now)
        waits = ExternalEventWaits(store=store, access_policy=Policy())
        scheduler = TaskStoreWaitScheduler(
            waits=waits, task_store=InMemoryTaskStore(), scheduler_id="timer"
        )
        event_only = await waits.reserve_correlation(reservation(), context=CONTEXT)
        registered = registration(event_only)
        await waits.register(registered, context=CONTEXT)
        assert await scheduler.schedule(registered, context=CONTEXT) is None
        timed = await waits.reserve_correlation(
            reservation(deadline=now + timedelta(days=1)), context=CONTEXT
        )
        registered = registration(timed)
        await waits.register(registered, context=CONTEXT)
        timer = await scheduler.schedule(registered, context=CONTEXT)
        assert (await scheduler.notify(timer, context=CONTEXT)).outcome is None
        await waits.deliver(
            ExternalEventDelivery(
                correlation=timed, delivery_id="event", payload_json='{"done":true}'
            ),
            context=CONTEXT,
        )
        assert (await scheduler.notify(timer, context=CONTEXT)).outcome.kind == "event"
        await waits.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("retirement_phase", ["retained", "pruned", "during_observation"])
def test_worker_completes_retired_timer_after_reconstruction(
    backend, retirement_phase, tmp_path, request, monkeypatch
):
    from tests.core.test_external_wait_retirement import AdministrationPolicy

    from cayu.tasks.worker import run_task_worker

    async def scenario():
        clock = [datetime.now(UTC)]
        async with stores(backend, tmp_path, request, clock) as (store, reopen):
            task_owners = []
            memory = InMemoryTaskStore()

            def open_tasks():
                if backend == "memory":
                    return memory
                if backend == "sqlite":
                    tasks = SQLiteTaskStore(tmp_path / "timer-tasks.sqlite")
                else:
                    tasks = PostgresTaskStore(
                        request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
                    )
                task_owners.append(tasks)
                return tasks

            policy = AdministrationPolicy()
            waits = ExternalEventWaits(store=store, access_policy=policy)
            tasks = open_tasks()
            try:
                correlation = await waits.reserve_correlation(
                    reservation(deadline=clock[0] - timedelta(seconds=1)), context=CONTEXT
                )
                registered = registration(correlation)
                await waits.register(registered, context=CONTEXT)
                scheduler = TaskStoreWaitScheduler(
                    waits=waits, task_store=tasks, scheduler_id="timer"
                )
                timer = await scheduler.schedule(registered, context=CONTEXT)
                assert timer is not None
                missing = timer.model_copy(
                    update={
                        "correlation": correlation.model_copy(
                            update={
                                "request": correlation.request.model_copy(
                                    update={"correlation_key": "missing"}
                                )
                            }
                        )
                    }
                )
                with pytest.raises(ExternalWaitConflict):
                    await scheduler.notify(missing, context=CONTEXT)

                async def retire():
                    receipt = await waits.retire_scope(
                        correlation.request.scope, operation_key="retire", context=CONTEXT
                    )
                    if retirement_phase != "retained":
                        assert (
                            await waits.prune_retired_scope(receipt, context=CONTEXT)
                        ).remaining == 0

                if retirement_phase != "during_observation":
                    await retire()
                restored = ExternalEventWaits(store=reopen(), access_policy=policy)
                reopened_tasks = open_tasks()
                scheduler = TaskStoreWaitScheduler(
                    waits=restored, task_store=reopened_tasks, scheduler_id="timer"
                )
                if retirement_phase == "during_observation":
                    observe = restored.observe

                    async def retire_before_observation(*args, **kwargs):
                        await retire()
                        return await observe(*args, **kwargs)

                    monkeypatch.setattr(restored, "observe", retire_before_observation)

                app = CayuApp(task_store=reopened_tasks, enable_logging=False)
                assert (
                    await run_task_worker(
                        app,
                        reopened_tasks,
                        scheduler.worker_handler(context=CONTEXT),
                        worker_id="timer-worker",
                        query=TaskQuery(type=EXTERNAL_WAIT_TASK_TYPE),
                        max_tasks=1,
                    )
                    == 1
                )
                final = await reopened_tasks.load_task(timer.task_id)
                assert final.status.value == "completed"
                assert final.result == {
                    "outcome_kind": None,
                    "pending_handoff": False,
                    "obsolete": True,
                }
                assert await scheduler.notify(timer, context=CONTEXT) is None
                assert await reopened_tasks.load_task(timer.task_id) == final
                for field, value in (
                    ("scheduler_id", "other"),
                    ("task_id", "missing-task"),
                    ("registration_sha256", "0" * 64),
                    ("correlation", missing.correlation),
                ):
                    with pytest.raises(ExternalWaitConflict):
                        await scheduler.notify(
                            timer.model_copy(update={field: value}), context=CONTEXT
                        )
                policy.revoked = True
                with pytest.raises(PermissionError):
                    await scheduler.notify(timer, context=CONTEXT)
                await restored.aclose()
                await waits.aclose()
            finally:
                for owner in task_owners:
                    await owner.close()

    asyncio.run(scenario())
