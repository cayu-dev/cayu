"""Concurrent workers keep per-task ownership and a shared handling limit."""

from __future__ import annotations

import asyncio

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskStatus,
    complete_managed_task,
    run_task_worker,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("limit", [1, 4, 7])
def test_concurrency_bounds_total_claims_and_isolates_handler_failures(
    backend, limit, sqlite_resources
):
    async def scenario():
        async with sqlite_resources as resources:
            store = (
                resources.own(SQLiteTaskStore(resources.path()))
                if backend == "sqlite"
                else InMemoryTaskStore()
            )
            app = CayuApp(task_store=store, enable_logging=False)
            tasks = [await store.create_task(TaskCreate(type="job")) for _ in range(10)]
            active = 0
            peak = 0
            entered = 0
            first_wave = asyncio.Event()
            release = asyncio.Event()
            settled = []

            async def handler(_app, task, worker_id):
                nonlocal active, peak, entered
                active += 1
                entered += 1
                peak = max(peak, active)
                if active == min(4, limit):
                    first_wave.set()
                try:
                    await release.wait()
                    if task.id == tasks[0].id:
                        raise RuntimeError("one job failed")
                    await complete_managed_task(store, task, worker_id, {"ok": True})
                    settled.append(task.id)
                finally:
                    active -= 1

            worker = asyncio.create_task(
                run_task_worker(
                    app,
                    store,
                    handler,
                    worker_id="pool",
                    concurrency=4,
                    max_tasks=limit,
                    poll_interval_s=0.01,
                )
            )
            try:
                await asyncio.wait_for(first_wave.wait(), 10)
                assert entered == min(4, limit)
                release.set()
                assert await asyncio.wait_for(worker, 10) == limit
            finally:
                release.set()
                if not worker.done():
                    worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            states = [(await store.load_task(task.id)).status for task in tasks]
            assert peak == min(4, limit)
            assert states.count(TaskStatus.FAILED) == 1
            assert states.count(TaskStatus.COMPLETED) == limit - 1
            assert states.count(TaskStatus.PENDING) == 10 - limit
            assert len(settled) == limit - 1

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_concurrency_shutdown_drains_every_active_handler(cancel):
    async def scenario():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        tasks = [await store.create_task(TaskCreate(type="job")) for _ in range(6)]
        entered = set()
        drained = set()
        ready = asyncio.Event()
        release = asyncio.Event()
        stop = asyncio.Event()

        async def handler(_app, task, worker_id):
            entered.add(task.id)
            if len(entered) == 4:
                ready.set()
            try:
                await release.wait()
                if not cancel:
                    await complete_managed_task(store, task, worker_id, {})
            finally:
                drained.add(task.id)

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="pool",
                concurrency=4,
                stop=stop,
                poll_interval_s=0.01,
            )
        )
        try:
            await asyncio.wait_for(ready.wait(), 10)
            if cancel:
                worker.cancel()
                async with asyncio.timeout(10):
                    while True:
                        states = [await store.load_task(task_id) for task_id in entered]
                        if all(task.status_reason == "cancellation_requested" for task in states):
                            break
                        await asyncio.sleep(0)
            else:
                stop.set()
            assert not worker.done()
            assert not drained
            release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(worker, 10)
            else:
                assert await asyncio.wait_for(worker, 10) == 4
            assert drained == entered
            states = [(await store.load_task(task.id)).status for task in tasks]
            expected = TaskStatus.CANCELLED if cancel else TaskStatus.COMPLETED
            assert states.count(expected) == 4
            assert states.count(TaskStatus.PENDING) == 2
        finally:
            stop.set()
            release.set()
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4"])
def test_concurrency_rejects_invalid_counts_even_for_empty_drains(value):
    async def scenario():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)

        async def handler(*_args):
            raise AssertionError("must not claim")

        with pytest.raises(ValueError, match="concurrency must be a positive int"):
            await run_task_worker(
                app, store, handler, worker_id="pool", concurrency=value, max_tasks=0
            )

    asyncio.run(scenario())


def test_concurrent_handlers_renew_their_own_leases(monkeypatch):
    async def scenario():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        tasks = [await store.create_task(TaskCreate(type="job")) for _ in range(4)]
        renewed = set()
        all_renewed = asyncio.Event()
        original = store.heartbeat

        async def heartbeat(task_id, *args, **kwargs):
            result = await original(task_id, *args, **kwargs)
            renewed.add(task_id)
            if len(renewed) == 4:
                all_renewed.set()
            return result

        monkeypatch.setattr(store, "heartbeat", heartbeat)

        async def handler(_app, task, worker_id):
            await all_renewed.wait()
            await complete_managed_task(store, task, worker_id, {})

        count = await asyncio.wait_for(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="pool",
                concurrency=4,
                max_tasks=4,
                lease_seconds=3,
                poll_interval_s=0.01,
            ),
            10,
        )
        assert count == 4
        assert renewed == {task.id for task in tasks}
        states = [(await store.load_task(task.id)).status for task in tasks]
        assert all(status is TaskStatus.COMPLETED for status in states)

    asyncio.run(scenario())


def test_infrastructure_failure_drains_healthy_handlers_without_cancelling():
    async def scenario():
        release_first = asyncio.Event()
        release_second = asyncio.Event()
        ready = asyncio.Event()
        failed = asyncio.Event()
        entered = []

        class FailingClaimStore(InMemoryTaskStore):
            verified_work_mutations_are_cancellation_quiescent = True

            async def claim_task(self, *args, **kwargs):
                if len(entered) == 2:
                    failed.set()
                    raise RuntimeError("claim transport failed")
                return await super().claim_task(*args, **kwargs)

        store = FailingClaimStore()
        app = CayuApp(task_store=store, enable_logging=False)
        tasks = [await store.create_task(TaskCreate(type="job")) for _ in range(3)]

        async def handler(_app, task, worker_id):
            entered.append(task.id)
            if len(entered) == 2:
                ready.set()
            await (release_first if task.id == entered[0] else release_second).wait()
            await complete_managed_task(store, task, worker_id, {})

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="pool",
                concurrency=2,
                reclaim=False,
                recover_interrupted_handoffs=False,
                poll_interval_s=0.01,
            )
        )
        try:
            await asyncio.wait_for(ready.wait(), 10)
            release_first.set()
            await asyncio.wait_for(failed.wait(), 10)
            assert not worker.done()
            healthy = await store.load_task(entered[1])
            assert healthy.status_reason != "cancellation_requested"
            release_second.set()
            with pytest.raises(RuntimeError, match="claim transport failed"):
                await asyncio.wait_for(worker, 10)
            assert [(await store.load_task(task.id)).status for task in tasks] == [
                TaskStatus.COMPLETED,
                TaskStatus.COMPLETED,
                TaskStatus.PENDING,
            ]
        finally:
            release_first.set()
            release_second.set()
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(scenario())


def test_budget_waiting_slot_yields_poller_turn_to_another_worker():
    async def scenario():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        tasks = [await store.create_task(TaskCreate(type="job")) for _ in range(3)]
        release_first = asyncio.Event()
        release_second = asyncio.Event()
        ready = asyncio.Event()
        first_finished = asyncio.Event()
        entered = []

        async def handler(_app, task, worker_id):
            entered.append(task.id)
            if len(entered) == 2:
                ready.set()
            await (release_first if task.id == entered[0] else release_second).wait()
            await complete_managed_task(store, task, worker_id, {})
            if task.id == entered[0]:
                first_finished.set()

        async def independent_handler(_app, task, worker_id):
            assert task.id == tasks[2].id
            await complete_managed_task(store, task, worker_id, {})

        pool = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="pool",
                concurrency=2,
                max_tasks=2,
                poll_interval_s=0.01,
            )
        )
        independent = None
        try:
            await asyncio.wait_for(ready.wait(), 10)
            release_first.set()
            await asyncio.wait_for(first_finished.wait(), 10)
            independent = asyncio.create_task(
                run_task_worker(
                    app,
                    store,
                    independent_handler,
                    worker_id="independent",
                    max_tasks=1,
                    poll_interval_s=0.01,
                )
            )
            assert await asyncio.wait_for(independent, 2) == 1
            assert not pool.done()
            release_second.set()
            assert await asyncio.wait_for(pool, 10) == 2
        finally:
            release_first.set()
            release_second.set()
            for worker in (pool, independent):
                if worker is not None:
                    if not worker.done():
                        worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(scenario())


def test_cancellation_while_draining_retains_settlement_retry_handle(monkeypatch):
    import cayu.tasks.worker as worker_module
    from cayu._exception_groups import exception_cause, iter_exception_tree
    from cayu.tasks._execution_settlement import (
        TaskExecutionSettlement,
        TaskExecutionSettlementPending,
    )

    async def scenario():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        tasks = [await store.create_task(TaskCreate(type="job")) for _ in range(2)]
        release_first = asyncio.Event()
        release_second = asyncio.Event()
        ready = asyncio.Event()
        failed = asyncio.Event()
        entered = []
        pending = TaskExecutionSettlementPending(TaskExecutionSettlement(store))
        original = worker_module._handle_with_heartbeat

        async def pending_ack(app, task_store, task, handler, worker_id, lease_seconds):
            await original(app, task_store, task, handler, worker_id, lease_seconds)
            if task.id == tasks[0].id:
                failed.set()
                raise pending

        monkeypatch.setattr(worker_module, "_handle_with_heartbeat", pending_ack)

        async def handler(_app, task, worker_id):
            entered.append(task.id)
            if len(entered) == 2:
                ready.set()
            await (release_first if task.id == tasks[0].id else release_second).wait()
            await complete_managed_task(store, task, worker_id, {})

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="pool",
                concurrency=2,
                reclaim=False,
                recover_interrupted_handoffs=False,
                poll_interval_s=0.01,
            )
        )
        try:
            await asyncio.wait_for(ready.wait(), 5)
            release_first.set()
            await asyncio.wait_for(failed.wait(), 5)
            worker.cancel()
            release_second.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                await asyncio.wait_for(worker, 5)
            assert any(
                error is pending for error in iter_exception_tree(exception_cause(caught.value))
            )
            assert pending.settlement is not None
        finally:
            release_first.set()
            release_second.set()
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(scenario())


def test_infrastructure_failure_wakes_idle_slots(monkeypatch):
    import cayu.tasks.worker as worker_module

    async def scenario():
        idle = asyncio.Event()
        original_loop = worker_module.run_durable_worker_loop

        async def observed_loop(*args, **kwargs):
            original_wait = kwargs["wait"]

            async def observed_wait(*wait_args, **wait_kwargs):
                idle.set()
                return await original_wait(*wait_args, **wait_kwargs)

            kwargs["wait"] = observed_wait
            return await original_loop(*args, **kwargs)

        monkeypatch.setattr(worker_module, "run_durable_worker_loop", observed_loop)

        class FailingReclaimStore(InMemoryTaskStore):
            verified_work_mutations_are_cancellation_quiescent = True

            async def reclaim_expired(self, *args, **kwargs):
                await idle.wait()
                raise RuntimeError("reclaim transport failed")

        store = FailingReclaimStore()
        app = CayuApp(task_store=store, enable_logging=False)

        async def handler(*args):
            raise AssertionError("There is no queued work.")

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="pool",
                concurrency=2,
                poll_interval_s=60,
                recover_interrupted_handoffs=False,
                minimum_idle_delay_s=60,
            )
        )
        try:
            await asyncio.wait_for(idle.wait(), 5)
            with pytest.raises(RuntimeError, match="reclaim transport failed"):
                await asyncio.wait_for(worker, 1)
        finally:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(scenario())


def test_handler_cancellation_does_not_cancel_healthy_peer():
    async def scenario():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        tasks = [await store.create_task(TaskCreate(type="job")) for _ in range(3)]
        peer_ready = asyncio.Event()
        release = asyncio.Event()

        async def handler(_app, task, worker_id):
            if task.id == tasks[0].id:
                await peer_ready.wait()
                raise asyncio.CancelledError()
            peer_ready.set()
            await release.wait()
            await complete_managed_task(store, task, worker_id, {})

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="pool",
                concurrency=2,
                max_tasks=3,
                reclaim=False,
                recover_interrupted_handoffs=False,
            )
        )
        try:
            await asyncio.wait_for(peer_ready.wait(), 5)
            async with asyncio.timeout(5):
                while (await store.load_task(tasks[0].id)).status is not TaskStatus.CANCELLED:
                    await asyncio.sleep(0.01)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(worker, 5)
            assert (await store.load_task(tasks[1].id)).status is TaskStatus.COMPLETED
            assert (await store.load_task(tasks[2].id)).status is TaskStatus.PENDING
        finally:
            release.set()
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    asyncio.run(scenario())
