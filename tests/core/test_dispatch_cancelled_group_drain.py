"""An exceptional natural drain still owns its task-group execution acknowledgement."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_dispatch import _batch, _build, _dispatch_request
from tests.core.test_task_execution_settlement_owner import create_group

from cayu.messages import Message
from cayu.sessions.invocation import TaskExecutionSource
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks._execution_settlement import pending_task_execution_settlements
from cayu.tasks.creation import TaskCreate, task_create_with_runtime_invocation
from cayu.tasks.dispatch import _queued_dispatch_persisted_envelope, _queued_dispatch_task_id
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("cancel_owner", [False, True])
def test_exceptional_cancellation_drain_releases_group_finalizer(
    monkeypatch, tmp_path, backend, cancel_owner
):
    async def scenario():
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "group-drain.sqlite")
        )
        h = _build([_batch("initial")], task_store=tasks)
        session_id = "group-cancellation-drain"
        async for _ in h.app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "initial")],
            )
        ):
            pass
        request = _dispatch_request(session_id, "group-drain-member")
        task_id = _queued_dispatch_task_id(request)
        envelope = await h.app._prepare_queued_dispatch(request, queue_task_id=task_id)
        await create_group(
            h.app,
            task_create_with_runtime_invocation(
                TaskCreate(
                    task_id=task_id,
                    type=h.dispatcher.task_type,
                    input={"dispatch": _queued_dispatch_persisted_envelope(envelope)},
                ),
                source=TaskExecutionSource.TASK_DISPATCH,
                session_invocation=await h.app.session_invocation_for_dispatch(session_id),
            ),
        )
        entered = asyncio.Event()
        release = asyncio.Event()
        fenced = asyncio.Event()
        coordinator = h.app._queued_dispatch_coordinator
        load_snapshot = coordinator._load_session_snapshot
        reads = 0

        async def delayed_failed_snapshot(identity):
            nonlocal reads
            reads += 1
            if reads == 2:
                # The first read classified initial settlement. This read is
                # inside execute(), after durable execution entry, before admission.
                entered.set()
                await release.wait()
                raise ConnectionError("session snapshot connection dropped")
            return await load_snapshot(identity)

        request_fence = h.dispatcher._request_dispatch_cancellation_fence

        async def observe_fence(*args, **kwargs):
            result = await request_fence(*args, **kwargs)
            fenced.set()
            return result

        monkeypatch.setattr(coordinator, "_load_session_snapshot", delayed_failed_snapshot)
        monkeypatch.setattr(h.dispatcher, "_request_dispatch_cancellation_fence", observe_fence)
        processing = asyncio.create_task(h.dispatcher.process_next(h.app, worker_id="owner"))
        waiter = asyncio.create_task(entered.wait())
        try:
            done, _ = await asyncio.wait(
                {waiter, processing}, timeout=10, return_when=asyncio.FIRST_COMPLETED
            )
            if processing in done:
                raise AssertionError(("processing exited before read", processing.result(), reads))
            assert waiter in done
            await waiter
            current = await tasks.load_task(task_id)
            assert current is not None and current.started_at is not None
            if cancel_owner:
                processing.cancel("worker shutdown")
                await asyncio.wait_for(fenced.wait(), 10)
            release.set()
            with pytest.raises(
                asyncio.CancelledError if cancel_owner else ConnectionError
            ) as interrupted:
                await asyncio.wait_for(processing, 10)
            # A fix can retain an acknowledgement-only retry owner on the
            # primary signal instead of completing acknowledgement inline.
            for settlement in pending_task_execution_settlements(interrupted.value):
                await settlement.retry()
            assert len(h.provider.requests) == 1  # no queued provider call was admitted
            current = await tasks.load_task(task_id)
            assert current is not None
            assert current.status is (TaskStatus.CANCELLED if cancel_owner else TaskStatus.PENDING)

            # A healthy later group decision must not wait on the dead dispatcher.
            await tasks.complete_task("winner", {})
            for _ in range(3):
                await h.app.reconcile_task_group("owned")
            finalizer = await tasks.load_task("finalizer")
            group = await h.app.load_task_group("owned")
            assert finalizer is not None and group is not None
            assert finalizer.status is TaskStatus.PENDING, group.quiescence
        finally:
            release.set()
            if not processing.done():
                processing.cancel()
            if not waiter.done():
                waiter.cancel()
            await asyncio.gather(processing, waiter, return_exceptions=True)
            if backend == "sqlite":
                await tasks.close()

    asyncio.run(scenario())
