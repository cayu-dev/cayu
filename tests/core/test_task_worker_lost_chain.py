"""Expired task cancellation settles lost runs without releasing uncertain descendants."""

from __future__ import annotations

import asyncio
import os

import pytest
from tests.runtime.test_orphaned_session_fast_reconciliation import _foreground_application
from tests.runtime.test_orphaned_session_reconciliation import _Crashing, _orphan, _request
from tests.runtime.test_orphaned_session_reconciliation import database_url as database_url

from cayu import SessionStatus
from cayu.sessions.queries import SessionQuery
from cayu.storage import open_application_stores
from cayu.tasks import TaskCreate, TaskStatus, run_task_worker, run_task_worker_until_idle


def _crash_attached_foreground_worker():
    async def scenario():
        stores = open_application_stores(
            os.environ["CAYU_ORPHAN_TEST_DATABASE_URL"],
            sqlite_path=os.environ["CAYU_ORPHAN_TEST_STORE"],
            event_watchers=False,
        )
        app = _foreground_application(stores, register_environment=True)
        await app.create_task(TaskCreate(task_id="job", type="scan"))

        async def handler(app, task, worker_id):
            request = _request("parent").model_copy(
                update={
                    "task_id": task.id,
                    "task_worker_id": worker_id,
                    "task_lease_expires_at": task.lease_expires_at,
                }
            )
            async for _ in app.run(request):
                pass

        await run_task_worker(
            app,
            stores.task_store,
            handler,
            worker_id="lost-worker",
            lease_seconds=1,
            max_tasks=1,
            reclaim=False,
        )
        raise AssertionError("The foreground child did not reach its crashing tool")

    asyncio.run(scenario())


@pytest.mark.parametrize("registered", [True, False])
def test_cancelled_lost_foreground_task_interrupts_parent_before_chain_settlement(
    tmp_path, database_url, registered, monkeypatch
):
    path = tmp_path / "cayu.db"
    _orphan(
        path,
        "parent",
        database_url=database_url,
        entry="_crash_attached_foreground_worker",
        module=__name__,
    )

    async def scenario():
        stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
        app = _foreground_application(stores, register_environment=registered)
        tool_calls = []

        async def no_tool(*_args):
            tool_calls.append(True)
            raise AssertionError("Recovery must not redispatch the child's external effect")

        monkeypatch.setattr(_Crashing, "run", no_tool)
        try:
            before = (await stores.session_store.list_sessions(SessionQuery(limit=10))).sessions
            assert len(before) == 2 and all(item.status is SessionStatus.RUNNING for item in before)
            async with asyncio.timeout(10):
                while (
                    await stores.task_store.load_expired_interrupted_task_handoff_candidate("job")
                    is None
                ):
                    await asyncio.sleep(0.02)
            await stores.task_store.cancel_task("job", {"code": "operator_cancelled"})

            async def no_handler(*_args):
                raise AssertionError("Cancelled chain must not redispatch a task handler")

            await asyncio.wait_for(
                run_task_worker_until_idle(
                    app,
                    stores.task_store,
                    no_handler,
                    worker_id="replacement",
                    reclaim=False,
                ),
                15,
            )
            parent = await stores.session_store.load("parent")
            assert parent.status is SessionStatus.INTERRUPTED
            current = await stores.task_store.load_task("job")
            # The unresolved child remains fenced. Interrupting a parent is not
            # positive proof that its descendant's dispatched effect stopped.
            assert current.status is TaskStatus.RUNNING
            assert current.status_reason == "cancellation_requested"
            assert current.worker_id == "lost-worker"
            events = await stores.session_store.load_events("parent")
            await asyncio.wait_for(
                run_task_worker_until_idle(
                    app,
                    stores.task_store,
                    no_handler,
                    worker_id="replacement",
                    reclaim=False,
                ),
                15,
            )
            assert await stores.task_store.load_task("job") == current
            assert await stores.session_store.load_events("parent") == events
            assert (await stores.session_store.load("parent")).run_epoch == parent.run_epoch
            assert app.get_provider().requests == [] and tool_calls == []
            await app.reconcile_orphaned_sessions()
            after = (await stores.session_store.list_sessions(SessionQuery(limit=10))).sessions
            assert all(
                item.status in {SessionStatus.INTERRUPTED, SessionStatus.FAILED} for item in after
            )
            assert app.get_provider().requests == [] and tool_calls == []
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("registered", [True, False])
def test_expired_parent_task_keeps_live_child_fenced_through_late_completion(
    tmp_path, database_url, monkeypatch, registered
):
    """A real child that outlives its task's lease cannot resume the fenced parent."""
    from cayu import EventType
    from cayu.providers.base import ModelStreamEvent
    from cayu.tasks.records import TaskClaimLost
    from cayu.tasks.worker import _recover_expired_interrupted_task_handoffs
    from cayu.tools.base import ToolResult

    async def scenario():
        stores = open_application_stores(
            database_url, sqlite_path=tmp_path / "cayu.db", event_watchers=False
        )
        original = _foreground_application(stores, register_environment=True)
        scanner = _foreground_application(stores, register_environment=registered)
        entered, release = asyncio.Event(), asyncio.Event()
        external_calls = []

        async def blocked_tool(*_args):
            external_calls.append(True)
            entered.set()
            await release.wait()
            return ToolResult(content="recorded")

        monkeypatch.setattr(_Crashing, "run", blocked_tool)
        original.get_provider()._responses.append(
            [
                ModelStreamEvent.text_delta("child finished"),
                ModelStreamEvent.completed(),
            ]
        )
        task_run = None
        try:
            await original.create_task(TaskCreate(task_id="live-child-job", type="scan"))
            task = await stores.task_store.claim_task("original", lease_seconds=1)
            await stores.task_store.mark_claimed_task_execution_started(
                task.id,
                "original",
                task.lease_expires_at,
                claim_generation=task.claim_generation,
            )

            async def run():
                request = _request("parent").model_copy(
                    update={
                        "task_id": task.id,
                        "task_worker_id": "original",
                        "task_lease_expires_at": task.lease_expires_at,
                    }
                )
                async for _ in original.run(request):
                    pass

            task_run = asyncio.create_task(run())
            await asyncio.wait_for(entered.wait(), 10)
            [child] = (
                await stores.session_store.list_sessions(SessionQuery(parent_session_id="parent"))
            ).sessions
            assert (
                await stores.session_store.inspect_session_execution(child.id)
            ).state == "executing"
            async with asyncio.timeout(10):
                while (
                    await stores.task_store.load_expired_interrupted_task_handoff_candidate(task.id)
                    is None
                ):
                    await asyncio.sleep(0.02)
            with pytest.raises(TaskClaimLost):
                await stores.task_store.heartbeat(
                    task.id,
                    "original",
                    claim_generation=task.claim_generation,
                    extend_seconds=30,
                )
            cancelled = await stores.task_store.cancel_task(task.id, {"code": "operator_cancelled"})
            assert cancelled.status_reason == "cancellation_requested"
            page = await _recover_expired_interrupted_task_handoffs(
                scanner,
                stores.task_store,
                worker_id="scanner",
                after=None,
                limit=10,
                stop=None,
            )
            assert not page.failed and page.recovered == 0
            parent = await stores.session_store.load("parent")
            assert parent.status is SessionStatus.INTERRUPTED
            assert (await stores.session_store.load(child.id)).status is SessionStatus.RUNNING
            held = await stores.task_store.load_task(task.id)
            assert held.worker_id == "original" and held.claim_generation == task.claim_generation
            assert held.status_reason == "cancellation_requested"
            assert await stores.task_store.claim_task("competitor", lease_seconds=30) is None
            release.set()
            await asyncio.wait_for(asyncio.gather(task_run, return_exceptions=True), 15)
            # The child's real completion/terminal callback has now crossed the
            # durable boundary; its parent's old invocation cannot regain ownership.
            assert (await stores.session_store.load(child.id)).status is SessionStatus.COMPLETED
            current = await stores.session_store.load("parent")
            assert (
                current.status is SessionStatus.INTERRUPTED
                and current.run_epoch == parent.run_epoch
            )
            events = await stores.session_store.load_events("parent")
            assert sum(event.type is EventType.MODEL_STARTED for event in events) == 1
            assert scanner.get_provider().requests == []
            assert len(external_calls) == 1
            current_task = await stores.task_store.load_task(task.id)
            assert (
                current_task.worker_id == "original"
                and current_task.claim_generation == held.claim_generation
            )
            # Only the next native pass, after observing the completed child,
            # may settle the retained cancellation; no successor receives it.
            settled = await _recover_expired_interrupted_task_handoffs(
                scanner,
                stores.task_store,
                worker_id="scanner",
                after=None,
                limit=10,
                stop=None,
            )
            assert not settled.failed and settled.recovered == 1
            assert (await stores.task_store.load_task(task.id)).status is TaskStatus.CANCELLED
        finally:
            release.set()
            if task_run is not None and not task_run.done():
                task_run.cancel()
                await asyncio.gather(task_run, return_exceptions=True)
            await original.aclose()
            await scanner.aclose()
            await stores.close()

    asyncio.run(scenario())
