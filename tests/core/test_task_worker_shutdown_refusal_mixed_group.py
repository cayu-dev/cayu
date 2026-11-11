"""Only refusal-only TaskGroup failures may return undispatched work to pending."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_task_worker_shutdown import _app, _run_task, _task_store

from cayu._exception_groups import exception_tree_leaves
from cayu.runtime.application_lifecycle import ApplicationAdmissionsSealed
from cayu.tasks import TaskCreate, TaskStatus, run_task_worker


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("failure_mode", ["none", "independent", "cleanup"])
def test_shutdown_refusal_group_preserves_other_failures(backend, nested, failure_mode, tmp_path):
    async def scenario():
        store = _task_store(backend, tmp_path)
        app = _app(store)
        created = await app.create_task(TaskCreate(type="demo", title="undispatched"))
        sibling_ready = asyncio.Event()
        refused = asyncio.Event()
        observed = []
        failure = ValueError("independent handler failure")

        async def sibling():
            sibling_ready.set()
            if failure_mode == "independent":
                await refused.wait()
                raise failure
            try:
                await asyncio.Future()
            finally:
                raise failure

        async def run_group(owned_app, task, worker_id):
            async def refuse():
                if failure_mode != "none":
                    await sibling_ready.wait()
                try:
                    await _run_task(owned_app, task, worker_id)
                finally:
                    # Wake the independent sibling before TaskGroup observes
                    # the refusal, or let TaskGroup cancel its cleanup sibling.
                    refused.set()

            async with asyncio.TaskGroup() as group:
                group.create_task(refuse())
                if failure_mode != "none":
                    group.create_task(sibling())

        async def handler(owned_app, task, worker_id):
            owned_app.seal_admissions()
            try:
                if nested:
                    async with asyncio.TaskGroup() as group:
                        group.create_task(run_group(owned_app, task, worker_id))
                else:
                    await run_group(owned_app, task, worker_id)
            except ExceptionGroup as error:
                observed.extend(exception_tree_leaves(error))
                raise

        try:
            assert (
                await asyncio.wait_for(
                    run_task_worker(app, store, handler, worker_id="closing", max_tasks=1),
                    10,
                )
                == 1
            )
            assert any(isinstance(error, ApplicationAdmissionsSealed) for error in observed)
            current = await store.load_task(created.id)
            assert current is not None
            assert current.worker_id is None
            assert current.session_id is None
            if failure_mode == "none":
                assert current.status is TaskStatus.PENDING
                assert current.error is None
            else:
                assert failure in observed
                assert current.status is TaskStatus.FAILED
                assert current.error is not None
                assert current.error["error"] == "ExceptionGroup"
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
