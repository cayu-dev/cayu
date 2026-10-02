"""Queued dispatches stay queued when the application starts shutting down."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_dispatch import _batch, _build, _create_resumable_session, _dispatch_request

from cayu.runtime.application_lifecycle import ApplicationAdmissionsSealed
from cayu.tasks import TaskStatus


def test_a_dispatch_claimed_by_a_closing_app_returns_to_the_queue() -> None:
    harness = _build([_batch("first"), _batch("second")])
    _create_resumable_session(harness.app, "session")

    async def scenario() -> None:
        submitted = await harness.app.dispatch(_dispatch_request("session", "queued"))
        task_id = submitted.metadata["queue_task_id"]
        harness.app.seal_admissions()
        with pytest.raises(ApplicationAdmissionsSealed):
            await harness.dispatcher.process_next(harness.app, worker_id="worker")
        task = await harness.tasks.load_task(task_id)
        assert task is not None and task.status is TaskStatus.PENDING
        assert task.worker_id is None

    asyncio.run(scenario())


def test_the_dispatch_worker_stops_claiming_once_the_app_closes() -> None:
    harness = _build([_batch("first"), _batch("second")])
    _create_resumable_session(harness.app, "session")

    async def scenario() -> None:
        submitted = await harness.app.dispatch(_dispatch_request("session", "queued"))
        task_id = submitted.metadata["queue_task_id"]
        harness.app.seal_admissions()
        # Returns on its own: a closing app ends the loop without a stop event.
        await asyncio.wait_for(
            harness.dispatcher.run_worker(
                harness.app, worker_id="worker", stop=asyncio.Event(), poll_interval_s=0.01
            ),
            5,
        )
        task = await harness.tasks.load_task(task_id)
        assert task is not None and task.status is TaskStatus.PENDING

    asyncio.run(scenario())
