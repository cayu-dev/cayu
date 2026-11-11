"""Verified-work admission must consume the managed schedule's authority."""

from datetime import UTC, datetime, timedelta

import pytest
from tests.core.task_invocation_fixtures import task_backed_session_invocation
from tests.core.test_verified_work_contracts import _contract
from tests.core.test_work_attempt_admission import _prepare_request

from cayu.storage.tasks_sqlite import SQLiteTaskStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.scheduling import TaskMisfirePolicy, TaskScheduleConflict, TaskSchedulePolicy

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize("ineligible", ["expired", "skipped"])
async def test_verified_admission_rejects_ineligible_schedule(tmp_path, ineligible):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = SQLiteTaskStore(
        tmp_path / "schedules.sqlite", clock=lambda: now, ownership_clock=lambda: now
    )
    try:
        contract = await store.publish_work_contract(_contract())
        task = await store.create_task(
            TaskCreate(
                task_id="scheduled-work",
                type="verified-work",
                work_contract=contract.reference(),
                available_at=now - timedelta(minutes=2),
                schedule_policy=TaskSchedulePolicy(
                    expires_at=now - timedelta(minutes=1) if ineligible == "expired" else None,
                    misfire_policy=TaskMisfirePolicy.SKIP,
                    misfire_grace_seconds=30,
                ),
            )
        )
        request = _prepare_request(
            task_id=task.id,
            session_id="scheduled-session",
            session_invocation=await task_backed_session_invocation(
                store, task.id, "scheduled-session"
            ),
        )
        with pytest.raises(TaskScheduleConflict):
            await store.prepare_work_attempt_admission(request)
        assert await store.load_work_attempt_admission(request.admission_id) is None
        assert await store.load_task(task.id) == task
    finally:
        await store.close()


async def test_verified_admission_records_first_schedule_admission(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = SQLiteTaskStore(
        tmp_path / "admitted.sqlite", clock=lambda: now, ownership_clock=lambda: now
    )
    try:
        contract = await store.publish_work_contract(_contract())
        task = await store.create_task(
            TaskCreate(
                task_id="scheduled-work",
                type="verified-work",
                work_contract=contract.reference(),
                available_at=now,
                schedule_policy=TaskSchedulePolicy(),
            )
        )
        request = _prepare_request(
            task_id=task.id,
            session_id="scheduled-session",
            session_invocation=await task_backed_session_invocation(
                store, task.id, "scheduled-session"
            ),
        )
        await store.prepare_work_attempt_admission(request)
        admitted = await store.load_task(task.id)
        assert admitted.schedule.admitted_at == now
        assert admitted.schedule.revision == task.schedule.revision + 1
    finally:
        await store.close()
