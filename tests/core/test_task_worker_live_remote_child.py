"""An expired parent task never takes over its still-live remote child."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.runtime.test_orphaned_session_fast_reconciliation import _foreground_application
from tests.runtime.test_orphaned_session_reconciliation import _Crashing, _request
from tests.runtime.test_orphaned_session_reconciliation import database_url as database_url

from cayu import EventType, SessionStatus
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.queries import SessionQuery
from cayu.storage import open_application_stores
from cayu.tasks import TaskCreate, TaskStatus
from cayu.tasks.worker import _recover_expired_interrupted_task_handoffs
from cayu.tools.base import ToolResult


def _remote_child_worker():
    async def scenario():
        path = Path(os.environ["CAYU_REMOTE_CHILD_STORE"])
        stores = open_application_stores(
            os.environ["CAYU_REMOTE_CHILD_DATABASE"], sqlite_path=path, event_watchers=False
        )
        app = _foreground_application(stores, register_environment=True)

        async def blocked_tool(*_args):
            path.with_suffix(".entered").write_text("dispatched")
            async with asyncio.timeout(45):
                while not path.with_suffix(".release").exists():
                    await asyncio.sleep(0.02)
            return ToolResult(content="remote tool completed")

        _Crashing.run = blocked_tool
        app.get_provider()._responses.append(
            [ModelStreamEvent.text_delta("child finished"), ModelStreamEvent.completed()]
        )
        try:
            await app.create_task(TaskCreate(task_id="remote-child-job", type="scan"))
            task = await stores.task_store.claim_task("remote-worker", lease_seconds=1)
            await stores.task_store.mark_claimed_task_execution_started(
                task.id,
                "remote-worker",
                task.lease_expires_at,
                claim_generation=task.claim_generation,
            )
            request = _request("parent").model_copy(
                update={
                    "task_id": task.id,
                    "task_worker_id": "remote-worker",
                    "task_lease_expires_at": task.lease_expires_at,
                }
            )
            try:
                async for _ in app.run(request):
                    pass
            except Exception:
                # The parent was fenced by the recovery process. Its child's
                # separately owned completion must still be durably observable.
                pass
            path.with_suffix(".finished").write_text("returned")
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("registered", [True, False])
def test_expired_parent_preserves_child_on_another_process(tmp_path, database_url, registered):
    path = tmp_path / "cayu.db"
    with (tmp_path / "worker.log").open("w+") as log:
        worker = subprocess.Popen(
            [
                sys.executable,
                "-c",
                f"from {__name__} import _remote_child_worker; _remote_child_worker()",
            ],
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(item for item in sys.path if item),
                "CAYU_REMOTE_CHILD_STORE": str(path),
                "CAYU_REMOTE_CHILD_DATABASE": database_url,
            },
            stdout=log,
            stderr=subprocess.STDOUT,
        )

        async def scenario():
            async with asyncio.timeout(20):
                while not path.with_suffix(".entered").exists():
                    assert worker.poll() is None
                    await asyncio.sleep(0.02)
            stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
            scanner = _foreground_application(stores, register_environment=registered)
            try:
                [child] = (
                    await stores.session_store.list_sessions(
                        SessionQuery(parent_session_id="parent")
                    )
                ).sessions
                assert (
                    await stores.session_store.inspect_session_execution(child.id)
                ).state == "executing"
                child_before = await stores.session_store.load(child.id)
                checkpoint_before = await stores.session_store.load_checkpoint(child.id)
                events_before = await stores.session_store.load_events(child.id)
                async with asyncio.timeout(10):
                    while (
                        await stores.task_store.load_expired_interrupted_task_handoff_candidate(
                            "remote-child-job"
                        )
                        is None
                    ):
                        await asyncio.sleep(0.02)
                await stores.task_store.cancel_task(
                    "remote-child-job", {"code": "operator_cancelled"}
                )
                held = await stores.task_store.load_task("remote-child-job")
                page = await _recover_expired_interrupted_task_handoffs(
                    scanner, stores.task_store, worker_id="scanner", after=None, limit=10, stop=None
                )
                assert not page.failed and page.recovered == 0
                parent = await stores.session_store.load("parent")
                assert parent.status is SessionStatus.INTERRUPTED
                assert await stores.session_store.load(child.id) == child_before
                assert await stores.session_store.load_checkpoint(child.id) == checkpoint_before
                assert await stores.session_store.load_events(child.id) == events_before
                assert (
                    await stores.session_store.inspect_session_execution(child.id)
                ).state == "executing"
                assert await stores.task_store.load_task("remote-child-job") == held
                assert held.status_reason == "cancellation_requested"
                assert await stores.task_store.claim_task("competitor", lease_seconds=30) is None
                path.with_suffix(".release").write_text("release remote tool")
                async with asyncio.timeout(20):
                    while worker.poll() is None:
                        await asyncio.sleep(0.02)
                assert worker.returncode == 0
                assert path.with_suffix(".finished").exists()
                assert (await stores.session_store.load(child.id)).status is SessionStatus.COMPLETED
                current = await stores.session_store.load("parent")
                assert (
                    current.status is SessionStatus.INTERRUPTED
                    and current.run_epoch == parent.run_epoch
                )
                assert (
                    sum(
                        e.type is EventType.MODEL_STARTED
                        for e in await stores.session_store.load_events("parent")
                    )
                    == 1
                )
                assert scanner.get_provider().requests == []
                settled = await _recover_expired_interrupted_task_handoffs(
                    scanner, stores.task_store, worker_id="scanner", after=None, limit=10, stop=None
                )
                assert not settled.failed and settled.recovered == 1
                assert (
                    await stores.task_store.load_task("remote-child-job")
                ).status is TaskStatus.CANCELLED
            finally:
                path.with_suffix(".release").touch()
                await scanner.aclose()
                await stores.close()

        try:
            asyncio.run(scenario())
        except BaseException:
            log.seek(0)
            print(log.read())
            raise
        finally:
            if worker.poll() is None:
                worker.terminate()
            worker.wait(timeout=10)
