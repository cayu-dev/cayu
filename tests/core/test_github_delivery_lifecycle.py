"""Public connector lifetime checks; controlled providers, no GitHub writes."""

import asyncio
import threading

import pytest
from tests.core.test_github_delivery import FakeTransport, _connector, _pr, _request

from cayu import CayuApp, InMemoryTaskStore, TaskCreate, TaskQuery, TaskStatus, run_task_worker
from cayu.artifacts.settlement import (
    ArtifactWriteSettlementPhase,
    ArtifactWriteSettlementStatus,
    register_artifact_write_operation,
)
from cayu.github_delivery import GitHubDeliveryAdmissionError, approve_github_delivery
from cayu.storage.sqlite import SQLiteTaskStore


@pytest.mark.parametrize("timeout", [True, -1, float("inf"), float("nan"), "1", 10**400])
def test_invalid_close_does_not_seal(tmp_path, timeout):
    request = _request()
    connector, _ = _connector(tmp_path, request, FakeTransport(request))

    async def scenario():
        with pytest.raises(ValueError, match="finite non-negative"):
            await connector.aclose(timeout_s=timeout)
        # Validation must not alter dispatch authority.
        await connector.run(request, object(), object())
        assert await connector.aclose(timeout_s=0)
        assert await connector.aclose(timeout_s=0)
        with pytest.raises(GitHubDeliveryAdmissionError, match="sealed"):
            await connector.run(request, object(), object())

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_managed_worker_retains_delivery_after_waiter_timeout(tmp_path, monkeypatch, backend):
    request = _request(reviewers=("reviewer",))
    request = request.model_copy(
        update={"limits": request.limits.model_copy(update={"timeout_seconds": 1})}
    )
    transport = FakeTransport(request)
    connector, _ = _connector(tmp_path, request, transport)
    approval = approve_github_delivery(request, approval_id="managed-lifetime")

    async def scenario():
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "worker.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(TaskCreate(task_id="github-delivery-task", type="github-delivery"))
        entered, release, draining = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def delayed_create(config, current):
            del config
            transport.create_calls += 1
            entered.set()
            await release.wait()
            transport.pull_request = _pr(current)
            return transport.pull_request, "managed-create"

        monkeypatch.setattr(transport, "create_pull_request", delayed_create)

        async def handler(_app, _task, _worker_id):
            try:
                await connector.run(request, object(), object(), approval=approval)
            finally:
                while not await connector.aclose(timeout_s=0.01):
                    draining.set()

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="github-owner",
                lease_seconds=1,
                poll_interval_s=0.001,
                reclaim=False,
                max_tasks=1,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            await asyncio.wait_for(draining.wait(), timeout=5)
            worker.cancel("stop delivery worker")
            for _ in range(100):
                current = await store.load_task("github-delivery-task")
                if current is not None and current.status_reason == "cancellation_requested":
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Worker did not publish the cancellation fence.")
            await asyncio.sleep(1.05)
            assert not worker.done()
            assert transport.pull_request is None
            query = TaskQuery(type="github-delivery")
            assert await store.reclaim_expired(query=query) == []
            assert await store.claim_task("replacement", query, lease_seconds=1) is None
            release.set()
            done, _ = await asyncio.wait({worker}, timeout=5)
            assert worker in done
            with pytest.raises(asyncio.CancelledError) as caught:
                await worker
            assert caught.value.args == ("stop delivery worker",)
            assert worker.cancelled() and worker.cancelling() == 1
            terminal = await store.load_task("github-delivery-task")
            assert terminal is not None and terminal.status is TaskStatus.CANCELLED
            assert terminal.worker_id is None and terminal.lease_expires_at is None
            assert transport.create_calls == 1
            assert transport.reviewer_calls == 0
            assert await connector.aclose(timeout_s=0)
        finally:
            release.set()
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            assert await connector.aclose(timeout_s=5)
            if isinstance(store, SQLiteTaskStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_caller", [False, True])
def test_close_retains_dispatched_work_across_deadline_and_cancellation(
    tmp_path, monkeypatch, cancel_caller
):
    request = _request(reviewers=("reviewer",))
    transport = FakeTransport(request)
    connector, _ = _connector(tmp_path, request, transport)
    approval = approve_github_delivery(request, approval_id="lifetime-approval")

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed_create(config, current):
            del config
            transport.create_calls += 1
            entered.set()
            await release.wait()
            transport.pull_request = _pr(current)
            return transport.pull_request, "delayed-create"

        monkeypatch.setattr(transport, "create_pull_request", delayed_create)
        caller = asyncio.create_task(connector.run(request, object(), object(), approval=approval))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if cancel_caller:
                caller.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await caller
                assert caller.cancelled() and caller.cancelling() == 1
            started = asyncio.get_running_loop().time()
            assert not await connector.aclose(timeout_s=0.01)
            assert asyncio.get_running_loop().time() - started < 1
            for run_id in (request.connector_run_id, "another-run"):
                changed = request.model_copy(update={"connector_run_id": run_id})
                with pytest.raises(GitHubDeliveryAdmissionError, match="sealed"):
                    await connector.run(changed, object(), object(), approval=approval)
            closer = asyncio.create_task(connector.aclose(timeout_s=10))
            await asyncio.sleep(0)
            closer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closer
            assert closer.cancelled() and closer.cancelling() == 1
            assert transport.pull_request is None
            assert not await connector.aclose(timeout_s=0)
        finally:
            release.set()
            assert await connector.aclose(timeout_s=5)
            if not cancel_caller:
                await caller
                assert not caller.cancelled() and caller.cancelling() == 0
        assert transport.create_calls == 1
        assert transport.reviewer_calls == 0
        assert await asyncio.gather(
            connector.aclose(timeout_s=0), connector.aclose(timeout_s=0)
        ) == [
            True,
            True,
        ]

    asyncio.run(scenario())


def test_close_waits_for_registered_artifact_after_coroutine_settles(tmp_path, monkeypatch):
    request = _request()
    connector, _ = _connector(tmp_path, request, FakeTransport(request))
    original = connector._run_once
    release, finished = threading.Event(), threading.Event()
    threads = []

    async def with_late_artifact(*args, **kwargs):
        result = await original(*args, **kwargs)
        registration = register_artifact_write_operation(
            artifact_id="late-artifact", store_id="fixture-store"
        )

        def settle():
            try:
                release.wait()
                registration.record(
                    status=ArtifactWriteSettlementStatus.COMMITTED,
                    phase=ArtifactWriteSettlementPhase.SETTLED,
                )
            finally:
                finished.set()

        thread = threading.Thread(target=settle)
        threads.append(thread)
        thread.start()
        return result

    monkeypatch.setattr(connector, "_run_once", with_late_artifact)

    async def scenario():
        try:
            await connector.run(request, object(), object())
            assert not finished.is_set()
            assert not await connector.aclose(timeout_s=0.01)
            release.set()
            assert await connector.aclose(timeout_s=5)
        finally:
            release.set()
            for thread in threads:
                thread.join(timeout=5)
                assert not thread.is_alive()
        assert finished.is_set()

    asyncio.run(scenario())
