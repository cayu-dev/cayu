"""Native connector lifetime; product validation is controlled in these unit cases."""

import asyncio
import importlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from cayu import (
    CayuApp,
    GitHubCheckBundle,
    GitHubCheckObservation,
    GitHubDeliveryState,
    GitHubReviewPolicy,
    InMemoryTaskStore,
    SQLiteTaskStore,
    TaskCreate,
    TaskQuery,
    TaskStatus,
    run_task_worker,
)
from cayu.cli.project import project_context
from cayu.github_delivery import approve_github_delivery
from tests.core.test_github_delivery import FakeTransport, _connector, _pr, _request
from tests.qualification.test_repository_maintenance_application import project as project


@pytest.fixture
def phase(project, tmp_path):
    request = _request()
    request = request.model_copy(
        update={
            "requested_at": datetime.now(UTC).isoformat(),
            "limits": request.limits.model_copy(
                update={"poll_interval_seconds": 1, "timeout_seconds": 1, "max_elapsed_seconds": 30}
            ),
            "reviews": GitHubReviewPolicy(),
        }
    )
    transport = FakeTransport(request)
    connector, _store = _connector(tmp_path, request, transport)
    connector.clock = lambda: datetime.now(UTC)

    async def verify(*args):
        return None

    with project_context(project):
        module = importlib.import_module("operations.maintenance_github")
        yield module, SimpleNamespace(verify=verify), connector, transport, request


def invoke(phase, *, request=None):
    module, application, connector, _transport, original = phase
    selected = original if request is None else request
    return module.observe_github_delivery(
        application,
        object(),
        object(),
        object(),
        connector,
        selected,
        approval=approve_github_delivery(selected, approval_id="explicit-fixture-approval"),
    )


@pytest.mark.parametrize("passed", [True, False])
def test_pending_checks_observe_same_request_then_close(phase, passed):
    _module, _application, connector, transport, request = phase
    transport.check_bundles.append(
        GitHubCheckBundle(
            head_commit=request.repository.head_commit,
            checks=(
                GitHubCheckObservation(
                    provider_id="passed",
                    name="test",
                    head_commit=request.repository.head_commit,
                    status="completed",
                    conclusion="success" if passed else "failure",
                ),
            ),
        )
    )

    async def scenario():
        result = await invoke(phase)
        assert result.result.state is (
            GitHubDeliveryState.CHECKS_PASSED if passed else GitHubDeliveryState.CHECKS_FAILED
        )
        assert result.result.connector_run_id == request.connector_run_id
        assert transport.create_calls == 1
        assert await connector.aclose(timeout_s=0) is True
        assert await connector.repository.latest(request) == result

    asyncio.run(scenario())


def test_cancel_during_dispatched_create_waits_for_positive_close(phase, monkeypatch):
    _module, _application, connector, transport, _request_value = phase

    async def scenario():
        entered, release, closing = asyncio.Event(), asyncio.Event(), asyncio.Event()
        close = connector.aclose

        async def create(config, request):
            transport.create_calls += 1
            entered.set()
            await release.wait()
            transport.pull_request = _pr(request)
            return transport.pull_request, "retained-create"

        async def observe_close(**kwargs):
            closing.set()
            return await close(**kwargs)

        monkeypatch.setattr(transport, "create_pull_request", create)
        monkeypatch.setattr(connector, "aclose", observe_close)
        owner = asyncio.create_task(invoke(phase))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            owner.cancel("first-stop")
            await asyncio.wait_for(closing.wait(), timeout=5)
            owner.cancel("second-stop")
            await asyncio.sleep(0)
            assert not owner.done() and owner.cancelling() == 2
            assert transport.pull_request is None
            release.set()
            done, _ = await asyncio.wait((owner,), timeout=5)
            assert owner in done
            with pytest.raises(asyncio.CancelledError, match="first-stop"):
                await owner
            assert owner.cancelled() and owner.cancelling() == 2
            assert transport.create_calls == 1 and transport.reviewer_calls == 0
            assert await close(timeout_s=0) is True
        finally:
            release.set()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            assert await close(timeout_s=5) is True

    asyncio.run(scenario())


def test_cancel_during_poll_wait_prevents_next_observation(phase, monkeypatch):
    _module, _application, connector, transport, _request_value = phase

    async def scenario():
        observed = asyncio.Event()
        original = connector.run
        calls = 0

        async def observe(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = await original(*args, **kwargs)
            observed.set()
            return result

        monkeypatch.setattr(connector, "run", observe)
        owner = asyncio.create_task(invoke(phase))
        try:
            await asyncio.wait_for(observed.wait(), timeout=5)
            await asyncio.sleep(0)
            owner.cancel("stop-poll")
            with pytest.raises(asyncio.CancelledError, match="stop-poll"):
                await owner
            assert owner.cancelled() and owner.cancelling() == 1
            assert calls == 1 and transport.create_calls == 1
            assert await connector.aclose(timeout_s=0) is True
        finally:
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("expired", [True, False])
def test_original_deadline_is_not_renewed(phase, expired):
    _module, _application, connector, transport, request = phase
    request = request.model_copy(
        update={
            "requested_at": (
                datetime.now(UTC) - timedelta(seconds=31 if expired else 29.6)
            ).isoformat(),
        }
    )

    async def scenario():
        with pytest.raises(TimeoutError):
            await invoke(phase, request=request)
        assert transport.create_calls == (0 if expired else 1)
        assert await connector.aclose(timeout_s=0) is True

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_runtime_worker_retains_fence(phase, monkeypatch, tmp_path, backend):
    _module, _application, connector, transport, _request_value = phase

    async def scenario():
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "phase.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(TaskCreate(task_id="phase", type="maintenance.github"))
        entered, release, closing = asyncio.Event(), asyncio.Event(), asyncio.Event()
        close = connector.aclose

        async def create(config, request):
            transport.create_calls += 1
            entered.set()
            await release.wait()
            transport.pull_request = _pr(request)
            return transport.pull_request, "owned-phase-create"

        async def observe_close(**kwargs):
            closing.set()
            return await close(**kwargs)

        async def handle(_app, task, worker_id):
            await invoke(phase)

        monkeypatch.setattr(transport, "create_pull_request", create)
        monkeypatch.setattr(connector, "aclose", observe_close)
        query = TaskQuery(type="maintenance.github")
        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handle,
                worker_id="phase-owner",
                query=query,
                lease_seconds=1,
                poll_interval_s=0.001,
                reclaim=False,
                max_tasks=1,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            await asyncio.wait_for(closing.wait(), timeout=5)
            worker.cancel("stop-phase-worker")
            for _ in range(100):
                current = await store.load_task("phase")
                if current is not None and current.status_reason == "cancellation_requested":
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("Worker did not record its cancellation fence")
            await asyncio.sleep(1.05)
            assert not worker.done() and transport.pull_request is None
            assert await store.reclaim_expired(query=query) == []
            assert await store.claim_task("replacement", query, lease_seconds=1) is None
            release.set()
            done, _ = await asyncio.wait((worker,), timeout=5)
            assert worker in done
            with pytest.raises(asyncio.CancelledError, match="stop-phase-worker"):
                await worker
            assert worker.cancelled() and worker.cancelling() == 1
            final = await store.load_task("phase")
            assert final is not None and final.status is TaskStatus.CANCELLED
            assert final.worker_id is None and final.lease_expires_at is None
            assert transport.create_calls == 1 and transport.reviewer_calls == 0
        finally:
            release.set()
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            assert await close(timeout_s=5) is True
            if isinstance(store, SQLiteTaskStore):
                await store.close()

    asyncio.run(scenario())


def test_primary_and_cleanup_errors_preserve_original_objects(phase, monkeypatch):
    module, _application, connector, _transport, _request_value = phase
    primary, cleanup = ValueError("verification failed"), RuntimeError("close failed")

    async def fail(*args, **kwargs):
        raise primary

    async def fail_close(**kwargs):
        raise cleanup

    monkeypatch.setattr(module, "run_verified_github_delivery", fail)
    monkeypatch.setattr(connector, "aclose", fail_close)

    async def scenario():
        with pytest.raises(ExceptionGroup) as caught:
            await invoke(phase)
        assert caught.value.exceptions == (primary, cleanup)

    asyncio.run(scenario())
