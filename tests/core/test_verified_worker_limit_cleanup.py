"""Replacement discovery after a limit stop, before invocation release."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest
from tests.core.test_completion_verifier_adapters import (
    RecordingVerifier,
    _accepted_decision,
    _contract,
)
from tests.core.test_verified_task_worker import _StaticHandler
from tests.core.test_verified_work_contracts import _RecordingProvider
from tests.core.verified_worker_fixtures import (
    VerifiedWorkerStoreFactory,
    wait_for_verified_worker_lease_expiry,
)
from tests.core.verified_worker_fixtures import (
    verified_work_postgres_dsn as verified_work_postgres_dsn,
)
from tests.core.verified_worker_fixtures import (
    verified_worker_store_factory as verified_worker_store_factory,
)

from cayu import (
    AgentSpec,
    BudgetLimit,
    BudgetPolicy,
    CayuApp,
    InMemorySessionStore,
    InMemoryTaskStore,
    ModelPrice,
    PriceBook,
    SessionStatus,
    TaskCreate,
    TaskQuery,
    TaskStatus,
    VerifiedTaskWorker,
)
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.loop_policies import LoopPolicy


class _WaitAfterModel(LoopPolicy):
    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests:limit-cleanup-wait", behavior_version="1", implementation_version="1"
        )

    async def before_stop(self, context):
        await asyncio.Event().wait()


def _limit_app(sessions, tasks, reason):
    policy = BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("1"),
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name="unregistered-provider",
                            model="unregistered-model",
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("1"),
                        ),
                    )
                ),
            ),
        )
        if reason == "budget_limit"
        else ()
    )
    app = CayuApp(
        session_store=sessions,
        task_store=tasks,
        budget_policy=policy,
        enable_logging=False,
        loop_policies=[_WaitAfterModel()] if reason == "elapsed_limit" else [],
    )
    provider = _RecordingProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
    verifier = RecordingVerifier(_accepted_decision())
    app.register_completion_verifier(_contract().verifier, verifier)
    return app, provider, verifier


def _crash_after_limit_stop(directory, reason, postgres_dsn=None, unknown_dispatch=False):
    async def scenario():
        factory = VerifiedWorkerStoreFactory(
            "postgres" if postgres_dsn else "sqlite", Path(directory), postgres_dsn
        )
        sessions, tasks = factory()
        app, provider, _ = _limit_app(sessions, tasks, reason)
        provider_settled = False
        if unknown_dispatch:

            async def stream(self, request):
                nonlocal provider_settled
                self.requests.append(request)
                try:
                    await asyncio.Event().wait()
                    yield None
                finally:
                    provider_settled = True

            _RecordingProvider.stream = stream
        await tasks.publish_work_contract(_contract())
        await tasks.create_task(TaskCreate(type="verified", work_contract=_contract().reference()))
        record_stop = type(tasks).record_work_attempt_execution_stop

        async def crash(store, request):
            result = await record_stop(store, request)
            assert request.reason == reason
            if reason == "budget_limit":
                assert provider.requests == []
            else:
                assert len(provider.requests) == 1
                if unknown_dispatch:
                    assert provider_settled
            # No finally blocks: leave exactly the committed stop without release.
            os._exit(79)
            return result  # pragma: no cover

        type(tasks).record_work_attempt_execution_stop = crash
        async with VerifiedTaskWorker(
            app,
            _StaticHandler(),
            worker_id="limit-crash-worker",
            lease_seconds=5,
            callback_timeout_seconds=1,
            max_elapsed_seconds=3 if reason == "elapsed_limit" else 3600,
        ) as worker:
            await worker.run(max_tasks=1)

    asyncio.run(scenario())


@pytest.mark.parametrize("reason", ["budget_limit", "elapsed_limit"])
def test_memory_worker_recovers_stop_acknowledgement_loss(reason, monkeypatch):
    async def scenario():
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        app, provider, verifier = _limit_app(sessions, tasks, reason)
        await tasks.publish_work_contract(_contract())
        task = await tasks.create_task(
            TaskCreate(type="verified", work_contract=_contract().reference())
        )
        stop_committed = False
        record_stop = type(tasks).record_work_attempt_execution_stop
        publish_transition = type(sessions).publish_interaction_transition

        async def lose_stop_reply(store, request):
            nonlocal stop_committed
            await record_stop(store, request)
            stop_committed = True
            raise ConnectionError("lost stop acknowledgement")

        async def unavailable_terminalization(store, request):
            if stop_committed:
                raise ConnectionError("terminalization unavailable after stop")
            return await publish_transition(store, request)

        with monkeypatch.context() as faults:
            faults.setattr(type(tasks), "record_work_attempt_execution_stop", lose_stop_reply)
            faults.setattr(
                type(sessions), "publish_interaction_transition", unavailable_terminalization
            )
            async with VerifiedTaskWorker(
                app,
                _StaticHandler(),
                worker_id="lost-stop-reply",
                lease_seconds=5,
                callback_timeout_seconds=1,
                max_elapsed_seconds=3 if reason == "elapsed_limit" else 3600,
            ) as worker:
                with pytest.raises(Exception):
                    await asyncio.wait_for(worker.run(max_tasks=1), 30)
        assert stop_committed
        admission = await tasks.load_latest_work_attempt_admission(task.id)
        assert admission.execution_stop.request.reason == reason
        assert (
            await app._session_engine.load_work_attempt_released_recovery_evidence(admission)
            is None
        )
        assert await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id) is None
        await wait_for_verified_worker_lease_expiry(tasks, admission.claim.lease_expires_at)
        replacement, replacement_provider, replacement_verifier = _limit_app(
            sessions, tasks, reason
        )
        handler = _StaticHandler()
        async with VerifiedTaskWorker(
            replacement,
            handler,
            worker_id="memory-replacement",
            lease_seconds=5,
            callback_timeout_seconds=1,
        ) as worker:
            assert await asyncio.wait_for(worker.run(max_tasks=1), 30) == 1
        final = await tasks.load_task(task.id)
        current = await tasks.load_latest_work_attempt_admission(task.id)
        receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
        assert final.status is TaskStatus.NEEDS_ATTENTION
        assert final.status_reason == "work_contract_" + reason
        assert receipt.task == final
        assert current.execution_stop == admission.execution_stop
        assert current.execution_entry == admission.execution_entry
        assert current.claim.generation == 2
        assert handler.preparations == handler.proposals == []
        assert (
            replacement_provider.requests
            == replacement_verifier.requests
            == verifier.requests
            == []
        )
        assert len(provider.requests) == int(reason == "elapsed_limit")
        assert await tasks.settle_work_attempt_lifecycle(receipt.request) == receipt

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize(
    ("reason", "unknown_dispatch"),
    [("budget_limit", False), ("elapsed_limit", False), ("elapsed_limit", True)],
)
def test_replacement_worker_finishes_unreleased_limit_stop(
    backend, reason, unknown_dispatch, verified_worker_store_factory
):
    factory = verified_worker_store_factory
    repository = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "PYTHONPATH": str(repository / "src")}
    environment.pop("CAYU_TEST_VERIFIED_WORKER_DSN", None)
    if factory.postgres_dsn is not None:
        environment["CAYU_TEST_VERIFIED_WORKER_DSN"] = factory.postgres_dsn
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.core.test_verified_worker_limit_cleanup import _crash_after_limit_stop; "
            "import os, sys; _crash_after_limit_stop(sys.argv[1], sys.argv[2], "
            "os.environ.get('CAYU_TEST_VERIFIED_WORKER_DSN'), sys.argv[3] == 'True')",
            str(factory.directory),
            reason,
            str(unknown_dispatch),
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert child.returncode == 79, (child.stdout, child.stderr)

    async def scenario():
        sessions, tasks = factory()
        try:
            (task,) = await tasks.list_tasks(TaskQuery())
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            assert admission.execution_stop.request.reason == reason
            assert (await tasks.load_task(task.id)).status is TaskStatus.RUNNING
            assert await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id) is None
            app, provider, verifier = _limit_app(sessions, tasks, reason)
            engine = app._session_engine
            assert await engine.load_work_attempt_released_recovery_evidence(admission) is None
            assert not await engine.has_recoverable_work_attempt_model_result(admission)
            active = await sessions.load_active_model_completion_stage(admission.session_id)
            assert await engine.has_recoverable_work_attempt_cleanup(admission) is (
                not unknown_dispatch
            )
            await wait_for_verified_worker_lease_expiry(tasks, admission.claim.lease_expires_at)
            handler = _StaticHandler()
            if unknown_dispatch:
                assert active.stage.state == "in_flight"
                assert (
                    await sessions.load_model_completion_stage_dispatch(
                        admission.session_id, active.stage.stage_id
                    )
                    is not None
                )
                checkpoint = await sessions.load_checkpoint(admission.session_id)
                stop = asyncio.Event()
                async with VerifiedTaskWorker(
                    app, handler, worker_id="fenced-replacement", poll_interval_s=0.01
                ) as worker:
                    discover = worker._discover_unfinished_attempt
                    scans = 0

                    async def observe_scan():
                        nonlocal scans
                        result = await discover()
                        scans += 1
                        if scans == 3:
                            stop.set()
                        return result

                    worker._discover_unfinished_attempt = observe_scan
                    assert await asyncio.wait_for(worker.run(stop=stop), 30) == 0
                    assert scans >= 3
                assert await tasks.load_latest_work_attempt_admission(task.id) == admission
                assert (
                    await sessions.load_active_model_completion_stage(admission.session_id)
                    == active
                )
                assert await sessions.load_checkpoint(admission.session_id) == checkpoint
                assert await tasks.load_task(task.id) == task
                assert (
                    await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id) is None
                )
                assert provider.requests == verifier.requests == []
                assert handler.preparations == handler.proposals == []
                return
            assert active is None
            async with VerifiedTaskWorker(
                app,
                handler,
                worker_id="limit-replacement",
                lease_seconds=5,
                callback_timeout_seconds=1,
            ) as worker:
                assert await asyncio.wait_for(worker.run(max_tasks=1), 60) == 1
            final = await tasks.load_task(task.id)
            current = await tasks.load_latest_work_attempt_admission(task.id)
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            assert final.status is TaskStatus.NEEDS_ATTENTION
            assert final.status_reason == "work_contract_" + reason
            assert receipt.task == final
            assert not receipt.retired_contract_binding
            assert current.execution_stop == admission.execution_stop
            assert current.execution_entry == admission.execution_entry
            assert current.claim.generation == 2
            assert provider.requests == verifier.requests == []
            assert handler.preparations == handler.proposals == []
            assert await tasks.load_completion_proposal_for_attempt(admission.attempt_id) is None
            assert (await sessions.load(admission.session_id)).status in {
                SessionStatus.FAILED,
                SessionStatus.INTERRUPTED,
            }
            assert await tasks.settle_work_attempt_lifecycle(receipt.request) == receipt
        finally:
            await tasks.close()
            await sessions.close()

    asyncio.run(scenario())
