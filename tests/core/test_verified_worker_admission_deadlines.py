"""Public worker deadline regressions at the two admission handoffs."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_completion_verifier_adapters import _contract, _rejected_decision
from tests.core.test_verified_task_worker import _ContinueOnceVerifier, _StaticHandler
from tests.core.test_verified_work_contracts import _RecordingProvider
from tests.core.verified_worker_fixtures import (
    verified_work_postgres_dsn as verified_work_postgres_dsn,
)
from tests.core.verified_worker_fixtures import (
    verified_worker_store_factory as verified_worker_store_factory,
)

from cayu import (
    AgentSpec,
    CayuApp,
    CompletionContinuationPolicy,
    CompletionRejectionAction,
    TaskCreate,
    TaskStatus,
    VerifiedTaskWorker,
)
from cayu.runtime._session_engine import SessionEngine
from cayu.runtime.work_attempt_admission import WorkAttemptAdmissionConflict
from cayu.runtime.work_attempt_lifecycle import (
    WorkAttemptLifecycleSettlement,
    work_attempt_admission_authority_sha256,
)


async def _wait_past(expires_at):
    await asyncio.sleep(max(0, (expires_at - datetime.now(UTC)).total_seconds()) + 0.05)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("boundary", ["callback", "admission"])
def test_worker_receipts_elapsed_preparation_without_dispatch(
    backend, boundary, verified_worker_store_factory, monkeypatch
):
    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        provider = _RecordingProvider()
        holds = []
        original_hold = type(tasks).hold_work_attempt_preparation
        original_prepare = SessionEngine._prepare_initial_run

        async def observe_hold(store, request):
            receipt = await original_hold(store, request)
            holds.append(receipt)
            return receipt

        async def delayed_admission(engine, request, **kwargs):
            if boundary == "admission":
                await _wait_past(request.execution_deadline.expires_at)
            return await original_prepare(engine, request, **kwargs)

        class Handler(_StaticHandler):
            async def prepare(self, context):
                if boundary == "callback":
                    await asyncio.sleep(1.1)
                return await super().prepare(context)

        monkeypatch.setattr(type(tasks), "hold_work_attempt_preparation", observe_hold)
        monkeypatch.setattr(SessionEngine, "_prepare_initial_run", delayed_admission)
        try:
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            contract = _contract()
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="work", work_contract=contract.reference())
            )
            handler = Handler()
            async with VerifiedTaskWorker(
                app,
                handler,
                worker_id="preparing",
                max_elapsed_seconds=1,
                callback_timeout_seconds=10,
            ) as worker:
                assert await asyncio.wait_for(worker.run(max_tasks=1), 30) == 1
            assert len(holds) == 1
            receipt = holds[0]
            assert receipt.request.deadline_expires_at <= datetime.now(UTC)
            assert receipt.task == await tasks.load_task(task.id)
            assert receipt.task.status is TaskStatus.NEEDS_ATTENTION
            assert receipt.task.status_reason == "work_contract_elapsed_limit"
            assert receipt.task.session_id is None
            assert await tasks.load_latest_work_attempt_admission(task.id) is None
            assert provider.requests == handler.proposals == []
            assert len(handler.preparations) == 1
            if backend != "memory":
                await sessions.close()
                await tasks.close()
                sessions, tasks = verified_worker_store_factory()
            assert await tasks.hold_work_attempt_preparation(receipt.request) == receipt
            with pytest.raises(WorkAttemptAdmissionConflict):
                await tasks.hold_work_attempt_preparation(
                    receipt.request.model_copy(
                        update={
                            "deadline_expires_at": receipt.request.deadline_expires_at
                            - timedelta(seconds=1)
                        }
                    )
                )
            assert await tasks.load_task(task.id) == receipt.task
        finally:
            if backend != "memory":
                await sessions.close()
                await tasks.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("boundary", ["handoff", "restart", "successor_ack_loss", "other_failure"])
def test_worker_settles_expired_applied_rejection_without_successor(
    backend, boundary, verified_worker_store_factory, monkeypatch
):
    async def scenario():
        sessions, tasks = verified_worker_store_factory()
        provider = _RecordingProvider()
        verifier = _ContinueOnceVerifier(_rejected_decision())
        contract = _contract(
            continuation_policy=CompletionContinuationPolicy(
                rejection_action=CompletionRejectionAction.CONTINUE,
                max_attempts=3,
                max_repeated_gap_count=3,
            )
        )
        original_continue = CayuApp._continue_verified_task
        original_settle = type(tasks).settle_work_attempt_lifecycle
        predecessor = None
        application = None
        lost_reply = False
        restart = boundary == "restart"
        successor = None
        expected_stop = None

        def make_app():
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
            app.register_completion_verifier(contract.verifier, verifier)
            return app

        async def delay_continuation(app, admission_id, decision_id, **kwargs):
            nonlocal predecessor, application, successor, expected_stop
            predecessor = await tasks.load_work_attempt_admission(admission_id)
            from cayu.runtime._verified_task_decision_coordinator import verified_task_operation_id

            application = await tasks.load_completion_decision_application_receipt(
                predecessor.task_id,
                verified_task_operation_id("application", admission_id, decision_id),
            )
            assert application.task.status is TaskStatus.RUNNING
            proposal = await tasks.load_completion_proposal_for_attempt(predecessor.attempt_id)
            expected_stop = WorkAttemptLifecycleSettlement(
                settlement_id=verified_task_operation_id("settlement", admission_id),
                task_id=predecessor.task_id,
                admission_id=admission_id,
                expected_admission_sha256=work_attempt_admission_authority_sha256(predecessor),
                release_evidence=await app._session_engine.load_work_attempt_release_evidence(
                    predecessor
                ),
                kind="continuation_deadline_stop",
                proposal_id=proposal.proposal_id,
                proposal_request_sha256=proposal.request_sha256,
                decision_id=decision_id,
                application_idempotency_key=application.idempotency_key,
                stop_reason="work_contract_elapsed_limit",
            )
            assert not predecessor.run_semantics.deadline.expired
            with pytest.raises(WorkAttemptAdmissionConflict):
                await original_settle(tasks, expected_stop)
            assert await tasks.load_task(predecessor.task_id) == application.task
            if restart:
                raise ConnectionError("restart before successor admission")
            if boundary == "successor_ack_loss":
                successor = await original_continue(app, admission_id, decision_id, **kwargs)
            await _wait_past(predecessor.run_semantics.deadline_expires_at)
            if boundary in {"successor_ack_loss", "other_failure"}:
                raise ConnectionError("continuation acknowledgement unavailable")
            return await original_continue(app, admission_id, decision_id, **kwargs)

        async def lose_stop_reply(store, request):
            nonlocal lost_reply
            if request.kind == "continuation_deadline_stop" and not lost_reply:
                before = await store.load_task(request.task_id)
                for field, value in (
                    ("decision_id", "another-decision"),
                    ("application_idempotency_key", "another-application"),
                    ("proposal_id", "another-proposal"),
                    ("proposal_request_sha256", "0" * 64),
                    ("expected_admission_sha256", "0" * 64),
                ):
                    with pytest.raises(WorkAttemptAdmissionConflict):
                        await original_settle(store, request.model_copy(update={field: value}))
                    assert await store.load_task(request.task_id) == before
                    assert (
                        await store.load_work_attempt_lifecycle_receipt(request.admission_id)
                        is None
                    )
            receipt = await original_settle(store, request)
            if request.kind == "continuation_deadline_stop" and not lost_reply:
                lost_reply = True
                raise ConnectionError("continuation stop acknowledgement lost")
            return receipt

        try:
            await tasks.publish_work_contract(contract)
            task = await tasks.create_task(
                TaskCreate(type="work", work_contract=contract.reference())
            )
            handler = _StaticHandler()
            monkeypatch.setattr(type(tasks), "settle_work_attempt_lifecycle", lose_stop_reply)
            with monkeypatch.context() as patch:
                patch.setattr(CayuApp, "_continue_verified_task", delay_continuation)
                async with VerifiedTaskWorker(
                    make_app(), handler, worker_id="original", max_elapsed_seconds=5
                ) as worker:
                    if restart:
                        with pytest.raises(ConnectionError, match="restart before successor"):
                            await worker.run(max_tasks=1)
                    elif boundary == "successor_ack_loss":
                        with pytest.raises(ExceptionGroup) as conflict:
                            await worker.run(max_tasks=1)
                        from cayu._exception_groups import iter_exception_tree

                        assert any(
                            isinstance(error, ConnectionError)
                            for error in iter_exception_tree(conflict.value)
                        )
                        # Runtime release ownership rejects the stale epoch
                        # before the store mutation is even dispatched.
                        assert any(
                            isinstance(error, RuntimeError) and "SessionRunFenced" in str(error)
                            for error in iter_exception_tree(conflict.value)
                        )
                    elif boundary == "other_failure":
                        with pytest.raises(ConnectionError, match="continuation acknowledgement"):
                            await worker.run(max_tasks=1)
                    else:
                        assert await asyncio.wait_for(worker.run(max_tasks=1), 30) == 1
            assert predecessor is not None and application is not None
            if boundary == "successor_ack_loss":
                assert successor is not None
                assert expected_stop is not None
                with pytest.raises(WorkAttemptAdmissionConflict):
                    await original_settle(tasks, expected_stop)
                assert await tasks.load_latest_work_attempt_admission(task.id) == successor
                assert (
                    await tasks.load_work_attempt_lifecycle_receipt(predecessor.admission_id)
                    is None
                )
                assert (await tasks.load_task(task.id)).status is TaskStatus.RUNNING
                assert len(provider.requests) == len(verifier.requests) == 1
                return
            if restart:
                if backend != "memory":
                    await sessions.close()
                    await tasks.close()
                    sessions, tasks = verified_worker_store_factory()
                await _wait_past(predecessor.run_semantics.deadline_expires_at)
                async with VerifiedTaskWorker(
                    make_app(), handler, worker_id="replacement"
                ) as worker:
                    assert await asyncio.wait_for(worker.run(max_tasks=1), 30) == 1
            assert lost_reply
            receipt = await tasks.load_work_attempt_lifecycle_receipt(predecessor.admission_id)
            assert receipt.request.kind == "continuation_deadline_stop"
            assert receipt.task.status is TaskStatus.NEEDS_ATTENTION
            assert receipt.task.status_reason == "work_contract_elapsed_limit"
            assert receipt.task == await tasks.load_task(task.id)
            assert not receipt.retired_contract_binding
            assert await tasks.load_latest_work_attempt_admission(task.id) == predecessor
            assert (
                await tasks.load_completion_decision_application_receipt(
                    task.id, application.idempotency_key
                )
                == application
            )
            assert len(provider.requests) == len(verifier.requests) == 1
            assert len(handler.preparations) == len(handler.proposals) == 1
            assert await tasks.settle_work_attempt_lifecycle(receipt.request) == receipt
            for field, value in (
                ("decision_id", "another-decision"),
                ("application_idempotency_key", "another-application"),
                ("proposal_request_sha256", "0" * 64),
            ):
                with pytest.raises(WorkAttemptAdmissionConflict):
                    await tasks.settle_work_attempt_lifecycle(
                        receipt.request.model_copy(update={field: value})
                    )
            assert await tasks.list_unsettled_work_attempt_admissions() == []
        finally:
            if backend != "memory":
                await sessions.close()
                await tasks.close()

    asyncio.run(scenario())
