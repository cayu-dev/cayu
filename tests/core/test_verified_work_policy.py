"""The same verdict/replay contract runs through all three backend boundaries."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_verified_work_contracts import assert_completion_rejection_policy
from tests.core.test_verified_work_store_hardening import _contract, _prepare_decision

from cayu import (
    CompletionDecisionApplicationRequest,
    CompletionVerdict,
    InMemoryTaskStore,
    SQLiteTaskStore,
    TaskStatus,
    TaskTerminalizationConflict,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_completion_decision_policy_conformance(backend, request):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def exercise():
        if dsn is not None:
            from tests.core.test_postgres_task_store import _new_store, _truncate

            await _truncate(dsn)

            def factory():
                return _new_store(dsn)
        elif backend == "sqlite":

            def factory():
                return SQLiteTaskStore(":memory:")
        else:
            factory = InMemoryTaskStore
        store = factory()
        try:
            for verdict, status in (
                (CompletionVerdict.ACCEPTED, TaskStatus.COMPLETED),
                (CompletionVerdict.BLOCKED, TaskStatus.BLOCKED),
                (CompletionVerdict.NEEDS_REVIEW, TaskStatus.NEEDS_ATTENTION),
                (CompletionVerdict.REJECTED, TaskStatus.PAUSED),
            ):
                contract = _contract(contract_id=f"policy-{verdict.value}")
                await store.publish_work_contract(contract)
                task, _, proposal, _, decision, result = await _prepare_decision(
                    store, contract, verdict=verdict, suffix=f"policy-{verdict.value}"
                )
                accepted = verdict is CompletionVerdict.ACCEPTED
                application = CompletionDecisionApplicationRequest(
                    task_id=task.id,
                    decision_id=decision.decision_id,
                    idempotency_key=f"apply-{verdict.value}",
                    result=result if accepted else None,
                    result_reference=proposal.result if accepted else None,
                )
                applied = await store.apply_completion_decision(application)
                assert applied.status is status
                assert applied.worker_id is None and applied.lease_expires_at is None
                if accepted:
                    assert applied.retry_series is None
                    assert applied.interrupted_handoff_id is None
                    assert applied.error is None and applied.status_payload is None
                receipt = await store.load_completion_decision_application_receipt(
                    task.id, application.idempotency_key
                )
                assert receipt is not None and receipt.task == applied
                assert receipt.applied_at == applied.updated_at
                assert await store.apply_completion_decision(application) == applied
                assert (
                    await store.load_completion_decision_application_receipt(
                        task.id, application.idempotency_key
                    )
                    == receipt
                )
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()
        await assert_completion_rejection_policy(factory)

    asyncio.run(exercise())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("verdict", list(CompletionVerdict))
def test_completion_decision_preserves_draining_cancellation(backend, verdict, request):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def exercise():
        if dsn is not None:
            from tests.core.test_postgres_task_store import _new_store, _truncate

            await _truncate(dsn)
            store = _new_store(dsn)
        elif backend == "sqlite":
            store = SQLiteTaskStore(":memory:")
        else:
            store = InMemoryTaskStore()
        try:
            contract = _contract(contract_id="draining-cancellation")
            await store.publish_work_contract(contract)
            task, _, proposal, _, decision, result = await _prepare_decision(
                store,
                contract,
                verdict=verdict,
                suffix="draining-cancellation",
                task_worker_id="task-worker",
            )
            cancelling = await store.cancel_task(task.id, error={"reason": "operator cancelled"})
            assert cancelling.status is TaskStatus.RUNNING
            assert cancelling.status_reason == "cancellation_requested"
            assert cancelling.worker_id == task.worker_id
            assert cancelling.lease_expires_at == task.lease_expires_at
            accepted = verdict is CompletionVerdict.ACCEPTED
            application = CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=decision.decision_id,
                idempotency_key="apply-during-cancellation",
                result=result if accepted else None,
                result_reference=proposal.result if accepted else None,
            )
            for _ in range(2):
                with pytest.raises(
                    TaskTerminalizationConflict, match="cancellation is still draining"
                ):
                    await store.apply_completion_decision(application)
                assert await store.load_task(task.id) == cancelling
                assert (
                    await store.load_completion_decision_application_receipt(
                        task.id, application.idempotency_key
                    )
                    is None
                )
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    asyncio.run(exercise())
