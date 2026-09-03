from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

import pytest
from tests.core.task_invocation_fixtures import task_backed_session_invocation

from cayu import (
    ResolutionActor,
    ResolutionActorSource,
    Task,
    TaskCancellationReconciliationConflict,
    TaskCancellationReconciliationEventType,
    TaskCancellationReconciliationEvidence,
    TaskCancellationReconciliationOutcome,
    TaskCancellationReconciliationRejected,
    TaskCancellationReconciliationRequest,
    TaskCancellationReconciliationResult,
    TaskClaimLost,
    TaskCreate,
    TaskStatus,
    TaskStore,
    TaskTerminalizationConflict,
    TaskTerminalizationRequest,
    TaskTerminalizationRetryPolicy,
    TaskTerminalizationUncertain,
    TaskTerminalKind,
    interrupted_task_handoff_request,
    terminalize_task_with_retry,
)


def ordinary_cancellation_reconciliation_request(
    task: Task,
    *,
    reconciliation_idempotency_key: str = "ordinary-reconciliation-1",
    outcome: TaskCancellationReconciliationOutcome = (
        TaskCancellationReconciliationOutcome.QUIESCENT
    ),
    evidence_sha256: str = "a" * 64,
) -> TaskCancellationReconciliationRequest:
    assert task.retry_series is None
    assert task.worker_id is not None
    assert task.lease_expires_at is not None
    assert task.status_payload is not None
    cancellation_idempotency_key = task.status_payload["terminalization_idempotency_key"]
    event = task.status_payload["event"]
    assert isinstance(cancellation_idempotency_key, str)
    assert isinstance(event, dict)
    cancellation_requested_at = event["occurred_at"]
    assert isinstance(cancellation_requested_at, str)
    reconciliation_requested_at = task.updated_at
    return TaskCancellationReconciliationRequest(
        task_id=task.id,
        original_worker_id=task.worker_id,
        original_handoff_id=task.interrupted_handoff_id,
        original_lease_expires_at=task.lease_expires_at,
        cancellation_requested_at=datetime.fromisoformat(cancellation_requested_at),
        cancellation_idempotency_key=cancellation_idempotency_key,
        reconciliation_idempotency_key=reconciliation_idempotency_key,
        reconciliation_requested_at=reconciliation_requested_at,
        reconciled_by=ResolutionActor(
            subject="operator:ordinary-reconciler",
            tenant="tenant-a",
            source=ResolutionActorSource.REQUEST,
            claims={"role": "operator"},
        ),
        evidence=TaskCancellationReconciliationEvidence(
            outcome=outcome,
            validator_id="runtime.effect-receipt",
            validator_version="1",
            evidence_id="ordinary-effect-receipt-1",
            evidence_sha256=evidence_sha256,
            validated_at=reconciliation_requested_at,
            execution_profile_fingerprint="b" * 64,
            effect_fingerprint="c" * 64,
        ),
        expected_execution_profile_fingerprint="b" * 64,
        expected_effect_fingerprint="c" * 64,
    )


async def assert_recovered_continuation_terminalization_conformance(
    store: TaskStore,
) -> None:
    """Prove every ordinary terminal kind consumes recovery lineage."""

    for kind in TaskTerminalKind:
        suffix = kind.value
        task_id = f"recovered_terminal_{suffix}"
        worker_id = f"prior_{suffix}"
        session_id = f"recovered_terminal_session_{suffix}"
        await store.create_task(TaskCreate(task_id=task_id, type="review"))
        claimed = await store.claim_task(worker_id, lease_seconds=60)
        assert claimed is not None
        attached = await store.attach_task(
            task_id,
            session_id=session_id,
            session_invocation=await task_backed_session_invocation(
                store,
                task_id,
                session_id,
            ),
            worker_id=worker_id,
            lease_expires_at=claimed.lease_expires_at,
        )
        await store.release_interrupted_task_worker(
            interrupted_task_handoff_request(attached, session_run_epoch=1)
        )
        recovery_worker = f"recovery_{suffix}"
        recovered = (
            await store.claim_interrupted_task_continuation(
                recovery_worker,
                handoff_id=str(uuid4()),
                lease_seconds=60,
            )
        ).task
        assert recovered is not None
        assert recovered.interrupted_handoff_id is not None
        idempotency_key = f"recovered-terminal-{suffix}"
        if kind is TaskTerminalKind.CANCELLED:
            recovered = await store.cancel_task(task_id, {"outcome": suffix})
            assert recovered.status_payload is not None
            stored_key = recovered.status_payload["terminalization_idempotency_key"]
            assert isinstance(stored_key, str)
            idempotency_key = stored_key
        request = TaskTerminalizationRequest(
            task_id=task_id,
            worker_id=recovery_worker,
            lease_expires_at=recovered.lease_expires_at,
            handoff_id=recovered.interrupted_handoff_id,
            kind=kind,
            result={"outcome": suffix} if kind is TaskTerminalKind.COMPLETED else None,
            error=None if kind is TaskTerminalKind.COMPLETED else {"outcome": suffix},
            idempotency_key=idempotency_key,
        )
        terminal = await store.terminalize_task(request)
        assert terminal.status is TaskStatus(kind.value)
        assert terminal.worker_id is None
        assert terminal.lease_expires_at is None
        assert terminal.interrupted_handoff_id is None
        receipt = await store.load_task_terminalization_receipt(
            task_id,
            request.idempotency_key,
        )
        assert receipt is not None
        assert receipt.task == terminal


async def assert_attached_task_recovery_terminalization_conformance(
    store: TaskStore,
) -> None:
    """Prove exact session/lease fencing and durable replay for crash recovery."""

    assert store.supports_attached_task_recovery_terminalization is True
    await store.create_task(TaskCreate(task_id="attached-recovery", type="review"))
    claimed = await store.claim_task("attached-recovery-worker", lease_seconds=1)
    assert claimed is not None
    assert claimed.lease_expires_at is not None
    attached = await store.attach_task(
        claimed.id,
        session_id="attached-recovery-session",
        session_invocation=await task_backed_session_invocation(
            store,
            claimed.id,
            "attached-recovery-session",
        ),
        worker_id=claimed.worker_id,
        lease_expires_at=claimed.lease_expires_at,
    )
    assert attached.session_instance_id is not None
    request = TaskTerminalizationRequest(
        task_id=attached.id,
        worker_id=claimed.worker_id,
        lease_expires_at=claimed.lease_expires_at,
        kind=TaskTerminalKind.FAILED,
        error={"code": "session_recovery"},
        idempotency_key="attached-recovery-failure",
    )

    with pytest.raises(TaskClaimLost, match="lease is still active"):
        await store.recover_attached_task_failure(
            request,
            session_id=attached.session_id,
            session_instance_id=attached.session_instance_id,
        )
    await asyncio.sleep(1.05)
    terminal = await store.recover_attached_task_failure(
        request,
        session_id=attached.session_id,
        session_instance_id=attached.session_instance_id,
    )
    replayed = await store.recover_attached_task_failure(
        request,
        session_id=attached.session_id,
        session_instance_id=attached.session_instance_id,
    )
    receipt = await store.load_task_terminalization_receipt(
        request.task_id,
        request.idempotency_key,
    )

    assert terminal.status is TaskStatus.FAILED
    assert terminal.error == {"code": "session_recovery"}
    assert terminal.worker_id is None
    assert terminal.lease_expires_at is None
    assert replayed == terminal
    assert receipt is not None
    assert receipt.task == terminal
    with pytest.raises(TaskClaimLost, match="session incarnation"):
        await store.recover_attached_task_failure(
            request,
            session_id="different-session",
            session_instance_id=attached.session_instance_id,
        )


async def assert_owner_lost_ordinary_cancellation_reconciliation_conformance(
    store: TaskStore,
) -> None:
    """Exercise evidence fencing, durable replay, and both terminalization races."""

    assert store.supports_task_cancellation_reconciliation is True
    assert store.supports_idempotent_terminalization is True
    await store.create_task(
        TaskCreate(
            task_id="ordinary_owner_lost",
            type="review",
            metadata={
                "execution_profile_fingerprint": "b" * 64,
                "effect_fingerprint": "c" * 64,
            },
        )
    )
    claimed = await store.claim_task("ordinary-lost-worker", lease_seconds=1)
    assert claimed is not None
    requested = await store.cancel_task(claimed.id, {"code": "operator"})
    stale_request = ordinary_cancellation_reconciliation_request(requested)
    renewed = await store.heartbeat(
        claimed.id,
        "ordinary-lost-worker",
        lease_expires_at=claimed.lease_expires_at,
        extend_seconds=1,
    )
    assert renewed is not None
    assert renewed.lease_expires_at != requested.lease_expires_at
    assert renewed.status_payload is not None
    assert requested.status_payload is not None
    assert renewed.status_payload["event"] == requested.status_payload["event"]
    request = ordinary_cancellation_reconciliation_request(renewed)

    with pytest.raises(
        TaskCancellationReconciliationConflict,
        match="identity is stale",
    ):
        await store.reconcile_task_cancellation(stale_request)

    with pytest.raises(
        TaskCancellationReconciliationConflict,
        match="lease is still active",
    ) as active_conflict:
        await store.reconcile_task_cancellation(request)
    assert active_conflict.value.event.type is TaskCancellationReconciliationEventType.CONFLICT
    assert await store.load_task(request.task_id) == renewed

    await asyncio.sleep(1.05)
    assert await store.claim_task("replacement-before-evidence") is None
    expired_but_unsettled = await store.load_task(request.task_id)
    assert expired_but_unsettled == renewed

    mismatched_profile = request.model_copy(
        update={
            "evidence": request.evidence.model_copy(
                update={"execution_profile_fingerprint": "d" * 64}
            ),
            "expected_execution_profile_fingerprint": "d" * 64,
        }
    )
    with pytest.raises(
        TaskCancellationReconciliationConflict,
        match="stored execution_profile_fingerprint",
    ):
        await store.reconcile_task_cancellation(mismatched_profile)

    result = await store.reconcile_task_cancellation(request)
    replayed = await store.reconcile_task_cancellation(request)
    receipt = await store.load_task_terminalization_receipt(
        request.task_id,
        request.cancellation_idempotency_key,
    )

    assert replayed == result
    assert receipt == result.terminalization_receipt
    assert result.task.status is TaskStatus.CANCELLED
    assert result.task.worker_id is None
    assert result.task.lease_expires_at is None
    assert result.task.error == {"code": "operator"}
    assert result.task.status_payload == {
        "cancellation_reconciliation": result.reconciliation.model_dump(
            mode="json",
            warnings=False,
        )
    }
    assert result.reconciliation.reconciled_by.claims == {}
    assert [event.type for event in result.reconciliation.events] == [
        TaskCancellationReconciliationEventType.CANCELLATION_REQUESTED,
        TaskCancellationReconciliationEventType.STARTED,
        TaskCancellationReconciliationEventType.RECONCILED,
    ]
    assert all("operator" not in event.model_dump_json() for event in result.reconciliation.events)
    assert (
        TaskCancellationReconciliationResult.model_validate_json(result.model_dump_json()) == result
    )
    tampered = result.model_dump(mode="json")
    terminalization_receipt = tampered["terminalization_receipt"]
    assert isinstance(terminalization_receipt, dict)
    terminalization_receipt["request_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="terminal receipt"):
        TaskCancellationReconciliationResult.model_validate(tampered)

    late_worker = await store.terminalize_task(
        TaskTerminalizationRequest(
            task_id=request.task_id,
            worker_id=request.original_worker_id,
            lease_expires_at=request.original_lease_expires_at,
            kind=TaskTerminalKind.CANCELLED,
            error={"code": "operator"},
            idempotency_key=request.cancellation_idempotency_key,
        )
    )
    assert late_worker == result.task

    changed = request.model_copy(
        update={"evidence": request.evidence.model_copy(update={"evidence_sha256": "d" * 64})}
    )
    with pytest.raises(TaskCancellationReconciliationConflict, match="another intent"):
        await store.reconcile_task_cancellation(changed)

    await store.create_task(
        TaskCreate(
            task_id="ordinary_recovery_owner_lost",
            type="review",
            metadata={
                "execution_profile_fingerprint": "b" * 64,
                "effect_fingerprint": "c" * 64,
            },
        )
    )
    prior_owner = await store.claim_task("ordinary-prior-recovery-worker", lease_seconds=60)
    assert prior_owner is not None
    attached = await store.attach_task(
        prior_owner.id,
        session_id="ordinary-recovery-owner-lost-session",
        session_invocation=await task_backed_session_invocation(
            store,
            prior_owner.id,
            "ordinary-recovery-owner-lost-session",
        ),
        worker_id="ordinary-prior-recovery-worker",
        lease_expires_at=prior_owner.lease_expires_at,
    )
    await store.release_interrupted_task_worker(
        interrupted_task_handoff_request(attached, session_run_epoch=1)
    )
    recovery_owner = (
        await store.claim_interrupted_task_continuation(
            "ordinary-recovery-lost-worker",
            handoff_id=str(uuid4()),
            lease_seconds=1,
        )
    ).task
    assert recovery_owner is not None
    assert recovery_owner.interrupted_handoff_id is not None
    recovery_requested = await store.cancel_task(
        recovery_owner.id,
        {"code": "operator"},
    )
    recovery_request = ordinary_cancellation_reconciliation_request(
        recovery_requested,
        reconciliation_idempotency_key="ordinary-recovery-reconciliation",
    )
    await asyncio.sleep(1.05)
    recovery_result = await store.reconcile_task_cancellation(recovery_request)
    assert recovery_result.task.status is TaskStatus.CANCELLED
    assert recovery_result.task.interrupted_handoff_id is None
    assert recovery_result.terminalization_receipt.task == recovery_result.task

    await store.create_task(TaskCreate(task_id="ordinary_worker_wins", type="review"))
    worker_claim = await store.claim_task("ordinary-winning-worker", lease_seconds=60)
    assert worker_claim is not None
    worker_requested = await store.cancel_task(worker_claim.id, {"code": "operator"})
    losing_reconciliation = ordinary_cancellation_reconciliation_request(
        worker_requested,
        reconciliation_idempotency_key="ordinary-reconciliation-worker-lost",
    )
    worker_terminal = await store.terminalize_task(
        TaskTerminalizationRequest(
            task_id=worker_requested.id,
            worker_id="ordinary-winning-worker",
            lease_expires_at=worker_requested.lease_expires_at,
            kind=TaskTerminalKind.CANCELLED,
            error={"code": "operator"},
            idempotency_key=losing_reconciliation.cancellation_idempotency_key,
        )
    )
    with pytest.raises(
        TaskCancellationReconciliationConflict,
        match="without reconciliation evidence",
    ):
        await store.reconcile_task_cancellation(losing_reconciliation)
    assert await store.load_task(worker_requested.id) == worker_terminal

    await store.create_task(TaskCreate(task_id="ordinary_rejected", type="review"))
    rejected_claim = await store.claim_task("ordinary-rejected-worker", lease_seconds=60)
    assert rejected_claim is not None
    rejected_task = await store.cancel_task(rejected_claim.id, {"code": "operator"})
    rejected_request = ordinary_cancellation_reconciliation_request(
        rejected_task,
        reconciliation_idempotency_key="ordinary-reconciliation-rejected",
        outcome=TaskCancellationReconciliationOutcome.UNRESOLVED,
    )
    with pytest.raises(TaskCancellationReconciliationRejected) as first_rejection:
        await store.reconcile_task_cancellation(rejected_request)
    with pytest.raises(TaskCancellationReconciliationRejected) as replayed_rejection:
        await store.reconcile_task_cancellation(rejected_request)
    assert replayed_rejection.value.event == first_rejection.value.event
    assert await store.load_task(rejected_task.id) == rejected_task
    with pytest.raises(TaskCancellationReconciliationConflict, match="another request"):
        await store.reconcile_task_cancellation(
            rejected_request.model_copy(
                update={
                    "evidence": rejected_request.evidence.model_copy(
                        update={"evidence_sha256": "e" * 64}
                    )
                }
            )
        )


async def assert_live_ordinary_cancellation_conformance(store: TaskStore) -> None:
    """Exercise request fencing, cancellation precedence, and receipt replay."""

    await store.create_task(TaskCreate(task_id="task_live_cancel", type="review"))
    claimed = await store.claim_task("worker_live_cancel")
    assert claimed is not None
    requested = await store.cancel_task(
        claimed.id,
        {"reason": "operator cancelled live work"},
    )
    assert requested.status is TaskStatus.CLAIMED
    assert requested.status_reason == "cancellation_requested"
    assert requested.worker_id == claimed.worker_id
    assert requested.lease_expires_at == claimed.lease_expires_at
    assert requested.status_payload is not None

    with pytest.raises(TaskTerminalizationConflict):
        await store.terminalize_task(
            TaskTerminalizationRequest(
                task_id=requested.id,
                worker_id="worker_live_cancel",
                lease_expires_at=requested.lease_expires_at,
                kind=TaskTerminalKind.COMPLETED,
                result={"summary": "late completion"},
                idempotency_key="late-completion",
            )
        )

    cancellation = TaskTerminalizationRequest(
        task_id=requested.id,
        worker_id="worker_live_cancel",
        lease_expires_at=requested.lease_expires_at,
        kind=TaskTerminalKind.CANCELLED,
        error={"reason": "operator cancelled live work"},
        idempotency_key=requested.status_payload["terminalization_idempotency_key"],
    )
    terminal = await store.terminalize_task(cancellation)
    replayed = await store.terminalize_task(cancellation)
    receipt = await store.load_task_terminalization_receipt(
        cancellation.task_id,
        cancellation.idempotency_key,
    )

    assert terminal.status is TaskStatus.CANCELLED
    assert terminal.worker_id is None
    assert terminal.lease_expires_at is None
    assert terminal.error == {"reason": "operator cancelled live work"}
    assert replayed == terminal
    assert receipt is not None
    assert receipt.kind is TaskTerminalKind.CANCELLED
    assert receipt.task == terminal


async def assert_task_terminalization_acknowledgement_conformance(
    store: TaskStore,
) -> None:
    """Exercise acknowledgement loss, bounded exhaustion, and cancellation."""

    await store.create_task(TaskCreate(task_id="task_commit_ack", type="review"))
    commit_claim = await store.claim_task("worker_a")
    assert commit_claim is not None
    commit_request = TaskTerminalizationRequest(
        task_id="task_commit_ack",
        worker_id="worker_a",
        lease_expires_at=commit_claim.lease_expires_at,
        kind=TaskTerminalKind.COMPLETED,
        result={"summary": "done"},
        idempotency_key="commit-ack",
    )
    terminalize = store.terminalize_task
    terminalize_calls = 0

    async def commit_then_raise(request: TaskTerminalizationRequest) -> Task:
        nonlocal terminalize_calls
        terminalize_calls += 1
        await terminalize(request)
        raise ConnectionError("acknowledgement lost")

    store.terminalize_task = commit_then_raise  # type: ignore[method-assign]
    reconciled = await terminalize_task_with_retry(
        store,
        commit_request,
        policy=TaskTerminalizationRetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
        ),
    )
    assert reconciled.receipt_reconciled is True
    assert reconciled.attempt_count == 1
    assert terminalize_calls == 1

    store.terminalize_task = terminalize  # type: ignore[method-assign]
    await store.create_task(TaskCreate(task_id="task_precommit_ack", type="review"))
    precommit_claim = await store.claim_task("worker_b")
    assert precommit_claim is not None
    precommit_calls = 0

    async def fail_before_once(request: TaskTerminalizationRequest) -> Task:
        nonlocal precommit_calls
        precommit_calls += 1
        if precommit_calls == 1:
            raise ConnectionError("write unavailable")
        return await terminalize(request)

    store.terminalize_task = fail_before_once  # type: ignore[method-assign]
    retried = await terminalize_task_with_retry(
        store,
        TaskTerminalizationRequest(
            task_id="task_precommit_ack",
            worker_id="worker_b",
            lease_expires_at=precommit_claim.lease_expires_at,
            kind=TaskTerminalKind.FAILED,
            error={"message": "failed"},
            idempotency_key="precommit-ack",
        ),
        policy=TaskTerminalizationRetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
        ),
    )
    assert retried.receipt_reconciled is False
    assert retried.attempt_count == 2
    assert precommit_calls == 2

    store.terminalize_task = terminalize  # type: ignore[method-assign]
    await store.create_task(TaskCreate(task_id="task_repeated_ack", type="review"))
    repeated_claim = await store.claim_task("worker_c")
    assert repeated_claim is not None
    repeated_calls = 0
    load_receipt = store.load_task_terminalization_receipt

    async def repeat_commit_ack_loss(request: TaskTerminalizationRequest) -> Task:
        nonlocal repeated_calls
        repeated_calls += 1
        await terminalize(request)
        raise ConnectionError("write acknowledgement unavailable")

    async def receipt_ack_loss(task_id: str, idempotency_key: str):
        del task_id, idempotency_key
        raise TimeoutError("receipt acknowledgement unavailable")

    store.terminalize_task = repeat_commit_ack_loss  # type: ignore[method-assign]
    store.load_task_terminalization_receipt = receipt_ack_loss  # type: ignore[method-assign]
    with pytest.raises(TaskTerminalizationUncertain) as captured:
        await terminalize_task_with_retry(
            store,
            TaskTerminalizationRequest(
                task_id="task_repeated_ack",
                worker_id="worker_c",
                lease_expires_at=repeated_claim.lease_expires_at,
                kind=TaskTerminalKind.COMPLETED,
                result={"summary": "done"},
                idempotency_key="repeated-ack",
            ),
            policy=TaskTerminalizationRetryPolicy(
                max_attempts=2,
                initial_backoff_seconds=0,
                max_backoff_seconds=0,
            ),
        )
    assert captured.value.attempt_count == 2
    assert captured.value.error_category == "timeout"
    assert "write acknowledgement unavailable" not in str(captured.value)
    assert "receipt acknowledgement unavailable" not in str(captured.value)
    assert repeated_calls == 2
    store.load_task_terminalization_receipt = load_receipt  # type: ignore[method-assign]
    receipt = await store.load_task_terminalization_receipt("task_repeated_ack", "repeated-ack")
    assert receipt is not None

    store.terminalize_task = terminalize  # type: ignore[method-assign]
    await store.create_task(TaskCreate(task_id="task_cancelled_ack", type="review"))
    cancelled_claim = await store.claim_task("worker_d")
    assert cancelled_claim is not None
    cancellation_calls = 0

    async def cancel_terminalization(request: TaskTerminalizationRequest) -> Task:
        del request
        nonlocal cancellation_calls
        cancellation_calls += 1
        raise asyncio.CancelledError

    store.terminalize_task = cancel_terminalization  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        await terminalize_task_with_retry(
            store,
            TaskTerminalizationRequest(
                task_id="task_cancelled_ack",
                worker_id="worker_d",
                lease_expires_at=cancelled_claim.lease_expires_at,
                kind=TaskTerminalKind.COMPLETED,
                result={"summary": "done"},
                idempotency_key="cancelled-ack",
            ),
            policy=TaskTerminalizationRetryPolicy(
                max_attempts=2,
                initial_backoff_seconds=0,
                max_backoff_seconds=0,
            ),
        )
    assert cancellation_calls == 1
    store.terminalize_task = terminalize  # type: ignore[method-assign]
