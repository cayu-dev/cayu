from __future__ import annotations

import asyncio

import pytest

from cayu import (
    Task,
    TaskCreate,
    TaskStore,
    TaskTerminalizationRequest,
    TaskTerminalizationRetryPolicy,
    TaskTerminalizationUncertain,
    TaskTerminalKind,
    terminalize_task_with_retry,
)


async def assert_task_terminalization_acknowledgement_conformance(
    store: TaskStore,
) -> None:
    """Exercise acknowledgement loss, bounded exhaustion, and cancellation."""

    await store.create_task(TaskCreate(task_id="task_commit_ack", type="review"))
    assert await store.claim_task("worker_a") is not None
    commit_request = TaskTerminalizationRequest(
        task_id="task_commit_ack",
        worker_id="worker_a",
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
    assert await store.claim_task("worker_b") is not None
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
    assert await store.claim_task("worker_c") is not None
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
    assert await store.claim_task("worker_d") is not None
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
