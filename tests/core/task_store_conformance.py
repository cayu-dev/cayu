from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import cast
from uuid import uuid4

import pytest
from tests.core.task_invocation_fixtures import (
    task_backed_session_invocation,
    unattributed_session_invocation_binding,
)

from cayu.sessions.invocation import TaskExecutionSource
from cayu.tasks.contracts import (
    CompletionResultResolverRef,
    CompletionVerifierRef,
    WorkContractDraft,
    WorkCriterion,
    work_contract_from_draft,
)
from cayu.tasks.creation import TaskCreate, task_create_with_runtime_invocation
from cayu.tasks.handoff import interrupted_task_handoff_request
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import Task, TaskClaimLost, TaskRetryPolicy, TaskStatus
from cayu.tasks.store import TaskStore
from cayu.tasks.terminalization import (
    TaskTerminalizationConflict,
    TaskTerminalizationRequest,
    TaskTerminalKind,
)


async def assert_task_contract_queue_filter_conformance(store: TaskStore) -> None:
    from tests.core.test_verified_work_contracts import _contract

    contract = await store.publish_work_contract(_contract(contract_id="queue-filter-contract"))
    ordinary = await store.create_task(TaskCreate(task_id="queue-ordinary", type="mixed"))
    contracted = await store.create_task(
        TaskCreate(task_id="queue-contracted", type="mixed", work_contract=contract.reference())
    )
    for selection, expected in (
        (None, {ordinary.id, contracted.id}),
        (False, {ordinary.id}),
        (True, {contracted.id}),
    ):
        tasks = await store.list_tasks(TaskQuery(type="mixed", has_work_contract=selection))
        assert {task.id for task in tasks} == expected

    claimed = await store.claim_task(
        "verified-worker", TaskQuery(type="mixed", has_work_contract=True)
    )
    assert claimed is not None and claimed.id == contracted.id
    assert (
        await store.claim_task("verified-peer", TaskQuery(type="mixed", has_work_contract=True))
        is None
    )
    claimed = await store.claim_task(
        "ordinary-worker", TaskQuery(type="mixed", has_work_contract=False)
    )
    assert claimed is not None and claimed.id == ordinary.id


async def assert_interrupted_continuation_scan_bound_conformance(
    store: TaskStore,
) -> None:
    """Apply claim filters after the bounded physical candidate page."""

    async def release(task_id: str, task_type: str) -> None:
        session_id = f"session-{task_id}"
        worker_id = f"prior-{task_id}"
        await store.create_task(TaskCreate(task_id=task_id, type=task_type))
        claimed = await store.claim_task(worker_id, TaskQuery(type=task_type))
        assert claimed is not None and claimed.id == task_id
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

    await release("continuation-scan-filtered", "other")
    await release("continuation-scan-target", "target")

    first = await store.claim_interrupted_task_continuation(
        "continuation-worker",
        TaskQuery(type="target"),
        handoff_id=str(uuid4()),
        scan_limit=1,
    )
    assert first.task is None
    assert first.scanned_candidates == 1
    assert first.filtered_candidates == 1
    assert first.rejected_candidates == 0
    assert first.next_after is not None
    assert not first.exhausted

    claim_handoff_id = str(uuid4())
    second = await store.claim_interrupted_task_continuation(
        "continuation-worker",
        TaskQuery(type="target"),
        handoff_id=claim_handoff_id,
        after=first.next_after,
        scan_limit=1,
    )
    assert second.task is not None
    assert second.task.id == "continuation-scan-target"
    assert second.scanned_candidates == 1
    assert second.filtered_candidates == 0
    assert second.rejected_candidates == 0
    assert not second.replayed

    replay = await store.claim_interrupted_task_continuation(
        "continuation-worker",
        TaskQuery(type="target"),
        handoff_id=claim_handoff_id,
    )
    assert replay.replayed
    assert replay.task == second.task
    with pytest.raises(TaskClaimLost):
        await store.claim_interrupted_task_continuation(
            "foreign-worker",
            TaskQuery(type="target"),
            handoff_id=claim_handoff_id,
        )

    await store.release_interrupted_task_worker(
        interrupted_task_handoff_request(second.task, session_run_epoch=2)
    )
    successor_handoff_id = str(uuid4())
    successor = await store.claim_interrupted_task_continuation(
        "continuation-worker",
        TaskQuery(type="target"),
        handoff_id=successor_handoff_id,
    )
    assert successor.task is not None
    await store.release_interrupted_task_worker(
        interrupted_task_handoff_request(successor.task, session_run_epoch=3)
    )
    with pytest.raises(TaskClaimLost):
        await store.claim_interrupted_task_continuation(
            "continuation-worker",
            TaskQuery(type="target"),
            handoff_id=claim_handoff_id,
        )
    current_handoff_id = str(uuid4())
    current = await store.claim_interrupted_task_continuation(
        "continuation-worker",
        TaskQuery(type="target"),
        handoff_id=current_handoff_id,
    )
    assert current.task is not None
    with pytest.raises(TaskClaimLost):
        await store.heartbeat(
            current.task.id,
            "continuation-worker",
            lease_expires_at=current.task.lease_expires_at,
            handoff_id=successor_handoff_id,
        )
    with pytest.raises(TaskClaimLost):
        await store.complete_task(
            current.task.id,
            {"authority": "stale"},
            worker_id="continuation-worker",
            lease_expires_at=current.task.lease_expires_at,
            handoff_id=successor_handoff_id,
        )
    with pytest.raises(TaskClaimLost):
        await store.fail_task(
            current.task.id,
            {"authority": "stale"},
            worker_id="continuation-worker",
            lease_expires_at=current.task.lease_expires_at,
            handoff_id=successor_handoff_id,
        )
    stale_terminalization = TaskTerminalizationRequest(
        task_id=current.task.id,
        worker_id="continuation-worker",
        lease_expires_at=current.task.lease_expires_at,
        handoff_id=successor_handoff_id,
        kind=TaskTerminalKind.COMPLETED,
        result={"authority": "stale"},
        idempotency_key="stale-continuation-generation",
    )
    with pytest.raises(TaskClaimLost):
        await store.terminalize_task(stale_terminalization)
    completed = await store.terminalize_task(
        stale_terminalization.model_copy(
            update={
                "handoff_id": current_handoff_id,
                "result": {"authority": "current"},
                "idempotency_key": "current-continuation-generation",
            }
        )
    )
    assert completed.status is TaskStatus.COMPLETED


async def assert_task_session_invocation_binding_conformance(store: TaskStore) -> None:
    """Reject a provenance value bound to a different immediate session ID."""

    task = await store.create_task(TaskCreate(task_id="session_binding_task", type="conformance"))
    binding = await task_backed_session_invocation(
        store,
        task.id,
        "bound_session",
    )

    with pytest.raises(ValueError, match="session identity conflicts"):
        await store.start_task(
            task.id,
            session_id="different_session",
            session_invocation=binding,
        )
    unchanged = await store.load_task(task.id)
    assert unchanged is not None
    assert unchanged.status is TaskStatus.PENDING
    assert unchanged.session_id is None

    mismatched_create = task_create_with_runtime_invocation(
        TaskCreate(
            task_id="session_binding_create",
            type="conformance",
            session_id="different_session",
        ),
        source=TaskExecutionSource.SDK_TASK,
        session_invocation=binding,
    )
    with pytest.raises(ValueError, match="session identity conflicts"):
        await store.create_task(mismatched_create)
    assert await store.load_task("session_binding_create") is None

    with pytest.raises(ValueError, match="session identity conflicts"):
        await store.create_running_task(
            TaskCreate(
                task_id="session_binding_running_create",
                type="conformance",
                session_id="running_session",
            ),
            session_invocation=unattributed_session_invocation_binding("different_session"),
        )
    assert await store.load_task("session_binding_running_create") is None

    prebound = await store.create_task(
        TaskCreate(
            task_id="session_binding_prebound_start",
            type="conformance",
            session_id="prebound_session",
        )
    )
    with pytest.raises(ValueError, match="binding is required"):
        await store.start_task(prebound.id)
    with pytest.raises(ValueError, match="already bound to a different session"):
        await store.start_task(
            prebound.id,
            session_id="replacement_session",
            session_invocation=unattributed_session_invocation_binding("replacement_session"),
        )
    unchanged_prebound = await store.load_task(prebound.id)
    assert unchanged_prebound is not None
    assert unchanged_prebound.status is TaskStatus.PENDING
    assert unchanged_prebound.session_id == "prebound_session"

    started_prebound = await store.start_task(
        prebound.id,
        session_invocation=await task_backed_session_invocation(
            store,
            prebound.id,
            "prebound_session",
        ),
    )
    assert started_prebound.status is TaskStatus.RUNNING
    assert started_prebound.session_id == "prebound_session"


async def assert_task_claim_lost_conformance(store: TaskStore) -> None:
    """Assert the public stale-worker contract shared by every task store."""

    await store.create_task(TaskCreate(task_id="claim_lost", type="conformance"))
    claimed = await store.claim_task("worker_a", lease_seconds=300)
    assert claimed is not None
    assert claimed.worker_id == "worker_a"
    assert claimed.lease_expires_at is not None

    with pytest.raises(TaskClaimLost, match="does not own|no longer owns"):
        await store.attach_task(
            claimed.id,
            session_id="claim_lost_session",
            session_invocation=await task_backed_session_invocation(
                store,
                claimed.id,
                "claim_lost_session",
            ),
            worker_id="worker_b",
            lease_expires_at=claimed.lease_expires_at,
        )
    with pytest.raises(TaskClaimLost, match="does not own|no longer owns"):
        await store.heartbeat(
            claimed.id,
            "worker_b",
            lease_expires_at=claimed.lease_expires_at,
        )
    with pytest.raises(TaskClaimLost, match="does not own|no longer owns"):
        await store.release_task(
            claimed.id,
            "worker_b",
            lease_expires_at=claimed.lease_expires_at,
        )
    with pytest.raises(TaskClaimLost, match="does not own|no longer owns"):
        await store.complete_task(
            claimed.id,
            {"ok": True},
            worker_id="worker_b",
            lease_expires_at=claimed.lease_expires_at,
        )
    with pytest.raises(TaskClaimLost, match="does not own|no longer owns"):
        await store.fail_task(
            claimed.id,
            {"message": "failed"},
            worker_id="worker_b",
            lease_expires_at=claimed.lease_expires_at,
        )

    attached = await store.attach_task(
        claimed.id,
        session_id="claim_lost_session",
        session_invocation=await task_backed_session_invocation(
            store,
            claimed.id,
            "claim_lost_session",
        ),
        worker_id="worker_a",
        lease_expires_at=claimed.lease_expires_at,
    )
    assert attached.status is TaskStatus.RUNNING
    with pytest.raises(TaskClaimLost, match="does not own|no longer owns"):
        await store.attach_task(
            attached.id,
            session_id="claim_lost_replacement_session",
            session_invocation=await task_backed_session_invocation(
                store,
                attached.id,
                "claim_lost_replacement_session",
            ),
            worker_id="worker_b",
            lease_expires_at=attached.lease_expires_at,
        )
    with pytest.raises(ValueError, match="already attached") as duplicate_attach:
        await store.attach_task(
            attached.id,
            session_id="claim_lost_duplicate_session",
            session_invocation=await task_backed_session_invocation(
                store,
                attached.id,
                "claim_lost_duplicate_session",
            ),
            worker_id="worker_a",
            lease_expires_at=attached.lease_expires_at,
        )
    assert type(duplicate_attach.value) is ValueError
    with pytest.raises(TaskClaimLost, match="does not own|no longer owns"):
        await store.release_attached_task_worker(
            attached.id,
            "worker_b",
            lease_expires_at=claimed.lease_expires_at,
        )

    unchanged = await store.load_task(attached.id)
    assert unchanged is not None
    assert unchanged.status is TaskStatus.RUNNING
    assert unchanged.worker_id == "worker_a"

    await store.create_task(TaskCreate(task_id="claim_lost_reclaimed", type="claim_lost_reclaimed"))
    stale_claim = await store.claim_task(
        "worker_a",
        TaskQuery(type="claim_lost_reclaimed"),
        lease_seconds=1,
    )
    assert stale_claim is not None
    await asyncio.sleep(1.05)
    with pytest.raises(TaskClaimLost, match="lease for worker worker_a has expired"):
        await store.attach_task(
            stale_claim.id,
            session_id="claim_lost_expired_session",
            session_invocation=await task_backed_session_invocation(
                store,
                stale_claim.id,
                "claim_lost_expired_session",
            ),
            worker_id="worker_a",
            lease_expires_at=stale_claim.lease_expires_at,
        )
    reclaimed = await store.reclaim_expired(
        query=TaskQuery(type="claim_lost_reclaimed"),
    )
    assert [task.id for task in reclaimed] == [stale_claim.id]
    with pytest.raises(TaskClaimLost, match="not claimed or running"):
        await store.release_task(
            stale_claim.id,
            "worker_a",
            lease_expires_at=stale_claim.lease_expires_at,
        )
    pending = await store.load_task(stale_claim.id)
    assert pending is not None
    assert pending.status is TaskStatus.PENDING
    assert pending.worker_id is None

    await store.create_task(
        TaskCreate(task_id="claim_lost_terminalized", type="claim_lost_terminalized")
    )
    terminal_claim = await store.claim_task(
        "worker_a",
        TaskQuery(type="claim_lost_terminalized"),
        lease_seconds=300,
    )
    assert terminal_claim is not None
    attached_terminal = await store.attach_task(
        terminal_claim.id,
        session_id="claim_lost_terminalized_session",
        session_invocation=await task_backed_session_invocation(
            store,
            terminal_claim.id,
            "claim_lost_terminalized_session",
        ),
        worker_id="worker_a",
        lease_expires_at=terminal_claim.lease_expires_at,
    )
    terminal = await store.complete_task(
        attached_terminal.id,
        {"winner": "control-plane"},
    )
    assert terminal.status is TaskStatus.COMPLETED
    with pytest.raises(TaskClaimLost, match="not claimed or running"):
        await store.release_attached_task_worker(
            attached_terminal.id,
            "worker_a",
            lease_expires_at=attached_terminal.lease_expires_at,
        )

    await store.create_task(
        TaskCreate(task_id="claim_lost_terminal_attach", type="claim_lost_terminal_attach")
    )
    terminal_attach_claim = await store.claim_task(
        "worker_a",
        TaskQuery(type="claim_lost_terminal_attach"),
        lease_seconds=300,
    )
    assert terminal_attach_claim is not None
    terminal_before_attach = await store.complete_task(
        terminal_attach_claim.id,
        {"winner": "control-plane"},
    )
    assert terminal_before_attach.status is TaskStatus.COMPLETED
    with pytest.raises(TaskClaimLost, match="not claimed by worker"):
        await store.attach_task(
            terminal_before_attach.id,
            session_id="claim_lost_too_late_session",
            session_invocation=await task_backed_session_invocation(
                store,
                terminal_before_attach.id,
                "claim_lost_too_late_session",
            ),
            worker_id="worker_a",
            lease_expires_at=terminal_attach_claim.lease_expires_at,
        )

    await store.create_task(
        TaskCreate(task_id="claim_lost_structural", type="claim_lost_structural")
    )
    structural_claim = await store.claim_task(
        "worker_a",
        TaskQuery(type="claim_lost_structural"),
        lease_seconds=300,
    )
    assert structural_claim is not None
    with pytest.raises(ValueError, match="not running") as unattached_release:
        await store.release_attached_task_worker(
            structural_claim.id,
            "worker_a",
            lease_expires_at=structural_claim.lease_expires_at,
        )
    assert type(unattached_release.value) is ValueError
    structural_attached = await store.attach_task(
        structural_claim.id,
        session_id="claim_lost_structural_session",
        session_invocation=await task_backed_session_invocation(
            store,
            structural_claim.id,
            "claim_lost_structural_session",
        ),
        worker_id="worker_a",
        lease_expires_at=structural_claim.lease_expires_at,
    )
    with pytest.raises(ValueError, match="already attached") as attached_release:
        await store.release_task(
            structural_attached.id,
            "worker_a",
            lease_expires_at=structural_attached.lease_expires_at,
        )
    assert type(attached_release.value) is ValueError


async def assert_worker_terminalization_generation_conformance(store: TaskStore) -> None:
    """Worker completion and failure require their exact live lease generation."""

    async def renewed_claim(task_id: str) -> tuple[Task, Task]:
        await store.create_task(TaskCreate(task_id=task_id, type=task_id))
        initial = await store.claim_task(
            "stable-worker",
            TaskQuery(type=task_id),
            lease_seconds=2,
        )
        assert initial is not None
        assert initial.lease_expires_at is not None
        renewed = await store.heartbeat(
            initial.id,
            "stable-worker",
            lease_expires_at=initial.lease_expires_at,
            extend_seconds=300,
        )
        assert renewed.lease_expires_at is not None
        assert renewed.lease_expires_at != initial.lease_expires_at
        return initial, renewed

    stale_completion, current_completion = await renewed_claim("worker_terminalization_completion")
    with pytest.raises(TaskClaimLost):
        await store.complete_task(
            stale_completion.id,
            {"winner": "stale"},
            worker_id="stable-worker",
            lease_expires_at=stale_completion.lease_expires_at,
        )
    with pytest.raises(TaskClaimLost, match="exact lease generation"):
        await store.complete_task(
            current_completion.id,
            {"winner": "missing-generation"},
            worker_id="stable-worker",
        )
    completed = await store.complete_task(
        current_completion.id,
        {"winner": "current"},
        worker_id="stable-worker",
        lease_expires_at=current_completion.lease_expires_at,
    )
    assert completed.status is TaskStatus.COMPLETED
    assert completed.result == {"winner": "current"}

    stale_failure, current_failure = await renewed_claim("worker_terminalization_failure")
    with pytest.raises(TaskClaimLost):
        await store.fail_task(
            stale_failure.id,
            {"winner": "stale"},
            worker_id="stable-worker",
            lease_expires_at=stale_failure.lease_expires_at,
        )
    failed = await store.fail_task(
        current_failure.id,
        {"winner": "current"},
        worker_id="stable-worker",
        lease_expires_at=current_failure.lease_expires_at,
    )
    assert failed.status is TaskStatus.FAILED
    assert failed.error == {"winner": "current"}


async def assert_exact_claimed_task_cancellation_conformance(store: TaskStore) -> None:
    """A stale cancellation request cannot mutate a renewed or replacement claim."""

    await store.create_task(
        TaskCreate(task_id="exact_cancellation_renewed", type="exact-cancellation")
    )
    initial = await store.claim_task(
        "worker-a",
        TaskQuery(type="exact-cancellation"),
        lease_seconds=2,
    )
    assert initial is not None
    assert initial.lease_expires_at is not None
    renewed = await store.heartbeat(
        initial.id,
        "worker-a",
        lease_expires_at=initial.lease_expires_at,
        extend_seconds=300,
    )
    assert renewed.lease_expires_at is not None
    assert renewed.lease_expires_at != initial.lease_expires_at

    with pytest.raises(TaskClaimLost, match="expected worker lease"):
        await store.request_claimed_task_cancellation(
            initial.id,
            "worker-a",
            initial.lease_expires_at,
            {"reason": "stale-renewal"},
        )
    unchanged = await store.load_task(initial.id)
    assert unchanged is not None
    assert unchanged.worker_id == "worker-a"
    assert unchanged.lease_expires_at == renewed.lease_expires_at
    assert unchanged.status_reason is None

    requested = await store.request_claimed_task_cancellation(
        renewed.id,
        "worker-a",
        renewed.lease_expires_at,
        {"reason": "current-owner"},
    )
    assert requested.worker_id == "worker-a"
    assert requested.lease_expires_at == renewed.lease_expires_at
    assert requested.status_reason == "cancellation_requested"

    with pytest.raises(TaskClaimLost, match="expected worker lease"):
        await store.request_claimed_task_cancellation(
            renewed.id,
            "worker-b",
            renewed.lease_expires_at,
            {"reason": "wrong-worker"},
        )
    requested_after_wrong_worker = await store.load_task(renewed.id)
    assert requested_after_wrong_worker == requested

    await store.create_task(TaskCreate(task_id="exact_cancellation_replay", type="exact-replay"))
    replay_claim = await store.claim_task(
        "worker-a",
        TaskQuery(type="exact-replay"),
        lease_seconds=1,
    )
    assert replay_claim is not None
    assert replay_claim.lease_expires_at is not None
    replay_marker = await store.request_claimed_task_cancellation(
        replay_claim.id,
        "worker-a",
        replay_claim.lease_expires_at,
        {"reason": "acknowledgement-lost"},
    )

    await store.create_task(
        TaskCreate(task_id="exact_cancellation_reclaimed", type="exact-reclaim")
    )
    stale = await store.claim_task(
        "worker-a",
        TaskQuery(type="exact-reclaim"),
        lease_seconds=1,
    )
    assert stale is not None
    assert stale.lease_expires_at is not None
    await asyncio.sleep(1.05)
    replayed = await store.request_claimed_task_cancellation(
        replay_claim.id,
        "worker-a",
        replay_claim.lease_expires_at,
        {"reason": "acknowledgement-lost"},
    )
    assert replayed == replay_marker
    reclaimed = await store.reclaim_expired(query=TaskQuery(type="exact-reclaim"))
    assert [task.id for task in reclaimed] == [stale.id]
    replacement = await store.claim_task(
        "worker-b",
        TaskQuery(type="exact-reclaim"),
        lease_seconds=300,
    )
    assert replacement is not None

    with pytest.raises(TaskClaimLost, match="expected worker lease"):
        await store.request_claimed_task_cancellation(
            stale.id,
            "worker-a",
            stale.lease_expires_at,
            {"reason": "stale-owner"},
        )
    replacement_after = await store.load_task(stale.id)
    assert replacement_after is not None
    assert replacement_after.worker_id == "worker-b"
    assert replacement_after.lease_expires_at == replacement.lease_expires_at
    assert replacement_after.status_reason is None

    await store.create_task(
        TaskCreate(task_id="exact_started_reclaim", type="exact-started-reclaim")
    )
    started_claim = await store.claim_task(
        "worker-a",
        TaskQuery(type="exact-started-reclaim"),
        lease_seconds=1,
    )
    assert started_claim is not None
    assert started_claim.lease_expires_at is not None
    with pytest.raises(TaskClaimLost, match="expected worker lease"):
        await store.mark_claimed_task_execution_started(
            started_claim.id,
            "worker-a",
            started_claim.lease_expires_at + timedelta(seconds=1),
        )
    started = await store.mark_claimed_task_execution_started(
        started_claim.id,
        "worker-a",
        started_claim.lease_expires_at,
    )
    assert started.started_at is not None
    assert (
        await store.mark_claimed_task_execution_started(
            started_claim.id,
            "worker-a",
            started_claim.lease_expires_at,
        )
        == started
    )
    await asyncio.sleep(1.05)
    assert await store.reclaim_expired(query=TaskQuery(type="exact-started-reclaim")) == []
    draining = await store.load_task(started_claim.id)
    assert draining is not None
    assert draining.worker_id == "worker-a"
    assert draining.lease_expires_at == started_claim.lease_expires_at
    expected_reason = (
        "retry_cancellation_requested"
        if draining.retry_series is not None
        else "cancellation_requested"
    )
    assert draining.status_reason == expected_reason
    assert (
        await store.claim_task(
            "worker-b",
            TaskQuery(type="exact-started-reclaim"),
            lease_seconds=300,
        )
        is None
    )
    replayed_drain = await store.request_claimed_task_cancellation(
        started_claim.id,
        "worker-a",
        started_claim.lease_expires_at,
        {"reason": "late-owner-fence"},
    )
    assert replayed_drain == draining


async def assert_hold_reason_cancellation_fence_conformance(store: TaskStore) -> None:
    """The fence texts fence a claimed or running task, never a held one.

    A hold stores its caller's reason in ``status_reason``, the field that also
    carries a requested cancellation. A held task has no worker, so the same
    text there must leave it cancellable and holdable. A claimed task whose
    cancellation was requested must still refuse both.
    """

    holds = ("pause_task", "block_task", "mark_task_needs_attention")
    fence_reasons = ("cancellation_requested", "retry_cancellation_requested")
    for hold in holds:
        for reason in fence_reasons:
            task = await store.create_task(TaskCreate(type="hold-reason-cancel"))
            held = await getattr(store, hold)(task.id, reason=reason)
            assert held.status_reason == reason
            assert held.worker_id is None
            assert held.lease_expires_at is None
            cancelled = await store.cancel_task(task.id)
            assert cancelled.status is TaskStatus.CANCELLED

    for reason in fence_reasons:
        task = await store.create_task(TaskCreate(type="hold-reason-hold"))
        await store.pause_task(task.id, reason=reason)
        blocked = await store.block_task(task.id, reason="waiting for a human")
        assert blocked.status is TaskStatus.BLOCKED
        assert blocked.status_reason == "waiting for a human"
        attention = await store.mark_task_needs_attention(task.id, reason=reason)
        assert attention.status is TaskStatus.NEEDS_ATTENTION
        assert attention.status_reason == reason
        resumed = await store.resume_task(task.id)
        assert resumed.status is TaskStatus.PENDING
        assert resumed.status_reason is None

    # A real fence: the worker still holds the task, so it refuses what it did before.
    await store.create_task(TaskCreate(task_id="hold_reason_fenced", type="hold-reason-fenced"))
    claimed = await store.claim_task(
        "worker-a", TaskQuery(type="hold-reason-fenced"), lease_seconds=300
    )
    assert claimed is not None
    requested = await store.cancel_task(claimed.id)
    assert requested.status is TaskStatus.CLAIMED
    assert requested.status_reason == "cancellation_requested"
    assert requested.worker_id == "worker-a"
    for hold in holds:
        with pytest.raises(TaskTerminalizationConflict, match="still draining"):
            await getattr(store, hold)(claimed.id, reason="operator hold")
    with pytest.raises(TaskTerminalizationConflict, match="still draining"):
        await store.cancel_task(claimed.id)
    assert await store.load_task(claimed.id) == requested

    await store.create_task(
        TaskCreate(
            task_id="hold_reason_retry_fenced",
            type="hold-reason-retry-fenced",
            retry_policy=TaskRetryPolicy(max_attempts=2),
        )
    )
    retry_claimed = await store.claim_task(
        "worker-a", TaskQuery(type="hold-reason-retry-fenced"), lease_seconds=300
    )
    assert retry_claimed is not None
    retry_requested = await store.cancel_task(retry_claimed.id)
    assert retry_requested.status is TaskStatus.CLAIMED
    assert retry_requested.status_reason == "retry_cancellation_requested"
    for hold in holds:
        with pytest.raises(TaskTerminalizationConflict, match="still draining"):
            await getattr(store, hold)(retry_claimed.id, reason="operator hold")
    assert await store.load_task(retry_claimed.id) == retry_requested


async def assert_task_store_time_conformance(
    store: TaskStore,
    *,
    initial_time: datetime,
    set_evidence_time: Callable[[datetime], None],
    set_ownership_time: Callable[[datetime], None],
    contender_store: TaskStore | None = None,
) -> None:
    """Exercise the shared authoritative worker-lease clock contract."""

    await store.create_task(TaskCreate(task_id="owned", type="scheduled"))
    claimed = await store.claim_task("worker-a", lease_seconds=60)
    assert claimed is not None
    assert claimed.lease_expires_at == initial_time + timedelta(seconds=60)

    set_evidence_time(initial_time + timedelta(days=3650))
    assert await store.reclaim_expired() == []
    renewed = await store.heartbeat(
        "owned",
        "worker-a",
        lease_expires_at=claimed.lease_expires_at,
        extend_seconds=60,
    )
    assert renewed.lease_expires_at == initial_time + timedelta(seconds=60)

    set_ownership_time(initial_time + timedelta(seconds=60))
    with pytest.raises(TaskClaimLost, match="expired"):
        await store.heartbeat(
            "owned",
            "worker-a",
            lease_expires_at=renewed.lease_expires_at,
        )
    with pytest.raises(TaskClaimLost, match="expired"):
        await store.release_task(
            "owned",
            "worker-a",
            lease_expires_at=renewed.lease_expires_at,
        )

    reclaimed = await store.reclaim_expired()
    assert [task.id for task in reclaimed] == ["owned"]
    replacement = await store.claim_task("worker-b", lease_seconds=60)
    assert replacement is not None
    assert replacement.worker_id == "worker-b"

    renewed, released = await asyncio.gather(
        store.heartbeat(
            "owned",
            "worker-b",
            lease_expires_at=replacement.lease_expires_at,
            extend_seconds=60,
        ),
        (contender_store or store).release_task(
            "owned",
            "worker-b",
            lease_expires_at=replacement.lease_expires_at,
        ),
        return_exceptions=True,
    )
    if isinstance(renewed, BaseException):
        assert isinstance(renewed, TaskClaimLost)
        assert not isinstance(released, BaseException)
        assert released.status is TaskStatus.PENDING
    else:
        assert renewed.worker_id == "worker-b"
        if isinstance(released, TaskClaimLost):
            released = await (contender_store or store).release_task(
                "owned",
                "worker-b",
                lease_expires_at=renewed.lease_expires_at,
            )
        else:
            assert renewed.lease_expires_at == replacement.lease_expires_at
        assert released.status is TaskStatus.PENDING
    pending = await store.load_task("owned")
    assert pending is not None
    assert pending.status is TaskStatus.PENDING
    assert pending.worker_id is None

    with pytest.raises(TaskClaimLost):
        await store.release_task(
            "owned",
            "worker-a",
            lease_expires_at=replacement.lease_expires_at,
        )

    set_evidence_time(initial_time)
    set_ownership_time(initial_time)
    await store.create_task(TaskCreate(task_id="same-worker-reclaim", type="same-worker"))
    stale = await store.claim_task(
        "shared-worker",
        TaskQuery(type="same-worker"),
        lease_seconds=10,
    )
    assert stale is not None
    assert stale.lease_expires_at is not None
    set_ownership_time(initial_time + timedelta(seconds=10))
    assert [task.id for task in await store.reclaim_expired()] == [stale.id]
    successor = await store.claim_task(
        "shared-worker",
        TaskQuery(type="same-worker"),
        lease_seconds=60,
    )
    assert successor is not None
    assert successor.lease_expires_at is not None
    assert successor.lease_expires_at != stale.lease_expires_at
    with pytest.raises(TaskClaimLost, match="lease generation"):
        await store.attach_task(
            successor.id,
            session_id="same-worker-successor-session",
            session_invocation=await task_backed_session_invocation(
                store,
                successor.id,
                "same-worker-successor-session",
            ),
            worker_id="shared-worker",
            lease_expires_at=stale.lease_expires_at,
        )
    with pytest.raises(TaskClaimLost, match="lease generation"):
        await store.terminalize_task(
            TaskTerminalizationRequest(
                task_id=successor.id,
                worker_id="shared-worker",
                lease_expires_at=stale.lease_expires_at,
                kind=TaskTerminalKind.FAILED,
                error={"code": "stale-generation"},
                idempotency_key="same-worker-stale-terminalization",
            )
        )
    assert await store.load_task(successor.id) == successor
    successor = await store.mark_claimed_task_execution_started(
        successor.id,
        "shared-worker",
        successor.lease_expires_at,
    )

    with pytest.raises(TaskClaimLost, match="expected worker lease"):
        await store.heartbeat(
            successor.id,
            "shared-worker",
            lease_expires_at=stale.lease_expires_at,
        )
    with pytest.raises(TaskClaimLost, match="expected worker lease"):
        await (contender_store or store).release_task(
            successor.id,
            "shared-worker",
            lease_expires_at=stale.lease_expires_at,
        )
    unchanged_successor = await store.load_task(successor.id)
    assert unchanged_successor == successor

    set_evidence_time(initial_time + timedelta(days=3650))
    set_ownership_time(initial_time)
    await store.create_task(TaskCreate(task_id="interrupted-owned", type="interrupted-scheduled"))
    source_claim = await store.claim_task(
        "source-worker",
        TaskQuery(type="interrupted-scheduled"),
        lease_seconds=60,
    )
    assert source_claim is not None
    attached = await store.attach_task(
        source_claim.id,
        session_id="interrupted-owned-session",
        session_invocation=await task_backed_session_invocation(
            store,
            source_claim.id,
            "interrupted-owned-session",
        ),
        worker_id="source-worker",
        lease_expires_at=source_claim.lease_expires_at,
    )
    await store.release_interrupted_task_worker(
        interrupted_task_handoff_request(attached, session_run_epoch=1)
    )

    handoff_id = "store-time-interrupted-continuation"
    continuation = await store.claim_interrupted_task_continuation(
        "continuation-worker",
        TaskQuery(type="interrupted-scheduled"),
        handoff_id=handoff_id,
        lease_seconds=30,
    )
    assert continuation.task is not None
    assert continuation.task.id == source_claim.id
    assert continuation.task.session_instance_id is not None
    assert continuation.task.lease_expires_at == initial_time + timedelta(seconds=30)
    active = await store.load_active_attached_task_worker(
        source_claim.id,
        "continuation-worker",
        session_id="interrupted-owned-session",
        session_instance_id=continuation.task.session_instance_id,
    )
    assert active == continuation.task
    replayed = await store.claim_interrupted_task_continuation(
        "continuation-worker",
        TaskQuery(type="interrupted-scheduled"),
        handoff_id=handoff_id,
        lease_seconds=30,
    )
    assert replayed.replayed
    assert replayed.task == continuation.task

    set_ownership_time(initial_time + timedelta(seconds=30))
    with pytest.raises(TaskClaimLost, match="expired"):
        await store.load_active_attached_task_worker(
            source_claim.id,
            "continuation-worker",
            session_id="interrupted-owned-session",
            session_instance_id=continuation.task.session_instance_id,
        )
    with pytest.raises(TaskClaimLost, match="no longer live"):
        await store.claim_interrupted_task_continuation(
            "continuation-worker",
            TaskQuery(type="interrupted-scheduled"),
            handoff_id=handoff_id,
            lease_seconds=30,
        )


async def assert_claim_generation_conformance(store: TaskStore) -> None:
    """A claim is identified by its generation, which renewals keep and holders change."""

    await store.create_task(TaskCreate(task_id="generation_job", type="generation"))
    first = await store.claim_task("stable-worker", TaskQuery(type="generation"), lease_seconds=2)
    assert first is not None and first.claim_generation == 1
    renewed = await store.heartbeat(
        first.id, "stable-worker", claim_generation=first.claim_generation, extend_seconds=300
    )
    assert renewed.claim_generation == first.claim_generation
    assert renewed.lease_expires_at != first.lease_expires_at
    # A write the holder sends with its generation is accepted after any renewal,
    # including one whose acknowledgement it never saw.
    renewed_again = await store.heartbeat(
        first.id, "stable-worker", claim_generation=first.claim_generation, extend_seconds=300
    )
    assert renewed_again.claim_generation == first.claim_generation
    released = await store.release_task(
        first.id, "stable-worker", claim_generation=first.claim_generation
    )
    assert (released.status, released.claim_generation) == (TaskStatus.PENDING, 2)

    # The same worker name claims again: the old generation is refused everywhere.
    second = await store.claim_task("stable-worker", TaskQuery(type="generation"))
    assert second is not None and second.claim_generation == 3
    stale = first.claim_generation
    for attempt in (
        lambda: store.heartbeat(second.id, "stable-worker", claim_generation=stale),
        lambda: store.release_task(second.id, "stable-worker", claim_generation=stale),
        lambda: store.mark_claimed_task_execution_started(
            second.id, "stable-worker", claim_generation=stale
        ),
        lambda: store.request_claimed_task_cancellation(
            second.id, "stable-worker", error={"reason": "stale"}, claim_generation=stale
        ),
        lambda: store.complete_task(
            second.id, {"winner": "stale"}, worker_id="stable-worker", claim_generation=stale
        ),
        lambda: store.fail_task(
            second.id, {"winner": "stale"}, worker_id="stable-worker", claim_generation=stale
        ),
        lambda: store.terminalize_task(
            TaskTerminalizationRequest(
                task_id=second.id,
                worker_id="stable-worker",
                claim_generation=stale,
                kind=TaskTerminalKind.FAILED,
                error={"winner": "stale"},
                idempotency_key="stale-generation",
            )
        ),
    ):
        with pytest.raises(TaskClaimLost, match="superseded"):
            await attempt()
    with pytest.raises(TaskClaimLost, match="superseded"):
        await store.attach_task(
            second.id,
            session_id="generation_session",
            session_invocation=await task_backed_session_invocation(
                store, second.id, "generation_session"
            ),
            worker_id="stable-worker",
            claim_generation=stale,
        )
    with pytest.raises(TaskClaimLost, match="does not own"):
        await store.heartbeat(second.id, "other-worker", claim_generation=second.claim_generation)
    unchanged = await store.load_task(second.id)
    assert unchanged == second

    # A terminalization keeps its identity across a renewal and replays exactly.
    request = TaskTerminalizationRequest(
        task_id=second.id,
        worker_id="stable-worker",
        claim_generation=second.claim_generation,
        kind=TaskTerminalKind.COMPLETED,
        result={"winner": "current"},
        idempotency_key="current-generation",
    )
    await store.heartbeat(second.id, "stable-worker", claim_generation=second.claim_generation)
    completed = await store.terminalize_task(request)
    assert completed.status is TaskStatus.COMPLETED
    assert completed.claim_generation == second.claim_generation
    assert await store.terminalize_task(request) == completed

    # Attaching keeps the generation; releasing the attached worker advances it.
    await store.create_task(TaskCreate(task_id="generation_attached", type="generation-attached"))
    attached_claim = await store.claim_task(
        "attached-worker", TaskQuery(type="generation-attached")
    )
    assert attached_claim is not None
    attached = await store.attach_task(
        attached_claim.id,
        session_id="generation_attached_session",
        session_invocation=await task_backed_session_invocation(
            store, attached_claim.id, "generation_attached_session"
        ),
        worker_id="attached-worker",
        claim_generation=attached_claim.claim_generation,
    )
    assert attached.claim_generation == attached_claim.claim_generation
    detached = await store.release_attached_task_worker(
        attached.id, "attached-worker", claim_generation=attached.claim_generation
    )
    assert detached.worker_id is None
    assert detached.claim_generation == attached.claim_generation + 1

    # A restart requeues the task under a new generation.
    await store.create_task(
        TaskCreate(task_id="generation_restart", type="generation-restart", restartable=True)
    )
    restart_claim = await store.claim_task("restart-worker", TaskQuery(type="generation-restart"))
    assert restart_claim is not None
    restarted = await store.restart_task(
        restart_claim.id, "restart-worker", claim_generation=restart_claim.claim_generation
    )
    assert restarted.status is TaskStatus.PENDING
    assert restarted.claim_generation == restart_claim.claim_generation + 1


async def assert_claim_generation_increments_conformance(
    store: TaskStore,
    *,
    outlive_lease: Callable[[timedelta], Awaitable[None]],
) -> None:
    """Reclaim, handoff, continuation and contract parking each start a new generation.

    ``outlive_lease`` moves the store's ownership time past a lease of that length.
    """

    # Reclaim fences the lost holder whether or not it started.
    await store.create_task(TaskCreate(task_id="reclaim_unstarted", type="reclaim"))
    await store.create_task(TaskCreate(task_id="reclaim_started", type="reclaim"))
    unstarted = await store.claim_task("lost", TaskQuery(type="reclaim"), lease_seconds=1)
    started = await store.claim_task("lost", TaskQuery(type="reclaim"), lease_seconds=1)
    assert unstarted is not None and started is not None
    await store.mark_claimed_task_execution_started(
        started.id, "lost", claim_generation=started.claim_generation
    )
    await outlive_lease(timedelta(seconds=1))
    await store.reclaim_expired()
    requeued = await store.load_task(unstarted.id)
    fenced = await store.load_task(started.id)
    assert requeued is not None and fenced is not None
    assert requeued.status is TaskStatus.PENDING
    assert requeued.claim_generation == unstarted.claim_generation + 1
    assert fenced.status is TaskStatus.CLAIMED
    assert fenced.status_reason == "cancellation_requested"
    assert fenced.claim_generation == started.claim_generation + 1
    with pytest.raises(TaskClaimLost, match="superseded"):
        await store.request_claimed_task_cancellation(
            started.id, "lost", claim_generation=started.claim_generation
        )

    # An interrupted-task handoff and its continuation claim each advance it.
    await store.create_task(TaskCreate(task_id="handed_off", type="handoff"))
    source = await store.claim_task("source", TaskQuery(type="handoff"), lease_seconds=60)
    assert source is not None
    attached = await store.attach_task(
        source.id,
        session_id="handed_off_session",
        session_invocation=await task_backed_session_invocation(
            store, source.id, "handed_off_session"
        ),
        worker_id="source",
        claim_generation=source.claim_generation,
    )
    receipt = await store.release_interrupted_task_worker(
        interrupted_task_handoff_request(attached, session_run_epoch=1)
    )
    assert receipt.task.worker_id is None
    assert receipt.task.claim_generation == source.claim_generation + 1
    continuation = await store.claim_interrupted_task_continuation(
        "source",
        TaskQuery(type="handoff"),
        handoff_id="generation-increments-continuation",
        lease_seconds=60,
    )
    assert continuation.task is not None and continuation.task.id == source.id
    assert continuation.task.claim_generation == source.claim_generation + 2
    with pytest.raises(TaskClaimLost, match="superseded"):
        await store.heartbeat(source.id, "source", claim_generation=source.claim_generation)

    # Parking a contracted task releases its worker under a new generation.
    contract = work_contract_from_draft(
        WorkContractDraft(
            contract_id="generation-increments",
            version=1,
            objective="Park the contracted task.",
            criteria=(
                WorkCriterion(criterion_id="parked", ordinal=1, description="It is parked."),
            ),
            verifier=CompletionVerifierRef(
                verifier_id="generation-verifier",
                version="v1",
                configuration_fingerprint="0" * 64,
            ),
            result_resolver=CompletionResultResolverRef(
                resolver_id="generation-result",
                version="v1",
                configuration_fingerprint="1" * 64,
            ),
        )
    )
    await store.publish_work_contract(contract)
    await store.create_task(
        TaskCreate(task_id="parked", type="contracted", work_contract=contract.reference())
    )
    contracted = await store.claim_task("ordinary", TaskQuery(type="contracted"))
    assert contracted is not None
    parked = await store.hold_claimed_work_contract_task(
        contracted.id,
        worker_id="ordinary",
        claim_generation=contracted.claim_generation,
        contract=contract.reference(),
    )
    assert parked.worker_id is None
    assert parked.claim_generation == contracted.claim_generation + 1


async def assert_task_types_latest_and_once_conformance(store: TaskStore) -> None:
    """Several task types in one query, the latest finished tasks, and queue-once keys."""

    from cayu.tasks.once import create_task_once

    async def finish(task_type: str, *, failed: bool = False, result: dict | None = None):
        claimed = await store.claim_task("conformance", TaskQuery(type=task_type))
        assert claimed is not None
        if failed:
            return await store.fail_task(
                claimed.id,
                {"message": "failed"},
                worker_id="conformance",
                claim_generation=claimed.claim_generation,
            )
        return await store.complete_task(
            claimed.id,
            result or {},
            worker_id="conformance",
            claim_generation=claimed.claim_generation,
        )

    # Several types: list and claim match any of them, in the usual claim order.
    for task_id, task_type in (("t-a", "alpha"), ("t-b", "beta"), ("t-g", "gamma")):
        await store.create_task(TaskCreate(task_id=task_id, type=task_type))
    listed = await store.list_tasks(TaskQuery(types=("alpha", "beta")))
    assert {task.id for task in listed} == {"t-a", "t-b"}
    first = await store.claim_task("several", TaskQuery(types=("beta", "alpha")))
    second = await store.claim_task("several", TaskQuery(types=("beta", "alpha")))
    assert [first.id if first else None, second.id if second else None] == ["t-a", "t-b"]
    assert await store.claim_task("several", TaskQuery(types=("beta", "alpha"))) is None

    # The latest finished task, and the latest per input key.
    for index, (repository, failed) in enumerate(
        (("o/a", False), ("o/b", False), ("o/a", False), ("o/c", True), (7, False))
    ):
        task_input = {"repository": repository}
        await store.create_task(TaskCreate(task_id=f"pass-{index}", type="pass", input=task_input))
        await finish("pass", failed=failed, result={"n": index})
    await store.create_task(
        TaskCreate(task_id="pass-open", type="pass", input={"repository": "o/d"})
    )

    completed = TaskQuery(type="pass", status=TaskStatus.COMPLETED)
    latest = await store.latest_task(completed)
    assert latest is not None and latest.id == "pass-4" and latest.result == {"n": 4}
    assert [task.id for task in await store.latest_tasks(TaskQuery(type="pass", limit=2))] == [
        "pass-4",
        "pass-3",
    ]
    by_repository = await store.latest_tasks(completed, by="input.repository")
    assert [(task.id, task.input["repository"]) for task in by_repository] == [
        ("pass-2", "o/a"),
        ("pass-1", "o/b"),
    ]
    any_status = await store.latest_tasks(TaskQuery(type="pass"), by="input.repository")
    assert [task.id for task in any_status] == ["pass-3", "pass-2", "pass-1"]
    paged = await store.latest_tasks(
        TaskQuery(type="pass", limit=1, offset=1), by="input.repository"
    )
    assert [task.id for task in paged] == ["pass-2"]
    assert await store.latest_tasks(TaskQuery(type="pass"), by="input.missing") == []
    assert await store.latest_task(TaskQuery(type="none")) is None
    for wrong in ("result.repository", "input.items.0", "input.items.-1", "input.-"):
        with pytest.raises(ValueError, match="field of the task's input"):
            await store.latest_tasks(TaskQuery(type="pass"), by=wrong)

    # Queue once per key: one task however often it is asked for, also when callers race.
    racing = await asyncio.gather(
        *(create_task_once(store, TaskCreate(type="digest", retry_key="slot-am")) for _ in range(3))
    )
    assert sum(item.created for item in racing) == 1
    assert len({item.task.id for item in racing if item.task is not None}) == 1
    done = await finish("digest")
    again = await create_task_once(store, TaskCreate(type="digest", retry_key="slot-am"))
    assert (again.created, again.task.id if again.task else None) == (False, done.id)
    assert again.task is not None and again.task.status is TaskStatus.COMPLETED

    # A failed key stays failed with one attempt; with more, the next call queues again.
    await create_task_once(store, TaskCreate(type="digest", retry_key="slot-pm"))
    failed = await finish("digest", failed=True)
    stays = await create_task_once(store, TaskCreate(type="digest", retry_key="slot-pm"))
    assert (stays.created, stays.task.id if stays.task else None) == (False, failed.id)

    await create_task_once(store, TaskCreate(type="digest", retry_key="slot-x"), attempts=2)
    await finish("digest", failed=True)
    requeued = await asyncio.gather(
        *(
            create_task_once(store, TaskCreate(type="digest", retry_key="slot-x"), attempts=2)
            for _ in range(3)
        )
    )
    assert sum(item.created for item in requeued) == 1
    [new_id] = {item.task.id for item in requeued if item.task is not None}
    await finish("digest", failed=True)
    # A later call keeps the key's budget whatever attempts it passes.
    last = await create_task_once(store, TaskCreate(type="digest", retry_key="slot-x"))
    assert (last.created, last.task.id if last.task else None) == (False, new_id)
    assert last.key is not None and last.key.attempts == 2

    await create_task_once(store, TaskCreate(type="digest", retry_key="slot-y"), attempts=3)
    await finish("digest", failed=True)
    retried = await create_task_once(store, TaskCreate(type="digest", retry_key="slot-y"))
    assert retried.created and retried.key is not None and retried.key.attempts == 2
    with pytest.raises(ValueError, match="leave task_id unset"):
        await create_task_once(store, TaskCreate(task_id="mine", type="digest", retry_key="k"))

    # A caller that read the key before others queued attempts 1 and 2 collides on
    # attempt 1's ID; it gets the key's current task back, not an error.
    late = TaskCreate(type="digest-late", retry_key="slot-z")
    await create_task_once(store, late, attempts=2)
    await finish("digest-late", failed=True)
    current = await create_task_once(store, late)
    assert current.created and current.task is not None
    delayed = await create_task_once(cast("TaskStore", _StaleKeyRead(store)), late)
    assert (delayed.created, delayed.task.id if delayed.task else None) == (False, current.task.id)
    assert delayed.key is not None and delayed.key.attempts == 2


class _StaleKeyRead:
    """A store whose first retry-key read is from before any task was queued for it."""

    def __init__(self, store: TaskStore) -> None:
        self._store = store
        self._stale = True

    def __getattr__(self, name: str):
        return getattr(self._store, name)

    async def load_task_retry_key(self, retry_key: str):
        if self._stale:
            self._stale = False
            return None
        return await self._store.load_task_retry_key(retry_key)
