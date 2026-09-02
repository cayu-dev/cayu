from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from tests.core.task_invocation_fixtures import (
    task_backed_session_invocation,
    unattributed_session_invocation_binding,
)

from cayu import (
    Task,
    TaskClaimLost,
    TaskCreate,
    TaskExecutionSource,
    TaskQuery,
    TaskStatus,
    TaskStore,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    interrupted_task_handoff_request,
)
from cayu.runtime.tasks import task_create_with_runtime_invocation


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

    with pytest.raises(TaskClaimLost, match="does not own"):
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
    with pytest.raises(TaskClaimLost, match="does not own"):
        await store.heartbeat(
            claimed.id,
            "worker_b",
            lease_expires_at=claimed.lease_expires_at,
        )
    with pytest.raises(TaskClaimLost, match="does not own"):
        await store.release_task(
            claimed.id,
            "worker_b",
            lease_expires_at=claimed.lease_expires_at,
        )
    with pytest.raises(TaskClaimLost, match="does not own"):
        await store.complete_task(
            claimed.id,
            {"ok": True},
            worker_id="worker_b",
            lease_expires_at=claimed.lease_expires_at,
        )
    with pytest.raises(TaskClaimLost, match="does not own"):
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
    with pytest.raises(TaskClaimLost, match="does not own"):
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
    with pytest.raises(TaskClaimLost, match="does not own"):
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
