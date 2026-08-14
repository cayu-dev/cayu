from __future__ import annotations

import asyncio

import pytest
from tests.core.task_invocation_fixtures import (
    task_backed_session_invocation,
    unattributed_session_invocation_binding,
)

from cayu import (
    TaskClaimLost,
    TaskCreate,
    TaskExecutionSource,
    TaskQuery,
    TaskStatus,
    TaskStore,
)
from cayu.runtime.tasks import task_create_with_runtime_invocation


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
        )
    with pytest.raises(TaskClaimLost, match="does not own"):
        await store.heartbeat(claimed.id, "worker_b")
    with pytest.raises(TaskClaimLost, match="does not own"):
        await store.release_task(claimed.id, "worker_b")
    with pytest.raises(TaskClaimLost, match="does not own"):
        await store.complete_task(claimed.id, {"ok": True}, worker_id="worker_b")
    with pytest.raises(TaskClaimLost, match="does not own"):
        await store.fail_task(claimed.id, {"message": "failed"}, worker_id="worker_b")

    attached = await store.attach_task(
        claimed.id,
        session_id="claim_lost_session",
        session_invocation=await task_backed_session_invocation(
            store,
            claimed.id,
            "claim_lost_session",
        ),
        worker_id="worker_a",
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
        )
    assert type(duplicate_attach.value) is ValueError
    with pytest.raises(TaskClaimLost, match="does not own"):
        await store.release_attached_task_worker(attached.id, "worker_b")

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
        )
    reclaimed = await store.reclaim_expired(
        query=TaskQuery(type="claim_lost_reclaimed"),
    )
    assert [task.id for task in reclaimed] == [stale_claim.id]
    with pytest.raises(TaskClaimLost, match="not claimed or running"):
        await store.release_task(stale_claim.id, "worker_a")
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
    )
    terminal = await store.complete_task(
        attached_terminal.id,
        {"winner": "control-plane"},
    )
    assert terminal.status is TaskStatus.COMPLETED
    with pytest.raises(TaskClaimLost, match="not claimed or running"):
        await store.release_attached_task_worker(attached_terminal.id, "worker_a")

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
        await store.release_attached_task_worker(structural_claim.id, "worker_a")
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
    )
    with pytest.raises(ValueError, match="already attached") as attached_release:
        await store.release_task(structural_attached.id, "worker_a")
    assert type(attached_release.value) is ValueError
