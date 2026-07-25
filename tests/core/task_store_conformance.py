from __future__ import annotations

import asyncio

import pytest

from cayu import TaskClaimLost, TaskCreate, TaskQuery, TaskStatus, TaskStore


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
        worker_id="worker_a",
    )
    assert attached.status is TaskStatus.RUNNING
    with pytest.raises(TaskClaimLost, match="does not own"):
        await store.attach_task(
            attached.id,
            session_id="claim_lost_replacement_session",
            worker_id="worker_b",
        )
    with pytest.raises(ValueError, match="already attached") as duplicate_attach:
        await store.attach_task(
            attached.id,
            session_id="claim_lost_duplicate_session",
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
        worker_id="worker_a",
    )
    with pytest.raises(ValueError, match="already attached") as attached_release:
        await store.release_task(structural_attached.id, "worker_a")
    assert type(attached_release.value) is ValueError
