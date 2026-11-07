"""Task claims fenced by their durable claim generation.

A claim is identified by ``Task.claim_generation``, not by the worker name or by
the exact lease expiry. Claim, continuation claim, release, reclaim, handoff and
restart advance the generation; a heartbeat keeps it and only moves the expiry.
A claimed mutation that presents its generation is accepted only while the row
still carries that generation, is claimed or running under the same worker, and
its lease is live. A renewal whose acknowledgement was lost therefore never
fences its own holder, and a same-name replacement never accepts a write meant
for an earlier claim.

Stores check the fence on the row they read under their own transaction lock,
then write against that same row's lease and handoff values. Callers that
present only an exact lease expiry use the deprecated exact-token path.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime

from cayu._clock import normalize_utc_datetime
from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.tasks._lifecycle import (
    _ensure_active_task_lease,
    _ensure_exact_owned_active_task_lease,
)
from cayu.tasks.records import Task, TaskClaimLost, TaskStatus

_HELD_STATUSES = frozenset({TaskStatus.CLAIMED, TaskStatus.RUNNING})
_legacy_lease_fence_warned = False


@dataclass(frozen=True, slots=True)
class TaskClaimFence:
    """The claim a mutation belongs to: its generation, plus its worker as a sanity check."""

    worker_id: str
    claim_generation: int


def validate_claim_generation(value: int | None) -> int | None:
    """Validate one presented claim generation, or pass ``None`` through."""

    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError("claim_generation must be a non-negative durable integer.")
    return value


def ensure_claim_fence(
    task: Task,
    fence: TaskClaimFence,
    *,
    now: datetime,
    require_live: bool = True,
) -> datetime:
    """Require the row to still belong to the fenced claim and return its lease.

    The lease must be live unless ``require_live`` is false.
    """

    if task.status not in _HELD_STATUSES:
        raise TaskClaimLost(f"Task {task.id} is not claimed or running.")
    if task.claim_generation != fence.claim_generation:
        raise TaskClaimLost(
            f"Task {task.id} claim generation {fence.claim_generation} was superseded "
            f"by {task.claim_generation}."
        )
    if task.worker_id != fence.worker_id:
        raise TaskClaimLost(f"Worker {fence.worker_id} does not own task {task.id}.")
    if require_live:
        _ensure_active_task_lease(task, fence.worker_id, now=now)
    if task.lease_expires_at is None:
        raise TaskClaimLost(f"Task {task.id} has no worker lease.")
    return task.lease_expires_at


def warn_legacy_lease_fence() -> None:
    """Warn once per process that exact-lease fencing without a generation is deprecated."""

    global _legacy_lease_fence_warned
    if _legacy_lease_fence_warned:
        return
    _legacy_lease_fence_warned = True
    warnings.warn(
        "Fencing a claimed task mutation by its exact lease_expires_at alone is "
        "deprecated; present the claim's claim_generation instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def fenced_claim_lease(
    task: Task,
    worker_id: str,
    *,
    lease_expires_at: datetime | None,
    claim_generation: int | None,
    now: datetime,
) -> tuple[datetime, str | None]:
    """Authenticate one claimed mutation and return the lease and handoff it writes against.

    With a generation, the generation fence decides, and the write is bound to
    the lease and handoff of the row the caller read under its lock. Without
    one, the exact presented lease must match (the deprecated path), and the
    write carries no handoff.
    """

    if claim_generation is not None:
        lease = ensure_claim_fence(task, TaskClaimFence(worker_id, claim_generation), now=now)
        return lease, task.interrupted_handoff_id
    if lease_expires_at is None:
        raise TaskClaimLost("A claimed task mutation requires its claim generation.")
    warn_legacy_lease_fence()
    _ensure_exact_owned_active_task_lease(task, worker_id, lease_expires_at, now=now)
    return lease_expires_at, None


def released_claim_lease(
    task: Task,
    worker_id: str,
    lease_expires_at: datetime | None,
    claim_generation: int | None,
    *,
    now: datetime,
    message: str,
) -> datetime:
    """Authenticate a release and return the lease the release writes against.

    With a generation, the generation fence decides. Without one, the exact
    presented lease must match; ``message`` is the refusal when it does not.
    """

    if claim_generation is not None:
        return ensure_claim_fence(task, TaskClaimFence(worker_id, claim_generation), now=now)
    if lease_expires_at is None:
        raise TaskClaimLost("Task release requires its claim generation.")
    expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")
    if task.lease_expires_at != expected_lease:
        raise TaskClaimLost(message)
    return expected_lease
