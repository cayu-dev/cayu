"""Settle every exit of a claimed task's handler through the shared exit table.

A worker or dispatcher that runs a claimed task owns two things once the
handler stops: the task's disposition and, for a quiescent task group, the
acknowledgement that this execution stopped. These helpers classify the exit
with :func:`cayu.runtime._run_exit.classify_run_exit` and settle both
obligations on every path.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from cayu._exception_groups import iter_exception_tree
from cayu.runtime._run_exit import RunExitKind, classify_run_exit
from cayu.tasks._execution_settlement import has_task_execution_settlement_pending
from cayu.tasks.records import TaskClaimLost, TaskStatus

if TYPE_CHECKING:
    from cayu.tasks._execution_settlement import TaskExecutionSettlement
    from cayu.tasks.records import Task
    from cayu.tasks.store import TaskStore


@dataclass(slots=True)
class UnstartedClaim:
    """One claim this worker holds, by identity: never the task's content.

    ``leases`` holds every lease expiry this worker was acknowledged for this
    claim, the claim's own and each renewal, so a renewed claim is still this
    worker's while a newer claim under the same worker id is not.
    """

    task_id: str
    leases: set[datetime] = field(default_factory=set)

    @classmethod
    def of(cls, task: Task) -> UnstartedClaim:
        claim = cls(task.id)
        claim.renewed(task)
        return claim

    def renewed(self, task: Task) -> None:
        if task.id == self.task_id and task.lease_expires_at is not None:
            self.leases.add(task.lease_expires_at)


async def acknowledge_stopped_handler(
    settlement: TaskExecutionSettlement,
    authority: Task,
    failure: BaseException,
) -> None:
    """Acknowledge the execution of a handler that stopped by raising.

    A raised exception proves the callback returned, which is the proof the
    group barrier needs. A process-level stop is left to recovery. The
    acknowledgement never replaces the handler's own failure: a non-retained
    acknowledgement failure is attached to it as a note.
    """

    if classify_run_exit(failure).kind is RunExitKind.PROCESS_CONTROL:
        return
    settlement.observe(authority, None)
    try:
        await settlement.attempt()
    except Exception as acknowledgement_failure:
        failure.add_note(
            "Task-group execution acknowledgement also failed: "
            f"{type(acknowledgement_failure).__name__}."
        )


def cancellation_carried_by(
    error: BaseException,
    *,
    cancellation_baseline: int,
) -> asyncio.CancelledError | None:
    """Return the cancellation a handler's grouped exit carries, if it settles as one.

    A group with a process-control leaf is never a cancellation. Otherwise
    a delivered cancellation, or a ``CancelledError`` the handler raised
    beside cleanup failures, settles the claim exactly as a bare cancellation
    would.
    """

    run_exit = classify_run_exit(error, cancellation_baseline=cancellation_baseline)
    if run_exit.kind not in {RunExitKind.CANCELLED, RunExitKind.FAILED}:
        return None
    return next(
        (
            candidate
            for candidate in iter_exception_tree(error)
            if isinstance(candidate, asyncio.CancelledError)
        ),
        None,
    )


async def release_unstarted_claims(
    task_store: TaskStore,
    worker_id: str,
    failure: BaseException,
    *,
    claims: list[UnstartedClaim],
) -> None:
    """Return claims this worker holds but never started to the queue.

    Between ``claim_task`` and the execution-start mark no handler owns a
    claim, so any exit there, including one that arrives with the claim's own
    acknowledgement, would strand it with the exited worker until its lease
    expires. Durable state decides: only a claim this worker still holds, not
    started, is released, shielded from the caller's cancellation. ``claims``
    names the claims this worker's claim calls returned, and only that claim (a
    lease this worker was acknowledged for it) is released: a newer claim under
    the same worker id belongs to its own owner. Unacknowledged claims are left
    to lease expiry because their authority cannot be recovered by worker id. A process-level stop is left
    to lease expiry, and a claim whose execution entry is still pending stays
    with that entry's settlement owner.
    """

    from cayu.runtime._recovery_ownership import _run_recovery_cleanup_steps

    if classify_run_exit(
        failure
    ).kind is RunExitKind.PROCESS_CONTROL or has_task_execution_settlement_pending(failure):
        # A pending execution-entry settlement still owns the claim: its retry
        # proves nondispatch before anyone relinquishes it.
        return

    async def release() -> None:
        for candidate in claims:
            current = await task_store.load_task(candidate.task_id)
            if (
                current is None
                or current.status is not TaskStatus.CLAIMED
                or current.started_at is not None
                or current.worker_id != worker_id
                or current.lease_expires_at not in candidate.leases
            ):
                continue
            try:
                await task_store.release_task(
                    current.id, worker_id, lease_expires_at=current.lease_expires_at
                )
            except TaskClaimLost:
                continue

    await _run_recovery_cleanup_steps(
        authoritative_failure=failure,
        steps=(("unstarted task claim release", release),),
    )
