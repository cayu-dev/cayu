"""Settle every exit of a claimed task's handler through the shared exit table.

A worker or dispatcher that runs a claimed task owns two things once the
handler stops: the task's disposition and, for a quiescent task group, the
acknowledgement that this execution stopped. These helpers classify the exit
with :func:`cayu.runtime._run_exit.classify_run_exit` and settle both
obligations on every path.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cayu._exception_groups import iter_exception_tree
from cayu.runtime._run_exit import RunExitKind, classify_run_exit
from cayu.tasks._execution_settlement import pending_task_execution_settlements
from cayu.tasks.cancellation import (
    _task_cancellation_requested,
    _task_retry_cancellation_requested,
)
from cayu.tasks.records import TaskClaimLost, TaskStatus

if TYPE_CHECKING:
    from cayu.tasks._execution_settlement import TaskExecutionSettlement
    from cayu.tasks.records import Task
    from cayu.tasks.store import TaskStore


@dataclass(frozen=True, slots=True)
class UnstartedClaim:
    """One claim this worker holds, by identity: never the task's content.

    ``claim_generation`` is the generation the claim call acknowledged. Renewals
    keep it, so a renewed claim is still this worker's, while a newer claim
    under the same worker id carries a later one.
    """

    task_id: str
    claim_generation: int

    @classmethod
    def of(cls, task: Task) -> UnstartedClaim:
        return cls(task.id, task.claim_generation)


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
    """Return claims this worker holds but never dispatched to the queue.

    Between ``claim_task`` and the acknowledged execution-start mark no handler
    owns a claim, so any exit there, including one that arrives with the
    claim's own acknowledgement, would strand it with the exited worker until
    its lease expires. Durable state decides: only a claim this worker still
    holds is released, shielded from the caller's cancellation. ``claims``
    names the claims this worker's claim calls returned, each released only
    under the claim generation it was acknowledged for: renewals keep the
    generation, and a newer claim under the same worker id carries its own.
    Unacknowledged claims are left to lease expiry because their authority
    cannot be recovered by worker id. A process-level stop is left to lease
    expiry.

    A cancellation while the execution-start mark is in flight leaves that
    entry's settlement pending. The entry raised, so no handler was dispatched:
    its own settlement joins the mark and acknowledges any group execution it
    opened before the claim, start marker included, is released. An entry that
    failed on its own keeps its claim with the pending settlement, which its
    caller retries, and any other pending settlement belongs to a dispatched
    handler and keeps the claim.
    """

    from cayu.runtime._recovery_ownership import _run_recovery_cleanup_steps

    if classify_run_exit(failure).kind is RunExitKind.PROCESS_CONTROL:
        return
    pending = pending_task_execution_settlements(failure)
    if pending and (
        not all(settlement.undispatched_entry for settlement in pending)
        or not any(
            isinstance(error, asyncio.CancelledError) for error in iter_exception_tree(failure)
        )
    ):
        return

    async def release() -> None:
        entered: dict[str, Task] = {}
        for settlement in pending:
            recorded = await settlement.settle_undispatched_entry()
            entered[recorded.id] = recorded
        for candidate in claims:
            current = await task_store.load_task(candidate.task_id)
            # Only the start marker of an undispatched entry may be released.
            entry = entered.get(candidate.task_id)
            undispatched_start = None if entry is None else entry.started_at
            if (
                current is None
                or current.status is not TaskStatus.CLAIMED
                or current.started_at not in {None, undispatched_start}
                or current.worker_id != worker_id
                or current.claim_generation != candidate.claim_generation
                # A requested cancellation keeps the claim with its drain.
                or _task_cancellation_requested(current)
                or _task_retry_cancellation_requested(current)
            ):
                continue
            try:
                # Fenced by the generation this worker's claim call returned.
                await task_store.release_task(
                    current.id, worker_id, claim_generation=candidate.claim_generation
                )
            except TaskClaimLost:
                continue

    await _run_recovery_cleanup_steps(
        authoritative_failure=failure,
        steps=(("unstarted task claim release", release),),
    )
