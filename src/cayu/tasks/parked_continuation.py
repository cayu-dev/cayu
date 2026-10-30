"""Continue a task parked on a wait of its session, under a claim bound to that wait.

A worker hands a task off *parked* when its session waits on a trigger of its own,
such as a ``sleep_until`` wake. No generic continuation claims a parked task: only
the trigger it names does, through a bound claim that presents the same
:class:`~cayu.tasks.handoff.TaskParkedWait`, and continues the session under the
task. These helpers are shared by every kind of parked wait.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from cayu.tasks.continuation_claims import TaskContinuationExpectation
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import TaskStatus

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.tasks.handoff import TaskInterruptedHandoffReceipt, TaskParkedWait
    from cayu.tasks.records import Task
    from cayu.tasks.worker import RecoveredInterruptedTaskHandler

TERMINAL_TASK_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED})
# A trigger can come due before the worker that ran its session has handed the task
# off. Its attempt waits this long for the handoff before failing into a successor.
OWNER_HANDOFF_WAIT = timedelta(seconds=5)
OWNER_HANDOFF_POLL = timedelta(milliseconds=100)


class ParkedOwnerBusy(RuntimeError):
    """The task that owns the session has not handed it off yet; the trigger is retried."""

    def __init__(self, task_id: str, message: str) -> None:
        self.task_id = task_id
        super().__init__(message)


def declares_parked_handoffs(task_store: object) -> bool:
    """Whether a task store's class declares bound and parked interrupted-task handoffs.

    Only a store that declares both itself can hand a task off parked on its
    session's wait and continue it through a claim bound to that wait.
    """

    declared = type.__getattribute__(type(task_store), "__dict__")
    return all(
        declared.get(capability) is True
        for capability in (
            "supports_bound_interrupted_task_handoffs",
            "supports_parked_interrupted_task_handoffs",
        )
    )


async def owning_task(app: CayuApp, session_id: str) -> Task | None:
    """Return the running task that owns a session, or ``None`` for an unowned one."""

    task_store = app.task_store
    if task_store is None:
        return None
    owners = await task_store.list_tasks(
        TaskQuery(session_id=session_id, status=TaskStatus.RUNNING, limit=2)
    )
    if len(owners) > 1:
        raise RuntimeError("The waiting session has ambiguous task ownership.")
    return owners[0] if owners else None


def continuation_suffix(wait_id: str) -> str:
    """How the worker id of a trigger's claim on its owning task ends."""

    return f":continue:{wait_id}"


async def handed_off_owner(
    app: CayuApp,
    owner: Task,
    *,
    parked: TaskParkedWait,
    still_waiting: bool,
    busy: Callable[[str], ParkedOwnerBusy],
) -> tuple[Task, TaskParkedWait | None] | None:
    """Wait briefly for the owner's handoff and return it with the wait to present.

    A trigger continues a handoff parked on it, and repairs an unparked one while the
    session still waits on it. A handoff parked on another wait, or an unparked one
    after the wait closed, belongs to someone else: ``None``. While the owner still
    holds the task, this waits up to ``OWNER_HANDOFF_WAIT`` and then raises ``busy``.
    """

    task_store = app.task_store
    assert task_store is not None
    waited = timedelta(0)
    current: Task | None = owner
    while current is not None and current.status is TaskStatus.RUNNING:
        if current.worker_id is None and current.interrupted_handoff_id is not None:
            receipt = await task_store.load_interrupted_task_handoff_receipt(
                current.id, current.interrupted_handoff_id
            )
            if receipt is None:
                return None
            presented = receipt.request.parked
            if presented == parked or (presented is None and still_waiting):
                return current, presented
            return None
        if not still_waiting:
            return None
        if waited >= OWNER_HANDOFF_WAIT:
            raise busy(current.id)
        await asyncio.sleep(OWNER_HANDOFF_POLL.total_seconds())
        waited += OWNER_HANDOFF_POLL
        current = await task_store.load_task(owner.id)
    return None


async def run_under_parked_claim(
    app: CayuApp,
    *,
    task_id: str,
    session_id: str,
    session_instance_id: str,
    handoff_id: str,
    parked: TaskParkedWait | None,
    operation: str,
    worker_id: str,
    lease_seconds: int,
    step: RecoveredInterruptedTaskHandler,
) -> bool:
    """Claim a task's handoff with the wait it presents and run ``step`` under it.

    ``handoff_id`` names the handoff to claim. ``operation`` names the continuation for the claim's
    idempotency. The worker hands the task off again, parked when its session waits
    on a trigger again, once ``step`` returns ``SESSION_INTERRUPTED``. Returns whether
    this call held the claim; ``False`` when another one holds or finished it.
    """

    import hashlib

    from cayu.tasks.worker import _run_recovered_interrupted_task_handler

    task_store = app.task_store
    assert task_store is not None
    expected = TaskContinuationExpectation(
        task_id=task_id,
        session_id=session_id,
        session_instance_id=session_instance_id,
        prior_handoff_id=handoff_id,
        operation_sha256=hashlib.sha256(operation.encode()).hexdigest(),
        parked_wait=parked,
    )
    page = await _run_recovered_interrupted_task_handler(
        app,
        task_store,
        step,
        worker_id=worker_id,
        expected=expected,
        apply_unproductive_backoff=False,
        lease_seconds=lease_seconds,
        query=None,
        after=None,
    )
    return page.task is not None


async def release_parked_handoff(
    app: CayuApp, receipt: TaskInterruptedHandoffReceipt, *, lease_seconds: int
) -> bool:
    """Claim one stale parked handoff with its own wait and hand it off again."""

    from cayu.tasks.worker import TaskHandlerOutcome

    request = receipt.request

    async def hand_off_again(owned_app: CayuApp, claimed: Task, claimed_by: str):
        # The claim already proved the session interrupted; the worker hands the
        # task off for whatever the session waits on now.
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    return await run_under_parked_claim(
        app,
        task_id=request.task_id,
        session_id=request.session_id,
        session_instance_id=request.session_instance_id,
        handoff_id=request.handoff_id,
        parked=request.parked,
        operation=f"cayu.parked-handoff-release.v1:{request.handoff_id}",
        worker_id=f"parked-handoff-release:{request.handoff_id[:16]}",
        lease_seconds=lease_seconds,
        step=hand_off_again,
    )


def attempt_is_live(task: Task, *, now: datetime) -> bool:
    """An attempt that ended, lost its worker's lease or was fenced is not live."""

    from cayu.tasks import is_fenced_after_worker_loss

    if task.status in TERMINAL_TASK_STATUSES or is_fenced_after_worker_loss(task):
        return False
    return not (
        task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        and task.lease_expires_at is not None
        and task.lease_expires_at <= now
    )
