"""External-event waits in sessions a task owns: the task parks until its wait is serviced.

A task handler runs its session to a wait with ``hosting.run_to_wait(...)`` or
``hosting.resume_to_wait(...)`` under its worker's claim and returns
``TaskHandlerOutcome.SESSION_INTERRUPTED``. The worker hands the task off parked on
the wait (``TaskParkedWait(kind="external_wait")``), so no generic continuation
claims it. When the wait's event or timeout is serviced, the servicing claims the
task with that same wait and continues the session under it; the task then
completes, fails or is handed off again with its session. Nothing else continues
the session: deliveries and sweeps only schedule the service task.

A wait whose task ended is retired instead of continued. Interrupting the session
retires its wait, and the worker then hands the task back unparked. Deleting it is
refused while its task runs.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING, NamedTuple
from uuid import uuid4
from weakref import WeakKeyDictionary

from cayu.sessions.external_waits import (
    EXTERNAL_RECOVERY_INACTIVITY_S,
    ExternalWaitRegistration,
    ExternalWaitScope,
    ExternalWaitUnavailable,
    external_wait_digest,
)
from cayu.sessions.records import SessionStatus
from cayu.tasks.handoff import TaskParkedWait
from cayu.tasks.parked_continuation import (
    TERMINAL_TASK_STATUSES,
    ParkedOwnerBusy,
    continuation_suffix,
    handed_off_owner,
    owning_task,
    run_under_parked_claim,
)
from cayu.tasks.queries import TaskQuery

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.external_wait_hosting import ExternalWaitHosting
    from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait
    from cayu.session_external_waits import SessionExternalWaitAdapter
    from cayu.sessions.external_waits import ExternalWaitRecord
    from cayu.sessions.store import SessionStore
    from cayu.tasks.records import Task
    from cayu.tasks.store import TaskStore

_EXTERNAL_PURPOSE = "external-event-v1"
_ARMING = frozenset({"ARMING"})
_HELD = frozenset({"ARMING", "WAITING"})
_REGISTRATION_PREFIX = "external-wait:"
# How long the servicing that continues a task-owned session holds its claim at once.
_CONTINUATION_LEASE_SECONDS = 300
_logger = logging.getLogger(__name__)
# The sessions each application that hosts no waits already reported as left to a
# worker that does, so its expired-lease passes don't repeat the warning.
_UNHOSTED_TURNS_REPORTED: WeakKeyDictionary[CayuApp, set[str]] = WeakKeyDictionary()

CHILD_SESSION_REFUSAL = (
    "External session waits don't support subagent sessions. Only application code "
    "enters a wait, and a subagent's run is started by its parent's call inside the "
    "runtime, so nothing could park it where that call waits. Wait in the session "
    "at the top of the chain (or the task's session) instead."
)


class ExternalWaitOwnerBusy(ParkedOwnerBusy, ExternalWaitUnavailable):
    """The task that owns the waiting session has not handed it off yet; it is retried."""

    def __init__(self, task_id: str) -> None:
        super().__init__(
            task_id,
            f"Task {task_id} still holds the waiting session; its wait's service continues "
            "it once the task is handed off.",
        )


class ParkedExternalWait(NamedTuple):
    """The external wait a session's turn parked on, from its continuation index."""

    wait_id: str
    scope: ExternalWaitScope
    source_key: str
    #: Its event's delivery is already admitting the continuation; only that
    #: service may settle it now.
    delivering: bool


async def parked_external_wait(
    app: CayuApp, session_id: str, *, states: frozenset[str] = frozenset({"WAITING"})
) -> ParkedExternalWait | None:
    """The external wait the session's current turn holds in one of ``states``, if any."""

    return await _held_wait(app.session_store, session_id, states)


async def turn_held_by_task_wait(
    session_store: SessionStore, task_store: TaskStore | None, session_id: str
) -> str | None:
    """The task whose session's turn died before parking on its external wait, if any.

    Only that wait's own recovery may bring the turn to it, or stop the wait; plain
    incomplete-session recovery would leave the turn to run again blind, or leave
    the wait able to continue a session its task's handler now owns.
    """

    held = await _held_wait(session_store, session_id, _ARMING)
    if held is None or await _registration_in(session_store, held) is None:
        # No wait is left to protect: plain recovery is safe.
        return None
    if task_store is None:
        # This process can't see the tasks: the wait itself names the turn's task.
        return await _wait_task_id(session_store, held)
    for task in await task_store.list_tasks(TaskQuery(session_id=session_id, limit=10)):
        if task.status not in TERMINAL_TASK_STATUSES:
            return task.id
    return None


async def _wait_task_id(session_store: SessionStore, held: ParkedExternalWait) -> str | None:
    """The task a held wait's turn runs under, as the wait recorded before the turn began.

    The session store holds it from the moment the wait is prepared, so a process
    without the task store tells a task's turn from a root one without guessing.
    """

    record: ExternalWaitRecord | None = await session_store._read_external_wait(
        held.scope, held.source_key
    )
    if record is None or record.execution is None:
        return None
    return record.execution.intent.task_id


_TASK_STORE_REQUIRED = (
    "Session {session_id} waits on an external event inside task {task_id}, which this "
    "process can't see without the application's task store; {action} from a process "
    "that has it."
)


async def _require_visible_task(app: CayuApp, session_id: str, action: str) -> None:
    """Refuse an operation on a task-owned waiting session whose task this process can't see."""

    if app.task_store is not None:
        return
    held = await _held_wait(app.session_store, session_id, _HELD)
    task_id = None if held is None else await _wait_task_id(app.session_store, held)
    if task_id is not None:
        raise ValueError(
            _TASK_STORE_REQUIRED.format(session_id=session_id, task_id=task_id, action=action)
        )


async def _held_wait(
    session_store: SessionStore, session_id: str, states: frozenset[str]
) -> ParkedExternalWait | None:
    from cayu.sessions._session_continuation import ContinuationRecord
    from cayu.sessions._session_continuation_store import ROOT_KEY, ContinuationRoot

    checkpoint = await session_store.load_checkpoint(session_id)
    raw = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    if raw is None:
        return None
    root = ContinuationRoot.model_validate(raw)
    for entry in root.entries:
        if entry.purpose != _EXTERNAL_PURPOSE or entry.state not in states:
            continue
        record = await session_store.load_session_operation(session_id, entry.ticket_key)
        if record is None:
            raise ExternalWaitUnavailable("The session's external wait record is unavailable.")
        continuation = ContinuationRecord.model_validate(record)
        ticket = continuation.ticket
        if len(ticket.targets) != 1 or not ticket.registration_key.startswith(_REGISTRATION_PREFIX):
            raise ExternalWaitUnavailable("The session's external wait is malformed.")
        [target] = ticket.targets
        return ParkedExternalWait(
            wait_id=ticket.registration_key.removeprefix(_REGISTRATION_PREFIX),
            scope=ExternalWaitScope(
                application_scope=target.owner.application_scope,
                generation=int(target.owner.incarnation),
            ),
            source_key=target.object_id,
            delivering=continuation.consumption is not None,
        )
    return None


async def external_parked_wait(app: CayuApp, session_id: str) -> TaskParkedWait | None:
    """The parked wait of a task whose session waits on an external event.

    It is read from the session itself, so a process that doesn't host the waits
    still sees the task parked and never hands it to a generic continuation.
    """

    parked = await parked_external_wait(app, session_id)
    if parked is None or await _registration(app, parked) is None:
        return None
    return TaskParkedWait(kind="external_wait", wait_id=parked.wait_id)


async def _registration(
    app: CayuApp, parked: ParkedExternalWait
) -> ExternalWaitRegistration | None:
    """The registration of the wait a session parked on; ``None`` once it's gone.

    A wait whose scope was pruned, or whose key now names another registration, has
    nothing left to continue or protect.
    """

    return await _registration_in(app.session_store, parked)


async def _registration_in(
    session_store: SessionStore, parked: ParkedExternalWait
) -> ExternalWaitRegistration | None:
    record: ExternalWaitRecord | None = await session_store._read_external_wait(
        parked.scope, parked.source_key
    )
    if (
        record is None
        or record.registration is None
        or external_wait_digest(record.registration) != parked.wait_id
    ):
        return None
    return record.registration


def ended_during_delivery(session_id: str, task_id: str) -> ExternalWaitUnavailable:
    """A task that ended while its event's delivery was being admitted: never continued.

    The prepared admission can only be settled by delivering it, which needs the
    task, so the wait stays pending and its session is left as it is.
    """

    return ExternalWaitUnavailable(
        f"Task {task_id} ended while its event was being delivered to session {session_id}; "
        "the session is not continued."
    )


async def task_ended_during_delivery(
    hosting: ExternalWaitHosting, registration: ExternalWaitRegistration
) -> str | None:
    """The task that ended while this wait's event was being delivered, if any.

    Nothing can settle such a wait any more, so it must not be scheduled again.
    """

    from cayu.runtime._external_wait_binding import load_prepared_continuation

    app = hosting.app
    request = registration.correlation.request
    retained = await app.session_store._read_external_wait(request.scope, request.correlation_key)
    if retained is None or retained.registration != registration or retained.continuation is None:
        return None
    native = await load_prepared_continuation(app.session_store, retained.continuation)
    if native is None or native.consumption is None or native.ticket.state != "WAITING":
        return None
    session_id = retained.continuation.intent.session_id
    try:
        task_id = await hosting.adapter._waiting_turn_task_id(retained, context=hosting.context)
        if task_id is None or not await task_ended(app, task_id, session_id):
            return None
    except ExternalWaitUnavailable:
        return None
    return task_id


async def task_ended(app: CayuApp, task_id: str, session_id: str) -> bool:
    """Whether the task a waiting turn ran under ended, so its wait must not continue it."""

    task_store = app.task_store
    if task_store is None:
        # Not seeing the task is not proof it ended: leave the wait to its owner.
        raise ExternalWaitUnavailable(
            "Servicing a task-owned external wait requires its task store."
        )
    owner = await owning_task(app, session_id)
    if owner is not None:
        if owner.id != task_id:
            raise ExternalWaitUnavailable("The waiting session belongs to another task.")
        return False
    task = await task_store.load_task(task_id)
    if task is None or task.status in TERMINAL_TASK_STATUSES:
        return True
    raise ExternalWaitUnavailable("The task that owns the waiting session is not running.")


async def service_under_task(
    adapter: SessionExternalWaitAdapter,
    boundary: _ExternalExecutionToWait,
    retained: ExternalWaitRecord,
    *,
    session_id: str,
    stage_id: str | None,
) -> None:
    """Continue a task-owned session under its task's claim bound to this wait.

    The worker that ran the session may still be handing the task off; this waits
    briefly, then raises :class:`ExternalWaitOwnerBusy` for the service to retry. A
    competing servicing that holds or finished the claim leaves nothing to do.
    """

    from cayu.tasks.worker import TaskHandlerOutcome

    app = adapter.app
    assert retained.registration is not None and retained.service is not None
    service = retained.service
    owner = await owning_task(app, session_id)
    if owner is None:
        raise ExternalWaitUnavailable("The task that owns the waiting session is not running.")
    wait_id = external_wait_digest(retained.registration)
    parked = TaskParkedWait(kind="external_wait", wait_id=wait_id)
    handed_off = await handed_off_owner(
        app, owner, parked=parked, still_waiting=True, busy=ExternalWaitOwnerBusy
    )
    if handed_off is None:
        raise ExternalWaitUnavailable("The waiting session's task is parked on another trigger.")
    owner, presented = handed_off
    assert owner.session_id is not None and owner.session_instance_id is not None
    assert owner.interrupted_handoff_id is not None
    failures: list[Exception] = []

    async def continue_owned(owned_app: CayuApp, claimed: Task, claimed_by: str):
        from cayu.runtime._session_continuation_owner import _ContinuationServiceNotStarted

        try:
            request = await boundary.service_request(
                retained,
                stage_id,
                task_claim=(claimed_by, claimed.interrupted_handoff_id),
            )
            # The owned path waits for the continuation to finish, however long its
            # model call or tool runs, so the claim's heartbeat supervises it; the
            # public path stops observing after its bound and would leave it unowned.
            await boundary.owner._service_owned(
                owned_app, request, service, participant_context=None
            )
        except _ContinuationServiceNotStarted as error:
            failures.append(error.failure)
        except Exception as error:
            failures.append(error)
        current = await owned_app.session_store.load(session_id)
        if current is not None and current.status is SessionStatus.INTERRUPTED:
            # Paused again, or never continued: the worker hands the task off,
            # parked while the session still waits on this wait.
            return TaskHandlerOutcome.SESSION_INTERRUPTED
        if failures:
            raise failures[0]
        return None

    claimed = await run_under_parked_claim(
        app,
        task_id=owner.id,
        session_id=owner.session_id,
        session_instance_id=owner.session_instance_id,
        handoff_id=owner.interrupted_handoff_id,
        parked=presented,
        operation=f"cayu.external-wait-continuation.v1:{wait_id}",
        worker_id=f"external-wait-{uuid4().hex[:12]}{continuation_suffix(wait_id)}",
        lease_seconds=_CONTINUATION_LEASE_SECONDS,
        step=continue_owned,
    )
    if failures:
        raise failures[0]
    if not claimed and await parked_external_wait(app, session_id) is not None:
        # Another servicing holds the claim; the service retries.
        raise ExternalWaitOwnerBusy(owner.id)


async def stop_wait(
    hosting: ExternalWaitHosting, registration: ExternalWaitRegistration, operation_key: str
) -> None:
    """Retire a wait so it never continues its session, as the hosting context.

    A wait still open is cancelled first, which needs the access policy to grant the
    hosting context ``cancel``. A parked session's wait is then retired; a
    preparation that never reached its wait is excluded.
    """

    correlation = registration.correlation
    current = await hosting.waits.inspect(correlation, context=hosting.context)
    if current.outcome is None:
        try:
            current = await hosting.waits.cancel(
                correlation, operation_key=operation_key, context=hosting.context
            )
        except PermissionError:
            raise PermissionError(
                "Stopping an external wait in a session a task owns cancels the wait as "
                "the hosting context; the access policy must grant it 'cancel'."
            ) from None
    assert current.outcome is not None
    if current.handoff == "unbound":
        if current.pending_handoff and not current.execution_excluded:
            await hosting.adapter.exclude_prepared_execution(registration, context=hosting.context)
    elif current.outcome.kind in {"cancelled", "unavailable"}:
        await hosting.adapter.service_wait(registration, context=hosting.context)
    elif current.pending_handoff:
        await hosting.adapter.retire_execution(
            registration, operation_key=operation_key, context=hosting.context
        )


async def stop_wait_of_ended_task(
    hosting: ExternalWaitHosting, registration: ExternalWaitRegistration
) -> bool:
    """Stop a parked wait whose task ended before its event or deadline; whether it did."""

    app = hosting.app
    request = registration.correlation.request
    retained = await app.session_store._read_external_wait(request.scope, request.correlation_key)
    if retained is None or retained.registration != registration or retained.continuation is None:
        return False
    session_id = retained.continuation.intent.session_id
    if app.task_store is None or not await app.task_store.list_tasks(
        TaskQuery(session_id=session_id, limit=1)
    ):
        # No task ever ran this session: a root wait, with nothing more to read.
        return False
    task_id = await hosting.adapter._waiting_turn_task_id(retained, context=hosting.context)
    if task_id is None or not await task_ended(app, task_id, session_id):
        return False
    await stop_wait(hosting, registration, f"task-ended:{task_id}")
    return True


async def stop_parked_external_wait(app: CayuApp, session_id: str) -> None:
    """Retire the wait of a task-owned session an admitted interruption stops.

    The task stays parked on the retired wait until the worker's
    ``reconcile_parked_handoffs`` hands it back unparked. A root session keeps its
    wait, as before.
    """

    parked = await parked_external_wait(app, session_id)
    if parked is None:
        return
    await _require_visible_task(app, session_id, "interrupt it")
    if await owning_task(app, session_id) is None:
        return
    if parked.delivering:
        raise ValueError(
            f"Session {session_id}'s external event is already being delivered to it; "
            "interrupt the session once it runs, or cancel its task."
        )
    hosting = app._external_wait_hosting
    if hosting is None:
        raise ValueError(
            f"Session {session_id} waits on an external event inside a task. Interrupt it "
            "from a process that hosts its waits (app.host_external_waits), which stops "
            "the wait, or cancel the task."
        )
    registration = await _registration(app, parked)
    if registration is None:
        return
    session = await app.session_store.load(session_id)
    assert session is not None
    await stop_wait(hosting, registration, f"interrupt:{session.instance_id}:{session.run_epoch}")


async def require_deletable(app: CayuApp, session_id: str) -> None:
    """Refuse to delete a session waiting on an external event while its task runs."""

    if await external_parked_wait(app, session_id) is None:
        return
    await _require_visible_task(app, session_id, "delete it")
    owner = await owning_task(app, session_id)
    if owner is not None:
        raise ValueError(
            f"Session {session_id} waits on an external event inside task {owner.id}, which "
            "is still running. Cancel the task first (a parked task is cancelled at once), "
            "then delete the session."
        )


async def stop_wait_left_arming(app: CayuApp, session_id: str) -> None:
    """Stop the wait of an interrupted session whose turn died before parking on it.

    Another process interrupted the turn plainly, so it can't be brought to its wait
    any more, and the still-armed wait would fence the task's own continuation.
    """

    hosting = app._external_wait_hosting
    if hosting is None:
        return
    held = await parked_external_wait(app, session_id, states=_ARMING)
    registration = None if held is None else await _registration(app, held)
    if registration is None:
        return
    _logger.warning(
        "The turn of session %s was interrupted before its external wait parked; its wait "
        "is stopped and its task handed back.",
        session_id,
    )
    await stop_wait(hosting, registration, f"lost-before-wait:{registration.operation_key}")


async def recover_before_handoff(app: CayuApp, session_id: str) -> bool:
    """Bring a task-owned turn that died before its wait parked to that wait, or retire it.

    The turn is recovered to its wait when its model step completed. One that can't
    reach it never runs again blind: its wait is retired, which recovers the session
    through the ordinary interruption, and the task is handed back unparked. Returns
    whether the session held such a wait; while its writer may still be alive this
    returns ``True`` and changes nothing, for the next pass to retry.
    """

    parked = await parked_external_wait(app, session_id, states=_ARMING)
    if parked is None:
        return False
    hosting = app._external_wait_hosting
    if hosting is None:
        # Only the wait's own recovery may bring the turn to it; plain recovery
        # would leave it to run again blind. A worker that hosts the waits does it.
        reported = _UNHOSTED_TURNS_REPORTED.setdefault(app, set())
        (_logger.debug if session_id in reported else _logger.warning)(
            "The turn of session %s died before its external wait parked; a worker that "
            "hosts the waits (app.host_external_waits) recovers it.",
            session_id,
        )
        reported.add(session_id)
        return True
    store = app.session_store
    if store.supports_session_execution:
        if (await store.inspect_session_execution(session_id)).state == "executing":
            return True
        inactive_for_seconds = 0
    else:
        # Without execution leases, only inactivity says its writer is gone.
        session = await store.load(session_id)
        inactive_for_seconds = EXTERNAL_RECOVERY_INACTIVITY_S
        if session is None or session.updated_at > app._clock() - timedelta(
            seconds=inactive_for_seconds
        ):
            return True
    registration = await _registration(app, parked)
    if registration is None:
        # No wait is left to bring the turn to: recover the session plainly.
        return False
    try:
        await hosting.adapter.recover_to_wait(
            registration, context=hosting.context, inactive_for_seconds=inactive_for_seconds
        )
    except ExternalWaitUnavailable:
        _logger.warning(
            "The turn of session %s died before its external wait parked and can't be "
            "recovered to it; its wait is retired and its task handed back.",
            session_id,
        )
        await stop_wait(hosting, registration, f"lost-before-wait:{registration.operation_key}")
    return True
