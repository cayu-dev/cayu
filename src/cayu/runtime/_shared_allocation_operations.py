"""List and release a workflow run's recorded shared allocations by hand.

A workflow run that crashed and is never resumed keeps the allocation its
steps shared (`cayu.runtime._shared_allocation_records`): nothing ends the run,
so nothing releases it, and recovery only reaps it once the run has ended.
These are the operator's way out, behind ``CayuApp.list_shared_allocations``
and ``CayuApp.release_shared_allocation`` (and the HTTP routes and CLI built on
them). They read the same records and go through the same fences as
``reap_recorded``:

- Nothing is released while this process runs the workflow run or holds the
  allocation; the run releases it itself. ``force`` doesn't change that.
- Nothing is released while the run looks like it is running somewhere else:
  one of its step sessions is running under an owner that hasn't been lost,
  or the newest event in its journal is within ``inactive_for_seconds``. (Not
  the run session's ``last_activity_at``: writing these records bumps it.)
  ``force`` skips this check, for an operator who knows the process is gone.
- The release claims the record before anything is disposed of
  (``claim_release``): a compare-and-set against the record that was read
  that fences it as ``reaping``, raises its claim generation and holds it
  under a lease the release renews while it runs. A run that resumes after
  that never reattaches to it (its claim moves the record to the orphans and
  starts a fresh allocation), a process that still holds it lets go without
  copying back or disposing of anything, and neither another release nor
  recovery's reaping touches it until the lease is given up or expires.
- A discard reconnects and discards without copy-back, or disposes of an
  unreachable allocation through ``discard_unreachable_allocation``, exactly as
  recovery does. A copy-back reconnects, rebuilds the binding from its
  recorded state (nothing is copied in again), and finalizes it as a completed
  run would. A failure gives the lease up and keeps the record, still fenced,
  so the release can be tried again (or recovery discards it once it is an
  orphan or the run ends). A binding that a failed finalize left holding its
  finalization state (``abandon`` returned False) is kept in this process, and
  the next release of that allocation here finalizes it again, which also
  releases its workspace reservations.
"""

from __future__ import annotations

import contextlib
import logging
import secrets
import weakref
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from cayu.environments._finalization_disposal import finalization_disposal_checkpoint
from cayu.environments.factory import EnvironmentFactoryOperation, EnvironmentFactoryRequest
from cayu.environments.shared_allocations import (
    SharedAllocation,
    SharedAllocationPage,
    SharedAllocationRelease,
    SharedAllocationReleaseMode,
    SharedAllocationReleaseRefused,
    SharedAllocationRunState,
    SharedAllocationState,
)
from cayu.runtime._shared_allocation_records import (
    DISPOSAL_LEASE_SECONDS,
    DisposalHold,
    SharedAllocationRecord,
    SharedAllocationRecords,
    SharedAllocationSnapshot,
    keeping_disposal_lease,
    shared_allocation_owner,
)
from cayu.runtime._shared_environment_allocations import (
    _discard_resources,
    _dispose_recorded,
    _positively_disposed,
    _require_result,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.queries import SessionOrder, SessionQuery
from cayu.sessions.records import Session, SessionStatus

if TYPE_CHECKING:
    from cayu.environments.factory import EnvironmentFactory
    from cayu.runtime._shared_environment_allocations import SharedEnvironmentAllocations
    from cayu.sessions.store import SessionStore

_LOGGER = logging.getLogger(__name__)
_WORKFLOW_JOURNAL_PROVIDER = "cayu.workflow"

#: How long a run must have journaled nothing before a release treats it as stopped.
DEFAULT_INACTIVE_FOR_SECONDS = 900
#: The most workflow runs one listing page reads.
MAX_SHARED_ALLOCATION_RUNS_PER_PAGE = 200
# The most running step sessions of one run whose liveness a check reads, and
# how many it reads per page.
_MAX_STEP_SESSIONS_CHECKED = 1000
_STEP_SESSION_PAGE = 100
# Bindings a failed copy-back left mid-finalization, per registry, by allocation.
_RETAINED_BINDINGS: weakref.WeakKeyDictionary[Any, dict[str, tuple[Any, Any]]] = (
    weakref.WeakKeyDictionary()
)

_Redact = Callable[[Any], Any]


def inactivity_window(seconds: int | None) -> int:
    """``seconds``, or the default when None; refuses anything but a non-negative int."""

    window = DEFAULT_INACTIVE_FOR_SECONDS if seconds is None else seconds
    if type(window) is not int or window < 0:
        raise ValueError("inactive_for_seconds must be a non-negative integer.")
    return window


async def list_shared_allocations(
    registry: SharedEnvironmentAllocations,
    *,
    session_store: SessionStore,
    redact: _Redact,
    now: datetime,
    workflow_run_id: str | None,
    environment_name: str | None,
    limit: int,
    cursor: str | None,
    inactive_for_seconds: int = DEFAULT_INACTIVE_FOR_SECONDS,
) -> SharedAllocationPage:
    """Return the recorded allocations of up to ``limit`` workflow runs, newest run first."""

    records = registry.durable_records
    if records is None:
        return SharedAllocationPage(())
    next_cursor: str | None = None
    if workflow_run_id is not None:
        run = await session_store.load(workflow_run_id)
        runs = [run] if run is not None and run.provider_name == _WORKFLOW_JOURNAL_PROVIDER else []
    else:
        page = await session_store.list_sessions(
            SessionQuery(
                provider_name=_WORKFLOW_JOURNAL_PROVIDER,
                order_by=SessionOrder.CREATED_AT_DESC,
                limit=limit,
                cursor=cursor,
            )
        )
        runs = page.sessions
        next_cursor = page.next_cursor
    allocations: list[SharedAllocation] = []
    for run in runs:
        snapshot = await records.load(run.id)
        if snapshot is None:
            continue
        entries = [
            (record, orphan)
            for record, orphan in _entries(snapshot)
            if environment_name is None or record.environment_name == environment_name
        ]
        if not entries:
            continue
        run_state, reason = await _run_state(
            registry, session_store, run, now=now, inactive_for_seconds=inactive_for_seconds
        )
        allocations.extend(
            _describe(
                record, orphan=orphan, run=run, run_state=run_state, reason=reason, redact=redact
            )
            for record, orphan in entries
        )
    return SharedAllocationPage(tuple(allocations), next_cursor)


async def release_shared_allocation(
    registry: SharedEnvironmentAllocations,
    allocation_id: str,
    *,
    session_store: SessionStore,
    redact: _Redact,
    now: datetime,
    mode: SharedAllocationReleaseMode,
    force: bool,
    inactive_for_seconds: int,
) -> SharedAllocationRelease:
    """Release one recorded allocation; raises ``KeyError`` when there is no such record."""

    records = registry.durable_records
    owner_id = shared_allocation_owner(allocation_id)
    if records is None or owner_id is None:
        raise KeyError(allocation_id)
    run = await session_store.load(owner_id)
    if run is None or run.provider_name != _WORKFLOW_JOURNAL_PROVIDER:
        raise KeyError(allocation_id)
    snapshot = await records.load(owner_id)
    found = None if snapshot is None else _find(snapshot, allocation_id)
    if found is None:
        raise KeyError(allocation_id)
    record_id, record, orphan = found
    run_state, reason = await _run_state(
        registry, session_store, run, now=now, inactive_for_seconds=inactive_for_seconds
    )
    allocation = _describe(
        record, orphan=orphan, run=run, run_state=run_state, reason=reason, redact=redact
    )
    environment = record.environment_name
    if registry.owner_active_here(owner_id) or registry.holds_allocation(allocation_id):
        raise SharedAllocationReleaseRefused(
            "held_here",
            f"Workflow run {owner_id!r} is running in this process, which holds its shared "
            f"allocation of environment {environment!r}. The run releases it when it ends; "
            "stop the run (or call ctx.release_environments()) instead.",
        )
    if run_state == "running" and not force:
        raise SharedAllocationReleaseRefused(
            "owner_running",
            f"Workflow run {owner_id!r} may still be running: {reason} Releasing its "
            "allocation now would pull the sandbox out from under it. Wait until it has "
            "stopped, or release with force if you know its process is gone.",
        )
    factory = registry.registered_factory(environment)
    if factory is None:
        raise SharedAllocationReleaseRefused(
            "environment_not_registered",
            f"Environment {environment!r} is not registered in this application, so its "
            "allocation can't be reached. Release it from an application that registers it.",
        )
    if mode is SharedAllocationReleaseMode.COPY_BACK:
        refusal = _copy_back_refusal(record, orphan=orphan)
        if refusal is not None:
            raise SharedAllocationReleaseRefused("not_copyable", refusal)
    elif not (
        record.reconnect_metadata
        or record.result_metadata
        or record.disposal_state
        or record.allocation_intent
    ):
        raise SharedAllocationReleaseRefused(
            "not_disposable",
            f"The allocation of environment {environment!r} was recorded before its factory "
            "returned, so there is nothing to reach it by. What the creation made lapses on "
            "its own, and recovery drops the record once the run has ended.",
        )
    token = f"release-{secrets.token_hex(8)}"
    held = await records.claim_release(record, token=token, lease_seconds=DISPOSAL_LEASE_SECONDS)
    if held is None:
        raise SharedAllocationReleaseRefused(
            "changed",
            "The allocation's record changed while it was read, or recovery or another "
            "release is disposing of it (another attempt claimed it, or something is "
            "releasing it). List the allocations again.",
        )
    async with keeping_disposal_lease(records, held) as hold:
        message = await _release_held(
            registry, allocation_id, records, record_id, hold, factory, mode=mode
        )
    fenced = hold.record
    await records.remove_reaped(fenced)
    forced = force and run_state == "running"
    _LOGGER.warning(
        "Released shared allocation %s of workflow run %s by hand (%s%s).",
        allocation_id,
        owner_id,
        mode.value,
        ", forced" if forced else "",
    )
    return SharedAllocationRelease(allocation=allocation, mode=mode, forced=forced, message=message)


async def _release_held(
    registry: SharedEnvironmentAllocations,
    allocation_id: str,
    records: SharedAllocationRecords,
    record_id: str | None,
    hold: DisposalHold,
    factory: EnvironmentFactory,
    *,
    mode: SharedAllocationReleaseMode,
) -> str:
    """Copy back or discard the allocation this release holds the lease on."""

    environment = hold.record.environment_name
    owner_id = hold.record.owner_session_id
    try:
        message = await _settle_retained(registry, allocation_id, records, record_id, hold)
        if mode is SharedAllocationReleaseMode.COPY_BACK and message is None:
            assert record_id is not None
            message = await _copy_back(registry, records, factory, record_id, hold)
        elif message is None:
            if not await _dispose_recorded(factory, hold.record, records):
                raise RuntimeError(
                    f"Environment {environment!r} couldn't confirm that the allocation was "
                    "disposed of. Its record is kept, fenced so no run reattaches to it; "
                    "release it again or let recovery retry once the run has ended."
                )
            message = (
                f"Discarded the shared allocation of environment {environment!r} that "
                f"workflow run {owner_id!r} recorded; nothing was copied back."
            )
    except BaseException:
        # Give the lease up so the release can be tried again at once; the record
        # stays fenced as reaping, so no run reattaches to it meanwhile.
        with contextlib.suppress(Exception):
            await records.drop_release(hold.record)
        raise
    assert message is not None
    return message


def _retained(registry: SharedEnvironmentAllocations) -> dict[str, tuple[Any, Any]]:
    retained = _RETAINED_BINDINGS.get(registry)
    if retained is None:
        retained = _RETAINED_BINDINGS[registry] = {}
    return retained


async def _settle_retained(
    registry: SharedEnvironmentAllocations,
    allocation_id: str,
    records: SharedAllocationRecords,
    record_id: str | None,
    hold: DisposalHold,
) -> str | None:
    """Finish an earlier release's finalization that this process still holds.

    A binding whose ``abandon`` returned False after a failed finalize (Docker
    coding, when the copy-back succeeded but container removal didn't) keeps
    its finalization state and its process-local workspace reservations. Only
    finalizing that same bound workspace again settles both, so it is retried
    here, with the outcome it was finalized with, before anything else.
    """

    retained = _retained(registry).get(allocation_id)
    if retained is None:
        return None
    binding, bound = retained
    await _finalize(records, record_id, hold, binding, bound)
    _retained(registry).pop(allocation_id, None)
    return (
        f"Finished the copy-back and disposal of the shared allocation of environment "
        f"{hold.record.environment_name!r} that an earlier release in this process left "
        "unfinished."
    )


async def _finalize(
    records: SharedAllocationRecords,
    record_id: str | None,
    hold: DisposalHold,
    binding: Any,
    bound: Any,
) -> None:
    async def checkpoint_disposal(state: dict[str, Any]) -> None:
        if record_id is not None:
            hold.record = await records.update(record_id, hold.record, disposal_state=state)

    token = finalization_disposal_checkpoint.set(checkpoint_disposal)
    try:
        await binding.finalize(bound, outcome="completed")
    finally:
        finalization_disposal_checkpoint.reset(token)


def _entries(
    snapshot: SharedAllocationSnapshot,
) -> list[tuple[SharedAllocationRecord, bool]]:
    return [
        *((record, False) for record in snapshot.records.values()),
        *((record, True) for record in snapshot.orphans.values()),
    ]


def _find(
    snapshot: SharedAllocationSnapshot, allocation_id: str
) -> tuple[str | None, SharedAllocationRecord, bool] | None:
    for record_id, record in snapshot.records.items():
        if record.allocation_session_id == allocation_id:
            return record_id, record, False
    record = snapshot.orphans.get(allocation_id)
    return None if record is None else (None, record, True)


def _state(record: SharedAllocationRecord, *, orphan: bool) -> SharedAllocationState:
    if record.state == "reaping":
        return "reaping"
    if orphan:
        return "orphan"
    return "creating" if record.state == "creating" else "live"


def _describe(
    record: SharedAllocationRecord,
    *,
    orphan: bool,
    run: Session,
    run_state: SharedAllocationRunState,
    reason: str | None,
    redact: _Redact,
) -> SharedAllocation:
    workflow_name = run.metadata.get("cayu.workflow")
    created_at: datetime | None = None
    if record.created_at is not None:
        with contextlib.suppress(ValueError):
            created_at = datetime.fromisoformat(record.created_at)
    metadata = redact(dict(record.result_metadata))
    return SharedAllocation(
        allocation_id=record.allocation_session_id,
        workflow_run_id=record.owner_session_id,
        workflow_name=workflow_name if type(workflow_name) is str else None,
        environment_name=record.environment_name,
        agent_name=record.agent_name,
        scope=record.scope,
        scope_id=record.scope_id,
        state=_state(record, orphan=orphan),
        reconnectable=not orphan and record.state != "reaping" and record.reconnectable,
        attempt_id=record.attempt_id,
        claim_generation=record.claim_generation,
        created_at=created_at,
        run_state=run_state,
        run_state_reason=reason,
        metadata=metadata if type(metadata) is dict else {},
    )


async def _run_state(
    registry: SharedEnvironmentAllocations,
    session_store: SessionStore,
    run: Session,
    *,
    now: datetime,
    inactive_for_seconds: int,
) -> tuple[SharedAllocationRunState, str | None]:
    """Whether the run looks like it is running, has stopped, or has ended, and why."""

    if registry.owner_active_here(run.id):
        return "running", "This process runs it or holds its shared allocations."
    if await registry.owner_ended(run):
        return "ended", "Its newest attempt has completed."
    step = await _running_step_reason(session_store, run.id)
    if step is not None:
        return "running", step
    # The run's own journal, not its last_activity_at: writing the shared
    # allocation records (a release, recovery's reaping) bumps that, and isn't
    # the run doing anything.
    newest = await session_store.query_events(
        EventQuery(session_id=run.id, order_by=EventOrder.SEQUENCE_DESC, limit=1)
    )
    last_journaled = newest[0].event.timestamp if newest else run.created_at
    quiet = (now - last_journaled).total_seconds()
    if quiet < inactive_for_seconds:
        return (
            "running",
            f"It journaled activity {max(0, int(quiet))}s ago, within the "
            f"{inactive_for_seconds}s inactivity window.",
        )
    return "stopped", None


async def _running_step_reason(session_store: SessionStore, run_id: str) -> str | None:
    """Why a step session of the run may be running, or None when none can be.

    Every running or interrupting step session is checked, page by page. Past
    ``_MAX_STEP_SESSIONS_CHECKED`` of them the run counts as running, rather
    than as stopped on a partial look.
    """

    checked = 0
    for status in (SessionStatus.RUNNING, SessionStatus.INTERRUPTING):
        cursor: str | None = None
        while True:
            page = await session_store.list_sessions(
                SessionQuery(
                    parent_session_id=run_id,
                    status=status,
                    limit=_STEP_SESSION_PAGE,
                    cursor=cursor,
                )
            )
            for child in page.sessions:
                checked += 1
                if checked > _MAX_STEP_SESSIONS_CHECKED:
                    return (
                        f"It has more than {_MAX_STEP_SESSIONS_CHECKED} running step sessions, "
                        "too many to check."
                    )
                try:
                    execution = await session_store.inspect_session_execution(child.id)
                except KeyError:
                    continue
                # A store without execution leases reports "unknown"; treat that as live.
                if execution.state not in {"owner_lost", "terminal"}:
                    return f"Its step session {child.id!r} is running."
            cursor = page.next_cursor
            if cursor is None or not page.sessions:
                break
    return None


def _copy_back_refusal(record: SharedAllocationRecord, *, orphan: bool) -> str | None:
    environment = record.environment_name
    if orphan:
        return (
            f"This allocation of environment {environment!r} is an orphan: a later attempt "
            "of the run replaced it, so its work isn't the run's to copy back. Discard it."
        )
    if record.disposal_state is not None:
        # Its copy-back already happened; only the disposal is left.
        return None
    if not record.reconnect_metadata:
        return (
            f"Environment {environment!r} can't reconnect to this allocation (its factory "
            "published no reconnect metadata), so its work can't be copied back. Discard it."
        )
    if record.binding_required and record.binding_state is None:
        return (
            f"This allocation of environment {environment!r} was never bound (or its binding "
            "left no recovery state), so there is no work to copy back. Discard it."
        )
    return None


async def _copy_back(
    registry: SharedEnvironmentAllocations,
    records: SharedAllocationRecords,
    factory: EnvironmentFactory,
    record_id: str,
    hold: DisposalHold,
) -> str:
    """Reconnect, rebuild the binding from its recorded state, and finalize it as completed."""

    fenced = hold.record
    environment = fenced.environment_name
    request = EnvironmentFactoryRequest(
        session_id=fenced.allocation_session_id,
        agent_name=fenced.agent_name,
        environment_name=environment,
        operation=EnvironmentFactoryOperation.RECONNECT,
        reconnect_metadata=fenced.reconnect_metadata,
        ownership_epoch=fenced.claim_generation,
        # The allocation's identity isn't a session; events about it belong on
        # the run that owned it.
        event_session_id=fenced.owner_session_id,
    )
    if fenced.disposal_state is not None:
        await factory.recover_finalization_disposal(request, fenced.disposal_state)
        return (
            f"The shared allocation of environment {environment!r} had already copied its "
            "work back; its disposal is now finished."
        )
    if await _positively_disposed(factory, request):
        return (
            f"The shared allocation of environment {environment!r} was already gone, so "
            "nothing could be copied back; its record is removed."
        )
    result = await factory.create(request)
    _require_result(result)
    environment_result = result.environment
    binding = environment_result.binding
    if binding is None or not fenced.binding_required:
        await _discard_resources(result)
        return (
            f"Environment {environment!r} has no binding that copies work back; the shared "
            "allocation is disposed of."
        )
    assert fenced.binding_state is not None
    bound = await binding._recover_completion_finalization(
        environment_result.workspace,
        environment_result.runner,
        session_id=fenced.allocation_session_id,
        agent_name=fenced.agent_name,
        environment_name=environment,
        recovery_state=fenced.binding_state,
    )
    try:
        await _finalize(records, record_id, hold, binding, bound)
    except BaseException:
        # The allocation, and its fenced record, stay for another try. A binding
        # that keeps its finalization state (and its workspace reservations)
        # can only be settled by finalizing it again, so keep it for that.
        abandoned = False
        with contextlib.suppress(Exception):
            abandoned = binding.abandon(bound)
        if not abandoned:
            _retained(registry)[fenced.allocation_session_id] = (binding, bound)
        raise
    return (
        f"Copied the work of the shared allocation of environment {environment!r} back to "
        "the source workspace and disposed of the allocation."
    )
