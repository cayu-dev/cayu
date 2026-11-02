"""Process-local owner of factory allocations that several sessions share.

An environment registered with an ``allocation_scope`` other than ``session``
(or a workflow step given one through ``StepRunOptions``) gets one allocation
per scope instead of one per session. The first session of a scope asks the
registered factory for the allocation, under a request whose ``session_id``
names the scope's owner session plus a fresh suffix, and binds the factory's
workspace once. Every session of the scope, the first one included, then attaches to that
live allocation: it gets the bound workspace and runner, and a binding whose
``finalize`` only detaches. The allocation is finalized through the factory's
own binding (copy-back and disposal) when the scope's owner session ends: the
workflow run for a workflow anchor, otherwise the owner's run reaching its
terminal event. A session ending never tears the allocation down.

A workflow run's shared allocations are also recorded durably, in the run's
checkpoint (``cayu.runtime._shared_allocation_records``), before the factory is
called. A run resumed in another process finds the record and, when the
factory published reconnect metadata and the binding can rebuild its bound
workspace from durable state, reconnects to the same allocation through the
factory's RECONNECT path instead of creating one; nothing is copied in again.
Each record names the workflow attempt that owns it, and every change is a
compare-and-set, so a superseded attempt never disposes of an allocation a
newer one adopted. An allocation that can't be reconnected (a brokered
sandbox) is left behind as an orphan record when another attempt takes over,
and ``recover_incomplete_sessions`` disposes of it, as it does of records whose
attempt completed.

An agent session that owns a scope (its subagents share its sandbox) records
it the same way in its own checkpoint when the session store outlives the
process. The claim then names one run epoch of the session instead of a
workflow attempt, and the attempt fence checks that the session hasn't started
a newer run since. A run of the session in another process (after a crash)
reconnects to the allocation, and ``recover_incomplete_sessions`` reaps it
once the session has ended (completed, or failed, which includes a close or an
abandonment). With a store that ends with the process, an agent session's
scope stays process-local, unless its factory creates only recoverably and so
needs a record for its allocation intent. Requests carry this process's ``process_local_owner``
token only when the session store doesn't outlive the process, as an ordinary
session's do, so ``reap_orphaned_docker_coding_containers`` never removes an
allocation a durable record still owns.

A factory that declares an ``allocation_scope`` creates the shared allocation
through ``create_recoverable``, with an allocation intent kept in the record
(``cayu.runtime._shared_allocation_intents``), so a crash during the factory's
own create can be recovered by the next attempt or reaped by recovery.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from cayu.environments._finalization_disposal import finalization_disposal_checkpoint
from cayu.environments.base import Environment
from cayu.environments.bindings import (
    BoundWorkspace,
    WorkspaceBinding,
    WorkspaceSnapshot,
    _binding_copies_back,
    _validate_bind_request,
    _validate_finalize_request,
)
from cayu.environments.deferred import DeferredWorkspaceBinding
from cayu.environments.factory import (
    EnvironmentAllocationScope,
    EnvironmentAllocationState,
    EnvironmentAllocationUnsupportedError,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.environments.process_ownership import current_process_local_owner
from cayu.environments.shared_allocations import (
    AllocationSharingScope,
    allocation_sharing_scope,
)
from cayu.runtime._shared_allocation_intents import SharedAllocationIntentContext
from cayu.runtime._shared_allocation_records import (
    MAX_COPIED_BACK_SESSIONS,
    SharedAllocationClaimLost,
    SharedAllocationRecord,
    SharedAllocationRecords,
    keeping_disposal_lease,
    shared_allocation_record_id,
    shared_allocation_session_id,
)
from cayu.runtime._shared_allocation_writers import SharedAllocationWriterTurns
from cayu.runtime._shared_session_parts import (
    require_shareable_factory,
    require_shareable_result,
    session_environment,
    session_parts,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.forks import session_fork_profile_relationship
from cayu.sessions.records import Session, SessionStatus

if TYPE_CHECKING:
    from cayu.environments.admission import (
        ExecutionAdmissionCandidate,
        ExecutionEnvironmentAuthority,
    )
    from cayu.immutable_inputs import ImmutableInputAdapterCapability
    from cayu.runners.base import Runner, RunnerWorkloadAuthority
    from cayu.sessions.store import SessionStore
    from cayu.workspaces.revisions import WorkspaceRevisionObservation

_LOGGER = logging.getLogger(__name__)
_MAX_LINEAGE_DEPTH = 64
_MAX_ENDED_OWNERS = 4096
_MAX_JOURNALS_PER_WORKFLOW = 8
_WORKFLOW_JOURNAL_PROVIDER = "cayu.workflow"


class SharedAllocationScopeClosedError(RuntimeError):
    """Raised when a session asks to share an allocation whose owner isn't running here."""


class SharedAllocationCopyBackPendingError(RuntimeError):
    """Raised when a release would dispose of work a failed copy-back after a step left."""


__all__ = [
    "SharedAllocationKey",
    "SharedAllocationScopeClosedError",
    "SharedEnvironmentAllocations",
    "shared_allocation_session_id",
]


@dataclass(frozen=True)
class _Claimant:
    """The owner attempt that may claim a scope's record, and its attempt fence.

    For a workflow run it is the workflow attempt, ordered by the journal's
    attempt markers. For an agent session it is one run epoch of the session
    (``run_epoch`` set), and the fence checks the session hasn't run again.
    """

    attempt_id: str
    fence: Callable[[], Awaitable[None]]
    run_epoch: int | None = None

    @property
    def workflow(self) -> bool:
        return self.run_epoch is None


@dataclass(frozen=True)
class SharedAllocationKey:
    """One shared allocation: its scope, the session that owns it, and its environment."""

    scope: AllocationSharingScope
    scope_id: str
    owner_session_id: str
    environment_name: str

    def payload(self) -> dict[str, str]:
        return {
            "scope": self.scope.value,
            "scope_id": self.scope_id,
            "owner_session_id": self.owner_session_id,
        }

    @property
    def record_id(self) -> str:
        return shared_allocation_record_id(self.scope.value, self.scope_id, self.environment_name)


@dataclass(eq=False)
class _Attachment:
    session_id: str
    released: bool = False
    # A copy-back's own attachment, which never waits for the copy-back gate.
    copy_back: bool = False


def _open_gate() -> asyncio.Event:
    gate = asyncio.Event()
    gate.set()
    return gate


@dataclass(eq=False)
class _SharedAllocation:
    key: SharedAllocationKey
    factory: EnvironmentFactory
    result: EnvironmentFactoryResult
    bound: BoundWorkspace | None
    attachments: set[_Attachment] = field(default_factory=set)
    release_outcome: str | None = None
    release_task: asyncio.Task[None] | None = None
    discard_task: asyncio.Task[bool] | None = None
    # Discard progress, so a retry skips what already succeeded: the binding
    # released its state, and the factory disposed of the resources.
    binding_released: bool = False
    resources_discarded: bool = False
    # The durable record of a workflow-owned allocation, and the attempts of
    # this process that may act on it; None for a process-local allocation.
    record: SharedAllocationRecord | None = None
    attempts: set[str] = field(default_factory=set)
    # Serializes this process's reclaims of the record, so two sessions of a
    # new attempt never both decide what to do with the entry.
    reclaim_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Serializes copy-backs after steps.
    copy_back_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Closed while a copy-back after a step runs, so no session or command
    # attaches and writes in the middle of it; set by every detach, so the
    # copy-back can wait for the ones already attached to leave.
    copy_back_gate: asyncio.Event = field(default_factory=_open_gate)
    detached: asyncio.Event = field(default_factory=asyncio.Event)
    # Step sessions whose copy-back failed in this process and hasn't been
    # published since; the record holds the durable copy.
    pending_copy_back: set[str] = field(default_factory=set)

    @property
    def environment(self) -> Environment:
        return self.result.environment

    @property
    def runner(self) -> Runner | None:
        if self.bound is not None and self.bound.runner is not None:
            return self.bound.runner
        return self.environment.runner


class SharedEnvironmentAllocations:
    """Hold the allocations shared by sessions of one scope in this process."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        session_running_here: Callable[[str], bool] | None = None,
    ) -> None:
        self._session_store = session_store
        self._session_running_here = session_running_here
        self._default_scopes: dict[str, AllocationSharingScope] = {}
        # Scopes the workflow layer chose for its step sessions, by owner. Only
        # trusted in-process callers set these; no session or model metadata can.
        self._session_scopes: dict[str, AllocationSharingScope] = {}
        self._sessions_by_owner: dict[str, set[str]] = {}
        self._open_owners: set[str] = set()
        # Owner session -> the run epoch whose end released its scope (None for
        # an explicitly opened owner such as a workflow run).
        self._ended_owners: OrderedDict[str, int | None] = OrderedDict()
        self._owner_generations: dict[str, int] = {}
        self._live: dict[SharedAllocationKey, _SharedAllocation] = {}
        self._creating: dict[SharedAllocationKey, asyncio.Task[_SharedAllocation]] = {}
        self._closing: set[_SharedAllocation] = set()
        self._background: set[asyncio.Task[None]] = set()
        # The identity of an allocation that hasn't been adopted or positively
        # discarded yet. A retry after a failed creation asks for the same one,
        # so it adopts or cleans up whatever the failed attempt left behind.
        self._allocation_ids: dict[SharedAllocationKey, str] = {}
        self._in_use = False
        # Owner session -> the workflow attempt that opened it most recently,
        # and the check that it is still the run's newest attempt.
        self._owner_attempts: dict[str, str] = {}
        self._owner_fences: dict[str, Callable[[], Awaitable[None]]] = {}
        # Durable records of workflow-owned allocations, once the runtime's
        # checkpoint store is bound.
        self._records: SharedAllocationRecords | None = None
        self._process_local_store: Callable[[], bool] = lambda: True
        self._factory_lookup: Callable[[str], EnvironmentFactory | None] = lambda name: None
        # Agent session owner -> (incarnation, run epoch) of the run this
        # process executes for it, as its own environment setup saw it.
        self._owner_runs: dict[str, tuple[str, int]] = {}
        self._session_run_here: Callable[[str], tuple[str, int] | None] = lambda session_id: None
        # Custom journal factories of this process's workflows, by workflow
        # name, so recovery can ask whether a run they journal has ended.
        self._workflow_journals: dict[str, list[Callable[[Any], Any]]] = {}
        # Step sessions the workflow asked to copy back after their step, and
        # the allocations each attached to in this process.
        self._copy_back_sessions: dict[str, set[SharedAllocationKey]] = {}
        # The writer turns that steps and commands opting in take, per run.
        self.writer_turns = SharedAllocationWriterTurns()

    def bind_session_presence(
        self,
        session_running_here: Callable[[str], bool],
        *,
        session_run_here: Callable[[str], tuple[str, int] | None] | None = None,
    ) -> None:
        """Tell the registry how to see whether a session's run is live in this process.

        ``session_run_here`` names the incarnation and run epoch of the run this
        process executes for a session, when it knows one.
        """

        self._session_running_here = session_running_here
        if session_run_here is not None:
            self._session_run_here = session_run_here

    def bind_durable_records(
        self, checkpoint_store: SessionStore, *, process_local_store: Callable[[], bool]
    ) -> None:
        """Record workflow-owned allocations in ``checkpoint_store``.

        ``process_local_store`` says whether that store ends with this process,
        in which case requests still carry this process's owner token.
        """

        self._records = SharedAllocationRecords(checkpoint_store)
        self._process_local_store = process_local_store

    def bind_factory_lookup(self, lookup: Callable[[str], EnvironmentFactory | None]) -> None:
        """Tell the registry how to find an environment's registered factory, for cleanup."""

        self._factory_lookup = lookup

    # Registration ---------------------------------------------------------

    def set_default_scope(self, environment_name: str, scope: AllocationSharingScope | str) -> None:
        resolved = allocation_sharing_scope(scope)
        if resolved is AllocationSharingScope.SESSION:
            self._default_scopes.pop(environment_name, None)
        else:
            self._default_scopes[environment_name] = resolved
            self._in_use = True

    def clear_default_scope(self, environment_name: str) -> None:
        self._default_scopes.pop(environment_name, None)

    def default_scope(self, environment_name: str) -> AllocationSharingScope:
        return self._default_scopes.get(environment_name, AllocationSharingScope.SESSION)

    # Scope resolution -----------------------------------------------------

    def requested_scope(self, session: Session, environment_name: str) -> AllocationSharingScope:
        """Return the scope ``session`` runs ``environment_name`` under, without I/O."""

        if session_fork_profile_relationship(session) is not None:
            # A fork works on its own copy of its source; it never joins the
            # source's scope.
            return AllocationSharingScope.SESSION
        assigned = self._session_scopes.get(session.id)
        return assigned if assigned is not None else self.default_scope(environment_name)

    def assign_session_scope(
        self,
        session_id: str,
        scope: AllocationSharingScope | str,
        *,
        owner_session_id: str,
    ) -> None:
        """Run ``session_id``'s environment under ``scope``, overriding its registration.

        The workflow layer calls this for a step's child session. The
        assignment is dropped when ``owner_session_id`` ends.
        """

        self._in_use = True
        self._session_scopes[session_id] = allocation_sharing_scope(scope)
        self._sessions_by_owner.setdefault(owner_session_id, set()).add(session_id)

    def assign_session_copy_back(self, session_id: str, *, owner_session_id: str) -> None:
        """Copy ``session_id``'s shared work back after its step, through ``copy_back_session``.

        The workflow layer calls this for a step run with
        ``StepRunOptions(copy_back="after_step")``. The assignment is dropped
        when ``owner_session_id`` ends.
        """

        self._in_use = True
        self._copy_back_sessions.setdefault(session_id, set())
        self._sessions_by_owner.setdefault(owner_session_id, set()).add(session_id)

    async def resolve_key(
        self,
        session: Session,
        environment_name: str,
        scope: AllocationSharingScope,
    ) -> SharedAllocationKey:
        if scope is AllocationSharingScope.PARENT:
            parent = session.parent_session_id
            if parent is not None:
                attached = self._attached_parent_key(parent, environment_name)
                if attached is not None:
                    return attached
            owner = parent or session.id
            return SharedAllocationKey(scope, owner, owner, environment_name)
        if scope is AllocationSharingScope.CAUSAL_BUDGET:
            owner = await self._top_ancestor(session)
            return SharedAllocationKey(
                scope, session.causal_budget_id or owner, owner, environment_name
            )
        raise ValueError("A session-scoped allocation is not shared.")

    def _attached_parent_key(
        self, parent_session_id: str, environment_name: str
    ) -> SharedAllocationKey | None:
        # A child of a session that is itself attached (a workflow step's
        # subagent) joins that session's allocation, so an agent and its
        # subagents share one sandbox wherever the agent runs.
        for key, entry in self._live.items():
            if (
                key.scope is AllocationSharingScope.PARENT
                and key.environment_name == environment_name
                and any(
                    attachment.session_id == parent_session_id and not attachment.released
                    for attachment in entry.attachments
                )
            ):
                return key
        return None

    async def _top_ancestor(self, session: Session) -> str:
        current = session
        seen = {session.id}
        for _ in range(_MAX_LINEAGE_DEPTH):
            parent_id = current.parent_session_id
            if parent_id is None or session_fork_profile_relationship(current) is not None:
                # A fork keeps its source's causal budget and names the source
                # as its parent, but works on its own copy: its descendants'
                # scope stops at the fork and never reaches the source's.
                return current.id
            if parent_id in seen:
                raise RuntimeError("Session lineage contains a cycle.")
            seen.add(parent_id)
            parent = await self._session_store.load(parent_id)
            if parent is None:
                return parent_id
            current = parent
        raise RuntimeError("Session lineage is too deep to resolve a causal_budget scope.")

    def attachment_factory(
        self,
        factory: EnvironmentFactory,
        *,
        session: Session,
        environment_name: str,
    ) -> EnvironmentFactory | None:
        """Return the factory a shared-scope session uses in place of ``factory``."""

        scope = self.requested_scope(session, environment_name)
        if scope is AllocationSharingScope.SESSION:
            return None
        self._in_use = True
        return _SharedAllocationFactory(
            self,
            factory,
            session=session,
            environment_name=environment_name,
            scope=scope,
        )

    # Owner lifetime -------------------------------------------------------

    def open_owner(
        self,
        owner_session_id: str,
        *,
        attempt_id: str | None = None,
        fence: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Let sessions share allocations owned by ``owner_session_id`` until it ends.

        Workflow runs open their anchor session this way, naming the workflow
        attempt and passing ``fence``, which raises once a newer attempt has
        taken the run over; their allocations are recorded durably. An agent
        session that runs in this process is open while its run is.
        """

        self._in_use = True
        self._open_owners.add(owner_session_id)
        self._ended_owners.pop(owner_session_id, None)
        if attempt_id is not None:
            self._owner_attempts[owner_session_id] = attempt_id
            if fence is not None:
                self._owner_fences[owner_session_id] = fence
            else:
                # Without its own fence an attempt's allocations aren't recorded.
                self._owner_fences.pop(owner_session_id, None)

    def _durable_claimant(self, owner_session_id: str) -> _Claimant | None:
        """The workflow attempt that records ``owner_session_id``'s allocations, if any."""

        attempt_id = self._owner_attempts.get(owner_session_id)
        fence = self._owner_fences.get(owner_session_id)
        if self._records is None or attempt_id is None or fence is None:
            return None
        return _Claimant(attempt_id, fence)

    async def _agent_claimant(
        self, owner_session_id: str, *, durable_only: bool = True
    ) -> _Claimant | None:
        """The run of an agent session owner that records its allocations, if any.

        The claimant is the owner's run that this process executes, as its own
        environment setup (or its invocation) saw it, never whatever run the
        store holds now: a process whose run was superseded must not act as
        the newer run. Without a known run here, the scope stays process-local.

        Only with a session store that outlives the process: otherwise nothing
        could resume the session elsewhere, and the scope stays process-local.
        ``durable_only=False`` records it anyway, for a factory that creates
        only recoverably and so needs a record to hold its allocation intent.
        """

        if (
            self._records is None
            or (durable_only and self._process_local_store())
            or owner_session_id in self._owner_attempts
        ):
            return None
        run = self._owner_runs.get(owner_session_id) or self._session_run_here(owner_session_id)
        if run is None:
            return None
        store = self._session_store
        instance_id, run_epoch = run

        async def fence() -> None:
            current = await store.load(owner_session_id)
            if current is None or (current.instance_id, current.run_epoch) != (
                instance_id,
                run_epoch,
            ):
                raise SharedAllocationClaimLost(
                    f"Session {owner_session_id!r} has started a newer run, which owns "
                    "its shared allocations now."
                )

        return _Claimant(f"run:{instance_id}:{run_epoch}", fence, run_epoch=run_epoch)

    async def _claimant(self, owner_session_id: str) -> _Claimant | None:
        return self._durable_claimant(owner_session_id) or await self._agent_claimant(
            owner_session_id
        )

    def _owner_generation(self, owner_session_id: str) -> int:
        return self._owner_generations.get(owner_session_id, 0)

    async def _require_open(self, key: SharedAllocationKey, session: Session | None) -> None:
        """Refuse unless ``key``'s owner is open; ``session`` is None for a workflow command."""

        owner = key.owner_session_id
        if owner in self._open_owners:
            return
        if session is not None and owner == session.id:
            # The owner itself is being set up for its own run.
            if self._ended_owners.get(owner, -1) != session.run_epoch:
                return
        elif self._session_running_here is not None and self._session_running_here(owner):
            root = await self._session_store.load(owner)
            if (
                root is not None
                and root.provider_name != _WORKFLOW_JOURNAL_PROVIDER
                and self._ended_owners.get(owner, -1) != root.run_epoch
            ):
                return
        raise SharedAllocationScopeClosedError(
            f"Environment {key.environment_name!r} shares one allocation per "
            f"{key.scope.value} scope, but the scope's owner session {owner!r} is not "
            "running in this process (or has already ended), so nothing would release "
            "the allocation. Run the session inside its workflow run or parent session, "
            "or give it allocation_scope='session'."
        )

    async def end_owner(
        self,
        owner_session_id: str,
        *,
        outcome: str,
        run_epoch: int | None = None,
        attempt_id: str | None = None,
    ) -> None:
        """Release every allocation ``owner_session_id`` owns.

        An allocation that sessions are still attached to is released when the
        last of them detaches. With ``attempt_id``, nothing happens when a
        newer workflow attempt has opened the owner since. Raises when
        finalizing an allocation fails; the allocation is then kept for a
        retry by a later ``end_owner`` or by ``close``.
        """

        if not self._in_use:
            # Nothing in this process has asked to share an allocation yet.
            return
        if (
            attempt_id is not None
            and self._owner_attempts.get(owner_session_id, attempt_id) != attempt_id
        ):
            # A newer attempt took the run over; its steps use the allocation.
            return
        claimant = self._durable_claimant(owner_session_id)
        if claimant is None and run_epoch is not None:
            # An agent session owner's run, taken before its run identity is
            # cleared below: what an earlier run of it recorded and this run
            # never reattached to (a subagent's environment it didn't use
            # again after resuming) is still this run's to release.
            claimant = await self._agent_claimant(owner_session_id)
        self._open_owners.discard(owner_session_id)
        if run_epoch is not None:
            self._owner_runs.pop(owner_session_id, None)
        for session_id in self._sessions_by_owner.pop(owner_session_id, ()):
            self._session_scopes.pop(session_id, None)
            self._copy_back_sessions.pop(session_id, None)
        self._ended_owners[owner_session_id] = run_epoch
        self._ended_owners.move_to_end(owner_session_id)
        while len(self._ended_owners) > _MAX_ENDED_OWNERS:
            self._ended_owners.popitem(last=False)
        if any(key.owner_session_id == owner_session_id for key in self._creating) or any(
            key.owner_session_id == owner_session_id for key in self._live
        ):
            self._owner_generations[owner_session_id] = self._owner_generation(owner_session_id) + 1
        ending = [
            entry
            for key, entry in tuple(self._live.items())
            if key.owner_session_id == owner_session_id
        ]
        for entry in ending:
            del self._live[entry.key]
            entry.release_outcome = outcome
            self._closing.add(entry)
        retry = [
            entry
            for entry in self._closing
            if entry.key.owner_session_id == owner_session_id and entry not in ending
        ]
        if outcome == "completed":
            # A retry asked to publish (a failed run's release kept the work,
            # and the caller fixed the source) takes that outcome; one never
            # moves the other way, which could dispose of unpublished work.
            for entry in retry:
                task = entry.release_task
                if task is None or task.done():
                    entry.release_outcome = outcome
        failures: list[Exception] = []
        for entry in (*ending, *retry):
            if entry.attachments:
                continue
            if entry.discard_task is not None:
                if not await self._retry_discard(entry):
                    failures.append(
                        RuntimeError(
                            "Discarding the shared allocation of environment "
                            f"{entry.key.environment_name!r} failed; it is kept for a retry."
                        )
                    )
                continue
            try:
                await self._settle(entry)
            except Exception as error:
                failures.append(error)
        if claimant is not None and not (
            outcome == "interrupted" and not self._process_local_store()
        ):
            # Allocations an earlier attempt recorded in another process, which
            # no step of this attempt attached to. An interrupted run can
            # resume, so with a durable store those stay for its resume, as
            # an interrupted release keeps what its binding left in place.
            failures.extend(await self._release_recorded(owner_session_id, claimant, outcome))
        if not self._owner_has_entries(owner_session_id):
            self._owner_generations.pop(owner_session_id, None)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ExceptionGroup("Releasing shared environment allocations failed.", failures)
        if claimant is not None and claimant == self._durable_claimant(owner_session_id):
            self._owner_attempts.pop(owner_session_id, None)
            self._owner_fences.pop(owner_session_id, None)

    def _owner_has_entries(self, owner_session_id: str) -> bool:
        return any(
            key.owner_session_id == owner_session_id
            for key in (*self._live, *self._creating, *(entry.key for entry in self._closing))
        )

    async def close(self, *, outcome: str = "interrupted", deadline: float | None = None) -> bool:
        """Release every shared allocation at shutdown, attached sessions or not.

        ``deadline`` is a loop time. Releases run as owned tasks that are never
        cancelled: one still running at the deadline stays owned (the result
        is False), and a later call waits for that same task. A release that
        fails is discarded instead, so nothing outlives the process. Returns
        whether every allocation finalized cleanly by the deadline.
        """

        loop = asyncio.get_running_loop()

        def remaining() -> float | None:
            return None if deadline is None else max(0.0, deadline - loop.time())

        clean = True
        self._open_owners.clear()
        self._session_scopes.clear()
        self._sessions_by_owner.clear()
        self._copy_back_sessions.clear()
        if self._creating:
            _, pending = await asyncio.wait(tuple(self._creating.values()), timeout=remaining())
            if pending:
                # A creation still running publishes its entry when it ends; a
                # later close releases it.
                clean = False
        keep_recorded = self._records is not None and not self._process_local_store()
        for entry in tuple(self._live.values()):
            if keep_recorded and entry.record is not None and entry.record.reconnectable:
                # Its run hasn't ended and can resume in another process, which
                # reconnects to it through the durable record; leave it there.
                self._forget(entry)
                continue
            entry.release_outcome = entry.release_outcome or outcome
            self._closing.add(entry)
        self._live.clear()
        releasing = tuple(entry for entry in self._closing if entry.discard_task is None)
        tasks = [self._release_task(entry) for entry in releasing]
        if tasks:
            await asyncio.wait(tasks, timeout=remaining())
        for entry, task in zip(releasing, tasks, strict=True):
            if not task.done():
                clean = False
                continue
            if not task.cancelled() and task.exception() is None:
                self._closing.discard(entry)
                continue
            clean = False
            if (
                keep_recorded
                and entry.record is not None
                and entry.record.reconnectable
                and _pending_copy_backs(entry)
            ):
                # A failed copy-back after a step left work only this
                # allocation has; its run can resume elsewhere and retry it.
                self._closing.discard(entry)
                self._forget(entry)
                continue
            _LOGGER.warning(
                "Finalizing shared environment allocation of %r failed; discarding it.",
                entry.key.environment_name,
            )
            entry.discard_task = asyncio.ensure_future(self._discard(entry))
        # Discards are owned like releases: one still running at the deadline
        # is kept, and a later close waits for that same task. One that failed
        # is retried by the next close; the entry stays owned until then.
        discards: list[tuple[_SharedAllocation, asyncio.Task[bool]]] = []
        for entry in tuple(self._closing):
            task = entry.discard_task
            if task is None:
                continue
            if task.done() and not _discarded(task):
                task = asyncio.ensure_future(self._discard(entry))
                entry.discard_task = task
            discards.append((entry, task))
        if discards:
            await asyncio.wait([task for _, task in discards], timeout=remaining())
            for entry, task in discards:
                if task.done() and _discarded(task):
                    self._closing.discard(entry)
                else:
                    clean = False
        if self._background:
            _, pending = await asyncio.wait(tuple(self._background), timeout=remaining())
            if pending:
                clean = False
        return clean

    # Attach and detach ----------------------------------------------------

    async def _acquire(
        self,
        key: SharedAllocationKey,
        factory: EnvironmentFactory,
        request: EnvironmentFactoryRequest,
        session: Session | None,
    ) -> _SharedAllocation:
        if (
            session is not None
            and session.id == key.owner_session_id
            and session.provider_name != _WORKFLOW_JOURNAL_PROVIDER
        ):
            # The owner's own run in this process: its scope's records are
            # claimed as this run, whatever the store says later.
            self._owner_runs[session.id] = (session.instance_id, session.run_epoch)
        while True:
            # The owner may have ended while this session waited for another
            # session's creation; never create or join after that.
            await self._require_open(key, session)
            entry = self._live.get(key)
            if entry is not None:
                if entry.factory is not factory:
                    raise RuntimeError(
                        f"Environment {key.environment_name!r} was registered again with "
                        "another factory while its shared allocation is live."
                    )
                if entry.record is not None and not await self._reclaim(entry):
                    continue
                return entry
            creating = self._creating.get(key)
            if creating is not None:
                try:
                    await asyncio.shield(creating)
                except asyncio.CancelledError:
                    if not creating.done() or not creating.cancelled():
                        raise
                except SharedAllocationScopeClosedError:
                    raise
                except Exception:
                    # The creating session reports its own failure; try again.
                    pass
                continue
            # Read the owner generation before the creation task can be
            # scheduled: an owner that ends from here on bumps it, since the
            # task is registered as creating with no await in between.
            generation = self._owner_generation(key.owner_session_id)
            task = asyncio.ensure_future(self._create(key, factory, request, generation))
            self._creating[key] = task
            await asyncio.shield(task)
            # Like a waiter, check the owner and the live entry again: the
            # owner may have ended (and released the new allocation) before
            # this session resumed.

    async def _unordered_claim_generation(
        self, owner_session_id: str, record_id: str, sequence: int
    ) -> int | None:
        """The record's claim generation before the fence check, for an unordered attempt.

        An attempt with no journal sequence (a custom journal that keeps its
        markers elsewhere) can't be ordered against the record's holder. It
        claims only if nothing claimed the record after this read, which comes
        before its fence check: a holder that claimed earlier is older than an
        attempt that then passes the fence, and one that claims later changes
        the generation, so the claim is refused instead of taking it over.
        """

        if sequence != 0 or self._records is None:
            return None
        snapshot = await self._records.load(owner_session_id)
        current = None if snapshot is None else snapshot.records.get(record_id)
        if current is not None and current.owner_session_id != owner_session_id:
            current = None
        return 0 if current is None else current.claim_generation

    async def _attempt_sequence(self, owner_session_id: str, claimant: _Claimant) -> int:
        """The claimant's order among the owner's attempts; a newer one is larger."""

        if claimant.run_epoch is not None:
            return claimant.run_epoch
        from cayu.workflows.base import WORKFLOW_ATTEMPT_EVENT_TYPE

        markers = await self._session_store.query_events(
            EventQuery(
                session_id=owner_session_id,
                event_type=WORKFLOW_ATTEMPT_EVENT_TYPE,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not markers:
            # A custom journal that keeps its attempt markers elsewhere. Its
            # attempt fence, checked on both sides of every claim, is then
            # what keeps a superseded attempt from holding the record.
            return 0
        if markers[0].event.payload.get("attempt_id") != claimant.attempt_id:
            raise SharedAllocationClaimLost("The workflow attempt has been superseded.")
        return markers[0].sequence

    async def _reclaim(self, entry: _SharedAllocation) -> bool:
        """Claim a live recorded allocation for the attempt now attaching to it.

        A newer attempt of the run in this process uses the allocation the
        older one holds only once it holds the record itself, so another
        process that took the run over in between is noticed: the record then
        names another allocation (or none), and this stale one is dropped, and
        discarded only if this process still owns it. Returns whether the
        entry is still usable.
        """

        if entry.record is None:
            return True
        async with entry.reclaim_lock:
            if self._live.get(entry.key) is not entry:
                # A session of this attempt that reclaimed first dropped the
                # entry; the caller looks the allocation up again.
                return False
            return await self._reclaim_held(entry)

    async def _reclaim_held(self, entry: _SharedAllocation) -> bool:
        record = entry.record
        assert record is not None
        claimant = await self._claimant(entry.key.owner_session_id)
        if claimant is None or claimant.attempt_id in entry.attempts:
            return True
        records = self._records
        assert records is not None
        attempt_id, fence = claimant.attempt_id, claimant.fence
        sequence = await self._attempt_sequence(record.owner_session_id, claimant)
        expected = await self._unordered_claim_generation(
            record.owner_session_id, entry.key.record_id, sequence
        )
        await fence()
        claim = await records.claim(
            record_id=entry.key.record_id,
            scope=record.scope,
            scope_id=record.scope_id,
            owner_session_id=record.owner_session_id,
            environment_name=record.environment_name,
            agent_name=record.agent_name,
            attempt_id=attempt_id,
            attempt_sequence=sequence,
            keep_allocation_session_id=record.allocation_session_id,
            expected_generation=expected,
        )
        if claim is None:
            if self._live.get(entry.key) is entry:
                del self._live[entry.key]
                await self._discard_failed_creation(entry)
            return False
        try:
            await fence()
        except BaseException:
            with contextlib.suppress(Exception):
                await records.restore(entry.key.record_id, claim)
            raise
        previous = claim.previous
        if (
            previous is not None
            and previous.attempt_id != attempt_id
            and previous.attempt_id not in entry.attempts
        ):
            # An attempt in another process held the allocation since this
            # process last did. Its RECONNECT may have taken provider-side
            # ownership away from this process's runner (a Lambda MicroVM's
            # owner claim), and its steps moved the binding state on. So this
            # attempt, which now holds the record, reconnects under the new
            # claim generation instead of reusing the stale handle; the
            # allocation itself is kept.
            if self._live.get(entry.key) is entry:
                del self._live[entry.key]
            self._forget(entry)
            return False
        entry.record = claim.record
        entry.attempts.add(attempt_id)
        return True

    async def _create(
        self,
        key: SharedAllocationKey,
        factory: EnvironmentFactory,
        request: EnvironmentFactoryRequest,
        generation: int,
    ) -> _SharedAllocation:
        try:
            claimant = await self._claimant(key.owner_session_id)
            if claimant is None and _creates_recoverably(factory, request):
                # A recoverable factory may refuse a plain create (a
                # process-external allocation, for example), and its intent
                # needs a record to live in, even if only for this process.
                claimant = await self._agent_claimant(key.owner_session_id, durable_only=False)
            if claimant is None:
                return await self._create_process_local(key, factory, request, generation)
            entry = await self._open_recorded(
                key, factory, request, generation, claimant, create=True
            )
            assert entry is not None
            self._live[key] = entry
            return entry
        finally:
            self._creating.pop(key, None)

    def _shared_request(
        self,
        request: EnvironmentFactoryRequest,
        allocation_session_id: str,
        *,
        owner_session_id: str,
        recorded: bool,
        ownership_epoch: int | None = None,
    ) -> EnvironmentFactoryRequest:
        # ``ownership_epoch`` is a recorded allocation's claim generation, which
        # goes up with every change of owner, so a provider that fences attaches
        # by epoch refuses an owner that lost the allocation.
        # A recorded allocation outlives this process when the session store
        # does, so it carries the owner token only when the store doesn't, as
        # an ordinary session's request does; the reaper then never removes an
        # allocation a durable record still owns. A process-local one always
        # carries it, so the reaper can remove it after a crash.
        owner = (
            current_process_local_owner() if not recorded or self._process_local_store() else None
        )
        return replace(
            request,
            session_id=allocation_session_id,
            parent_session_id=None,
            operation=EnvironmentFactoryOperation.CREATE,
            reconnect_metadata={},
            replacement_predecessor={},
            interaction_id=None,
            process_local_owner=owner,
            ownership_epoch=ownership_epoch,
            # The allocation's identity isn't a session; events about it (a
            # broker's egress audit, say) belong on the scope's owner.
            event_session_id=owner_session_id,
        )

    async def _bind_and_check(
        self,
        key: SharedAllocationKey,
        factory: EnvironmentFactory,
        result: EnvironmentFactoryResult,
        *,
        allocation_session_id: str,
        agent_name: str,
        generation: int,
        recovery_state: dict[str, Any] | None = None,
        workflow_owner: bool = False,
    ) -> BoundWorkspace | None:
        """Bind a new allocation, or rebuild a reconnected one's bound workspace.

        With ``recovery_state`` the binding restores the bound workspace it
        recorded, without copying the source in again. On failure the bound
        workspace is abandoned; the allocation itself is the caller's.
        """

        environment = result.environment
        binding = environment.binding
        bound: BoundWorkspace | None = None
        try:
            require_shareable_result(factory, environment, key.environment_name)
            if binding is not None:
                if recovery_state is not None:
                    bound = await binding._recover_completion_finalization(
                        environment.workspace,
                        environment.runner,
                        session_id=allocation_session_id,
                        agent_name=agent_name,
                        environment_name=key.environment_name,
                        recovery_state=recovery_state,
                    )
                else:
                    bound = await binding.bind(
                        environment.workspace,
                        environment.runner,
                        session_id=allocation_session_id,
                        agent_name=agent_name,
                        environment_name=key.environment_name,
                    )
                    if isinstance(binding, DeferredWorkspaceBinding):
                        # A deferred factory creates its runner on first use.
                        # Every session of the scope uses this one, so it is
                        # created and bound once, now, rather than by whichever
                        # session's tool call happens to come first.
                        await binding.materialization.materialize("use")
            if (
                not workflow_owner
                and key.owner_session_id not in self._open_owners
                and bound is not None
                and binding is not None
                and binding._completion_requires_successful_finalization(bound)
            ):
                # An agent session's success is published before its scope
                # is released, so a failed copy-back couldn't fail it. Only
                # an explicitly opened owner (a workflow run, which releases
                # before journaling completion) may share such a binding.
                raise RuntimeError(
                    f"Environment {key.environment_name!r} copies its work back when "
                    "a session completes, so only a workflow run can own a shared "
                    "allocation of it; an agent session's scope can't. Run the "
                    "sessions as steps of one workflow run, or give them "
                    "allocation_scope='session'."
                )
            if self._owner_generation(key.owner_session_id) != generation:
                raise SharedAllocationScopeClosedError(
                    f"The scope's owner session {key.owner_session_id!r} ended while its "
                    "shared allocation was being created."
                )
        except BaseException:
            if bound is not None and binding is not None:
                with contextlib.suppress(Exception):
                    binding.abandon(bound)
            raise
        return bound

    async def _discard_failed_creation(self, entry: _SharedAllocation) -> None:
        """Own a creation that failed after the factory returned until a discard succeeds."""

        # The entry owns the allocation until a discard positively succeeds: a
        # failed one stays held, and a later end_owner or close retries it.
        entry.release_outcome = "failed"
        entry.discard_task = asyncio.ensure_future(self._discard(entry))
        self._closing.add(entry)
        if await asyncio.shield(entry.discard_task):
            self._closing.discard(entry)

    async def _create_process_local(
        self,
        key: SharedAllocationKey,
        factory: EnvironmentFactory,
        request: EnvironmentFactoryRequest,
        generation: int,
    ) -> _SharedAllocation:
        allocation_session_id = self._allocation_ids.setdefault(
            key, shared_allocation_session_id(key.owner_session_id)
        )
        result = await _create_shared(
            factory,
            self._shared_request(
                request,
                allocation_session_id,
                owner_session_id=key.owner_session_id,
                recorded=False,
            ),
        )
        _require_result(result)
        try:
            bound = await self._bind_and_check(
                key,
                factory,
                result,
                allocation_session_id=allocation_session_id,
                agent_name=request.agent_name,
                generation=generation,
            )
        except BaseException:
            # A retried creation needs a fresh identity, since the failed
            # entry still owns the old one until its discard succeeds.
            self._allocation_ids.pop(key, None)
            await self._discard_failed_creation(
                _SharedAllocation(key=key, factory=factory, result=result, bound=None)
            )
            raise
        entry = _SharedAllocation(key=key, factory=factory, result=result, bound=bound)
        self._allocation_ids.pop(key, None)
        self._live[key] = entry
        return entry

    async def _open_recorded(
        self,
        key: SharedAllocationKey,
        factory: EnvironmentFactory,
        request: EnvironmentFactoryRequest,
        generation: int,
        claimant: _Claimant,
        *,
        create: bool,
    ) -> _SharedAllocation | None:
        """Claim a recorded scope's record, then reconnect to or create its allocation.

        The record is claimed for this attempt before the factory is called,
        and the attempt fence is checked on both sides of the claim, so a
        superseded attempt never keeps a claim. ``create=False`` only
        reconnects (to release what an earlier attempt left): it returns None
        when there is nothing reconnectable.
        """

        records = self._records
        assert records is not None
        attempt_id, fence = claimant.attempt_id, claimant.fence
        record_id = key.record_id
        sequence = await self._attempt_sequence(key.owner_session_id, claimant)
        expected = await self._unordered_claim_generation(key.owner_session_id, record_id, sequence)
        await fence()
        claim = await records.claim(
            record_id=record_id,
            scope=key.scope.value,
            scope_id=key.scope_id,
            owner_session_id=key.owner_session_id,
            environment_name=key.environment_name,
            agent_name=request.agent_name,
            attempt_id=attempt_id,
            attempt_sequence=sequence,
            replace_unreachable=create,
            expected_generation=expected,
        )
        if claim is None:
            return None
        try:
            await fence()
        except BaseException:
            with contextlib.suppress(Exception):
                await records.restore(record_id, claim)
            raise
        record = claim.record
        if claim.orphaned_previous:
            _LOGGER.warning(
                "The shared allocation of environment %r that an earlier attempt of "
                "session %s left can't be reconnected; this attempt starts a new "
                "one, and work that existed only in the old one is lost.",
                key.environment_name,
                key.owner_session_id,
            )
        if record.disposal_state is not None:
            await factory.recover_finalization_disposal(
                replace(
                    request,
                    session_id=record.allocation_session_id,
                    operation=EnvironmentFactoryOperation.RECONNECT,
                    reconnect_metadata=record.reconnect_metadata,
                    ownership_epoch=record.claim_generation,
                ),
                record.disposal_state,
            )
            if not create:
                await records.remove(
                    key.owner_session_id, record.allocation_session_id, {attempt_id}
                )
                return None
            record = await records.replace_with_fresh(record_id, record)
        if record.reconnectable:
            reconnect = replace(
                self._shared_request(
                    request,
                    record.allocation_session_id,
                    owner_session_id=key.owner_session_id,
                    recorded=True,
                    ownership_epoch=record.claim_generation,
                ),
                operation=EnvironmentFactoryOperation.RECONNECT,
                reconnect_metadata=record.reconnect_metadata,
            )
            try:
                result = await factory.create(reconnect)
                _require_result(result)
                # A deferred factory's RECONNECT only rebuilds a placeholder; it
                # finds a missing resource when the binding recovers, here. So a
                # failed adoption gets the same disposal check as a failed
                # reconnect.
                return await self._adopt_reconnected(
                    key,
                    factory,
                    result,
                    record,
                    generation,
                    attempt_id,
                    bind=create,
                    workflow_owner=claimant.workflow,
                )
            except (SharedAllocationClaimLost, SharedAllocationScopeClosedError):
                raise
            except Exception:
                if record.kept_for_resume and not await _positively_disposed(factory, reconnect):
                    # An interrupted release left this for the resume, but it
                    # can't be reconnected (its binding may have disposed of it
                    # in a way the factory can't prove). Continue on a new
                    # allocation; the old one stays owned as an orphan so
                    # recovery or a hand release can dispose of it.
                    _LOGGER.warning(
                        "The shared allocation of environment %r that workflow run %s's "
                        "interrupted release kept can't be reconnected; this attempt "
                        "starts a new one, and work that existed only in the old one is "
                        "lost. The old one is kept as an orphan for cleanup.",
                        key.environment_name,
                        key.owner_session_id,
                    )
                    if not create:
                        await records.orphan(record_id, record)
                        return None
                    record = await records.orphan_and_replace_with_fresh(record_id, record)
                    return await self._create_recorded(
                        key,
                        factory,
                        request,
                        record,
                        generation,
                        attempt_id,
                        workflow_owner=claimant.workflow,
                    )
                if not await _positively_disposed(factory, reconnect):
                    # The allocation may hold the run's only copy of its work;
                    # keep the record for a retry.
                    raise
                _LOGGER.warning(
                    "The shared allocation of environment %r that workflow run %s "
                    "recorded is gone; this attempt starts a new one, and work that "
                    "existed only in the old one is lost.",
                    key.environment_name,
                    key.owner_session_id,
                )
                if not create:
                    await records.remove(
                        key.owner_session_id, record.allocation_session_id, {attempt_id}
                    )
                    return None
                record = await records.replace_with_fresh(record_id, record)
        if not create:
            return None
        previous = claim.previous
        return await self._create_recorded(
            key,
            factory,
            request,
            record,
            generation,
            attempt_id,
            workflow_owner=claimant.workflow,
            # A record an earlier attempt left mid-create without an intent:
            # that creation (perhaps by an older Cayu) went through plain
            # create() under this identity, which adopts what it made.
            interrupted_plain_create=(
                previous is not None
                and previous.allocation_session_id == record.allocation_session_id
                and previous.state == "creating"
                and previous.allocation_intent is None
            ),
        )

    async def _adopt_reconnected(
        self,
        key: SharedAllocationKey,
        factory: EnvironmentFactory,
        result: EnvironmentFactoryResult,
        record: SharedAllocationRecord,
        generation: int,
        attempt_id: str,
        *,
        bind: bool,
        workflow_owner: bool,
    ) -> _SharedAllocation:
        """Attach to a reconnected allocation without disposing of it on failure."""

        records = self._records
        assert records is not None
        binding = result.environment.binding
        bound: BoundWorkspace | None = None
        if record.state == "live" or bind:
            # A live allocation's bound workspace is rebuilt from its recorded
            # state, so nothing is copied in again. One that was created but
            # never bound has no work yet, so binding it now copies the source in.
            bound = await self._bind_and_check(
                key,
                factory,
                result,
                allocation_session_id=record.allocation_session_id,
                agent_name=record.agent_name,
                generation=generation,
                recovery_state=record.binding_state if record.state == "live" else None,
                workflow_owner=workflow_owner,
            )
        try:
            if record.state == "created" and bound is not None and binding is not None:
                record = await records.update(
                    key.record_id,
                    record,
                    state="live",
                    binding_state=_binding_recovery_state(binding, bound),
                    kept_for_resume=False,
                )
            elif record.kept_for_resume:
                # The resume this was kept for has reconnected: from here on a
                # failed reconnect may be all that stands between the run and
                # newer work in it, so it is kept for a retry again.
                record = await records.update(key.record_id, record, kept_for_resume=False)
        except BaseException:
            if bound is not None and binding is not None:
                with contextlib.suppress(Exception):
                    binding.abandon(bound)
            raise
        return _SharedAllocation(
            key=key,
            factory=factory,
            result=result,
            bound=bound,
            record=record,
            attempts={attempt_id},
        )

    async def _create_recorded(
        self,
        key: SharedAllocationKey,
        factory: EnvironmentFactory,
        request: EnvironmentFactoryRequest,
        record: SharedAllocationRecord,
        generation: int,
        attempt_id: str,
        *,
        workflow_owner: bool,
        interrupted_plain_create: bool = False,
    ) -> _SharedAllocation:
        """Create a recorded allocation under its durable identity and record what it returned."""

        records = self._records
        assert records is not None
        record_id = key.record_id
        result, record = await self._create_allocation(
            key, factory, request, record, interrupted_plain_create=interrupted_plain_create
        )
        entry = _SharedAllocation(
            key=key, factory=factory, result=result, bound=None, attempts={attempt_id}
        )
        try:
            record = await records.update(
                record_id,
                record,
                state="created",
                reconnect_metadata=dict(result.reconnect_metadata),
                result_metadata=dict(result.metadata),
            )
            entry.record = record
            bound = await self._bind_and_check(
                key,
                factory,
                result,
                allocation_session_id=record.allocation_session_id,
                agent_name=request.agent_name,
                generation=generation,
                workflow_owner=workflow_owner,
            )
            binding = result.environment.binding
            try:
                record = await records.update(
                    record_id,
                    record,
                    state="live",
                    binding_required=binding is not None and bound is not None,
                    binding_state=(
                        None
                        if bound is None or binding is None or not record.reconnect_metadata
                        else _binding_recovery_state(binding, bound)
                    ),
                )
            except BaseException:
                if bound is not None and binding is not None:
                    with contextlib.suppress(Exception):
                        binding.abandon(bound)
                raise
        except SharedAllocationClaimLost:
            # A newer attempt adopted this identity while it was being created;
            # the allocation is its now, so leave it alone.
            raise
        except BaseException:
            entry.record = record
            # Keep the failed allocation as an orphan until it is discarded, so
            # a retry gets a fresh identity and a crash leaves it to recovery.
            # The discard checks the claim first, so it never disposes of an
            # allocation another attempt has taken over.
            with contextlib.suppress(Exception):
                await records.orphan(record_id, record)
            await self._discard_failed_creation(entry)
            raise
        entry.record = record
        entry.bound = bound
        return entry

    async def _create_allocation(
        self,
        key: SharedAllocationKey,
        factory: EnvironmentFactory,
        request: EnvironmentFactoryRequest,
        record: SharedAllocationRecord,
        *,
        interrupted_plain_create: bool = False,
    ) -> tuple[EnvironmentFactoryResult, SharedAllocationRecord]:
        """Ask the factory for the record's allocation, recoverably when it can.

        ``interrupted_plain_create`` means an earlier attempt left the record
        mid-create with no intent, so its creation went through plain
        ``create()`` under this identity (an older Cayu did that for every
        factory). That create is retried first, so it adopts what the earlier
        one made instead of leaking it under a new intent's name; a factory
        that refuses a plain CREATE never made anything that way.

        A factory that declares an ``allocation_scope`` gets ``create_recoverable``
        with an intent kept in the record. An intent an earlier attempt left
        (it crashed, or its create failed) goes back to the factory, which
        recovers that allocation. When it can't, nothing has been bound to the
        allocation yet, so it is reaped through ``reap_allocation`` and a fresh
        one is created under a new identity. Returns the result and the record
        as it now stands.
        """

        records = self._records
        assert records is not None
        create = self._shared_request(
            request,
            record.allocation_session_id,
            owner_session_id=key.owner_session_id,
            recorded=True,
            ownership_epoch=record.claim_generation,
        )
        scope = factory.allocation_scope(create)
        if scope is not None and type(scope) is not EnvironmentAllocationScope:
            raise TypeError(
                "EnvironmentFactory.allocation_scope must return EnvironmentAllocationScope or None."
            )
        if scope is None:
            if record.allocation_intent is not None:
                raise RuntimeError(
                    f"Environment {key.environment_name!r} has an incomplete recoverable "
                    "shared allocation, but its factory no longer declares an allocation scope."
                )
            result = await factory.create(create)
            _require_result(result)
            return result, record
        if interrupted_plain_create and record.allocation_intent is None:
            try:
                result = await factory.create(create)
            except EnvironmentAllocationUnsupportedError:
                pass
            else:
                _require_result(result)
                return result, record
        context = SharedAllocationIntentContext.open(records, record, scope)
        recovering = context.state is not EnvironmentAllocationState.UNPREPARED
        try:
            result = await factory.create_recoverable(create, context)
        except SharedAllocationClaimLost:
            raise
        except Exception as error:
            if not recovering:
                # The intent stays in the record: a retry recovers it, and
                # recovery reaps it if nothing does.
                raise
            _LOGGER.warning(
                "Recovering the interrupted creation of environment %r's shared allocation "
                "failed (%s); reaping it and creating a new one.",
                key.environment_name,
                type(error).__name__,
            )
            await factory.reap_allocation(create, context)
            if context.state is not EnvironmentAllocationState.REAPED:
                raise RuntimeError(
                    f"Environment {key.environment_name!r} couldn't recover or reap the "
                    "interrupted creation of its shared allocation; it is kept for a retry."
                ) from error
            record = await records.replace_with_fresh(key.record_id, context.record)
            create = self._shared_request(
                request,
                record.allocation_session_id,
                owner_session_id=key.owner_session_id,
                recorded=True,
                ownership_epoch=record.claim_generation,
            )
            context = SharedAllocationIntentContext.open(records, record, scope)
            result = await factory.create_recoverable(create, context)
        _require_result(result)
        if context.state is not EnvironmentAllocationState.ACKNOWLEDGED:
            raise RuntimeError(
                "Recoverable environment factory returned before provider acknowledgement."
            )
        return result, context.record

    async def _release_recorded(
        self, owner_session_id: str, claimant: _Claimant, outcome: str
    ) -> list[Exception]:
        """Finalize the reconnectable allocations an earlier attempt recorded and left.

        A run resumed in another process may end without any step attaching to
        what the crashed process's steps shared; its copy-back still belongs
        to this run's end. Anything that can't be reconnected is left to
        ``recover_incomplete_sessions``.
        """

        records = self._records
        assert records is not None
        snapshot = await records.load(owner_session_id)
        if snapshot is None:
            return []
        held = self._held_allocation_ids()
        creating = {key.record_id for key in self._creating}
        failures: list[Exception] = []
        for record_id, record in snapshot.records.items():
            if (
                record.owner_session_id != owner_session_id
                or record.allocation_session_id in held
                or record_id in creating
                or not record.reconnectable
            ):
                continue
            factory = self._factory_lookup(record.environment_name)
            if factory is None:
                failures.append(
                    RuntimeError(
                        f"Environment {record.environment_name!r} is not registered, so its "
                        "recorded shared allocation can't be released."
                    )
                )
                continue
            key = SharedAllocationKey(
                AllocationSharingScope(record.scope),
                record.scope_id,
                owner_session_id,
                record.environment_name,
            )
            try:
                entry = await self._open_recorded(
                    key,
                    factory,
                    EnvironmentFactoryRequest(
                        session_id=record.allocation_session_id,
                        agent_name=record.agent_name,
                        environment_name=record.environment_name,
                    ),
                    self._owner_generation(owner_session_id),
                    claimant,
                    create=False,
                )
                if entry is None:
                    continue
                entry.release_outcome = outcome
                self._closing.add(entry)
                await self._settle(entry)
            except Exception as error:
                failures.append(error)
        return failures

    async def _acquire_attachable(
        self,
        key: SharedAllocationKey,
        factory: EnvironmentFactory,
        request: EnvironmentFactoryRequest,
        session: Session | None,
    ) -> _SharedAllocation:
        """``_acquire``, then wait out any copy-back after a step before attaching.

        The caller attaches with no await in between. A subagent of a session
        that is attached doesn't wait: the copy-back is waiting for its parent
        to detach.
        """

        while True:
            entry = await self._acquire(key, factory, request, session)
            parent = None if session is None else session.parent_session_id
            # Not a copy-back's own attachment, which names the finished step:
            # a subagent it left behind waits like anything else.
            if entry.copy_back_gate.is_set() or any(
                attachment.session_id == parent
                and not attachment.released
                and not attachment.copy_back
                for attachment in entry.attachments
            ):
                return entry
            await entry.copy_back_gate.wait()

    def _attach(
        self, entry: _SharedAllocation, session_id: str, *, copy_back: bool = False
    ) -> _Attachment:
        attachment = _Attachment(session_id=session_id, copy_back=copy_back)
        entry.attachments.add(attachment)
        return attachment

    def _detach(self, entry: _SharedAllocation, attachment: _Attachment) -> None:
        if attachment.released:
            return
        attachment.released = True
        entry.attachments.discard(attachment)
        entry.detached.set()
        if entry.attachments or entry.release_outcome is None or entry not in self._closing:
            return
        task = asyncio.get_running_loop().create_task(
            self._settle_in_background(entry),
            name="cayu-shared-environment-release",
        )
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _settle_in_background(self, entry: _SharedAllocation) -> None:
        try:
            await self._settle(entry)
        except Exception as error:
            _LOGGER.warning(
                "Releasing the shared allocation of environment %r failed (%s); "
                "it is kept for a retry when its owner ends again or the app closes.",
                entry.key.environment_name,
                type(error).__name__,
            )

    def _release_task(self, entry: _SharedAllocation) -> asyncio.Task[None]:
        """Return the entry's owned release task, starting it (again after a failure)."""

        task = entry.release_task
        if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
            outcome = entry.release_outcome or "interrupted"
            task = asyncio.ensure_future(self._release(entry, outcome))
            entry.release_task = task
        return task

    async def _settle(self, entry: _SharedAllocation) -> None:
        await asyncio.shield(self._release_task(entry))
        self._closing.discard(entry)

    async def _release(self, entry: _SharedAllocation, outcome: str) -> None:
        if entry.record is not None and not await self._still_holds(entry):
            # A newer attempt of the run took the allocation over in another
            # process; finalizing it here would dispose of that attempt's work.
            self._forget(entry)
            return
        binding = entry.environment.binding
        if (
            entry.bound is not None
            and binding is not None
            and _pending_copy_backs(entry)
            and not _binding_copies_back(binding, outcome)
        ):
            # A step's copy-back failed, so the allocation holds work the
            # source hasn't got; this outcome would dispose of it unpublished.
            raise SharedAllocationCopyBackPendingError(
                f"A step's copy-back of environment {entry.key.environment_name!r} failed "
                f"and its work isn't in the source yet, so the shared allocation isn't "
                f"released with outcome {outcome!r}, which wouldn't copy it back; it is "
                "kept. Resolve the conflict and resume the run, or release it with "
                "outcome='completed'."
            )
        if entry.bound is not None and binding is not None:

            async def checkpoint_disposal(state: dict[str, Any]) -> None:
                if entry.record is not None:
                    assert self._records is not None
                    entry.record = await self._records.update(
                        entry.key.record_id, entry.record, disposal_state=state
                    )

            token = finalization_disposal_checkpoint.set(checkpoint_disposal)
            try:
                await binding.finalize(entry.bound, outcome=outcome)
            finally:
                finalization_disposal_checkpoint.reset(token)
            if await self._keep_for_resume(entry, binding, outcome):
                # The binding left the allocation in place for the owner to
                # resume (Lambda MicroVM suspends an interrupted one), so its
                # record stays, holding the binding state this release left.
                return
            if not await self._dispose_what_finalize_kept(entry, outcome):
                # Its disposal isn't confirmed, so the record keeps the
                # allocation owned, as an orphan: recovery and a hand release
                # retry the disposal, and a resumed owner, which never claims
                # an orphan, starts a fresh allocation instead.
                await self._orphan_record(entry)
                return
        elif entry.result.release is not None:
            await entry.result.release(EnvironmentFactoryReleaseAction.DISCARD)
        elif (runner := entry.environment.runner) is not None:
            await runner.close()
        await self._forget_record(entry)

    async def _keep_for_resume(
        self, entry: _SharedAllocation, binding: WorkspaceBinding, outcome: str
    ) -> bool:
        """Keep an interrupted owner's allocation and its record for the owner's resume.

        Only an interruption can leave an allocation for its owner, which may
        resume; any other outcome ends the scope. The record must outlive this
        process and be reachable by RECONNECT, the factory must not prove the
        allocation gone, and the binding must still describe it after finalize
        (a binding that disposed of it, as Docker coding removes its container,
        no longer can). The record then takes the binding state this finalize
        left, so a resumed copy-back compares against what the interrupted one
        published, not against the baselines from before it.
        """

        record = entry.record
        if (
            outcome != "interrupted"
            or record is None
            or self._records is None
            or self._process_local_store()
            or not record.reconnect_metadata
            or record.disposal_state is not None
            or entry.bound is None
        ):
            return False
        binding_state = record.binding_state
        if binding_state is not None:
            binding_state = _binding_recovery_state(binding, entry.bound)
            if binding_state is None:
                return False
        reconnect = EnvironmentFactoryRequest(
            session_id=record.allocation_session_id,
            agent_name=record.agent_name,
            environment_name=record.environment_name,
            operation=EnvironmentFactoryOperation.RECONNECT,
            reconnect_metadata=record.reconnect_metadata,
            ownership_epoch=record.claim_generation,
        )
        if await _positively_disposed(entry.factory, reconnect):
            return False
        try:
            entry.record = await self._records.update(
                entry.key.record_id,
                record,
                binding_state=binding_state,
                kept_for_resume=True,
            )
        except SharedAllocationClaimLost:
            # A newer attempt took the allocation over meanwhile; it is that
            # attempt's to keep or release now.
            return True
        return True

    async def _orphan_record(self, entry: _SharedAllocation) -> None:
        record = entry.record
        if record is None or self._records is None:
            return
        try:
            await self._records.orphan(entry.key.record_id, record)
        except Exception as error:
            _LOGGER.warning(
                "Keeping the record of environment %r's shared allocation, whose disposal "
                "isn't confirmed, as an orphan failed (%s).",
                entry.key.environment_name,
                type(error).__name__,
            )

    async def _dispose_what_finalize_kept(self, entry: _SharedAllocation, outcome: str) -> bool:
        """Dispose of a recorded allocation an interrupted finalize may have kept.

        A binding may keep its allocation on ``interrupted`` so an ordinary
        session can resume it (Lambda MicroVM suspends the MicroVM). A shared
        allocation's record is forgotten once its owner releases it, so nothing
        would own what was kept: unless the factory proves it gone, reconnect
        and discard it, as recovery would. Returns False when that disposal
        failed or can't be confirmed (the binding may have disposed of it
        already, or the provider may be briefly unreachable); the caller then
        keeps the record so the disposal can be retried.
        """

        record = entry.record
        if (
            outcome != "interrupted"
            or record is None
            or not record.reconnect_metadata
            or record.disposal_state is not None
        ):
            return True
        request = EnvironmentFactoryRequest(
            session_id=record.allocation_session_id,
            agent_name=record.agent_name,
            environment_name=record.environment_name,
            operation=EnvironmentFactoryOperation.RECONNECT,
            reconnect_metadata=record.reconnect_metadata,
            ownership_epoch=record.claim_generation,
        )
        if await _positively_disposed(entry.factory, request):
            return True
        try:
            return await _dispose_recorded(entry.factory, record, self._records)
        except Exception as error:
            _LOGGER.warning(
                "After the interrupted release of environment %r's shared allocation, "
                "disposing of anything its binding kept failed (%s); its record is kept "
                "as an orphan so recovery or a hand release can retry.",
                entry.key.environment_name,
                type(error).__name__,
            )
            return False

    async def _still_holds(self, entry: _SharedAllocation) -> bool:
        record = entry.record
        assert record is not None and self._records is not None
        return await self._records.holds(
            record.owner_session_id, record.allocation_session_id, entry.attempts
        )

    async def _forget_record(self, entry: _SharedAllocation) -> None:
        """Drop a disposed allocation's record; a record left behind is reaped later."""

        record = entry.record
        if record is None or self._records is None:
            return
        try:
            await self._records.remove(
                record.owner_session_id, record.allocation_session_id, entry.attempts
            )
        except Exception as error:
            _LOGGER.warning(
                "Forgetting the record of a disposed shared allocation of %r failed (%s); "
                "incomplete-session recovery removes it later.",
                entry.key.environment_name,
                type(error).__name__,
            )

    def _forget(self, entry: _SharedAllocation) -> None:
        """Drop this process's hold on an allocation without touching the allocation."""

        binding = entry.environment.binding
        if entry.bound is not None and binding is not None:
            with contextlib.suppress(Exception):
                binding.abandon(entry.bound)

    def _held_allocation_ids(self) -> set[str]:
        return {
            entry.record.allocation_session_id
            for entry in self._entries()
            if entry.record is not None
        }

    async def _retry_discard(self, entry: _SharedAllocation) -> bool:
        """Wait for ``entry``'s discard, restarting a failed one; True once it is gone."""

        task = entry.discard_task
        if task is None or (task.done() and not _discarded(task)):
            task = asyncio.ensure_future(self._discard(entry))
            entry.discard_task = task
        await asyncio.wait([task])
        if _discarded(task):
            self._closing.discard(entry)
            return True
        return False

    async def _discard(self, entry: _SharedAllocation) -> bool:
        """Discard ``entry``'s allocation; return whether that positively succeeded.

        Success means the binding released what it holds (such as a source
        workspace reservation) and the factory disposed of the resources.
        Anything less keeps the entry owned for a retry, which skips whichever
        step already succeeded.
        """

        name = entry.key.environment_name
        binding = entry.environment.binding
        if entry.record is not None and not entry.resources_discarded:
            try:
                held = await self._still_holds(entry)
            except Exception as error:
                _LOGGER.warning(
                    "Checking the claim on environment %r's shared allocation failed (%s); "
                    "it is kept for a retry.",
                    name,
                    type(error).__name__,
                )
                return False
            if not held:
                # Another attempt took it over: it isn't this process's to discard.
                self._forget(entry)
                return True
        if not entry.binding_released and entry.bound is not None and binding is not None:
            try:
                if not binding.abandon(entry.bound):
                    # The binding keeps its lifecycle owner for its own cleanup
                    # retry (Docker coding after a finalize whose disposal
                    # failed). Run that first: it disposes through the
                    # binding's retained state, which a factory discard would
                    # move past (releasing immutable inputs the retry then
                    # can't mark closing).
                    await binding.finalize(
                        entry.bound, outcome=entry.release_outcome or "interrupted"
                    )
            except Exception as error:
                _LOGGER.warning(
                    "Releasing the binding state of environment %r's shared allocation "
                    "failed (%s); it is kept for a retry.",
                    name,
                    type(error).__name__,
                )
                return False
            entry.binding_released = True
        if not entry.resources_discarded:
            try:
                await _discard_resources(entry.result)
            except Exception as error:
                _LOGGER.warning(
                    "Discarding the shared allocation of environment %r failed (%s).",
                    name,
                    type(error).__name__,
                )
                return False
            entry.resources_discarded = True
        await self._forget_record(entry)
        return True

    # Recovery -------------------------------------------------------------

    async def reap_recorded(self, owner_session_id: str) -> int:
        """Dispose of recorded allocations of ``owner_session_id`` that no run can use.

        That is every orphan (left by an attempt that couldn't reconnect, or by
        a failed creation), and every record once the owner has ended: a
        workflow run whose newest attempt journaled its completion, or an agent
        session that completed or failed (a close or an abandonment fails it).
        A record whose owner can still resume is left alone, and so is anything
        this process holds. Returns how many were disposed of.
        ``recover_incomplete_sessions`` calls it for each session it inspects.
        """

        records = self._records
        if records is None:
            return 0
        snapshot = await records.load(owner_session_id)
        if snapshot is None or not (snapshot.records or snapshot.orphans):
            return 0
        # Read after the snapshot: a run that claimed a record before it was
        # read is already running, and one that claims after changes the claim,
        # so ``mark_reaping`` refuses the record.
        owner = await self._session_store.load(owner_session_id)
        if owner is None:
            return 0

        def owned(record: SharedAllocationRecord) -> bool:
            # A fork starts from a copy of its source's checkpoint, records
            # included; those still belong to the source.
            return record.owner_session_id == owner_session_id

        candidates = [record for record in snapshot.orphans.values() if owned(record)]
        if await self._owner_ended(owner):
            candidates.extend(record for record in snapshot.records.values() if owned(record))
        held = self._held_allocation_ids()
        reaped = 0
        for record in candidates:
            if record.allocation_session_id in held:
                continue
            factory = self._factory_lookup(record.environment_name)
            if factory is None:
                continue
            # The same disposal lease a hand release takes: while recovery holds
            # it, no release (and no other reaper) touches the allocation.
            fenced = await records.mark_reaping(record)
            if fenced is None:
                continue
            try:
                async with keeping_disposal_lease(records, fenced) as hold:
                    try:
                        disposed = await _dispose_recorded(factory, hold.record, records)
                    except Exception as error:
                        _LOGGER.warning(
                            "Disposing of a recorded shared allocation of %r failed (%s); "
                            "the next recovery retries it.",
                            record.environment_name,
                            type(error).__name__,
                        )
                        disposed = False
                    except BaseException:
                        # Cancelled mid-disposal: give the lease up so the next recovery
                        # or a release can try again at once.
                        with contextlib.suppress(Exception):
                            await records.drop_release(hold.record)
                        raise
                    if not disposed:
                        with contextlib.suppress(Exception):
                            await records.drop_release(hold.record)
                        continue
                    await hold.remove_record(records)
            except SharedAllocationClaimLost:
                _LOGGER.warning(
                    "Lost the disposal lease for shared allocation %s; leaving it for retry.",
                    record.allocation_session_id,
                )
                continue
            reaped += 1
        return reaped

    async def _owner_ended(self, owner: Session) -> bool:
        """Whether no run of the owner will use its recorded allocations again."""

        if owner.provider_name != _WORKFLOW_JOURNAL_PROVIDER:
            # An agent session's run ends its scope at its terminal event; a
            # record is left only by a crash. An interrupted or pending session
            # can still continue and reconnect, so only an ended one is reaped.
            return owner.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}
        workflow_name = owner.metadata.get("cayu.workflow")
        if type(workflow_name) is not str:
            return False
        # Imported here: the workflow package imports the runtime.
        from cayu.workflows.journal import (
            EventStoreJournal,
            WorkflowJournalContext,
            WorkflowJournalRunEnd,
        )

        if await EventStoreJournal(self._session_store, owner.id, workflow_name).run_ended():
            return True
        # A custom journal keeps its attempts elsewhere; ask the journals that
        # this process's workflows of that name use, when they can tell.
        for factory in tuple(self._workflow_journals.get(workflow_name, ())):
            try:
                journal = factory(
                    WorkflowJournalContext(
                        session_store=self._session_store,
                        session_id=owner.id,
                        workflow_name=workflow_name,
                        emit_events=_refuse_journal_writes,
                    )
                )
                if isinstance(journal, WorkflowJournalRunEnd) and await journal.run_ended() is True:
                    return True
            except Exception as error:
                _LOGGER.warning(
                    "Asking the journal of workflow run %s whether it ended failed (%s); "
                    "its shared allocations are kept.",
                    owner.id,
                    type(error).__name__,
                )
        return False

    def register_workflow_journal(
        self, workflow_name: str, journal_factory: Callable[[Any], Any]
    ) -> None:
        """Let recovery ask a custom journal of ``workflow_name`` whether a run ended.

        ``WorkflowBase`` calls this when it is given a ``journal_factory``. The
        newest few distinct factories per workflow name are kept.
        """

        factories = self._workflow_journals.setdefault(workflow_name, [])
        if any(existing is journal_factory for existing in factories):
            return
        factories.append(journal_factory)
        del factories[:-_MAX_JOURNALS_PER_WORKFLOW]

    # Copy-back after a step ----------------------------------------------

    async def copy_back_session(
        self,
        session_id: str,
        *,
        owner_session_id: str,
        environment_name: str | None,
    ) -> int:
        """Copy the work in the allocations ``session_id`` shared back to their source.

        The allocations stay live for the run's next steps; the owner's release
        later copies back only what changed after this. Each copy-back goes
        through the factory binding's own copy-back. It waits until no other
        session or command is attached and keeps new ones out until it is
        done, so nothing writes in the sandbox while it is copied: it copies
        what parallel steps finished, never half a write. A recorded
        allocation's binding recovery state is then rewritten, so a process
        that reconnects after a crash continues from this copy-back instead of
        the original copy-in.

        A session that attached in this process copies back what it attached
        to. One that didn't (its step is being replayed after a crash, or after
        its run ended on a failed copy-back) copies back the allocations
        ``owner_session_id`` holds for ``environment_name``: a live one, one
        kept after a failed release, or a recorded one, which it reconnects
        first. A conflict raises, as at the owner's release: the allocation
        stays live with the work in it, marked as holding unpublished work, so
        no release with an outcome that wouldn't copy back disposes of it.
        Returns how many allocations were copied back.
        """

        keys = self._copy_back_sessions.get(session_id) or await self._replayed_copy_back_keys(
            session_id, owner_session_id, environment_name
        )
        copied = 0
        for key in sorted(keys, key=lambda item: item.record_id):
            entry = await self._copy_back_entry(key, session_id)
            if entry is None:
                continue
            attachment = self._attach(entry, session_id, copy_back=True)
            try:
                async with entry.copy_back_lock:
                    entry.copy_back_gate.clear()
                    try:
                        await _until_quiet(entry, session_id)
                        if self._live.get(key) is not entry:
                            # The owner ended meanwhile; its release copies back.
                            continue
                        if await self._copy_back(entry, session_id):
                            copied += 1
                    finally:
                        entry.copy_back_gate.set()
            finally:
                self._detach(entry, attachment)
        return copied

    async def _replayed_copy_back_keys(
        self, session_id: str, owner_session_id: str, environment_name: str | None
    ) -> set[SharedAllocationKey]:
        scope = self._session_scopes.get(session_id)
        if environment_name is None or (scope or self.default_scope(environment_name)) is (
            AllocationSharingScope.SESSION
        ):
            return set()
        keys = {
            entry.key
            for entry in self._entries()
            if entry.key.owner_session_id == owner_session_id
            and entry.key.environment_name == environment_name
        }
        if self._records is None:
            return keys
        snapshot = await self._records.load(owner_session_id)
        held = self._held_allocation_ids()
        for record in () if snapshot is None else snapshot.records.values():
            if (
                record.environment_name == environment_name
                and record.state == "live"
                and record.reconnectable
                and record.allocation_session_id not in held
            ):
                keys.add(
                    SharedAllocationKey(
                        AllocationSharingScope(record.scope),
                        record.scope_id,
                        owner_session_id,
                        environment_name,
                    )
                )
        return keys

    async def _copy_back_entry(
        self, key: SharedAllocationKey, session_id: str
    ) -> _SharedAllocation | None:
        """The live entry for ``key``, revived or reconnected if need be; None if none."""

        if key not in self._live:
            retained = next((entry for entry in self._closing if entry.key == key), None)
            if retained is not None and not self._revive(retained):
                # Its release or discard is running now.
                return None
        entry = self._live.get(key)
        factory = self._factory_lookup(key.environment_name) if entry is None else entry.factory
        if factory is None:
            return None
        agent_name = (
            session_id if entry is None or entry.record is None else entry.record.agent_name
        )
        if entry is None and key not in self._creating:
            # A record an earlier process left: reconnect to it through the
            # path a step attaching to it takes, so the run's next steps reuse it.
            records = self._records
            if records is None or self._durable_claimant(key.owner_session_id) is None:
                return None
            snapshot = await records.load(key.owner_session_id)
            record = None if snapshot is None else snapshot.records.get(key.record_id)
            if (
                record is None
                or not record.reconnectable
                or record.state != "live"
                or record.allocation_session_id in self._held_allocation_ids()
            ):
                return None
            agent_name = record.agent_name
        # A live entry, one a sibling step is creating, or a record to
        # reconnect, all through the path a step attaching takes, which also
        # reclaims a record an earlier attempt of this run holds.
        return await self._acquire(
            key,
            factory,
            EnvironmentFactoryRequest(
                session_id=session_id,
                agent_name=agent_name,
                environment_name=key.environment_name,
            ),
            None,
        )

    def _revive(self, entry: _SharedAllocation) -> bool:
        """Make an allocation kept after a failed release live again for its reopened owner.

        A release that failed (a copy-back conflict, or one refused because a
        step's copy-back is pending) keeps the allocation; a resumed run's
        step then needs it back to retry its copy-back. One whose release or
        discard is still running is left to it.
        """

        task = entry.release_task
        if (
            entry.key.owner_session_id not in self._open_owners
            or entry.key in self._live
            or entry.discard_task is not None
            or (task is not None and not task.done())
        ):
            return False
        self._closing.discard(entry)
        entry.release_outcome = None
        entry.release_task = None
        self._live[entry.key] = entry
        return True

    async def _copy_back(self, entry: _SharedAllocation, session_id: str) -> bool:
        binding = entry.environment.binding
        bound = entry.bound
        record = entry.record
        if binding is None or bound is None:
            return False
        if record is not None and session_id in record.copied_back:
            # Done before a crash that came ahead of the step's journaling.
            return False
        _require_copy_back_while_bound(entry)
        if record is not None and not await self._still_holds(entry):
            raise SharedAllocationClaimLost(
                "Another workflow attempt took over the shared allocation of environment "
                f"{entry.key.environment_name!r}; this one can't copy it back."
            )
        try:
            await binding._copy_back_while_bound(bound, outcome="completed")
        except BaseException:
            await self._mark_copy_back_pending(entry, session_id)
            raise
        # Everything in the allocation is published now, including the work of
        # steps whose copy-back failed earlier.
        published = (*sorted(_pending_copy_backs(entry) - {session_id}), session_id)
        entry.pending_copy_back.clear()
        record = entry.record
        if record is not None:
            assert self._records is not None
            # The bound workspace's baseline moved: a process that reconnects
            # must rebuild it from here, or its next copy-back would see this
            # one's writes to the source as conflicts. Recorded together with
            # the steps it published, so replaying one doesn't copy back again.
            copied_back = (
                *(item for item in record.copied_back if item not in published),
                *published,
            )
            entry.record = await self._records.update(
                entry.key.record_id,
                record,
                binding_state=(
                    None
                    if record.binding_state is None
                    else _binding_recovery_state(binding, bound)
                ),
                copied_back=copied_back[-MAX_COPIED_BACK_SESSIONS:],
                pending_copy_back=(),
            )
        return True

    async def _mark_copy_back_pending(self, entry: _SharedAllocation, session_id: str) -> None:
        """Remember that ``entry`` holds work a failed copy-back didn't publish."""

        entry.pending_copy_back.add(session_id)
        record = entry.record
        if record is None or self._records is None or session_id in record.pending_copy_back:
            return
        try:
            entry.record = await self._records.update(
                entry.key.record_id,
                record,
                pending_copy_back=(*record.pending_copy_back, session_id)[
                    -MAX_COPIED_BACK_SESSIONS:
                ],
            )
        except Exception as error:
            # This process still knows, so its own releases keep the work.
            _LOGGER.warning(
                "Recording the failed copy-back of environment %r failed (%s).",
                entry.key.environment_name,
                type(error).__name__,
            )

    # Operator inspection and release ---------------------------------------
    # ``cayu.runtime._shared_allocation_operations`` lists and releases
    # recorded allocations through these, on the same fences as recovery.

    @property
    def durable_records(self) -> SharedAllocationRecords | None:
        return self._records

    def registered_factory(self, environment_name: str) -> EnvironmentFactory | None:
        return self._factory_lookup(environment_name)

    def owner_active_here(self, owner_session_id: str) -> bool:
        """Whether this process runs ``owner_session_id`` or holds any of its allocations."""

        return owner_session_id in self._open_owners or self._owner_has_entries(owner_session_id)

    def holds_allocation(self, allocation_session_id: str) -> bool:
        return allocation_session_id in self._held_allocation_ids()

    async def owner_ended(self, owner: Session) -> bool:
        return await self._owner_ended(owner)

    # Workflow commands ----------------------------------------------------

    @contextlib.asynccontextmanager
    async def attach_command(
        self,
        factory: EnvironmentFactory,
        *,
        scope: AllocationSharingScope,
        parent: Session,
        request: EnvironmentFactoryRequest,
    ) -> AsyncIterator[Runner]:
        """Lend a workflow ``command`` step the runner of its scope's shared allocation.

        ``parent`` is the session the command runs under, the workflow run's
        anchor, so the key is the one that run's step sessions resolve. When no
        step has created the allocation yet, the command creates it through the
        same path, owner and refusals a step would, with ``request``. The
        command only attaches while it runs and detaches after; releasing the
        allocation stays with the scope's owner, which must be open already.
        """

        environment_name = request.environment_name
        if scope is AllocationSharingScope.PARENT:
            key = self._attached_parent_key(parent.id, environment_name) or SharedAllocationKey(
                scope, parent.id, parent.id, environment_name
            )
        elif scope is AllocationSharingScope.CAUSAL_BUDGET:
            owner = await self._top_ancestor(parent)
            key = SharedAllocationKey(
                scope, request.causal_budget_id or owner, owner, environment_name
            )
        else:
            raise ValueError("A session-scoped allocation is not shared.")
        require_shareable_factory(factory)
        self._in_use = True
        entry = await self._acquire_attachable(key, factory, request, None)
        attachment = self._attach(entry, request.session_id)
        try:
            if self._live.get(key) is not entry:
                # The owner ended between the creation and this attachment.
                raise SharedAllocationScopeClosedError(
                    f"The scope's owner session {key.owner_session_id!r} ended before the "
                    "command could attach to its shared allocation."
                )
            runner = entry.runner
            if runner is None:
                raise RuntimeError(
                    f"The shared allocation of environment {environment_name!r} has no runner."
                )
            yield runner
        finally:
            self._detach(entry, attachment)

    # Inspection -----------------------------------------------------------

    def _entries(self) -> tuple[_SharedAllocation, ...]:
        return (*self._live.values(), *self._closing)

    def held_runners(self) -> list[Runner]:
        return [runner for entry in self._entries() if (runner := entry.runner) is not None]

    def held_runners_source(self) -> Callable[[], list[Runner]]:
        """Return a callable listing held runners that keeps no reference to this registry.

        The process-exit hook outlives the application, so it must not keep the
        registry (and through it the application) alive.
        """

        live = self._live
        closing = self._closing

        def held() -> list[Runner]:
            return [
                runner
                for entry in (*live.values(), *closing)
                if (runner := entry.runner) is not None
            ]

        return held

    def binds_source_workspace(self, workspace_id: str) -> bool:
        return any(
            entry.bound is not None
            and entry.bound.source_workspace is not None
            and entry.bound.source_workspace.id == workspace_id
            for entry in self._entries()
        )

    def holds_environment(self, environment_name: str) -> bool:
        return any(
            key.environment_name == environment_name
            for key in (*self._creating, *(entry.key for entry in self._entries()))
        )

    def live_keys(self) -> tuple[SharedAllocationKey, ...]:
        """Keys of the allocations that are live or still being released."""

        return tuple(entry.key for entry in self._entries())


def _creates_recoverably(factory: EnvironmentFactory, request: EnvironmentFactoryRequest) -> bool:
    """Whether ``factory`` declares an allocation scope for a CREATE like ``request``."""

    return (
        factory.allocation_scope(
            replace(
                request,
                operation=EnvironmentFactoryOperation.CREATE,
                reconnect_metadata={},
                replacement_predecessor={},
            )
        )
        is not None
    )


async def _create_shared(
    factory: EnvironmentFactory, request: EnvironmentFactoryRequest
) -> EnvironmentFactoryResult:
    try:
        return await factory.create(request)
    except EnvironmentAllocationUnsupportedError as error:
        raise RuntimeError(
            f"Environment {request.environment_name!r} can't share its allocation: its "
            "factory creates a process-external resource (a Lambda MicroVM, say) only "
            "through create_recoverable, under a session's durable allocation intent, "
            "and a shared allocation is created through create() under the scope's "
            "own identity. Give it allocation_scope='session' so each session gets "
            "its own."
        ) from error


async def _refuse_journal_writes(events: list[Any]) -> list[Any]:
    """The event emitter of a journal recovery only reads from."""

    del events
    raise RuntimeError("Recovery reads a workflow journal; it never writes to it.")


def _pending_copy_backs(entry: _SharedAllocation) -> set[str]:
    """Step sessions whose copy-back failed and whose work only ``entry`` holds."""

    recorded = () if entry.record is None else entry.record.pending_copy_back
    return {*entry.pending_copy_back, *recorded}


async def _until_quiet(entry: _SharedAllocation, session_id: str) -> None:
    """Wait until nothing but ``session_id`` and copy-backs is attached to ``entry``."""

    while any(
        not attachment.copy_back and attachment.session_id != session_id
        for attachment in entry.attachments
    ):
        entry.detached.clear()
        await entry.detached.wait()


def _require_copy_back_while_bound(entry: _SharedAllocation) -> None:
    binding = entry.environment.binding
    if binding is None or entry.bound is None or binding._copies_back_while_bound():
        return
    raise RuntimeError(
        f"Environment {entry.key.environment_name!r} uses {type(binding).__name__}, which "
        "can't copy its work back while its allocation stays live, so a step can't use "
        "copy_back='after_step' with it. Use the default, copy_back='on_release'."
    )


def _discarded(task: asyncio.Task[bool]) -> bool:
    """Whether a finished discard task positively disposed of its allocation."""

    return not task.cancelled() and task.exception() is None and task.result() is True


def _require_result(result: object) -> None:
    if type(result) is not EnvironmentFactoryResult:
        raise TypeError("Environment factory create must return EnvironmentFactoryResult.")


def _binding_recovery_state(
    binding: WorkspaceBinding, bound: BoundWorkspace
) -> dict[str, Any] | None:
    """The binding's durable state for rebuilding ``bound`` elsewhere, or None if it has none.

    This is the state completion recovery already uses to retry a copy-back
    without copying in again. A binding that can't provide it can't be
    reattached from another process.
    """

    try:
        return binding._completion_finalization_recovery_state(bound)
    except Exception:
        return None


async def _positively_disposed(
    factory: EnvironmentFactory, request: EnvironmentFactoryRequest
) -> bool:
    """Whether the factory proves the recorded allocation gone; a failed check is not proof."""

    try:
        return await factory.is_allocation_disposed(request) is True
    except Exception:
        return False


async def _discard_resources(result: EnvironmentFactoryResult) -> bool:
    if result.release is not None:
        await asyncio.wait_for(
            result.release(EnvironmentFactoryReleaseAction.DISCARD),
            timeout=result.release_timeout_s,
        )
        return True
    if result.environment.runner is not None:
        await result.environment.runner.close()
        return True
    return False


async def _dispose_recorded(
    factory: EnvironmentFactory,
    record: SharedAllocationRecord,
    records: SharedAllocationRecords | None = None,
) -> bool:
    """Dispose of a recorded allocation that no run will use again, without copying back.

    One whose creation never finished but left an allocation intent is reaped
    through the factory's ``reap_allocation``, as an ordinary session's
    unpublished allocation is. A reconnectable one goes through the factory's
    RECONNECT path and is discarded. Anything else can only be disposed of from
    the metadata of the result that created it, through
    ``discard_unreachable_allocation``, or through ``reap_allocation`` when its
    creation left an intent; a factory that can do neither leaves it to lapse
    (a brokered sandbox's lease does).
    """

    request = EnvironmentFactoryRequest(
        session_id=record.allocation_session_id,
        agent_name=record.agent_name,
        environment_name=record.environment_name,
        ownership_epoch=record.claim_generation,
        # Events about the allocation (a broker's grant revocations) belong on
        # the run that owned it; its own identity isn't a session.
        event_session_id=record.owner_session_id,
    )

    if (
        records is not None
        and record.allocation_intent is not None
        and not record.reconnect_metadata
        and record.disposal_state is None
    ):
        context = SharedAllocationIntentContext.open(records, record)
        await factory.reap_allocation(request, context)
        return context.state is EnvironmentAllocationState.REAPED
    if record.reconnect_metadata:
        reconnect = replace(
            request,
            operation=EnvironmentFactoryOperation.RECONNECT,
            reconnect_metadata=record.reconnect_metadata,
        )
        if record.disposal_state is not None:
            await factory.recover_finalization_disposal(reconnect, record.disposal_state)
            return True
        if await factory.is_allocation_disposed(reconnect):
            return True
        result = await factory.create(reconnect)
        _require_result(result)
        return await _discard_resources(result)
    if record.result_metadata and await factory.discard_unreachable_allocation(
        request, record.result_metadata
    ):
        return True
    _LOGGER.info(
        "Environment %r can't dispose of an unreachable shared allocation; it is left to lapse.",
        record.environment_name,
    )
    return False


class _SharedAllocationFactory(EnvironmentFactory):
    """Attach one session to its scope's shared allocation, creating it first if needed."""

    def __init__(
        self,
        registry: SharedEnvironmentAllocations,
        inner: EnvironmentFactory,
        *,
        session: Session,
        environment_name: str,
        scope: AllocationSharingScope,
    ) -> None:
        self._registry = registry
        self._inner = inner
        self._session = session
        self._environment_name = environment_name
        self._scope = scope

    @property
    def source_workspace(self):  # type: ignore[override]
        return self._inner.source_workspace

    @property
    def secret_resolution_scope(self) -> Literal["static", "dynamic"]:
        return self._inner.secret_resolution_scope

    @property
    def execution_profile_identity(self):
        return self._inner.execution_profile_identity

    @property
    def egress_authority_identity(self):
        return self._inner.egress_authority_identity

    @property
    def egress_authority_follows_allocation(self) -> bool:
        return self._inner.egress_authority_follows_allocation

    @property
    def configured_artifact_store(self):
        return self._inner.configured_artifact_store

    def execution_admission_candidate(
        self, request: EnvironmentFactoryRequest
    ) -> ExecutionAdmissionCandidate | None:
        return self._inner.execution_admission_candidate(request)

    def construction_admission_candidate(self) -> ExecutionAdmissionCandidate | None:
        return self._inner.construction_admission_candidate()

    def execution_environment_authority(self) -> ExecutionEnvironmentAuthority | None:
        return self._inner.execution_environment_authority()

    def workload_authority(self, name: str) -> RunnerWorkloadAuthority | None:
        return self._inner.workload_authority(name)

    def allocation_scope(
        self, request: EnvironmentFactoryRequest
    ) -> EnvironmentAllocationScope | None:
        # The shared allocation has no per-session durable intent.
        del request
        return None

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        if request.reconnect_metadata:
            raise RuntimeError(
                f"Session {request.session_id!r} already has its own allocation of "
                f"environment {request.environment_name!r}; it can't move to a shared "
                f"{self._scope.value} allocation."
            )
        require_shareable_factory(self._inner)
        registry = self._registry
        key = await registry.resolve_key(self._session, self._environment_name, self._scope)
        entry = await registry._acquire_attachable(key, self._inner, request, self._session)
        attachment = registry._attach(entry, request.session_id)
        try:
            copy_back = registry._copy_back_sessions.get(request.session_id)
            if copy_back is not None:
                _require_copy_back_while_bound(entry)
                copy_back.add(key)
            # This session's own stores and MCP servers, from its own request;
            # never the ones of the session that created the allocation.
            parts = await session_parts(self._inner, request, entry.result)
            attached = session_environment(
                entry.environment,
                parts,
                bound=entry.bound,
                runner=entry.runner,
                binding=_SharedAllocationBinding(registry, entry, attachment),
            )

            async def release(action: EnvironmentFactoryReleaseAction) -> None:
                # Detach only: the scope's owner releases the allocation.
                del action
                registry._detach(entry, attachment)

            return EnvironmentFactoryResult(
                environment=attached,
                metadata={
                    **entry.result.metadata,
                    "shared_allocation": key.payload(),
                },
                release=release,
            )
        except BaseException:
            registry._detach(entry, attachment)
            raise


class _SharedAllocationBinding(WorkspaceBinding):
    """Expose a shared allocation's bound workspace; finalizing only detaches."""

    def __init__(
        self,
        registry: SharedEnvironmentAllocations,
        entry: _SharedAllocation,
        attachment: _Attachment,
    ) -> None:
        self._registry = registry
        self._entry = entry
        self._attachment = attachment

    async def bind(
        self,
        workspace,
        runner,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoundWorkspace:
        copied = _validate_bind_request(
            workspace,
            runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata,
        )
        if self._attachment.released:
            raise RuntimeError("This shared allocation attachment was already released.")
        shared = self._entry.bound
        return BoundWorkspace(
            workspace=workspace,
            source_workspace=workspace,
            runner=runner,
            path=None if shared is None else shared.path,
            metadata={**copied, "shared_allocation": self._entry.key.payload()},
        )

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        _validate_finalize_request(bound, outcome=outcome, metadata=metadata)
        self._registry._detach(self._entry, self._attachment)
        return None

    async def observe_revision(self, bound: BoundWorkspace) -> WorkspaceRevisionObservation:
        inner = self._entry.environment.binding
        shared = self._entry.bound
        if inner is not None and shared is not None and not self._attachment.released:
            return await inner.observe_revision(shared)
        return await super().observe_revision(bound)

    def input_capability(self) -> ImmutableInputAdapterCapability:
        inner = self._entry.environment.binding
        if inner is not None:
            return inner.input_capability()
        return super().input_capability()

    def abandon(self, bound: BoundWorkspace) -> bool:
        super().abandon(bound)
        self._registry._detach(self._entry, self._attachment)
        return True
