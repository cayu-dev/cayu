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

An agent session that owns a scope stays process-local: its request carries
this process's ``process_local_owner`` token, so a factory that labels its
resources with it (Docker coding does) leaves them to
``reap_orphaned_docker_coding_containers`` after a crash, and a leased provider
(a brokered sandbox) lets the lease lapse. A workflow run's request carries the
token only when the session store doesn't outlive the process, as an ordinary
session's does, so the reaper never removes an allocation a durable record
still owns.
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
from cayu.environments.base import Environment, copy_environment_spec
from cayu.environments.bindings import (
    BoundWorkspace,
    WorkspaceBinding,
    WorkspaceSnapshot,
    _validate_bind_request,
    _validate_finalize_request,
)
from cayu.environments.deferred import DeferredWorkspaceBinding
from cayu.environments.factory import (
    EnvironmentAllocationScope,
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
from cayu.events import EventType
from cayu.runtime._shared_allocation_records import (
    SharedAllocationClaimLost,
    SharedAllocationRecord,
    SharedAllocationRecords,
    shared_allocation_record_id,
    shared_allocation_session_id,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.forks import session_fork_profile_relationship
from cayu.sessions.records import Session

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
_WORKFLOW_JOURNAL_PROVIDER = "cayu.workflow"


class SharedAllocationScopeClosedError(RuntimeError):
    """Raised when a session asks to share an allocation whose owner isn't running here."""


__all__ = [
    "SharedAllocationKey",
    "SharedAllocationScopeClosedError",
    "SharedEnvironmentAllocations",
    "shared_allocation_session_id",
]

# A workflow owner's attempt id, and the check that it is still the newest attempt.
_Claimant = tuple[str, Callable[[], Awaitable[None]]]


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

    def bind_session_presence(self, session_running_here: Callable[[str], bool]) -> None:
        """Tell the registry how to see whether a session's run is live in this process."""

        self._session_running_here = session_running_here

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
        return attempt_id, fence

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
        self._open_owners.discard(owner_session_id)
        for session_id in self._sessions_by_owner.pop(owner_session_id, ()):
            self._session_scopes.pop(session_id, None)
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
        if claimant is not None:
            # Allocations an earlier attempt recorded in another process, which
            # no step of this attempt attached to.
            failures.extend(await self._release_recorded(owner_session_id, claimant, outcome))
        if not self._owner_has_entries(owner_session_id):
            self._owner_generations.pop(owner_session_id, None)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ExceptionGroup("Releasing shared environment allocations failed.", failures)
        if self._durable_claimant(owner_session_id) == claimant:
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

    async def _attempt_sequence(self, owner_session_id: str, attempt_id: str) -> int:
        from cayu.workflows.base import WORKFLOW_ATTEMPT_EVENT_TYPE

        markers = await self._session_store.query_events(
            EventQuery(
                session_id=owner_session_id,
                event_type=WORKFLOW_ATTEMPT_EVENT_TYPE,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not markers or markers[0].event.payload.get("attempt_id") != attempt_id:
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

        claimant = self._durable_claimant(entry.key.owner_session_id)
        record = entry.record
        if claimant is None or record is None or claimant[0] in entry.attempts:
            return True
        records = self._records
        assert records is not None
        attempt_id, fence = claimant
        await fence()
        claim = await records.claim(
            record_id=entry.key.record_id,
            scope=record.scope,
            scope_id=record.scope_id,
            owner_session_id=record.owner_session_id,
            environment_name=record.environment_name,
            agent_name=record.agent_name,
            attempt_id=attempt_id,
            attempt_sequence=await self._attempt_sequence(record.owner_session_id, attempt_id),
            keep_allocation_session_id=record.allocation_session_id,
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
            claimant = self._durable_claimant(key.owner_session_id)
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
        recorded: bool,
    ) -> EnvironmentFactoryRequest:
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
            if isinstance(binding, DeferredWorkspaceBinding):
                raise RuntimeError(
                    "A deferred-materialization environment can't share its allocation."
                )
            _require_shareable(factory, environment, key.environment_name)
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
        result = await factory.create(
            self._shared_request(request, allocation_session_id, recorded=False)
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
        """Claim a workflow-owned scope's record, then reconnect to or create its allocation.

        The record is claimed for this attempt before the factory is called,
        and the attempt fence is checked on both sides of the claim, so a
        superseded attempt never keeps a claim. ``create=False`` only
        reconnects (to release what an earlier attempt left): it returns None
        when there is nothing reconnectable.
        """

        records = self._records
        assert records is not None
        attempt_id, fence = claimant
        record_id = key.record_id
        await fence()
        claim = await records.claim(
            record_id=record_id,
            scope=key.scope.value,
            scope_id=key.scope_id,
            owner_session_id=key.owner_session_id,
            environment_name=key.environment_name,
            agent_name=request.agent_name,
            attempt_id=attempt_id,
            attempt_sequence=await self._attempt_sequence(key.owner_session_id, attempt_id),
            replace_unreachable=create,
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
                "workflow run %s left can't be reconnected; this attempt starts a new "
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
                self._shared_request(request, record.allocation_session_id, recorded=True),
                operation=EnvironmentFactoryOperation.RECONNECT,
                reconnect_metadata=record.reconnect_metadata,
            )
            try:
                result = await factory.create(reconnect)
                _require_result(result)
            except Exception:
                if not await _positively_disposed(factory, reconnect):
                    # The allocation may hold the run's only copy of its work;
                    # keep the record for a retry.
                    raise
                if not create:
                    await records.remove(
                        key.owner_session_id, record.allocation_session_id, {attempt_id}
                    )
                    return None
                record = await records.replace_with_fresh(record_id, record)
            else:
                return await self._adopt_reconnected(
                    key, factory, result, record, generation, attempt_id, bind=create
                )
        if not create:
            return None
        return await self._create_recorded(key, factory, request, record, generation, attempt_id)

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
                workflow_owner=True,
            )
        try:
            if record.state == "created" and bound is not None and binding is not None:
                record = await records.update(
                    key.record_id,
                    record,
                    state="live",
                    binding_state=_binding_recovery_state(binding, bound),
                )
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
    ) -> _SharedAllocation:
        """Create a recorded allocation under its durable identity and record what it returned."""

        records = self._records
        assert records is not None
        record_id = key.record_id
        result = await factory.create(
            self._shared_request(request, record.allocation_session_id, recorded=True)
        )
        _require_result(result)
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
                workflow_owner=True,
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
                record.allocation_session_id in held
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

    def _attach(self, entry: _SharedAllocation, session_id: str) -> _Attachment:
        attachment = _Attachment(session_id=session_id)
        entry.attachments.add(attachment)
        return attachment

    def _detach(self, entry: _SharedAllocation, attachment: _Attachment) -> None:
        if attachment.released:
            return
        attachment.released = True
        entry.attachments.discard(attachment)
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
        elif entry.result.release is not None:
            await entry.result.release(EnvironmentFactoryReleaseAction.DISCARD)
        elif (runner := entry.environment.runner) is not None:
            await runner.close()
        await self._forget_record(entry)

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
        a failed creation), and every record once the run's newest attempt has
        completed. A record whose run can still resume is left alone, and so is
        anything this process holds. Returns how many were disposed of.
        ``recover_incomplete_sessions`` calls it for each session it inspects.
        """

        records = self._records
        if records is None:
            return 0
        owner = await self._session_store.load(owner_session_id)
        if owner is None or owner.provider_name != _WORKFLOW_JOURNAL_PROVIDER:
            # Only workflow runs record their shared allocations.
            return 0
        snapshot = await records.load(owner_session_id)
        if snapshot is None or not (snapshot.records or snapshot.orphans):
            return 0
        candidates = list(snapshot.orphans.values())
        if await self._owner_ended(owner):
            candidates.extend(snapshot.records.values())
        held = self._held_allocation_ids()
        reaped = 0
        for record in candidates:
            if record.allocation_session_id in held:
                continue
            factory = self._factory_lookup(record.environment_name)
            if factory is None:
                continue
            fenced = await records.mark_reaping(record)
            if fenced is None:
                continue
            try:
                if not await _dispose_recorded(factory, fenced):
                    continue
            except Exception as error:
                _LOGGER.warning(
                    "Disposing of a recorded shared allocation of %r failed (%s); "
                    "the next recovery retries it.",
                    record.environment_name,
                    type(error).__name__,
                )
                continue
            await records.remove_reaped(fenced)
            reaped += 1
        return reaped

    async def _owner_ended(self, owner: Session) -> bool:
        """Whether the workflow run's newest attempt has journaled its completion."""

        owner_session_id = owner.id
        workflow_name = owner.metadata.get("cayu.workflow")
        if type(workflow_name) is not str:
            return False
        # Imported here: the workflow package imports the runtime.
        from cayu.workflows.journal import EventStoreJournal

        latest = await EventStoreJournal(
            self._session_store, owner_session_id, workflow_name
        ).latest_attempt_id()
        if latest is None:
            return False
        completions = await self._session_store.query_events(
            EventQuery(
                session_id=owner_session_id,
                event_type=EventType.WORKFLOW_COMPLETED,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=16,
            )
        )
        return any(record.event.payload.get("attempt_id") == latest for record in completions)

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
        _require_shareable_factory(factory)
        self._in_use = True
        entry = await self._acquire(key, factory, request, None)
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


def _require_shareable_factory(factory: EnvironmentFactory) -> None:
    """Refuse a factory whose allocations can't be shared, before asking it for one."""

    if factory.egress_authority_identity is not None:
        raise RuntimeError(
            "An environment factory with an egress authority can't share its allocation."
        )
    if getattr(factory, "deferred_materialization", False) is True:
        raise RuntimeError(
            "A deferred-materialization environment factory can't share its allocation."
        )


def _require_shareable(
    factory: EnvironmentFactory, environment: Environment, environment_name: str
) -> None:
    """Refuse an allocation whose authority might depend on the session that created it.

    Later sessions of the scope never reach the factory, so whatever the first
    request was granted must be the factory's fixed configuration: no vault or
    credential proxy, the factory's configured artifact store, knowledge store
    and scope, and no MCP servers.
    """

    reasons: list[str] = []
    if factory.secret_resolution_scope != "static":
        reasons.append("its secret resolution scope is dynamic")
    if environment.vault is not None or environment.proxy is not None:
        reasons.append("it exposes a vault or credential proxy")
    if environment.knowledge_store is not getattr(
        factory, "configured_knowledge_store", None
    ) or environment.knowledge_access_scope != getattr(
        factory, "configured_knowledge_access_scope", None
    ):
        reasons.append("its knowledge store or scope isn't the factory's configured one")
    if environment.artifact_store is not factory.configured_artifact_store:
        reasons.append("its artifact store isn't the factory's configured one")
    if environment.mcp_servers:
        reasons.append("it exposes MCP servers")
    if reasons:
        raise RuntimeError(
            f"Environment {environment_name!r} can't share its allocation: "
            + "; ".join(reasons)
            + ". Every session of a scope would get the first session's authority."
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


async def _dispose_recorded(factory: EnvironmentFactory, record: SharedAllocationRecord) -> bool:
    """Dispose of a recorded allocation that no run will use again, without copying back.

    A reconnectable one goes through the factory's RECONNECT path and is
    discarded. Anything else can only be disposed of from the metadata of the
    result that created it, through ``discard_unreachable_allocation``; a
    factory that can't leaves it to lapse (a brokered sandbox's lease does).
    """

    request = EnvironmentFactoryRequest(
        session_id=record.allocation_session_id,
        agent_name=record.agent_name,
        environment_name=record.environment_name,
    )
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
    else:
        _LOGGER.info(
            "Environment %r can't dispose of an unreachable shared allocation; it is "
            "left to lapse.",
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
        _require_shareable_factory(self._inner)
        registry = self._registry
        key = await registry.resolve_key(self._session, self._environment_name, self._scope)
        entry = await registry._acquire(key, self._inner, request, self._session)
        attachment = registry._attach(entry, request.session_id)
        try:
            environment = entry.environment
            bound = entry.bound
            attached = Environment(
                copy_environment_spec(environment.spec),
                workspace=bound.workspace if bound is not None else environment.workspace,
                artifact_store=environment.artifact_store,
                runner=entry.runner,
                vault=environment.vault,
                proxy=environment.proxy,
                knowledge_store=environment.knowledge_store,
                knowledge_access_scope=environment.knowledge_access_scope,
                binding=_SharedAllocationBinding(registry, entry, attachment),
                mcp_servers=environment.mcp_servers,
                workspace_instructions=environment.workspace_instructions,
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
