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

Ownership is process-local by design. The shared allocation is not part of any
session's durable allocation records; its request carries this process's
``process_local_owner`` token, so a factory that labels its resources with it
(Docker coding does) leaves them to ``reap_orphaned_docker_coding_containers``
after a crash, and a leased provider (a brokered sandbox) lets the lease lapse.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

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


def shared_allocation_session_id(owner_session_id: str) -> str:
    """Return the allocation identity one shared allocation is requested under.

    Factories key their allocation off ``request.session_id`` (Docker coding
    names its container after it). The identity names the scope's owner and
    is fresh for every shared allocation, so a run resumed in another process
    never reattaches the container a crashed process left behind; that one
    keeps its dead owner's process-local label and only the reaper removes it.
    Within one process, a retry after a failed creation reuses the identity
    until the allocation is adopted or positively discarded, so it adopts or
    cleans up what the failed attempt left.
    """

    return f"{owner_session_id}.shared-{secrets.token_hex(8)}"


class SharedAllocationScopeClosedError(RuntimeError):
    """Raised when a session asks to share an allocation whose owner isn't running here."""


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
        # Owner session -> the workflow attempt that opened it most recently.
        self._owner_attempts: dict[str, str] = {}

    def bind_session_presence(self, session_running_here: Callable[[str], bool]) -> None:
        """Tell the registry how to see whether a session's run is live in this process."""

        self._session_running_here = session_running_here

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
                # A child of a session that is itself attached (a workflow
                # step's subagent) joins that session's allocation, so an agent
                # and its subagents share one sandbox wherever the agent runs.
                for key, entry in self._live.items():
                    if (
                        key.scope is scope
                        and key.environment_name == environment_name
                        and any(
                            attachment.session_id == parent and not attachment.released
                            for attachment in entry.attachments
                        )
                    ):
                        return key
            owner = parent or session.id
            return SharedAllocationKey(scope, owner, owner, environment_name)
        if scope is AllocationSharingScope.CAUSAL_BUDGET:
            owner = await self._top_ancestor(session)
            return SharedAllocationKey(
                scope, session.causal_budget_id or owner, owner, environment_name
            )
        raise ValueError("A session-scoped allocation is not shared.")

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

    def open_owner(self, owner_session_id: str, *, attempt_id: str | None = None) -> None:
        """Let sessions share allocations owned by ``owner_session_id`` until it ends.

        Workflow runs open their anchor session this way, naming the workflow
        attempt; an agent session that runs in this process is open while its
        run is.
        """

        self._in_use = True
        self._open_owners.add(owner_session_id)
        self._ended_owners.pop(owner_session_id, None)
        if attempt_id is not None:
            self._owner_attempts[owner_session_id] = attempt_id

    def _owner_generation(self, owner_session_id: str) -> int:
        return self._owner_generations.get(owner_session_id, 0)

    async def _require_open(self, key: SharedAllocationKey, session: Session) -> None:
        owner = key.owner_session_id
        if owner in self._open_owners:
            return
        if owner == session.id:
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
        self._owner_attempts.pop(owner_session_id, None)
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
        if not self._owner_has_entries(owner_session_id):
            self._owner_generations.pop(owner_session_id, None)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ExceptionGroup("Releasing shared environment allocations failed.", failures)

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
        for entry in tuple(self._live.values()):
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
        session: Session,
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

    async def _create(
        self,
        key: SharedAllocationKey,
        factory: EnvironmentFactory,
        request: EnvironmentFactoryRequest,
        generation: int,
    ) -> _SharedAllocation:
        allocation_session_id = self._allocation_ids.setdefault(
            key, shared_allocation_session_id(key.owner_session_id)
        )
        try:
            shared_request = replace(
                request,
                session_id=allocation_session_id,
                parent_session_id=None,
                operation=EnvironmentFactoryOperation.CREATE,
                reconnect_metadata={},
                replacement_predecessor={},
                interaction_id=None,
                # The shared allocation's records end with this process.
                process_local_owner=current_process_local_owner(),
            )
            result = await factory.create(shared_request)
            if type(result) is not EnvironmentFactoryResult:
                raise TypeError("Environment factory create must return EnvironmentFactoryResult.")
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
                    bound = await binding.bind(
                        environment.workspace,
                        environment.runner,
                        session_id=allocation_session_id,
                        agent_name=request.agent_name,
                        environment_name=key.environment_name,
                    )
                if (
                    key.owner_session_id not in self._open_owners
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
                entry = _SharedAllocation(key=key, factory=factory, result=result, bound=bound)
                if bound is not None and binding is not None:
                    with contextlib.suppress(Exception):
                        binding.abandon(bound)
                # The entry owns the allocation until a discard positively
                # succeeds: a failed one stays held, and a later end_owner or
                # close retries it. A retried creation then needs a fresh
                # identity, since this entry still owns the old one.
                entry.release_outcome = "failed"
                entry.discard_task = asyncio.ensure_future(self._discard(entry))
                self._closing.add(entry)
                self._allocation_ids.pop(key, None)
                if await asyncio.shield(entry.discard_task):
                    self._closing.discard(entry)
                raise
            entry = _SharedAllocation(key=key, factory=factory, result=result, bound=bound)
            self._allocation_ids.pop(key, None)
            self._live[key] = entry
            return entry
        finally:
            self._creating.pop(key, None)

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
        binding = entry.environment.binding
        if entry.bound is not None and binding is not None:
            await binding.finalize(entry.bound, outcome=outcome)
            return
        if entry.result.release is not None:
            await entry.result.release(EnvironmentFactoryReleaseAction.DISCARD)
            return
        runner = entry.environment.runner
        if runner is not None:
            await runner.close()

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
                if entry.result.release is not None:
                    await asyncio.wait_for(
                        entry.result.release(EnvironmentFactoryReleaseAction.DISCARD),
                        timeout=entry.result.release_timeout_s,
                    )
                elif entry.environment.runner is not None:
                    await entry.environment.runner.close()
            except Exception as error:
                _LOGGER.warning(
                    "Discarding the shared allocation of environment %r failed (%s).",
                    name,
                    type(error).__name__,
                )
                return False
            entry.resources_discarded = True
        return True

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
        if self._inner.egress_authority_identity is not None:
            raise RuntimeError(
                "An environment factory with an egress authority can't share its allocation."
            )
        if getattr(self._inner, "deferred_materialization", False) is True:
            raise RuntimeError(
                "A deferred-materialization environment factory can't share its allocation."
            )
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
