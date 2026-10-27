"""Environment registrations that change while the application runs.

Owns the parts of the environment surface that act on registrations after
setup: trimming or closing factories' idle resources, unregistering an
environment once nothing can still need it, and recreating a per-job
environment from a session's durable rebuild record. Kept apart from
``CayuApp`` so each part stays usable and testable on its own.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractContextManager, asynccontextmanager, nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from cayu._validation import require_clean_nonblank
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.environments.rebuild import (
    EnvironmentRebuild,
    EnvironmentRebuildMismatch,
    EnvironmentRebuildRecord,
    EnvironmentRebuildRequest,
    declared_rebuild_record,
)
from cayu.events import EventType
from cayu.exceptions import EnvironmentInUse
from cayu.runtime._environment_allocation import environment_allocation_parent_session_id
from cayu.runtime._environment_lifecycle import _PENDING_ALLOCATION_DISPOSAL_KEY
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions._completion_finalization import PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import Session, SessionStatus

if TYPE_CHECKING:
    from cayu._application_environment_registry import ApplicationEnvironmentRegistry
    from cayu.applications import CayuApp

logger = logging.getLogger(__name__)

# Terminal sessions start a new invocation on resume, which resolves the
# environment again and fails before any work when it is not registered.
_TERMINAL_STATUSES = frozenset({SessionStatus.COMPLETED, SessionStatus.FAILED})
_SESSION_PAGE_SIZE = 500
# Forks and subagents of forks rarely nest deeply; stop looking after this many.
_MAX_REBUILD_LINEAGE_DEPTH = 32
_FACTORY_EVENT_TYPES = (
    EventType.ENVIRONMENT_FACTORY_STARTED,
    EventType.ENVIRONMENT_FACTORY_COMPLETED,
    EventType.ENVIRONMENT_FACTORY_FAILED,
)


async def _release_after(
    previous: asyncio.Task[None], release: Callable[[], Awaitable[None]]
) -> None:
    """Run ``release`` once ``previous`` settled, then report ``previous`` too.

    The later release runs whatever the earlier one's outcome. If the earlier
    one failed, this task fails as well (chained to that error), so the caller
    that replaced it in the ownership map still reports it.
    """

    earlier: BaseException | None = None
    try:
        await asyncio.shield(previous)
    except asyncio.CancelledError:
        if not previous.cancelled():
            raise  # This chained release itself was cancelled.
        earlier = RuntimeError("An earlier idle-resource release was cancelled.")
    except Exception as error:
        earlier = error
    await release()
    if earlier is not None:
        raise RuntimeError(
            "An earlier idle-resource release failed before this one ran."
        ) from earlier


def _retrieve_idle_release_outcome(task: asyncio.Task[None]) -> None:
    # A release that settles after its caller stopped waiting must not be
    # logged as an unretrieved task exception; the next drain or close reports it.
    if not task.cancelled():
        task.exception()


async def release_idle_environment_resources(
    registry: ApplicationEnvironmentRegistry,
    releases: dict[str, tuple[bool, asyncio.Task[None]]],
    *,
    close: bool,
    deadline: float,
) -> bool:
    """Trim (or close) factories' idle resources, such as warm spares, by ``deadline``.

    Each factory's release runs as an owned task that is never cancelled:
    one still running at the deadline stays owned in ``releases`` (the result
    is False), and a later call waits for that same task instead of starting
    another. A close requested while a trim is still running runs after it.
    """

    loop = asyncio.get_running_loop()
    # A factory a recovery rebuilt for itself serves no later session, so its
    # idle resources are closed (not just trimmed) and it is then forgotten.
    retired = dict(registry.retired_scoped_registrations)
    current_registrations = (*registry.registrations.items(), *retired.items())
    if not any(registration.factory is not None for _, registration in current_registrations):
        return True
    if deadline - loop.time() <= 0:
        return False
    owned: list[tuple[str, asyncio.Task[None]]] = []
    released = True
    for name, registration in current_registrations:
        factory = registration.factory
        if factory is None:
            continue
        closing = close or name in retired
        release = factory.close_idle_resources if closing else factory.release_idle_resources
        previous = releases.pop(name, None)
        if previous is not None and previous[1].done():
            # Settled after an earlier caller stopped waiting: report it once.
            if previous[1].cancelled() or previous[1].exception() is not None:
                released = False
            previous = None
        if previous is not None:
            previous_close, previous_task = previous
            if previous_close or not closing:
                # The same release, or a close that already covers a trim.
                releases[name] = previous
                owned.append((name, previous_task))
                continue
            task = loop.create_task(
                _release_after(previous_task, release),
                name=f"cayu-idle-environment-release:{name}",
            )
        else:
            task = loop.create_task(release(), name=f"cayu-idle-environment-release:{name}")
        task.add_done_callback(_retrieve_idle_release_outcome)
        releases[name] = (closing, task)
        owned.append((name, task))
    if not owned:
        return released
    remaining = deadline - loop.time()
    if remaining > 0:
        await asyncio.wait([task for _, task in owned], timeout=remaining)
    for name, task in owned:
        if not task.done():
            released = False  # Still owned; a later drain observes its settlement.
            continue
        current = releases.get(name)
        if current is not None and current[1] is task:
            del releases[name]
        if task.cancelled() or task.exception() is not None:
            released = False
        elif name in retired:
            registry.forget_retired_scoped_registration(name)
    return released


def _checkpoint_owes_environment_cleanup(
    checkpoint: Mapping[str, Any] | None, environment_name: str
) -> bool:
    """Whether a terminal session's checkpoint still owes the factory a disposal.

    Pending allocation setup (unreaped intents, and published receipts whose
    setup failed before the first transcript) is the lifecycle's own census,
    ``EnvironmentLifecycle.pending_allocation_names``.
    """

    if not checkpoint:
        return False
    marker = checkpoint.get(PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY)
    if isinstance(marker, Mapping) and marker.get("environment_name") == environment_name:
        return True
    disposals = checkpoint.get(_PENDING_ALLOCATION_DISPOSAL_KEY)
    return isinstance(disposals, Mapping) and disposals.get(environment_name) is not None


async def _session_owes_environment_cleanup(
    app: CayuApp, session: Session, environment_name: str
) -> bool:
    if _checkpoint_owes_environment_cleanup(
        await app.session_store.load_checkpoint(session.id), environment_name
    ):
        return True
    try:
        pending = await app._environment_lifecycle.pending_allocation_names(session)
    except ValueError:
        return True  # State the lifecycle cannot read cannot be shown to be clean.
    return environment_name in pending


async def _refuse_while_sessions_need(app: CayuApp, environment_name: str) -> None:
    """Refuse while a stored session could still run in, or owe cleanup to, the environment."""

    store = app.session_store
    cursor: str | None = None
    while True:
        page = await store.list_sessions(
            SessionQuery(environment_name=environment_name, limit=_SESSION_PAGE_SIZE, cursor=cursor)
        )
        for session in page.sessions:
            if session.environment_name != environment_name:
                continue
            if await _built_from_another_record(app, session, environment_name):
                # It can never run or be cleaned up under this registration;
                # its recovery rebuilds its own environment (or is refused).
                continue
            if session.status not in _TERMINAL_STATUSES:
                raise EnvironmentInUse(
                    environment_name,
                    "unfinished_session",
                    f"session {session.id} is {session.status.value}; it can still run, resume "
                    "or be recovered in this environment",
                )
            if await _session_owes_environment_cleanup(app, session, environment_name):
                raise EnvironmentInUse(
                    environment_name,
                    "pending_cleanup",
                    f"session {session.id} still owes environment cleanup; run "
                    "recover_incomplete_session for it first",
                )
        cursor = page.next_cursor
        if cursor is None or not page.sessions:
            return


async def _refuse_while_process_needs(app: CayuApp, environment_name: str) -> None:
    """Refuse while this process still owns setup or cleanup for a session in it."""

    _refuse_while_idle_release_runs(app, environment_name)

    named, unattributed = app._environment_lifecycle.sessions_with_environment_work(
        environment_name
    )
    in_use = set(named)
    for session_id in unattributed:
        session = await app.session_store.load(session_id)
        # Cleanup for a session that no longer has a record cannot be attributed.
        if session is None or session.environment_name == environment_name:
            in_use.add(session_id)
    if in_use:
        raise EnvironmentInUse(
            environment_name,
            "in_process_work",
            "this process still owns environment setup or cleanup for "
            f"{len(in_use)} session(s); let them finish or call drain_environment_cleanups",
        )


def _refuse_while_idle_release_runs(app: CayuApp, environment_name: str) -> None:
    pending_release = app._idle_resource_releases.get(environment_name)
    if pending_release is not None and not pending_release[1].done():
        raise EnvironmentInUse(
            environment_name,
            "in_process_work",
            "an idle-resource release for its factory is still running",
        )


async def unregister_application_environment(app: CayuApp, name: str) -> None:
    """Remove one registration once no session can still need it.

    See ``CayuApp.unregister_environment`` for the rule.
    """

    environment_name = require_clean_nonblank(name, "environment.name")
    registry = app._environment_registry
    if environment_name not in registry.registrations:
        raise KeyError(f"Environment not registered: {environment_name}")
    if registry.default_name == environment_name:
        raise EnvironmentInUse(
            environment_name, "default_environment", "it is the application's default"
        )
    for agent_name, agent in app._agent_registry.registrations.items():
        if agent.default_environment_name == environment_name:
            raise EnvironmentInUse(
                environment_name,
                "agent_default",
                f"agent {agent_name} names it as its default environment",
            )
    await _refuse_while_process_needs(app, environment_name)
    await _refuse_while_sessions_need(app, environment_name)
    # Setup that started while the store was read is owned by this process by now.
    named, _ = app._environment_lifecycle.sessions_with_environment_work(environment_name)
    if named:
        raise EnvironmentInUse(
            environment_name,
            "in_process_work",
            "a session started setup in it while it was being unregistered",
        )
    # A drain or close may have started a release while the store was read.
    _refuse_while_idle_release_runs(app, environment_name)
    registry.unregister(environment_name)
    settled = app._idle_resource_releases.get(environment_name)
    if settled is not None and settled[1].done():
        del app._idle_resource_releases[environment_name]


async def _session_rebuild_record(
    app: CayuApp, session: Session, environment_name: str
) -> tuple[str, EnvironmentRebuildRecord | None] | None:
    """Find the record of the factory that last built ``environment_name`` for the session.

    Reads the latest factory event (started, completed or failed, so a crash
    during allocation still counts) or its own fork event's record. Older forks
    and spawned sessions without their own evidence follow their source. Returns
    ``None`` without such evidence, and ``(session_id, None)`` when that factory
    declared no record.
    """

    store = app.session_store
    current: Session | None = session
    seen: set[str] = set()
    for _ in range(_MAX_REBUILD_LINEAGE_DEPTH):
        if current is None or current.id in seen or current.environment_name != environment_name:
            return None
        seen.add(current.id)
        events = await store.query_events(
            EventQuery(
                session_id=current.id,
                event_types=(*_FACTORY_EVENT_TYPES, EventType.SESSION_FORKED),
                environment_name=environment_name,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if events and (
            events[0].event.type is not EventType.SESSION_FORKED
            or "environment_rebuild" in events[0].event.payload
        ):
            payload = events[0].event.payload.get("environment_rebuild")
            record = None if payload is None else EnvironmentRebuildRecord.from_payload(payload)
            return current.id, record
        # Older forks and spawned children without a snapshot inherit their source's.
        parent_id = environment_allocation_parent_session_id(current)
        current = None if parent_id is None else await store.load(parent_id)
    return None


async def fork_environment_rebuild_record(
    app: CayuApp, session: Session, environment_name: str | None
) -> EnvironmentRebuildRecord | None:
    """Snapshot the selected environment's record into the fork's own durable event."""

    if environment_name is None:
        return None
    if session.environment_name == environment_name:
        found = await _session_rebuild_record(app, session, environment_name)
        if found is not None:
            return found[1]
    return app._environment_registry.rebuild_record(environment_name)


async def _build_from_record(
    app: CayuApp, session: Session, name: str
) -> tuple[EnvironmentRebuild, EnvironmentRebuildRecord] | None:
    """Ask the rebuilder for the session's own record; ``None`` when none applies."""

    found = await _session_rebuild_record(app, session, name)
    if found is None:
        return None
    source_session_id, record = found
    if record is None:
        return None
    rebuilt = await _rebuild_from(app, name, source_session_id, record)
    return None if rebuilt is None else (rebuilt, record)


async def _rebuild_from(
    app: CayuApp, name: str, source_session_id: str, record: EnvironmentRebuildRecord
) -> EnvironmentRebuild | None:
    """Ask the rebuilder for ``record``'s kind; ``None`` when no rebuilder handles it."""

    rebuilder = app._environment_registry.rebuilders.get(record.kind)
    if rebuilder is None:
        return None
    rebuilt = await rebuilder(
        EnvironmentRebuildRequest(
            environment_name=name, record=record, session_id=source_session_id
        )
    )
    if type(rebuilt) is not EnvironmentRebuild:
        raise TypeError("An environment rebuilder must return an EnvironmentRebuild.")
    if rebuilt.spec.name != name:
        raise ValueError(
            f"Environment rebuilder for {record.kind!r} built {rebuilt.spec.name!r}, not {name!r}."
        )
    if declared_rebuild_record(rebuilt.factory) != record:
        raise ValueError(
            "A rebuilt environment factory must declare the rebuild record it was built from."
        )
    return replace(
        rebuilt,
        artifact_store=app._environment_registry.artifact_store_for_rebuild(rebuilt.artifact_store),
    )


async def _built_from_another_record(app: CayuApp, session: Session, name: str) -> bool:
    registered_record = app._environment_registry.rebuild_record(name)
    if registered_record is None:
        return False
    found = await _session_rebuild_record(app, session, name)
    return found is not None and found[1] is not None and found[1] != registered_record


def _session_environment_name(session: Session, environment_name: str | None) -> str | None:
    name = session.environment_name
    if name is None or (environment_name is not None and environment_name != name):
        return None
    return name


async def ensure_session_environment(
    app: CayuApp, session: Session | str | None, environment_name: str | None = None
) -> None:
    """Let a session run only in an environment built from its own rebuild record.

    When the session's environment is not registered and a rebuilder handles
    its record's kind, register it rebuilt from that record. When the name is
    registered by a factory that declares a record and the session's
    environment was built from a different one, raise
    ``EnvironmentRebuildMismatch`` before any work: per-job names are reused,
    and another job's environment must never be reused silently. A side
    without a record makes no claim. Reads nothing unless this process has a
    rebuilder or a registered factory with a record. ``session`` is the session
    or its private id; ``environment_name`` is the environment the caller will
    run in, and only the session's own environment is checked.
    """

    registry = app._environment_registry
    if session is None or not (registry.rebuilders or registry.has_rebuild_records):
        return
    if isinstance(session, str):
        session = await app.session_store.load(session)
        if session is None:
            return
    name = _session_environment_name(session, environment_name)
    if name is None:
        return
    if name not in registry.registrations and registry.rebuilders:
        async with registry.rebuild_lock(name):
            if name not in registry.registrations:
                built = await _build_from_record(app, session, name)
                if built is not None:
                    rebuilt, record = built
                    registry.register_factory(
                        rebuilt.spec,
                        rebuilt.factory,
                        artifact_store=rebuilt.artifact_store,
                        registration_site=("cayu.environments.rebuild", record.kind),
                    )
                    return  # Built from exactly this session's record.
    if name in registry.registrations and await _built_from_another_record(app, session, name):
        raise EnvironmentRebuildMismatch(session.id, name)


class SessionEnvironments:
    """Recovery's view of per-job environments for one application.

    ``recovery_scope`` gives one session's recovery the environment built from
    its own record. ``rebuild_required`` reports, without calling anything,
    which rebuild kind recovery would need, for planning and preflight.
    """

    def __init__(self, app: CayuApp) -> None:
        self._app = app

    async def rebuild_required(self, session: Session) -> str | None:
        """The kind recovery would rebuild for ``session``, or ``None``.

        Reads only the session store and this process's registrations: it
        never calls a rebuilder, a factory or registers anything.
        """

        app = self._app
        registry = app._environment_registry
        name = session.environment_name
        if name is None or not registry.rebuilders:
            return None
        try:
            found = await _session_rebuild_record(app, session, name)
        except ValueError:
            return None
        if found is None or found[1] is None or found[1].kind not in registry.rebuilders:
            return None
        record = found[1]
        registered_record = registry.rebuild_record(name)
        if name not in registry.registrations or (
            registered_record is not None and registered_record != record
        ):
            return record.kind
        return None

    @asynccontextmanager
    async def recovery_scope(
        self, session: Session, *, read_only: bool = False
    ) -> AsyncIterator[None]:
        """Resolve the environment built from the session's own record during recovery.

        Recovery settles a crashed session and releases what it allocated, so
        it must not be stranded when its slot name was reused for another job.
        A session whose environment is not registered is rebuilt from its
        record, as ``ensure_session_environment`` does. When the name is
        registered from another record, the session's own environment is
        rebuilt separately and resolved only inside this recovery (and the
        cleanup it starts); the shared registration and its sessions are
        untouched. When that can't be done (no rebuilder for the kind, or the
        rebuilder fails), the name resolves to a stand-in that refuses every
        factory call and carries the session's record, so its refusals never
        erase which environment the session belongs to. If the session's record
        can't be read while another record is registered, recovery is refused
        with ``EnvironmentRebuildMismatch`` rather than trusting the
        registration. A failing rebuilder never aborts other sessions.

        ``read_only`` (planning and preflight) never calls a rebuilder or a
        factory and registers nothing. Planning skips preflight for sessions
        that need a rebuild (see ``rebuild_required``).
        """

        app = self._app
        registry = app._environment_registry
        name = session.environment_name
        scope: AbstractContextManager[None] = nullcontext()
        if name is not None and (registry.rebuilders or registry.has_rebuild_records):
            scope = await self._scope_for(session, name, read_only=read_only)
        with scope:
            yield

    async def _scope_for(
        self, session: Session, name: str, *, read_only: bool
    ) -> AbstractContextManager[None]:
        app = self._app
        registry = app._environment_registry
        if name not in registry.registrations:
            if not read_only and registry.rebuilders:
                try:
                    await ensure_session_environment(app, session)
                except Exception:
                    # No other job's factory is involved: recovery continues
                    # as it would without a registration.
                    logger.warning(
                        "Rebuilding environment %r for session recovery failed.",
                        name,
                        exc_info=True,
                    )
            return nullcontext()
        registered_record = registry.rebuild_record(name)
        if registered_record is None:
            return nullcontext()  # The registration makes no claim.
        try:
            found = await _session_rebuild_record(app, session, name)
        except Exception as exc:
            # Unknown ownership must not hand the session's allocation to a
            # registration that may belong to another job.
            raise EnvironmentRebuildMismatch(session.id, name, verified=False) from exc
        if found is None:
            return nullcontext()
        source_session_id, record = found
        if record is None or record == registered_record:
            return nullcontext()
        if not read_only:
            rebuilt = None
            try:
                rebuilt = await _rebuild_from(app, name, source_session_id, record)
                if rebuilt is not None:
                    return registry.scoped_factory_registration(
                        rebuilt.spec,
                        rebuilt.factory,
                        artifact_store=rebuilt.artifact_store,
                        registration_site=("cayu.environments.rebuild", record.kind),
                        retain_idle_resources=True,
                    )
            except Exception:
                logger.warning(
                    "Rebuilding environment %r for session recovery failed; recovering without it.",
                    name,
                    exc_info=True,
                )
        registered = registry.registrations[name]
        assert registered.factory is not None
        return registry.scoped_factory_registration(
            registered.spec,
            _ForeignRecordFactory(registered.factory, record, session.id, name),
            registration_site=("cayu.environments.rebuild", record.kind),
        )


class _ForeignRecordFactory(EnvironmentFactory):
    """Stands in for a session's own environment when nothing here can rebuild it.

    Recovery can still settle the session (interrupt it, finish its records)
    as long as that needs no environment. Any factory call is refused, so the
    registered factory, built for another job, never creates, reconnects to or
    disposes of this session's allocation; such cleanup stays recorded as
    pending until the session is recovered where its environment exists.
    """

    def __init__(
        self,
        registered: EnvironmentFactory,
        record: EnvironmentRebuildRecord,
        session_id: str,
        environment_name: str,
    ) -> None:
        self._identity = registered.execution_profile_identity
        self._secret_resolution_scope = registered.secret_resolution_scope
        self._record = record
        self._session_id = session_id
        self._environment_name = environment_name

    @property
    def rebuild_record(self) -> EnvironmentRebuildRecord:
        # Factory events of a refused call keep naming the session's own record.
        return self._record

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity | None:
        return self._identity

    @property
    def secret_resolution_scope(self) -> Literal["static", "dynamic"]:
        return self._secret_resolution_scope

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        del request
        raise EnvironmentRebuildMismatch(self._session_id, self._environment_name)
