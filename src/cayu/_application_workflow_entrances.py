"""CayuApp entrances that workflow code uses to raise approvals and react to them.

``CayuApp`` inherits these; they live here so the application module stays
within its size limit. Each one composes existing parts: a proposal is an
ordinary session run, watchers are delivered by ``dispatch_event_watchers``,
and the shared-allocation entrances read and release the records a workflow
run keeps (``cayu.runtime._shared_allocation_operations``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, TypeVar

from cayu.observability.watchers import EventWatcher, EventWatcherRunResult
from cayu.runtime._event_watcher_delivery import dispatch_event_watchers
from cayu.runtime.application_lifecycle import _admitted_entrance, _tracked_entrance

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.approvals.proposals import ToolCallPreview, ToolCallProposalOutcome
    from cayu.environments.shared_allocations import (
        SharedAllocationPage,
        SharedAllocationRelease,
        SharedAllocationReleaseMode,
    )
    from cayu.sessions.invocation import InvocationOriginClaim
    from cayu.tools.exposure import ToolCapabilityCeiling


_T = TypeVar("_T")


class WorkflowEntrances:
    """Tool-call proposals and registered event watchers for ``CayuApp``."""

    _event_watchers: tuple[EventWatcher, ...] = ()

    def register_event_watcher(self: CayuApp, watcher: EventWatcher) -> EventWatcher:
        """Declare a watcher that the hosting process runs.

        Registration starts nothing. ``create_server``, ``mount_control_plane``
        (and so ``cayu serve``) run registered watchers for the server's
        lifetime, and ``cayu.observability.serve_event_watchers(app, stop)``
        runs them from a worker. Watcher names must be unique.
        """
        # Delivery accepts exactly EventWatcher; refuse anything else up front.
        if type(watcher) is not EventWatcher:
            raise TypeError("register_event_watcher requires an EventWatcher.")
        if self.lifecycle_state != "open":
            raise RuntimeError("Register event watchers before the application shuts down.")
        if any(existing.name == watcher.name for existing in self._event_watchers):
            raise ValueError(f"Event watcher already registered: {watcher.name}")
        self._event_watchers = (*self._event_watchers, watcher)
        return watcher

    @property
    def event_watchers(self) -> tuple[EventWatcher, ...]:
        """Watchers declared with ``register_event_watcher``, in registration order."""
        return self._event_watchers

    @_admitted_entrance
    async def _run_admitted_workflow_command(
        self: CayuApp, operation: Callable[[], Awaitable[_T]]
    ) -> _T:
        """Run one workflow ``command`` step as admitted application work.

        Shutdown waits for it like an agent step's run, and a sealed application
        refuses a new one unless admitted work started it.
        """
        return await operation()

    @_admitted_entrance
    async def run_event_watchers(
        self: CayuApp,
        watchers: Iterable[EventWatcher],
        *,
        limit: int = 100,
    ) -> list[EventWatcherRunResult]:
        """Deliver ordered durable events under renewable, fenced watcher leases."""
        return await dispatch_event_watchers(
            watchers,
            limit=limit,
            get_session_store=lambda: self.session_store,
            get_watcher_store=lambda: self.event_watcher_store,
            supervisor=self._event_watcher_supervisor,
            redactor=self._secret_redactor,
            project_record=self._project_persisted_event_record_for_exposure,
        )

    @_admitted_entrance
    async def propose_tool_call(
        self: CayuApp,
        agent_name: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        tool_call_id: str | None = None,
        **request_fields: Any,
    ) -> ToolCallProposalOutcome:
        """Raise one exact tool call for ``agent_name`` without a model turn.

        The call goes through the agent's tool policy like a model-raised call.
        When the policy requires approval, this returns ``pending_approval`` and
        the approval is resolved like any other, through
        ``resolve_tool_approval`` or the Developer dashboard; approval runs the
        tool and a denial is recorded. ``request_fields`` pass through to the
        ``RunRequest`` built by ``cayu.approvals.tool_call_proposal_request``.
        A proposal with a registered secret anywhere in its tool name, call ID,
        argument keys or values is refused, because it is recorded in the
        session transcript.
        """
        from cayu.approvals.proposals import (
            ToolCallProposalOutcome,
            _require_secret_free_proposal,
            _require_unredacted_proposal_message,
            tool_call_proposal_request,
        )

        _require_secret_free_proposal(
            self._secret_redactor,
            tool_name=tool_name,
            arguments=arguments,
            tool_call_id=tool_call_id,
        )
        self._get_registered_agent(agent_name)
        request = tool_call_proposal_request(
            agent_name, tool_name, arguments, tool_call_id=tool_call_id, **request_fields
        )
        _require_unredacted_proposal_message(self._secret_redactor, request)
        async with aclosing(self.run(request)) as stream:
            events = [event async for event in stream]
        return ToolCallProposalOutcome.from_events(events)

    @_tracked_entrance
    async def read_tool_call_proposal(
        self: CayuApp, session_id: str
    ) -> ToolCallProposalOutcome | None:
        """Where a proposal session's call stands now, or None when no such session exists.

        It reads the session's persisted events as the public API shows them, so its
        identifiers are the ones ``propose_tool_call`` returned and
        ``resolve_tool_approval`` takes. A session that isn't a tool-call proposal
        raises ``ValueError``.
        """
        from cayu.approvals.proposals import ToolCallProposalOutcome
        from cayu.runtime._session_queries import query_all_event_records
        from cayu.runtime._tool_call_proposals import TOOL_CALL_PROPOSAL_PROVIDER_NAME
        from cayu.sessions.event_queries import EventQuery

        if type(session_id) is not str or not session_id.strip():
            raise ValueError("session_id must be a non-blank string.")
        session = await self.session_store.load(session_id)
        if session is None:
            return None
        if session.provider_name != TOOL_CALL_PROPOSAL_PROVIDER_NAME:
            raise ValueError(f"Session {session_id} is not a tool-call proposal.")
        records = await query_all_event_records(
            self.session_store, EventQuery(session_id=session_id, limit=1000)
        )
        return ToolCallProposalOutcome.from_events(
            self._project_persisted_event_record_for_exposure(record).event for record in records
        )

    @_tracked_entrance
    async def preview_tool_call(
        self: CayuApp,
        agent_name: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        tool_call_id: str | None = None,
        session_id: str | None = None,
        invocation_origin: InvocationOriginClaim | None = None,
        tool_capability_ceiling: ToolCapabilityCeiling | None = None,
        labels: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ToolCallPreview:
        """What ``propose_tool_call`` with these values would decide, recording nothing.

        It runs the same checks (registered, exposed or proposal-only, inside the
        ceiling, arguments matching the schema, no registered secret) and calls the
        agent's tool policy with ``metadata["cayu:dry_run"] = True``. No session,
        event or approval is created and no tool runs. A policy reading live state
        decides from it as it is now, so the proposal itself can still differ.
        """
        from cayu.approvals.proposals import _require_secret_free_proposal
        from cayu.runtime._tool_call_preview import preview_tool_call

        _require_secret_free_proposal(
            self._secret_redactor,
            tool_name=tool_name,
            arguments=arguments,
            tool_call_id=tool_call_id,
        )
        return await preview_tool_call(
            self,
            agent_name,
            tool_name,
            dict(arguments or {}),
            tool_call_id=tool_call_id or "preview",
            session_id=session_id,
            invocation_origin=invocation_origin,
            tool_capability_ceiling=tool_capability_ceiling,
            labels=dict(labels or {}),
            metadata=dict(metadata or {}),
        )

    @_tracked_entrance
    async def list_shared_allocations(
        self: CayuApp,
        *,
        workflow_run_id: str | None = None,
        environment_name: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        inactive_for_seconds: int | None = None,
    ) -> SharedAllocationPage:
        """List the shared allocations workflow runs have recorded, newest run first.

        Each page reads up to ``limit`` workflow runs (at most 200) and returns
        every allocation they recorded, so it can be short or empty and still
        have a ``next_cursor``. ``workflow_run_id`` reads one run. Each entry
        says whether its run looks like it is running, has stopped, or has
        ended, judged as ``release_shared_allocation`` judges it with the same
        ``inactive_for_seconds``. Reconnect metadata and binding state are never returned; the
        factory's result metadata is, redacted, as its
        ``environment.factory.completed`` event already shows it.
        """
        from cayu.runtime._shared_allocation_operations import (
            MAX_SHARED_ALLOCATION_RUNS_PER_PAGE,
            inactivity_window,
            list_shared_allocations,
        )

        if type(limit) is not int or not 1 <= limit <= MAX_SHARED_ALLOCATION_RUNS_PER_PAGE:
            raise ValueError(
                f"limit must be an integer from 1 to {MAX_SHARED_ALLOCATION_RUNS_PER_PAGE}."
            )
        for name, value in (
            ("workflow_run_id", workflow_run_id),
            ("environment_name", environment_name),
            ("cursor", cursor),
        ):
            if value is not None and (type(value) is not str or not value.strip()):
                raise ValueError(f"{name} must be a non-blank string.")
        return await list_shared_allocations(
            self._environment_lifecycle.shared_allocations,
            session_store=self.session_store,
            redact=self.redact_json,
            now=self._clock(),
            workflow_run_id=workflow_run_id,
            environment_name=environment_name,
            limit=limit,
            cursor=cursor,
            inactive_for_seconds=inactivity_window(inactive_for_seconds),
        )

    @_admitted_entrance
    async def release_shared_allocation(
        self: CayuApp,
        allocation_id: str,
        *,
        mode: SharedAllocationReleaseMode | str,
        force: bool = False,
        inactive_for_seconds: int | None = None,
    ) -> SharedAllocationRelease:
        """Release one recorded shared allocation by hand: copy its work back, or discard it.

        For a workflow run that crashed and won't be resumed; a running run
        releases its own. ``mode`` is ``copy_back`` (reconnect, copy the work
        back to the source workspace as a completed run would, then dispose)
        or ``discard`` (dispose without copying back, as recovery does).

        Refused (``SharedAllocationReleaseRefused``, before anything changes)
        while this process runs the workflow run or holds the allocation; while
        the run looks like it is running elsewhere (a step session runs under
        a live owner, or it journaled something within
        ``inactive_for_seconds``, 900 by default), unless ``force``; when its
        environment isn't registered here; and for a copy-back of an
        allocation that can't be reconnected or holds no work. The record is
        fenced as ``reaping`` before disposal, so a run that resumes later
        starts a fresh allocation instead of reattaching. Raises ``KeyError``
        when no such allocation is recorded, which includes one already
        released.
        """
        from cayu.environments.shared_allocations import SharedAllocationReleaseMode
        from cayu.runtime._shared_allocation_operations import (
            inactivity_window,
            release_shared_allocation,
        )

        if type(allocation_id) is not str or not allocation_id.strip():
            raise ValueError("allocation_id must be a non-blank string.")
        try:
            resolved_mode = SharedAllocationReleaseMode(mode)
        except ValueError:
            raise ValueError("mode must be 'copy_back' or 'discard'.") from None
        if type(force) is not bool:
            raise TypeError("force must be a bool.")
        return await release_shared_allocation(
            self._environment_lifecycle.shared_allocations,
            allocation_id,
            session_store=self.session_store,
            redact=self.redact_json,
            now=self._clock(),
            mode=resolved_mode,
            force=force,
            inactive_for_seconds=inactivity_window(inactive_for_seconds),
        )


__all__ = ["WorkflowEntrances"]
