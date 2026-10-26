"""CayuApp entrances that workflow code uses to raise approvals and react to them.

``CayuApp`` inherits these; they live here so the application module stays
within its size limit. Each one composes existing parts: a proposal is an
ordinary session run, and watchers are delivered by ``dispatch_event_watchers``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from cayu.observability.watchers import EventWatcher, EventWatcherRunResult
from cayu.runtime._event_watcher_delivery import dispatch_event_watchers
from cayu.runtime.application_lifecycle import _admitted_entrance

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.approvals.proposals import ToolCallProposalOutcome


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
        async with aclosing(self.run(request)) as stream:
            events = [event async for event in stream]
        return ToolCallProposalOutcome.from_events(events)


__all__ = ["WorkflowEntrances"]
