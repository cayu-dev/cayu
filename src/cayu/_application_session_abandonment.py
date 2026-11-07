"""``CayuApp`` entrances that end sessions nobody will continue.

``recover_abandoned_sessions`` and ``reconcile_orphaned_sessions`` sweep sessions a
dead process left behind; ``abandon_session`` and ``close_session`` end one session.
All of them go through ``SessionAbandonment``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cayu.runtime.application_lifecycle import _admitted_entrance
from cayu.sessions.execution import execution_owned_by

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.approvals.actors import ResolutionActor
    from cayu.sessions.recovery import IncompleteSessionRecoveryResult


class SessionAbandonmentEntrances:
    @_admitted_entrance
    @execution_owned_by("recovery")
    async def recover_abandoned_sessions(
        self: CayuApp,
        *,
        inactive_for_seconds: int = 3600,
        abandon_sessions_of_terminal_tasks: bool = False,
    ) -> tuple[IncompleteSessionRecoveryResult, ...]:
        """Recover running sessions whose process died and that nothing has resumed since.

        A continuation recovers its own session when it takes it over, so this
        is for conversations an application never returns to: run it at startup
        or on a schedule. It uses paginated incomplete-session recovery for
        running and interrupting sessions inactive for ``inactive_for_seconds``,
        which skips sessions with a live execution owner. Ordinary lost model
        calls are settled as unknown without redispatch, including their budget
        reservations; provider operations and contracted tasks keep their own
        recovery paths. Keep the window longer than the gap after which a
        conversation's next message would still take
        its session over, so that message's takeover replays interrupted calls.

        A session stopped on an unknown external tool outcome is left
        ``interrupted`` with ``pending_tool_effect``. With
        ``abandon_sessions_of_terminal_tasks``, such a session is abandoned
        as by :meth:`abandon_session` when it has at least one task and every
        one of its tasks is terminal, which requires a ``task_store``. The
        sweep then also revisits interrupted sessions whose tasks finished
        later, and abandoned sessions whose cleanup did not finish. Every sweep
        finishes a ``close_session`` that stopped before closing the children,
        and every completed turn whose process stopped before it completed its
        task, published its terminal events or released its session.
        """

        return tuple(
            [
                await self._project_incomplete_recovery_result_for_public_api(result)
                for result in await self._session_abandonment.recover_abandoned(
                    inactive_for_seconds=inactive_for_seconds,
                    abandon_sessions_of_terminal_tasks=abandon_sessions_of_terminal_tasks,
                )
            ]
        )

    @_admitted_entrance
    @execution_owned_by("recovery")
    async def reconcile_orphaned_sessions(
        self: CayuApp, *, inactive_for_seconds: int = 3600
    ) -> tuple[IncompleteSessionRecoveryResult, ...]:
        """End the runs a lost worker left holding sessions ``running``.

        A session stays ``running`` when the worker that ran it dies, for example
        when a deploy replaces it. This reconciles each ``running`` or
        ``interrupting`` session inactive for ``inactive_for_seconds`` whose owner is
        provably gone: the store records its owner's execution lease as lapsed or its
        process as dead, or the session has tasks and every one is terminal. Such a
        session is recovered as :meth:`recover_abandoned_sessions` recovers it, left
        ``interrupted`` (with ``pending_tool_effect``, an inbox recovery item, when a
        tool call it was running has an unknown outcome). Where its environment isn't
        registered, as for a per-run environment only its worker registered, recovery
        can't take it over, so it fails as abandoned instead (``session.failed`` with
        ``error_type: "SessionAbandoned"`` and reason ``orphaned_session_reconciled``,
        listing any unknown tool calls, and the assistant model call it was making in
        ``model_call_outcome_unknown``); its environment binding stays recorded.

        A session with a live owner, an unfinished task, or activity within the
        window is left alone, as is one without attested ownership and tasks. The
        server runs this on a schedule (``ServerLifecycleConfig``); call it yourself
        when you run Cayu without the server. Results list only the sessions it
        acted on.
        """

        return tuple(
            [
                await self._project_incomplete_recovery_result_for_public_api(result)
                for result in await self._session_abandonment.reconcile_orphaned(
                    inactive_for_seconds=inactive_for_seconds
                )
            ]
        )

    @_admitted_entrance
    @execution_owned_by("recovery")
    async def abandon_session(
        self: CayuApp,
        session_id: str,
        *,
        reason: str = "session_abandoned",
        metadata: dict[str, Any] | None = None,
        inactive_for_seconds: int = 0,
    ) -> IncompleteSessionRecoveryResult:
        """End a session stopped on an unknown tool outcome and release its environment.

        Use this when the application will never continue the session, for
        example after cancelling its work across a restart. The session may be
        ``interrupted`` with ``pending_tool_effect`` or not yet recovered after
        its process died. Each unknown external effect keeps its
        ``outcome_unknown`` record and is never dispatched again; the pending
        round is retained as evidence but leaves resumable work. The session
        then fails with a ``session.failed`` event whose payload has
        ``error_type: "SessionAbandoned"`` and lists the unknown calls, and
        recovery finalizes its environment binding through the factory. No
        model call is made.

        The result reports ``abandoned_tool_effect`` on success. A session
        with no unknown tool outcome is recovered normally and left as it
        was. Refuses while any task of the session is not terminal, and skips a
        session with a live execution owner or activity within
        ``inactive_for_seconds``. Repeating the call after an interruption
        finishes the cleanup. An abandoned session can no longer be resumed;
        fork it to continue its conversation.
        """

        if type(inactive_for_seconds) is not int or inactive_for_seconds < 0:
            raise ValueError("inactive_for_seconds must be a non-negative integer.")
        result = await self._session_abandonment.abandon(
            await self._resolve_public_session_id(session_id),
            reason=reason,
            metadata={} if metadata is None else metadata,
            inactive_for_seconds=inactive_for_seconds,
        )
        return await self._project_incomplete_recovery_result_for_public_api(result)

    @_admitted_entrance
    @execution_owned_by("recovery")
    async def close_session(
        self: CayuApp,
        session_id: str,
        *,
        reason: str,
        closed_by: ResolutionActor | None = None,
        waiting_on: str | None = None,
    ) -> IncompleteSessionRecoveryResult:
        """End a pending or interrupted session nobody will continue, keeping its history.

        A ``running`` session is closed only when its owner is provably gone (a
        lapsed execution lease, a process that died, or every task of it finished),
        including where its environment isn't registered; a live one is refused.
        ``waiting_on`` is an approval or input ID the session must still wait on.
        ``docs/runtime-contracts.md`` says what it records, closes and refuses.
        """
        result = await self._session_abandonment.close(
            await self._resolve_public_session_id(session_id),
            reason=reason,
            closed_by=closed_by,
            waiting_on=waiting_on,
        )
        return await self._project_incomplete_recovery_result_for_public_api(result)
