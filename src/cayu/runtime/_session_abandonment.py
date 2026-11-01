"""Abandon sessions stopped on unknown tool outcomes, and their durable record."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid5

from cayu._validation import copy_durable_metadata
from cayu.approvals.actors import ResolutionActor, resolution_actor_payload
from cayu.approvals.user_input import pending_user_input_from_checkpoint
from cayu.approvals.wakes import PENDING_WAKE_CHECKPOINT_KEY
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.runtime._event_projection import PUBLIC_EVENT_ID_PREFIX
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.sessions._abandonment_checkpoint import ABANDONED_SESSION_CHECKPOINT_KEY
from cayu.sessions._completion_finalization import (
    PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY,
)
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._foreground_child_checkpoint import FOREGROUND_CHILD_WAIT_KEY
from cayu.sessions._pending_approval_reader import pending_approval_from_checkpoint
from cayu.sessions._pending_tool_round_reader import pending_tool_round_from_checkpoint
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.queries import SessionOrder, SessionQuery
from cayu.sessions.records import Session, SessionStatus, SessionStatusConflict
from cayu.sessions.recovery import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
    IncompleteSessionsRecoveryRequest,
)
from cayu.tasks.queries import TaskOrder, TaskQuery
from cayu.tasks.records import _TERMINAL_TASK_STATUSES

if TYPE_CHECKING:
    from cayu.runtime._environment_lifecycle import EnvironmentLifecycle
    from cayu.runtime._incomplete_session_recovery import IncompleteSessionRecovery
    from cayu.runtime._pending_tool_round_recovery import RegisteredEnvironmentResolver
    from cayu.runtime._session_recovery import SessionRecovery
    from cayu.sessions.store import SessionStore
    from cayu.tasks.store import TaskStore


# The abandonment reason a session closed by a person fails with; the person and
# their reason are its metadata.
SESSION_CLOSED_REASON = "session_closed"
STRANDED_APPROVAL_DENIAL_REASON = "tool_approval_denied_after_profile_change"
# The session's execution deadline passed while its approval waited, so the
# denied round could not continue either.
EXPIRED_APPROVAL_DENIAL_REASON = "tool_approval_denied_after_deadline"
_STRANDED_DENIAL_ERRORS = {
    STRANDED_APPROVAL_DENIAL_REASON: (
        "The pending approval was denied after the application's execution profile "
        "changed. The session ended without continuing, because the paused round "
        "could not continue under the changed profile."
    ),
    EXPIRED_APPROVAL_DENIAL_REASON: (
        "The pending approval was denied after the session's execution deadline "
        "passed. The session ended without continuing, because the paused round "
        "could not continue past its deadline."
    ),
}


def stranded_approval_failure_event(session: Session, record: dict[str, Any]) -> Event:
    """Use one terminal event for direct denial publication and crash recovery."""
    denial = record["denied_approval"]
    return event_with_runtime_payload_authority(
        event_with_runtime_envelope_authority(
            event_with_runtime_generated_id(
                Event(
                    id=str(
                        uuid5(
                            UUID("951440b9-3a08-4b67-b436-b5310cc4c832"),
                            f"{session.id}\0failed\0{denial['approval_id']}",
                        )
                    ),
                    type=EventType.SESSION_FAILED,
                    session_id=session.id,
                    timestamp=datetime.fromisoformat(denial["denied_at"]),
                    agent_name=session.agent_name,
                    payload=abandoned_session_failure_payload(record),
                )
            ),
            "session_id",
        ),
        "approval_id",
        "tool_call_id",
        "tool_round_id",
    )


def abandoned_session_record(
    *,
    reason: str,
    metadata: dict[str, Any],
    tool_round: dict[str, Any],
    unknown_tool_calls: list[dict[str, str]],
    denied_approval: dict[str, Any] | None = None,
    closed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        "reason": reason,
        "metadata": metadata,
        "tool_round": tool_round,
        "unknown_tool_calls": unknown_tool_calls,
    }
    if denied_approval is not None:
        # A person denied the round's approval, but the round could not continue:
        # the application's execution profile had changed, or the session's
        # execution deadline had passed (``reason`` says which).
        record["denied_approval"] = denied_approval
    if closed is not None:
        # A person closed the session while it waited; names what it waited on.
        record["closed"] = closed
    return record


def abandoned_session_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    record = checkpoint.get(ABANDONED_SESSION_CHECKPOINT_KEY)
    if record is None:
        return None
    if (
        type(record) is not dict
        or record.get("schema_version") != 1
        or type(record.get("reason")) is not str
        or type(record.get("metadata")) is not dict
        or type(record.get("tool_round")) is not dict
        or type(record.get("unknown_tool_calls")) is not list
        or any(type(call) is not dict for call in record["unknown_tool_calls"])
        or ("denied_approval" in record and not _valid_denied_approval(record["denied_approval"]))
        or ("closed" in record and not _valid_closed(record["closed"]))
    ):
        raise RuntimeError("Session retains malformed abandonment evidence.")
    return record


_DENIED_APPROVAL_STRING_FIELDS = (
    "approval_id",
    "tool_round_id",
    "tool_call_id",
    "tool_name",
    "model_step_id",
    "model_attempt_id",
    "resolution_request_digest",
    "denied_at",
)


_CLOSED_FIELDS = frozenset({"tool_round_id", "approval_id", "input_id"})


def _valid_closed(closed: Any) -> bool:
    return (
        type(closed) is dict
        and set(closed) <= _CLOSED_FIELDS
        and all(type(value) is str for value in closed.values())
    )


def _valid_denied_approval(denial: Any) -> bool:
    return (
        type(denial) is dict
        and all(type(denial.get(field)) is str for field in _DENIED_APPROVAL_STRING_FIELDS)
        and (denial.get("reason") is None or type(denial.get("reason")) is str)
        and (denial.get("resolved_by") is None or type(denial.get("resolved_by")) is dict)
        and type(denial.get("result")) is dict
    )


def stranded_approval_denial(
    checkpoint: dict[str, Any] | None, approval_id: str
) -> dict[str, Any] | None:
    """The recorded denial of ``approval_id`` that ended its session, if any."""

    record = abandoned_session_from_checkpoint(checkpoint)
    denial = None if record is None else record.get("denied_approval")
    if denial is None or denial["approval_id"] != approval_id:
        return None
    return denial


def abandoned_session_failure_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Terminal ``session.failed`` payload for an abandoned session."""

    if "denied_approval" in record:
        denial = record["denied_approval"]
        return {
            "error": _STRANDED_DENIAL_ERRORS.get(
                record["reason"], _STRANDED_DENIAL_ERRORS[STRANDED_APPROVAL_DENIAL_REASON]
            ),
            "error_type": "SessionAbandoned",
            "abandoned": True,
            "reason": record["reason"],
            "metadata": copy_durable_metadata(record["metadata"]),
            "approval_id": denial["approval_id"],
            "tool_round_id": denial["tool_round_id"],
            "tool_call_id": denial["tool_call_id"],
            "tool_name": denial["tool_name"],
            "tool_effect_outcome_unknown": [],
            "recovered": True,
        }
    if "closed" in record:
        return {
            "error": "Session was closed without continuing.",
            "error_type": "SessionAbandoned",
            "abandoned": True,
            "closed": True,
            "reason": record["reason"],
            "metadata": copy_durable_metadata(record["metadata"]),
            **record["closed"],
            "tool_effect_outcome_unknown": [dict(call) for call in record["unknown_tool_calls"]],
            "recovered": True,
        }
    return {
        "error": "Session was abandoned with unknown tool outcomes.",
        "error_type": "SessionAbandoned",
        "abandoned": True,
        "reason": record["reason"],
        "metadata": copy_durable_metadata(record["metadata"]),
        "tool_effect_outcome_unknown": [dict(call) for call in record["unknown_tool_calls"]],
        "recovered": True,
    }


def _abandonable_tool_round(
    checkpoint: dict[str, Any] | None,
) -> tuple[str, tuple[str, ...], str | None] | None:
    """The round abandonment would end, as its ID, tool call IDs and attached task.

    An approval or user-input gate holds its round while the approved or
    answered calls run, so an unknown outcome can be retained under either.
    """

    held = (
        pending_approval_from_checkpoint(checkpoint)
        or pending_user_input_from_checkpoint(checkpoint)
        or pending_tool_round_from_checkpoint(checkpoint)
    )
    if held is None:
        return None
    return (
        held.tool_round_id,
        tuple(call.tool_call_id for call in held.tool_calls),
        held.task_id,
    )


logger = logging.getLogger(__name__)
# A child in one of these statuses still runs, so its parent can't be closed.
_CLOSE_BLOCKING_STATUSES = frozenset({SessionStatus.RUNNING, SessionStatus.INTERRUPTING})


def _merged_recovery_results(
    results: list[IncompleteSessionRecoveryResult],
) -> IncompleteSessionRecoveryResult:
    """Report consecutive recovery passes of one session as one result."""

    if len(results) == 1:
        return results[0]
    last = results[-1]
    message = last.message
    if IncompleteSessionRecoveryAction.FAILED not in last.actions:
        message = next(
            (
                result.message
                for result in results
                if IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT in result.actions
                or IncompleteSessionRecoveryAction.CLOSED in result.actions
            ),
            message,
        )
    return last.model_copy(
        update={
            "previous_status": results[0].previous_status,
            "actions": tuple(action for result in results for action in result.actions),
            "events": tuple(event for result in results for event in result.events),
            "message": message,
        }
    )


class SessionAbandonment:
    """End sessions stopped on unknown tool outcomes through ordinary recovery."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        task_store: TaskStore | None,
        session_recovery: SessionRecovery,
        environment_lifecycle: EnvironmentLifecycle,
        resolve_registered_environment: RegisteredEnvironmentResolver,
        incomplete_recovery: IncompleteSessionRecovery,
        agent_registered: Callable[[str], bool],
        resolve_action_linkage: Callable[..., Awaitable[str]],
    ) -> None:
        self._resolve_action_linkage = resolve_action_linkage
        self._incomplete_recovery = incomplete_recovery
        self._agent_registered = agent_registered
        self._session_store = session_store
        self._task_store = task_store
        self._session_recovery = session_recovery
        self._environment_lifecycle = environment_lifecycle
        self._resolve_registered_environment = resolve_registered_environment

    async def abandon(
        self,
        session_id: str,
        *,
        reason: str,
        metadata: dict[str, Any],
        inactive_for_seconds: int,
        prior: IncompleteSessionRecoveryResult | None = None,
    ) -> IncompleteSessionRecoveryResult:
        request = IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_for_seconds=inactive_for_seconds,
            reason=reason,
            metadata=metadata,
        )
        session = await self._session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        pending_round = _abandonable_tool_round(
            await self._session_store.load_checkpoint(session_id)
        )
        if not await self.tasks_are_terminal(
            session_id, task_id=None if pending_round is None else pending_round[2]
        ):
            raise SessionStatusConflict(
                "Only a session whose tasks are all terminal can be abandoned."
            )
        abandoned = (
            session.status is SessionStatus.FAILED
            and abandoned_session_from_checkpoint(
                await self._session_store.load_checkpoint(session_id)
            )
            is not None
        )
        passes = [] if prior is None else [prior]
        if not abandoned:
            passes.append(
                await self._session_recovery.recover_incomplete_session(
                    request, abandon_unknown_tool_effects=True
                )
            )
            if IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT not in passes[-1].actions:
                return _merged_recovery_results(passes)
        # A second, ordinary pass publishes the terminal evidence and finalizes
        # the environment binding of the now-failed session. Admission refuses
        # an abandoned session, so nothing can have continued it in between.
        current = await self._session_store.load(session_id)
        if (
            current is None
            or current.instance_id != session.instance_id
            or current.status is not SessionStatus.FAILED
            or abandoned_session_from_checkpoint(
                await self._session_store.load_checkpoint(session_id)
            )
            is None
        ):
            raise SessionStatusConflict("Abandoned session changed before its cleanup.")
        passes.append(
            await self._session_recovery.recover_incomplete_session(
                request.model_copy(update={"inactive_for_seconds": 0})
            )
        )
        return _merged_recovery_results(passes)

    async def close(
        self,
        session_id: str,
        *,
        reason: str,
        closed_by: ResolutionActor | None,
        waiting_on: str | None = None,
        _from_parent: bool = False,
    ) -> IncompleteSessionRecoveryResult:
        """End a non-running session, including one waiting on a person, as abandoned.

        A session stopped on an unknown tool outcome is abandoned as by ``abandon``.
        Any other pending or interrupted session is failed by
        ``IncompleteSessionRecovery.close_waiting_session``, after its pending or
        interrupted child sessions are closed the same way. A second, ordinary
        recovery pass then publishes the terminal evidence and releases the
        environment binding. Repeating the call after a close finishes that cleanup.
        """

        if type(reason) is not str or not reason.strip():
            raise ValueError("A reason is required to close a session.")
        if closed_by is not None and type(closed_by) is not ResolutionActor:
            raise TypeError("closed_by must be a ResolutionActor.")
        if waiting_on is not None and (type(waiting_on) is not str or not waiting_on):
            raise ValueError("waiting_on must be a non-empty approval or input ID.")
        metadata = {"reason": reason, "resolved_by": resolution_actor_payload(closed_by)}
        session = await self._session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        checkpoint = await self._session_store.load_checkpoint(session_id)
        record = abandoned_session_from_checkpoint(checkpoint)
        if session.status is SessionStatus.FAILED and record is not None:
            if waiting_on is not None and waiting_on not in (record.get("closed") or {}).values():
                raise SessionStatusConflict("Session ended without that approval or question.")
            return await self._finish(session, metadata=metadata, passes=[])
        if session.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
            raise SessionStatusConflict(
                f"Session already {session.status.value}; there is nothing to close."
            )
        if waiting_on is not None and waiting_on.startswith(PUBLIC_EVENT_ID_PREFIX):
            # The pending-action API shows these as public aliases.
            waiting_on = await self._resolve_action_linkage(
                session_id=session_id,
                value=waiting_on,
                field_name="approval_id" if waiting_on.endswith(":approval_id") else "input_id",
            )
        # Every refusal, the children's included, comes before anything is closed.
        await self._require_closable(
            session, checkpoint, from_parent=_from_parent, waiting_on=waiting_on
        )
        passes: list[IncompleteSessionRecoveryResult] = []
        if session.status in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
            # Only a session whose process died can be closed; recovery takes it
            # over (skipping a live owner) and leaves it interrupted. A session that
            # is already pending or interrupted is closed as it is: its continuation
            # may be refused (for example after its execution profile changed), and
            # closing never continues it.
            passes.append(
                await self._session_recovery.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id=session_id,
                        inactive_for_seconds=0,
                        reason=SESSION_CLOSED_REASON,
                        metadata=metadata,
                    ),
                    abandon_unknown_tool_effects=True,
                )
            )
        if not passes or (
            IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT not in passes[-1].actions
        ):
            current = await self._session_store.load(session_id)
            if current is None or current.instance_id != session.instance_id:
                raise SessionStatusConflict("Session changed before it was closed.")
            if current.status in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
                raise SessionStatusConflict(
                    f"Session is {current.status.value}; interrupt it first, then close it."
                )
            if current.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
                # Recovery found the session had already ended.
                return _merged_recovery_results(passes)
            closed = await self._incomplete_recovery.close_waiting_session(
                session_id, reason=SESSION_CLOSED_REASON, metadata=metadata, waiting_on=waiting_on
            )
            # The parent closes first, so a refused parent never closes its children;
            # each child was checked above and is closable now that its parent ended.
            for child in await self._open_children(session_id):
                await self.close(child.id, reason=reason, closed_by=closed_by, _from_parent=True)
            passes.append(
                IncompleteSessionRecoveryResult(
                    session_id=session_id,
                    previous_status=current.status,
                    status=closed.status,
                    actions=(IncompleteSessionRecoveryAction.CLOSED,),
                    message="Closed the session; it can no longer continue.",
                )
            )
        return await self._finish(session, metadata=metadata, passes=passes)

    async def _open_children(self, session_id: str) -> list[Session]:
        page = await self._session_store.list_sessions(
            SessionQuery(parent_session_id=session_id, limit=1000)
        )
        return [
            child
            for child in page.sessions
            if child.status not in {SessionStatus.COMPLETED, SessionStatus.FAILED}
        ]

    async def _require_closable(
        self,
        session: Session,
        checkpoint: dict[str, Any] | None,
        *,
        from_parent: bool,
        waiting_on: str | None = None,
    ) -> None:
        """Refuse a close that couldn't finish, checking every open descendant too."""

        if PENDING_WAKE_CHECKPOINT_KEY in (checkpoint or {}):
            raise SessionStatusConflict(
                "Session is asleep until its wake; interrupt it first, which cancels the "
                "wake, then close it."
            )
        pending_round = _abandonable_tool_round(checkpoint)
        if not await self.tasks_are_terminal(
            session.id, task_id=None if pending_round is None else pending_round[2]
        ):
            raise SessionStatusConflict(
                "Session has a task that is not finished; cancel it first "
                "(`cayu task cancel`), then close the session."
            )
        if not from_parent and await self._foreground_parent_waits(session):
            raise SessionStatusConflict(
                "This subagent's parent session waits on it; close the parent session "
                "instead, which closes this one too."
            )
        if not self._agent_registered(session.agent_name):
            # Its terminal evidence and environment are settled through its agent.
            raise SessionStatusConflict(
                f"The session's agent {session.agent_name!r} isn't registered in this "
                "application; close it from the application that runs that agent."
            )
        if session.status in {SessionStatus.PENDING, SessionStatus.INTERRUPTED}:
            # The pause itself (and, for a child, that it still waits) is checked
            # without changing anything.
            await self._incomplete_recovery.close_waiting_session(
                session.id,
                reason=SESSION_CLOSED_REASON,
                metadata={},
                waiting_on=waiting_on,
                check_only=True,
            )
        for child in await self._open_children(session.id):
            if child.status in _CLOSE_BLOCKING_STATUSES:
                raise SessionStatusConflict(
                    "Session has child sessions that are still running; interrupt them first."
                )
            await self._require_closable(
                child, await self._session_store.load_checkpoint(child.id), from_parent=True
            )

    async def _foreground_parent_waits(self, session: Session) -> bool:
        """Whether a parent that hasn't ended waits on this session as a foreground child."""

        if session.parent_session_id is None:
            return False
        parent = await self._session_store.load(session.parent_session_id)
        if parent is None or parent.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
            return False
        wait = (await self._session_store.load_checkpoint(parent.id) or {}).get(
            FOREGROUND_CHILD_WAIT_KEY
        )
        return isinstance(wait, dict) and wait.get("child_session_id") == session.id

    async def _finish(
        self,
        session: Session,
        *,
        metadata: dict[str, Any],
        passes: list[IncompleteSessionRecoveryResult],
    ) -> IncompleteSessionRecoveryResult:
        """Publish an abandoned session's terminal evidence and release its environment."""

        current = await self._session_store.load(session.id)
        if (
            current is None
            or current.instance_id != session.instance_id
            or current.status is not SessionStatus.FAILED
            or abandoned_session_from_checkpoint(
                await self._session_store.load_checkpoint(session.id)
            )
            is None
        ):
            raise SessionStatusConflict("Closed session changed before its cleanup.")
        passes.append(
            await self._session_recovery.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session.id,
                    inactive_for_seconds=0,
                    reason=SESSION_CLOSED_REASON,
                    metadata=metadata,
                )
            )
        )
        return _merged_recovery_results(passes)

    async def recover_abandoned(
        self, *, inactive_for_seconds: int, abandon_sessions_of_terminal_tasks: bool
    ) -> list[IncompleteSessionRecoveryResult]:
        """Run ``CayuApp.recover_abandoned_sessions``' sweep, unprojected."""

        if type(inactive_for_seconds) is not int or inactive_for_seconds < 0:
            raise ValueError("inactive_for_seconds must be a non-negative integer.")
        if type(abandon_sessions_of_terminal_tasks) is not bool:
            raise TypeError("abandon_sessions_of_terminal_tasks must be a bool.")
        if abandon_sessions_of_terminal_tasks and self._task_store is None:
            raise ValueError("abandon_sessions_of_terminal_tasks requires a task_store.")
        results: list[IncompleteSessionRecoveryResult] = []
        cursor: str | None = None
        while True:
            page = await self._session_recovery.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(
                    statuses={SessionStatus.RUNNING, SessionStatus.INTERRUPTING},
                    inactive_for_seconds=inactive_for_seconds,
                    cursor=cursor,
                    reason="abandoned_session_recovered",
                ),
                recover_abandoned_models=True,
            )
            for result in page.results:
                if abandon_sessions_of_terminal_tasks:
                    result = await self.abandon_swept(result)
                results.append(result)
            if page.next_cursor is None:
                if abandon_sessions_of_terminal_tasks:
                    results.extend(
                        await self.abandon_retained(inactive_for_seconds=inactive_for_seconds)
                    )
                return results
            if page.next_cursor == cursor:
                raise RuntimeError("Abandoned-session recovery returned a repeated cursor.")
            cursor = page.next_cursor

    async def abandon_swept(
        self, result: IncompleteSessionRecoveryResult
    ) -> IncompleteSessionRecoveryResult:
        """Abandon one sweep result left on an unknown outcome of finished tasks."""

        if IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT not in result.actions:
            return result
        # The preceding recovery already claimed inactivity and refreshed it.
        return await self._abandon_isolated(
            result.session_id, result.status, inactive_for_seconds=0, prior=result
        )

    async def _abandon_isolated(
        self,
        session_id: str,
        status: SessionStatus,
        *,
        inactive_for_seconds: int,
        prior: IncompleteSessionRecoveryResult | None = None,
    ) -> IncompleteSessionRecoveryResult:
        """Abandon one swept session; a failure is reported, not raised.

        A later sweep revisits the session, so one failure cannot strand the
        allocations of the sessions after it.
        """

        try:
            if prior is not None:
                pending_round = _abandonable_tool_round(
                    await self._session_store.load_checkpoint(session_id)
                )
                if not await self.tasks_are_terminal(
                    session_id,
                    require_task=True,
                    task_id=None if pending_round is None else pending_round[2],
                ):
                    return prior
            return await self.abandon(
                session_id,
                reason="abandoned_session_recovered",
                metadata={},
                inactive_for_seconds=inactive_for_seconds,
                prior=prior,
            )
        except Exception as exc:
            logger.warning(
                "Session abandonment failed for session %s: error_type=%s",
                session_id,
                type(exc).__name__,
            )
            failed = IncompleteSessionRecoveryResult(
                session_id=session_id,
                previous_status=status,
                status=status,
                actions=(IncompleteSessionRecoveryAction.FAILED,),
                message=f"Session abandonment failed: {type(exc).__name__}.",
            )
            return failed if prior is None else _merged_recovery_results([prior, failed])

    async def abandon_retained(
        self, *, inactive_for_seconds: int
    ) -> list[IncompleteSessionRecoveryResult]:
        """Revisit sessions an earlier sweep or abandonment left unfinished.

        Selects interrupted sessions held by an unknown tool outcome whose
        tasks have since become terminal, and failed abandoned sessions whose
        terminal evidence or environment cleanup did not finish.
        """

        results: list[IncompleteSessionRecoveryResult] = []
        for status in (SessionStatus.INTERRUPTED, SessionStatus.FAILED):
            cursor: str | None = None
            while True:
                page = await self._session_store.list_sessions(
                    SessionQuery(
                        status=status,
                        inactive_for_seconds=inactive_for_seconds,
                        order_by=SessionOrder.CREATED_AT_ASC,
                        limit=1000,
                        cursor=cursor,
                    )
                )
                for session in page.sessions:
                    try:
                        retained = await self._retained(session)
                    except Exception as exc:
                        logger.warning(
                            "Abandonment inspection failed for session %s: error_type=%s",
                            session.id,
                            type(exc).__name__,
                        )
                        continue
                    if retained:
                        results.append(
                            await self._abandon_isolated(
                                session.id,
                                session.status,
                                inactive_for_seconds=inactive_for_seconds,
                            )
                        )
                if page.next_cursor is None or page.next_cursor == cursor:
                    break
                cursor = page.next_cursor
        return results

    async def _retained(self, session: Session) -> bool:
        checkpoint = await self._session_store.load_checkpoint(session.id)
        if session.status is SessionStatus.FAILED:
            if abandoned_session_from_checkpoint(checkpoint) is None:
                return False
            assert checkpoint is not None
            if PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY in checkpoint:
                return True
            if checkpoint.get("environment_factory_reconnect"):
                active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
                if active_profile is not None and (
                    await self._environment_lifecycle.prepare_terminal_binding_finalization(
                        session=session,
                        execution_profile=active_profile.profile,
                        registered_environment=self._resolve_registered_environment(
                            session.environment_name
                        ),
                    )
                    is not None
                ):
                    return True
            failures = await self._session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    event_type=EventType.SESSION_FAILED,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=1,
                )
            )
            return not failures or failures[0].event.payload.get("error_type") != (
                "SessionAbandoned"
            )
        pending_round = _abandonable_tool_round(checkpoint)
        if pending_round is None:
            return False
        tool_round_id, tool_calls, task_id = pending_round
        if not await self.tasks_are_terminal(session.id, require_task=True, task_id=task_id):
            return False
        effects = ToolEffectStateOwner(self._session_store)
        for tool_call_id in tool_calls:
            record = await effects.resolve_call(
                session,
                tool_round_id=tool_round_id,
                tool_call_id=tool_call_id,
            )
            if record is not None and record.state in {"executing", "outcome_unknown"}:
                return True
        return False

    async def tasks_are_terminal(
        self, session_id: str, *, require_task: bool = False, task_id: str | None = None
    ) -> bool:
        """Whether every task of the session is terminal, optionally requiring one.

        ``task_id`` names a task the session's durable state is attached to;
        without a task store its status cannot be verified.
        """

        if self._task_store is None:
            if task_id is not None:
                raise SessionStatusConflict(
                    "Abandoning a session attached to a task requires its task store."
                )
            return not require_task
        found = False
        if task_id is not None:
            task = await self._task_store.load_task(task_id)
            if task is not None:
                if task.status not in _TERMINAL_TASK_STATUSES:
                    return False
                # A restart detaches its task from this session, so the query
                # below no longer finds it; the attached task still counts.
                found = True
        offset = 0
        while True:
            page = await self._task_store.list_tasks(
                TaskQuery(
                    session_id=session_id,
                    limit=1000,
                    offset=offset,
                    order_by=TaskOrder.CREATED_AT_ASC,
                )
            )
            for task in page:
                if task.status not in _TERMINAL_TASK_STATUSES:
                    return False
                found = True
            if len(page) < 1000:
                return found or not require_task
            offset += len(page)
