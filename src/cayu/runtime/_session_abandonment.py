"""Abandon sessions stopped on unknown tool outcomes, and their durable record."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
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
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
)
from cayu.sessions.invocation import SessionExecutionSource
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
# The abandonment reason a running session whose owner was provably gone fails with
# when its environment isn't registered where it was reconciled.
ORPHANED_SESSION_RECONCILED_REASON = "orphaned_session_reconciled"
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
    model_call: dict[str, Any] | None = None,
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
    if model_call is not None:
        # The assistant model call the lost run was making, abandoned with it: its
        # stage, and whether it reached the provider (its outcome is then unknown).
        record["model_call"] = model_call
    return record


def _unknown_model_calls(record: dict[str, Any]) -> list[dict[str, str]]:
    call = record.get("model_call")
    if call is None or not call["dispatched"]:
        return []
    return [{key: call[key] for key in _MODEL_CALL_STRING_FIELDS}]


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
        or ("model_call" in record and not _valid_model_call(record["model_call"]))
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
_MODEL_CALL_STRING_FIELDS = ("stage_id", "model_step_id", "model_attempt_id", "provider_name")


def _valid_model_call(call: Any) -> bool:
    return (
        type(call) is dict
        and set(call) == {*_MODEL_CALL_STRING_FIELDS, "dispatched"}
        and all(type(call[field]) is str for field in _MODEL_CALL_STRING_FIELDS)
        and type(call["dispatched"]) is bool
    )


def _unresumed_close(
    session_id: str, action: IncompleteSessionRecoveryAction, message: str
) -> IncompleteSessionRecoveryResult:
    return IncompleteSessionRecoveryResult(
        session_id=session_id,
        previous_status=SessionStatus.FAILED,
        status=SessionStatus.FAILED,
        actions=(action,),
        message=message,
    )


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
            **_model_call_payload(record),
            "recovered": True,
        }
    return {
        "error": "Session was abandoned with unknown tool outcomes.",
        "error_type": "SessionAbandoned",
        "abandoned": True,
        "reason": record["reason"],
        "metadata": copy_durable_metadata(record["metadata"]),
        "tool_effect_outcome_unknown": [dict(call) for call in record["unknown_tool_calls"]],
        **_model_call_payload(record),
        "recovered": True,
    }


def _model_call_payload(record: dict[str, Any]) -> dict[str, Any]:
    """``model_call_outcome_unknown`` for a run abandoned inside a model call."""

    # Absent otherwise, so abandonments recorded before it kept their payload.
    unknown = _unknown_model_calls(record)
    return {"model_call_outcome_unknown": unknown} if unknown else {}


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
# How far back, before its inactivity window, the reconciliation sweep looks for an
# abandonment that failed a session but stopped before publishing its evidence.
_UNFINISHED_ABANDONMENT_LOOKBACK = timedelta(days=7)
_OPEN_STATUSES = (SessionStatus.PENDING, SessionStatus.INTERRUPTED)


class _ClosedSessionChildRunning(SessionStatusConflict):
    """A closed session's remaining child runs again, so none of them is closed yet."""


def _close_record(checkpoint: dict[str, Any] | None) -> dict[str, Any] | None:
    """The abandonment record of a session ``close`` ended, if it has one."""

    record = abandoned_session_from_checkpoint(checkpoint)
    if record is None or record["reason"] != SESSION_CLOSED_REASON or "closed" not in record:
        return None
    return record


def _recorded_close(record: dict[str, Any]) -> tuple[str, ResolutionActor | None]:
    """The reason and person a close recorded, to close its children with."""

    metadata = record["metadata"]
    reason = metadata.get("reason")
    actor = metadata.get("resolved_by")
    return (
        reason if type(reason) is str and reason.strip() else "Parent closed.",
        ResolutionActor.model_validate(actor) if type(actor) is dict else None,
    )


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
            try:
                passes.append(
                    await self._session_recovery.recover_incomplete_session(
                        request, abandon_unknown_tool_effects=True
                    )
                )
            except KeyError:
                # Recovery reconnects the session's environment, so it can't run here
                # without a registration of it. Abandoning continues nothing and
                # needs no environment.
                if not await self._environment_unregistered(session_id):
                    raise
                passes.append(
                    await self._abandon_without_environment(
                        session,
                        reason=reason,
                        metadata=metadata,
                        inactive_for_seconds=inactive_for_seconds,
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

    async def _environment_unregistered(self, session_id: str) -> bool:
        """Whether the session exists and names an environment this application lacks."""

        session = await self._session_store.load(session_id)
        if session is None or session.environment_name is None:
            return False
        try:
            self._resolve_registered_environment(session.environment_name)
        except KeyError:
            return True
        return False

    async def _abandon_without_environment(
        self,
        session: Session,
        *,
        reason: str,
        metadata: dict[str, Any],
        inactive_for_seconds: int,
    ) -> IncompleteSessionRecoveryResult:
        """Abandon a session whose environment ordinary recovery couldn't resolve here.

        ``session`` is the session as read before ordinary recovery, which refused a
        live owner or active work in this process before it needed the environment.
        Child sessions other than forks that haven't ended are refused, since
        without its environment the parent's wait on them can't be checked; see
        ``IncompleteSessionRecovery.abandon_without_environment`` for the rest.
        """

        if not self._agent_registered(session.agent_name):
            raise SessionStatusConflict(
                f"The session's agent {session.agent_name!r} isn't registered in this "
                "application; abandon it from the application that runs that agent."
            )
        if any(
            # A fork is an independent run; nothing waits on it.
            child.invocation.source is not SessionExecutionSource.FORK
            for child in await self._open_children(session.id)
        ):
            raise SessionStatusConflict(
                f"The session's environment {session.environment_name!r} isn't registered "
                "in this application, and the session has child sessions that haven't "
                "ended; end them first."
            )
        abandoned = await self._incomplete_recovery.abandon_without_environment(
            session,
            reason=reason,
            metadata=metadata,
            inactive_for_seconds=inactive_for_seconds,
        )
        return IncompleteSessionRecoveryResult(
            session_id=session.id,
            previous_status=SessionStatus.INTERRUPTED,
            status=abandoned.status,
            actions=(IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT,),
            message=(
                "Abandoned the session; its unknown tool outcomes were not dispatched "
                f"again. Its environment {session.environment_name!r} isn't registered in "
                "this application, so its binding is left to be released where it is."
            ),
        )

    async def owner_gone(self, session: Session) -> bool:
        """Whether the run holding a running session has provably ended.

        It has when the store attests its owner is gone (the execution lease lapsed
        or the owner's process died), or, for a store or run without such an owner,
        when the session has tasks and every one of them is terminal. A live owner,
        or the session running in this process, keeps it.
        """

        owner = await self._incomplete_recovery.execution_owner(session)
        if owner == "lost":
            return True
        if owner == "live":
            return False
        return await self.tasks_are_terminal(session.id, require_task=True)

    async def _abandon_orphan_without_environment(
        self,
        session: Session,
        *,
        reason: str,
        metadata: dict[str, Any],
        inactive_for_seconds: int,
        closed: dict[str, str] | None,
    ) -> IncompleteSessionRecoveryResult:
        """Fail a running session whose owner is gone, where its environment isn't.

        ``session`` is the session as read before ordinary recovery, which refused a
        live owner or active work in this process before it needed the environment.
        The caller has proved its owner gone (``owner_gone``) and checked its agent is
        registered. Child sessions other than forks that haven't ended, and a subagent
        whose parent waits on it, are refused; see
        ``IncompleteSessionRecovery.abandon_orphaned_without_environment`` for the rest.
        """

        if any(
            child.invocation.source is not SessionExecutionSource.FORK
            for child in await self._open_children(session.id)
        ):
            raise SessionStatusConflict(
                f"The session's environment {session.environment_name!r} isn't registered "
                "in this application, and the session has child sessions that haven't "
                "ended; end them first."
            )
        if await self._foreground_parent_waits(session):
            raise SessionStatusConflict(
                "This subagent's parent session waits on it; end the parent session instead."
            )
        abandoned = await self._incomplete_recovery.abandon_orphaned_without_environment(
            session,
            reason=reason,
            metadata=metadata,
            inactive_for_seconds=inactive_for_seconds,
            closed=closed,
        )
        # No recovery owner reconstructed the run, so nothing closed its interaction.
        await self._session_recovery.settle_released_open_interaction(session.id)
        record = abandoned_session_from_checkpoint(
            await self._session_store.load_checkpoint(session.id)
        )
        unknown = [] if record is None else record["unknown_tool_calls"]
        unknown_model = [] if record is None else _unknown_model_calls(record)
        return IncompleteSessionRecoveryResult(
            session_id=session.id,
            previous_status=session.status,
            status=abandoned.status,
            actions=(
                IncompleteSessionRecoveryAction.CLOSED
                if closed is not None
                else IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT
                if unknown
                else IncompleteSessionRecoveryAction.ABANDONED_ORPHANED_RUN,
            ),
            message=(
                ("Closed" if closed is not None else "Failed")
                + " the session its lost worker left running; it can no longer continue."
                + (
                    f" {len(unknown)} tool call(s) it was running have unknown outcomes and "
                    "were not dispatched again."
                    if unknown
                    else ""
                )
                + (
                    " The model call it was making has an unknown outcome and was not made again."
                    if unknown_model
                    else ""
                )
                + f" Its environment {session.environment_name!r} isn't registered in this "
                "application, so its binding is left to be released where it is."
            ),
        )

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
        environment binding. Repeating the call after a close finishes that cleanup,
        and so does ``resume_stopped_closes``, which the abandonment sweep runs.
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
            # A repeated close replays. The ID may be the private one the close
            # recorded, including a legacy private ID that looks like an alias, or the
            # public alias the pending-action API showed for it.
            closed_ids = set((record.get("closed") or {}).values())
            if (
                waiting_on is not None
                and waiting_on not in closed_ids
                and await self._private_waiting_on(session_id, waiting_on) not in closed_ids
            ):
                raise SessionStatusConflict("Session ended without that approval or question.")
            if "closed" in record:
                # A close fails the parent before its children, so one that stopped in
                # between left children open; repeating it closes the rest.
                await self._close_remaining_children(session_id, record)
            return await self._finish(session, metadata=metadata, passes=[])
        if waiting_on is not None:
            waiting_on = await self._private_waiting_on(session_id, waiting_on)
        if session.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
            raise SessionStatusConflict(
                f"Session already {session.status.value}; there is nothing to close."
            )
        # Every refusal, the children's included, comes before anything is closed.
        await self._require_closable(
            session, checkpoint, from_parent=_from_parent, waiting_on=waiting_on
        )
        passes: list[IncompleteSessionRecoveryResult] = []
        if session.status in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
            # Only a session whose owner is provably gone can be closed; recovery takes
            # it over (skipping a live owner) and leaves it interrupted. A session that
            # is already pending or interrupted is closed as it is: its continuation
            # may be refused (for example after its execution profile changed), and
            # closing never continues it.
            if not await self.owner_gone(session):
                raise SessionStatusConflict(
                    f"Session is {session.status.value}, and nothing shows its owner is gone "
                    "(a lapsed execution lease, a process that died, or a finished task); "
                    "interrupt it first, then close it."
                )
            try:
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
            except KeyError:
                # Recovery reconnects the session's environment, which a per-run one
                # (registered only by the worker that ran it) never is here. Closing
                # continues nothing and needs no environment.
                if not await self._environment_unregistered(session_id):
                    raise
                if waiting_on is not None:
                    # A running session waits on no approval or question.
                    raise SessionStatusConflict(
                        "Session no longer waits on that approval or question."
                    ) from None
                return await self._abandon_orphan_without_environment(
                    session,
                    reason=SESSION_CLOSED_REASON,
                    metadata=metadata,
                    inactive_for_seconds=0,
                    closed={},
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
            try:
                closed = await self._incomplete_recovery.close_waiting_session(
                    session_id,
                    reason=SESSION_CLOSED_REASON,
                    metadata=metadata,
                    waiting_on=waiting_on,
                )
            except SessionStatusConflict:
                # Another close got there first, such as the sweep finishing a stopped
                # close while a person repeats it. Its close stands; replay it.
                current = await self._session_store.load(session_id)
                if (
                    current is None
                    or current.instance_id != session.instance_id
                    or current.status is not SessionStatus.FAILED
                    or _close_record(await self._session_store.load_checkpoint(session_id)) is None
                ):
                    raise
                return await self.close(
                    session_id,
                    reason=reason,
                    closed_by=closed_by,
                    waiting_on=waiting_on,
                    _from_parent=_from_parent,
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

    async def _close_remaining_children(self, session_id: str, record: dict[str, Any]) -> None:
        """Close the open children a closed session left, as its own close would have.

        They are closed with the reason and person the parent's close recorded. A child
        that runs again is refused before any other child is closed.
        """

        remaining = await self._remaining_close_descendants(session_id, record)
        if any(child.status in _CLOSE_BLOCKING_STATUSES for child, _ in remaining):
            raise _ClosedSessionChildRunning(
                "The session is closed, but a child session is still running; interrupt "
                "it, then repeat the close to close the rest."
            )
        for child, _ in remaining:
            if child.status is SessionStatus.FAILED:
                continue
            await self._require_closable(
                child, await self._session_store.load_checkpoint(child.id), from_parent=True
            )
        for child, close_record in remaining:
            # The full tree was preflighted above. Recurse only through direct
            # children so each descendant and finalization is executed once.
            if child.parent_session_id != session_id:
                continue
            reason, closed_by = _recorded_close(close_record)
            await self.close(child.id, reason=reason, closed_by=closed_by, _from_parent=True)

    async def _remaining_close_descendants(
        self, session_id: str, record: dict[str, Any]
    ) -> list[tuple[Session, dict[str, Any]]]:
        """Find descendants needing closure or finalization after a stopped cascade."""

        pending = [(session_id, record)]
        seen = {session_id}
        remaining: list[tuple[Session, dict[str, Any]]] = []
        while pending:
            parent_id, close_record = pending.pop()
            for child in await self._open_children(parent_id, include_failed=True):
                # A fork is an independent run, even when created after the close.
                if child.invocation.source is SessionExecutionSource.FORK:
                    continue
                if child.id in seen:
                    raise SessionStatusConflict("Session ancestry contains a repeated child.")
                seen.add(child.id)
                inherited = close_record
                if child.status is SessionStatus.FAILED:
                    child_record = abandoned_session_from_checkpoint(
                        await self._session_store.load_checkpoint(child.id)
                    )
                    if child_record is None or "closed" not in child_record:
                        continue
                    inherited = child_record
                # Retain closed ancestors too: replay must publish their terminal
                # evidence and release environments after finishing descendants.
                remaining.append((child, inherited))
                pending.append((child.id, inherited))
        return remaining

    async def _private_waiting_on(self, session_id: str, waiting_on: str) -> str:
        """The private approval or input ID for ``waiting_on``, which may be a public alias.

        The pending-action API shows these as ``cayu_event_…`` aliases; the resolver
        reads the alias's own event, so it still works after the pause is gone.
        """

        if not waiting_on.startswith(PUBLIC_EVENT_ID_PREFIX):
            return waiting_on
        return await self._resolve_action_linkage(
            session_id=session_id,
            value=waiting_on,
            field_name="approval_id" if waiting_on.endswith(":approval_id") else "input_id",
        )

    async def _open_children(
        self, session_id: str, *, include_failed: bool = False
    ) -> list[Session]:
        """List children across every page, optionally retaining failed ancestors."""

        children: list[Session] = []
        cursor: str | None = None
        while True:
            page = await self._session_store.list_sessions(
                SessionQuery(
                    parent_session_id=session_id,
                    order_by=SessionOrder.CREATED_AT_ASC,
                    limit=1000,
                    cursor=cursor,
                )
            )
            children.extend(
                child
                for child in page.sessions
                if child.status is not SessionStatus.COMPLETED
                and (include_failed or child.status is not SessionStatus.FAILED)
            )
            if page.next_cursor is None:
                return children
            if page.next_cursor == cursor:
                raise RuntimeError("Child session listing returned a repeated cursor.")
            cursor = page.next_cursor

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
        record = abandoned_session_from_checkpoint(
            await self._session_store.load_checkpoint(session.id)
        )
        if (
            current is None
            or current.instance_id != session.instance_id
            or current.status is not SessionStatus.FAILED
            or record is None
        ):
            raise SessionStatusConflict("Closed session changed before its cleanup.")
        if "model_call" in record:
            # The model call the run was abandoned inside settles with its interaction,
            # which ordinary recovery would otherwise leave to manual model recovery.
            # An abandonment that died before it released its invocation is released
            # first.
            await self._incomplete_recovery.release_abandoned_model_call_invocation(session.id)
            await self._session_recovery.settle_released_open_interaction(session.id)
        try:
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
        except KeyError:
            # Recovery repairs missing terminal evidence without the environment, so
            # what remains needs it: releasing the binding or an allocation the run
            # started, which happens where the environment is registered.
            if not await self._environment_unregistered(session.id):
                raise
            passes.append(
                _unresumed_close(
                    session.id,
                    IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,
                    "The session is closed. Its environment "
                    f"{session.environment_name!r} isn't registered in this application, "
                    "so its binding is left to be released where it is.",
                )
            )
        if await self._environment_unregistered(session.id):
            # No recovery owner reconstructed the run, so nothing closed its
            # interaction; one an abandonment that stopped part way left open is too.
            await self._session_recovery.settle_released_open_interaction(session.id)
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
                results.extend(
                    await self.resume_stopped_closes(inactive_for_seconds=inactive_for_seconds)
                )
                return results
            if page.next_cursor == cursor:
                raise RuntimeError("Abandoned-session recovery returned a repeated cursor.")
            cursor = page.next_cursor

    async def reconcile_orphaned(
        self, *, inactive_for_seconds: int
    ) -> list[IncompleteSessionRecoveryResult]:
        """Run ``CayuApp.reconcile_orphaned_sessions``' sweep, unprojected.

        Each running or interrupting session inactive for ``inactive_for_seconds``
        whose owner is provably gone (``owner_gone``) and whose tasks are all terminal
        is recovered as ``recover_abandoned_sessions`` would: interrupted, with
        ``pending_tool_effect`` when a tool call it was running has an unknown outcome.
        Where its environment isn't registered, so recovery can't take it over, it
        fails as abandoned instead (``abandon_orphaned_without_environment``). Any
        other session is left alone and not reported. It then finishes abandonments
        that failed a session but stopped before its terminal evidence
        (``_finish_unfinished_abandonments``). One session's failure is reported, not
        raised, so it can't hold up the rest.
        """

        if type(inactive_for_seconds) is not int or inactive_for_seconds < 0:
            raise ValueError("inactive_for_seconds must be a non-negative integer.")
        results: list[IncompleteSessionRecoveryResult] = []
        for status in (SessionStatus.RUNNING, SessionStatus.INTERRUPTING):
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
                    result = await self._reconcile_orphaned_isolated(
                        session, inactive_for_seconds=inactive_for_seconds
                    )
                    if result is not None:
                        results.append(result)
                if page.next_cursor is None or page.next_cursor == cursor:
                    break
                cursor = page.next_cursor
        results.extend(await self._finish_unfinished_abandonments(inactive_for_seconds))
        return results

    async def _finish_unfinished_abandonments(
        self, inactive_for_seconds: int
    ) -> list[IncompleteSessionRecoveryResult]:
        """Finish abandonments that failed a session but stopped before its evidence.

        An abandonment commits ``failed`` before it publishes ``session.failed`` and
        settles the run's interaction, so a process that died in between, or a
        publication that failed, leaves a failed session no running-session sweep
        selects again. This revisits failed sessions that ended within
        ``_UNFINISHED_ABANDONMENT_LOOKBACK`` before the inactivity window, newest
        first, and finishes each abandoned one still missing that evidence.
        """

        results: list[IncompleteSessionRecoveryResult] = []
        cutoff = datetime.now(UTC) - timedelta(seconds=inactive_for_seconds)
        cutoff -= _UNFINISHED_ABANDONMENT_LOOKBACK
        cursor: str | None = None
        while True:
            page = await self._session_store.list_sessions(
                SessionQuery(
                    status=SessionStatus.FAILED,
                    inactive_for_seconds=inactive_for_seconds,
                    order_by=SessionOrder.UPDATED_AT_DESC,
                    limit=1000,
                    cursor=cursor,
                )
            )
            for session in page.sessions:
                if session.updated_at < cutoff:
                    return results
                try:
                    record = await self._unfinished_abandonment(session)
                    if record is None:
                        continue
                    results.append(
                        await self._finish(session, metadata=record["metadata"], passes=[])
                    )
                except Exception as exc:
                    logger.warning(
                        "Finishing the abandonment of session %s failed: error_type=%s",
                        session.id,
                        type(exc).__name__,
                    )
                    results.append(
                        _unresumed_close(
                            session.id,
                            IncompleteSessionRecoveryAction.FAILED,
                            f"Finishing the session's abandonment failed: {type(exc).__name__}.",
                        )
                    )
            if page.next_cursor is None or page.next_cursor == cursor:
                return results
            cursor = page.next_cursor

    async def _unfinished_abandonment(self, session: Session) -> dict[str, Any] | None:
        """The abandonment record of a failed session still missing its evidence, if any."""

        record = abandoned_session_from_checkpoint(
            await self._session_store.load_checkpoint(session.id)
        )
        if record is None:
            return None
        [failure] = await self._session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_type=EventType.SESSION_FAILED,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        ) or [None]
        if failure is None or failure.event.payload.get("error_type") != "SessionAbandoned":
            return record
        [latest] = await self._session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        ) or [None]
        if latest is not None and latest.event.type not in INTERACTION_TERMINAL_EVENT_TYPES:
            return record
        return None

    async def _reconcile_orphaned_isolated(
        self, session: Session, *, inactive_for_seconds: int
    ) -> IncompleteSessionRecoveryResult | None:
        try:
            if (
                session.runtime_name == "external_agent"
                or not self._agent_registered(session.agent_name)
                or not await self.owner_gone(session)
                or not await self.tasks_are_terminal(session.id)
            ):
                return None
            try:
                return await self._session_recovery.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id=session.id,
                        inactive_for_seconds=inactive_for_seconds,
                        reason=ORPHANED_SESSION_RECONCILED_REASON,
                    ),
                    recover_abandoned_models=True,
                )
            except KeyError:
                if not await self._environment_unregistered(session.id):
                    raise
                return await self._abandon_orphan_without_environment(
                    session,
                    reason=ORPHANED_SESSION_RECONCILED_REASON,
                    metadata={},
                    inactive_for_seconds=inactive_for_seconds,
                    closed=None,
                )
        except Exception as exc:
            # A refusal (the session moved on, or holds work this path leaves to its
            # own recovery) is expected; anything else is worth a warning.
            refused = isinstance(exc, SessionStatusConflict)
            logger.log(
                logging.INFO if refused else logging.WARNING,
                "Orphaned-session reconciliation %s session %s: error_type=%s",
                "left" if refused else "failed for",
                session.id,
                type(exc).__name__,
            )
            try:
                current = await self._session_store.load(session.id)
            except Exception:
                current = None
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=session.status,
                status=session.status if current is None else current.status,
                actions=(IncompleteSessionRecoveryAction.FAILED,),
                message=(
                    f"Orphaned-session reconciliation refused: {exc}"
                    if refused
                    else f"Orphaned-session reconciliation failed: {type(exc).__name__}."
                ),
            )

    async def resume_stopped_closes(
        self, *, inactive_for_seconds: int
    ) -> list[IncompleteSessionRecoveryResult]:
        """Finish closes that failed a session but stopped before closing its children.

        Open children discover interrupted cascades. Failed sessions also retain
        discoverable cleanup work after the last child changes status. Completed
        closes are skipped, and the root must satisfy the sweep's inactivity window.
        """

        results: list[IncompleteSessionRecoveryResult] = []
        visited: set[str] = set()
        for status in (*_OPEN_STATUSES, SessionStatus.FAILED):
            cursor: str | None = None
            while True:
                page = await self._session_store.list_sessions(
                    SessionQuery(
                        status=status,
                        order_by=SessionOrder.CREATED_AT_ASC,
                        limit=1000,
                        cursor=cursor,
                    )
                )
                for candidate in page.sessions:
                    if candidate.status is not SessionStatus.FAILED and (
                        candidate.parent_session_id is None
                        or candidate.parent_session_id in visited
                        or candidate.invocation.source is SessionExecutionSource.FORK
                    ):
                        continue
                    if candidate.status is SessionStatus.FAILED and candidate.id in visited:
                        continue
                    result = await self._resume_stopped_close(
                        candidate, visited, inactive_for_seconds=inactive_for_seconds
                    )
                    if result is not None:
                        results.append(result)
                if page.next_cursor is None or page.next_cursor == cursor:
                    break
                cursor = page.next_cursor
        return results

    async def _resume_stopped_close(
        self, child: Session, visited: set[str], *, inactive_for_seconds: int
    ) -> IncompleteSessionRecoveryResult | None:
        """Finish the close of ``child``'s closed parent; a failure is reported, not raised.

        ``visited`` collects the sessions this pass already looked at.
        """

        failed_candidate = child.status is SessionStatus.FAILED
        session_id = child.id if failed_candidate else child.parent_session_id
        assert session_id is not None
        try:
            if failed_candidate and (
                _close_record(await self._session_store.load_checkpoint(child.id)) is None
                or not await self._retained(child)
            ):
                return None
            root = await self._closed_ancestor(session_id)
            if root is None:
                # The parent wasn't closed, so its other children needn't look again.
                visited.add(session_id)
                return None
            current = await self._session_store.load(child.id)
            if (
                current is None
                or current.instance_id != child.instance_id
                or (
                    current.status is not SessionStatus.FAILED
                    if failed_candidate
                    else current.status not in _OPEN_STATUSES
                )
            ):
                # Closed (or resumed) since it was listed; a sibling may still be open.
                return None
            visited.add(session_id)
            if root[0] in visited and root[0] != session_id:
                return None
            session_id, record = root
            visited.add(session_id)
            owner = await self._session_store.load(session_id)
            if (
                owner is None
                or owner.status is not SessionStatus.FAILED
                or not self._incomplete_recovery.session_is_inactive(
                    owner, inactive_for_seconds=inactive_for_seconds
                )
            ):
                return None
            reason, closed_by = _recorded_close(record)
            result = await self.close(session_id, reason=reason, closed_by=closed_by)
        except _ClosedSessionChildRunning as exc:
            logger.info("Session close not resumed for session %s: a child is running", session_id)
            return _unresumed_close(
                session_id, IncompleteSessionRecoveryAction.SKIPPED_ACTIVE, str(exc)
            )
        except SessionRunFenced:
            # Another process, such as a person repeating the close, holds the
            # session's recovery claim and finishes the close itself.
            logger.info("Session close not resumed for session %s: another owner", session_id)
            return _unresumed_close(
                session_id,
                IncompleteSessionRecoveryAction.SKIPPED_EXECUTION_OWNER,
                "Another process is finishing the session's close.",
            )
        except Exception as exc:
            logger.warning(
                "Resuming the close of session %s failed: error_type=%s",
                session_id,
                type(exc).__name__,
            )
            return _unresumed_close(
                session_id,
                IncompleteSessionRecoveryAction.FAILED,
                f"Resuming the session's close failed: {type(exc).__name__}.",
            )
        return result.model_copy(
            update={
                "actions": (IncompleteSessionRecoveryAction.CLOSED, *result.actions),
                "message": "Finished the session's close: closed the child sessions it left open.",
            }
        )

    async def _closed_ancestor(self, session_id: str) -> tuple[str, dict[str, Any]] | None:
        """The topmost closed session in the closed chain from ``session_id`` up.

        Resuming there also finishes the closed ancestors a nested close left.
        """

        found: tuple[str, dict[str, Any]] | None = None
        seen: set[str] = set()
        current: str | None = session_id
        while current is not None and current not in seen:
            seen.add(current)
            session = await self._session_store.load(current)
            if session is None or session.status is not SessionStatus.FAILED:
                return found
            record = _close_record(await self._session_store.load_checkpoint(current))
            if record is None:
                return found
            found = (current, record)
            if session.invocation.source is SessionExecutionSource.FORK:
                return found
            current = session.parent_session_id
        return found

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
            if checkpoint.get("environment_factory_reconnect") and not (
                await self._environment_unregistered(session.id)
            ):
                # Where the environment isn't registered, nothing here can release the
                # binding; recovery where it is does.
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
