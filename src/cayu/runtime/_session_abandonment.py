"""Abandon sessions stopped on unknown tool outcomes, and their durable record."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from cayu._validation import copy_durable_metadata
from cayu.events import EventType
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.sessions._completion_finalization import (
    PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY,
)
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._pending_tool_round_reader import pending_tool_round_from_checkpoint
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.queries import SessionOrder, SessionQuery
from cayu.sessions.records import Session, SessionStatus, SessionStatusConflict
from cayu.sessions.recovery import (
    _ABANDONED_SESSION_CHECKPOINT_KEY,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
)
from cayu.tasks.queries import TaskOrder, TaskQuery
from cayu.tasks.records import _TERMINAL_TASK_STATUSES

if TYPE_CHECKING:
    from cayu.runtime._environment_lifecycle import EnvironmentLifecycle
    from cayu.runtime._pending_tool_round_recovery import RegisteredEnvironmentResolver
    from cayu.runtime._session_recovery import SessionRecovery
    from cayu.sessions.store import SessionStore
    from cayu.tasks.store import TaskStore

# Retains the abandoned round and the calls whose outcome stays unknown. Its
# presence on a failed session tells recovery to release the bound environment
# and to publish this reason as the session's terminal failure.
ABANDONED_SESSION_CHECKPOINT_KEY = _ABANDONED_SESSION_CHECKPOINT_KEY


def abandoned_session_record(
    *,
    reason: str,
    metadata: dict[str, Any],
    tool_round: dict[str, Any],
    unknown_tool_calls: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "reason": reason,
        "metadata": metadata,
        "tool_round": tool_round,
        "unknown_tool_calls": unknown_tool_calls,
    }


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
    ):
        raise RuntimeError("Session retains malformed abandonment evidence.")
    return record


def abandoned_session_failure_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Terminal ``session.failed`` payload for an abandoned session."""

    return {
        "error": "Session was abandoned with unknown tool outcomes.",
        "error_type": "SessionAbandoned",
        "abandoned": True,
        "reason": record["reason"],
        "metadata": copy_durable_metadata(record["metadata"]),
        "tool_effect_outcome_unknown": [dict(call) for call in record["unknown_tool_calls"]],
        "recovered": True,
    }


logger = logging.getLogger(__name__)


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
    ) -> None:
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
        pending_round = pending_tool_round_from_checkpoint(
            await self._session_store.load_checkpoint(session_id)
        )
        if not await self.tasks_are_terminal(
            session_id, task_id=None if pending_round is None else pending_round.task_id
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
            if prior is not None and not await self.tasks_are_terminal(
                session_id, require_task=True
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
        if (
            checkpoint is None
            or checkpoint.get("pending_tool_approval") is not None
            or checkpoint.get("pending_user_input") is not None
        ):
            return False
        pending_round = pending_tool_round_from_checkpoint(checkpoint)
        if pending_round is None or not await self.tasks_are_terminal(
            session.id, require_task=True, task_id=pending_round.task_id
        ):
            return False
        effects = ToolEffectStateOwner(self._session_store)
        for call in pending_round.tool_calls:
            record = await effects.resolve_call(
                session,
                tool_round_id=pending_round.tool_round_id,
                tool_call_id=call.tool_call_id,
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
        if task_id is not None:
            task = await self._task_store.load_task(task_id)
            if task is not None and task.status not in _TERMINAL_TASK_STATUSES:
                return False
        found = False
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
