"""Evidence-only late-dispatch audit; this grants no session execution authority."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Never

from cayu._exception_groups import exception_cause
from cayu._task_wait import (
    await_shielded_task_outcome,
    restore_task_cancellation_requests,
    unexpected_child_cancellation_error,
)
from cayu.core.events import Event, EventType, copy_event
from cayu.runtime._tool_effect_state import (
    ToolEffectRecord,
    _copy_model,
    _digest,
    effect_storage_key,
)
from cayu.runtime.sessions import Session, SessionStore
from cayu.runtime.tool_effects import ToolEffectConflict


@dataclass(frozen=True, slots=True)
class ToolEffectConflictAudit:
    """Exact runtime dispatch and retained winner, never an untrusted receipt.

    The execution owner retains ``executing`` from ``begin``. Store transactions
    verify ``selected`` against the retained operation before appending evidence.
    Neither record authorizes a checkpoint write or another tool invocation.
    """

    executing: ToolEffectRecord
    selected: ToolEffectRecord

    def __post_init__(self) -> None:
        executing = _copy_model(self.executing, ToolEffectRecord)
        selected = _copy_model(self.selected, ToolEffectRecord)
        if (
            executing.state != "executing"
            or executing.revision != 1
            or selected.terminal is None
            or selected.revision <= executing.revision
            or executing.intent != selected.intent
            or executing.dispatch_id != selected.dispatch_id
        ):
            raise ToolEffectConflict("Late-dispatch audit evidence does not identify one call.")
        object.__setattr__(self, "executing", executing)
        object.__setattr__(self, "selected", selected)

    @property
    def storage_key(self) -> str:
        return effect_storage_key(self.executing.intent)

    def prepare_event(self, session: Session, current: object, *, now: datetime) -> Event:
        """Called under the transaction lock; emits no arbitrary result material."""
        intent = self.executing.intent
        if session.id != intent.session_id or session.instance_id != intent.session_instance_id:
            raise ToolEffectConflict("Late-dispatch audit lost its session incarnation.")
        if current != self.selected.model_dump(mode="json"):
            raise ToolEffectConflict("Late-dispatch audit winner changed.")
        payload = {
            "schema_version": 1,
            "kind": "late_dispatch",
            "intent_digest": _digest(intent.model_dump(mode="json")),
            "dispatch_digest": _digest(self.executing.model_dump(mode="json")),
            "winner_digest": _digest(self.selected.model_dump(mode="json")),
            "source_run_epoch": intent.source_run_epoch,
            "model_step_id": intent.model_step_id,
            "model_attempt_id": intent.model_attempt_id,
            "tool_round_id": intent.tool_round_id,
            "tool_call_id": intent.tool_call_id,
            "approval_id": intent.approval_id,
        }
        return Event(
            id="tool-effect-conflict:v1:" + _digest(payload),
            type=EventType.TOOL_EFFECT_RECONCILIATION_CONFLICT,
            timestamp=now,
            session_id=intent.session_id,
            interaction_id=intent.interaction_id,
            agent_name=intent.agent_name,
            environment_name=intent.environment_name,
            tool_name=intent.tool_name,
            payload=payload,
        )


def copy_tool_effect_conflict_audit(value: object) -> ToolEffectConflictAudit:
    if type(value) is not ToolEffectConflictAudit:
        raise TypeError("Tool effect conflict audit requires exact runtime evidence.")
    return ToolEffectConflictAudit(value.executing, value.selected)


def reconcile_tool_effect_conflict_event(expected: Event, stored: Event) -> Event:
    """Store owns first-publication time; all other event material is exact."""
    expected = copy_event(expected)
    stored = copy_event(stored)
    if expected.model_copy(update={"timestamp": stored.timestamp}) != stored:
        raise ToolEffectConflict("Late-dispatch audit identity conflicts with stored evidence.")
    return stored


async def audit_rejected_tool_dispatch(
    store: SessionStore, executing: ToolEffectRecord, *, candidate_event_id: str | None = None
) -> Event | None:
    """Resolve a winner, then let its store atomically authenticate the audit.

    A nonterminal record is not winning settlement evidence. Reads do not confer
    current-run authority and the append repeats the complete retained comparison.
    """
    executing = _copy_model(executing, ToolEffectRecord)
    current = await store.load_session_operation(
        executing.intent.session_id, effect_storage_key(executing.intent)
    )
    if current is None:
        return None
    selected = ToolEffectRecord.model_validate(current)
    if selected.intent != executing.intent or selected.dispatch_id != executing.dispatch_id:
        raise ToolEffectConflict("Rejected dispatch conflicts with retained call authority.")
    if selected.terminal is None:
        return None
    if (
        selected.state in {"completed", "failed"}
        and candidate_event_id == selected.terminal.event_id
    ):
        # This invocation already selected its own terminal. A later hook,
        # cleanup, or stream-close failure is not a competing settlement.
        return None
    return await store.append_tool_effect_conflict(ToolEffectConflictAudit(executing, selected))


async def raise_after_tool_dispatch_audit(
    store: SessionStore,
    executing: ToolEffectRecord,
    publication_failure: BaseException,
    *,
    candidate_event_id: str | None,
) -> Never:
    """Own diagnostic settlement while retaining the original control signal."""
    audit_task = asyncio.create_task(
        audit_rejected_tool_dispatch(store, executing, candidate_event_id=candidate_event_id)
    )
    outcome = await await_shielded_task_outcome(audit_task)
    audit_failure = outcome.error
    if isinstance(audit_failure, asyncio.CancelledError):
        audit_failure = unexpected_child_cancellation_error(
            audit_failure, operation="Tool effect conflict audit"
        )
    if outcome.cancellation is not None:
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
        )
    cancellation = (
        publication_failure
        if isinstance(publication_failure, asyncio.CancelledError)
        else outcome.cancellation
    )
    if cancellation is not None:
        primary_evidence = (
            exception_cause(cancellation)
            if cancellation is publication_failure
            else publication_failure
        )
        failures = [
            error
            for error in (primary_evidence, audit_failure)
            if error is not None and error is not cancellation
        ]
        if failures:
            cause = (
                failures[0]
                if len(failures) == 1
                else BaseExceptionGroup(
                    "Tool publication rejection and conflict audit failures.", failures
                )
            )
            raise cancellation from cause
        raise cancellation from exception_cause(cancellation)
    if audit_failure is not None and audit_failure is not publication_failure:
        raise BaseExceptionGroup(
            "Tool publication rejection and conflict audit failures.",
            [publication_failure, audit_failure],
        ) from None
    raise publication_failure
