"""Positive completion proof for receipt-only recovery after deadline expiry."""

from __future__ import annotations

from typing import Any, Literal

from cayu.runtime._tool_effect_state import ToolEffectConflict, ToolEffectRecord
from cayu.sessions._runtime_publication import SessionOperationPublication
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.store import SessionStore


def _completion(
    record: ToolEffectRecord, phase: Literal["settlement", "tool-hooks"]
) -> tuple[str, dict[str, Any]]:
    terminal = record.terminal
    if terminal is None or terminal.reconciliation_request_digest is None:
        raise ToolEffectConflict("Deadline settlement requires an exact reconciled terminal.")
    key = f"tool-effect-deadline-{phase}:" + terminal.event_id
    return key, {
        "schema_version": 1,
        "session_instance_id": record.intent.session_instance_id,
        "tool_round_id": record.intent.tool_round_id,
        "tool_call_id": record.intent.tool_call_id,
        "terminal_event_id": terminal.event_id,
        "request_digest": terminal.reconciliation_request_digest,
    }


async def deadline_effect_settlement_completed(
    store: SessionStore,
    session: Session,
    record: ToolEffectRecord,
    *,
    phase: Literal["settlement", "tool-hooks"] = "settlement",
) -> bool:
    """Read completion without admitting another environment, hook or task mutation."""
    if phase == "settlement" and not session.execution_deadline.expired:
        return False
    key, expected = _completion(record, phase)
    observed = await store.load_session_operation(session.id, key)
    if observed is not None and observed != expected:
        raise ToolEffectConflict("Deadline settlement has conflicting completion evidence.")
    return observed is not None


async def complete_deadline_effect_settlement(
    store: SessionStore,
    session: Session,
    record: ToolEffectRecord,
    *,
    phase: Literal["settlement", "tool-hooks"] = "settlement",
) -> None:
    """Publish a completed phase, with final settlement requiring interruption."""
    key, completion = _completion(record, phase)

    def transform(current: Session, checkpoint: dict[str, Any] | None, previous: dict | None):
        if current.instance_id != record.intent.session_instance_id:
            raise ToolEffectConflict("Deadline settlement changed session incarnation.")
        if previous is not None and previous != completion:
            raise ToolEffectConflict("Deadline settlement has conflicting completion evidence.")
        return SessionOperationPublication(
            checkpoint={} if checkpoint is None else checkpoint,
            operation_records={key: completion},
        )

    await store.publish_session_operation(
        session.id,
        idempotency_key=key,
        operation_transform=transform,
        events=[],
        expected_statuses=(
            {SessionStatus.INTERRUPTED}
            if phase == "settlement"
            else {SessionStatus.RUNNING, SessionStatus.INTERRUPTED}
        ),
        expected_run_epoch=session.run_epoch,
    )
