"""Content-free notification references over Runtime-owned pending actions.

This is a read projection, not a pause or delivery state machine. Callers own
recipient authorization and delivery; references cannot authorize execution.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cayu import EventType
from cayu.approvals.user_input import (
    USER_INPUT_SUPERSESSION_INTENT_KEY,
    UserInputSupersessionIntent,
)
from cayu.sessions.base import (
    EventQuery,
    PendingActionKind,
    PendingActionQuery,
    PendingActionRecord,
    SessionStore,
)
from cayu.tools.base import ToolResult

AttentionKind = Literal["user_input", "tool_approval", "manual_recovery"]
AttentionState = Literal["active", "resolved", "cancelled", "superseded", "expired", "unavailable"]

AttentionReason = Literal[
    "current_action",
    "durable_closure",
    "durable_supersession",
    "session_unavailable",
    "session_replaced",
    "incomplete_query",
    "no_terminal_evidence",
    "store_unavailable",
]


class HumanAttentionReference(BaseModel):
    """Trusted SDK correlation data, not a bearer credential or decision receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    attention_id: str = Field(pattern=r"^attention_[0-9a-f]{64}$")
    session_id: str = Field(min_length=1, max_length=2048)
    session_instance_id: str = Field(min_length=1, max_length=512)
    kind: AttentionKind
    action_id: str = Field(min_length=1, max_length=4096)
    round_id: str | None = Field(default=None, max_length=512)
    tool_call_id: str | None = Field(default=None, max_length=512)
    source_sequence: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_identity(self):
        identity = [
            self.session_id,
            self.session_instance_id,
            self.kind,
            self.action_id,
            self.tool_call_id if self.kind == "manual_recovery" else None,
        ]
        expected = (
            "attention_"
            + hashlib.sha256(
                json.dumps(identity, ensure_ascii=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        if self.attention_id != expected:
            raise ValueError("Attention identity does not match its exact reference.")
        return self


class HumanAttentionRequest(BaseModel):
    """Safe default summary. Detailed human disclosure remains policy-controlled."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    reference: HumanAttentionReference
    summary: Literal["User input required.", "Tool approval required.", "Manual recovery required."]

    @classmethod
    def from_pending_action(cls, action: PendingActionRecord) -> HumanAttentionRequest | None:
        """Project canonical query results; delegated rows are navigation only."""
        if action.attention_id is None or action.session.instance_id is None:
            return None
        if action.kind is PendingActionKind.USER_INPUT:
            action_id, summary = action.input_id, "User input required."
        elif action.kind is PendingActionKind.TOOL_APPROVAL:
            action_id, summary = action.approval_id, "Tool approval required."
        elif action.kind is PendingActionKind.MANUAL_RECOVERY:
            action_id, summary = action.round_id or action.id, "Manual recovery required."
        else:
            return None
        if action_id is None:
            return None
        return cls(
            reference=HumanAttentionReference(
                attention_id=action.attention_id,
                session_id=action.session.id,
                session_instance_id=action.session.instance_id,
                kind=action.kind.value,
                action_id=action_id,
                round_id=action.round_id,
                tool_call_id=action.tool_call_id,
                source_sequence=action.event.sequence,
            ),
            summary=summary,
        )


class HumanAttentionObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    attention_id: str
    state: AttentionState
    observed_at: datetime
    reason: AttentionReason


async def observe_human_attention(
    store: SessionStore,
    reference: HumanAttentionReference,
) -> HumanAttentionObservation:
    """Read current authority or exact terminal evidence, never infer closure from absence.

    State may change immediately after observation. Inspection is bounded and
    deliberately returns unavailable when retained evidence cannot prove a result.
    No health/notification event, claim, or delivery mutation is made here.
    """
    reference = HumanAttentionReference.model_validate(reference)
    now = datetime.now(UTC)

    def result(state: AttentionState, reason: AttentionReason) -> HumanAttentionObservation:
        return HumanAttentionObservation(
            attention_id=reference.attention_id, state=state, observed_at=now, reason=reason
        )

    try:
        session = await store.load(reference.session_id)
        if session is None:
            return result("unavailable", "session_unavailable")
        if session.instance_id != reference.session_instance_id:
            return result("unavailable", "session_replaced")
        pending = await store.query_pending_actions(PendingActionQuery(session_id=session.id))
        if pending.issues or pending.has_more:
            return result("unavailable", "incomplete_query")
        candidate = next(
            (action for action in pending.actions if action.attention_id == reference.attention_id),
            None,
        )
        if candidate is not None:
            projected = HumanAttentionRequest.from_pending_action(candidate)
            # Do not let a forged attention ID mask different exact-action fields.
            if projected is None or any(
                getattr(projected.reference, name) != getattr(reference, name)
                for name in ("session_instance_id", "kind", "action_id", "round_id", "tool_call_id")
            ):
                return result("unavailable", "no_terminal_evidence")
            state, reason = "active", "current_action"
        else:
            state, reason = await _terminal_state(store, reference)
        # Deletion/recreation between reads must not borrow another incarnation's evidence.
        current = await store.load(reference.session_id)
        if current is None:
            return result("unavailable", "session_unavailable")
        if current.instance_id != reference.session_instance_id:
            return result("unavailable", "session_replaced")
        return result(state, reason)
    except Exception:
        # A temporarily unavailable canonical store is not proof of resolution.
        # Exception text may contain secrets; expose a fixed reason only.
        return result("unavailable", "store_unavailable")


async def _terminal_state(
    store: SessionStore, ref: HumanAttentionReference
) -> tuple[AttentionState, AttentionReason]:
    if ref.kind == "user_input":
        receipt = await store.load_runtime_publication_receipt(
            ref.session_id, f"user-input-close:{ref.action_id}"
        )
        if (
            receipt is not None
            and receipt.kind == "user-input-close"
            and receipt.intent.get("input_id") == ref.action_id
        ):
            return "resolved", "durable_closure"
        events = await store.load_user_input_supersession_events(ref.session_id, ref.action_id)
        if len(events) == 1:
            intent = UserInputSupersessionIntent.model_validate(
                events[0].payload.get(USER_INPUT_SUPERSESSION_INTENT_KEY)
            )
            if (
                intent.session_instance_id == ref.session_instance_id
                and intent.input_id == ref.action_id
            ):
                return "superseded", "durable_supersession"
    elif ref.kind == "tool_approval":
        receipt = await store.load_runtime_publication_receipt(
            ref.session_id, f"approval-close:{ref.action_id}"
        )
        if (
            receipt is not None
            and receipt.kind == "approval-close"
            and receipt.intent.get("approval_id") == ref.action_id
        ):
            # Expiry is a Runtime decision, not a notification-service clock rule.
            # It gates the first grant; an in-window grant can recover after expiry.
            expired = await store.query_events(
                EventQuery(
                    session_id=ref.session_id,
                    event_type=EventType.TOOL_CALL_APPROVAL_EXPIRED,
                    after_sequence=ref.source_sequence,
                    limit=100,
                )
            )
            if any(row.event.payload.get("approval_id") == ref.action_id for row in expired):
                return "expired", "durable_closure"
            if len(expired) == 100:
                return "unavailable", "incomplete_query"
            decision = receipt.intent.get("decision")
            if decision == "approve":
                return "resolved", "durable_closure"
            if decision == "deny":
                return "cancelled", "durable_closure"
    else:
        rows = await store.query_events(
            EventQuery(
                session_id=ref.session_id,
                after_sequence=ref.source_sequence,
                event_types=(
                    EventType.TOOL_CALL_COMPLETED,
                    EventType.TOOL_CALL_BLOCKED,
                    EventType.TOOL_CALL_FAILED,
                ),
                limit=100,
            )
        )
        for row in rows:
            payload = row.event.payload
            if (
                ref.round_id is not None
                and ref.tool_call_id is not None
                and payload.get("tool_round_id") == ref.round_id
                and payload.get("tool_call_id") == ref.tool_call_id
            ):
                if row.event.type == EventType.TOOL_CALL_COMPLETED:
                    return "resolved", "durable_closure"
                if (
                    row.event.type == EventType.TOOL_CALL_FAILED
                    and payload.get("manual_recovery") is True
                ):
                    result = ToolResult.model_validate(payload.get("result"))
                    # A committed operator failure settles the attention request,
                    # but synthetic outcome-unknown failures do not prove closure.
                    if result.is_error and all(
                        controls.get("outcome_unknown", False) is False
                        and controls.get("manual_reconciliation_required", False) is False
                        for controls in (payload, result.structured or {})
                    ):
                        return "resolved", "durable_closure"
                if row.event.type == EventType.TOOL_CALL_BLOCKED:
                    return "cancelled", "durable_closure"
        if len(rows) == 100:
            return "unavailable", "incomplete_query"
    return "unavailable", "no_terminal_evidence"
