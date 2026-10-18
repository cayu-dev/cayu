"""Settle positively unconsumed preparations, without invoking external code."""

from __future__ import annotations

from hashlib import sha256

from cayu._validation import canonical_durable_json_bytes
from cayu.approvals.tools import PendingToolApproval
from cayu.approvals.user_input import PendingUserInput
from cayu.events import Event, EventType
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.execution_units import ToolRoundIdentity
from cayu.runtime._approval_support import (
    approved_arguments_for_call,
    tool_call_request_from_pending,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._tool_effect_state import (
    ToolEffectRecord,
    ToolEffectStateOwner,
    ToolEffectTerminal,
)
from cayu.runtime._tool_execution import (
    APPROVED_ARGUMENTS_TERMINAL_FIELD,
    with_approved_arguments_notice,
)
from cayu.runtime._tool_invocation.context import (
    _event_with_targeted_tool_invocation_authority,
    _targeted_tool_invocation_payload,
)
from cayu.runtime._tool_round_staging import _event_with_tool_round_authority
from cayu.runtime.tool_effects import ToolEffectConflict
from cayu.sessions._pending_approval_reader import approval_resolution_intent_from_checkpoint
from cayu.sessions._pending_tool_round import PendingToolRound
from cayu.sessions.records import Session
from cayu.sessions.store import SessionStore
from cayu.tools._argument_publication import unavailable_argument_projection
from cayu.tools.base import ToolResult


def requires_explicit_effect_continuation(
    record: ToolEffectRecord, pending: PendingToolRound
) -> bool:
    """Receipt consumption and targeted grants still require a continuation owner."""
    if record.terminal is None:
        return False
    if record.terminal.receipt is not None:
        return True
    return record.dispatch_id is None and any(
        call.targeted_tool_grant_id is not None
        or call.targeted_tool_invocation is not None
        or call.targeted_tool_rejection is not None
        for call in pending.tool_calls
    )


async def settle_prepared_tool_effects(
    *,
    store: SessionStore,
    writer: RuntimeEventWriter,
    session: Session,
    pending: PendingToolRound | PendingToolApproval | PendingUserInput,
    profile: ExecutionProfileIdentity | None,
) -> list[Event]:
    """Use the same exact CAS as dispatch; leave pause/round consumption to its owner."""
    owner = ToolEffectStateOwner(store)
    events: list[Event] = []
    # The model learns a person changed the gating call even when it never ran.
    resolution_intent = (
        approval_resolution_intent_from_checkpoint(await store.load_checkpoint(session.id))
        if isinstance(pending, PendingToolApproval)
        else None
    )
    identity = ToolRoundIdentity(
        model_step_id=pending.model_step_id,
        model_attempt_id=pending.model_attempt_id,
        tool_round_id=pending.tool_round_id,
    )
    for call in pending.tool_calls:
        record = await owner.resolve_call(
            session, tool_round_id=pending.tool_round_id, tool_call_id=call.tool_call_id
        )
        if record is None or record.state != "prepared":
            continue
        intent = record.intent
        tool_call = tool_call_request_from_pending(call)
        targeted = _targeted_tool_invocation_payload(tool_call)
        targeted_digest = (
            sha256(canonical_durable_json_bytes(targeted, "effect_targeted_invocation")).hexdigest()
            if targeted
            else None
        )
        if (
            profile is None
            or intent.execution_profile_fingerprint != profile.fingerprint
            or intent.tool_name != call.tool_name
            or intent.targeted_invocation_digest != targeted_digest
            or intent.agent_name != pending.agent_name
            or intent.environment_name != pending.environment_name
            or any(getattr(intent, key) != value for key, value in identity.payload().items())
            or (
                isinstance(pending, PendingToolApproval)
                and intent.approval_id != pending.approval_id
            )
            or intent.pause_id
            != (pending.input_id if isinstance(pending, PendingUserInput) else None)
        ):
            raise ToolEffectConflict("Prepared recovery lost its exact invocation authority.")
        event = Event(
            type=EventType.TOOL_CALL_FAILED,
            session_id=session.id,
            interaction_id=intent.interaction_id,
            agent_name=intent.agent_name,
            environment_name=intent.environment_name,
            tool_name=intent.tool_name,
            payload={
                **identity.payload(),
                **targeted,
                **unavailable_argument_projection().payload_fields(),
                "tool_call_id": intent.tool_call_id,
                "idempotency_key": intent.idempotency_key,
                **({"approval_id": intent.approval_id} if intent.approval_id else {}),
                **({"input_id": intent.pause_id} if intent.pause_id else {}),
                "recovered": True,
                **(
                    {APPROVED_ARGUMENTS_TERMINAL_FIELD: True}
                    if approved_arguments_for_call(resolution_intent, call.tool_call_id) is not None
                    else {}
                ),
                "result": with_approved_arguments_notice(
                    ToolResult(
                        content="Tool was not invoked before interrupted preparation was recovered.",
                        is_error=True,
                        structured={
                            "recovery_reason": "tool_effect_not_dispatched",
                            "executed": False,
                            "outcome_unknown": False,
                        },
                    ),
                    approved=approved_arguments_for_call(resolution_intent, call.tool_call_id)
                    is not None,
                ).model_dump(mode="json"),
            },
        )
        event = writer.prepare(
            event_with_execution_profile_authority(
                _event_with_targeted_tool_invocation_authority(
                    _event_with_tool_round_authority(event, identity, "approval_id", "input_id"),
                    tool_call,
                ),
                profile,
            )
        )
        await owner.transition(
            record,
            state="failed",
            run_epoch=session.run_epoch,
            terminal=ToolEffectTerminal(
                event_id=event.id,
                result_digest=sha256(
                    canonical_durable_json_bytes(event.payload["result"], "undispatched_result")
                ).hexdigest(),
            ),
            events=(event,),
        )
        events.extend(await writer.fan_out_persisted([event]))
    return events
