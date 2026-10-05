from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal, NamedTuple

from cayu._command_diagnostics import COMMAND_DENIAL_HINTS
from cayu._validation import (
    copy_durable_json_value,
    copy_durable_metadata,
)
from cayu.approvals.tools import (
    PendingToolApproval,
    PendingToolCallApproval,
    ResolutionActor,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
    ToolPolicyEvidence,
    pending_tool_call_for_approval_event,
    resolution_actor_payload,
)
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_nested_payload_authority,
    event_with_runtime_payload_authority,
)
from cayu.runtime import _resume_ledger as resume_ledger
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions.base import (
    Session,
    SessionStore,
    runtime_publication_checkpoint_value_digest,
)
from cayu.tools import _argument_publication as tool_argument_publication
from cayu.tools.base import ToolResult
from cayu.tools.policy import ToolPolicyDecision, ToolPolicyResult
from cayu.vaults.redaction import SecretRedactor

APPROVAL_INTERRUPT_CLOSE_INTENT_KEY = "approval_close_intent"


def approval_interrupt_close_intent(approval: PendingToolApproval) -> dict[str, str]:
    """Bind interruption closure to the exact approval and model/tool round."""
    return {
        "approval_id": approval.approval_id,
        "tool_call_id": approval.tool_call_id,
        "tool_round_id": approval.tool_round_id,
        "model_step_id": approval.model_step_id,
        "model_attempt_id": approval.model_attempt_id,
    }


BUSINESS_APPROVAL_RESOLUTION_METADATA_KEY = "cayu:business_approval"
_BUSINESS_APPROVAL_STAMP_PRIORITY_FIELDS = (
    "kind",
    "outcome",
    "approver_id",
    "approver_tier",
    "required_tier",
    "chain",
    "condition_text",
)
_APPROVAL_TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }
)
_USER_INPUT_ROUND_TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
    }
)
_APPROVAL_HISTORY_EVENT_TYPES = frozenset(
    {
        EventType.SESSION_RESUMED,
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.TOOL_CALL_APPROVAL_EXPIRED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
    }
)

_RUNTIME_APPROVAL_IDENTITY_FIELDS = (
    "approval_id",
    "model_step_id",
    "model_attempt_id",
    "tool_round_id",
)
_EXECUTION_PROFILE_FINGERPRINT_FIELD = "execution_profile_fingerprint"


def event_with_pending_approval_authority(
    event: Event,
    approval: PendingToolApproval,
) -> Event:
    """Attest approval identities from one validated runtime checkpoint model."""

    if type(approval) is not PendingToolApproval:
        raise TypeError("approval must be a PendingToolApproval.")
    top_level_fields = tuple(
        field_name
        for field_name in _RUNTIME_APPROVAL_IDENTITY_FIELDS
        if event.payload.get(field_name) == getattr(approval, field_name)
    )
    if (
        approval.execution_profile_fingerprint is not None
        and event.payload.get(_EXECUTION_PROFILE_FINGERPRINT_FIELD)
        == approval.execution_profile_fingerprint
    ):
        top_level_fields = (*top_level_fields, _EXECUTION_PROFILE_FINGERPRINT_FIELD)
    if top_level_fields:
        event = event_with_runtime_payload_authority(event, *top_level_fields)
    nested = event.payload.get("approval")
    nested_paths = tuple(
        ("approval", field_name)
        for field_name in _RUNTIME_APPROVAL_IDENTITY_FIELDS
        if type(nested) is dict and nested.get(field_name) == getattr(approval, field_name)
    )
    if nested_paths:
        event = event_with_runtime_nested_payload_authority(event, *nested_paths)
    return event


def approval_resolution_intent_for(
    approval: PendingToolApproval,
    *,
    decision: ToolApprovalDecision,
    resolution_request_digest: str | None,
    reviewed_approval_digest: str | None = None,
    pause_resolved_at: datetime | None = None,
) -> pending_approval_reader.ApprovalResolutionIntent:
    if type(approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if type(decision) is not ToolApprovalDecision:
        raise TypeError("Approval resolution decision must be a ToolApprovalDecision.")
    return pending_approval_reader.ApprovalResolutionIntent(
        approval_id=approval.approval_id,
        tool_call_id=approval.tool_call_id,
        tool_round_id=approval.tool_round_id,
        model_step_id=approval.model_step_id,
        model_attempt_id=approval.model_attempt_id,
        decision=decision,
        resolution_request_digest=resolution_request_digest,
        reviewed_approval_digest=reviewed_approval_digest,
        pause_resolved_at=pause_resolved_at,
    )


def approval_resolution_request_digest(request: ToolApprovalRequest) -> str:
    """Bind retry-visible audit input without copying it into checkpoint state."""

    if type(request) is not ToolApprovalRequest:
        raise TypeError("request must be a ToolApprovalRequest.")
    return runtime_publication_checkpoint_value_digest(
        request.model_dump(
            mode="json",
            include={
                "review_reference",
                "decision",
                "reason",
                "metadata",
                "resolved_by",
            },
        )
    )


def require_resolution_intent_matches_approval(
    intent: pending_approval_reader.ApprovalResolutionIntent,
    *,
    approval: PendingToolApproval,
) -> None:
    expected = approval_resolution_intent_for(
        approval,
        decision=intent.decision,
        resolution_request_digest=intent.resolution_request_digest,
        pause_resolved_at=intent.pause_resolved_at,
        reviewed_approval_digest=(
            None
            if intent.reviewed_approval_digest is None
            else runtime_publication_checkpoint_value_digest(approval.model_dump(mode="json"))
        ),
    )
    if intent != expected:
        raise RuntimeError("Approval resolution intent conflicts with its pending approval.")


def checkpoint_with_approval_resolution_intent(
    checkpoint: dict[str, Any] | None,
    *,
    approval: PendingToolApproval,
    decision: ToolApprovalDecision,
    resolution_request_digest: str,
    redactor: SecretRedactor,
    runtime_session: Session | None = None,
    reviewed_approval_digest: str | None = None,
    pause_resolved_at: datetime | None = None,
) -> dict[str, Any]:
    """Set or validate one immutable decision inside an approval claim."""

    copied = _checkpoint_with_exact_pending_approval_round(
        checkpoint,
        approval=approval,
        redactor=redactor,
        runtime_session=runtime_session,
    )
    expected = approval_resolution_intent_for(
        approval,
        decision=decision,
        resolution_request_digest=resolution_request_digest,
        reviewed_approval_digest=reviewed_approval_digest,
        pause_resolved_at=pause_resolved_at,
    )
    current = pending_approval_reader.approval_resolution_intent_from_checkpoint(
        copied, redactor=redactor
    )
    if current is not None:
        require_resolution_intent_matches_approval(current, approval=approval)
        if current.decision is not decision:
            raise RuntimeError(
                "Tool approval was already claimed with a different resolution decision."
            )
        if current.resolution_request_digest != resolution_request_digest:
            raise RuntimeError(
                "Tool approval was already claimed with a different resolution request."
            )
        if current.reviewed_approval_digest != reviewed_approval_digest:
            raise RuntimeError("Tool approval cannot replace its accepted content binding.")
        expected = current
    copied[pending_approval_reader.APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY] = expected.model_dump(
        mode="json"
    )
    return copied


def bounded_resolution_metadata_payload(
    metadata: dict[str, Any],
    *,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    """Return bounded, redacted audit metadata without touching identity fields."""

    prioritized = metadata
    if BUSINESS_APPROVAL_RESOLUTION_METADATA_KEY in metadata:
        business_stamp = metadata[BUSINESS_APPROVAL_RESOLUTION_METADATA_KEY]
        if type(business_stamp) is dict:
            ordered_stamp = {
                key: business_stamp[key]
                for key in _BUSINESS_APPROVAL_STAMP_PRIORITY_FIELDS
                if key in business_stamp
            }
            ordered_stamp.update(
                (key, value) for key, value in business_stamp.items() if key not in ordered_stamp
            )
            business_stamp = ordered_stamp
        prioritized = {
            BUSINESS_APPROVAL_RESOLUTION_METADATA_KEY: business_stamp,
            **{
                key: value
                for key, value in metadata.items()
                if key != BUSINESS_APPROVAL_RESOLUTION_METADATA_KEY
            },
        }
    evidence = tool_results.portable_result_evidence(prioritized, redactor=redactor)
    bounded = evidence.value if evidence.included and type(evidence.value) is dict else {}
    payload: dict[str, Any] = {"metadata": bounded}
    if evidence.incomplete:
        payload["metadata_truncated"] = True
    return payload


def bounded_pending_approval_event_payload(
    approval: PendingToolApproval,
    *,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    """Return an event-safe approval copy with bounded policy metadata.

    The checkpoint remains the authority for resumption. The event copy is
    intentionally bounded because policy metadata is adapter-owned and may
    otherwise turn an audit event into an unbounded storage surface.
    """

    if type(approval) is not PendingToolApproval:
        raise TypeError("approval must be a PendingToolApproval.")
    publish_arguments = approval.publish_arguments is True
    payload = approval.model_dump(
        mode="json",
        exclude={"publish_arguments"},
        warnings=False,
    )
    # Secret-scope provenance is private checkpoint evidence, not public
    # approval content.  Only a positively static scope proves that a policy's
    # argument-derived output cannot become a late-resolved workload secret.
    payload.pop("secret_resolution_scope", None)
    payload.pop("run_limit_accounting", None)
    publish_policy_output = approval.secret_resolution_scope == "static" and publish_arguments
    # Pause events precede execution, so arguments remain private even when
    # static scope permits publishing bounded policy output. Keep this payload
    # canonical before recording a paired terminal decision.
    payload.pop("arguments", None)
    payload[tool_argument_publication.ARGUMENTS_STATE_FIELD] = "quarantined"
    if not publish_policy_output:
        payload.pop("reason", None)
        payload.pop("metadata", None)
    bounded = (
        bounded_resolution_metadata_payload(approval.metadata, redactor=redactor)
        if publish_policy_output
        else {}
    )
    if publish_policy_output:
        payload["metadata"] = bounded["metadata"]
    truncated_tool_call_ids: list[str] = []
    tool_calls = payload.get("tool_calls")
    if type(tool_calls) is not list:
        raise TypeError("Pending approval event payload must contain tool_calls.")
    for index, pending_call in enumerate(approval.tool_calls):
        raw_call = tool_calls[index]
        if type(raw_call) is not dict:
            raise TypeError("Pending approval event tool calls must be objects.")
        raw_call.pop("model_tool_name", None)
        raw_call.pop("targeted_tool_grant_id", None)
        raw_call.pop("targeted_tool_invocation", None)
        raw_call.pop("targeted_tool_rejection", None)
        raw_call.pop("arguments", None)
        raw_call[tool_argument_publication.ARGUMENTS_STATE_FIELD] = "quarantined"
        if not publish_policy_output:
            raw_call.pop("reason", None)
            raw_call.pop("metadata", None)
            continue
        call_metadata = bounded_resolution_metadata_payload(
            pending_call.metadata,
            redactor=redactor,
        )
        raw_call["metadata"] = call_metadata["metadata"]
        if call_metadata.get("metadata_truncated") is True:
            truncated_tool_call_ids.append(pending_call.tool_call_id)
    result: dict[str, Any] = {"approval": payload}
    if approval.execution_profile_fingerprint is not None:
        result[_EXECUTION_PROFILE_FINGERPRINT_FIELD] = approval.execution_profile_fingerprint
    if publish_policy_output and bounded.get("metadata_truncated") is True:
        result["approval_metadata_truncated"] = True
    if truncated_tool_call_ids:
        result["tool_call_metadata_truncated"] = truncated_tool_call_ids
    return result


def public_policy_denial_result(
    *,
    secret_resolution_scope: Literal["static", "dynamic", "unknown"],
    policy_result: ToolPolicyResult,
    publish_arguments: bool = True,
) -> ToolPolicyResult:
    """Remove argument-derived denial output without positive static evidence."""

    if secret_resolution_scope not in {"static", "dynamic", "unknown"}:
        raise ValueError("secret_resolution_scope must be static, dynamic, or unknown.")
    if type(policy_result) is not ToolPolicyResult:
        raise TypeError("policy_result must be a ToolPolicyResult.")
    if type(publish_arguments) is not bool:
        raise TypeError("publish_arguments must be a bool.")
    if policy_result.decision is not ToolPolicyDecision.DENY:
        raise ValueError("Public policy result must be a denial.")
    if policy_result.command_denial_code is not None:
        code = policy_result.command_denial_code
        return ToolPolicyResult(
            decision=ToolPolicyDecision.DENY,
            reason=COMMAND_DENIAL_HINTS[code],
            metadata={"command_denial_code": code.value},
            command_denial_code=code,
        )
    if secret_resolution_scope == "static" and publish_arguments:
        return policy_result.model_copy(deep=True)
    return ToolPolicyResult(decision=ToolPolicyDecision.DENY)


async def checkpoint_without_pending_approval(
    session_store: SessionStore,
    session_id: str,
) -> dict[str, Any]:
    """Copy a session checkpoint without its pending approval marker."""
    checkpoint = await session_store.load_checkpoint(session_id)
    copied = {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
    copied.pop(pending_approval_reader.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY, None)
    copied.pop(pending_approval_reader.APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY, None)
    return copied


def _checkpoint_with_exact_pending_approval_round(
    checkpoint: dict[str, Any] | None,
    *,
    approval: PendingToolApproval,
    redactor: SecretRedactor,
    runtime_session: Session | None = None,
) -> dict[str, Any]:
    copied = {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
    current_approval = pending_approval_reader.pending_approval_from_checkpoint(
        copied, redactor=redactor
    )
    if current_approval != approval:
        raise RuntimeError("Pending tool approval changed before exact checkpoint clearing.")
    current_round = pending_round_reader.pending_tool_round_from_checkpoint(
        copied,
        redactor=redactor,
        runtime_session=runtime_session,
    )
    if current_round is None or current_round.policy_state != "planned":
        raise RuntimeError("Pending tool approval has no policy-planned round to clear.")
    if (
        current_round.tool_round_id != approval.tool_round_id
        or current_round.model_step_id != approval.model_step_id
        or current_round.model_attempt_id != approval.model_attempt_id
        or [call.model_dump(mode="json") for call in current_round.tool_calls]
        != [call.model_dump(mode="json") for call in approval.tool_calls]
    ):
        raise RuntimeError("Pending tool approval round changed before exact checkpoint clearing.")
    intent = pending_approval_reader.approval_resolution_intent_from_checkpoint(
        copied, redactor=redactor
    )
    if intent is not None:
        require_resolution_intent_matches_approval(intent, approval=approval)
    return copied


def checkpoint_without_exact_pending_approval(
    checkpoint: dict[str, Any] | None,
    *,
    approval: PendingToolApproval,
    redactor: SecretRedactor,
    runtime_session: Session | None = None,
) -> dict[str, Any]:
    """Clear only the exact approval while retaining its planned round."""

    copied = _checkpoint_with_exact_pending_approval_round(
        checkpoint,
        approval=approval,
        redactor=redactor,
        runtime_session=runtime_session,
    )
    copied.pop(pending_approval_reader.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY)
    copied.pop(pending_approval_reader.APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY, None)
    return copied


def checkpoint_without_exact_pending_approval_round(
    checkpoint: dict[str, Any] | None,
    *,
    approval: PendingToolApproval,
    redactor: SecretRedactor,
    runtime_session: Session | None = None,
) -> dict[str, Any]:
    """Clear only the exact paired approval and policy-planned round."""

    copied = _checkpoint_with_exact_pending_approval_round(
        checkpoint,
        approval=approval,
        redactor=redactor,
        runtime_session=runtime_session,
    )
    copied.pop(pending_approval_reader.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY)
    copied.pop(pending_approval_reader.APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY, None)
    copied.pop(pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY)
    return copied


def pending_approval_expired(approval: PendingToolApproval, now: datetime) -> bool:
    """Whether a pending approval's window has closed at ``now``.

    Pure access-time check (no daemon); the resolution winner evaluates it
    after the atomic status claim. A future lifecycle sweep (issue #104) can
    call this on interrupted sessions to proactively deny expired approvals.
    """
    return approval.expires_at is not None and now >= approval.expires_at


class ApprovalResolutionHistory(NamedTuple):
    has_resolution_attempt: bool
    has_denied_result: bool
    has_approved_call: bool
    has_executed_or_recovered_result: bool
    has_nonexecuting_approval_resolution: bool

    @property
    def has_resolution_activity(self) -> bool:
        """Whether durable history proves that an approval resolution already progressed."""

        return self.has_resolution_attempt or self.has_denied_result or self.has_granted_activity

    @property
    def has_granted_activity(self) -> bool:
        """The approval was already granted or produced executed results.

        Expiry gates only the FIRST grant: a retry after a mid-run crash
        re-resolves an approval that was authorized in-window, so coercing it
        to a denial would contradict the recorded grant (and deadlock against
        ``validate_retry_decision``).
        """
        return (
            self.has_approved_call
            or self.has_executed_or_recovered_result
            or self.has_nonexecuting_approval_resolution
        )


class ToolApprovalManualRecoveryRequired(RuntimeError):
    def __init__(self, *, tool_call_id: str, tool_name: str) -> None:
        super().__init__(
            "Tool approval cannot be retried automatically because a tool call "
            f"started without a terminal result: {tool_call_id} ({tool_name})."
        )
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name


class RoundToolManualRecoveryRequired(RuntimeError):
    def __init__(self, *, tool_call_id: str, tool_name: str) -> None:
        super().__init__(
            "A paused round cannot be resumed automatically because a tool call started "
            f"without a terminal result: {tool_call_id} ({tool_name})."
        )
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name


def resumed_event(
    *,
    session: Session,
    agent_name: str,
    environment_name: str | None,
    approval: PendingToolApproval,
    decision: ToolApprovalDecision,
    resolved_by: ResolutionActor | None = None,
    expired: bool = False,
) -> Event:
    event = Event(
        type=EventType.SESSION_RESUMED,
        session_id=session.id,
        agent_name=agent_name,
        environment_name=environment_name,
        payload={
            "model_step_id": approval.model_step_id,
            "model_attempt_id": approval.model_attempt_id,
            "tool_round_id": approval.tool_round_id,
            "agent_name": agent_name,
            "approval_id": approval.approval_id,
            "tool_call_id": approval.tool_call_id,
            "decision": decision.value,
            **(
                {_EXECUTION_PROFILE_FINGERPRINT_FIELD: (approval.execution_profile_fingerprint)}
                if approval.execution_profile_fingerprint is not None
                else {}
            ),
            "resolved_by": resolution_actor_payload(resolved_by),
            "expired": expired,
        },
    )
    return event_with_runtime_payload_authority(
        event,
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "approval_id",
        *(
            (_EXECUTION_PROFILE_FINGERPRINT_FIELD,)
            if approval.execution_profile_fingerprint is not None
            else ()
        ),
    )


def cleared_event(
    *,
    session: Session,
    agent_name: str,
    environment_name: str | None,
    approval: PendingToolApproval,
) -> Event:
    event = Event(
        type=EventType.SESSION_CHECKPOINTED,
        session_id=session.id,
        agent_name=agent_name,
        environment_name=environment_name,
        payload={
            "model_step_id": approval.model_step_id,
            "model_attempt_id": approval.model_attempt_id,
            "tool_round_id": approval.tool_round_id,
            "checkpoint": pending_approval_reader.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY,
            "approval_id": approval.approval_id,
            "tool_call_id": approval.tool_call_id,
            "cleared": True,
        },
    )
    return event_with_runtime_payload_authority(
        event,
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "approval_id",
    )


def checkpoint_for_fork(
    *,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    copied_checkpoint = copy_durable_json_value(checkpoint, "checkpoint")
    pending_approval = pending_approval_reader.pending_approval_from_checkpoint(copied_checkpoint)
    if pending_approval is not None:
        raise RuntimeError(
            "Session awaiting tool approval cannot be forked; resolve it with "
            "resolve_tool_approval(...) first."
        )
    return copied_checkpoint


def approval_denied_tool_result(
    request: ToolApprovalRequest,
    *,
    approval: PendingToolApproval,
    tool_call: runtime_records.ToolCallRequest,
    approval_required: bool,
) -> ToolResult:
    if request.reason:
        reason = request.reason
        if approval_required:
            content = f"Tool call denied by approval: {request.reason}"
        else:
            content = (
                "Tool call skipped because approval was denied for the same tool round: "
                f"{request.reason}"
            )
    elif approval_required:
        reason = "Tool call denied by approval."
        content = reason
    else:
        reason = "Tool call skipped because approval was denied for the same tool round."
        content = reason

    return ToolResult(
        content=content,
        structured={
            "model_step_id": approval.model_step_id,
            "model_attempt_id": approval.model_attempt_id,
            "tool_round_id": approval.tool_round_id,
            "decision": request.decision.value,
            "approval_id": approval.approval_id,
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "approval_required": approval_required,
            "denied_by_approval": approval_required,
            "skipped_due_to_approval_denial": not approval_required,
            "denied_tool_call_id": approval.tool_call_id,
            "denied_tool_name": approval.tool_name,
            "reason": reason,
            "metadata": request.metadata,
        },
        is_error=True,
    )


def user_input_resume_events(events: list[Event], input_id: str) -> list[Event]:
    """Return only the events belonging to a user-input pause's resume attempts.

    User-input round terminal events carry no ``approval_id``, and tool-call ids are only unique
    within one assistant message — not per session — so a round's events cannot be identified by
    id alone. The round runs no tools before it pauses, so every ``started``/terminal event for
    the round is emitted AFTER the pause boundary; events before it — prior rounds that may reuse
    the same ids — are excluded.

    The boundary is the FIRST event that marks this pause: either ``session.awaiting_user_input``
    (payload ``input_id``) or a ``session.interrupted`` carrying ``user_input.input_id`` for it.
    Both are accepted because a bounded recovery window may begin at the terminal
    ``session.interrupted`` evidence rather than the earlier atomic open publication. The
    ``pending_user_input`` checkpoint and awaiting event cannot split: the exact user-input-open
    receipt publishes them together. Anchoring on the terminal pause too keeps the retry ledger
    scoped rather than empty, which would re-run an already-completed sibling.
    """
    for index, event in enumerate(events):
        if _event_marks_user_input_pause(event, input_id):
            return events[index + 1 :]
    return []


def _event_marks_user_input_pause(event: Event, input_id: str) -> bool:
    if event.type == EventType.SESSION_AWAITING_USER_INPUT:
        return event.payload.get("input_id") == input_id
    if event.type == EventType.SESSION_INTERRUPTED:
        user_input = event.payload.get("user_input")
        return isinstance(user_input, dict) and user_input.get("input_id") == input_id
    return False


def recorded_round_tool_outcomes(
    *,
    events: list[Event],
    pending_calls: list[PendingToolCallApproval],
    input_id: str,
    tool_round_identity: ToolRoundIdentity,
    staged_terminals: list[tool_round_recovery.StagedToolCallTerminal] | None = None,
) -> dict[str, runtime_records.ToolCallOutcome]:
    """Reconstruct already-recorded terminal outcomes for a paused user-input round, keyed by
    ``tool_call_id``, scoped to the pause's resume window (see ``user_input_resume_events``).

    Lets a retried resume skip re-executing a tool that already completed before a mid-resume
    failure, without colliding with a prior round that reused the same ids.
    """
    identity = copy_tool_round_identity(tool_round_identity)
    pending_by_id = {call.tool_call_id: call for call in pending_calls}
    resume_events = user_input_resume_events(events, input_id)
    ledger = resume_ledger.scan_tool_call_events(
        events=resume_events,
        pending_calls=pending_calls,
        in_scope=lambda event: identity.matches_payload(event.payload),
        candidate_scope=lambda event: (
            identity.matches_payload(event.payload) or event.payload.get("input_id") == input_id
        ),
        terminal_event_types=_USER_INPUT_ROUND_TERMINAL_EVENT_TYPES,
    )
    if ledger.scope_conflicting:
        raise resume_ledger.ToolCallEvidenceConflict(
            "User-input recovery evidence contains a call outside the pending tool round."
        )
    staged_ids = _validated_staged_terminal_ids(
        staged_terminals,
        pending_by_id=pending_by_id,
        identity=identity,
        pause_field="input_id",
        pause_id=input_id,
        recorded_outcomes=ledger.outcomes,
        durable_events=resume_events,
        conflicting_ids=ledger.conflicting_ids,
    )
    # A tool that started on a prior resume attempt but has no terminal event (a crash mid-tool)
    # cannot be safely re-run — fail loudly instead of silently double-executing a side effect.
    for tool_call_id in ledger.started_without_terminal_ids:
        if tool_call_id in staged_ids:
            continue
        pending_call = pending_by_id[tool_call_id]
        raise RoundToolManualRecoveryRequired(
            tool_call_id=tool_call_id,
            tool_name=pending_call.tool_name,
        )
    return ledger.outcomes


def recorded_tool_outcomes(
    *,
    events: list[Event],
    approval: PendingToolApproval,
    staged_terminals: list[tool_round_recovery.StagedToolCallTerminal] | None = None,
) -> dict[str, runtime_records.ToolCallOutcome]:
    identity = _approval_tool_round_identity(approval)
    pending_calls = pending_round_tool_calls(approval)
    pending_by_id = {call.tool_call_id: call for call in pending_calls}
    ledger = resume_ledger.scan_tool_call_events(
        events=events,
        pending_calls=pending_calls,
        in_scope=lambda event: (
            event.payload.get("approval_id") == approval.approval_id
            and identity.matches_payload(event.payload)
        ),
        candidate_scope=lambda event: (
            identity.matches_payload(event.payload)
            or event.payload.get("approval_id") == approval.approval_id
        ),
        terminal_event_types=_APPROVAL_TERMINAL_EVENT_TYPES,
    )
    if ledger.scope_conflicting:
        raise resume_ledger.ToolCallEvidenceConflict(
            "Tool approval evidence contains a call outside the pending tool round."
        )
    staged_ids = _validated_staged_terminal_ids(
        staged_terminals,
        pending_by_id=pending_by_id,
        identity=identity,
        pause_field="approval_id",
        pause_id=approval.approval_id,
        recorded_outcomes=ledger.outcomes,
        durable_events=events,
        conflicting_ids=ledger.conflicting_ids,
    )

    for tool_call_id in ledger.started_without_terminal_ids:
        if tool_call_id in staged_ids:
            continue
        pending_tool_call = pending_by_id[tool_call_id]
        raise ToolApprovalManualRecoveryRequired(
            tool_call_id=tool_call_id,
            tool_name=pending_tool_call.tool_name,
        )

    return ledger.outcomes


def _validated_staged_terminal_ids(
    staged_terminals: list[tool_round_recovery.StagedToolCallTerminal] | None,
    *,
    pending_by_id: dict[str, PendingToolCallApproval],
    identity: ToolRoundIdentity,
    pause_field: Literal["approval_id", "input_id"],
    pause_id: str,
    recorded_outcomes: dict[str, runtime_records.ToolCallOutcome],
    durable_events: list[Event],
    conflicting_ids: set[str],
) -> set[str]:
    """Return only positively owned private terminals for retry classification."""

    if staged_terminals is None:
        return set()
    staged_ids: set[str] = set()
    for candidate in staged_terminals:
        staged = tool_round_recovery.StagedToolCallTerminal.model_validate(
            candidate.model_dump(mode="json")
        )
        pending_call = pending_by_id.get(staged.tool_call_id)
        if (
            pending_call is None
            or staged.tool_call_id in staged_ids
            or staged.tool_call_id in conflicting_ids
            or staged.event.tool_name != pending_call.tool_name
            or not identity.matches_payload(staged.event.payload)
            or staged.event.payload.get(pause_field) != pause_id
        ):
            raise resume_ledger.ToolCallEvidenceConflict(
                "Private staged terminal evidence conflicts with its paused tool round."
            )
        recorded = recorded_outcomes.get(staged.tool_call_id)
        if recorded is not None:
            staged_outcome = resume_ledger.tool_call_outcome_from_terminal_event(
                event=staged.event,
                pending_tool_call=pending_call,
            )
            durable_matches = [event for event in durable_events if event.id == staged.event.id]
            if (
                recorded != staged_outcome
                or len(durable_matches) != 1
                or durable_matches[0] != staged.event
            ):
                raise resume_ledger.ToolCallEvidenceConflict(
                    "Durable and staged terminal evidence conflict for one paused tool call."
                )
        staged_ids.add(staged.tool_call_id)
    return staged_ids


def approval_resolution_history(
    *,
    events: list[Event],
    approval: PendingToolApproval,
) -> ApprovalResolutionHistory:
    identity = _approval_tool_round_identity(approval)
    has_resolution_attempt = False
    has_denied_result = False
    has_approved_call = False
    has_executed_or_recovered_result = False
    has_nonexecuting_approval_resolution = False

    for event in events:
        if event.type not in _APPROVAL_HISTORY_EVENT_TYPES:
            continue
        approval_matches = event.payload.get("approval_id") == approval.approval_id
        identity_matches = identity.matches_payload(event.payload)
        if not approval_matches and not identity_matches:
            continue
        if not approval_matches or not identity_matches:
            raise resume_ledger.ToolCallEvidenceConflict(
                "Tool approval history contains contradictory execution identity."
            )
        if event.type == EventType.SESSION_RESUMED:
            if (
                event.payload.get("tool_call_id") != approval.tool_call_id
                or event.agent_name != approval.agent_name
                or event.environment_name != approval.environment_name
            ):
                raise resume_ledger.ToolCallEvidenceConflict(
                    "Tool approval history contains a contradictory resolution attempt."
                )
            has_resolution_attempt = True
            continue
        try:
            pending_tool_call_for_approval_event(
                event=event,
                approval=approval,
            )
        except ValueError:
            raise resume_ledger.ToolCallEvidenceConflict(
                "Tool approval history contains a contradictory tool-call descriptor."
            ) from None
        if event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED:
            has_resolution_attempt = True
        elif event.type == EventType.TOOL_CALL_APPROVAL_DENIED:
            has_denied_result = True
        elif event.type == EventType.TOOL_CALL_APPROVED:
            has_approved_call = True
        elif event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}:
            has_executed_or_recovered_result = True
        elif (
            event.type == EventType.TOOL_CALL_BLOCKED
            and event.payload.get("blocked_by") == "policy_evaluation_ambiguous"
            and event.payload.get("requested_decision") == ToolApprovalDecision.APPROVE.value
        ):
            has_nonexecuting_approval_resolution = True

    return ApprovalResolutionHistory(
        has_resolution_attempt=has_resolution_attempt,
        has_denied_result=has_denied_result,
        has_approved_call=has_approved_call,
        has_executed_or_recovered_result=has_executed_or_recovered_result,
        has_nonexecuting_approval_resolution=has_nonexecuting_approval_resolution,
    )


def validate_retry_decision(
    *,
    history: ApprovalResolutionHistory,
    approval: PendingToolApproval,
    decision: ToolApprovalDecision,
) -> None:
    if decision == ToolApprovalDecision.APPROVE and history.has_denied_result:
        raise RuntimeError(
            "Tool approval was already denied and cannot be retried as approved: "
            f"{approval.approval_id}"
        )
    if decision == ToolApprovalDecision.DENY and history.has_granted_activity:
        raise RuntimeError(
            "Tool approval already has approved or executed tool results and "
            f"cannot be retried as denied: {approval.approval_id}"
        )


def pending_tool_call_for_recovery(
    *,
    approval: PendingToolApproval,
    tool_call_id: str,
) -> PendingToolCallApproval:
    for pending_tool_call in pending_round_tool_calls(approval):
        if pending_tool_call.tool_call_id == tool_call_id:
            return pending_tool_call
    raise ValueError(f"Tool call is not part of the pending approval: {tool_call_id}")


def validate_recovery_target(
    *,
    events: list[Event],
    approval: PendingToolApproval,
    tool_call_id: str,
) -> None:
    identity = _approval_tool_round_identity(approval)
    pending_tool_call = pending_tool_call_for_recovery(
        approval=approval,
        tool_call_id=tool_call_id,
    )
    state = resume_ledger.tool_call_recovery_state(
        events=events,
        pending_calls=pending_round_tool_calls(approval),
        tool_call_id=pending_tool_call.tool_call_id,
        in_scope=lambda event: (
            event.payload.get("approval_id") == approval.approval_id
            and identity.matches_payload(event.payload)
        ),
        candidate_scope=lambda event: (
            identity.matches_payload(event.payload)
            or event.payload.get("approval_id") == approval.approval_id
        ),
        terminal_event_types=_APPROVAL_TERMINAL_EVENT_TYPES,
    )

    if state.conflicting:
        return
    if state.terminal:
        raise RuntimeError(
            f"Tool call already has a terminal event and does not need recovery: {tool_call_id}"
        )
    if not state.started:
        raise RuntimeError(
            f"Tool approval recovery requires a recorded tool.call.started event: {tool_call_id}"
        )


def round_tool_call_for_recovery(
    *,
    pending_calls: list[PendingToolCallApproval],
    tool_call_id: str,
) -> PendingToolCallApproval:
    for pending_tool_call in pending_calls:
        if pending_tool_call.tool_call_id == tool_call_id:
            return PendingToolCallApproval(**pending_tool_call.model_dump())
    raise ValueError(f"Tool call is not part of the paused round: {tool_call_id}")


def validate_round_recovery_target(
    *,
    events: list[Event],
    pending_calls: list[PendingToolCallApproval],
    tool_call_id: str,
    input_id: str,
    tool_round_identity: ToolRoundIdentity,
) -> None:
    # Round terminal events carry no approval_id (user-input rounds) and tool-call ids are not
    # unique across the session, so scope to the pause's resume window (matching
    # recorded_round_tool_outcomes) — a prior round reusing this id must not be seen here.
    pending_tool_call = round_tool_call_for_recovery(
        pending_calls=pending_calls,
        tool_call_id=tool_call_id,
    )
    identity = copy_tool_round_identity(tool_round_identity)
    state = resume_ledger.tool_call_recovery_state(
        events=user_input_resume_events(events, input_id),
        pending_calls=pending_calls,
        tool_call_id=pending_tool_call.tool_call_id,
        in_scope=lambda event: identity.matches_payload(event.payload),
        candidate_scope=lambda event: (
            identity.matches_payload(event.payload) or event.payload.get("input_id") == input_id
        ),
        terminal_event_types=_USER_INPUT_ROUND_TERMINAL_EVENT_TYPES,
    )

    if state.conflicting:
        return
    if state.terminal:
        raise RuntimeError(
            f"Tool call already has a terminal event and does not need recovery: {tool_call_id}"
        )
    if not state.started:
        raise RuntimeError(
            f"User input recovery requires a recorded tool.call.started event: {tool_call_id}"
        )


def _approval_tool_round_identity(approval: PendingToolApproval) -> ToolRoundIdentity:
    return ToolRoundIdentity(
        tool_round_id=approval.tool_round_id,
        model_step_id=approval.model_step_id,
        model_attempt_id=approval.model_attempt_id,
    )


def recovered_tool_result(
    *,
    request: ToolApprovalRecoveryRequest,
) -> ToolResult:
    if request.outcome not in {
        ToolApprovalRecoveryOutcome.COMPLETED,
        ToolApprovalRecoveryOutcome.FAILED,
    }:
        raise ValueError(f"Unsupported tool approval recovery outcome: {request.outcome}")
    return ToolResult(
        content=request.message,
        structured=request.structured,
        artifacts=request.artifacts,
        is_error=request.outcome == ToolApprovalRecoveryOutcome.FAILED,
    )


def pending_tool_call_approvals(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] | None,
    default_policy_evidence: ToolPolicyEvidence = ToolPolicyEvidence.UNPLANNED,
    active_taint_by_id: Mapping[str, frozenset[str]] = MappingProxyType({}),
    redactor: SecretRedactor | None = None,
) -> list[PendingToolCallApproval]:
    policy_results_by_id: dict[str, ToolPolicyResult | None] = {}
    policy_evidence_by_id: dict[str, ToolPolicyEvidence] = {}
    if policy_outcomes is not None:
        policy_results_by_id = {outcome.call.id: outcome.result for outcome in policy_outcomes}
        policy_evidence_by_id = {outcome.call.id: outcome.evidence for outcome in policy_outcomes}
    pending_approvals: list[PendingToolCallApproval] = []
    for tool_call in tool_calls:
        policy_result = policy_results_by_id.get(tool_call.id)
        pending_approvals.append(
            PendingToolCallApproval(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                arguments=copy_durable_json_value(tool_call.arguments, "arguments"),
                targeted_tool_grant_id=tool_call.targeted_tool_grant_id,
                model_tool_name=tool_call.model_tool_name,
                targeted_tool_invocation=tool_call.targeted_tool_invocation,
                targeted_tool_rejection=tool_call.targeted_tool_rejection,
                policy_evidence=policy_evidence_by_id.get(
                    tool_call.id,
                    default_policy_evidence,
                ),
                policy_decision=policy_result.decision.value if policy_result is not None else None,
                command_denial_code=(
                    policy_result.command_denial_code if policy_result is not None else None
                ),
                reason=resume_ledger.policy_reason_for_pending_tool_call(
                    policy_result,
                    redactor=redactor,
                ),
                metadata=(
                    (
                        copy_durable_metadata(policy_result.metadata, "metadata")
                        if redactor is None
                        else copy_durable_json_value(
                            redactor.redact_json_values(policy_result.metadata),
                            "metadata",
                        )
                    )
                    if policy_result is not None
                    else {}
                ),
                active_taint_labels=sorted(active_taint_by_id.get(tool_call.id, frozenset())),
            )
        )
    return pending_approvals


def pending_round_tool_calls(
    approval: PendingToolApproval,
) -> list[PendingToolCallApproval]:
    return [PendingToolCallApproval(**call.model_dump()) for call in approval.tool_calls]


def tool_call_request_from_pending(
    call: PendingToolCallApproval,
    *,
    arguments: dict[str, Any] | None = None,
) -> runtime_records.ToolCallRequest:
    """Restore one private pending call without dropping gateway authority."""

    if type(call) is not PendingToolCallApproval:
        raise TypeError("call must be a PendingToolCallApproval.")
    return runtime_records.ToolCallRequest(
        id=call.tool_call_id,
        name=call.tool_name,
        arguments=copy_durable_json_value(
            call.arguments if arguments is None else arguments,
            "arguments",
        ),
        targeted_tool_grant_id=call.targeted_tool_grant_id,
        model_tool_name=call.model_tool_name,
        targeted_tool_invocation=call.targeted_tool_invocation,
        targeted_tool_rejection=call.targeted_tool_rejection,
    )


def policy_result_from_pending_tool_call(
    pending_tool_call: PendingToolCallApproval,
) -> ToolPolicyResult | None:
    return resume_ledger.policy_result_from_pending_tool_call(pending_tool_call)


def taint_labels_from_pending_tool_call(
    pending_tool_call: PendingToolCallApproval,
) -> frozenset[str]:
    """Active taint labels persisted for this call, restored so the resumed tool is gated with the
    same taint the policy used before the pause."""
    return frozenset(pending_tool_call.active_taint_labels)


def _pending_approval_and_round_for_atomic_claim(
    checkpoint: dict[str, Any] | None,
    *,
    approval_id: str,
    tool_round_id: str,
    gating_tool_call_id: str | None = None,
    recovery_tool_call_id: str | None = None,
    redactor: SecretRedactor,
    runtime_session: Session | None = None,
) -> tuple[PendingToolApproval, pending_rounds.PendingToolRound]:
    if (gating_tool_call_id is None) == (recovery_tool_call_id is None):
        raise TypeError("Exactly one approval gating or recovery tool-call identity is required.")
    approval = pending_approval_reader.pending_approval_from_checkpoint(
        checkpoint,
        redactor=redactor,
    )
    if approval is None:
        raise RuntimeError("Session has no pending tool approval.")
    if approval.approval_id != approval_id or approval.tool_round_id != tool_round_id:
        raise ValueError("Tool approval identity does not match the current pending approval.")
    pending_round = pending_round_reader.pending_tool_round_from_checkpoint(
        checkpoint,
        redactor=redactor,
        runtime_session=runtime_session,
    )
    reconstructed_approval_only_round = pending_round is None
    if reconstructed_approval_only_round:
        # Compatibility boundary for checkpoints written before the paired
        # approval/round contract. PendingToolApproval is itself validated and
        # carries the complete policy-planned call list; the atomic claim below
        # persists this projection before any resolution work can begin.
        pending_round = pending_approval_reader.planned_tool_round_from_pending_approval(approval)
    if pending_round.policy_state != "planned":
        raise RuntimeError("Pending tool approval has no durable policy plan.")
    if (
        pending_round.tool_round_id != approval.tool_round_id
        or pending_round.model_step_id != approval.model_step_id
        or pending_round.model_attempt_id != approval.model_attempt_id
        or (
            not reconstructed_approval_only_round
            and not pending_approval_reader.pending_approval_scope_matches_round(
                approval,
                pending_round,
            )
        )
        or [call.model_dump(mode="json") for call in pending_round.tool_calls]
        != [call.model_dump(mode="json") for call in approval.tool_calls]
    ):
        raise RuntimeError("Pending tool approval conflicts with its durable tool round.")
    gating_calls = [
        call for call in pending_round.tool_calls if call.tool_call_id == approval.tool_call_id
    ]
    gating_evidence = (
        None
        if len(gating_calls) != 1
        else pending_approval_reader.effective_tool_policy_evidence(gating_calls[0])
    )
    if len(gating_calls) != 1 or not (
        (
            gating_evidence is ToolPolicyEvidence.AUTHORITATIVE
            and gating_calls[0].policy_decision == ToolPolicyDecision.REQUIRE_APPROVAL.value
        )
        or gating_evidence is ToolPolicyEvidence.AMBIGUOUS
    ):
        raise RuntimeError(
            "Pending approval call is neither authoritatively approval-gated "
            "nor explicitly ambiguous."
        )
    resolution_intent = pending_approval_reader.approval_resolution_intent_from_checkpoint(
        checkpoint,
        redactor=redactor,
    )
    if resolution_intent is not None:
        require_resolution_intent_matches_approval(
            resolution_intent,
            approval=approval,
        )
    if gating_tool_call_id is not None and approval.tool_call_id != gating_tool_call_id:
        raise ValueError("Tool approval identity does not match the current pending approval.")
    if recovery_tool_call_id is not None and not any(
        call.tool_call_id == recovery_tool_call_id for call in pending_round.tool_calls
    ):
        raise ValueError("Recovery tool call is not part of the pending approval round.")
    return approval, pending_round
