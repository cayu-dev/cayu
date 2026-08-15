from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cayu._validation import copy_durable_json_value, require_durable_clean_nonblank
from cayu.core.events import (
    Event,
    EventType,
    event_with_runtime_nested_payload_authority,
    event_with_runtime_payload_authority,
)
from cayu.core.tools import ToolResult
from cayu.runtime import _resume_ledger as resume_ledger
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_argument_publication as tool_argument_publication
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._checkpoint_redaction import durable_value_contains_secret
from cayu.runtime.approvals import (
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
from cayu.runtime.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.runtime.sessions import (
    Session,
    SessionStore,
    runtime_publication_checkpoint_value_digest,
)
from cayu.runtime.tool_policy import ToolPolicyDecision, ToolPolicyResult
from cayu.vaults import SecretRedactor, contains_redacted_secret

PENDING_TOOL_APPROVAL_CHECKPOINT_KEY = "pending_tool_approval"
APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY = "approval_resolution_intent"
APPROVAL_INTERRUPT_CLOSE_INTENT_KEY = "approval_close_intent"
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


class ApprovalResolutionIntent(BaseModel):
    """Immutable resolution authority retained with one pending approval."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    approval_id: str
    tool_call_id: str
    tool_round_id: str
    model_step_id: str
    model_attempt_id: str
    decision: ToolApprovalDecision
    # ``None`` loads checkpoints written before request digests existed. It is
    # intentionally non-authoritative and must never be upgraded after the fact.
    resolution_request_digest: str | None = None

    @field_validator("approval_id", "tool_call_id")
    @classmethod
    def validate_nonblank_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("resolution_request_digest")
    @classmethod
    def validate_resolution_request_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("resolution_request_digest must be a lowercase SHA-256 digest.")
        return value

    @model_validator(mode="after")
    def validate_execution_identity(self) -> ApprovalResolutionIntent:
        ToolRoundIdentity(
            tool_round_id=self.tool_round_id,
            model_step_id=self.model_step_id,
            model_attempt_id=self.model_attempt_id,
        )
        return self


def approval_resolution_intent_for(
    approval: PendingToolApproval,
    *,
    decision: ToolApprovalDecision,
    resolution_request_digest: str | None,
) -> ApprovalResolutionIntent:
    if type(approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if type(decision) is not ToolApprovalDecision:
        raise TypeError("Approval resolution decision must be a ToolApprovalDecision.")
    return ApprovalResolutionIntent(
        approval_id=approval.approval_id,
        tool_call_id=approval.tool_call_id,
        tool_round_id=approval.tool_round_id,
        model_step_id=approval.model_step_id,
        model_attempt_id=approval.model_attempt_id,
        decision=decision,
        resolution_request_digest=resolution_request_digest,
    )


def approval_resolution_request_digest(request: ToolApprovalRequest) -> str:
    """Bind retry-visible audit input without copying it into checkpoint state."""

    if type(request) is not ToolApprovalRequest:
        raise TypeError("request must be a ToolApprovalRequest.")
    return runtime_publication_checkpoint_value_digest(
        request.model_dump(
            mode="json",
            include={
                "decision",
                "reason",
                "metadata",
                "resolved_by",
            },
        )
    )


def approval_resolution_intent_from_checkpoint(
    checkpoint: dict[str, Any] | None,
    *,
    redactor: SecretRedactor | None = None,
) -> ApprovalResolutionIntent | None:
    if checkpoint is None:
        return None
    copied = copy_durable_json_value(checkpoint, "checkpoint")
    value = copied.get(APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY)
    if value is None:
        return None
    if redactor is not None and durable_value_contains_secret(
        value,
        redactor=redactor,
        path=(APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY,),
    ):
        raise ValueError(
            "Approval resolution intent contains a workload secret and cannot be executed."
        )
    if type(value) is not dict:
        raise ValueError("Approval resolution intent checkpoint must be an object.")
    try:
        return ApprovalResolutionIntent.model_validate(value)
    except Exception:
        if redactor is None:
            raise
        raise ValueError(
            "Approval resolution intent checkpoint is invalid and cannot be executed."
        ) from None


def require_resolution_intent_matches_approval(
    intent: ApprovalResolutionIntent,
    *,
    approval: PendingToolApproval,
) -> None:
    expected = approval_resolution_intent_for(
        approval,
        decision=intent.decision,
        resolution_request_digest=intent.resolution_request_digest,
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
) -> dict[str, Any]:
    """Set or validate one immutable decision inside an approval claim."""

    copied = _checkpoint_with_exact_pending_approval_round(
        checkpoint,
        approval=approval,
        redactor=redactor,
    )
    expected = approval_resolution_intent_for(
        approval,
        decision=decision,
        resolution_request_digest=resolution_request_digest,
    )
    current = approval_resolution_intent_from_checkpoint(copied, redactor=redactor)
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
    copied[APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY] = expected.model_dump(mode="json")
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
    if not publish_arguments:
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
        if not publish_arguments:
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
    if publish_policy_output and bounded.get("metadata_truncated") is True:
        result["approval_metadata_truncated"] = True
    if truncated_tool_call_ids:
        result["tool_call_metadata_truncated"] = truncated_tool_call_ids
    return result


def tool_round_secret_resolution_scope(
    pending_round: tool_round_recovery.PendingToolRound,
) -> Literal["static", "dynamic", "unknown"]:
    """Return positive durable secret-scope evidence for one tool round."""

    if type(pending_round) is not tool_round_recovery.PendingToolRound:
        raise TypeError("pending_round must be a PendingToolRound.")
    publication = pending_round.assistant_publication
    return "unknown" if publication is None else publication.secret_resolution_scope


def pending_approval_scope_matches_round(
    approval: PendingToolApproval,
    pending_round: tool_round_recovery.PendingToolRound,
) -> bool:
    """Accept legacy unknown scope, otherwise require paired positive evidence."""

    if type(approval) is not PendingToolApproval:
        raise TypeError("approval must be a PendingToolApproval.")
    round_scope = tool_round_secret_resolution_scope(pending_round)
    return (
        approval.secret_resolution_scope == "unknown"
        or approval.secret_resolution_scope == round_scope
    )


def public_pending_approval_reason(
    approval: PendingToolApproval,
    *,
    tool_call_id: str | None = None,
) -> str | None:
    """Return policy reason only with positive static-scope evidence."""

    if type(approval) is not PendingToolApproval:
        raise TypeError("approval must be a PendingToolApproval.")
    if approval.publish_arguments is not True or approval.secret_resolution_scope != "static":
        return None
    if tool_call_id is None or tool_call_id == approval.tool_call_id:
        return approval.reason
    for call in approval.tool_calls:
        if call.tool_call_id == tool_call_id:
            return call.reason
    return None


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
    if secret_resolution_scope == "static" and publish_arguments:
        return policy_result.model_copy(deep=True)
    return ToolPolicyResult(decision=ToolPolicyDecision.DENY)


def planned_tool_round_from_pending_approval(
    approval: PendingToolApproval,
) -> tool_round_recovery.PendingToolRound:
    """Project the complete planned round carried by an approval checkpoint.

    Releases before the paired-checkpoint contract stored the approval as the
    only round authority. The approval already contains every call and its
    policy decision, so it is sufficient positive evidence to reconstruct the
    missing round during an atomic claim without re-running policy.
    """

    if type(approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    return tool_round_recovery.PendingToolRound(
        tool_round_id=approval.tool_round_id,
        model_step_id=approval.model_step_id,
        model_attempt_id=approval.model_attempt_id,
        agent_name=approval.agent_name,
        environment_name=approval.environment_name,
        task_id=approval.task_id,
        execution_profile_fingerprint=approval.execution_profile_fingerprint,
        tool_calls=approval.tool_calls,
        policy_state="planned",
        policy_context_version=1,
        structured_output=approval.structured_output,
        thinking=approval.thinking,
        max_steps=approval.max_steps,
        limits=approval.limits,
        run_limit_accounting=approval.run_limit_accounting,
        budget_limits=approval.budget_limits,
        retry_policy=approval.retry_policy,
    )


async def checkpoint_without_pending_approval(
    session_store: SessionStore,
    session_id: str,
) -> dict[str, Any]:
    """Copy a session checkpoint without its pending approval marker."""
    checkpoint = await session_store.load_checkpoint(session_id)
    copied = {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
    copied.pop(PENDING_TOOL_APPROVAL_CHECKPOINT_KEY, None)
    copied.pop(APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY, None)
    return copied


def _checkpoint_with_exact_pending_approval_round(
    checkpoint: dict[str, Any] | None,
    *,
    approval: PendingToolApproval,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    copied = {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
    current_approval = pending_approval_from_checkpoint(copied, redactor=redactor)
    if current_approval != approval:
        raise RuntimeError("Pending tool approval changed before exact checkpoint clearing.")
    current_round = tool_round_recovery.pending_tool_round_from_checkpoint(
        copied,
        redactor=redactor,
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
    intent = approval_resolution_intent_from_checkpoint(copied, redactor=redactor)
    if intent is not None:
        require_resolution_intent_matches_approval(intent, approval=approval)
    return copied


def checkpoint_without_exact_pending_approval(
    checkpoint: dict[str, Any] | None,
    *,
    approval: PendingToolApproval,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    """Clear only the exact approval while retaining its planned round."""

    copied = _checkpoint_with_exact_pending_approval_round(
        checkpoint,
        approval=approval,
        redactor=redactor,
    )
    copied.pop(PENDING_TOOL_APPROVAL_CHECKPOINT_KEY)
    copied.pop(APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY, None)
    return copied


def checkpoint_without_exact_pending_approval_round(
    checkpoint: dict[str, Any] | None,
    *,
    approval: PendingToolApproval,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    """Clear only the exact paired approval and policy-planned round."""

    copied = _checkpoint_with_exact_pending_approval_round(
        checkpoint,
        approval=approval,
        redactor=redactor,
    )
    copied.pop(PENDING_TOOL_APPROVAL_CHECKPOINT_KEY)
    copied.pop(APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY, None)
    copied.pop(tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY)
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
            "checkpoint": PENDING_TOOL_APPROVAL_CHECKPOINT_KEY,
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
    pending_approval = pending_approval_from_checkpoint(copied_checkpoint)
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
    Both are needed: ``pending_user_input`` is checkpointed BEFORE the awaiting event is appended,
    so a worker that crashed in between has no awaiting event, but recovery finalizes the pause
    with a durable ``session.interrupted``. Anchoring on that too keeps the retry ledger scoped
    (rather than empty, which would re-run an already-completed sibling).
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


def pending_approval_from_checkpoint(
    checkpoint: dict[str, Any] | None,
    *,
    redactor: SecretRedactor | None = None,
    consume_on_rejection: bool = False,
) -> PendingToolApproval | None:
    if type(consume_on_rejection) is not bool:
        raise TypeError("consume_on_rejection must be a bool.")
    if checkpoint is None:
        return None
    copied_checkpoint = copy_durable_json_value(checkpoint, "checkpoint")
    value = copied_checkpoint.get(PENDING_TOOL_APPROVAL_CHECKPOINT_KEY)
    if value is None:
        return None
    if redactor is not None and durable_value_contains_secret(
        value,
        redactor=redactor,
        path=(PENDING_TOOL_APPROVAL_CHECKPOINT_KEY,),
    ):
        # Public callers retain their input by default. Runtime callers opt in
        # to consuming their private checkpoint copy so no outer traceback
        # frame keeps executable secret-bearing state.
        if type(value) is dict:
            value.clear()
        value = None
        copied_checkpoint.clear()
        if consume_on_rejection:
            checkpoint.clear()
        checkpoint = None
        raise ValueError(
            "Pending tool approval checkpoint contains a workload secret and cannot be executed."
        ) from None
    if type(value) is not dict:
        raise ValueError("Pending tool approval checkpoint must be an object.")
    validation_rejected = False
    try:
        approval = PendingToolApproval(**value)
    except Exception:
        if redactor is None:
            raise
        validation_rejected = True
    if validation_rejected:
        value.clear()
        value = None
        copied_checkpoint.clear()
        if consume_on_rejection:
            checkpoint.clear()
        checkpoint = None
        raise ValueError(
            "Pending tool approval checkpoint is invalid and cannot be executed."
        ) from None
    if contains_redacted_secret(approval.arguments) or any(
        contains_redacted_secret(call.arguments) for call in approval.tool_calls
    ):
        raise ValueError(
            "Pending approval arguments contain a redaction marker and cannot be executed."
        )
    return approval


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
                policy_evidence=policy_evidence_by_id.get(
                    tool_call.id,
                    default_policy_evidence,
                ),
                policy_decision=policy_result.decision.value if policy_result is not None else None,
                reason=resume_ledger.policy_reason_for_pending_tool_call(
                    policy_result,
                    redactor=redactor,
                ),
                metadata=(
                    (
                        copy_durable_json_value(policy_result.metadata, "metadata")
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


def policy_result_from_pending_tool_call(
    pending_tool_call: PendingToolCallApproval,
) -> ToolPolicyResult | None:
    return resume_ledger.policy_result_from_pending_tool_call(pending_tool_call)


def effective_tool_policy_evidence(
    pending_tool_call: PendingToolCallApproval,
) -> ToolPolicyEvidence:
    """Return explicit evidence, conservatively classifying legacy records.

    A legacy recognized decision is authoritative because it is durable.
    Absence of a decision in an old paused checkpoint is never positive
    authorization; it represents a call that was unregistered when planned.
    Raw legacy rounds are separately promoted to ``AMBIGUOUS`` by recovery.
    """

    if pending_tool_call.policy_evidence is not None:
        return pending_tool_call.policy_evidence
    if pending_tool_call.policy_decision in {
        ToolPolicyDecision.ALLOW.value,
        ToolPolicyDecision.DENY.value,
        ToolPolicyDecision.REQUIRE_APPROVAL.value,
    }:
        return ToolPolicyEvidence.AUTHORITATIVE
    return ToolPolicyEvidence.UNREGISTERED


def taint_labels_from_pending_tool_call(
    pending_tool_call: PendingToolCallApproval,
) -> frozenset[str]:
    """Active taint labels persisted for this call, restored so the resumed tool is gated with the
    same taint the policy used before the pause."""
    return frozenset(pending_tool_call.active_taint_labels)
