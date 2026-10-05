"""Read saved approvals and interpret their pending-action evidence."""

from __future__ import annotations

from typing import Any, Literal

from cayu._validation import copy_durable_json_value
from cayu.approvals.tools import PendingToolApproval, PendingToolCallApproval, ToolPolicyEvidence
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions._checkpoint_secret_validation import durable_value_contains_secret
from cayu.tools.policy import ToolPolicyDecision
from cayu.vaults.redaction import SecretRedactor, contains_redacted_secret

PENDING_TOOL_APPROVAL_CHECKPOINT_KEY = "pending_tool_approval"


def tool_round_secret_resolution_scope(
    pending_round: pending_rounds.PendingToolRound,
) -> Literal["static", "dynamic", "unknown"]:
    """Return positive durable secret-scope evidence for one tool round."""

    if type(pending_round) is not pending_rounds.PendingToolRound:
        raise TypeError("pending_round must be a PendingToolRound.")
    publication = pending_round.assistant_publication
    return "unknown" if publication is None else publication.secret_resolution_scope


def pending_approval_scope_matches_round(
    approval: PendingToolApproval,
    pending_round: pending_rounds.PendingToolRound,
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


def planned_tool_round_from_pending_approval(
    approval: PendingToolApproval,
) -> pending_rounds.PendingToolRound:
    """Project the complete planned round carried by an approval checkpoint.

    Releases before the paired-checkpoint contract stored the approval as the
    only round authority. The approval already contains every call and its
    policy decision, so it is sufficient positive evidence to reconstruct the
    missing round during an atomic claim without re-running policy.
    """

    if type(approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    return pending_rounds.PendingToolRound(
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
    try:
        return _pending_approval_from_owned_checkpoint(
            checkpoint,
            copied_checkpoint,
            redactor=redactor,
            consume_on_rejection=consume_on_rejection,
        )
    finally:
        # The inner parser clears rejected private data. Do not retain the
        # caller-owned source in this wrapper's exception traceback.
        checkpoint = None
        copied_checkpoint = None


def _pending_approval_from_owned_checkpoint(
    checkpoint: dict[str, Any] | None,
    copied_checkpoint: dict[str, Any],
    *,
    redactor: SecretRedactor | None = None,
    consume_on_rejection: bool = False,
) -> PendingToolApproval | None:
    """Parse an immediately owned, validated snapshot; never retain or cache it."""
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
        if consume_on_rejection and checkpoint is not None:
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
        if consume_on_rejection and checkpoint is not None:
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
