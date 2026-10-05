"""Read saved approvals and interpret their pending-action evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import copy_durable_json_value, require_durable_clean_nonblank
from cayu.approvals.tools import (
    PendingToolApproval,
    PendingToolCallApproval,
    ToolApprovalDecision,
    ToolPolicyEvidence,
)
from cayu.runtime.execution_units import ToolRoundIdentity
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions._checkpoint_secret_validation import durable_value_contains_secret
from cayu.tools.policy import ToolPolicyDecision
from cayu.vaults.redaction import SecretRedactor, contains_redacted_secret

PENDING_TOOL_APPROVAL_CHECKPOINT_KEY = "pending_tool_approval"
APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY = "approval_resolution_intent"


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


class ApprovalResolutionIntent(BaseModel):
    """Immutable resolution authority retained with one pending approval."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    approval_id: str
    tool_call_id: str
    tool_round_id: str
    model_step_id: str
    model_attempt_id: str
    decision: ToolApprovalDecision
    pause_resolved_at: datetime | None = Field(default=None, exclude_if=lambda value: value is None)
    # ``None`` loads checkpoints written before request digests existed. It is
    # intentionally non-authoritative and must never be upgraded after the fact.
    resolution_request_digest: str | None = None
    # Reviewed tool arguments remain immutable even as the paired round's
    # publication coverage advances. Legacy/unreviewed claims do not acquire
    # this authority retroactively.
    reviewed_approval_digest: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("pause_resolved_at")
    @classmethod
    def validate_pause_resolved_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("pause_resolved_at must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("approval_id", "tool_call_id")
    @classmethod
    def validate_nonblank_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("resolution_request_digest", "reviewed_approval_digest")
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
