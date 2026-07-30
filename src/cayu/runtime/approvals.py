from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema  # noqa: TC002 - Pydantic needs this at runtime.

from cayu._validation import (
    copy_durable_json_value,
    require_durable_clean_nonblank,
    require_durable_nonblank,
)
from cayu.core.events import Event, EventType
from cayu.core.thinking import ThinkingConfig
from cayu.runtime._policy_evidence import ToolPolicyEvidence
from cayu.runtime.budgets import BudgetLimit, copy_budget_limits, copy_request_budget_limits
from cayu.runtime.execution_units import ToolRoundIdentity
from cayu.runtime.loop_policies import LoopPolicy, validate_loop_policies
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.runtime.structured_output import StructuredOutputSpec, copy_structured_output_spec

RESOLUTION_ACTOR_RESERVED_SUBJECT_PREFIX = "cayu:"
EXPIRY_RESOLUTION_ACTOR_SUBJECT = "cayu:approval-expiry"


class ToolApprovalDecision(StrEnum):
    APPROVE = "approve"
    DENY = "deny"


class ToolApprovalRecoveryOutcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class ResolutionActorSource(StrEnum):
    """How a ``ResolutionActor``'s identity claim was established.

    ``HTTP_AUTH`` is produced only by the server layer from a verified
    ``AuthContext``; ``REQUEST`` marks a caller-asserted identity (SDK or
    open-access HTTP body); ``SYSTEM`` marks runtime-generated actors such as
    deterministic approval expiry. Direct SDK callers are a trusted boundary
    and may construct system actors; HTTP bodies cannot — the server re-stamps
    open-access bodies to ``REQUEST`` and rejects them entirely under auth.
    """

    HTTP_AUTH = "http_auth"
    REQUEST = "request"
    SYSTEM = "system"


class ResolutionActor(BaseModel):
    """Typed actor identity for trusted runtime operator actions.

    Stamped into approval, user-input, recovery, and interruption event payloads
    so the audit trail answers who performed an operator action without
    consulting app-side state. ``reason`` and ``metadata`` on requests remain
    caller-claimed free-form data; this model is the provenance field.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    subject: str
    tenant: str | None = None
    source: ResolutionActorSource | None = None
    claims: dict[str, Any] = Field(default_factory=dict)

    @field_validator("subject", "tenant")
    @classmethod
    def validate_nonblank_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("claims", mode="before")
    @classmethod
    def copy_claims(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_value(value, "claims")

    @model_validator(mode="after")
    def validate_reserved_subject(self) -> ResolutionActor:
        if (
            self.subject.startswith(RESOLUTION_ACTOR_RESERVED_SUBJECT_PREFIX)
            and self.source != ResolutionActorSource.SYSTEM
        ):
            raise ValueError(
                "ResolutionActor subjects prefixed "
                f"{RESOLUTION_ACTOR_RESERVED_SUBJECT_PREFIX!r} are reserved for system actors."
            )
        return self


def copy_resolution_actor(actor: ResolutionActor | None) -> ResolutionActor | None:
    if actor is None:
        return None
    if type(actor) is not ResolutionActor:
        raise TypeError("Resolution actors must be ResolutionActor instances.")
    return ResolutionActor(
        subject=actor.subject,
        tenant=actor.tenant,
        source=actor.source,
        claims=copy_durable_json_value(actor.claims, "claims"),
    )


def expiry_resolution_actor() -> ResolutionActor:
    """The system actor stamped on deterministic approval-expiry resolutions."""

    return ResolutionActor(
        subject=EXPIRY_RESOLUTION_ACTOR_SUBJECT,
        source=ResolutionActorSource.SYSTEM,
    )


def resolution_actor_payload(actor: ResolutionActor | None) -> dict[str, Any] | None:
    """JSON-safe event payload form of an actor (``None`` stays ``None``).

    ``claims`` are deliberately excluded: they carry deployment authorization
    state (scopes/roles) for in-process use on the request, and nothing
    redacts durable event payloads. The audit trail's who/how is
    ``subject``/``tenant``/``source``.
    """

    if actor is None:
        return None
    return actor.model_dump(mode="json", exclude={"claims"})


class ToolApprovalRequest(BaseModel):
    """Caller decision for a pending tool approval.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default
    to ``None``, which means "inherit the original run's configuration" as
    persisted on the pending approval checkpoint. Passing an explicit value
    overrides the persisted configuration for the resumed run.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        hide_input_in_errors=True,
    )

    session_id: str
    approval_id: str
    tool_round_id: str
    tool_call_id: str
    decision: ToolApprovalDecision
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    loop_policies: SkipJsonSchema[tuple[LoopPolicy, ...]] = Field(
        default_factory=tuple,
        exclude=True,
    )

    @field_validator("session_id", "approval_id", "tool_round_id", "tool_call_id")
    @classmethod
    def validate_nonblank_ids(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_optional_reason(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_value(value, "metadata")

    @field_validator("resolved_by")
    @classmethod
    def copy_resolved_by(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("limits")
    @classmethod
    def copy_limits(cls, value: RunLimits | None) -> RunLimits | None:
        if value is None:
            return None
        return copy_run_limits(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_request_budget_limits(value)

    @field_validator("loop_policies", mode="before")
    @classmethod
    def copy_loop_policies(cls, value) -> tuple[LoopPolicy, ...]:
        return validate_loop_policies(value, field_name="loop_policies")


class ToolApprovalRecoveryRequest(BaseModel):
    """Caller-supplied terminal outcome for an approved tool with unknown result.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default
    to ``None``, which means "inherit the original run's configuration" as
    persisted on the pending approval checkpoint. Passing an explicit value
    overrides the persisted configuration for the resumed run.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        hide_input_in_errors=True,
    )

    session_id: str
    approval_id: str
    tool_round_id: str
    tool_call_id: str
    outcome: ToolApprovalRecoveryOutcome
    message: str
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    loop_policies: SkipJsonSchema[tuple[LoopPolicy, ...]] = Field(
        default_factory=tuple,
        exclude=True,
    )

    @field_validator("session_id", "approval_id", "tool_round_id", "tool_call_id")
    @classmethod
    def validate_nonblank_ids(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str, info) -> str:
        return require_durable_nonblank(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_optional_reason(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_durable_nonblank(value, info.field_name)

    @field_validator("structured", "artifacts", "metadata", mode="before")
    @classmethod
    def copy_json_fields(cls, value, info):
        return copy_durable_json_value(value, info.field_name)

    @field_validator("resolved_by")
    @classmethod
    def copy_resolved_by(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("limits")
    @classmethod
    def copy_limits(cls, value: RunLimits | None) -> RunLimits | None:
        if value is None:
            return None
        return copy_run_limits(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_request_budget_limits(value)

    @field_validator("loop_policies", mode="before")
    @classmethod
    def copy_loop_policies(cls, value) -> tuple[LoopPolicy, ...]:
        return validate_loop_policies(value, field_name="loop_policies")


class PendingToolCallApproval(BaseModel):
    """One tool call captured inside a pending approval round."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # ``None`` is accepted only as a legacy representation. Callers must use
    # ``effective_tool_policy_evidence`` before making an authority decision.
    policy_evidence: ToolPolicyEvidence | None = None
    policy_decision: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Taint labels active for this call when the round paused, so the resumed tool sees the same
    # taint the policy gated it with (a run-request seed would otherwise not survive the checkpoint).
    active_taint_labels: list[str] = Field(default_factory=list)

    @field_validator("tool_call_id", "tool_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("active_taint_labels", mode="before")
    @classmethod
    def validate_active_taint_labels(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("active_taint_labels must be a list of strings.")
        return [require_durable_clean_nonblank(item, "active_taint_labels") for item in value]

    @field_validator("policy_decision", "reason")
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        if info.field_name == "reason":
            return require_durable_nonblank(value, info.field_name)
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("arguments", "metadata", mode="before")
    @classmethod
    def copy_json_fields(cls, value: dict[str, Any], info) -> dict[str, Any]:
        return copy_durable_json_value(value, info.field_name)

    @model_validator(mode="after")
    def validate_policy_evidence(self) -> PendingToolCallApproval:
        recognized_decisions = {
            "allow",
            "deny",
            "require_approval",
        }
        if (
            self.policy_evidence is ToolPolicyEvidence.AUTHORITATIVE
            and self.policy_decision not in recognized_decisions
        ):
            raise ValueError("Authoritative tool-policy evidence requires a recognized decision.")
        if (
            self.policy_evidence
            in {
                ToolPolicyEvidence.UNPLANNED,
                ToolPolicyEvidence.UNREGISTERED,
                ToolPolicyEvidence.AMBIGUOUS,
            }
            and self.policy_decision is not None
        ):
            raise ValueError(
                "Non-authoritative tool-policy evidence cannot carry a policy decision."
            )
        return self


class PendingToolApproval(BaseModel):
    """Durable checkpoint state for a tool call waiting on caller approval.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` persist
    the original run's configuration across the approval pause so resolving
    the approval resumes with the same config instead of fresh defaults
    (unless the resolution request overrides them explicitly). They are
    optional so checkpoints written before this state existed still load.
    Loop policies are runtime callables and cannot be checkpointed; they must
    be re-supplied on the resolution request when needed.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    approval_id: str
    tool_round_id: str
    model_step_id: str
    model_attempt_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    agent_name: str
    environment_name: str | None = None
    workspace_id: str | None = None
    task_id: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[PendingToolCallApproval]
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    expires_at: datetime | None = None

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expires_at must be timezone-aware.")
        return value.astimezone(UTC)

    @classmethod
    def from_event(cls, event: Event) -> PendingToolApproval:
        """Build a ``PendingToolApproval`` from a ``tool.call.approval_requested`` event.

        The complete approval remains nested under ``"approval"`` while its
        lookup and execution-unit identities are repeated as first-class fields.
        Both representations must agree.
        """
        if event.type != EventType.TOOL_CALL_APPROVAL_REQUESTED:
            raise ValueError(
                "PendingToolApproval.from_event expects a "
                f"{EventType.TOOL_CALL_APPROVAL_REQUESTED.value} event, got {str(event.type)!r}."
            )
        approval = (event.payload or {}).get("approval")
        if not isinstance(approval, dict):
            raise ValueError(
                "Event payload has no 'approval' object; expected a "
                "tool.call.approval_requested event payload."
            )
        pending = cls.model_validate(approval)
        expected = {
            "approval_id": pending.approval_id,
            "tool_call_id": pending.tool_call_id,
            "model_step_id": pending.model_step_id,
            "model_attempt_id": pending.model_attempt_id,
            "tool_round_id": pending.tool_round_id,
        }
        if any(event.payload.get(key) != value for key, value in expected.items()):
            raise ValueError("Approval-request event identity does not match its nested approval.")
        pending_tool_call_for_approval_event(
            event=event,
            approval=pending,
        )
        return pending

    @field_validator("approval_id", "tool_call_id", "tool_name", "agent_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_tool_round_identity(self) -> PendingToolApproval:
        ToolRoundIdentity(
            tool_round_id=self.tool_round_id,
            model_step_id=self.model_step_id,
            model_attempt_id=self.model_attempt_id,
        )
        gating_calls = [call for call in self.tool_calls if call.tool_call_id == self.tool_call_id]
        if len(gating_calls) != 1:
            raise ValueError(
                "Pending tool approval must identify exactly one call in its tool round."
            )
        gating_call = gating_calls[0]
        if gating_call.tool_name != self.tool_name or gating_call.arguments != self.arguments:
            raise ValueError(
                "Pending tool approval call details do not match its tool-round record."
            )
        return self

    @field_validator("environment_name", "workspace_id", "task_id", "reason")
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        if info.field_name == "reason":
            return require_durable_nonblank(value, info.field_name)
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("arguments", "metadata", mode="before")
    @classmethod
    def copy_json_fields(cls, value: dict[str, Any], info) -> dict[str, Any]:
        return copy_durable_json_value(value, info.field_name)

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("tool_calls")
    @classmethod
    def copy_tool_calls(
        cls,
        value: list[PendingToolCallApproval],
    ) -> list[PendingToolCallApproval]:
        return copy_distinct_pending_tool_call_approvals(
            value,
            owner="Pending tool approval",
        )

    @field_validator("limits")
    @classmethod
    def copy_limits(cls, value: RunLimits | None) -> RunLimits | None:
        if value is None:
            return None
        return copy_run_limits(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_budget_limits(value, field_name="budget_limits")

    @field_validator("retry_policy")
    @classmethod
    def copy_retry(cls, value: RetryPolicy | None) -> RetryPolicy | None:
        if value is None:
            return None
        return copy_retry_policy(value)


_PENDING_TOOL_APPROVAL_EVENT_PROJECTION_KEYS = tuple(PendingToolApproval.model_fields)


def _copy_approval_resume_fields(
    request: ToolApprovalRequest | ToolApprovalRecoveryRequest,
) -> dict[str, Any]:
    """Copy the run configuration shared by both approval continuation requests."""

    return {
        "reason": request.reason,
        "metadata": copy_durable_json_value(request.metadata, "metadata"),
        "resolved_by": copy_resolution_actor(request.resolved_by),
        "max_steps": request.max_steps,
        "limits": copy_run_limits(request.limits) if request.limits is not None else None,
        "budget_limits": (
            copy_request_budget_limits(request.budget_limits)
            if request.budget_limits is not None
            else None
        ),
        "retry_policy": (copy_retry_policy(request.retry_policy) if request.retry_policy else None),
        "structured_output": copy_structured_output_spec(request.structured_output),
        "thinking": request.thinking,
        "loop_policies": validate_loop_policies(
            request.loop_policies,
            field_name="loop_policies",
        ),
    }


def copy_tool_approval_request(request: ToolApprovalRequest) -> ToolApprovalRequest:
    if type(request) is not ToolApprovalRequest:
        raise TypeError("Tool approval resolution requires a ToolApprovalRequest.")
    return ToolApprovalRequest(
        session_id=request.session_id,
        approval_id=request.approval_id,
        tool_round_id=request.tool_round_id,
        tool_call_id=request.tool_call_id,
        decision=request.decision,
        **_copy_approval_resume_fields(request),
    )


def copy_tool_approval_recovery_request(
    request: ToolApprovalRecoveryRequest,
) -> ToolApprovalRecoveryRequest:
    if type(request) is not ToolApprovalRecoveryRequest:
        raise TypeError("Tool approval recovery requires a ToolApprovalRecoveryRequest.")
    return ToolApprovalRecoveryRequest(
        session_id=request.session_id,
        approval_id=request.approval_id,
        tool_round_id=request.tool_round_id,
        tool_call_id=request.tool_call_id,
        outcome=request.outcome,
        message=request.message,
        structured=copy_durable_json_value(request.structured, "structured"),
        artifacts=copy_durable_json_value(request.artifacts, "artifacts"),
        **_copy_approval_resume_fields(request),
    )


def copy_pending_tool_approval(approval: PendingToolApproval) -> PendingToolApproval:
    if type(approval) is not PendingToolApproval:
        raise TypeError("Pending tool approval must be a PendingToolApproval.")
    return PendingToolApproval(
        approval_id=approval.approval_id,
        tool_round_id=approval.tool_round_id,
        model_step_id=approval.model_step_id,
        model_attempt_id=approval.model_attempt_id,
        tool_call_id=approval.tool_call_id,
        tool_name=approval.tool_name,
        arguments=copy_durable_json_value(approval.arguments, "arguments"),
        agent_name=approval.agent_name,
        environment_name=approval.environment_name,
        workspace_id=approval.workspace_id,
        task_id=approval.task_id,
        reason=approval.reason,
        metadata=copy_durable_json_value(approval.metadata, "metadata"),
        tool_calls=[copy_pending_tool_call_approval(call) for call in approval.tool_calls],
        structured_output=copy_structured_output_spec(approval.structured_output),
        thinking=approval.thinking,
        max_steps=approval.max_steps,
        limits=copy_run_limits(approval.limits) if approval.limits is not None else None,
        budget_limits=(
            copy_budget_limits(approval.budget_limits, field_name="budget_limits")
            if approval.budget_limits is not None
            else None
        ),
        retry_policy=copy_retry_policy(approval.retry_policy) if approval.retry_policy else None,
        expires_at=approval.expires_at,
    )


def copy_pending_tool_call_approval(
    call: PendingToolCallApproval,
) -> PendingToolCallApproval:
    if type(call) is not PendingToolCallApproval:
        raise TypeError("Pending tool call approval must be a PendingToolCallApproval.")
    return PendingToolCallApproval(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        arguments=copy_durable_json_value(call.arguments, "arguments"),
        policy_evidence=call.policy_evidence,
        policy_decision=call.policy_decision,
        reason=call.reason,
        metadata=copy_durable_json_value(call.metadata, "metadata"),
        active_taint_labels=list(call.active_taint_labels),
    )


def pending_tool_call_for_approval_event(
    *,
    event: Event,
    approval: PendingToolApproval,
) -> PendingToolCallApproval:
    """Return the exact pending call described by one approval lifecycle event."""

    tool_call_id = event.payload.get("tool_call_id")
    pending_call = next(
        (
            call
            for call in approval.tool_calls
            if type(tool_call_id) is str and call.tool_call_id == tool_call_id
        ),
        None,
    )
    if (
        pending_call is None
        or event.tool_name != pending_call.tool_name
        or event.agent_name != approval.agent_name
        or event.environment_name != approval.environment_name
    ):
        raise ValueError(
            "Approval lifecycle event descriptor does not match its pending tool call."
        )
    return pending_call


def copy_distinct_pending_tool_call_approvals(
    calls: list[PendingToolCallApproval],
    *,
    owner: str,
) -> list[PendingToolCallApproval]:
    copied = [copy_pending_tool_call_approval(call) for call in calls]
    if not copied:
        raise ValueError(f"{owner} must include tool calls.")
    tool_call_ids = [call.tool_call_id for call in copied]
    if len(tool_call_ids) != len(set(tool_call_ids)):
        raise ValueError(f"{owner} contains duplicate tool-call identities.")
    return copied
