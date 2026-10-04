"""Saved pending-tool-round state and validation shared with checkpoint readers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_durable_metadata,
    require_clean_nonblank,
    require_durable_text,
)
from cayu.approvals.tools import PendingToolCallApproval, copy_distinct_pending_tool_call_approvals
from cayu.budgets._run_limit_accounting import (
    RunLimitAccountingContext,
    has_run_limit_accounting_authority,
)
from cayu.budgets.base import BudgetLimit, copy_request_budget_limits
from cayu.budgets.run_limits import RunLimits, copy_run_limits
from cayu.configuration import MAX_STEPS
from cayu.context.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
    StructuredOutputValidation,
    copy_structured_output_spec,
)
from cayu.context.thinking import ThinkingConfig
from cayu.messages import Message, detach_message
from cayu.runtime.execution_units import ToolRoundIdentity
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.sessions._assistant_tool_round_publication import (
    AssistantToolRoundPublication,
    StagedToolCallTerminal,
    validate_staged_tool_exposure_terminal,
)
from cayu.tools._policy_evidence import ToolPolicyEvidence
from cayu.tools.catalogue import CALL_TOOL_NAME, SEARCH_TOOLS_NAME
from cayu.tools.exposure import ResolvedToolExposureAuthority, copy_resolved_tool_exposure_authority

PENDING_TOOL_ROUND_CHECKPOINT_KEY = "pending_tool_round"
# Only checkpoint parsers supply this marker, after owning a complete durable
# JSON document. Nested models then originate in plain JSON, not caller-owned
# instances, and need no second dump/validate cycle merely to detach them.
_OWNED_ROUND_JSON_CONTEXT = object()


class PendingToolRound(BaseModel):
    """Durable checkpoint state for an ordinary tool round in progress."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    tool_round_id: str
    model_step_id: str
    model_attempt_id: str
    agent_name: str
    interaction_id: str | None = None
    environment_name: str | None = None
    task_id: str | None = None
    source_run_epoch: StrictInt | None = Field(
        default=None,
        ge=1,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    execution_profile_fingerprint: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    tool_exposure: ResolvedToolExposureAuthority | None = None
    tool_calls: list[PendingToolCallApproval]
    policy_state: Literal["unplanned", "planned"] = "unplanned"
    policy_context_version: Literal[1] | None = None
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    deferred_messages: list[Message] = Field(default_factory=list)
    assistant_message_state: Literal["published", "quarantined"] = "published"
    quarantined_assistant_message: Message | None = None
    assistant_publication: AssistantToolRoundPublication | None = None
    staged_terminals: list[StagedToolCallTerminal] = Field(default_factory=list)
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=MAX_STEPS)
    limits: RunLimits | None = None
    run_limit_accounting: RunLimitAccountingContext | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    source_model_step_id: str | None = None
    source_transcript_cursor: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    model_step: StrictInt | None = Field(
        default=None,
        ge=1,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    structured_output_attempt: StrictInt | None = Field(
        default=None,
        ge=1,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    structured_output_retries: StrictInt = Field(
        default=0,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    structured_output_validation: StructuredOutputValidation | None = None

    @field_validator("agent_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(
            require_durable_text(value, info.field_name),
            info.field_name,
        )

    @model_validator(mode="after")
    def validate_tool_round_identity(self) -> PendingToolRound:
        ToolRoundIdentity(
            tool_round_id=self.tool_round_id,
            model_step_id=self.model_step_id,
            model_attempt_id=self.model_attempt_id,
        )
        if self.run_limit_accounting is not None and not has_run_limit_accounting_authority(
            self.limits,
            self.budget_limits,
        ):
            raise ValueError("run_limit_accounting requires active run-scoped authority.")
        has_targeted_call = any(
            call.tool_name == CALL_TOOL_NAME
            or call.targeted_tool_grant_id is not None
            or call.targeted_tool_invocation is not None
            or call.targeted_tool_rejection is not None
            for call in self.tool_calls
        )
        if has_targeted_call != (self.interaction_id is not None):
            raise ValueError(
                "Pending targeted calls and interaction identity authority must be present "
                "together."
            )
        return self

    @field_validator("interaction_id", "environment_name", "task_id", "source_model_step_id")
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(
            require_durable_text(value, info.field_name),
            info.field_name,
        )

    @field_validator("tool_calls")
    @classmethod
    def copy_tool_calls(
        cls,
        value: list[PendingToolCallApproval],
        info: ValidationInfo,
    ) -> list[PendingToolCallApproval]:
        if info.context is _OWNED_ROUND_JSON_CONTEXT:
            if not value:
                raise ValueError("Pending tool round must include tool calls.")
            ids = [call.tool_call_id for call in value]
            if len(ids) != len(set(ids)):
                raise ValueError("Pending tool round contains duplicate tool-call identities.")
            return value
        return copy_distinct_pending_tool_call_approvals(
            value,
            owner="Pending tool round",
        )

    @field_validator("request_metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_metadata(value, "request_metadata")

    @field_validator("deferred_messages")
    @classmethod
    def copy_deferred_messages(cls, value: list[Message]) -> list[Message]:
        return [detach_message(message) for message in value]

    @field_validator("quarantined_assistant_message")
    @classmethod
    def copy_quarantined_assistant_message(
        cls,
        value: Message | None,
    ) -> Message | None:
        return None if value is None else detach_message(value)

    @field_validator("staged_terminals")
    @classmethod
    def copy_staged_terminals(
        cls,
        value: list[StagedToolCallTerminal],
        info: ValidationInfo,
    ) -> list[StagedToolCallTerminal]:
        copied = (
            value
            if info.context is _OWNED_ROUND_JSON_CONTEXT
            else [
                StagedToolCallTerminal.model_validate(item.model_dump(mode="json"))
                for item in value
            ]
        )
        ids = [item.tool_call_id for item in copied]
        if len(ids) != len(set(ids)):
            raise ValueError("Pending tool round cannot repeat staged terminal calls.")
        event_ids = [item.event.id for item in copied]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Pending tool round cannot repeat staged terminal event ids.")
        return copied

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("tool_exposure")
    @classmethod
    def copy_tool_exposure(
        cls,
        value: ResolvedToolExposureAuthority | None,
    ) -> ResolvedToolExposureAuthority | None:
        if value is None:
            return None
        return copy_resolved_tool_exposure_authority(value)

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

    @field_validator("retry_policy")
    @classmethod
    def copy_retry_policy(cls, value: RetryPolicy | None) -> RetryPolicy | None:
        if value is None:
            return None
        return copy_retry_policy(value)

    @field_validator("structured_output_validation")
    @classmethod
    def copy_structured_output_validation(
        cls,
        value: StructuredOutputValidation | None,
    ) -> StructuredOutputValidation | None:
        if value is None:
            return None
        return value.model_copy(deep=True)

    @model_validator(mode="after")
    def validate_model_step_link(self) -> PendingToolRound:
        source_fields = (
            self.source_model_step_id,
            self.source_transcript_cursor,
            self.model_step,
        )
        if any(value is not None for value in source_fields) and any(
            value is None for value in source_fields
        ):
            raise ValueError(
                "Pending tool-round model-step identity fields must be supplied together."
            )
        if self.structured_output_attempt is not None and (
            self.structured_output is None
            or not any(call.tool_name == STRUCTURED_OUTPUT_TOOL_NAME for call in self.tool_calls)
        ):
            raise ValueError(
                "structured_output_attempt requires a structured-output finalizer call."
            )
        if self.structured_output_validation is not None and (
            self.structured_output_attempt is None
            or self.structured_output is None
            or not any(call.tool_name == STRUCTURED_OUTPUT_TOOL_NAME for call in self.tool_calls)
        ):
            raise ValueError(
                "structured_output_validation requires a structured-output finalizer attempt."
            )
        if self.structured_output_validation is not None:
            validation = self.structured_output_validation
            if validation.valid and validation.errors:
                raise ValueError(
                    "Valid structured-output evidence cannot contain validation errors."
                )
            if not validation.valid and (validation.output is not None or not validation.errors):
                raise ValueError(
                    "Invalid structured-output evidence requires errors and no output."
                )
        if self.structured_output_attempt is not None and self.structured_output_validation is None:
            raise ValueError(
                "A structured-output finalizer attempt requires authoritative validation."
            )
        if self.structured_output_retries and self.structured_output is None:
            raise ValueError("Structured-output retry state requires a structured-output contract.")
        if self.policy_state == "planned" and self.policy_context_version != 1:
            raise ValueError("A planned pending tool round requires policy context version 1.")
        if self.policy_state == "unplanned":
            if any(
                call.policy_evidence not in {None, ToolPolicyEvidence.UNPLANNED}
                or (
                    call.policy_evidence is ToolPolicyEvidence.UNPLANNED
                    and call.policy_decision is not None
                )
                for call in self.tool_calls
            ):
                raise ValueError("An unplanned pending tool round cannot contain policy authority.")
        elif any(call.policy_evidence is ToolPolicyEvidence.UNPLANNED for call in self.tool_calls):
            raise ValueError("A planned pending tool round cannot contain unplanned calls.")
        unexposed_calls = [
            call for call in self.tool_calls if call.policy_evidence is ToolPolicyEvidence.UNEXPOSED
        ]
        if unexposed_calls and self.tool_exposure is None:
            raise ValueError("Unexposed tool calls require a frozen exposure snapshot.")
        if self.tool_exposure is not None:
            exposed_names = frozenset((*self.tool_exposure.tool_names, SEARCH_TOOLS_NAME))
            if any(call.tool_name in exposed_names for call in unexposed_calls):
                raise ValueError("Unexposed tool-call evidence conflicts with the snapshot.")
            if self.policy_state == "planned" and any(
                call.policy_evidence is ToolPolicyEvidence.AUTHORITATIVE
                and call.tool_name not in exposed_names
                and call.targeted_tool_invocation is None
                for call in self.tool_calls
            ):
                raise ValueError(
                    "Authoritative tool-policy evidence names a tool outside the snapshot."
                )
        if self.assistant_message_state == "quarantined":
            if self.quarantined_assistant_message is None:
                raise ValueError(
                    "A quarantined pending tool round requires its private assistant message."
                )
        elif self.quarantined_assistant_message is not None:
            raise ValueError(
                "A published pending tool round cannot retain a quarantined assistant message."
            )
        if self.assistant_message_state == "published" and self.assistant_publication is not None:
            raise ValueError(
                "A published pending tool round cannot retain assistant publication state."
            )
        if self.assistant_publication is not None:
            expected_ids = {call.tool_call_id for call in self.tool_calls}
            covered_ids = set(self.assistant_publication.covered_tool_call_ids)
            if not covered_ids <= expected_ids:
                raise ValueError("Assistant publication covers a call outside its pending round.")
            expected_state = "ready" if covered_ids == expected_ids else "pending"
            if (
                self.assistant_publication.state != "blocked"
                and self.assistant_publication.state != expected_state
            ):
                raise ValueError("Assistant publication readiness conflicts with call coverage.")
        expected_ids = {call.tool_call_id for call in self.tool_calls}
        if any(item.tool_call_id not in expected_ids for item in self.staged_terminals):
            raise ValueError("Staged terminal evidence names a call outside its pending round.")
        identity = pending_tool_round_identity(self)
        calls_by_id = {call.tool_call_id: call for call in self.tool_calls}
        for item in self.staged_terminals:
            event = item.event
            if not identity.matches_payload(event.payload):
                raise ValueError("Staged terminal evidence has a conflicting round identity.")
            call = calls_by_id[item.tool_call_id]
            if event.tool_name != call.tool_name:
                raise ValueError("Staged terminal evidence has a conflicting tool name.")
            validate_staged_tool_exposure_terminal(
                item,
                policy_evidence=call.policy_evidence,
                tool_exposure=self.tool_exposure,
            )
        return self


def pending_tool_round_identity(pending_round: PendingToolRound) -> ToolRoundIdentity:
    if type(pending_round) is not PendingToolRound:
        raise TypeError("Pending tool round must be a PendingToolRound.")
    return ToolRoundIdentity(
        tool_round_id=pending_round.tool_round_id,
        model_step_id=pending_round.model_step_id,
        model_attempt_id=pending_round.model_attempt_id,
    )
