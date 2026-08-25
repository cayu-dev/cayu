from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema  # noqa: TC002 - Pydantic needs this at runtime.

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_durable_json_value,
    require_durable_clean_nonblank,
    require_durable_nonblank,
)
from cayu.core.events import (
    Event,
    event_with_runtime_nested_payload_authority,
    event_with_runtime_payload_authority,
)
from cayu.core.messages import Message, detach_message
from cayu.core.thinking import ThinkingConfig
from cayu.runtime._assistant_tool_round_publication import (
    AssistantToolRoundPublication,
    StagedToolCallTerminal,
    copy_assistant_tool_round_publication,
    validate_staged_tool_exposure_terminal,
)
from cayu.runtime._checkpoint_redaction import durable_value_contains_secret
from cayu.runtime._policy_evidence import ToolPolicyEvidence
from cayu.runtime._run_limit_accounting import (
    RunLimitAccountingContext,
    has_run_limit_accounting_authority,
)
from cayu.runtime.approvals import (
    PendingToolCallApproval,
    ResolutionActor,
    ToolApprovalRecoveryOutcome,
    copy_distinct_pending_tool_call_approvals,
    copy_pending_tool_call_approval,
    copy_resolution_actor,
)
from cayu.runtime.budgets import BudgetLimit, copy_budget_limits, copy_request_budget_limits
from cayu.runtime.execution_units import ToolRoundIdentity
from cayu.runtime.loop_policies import LoopPolicy, validate_loop_policies
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.runtime.structured_output import StructuredOutputSpec, copy_structured_output_spec
from cayu.runtime.tool_catalogue import CALL_TOOL_NAME
from cayu.runtime.tool_exposure import (
    ResolvedToolExposureAuthority,
    copy_resolved_tool_exposure_authority,
)
from cayu.vaults import SecretRedactor, contains_redacted_secret

PENDING_USER_INPUT_CHECKPOINT_KEY = "pending_user_input"


class UserInputResponse(BaseModel):
    """Caller-supplied answer that resumes a session paused by ``ask_user``.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default to ``None``,
    which means "inherit the original run's configuration" as persisted on the pending
    user-input checkpoint. An explicit value is accepted only when it preserves the
    invocation's frozen execution profile.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        hide_input_in_errors=True,
    )

    session_id: str
    input_id: str
    answer: str
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
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

    @field_validator("session_id", "input_id")
    @classmethod
    def validate_nonblank_ids(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str, info) -> str:
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


class PendingUserInput(BaseModel):
    """Durable checkpoint state for a session paused on an ``ask_user`` question.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` persist the original
    run's configuration across the pause so resolving the question resumes with the same
    config instead of fresh defaults. A resolution request may restate those values only
    when they preserve the invocation's frozen execution profile. They are optional so
    checkpoints written before this state existed still load.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    input_id: str
    tool_round_id: str
    model_step_id: str
    model_attempt_id: str
    model_step: StrictInt | None = Field(
        default=None,
        ge=1,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    tool_call_id: str
    tool_name: str
    question: str
    options: list[str] = Field(default_factory=list)
    arguments: dict[str, Any] = Field(default_factory=dict)
    agent_name: str
    environment_name: str | None = None
    workspace_id: str | None = None
    task_id: str | None = None
    interaction_id: str | None = None
    execution_profile_fingerprint: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    tool_exposure: ResolvedToolExposureAuthority | None = None
    tool_calls: list[PendingToolCallApproval]
    assistant_message_state: Literal["published", "quarantined"] = "published"
    quarantined_assistant_message: Message | None = None
    assistant_publication: AssistantToolRoundPublication | None = None
    staged_terminals: list[StagedToolCallTerminal] = Field(default_factory=list)
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    run_limit_accounting: RunLimitAccountingContext | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None

    @field_validator("input_id", "tool_call_id", "tool_name", "agent_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_tool_round_identity(self) -> PendingUserInput:
        ToolRoundIdentity(
            tool_round_id=self.tool_round_id,
            model_step_id=self.model_step_id,
            model_attempt_id=self.model_attempt_id,
        )
        gating_calls = [call for call in self.tool_calls if call.tool_call_id == self.tool_call_id]
        if len(gating_calls) != 1:
            raise ValueError("Pending user input must identify exactly one call in its tool round.")
        gating_call = gating_calls[0]
        if gating_call.tool_name != self.tool_name or gating_call.arguments != self.arguments:
            raise ValueError("Pending user-input call details do not match its tool-round record.")
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
        if self.assistant_message_state == "quarantined":
            if self.quarantined_assistant_message is None:
                raise ValueError(
                    "A quarantined pending user input requires its private assistant message."
                )
        elif self.quarantined_assistant_message is not None:
            raise ValueError(
                "A published pending user input cannot retain a quarantined assistant message."
            )
        if self.assistant_message_state == "published" and self.assistant_publication is not None:
            raise ValueError(
                "Published pending user input cannot retain assistant publication state."
            )
        expected_ids = {call.tool_call_id for call in self.tool_calls}
        if any(item.tool_call_id not in expected_ids for item in self.staged_terminals):
            raise ValueError("Staged terminal evidence names a call outside its user-input round.")
        calls_by_id = {call.tool_call_id: call for call in self.tool_calls}
        identity = ToolRoundIdentity(
            tool_round_id=self.tool_round_id,
            model_step_id=self.model_step_id,
            model_attempt_id=self.model_attempt_id,
        )
        event_ids: set[str] = set()
        for item in self.staged_terminals:
            if item.event.id in event_ids:
                raise ValueError("Pending user input cannot repeat staged terminal event ids.")
            event_ids.add(item.event.id)
            if not identity.matches_payload(item.event.payload):
                raise ValueError("Staged terminal evidence has a conflicting round identity.")
            if item.event.tool_name != calls_by_id[item.tool_call_id].tool_name:
                raise ValueError("Staged terminal evidence has a conflicting tool name.")
            call = calls_by_id[item.tool_call_id]
            validate_staged_tool_exposure_terminal(
                item,
                policy_evidence=call.policy_evidence,
                tool_exposure=self.tool_exposure,
            )
        unexposed_calls = [
            call for call in self.tool_calls if call.policy_evidence is ToolPolicyEvidence.UNEXPOSED
        ]
        if unexposed_calls and self.tool_exposure is None:
            raise ValueError("Unexposed user-input siblings require a frozen exposure snapshot.")
        if self.tool_exposure is not None:
            exposed_names = frozenset(self.tool_exposure.tool_names)
            if any(call.tool_name in exposed_names for call in unexposed_calls):
                raise ValueError(
                    "Unexposed user-input sibling evidence conflicts with the snapshot."
                )
            if any(
                call.policy_evidence is ToolPolicyEvidence.AUTHORITATIVE
                and call.tool_name not in exposed_names
                and call.targeted_tool_invocation is None
                for call in self.tool_calls
            ):
                raise ValueError(
                    "Authoritative user-input sibling evidence names a tool outside the snapshot."
                )
        return self

    @field_validator("tool_exposure")
    @classmethod
    def copy_tool_exposure(
        cls,
        value: ResolvedToolExposureAuthority | None,
    ) -> ResolvedToolExposureAuthority | None:
        if value is None:
            return None
        return copy_resolved_tool_exposure_authority(value)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str, info) -> str:
        return require_durable_nonblank(value, info.field_name)

    @field_validator("environment_name", "workspace_id", "task_id", "interaction_id")
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: list[str], info) -> list[str]:
        return [require_durable_nonblank(option, "option") for option in value]

    @field_validator("arguments", mode="before")
    @classmethod
    def copy_arguments(cls, value: dict[str, Any], info) -> dict[str, Any]:
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
            owner="Pending user input",
        )

    @field_validator("quarantined_assistant_message")
    @classmethod
    def copy_quarantined_assistant_message(
        cls,
        value: Message | None,
    ) -> Message | None:
        return None if value is None else detach_message(value)

    @field_validator("assistant_publication")
    @classmethod
    def copy_assistant_publication(
        cls,
        value: AssistantToolRoundPublication | None,
    ) -> AssistantToolRoundPublication | None:
        return copy_assistant_tool_round_publication(value)

    @field_validator("staged_terminals")
    @classmethod
    def copy_staged_terminals(
        cls,
        value: list[StagedToolCallTerminal],
    ) -> list[StagedToolCallTerminal]:
        copied = [
            StagedToolCallTerminal.model_validate(item.model_dump(mode="json")) for item in value
        ]
        ids = [item.tool_call_id for item in copied]
        if len(ids) != len(set(ids)):
            raise ValueError("Pending user input cannot repeat staged terminal calls.")
        return copied

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


_RUNTIME_USER_INPUT_IDENTITY_FIELDS = (
    "input_id",
    "model_step_id",
    "model_attempt_id",
    "tool_round_id",
)
_EXECUTION_PROFILE_FINGERPRINT_FIELD = "execution_profile_fingerprint"


def public_pending_user_input_prompt(
    pending: PendingUserInput,
) -> tuple[str | None, list[str]]:
    """Project a pause prompt only when its durable secret scope is complete.

    The private checkpoint retains the original prompt for continuation.  A
    dynamic, unknown, or legacy scope cannot prove that a sibling invocation
    will not resolve the prompt as a workload secret after the pause was
    published, so public representations must withhold it.
    """

    if type(pending) is not PendingUserInput:
        raise TypeError("pending must be a PendingUserInput.")
    publication = pending.assistant_publication
    if publication is None or publication.secret_resolution_scope != "static":
        return None, []
    return pending.question, list(pending.options)


def public_pending_user_input_event_payload(
    pending: PendingUserInput,
) -> dict[str, Any]:
    """Copy a pending-input payload without unproven prompt or policy output."""

    payload = pending.model_dump(mode="json")
    payload.pop("run_limit_accounting", None)
    # Interaction identity is private checkpoint authority for targeted
    # continuation. The enclosing event carries its own attested interaction
    # envelope, while this public descriptor deliberately omits targeted-call
    # bindings and must not retain only half of that invariant.
    payload.pop("interaction_id", None)
    # The profile reference is runtime authority for the enclosing event, not
    # untrusted pause content. Interruption payloads publish it once at the top
    # level through ``pending_user_input_interruption_payload``.
    payload.pop(_EXECUTION_PROFILE_FINGERPRINT_FIELD, None)
    # Staged terminals are private crash-recovery evidence. They are published
    # only through the terminal event boundary after the round-wide secret
    # scope is finalized, never as part of a pending-input representation.
    payload.pop("staged_terminals", None)
    question, options = public_pending_user_input_prompt(pending)
    tool_calls = payload.get("tool_calls")
    if type(tool_calls) is not list:
        raise TypeError("Pending user-input event payload must contain tool_calls.")
    for pending_call in tool_calls:
        if type(pending_call) is not dict:
            raise TypeError("Pending user-input event tool calls must be objects.")
        pending_call.pop("model_tool_name", None)
        pending_call.pop("targeted_tool_grant_id", None)
        pending_call.pop("targeted_tool_invocation", None)
        pending_call.pop("targeted_tool_rejection", None)
    if question is None:
        payload.pop("question", None)
        payload.pop("options", None)
        for pending_call in tool_calls:
            pending_call.pop("reason", None)
            pending_call.pop("metadata", None)
    else:
        payload["question"] = question
        payload["options"] = options
    return payload


def pending_user_input_interruption_payload(
    pending: PendingUserInput,
) -> dict[str, Any]:
    """Return the bounded public pause descriptor and its profile authority."""

    if type(pending) is not PendingUserInput:
        raise TypeError("pending must be a PendingUserInput.")
    payload: dict[str, Any] = {
        "user_input": public_pending_user_input_event_payload(pending),
    }
    if pending.execution_profile_fingerprint is not None:
        payload[_EXECUTION_PROFILE_FINGERPRINT_FIELD] = pending.execution_profile_fingerprint
    return payload


def event_with_pending_user_input_authority(
    event: Event,
    pending: PendingUserInput,
) -> Event:
    """Attest user-input identities from one validated runtime checkpoint model."""

    if type(pending) is not PendingUserInput:
        raise TypeError("pending must be a PendingUserInput.")
    top_level_fields = tuple(
        field_name
        for field_name in _RUNTIME_USER_INPUT_IDENTITY_FIELDS
        if event.payload.get(field_name) == getattr(pending, field_name)
    )
    if (
        pending.execution_profile_fingerprint is not None
        and event.payload.get(_EXECUTION_PROFILE_FINGERPRINT_FIELD)
        == pending.execution_profile_fingerprint
    ):
        top_level_fields = (*top_level_fields, _EXECUTION_PROFILE_FINGERPRINT_FIELD)
    if top_level_fields:
        event = event_with_runtime_payload_authority(event, *top_level_fields)
    nested = event.payload.get("user_input")
    nested_paths = tuple(
        ("user_input", field_name)
        for field_name in _RUNTIME_USER_INPUT_IDENTITY_FIELDS
        if type(nested) is dict and nested.get(field_name) == getattr(pending, field_name)
    )
    if nested_paths:
        event = event_with_runtime_nested_payload_authority(event, *nested_paths)
    return event


def copy_user_input_response(response: UserInputResponse) -> UserInputResponse:
    if type(response) is not UserInputResponse:
        raise TypeError("User input resolution requires a UserInputResponse.")
    return UserInputResponse(
        session_id=response.session_id,
        input_id=response.input_id,
        answer=response.answer,
        structured=copy_durable_json_value(response.structured, "structured"),
        artifacts=copy_durable_json_value(response.artifacts, "artifacts"),
        metadata=copy_durable_json_value(response.metadata, "metadata"),
        resolved_by=copy_resolution_actor(response.resolved_by),
        max_steps=response.max_steps,
        limits=copy_run_limits(response.limits) if response.limits is not None else None,
        budget_limits=(
            copy_request_budget_limits(response.budget_limits)
            if response.budget_limits is not None
            else None
        ),
        retry_policy=copy_retry_policy(response.retry_policy) if response.retry_policy else None,
        structured_output=copy_structured_output_spec(response.structured_output),
        thinking=response.thinking,
        loop_policies=validate_loop_policies(response.loop_policies, field_name="loop_policies"),
    )


def copy_pending_user_input(pending: PendingUserInput) -> PendingUserInput:
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    return PendingUserInput(
        input_id=pending.input_id,
        tool_round_id=pending.tool_round_id,
        model_step_id=pending.model_step_id,
        model_attempt_id=pending.model_attempt_id,
        model_step=pending.model_step,
        tool_call_id=pending.tool_call_id,
        tool_name=pending.tool_name,
        question=pending.question,
        options=list(pending.options),
        arguments=copy_durable_json_value(pending.arguments, "arguments"),
        agent_name=pending.agent_name,
        environment_name=pending.environment_name,
        workspace_id=pending.workspace_id,
        task_id=pending.task_id,
        interaction_id=pending.interaction_id,
        execution_profile_fingerprint=pending.execution_profile_fingerprint,
        tool_exposure=pending.tool_exposure,
        tool_calls=[copy_pending_tool_call_approval(call) for call in pending.tool_calls],
        assistant_message_state=pending.assistant_message_state,
        quarantined_assistant_message=(
            None
            if pending.quarantined_assistant_message is None
            else detach_message(pending.quarantined_assistant_message)
        ),
        assistant_publication=copy_assistant_tool_round_publication(pending.assistant_publication),
        staged_terminals=[
            StagedToolCallTerminal.model_validate(item.model_dump(mode="json"))
            for item in pending.staged_terminals
        ],
        structured_output=copy_structured_output_spec(pending.structured_output),
        thinking=pending.thinking,
        max_steps=pending.max_steps,
        limits=copy_run_limits(pending.limits) if pending.limits is not None else None,
        run_limit_accounting=pending.run_limit_accounting,
        budget_limits=(
            copy_budget_limits(pending.budget_limits, field_name="budget_limits")
            if pending.budget_limits is not None
            else None
        ),
        retry_policy=copy_retry_policy(pending.retry_policy)
        if pending.retry_policy is not None
        else None,
    )


def pending_user_input_from_checkpoint(
    checkpoint: dict[str, Any] | None,
    *,
    redactor: SecretRedactor | None = None,
    consume_on_rejection: bool = False,
) -> PendingUserInput | None:
    if type(consume_on_rejection) is not bool:
        raise TypeError("consume_on_rejection must be a bool.")
    if checkpoint is None:
        return None
    copied_checkpoint = copy_durable_json_value(checkpoint, "checkpoint")
    value = copied_checkpoint.get(PENDING_USER_INPUT_CHECKPOINT_KEY)
    if value is None:
        return None
    if redactor is not None and durable_value_contains_secret(
        value,
        redactor=redactor,
        path=(PENDING_USER_INPUT_CHECKPOINT_KEY,),
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
            "Pending user-input checkpoint contains a workload secret and cannot be executed."
        ) from None
    if type(value) is not dict:
        raise ValueError("Pending user input checkpoint must be an object.")
    validation_rejected = False
    try:
        pending = PendingUserInput(**value)
    except Exception:
        if redactor is None:
            raise
        validation_rejected = True
    if validation_rejected:
        # A malformed legacy object can place a secret-looking schema field at
        # an invalid position. Clear every private copy after validation, and
        # optionally the runtime-owned source, before raising a fresh error so
        # neither the Pydantic failure nor this traceback retains the payload.
        value.clear()
        value = None
        copied_checkpoint.clear()
        if consume_on_rejection:
            checkpoint.clear()
        checkpoint = None
        raise ValueError(
            "Pending user-input checkpoint is invalid and cannot be executed."
        ) from None
    if contains_redacted_secret(pending.arguments) or any(
        contains_redacted_secret(call.arguments) for call in pending.tool_calls
    ):
        raise ValueError(
            "Pending user-input arguments contain a redaction marker and cannot be executed."
        )
    return pending


class UserInputRecoveryRequest(BaseModel):
    """Caller-supplied terminal outcome for a paused round's tool with an unknown result.

    Used when `resolve_user_input` reports `manual_recovery_required`: a tool in the paused
    round started on a prior resume but recorded no terminal event (a crash mid-tool), so it
    cannot be re-run automatically. The caller supplies the externally verified outcome for
    that `tool_call_id`; `answer` is re-supplied so the `ask_user` result is available if it
    was not already recorded before the crash.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default to ``None`` =
    "inherit the original run's configuration" from the pending checkpoint.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        hide_input_in_errors=True,
    )

    session_id: str
    input_id: str
    answer: str
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

    @field_validator("session_id", "input_id", "tool_call_id")
    @classmethod
    def validate_nonblank_ids(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("resolved_by")
    @classmethod
    def copy_resolved_by(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)

    @field_validator("answer", "message")
    @classmethod
    def validate_nonblank_text(cls, value: str, info) -> str:
        return require_durable_nonblank(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_optional_reason(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_nonblank(value, info.field_name)

    @field_validator("structured", "artifacts", "metadata", mode="before")
    @classmethod
    def copy_json_fields(cls, value, info):
        return copy_durable_json_value(value, info.field_name)

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


def copy_user_input_recovery_request(
    request: UserInputRecoveryRequest,
) -> UserInputRecoveryRequest:
    if type(request) is not UserInputRecoveryRequest:
        raise TypeError("User input recovery requires a UserInputRecoveryRequest.")
    return UserInputRecoveryRequest(
        session_id=request.session_id,
        input_id=request.input_id,
        answer=request.answer,
        tool_call_id=request.tool_call_id,
        outcome=request.outcome,
        message=request.message,
        structured=copy_durable_json_value(request.structured, "structured"),
        artifacts=copy_durable_json_value(request.artifacts, "artifacts"),
        reason=request.reason,
        metadata=copy_durable_json_value(request.metadata, "metadata"),
        resolved_by=copy_resolution_actor(request.resolved_by),
        max_steps=request.max_steps,
        limits=copy_run_limits(request.limits) if request.limits is not None else None,
        budget_limits=(
            copy_request_budget_limits(request.budget_limits)
            if request.budget_limits is not None
            else None
        ),
        retry_policy=copy_retry_policy(request.retry_policy) if request.retry_policy else None,
        structured_output=copy_structured_output_spec(request.structured_output),
        thinking=request.thinking,
        loop_policies=validate_loop_policies(request.loop_policies, field_name="loop_policies"),
    )
