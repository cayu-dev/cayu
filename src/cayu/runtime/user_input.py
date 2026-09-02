from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema  # noqa: TC002 - Pydantic needs this at runtime.

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
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
from cayu.runtime.checkpoints import AMBIGUOUS_PENDING_USER_INPUT_CHECKPOINT_KEY
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
USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY = "user_input_resolution_intent"
USER_INPUT_SUPERSESSION_INTENT_KEY = "user_input_supersession_intent"
AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY = "ambiguous_user_input_supersession_intent"


class UserInputPauseState(StrEnum):
    """Durable lifecycle classification for one exact user-input pause."""

    ACTIVE = "active"
    ANSWERING = "answering"
    ANSWERED = "answered"
    SUPERSEDED = "superseded"
    AMBIGUOUS = "ambiguous"


class AmbiguousUserInputPauseAuthorityError(RuntimeError):
    """A supported checkpoint retains a pause without executable authority."""

    state = UserInputPauseState.AMBIGUOUS

    def __init__(self, source_checkpoint_digest: str) -> None:
        self.source_checkpoint_digest = source_checkpoint_digest
        super().__init__(
            "Pending user input has ambiguous historical authority and cannot be resumed."
        )


class AmbiguousPendingUserInput(BaseModel):
    """Content-free tombstone for a pre-authority supported pause."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    source_checkpoint_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    reason: Literal["missing_exact_pause_authority"]


class AmbiguousUserInputSupersessionIntent(BaseModel):
    """Explicit operator retirement of one ambiguous historical pause."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    session_id: str
    session_instance_id: str
    source_checkpoint_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    state: Literal["ambiguous"] = "ambiguous"

    @field_validator("session_id", "session_instance_id")
    @classmethod
    def validate_nonblank_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


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
    task_worker_id: str | None = None
    task_handoff_id: str | None = None
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

    @field_validator("task_worker_id", "task_handoff_id")
    @classmethod
    def validate_task_worker_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, "task continuation authority")

    @model_validator(mode="after")
    def validate_task_handoff_authority(self) -> UserInputResponse:
        if self.task_handoff_id is not None and self.task_worker_id is None:
            raise ValueError("task_handoff_id requires task_worker_id.")
        return self

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
    when they preserve the invocation's frozen execution profile. Older pauses without
    exact authority migrate to a separate ambiguous tombstone and never instantiate this
    executable model.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    session_id: str
    session_instance_id: str
    source_interaction_id: str
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
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
    execution_profile_fingerprint: str = Field(
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

    @field_validator(
        "session_id",
        "session_instance_id",
        "source_interaction_id",
        "input_id",
        "tool_call_id",
        "tool_name",
        "agent_name",
    )
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
    "tool_call_id",
    "model_step_id",
    "model_attempt_id",
    "tool_round_id",
)
_EXECUTION_PROFILE_FINGERPRINT_FIELD = "execution_profile_fingerprint"


class UserInputResolutionIntent(BaseModel):
    """Immutable request authority retained while one exact pause is resolving."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    session_id: str
    session_instance_id: str
    source_interaction_id: str
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    claim_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    input_id: str
    tool_call_id: str
    tool_round_id: str
    model_step_id: str
    model_attempt_id: str
    execution_profile_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    pause_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    answer_request_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    resolution_stage: Literal["answer", "manual-recovery"]
    execution_state: Literal["claimed", "executing"]
    resolution_request_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @field_validator(
        "session_id",
        "session_instance_id",
        "source_interaction_id",
        "input_id",
        "tool_call_id",
    )
    @classmethod
    def validate_nonblank_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_round_identity(self) -> UserInputResolutionIntent:
        ToolRoundIdentity(
            tool_round_id=self.tool_round_id,
            model_step_id=self.model_step_id,
            model_attempt_id=self.model_attempt_id,
        )
        return self


class UserInputSupersessionIntent(BaseModel):
    """Bounded proof that an operator interruption closed one exact pause."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    session_id: str
    session_instance_id: str
    source_interaction_id: str
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    input_id: str
    tool_call_id: str
    tool_round_id: str
    model_step_id: str
    model_attempt_id: str
    execution_profile_fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    pause_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    state: Literal["active", "answering"]
    claim_run_epoch: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    resolution_request_digest: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @field_validator(
        "session_id",
        "session_instance_id",
        "source_interaction_id",
        "input_id",
        "tool_call_id",
    )
    @classmethod
    def validate_nonblank_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_authority(self) -> UserInputSupersessionIntent:
        ToolRoundIdentity(
            tool_round_id=self.tool_round_id,
            model_step_id=self.model_step_id,
            model_attempt_id=self.model_attempt_id,
        )
        has_claim = self.claim_run_epoch is not None
        has_request = self.resolution_request_digest is not None
        if has_claim != has_request or (self.state == "answering") != has_claim:
            raise ValueError(
                "User-input supersession answering state requires exact answer-claim authority."
            )
        return self


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
    payload.pop("session_id", None)
    payload.pop("session_instance_id", None)
    payload.pop("source_interaction_id", None)
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
        "source_run_epoch": pending.source_run_epoch,
        "pause_digest": pending_user_input_digest(pending),
    }
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
        task_worker_id=response.task_worker_id,
        task_handoff_id=response.task_handoff_id,
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
        schema_version=pending.schema_version,
        session_id=pending.session_id,
        session_instance_id=pending.session_instance_id,
        source_interaction_id=pending.source_interaction_id,
        source_run_epoch=pending.source_run_epoch,
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


def ambiguous_pending_user_input_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> AmbiguousPendingUserInput | None:
    """Load the bounded tombstone for a supported pre-authority pause."""

    if checkpoint is None:
        return None
    value = checkpoint.get(AMBIGUOUS_PENDING_USER_INPUT_CHECKPOINT_KEY)
    if value is None:
        return None
    if PENDING_USER_INPUT_CHECKPOINT_KEY in checkpoint:
        raise ValueError(
            "Checkpoint contains both exact and ambiguous pending user-input authority."
        )
    if type(value) is not dict:
        raise ValueError("Ambiguous pending user-input checkpoint must be an object.")
    try:
        return AmbiguousPendingUserInput.model_validate(value)
    except (TypeError, ValueError):
        raise ValueError("Ambiguous pending user-input checkpoint is malformed.") from None


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
    ambiguous = ambiguous_pending_user_input_from_checkpoint(copied_checkpoint)
    if ambiguous is not None:
        raise AmbiguousUserInputPauseAuthorityError(ambiguous.source_checkpoint_digest) from None
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
    task_worker_id: str | None = None
    task_handoff_id: str | None = None
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

    @field_validator("task_worker_id", "task_handoff_id")
    @classmethod
    def validate_task_worker_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, "task continuation authority")

    @model_validator(mode="after")
    def validate_task_handoff_authority(self) -> UserInputRecoveryRequest:
        if self.task_handoff_id is not None and self.task_worker_id is None:
            raise ValueError("task_handoff_id requires task_worker_id.")
        return self

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
        task_worker_id=request.task_worker_id,
        task_handoff_id=request.task_handoff_id,
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


def pending_user_input_digest(pending: PendingUserInput) -> str:
    """Bind every immutable decision-bearing field of one durable pause.

    Terminal stages and the assistant publication projection are separately
    CAS-owned continuation evidence and may advance while the pause identity
    remains stable.
    """

    if type(pending) is not PendingUserInput:
        raise TypeError("pending must be a PendingUserInput.")
    document = pending.model_dump(mode="json")
    document.pop("staged_terminals", None)
    document.pop("assistant_publication", None)
    return sha256(
        canonical_durable_json_bytes(
            document,
            "pending_user_input",
        )
    ).hexdigest()


def pending_user_input_identity(pending: PendingUserInput) -> dict[str, Any]:
    """Return the complete bounded authority tuple for one pause."""

    if type(pending) is not PendingUserInput:
        raise TypeError("pending must be a PendingUserInput.")
    return {
        "schema_version": 1,
        "session_id": pending.session_id,
        "session_instance_id": pending.session_instance_id,
        "source_interaction_id": pending.source_interaction_id,
        "source_run_epoch": pending.source_run_epoch,
        "input_id": pending.input_id,
        "tool_call_id": pending.tool_call_id,
        "tool_round_id": pending.tool_round_id,
        "model_step_id": pending.model_step_id,
        "model_attempt_id": pending.model_attempt_id,
        "execution_profile_fingerprint": pending.execution_profile_fingerprint,
        "pause_digest": pending_user_input_digest(pending),
    }


def user_input_supersession_intent_for(
    pending: PendingUserInput,
    *,
    resolution_intent: UserInputResolutionIntent | None,
) -> UserInputSupersessionIntent:
    """Construct the exact operator-owned supersession of one active answer state."""

    if resolution_intent is not None:
        require_resolution_intent_matches_pending(resolution_intent, pending=pending)
        if resolution_intent.execution_state != "claimed":
            raise RuntimeError("Executing user-input resolution authority cannot be superseded.")
    return UserInputSupersessionIntent(
        **pending_user_input_identity(pending),
        state="answering" if resolution_intent is not None else "active",
        claim_run_epoch=(None if resolution_intent is None else resolution_intent.claim_run_epoch),
        resolution_request_digest=(
            None if resolution_intent is None else resolution_intent.resolution_request_digest
        ),
    )


def ambiguous_user_input_supersession_intent_for(
    pending: AmbiguousPendingUserInput,
    *,
    session_id: str,
    session_instance_id: str,
) -> AmbiguousUserInputSupersessionIntent:
    """Bind explicit operator retirement to one historical pause tombstone."""

    if type(pending) is not AmbiguousPendingUserInput:
        raise TypeError("pending must be an AmbiguousPendingUserInput.")
    return AmbiguousUserInputSupersessionIntent(
        session_id=session_id,
        session_instance_id=session_instance_id,
        source_checkpoint_digest=pending.source_checkpoint_digest,
    )


def event_with_user_input_supersession_authority(
    event: Event,
    intent: UserInputSupersessionIntent,
) -> Event:
    """Attest the bounded string identities in a validated supersession marker."""

    if type(intent) is not UserInputSupersessionIntent:
        raise TypeError("intent must be a UserInputSupersessionIntent.")
    nested = event.payload.get(USER_INPUT_SUPERSESSION_INTENT_KEY)
    expected = intent.model_dump(mode="json", exclude_none=True)
    if nested != expected:
        raise ValueError("Interruption event supersession marker changed before attestation.")
    paths = tuple(
        (USER_INPUT_SUPERSESSION_INTENT_KEY, field_name)
        for field_name in (
            "session_id",
            "session_instance_id",
            "source_interaction_id",
            "input_id",
            "tool_call_id",
            "tool_round_id",
            "model_step_id",
            "model_attempt_id",
            "execution_profile_fingerprint",
            "pause_digest",
            "state",
            "resolution_request_digest",
        )
        if type(expected.get(field_name)) is str
    )
    return event_with_runtime_nested_payload_authority(event, *paths)


def event_with_ambiguous_user_input_supersession_authority(
    event: Event,
    intent: AmbiguousUserInputSupersessionIntent,
) -> Event:
    """Attest content-free retirement evidence created by the runtime."""

    if type(intent) is not AmbiguousUserInputSupersessionIntent:
        raise TypeError("intent must be an AmbiguousUserInputSupersessionIntent.")
    nested = event.payload.get(AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY)
    expected = intent.model_dump(mode="json")
    if nested != expected:
        raise ValueError("Ambiguous user-input supersession changed before attestation.")
    return event_with_runtime_nested_payload_authority(
        event,
        *(
            (AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY, field_name)
            for field_name in (
                "session_id",
                "session_instance_id",
                "source_checkpoint_digest",
                "state",
            )
        ),
    )


def _user_input_request_payload(
    request: UserInputResponse | UserInputRecoveryRequest,
) -> tuple[UserInputResponse | UserInputRecoveryRequest, dict[str, Any]]:
    """Copy one request and return its shared answer-semantics projection."""

    if type(request) not in (UserInputResponse, UserInputRecoveryRequest):
        raise TypeError("request must be a user-input resolution request.")
    if isinstance(request, UserInputResponse):
        copied: UserInputResponse | UserInputRecoveryRequest = copy_user_input_response(request)
    else:
        assert isinstance(request, UserInputRecoveryRequest)
        copied = copy_user_input_recovery_request(request)
    document = copied.model_dump(
        mode="json",
        include={
            "session_id",
            "input_id",
            "answer",
            "structured",
            "artifacts",
            "metadata",
            "resolved_by",
            "max_steps",
            "limits",
            "budget_limits",
            "retry_policy",
            "structured_output",
            "thinking",
        },
    )
    document["loop_policies"] = [
        {
            "name": require_durable_clean_nonblank(policy.name, "loop_policies.name"),
            "implementation": (
                f"{require_durable_clean_nonblank(type(policy).__module__, 'loop_policies.module')}:"
                f"{require_durable_clean_nonblank(type(policy).__qualname__, 'loop_policies.qualname')}"
            ),
            "replay_identity": policy.adoption_replay_identity,
        }
        for policy in copied.loop_policies
    ]
    return copied, document


def user_input_answer_request_digest(
    request: UserInputResponse | UserInputRecoveryRequest,
) -> str:
    """Bind the answer and continuation semantics shared by both entrances."""

    _copied, document = _user_input_request_payload(request)
    answer_document = {
        field_name: document[field_name]
        for field_name in (
            "session_id",
            "input_id",
            "answer",
            "structured",
            "artifacts",
            "metadata",
            "resolved_by",
        )
    }
    return sha256(
        canonical_durable_json_bytes(answer_document, "user_input_answer_request")
    ).hexdigest()


def user_input_resolution_request_digest(
    request: UserInputResponse | UserInputRecoveryRequest,
) -> str:
    """Bind every caller field that can affect one exact resolution operation."""

    copied, document = _user_input_request_payload(request)
    operation: dict[str, Any] = {
        "kind": "answer" if type(copied) is UserInputResponse else "manual-recovery",
        "answer": document,
    }
    if type(copied) is UserInputRecoveryRequest:
        operation.update(
            {
                "tool_call_id": copied.tool_call_id,
                "outcome": copied.outcome.value,
                "message": copied.message,
                "reason": copied.reason,
            }
        )
    return sha256(
        canonical_durable_json_bytes(operation, "user_input_resolution_request")
    ).hexdigest()


def user_input_resolution_intent_for(
    pending: PendingUserInput,
    *,
    answer_request_digest: str,
    resolution_stage: Literal["answer", "manual-recovery"],
    resolution_request_digest: str,
    claim_run_epoch: int,
    execution_state: Literal["claimed", "executing"] = "claimed",
) -> UserInputResolutionIntent:
    """Construct the immutable resolution claim for an exact pause."""

    identity = pending_user_input_identity(pending)
    return UserInputResolutionIntent(
        **identity,
        claim_run_epoch=claim_run_epoch,
        answer_request_digest=answer_request_digest,
        resolution_stage=resolution_stage,
        execution_state=execution_state,
        resolution_request_digest=resolution_request_digest,
    )


def user_input_resolution_intent_from_checkpoint(
    checkpoint: dict[str, Any] | None,
    *,
    redactor: SecretRedactor | None = None,
) -> UserInputResolutionIntent | None:
    if checkpoint is None:
        return None
    copied = copy_durable_json_value(checkpoint, "checkpoint")
    value = copied.get(USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY)
    if value is None:
        return None
    if redactor is not None and durable_value_contains_secret(
        value,
        redactor=redactor,
        path=(USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY,),
    ):
        raise ValueError(
            "User-input resolution intent contains a workload secret and cannot be executed."
        )
    if type(value) is not dict:
        raise ValueError("User-input resolution intent checkpoint must be an object.")
    try:
        return UserInputResolutionIntent.model_validate(value)
    except Exception:
        if redactor is None:
            raise
        raise ValueError(
            "User-input resolution intent checkpoint is invalid and cannot be executed."
        ) from None


def require_resolution_intent_matches_pending(
    intent: UserInputResolutionIntent,
    *,
    pending: PendingUserInput,
    answer_request_digest: str | None = None,
    resolution_stage: Literal["answer", "manual-recovery"] | None = None,
    resolution_request_digest: str | None = None,
) -> None:
    if type(intent) is not UserInputResolutionIntent:
        raise TypeError("intent must be a UserInputResolutionIntent.")
    expected = user_input_resolution_intent_for(
        pending,
        answer_request_digest=(
            intent.answer_request_digest if answer_request_digest is None else answer_request_digest
        ),
        resolution_stage=(
            intent.resolution_stage if resolution_stage is None else resolution_stage
        ),
        resolution_request_digest=(
            intent.resolution_request_digest
            if resolution_request_digest is None
            else resolution_request_digest
        ),
        claim_run_epoch=intent.claim_run_epoch,
        execution_state=intent.execution_state,
    )
    if intent != expected:
        raise RuntimeError("User-input resolution intent conflicts with its pending pause.")


def user_input_lifecycle_authority_from_checkpoint(
    checkpoint: dict[str, Any] | None,
    *,
    redactor: SecretRedactor | None = None,
    consume_on_rejection: bool = False,
    current_run_epoch: int | None = None,
) -> tuple[PendingUserInput | None, UserInputResolutionIntent | None]:
    """Load one coherent pause/answer-claim topology from a checkpoint.

    An answer claim is never independent authority: it is meaningful only
    while its exact pending pause remains present.  Keep this validation at
    every lifecycle entrance so generic recovery, interruption, and profile
    admission cannot reinterpret an orphan claim as ordinary checkpoint data.
    """

    pending = pending_user_input_from_checkpoint(
        checkpoint,
        redactor=redactor,
        consume_on_rejection=consume_on_rejection,
    )
    intent = user_input_resolution_intent_from_checkpoint(
        checkpoint,
        redactor=redactor,
    )
    if intent is not None:
        if pending is None:
            raise RuntimeError("User-input resolution intent has no exact pending pause authority.")
        require_resolution_intent_matches_pending(intent, pending=pending)
        if intent.claim_run_epoch <= pending.source_run_epoch:
            raise RuntimeError("User-input resolution intent does not follow its pause run epoch.")
    if current_run_epoch is not None:
        if type(current_run_epoch) is not int or current_run_epoch < 0:
            raise TypeError("current_run_epoch must be a non-negative integer or None.")
        if pending is not None:
            expected_epoch = pending.source_run_epoch if intent is None else intent.claim_run_epoch
            # The active owner starts at ``expected_epoch``. Exact lifecycle
            # release and later recovery fences may advance the durable epoch
            # repeatedly while retaining the same pause or answer request, but
            # durable pause authority can never originate in a future epoch.
            if current_run_epoch < expected_epoch:
                raise RuntimeError("User-input lifecycle conflicts with the session run epoch.")
    return pending, intent


def checkpoint_with_user_input_resolution_intent(
    checkpoint: dict[str, Any] | None,
    *,
    pending: PendingUserInput,
    answer_request_digest: str,
    resolution_stage: Literal["answer", "manual-recovery"],
    resolution_request_digest: str,
    claim_run_epoch: int,
    redactor: SecretRedactor,
    allow_answer_to_manual_recovery: bool = False,
    allow_manual_recovery_to_answer: bool = False,
) -> tuple[dict[str, Any], UserInputResolutionIntent]:
    """Set or validate one exact resolution claim under the session transition lock."""

    if type(allow_answer_to_manual_recovery) is not bool:
        raise TypeError("allow_answer_to_manual_recovery must be a boolean.")
    if type(allow_manual_recovery_to_answer) is not bool:
        raise TypeError("allow_manual_recovery_to_answer must be a boolean.")
    copied = {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
    current_pending = pending_user_input_from_checkpoint(copied, redactor=redactor)
    if current_pending != pending:
        raise RuntimeError("Pending user input changed before the answer was claimed.")
    current_intent = user_input_resolution_intent_from_checkpoint(copied, redactor=redactor)
    if current_intent is not None:
        require_resolution_intent_matches_pending(current_intent, pending=pending)
        if current_intent.answer_request_digest != answer_request_digest:
            raise RuntimeError(
                "User input was already claimed with a different resolution request."
            )
        same_stage = current_intent.resolution_stage == resolution_stage
        allowed_stage_transition = (
            allow_answer_to_manual_recovery
            and current_intent.resolution_stage == "answer"
            and resolution_stage == "manual-recovery"
        ) or (
            allow_manual_recovery_to_answer
            and current_intent.resolution_stage == "manual-recovery"
            and resolution_stage == "answer"
        )
        if not same_stage and not allowed_stage_transition:
            raise RuntimeError(
                "User input was already claimed with a different resolution request."
            )
        if same_stage and current_intent.resolution_request_digest != resolution_request_digest:
            raise RuntimeError(
                "User input was already claimed with a different resolution request."
            )
        if current_intent.claim_run_epoch == claim_run_epoch:
            if (
                not same_stage
                or current_intent.resolution_request_digest != resolution_request_digest
            ):
                raise RuntimeError(
                    "User input cannot replace resolution authority within one run epoch."
                )
            return copied, current_intent
        current_intent = user_input_resolution_intent_for(
            pending,
            answer_request_digest=answer_request_digest,
            resolution_stage=resolution_stage,
            resolution_request_digest=resolution_request_digest,
            claim_run_epoch=claim_run_epoch,
        )
        copied[USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY] = current_intent.model_dump(mode="json")
        return copied, current_intent
    intent = user_input_resolution_intent_for(
        pending,
        answer_request_digest=answer_request_digest,
        resolution_stage=resolution_stage,
        resolution_request_digest=resolution_request_digest,
        claim_run_epoch=claim_run_epoch,
    )
    copied[USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY] = intent.model_dump(mode="json")
    return copied, intent


def checkpoint_with_executing_user_input_resolution_intent(
    checkpoint: dict[str, Any] | None,
    *,
    current_run_epoch: int,
    pending: PendingUserInput,
    intent: UserInputResolutionIntent,
    redactor: SecretRedactor,
) -> tuple[dict[str, Any], UserInputResolutionIntent]:
    """Atomically admit governed continuation work for one exact resolution claim."""

    if type(current_run_epoch) is not int or current_run_epoch < 0:
        raise TypeError("current_run_epoch must be a non-negative integer.")
    if current_run_epoch != intent.claim_run_epoch:
        raise RuntimeError("User-input resolution lost its claimed run epoch.")
    copied = {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
    current_pending, current_intent = user_input_lifecycle_authority_from_checkpoint(
        copied,
        redactor=redactor,
        current_run_epoch=current_run_epoch,
    )
    if current_pending != pending or current_intent is None or current_intent != intent:
        raise RuntimeError("User-input resolution authority changed before execution admission.")
    if current_intent.execution_state == "executing":
        return copied, current_intent
    executing = current_intent.model_copy(update={"execution_state": "executing"})
    copied[USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY] = executing.model_dump(mode="json")
    return copied, executing


def checkpoint_without_exact_pending_user_input(
    checkpoint: dict[str, Any] | None,
    *,
    pending: PendingUserInput,
    intent: UserInputResolutionIntent,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    """Clear only the exact pause and answer claim that own a close publication."""

    copied = {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
    current_pending = pending_user_input_from_checkpoint(copied, redactor=redactor)
    current_intent = user_input_resolution_intent_from_checkpoint(copied, redactor=redactor)
    if (
        current_pending is None
        or current_intent is None
        or current_pending != pending
        or current_intent != intent
    ):
        raise RuntimeError("Pending user-input authority changed before atomic closure.")
    require_resolution_intent_matches_pending(current_intent, pending=current_pending)
    copied.pop(PENDING_USER_INPUT_CHECKPOINT_KEY)
    copied.pop(USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY)
    return copied
