"""Session request contracts, input evidence and authenticated copying rules."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from hashlib import sha256
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema  # noqa: TC002 - Pydantic needs this at runtime.

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_metadata,
    copy_label_map,
    require_durable_json_text,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu._validation import require_durable_nonblank as require_nonblank
from cayu.approvals.actors import ResolutionActor, copy_resolution_actor
from cayu.budgets.base import BudgetLimit, copy_request_budget_limits
from cayu.budgets.run_limits import RunLimits, copy_run_limits
from cayu.build_provenance import RuntimeBuildProvenance
from cayu.configuration import DEFAULT_MAX_STEPS, MAX_STEPS
from cayu.context.structured_output import StructuredOutputSpec, copy_structured_output_spec
from cayu.context.thinking import ThinkingConfig
from cayu.deadlines import (
    EXECUTION_DEADLINE_METADATA_KEY,
    ExecutionDeadline,
    current_execution_deadline,
)
from cayu.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    copy_execution_profile_adoption_intent,
)
from cayu.messages import Message, detach_message
from cayu.providers.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.loop_policies import LoopPolicy, validate_loop_policies
from cayu.runtime.tool_completion import ToolCompletionPolicy, copy_tool_completion_policy
from cayu.sessions import records as session_record_rules
from cayu.sessions import transcript_input as session_transcript_input
from cayu.sessions._execution_profile_checkpoint import EXECUTION_PROFILE_METADATA_KEY
from cayu.sessions._model_failover import (
    MODEL_TARGET_PROJECTION_METADATA_KEY,
    ModelFailoverPolicy,
    ModelTarget,
    copy_optional_model_failover_policy,
)
from cayu.sessions.forks import (
    FORK_EXECUTION_PROFILE_METADATA_KEY,
    FORK_SOURCE_SNAPSHOT_METADATA_KEY,
    PROMPT_ANATOMY_TRANSITION_METADATA_KEY,
    ForkExecutionProfileSelection,
    ForkSourceSnapshot,
    ForkSystemPromptPolicy,
)
from cayu.sessions.invocation import (
    InvocationOrigin,
    InvocationOriginClaim,
    SessionExecutionSource,
    copy_invocation_origin,
    copy_invocation_origin_claim,
    copy_task_invocation,
)
from cayu.sessions.records import Session, SessionIdentity
from cayu.sessions.transcript_input import session_messages_input_contract_evidence
from cayu.tasks.creation import TaskInvocationSnapshot
from cayu.tools.exposure import (
    TOOL_CAPABILITY_CEILING_METADATA_KEY,
    ToolCapabilityCeiling,
    copy_tool_capability_ceiling,
)
from cayu.tools.grants import TargetedToolGrant, validate_targeted_tool_grants

# Compaction guidance can carry a domain glossary or a list of facts to keep.
COMPACTION_INSTRUCTIONS_MAX_CHARS = 32_768
_RUNTIME_SESSION_CREATE_CLAIM_TOKEN = object()
_RUNTIME_SESSION_INSTANCE_AUTHORITY_TOKEN = object()
_RUNTIME_INITIAL_TRANSCRIPT_AUTHORITY_TOKEN = object()
_RUNTIME_PREPARED_SESSION_AUTHORITY_TOKEN = object()
_RUNTIME_RESUME_TRANSPORT_METADATA_TOKEN = object()


class _RuntimeSessionCreateClaim:
    """Authenticated process-local authority for one durable create readback."""

    expected_session_material: _SessionCreateMaterial | None
    interaction_id: str | None
    messages_sha256: str | None
    request_sha256: str | None

    __slots__ = (
        "claim_id",
        "expected_session_material",
        "interaction_id",
        "messages_sha256",
        "request_sha256",
        "session_id",
        "token",
    )

    def __init__(self, *, session_id: str, claim_id: str) -> None:
        self.token = _RUNTIME_SESSION_CREATE_CLAIM_TOKEN
        self.session_id = session_id
        self.claim_id = require_clean_nonblank(claim_id, "session create claim_id")
        self.expected_session_material = None
        self.interaction_id = None
        self.messages_sha256 = None
        self.request_sha256 = None

    def __deepcopy__(self, memo: dict[int, Any]) -> None:
        # Generic deep copies are caller-controlled and must not duplicate
        # runtime authority. The explicit RunRequest copier authenticates and
        # preserves the original handoff instead.
        return None


class _RuntimeSessionInstanceAuthority:
    """Authenticated process-local authority for one preallocated incarnation."""

    __slots__ = ("session_id", "session_instance_id", "token")

    def __init__(self, *, session_id: str, session_instance_id: str) -> None:
        self.token = _RUNTIME_SESSION_INSTANCE_AUTHORITY_TOKEN
        self.session_id = session_id
        self.session_instance_id = session_instance_id

    def __deepcopy__(self, memo: dict[int, Any]) -> None:
        return None


class _RuntimeInitialTranscriptAuthority:
    """Authenticated process-local handoff for an exact initial transcript."""

    __slots__ = (
        "initial_transcript_messages",
        "interaction_id",
        "session_id",
        "source_messages",
        "token",
    )

    def __init__(
        self,
        *,
        session_id: str,
        interaction_id: str,
        source_messages: list[Message],
        initial_transcript_messages: list[Message],
    ) -> None:
        self.token = _RUNTIME_INITIAL_TRANSCRIPT_AUTHORITY_TOKEN
        self.session_id = session_id
        self.interaction_id = interaction_id
        self.source_messages = tuple(detach_message(message) for message in source_messages)
        self.initial_transcript_messages = tuple(
            detach_message(message) for message in initial_transcript_messages
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _RuntimePreparedSessionAuthority:
    """Process-local proof that a claimed queue task may start one PENDING session."""

    token: object
    session_id: str
    queue_task_id: str
    dispatch_operation_id: str
    terminal_event_id: str
    interaction_id: str
    interaction_started_event_id: str
    idempotency_key: str
    submission_sha256: str
    provider_name: str
    model: str
    policy_evidence: bytes | None

    def __post_init__(self) -> None:
        if self.policy_evidence is not None and type(self.policy_evidence) is not bytes:
            raise TypeError("Prepared policy evidence must be immutable bytes.")
        for field_name in (
            "provider_name",
            "model",
            "session_id",
            "queue_task_id",
            "dispatch_operation_id",
            "terminal_event_id",
            "interaction_id",
            "interaction_started_event_id",
            "idempotency_key",
            "submission_sha256",
        ):
            require_clean_nonblank(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class _RuntimeResumeTransportMetadataAuthority:
    """Positive provenance for per-attempt metadata kept outside semantic input."""

    token: object
    values: tuple[tuple[str, str], ...]


def _empty_run_request_authority() -> frozenset[tuple[str, str]]:
    return frozenset()


def _copy_optional_tool_capability_ceiling(
    value: object,
) -> ToolCapabilityCeiling | None:
    if value is None:
        return None
    if isinstance(value, ToolCapabilityCeiling):
        return copy_tool_capability_ceiling(value)
    return ToolCapabilityCeiling.model_validate(value)


class RunRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        hide_input_in_errors=True,
    )

    agent_name: str
    messages: list[Message]
    execution_deadline: ExecutionDeadline = Field(
        default_factory=current_execution_deadline,
        exclude_if=lambda boundary: boundary.expires_at is None,
    )
    # Optional caller-provided id for a new session. It must be unique.
    session_id: str | None = None
    parent_session_id: str | None = None
    # Durable budget/accounting identity shared by related sessions. Defaults to
    # task_id when present, otherwise session_id. Forks inherit the source value.
    causal_budget_id: str | None = None
    task_id: str | None = None
    task_worker_id: str | None = None
    # Exact claim generation for worker-owned fresh task attachment. Worker
    # identifiers are reusable after lease expiry and are not authority alone.
    task_lease_expires_at: datetime | None = None
    # Exact per-run execution target. When omitted, the agent model and provider
    # routing/defaults select the initial target.
    target: ModelTarget | None = None
    failover: ModelFailoverPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # Durable application-tool maximum. None selects the current registered catalog.
    tool_capability_ceiling: ToolCapabilityCeiling | None = None
    # Interaction-scoped addressability requests resolved from the registered catalogue.
    tool_grants: tuple[TargetedToolGrant, ...] = ()
    environment_name: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    invocation_origin: InvocationOriginClaim | None = None
    max_steps: StrictInt = Field(default=DEFAULT_MAX_STEPS, ge=1, le=MAX_STEPS)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    tool_completion: ToolCompletionPolicy | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Complete a single-call round from a designated successful none or idempotent tool.",
    )
    loop_policies: SkipJsonSchema[tuple[LoopPolicy, ...]] = Field(
        default_factory=tuple,
        exclude=True,
    )
    _runtime_generated_authority: frozenset[tuple[str, str]] = PrivateAttr(
        default_factory=_empty_run_request_authority
    )
    _runtime_session_create_claim: object | None = PrivateAttr(default=None)
    _runtime_session_instance_authority: object | None = PrivateAttr(default=None)
    _runtime_initial_transcript_authority: object | None = PrivateAttr(default=None)
    _runtime_work_attempt_creation: object | None = PrivateAttr(default=None)
    _input_redactions_applied: bool = PrivateAttr(default=False)
    _verified_invocation_origin: InvocationOrigin | None = PrivateAttr(default=None)
    _runtime_invocation_source: SessionExecutionSource | None = PrivateAttr(default=None)
    _runtime_task_invocation: TaskInvocationSnapshot | None = PrivateAttr(default=None)
    _runtime_prepared_session_authority: object | None = PrivateAttr(default=None)

    @field_validator("failover", mode="before")
    @classmethod
    def copy_failover(cls, value: object) -> ModelFailoverPolicy | None:
        return copy_optional_model_failover_policy(value)

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        return [session_transcript_input._copy_caller_input_message(message) for message in value]

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_durable_metadata(value)
        reserved_key = next(
            (
                key
                for key in (
                    EXECUTION_DEADLINE_METADATA_KEY,
                    MODEL_TARGET_PROJECTION_METADATA_KEY,
                    EXECUTION_PROFILE_METADATA_KEY,
                    TOOL_CAPABILITY_CEILING_METADATA_KEY,
                    PROMPT_ANATOMY_TRANSITION_METADATA_KEY,
                    FORK_EXECUTION_PROFILE_METADATA_KEY,
                )
                if key in copied
            ),
            None,
        )
        if reserved_key is not None:
            copied.clear()
            if reserved_key == EXECUTION_DEADLINE_METADATA_KEY:
                authority_name = "execution-deadline authority"
            elif reserved_key == MODEL_TARGET_PROJECTION_METADATA_KEY:
                authority_name = "model-target authority"
            elif reserved_key == EXECUTION_PROFILE_METADATA_KEY:
                authority_name = "execution-profile authority"
            elif reserved_key == FORK_EXECUTION_PROFILE_METADATA_KEY:
                authority_name = "fork-profile authority"
            elif reserved_key == TOOL_CAPABILITY_CEILING_METADATA_KEY:
                authority_name = "tool-capability-ceiling authority"
            else:
                authority_name = "prompt-transition authority"
            raise ValueError(f"metadata[{reserved_key!r}] is runtime-owned {authority_name}.")
        return copied

    @field_validator("labels", mode="before")
    @classmethod
    def copy_request_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels", allow_reserved=False)

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("tool_capability_ceiling", mode="before")
    @classmethod
    def copy_tool_capability_ceiling(
        cls,
        value: object,
    ) -> ToolCapabilityCeiling | None:
        return _copy_optional_tool_capability_ceiling(value)

    @field_validator("tool_grants", mode="before")
    @classmethod
    def copy_tool_grants(cls, value: object) -> tuple[TargetedToolGrant, ...]:
        return validate_targeted_tool_grants(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @field_validator("limits")
    @classmethod
    def copy_limits(cls, value: RunLimits) -> RunLimits:
        return copy_run_limits(value)

    @field_validator("tool_completion", mode="before")
    @classmethod
    def copy_tool_completion(cls, value: object) -> ToolCompletionPolicy | None:
        return copy_tool_completion_policy(value)

    @field_validator("loop_policies", mode="before")
    @classmethod
    def copy_loop_policies(cls, value) -> tuple[LoopPolicy, ...]:
        return validate_loop_policies(value, field_name="loop_policies")

    @field_validator("agent_name")
    @classmethod
    def validate_nonblank_agent_name(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("session_id")
    @classmethod
    def validate_optional_session_id(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return session_record_rules._require_bounded_session_id(value, info.field_name)

    @field_validator(
        "parent_session_id",
        "causal_budget_id",
        "task_id",
        "task_worker_id",
        "environment_name",
    )
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_task_worker_handoff(self) -> RunRequest:
        if self.task_worker_id is not None and self.task_id is None:
            raise ValueError("RunRequest.task_worker_id requires task_id.")
        if (self.task_worker_id is None) != (self.task_lease_expires_at is None):
            raise ValueError(
                "RunRequest.task_worker_id and task_lease_expires_at must be supplied together."
            )
        return self

    @field_validator("task_lease_expires_at")
    @classmethod
    def normalize_task_lease_expires_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, "task_lease_expires_at")


def session_input_contract_evidence(
    request: RunRequest | ResumeRequest,
    *,
    message_start_index: int,
) -> str:
    """Return the canonical runtime-owned fresh-input contract for one request."""

    if type(request) not in {RunRequest, ResumeRequest}:
        raise TypeError("request must be an exact RunRequest or ResumeRequest.")
    return session_messages_input_contract_evidence(
        request.messages,
        message_start_index=message_start_index,
        redactions_applied=request._input_redactions_applied,
        structured_output_requested=request.structured_output is not None,
    )


class ResumeRequest(BaseModel):
    """Continue a conversation; omitted loop controls inherit verified prior settings.

    ``max_steps``, ``limits``, and ``retry_policy`` inherit from the latest
    profile-bound model completion when available. Explicit values still pass
    normal execution-profile admission. Legacy/fork sessions without matching
    evidence use application defaults. This does not reset session-scoped limits.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        hide_input_in_errors=True,
    )

    session_id: str
    # Optional exact task lease owner for a receipt-backed task continuation.
    task_worker_id: str | None = None
    task_handoff_id: str | None = None
    messages: list[Message]
    target: ModelTarget | None = None
    failover: ModelFailoverPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # None preserves the durable maximum; an explicit subset narrows it permanently.
    tool_capability_ceiling: ToolCapabilityCeiling | None = None
    # Fresh grants apply only to the newly admitted ordinary interaction.
    tool_grants: tuple[TargetedToolGrant, ...] = ()
    profile_adoption: ExecutionProfileAdoptionIntent | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_steps: StrictInt = Field(default=DEFAULT_MAX_STEPS, ge=1, le=MAX_STEPS)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    tool_completion: ToolCompletionPolicy | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Complete a single-call round from a designated successful none or idempotent tool.",
    )
    loop_policies: SkipJsonSchema[tuple[LoopPolicy, ...]] = Field(
        default_factory=tuple,
        exclude=True,
    )
    _runtime_transport_metadata_authority: object | None = PrivateAttr(default=None)
    _input_redactions_applied: bool = PrivateAttr(default=False)

    @field_validator("failover", mode="before")
    @classmethod
    def copy_failover(cls, value: object) -> ModelFailoverPolicy | None:
        return copy_optional_model_failover_policy(value)

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        copied_messages = [
            session_transcript_input._copy_caller_input_message(message) for message in value
        ]
        if not copied_messages:
            raise ValueError("ResumeRequest messages cannot be empty.")
        return copied_messages

    @field_validator("task_worker_id", "task_handoff_id")
    @classmethod
    def validate_optional_task_worker_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "task continuation authority")

    @model_validator(mode="after")
    def validate_task_handoff_authority(self) -> ResumeRequest:
        if self.task_handoff_id is not None and self.task_worker_id is None:
            raise ValueError("task_handoff_id requires task_worker_id.")
        return self

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_durable_metadata(value)
        if EXECUTION_DEADLINE_METADATA_KEY in copied:
            raise ValueError("Session metadata contains runtime-owned deadline authority.")
        return copied

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("tool_capability_ceiling", mode="before")
    @classmethod
    def copy_tool_capability_ceiling(
        cls,
        value: object,
    ) -> ToolCapabilityCeiling | None:
        return _copy_optional_tool_capability_ceiling(value)

    @field_validator("tool_grants", mode="before")
    @classmethod
    def copy_tool_grants(cls, value: object) -> tuple[TargetedToolGrant, ...]:
        return validate_targeted_tool_grants(value)

    @field_validator("profile_adoption", mode="before")
    @classmethod
    def copy_profile_adoption(
        cls,
        value: object,
    ) -> ExecutionProfileAdoptionIntent | None:
        if value is None:
            return None
        if isinstance(value, ExecutionProfileAdoptionIntent):
            return copy_execution_profile_adoption_intent(value)
        return ExecutionProfileAdoptionIntent.model_validate(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @field_validator("limits")
    @classmethod
    def copy_limits(cls, value: RunLimits) -> RunLimits:
        return copy_run_limits(value)

    @field_validator("tool_completion", mode="before")
    @classmethod
    def copy_tool_completion(cls, value: object) -> ToolCompletionPolicy | None:
        return copy_tool_completion_policy(value)

    @field_validator("loop_policies", mode="before")
    @classmethod
    def copy_loop_policies(cls, value) -> tuple[LoopPolicy, ...]:
        return validate_loop_policies(value, field_name="loop_policies")

    @field_validator("session_id")
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class CompactSessionRequest(BaseModel):
    """Request an explicit, application-owned compaction of durable session context."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str
    idempotency_key: str = Field(max_length=256)
    expected_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    expected_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    reason: Literal["application_requested"] = "application_requested"
    instructions: str | None = Field(default=None, max_length=COMPACTION_INSTRUCTIONS_MAX_CHARS)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    requested_by: ResolutionActor | None = None

    @field_validator("session_id", "idempotency_key")
    @classmethod
    def validate_required_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("instructions")
    @classmethod
    def validate_optional_instructions(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, "instructions")

    @field_validator("limits")
    @classmethod
    def copy_limits(cls, value: RunLimits) -> RunLimits:
        return copy_run_limits(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @field_validator("requested_by")
    @classmethod
    def copy_requested_by(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)

    @model_validator(mode="after")
    def validate_durable_text(self) -> CompactSessionRequest:
        require_durable_json_text(
            self.model_dump(mode="json", exclude={"requested_by"}),
            "CompactSessionRequest",
        )
        if self.requested_by is not None:
            require_durable_json_text(
                self.requested_by.model_dump(mode="json", exclude={"claims"}),
                "CompactSessionRequest.requested_by",
            )
        return self


class InterruptSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    requested_by: ResolutionActor | None = None

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_metadata(value)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_optional_reason(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("requested_by")
    @classmethod
    def copy_requested_by(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)


class ForkSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_session_id: str
    session_id: str | None = None
    agent_name: str | None = None
    model: str | None = None
    environment_name: str | None = None
    # None inherits the source maximum; an explicit subset narrows it for the child.
    tool_capability_ceiling: ToolCapabilityCeiling | None = None
    transcript_cursor: StrictInt | None = Field(default=None, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    copy_checkpoint: StrictBool = True
    system_prompt_policy: ForkSystemPromptPolicy = ForkSystemPromptPolicy.INHERIT_SOURCE
    execution_profile_selection: ForkExecutionProfileSelection = (
        ForkExecutionProfileSelection.INHERIT_PARENT
    )
    profile_adoption: ExecutionProfileAdoptionIntent | None = None
    expected_source: ForkSourceSnapshot | None = None
    initial_invocation: ResumeRequest | None = None
    initial_dispatch_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id")
    @classmethod
    def validate_optional_destination_session_id(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return session_record_rules._require_bounded_session_id(value, info.field_name)

    @field_validator(
        "source_session_id",
        "agent_name",
        "model",
        "environment_name",
        "initial_dispatch_id",
    )
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_durable_metadata(value)
        reserved_authority_kinds = {
            EXECUTION_DEADLINE_METADATA_KEY: "execution-deadline authority",
            MODEL_TARGET_PROJECTION_METADATA_KEY: "model-target authority",
            EXECUTION_PROFILE_METADATA_KEY: "execution-profile authority",
            TOOL_CAPABILITY_CEILING_METADATA_KEY: "tool-capability-ceiling authority",
            PROMPT_ANATOMY_TRANSITION_METADATA_KEY: "prompt-transition authority",
            FORK_EXECUTION_PROFILE_METADATA_KEY: "fork-profile authority",
            FORK_SOURCE_SNAPSHOT_METADATA_KEY: "fork-source authority",
        }
        reserved_key = next(
            (key for key in reserved_authority_kinds if key in copied),
            None,
        )
        if reserved_key is not None:
            copied.clear()
            raise ValueError(
                f"metadata[{reserved_key!r}] is runtime-owned "
                f"{reserved_authority_kinds[reserved_key]}."
            )
        return copied

    @field_validator("tool_capability_ceiling", mode="before")
    @classmethod
    def copy_tool_capability_ceiling(
        cls,
        value: object,
    ) -> ToolCapabilityCeiling | None:
        return _copy_optional_tool_capability_ceiling(value)

    @field_validator("profile_adoption", mode="before")
    @classmethod
    def copy_profile_adoption(
        cls,
        value: object,
    ) -> ExecutionProfileAdoptionIntent | None:
        if value is None:
            return None
        if isinstance(value, ExecutionProfileAdoptionIntent):
            return copy_execution_profile_adoption_intent(value)
        return ExecutionProfileAdoptionIntent.model_validate(value)

    @field_validator("expected_source", mode="before")
    @classmethod
    def copy_expected_source(cls, value: object) -> ForkSourceSnapshot | None:
        if value is None:
            return None
        if isinstance(value, ForkSourceSnapshot):
            value = value.model_dump(mode="json")
        return ForkSourceSnapshot.model_validate(value)

    @field_validator("initial_invocation", mode="before")
    @classmethod
    def copy_initial_invocation(cls, value: object) -> ResumeRequest | None:
        if value is None:
            return None
        if isinstance(value, ResumeRequest):
            return copy_resume_request(value)
        return ResumeRequest.model_validate(value)

    @model_validator(mode="after")
    def validate_execution_profile_selection(self) -> ForkSessionRequest:
        if (
            self.expected_source is not None
            and self.expected_source.source_session_id != self.source_session_id
        ):
            raise ValueError("expected_source must identify source_session_id exactly.")
        if self.expected_source is not None and (
            self.transcript_cursor is not None or not self.copy_checkpoint
        ):
            raise ValueError(
                "An exact expected_source requires the complete transcript and checkpoint."
            )
        if self.initial_invocation is not None and (
            self.session_id is None or self.initial_invocation.session_id != self.session_id
        ):
            raise ValueError("initial_invocation requires an explicit matching child session_id.")
        if (self.initial_invocation is None) != (self.initial_dispatch_id is None):
            raise ValueError(
                "initial_invocation and initial_dispatch_id must be supplied together."
            )
        if self.initial_invocation is not None and self.expected_source is None:
            raise ValueError("initial_invocation requires an exact expected_source snapshot.")
        if self.initial_invocation is not None and (
            self.initial_invocation.task_worker_id is not None
            or self.initial_invocation.task_handoff_id is not None
        ):
            raise ValueError(
                "initial_invocation cannot contain task continuation authority because "
                "durable dispatch cannot reconstruct it."
            )
        if self.initial_invocation is not None and (
            self.initial_invocation.target is not None
            or self.initial_invocation.profile_adoption is not None
        ):
            raise ValueError(
                "Configure child profile adoption on the fork; initial_invocation cannot "
                "change the model target or adopt another profile."
            )
        if self.initial_invocation is not None and self.initial_invocation.loop_policies:
            raise ValueError(
                "initial_invocation cannot contain process-local loop_policies because "
                "durable dispatch cannot reconstruct them."
            )
        if self.execution_profile_selection is ForkExecutionProfileSelection.INHERIT_PARENT:
            if self.profile_adoption is not None:
                raise ValueError("Parent-profile inheritance cannot include profile_adoption.")
            if (
                self.agent_name is not None
                or self.model is not None
                or self.environment_name is not None
            ):
                raise ValueError(
                    "Fork agent/model/environment overrides require execution_profile_selection="
                    "'current_child'."
                )
            if self.system_prompt_policy is ForkSystemPromptPolicy.CURRENT_AGENT:
                raise ValueError(
                    "The current agent system prompt requires execution_profile_selection="
                    "'current_child'."
                )
        elif self.profile_adoption is None:
            raise ValueError("Current-child profile selection requires profile_adoption.")
        return self


@dataclass(frozen=True, slots=True)
class _SessionCreateMaterial:
    agent_name: str
    provider_name: str
    model: str
    parent_session_id: str | None
    causal_budget_id: str
    runtime_name: str
    runtime_version: str | None
    runtime_build_provenance: RuntimeBuildProvenance
    environment_name: str | None

    @classmethod
    def from_request(
        cls,
        request: RunRequest,
        *,
        identity: SessionIdentity,
        session_id: str,
    ) -> _SessionCreateMaterial:
        return cls(
            agent_name=request.agent_name,
            provider_name=identity.provider_name,
            model=identity.model,
            parent_session_id=request.parent_session_id,
            causal_budget_id=request.causal_budget_id or request.task_id or session_id,
            runtime_name=identity.runtime_name,
            runtime_version=identity.runtime_version,
            runtime_build_provenance=identity.runtime_build_provenance,
            environment_name=request.environment_name,
        )

    @classmethod
    def from_session(cls, session: Session) -> _SessionCreateMaterial:
        return cls(
            agent_name=session.agent_name,
            provider_name=session.provider_name,
            model=session.model,
            parent_session_id=session.parent_session_id,
            causal_budget_id=session.causal_budget_id,
            runtime_name=session.runtime_name,
            runtime_version=session.runtime_version,
            runtime_build_provenance=session.runtime_build_provenance,
            environment_name=session.environment_name,
        )


def copy_run_request(request: RunRequest) -> RunRequest:
    if type(request) is not RunRequest:
        raise TypeError("Session creation requires a RunRequest.")
    messages = getattr(request, "messages", None)
    if type(messages) is not list:
        raise ValueError("RunRequest messages must be a list.")
    copied = RunRequest(
        agent_name=request.agent_name,
        messages=[detach_message(message) for message in messages],
        session_id=request.session_id,
        execution_deadline=request.execution_deadline,
        parent_session_id=request.parent_session_id,
        causal_budget_id=request.causal_budget_id,
        task_id=request.task_id,
        task_worker_id=request.task_worker_id,
        task_lease_expires_at=request.task_lease_expires_at,
        failover=copy_optional_model_failover_policy(request.failover),
        target=(
            None
            if request.target is None
            else ModelTarget(
                provider_name=request.target.provider_name,
                model=request.target.model,
            )
        ),
        tool_capability_ceiling=_copy_optional_tool_capability_ceiling(
            request.tool_capability_ceiling
        ),
        tool_grants=validate_targeted_tool_grants(request.tool_grants),
        environment_name=request.environment_name,
        labels=copy_label_map(request.labels, "labels"),
        metadata=copy_durable_metadata(request.metadata),
        invocation_origin=copy_invocation_origin_claim(request.invocation_origin),
        budget_limits=copy_request_budget_limits(request.budget_limits),
        retry_policy=copy_retry_policy(request.retry_policy) if request.retry_policy else None,
        structured_output=copy_structured_output_spec(request.structured_output),
        tool_completion=copy_tool_completion_policy(request.tool_completion),
        loop_policies=validate_loop_policies(request.loop_policies, field_name="loop_policies"),
    )
    copied = copied.model_copy(
        update={
            "max_steps": request.max_steps,
            "limits": copy_run_limits(request.limits),
            "thinking": (
                None
                if request.thinking is None
                else ThinkingConfig(
                    enabled=request.thinking.enabled,
                    effort=request.thinking.effort,
                    max_tokens=request.thinking.max_tokens,
                    include_in_transcript=request.thinking.include_in_transcript,
                )
            ),
        }
    )
    copied_fields_set = set(copied.model_fields_set)
    for field_name in ("max_steps", "limits", "thinking", "tool_completion"):
        if field_name not in request.model_fields_set:
            copied_fields_set.discard(field_name)
    object.__setattr__(copied, "__pydantic_fields_set__", copied_fields_set)
    copied._runtime_generated_authority = request._runtime_generated_authority
    prepared_creation = request._runtime_work_attempt_creation
    copied._runtime_work_attempt_creation = (
        prepared_creation
        if type(prepared_creation) is _PreparedWorkAttemptCreation
        and prepared_creation.token is _PREPARED_WORK_ATTEMPT_CREATION_TOKEN
        else None
    )
    create_claim = request._runtime_session_create_claim
    copied._runtime_session_create_claim = (
        create_claim
        if type(create_claim) is _RuntimeSessionCreateClaim
        and create_claim.token is _RUNTIME_SESSION_CREATE_CLAIM_TOKEN
        else None
    )
    session_instance_authority = request._runtime_session_instance_authority
    copied._runtime_session_instance_authority = (
        session_instance_authority
        if type(session_instance_authority) is _RuntimeSessionInstanceAuthority
        and session_instance_authority.token is _RUNTIME_SESSION_INSTANCE_AUTHORITY_TOKEN
        and session_instance_authority.session_id == request.session_id
        else None
    )
    initial_transcript_authority = request._runtime_initial_transcript_authority
    copied._runtime_initial_transcript_authority = (
        initial_transcript_authority
        if type(initial_transcript_authority) is _RuntimeInitialTranscriptAuthority
        and initial_transcript_authority.token is _RUNTIME_INITIAL_TRANSCRIPT_AUTHORITY_TOKEN
        and initial_transcript_authority.session_id == request.session_id
        and initial_transcript_authority.source_messages == tuple(request.messages)
        else None
    )
    copied._input_redactions_applied = request._input_redactions_applied
    copied._verified_invocation_origin = (
        None
        if request._verified_invocation_origin is None
        else copy_invocation_origin(request._verified_invocation_origin)
    )
    copied._runtime_invocation_source = request._runtime_invocation_source
    copied._runtime_task_invocation = (
        None
        if request._runtime_task_invocation is None
        else TaskInvocationSnapshot(
            id=request._runtime_task_invocation.id,
            session_id=request._runtime_task_invocation.session_id,
            session_instance_id=request._runtime_task_invocation.session_instance_id,
            invocation=copy_task_invocation(request._runtime_task_invocation.invocation),
        )
    )
    prepared_authority = request._runtime_prepared_session_authority
    copied._runtime_prepared_session_authority = (
        prepared_authority
        if type(prepared_authority) is _RuntimePreparedSessionAuthority
        and prepared_authority.token is _RUNTIME_PREPARED_SESSION_AUTHORITY_TOKEN
        else None
    )
    return copied


def copy_resume_request(request: ResumeRequest) -> ResumeRequest:
    if type(request) is not ResumeRequest:
        raise TypeError("Session resume requires a ResumeRequest.")
    messages = getattr(request, "messages", None)
    if type(messages) is not list:
        raise ValueError("ResumeRequest messages must be a list.")
    copied = ResumeRequest(
        session_id=request.session_id,
        task_worker_id=request.task_worker_id,
        task_handoff_id=request.task_handoff_id,
        failover=copy_optional_model_failover_policy(request.failover),
        messages=[detach_message(message) for message in messages],
        target=(
            None
            if request.target is None
            else ModelTarget(
                provider_name=request.target.provider_name,
                model=request.target.model,
            )
        ),
        tool_capability_ceiling=_copy_optional_tool_capability_ceiling(
            request.tool_capability_ceiling
        ),
        tool_grants=validate_targeted_tool_grants(request.tool_grants),
        profile_adoption=(
            None
            if request.profile_adoption is None
            else copy_execution_profile_adoption_intent(request.profile_adoption)
        ),
        metadata=copy_durable_metadata(request.metadata),
        budget_limits=copy_request_budget_limits(request.budget_limits),
        retry_policy=copy_retry_policy(request.retry_policy) if request.retry_policy else None,
        structured_output=copy_structured_output_spec(request.structured_output),
        tool_completion=copy_tool_completion_policy(request.tool_completion),
        loop_policies=validate_loop_policies(request.loop_policies, field_name="loop_policies"),
    )
    copied = copied.model_copy(
        update={
            "max_steps": request.max_steps,
            "limits": copy_run_limits(request.limits),
            "thinking": (
                None
                if request.thinking is None
                else ThinkingConfig(
                    enabled=request.thinking.enabled,
                    effort=request.thinking.effort,
                    max_tokens=request.thinking.max_tokens,
                    include_in_transcript=request.thinking.include_in_transcript,
                )
            ),
        }
    )
    copied_fields_set = set(copied.model_fields_set)
    for field_name in ("max_steps", "limits", "thinking", "retry_policy", "tool_completion"):
        if field_name not in request.model_fields_set:
            copied_fields_set.discard(field_name)
    object.__setattr__(copied, "__pydantic_fields_set__", copied_fields_set)
    authority = request._runtime_transport_metadata_authority
    if (
        type(authority) is _RuntimeResumeTransportMetadataAuthority
        and authority.token is _RUNTIME_RESUME_TRANSPORT_METADATA_TOKEN
    ):
        copied._runtime_transport_metadata_authority = authority
    copied._input_redactions_applied = request._input_redactions_applied
    return copied


def copy_compact_session_request(request: CompactSessionRequest) -> CompactSessionRequest:
    if type(request) is not CompactSessionRequest:
        raise TypeError("Session compaction requires a CompactSessionRequest.")
    return CompactSessionRequest(
        session_id=request.session_id,
        idempotency_key=request.idempotency_key,
        expected_run_epoch=request.expected_run_epoch,
        expected_transcript_cursor=request.expected_transcript_cursor,
        reason=request.reason,
        instructions=request.instructions,
        limits=copy_run_limits(request.limits),
        budget_limits=copy_request_budget_limits(request.budget_limits),
        requested_by=copy_resolution_actor(request.requested_by),
    )


def copy_interrupt_session_request(request: InterruptSessionRequest) -> InterruptSessionRequest:
    if type(request) is not InterruptSessionRequest:
        raise TypeError("Session interruption requires an InterruptSessionRequest.")
    return InterruptSessionRequest(
        session_id=request.session_id,
        reason=request.reason,
        metadata=copy_durable_metadata(request.metadata),
        requested_by=copy_resolution_actor(request.requested_by),
    )


def copy_fork_session_request(request: ForkSessionRequest) -> ForkSessionRequest:
    if type(request) is not ForkSessionRequest:
        raise TypeError("Session fork requires a ForkSessionRequest.")
    copied = ForkSessionRequest(
        source_session_id=request.source_session_id,
        session_id=request.session_id,
        agent_name=request.agent_name,
        model=request.model,
        environment_name=request.environment_name,
        tool_capability_ceiling=_copy_optional_tool_capability_ceiling(
            request.tool_capability_ceiling
        ),
        transcript_cursor=request.transcript_cursor,
        copy_checkpoint=request.copy_checkpoint,
        system_prompt_policy=request.system_prompt_policy,
        execution_profile_selection=request.execution_profile_selection,
        profile_adoption=(
            None
            if request.profile_adoption is None
            else copy_execution_profile_adoption_intent(request.profile_adoption)
        ),
        expected_source=(
            None
            if request.expected_source is None
            else ForkSourceSnapshot.model_validate(request.expected_source.model_dump(mode="json"))
        ),
        initial_invocation=(
            None
            if request.initial_invocation is None
            else copy_resume_request(request.initial_invocation)
        ),
        initial_dispatch_id=request.initial_dispatch_id,
        metadata=copy_durable_metadata(request.metadata),
    )
    return copied


_PREPARED_WORK_ATTEMPT_CREATION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _PreparedWorkAttemptCreation:
    """Permission to finish exactly one already-admitted metadata creation.

    This capability is not serialized and never authorizes execution. Recovery
    must reconstruct it from the acknowledged durable preparation each time.
    """

    request_sha256: str
    token: object = dataclass_field(repr=False)

    def __deepcopy__(self, memo: dict[int, Any]) -> _PreparedWorkAttemptCreation:
        return self


def _run_request_invocation_lifecycle_authority_sha256(request: RunRequest) -> str:
    """Hash authenticated private create authority omitted from model serialization."""

    copied = copy_run_request(request)
    create_claim = copied._runtime_session_create_claim
    create_claim_material: dict[str, Any] | None = None
    if type(create_claim) is _RuntimeSessionCreateClaim and (
        create_claim.token is _RUNTIME_SESSION_CREATE_CLAIM_TOKEN
    ):
        expected = create_claim.expected_session_material
        create_claim_material = {
            "claim_id": create_claim.claim_id,
            "session_id": create_claim.session_id,
            "interaction_id": create_claim.interaction_id,
            "messages_sha256": create_claim.messages_sha256,
            "request_sha256": create_claim.request_sha256,
            "expected_session_material": (
                None
                if expected is None
                else {
                    "agent_name": expected.agent_name,
                    "provider_name": expected.provider_name,
                    "model": expected.model,
                    "parent_session_id": expected.parent_session_id,
                    "causal_budget_id": expected.causal_budget_id,
                    "runtime_name": expected.runtime_name,
                    "runtime_version": expected.runtime_version,
                    "environment_name": expected.environment_name,
                }
            ),
        }
    instance = copied._runtime_session_instance_authority
    instance_material = (
        None
        if type(instance) is not _RuntimeSessionInstanceAuthority
        or instance.token is not _RUNTIME_SESSION_INSTANCE_AUTHORITY_TOKEN
        else {
            "session_id": instance.session_id,
            "session_instance_id": instance.session_instance_id,
        }
    )
    transcript = copied._runtime_initial_transcript_authority
    transcript_material = (
        None
        if type(transcript) is not _RuntimeInitialTranscriptAuthority
        or transcript.token is not _RUNTIME_INITIAL_TRANSCRIPT_AUTHORITY_TOKEN
        else {
            "session_id": transcript.session_id,
            "interaction_id": transcript.interaction_id,
            "source_messages": [
                item.model_dump(mode="json") for item in transcript.source_messages
            ],
            "initial_transcript_messages": [
                item.model_dump(mode="json") for item in transcript.initial_transcript_messages
            ],
        }
    )
    prepared = runtime_prepared_session_authority(copied)
    prepared_material = (
        None
        if prepared is None
        else {
            "session_id": prepared.session_id,
            "queue_task_id": prepared.queue_task_id,
            "dispatch_operation_id": prepared.dispatch_operation_id,
            "terminal_event_id": prepared.terminal_event_id,
            "interaction_id": prepared.interaction_id,
            "interaction_started_event_id": prepared.interaction_started_event_id,
            "idempotency_key": prepared.idempotency_key,
            "submission_sha256": prepared.submission_sha256,
        }
    )
    material = {
        "runtime_generated_authority": [
            list(item) for item in sorted(copied._runtime_generated_authority)
        ],
        "session_create_claim": create_claim_material,
        "session_instance_authority": instance_material,
        "initial_transcript_authority": transcript_material,
        "input_redactions_applied": copied._input_redactions_applied,
        "verified_invocation_origin": (
            None
            if copied._verified_invocation_origin is None
            else copied._verified_invocation_origin.model_dump(mode="json")
        ),
        "runtime_invocation_source": (
            None
            if copied._runtime_invocation_source is None
            else copied._runtime_invocation_source.value
        ),
        "task_invocation": (
            None
            if copied._runtime_task_invocation is None
            else copied._runtime_task_invocation.model_dump(mode="json")
        ),
        "prepared_session_authority": prepared_material,
    }
    return sha256(
        canonical_durable_json_bytes(material, "run request lifecycle authority")
    ).hexdigest()


def run_request_with_prepared_session_authority(
    request: RunRequest,
    *,
    session_id: str,
    queue_task_id: str,
    dispatch_operation_id: str,
    terminal_event_id: str,
    interaction_id: str,
    interaction_started_event_id: str,
    idempotency_key: str,
    submission_sha256: str,
    provider_name: str,
    model: str,
    policy_evidence: bytes | None,
) -> RunRequest:
    """Bind a validated claimed queue operation to one pre-created child session."""

    copied = copy_run_request(request)
    if copied.session_id != session_id:
        raise ValueError("Prepared session authority conflicts with the run request.")
    copied._runtime_prepared_session_authority = _RuntimePreparedSessionAuthority(
        token=_RUNTIME_PREPARED_SESSION_AUTHORITY_TOKEN,
        session_id=session_id,
        queue_task_id=queue_task_id,
        dispatch_operation_id=dispatch_operation_id,
        terminal_event_id=terminal_event_id,
        interaction_id=interaction_id,
        interaction_started_event_id=interaction_started_event_id,
        idempotency_key=idempotency_key,
        submission_sha256=submission_sha256,
        provider_name=provider_name,
        model=model,
        policy_evidence=policy_evidence,
    )
    return copied


def runtime_prepared_session_authority(
    request: RunRequest,
) -> _RuntimePreparedSessionAuthority | None:
    """Return authenticated prepared-session authority, clearing malformed copies."""

    if type(request) is not RunRequest:
        raise TypeError("Prepared session authority requires a RunRequest.")
    authority = request._runtime_prepared_session_authority
    if authority is None:
        return None
    if (
        type(authority) is not _RuntimePreparedSessionAuthority
        or authority.token is not _RUNTIME_PREPARED_SESSION_AUTHORITY_TOKEN
        or authority.session_id != request.session_id
    ):
        request._runtime_prepared_session_authority = None
        return None
    return authority


def run_request_with_runtime_generated_authority(
    request: RunRequest,
    *field_names: str,
) -> RunRequest:
    """Attest exact run authority selected by a trusted runtime boundary."""

    if type(request) is not RunRequest:
        raise TypeError("Runtime authority requires a RunRequest.")
    authority = set(request._runtime_generated_authority)
    for field_name in field_names:
        if field_name not in {
            "session_id",
            "task_id",
            "parent_session_id",
            "causal_budget_id",
        }:
            raise ValueError("Unsupported runtime-generated run authority field.")
        value = getattr(request, field_name)
        if type(value) is not str or not value.strip():
            raise ValueError(
                f"RunRequest.{field_name} must be a non-empty string before attestation."
            )
        authority.add((field_name, value))
    copied = copy_run_request(request)
    copied._runtime_generated_authority = frozenset(authority)
    return copied


def run_request_authority_is_runtime_generated(
    request: RunRequest,
    *,
    field_name: str,
    value: str,
) -> bool:
    """Return positive in-process provenance for exact generated run authority."""

    return (
        type(request) is RunRequest
        and type(field_name) is str
        and type(value) is str
        and getattr(request, field_name, None) == value
        and (field_name, value) in request._runtime_generated_authority
    )
