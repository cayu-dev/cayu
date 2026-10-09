"""Shared model dispatch, recovery-context and completion publication contracts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._validation import (
    DURABLE_DOCUMENT_LIMITS,
    copy_durable_json_object,
    copy_durable_metadata,
    inspect_bounded_durable_json,
    require_durable_clean_nonblank,
)
from cayu.budgets._run_limit_accounting import (
    RunLimitAccountingContext,
    has_run_limit_accounting_authority,
)
from cayu.budgets.base import (
    MAX_REQUEST_BUDGET_LIMITS,
    BudgetLimit,
    BudgetReservationRecoveryContext,
    copy_request_budget_limits,
)
from cayu.budgets.billing import BillingIdentity, copy_billing_identity
from cayu.budgets.run_limits import RunLimits
from cayu.configuration import MAX_STEPS
from cayu.context.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
    StructuredOutputValidation,
)
from cayu.context.thinking import ThinkingConfig
from cayu.events import Event, EventType, copy_event
from cayu.execution_units import copy_tool_round_identity
from cayu.memory.evidence import ContextExposure, ContextExposureState
from cayu.messages import Message, detach_message
from cayu.providers.base import (
    OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL,
    TOOL_DISCOVERY_PROJECTION_MAX_TOOLS,
    ModelRequest,
    copy_model_completion,
)
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._child_session_notifications import (
    ChildSessionNotificationStageBinding,
    child_session_notification_stage_binding,
)
from cayu.runtime._memory_evidence import (
    MemoryEvidenceReference,
    context_exposure_identity_payload,
    validate_context_exposure_stage_scope,
)
from cayu.runtime._model_execution_selection import ModelFailoverAttempt
from cayu.runtime._run_limits import BudgetStepReservation
from cayu.runtime.model_steps import AssistantStepResult
from cayu.runtime.tool_completion import ToolCompletionPolicy
from cayu.sessions import _model_completion_publication as model_completion_publication
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions.base import (
    MODEL_COMPLETION_RECOVERY_CONTEXT_MAX_BYTES,
    ModelCompletionStage,
    ModelCompletionStageResult,
    RuntimePublicationOperationRecordMutation,
    RuntimePublicationReceipt,
    RuntimePublicationResult,
)
from cayu.sessions.records import Session
from cayu.tools.exposure import ResolvedToolExposureAuthority, copy_resolved_tool_exposure_authority

MAX_MODEL_COMPLETION_RECOVERY_CONTEXT_BYTES = MODEL_COMPLETION_RECOVERY_CONTEXT_MAX_BYTES
# RunRequest metadata is bounded only by DURABLE_METADATA_LIMITS and its budget
# limits by MAX_REQUEST_BUDGET_LIMITS; recovery copies both the same way, so an
# admitted run cannot fail at its first model step.
MAX_MODEL_COMPLETION_RECOVERY_BUDGET_LIMITS = MAX_REQUEST_BUDGET_LIMITS
_MAX_MODEL_COMPLETION_RECOVERY_PRICE_ENTRIES = 512
_MAX_MODEL_COMPLETION_RECOVERY_PRICING_CONTEXTS = 128
_MAX_MODEL_COMPLETION_RECOVERY_EVIDENCE_ENTRIES = 256


class HostedToolDiscoveryRecoveryAuthority(BaseModel):
    """Compact authority for reconstructing one hosted Tool Search request."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    protocol: Literal["openai.tool_search.hosted.v1"] = OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL
    projection_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    targeted_tool_name_sha256s: tuple[str, ...] = Field(
        default=(),
        max_length=TOOL_DISCOVERY_PROJECTION_MAX_TOOLS,
    )
    loaded_tool_name_sha256s: tuple[str, ...] = Field(
        default=(),
        max_length=TOOL_DISCOVERY_PROJECTION_MAX_TOOLS,
    )

    @field_validator(
        "targeted_tool_name_sha256s",
        "loaded_tool_name_sha256s",
        mode="before",
    )
    @classmethod
    def copy_tool_name_sha256s(cls, value: object, info) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{info.field_name} must be a sequence.")
        copied = tuple(value)
        if any(
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in copied
        ):
            raise ValueError(f"{info.field_name} must contain SHA-256 digests.")
        if copied != tuple(sorted(set(copied))):
            raise ValueError(f"{info.field_name} must be unique and sorted.")
        return cast("tuple[str, ...]", copied)


_MODEL_COMPLETION_RECOVERY_V1_DEFAULT_MAX_STEPS = 16


class ModelCompletionRecoveryContext(BaseModel):
    """Secret-free run semantics required to publish an offline completion."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    interaction_id: str | None = Field(default=None, min_length=1, max_length=256)
    execution_profile_fingerprint: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    tool_exposure: ResolvedToolExposureAuthority | None = None
    hosted_tool_discovery: HostedToolDiscoveryRecoveryAuthority | None = None
    task_id: str | None = None
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    tool_completion: ToolCompletionPolicy | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    # A missing field can be a persisted schema-v1 payload, so changing this
    # default would rewrite historical run semantics during recovery.
    max_steps: StrictInt = Field(
        default=_MODEL_COMPLETION_RECOVERY_V1_DEFAULT_MAX_STEPS,
        ge=1,
        le=MAX_STEPS,
    )
    limits: RunLimits = Field(default_factory=RunLimits)
    run_limit_accounting: RunLimitAccountingContext | None = None
    budget_limits: tuple[BudgetLimit, ...] = ()
    budget_reservations: tuple[BudgetReservationRecoveryContext, ...] = ()
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    structured_output_attempt: StrictInt | None = Field(default=None, ge=1)
    billing_identity: BillingIdentity | None = None

    @field_validator("interaction_id", "task_id")
    @classmethod
    def validate_recovery_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("request_metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: object) -> dict[str, Any]:
        return copy_durable_metadata(value, "request_metadata")

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value: Any) -> tuple[BudgetLimit, ...]:
        if (
            type(value) in (list, tuple)
            and len(value) > MAX_MODEL_COMPLETION_RECOVERY_BUDGET_LIMITS
        ):
            raise ValueError(
                "budget_limits cannot contain more than "
                f"{MAX_MODEL_COMPLETION_RECOVERY_BUDGET_LIMITS} limits."
            )
        return copy_request_budget_limits(value)

    @field_validator("budget_reservations", mode="before")
    @classmethod
    def copy_budget_reservations(
        cls,
        value: Any,
    ) -> tuple[BudgetReservationRecoveryContext, ...]:
        if type(value) not in (list, tuple):
            raise TypeError("budget_reservations must be a list or tuple.")
        if len(value) > MAX_MODEL_COMPLETION_RECOVERY_BUDGET_LIMITS:
            raise ValueError(
                "budget_reservations cannot contain more than "
                f"{MAX_MODEL_COMPLETION_RECOVERY_BUDGET_LIMITS} entries."
            )
        copied = tuple(BudgetReservationRecoveryContext.model_validate(item) for item in value)
        reservation_ids = [item.reservation_id for item in copied]
        budget_limit_ids = [item.budget_limit_id for item in copied]
        if len(set(reservation_ids)) != len(reservation_ids):
            raise ValueError("budget_reservations must not repeat reservation ids.")
        if len(set(budget_limit_ids)) != len(budget_limit_ids):
            raise ValueError("budget_reservations must not repeat budget limit ids.")
        return copied

    @field_validator("billing_identity", mode="after")
    @classmethod
    def copy_context_billing_identity(
        cls,
        value: BillingIdentity | None,
    ) -> BillingIdentity | None:
        return copy_billing_identity(value)

    @model_validator(mode="after")
    def validate_durable_bounds(self) -> ModelCompletionRecoveryContext:
        if self.run_limit_accounting is not None and not has_run_limit_accounting_authority(
            self.limits,
            self.budget_limits,
        ):
            raise ValueError("run_limit_accounting requires active run-scoped authority.")
        for limit in (
            *self.budget_limits,
            *(
                reservation.limit
                for reservation in self.budget_reservations
                if reservation.limit is not None
            ),
        ):
            price_book = limit.pricing
            if (
                len(price_book.prices) > _MAX_MODEL_COMPLETION_RECOVERY_PRICE_ENTRIES
                or len(price_book.resource_mappings) > _MAX_MODEL_COMPLETION_RECOVERY_PRICE_ENTRIES
                or len(price_book.contextual_pricing_requirements)
                > _MAX_MODEL_COMPLETION_RECOVERY_PRICE_ENTRIES
            ):
                raise ValueError("budget limit pricing collections exceed recovery bounds.")
        if self.billing_identity is not None:
            identity = self.billing_identity
            if (
                len(identity.request_evidence) > _MAX_MODEL_COMPLETION_RECOVERY_EVIDENCE_ENTRIES
                or len(identity.completion_evidence)
                > _MAX_MODEL_COMPLETION_RECOVERY_EVIDENCE_ENTRIES
                or len(identity.pricing_contexts) > _MAX_MODEL_COMPLETION_RECOVERY_PRICING_CONTEXTS
                or any(
                    len(context.dimensions) > _MAX_MODEL_COMPLETION_RECOVERY_EVIDENCE_ENTRIES
                    for context in identity.pricing_contexts
                )
            ):
                raise ValueError("billing identity collections exceed recovery bounds.")
        inspect_bounded_durable_json(
            self.model_dump(mode="json"),
            "model_completion_recovery_context",
            max_bytes=MAX_MODEL_COMPLETION_RECOVERY_CONTEXT_BYTES,
            max_nodes=DURABLE_DOCUMENT_LIMITS.max_nodes,
            max_nesting=DURABLE_DOCUMENT_LIMITS.max_nesting,
        )
        return self


ModelCompletionRecoveryContextFactory = Callable[
    [BillingIdentity | None, tuple[BudgetStepReservation, ...]],
    ModelCompletionRecoveryContext | None,
]


def model_completion_recovery_context_from_stage(
    stage: ModelCompletionStage,
) -> ModelCompletionRecoveryContext | None:
    """Reconstruct the typed, secret-free continuation context from one stage."""

    raw_context = stage.intent.get("recovery_context")
    if raw_context is None:
        return None
    context = ModelCompletionRecoveryContext.model_validate(
        copy_durable_json_object(raw_context, "recovery_context")
    )
    if (
        context.run_limit_accounting is not None
        and context.run_limit_accounting.baseline.session_id != stage.session_id
    ):
        raise ValueError("Model completion run-limit accounting belongs to another session.")
    return context


def _copy_model_completion_stage(stage: ModelCompletionStage) -> ModelCompletionStage:
    if type(stage) is not ModelCompletionStage:
        raise TypeError("Model completion dispatch requires a ModelCompletionStage.")
    return stage.model_copy(deep=True)


def _copy_model_completion_stage_result(
    result: ModelCompletionStageResult,
) -> ModelCompletionStageResult:
    if type(result) is not ModelCompletionStageResult:
        raise TypeError("Model completion publication requires a ModelCompletionStageResult.")
    return result.model_copy(deep=True)


def _copy_runtime_publication_result(
    result: RuntimePublicationResult,
) -> RuntimePublicationResult:
    if type(result) is not RuntimePublicationResult:
        raise TypeError("Model completion publication requires a RuntimePublicationResult.")
    return result.model_copy(deep=True)


def _copy_assistant_step_result(result: AssistantStepResult) -> AssistantStepResult:
    if type(result) is not AssistantStepResult:
        raise TypeError("Model completion publication requires an AssistantStepResult.")
    completion = copy_model_completion(result.completion)
    if completion is None:  # pragma: no cover - AssistantStepResult requires a completion
        raise RuntimeError("Assistant step result lost its completion metadata.")
    return AssistantStepResult(
        session_id=result.session_id,
        step=result.step,
        model_step_id=result.model_step_id,
        model_attempt_id=result.model_attempt_id,
        tool_round_identity=(
            None
            if result.tool_round_identity is None
            else copy_tool_round_identity(result.tool_round_identity)
        ),
        assistant_message=(
            None if result.assistant_message is None else detach_message(result.assistant_message)
        ),
        tool_calls=[
            runtime_records.ToolCallRequest(
                id=call.id,
                name=call.name,
                arguments=copy_durable_json_object(
                    call.arguments,
                    "tool_call_arguments",
                ),
                targeted_tool_grant_id=call.targeted_tool_grant_id,
                model_tool_name=call.model_tool_name,
                targeted_tool_invocation=call.targeted_tool_invocation,
                targeted_tool_rejection=call.targeted_tool_rejection,
            )
            for call in result.tool_calls
        ],
        completion=completion,
        text_content=result.text_content,
        has_user_visible_content=result.has_user_visible_content,
        provider_state_count=result.provider_state_count,
        thinking_count=result.thinking_count,
    )


@dataclass(frozen=True, slots=True)
class ModelCompletionDispatch:
    """Detached proof that one exact provider dispatch was durably prepared."""

    stage: ModelCompletionStage
    request_fingerprint: str
    context_exposure: ContextExposure | None = None
    child_session_notifications_consumed: bool = True
    prepared_events: tuple[Event, ...] = ()

    def __post_init__(self) -> None:
        stage = _copy_model_completion_stage(self.stage)
        request_fingerprint = require_durable_clean_nonblank(
            self.request_fingerprint,
            "request_fingerprint",
        )
        if len(request_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in request_fingerprint
        ):
            raise ValueError("request_fingerprint must be a lowercase SHA-256 digest.")
        if stage.state != "in_flight":
            raise ValueError("A provider dispatch requires an in-flight completion stage.")
        if stage.intent.get("request_fingerprint") != request_fingerprint:
            raise ValueError("Completion-stage intent does not match its request fingerprint.")
        exposure = self.context_exposure
        if exposure is not None:
            exposure = ContextExposure.model_validate(exposure.model_dump(mode="python"))
            if exposure.state is not ContextExposureState.DISPATCH_STARTED:
                raise ValueError("A dispatched context exposure must be dispatch_started.")
            if stage.intent.get("context_exposure") != context_exposure_identity_payload(exposure):
                raise ValueError("Completion-stage intent does not match its context exposure.")
            validate_context_exposure_stage_scope(exposure, stage.intent)
        notifications_consumed = self.child_session_notifications_consumed
        prepared_events = tuple(copy_event(event) for event in self.prepared_events)
        if len(prepared_events) > 1 or any(
            event.type is not EventType.MODEL_FAILOVER_SELECTED
            or event.session_id != stage.session_id
            or event.payload.get("stage_id") != stage.stage_id
            for event in prepared_events
        ):
            raise ValueError("Model dispatch contains conflicting preparation events.")
        object.__setattr__(self, "prepared_events", prepared_events)
        if type(notifications_consumed) is not bool:
            raise TypeError("child_session_notifications_consumed must be a boolean.")
        if (
            not notifications_consumed
            and child_session_notification_stage_binding(stage.intent) is None
        ):
            raise ValueError("An unconsumed dispatch must bind child-session notifications.")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "request_fingerprint", request_fingerprint)
        object.__setattr__(self, "context_exposure", exposure)
        object.__setattr__(
            self,
            "child_session_notifications_consumed",
            notifications_consumed,
        )

    @property
    def stage_id(self) -> str:
        return self.stage.stage_id

    @property
    def logical_step_id(self) -> str:
        return self.stage.logical_step_id

    @property
    def dispatch_ordinal(self) -> int:
        return self.stage.dispatch_ordinal

    @property
    def reservation_ids(self) -> tuple[str, ...]:
        return self.stage.reservation_ids

    @property
    def intent(self) -> dict[str, Any]:
        return copy_durable_json_object(self.stage.intent, "model_completion_intent")


class ModelCompletionDispatchPreparer(Protocol):
    def __call__(
        self,
        request: ModelRequest,
        reference: MemoryEvidenceReference | None,
        notifications: ChildSessionNotificationStageBinding | None,
        consume_notifications: bool,
        /,
        *,
        failover_attempt: ModelFailoverAttempt | None = None,
    ) -> Awaitable[ModelCompletionDispatch]: ...


@dataclass(frozen=True, slots=True)
class ModelCompletionPublicationRequest:
    """Immutable, detached terminal material handed to the session owner."""

    dispatch: ModelCompletionDispatch
    assistant_step_result: AssistantStepResult | None
    completion_event: Event
    authoritative_assistant_message: Message | None
    defer_assistant_message: bool
    structured_output_validation: StructuredOutputValidation | None
    tool_exposure: ResolvedToolExposureAuthority | None = None
    operation_record_mutations: tuple[RuntimePublicationOperationRecordMutation, ...] = ()

    def __post_init__(self) -> None:
        if type(self.dispatch) is not ModelCompletionDispatch:
            raise TypeError("dispatch must be a ModelCompletionDispatch.")
        dispatch = ModelCompletionDispatch(
            stage=self.dispatch.stage,
            request_fingerprint=self.dispatch.request_fingerprint,
            context_exposure=self.dispatch.context_exposure,
            child_session_notifications_consumed=(
                self.dispatch.child_session_notifications_consumed
            ),
            prepared_events=self.dispatch.prepared_events,
        )
        if not dispatch.child_session_notifications_consumed:
            raise ValueError(
                "A model completion cannot be published before child notifications "
                "cross the provider-start fence."
            )
        result = (
            None
            if self.assistant_step_result is None
            else _copy_assistant_step_result(self.assistant_step_result)
        )
        event = copy_event(self.completion_event)
        assistant_message = (
            None
            if self.authoritative_assistant_message is None
            else detach_message(self.authoritative_assistant_message)
        )
        if type(self.defer_assistant_message) is not bool:
            raise TypeError("defer_assistant_message must be a bool.")
        if (
            self.structured_output_validation is not None
            and type(self.structured_output_validation) is not StructuredOutputValidation
        ):
            raise TypeError("structured_output_validation must be a StructuredOutputValidation.")
        structured_output_validation = (
            None
            if self.structured_output_validation is None
            else self.structured_output_validation.model_copy(deep=True)
        )
        tool_exposure = (
            None
            if self.tool_exposure is None
            else copy_resolved_tool_exposure_authority(self.tool_exposure)
        )
        operation_record_mutations = tuple(
            RuntimePublicationOperationRecordMutation.model_validate(
                mutation.model_dump(mode="python")
                if type(mutation) is RuntimePublicationOperationRecordMutation
                else mutation
            )
            for mutation in self.operation_record_mutations
        )
        if result is not None and result.session_id != dispatch.stage.session_id:
            raise ValueError("Assistant result session does not match its completion stage.")
        if event.session_id != dispatch.stage.session_id:
            raise ValueError("Completion event session does not match its completion stage.")
        if event.type != EventType.MODEL_COMPLETED:
            raise ValueError("Completion publication requires a model.completed event.")
        if assistant_message is not None:
            if result is None:
                raise ValueError(
                    "An authoritative assistant message requires an assistant step result."
                )
            if assistant_message != result.assistant_message:
                raise ValueError(
                    "Authoritative assistant message does not match the detached step result."
                )
        if self.defer_assistant_message and (
            result is None or assistant_message is None or not result.tool_calls
        ):
            raise ValueError(
                "Deferred assistant publication requires an ordinary tool-call message."
            )
        if structured_output_validation is not None and (
            result is None
            or assistant_message is None
            or not any(call.name == STRUCTURED_OUTPUT_TOOL_NAME for call in result.tool_calls)
        ):
            raise ValueError(
                "Structured-output validation requires a published finalizer tool round."
            )
        object.__setattr__(self, "dispatch", dispatch)
        object.__setattr__(self, "assistant_step_result", result)
        object.__setattr__(self, "completion_event", event)
        object.__setattr__(self, "authoritative_assistant_message", assistant_message)
        object.__setattr__(self, "defer_assistant_message", self.defer_assistant_message)
        object.__setattr__(
            self,
            "structured_output_validation",
            structured_output_validation,
        )
        object.__setattr__(self, "tool_exposure", tool_exposure)
        object.__setattr__(self, "operation_record_mutations", operation_record_mutations)


@dataclass(frozen=True, slots=True)
class ModelCompletionPublicationResult:
    """Durable terminal-stage and atomic-promotion acknowledgements."""

    completion: ModelCompletionStageResult
    publication: RuntimePublicationResult

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "completion",
            _copy_model_completion_stage_result(self.completion),
        )
        object.__setattr__(
            self,
            "publication",
            _copy_runtime_publication_result(self.publication),
        )


ModelCompletionPublisher = Callable[
    [ModelCompletionPublicationRequest],
    Awaitable[ModelCompletionPublicationResult],
]


class ModelCompletionDispatchNotAuthorized(RuntimeError):
    """A prior or ambiguous preparation must never cause another provider call."""

    def __init__(
        self,
        *,
        stage: ModelCompletionStage,
        request_fingerprint: str,
    ) -> None:
        self.stage = _copy_model_completion_stage(stage)
        self.request_fingerprint = require_durable_clean_nonblank(
            request_fingerprint,
            "request_fingerprint",
        )
        super().__init__(
            "Model completion dispatch was not authorized because its durable "
            f"stage already exists: {stage.stage_id}."
        )


class ModelCompletionManualRecoveryRequired(RuntimeError):
    """A model dispatch cannot be reconstructed safely without operator input."""

    def __init__(self, message: str) -> None:
        super().__init__(
            f"{message} Inspect the registered application with `cayu recovery plan` "
            "before selecting an operator recovery decision."
        )


@dataclass(frozen=True)
class ModelCompletionBoundaryReconciliation:
    """Verified state at the durable model-completion publication boundary."""

    state: Literal[
        "none",
        "prepared_abandoned",
        "promoted",
        "already_promoted",
        "provider_operation_pending",
        "provider_operation_reconciled",
        "provider_operation_unavailable",
    ]
    session: Session
    pointer: model_completion_publication.ModelStepPublicationCheckpoint | None = None
    completion_event: Event | None = None
    pending_tool_round: pending_rounds.PendingToolRound | None = None
    transcript_cursor: int = 0
    recovery_events: tuple[Event, ...] = ()
    # Retain the exact validated publication and its original execution
    # identities for governed replay; this is evidence, not dispatch authority.
    completed_stage: ModelCompletionStage | None = None
    closed_tool_receipt: RuntimePublicationReceipt | None = None
    structured_output_events: tuple[Event, ...] = ()

    @property
    def blocks_provider_dispatch(self) -> bool:
        return (
            self.pointer is not None
            and self.pending_tool_round is None
            and self.transcript_cursor == self.pointer.transcript_end_cursor
        )
