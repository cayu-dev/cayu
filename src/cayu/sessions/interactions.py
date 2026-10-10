from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import (
    copy_durable_json_object,
    require_clean_nonblank,
    require_durable_clean_nonblank,
)
from cayu.budgets.usage import (
    USAGE_BEARING_EVENT_TYPES,
    AggregateUsageMetrics,
    SessionUsageSummary,
    aggregate_usage_metrics_from_durable_payload,
    build_aggregate_usage_metrics,
    session_usage_summary,
)
from cayu.events import Event, EventType, copy_event
from cayu.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileDecision,
    ExecutionProfileDecisionKind,
    ExecutionProfileIdentity,
    copy_execution_profile_decision,
    execution_profile_runtime_component,
)
from cayu.messages import Message, detach_message
from cayu.runtime.tool_completion import ToolCompletionResult
from cayu.sessions import _completion_finalization as completion_finalization
from cayu.sessions import records as session_record_rules
from cayu.sessions._execution_profile_checkpoint import ActiveInvocationExecutionProfile
from cayu.sessions._execution_profile_rules import SessionModelTransition
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalDecision,
    InvocationTerminalOutcome,
)
from cayu.sessions._model_completion_publication import ModelCompletionStageSettlementRequest
from cayu.sessions.records import (
    CheckpointTransform,
    Session,
    SessionRuntimeIdentity,
    SessionStatus,
    StoreTimeCheckpointTransform,
    copy_session,
    copy_session_runtime_identity,
)
from cayu.sessions.transcript_input import copy_transcript_messages
from cayu.tools.exposure import ToolCapabilityCeiling, copy_tool_capability_ceiling
from cayu.tools.grants import TARGETED_TOOL_GRANT_MAX_REQUESTS

if TYPE_CHECKING:
    from cayu.sessions._temporary_continuation import TemporaryServiceAdmission


class InteractionStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


INTERACTION_LIFECYCLE_EVENT_TYPES: tuple[EventType, ...] = (
    EventType.INTERACTION_STARTED,
    EventType.INTERACTION_RESUMED,
    EventType.INTERACTION_PAUSED,
    EventType.INTERACTION_COMPLETED,
    EventType.INTERACTION_FAILED,
    EventType.INTERACTION_INTERRUPTED,
)
INTERACTION_TERMINAL_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.INTERACTION_COMPLETED,
        EventType.INTERACTION_FAILED,
        EventType.INTERACTION_INTERRUPTED,
    }
)
INTERACTION_OPEN_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.INTERACTION_STARTED,
        EventType.INTERACTION_RESUMED,
        EventType.INTERACTION_PAUSED,
    }
)
INTERACTION_SUMMARY_EVENT_TYPES: tuple[EventType, ...] = (
    EventType.MODEL_STARTED,
    *USAGE_BEARING_EVENT_TYPES,
)


def interaction_usage_summary(
    session_id: str,
    events: list[Event],
) -> SessionUsageSummary:
    """Aggregate response-scoped attempts, tool calls, and successful usage."""

    usage_summary = session_usage_summary(session_id, events)
    provider_names: list[str] = []
    models: list[str] = []
    model_steps = 0
    for event in events:
        if event.type != EventType.MODEL_STARTED:
            continue
        model_steps += 1
        for target, value in (
            (provider_names, event.payload.get("provider")),
            (models, event.payload.get("model")),
        ):
            if type(value) is str and value.strip() and value not in target:
                target.append(value)
    for target, values in (
        (provider_names, usage_summary.provider_names),
        (models, usage_summary.models),
    ):
        for value in values:
            if value not in target:
                target.append(value)
    return usage_summary.model_copy(
        update={
            "model_steps": model_steps,
            "provider_names": provider_names,
            "models": models,
        }
    )


class InteractionSummaryEvidence(BaseModel):
    """Validated, durable projection carried by interaction lifecycle events."""

    model_config = ConfigDict(extra="forbid")

    model_policy: dict[str, object] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("model_policy", mode="before")
    @classmethod
    def validate_model_policy(cls, value):
        if value is None:
            return None
        from cayu._validation import copy_durable_metadata

        return copy_durable_metadata(value, "model_policy")

    tool_completion: ToolCompletionResult | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    status: InteractionStatus
    start_event_id: str
    start_event_sequence: StrictInt | None = Field(default=None, ge=1)
    source_transcript_start: StrictInt | None = Field(default=None, ge=0)
    source_transcript_end: StrictInt | None = Field(default=None, ge=0)
    result_transcript_start: StrictInt | None = Field(default=None, ge=0)
    result_transcript_end: StrictInt | None = Field(default=None, ge=0)
    started_at: datetime
    completed_at: datetime | None = None
    active_duration_ms: StrictInt = Field(default=0, ge=0)
    wall_duration_ms: StrictInt | None = Field(default=None, ge=0)
    model_step_count: StrictInt = Field(default=0, ge=0)
    tool_call_count: StrictInt = Field(default=0, ge=0)
    token_usage: AggregateUsageMetrics = Field(default_factory=build_aggregate_usage_metrics)
    provider_names: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    pending_action_kind: str | None = None
    targeted_tool_grant_count: StrictInt | None = Field(
        default=None,
        ge=1,
        le=TARGETED_TOOL_GRANT_MAX_REQUESTS,
        exclude_if=lambda value: value is None,
    )
    targeted_tool_grant_batch_fingerprint: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )
    queued_interaction_profile_handoff: dict[str, str | int] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @field_validator("token_usage", mode="before")
    @classmethod
    def validate_token_usage(cls, value: object) -> AggregateUsageMetrics:
        if type(value) is AggregateUsageMetrics:
            return value
        return aggregate_usage_metrics_from_durable_payload(value)

    @field_validator("start_event_id", "pending_action_kind")
    @classmethod
    def validate_optional_nonblank(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("provider_names", "models")
    @classmethod
    def validate_names(cls, value: list[str], info) -> list[str]:
        names = [
            require_clean_nonblank(item, f"{info.field_name}[{index}]")
            for index, item in enumerate(value)
        ]
        if len(names) != len(set(names)):
            raise ValueError(f"{info.field_name} must not contain duplicates.")
        return names

    @field_validator("started_at", "completed_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_ranges_and_terminal_time(self) -> InteractionSummaryEvidence:
        if self.tool_completion is not None and self.status is not InteractionStatus.COMPLETED:
            raise ValueError("Tool completion requires a completed interaction.")
        if (self.targeted_tool_grant_count is None) != (
            self.targeted_tool_grant_batch_fingerprint is None
        ):
            raise ValueError("Targeted grant count and batch fingerprint must be present together.")
        for label, start, end in (
            ("source transcript", self.source_transcript_start, self.source_transcript_end),
            ("result transcript", self.result_transcript_start, self.result_transcript_end),
        ):
            if (start is None) != (end is None):
                raise ValueError(f"{label} range endpoints must both be present or absent.")
            if start is not None and end is not None and start > end:
                raise ValueError(f"{label} range start must not exceed its end.")
        terminal = self.status in {
            InteractionStatus.COMPLETED,
            InteractionStatus.FAILED,
            InteractionStatus.INTERRUPTED,
        }
        if terminal != (self.completed_at is not None):
            raise ValueError("completed_at must be present exactly for terminal interactions.")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at.")
        return self


class QueuedInteractionProfileHandoff(BaseModel):
    """Exact authority for one same-epoch queued interaction handoff."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    schema_version: Literal[1] = 1
    expected_session_instance_id: str
    predecessor_settlement_event_id: str
    expected_active_profile: ActiveInvocationExecutionProfile
    target_active_profile: ActiveInvocationExecutionProfile

    @field_validator("expected_session_instance_id", "predecessor_settlement_event_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_handoff(self) -> QueuedInteractionProfileHandoff:
        expected = self.expected_active_profile
        target = self.target_active_profile
        if (
            expected.session_id != target.session_id
            or expected.run_epoch != target.run_epoch
            or expected.profile != target.profile
        ):
            raise ValueError(
                "Queued interaction handoff must retain the exact session, epoch, and profile."
            )
        if expected.interaction_id == target.interaction_id:
            raise ValueError("Queued interaction handoff requires a new interaction.")
        return self


QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY = "queued_interaction_profile_handoff"


def queued_interaction_profile_handoff_evidence(
    handoff: QueuedInteractionProfileHandoff,
) -> dict[str, Any]:
    """Return bounded public evidence that binds a delivery receipt to its CAS."""

    copied = QueuedInteractionProfileHandoff.model_validate(handoff)
    expected = copied.expected_active_profile
    target = copied.target_active_profile
    return {
        "schema_version": 1,
        "session_instance_id": copied.expected_session_instance_id,
        "run_epoch": expected.run_epoch,
        "predecessor_interaction_id": expected.interaction_id,
        "predecessor_settlement_event_id": copied.predecessor_settlement_event_id,
        "target_interaction_id": target.interaction_id,
        "execution_profile_fingerprint": target.profile.fingerprint,
    }


class InteractionTransitionResult(BaseModel):
    """Exact result of an atomic interaction-lifecycle/session transition."""

    model_config = ConfigDict(extra="forbid")

    session: Session
    event: Event
    terminal_event: Event | None = None
    status_changed: StrictBool
    replayed: StrictBool = False

    @field_validator("session")
    @classmethod
    def copy_session(cls, value: Session) -> Session:
        return copy_session(value)

    @field_validator("event")
    @classmethod
    def copy_event(cls, value: Event) -> Event:
        return copy_event(value)

    @field_validator("terminal_event")
    @classmethod
    def copy_terminal_event(cls, value: Event | None) -> Event | None:
        return None if value is None else copy_event(value)


class InteractionTransitionSpec(BaseModel):
    """Complete caller-owned identity of one interaction transition."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    event: Event
    from_statuses: tuple[SessionStatus, ...]
    to_status: SessionStatus
    only_if_no_queued_messages: StrictBool = False
    model_completion_stage_settlement: ModelCompletionStageSettlementRequest | None = None
    checkpoint_mutation: dict[str, Any] | None = None
    terminal_event: Event | None = None
    terminal_decision: InvocationTerminalDecision | None = None

    @field_validator("event")
    @classmethod
    def copy_event(cls, value: Event) -> Event:
        return copy_event(value)

    @field_validator("terminal_event")
    @classmethod
    def copy_terminal_event(cls, value: Event | None) -> Event | None:
        return None if value is None else copy_event(value)

    @field_validator("terminal_decision")
    @classmethod
    def copy_terminal_decision(
        cls,
        value: InvocationTerminalDecision | None,
    ) -> InvocationTerminalDecision | None:
        return None if value is None else value.model_copy(deep=True)

    @field_validator("from_statuses")
    @classmethod
    def canonicalize_source_statuses(
        cls,
        value: tuple[SessionStatus, ...],
    ) -> tuple[SessionStatus, ...]:
        if not value:
            raise ValueError("Interaction transition requires source statuses.")
        if len(set(value)) != len(value):
            raise ValueError("Interaction transition source statuses must be unique.")
        return tuple(sorted(value, key=str))

    @field_validator("checkpoint_mutation", mode="before")
    @classmethod
    def copy_checkpoint_mutation(cls, value: object) -> dict[str, Any] | None:
        if value is None:
            return None
        if type(value) is not dict:
            raise TypeError("checkpoint_mutation must be an object or None.")
        return copy_durable_json_object(value, "checkpoint_mutation")

    @model_validator(mode="after")
    def validate_transition_shape(self) -> InteractionTransitionSpec:
        if self.event.interaction_id is None:
            raise ValueError("Interaction transition event must have an interaction_id.")
        statuses_by_event_type: dict[str, set[SessionStatus]] = {
            EventType.INTERACTION_PAUSED: {
                SessionStatus.FAILED,
                SessionStatus.INTERRUPTED,
            },
            EventType.INTERACTION_COMPLETED: {
                SessionStatus.RUNNING,
                SessionStatus.COMPLETED,
            },
            EventType.INTERACTION_FAILED: {SessionStatus.FAILED},
            EventType.INTERACTION_INTERRUPTED: {SessionStatus.INTERRUPTED},
        }
        expected_statuses = statuses_by_event_type.get(self.event.type)
        if expected_statuses is None:
            raise ValueError("event must be a terminal or paused interaction lifecycle event.")
        if self.to_status not in expected_statuses:
            rendered_statuses = ", ".join(sorted(str(status) for status in expected_statuses))
            raise ValueError(
                f"{self.event.type} requires session status in "
                f"{{{rendered_statuses}}}, not {self.to_status}."
            )
        if self.only_if_no_queued_messages and self.event.type != EventType.INTERACTION_COMPLETED:
            raise ValueError("only_if_no_queued_messages is valid only for interaction.completed.")
        if self.to_status is SessionStatus.RUNNING and (
            self.event.type != EventType.INTERACTION_COMPLETED
            or not self.only_if_no_queued_messages
            or self.checkpoint_mutation is None
        ):
            raise ValueError(
                "A running interaction completion requires a queue-conditional checkpoint mutation."
            )
        if self.checkpoint_mutation is not None and (
            self.event.type != EventType.INTERACTION_COMPLETED
            or not self.only_if_no_queued_messages
            or self.to_status is not SessionStatus.RUNNING
        ):
            raise ValueError(
                "checkpoint_mutation is valid only for a queue-conditional running "
                "interaction completion."
            )
        if self.checkpoint_mutation is not None:
            completion_finalization._validate_completion_finalization_mutation_shape(
                self.checkpoint_mutation
            )
        if self.model_completion_stage_settlement is not None:
            if self.only_if_no_queued_messages:
                raise ValueError(
                    "A model-completion settlement cannot use a conditional interaction transition."
                )
            if self.to_status not in {SessionStatus.FAILED, SessionStatus.INTERRUPTED}:
                raise ValueError(
                    "A model-completion settlement requires a failed or interrupted transition."
                )
            if self.model_completion_stage_settlement.interaction_id != self.event.interaction_id:
                raise ValueError("Model-completion settlement belongs to a different interaction.")
        if (self.terminal_event is None) != (self.terminal_decision is None):
            raise ValueError(
                "A terminal session event and terminal decision must be supplied together."
            )
        if self.terminal_event is not None:
            terminal_event = self.terminal_event
            decision = self.terminal_decision
            assert decision is not None
            expected_event_type = {
                SessionStatus.FAILED: EventType.SESSION_FAILED,
                SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
            }.get(self.to_status)
            expected_outcome = {
                SessionStatus.FAILED: InvocationTerminalOutcome.FAILED,
                SessionStatus.INTERRUPTED: InvocationTerminalOutcome.INTERRUPTED,
            }.get(self.to_status)
            if expected_event_type is None or expected_outcome is None:
                raise ValueError(
                    "Paired terminal publication requires failed or interrupted status."
                )
            if (
                terminal_event.type is not expected_event_type
                or terminal_event.session_id != self.event.session_id
                or terminal_event.interaction_id is not None
                or decision.interaction_event_id is None
                or terminal_event.id != decision.terminal_event_id
                or self.event.id != decision.interaction_event_id
                or self.event.interaction_id != decision.interaction_id
                or decision.session_id != self.event.session_id
                or decision.outcome is not expected_outcome
                or terminal_event.timestamp != decision.observed_at
                or self.event.timestamp != decision.observed_at
                or any(
                    terminal_event.payload.get(key) != value
                    for key, value in decision.terminal_payload.items()
                )
            ):
                raise ValueError("Paired terminal publication conflicts with its decision.")
        return self


class InteractionTransitionReceiptResult(BaseModel):
    """Read-only exact receipt evidence for one interaction transition."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    session: Session
    transition: InteractionTransitionSpec
    status_changed: StrictBool
    replayed: Literal[True] = True

    @field_validator("session")
    @classmethod
    def copy_session(cls, value: Session) -> Session:
        return copy_session(value)

    @field_validator("transition")
    @classmethod
    def copy_transition(
        cls,
        value: InteractionTransitionSpec,
    ) -> InteractionTransitionSpec:
        return copy_interaction_transition_spec(value)

    @model_validator(mode="after")
    def validate_result(self) -> InteractionTransitionReceiptResult:
        if self.session.id != self.transition.event.session_id:
            raise ValueError("Interaction transition receipt belongs to a different session.")
        if self.status_changed:
            if self.session.status is not self.transition.to_status:
                raise ValueError(
                    "Interaction transition receipt changed status does not match its session."
                )
        elif (
            not self.transition.only_if_no_queued_messages
            or self.session.status not in self.transition.from_statuses
        ):
            raise ValueError("Interaction transition receipt unchanged status is inconsistent.")
        return self


@dataclass(frozen=True)
class SessionInvocationAdmission:
    """One valid atomic admission of a profiled session invocation.

    A clean resume starts a new interaction and compares ``execution_profile``
    with session metadata. A continuation keeps the historical interaction and
    compares ``expected_active_invocation_profile`` with checkpoint authority.
    The store binds either profile to the newly claimed run epoch in the same
    transaction as status and interaction admission.
    Exactly one checkpoint callback is required; the store-time callback receives
    the receiving owner's time inside that transaction for deadline validation.
    """

    from_statuses: frozenset[SessionStatus]
    checkpoint_transform: CheckpointTransform | None
    execution_profile: ExecutionProfileIdentity
    interaction_source_messages: tuple[Message, ...]
    tool_capability_ceiling: ToolCapabilityCeiling
    result_checkpoint_transform: CheckpointTransform | None = None
    interaction_started_event: Event | None = None
    continued_interaction_id: str | None = None
    defer_interaction_source: bool = False
    model_transition: SessionModelTransition | None = None
    execution_profile_decision: ExecutionProfileDecision | None = None
    adopted_runtime_identity: SessionRuntimeIdentity | None = None
    expected_active_invocation_profile: ActiveInvocationExecutionProfile | None = None
    allow_pending_initial_interaction: bool = False
    temporary_service_admission: TemporaryServiceAdmission | None = None
    store_time_checkpoint_transform: StoreTimeCheckpointTransform | None = None

    def __post_init__(self) -> None:
        if self.temporary_service_admission is not None:
            from cayu.collaboration._preparation import prepare_contract
            from cayu.sessions._temporary_continuation import TemporaryServiceAdmission
            from cayu.sessions._temporary_continuation_scope import require_temporary_transition
            from cayu.vaults.redaction import SecretRedactor

            prepared_service = prepare_contract(
                TemporaryServiceAdmission,
                self.temporary_service_admission,
                redactor=SecretRedactor(),
            )
            require_temporary_transition(prepared_service)
            object.__setattr__(self, "temporary_service_admission", prepared_service)
        if type(self.from_statuses) is not frozenset or not self.from_statuses:
            raise ValueError("from_statuses must be a non-empty frozenset.")
        if any(not isinstance(status, SessionStatus) for status in self.from_statuses):
            raise TypeError("from_statuses must contain only SessionStatus values.")
        if (self.checkpoint_transform is None) == (self.store_time_checkpoint_transform is None):
            raise TypeError("Exactly one admission checkpoint transform is required.")
        selected_transform = (
            self.checkpoint_transform
            if self.checkpoint_transform is not None
            else self.store_time_checkpoint_transform
        )
        if not callable(selected_transform):
            raise TypeError("Admission checkpoint transform must be callable.")
        if self.result_checkpoint_transform is not None and not callable(
            self.result_checkpoint_transform
        ):
            raise TypeError("result_checkpoint_transform must be callable.")
        if type(self.execution_profile) is not ExecutionProfileIdentity:
            raise TypeError("execution_profile must be an ExecutionProfileIdentity.")
        object.__setattr__(
            self,
            "execution_profile",
            ExecutionProfileIdentity.model_validate(self.execution_profile.model_dump(mode="json")),
        )
        object.__setattr__(
            self,
            "tool_capability_ceiling",
            copy_tool_capability_ceiling(self.tool_capability_ceiling),
        )
        if type(self.allow_pending_initial_interaction) is not bool:
            raise TypeError("allow_pending_initial_interaction must be a boolean.")
        if type(self.interaction_source_messages) is not tuple:
            raise TypeError("interaction_source_messages must be a tuple.")
        object.__setattr__(
            self,
            "interaction_source_messages",
            tuple(detach_message(message) for message in self.interaction_source_messages),
        )
        if self.interaction_started_event is not None:
            if type(self.interaction_started_event) is not Event:
                raise TypeError("interaction_started_event must be an Event.")
            object.__setattr__(
                self,
                "interaction_started_event",
                copy_event(self.interaction_started_event),
            )
        if (
            self.model_transition is not None
            and type(self.model_transition) is not SessionModelTransition
        ):
            raise TypeError("model_transition must be a SessionModelTransition.")
        if (
            self.execution_profile_decision is not None
            and type(self.execution_profile_decision) is not ExecutionProfileDecision
        ):
            raise TypeError("execution_profile_decision must be an ExecutionProfileDecision.")
        if self.execution_profile_decision is not None:
            object.__setattr__(
                self,
                "execution_profile_decision",
                copy_execution_profile_decision(self.execution_profile_decision),
            )
        if self.adopted_runtime_identity is not None:
            object.__setattr__(
                self,
                "adopted_runtime_identity",
                copy_session_runtime_identity(self.adopted_runtime_identity),
            )
        decision = self.execution_profile_decision
        runtime_changed = (
            decision is not None
            and ExecutionProfileComponentClass.RUNTIME in decision.changed_component_classes
        )
        requires_runtime_identity = (
            runtime_changed
            and decision is not None
            and decision.kind is ExecutionProfileDecisionKind.ADOPTED
        )
        if requires_runtime_identity != (self.adopted_runtime_identity is not None):
            raise ValueError(
                "An adopted runtime profile change requires its exact target runtime identity."
            )
        if self.adopted_runtime_identity is not None:
            if decision is None or decision.kind is not ExecutionProfileDecisionKind.ADOPTED:
                raise ValueError("Runtime identity can change only through an adopted profile.")
            runtime_component = execution_profile_runtime_component(
                self.adopted_runtime_identity.runtime_name,
                self.adopted_runtime_identity.runtime_version,
                self.adopted_runtime_identity.runtime_build_provenance,
            )
            if self.execution_profile.component(ExecutionProfileComponentClass.RUNTIME) != (
                runtime_component
            ):
                raise ValueError("Adopted runtime identity conflicts with the target profile.")
        if (
            self.expected_active_invocation_profile is not None
            and type(self.expected_active_invocation_profile)
            is not ActiveInvocationExecutionProfile
        ):
            raise TypeError(
                "expected_active_invocation_profile must be an ActiveInvocationExecutionProfile."
            )
        if self.expected_active_invocation_profile is not None:
            object.__setattr__(
                self,
                "expected_active_invocation_profile",
                ActiveInvocationExecutionProfile.model_validate(
                    self.expected_active_invocation_profile.model_dump(mode="json")
                ),
            )
        starts_interaction = self.interaction_started_event is not None
        continues_interaction = self.continued_interaction_id is not None
        if starts_interaction == continues_interaction:
            raise ValueError(
                "Invocation admission requires exactly one new or continued interaction."
            )
        if starts_interaction:
            if self.defer_interaction_source and not self.allow_pending_initial_interaction:
                raise ValueError("A new interaction cannot defer its source messages.")
            if self.allow_pending_initial_interaction and (
                self.from_statuses != frozenset({SessionStatus.PENDING})
                or not self.defer_interaction_source
                or self.execution_profile_decision is not None
            ):
                raise ValueError("A prepared initial interaction requires exact PENDING admission.")
            return
        if self.allow_pending_initial_interaction:
            raise ValueError("Only a new interaction can admit a prepared PENDING session.")
        if self.expected_active_invocation_profile is None:
            raise ValueError(
                "A continued interaction requires active invocation profile authority."
            )
        if self.execution_profile_decision is not None:
            raise ValueError("A continued interaction cannot adopt an execution profile.")
        if self.model_transition is not None:
            raise ValueError("A continued interaction cannot change its model target.")
        if not self.defer_interaction_source:
            raise ValueError("A continued interaction must defer its source messages.")
        continued_interaction_id = self.continued_interaction_id
        if continued_interaction_id is None:
            raise AssertionError("Continued invocation admission lost its interaction identity.")
        continued_interaction_id = require_durable_clean_nonblank(
            continued_interaction_id,
            "continued_interaction_id",
        )
        active_profile = self.expected_active_invocation_profile
        if active_profile.interaction_id != continued_interaction_id:
            raise ValueError(
                "Continued interaction identity conflicts with active profile authority."
            )
        if active_profile.profile != self.execution_profile:
            raise ValueError("Continued execution profile conflicts with active profile authority.")


def _copy_queued_interaction_profile_handoff(
    session_id: str,
    delivery_id: str,
    interaction_id: str | None,
    interaction_started_event: Event | None,
    handoff: QueuedInteractionProfileHandoff | None,
) -> QueuedInteractionProfileHandoff | None:
    if handoff is None:
        if (
            interaction_started_event is not None
            and QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY in interaction_started_event.payload
        ):
            raise ValueError(
                "Queued interaction start evidence requires its atomic profile handoff."
            )
        return None
    copied = QueuedInteractionProfileHandoff.model_validate(
        handoff.model_dump(mode="json")
        if isinstance(handoff, QueuedInteractionProfileHandoff)
        else handoff
    )
    if (
        copied.expected_active_profile.session_id != session_id
        or copied.target_active_profile.session_id != session_id
        or interaction_id != copied.target_active_profile.interaction_id
        or interaction_started_event is None
        or interaction_started_event.interaction_id != interaction_id
        or delivery_id != interaction_id
        or interaction_started_event.payload.get(QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY)
        != queued_interaction_profile_handoff_evidence(copied)
    ):
        raise ValueError(
            "Queued interaction profile handoff conflicts with its first delivery identity."
        )
    return copied


def _copy_historical_queued_interaction_profile_handoff(
    session_id: str,
    interaction_started_event: Event,
    handoff: QueuedInteractionProfileHandoff,
) -> tuple[Event, QueuedInteractionProfileHandoff]:
    """Validate repair input while accepting legacy starts without handoff evidence."""

    copied_handoff = QueuedInteractionProfileHandoff.model_validate(
        handoff.model_dump(mode="json")
        if isinstance(handoff, QueuedInteractionProfileHandoff)
        else handoff
    )
    target = copied_handoff.target_active_profile
    copied_event = _copy_queued_interaction_started_event(
        session_id,
        target.interaction_id,
        interaction_started_event,
    )
    if copied_event is None:
        raise TypeError("interaction_started_event is required.")
    has_persisted_evidence = QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY in copied_event.payload
    persisted_evidence = copied_event.payload.get(QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY)
    if (
        copied_handoff.expected_active_profile.session_id != session_id
        or target.session_id != session_id
        or (
            has_persisted_evidence
            and persisted_evidence != queued_interaction_profile_handoff_evidence(copied_handoff)
        )
    ):
        raise ValueError("Historical queued interaction handoff evidence conflicts.")
    return copied_event, copied_handoff


def _copy_queued_interaction_started_event(
    session_id: str,
    interaction_id: str | None,
    event: Event | None,
) -> Event | None:
    if event is None:
        return None
    if interaction_id is None:
        raise ValueError("interaction_id is required with interaction_started_event.")
    copied = copy_event(event)
    if copied.session_id != session_id:
        raise ValueError("interaction_started_event belongs to a different session.")
    if copied.interaction_id != interaction_id:
        raise ValueError("interaction_started_event belongs to a different interaction.")
    if copied.type != EventType.INTERACTION_STARTED:
        raise ValueError("interaction_started_event must be interaction.started.")
    return copied


def copy_interaction_transition_spec(
    value: InteractionTransitionSpec,
) -> InteractionTransitionSpec:
    """Return a validated detached copy of one complete transition identity."""

    if type(value) is not InteractionTransitionSpec:
        raise TypeError("transition must be an InteractionTransitionSpec.")
    return InteractionTransitionSpec.model_validate(value.model_dump(mode="python"))


def _prepare_interaction_transition_receipt_lookup(
    session_id: str,
    *,
    transition: InteractionTransitionSpec,
) -> tuple[str, InteractionTransitionSpec]:
    session_id = require_durable_clean_nonblank(session_id, "session_id")
    copied_transition = copy_interaction_transition_spec(transition)
    if copied_transition.event.session_id != session_id:
        raise ValueError("Interaction transition event belongs to a different session.")
    return session_id, copied_transition


def _prepare_interaction_transition(
    session_id: str,
    *,
    event: Event,
    from_statuses: set[SessionStatus],
    to_status: SessionStatus,
    only_if_no_queued_messages: bool,
    model_completion_stage_settlement: ModelCompletionStageSettlementRequest | None = None,
    checkpoint_mutation: dict[str, Any] | None = None,
    terminal_event: Event | None = None,
    terminal_decision: InvocationTerminalDecision | None = None,
) -> tuple[str, InteractionTransitionSpec]:
    session_id = require_durable_clean_nonblank(session_id, "session_id")
    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    if type(from_statuses) is not set:
        raise TypeError("from_statuses must be a set.")
    if not isinstance(to_status, SessionStatus):
        raise ValueError("to_status must be a SessionStatus.")
    if type(only_if_no_queued_messages) is not bool:
        raise TypeError("only_if_no_queued_messages must be a bool.")
    if (
        model_completion_stage_settlement is not None
        and type(model_completion_stage_settlement) is not ModelCompletionStageSettlementRequest
    ):
        raise TypeError(
            "model_completion_stage_settlement must be a "
            "ModelCompletionStageSettlementRequest or None."
        )
    transition = InteractionTransitionSpec(
        event=copy_event(event),
        from_statuses=tuple(
            session_record_rules._validate_status_set(from_statuses, "from_statuses")
        ),
        to_status=to_status,
        only_if_no_queued_messages=only_if_no_queued_messages,
        model_completion_stage_settlement=model_completion_stage_settlement,
        checkpoint_mutation=checkpoint_mutation,
        terminal_event=terminal_event,
        terminal_decision=terminal_decision,
    )
    if transition.event.session_id != session_id:
        raise ValueError("Interaction transition event belongs to a different session.")
    return session_id, transition


def _copy_interaction_admission(
    session_id: str,
    started_event: Event,
    source_messages: list[Message],
    *,
    defer_transcript: bool,
) -> tuple[Event, list[Message]]:
    session_id = require_durable_clean_nonblank(session_id, "session_id")
    if type(defer_transcript) is not bool:
        raise TypeError("defer_transcript must be a bool.")
    if type(started_event) is not Event:
        raise TypeError("started_event must be an Event.")
    copied_event = copy_event(started_event)
    if copied_event.session_id != session_id:
        raise ValueError("started_event belongs to a different session.")
    if copied_event.type != EventType.INTERACTION_STARTED:
        raise ValueError("started_event must be interaction.started.")
    if copied_event.interaction_id is None:
        raise ValueError("started_event must have an interaction_id.")
    copied_messages = copy_transcript_messages(source_messages)
    return copied_event, copied_messages


def _copy_optional_interaction_admission(
    session_id: str,
    started_event: Event | None,
    source_messages: list[Message] | None,
    *,
    defer_transcript: bool,
) -> tuple[Event, list[Message]] | None:
    if (started_event is None) != (source_messages is None):
        raise ValueError(
            "interaction_started_event and interaction_source_messages must be supplied together."
        )
    if started_event is None or source_messages is None:
        return None
    return _copy_interaction_admission(
        session_id,
        started_event,
        source_messages,
        defer_transcript=defer_transcript,
    )


def _copy_transition_interaction_admission(
    session_id: str,
    started_event: Event | None,
    source_messages: list[Message] | None,
    *,
    continued_interaction_id: str | None,
    defer_interaction_source: bool,
) -> tuple[Event | None, str, list[Message], bool] | None:
    if type(defer_interaction_source) is not bool:
        raise TypeError("defer_interaction_source must be a bool.")
    if started_event is not None and continued_interaction_id is not None:
        raise ValueError(
            "interaction_started_event and continued_interaction_id are mutually exclusive."
        )
    if started_event is None and continued_interaction_id is None:
        if source_messages is not None or defer_interaction_source:
            raise ValueError("Interaction source messages require an interaction identity.")
        return None
    if source_messages is None:
        raise ValueError("interaction_source_messages are required for interaction admission.")
    if started_event is not None:
        event, messages = _copy_interaction_admission(
            session_id,
            started_event,
            source_messages,
            defer_transcript=defer_interaction_source,
        )
        interaction_id = event.interaction_id
        if interaction_id is None:
            raise AssertionError("Interaction admission lost its identity.")
        return event, interaction_id, messages, defer_interaction_source
    if continued_interaction_id is None:
        raise AssertionError("Continued interaction admission lost its identity.")
    interaction_id = require_durable_clean_nonblank(
        continued_interaction_id, "continued_interaction_id"
    )
    return (
        None,
        interaction_id,
        copy_transcript_messages(source_messages),
        defer_interaction_source,
    )
