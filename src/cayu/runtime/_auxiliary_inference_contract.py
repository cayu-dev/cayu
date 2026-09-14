"""Durable auxiliary-attempt publication shape, independent of dispatch ownership."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

from pydantic import BaseModel, ConfigDict, field_validator

from cayu._validation import require_durable_clean_nonblank
from cayu.budgets.base import (
    BudgetReservationRecoveryContext,
    model_completion_budget_settlements,
)
from cayu.events import Event, EventType
from cayu.runtime.execution_units import ModelAttemptIdentity, ToolRoundIdentity
from cayu.tools.inference import validate_inference_purpose

if TYPE_CHECKING:
    from cayu.sessions.base import (
        ModelCompletionStage,
        RuntimePublicationRequest,
        _ModelCompletionStagePreparationRecord,
    )


AuxiliaryInferenceOutcome = Literal[
    "completed", "failed", "cancelled", "timed_out", "outcome_unknown"
]
AuxiliaryInferenceUsageStatus = Literal["observed", "missing", "malformed"]

AUXILIARY_ATTRIBUTION_AUTHORITY_PATHS = (
    ("auxiliary_inference", "operation_id"),
    ("auxiliary_inference", "tool_call_id"),
    ("auxiliary_inference", "parent", "model_step_id"),
    ("auxiliary_inference", "parent", "model_attempt_id"),
    ("auxiliary_inference", "parent", "tool_round_id"),
)


def auxiliary_budget_recovery_contexts(
    stage: ModelCompletionStage,
) -> tuple[BudgetReservationRecoveryContext, ...]:
    """Recover exact reservation authority from the auxiliary stage, never defaults."""

    if stage.purpose != "auxiliary-inference":
        raise ValueError("Auxiliary budget recovery requires an auxiliary stage.")
    raw = stage.intent.get("budget_reservations", [])
    if type(raw) is not list:
        raise ValueError("Auxiliary budget recovery contexts must be a list.")
    contexts = tuple(BudgetReservationRecoveryContext.model_validate(item) for item in raw)
    if tuple(item.reservation_id for item in contexts) != stage.reservation_ids:
        raise ValueError("Auxiliary stage lost its exact budget recovery authority.")
    return contexts


class AuxiliaryInferenceAttribution(BaseModel):
    """Parent attribution authored by the runtime, never proof of authority itself."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    operation_id: str
    purpose: str
    parent: ToolRoundIdentity
    tool_call_id: str

    @field_validator("operation_id", "tool_call_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("purpose", mode="before")
    @classmethod
    def validate_purpose(cls, value: object) -> str:
        return validate_inference_purpose(value)

    @field_validator("parent", mode="before")
    @classmethod
    def copy_parent(cls, value: object) -> ToolRoundIdentity:
        if type(value) is ToolRoundIdentity:
            return ToolRoundIdentity(**value.payload())
        if type(value) is dict:
            return ToolRoundIdentity.model_validate(value)
        raise ValueError("Auxiliary inference requires an exact parent tool-round identity.")


def auxiliary_terminal_publication(
    stage: ModelCompletionStage,
    event: Event,
) -> RuntimePublicationRequest:
    """Package prepared accounting without mutating the parent conversation."""
    from cayu.sessions.base import RuntimePublicationMutation, RuntimePublicationRequest

    publication = RuntimePublicationRequest(
        publication_id=stage.logical_step_id,
        kind="auxiliary-inference",
        interaction_id=event.interaction_id,
        intent=stage.intent,
        mutation=RuntimePublicationMutation(),
        transcript_messages=(),
        events=(event,),
    )
    validate_auxiliary_publication(publication, session_id=stage.session_id, stage=stage)
    return publication


def validate_auxiliary_publication(
    publication: RuntimePublicationRequest,
    *,
    session_id: str,
    stage: ModelCompletionStage | _ModelCompletionStagePreparationRecord,
) -> None:
    """Require one exact outcome without touching the parent conversation state."""

    if (
        publication.kind != "auxiliary-inference"
        or publication.transcript_messages
        or publication.mutation.operations
        or publication.operation_record_mutations
        or publication.referenced_events
        or publication.argument_continuity is not None
    ):
        raise ValueError(
            "Auxiliary publication cannot mutate transcript or parent checkpoint state."
        )
    if len(publication.events) != 1:
        raise ValueError("Auxiliary publication requires exactly one attempt-settled event.")
    event = publication.events[0]
    if event.type != EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED or event.session_id != session_id:
        raise ValueError(
            "Auxiliary publication requires its own matching session settlement event."
        )
    expected = AuxiliaryInferenceAttribution.model_validate(stage.intent.get("auxiliary_inference"))
    actual = AuxiliaryInferenceAttribution.model_validate(event.payload.get("auxiliary_inference"))
    if actual != expected:
        raise ValueError("Auxiliary settlement attribution conflicts with its preparation.")
    identity = ModelAttemptIdentity.model_validate(
        {
            "model_step_id": stage.intent.get("model_step_id"),
            "model_attempt_id": stage.intent.get("model_attempt_id"),
        }
    )
    if any(event.payload.get(key) != value for key, value in identity.payload().items()):
        raise ValueError("Auxiliary settlement attempt conflicts with its preparation.")
    for key in ("provider_name", "requested_model", "execution_profile_fingerprint"):
        expected_value = stage.intent.get(key)
        if (
            type(expected_value) is not str
            or not expected_value
            or event.payload.get(key) != expected_value
        ):
            raise ValueError("Auxiliary settlement target/profile conflicts with its preparation.")
    outcome = event.payload.get("auxiliary_outcome")
    ordinal = event.payload.get("attempt")
    if type(ordinal) is not int or ordinal != stage.dispatch_ordinal + 1:
        raise ValueError("Auxiliary settlement ordinal conflicts with its dispatch.")
    if type(outcome) is not str or outcome not in get_args(AuxiliaryInferenceOutcome):
        raise ValueError("Auxiliary settlement requires an explicit known outcome.")
    usage_status = event.payload.get("usage_status")
    if type(usage_status) is not str or usage_status not in get_args(AuxiliaryInferenceUsageStatus):
        raise ValueError("Auxiliary settlement requires an explicit usage observation status.")
    if "transcript_cursor" in event.payload or "input_coverage" in event.payload:
        raise ValueError("Auxiliary settlement cannot claim conversational context coverage.")
    model_completion_budget_settlements(event, reservation_ids=stage.reservation_ids)
