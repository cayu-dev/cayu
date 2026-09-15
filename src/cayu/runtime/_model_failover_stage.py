"""Private ordinary-model route admission at the model-stage transaction.

Public stage intents/checkpoints are data. Only a live admitted invocation with
the exact preflighted candidate profiles can construct a routed preparation.
Backends retain their existing transaction and external-operation ownership.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

from cayu.providers.base import ModelProviderError
from cayu.runtime._execution_profile_admission import bind_model_failover_execution_profile
from cayu.runtime._invocation_lifecycle import AdmittedInvocationBinding, InvocationContext
from cayu.runtime._model_failover import (
    FailoverDisposition,
    FailoverObservation,
    decide_model_failover,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileIdentity,
    active_invocation_execution_profile_from_checkpoint,
    execution_profile_from_session_metadata,
)
from cayu.runtime.retry_policy import RetryDecision
from cayu.sessions._model_failover import (
    MODEL_FAILOVER_CHECKPOINT_KEY,
    ModelFailoverProgress,
    ModelFailoverSelection,
    ModelTarget,
    copy_model_failover_state,
    validate_model_failover_successor,
)
from cayu.sessions.base import (
    ModelCompletionStage,
    Session,
    SessionModelCompletionStageConflict,
    SessionRunFenced,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    decode_runtime_checkpoint,
)


@dataclass(frozen=True, slots=True, repr=False)
class ModelFailoverStageAdmission:
    """In-process proof, intentionally not accepted by public JSON entrances."""

    invocation_context: InvocationContext
    candidate_profiles: tuple[ExecutionProfileIdentity, ...]
    expected: ModelFailoverProgress | ModelFailoverSelection | None
    successor: ModelFailoverProgress
    transition: Literal["initial", "retry", "fallback", "next_step", "reprepare"]
    source_preparation_digest: str | None
    failure: BaseException | None = field(default=None, repr=False)
    retry: RetryDecision | None = None
    observation: FailoverObservation | None = None

    def __post_init__(self) -> None:
        if type(self.invocation_context) is not InvocationContext:
            raise TypeError("Model failover requires an authenticated InvocationContext.")
        self.invocation_context._validate()
        binding = self.invocation_context.binding
        if type(binding) is not AdmittedInvocationBinding:
            raise TypeError("Model failover requires an admitted invocation.")
        successor = ModelFailoverProgress.model_validate(self.successor)
        expected = None if self.expected is None else copy_model_failover_state(self.expected)
        object.__setattr__(self, "successor", successor)
        object.__setattr__(self, "expected", expected)
        bound_profile = bind_model_failover_execution_profile(
            plan=successor.plan, candidate_profiles=self.candidate_profiles
        )
        if (
            bound_profile != self.invocation_context.profile
            or successor.execution_profile_fingerprint != bound_profile.fingerprint
            or successor.session_id != binding.session_id
            or successor.session_instance_id != binding.session_instance_id
            or successor.interaction_id != binding.interaction_id
            or successor.source_run_epoch != binding.run_epoch
        ):
            raise ValueError("Model failover conflicts with its admitted invocation.")
        if self.transition == "initial":
            if (
                isinstance(expected, ModelFailoverProgress)
                or self.source_preparation_digest is not None
                or successor.generation != 1
                or successor.candidate_index
                != (0 if expected is None else expected.candidate_index)
                or successor.attempts_used != 1
                or successor.candidate_attempt != 1
                or successor.provider_effect_observed
            ):
                raise ValueError("Initial model failover progress is inconsistent.")
            if expected is not None and (
                expected.session_id != successor.session_id
                or expected.session_instance_id != successor.session_instance_id
                or expected.execution_profile_fingerprint != successor.execution_profile_fingerprint
                or expected.plan != successor.plan
                or expected.source_run_epoch > successor.source_run_epoch
                or expected.source_transcript_cursor > successor.source_transcript_cursor
                or expected.projection_cursor != successor.projection_cursor
            ):
                raise ValueError("Initial model preparation changed its durable selection.")
        else:
            if not isinstance(expected, ModelFailoverProgress):
                raise ValueError("Model failover requires an exact predecessor.")
            digest = self.source_preparation_digest
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("Model failover requires the exact predecessor preparation.")
            validate_model_failover_successor(expected, successor, transition=self.transition)
        if self.transition == "fallback":
            assert isinstance(expected, ModelFailoverProgress)
            if self.failure is None or self.retry is None or self.observation is None:
                raise ValueError("Fallback requires positive live failure and ownership evidence.")
            decision = decide_model_failover(
                failure=self.failure,
                provider_name=expected.plan.candidates[expected.candidate_index].provider_name,
                retry=self.retry,
                observation=self.observation,
                candidate_index=expected.candidate_index,
                candidate_count=len(expected.plan.candidates),
                attempts_used=expected.attempts_used,
                max_total_attempts=expected.plan.max_total_attempts,
            )
            if decision.disposition is not FailoverDisposition.SELECT_NEXT:
                raise ValueError("The observed failure does not authorize provider fallback.")
            assert isinstance(self.failure, ModelProviderError)
            # The original exception remains owned by the model failure path.
            # Durable preparation needs only a detached eligibility snapshot;
            # a provider may still retain and mutate its original exception.
            object.__setattr__(
                self,
                "failure",
                ModelProviderError(
                    "Eligible provider service rejection",
                    provider=self.failure.provider,
                    status_code=self.failure.status_code,
                    retryable=True,
                ),
            )
            object.__setattr__(
                self,
                "retry",
                RetryDecision.model_validate(
                    {name: getattr(self.retry, name) for name in RetryDecision.model_fields}
                ),
            )
            object.__setattr__(self, "observation", replace(self.observation))
        elif self.failure is not None or self.retry is not None or self.observation is not None:
            raise ValueError("Only a fallback transition carries cross-provider failure evidence.")

    def __repr__(self) -> str:
        return "ModelFailoverStageAdmission(<authenticated>)"

    def __reduce_ex__(self, _protocol):
        raise TypeError("Model failover admission has no serialization form.")

    def intent_payload(self) -> dict[str, Any]:
        payload = {
            "transition": self.transition,
            "expected": None if self.expected is None else self.expected.payload(),
            "successor": self.successor.payload(),
            "source_preparation_digest": self.source_preparation_digest,
        }
        if self.transition == "fallback":
            assert isinstance(self.failure, ModelProviderError)
            assert self.retry is not None and self.observation is not None
            payload["eligibility"] = {
                "failure_kind": "typed_provider_service_failure",
                "provider_name": self.observation.provider_name,
                "status_code": self.failure.status_code,
                "retryable": self.failure.retryable,
                "retry_disposition": str(self.retry.disposition),
                "retry_status_code": self.retry.status_code,
                "provider_retryable": self.retry.provider_retryable,
                "retry": self.retry.retry,
                "retry_suppression": self.retry.suppression,
                "caller_cancelled": self.observation.caller_cancelled,
                "completion_observed": self.observation.completion_observed,
                "provider_effect_observed": self.observation.provider_effect_observed,
                "provider_operation_owned": self.observation.provider_operation_owned,
                "cleanup_settled": self.observation.cleanup_settled,
            }
        return payload


def model_failover_checkpoint_after_preparation(
    admission: ModelFailoverStageAdmission,
    *,
    session: Session,
    checkpoint: dict[str, Any] | None,
    transcript_cursor: int,
    replayed: bool,
) -> dict[str, Any]:
    """Pure projection, called under the existing backend transaction/lock.

    The stage owner additionally validates the exact active predecessor, its
    settlement, and its winner/receipt state. No caller may commit this returned
    projection through a generic checkpoint callback.
    """

    if type(admission) is not ModelFailoverStageAdmission:
        raise TypeError("Model failover requires authenticated stage admission.")
    admission = replace(admission)
    successor = admission.successor
    current = decode_runtime_checkpoint(checkpoint, session_id=session.id)
    if (
        session.id != successor.session_id
        or session.instance_id != successor.session_instance_id
        or session.run_epoch != successor.source_run_epoch
        or transcript_cursor != successor.source_transcript_cursor
        or active_invocation_execution_profile_from_checkpoint(current)
        != admission.invocation_context.active_profile
    ):
        raise SessionRunFenced("Model failover lost its exact invocation or transcript fence.")
    observed = (
        None
        if current is None or MODEL_FAILOVER_CHECKPOINT_KEY not in current
        else copy_model_failover_state(current[MODEL_FAILOVER_CHECKPOINT_KEY])
    )
    expected = successor if replayed else admission.expected
    if observed != expected:
        raise SessionModelCompletionStageConflict("Model failover predecessor changed.")
    updated = {} if current is None else current
    updated[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION
    updated[MODEL_FAILOVER_CHECKPOINT_KEY] = successor.payload()
    return updated


def model_failover_target_for_stored_stage(
    *, session: Session, stage: ModelCompletionStage
) -> ModelTarget | None:
    """Resolve a store-read stage's target, never authenticate a caller record.

    The caller owns the exact store lookup and any active-stage/replay fence.
    Immutable stage/profile evidence remains usable after the recovery owner
    advances the run epoch; unlike dispatch, settlement cannot require the old
    epoch to still be current.
    """

    profile = execution_profile_from_session_metadata(session.metadata)
    binding = profile.model_failover
    if MODEL_FAILOVER_CHECKPOINT_KEY not in stage.intent:
        if stage.purpose == "assistant-turn" and binding is not None:
            raise SessionModelCompletionStageConflict("Stored model stage lost its failover plan.")
        return None
    capsule = stage.intent[MODEL_FAILOVER_CHECKPOINT_KEY]
    if type(capsule) is not dict or "successor" not in capsule:
        raise SessionModelCompletionStageConflict("Model failover stage binding is malformed.")
    progress = ModelFailoverProgress.model_validate(capsule["successor"])
    candidate = progress.plan.candidates[progress.candidate_index]
    if (
        binding is None
        or binding.plan != progress.plan
        or profile.fingerprint != progress.execution_profile_fingerprint
        or progress.session_id != session.id
        or progress.session_instance_id != session.instance_id
        or progress.plan.candidates[0].provider_name != session.provider_name
        or progress.plan.candidates[0].model != session.model
        or stage.session_id != session.id
        or stage.purpose != "assistant-turn"
        or progress.source_run_epoch != stage.source_run_epoch
        or progress.source_run_epoch > session.run_epoch
        or progress.stage_id != stage.stage_id
        or progress.logical_step_id != stage.logical_step_id
        or progress.dispatch_ordinal != stage.dispatch_ordinal
        or progress.source_transcript_cursor != stage.source_transcript_cursor
        or stage.intent.get("interaction_id") != progress.interaction_id
        or stage.intent.get("request_fingerprint") != progress.request_fingerprint
        or stage.intent.get("provider_name") != candidate.provider_name
        or stage.intent.get("requested_model") != candidate.model
    ):
        raise SessionModelCompletionStageConflict("Stored model stage lost its admitted target.")
    return ModelTarget(provider_name=candidate.provider_name, model=candidate.model)


def model_failover_retry_available_for_stored_stage(
    *, session: Session, stage: ModelCompletionStage
) -> bool:
    """Check the remaining step budget for an exact store-owned recovery stage.

    This does not grant retry authority. The caller's disposition transaction
    still owns the active-stage fence, and preparation independently consumes
    the next attempt under its existing compare-and-swap.
    """

    if MODEL_FAILOVER_CHECKPOINT_KEY not in stage.intent:
        return True
    if model_failover_target_for_stored_stage(session=session, stage=stage) is None:
        raise SessionModelCompletionStageConflict("Model retry lost its failover route.")
    progress = ModelFailoverProgress.model_validate(
        stage.intent[MODEL_FAILOVER_CHECKPOINT_KEY]["successor"]
    )
    return progress.attempts_used < progress.plan.max_total_attempts


def validate_model_failover_dispatch(
    *,
    session: Session,
    checkpoint: dict[str, Any] | None,
    stage: ModelCompletionStage,
) -> None:
    """Bind a stored ordinary stage to the exact current route before dispatch."""

    current = (
        decode_runtime_checkpoint(checkpoint, session_id=session.id)
        if checkpoint is not None and MODEL_FAILOVER_CHECKPOINT_KEY in checkpoint
        else checkpoint
    )
    has_route = current is not None and MODEL_FAILOVER_CHECKPOINT_KEY in current
    if MODEL_FAILOVER_CHECKPOINT_KEY not in stage.intent:
        if has_route and stage.purpose == "assistant-turn":
            raise SessionModelCompletionStageConflict(
                "An ordinary model dispatch cannot bypass its durable failover route."
            )
        return
    if stage.purpose != "assistant-turn" or not has_route:
        raise SessionModelCompletionStageConflict("Model dispatch lost its failover route.")
    model_failover_target_for_stored_stage(session=session, stage=stage)
    capsule = stage.intent[MODEL_FAILOVER_CHECKPOINT_KEY]
    if type(capsule) is not dict or "successor" not in capsule:
        raise SessionModelCompletionStageConflict("Model failover stage binding is malformed.")
    progress = ModelFailoverProgress.model_validate(capsule["successor"])
    if current is None or MODEL_FAILOVER_CHECKPOINT_KEY not in current:
        raise SessionModelCompletionStageConflict(
            "Model failover stage lacks current-schema authority."
        )
    observed = ModelFailoverProgress.model_validate(current[MODEL_FAILOVER_CHECKPOINT_KEY])
    active_profile = active_invocation_execution_profile_from_checkpoint(current)
    candidate = progress.plan.candidates[progress.candidate_index]
    if (
        observed != progress
        or progress.session_id != session.id
        or progress.session_instance_id != session.instance_id
        or progress.source_run_epoch != session.run_epoch
        or progress.stage_id != stage.stage_id
        or progress.logical_step_id != stage.logical_step_id
        or progress.dispatch_ordinal != stage.dispatch_ordinal
        or progress.source_transcript_cursor != stage.source_transcript_cursor
        or stage.intent.get("request_fingerprint") != progress.request_fingerprint
        or stage.intent.get("provider_name") != candidate.provider_name
        or stage.intent.get("requested_model") != candidate.model
        or active_profile is None
        or active_profile.session_id != session.id
        or active_profile.run_epoch != session.run_epoch
        or active_profile.interaction_id != progress.interaction_id
        or active_profile.profile.fingerprint != progress.execution_profile_fingerprint
    ):
        raise SessionModelCompletionStageConflict("Model dispatch lost its exact selected route.")
