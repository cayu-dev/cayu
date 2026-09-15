"""Effective model configuration inside an immutable admitted invocation.

This value authorizes request preparation for a configured candidate, not a
provider dispatch. The existing model-stage preparation/dispatch transaction is
still required to consume an exact attempt and advance durable route progress.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from cayu.providers.base import ModelProviderError
from cayu.runtime._execution_profile_admission import ModelFailoverProfileResolution
from cayu.runtime._invocation_lifecycle import AdmittedInvocationBinding, InvocationContext
from cayu.runtime._model_failover import (
    FailoverDisposition,
    FailoverObservation,
    decide_model_failover,
)
from cayu.runtime._model_failover_stage import (
    ModelFailoverStageAdmission,
    model_failover_target_for_stored_stage,
)
from cayu.runtime._runtime_records import RegisteredAgentState, RegisteredProvider
from cayu.runtime.execution_profiles import ExecutionProfileIdentity
from cayu.runtime.execution_units import ModelAttemptIdentity, copy_model_attempt_identity
from cayu.runtime.retry_policy import RetryDecision
from cayu.sessions._model_failover import (
    MODEL_FAILOVER_CHECKPOINT_KEY,
    ModelFailoverProgress,
    ModelFailoverSelection,
    copy_model_failover_state,
)
from cayu.sessions.base import ModelCompletionStage, Session
from cayu.sessions.checkpoints import decode_runtime_checkpoint


def model_failover_progress_for_session(
    *,
    session: Session,
    execution_profile: ExecutionProfileIdentity,
    checkpoint: dict[str, Any] | None,
) -> ModelFailoverProgress | ModelFailoverSelection | None:
    """Resolve stored selection, not permission to resume or repeat a dispatch.

    The caller owns snapshot provenance and invocation admission. The stage
    transaction must still prove the exact predecessor and its settlement.
    """

    if checkpoint is None or MODEL_FAILOVER_CHECKPOINT_KEY not in checkpoint:
        return None
    current = decode_runtime_checkpoint(checkpoint, session_id=session.id)
    if current is None or MODEL_FAILOVER_CHECKPOINT_KEY not in current:
        return None
    progress = copy_model_failover_state(current[MODEL_FAILOVER_CHECKPOINT_KEY])
    binding = execution_profile.model_failover
    if (
        binding is None
        or progress.session_id != session.id
        or progress.session_instance_id != session.instance_id
        or progress.execution_profile_fingerprint != execution_profile.fingerprint
        or progress.plan != binding.plan
        or progress.source_run_epoch > session.run_epoch
        or progress.plan.candidates[0].provider_name != session.provider_name
        or progress.plan.candidates[0].model != session.model
    ):
        raise ValueError("Stored model selection conflicts with its session/profile authority.")
    return progress


@dataclass(frozen=True, slots=True, repr=False)
class ModelExecutionSelection:
    invocation_context: InvocationContext
    resolution: ModelFailoverProfileResolution
    candidate_index: int

    def __post_init__(self) -> None:
        if type(self.invocation_context) is not InvocationContext:
            raise TypeError("Model selection requires admitted invocation authority.")
        self.invocation_context._validate()
        if type(self.invocation_context.binding) is not AdmittedInvocationBinding:
            raise TypeError("Model selection requires an admitted invocation.")
        if type(self.resolution) is not ModelFailoverProfileResolution:
            raise TypeError("Model selection requires resolved candidate configuration.")
        resolution = replace(self.resolution)
        if type(self.candidate_index) is not int or not (
            0 <= self.candidate_index < len(resolution.plan.candidates)
        ):
            raise ValueError("Model selection index is outside the admitted plan.")
        if (
            resolution.profile != self.invocation_context.profile
            or resolution.registered_providers[0] is not self.invocation_context.registered_provider
        ):
            raise ValueError("Model selection conflicts with frozen invocation authority.")
        object.__setattr__(self, "resolution", resolution)

    @property
    def registered_provider(self) -> RegisteredProvider:
        return self.resolution.registered_providers[self.candidate_index]

    @property
    def model(self) -> str:
        return self.resolution.plan.candidates[self.candidate_index].model

    def require_request_scope(
        self,
        *,
        invocation_context: InvocationContext | None,
        session: Session,
        registered_agent: RegisteredAgentState,
        registered_provider: RegisteredProvider,
        execution_profile: ExecutionProfileIdentity | None,
        model: str,
    ) -> None:
        selection = replace(self)
        context = selection.invocation_context
        binding = context.binding
        if (
            invocation_context is not context
            or registered_agent is not context.registered_agent
            or registered_provider is not selection.registered_provider
            or execution_profile is not context.profile
            or type(session) is not Session
            or type(model) is not str
            or session.id != binding.session_id
            or session.instance_id != binding.session_instance_id
            or session.run_epoch != binding.run_epoch
            or session.provider_name != binding.provider_name
            or session.model != binding.model
            or model != selection.model
        ):
            raise ValueError("Effective model request substituted admitted invocation authority.")

    def __repr__(self) -> str:
        return "ModelExecutionSelection(<admitted candidate>)"

    def require_prepared_stage(self, stage: ModelCompletionStage) -> None:
        """Check the store callback's result before entering provider-controlled code.

        This verifies handoff consistency, not store provenance. The stage
        transaction and dispatch marker remain the sole dispatch authority.
        """

        selection = replace(self)
        if type(stage) is not ModelCompletionStage:
            raise TypeError("Selected execution requires an exact prepared model stage.")
        capsule = stage.intent.get(MODEL_FAILOVER_CHECKPOINT_KEY)
        if type(capsule) is not dict:
            raise ValueError("Selected execution requires a routed prepared model stage.")
        progress = ModelFailoverProgress.model_validate(capsule.get("successor"))
        binding = selection.invocation_context.binding
        if (
            progress.plan != selection.resolution.plan
            or progress.candidate_index != selection.candidate_index
            or progress.session_id != binding.session_id
            or progress.session_instance_id != binding.session_instance_id
            or progress.interaction_id != binding.interaction_id
            or progress.source_run_epoch != binding.run_epoch
            or progress.execution_profile_fingerprint
            != selection.invocation_context.profile.fingerprint
            or stage.session_id != binding.session_id
            or stage.source_run_epoch != binding.run_epoch
            or stage.source_transcript_cursor != progress.source_transcript_cursor
            or stage.stage_id != progress.stage_id
            or stage.logical_step_id != progress.logical_step_id
            or stage.dispatch_ordinal != progress.dispatch_ordinal
            or stage.purpose != "assistant-turn"
            or stage.intent.get("request_fingerprint") != progress.request_fingerprint
            or stage.intent.get("provider_name") != selection.registered_provider.name
            or stage.intent.get("requested_model") != selection.model
        ):
            raise ValueError("Prepared stage conflicts with the selected model request.")

    def __reduce_ex__(self, _protocol):
        raise TypeError("Model execution selection has no serialization form.")

    def require_recovery_scope(
        self,
        *,
        session: Session,
        stage: ModelCompletionStage,
        invocation_context: InvocationContext | None,
        registered_provider: RegisteredProvider,
    ) -> None:
        """Bind reconstructed collaborators to a store-owned stage, not a new dispatch."""

        selection = replace(self)
        target = model_failover_target_for_stored_stage(session=session, stage=stage)
        binding = selection.invocation_context.binding
        if (
            invocation_context is not selection.invocation_context
            or registered_provider is not selection.registered_provider
            or target is None
            or target.provider_name != registered_provider.name
            or target.model != selection.model
            or binding.session_id != session.id
            or binding.session_instance_id != session.instance_id
            or binding.run_epoch != session.run_epoch
            or binding.interaction_id != stage.intent.get("interaction_id")
            or selection.invocation_context.profile.fingerprint
            != stage.intent[MODEL_FAILOVER_CHECKPOINT_KEY]["successor"][
                "execution_profile_fingerprint"
            ]
        ):
            raise ValueError(
                "Provider-operation recovery substituted admitted selection authority."
            )


@dataclass(frozen=True, slots=True, repr=False)
class ModelFailoverTransition:
    """Live settled-failure handoff, bound to the exact predecessor preparation.

    This is not a dispatch permit. Preparation must compare this predecessor
    under the existing store transaction before admitting the next candidate.
    """

    source: ModelFailoverProgress
    source_preparation_digest: str
    failure: BaseException
    retry: RetryDecision
    observation: FailoverObservation

    def __post_init__(self) -> None:
        source = ModelFailoverProgress.model_validate(self.source)
        object.__setattr__(self, "source", source)
        digest = self.source_preparation_digest
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Fallback requires the exact source preparation digest.")
        decision = decide_model_failover(
            failure=self.failure,
            provider_name=source.plan.candidates[source.candidate_index].provider_name,
            retry=self.retry,
            observation=self.observation,
            candidate_index=source.candidate_index,
            candidate_count=len(source.plan.candidates),
            attempts_used=source.attempts_used,
            max_total_attempts=source.plan.max_total_attempts,
        )
        if decision.disposition is not FailoverDisposition.SELECT_NEXT:
            raise ValueError("Live failure does not authorize a candidate transition.")
        assert isinstance(self.failure, ModelProviderError)
        # Keep the original exception at the execution owner. Provider-retained
        # mutable exceptions must not rewrite an already accepted eligibility
        # decision while another target constructs its request.
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

    def __repr__(self) -> str:
        return "ModelFailoverTransition(<settled candidate>)"

    def __reduce_ex__(self, _protocol):
        raise TypeError("Model failover transition has no serialization form.")


@dataclass(frozen=True, slots=True, repr=False)
class ModelFailoverAttempt:
    """One live local retry handed to the existing stage-preparation owner."""

    selection: ModelExecutionSelection
    identity: ModelAttemptIdentity
    prior_provider_effect_observed: bool

    def __post_init__(self) -> None:
        if type(self.selection) is not ModelExecutionSelection:
            raise TypeError("Failover attempt requires an admitted selection.")
        replace(self.selection)
        object.__setattr__(self, "identity", copy_model_attempt_identity(self.identity))
        if type(self.prior_provider_effect_observed) is not bool:
            raise TypeError("Failover attempt requires explicit prior-effect evidence.")

    def stage_admission(
        self,
        *,
        previous: ModelFailoverProgress | ModelFailoverSelection | None,
        source_preparation_digest: str | None,
        source_transcript_cursor: int,
        request_fingerprint: str,
        fallback: ModelFailoverTransition | None = None,
    ) -> ModelFailoverStageAdmission:
        attempt = replace(self)
        selection = attempt.selection
        binding = selection.invocation_context.binding
        if previous is not None:
            previous = copy_model_failover_state(previous)
        prepared_previous = previous if isinstance(previous, ModelFailoverProgress) else None
        same_step = (
            prepared_previous is not None
            and prepared_previous.logical_step_id == attempt.identity.model_step_id
        )
        if fallback is not None:
            if type(fallback) is not ModelFailoverTransition:
                raise TypeError("Fallback preparation requires a live failure handoff.")
            fallback = replace(fallback)
            if (
                previous != fallback.source
                or source_preparation_digest != fallback.source_preparation_digest
                or not same_step
                or selection.candidate_index != fallback.source.candidate_index + 1
            ):
                raise ValueError("Fallback preparation changed its exact failed predecessor.")
            transition = "fallback"
        elif previous is not None and previous.candidate_index != selection.candidate_index:
            raise ValueError("Cross-candidate preparation requires an authorized failure handoff.")
        elif prepared_previous is None:
            transition = "initial"
        elif same_step:
            transition = "retry"
        else:
            transition = "next_step"
        ordinal = (
            prepared_previous.dispatch_ordinal + 1
            if same_step and prepared_previous is not None
            else 0
        )
        successor = ModelFailoverProgress(
            session_id=binding.session_id,
            session_instance_id=binding.session_instance_id,
            interaction_id=binding.interaction_id,
            execution_profile_fingerprint=selection.invocation_context.profile.fingerprint,
            plan=selection.resolution.plan,
            generation=1 if prepared_previous is None else prepared_previous.generation + 1,
            candidate_index=selection.candidate_index,
            logical_step_id=attempt.identity.model_step_id,
            stage_id=f"{attempt.identity.model_step_id}:dispatch:{ordinal}",
            request_fingerprint=request_fingerprint,
            source_run_epoch=binding.run_epoch,
            source_transcript_cursor=source_transcript_cursor,
            projection_cursor=(
                source_transcript_cursor
                if fallback is not None
                else 0
                if previous is None
                else previous.projection_cursor
            ),
            dispatch_ordinal=ordinal,
            attempts_used=(
                prepared_previous.attempts_used + 1
                if same_step and prepared_previous is not None
                else 1
            ),
            candidate_attempt=(
                prepared_previous.candidate_attempt + 1
                if same_step and prepared_previous is not None and fallback is None
                else 1
            ),
            provider_effect_observed=(
                attempt.prior_provider_effect_observed
                or (
                    same_step
                    and prepared_previous is not None
                    and prepared_previous.provider_effect_observed
                )
            ),
        )
        return ModelFailoverStageAdmission(
            invocation_context=selection.invocation_context,
            candidate_profiles=selection.resolution.candidate_profiles,
            expected=previous,
            successor=successor,
            transition=transition,
            source_preparation_digest=source_preparation_digest,
            failure=None if fallback is None else fallback.failure,
            retry=None if fallback is None else fallback.retry,
            observation=None if fallback is None else fallback.observation,
        )

    def __repr__(self) -> str:
        return "ModelFailoverAttempt(<admitted attempt>)"
