"""Shared execution-profile classification for resume and fork admission."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime

from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
from cayu.egress.authority import EgressAuthorityChangeKind
from cayu.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileComponentClass,
    ExecutionProfileDecision,
    ExecutionProfileDecisionKind,
    ExecutionProfileIdentity,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
    copy_execution_profile_policy_result,
    execution_profile_changes_authority,
    execution_profile_egress_authority_change,
)
from cayu.runtime._execution_profile_continuation import _execution_profile_decision_event
from cayu.runtime.execution_profiles import ExecutionProfilePolicy, ExecutionProfilePolicyError
from cayu.sessions.records import Session
from cayu.vaults.redaction import SecretRedactor

_EXACT_PROFILE_POLICY_ID = "cayu:execution-profile-exact-reuse:v1"
_DEFAULT_PROFILE_POLICY_ID = "cayu:execution-profile-default-reject:v1"
_MODEL_TARGET_PROFILE_POLICY_ID = "cayu:model-target-adoption:v1"
_TOOL_CAPABILITY_CEILING_PROFILE_POLICY_ID = "cayu:tool-capability-ceiling-narrowing:v1"
_MODEL_TARGET_AND_TOOL_CAPABILITY_CEILING_PROFILE_POLICY_ID = (
    "cayu:model-target-and-tool-capability-ceiling-adoption:v1"
)
_MODEL_TARGET_BUILT_IN_COMPONENTS = frozenset(
    {
        ExecutionProfileComponentClass.PROVIDER_ADAPTER,
        ExecutionProfileComponentClass.PROVIDER_TARGET,
    }
)


def _model_target_profile_actor() -> ResolutionActor:
    return ResolutionActor(
        subject="cayu:model-target-adoption",
        source=ResolutionActorSource.SYSTEM,
    )


def _tool_capability_ceiling_profile_actor() -> ResolutionActor:
    return ResolutionActor(
        subject="cayu:tool-capability-ceiling-narrowing",
        source=ResolutionActorSource.SYSTEM,
    )


def _model_target_and_tool_capability_ceiling_profile_actor() -> ResolutionActor:
    return ResolutionActor(
        subject="cayu:model-target-and-tool-capability-ceiling-adoption",
        source=ResolutionActorSource.SYSTEM,
    )


def _execution_profile_policy_actor(policy_identity: str) -> ResolutionActor:
    return ResolutionActor(
        subject=f"cayu:execution-profile-policy:{policy_identity}",
        source=ResolutionActorSource.SYSTEM,
    )


class ExecutionProfileClassifier:
    """Own exact reuse, adoption and authority decisions before admission."""

    def __init__(
        self,
        *,
        policy: ExecutionProfilePolicy | None,
        policy_identity: str | None,
        redactor: SecretRedactor,
        clock: Callable[[], datetime],
    ) -> None:
        self._execution_profile_policy = policy
        self._execution_profile_policy_identity = policy_identity
        self._secret_redactor = redactor
        self._clock = clock

    @property
    def policy_identity(self) -> str | None:
        return self._execution_profile_policy_identity

    async def classify(
        self,
        *,
        session: Session,
        expected_profile: ExecutionProfileIdentity,
        candidate_profile: ExecutionProfileIdentity,
        changed_component_classes: tuple[ExecutionProfileComponentClass, ...],
        intent: ExecutionProfileAdoptionIntent | None,
        adoption_request_fingerprint: str | None,
        target_changed: bool,
        target_provider_name: str,
        target_model: str,
        built_in_model_target_transition: bool = False,
        decision_session: Session | None = None,
        source_provider_name: str | None = None,
        source_model: str | None = None,
        force_authority_review: bool = False,
        tool_capability_ceiling_narrowed: bool = False,
        decision_clock: Callable[[], datetime] | None = None,
    ) -> ExecutionProfileDecision:
        event_session = session if decision_session is None else decision_session
        egress_authority_change = execution_profile_egress_authority_change(
            expected_profile,
            candidate_profile,
            changed_component_classes=changed_component_classes,
        )
        if not changed_component_classes and not force_authority_review:
            return _execution_profile_decision_event(
                session=event_session,
                expected_profile=expected_profile,
                candidate_profile=candidate_profile,
                changed_component_classes=changed_component_classes,
                kind=ExecutionProfileDecisionKind.EXACT_REUSE,
                policy_identity=_EXACT_PROFILE_POLICY_ID,
                policy_reason="The execution profile is exactly equal.",
                authority_decision=ExecutionProfileAuthorityDecision.NOT_REQUIRED,
                intent=intent,
                adoption_request_fingerprint=adoption_request_fingerprint,
                fallback_actor=None,
                fallback_reason="The execution profile is exactly equal.",
                clock=self._clock if decision_clock is None else decision_clock,
            )

        changed_component_set = frozenset(changed_component_classes)
        model_target_only = (
            target_changed
            and built_in_model_target_transition
            and ExecutionProfileComponentClass.PROVIDER_TARGET in changed_component_set
            and changed_component_set <= _MODEL_TARGET_BUILT_IN_COMPONENTS
        )
        built_in_model_target_adoption = model_target_only and not force_authority_review
        built_in_ceiling_narrowing = (
            tool_capability_ceiling_narrowed
            and not force_authority_review
            and changed_component_set
            == frozenset({ExecutionProfileComponentClass.TOOL_VIEW_GRANTS})
        )
        built_in_model_target_and_ceiling_adoption = (
            target_changed
            and built_in_model_target_transition
            and tool_capability_ceiling_narrowed
            and not force_authority_review
            and {
                ExecutionProfileComponentClass.PROVIDER_TARGET,
                ExecutionProfileComponentClass.TOOL_VIEW_GRANTS,
            }
            <= changed_component_set
            and changed_component_set
            <= _MODEL_TARGET_BUILT_IN_COMPONENTS | {ExecutionProfileComponentClass.TOOL_VIEW_GRANTS}
        )
        if built_in_model_target_and_ceiling_adoption:
            fallback_actor = _model_target_and_tool_capability_ceiling_profile_actor()
            fallback_reason = (
                "The caller explicitly requested a model-target transition and narrowed the "
                "durable tool capability ceiling."
            )
            policy_identity = _MODEL_TARGET_AND_TOOL_CAPABILITY_CEILING_PROFILE_POLICY_ID
        elif built_in_model_target_adoption:
            fallback_actor = _model_target_profile_actor()
            fallback_reason = "The caller explicitly requested a model-target transition."
            policy_identity = _MODEL_TARGET_PROFILE_POLICY_ID
        elif built_in_ceiling_narrowing:
            fallback_actor = _tool_capability_ceiling_profile_actor()
            fallback_reason = "The caller explicitly narrowed the durable tool capability ceiling."
            policy_identity = _TOOL_CAPABILITY_CEILING_PROFILE_POLICY_ID
        else:
            fallback_actor = None
            fallback_reason = "The execution profile changed without authorized adoption."
            policy_identity = _DEFAULT_PROFILE_POLICY_ID
        policy_reason = fallback_reason
        built_in_profile_adoption = (
            built_in_model_target_adoption
            or built_in_ceiling_narrowing
            or built_in_model_target_and_ceiling_adoption
        )
        authority_decision = (
            ExecutionProfileAuthorityDecision.AUTHORIZED
            if built_in_profile_adoption
            else ExecutionProfileAuthorityDecision.NOT_REQUIRED
        )
        action = (
            ExecutionProfilePolicyAction.ADOPT
            if built_in_profile_adoption
            else ExecutionProfilePolicyAction.REJECT
        )

        policy = self._execution_profile_policy
        if policy is not None:
            stable_identity = self._execution_profile_policy_identity
            try:
                current_policy_identity = policy.identity
            except Exception as exc:
                raise ExecutionProfilePolicyError(
                    "The configured execution-profile policy identity could not be read."
                ) from exc
            if stable_identity is None or current_policy_identity != stable_identity:
                raise ExecutionProfilePolicyError(
                    "The configured execution-profile policy identity changed."
                )
            policy_request = ExecutionProfilePolicyRequest(
                session_id=event_session.id,
                expected_profile=expected_profile,
                candidate_profile=candidate_profile,
                changed_component_classes=changed_component_classes,
                intent=intent,
                authority_review_required=(
                    force_authority_review
                    or execution_profile_changes_authority(changed_component_classes)
                ),
                egress_authority_change=egress_authority_change,
                source_provider_name=(
                    session.provider_name if source_provider_name is None else source_provider_name
                ),
                source_model=session.model if source_model is None else source_model,
                target_provider_name=target_provider_name,
                target_model=target_model,
            )
            try:
                raw_result = await policy.decide(policy_request)
                result = copy_execution_profile_policy_result(raw_result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise ExecutionProfilePolicyError(
                    "The execution-profile policy failed before admission."
                ) from exc
            try:
                current_policy_identity = policy.identity
            except Exception as exc:
                raise ExecutionProfilePolicyError(
                    "The configured execution-profile policy identity could not be re-read."
                ) from exc
            if current_policy_identity != stable_identity:
                raise ExecutionProfilePolicyError(
                    "The configured execution-profile policy identity changed during review."
                )
            try:
                result = ExecutionProfilePolicyResult(
                    action=result.action,
                    reason=self._secret_redactor.redact_text(result.reason),
                    authority_decision=result.authority_decision,
                )
            except (TypeError, ValueError) as exc:
                raise ExecutionProfilePolicyError(
                    "The execution-profile policy result could not be made durable."
                ) from exc
            action = result.action
            policy_identity = stable_identity
            policy_reason = result.reason
            authority_decision = result.authority_decision

        non_egress_authority_change = execution_profile_changes_authority(
            component
            for component in changed_component_classes
            if component is not ExecutionProfileComponentClass.EGRESS_AUTHORITY
        )
        egress_broadens_or_is_ambiguous = egress_authority_change in {
            EgressAuthorityChangeKind.WIDER,
            EgressAuthorityChangeKind.INCOMPARABLE,
            EgressAuthorityChangeKind.REFUSED,
        }
        broadens_authority = (
            force_authority_review or non_egress_authority_change or egress_broadens_or_is_ambiguous
        )
        automatic_egress_narrowing = (
            policy is not None
            and intent is None
            and action is ExecutionProfilePolicyAction.ADOPT
            and authority_decision is not ExecutionProfileAuthorityDecision.DENIED
            and changed_component_set
            == frozenset({ExecutionProfileComponentClass.EGRESS_AUTHORITY})
            and egress_authority_change is EgressAuthorityChangeKind.NARROWER
        )
        if automatic_egress_narrowing:
            fallback_actor = _execution_profile_policy_actor(policy_identity)
            fallback_reason = policy_reason
        if action is ExecutionProfilePolicyAction.COMPATIBLE_REUSE:
            changes_persistent_target = (
                ExecutionProfileComponentClass.PROVIDER_TARGET in changed_component_classes
            )
            changes_egress_generation = (
                ExecutionProfileComponentClass.EGRESS_AUTHORITY in changed_component_classes
            )
            kind = (
                ExecutionProfileDecisionKind.REJECTED
                if (
                    authority_decision is ExecutionProfileAuthorityDecision.DENIED
                    or broadens_authority
                    or changes_persistent_target
                    or changes_egress_generation
                )
                else ExecutionProfileDecisionKind.COMPATIBLE_REUSE
            )
            if (
                authority_decision is not ExecutionProfileAuthorityDecision.DENIED
                and broadens_authority
            ):
                policy_reason = (
                    "A compatibility decision cannot authorize broader execution authority."
                )
            elif (
                authority_decision is not ExecutionProfileAuthorityDecision.DENIED
                and changes_persistent_target
            ):
                policy_reason = (
                    "A provider-target change must be adopted because it changes the "
                    "session's persistent execution target."
                )
            elif (
                authority_decision is not ExecutionProfileAuthorityDecision.DENIED
                and changes_egress_generation
            ):
                policy_reason = (
                    "An egress-authority generation change must be durably adopted even "
                    "when its effective permissions are unchanged."
                )
        elif action is ExecutionProfilePolicyAction.ADOPT:
            has_adoption_authority = (
                intent is not None
                or model_target_only
                or built_in_ceiling_narrowing
                or built_in_model_target_and_ceiling_adoption
                or automatic_egress_narrowing
            )
            authority_is_sufficient = (
                authority_decision is not ExecutionProfileAuthorityDecision.DENIED
                and (
                    not broadens_authority
                    or authority_decision is ExecutionProfileAuthorityDecision.AUTHORIZED
                )
            )
            kind = (
                ExecutionProfileDecisionKind.ADOPTED
                if has_adoption_authority and authority_is_sufficient
                else ExecutionProfileDecisionKind.REJECTED
            )
            if not has_adoption_authority:
                policy_reason = "Explicit execution-profile adoption intent is required."
            elif (
                not authority_is_sufficient
                and authority_decision is not ExecutionProfileAuthorityDecision.DENIED
            ):
                policy_reason = "Authority-broadening adoption requires explicit authorization."
        elif action is ExecutionProfilePolicyAction.MIGRATION_REQUIRED:
            # Migration is a non-admitting classification, not an adoption.
            # A policy may require migration for ordinary drift even when the
            # caller did not ask Cayu to adopt the candidate profile.
            kind = ExecutionProfileDecisionKind.MIGRATION_REQUIRED
        else:
            kind = ExecutionProfileDecisionKind.REJECTED

        return _execution_profile_decision_event(
            session=event_session,
            expected_profile=expected_profile,
            candidate_profile=candidate_profile,
            changed_component_classes=changed_component_classes,
            kind=kind,
            policy_identity=policy_identity,
            policy_reason=policy_reason,
            authority_decision=authority_decision,
            intent=intent,
            adoption_request_fingerprint=adoption_request_fingerprint,
            fallback_actor=fallback_actor,
            fallback_reason=fallback_reason,
            clock=self._clock if decision_clock is None else decision_clock,
        )
