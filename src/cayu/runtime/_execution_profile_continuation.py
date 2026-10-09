"""Execution-profile continuation authority shared below runtime orchestration."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

from cayu._validation import canonical_durable_json_bytes, copy_durable_json_value
from cayu.approvals.tools import ResolutionActor
from cayu.budgets.base import (
    BudgetLimit,
    BudgetPolicy,
    budget_limits_for_session,
    request_budget_execution_profile_ids,
)
from cayu.budgets.run_limits import RunLimits, copy_run_limits
from cayu.context.structured_output import StructuredOutputSpec, copy_structured_output_spec
from cayu.context.thinking import ThinkingConfig, thinking_config_payload
from cayu.egress.authority import EgressAuthorityChangeKind
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileComponentClass,
    ExecutionProfileDecision,
    ExecutionProfileDecisionKind,
    ExecutionProfileIdentity,
    execution_profile_decision_payload,
    execution_profile_egress_authority_change,
)
from cayu.providers.base import ModelProvider, ModelRequest, privacy_safe_provider_option_projection
from cayu.providers.cache import CacheBreakpoint, CachePolicy
from cayu.providers.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime import _execution_profile_admission as execution_profile_admission
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._execution_profile_identity_validation import (
    copy_secret_free_execution_profile_behavior_identity,
)
from cayu.runtime._model_completion_contracts import model_completion_recovery_context_from_stage
from cayu.runtime._tool_completion import (
    load_recorded_tool_completion_policy,
    require_registered_completion_tools,
)
from cayu.runtime.build_provenance import current_runtime_build_provenance
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import (
    ExecutionProfileMismatchError,
    _with_runtime_execution_profile_decision_authority,
)
from cayu.runtime.loop_policies import LoopPolicy
from cayu.runtime.tool_completion import ToolCompletionPolicy, copy_tool_completion_policy
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._invocation_lifecycle import RejectInvocationCommand
from cayu.sessions.base import (
    QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY,
    ActiveModelCompletionStage,
    ModelFailoverPolicy,
    QueuedInteractionProfileHandoff,
    SessionRunFenced,
    SessionStore,
    queued_interaction_profile_handoff_evidence,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
)
from cayu.sessions.records import Session
from cayu.tools.exposure import tool_capability_ceiling_from_session_metadata
from cayu.vaults.redaction import SecretRedactor


def _execution_profile_provider_options(
    options: dict[str, Any],
    *,
    provider: ModelProvider | None = None,
    model: str = "execution-profile-options",
    process_identity: str,
) -> tuple[dict[str, Any], bool]:
    """Project the selected adapter's effective request-option policy.

    Built-in adapters remove inactive namespaces, normalize controls, and add
    provider defaults through ``request_fingerprint_options``.  The optional
    provider preserves the conservative raw-options behavior for internal
    helper callers that do not yet have a resolved adapter.
    """

    effective_options = options
    cache_policy_material: dict[str, Any] | None = None
    private_options_material: dict[str, Any] = effective_options
    provider_defined_cache_policy = False
    if provider is not None:
        detached_request = ModelRequest(
            model=model,
            messages=[],
            options=options,
        )
        projected_options = provider.request_fingerprint_options(detached_request)
        if type(projected_options) is not dict:
            raise TypeError("ModelProvider.request_fingerprint_options() must return a dict.")
        copied_options = copy_durable_json_value(
            projected_options,
            "execution profile provider options",
        )
        if type(copied_options) is not dict:  # pragma: no cover - copier invariant.
            raise TypeError("ModelProvider.request_fingerprint_options() must return a dict.")
        effective_options = copied_options
        private_options_material = effective_options
        effective_cache_policy, cache_policy_is_authoritative = (
            _execution_profile_effective_cache_policy(
                provider,
                model=model,
                options=detached_request.options,
            )
        )
        cache_policy_material = _execution_profile_cache_policy(effective_cache_policy)
        raw_cache_policy = detached_request.options.get("cache_policy")
        if not cache_policy_is_authoritative and raw_cache_policy is not None:
            copied_cache_policy = copy_durable_json_value(
                raw_cache_policy,
                "execution profile provider-defined cache policy",
            )
            private_options_material = {
                "provider_options": effective_options,
                "provider_defined_cache_policy": copied_cache_policy,
            }
            provider_defined_cache_policy = True

    projected: dict[str, Any] = {}
    opaque = False
    for namespace, value in effective_options.items():
        if type(value) is dict:
            safe = privacy_safe_provider_option_projection(value)
            if safe:
                projected[namespace] = safe
            visible_source = {key: item for key, item in value.items() if item is not None}
            opaque |= safe != visible_source
        elif value is not None:
            opaque = True
    if cache_policy_material is not None:
        projected["cache_policy"] = cache_policy_material
    elif provider_defined_cache_policy:
        projected["cache_policy"] = {"configuration": "provider_defined"}
        opaque = True
    if not opaque:
        return projected, False
    private_digest = hmac.new(
        process_identity.encode("utf-8"),
        canonical_durable_json_bytes(private_options_material, "provider_options"),
        hashlib.sha256,
    ).hexdigest()
    return (
        {
            "public_projection": projected,
            "private_configuration_hmac_sha256": private_digest,
            "process_scope": hashlib.sha256(process_identity.encode("utf-8")).hexdigest(),
        },
        True,
    )


def _execution_profile_cache_policy(
    policy: CachePolicy | None,
) -> dict[str, Any] | None:
    """Canonicalize the effective provider cache policy without request content."""

    if policy is None:
        return None
    if type(policy) is not CachePolicy:
        raise TypeError("ModelProvider.request_cache_policy() must return CachePolicy or None.")
    copied = CachePolicy.model_validate(policy.model_dump(mode="python"))
    breakpoints = set(copied.breakpoints)
    if copied.conversation_prefix_strategy == "none":
        breakpoints.discard(CacheBreakpoint.CONVERSATION_PREFIX)
    if not breakpoints:
        return None
    ordered = tuple(sorted(breakpoints, key=lambda item: item.value))
    material: dict[str, Any] = {
        "breakpoints": [breakpoint.value for breakpoint in ordered],
        "ttl": "extended" if copied.ttl == "extended" else "standard",
    }
    if CacheBreakpoint.CONVERSATION_PREFIX in breakpoints:
        material["conversation_prefix_strategy"] = copied.conversation_prefix_strategy
        if copied.conversation_prefix_strategy == "all_but_last_n":
            material["conversation_prefix_n"] = copied.conversation_prefix_n
    return material


def _execution_profile_effective_cache_policy(
    provider: ModelProvider,
    *,
    model: str,
    options: dict[str, Any],
) -> tuple[CachePolicy | None, bool]:
    """Resolve cache configuration only for an adapter whose complete contract is known."""

    from cayu.providers.anthropic import AnthropicProvider
    from cayu.providers.bedrock import BedrockProvider
    from cayu.providers.cache import resolve_cache_policy
    from cayu.providers.vertex import VertexProvider

    if type(provider) is AnthropicProvider or type(provider) is VertexProvider:
        return resolve_cache_policy(provider.cache_policy, options), True
    if type(provider) is BedrockProvider:
        return provider._resolved_cache_policy(model, options), True
    provider_type = type(provider)
    uses_default_cache_contract = (
        provider_type.request_cache_policy is ModelProvider.request_cache_policy
        and provider_type.request_cache_projection is ModelProvider.request_cache_projection
    )
    return None, uses_default_cache_contract


def _execution_profile_structured_output(
    spec: StructuredOutputSpec | None,
) -> dict[str, Any] | None:
    if spec is None:
        return None
    copied = copy_structured_output_spec(spec)
    if copied is None:
        return None
    schema_digest = hashlib.sha256(
        canonical_durable_json_bytes(copied.json_schema, "structured_output.json_schema")
    ).hexdigest()
    return {
        "kind": "structured_output",
        "version": 1,
        "schema_sha256": schema_digest,
        "strategy": copied.strategy.value,
        "max_retries": copied.max_retries,
        "name_sha256": (
            None if copied.name is None else hashlib.sha256(copied.name.encode("utf-8")).hexdigest()
        ),
        "repair_prompt_sha256": (
            None
            if copied.repair_prompt is None
            else hashlib.sha256(copied.repair_prompt.encode("utf-8")).hexdigest()
        ),
    }


def _execution_profile_decision_event_id(
    *,
    session_id: str,
    run_epoch: int,
    expected_profile: ExecutionProfileIdentity,
    candidate_profile: ExecutionProfileIdentity,
    intent: ExecutionProfileAdoptionIntent | None,
    policy_identity: str | None = None,
) -> tuple[str, str]:
    if intent is not None:
        idempotency_identity = intent.idempotency_key
        material = {
            "session_id": session_id,
            "idempotency_identity": idempotency_identity,
        }
    else:
        if policy_identity is None:
            raise ValueError(
                "A runtime-generated execution-profile decision requires policy authority."
            )
        policy_identity_digest = hashlib.sha256(policy_identity.encode("utf-8")).hexdigest()
        idempotency_identity = (
            f"run-epoch:{run_epoch}:{expected_profile.fingerprint}:"
            f"{candidate_profile.fingerprint}:{policy_identity_digest}"
        )
        material = {
            "session_id": session_id,
            "run_epoch": run_epoch,
            "expected_profile_fingerprint": expected_profile.fingerprint,
            "candidate_profile_fingerprint": candidate_profile.fingerprint,
            "policy_identity": policy_identity,
        }
    event_material = canonical_durable_json_bytes(material, "execution_profile_decision")
    return "epd_" + hashlib.sha256(event_material).hexdigest(), idempotency_identity


def _execution_profile_decision_event(
    *,
    session: Session,
    expected_profile: ExecutionProfileIdentity,
    candidate_profile: ExecutionProfileIdentity,
    changed_component_classes: tuple[ExecutionProfileComponentClass, ...],
    kind: ExecutionProfileDecisionKind,
    policy_identity: str,
    policy_reason: str,
    authority_decision: ExecutionProfileAuthorityDecision,
    intent: ExecutionProfileAdoptionIntent | None,
    adoption_request_fingerprint: str | None,
    fallback_actor: ResolutionActor | None,
    fallback_reason: str,
    clock: Callable[[], datetime],
) -> ExecutionProfileDecision:
    if (intent is None) != (adoption_request_fingerprint is None):
        raise ValueError(
            "Explicit execution-profile adoption intent and request fingerprint must "
            "be supplied together."
        )
    event_id, idempotency_identity = _execution_profile_decision_event_id(
        session_id=session.id,
        run_epoch=session.run_epoch,
        expected_profile=expected_profile,
        candidate_profile=candidate_profile,
        intent=intent,
        policy_identity=policy_identity,
    )
    actor = intent.requested_by if intent is not None else fallback_actor
    reason = intent.reason if intent is not None else fallback_reason
    egress_authority_change = execution_profile_egress_authority_change(
        expected_profile,
        candidate_profile,
        changed_component_classes=changed_component_classes,
    )
    if kind is ExecutionProfileDecisionKind.REJECTED and egress_authority_change is not None:
        egress_authority_change = EgressAuthorityChangeKind.REFUSED
    event = Event(
        id=event_id,
        type=(
            EventType.SESSION_EXECUTION_PROFILE_REJECTED
            if kind is ExecutionProfileDecisionKind.REJECTED
            else EventType.SESSION_EXECUTION_PROFILE_DECIDED
        ),
        session_id=session.id,
        timestamp=clock(),
        agent_name=session.agent_name,
        environment_name=session.environment_name,
        payload=execution_profile_decision_payload(
            kind=kind,
            expected_profile=expected_profile,
            candidate_profile=candidate_profile,
            changed_component_classes=changed_component_classes,
            policy_identity=policy_identity,
            policy_reason=policy_reason,
            authority_decision=authority_decision,
            egress_authority_change=egress_authority_change,
            idempotency_identity=idempotency_identity,
            adoption_request_fingerprint=adoption_request_fingerprint,
            actor=actor,
            reason=reason,
        ),
    )
    authority_fields = [
        "decision",
        "policy_identity",
        "authority_decision",
        "idempotency_identity",
    ]
    if adoption_request_fingerprint is not None:
        authority_fields.append("adoption_request_fingerprint")
    if egress_authority_change is not None:
        authority_fields.append("egress_authority_change")
    event = event_with_runtime_payload_authority(
        event_with_runtime_envelope_authority(
            event_with_runtime_generated_id(event),
            "session_id",
        ),
        *authority_fields,
    )
    return _with_runtime_execution_profile_decision_authority(
        ExecutionProfileDecision(
            kind=kind,
            expected_profile=expected_profile,
            candidate_profile=candidate_profile,
            changed_component_classes=changed_component_classes,
            policy_identity=policy_identity,
            policy_reason=policy_reason,
            authority_decision=authority_decision,
            egress_authority_change=egress_authority_change,
            idempotency_identity=idempotency_identity,
            adoption_request_fingerprint=adoption_request_fingerprint,
            actor=actor,
            reason=reason,
            event=event,
        )
    )


class ExecutionProfileContinuation:
    """Resolve saved execution settings and reject conflicting continuation authority."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        execution_profile_process_identity: str,
        get_registered_environment_for_session: Callable[
            [str | None], runtime_records.RegisteredEnvironment | None
        ],
        get_registered_provider: Callable[[str], runtime_records.RegisteredProvider],
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        loop_policies: tuple[LoopPolicy, ...],
        loop_policy_execution_profile_identities: tuple[
            ExecutionProfileBehaviorIdentity | None, ...
        ],
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._secret_redactor = secret_redactor
        self._clock = clock
        self._execution_profile_process_identity = execution_profile_process_identity
        self._get_registered_environment_for_session = get_registered_environment_for_session
        self._get_registered_provider = get_registered_provider
        self._runtime_hooks = runtime_hooks
        self._loop_policies = loop_policies
        self._loop_policy_execution_profile_identities = loop_policy_execution_profile_identities
        self.policy_identity_registry = (
            execution_profile_admission.ProcessLocalBehaviorIdentityRegistry()
        )

    def request_loop_policy_instance_identities(
        self,
        policies: tuple[LoopPolicy, ...],
    ) -> tuple[str, ...]:
        return tuple(self.policy_identity_registry.identity_for(policy) for policy in policies)

    async def validate(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        request_loop_policies: tuple[LoopPolicy, ...] | None = None,
        budget_policy: BudgetPolicy | None = None,
        request_budget_limits: tuple[BudgetLimit, ...] = (),
        structured_output: StructuredOutputSpec | None = None,
        thinking: ThinkingConfig | None = None,
        max_steps: int | None = None,
        limits: RunLimits | None = None,
        retry_policy: RetryPolicy | None = None,
        invocation_semantics_available: bool = False,
        frozen_candidate_profile: ExecutionProfileIdentity | None = None,
        require_open_interaction: bool = True,
        additional_profile_fingerprints: tuple[str, ...] = (),
        record_rejection: bool = True,
        tool_completion: ToolCompletionPolicy | None = None,
    ) -> ActiveInvocationExecutionProfile:
        """Validate without exporting process-local candidate collaborators."""

        resolved = await self.resolve(
            session=session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            request_loop_policies=request_loop_policies,
            budget_policy=budget_policy,
            request_budget_limits=request_budget_limits,
            structured_output=structured_output,
            thinking=thinking,
            max_steps=max_steps,
            limits=limits,
            retry_policy=retry_policy,
            tool_completion=tool_completion,
            invocation_semantics_available=invocation_semantics_available,
            frozen_candidate_profile=frozen_candidate_profile,
            require_open_interaction=require_open_interaction,
            additional_profile_fingerprints=additional_profile_fingerprints,
            record_rejection=record_rejection,
        )
        return resolved.snapshot

    async def resolve(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        request_failover: ModelFailoverPolicy | None = None,
        request_loop_policies: tuple[LoopPolicy, ...] | None = None,
        budget_policy: BudgetPolicy | None = None,
        request_budget_limits: tuple[BudgetLimit, ...] = (),
        structured_output: StructuredOutputSpec | None = None,
        thinking: ThinkingConfig | None = None,
        max_steps: int | None = None,
        limits: RunLimits | None = None,
        retry_policy: RetryPolicy | None = None,
        invocation_semantics_available: bool = False,
        frozen_candidate_profile: ExecutionProfileIdentity | None = None,
        require_open_interaction: bool = True,
        additional_profile_fingerprints: tuple[str, ...] = (),
        record_rejection: bool = True,
        tool_completion: ToolCompletionPolicy | None = None,
    ) -> execution_profile_admission.ExecutionProfileContinuationPlan:
        """Resolve a recovery continuation against its durable invocation profile."""

        if invocation_semantics_available and max_steps is None:
            raise ValueError("Recovery requires recorded invocation max_steps.")
        active_model_completion = await self._session_store.load_active_model_completion_stage(
            session.id
        )
        model_completion_context = (
            None
            if active_model_completion is None
            else model_completion_recovery_context_from_stage(active_model_completion.stage)
        )
        completion_profile_fingerprint = (
            None
            if model_completion_context is None
            else model_completion_context.execution_profile_fingerprint
        )
        if (
            active_model_completion is not None
            and active_model_completion.stage.purpose == "auxiliary-inference"
        ):
            completion_profile_fingerprint = active_model_completion.stage.intent.get(
                "execution_profile_fingerprint"
            )
            if type(completion_profile_fingerprint) is not str:
                raise ValueError("Auxiliary recovery stage lost its execution profile fingerprint.")
        provider_options, provider_options_process_local = _execution_profile_provider_options(
            registered_agent.spec.provider_options,
            provider=registered_provider.provider,
            model=session.model,
            process_identity=self._execution_profile_process_identity,
        )
        effective_thinking = thinking if thinking is not None else registered_agent.spec.thinking
        app_limit_ids = tuple(
            limit.budget_limit_id
            for limit in budget_limits_for_session(
                policy=budget_policy,
                agent_name=registered_agent.spec.name,
                causal_budget_id=session.causal_budget_id,
            )
        )
        request_limit_ids: tuple[str, ...] = ()
        if invocation_semantics_available:
            request_limit_ids = request_budget_execution_profile_ids(
                limits=request_budget_limits,
                agent_name=registered_agent.spec.name,
                causal_budget_id=session.causal_budget_id,
            )
        tool_completion = copy_tool_completion_policy(tool_completion)
        require_registered_completion_tools(tool_completion, registered_agent)
        recorded_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if (
            invocation_semantics_available
            and recorded_profile is not None
            and tool_completion is None
        ):
            tool_completion = await load_recorded_tool_completion_policy(
                self._session_store,
                session,
                checkpoint,
                execution_profile=recorded_profile.profile,
                max_steps=cast("int", max_steps),
                limits=copy_run_limits(limits),
                retry_policy=copy_retry_policy(retry_policy),
                context=model_completion_context,
            )
        plan = execution_profile_admission.prepare_execution_profile_continuation(
            session=session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            request_failover=request_failover,
            runtime_version=_runtime_version(),
            runtime_build_provenance=current_runtime_build_provenance(),
            redactor=self._secret_redactor,
            process_identity=self._execution_profile_process_identity,
            registered_environment=self._get_registered_environment_for_session(
                session.environment_name
            ),
            runtime_hooks=self._runtime_hooks,
            loop_policies=self._loop_policies,
            loop_policy_identities=self._loop_policy_execution_profile_identities,
            invocation_loop_policies=request_loop_policies,
            invocation_loop_policy_identities=(
                ()
                if request_loop_policies is None or frozen_candidate_profile is not None
                else tuple(
                    copy_secret_free_execution_profile_behavior_identity(
                        policy.execution_profile_identity,
                        redactor=self._secret_redactor,
                        field_name=(f"request.loop_policies[{index}].execution_profile_identity"),
                    )
                    for index, policy in enumerate(request_loop_policies)
                )
            ),
            invocation_loop_policy_instance_identities=(
                ()
                if request_loop_policies is None or frozen_candidate_profile is not None
                else self.request_loop_policy_instance_identities(request_loop_policies)
            ),
            additional_profile_fingerprints=(
                *additional_profile_fingerprints,
                *(() if active_model_completion is None else (completion_profile_fingerprint,)),
            ),
            frozen_candidate_profile=frozen_candidate_profile,
            provider_options=provider_options,
            provider_options_process_local=provider_options_process_local,
            thinking=(
                None if effective_thinking is None else thinking_config_payload(effective_thinking)
            ),
            app_budget_limit_ids=app_limit_ids,
            request_budget_limit_ids=request_limit_ids,
            structured_output=_execution_profile_structured_output(structured_output),
            finalization=(
                execution_profile_admission.model_finalization_material(
                    max_steps=cast("int", max_steps),
                    limits=copy_run_limits(limits),
                    retry_policy=copy_retry_policy(retry_policy),
                    tool_completion=tool_completion,
                )
                if invocation_semantics_available
                else None
            ),
            invocation_semantics_available=invocation_semantics_available,
            tool_capability_ceiling=tool_capability_ceiling_from_session_metadata(
                session.metadata
            ).tool_names,
            resolve_provider=self._get_registered_provider,
            resolve_candidate_provider_options=lambda provider, model: (
                _execution_profile_provider_options(
                    registered_agent.spec.provider_options,
                    provider=provider.provider,
                    model=model,
                    process_identity=self._execution_profile_process_identity,
                )
            ),
        )
        snapshot = plan.snapshot
        if snapshot.profile.model_failover is not None or plan.model_failover is not None:
            self._session_store._require_model_failover_stage_protocol()
        candidate = plan.candidate_profile
        changed = plan.changed_component_classes
        if not changed:
            if require_open_interaction:
                latest_interactions = await self._session_store.query_events(
                    EventQuery(
                        session_id=session.id,
                        event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                        order_by=EventOrder.SEQUENCE_DESC,
                        limit=1,
                    )
                )
                if not latest_interactions or (
                    latest_interactions[0].event.type in INTERACTION_TERMINAL_EVENT_TYPES
                ):
                    raise RuntimeError(
                        "Active invocation execution profile has no authoritative open interaction."
                    )
                open_interaction_id = latest_interactions[0].event.interaction_id
                if open_interaction_id is None:
                    raise RuntimeError("Interaction lifecycle event has no interaction identity.")
                if snapshot.interaction_id != open_interaction_id:
                    snapshot = await self._repair_historical_queued_profile_handoff(
                        session=session,
                        snapshot=snapshot,
                        open_interaction_event=latest_interactions[0].event,
                        active_model_completion=active_model_completion,
                    )
            if frozen_candidate_profile is not None:
                snapshot = snapshot.model_copy(update={"profile": frozen_candidate_profile})
            return execution_profile_admission.ExecutionProfileContinuationPlan(
                snapshot=snapshot,
                candidate_profile=candidate,
                changed_component_classes=(),
                model_failover=plan.model_failover,
            )

        if not record_rejection:
            raise ExecutionProfileMismatchError(
                session_id=session.id,
                expected_profile_fingerprint=snapshot.profile.fingerprint,
                candidate_profile_fingerprint=candidate.fingerprint,
                changed_component_classes=changed,
                expected_profile=snapshot.profile,
                candidate_profile=candidate,
            )

        policy_identity = "cayu:active-invocation-profile:v1"
        decision = _execution_profile_decision_event(
            session=session,
            expected_profile=snapshot.profile,
            candidate_profile=candidate,
            changed_component_classes=changed,
            kind=ExecutionProfileDecisionKind.REJECTED,
            policy_identity=policy_identity,
            policy_reason="Active invocation profiles require exact recovery reuse.",
            authority_decision=ExecutionProfileAuthorityDecision.NOT_REQUIRED,
            intent=None,
            adoption_request_fingerprint=None,
            fallback_actor=None,
            fallback_reason="The active invocation profile changed before continuation.",
            clock=self._clock,
        )
        rejection = await self._session_store.apply_invocation_lifecycle_command(
            RejectInvocationCommand(
                session_id=session.id,
                expected_session_instance_id=session.instance_id,
                expected_statuses=(session.status,),
                expected_run_epoch=session.run_epoch,
                expected_profile=snapshot.profile,
                candidate_profile=candidate,
                event=decision.event,
                decision=decision,
                expected_active_profile=snapshot,
            )
        )
        if not rejection.replayed:
            await self._event_writer.fan_out_persisted([rejection.event])
        raise ExecutionProfileMismatchError(
            session_id=session.id,
            expected_profile_fingerprint=snapshot.profile.fingerprint,
            candidate_profile_fingerprint=candidate.fingerprint,
            changed_component_classes=changed,
            expected_profile=snapshot.profile,
            candidate_profile=candidate,
        )

    async def _repair_historical_queued_profile_handoff(
        self,
        *,
        session: Session,
        snapshot: ActiveInvocationExecutionProfile,
        open_interaction_event: Event,
        active_model_completion: ActiveModelCompletionStage | None,
    ) -> ActiveInvocationExecutionProfile:
        """Repair only the legacy A-profile/B-delivery provider-dispatch split."""

        target_interaction_id = open_interaction_event.interaction_id
        recovery_context = (
            None
            if active_model_completion is None
            else model_completion_recovery_context_from_stage(active_model_completion.stage)
        )
        if (
            open_interaction_event.type is not EventType.INTERACTION_STARTED
            or target_interaction_id is None
            or recovery_context is None
            or recovery_context.interaction_id != target_interaction_id
            or recovery_context.execution_profile_fingerprint != snapshot.profile.fingerprint
        ):
            raise RuntimeError(
                "Active invocation execution profile belongs to another interaction."
            )
        predecessor_events = await self._session_store.query_events(
            EventQuery(
                session_id=session.id,
                interaction_id=snapshot.interaction_id,
                event_types=tuple(INTERACTION_TERMINAL_EVENT_TYPES),
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if (
            not predecessor_events
            or predecessor_events[0].event.type is not EventType.INTERACTION_COMPLETED
        ):
            raise RuntimeError(
                "Historical queued interaction handoff has no exact terminal predecessor."
            )
        target = ActiveInvocationExecutionProfile(
            session_id=session.id,
            interaction_id=target_interaction_id,
            run_epoch=session.run_epoch,
            profile=snapshot.profile,
        )
        handoff = QueuedInteractionProfileHandoff(
            expected_session_instance_id=session.instance_id,
            predecessor_settlement_event_id=predecessor_events[0].event.id,
            expected_active_profile=snapshot,
            target_active_profile=target,
        )
        has_durable_evidence = (
            QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY in open_interaction_event.payload
        )
        durable_evidence = open_interaction_event.payload.get(
            QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY
        )
        if has_durable_evidence and durable_evidence != queued_interaction_profile_handoff_evidence(
            handoff
        ):
            raise RuntimeError("Historical queued interaction handoff evidence is malformed.")
        repaired = await self._session_store.repair_queued_interaction_profile_handoff(
            session.id,
            interaction_started_event=open_interaction_event,
            profile_handoff=handoff,
        )
        if repaired != target:
            raise SessionRunFenced(
                "Historical queued interaction handoff repair returned different authority."
            )
        return target


def _runtime_version() -> str | None:
    try:
        return version("cayu")
    except PackageNotFoundError:
        return None
