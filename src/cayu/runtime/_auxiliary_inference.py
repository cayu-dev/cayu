"""Auxiliary inference orchestration over the existing durable model-stage owner."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from weakref import WeakValueDictionary

from cayu._exception_groups import exception_cause, exception_tree_contains, set_exception_cause
from cayu._task_wait import consume_pending_task_cancellation, unexpected_child_cancellation_error
from cayu._validation import (
    canonical_bounded_durable_json_bytes,
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
)
from cayu.budgets.base import (
    BudgetLimit,
    BudgetReservationRecoveryContext,
    budget_reservation_authority_sha256,
)
from cayu.budgets.billing import BillingIdentity, copy_billing_identity, resolved_billing_identity
from cayu.budgets.binding import BudgetBinding, copy_budget_binding
from cayu.budgets.usage import normalize_usage_metrics, usage_metrics_payload
from cayu.deadlines import ExecutionDeadline, execution_deadline_scope, expired_execution_deadline
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_nested_payload_authority,
    event_with_runtime_payload_authority,
)
from cayu.providers import (
    ModelProviderError,
    ModelRequest,
    ModelStreamDeadlineError,
    ModelStreamEvent,
    ModelStreamEventType,
)
from cayu.providers._credential_boundary import (
    release_provider_stream_cleanup,
    reserve_provider_stream_cleanup,
)
from cayu.providers.base import _copy_auxiliary_request
from cayu.providers.deadlines import ProviderStreamDeadlineAdmission
from cayu.providers.response import ModelResponse
from cayu.runtime._auxiliary_inference_contract import (
    AUXILIARY_ATTRIBUTION_AUTHORITY_PATHS,
    AuxiliaryInferenceAttribution,
)
from cayu.runtime._auxiliary_invocation import (
    AuxiliaryInferenceScope,
    AuxiliaryInvocationPolicy,
)
from cayu.runtime._auxiliary_response import AuxiliaryResponseCollector
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._execution_profile_admission import resolve_provider_adapter_component
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._model_errors import (
    model_provider_error_from_payload,
    resolve_request_billing_identity,
)
from cayu.runtime._provider_stream import (
    _admitted_model_provider_events,
    _owned_model_provider_events,
    _provider_stream_self_cancellation_error,
    _ProviderStreamSelfCancellation,
)
from cayu.runtime._run_limit_accounting import (
    has_run_limit_accounting_authority,
    restore_run_limit_accounting_context,
)
from cayu.runtime._run_limits import (
    _TRUSTED_BINDING_PROVENANCE,
    BudgetStepReservation,
    LimitEvaluation,
    OperationReservationSetup,
    RunLimitController,
    _model_completion_with_budget_settlement_evidence,
)
from cayu.runtime._runtime_records import RegisteredTool
from cayu.runtime.execution_profiles import (
    ExecutionProfileMismatchError,
    event_with_execution_profile_authority,
    execution_profile_with_component,
)
from cayu.runtime.execution_units import (
    ModelAttemptIdentity,
    ToolRoundIdentity,
    copy_model_attempt_identity,
    copy_tool_round_identity,
)
from cayu.runtime.retry_policy import RetryDecision, RetryPolicy, retry_decision
from cayu.sessions.base import (
    ModelCompletionStage,
    ModelCompletionStageRequest,
    RuntimePublicationRequest,
    Session,
    SessionModelCompletionStageConflict,
    SessionStatus,
    SessionStore,
)
from cayu.tools.inference import InferenceLimits, copy_inference_limits, validate_inference_purpose
from cayu.vaults.redaction import SecretRedactor


@dataclass(frozen=True)
class AuxiliaryAttemptResult:
    """A published, settled attempt; unresolved failures are raised instead."""

    result: ModelResponse | ModelProviderError
    events: tuple[Event, ...]
    retry: RetryDecision | None = None


class AuxiliaryInferenceOwner:
    """Use model stages and the budget outbox as the sole durable authorities.

    No alternate completion marker or result cache is maintained here. Preparation
    of terminal material is separate from publication so an exact retry retains
    event identities, timestamps, pricing and the entire expected publication.
    This internal owner does not itself confer authority on caller-supplied data.
    """

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        run_limit_controller: RunLimitController,
        clock: Callable[[], datetime],
        execution_profile_process_identity: str | None = None,
        profile_redactor: SecretRedactor | None = None,
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._run_limit_controller = run_limit_controller
        self._clock = clock
        self._execution_profile_process_identity = execution_profile_process_identity
        self._profile_redactor = profile_redactor
        self._session_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    def create_scope(
        self,
        *,
        session: Session,
        invocation: InvocationContext,
        policy: AuxiliaryInvocationPolicy,
        registered_tool: RegisteredTool,
        parent: ToolRoundIdentity,
        tool_call_id: str,
        idempotency_key: str,
        budget_limits: tuple[BudgetLimit, ...],
        budget_binding: BudgetBinding | None = None,
        redactor: Callable[[], SecretRedactor],
        refresh: Callable[[], Awaitable[None]],
        observe_event: Callable[[Event], None],
    ) -> AuxiliaryInferenceScope:
        """Bind the public capability to one frozen, runtime-owned tool invocation."""

        invocation._validate()
        session = session.model_copy(deep=True)
        budget_binding = None if budget_binding is None else copy_budget_binding(budget_binding)
        parent = copy_tool_round_identity(parent)

        def validate_provider() -> None:
            invocation._validate()
            if self._execution_profile_process_identity is None or self._profile_redactor is None:
                raise RuntimeError("Auxiliary dispatch requires profile resolution dependencies.")
            component = resolve_provider_adapter_component(
                registered_provider=invocation.registered_provider,
                runtime_version=session.runtime_version,
                process_identity=self._execution_profile_process_identity,
                redactor=self._profile_redactor,
            )
            if component != invocation.profile.component(component.component_class):
                candidate = execution_profile_with_component(invocation.profile, component)
                raise ExecutionProfileMismatchError(
                    session_id=session.id,
                    expected_profile_fingerprint=invocation.profile.fingerprint,
                    candidate_profile_fingerprint=candidate.fingerprint,
                    changed_component_classes=(component.component_class,),
                    expected_profile=invocation.profile,
                    candidate_profile=candidate,
                )

        async def refresh_execution() -> None:
            validate_provider()
            await refresh()
            validate_provider()

        async def invoke(
            request: ModelRequest, purpose: str, limits: InferenceLimits
        ) -> ModelResponse:
            validate_provider()
            request, limits = self.prepare_request(
                invocation=invocation,
                registered_tool=registered_tool,
                request=request,
                purpose=purpose,
                limits=limits,
            )
            # Scheduling only: the store remains the atomic dispatch authority.
            lock = self._session_locks.setdefault(session.id, asyncio.Lock())
            effective_budget_binding = await self._run_limit_controller._binding_for_dispatch(
                request={
                    "session_id": session.id,
                    "agent_name": invocation.binding.agent_name,
                    "kind": "auxiliary",
                },
                binding=budget_binding,
            )

            async def run_attempt(attempt: int) -> AuxiliaryAttemptResult:
                await refresh_execution()
                provider = invocation.registered_provider.provider
                billing_identity = await resolve_request_billing_identity(
                    provider,
                    request.model_copy(deep=True),
                    provider_name=invocation.binding.provider_name,
                )
                policy_check = await self._run_limit_controller.evaluate_policy_budgets(
                    session=session,
                    agent_name=invocation.binding.agent_name,
                    environment_name=invocation.binding.environment_name,
                    budget_policy=invocation.budget_policy,
                    billing_identity_state=resolved_billing_identity(billing_identity),
                    pricing_provider_name=provider.billing_provider_name
                    or invocation.binding.provider_name,
                    model=invocation.binding.model,
                    execution_profile_fingerprint=invocation.profile.fingerprint,
                )
                for event in policy_check.events:
                    observe_event(event)
                if policy_check.check is not None:
                    raise RuntimeError(
                        "Auxiliary inference was refused by the invocation budget policy."
                    )
                evaluation = await self.evaluate_admission(
                    session=session,
                    invocation=invocation,
                    policy=policy,
                    budget_limits=budget_limits,
                    budget_binding=effective_budget_binding,
                    request_limits=limits,
                )
                for event in evaluation.events:
                    observe_event(event)
                if evaluation.decision is not None:
                    raise RuntimeError("Auxiliary inference was refused by invocation limits.")

                def make_stage(
                    reservations: tuple[BudgetStepReservation, ...] = (),
                ) -> ModelCompletionStageRequest:
                    return self.stage_request(
                        session=session,
                        invocation=invocation,
                        policy=policy,
                        parent=parent,
                        tool_name=registered_tool.name,
                        tool_call_id=tool_call_id,
                        idempotency_key=idempotency_key,
                        purpose=purpose,
                        request=request,
                        limits=limits,
                        attempt=attempt,
                        billing_identity=billing_identity,
                        reservations=reservations,
                        budget_binding=effective_budget_binding,
                    )

                preliminary = make_stage()
                setup = await self.reserve_attempt(
                    session=session,
                    invocation=invocation,
                    budget_limits=budget_limits,
                    request_limits=limits,
                    billing_identity=billing_identity,
                    budget_binding=effective_budget_binding,
                    identity=ModelAttemptIdentity(
                        model_step_id=preliminary.intent["model_step_id"],
                        model_attempt_id=preliminary.intent["model_attempt_id"],
                    ),
                )
                try:
                    if setup.error is not None:
                        raise setup.error
                    for event in setup.events:
                        observe_event(await self._event_writer.emit(event))
                    if setup.failure is not None:
                        raise RuntimeError("Auxiliary budget reservation was refused.")
                    staged = make_stage(setup.reservations)
                    cursor = await self._session_store.load_transcript_cursor(session.id)
                    await refresh_execution()
                except BaseException as failure:
                    try:
                        async for event in self._run_limit_controller.release_reservations(
                            list(setup.reservations),
                            session=session,
                            agent_name=invocation.binding.agent_name,
                            environment_name=invocation.binding.environment_name,
                            reason="Auxiliary inference stopped before stage preparation.",
                        ):
                            observe_event(event)
                    except BaseException as release_failure:
                        if isinstance(failure, asyncio.CancelledError):
                            raise failure from release_failure
                        raise BaseExceptionGroup(
                            "Auxiliary setup and release failed", [failure, release_failure]
                        ) from None
                    raise
                return await self.execute_attempt(
                    invocation=invocation,
                    request=request,
                    stage_request=staged,
                    reservations=setup.reservations,
                    limits=limits,
                    expected_transcript_cursor=cursor,
                    redactor=redactor(),
                    billing_identity=billing_identity,
                    refresh_live=refresh_execution,
                )

            async with (
                execution_deadline_scope(
                    ExecutionDeadline.after(
                        limits.timeout_seconds, source="auxiliary", scope="invocation"
                    )
                ),
                lock,
            ):
                for attempt in range(policy.retry_policy.max_attempts):
                    settled = await run_attempt(attempt)
                    for event in settled.events:
                        observe_event(event)
                    if isinstance(settled.result, ModelResponse):
                        return settled.result
                    error = settled.result
                    decision = settled.retry
                    if decision is None or decision.attempt != attempt + 1:
                        raise RuntimeError(
                            "Auxiliary attempt lost its settled retry decision."
                        ) from error
                    if not decision.retry:
                        raise error
                    await asyncio.sleep(decision.delay_seconds)
            raise AssertionError("Auxiliary retry policy exhausted without a terminal decision.")

        return AuxiliaryInferenceScope(invoke)

    def prepare_request(
        self,
        *,
        invocation: InvocationContext,
        registered_tool: RegisteredTool,
        request: ModelRequest,
        purpose: str,
        limits: InferenceLimits,
    ) -> tuple[ModelRequest, InferenceLimits]:
        """Resolve bounded content against the frozen invocation, without I/O.

        This is not dispatch admission. The eventual provider call still needs
        current epoch, budget, deadline and durable stage fences.
        """

        if type(invocation) is not InvocationContext:
            raise TypeError("Auxiliary inference requires runtime invocation authority.")
        invocation._validate()
        if not self._session_store._supports_model_completion_recovery_fence_protocol():
            raise NotImplementedError("Auxiliary inference requires atomic model recovery fences.")
        if (
            type(registered_tool) is not RegisteredTool
            or invocation.registered_agent.executable_tool(registered_tool.name)
            is not registered_tool
            or registered_tool.auxiliary_inference is None
        ):
            raise ValueError("Tool has no auxiliary inference authority in this invocation.")
        policy = registered_tool.auxiliary_inference
        purpose = validate_inference_purpose(purpose)
        if purpose not in policy.purposes:
            raise ValueError("Auxiliary inference purpose is not allowed by the tool policy.")
        limits = copy_inference_limits(limits).bounded_by(policy.limits)
        if type(request) is not ModelRequest:
            raise TypeError("Auxiliary inference requires a ModelRequest.")
        if type(request.model) is not str or request.model != invocation.binding.model:
            raise ValueError("Auxiliary model target is outside the inherited invocation profile.")
        if (
            type(request.tools) is not list
            or len(request.tools) != 0
            or type(request.hosted_tools) is not tuple
            or len(request.hosted_tools) != 0
            or request.targeted_tool_projection is not None
            or request.tool_discovery_projection is not None
            or type(request.options) is not dict
            or len(request.options) != 0
            or type(request.messages) is not list
        ):
            raise ValueError("Auxiliary requests cannot supply tools, raw options or projections.")
        copied = ModelRequest(model=request.model, messages=request.messages)
        canonical_bounded_durable_json_bytes(
            copied.model_dump(mode="json"),
            "auxiliary_request",
            max_bytes=limits.max_request_bytes,
            max_nodes=limits.max_request_bytes,
        )
        provider = invocation.registered_provider.provider
        prepared = provider.prepare_auxiliary_request(
            ModelRequest(model=copied.model, messages=copied.messages),
            max_output_tokens=limits.max_output_tokens,
        )
        if type(prepared) is not ModelRequest:
            raise TypeError("Provider auxiliary preparation must return a ModelRequest.")
        prepared = _copy_auxiliary_request(prepared)
        if (
            prepared.model != copied.model
            or prepared.messages != copied.messages
            or prepared.tools
            or prepared.hosted_tools
            or prepared.targeted_tool_projection is not None
            or prepared.tool_discovery_projection is not None
        ):
            raise ValueError("Provider auxiliary preparation changed the authorized request.")
        canonical_bounded_durable_json_bytes(
            prepared.model_dump(mode="json"),
            "auxiliary_request",
            max_bytes=limits.max_request_bytes,
            max_nodes=limits.max_request_bytes,
        )
        return prepared, limits

    async def evaluate_admission(
        self,
        *,
        session: Session,
        invocation: InvocationContext,
        policy: AuxiliaryInvocationPolicy,
        budget_limits: tuple[BudgetLimit, ...],
        request_limits: InferenceLimits,
        budget_binding: BudgetBinding | None = None,
    ) -> LimitEvaluation:
        """Use the normal bounded accounting owner, including original run scope.

        A successful evaluation is only a preflight. Reservations and the atomic
        model-stage dispatch fence must still succeed before provider entry.
        """

        if (
            type(invocation) is not InvocationContext
            or type(policy) is not AuxiliaryInvocationPolicy
        ):
            raise TypeError("Auxiliary admission requires frozen runtime invocation inputs.")
        invocation._validate()
        binding = invocation.binding
        common_binding = None if budget_binding is None else copy_budget_binding(budget_binding)
        if (
            session.id != binding.session_id
            or session.run_epoch != binding.run_epoch
            or session.instance_id != binding.session_instance_id
        ):
            raise ValueError("Auxiliary admission conflicts with the active invocation.")
        limits = policy.limits
        accounting = policy.accounting
        run_started_at = time.monotonic()
        baseline = None
        authorities = None
        if accounting is not None:
            run_started_at, baseline, authorities = restore_run_limit_accounting_context(
                accounting, session_id=session.id, budget_limits=budget_limits, now=self._clock()
            )
        elif has_run_limit_accounting_authority(limits, budget_limits):
            raise ValueError("Auxiliary admission lost the original run accounting authority.")
        provider = invocation.registered_provider.provider
        if common_binding is not None and (
            (
                common_binding.provider_name is not None
                and common_binding.provider_name != binding.provider_name
            )
            or (common_binding.model is not None and common_binding.model != binding.model)
            or (
                common_binding.environment_name is not None
                and common_binding.environment_name != binding.environment_name
            )
        ):
            raise ValueError("Budget binding conflicts with the auxiliary invocation.")
        return await self._run_limit_controller.evaluate_request_limits(
            session=session,
            agent_name=binding.agent_name,
            environment_name=binding.environment_name,
            limits=limits,
            budget_limits=budget_limits,
            run_started_at=run_started_at,
            run_baseline=baseline,
            run_budget_authorities=authorities,
            pricing_provider_name=provider.billing_provider_name or binding.provider_name,
            model=binding.model,
            execution_profile_fingerprint=invocation.profile.fingerprint,
            auxiliary_request_limits=request_limits,
        )

    async def reserve_attempt(
        self,
        *,
        session: Session,
        invocation: InvocationContext,
        budget_limits: tuple[BudgetLimit, ...],
        request_limits: InferenceLimits,
        identity: ModelAttemptIdentity,
        billing_identity: BillingIdentity | None,
        budget_binding: BudgetBinding | None = None,
    ) -> OperationReservationSetup:
        """Reserve through the existing ledger, never silently skip a hard cap.

        Billing identity and attempt identity come from the runtime, not request
        content. The caller owns the returned reservations until they are bound
        to a durable stage or released as positively not dispatched.
        """

        if type(invocation) is not InvocationContext or type(session) is not Session:
            raise TypeError("Auxiliary reservations require runtime session authority.")
        invocation._validate()
        binding = invocation.binding
        if (
            session.id != binding.session_id
            or session.instance_id != binding.session_instance_id
            or session.run_epoch != binding.run_epoch
        ):
            raise ValueError("Auxiliary reservation conflicts with the active invocation.")
        request_limits = copy_inference_limits(request_limits)
        identity = copy_model_attempt_identity(identity)
        budget_limits = self._run_limit_controller.provider_budget_limits(
            session=session,
            agent_name=binding.agent_name,
            budget_policy=invocation.budget_policy,
            request_budget_limits=budget_limits,
        )
        for limit in budget_limits:
            if limit.action != "interrupt":
                continue
            reservation = limit.reservation
            if reservation is None:
                raise ValueError(
                    "Auxiliary hard budgets require a configured reservation envelope."
                )
            if (
                reservation.max_input_tokens < request_limits.max_input_tokens
                or reservation.max_output_tokens < request_limits.max_output_tokens
            ):
                raise ValueError("Auxiliary request exceeds its hard-budget reservation envelope.")
        provider = invocation.registered_provider.provider
        return await self._run_limit_controller.reserve_operation_budgets(
            budget_limits=budget_limits,
            session_id=session.id,
            agent_name=binding.agent_name,
            environment_name=binding.environment_name,
            provider_name=provider.billing_provider_name or binding.provider_name,
            model=binding.model,
            model_attempt_identity=identity,
            execution_profile_fingerprint=invocation.profile.fingerprint,
            settlement_event_payload={"interaction_id": binding.interaction_id},
            billing_identity=billing_identity,
            binding=budget_binding,
            rejection_release_reason="Auxiliary inference admission was rejected before dispatch.",
            accepted_record_error="Auxiliary inference reservation lost its durable identity.",
            _binding_provenance=_TRUSTED_BINDING_PROVENANCE,
        )

    def stage_request(
        self,
        *,
        session: Session,
        invocation: InvocationContext,
        policy: AuxiliaryInvocationPolicy,
        parent: ToolRoundIdentity,
        tool_name: str,
        tool_call_id: str,
        idempotency_key: str,
        purpose: str,
        request: ModelRequest,
        limits: InferenceLimits,
        attempt: int,
        reservations: tuple[BudgetStepReservation, ...] = (),
        billing_identity: BillingIdentity | None = None,
        budget_binding: BudgetBinding | None = None,
    ) -> ModelCompletionStageRequest:
        """Build the complete private preparation tuple, without persisting content.

        Operation identity stays stable across enclosing-tool recovery. Epoch,
        request, policy and target remain decision-bearing intent fields rather
        than creating a fresh identity that could bypass an existing fence.
        """

        if type(invocation) is not InvocationContext or type(session) is not Session:
            raise TypeError("Auxiliary stage requires runtime session authority.")
        invocation._validate()
        binding = invocation.binding
        if (
            session.id != binding.session_id
            or session.instance_id != binding.session_instance_id
            or session.run_epoch != binding.run_epoch
        ):
            raise ValueError("Auxiliary stage conflicts with its invocation.")
        if type(attempt) is not int or not 0 <= attempt < policy.retry_policy.max_attempts:
            raise ValueError("Auxiliary attempt ordinal is outside the invocation retry policy.")
        parent = copy_tool_round_identity(parent)
        tool_name = require_durable_clean_nonblank(tool_name, "tool_name")
        tool_call_id = require_durable_clean_nonblank(tool_call_id, "tool_call_id")
        idempotency_key = require_durable_clean_nonblank(idempotency_key, "idempotency_key")
        purpose = validate_inference_purpose(purpose)
        limits = copy_inference_limits(limits)
        if type(request) is not ModelRequest:
            raise TypeError("Auxiliary stage requires its prepared model request.")
        request = ModelRequest(
            **{name: getattr(request, name) for name in ModelRequest.model_fields}
        )
        if request.model != binding.model:
            raise ValueError("Auxiliary stage request conflicts with the inherited model.")
        billing_identity = copy_billing_identity(billing_identity)
        operation_material = {
            "session_instance_id": session.instance_id,
            "parent": parent.payload(),
            "tool_call_id": tool_call_id,
        }
        operation_digest = sha256(
            canonical_durable_json_bytes(operation_material, "auxiliary_operation")
        ).hexdigest()
        attempt_digest = sha256(
            canonical_durable_json_bytes(
                {"operation_id": operation_digest, "attempt": attempt}, "auxiliary_attempt"
            )
        ).hexdigest()
        attribution = AuxiliaryInferenceAttribution(
            operation_id=f"aux_{operation_digest}",
            purpose=purpose,
            parent=parent,
            tool_call_id=tool_call_id,
        )
        request_digest = sha256(
            canonical_bounded_durable_json_bytes(
                request.model_dump(mode="json"),
                "auxiliary_request",
                max_bytes=limits.max_request_bytes,
                max_nodes=limits.max_request_bytes,
            )
        ).hexdigest()
        accounting = policy.accounting
        intent = {
            "model_step_id": f"mstep_{operation_digest[:32]}",
            "model_attempt_id": f"matt_{attempt_digest[:32]}",
            "auxiliary_inference": attribution.model_dump(mode="json"),
            "session_instance_id": session.instance_id,
            "invocation": session.invocation.model_dump(mode="json"),
            "interaction_id": binding.interaction_id,
            "source_run_epoch": binding.run_epoch,
            "execution_profile_fingerprint": invocation.profile.fingerprint,
            "causal_budget_id": session.causal_budget_id,
            "agent_name": binding.agent_name,
            "environment_name": binding.environment_name,
            "provider_name": binding.provider_name,
            "pricing_provider_name": invocation.registered_provider.provider.billing_provider_name
            or binding.provider_name,
            "billing_identity": None
            if billing_identity is None
            else billing_identity.model_dump(mode="json"),
            "budget_reservations": [
                BudgetReservationRecoveryContext(
                    reservation_id=item.record.reservation_id,
                    budget_limit_id=item.record.budget_limit_id,
                    limit=item.limit,
                    reservation_authority_sha256=budget_reservation_authority_sha256(item.record),
                ).model_dump(mode="json")
                for item in reservations
            ],
            "allocation_fingerprint": (
                None
                if invocation.registered_environment is None
                else invocation.registered_environment.live_allocation_fingerprint
            ),
            "requested_model": binding.model,
            "tool_name": tool_name,
            "idempotency_key": idempotency_key,
            "request_fingerprint": request_digest,
            "limits": limits.model_dump(mode="json"),
            "run_limits": policy.limits.model_dump(mode="json"),
            "retry_policy": policy.retry_policy.model_dump(mode="json"),
            "run_limit_accounting": None
            if accounting is None
            else accounting.model_dump(mode="json"),
        }
        if budget_binding is not None:
            intent["budget_binding_id"] = budget_binding.binding_id
            intent["budget_binding_authority_sha256"] = budget_binding.authority_digest
        return ModelCompletionStageRequest(
            stage_id=f"auxiliary:{attempt_digest}",
            logical_step_id=f"auxiliary:{attempt_digest}",
            dispatch_ordinal=attempt,
            purpose="auxiliary-inference",
            intent=intent,
            reservation_ids=tuple(item.record.reservation_id for item in reservations),
        )

    async def prepare_dispatch(
        self,
        *,
        session_id: str,
        request: ModelCompletionStageRequest,
        reservations: tuple[BudgetStepReservation, ...],
        expected_run_epoch: int,
        expected_transcript_cursor: int,
    ) -> ModelCompletionStage:
        """Consume a new preparation through both existing dispatch fences.

        Only normal return permits the owning caller to enter provider code.
        Neither an exact preparation replay nor a lost acknowledgement grants a
        second dispatch. On failure the durable stage remains the recovery owner;
        the caller must not release reservations merely because its wait ended.
        """

        if (
            type(request) is not ModelCompletionStageRequest
            or request.purpose != "auxiliary-inference"
        ):
            raise ValueError("Auxiliary dispatch requires its exact stage request.")
        request = ModelCompletionStageRequest.model_validate(request.model_dump(mode="python"))
        if tuple(item.record.reservation_id for item in reservations) != request.reservation_ids:
            raise ValueError("Auxiliary dispatch lost its prepared reservation identities.")
        identity = ModelAttemptIdentity.model_validate(
            {
                "model_step_id": request.intent.get("model_step_id"),
                "model_attempt_id": request.intent.get("model_attempt_id"),
            }
        )
        prepared = await self._session_store.prepare_model_completion_stage(
            session_id,
            request=request,
            expected_statuses={SessionStatus.RUNNING},
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )
        if not prepared.dispatch_authorized:
            raise SessionModelCompletionStageConflict(
                "Auxiliary preparation replay cannot authorize another provider dispatch."
            )
        deferred_failure = await self._run_limit_controller.mark_reservations_dispatched(
            reservations, dispatch_id=identity.model_attempt_id
        )
        if deferred_failure is not None:
            raise deferred_failure
        await self._session_store.mark_model_completion_stage_dispatched(
            session_id,
            stage=prepared.stage,
            consume_child_session_notifications=False,
        )
        return prepared.stage

    async def execute_attempt(
        self,
        *,
        invocation: InvocationContext,
        request: ModelRequest,
        stage_request: ModelCompletionStageRequest,
        reservations: tuple[BudgetStepReservation, ...],
        limits: InferenceLimits,
        expected_transcript_cursor: int,
        redactor: SecretRedactor,
        billing_identity: BillingIdentity | None = None,
        refresh_live: Callable[[], Awaitable[None]] | None = None,
    ) -> AuxiliaryAttemptResult:
        """Run one admitted attempt; retain an ambiguous stage for recovery.

        The scoped invocation owner supplies the prepared request and reserved
        budget envelope. This method owns stream capacity, dispatch and terminal
        accounting, not admission policy, retries, or enclosing-tool lifetime.
        """

        admission: ProviderStreamDeadlineAdmission | None = None
        cleanup = None
        try:
            invocation._validate()
            binding = invocation.binding
            limits = copy_inference_limits(limits)
            billing_identity = copy_billing_identity(billing_identity)
            if (
                type(request) is not ModelRequest
                or type(stage_request) is not ModelCompletionStageRequest
            ):
                raise TypeError("Auxiliary execution requires prepared request and stage values.")
            request = ModelRequest(
                **{name: getattr(request, name) for name in ModelRequest.model_fields}
            )
            stage_request = ModelCompletionStageRequest.model_validate(
                stage_request.model_dump(mode="python")
            )
            retry_policy = RetryPolicy.model_validate(stage_request.intent["retry_policy"])
            fingerprint = sha256(
                canonical_bounded_durable_json_bytes(
                    request.model_dump(mode="json"),
                    "auxiliary_request",
                    max_bytes=limits.max_request_bytes,
                    max_nodes=limits.max_request_bytes,
                )
            ).hexdigest()
            if (
                stage_request.intent.get("request_fingerprint") != fingerprint
                or stage_request.intent.get("source_run_epoch") != binding.run_epoch
                or stage_request.intent.get("session_instance_id") != binding.session_instance_id
                or stage_request.intent.get("execution_profile_fingerprint")
                != invocation.profile.fingerprint
                or stage_request.intent.get("provider_name") != binding.provider_name
                or request.model != binding.model
                or stage_request.intent.get("limits") != limits.model_dump(mode="json")
                or stage_request.intent.get("billing_identity")
                != (None if billing_identity is None else billing_identity.model_dump(mode="json"))
            ):
                raise ValueError("Auxiliary dispatch conflicts with its prepared invocation.")
            provider = invocation.registered_provider.provider
            admission = ProviderStreamDeadlineAdmission(provider.stream_deadlines)
            transferred = False
            collector = AuxiliaryResponseCollector(max_bytes=limits.max_response_bytes)
            terminal_seen = False
            terminal_completed = False
            metrics = None
            usage_status = "missing"
            failure: BaseException | None = None
            provider_failure: ModelProviderError | None = None
            conclusive_failure = False
            deadline_expired = False
            terminal_cancellation: asyncio.CancelledError | None = None
            response: ModelResponse | None = None
            task = asyncio.current_task()
            cancellation_baseline = 0 if task is None else task.cancelling()

            def classify_cancellation(error: BaseException) -> BaseException:
                if not isinstance(error, asyncio.CancelledError) or (
                    task is not None and task.cancelling() > cancellation_baseline
                ):
                    return error
                if isinstance(error, _ProviderStreamSelfCancellation):
                    return _provider_stream_self_cancellation_error(binding.provider_name)
                return unexpected_child_cancellation_error(
                    error, operation="Auxiliary inference operation"
                )

            async def refresh() -> None:
                invocation._validate()
                if refresh_live is not None:
                    await refresh_live()
                invocation._validate()

            cleanup = reserve_provider_stream_cleanup(admission.max_concurrent_streams)
        except BaseException as preparation_failure:
            if admission is not None:
                admission.close()
            if cleanup is not None:
                release_provider_stream_cleanup(cleanup)
            try:
                async for _ in self._run_limit_controller.release_operation_reservations(
                    list(reservations),
                    reason="Auxiliary inference stopped before stage preparation.",
                ):
                    pass
                await self._run_limit_controller.recover_pending_budget_settlements()
            except BaseException as release_failure:
                if isinstance(preparation_failure, asyncio.CancelledError):
                    raise preparation_failure from release_failure
                raise BaseExceptionGroup(
                    "Auxiliary preparation and reservation release failed",
                    [preparation_failure, release_failure],
                ) from None
            raise

        try:
            stage = await self.prepare_dispatch(
                session_id=binding.session_id,
                request=stage_request,
                reservations=reservations,
                expected_run_epoch=binding.run_epoch,
                expected_transcript_cursor=expected_transcript_cursor,
            )
            try:
                async with execution_deadline_scope(
                    ExecutionDeadline.after(
                        limits.timeout_seconds, source="auxiliary", scope="model"
                    )
                ):
                    started = Event(
                        type=EventType.MODEL_AUXILIARY_ATTEMPT_STARTED,
                        session_id=binding.session_id,
                        interaction_id=binding.interaction_id,
                        agent_name=binding.agent_name,
                        environment_name=binding.environment_name,
                        payload={
                            key: stage.intent[key]
                            for key in (
                                "model_step_id",
                                "model_attempt_id",
                                "auxiliary_inference",
                                "provider_name",
                                "requested_model",
                            )
                        }
                        | {"attempt": stage.dispatch_ordinal + 1},
                    )
                    started = event_with_runtime_payload_authority(
                        started, "model_step_id", "model_attempt_id"
                    )
                    started = event_with_execution_profile_authority(started, invocation.profile)
                    started = event_with_runtime_nested_payload_authority(
                        started, *AUXILIARY_ATTRIBUTION_AUTHORITY_PATHS
                    )
                    started = await self._event_writer.emit(started)
                    events = _owned_model_provider_events(
                        lambda: _admitted_model_provider_events(
                            provider, request, admission, refresh
                        ),
                        cancellation_baseline=cancellation_baseline,
                        max_concurrent_streams=admission.max_concurrent_streams,
                        cleanup_ownership=cleanup,
                    )
                    transferred = True
                    async with aclosing(events):
                        async for event in events:
                            if (
                                type(event) is ModelStreamEvent
                                and event.type
                                in {
                                    ModelStreamEventType.COMPLETED,
                                    ModelStreamEventType.ERROR,
                                }
                                and not terminal_seen
                            ):
                                terminal_seen = True
                                raw_usage = (
                                    event.payload.get("usage")
                                    if type(event.payload) is dict
                                    else None
                                )
                                metrics = normalize_usage_metrics(
                                    provider_name=provider.billing_provider_name
                                    or binding.provider_name,
                                    model=binding.model,
                                    requested_model=binding.model,
                                    raw_usage=raw_usage,
                                    usage_dialect=invocation.registered_provider.usage_dialect,
                                    billing_identity=billing_identity,
                                )
                                usage_status = (
                                    "observed"
                                    if metrics is not None
                                    else ("missing" if raw_usage is None else "malformed")
                                )
                            if (
                                type(event) is ModelStreamEvent
                                and event.type is ModelStreamEventType.ERROR
                            ):
                                accepted = collector.accept_error(event)
                                safe_payload = redactor.redact_json_values(accepted.payload)
                                provider_failure = model_provider_error_from_payload(
                                    safe_payload, fallback_provider=binding.provider_name
                                ) or ModelProviderError(
                                    "Auxiliary model provider reported a failure.",
                                    provider=binding.provider_name,
                                )
                                if isinstance(provider_failure, ModelStreamDeadlineError):
                                    raise provider_failure
                            else:
                                collector.add(event)
                                # Usage is observed before validation, but only
                                # accepted terminal content authorizes completion
                                # publication and release of the recovery fence.
                                if event.type is ModelStreamEventType.COMPLETED:
                                    terminal_completed = True
                    if provider_failure is None:
                        raw_response = collector.finish()
                        public_response = AuxiliaryResponseCollector(
                            max_bytes=limits.max_response_bytes
                        )
                        for event in raw_response.events:
                            public_response.add(
                                ModelStreamEvent.model_validate(
                                    redactor.redact_json_values(event.model_dump(mode="json"))
                                )
                            )
                        response = public_response.finish()
                    else:
                        # Only validated terminal error + normal stream/cleanup
                        # completion permits publication followed by a retry.
                        failure = provider_failure
                        conclusive_failure = True
            except BaseException as exc:
                conclusive_failure = False
                original = exc
                if isinstance(original, asyncio.CancelledError) and terminal_completed:
                    terminal_cancellation = consume_pending_task_cancellation(original)
                    exc = None
                if exc is not None:
                    exc = classify_cancellation(exc)
                if (
                    isinstance(original, _ProviderStreamSelfCancellation)
                    and isinstance(exc, ModelProviderError)
                    and provider_failure is None
                ):
                    provider_failure = exc
                failure = (
                    None
                    if exc is None
                    else (
                        BaseExceptionGroup(
                            "Auxiliary provider error and stream termination failed",
                            [provider_failure, exc],
                        )
                        if provider_failure is not None and provider_failure is not exc
                        else exc
                    )
                )
                if isinstance(exc, asyncio.CancelledError) and provider_failure is not None:
                    failure = exc
                    prior = exception_cause(exc)
                    set_exception_cause(
                        failure,
                        provider_failure
                        if prior is None
                        else BaseExceptionGroup(
                            "Auxiliary provider error and cancellation diagnostics",
                            [provider_failure, prior],
                        ),
                    )
                deadline_expired = (
                    expired_execution_deadline() is not None
                    or isinstance(exc, TimeoutError)
                    or (exc is not None and exception_tree_contains(exc, ModelStreamDeadlineError))
                )
            outcome = (
                "completed"
                if failure is None or terminal_completed
                else (
                    "timed_out"
                    if deadline_expired
                    else "cancelled"
                    if isinstance(failure, asyncio.CancelledError)
                    else "failed"
                )
            )
            decision = (
                retry_decision(
                    policy=retry_policy,
                    attempt=stage.dispatch_ordinal + 1,
                    error=str(provider_failure),
                    status_code=provider_failure.status_code,
                    retryable=provider_failure.retryable,
                    retry_after_s=provider_failure.retry_after_s,
                    unknown_provider_error=(
                        provider_failure.status_code is None and provider_failure.retryable is None
                    ),
                )
                if conclusive_failure and provider_failure is not None
                else None
            )
            event = Event(
                type=EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
                session_id=binding.session_id,
                interaction_id=binding.interaction_id,
                agent_name=binding.agent_name,
                environment_name=binding.environment_name,
                payload={
                    **{
                        key: stage.intent[key]
                        for key in (
                            "model_step_id",
                            "model_attempt_id",
                            "auxiliary_inference",
                            "provider_name",
                            "requested_model",
                            "execution_profile_fingerprint",
                        )
                    },
                    "provider": provider.billing_provider_name or binding.provider_name,
                    "model": binding.model,
                    "auxiliary_outcome": outcome,
                    "attempt": stage.dispatch_ordinal + 1,
                    "usage_status": usage_status,
                    "usage_metrics": usage_metrics_payload(metrics),
                    **(
                        {"retry_decision": decision.model_dump(mode="json")}
                        if decision is not None
                        else {}
                    ),
                    **(
                        {
                            "provider_error": {
                                "error": str(provider_failure),
                                **provider_failure.error_payload_fields(),
                            }
                        }
                        if provider_failure is not None
                        else {}
                    ),
                },
            )
            try:
                event = event_with_runtime_payload_authority(
                    event, "model_step_id", "model_attempt_id"
                )
                event = event_with_execution_profile_authority(event, invocation.profile)
                event = event_with_runtime_nested_payload_authority(
                    event, *AUXILIARY_ATTRIBUTION_AUTHORITY_PATHS
                )
                publication = self.prepare_terminal(
                    stage=stage, event=event, reservations=reservations
                )
                # Once a validated terminal completion has been observed, its
                # accounting evidence must win over a later stream-cleanup
                # failure.  Publish it before propagating that secondary
                # failure so cancellation cannot erase accepted provider work.
                if failure is None or conclusive_failure or terminal_completed:
                    if (
                        terminal_completed
                        and task is not None
                        and task.cancelling() > cancellation_baseline
                        and terminal_cancellation is None
                    ):
                        terminal_cancellation = consume_pending_task_cancellation()
                    published = await self.publish_terminal(
                        stage=stage, publication=publication, expected_run_epoch=binding.run_epoch
                    )
                else:
                    # A failed wait does not establish provider quiescence. Record
                    # accounting but keep the stage active for explicit recovery.
                    await self._session_store.complete_model_completion_stage(
                        binding.session_id, stage_id=stage.stage_id, publication=publication
                    )
                    await self._run_limit_controller.reconcile_model_completion_settlements(
                        publication.events[0], reservation_ids=stage.reservation_ids
                    )
            except BaseException as settlement_failure:
                settlement_failure = classify_cancellation(settlement_failure)
                if failure is None:
                    raise settlement_failure
                if isinstance(failure, asyncio.CancelledError):
                    prior = exception_cause(failure)
                    if prior is not None:
                        raise failure from BaseExceptionGroup(
                            "Auxiliary provider error and terminal accounting failed",
                            [prior, settlement_failure],
                        )
                    raise failure from settlement_failure
                raise BaseExceptionGroup(
                    "Auxiliary inference and terminal accounting failed",
                    [failure, settlement_failure],
                ) from None
            # A provider may consume cancellation and still yield a terminal
            # completion.  Terminal evidence is authoritative and must be
            # committed first, but the caller's interruption remains
            # authoritative for the operation's result.
            if failure is None and task is not None and task.cancelling() > cancellation_baseline:
                raise asyncio.CancelledError("Auxiliary inference cancelled")
            if failure is None and terminal_cancellation is not None:
                raise terminal_cancellation
            if conclusive_failure:
                assert provider_failure is not None
                return AuxiliaryAttemptResult(provider_failure, (started, *published), decision)
            if failure is not None:
                raise failure
            assert response is not None
            return AuxiliaryAttemptResult(response, (started, *published))
        finally:
            admission.close()
            if cleanup is not None and not transferred:
                release_provider_stream_cleanup(cleanup)

    def prepare_terminal(
        self,
        *,
        stage: ModelCompletionStage,
        event: Event,
        reservations: tuple[BudgetStepReservation, ...],
    ) -> RuntimePublicationRequest:
        """Freeze exact terminal accounting without changing the parent transcript."""

        if type(stage) is not ModelCompletionStage or stage.purpose != "auxiliary-inference":
            raise ValueError("Auxiliary completion requires its exact model stage.")
        if (
            type(event) is not Event
            or event.type is not EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
            or event.session_id != stage.session_id
        ):
            raise ValueError("Auxiliary completion requires its matching terminal event.")
        if (
            tuple(reservation.record.reservation_id for reservation in reservations)
            != stage.reservation_ids
        ):
            raise ValueError("Auxiliary completion lost its prepared reservation identities.")
        prepared = _model_completion_with_budget_settlement_evidence(
            event,
            reservations,
            prepare_event=self._event_writer.prepare,
            settled_at=self._clock(),
        )
        # Fail before writes; the store repeats validation at the atomic boundary.
        from cayu.runtime._auxiliary_inference_contract import auxiliary_terminal_publication

        return auxiliary_terminal_publication(stage, prepared)

    async def publish_terminal(
        self,
        *,
        stage: ModelCompletionStage,
        publication: RuntimePublicationRequest,
        expected_run_epoch: int,
    ) -> list[Event]:
        """Persist terminal evidence, settle budgets, then release the active stage.

        On any exception the caller retains the original stage and publication.
        In particular a cancelled waiter or lost acknowledgement never authorizes
        another provider dispatch. Recovery can replay the same publication or
        reconstruct the committed terminal from the store-owned stage.
        """

        if type(stage) is not ModelCompletionStage or stage.purpose != "auxiliary-inference":
            raise ValueError("Auxiliary completion requires its exact model stage.")
        from cayu.runtime._auxiliary_inference_contract import validate_auxiliary_publication

        validate_auxiliary_publication(publication, session_id=stage.session_id, stage=stage)
        result = await self._session_store.complete_model_completion_stage(
            stage.session_id, stage_id=stage.stage_id, publication=publication
        )
        completed = result.stage
        if completed.publication is None:
            raise RuntimeError("Completed auxiliary stage has no durable publication.")
        event = completed.publication.events[0]
        settlement_events = await self._run_limit_controller.reconcile_model_completion_settlements(
            event, reservation_ids=completed.reservation_ids
        )
        await self._session_store.promote_model_completion_stage(
            completed.session_id,
            stage_id=completed.stage_id,
            expected_run_epoch=expected_run_epoch,
        )
        terminal_events = await self._event_writer.fan_out_persisted([event])
        return [*settlement_events, *terminal_events]
