"""Live model-attempt streaming, cancellation cleanup and completion publication."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import replace
from datetime import datetime
from hashlib import sha256
from typing import Any, cast

from cayu._exception_groups import (
    add_exception_note_safely,
    exception_cause,
    exception_tree_contains,
    iter_exception_tree,
    set_exception_cause,
)
from cayu._task_wait import unexpected_child_cancellation_error
from cayu._validation import (
    DurableValueError,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_json_value,
    extract_durable_value_error,
    safe_durable_value_error_details,
)
from cayu.budgets.billing import BillingIdentity
from cayu.context.base import (
    ContextPressureEstimate,
    context_input_coverage,
    copy_context_pressure_estimate,
)
from cayu.context.footprints import analyze_request_context_pressure
from cayu.context.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
    StructuredOutputStrategy,
)
from cayu.environments.admission import ExecutionAdmissionError
from cayu.events import Event, EventType, copy_event, event_with_runtime_generated_id
from cayu.execution_profiles import ExecutionProfileIdentity, event_with_execution_profile_authority
from cayu.memory.evidence import ContextExposureEvidenceKind, ContextExposureState
from cayu.messages import ProviderStatePart
from cayu.providers._credential_boundary import (
    aclosing_provider_stream,
    credential_safe_provider_cancellation,
    detach_credential_safe_provider_cancellation,
    provider_cancellation_admission_deadline,
    provider_cancellation_failures,
    stream_cleanup_cancelled_after_provider_failure,
)
from cayu.providers._http import (
    bind_provider_error_workload_redactor,
    reset_provider_error_workload_redactor,
)
from cayu.providers._stream_cleanup import _LocalHttpCleanupObserver
from cayu.providers.base import (
    EXACT_MODEL_STREAM_RECOVERY_DISPOSITION,
    MANUAL_MODEL_STREAM_RECOVERY_DISPOSITION,
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamDeadlineError,
    ModelStreamEvent,
    ModelStreamEventType,
)
from cayu.providers.deadlines import ProviderStreamDeadlineAdmission, ProviderStreamDeadlineEvidence
from cayu.providers.operations import (
    ProviderOperationAdapter,
    ProviderOperationMode,
    ProviderOperationSnapshot,
    ProviderOperationState,
    ProviderOperationStatus,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._child_session_notifications import ChildSessionNotificationStageBinding
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._memory_evidence import MemoryEvidenceReference, transition_context_exposure
from cayu.runtime._model_completion_contracts import (
    ModelCompletionDispatch,
    ModelCompletionDispatchPreparer,
    ModelCompletionPublicationRequest,
    ModelCompletionPublisher,
    _copy_assistant_step_result,
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._model_completion_delivery import (
    ModelAttemptFailed,
    _combine_authoritative_model_failure,
    _combine_post_completion_failures,
    _durable_assistant_step_result,
    _non_turn_model_completion_event,
    _publish_model_completion,
    _take_model_completion_cancellation,
)
from cayu.runtime._model_errors import (
    copy_provider_exception_control,
    nonportable_model_provider_error,
    resolve_completion_billing_identity,
    runtime_owned_model_stream_error_event,
)
from cayu.runtime._model_event_authority import _event_with_model_identity_authority
from cayu.runtime._model_execution_selection import ModelExecutionSelection
from cayu.runtime._model_stream_events import (
    _assistant_step_result,
    _citation_part,
    _hosted_tool_call_part,
    _model_stream_event_to_runtime_event,
    _provider_operation_generated_tool_call_id,
    _require_unique_tool_call_ids,
    _retry_attempt_payload,
    _stream_event_completion,
    _validate_assistant_stream_event,
    _validate_stream_event,
)
from cayu.runtime._model_tool_discovery import _hosted_tool_discovery_publication_authority
from cayu.runtime._provider_cleanup_evidence import local_http_cleanup_event_id
from cayu.runtime._provider_operation_cancellation_owner import ProviderOperationCancellationOwner
from cayu.runtime._provider_operation_recovery_owner import (
    ProviderOperationRecoveryOwner,
    _provider_operation_progress_contains_secret,
)
from cayu.runtime._provider_operation_start_owner import (
    ProviderOperationStartOwner,
    ProviderOperationStartState,
)
from cayu.runtime._provider_stream import (
    _admitted_model_provider_events,
    _owned_model_provider_events,
    _provider_stream_self_cancellation_error,
    _ProviderStreamSelfCancellation,
)
from cayu.runtime._run_limits import SessionUsageTracker
from cayu.runtime._session_control import SessionControl, SessionInterruptedByRequest
from cayu.runtime._structured_output_tool_round import (
    _redact_structured_output_validation,
    _validate_structured_output_tool_round,
)
from cayu.runtime.execution_units import ModelAttemptIdentity, copy_model_attempt_identity
from cayu.runtime.model_steps import AssistantStepResult, classify_assistant_step
from cayu.runtime.provider_operations import (
    ProviderOperationEvidenceError,
    ProviderOperationRecoveryStatus,
    load_recoverable_provider_operation,
    provider_operation_progress_envelope,
)
from cayu.runtime.retry_policy import (
    RetryDecision,
    RetryPolicy,
    RetrySuppression,
    copy_retry_policy,
    retry_decision,
)
from cayu.sessions.base import RuntimePublicationOperationRecordMutation, Session, SessionStore
from cayu.tools.exposure import ResolvedToolExposure, resolved_tool_exposure_authority
from cayu.tools.gateway import TargetedToolGatewayProjection
from cayu.vaults.redaction import SecretRedactor


def _provider_failure_proves_no_model_effect(failure: BaseException) -> bool:
    """Return whether typed provider evidence proves dispatch was rejected pre-effect."""

    authentication_rejection_observed = False
    for candidate in iter_exception_tree(failure):
        if isinstance(candidate, BaseExceptionGroup):
            continue
        attempt_failure = (
            candidate if isinstance(candidate, ModelAttemptFailed) else exception_cause(candidate)
        )
        if isinstance(attempt_failure, ModelAttemptFailed):
            if attempt_failure.provider_effect_observed:
                return False
            provider_failure = (
                attempt_failure.cause if isinstance(candidate, ModelAttemptFailed) else candidate
            )
        else:
            provider_failure = candidate
        if not isinstance(provider_failure, ModelProviderError):
            return False
        if not (
            provider_failure.status_code == 401
            or provider_failure.error_type == "authentication_error"
            or provider_failure.error_code in {"authentication_error", "invalid_api_key"}
        ):
            return False
        authentication_rejection_observed = True
    return authentication_rejection_observed


def _assistant_step_result_with_published_targeted_authority(
    live_result: AssistantStepResult,
    published_result: AssistantStepResult,
) -> AssistantStepResult:
    """Pair live validation values with exact published targeted-tool authority."""

    live = _copy_assistant_step_result(live_result)
    published = _copy_assistant_step_result(published_result)
    if (
        live.session_id != published.session_id
        or live.step != published.step
        or live.model_step_id != published.model_step_id
        or live.model_attempt_id != published.model_attempt_id
        or live.tool_round_identity != published.tool_round_identity
        or len(live.tool_calls) != len(published.tool_calls)
    ):
        raise RuntimeError("Published model completion conflicts with its live result.")
    merged_calls: list[runtime_records.ToolCallRequest] = []
    for live_call, published_call in zip(live.tool_calls, published.tool_calls, strict=True):
        if live_call.id != published_call.id or live_call.name != published_call.name:
            raise RuntimeError("Published model tool calls conflict with their live result.")
        merged_calls.append(
            runtime_records.copy_tool_call_request(
                published_call,
                arguments=live_call.arguments,
            )
        )
    return AssistantStepResult(
        session_id=live.session_id,
        step=live.step,
        model_step_id=live.model_step_id,
        model_attempt_id=live.model_attempt_id,
        tool_round_identity=live.tool_round_identity,
        assistant_message=published.assistant_message,
        tool_calls=merged_calls,
        completion=live.completion,
        text_content=live.text_content,
        has_user_visible_content=live.has_user_visible_content,
        provider_state_count=live.provider_state_count,
        thinking_count=live.thinking_count,
    )


def _model_request_fingerprint(
    *,
    provider_name: str,
    model_request: ModelRequest,
) -> str:
    material = canonical_durable_json_bytes(
        {
            "schema_version": 1,
            "provider_name": provider_name,
            "request": model_request.model_dump(mode="json"),
        },
        "model_request_fingerprint",
    )
    return sha256(material).hexdigest()


def _deadline_with_runtime_recovery_authority(
    error: ModelStreamDeadlineError,
    *,
    exact_operation: bool,
) -> ModelStreamDeadlineError:
    """Project recovery disposition from runtime-owned durable operation state."""

    disposition = (
        EXACT_MODEL_STREAM_RECOVERY_DISPOSITION
        if exact_operation
        else MANUAL_MODEL_STREAM_RECOVERY_DISPOSITION
    )
    if error.recovery_disposition == disposition:
        return error
    return ModelStreamDeadlineError(
        provider=error.provider,
        evidence=error.deadline_evidence,
        stream_cleanup_failed=error.stream_cleanup_failed,
        recovery_disposition=disposition,
    )


def _model_context_overflow_error_event(
    error: ModelContextOverflowError,
    *,
    session: Session,
    provider_name: str,
    requested_model: str,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    step: int,
    attempt: int,
    max_attempts: int,
    model_attempt_identity: ModelAttemptIdentity,
) -> Event:
    """Terminalize one dispatched context-overflow attempt without retrying it."""

    if type(error) is not ModelContextOverflowError:
        raise TypeError("Context-overflow terminalization requires a runtime-owned error.")
    payload = {
        "error": str(error),
        "error_type": type(error).__name__,
        "stage": "provider_dispatch",
        "context_overflow": True,
        **ModelProviderError.error_payload_fields(error),
    }
    return Event(
        type=EventType.MODEL_ERROR,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=_retry_attempt_payload(
            payload,
            execution_provider_name=provider_name,
            requested_model=requested_model,
            step=step,
            attempt=attempt,
            max_attempts=max_attempts,
            model_attempt_identity=model_attempt_identity,
        ),
    )


class LiveModelAttempt:
    """Own one live provider attempt through stream cleanup and outcome publication.

    Retry and failover scheduling remain with the caller. Startup, durable
    cancellation and recovery compose their existing owners; run-local authority
    callbacks retain the same dispatch and publication boundaries.
    """

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        session_control: SessionControl[SessionUsageTracker],
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        provider_operation_start: ProviderOperationStartOwner,
        provider_operation_cancellation: ProviderOperationCancellationOwner,
        provider_operation_recovery: ProviderOperationRecoveryOwner,
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._session_control = session_control
        self._secret_redactor = secret_redactor
        self._clock = clock
        self._provider_operation_start = provider_operation_start
        self._provider_operation_cancellation = provider_operation_cancellation
        self._provider_operation_recovery = provider_operation_recovery

    async def execute(
        self,
        *,
        provider: ModelProvider,
        deadline_admission: ProviderStreamDeadlineAdmission,
        model_request: ModelRequest,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
        retry_policy: RetryPolicy,
        model_attempt_identity: ModelAttemptIdentity,
        transcript_cursor_before_request: int,
        record_model_completion: Callable[[Event], Event],
        before_provider_dispatch: Callable[[ModelAttemptIdentity], Awaitable[None]],
        validate_live_model_semantics: Callable[[], None],
        refresh_live_model_semantics: Callable[[], Awaitable[None]],
        billing_identity: BillingIdentity | None,
        structured_output: StructuredOutputSpec | None,
        context_pressure_estimate: ContextPressureEstimate | None,
        prepare_model_completion_dispatch: ModelCompletionDispatchPreparer | None,
        model_completion_publisher: ModelCompletionPublisher | None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None,
        model_execution_selection: ModelExecutionSelection | None,
        tool_exposure: ResolvedToolExposure | None,
        targeted_tool_gateway: TargetedToolGatewayProjection | None,
        native_tool_grant_ids: Mapping[str, str],
        memory_evidence_reference: MemoryEvidenceReference | None,
        child_session_notification_binding: ChildSessionNotificationStageBinding | None,
        prepared_model_completion_dispatch: ModelCompletionDispatch | None,
    ) -> AsyncIterator[tuple[Event | None, AssistantStepResult | None]]:
        input_coverage = context_input_coverage(
            model_request.messages, transcript_cursor=transcript_cursor_before_request
        )
        retry_policy = copy_retry_policy(retry_policy)
        if type(deadline_admission) is not ProviderStreamDeadlineAdmission:
            raise TypeError("deadline_admission must be ProviderStreamDeadlineAdmission.")
        targeted_tool_reference_grant_ids = (
            None if targeted_tool_gateway is None else targeted_tool_gateway.reference_grant_ids()
        )
        native_tool_name_grant_ids = dict(native_tool_grant_ids)
        if retry_policy.max_attempts != max_attempts:
            raise ValueError("Retry policy does not match the model-attempt ceiling.")
        assistant_parts: list[transcript_helpers.AssistantContentPart] = []
        thinking_options = model_request.options.get("thinking")
        include_thinking_in_transcript = (
            thinking_options.get("include_in_transcript", True)
            if isinstance(thinking_options, dict)
            else True
        )
        tool_calls: list[runtime_records.ToolCallRequest] = []
        provider_state_parts: list[ProviderStatePart] = []
        hosted_discovery_mutations: tuple[RuntimePublicationOperationRecordMutation, ...] = ()
        hosted_discovery_grant_ids: dict[str, str] = {}
        completed_stream_event: ModelStreamEvent | None = None
        step_result: AssistantStepResult | None = None
        completion_event: Event | None = None
        completion_dispatch = (
            None
            if prepared_model_completion_dispatch is None
            else ModelCompletionDispatch(
                stage=prepared_model_completion_dispatch.stage,
                request_fingerprint=prepared_model_completion_dispatch.request_fingerprint,
                context_exposure=prepared_model_completion_dispatch.context_exposure,
                child_session_notifications_consumed=(
                    prepared_model_completion_dispatch.child_session_notifications_consumed
                ),
            )
        )
        model_completed = False
        # Request analysis invokes provider-owned projection hooks.  A mutable
        # built-in adapter must still match the profile admitted for this
        # invocation before any of those semantics are consulted.
        await refresh_live_model_semantics()
        context_pressure_estimate = copy_context_pressure_estimate(context_pressure_estimate)
        if context_pressure_estimate is None:
            context_pressure_estimate = analyze_request_context_pressure(
                model_request,
                provider=registered_provider.provider,
            )
        interrupt_poll = self._session_control.stream_interrupt_poll(session.id)
        if (prepare_model_completion_dispatch is None) != (model_completion_publisher is None):
            raise RuntimeError(
                "Model completion staging and publication must be configured together."
            )
        # Deliver a cancellation already pending before the dispatch boundary.
        # A request count retained after this checkpoint is historical; only a
        # later generation can classify provider or cleanup failure as caller
        # cancellation.
        await asyncio.sleep(0)
        current_task = asyncio.current_task()
        provider_cancellation_baseline = 0 if current_task is None else current_task.cancelling()
        # This is the final in-method accounting boundary. A memory-bearing
        # request may already have crossed the same durable fence before an
        # optional provider-backed token count; in that case reuse its dispatch.
        model_attempt_identity = copy_model_attempt_identity(model_attempt_identity)
        if completion_dispatch is not None:
            expected_request_fingerprint = _model_request_fingerprint(
                provider_name=registered_provider.name,
                model_request=model_request,
            )
            if completion_dispatch.request_fingerprint != expected_request_fingerprint:
                raise RuntimeError(
                    "Prepared model-completion dispatch does not match the provider request."
                )
        elif prepare_model_completion_dispatch is None:
            await before_provider_dispatch(model_attempt_identity)
        else:
            completion_dispatch = await prepare_model_completion_dispatch(
                model_request,
                memory_evidence_reference,
                child_session_notification_binding,
                True,
            )
            for prepared_event in completion_dispatch.prepared_events:
                yield prepared_event, None
        # Durable dispatch admission and interruption-decision election share
        # one store-owned fence. If dispatch won that fence, interruption may
        # move the session to INTERRUPTING without yet electing a terminal
        # outcome. Re-read that authority before entering provider-owned code
        # so a dispatch that was only durably prepared cannot start after the
        # interruption request became authoritative.
        await self._session_control.raise_if_interrupted(session.id)
        context_exposure = (
            None if completion_dispatch is None else completion_dispatch.context_exposure
        )
        stream_deadline_failure: ModelStreamDeadlineError | None = None

        async def advance_context_exposure(
            state: ContextExposureState,
            evidence_kind: ContextExposureEvidenceKind,
            evidence_ref: str,
            *,
            provider_request_id: str | None = None,
        ) -> None:
            nonlocal context_exposure
            if context_exposure is None or context_exposure.state.terminal:
                return
            if context_exposure.state is state:
                return
            context_exposure = await transition_context_exposure(
                store=self._session_store,
                exposure=context_exposure,
                state=state,
                evidence_kind=evidence_kind,
                evidence_ref=evidence_ref,
                provider_request_id=provider_request_id,
            )

        def context_exposure_ref(suffix: str) -> str:
            stage_ref = (
                model_attempt_identity.model_attempt_id
                if completion_dispatch is None
                else completion_dispatch.stage_id
            )
            return f"model-stage:{stage_ref}:{suffix}"

        async def advance_context_exposure_preserving_deadline(
            state: ContextExposureState,
            evidence_kind: ContextExposureEvidenceKind,
            evidence_ref: str,
            *,
            authoritative_failure: BaseException | None = None,
            provider_request_id: str | None = None,
        ) -> None:
            try:
                await advance_context_exposure(
                    state,
                    evidence_kind,
                    evidence_ref,
                    provider_request_id=provider_request_id,
                )
            except Exception as evidence_failure:
                authoritative = (
                    stream_deadline_failure
                    if authoritative_failure is None
                    else authoritative_failure
                )
                if authoritative is None:
                    raise
                raise _combine_authoritative_model_failure(
                    authoritative,
                    evidence_failure,
                    message=("Model stream deadline and context-exposure persistence both failed."),
                ) from None

        async def consume_child_session_notifications() -> None:
            nonlocal completion_dispatch
            if (
                completion_dispatch is None
                or completion_dispatch.child_session_notifications_consumed
            ):
                return
            await self._session_store.mark_model_completion_stage_dispatched(
                session.id,
                stage=completion_dispatch.stage,
                consume_child_session_notifications=True,
            )
            completion_dispatch = ModelCompletionDispatch(
                stage=completion_dispatch.stage,
                request_fingerprint=completion_dispatch.request_fingerprint,
                context_exposure=completion_dispatch.context_exposure,
                child_session_notifications_consumed=True,
            )

        async def refresh_context_exposure() -> None:
            nonlocal context_exposure
            if context_exposure is None:
                return
            durable = await self._session_store.load_context_exposure(
                context_exposure.session_id,
                context_exposure.exposure_id,
            )
            if durable is None:
                raise RuntimeError("Provider dispatch lost its context exposure evidence.")
            context_exposure = durable

        async def record_provider_cancellation_outcome(
            snapshot: ProviderOperationSnapshot | None,
            *,
            ambiguous_suffix: str,
        ) -> None:
            if snapshot is not None and snapshot.status is ProviderOperationStatus.CANCELLED:
                await advance_context_exposure(
                    ContextExposureState.CANCELLED,
                    ContextExposureEvidenceKind.CONCLUSIVE_CANCELLATION,
                    context_exposure_ref("provider-cancelled"),
                )
            elif snapshot is not None and snapshot.status in {
                ProviderOperationStatus.FAILED,
                ProviderOperationStatus.EXPIRED,
            }:
                await advance_context_exposure(
                    ContextExposureState.FAILED,
                    ContextExposureEvidenceKind.CONCLUSIVE_FAILURE,
                    context_exposure_ref(f"provider-{snapshot.status.value}"),
                )
            elif snapshot is None or snapshot.status is not ProviderOperationStatus.COMPLETED:
                await advance_context_exposure(
                    ContextExposureState.INDETERMINATE,
                    ContextExposureEvidenceKind.AMBIGUOUS_TRANSPORT,
                    context_exposure_ref(ambiguous_suffix),
                )

        provider_events: AsyncIterator[ModelStreamEvent] | None = None
        provider_events_owned = False
        provider_operation_adapter: ProviderOperationAdapter | None = None
        provider_operation_state: ProviderOperationState | None = None
        provider_operation_interaction_id: str | None = None
        provider_operation_identity_durable = False
        provider_exhausted = False
        background_dispatch_invoked = False
        durable_stream_failure: ModelAttemptFailed | None = None
        provider_control_failure: ModelProviderError | None = None
        provider_control_error_emitted = False
        provider_effect_observed = False
        post_completion_failure: BaseException | None = None

        def background_exposure_recovery_pending() -> bool:
            return bool(
                provider_operation_state is not None
                and provider_operation_identity_durable
                and not model_completed
            )

        async def reconcile_completion_that_won_cancellation(
            snapshot: ProviderOperationSnapshot | None,
        ) -> None:
            if snapshot is None or snapshot.status is not ProviderOperationStatus.COMPLETED:
                return
            if completion_dispatch is None or model_completion_publisher is None:
                raise RuntimeError(
                    "Provider completion won cancellation without a durable publication stage."
                )
            recoverable = await load_recoverable_provider_operation(
                self._session_store,
                completion_dispatch.stage,
            )
            if recoverable is None:
                raise ProviderOperationEvidenceError(
                    "Provider completion won cancellation without recoverable operation evidence."
                )
            recovered = await self._provider_operation_recovery.recover(
                session=session,
                stage=completion_dispatch.stage,
                operation=recoverable,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                environment_name=environment_name,
                recovery_context=model_completion_recovery_context_from_stage(
                    completion_dispatch.stage
                ),
                model_completion_publisher=model_completion_publisher,
                invocation_context=invocation_context,
                model_execution_selection=model_execution_selection,
            )
            if recovered.status is not ProviderOperationRecoveryStatus.RECONCILED:
                raise ProviderOperationEvidenceError(
                    "Provider completion won cancellation but remained unreconciled."
                )

        try:
            validate_live_model_semantics()
            provider_operation_mode = provider.provider_operation_mode
            if type(provider_operation_mode) is not ProviderOperationMode:
                raise TypeError(
                    "ModelProvider.provider_operation_mode must return a ProviderOperationMode."
                )
            if provider_operation_mode is ProviderOperationMode.SYNCHRONOUS:
                await consume_child_session_notifications()
                cleanup_observer = None
                if completion_dispatch is not None:
                    cleanup_stage = completion_dispatch.stage
                    cleanup_identity = copy_model_attempt_identity(model_attempt_identity)
                    cleanup_writer = self._event_writer
                    cleanup_session_id = session.id
                    cleanup_agent_name = registered_agent.spec.name
                    cleanup_provider_name = registered_provider.name

                    async def publish_http_cleanup(
                        evidence: ProviderStreamDeadlineEvidence, succeeded: bool
                    ) -> None:
                        event = Event(
                            id=local_http_cleanup_event_id(cleanup_stage.stage_id),
                            type=EventType.MODEL_HTTP_CLEANUP,
                            session_id=cleanup_session_id,
                            agent_name=cleanup_agent_name,
                            environment_name=environment_name,
                            payload={
                                **cleanup_identity.payload(),
                                **evidence.payload(),
                                "source_run_epoch": cleanup_stage.source_run_epoch,
                                "provider": cleanup_provider_name,
                                "local_http_cleanup": "succeeded" if succeeded else "failed",
                                "provider_effect_outcome": "unknown",
                            },
                        )
                        event = _event_with_model_identity_authority(event, cleanup_identity)
                        event = event_with_execution_profile_authority(event, execution_profile)
                        # Retained close ownership also bounds this write. Persist
                        # without calling user sinks from a transport finalizer.
                        await cleanup_writer.persist(event_with_runtime_generated_id(event))

                    cleanup_observer = _LocalHttpCleanupObserver(publish_http_cleanup)
                provider_events = _owned_model_provider_events(
                    lambda: _admitted_model_provider_events(
                        provider,
                        model_request,
                        deadline_admission,
                        refresh_live_model_semantics,
                        cleanup_observer,
                        error_redactor=self._secret_redactor,
                    ),
                    cancellation_baseline=provider_cancellation_baseline,
                    max_concurrent_streams=deadline_admission.max_concurrent_streams,
                )
                provider_events_owned = True
            else:
                start_progress = ProviderOperationStartState()
                try:
                    async with contextlib.aclosing(
                        self._provider_operation_start.start(
                            progress=start_progress,
                            provider=provider,
                            model_request=model_request,
                            deadline_admission=deadline_admission,
                            completion_dispatch=completion_dispatch,
                            session=session,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                            execution_profile=execution_profile,
                            refresh_live_model_semantics=refresh_live_model_semantics,
                            consume_child_session_notifications=consume_child_session_notifications,
                            acknowledge_context_exposure=lambda operation_id: (
                                advance_context_exposure(
                                    ContextExposureState.ACKNOWLEDGED,
                                    ContextExposureEvidenceKind.PROVIDER_ACKNOWLEDGEMENT,
                                    context_exposure_ref("provider-operation-started"),
                                    provider_request_id=operation_id,
                                )
                            ),
                        )
                    ) as starting:
                        async for event in starting:
                            yield event, None
                finally:
                    # Startup may have published identity or acquired a stream before it
                    # fails. Hand those exact effects to the existing live cleanup paths.
                    provider_operation_adapter = start_progress.adapter
                    provider_operation_state = start_progress.operation_state
                    provider_operation_interaction_id = start_progress.interaction_id
                    provider_events = start_progress.events
                    provider_operation_identity_durable = start_progress.identity_durable
                    background_dispatch_invoked = start_progress.dispatch_invoked
            if provider_events is None:  # pragma: no cover - startup invariant
                raise AssertionError("Provider startup completed without an event stream.")
            provider_iterator = aiter(provider_events)
            while True:
                redactor_token = bind_provider_error_workload_redactor(self._secret_redactor)
                try:
                    try:
                        raw_stream_event = await anext(provider_iterator)
                    except StopAsyncIteration:
                        break
                finally:
                    reset_provider_error_workload_redactor(redactor_token)
                boundary_value = _validate_stream_event(
                    raw_stream_event,
                    provider_name=registered_provider.name,
                    requested_model=model_request.model,
                    usage_dialect=registered_provider.usage_dialect,
                )
                generated_tool_call_id = None
                if provider_operation_state is not None and completion_dispatch is not None:
                    generated_tool_call_id = _provider_operation_generated_tool_call_id(
                        completion_dispatch.stage,
                        boundary_value.event,
                    )
                assistant_boundary = _validate_assistant_stream_event(
                    boundary_value.event,
                    generated_tool_call_id=generated_tool_call_id,
                )
                stream_event = assistant_boundary.event
                if stream_event.type is not ModelStreamEventType.ERROR:
                    provider_effect_observed = True
                    if (
                        context_exposure is not None
                        and context_exposure.state is ContextExposureState.DISPATCH_STARTED
                    ):
                        await advance_context_exposure(
                            ContextExposureState.ACKNOWLEDGED,
                            ContextExposureEvidenceKind.PROVIDER_ACKNOWLEDGEMENT,
                            context_exposure_ref("response"),
                        )
                await interrupt_poll.raise_if_interrupted()
                if model_completed:
                    if (
                        provider_operation_state is not None
                        and completed_stream_event is not None
                        and stream_event == completed_stream_event
                    ):
                        continue
                    message = f"Model provider emitted event after completed: {stream_event.type}"
                    raise ModelAttemptFailed(
                        message=message,
                        payload={"error": message, "error_type": "RuntimeError"},
                        emitted_error_event=False,
                        cause=RuntimeError(message),
                        completion_observed=model_completion_publisher is not None,
                        provider_effect_observed=provider_effect_observed,
                    )

                progress_emitted_event: Event | None = None

                if stream_event.type == ModelStreamEventType.TOOL_CALL:
                    if provider_operation_state is not None:
                        if completion_dispatch is None:  # pragma: no cover - checked at start
                            raise AssertionError("Background operation lost its completion stage.")
                        if provider_operation_interaction_id is None:  # pragma: no cover
                            raise AssertionError("Background operation lost its interaction.")
                        (
                            progress,
                            progress_emitted_event,
                        ) = await self._provider_operation_recovery.commit_stream_event(
                            stage=completion_dispatch.stage,
                            state=provider_operation_state,
                            stream_event=stream_event,
                            runtime_event=None,
                            session=session,
                            interaction_id=provider_operation_interaction_id,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                            targeted_tool_reference_grant_ids=targeted_tool_reference_grant_ids,
                        )
                        provider_operation_state = progress.state
                        if progress.replayed:
                            continue
                    tool_call = assistant_boundary.tool_call
                    tool_call_part = assistant_boundary.tool_call_part
                    if tool_call is None or tool_call_part is None:  # pragma: no cover
                        raise AssertionError("Validated tool-call projection disappeared.")
                    tool_calls.append(tool_call)
                    assistant_parts.append(tool_call_part)
                    if progress_emitted_event is not None:
                        yield progress_emitted_event, None
                    continue

                if stream_event.type in {
                    ModelStreamEventType.HOSTED_TOOL_CALL,
                    ModelStreamEventType.CITATION,
                }:
                    progress_runtime_event = _model_stream_event_to_runtime_event(
                        stream_event,
                        session=session,
                        requested_model=model_request.model,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        provider_name=registered_provider.name,
                        step=step,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        model_attempt_identity=model_attempt_identity,
                        usage_dialect=registered_provider.usage_dialect,
                    )
                    if provider_operation_state is not None:
                        if completion_dispatch is None:  # pragma: no cover - checked at start
                            raise AssertionError("Background operation lost its completion stage.")
                        if provider_operation_interaction_id is None:  # pragma: no cover
                            raise AssertionError("Background operation lost its interaction.")
                        (
                            progress,
                            progress_emitted_event,
                        ) = await self._provider_operation_recovery.commit_stream_event(
                            stage=completion_dispatch.stage,
                            state=provider_operation_state,
                            stream_event=stream_event,
                            runtime_event=progress_runtime_event,
                            session=session,
                            interaction_id=provider_operation_interaction_id,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                            targeted_tool_reference_grant_ids=targeted_tool_reference_grant_ids,
                        )
                        provider_operation_state = progress.state
                        if progress.replayed:
                            continue
                    if stream_event.type is ModelStreamEventType.HOSTED_TOOL_CALL:
                        hosted_part = _hosted_tool_call_part(
                            stream_event,
                            provider_name=registered_provider.name,
                            model=model_request.model,
                            model_attempt_identity=model_attempt_identity,
                        )
                        if hosted_part is not None:
                            assistant_parts.append(hosted_part)
                    else:
                        assistant_parts.append(
                            _citation_part(
                                stream_event,
                                provider_name=registered_provider.name,
                                model_attempt_identity=model_attempt_identity,
                                assistant_parts=assistant_parts,
                            )
                        )
                    emitted_event = (
                        await self._event_writer.emit(progress_runtime_event)
                        if progress_emitted_event is None
                        else progress_emitted_event
                    )
                    yield emitted_event, None
                    continue

                if stream_event.type == ModelStreamEventType.TEXT_DELTA:
                    if provider_operation_state is not None:
                        if completion_dispatch is None:  # pragma: no cover - checked at start
                            raise AssertionError("Background operation lost its completion stage.")
                        if provider_operation_interaction_id is None:  # pragma: no cover
                            raise AssertionError("Background operation lost its interaction.")
                        progress_runtime_event = _model_stream_event_to_runtime_event(
                            stream_event,
                            session=session,
                            requested_model=model_request.model,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            provider_name=registered_provider.name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                            usage_dialect=registered_provider.usage_dialect,
                            execution_profile_fingerprint=(
                                None if execution_profile is None else execution_profile.fingerprint
                            ),
                        )
                        (
                            progress,
                            progress_emitted_event,
                        ) = await self._provider_operation_recovery.commit_stream_event(
                            stage=completion_dispatch.stage,
                            state=provider_operation_state,
                            stream_event=stream_event,
                            runtime_event=progress_runtime_event,
                            session=session,
                            interaction_id=provider_operation_interaction_id,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                            targeted_tool_reference_grant_ids=targeted_tool_reference_grant_ids,
                        )
                        provider_operation_state = progress.state
                        if progress.replayed:
                            continue
                    transcript_helpers.append_assistant_text_delta(
                        assistant_parts,
                        stream_event.delta,
                    )
                elif stream_event.type == ModelStreamEventType.THINKING:
                    if provider_operation_state is not None:
                        if completion_dispatch is None:  # pragma: no cover - checked at start
                            raise AssertionError("Background operation lost its completion stage.")
                        if provider_operation_interaction_id is None:  # pragma: no cover
                            raise AssertionError("Background operation lost its interaction.")
                        progress_runtime_event = (
                            _model_stream_event_to_runtime_event(
                                stream_event,
                                session=session,
                                requested_model=model_request.model,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                provider_name=registered_provider.name,
                                step=step,
                                attempt=attempt,
                                max_attempts=max_attempts,
                                model_attempt_identity=model_attempt_identity,
                                usage_dialect=registered_provider.usage_dialect,
                                execution_profile_fingerprint=(
                                    None
                                    if execution_profile is None
                                    else execution_profile.fingerprint
                                ),
                            )
                            if stream_event.delta
                            else None
                        )
                        (
                            progress,
                            progress_emitted_event,
                        ) = await self._provider_operation_recovery.commit_stream_event(
                            stage=completion_dispatch.stage,
                            state=provider_operation_state,
                            stream_event=stream_event,
                            runtime_event=progress_runtime_event,
                            session=session,
                            interaction_id=provider_operation_interaction_id,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                            targeted_tool_reference_grant_ids=targeted_tool_reference_grant_ids,
                        )
                        provider_operation_state = progress.state
                        if progress.replayed:
                            continue
                    transcript_helpers.append_assistant_thinking_delta(
                        assistant_parts,
                        stream_event.delta,
                        provider_state=stream_event.payload.get("provider_state"),
                        include=include_thinking_in_transcript,
                    )
                    if not stream_event.delta:
                        # Opaque/redacted thinking state belongs in the transcript,
                        # but an empty readable delta should not reach consumers.
                        if progress_emitted_event is not None:
                            yield progress_emitted_event, None
                        continue
                elif stream_event.type == ModelStreamEventType.COMPLETED:
                    terminal_progress_failure: BaseException | None = None
                    terminal_progress_verified = False
                    if provider_operation_state is not None:
                        envelope = provider_operation_progress_envelope(
                            provider_operation_state,
                            stream_event,
                        )
                        if _provider_operation_progress_contains_secret(
                            envelope,
                            redactor=self._secret_redactor,
                            targeted_tool_reference_grant_ids=targeted_tool_reference_grant_ids,
                        ):
                            terminal_progress_failure = ProviderOperationEvidenceError(
                                "Provider-operation terminal recovery state contains a workload "
                                "secret."
                            )
                        else:
                            terminal_cursor = envelope.recovery_metadata.cursor
                            current_cursor = provider_operation_state.recovery_metadata.cursor
                            current_cursor = -1 if current_cursor is None else current_cursor
                            if terminal_cursor != current_cursor + 1:
                                terminal_progress_failure = ProviderOperationEvidenceError(
                                    "Provider-operation terminal cursor is not the next boundary."
                                )
                            else:
                                terminal_progress_verified = True
                    completion_terminal_error: ModelProviderError | None = None
                    completion_diagnostics: dict[str, Any] = {}
                    if boundary_value.completion_error is not None:
                        code, path = safe_durable_value_error_details(
                            boundary_value.completion_error
                        )
                        completion_terminal_error = ModelProviderError(
                            "Model provider emitted a non-portable completion value.",
                            provider=registered_provider.name,
                            error_type="DurableValueError",
                            error_code="invalid_model_completion_value",
                            retryable=False,
                        )
                        completion_diagnostics = {
                            "completion_outcome": "invalid_metadata",
                            "completion_error": {
                                "error": str(completion_terminal_error),
                                "error_type": type(completion_terminal_error).__name__,
                                "durable_value_error_code": code,
                                "durable_value_path": path,
                                **completion_terminal_error.error_payload_fields(),
                            },
                        }
                    else:
                        try:
                            billing_identity = resolve_completion_billing_identity(
                                provider,
                                billing_identity,
                                copy_durable_json_object(
                                    stream_event.payload,
                                    "completed_payload",
                                ),
                                provider_name=registered_provider.name,
                            )
                        except ModelProviderError as exc:
                            completion_terminal_error = exc
                            completion_diagnostics = {
                                "completion_outcome": "billing_identity_resolution_failed",
                                "completion_error": {
                                    "error": str(completion_terminal_error),
                                    "error_type": type(completion_terminal_error).__name__,
                                    "stage": "billing_identity_for_completion",
                                    **completion_terminal_error.error_payload_fields(),
                                },
                            }
                    model_completed = True
                    completed_stream_event = stream_event
                    await advance_context_exposure(
                        ContextExposureState.COMPLETED,
                        ContextExposureEvidenceKind.PROVIDER_COMPLETION,
                        context_exposure_ref("completed"),
                    )
                    assistant_message = None
                    classification = None
                    if completion_terminal_error is None:
                        try:
                            _require_unique_tool_call_ids(tool_calls)
                            provider_state_parts = transcript_helpers.provider_state_parts(
                                stream_event.payload
                            )
                            assistant_message = transcript_helpers.assistant_message(
                                content_parts=assistant_parts,
                                provider_state_parts=provider_state_parts,
                            )
                        except (TypeError, ValueError):
                            completion_terminal_error = ModelProviderError(
                                "Model provider emitted invalid completion transcript state.",
                                provider=registered_provider.name,
                                error_type="ValueError",
                                error_code="invalid_model_completion_transcript",
                                retryable=False,
                            )
                            completion_diagnostics = {
                                "completion_outcome": "invalid_transcript_state",
                                "completion_error": {
                                    "error": str(completion_terminal_error),
                                    "error_type": type(completion_terminal_error).__name__,
                                    "stage": "completion_transcript_projection",
                                    **completion_terminal_error.error_payload_fields(),
                                },
                            }
                    if completion_terminal_error is None:
                        try:
                            (
                                hosted_discovery_mutations,
                                hosted_discovery_grant_ids,
                            ) = await _hosted_tool_discovery_publication_authority(
                                session_store=self._session_store,
                                session=session,
                                registered_agent=registered_agent,
                                projection=model_request.tool_discovery_projection,
                                stream_event=stream_event,
                                tool_calls=tool_calls,
                                model_step_id=model_attempt_identity.model_step_id,
                                created_at=self._clock(),
                            )
                        except (TypeError, ValueError) as exc:
                            completion_terminal_error = ModelProviderError(
                                "Model provider emitted invalid hosted Tool Search evidence.",
                                provider=registered_provider.name,
                                error_type=type(exc).__name__,
                                error_code="invalid_tool_discovery_projection",
                                retryable=False,
                            )
                            completion_diagnostics = {
                                "completion_outcome": "invalid_tool_discovery_projection",
                                "completion_error": {
                                    "error": str(completion_terminal_error),
                                    "error_type": type(completion_terminal_error).__name__,
                                    "stage": "hosted_tool_discovery_validation",
                                    **completion_terminal_error.error_payload_fields(),
                                },
                            }
                    if completion_terminal_error is None:
                        step_result = _assistant_step_result(
                            session_id=session.id,
                            step=step,
                            model_attempt_identity=model_attempt_identity,
                            assistant_message=assistant_message,
                            tool_calls=tool_calls,
                            completion=_stream_event_completion(completed_stream_event),
                        )
                        classification = classify_assistant_step(step_result)
                    defer_assistant_message = bool(tool_calls)
                    completion_event = _model_stream_event_to_runtime_event(
                        stream_event,
                        session=session,
                        requested_model=model_request.model,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        provider_name=registered_provider.name,
                        step=step,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        model_attempt_identity=model_attempt_identity,
                        tool_round_identity=(
                            step_result.tool_round_identity if step_result is not None else None
                        ),
                        classification=(
                            classification.payload() if classification is not None else None
                        ),
                        context_pressure_estimate=context_pressure_estimate,
                        transcript_cursor_after_completion=(
                            transcript_cursor_before_request
                            + (
                                1
                                if assistant_message is not None and not defer_assistant_message
                                else 0
                            )
                        ),
                        input_coverage=input_coverage,
                        usage_dialect=registered_provider.usage_dialect,
                        billing_identity=billing_identity,
                        accounting_usage_metrics=boundary_value.accounting_usage_metrics,
                        accounting_usage_rejected=boundary_value.accounting_usage_rejected,
                        usage_normalization_failed=(boundary_value.usage_normalization_failed),
                        completion_diagnostics=completion_diagnostics,
                        execution_profile_fingerprint=(
                            None if execution_profile is None else execution_profile.fingerprint
                        ),
                    )
                    if provider_operation_state is not None and terminal_progress_verified:
                        if completion_dispatch is None:  # pragma: no cover - checked at start
                            raise AssertionError("Background operation lost its completion stage.")
                        if provider_operation_interaction_id is None:  # pragma: no cover
                            raise AssertionError("Background operation lost its interaction.")
                        completion_event = self._provider_operation_recovery.progress_event(
                            stage=completion_dispatch.stage,
                            state=provider_operation_state,
                            stream_event=stream_event,
                            runtime_event=completion_event,
                            session=session,
                            interaction_id=provider_operation_interaction_id,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                            targeted_tool_reference_grant_ids=targeted_tool_reference_grant_ids,
                        )
                    if model_completion_publisher is None:
                        completion_event = record_model_completion(completion_event)
                        yield await self._event_writer.emit(completion_event), None
                    if terminal_progress_failure is not None:
                        post_completion_failure = _combine_post_completion_failures(
                            post_completion_failure,
                            terminal_progress_failure,
                        )
                    if completion_terminal_error is not None:
                        if post_completion_failure is None:
                            provider_control_failure = completion_terminal_error
                        else:
                            post_completion_failure = _combine_post_completion_failures(
                                post_completion_failure,
                                completion_terminal_error,
                            )
                        break
                    if provider_operation_state is not None:
                        break
                    continue

                stream_retry_decision: RetryDecision | None = None
                provider_error: ModelProviderError | None = None
                message = ""
                if stream_event.type == ModelStreamEventType.ERROR:
                    untrusted_message = str(
                        stream_event.payload.get("error") or "Model provider error"
                    )
                    stream_event, provider_error = runtime_owned_model_stream_error_event(
                        stream_event,
                        fallback_provider=registered_provider.name,
                        fallback_message=untrusted_message,
                    )
                    message = str(stream_event.payload.get("error") or "Model provider error")
                    if isinstance(provider_error, ModelStreamDeadlineError):
                        provider_error = _deadline_with_runtime_recovery_authority(
                            provider_error,
                            exact_operation=background_exposure_recovery_pending(),
                        )
                        stream_event = ModelStreamEvent.error(
                            str(provider_error),
                            cause=provider_error,
                            recovery_metadata=stream_event.recovery_metadata,
                            provider_operation_status=stream_event.provider_operation_status,
                        )
                        message = str(provider_error)
                        stream_deadline_failure = provider_error
                        if not background_exposure_recovery_pending():
                            await advance_context_exposure_preserving_deadline(
                                ContextExposureState.INDETERMINATE,
                                ContextExposureEvidenceKind.AMBIGUOUS_TRANSPORT,
                                context_exposure_ref("provider-stream-deadline"),
                            )
                    else:
                        await advance_context_exposure(
                            ContextExposureState.FAILED,
                            ContextExposureEvidenceKind.CONCLUSIVE_FAILURE,
                            context_exposure_ref("provider-error"),
                        )
                    if (
                        isinstance(provider_error, ModelContextOverflowError)
                        and provider_operation_state is None
                    ):
                        # Providers may flatten a typed overflow into an error
                        # event. Rehydrate it so bounded recovery can shrink the
                        # request instead of spending generic retries on it.
                        raise provider_error

                    stream_retry_decision = retry_decision(
                        policy=retry_policy,
                        attempt=attempt,
                        error=message,
                        status_code=(
                            provider_error.status_code
                            if isinstance(provider_error, ModelProviderError)
                            else None
                        ),
                        suppression=(
                            RetrySuppression.DEADLINE
                            if isinstance(provider_error, ModelStreamDeadlineError)
                            else RetrySuppression.PROVIDER_OPERATION
                            if provider_operation_state is not None
                            else None
                        ),
                        retryable=(
                            provider_error.retryable
                            if isinstance(provider_error, ModelProviderError)
                            else None
                        ),
                        retry_after_s=(
                            None
                            if provider_operation_state is not None
                            else (
                                provider_error.retry_after_s
                                if isinstance(provider_error, ModelProviderError)
                                else None
                            )
                        ),
                        unknown_provider_error=(
                            provider_operation_state is None
                            and isinstance(provider_error, ModelProviderError)
                            and provider_error.status_code is None
                            and provider_error.retryable is None
                        ),
                    )

                    if provider_operation_state is not None:
                        if completion_dispatch is None:  # pragma: no cover - checked at start
                            raise AssertionError("Background operation lost its completion stage.")
                        if provider_operation_interaction_id is None:  # pragma: no cover
                            raise AssertionError("Background operation lost its interaction.")
                        progress_runtime_event = _model_stream_event_to_runtime_event(
                            stream_event,
                            session=session,
                            requested_model=model_request.model,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            provider_name=registered_provider.name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                            usage_dialect=registered_provider.usage_dialect,
                            execution_profile_fingerprint=(
                                None if execution_profile is None else execution_profile.fingerprint
                            ),
                            retry_decision=stream_retry_decision,
                        )
                        (
                            progress,
                            progress_emitted_event,
                        ) = await self._provider_operation_recovery.commit_stream_event(
                            stage=completion_dispatch.stage,
                            state=provider_operation_state,
                            stream_event=stream_event,
                            runtime_event=progress_runtime_event,
                            session=session,
                            interaction_id=provider_operation_interaction_id,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                            targeted_tool_reference_grant_ids=targeted_tool_reference_grant_ids,
                        )
                        provider_operation_state = progress.state
                        if progress.replayed:
                            continue

                if progress_emitted_event is None:
                    event = _model_stream_event_to_runtime_event(
                        stream_event,
                        session=session,
                        requested_model=model_request.model,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        provider_name=registered_provider.name,
                        step=step,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        model_attempt_identity=model_attempt_identity,
                        usage_dialect=registered_provider.usage_dialect,
                        execution_profile_fingerprint=(
                            None if execution_profile is None else execution_profile.fingerprint
                        ),
                        retry_decision=(
                            stream_retry_decision
                            if stream_event.type == ModelStreamEventType.ERROR
                            else None
                        ),
                    )
                    emitted_event = await self._event_writer.emit(event)
                else:
                    emitted_event = progress_emitted_event
                if stream_event.type == ModelStreamEventType.ERROR:
                    if stream_retry_decision is None:  # pragma: no cover - set above
                        raise AssertionError("Model error lost its retry decision.")
                    yield emitted_event, None
                    if provider_operation_state is not None and isinstance(
                        provider_error, ModelContextOverflowError
                    ):
                        provider_control_failure = provider_error
                        provider_control_error_emitted = True
                        break
                    raise ModelAttemptFailed(
                        message=message,
                        payload=copy_json_value(stream_event.payload, "payload"),
                        emitted_error_event=True,
                        cause=provider_error or RuntimeError(message),
                        retry_decision=stream_retry_decision,
                        provider_effect_observed=provider_effect_observed,
                    )
                yield emitted_event, None
            else:
                provider_exhausted = True

        except SessionInterruptedByRequest as exc:
            if model_completion_publisher is None or not model_completed:
                cancellation_snapshot = None
                if (
                    provider_operation_adapter is not None
                    and provider_operation_state is not None
                    and provider_operation_interaction_id is not None
                    and provider_operation_identity_durable
                    and completion_dispatch is not None
                ):
                    cancellation_snapshot = (
                        await self._provider_operation_cancellation.cancel_started_operation(
                            adapter=provider_operation_adapter,
                            state=provider_operation_state,
                            failure=exc,
                            session=session,
                            stage=completion_dispatch.stage,
                            interaction_id=provider_operation_interaction_id,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                        )
                    )
                    await reconcile_completion_that_won_cancellation(cancellation_snapshot)
                try:
                    await refresh_context_exposure()
                except Exception as evidence_failure:
                    add_exception_note_safely(
                        exc,
                        "Context-exposure cancellation readback failed: "
                        f"{type(evidence_failure).__name__}.",
                    )
                    raise exc from evidence_failure
                await record_provider_cancellation_outcome(
                    cancellation_snapshot,
                    ambiguous_suffix="interrupted",
                )
                raise
            post_completion_failure = exc
        except asyncio.CancelledError as exc:
            current_task = asyncio.current_task()
            caller_cancellation = (
                current_task is not None
                and current_task.cancelling() > provider_cancellation_baseline
            )
            if caller_cancellation:
                # The provider executes inside this task and can replace the
                # injected cancellation with arbitrary text. Task state proves
                # caller cancellation, but not that the returned exception
                # object or its arguments are caller-owned.
                detached_cancellation = detach_credential_safe_provider_cancellation(exc)
                safe_message = (
                    "Provider operation cancelled"
                    if detached_cancellation is None
                    else detached_cancellation.args[0]
                )
                exc = credential_safe_provider_cancellation(
                    safe_message,
                    preserve_empty_artifacts=False,
                    stream_cleanup_cancelled_after_failure=(
                        stream_cleanup_cancelled_after_provider_failure(exc)
                    ),
                    native_admission_deadline=provider_cancellation_admission_deadline(exc),
                    provider_cancellation_failures=tuple(
                        {**failure, **model_attempt_identity.payload()}
                        if "cleanup_diagnostic_version" in failure
                        else failure
                        for failure in provider_cancellation_failures(exc)
                    ),
                )
            elif isinstance(exc, _ProviderStreamSelfCancellation):
                # Only the exact provider iterator boundary can mint this
                # marker. Keep provider-created cancellation on the ordinary,
                # non-retryable provider-failure path without granting it
                # caller cancellation authority.
                provider_failure = _provider_stream_self_cancellation_error(
                    registered_provider.name
                )
                if model_completion_publisher is None or not model_completed:
                    durable_stream_failure = ModelAttemptFailed(
                        message=str(provider_failure),
                        payload={
                            "error": str(provider_failure),
                            "error_type": "ProviderStreamCancellationError",
                            **provider_failure.error_payload_fields(),
                        },
                        emitted_error_event=False,
                        cause=provider_failure,
                        provider_effect_observed=provider_effect_observed,
                        automatic_retry_disabled=True,
                        retry_suppression=RetrySuppression.CANCELLATION,
                    )
                else:
                    post_completion_failure = provider_failure
            else:
                # Provider-originated cancellation is converted at the exact
                # provider iterator boundary. A raw cancellation reaching this
                # wider attempt owner came from another child operation (for
                # example event persistence) and must not be mislabeled as a
                # provider failure or caller control.
                operational_failure = unexpected_child_cancellation_error(
                    exc,
                    operation="Model attempt operation",
                )
                if model_completion_publisher is None or not model_completed:
                    durable_stream_failure = ModelAttemptFailed(
                        message=str(operational_failure),
                        payload={
                            "error": str(operational_failure),
                            "error_type": type(operational_failure).__name__,
                        },
                        emitted_error_event=False,
                        cause=operational_failure,
                        provider_effect_observed=provider_effect_observed,
                        automatic_retry_disabled=True,
                        retry_suppression=RetrySuppression.CANCELLATION,
                    )
                else:
                    post_completion_failure = operational_failure
            if caller_cancellation and (model_completion_publisher is None or not model_completed):
                cancellation_snapshot = None
                if (
                    provider_operation_adapter is not None
                    and provider_operation_state is not None
                    and provider_operation_interaction_id is not None
                    and provider_operation_identity_durable
                    and completion_dispatch is not None
                ):
                    cancellation_snapshot = (
                        await self._provider_operation_cancellation.cancel_started_operation(
                            adapter=provider_operation_adapter,
                            state=provider_operation_state,
                            failure=exc,
                            session=session,
                            stage=completion_dispatch.stage,
                            interaction_id=provider_operation_interaction_id,
                            registered_agent=registered_agent,
                            registered_provider=registered_provider,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                        )
                    )
                    await reconcile_completion_that_won_cancellation(cancellation_snapshot)
                try:
                    await refresh_context_exposure()
                except Exception as evidence_failure:
                    add_exception_note_safely(
                        exc,
                        "Context-exposure cancellation readback failed: "
                        f"{type(evidence_failure).__name__}.",
                    )
                    raise exc from evidence_failure
                await record_provider_cancellation_outcome(
                    cancellation_snapshot,
                    ambiguous_suffix="cancelled-with-unknown-provider-outcome",
                )
                raise exc from None
            if caller_cancellation:
                post_completion_failure = exc
        except GeneratorExit as exc:
            if model_completion_publisher is None or not model_completed:
                if not background_exposure_recovery_pending():
                    await advance_context_exposure(
                        ContextExposureState.INDETERMINATE,
                        ContextExposureEvidenceKind.AMBIGUOUS_TRANSPORT,
                        context_exposure_ref("consumer-abandoned-stream"),
                    )
                raise
            post_completion_failure = exc
        except BaseExceptionGroup as exc:
            grouped_deadline = next(
                (
                    candidate
                    for candidate in iter_exception_tree(exc)
                    if isinstance(candidate, ModelStreamDeadlineError)
                ),
                None,
            )
            if grouped_deadline is not None:
                stream_deadline_failure = grouped_deadline
            if model_completion_publisher is None or not model_completed:
                if not background_exposure_recovery_pending():
                    authentication_rejection = _provider_failure_proves_no_model_effect(exc)
                    if authentication_rejection and not provider_effect_observed:
                        await advance_context_exposure_preserving_deadline(
                            ContextExposureState.FAILED,
                            ContextExposureEvidenceKind.CONCLUSIVE_FAILURE,
                            context_exposure_ref("provider-rejected-before-effect"),
                            authoritative_failure=(exc if grouped_deadline is not None else None),
                        )
                    else:
                        await advance_context_exposure_preserving_deadline(
                            ContextExposureState.INDETERMINATE,
                            ContextExposureEvidenceKind.AMBIGUOUS_TRANSPORT,
                            context_exposure_ref("exception-group-without-provider-outcome"),
                            authoritative_failure=(exc if grouped_deadline is not None else None),
                        )
                    if authentication_rejection and provider_effect_observed:
                        late_failure = ModelProviderError(
                            "A model provider exception group was raised after provider output "
                            "was observed.",
                            provider=registered_provider.name,
                            error_type="ProviderExceptionGroup",
                            error_code="provider_exception_group_after_effect",
                            retryable=False,
                        )
                        set_exception_cause(late_failure, exc)
                        raise ModelAttemptFailed(
                            message=str(late_failure),
                            payload={
                                "error": str(late_failure),
                                "error_type": "ProviderExceptionGroup",
                            },
                            emitted_error_event=False,
                            cause=late_failure,
                            provider_effect_observed=True,
                            automatic_retry_disabled=True,
                            retry_suppression=RetrySuppression.PROVIDER_EFFECT_OBSERVED,
                        ) from exc
                raise
            post_completion_failure = exc
        except ModelAttemptFailed as exc:
            if isinstance(exc.cause, ModelStreamDeadlineError):
                stream_deadline_failure = exc.cause
            if model_completion_publisher is None or not model_completed:
                if not background_exposure_recovery_pending():
                    await advance_context_exposure_preserving_deadline(
                        ContextExposureState.INDETERMINATE,
                        ContextExposureEvidenceKind.AMBIGUOUS_TRANSPORT,
                        context_exposure_ref("attempt-failed-without-terminal-provider-evidence"),
                        authoritative_failure=(
                            exc.cause if isinstance(exc.cause, ModelStreamDeadlineError) else None
                        ),
                    )
                if background_dispatch_invoked and not exc.automatic_retry_disabled:
                    raise ModelAttemptFailed(
                        message=exc.message,
                        payload=exc.payload,
                        emitted_error_event=exc.emitted_error_event,
                        cause=exc.cause,
                        completion_observed=exc.completion_observed,
                        provider_effect_observed=exc.provider_effect_observed,
                        automatic_retry_disabled=True,
                        retry_decision=(
                            exc.retry_decision
                            if exc.retry_decision is not None and not exc.retry_decision.retry
                            else None
                        ),
                        retry_suppression=RetrySuppression.PROVIDER_OPERATION,
                    ) from exc
                raise
            durable_stream_failure = exc
        except Exception as exc:
            if stream_deadline_failure is not None and exc is not stream_deadline_failure:
                raise _combine_authoritative_model_failure(
                    stream_deadline_failure,
                    exc,
                    message="Model stream deadline and diagnostic publication both failed.",
                ) from None
            provider_failure = None
            durable_error = extract_durable_value_error(exc)
            invalid_provider_error = False
            if isinstance(exc, ModelProviderError) or durable_error is None:
                try:
                    provider_failure = copy_provider_exception_control(exc)
                except DurableValueError as portability_error:
                    durable_error = portability_error
                    invalid_provider_error = True
            if provider_failure is not None and isinstance(
                provider_failure.cause,
                ModelStreamDeadlineError,
            ):
                deadline_error = _deadline_with_runtime_recovery_authority(
                    provider_failure.cause,
                    exact_operation=background_exposure_recovery_pending(),
                )
                provider_failure = replace(
                    provider_failure,
                    message=str(deadline_error),
                    error_type=type(deadline_error).__name__,
                    cause=deadline_error,
                )
            elif provider_failure is not None and isinstance(exc, ExecutionAdmissionError):
                # Admission is runtime-owned control, not provider output. Keep
                # its exact settlement handoff and structured decision through
                # MODEL_ERROR publication and terminal environment cleanup.
                provider_failure = replace(
                    provider_failure,
                    message=str(exc),
                    error_type=type(exc).__name__,
                    cause=exc,
                )
            if provider_failure is not None:
                if isinstance(provider_failure.cause, ModelContextOverflowError):
                    provider_control_failure = provider_failure.cause
                else:
                    deadline_failure = isinstance(
                        provider_failure.cause,
                        ModelStreamDeadlineError,
                    )
                    if deadline_failure:
                        stream_deadline_failure = cast(
                            "ModelStreamDeadlineError",
                            provider_failure.cause,
                        )
                    durable_stream_failure = ModelAttemptFailed(
                        message=provider_failure.message,
                        payload={
                            "error": provider_failure.message,
                            "error_type": provider_failure.error_type,
                            **(
                                {"execution_admission": exc.decision.model_dump(mode="json")}
                                if isinstance(exc, ExecutionAdmissionError)
                                else {}
                            ),
                            **(
                                provider_failure.cause.error_payload_fields()
                                if isinstance(provider_failure.cause, ModelProviderError)
                                else {}
                            ),
                        },
                        emitted_error_event=False,
                        cause=provider_failure.cause,
                        completion_observed=(
                            model_completion_publisher is not None and model_completed
                        ),
                        provider_effect_observed=provider_effect_observed,
                        automatic_retry_disabled=(
                            deadline_failure or isinstance(exc, ExecutionAdmissionError)
                        ),
                    )
            elif durable_error is not None:
                if invalid_provider_error:
                    provider_error, durable_diagnostics = nonportable_model_provider_error(
                        durable_error,
                        fallback_provider=registered_provider.name,
                    )
                else:
                    durable_error_code, durable_error_path = safe_durable_value_error_details(
                        durable_error
                    )
                    provider_error = ModelProviderError(
                        "Model provider emitted a non-portable stream value.",
                        provider=registered_provider.name,
                        error_type="DurableValueError",
                        error_code="invalid_model_stream_value",
                        retryable=False,
                    )
                    durable_diagnostics = {
                        "durable_value_error_code": durable_error_code,
                        "durable_value_path": durable_error_path,
                    }
                error_payload = {
                    "error": str(provider_error),
                    "error_type": type(provider_error).__name__,
                    "stage": "model_stream_validation",
                    **durable_diagnostics,
                    **provider_error.error_payload_fields(),
                }
                durable_stream_failure = ModelAttemptFailed(
                    message=str(provider_error),
                    payload=error_payload,
                    emitted_error_event=not (
                        model_completion_publisher is not None and model_completed
                    ),
                    cause=provider_error,
                    completion_observed=(
                        model_completion_publisher is not None and model_completed
                    ),
                    provider_effect_observed=provider_effect_observed,
                    retry_decision=retry_decision(
                        policy=retry_policy,
                        attempt=attempt,
                        error=str(provider_error),
                        retryable=provider_error.retryable,
                        suppression=(
                            RetrySuppression.COMPLETION_OBSERVED
                            if model_completion_publisher is not None and model_completed
                            else RetrySuppression.PROVIDER_OPERATION
                            if background_dispatch_invoked
                            else None
                        ),
                    ),
                )
                if model_completion_publisher is None or not model_completed:
                    yield (
                        await self._event_writer.emit(
                            event_with_execution_profile_authority(
                                Event(
                                    type=EventType.MODEL_ERROR,
                                    session_id=session.id,
                                    agent_name=registered_agent.spec.name,
                                    environment_name=environment_name,
                                    payload=_retry_attempt_payload(
                                        error_payload,
                                        execution_provider_name=registered_provider.name,
                                        requested_model=model_request.model,
                                        step=step,
                                        attempt=attempt,
                                        max_attempts=max_attempts,
                                        model_attempt_identity=model_attempt_identity,
                                        decision=durable_stream_failure.retry_decision,
                                    ),
                                ),
                                execution_profile,
                            )
                        ),
                        None,
                    )
            else:  # pragma: no cover - every Exception has a control or durable failure
                raise RuntimeError("Provider exception handling lost its failure state.") from None
        except BaseException as exc:
            if model_completion_publisher is None or not model_completed:
                if not background_exposure_recovery_pending():
                    await advance_context_exposure(
                        ContextExposureState.INDETERMINATE,
                        ContextExposureEvidenceKind.AMBIGUOUS_TRANSPORT,
                        context_exposure_ref("fatal-stream-failure"),
                    )
                raise
            post_completion_failure = exc
        finally:
            if provider_events is not None and not provider_exhausted:
                active_failure = sys.exception()
                try:
                    if provider_events_owned:
                        # The wrapper owns the one credential-safe provider close.
                        await cast(
                            "AsyncGenerator[ModelStreamEvent, None]",
                            provider_events,
                        ).aclose()
                    else:
                        async with aclosing_provider_stream(provider_events):
                            pass
                except BaseException as cleanup_failure:
                    if isinstance(cleanup_failure, asyncio.CancelledError):
                        raise
                    primary_failure = (
                        post_completion_failure
                        or provider_control_failure
                        or durable_stream_failure
                        or active_failure
                    )
                    combined_failure = (
                        cleanup_failure
                        if primary_failure is None or primary_failure is cleanup_failure
                        else _combine_post_completion_failures(
                            primary_failure,
                            cleanup_failure,
                        )
                    )
                    if background_dispatch_invoked and model_completed:
                        post_completion_failure = combined_failure
                    elif model_completion_publisher is None or not model_completed:
                        raise combined_failure from None
                    else:
                        post_completion_failure = combined_failure

        if (
            not model_completed
            and context_exposure is not None
            and not background_exposure_recovery_pending()
            and (
                provider_control_failure is not None
                or durable_stream_failure is not None
                or post_completion_failure is not None
                or provider_exhausted
            )
        ):
            if isinstance(provider_control_failure, ModelContextOverflowError):
                await advance_context_exposure(
                    ContextExposureState.FAILED,
                    ContextExposureEvidenceKind.CONCLUSIVE_FAILURE,
                    context_exposure_ref("provider-context-overflow"),
                )
            elif durable_stream_failure is not None and _provider_failure_proves_no_model_effect(
                durable_stream_failure
            ):
                await advance_context_exposure(
                    ContextExposureState.FAILED,
                    ContextExposureEvidenceKind.CONCLUSIVE_FAILURE,
                    context_exposure_ref("provider-rejected-before-effect"),
                )
            else:
                await advance_context_exposure_preserving_deadline(
                    ContextExposureState.INDETERMINATE,
                    ContextExposureEvidenceKind.AMBIGUOUS_TRANSPORT,
                    context_exposure_ref("stream-ended-without-completion"),
                )

        if model_completed and model_completion_publisher is not None:
            if completed_stream_event is None:
                raise RuntimeError("Model provider completed without completion metadata.")
            if completion_event is None:
                raise RuntimeError("Model provider completed without a completion event.")
            if completion_dispatch is None:
                raise RuntimeError("Model provider completed without a prepared dispatch.")

            terminal_failure: BaseException | None = (
                post_completion_failure or provider_control_failure or durable_stream_failure
            )
            if terminal_failure is None:
                try:
                    await self._session_control.raise_if_interrupted(session.id)
                except (SessionInterruptedByRequest, asyncio.CancelledError) as exc:
                    terminal_failure = exc
            publication_cancellation = _take_model_completion_cancellation(
                terminal_failure,
                cancellation_baseline=provider_cancellation_baseline,
            )
            if publication_cancellation is not None and terminal_failure is None:
                terminal_failure = publication_cancellation
            elif publication_cancellation is None and isinstance(
                terminal_failure,
                asyncio.CancelledError,
            ):
                terminal_failure = unexpected_child_cancellation_error(
                    terminal_failure,
                    operation="Model provider stream",
                )

            durable_step_result = None
            live_continuation_result = None
            structured_output_validation = None
            if step_result is not None:
                try:
                    overlapping_hosted_grants = (
                        native_tool_name_grant_ids.keys() & hosted_discovery_grant_ids.keys()
                    )
                    if overlapping_hosted_grants:
                        raise ValueError(
                            "Hosted discovery duplicated existing native grant authority."
                        )
                    publication_native_grant_ids = {
                        **native_tool_name_grant_ids,
                        **hosted_discovery_grant_ids,
                    }
                    candidate_step_result = _durable_assistant_step_result(
                        step_result,
                        redactor=self._secret_redactor,
                        targeted_tool_reference_grant_ids=targeted_tool_reference_grant_ids,
                        native_tool_name_grant_ids=publication_native_grant_ids,
                    )
                    candidate_live_continuation = (
                        None
                        if targeted_tool_gateway is None and not publication_native_grant_ids
                        else _assistant_step_result_with_published_targeted_authority(
                            step_result,
                            candidate_step_result,
                        )
                    )
                    candidate_validation = None
                    if (
                        terminal_failure is None
                        and structured_output is not None
                        and structured_output.strategy == StructuredOutputStrategy.TOOL
                        and any(
                            call.name == STRUCTURED_OUTPUT_TOOL_NAME
                            for call in step_result.tool_calls
                        )
                    ):
                        candidate_validation = _redact_structured_output_validation(
                            _validate_structured_output_tool_round(
                                tool_calls=step_result.tool_calls,
                                spec=structured_output,
                            ),
                            self._secret_redactor,
                        )
                    durable_step_result = candidate_step_result
                    live_continuation_result = candidate_live_continuation
                    structured_output_validation = candidate_validation
                except (TypeError, ValueError):
                    if terminal_failure is None:
                        terminal_failure = ModelProviderError(
                            "Model provider emitted assistant output that cannot cross "
                            "the durable publication boundary.",
                            provider=registered_provider.name,
                            error_type="DurableBoundaryError",
                            error_code="invalid_model_completion_transcript",
                            retryable=False,
                        )
            authoritative_assistant_message = (
                durable_step_result.assistant_message
                if durable_step_result is not None and terminal_failure is None
                else None
            )
            defer_assistant_message = bool(
                authoritative_assistant_message is not None
                and durable_step_result is not None
                and durable_step_result.tool_calls
            )
            publication_event = (
                completion_event
                if terminal_failure is None
                else _non_turn_model_completion_event(
                    completion_event,
                    failure=terminal_failure,
                    cancellation=publication_cancellation,
                    transcript_cursor=completion_dispatch.stage.source_transcript_cursor,
                )
            )
            publication_event = record_model_completion(publication_event)
            publication_request = ModelCompletionPublicationRequest(
                dispatch=completion_dispatch,
                assistant_step_result=durable_step_result,
                completion_event=publication_event,
                authoritative_assistant_message=authoritative_assistant_message,
                defer_assistant_message=defer_assistant_message,
                structured_output_validation=structured_output_validation,
                tool_exposure=(
                    None
                    if tool_exposure is None
                    else resolved_tool_exposure_authority(tool_exposure)
                ),
                operation_record_mutations=(
                    () if terminal_failure is not None else hosted_discovery_mutations
                ),
            )
            await _publish_model_completion(
                model_completion_publisher,
                publication_request,
                terminal_failure=terminal_failure,
                publication_cancellation=publication_cancellation,
            )
            if live_continuation_result is not None:
                # Structured-output validation still needs the live values,
                # while subsequent gateway routing needs the exact private
                # authority that crossed the atomic publication boundary.
                step_result = live_continuation_result
            if terminal_failure is None or not exception_tree_contains(
                terminal_failure,
                GeneratorExit,
            ):
                yield copy_event(publication_event), None
            if terminal_failure is not None:
                raise terminal_failure

        if provider_control_failure is not None and background_dispatch_invoked:
            post_dispatch_failure = ModelProviderError(
                "A background provider operation failed after dispatch; automatic retry and "
                "context-overflow recovery are disabled while the original operation may "
                "remain active.",
                provider=registered_provider.name,
                error_type=type(provider_control_failure).__name__,
                error_code="provider_operation_failed_after_dispatch",
                retryable=False,
            )
            set_exception_cause(post_dispatch_failure, provider_control_failure)
            raise ModelAttemptFailed(
                message=str(post_dispatch_failure),
                payload={
                    "error": str(post_dispatch_failure),
                    "error_type": type(provider_control_failure).__name__,
                },
                emitted_error_event=provider_control_error_emitted,
                cause=post_dispatch_failure,
                provider_effect_observed=provider_effect_observed,
                automatic_retry_disabled=True,
                retry_suppression=RetrySuppression.PROVIDER_OPERATION,
            ) from post_dispatch_failure
        if type(provider_control_failure) is ModelContextOverflowError:
            yield (
                await self._event_writer.emit(
                    event_with_execution_profile_authority(
                        _model_context_overflow_error_event(
                            provider_control_failure,
                            session=session,
                            provider_name=registered_provider.name,
                            requested_model=model_request.model,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            step=step,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            model_attempt_identity=model_attempt_identity,
                        ),
                        execution_profile,
                    )
                ),
                None,
            )
        if provider_control_failure is not None:
            if isinstance(provider_control_failure, ModelContextOverflowError):
                raise ModelAttemptFailed(
                    message=str(provider_control_failure),
                    payload=provider_control_failure.error_payload_fields(),
                    emitted_error_event=True,
                    cause=provider_control_failure,
                    completion_observed=model_completed,
                    provider_effect_observed=provider_effect_observed,
                    automatic_retry_disabled=background_dispatch_invoked,
                ) from None
            raise provider_control_failure from None
        if durable_stream_failure is not None:
            if background_dispatch_invoked and not durable_stream_failure.automatic_retry_disabled:
                durable_stream_failure = ModelAttemptFailed(
                    message=durable_stream_failure.message,
                    payload=durable_stream_failure.payload,
                    emitted_error_event=durable_stream_failure.emitted_error_event,
                    cause=durable_stream_failure.cause,
                    completion_observed=durable_stream_failure.completion_observed,
                    provider_effect_observed=(durable_stream_failure.provider_effect_observed),
                    automatic_retry_disabled=True,
                    retry_decision=(
                        durable_stream_failure.retry_decision
                        if durable_stream_failure.retry_decision is not None
                        and not durable_stream_failure.retry_decision.retry
                        else None
                    ),
                    retry_suppression=RetrySuppression.PROVIDER_OPERATION,
                )
            raise durable_stream_failure from None
        if post_completion_failure is not None:
            raise post_completion_failure
        if not model_completed:
            message = "Model provider stream ended without a completed event."
            raise ModelAttemptFailed(
                message=message,
                payload={"error": message, "error_type": "RuntimeError"},
                emitted_error_event=False,
                cause=RuntimeError(message),
                provider_effect_observed=provider_effect_observed,
                automatic_retry_disabled=background_dispatch_invoked,
                retry_suppression=(
                    RetrySuppression.PROVIDER_OPERATION if background_dispatch_invoked else None
                ),
            )
        await self._session_control.raise_if_interrupted(session.id)
        if completed_stream_event is None:
            raise RuntimeError("Model provider completed without completion metadata.")
        if step_result is None:
            raise RuntimeError("Model provider completed without an assistant step result.")
        yield None, step_result
