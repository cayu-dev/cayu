"""Complete context, accounting, retry, and provider-stream model-step ownership.

This module sits below :class:`CayuApp`: it never imports or accepts the
application facade.  The complete executor owns provider-facing request
construction, attachment resolution, context projection and recovery, budget
reservation settlement, retry isolation, and stream normalization. Session-loop
decisions and transcript commits stay with :class:`SessionEngine`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

from cayu._task_wait import (
    await_shielded_task_outcome,
    consume_pending_task_cancellation,
    unexpected_child_cancellation_error,
)
from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.artifacts import (
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    FileAttachment,
    InvalidArtifactIdError,
    copy_artifact_read_result,
    file_attachment_from_payload,
    resolved_file_attachment,
)
from cayu.core.agents import AgentSpec
from cayu.core.billing import (
    BillingIdentity,
    completed_billing_identity,
    copy_billing_identity,
    resolved_billing_identity,
)
from cayu.core.events import Event, EventType
from cayu.core.messages import (
    FilePart,
    Message,
    MessageRole,
    ProviderStatePart,
    ToolCallPart,
    ToolResultPart,
)
from cayu.core.thinking import ThinkingConfig, thinking_config_payload
from cayu.providers import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelCompletion,
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    copy_input_token_count_result,
    copy_model_context_pressure_profile,
    copy_model_stream_event,
    normalize_model_completion,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._model_errors import model_provider_error_from_payload
from cayu.runtime._run_limits import (
    UNKNOWN_POST_DISPATCH_BUDGET_REASON,
    BudgetDispatchReservationFailed,
    BudgetedOperationRejected,
    BudgetedOperationSucceeded,
    BudgetEvaluation,
    BudgetModelStepLifecycle,
    BudgetReservationLeaseLost,
    BudgetReservationLeaseLostBeforeModelDispatch,
    LimitEvaluation,
    RunLimitController,
    RunLimitGate,
    SessionUsageTracker,
    add_budget_failure_note,
)
from cayu.runtime._session_control import (
    ActiveSessionRun,
    SessionControl,
    SessionInterruptedByRequest,
)
from cayu.runtime.budgets import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservationResult,
    budget_limits_for_session,
    copy_request_budget_limits,
    has_deferred_contextual_price,
)
from cayu.runtime.context import (
    _COMPACTION_ATTEMPT_ID_KEY,
    CompactionRequest,
    CompactionResult,
    ContextBuildError,
    ContextCompactionTelemetry,
    ContextCompactor,
    ContextKnowledgeTelemetry,
    ContextPolicy,
    ContextPressureEstimate,
    ContextPressureOverhead,
    ContextRequest,
    ContextUsageState,
    RuntimeManagedContextPolicy,
    _automatic_compaction_dispatch_runner_scope,
    _automatic_compaction_runner_scope,
    _AutomaticCompactionRunner,
    _compaction_completion_publisher_scope,
    context_build_termination_compaction_telemetry,
    copy_context_messages,
    estimate_context_pressure,
    estimate_model_request_context_pressure,
    noteify_unresolvable_prompt_files,
    sanitize_context_compaction_telemetry,
)
from cayu.runtime.context_counting import ContextCountingConfig, ContextCountingMode
from cayu.runtime.model_steps import (
    AssistantStepResult,
    assistant_text_content,
    classify_assistant_step,
    provider_state_count,
    thinking_count,
)
from cayu.runtime.retry_policy import (
    RetryDecision,
    RetryPolicy,
    copy_retry_policy,
    retry_decision,
    retry_event_payload,
)
from cayu.runtime.sessions import (
    CheckpointTransform,
    EventOrder,
    EventQuery,
    Session,
    SessionStore,
)
from cayu.runtime.structured_output import (
    StructuredOutputSpec,
    StructuredOutputStrategy,
    structured_output_spec_payload,
    structured_output_tool_instruction,
    structured_output_tool_spec,
)
from cayu.runtime.usage import (
    normalize_usage_metrics,
    usage_metrics_from_event_payload,
    usage_metrics_payload,
)

logger = logging.getLogger(__name__)


class ModelAttemptFailed(Exception):
    """A single provider attempt failed after zero or more streamed events."""

    def __init__(
        self,
        *,
        message: str,
        payload: dict[str, Any],
        emitted_error_event: bool,
        cause: Exception | None = None,
    ) -> None:
        self.message = require_nonblank(message, "message")
        self.payload = copy_json_value(payload, "payload")
        self.emitted_error_event = emitted_error_event
        self.cause = cause
        super().__init__(self.message)


@dataclass(frozen=True)
class _ContextCountObservation:
    result: InputTokenCountResult
    observation_id: str


@dataclass(frozen=True)
class _ContextPressureObservation:
    estimate: ContextPressureEstimate
    observation_id: str


@dataclass(frozen=True)
class ModelStepFlowOutcome:
    """Terminal outcome of one logical model step."""

    assistant_step_result: AssistantStepResult | None = None
    stop_session: bool = False

    def __post_init__(self) -> None:
        if self.stop_session == (self.assistant_step_result is not None):
            raise ValueError(
                "A model-step flow outcome must contain either a result or a stop signal."
            )


_CONTEXT_TERMINATION_PERSIST_TIMEOUT_S = 5.0
_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S = 5.0
_CONTEXT_USAGE_AUXILIARY_PAGE_SIZE = 100


def _consume_detached_task_outcome(task: asyncio.Task[Any]) -> None:
    """Retrieve a timed-out task's eventual result after requesting cancellation."""

    with contextlib.suppress(asyncio.CancelledError):
        task.exception()


@dataclass(frozen=True)
class ModelStepBudgetEvaluationRequest:
    evaluation: BudgetEvaluation
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    messages: list[Message]
    run_started_at: float
    turn_usage_tracker: SessionUsageTracker | None
    active_run: ActiveSessionRun[SessionUsageTracker] | None


@dataclass(frozen=True)
class ModelStepLimitEvaluationRequest:
    evaluation: LimitEvaluation
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    messages: list[Message]
    run_started_at: float
    turn_usage_tracker: SessionUsageTracker | None
    active_run: ActiveSessionRun[SessionUsageTracker] | None


@dataclass(frozen=True)
class ModelStepBudgetReservationFailureRequest:
    result: BudgetReservationResult
    session: Session
    registered_agent: runtime_records.RegisteredAgentState
    registered_environment: runtime_records.RegisteredEnvironment | None
    environment_name: str | None
    messages: list[Message]
    run_started_at: float
    turn_usage_tracker: SessionUsageTracker | None
    active_run: ActiveSessionRun[SessionUsageTracker] | None


BudgetEvaluationEventStream = Callable[
    [ModelStepBudgetEvaluationRequest],
    AsyncIterator[Event],
]
LimitEvaluationEventStream = Callable[
    [ModelStepLimitEvaluationRequest],
    AsyncIterator[Event],
]
BudgetReservationFailureEventStream = Callable[
    [ModelStepBudgetReservationFailureRequest],
    AsyncIterator[Event],
]
CheckpointTransformFactory = Callable[[dict[str, Any]], CheckpointTransform]


class _AutomaticCompactionBudgetReservationFailed(RuntimeError):
    def __init__(self, result: BudgetReservationResult) -> None:
        super().__init__(f"Context compaction budget reservation failed: {result.message}")
        self.result = result


class _AutomaticCompactionAdmissionStopped(RuntimeError):
    """The session was stopped by a limit before a compactor provider dispatch."""

    def __init__(
        self,
        *,
        budget_evaluation: BudgetEvaluation | None = None,
        limit_evaluation: LimitEvaluation | None = None,
    ) -> None:
        if (budget_evaluation is None) == (limit_evaluation is None):
            raise ValueError(
                "Automatic compaction admission must contain one rejecting evaluation."
            )
        self.budget_evaluation = budget_evaluation
        self.limit_evaluation = limit_evaluation
        super().__init__("Automatic compaction provider dispatch was stopped by a limit.")


class ModelStepExecutor:
    """Build and execute provider requests for one logical model step."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        session_control: SessionControl[SessionUsageTracker],
        run_limit_controller: RunLimitController,
        context_counting: ContextCountingConfig,
        max_file_attachment_bytes: int,
        max_total_file_attachment_bytes: int,
        max_file_attachments_per_request: int,
        checkpoint_transform: CheckpointTransformFactory,
        apply_budget_evaluation: BudgetEvaluationEventStream,
        apply_limit_evaluation: LimitEvaluationEventStream,
        stop_for_budget_reservation_failure: BudgetReservationFailureEventStream,
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._session_control = session_control
        self._run_limit_controller = run_limit_controller
        self._context_counting = context_counting.model_copy(deep=True)
        self._max_file_attachment_bytes = max_file_attachment_bytes
        self._max_total_file_attachment_bytes = max_total_file_attachment_bytes
        self._max_file_attachments_per_request = max_file_attachments_per_request
        self._checkpoint_transform = checkpoint_transform
        self._apply_budget_evaluation = apply_budget_evaluation
        self._apply_limit_evaluation = apply_limit_evaluation
        self._stop_for_budget_reservation_failure = stop_for_budget_reservation_failure

    def create_run(
        self,
        *,
        provider: ModelProvider,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        knowledge_store: Any,
        request_metadata: dict[str, Any],
        retry_policy: RetryPolicy,
        request_budget_limits: tuple[BudgetLimit, ...],
        limit_gate: RunLimitGate,
        budget_policy: BudgetPolicy | None,
        run_started_at: float,
        turn_usage_tracker: SessionUsageTracker | None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None,
    ) -> ModelStepRun:
        return ModelStepRun(
            self,
            provider=provider,
            session=session,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            environment_name=environment_name,
            structured_output=structured_output,
            thinking=thinking,
            knowledge_store=knowledge_store,
            request_metadata=request_metadata,
            retry_policy=retry_policy,
            request_budget_limits=request_budget_limits,
            limit_gate=limit_gate,
            budget_policy=budget_policy,
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
        )

    async def build_request(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        context_messages: list[Message],
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        step: int,
    ) -> ModelRequest:
        model_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": deepcopy(tool.schema),
            }
            for tool in registered_agent.tools.values()
        ]
        model_messages = context_messages
        if (
            structured_output is not None
            and structured_output.strategy == StructuredOutputStrategy.TOOL
        ):
            model_tools.append(structured_output_tool_spec(structured_output))
            model_messages = _with_structured_output_tool_instruction(
                context_messages,
                structured_output,
            )

        resolved_attachments, unresolvable_prompt_ids = await _resolved_file_attachments(
            messages=model_messages,
            session=session,
            registered_environment=registered_environment,
            max_file_attachment_bytes=self._max_file_attachment_bytes,
            max_total_file_attachment_bytes=self._max_total_file_attachment_bytes,
            max_file_attachments_per_request=self._max_file_attachments_per_request,
        )
        if unresolvable_prompt_ids:
            model_messages = noteify_unresolvable_prompt_files(
                model_messages,
                unresolvable_prompt_ids,
            )
            logger.warning(
                "Prompt file attachment(s) could not be resolved and were omitted from the "
                "provider request (check the session_id used at attach time, or whether the "
                "artifact still exists): %s",
                ", ".join(sorted(unresolvable_prompt_ids)),
            )

        request_options: dict[str, Any] = {
            **copy_json_value(
                registered_agent.spec.provider_options,
                "provider_options",
            ),
            "agent_metadata": deepcopy(registered_agent.spec.metadata),
            "environment_metadata": (
                deepcopy(registered_environment.spec.metadata)
                if registered_environment is not None
                else {}
            ),
            "step": step,
            "structured_output": (
                structured_output_spec_payload(structured_output)
                if structured_output is not None
                else None
            ),
            RESOLVED_FILE_ATTACHMENTS_OPTION: resolved_attachments,
        }
        if thinking is not None:
            request_options["thinking"] = thinking_config_payload(thinking)
        return ModelRequest(
            model=session.model,
            messages=model_messages,
            tools=model_tools,
            options=request_options,
        )

    async def run_with_retries(
        self,
        *,
        provider: ModelProvider,
        model_request: ModelRequest,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        retry_policy: RetryPolicy,
        transcript_cursor_before_request: int,
        record_model_completion: Callable[[Event], None],
        prepare_provider_dispatch: Callable[
            [], Awaitable[tuple[list[Event], BudgetReservationResult | None, Exception | None]]
        ],
        before_provider_dispatch: Callable[[], Awaitable[None]],
        billing_identity: BillingIdentity | None = None,
    ) -> AsyncIterator[tuple[Event | None, AssistantStepResult | None]]:
        retry_policy = copy_retry_policy(retry_policy)
        attempt = 1
        prior_retry_failure: ModelAttemptFailed | None = None
        while True:
            try:
                (
                    reservation_events,
                    reservation_failure,
                    preparation_error,
                ) = await prepare_provider_dispatch()
            except Exception as accounting_exc:
                reservation_events = []
                reservation_failure = None
                preparation_error = accounting_exc
            for reservation_event in reservation_events:
                yield reservation_event, None
            if preparation_error is not None:
                if prior_retry_failure is None:
                    raise preparation_error
                authoritative_failure = prior_retry_failure.cause
                if authoritative_failure is None:
                    authoritative_failure = RuntimeError(prior_retry_failure.message)
                add_budget_failure_note(
                    authoritative_failure,
                    operation="retry preparation",
                    accounting_failure=preparation_error,
                )
                raise authoritative_failure from prior_retry_failure
            prior_retry_failure = None
            if reservation_failure is not None:
                raise BudgetDispatchReservationFailed(reservation_failure)

            (
                context_pressure_observation,
                context_pressure_event,
            ) = await self._observe_context_pressure(
                model_request=model_request,
                session=session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                environment_name=environment_name,
                step=step,
                attempt=attempt,
                max_attempts=retry_policy.max_attempts,
            )
            if context_pressure_event is not None:
                yield context_pressure_event, None
            context_count_observation, context_count_event = await self._observe_context_count(
                provider=provider,
                model_request=model_request,
                session=session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                environment_name=environment_name,
                step=step,
                attempt=attempt,
                max_attempts=retry_policy.max_attempts,
            )
            if context_count_event is not None:
                yield context_count_event, None
            yield (
                await self._event_writer.emit(
                    Event(
                        type=EventType.MODEL_STARTED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        payload={
                            "model": session.model,
                            "provider": registered_provider.name,
                            "step": step,
                            "attempt": attempt,
                            "max_attempts": retry_policy.max_attempts,
                        },
                        environment_name=environment_name,
                    )
                ),
                None,
            )
            attempt_events = self._run_once(
                provider=provider,
                model_request=model_request,
                session=session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                environment_name=environment_name,
                step=step,
                attempt=attempt,
                max_attempts=retry_policy.max_attempts,
                transcript_cursor_before_request=transcript_cursor_before_request,
                before_provider_dispatch=before_provider_dispatch,
                billing_identity=billing_identity,
            )
            try:
                result: AssistantStepResult | None = None
                async for event, step_result in attempt_events:
                    if event is not None:
                        if event.type == EventType.MODEL_COMPLETED:
                            record_model_completion(event)
                        yield event, None
                        if (
                            event.type == EventType.MODEL_COMPLETED
                            and context_pressure_observation is not None
                        ):
                            yield (
                                await self._event_writer.emit(
                                    _context_pressure_reconciled_event(
                                        event,
                                        observation=context_pressure_observation,
                                        session=session,
                                        registered_agent=registered_agent,
                                        registered_provider=registered_provider,
                                        environment_name=environment_name,
                                        step=step,
                                        attempt=attempt,
                                        max_attempts=retry_policy.max_attempts,
                                    )
                                ),
                                None,
                            )
                        if (
                            event.type == EventType.MODEL_COMPLETED
                            and context_count_observation is not None
                        ):
                            yield (
                                await self._event_writer.emit(
                                    _context_count_reconciled_event(
                                        event,
                                        observation=context_count_observation,
                                        session=session,
                                        registered_agent=registered_agent,
                                        registered_provider=registered_provider,
                                        environment_name=environment_name,
                                        step=step,
                                        attempt=attempt,
                                        max_attempts=retry_policy.max_attempts,
                                    )
                                ),
                                None,
                            )
                    if step_result is not None:
                        result = step_result
                if result is None:
                    raise RuntimeError("Model step finished without a result.")
                yield None, result
                return
            except ModelAttemptFailed as exc:
                status_code, retryable, retry_after_s = _typed_retry_fields(exc)
                decision = retry_decision(
                    policy=retry_policy,
                    attempt=attempt,
                    error=exc.message,
                    status_code=status_code,
                    retryable=retryable,
                    retry_after_s=retry_after_s,
                )
                if decision.reason is not None and not exc.emitted_error_event:
                    yield (
                        await self._event_writer.emit(
                            Event(
                                type=EventType.MODEL_ERROR,
                                session_id=session.id,
                                agent_name=registered_agent.spec.name,
                                environment_name=environment_name,
                                payload=_retry_attempt_payload(
                                    exc.payload,
                                    step=step,
                                    attempt=attempt,
                                    max_attempts=retry_policy.max_attempts,
                                ),
                            )
                        ),
                        None,
                    )
                if not decision.retry:
                    if exc.cause is not None:
                        raise exc.cause from exc
                    raise RuntimeError(exc.message) from exc
                yield (
                    await self._event_writer.emit(
                        _model_retry_event(
                            session=session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            registered_provider=registered_provider,
                            step=step,
                            decision=decision,
                            error=exc.message,
                        )
                    ),
                    None,
                )
                yield (
                    await self._event_writer.emit(
                        _model_attempt_discarded_event(
                            session=session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            registered_provider=registered_provider,
                            step=step,
                            decision=decision,
                        )
                    ),
                    None,
                )
                await self._sleep_before_retry(session.id, decision)
                prior_retry_failure = exc
                attempt += 1
            finally:
                await _close_async_iterator(attempt_events)

    async def _observe_context_pressure(
        self,
        *,
        model_request: ModelRequest,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
    ) -> tuple[_ContextPressureObservation | None, Event | None]:
        if self._context_counting.mode == ContextCountingMode.OFF:
            return None, None
        observation_id = str(uuid4())
        profile = copy_model_context_pressure_profile(
            registered_provider.provider.context_pressure_profile
        )
        estimate = estimate_model_request_context_pressure(
            model_request=model_request,
            image_min_tokens=profile.image_min_tokens,
            document_min_tokens=profile.document_min_tokens,
            document_bytes_per_token=profile.document_bytes_per_token,
            tool_schema_chars_per_token=profile.tool_schema_chars_per_token,
        )
        observation = _ContextPressureObservation(
            estimate=estimate,
            observation_id=observation_id,
        )
        event = await self._event_writer.emit(
            Event(
                type=EventType.CONTEXT_PRESSURE_ESTIMATED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload={
                    **_context_count_base_payload(
                        model_request=model_request,
                        provider_name=registered_provider.name,
                        step=step,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        observation_id=observation_id,
                    ),
                    "estimate": estimate.model_dump(mode="json"),
                },
            )
        )
        return observation, event

    async def _observe_context_count(
        self,
        *,
        provider: ModelProvider,
        model_request: ModelRequest,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
    ) -> tuple[_ContextCountObservation | None, Event | None]:
        if self._context_counting.mode == ContextCountingMode.OFF:
            return None, None
        observation_id = str(uuid4())
        base_payload = _context_count_base_payload(
            model_request=model_request,
            provider_name=registered_provider.name,
            step=step,
            attempt=attempt,
            max_attempts=max_attempts,
            observation_id=observation_id,
        )
        try:
            provider_result = await provider.count_input_tokens(
                _copy_model_request_for_counting(model_request)
            )
            provider_result = copy_input_token_count_result(provider_result)
            result = (
                provider_result
                if provider_result is not None
                else InputTokenCountResult(
                    input_tokens=None,
                    method=InputTokenCountMethod.UNAVAILABLE,
                    confidence=InputTokenCountConfidence.UNAVAILABLE,
                )
            )
        except Exception as exc:
            event = await self._event_writer.emit(
                Event(
                    type=EventType.CONTEXT_COUNT_FAILED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        **base_payload,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
            )
            return None, event

        observation = _ContextCountObservation(
            result=result,
            observation_id=observation_id,
        )
        event = await self._event_writer.emit(
            Event(
                type=EventType.CONTEXT_COUNTED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload={
                    **base_payload,
                    "count": result.model_dump(mode="json"),
                },
            )
        )
        return observation, event

    async def _run_once(
        self,
        *,
        provider: ModelProvider,
        model_request: ModelRequest,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
        transcript_cursor_before_request: int,
        before_provider_dispatch: Callable[[], Awaitable[None]],
        billing_identity: BillingIdentity | None,
    ) -> AsyncIterator[tuple[Event | None, AssistantStepResult | None]]:
        assistant_parts: list[
            transcript_helpers.AssistantTextPart
            | transcript_helpers.AssistantThinkingPart
            | ToolCallPart
        ] = []
        thinking_options = model_request.options.get("thinking")
        include_thinking_in_transcript = (
            thinking_options.get("include_in_transcript", True)
            if isinstance(thinking_options, dict)
            else True
        )
        tool_calls: list[runtime_records.ToolCallRequest] = []
        provider_state_parts: list[ProviderStatePart] = []
        completed_stream_event: ModelStreamEvent | None = None
        step_result: AssistantStepResult | None = None
        model_completed = False
        profile = copy_model_context_pressure_profile(
            registered_provider.provider.context_pressure_profile
        )
        context_pressure_estimate = estimate_model_request_context_pressure(
            model_request=model_request,
            image_min_tokens=profile.image_min_tokens,
            document_min_tokens=profile.document_min_tokens,
            document_bytes_per_token=profile.document_bytes_per_token,
            tool_schema_chars_per_token=profile.tool_schema_chars_per_token,
        )
        interrupt_poll = self._session_control.stream_interrupt_poll(session.id)
        # This is the accounting boundary: after the callback returns, the next
        # expression enters provider-controlled code and billable work may occur.
        await before_provider_dispatch()
        provider_events: AsyncIterator[ModelStreamEvent] | None = None
        provider_exhausted = False
        try:
            provider_events = provider.stream(model_request)
            async for raw_stream_event in provider_events:
                stream_event = _validate_stream_event(raw_stream_event)
                await interrupt_poll.raise_if_interrupted()
                if model_completed:
                    message = f"Model provider emitted event after completed: {stream_event.type}"
                    raise ModelAttemptFailed(
                        message=message,
                        payload={"error": message, "error_type": "RuntimeError"},
                        emitted_error_event=False,
                        cause=RuntimeError(message),
                    )

                if stream_event.type == ModelStreamEventType.TOOL_CALL:
                    tool_call = transcript_helpers.parse_tool_call(stream_event.payload)
                    tool_calls.append(tool_call)
                    assistant_parts.append(transcript_helpers.tool_call_part(tool_call))
                    continue

                if stream_event.type == ModelStreamEventType.TEXT_DELTA:
                    transcript_helpers.append_assistant_text_delta(
                        assistant_parts,
                        stream_event.delta,
                    )
                elif stream_event.type == ModelStreamEventType.THINKING:
                    transcript_helpers.append_assistant_thinking_delta(
                        assistant_parts,
                        stream_event.delta,
                        provider_state=stream_event.payload.get("provider_state"),
                        include=include_thinking_in_transcript,
                    )
                    if not stream_event.delta:
                        # Opaque/redacted thinking state belongs in the transcript,
                        # but an empty readable delta should not reach consumers.
                        continue
                elif stream_event.type == ModelStreamEventType.COMPLETED:
                    try:
                        billing_identity = completed_billing_identity(
                            billing_identity,
                            provider.billing_identity_for_completion(
                                billing_identity,
                                stream_event.payload,
                            ),
                        )
                    except Exception as exc:
                        provider_error = ModelProviderError(
                            str(exc),
                            provider=registered_provider.name,
                            error_type=type(exc).__name__,
                            error_code="billing_identity_resolution_failed",
                            retryable=False,
                        )
                        error_payload = {
                            "error": str(provider_error),
                            "error_type": type(provider_error).__name__,
                            "stage": "billing_identity_for_completion",
                            **provider_error.error_payload_fields(),
                        }
                        yield (
                            await self._event_writer.emit(
                                Event(
                                    type=EventType.MODEL_ERROR,
                                    session_id=session.id,
                                    agent_name=registered_agent.spec.name,
                                    environment_name=environment_name,
                                    payload=_retry_attempt_payload(
                                        error_payload,
                                        step=step,
                                        attempt=attempt,
                                        max_attempts=max_attempts,
                                    ),
                                )
                            ),
                            None,
                        )
                        raise ModelAttemptFailed(
                            message=str(provider_error),
                            payload=error_payload,
                            emitted_error_event=True,
                            cause=provider_error,
                        ) from exc
                    model_completed = True
                    completed_stream_event = stream_event
                    provider_state_parts = transcript_helpers.provider_state_parts(
                        stream_event.payload
                    )
                    assistant_message = transcript_helpers.assistant_message(
                        content_parts=assistant_parts,
                        provider_state_parts=provider_state_parts,
                    )
                    step_result = _assistant_step_result(
                        session_id=session.id,
                        step=step,
                        assistant_message=assistant_message,
                        tool_calls=tool_calls,
                        completion=_stream_event_completion(completed_stream_event),
                    )
                    classification = classify_assistant_step(step_result)
                    event = _model_stream_event_to_runtime_event(
                        stream_event,
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        provider_name=registered_provider.name,
                        step=step,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        classification=classification.payload(),
                        context_pressure_estimate=context_pressure_estimate,
                        transcript_cursor_after_completion=(
                            transcript_cursor_before_request
                            + (1 if assistant_message is not None else 0)
                        ),
                        usage_dialect=registered_provider.provider.usage_dialect,
                        billing_identity=billing_identity,
                    )
                    yield await self._event_writer.emit(event), None
                    continue

                if stream_event.type == ModelStreamEventType.ERROR:
                    provider_error = model_provider_error_from_payload(
                        stream_event.payload,
                        fallback_provider=registered_provider.name,
                    )
                    if isinstance(provider_error, ModelContextOverflowError):
                        # Providers may flatten a typed overflow into an error
                        # event. Rehydrate it so bounded recovery can shrink the
                        # request instead of spending generic retries on it.
                        raise provider_error

                event = _model_stream_event_to_runtime_event(
                    stream_event,
                    session=session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    provider_name=registered_provider.name,
                    step=step,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    usage_dialect=registered_provider.provider.usage_dialect,
                )
                emitted_event = await self._event_writer.emit(event)
                if stream_event.type == ModelStreamEventType.ERROR:
                    message = str(stream_event.payload.get("error") or "Model provider error")
                    provider_error = model_provider_error_from_payload(
                        stream_event.payload,
                        fallback_provider=registered_provider.name,
                        fallback_message=message,
                    )
                    yield emitted_event, None
                    raise ModelAttemptFailed(
                        message=message,
                        payload=copy_json_value(stream_event.payload, "payload"),
                        emitted_error_event=True,
                        cause=provider_error or RuntimeError(message),
                    )
                yield emitted_event, None
            provider_exhausted = True
        except SessionInterruptedByRequest:
            raise
        except asyncio.CancelledError:
            raise
        except ModelAttemptFailed:
            raise
        except ModelContextOverflowError:
            raise
        except Exception as exc:
            raise ModelAttemptFailed(
                message=str(exc),
                payload={"error": str(exc), "error_type": type(exc).__name__},
                emitted_error_event=False,
                cause=exc,
            ) from exc
        finally:
            if provider_events is not None and not provider_exhausted:
                await _close_async_iterator(provider_events)

        if not model_completed:
            message = "Model provider stream ended without a completed event."
            raise ModelAttemptFailed(
                message=message,
                payload={"error": message, "error_type": "RuntimeError"},
                emitted_error_event=False,
                cause=RuntimeError(message),
            )
        await self._session_control.raise_if_interrupted(session.id)
        if completed_stream_event is None:
            raise RuntimeError("Model provider completed without completion metadata.")
        if step_result is None:
            raise RuntimeError("Model provider completed without an assistant step result.")
        yield None, step_result

    async def _sleep_before_retry(self, session_id: str, decision: RetryDecision) -> None:
        await self._session_control.raise_if_interrupted(session_id)
        if decision.delay_seconds > 0:
            await asyncio.sleep(decision.delay_seconds)
        await self._session_control.raise_if_interrupted(session_id)


class ModelStepRun:
    """Per-run model-step dependencies and accounting state."""

    def __init__(
        self,
        executor: ModelStepExecutor,
        *,
        provider: ModelProvider,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        knowledge_store: Any,
        request_metadata: dict[str, Any],
        retry_policy: RetryPolicy,
        request_budget_limits: tuple[BudgetLimit, ...],
        limit_gate: RunLimitGate,
        budget_policy: BudgetPolicy | None,
        run_started_at: float,
        turn_usage_tracker: SessionUsageTracker | None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None,
    ) -> None:
        self._executor = executor
        self._provider = provider
        self._session = session
        self._registered_agent = registered_agent
        self._registered_provider = registered_provider
        self._registered_environment = registered_environment
        self._environment_name = environment_name
        self._structured_output = structured_output
        self._thinking = thinking
        self._knowledge_store = knowledge_store
        self._request_metadata = copy_json_value(request_metadata, "metadata")
        self._retry_policy = copy_retry_policy(retry_policy)
        self._request_budget_limits = copy_request_budget_limits(request_budget_limits)
        self._limit_gate = limit_gate
        self._budget_policy = budget_policy
        self._run_started_at = run_started_at
        self._turn_usage_tracker = turn_usage_tracker
        self._active_run = active_run
        contextual_limits = (
            *budget_limits_for_session(
                policy=self._budget_policy,
                agent_name=self._registered_agent.spec.name,
                causal_budget_id=self._session.causal_budget_id,
            ),
            *self._request_budget_limits,
        )
        self._deferred_contextual_price = any(
            has_deferred_contextual_price(
                limit.pricing,
                provider_name=(self._provider.billing_provider_name or self._session.provider_name),
                model=self._session.model,
            )
            for limit in contextual_limits
        )

    async def execute(
        self,
        *,
        step: int,
        messages: list[Message],
    ) -> AsyncIterator[tuple[Event | None, ModelStepFlowOutcome | None]]:
        context_messages: list[Message]
        compaction_budget_events: list[Event] = []
        published_compaction_attempt_ids: set[str] = set()
        compaction_start_events: list[Event] = []
        compaction_completion_events: dict[str, Event] = {}

        async def run_automatic_compaction(
            compactor: ContextCompactor,
            compaction_request: CompactionRequest,
            compaction_started: ContextCompactionTelemetry,
            execute: Callable[[], Awaitable[CompactionResult]],
            completed_payloads: Callable[[], list[dict[str, Any]]],
        ) -> CompactionResult:
            await self._persist_automatic_compaction_started(
                compaction_started,
                published_events=compaction_budget_events,
                start_events=compaction_start_events,
            )

            async def publish_completions(payloads: list[dict[str, Any]]) -> None:
                await self._persist_automatic_compaction_completions(
                    payloads,
                    published_attempt_ids=published_compaction_attempt_ids,
                    published_events=compaction_budget_events,
                    completion_events=compaction_completion_events,
                )

            async def run() -> CompactionResult:
                return await self._run_automatic_compaction_with_budget(
                    compactor=compactor,
                    compaction_request=compaction_request,
                    execute=execute,
                    completed_payloads=completed_payloads,
                    budget_events=compaction_budget_events,
                    messages=messages,
                )

            if not compactor._uses_runtime_provider_dispatch_runner_for_request(compaction_request):
                return await run()
            with _compaction_completion_publisher_scope(publish_completions):
                return await run()

        current_task = asyncio.current_task()
        context_build_cancellation_requests = (
            0 if current_task is None else current_task.cancelling()
        )
        try:
            (
                context_messages,
                checkpoint_update,
                checkpoint_event_payload,
                context_compaction_telemetry,
                context_knowledge_telemetry,
            ) = await _build_context(
                context_policy=self._registered_agent.context_policy,
                session_store=self._executor._session_store,
                session=self._session,
                agent_spec=_session_agent_spec(
                    registered_agent=self._registered_agent,
                    session=self._session,
                ),
                messages=messages,
                step=step,
                environment_name=self._environment_name,
                knowledge_store=self._knowledge_store,
                request_metadata=self._request_metadata,
                pressure_overhead=_context_pressure_overhead(
                    registered_provider=self._registered_provider,
                    registered_agent=self._registered_agent,
                    registered_environment=self._registered_environment,
                    structured_output=self._structured_output,
                    thinking=self._thinking,
                    step=step,
                ),
                count_input_tokens=self._context_input_token_counter(step=step),
                build_cache_prefix_request=self._cache_prefix_request_builder(step=step),
                run_compaction=run_automatic_compaction,
            )
        except ContextBuildError as exc:
            (
                context_failure_events,
                context_failure_persistence,
            ) = await self._context_build_failure_events(
                exc,
                published_compaction_attempt_ids=published_compaction_attempt_ids,
                compaction_completion_events=compaction_completion_events,
                compaction_start_event=(
                    compaction_start_events[0] if compaction_start_events else None
                ),
                compaction_started_published=any(
                    event.type == EventType.CONTEXT_COMPACTION_STARTED
                    for event in compaction_budget_events
                ),
            )
            for event in compaction_budget_events:
                yield event, None
            for event in context_failure_events:
                yield event, None
            if context_failure_persistence is not None:
                raise context_failure_persistence from exc
            if isinstance(exc.cause, _AutomaticCompactionBudgetReservationFailed):
                async for event in self._stop_for_budget_reservation_failure(
                    result=exc.cause.result,
                    messages=messages,
                ):
                    yield event, None
                yield None, ModelStepFlowOutcome(stop_session=True)
                return
            if isinstance(exc.cause, _AutomaticCompactionAdmissionStopped):
                admission_events = self._automatic_compaction_admission_events(
                    exc.cause,
                    messages=messages,
                )
                try:
                    async for event in admission_events:
                        yield event, None
                finally:
                    await _close_async_iterator(admission_events)
                yield None, ModelStepFlowOutcome(stop_session=True)
                return
            raise exc.cause from exc
        except BaseException as exc:
            await self._persist_context_build_termination_events(
                exc,
                published_compaction_attempt_ids=published_compaction_attempt_ids,
                compaction_completion_events=compaction_completion_events,
                compaction_start_event=(
                    compaction_start_events[0] if compaction_start_events else None
                ),
                compaction_started_published=any(
                    event.type == EventType.CONTEXT_COMPACTION_STARTED
                    for event in compaction_budget_events
                ),
                cancellation_requests_before_build=context_build_cancellation_requests,
            )
            raise

        context_success_events, context_success_persistence = await self._context_success_events(
            checkpoint_update=checkpoint_update,
            checkpoint_event_payload=checkpoint_event_payload,
            compaction_telemetry=context_compaction_telemetry,
            knowledge_telemetry=context_knowledge_telemetry,
            published_compaction_attempt_ids=published_compaction_attempt_ids,
            compaction_completion_events=compaction_completion_events,
            compaction_start_event=(
                compaction_start_events[0] if compaction_start_events else None
            ),
            compaction_started_published=any(
                event.type == EventType.CONTEXT_COMPACTION_STARTED
                for event in compaction_budget_events
            ),
        )
        for event in compaction_budget_events:
            yield event, None
        for event in context_success_events:
            yield event, None
        if context_success_persistence is not None:
            raise context_success_persistence
        await self._executor._session_control.raise_if_interrupted(self._session.id)

        if _has_provider_backed_context_compaction(context_compaction_telemetry):
            should_stop: bool | None = None
            gate_events = self._post_compaction_gate(messages=messages)
            try:
                async for event, gate_outcome in gate_events:
                    if event is not None:
                        yield event, None
                    if gate_outcome is not None:
                        should_stop = gate_outcome
            finally:
                await _close_async_iterator(gate_events)
            if should_stop is None:
                raise RuntimeError("Post-compaction gate finished without an outcome.")
            if should_stop:
                yield None, ModelStepFlowOutcome(stop_session=True)
                return

        model_request = await self._executor.build_request(
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            context_messages=context_messages,
            structured_output=self._structured_output,
            thinking=self._thinking,
            step=step,
        )
        request_events = self._execute_request(
            model_request=model_request,
            step=step,
            messages=messages,
        )
        try:
            async for event, outcome in request_events:
                yield event, outcome
        finally:
            await _close_async_iterator(request_events)

    async def _execute_request(
        self,
        *,
        model_request: ModelRequest,
        step: int,
        messages: list[Message],
    ) -> AsyncIterator[tuple[Event | None, ModelStepFlowOutcome | None]]:
        controller = self._executor._run_limit_controller
        try:
            billing_identity = copy_billing_identity(
                await self._provider.billing_identity_for_request(model_request)
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            provider_error = (
                exc
                if isinstance(exc, ModelProviderError)
                else ModelProviderError(
                    str(exc),
                    provider=self._registered_provider.name,
                    error_type=type(exc).__name__,
                    error_code="billing_identity_resolution_failed",
                    retryable=False,
                )
            )
            payload = {
                "error": str(provider_error),
                "error_type": type(provider_error).__name__,
                "stage": "billing_identity_for_request",
                **provider_error.error_payload_fields(),
            }
            yield (
                await self._executor._event_writer.emit(
                    Event(
                        type=EventType.MODEL_ERROR,
                        session_id=self._session.id,
                        agent_name=self._registered_agent.spec.name,
                        environment_name=self._environment_name,
                        payload=_retry_attempt_payload(
                            payload,
                            step=step,
                            attempt=1,
                            max_attempts=self._retry_policy.max_attempts,
                        ),
                    )
                ),
                None,
            )
            if provider_error is exc:
                raise
            raise provider_error from exc
        if billing_identity is not None or self._has_deferred_contextual_price():
            should_stop: bool | None = None
            gate_events = self._billing_identity_budget_gate(
                messages=messages,
                billing_identity=billing_identity,
            )
            try:
                async for event, gate_outcome in gate_events:
                    if event is not None:
                        yield event, None
                    if gate_outcome is not None:
                        should_stop = gate_outcome
            finally:
                await _close_async_iterator(gate_events)
            if should_stop is None:
                raise RuntimeError("Billing-identity budget gate finished without an outcome.")
            if should_stop:
                yield None, ModelStepFlowOutcome(stop_session=True)
                return
        reservation_setup = await controller.reserve_for_model_step(
            session=self._session,
            agent_name=self._registered_agent.spec.name,
            provider_name=self._registered_provider.name,
            environment_name=self._environment_name,
            budget_policy=self._budget_policy,
            request_budget_limits=self._request_budget_limits,
            billing_identity=billing_identity,
        )
        budget_reservations = list(reservation_setup.reservations)
        try:
            for event in reservation_setup.events:
                yield event, None
        except (GeneratorExit, asyncio.CancelledError) as authoritative_exc:
            if reservation_setup.failure is None and reservation_setup.error is None:
                async for _ in controller.settlement_events_preserving_failure(
                    controller.release_reservations(
                        budget_reservations,
                        session=self._session,
                        agent_name=self._registered_agent.spec.name,
                        environment_name=self._environment_name,
                        reason="model step abandoned before provider dispatch",
                    ),
                    authoritative_failure=authoritative_exc,
                ):
                    pass
            raise
        if reservation_setup.error is not None:
            raise reservation_setup.error
        if reservation_setup.failure is not None:
            async for event in self._stop_for_budget_reservation_failure(
                result=reservation_setup.failure,
                messages=messages,
            ):
                yield event, None
            yield None, ModelStepFlowOutcome(stop_session=True)
            return

        if budget_reservations and controller.reservation_ttl_seconds is not None:
            try:
                await controller.renew_reservations(budget_reservations)
            except asyncio.CancelledError as authoritative_exc:
                async for event in controller.settlement_events_preserving_failure(
                    controller.release_reservations(
                        budget_reservations,
                        session=self._session,
                        agent_name=self._registered_agent.spec.name,
                        environment_name=self._environment_name,
                        reason="model step cancelled before provider dispatch",
                    ),
                    authoritative_failure=authoritative_exc,
                ):
                    yield event, None
                raise
            except BudgetReservationLeaseLost as authoritative_exc:
                async for event in controller.settlement_events_preserving_failure(
                    controller.release_reservations(
                        budget_reservations,
                        session=self._session,
                        agent_name=self._registered_agent.spec.name,
                        environment_name=self._environment_name,
                        reason="reservation lease expired before model step",
                    ),
                    authoritative_failure=authoritative_exc,
                ):
                    yield event, None
                raise

        lifecycle = BudgetModelStepLifecycle()
        lifecycle.prepare_provider_dispatch(budget_reservations)

        async def settle_provider_dispatch() -> tuple[list[Event], Exception | None]:
            if lifecycle.pending_reservations is not None:
                return [], None
            settlement_events: list[Event] = []
            try:
                async for event in controller.reconcile_dispatched_reservations(
                    budget_reservations,
                    lifecycle=lifecycle,
                    session=self._session,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    unknown_reason=UNKNOWN_POST_DISPATCH_BUDGET_REASON,
                ):
                    settlement_events.append(event)
            except Exception as settlement_error:
                return settlement_events, settlement_error
            return settlement_events, None

        async def prepare_provider_dispatch() -> tuple[
            list[Event],
            BudgetReservationResult | None,
            Exception | None,
        ]:
            if lifecycle.pending_reservations is not None:
                return [], None, None
            settlement_events, settlement_error = await settle_provider_dispatch()
            if settlement_error is not None:
                return settlement_events, None, settlement_error
            retry_setup = await controller.reserve_for_model_step(
                session=self._session,
                agent_name=self._registered_agent.spec.name,
                provider_name=self._registered_provider.name,
                environment_name=self._environment_name,
                budget_policy=self._budget_policy,
                request_budget_limits=self._request_budget_limits,
                billing_identity=billing_identity,
            )
            if retry_setup.error is not None:
                return settlement_events + list(retry_setup.events), None, retry_setup.error
            if retry_setup.failure is not None:
                return settlement_events + list(retry_setup.events), retry_setup.failure, None
            retry_reservations = list(retry_setup.reservations)
            budget_reservations.extend(retry_reservations)
            lifecycle.prepare_provider_dispatch(retry_reservations)
            return settlement_events + list(retry_setup.events), None, None

        async def before_provider_dispatch() -> None:
            await controller.before_provider_dispatch(
                budget_reservations,
                lifecycle=lifecycle,
            )

        flow_outcome: ModelStepFlowOutcome | None = None
        model_step_events = self._run_with_context_overflow_recovery(
            provider=self._provider,
            model_request=model_request,
            messages=messages,
            step=step,
            transcript_cursor_before_request=len(messages),
            record_model_completion=lifecycle.record_model_completion,
            settle_provider_dispatch=settle_provider_dispatch,
            prepare_provider_dispatch=prepare_provider_dispatch,
            before_provider_dispatch=before_provider_dispatch,
            billing_identity=billing_identity,
        )
        guarded_events = controller.model_step_events_with_heartbeat(
            model_step_events,
            reservations=budget_reservations,
            lifecycle=lifecycle,
        )
        try:
            async for event, outcome in guarded_events:
                if event is not None:
                    yield event, None
                if outcome is not None:
                    if flow_outcome is not None:
                        raise RuntimeError(
                            "Model step produced more than one terminal flow outcome."
                        )
                    flow_outcome = outcome
        except GeneratorExit as authoritative_exc:
            async for _ in controller.settlement_events_preserving_failure(
                controller.settle_after_model_failure(
                    budget_reservations,
                    lifecycle=lifecycle,
                    session=self._session,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    release_reason="model step abandoned before provider dispatch",
                ),
                authoritative_failure=authoritative_exc,
            ):
                pass
            raise
        except BudgetDispatchReservationFailed as exc:
            async for event in controller.settle_after_model_failure(
                budget_reservations,
                lifecycle=lifecycle,
                session=self._session,
                agent_name=self._registered_agent.spec.name,
                environment_name=self._environment_name,
                release_reason="retry reservation failed before provider dispatch",
            ):
                yield event, None
            async for event in self._stop_for_budget_reservation_failure(
                result=exc.result,
                messages=messages,
            ):
                yield event, None
            yield None, ModelStepFlowOutcome(stop_session=True)
            return
        except BudgetReservationLeaseLostBeforeModelDispatch as authoritative_exc:
            async for event in controller.settlement_events_preserving_failure(
                controller.release_reservations(
                    budget_reservations,
                    session=self._session,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    reason="reservation lease expired before model dispatch",
                ),
                authoritative_failure=authoritative_exc,
            ):
                yield event, None
            raise
        except BudgetReservationLeaseLost as authoritative_exc:
            async for event in controller.settlement_events_preserving_failure(
                controller.settle_after_model_failure(
                    budget_reservations,
                    lifecycle=lifecycle,
                    session=self._session,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    release_reason="reservation heartbeat lost before provider dispatch",
                    unknown_reason="reservation heartbeat lost; charged reserved amount",
                ),
                authoritative_failure=authoritative_exc,
            ):
                yield event, None
            raise
        except SessionInterruptedByRequest as authoritative_exc:
            async for event in controller.settlement_events_preserving_failure(
                controller.settle_after_model_failure(
                    budget_reservations,
                    lifecycle=lifecycle,
                    session=self._session,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    release_reason="session interrupted before provider dispatch",
                ),
                authoritative_failure=authoritative_exc,
            ):
                yield event, None
            raise
        except asyncio.CancelledError as authoritative_exc:
            async for event in controller.settlement_events_preserving_failure(
                controller.settle_after_model_failure(
                    budget_reservations,
                    lifecycle=lifecycle,
                    session=self._session,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    release_reason="model step cancelled before provider dispatch",
                ),
                authoritative_failure=authoritative_exc,
            ):
                yield event, None
            raise
        except Exception as provider_exc:
            async for event in controller.settlement_events_preserving_failure(
                controller.settle_after_model_failure(
                    budget_reservations,
                    lifecycle=lifecycle,
                    session=self._session,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    release_reason="model step failed before provider dispatch",
                ),
                authoritative_failure=provider_exc,
            ):
                yield event, None
            raise
        finally:
            try:
                await _close_async_iterator(guarded_events)
            finally:
                await _close_async_iterator(model_step_events)

        if lifecycle.dispatches:
            async for event in controller.reconcile_dispatched_reservations(
                budget_reservations,
                lifecycle=lifecycle,
                session=self._session,
                agent_name=self._registered_agent.spec.name,
                environment_name=self._environment_name,
                unknown_reason=UNKNOWN_POST_DISPATCH_BUDGET_REASON,
            ):
                yield event, None
        if flow_outcome is None:
            raise RuntimeError("Model step finished without a terminal flow outcome.")
        yield None, flow_outcome

    async def _run_with_context_overflow_recovery(
        self,
        *,
        provider: ModelProvider,
        model_request: ModelRequest,
        messages: list[Message],
        step: int,
        transcript_cursor_before_request: int,
        record_model_completion: Callable[[Event], None],
        settle_provider_dispatch: Callable[[], Awaitable[tuple[list[Event], Exception | None]]],
        prepare_provider_dispatch: Callable[
            [], Awaitable[tuple[list[Event], BudgetReservationResult | None, Exception | None]]
        ],
        before_provider_dispatch: Callable[[], Awaitable[None]],
        billing_identity: BillingIdentity | None,
    ) -> AsyncIterator[tuple[Event | None, ModelStepFlowOutcome | None]]:
        overflow_policy = self._registered_agent.context_overflow_policy
        compaction_budget_events: list[Event] = []
        published_compaction_attempt_ids: set[str] = set()
        compaction_start_events: list[Event] = []
        compaction_completion_events: dict[str, Event] = {}

        def run_attempt(
            request: ModelRequest,
        ) -> AsyncIterator[tuple[Event | None, AssistantStepResult | None]]:
            return self._executor.run_with_retries(
                provider=provider,
                model_request=request,
                session=self._session,
                registered_agent=self._registered_agent,
                registered_provider=self._registered_provider,
                environment_name=self._environment_name,
                step=step,
                retry_policy=self._retry_policy,
                transcript_cursor_before_request=transcript_cursor_before_request,
                record_model_completion=record_model_completion,
                prepare_provider_dispatch=prepare_provider_dispatch,
                before_provider_dispatch=before_provider_dispatch,
                billing_identity=billing_identity,
            )

        attempt_events = run_attempt(model_request)
        try:
            try:
                async for event, result in attempt_events:
                    yield (
                        event,
                        ModelStepFlowOutcome(assistant_step_result=result)
                        if result is not None
                        else None,
                    )
                return
            except ModelContextOverflowError as exc:
                if overflow_policy is None:
                    raise
                yield (
                    await self._executor._event_writer.emit(
                        Event(
                            type=EventType.CONTEXT_OVERFLOW_DETECTED,
                            session_id=self._session.id,
                            agent_name=self._registered_agent.spec.name,
                            environment_name=self._environment_name,
                            payload=_context_overflow_event_payload(
                                exc,
                                step=step,
                                phase="initial",
                                original_message_count=len(model_request.messages),
                            ),
                        )
                    ),
                    None,
                )
        finally:
            await _close_async_iterator(attempt_events)

        async def run_automatic_compaction(
            compactor: ContextCompactor,
            compaction_request: CompactionRequest,
            compaction_started: ContextCompactionTelemetry,
            execute: Callable[[], Awaitable[CompactionResult]],
            completed_payloads: Callable[[], list[dict[str, Any]]],
        ) -> CompactionResult:
            await self._persist_automatic_compaction_started(
                compaction_started,
                published_events=compaction_budget_events,
                start_events=compaction_start_events,
            )

            async def publish_completions(payloads: list[dict[str, Any]]) -> None:
                await self._persist_automatic_compaction_completions(
                    payloads,
                    published_attempt_ids=published_compaction_attempt_ids,
                    published_events=compaction_budget_events,
                    completion_events=compaction_completion_events,
                )

            async def run() -> CompactionResult:
                return await self._run_automatic_compaction_with_budget(
                    compactor=compactor,
                    compaction_request=compaction_request,
                    execute=execute,
                    completed_payloads=completed_payloads,
                    budget_events=compaction_budget_events,
                    messages=messages,
                )

            if not compactor._uses_runtime_provider_dispatch_runner_for_request(compaction_request):
                return await run()
            with _compaction_completion_publisher_scope(publish_completions):
                return await run()

        current_task = asyncio.current_task()
        context_build_cancellation_requests = (
            0 if current_task is None else current_task.cancelling()
        )
        try:
            (
                recovery_context_messages,
                checkpoint_update,
                checkpoint_event_payload,
                compaction_telemetry,
                knowledge_telemetry,
            ) = await _build_context(
                context_policy=overflow_policy,
                session_store=self._executor._session_store,
                session=self._session,
                agent_spec=_session_agent_spec(
                    registered_agent=self._registered_agent,
                    session=self._session,
                ),
                messages=messages,
                step=step,
                environment_name=self._environment_name,
                knowledge_store=self._knowledge_store,
                request_metadata=self._request_metadata,
                pressure_overhead=_context_pressure_overhead(
                    registered_provider=self._registered_provider,
                    registered_agent=self._registered_agent,
                    registered_environment=self._registered_environment,
                    structured_output=self._structured_output,
                    thinking=self._thinking,
                    step=step,
                ),
                count_input_tokens=self._context_input_token_counter(step=step),
                build_cache_prefix_request=self._cache_prefix_request_builder(step=step),
                run_compaction=run_automatic_compaction,
                force_bounded_compaction=True,
            )
        except ContextBuildError as exc:
            (
                context_failure_events,
                context_failure_persistence,
            ) = await self._context_build_failure_events(
                exc,
                published_compaction_attempt_ids=published_compaction_attempt_ids,
                compaction_completion_events=compaction_completion_events,
                compaction_start_event=(
                    compaction_start_events[0] if compaction_start_events else None
                ),
                compaction_started_published=any(
                    event.type == EventType.CONTEXT_COMPACTION_STARTED
                    for event in compaction_budget_events
                ),
            )
            for event in compaction_budget_events:
                yield event, None
            for event in context_failure_events:
                yield event, None
            if context_failure_persistence is not None:
                raise context_failure_persistence from exc
            if isinstance(exc.cause, _AutomaticCompactionBudgetReservationFailed):
                settlement_events, settlement_error = await settle_provider_dispatch()
                for event in settlement_events:
                    yield event, None
                if settlement_error is not None:
                    raise settlement_error from exc.cause
                async for event in self._stop_for_budget_reservation_failure(
                    result=exc.cause.result,
                    messages=messages,
                ):
                    yield event, None
                yield None, ModelStepFlowOutcome(stop_session=True)
                return
            if isinstance(exc.cause, _AutomaticCompactionAdmissionStopped):
                settlement_events, settlement_error = await settle_provider_dispatch()
                for event in settlement_events:
                    yield event, None
                if settlement_error is not None:
                    raise settlement_error from exc.cause
                admission_events = self._automatic_compaction_admission_events(
                    exc.cause,
                    messages=messages,
                )
                try:
                    async for event in admission_events:
                        yield event, None
                finally:
                    await _close_async_iterator(admission_events)
                yield None, ModelStepFlowOutcome(stop_session=True)
                return
            yield (
                await self._executor._event_writer.emit(
                    Event(
                        type=EventType.CONTEXT_OVERFLOW_FAILED,
                        session_id=self._session.id,
                        agent_name=self._registered_agent.spec.name,
                        environment_name=self._environment_name,
                        payload={
                            "step": step,
                            "phase": "context_build",
                            "error": str(exc.cause),
                            "error_type": type(exc.cause).__name__,
                            "policy": type(overflow_policy).__name__,
                        },
                    )
                ),
                None,
            )
            raise exc.cause from exc
        except BaseException as exc:
            await self._persist_context_build_termination_events(
                exc,
                published_compaction_attempt_ids=published_compaction_attempt_ids,
                compaction_completion_events=compaction_completion_events,
                compaction_start_event=(
                    compaction_start_events[0] if compaction_start_events else None
                ),
                compaction_started_published=any(
                    event.type == EventType.CONTEXT_COMPACTION_STARTED
                    for event in compaction_budget_events
                ),
                cancellation_requests_before_build=context_build_cancellation_requests,
            )
            raise

        context_success_events, context_success_persistence = await self._context_success_events(
            checkpoint_update=checkpoint_update,
            checkpoint_event_payload=checkpoint_event_payload,
            compaction_telemetry=compaction_telemetry,
            knowledge_telemetry=knowledge_telemetry,
            published_compaction_attempt_ids=published_compaction_attempt_ids,
            compaction_completion_events=compaction_completion_events,
            compaction_start_event=(
                compaction_start_events[0] if compaction_start_events else None
            ),
            compaction_started_published=any(
                event.type == EventType.CONTEXT_COMPACTION_STARTED
                for event in compaction_budget_events
            ),
        )
        for event in compaction_budget_events:
            yield event, None
        for event in context_success_events:
            yield event, None
        if context_success_persistence is not None:
            raise context_success_persistence
        await self._executor._session_control.raise_if_interrupted(self._session.id)
        if _has_provider_backed_context_compaction(compaction_telemetry):
            settlement_events, settlement_error = await settle_provider_dispatch()
            for event in settlement_events:
                yield event, None
            if settlement_error is not None:
                raise settlement_error
            should_stop: bool | None = None
            gate_events = self._post_compaction_gate(messages=messages)
            try:
                async for event, gate_outcome in gate_events:
                    if event is not None:
                        yield event, None
                    if gate_outcome is not None:
                        should_stop = gate_outcome
            finally:
                await _close_async_iterator(gate_events)
            if should_stop is None:
                raise RuntimeError("Post-compaction gate finished without an outcome.")
            if should_stop:
                yield None, ModelStepFlowOutcome(stop_session=True)
                return

        recovery_request = await self._executor.build_request(
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            context_messages=recovery_context_messages,
            structured_output=self._structured_output,
            thinking=self._thinking,
            step=step,
        )
        yield (
            await self._executor._event_writer.emit(
                Event(
                    type=EventType.CONTEXT_OVERFLOW_RECOVERING,
                    session_id=self._session.id,
                    agent_name=self._registered_agent.spec.name,
                    environment_name=self._environment_name,
                    payload={
                        "step": step,
                        "original_message_count": len(model_request.messages),
                        "recovery_message_count": len(recovery_request.messages),
                        "policy": type(overflow_policy).__name__,
                    },
                )
            ),
            None,
        )
        recovery_events = run_attempt(recovery_request)
        try:
            try:
                async for event, result in recovery_events:
                    yield (
                        event,
                        ModelStepFlowOutcome(assistant_step_result=result)
                        if result is not None
                        else None,
                    )
            except ModelContextOverflowError as exc:
                yield (
                    await self._executor._event_writer.emit(
                        Event(
                            type=EventType.CONTEXT_OVERFLOW_FAILED,
                            session_id=self._session.id,
                            agent_name=self._registered_agent.spec.name,
                            environment_name=self._environment_name,
                            payload=_context_overflow_event_payload(
                                exc,
                                step=step,
                                phase="recovery",
                                original_message_count=len(model_request.messages),
                                recovery_message_count=len(recovery_request.messages),
                            ),
                        )
                    ),
                    None,
                )
                raise
        finally:
            await _close_async_iterator(recovery_events)

    def _automatic_compaction_admission_events(
        self,
        rejection: _AutomaticCompactionAdmissionStopped,
        *,
        messages: list[Message],
    ) -> AsyncIterator[Event]:
        if rejection.budget_evaluation is not None:
            return self._executor._apply_budget_evaluation(
                ModelStepBudgetEvaluationRequest(
                    evaluation=rejection.budget_evaluation,
                    session=self._session,
                    registered_agent=self._registered_agent,
                    registered_environment=self._registered_environment,
                    environment_name=self._environment_name,
                    messages=messages,
                    run_started_at=self._run_started_at,
                    turn_usage_tracker=self._turn_usage_tracker,
                    active_run=self._active_run,
                )
            )
        if rejection.limit_evaluation is None:
            raise RuntimeError(
                "Automatic compaction admission rejection lost its evaluation."
            ) from rejection
        return self._executor._apply_limit_evaluation(
            ModelStepLimitEvaluationRequest(
                evaluation=rejection.limit_evaluation,
                session=self._session,
                registered_agent=self._registered_agent,
                registered_environment=self._registered_environment,
                environment_name=self._environment_name,
                messages=messages,
                run_started_at=self._run_started_at,
                turn_usage_tracker=self._turn_usage_tracker,
                active_run=self._active_run,
            )
        )

    async def _reconcile_automatic_compaction_events(
        self,
        events: list[Event],
        *,
        cancellation: asyncio.CancelledError | None,
        operation: str,
    ) -> tuple[
        list[bool] | None,
        BaseException | None,
        asyncio.CancelledError | None,
    ]:
        """Read durable event state without losing cancellation during the read."""

        async def reconcile() -> list[bool]:
            return [await self._executor._event_writer.is_persisted(event) for event in events]

        reconciliation_task = asyncio.create_task(reconcile())
        outcome = await await_shielded_task_outcome(
            reconciliation_task,
            cancellation=cancellation,
            timeout_s=_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S,
        )
        if outcome.timed_out:
            reconciliation_task.cancel()
            reconciliation_task.add_done_callback(_consume_detached_task_outcome)
            reconciliation_error = TimeoutError(
                f"{operation} reconciliation exceeded "
                f"{_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S:g} seconds."
            )
            return None, reconciliation_error, outcome.cancellation
        reconciliation_error = outcome.error
        if isinstance(reconciliation_error, asyncio.CancelledError):
            reconciliation_error = unexpected_child_cancellation_error(
                reconciliation_error,
                operation=f"{operation} reconciliation",
            )
        if outcome.result is None and reconciliation_error is None:
            reconciliation_error = RuntimeError(f"{operation} reconciliation returned no result.")
        return outcome.result, reconciliation_error, outcome.cancellation

    async def _fan_out_reconciled_automatic_compaction_events(
        self,
        events: list[Event],
        *,
        cancellation: asyncio.CancelledError | None,
        operation: str,
    ) -> tuple[BaseException | None, asyncio.CancelledError | None]:
        """Retry durable side effects with a bounded cancellation-safe wait."""

        fan_out_task = asyncio.create_task(self._executor._event_writer.fan_out_persisted(events))
        outcome = await await_shielded_task_outcome(
            fan_out_task,
            cancellation=cancellation,
            timeout_s=_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S,
        )
        if outcome.timed_out:
            fan_out_task.cancel()
            fan_out_task.add_done_callback(_consume_detached_task_outcome)
            return (
                TimeoutError(
                    f"{operation} side-effect delivery exceeded "
                    f"{_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S:g} seconds."
                ),
                outcome.cancellation,
            )
        error = outcome.error
        if isinstance(error, asyncio.CancelledError):
            error = unexpected_child_cancellation_error(
                error,
                operation=f"{operation} side-effect delivery",
            )
        return error, outcome.cancellation

    async def _persist_automatic_compaction_started(
        self,
        telemetry: ContextCompactionTelemetry,
        *,
        published_events: list[Event],
        start_events: list[Event],
    ) -> None:
        """Make the causal start durable before the first provider dispatch."""

        if telemetry.event_type != EventType.CONTEXT_COMPACTION_STARTED:
            raise TypeError("Automatic compaction start telemetry has the wrong event type.")
        if any(event.type == EventType.CONTEXT_COMPACTION_STARTED for event in published_events):
            return
        if start_events:
            event = start_events[0].model_copy(deep=True)
        else:
            event = _context_compaction_telemetry_event(
                telemetry=telemetry,
                session=self._session,
                registered_agent=self._registered_agent,
                environment_name=self._environment_name,
            )
            start_events.append(event.model_copy(deep=True))
        persistence_task = asyncio.create_task(
            self._executor._event_writer.emit_many(self._session.id, [event])
        )
        outcome = await await_shielded_task_outcome(
            persistence_task,
            timeout_s=_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S,
        )
        cancellation = outcome.cancellation
        if outcome.timed_out:
            persistence_task.cancel()
            persistence_task.add_done_callback(_consume_detached_task_outcome)
            publication_error: BaseException = TimeoutError(
                "Compaction start publication exceeded "
                f"{_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S:g} seconds."
            )
        else:
            publication_error = outcome.error
        try:
            if publication_error is not None:
                if isinstance(publication_error, asyncio.CancelledError) and cancellation is None:
                    raise unexpected_child_cancellation_error(
                        publication_error,
                        operation="Compaction start publication",
                    )
                raise publication_error
            persisted = outcome.result
            if persisted is None:
                raise RuntimeError("Compaction start publication returned no result.")
        except BaseException as publication_error:
            try:
                (
                    commit_states,
                    reconciliation_error,
                    cancellation,
                ) = await self._reconcile_automatic_compaction_events(
                    [event],
                    cancellation=cancellation,
                    operation="Compaction start publication",
                )
                if reconciliation_error is not None:
                    raise reconciliation_error
            except BaseException as reconciliation_error:
                publication_error.add_note(
                    "Compaction start publication reconciliation also failed: "
                    f"{type(reconciliation_error).__name__}: {reconciliation_error}"
                )
                if cancellation is not None:
                    cancellation.add_note(
                        "Compaction start publication and reconciliation also "
                        "failed during cancellation."
                    )
                    raise cancellation from publication_error
                raise publication_error from reconciliation_error
            if commit_states is None:
                raise AssertionError(
                    "Compaction start reconciliation lost its result."
                ) from publication_error
            if not commit_states[0]:
                if cancellation is not None:
                    cancellation.add_note(
                        "Compaction start could not be confirmed durable during cancellation."
                    )
                    raise cancellation from publication_error
                raise publication_error
            published_events.append(event.model_copy(deep=True))
            (
                fan_out_error,
                cancellation,
            ) = await self._fan_out_reconciled_automatic_compaction_events(
                [event],
                cancellation=cancellation,
                operation="Compaction start publication",
            )
            if fan_out_error is not None:
                publication_error.add_note(
                    "Committed compaction start side-effect delivery also failed: "
                    f"{type(fan_out_error).__name__}: {fan_out_error}"
                )
            publication_error.add_note(
                "Compaction start was durable; no provider dispatch followed the "
                "failed publication acknowledgement."
            )
            if cancellation is not None:
                cancellation.add_note("Compaction start was durable before cancellation.")
                raise cancellation from publication_error
            raise publication_error
        published_events.extend(persisted)
        if cancellation is not None:
            raise cancellation

    async def _persist_automatic_compaction_completions(
        self,
        payloads: list[dict[str, Any]],
        *,
        published_attempt_ids: set[str],
        published_events: list[Event],
        completion_events: dict[str, Event],
    ) -> None:
        """Commit finalized provider evidence before another compactor dispatch."""

        pending: list[tuple[str, Event]] = []
        for payload in payloads:
            attempt_id = payload.get(_COMPACTION_ATTEMPT_ID_KEY)
            if type(attempt_id) is not str:
                raise RuntimeError("Compaction completion evidence lost its attempt identity.")
            if attempt_id in published_attempt_ids:
                continue
            event = completion_events.get(attempt_id)
            if event is None:
                event = _context_compaction_telemetry_event(
                    telemetry=ContextCompactionTelemetry(
                        event_type=EventType.MODEL_COMPLETED,
                        payload=payload,
                    ),
                    session=self._session,
                    registered_agent=self._registered_agent,
                    environment_name=self._environment_name,
                )
                completion_events[attempt_id] = event.model_copy(deep=True)
            pending.append(
                (
                    attempt_id,
                    event.model_copy(deep=True),
                )
            )
        if not pending:
            return

        events = [event for _attempt_id, event in pending]
        persistence_task = asyncio.create_task(
            self._executor._event_writer.persist_many(self._session.id, events)
        )
        outcome = await await_shielded_task_outcome(
            persistence_task,
            timeout_s=_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S,
        )
        cancellation = outcome.cancellation
        if outcome.timed_out:
            persistence_task.cancel()
            # Only this physical write receives unbounded ownership. Sink and
            # budget side-effect delivery happens separately below and remains
            # bounded because the durable handoff can recover it after restart.
            drain_outcome = await await_shielded_task_outcome(
                persistence_task,
                cancellation=cancellation,
            )
            cancellation = drain_outcome.cancellation
            publication_error: BaseException = TimeoutError(
                "Compaction completion publication exceeded "
                f"{_CONTEXT_EVENT_STORE_WAIT_TIMEOUT_S:g} seconds."
            )
        else:
            publication_error = outcome.error
        try:
            if publication_error is not None:
                if isinstance(publication_error, asyncio.CancelledError) and cancellation is None:
                    raise unexpected_child_cancellation_error(
                        publication_error,
                        operation="Compaction completion publication",
                    )
                raise publication_error
            persisted = outcome.result
            if persisted is None:
                raise RuntimeError("Compaction completion publication returned no result.")
        except BaseException as publication_error:
            try:
                (
                    commit_states,
                    reconciliation_error,
                    cancellation,
                ) = await self._reconcile_automatic_compaction_events(
                    events,
                    cancellation=cancellation,
                    operation="Compaction completion publication",
                )
                if reconciliation_error is not None:
                    raise reconciliation_error
            except BaseException as reconciliation_error:
                publication_error.add_note(
                    "Compaction completion publication reconciliation also failed: "
                    f"{type(reconciliation_error).__name__}: {reconciliation_error}"
                )
                if cancellation is not None:
                    cancellation.add_note(
                        "Compaction completion publication and reconciliation also "
                        "failed during cancellation."
                    )
                    raise cancellation from publication_error
                raise publication_error from reconciliation_error
            if commit_states is None:
                raise AssertionError(
                    "Compaction completion reconciliation lost its result."
                ) from publication_error
            if not all(commit_states):
                if any(commit_states):
                    publication_error.add_note(
                        "The event store violated atomic compaction completion publication."
                    )
                if cancellation is not None:
                    cancellation.add_note(
                        "Compaction completion evidence could not be confirmed durable "
                        "during cancellation."
                    )
                    raise cancellation from publication_error
                raise publication_error
            # The provider evidence reached the durable handoff even though the
            # publication acknowledgement or a downstream side effect failed.
            # Remember it before propagating the failure so no failure path can
            # publish a duplicate completion and no retry can dispatch again.
            published_attempt_ids.update(attempt_id for attempt_id, _event in pending)
            published_events.extend(event.model_copy(deep=True) for event in events)
            (
                fan_out_error,
                cancellation,
            ) = await self._fan_out_reconciled_automatic_compaction_events(
                events,
                cancellation=cancellation,
                operation="Compaction completion publication",
            )
            if fan_out_error is not None:
                publication_error.add_note(
                    "Committed compaction completion side-effect delivery also failed: "
                    f"{type(fan_out_error).__name__}: {fan_out_error}"
                )
            publication_error.add_note(
                "Compaction completion evidence was durable; the operation will "
                "fail closed without another provider dispatch."
            )
            if cancellation is not None:
                cancellation.add_note(
                    "Compaction completion evidence was durable before cancellation."
                )
                raise cancellation from publication_error
            raise publication_error
        published_attempt_ids.update(attempt_id for attempt_id, _event in pending)
        published_events.extend(persisted)
        (
            fan_out_error,
            cancellation,
        ) = await self._fan_out_reconciled_automatic_compaction_events(
            events,
            cancellation=cancellation,
            operation="Compaction completion publication",
        )
        if cancellation is not None:
            if fan_out_error is not None:
                cancellation.add_note(
                    "Committed compaction completion side-effect delivery also failed "
                    f"during cancellation: {type(fan_out_error).__name__}: "
                    f"{fan_out_error}"
                )
                raise cancellation from fan_out_error
            raise cancellation
        if fan_out_error is not None:
            raise fan_out_error

    async def _emit_context_events_reconciling_late_start(
        self,
        events: list[Event],
        *,
        compaction_start_event: Event | None,
    ) -> list[Event]:
        """Persist a batch without duplicating a concurrently committed start."""

        try:
            return await self._executor._event_writer.emit_many(self._session.id, events)
        except BaseException as publication_error:
            if (
                not isinstance(publication_error, ValueError)
                or compaction_start_event is None
                or all(event.id != compaction_start_event.id for event in events)
            ):
                raise
            try:
                start_durable = await self._executor._event_writer.is_persisted(
                    compaction_start_event
                )
                remaining = [event for event in events if event.id != compaction_start_event.id]
                remaining_states = [
                    await self._executor._event_writer.is_persisted(event) for event in remaining
                ]
            except BaseException as reconciliation_error:
                publication_error.add_note(
                    "Context start conflict reconciliation also failed: "
                    f"{type(reconciliation_error).__name__}: {reconciliation_error}"
                )
                raise publication_error from reconciliation_error
            if not start_durable:
                raise
            if any(remaining_states):
                if not all(remaining_states):
                    publication_error.add_note(
                        "The context store violated atomic event-batch publication."
                    )
                    raise publication_error
                persisted_remaining = [event.model_copy(deep=True) for event in remaining]
            else:
                try:
                    persisted_remaining = await self._executor._event_writer.persist_many(
                        self._session.id,
                        remaining,
                    )
                except BaseException as retry_error:
                    retry_error.add_note(
                        "Context event publication retried after the original compaction "
                        "start committed concurrently."
                    )
                    raise retry_error from publication_error
            reconciled = [compaction_start_event.model_copy(deep=True), *persisted_remaining]
            await self._executor._event_writer.fan_out_persisted(reconciled)
            return [
                next(event for event in reconciled if event.id == requested.id).model_copy(
                    deep=True
                )
                for requested in events
            ]

    async def _persist_context_events(
        self,
        *,
        compaction_telemetry: list[ContextCompactionTelemetry],
        knowledge_telemetry: list[ContextKnowledgeTelemetry],
        checkpoint_update: dict[str, Any] | None,
        checkpoint_event_payload: dict[str, Any] | None,
        published_compaction_attempt_ids: set[str],
        compaction_completion_events: dict[str, Event],
        compaction_start_event: Event | None,
        compaction_started_published: bool,
        checkpoint_invariant_cause: BaseException | None = None,
    ) -> tuple[list[Event], BaseException | None]:
        """Persist one context outcome completely before exposing its first event."""

        reconciled_start_events: list[Event] = []
        compaction_start_durable = compaction_started_published
        if not compaction_start_durable and compaction_start_event is not None:
            (
                commit_states,
                reconciliation_error,
                cancellation,
            ) = await self._reconcile_automatic_compaction_events(
                [compaction_start_event],
                cancellation=None,
                operation="Compaction start cleanup",
            )
            if cancellation is not None:
                if reconciliation_error is not None:
                    raise cancellation from reconciliation_error
                raise cancellation
            if reconciliation_error is not None:
                return [], reconciliation_error
            if commit_states is None:
                return [], RuntimeError(
                    "Compaction start cleanup reconciliation returned no result."
                )
            compaction_start_durable = commit_states[0]
            if compaction_start_durable:
                reconciled_start_events.append(compaction_start_event.model_copy(deep=True))

        prepared_events: list[Event] = []
        for telemetry in compaction_telemetry:
            if (
                telemetry.event_type == EventType.MODEL_COMPLETED
                and telemetry.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                in published_compaction_attempt_ids
            ) or (
                telemetry.event_type == EventType.CONTEXT_COMPACTION_STARTED
                and compaction_start_durable
            ):
                continue
            compaction_attempt_id = telemetry.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
            event = (
                compaction_completion_events.get(compaction_attempt_id)
                if telemetry.event_type == EventType.MODEL_COMPLETED
                and type(compaction_attempt_id) is str
                else (
                    compaction_start_event
                    if telemetry.event_type == EventType.CONTEXT_COMPACTION_STARTED
                    else None
                )
            )
            if event is None:
                event = _context_compaction_telemetry_event(
                    telemetry=telemetry,
                    session=self._session,
                    registered_agent=self._registered_agent,
                    environment_name=self._environment_name,
                )
                if (
                    telemetry.event_type == EventType.MODEL_COMPLETED
                    and type(compaction_attempt_id) is str
                ):
                    compaction_completion_events[compaction_attempt_id] = event.model_copy(
                        deep=True
                    )
            prepared_events.append(event.model_copy(deep=True))
        prepared_events.extend(
            _context_knowledge_telemetry_event(
                telemetry=telemetry,
                session=self._session,
                registered_agent=self._registered_agent,
                environment_name=self._environment_name,
            )
            for telemetry in knowledge_telemetry
        )

        async def persist() -> tuple[list[Event], BaseException | None]:
            if checkpoint_event_payload is None:
                persisted = await self._emit_context_events_reconciling_late_start(
                    prepared_events,
                    compaction_start_event=compaction_start_event,
                )
                if reconciled_start_events:
                    await self._executor._event_writer.fan_out_persisted(reconciled_start_events)
                return [*reconciled_start_events, *persisted], None
            if checkpoint_update is None:
                error = RuntimeError("Context checkpoint event payload requires checkpoint state.")
                if checkpoint_invariant_cause is not None:
                    error.__cause__ = checkpoint_invariant_cause
                return [], error

            checkpoint_event = Event(
                type=EventType.SESSION_CHECKPOINTED,
                session_id=self._session.id,
                agent_name=self._registered_agent.spec.name,
                environment_name=self._environment_name,
                payload=checkpoint_event_payload,
            )
            atomic_events = [*prepared_events, checkpoint_event]
            checkpoint_transform = self._executor._checkpoint_transform(checkpoint_update)
            try:
                await self._executor._session_store.publish_checkpoint_and_events(
                    self._session.id,
                    checkpoint_transform=checkpoint_transform,
                    events=atomic_events,
                )
            except BaseException as publication_error:
                try:
                    event_commit_states = [
                        await self._executor._event_writer.is_persisted(event)
                        for event in atomic_events
                    ]
                    events_committed = all(event_commit_states)
                    durable_checkpoint = await self._executor._session_store.load_checkpoint(
                        self._session.id
                    )
                    durable_session = await self._executor._session_store.load(self._session.id)
                    expected_checkpoint = (
                        None
                        if durable_checkpoint is None or durable_session is None
                        else checkpoint_transform(durable_session, durable_checkpoint)
                    )
                    checkpoint_committed = (
                        durable_checkpoint is not None and expected_checkpoint == durable_checkpoint
                    )
                except BaseException as reconciliation_error:
                    publication_error.add_note(
                        "Context checkpoint publication reconciliation also failed: "
                        f"{type(reconciliation_error).__name__}: {reconciliation_error}"
                    )
                    return [], publication_error
                if not (events_committed and checkpoint_committed):
                    if events_committed != checkpoint_committed:
                        publication_error.add_note(
                            "The context store violated atomic checkpoint/event publication."
                        )
                    return [], publication_error
            await self._executor._event_writer.fan_out_persisted(
                [*reconciled_start_events, *atomic_events]
            )
            return [
                *(event.model_copy(deep=True) for event in reconciled_start_events),
                *(event.model_copy(deep=True) for event in atomic_events),
            ], None

        persistence_task = asyncio.create_task(persist())
        outcome = await await_shielded_task_outcome(persistence_task)
        cancellation = outcome.cancellation
        if cancellation is not None:
            if outcome.error is not None:
                cancellation.add_note(
                    "Context outcome persistence also failed during cancellation: "
                    f"{type(outcome.error).__name__}."
                )
                raise cancellation from outcome.error
            persisted_outcome = outcome.result
            if persisted_outcome is None:
                persistence_error = RuntimeError("Context persistence returned no result.")
                raise cancellation from persistence_error
            _, persistence_failure = persisted_outcome
            if persistence_failure is not None:
                cancellation.add_note(
                    "Context checkpoint persistence also failed during cancellation: "
                    f"{type(persistence_failure).__name__}."
                )
                raise cancellation from persistence_failure
            raise cancellation
        if outcome.error is not None:
            if isinstance(outcome.error, asyncio.CancelledError):
                return (
                    [],
                    unexpected_child_cancellation_error(
                        outcome.error,
                        operation="Context outcome persistence",
                    ),
                )
            return [], outcome.error
        if outcome.result is None:
            return [], RuntimeError("Context persistence returned no result.")
        persisted_events, persistence_failure = outcome.result
        if isinstance(persistence_failure, asyncio.CancelledError):
            persistence_failure = unexpected_child_cancellation_error(
                persistence_failure,
                operation="Context outcome persistence",
            )
        return persisted_events, persistence_failure

    async def _context_build_failure_events(
        self,
        error: ContextBuildError,
        *,
        published_compaction_attempt_ids: set[str],
        compaction_completion_events: dict[str, Event],
        compaction_start_event: Event | None,
        compaction_started_published: bool,
    ) -> tuple[list[Event], BaseException | None]:
        return await self._persist_context_events(
            compaction_telemetry=list(error.compaction_telemetry),
            knowledge_telemetry=list(error.knowledge_telemetry),
            checkpoint_update=error.checkpoint,
            checkpoint_event_payload=error.checkpoint_event_payload,
            published_compaction_attempt_ids=published_compaction_attempt_ids,
            compaction_completion_events=compaction_completion_events,
            compaction_start_event=compaction_start_event,
            compaction_started_published=compaction_started_published,
            checkpoint_invariant_cause=error,
        )

    async def _persist_context_build_termination_events(
        self,
        error: BaseException,
        *,
        published_compaction_attempt_ids: set[str],
        compaction_completion_events: dict[str, Event],
        compaction_start_event: Event | None,
        compaction_started_published: bool,
        cancellation_requests_before_build: int,
    ) -> None:
        """Persist completed compaction evidence without replacing a fatal signal."""

        telemetry = context_build_termination_compaction_telemetry(error)
        compaction_start_durable = compaction_started_published
        if not compaction_start_durable and compaction_start_event is not None:
            if isinstance(error, asyncio.CancelledError):
                # Remove only the cancellation already represented by ``error``.
                # The bounded reconciliation below must observe a genuinely later
                # Task.cancel() as a distinct signal rather than folding it into
                # the historical/provider cancellation.
                consume_pending_task_cancellation(
                    error,
                    preserve_requests=cancellation_requests_before_build,
                )
            (
                commit_states,
                reconciliation_error,
                reconciliation_cancellation,
            ) = await self._reconcile_automatic_compaction_events(
                [compaction_start_event],
                cancellation=None,
                operation="Compaction start termination cleanup",
            )
            if reconciliation_error is not None:
                error.add_note(
                    "Context compaction start reconciliation also failed during "
                    f"termination: {type(reconciliation_error).__name__}: "
                    f"{reconciliation_error}"
                )
            elif commit_states is not None:
                compaction_start_durable = commit_states[0]
            if reconciliation_cancellation is not None and reconciliation_cancellation is not error:
                raise BaseExceptionGroup(
                    "Context compaction start reconciliation observed a later cancellation.",
                    [error, reconciliation_cancellation],
                )
        unpublished_telemetry = [
            item
            for item in telemetry
            if not (
                (
                    item.event_type == EventType.MODEL_COMPLETED
                    and item.payload.get(_COMPACTION_ATTEMPT_ID_KEY)
                    in published_compaction_attempt_ids
                )
                or (
                    item.event_type == EventType.CONTEXT_COMPACTION_STARTED
                    and compaction_start_durable
                )
            )
        ]
        if not unpublished_telemetry:
            return

        async def persist() -> None:
            events = [
                (
                    compaction_completion_events[
                        cast("str", item.payload.get(_COMPACTION_ATTEMPT_ID_KEY))
                    ].model_copy(deep=True)
                    if item.event_type == EventType.MODEL_COMPLETED
                    and type(item.payload.get(_COMPACTION_ATTEMPT_ID_KEY)) is str
                    and item.payload.get(_COMPACTION_ATTEMPT_ID_KEY) in compaction_completion_events
                    else compaction_start_event.model_copy(deep=True)
                    if item.event_type == EventType.CONTEXT_COMPACTION_STARTED
                    and compaction_start_event is not None
                    else _context_compaction_telemetry_event(
                        telemetry=item,
                        session=self._session,
                        registered_agent=self._registered_agent,
                        environment_name=self._environment_name,
                    )
                )
                for item in unpublished_telemetry
            ]
            await self._emit_context_events_reconciling_late_start(
                events,
                compaction_start_event=compaction_start_event,
            )

        if isinstance(error, asyncio.CancelledError):
            # This cancellation has already crossed the context-build boundary and
            # remains authoritative. Clear only its task-level delivery state before
            # shielding cleanup so a genuinely later Task.cancel() is observed as a
            # distinct signal instead of being normalized back into ``error``.
            consume_pending_task_cancellation(
                error,
                preserve_requests=cancellation_requests_before_build,
            )
        task = asyncio.create_task(persist())
        outcome = await await_shielded_task_outcome(
            task,
            timeout_s=_CONTEXT_TERMINATION_PERSIST_TIMEOUT_S,
        )
        later_cancellation = (
            outcome.cancellation
            if outcome.cancellation is not None and outcome.cancellation is not error
            else None
        )
        if outcome.timed_out:
            task.add_done_callback(_consume_detached_task_outcome)
            task.cancel()
            error.add_note(
                "Context compaction termination telemetry persistence exceeded "
                f"{_CONTEXT_TERMINATION_PERSIST_TIMEOUT_S:g} seconds."
            )
            if later_cancellation is not None:
                raise BaseExceptionGroup(
                    "Context termination telemetry timed out after a later cancellation.",
                    [error, later_cancellation],
                )
            return
        if outcome.error is not None:
            error.add_note(
                "Context compaction termination telemetry also failed to persist: "
                f"{type(outcome.error).__name__}: {outcome.error}"
            )
            if later_cancellation is not None:
                raise BaseExceptionGroup(
                    "Context termination telemetry failed after a later cancellation.",
                    [error, outcome.error, later_cancellation],
                )
        elif later_cancellation is not None:
            raise BaseExceptionGroup(
                "Context termination telemetry completed after a later cancellation.",
                [error, later_cancellation],
            )

    async def _context_success_events(
        self,
        *,
        checkpoint_update: dict[str, Any] | None,
        checkpoint_event_payload: dict[str, Any] | None,
        compaction_telemetry: list[ContextCompactionTelemetry],
        knowledge_telemetry: list[ContextKnowledgeTelemetry],
        published_compaction_attempt_ids: set[str],
        compaction_completion_events: dict[str, Event],
        compaction_start_event: Event | None,
        compaction_started_published: bool,
    ) -> tuple[list[Event], BaseException | None]:
        return await self._persist_context_events(
            compaction_telemetry=compaction_telemetry,
            knowledge_telemetry=knowledge_telemetry,
            checkpoint_update=checkpoint_update,
            checkpoint_event_payload=checkpoint_event_payload,
            published_compaction_attempt_ids=published_compaction_attempt_ids,
            compaction_completion_events=compaction_completion_events,
            compaction_start_event=compaction_start_event,
            compaction_started_published=compaction_started_published,
        )

    async def _post_compaction_gate(
        self,
        *,
        messages: list[Message],
    ) -> AsyncIterator[tuple[Event | None, bool | None]]:
        budget_evaluation = await self._limit_gate.evaluate_budget(self._budget_policy)
        request = ModelStepBudgetEvaluationRequest(
            evaluation=budget_evaluation,
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            environment_name=self._environment_name,
            messages=messages,
            run_started_at=self._run_started_at,
            turn_usage_tracker=self._turn_usage_tracker,
            active_run=self._active_run,
        )
        budget_events = self._executor._apply_budget_evaluation(request)
        try:
            async for event in budget_events:
                yield event, None
        finally:
            await _close_async_iterator(budget_events)
        if budget_evaluation.check is not None:
            yield None, True
            return
        limit_evaluation = await self._limit_gate.evaluate_limits()
        request = ModelStepLimitEvaluationRequest(
            evaluation=limit_evaluation,
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            environment_name=self._environment_name,
            messages=messages,
            run_started_at=self._run_started_at,
            turn_usage_tracker=self._turn_usage_tracker,
            active_run=self._active_run,
        )
        limit_events = self._executor._apply_limit_evaluation(request)
        try:
            async for event in limit_events:
                yield event, None
        finally:
            await _close_async_iterator(limit_events)
        yield None, limit_evaluation.decision is not None

    async def _billing_identity_budget_gate(
        self,
        *,
        messages: list[Message],
        billing_identity: BillingIdentity | None,
    ) -> AsyncIterator[tuple[Event | None, bool | None]]:
        budget_evaluation = await self._limit_gate.evaluate_budget(
            self._budget_policy,
            billing_identity_state=resolved_billing_identity(billing_identity),
        )
        request = ModelStepBudgetEvaluationRequest(
            evaluation=budget_evaluation,
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            environment_name=self._environment_name,
            messages=messages,
            run_started_at=self._run_started_at,
            turn_usage_tracker=self._turn_usage_tracker,
            active_run=self._active_run,
        )
        budget_events = self._executor._apply_budget_evaluation(request)
        try:
            async for event in budget_events:
                yield event, None
        finally:
            await _close_async_iterator(budget_events)
        if budget_evaluation.check is not None:
            yield None, True
            return
        limit_evaluation = await self._limit_gate.evaluate_limits(
            billing_identity_state=resolved_billing_identity(billing_identity),
        )
        request = ModelStepLimitEvaluationRequest(
            evaluation=limit_evaluation,
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            environment_name=self._environment_name,
            messages=messages,
            run_started_at=self._run_started_at,
            turn_usage_tracker=self._turn_usage_tracker,
            active_run=self._active_run,
        )
        limit_events = self._executor._apply_limit_evaluation(request)
        try:
            async for event in limit_events:
                yield event, None
        finally:
            await _close_async_iterator(limit_events)
        yield None, limit_evaluation.decision is not None

    def _has_deferred_contextual_price(self) -> bool:
        return self._deferred_contextual_price

    async def _stop_for_budget_reservation_failure(
        self,
        *,
        result: BudgetReservationResult,
        messages: list[Message],
    ) -> AsyncIterator[Event]:
        request = ModelStepBudgetReservationFailureRequest(
            result=result,
            session=self._session,
            registered_agent=self._registered_agent,
            registered_environment=self._registered_environment,
            environment_name=self._environment_name,
            messages=messages,
            run_started_at=self._run_started_at,
            turn_usage_tracker=self._turn_usage_tracker,
            active_run=self._active_run,
        )
        terminal_events = self._executor._stop_for_budget_reservation_failure(request)
        try:
            async for event in terminal_events:
                yield event
        finally:
            await _close_async_iterator(terminal_events)

    def _context_input_token_counter(
        self,
        *,
        step: int,
    ) -> Callable[[list[Message]], Awaitable[int | None]]:
        async def count_input_tokens(context_messages: list[Message]) -> int | None:
            request = await self._executor.build_request(
                session=self._session,
                registered_agent=self._registered_agent,
                registered_environment=self._registered_environment,
                context_messages=copy_context_messages(context_messages),
                structured_output=self._structured_output,
                thinking=self._thinking,
                step=step,
            )
            result = await self._provider.count_input_tokens(request)
            return None if result is None else result.input_tokens

        return count_input_tokens

    def _cache_prefix_request_builder(
        self,
        *,
        step: int,
    ) -> Callable[[list[Message]], Awaitable[ModelRequest]]:
        async def build_cache_prefix_request(context_messages: list[Message]) -> ModelRequest:
            return await self._executor.build_request(
                session=self._session,
                registered_agent=self._registered_agent,
                registered_environment=self._registered_environment,
                context_messages=copy_context_messages(context_messages),
                structured_output=self._structured_output,
                thinking=self._thinking,
                step=step,
            )

        return build_cache_prefix_request

    async def _run_automatic_compaction_with_budget(
        self,
        *,
        compactor: ContextCompactor,
        compaction_request: CompactionRequest,
        execute: Callable[[], Awaitable[CompactionResult]],
        completed_payloads: Callable[[], list[dict[str, Any]]],
        budget_events: list[Event],
        messages: list[Message],
    ) -> CompactionResult:
        controller = self._executor._run_limit_controller
        all_limits = controller.provider_budget_limits(
            session=self._session,
            agent_name=self._registered_agent.spec.name,
            budget_policy=self._budget_policy,
            request_budget_limits=self._request_budget_limits,
        )
        if not all_limits and not self._limit_gate.has_run_limits():
            return await execute()
        strict_contextual_candidates = tuple(
            limit
            for limit in all_limits
            if limit.action == "interrupt"
            and not limit.allow_unpriced
            and any(price.pricing_context is not None for price in limit.pricing.prices)
        )
        try:
            identity = compactor._provider_budget_identity_for_request(compaction_request)
        except NotImplementedError as exc:
            raise RuntimeError(
                "Automatic provider-backed compaction under run or budget limits requires the "
                "ContextCompactor to declare provider_budget_identity(session), "
                "returning provider/model or None for deterministic execution."
            ) from exc
        uses_dispatch_boundary = compactor._uses_runtime_provider_dispatch_runner_for_request(
            compaction_request
        )
        if identity is None:
            if uses_dispatch_boundary:
                raise RuntimeError(
                    "Provider-backed compaction cannot declare a deterministic budget "
                    "identity under run or cost limits."
                )
            return await execute()
        if type(identity) is not tuple or len(identity) != 2:
            raise TypeError(
                "ContextCompactor.provider_budget_identity must return a "
                "(provider_name, model) tuple or None."
            )
        require_clean_nonblank(identity[0], "compactor_provider_name")
        require_clean_nonblank(identity[1], "compactor_model")
        pricing_provider_name = identity[0]
        declared_model = identity[1]
        contextual_limits = tuple(
            limit
            for limit in strict_contextual_candidates
            if has_deferred_contextual_price(
                limit.pricing,
                provider_name=pricing_provider_name,
                model=declared_model,
            )
        )
        limits = tuple(
            limit
            for limit in all_limits
            if limit.reservation is not None or limit in contextual_limits
        )
        if not uses_dispatch_boundary:
            raise RuntimeError(
                "Automatic provider-backed compaction under run or budget limits cannot safely run "
                f"opaque provider-backed compactor {type(compactor).__name__}: "
                "Cayu cannot admit each provider dispatch independently. Use an "
                "unmodified built-in provider compactor or remove the applicable "
                "run and budget limits."
            )

        policy_limits = budget_limits_for_session(
            policy=self._budget_policy,
            agent_name=self._registered_agent.spec.name,
            causal_budget_id=self._session.causal_budget_id,
        )
        dispatch_policy_limits = tuple(limit for limit in policy_limits if limit not in limits)
        dispatch_request_limits = tuple(
            limit for limit in self._request_budget_limits if limit not in limits
        )

        async def run_provider_dispatch(
            actual_provider: ModelProvider,
            actual_model: str,
            billing_identity: BillingIdentity | None,
            dispatch: Callable[[], Awaitable[tuple[str, dict[str, Any]]]],
        ) -> tuple[str, dict[str, Any]]:
            before_count = len(completed_payloads())

            def completion_events(payloads: list[dict[str, Any]]) -> list[Event]:
                return [
                    Event(
                        type=EventType.MODEL_COMPLETED,
                        session_id=self._session.id,
                        agent_name=self._registered_agent.spec.name,
                        environment_name=self._environment_name,
                        payload=payload,
                    )
                    for payload in payloads
                ]

            prior_completion_events = [
                event.model_copy(deep=True)
                for event in budget_events
                if event.type == EventType.MODEL_COMPLETED
            ]

            def completed_events() -> list[Event]:
                return completion_events(completed_payloads()[before_count:])

            actual_pricing_provider_name = require_clean_nonblank(
                actual_provider.billing_provider_name or actual_provider.name,
                "compactor_provider_name",
            )
            billing_identity_state = resolved_billing_identity(billing_identity)
            budget_evaluation = await self._limit_gate.evaluate_budget(
                BudgetPolicy(limits=dispatch_policy_limits),
                billing_identity_state=billing_identity_state,
                pricing_provider_name=actual_pricing_provider_name,
                model=actual_model,
                additional_usage_events=prior_completion_events,
            )
            if budget_evaluation.check is not None:
                raise _AutomaticCompactionAdmissionStopped(budget_evaluation=budget_evaluation)
            budget_events.extend(budget_evaluation.events)
            limit_evaluation = await self._limit_gate.evaluate_limits(
                billing_identity_state=billing_identity_state,
                pricing_provider_name=actual_pricing_provider_name,
                model=actual_model,
                additional_usage_events=prior_completion_events,
                budget_limits=dispatch_request_limits,
            )
            if limit_evaluation.decision is not None:
                raise _AutomaticCompactionAdmissionStopped(limit_evaluation=limit_evaluation)
            budget_events.extend(limit_evaluation.events)
            if not limits:
                return await dispatch()
            outcome = await controller.run_automatic_compaction_dispatch(
                dispatch,
                completed_events=completed_events,
                prior_completion_events=prior_completion_events,
                budget_limits=limits,
                session=self._session,
                agent_name=self._registered_agent.spec.name,
                environment_name=self._environment_name,
                provider_name=actual_pricing_provider_name,
                model=require_clean_nonblank(actual_model, "compactor_model"),
                billing_identity=billing_identity,
                pricing_provider_name=pricing_provider_name,
                authoritative_failure_types=(ContextBuildError,),
            )
            budget_events.extend(outcome.events)
            if isinstance(outcome, BudgetedOperationSucceeded):
                return cast("tuple[str, dict[str, Any]]", outcome.result)
            if isinstance(outcome, BudgetedOperationRejected):
                raise _AutomaticCompactionBudgetReservationFailed(outcome.failure)
            if outcome.cause is not None:
                raise outcome.error from outcome.cause
            raise outcome.error

        with _automatic_compaction_dispatch_runner_scope(run_provider_dispatch):
            return await execute()


def _session_agent_spec(
    *,
    registered_agent: runtime_records.RegisteredAgentState,
    session: Session,
) -> AgentSpec:
    return AgentSpec(
        name=registered_agent.spec.name,
        model=session.model,
        provider_name=session.provider_name,
        system_prompt=registered_agent.spec.system_prompt,
        metadata=copy_json_value(registered_agent.spec.metadata, "metadata"),
        provider_options=copy_json_value(
            registered_agent.spec.provider_options,
            "provider_options",
        ),
    )


def _context_pressure_overhead(
    *,
    registered_provider: runtime_records.RegisteredProvider,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    structured_output: StructuredOutputSpec | None,
    thinking: ThinkingConfig | None,
    step: int,
) -> ContextPressureOverhead:
    profile = copy_model_context_pressure_profile(
        registered_provider.provider.context_pressure_profile
    )
    tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": deepcopy(tool.schema),
        }
        for tool in registered_agent.tools.values()
    ]
    structured_output_instruction: str | None = None
    if (
        structured_output is not None
        and structured_output.strategy == StructuredOutputStrategy.TOOL
    ):
        tools.append(structured_output_tool_spec(structured_output))
        structured_output_instruction = structured_output_tool_instruction(structured_output)

    request_options: dict[str, Any] = {
        **copy_json_value(
            registered_agent.spec.provider_options,
            "provider_options",
        ),
        "agent_metadata": deepcopy(registered_agent.spec.metadata),
        "environment_metadata": (
            deepcopy(registered_environment.spec.metadata)
            if registered_environment is not None
            else {}
        ),
        "step": step,
        "structured_output": (
            structured_output_spec_payload(structured_output)
            if structured_output is not None
            else None
        ),
    }
    if thinking is not None:
        request_options["thinking"] = thinking_config_payload(thinking)
    return ContextPressureOverhead(
        tools=tools,
        structured_output_instruction=structured_output_instruction,
        request_options=request_options,
        image_min_tokens=profile.image_min_tokens,
        document_min_tokens=profile.document_min_tokens,
        document_bytes_per_token=profile.document_bytes_per_token,
        tool_schema_chars_per_token=profile.tool_schema_chars_per_token,
    )


async def _build_context(
    *,
    context_policy: ContextPolicy,
    session_store: SessionStore,
    session: Session,
    agent_spec: AgentSpec,
    messages: list[Message],
    step: int,
    environment_name: str | None,
    knowledge_store: Any,
    request_metadata: dict[str, Any],
    pressure_overhead: ContextPressureOverhead,
    count_input_tokens: Callable[[list[Message]], Awaitable[int | None]] | None,
    build_cache_prefix_request: Callable[[list[Message]], Awaitable[ModelRequest]] | None,
    run_compaction: _AutomaticCompactionRunner | None = None,
    force_bounded_compaction: bool = False,
) -> tuple[
    list[Message],
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[ContextCompactionTelemetry],
    list[ContextKnowledgeTelemetry],
]:
    context_usage = await _context_usage_state_for_session(
        session_store=session_store,
        session_id=session.id,
    )
    context_usage = estimate_context_pressure(
        usage=context_usage,
        messages=messages,
        image_min_tokens=pressure_overhead.image_min_tokens,
        document_min_tokens=pressure_overhead.document_min_tokens,
        document_bytes_per_token=pressure_overhead.document_bytes_per_token,
    )
    request = ContextRequest(
        session=session.model_copy(deep=True),
        agent=agent_spec.model_copy(deep=True),
        messages=[message.model_copy(deep=True) for message in messages],
        step=step,
        environment_name=environment_name,
        knowledge_store=knowledge_store,
        metadata=copy_json_value(request_metadata, "metadata"),
        context_usage=context_usage,
        pressure_overhead=pressure_overhead,
        count_input_tokens=count_input_tokens,
        build_cache_prefix_request=build_cache_prefix_request,
        force_bounded_compaction=force_bounded_compaction,
    )
    if isinstance(context_policy, RuntimeManagedContextPolicy):
        checkpoint = await session_store.load_checkpoint(session.id)
        with _automatic_compaction_runner_scope(run_compaction):
            result = await context_policy.build_with_checkpoint(
                request,
                checkpoint=checkpoint,
            )
        return (
            copy_context_messages(result.messages),
            copy_json_value(result.checkpoint, "checkpoint"),
            result.checkpoint_event_payload,
            [telemetry.model_copy(deep=True) for telemetry in result.compaction_telemetry],
            [telemetry.model_copy(deep=True) for telemetry in result.knowledge_telemetry],
        )

    result = await context_policy.build(request)
    return copy_context_messages(result), None, None, [], []


async def _context_usage_state_for_session(
    *,
    session_store: SessionStore,
    session_id: str,
) -> ContextUsageState:
    before_sequence: int | None = None
    page_size = 1
    while True:
        records = await session_store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.MODEL_COMPLETED,
                before_sequence=before_sequence,
                limit=page_size,
                order_by=EventOrder.SEQUENCE_DESC,
            )
        )
        if not records:
            return ContextUsageState()
        for record in records:
            payload = record.event.payload
            is_compaction_completion = (
                payload.get("purpose") == "context_compaction"
                and _transcript_cursor_from_model_completed_event(record.event) is None
            )
            if not is_compaction_completion:
                return _context_usage_state_from_model_completed_event(record.event)
        before_sequence = records[-1].sequence
        page_size = _CONTEXT_USAGE_AUXILIARY_PAGE_SIZE


def _context_usage_state_from_model_completed_event(event: Event) -> ContextUsageState:
    if event.type != EventType.MODEL_COMPLETED:
        return ContextUsageState()
    metrics = usage_metrics_from_event_payload(event.payload)
    if metrics is None:
        return ContextUsageState(
            last_transcript_cursor=_transcript_cursor_from_model_completed_event(event)
        )
    return ContextUsageState(
        last_input_tokens=metrics.input_tokens,
        last_output_tokens=metrics.output_tokens,
        last_total_tokens=metrics.total_tokens,
        last_transcript_cursor=_transcript_cursor_from_model_completed_event(event),
        last_context_overhead_input_tokens=(
            _context_overhead_input_tokens_from_model_completed_event(event)
        ),
        last_provider_name=metrics.provider_name,
        last_requested_model=metrics.requested_model,
        last_model=metrics.model,
    )


def _transcript_cursor_from_model_completed_event(event: Event) -> int | None:
    cursor = event.payload.get("transcript_cursor")
    if type(cursor) is not int or cursor < 0:
        return None
    return cursor


def _context_overhead_input_tokens_from_model_completed_event(event: Event) -> int | None:
    pressure = event.payload.get("context_pressure")
    if type(pressure) is not dict:
        return None
    tokens = pressure.get("estimated_request_overhead_input_tokens")
    if type(tokens) is not int or tokens < 0:
        return None
    return tokens


def _has_provider_backed_context_compaction(
    compaction_telemetry: list[ContextCompactionTelemetry],
) -> bool:
    return any(
        telemetry.event_type == EventType.MODEL_COMPLETED for telemetry in compaction_telemetry
    )


def _context_compaction_telemetry_event(
    *,
    telemetry: ContextCompactionTelemetry,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
) -> Event:
    if type(telemetry) is not ContextCompactionTelemetry:
        raise TypeError(
            "Context compaction telemetry must be ContextCompactionTelemetry instances."
        )
    sanitized = sanitize_context_compaction_telemetry(telemetry)
    return Event(
        type=sanitized.event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=copy_json_value(sanitized.payload, "payload"),
    )


def _context_knowledge_telemetry_event(
    *,
    telemetry: ContextKnowledgeTelemetry,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
) -> Event:
    if type(telemetry) is not ContextKnowledgeTelemetry:
        raise TypeError("Context knowledge telemetry must be ContextKnowledgeTelemetry instances.")
    return Event(
        type=telemetry.event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=copy_json_value(telemetry.payload, "payload"),
    )


def _context_overflow_event_payload(
    error: ModelContextOverflowError,
    *,
    step: int,
    phase: str,
    original_message_count: int,
    recovery_message_count: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "step": step,
        "phase": require_clean_nonblank(phase, "phase"),
        "error": str(error),
        "error_type": type(error).__name__,
        "provider": error.provider,
        "original_message_count": original_message_count,
    }
    if error.status_code is not None:
        payload["status_code"] = error.status_code
    if error.error_type is not None:
        payload["provider_error_type"] = error.error_type
    if error.error_code is not None:
        payload["provider_error_code"] = error.error_code
    if error.request_id is not None:
        payload["request_id"] = error.request_id
    if recovery_message_count is not None:
        payload["recovery_message_count"] = recovery_message_count
    return payload


class _FileAttachmentUnavailable(RuntimeError):
    """An attachment reference cannot be resolved in its declared scope."""


async def _resolved_file_attachments(
    *,
    messages: list[Message],
    session: Session,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    max_file_attachment_bytes: int,
    max_total_file_attachment_bytes: int,
    max_file_attachments_per_request: int,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Resolve model-facing files while failing open only for prompt files.

    Missing prompt files become visible text notes so a stale prompt reference
    cannot brick a session forever. Tool-result files remain fail-closed because
    silently omitting tool evidence would let the model answer from incomplete
    state. A reference used by both paths therefore remains fail-closed.
    """
    attachment_refs, prompt_file_artifact_ids, tool_result_artifact_ids = _file_attachment_refs(
        messages
    )
    if not attachment_refs:
        return {}, set()
    if len(attachment_refs) > max_file_attachments_per_request:
        raise RuntimeError(
            "File attachment count exceeds the runtime attachment limit: "
            f"{len(attachment_refs)} > {max_file_attachments_per_request}"
        )
    artifact_store = (
        None
        if registered_environment is None
        else registered_environment.environment.artifact_store
    )
    if artifact_store is None:
        raise RuntimeError("File attachments require an artifact store.")

    environment_name = None if registered_environment is None else registered_environment.spec.name
    resolved: dict[str, dict[str, Any]] = {}
    unresolvable_prompt_ids: set[str] = set()
    total_attachment_bytes = 0
    for attachment in attachment_refs:
        if attachment.size_bytes > max_file_attachment_bytes:
            raise RuntimeError(
                "File attachment exceeds the runtime attachment byte limit: "
                f"{attachment.artifact_id}"
            )
        total_attachment_bytes += attachment.size_bytes
        if total_attachment_bytes > max_total_file_attachment_bytes:
            raise RuntimeError("File attachments exceed the runtime total attachment byte limit.")
        if attachment.artifact_id in resolved or attachment.artifact_id in unresolvable_prompt_ids:
            continue
        try:
            result = copy_artifact_read_result(
                await artifact_store.read_bytes(
                    attachment.artifact_id,
                    max_bytes=attachment.size_bytes,
                ),
                expected_artifact_id=attachment.artifact_id,
                max_content_bytes=attachment.size_bytes,
            )
            artifact = result.metadata
            if artifact.scope.value == "session" and artifact.session_id != session.id:
                raise _FileAttachmentUnavailable(
                    "File attachment is not available in this session."
                )
            if (
                artifact.scope.value == "environment"
                and artifact.environment_name != environment_name
            ):
                raise _FileAttachmentUnavailable(
                    "File attachment is not available in this environment."
                )
            if artifact.content_type != attachment.content_type:
                raise _FileAttachmentUnavailable(
                    "File attachment content type changed before provider request."
                )
            if artifact.size_bytes != attachment.size_bytes:
                raise _FileAttachmentUnavailable(
                    "File attachment size changed before provider request."
                )
        except (FileNotFoundError, InvalidArtifactIdError, _FileAttachmentUnavailable):
            is_exclusively_prompt = (
                attachment.artifact_id in prompt_file_artifact_ids
                and attachment.artifact_id not in tool_result_artifact_ids
            )
            if not is_exclusively_prompt:
                raise
            unresolvable_prompt_ids.add(attachment.artifact_id)
            continue
        resolved[attachment.artifact_id] = resolved_file_attachment(attachment, result)
    return resolved, unresolvable_prompt_ids


def _file_attachment_refs(
    messages: list[Message],
) -> tuple[tuple[FileAttachment, ...], set[str], set[str]]:
    """Collect ordered references and their prompt/tool-result provenance."""
    refs: dict[str, FileAttachment] = {}
    ordered_refs: list[FileAttachment] = []
    prompt_artifact_ids: set[str] = set()
    tool_result_artifact_ids: set[str] = set()
    for message in messages:
        for part in message.content:
            if type(part) is ToolResultPart:
                payloads: list[dict[str, Any]] = part.artifacts
                origin_ids = tool_result_artifact_ids
            elif type(part) is FilePart:
                payloads = [part.attachment]
                origin_ids = prompt_artifact_ids
            else:
                continue
            for payload in payloads:
                attachment = file_attachment_from_payload(payload)
                if attachment is None:
                    continue
                origin_ids.add(attachment.artifact_id)
                existing = refs.get(attachment.artifact_id)
                if existing is not None and not _same_file_attachment_ref(existing, attachment):
                    raise RuntimeError(
                        "Conflicting file attachment references for artifact: "
                        f"{attachment.artifact_id}"
                    )
                refs[attachment.artifact_id] = attachment
                ordered_refs.append(attachment)
    return tuple(ordered_refs), prompt_artifact_ids, tool_result_artifact_ids


def _same_file_attachment_ref(left: FileAttachment, right: FileAttachment) -> bool:
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def _validate_stream_event(value: object) -> ModelStreamEvent:
    if type(value) is not ModelStreamEvent:
        raise TypeError("Model providers must yield ModelStreamEvent instances.")
    return copy_model_stream_event(value)


def _copy_model_request_for_counting(request: ModelRequest) -> ModelRequest:
    if type(request) is not ModelRequest:
        raise TypeError("request must be a ModelRequest.")
    return ModelRequest(
        model=request.model,
        messages=request.messages,
        tools=request.tools,
        options=request.options,
    )


def _context_count_base_payload(
    *,
    model_request: ModelRequest,
    provider_name: str,
    step: int,
    attempt: int,
    max_attempts: int,
    observation_id: str,
) -> dict[str, Any]:
    roles = [
        message.role.value if isinstance(message.role, MessageRole) else str(message.role)
        for message in model_request.messages
    ]
    return {
        "model": model_request.model,
        "provider": provider_name,
        "step": step,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "observation_id": observation_id,
        "messages": {"count": len(model_request.messages), "roles": roles},
        "tools": {"count": len(model_request.tools)},
        "options": {"keys": sorted(model_request.options.keys())},
    }


def _context_count_reconciled_event(
    model_completed_event: Event,
    *,
    observation: _ContextCountObservation,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_provider: runtime_records.RegisteredProvider,
    environment_name: str | None,
    step: int,
    attempt: int,
    max_attempts: int,
) -> Event:
    if model_completed_event.type != EventType.MODEL_COMPLETED:
        raise ValueError("Context count reconciliation requires a model.completed event.")
    actual_input_tokens = _actual_input_tokens_from_completed_event(model_completed_event)
    estimated_input_tokens = observation.result.input_tokens
    delta_tokens = (
        None
        if actual_input_tokens is None or estimated_input_tokens is None
        else actual_input_tokens - estimated_input_tokens
    )
    relative_error = (
        None
        if delta_tokens is None or actual_input_tokens is None or actual_input_tokens <= 0
        else delta_tokens / actual_input_tokens
    )
    return Event(
        type=EventType.CONTEXT_COUNT_RECONCILED,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload={
            "model": session.model,
            "provider": registered_provider.name,
            "step": step,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "observation_id": observation.observation_id,
            "pre_call_count": observation.result.model_dump(mode="json"),
            "actual_input_tokens": actual_input_tokens,
            "delta_tokens": delta_tokens,
            "relative_error": relative_error,
            "reconciled": actual_input_tokens is not None and estimated_input_tokens is not None,
        },
    )


def _context_pressure_reconciled_event(
    model_completed_event: Event,
    *,
    observation: _ContextPressureObservation,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_provider: runtime_records.RegisteredProvider,
    environment_name: str | None,
    step: int,
    attempt: int,
    max_attempts: int,
) -> Event:
    if model_completed_event.type != EventType.MODEL_COMPLETED:
        raise ValueError("Context pressure reconciliation requires a model.completed event.")
    actual_input_tokens = _actual_input_tokens_from_completed_event(model_completed_event)
    estimated_input_tokens = observation.estimate.estimated_context_input_tokens
    delta_tokens = (
        None if actual_input_tokens is None else actual_input_tokens - estimated_input_tokens
    )
    relative_error = (
        None
        if delta_tokens is None or actual_input_tokens is None or actual_input_tokens <= 0
        else delta_tokens / actual_input_tokens
    )
    return Event(
        type=EventType.CONTEXT_PRESSURE_RECONCILED,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload={
            "model": session.model,
            "provider": registered_provider.name,
            "step": step,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "observation_id": observation.observation_id,
            "pre_call_estimate": observation.estimate.model_dump(mode="json"),
            "actual_input_tokens": actual_input_tokens,
            "delta_tokens": delta_tokens,
            "relative_error": relative_error,
            "reconciled": actual_input_tokens is not None,
        },
    )


def _actual_input_tokens_from_completed_event(event: Event) -> int | None:
    usage_metrics = event.payload.get("usage_metrics")
    if type(usage_metrics) is not dict:
        return None
    input_tokens = usage_metrics.get("input_tokens")
    if type(input_tokens) is not int or input_tokens < 0:
        return None
    return input_tokens


def _model_stream_event_to_runtime_event(
    stream_event: ModelStreamEvent,
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    provider_name: str | None,
    step: int,
    attempt: int,
    max_attempts: int,
    classification: dict[str, str] | None = None,
    context_pressure_estimate: ContextPressureEstimate | None = None,
    transcript_cursor_after_completion: int | None = None,
    usage_dialect: str | None = None,
    billing_identity: BillingIdentity | None = None,
) -> Event:
    if type(stream_event) is not ModelStreamEvent:
        raise TypeError("Model stream events must be ModelStreamEvent instances.")
    if stream_event.type == ModelStreamEventType.TEXT_DELTA:
        event_type = EventType.MODEL_TEXT_DELTA
        payload = {"delta": stream_event.delta}
    elif stream_event.type == ModelStreamEventType.THINKING:
        event_type = EventType.MODEL_THINKING_DELTA
        payload = {"delta": stream_event.delta}
    elif stream_event.type == ModelStreamEventType.COMPLETED:
        payload = transcript_helpers.model_completed_event_payload(stream_event.payload)
        resolved_model = _payload_model(payload, fallback=session.model)
        payload["requested_model"] = session.model
        # Billing identity is runtime-owned. Providers may report completion facts
        # consumed by their hook, but cannot inject an identity in the raw payload.
        payload.pop("billing_identity", None)
        if billing_identity is not None:
            payload["billing_identity"] = billing_identity.model_dump(mode="json")
        completion = _stream_event_completion(stream_event)
        payload["completion"] = {
            "finish_reason": completion.finish_reason.value,
            "raw_finish_reason": completion.raw_finish_reason,
            "status": completion.status,
        }
        if classification is not None:
            payload["step_classification"] = classification
        metrics = usage_metrics_payload(
            normalize_usage_metrics(
                provider_name=provider_name,
                model=resolved_model,
                requested_model=session.model,
                raw_usage=payload.get("usage"),
                usage_dialect=usage_dialect,
                billing_identity=billing_identity,
            )
        )
        if metrics is not None:
            # The event-level identity is authoritative. Keeping a second nested
            # copy would let an untrusted provider payload create conflicting
            # accounting evidence when normalized usage is unavailable.
            metrics.pop("billing_identity", None)
            payload["usage_metrics"] = metrics
        if context_pressure_estimate is not None:
            payload["context_pressure"] = {
                "estimated_tool_schema_input_tokens": (
                    context_pressure_estimate.estimated_tool_schema_input_tokens
                ),
                "estimated_structured_output_input_tokens": (
                    context_pressure_estimate.estimated_structured_output_input_tokens
                ),
                "estimated_request_options_input_tokens": (
                    context_pressure_estimate.estimated_request_options_input_tokens
                ),
                "estimated_request_overhead_input_tokens": (
                    context_pressure_estimate.estimated_request_overhead_input_tokens
                ),
            }
        if transcript_cursor_after_completion is not None:
            payload["transcript_cursor"] = transcript_cursor_after_completion
        event_type = EventType.MODEL_COMPLETED
    elif stream_event.type == ModelStreamEventType.ERROR:
        event_type = EventType.MODEL_ERROR
        payload = copy_json_value(stream_event.payload, "payload")
    else:
        raise ValueError(f"Unsupported model stream event type: {stream_event.type}")
    return Event(
        type=event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=_retry_attempt_payload(
            payload,
            step=step,
            attempt=attempt,
            max_attempts=max_attempts,
        ),
    )


def _with_structured_output_tool_instruction(
    messages: list[Message],
    spec: StructuredOutputSpec,
) -> list[Message]:
    copied_messages = copy_context_messages(messages)
    instruction = Message.text(MessageRole.SYSTEM, structured_output_tool_instruction(spec))
    insert_at = 0
    while (
        insert_at < len(copied_messages) and copied_messages[insert_at].role == MessageRole.SYSTEM
    ):
        insert_at += 1
    copied_messages.insert(insert_at, instruction)
    return copied_messages


def _stream_event_completion(stream_event: ModelStreamEvent) -> ModelCompletion:
    if type(stream_event) is not ModelStreamEvent:
        raise TypeError("Model stream events must be ModelStreamEvent instances.")
    if stream_event.type != ModelStreamEventType.COMPLETED:
        raise ValueError("Only completed model stream events have completion metadata.")
    if stream_event.completion is not None:
        return stream_event.completion
    return normalize_model_completion(stream_event.payload)


def _assistant_step_result(
    *,
    session_id: str,
    step: int,
    assistant_message: Message | None,
    tool_calls: list[runtime_records.ToolCallRequest],
    completion: ModelCompletion,
) -> AssistantStepResult:
    text_content = assistant_text_content(assistant_message)
    return AssistantStepResult(
        session_id=session_id,
        step=step,
        assistant_message=assistant_message,
        tool_calls=list(tool_calls),
        completion=completion,
        text_content=text_content,
        has_user_visible_content=bool(text_content.strip()),
        provider_state_count=provider_state_count(assistant_message),
        thinking_count=thinking_count(assistant_message),
    )


def _typed_retry_fields(
    exc: ModelAttemptFailed,
) -> tuple[int | None, bool | None, float | None]:
    cause = exc.cause
    if isinstance(cause, ModelProviderError):
        return cause.status_code, cause.retryable, cause.retry_after_s
    status_code = exc.payload.get("status_code")
    retryable = exc.payload.get("retryable")
    retry_after_s = exc.payload.get("retry_after_s")
    return (
        status_code if type(status_code) is int else None,
        retryable if type(retryable) is bool else None,
        float(retry_after_s) if type(retry_after_s) in {int, float} else None,
    )


def _model_retry_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    registered_provider: runtime_records.RegisteredProvider,
    step: int,
    decision: RetryDecision,
    error: str,
) -> Event:
    return Event(
        type=EventType.MODEL_RETRY,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=retry_event_payload(
            decision=decision,
            provider_name=registered_provider.name,
            model=session.model,
            step=step,
            error=error,
        ),
    )


def _model_attempt_discarded_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    registered_provider: runtime_records.RegisteredProvider,
    step: int,
    decision: RetryDecision,
) -> Event:
    return Event(
        type=EventType.MODEL_ATTEMPT_DISCARDED,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload={
            "provider": registered_provider.name,
            "model": session.model,
            "step": step,
            "attempt": decision.attempt,
            "next_attempt": decision.next_attempt,
            "max_attempts": decision.max_attempts,
            "reason": None if decision.reason is None else decision.reason.value,
            "status_code": decision.status_code,
        },
    )


def _retry_attempt_payload(
    payload: dict[str, Any],
    *,
    step: int,
    attempt: int,
    max_attempts: int,
) -> dict[str, Any]:
    if max_attempts <= 1:
        return payload
    enriched = dict(payload)
    enriched["step"] = step
    enriched["attempt"] = attempt
    enriched["max_attempts"] = max_attempts
    return enriched


def _payload_model(payload: dict[str, Any], *, fallback: str) -> str:
    model = payload.get("model")
    if type(model) is str and model.strip():
        return model
    return fallback


async def _close_async_iterator(iterator: AsyncIterator[Any]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        # Iterator disposal runs while a more authoritative provider, budget,
        # cancellation, or GeneratorExit outcome is already propagating.
        with contextlib.suppress(Exception):
            await close()
