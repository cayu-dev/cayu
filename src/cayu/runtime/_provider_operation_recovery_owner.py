"""Durable provider-operation start recovery, reconnect and terminal reconciliation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import datetime
from typing import Any, cast

from cayu._exception_groups import (
    add_exception_note_safely,
    exception_cause,
    exception_context,
    exception_group_children,
    iter_exception_tree,
    set_exception_cause,
    set_exception_context,
)
from cayu._task_wait import unexpected_child_cancellation_error
from cayu._validation import (
    copy_durable_json_object,
    safe_durable_value_error_details,
)
from cayu.context.base import ContextInputCoverage
from cayu.context.structured_output import STRUCTURED_OUTPUT_TOOL_NAME, StructuredOutputStrategy
from cayu.events import (
    Event,
    EventType,
    copy_event,
    event_with_runtime_generated_id,
    event_with_runtime_nested_payload_authority,
    event_with_runtime_payload_authority,
)
from cayu.memory.evidence import ContextExposureEvidenceKind, ContextExposureState
from cayu.messages import Message
from cayu.providers._credential_boundary import aclosing_provider_stream
from cayu.providers._http import (
    bind_provider_error_workload_redactor,
    reset_provider_error_workload_redactor,
)
from cayu.providers._openai_protocol import protocol_exception_fields
from cayu.providers.base import (
    EXACT_MODEL_STREAM_RECOVERY_DISPOSITION,
    ModelProviderError,
    ModelStreamDeadlineError,
    ModelStreamEvent,
    ModelStreamEventType,
    ToolDiscoveryProjectionRequest,
)
from cayu.providers.operations import (
    ProviderOperationAdapter,
    ProviderOperationConnection,
    ProviderOperationMalformedError,
    ProviderOperationMode,
    ProviderOperationStartIdempotencySupport,
    ProviderOperationStartRecoveryRequest,
    ProviderOperationState,
    ProviderOperationStatus,
    copy_provider_operation_connection,
    copy_provider_operation_snapshot,
    copy_provider_operation_state,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._checkpoint_redaction import durable_value_contains_secret
from cayu.runtime._diagnostics import exception_diagnostic
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._memory_evidence import recover_context_exposure
from cayu.runtime._model_completion_contracts import (
    ModelCompletionDispatch,
    ModelCompletionPublicationRequest,
    ModelCompletionPublisher,
    ModelCompletionRecoveryContext,
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._model_completion_delivery import (
    _combine_authoritative_model_failure,
    _combine_post_completion_failures,
    _durable_assistant_step_result,
    _non_turn_model_completion_event,
    _publish_model_completion,
    _take_model_completion_cancellation,
)
from cayu.runtime._model_errors import (
    copy_provider_exception_control,
    resolve_completion_billing_identity,
    runtime_owned_model_stream_error_event,
)
from cayu.runtime._model_event_authority import _event_with_model_identity_authority
from cayu.runtime._model_execution_selection import ModelExecutionSelection
from cayu.runtime._model_failover_stage import model_failover_target_for_stored_stage
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
from cayu.runtime._model_tool_discovery import (
    _hosted_tool_discovery_projection,
    _hosted_tool_discovery_projection_digest,
    _hosted_tool_discovery_publication_authority,
    _hosted_tool_name_sha256,
)
from cayu.runtime._provider_operation_cancellation_owner import (
    ProviderOperationCancellationOwner,
    _provider_operation_target_model,
)
from cayu.runtime._run_limits import RunLimitController
from cayu.runtime._session_control import SessionInterruptedByRequest
from cayu.runtime._structured_output_tool_round import (
    _redact_structured_output_validation,
    _validate_structured_output_tool_round,
)
from cayu.runtime.execution_profiles import event_with_execution_profile_fingerprint_authority
from cayu.runtime.execution_units import ModelAttemptIdentity
from cayu.runtime.model_steps import AssistantStepResult, classify_assistant_step
from cayu.runtime.provider_operations import (
    ProviderOperationEvidenceError,
    ProviderOperationProgressCommit,
    ProviderOperationProgressEnvelope,
    ProviderOperationRecoveryResult,
    ProviderOperationRecoveryStatus,
    ProviderOperationUnavailableReason,
    RecoverableProviderOperation,
    RecoverableProviderOperationStart,
    commit_provider_operation_progress,
    load_recoverable_provider_operation,
    provider_operation_progress_envelope,
    provider_operation_progress_event_id,
    provider_operation_started_event_id,
    provider_operation_unavailable_reason,
)
from cayu.sessions._provider_operation_cancellation_claim import (
    ProviderOperationCancellationClaim,
    provider_operation_cancellation_claim_from_checkpoint,
)
from cayu.sessions.base import (
    ModelCompletionStage,
    RuntimePublicationOperationRecordMutation,
    Session,
    SessionRunFenced,
    SessionStatus,
    SessionStore,
)
from cayu.tools.catalogue import CALL_TOOL_NAME
from cayu.tools.discovery import (
    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
    current_tool_discovery_view,
    tool_discovery_generation_id,
    tool_discovery_record_matches_descriptor,
)
from cayu.tools.exposure import tool_capability_ceiling_from_session_metadata
from cayu.vaults.redaction import SecretRedactor

logger = logging.getLogger(__name__)


def _provider_operation_progress_contains_secret(
    envelope: ProviderOperationProgressEnvelope,
    *,
    redactor: SecretRedactor,
    targeted_tool_reference_grant_ids: Mapping[str, str] | None = None,
) -> bool:
    """Check adapter-owned continuation and output without scanning schema keys."""

    stream_event = envelope.stream_event
    payload = copy_durable_json_object(stream_event.payload, "provider_operation_stream_payload")
    if (
        stream_event.type is ModelStreamEventType.TOOL_CALL
        and payload.get("name") == CALL_TOOL_NAME
    ):
        arguments = payload.get("arguments")
        if type(arguments) is dict:
            tool_ref = arguments.get("tool_ref")
            if (
                type(tool_ref) is str
                and targeted_tool_reference_grant_ids is not None
                and tool_ref in targeted_tool_reference_grant_ids
            ):
                # This exact value was projected by the runtime for the model
                # request. The surrounding model-authored envelope and all
                # inner arguments remain untrusted and are still scanned.
                arguments = copy_durable_json_object(arguments, "call_tool.arguments")
                arguments.pop("tool_ref", None)
                payload["arguments"] = arguments
    return (
        redactor.redact_text(stream_event.delta) != stream_event.delta
        or durable_value_contains_secret(
            payload,
            redactor=redactor,
            path=("provider_operation_stream_payload",),
        )
        or durable_value_contains_secret(
            envelope.recovery_metadata.opaque,
            redactor=redactor,
            path=("provider_operation_recovery_opaque",),
        )
    )


def _classify_provider_recovery_failure(
    failure: BaseException,
    *,
    cancellation_baseline: int,
    operation: str,
) -> BaseException:
    """Keep current caller cancellation distinct from child-only cancellation."""

    fatal_leaves = [
        candidate
        for candidate in iter_exception_tree(failure)
        if not isinstance(candidate, BaseExceptionGroup)
        and not isinstance(candidate, (Exception, asyncio.CancelledError))
    ]
    if fatal_leaves:
        return failure
    task = asyncio.current_task()
    if task is not None and task.cancelling() > cancellation_baseline:
        cancellation = _current_provider_recovery_cancellation(
            failure,
            cancellation_baseline=cancellation_baseline,
        )
        if cancellation is None:  # pragma: no cover - guarded by the task count
            raise AssertionError("Provider recovery lost current task cancellation.")
        secondary = _provider_recovery_failure_without_identity(
            failure,
            excluded_identity=id(cancellation),
        )
        if secondary is not None and not _attach_provider_recovery_secondary_failure(
            cancellation,
            secondary,
        ):
            add_exception_note_safely(
                cancellation,
                "Provider recovery also reported additional failures that could not be "
                "attached to caller cancellation.",
            )
        return cancellation
    cancellations = [
        candidate
        for candidate in iter_exception_tree(failure)
        if isinstance(candidate, asyncio.CancelledError)
    ]
    if not cancellations:
        return failure
    unexpected = unexpected_child_cancellation_error(cancellations[0], operation=operation)
    if failure is not cancellations[0]:
        set_exception_cause(unexpected, failure)
    return unexpected


def _raise_pending_provider_recovery_cancellation(*, cancellation_baseline: int) -> None:
    """Propagate caller cancellation even when a provider await suppressed delivery."""

    cancellation = _current_provider_recovery_cancellation(
        None,
        cancellation_baseline=cancellation_baseline,
    )
    if cancellation is None:
        return
    raise cancellation


def _current_provider_recovery_cancellation(
    failure: BaseException | None,
    *,
    cancellation_baseline: int,
) -> asyncio.CancelledError | None:
    """Return current cancellation without consuming and re-arming its task request."""

    task = asyncio.current_task()
    if task is None or task.cancelling() <= cancellation_baseline:
        return None
    cancellation = next(
        (
            candidate
            for candidate in (() if failure is None else iter_exception_tree(failure))
            if isinstance(candidate, asyncio.CancelledError)
        ),
        None,
    )
    if cancellation is not None:
        return cancellation
    cancel_message = getattr(task, "_cancel_message", None)
    return (
        asyncio.CancelledError()
        if cancel_message is None
        else asyncio.CancelledError(cancel_message)
    )


def _provider_recovery_cleanup_cancellation_baseline(
    publication_failure: BaseException | None,
    *,
    cancellation_baseline: int,
) -> int:
    """Ignore one cancellation already delivered by the publication await."""

    task = asyncio.current_task()
    if (
        not isinstance(publication_failure, asyncio.CancelledError)
        or task is None
        or task.cancelling() <= cancellation_baseline
    ):
        return cancellation_baseline
    return task.cancelling()


async def _close_provider_recovery_iterator(
    iterator: AsyncIterator[Any],
    *,
    cancellation_baseline: int,
    operation: str,
) -> Exception | None:
    """Close one provider-owned recovery iterator without forging caller cancellation."""

    try:
        close = getattr(iterator, "aclose", None)
        if callable(close):
            await close()
    except BaseException as close_failure:
        classified = _classify_provider_recovery_failure(
            close_failure,
            cancellation_baseline=cancellation_baseline,
            operation=operation,
        )
        if not isinstance(classified, Exception):
            if classified is close_failure:
                raise
            raise classified from close_failure
        return classified
    _raise_pending_provider_recovery_cancellation(cancellation_baseline=cancellation_baseline)
    return None


def _provider_recovery_cleanup_payload(
    failure: Exception,
    *,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    diagnostic = exception_diagnostic(
        failure,
        empty_message="provider recovery stream cleanup failed",
        nonportable_message=(
            "Provider recovery stream cleanup failed with a non-portable diagnostic."
        ),
        redactor=redactor,
    )
    return {
        **diagnostic.payload_fields(),
        "phase": "provider_recovery_stream_cleanup",
    }


class _ProviderRecoveryRequiredPublicationFailureEvidence(RuntimeError):
    """Sanitized cleanup evidence retained when typed publication also fails."""

    def __init__(
        self,
        *,
        recovery_reason: ProviderOperationUnavailableReason,
        cleanup_diagnostic: dict[str, Any],
    ) -> None:
        copied = copy_durable_json_object(cleanup_diagnostic, "cleanup_diagnostic")
        error_type = copied.get("error_type")
        if type(error_type) is not str or not error_type.strip():
            raise ValueError("Provider recovery cleanup diagnostic has no error type.")
        super().__init__(
            "Provider recovery stream cleanup failed before "
            f"{recovery_reason.value} recovery evidence was acknowledged "
            f"({error_type})."
        )
        self.recovery_reason = recovery_reason
        self.cleanup_diagnostic = copied


class _ProviderOperationStreamStatusError(RuntimeError):
    """A validated operation error boundary carrying the provider's typed status."""

    def __init__(
        self,
        status: ProviderOperationStatus,
        provider_error: ModelProviderError,
    ) -> None:
        super().__init__(str(provider_error))
        self.status = status
        self.provider_error = provider_error


async def _emit_provider_recovery_required_event(
    event_writer: RuntimeEventWriter,
    event: Event,
    *,
    recovery_reason: ProviderOperationUnavailableReason,
    cleanup_failure: Exception | None,
    redactor: SecretRedactor,
) -> Event:
    """Publish typed recovery evidence without losing sanitized cleanup failure."""

    try:
        return await event_writer.emit(event)
    except BaseException as publication_failure:
        if cleanup_failure is None:
            raise
        cleanup_evidence = _ProviderRecoveryRequiredPublicationFailureEvidence(
            recovery_reason=recovery_reason,
            cleanup_diagnostic=_provider_recovery_cleanup_payload(
                cleanup_failure,
                redactor=redactor,
            ),
        )
        if not _attach_provider_recovery_secondary_failure(
            publication_failure,
            cleanup_evidence,
        ):
            raise BaseExceptionGroup(
                "Provider recovery evidence publication and stream cleanup both failed.",
                [publication_failure, cleanup_evidence],
            ) from None
        add_exception_note_safely(
            publication_failure,
            "Provider recovery-required publication also retained sanitized "
            f"{recovery_reason.value} stream-cleanup evidence.",
        )
        raise


def _provider_recovery_failure_without_identity(
    error: BaseException,
    *,
    excluded_identity: int,
) -> BaseException | None:
    """Remove one owned failure while retaining ordered non-overlapping subgroups."""

    pending: list[tuple[BaseException, bool]] = [(error, False)]
    children_by_group: dict[int, tuple[BaseException, ...]] = {}
    retained_by_identity: dict[int, BaseException | None] = {}
    while pending:
        candidate, expanded = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in retained_by_identity:
            continue
        if candidate_id == excluded_identity:
            retained_by_identity[candidate_id] = None
            continue
        if not isinstance(candidate, BaseExceptionGroup):
            retained_by_identity[candidate_id] = candidate
            continue
        if expanded:
            children = children_by_group.pop(candidate_id, ())
            retained_children = [
                retained
                for child in children
                if (retained := retained_by_identity.get(id(child))) is not None
            ]
            if not retained_children:
                retained_by_identity[candidate_id] = None
            elif len(retained_children) == len(children) and all(
                retained is child
                for retained, child in zip(retained_children, children, strict=True)
            ):
                retained_by_identity[candidate_id] = candidate
            else:
                retained_by_identity[candidate_id] = BaseExceptionGroup(
                    "Provider recovery additional non-cancellation failures.",
                    retained_children,
                )
            continue
        children = exception_group_children(candidate)
        if children is None:
            retained_by_identity[candidate_id] = RuntimeError(
                "Provider recovery received an unreadable exception group."
            )
            continue
        children_by_group[candidate_id] = children
        pending.append((candidate, True))
        pending.extend((child, False) for child in reversed(children))

    return retained_by_identity.get(id(error))


def _provider_recovery_failure_graph_contains_identity(
    error: BaseException,
    *,
    target_identity: int,
) -> bool:
    """Return whether one safe exception graph contains an exact object identity."""

    pending = [error]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        candidate_id = id(candidate)
        if candidate_id == target_identity:
            return True
        if candidate_id in visited:
            continue
        visited.add(candidate_id)
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is not None:
                pending.extend(children)
        cause = exception_cause(candidate)
        if cause is not None:
            pending.append(cause)
        context = exception_context(candidate)
        if context is not None:
            pending.append(context)
    return False


def _detach_provider_recovery_back_edges(
    error: BaseException,
    *,
    target: BaseException,
) -> bool:
    """Remove causal links back to a primary error before attaching this graph."""

    pending = [error]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in visited:
            continue
        visited.add(candidate_id)
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is not None:
                if any(child is target for child in children):
                    return False
                pending.extend(children)
        cause = exception_cause(candidate)
        if cause is target:
            if not set_exception_cause(candidate, None):
                return False
        elif cause is not None:
            pending.append(cause)
        context = exception_context(candidate)
        if context is target:
            if not set_exception_context(candidate, None):
                return False
        elif context is not None:
            pending.append(context)
    return True


def _attach_provider_recovery_secondary_failure(
    primary: BaseException,
    secondary: BaseException,
) -> bool:
    """Retain one ordered recovery failure as an acyclic causal graph."""

    if primary is secondary:
        return True
    if not _detach_provider_recovery_back_edges(secondary, target=primary):
        return False
    if _provider_recovery_failure_graph_contains_identity(
        secondary,
        target_identity=id(primary),
    ):
        return False
    prior_cause = exception_cause(primary)
    prior_context = None if prior_cause is not None else exception_context(primary)
    prior_failure = prior_cause if prior_cause is not None else prior_context
    if prior_failure is secondary or (
        prior_failure is not None
        and _provider_recovery_failure_graph_contains_identity(
            prior_failure,
            target_identity=id(secondary),
        )
    ):
        return True
    if prior_failure is None or _provider_recovery_failure_graph_contains_identity(
        secondary,
        target_identity=id(prior_failure),
    ):
        combined = secondary
    else:
        combined = BaseExceptionGroup(
            "Provider recovery publication and stream cleanup both failed.",
            [prior_failure, secondary],
        )
    if not set_exception_cause(primary, combined):
        return False
    if prior_context is not None:
        set_exception_context(primary, None)
    return True


class ProviderOperationRecoveryOwner:
    """Recover exact durable operations through shared live-execution boundaries.

    The caller supplies its existing cancellation owner and backend-bound
    dependencies. Recovery never owns a second cancellation registry or changes
    the store's transaction and checkpoint authority boundaries.
    """

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        run_limit_controller: RunLimitController,
        cancellation: ProviderOperationCancellationOwner,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._run_limit_controller = run_limit_controller
        self._provider_operation_cancellation = cancellation
        self._secret_redactor = secret_redactor
        self._clock = clock

    def progress_event(
        self,
        *,
        stage: ModelCompletionStage,
        state: ProviderOperationState,
        stream_event: ModelStreamEvent,
        runtime_event: Event | None,
        session: Session,
        interaction_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
        model_attempt_identity: ModelAttemptIdentity,
        targeted_tool_reference_grant_ids: Mapping[str, str] | None = None,
    ) -> Event:
        """Attach private reconnect state to its corresponding normalized event."""

        metadata = stream_event.recovery_metadata
        if metadata is None or metadata.cursor is None:
            raise ProviderOperationEvidenceError(
                "Reconnectable provider events must carry a monotonic recovery cursor."
            )
        event_id = provider_operation_progress_event_id(stage.stage_id, metadata.cursor)
        progress_envelope = provider_operation_progress_envelope(state, stream_event)
        if _provider_operation_progress_contains_secret(
            progress_envelope,
            redactor=self._secret_redactor,
            targeted_tool_reference_grant_ids=targeted_tool_reference_grant_ids,
        ):
            raise ProviderOperationEvidenceError(
                "Provider-operation recovery metadata or normalized output contains a "
                "workload secret and cannot cross the durable recovery boundary."
            )
        envelope = progress_envelope.model_dump(mode="json")
        if runtime_event is None:
            event = _event_with_model_identity_authority(
                Event(
                    id=event_id,
                    type=EventType.PROVIDER_OPERATION_PROGRESS,
                    session_id=session.id,
                    interaction_id=interaction_id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        "provider": registered_provider.name,
                        "step": step,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        **model_attempt_identity.payload(),
                        "operation_id": state.operation_id,
                        "stream_protocol": state.stream_protocol,
                        "provider_operation_progress": envelope,
                    },
                ),
                model_attempt_identity,
            )
            event = event_with_runtime_payload_authority(
                event,
                "operation_id",
                "stream_protocol",
            )
        else:
            event = copy_event(runtime_event)
            if event.interaction_id not in {None, interaction_id}:
                raise ProviderOperationEvidenceError(
                    "Provider-operation progress changed its owning interaction."
                )
            payload = copy_durable_json_object(event.payload, "event.payload")
            payload["provider_operation_progress"] = envelope
            event = event.model_copy(
                update={
                    "id": event_id,
                    "interaction_id": interaction_id,
                    "payload": payload,
                },
                deep=True,
            )
        recovery_context = model_completion_recovery_context_from_stage(stage)
        event = event_with_execution_profile_fingerprint_authority(
            event,
            (None if recovery_context is None else recovery_context.execution_profile_fingerprint),
        )
        if (
            stream_event.type is ModelStreamEventType.TOOL_CALL
            and stream_event.payload.get("name") == CALL_TOOL_NAME
            and targeted_tool_reference_grant_ids is not None
        ):
            arguments = stream_event.payload.get("arguments")
            tool_ref = arguments.get("tool_ref") if type(arguments) is dict else None
            if type(tool_ref) is str and tool_ref in targeted_tool_reference_grant_ids:
                event = event_with_runtime_nested_payload_authority(
                    event,
                    (
                        "provider_operation_progress",
                        "stream_event",
                        "payload",
                        "arguments",
                        "tool_ref",
                    ),
                )
        return event_with_runtime_generated_id(event)

    async def commit_stream_event(
        self,
        *,
        stage: ModelCompletionStage,
        state: ProviderOperationState,
        stream_event: ModelStreamEvent,
        runtime_event: Event | None,
        session: Session,
        interaction_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
        model_attempt_identity: ModelAttemptIdentity,
        targeted_tool_reference_grant_ids: Mapping[str, str] | None = None,
    ) -> tuple[ProviderOperationProgressCommit, Event | None]:
        event = self.progress_event(
            stage=stage,
            state=state,
            stream_event=stream_event,
            runtime_event=runtime_event,
            session=session,
            interaction_id=interaction_id,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=environment_name,
            step=step,
            attempt=attempt,
            max_attempts=max_attempts,
            model_attempt_identity=model_attempt_identity,
            targeted_tool_reference_grant_ids=targeted_tool_reference_grant_ids,
        )
        prepared = self._event_writer.prepare(event)
        commit = await commit_provider_operation_progress(
            self._session_store,
            stage=stage,
            model_attempt_identity=model_attempt_identity,
            current_state=state,
            stream_event=stream_event,
            event=prepared,
            expected_run_epoch=session.run_epoch,
        )
        if commit.replayed:
            return commit, None
        [emitted] = await self._event_writer.fan_out_persisted([commit.event])
        return commit, emitted

    async def recover_start(
        self,
        *,
        session: Session,
        stage: ModelCompletionStage,
        start: RecoverableProviderOperationStart,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        model_completion_publisher: ModelCompletionPublisher,
        invocation_context: InvocationContext | None = None,
        model_execution_selection: ModelExecutionSelection | None = None,
    ) -> ProviderOperationRecoveryResult:
        """Recover start-only evidence without persisting or replaying a raw request."""

        selected_model = _provider_operation_target_model(session, stage)
        if model_execution_selection is not None:
            model_execution_selection.require_recovery_scope(
                session=session,
                stage=stage,
                invocation_context=invocation_context,
                registered_provider=registered_provider,
            )
        elif model_failover_target_for_stored_stage(session=session, stage=stage) is not None:
            raise RuntimeError("Routed provider-operation recovery requires admitted selection.")
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or registered_agent is not invocation_context.registered_agent
            or (
                model_execution_selection is None
                and registered_provider is not invocation_context.registered_provider
            )
            or environment_name
            != (
                None
                if invocation_context.registered_environment is None
                else invocation_context.registered_environment.spec.name
            )
        ):
            raise RuntimeError(
                "Provider-operation start recovery substituted frozen invocation authority."
            )

        provider = registered_provider.provider
        adapter = provider.provider_operations
        recovery_context = model_completion_recovery_context_from_stage(stage)
        exact_recovery = start.idempotency_support is ProviderOperationStartIdempotencySupport.EXACT
        if (
            provider.provider_operation_mode is not ProviderOperationMode.BACKGROUND
            or not isinstance(adapter, ProviderOperationAdapter)
        ):
            raise ProviderOperationEvidenceError(
                "Provider-operation start recovery is unavailable."
            )
        if start.provider != registered_provider.name or start.model != selected_model:
            raise ProviderOperationEvidenceError(
                "Provider-operation start recovery resolved a different provider scope."
            )

        async def unavailable(
            reason: ProviderOperationUnavailableReason,
            *,
            cleanup_failure: Exception | None = None,
        ) -> ProviderOperationRecoveryResult:
            payload: dict[str, Any] = {
                "provider": registered_provider.name,
                "model": selected_model,
                "step": start.step,
                "attempt": start.attempt,
                "max_attempts": start.max_attempts,
                **start.model_attempt_identity.payload(),
                "source_run_epoch": start.source_run_epoch,
                "run_epoch": session.run_epoch,
                "start_id": start.start_id,
                "status": reason.value,
                "recovery_reason": reason.value,
                "idempotent_start_recovery": exact_recovery,
            }
            if cleanup_failure is not None:
                payload["provider_cleanup_failure"] = _provider_recovery_cleanup_payload(
                    cleanup_failure,
                    redactor=self._secret_redactor,
                )
            required = _event_with_model_identity_authority(
                Event(
                    type=EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED,
                    session_id=session.id,
                    interaction_id=start.interaction_id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=payload,
                ),
                start.model_attempt_identity,
            )
            required = event_with_execution_profile_fingerprint_authority(
                required,
                (
                    None
                    if recovery_context is None
                    else recovery_context.execution_profile_fingerprint
                ),
            )
            required = event_with_runtime_payload_authority(required, "start_id")
            emitted = await _emit_provider_recovery_required_event(
                self._event_writer,
                required,
                recovery_reason=reason,
                cleanup_failure=cleanup_failure,
                redactor=self._secret_redactor,
            )
            return ProviderOperationRecoveryResult(
                status=ProviderOperationRecoveryStatus.UNAVAILABLE,
                events=(emitted,),
                unavailable_reason=reason,
            )

        if start.idempotency_support is not ProviderOperationStartIdempotencySupport.EXACT:
            return await unavailable(ProviderOperationUnavailableReason.AMBIGUOUS_SUBMISSION)
        if adapter.start_idempotency_support is not ProviderOperationStartIdempotencySupport.EXACT:
            return await unavailable(ProviderOperationUnavailableReason.AMBIGUOUS_SUBMISSION)
        recovery_task = asyncio.current_task()
        recovery_cancellation_baseline = 0 if recovery_task is None else recovery_task.cancelling()
        try:
            raw_connection = await adapter.recover_start(
                ProviderOperationStartRecoveryRequest(idempotency_key=start.start_id)
            )
        except BaseException as recovery_failure:
            recovery_failure = _classify_provider_recovery_failure(
                recovery_failure,
                cancellation_baseline=recovery_cancellation_baseline,
                operation="Provider operation start recovery",
            )
            if not isinstance(recovery_failure, Exception):
                raise recovery_failure
            return await unavailable(ProviderOperationUnavailableReason.UNAVAILABLE)
        try:
            connection = copy_provider_operation_connection(raw_connection)
        except Exception as malformed_failure:
            cleanup_failure = None
            if type(raw_connection) is ProviderOperationConnection:
                cleanup_failure = await _close_provider_recovery_iterator(
                    raw_connection.events,
                    cancellation_baseline=recovery_cancellation_baseline,
                    operation="Provider operation malformed start recovery stream cleanup",
                )
            _raise_pending_provider_recovery_cancellation(
                cancellation_baseline=recovery_cancellation_baseline
            )
            if cleanup_failure is not None:
                add_exception_note_safely(
                    malformed_failure,
                    "Provider recovery stream cleanup also failed with "
                    f"{type(cleanup_failure).__name__}.",
                )
            return await unavailable(
                ProviderOperationUnavailableReason.MALFORMED,
                cleanup_failure=cleanup_failure,
            )

        operation_event = _event_with_model_identity_authority(
            Event(
                id=provider_operation_started_event_id(start.start_id),
                type=EventType.PROVIDER_OPERATION_STARTED,
                session_id=session.id,
                interaction_id=start.interaction_id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload={
                    "provider": registered_provider.name,
                    "model": selected_model,
                    "step": start.step,
                    "attempt": start.attempt,
                    "max_attempts": start.max_attempts,
                    **start.model_attempt_identity.payload(),
                    "source_run_epoch": start.source_run_epoch,
                    "start_id": start.start_id,
                    "state_version": connection.state.version,
                    "operation_id": connection.state.operation_id,
                    "stream_protocol": connection.state.stream_protocol,
                    "status": connection.status.value,
                    "recovery_metadata": connection.state.recovery_metadata.model_dump(
                        mode="json",
                        exclude_none=True,
                    ),
                    "idempotent_start_recovery": True,
                },
            ),
            start.model_attempt_identity,
        )
        operation_event = event_with_execution_profile_fingerprint_authority(
            operation_event,
            (None if recovery_context is None else recovery_context.execution_profile_fingerprint),
        )
        operation_event = event_with_runtime_payload_authority(
            operation_event,
            "start_id",
        )
        publication_failure: BaseException | None = None
        try:
            persisted = await self._event_writer.persist_exact_replay(operation_event)
            [emitted] = await self._event_writer.fan_out_persisted([persisted])
        except BaseException as failure:
            publication_failure = failure
            raise
        finally:
            cleanup_cancellation_baseline = _provider_recovery_cleanup_cancellation_baseline(
                publication_failure,
                cancellation_baseline=recovery_cancellation_baseline,
            )
            try:
                cleanup_failure = await _close_provider_recovery_iterator(
                    raw_connection.events,
                    cancellation_baseline=cleanup_cancellation_baseline,
                    operation="Provider operation start recovery stream cleanup",
                )
            except BaseException as cleanup_signal:
                if (
                    publication_failure is not None
                    and cleanup_signal is not publication_failure
                    and not any(
                        candidate is publication_failure
                        for candidate in iter_exception_tree(cleanup_signal)
                    )
                ):
                    cleanup_cause = exception_cause(cleanup_signal)
                    if (
                        cleanup_cause is not None
                        and cleanup_cause is not publication_failure
                        and not _attach_provider_recovery_secondary_failure(
                            publication_failure,
                            cleanup_cause,
                        )
                    ):
                        raise BaseExceptionGroup(
                            "Provider recovery publication and stream cleanup both failed.",
                            [publication_failure, cleanup_signal],
                        ) from cleanup_signal
                    if not set_exception_cause(cleanup_signal, publication_failure):
                        raise BaseExceptionGroup(
                            "Provider recovery publication and stream cleanup both failed.",
                            [publication_failure, cleanup_signal],
                        ) from cleanup_signal
                raise
            if cleanup_failure is not None:
                diagnostic = _provider_recovery_cleanup_payload(
                    cleanup_failure,
                    redactor=self._secret_redactor,
                )
                if publication_failure is not None:
                    if not _attach_provider_recovery_secondary_failure(
                        publication_failure,
                        cleanup_failure,
                    ):
                        raise BaseExceptionGroup(
                            "Provider recovery publication and stream cleanup both failed.",
                            [publication_failure, cleanup_failure],
                        )
                    add_exception_note_safely(
                        publication_failure,
                        "Provider recovery stream cleanup also failed with "
                        f"{diagnostic['error_type']}.",
                    )
                else:
                    logger.warning(
                        "Provider recovery stream cleanup failed after exact start publication: %s",
                        diagnostic["error_type"],
                    )
        operation = await load_recoverable_provider_operation(
            self._session_store,
            stage,
        )
        if operation is None:
            raise ProviderOperationEvidenceError(
                "Idempotent provider start did not produce recoverable operation evidence."
            )
        recovered = await self.recover(
            session=session,
            stage=stage,
            operation=operation,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=environment_name,
            recovery_context=recovery_context,
            model_completion_publisher=model_completion_publisher,
            invocation_context=invocation_context,
            model_execution_selection=model_execution_selection,
        )
        return ProviderOperationRecoveryResult(
            status=recovered.status,
            events=(emitted, *recovered.events),
            completion_event=recovered.completion_event,
            unavailable_reason=recovered.unavailable_reason,
        )

    async def recover(
        self,
        *,
        session: Session,
        stage: ModelCompletionStage,
        operation: RecoverableProviderOperation,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        recovery_context: ModelCompletionRecoveryContext | None,
        model_completion_publisher: ModelCompletionPublisher,
        invocation_context: InvocationContext | None = None,
        model_execution_selection: ModelExecutionSelection | None = None,
    ) -> ProviderOperationRecoveryResult:
        """Retrieve and atomically publish one exact offline provider operation."""

        selected_model = _provider_operation_target_model(session, stage)
        if model_execution_selection is not None:
            model_execution_selection.require_recovery_scope(
                session=session,
                stage=stage,
                invocation_context=invocation_context,
                registered_provider=registered_provider,
            )
        elif model_failover_target_for_stored_stage(session=session, stage=stage) is not None:
            raise RuntimeError("Routed provider-operation recovery requires admitted selection.")
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or registered_agent is not invocation_context.registered_agent
            or (
                model_execution_selection is None
                and registered_provider is not invocation_context.registered_provider
            )
            or environment_name
            != (
                None
                if invocation_context.registered_environment is None
                else invocation_context.registered_environment.spec.name
            )
        ):
            raise RuntimeError(
                "Provider-operation recovery substituted frozen invocation authority."
            )

        provider = registered_provider.provider
        adapter = provider.provider_operations
        if (
            provider.provider_operation_mode is not ProviderOperationMode.BACKGROUND
            or not isinstance(
                adapter,
                ProviderOperationAdapter,
            )
        ):
            raise RuntimeError(
                "The registered provider no longer supports its durable background operation."
            )
        if operation.provider != registered_provider.name:
            raise RuntimeError("Provider-operation recovery resolved a different provider.")
        if operation.model != selected_model:
            raise RuntimeError("Provider-operation recovery resolved a different model.")
        if operation.model_attempt_identity.model_step_id != stage.logical_step_id:
            raise RuntimeError("Provider-operation recovery belongs to a different model stage.")
        if (
            recovery_context is not None
            and type(recovery_context) is not ModelCompletionRecoveryContext
        ):
            raise TypeError("Provider-operation recovery context is invalid.")
        if (
            recovery_context is not None
            and recovery_context.structured_output is not None
            and recovery_context.structured_output.strategy is StructuredOutputStrategy.NATIVE
        ):
            raise ProviderOperationEvidenceError(
                "Offline recovery of native structured output requires manual reconciliation."
            )
        targeted_tool_reference_grant_ids = {
            record.tool_ref: record.grant_id
            for record in await self._session_store.list_targeted_tool_grants(
                session.id,
                interaction_id=operation.interaction_id,
            )
        }
        hosted_discovery_projection: ToolDiscoveryProjectionRequest | None = None
        hosted_replay_grant_ids: dict[str, str] = {}
        if recovery_context is not None and recovery_context.hosted_tool_discovery is not None:
            if recovery_context.tool_exposure is None:
                raise ProviderOperationEvidenceError(
                    "Hosted Tool Search recovery has no durable tool-exposure authority."
                )
            try:
                targeted_name_digests = frozenset(
                    recovery_context.hosted_tool_discovery.targeted_tool_name_sha256s
                )
                targeted_names = frozenset(
                    descriptor.name
                    for descriptor in registered_agent.tool_catalogue.descriptors
                    if _hosted_tool_name_sha256(descriptor.name) in targeted_name_digests
                )
                if {
                    _hosted_tool_name_sha256(name) for name in targeted_names
                } != targeted_name_digests:
                    raise ValueError("Hosted Tool Search recovery lost a targeted-tool exclusion.")
                loaded_name_digests = frozenset(
                    recovery_context.hosted_tool_discovery.loaded_tool_name_sha256s
                )
                loaded_names = frozenset(
                    descriptor.name
                    for descriptor in registered_agent.tool_catalogue.descriptors
                    if _hosted_tool_name_sha256(descriptor.name) in loaded_name_digests
                )
                if {_hosted_tool_name_sha256(name) for name in loaded_names} != loaded_name_digests:
                    raise ValueError("Hosted Tool Search recovery lost replay-loaded authority.")
                hosted_discovery_projection = _hosted_tool_discovery_projection(
                    session=session,
                    registered_agent=registered_agent,
                    excluded_tool_names=(
                        *recovery_context.tool_exposure.tool_names,
                        *targeted_names,
                    ),
                    redactor=self._secret_redactor,
                )
                if loaded_names:
                    recovery_view = current_tool_discovery_view(
                        await self._session_store.load_session_operation(
                            session.id,
                            TOOL_DISCOVERY_VIEW_OPERATION_KEY,
                        ),
                        session_id=session.id,
                        generation_id=tool_discovery_generation_id(
                            session_id=session.id,
                            root_invocation_id=session.invocation.root_invocation_id,
                        ),
                        agent_name=registered_agent.spec.name,
                        catalogue=registered_agent.tool_catalogue,
                        ceiling=tool_capability_ceiling_from_session_metadata(session.metadata),
                    )
                    grants_by_name = {grant.tool_name: grant for grant in recovery_view.grants}
                    if any(
                        name not in grants_by_name
                        or not tool_discovery_record_matches_descriptor(
                            grants_by_name[name],
                            registered_agent.tool_catalogue.descriptor_for_name(name),
                        )
                        for name in loaded_names
                    ):
                        raise ValueError(
                            "Hosted Tool Search recovery lost a replay-loaded durable grant."
                        )
                    hosted_replay_grant_ids = {
                        name: grants_by_name[name].grant_id for name in sorted(loaded_names)
                    }
                    candidates_by_name = {
                        cast("str", tool["name"]): tool
                        for tool in hosted_discovery_projection.candidate_tools
                    }
                    if any(name not in candidates_by_name for name in loaded_names):
                        raise ValueError(
                            "Hosted Tool Search recovery loaded authority is not a candidate."
                        )
                    hosted_discovery_projection = ToolDiscoveryProjectionRequest.model_validate(
                        {
                            **hosted_discovery_projection.model_dump(mode="python"),
                            "loaded_tools": tuple(
                                candidates_by_name[name] for name in sorted(loaded_names)
                            ),
                        }
                    )
                recovered_projection_digest = _hosted_tool_discovery_projection_digest(
                    hosted_discovery_projection
                )
            except (TypeError, ValueError) as exc:
                raise ProviderOperationEvidenceError(
                    "Hosted Tool Search recovery could not reconstruct its original "
                    "candidate projection."
                ) from exc
            if (
                recovered_projection_digest
                != recovery_context.hosted_tool_discovery.projection_sha256
            ):
                raise ProviderOperationEvidenceError(
                    "Hosted Tool Search recovery candidate authority no longer matches "
                    "the dispatched request."
                )
        if durable_value_contains_secret(
            operation.state.recovery_metadata.opaque,
            redactor=self._secret_redactor,
            path=("provider_operation_recovery_opaque",),
        ) or any(
            _provider_operation_progress_contains_secret(
                provider_operation_progress_envelope(operation.state, accepted_event),
                redactor=self._secret_redactor,
                targeted_tool_reference_grant_ids=targeted_tool_reference_grant_ids,
            )
            for accepted_event in operation.accepted_stream_events
        ):
            raise ProviderOperationEvidenceError(
                "Stored provider-operation recovery state conflicts with a current workload secret."
            )

        cancellation_claim = ProviderOperationCancellationClaim(
            claim_id=(
                f"provider-cancel:{stage.stage_id}:{session.run_epoch}:"
                f"{operation.state.operation_id}:{operation.state.stream_protocol}"
            ),
            stage_id=stage.stage_id,
            run_epoch=session.run_epoch,
            operation_id=operation.state.operation_id,
            stream_protocol=operation.state.stream_protocol,
            expires_at=session.updated_at,
        )
        recovery_under_cancellation_claim = False

        async def require_recovery_owner() -> None:
            nonlocal recovery_under_cancellation_claim
            current = await self._session_store.load(session.id)
            if current is None:
                raise KeyError(f"Session not found: {session.id}")
            if current.run_epoch != session.run_epoch:
                raise SessionRunFenced(
                    "Provider-operation recovery run epoch is stale: expected "
                    f"{session.run_epoch}, current {current.run_epoch}."
                )
            if current.status is SessionStatus.INTERRUPTING:
                checkpoint = await self._session_store.load_checkpoint(session.id)
                if (
                    stored_claim := provider_operation_cancellation_claim_from_checkpoint(
                        checkpoint
                    )
                ) is None or not stored_claim.same_owner(cancellation_claim):
                    raise SessionInterruptedByRequest(session.id)
                recovery_under_cancellation_claim = True

        def recovery_event(
            event_type: EventType,
            *,
            status: str,
            recovery_reason: ProviderOperationUnavailableReason | None = None,
            cleanup_failure: Exception | None = None,
            protocol_failure: Exception | None = None,
        ) -> Event:
            payload: dict[str, Any] = {
                "provider": registered_provider.name,
                "model": selected_model,
                "step": operation.step,
                "attempt": operation.attempt,
                "max_attempts": operation.max_attempts,
                **operation.model_attempt_identity.payload(),
                "source_run_epoch": operation.source_run_epoch,
                "run_epoch": session.run_epoch,
                "operation_id": operation.state.operation_id,
                "stream_protocol": operation.state.stream_protocol,
                "status": status,
            }
            payload.update(protocol_exception_fields(protocol_failure))
            if recovery_reason is not None:
                payload["recovery_reason"] = recovery_reason.value
            if cleanup_failure is not None:
                payload["provider_cleanup_failure"] = _provider_recovery_cleanup_payload(
                    cleanup_failure,
                    redactor=self._secret_redactor,
                )
            event = _event_with_model_identity_authority(
                Event(
                    type=event_type,
                    session_id=session.id,
                    interaction_id=operation.interaction_id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=payload,
                ),
                operation.model_attempt_identity,
            )
            event = event_with_execution_profile_fingerprint_authority(
                event,
                None
                if recovery_context is None
                else recovery_context.execution_profile_fingerprint,
            )
            return event_with_runtime_payload_authority(
                event,
                "operation_id",
                "stream_protocol",
            )

        await require_recovery_owner()
        await recover_context_exposure(
            store=self._session_store,
            session_id=session.id,
            stage_id=stage.stage_id,
            stage_intent=stage.intent,
            state=ContextExposureState.ACKNOWLEDGED,
            evidence_kind=ContextExposureEvidenceKind.RECOVERY_ACKNOWLEDGEMENT,
            evidence_ref=f"provider-operation:{operation.state.operation_id}:recovered",
            provider_request_id=operation.state.operation_id,
        )
        scheduled = await self._event_writer.emit(
            recovery_event(
                EventType.PROVIDER_OPERATION_RECONNECT_SCHEDULED,
                status=operation.status.value,
            )
        )
        await require_recovery_owner()
        started = await self._event_writer.emit(
            recovery_event(
                EventType.PROVIDER_OPERATION_RECONNECT_STARTED,
                status=operation.status.value,
            )
        )
        assistant_parts: list[transcript_helpers.AssistantContentPart] = []
        tool_calls: list[runtime_records.ToolCallRequest] = []
        completed_boundary = None
        completed_event: ModelStreamEvent | None = None
        completion_diagnostics: dict[str, Any] = {}
        terminal_progress_verified = False
        current_state = operation.state
        recovered_events: list[Event] = [scheduled, started]
        recovery_task = asyncio.current_task()
        recovery_cancellation_baseline = 0 if recovery_task is None else recovery_task.cancelling()
        post_completion_failure: BaseException | None = None

        async def pending_recovery_result(
            status: ProviderOperationStatus,
        ) -> ProviderOperationRecoveryResult:
            await require_recovery_owner()
            rescheduled = await self._event_writer.emit(
                recovery_event(
                    EventType.PROVIDER_OPERATION_RECONNECT_SCHEDULED,
                    status=status.value,
                )
            )
            recovered_events.append(rescheduled)
            return ProviderOperationRecoveryResult(
                status=ProviderOperationRecoveryStatus.PENDING,
                events=tuple(recovered_events),
            )

        async def unavailable_recovery_result(
            reason: ProviderOperationUnavailableReason,
            status: ProviderOperationStatus | str,
            *,
            cleanup_failure: Exception | None = None,
            deadline_failure: ModelStreamDeadlineError | None = None,
            protocol_failure: Exception | None = None,
        ) -> ProviderOperationRecoveryResult:
            authoritative_deadline: ModelStreamDeadlineError | None = None
            if deadline_failure is not None:
                controlled_failure = copy_provider_exception_control(deadline_failure)
                if type(controlled_failure.cause) is not ModelStreamDeadlineError:
                    raise RuntimeError(
                        "Provider reconnect deadline lost its typed control evidence."
                    )
                copied_deadline = controlled_failure.cause
                authoritative_deadline = ModelStreamDeadlineError(
                    provider=registered_provider.name,
                    evidence=copied_deadline.deadline_evidence,
                    stream_cleanup_failed=copied_deadline.stream_cleanup_failed,
                    recovery_disposition=EXACT_MODEL_STREAM_RECOVERY_DISPOSITION,
                )
            try:
                await require_recovery_owner()
                if authoritative_deadline is not None:
                    deadline_event = _event_with_model_identity_authority(
                        Event(
                            type=EventType.MODEL_ERROR,
                            session_id=session.id,
                            interaction_id=operation.interaction_id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            payload=_retry_attempt_payload(
                                {
                                    "error": str(authoritative_deadline),
                                    "error_type": type(authoritative_deadline).__name__,
                                    **authoritative_deadline.error_payload_fields(),
                                },
                                execution_provider_name=registered_provider.name,
                                requested_model=operation.model,
                                step=operation.step,
                                attempt=operation.attempt,
                                max_attempts=operation.max_attempts,
                                model_attempt_identity=operation.model_attempt_identity,
                            ),
                        ),
                        operation.model_attempt_identity,
                    )
                    deadline_event = event_with_execution_profile_fingerprint_authority(
                        deadline_event,
                        None
                        if recovery_context is None
                        else recovery_context.execution_profile_fingerprint,
                    )
                    recovered_events.append(await self._event_writer.emit(deadline_event))
                await recover_context_exposure(
                    store=self._session_store,
                    session_id=session.id,
                    stage_id=stage.stage_id,
                    stage_intent=stage.intent,
                    state=ContextExposureState.INDETERMINATE,
                    evidence_kind=ContextExposureEvidenceKind.RECOVERY_INDETERMINATE,
                    evidence_ref=f"provider-operation:{operation.state.operation_id}:unavailable",
                )
                status_value = (
                    status.value if isinstance(status, ProviderOperationStatus) else status
                )
                required = await _emit_provider_recovery_required_event(
                    self._event_writer,
                    recovery_event(
                        EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED,
                        status=status_value,
                        recovery_reason=reason,
                        cleanup_failure=cleanup_failure,
                        protocol_failure=protocol_failure,
                    ),
                    recovery_reason=reason,
                    cleanup_failure=cleanup_failure,
                    redactor=self._secret_redactor,
                )
                recovered_events.append(required)
            except Exception as diagnostic_failure:
                if authoritative_deadline is None:
                    raise
                raise _combine_authoritative_model_failure(
                    authoritative_deadline,
                    diagnostic_failure,
                    message=("Provider reconnect deadline and recovery diagnostics both failed."),
                ) from None
            return ProviderOperationRecoveryResult(
                status=ProviderOperationRecoveryStatus.UNAVAILABLE,
                events=tuple(recovered_events),
                unavailable_reason=reason,
            )

        def retain_post_completion_failure(failure: BaseException) -> None:
            nonlocal post_completion_failure
            post_completion_failure = _combine_post_completion_failures(
                post_completion_failure,
                failure,
            )

        def recovered_event_failure_reason(
            failure: Exception,
        ) -> ProviderOperationUnavailableReason:
            if isinstance(failure, ModelProviderError):
                return ProviderOperationUnavailableReason.FAILED
            return ProviderOperationUnavailableReason.MALFORMED

        async def accept_recovered_event(
            raw_event: object,
            *,
            persist_progress: bool,
        ) -> None:
            nonlocal completed_boundary, completed_event, completion_diagnostics
            nonlocal current_state, terminal_progress_verified
            if completed_event is not None:
                candidate = _validate_stream_event(
                    raw_event,
                    provider_name=registered_provider.name,
                    requested_model=selected_model,
                    usage_dialect=registered_provider.usage_dialect,
                ).event
                if candidate == completed_event:
                    return
                raise RuntimeError("Recovered provider operation emitted output after completion.")
            boundary = _validate_stream_event(
                raw_event,
                provider_name=registered_provider.name,
                requested_model=selected_model,
                usage_dialect=registered_provider.usage_dialect,
            )
            assistant_boundary = _validate_assistant_stream_event(
                boundary.event,
                generated_tool_call_id=_provider_operation_generated_tool_call_id(
                    stage,
                    boundary.event,
                ),
            )
            stream_event = assistant_boundary.event
            if (
                stream_event.type
                in {
                    ModelStreamEventType.THINKING,
                    ModelStreamEventType.TOOL_CALL,
                    ModelStreamEventType.HOSTED_TOOL_CALL,
                    ModelStreamEventType.CITATION,
                }
                and recovery_context is None
            ):
                kind = (
                    "thinking transcript policy"
                    if (stream_event.type is ModelStreamEventType.THINKING)
                    else (
                        "a tool continuation"
                        if stream_event.type is ModelStreamEventType.TOOL_CALL
                        else "hosted execution evidence"
                    )
                )
                raise ProviderOperationEvidenceError(
                    f"Legacy provider-operation evidence cannot safely reconstruct {kind}."
                )
            recovered_provider_error: ModelProviderError | None = None
            if stream_event.type is ModelStreamEventType.ERROR:
                stream_event, recovered_provider_error = runtime_owned_model_stream_error_event(
                    stream_event,
                    fallback_provider=registered_provider.name,
                )
            if persist_progress and stream_event.type is not ModelStreamEventType.COMPLETED:
                runtime_event = None
                if stream_event.type in {
                    ModelStreamEventType.TEXT_DELTA,
                    ModelStreamEventType.ERROR,
                    ModelStreamEventType.HOSTED_TOOL_CALL,
                    ModelStreamEventType.CITATION,
                } or (
                    stream_event.type is ModelStreamEventType.THINKING and bool(stream_event.delta)
                ):
                    runtime_event = _model_stream_event_to_runtime_event(
                        stream_event,
                        session=session,
                        requested_model=operation.model,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        provider_name=registered_provider.name,
                        step=operation.step,
                        attempt=operation.attempt,
                        max_attempts=operation.max_attempts,
                        model_attempt_identity=operation.model_attempt_identity,
                        usage_dialect=registered_provider.usage_dialect,
                        execution_profile_fingerprint=(
                            None
                            if recovery_context is None
                            else recovery_context.execution_profile_fingerprint
                        ),
                    )
                progress, emitted = await self.commit_stream_event(
                    stage=stage,
                    state=current_state,
                    stream_event=stream_event,
                    runtime_event=runtime_event,
                    session=session,
                    interaction_id=operation.interaction_id,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    environment_name=environment_name,
                    step=operation.step,
                    attempt=operation.attempt,
                    max_attempts=operation.max_attempts,
                    model_attempt_identity=operation.model_attempt_identity,
                    targeted_tool_reference_grant_ids=targeted_tool_reference_grant_ids,
                )
                current_state = progress.state
                if progress.replayed:
                    return
                if emitted is not None:
                    recovered_events.append(emitted)
            if stream_event.type is ModelStreamEventType.TEXT_DELTA:
                transcript_helpers.append_assistant_text_delta(
                    assistant_parts,
                    stream_event.delta,
                )
            elif stream_event.type is ModelStreamEventType.THINKING:
                assert recovery_context is not None
                transcript_helpers.append_assistant_thinking_delta(
                    assistant_parts,
                    stream_event.delta,
                    provider_state=stream_event.payload.get("provider_state"),
                    include=(
                        recovery_context.thinking.include_in_transcript
                        if recovery_context.thinking is not None
                        else True
                    ),
                )
            elif stream_event.type is ModelStreamEventType.TOOL_CALL:
                assert recovery_context is not None
                tool_call = assistant_boundary.tool_call
                tool_call_part = assistant_boundary.tool_call_part
                if tool_call is None or tool_call_part is None:  # pragma: no cover - helper owns it
                    raise AssertionError("Validated tool-call projection disappeared.")
                tool_calls.append(tool_call)
                assistant_parts.append(tool_call_part)
            elif stream_event.type is ModelStreamEventType.HOSTED_TOOL_CALL:
                hosted_part = _hosted_tool_call_part(
                    stream_event,
                    provider_name=registered_provider.name,
                    model=selected_model,
                    model_attempt_identity=operation.model_attempt_identity,
                )
                if hosted_part is not None:
                    assistant_parts.append(hosted_part)
            elif stream_event.type is ModelStreamEventType.CITATION:
                assistant_parts.append(
                    _citation_part(
                        stream_event,
                        provider_name=registered_provider.name,
                        model_attempt_identity=operation.model_attempt_identity,
                        assistant_parts=assistant_parts,
                    )
                )
            elif stream_event.type is ModelStreamEventType.ERROR:
                if stream_event.provider_operation_status is not None:
                    if not isinstance(recovered_provider_error, ModelProviderError):
                        raise RuntimeError(
                            "Provider-operation status requires a typed provider error."
                        )
                    raise _ProviderOperationStreamStatusError(
                        stream_event.provider_operation_status,
                        recovered_provider_error,
                    ) from recovered_provider_error
                raise recovered_provider_error or RuntimeError(
                    "Recovered provider operation failed."
                )
            elif stream_event.type is ModelStreamEventType.COMPLETED:
                completed_boundary = boundary
                completed_event = stream_event
                if persist_progress:
                    envelope = provider_operation_progress_envelope(
                        current_state,
                        stream_event,
                    )
                    if _provider_operation_progress_contains_secret(
                        envelope,
                        redactor=self._secret_redactor,
                        targeted_tool_reference_grant_ids=targeted_tool_reference_grant_ids,
                    ):
                        retain_post_completion_failure(
                            ProviderOperationEvidenceError(
                                "Provider-operation terminal recovery state contains a workload "
                                "secret."
                            )
                        )
                    else:
                        terminal_cursor = envelope.recovery_metadata.cursor
                        current_cursor = current_state.recovery_metadata.cursor
                        current_cursor = -1 if current_cursor is None else current_cursor
                        if terminal_cursor != current_cursor + 1:
                            retain_post_completion_failure(
                                ProviderOperationEvidenceError(
                                    "Provider-operation terminal cursor is not the next boundary."
                                )
                            )
                        else:
                            terminal_progress_verified = True
                if boundary.completion_error is not None:
                    code, path = safe_durable_value_error_details(boundary.completion_error)
                    completion_error = ModelProviderError(
                        "Recovered provider operation returned invalid completion metadata.",
                        provider=registered_provider.name,
                        error_type="DurableValueError",
                        error_code="invalid_model_completion_value",
                        retryable=False,
                    )
                    completion_diagnostics = {
                        "completion_outcome": "invalid_metadata",
                        "completion_error": {
                            "error": str(completion_error),
                            "error_type": type(completion_error).__name__,
                            "durable_value_error_code": code,
                            "durable_value_path": path,
                            **completion_error.error_payload_fields(),
                        },
                    }
                    retain_post_completion_failure(completion_error)
            else:  # pragma: no cover - boundary validation owns the closed event vocabulary
                raise RuntimeError("Recovered provider operation returned an unsupported event.")

        for accepted_event in operation.accepted_stream_events:
            if (
                accepted_event.type is ModelStreamEventType.ERROR
                and accepted_event.provider_operation_status
                in {
                    ProviderOperationStatus.QUEUED,
                    ProviderOperationStatus.IN_PROGRESS,
                }
            ):
                # The error evidence and cursor are already durable. A
                # nonterminal provider status means recovery should continue
                # after that boundary rather than replaying the model error.
                continue
            await accept_recovered_event(accepted_event, persist_progress=False)

        await require_recovery_owner()
        if operation.accepted_stream_events:
            try:
                raw_connection = await adapter.reconnect(
                    copy_provider_operation_state(operation.state)
                )
            except BaseException as recovery_failure:
                recovery_failure = _classify_provider_recovery_failure(
                    recovery_failure,
                    cancellation_baseline=recovery_cancellation_baseline,
                    operation="Provider operation reconnect",
                )
                if not isinstance(recovery_failure, Exception):
                    raise recovery_failure
                malformed = isinstance(recovery_failure, ProviderOperationMalformedError)
                return await unavailable_recovery_result(
                    (
                        ProviderOperationUnavailableReason.MALFORMED
                        if malformed
                        else ProviderOperationUnavailableReason.UNAVAILABLE
                    ),
                    "malformed" if malformed else ProviderOperationStatus.UNAVAILABLE,
                    protocol_failure=recovery_failure,
                    deadline_failure=(
                        recovery_failure
                        if isinstance(recovery_failure, ModelStreamDeadlineError)
                        else None
                    ),
                )
            try:
                connection = copy_provider_operation_connection(raw_connection)
            except Exception as malformed_failure:
                cleanup_failure = None
                if type(raw_connection) is ProviderOperationConnection:
                    cleanup_failure = await _close_provider_recovery_iterator(
                        raw_connection.events,
                        cancellation_baseline=recovery_cancellation_baseline,
                        operation="Provider operation malformed reconnect stream cleanup",
                    )
                _raise_pending_provider_recovery_cancellation(
                    cancellation_baseline=recovery_cancellation_baseline
                )
                if cleanup_failure is not None:
                    add_exception_note_safely(
                        malformed_failure,
                        "Provider recovery stream cleanup also failed with "
                        f"{type(cleanup_failure).__name__}.",
                    )
                return await unavailable_recovery_result(
                    ProviderOperationUnavailableReason.MALFORMED,
                    "malformed",
                    cleanup_failure=cleanup_failure,
                )
            recovery_task = asyncio.current_task()
            if (
                recovery_task is not None
                and recovery_task.cancelling() > recovery_cancellation_baseline
            ):
                await _close_provider_recovery_iterator(
                    connection.events,
                    cancellation_baseline=recovery_cancellation_baseline,
                    operation=(
                        "Provider operation reconnect stream cleanup after suppressed caller "
                        "cancellation"
                    ),
                )
                raise AssertionError(
                    "Provider reconnect cleanup returned with caller cancellation pending."
                )
            reconnect_unavailable: (
                tuple[
                    ProviderOperationUnavailableReason,
                    ProviderOperationStatus | str,
                ]
                | None
            ) = None
            reconnect_pending_status: ProviderOperationStatus | None = None
            reconnect_cleanup_failure: Exception | None = None
            reconnect_protocol_failure: Exception | None = None
            reconnect_deadline_failure: ModelStreamDeadlineError | None = None
            try:
                async with aclosing_provider_stream(connection.events) as reconnect_events:
                    if connection.state != operation.state:
                        reconnect_unavailable = (
                            ProviderOperationUnavailableReason.WRONG_PROVIDER,
                            "wrong_provider",
                        )
                    else:
                        recovery_status = connection.status
                        reconnect_iterator = aiter(reconnect_events)
                        while True:
                            redactor_token = bind_provider_error_workload_redactor(
                                self._secret_redactor
                            )
                            try:
                                try:
                                    raw_event = await anext(reconnect_iterator)
                                except StopAsyncIteration:
                                    break
                            finally:
                                reset_provider_error_workload_redactor(redactor_token)
                            await require_recovery_owner()
                            try:
                                await accept_recovered_event(raw_event, persist_progress=True)
                            except _ProviderOperationStreamStatusError as status_failure:
                                if completed_event is None:
                                    if status_failure.status in {
                                        ProviderOperationStatus.QUEUED,
                                        ProviderOperationStatus.IN_PROGRESS,
                                    }:
                                        reconnect_pending_status = status_failure.status
                                    else:
                                        reason = provider_operation_unavailable_reason(
                                            status_failure.status
                                        )
                                        if reason is None:
                                            reason = ProviderOperationUnavailableReason.MALFORMED
                                        reconnect_unavailable = (
                                            reason,
                                            status_failure.status,
                                        )
                                else:
                                    retain_post_completion_failure(status_failure.provider_error)
                                break
                            except Exception as event_failure:
                                if completed_event is None:
                                    reason = recovered_event_failure_reason(event_failure)
                                    reconnect_unavailable = (reason, reason.value)
                                else:
                                    retain_post_completion_failure(event_failure)
                                break
                            if completed_event is None:
                                continue
                            break
            except BaseException as stream_failure:
                stream_failure = _classify_provider_recovery_failure(
                    stream_failure,
                    cancellation_baseline=recovery_cancellation_baseline,
                    operation="Provider operation recovery stream",
                )
                if completed_event is None:
                    if isinstance(stream_failure, Exception):
                        if reconnect_unavailable is None:
                            if isinstance(stream_failure, ModelStreamDeadlineError):
                                reconnect_deadline_failure = stream_failure
                            reconnect_protocol_failure = stream_failure
                            malformed = isinstance(stream_failure, ProviderOperationMalformedError)
                            reconnect_unavailable = (
                                (
                                    ProviderOperationUnavailableReason.MALFORMED
                                    if malformed
                                    else ProviderOperationUnavailableReason.UNAVAILABLE
                                ),
                                ("malformed" if malformed else ProviderOperationStatus.UNAVAILABLE),
                            )
                        else:
                            reconnect_cleanup_failure = stream_failure
                    else:
                        raise
                else:
                    retain_post_completion_failure(stream_failure)
            if reconnect_unavailable is not None and completed_event is None:
                return await unavailable_recovery_result(
                    *reconnect_unavailable,
                    cleanup_failure=reconnect_cleanup_failure,
                    deadline_failure=reconnect_deadline_failure,
                    protocol_failure=reconnect_protocol_failure,
                )
            if reconnect_pending_status is not None and completed_event is None:
                return await pending_recovery_result(reconnect_pending_status)
        else:
            try:
                raw_snapshot = await adapter.retrieve(
                    copy_provider_operation_state(operation.state)
                )
                _raise_pending_provider_recovery_cancellation(
                    cancellation_baseline=recovery_cancellation_baseline
                )
            except BaseException as recovery_failure:
                recovery_failure = _classify_provider_recovery_failure(
                    recovery_failure,
                    cancellation_baseline=recovery_cancellation_baseline,
                    operation="Provider operation retrieval",
                )
                if not isinstance(recovery_failure, Exception):
                    raise recovery_failure
                malformed = isinstance(recovery_failure, ProviderOperationMalformedError)
                return await unavailable_recovery_result(
                    (
                        ProviderOperationUnavailableReason.MALFORMED
                        if malformed
                        else ProviderOperationUnavailableReason.UNAVAILABLE
                    ),
                    "malformed" if malformed else ProviderOperationStatus.UNAVAILABLE,
                    protocol_failure=recovery_failure,
                )
            try:
                snapshot = copy_provider_operation_snapshot(raw_snapshot)
            except Exception:
                return await unavailable_recovery_result(
                    ProviderOperationUnavailableReason.MALFORMED,
                    "malformed",
                )
            if snapshot.state != operation.state:
                return await unavailable_recovery_result(
                    ProviderOperationUnavailableReason.WRONG_PROVIDER,
                    "wrong_provider",
                )
            recovery_status = snapshot.status
            if snapshot.status in {
                ProviderOperationStatus.QUEUED,
                ProviderOperationStatus.IN_PROGRESS,
            }:
                return await pending_recovery_result(snapshot.status)
            for raw_event in snapshot.events:
                try:
                    await accept_recovered_event(raw_event, persist_progress=False)
                except _ProviderOperationStreamStatusError as status_failure:
                    if completed_event is None:
                        if status_failure.status in {
                            ProviderOperationStatus.QUEUED,
                            ProviderOperationStatus.IN_PROGRESS,
                        }:
                            return await pending_recovery_result(status_failure.status)
                        reason = provider_operation_unavailable_reason(status_failure.status)
                        if reason is None:
                            reason = ProviderOperationUnavailableReason.MALFORMED
                        return await unavailable_recovery_result(
                            reason,
                            status_failure.status,
                        )
                    retain_post_completion_failure(status_failure.provider_error)
                    break
                except Exception as snapshot_failure:
                    if completed_event is None:
                        reason = recovered_event_failure_reason(snapshot_failure)
                        return await unavailable_recovery_result(reason, reason.value)
                    retain_post_completion_failure(snapshot_failure)
                    break

        try:
            await require_recovery_owner()
        except (SessionInterruptedByRequest, asyncio.CancelledError) as recovery_failure:
            if completed_event is None:
                raise
            post_completion_failure = _combine_post_completion_failures(
                post_completion_failure,
                recovery_failure,
            )
        if completed_event is None or completed_boundary is None:
            if recovery_status in {
                ProviderOperationStatus.QUEUED,
                ProviderOperationStatus.IN_PROGRESS,
            }:
                return await pending_recovery_result(recovery_status)
            unavailable_reason = provider_operation_unavailable_reason(recovery_status)
            if unavailable_reason is None:
                raise RuntimeError("Provider operation returned an unknown terminal status.")
            return await unavailable_recovery_result(unavailable_reason, recovery_status)
        if recovery_status not in {
            ProviderOperationStatus.QUEUED,
            ProviderOperationStatus.IN_PROGRESS,
            ProviderOperationStatus.COMPLETED,
        }:
            retain_post_completion_failure(
                RuntimeError(
                    "Provider operation returned completion output with conflicting terminal "
                    f"status: {recovery_status.value}."
                )
            )

        completion_semantics_valid = completed_boundary.completion_error is None
        billing_identity = None if recovery_context is None else recovery_context.billing_identity
        if completion_semantics_valid:
            try:
                billing_identity = resolve_completion_billing_identity(
                    provider,
                    billing_identity,
                    copy_durable_json_object(completed_event.payload, "completed_payload"),
                    provider_name=registered_provider.name,
                )
            except ModelProviderError as billing_error:
                completion_semantics_valid = False
                completion_diagnostics = {
                    "completion_outcome": "billing_identity_resolution_failed",
                    "completion_error": {
                        "error": str(billing_error),
                        "error_type": type(billing_error).__name__,
                        "stage": "billing_identity_for_completion",
                        **billing_error.error_payload_fields(),
                    },
                }
                retain_post_completion_failure(billing_error)

        assistant_message: Message | None = None
        step_result: AssistantStepResult | None = None
        classification = None
        hosted_discovery_mutations: tuple[RuntimePublicationOperationRecordMutation, ...] = ()
        hosted_discovery_grant_ids: dict[str, str] = {}
        if completion_semantics_valid:
            try:
                (
                    hosted_discovery_mutations,
                    hosted_discovery_grant_ids,
                ) = await _hosted_tool_discovery_publication_authority(
                    session_store=self._session_store,
                    session=session,
                    registered_agent=registered_agent,
                    projection=hosted_discovery_projection,
                    stream_event=completed_event,
                    tool_calls=tool_calls,
                    model_step_id=operation.model_attempt_identity.model_step_id,
                    created_at=self._clock(),
                )
            except (TypeError, ValueError) as exc:
                completion_semantics_valid = False
                hosted_discovery_error = ModelProviderError(
                    "Recovered provider operation returned invalid hosted Tool Search evidence.",
                    provider=registered_provider.name,
                    error_type=type(exc).__name__,
                    error_code="invalid_tool_discovery_projection",
                    retryable=False,
                )
                completion_diagnostics = {
                    "completion_outcome": "invalid_tool_discovery_projection",
                    "completion_error": {
                        "error": str(hosted_discovery_error),
                        "error_type": type(hosted_discovery_error).__name__,
                        "stage": "hosted_tool_discovery_validation",
                        **hosted_discovery_error.error_payload_fields(),
                    },
                }
                retain_post_completion_failure(hosted_discovery_error)
        if completion_semantics_valid:
            try:
                _require_unique_tool_call_ids(tool_calls)
                provider_state_parts = transcript_helpers.provider_state_parts(
                    completed_event.payload
                )
                assistant_message = transcript_helpers.assistant_message(
                    content_parts=assistant_parts,
                    provider_state_parts=provider_state_parts,
                )
                step_result = _assistant_step_result(
                    session_id=session.id,
                    step=operation.step,
                    model_attempt_identity=operation.model_attempt_identity,
                    assistant_message=assistant_message,
                    tool_calls=tool_calls,
                    completion=_stream_event_completion(completed_event),
                )
                classification = classify_assistant_step(step_result)
            except (TypeError, ValueError):
                completion_semantics_valid = False
                transcript_error = ModelProviderError(
                    "Recovered provider operation returned invalid completion transcript state.",
                    provider=registered_provider.name,
                    error_type="ValueError",
                    error_code="invalid_model_completion_transcript",
                    retryable=False,
                )
                completion_diagnostics = {
                    "completion_outcome": "invalid_transcript_state",
                    "completion_error": {
                        "error": str(transcript_error),
                        "error_type": type(transcript_error).__name__,
                        "stage": "completion_transcript_projection",
                        **transcript_error.error_payload_fields(),
                    },
                }
                retain_post_completion_failure(transcript_error)

        completion_event = _model_stream_event_to_runtime_event(
            completed_event,
            session=session,
            requested_model=operation.model,
            registered_agent=registered_agent,
            environment_name=environment_name,
            provider_name=registered_provider.name,
            step=operation.step,
            attempt=operation.attempt,
            max_attempts=operation.max_attempts,
            model_attempt_identity=operation.model_attempt_identity,
            tool_round_identity=(None if step_result is None else step_result.tool_round_identity),
            classification=None if classification is None else classification.payload(),
            transcript_cursor_after_completion=(
                stage.source_transcript_cursor
                + int(assistant_message is not None and not tool_calls)
            ),
            input_coverage=(
                ContextInputCoverage.model_validate(stage.intent["input_coverage"])
                if "input_coverage" in stage.intent
                else None
            ),
            usage_dialect=registered_provider.usage_dialect,
            billing_identity=billing_identity,
            accounting_usage_metrics=completed_boundary.accounting_usage_metrics,
            accounting_usage_rejected=completed_boundary.accounting_usage_rejected,
            usage_normalization_failed=completed_boundary.usage_normalization_failed,
            completion_diagnostics=completion_diagnostics,
            execution_profile_fingerprint=(
                None if recovery_context is None else recovery_context.execution_profile_fingerprint
            ),
        )
        completion_event = completion_event.model_copy(
            update={"interaction_id": operation.interaction_id},
            deep=True,
        )
        if terminal_progress_verified:
            completion_event = self.progress_event(
                stage=stage,
                state=current_state,
                stream_event=completed_event,
                runtime_event=completion_event,
                session=session,
                interaction_id=operation.interaction_id,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                environment_name=environment_name,
                step=operation.step,
                attempt=operation.attempt,
                max_attempts=operation.max_attempts,
                model_attempt_identity=operation.model_attempt_identity,
                targeted_tool_reference_grant_ids=targeted_tool_reference_grant_ids,
            )
        publication_cancellation = _take_model_completion_cancellation(
            post_completion_failure,
            cancellation_baseline=recovery_cancellation_baseline,
        )
        if publication_cancellation is not None and post_completion_failure is None:
            post_completion_failure = publication_cancellation
        elif publication_cancellation is None and isinstance(
            post_completion_failure,
            asyncio.CancelledError,
        ):
            post_completion_failure = unexpected_child_cancellation_error(
                post_completion_failure,
                operation="Provider operation recovery stream",
            )
        durable_step_result = None
        if step_result is not None:
            try:
                durable_step_result = _durable_assistant_step_result(
                    step_result,
                    redactor=self._secret_redactor,
                    targeted_tool_reference_grant_ids=targeted_tool_reference_grant_ids,
                    native_tool_name_grant_ids={
                        **hosted_replay_grant_ids,
                        **hosted_discovery_grant_ids,
                    },
                )
            except (TypeError, ValueError):
                durable_boundary_error = ModelProviderError(
                    "Recovered provider operation returned assistant output that cannot cross "
                    "the durable publication boundary.",
                    provider=registered_provider.name,
                    error_type="DurableBoundaryError",
                    error_code="invalid_model_completion_transcript",
                    retryable=False,
                )
                retain_post_completion_failure(durable_boundary_error)
        structured_output_validation = None
        if (
            post_completion_failure is None
            and recovery_context is not None
            and recovery_context.structured_output is not None
            and recovery_context.structured_output.strategy is StructuredOutputStrategy.TOOL
            and any(call.name == STRUCTURED_OUTPUT_TOOL_NAME for call in tool_calls)
        ):
            try:
                structured_output_validation = _redact_structured_output_validation(
                    _validate_structured_output_tool_round(
                        tool_calls=tool_calls,
                        spec=recovery_context.structured_output,
                    ),
                    self._secret_redactor,
                )
            except (TypeError, ValueError):
                structured_output_error = ModelProviderError(
                    "Recovered provider operation returned invalid structured output.",
                    provider=registered_provider.name,
                    error_type="ValueError",
                    error_code="invalid_model_completion_transcript",
                    retryable=False,
                )
                retain_post_completion_failure(structured_output_error)
        request_fingerprint = stage.intent.get("request_fingerprint")
        if type(request_fingerprint) is not str:
            raise RuntimeError("Provider-operation stage lost its request fingerprint.")
        authoritative_assistant_message = (
            durable_step_result.assistant_message
            if durable_step_result is not None and post_completion_failure is None
            else None
        )
        publication_event = (
            completion_event
            if post_completion_failure is None
            else _non_turn_model_completion_event(
                completion_event,
                failure=post_completion_failure,
                cancellation=publication_cancellation,
                transcript_cursor=stage.source_transcript_cursor,
            )
        )
        if stage.reservation_ids:
            if recovery_context is None:
                raise ProviderOperationEvidenceError(
                    "Budgeted provider-operation recovery has no durable accounting context."
                )
            try:
                publication_event = (
                    await self._run_limit_controller.recover_model_completion_budget_evidence(
                        publication_event,
                        reservation_ids=stage.reservation_ids,
                        recovery_contexts=recovery_context.budget_reservations,
                        session=session,
                        provider_name=registered_provider.name,
                        model_attempt_identity=operation.model_attempt_identity,
                        dispatch_id=stage.stage_id,
                        request_billing_identity=recovery_context.billing_identity,
                    )
                )
            except (KeyError, NotImplementedError, TypeError, ValueError) as accounting_error:
                raise ProviderOperationEvidenceError(
                    "Provider-operation recovery could not reconstruct its original budget "
                    "reservation and pricing context."
                ) from accounting_error
        tool_exposure = None if recovery_context is None else recovery_context.tool_exposure
        if (
            durable_step_result is not None
            and durable_step_result.tool_calls
            and tool_exposure is None
        ):
            raise ProviderOperationEvidenceError(
                "Provider-operation recovery has no durable tool-exposure authority."
            )
        publication_event = self._event_writer.prepare(publication_event)
        publication = ModelCompletionPublicationRequest(
            dispatch=ModelCompletionDispatch(
                stage=stage,
                request_fingerprint=request_fingerprint,
            ),
            assistant_step_result=durable_step_result,
            completion_event=publication_event,
            authoritative_assistant_message=authoritative_assistant_message,
            defer_assistant_message=bool(
                durable_step_result is not None
                and durable_step_result.tool_calls
                and post_completion_failure is None
            ),
            structured_output_validation=structured_output_validation,
            tool_exposure=tool_exposure,
            operation_record_mutations=(
                () if post_completion_failure is not None else hosted_discovery_mutations
            ),
        )
        await recover_context_exposure(
            store=self._session_store,
            session_id=session.id,
            stage_id=stage.stage_id,
            stage_intent=stage.intent,
            state=ContextExposureState.COMPLETED,
            evidence_kind=ContextExposureEvidenceKind.RECOVERY_COMPLETION,
            evidence_ref=f"provider-operation:{operation.state.operation_id}:completed",
            provider_request_id=operation.state.operation_id,
        )
        await _publish_model_completion(
            model_completion_publisher,
            publication,
            terminal_failure=post_completion_failure,
            publication_cancellation=publication_cancellation,
        )
        if post_completion_failure is not None:
            raise post_completion_failure
        reconciled = recovery_event(
            EventType.PROVIDER_OPERATION_RECONCILED,
            status=recovery_status.value,
        )
        try:
            reconciled = await self._event_writer.emit(reconciled)
        except Exception as delivery_error:
            logger.warning(
                "Provider operation completed durably but reconciliation telemetry failed: "
                "session_id=%s operation_id=%s error_type=%s",
                session.id,
                operation.state.operation_id,
                type(delivery_error).__name__,
            )
        recovered_events.append(reconciled)
        if recovery_under_cancellation_claim:
            if stage.reservation_ids:
                await self._run_limit_controller.reconcile_model_completion_settlements(
                    publication_event,
                    reservation_ids=stage.reservation_ids,
                )
            await self._provider_operation_cancellation.release_claim(
                session=session,
                claim=cancellation_claim,
            )
        return ProviderOperationRecoveryResult(
            status=ProviderOperationRecoveryStatus.RECONCILED,
            events=tuple(recovered_events),
            completion_event=publication_event,
        )
