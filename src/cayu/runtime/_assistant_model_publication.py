"""Shared durable assistant publication for live execution and provider recovery."""

from __future__ import annotations

import asyncio
from typing import Any

from cayu._task_wait import await_shielded_task_outcome, unexpected_child_cancellation_error
from cayu._validation import copy_durable_record
from cayu.budgets._run_limit_accounting import RunLimitAccountingContext
from cayu.budgets.base import BudgetLimit
from cayu.budgets.run_limits import RunLimits
from cayu.context.structured_output import StructuredOutputSpec
from cayu.context.thinking import ThinkingConfig
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._durable_tool_round import _environment_name as _environment_name
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._model_completion_contracts import (
    ModelCompletionPublicationRequest,
    ModelCompletionPublicationResult,
    ModelCompletionPublisher,
    ModelCompletionRecoveryContext,
)
from cayu.runtime._structured_output_tool_round import _has_structured_output_tool_call
from cayu.runtime._tool_round_staging import _redactor_for_tool_calls
from cayu.sessions import _model_completion_publication as model_completion_publication
from cayu.sessions.base import (
    RuntimePublicationRequest,
    SessionStore,
    runtime_publication_checkpoint_mutation,
)
from cayu.sessions.records import Session
from cayu.tools.catalogue import CALL_TOOL_NAME
from cayu.tools.exposure import validate_resolved_tool_exposure_authority
from cayu.vaults.redaction import SecretRedactor


class AssistantModelPublication:
    """Publish a model result and its next action through the store's exact protocol.

    Live execution supplies its current semantics. Provider recovery binds its
    saved semantics with recovery_publisher. Both use the same completion,
    promotion and event delivery sequence without depending on a session engine.
    """

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        secret_redactor: SecretRedactor,
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._secret_redactor = secret_redactor

    async def publish(
        self,
        publication: ModelCompletionPublicationRequest,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        task_id: str | None,
        request_metadata: dict[str, Any],
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        max_steps: int,
        limits: RunLimits,
        budget_limits: tuple[BudgetLimit, ...],
        retry_policy: RetryPolicy,
        structured_output_attempt: int | None,
        structured_output_retries: int,
        run_limit_accounting: RunLimitAccountingContext | None,
    ) -> ModelCompletionPublicationResult:
        """Commit one staged model completion and its next durable action atomically."""

        if publication.dispatch.stage.session_id != session.id:
            raise RuntimeError("Model completion publication belongs to a different session.")
        assistant_step_result = publication.assistant_step_result
        assistant_message = publication.authoritative_assistant_message
        tool_calls = (
            assistant_step_result.tool_calls
            if assistant_message is not None and assistant_step_result is not None
            else []
        )
        tool_round_identity = (
            assistant_step_result.tool_round_identity
            if assistant_message is not None and assistant_step_result is not None
            else None
        )
        if bool(tool_calls) != (tool_round_identity is not None):
            raise RuntimeError(
                "Model completion tool calls and tool-round identity must be published together."
            )
        source_checkpoint = await self._session_store.load_checkpoint(session.id)
        target_checkpoint = source_checkpoint
        tool_round_id = None if tool_round_identity is None else tool_round_identity.tool_round_id
        if tool_calls:
            if tool_round_identity is None or assistant_step_result is None:
                raise RuntimeError("Model completion lost its tool-round execution material.")
            if publication.tool_exposure is None:
                raise RuntimeError("Model completion lost its frozen tool exposure.")
            tool_exposure = validate_resolved_tool_exposure_authority(
                publication.tool_exposure,
                registered_agent.tool_capabilities,
                catalogue_revision=registered_agent.tool_catalogue.revision,
            )
            tool_redactor = _redactor_for_tool_calls(
                self._secret_redactor,
                registered_agent=registered_agent,
                tool_calls=tool_calls,
            )
            target_checkpoint, _pending_round = (
                tool_round_recovery.checkpoint_with_pending_tool_round(
                    source_checkpoint,
                    agent_name=registered_agent.spec.name,
                    interaction_id=(
                        publication.completion_event.interaction_id
                        if any(
                            call.name == CALL_TOOL_NAME
                            or call.targeted_tool_grant_id is not None
                            or call.targeted_tool_invocation is not None
                            or call.targeted_tool_rejection is not None
                            for call in tool_calls
                        )
                        else None
                    ),
                    environment_name=_environment_name(registered_environment),
                    task_id=task_id,
                    source_run_epoch=publication.dispatch.stage.source_run_epoch,
                    tool_calls=tool_calls,
                    policy_outcomes=None,
                    tool_exposure=tool_exposure,
                    policy_context_version=1,
                    request_metadata=request_metadata,
                    assistant_message_state=(
                        "quarantined" if publication.defer_assistant_message else "published"
                    ),
                    quarantined_assistant_message=(
                        assistant_message if publication.defer_assistant_message else None
                    ),
                    secret_resolution_scope=invocation_secrets.registered_environment_secret_resolution_scope(
                        registered_environment
                    ),
                    continuity_tool_names=frozenset(
                        tool.name
                        for tool in registered_agent.tools.values()
                        if tool.retain_arguments_for_model and not tool.publish_arguments
                    ),
                    continuity_knowledge_scope=(
                        None
                        if registered_environment is None
                        else registered_environment.environment.knowledge_access_scope
                    ),
                    structured_output=structured_output,
                    thinking=thinking,
                    max_steps=max_steps,
                    limits=limits,
                    run_limit_accounting=run_limit_accounting,
                    budget_limits=budget_limits,
                    retry_policy=retry_policy,
                    tool_round_identity=tool_round_identity,
                    redactor=tool_redactor,
                    source_model_step_id=tool_round_identity.model_step_id,
                    source_transcript_cursor=(publication.dispatch.stage.source_transcript_cursor),
                    model_step=assistant_step_result.step,
                    structured_output_attempt=structured_output_attempt,
                    structured_output_retries=structured_output_retries,
                    structured_output_validation=(publication.structured_output_validation),
                    runtime_session=session,
                )
            )
            if (
                _pending_round.assistant_publication is not None
                and _pending_round.assistant_publication.argument_continuity is not None
                and not self._session_store.supports_private_argument_continuity
            ):
                raise RuntimeError("Session store does not support private argument continuity.")
        target_checkpoint = (
            {}
            if target_checkpoint is None
            else copy_durable_record(target_checkpoint, "model_completion_checkpoint")
        )
        classification = publication.completion_event.payload.get("step_classification")
        if type(classification) is not dict:
            raise RuntimeError(
                "Model completion publication requires a durable step classification."
            )
        pointer = model_completion_publication.ModelStepPublicationCheckpoint(
            logical_step_id=publication.dispatch.logical_step_id,
            stage_id=publication.dispatch.stage_id,
            source_transcript_cursor=(publication.dispatch.stage.source_transcript_cursor),
            transcript_end_cursor=(
                publication.dispatch.stage.source_transcript_cursor
                + int(assistant_message is not None and not publication.defer_assistant_message)
            ),
            completion_event_id=publication.completion_event.id,
            classification=classification,
            assistant_message_published=(
                assistant_message is not None and not publication.defer_assistant_message
            ),
            assistant_message_deferred=publication.defer_assistant_message,
            tool_round_id=tool_round_id,
        )
        target_checkpoint[
            model_completion_publication.LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY
        ] = pointer.model_dump(mode="json")
        runtime_publication = RuntimePublicationRequest(
            publication_id=publication.dispatch.logical_step_id,
            kind="model-step",
            interaction_id=publication.completion_event.interaction_id,
            intent=publication.dispatch.intent,
            mutation=runtime_publication_checkpoint_mutation(
                source_checkpoint,
                target_checkpoint,
            ),
            transcript_messages=(
                ()
                if assistant_message is None or publication.defer_assistant_message
                else (assistant_message,)
            ),
            events=(publication.completion_event,),
            operation_record_mutations=publication.operation_record_mutations,
        )

        async def commit_and_fan_out_once() -> ModelCompletionPublicationResult:
            completion = await self._session_store.complete_model_completion_stage(
                session.id,
                stage_id=publication.dispatch.stage_id,
                publication=runtime_publication,
            )
            promoted = await self._session_store.promote_model_completion_stage(
                session.id,
                stage_id=publication.dispatch.stage_id,
                expected_run_epoch=session.run_epoch,
            )
            await self._event_writer.fan_out_persisted([publication.completion_event])
            return ModelCompletionPublicationResult(
                completion=completion,
                publication=promoted,
            )

        async def commit_and_fan_out() -> ModelCompletionPublicationResult:
            try:
                return await commit_and_fan_out_once()
            except Exception as first_error:
                try:
                    return await commit_and_fan_out_once()
                except Exception as replay_error:
                    replay_error.add_note(
                        "Exact model-completion publication replay also failed after "
                        f"{type(first_error).__name__}: {first_error}"
                    )
                    raise replay_error from first_error

        commit_task = asyncio.create_task(commit_and_fan_out())
        outcome = await await_shielded_task_outcome(commit_task)
        cancellation = outcome.cancellation
        error = outcome.error
        if isinstance(error, asyncio.CancelledError) and cancellation is None:
            error = unexpected_child_cancellation_error(
                error,
                operation="Model completion durable publication",
            )
        if error is not None:
            if cancellation is not None:
                cancellation.add_note(
                    "Model completion durable publication also failed: "
                    f"{type(error).__name__}: {error}"
                )
                raise cancellation from error
            raise error
        if outcome.result is None:
            result_error = RuntimeError(
                "Model completion durable publication returned no acknowledgement."
            )
            if cancellation is not None:
                cancellation.add_note(str(result_error))
                raise cancellation from result_error
            raise result_error
        if cancellation is not None:
            raise cancellation
        return outcome.result

    def recovery_publisher(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        publication_context: ModelCompletionRecoveryContext,
    ) -> ModelCompletionPublisher:
        """Bind saved recovery semantics to this same publication operation."""

        async def publish(
            publication: ModelCompletionPublicationRequest,
        ) -> ModelCompletionPublicationResult:
            return await self.publish(
                publication,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                task_id=publication_context.task_id,
                request_metadata=publication_context.request_metadata,
                structured_output=publication_context.structured_output,
                thinking=publication_context.thinking,
                max_steps=publication_context.max_steps,
                limits=publication_context.limits,
                budget_limits=publication_context.budget_limits,
                retry_policy=publication_context.retry_policy,
                structured_output_attempt=(
                    publication_context.structured_output_attempt
                    if (
                        publication.assistant_step_result is not None
                        and _has_structured_output_tool_call(
                            publication.assistant_step_result.tool_calls
                        )
                    )
                    else None
                ),
                structured_output_retries=(
                    max(publication_context.structured_output_attempt - 1, 0)
                    if publication_context.structured_output_attempt is not None
                    else 0
                ),
                run_limit_accounting=publication_context.run_limit_accounting,
            )

        return publish
