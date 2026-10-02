"""Durable assistant projection and exact model-completion acknowledgement."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from cayu._exception_groups import add_exception_note_safely, iter_exception_tree
from cayu._task_wait import (
    await_shielded_task_outcome,
    consume_pending_task_cancellation,
    unexpected_child_cancellation_error,
)
from cayu._validation import copy_durable_json_object, copy_json_value, require_nonblank
from cayu.events import Event, copy_event
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._model_completion_contracts import (
    ModelCompletionPublicationRequest,
    ModelCompletionPublicationResult,
    ModelCompletionPublisher,
    _copy_assistant_step_result,
)
from cayu.runtime._session_control import SessionInterruptedByRequest
from cayu.runtime.model_steps import (
    AssistantStepResult,
    assistant_text_content,
    provider_state_count,
    thinking_count,
)
from cayu.runtime.retry_policy import RetryDecision, RetrySuppression
from cayu.tools.catalogue import CALL_TOOL_NAME
from cayu.vaults.redaction import SecretRedactor


class ModelAttemptFailed(Exception):
    """A single provider attempt failed after zero or more streamed events.

    ``completion_observed`` is set only by the durable publication path. Once a
    valid completed frame has crossed that boundary, a later transport/control
    error is terminal and cannot authorize another provider dispatch.
    ``provider_effect_observed`` records whether any valid non-error provider
    frame was observed, so terminal classification cannot treat a late auth
    error as proof that the provider rejected the request before all effects.
    """

    def __init__(
        self,
        *,
        message: str,
        payload: dict[str, Any],
        emitted_error_event: bool,
        cause: Exception | None = None,
        completion_observed: bool = False,
        provider_effect_observed: bool = False,
        automatic_retry_disabled: bool = False,
        retry_decision: RetryDecision | None = None,
        retry_suppression: RetrySuppression | None = None,
    ) -> None:
        if type(provider_effect_observed) is not bool:
            raise TypeError("provider_effect_observed must be a bool.")
        if type(automatic_retry_disabled) is not bool:
            raise TypeError("automatic_retry_disabled must be a bool.")
        self.message = require_nonblank(message, "message")
        self.payload = copy_json_value(payload, "payload")
        self.emitted_error_event = emitted_error_event
        self.cause = cause
        self.completion_observed = completion_observed
        self.provider_effect_observed = provider_effect_observed
        self.automatic_retry_disabled = automatic_retry_disabled
        if retry_decision is not None and type(retry_decision) is not RetryDecision:
            raise TypeError("retry_decision must be a RetryDecision or None.")
        if retry_suppression is not None and type(retry_suppression) is not RetrySuppression:
            raise TypeError("retry_suppression must be a RetrySuppression or None.")
        self.retry_suppression = retry_suppression
        self.retry_decision = retry_decision
        super().__init__(self.message)


def _durable_assistant_step_result(
    result: AssistantStepResult,
    *,
    redactor: SecretRedactor,
    targeted_tool_reference_grant_ids: Mapping[str, str] | None = None,
    native_tool_name_grant_ids: Mapping[str, str] | None = None,
) -> AssistantStepResult:
    """Project one assistant result across the durable publication boundary."""

    copied = _copy_assistant_step_result(result)
    if copied.assistant_message is None:
        if copied.tool_calls:
            raise ValueError("Assistant tool calls require an assistant message.")
        return copied
    # The model never owned tool-round identifiers. Evaluate only its content
    # against workload secrets, then restore the exact runtime lineage.
    assistant_message = transcript_helpers.redact_untrusted_assistant_message_for_boundary(
        copied.assistant_message,
        tool_round_identity=copied.tool_round_identity,
        redactor=redactor,
        field_name="assistant_message",
    )
    reference_grant_ids = (
        {} if targeted_tool_reference_grant_ids is None else dict(targeted_tool_reference_grant_ids)
    )
    name_grant_ids = {} if native_tool_name_grant_ids is None else dict(native_tool_name_grant_ids)
    tool_calls: list[runtime_records.ToolCallRequest] = []
    for call in copied.tool_calls:
        projected_arguments = redactor.redact_json_values(call.arguments)
        if type(projected_arguments) is not dict:
            raise AssertionError("Tool-call argument redaction returned a non-object.")
        tool_ref = call.arguments.get("tool_ref") if call.name == CALL_TOOL_NAME else None
        targeted_tool_grant_id = (
            reference_grant_ids.get(tool_ref) if type(tool_ref) is str else None
        )
        native_grant_id = name_grant_ids.get(call.name)
        if targeted_tool_grant_id is not None and native_grant_id is not None:
            raise ValueError(
                "A model tool call matched both reference and native dynamic-tool authority."
            )
        if native_grant_id is not None:
            targeted_tool_grant_id = native_grant_id
        if targeted_tool_grant_id is not None and tool_ref is not None:
            # The exact reference was issued in this request by the runtime.
            # Retain it only as private executable material paired with its
            # grant id; transcript projection below replaces it with a
            # non-authoritative placeholder.
            projected_arguments["tool_ref"] = tool_ref
        tool_calls.append(
            runtime_records.ToolCallRequest(
                id=call.id,
                name=call.name,
                arguments=projected_arguments,
                targeted_tool_grant_id=targeted_tool_grant_id,
            )
        )
    assistant_message = transcript_helpers.assistant_message_with_tool_call_arguments(
        assistant_message,
        tool_calls,
    )
    text_content = assistant_text_content(assistant_message)
    return AssistantStepResult(
        session_id=copied.session_id,
        step=copied.step,
        model_step_id=copied.model_step_id,
        model_attempt_id=copied.model_attempt_id,
        tool_round_identity=copied.tool_round_identity,
        assistant_message=assistant_message,
        tool_calls=tool_calls,
        completion=copied.completion,
        text_content=text_content,
        has_user_visible_content=bool(text_content.strip()),
        provider_state_count=provider_state_count(assistant_message),
        thinking_count=thinking_count(assistant_message),
    )


def _non_turn_model_completion_event(
    event: Event,
    *,
    failure: BaseException,
    cancellation: asyncio.CancelledError | None,
    transcript_cursor: int,
) -> Event:
    payload = copy_durable_json_object(event.payload, "model_completion_payload")
    if cancellation is not None:
        reason = "model stream was cancelled before terminal validation completed"
    elif isinstance(failure, SessionInterruptedByRequest):
        reason = "session interruption won before terminal validation completed"
    elif isinstance(failure, ModelAttemptFailed):
        reason = "provider emitted an invalid event after model completion"
    else:
        reason = "model completion failed terminal validation"
    payload["step_classification"] = {
        "type": "failed",
        "reason": reason,
    }
    payload["transcript_cursor"] = transcript_cursor
    return copy_event(event).model_copy(update={"payload": payload}, deep=True)


def _validate_model_completion_publication_result(
    request: ModelCompletionPublicationRequest,
    result: ModelCompletionPublicationResult,
) -> None:
    if type(result) is not ModelCompletionPublicationResult:
        raise TypeError("Model completion publisher must return ModelCompletionPublicationResult.")
    detached_result = ModelCompletionPublicationResult(
        completion=result.completion,
        publication=result.publication,
    )
    prepared = request.dispatch.stage
    completed = detached_result.completion.stage
    for field_name in (
        "session_id",
        "stage_id",
        "logical_step_id",
        "dispatch_ordinal",
        "purpose",
        "intent",
        "reservation_ids",
        "preparation_request_digest",
        "preparation_digest",
        "source_status",
        "source_run_epoch",
        "source_transcript_cursor",
        "prepared_at",
    ):
        if getattr(completed, field_name) != getattr(prepared, field_name):
            raise RuntimeError(
                "Model completion publisher acknowledged a different prepared "
                f"stage field: {field_name}."
            )
    if completed.state != "completed" or completed.publication is None:
        raise RuntimeError("Model completion publisher did not return a terminal stage.")

    publication_request = completed.publication
    expected_messages = (
        ()
        if request.authoritative_assistant_message is None or request.defer_assistant_message
        else (request.authoritative_assistant_message,)
    )
    if publication_request.publication_id != request.dispatch.logical_step_id:
        raise RuntimeError("Model completion publication acknowledged a different logical step.")
    if publication_request.kind != "model-step":
        raise RuntimeError("Model completion publication has the wrong publication kind.")
    if publication_request.intent != request.dispatch.intent:
        raise RuntimeError("Model completion publication changed the prepared dispatch intent.")
    if publication_request.transcript_messages != expected_messages:
        raise RuntimeError("Model completion publication changed the authoritative assistant turn.")
    if publication_request.events != (request.completion_event,):
        raise RuntimeError("Model completion publication changed its completion event.")
    pending_round_operations = [
        operation
        for operation in publication_request.mutation.operations
        if operation.key == "pending_tool_round"
    ]
    expects_pending_round = bool(
        request.assistant_step_result is not None
        and request.authoritative_assistant_message is not None
        and request.assistant_step_result.tool_calls
    )
    if bool(pending_round_operations) != expects_pending_round:
        raise RuntimeError("Model completion publication changed its pending tool-round mutation.")
    durable_validation = None
    durable_tool_exposure = None
    if pending_round_operations:
        if len(pending_round_operations) != 1:
            raise RuntimeError(
                "Model completion publication changed its pending tool-round mutation."
            )
        pending_round_value = pending_round_operations[0].value
        if type(pending_round_value) is not dict:
            raise RuntimeError(
                "Model completion publication returned a malformed pending tool round."
            )
        durable_validation = pending_round_value.get("structured_output_validation")
        durable_tool_exposure = pending_round_value.get("tool_exposure")
    expected_validation = (
        None
        if request.structured_output_validation is None
        else request.structured_output_validation.model_dump(mode="json")
    )
    if durable_validation != expected_validation:
        raise RuntimeError("Model completion publication changed its structured-output validation.")
    expected_tool_exposure = (
        None
        if not expects_pending_round or request.tool_exposure is None
        else request.tool_exposure.model_dump(mode="json")
    )
    if durable_tool_exposure != expected_tool_exposure:
        raise RuntimeError("Model completion publication changed its frozen tool exposure.")

    promoted = detached_result.publication
    receipt = promoted.receipt
    if promoted.session.id != prepared.session_id:
        raise RuntimeError("Model completion publication returned a different session.")
    if receipt.session_id != prepared.session_id:
        raise RuntimeError("Model completion receipt belongs to a different session.")
    if receipt.publication_id != request.dispatch.logical_step_id:
        raise RuntimeError("Model completion receipt belongs to a different logical step.")
    if receipt.kind != "model-step":
        raise RuntimeError("Model completion receipt has the wrong publication kind.")
    if receipt.appended_event_ids != (request.completion_event.id,):
        raise RuntimeError("Model completion receipt does not bind the exact completion event.")
    if receipt.transcript_start_cursor != prepared.source_transcript_cursor:
        raise RuntimeError("Model completion publication started at a different transcript cursor.")
    if receipt.transcript_end_cursor != (
        prepared.source_transcript_cursor + len(expected_messages)
    ):
        raise RuntimeError("Model completion publication ended at a different transcript cursor.")


async def _publish_model_completion(
    publisher: ModelCompletionPublisher,
    request: ModelCompletionPublicationRequest,
    *,
    terminal_failure: BaseException | None,
    publication_cancellation: asyncio.CancelledError | None,
) -> None:
    expected_request = request
    callback_request = ModelCompletionPublicationRequest(
        dispatch=request.dispatch,
        assistant_step_result=request.assistant_step_result,
        completion_event=request.completion_event,
        authoritative_assistant_message=request.authoritative_assistant_message,
        defer_assistant_message=request.defer_assistant_message,
        structured_output_validation=request.structured_output_validation,
        tool_exposure=request.tool_exposure,
        operation_record_mutations=request.operation_record_mutations,
    )
    if publication_cancellation is not None:
        assert terminal_failure is not None

        async def publish() -> ModelCompletionPublicationResult:
            return await publisher(callback_request)

        publication_task = asyncio.create_task(publish())
        outcome = await await_shielded_task_outcome(
            publication_task,
            cancellation=publication_cancellation,
        )
        if outcome.error is not None:
            callback_error = outcome.error
            if isinstance(callback_error, asyncio.CancelledError):
                callback_error = unexpected_child_cancellation_error(
                    callback_error,
                    operation="model completion publication",
                )
            add_exception_note_safely(
                terminal_failure,
                "Model completion publication also failed while preserving cancellation: "
                f"{type(callback_error).__name__}: {callback_error}",
            )
        elif outcome.result is None:
            add_exception_note_safely(
                terminal_failure,
                (
                    "Model completion publication returned no acknowledgement while preserving "
                    "cancellation."
                ),
            )
        else:
            try:
                _validate_model_completion_publication_result(
                    expected_request,
                    outcome.result,
                )
            except BaseException as validation_error:
                add_exception_note_safely(
                    terminal_failure,
                    (
                        "Model completion publication acknowledgement was invalid while "
                        "preserving cancellation: "
                        f"{type(validation_error).__name__}: {validation_error}"
                    ),
                )
        if terminal_failure is publication_cancellation or (
            isinstance(terminal_failure, BaseExceptionGroup)
            and any(
                candidate is publication_cancellation
                for candidate in iter_exception_tree(terminal_failure)
            )
        ):
            raise terminal_failure
        add_exception_note_safely(
            publication_cancellation,
            "The provider suppressed caller cancellation before raising "
            f"{type(terminal_failure).__name__}.",
        )
        raise publication_cancellation from terminal_failure

    try:
        result = await publisher(callback_request)
        _validate_model_completion_publication_result(expected_request, result)
    except BaseException as publication_error:
        if terminal_failure is not None:
            publication_error.add_note(
                "The provider stream had already reached a terminal failure after emitting "
                "completion evidence."
            )
        raise


def _take_model_completion_cancellation(
    failure: BaseException | None,
    *,
    cancellation_baseline: int,
) -> asyncio.CancelledError | None:
    """Take caller cancellation newer than the provider-boundary baseline."""

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
    return consume_pending_task_cancellation(
        cancellation,
        preserve_requests=cancellation_baseline,
    )


def _combine_post_completion_failures(
    current: BaseException | None,
    subsequent: BaseException,
) -> BaseException:
    if current is None or current is subsequent:
        return subsequent
    return BaseExceptionGroup(
        "Model completion encountered multiple terminal failures.",
        [current, subsequent],
    )


def _combine_authoritative_model_failure(
    authoritative: BaseException,
    secondary: BaseException,
    *,
    message: str,
) -> BaseException:
    """Preserve one authoritative model failure beside later diagnostics."""

    if authoritative is secondary:
        return authoritative
    if any(candidate is authoritative for candidate in iter_exception_tree(secondary)):
        return secondary
    if any(candidate is secondary for candidate in iter_exception_tree(authoritative)):
        return authoritative
    return BaseExceptionGroup(message, [authoritative, secondary])
