from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from cayu._validation import (
    copy_durable_json_value,
    copy_durable_metadata,
    require_clean_nonblank,
)
from cayu.approvals.tools import (
    PendingToolCallApproval,
    ToolPolicyEvidence,
)
from cayu.budgets._run_limit_accounting import (
    RunLimitAccountingContext,
)
from cayu.budgets.base import BudgetLimit, copy_request_budget_limits
from cayu.budgets.run_limits import RunLimits, copy_run_limits
from cayu.context.structured_output import (
    StructuredOutputSpec,
    StructuredOutputValidation,
    copy_structured_output_spec,
)
from cayu.context.thinking import ThinkingConfig
from cayu.events import Event, EventType, copy_event
from cayu.messages import Message, detach_message
from cayu.runtime import _resume_ledger as resume_ledger
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _transcript as transcript_support
from cayu.runtime._argument_continuity import capture_arguments, redact_continuity
from cayu.runtime.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions import _staged_tool_terminal_reader as staged_terminal_reader
from cayu.sessions._assistant_tool_round_publication import (
    AssistantToolRoundPublication,
    StagedToolCallTerminal,
)
from cayu.sessions._checkpoint_secret_validation import (
    require_secret_free_durable_object,
)
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions.base import Session, SessionStatus, SessionStore
from cayu.sessions.checkpoints import WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY
from cayu.tools import _shared_artifact_result_schema as shared_artifact_result_schema
from cayu.tools import _web_access_result_schema as web_access_result_schema
from cayu.tools.base import ToolEffect, ToolResult
from cayu.tools.exposure import (
    ResolvedToolExposureAuthority,
)
from cayu.tools.policy import ToolPolicyResult
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    from cayu.approvals.user_input import PendingUserInput

_TOOL_ROUND_TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }
)


class UnsafeToolRoundContinuationError(RuntimeError):
    """A quarantined round cannot be replayed without inventing provider state."""


def ready_assistant_publication_message(
    pending_round: pending_rounds.PendingToolRound,
) -> Message:
    """Return positively complete assistant evidence or fence provider continuation."""

    if type(pending_round) is not pending_rounds.PendingToolRound:
        raise TypeError("pending_round must be a PendingToolRound.")
    publication = pending_round.assistant_publication
    if publication is None:
        raise RuntimeError("Quarantined tool round has no durable assistant publication evidence.")
    if publication.state == "blocked":
        raise UnsafeToolRoundContinuationError(
            "Quarantined tool round cannot safely continue with opaque provider state."
        )
    if publication.state != "ready" or publication.message is None:
        raise RuntimeError("Quarantined tool round assistant publication is incomplete.")
    return detach_message(publication.message)


def checkpoint_with_assistant_publication_snapshot(
    checkpoint: dict[str, Any] | None,
    *,
    tool_round_identity: ToolRoundIdentity,
    tool_call_id: str,
    redactor: SecretRedactor,
    unsafe_output: bool,
) -> dict[str, Any]:
    """Durably fold one sealed invocation scope into its assistant projection."""

    copied_checkpoint = (
        {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
    )
    tool_call_id = require_clean_nonblank(tool_call_id, "tool_call_id")
    pending_round = pending_round_reader._pending_tool_round_from_owned_checkpoint(
        checkpoint, copied_checkpoint
    )
    if pending_round is not None:
        if pending_rounds.pending_tool_round_identity(pending_round) != copy_tool_round_identity(
            tool_round_identity
        ):
            raise RuntimeError("Assistant publication update targets a different tool round.")
        if pending_round.assistant_message_state == "published":
            return copied_checkpoint
        expected_ids = {call.tool_call_id for call in pending_round.tool_calls}
        updated_publication = _updated_assistant_publication(
            _publication_with_legacy_projection(
                pending_round.assistant_publication,
                message=pending_round.quarantined_assistant_message,
                redactor=redactor,
            ),
            expected_ids=expected_ids,
            tool_call_id=tool_call_id,
            redactor=redactor,
            unsafe_output=unsafe_output,
            cover_call=True,
        )
        updated_round = pending_round.model_copy(
            update={"assistant_publication": updated_publication},
        )
        updated_round = pending_rounds.PendingToolRound.model_validate(
            updated_round.model_dump(mode="json")
        )
        copied_checkpoint[pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY] = (
            updated_round.model_dump(mode="json")
        )
        return copied_checkpoint

    # User-input pauses move the same round-owned projection into their single
    # pending checkpoint. Import lazily to keep the schema modules acyclic.
    from cayu.approvals.user_input import (
        PENDING_USER_INPUT_CHECKPOINT_KEY,
        PendingUserInput,
    )

    pending_input_payload = copied_checkpoint.get(PENDING_USER_INPUT_CHECKPOINT_KEY)
    if type(pending_input_payload) is not dict:
        raise RuntimeError("Assistant publication update has no pending tool round.")
    pending_input = PendingUserInput.model_validate(pending_input_payload)
    if ToolRoundIdentity(
        tool_round_id=pending_input.tool_round_id,
        model_step_id=pending_input.model_step_id,
        model_attempt_id=pending_input.model_attempt_id,
    ) != copy_tool_round_identity(tool_round_identity):
        raise RuntimeError("Assistant publication update targets a different user-input round.")
    if pending_input.assistant_message_state == "published":
        return copied_checkpoint
    expected_ids = {call.tool_call_id for call in pending_input.tool_calls}
    updated_publication = _updated_assistant_publication(
        _publication_with_legacy_projection(
            pending_input.assistant_publication,
            message=pending_input.quarantined_assistant_message,
            redactor=redactor,
        ),
        expected_ids=expected_ids,
        tool_call_id=tool_call_id,
        redactor=redactor,
        unsafe_output=unsafe_output,
        cover_call=True,
    )
    updated_input = pending_input.model_copy(
        update={"assistant_publication": updated_publication},
    )
    updated_input = PendingUserInput.model_validate(updated_input.model_dump(mode="json"))
    copied_checkpoint[PENDING_USER_INPUT_CHECKPOINT_KEY] = updated_input.model_dump(mode="json")
    return copied_checkpoint


def checkpoint_with_assistant_publication_redactor(
    checkpoint: dict[str, Any] | None,
    *,
    tool_round_identity: ToolRoundIdentity,
    tool_call_id: str,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    """Durably apply a newly resolved secret before returning it to a tool."""

    copied_checkpoint = (
        {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
    )
    tool_call_id = require_clean_nonblank(tool_call_id, "tool_call_id")
    pending_round = pending_round_reader._pending_tool_round_from_owned_checkpoint(
        checkpoint, copied_checkpoint
    )
    if pending_round is not None:
        if pending_rounds.pending_tool_round_identity(pending_round) != copy_tool_round_identity(
            tool_round_identity
        ):
            raise RuntimeError("Assistant publication update targets a different tool round.")
        if pending_round.assistant_message_state == "published":
            return copied_checkpoint
        updated_round = pending_round.model_copy(
            update={
                "assistant_publication": _updated_assistant_publication(
                    _publication_with_legacy_projection(
                        pending_round.assistant_publication,
                        message=pending_round.quarantined_assistant_message,
                        redactor=redactor,
                    ),
                    expected_ids={call.tool_call_id for call in pending_round.tool_calls},
                    tool_call_id=tool_call_id,
                    redactor=redactor,
                    unsafe_output=False,
                    cover_call=False,
                )
            },
        )
        updated_round = pending_rounds.PendingToolRound.model_validate(
            updated_round.model_dump(mode="json")
        )
        copied_checkpoint[pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY] = (
            updated_round.model_dump(mode="json")
        )
        return copied_checkpoint

    from cayu.approvals.user_input import (
        PENDING_USER_INPUT_CHECKPOINT_KEY,
        PendingUserInput,
    )

    pending_input_payload = copied_checkpoint.get(PENDING_USER_INPUT_CHECKPOINT_KEY)
    if type(pending_input_payload) is not dict:
        raise RuntimeError("Assistant publication update has no pending tool round.")
    pending_input = PendingUserInput.model_validate(pending_input_payload)
    input_identity = ToolRoundIdentity(
        tool_round_id=pending_input.tool_round_id,
        model_step_id=pending_input.model_step_id,
        model_attempt_id=pending_input.model_attempt_id,
    )
    if input_identity != copy_tool_round_identity(tool_round_identity):
        raise RuntimeError("Assistant publication update targets a different user-input round.")
    if pending_input.assistant_message_state == "published":
        return copied_checkpoint
    updated_input = pending_input.model_copy(
        update={
            "assistant_publication": _updated_assistant_publication(
                _publication_with_legacy_projection(
                    pending_input.assistant_publication,
                    message=pending_input.quarantined_assistant_message,
                    redactor=redactor,
                ),
                expected_ids={call.tool_call_id for call in pending_input.tool_calls},
                tool_call_id=tool_call_id,
                redactor=redactor,
                unsafe_output=False,
                cover_call=False,
            )
        },
    )
    updated_input = PendingUserInput.model_validate(updated_input.model_dump(mode="json"))
    copied_checkpoint[PENDING_USER_INPUT_CHECKPOINT_KEY] = updated_input.model_dump(mode="json")
    return copied_checkpoint


def assistant_publication_snapshot_transform(
    *,
    tool_round_identity: ToolRoundIdentity,
    tool_call_id: str,
    redactor: SecretRedactor,
    unsafe_output: bool,
) -> Callable[[Session, dict[str, Any] | None], dict[str, Any]]:
    """Build one detached store transform for a sealed invocation snapshot."""

    copied_identity = copy_tool_round_identity(tool_round_identity)
    copied_tool_call_id = require_clean_nonblank(tool_call_id, "tool_call_id")
    if type(redactor) is not SecretRedactor:
        raise TypeError("redactor must be a SecretRedactor.")
    if type(unsafe_output) is not bool:
        raise TypeError("unsafe_output must be a boolean.")

    def transform(
        _current_session: Session,
        current_checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return checkpoint_with_assistant_publication_snapshot(
            current_checkpoint,
            tool_round_identity=copied_identity,
            tool_call_id=copied_tool_call_id,
            redactor=redactor,
            unsafe_output=unsafe_output,
        )

    return transform


def assistant_publication_redactor_transform(
    *,
    tool_round_identity: ToolRoundIdentity,
    tool_call_id: str,
    redactor: SecretRedactor,
) -> Callable[[Session, dict[str, Any] | None], dict[str, Any]]:
    """Build a detached transform for one pre-return secret registration."""

    copied_identity = copy_tool_round_identity(tool_round_identity)
    copied_tool_call_id = require_clean_nonblank(tool_call_id, "tool_call_id")
    if type(redactor) is not SecretRedactor:
        raise TypeError("redactor must be a SecretRedactor.")

    def transform(
        _current_session: Session,
        current_checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return checkpoint_with_assistant_publication_redactor(
            current_checkpoint,
            tool_round_identity=copied_identity,
            tool_call_id=copied_tool_call_id,
            redactor=redactor,
        )

    return transform


def _updated_assistant_publication(
    publication: AssistantToolRoundPublication | None,
    *,
    expected_ids: set[str],
    tool_call_id: str,
    redactor: SecretRedactor,
    unsafe_output: bool,
    cover_call: bool,
) -> AssistantToolRoundPublication:
    if tool_call_id not in expected_ids:
        raise RuntimeError("Assistant publication update targets a call outside its round.")
    if publication is None:
        publication = AssistantToolRoundPublication(
            state="blocked",
            reason="projection_evidence_unavailable",
        )
    covered_ids = list(publication.covered_tool_call_ids)
    if cover_call and tool_call_id in covered_ids:
        return publication
    if cover_call:
        covered_ids.append(tool_call_id)
    publication_message = publication.message
    blocked_reason = publication.reason
    if publication.state != "blocked":
        if unsafe_output:
            publication_message = None
            blocked_reason = "incomplete_invocation_secret_scope"
        else:
            if publication_message is None:
                raise AssertionError("Publishable assistant projection lost its message.")
            publication_message = (
                transcript_support.project_assistant_message_for_tool_round_publication(
                    publication_message,
                    redactor=redactor,
                )
            )
            if publication_message is None:
                blocked_reason = "opaque_provider_state_secret"
    if blocked_reason is not None:
        updated_publication = AssistantToolRoundPublication(
            state="blocked",
            covered_tool_call_ids=covered_ids,
            reason=blocked_reason,
            secret_resolution_scope=publication.secret_resolution_scope,
        )
    else:
        updated_publication = AssistantToolRoundPublication(
            state="ready" if set(covered_ids) == expected_ids else "pending",
            message=publication_message,
            argument_continuity=redact_continuity(publication.argument_continuity, redactor),
            covered_tool_call_ids=covered_ids,
            secret_resolution_scope=publication.secret_resolution_scope,
        )
    return updated_publication


def _publication_with_legacy_projection(
    publication: AssistantToolRoundPublication | None,
    *,
    message: Message | None,
    redactor: SecretRedactor,
) -> AssistantToolRoundPublication:
    if publication is not None:
        return publication
    if message is None:
        return AssistantToolRoundPublication(
            state="blocked",
            reason="projection_evidence_unavailable",
        )
    projected = transcript_support.project_assistant_message_for_tool_round_publication(
        message,
        redactor=redactor,
    )
    if projected is None:
        return AssistantToolRoundPublication(
            state="blocked",
            reason="opaque_provider_state_secret",
        )
    return AssistantToolRoundPublication(state="pending", message=projected)


def checkpoint_with_pending_tool_round(
    checkpoint: dict[str, Any] | None,
    *,
    agent_name: str,
    interaction_id: str | None = None,
    environment_name: str | None,
    task_id: str | None,
    source_run_epoch: int | None = None,
    tool_calls: list[runtime_records.ToolCallRequest],
    policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] | None,
    tool_exposure: ResolvedToolExposureAuthority | None = None,
    policy_state: Literal["unplanned", "planned"] = "unplanned",
    policy_context_version: Literal[1] | None = None,
    request_metadata: dict[str, Any] | None = None,
    deferred_messages: list[Message] | None = None,
    assistant_message_state: Literal["published", "quarantined"] = "published",
    quarantined_assistant_message: Message | None = None,
    secret_resolution_scope: Literal["static", "dynamic", "unknown"] = "unknown",
    continuity_tool_names: frozenset[str] = frozenset(),
    continuity_knowledge_scope: Any = None,
    structured_output: StructuredOutputSpec | None = None,
    thinking: ThinkingConfig | None = None,
    max_steps: int | None = None,
    limits: RunLimits | None = None,
    run_limit_accounting: RunLimitAccountingContext | None = None,
    budget_limits: tuple[BudgetLimit, ...] | None = None,
    retry_policy: RetryPolicy | None = None,
    tool_round_identity: ToolRoundIdentity,
    redactor: SecretRedactor | None = None,
    runtime_session: Session | None = None,
    source_model_step_id: str | None = None,
    source_transcript_cursor: int | None = None,
    model_step: int | None = None,
    structured_output_attempt: int | None = None,
    structured_output_retries: int = 0,
    structured_output_validation: StructuredOutputValidation | None = None,
) -> tuple[dict[str, Any], pending_rounds.PendingToolRound]:
    copied_checkpoint = (
        {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
    )
    resolved_redactor = redactor or SecretRedactor()
    if (
        pending_round_reader.pending_tool_round_from_checkpoint(
            copied_checkpoint,
            redactor=resolved_redactor,
            consume_on_rejection=True,
            runtime_session=runtime_session,
        )
        is not None
    ):
        raise RuntimeError("Session already has a pending tool round.")

    identity = copy_tool_round_identity(tool_round_identity)
    active_profile = active_invocation_execution_profile_from_checkpoint(copied_checkpoint)
    durable_request_metadata = (
        {} if request_metadata is None else resolved_redactor.redact_json_values(request_metadata)
    )
    if type(durable_request_metadata) is not dict:
        raise AssertionError("Pending tool-round request metadata must remain an object.")
    assistant_publication = None
    if assistant_message_state == "quarantined":
        if quarantined_assistant_message is None:
            raise ValueError("Quarantined assistant publication requires its private message.")
        publication_message = (
            transcript_support.project_assistant_message_for_tool_round_publication(
                quarantined_assistant_message,
                redactor=resolved_redactor,
            )
        )
        assistant_publication = AssistantToolRoundPublication(
            state="blocked" if publication_message is None else "pending",
            message=publication_message,
            argument_continuity=(
                capture_arguments(
                    tool_calls,
                    names=continuity_tool_names,
                    scope=continuity_knowledge_scope,
                    profile=None if active_profile is None else active_profile.profile.fingerprint,
                    redactor=resolved_redactor,
                )
                if publication_message is not None
                else None
            ),
            reason=("opaque_provider_state_secret" if publication_message is None else None),
            secret_resolution_scope=secret_resolution_scope,
        )
    pending_round = pending_rounds.PendingToolRound(
        tool_round_id=identity.tool_round_id,
        model_step_id=identity.model_step_id,
        model_attempt_id=identity.model_attempt_id,
        agent_name=agent_name,
        interaction_id=interaction_id,
        environment_name=environment_name,
        task_id=task_id,
        source_run_epoch=source_run_epoch,
        execution_profile_fingerprint=(
            None if active_profile is None else active_profile.profile.fingerprint
        ),
        tool_exposure=tool_exposure,
        tool_calls=pending_tool_call_records(
            tool_calls=tool_calls,
            policy_outcomes=policy_outcomes,
            default_policy_evidence=(
                ToolPolicyEvidence.UNPLANNED
                if policy_state == "unplanned"
                else ToolPolicyEvidence.UNREGISTERED
            ),
            redactor=redactor,
        ),
        policy_state=policy_state,
        policy_context_version=policy_context_version,
        request_metadata=durable_request_metadata,
        deferred_messages=(
            []
            if deferred_messages is None
            else [detach_message(message) for message in deferred_messages]
        ),
        assistant_message_state=assistant_message_state,
        quarantined_assistant_message=quarantined_assistant_message,
        assistant_publication=assistant_publication,
        structured_output=copy_structured_output_spec(structured_output),
        thinking=thinking,
        max_steps=max_steps,
        limits=copy_run_limits(limits) if limits is not None else None,
        run_limit_accounting=run_limit_accounting,
        budget_limits=(
            copy_request_budget_limits(budget_limits) if budget_limits is not None else None
        ),
        retry_policy=copy_retry_policy(retry_policy) if retry_policy is not None else None,
        source_model_step_id=source_model_step_id,
        source_transcript_cursor=source_transcript_cursor,
        model_step=model_step,
        structured_output_attempt=structured_output_attempt,
        structured_output_retries=structured_output_retries,
        structured_output_validation=structured_output_validation,
    )
    pending_round_reader._require_executable_pending_tool_round(pending_round)
    pending_payload = pending_round.model_dump(mode="json")
    serialized_calls = pending_payload.get("tool_calls")
    if not isinstance(serialized_calls, list):
        raise AssertionError("Pending tool round serialized tool_calls as a non-list.")
    for serialized_call in serialized_calls:
        if type(serialized_call) is not dict:
            raise AssertionError("Pending tool round serialized a non-object tool call.")
        reason = serialized_call.get("reason")
        if type(reason) is str:
            serialized_call["reason"] = resolved_redactor.redact_text(reason)
        metadata = serialized_call.get("metadata")
        if type(metadata) is dict:
            serialized_call["metadata"] = resolved_redactor.redact_json(metadata)
    pending_payload = require_secret_free_durable_object(
        pending_payload,
        redactor=resolved_redactor,
        field_name="pending_tool_round",
        schema_root=pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY,
        runtime_session=runtime_session,
    )
    copied_checkpoint[pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY] = pending_payload
    copied_checkpoint = require_secret_free_durable_object(
        copied_checkpoint,
        redactor=resolved_redactor,
        field_name="checkpoint",
        runtime_session=runtime_session,
    )
    return copied_checkpoint, pending_round


def checkpoint_without_pending_tool_round(
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    copied_checkpoint = (
        {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
    )
    # Workspace settlement retains the pending round as its recovery authority.
    # A completed tool event alone does not authorize retiring that owner.
    if copied_checkpoint.get(WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY):
        raise RuntimeError("Cannot retire a tool round with unsettled workspace observations.")
    from cayu.sessions._foreground_child_checkpoint import (
        FOREGROUND_CHILD_TERMINAL_KEY,
        FOREGROUND_CHILD_WAIT_KEY,
        foreground_child_state_from_checkpoint,
    )

    wait, selected = foreground_child_state_from_checkpoint(copied_checkpoint)
    if wait is not None:
        pending = pending_round_reader.pending_tool_round_from_checkpoint(copied_checkpoint)
        if pending is None or pending.tool_round_id != wait.parent_effect.tool_round_id:
            raise RuntimeError("Cannot retire a foreground wait belonging to another round.")
        copied_checkpoint.pop(FOREGROUND_CHILD_WAIT_KEY)
        if selected is not None:
            copied_checkpoint.pop(FOREGROUND_CHILD_TERMINAL_KEY)
    copied_checkpoint.pop(pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY, None)
    return copied_checkpoint


def pending_tool_call_records(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] | None,
    default_policy_evidence: ToolPolicyEvidence = ToolPolicyEvidence.UNPLANNED,
    redactor: SecretRedactor | None = None,
) -> list[PendingToolCallApproval]:
    policy_results_by_id: dict[str, ToolPolicyResult | None] = {}
    policy_evidence_by_id: dict[str, ToolPolicyEvidence] = {}
    if policy_outcomes is not None:
        policy_results_by_id = {outcome.call.id: outcome.result for outcome in policy_outcomes}
        policy_evidence_by_id = {outcome.call.id: outcome.evidence for outcome in policy_outcomes}

    records: list[PendingToolCallApproval] = []
    for tool_call in tool_calls:
        policy_result = policy_results_by_id.get(tool_call.id)
        records.append(
            PendingToolCallApproval(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                arguments=copy_durable_json_value(tool_call.arguments, "arguments"),
                targeted_tool_grant_id=tool_call.targeted_tool_grant_id,
                model_tool_name=tool_call.model_tool_name,
                targeted_tool_invocation=tool_call.targeted_tool_invocation,
                targeted_tool_rejection=tool_call.targeted_tool_rejection,
                policy_evidence=policy_evidence_by_id.get(
                    tool_call.id,
                    default_policy_evidence,
                ),
                policy_decision=policy_result.decision.value if policy_result is not None else None,
                command_denial_code=(
                    policy_result.command_denial_code if policy_result is not None else None
                ),
                reason=resume_ledger.policy_reason_for_pending_tool_call(
                    policy_result,
                    redactor=redactor,
                ),
                metadata=(
                    copy_durable_metadata(policy_result.metadata, "metadata")
                    if policy_result is not None
                    else {}
                ),
            )
        )
    return records


def pending_round_tool_calls(
    pending_round: pending_rounds.PendingToolRound,
) -> list[runtime_records.ToolCallRequest]:
    return [
        runtime_records.ToolCallRequest(
            id=call.tool_call_id,
            name=call.tool_name,
            arguments=copy_durable_json_value(call.arguments, "arguments"),
            targeted_tool_grant_id=call.targeted_tool_grant_id,
            model_tool_name=call.model_tool_name,
            targeted_tool_invocation=call.targeted_tool_invocation,
            targeted_tool_rejection=call.targeted_tool_rejection,
        )
        for call in pending_round.tool_calls
    ]


def recorded_tool_outcomes(
    *,
    events: list[Event],
    pending_round: pending_rounds.PendingToolRound,
) -> tuple[dict[str, runtime_records.ToolCallOutcome], set[str]]:
    identity = pending_rounds.pending_tool_round_identity(pending_round)
    ledger = resume_ledger.scan_tool_call_events(
        events=events,
        pending_calls=pending_round.tool_calls,
        in_scope=lambda event: identity.matches_payload(event.payload),
        candidate_scope=lambda event: (
            event.payload.get("tool_round_id") == pending_round.tool_round_id
            or (
                event.payload.get("model_step_id") == pending_round.model_step_id
                and event.payload.get("model_attempt_id") == pending_round.model_attempt_id
            )
        ),
        terminal_event_types=_TOOL_ROUND_TERMINAL_EVENT_TYPES,
    )
    if ledger.scope_conflicting:
        raise resume_ledger.ToolCallEvidenceConflict(
            "Tool-round recovery evidence contains a call outside the pending tool round."
        )
    outcomes = dict(ledger.outcomes)
    pending_by_id = {call.tool_call_id: call for call in pending_round.tool_calls}
    for staged in staged_terminal_reader._recovery_safe_staged_terminals(pending_round):
        pending_call = pending_by_id[staged.tool_call_id]
        staged_outcome = resume_ledger.tool_call_outcome_from_terminal_event(
            event=staged.event,
            pending_tool_call=pending_call,
        )
        recorded = outcomes.get(staged.tool_call_id)
        if recorded is not None:
            if recorded != staged_outcome:
                raise resume_ledger.ToolCallEvidenceConflict(
                    "Durable and staged terminal evidence conflict for one tool call."
                )
            continue
        outcomes[staged.tool_call_id] = staged_outcome
    return outcomes, ledger.started_ids


def checkpoint_with_staged_terminals(
    checkpoint: dict[str, Any] | None,
    *,
    tool_round_identity: ToolRoundIdentity,
    staged_terminals: list[StagedToolCallTerminal],
) -> dict[str, Any]:
    """Replace stages on the checkpoint boundary that owns the round."""

    copied = {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
    if type(copied) is not dict:
        raise AssertionError("Checkpoint copied as a non-object.")
    owner_key, owner = staged_terminal_reader._staged_terminal_owner_from_owned_checkpoint(
        copied,
        tool_round_identity=tool_round_identity,
    )
    return _replace_owned_staged_terminals(copied, owner_key, owner, staged_terminals)


def _replace_owned_staged_terminals(
    copied: dict[str, Any],
    owner_key: str,
    owner: pending_rounds.PendingToolRound | PendingUserInput,
    staged_terminals: list[StagedToolCallTerminal],
) -> dict[str, Any]:
    """Replace stages on a call-local owned checkpoint and revalidate its owner."""
    copied_stages = [
        StagedToolCallTerminal.model_validate(item.model_dump(mode="json"))
        for item in staged_terminals
    ]
    updated_owner = owner.model_copy(update={"staged_terminals": copied_stages})
    updated_owner = type(owner).model_validate(updated_owner.model_dump(mode="json"))
    copied[owner_key] = updated_owner.model_dump(mode="json")
    return copied


def completed_staged_terminal_transform(
    *,
    tool_round_identity: ToolRoundIdentity,
    event: Event,
    payload_bytes: int | None = None,
) -> Callable[[Session, dict[str, Any] | None], dict[str, Any]]:
    """Record that terminal hooks completed before public event append."""

    return _updated_staged_terminal_transform(
        tool_round_identity=tool_round_identity,
        event=event,
        payload_bytes=payload_bytes,
        hooks_state="completed",
        operation="Hook completion",
    )


def projected_staged_terminal_transform(
    *,
    tool_round_identity: ToolRoundIdentity,
    event: Event,
    payload_bytes: int | None = None,
) -> Callable[[Session, dict[str, Any] | None], dict[str, Any]]:
    """Persist a projected terminal without claiming its hooks completed."""

    return _updated_staged_terminal_transform(
        tool_round_identity=tool_round_identity,
        event=event,
        payload_bytes=payload_bytes,
        hooks_state="preserve",
        operation="Projection",
    )


def started_staged_terminal_publication_transform(
    *,
    tool_round_identity: ToolRoundIdentity,
    tool_call_id: str,
    event: Event,
    payload_bytes: int,
    effect_completed_at: datetime,
    staged_at: datetime,
    publication_started_at: datetime,
) -> Callable[[Session, dict[str, Any] | None], dict[str, Any]]:
    """Persist exact public-event timing before its first append attempt."""

    identity = copy_tool_round_identity(tool_round_identity)
    copied_event = copy_event(event)

    def transform(
        _session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any]:
        copied = {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
        if type(copied) is not dict:
            raise AssertionError("Checkpoint copied as a non-object.")
        # Pinning publication timing changes the event digest. Keep the exact
        # workspace-bound stage until observation recovery has consumed it.
        if copied.get(WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY):
            raise RuntimeError("Cannot publish tool stages before workspace settlement.")
        owner_key, owner = staged_terminal_reader._staged_terminal_owner_from_owned_checkpoint(
            copied, tool_round_identity=identity
        )
        existing = owner.staged_terminals
        updated: list[StagedToolCallTerminal] = []
        found = False
        for item in existing:
            if item.tool_call_id != tool_call_id:
                updated.append(item)
                continue
            if item.event.id != copied_event.id:
                raise RuntimeError("Publication timing conflicts with staged terminal evidence.")
            found = True
            if item.publication_started_at is not None:
                if (
                    item.publication_started_at != publication_started_at
                    or item.event != copied_event
                ):
                    raise RuntimeError("Staged terminal has conflicting publication timing.")
                updated.append(item)
            else:
                updated.append(
                    item.model_copy(
                        update={
                            "event": copied_event,
                            "payload_bytes": payload_bytes,
                            "effect_completed_at": effect_completed_at,
                            "staged_at": staged_at,
                            "publication_started_at": publication_started_at,
                        },
                        deep=True,
                    )
                )
        if not found:
            raise RuntimeError("Publication timing has no staged terminal owner.")
        return _replace_owned_staged_terminals(copied, owner_key, owner, updated)

    return transform


def _updated_staged_terminal_transform(
    *,
    tool_round_identity: ToolRoundIdentity,
    event: Event,
    payload_bytes: int | None,
    hooks_state: Literal["preserve", "completed"],
    operation: str,
) -> Callable[[Session, dict[str, Any] | None], dict[str, Any]]:
    """Replace one owned stage while preserving or completing its hook state."""

    identity = copy_tool_round_identity(tool_round_identity)
    copied_event = copy_event(event)
    if payload_bytes is not None and (type(payload_bytes) is not int or payload_bytes < 0):
        raise ValueError("payload_bytes must be a non-negative integer or None.")

    def transform(
        _session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any]:
        copied = {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
        if type(copied) is not dict:
            raise AssertionError("Checkpoint copied as a non-object.")
        owner_key, owner = staged_terminal_reader._staged_terminal_owner_from_owned_checkpoint(
            copied,
            tool_round_identity=identity,
        )
        existing_stages = owner.staged_terminals
        tool_call_id = copied_event.payload.get("tool_call_id")
        if type(tool_call_id) is not str:
            raise ValueError(f"{operation} staged terminal lost its tool-call identity.")
        found = False
        staged: list[StagedToolCallTerminal] = []
        for item in existing_stages:
            if item.tool_call_id != tool_call_id:
                staged.append(item)
                continue
            if item.event.id != copied_event.id:
                raise RuntimeError(f"{operation} conflicts with staged terminal evidence.")
            found = True
            staged.append(
                item.model_copy(
                    update={
                        "event": copied_event,
                        **({} if payload_bytes is None else {"payload_bytes": payload_bytes}),
                        "hooks_state": (
                            item.hooks_state if hooks_state == "preserve" else "completed"
                        ),
                    },
                    deep=True,
                )
            )
        if not found:
            raise RuntimeError(f"{operation} has no staged terminal owner.")
        return _replace_owned_staged_terminals(copied, owner_key, owner, staged)

    return transform


def validate_tool_round_recovery_target(
    *,
    events: list[Event],
    pending_round: pending_rounds.PendingToolRound,
    tool_call_id: str,
    execution_started: bool | None = None,
) -> None:
    """Reject manual recovery targets that need no recovery or never started.

    Scoped by the round's session-unique ``tool_round_id`` payload key — the
    same ledger key `recorded_tool_outcomes` reads, so a call this guard
    accepts is exactly one the automatic close would otherwise synthesize an
    unknown outcome for.
    """
    identity = pending_rounds.pending_tool_round_identity(pending_round)
    pending_tool_call = next(
        (call for call in pending_round.tool_calls if call.tool_call_id == tool_call_id),
        None,
    )
    if pending_tool_call is None:
        raise ValueError(f"Tool call is not part of the pending tool round: {tool_call_id}")
    state = resume_ledger.tool_call_recovery_state(
        events=events,
        pending_calls=pending_round.tool_calls,
        tool_call_id=pending_tool_call.tool_call_id,
        in_scope=lambda event: identity.matches_payload(event.payload),
        candidate_scope=lambda event: (
            event.payload.get("tool_round_id") == pending_round.tool_round_id
            or (
                event.payload.get("model_step_id") == pending_round.model_step_id
                and event.payload.get("model_attempt_id") == pending_round.model_attempt_id
            )
        ),
        terminal_event_types=_TOOL_ROUND_TERMINAL_EVENT_TYPES,
    )

    # Runtime-authenticated positive zero-dispatch evidence is stronger than
    # an ambiguous event ledger.  Conflict recovery must not let an operator
    # assign an executed outcome to a worker known never to have been admitted.
    if execution_started is False:
        raise RuntimeError(
            f"Tool round recovery requires a recorded tool.call.started event: {tool_call_id}"
        )
    if state.conflicting:
        return
    if state.terminal:
        raise RuntimeError(
            f"Tool call already has a terminal event and does not need recovery: {tool_call_id}. "
            "Resume the session to close the round from the persisted outcome."
        )
    effective_execution_started = state.started if execution_started is None else execution_started
    if not effective_execution_started:
        raise RuntimeError(
            f"Tool round recovery requires a recorded tool.call.started event: {tool_call_id}"
        )


def unknown_recovered_tool_result(
    *,
    pending_tool_call: PendingToolCallApproval,
    pending_round: pending_rounds.PendingToolRound,
    started: bool,
    effect: ToolEffect | None = None,
) -> ToolResult:
    if not started:
        return ToolResult(
            content=(
                f"Tool call {pending_tool_call.tool_name} "
                f"({pending_tool_call.tool_call_id}) was not executed before Cayu "
                "recovered an incomplete tool round."
            ),
            structured={
                "recovered": True,
                "recovery_reason": "pending_tool_round_not_started",
                **pending_rounds.pending_tool_round_identity(pending_round).payload(),
                "tool_call_id": pending_tool_call.tool_call_id,
                "tool_name": pending_tool_call.tool_name,
                "started": False,
                "executed": False,
                "outcome_unknown": False,
            },
            is_error=True,
        )

    if effect in (ToolEffect.NONE, ToolEffect.IDEMPOTENT):
        # Recovery cannot republish the original arguments (their redaction scope
        # is gone), but the declared effect makes another call for the same
        # operation safe; "inspect external state" assumes a tool apps rarely have.
        guidance = (
            f"Its outcome is unknown and its original arguments are not shown after "
            f"recovery. {pending_tool_call.tool_name} declares {effect.value} effects, "
            "so calling it again for the same operation is safe."
        )
    else:
        guidance = (
            "The external side-effect outcome is unknown; inspect external state before retrying."
        )
    return ToolResult(
        content=(
            f"Tool call {pending_tool_call.tool_name} ({pending_tool_call.tool_call_id}) "
            "started but did not record a terminal result before Cayu recovered an "
            f"incomplete tool round. {guidance}"
        ),
        structured={
            "recovered": True,
            "recovery_reason": "pending_tool_round_missing_terminal_event",
            **pending_rounds.pending_tool_round_identity(pending_round).payload(),
            "tool_call_id": pending_tool_call.tool_call_id,
            "tool_name": pending_tool_call.tool_name,
            "started": True,
            "outcome_unknown": True,
        },
        is_error=True,
    )


def hook_scope_unavailable_recovery_event(event: Event) -> Event:
    """Fail closed when restart erased a dynamic round's hook redactor."""

    if type(event) is not Event or event.type not in _TOOL_ROUND_TERMINAL_EVENT_TYPES:
        raise TypeError("Recovered hook quarantine requires a terminal tool event.")
    # Quarantine removes unsafe output, not the durable fact that a denied call
    # never executed. Reclassifying it as failed would invent an executed outcome
    # without a started event and make continuation reconciliation reject it.
    never_executed = event.type in staged_terminal_reader._NONEXECUTED_TERMINAL_EVENT_TYPES
    result = ToolResult(
        content=(
            "Tool result unavailable because its invocation-secret scope could not "
            "be reconstructed before recovery hooks."
        ),
        structured={
            **({} if never_executed else {"error": "invalid_tool_output"}),
            "outcome_unknown": not never_executed,
            **({"executed": False} if never_executed else {}),
            "recovered": True,
            "reason": "recovery_hook_secret_scope_unavailable",
        },
        is_error=True,
    )
    payload = copy_durable_json_value(event.payload, "recovered_terminal.payload")
    payload.pop(web_access_result_schema.WEB_ACCESS_RESULT_AUTHORITY_FIELD, None)
    payload.pop(shared_artifact_result_schema.SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD, None)
    payload["result"] = result.model_dump(mode="json")
    payload["recovered"] = True
    return copy_event(
        event.model_copy(
            update={
                "type": event.type if never_executed else EventType.TOOL_CALL_FAILED,
                "payload": payload,
            }
        )
    )


_SUBAGENT_RECOVERY_TERMINAL_STATUSES = frozenset(
    {SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.INTERRUPTED}
)


def subagent_child_idempotency_key(child: Session) -> str | None:
    """The tool-execution ``idempotency_key`` a child subagent session records, or None if unlinked.

    The key encodes (session, tool_round, tool_call), so matching on it binds a recovered child to the
    exact pending spawn call — round-scoped, immune to providers reusing a ``tool_call_id`` across rounds.
    """
    subagent = child.metadata.get("subagent")
    if not isinstance(subagent, dict):
        return None
    idempotency_key = subagent.get("idempotency_key")
    return idempotency_key if type(idempotency_key) is str and idempotency_key else None


def recovered_subagent_tool_result(
    *,
    tool_call_id: str,
    tool_name: str,
    tool_round_id: str,
    child: Session,
) -> ToolResult:
    """Re-attach a recovered subagent-spawn tool call to its durably-created child session.

    Closes the parent->child linkage window: instead of resolving an incomplete spawn call as an unknown
    (or generic interrupted) outcome, record the discovered child (id + terminal status) so the parent
    transcript keeps a durable reference. Shared by the crash-recovery and live-interrupt close paths.
    """
    status = child.status
    terminal = status in _SUBAGENT_RECOVERY_TERMINAL_STATUSES
    subagent = child.metadata.get("subagent")
    mode = subagent.get("mode") if isinstance(subagent, dict) else None
    durable_dispatch = subagent.get("durable_dispatch") if isinstance(subagent, dict) else None
    queue_task_id = (
        durable_dispatch.get("queue_task_id") if isinstance(durable_dispatch, dict) else None
    )
    if terminal:
        retrieval = (
            "Use subagent_result for its full output."
            if mode in {"background", "durable"}
            else "Its durable session and transcript remain available for inspection."
        )
        content = (
            f"Subagent {child.id} was recovered with terminal status {status.value} after Cayu "
            f"recovered an incomplete tool round. {retrieval}"
        )
    else:
        # A non-terminal child means its in-process execution did not survive the crash. The linkage is
        # still recorded so the parent can inspect or re-run the child rather than losing the reference.
        content = (
            f"Subagent {child.id} was spawned but did not reach a terminal status before Cayu recovered "
            f"an incomplete tool round (status {status.value}); its outcome is unknown."
        )
    return ToolResult(
        content=content,
        structured={
            "recovered": True,
            "recovery_reason": "pending_tool_round_reattached_subagent",
            "tool_round_id": tool_round_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "child_session_id": child.id,
            "parent_session_id": child.parent_session_id,
            "mode": mode,
            **({"queue_task_id": queue_task_id} if isinstance(queue_task_id, str) else {}),
            "status": status.value,
            "outcome_unknown": not terminal,
        },
        is_error=status is not SessionStatus.COMPLETED,
    )


async def load_tool_round_lifecycle_events(
    session_store: SessionStore,
    *,
    session_id: str,
    pending_round: pending_rounds.PendingToolRound,
) -> list[Event]:
    """Load bounded lifecycle evidence and scope reused call IDs by round."""
    candidates = await session_store.load_tool_round_lifecycle_events_for_round(
        session_id,
        [call.tool_call_id for call in pending_round.tool_calls],
        tool_round_identity=pending_rounds.pending_tool_round_identity(pending_round),
    )
    lifecycle_events: list[Event] = []
    for event in candidates:
        event_round_id = event.payload.get("tool_round_id")
        if event_round_id == pending_round.tool_round_id:
            lifecycle_events.append(event)
            continue
        if (
            type(event_round_id) is not str
            or not event_round_id.strip()
            or event_round_id.strip() != event_round_id
        ):
            raise RuntimeError("Indexed tool-round lifecycle evidence has no valid round identity.")
        raise RuntimeError(
            "Round-scoped lifecycle lookup returned evidence for a different tool round."
        )
    return lifecycle_events
