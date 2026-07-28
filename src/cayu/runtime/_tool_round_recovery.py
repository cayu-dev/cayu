from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_durable_json_value,
    require_clean_nonblank,
    require_durable_text,
)
from cayu.core.events import Event, EventType
from cayu.core.tools import ToolResult
from cayu.runtime import _resume_ledger as resume_ledger
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._checkpoint_redaction import (
    durable_value_contains_secret,
    require_secret_free_durable_object,
)
from cayu.runtime.approvals import (
    PendingToolCallApproval,
    copy_distinct_pending_tool_call_approvals,
)
from cayu.runtime.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.runtime.sessions import Session, SessionStatus
from cayu.runtime.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
    StructuredOutputValidation,
    copy_structured_output_spec,
)
from cayu.runtime.tool_policy import ToolPolicyResult
from cayu.vaults import SecretRedactor, contains_redacted_secret

PENDING_TOOL_ROUND_CHECKPOINT_KEY = "pending_tool_round"
_TOOL_ROUND_TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }
)


class PendingToolRound(BaseModel):
    """Durable checkpoint state for an ordinary tool round in progress."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    tool_round_id: str
    model_step_id: str
    model_attempt_id: str
    agent_name: str
    environment_name: str | None = None
    task_id: str | None = None
    tool_calls: list[PendingToolCallApproval]
    structured_output: StructuredOutputSpec | None = None
    source_model_step_id: str | None = None
    source_transcript_cursor: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    model_step: StrictInt | None = Field(
        default=None,
        ge=1,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    structured_output_attempt: StrictInt | None = Field(
        default=None,
        ge=1,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    structured_output_validation: StructuredOutputValidation | None = None

    @field_validator("agent_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(
            require_durable_text(value, info.field_name),
            info.field_name,
        )

    @model_validator(mode="after")
    def validate_tool_round_identity(self) -> PendingToolRound:
        ToolRoundIdentity(
            tool_round_id=self.tool_round_id,
            model_step_id=self.model_step_id,
            model_attempt_id=self.model_attempt_id,
        )
        return self

    @field_validator("environment_name", "task_id", "source_model_step_id")
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(
            require_durable_text(value, info.field_name),
            info.field_name,
        )

    @field_validator("tool_calls")
    @classmethod
    def copy_tool_calls(
        cls,
        value: list[PendingToolCallApproval],
    ) -> list[PendingToolCallApproval]:
        return copy_distinct_pending_tool_call_approvals(
            value,
            owner="Pending tool round",
        )

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("structured_output_validation")
    @classmethod
    def copy_structured_output_validation(
        cls,
        value: StructuredOutputValidation | None,
    ) -> StructuredOutputValidation | None:
        if value is None:
            return None
        return value.model_copy(deep=True)

    @model_validator(mode="after")
    def validate_model_step_link(self) -> PendingToolRound:
        source_fields = (
            self.source_model_step_id,
            self.source_transcript_cursor,
            self.model_step,
        )
        if any(value is not None for value in source_fields) and any(
            value is None for value in source_fields
        ):
            raise ValueError(
                "Pending tool-round model-step identity fields must be supplied together."
            )
        if self.structured_output_attempt is not None and (
            self.structured_output is None
            or not any(call.tool_name == STRUCTURED_OUTPUT_TOOL_NAME for call in self.tool_calls)
        ):
            raise ValueError(
                "structured_output_attempt requires a structured-output finalizer call."
            )
        if self.structured_output_validation is not None and (
            self.structured_output_attempt is None
            or self.structured_output is None
            or not any(call.tool_name == STRUCTURED_OUTPUT_TOOL_NAME for call in self.tool_calls)
        ):
            raise ValueError(
                "structured_output_validation requires a structured-output finalizer attempt."
            )
        if self.structured_output_validation is not None:
            validation = self.structured_output_validation
            if validation.valid and validation.errors:
                raise ValueError(
                    "Valid structured-output evidence cannot contain validation errors."
                )
            if not validation.valid and (validation.output is not None or not validation.errors):
                raise ValueError(
                    "Invalid structured-output evidence requires errors and no output."
                )
        if self.structured_output_attempt is not None and self.structured_output_validation is None:
            raise ValueError(
                "A structured-output finalizer attempt requires authoritative validation."
            )
        return self


def pending_tool_round_identity(pending_round: PendingToolRound) -> ToolRoundIdentity:
    if type(pending_round) is not PendingToolRound:
        raise TypeError("Pending tool round must be a PendingToolRound.")
    return ToolRoundIdentity(
        tool_round_id=pending_round.tool_round_id,
        model_step_id=pending_round.model_step_id,
        model_attempt_id=pending_round.model_attempt_id,
    )


def pending_tool_round_from_checkpoint(
    checkpoint: dict[str, Any] | None,
    *,
    redactor: SecretRedactor | None = None,
    consume_on_rejection: bool = False,
) -> PendingToolRound | None:
    if type(consume_on_rejection) is not bool:
        raise TypeError("consume_on_rejection must be a bool.")
    if checkpoint is None:
        return None
    copied_checkpoint = copy_durable_json_value(checkpoint, "checkpoint")
    value = copied_checkpoint.get(PENDING_TOOL_ROUND_CHECKPOINT_KEY)
    if value is None:
        return None
    if redactor is not None and durable_value_contains_secret(
        value,
        redactor=redactor,
        path=(PENDING_TOOL_ROUND_CHECKPOINT_KEY,),
    ):
        # Public callers retain their input by default. Runtime callers opt in
        # to consuming their private checkpoint copy so no outer traceback
        # frame keeps executable secret-bearing state.
        if type(value) is dict:
            value.clear()
        value = None
        copied_checkpoint.clear()
        if consume_on_rejection:
            checkpoint.clear()
        checkpoint = None
        raise ValueError(
            "Pending tool-round checkpoint contains a workload secret and cannot be executed."
        ) from None
    if type(value) is not dict:
        raise ValueError("Pending tool round checkpoint must be an object.")
    validation_rejected = False
    try:
        pending_round = PendingToolRound(**value)
    except Exception:
        if redactor is None:
            raise
        validation_rejected = True
    if validation_rejected:
        value.clear()
        value = None
        copied_checkpoint.clear()
        if consume_on_rejection:
            checkpoint.clear()
        checkpoint = None
        raise ValueError(
            "Pending tool-round checkpoint is invalid and cannot be executed."
        ) from None
    _require_executable_pending_tool_round(pending_round)
    return pending_round


def checkpoint_with_pending_tool_round(
    checkpoint: dict[str, Any] | None,
    *,
    agent_name: str,
    environment_name: str | None,
    task_id: str | None,
    tool_calls: list[runtime_records.ToolCallRequest],
    policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] | None,
    structured_output: StructuredOutputSpec | None,
    tool_round_identity: ToolRoundIdentity,
    redactor: SecretRedactor | None = None,
    source_model_step_id: str | None = None,
    source_transcript_cursor: int | None = None,
    model_step: int | None = None,
    structured_output_attempt: int | None = None,
    structured_output_validation: StructuredOutputValidation | None = None,
) -> tuple[dict[str, Any], PendingToolRound]:
    copied_checkpoint = (
        {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
    )
    resolved_redactor = redactor or SecretRedactor()
    if (
        pending_tool_round_from_checkpoint(
            copied_checkpoint,
            redactor=resolved_redactor,
            consume_on_rejection=True,
        )
        is not None
    ):
        raise RuntimeError("Session already has a pending tool round.")

    identity = copy_tool_round_identity(tool_round_identity)
    pending_round = PendingToolRound(
        **identity.payload(),
        agent_name=agent_name,
        environment_name=environment_name,
        task_id=task_id,
        tool_calls=pending_tool_call_records(
            tool_calls=tool_calls,
            policy_outcomes=policy_outcomes,
            redactor=redactor,
        ),
        structured_output=copy_structured_output_spec(structured_output),
        source_model_step_id=source_model_step_id,
        source_transcript_cursor=source_transcript_cursor,
        model_step=model_step,
        structured_output_attempt=structured_output_attempt,
        structured_output_validation=structured_output_validation,
    )
    _require_executable_pending_tool_round(pending_round)
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
        schema_root=PENDING_TOOL_ROUND_CHECKPOINT_KEY,
    )
    copied_checkpoint[PENDING_TOOL_ROUND_CHECKPOINT_KEY] = pending_payload
    copied_checkpoint = require_secret_free_durable_object(
        copied_checkpoint,
        redactor=resolved_redactor,
        field_name="checkpoint",
    )
    return copied_checkpoint, pending_round


def _require_executable_pending_tool_round(pending_round: PendingToolRound) -> None:
    has_redacted_arguments = any(
        contains_redacted_secret(call.arguments) for call in pending_round.tool_calls
    )
    is_internal_structured_output_round = pending_round.structured_output is not None and any(
        call.tool_name == STRUCTURED_OUTPUT_TOOL_NAME for call in pending_round.tool_calls
    )
    if has_redacted_arguments and not is_internal_structured_output_round:
        raise ValueError(
            "Pending tool-round arguments contain a redaction marker and cannot be executed."
        )


def checkpoint_without_pending_tool_round(
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    copied_checkpoint = (
        {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
    )
    copied_checkpoint.pop(PENDING_TOOL_ROUND_CHECKPOINT_KEY, None)
    return copied_checkpoint


def pending_tool_call_records(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] | None,
    redactor: SecretRedactor | None = None,
) -> list[PendingToolCallApproval]:
    policy_results_by_id: dict[str, ToolPolicyResult | None] = {}
    if policy_outcomes is not None:
        policy_results_by_id = {outcome.call.id: outcome.result for outcome in policy_outcomes}

    records: list[PendingToolCallApproval] = []
    for tool_call in tool_calls:
        policy_result = policy_results_by_id.get(tool_call.id)
        records.append(
            PendingToolCallApproval(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                arguments=copy_durable_json_value(tool_call.arguments, "arguments"),
                policy_decision=policy_result.decision.value if policy_result is not None else None,
                reason=resume_ledger.policy_reason_for_pending_tool_call(
                    policy_result,
                    redactor=redactor,
                ),
                metadata=(
                    copy_durable_json_value(policy_result.metadata, "metadata")
                    if policy_result is not None
                    else {}
                ),
            )
        )
    return records


def pending_round_tool_calls(
    pending_round: PendingToolRound,
) -> list[runtime_records.ToolCallRequest]:
    return [
        runtime_records.ToolCallRequest(
            id=call.tool_call_id,
            name=call.tool_name,
            arguments=copy_durable_json_value(call.arguments, "arguments"),
        )
        for call in pending_round.tool_calls
    ]


def recorded_tool_outcomes(
    *,
    events: list[Event],
    pending_round: PendingToolRound,
) -> tuple[dict[str, runtime_records.ToolCallOutcome], set[str]]:
    identity = pending_tool_round_identity(pending_round)
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
    return ledger.outcomes, ledger.started_ids


def validate_tool_round_recovery_target(
    *,
    events: list[Event],
    pending_round: PendingToolRound,
    tool_call_id: str,
) -> None:
    """Reject manual recovery targets that need no recovery or never started.

    Scoped by the round's session-unique ``tool_round_id`` payload key — the
    same ledger key `recorded_tool_outcomes` reads, so a call this guard
    accepts is exactly one the automatic close would otherwise synthesize an
    unknown outcome for.
    """
    identity = pending_tool_round_identity(pending_round)
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

    if state.conflicting:
        return
    if state.terminal:
        raise RuntimeError(
            f"Tool call already has a terminal event and does not need recovery: {tool_call_id}. "
            "Resume the session to close the round from the persisted outcome."
        )
    if not state.started:
        raise RuntimeError(
            f"Tool round recovery requires a recorded tool.call.started event: {tool_call_id}"
        )


def unknown_recovered_tool_result(
    *,
    pending_tool_call: PendingToolCallApproval,
    pending_round: PendingToolRound,
    started: bool,
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
                **pending_tool_round_identity(pending_round).payload(),
                "tool_call_id": pending_tool_call.tool_call_id,
                "tool_name": pending_tool_call.tool_name,
                "started": False,
                "executed": False,
                "outcome_unknown": False,
            },
            is_error=True,
        )

    return ToolResult(
        content=(
            f"Tool call {pending_tool_call.tool_name} ({pending_tool_call.tool_call_id}) "
            "started but did not record a terminal result before Cayu recovered an "
            "incomplete tool round. The external "
            "side-effect outcome is unknown; inspect external state before retrying."
        ),
        structured={
            "recovered": True,
            "recovery_reason": "pending_tool_round_missing_terminal_event",
            **pending_tool_round_identity(pending_round).payload(),
            "tool_call_id": pending_tool_call.tool_call_id,
            "tool_name": pending_tool_call.tool_name,
            "started": True,
            "outcome_unknown": True,
        },
        is_error=True,
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
    transcript keeps a durable reference. The parent can fetch the child's full output later via
    ``subagent_result``. Shared by the crash-recovery and live-interrupt close paths.
    """
    status = child.status
    terminal = status in _SUBAGENT_RECOVERY_TERMINAL_STATUSES
    if terminal:
        content = (
            f"Subagent {child.id} was recovered with terminal status {status.value} after Cayu "
            "recovered an incomplete tool round. Use subagent_result for its full output."
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
            "status": status.value,
            "outcome_unknown": not terminal,
        },
        is_error=status is not SessionStatus.COMPLETED,
    )
