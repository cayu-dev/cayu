from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_durable_json_value,
    require_clean_nonblank,
    require_durable_text,
)
from cayu.core.events import Event, EventType, copy_event
from cayu.core.messages import Message, detach_message
from cayu.core.thinking import ThinkingConfig
from cayu.core.tools import ToolResult
from cayu.runtime import _resume_ledger as resume_ledger
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _transcript as transcript_support
from cayu.runtime._assistant_tool_round_publication import (
    AssistantToolRoundPublication,
    StagedToolCallTerminal,
)
from cayu.runtime._checkpoint_redaction import (
    durable_value_contains_secret,
    require_secret_free_durable_object,
)
from cayu.runtime.approvals import (
    PendingToolCallApproval,
    ToolPolicyEvidence,
    copy_distinct_pending_tool_call_approvals,
)
from cayu.runtime.budgets import BudgetLimit, copy_request_budget_limits
from cayu.runtime.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.sessions import Session, SessionStatus
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.runtime.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
    StructuredOutputValidation,
    copy_structured_output_spec,
)
from cayu.runtime.tool_policy import ToolPolicyResult
from cayu.vaults import SecretRedactor, contains_redacted_secret

if TYPE_CHECKING:
    from cayu.runtime.user_input import PendingUserInput

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
    policy_state: Literal["unplanned", "planned"] = "unplanned"
    policy_context_version: Literal[1] | None = None
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    deferred_messages: list[Message] = Field(default_factory=list)
    assistant_message_state: Literal["published", "quarantined"] = "published"
    quarantined_assistant_message: Message | None = None
    assistant_publication: AssistantToolRoundPublication | None = None
    staged_terminals: list[StagedToolCallTerminal] = Field(default_factory=list)
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
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

    @field_validator("request_metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_value(value, "request_metadata")

    @field_validator("deferred_messages")
    @classmethod
    def copy_deferred_messages(cls, value: list[Message]) -> list[Message]:
        return [detach_message(message) for message in value]

    @field_validator("quarantined_assistant_message")
    @classmethod
    def copy_quarantined_assistant_message(
        cls,
        value: Message | None,
    ) -> Message | None:
        return None if value is None else detach_message(value)

    @field_validator("staged_terminals")
    @classmethod
    def copy_staged_terminals(
        cls,
        value: list[StagedToolCallTerminal],
    ) -> list[StagedToolCallTerminal]:
        copied = [
            StagedToolCallTerminal.model_validate(item.model_dump(mode="json")) for item in value
        ]
        ids = [item.tool_call_id for item in copied]
        if len(ids) != len(set(ids)):
            raise ValueError("Pending tool round cannot repeat staged terminal calls.")
        event_ids = [item.event.id for item in copied]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Pending tool round cannot repeat staged terminal event ids.")
        return copied

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("limits")
    @classmethod
    def copy_limits(cls, value: RunLimits | None) -> RunLimits | None:
        if value is None:
            return None
        return copy_run_limits(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_request_budget_limits(value)

    @field_validator("retry_policy")
    @classmethod
    def copy_retry_policy(cls, value: RetryPolicy | None) -> RetryPolicy | None:
        if value is None:
            return None
        return copy_retry_policy(value)

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
        if self.policy_state == "planned" and self.policy_context_version != 1:
            raise ValueError("A planned pending tool round requires policy context version 1.")
        if self.policy_state == "unplanned":
            if any(
                call.policy_evidence not in {None, ToolPolicyEvidence.UNPLANNED}
                or (
                    call.policy_evidence is ToolPolicyEvidence.UNPLANNED
                    and call.policy_decision is not None
                )
                for call in self.tool_calls
            ):
                raise ValueError("An unplanned pending tool round cannot contain policy authority.")
        elif any(call.policy_evidence is ToolPolicyEvidence.UNPLANNED for call in self.tool_calls):
            raise ValueError("A planned pending tool round cannot contain unplanned calls.")
        if self.assistant_message_state == "quarantined":
            if self.quarantined_assistant_message is None:
                raise ValueError(
                    "A quarantined pending tool round requires its private assistant message."
                )
        elif self.quarantined_assistant_message is not None:
            raise ValueError(
                "A published pending tool round cannot retain a quarantined assistant message."
            )
        if self.assistant_message_state == "published" and self.assistant_publication is not None:
            raise ValueError(
                "A published pending tool round cannot retain assistant publication state."
            )
        if self.assistant_publication is not None:
            expected_ids = {call.tool_call_id for call in self.tool_calls}
            covered_ids = set(self.assistant_publication.covered_tool_call_ids)
            if not covered_ids <= expected_ids:
                raise ValueError("Assistant publication covers a call outside its pending round.")
            expected_state = "ready" if covered_ids == expected_ids else "pending"
            if (
                self.assistant_publication.state != "blocked"
                and self.assistant_publication.state != expected_state
            ):
                raise ValueError("Assistant publication readiness conflicts with call coverage.")
        expected_ids = {call.tool_call_id for call in self.tool_calls}
        if any(item.tool_call_id not in expected_ids for item in self.staged_terminals):
            raise ValueError("Staged terminal evidence names a call outside its pending round.")
        identity = pending_tool_round_identity(self)
        calls_by_id = {call.tool_call_id: call for call in self.tool_calls}
        for item in self.staged_terminals:
            event = item.event
            if not identity.matches_payload(event.payload):
                raise ValueError("Staged terminal evidence has a conflicting round identity.")
            call = calls_by_id[item.tool_call_id]
            if event.tool_name != call.tool_name:
                raise ValueError("Staged terminal evidence has a conflicting tool name.")
        return self


def pending_tool_round_identity(pending_round: PendingToolRound) -> ToolRoundIdentity:
    if type(pending_round) is not PendingToolRound:
        raise TypeError("Pending tool round must be a PendingToolRound.")
    return ToolRoundIdentity(
        tool_round_id=pending_round.tool_round_id,
        model_step_id=pending_round.model_step_id,
        model_attempt_id=pending_round.model_attempt_id,
    )


def ready_assistant_publication_message(pending_round: PendingToolRound) -> Message:
    """Return positively complete assistant evidence or fence provider continuation."""

    if type(pending_round) is not PendingToolRound:
        raise TypeError("pending_round must be a PendingToolRound.")
    publication = pending_round.assistant_publication
    if publication is None:
        raise RuntimeError("Quarantined tool round has no durable assistant publication evidence.")
    if publication.state == "blocked":
        raise RuntimeError(
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
    pending_round = pending_tool_round_from_checkpoint(copied_checkpoint)
    if pending_round is not None:
        if pending_tool_round_identity(pending_round) != copy_tool_round_identity(
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
        updated_round = PendingToolRound.model_validate(updated_round.model_dump(mode="json"))
        copied_checkpoint[PENDING_TOOL_ROUND_CHECKPOINT_KEY] = updated_round.model_dump(mode="json")
        return copied_checkpoint

    # User-input pauses move the same round-owned projection into their single
    # pending checkpoint. Import lazily to keep the schema modules acyclic.
    from cayu.runtime.user_input import (
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
    pending_round = pending_tool_round_from_checkpoint(copied_checkpoint)
    if pending_round is not None:
        if pending_tool_round_identity(pending_round) != copy_tool_round_identity(
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
        updated_round = PendingToolRound.model_validate(updated_round.model_dump(mode="json"))
        copied_checkpoint[PENDING_TOOL_ROUND_CHECKPOINT_KEY] = updated_round.model_dump(mode="json")
        return copied_checkpoint

    from cayu.runtime.user_input import (
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
    policy_state: Literal["unplanned", "planned"] = "unplanned",
    policy_context_version: Literal[1] | None = None,
    request_metadata: dict[str, Any] | None = None,
    deferred_messages: list[Message] | None = None,
    assistant_message_state: Literal["published", "quarantined"] = "published",
    quarantined_assistant_message: Message | None = None,
    secret_resolution_scope: Literal["static", "dynamic", "unknown"] = "unknown",
    structured_output: StructuredOutputSpec | None = None,
    thinking: ThinkingConfig | None = None,
    max_steps: int | None = None,
    limits: RunLimits | None = None,
    budget_limits: tuple[BudgetLimit, ...] | None = None,
    retry_policy: RetryPolicy | None = None,
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
            reason=("opaque_provider_state_secret" if publication_message is None else None),
            secret_resolution_scope=secret_resolution_scope,
        )
    pending_round = PendingToolRound(
        tool_round_id=identity.tool_round_id,
        model_step_id=identity.model_step_id,
        model_attempt_id=identity.model_attempt_id,
        agent_name=agent_name,
        environment_name=environment_name,
        task_id=task_id,
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
        budget_limits=(
            copy_request_budget_limits(budget_limits) if budget_limits is not None else None
        ),
        retry_policy=copy_retry_policy(retry_policy) if retry_policy is not None else None,
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
                policy_evidence=policy_evidence_by_id.get(
                    tool_call.id,
                    default_policy_evidence,
                ),
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
    outcomes = dict(ledger.outcomes)
    pending_by_id = {call.tool_call_id: call for call in pending_round.tool_calls}
    for staged in _recovery_safe_staged_terminals(pending_round):
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


def staged_terminal_events(pending_round: PendingToolRound) -> list[Event]:
    """Return detached private terminal projections in pending-call order."""

    staged_by_id = {
        item.tool_call_id: item.event for item in staged_terminal_records(pending_round)
    }
    return [
        copy_event(staged_by_id[call.tool_call_id])
        for call in pending_round.tool_calls
        if call.tool_call_id in staged_by_id
    ]


def staged_terminal_records(pending_round: PendingToolRound) -> list[StagedToolCallTerminal]:
    """Return recovery-safe staged records in pending-call order."""

    staged_by_id = {
        item.tool_call_id: item for item in _recovery_safe_staged_terminals(pending_round)
    }
    return [
        StagedToolCallTerminal.model_validate(
            staged_by_id[call.tool_call_id].model_dump(mode="json")
        )
        for call in pending_round.tool_calls
        if call.tool_call_id in staged_by_id
    ]


def checkpoint_staged_terminals(
    checkpoint: dict[str, Any] | None,
    *,
    tool_round_identity: ToolRoundIdentity,
) -> list[StagedToolCallTerminal]:
    """Read stages from an ordinary or user-input-owned tool round."""

    _owner_key, owner = _checkpoint_staged_terminal_owner(
        checkpoint,
        tool_round_identity=tool_round_identity,
    )
    return [
        StagedToolCallTerminal.model_validate(item.model_dump(mode="json"))
        for item in owner.staged_terminals
    ]


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
    owner_key, owner = _checkpoint_staged_terminal_owner(
        copied,
        tool_round_identity=tool_round_identity,
    )
    copied_stages = [
        StagedToolCallTerminal.model_validate(item.model_dump(mode="json"))
        for item in staged_terminals
    ]
    updated_owner = owner.model_copy(update={"staged_terminals": copied_stages})
    updated_owner = type(owner).model_validate(updated_owner.model_dump(mode="json"))
    copied[owner_key] = updated_owner.model_dump(mode="json")
    return copied


def _checkpoint_staged_terminal_owner(
    checkpoint: dict[str, Any] | None,
    *,
    tool_round_identity: ToolRoundIdentity,
) -> tuple[str, PendingToolRound | PendingUserInput]:
    identity = copy_tool_round_identity(tool_round_identity)
    pending_round = pending_tool_round_from_checkpoint(checkpoint)
    from cayu.runtime.user_input import (
        PENDING_USER_INPUT_CHECKPOINT_KEY,
        pending_user_input_from_checkpoint,
    )

    pending_input = pending_user_input_from_checkpoint(checkpoint)
    if pending_round is not None and pending_input is not None:
        raise RuntimeError("Checkpoint has multiple staged-terminal owners.")
    if pending_round is not None:
        if pending_tool_round_identity(pending_round) != identity:
            raise RuntimeError("Staged terminals target a different pending tool round.")
        return PENDING_TOOL_ROUND_CHECKPOINT_KEY, pending_round
    if pending_input is not None:
        input_identity = ToolRoundIdentity(
            tool_round_id=pending_input.tool_round_id,
            model_step_id=pending_input.model_step_id,
            model_attempt_id=pending_input.model_attempt_id,
        )
        if input_identity != identity:
            raise RuntimeError("Staged terminals target a different pending user-input round.")
        return PENDING_USER_INPUT_CHECKPOINT_KEY, pending_input
    raise RuntimeError("Staged terminals have no pending tool-round owner.")


def completed_staged_terminal_transform(
    *,
    tool_round_identity: ToolRoundIdentity,
    event: Event,
) -> Callable[[Session, dict[str, Any] | None], dict[str, Any]]:
    """Record that terminal hooks completed before public event append."""

    return _updated_staged_terminal_transform(
        tool_round_identity=tool_round_identity,
        event=event,
        hooks_state="completed",
        operation="Hook completion",
    )


def projected_staged_terminal_transform(
    *,
    tool_round_identity: ToolRoundIdentity,
    event: Event,
) -> Callable[[Session, dict[str, Any] | None], dict[str, Any]]:
    """Persist a projected terminal without claiming its hooks completed."""

    return _updated_staged_terminal_transform(
        tool_round_identity=tool_round_identity,
        event=event,
        hooks_state="preserve",
        operation="Projection",
    )


def _updated_staged_terminal_transform(
    *,
    tool_round_identity: ToolRoundIdentity,
    event: Event,
    hooks_state: Literal["preserve", "completed"],
    operation: str,
) -> Callable[[Session, dict[str, Any] | None], dict[str, Any]]:
    """Replace one owned stage while preserving or completing its hook state."""

    identity = copy_tool_round_identity(tool_round_identity)
    copied_event = copy_event(event)

    def transform(
        _session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any]:
        copied = {} if checkpoint is None else copy_durable_json_value(checkpoint, "checkpoint")
        if type(copied) is not dict:
            raise AssertionError("Checkpoint copied as a non-object.")
        existing_stages = checkpoint_staged_terminals(
            copied,
            tool_round_identity=identity,
        )
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
                StagedToolCallTerminal(
                    tool_call_id=tool_call_id,
                    event=copied_event,
                    hooks_state=(item.hooks_state if hooks_state == "preserve" else "completed"),
                )
            )
        if not found:
            raise RuntimeError(f"{operation} has no staged terminal owner.")
        return checkpoint_with_staged_terminals(
            copied,
            tool_round_identity=identity,
            staged_terminals=staged,
        )

    return transform


def _recovery_safe_staged_terminals(
    pending_round: PendingToolRound,
) -> list[StagedToolCallTerminal]:
    publication = pending_round.assistant_publication
    expected_ids = {call.tool_call_id for call in pending_round.tool_calls}
    covered_ids = set() if publication is None else set(publication.covered_tool_call_ids)
    scope = "unknown" if publication is None else publication.secret_resolution_scope
    if scope == "static" or covered_ids == expected_ids:
        return [
            StagedToolCallTerminal.model_validate(item.model_dump(mode="json"))
            for item in pending_round.staged_terminals
        ]
    safe: list[StagedToolCallTerminal] = []
    for item in pending_round.staged_terminals:
        terminal_controls = tool_results.runtime_terminal_controls(item.event.payload)
        fixed_result = ToolResult(
            content="Tool result unavailable because invocation secret scope was incomplete.",
            structured={
                "error": "invalid_tool_output",
                "outcome_unknown": True,
                **terminal_controls,
            },
            is_error=True,
        )
        payload = copy_durable_json_value(item.event.payload, "staged_terminal.payload")
        if type(payload) is not dict:
            raise AssertionError("Staged terminal payload copied as a non-object.")
        payload["result"] = fixed_result.model_dump(mode="json")
        payload["recovered"] = True
        payload["secret_scope_incomplete"] = True
        event = item.event.model_copy(
            update={"type": EventType.TOOL_CALL_FAILED, "payload": payload},
            deep=True,
        )
        safe.append(
            StagedToolCallTerminal(
                tool_call_id=item.tool_call_id,
                event=event,
                hooks_state="finalized",
            )
        )
    return safe


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


def hook_scope_unavailable_recovery_event(event: Event) -> Event:
    """Fail closed when restart erased a dynamic round's hook redactor."""

    if type(event) is not Event or event.type not in _TOOL_ROUND_TERMINAL_EVENT_TYPES:
        raise TypeError("Recovered hook quarantine requires a terminal tool event.")
    result = ToolResult(
        content=(
            "Tool result unavailable because its invocation-secret scope could not "
            "be reconstructed before recovery hooks."
        ),
        structured={
            "error": "invalid_tool_output",
            "outcome_unknown": True,
            "recovered": True,
            "reason": "recovery_hook_secret_scope_unavailable",
        },
        is_error=True,
    )
    payload = copy_durable_json_value(event.payload, "recovered_terminal.payload")
    payload["result"] = result.model_dump(mode="json")
    payload["recovered"] = True
    return copy_event(
        event.model_copy(
            update={
                "type": EventType.TOOL_CALL_FAILED,
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
