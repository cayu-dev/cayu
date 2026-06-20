from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.core.events import Event, EventType
from cayu.core.tools import ToolResult
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_results as tool_results
from cayu.runtime.approvals import (
    PendingToolCallApproval,
    copy_pending_tool_call_approval,
)
from cayu.runtime.structured_output import (
    StructuredOutputSpec,
    copy_structured_output_spec,
)
from cayu.runtime.tool_policy import ToolPolicyDecision, ToolPolicyResult

PENDING_TOOL_ROUND_CHECKPOINT_KEY = "pending_tool_round"


class PendingToolRound(BaseModel):
    """Durable checkpoint state for an ordinary tool round in progress."""

    model_config = ConfigDict(extra="forbid")

    round_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_name: str
    environment_name: str | None = None
    task_id: str | None = None
    tool_calls: list[PendingToolCallApproval]
    structured_output: StructuredOutputSpec | None = None

    @field_validator("round_id", "agent_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("environment_name", "task_id")
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("tool_calls")
    @classmethod
    def copy_tool_calls(
        cls,
        value: list[PendingToolCallApproval],
    ) -> list[PendingToolCallApproval]:
        copied = [copy_pending_tool_call_approval(call) for call in value]
        if not copied:
            raise ValueError("Pending tool round must include tool calls.")
        return copied

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)


def pending_tool_round_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> PendingToolRound | None:
    if checkpoint is None:
        return None
    copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
    value = copied_checkpoint.get(PENDING_TOOL_ROUND_CHECKPOINT_KEY)
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("Pending tool round checkpoint must be an object.")
    return PendingToolRound(**value)


def checkpoint_with_pending_tool_round(
    checkpoint: dict[str, Any] | None,
    *,
    agent_name: str,
    environment_name: str | None,
    task_id: str | None,
    tool_calls: list[runtime_records.ToolCallRequest],
    policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] | None,
    structured_output: StructuredOutputSpec | None,
) -> tuple[dict[str, Any], PendingToolRound]:
    copied_checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
    if pending_tool_round_from_checkpoint(copied_checkpoint) is not None:
        raise RuntimeError("Session already has a pending tool round.")

    pending_round = PendingToolRound(
        agent_name=agent_name,
        environment_name=environment_name,
        task_id=task_id,
        tool_calls=pending_tool_call_records(
            tool_calls=tool_calls,
            policy_outcomes=policy_outcomes,
        ),
        structured_output=copy_structured_output_spec(structured_output),
    )
    copied_checkpoint[PENDING_TOOL_ROUND_CHECKPOINT_KEY] = pending_round.model_dump(mode="json")
    return copied_checkpoint, pending_round


def checkpoint_without_pending_tool_round(
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    copied_checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
    copied_checkpoint.pop(PENDING_TOOL_ROUND_CHECKPOINT_KEY, None)
    return copied_checkpoint


def pending_tool_call_records(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] | None,
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
                arguments=copy_json_value(tool_call.arguments, "arguments"),
                policy_decision=policy_result.decision.value if policy_result is not None else None,
                reason=policy_result.reason if policy_result is not None else None,
                metadata=(
                    copy_json_value(policy_result.metadata, "metadata")
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
            arguments=copy_json_value(call.arguments, "arguments"),
        )
        for call in pending_round.tool_calls
    ]


def recorded_tool_outcomes(
    *,
    events: list[Event],
    pending_round: PendingToolRound,
) -> tuple[dict[str, runtime_records.ToolCallOutcome], set[str]]:
    pending_calls = {call.tool_call_id: call for call in pending_round.tool_calls}
    started_ids: set[str] = set()
    outcomes: dict[str, runtime_records.ToolCallOutcome] = {}
    terminal_event_types = {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }

    for event in events:
        if event.payload.get("tool_round_id") != pending_round.round_id:
            continue
        tool_call_id = event.payload.get("tool_call_id")
        if type(tool_call_id) is not str or tool_call_id not in pending_calls:
            continue

        if event.type == EventType.TOOL_CALL_STARTED:
            started_ids.add(tool_call_id)
            continue

        if event.type in terminal_event_types:
            outcomes[tool_call_id] = _tool_call_outcome_from_terminal_event(
                event=event,
                pending_tool_call=pending_calls[tool_call_id],
            )

    return outcomes, started_ids


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
                "tool_round_id": pending_round.round_id,
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
            "tool_round_id": pending_round.round_id,
            "tool_call_id": pending_tool_call.tool_call_id,
            "tool_name": pending_tool_call.tool_name,
            "started": True,
            "outcome_unknown": True,
        },
        is_error=True,
    )


def policy_result_from_pending_tool_call(
    pending_tool_call: PendingToolCallApproval,
) -> ToolPolicyResult | None:
    if pending_tool_call.policy_decision is None:
        return None
    return ToolPolicyResult(
        decision=ToolPolicyDecision(pending_tool_call.policy_decision),
        reason=pending_tool_call.reason,
        metadata=copy_json_value(pending_tool_call.metadata, "metadata"),
    )


def _tool_call_outcome_from_terminal_event(
    *,
    event: Event,
    pending_tool_call: PendingToolCallApproval,
) -> runtime_records.ToolCallOutcome:
    result_payload = event.payload.get("result")
    if type(result_payload) is not dict:
        raise ValueError(
            f"Terminal tool event is missing result payload: {pending_tool_call.tool_call_id}"
        )
    result = tool_results.tool_result_from_payload(result_payload)
    return runtime_records.ToolCallOutcome(
        call=runtime_records.ToolCallRequest(
            id=pending_tool_call.tool_call_id,
            name=pending_tool_call.tool_name,
            arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
        ),
        result=result,
    )
