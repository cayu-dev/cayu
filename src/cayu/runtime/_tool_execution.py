from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cayu._validation import (
    FrozenJsonDict,
    copy_durable_json_object,
    copy_json_value,
    freeze_json_value,
    thaw_json_value,
)
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult
from cayu.runtime import _tool_results as tool_results
from cayu.runtime.tool_policy import ToolPolicyResult
from cayu.vaults import SecretRedactor

_OUTCOME_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class ToolExecutionOutcome:
    """Runtime-owned result plus terminal controls a tool cannot supply.

    Tools are accepted only when they return an exact ``ToolResult``. This
    envelope is created after execution by this module and carries control
    evidence separately, so a forged result or hook cannot clear replay and
    reconciliation semantics.
    """

    result: ToolResult
    _terminal_payload: FrozenJsonDict

    def __init__(
        self,
        result: ToolResult,
        terminal_payload: dict[str, Any],
        *,
        _token: object,
    ) -> None:
        if _token is not _OUTCOME_CONSTRUCTION_TOKEN:
            raise TypeError("ToolExecutionOutcome is runtime-owned.")
        validated_result = tool_results.normalize_tool_result(
            tool_results.validate_tool_result(result)
        )
        copied_payload = copy_durable_json_object(
            terminal_payload,
            "tool_execution_terminal_payload",
        )
        frozen_payload = freeze_json_value(copied_payload)
        if type(frozen_payload) is not FrozenJsonDict:
            raise AssertionError("Frozen terminal tool payload must be an object.")
        object.__setattr__(self, "result", validated_result)
        object.__setattr__(self, "_terminal_payload", frozen_payload)

    @property
    def allows_hook_modification(self) -> bool:
        return not self.publish_before_hooks

    @property
    def publish_before_hooks(self) -> bool:
        """Whether authoritative terminal evidence must precede observational hooks."""

        return bool(self._terminal_payload) and (
            self._terminal_payload.get("tool_effect") != ToolEffect.NONE.value
        )

    def terminal_payload_fields(self) -> dict[str, Any]:
        fields = thaw_json_value(self._terminal_payload)
        if type(fields) is not dict:
            raise AssertionError("Terminal tool payload must thaw to an object.")
        return copy_durable_json_object(fields, "tool_execution_terminal_payload")


def _execution_outcome(
    result: ToolResult,
    terminal_payload: dict[str, Any] | None = None,
) -> ToolExecutionOutcome:
    return ToolExecutionOutcome(
        result,
        terminal_payload or {},
        _token=_OUTCOME_CONSTRUCTION_TOKEN,
    )


def tool_idempotency_key(
    *,
    session_id: str,
    tool_call_id: str,
    tool_round_id: str | None = None,
    approval_id: str | None = None,
    pause_id: str | None = None,
) -> str:
    """Stable, bounded key for one runtime-owned tool execution identity."""

    components = (
        "cayu-tool-idempotency-v1",
        session_id,
        tool_round_id or "",
        approval_id or "",
        pause_id or "",
        tool_call_id,
    )
    material = json.dumps(components, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "cayu-tool:v1:" + hashlib.sha256(material).hexdigest()


async def run_tool(
    *,
    tool: Tool,
    effect: ToolEffect,
    ctx: ToolContext,
    arguments: dict[str, Any],
    redactor: Callable[[], SecretRedactor],
    timeout_seconds: float | None = None,
) -> ToolExecutionOutcome:
    timer: asyncio.Timeout | None = None
    if type(effect) is not ToolEffect:
        raise TypeError("effect must be a ToolEffect.")
    if not callable(redactor):
        raise TypeError("redactor must be a callable returning SecretRedactor.")
    try:
        if timeout_seconds is None:
            raw_result = await tool.run(ctx, arguments)
        else:
            async with asyncio.timeout(timeout_seconds) as timer:
                raw_result = await tool.run(ctx, arguments)
    except TimeoutError as exc:
        ctx._discard_policy_denials_for(tool)
        if timer is not None and timer.expired():
            active_redactor = _active_redactor(redactor)
            result, controls = tool_results.terminal_failure_result(
                terminal_outcome="tool_execution_timeout",
                effect=effect,
                message=f"Tool call timed out after {timeout_seconds} seconds.",
                redactor=active_redactor,
            )
            return _execution_outcome(result, controls)
        active_redactor = _active_redactor(redactor)
        diagnostic = tool_results.exception_diagnostic(
            exc,
            empty_message="tool execution failed",
            nonportable_message="Tool execution failed with a non-portable diagnostic.",
            redactor=active_redactor,
        )
        result, controls = tool_results.terminal_failure_result(
            terminal_outcome="tool_execution_error",
            effect=effect,
            message=diagnostic.message,
            diagnostic=diagnostic,
            redactor=active_redactor,
        )
        return _execution_outcome(result, controls)
    except Exception as exc:
        ctx._discard_policy_denials_for(tool)
        active_redactor = _active_redactor(redactor)
        diagnostic = tool_results.exception_diagnostic(
            exc,
            empty_message="tool execution failed",
            nonportable_message="Tool execution failed with a non-portable diagnostic.",
            redactor=active_redactor,
        )
        result, controls = tool_results.terminal_failure_result(
            terminal_outcome="tool_execution_error",
            effect=effect,
            message=diagnostic.message,
            diagnostic=diagnostic,
            redactor=active_redactor,
        )
        return _execution_outcome(result, controls)

    if type(raw_result) is not ToolResult:
        ctx._discard_policy_denials_for(tool)
        active_redactor = _active_redactor(redactor)
        result, controls = tool_results.terminal_failure_result(
            terminal_outcome="invalid_tool_output",
            effect=effect,
            message=(
                "Tool returned invalid result type: "
                f"{_safe_type_name(raw_result)}. Expected ToolResult."
            ),
            raw_evidence=tool_results.raw_tool_result_evidence(raw_result),
            redactor=active_redactor,
        )
        return _execution_outcome(result, controls)
    try:
        validated_result = tool_results.normalize_tool_result(
            tool_results.validate_tool_result(raw_result)
        )
    except Exception as exc:
        ctx._discard_policy_denials_for(tool)
        active_redactor = _active_redactor(redactor)
        diagnostic = tool_results.exception_diagnostic(
            exc,
            empty_message="tool result validation failed",
            nonportable_message="Tool returned a non-portable result after execution.",
            redactor=active_redactor,
        )
        result, controls = tool_results.terminal_failure_result(
            terminal_outcome="invalid_tool_output",
            effect=effect,
            message=diagnostic.message,
            raw_evidence=tool_results.raw_tool_result_evidence(raw_result),
            diagnostic=diagnostic,
            redactor=active_redactor,
        )
        return _execution_outcome(result, controls)
    return _execution_outcome(validated_result)


def _active_redactor(redactor: Callable[[], SecretRedactor]) -> SecretRedactor:
    active = redactor()
    if not isinstance(active, SecretRedactor):
        raise TypeError("redactor must return a SecretRedactor.")
    return active


def _safe_type_name(value: Any) -> str:
    try:
        name = type.__getattribute__(type(value), "__name__")
    except Exception:
        return "unknown"
    return name if type(name) is str and name.isidentifier() else "unknown"


def policy_denial_reason(policy_result: ToolPolicyResult) -> str:
    return policy_result.reason or "Tool call denied by policy."


def blocked_tool_result(policy_result: ToolPolicyResult, *, reason: str) -> ToolResult:
    return ToolResult(
        content=reason,
        structured={
            "decision": policy_result.decision.value,
            "reason": reason,
            "metadata": policy_result.metadata,
        },
        is_error=True,
    )


def context_metadata(
    *,
    request_metadata: dict[str, Any] | None = None,
    tool_call_id: str,
    approval_id: str | None,
    idempotency_key: str | None = None,
    tool_effect: ToolEffect | None = None,
    input_id: str | None = None,
) -> dict[str, Any]:
    metadata = copy_json_value(request_metadata or {}, "request_metadata")
    metadata["tool_call_id"] = tool_call_id
    if idempotency_key is not None:
        metadata["idempotency_key"] = idempotency_key
    if tool_effect is not None:
        metadata["tool_effect"] = tool_effect.value
    if approval_id is not None:
        metadata["approval_id"] = approval_id
    if input_id is not None:
        metadata["input_id"] = input_id
    return metadata


def validate_tool_policy_result(result: ToolPolicyResult) -> ToolPolicyResult:
    if type(result) is not ToolPolicyResult:
        raise TypeError(
            "Tool policies must return ToolPolicyResult instances. "
            f"Received {type(result).__name__}."
        )
    return ToolPolicyResult(
        decision=result.decision,
        reason=result.reason,
        metadata=copy_json_value(result.metadata, "metadata"),
        approval_expires_in_seconds=result.approval_expires_in_seconds,
    )
