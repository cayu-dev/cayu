from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from cayu._validation import copy_json_value
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult
from cayu.runtime import _tool_results as tool_results
from cayu.runtime.tool_policy import ToolPolicyResult


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
    ctx: ToolContext,
    arguments: dict[str, Any],
    timeout_seconds: float | None = None,
) -> ToolResult:
    timer: asyncio.Timeout | None = None
    try:
        if timeout_seconds is None:
            result = await tool.run(ctx, arguments)
        else:
            async with asyncio.timeout(timeout_seconds) as timer:
                result = await tool.run(ctx, arguments)
        if type(result) is not ToolResult:
            ctx._discard_policy_denials_for(tool)
            return ToolResult(
                content=(
                    "Tool returned invalid result type: "
                    f"{type(result).__name__}. Expected ToolResult."
                ),
                is_error=True,
            )
        return tool_results.normalize_tool_result(tool_results.validate_tool_result(result))
    except TimeoutError as exc:
        ctx._discard_policy_denials_for(tool)
        if timer is not None and timer.expired():
            return ToolResult(
                content=f"Tool call timed out after {timeout_seconds} seconds.",
                is_error=True,
            )
        return ToolResult(content=tool_results.exception_message(exc), is_error=True)
    except Exception as exc:
        ctx._discard_policy_denials_for(tool)
        return ToolResult(content=tool_results.exception_message(exc), is_error=True)


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
