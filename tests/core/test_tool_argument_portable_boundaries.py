from __future__ import annotations

import asyncio
from typing import Any, Literal, cast

import pytest
from tests._session_provenance import fixture_session_invocation

from cayu import (
    AgentSpec,
    BeforeToolCallDecision,
    BeforeToolCallHookContext,
    CayuApp,
    Event,
    EventType,
    RuntimeHookPhase,
    Session,
    Tool,
    ToolCallHookContext,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    extract_durable_value_error,
)
from cayu._validation import MAX_DURABLE_JSON_INTEGER, MIN_DURABLE_JSON_INTEGER
from cayu.testing import verify_tool_effect

ArgumentBoundary = Literal[
    "before_decision",
    "before_context",
    "after_context",
    "verify_tool_effect",
]

_BOUNDARIES: tuple[ArgumentBoundary, ...] = (
    "before_decision",
    "before_context",
    "after_context",
    "verify_tool_effect",
)
_SENSITIVE_CANARY = "private-tool-argument-canary"


class _RecordingTool(Tool):
    spec = ToolSpec(
        name="portable_probe",
        effect=ToolEffect.NONE,
        input_schema={"type": "object", "additionalProperties": True},
    )

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        super().__init__()
        self._calls = calls

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del ctx
        self._calls.append(args)
        return ToolResult(content="ok")


def _session() -> Session:
    return Session(
        id="sess_portable_tool_arguments",
        agent_name="worker",
        provider_name="fake",
        model="fake-model",
        invocation=fixture_session_invocation("sess_portable_tool_arguments"),
    )


async def _capture_arguments(
    boundary: ArgumentBoundary,
    arguments: dict[str, Any],
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    if boundary == "before_decision":
        decision = BeforeToolCallDecision(
            action="proceed_modified",
            modified_arguments=arguments,
        )
        assert decision.modified_arguments is not None
        return decision.modified_arguments

    if boundary == "before_context":
        context = BeforeToolCallHookContext(
            runtime=cast("Any", object()),
            hook_name="portable-hook",
            phase=RuntimeHookPhase.BEFORE_TOOL_CALL,
            session=_session(),
            tool_name="portable_probe",
            tool_call_id="call_1",
            arguments=arguments,
            task_id=None,
        )
        return context.arguments

    if boundary == "after_context":
        context = ToolCallHookContext(
            runtime=cast("Any", object()),
            hook_name="portable-hook",
            phase=RuntimeHookPhase.AFTER_TOOL_CALL,
            session=_session(),
            tool_event=Event(
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="sess_portable_tool_arguments",
            ),
            tool_name="portable_probe",
            tool_call_id="call_1",
            arguments=arguments,
            result=ToolResult(content="ok"),
            task_id=None,
        )
        return context.arguments

    tool = _RecordingTool(tool_calls)
    app = CayuApp(enable_logging=False)
    app.register_agent(AgentSpec(name="worker", model="fake-model"), tools=[tool])
    await verify_tool_effect(
        app,
        agent_name="worker",
        tool_name="portable_probe",
        arguments=arguments,
    )
    assert len(tool_calls) == 1
    return tool_calls[0]


@pytest.mark.parametrize("boundary", _BOUNDARIES)
@pytest.mark.parametrize(
    ("arguments", "expected_code"),
    [
        ({"value": MAX_DURABLE_JSON_INTEGER + 1}, "integer_out_of_range"),
        ({"value": MIN_DURABLE_JSON_INTEGER - 1}, "integer_out_of_range"),
        ({"value": float(2**63)}, "integral_float_out_of_range"),
        ({"value": -(float(2**63) + 2048.0)}, "integral_float_out_of_range"),
        ({"value": f"{_SENSITIVE_CANARY}\x00"}, "nul_character"),
        ({"value": "bad\ud800value"}, "unicode_surrogate"),
        ({"bad\x00key": "value"}, "nul_character"),
        ({"bad\ud800key": "value"}, "unicode_surrogate"),
    ],
)
def test_tool_argument_boundaries_reject_nonportable_values_before_invocation(
    boundary: ArgumentBoundary,
    arguments: dict[str, Any],
    expected_code: str,
) -> None:
    tool_calls: list[dict[str, Any]] = []

    with pytest.raises(ValueError) as raised:
        asyncio.run(_capture_arguments(boundary, arguments, tool_calls))

    durable_error = extract_durable_value_error(raised.value)
    assert durable_error is not None
    assert durable_error.code == expected_code
    assert _SENSITIVE_CANARY not in str(raised.value)
    assert tool_calls == []


@pytest.mark.parametrize("boundary", _BOUNDARIES)
def test_tool_argument_boundaries_normalize_and_copy_portable_values(
    boundary: ArgumentBoundary,
) -> None:
    arguments: dict[str, Any] = {
        "minimum": MIN_DURABLE_JSON_INTEGER,
        "maximum": MAX_DURABLE_JSON_INTEGER,
        "integral": 42.0,
        "negative_zero": -0.0,
        "fractional": 1.25,
        "unicode": "Zażółć 😀",
        "nested": {"array": [1.0, {"value": "original"}]},
    }
    tool_calls: list[dict[str, Any]] = []

    captured = asyncio.run(_capture_arguments(boundary, arguments, tool_calls))

    assert captured == {
        "minimum": MIN_DURABLE_JSON_INTEGER,
        "maximum": MAX_DURABLE_JSON_INTEGER,
        "integral": 42,
        "negative_zero": 0,
        "fractional": 1.25,
        "unicode": "Zażółć 😀",
        "nested": {"array": [1, {"value": "original"}]},
    }
    assert type(captured["integral"]) is int
    assert type(captured["negative_zero"]) is int
    assert type(captured["fractional"]) is float

    cast("dict[str, Any]", arguments["nested"])["array"][1]["value"] = "mutated"
    assert captured["nested"]["array"][1]["value"] == "original"
