"""Verified Amazon Bedrock text, tool-structured-output, usage, and counting contract."""

from __future__ import annotations

import asyncio
import json
import os

from examples._live_checks import require

from cayu import (
    AgentSpec,
    BedrockProvider,
    CayuApp,
    Event,
    EventType,
    Message,
    RunRequest,
    StructuredOutputSpec,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelRequest

EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="


class EchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Return the supplied text unchanged.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        effect=ToolEffect.NONE,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        return ToolResult(content=args["text"], structured={"echoed": args["text"]})


async def main() -> None:
    if os.environ.get("CAYU_BEDROCK_LIVE") != "1":
        raise SystemExit("Set CAYU_BEDROCK_LIVE=1 to run the live Bedrock contract.")
    model = os.environ.get("CAYU_BEDROCK_MODEL")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not model:
        raise SystemExit("Set CAYU_BEDROCK_MODEL to a Claude model ID or inference profile ARN.")
    if not region:
        raise SystemExit("Set AWS_REGION or AWS_DEFAULT_REGION to run the Bedrock contract.")

    provider = BedrockProvider(region_name=region, max_tokens=256)
    try:
        counted = await provider.count_input_tokens(
            ModelRequest(
                model=model,
                messages=[Message.text("user", "Reply with a short structured answer.")],
            )
        )
        input_count = counted.input_tokens if counted is not None else None
        if input_count is not None:
            require(input_count > 0, "token count was zero")

        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(
                name="bedrock-live",
                model=model,
                provider_name="bedrock",
                system_prompt=(
                    "Use the echo tool whenever the user requests it, then answer only after "
                    "observing the tool result."
                ),
            ),
            tools=[EchoTool()],
        )
        tool_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="bedrock-live",
                    session_id="bedrock_provider_tool_live",
                    messages=[
                        Message.text(
                            "user",
                            "Call echo with text 'bedrock-tool', then report the returned text.",
                        )
                    ],
                    max_steps=3,
                )
            )
        ]
        _validate_tool_events(tool_events)
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="bedrock-live",
                    session_id="bedrock_provider_live",
                    messages=[
                        Message.text(
                            "user",
                            "Submit structured output with answer exactly 'bedrock-live'.",
                        )
                    ],
                    structured_output=StructuredOutputSpec(
                        name="bedrock_live_answer",
                        json_schema={
                            "type": "object",
                            "properties": {"answer": {"const": "bedrock-live"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                    ),
                )
            )
        ]
        evidence = _validate_events(events, model=model, input_count=input_count)
        evidence["tool_result"] = "validated"
        print(f"{EVIDENCE_PREFIX}{json.dumps(evidence, sort_keys=True)}")
    finally:
        await provider.aclose()


def _validate_events(
    events: list[Event], *, model: str, input_count: int | None
) -> dict[str, object]:
    require(bool(events), "Bedrock run emitted no events")
    require(
        events[-1].type == EventType.SESSION_COMPLETED,
        f"Bedrock session did not complete; tail={_event_tail(events)}",
    )
    validated = next(
        (event for event in events if event.type == EventType.STRUCTURED_OUTPUT_VALIDATED),
        None,
    )
    if validated is None:
        raise RuntimeError("Bedrock structured output was not validated")
    require(validated.payload.get("output") == {"answer": "bedrock-live"}, "wrong output")
    completed = next(
        (event for event in events if event.type == EventType.MODEL_COMPLETED),
        None,
    )
    if completed is None:
        raise RuntimeError("Bedrock run emitted no model.completed event")
    usage = completed.payload.get("usage_metrics")
    if not isinstance(usage, dict):
        raise RuntimeError("Bedrock model.completed omitted normalized usage")
    require(usage.get("provider_name") == "bedrock", "wrong normalized provider name")
    total_tokens = usage.get("total_tokens")
    if type(total_tokens) is not int:
        raise RuntimeError("Bedrock total token usage missing")
    require(total_tokens > 0, "Bedrock total token usage was zero")
    return {
        "provider": "bedrock",
        "model": model,
        "input_count": input_count,
        "token_counting": "validated" if input_count is not None else "unsupported",
        "total_tokens": total_tokens,
        "structured_output": "validated",
    }


def _validate_tool_events(events: list[Event]) -> None:
    require(bool(events), "Bedrock tool run emitted no events")
    require(
        events[-1].type == EventType.SESSION_COMPLETED,
        f"Bedrock tool session did not complete; tail={_event_tail(events)}",
    )
    completed = next(
        (event for event in events if event.type == EventType.TOOL_CALL_COMPLETED),
        None,
    )
    if completed is None:
        raise RuntimeError("Bedrock tool run emitted no tool.call.completed event")
    require(completed.tool_name == "echo", "Bedrock called the wrong tool")
    require(
        any(event.type == EventType.MODEL_TEXT_DELTA for event in events),
        "Bedrock emitted no final text after the tool result",
    )


def _event_tail(events: list[Event]) -> list[dict[str, object]]:
    return [
        {"type": event.type, "tool_name": event.tool_name, "payload": event.payload}
        for event in events[-20:]
    ]


if __name__ == "__main__":
    asyncio.run(main())
