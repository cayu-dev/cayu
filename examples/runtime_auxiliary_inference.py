"""Run a managed tool-owned model request without network access or credentials.

Run: PYTHONPATH=src python examples/runtime_auxiliary_inference.py
Direct provider SDK calls inside a tool are outside Cayu's accounting guarantee.
Use a bounded child session instead when the delegated work needs its own tools,
transcript, approvals, or recursive inference.
"""

from __future__ import annotations

import asyncio
import json

from cayu import (
    AgentSpec,
    AuxiliaryInferencePolicy,
    CayuApp,
    EventType,
    InferenceLimits,
    Message,
    ModelRequest,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)

MODEL = "demo-model"
LIMITS = InferenceLimits(max_input_tokens=100, max_output_tokens=50, timeout_seconds=10)


class Summarize(Tool):
    spec = ToolSpec(
        name="summarize",
        description="Summarize text through runtime-owned inference.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        auxiliary_inference=AuxiliaryInferencePolicy(limits=LIMITS, purposes=("tool.summary",)),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        if ctx.inference is None:
            raise RuntimeError("This tool requires managed inference.")
        response = await ctx.inference.invoke(
            ModelRequest(model=MODEL, messages=[Message.text("user", args["text"])]),
            purpose="tool.summary",
            limits=LIMITS,
        )
        return ToolResult(content=response.text)


async def run_demo() -> dict[str, object]:
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    name="summarize", id="summary-call", arguments={"text": "Long input"}
                ),
                ModelStreamEvent.completed({"usage": {"input_tokens": 2, "output_tokens": 1}}),
            ],
            [
                ModelStreamEvent.text_delta("Short summary"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 5, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("Finished"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 2, "output_tokens": 1}}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model=MODEL), tools=[Summarize()])
    events = [
        event
        async for event in app.run(
            RunRequest(
                session_id="auxiliary-demo",
                agent_name="assistant",
                messages=[Message.text("user", "Please summarize the input.")],
            )
        )
    ]
    usage = await app.get_session_usage("auxiliary-demo")
    attempts = [
        event for event in events if event.type == EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
    ]
    return {
        "provider_calls": len(provider.requests),
        "ordinary_model_steps": usage.model_steps,
        "total_tokens": usage.usage.total_tokens,
        "auxiliary_attempts": len(attempts),
        "auxiliary_purposes": [
            event.payload["auxiliary_inference"]["purpose"] for event in attempts
        ],
        "auxiliary_outcomes": [event.payload["auxiliary_outcome"] for event in attempts],
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_demo()), indent=2))
