from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from cayu import AgentSpec, CayuApp, OpenAIProvider, ScriptedModelProvider, WorkflowSpec
from cayu.providers import ModelStreamEvent
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.workflows import StepRunOptions, WorkflowBase, step

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["status", "summary"],
    "additionalProperties": False,
}


class SmokeWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="workflow-helper-example")

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        yield await ctx.emit_custom_event(
            "custom.workflow.example.started",
            payload={"session_id": session_id},
        )
        result = await step(
            ctx,
            agent="summarizer",
            step_id="summarize",
            prompt=(
                "Return structured output with status 'ok' and a one sentence "
                "summary of what Cayu workflow helpers do."
            ),
            schema=SUMMARY_SCHEMA,
            run_options=StepRunOptions(labels={"example": "workflow-helpers"}),
        )
        yield await ctx.completed({"result": result.output})


def _scripted_submit(output: dict[str, Any]) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(
            id="call_workflow_example",
            name=STRUCTURED_OUTPUT_TOOL_NAME,
            arguments={"output": output},
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


async def _build_app(provider_name: str) -> tuple[CayuApp, Any]:
    app = CayuApp(enable_logging=False)
    if provider_name == "scripted":
        provider = ScriptedModelProvider(
            [
                _scripted_submit(
                    {
                        "status": "ok",
                        "summary": "Workflow helpers journal and resume agent steps.",
                    }
                )
            ]
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="summarizer", model="scripted-model"))
        return app, provider
    if provider_name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Set OPENAI_API_KEY to run the live OpenAI workflow example.")
        provider = OpenAIProvider()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="summarizer", model="gpt-5.6-luna"))
        return app, provider
    raise SystemExit("CAYU_WORKFLOW_PROVIDER must be scripted or openai.")


async def main() -> None:
    provider_name = os.environ.get("CAYU_WORKFLOW_PROVIDER", "scripted")
    app, provider = await _build_app(provider_name)
    workflow = SmokeWorkflow(app)
    try:
        yielded = [event async for event in workflow.run(f"workflow-example-{provider_name}")]
        journal = await app.session_store.load_events(f"workflow-example-{provider_name}")
        completed_payload = yielded[-1].payload if yielded else {}
        print(
            json.dumps(
                {
                    "provider": provider_name,
                    "result": completed_payload.get("result"),
                    "yielded_events": [str(event.type) for event in yielded],
                    "journal_events": [str(event.type) for event in journal],
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            await aclose()


if __name__ == "__main__":
    asyncio.run(main())
