from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from cayu import AgentSpec, CayuApp
from cayu.context.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.environments import Environment, EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent, OpenAIProvider
from cayu.runners import LocalRunner
from cayu.workflows import StepRunOptions, WorkflowBase, WorkflowSpec, command, step

# A deterministic check the workflow runs after the agent step. Its exit code,
# not the model's own claim, decides whether the summary is accepted.
CHECK_SUMMARY = (
    "import json, sys; out = json.load(sys.stdin); "
    "sys.exit(0 if out.get('status') == 'ok' and out.get('summary') else 1)"
)

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
        # Journaled command step: runs in the app's default environment, and a
        # resumed run returns the journaled result instead of running it again.
        check = await command(
            ctx,
            step_id="check-summary",
            argv=[sys.executable, "-c", CHECK_SUMMARY],
            stdin=json.dumps(result.output),
            timeout_s=30,
        )
        yield await ctx.completed({"result": result.output, "check_passed": check.passed})


def _scripted_submit(output: dict[str, Any]) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(
            id="call_workflow_example",
            name=STRUCTURED_OUTPUT_TOOL_NAME,
            arguments={"output": output},
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


async def _build_app(provider_name: str, workdir: Path) -> tuple[CayuApp, Any]:
    app = CayuApp(enable_logging=False)
    # Swap this for a Docker or Lambda MicroVM environment to run the same check there.
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), runner=LocalRunner(workdir)),
        default=True,
    )
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
    with tempfile.TemporaryDirectory() as workdir:
        await _run(provider_name, Path(workdir))


async def _run(provider_name: str, workdir: Path) -> None:
    app, provider = await _build_app(provider_name, workdir)
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
                    "check_passed": completed_payload.get("check_passed"),
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
