from __future__ import annotations

import asyncio
from pathlib import Path

from examples._advanced_support import ScenarioResult, completed_batch
from examples.tainted_incident_response.scenario import run_scenario

from cayu import ScriptedModelProvider
from cayu.providers import ModelStreamEvent


def _tool_batch(call_id: str, name: str, arguments: dict) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(id=call_id, name=name, arguments=arguments),
        ModelStreamEvent.completed(
            {
                "finish_reason": "tool_calls",
                "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
            }
        ),
    ]


async def run(root: Path) -> ScenarioResult:
    provider = ScriptedModelProvider(
        [
            _tool_batch("read-incident", "read_untrusted_evidence", {}),
            completed_batch("The evidence is untrusted and has been isolated."),
            _tool_batch("rotate", "rotate_credentials", {"service": "payments"}),
            completed_batch("The credential rotation was blocked by the runtime policy."),
            _tool_batch("sanitize", "sanitize_incident", {}),
            completed_batch("Only the sanitized artifact may leave quarantine."),
            _tool_batch(
                "notify",
                "send_notification",
                {"artifact_id": "sanitized-incident-facts:incident-quarantine:v1"},
            ),
            completed_batch("The sanitized incident notification was sent."),
        ]
    )
    return await run_scenario(
        root,
        provider=provider,
        model="scripted-model",
        mode="deterministic",
    )


if __name__ == "__main__":
    asyncio.run(run(Path.cwd()))
