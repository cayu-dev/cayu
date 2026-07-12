from __future__ import annotations

import asyncio
from pathlib import Path

from examples._advanced_support import ScenarioResult, completed_batch, structured_batch
from examples.counterfactual_approval.scenario import run_scenario

from cayu import ScriptedModelProvider
from cayu.providers import ModelStreamEvent


async def run(root: Path) -> ScenarioResult:
    analyses = [
        {
            "future": "approve",
            "outcome": "The verified release is deployed after state revalidation.",
            "risks": ["a latent regression could require rollback"],
            "alternatives": ["canary the release first"],
            "evidence": ["the reconciliation regression test passes"],
            "recommendation": "approve after checking version 7 is still current",
            "uncertainties": ["production traffic differs from the fixture"],
            "external_state_version": 7,
        },
        {
            "future": "deny",
            "outcome": "The current release remains active and the fix is delayed.",
            "risks": ["payment reconciliation remains degraded"],
            "alternatives": ["request a smaller patch"],
            "evidence": ["no mutation is needed to deny"],
            "recommendation": "deny only if the state changed or evidence expires",
            "uncertainties": ["the delay cost is not quantified"],
            "external_state_version": 7,
        },
        {
            "future": "explain",
            "outcome": "Approval deploys one release; denial preserves current state.",
            "risks": ["rollback may be needed"],
            "alternatives": ["canary deployment", "defer for more evidence"],
            "evidence": ["tests passed", "expected external version is 7"],
            "recommendation": "approve with revalidation and post-action verification",
            "uncertainties": ["live load is not represented"],
            "external_state_version": 7,
        },
    ]
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="deploy-1",
                    name="deploy_service",
                    arguments={
                        "service": "payments",
                        "release": "2026.07.11",
                        "expected_version": 7,
                    },
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 25, "output_tokens": 8, "total_tokens": 33},
                    }
                ),
            ],
            *[
                structured_batch(analysis, call_id=f"analysis-{index}")
                for index, analysis in enumerate(analyses, start=1)
            ],
            completed_batch("Deployment completed after approval and revalidation."),
            [
                ModelStreamEvent.tool_call(
                    id="deploy-stale",
                    name="deploy_service",
                    arguments={
                        "service": "payments",
                        "release": "2026.07.12",
                        "expected_version": 7,
                    },
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 25, "output_tokens": 8, "total_tokens": 33},
                    }
                ),
            ],
            completed_batch("The stale deployment was rejected because version 7 is obsolete."),
            structured_batch(
                {
                    "confirmed": True,
                    "observed_version": 8,
                    "observed_mutations": 1,
                    "evidence": "The deployment receipt names release 2026.07.11.",
                },
                call_id="verification",
            ),
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
