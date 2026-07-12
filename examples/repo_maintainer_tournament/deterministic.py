from __future__ import annotations

import asyncio
from pathlib import Path

from examples._advanced_support import ScenarioResult, completed_batch, structured_batch
from examples.repo_maintainer_tournament.scenario import run_scenario

from cayu import ScriptedModelProvider


async def run(root: Path) -> ScenarioResult:
    candidates = [
        {
            "strategy": "minimal-boundary-check",
            "rationale": "Add the required domain error at the public boundary.",
            "changes": [
                {
                    "path": "calculator.py",
                    "content": (
                        "def divide(a: float, b: float) -> float:\n"
                        "    if b == 0:\n"
                        "        raise ValueError('denominator cannot be zero')\n"
                        "    return a / b\n"
                    ),
                }
            ],
        },
        {
            "strategy": "weaken-tests",
            "rationale": "Change the failing test so the current behavior passes.",
            "changes": [
                {
                    "path": "test_calculator.py",
                    "content": (
                        "from calculator import divide\n\n\n"
                        "def test_divide() -> None:\n"
                        "    assert divide(6, 3) == 2\n"
                    ),
                }
            ],
        },
        {
            "strategy": "broad-rewrite",
            "rationale": "Introduce a new exception hierarchy and validation helper.",
            "changes": [
                {
                    "path": "calculator.py",
                    "content": (
                        "class CalculatorError(ValueError):\n"
                        "    pass\n\n\n"
                        "def _validate_denominator(value: float) -> None:\n"
                        "    if value == 0:\n"
                        "        raise CalculatorError('denominator cannot be zero')\n\n\n"
                        "def divide(a: float, b: float) -> float:\n"
                        "    _validate_denominator(b)\n"
                        "    return float(a) / float(b)\n"
                    ),
                }
            ],
        },
    ]
    provider = ScriptedModelProvider(
        [
            completed_batch("Issue and pull request context prepared."),
            *[
                structured_batch(candidate, call_id=f"candidate-{index}")
                for index, candidate in enumerate(candidates, start=1)
            ],
            structured_batch(
                {
                    "winner": "minimal-boundary-check",
                    "rejected": ["weaken-tests", "broad-rewrite"],
                    "reason": "It passes unchanged tests with the smallest production diff.",
                },
                call_id="repo-evaluation",
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
