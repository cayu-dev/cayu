from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    EvalCase,
    EvalPlan,
    EvalStatus,
    EvalSuite,
    FinalOutputContains,
    Message,
    RunRequest,
    load_eval_run,
)
from cayu.cli import main
from cayu.cli.evals import add_eval_parser
from cayu.providers import ModelProvider, ModelStreamEvent


class _SlowProvider(ModelProvider):
    name = "slow"

    async def stream(self, request):
        await asyncio.sleep(0.2)
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def build_slow_eval_plan() -> EvalPlan:
    app = CayuApp(enable_logging=False)
    app.register_provider(_SlowProvider(), default=True)
    app.register_agent(AgentSpec(name="agent", model="fake-model"))
    suite = EvalSuite(
        id="slow-suite",
        cases=[
            EvalCase(
                id="slow-case",
                request=RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "go")],
                    max_steps=1,
                ),
                assertions=[FinalOutputContains("done")],
            )
        ],
    )
    return EvalPlan(app=app, suite=suite)


def test_eval_run_parses_optional_case_timeout_as_float() -> None:
    parser = argparse.ArgumentParser(prog="cayu")
    subparsers = parser.add_subparsers(dest="command")
    add_eval_parser(subparsers)

    configured = parser.parse_args(
        ["eval", "run", "example:build", "--case-timeout-seconds", "0.05"]
    )
    omitted = parser.parse_args(["eval", "run", "example:build"])

    assert configured.case_timeout_seconds == 0.05
    assert omitted.case_timeout_seconds is None


def test_eval_run_timeout_returns_nonzero_and_saves_actionable_error(tmp_path: Path) -> None:
    output = tmp_path / "eval-run.json"

    exit_code = main(
        [
            "eval",
            "run",
            f"{__name__}:build_slow_eval_plan",
            "--case-timeout-seconds",
            "0.01",
            "--output",
            str(output),
        ]
    )

    report = load_eval_run(output)
    assert exit_code == 1
    assert report.status == EvalStatus.ERROR
    assert report.cases[0].status == EvalStatus.ERROR
    assert report.cases[0].error == "Eval case timed out after 0.01 seconds."
