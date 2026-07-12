from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from examples._advanced_support.results import ScenarioResult

ScenarioRunner = Callable[[Path], Coroutine[Any, Any, ScenarioResult]]
LiveScenarioRunner = Callable[[Path, str | None], Coroutine[Any, Any, ScenarioResult]]


def run_cli(
    *,
    deterministic: ScenarioRunner,
    live: LiveScenarioRunner,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("deterministic", "live"), default="deterministic")
    parser.add_argument("--provider", choices=("gemini", "openai", "anthropic"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "deterministic" and args.provider is not None:
        raise SystemExit("--provider can only be used with --mode live")
    default_trials = int(os.environ.get("CAYU_ADVANCED_TRIALS", "1"))
    trials = args.trials if args.trials is not None else default_trials
    if trials < 1 or trials > 10:
        raise SystemExit("--trials must be between 1 and 10")
    results: list[ScenarioResult] = []
    failures: list[str] = []
    for trial in range(1, trials + 1):
        try:
            invocation = (
                deterministic(args.root.resolve())
                if args.mode == "deterministic"
                else live(args.root.resolve(), args.provider)
            )
            results.append(asyncio.run(invocation))
        except Exception as exc:
            failures.append(f"trial {trial}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError("; ".join(failures))
    result = results[-1]
    evidence = {
        "scenario": result.scenario,
        "mode": result.mode,
        "status": result.status,
        "assertions": result.assertions,
        "metrics": result.metrics,
        "result_path": str(result.output_path) if result.output_path else None,
        "trials": trials,
        "successful_trials": len(results),
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print("CAYU_NIGHTLY_EVIDENCE=" + json.dumps(evidence, sort_keys=True))
