from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from cayu.cli._output import add_output_options
from cayu.cli._targets import TargetResolutionError, load_target
from cayu.cli.project import project_context, resolve_eval_project
from cayu.evals import (
    EvalPlan,
    EvalStatus,
    EvalSuite,
    compare_eval_runs,
    comparison_to_json,
    eval_run_to_json,
    load_eval_run,
    render_comparison_html,
    render_html_report,
    run_eval_suite,
)
from cayu.runtime.app import CayuApp


def add_eval_parser(subparsers: Any) -> None:
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run and report Cayu runtime-native evals.",
        description=(
            "Run and report Cayu runtime-native evals. Start with `cayu eval run` "
            "for the project-configured hermetic proof."
        ),
    )
    inner = eval_parser.add_subparsers(dest="eval_command", required=True)

    run = inner.add_parser(
        "run",
        help="Run a configured or explicit eval plan.",
        description=(
            "Run a configured or explicit eval plan and emit a stable JSON result. "
            "Use `--output FILE` to save it."
        ),
    )
    run.add_argument(
        "target",
        nargs="?",
        help=(
            "Python target that returns EvalPlan, (CayuApp, EvalSuite), or an object "
            "with app and suite attributes. Defaults to [tool.cayu].eval_target."
        ),
    )
    add_output_options(run, formats=("json",))
    run.add_argument("--html-output", metavar="FILE", help="Also write an HTML report to FILE.")
    run.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Limit each eval case to SECONDS (default: no timeout).",
    )

    report = inner.add_parser(
        "report",
        help="Render a JSON or HTML report from eval results.",
        description=(
            "Render saved eval results as HTML by default or JSON explicitly. "
            "Use `--output FILE` to save the report."
        ),
    )
    report.add_argument("input", metavar="RESULTS_JSON", help="Eval JSON results file.")
    add_output_options(report, formats=("html", "json"), default="html")

    compare = inner.add_parser(
        "compare",
        help="Compare baseline and current eval results.",
        description=(
            "Compare baseline and current eval results. JSON is the default; "
            "a nonzero exit reports regressions."
        ),
    )
    compare.add_argument("baseline", metavar="BASELINE_JSON")
    compare.add_argument("current", metavar="CURRENT_JSON")
    add_output_options(compare, formats=("html", "json"))
    compare.add_argument(
        "--score-tolerance",
        type=float,
        default=0.0,
        metavar="DELTA",
        help="Allowed score drop before a regression is flagged (default: 0.0).",
    )


def run_eval_command(args: argparse.Namespace) -> int:
    try:
        if args.eval_command == "run":
            return asyncio.run(_run(args))
        if args.eval_command == "report":
            return _report(args)
        if args.eval_command == "compare":
            return _compare(args)
    except Exception as exc:
        if getattr(args, "output_format", None) == "json":
            print(
                json.dumps(
                    {
                        "schema_version": "1",
                        "error": {"code": "EVAL_COMMAND_FAILED", "message": str(exc)},
                    },
                    sort_keys=True,
                )
            )
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


async def _run(args: argparse.Namespace) -> int:
    project = resolve_eval_project(args.target)
    label = (
        "Command-line eval target"
        if args.target is not None
        else f"Configured eval target from {project.root / 'pyproject.toml'}"
    )
    with project_context(project.root):
        plan = await _load_eval_plan(project.target, label=label)
        run = await run_eval_suite(
            plan.app,
            plan.suite,
            case_timeout_seconds=args.case_timeout_seconds,
        )
        output = eval_run_to_json(run)
        _write_or_print(output, args.output)
        if args.html_output is not None:
            Path(args.html_output).write_text(render_html_report(run), encoding="utf-8")
        return 0 if run.status == EvalStatus.PASSED else 1


async def _load_eval_plan(target: str, *, label: str) -> EvalPlan:
    try:
        loaded = load_target(target, label=label)
    except TargetResolutionError:
        raise
    except Exception as exc:
        raise RuntimeError(f"{label} could not be loaded ({type(exc).__name__}): {exc}") from exc

    try:
        if callable(loaded):
            loaded = loaded()
        if inspect.isawaitable(loaded):
            loaded = await loaded
    except Exception as exc:
        raise RuntimeError(f"{label} failed ({type(exc).__name__}): {exc}") from exc

    try:
        return _coerce_plan(loaded)
    except Exception as exc:
        raise TypeError(f"{label} returned an invalid eval plan: {exc}") from exc


def _report(args: argparse.Namespace) -> int:
    run = load_eval_run(args.input)
    output = eval_run_to_json(run) if args.output_format == "json" else render_html_report(run)
    _write_or_print(output, args.output)
    return 0


def _compare(args: argparse.Namespace) -> int:
    baseline = load_eval_run(args.baseline)
    current = load_eval_run(args.current)
    comparison = compare_eval_runs(baseline, current, score_tolerance=args.score_tolerance)
    if args.output_format == "json":
        output = comparison_to_json(comparison)
    else:
        output = render_comparison_html(comparison)
    _write_or_print(output, args.output)
    if comparison.regressions or current.status != EvalStatus.PASSED:
        return 1
    return 0


def _coerce_plan(value: Any) -> EvalPlan:
    if isinstance(value, EvalPlan):
        return value
    if isinstance(value, tuple | list) and len(value) == 2:
        app, suite = value
        return _validate_plan(app, suite)
    app = getattr(value, "app", None)
    suite = getattr(value, "suite", None)
    if app is not None or suite is not None:
        return _validate_plan(app, suite)
    if isinstance(value, dict) and {"app", "suite"} <= set(value):
        return _validate_plan(value["app"], value["suite"])
    raise TypeError(
        "Eval target must return EvalPlan, (CayuApp, EvalSuite), or app/suite attributes."
    )


def _validate_plan(app: Any, suite: Any) -> EvalPlan:
    if not isinstance(app, CayuApp):
        raise TypeError("Eval plan app must be a CayuApp.")
    if type(suite) is not EvalSuite:
        suite = EvalSuite.model_validate(suite)
    return EvalPlan(app=app, suite=suite)


def _write_or_print(content: str, path: str | None) -> None:
    if path is None:
        print(content, end="")
        return
    Path(path).write_text(content, encoding="utf-8")
