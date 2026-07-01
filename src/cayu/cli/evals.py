from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

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
    )
    inner = eval_parser.add_subparsers(dest="eval_command", required=True)

    run = inner.add_parser(
        "run",
        help="Run an eval plan from a Python target such as package.module:build.",
    )
    run.add_argument(
        "target",
        help=(
            "Python target that returns EvalPlan, (CayuApp, EvalSuite), or an object "
            "with app and suite attributes."
        ),
    )
    run.add_argument("--output", "-o", metavar="FILE", help="Write JSON results to FILE.")
    run.add_argument("--html-output", metavar="FILE", help="Also write an HTML report to FILE.")

    report = inner.add_parser("report", help="Render a JSON or HTML report from eval results.")
    report.add_argument("input", metavar="RESULTS_JSON", help="Eval JSON results file.")
    report.add_argument(
        "--format",
        choices=("html", "json"),
        default="html",
        help="Report format (default: html).",
    )
    report.add_argument("--output", "-o", metavar="FILE", help="Write report to FILE.")

    compare = inner.add_parser("compare", help="Compare baseline and current eval results.")
    compare.add_argument("baseline", metavar="BASELINE_JSON")
    compare.add_argument("current", metavar="CURRENT_JSON")
    compare.add_argument(
        "--format",
        choices=("html", "json"),
        default="json",
        help="Comparison format (default: json).",
    )
    compare.add_argument("--output", "-o", metavar="FILE", help="Write comparison to FILE.")


def run_eval_command(args: argparse.Namespace) -> int:
    try:
        if args.eval_command == "run":
            return asyncio.run(_run(args))
        if args.eval_command == "report":
            return _report(args)
        if args.eval_command == "compare":
            return _compare(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


async def _run(args: argparse.Namespace) -> int:
    loaded = _load_target(args.target)
    if callable(loaded):
        loaded = loaded()
    if inspect.isawaitable(loaded):
        loaded = await loaded
    plan = _coerce_plan(loaded)
    run = await run_eval_suite(plan.app, plan.suite)
    output = eval_run_to_json(run)
    _write_or_print(output, args.output)
    if args.html_output is not None:
        Path(args.html_output).write_text(render_html_report(run), encoding="utf-8")
    return 0 if run.status == EvalStatus.PASSED else 1


def _report(args: argparse.Namespace) -> int:
    run = load_eval_run(args.input)
    output = eval_run_to_json(run) if args.format == "json" else render_html_report(run)
    _write_or_print(output, args.output)
    return 0


def _compare(args: argparse.Namespace) -> int:
    baseline = load_eval_run(args.baseline)
    current = load_eval_run(args.current)
    comparison = compare_eval_runs(baseline, current)
    if args.format == "json":
        output = comparison_to_json(comparison)
    else:
        output = render_comparison_html(comparison)
    _write_or_print(output, args.output)
    if comparison.regressions or current.status != EvalStatus.PASSED:
        return 1
    return 0


def _load_target(target: str) -> Any:
    if ":" not in target:
        raise ValueError("Eval target must use module:attribute syntax.")
    module_name, attr_path = target.split(":", 1)
    if not module_name or not attr_path:
        raise ValueError("Eval target must use module:attribute syntax.")
    module = importlib.import_module(module_name)
    value: Any = module
    for part in attr_path.split("."):
        if not part:
            raise ValueError("Eval target attribute path contains an empty segment.")
        value = getattr(value, part)
    return value


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
