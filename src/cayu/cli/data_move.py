"""`cayu data-move run MODULE:NAME`: run a data move (`cayu.data_moves`) and report it.

The target is a `DataMove`, or a function without arguments that returns one, in the project
in the working directory. The move connects wherever its own code does, usually
`CAYU_DATABASE_URL`, so the command runs the same on a laptop, in CI and as a deploy step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from shlex import quote
from typing import Any

from cayu.cli._targets import TargetResolutionError, load_target
from cayu.data_moves import DataMove, DataMoveReport, StepOutcome, run_data_move

#: Exit status of a run that read every page with no conflict, failure or verify problem.
EXIT_CLEAN = 0
#: Exit status of a run with conflicts, failures or verify problems, or that stopped early.
EXIT_NOT_CLEAN = 3


def add_data_move_parser(subparsers: Any) -> None:
    group = subparsers.add_parser(
        "data-move",
        help="Run a data move: copy or reshape records, item by item, safe to run again.",
        description=(
            "Run a data move defined with cayu.data_moves: a paged source and one idempotent "
            "step per item, with dry run, conflict reporting and a final check."
        ),
    )
    commands = group.add_subparsers(dest="data_move_command", required=True)
    run = commands.add_parser(
        "run",
        help="Run a data move and report what each item's step did.",
        description=(
            "Run the data move TARGET (module:name of a DataMove, or of a function that "
            "returns one) page by page, then its verify step. Exits 0 when every page was "
            f"read with no conflict, failure or verify problem, {EXIT_NOT_CLEAN} otherwise; "
            "rerun with --after RESUME_AFTER to go on from where a run stopped."
        ),
    )
    run.add_argument("target", help="module:name of the DataMove, or of a function returning one.")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Decide every item's outcome without writing anything.",
    )
    run.add_argument(
        "--page-size", type=_positive, default=100, help="Items per source page (100)."
    )
    run.add_argument(
        "--after", help="Start after this source cursor, from a report's resume_after."
    )
    run.add_argument(
        "--max-items",
        type=_positive,
        help="Stop after about this many items, at a page boundary.",
    )
    run.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop at the first item whose step fails, instead of reporting it and going on.",
    )
    run.add_argument("--json", action="store_true", help="Print the report as JSON.")


def run_data_move_command(args: argparse.Namespace) -> int:
    project = str(Path.cwd())
    if project not in sys.path:
        sys.path.insert(0, project)
    try:
        target = load_target(args.target, label="Data move", normalize_errors=True)
    except TargetResolutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    move = target if isinstance(target, DataMove) else target() if callable(target) else None
    if not isinstance(move, DataMove):
        print(
            f"error: {args.target} is not a DataMove or a function that returns one.",
            file=sys.stderr,
        )
        return 2
    report = run_data_move(
        move,
        dry_run=args.dry_run,
        page_size=args.page_size,
        after=args.after,
        max_items=args.max_items,
        stop_on_failure=args.stop_on_failure,
    )
    if args.json:
        print(json.dumps(report.to_json(), indent=2))
    else:
        _print_report(report)
    return EXIT_CLEAN if report.clean else EXIT_NOT_CLEAN


def _print_report(report: DataMoveReport) -> None:
    mode = " (dry run: nothing written)" if report.dry_run else ""
    print(f"Data move {report.name}{mode}: {report.pages} page(s).")
    for outcome in StepOutcome:
        if report.counts[outcome]:
            print(f"  {outcome.value}: {report.counts[outcome]}")
    for item in report.items:
        reason = f": {item.detail}" if item.detail else ""
        print(f"  {item.outcome.value} {item.key}{reason}")
    for problem in report.verify_problems:
        print(f"  verify: {problem}")
    if not report.complete:
        resume = "" if report.resume_after is None else f" --after={quote(report.resume_after)}"
        print(f"Stopped before the last page; go on with `cayu data-move run ...{resume}`.")


def _positive(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1:
        raise argparse.ArgumentTypeError("must be a whole number of at least 1")
    return number
