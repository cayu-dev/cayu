"""`cayu data-move run MODULE:NAME`: run a data move (`cayu.data_moves`) and report it.

The target is a `DataMove`, or a function without arguments that returns one, in the project
in the working directory. The move connects wherever its own code does, usually
`CAYU_DATABASE_URL`, so the command runs the same on a laptop, in CI and as a deploy step.
Each run is recorded as it goes (`cayu.data_move_runs`); `resume` goes on from a record and
`runs` lists them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from shlex import quote
from typing import Any

from cayu.cli._targets import TargetResolutionError, load_target
from cayu.data_move_runs import (
    DataMoveRun,
    DataMoveRunStore,
    FileDataMoveRunStore,
    RunOptions,
    default_run_store,
    resume_recorded,
    run_recorded,
)
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
    run.add_argument(
        "--no-record",
        action="store_true",
        help="Keep no run record, so the run can't be resumed with `resume`.",
    )
    _add_store(run)
    run.add_argument("--json", action="store_true", help="Print the report as JSON.")
    resume = commands.add_parser(
        "resume",
        help="Go on with a recorded run from where it stopped.",
        description=(
            "Go on with run --run-id, or the newest run of TARGET that didn't finish, from its "
            "record, with the options it started with. A run whose process stopped mid-page "
            "moves that page again; its steps are idempotent."
        ),
    )
    resume.add_argument(
        "target", help="module:name of the DataMove, or of a function returning one."
    )
    resume.add_argument("--run-id", help="The run to go on with (default: the newest open one).")
    _add_store(resume)
    resume.add_argument("--json", action="store_true", help="Print the run record as JSON.")
    runs = commands.add_parser(
        "runs",
        help="List recorded runs, newest first.",
        description="List the recorded data-move runs, newest first, or show one with --run-id.",
    )
    runs.add_argument("--move", help="Only the runs of the move with this name.")
    runs.add_argument("--run-id", help="Show this run's whole record.")
    runs.add_argument("--limit", type=_positive, default=20, help="Runs to list (20).")
    _add_store(runs)
    runs.add_argument("--json", action="store_true", help="Print JSON.")


def _add_store(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help=(
            "Where run records live (default: CAYU_DATA_VOLUME/data-moves, else "
            "data/data-moves in the working directory)."
        ),
    )


def run_data_move_command(args: argparse.Namespace) -> int:
    store: DataMoveRunStore = (
        default_run_store() if args.runs_dir is None else FileDataMoveRunStore(args.runs_dir)
    )
    if args.data_move_command == "runs":
        return _list_runs(args, store)
    move = _load_move(args.target)
    if move is None:
        return 2
    if args.data_move_command == "resume":
        try:
            run = resume_recorded(move, store, run_id=args.run_id)
        except (LookupError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return _finish(args, run.report, run)
    options = RunOptions(
        page_size=args.page_size, max_items=args.max_items, stop_on_failure=args.stop_on_failure
    )
    if args.no_record:
        report = run_data_move(
            move,
            dry_run=args.dry_run,
            page_size=options.page_size,
            after=args.after,
            max_items=options.max_items,
            stop_on_failure=options.stop_on_failure,
        )
        return _finish(args, report, None)
    run = run_recorded(move, store, dry_run=args.dry_run, after=args.after, options=options)
    return _finish(args, run.report, run)


def _load_move(target_name: str) -> DataMove[Any] | None:
    project = str(Path.cwd())
    if project not in sys.path:
        sys.path.insert(0, project)
    try:
        target = load_target(target_name, label="Data move", normalize_errors=True)
    except TargetResolutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None
    move = target if isinstance(target, DataMove) else target() if callable(target) else None
    if not isinstance(move, DataMove):
        print(
            f"error: {target_name} is not a DataMove or a function that returns one.",
            file=sys.stderr,
        )
        return None
    return move


def _finish(args: argparse.Namespace, report: DataMoveReport, run: DataMoveRun | None) -> int:
    if args.json:
        print(json.dumps(report.to_json() if run is None else run.to_json(), indent=2))
    else:
        _print_report(report, run)
    return EXIT_CLEAN if report.clean else EXIT_NOT_CLEAN


def _list_runs(args: argparse.Namespace, store: DataMoveRunStore) -> int:
    if args.run_id is not None:
        try:
            run = store.get(args.run_id)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if run is None:
            print(f"error: no run {args.run_id} is recorded", file=sys.stderr)
            return 3
        if args.json:
            print(json.dumps(run.to_json(), indent=2))
        else:
            _print_report(run.report, run)
        return 0
    runs = store.list_runs(args.move, limit=args.limit)
    if args.json:
        print(json.dumps({"runs": [run.to_json() for run in runs]}, indent=2))
        return 0
    if not runs:
        print("No data-move runs are recorded.")
    for run in runs:
        counts = ", ".join(
            f"{outcome.value} {count}" for outcome, count in run.report.counts.items() if count
        )
        dry = " dry-run" if run.report.dry_run else ""
        print(
            f"{run.run_id}  {run.move}  {run.status}{dry}  "
            f"{run.started_at.isoformat(timespec='seconds')}  {counts or 'no items'}"
        )
    return 0


def _print_report(report: DataMoveReport, run: DataMoveRun | None = None) -> None:
    mode = " (dry run: nothing written)" if report.dry_run else ""
    recorded = "" if run is None else f", run {run.run_id}"
    print(f"Data move {report.name}{mode}{recorded}: {report.pages} page(s).")
    for outcome in StepOutcome:
        if report.counts[outcome]:
            print(f"  {outcome.value}: {report.counts[outcome]}")
    for item in report.items:
        reason = f": {item.detail}" if item.detail else ""
        print(f"  {item.outcome.value} {item.key}{reason}")
    if report.stopped_by is not None:
        reason = f": {report.stopped_by.detail}" if report.stopped_by.detail else ""
        print(f"  stopped at {report.stopped_by.key}{reason}")
    for problem in report.verify_problems:
        print(f"  verify: {problem}")
    if not report.complete:
        if run is not None:
            print(
                "Stopped before the last page; go on with "
                f"`cayu data-move resume TARGET --run-id {run.run_id}`."
            )
        else:
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
