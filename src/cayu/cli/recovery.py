"""Registered-application recovery planning and operator execution."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cayu.cli._output import add_output_options, output_destination
from cayu.cli.project import ProjectError, build_project_app, project_context, resolve_project
from cayu.runtime import (
    RECOVERY_PLAN_MAX_CONCURRENCY,
    RECOVERY_PLAN_MAX_INSPECTIONS,
    RECOVERY_PLAN_MAX_ITEMS,
    RecoveryDecision,
    RecoveryExecutionRequest,
    RecoveryPlan,
    RecoveryPlanBounds,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
    SessionStatus,
)

_MAX_PLAN_FILE_BYTES = 16 * 1024 * 1024
_MAX_DECISIONS_FILE_BYTES = 2 * 1024 * 1024


def add_recovery_parser(subparsers: Any) -> None:
    recovery = subparsers.add_parser(
        "recovery",
        help="Plan and execute bounded recovery through the registered application.",
        description=(
            "Inspect recovery work through the canonical Cayu application factory, then "
            "execute an exact saved plan. Planning is read-only; execution rejects stale "
            "durable state and records per-session receipts."
        ),
    )
    commands = recovery.add_subparsers(dest="recovery_command", required=True)

    plan = commands.add_parser(
        "plan",
        help="Create a bounded, secret-free recovery plan without mutating storage.",
        description=(
            "Boot the registered application once and inspect exact session, registration, "
            "claim, model, tool, task, and interruption-cascade state. Save the JSON output "
            "and pass it unchanged to `cayu recovery execute`."
        ),
    )
    plan.add_argument(
        "target",
        nargs="?",
        help="Override project discovery with a module:factory target.",
    )
    selection = plan.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--session",
        dest="session_ids",
        action="append",
        metavar="SESSION_ID",
        help="Inspect one exact session; repeat for more sessions.",
    )
    selection.add_argument(
        "--status",
        dest="statuses",
        action="append",
        choices=tuple(status.value for status in SessionStatus),
        metavar="STATUS",
        help="Inspect one lifecycle status; repeat for more statuses.",
    )
    plan.add_argument(
        "--inactive-for-seconds",
        type=_nonnegative_int,
        help="Include only sessions inactive for at least this many seconds.",
    )
    plan.add_argument(
        "--limit",
        type=_bounded_item_limit,
        default=100,
        help=f"Maximum plan items (default: 100; max: {RECOVERY_PLAN_MAX_ITEMS}).",
    )
    plan.add_argument(
        "--inspection-limit",
        type=_bounded_inspection_limit,
        default=1000,
        help=(
            f"Maximum session rows inspected (default: 1000; max: {RECOVERY_PLAN_MAX_INSPECTIONS})."
        ),
    )
    plan.add_argument("--cursor", help="Continue a status-based plan page.")
    add_output_options(plan, formats=("json",))

    execute = commands.add_parser(
        "execute",
        help="Execute decisions bound to an exact saved recovery plan.",
        description=(
            "Load an immutable plan, boot the same registered application once, reject "
            "changed durable state, and emit replayable per-session receipts. No action "
            "with an unknown external effect is selected automatically."
        ),
    )
    execute.add_argument("plan_file", metavar="PLAN.json")
    execute.add_argument(
        "--target",
        help="Override project discovery with a module:factory target.",
    )
    execute.add_argument(
        "--execution-id",
        required=True,
        help="Stable operator-supplied id; reuse it to replay an ambiguous invocation.",
    )
    execute.add_argument(
        "--decisions",
        metavar="DECISIONS.json",
        help="Optional JSON array of RecoveryDecision objects.",
    )
    execute.add_argument(
        "--max-concurrency",
        type=_bounded_concurrency,
        default=1,
        help=(
            "Maximum independent sessions executed concurrently "
            f"(default: 1; max: {RECOVERY_PLAN_MAX_CONCURRENCY})."
        ),
    )
    add_output_options(execute, formats=("json",))


def run_recovery(args: argparse.Namespace) -> int:
    try:
        with output_destination(args.output):
            return asyncio.run(_run_recovery(args))
    except OSError:
        _render_error("OUTPUT_UNAVAILABLE", "Could not read or write the requested file.")
        return 1
    except (ProjectError, ValidationError, ValueError) as exc:
        _render_error(_error_code(exc), _safe_expected_error(exc))
        return 1
    except Exception as exc:
        _render_error(
            type(exc).__name__,
            "Recovery command failed. Run `cayu inspect` to verify the application factory "
            "and `cayu recovery plan` to refresh durable state.",
        )
        return 1


async def _run_recovery(args: argparse.Namespace) -> int:
    if args.recovery_command == "plan":
        request = RecoveryPlanRequest(
            selection=RecoveryPlanSelection(
                session_ids=tuple(args.session_ids or ()),
                statuses=frozenset(SessionStatus(value) for value in (args.statuses or ())),
                inactive_for_seconds=args.inactive_for_seconds,
                cursor=args.cursor,
            ),
            bounds=RecoveryPlanBounds(
                item_limit=args.limit,
                inspection_limit=args.inspection_limit,
            ),
        )
        project = resolve_project(args.target, command="cayu recovery plan")
        with project_context(project.root):
            app = build_project_app(project.target, command="Recovery plan")
            result = await app.plan_recovery(request)
    elif args.recovery_command == "execute":
        plan = RecoveryPlan.model_validate_json(
            _read_bounded_file(Path(args.plan_file), limit=_MAX_PLAN_FILE_BYTES)
        )
        decisions: tuple[RecoveryDecision, ...] = ()
        if args.decisions is not None:
            raw_decisions = json.loads(
                _read_bounded_file(
                    Path(args.decisions),
                    limit=_MAX_DECISIONS_FILE_BYTES,
                )
            )
            if type(raw_decisions) is not list:
                raise ValueError("Decisions input must be a JSON array.")
            decisions = tuple(RecoveryDecision.model_validate(item) for item in raw_decisions)
        request = RecoveryExecutionRequest(
            plan=plan,
            execution_id=args.execution_id,
            decisions=decisions,
            max_concurrency=args.max_concurrency,
        )
        project = resolve_project(args.target, command="cayu recovery execute")
        with project_context(project.root):
            app = build_project_app(project.target, command="Recovery execute")
            result = await app.execute_recovery(request)
    else:
        raise ValueError("Unknown recovery command.")
    print(result.model_dump_json(indent=2))
    return 0


def _read_bounded_file(path: Path, *, limit: int) -> bytes:
    stat = path.stat()
    if stat.st_size > limit:
        raise ValueError(f"{path.name} exceeds the {limit}-byte input limit.")
    value = path.read_bytes()
    if len(value) > limit:
        raise ValueError(f"{path.name} exceeds the {limit}-byte input limit.")
    return value


def _bounded_int(value: str, *, maximum: int, field_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{field_name} must be an integer") from exc
    if parsed < 1 or parsed > maximum:
        raise argparse.ArgumentTypeError(f"{field_name} must be between 1 and {maximum}")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _bounded_item_limit(value: str) -> int:
    return _bounded_int(value, maximum=RECOVERY_PLAN_MAX_ITEMS, field_name="limit")


def _bounded_inspection_limit(value: str) -> int:
    return _bounded_int(
        value,
        maximum=RECOVERY_PLAN_MAX_INSPECTIONS,
        field_name="inspection-limit",
    )


def _bounded_concurrency(value: str) -> int:
    return _bounded_int(
        value,
        maximum=RECOVERY_PLAN_MAX_CONCURRENCY,
        field_name="max-concurrency",
    )


def _error_code(exc: Exception) -> str:
    if isinstance(exc, ProjectError):
        return "PROJECT_BOOT_FAILED"
    if isinstance(exc, ValidationError):
        return "INVALID_RECOVERY_INPUT"
    return "INVALID_RECOVERY_INPUT"


def _safe_expected_error(exc: Exception) -> str:
    if isinstance(exc, ProjectError):
        return str(exc)
    if isinstance(exc, ValidationError):
        return "Recovery input failed contract validation."
    return str(exc)


def _render_error(code: str, message: str) -> None:
    print(
        json.dumps(
            {
                "record_type": "cayu.recovery-error",
                "schema_version": 1,
                "error": {"code": code, "message": message},
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )


__all__ = ["add_recovery_parser", "run_recovery"]
