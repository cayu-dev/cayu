"""`cayu task`: read durable tasks from a local store or a Cayu Cloud Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from cayu.cli._cloud_agent_api import open_cloud_agent_api
from cayu.cli._output import (
    add_output_options,
    print_table,
    terminal_safe,
)
from cayu.cli._read_targets import ReadTarget, add_read_target_options, run_read_command
from cayu.cli._task_readers import (
    MAX_TASK_PAGE,
    ApiTaskReader,
    TaskListRequest,
    TaskReader,
    open_store_task_reader,
)
from cayu.cli.session import _redact_sensitive
from cayu.tasks.records import TaskStatus

CLI_SCHEMA_VERSION = "1"
# A retry series has at most 100 attempts (`TaskRetryPolicy.max_attempts`).
_MAX_SERIES_ATTEMPTS = 100
_CANCELLATION_REQUEST_REASONS = frozenset(
    {"cancellation_requested", "retry_cancellation_requested"}
)


def add_task_parser(subparsers: Any) -> None:
    """Register the ``cayu task`` command group."""

    task = subparsers.add_parser(
        "task",
        help="Inspect and control durable tasks.",
        description=(
            "Read durable tasks from a local store (--sqlite, --postgres, or the project's "
            "store) or from a Cayu Cloud Agent (--application NAME). Start with "
            "`cayu task list`."
        ),
    )
    commands = task.add_subparsers(dest="task_command", required=True)

    from cayu.cli._control_plane_actions import add_action_target_options

    create = commands.add_parser(
        "create",
        help="Queue a declared task on a running agent.",
        description=(
            "Queue a declared task on a running agent. Inspect its result with "
            "`cayu task show TASK_ID`. Retain the operation ID and input for exact retry."
        ),
    )
    create.add_argument("type", help="Declared task type (see the dashboard task catalog).")
    create.add_argument("--input-file", type=Path, help="JSON object containing task input.")
    create.add_argument(
        "--operation-id", required=True, help="Retain this ID and input for exact retry."
    )
    add_action_target_options(create)
    graph = commands.add_parser(
        "graph-create",
        help="Atomically queue a declared task graph.",
        description=(
            "Atomically queue a declared task graph from a JSON file. Inspect an emitted "
            "task ID with `cayu task show TASK_ID`. Retain the operation ID and graph for exact retry."
        ),
    )
    graph.add_argument(
        "file", type=Path, help="JSON object with nodes {key,type,input,prerequisite_keys}."
    )
    graph.add_argument(
        "--operation-id", required=True, help="Retain this ID and graph for exact retry."
    )
    add_action_target_options(graph)
    for name, help_text in (("block", "Block a task."), ("hold", "Pause a task until resumed.")):
        action = commands.add_parser(
            name,
            help=help_text,
            description=(
                f"{help_text} Inspect the current result with `cayu task show TASK_ID` "
                "before retrying an uncertain request."
            ),
        )
        action.add_argument("task_id")
        action.add_argument("--reason")
        action.add_argument(
            "--payload-file", type=Path, help="Optional JSON object with hold details."
        )
        add_action_target_options(action)

    list_parser = commands.add_parser(
        "list",
        help="List tasks, most recently updated first.",
        description=(
            "List tasks, most recently updated first. Rows leave out each task's input and "
            "result; `cayu task show TASK_ID` has them."
        ),
    )
    list_parser.add_argument(
        "--status", choices=tuple(item.value for item in TaskStatus), help="Only this status."
    )
    list_parser.add_argument("--type", help="Only tasks of this exact type.")
    list_parser.add_argument("--agent", help="Only tasks assigned to this agent.")
    list_parser.add_argument("--session", metavar="SESSION_ID", help="Only this session's tasks.")
    list_parser.add_argument(
        "--parent", metavar="TASK_ID", help="Only tasks whose parent is this task."
    )
    list_parser.add_argument(
        "--limit",
        type=_page_limit,
        default=50,
        help=f"Tasks per page, 1-{MAX_TASK_PAGE - 1} (default: 50).",
    )
    list_parser.add_argument(
        "--offset", type=_nonnegative_int, default=0, help="Skip this many tasks."
    )
    add_read_target_options(list_parser)
    add_output_options(list_parser, formats=("table", "json", "jsonl"), default="table")

    show_parser = commands.add_parser(
        "show",
        help="Show one task with its result, cancellation and retry series.",
        description=(
            "Show one task: its status, result or error, why it was cancelled, and every "
            "attempt of its retry series."
        ),
    )
    show_parser.add_argument("task_id")
    add_read_target_options(show_parser)
    add_output_options(show_parser, formats=("table", "json"), default="table")


def run_task(args: argparse.Namespace) -> int:
    if args.task_command in {"create", "graph-create", "block", "hold"}:
        from cayu.cli._control_plane_actions import run_action

        return run_action(args)
    return run_read_command(
        args,
        _run_task_command,
        error_code="TASK_INSPECTION_FAILED",
        schema_version=CLI_SCHEMA_VERSION,
    )


async def _run_task_command(args: argparse.Namespace, target: ReadTarget) -> int:
    reader = open_task_reader(target)
    try:
        if args.task_command == "list":
            return await _list_tasks(args, reader)
        return await _show_task(args, reader)
    finally:
        await reader.close()


def open_task_reader(target: ReadTarget) -> TaskReader:
    """The task reader for a resolved ``--sqlite``/``--postgres``/``--application``."""

    if target.server_url is not None:
        from cayu.cli._cloud_agent_api import CayuApi
        from cayu.cli.diagnostics import read_authorization

        api = CayuApi(
            target.server_url,
            authorization=read_authorization(target.authorization_env),
            timeout_seconds=target.timeout_seconds,
        )
        return ApiTaskReader(api, source={"kind": "server", "url": target.server_url})
    if target.store is not None:
        return open_store_task_reader(target.store)
    assert target.application is not None
    api = open_cloud_agent_api(
        target.application, context=target.context, timeout_seconds=target.timeout_seconds
    )
    return ApiTaskReader(api, source={"kind": "cayu_cloud", "application": target.application})


async def _list_tasks(args: argparse.Namespace, reader: TaskReader) -> int:
    rows = await reader.list_tasks(
        TaskListRequest(
            status=None if args.status is None else TaskStatus(args.status),
            type=args.type,
            agent=args.agent,
            session_id=args.session,
            parent_task_id=args.parent,
            # One more than asked says whether another page follows.
            limit=args.limit + 1,
            offset=args.offset,
        )
    )
    has_more = len(rows) > args.limit
    tasks = [{**row, "cancellation": cancellation_summary(row)} for row in rows[: args.limit]]
    payload = _redact_sensitive(
        {
            "schema_version": CLI_SCHEMA_VERSION,
            "source": reader.source,
            "tasks": tasks,
            "has_more": has_more,
            "next_offset": args.offset + len(tasks) if has_more else None,
        }
    )
    if args.output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    elif args.output_format == "jsonl":
        for task in payload["tasks"]:
            print(json.dumps(task, ensure_ascii=False, sort_keys=True))
    elif not payload["tasks"]:
        print("No tasks.")
    else:
        print_table(
            ("id", "type", "status", "reason", "agent", "attempt", "worker", "updated_at"),
            [_table_row(task) for task in payload["tasks"]],
        )
        if has_more:
            print(f"More tasks follow; add --offset {payload['next_offset']}.")
    return 0


async def _show_task(args: argparse.Namespace, reader: TaskReader) -> int:
    task = await reader.load_task(args.task_id)
    if task is None:
        raise ValueError(f"Task not found: {args.task_id}")
    payload = _redact_sensitive(
        {
            "schema_version": CLI_SCHEMA_VERSION,
            "source": reader.source,
            "task": task,
            "cancellation": cancellation_summary(task),
            "retry_series": await _retry_series(reader, task),
        }
    )
    if args.output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        _print_task(payload)
    return 0


def cancellation_summary(task: dict[str, Any]) -> dict[str, Any] | None:
    """Why a task was cancelled, or that cancelling it was asked for; ``None`` otherwise.

    A live task whose cancellation was asked for carries ``status_reason``
    ``cancellation_requested`` (``retry_cancellation_requested`` in a retry series) until
    its worker stops, with the request's error and event in ``status_payload``. A task
    cancelled after its worker was lost carries the reconciliation in ``status_payload``,
    with its outcome (for example ``quiescent``); otherwise a cancelled task's reason is
    its ``error``.
    """

    status = task.get("status")
    reason = task.get("status_reason")
    requested = reason in _CANCELLATION_REQUEST_REASONS
    if status != TaskStatus.CANCELLED.value and not requested:
        return None
    payload = _object(task.get("status_payload"))
    reconciliation = _object(payload.get("cancellation_reconciliation"))
    # A pending request keeps its error and event in the payload until the worker stops.
    request_event = _object(payload.get("event"))
    error = _object(task.get("error")) or _object(payload.get("error"))
    actor = _object(reconciliation.get("reconciled_by"))
    return {
        "state": "cancelled" if status == TaskStatus.CANCELLED.value else "requested",
        "reason": reason,
        "requested_at": _text(reconciliation.get("cancellation_requested_at"))
        or _text(request_event.get("occurred_at")),
        "outcome": _text(_object(reconciliation.get("evidence")).get("outcome")),
        "reconciled_by": _text(actor.get("subject")),
        "error_code": _text(error.get("code")),
        "error_message": _text(error.get("message")),
    }


def _object(value: object) -> dict[str, Any]:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


async def _retry_series(reader: TaskReader, task: dict[str, Any]) -> dict[str, Any] | None:
    """The task's retry series with every attempt found through its predecessor links."""

    series = task.get("retry_series")
    if not isinstance(series, dict):
        return None
    attempts = {task["id"]: task}
    complete = True
    for link in ("predecessor_task_id", "successor_task_id"):
        current: dict[str, Any] = task
        while True:
            linked = _object(current.get("retry_series")).get(link)
            if not isinstance(linked, str) or linked in attempts:
                break
            if len(attempts) >= _MAX_SERIES_ATTEMPTS:
                # More links than a series can have; stop rather than follow them all.
                complete = False
                break
            loaded = await reader.load_task(linked)
            if loaded is None:
                # The linked attempt is gone (for example, retention removed it).
                complete = False
                break
            attempts[linked] = loaded
            current = loaded
    return {
        **series,
        "attempts": sorted(
            (_attempt_row(attempt) for attempt in attempts.values()),
            key=lambda row: row["attempt"] if isinstance(row["attempt"], int) else 0,
        ),
        "attempts_complete": complete,
    }


def _attempt_row(task: dict[str, Any]) -> dict[str, Any]:
    series = task.get("retry_series")
    series = series if isinstance(series, dict) else {}
    return {
        "task_id": task.get("id"),
        "attempt": series.get("attempt"),
        "status": task.get("status"),
        "disposition": series.get("disposition"),
        "worker_id": task.get("worker_id"),
        "updated_at": task.get("updated_at"),
        "completed_at": task.get("completed_at"),
    }


def _table_row(task: dict[str, Any]) -> dict[str, Any]:
    series = task.get("retry_series")
    attempt = None
    if isinstance(series, dict):
        policy = series.get("policy")
        limit = policy.get("max_attempts") if isinstance(policy, dict) else None
        attempt = f"{series.get('attempt')}/{limit}" if limit is not None else series.get("attempt")
    cancellation = task.get("cancellation")
    reason = task.get("status_reason")
    if isinstance(cancellation, dict) and cancellation.get("outcome"):
        reason = f"{reason or 'cancelled'} ({cancellation['outcome']})"
    return {
        "id": task.get("id"),
        "type": task.get("type"),
        "status": task.get("status"),
        "reason": reason,
        "agent": task.get("assigned_agent_name"),
        "attempt": attempt,
        "worker": task.get("worker_id"),
        "updated_at": task.get("updated_at"),
    }


def _print_task(payload: dict[str, Any]) -> None:
    task = payload["task"]
    fields = (
        "id",
        "type",
        "title",
        "status",
        "status_reason",
        "assigned_agent_name",
        "session_id",
        "parent_task_id",
        "worker_id",
        "lease_expires_at",
        "created_at",
        "started_at",
        "updated_at",
        "completed_at",
    )
    print_table(
        ("field", "value"),
        [{"field": name, "value": task.get(name)} for name in fields if task.get(name) is not None],
    )
    cancellation = payload["cancellation"]
    if cancellation is not None:
        print("\nCancellation")
        print_table(
            ("field", "value"),
            [{"field": key, "value": value} for key, value in cancellation.items() if value],
        )
    series = payload["retry_series"]
    if series is not None:
        print(
            "\n"
            + terminal_safe(f"Retry series {series.get('series_id')} ({series.get('disposition')})")
        )
        print_table(
            ("attempt", "task_id", "status", "disposition", "completed_at"),
            series["attempts"],
        )
        if not series["attempts_complete"]:
            print("Some attempts of this series could not be read.")
    for section in ("result", "error"):
        value = task.get(section)
        if value is not None:
            print(f"\n{section.capitalize()}")
            print(
                terminal_safe(
                    json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
                    keep_lines=True,
                )
            )


def _page_limit(value: str) -> int:
    limit = _nonnegative_int(value)
    if not 1 <= limit < MAX_TASK_PAGE:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_TASK_PAGE - 1}")
    return limit


def _nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if number < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return number
