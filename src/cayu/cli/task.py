"""`cayu task`: read durable tasks from a local store or a Cayu Cloud Agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from cayu.cli._cloud_agent_api import CayuApiError, open_cloud_agent_api
from cayu.cli._output import (
    add_output_options,
    note,
    print_table,
    terminal_safe,
)
from cayu.cli._read_targets import (
    ReadTarget,
    _positive_seconds,
    add_read_target_options,
    run_read_command,
)
from cayu.cli._task_readers import (
    MAX_TASK_PAGE,
    ApiTaskReader,
    TaskListRequest,
    TaskReader,
    open_store_task_reader,
)
from cayu.cli.session import _redact_sensitive
from cayu.tasks.cancellation import (
    TaskCancellationReconciliationEventType,
    TaskRetryCancellationReconciliationEventType,
)
from cayu.tasks.records import TaskStatus
from cayu.tasks.restart import WORKER_LOSS_FENCE_CODES

CLI_SCHEMA_VERSION = "1"
# A retry series has at most 100 attempts (`TaskRetryPolicy.max_attempts`).
_MAX_SERIES_ATTEMPTS = 100
_CANCELLATION_REQUEST_REASONS = frozenset(
    {"cancellation_requested", "retry_cancellation_requested"}
)


class WorkerLossEvidence(StrEnum):
    """What made it safe to settle a task whose worker lost its lease (``evidence_kind``)."""

    LEASE_EXPIRED = "lease_expired"
    RESTART_LIMIT_REACHED = "restart_limit_reached"
    SESSION_INTERRUPTED = "session_interrupted"
    TASK_ENDED = "task_ended"
    HANDLER_STOPPED = "handler_stopped"
    OTHER = "other"


#: How the task table output names what fenced a cancelled task.
_FENCED_BY_TEXT = {"worker_loss": "worker lost", "request": "requested"}

# The evidence id prefixes Cayu records when it settles such a task; see `_worker_loss`.
_WORKER_LOSS_EVIDENCE_PREFIXES = (
    ("lease-expired-at:", WorkerLossEvidence.LEASE_EXPIRED),
    ("restart-limit-reached:", WorkerLossEvidence.RESTART_LIMIT_REACHED),
    ("session-interrupted:", WorkerLossEvidence.SESSION_INTERRUPTED),
    ("task-ended:", WorkerLossEvidence.TASK_ENDED),
    ("task-worker-quiescence:", WorkerLossEvidence.HANDLER_STOPPED),
    ("task-worker-retry-quiescence:", WorkerLossEvidence.HANDLER_STOPPED),
    ("dispatch-quiescence:", WorkerLossEvidence.HANDLER_STOPPED),
)
# Evidence ids that end in ":<session instance>:<run epoch>".
_SESSION_RUN_EVIDENCE = frozenset(
    {WorkerLossEvidence.SESSION_INTERRUPTED, WorkerLossEvidence.TASK_ENDED}
)
_RECONCILED_EVENT_TYPES = frozenset(
    {
        TaskCancellationReconciliationEventType.RECONCILED.value,
        TaskRetryCancellationReconciliationEventType.RECONCILED.value,
    }
)

#: ``cayu task wait`` exit codes. 1 remains any other error and 2 an argparse usage error.
EXIT_TASK_COMPLETED = 0
#: The task failed, or was skipped because a prerequisite didn't complete.
EXIT_TASK_FAILED = 3
EXIT_TASK_CANCELLED = 4
#: The task is still unfinished after ``--wait-seconds``, as `timeout` reports it.
EXIT_WAIT_TIMED_OUT = 124
# A task ends in one of these statuses; each has its exit code.
_TERMINAL_EXIT_CODES = {
    TaskStatus.COMPLETED.value: EXIT_TASK_COMPLETED,
    TaskStatus.FAILED.value: EXIT_TASK_FAILED,
    TaskStatus.DEPENDENCY_SKIPPED.value: EXIT_TASK_FAILED,
    TaskStatus.CANCELLED.value: EXIT_TASK_CANCELLED,
}
_TIMED_OUT = "timed_out"
DEFAULT_WAIT_SECONDS = 6 * 60 * 60.0
DEFAULT_POLL_SECONDS = 5.0
# Commands that change a task through the agent's API (`_control_plane_actions`).
_ACTIONS = frozenset({"create", "graph-create", "block", "hold", "cancel", "resume", "replay"})


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

    cancel = commands.add_parser(
        "cancel",
        help="Cancel a task.",
        description=(
            "Cancel a task. Idle work is cancelled at once; a task a worker is running is "
            "asked to stop and is cancelled once its worker has stopped (its reason shows "
            "cancellation_requested until then). Tasks that depend on it are then skipped. "
            "Inspect the result with `cayu task show TASK_ID`."
        ),
    )
    cancel.add_argument("task_id")
    cancel.add_argument("--reason", help="Why, recorded as the task's cancellation message.")
    add_action_target_options(cancel)
    resume = commands.add_parser(
        "resume",
        help="Resume a held, blocked or attention-needed task.",
        description=(
            "Return a task that `cayu task hold` paused, `cayu task block` blocked or that "
            "needs attention to the pending queue, where a worker picks it up again."
        ),
    )
    resume.add_argument("task_id")
    add_action_target_options(resume)
    replay = commands.add_parser(
        "replay",
        help="Queue a declared task again with the same input.",
        description=(
            "Queue a new task of the same declared type with the same input as TASK_ID, "
            "as the dashboard's replay does. The agent must still declare the type and "
            "accept the input. Retain the operation ID for exact retry; the new task's ID "
            "is printed."
        ),
    )
    replay.add_argument("task_id")
    replay.add_argument("--operation-id", required=True, help="Retain this ID for exact retry.")
    add_action_target_options(replay)

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
            "Show one task: its status, result or error, why it was cancelled (for a task "
            "Cayu settled after its worker was lost, which worker, when its lease expired "
            "and what made settling it safe), and every attempt of its retry series."
        ),
    )
    show_parser.add_argument("task_id")
    add_read_target_options(show_parser)
    add_output_options(show_parser, formats=("table", "json"), default="table")

    wait_parser = commands.add_parser(
        "wait",
        help="Wait until a task ends, then show it.",
        description=(
            "Wait until a task completes, fails or is cancelled, then show it as "
            "`cayu task show` does. A failed attempt whose retry series goes on is followed "
            f"to its next attempt. Exits {EXIT_TASK_COMPLETED} when the task completed, "
            f"{EXIT_TASK_FAILED} when it failed or was skipped, {EXIT_TASK_CANCELLED} when it "
            f"was cancelled, {EXIT_WAIT_TIMED_OUT} when it is still unfinished after "
            "--wait-seconds, and 1 when it can't be read."
        ),
    )
    wait_parser.add_argument("task_id")
    wait_parser.add_argument(
        "--wait-seconds",
        type=_positive_seconds,
        default=DEFAULT_WAIT_SECONDS,
        metavar="SECONDS",
        help=f"How long to wait for the task (default: {DEFAULT_WAIT_SECONDS:.0f}).",
    )
    wait_parser.add_argument(
        "--poll-seconds",
        type=_positive_seconds,
        default=DEFAULT_POLL_SECONDS,
        metavar="SECONDS",
        help=f"How often to read the task (default: {DEFAULT_POLL_SECONDS:.0f}).",
    )
    add_read_target_options(wait_parser)
    add_output_options(wait_parser, formats=("table", "json"), default="table")


def run_task(args: argparse.Namespace) -> int:
    if args.task_command in _ACTIONS:
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
        if args.task_command == "wait":
            return await _wait_for_task(args, reader)
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
        target.application,
        context=target.context,
        timeout_seconds=target.timeout_seconds,
        wake_timeout_seconds=target.wake_timeout_seconds,
        offers_wake_option=True,
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
    _render_task(args, await _task_payload(reader, task))
    return 0


async def _wait_for_task(args: argparse.Namespace, reader: TaskReader) -> int:
    """Read the task until it ends or ``--wait-seconds`` pass, then show it.

    A terminal attempt with a successor in its retry series hands the wait to that attempt.
    Failures to reach a server say nothing about the task, so they are read through, except a
    sleeping Agent under ``--no-wait``, which fails at once as every read does.
    """

    deadline = time.monotonic() + args.wait_seconds
    task_id = args.task_id
    task: dict[str, Any] | None = None
    unread: CayuApiError | None = None
    shown: str | None = None
    followed = {task_id}
    no_wake_wait = args.wake_timeout_seconds == 0
    while True:
        try:
            current = await reader.load_task(task_id)
        except CayuApiError as exc:
            if not exc.transient or (exc.category == "starting" and no_wake_wait):
                raise
            if unread is None:
                note(f"{exc} Reading task {task_id} again.")
            unread = exc
        else:
            unread = None
            if current is None:
                raise ValueError(f"Task not found: {task_id}")
            task = current
            status = task["status"]
            successor = _object(task.get("retry_series")).get("successor_task_id")
            if status in _TERMINAL_EXIT_CODES and isinstance(successor, str):
                if successor in followed or len(followed) >= _MAX_SERIES_ATTEMPTS:
                    # A series links each attempt to a new one, at most 100 of them.
                    raise ValueError(
                        f"Task {args.task_id}'s retry series can't be followed past task "
                        f"{task_id}: its attempts link back to an earlier one or exceed "
                        f"{_MAX_SERIES_ATTEMPTS}."
                    )
                note(f"Task {task_id} {status}; its retry series goes on as task {successor}.")
                followed.add(successor)
                task_id = successor
                # Only the attempt waited on now says how the work ended.
                task = None
                continue
            line = f"Task {task_id} is {status}."
            if line != shown:
                note(line)
                shown = line
            if status in _TERMINAL_EXIT_CODES:
                break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(args.poll_seconds, remaining))
    if task is None:
        # Every read of the attempt waited on failed until the wait ran out.
        assert unread is not None
        raise unread
    outcome = task["status"] if task["status"] in _TERMINAL_EXIT_CODES else _TIMED_OUT
    payload = await _task_payload(reader, task)
    payload["wait"] = {"task_id": args.task_id, "outcome": outcome}
    _render_task(args, payload)
    if outcome == _TIMED_OUT:
        note(
            f"Task {task['id']} is still {task['status']} after waiting "
            f"{args.wait_seconds:g}s; it goes on."
        )
        return EXIT_WAIT_TIMED_OUT
    return _TERMINAL_EXIT_CODES[outcome]


async def _task_payload(reader: TaskReader, task: dict[str, Any]) -> dict[str, Any]:
    return _redact_sensitive(
        {
            "schema_version": CLI_SCHEMA_VERSION,
            "source": reader.source,
            "task": task,
            "cancellation": cancellation_summary(task),
            "retry_series": await _retry_series(reader, task),
        }
    )


def _render_task(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if args.output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        _print_task(payload)


def cancellation_summary(task: dict[str, Any]) -> dict[str, Any] | None:
    """Why a task was cancelled, or that cancelling it was asked for; ``None`` otherwise.

    A live task whose cancellation was asked for carries ``status_reason``
    ``cancellation_requested`` (``retry_cancellation_requested`` in a retry series) until
    its worker stops, with the request's error and event in ``status_payload``. A task
    cancelled after its worker was lost carries the reconciliation in ``status_payload``,
    with its outcome (for example ``quiescent``), and ``worker_loss`` tells what happened;
    otherwise a cancelled task's reason is its ``error``.

    ``fenced_by`` says what fenced a task that was asked to stop or settled after its worker
    was lost: ``worker_loss`` when the worker or expired-claim reclamation fenced it,
    ``request`` when someone asked to cancel it. Both settle the same way, so only the fence's
    error code tells them apart. Once a task is settled that code is only in its ``error``,
    which list rows leave out, so a settled task's list row has ``fenced_by`` ``None``.
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
    code = _text(error.get("code"))
    fenced_by = None
    if (requested or reconciliation) and code is not None:
        fenced_by = "worker_loss" if code in WORKER_LOSS_FENCE_CODES else "request"
    return {
        "state": "cancelled" if status == TaskStatus.CANCELLED.value else "requested",
        "reason": reason,
        "requested_at": _text(reconciliation.get("cancellation_requested_at"))
        or _text(request_event.get("occurred_at")),
        "outcome": _text(_object(reconciliation.get("evidence")).get("outcome")),
        "reconciled_by": _text(actor.get("subject")),
        "error_code": code,
        "error_message": _text(error.get("message")),
        "fenced_by": fenced_by,
        "worker_loss": _worker_loss(task, reconciliation),
    }


def _worker_loss(task: dict[str, Any], reconciliation: dict[str, Any]) -> dict[str, Any] | None:
    """How a task was settled after the worker holding it lost its lease; ``None`` otherwise.

    The reconciliation's evidence id says what made settling it safe. Cayu records:

    - ``lease-expired-at:TIME``: the worker was found gone once its lease expired, by a
      worker declared the only one (``--reconcile-abandoned-after-seconds``) or by the
      application (``reconcile_abandoned_tasks``).
    - ``restart-limit-reached:LIMIT:lease-expired-at:TIME``: the same, for a restartable
      task that had used up its restarts after a lost worker.
    - ``session-interrupted:INSTANCE:EPOCH``: the task's agent session is interrupted,
      which fences the lost worker's run.
    - ``task-ended:INSTANCE:EPOCH``: a session wake whose task had already ended.
    - ``task-worker-quiescence:``, ``task-worker-retry-quiescence:`` and
      ``dispatch-quiescence:``: the worker's lease expired before it settled the
      cancellation, and it saw its handler stop.

    An application that settles a task itself may record other evidence (``other``).
    """

    worker_id = _text(reconciliation.get("original_worker_id"))
    lease_expired_at = _text(reconciliation.get("original_lease_expires_at"))
    if worker_id is None or lease_expired_at is None:
        return None
    evidence = _object(reconciliation.get("evidence"))
    evidence_id = _text(evidence.get("evidence_id")) or ""
    kind = next(
        (kind for prefix, kind in _WORKER_LOSS_EVIDENCE_PREFIXES if evidence_id.startswith(prefix)),
        WorkerLossEvidence.OTHER,
    )
    # What follows the prefix: "LIMIT:lease-expired-at:TIME" or "INSTANCE:EPOCH".
    detail = evidence_id.partition(":")[2]
    limit = detail.partition(":")[0]
    instance, _, epoch = detail.rpartition(":")
    session_run = kind in _SESSION_RUN_EVIDENCE
    actor = _object(reconciliation.get("reconciled_by"))
    events = reconciliation.get("events")
    settled = next(
        (
            event
            for event in map(_object, events if isinstance(events, list) else ())
            if event.get("type") in _RECONCILED_EVENT_TYPES
        ),
        {},
    )
    loss: dict[str, Any] = {
        "worker_id": worker_id,
        "lease_expired_at": lease_expired_at,
        "settled_at": _text(settled.get("occurred_at")),
        "settled_by": _text(actor.get("subject")),
        "settled_by_source": _text(actor.get("source")),
        "evidence_kind": kind.value,
        "evidence_id": evidence_id or None,
        "validator_id": _text(evidence.get("validator_id")),
        "validator_version": _text(evidence.get("validator_version")),
        "session_instance_id": (instance or None) if session_run else None,
        "session_run_epoch": _count(epoch) if session_run else None,
        "restart_limit": (
            _count(limit) if kind is WorkerLossEvidence.RESTART_LIMIT_REACHED else None
        ),
    }
    loss["summary"] = _worker_loss_summary(task, loss, kind)
    return loss


def _worker_loss_summary(
    task: dict[str, Any], loss: dict[str, Any], kind: WorkerLossEvidence
) -> str:
    worker, expired = loss["worker_id"], loss["lease_expired_at"]
    if kind is WorkerLossEvidence.HANDLER_STOPPED:
        sentences = [
            f"Worker {worker}'s lease expired at {expired} before it settled the "
            "cancellation, and it saw its handler stop."
        ]
    else:
        sentences = [f"Worker {worker} stopped renewing its lease, which expired at {expired}."]
    limit = loss["restart_limit"]
    if limit is not None:
        sentences.append(
            f"It wasn't restarted: it had reached its limit of {limit} "
            f"restart{'' if limit == 1 else 's'} after a lost worker."
        )
    elif kind is WorkerLossEvidence.SESSION_INTERRUPTED:
        session = _text(task.get("session_id"))
        epoch = loss["session_run_epoch"]
        sentences.append(
            "Its session"
            + ("" if session is None else f" {session}")
            + " was interrupted"
            + ("" if epoch is None else f" (run epoch {epoch})")
            + ", so the lost run couldn't go on."
        )
    elif kind is WorkerLossEvidence.TASK_ENDED:
        sentences.append("The task this session wake would have continued had already ended.")
    cancelled = "Cayu cancelled the task"
    if loss["settled_at"] is not None:
        cancelled += f" at {loss['settled_at']}"
    if loss["settled_by"] is not None:
        cancelled += f" (settled by {loss['settled_by']})"
    sentences.append(cancelled + ".")
    return " ".join(sentences)


def _object(value: object) -> dict[str, Any]:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _count(text: str) -> int | None:
    # Only ASCII digits: `int` refuses others that `isdigit` accepts, such as "²".
    return int(text) if text.isascii() and text.isdigit() else None


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
            try:
                loaded = await reader.load_task(linked)
            except CayuApiError as exc:
                if not exc.transient:
                    raise
                # The server didn't answer for this attempt just now; the task itself was
                # read, so show it with the attempts read so far.
                loaded = None
            if loaded is None:
                # The linked attempt is gone (for example, retention removed it) or unread.
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
    if isinstance(cancellation, dict):
        details = [
            detail
            for detail in (
                cancellation.get("outcome"),
                _FENCED_BY_TEXT.get(cancellation.get("fenced_by")),
            )
            if detail
        ]
        if details:
            reason = f"{reason or 'cancelled'} ({', '.join(details)})"
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
            [
                {
                    "field": key,
                    "value": _FENCED_BY_TEXT.get(value, value) if key == "fenced_by" else value,
                }
                for key, value in cancellation.items()
                if value and key != "worker_loss"
            ],
        )
        loss = cancellation["worker_loss"]
        if loss is not None:
            print("\nWorker loss")
            print(terminal_safe(loss["summary"]))
            print_table(
                ("field", "value"),
                [
                    {"field": key, "value": value}
                    for key, value in loss.items()
                    if value is not None and key != "summary"
                ],
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
