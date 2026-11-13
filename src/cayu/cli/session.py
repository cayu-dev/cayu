"""Bounded inspection and explicit recovery of durable Cayu sessions."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import sys
import unicodedata
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx

from cayu._validation import compact_json_utf8_size
from cayu.budgets.pricing import PriceBook, default_price_book, load_price_book
from cayu.cli._output import add_output_options, output_destination, print_table, terminal_safe
from cayu.cli._session_reader import (
    OWNER_LOST_STATUSES,
    SessionReader,
    StoreSessionReader,
    TranscriptEntry,
    aggregate_usage_cli_payload,
    owner_lost_sessions,
)
from cayu.cli._workflow_run import SessionUsage as WorkflowSessionUsage
from cayu.cli._workflow_run import read_workflow_run, run_status, session_ended_at, step_times
from cayu.cli.storage import _sanitize
from cayu.cli.store_targets import (
    SessionStoreBackend,
    SessionStoreTarget,
    SessionStoreTargetError,
    resolve_session_store_target,
)
from cayu.events import EventType
from cayu.events import _validate_custom_event_type as validate_custom_event_type
from cayu.runtime.public_authority import public_authority_alias_codec_from_environment
from cayu.sessions._call_inspection import InspectionWindow
from cayu.sessions._call_inspection import redact_inspection_value as _redact_sensitive
from cayu.sessions._call_inspection import truncate_utf8 as _truncate_utf8
from cayu.sessions._event_loop_lag import event_loop_lag_text
from cayu.sessions._rollup import (
    RollupGroup,
    SessionRollupQuery,
    merge_groups,
    rollup_grouping,
    top_failure_kinds,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.interactions import InteractionStatus
from cayu.sessions.queries import SessionOrder, SessionQuery
from cayu.sessions.records import EventRecord, SessionStatus
from cayu.sessions.store import SessionStore
from cayu.sessions.transcript_queries import TranscriptQuery
from cayu.storage import SQLiteSessionStore
from cayu.storage import migrations as schema

FORMAT_CHOICES = ("json", "table", "jsonl")
CLI_SCHEMA_VERSION = "9"
_MAX_TRANSCRIPT_CONTENT_BYTES = 1_048_576
_CLOSE_REASON_MAX_CHARS = 4096
_MAX_SERVER_ERROR_BODY_BYTES = 128 * 1024
_MAX_SERVER_ERROR_DETAIL_BYTES = 8 * 1024
_MAX_SERVER_ERROR_DETAIL_SOURCE_BYTES = 64 * 1024
_MAX_TRANSCRIPT_SUMMARY_PARTS = 100
#: ``cayu session output`` exit codes besides 0 (printed), 1 (any other error) and 2
#: (argparse usage error).
EXIT_NO_STRUCTURED_OUTPUT = 3
EXIT_SESSION_NOT_FINISHED = 4
_UNFINISHED_SESSION_STATUSES = frozenset(
    {SessionStatus.PENDING.value, SessionStatus.RUNNING.value, SessionStatus.INTERRUPTING.value}
)
_UNFINISHED_INTERACTION_STATUSES = frozenset(
    {InteractionStatus.ACTIVE.value, InteractionStatus.PAUSED.value}
)


class SessionOutputUnavailable(ValueError):
    """`session output` has nothing to print yet, or nothing at all; ``code`` says which."""

    def __init__(self, code: str, message: str, *, exit_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


def add_session_parser(subparsers: Any) -> None:
    """Register the singular ``cayu session`` command group."""

    session = subparsers.add_parser(
        "session",
        help="Inspect durable sessions, start one, and resolve unavailable provider work.",
        description=(
            "Inspect durable Cayu sessions without mutating storage, in a local store or a "
            "Cayu Cloud Agent (--application NAME), start a conversation as a person "
            "(`cayu session start`), or send an explicit provider-operation disposition to a "
            "running Cayu server. Start with `cayu session list`; JSON is the default output."
        ),
    )
    commands = session.add_subparsers(dest="session_command", required=True)
    from cayu.cli._session_start import add_session_start_parser

    add_session_start_parser(commands)
    list_parser = commands.add_parser(
        "list",
        help="List sessions by newest activity.",
        description=(
            "List sessions by newest activity. Use a returned session id with "
            "`cayu session show SESSION_ID`."
        ),
    )
    _add_target_options(list_parser)
    list_parser.add_argument("--status", choices=tuple(item.value for item in SessionStatus))
    list_parser.add_argument("--agent", help="Filter by exact agent name.")
    list_parser.add_argument("--environment", help="Filter by exact environment name.")
    list_parser.add_argument(
        "--task",
        metavar="TASK_ID",
        help=(
            "Only sessions run under this task: attached to it, created while its handler "
            "ran, or children of such a session."
        ),
    )
    list_parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Filter by an exact label; repeat for multiple labels.",
    )
    _add_created_window_options(list_parser)
    list_parser.add_argument("--limit", type=_positive_limit, default=50)
    paging = list_parser.add_mutually_exclusive_group()
    paging.add_argument("--offset", type=_nonnegative_int, default=0)
    paging.add_argument("--cursor")
    list_parser.add_argument(
        "--overview",
        action="store_true",
        help=(
            "Add each session's overview, as the Sessions page shows it: state and why, "
            "title, summary, first user message, steps, tokens, cost, and whether it used "
            "a browser, subagents or artifacts."
        ),
    )
    list_parser.add_argument(
        "--price-book",
        "--price-table",
        dest="price_book",
        type=Path,
        metavar="PATH",
        help=(
            "With --overview on a local store: price costs with this price book instead of "
            "Cayu's bundled one. A Cloud Agent prices with its own."
        ),
    )
    list_parser.add_argument(
        "--owner-lost",
        action="store_true",
        help=(
            "Only running or interrupting sessions whose execution owner is gone (its "
            "lease expired or its process died), each with its execution presence. Reads "
            "every such session's state, newest activity first, until --limit are found."
        ),
    )
    add_output_options(list_parser, formats=FORMAT_CHOICES)

    rollup_parser = commands.add_parser(
        "rollup",
        help="Sum usage, time and tool failures across many sessions.",
        description=(
            "Sum token usage, time by phase (model calls, tool calls, environment setup and "
            "the runtime gaps between them) and tool failures by error kind across the "
            "sessions created in a window, grouped by agent, UTC day, Release or label. "
            "Sessions are read a page at a time and each session's events in bounded "
            "pages, so a large store is never loaded at once."
        ),
    )
    _add_target_options(rollup_parser)
    _add_created_window_options(rollup_parser)
    rollup_parser.add_argument(
        "--by",
        dest="group_by",
        default="agent",
        type=_group_by_argument,
        metavar="agent|day|release|label:KEY",
        help="Group sessions by agent (default), UTC creation day, Release, or a label.",
    )
    rollup_parser.add_argument("--status", choices=tuple(item.value for item in SessionStatus))
    rollup_parser.add_argument("--agent", help="Only sessions of this exact agent.")
    rollup_parser.add_argument("--environment", help="Only sessions of this exact environment.")
    rollup_parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Only sessions with this exact label; repeat for multiple labels.",
    )
    rollup_parser.add_argument(
        "--causal-budget",
        metavar="ID",
        help=(
            "Only sessions charged to this causal budget: a run and the child sessions it "
            "spawned share its id."
        ),
    )
    rollup_parser.add_argument(
        "--root",
        metavar="SESSION_ID",
        help="Only this root session and the sessions its invocation started.",
    )
    rollup_parser.add_argument(
        "--task",
        metavar="TASK_ID",
        help="Only the sessions run under this task (see `cayu session list --task`).",
    )
    rollup_parser.add_argument(
        "--max-sessions",
        type=_positive_int,
        default=10_000,
        help=(
            "Stop after this many sessions (default 10000); the output says whether the "
            "window was complete and gives a cursor to continue."
        ),
    )
    rollup_parser.add_argument(
        "--cursor", help="Continue a rollup after the sessions an earlier run read."
    )
    _add_price_book_option(rollup_parser)
    rollup_parser.add_argument(
        "--failure-kinds",
        type=_positive_int,
        default=10,
        metavar="N",
        help="Tool failure kinds shown per group in tables (default 10; JSON has all).",
    )
    add_output_options(rollup_parser, formats=FORMAT_CHOICES)

    show_parser = commands.add_parser(
        "show",
        help="Show a compact session overview.",
        description=(
            "Show a compact session overview, including execution ownership, lease expiry "
            "and owner_lost state. Also shows the run epoch, active model_stage "
            "(purpose, dispatch state, provider and model), and latest provider_operation "
            "by identity and state only. Use usage, tools, events, or transcript for bounded detail."
        ),
    )
    show_parser.add_argument("session_id")
    _add_target_options(show_parser)
    add_output_options(show_parser, formats=FORMAT_CHOICES)

    from cayu.cli._runtime_timing import add_runtime_timing_parser

    add_runtime_timing_parser(commands)

    workflow_parser = commands.add_parser(
        "workflow",
        help="Show a workflow run: its steps, their child sessions, usage and custom events.",
        description=(
            "Show one workflow run as a whole: each journaled step with its child session, "
            "agent, status, start and end, model calls and tokens (the child session and "
            "everything under it), totals for the run including children of children, and "
            "the custom.* events the workflow emitted, in order. Works on any session; one "
            "without a workflow journal lists its child sessions."
        ),
    )
    workflow_parser.add_argument("session_id", help="The workflow run's own session id.")
    _add_target_options(workflow_parser)
    _add_price_book_option(workflow_parser)
    workflow_parser.add_argument(
        "--max-sessions",
        type=_positive_int,
        default=1000,
        help=(
            "Stop walking child sessions after this many sessions (default 1000); the "
            "output then says the run is incomplete."
        ),
    )
    workflow_parser.add_argument(
        "--max-events",
        type=_positive_int,
        default=100_000,
        help="Read at most this many of the run's own events (default 100000).",
    )
    workflow_parser.add_argument(
        "--payload-bytes",
        type=_payload_limit,
        default=2048,
        metavar="BYTES",
        help="Bound each custom event's payload preview (default 2048 bytes).",
    )
    add_output_options(workflow_parser, formats=FORMAT_CHOICES)

    snapshot_parser = commands.add_parser(
        "snapshots",
        help="Inspect execution snapshots and unresolved operations.",
        description=(
            "Inspect execution snapshots and unresolved operations. Use the execution "
            "snapshot APIs to reconcile an unresolved operation before continuing the session."
        ),
    )
    snapshot_parser.add_argument("session_id")
    snapshot_parser.add_argument("--environment")
    snapshot_parser.add_argument("--limit", type=_positive_limit, default=25)
    _add_target_options(snapshot_parser, cloud=False)
    add_output_options(snapshot_parser, formats=FORMAT_CHOICES)

    interactions_parser = commands.add_parser(
        "interactions",
        help="Page response-scoped interaction summaries.",
        description=(
            "Page durable interaction summaries newest first. Use an interaction id "
            "with usage, tools, events, or transcript for response-scoped evidence."
        ),
    )
    interactions_parser.add_argument("session_id")
    _add_target_options(interactions_parser)
    interactions_parser.add_argument("--before-sequence", type=_positive_int)
    interactions_parser.add_argument("--limit", type=_positive_limit, default=50)
    add_output_options(interactions_parser, formats=FORMAT_CHOICES)

    usage_parser = commands.add_parser(
        "usage",
        help="Show per-model-call token usage.",
        description=(
            "Show bounded per-model-call token usage and pricing state. "
            "Use `--limit` and `--offset` to page results."
        ),
    )
    usage_parser.add_argument("session_id")
    _add_target_options(usage_parser)
    usage_parser.add_argument("--offset", type=_nonnegative_int, default=0)
    usage_parser.add_argument("--limit", type=_positive_limit, default=100)
    usage_parser.add_argument("--after-sequence", type=_nonnegative_int)
    usage_parser.add_argument("--before-sequence", type=_positive_int)
    usage_parser.add_argument("--interaction-id")
    _add_price_book_option(usage_parser)
    add_output_options(usage_parser, formats=FORMAT_CHOICES)

    tools_parser = commands.add_parser(
        "tools",
        help="Show paired durable tool calls.",
        description=(
            "Show paired durable tool calls without result bodies. "
            "Use event inspection for bounded payload metadata; for example, "
            "`cayu session events SESSION_ID --type tool.call.failed,tool.call.blocked "
            "--include-payload` shows why calls failed or were blocked."
        ),
    )
    tools_parser.add_argument("session_id")
    _add_target_options(tools_parser)
    tools_parser.add_argument("--offset", type=_nonnegative_int, default=0)
    tools_parser.add_argument("--limit", type=_positive_limit, default=100)
    tools_parser.add_argument("--after-sequence", type=_nonnegative_int)
    tools_parser.add_argument("--before-sequence", type=_positive_int)
    tools_parser.add_argument("--interaction-id")
    add_output_options(tools_parser, formats=FORMAT_CHOICES)

    events_parser = commands.add_parser(
        "events",
        help="Page durable session events.",
        description=(
            "Page durable session events with optional filters. "
            "Use `--include-payload` only for bounded payload previews."
        ),
    )
    events_parser.add_argument("session_id")
    _add_target_options(events_parser)
    events_parser.add_argument(
        "--type",
        action="extend",
        dest="event_types",
        default=[],
        type=_event_types_argument,
        metavar="EVENT_TYPE",
        help=(
            "Show only this event type, such as tool.call.failed or a custom.* type. "
            "Repeat --type or separate types with commas to show several."
        ),
    )
    events_parser.add_argument("--tool")
    events_parser.add_argument("--agent")
    events_parser.add_argument("--environment")
    events_parser.add_argument("--since", type=_datetime_argument)
    events_parser.add_argument("--until", type=_datetime_argument)
    events_parser.add_argument("--after-sequence", type=_nonnegative_int)
    events_parser.add_argument("--before-sequence", type=_positive_int)
    events_parser.add_argument("--interaction-id")
    events_parser.add_argument("--limit", type=_positive_limit, default=100)
    events_parser.add_argument(
        "--include-payload",
        nargs="?",
        const=2048,
        type=_payload_limit,
        metavar="BYTES",
        help=(
            "Include a bounded preview of each event's payload, such as a failed tool "
            "call's error (default 2048 bytes). Payloads are omitted without it."
        ),
    )
    add_output_options(events_parser, formats=FORMAT_CHOICES)

    transcript_parser = commands.add_parser(
        "transcript",
        help="Page bounded transcript metadata and previews.",
        description=(
            "Page bounded transcript metadata with redacted content by default. "
            "Use `--include-content` only for bounded, redacted previews."
        ),
    )
    transcript_parser.add_argument("session_id")
    _add_target_options(transcript_parser)
    transcript_parser.add_argument("--offset", type=_nonnegative_int, default=0)
    transcript_parser.add_argument("--limit", type=_positive_limit, default=100)
    transcript_parser.add_argument("--interaction-id")
    transcript_parser.add_argument(
        "--sizes",
        action="store_true",
        help="Include serialized size metadata for each content part.",
    )
    transcript_parser.add_argument(
        "--include-content",
        nargs="?",
        const=4096,
        type=_content_limit,
        metavar="BYTES",
        help="Include redacted serialized content, bounded per message.",
    )
    add_output_options(transcript_parser, formats=FORMAT_CHOICES)

    output_parser = commands.add_parser(
        "output",
        help="Print a finished session's structured output.",
        description=(
            "Print the structured output the session's latest interaction validated, or "
            "that of --interaction-id. Exits "
            f"{EXIT_NO_STRUCTURED_OUTPUT} when there is none and {EXIT_SESSION_NOT_FINISHED} "
            "when the session hasn't finished."
        ),
    )
    output_parser.add_argument("session_id")
    _add_target_options(output_parser)
    output_parser.add_argument(
        "--interaction-id",
        help="Read this interaction's output instead (see `cayu session interactions`).",
    )
    add_output_options(output_parser, formats=("json", "table"))

    close_parser = commands.add_parser(
        "close",
        help="End a session nobody will continue, or one a lost worker left running.",
        description=(
            "Close a pending or interrupted session nobody will continue, such as one "
            "waiting on a question or an approval it can no longer resume from, or a "
            "running session whose worker is provably gone (its execution lease lapsed, "
            "its process died, or every task of it finished), such as one a deploy cut "
            "off. The session fails with your reason and name recorded on its "
            "session.failed event, its pending approval or question leaves the inbox, and "
            "its transcript and events are kept. A running session whose owner is live or "
            "not known to be gone, one already ended, or one with unfinished tasks or "
            "child sessions is refused. With --application NAME it closes a Cayu "
            "Cloud Agent's session through the Agent's API (developer or admin). Without it, "
            "it opens the project's application in this process, as `cayu inbox` does, "
            "because closing settles the session's environment through that application; "
            "a bare --sqlite or --postgres store can't do that."
        ),
    )
    close_parser.add_argument("session_id")
    close_parser.add_argument(
        "--reason",
        required=True,
        help=f"Why, recorded with the session (at most {_CLOSE_REASON_MAX_CHARS} characters).",
    )
    close_target = close_parser.add_mutually_exclusive_group()
    close_target.add_argument(
        "--application",
        metavar="NAME",
        help="A Cayu Cloud Agent's id, name or slug, signed in like `cayu cloud` commands.",
    )
    close_target.add_argument(
        "--target",
        metavar="MODULE:FACTORY",
        help="Override project discovery with a module:factory target (local only).",
    )
    close_parser.add_argument(
        "--context",
        type=Path,
        metavar="PATH",
        help="Private Cayu Cloud context JSON for --application; defaults to `cayu cloud`'s.",
    )
    from cayu.cli._cloud_agent_api import add_wake_options

    add_wake_options(close_parser)
    close_parser.add_argument(
        "--as",
        dest="closed_by",
        metavar="SUBJECT",
        help=(
            "Who a local close is recorded as (default: local:<your user name>). "
            "A Cloud Agent records your login."
        ),
    )
    add_output_options(close_parser, formats=("json", "table"))

    resolution_parser = commands.add_parser(
        "resolve-provider-operation",
        help="Explicitly retry or fail unavailable provider work.",
        description=(
            "Send a run-epoch-fenced fallback_retry or fail disposition to the Cayu "
            "server. `cayu session show SESSION_ID` prints what it needs: --stage-id is "
            "model_stage.stage_id (also provider_operation.stage_id), --run-epoch is "
            "session.run_epoch, and provider_operation.allowed_resolutions lists the "
            "actions the operation accepts now. Authorization is read from "
            "CAYU_API_AUTHORIZATION by default and is never printed."
        ),
    )
    resolution_parser.add_argument("session_id")
    resolution_parser.add_argument("--stage-id", required=True)
    resolution_parser.add_argument("--run-epoch", required=True, type=_nonnegative_int)
    resolution_parser.add_argument(
        "--action",
        choices=("fallback_retry", "fail"),
        required=True,
    )
    resolution_parser.add_argument("--reason")
    resolution_parser.add_argument(
        "--metadata",
        type=_json_object_argument,
        default=None,
        metavar="JSON",
        help="Bounded JSON object recorded with the durable resolution.",
    )
    resolution_parser.add_argument(
        "--server-url",
        required=True,
        help="Cayu server root URL, including any mount prefix but excluding /api.",
    )
    resolution_parser.add_argument(
        "--authorization-env",
        default="CAYU_API_AUTHORIZATION",
        metavar="NAME",
        help="Environment variable containing the complete Authorization header value.",
    )
    resolution_parser.add_argument(
        "--mutation-id",
        help="Stable Cayu-Mutation-ID for correlating an ambiguous SSE reconnect.",
    )
    add_output_options(resolution_parser, formats=("json", "table"))


def run_session(args: argparse.Namespace) -> int:
    """Resolve a read-only target and dispatch one session-inspection command."""

    if args.session_command == "runtime-timing":
        from cayu.cli._runtime_timing import run_runtime_timing

        return run_runtime_timing(args)
    try:
        with output_destination(args.output):
            return _run_session(args)
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 1


def _run_session(args: argparse.Namespace) -> int:
    """Run after the optional output destination owns stdout."""

    if args.session_command == "start":
        from cayu.cli._session_start import run_session_start

        return run_session_start(args)
    if args.session_command == "close":
        try:
            return _close_session(args)
        except (ValueError, OSError, RuntimeError) as exc:
            _render_session_error(str(exc), args.output_format, code="SESSION_CLOSE_FAILED")
            return 1
    if args.session_command == "resolve-provider-operation":
        try:
            return _resolve_provider_operation(args)
        except (ValueError, OSError, RuntimeError) as exc:
            _render_session_error(_safe_error(str(exc), None), args.output_format)
            return 1
    dsn: str | None = None
    try:
        if getattr(args, "application", None) is not None:
            if args.sqlite is not None or args.postgres is not None:
                raise ValueError("--application can't be combined with --sqlite or --postgres.")
            return asyncio.run(_run_session_command(args, lambda: _open_cloud_reader(args)))
        if getattr(args, "context", None) is not None:
            raise ValueError("--context selects the Cayu Cloud for --application NAME.")
        if getattr(args, "server_url", None) is not None:
            return asyncio.run(_run_session_command(args, lambda: _open_server_reader(args)))
        target = resolve_session_store_target(
            sqlite=args.sqlite,
            postgres=args.postgres,
        )
        dsn = target.postgres_dsn
        return asyncio.run(
            _run_session_command(args, lambda: StoreSessionReader(_open_read_only_store(target)))
        )
    except SessionOutputUnavailable as exc:
        _render_session_error(str(exc), args.output_format, code=exc.code)
        return exc.exit_code
    except (SessionStoreTargetError, ValueError, OSError, RuntimeError) as exc:
        _render_session_error(_safe_error(str(exc), dsn), args.output_format)
        return 1
    except Exception as exc:
        # Driver and SQLite failures have backend-specific exception types. Keep
        # the CLI concise while preserving the existing DSN scrubbing contract.
        _render_session_error(_safe_error(str(exc), dsn), args.output_format)
        return 1


def _close_session(args: argparse.Namespace) -> int:
    reason = args.reason
    if not reason.strip():
        raise ValueError("--reason must not be blank.")
    if len(reason) > _CLOSE_REASON_MAX_CHARS:
        raise ValueError(f"--reason is longer than {_CLOSE_REASON_MAX_CHARS} characters.")
    if args.application is not None:
        if args.closed_by is not None:
            raise ValueError("--as applies to a local close; a Cloud Agent records your login.")
        answer = _close_cloud_session(args)
    else:
        if args.context is not None:
            raise ValueError("--context selects the Cayu Cloud for --application NAME.")
        answer = asyncio.run(_close_local_session(args))
    reported = answer.get("session")
    session: dict[str, Any] = reported if isinstance(reported, dict) else {}
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "closed": True,
        "where": "cloud" if args.application is not None else "local",
        "session_id": session.get("id", args.session_id),
        "status": session.get("status"),
        "actions": answer.get("actions", []),
        "message": answer.get("message"),
    }
    if args.output_format == "json":
        _render_detail("json", payload)
    else:
        print(
            f"Closed session {terminal_safe(str(payload['session_id']))}; "
            f"it is {terminal_safe(str(payload['status']))}."
        )
    return 0


def _close_cloud_session(args: argparse.Namespace) -> dict[str, Any]:
    from urllib.parse import quote

    from cayu.cli._cloud_agent_api import CayuApiError, CayuApiNotFound, open_cloud_agent_api

    try:
        api = open_cloud_agent_api(
            args.application,
            context=args.context,
            wake_timeout_seconds=args.wake_timeout_seconds,
            offers_wake_option=True,
        )
    except CayuApiError as exc:
        raise RuntimeError(str(exc)) from None
    try:
        answer = api.mutate(
            f"/sessions/{quote(args.session_id, safe='')}/close",
            {"reason": args.reason},
            resource="session close",
        )
    except CayuApiNotFound:
        raise ValueError(f"Session not found: {args.session_id}") from None
    except CayuApiError as exc:
        if exc.category == "request_rejected" and exc.detail:
            raise ValueError(exc.detail) from None
        if exc.category == "outcome_unknown":
            raise RuntimeError(
                f"{exc} Repeating the same close is safe: `cayu session show "
                f"{args.session_id} --application {args.application}` shows whether it closed."
            ) from None
        raise RuntimeError(str(exc)) from None
    finally:
        api.close()
    if not isinstance(answer, dict):
        raise RuntimeError("The Agent answered the close without a JSON object.")
    return answer


async def _close_local_session(args: argparse.Namespace) -> dict[str, Any]:
    import getpass

    from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
    from cayu.cli.project import (
        ProjectError,
        build_project_app,
        close_project_app,
        project_context,
        resolve_project,
    )

    subject = args.closed_by
    if subject is None:
        try:
            subject = f"local:{getpass.getuser()}"
        except (KeyError, OSError):
            raise ValueError("Can't tell your user name; name yourself with --as.") from None
    actor = ResolutionActor(subject=subject, source=ResolutionActorSource.REQUEST)
    try:
        project = resolve_project(args.target, command="cayu session close")
    except ProjectError as exc:
        raise ValueError(str(exc)) from None
    with project_context(project.root):
        try:
            app = build_project_app(project.target, command="Session close")
        except ProjectError as exc:
            raise ValueError(str(exc)) from None
        try:
            result = await app.close_session(args.session_id, reason=args.reason, closed_by=actor)
        except KeyError:
            raise ValueError(f"Session not found: {args.session_id}") from None
        finally:
            await close_project_app(app)
    return {
        "session": {"id": args.session_id, "status": result.status.value},
        "actions": [action.value for action in result.actions],
        "message": result.message,
    }


def _resolve_provider_operation(args: argparse.Namespace) -> int:
    endpoint = _provider_operation_resolution_endpoint(args.server_url)
    authorization = os.environ.get(args.authorization_env)
    if authorization is not None and (
        not authorization.strip()
        or authorization != authorization.strip()
        or "\r" in authorization
        or "\n" in authorization
    ):
        raise ValueError(f"{args.authorization_env} contains an invalid Authorization value.")
    headers = {
        "Accept": "text/event-stream",
        "Cayu-Mutation-ID": args.mutation_id or f"cli-provider-resolution-{uuid4().hex}",
    }
    if authorization is not None:
        headers["Authorization"] = authorization
    payload = {
        "session_id": args.session_id,
        "stage_id": args.stage_id,
        "expected_run_epoch": args.run_epoch,
        "action": args.action,
        "reason": args.reason,
        "metadata": args.metadata or {},
    }
    try:
        with (
            httpx.Client(follow_redirects=False, timeout=30.0) as client,
            client.stream(
                "POST",
                endpoint,
                json=payload,
                headers=headers,
            ) as response,
        ):
            if not 200 <= response.status_code < 300:
                detail = _safe_server_error_detail(
                    response,
                    authorization=authorization,
                )
                suffix = f": {detail}" if detail is not None else "."
                raise RuntimeError(f"Cayu server returned HTTP {response.status_code}{suffix}")
    except httpx.RequestError:
        raise RuntimeError("Cayu server is unavailable.") from None
    _render_detail(
        args.output_format,
        {
            "schema_version": CLI_SCHEMA_VERSION,
            "accepted": True,
            "session_id": args.session_id,
            "stage_id": args.stage_id,
            "run_epoch": args.run_epoch,
            "action": args.action,
            "mutation_id": headers["Cayu-Mutation-ID"],
        },
    )
    return 0


def _provider_operation_resolution_endpoint(server_url: str) -> str:
    parsed = urlsplit(server_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("A canonical HTTP(S) Cayu server URL is required.")
    hostname = parsed.hostname
    if parsed.scheme == "http" and not _is_loopback_host(hostname):
        raise ValueError("Cayu server URL must use HTTPS outside loopback.")
    path = parsed.path.rstrip("/") + "/api/provider-operations/resolve"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _safe_server_error_detail(
    response: httpx.Response,
    *,
    authorization: str | None,
) -> str | None:
    body = bytearray()
    for chunk in response.iter_bytes(chunk_size=1024):
        if len(body) + len(chunk) > _MAX_SERVER_ERROR_BODY_BYTES:
            return None
        body.extend(chunk)
    try:
        value = json.loads(body)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if type(value) is not dict:
        return None
    detail = value.get("detail")
    if type(detail) is not str or not detail:
        return None
    try:
        detail_size = len(detail.encode("utf-8"))
    except UnicodeEncodeError:
        return None
    if detail_size > _MAX_SERVER_ERROR_DETAIL_SOURCE_BYTES:
        return None
    # Keep tabs and line breaks; neutralize every other control or format
    # character: C0/C1 controls (including the 8-bit CSI U+009B), DEL, and bidi
    # overrides that could make the shown text differ from what was sent.
    detail = "".join(
        " "
        if character not in "\t\n" and unicodedata.category(character) in {"Cc", "Cf"}
        else character
        for character in detail
    )
    if authorization is not None:
        detail = detail.replace(authorization, "[REDACTED]")
    redacted = _redact_sensitive(detail)
    if type(redacted) is not str:
        return None
    # Redact the complete text first, then bound it.
    encoded = redacted.encode("utf-8")
    if len(encoded) <= _MAX_SERVER_ERROR_DETAIL_BYTES:
        return redacted
    marker = "...[truncated]"
    head = encoded[: _MAX_SERVER_ERROR_DETAIL_BYTES - len(marker)].decode("utf-8", "ignore")
    return head + marker


def _render_session_error(
    message: str, output_format: str, *, code: str = "SESSION_INSPECTION_FAILED"
) -> None:
    if output_format in {"json", "jsonl"}:
        print(
            json.dumps(
                {
                    "schema_version": CLI_SCHEMA_VERSION,
                    "error": {"code": code, "message": message},
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    print(f"error: {terminal_safe(message)}", file=sys.stderr)


def _open_cloud_reader(args: argparse.Namespace) -> SessionReader:
    from cayu.cli._agent_session_reader import AgentSessionReader
    from cayu.cli._cloud_agent_api import open_cloud_agent_api

    return AgentSessionReader(
        open_cloud_agent_api(
            args.application,
            context=args.context,
            wake_timeout_seconds=args.wake_timeout_seconds,
            offers_wake_option=True,
        )
    )


def _open_server_reader(args: argparse.Namespace) -> SessionReader:
    from cayu.cli._agent_session_reader import AgentSessionReader
    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.cli.diagnostics import read_authorization, request_cost_endpoint

    # Refuses credentials, a query or plain HTTP outside loopback in the URL.
    request_cost_endpoint(args.server_url)
    return AgentSessionReader(
        CayuApi(args.server_url, authorization=read_authorization(args.authorization_env))
    )


async def _run_session_command(
    args: argparse.Namespace,
    open_reader: Callable[[], SessionReader],
) -> int:
    reader = open_reader()
    try:
        if args.session_command == "list":
            return await _list_sessions(args, reader)
        if args.session_command == "show":
            return await _show_session(args, reader)
        if args.session_command == "rollup":
            return await _session_rollup(args, reader)
        if args.session_command == "workflow":
            return await _session_workflow(args, reader)
        if args.session_command == "snapshots":
            from cayu.environments.snapshot_lifecycle import inspect_execution_snapshots

            if not isinstance(reader, StoreSessionReader):
                raise ValueError("Execution snapshots are read from a local session store.")
            records = await inspect_execution_snapshots(
                reader.store, args.session_id, environment_name=args.environment, limit=args.limit
            )
            _render_detail(
                args.output_format,
                {
                    "schema_version": CLI_SCHEMA_VERSION,
                    "execution_snapshots": [record.model_dump(mode="json") for record in records],
                },
            )
            return 0
        if args.session_command == "interactions":
            return await _session_interactions(args, reader)
        if args.session_command == "usage":
            return await _session_usage(args, reader)
        if args.session_command == "tools":
            return await _session_tools(args, reader)
        if args.session_command == "events":
            return await _session_events(args, reader)
        if args.session_command == "transcript":
            return await _session_transcript(args, reader)
        if args.session_command == "output":
            return await _session_output(args, reader)
        return 1
    finally:
        await reader.close()


def _open_read_only_store(target: SessionStoreTarget) -> SessionStore:
    if target.backend is SessionStoreBackend.SQLITE:
        if target.sqlite_path is None:
            raise AssertionError("Resolved SQLite target has no path.")
        return SQLiteSessionStore(
            target.sqlite_path,
            schema_mode=schema.SchemaMode.VALIDATE,
            read_only=True,
            public_authority_alias_codec=public_authority_alias_codec_from_environment(),
        )
    if target.postgres_dsn is None:
        raise AssertionError("Resolved Postgres target has no DSN.")
    from cayu.storage import PostgresSessionStore

    return PostgresSessionStore(
        target.postgres_dsn,
        schema_mode=schema.SchemaMode.VALIDATE,
        read_only=True,
        public_authority_alias_codec=public_authority_alias_codec_from_environment(),
    )


async def _list_sessions(args: argparse.Namespace, reader: SessionReader) -> int:
    labels = _parse_labels(args.label)
    local = isinstance(reader, StoreSessionReader)
    if args.price_book is not None and not (args.overview and local):
        raise ValueError("--price-book prices --overview costs of a local store.")
    query = SessionQuery(
        status=None if args.status is None else SessionStatus(args.status),
        agent_name=args.agent,
        environment_name=args.environment,
        task_id=args.task,
        labels=labels,
        created_since=args.since,
        created_before=args.until,
        limit=args.limit,
        offset=args.offset,
        cursor=args.cursor,
        include_total_count=True,
        order_by=SessionOrder.LAST_ACTIVITY_AT_DESC,
    )
    if args.owner_lost:
        return await _list_owner_lost_sessions(args, reader, query)
    result = await reader.list_sessions(
        query,
        overview=args.overview,
        pricing=_price_book(args) if args.overview and local else None,
    )
    sessions = result.rows
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "sessions": sessions,
        "next_cursor": result.next_cursor,
        "has_more": result.next_cursor is not None,
        "total_count": result.total_count,
    }
    if args.overview:
        _render_collection(
            args.output_format,
            payload,
            sessions if args.output_format == "jsonl" else [_overview_cells(s) for s in sessions],
            headers=_OVERVIEW_HEADERS,
        )
        return 0
    _render_collection(
        args.output_format,
        payload,
        sessions,
        headers=(
            "id",
            "status",
            "agent",
            "provider",
            "model",
            "environment",
            "created_at",
            "updated_at",
            "last_activity_at",
            "run_epoch",
        ),
    )
    return 0


async def _list_owner_lost_sessions(
    args: argparse.Namespace, reader: SessionReader, query: SessionQuery
) -> int:
    if args.overview or args.offset or args.cursor is not None:
        raise ValueError("--owner-lost can't be combined with --overview, --offset or --cursor.")
    page = await owner_lost_sessions(
        reader,
        query,
        statuses=OWNER_LOST_STATUSES if query.status is None else (query.status,),
    )
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "sessions": page.rows,
        "next_cursor": None,
        "has_more": page.has_more,
        "total_count": None,
        "scanned": page.scanned,
    }
    _render_collection(
        args.output_format,
        payload,
        page.rows if args.output_format == "jsonl" else [_owner_lost_cells(r) for r in page.rows],
        headers=_OWNER_LOST_HEADERS,
    )
    return 0


_OWNER_LOST_HEADERS = (
    "id",
    "status",
    "agent",
    "owner_kind",
    "owner_id",
    "heartbeat_at",
    "lease_expires_at",
    "last_progress_kind",
    "last_activity_at",
)


def _owner_lost_cells(row: dict[str, Any]) -> dict[str, Any]:
    execution = row["execution"]
    return {
        **{key: row.get(key) for key in ("id", "status", "agent", "last_activity_at")},
        **{
            key: execution.get(key)
            for key in (
                "owner_kind",
                "owner_id",
                "heartbeat_at",
                "lease_expires_at",
                "last_progress_kind",
            )
        },
    }


_OVERVIEW_HEADERS = (
    "id",
    "state",
    "agent",
    "title",
    "first_message",
    "model_steps",
    "tool_calls",
    "tokens",
    "cost",
    "last_activity_at",
)
_OVERVIEW_TEXT_CELL_CHARS = 60


def _overview_cells(row: dict[str, Any]) -> dict[str, Any]:
    """One `session list --overview --table` line: short, single-line cells."""

    overview = row.get("overview")
    overview = overview if isinstance(overview, dict) else {}
    tokens = overview.get("tokens")
    tokens = tokens if isinstance(tokens, dict) else {}
    cost = overview.get("cost")
    first = overview.get("first_message")
    if first is None and overview.get("first_message_unavailable") is True:
        first = "[unavailable]"
    return {
        "id": row.get("id"),
        "state": overview.get("state"),
        "agent": row.get("agent"),
        "title": _one_line(overview.get("title")),
        "first_message": _one_line(first),
        "model_steps": overview.get("model_steps"),
        "tool_calls": overview.get("tool_calls"),
        "tokens": None if not tokens else f"{tokens.get('input')} in / {tokens.get('output')} out",
        "cost": None
        if not isinstance(cost, dict)
        else f"{cost.get('amount')} {cost.get('currency')}"
        + ("" if cost.get("complete") is True else " (partial)"),
        "last_activity_at": overview.get("last_activity_at") or row.get("last_activity_at"),
    }


def _one_line(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if len(text) > _OVERVIEW_TEXT_CELL_CHARS:
        text = text[: _OVERVIEW_TEXT_CELL_CHARS - 1] + "…"
    return text


# Sessions per rollup page: an Agent reads each page within one request.
_ROLLUP_PAGE_SESSIONS = {"local": 200, "cloud": 25}


async def _session_rollup(args: argparse.Namespace, reader: SessionReader) -> int:
    page_size = _ROLLUP_PAGE_SESSIONS[
        "local" if isinstance(reader, StoreSessionReader) else "cloud"
    ]
    pricing = _price_book(args)
    query = SessionRollupQuery(
        since=args.since,
        until=args.until,
        group_by=args.group_by,
        agent_name=args.agent,
        environment_name=args.environment,
        status=None if args.status is None else SessionStatus(args.status),
        labels=_parse_labels(args.label),
        causal_budget_id=args.causal_budget,
        root_session_id=args.root,
        task_id=args.task,
        cursor=args.cursor,
        max_sessions=min(page_size, args.max_sessions),
        pricing=pricing,
    )
    groups: list[RollupGroup] = []
    sessions_read = 0
    while True:
        page = await reader.rollup_page(query)
        groups = merge_groups([*groups, *page.groups])
        sessions_read += page.sessions_read
        cursor = page.next_cursor
        remaining = args.max_sessions - sessions_read
        if cursor is None or remaining <= 0 or not page.sessions_read:
            break
        query = query.model_copy(
            update={"cursor": cursor, "max_sessions": min(page_size, remaining)}
        )
    unpriced_by_agent = getattr(reader, "rollup_pricing_unsupported", False)
    rows = [_rollup_row(group, priced=not unpriced_by_agent) for group in groups]
    if unpriced_by_agent and args.output_format != "json":
        print(
            "warning: this Agent's Cayu can't price rollups, so costs are left out; "
            "redeploy it with a newer cayu release to see them.",
            file=sys.stderr,
        )
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "group_by": args.group_by,
        "since": None if args.since is None else args.since.isoformat(),
        "until": None if args.until is None else args.until.isoformat(),
        "sessions": sessions_read,
        "complete": cursor is None,
        "next_cursor": cursor,
        "groups": rows,
        "price_book": None
        if unpriced_by_agent
        else _price_book_source(
            args, {"version": pricing.price_book_version, "generated_at": pricing.generated_at}
        ),
    }
    _render_rollup(args.output_format, payload, rows, failure_kinds=args.failure_kinds)
    return 0


def _rollup_row(group: RollupGroup, *, priced: bool = True) -> dict[str, Any]:
    """One group's row; when the Agent priced nothing, every model call counts as unpriced."""

    return {
        "key": group.key,
        "sessions": group.sessions,
        "statuses": group.statuses,
        "model_calls": group.model_calls,
        "model_calls_with_usage": group.model_calls_with_usage,
        "model_errors": group.model_errors,
        "tool_calls": group.tool_calls,
        "tool_failures": group.tool_failures,
        "usage": aggregate_usage_cli_payload(group.usage),
        # Without pricing nothing says whether a group cost anything, so both are unknown.
        "estimated_cost": group.cost if priced else None,
        "unpriced_model_calls": group.unpriced_model_calls if priced else None,
        "durations": group.durations.model_dump(mode="json"),
        "failure_kinds": [item.model_dump(mode="json") for item in top_failure_kinds(group)],
    }


def _render_rollup(
    output: str, payload: dict[str, Any], rows: list[dict[str, Any]], *, failure_kinds: int
) -> None:
    if output in {"json", "jsonl"}:
        if output == "json":
            print(json.dumps(_redact_sensitive(payload), ensure_ascii=False, sort_keys=True))
            return
        for row in rows:
            record = {"record_type": "group", **row, "schema_version": CLI_SCHEMA_VERSION}
            print(json.dumps(_redact_sensitive(record), ensure_ascii=False, sort_keys=True))
        # The summary says whether the groups cover the whole window and how to continue.
        summary = {"record_type": "summary"} | {k: v for k, v in payload.items() if k != "groups"}
        print(json.dumps(_redact_sensitive(summary), ensure_ascii=False, sort_keys=True))
        return
    table = [
        {
            "group": row["key"],
            "sessions": row["sessions"],
            "model_calls": row["model_calls"],
            "tool_calls": row["tool_calls"],
            "tool_failures": row["tool_failures"],
            "input_tokens": row["usage"]["input_tokens"],
            "output_tokens": row["usage"]["output_tokens"],
            "cache_read_tokens": row["usage"]["cache_read_tokens"],
            "cost": _cost_total_text(row["estimated_cost"], row["unpriced_model_calls"]),
            "unpriced": row["unpriced_model_calls"],
            **{
                name: _duration_text(row["durations"][f"{name}_ms"])
                for name in ("active", "model", "tool", "setup", "teardown", "gap")
            },
        }
        for row in rows
    ]
    _print_table(
        (
            "group",
            "sessions",
            "model_calls",
            "tool_calls",
            "tool_failures",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cost",
            "unpriced",
            "active",
            "model",
            "tool",
            "setup",
            "teardown",
            "gap",
        ),
        cast("list[dict[str, Any]]", _redact_sensitive(table)),
    )
    kinds = [
        {"group": row["key"], **kind}
        for row in rows
        for kind in row["failure_kinds"][:failure_kinds]
    ]
    if kinds:
        print()
        print("Tool failures by kind")
        _print_table(("group", "tool", "kind", "count"), _redact_sensitive(kinds))
    print()
    state = (
        "complete"
        if payload["complete"]
        else f"stopped early; continue with --cursor {payload['next_cursor']}"
    )
    print(f"{payload['sessions']} sessions ({state}).")


def _optional_duration_text(milliseconds: object) -> str | None:
    """A duration, or nothing when it wasn't recorded (an older Agent or session)."""

    return _duration_text(milliseconds) if type(milliseconds) is int else None


def _duration_text(milliseconds: int) -> str:
    seconds, ms = divmod(milliseconds, 1000)
    if not seconds:
        return f"{ms}ms"
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}.{ms // 100}s"


async def _show_session(args: argparse.Namespace, reader: SessionReader) -> int:
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        **await reader.overview(args.session_id),
    }
    if args.output_format in {"json", "jsonl"}:
        _render_detail(args.output_format, payload)
        return 0
    timing = payload.pop("timing", None)
    _render_detail(args.output_format, payload)
    lag = event_loop_lag_text(payload.get("event_loop_lag"))
    if lag is not None:
        print("\n" + terminal_safe(lag))
    print()
    _print_timing(cast("dict[str, Any] | None", _redact_sensitive(timing)))
    return 0


_TIMING_PARTS = (
    ("setup", "environment setup: allocate, copy in, setup commands"),
    ("model", "model calls"),
    ("tool", "tool calls"),
    ("teardown", "environment teardown: copy back, release"),
    ("waiting", "waiting for a person"),
    ("other", "runtime work inside turns"),
    ("idle", "outside turns"),
)


def _print_timing(timing: dict[str, Any] | None) -> None:
    """Print where the session's time went and each environment operation's steps."""

    if timing is None:
        print(
            "Time: this Agent's Cayu doesn't report session timing; redeploy it with a newer "
            "cayu release to see it."
        )
        return
    print(f"Time ({_duration_text(timing['elapsed_ms'])} elapsed)")
    calls = {
        name: f"{count} call" + ("" if count == 1 else "s")
        for name, count in (("model", timing["model_calls"]), ("tool", timing["tool_calls"]))
    }
    first = timing.get("first_model_call_ms")
    if type(first) is int:
        calls["model"] += f", first after {_duration_text(first)}"
    single = _call_durations_text(timing.get("per_model_call"))
    if single is not None:
        calls["model"] += f", {single}"
    _print_timing_parts(timing, calls)
    if type(first) is int and type(timing.get("gap_ms")) is int:
        print(f"Before the first model call: {_duration_text(first)}. {_gap_text(timing)}")
    if timing.get("truncated"):
        print("The session has more events than one breakdown reads; it covers the first.")
    _print_tool_durations(timing)
    operations = timing.get("environment") or []
    if not operations:
        return
    print()
    print("Environment operations")
    rows: list[dict[str, Any]] = []
    for operation in operations:
        rows.append(
            {
                "step": f"{operation['kind']} ({operation.get('environment') or '-'})",
                "time": _optional_duration_text(operation.get("duration_ms")),
                "outcome": operation["outcome"],
                "work": None,
            }
        )
        for phase in operation.get("phases") or []:
            rows.append(
                {
                    "step": "  " * (1 + phase.get("depth", 0)) + phase["phase"],
                    "time": _duration_text(phase["duration_ms"]),
                    "outcome": phase["outcome"],
                    "work": _phase_work_text(phase),
                }
            )
    _print_table(("step", "time", "outcome", "work"), rows)
    if timing.get("environment_truncated"):
        print("More environment operations were left out.")


def print_timing_totals(totals: dict[str, Any]) -> None:
    """Print `SessionTimingTotals`: where the time of several sessions went, added up."""

    sessions = totals["sessions"]
    print(
        f"Time over {sessions} session{'' if sessions == 1 else 's'} "
        f"({_duration_text(totals['elapsed_ms'])} elapsed, added up)"
    )
    calls = {
        name: f"{count} call" + ("" if count == 1 else "s")
        for name, count in (("model", totals["model_calls"]), ("tool", totals["tool_calls"]))
    }
    single = _call_durations_text(totals.get("per_model_call"))
    if single is not None:
        calls["model"] += f", {single}"
    _print_timing_parts(totals, calls)
    print(
        f"Before each session's first model call: "
        f"{_duration_text(totals['first_model_call_ms'])}. {_gap_text(totals)}"
    )
    if totals.get("truncated"):
        print("Some sessions have more events than one breakdown reads; they cover the first.")
    _print_tool_durations(totals)


def _print_timing_parts(timing: dict[str, Any], notes: dict[str, str]) -> None:
    """The parts that add up to the elapsed time, with their shares of it."""

    _print_table(
        ("part", "time", "share", "notes"),
        [
            {
                "part": name,
                "time": _duration_text(timing[f"{name}_ms"]),
                "share": _share_text(timing[f"{name}_ms"], timing["elapsed_ms"]),
                "notes": notes.get(name, meaning),
            }
            for name, meaning in _TIMING_PARTS
        ],
    )


def _gap_text(timing: dict[str, Any]) -> str:
    return (
        f"After it, {_duration_text(timing['gap_ms'])} went to neither model nor tool calls "
        "(runtime work between steps, setup, teardown, waits and idle time)."
    )


def _call_durations_text(calls: object) -> str | None:
    """``27.0s average, longest 1m02s``, or nothing for no finished call."""

    if not isinstance(calls, dict):
        return None
    data = cast("dict[str, Any]", calls)
    average, longest = data.get("average_ms"), data.get("max_ms")
    if type(average) is not int or type(longest) is not int:
        return None
    return f"{_duration_text(average)} average, longest {_duration_text(longest)}"


def _print_tool_durations(timing: dict[str, Any]) -> None:
    tools = timing.get("per_tool") or []
    if not tools:
        return
    print()
    print("Tool calls by tool")
    _print_table(
        ("tool", "calls", "total", "average", "longest"),
        [
            {
                "tool": tool["tool"],
                "calls": tool["calls"],
                "total": _duration_text(tool["total_ms"]),
                "average": _optional_duration_text(tool.get("average_ms")),
                "longest": _optional_duration_text(tool.get("max_ms")),
            }
            for tool in tools
        ],
    )
    if timing.get("per_tool_truncated"):
        print("More tools were left out.")


def _share_text(part_ms: int, elapsed_ms: int) -> str:
    return f"{round(100 * part_ms / elapsed_ms)}%" if elapsed_ms else "-"


def _phase_work_text(phase: dict[str, Any]) -> str | None:
    work = []
    if type(phase.get("items")) is int:
        work.append(f"{phase['items']} items")
    if type(phase.get("bytes")) is int:
        work.append(f"{phase['bytes']} bytes")
    return ", ".join(work) or None


async def _session_workflow(args: argparse.Namespace, reader: SessionReader) -> int:
    pricing = _price_book(args)
    run = await read_workflow_run(
        reader,
        args.session_id,
        pricing=pricing,
        max_sessions=args.max_sessions,
        max_events=args.max_events,
    )
    priced = run.totals.unpriced_calls is not None
    # A gated-loop item usually names the child of a step it ran, whose row has the usage;
    # an item whose `do` ran a child without a journaled step carries that child's usage.
    step_children = {step.child_session_id for step in run.steps if step.kind == "step"}
    steps: list[dict[str, Any]] = []
    for step in run.steps:
        node = None if step.child_session_id is None else run.nodes.get(step.child_session_id)
        started_at, ended_at = step_times(step, node)
        usage = (
            run.subtree_usage.get(step.child_session_id)
            if step.child_session_id is not None
            and (step.kind == "step" or step.child_session_id not in step_children)
            else None
        )
        steps.append(
            {
                "step_id": step.step_id,
                "kind": step.kind,
                "agent": step.agent or (None if node is None else node.agent),
                "child_session_id": step.child_session_id,
                "status": None if node is None else node.status,
                "journal": "completed" if step.completed_at is not None else "started",
                "loop_name": step.loop_name,
                "item_key": step.item_key,
                "outcome": step.outcome,
                "has_output": step.has_output,
                "exit_code": step.exit_code,
                "started_at": started_at,
                "ended_at": ended_at,
                **_workflow_usage_fields(usage, priced=priced),
            }
        )
    for child_id in run.unlinked_children:
        node = run.nodes[child_id]
        steps.append(
            {
                "step_id": None,
                "kind": "child",
                "agent": node.agent,
                "child_session_id": child_id,
                "status": node.status,
                "journal": None,
                "loop_name": None,
                "item_key": None,
                "outcome": None,
                "has_output": None,
                "exit_code": None,
                "started_at": node.created_at,
                "ended_at": session_ended_at(node),
                **_workflow_usage_fields(run.subtree_usage.get(child_id), priced=priced),
            }
        )
    custom_events = [
        _event_row(record, payload_limit=args.payload_bytes) for record in run.custom_events
    ]
    session = run.session
    complete = run.sessions_complete and run.events_complete
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "workflow": {
            "session_id": args.session_id,
            "name": run.name,
            "agent": session.get("agent"),
            "status": run_status(run),
            # The journal's anchor session; the shipped journal marks it completed up front.
            "session_status": session.get("status"),
            "parent_session_id": session.get("parent_session_id"),
            "causal_budget_id": session.get("causal_budget_id"),
            "started_at": None if run.started_at is None else run.started_at.isoformat(),
            "completed_at": None if run.completed_at is None else run.completed_at.isoformat(),
        },
        "steps": steps,
        "custom_events": custom_events,
        "totals": {
            "statuses": dict(sorted(run.totals.statuses.items())),
            **_workflow_usage_fields(run.totals, priced=priced),
        },
        "step_ids_withheld": run.step_ids_withheld,
        "complete": complete,
        "sessions_complete": run.sessions_complete,
        "events_complete": run.events_complete,
        "price_book": _price_book_source(
            args, {"version": pricing.price_book_version, "generated_at": pricing.generated_at}
        )
        if priced
        else None,
    }
    _render_workflow(args.output_format, payload)
    return 0


def _workflow_usage_fields(usage: WorkflowSessionUsage | None, *, priced: bool) -> dict[str, Any]:
    if usage is None:
        # A gated-loop item whose child a step row covers, or a child that wasn't found.
        return dict.fromkeys(
            (
                "sessions",
                "model_calls",
                "model_calls_with_usage",
                "tool_calls",
                "usage",
                "estimated_cost",
                "unpriced_model_calls",
            )
        )
    return {
        "sessions": usage.sessions,
        "model_calls": usage.model_calls,
        "model_calls_with_usage": usage.model_calls_with_usage,
        "tool_calls": usage.tool_calls,
        "usage": aggregate_usage_cli_payload(usage.usage),
        "estimated_cost": usage.estimated_cost(priced=priced),
        "unpriced_model_calls": usage.unpriced_calls if priced else None,
    }


def _render_workflow(output: str, payload: dict[str, Any]) -> None:
    safe = cast("dict[str, Any]", _redact_sensitive(payload))
    if output == "json":
        print(json.dumps(safe, ensure_ascii=False, sort_keys=True))
        return
    if output == "jsonl":
        for row in safe["steps"]:
            record = {"record_type": "step", **row, "schema_version": CLI_SCHEMA_VERSION}
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))
        for row in safe["custom_events"]:
            record = {"record_type": "custom_event", **row, "schema_version": CLI_SCHEMA_VERSION}
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))
        summary = {"record_type": "summary"} | {
            key: value for key, value in safe.items() if key not in {"steps", "custom_events"}
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return
    workflow = safe["workflow"]
    window = " to ".join(
        value for value in (workflow["started_at"], workflow["completed_at"]) if value
    )
    print(
        terminal_safe(
            f"Workflow {workflow['name'] or '-'}, session {workflow['session_id']} "
            f"({workflow['status']})" + (f", {window}" if window else "")
        )
    )
    if safe["steps"]:
        print()
        _print_table(
            (
                "step",
                "agent",
                "child_session",
                "status",
                "started_at",
                "ended_at",
                "sessions",
                "model_calls",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "cost",
            ),
            [_workflow_table_row(row) for row in safe["steps"]],
        )
    if safe["custom_events"]:
        print()
        print("Custom events")
        _print_table(
            ("sequence", "timestamp", "type", "agent", "payload_preview"),
            safe["custom_events"],
        )
    totals = safe["totals"]
    usage = totals["usage"]
    statuses = ", ".join(f"{status} {count}" for status, count in totals["statuses"].items())
    print()
    print(
        terminal_safe(
            f"Totals: {totals['sessions']} sessions ({statuses or 'none'}), "
            f"{totals['model_calls']} model calls, {usage['input_tokens']} input, "
            f"{usage['output_tokens']} output and {usage['reasoning_tokens']} reasoning tokens, "
            f"cost {_cost_total_text(totals['estimated_cost'], totals['unpriced_model_calls'])}."
        )
    )
    if safe["step_ids_withheld"]:
        print(
            "Some workflow identifiers are withheld or redacted; available child sessions "
            "are listed where steps cannot be identified. Redeploy older Agents to expose "
            "step names; secret-redacted names remain unavailable."
        )
    if not safe["complete"]:
        print(
            "Incomplete: the run has more sessions or events than were read; raise "
            "--max-sessions or --max-events."
        )


def _workflow_table_row(row: dict[str, Any]) -> dict[str, Any]:
    usage = row["usage"] or {}
    step = row["step_id"] if row["step_id"] is not None else "(child)"
    if row["loop_name"] is not None and row["item_key"] is not None:
        step = f"{row['loop_name']}/{row['item_key']}"
    if row["outcome"]:
        step = f"{step} [{row['outcome']}]"
    return {
        "step": step,
        "agent": row["agent"],
        "child_session": row["child_session_id"],
        "status": row["status"] if row["status"] is not None else row["journal"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "sessions": row["sessions"],
        "model_calls": row["model_calls"],
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "cost": None
        if row["usage"] is None
        else _cost_total_text(row["estimated_cost"], row["unpriced_model_calls"]),
    }


async def _session_interactions(args: argparse.Namespace, reader: SessionReader) -> int:
    page = await reader.latest_interactions(
        args.session_id,
        before_sequence=args.before_sequence,
        limit=args.limit,
    )
    rows = page.rows
    has_more = page.has_more
    next_sequence = page.next_sequence
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "session_id": args.session_id,
        "interactions": rows,
        "next_sequence": next_sequence,
        "has_more": has_more,
    }
    _render_collection(
        args.output_format,
        payload,
        rows,
        headers=(
            "interaction_id",
            "status",
            "started_at",
            "completed_at",
            "active_duration_ms",
            "wall_duration_ms",
            "model_step_count",
            "tool_call_count",
            "source_transcript_start",
            "source_transcript_end",
            "result_transcript_start",
            "result_transcript_end",
            "pending_action_kind",
            "updated_at",
        ),
    )
    return 0


async def _session_usage(args: argparse.Namespace, reader: SessionReader) -> int:
    await reader.require_session(args.session_id)
    result = await reader.model_calls(
        args.session_id,
        _inspection_window(args),
        offset=args.offset,
        limit=args.limit,
        pricing=_price_book(args),
    )
    aggregate = result["aggregate"]
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "session_id": args.session_id,
        "interaction_id": args.interaction_id,
        "calls": result["calls"],
        "offset": result["offset"],
        "next_offset": result["next_offset"],
        "has_more": result["has_more"],
        "total_calls": result["total_calls"],
        "event_window": {
            "after_sequence": args.after_sequence,
            "before_sequence": args.before_sequence,
        },
        "unmatched_ledger": result["unmatched_ledger"],
        "unmatched_ledger_total": result["unmatched_ledger_total"],
        "unmatched_ledger_next_offset": result["unmatched_ledger_next_offset"],
        "unmatched_ledger_has_more": result["unmatched_ledger_has_more"],
        "aggregate": {
            "model_calls": aggregate["model_calls"],
            "model_calls_with_usage": aggregate["model_calls_with_usage"],
            "tool_calls": aggregate["tool_calls"],
            **aggregate_usage_cli_payload(aggregate["usage"]),
            "estimated_cost": aggregate.get("estimated_cost"),
            "unpriced_calls": aggregate.get("unpriced_calls"),
        },
        "price_book": _price_book_source(args, result.get("price_book")),
    }
    _render_usage(
        args.output_format,
        payload,
        result["calls"],
        result["unmatched_ledger"],
    )
    return 0


def _add_price_book_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--price-book",
        "--price-table",
        dest="price_book",
        type=Path,
        metavar="PATH",
        help=(
            "Price model calls with this price book (JSON written by "
            "cayu.budgets.dump_price_book) instead of Cayu's bundled one."
        ),
    )


def _price_book(args: argparse.Namespace) -> PriceBook:
    if args.price_book is None:
        return default_price_book()
    try:
        return load_price_book(args.price_book)
    except OSError as exc:
        raise ValueError(
            f"Could not read the price book {args.price_book}: {exc.strerror}."
        ) from None
    except ValueError:
        raise ValueError(f"{args.price_book} is not a valid Cayu price book.") from None


def _price_book_source(
    args: argparse.Namespace, identity: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Which price book priced the calls, or None when the source couldn't price them."""

    if not isinstance(identity, dict):
        return None
    return {
        "source": "bundled" if args.price_book is None else "custom",
        "version": identity.get("version"),
        "generated_at": identity.get("generated_at"),
    }


def _cost_text(call: dict[str, Any]) -> str | None:
    if call.get("cost_state") == "priced":
        return f"{call['cost']} {call['cost_currency']}"
    return call.get("cost_state")


def _cost_total_text(cost: dict[str, str] | None, unpriced: int | None) -> str:
    """A cost total that never shows calls without a price as free."""

    if cost is None:
        return "unknown"
    priced = ", ".join(f"{amount} {currency}" for currency, amount in cost.items()) or None
    if not unpriced:
        return priced or "0"
    return "unknown" if priced is None else f"{priced} + unpriced"


def _inspection_window(args: argparse.Namespace) -> InspectionWindow:
    return InspectionWindow(
        after_sequence=args.after_sequence,
        before_sequence=args.before_sequence,
        interaction_id=args.interaction_id,
    )


def _render_usage(
    output: str,
    payload: dict[str, Any],
    calls: list[dict[str, Any]],
    unmatched_ledger: list[dict[str, Any]],
) -> None:
    safe_payload = _redact_sensitive(payload)
    if output == "json":
        print(json.dumps(safe_payload, ensure_ascii=False, sort_keys=True))
        return
    if output == "jsonl":
        for call in calls:
            print(
                json.dumps(
                    _redact_sensitive(
                        {
                            "record_type": "model_call",
                            **call,
                            "schema_version": payload["schema_version"],
                        }
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        for ledger in unmatched_ledger:
            print(
                json.dumps(
                    _redact_sensitive(
                        {
                            "record_type": "unmatched_ledger",
                            **ledger,
                            "schema_version": payload["schema_version"],
                        }
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        print(
            json.dumps(
                _redact_sensitive(
                    {
                        "record_type": "aggregate",
                        **payload["aggregate"],
                        "schema_version": payload["schema_version"],
                    }
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    safe_calls = [
        {
            **call,
            "cost": _cost_text(call),
            "duration": _optional_duration_text(call.get("duration_ms")),
            "first_output": _optional_duration_text(call.get("first_output_ms")),
        }
        for call in cast("list[dict[str, Any]]", _redact_sensitive(calls))
    ]
    _print_table(
        (
            "sequence",
            "timestamp",
            "duration",
            "first_output",
            "provider",
            "requested_model",
            "resolved_model",
            "finish_reason",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "cached_input_tokens",
            "uncached_input_tokens",
            "transcript_cursor",
            "pricing_state",
            "cost",
        ),
        safe_calls,
    )
    if safe_calls:
        print()
    if _print_request_controls(safe_calls):
        print()
    print("Aggregate usage")
    safe_aggregate = cast("dict[str, Any]", safe_payload["aggregate"])
    _print_table(
        ("metric", "value"),
        [
            {
                "metric": metric,
                "value": _cost_total_text(value, safe_aggregate.get("unpriced_calls"))
                if metric == "estimated_cost" and isinstance(value, dict)
                else value,
            }
            for metric, value in safe_aggregate.items()
        ],
    )
    price_book = safe_payload.get("price_book")
    if isinstance(price_book, dict):
        print(
            f"Priced with the {price_book['source']} price book "
            f"{price_book['version']} ({price_book['generated_at']})."
        )
    if unmatched_ledger:
        print()
        print("Unmatched ledger")
        safe_ledger = cast("list[dict[str, Any]]", _redact_sensitive(unmatched_ledger))
        _print_table(
            (
                "reservation_id",
                "outcome",
                "reserved_amount",
                "actual_amount",
                "currency",
                "pricing_state",
            ),
            safe_ledger,
        )


def _print_request_controls(calls: list[dict[str, Any]]) -> bool:
    """Print the controls each call sent and its provider echoed, when any call has them."""

    rows = [
        {
            "sequence": call.get("sequence"),
            "sent": _request_controls_text(call.get("request_controls"), unset="provider defaults"),
            "echoed": _request_controls_text(call.get("echoed_request_controls"), unset=None),
            "mismatch": ", ".join(call.get("request_controls_mismatch") or ()) or None,
        }
        for call in calls
        if call.get("request_controls") is not None
        or call.get("echoed_request_controls") is not None
    ]
    if not rows:
        return False
    print("Request controls")
    _print_table(("sequence", "sent", "echoed", "mismatch"), rows)
    return True


def _request_controls_text(controls: object, *, unset: str | None) -> str | None:
    if not isinstance(controls, dict):
        return None
    if not controls:
        return unset
    return " ".join(
        f"{field}={'null' if value is None else _printable_control(value)}"
        for field, value in controls.items()
    )


def _printable_control(value: object) -> str:
    """Escape anything a terminal would interpret; echoed values come from the provider."""

    text = str(value)
    return text if text.isprintable() else text.encode("unicode_escape").decode("ascii")


async def _session_tools(args: argparse.Namespace, reader: SessionReader) -> int:
    await reader.require_session(args.session_id)
    result = await reader.tool_calls(
        args.session_id, _inspection_window(args), offset=args.offset, limit=args.limit
    )
    page = result["calls"]
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "session_id": args.session_id,
        "interaction_id": args.interaction_id,
        "calls": page,
        "offset": result["offset"],
        "next_offset": result["next_offset"],
        "has_more": result["has_more"],
        "total_calls": result["total_calls"],
        "event_window": {
            "after_sequence": args.after_sequence,
            "before_sequence": args.before_sequence,
        },
    }
    _render_collection(
        args.output_format,
        payload,
        (
            [{**call, "runner_commands": _runner_commands_cell(call)} for call in page]
            if args.output_format == "table"
            else page
        ),
        headers=(
            "sequence",
            "tool",
            "tool_call_id",
            "model_step_id",
            "model_attempt_id",
            "tool_round_id",
            "parallel_round_width",
            "arguments_state",
            "argument_summary",
            "started_at",
            "completed_at",
            "status",
            "approval_state",
            "duration_ms",
            "rendered_content_bytes",
            "structured_result_bytes",
            "artifact_bytes",
            "returned",
            "truncated",
            "runner_commands",
        ),
    )
    return 0


def _runner_commands_cell(call: dict[str, Any]) -> str | None:
    """A call's runner commands in one cell: each id (or runner), then its times."""

    commands = call.get("runner_commands")
    if not isinstance(commands, list) or not commands:
        return None
    cells = []
    for command in commands:
        if not isinstance(command, dict):
            continue
        times = " ".join(
            f"{label}={command[field]}ms"
            for label, field in (
                ("runtime", "runtime_ms"),
                ("backend", "backend_ms"),
                ("queue", "queue_ms"),
                ("run", "run_ms"),
            )
            if command.get(field) is not None
        )
        cells.append(f"{command.get('id') or command.get('adapter') or '-'} {times}".strip())
    return "; ".join(cells) or None


async def _session_events(args: argparse.Namespace, reader: SessionReader) -> int:
    await reader.require_session(args.session_id)
    records = await reader.query_events(
        EventQuery(
            session_id=args.session_id,
            interaction_id=args.interaction_id,
            event_types=tuple(args.event_types),
            tool_name=args.tool,
            agent_name=args.agent,
            environment_name=args.environment,
            since=args.since,
            until=args.until,
            after_sequence=args.after_sequence,
            before_sequence=args.before_sequence,
            limit=args.limit + 1,
            order_by=EventOrder.SEQUENCE_ASC,
        )
    )
    has_more = len(records) > args.limit
    page = records[: args.limit]
    rows = [_event_row(record, payload_limit=args.include_payload) for record in page]
    next_sequence = page[-1].sequence if has_more and page else None
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "session_id": args.session_id,
        "interaction_id": args.interaction_id,
        "events": rows,
        "order": "sequence_asc",
        "next_sequence": next_sequence,
        "has_more": has_more,
    }
    headers = (
        "sequence",
        "timestamp",
        "type",
        "interaction_id",
        "tool",
        "agent",
        "environment",
        "payload_bytes",
    )
    if args.include_payload is not None:
        headers += ("payload_preview", "payload_truncated")
    _render_collection(
        args.output_format,
        payload,
        rows,
        headers=headers,
    )
    return 0


def _event_row(record: EventRecord, *, payload_limit: int | None) -> dict[str, Any]:
    event = record.event
    payload_bytes = compact_json_utf8_size(event.payload)
    row: dict[str, Any] = {
        "sequence": record.sequence,
        "timestamp": event.timestamp.isoformat(),
        "type": str(event.type),
        "tool": event.tool_name,
        "agent": event.agent_name,
        "environment": event.environment_name,
        "interaction_id": event.interaction_id,
        "payload_bytes": payload_bytes,
    }
    if payload_limit is not None:
        safe_payload = _redact_sensitive(event.payload)
        preview = json.dumps(
            safe_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        row["payload_preview"] = _truncate_utf8(preview, max_bytes=payload_limit)
        row["payload_truncated"] = len(preview.encode("utf-8")) > payload_limit
    return row


async def _session_output(args: argparse.Namespace, reader: SessionReader) -> int:
    """Print the last structured output validated in the session's latest (or the named)
    interaction: the answer to the request that started it."""

    session_id = args.session_id
    status = await reader.session_status(session_id)
    if status in _UNFINISHED_SESSION_STATUSES:
        raise SessionOutputUnavailable(
            "SESSION_NOT_FINISHED",
            f"Session {session_id} hasn't finished: it is {status}.",
            exit_code=EXIT_SESSION_NOT_FINISHED,
        )
    latest = (await reader.latest_interactions(session_id, before_sequence=None, limit=1)).rows
    interaction_id = args.interaction_id or (latest[0]["interaction_id"] if latest else None)
    if latest and interaction_id == latest[0]["interaction_id"]:
        interaction_status = latest[0]["status"]
        if interaction_status in _UNFINISHED_INTERACTION_STATUSES:
            raise SessionOutputUnavailable(
                "SESSION_NOT_FINISHED",
                f"Session {session_id} hasn't finished: it is {status} and its latest "
                f"interaction is {interaction_status}"
                + (
                    ", waiting for an approval or an answer."
                    if interaction_status == InteractionStatus.PAUSED.value
                    else "."
                ),
                exit_code=EXIT_SESSION_NOT_FINISHED,
            )
    records = (
        []
        if interaction_id is None
        else await reader.query_events(
            EventQuery(
                session_id=session_id,
                interaction_id=interaction_id,
                event_types=(EventType.STRUCTURED_OUTPUT_VALIDATED,),
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
    )
    if not records:
        which = f"interaction {interaction_id}" if args.interaction_id else "its latest interaction"
        raise SessionOutputUnavailable(
            "NO_STRUCTURED_OUTPUT",
            f"Session {session_id} has no structured output from {which}; it is {status}.",
            exit_code=EXIT_NO_STRUCTURED_OUTPUT,
        )
    record = records[0]
    event = record.event
    if "output" not in event.payload:
        raise ValueError(
            f"Session {session_id}'s structured output (event {record.sequence}) was "
            "recorded without its value."
        )
    payload = _redact_sensitive(
        {
            "schema_version": CLI_SCHEMA_VERSION,
            "session_id": session_id,
            "session_status": status,
            "interaction_id": event.interaction_id,
            "name": event.payload.get("name"),
            "attempt": event.payload.get("attempt"),
            "sequence": record.sequence,
            "validated_at": event.timestamp.isoformat(),
            "output": event.payload["output"],
        }
    )
    if args.output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    print_table(
        ("field", "value"),
        [
            {"field": key, "value": value}
            for key, value in payload.items()
            if key not in {"schema_version", "output"}
        ],
    )
    print("\nOutput")
    print(
        terminal_safe(
            json.dumps(payload["output"], ensure_ascii=False, indent=2, sort_keys=True),
            keep_lines=True,
        )
    )
    return 0


async def _session_transcript(args: argparse.Namespace, reader: SessionReader) -> int:
    await reader.require_session(args.session_id)
    page = await reader.query_transcript(
        TranscriptQuery(
            session_id=args.session_id,
            interaction_id=args.interaction_id,
            offset=args.offset,
            limit=args.limit,
        )
    )
    remaining_content_bytes = _MAX_TRANSCRIPT_CONTENT_BYTES
    rows: list[dict[str, Any]] = []
    for entry in page.entries:
        content_limit = args.include_content
        if content_limit is not None:
            content_limit = min(content_limit, remaining_content_bytes)
        row = _transcript_row(
            entry,
            sizes=args.sizes,
            content_limit=content_limit,
        )
        if content_limit is not None:
            remaining_content_bytes -= len(row["content_json"].encode("utf-8"))
        rows.append(row)

    next_offset = args.offset + len(rows)
    has_more = next_offset < page.total_records
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "session_id": args.session_id,
        "interaction_id": args.interaction_id,
        "messages": rows,
        "offset": args.offset,
        "next_offset": next_offset if has_more else None,
        "has_more": has_more,
        "total_messages": page.total_records,
    }
    headers = (
        (
            "index",
            "interaction_id",
            "role",
            "message_bytes",
            "content_part_count",
            "content_parts_truncated",
            "largest_part_bytes",
            "content_kinds",
        )
        if args.sizes
        else (
            "index",
            "interaction_id",
            "role",
            "message_bytes",
            "content_part_count",
            "content_parts_truncated",
            "content_kinds",
            "preview",
        )
    )
    if args.include_content is not None:
        headers += ("content_json", "content_truncated")
    _render_collection(args.output_format, payload, rows, headers=headers)
    return 0


def _transcript_row(
    entry: TranscriptEntry,
    *,
    sizes: bool,
    content_limit: int | None,
) -> dict[str, Any]:
    serialized = {"role": entry.role, "content": entry.content}
    retained_parts = entry.content[:_MAX_TRANSCRIPT_SUMMARY_PARTS]
    part_sizes = [
        {
            "kind": part.get("type"),
            "bytes": compact_json_utf8_size(part),
        }
        for part in retained_parts
    ]
    row: dict[str, Any] = {
        "index": entry.index,
        "interaction_id": entry.interaction_id,
        "role": entry.role,
        "message_bytes": compact_json_utf8_size(serialized),
        "content_part_count": len(entry.content),
        "content_parts_truncated": len(entry.content) > len(retained_parts),
        "content_kinds": [part.get("type") for part in retained_parts],
        "preview": _message_preview(serialized, max_bytes=160),
    }
    if sizes:
        row["part_sizes"] = part_sizes
        row["largest_part_bytes"] = max(
            (compact_json_utf8_size(part) for part in entry.content),
            default=0,
        )
    if content_limit is not None:
        safe_content = _redact_sensitive(serialized["content"])
        rendered = json.dumps(
            safe_content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        row["content_json"] = _truncate_utf8(rendered, max_bytes=content_limit)
        row["content_truncated"] = len(rendered.encode("utf-8")) > content_limit
    return row


def _message_preview(serialized: dict[str, Any], *, max_bytes: int) -> str:
    previews: list[str] = []
    content = serialized["content"]
    for raw_part in content[:_MAX_TRANSCRIPT_SUMMARY_PARTS]:
        # A Cloud Agent's API withholds some parts' private fields, so read every field
        # defensively.
        part = _redact_sensitive(raw_part)
        kind = part.get("type")
        if kind == "text":
            previews.append(str(part.get("text", "")))
        elif kind == "thinking":
            previews.append("[thinking]")
        elif kind == "tool_call":
            arguments = json.dumps(
                part.get("arguments"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            previews.append(f"{part.get('tool_name')}({arguments})")
        elif kind == "tool_result":
            previews.append(f"{part.get('tool_name')}: {part.get('content')}")
        elif kind == "file":
            previews.append("[file]")
        elif kind == "citation":
            previews.append(f"[citation: {part.get('url')}]")
        elif kind == "hosted_tool_call":
            action = part.get("action")
            source_count = len(action.get("sources", [])) if isinstance(action, dict) else 0
            previews.append(f"[hosted web search {part.get('status')}: {source_count} sources]")
        else:
            previews.append(f"[{kind}]")
    if len(content) > _MAX_TRANSCRIPT_SUMMARY_PARTS:
        previews.append(f"[+{len(content) - _MAX_TRANSCRIPT_SUMMARY_PARTS} parts]")
    return _truncate_utf8(" | ".join(previews), max_bytes=max_bytes)


def _add_created_window_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--since",
        type=_window_instant,
        metavar="WHEN",
        help=(
            "Only sessions created at or after WHEN: an ISO 8601 time with an offset, "
            "or a date (UTC midnight)."
        ),
    )
    parser.add_argument(
        "--until",
        type=_window_instant,
        metavar="WHEN",
        help="Only sessions created before WHEN, in the same form as --since.",
    )


def _window_instant(value: str) -> datetime:
    try:
        return datetime.combine(date.fromisoformat(value), datetime.min.time(), tzinfo=UTC)
    except ValueError:
        return _datetime_argument(value)


def _group_by_argument(value: str) -> str:
    try:
        rollup_grouping(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return value


def _add_target_options(parser: argparse.ArgumentParser, *, cloud: bool = True) -> None:
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--sqlite", metavar="PATH", help="Read this local SQLite store.")
    target.add_argument("--postgres", metavar="DSN", help="Read this PostgreSQL store.")
    if not cloud:
        return
    target.add_argument(
        "--server-url",
        metavar="URL",
        help=(
            "Read the sessions of the running Cayu server at this mount URL through its Cayu API."
        ),
    )
    parser.add_argument(
        "--authorization-env",
        default="CAYU_API_AUTHORIZATION",
        metavar="NAME",
        help=(
            "With --server-url: the environment variable holding the complete "
            "Authorization header value (default: CAYU_API_AUTHORIZATION). Never printed."
        ),
    )
    target.add_argument(
        "--application",
        metavar="NAME",
        help=(
            "Read the sessions of this Cayu Cloud Agent (its id, name, or slug) through its "
            "own Cayu API, signed in like `cayu cloud` commands. Needs a developer or admin."
        ),
    )
    parser.add_argument(
        "--context",
        type=Path,
        metavar="PATH",
        help="Private Cayu Cloud context JSON for --application; defaults to `cayu cloud`'s.",
    )
    from cayu.cli._cloud_agent_api import add_wake_options

    add_wake_options(parser)


def _parse_labels(values: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for value in values:
        key, separator, label_value = value.partition("=")
        if not separator or not key or not label_value:
            raise ValueError("--label must use non-empty KEY=VALUE syntax.")
        if key in labels:
            raise ValueError(f"--label repeats key {key!r}.")
        labels[key] = label_value
    return labels


def _render_collection(
    output: str,
    payload: dict[str, Any],
    items: list[dict[str, Any]],
    *,
    headers: tuple[str, ...],
) -> None:
    safe_payload = _redact_sensitive(payload)
    safe_items = _redact_sensitive(items)
    if output == "json":
        print(json.dumps(safe_payload, ensure_ascii=False, sort_keys=True))
        return
    if output == "jsonl":
        for item in safe_items:
            print(json.dumps(item, ensure_ascii=False, sort_keys=True))
        return
    _print_table(headers, safe_items)


def _render_detail(output: str, payload: dict[str, Any]) -> None:
    safe_payload = _redact_sensitive(payload)
    if output in {"json", "jsonl"}:
        print(json.dumps(safe_payload, ensure_ascii=False, sort_keys=True))
        return
    rows: list[dict[str, Any]] = []
    for section, value in safe_payload.items():
        if section == "schema_version":
            continue
        if isinstance(value, dict):
            for key, item in value.items():
                rows.append({"field": f"{section}.{key}", "value": _display_value(item)})
        else:
            rows.append({"field": section, "value": _display_value(value)})
    _print_table(("field", "value"), rows)


def _display_value(value: object) -> str:
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _cell(value)


def _safe_error(message: str, dsn: str | None) -> str:
    return message if dsn is None else _sanitize(message, dsn)


def _print_table(headers: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    rendered = [[_cell(row.get(header)) for header in headers] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in rendered))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    for row in rendered:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _cell(value: object) -> str:
    if value is None:
        return "-"
    # Cells can hold journaled text, such as a gated-loop item key, so terminal controls
    # are escaped rather than sent to the operator's terminal.
    return terminal_safe(str(value))


def _positive_limit(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 1000:
        raise argparse.ArgumentTypeError("limit must be between 1 and 1000.")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive.")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative.")
    return parsed


def _json_object_argument(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("metadata must be a JSON object.") from exc
    if type(parsed) is not dict:
        raise argparse.ArgumentTypeError("metadata must be a JSON object.")
    return parsed


def _payload_limit(value: str) -> int:
    return _bounded_content_bytes(value, kind="payload")


def _content_limit(value: str) -> int:
    return _bounded_content_bytes(value, kind="content")


def _bounded_content_bytes(value: str, *, kind: str) -> int:
    parsed = int(value)
    if not 16 <= parsed <= 65_536:
        raise argparse.ArgumentTypeError(f"{kind} bytes must be between 16 and 65536.")
    return parsed


def _event_types_argument(value: str) -> list[EventType | str]:
    selected: list[EventType | str] = []
    for item in value.split(","):
        item = item.strip()
        try:
            selected.append(EventType(item))
        except ValueError:
            if not item.startswith("custom."):
                raise argparse.ArgumentTypeError(
                    f"unknown event type {item!r}; use a built-in type, such as "
                    "tool.call.failed, or a custom.* type."
                ) from None
            try:
                selected.append(validate_custom_event_type(item))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(str(exc)) from None
    return selected


def _datetime_argument(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timestamp must use ISO 8601 syntax.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone offset.")
    return parsed
