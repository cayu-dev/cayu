"""Bounded read-only inspection of durable Cayu sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

from cayu._validation import compact_json_utf8_size
from cayu.cli._output import add_output_options, output_destination
from cayu.cli.storage import _sanitize
from cayu.cli.store_targets import (
    SessionStoreBackend,
    SessionStoreTarget,
    SessionStoreTargetError,
    resolve_session_store_target,
)
from cayu.core import EventType
from cayu.runtime import (
    EventOrder,
    EventQuery,
    EventRecord,
    SessionOrder,
    SessionQuery,
    SessionStatus,
    SessionStore,
    TranscriptQuery,
    TranscriptRecord,
    session_usage_summary,
    usage_metrics_from_event_payload,
)
from cayu.runtime.usage import count_model_steps_with_usage
from cayu.storage import SQLiteSessionStore
from cayu.storage import migrations as schema

FORMAT_CHOICES = ("json", "table", "jsonl")
CLI_SCHEMA_VERSION = "1"
_MAX_COLLECTED_EVENT_BYTES = 64 * 1024 * 1024
_MAX_COLLECTED_EVENT_RECORDS = 100_000
_MAX_TRANSCRIPT_CONTENT_BYTES = 1_048_576
_MAX_TRANSCRIPT_SUMMARY_PARTS = 100
_EVENT_QUERY_PAGE_SIZE = 200


def add_session_parser(subparsers: Any) -> None:
    """Register the singular ``cayu session`` command group."""

    session = subparsers.add_parser(
        "session",
        help="Inspect durable Cayu sessions without mutating storage.",
        description=(
            "Inspect durable Cayu sessions without mutating storage. "
            "Start with `cayu session list`; JSON is the default output."
        ),
    )
    commands = session.add_subparsers(dest="session_command", required=True)
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
        "--label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Filter by an exact label; repeat for multiple labels.",
    )
    list_parser.add_argument("--limit", type=_positive_limit, default=50)
    paging = list_parser.add_mutually_exclusive_group()
    paging.add_argument("--offset", type=_nonnegative_int, default=0)
    paging.add_argument("--cursor")
    add_output_options(list_parser, formats=FORMAT_CHOICES)

    show_parser = commands.add_parser(
        "show",
        help="Show a compact session overview.",
        description=(
            "Show a compact session overview. Use usage, tools, events, or transcript "
            "for bounded detail."
        ),
    )
    show_parser.add_argument("session_id")
    _add_target_options(show_parser)
    add_output_options(show_parser, formats=FORMAT_CHOICES)

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
    add_output_options(usage_parser, formats=FORMAT_CHOICES)

    tools_parser = commands.add_parser(
        "tools",
        help="Show paired durable tool calls.",
        description=(
            "Show paired durable tool calls without result bodies. "
            "Use event inspection for bounded payload metadata."
        ),
    )
    tools_parser.add_argument("session_id")
    _add_target_options(tools_parser)
    tools_parser.add_argument("--offset", type=_nonnegative_int, default=0)
    tools_parser.add_argument("--limit", type=_positive_limit, default=100)
    tools_parser.add_argument("--after-sequence", type=_nonnegative_int)
    tools_parser.add_argument("--before-sequence", type=_positive_int)
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
    events_parser.add_argument("--type", action="append", dest="event_types", default=[])
    events_parser.add_argument("--tool")
    events_parser.add_argument("--agent")
    events_parser.add_argument("--environment")
    events_parser.add_argument("--since", type=_datetime_argument)
    events_parser.add_argument("--until", type=_datetime_argument)
    events_parser.add_argument("--after-sequence", type=_nonnegative_int)
    events_parser.add_argument("--before-sequence", type=_positive_int)
    events_parser.add_argument("--limit", type=_positive_limit, default=100)
    events_parser.add_argument(
        "--include-payload",
        nargs="?",
        const=2048,
        type=_payload_limit,
        metavar="BYTES",
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


def run_session(args: argparse.Namespace) -> int:
    """Resolve a read-only target and dispatch one session-inspection command."""

    try:
        with output_destination(args.output):
            return _run_session(args)
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 1


def _run_session(args: argparse.Namespace) -> int:
    """Run after the optional output destination owns stdout."""

    dsn: str | None = None
    try:
        target = resolve_session_store_target(
            sqlite=args.sqlite,
            postgres=args.postgres,
        )
        dsn = target.postgres_dsn
        return asyncio.run(_run_session_command(args, target))
    except (SessionStoreTargetError, ValueError, OSError, RuntimeError) as exc:
        _render_session_error(_safe_error(str(exc), dsn), args.output_format)
        return 1
    except Exception as exc:
        # Driver and SQLite failures have backend-specific exception types. Keep
        # the CLI concise while preserving the existing DSN scrubbing contract.
        _render_session_error(_safe_error(str(exc), dsn), args.output_format)
        return 1


def _render_session_error(message: str, output_format: str) -> None:
    if output_format in {"json", "jsonl"}:
        print(
            json.dumps(
                {
                    "schema_version": CLI_SCHEMA_VERSION,
                    "error": {"code": "SESSION_INSPECTION_FAILED", "message": message},
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    print(f"error: {message}", file=sys.stderr)


async def _run_session_command(
    args: argparse.Namespace,
    target: SessionStoreTarget,
) -> int:
    store = _open_read_only_store(target)
    try:
        if args.session_command == "list":
            return await _list_sessions(args, store)
        if args.session_command == "show":
            return await _show_session(args, store)
        if args.session_command == "usage":
            return await _session_usage(args, store)
        if args.session_command == "tools":
            return await _session_tools(args, store)
        if args.session_command == "events":
            return await _session_events(args, store)
        if args.session_command == "transcript":
            return await _session_transcript(args, store)
        return 1
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            await close()


def _open_read_only_store(target: SessionStoreTarget) -> SessionStore:
    if target.backend is SessionStoreBackend.SQLITE:
        if target.sqlite_path is None:
            raise AssertionError("Resolved SQLite target has no path.")
        return SQLiteSessionStore(
            target.sqlite_path,
            schema_mode=schema.SchemaMode.VALIDATE,
            read_only=True,
        )
    if target.postgres_dsn is None:
        raise AssertionError("Resolved Postgres target has no DSN.")
    from cayu.storage import PostgresSessionStore

    return PostgresSessionStore(
        target.postgres_dsn,
        schema_mode=schema.SchemaMode.VALIDATE,
        read_only=True,
    )


async def _list_sessions(args: argparse.Namespace, store: SessionStore) -> int:
    labels = _parse_labels(args.label)
    result = await store.list_sessions(
        SessionQuery(
            status=None if args.status is None else SessionStatus(args.status),
            agent_name=args.agent,
            environment_name=args.environment,
            labels=labels,
            limit=args.limit,
            offset=args.offset,
            cursor=args.cursor,
            include_total_count=True,
            order_by=SessionOrder.LAST_ACTIVITY_AT_DESC,
        )
    )
    sessions = [
        {
            "id": session.id,
            "status": session.status.value,
            "agent": session.agent_name,
            "provider": session.provider_name,
            "model": session.model,
            "environment": session.environment_name,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "last_activity_at": session.last_activity_at.isoformat(),
            "run_epoch": session.run_epoch,
        }
        for session in result.sessions
    ]
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "sessions": sessions,
        "next_cursor": result.next_cursor,
        "has_more": result.next_cursor is not None,
        "total_count": result.total_count,
    }
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


async def _show_session(args: argparse.Namespace, store: SessionStore) -> int:
    try:
        summary = await store.inspect_summary(args.session_id)
    except KeyError as exc:
        raise ValueError(f"Session not found: {args.session_id}") from exc
    identity = summary.session
    usage = summary.usage.usage
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "session": {
            "id": identity.id,
            "parent_session_id": identity.parent_session_id,
            "causal_budget_id": identity.causal_budget_id,
            "agent": identity.agent_name,
            "provider": identity.provider_name,
            "model": identity.model,
            "runtime_name": identity.runtime_name,
            "runtime_version": identity.runtime_version,
            "environment": identity.environment_name,
            "status": identity.status.value,
            "created_at": identity.created_at.isoformat(),
            "updated_at": identity.updated_at.isoformat(),
            "last_activity_at": identity.last_activity_at.isoformat(),
            "run_epoch": identity.run_epoch,
            "labels": identity.labels,
            "label_count": identity.label_count,
            "labels_truncated": identity.labels_truncated,
        },
        "transcript": {
            "message_count": summary.transcript.record_count,
            "total_message_bytes": summary.transcript.total_bytes,
            "largest_message_bytes": summary.transcript.largest_record_bytes,
        },
        "events": {
            "event_count": summary.events.record_count,
            "total_payload_bytes": summary.events.total_bytes,
            "largest_payload_bytes": summary.events.largest_record_bytes,
        },
        "activity": {
            "model_calls": summary.model_calls,
            "model_calls_with_usage": summary.model_calls_with_usage,
            "tool_calls": summary.tool_calls,
        },
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "reasoning_tokens": usage.reasoning_output_tokens,
            "cache_read_tokens": usage.cache.read_tokens,
            "cache_write_tokens": usage.cache.write_tokens,
            "cached_input_tokens": usage.cache.cached_input_tokens,
            "uncached_input_tokens": usage.cache.uncached_input_tokens,
        },
        "pending_action": {
            "count": summary.pending_action_count,
            "kinds": [kind.value for kind in summary.pending_action_kinds],
            "issue_count": summary.pending_action_issue_count,
        },
        "queue": {
            "queued": summary.queued_message_count,
            "delivered": summary.delivered_message_count,
            "outstanding": summary.outstanding_message_count,
        },
        "operation": {
            "accepted_event_count": summary.operation_event_count,
            "state": "present" if summary.operation_event_count else "none",
        },
        "terminal_failure": {"state": summary.terminal_failure_state},
        "budget": summary.budget.model_dump(mode="json"),
    }
    _render_detail(args.output_format, payload)
    return 0


_USAGE_EVENT_TYPES = (
    EventType.MODEL_COMPLETED,
    EventType.TOOL_CALL_STARTED,
    EventType.BUDGET_RESERVED,
    EventType.BUDGET_RECONCILED,
    EventType.BUDGET_RESERVATION_RELEASED,
)


async def _session_usage(args: argparse.Namespace, store: SessionStore) -> int:
    await _require_session(store, args.session_id)
    records = await _query_all_event_records(
        store,
        args.session_id,
        event_types=_USAGE_EVENT_TYPES,
        project_record=_usage_inspection_record,
        after_sequence=args.after_sequence,
        before_sequence=args.before_sequence,
    )
    usage_events = [
        record.event
        for record in records
        if record.event.type in {EventType.MODEL_COMPLETED, EventType.TOOL_CALL_STARTED}
    ]
    aggregate = session_usage_summary(args.session_id, usage_events)
    model_calls_with_usage = count_model_steps_with_usage(usage_events)
    calls, unmatched_ledger = _model_call_usage(records)
    page = calls[args.offset : args.offset + args.limit]
    ledger_page = unmatched_ledger[args.offset : args.offset + args.limit]
    next_offset = args.offset + len(page)
    ledger_next_offset = args.offset + len(ledger_page)
    has_more = next_offset < len(calls)
    ledger_has_more = ledger_next_offset < len(unmatched_ledger)
    aggregate_usage = aggregate.usage
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "session_id": args.session_id,
        "calls": page,
        "offset": args.offset,
        "next_offset": next_offset if has_more else None,
        "has_more": has_more,
        "total_calls": len(calls),
        "event_window": {
            "after_sequence": args.after_sequence,
            "before_sequence": args.before_sequence,
        },
        "unmatched_ledger": ledger_page,
        "unmatched_ledger_total": len(unmatched_ledger),
        "unmatched_ledger_next_offset": ledger_next_offset if ledger_has_more else None,
        "unmatched_ledger_has_more": ledger_has_more,
        "aggregate": {
            "model_calls": aggregate.model_steps,
            "model_calls_with_usage": model_calls_with_usage,
            "tool_calls": aggregate.tool_calls,
            "input_tokens": aggregate_usage.input_tokens,
            "output_tokens": aggregate_usage.output_tokens,
            "total_tokens": aggregate_usage.total_tokens,
            "reasoning_tokens": aggregate_usage.reasoning_output_tokens,
            "cache_read_tokens": aggregate_usage.cache.read_tokens,
            "cache_write_tokens": aggregate_usage.cache.write_tokens,
            "cached_input_tokens": aggregate_usage.cache.cached_input_tokens,
            "uncached_input_tokens": aggregate_usage.cache.uncached_input_tokens,
        },
    }
    _render_usage(
        args.output_format,
        payload,
        page,
        ledger_page,
    )
    return 0


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
                    _redact_sensitive({"record_type": "model_call", **call}),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        for ledger in unmatched_ledger:
            print(
                json.dumps(
                    _redact_sensitive({"record_type": "unmatched_ledger", **ledger}),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        print(
            json.dumps(
                _redact_sensitive({"record_type": "aggregate", **payload["aggregate"]}),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    safe_calls = cast("list[dict[str, Any]]", _redact_sensitive(calls))
    _print_table(
        (
            "sequence",
            "timestamp",
            "provider",
            "requested_model",
            "resolved_model",
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
        ),
        safe_calls,
    )
    if safe_calls:
        print()
    print("Aggregate usage")
    safe_aggregate = cast("dict[str, Any]", safe_payload["aggregate"])
    _print_table(
        ("metric", "value"),
        [{"metric": metric, "value": value} for metric, value in safe_aggregate.items()],
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


def _model_call_usage(
    records: list[EventRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    unmatched_ledger: list[dict[str, Any]] = []
    reservations: dict[str, dict[str, Any]] = {}
    settled_reservation_ids: set[str] = set()
    for record in records:
        event = record.event
        if event.type == EventType.BUDGET_RESERVED:
            reservation_id = event.payload.get("reservation_id")
            if type(reservation_id) is not str:
                continue
            reservations[reservation_id] = {
                "reservation_id": reservation_id,
                "reserved_amount": _optional_string(event.payload.get("requested")),
                "actual_amount": None,
                "currency": _optional_string(event.payload.get("currency")),
                "pricing_state": "unpriced",
            }
            continue
        if event.type == EventType.MODEL_COMPLETED:
            try:
                metrics = usage_metrics_from_event_payload(event.payload)
            except (TypeError, ValueError):
                metrics = None
            cache = None if metrics is None else metrics.cache
            call = {
                "sequence": record.sequence,
                "timestamp": event.timestamp.isoformat(),
                "provider": None if metrics is None else metrics.provider_name,
                "requested_model": None if metrics is None else metrics.requested_model,
                "resolved_model": None if metrics is None else metrics.model,
                "input_tokens": None if metrics is None else metrics.input_tokens,
                "output_tokens": None if metrics is None else metrics.output_tokens,
                "total_tokens": None if metrics is None else metrics.total_tokens,
                "reasoning_tokens": (None if metrics is None else metrics.reasoning_output_tokens),
                "cache_read_tokens": None if cache is None else cache.read_tokens,
                "cache_write_tokens": None if cache is None else cache.write_tokens,
                "cached_input_tokens": None if cache is None else cache.cached_input_tokens,
                "uncached_input_tokens": None if cache is None else cache.uncached_input_tokens,
                "transcript_cursor": _optional_nonnegative_int(
                    event.payload.get("transcript_cursor")
                ),
                "pricing_state": "unknown",
                "ledger": [],
            }
            calls.append(call)
            continue
        if event.type not in {
            EventType.BUDGET_RECONCILED,
            EventType.BUDGET_RESERVATION_RELEASED,
        }:
            continue
        reservation_id = event.payload.get("reservation_id")
        if type(reservation_id) is not str:
            continue
        ledger = reservations.get(reservation_id)
        if ledger is None:
            ledger = {
                "reservation_id": reservation_id,
                "reserved_amount": _optional_string(event.payload.get("reserved_amount")),
                "actual_amount": None,
                "currency": None,
                "pricing_state": "unpriced",
            }
        if event.type == EventType.BUDGET_RECONCILED:
            ledger["actual_amount"] = _optional_string(event.payload.get("actual_amount"))
            ledger["pricing_state"] = (
                "priced" if type(event.payload.get("pricing")) is dict else "unpriced"
            )
        ledger["outcome"] = (
            "reconciled" if event.type == EventType.BUDGET_RECONCILED else "released"
        )
        unmatched_ledger.append(ledger)
        settled_reservation_ids.add(reservation_id)
    for reservation_id, ledger in reservations.items():
        if reservation_id in settled_reservation_ids:
            continue
        unmatched_ledger.append({**ledger, "outcome": "open"})
    return calls, unmatched_ledger


_TOOL_EVENT_TYPES = (
    EventType.TOOL_CALL_STARTED,
    EventType.TOOL_CALL_COMPLETED,
    EventType.TOOL_CALL_FAILED,
    EventType.TOOL_CALL_BLOCKED,
    EventType.TOOL_CALL_APPROVAL_REQUESTED,
    EventType.TOOL_CALL_APPROVED,
    EventType.TOOL_CALL_APPROVAL_DENIED,
    EventType.TOOL_CALL_APPROVAL_EXPIRED,
    EventType.SESSION_AWAITING_USER_INPUT,
)
_TOOL_TERMINAL_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
    }
)


async def _session_tools(args: argparse.Namespace, store: SessionStore) -> int:
    await _require_session(store, args.session_id)
    records = await _query_all_event_records(
        store,
        args.session_id,
        event_types=_TOOL_EVENT_TYPES,
        project_record=_tool_inspection_record,
        after_sequence=args.after_sequence,
        before_sequence=args.before_sequence,
    )
    calls = _tool_call_rows(records)
    page = calls[args.offset : args.offset + args.limit]
    next_offset = args.offset + len(page)
    has_more = next_offset < len(calls)
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "session_id": args.session_id,
        "calls": page,
        "offset": args.offset,
        "next_offset": next_offset if has_more else None,
        "has_more": has_more,
        "total_calls": len(calls),
        "event_window": {
            "after_sequence": args.after_sequence,
            "before_sequence": args.before_sequence,
        },
    }
    _render_collection(
        args.output_format,
        payload,
        page,
        headers=(
            "sequence",
            "tool",
            "tool_call_id",
            "tool_round_id",
            "parallel_round_width",
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
        ),
    )
    return 0


def _tool_call_rows(records: list[EventRecord]) -> list[dict[str, Any]]:
    starts: dict[str, EventRecord] = {}
    terminals: dict[str, EventRecord] = {}
    approval_requests: dict[str, EventRecord] = {}
    approval_calls: dict[str, dict[str, Any]] = {}
    input_requests: dict[str, EventRecord] = {}
    input_calls: dict[str, dict[str, Any]] = {}
    decision_records: dict[str, EventRecord] = {}
    approval_states: dict[str, str] = {}
    for record in records:
        event = record.event
        if event.type == EventType.SESSION_AWAITING_USER_INPUT:
            nested_calls = event.payload.get("tool_calls")
            calls = (
                [item for item in nested_calls if type(item) is dict]
                if type(nested_calls) is list and nested_calls
                else [event.payload]
            )
            for call in calls:
                call_id = call.get("tool_call_id")
                if type(call_id) is not str:
                    continue
                input_requests.setdefault(call_id, record)
                input_calls.setdefault(call_id, call)
            continue
        if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED:
            approval = _tool_approval_payload(record)
            if approval is None:
                continue
            nested_calls = approval.get("tool_calls")
            calls = (
                [item for item in nested_calls if type(item) is dict]
                if type(nested_calls) is list and nested_calls
                else [approval]
            )
            for call in calls:
                call_id = call.get("tool_call_id")
                if type(call_id) is not str:
                    continue
                approval_requests.setdefault(call_id, record)
                approval_calls.setdefault(call_id, call)
                approval_states[call_id] = "requested"
            continue
        call_id = _tool_event_call_id(record)
        if call_id is None:
            continue
        if event.type == EventType.TOOL_CALL_STARTED:
            starts.setdefault(call_id, record)
        elif event.type in _TOOL_TERMINAL_TYPES:
            terminals.setdefault(call_id, record)
        elif event.type == EventType.TOOL_CALL_APPROVED:
            approval_states[call_id] = "approved"
            decision_records.setdefault(call_id, record)
        elif event.type == EventType.TOOL_CALL_APPROVAL_DENIED:
            approval_states[call_id] = "denied"
            decision_records.setdefault(call_id, record)
        elif event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED:
            approval_states[call_id] = "expired"
            decision_records.setdefault(call_id, record)

    call_id_set = (
        starts.keys()
        | terminals.keys()
        | approval_requests.keys()
        | input_requests.keys()
        | decision_records.keys()
    )
    anchors = {
        call_id: (
            starts.get(call_id)
            or approval_requests.get(call_id)
            or input_requests.get(call_id)
            or terminals.get(call_id)
            or decision_records[call_id]
        )
        for call_id in call_id_set
    }
    group_ids = {
        call_id: _tool_event_group_id(
            starts.get(call_id),
            terminals.get(call_id),
            approval_requests.get(call_id),
            input_requests.get(call_id),
            decision_records.get(call_id),
        )
        for call_id in call_id_set
    }
    round_widths: dict[str, int] = {}
    for round_id in group_ids.values():
        if round_id is not None:
            round_widths[round_id] = round_widths.get(round_id, 0) + 1

    rows: list[dict[str, Any]] = []
    call_ids = sorted(
        call_id_set,
        key=lambda call_id: (anchors[call_id].sequence, call_id),
    )
    for call_id in call_ids:
        started = starts.get(call_id)
        approval_request = approval_requests.get(call_id)
        approval_call = approval_calls.get(call_id)
        input_request = input_requests.get(call_id)
        input_call = input_calls.get(call_id)
        anchor = anchors[call_id]
        terminal = terminals.get(call_id)
        round_id = group_ids[call_id]
        approval_state = approval_states.get(call_id, "none")
        result = None
        inspection_result = None
        if terminal is not None:
            raw_inspection_result = terminal.event.payload.get("_inspection_result")
            if type(raw_inspection_result) is dict:
                inspection_result = raw_inspection_result
            raw_result = terminal.event.payload.get("result")
            if type(raw_result) is dict:
                result = raw_result
        structured = None if result is None else result.get("structured")
        structured = structured if type(structured) is dict else None
        content = None if result is None else result.get("content")
        artifacts = None if result is None else result.get("artifacts")
        approval = _tool_approval_payload(approval_request)
        argument_summary = (
            started.event.payload.get("_argument_summary")
            if started is not None
            else approval_call.get("_argument_summary")
            if approval_call is not None
            else input_call.get("_argument_summary")
            if input_call is not None
            else None
        )
        if type(argument_summary) is not str:
            arguments = (
                started.event.payload.get("arguments", {})
                if started is not None
                else approval_call.get("arguments", {})
                if approval_call is not None
                else input_call.get("arguments", {})
                if input_call is not None
                else {}
                if approval is None
                else approval.get("arguments", {})
            )
            argument_summary = _bounded_argument_summary(arguments)
        tool_name = (
            anchor.event.tool_name
            if approval_call is None and input_call is None
            else (approval_call or input_call or {}).get("tool_name", anchor.event.tool_name)
        )
        rows.append(
            {
                "sequence": anchor.sequence,
                "tool": tool_name,
                "tool_call_id": call_id,
                "tool_round_id": round_id,
                "parallel_round_width": (1 if round_id is None else round_widths[round_id]),
                "argument_summary": argument_summary,
                "started_at": (None if started is None else started.event.timestamp.isoformat()),
                "completed_at": (
                    None if terminal is None else terminal.event.timestamp.isoformat()
                ),
                "duration_ms": _duration_ms(started, terminal),
                "status": _tool_status(
                    started,
                    terminal,
                    approval_state,
                    awaiting_input=input_request is not None,
                ),
                "approval_state": approval_state,
                "rendered_content_bytes": (
                    _optional_nonnegative_int(inspection_result.get("rendered_content_bytes"))
                    if inspection_result is not None
                    else len(content.encode("utf-8"))
                    if type(content) is str
                    else 0
                ),
                "structured_result_bytes": (
                    _optional_nonnegative_int(inspection_result.get("structured_result_bytes"))
                    if inspection_result is not None
                    else compact_json_utf8_size(structured)
                    if structured is not None
                    else 0
                ),
                "artifact_bytes": (
                    _optional_nonnegative_int(inspection_result.get("artifact_bytes"))
                    if inspection_result is not None
                    else compact_json_utf8_size(artifacts)
                    if type(artifacts) is list
                    else 0
                ),
                "returned": (
                    _optional_nonnegative_int(inspection_result.get("returned"))
                    if inspection_result is not None
                    else None
                    if structured is None
                    else _optional_nonnegative_int(structured.get("returned"))
                ),
                "truncated": (
                    inspection_result.get("truncated")
                    if inspection_result is not None
                    and type(inspection_result.get("truncated")) is bool
                    else structured.get("truncated")
                    if structured is not None and type(structured.get("truncated")) is bool
                    else None
                ),
            }
        )
    return rows


def _tool_event_group_id(*records: EventRecord | None) -> str | None:
    for record in records:
        if record is None:
            continue
        round_id = _tool_event_round_id(record)
        if round_id is not None:
            return round_id
    for record in records:
        if record is None:
            continue
        approval_id = record.event.payload.get("approval_id")
        if type(approval_id) is not str:
            approval = _tool_approval_payload(record)
            approval_id = None if approval is None else approval.get("approval_id")
        if type(approval_id) is str:
            return f"approval:{approval_id}"
        input_id = record.event.payload.get("input_id")
        if type(input_id) is str:
            return f"input:{input_id}"
    return None


def _tool_event_call_id(record: EventRecord) -> str | None:
    call_id = record.event.payload.get("tool_call_id")
    if type(call_id) is str:
        return call_id
    approval = _tool_approval_payload(record)
    if approval is None:
        return None
    nested_call_id = approval.get("tool_call_id")
    return nested_call_id if type(nested_call_id) is str else None


def _tool_event_round_id(record: EventRecord) -> str | None:
    round_id = record.event.payload.get("tool_round_id")
    return round_id if type(round_id) is str else None


def _tool_approval_payload(record: EventRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    approval = record.event.payload.get("approval")
    return approval if type(approval) is dict else None


def _tool_status(
    started: EventRecord | None,
    terminal: EventRecord | None,
    approval_state: str,
    *,
    awaiting_input: bool,
) -> str:
    if terminal is None:
        if started is None and awaiting_input:
            return "awaiting_input"
        if started is None and approval_state == "requested":
            return "approval_pending"
        if started is None and approval_state in {"denied", "expired"}:
            return approval_state
        return "running"
    if terminal.event.type == EventType.TOOL_CALL_COMPLETED:
        return "success"
    if terminal.event.type == EventType.TOOL_CALL_BLOCKED:
        return "blocked"
    return "error"


def _duration_ms(
    started: EventRecord | None,
    terminal: EventRecord | None,
) -> int | None:
    if started is None or terminal is None:
        return None
    duration = terminal.event.timestamp - started.event.timestamp
    return max(round(duration.total_seconds() * 1000), 0)


def _bounded_json_summary(value: object, *, max_bytes: int) -> str:
    safe = _redact_sensitive(value)
    rendered = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _truncate_utf8(rendered, max_bytes=max_bytes)


def _truncate_utf8(value: str, *, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    if max_bytes <= 0:
        return ""
    marker = "…"
    if max_bytes < len(marker.encode("utf-8")):
        return "." * max_bytes
    retained = encoded[: max_bytes - len(marker.encode("utf-8"))]
    while retained:
        try:
            return retained.decode("utf-8") + marker
        except UnicodeDecodeError:
            retained = retained[:-1]
    return marker


async def _session_events(args: argparse.Namespace, store: SessionStore) -> int:
    await _require_session(store, args.session_id)
    records = await store.query_events(
        EventQuery(
            session_id=args.session_id,
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
        "events": rows,
        "order": "sequence_asc",
        "next_sequence": next_sequence,
        "has_more": has_more,
    }
    headers = (
        "sequence",
        "timestamp",
        "type",
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


async def _session_transcript(args: argparse.Namespace, store: SessionStore) -> int:
    await _require_session(store, args.session_id)
    page = await store.query_transcript(
        TranscriptQuery(
            session_id=args.session_id,
            offset=args.offset,
            limit=args.limit,
        )
    )
    remaining_content_bytes = _MAX_TRANSCRIPT_CONTENT_BYTES
    rows: list[dict[str, Any]] = []
    for record in page.records:
        content_limit = args.include_content
        if content_limit is not None:
            content_limit = min(content_limit, remaining_content_bytes)
        row = _transcript_row(
            record,
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
        "messages": rows,
        "offset": args.offset,
        "next_offset": next_offset if has_more else None,
        "has_more": has_more,
        "total_messages": page.total_records,
    }
    headers = (
        (
            "index",
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
    record: TranscriptRecord,
    *,
    sizes: bool,
    content_limit: int | None,
) -> dict[str, Any]:
    message = record.message
    serialized = message.model_dump(mode="json")
    retained_parts = message.content[:_MAX_TRANSCRIPT_SUMMARY_PARTS]
    part_sizes = [
        {
            "kind": part.type,
            "bytes": compact_json_utf8_size(part.model_dump(mode="json")),
        }
        for part in retained_parts
    ]
    row: dict[str, Any] = {
        "index": record.index,
        "role": str(message.role),
        "message_bytes": compact_json_utf8_size(serialized),
        "content_part_count": len(message.content),
        "content_parts_truncated": len(message.content) > len(retained_parts),
        "content_kinds": [part.type for part in retained_parts],
        "preview": _message_preview(serialized, max_bytes=160),
    }
    if sizes:
        row["part_sizes"] = part_sizes
        row["largest_part_bytes"] = max(
            (compact_json_utf8_size(part.model_dump(mode="json")) for part in message.content),
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
        part = _redact_sensitive(raw_part)
        kind = part["type"]
        if kind == "text":
            previews.append(part["text"])
        elif kind == "thinking":
            previews.append("[thinking]")
        elif kind == "tool_call":
            arguments = json.dumps(
                part["arguments"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            previews.append(f"{part['tool_name']}({arguments})")
        elif kind == "tool_result":
            previews.append(f"{part['tool_name']}: {part['content']}")
        elif kind == "file":
            previews.append("[file]")
        else:
            previews.append(f"[{kind}]")
    if len(content) > _MAX_TRANSCRIPT_SUMMARY_PARTS:
        previews.append(f"[+{len(content) - _MAX_TRANSCRIPT_SUMMARY_PARTS} parts]")
    return _truncate_utf8(" | ".join(previews), max_bytes=max_bytes)


async def _require_session(store: SessionStore, session_id: str) -> None:
    if await store.load(session_id) is None:
        raise ValueError(f"Session not found: {session_id}")


async def _query_all_event_records(
    store: SessionStore,
    session_id: str,
    *,
    event_types: tuple[EventType, ...] | None = None,
    project_record: Callable[[EventRecord], EventRecord] | None = None,
    after_sequence: int | None = None,
    before_sequence: int | None = None,
) -> list[EventRecord]:
    records: list[EventRecord] = []
    retained_bytes = 0
    cursor = 0 if after_sequence is None else after_sequence
    while True:
        page = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=() if event_types is None else event_types,
                after_sequence=cursor,
                before_sequence=before_sequence,
                limit=_EVENT_QUERY_PAGE_SIZE,
                order_by=EventOrder.SEQUENCE_ASC,
            )
        )
        if not page:
            return records
        for record in page:
            retained_record = record if project_record is None else project_record(record)
            retained_bytes += compact_json_utf8_size(retained_record.event.model_dump(mode="json"))
            if retained_bytes > _MAX_COLLECTED_EVENT_BYTES:
                raise ValueError(
                    "Session inspection exceeds the 64 MiB retained-event safety limit."
                    " Narrow the event window with --after-sequence or --before-sequence."
                )
            records.append(retained_record)
            if len(records) > _MAX_COLLECTED_EVENT_RECORDS:
                raise ValueError(
                    "Session inspection exceeds the "
                    f"{_MAX_COLLECTED_EVENT_RECORDS}-event safety limit. Narrow the event "
                    "window with --after-sequence or --before-sequence."
                )
        cursor = page[-1].sequence
        if len(page) < _EVENT_QUERY_PAGE_SIZE:
            return records


def _usage_inspection_record(record: EventRecord) -> EventRecord:
    event = record.event
    payload: dict[str, Any] = {}
    if event.type == EventType.MODEL_COMPLETED:
        transcript_cursor = event.payload.get("transcript_cursor")
        if transcript_cursor is not None:
            payload["transcript_cursor"] = transcript_cursor
        try:
            metrics = usage_metrics_from_event_payload(event.payload)
        except (TypeError, ValueError):
            payload["usage_metrics"] = {"_invalid": True}
        else:
            if metrics is not None:
                payload["usage_metrics"] = metrics.model_dump(mode="json")
    elif event.type == EventType.BUDGET_RESERVED:
        for key in ("reservation_id", "currency", "requested"):
            if key in event.payload:
                payload[key] = event.payload[key]
    elif event.type == EventType.BUDGET_RECONCILED:
        for key in ("reservation_id", "reserved_amount", "actual_amount"):
            if key in event.payload:
                payload[key] = event.payload[key]
        if type(event.payload.get("pricing")) is dict:
            payload["pricing"] = {}
    elif event.type == EventType.BUDGET_RESERVATION_RELEASED:
        if "reservation_id" in event.payload:
            payload["reservation_id"] = event.payload["reservation_id"]
    return EventRecord(
        sequence=record.sequence,
        event=event.model_copy(update={"payload": payload}),
    )


def _tool_inspection_record(record: EventRecord) -> EventRecord:
    event = record.event
    payload: dict[str, Any] = {}
    for key in ("tool_call_id", "tool_round_id", "approval_id", "input_id"):
        if key in event.payload:
            payload[key] = event.payload[key]
    if event.type == EventType.TOOL_CALL_STARTED:
        payload["_argument_summary"] = _bounded_argument_summary(event.payload.get("arguments", {}))
    if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED:
        approval = _tool_approval_payload(record)
        if approval is not None:
            compact_approval = {
                key: approval[key]
                for key in ("approval_id", "tool_call_id", "tool_name")
                if key in approval
            }
            compact_approval["_argument_summary"] = _bounded_argument_summary(
                approval.get("arguments", {})
            )
            nested_calls = approval.get("tool_calls")
            if type(nested_calls) is list:
                compact_approval["tool_calls"] = [
                    {
                        key: item[key]
                        for key in ("tool_call_id", "tool_name", "policy_decision")
                        if key in item
                    }
                    | {"_argument_summary": _bounded_argument_summary(item.get("arguments", {}))}
                    for item in nested_calls
                    if type(item) is dict
                ]
            payload["approval"] = compact_approval
    if event.type == EventType.SESSION_AWAITING_USER_INPUT:
        payload["question"] = _truncate_utf8(str(event.payload.get("question", "")), max_bytes=512)
        options = event.payload.get("options")
        if type(options) is list:
            payload["options"] = [
                _truncate_utf8(str(option), max_bytes=256) for option in options[:100]
            ]
        nested_calls = event.payload.get("tool_calls")
        if type(nested_calls) is list:
            payload["tool_calls"] = [
                {
                    key: item[key]
                    for key in ("tool_call_id", "tool_name", "policy_decision")
                    if key in item
                }
                | {"_argument_summary": _bounded_argument_summary(item.get("arguments", {}))}
                for item in nested_calls
                if type(item) is dict
            ]
    if event.type in _TOOL_TERMINAL_TYPES:
        result = event.payload.get("result")
        result = result if type(result) is dict else {}
        content = result.get("content")
        structured = result.get("structured")
        artifacts = result.get("artifacts")
        payload["_inspection_result"] = {
            "rendered_content_bytes": (len(content.encode("utf-8")) if type(content) is str else 0),
            "structured_result_bytes": (
                compact_json_utf8_size(structured) if type(structured) is dict else 0
            ),
            "artifact_bytes": (compact_json_utf8_size(artifacts) if type(artifacts) is list else 0),
            "returned": (
                _optional_nonnegative_int(structured.get("returned"))
                if type(structured) is dict
                else None
            ),
            "truncated": (
                structured.get("truncated")
                if type(structured) is dict and type(structured.get("truncated")) is bool
                else None
            ),
        }
    return EventRecord(
        sequence=record.sequence,
        event=event.model_copy(update={"payload": payload}),
    )


def _optional_string(value: object) -> str | None:
    return value if type(value) is str else None


def _optional_nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _add_target_options(parser: argparse.ArgumentParser) -> None:
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--sqlite", metavar="PATH")
    target.add_argument("--postgres", metavar="DSN")


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


_SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "aws_secret_access_key",
        "aws_access_key_id",
        "auth_token",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "encrypted_content",
        "password",
        "private_key",
        "provider_state",
        "refresh_token",
        "secret",
        "secret_key",
        "session_token",
        "signature",
        "signing_key",
        "token",
        "id_token",
    }
)

_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(api[-_ ]?key|auth(?:orization)?|access[-_ ]?token|refresh[-_ ]?token|"
    r"password|secret|credential)\b(\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_SECRET_TOKEN_PATTERN = re.compile(
    r"\b(?:sk|gh[pousr]|github_pat)_[A-Za-z0-9_-]{8,}\b|\bsk-[A-Za-z0-9_-]{8,}\b"
)
_POSTGRES_PASSWORD_PATTERN = re.compile(r"(?i)(postgres(?:ql)?://[^:/\s]+:)([^@\s]+)(@)")
_PEM_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?-----END [^-\r\n]*PRIVATE KEY-----",
    re.DOTALL,
)


def _redact_sensitive(value: Any, *, key: str | None = None) -> Any:
    normalized_key = "" if key is None else _normalize_sensitive_key(key)
    if key is not None and (
        normalized_key in _SENSITIVE_KEYS
        or normalized_key.endswith(("_api_key", "_password", "_secret", "_credential", "_token"))
    ):
        return "[REDACTED]"
    if isinstance(value, dict):
        if value.get("type") == "provider_state":
            return {
                item_key: (
                    "[REDACTED]"
                    if _normalize_sensitive_key(str(item_key)) == "state"
                    else _redact_sensitive(item, key=str(item_key))
                )
                for item_key, item in value.items()
            }
        return {
            item_key: _redact_sensitive(item, key=str(item_key)) for item_key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_redact_sensitive(item) for item in value]
    if isinstance(value, str):
        redacted = _PEM_PRIVATE_KEY_PATTERN.sub("[REDACTED PRIVATE KEY]", value)
        redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", redacted)
        redacted = _SECRET_ASSIGNMENT_PATTERN.sub(
            lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
            redacted,
        )
        redacted = _SECRET_TOKEN_PATTERN.sub("[REDACTED]", redacted)
        return _POSTGRES_PASSWORD_PATTERN.sub(r"\1***\3", redacted)
    return value


def _normalize_sensitive_key(key: str) -> str:
    separated = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", key)
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", separated)
    return re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")


def _bounded_argument_summary(value: Any) -> str:
    """Render a bounded tool-argument summary after structural redaction."""
    return _bounded_json_summary(_redact_sensitive(value), max_bytes=256)


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
    return str(value)


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


def _payload_limit(value: str) -> int:
    return _bounded_content_bytes(value, kind="payload")


def _content_limit(value: str) -> int:
    return _bounded_content_bytes(value, kind="content")


def _bounded_content_bytes(value: str, *, kind: str) -> int:
    parsed = int(value)
    if not 16 <= parsed <= 65_536:
        raise argparse.ArgumentTypeError(f"{kind} bytes must be between 16 and 65536.")
    return parsed


def _datetime_argument(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timestamp must use ISO 8601 syntax.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone offset.")
    return parsed
