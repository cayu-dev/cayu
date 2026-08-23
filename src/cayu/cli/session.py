"""Bounded inspection and explicit recovery of durable Cayu sessions."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import uuid4

import httpx

from cayu._validation import compact_json_utf8_size, require_clean_nonblank
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
    AggregateUsageMetrics,
    EventOrder,
    EventQuery,
    EventRecord,
    InteractionSummaryEvidence,
    SessionOrder,
    SessionQuery,
    SessionStatus,
    SessionStore,
    ToolPolicyDecision,
    ToolRoundIdentity,
    TranscriptQuery,
    TranscriptRecord,
    public_authority_alias_codec_from_environment,
    session_usage_summary,
    usage_metrics_from_event_payload,
)
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime.aggregates import summary_usage_metrics_from_event_payload
from cayu.runtime.budgets import is_complete_budget_reconciliation_pricing
from cayu.runtime.interactions import INTERACTION_TERMINAL_EVENT_TYPES
from cayu.runtime.provider_operations import inspect_provider_operation
from cayu.runtime.usage import count_model_steps_with_usage
from cayu.storage import SQLiteSessionStore
from cayu.storage import migrations as schema

FORMAT_CHOICES = ("json", "table", "jsonl")
CLI_SCHEMA_VERSION = "7"
_MAX_COLLECTED_EVENT_BYTES = 64 * 1024 * 1024
_MAX_COLLECTED_EVENT_RECORDS = 100_000
_MAX_TRANSCRIPT_CONTENT_BYTES = 1_048_576
_MAX_SERVER_ERROR_BODY_BYTES = 8 * 1024
_MAX_SERVER_ERROR_DETAIL_BYTES = 1024
_MAX_SERVER_FORK_GROUP_BODY_BYTES = 512 * 1024
_MAX_TRANSCRIPT_SUMMARY_PARTS = 100
_EVENT_QUERY_PAGE_SIZE = 200
_USAGE_INSPECTION_PRICING_STATE_KEY = "_cayu_pricing_state"
_TOOL_INSPECTION_IDENTITY_CONFLICT_KEY = "_cayu_tool_identity_conflict"


def add_session_parser(subparsers: Any) -> None:
    """Register the singular ``cayu session`` command group."""

    session = subparsers.add_parser(
        "session",
        help="Inspect durable sessions and resolve unavailable provider work.",
        description=(
            "Inspect durable Cayu sessions without mutating storage, or send an explicit "
            "provider-operation disposition to a running Cayu server. Start with "
            "`cayu session list`; JSON is the default output."
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
    events_parser.add_argument("--type", action="append", dest="event_types", default=[])
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

    fork_group_parser = commands.add_parser(
        "fork-group",
        help="Inspect one durable fork group and its task attempts.",
        description=(
            "Read bounded group, attempt, task, lease, recovery, and terminal state "
            "from a running Cayu server without claiming or advancing work."
        ),
    )
    fork_group_parser.add_argument("session_id", help="Source session id.")
    fork_group_parser.add_argument("group_id")
    fork_group_parser.add_argument(
        "--server-url",
        required=True,
        help="Cayu server root URL, including any mount prefix but excluding /api.",
    )
    fork_group_parser.add_argument(
        "--authorization-env",
        default="CAYU_API_AUTHORIZATION",
        metavar="NAME",
        help="Environment variable containing the complete Authorization header value.",
    )
    add_output_options(fork_group_parser, formats=("json", "table"))

    resolution_parser = commands.add_parser(
        "resolve-provider-operation",
        help="Explicitly retry or fail unavailable provider work.",
        description=(
            "Send a run-epoch-fenced fallback_retry or fail disposition to the Cayu "
            "server. Read stage_id and run_epoch from `cayu session show` or the "
            "session state API. Authorization is read from CAYU_API_AUTHORIZATION by "
            "default and is never printed."
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

    try:
        with output_destination(args.output):
            return _run_session(args)
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 1


def _run_session(args: argparse.Namespace) -> int:
    """Run after the optional output destination owns stdout."""

    if args.session_command == "resolve-provider-operation":
        try:
            return _resolve_provider_operation(args)
        except (ValueError, OSError, RuntimeError) as exc:
            _render_session_error(_safe_error(str(exc), None), args.output_format)
            return 1
    if args.session_command == "fork-group":
        try:
            return _inspect_fork_group(args)
        except (ValueError, OSError, RuntimeError) as exc:
            _render_session_error(_safe_error(str(exc), None), args.output_format)
            return 1

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


def _inspect_fork_group(args: argparse.Namespace) -> int:
    endpoint = _fork_group_inspection_endpoint(
        args.server_url,
        session_id=args.session_id,
        group_id=args.group_id,
    )
    authorization = os.environ.get(args.authorization_env)
    if authorization is not None and (
        not authorization.strip()
        or authorization != authorization.strip()
        or "\r" in authorization
        or "\n" in authorization
    ):
        raise ValueError(f"{args.authorization_env} contains an invalid Authorization value.")
    headers = {"Accept": "application/json"}
    if authorization is not None:
        headers["Authorization"] = authorization
    try:
        with (
            httpx.Client(follow_redirects=False, timeout=30.0) as client,
            client.stream("GET", endpoint, headers=headers) as response,
        ):
            if not 200 <= response.status_code < 300:
                detail = _safe_server_error_detail(
                    response,
                    authorization=authorization,
                )
                suffix = f": {detail}" if detail is not None else "."
                raise RuntimeError(f"Cayu server returned HTTP {response.status_code}{suffix}")
            body = bytearray()
            for chunk in response.iter_bytes(chunk_size=8192):
                if len(body) + len(chunk) > _MAX_SERVER_FORK_GROUP_BODY_BYTES:
                    raise RuntimeError("Cayu server returned an oversized fork-group response.")
                body.extend(chunk)
    except httpx.RequestError:
        raise RuntimeError("Cayu server is unavailable.") from None
    try:
        payload = json.loads(body)
    except (TypeError, ValueError, json.JSONDecodeError):
        raise RuntimeError("Cayu server returned malformed fork-group JSON.") from None
    if type(payload) is not dict:
        raise RuntimeError("Cayu server returned a non-object fork-group response.")
    _render_detail(args.output_format, payload)
    return 0


def _fork_group_inspection_endpoint(
    server_url: str,
    *,
    session_id: str,
    group_id: str,
) -> str:
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
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ValueError("Cayu server URL must use HTTPS outside loopback.")
    session_id = require_clean_nonblank(session_id, "session_id")
    group_id = require_clean_nonblank(group_id, "group_id")
    path = (
        parsed.path.rstrip("/")
        + "/api/sessions/"
        + quote(session_id, safe="")
        + "/fork-groups/"
        + quote(group_id, safe="")
    )
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
    if detail_size > _MAX_SERVER_ERROR_DETAIL_BYTES:
        return None
    if any(ord(character) < 32 and character not in "\t" for character in detail):
        return None
    if authorization is not None:
        detail = detail.replace(authorization, "[REDACTED]")
    redacted = _redact_sensitive(detail)
    return redacted if type(redacted) is str else None


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
        if args.session_command == "interactions":
            return await _session_interactions(args, store)
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
        summary = await runtime_checkpoint_session_store(store).inspect_summary(args.session_id)
    except KeyError as exc:
        raise ValueError(f"Session not found: {args.session_id}") from exc
    identity = summary.session
    provider_operation = await inspect_provider_operation(store, args.session_id)
    usage = _aggregate_usage_cli_payload(summary.usage.usage)
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
        "usage": usage,
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
        "provider_operation": provider_operation.model_dump(mode="json"),
        "terminal_failure": {"state": summary.terminal_failure_state},
        "budget": summary.budget.model_dump(mode="json"),
    }
    _render_detail(args.output_format, payload)
    return 0


async def _session_interactions(args: argparse.Namespace, store: SessionStore) -> int:
    await _require_session(store, args.session_id)
    records = await store.query_latest_interaction_events(
        args.session_id,
        before_sequence=args.before_sequence,
        limit=args.limit + 1,
    )
    has_more = len(records) > args.limit
    page = records[: args.limit]
    rows = [_interaction_row(record) for record in page]
    next_sequence = page[-1].sequence if has_more and page else None
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


def _interaction_row(record: EventRecord) -> dict[str, Any]:
    event = record.event
    if event.interaction_id is None:
        raise ValueError("Interaction lifecycle event has no interaction identity.")
    evidence = InteractionSummaryEvidence.model_validate(event.payload)
    terminal = event.type in INTERACTION_TERMINAL_EVENT_TYPES
    return {
        "interaction_id": event.interaction_id,
        "session_id": event.session_id,
        **evidence.model_dump(mode="json"),
        "start_event_sequence": evidence.start_event_sequence or record.sequence,
        "terminal_event_id": event.id if terminal else None,
        "terminal_event_sequence": record.sequence if terminal else None,
        "updated_at": event.timestamp.isoformat(),
    }


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
        interaction_id=args.interaction_id,
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
        "interaction_id": args.interaction_id,
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
            **_aggregate_usage_cli_payload(aggregate_usage),
        },
    }
    _render_usage(
        args.output_format,
        payload,
        page,
        ledger_page,
    )
    return 0


def _aggregate_usage_cli_payload(usage: AggregateUsageMetrics) -> dict[str, str]:
    """Project validated aggregate counters through their lossless JSON representation."""

    if type(usage) is not AggregateUsageMetrics:
        raise TypeError("CLI aggregate usage must be an AggregateUsageMetrics instance.")
    serialized = usage.model_dump(mode="json")
    cache = serialized["cache"]
    return {
        "input_tokens": serialized["input_tokens"],
        "output_tokens": serialized["output_tokens"],
        "total_tokens": serialized["total_tokens"],
        "reasoning_tokens": serialized["reasoning_output_tokens"],
        "cache_read_tokens": cache["read_tokens"],
        "cache_write_tokens": cache["write_tokens"],
        "cached_input_tokens": cache["cached_input_tokens"],
        "uncached_input_tokens": cache["uncached_input_tokens"],
    }


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
            pricing_state = event.payload.get(_USAGE_INSPECTION_PRICING_STATE_KEY)
            ledger["pricing_state"] = (
                pricing_state
                if type(pricing_state) is str and pricing_state in {"priced", "unpriced", "unknown"}
                else "unknown"
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
_TOOL_EXECUTION_TERMINAL_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
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
        interaction_id=args.interaction_id,
    )
    calls = _tool_call_rows(records)
    page = calls[args.offset : args.offset + args.limit]
    next_offset = args.offset + len(page)
    has_more = next_offset < len(calls)
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "session_id": args.session_id,
        "interaction_id": args.interaction_id,
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
        ),
    )
    return 0


def _tool_call_rows(records: list[EventRecord]) -> list[dict[str, Any]]:
    starts: dict[_ToolCallKey, EventRecord] = {}
    terminals: dict[_ToolCallKey, EventRecord] = {}
    approval_requests: dict[_ToolCallKey, EventRecord] = {}
    approval_calls: dict[_ToolCallKey, dict[str, Any]] = {}
    input_requests: dict[_ToolCallKey, EventRecord] = {}
    input_calls: dict[_ToolCallKey, dict[str, Any]] = {}
    decision_records: dict[_ToolCallKey, EventRecord] = {}
    approval_states: dict[_ToolCallKey, str] = {}
    approval_outcomes: dict[_ToolCallKey, str] = {}
    approval_completion_records: dict[_ToolCallKey, EventRecord] = {}
    unregistered_policy_calls: set[_ToolCallKey] = set()
    resolved_approval_keys: set[tuple[str, str, str, str]] = set()
    approval_request_calls: dict[
        tuple[str, str, str, str],
        set[_ToolCallKey],
    ] = {}
    approval_request_rounds: set[tuple[str, str, str]] = set()
    approval_gated_calls: set[_ToolCallKey] = set()
    evidence_conflicts: set[_ToolCallKey] = set()
    approval_decision_conflicts: set[_ToolCallKey] = set()
    execution_terminal_keys: set[_ToolCallKey] = set()
    tool_names: dict[_ToolCallKey, set[str]] = {}
    lifecycle_records: dict[_ToolCallKey, list[EventRecord]] = {}
    approval_scoped_keys: set[_ToolCallKey] = set()

    def retain_tool_name(call_key: _ToolCallKey, value: object) -> None:
        if type(value) is not str or not value or value.strip() != value:
            evidence_conflicts.add(call_key)
            return
        tool_names.setdefault(call_key, set()).add(value)

    def retain_lifecycle_record(call_key: _ToolCallKey, record: EventRecord) -> None:
        lifecycle_records.setdefault(call_key, []).append(record)

    def approval_resolution_key(
        record: EventRecord,
    ) -> tuple[str, str, str, str] | None:
        approval_id = record.event.payload.get("approval_id")
        identity = _tool_event_identity(record)
        if (
            type(approval_id) is not str
            or not approval_id
            or approval_id.strip() != approval_id
            or identity is None
        ):
            return None
        return (
            identity.model_step_id,
            identity.model_attempt_id,
            identity.tool_round_id,
            approval_id,
        )

    def retain_approval_decision_state(
        call_key: _ToolCallKey,
        record: EventRecord,
    ) -> None:
        if record.event.type == EventType.TOOL_CALL_APPROVED:
            approval_states[call_key] = "approved"
            return
        if record.event.type in {
            EventType.TOOL_CALL_APPROVAL_DENIED,
            EventType.TOOL_CALL_APPROVAL_EXPIRED,
        }:
            outcome = (
                "expired"
                if record.event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED
                or record.event.payload.get("expired") is True
                else "denied"
            )
            approval_states[call_key] = outcome
            approval_outcomes[call_key] = outcome
            return
        if _is_ambiguous_approval_block(record):
            approval_states[call_key] = "blocked"

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
                call_key = _tool_call_key(record, call_id)
                if call_key in input_requests:
                    evidence_conflicts.add(call_key)
                input_requests.setdefault(call_key, record)
                input_calls.setdefault(call_key, call)
                retain_tool_name(call_key, call.get("tool_name"))
                if call_id == event.payload.get("tool_call_id"):
                    retain_tool_name(call_key, event.tool_name)
            continue
        if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED:
            approval = _tool_approval_payload(record)
            if approval is None:
                continue
            nested_calls = approval.get("tool_calls")
            has_nested_calls = type(nested_calls) is list and bool(nested_calls)
            calls = (
                [item for item in nested_calls if type(item) is dict]
                if has_nested_calls
                else [approval]
            )
            for call in calls:
                call_id = call.get("tool_call_id")
                if type(call_id) is not str:
                    continue
                call_key = _tool_call_key(record, call_id)
                if call_key in approval_requests:
                    evidence_conflicts.add(call_key)
                    approval_decision_conflicts.add(call_key)
                approval_requests.setdefault(call_key, record)
                approval_calls.setdefault(call_key, call)
                resolution_key = approval_resolution_key(record)
                if resolution_key is None:
                    evidence_conflicts.add(call_key)
                    approval_decision_conflicts.add(call_key)
                else:
                    approval_request_calls.setdefault(resolution_key, set()).add(call_key)
                    approval_request_rounds.add(resolution_key[:3])
                policy_decision = call.get("policy_decision")
                policy_evidence = call.get("policy_evidence")
                is_ambiguous_call = policy_evidence == "ambiguous" and policy_decision is None
                is_unregistered_call = policy_evidence == "unregistered" and policy_decision is None
                if is_unregistered_call:
                    unregistered_policy_calls.add(call_key)
                if (
                    not has_nested_calls
                    or policy_decision == ToolPolicyDecision.REQUIRE_APPROVAL.value
                    or (is_ambiguous_call and call_id == approval.get("tool_call_id"))
                ):
                    approval_states[call_key] = "requested"
                    approval_gated_calls.add(call_key)
                elif (
                    policy_decision
                    not in {
                        ToolPolicyDecision.ALLOW.value,
                        ToolPolicyDecision.DENY.value,
                    }
                    and not is_ambiguous_call
                    and not is_unregistered_call
                ):
                    evidence_conflicts.add(call_key)
                    approval_decision_conflicts.add(call_key)
                approval_scoped_keys.add(call_key)
                retain_lifecycle_record(call_key, record)
                retain_tool_name(call_key, call.get("tool_name"))
                if call_id == approval.get("tool_call_id"):
                    retain_tool_name(call_key, event.tool_name)
            continue
        call_id = _tool_event_call_id(record)
        if call_id is None:
            continue
        call_key = _tool_call_key(record, call_id)
        retain_lifecycle_record(call_key, record)
        retain_tool_name(call_key, event.tool_name)
        if event.type == EventType.TOOL_CALL_STARTED:
            if call_key in starts or call_key in terminals:
                evidence_conflicts.add(call_key)
            starts.setdefault(call_key, record)
        elif event.type in _TOOL_TERMINAL_TYPES:
            if event.type in _TOOL_EXECUTION_TERMINAL_TYPES:
                execution_terminal_keys.add(call_key)
            if call_key in terminals:
                evidence_conflicts.add(call_key)
            terminals.setdefault(call_key, record)
            if _is_ambiguous_approval_block_candidate(record):
                approval_scoped_keys.add(call_key)
                if call_key in decision_records:
                    evidence_conflicts.add(call_key)
                    approval_decision_conflicts.add(call_key)
                decision_records.setdefault(call_key, record)
        elif event.type == EventType.TOOL_CALL_APPROVED:
            approval_scoped_keys.add(call_key)
            if call_key in decision_records:
                evidence_conflicts.add(call_key)
                approval_decision_conflicts.add(call_key)
            decision_records.setdefault(call_key, record)
        elif event.type == EventType.TOOL_CALL_APPROVAL_DENIED:
            approval_scoped_keys.add(call_key)
            prior_decision = decision_records.get(call_key)
            if prior_decision is not None and not _approval_denial_completes_expiry(
                prior_decision, record
            ):
                evidence_conflicts.add(call_key)
                approval_decision_conflicts.add(call_key)
            decision_records.setdefault(call_key, record)
        elif event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED:
            approval_scoped_keys.add(call_key)
            if call_key in decision_records:
                evidence_conflicts.add(call_key)
                approval_decision_conflicts.add(call_key)
            decision_records.setdefault(call_key, record)

    for call_key, decision_record in decision_records.items():
        resolution_key = approval_resolution_key(decision_record)
        if resolution_key is None:
            evidence_conflicts.add(call_key)
            approval_decision_conflicts.add(call_key)
            continue
        request_calls = approval_request_calls.get(resolution_key)
        if request_calls is None:
            if resolution_key[:3] in approval_request_rounds:
                evidence_conflicts.add(call_key)
                approval_decision_conflicts.add(call_key)
                continue
            if decision_record.event.type is EventType.TOOL_CALL_BLOCKED:
                # Every ambiguous sibling carries the round-level requested
                # decision. Without the request, the gating call cannot be
                # identified positively, so retain only the terminal block.
                continue
            # A bounded --after-sequence window may intentionally exclude the
            # request. Retain the decision's local state without inventing a
            # call-level join to evidence that is not present.
            retain_approval_decision_state(call_key, decision_record)
            resolved_approval_keys.add(resolution_key)
            continue
        if call_key not in request_calls:
            evidence_conflicts.add(call_key)
            approval_decision_conflicts.add(call_key)
            continue
        if call_key in approval_gated_calls:
            if decision_record.event.type is EventType.TOOL_CALL_BLOCKED:
                approval_call = approval_calls.get(call_key)
                if (
                    not _is_ambiguous_approval_block(decision_record)
                    or approval_call is None
                    or approval_call.get("policy_evidence") != "ambiguous"
                    or approval_call.get("policy_decision") is not None
                ):
                    evidence_conflicts.add(call_key)
                    approval_decision_conflicts.add(call_key)
                    continue
                retain_approval_decision_state(call_key, decision_record)
                resolved_approval_keys.add(resolution_key)
                continue
            if (
                decision_record.event.type == EventType.TOOL_CALL_APPROVAL_DENIED
                and decision_record.event.payload.get("approval_required") is not True
            ):
                evidence_conflicts.add(call_key)
                approval_decision_conflicts.add(call_key)
                continue
            retain_approval_decision_state(call_key, decision_record)
            resolved_approval_keys.add(resolution_key)
            continue

        approval_call = approval_calls.get(call_key)
        policy_decision = None if approval_call is None else approval_call.get("policy_decision")
        policy_evidence = None if approval_call is None else approval_call.get("policy_evidence")
        if (
            _is_ambiguous_approval_block(decision_record)
            and approval_call is not None
            and approval_call.get("policy_evidence") == "ambiguous"
            and policy_decision is None
        ):
            # The round-level acknowledgement belongs only to the gating call.
            # Ambiguous siblings still close as blocked terminal tool calls.
            continue
        if (
            decision_record.event.type == EventType.TOOL_CALL_APPROVAL_DENIED
            and (
                policy_decision == ToolPolicyDecision.ALLOW.value
                or (policy_evidence in {"ambiguous", "unregistered"} and policy_decision is None)
            )
            and decision_record.event.payload.get("approval_required") is False
        ):
            approval_outcomes[call_key] = (
                "expired" if decision_record.event.payload.get("expired") is True else "denied"
            )
            approval_completion_records[call_key] = decision_record
            continue
        evidence_conflicts.add(call_key)
        approval_decision_conflicts.add(call_key)

    for call_key in unregistered_policy_calls:
        started = starts.get(call_key)
        terminal = terminals.get(call_key)
        decision = decision_records.get(call_key)
        valid_failed_closure = (
            terminal is not None
            and terminal.event.type is EventType.TOOL_CALL_FAILED
            and terminal.event.payload.get("registration_state") == "unregistered_at_policy_plan"
        )
        valid_denied_closure = (
            decision is not None
            and decision.event.type is EventType.TOOL_CALL_APPROVAL_DENIED
            and decision.event.payload.get("approval_required") is False
        )
        if (
            started is not None
            or (terminal is not None and not valid_failed_closure)
            or (decision is not None and not valid_denied_closure)
            or (terminal is not None and decision is not None)
        ):
            evidence_conflicts.add(call_key)
            approval_decision_conflicts.add(call_key)

    for call_key in approval_outcomes.keys() & execution_terminal_keys:
        evidence_conflicts.add(call_key)
        approval_decision_conflicts.add(call_key)

    for call_key, names in tool_names.items():
        if len(names) != 1:
            evidence_conflicts.add(call_key)
    for call_key, evidence in lifecycle_records.items():
        if call_key not in approval_scoped_keys and not any(
            "approval_id" in record.event.payload for record in evidence
        ):
            continue
        approval_ids: set[str] = set()
        invalid_approval_identity = False
        for record in evidence:
            raw_approval_id = record.event.payload.get("approval_id")
            if (
                type(raw_approval_id) is not str
                or not raw_approval_id
                or raw_approval_id.strip() != raw_approval_id
            ):
                invalid_approval_identity = True
                continue
            approval_ids.add(raw_approval_id)
        if invalid_approval_identity or len(approval_ids) != 1:
            evidence_conflicts.add(call_key)
            approval_decision_conflicts.add(call_key)

    call_keys = (
        starts.keys()
        | terminals.keys()
        | approval_requests.keys()
        | input_requests.keys()
        | decision_records.keys()
    )
    evidence_conflicts.update(_contradictory_tool_call_keys(call_keys))
    anchors = {
        call_key: (
            starts.get(call_key)
            or approval_requests.get(call_key)
            or input_requests.get(call_key)
            or terminals.get(call_key)
            or decision_records[call_key]
        )
        for call_key in call_keys
    }
    round_widths: dict[tuple[str, str, str], int] = {}
    for model_step_id, model_attempt_id, round_id, _, invalid_event_id in call_keys:
        if round_id is None or invalid_event_id is not None:
            continue
        round_key = (model_step_id or "", model_attempt_id or "", round_id)
        round_widths[round_key] = round_widths.get(round_key, 0) + 1

    rows: list[dict[str, Any]] = []
    ordered_call_keys = sorted(
        call_keys,
        key=lambda call_key: (
            anchors[call_key].sequence,
            call_key[3],
            anchors[call_key].event.id,
        ),
    )
    for call_key in ordered_call_keys:
        model_step_id, model_attempt_id, round_id, call_id, invalid_event_id = call_key
        started = starts.get(call_key)
        approval_request = approval_requests.get(call_key)
        approval_call = approval_calls.get(call_key)
        input_request = input_requests.get(call_key)
        input_call = input_calls.get(call_key)
        anchor = anchors[call_key]
        evidence_conflict = call_key in evidence_conflicts
        terminal = None if evidence_conflict else terminals.get(call_key)
        completion_record = (
            None if evidence_conflict else terminal or approval_completion_records.get(call_key)
        )
        approval_state = (
            "unavailable"
            if call_key in approval_decision_conflicts
            else approval_states.get(call_key, "none")
        )
        approval_outcome = approval_outcomes.get(call_key)
        approval_id = (
            None if approval_request is None else approval_request.event.payload.get("approval_id")
        )
        approval_key = (
            (
                model_step_id,
                model_attempt_id,
                round_id,
                approval_id,
            )
            if invalid_event_id is None
            and type(model_step_id) is str
            and type(model_attempt_id) is str
            and type(round_id) is str
            and type(approval_id) is str
            else None
        )
        awaiting_approval = approval_request is not None and (
            approval_key is None or approval_key not in resolved_approval_keys
        )
        round_key = (
            None
            if round_id is None or invalid_event_id is not None
            else (model_step_id or "", model_attempt_id or "", round_id)
        )
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
        argument_summary = None
        arguments_state = None
        for candidate_state in (
            None if terminal is None else terminal.event.payload.get("_arguments_state"),
            None if started is None else started.event.payload.get("_arguments_state"),
            None if approval_call is None else approval_call.get("_arguments_state"),
            None if input_call is None else input_call.get("_arguments_state"),
        ):
            if candidate_state in {"finalized", "quarantined", "unavailable"}:
                arguments_state = candidate_state
                break
        for candidate_summary in (
            None if terminal is None else terminal.event.payload.get("_argument_summary"),
            None if started is None else started.event.payload.get("_argument_summary"),
            None if approval_call is None else approval_call.get("_argument_summary"),
            None if input_call is None else input_call.get("_argument_summary"),
        ):
            if type(candidate_summary) is str:
                argument_summary = candidate_summary
                break
        if type(argument_summary) is not str:
            argument_source = (
                terminal.event.payload
                if terminal is not None
                else started.event.payload
                if started is not None
                else approval_call
                if approval_call is not None
                else input_call
                if input_call is not None
                else approval
                if approval is not None
                else {}
            )
            arguments_state, argument_summary = _tool_argument_inspection_projection(
                argument_source
            )
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
                "model_step_id": model_step_id,
                "model_attempt_id": model_attempt_id,
                "tool_round_id": round_id,
                "parallel_round_width": 1 if round_key is None else round_widths[round_key],
                "arguments_state": arguments_state,
                "argument_summary": argument_summary,
                "started_at": (None if started is None else started.event.timestamp.isoformat()),
                "completed_at": (
                    None
                    if completion_record is None
                    else completion_record.event.timestamp.isoformat()
                ),
                "duration_ms": (None if evidence_conflict else _duration_ms(started, terminal)),
                "status": (
                    "unavailable"
                    if evidence_conflict
                    else _tool_status(
                        started,
                        terminal,
                        approval_state,
                        approval_outcome=approval_outcome,
                        awaiting_approval=awaiting_approval,
                        awaiting_input=input_request is not None,
                    )
                ),
                "approval_state": approval_state,
                "rendered_content_bytes": (
                    None
                    if evidence_conflict
                    else _optional_nonnegative_int(inspection_result.get("rendered_content_bytes"))
                    if inspection_result is not None
                    else len(content.encode("utf-8"))
                    if type(content) is str
                    else 0
                ),
                "structured_result_bytes": (
                    None
                    if evidence_conflict
                    else _optional_nonnegative_int(inspection_result.get("structured_result_bytes"))
                    if inspection_result is not None
                    else compact_json_utf8_size(structured)
                    if structured is not None
                    else 0
                ),
                "artifact_bytes": (
                    None
                    if evidence_conflict
                    else _optional_nonnegative_int(inspection_result.get("artifact_bytes"))
                    if inspection_result is not None
                    else compact_json_utf8_size(artifacts)
                    if type(artifacts) is list
                    else 0
                ),
                "returned": (
                    None
                    if evidence_conflict
                    else _optional_nonnegative_int(inspection_result.get("returned"))
                    if inspection_result is not None
                    else None
                    if structured is None
                    else _optional_nonnegative_int(structured.get("returned"))
                ),
                "truncated": (
                    None
                    if evidence_conflict
                    else inspection_result.get("truncated")
                    if inspection_result is not None
                    and type(inspection_result.get("truncated")) is bool
                    else structured.get("truncated")
                    if structured is not None and type(structured.get("truncated")) is bool
                    else None
                ),
            }
        )
    return rows


_ToolCallKey = tuple[str | None, str | None, str | None, str, str | None]


def _tool_call_key(record: EventRecord, call_id: str) -> _ToolCallKey:
    """Return an exact round/call join key, isolating malformed evidence."""

    if record.event.payload.get(_TOOL_INSPECTION_IDENTITY_CONFLICT_KEY) is True:
        return None, None, None, call_id, record.event.id
    identity = _tool_event_identity(record)
    if identity is None:
        return None, None, None, call_id, record.event.id
    return (
        identity.model_step_id,
        identity.model_attempt_id,
        identity.tool_round_id,
        call_id,
        None,
    )


def _contradictory_tool_call_keys(
    call_keys: set[_ToolCallKey],
) -> set[_ToolCallKey]:
    """Return rows whose execution-unit identity cannot be true simultaneously."""

    parents_by_round_id: dict[str, set[tuple[str, str]]] = {}
    rounds_by_attempt_id: dict[str, set[tuple[str, str]]] = {}
    conflicts = {call_key for call_key in call_keys if call_key[4] is not None}
    for model_step_id, model_attempt_id, round_id, _, invalid_event_id in call_keys:
        if (
            invalid_event_id is not None
            or model_step_id is None
            or model_attempt_id is None
            or round_id is None
        ):
            continue
        parents_by_round_id.setdefault(round_id, set()).add((model_step_id, model_attempt_id))
        rounds_by_attempt_id.setdefault(model_attempt_id, set()).add((model_step_id, round_id))

    conflicting_round_ids = {
        round_id for round_id, parents in parents_by_round_id.items() if len(parents) != 1
    }
    conflicting_attempt_ids = {
        model_attempt_id
        for model_attempt_id, step_rounds in rounds_by_attempt_id.items()
        if len(step_rounds) != 1
    }
    conflicts.update(
        call_key
        for call_key in call_keys
        if call_key[2] in conflicting_round_ids or call_key[1] in conflicting_attempt_ids
    )
    return conflicts


def _tool_event_identity(record: EventRecord) -> ToolRoundIdentity | None:
    payload = record.event.payload
    try:
        return ToolRoundIdentity.model_validate(
            {
                "model_step_id": payload.get("model_step_id"),
                "model_attempt_id": payload.get("model_attempt_id"),
                "tool_round_id": payload.get("tool_round_id"),
            }
        )
    except (TypeError, ValueError):
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


def _tool_approval_payload(record: EventRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    approval = record.event.payload.get("approval")
    return approval if type(approval) is dict else None


def _tool_approval_identity_matches_event(record: EventRecord) -> bool:
    """Validate the duplicated approval identity before bounded projection."""

    approval = _tool_approval_payload(record)
    if approval is None:
        return False
    identity_keys = (
        "approval_id",
        "tool_call_id",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
    )
    for key in identity_keys:
        event_value = record.event.payload.get(key)
        approval_value = approval.get(key)
        if (
            type(event_value) is not str
            or type(approval_value) is not str
            or event_value != approval_value
        ):
            return False
    return True


def _approval_denial_completes_expiry(
    prior_decision: EventRecord,
    denial: EventRecord,
) -> bool:
    """Recognize the runtime's expected expiry marker followed by denial."""

    approval_id = prior_decision.event.payload.get("approval_id")
    return (
        prior_decision.event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED
        and denial.event.type == EventType.TOOL_CALL_APPROVAL_DENIED
        and denial.event.payload.get("expired") is True
        and type(approval_id) is str
        and denial.event.payload.get("approval_id") == approval_id
    )


def _is_ambiguous_approval_block_candidate(record: EventRecord) -> bool:
    """Return whether a block claims to close an ambiguous approval."""

    payload = record.event.payload
    return record.event.type is EventType.TOOL_CALL_BLOCKED and (
        "requested_decision" in payload
        or payload.get("blocked_by") == "policy_evaluation_ambiguous"
        or payload.get("decision") == "ambiguous"
    )


def _is_ambiguous_approval_block(record: EventRecord) -> bool:
    """Recognize the runtime's exact non-authorizing approval resolution."""

    payload = record.event.payload
    return (
        record.event.type is EventType.TOOL_CALL_BLOCKED
        and payload.get("decision") == "ambiguous"
        and payload.get("blocked_by") == "policy_evaluation_ambiguous"
        and payload.get("requested_decision") == "approve"
    )


def _tool_status(
    started: EventRecord | None,
    terminal: EventRecord | None,
    approval_state: str,
    *,
    approval_outcome: str | None,
    awaiting_approval: bool,
    awaiting_input: bool,
) -> str:
    if terminal is None:
        if started is None and awaiting_input:
            return "awaiting_input"
        if started is None and approval_outcome in {"denied", "expired"}:
            return approval_outcome
        if started is None and approval_state in {"denied", "expired"}:
            return approval_state
        if started is None and (awaiting_approval or approval_state == "requested"):
            return "approval_pending"
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


async def _session_transcript(args: argparse.Namespace, store: SessionStore) -> int:
    await _require_session(store, args.session_id)
    page = await store.query_transcript(
        TranscriptQuery(
            session_id=args.session_id,
            interaction_id=args.interaction_id,
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
        "interaction_id": record.interaction_id,
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
        elif kind == "citation":
            previews.append(f"[citation: {part['url']}]")
        elif kind == "hosted_tool_call":
            action = part.get("action")
            source_count = len(action.get("sources", [])) if isinstance(action, dict) else 0
            previews.append(f"[hosted web search {part['status']}: {source_count} sources]")
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
    interaction_id: str | None = None,
) -> list[EventRecord]:
    records: list[EventRecord] = []
    retained_bytes = 0
    cursor = 0 if after_sequence is None else after_sequence
    while True:
        page = await store.query_events(
            EventQuery(
                session_id=session_id,
                interaction_id=interaction_id,
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
            metrics = summary_usage_metrics_from_event_payload(event.payload)
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
        pricing = event.payload.get("pricing")
        if pricing is None:
            payload[_USAGE_INSPECTION_PRICING_STATE_KEY] = "unpriced"
        elif is_complete_budget_reconciliation_pricing(pricing):
            payload[_USAGE_INSPECTION_PRICING_STATE_KEY] = "priced"
        else:
            payload[_USAGE_INSPECTION_PRICING_STATE_KEY] = "unknown"
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
    for key in (
        "tool_call_id",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "approval_id",
        "input_id",
        "approval_required",
        "registration_state",
    ):
        if key in event.payload:
            payload[key] = event.payload[key]
    if event.type == EventType.TOOL_CALL_STARTED:
        state, summary = _tool_argument_inspection_projection(event.payload)
        payload["_arguments_state"] = state
        payload["_argument_summary"] = summary
    if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED:
        approval = _tool_approval_payload(record)
        if approval is not None:
            if not _tool_approval_identity_matches_event(record):
                payload[_TOOL_INSPECTION_IDENTITY_CONFLICT_KEY] = True
            compact_approval = {
                key: approval[key]
                for key in ("approval_id", "tool_call_id", "tool_name")
                if key in approval
            }
            state, summary = _tool_argument_inspection_projection(approval)
            compact_approval["_arguments_state"] = state
            compact_approval["_argument_summary"] = summary
            nested_calls = approval.get("tool_calls")
            if type(nested_calls) is list:
                compact_approval["tool_calls"] = [
                    {
                        key: item[key]
                        for key in (
                            "tool_call_id",
                            "tool_name",
                            "policy_decision",
                            "policy_evidence",
                        )
                        if key in item
                    }
                    | _tool_argument_inspection_fields(item)
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
                | _tool_argument_inspection_fields(item)
                for item in nested_calls
                if type(item) is dict
            ]
    if (
        event.type == EventType.TOOL_CALL_APPROVAL_DENIED
        and type(event.payload.get("expired")) is bool
    ):
        payload["expired"] = event.payload["expired"]
    if event.type == EventType.TOOL_CALL_BLOCKED:
        for key in ("decision", "blocked_by", "requested_decision"):
            if key in event.payload:
                payload[key] = event.payload[key]
    if event.type in _TOOL_TERMINAL_TYPES:
        if "arguments_state" in event.payload or "arguments" in event.payload:
            state, summary = _tool_argument_inspection_projection(event.payload)
            payload["_arguments_state"] = state
            payload["_argument_summary"] = summary
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


def _tool_argument_inspection_projection(value: Any) -> tuple[str | None, str]:
    if type(value) is not dict:
        return "unavailable", "[unavailable]"
    state = value.get("arguments_state")
    if state == "quarantined":
        return state, "[quarantined]"
    if state == "unavailable":
        return state, "[unavailable]"
    arguments = value.get("arguments")
    if state == "finalized":
        if type(arguments) is dict:
            return state, _bounded_argument_summary(arguments)
        return "unavailable", "[unavailable]"
    if state is not None:
        return "unavailable", "[unavailable]"
    if type(arguments) is dict:
        return None, _bounded_argument_summary(arguments)
    return "unavailable", "[unavailable]"


def _tool_argument_inspection_fields(value: Any) -> dict[str, str | None]:
    state, summary = _tool_argument_inspection_projection(value)
    return {
        "_arguments_state": state,
        "_argument_summary": summary,
    }


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


def _datetime_argument(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timestamp must use ISO 8601 syntax.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone offset.")
    return parsed
