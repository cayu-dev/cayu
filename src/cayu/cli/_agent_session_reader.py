"""`SessionReader` over a deployed Agent's `/api/sessions/...` endpoints.

The Agent API pages each read with its own bounds (events and transcripts up to 1000 records,
interactions up to 100), filters events by one type at a time and omits a few private fields a
local store reports (`run_epoch`, record byte sizes, queue and budget details). This reader
follows the API's pages to answer the same queries the CLI asks a local store, reads one stream
per requested event type and merges them in sequence order, applies `since`/`until` to events
itself, and leaves what the API doesn't report as `None`.
"""

from __future__ import annotations

import heapq
import json
from collections.abc import Iterator, Mapping
from datetime import datetime
from itertools import islice
from typing import Any, cast
from urllib.parse import quote

from cayu.budgets.pricing import PriceBook
from cayu.budgets.usage import AggregateUsageMetrics
from cayu.cli._cloud_agent_api import CayuApi, CayuApiError, CayuApiNotFound
from cayu.cli._session_reader import (
    InteractionPage,
    SessionListPage,
    TranscriptEntry,
    TranscriptView,
    aggregate_usage_cli_payload,
)
from cayu.events import Event
from cayu.sessions._call_inspection import InspectionWindow
from cayu.sessions._rollup import RollupPage, SessionRollupQuery
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import EventRecord
from cayu.sessions.transcript_queries import TranscriptQuery

_EVENT_PAGE_LIMIT = 1000
_INTERACTION_PAGE_LIMIT = 100
_PENDING_ACTION_PAGE_LIMIT = 200
_MAX_SHOWN_LABELS = 200
_SESSION_NOT_FOUND = "Session not found"


class AgentSessionReader:
    """Read one Cloud Agent's sessions through its Cayu API."""

    def __init__(self, api: CayuApi) -> None:
        self.api = api
        # Set once an Agent refuses a priced rollup page; later pages are asked unpriced.
        self.rollup_pricing_unsupported = False

    async def list_sessions(self, query: SessionQuery) -> SessionListPage:
        params: dict[str, Any] = {
            "limit": query.limit,
            "offset": query.offset or None,
            "cursor": query.cursor,
            "status": None if query.status is None else query.status.value,
            "agent_name": query.agent_name,
            "provider_name": query.provider_name,
            "model": query.model,
            "environment_name": query.environment_name,
            "label": [f"{key}={value}" for key, value in query.labels.items()],
            "order_by": query.order_by.value,
            "created_since": _instant(query.created_since),
            "created_before": _instant(query.created_before),
        }
        windowed = query.created_since is not None or query.created_before is not None
        # An Agent whose Cayu predates the creation-time filter ignores it, so with a window
        # the CLI filters rows itself and applies the offset to the filtered rows.
        skip = query.offset if windowed else 0
        if skip:
            params["offset"] = None
        rows: list[dict[str, Any]] = []
        filtered_here = False
        while True:
            # Ask only for what the page still needs, so no row is read past the cursor.
            params["limit"] = min(query.limit - len(rows) + skip, 1000)
            body = self.api.get("/sessions", query=_query(params), resource="sessions")
            sessions = body.get("sessions")
            if not isinstance(sessions, list):
                raise CayuApiError("not_cayu", "The Agent's session list is invalid.")
            for item in sessions:
                if not isinstance(item, dict):
                    continue
                row = _session_list_row(item)
                if not _created_in_window(row, query):
                    filtered_here = True
                elif skip:
                    skip -= 1
                else:
                    rows.append(row)
            next_cursor = body.get("next_cursor")
            next_cursor = next_cursor if isinstance(next_cursor, str) else None
            if (
                not (filtered_here or (query.offset and windowed))
                or len(rows) >= query.limit
                or next_cursor is None
            ):
                break
            params = {**params, "offset": None, "cursor": next_cursor}
        total_count = body.get("total_count")
        return SessionListPage(
            rows=rows,
            next_cursor=next_cursor,
            # Matching rows don't prove an Agent applied the window, so a windowed total
            # from the Agent is never trusted.
            total_count=None if windowed or type(total_count) is not int else total_count,
        )

    async def require_session(self, session_id: str) -> None:
        self._session_get(session_id)

    async def overview(self, session_id: str) -> dict[str, Any]:
        detail = self._session_get(session_id)
        summary = self._session_get(session_id, "summary")
        state = self._session_get(session_id, "state")
        pending = self._get_supported(
            "/pending-actions",
            {"session_id": session_id, "limit": _PENDING_ACTION_PAGE_LIMIT},
        )
        usage = _mapping(summary.get("usage"))
        labels = _mapping(detail.get("labels"))
        shown_labels = dict(sorted(labels.items())[:_MAX_SHOWN_LABELS])
        actions = [item for item in pending.get("actions", []) if isinstance(item, dict)]
        issues = [item for item in pending.get("issues", []) if isinstance(item, dict)]
        status = state.get("status", detail.get("status"))
        return {
            "session": {
                "id": detail.get("id"),
                "parent_session_id": detail.get("parent_session_id"),
                "causal_budget_id": detail.get("causal_budget_id"),
                "agent": detail.get("agent_name"),
                "provider": detail.get("provider_name"),
                "model": detail.get("model"),
                "runtime_name": detail.get("runtime_name"),
                "runtime_version": detail.get("runtime_version"),
                "runtime_build_provenance": detail.get("runtime_build_provenance"),
                "environment": detail.get("environment_name"),
                "status": status,
                "created_at": _iso(detail.get("created_at")),
                "updated_at": _iso(state.get("updated_at", detail.get("updated_at"))),
                "last_activity_at": _iso(state.get("last_activity_at")),
                "run_epoch": None,
                "labels": shown_labels,
                "label_count": len(labels),
                "labels_truncated": len(labels) > len(shown_labels),
            },
            "transcript": {
                "message_count": _mapping(summary.get("transcript")).get("total_messages"),
                "total_message_bytes": None,
                "largest_message_bytes": None,
            },
            "events": {
                "event_count": _mapping(summary.get("events")).get("total_events"),
                "total_payload_bytes": None,
                "largest_payload_bytes": None,
            },
            "activity": {
                "model_calls": usage.get("model_steps"),
                "model_calls_with_usage": None,
                "tool_calls": usage.get("tool_calls"),
            },
            "usage": aggregate_usage_cli_payload(_aggregate_usage(usage.get("usage") or {})),
            "pending_action": {
                "count": len(actions),
                "kinds": sorted({str(item["kind"]) for item in actions if "kind" in item}),
                "issue_count": len(issues),
            },
            "queue": {"queued": None, "delivered": None, "outstanding": None},
            "operation": {"accepted_event_count": None, "state": None},
            "provider_operation": state.get("provider_operation"),
            "terminal_failure": {
                "state": status if status in {"failed", "interrupted"} else "none"
            },
            "budget": None,
        }

    async def latest_interactions(
        self, session_id: str, *, before_sequence: int | None, limit: int
    ) -> InteractionPage:
        rows: list[dict[str, Any]] = []
        next_sequence = before_sequence
        has_more = True
        while has_more and len(rows) < limit:
            body = self._session_get(
                session_id,
                "interactions",
                params={
                    "before_sequence": next_sequence,
                    "limit": min(_INTERACTION_PAGE_LIMIT, limit - len(rows)),
                },
            )
            page = [
                {**item, "updated_at": _iso(item.get("updated_at"))}
                for item in body.get("interactions", [])
                if isinstance(item, dict)
            ]
            rows.extend(page)
            raw_next = body.get("next_sequence")
            next_sequence = raw_next if type(raw_next) is int else None
            has_more = body.get("has_more") is True and bool(page) and next_sequence is not None
        return InteractionPage(
            rows=rows, has_more=has_more, next_sequence=next_sequence if has_more else None
        )

    async def model_calls(
        self,
        session_id: str,
        window: InspectionWindow,
        *,
        offset: int,
        limit: int,
        pricing: PriceBook | None = None,
    ) -> dict[str, Any]:
        params = _window_params(window, offset, limit)
        body = None
        if pricing is not None:
            try:
                body = self._session_post(
                    session_id,
                    "model-calls",
                    params=params,
                    body={"pricing": pricing.model_dump(mode="json")},
                )
            except CayuApiError as exc:
                # An Agent whose Cayu predates pricing here serves only GET; its calls
                # come back without cost.
                if exc.status_code != 405:
                    raise
        if body is None:
            body = self._session_get(session_id, "model-calls", params=params)
        aggregate = _mapping(body.get("aggregate"))
        return {
            **body,
            "aggregate": {**aggregate, "usage": _aggregate_usage(aggregate.get("usage"))},
        }

    async def tool_calls(
        self, session_id: str, window: InspectionWindow, *, offset: int, limit: int
    ) -> dict[str, Any]:
        return self._session_get(
            session_id, "tool-calls", params=_window_params(window, offset, limit)
        )

    async def rollup_page(self, query: SessionRollupQuery) -> RollupPage:
        if self.rollup_pricing_unsupported:
            query = query.model_copy(update={"pricing": None})
        try:
            try:
                body = self.api.post(
                    "/sessions/rollup", _rollup_body(query), resource="session rollups"
                )
            except CayuApiError as exc:
                # An Agent whose Cayu predates priced rollups rejects the `pricing` field;
                # its pages come back without cost.
                if query.pricing is None or exc.status_code not in {400, 422}:
                    raise
                self.rollup_pricing_unsupported = True
                query = query.model_copy(update={"pricing": None})
                body = self.api.post(
                    "/sessions/rollup", _rollup_body(query), resource="session rollups"
                )
        except CayuApiError as exc:
            if exc.category == "unsupported" or exc.status_code in {404, 405}:
                raise CayuApiError(
                    "unsupported",
                    "This Agent's Cayu doesn't serve session rollups yet; redeploy it with a "
                    "newer cayu release, or run `cayu session rollup --sqlite PATH` on a copy "
                    "of its session store.",
                    status_code=exc.status_code,
                ) from None
            raise
        try:
            return RollupPage.model_validate_json(json.dumps(body))
        except ValueError as exc:
            raise CayuApiError("not_cayu", "The Agent's session rollup is invalid.") from exc

    async def query_events(self, query: EventQuery) -> list[EventRecord]:
        if query.order_by is not EventOrder.SEQUENCE_ASC:
            raise ValueError("Cloud Agent event reads are in ascending sequence order.")
        if query.session_id is None:
            raise ValueError("Cloud Agent event reads name one session.")
        types: tuple[str | None, ...] = tuple(str(item) for item in query.event_types) or (
            (str(query.event_type),) if query.event_type is not None else (None,)
        )
        if len(types) > 1 and query.before_sequence is None:
            # Separate per-type reads of a live session must share one upper bound, or an
            # event appended between them could fall behind the merged page's cursor.
            latest = self._session_get(
                query.session_id,
                "events",
                params={"order_by": EventOrder.SEQUENCE_DESC.value, "limit": 1},
            )
            newest = [item for item in latest.get("events", []) if isinstance(item, dict)]
            if not newest:
                return []
            query = query.model_copy(update={"before_sequence": int(newest[0]["sequence"]) + 1})
        streams = [self._event_stream(query, event_type) for event_type in types]
        merged = heapq.merge(*streams, key=lambda record: record.sequence)
        return list(islice(merged, query.limit))

    async def query_transcript(self, query: TranscriptQuery) -> TranscriptView:
        body = self._session_get(
            query.session_id,
            "transcript",
            params={
                "offset": query.offset,
                "limit": query.limit,
                "interaction_id": query.interaction_id,
                "role": None if query.role is None else str(query.role),
            },
        )
        entries = [
            TranscriptEntry(
                index=item["index"],
                interaction_id=item.get("interaction_id"),
                role=str(item.get("role")),
                content=[part for part in item.get("content", []) if isinstance(part, dict)],
            )
            for item in body.get("messages", [])
            if isinstance(item, dict) and type(item.get("index")) is int
        ]
        total = body.get("total_messages")
        return TranscriptView(entries=entries, total_records=total if type(total) is int else 0)

    async def close(self) -> None:
        self.api.close()

    def _event_stream(self, query: EventQuery, event_type: str | None) -> Iterator[EventRecord]:
        cursor = query.after_sequence
        page_size = min(_EVENT_PAGE_LIMIT, query.limit)
        while True:
            body = self._session_get(
                query.session_id or "",
                "events",
                params={
                    "event_type": event_type,
                    "interaction_id": query.interaction_id,
                    "tool_name": query.tool_name,
                    "agent_name": query.agent_name,
                    "environment_name": query.environment_name,
                    "after_sequence": cursor,
                    "before_sequence": query.before_sequence,
                    "order_by": EventOrder.SEQUENCE_ASC.value,
                    "limit": page_size,
                },
            )
            page = [
                _event_record(item) for item in body.get("events", []) if isinstance(item, dict)
            ]
            for record in page:
                timestamp = record.event.timestamp
                if query.since is not None and timestamp < query.since:
                    continue
                if query.until is not None and timestamp >= query.until:
                    continue
                yield record
            if not page or body.get("has_more") is not True:
                return
            cursor = page[-1].sequence

    def _session_get(
        self,
        session_id: str,
        *resource: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = _path("sessions", session_id, *resource)
        try:
            return self.api.get(path, query=_query(params), resource=_resource(path))
        except CayuApiNotFound as exc:
            if _SESSION_NOT_FOUND in str(exc):
                raise ValueError(f"Session not found: {session_id}") from None
            raise _unsupported(path) from None

    def _session_post(
        self,
        session_id: str,
        *resource: str,
        params: dict[str, Any],
        body: dict[str, Any],
    ) -> dict[str, Any]:
        path = _path("sessions", session_id, *resource)
        try:
            return self.api.post(path, body, query=_query(params), resource=_resource(path))
        except CayuApiNotFound as exc:
            if _SESSION_NOT_FOUND in str(exc):
                raise ValueError(f"Session not found: {session_id}") from None
            raise _unsupported(path) from None

    def _get_supported(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.api.get(path, query=_query(params), resource=_resource(path))
        except CayuApiNotFound:
            raise _unsupported(path) from None


def _instant(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _created_in_window(row: dict[str, Any], query: SessionQuery) -> bool:
    if query.created_since is None and query.created_before is None:
        return True
    try:
        created = datetime.fromisoformat(str(row.get("created_at")))
    except ValueError:
        return False
    return (query.created_since is None or created >= query.created_since) and (
        query.created_before is None or created < query.created_before
    )


def _window_params(window: InspectionWindow, offset: int, limit: int) -> dict[str, Any]:
    return {
        "offset": offset,
        "limit": limit,
        "after_sequence": window.after_sequence,
        "before_sequence": window.before_sequence,
        "interaction_id": window.interaction_id,
    }


def _session_list_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "status": item.get("status"),
        "agent": item.get("agent_name"),
        "provider": item.get("provider_name"),
        "model": item.get("model"),
        "environment": item.get("environment_name"),
        "created_at": _iso(item.get("created_at")),
        "updated_at": _iso(item.get("updated_at")),
        "last_activity_at": _iso(item.get("last_activity_at")),
        "run_epoch": item.get("run_epoch"),
    }


def _event_record(item: dict[str, Any]) -> EventRecord:
    sequence = item.get("sequence")
    fields = {key: value for key, value in item.items() if key != "sequence"}
    try:
        return EventRecord.model_validate(
            {"sequence": sequence, "event": Event.model_validate(fields)}
        )
    except ValueError as exc:
        raise CayuApiError(
            "not_cayu", f"The Agent returned an event the CLI can't read (sequence {sequence})."
        ) from exc


def _aggregate_usage(value: object) -> AggregateUsageMetrics:
    # Aggregate counters travel as decimal strings; JSON-mode validation reads them back.
    try:
        return AggregateUsageMetrics.model_validate_json(json.dumps(value))
    except ValueError as exc:
        raise CayuApiError("not_cayu", "The Agent's usage counters are invalid.") from exc


def _iso(value: object) -> object:
    """A timestamp in the form a local store reports it (`+00:00`, not `Z`)."""

    if not isinstance(value, str):
        return value
    try:
        return datetime.fromisoformat(value).isoformat()
    except ValueError:
        return value


def _mapping(value: object) -> dict[str, Any]:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _rollup_body(query: SessionRollupQuery) -> dict[str, Any]:
    """A rollup page request; without pricing the key is left out, since an older Agent
    rejects the field whatever its value."""

    body = query.model_dump(mode="json")
    if query.pricing is None:
        body.pop("pricing", None)
    return body


def _unsupported(path: str) -> CayuApiError:
    return CayuApiError(
        "unsupported",
        f"This Agent's Cayu doesn't serve session {path.split('/')[-1]} yet; "
        "redeploy it with a newer cayu release to read them from the CLI.",
        status_code=404,
    )


def _path(*segments: str) -> str:
    """An API path whose variable segments (such as a session id) are escaped."""

    return "".join("/" + quote(segment, safe="") for segment in segments)


def _resource(path: str) -> str:
    """How messages name what ``path`` reads, for example ``session events``."""

    segments = path.strip("/").split("/")
    return "the session" if len(segments) <= 2 else f"session {segments[-1]}"


def _query(params: Mapping[str, Any] | None) -> dict[str, str | list[str]]:
    """Query parameters as strings, leaving out unset ones."""

    query: dict[str, str | list[str]] = {}
    for key, value in (params or {}).items():
        if value is None:
            continue
        values = list(value) if isinstance(value, list | tuple) else [value]
        strings = ["true" if v is True else "false" if v is False else str(v) for v in values]
        if isinstance(value, list | tuple):
            if strings:
                query[key] = strings
        else:
            query[key] = strings[0]
    return query
