"""One read-only seam that `cayu session` commands read sessions through.

A `SessionReader` answers the handful of reads the inspection commands need. `StoreSessionReader`
reads a local SQLite or PostgreSQL store through the `SessionStore` contract;
`AgentSessionReader` (`cayu.cli._agent_session_reader`) reads a deployed Agent's
`/api/sessions/...` endpoints. The commands build their tables from what a reader returns, so a
local store and a Cloud Agent show the same rows.

Records cross the seam as Cayu's own types where the HTTP API returns them losslessly (event
records) and as plain JSON shapes otherwise (session rows, transcript messages, the `show`
overview), because the API withholds private parts of messages a store would return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from cayu.budgets.pricing import PriceBook
from cayu.budgets.usage import AggregateUsageMetrics
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime.provider_operations import inspect_provider_operation
from cayu.sessions._call_inspection import (
    InspectionWindow,
    inspect_model_calls,
    inspect_tool_calls,
)
from cayu.sessions._rollup import RollupPage, SessionRollupQuery, rollup_page
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.interactions import INTERACTION_TERMINAL_EVENT_TYPES, InteractionSummaryEvidence
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import EventRecord, Session
from cayu.sessions.store import SessionStore
from cayu.sessions.transcript_queries import TranscriptQuery


@dataclass(frozen=True)
class SessionListPage:
    """One page of `session list` rows, newest activity first unless the query says otherwise."""

    rows: list[dict[str, Any]]
    next_cursor: str | None
    total_count: int | None


@dataclass(frozen=True)
class TranscriptEntry:
    """One transcript message as JSON: `content` is a list of serialized message parts."""

    index: int
    interaction_id: str | None
    role: str
    content: list[dict[str, Any]]


@dataclass(frozen=True)
class TranscriptView:
    entries: list[TranscriptEntry]
    total_records: int


@dataclass(frozen=True)
class InteractionPage:
    rows: list[dict[str, Any]]
    has_more: bool
    next_sequence: int | None


class SessionReader(Protocol):
    """The reads `cayu session` inspection commands make, from any source."""

    async def list_sessions(self, query: SessionQuery) -> SessionListPage: ...

    async def require_session(self, session_id: str) -> None:
        """Raise `ValueError("Session not found: ...")` unless the session exists."""
        ...

    async def overview(self, session_id: str) -> dict[str, Any]: ...

    async def latest_interactions(
        self, session_id: str, *, before_sequence: int | None, limit: int
    ) -> InteractionPage: ...

    async def model_calls(
        self,
        session_id: str,
        window: InspectionWindow,
        *,
        offset: int,
        limit: int,
        pricing: PriceBook | None = None,
    ) -> dict[str, Any]:
        """`inspect_model_calls`: usage (and cost) per model call, unclaimed ledger, aggregate."""
        ...

    async def tool_calls(
        self, session_id: str, window: InspectionWindow, *, offset: int, limit: int
    ) -> dict[str, Any]:
        """`inspect_tool_calls`: paired tool calls without result bodies."""
        ...

    async def rollup_page(self, query: SessionRollupQuery) -> RollupPage:
        """One page of `rollup_page`: groups for the next sessions and where to continue."""
        ...

    async def query_events(self, query: EventQuery) -> list[EventRecord]: ...

    async def query_transcript(self, query: TranscriptQuery) -> TranscriptView: ...

    async def close(self) -> None: ...


def session_list_row(session: Session) -> dict[str, Any]:
    return {
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


def interaction_row(record: EventRecord) -> dict[str, Any]:
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


def aggregate_usage_cli_payload(usage: AggregateUsageMetrics) -> dict[str, str]:
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


class StoreSessionReader:
    """Read a local session store opened read-only."""

    def __init__(self, store: SessionStore) -> None:
        self.store = store

    async def list_sessions(self, query: SessionQuery) -> SessionListPage:
        result = await self.store.list_sessions(query)
        return SessionListPage(
            rows=[session_list_row(session) for session in result.sessions],
            next_cursor=result.next_cursor,
            total_count=result.total_count,
        )

    async def require_session(self, session_id: str) -> None:
        if await self.store.load(session_id) is None:
            raise ValueError(f"Session not found: {session_id}")

    async def overview(self, session_id: str) -> dict[str, Any]:
        try:
            summary = await runtime_checkpoint_session_store(self.store).inspect_summary(session_id)
        except KeyError as exc:
            raise ValueError(f"Session not found: {session_id}") from exc
        identity = summary.session
        provider_operation = await inspect_provider_operation(self.store, session_id)
        return {
            "session": {
                "id": identity.id,
                "parent_session_id": identity.parent_session_id,
                "causal_budget_id": identity.causal_budget_id,
                "agent": identity.agent_name,
                "provider": identity.provider_name,
                "model": identity.model,
                "runtime_name": identity.runtime_name,
                "runtime_version": identity.runtime_version,
                "runtime_build_provenance": identity.runtime_build_provenance.model_dump(
                    mode="json"
                ),
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
            "usage": aggregate_usage_cli_payload(summary.usage.usage),
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

    async def latest_interactions(
        self, session_id: str, *, before_sequence: int | None, limit: int
    ) -> InteractionPage:
        await self.require_session(session_id)
        records = await self.store.query_latest_interaction_events(
            session_id, before_sequence=before_sequence, limit=limit + 1
        )
        has_more = len(records) > limit
        page = records[:limit]
        return InteractionPage(
            rows=[interaction_row(record) for record in page],
            has_more=has_more,
            next_sequence=page[-1].sequence if has_more and page else None,
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
        return await inspect_model_calls(
            self.store, session_id, window, offset=offset, limit=limit, pricing=pricing
        )

    async def tool_calls(
        self, session_id: str, window: InspectionWindow, *, offset: int, limit: int
    ) -> dict[str, Any]:
        return await inspect_tool_calls(self.store, session_id, window, offset=offset, limit=limit)

    async def rollup_page(self, query: SessionRollupQuery) -> RollupPage:
        return await rollup_page(self.store, query)

    async def query_events(self, query: EventQuery) -> list[EventRecord]:
        return await self.store.query_events(query)

    async def query_transcript(self, query: TranscriptQuery) -> TranscriptView:
        page = await self.store.query_transcript(query)
        return TranscriptView(
            entries=[
                TranscriptEntry(
                    index=record.index,
                    interaction_id=record.interaction_id,
                    role=str(record.message.role),
                    content=[part.model_dump(mode="json") for part in record.message.content],
                )
                for record in page.records
            ],
            total_records=page.total_records,
        )

    async def close(self) -> None:
        close = getattr(self.store, "close", None)
        if close is not None:
            await close()
