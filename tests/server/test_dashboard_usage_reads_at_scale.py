"""The dashboard's Usage and Tools reads at review.cayu.ai's scale.

review.cayu.ai keeps about 8,000 sessions with about 110,000 tool calls and
80,000 model steps over 30 days. Its Usage page (a 30-day rollup with a daily
series) and Tools page (retained per-tool totals) read the narrow
``cayu_usage_events`` copy rather than whole event rows.

The store is written with plain SQL inserts into ``cayu_events``, so the copy is
filled by the same triggers that maintain it for the Runtime's writes.

What the test holds is which columns the reads touch: SQLite's authorizer reports
every table column a statement reads when it is prepared, and the page reads must
never read an event payload from ``cayu_events``. That catches a return to the old
full-row scans on any runner. Wall time is also bounded, with room for a busy CI
worker: these reads take about 0.4 s (Tools) and 2.5 s (Usage) on a developer
laptop and up to about 3.6 s on CI, while the old full scans took about 10 s
(Usage) and more than two minutes (Tools) on this store.
"""

from __future__ import annotations

import asyncio
import json
import random
import sqlite3
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Event,
    EventType,
    Message,
    RunRequest,
    Tool,
    ToolResult,
    ToolSpec,
)
from cayu.server import ServerConfig, create_server
from cayu.sessions.records import SessionIdentity
from cayu.storage import SQLiteSessionStore
from cayu.storage._sqlite_functions import _register_sqlite_functions
from cayu.storage._sqlite_public_authority import register_sql_functions

SESSIONS = 8_000
TOOL_CALLS = 110_000
MODEL_STEPS = 80_000
DAYS = 30
TOOLS = ("exec_command", "read_file", "search_text", "list_files", "edit_file", "write_file")
MODELS = ("claude-a", "claude-b", "gpt-x")
# A sanity bound, not the dashboard's 3-second target: CI workers share hosts.
PAGE_LIMIT_SECONDS = 15.0


class _NamedTool(Tool):
    def __init__(self, name: str) -> None:
        self.spec = ToolSpec(
            name=name,
            description=f"The {name} tool.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        )
        super().__init__()

    async def run(self, ctx, args) -> ToolResult:
        return ToolResult(content="done")


async def _template(path) -> None:
    store = SQLiteSessionStore(path)
    try:
        await store.create(
            RunRequest(
                session_id="template",
                agent_name="reviewer",
                messages=[Message.text("user", "review")],
            ),
            identity=SessionIdentity(provider_name="anthropic", model="claude-a"),
        )
        for event_type, tool_name in (
            (EventType.MODEL_COMPLETED, None),
            (EventType.TOOL_CALL_STARTED, "exec_command"),
        ):
            await store.append_event(
                "template",
                Event(
                    id=str(event_type),
                    session_id="template",
                    type=event_type,
                    tool_name=tool_name,
                    payload={},
                ),
            )
    finally:
        await store.close()


def _fill(path, end: datetime) -> None:
    """Insert the sessions and events in one transaction, as raw rows."""

    rng = random.Random(3843)
    connection = sqlite3.connect(path, isolation_level=None)
    connection.row_factory = sqlite3.Row
    _register_sqlite_functions(connection)
    register_sql_functions(connection, codec=None)
    session = dict(connection.execute("SELECT * FROM cayu_sessions").fetchone())
    template = dict(connection.execute("SELECT * FROM cayu_events LIMIT 1").fetchone())
    session_columns = list(session)
    event_columns = [column for column in template if column != "sequence"]
    insert_session = (
        f"INSERT INTO cayu_sessions ({', '.join(session_columns)}) "
        f"VALUES ({', '.join('?' * len(session_columns))})"
    )
    insert_event = (
        f"INSERT INTO cayu_events ({', '.join(event_columns)}) "
        f"VALUES ({', '.join('?' * len(event_columns))})"
    )
    content = "x" * 512
    output = "y" * 1024
    start = end - timedelta(days=DAYS)
    span = (end - start).total_seconds() - 3600
    connection.execute("BEGIN")
    connection.execute("DELETE FROM cayu_events")
    connection.execute("DELETE FROM cayu_sessions")
    rows: list[list[object]] = []

    def event(session_id, index, event_type, at, tool_name, payload) -> None:
        row = dict(
            template,
            session_id=session_id,
            event_id=f"{session_id}-{index}",
            interaction_id=None,
            event_type=str(event_type),
            timestamp=at.isoformat(),
            tool_name=tool_name,
            payload_json=json.dumps(payload),
        )
        rows.append([row[column] for column in event_columns])

    for number in range(SESSIONS):
        session_id = f"session-{number:05d}"
        at = start + timedelta(seconds=rng.random() * span)
        connection.execute(
            insert_session,
            [
                dict(
                    session,
                    id=session_id,
                    instance_id=f"instance-{number:05d}",
                    status="completed",
                    created_at=at.isoformat(),
                    updated_at=at.isoformat(),
                    last_activity_at=at.isoformat(),
                )[column]
                for column in session_columns
            ],
        )
        index = 0
        for _ in range(MODEL_STEPS // SESSIONS):
            at += timedelta(seconds=5)
            input_tokens = rng.randint(500, 20_000)
            output_tokens = rng.randint(50, 2_000)
            event(
                session_id,
                index,
                EventType.MODEL_COMPLETED,
                at,
                None,
                {
                    "usage_metrics": {
                        "provider_name": "anthropic",
                        "model": rng.choice(MODELS),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                    "response": content,
                },
            )
            index += 1
        for _ in range(TOOL_CALLS // SESSIONS + (1 if number < TOOL_CALLS % SESSIONS else 0)):
            at += timedelta(seconds=2)
            tool_name = rng.choice(TOOLS)
            event(session_id, index, EventType.TOOL_CALL_STARTED, at, tool_name, {"cmd": content})
            failed = rng.random() < 0.05
            event(
                session_id,
                index + 1,
                EventType.TOOL_CALL_FAILED if failed else EventType.TOOL_CALL_COMPLETED,
                at + timedelta(seconds=1),
                tool_name,
                {"output": output},
            )
            index += 2
        if len(rows) >= 20_000:
            connection.executemany(insert_event, rows)
            rows.clear()
    connection.executemany(insert_event, rows)
    connection.execute("COMMIT")
    connection.close()


@pytest.fixture(scope="module")
def review_scale_store(tmp_path_factory):
    path = tmp_path_factory.mktemp("review-scale") / "sessions.sqlite"
    end = datetime.now(UTC).replace(microsecond=0)
    asyncio.run(_template(path))
    _fill(path, end)
    return path, end


class _ColumnReads:
    """Record every table column the store's connections prepare a read of."""

    def __init__(self, store: SQLiteSessionStore) -> None:
        self.columns: set[tuple[str, str]] = set()
        self.connections = [connection for _, connection in store._readers]
        self.connections.append(store._connection)

    def _authorize(self, action, table, column, database, trigger) -> int:
        if action == sqlite3.SQLITE_READ and table is not None:
            self.columns.add((table, column or ""))
        return sqlite3.SQLITE_OK

    def __enter__(self) -> _ColumnReads:
        for connection in self.connections:
            connection.set_authorizer(self._authorize)
        return self

    def __exit__(self, *exc_info) -> None:
        for connection in self.connections:
            connection.set_authorizer(None)

    def tables(self) -> set[str]:
        return {table for table, _ in self.columns}


async def _timed(client: httpx.AsyncClient, method: str, url: str, body=None):
    timings = []
    for _ in range(2):
        began = time.perf_counter()
        response = await client.request(method, url, json=body)
        timings.append(time.perf_counter() - began)
        assert response.status_code == 200, response.text
    return response.json(), min(timings)


def test_usage_and_tools_pages_read_only_the_narrow_copy_at_review_scale(review_scale_store):
    path, end = review_scale_store

    async def run():
        store = SQLiteSessionStore(path)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="reviewer", model="model"),
            tools=[_NamedTool(name) for name in TOOLS],
        )
        server = create_server(app, config=ServerConfig.local_development())
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test", timeout=600
            ) as client:
                with _ColumnReads(store) as tools_reads:
                    tools, tools_seconds = await _timed(client, "GET", "/api/tools/usage")
                with _ColumnReads(store) as rollup_reads:
                    rollup, rollup_seconds = await _timed(
                        client,
                        "POST",
                        "/api/usage/rollup",
                        {
                            "start_at": (end - timedelta(days=DAYS)).isoformat(),
                            "end_at": end.isoformat(),
                            "session_filter": {},
                            "group_limit": 20,
                            "pricing_input_limit": 1000,
                            "session_group_limit": 10,
                            "series": {"unit": "day", "timezone": "America/Los_Angeles"},
                        },
                    )
        finally:
            await app.aclose()
        return tools, tools_seconds, tools_reads, rollup, rollup_seconds, rollup_reads

    tools, tools_seconds, tools_reads, rollup, rollup_seconds, rollup_reads = asyncio.run(run())
    print(f"\nTools page read: {tools_seconds:.2f} s; Usage page rollup: {rollup_seconds:.2f} s")
    assert {row["name"] for row in tools["tools"]} == set(TOOLS)
    assert sum(row["calls"] for row in tools["tools"]) == TOOL_CALLS
    assert tools["through_sequence"] == MODEL_STEPS + 2 * TOOL_CALLS + 2
    assert int(rollup["matching_session_count"]) == SESSIONS
    assert int(rollup["totals"]["model_steps"]) == MODEL_STEPS
    assert int(rollup["totals"]["tool_calls"]) == TOOL_CALLS
    assert len(rollup["series"]["buckets"]) in (DAYS, DAYS + 1)
    for reads in (tools_reads, rollup_reads):
        assert "cayu_usage_events" in reads.tables(), sorted(reads.columns)
        assert ("cayu_events", "payload_json") not in reads.columns, sorted(reads.columns)
    assert tools_seconds < PAGE_LIMIT_SECONDS
    assert rollup_seconds < PAGE_LIMIT_SECONDS


def test_store_tallies_every_tool_call_at_review_scale(review_scale_store):
    path, _ = review_scale_store

    async def run():
        store = SQLiteSessionStore(path)
        try:
            timings = []
            for _ in range(2):
                began = time.perf_counter()
                tallies, _ = await store.tool_usage_totals()
                timings.append(time.perf_counter() - began)
            return tallies, min(timings)
        finally:
            await store.close()

    tallies, seconds = asyncio.run(run())
    assert set(tallies) == set(TOOLS)
    assert sum(tally.calls for tally in tallies.values()) == TOOL_CALLS
    assert 0 < sum(tally.failures for tally in tallies.values()) < TOOL_CALLS
    assert seconds < PAGE_LIMIT_SECONDS
