"""Time the Developer dashboard's session reads on a SQLite store of production size.

Run with ``uv run python scripts/benchmark_session_overview_reads.py``. It builds a
store shaped like a busy deployment (by default 8,000 sessions over 60 days, each
with five model steps, four tool calls, a transcript and a checkpoint), then times
``GET /api/sessions/overview/counts`` for the last 24 hours and
``GET /api/sessions?include=overview`` for a page of 100 sessions through the real
server. Pass ``--keep PATH`` to keep the store and ``--store PATH`` to reuse one.

A few template sessions run through the real runtime and are then copied with
SQL, with their IDs, incarnations, sequences and times rewritten, so building the
store takes seconds rather than the minutes that thousands of runs would take.
The copy is for measurement only; it is not a supported way to write a store.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import statistics
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from cayu import AgentSpec, CayuApp, Message, RunRequest, Tool, ToolResult, ToolSpec
from cayu.budgets.pricing import default_price_book
from cayu.evals import ScriptedModelProvider
from cayu.messages import MessageRole
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.server import ServerConfig, create_server
from cayu.server.config import DashboardConfig
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.storage import SQLiteSessionStore
from cayu.storage._sqlite_catalog import _iter_statements
from cayu.storage._usage_projection_schema import (
    PROJECTION_TABLES,
    SQLITE_USAGE_PROJECTION_REBUILD,
)
from cayu.tools.base import ToolContext

SESSIONS = 8_000
DAYS = 60
TEMPLATES = 4
RUNNING = 3
MODEL_STEPS = 5
PAGE = 100

_PRICE = next(
    price
    for price in default_price_book().prices
    if price.provider_name == "anthropic" and price.match == "exact"
)
_USAGE = {"input_tokens": 12_000, "output_tokens": 800}
# Realistic payload weight: tool output and the first user message dominate.
_TOOL_OUTPUT = "diff --git a/src/module.py b/src/module.py\n" + "+    changed line\n" * 300
_PROMPT = "Review this pull request.\n\n" + "Context line for the reviewer.\n" * 60


class _Inspect(Tool):
    spec = ToolSpec(
        name="inspect",
        description="Read part of the change under review.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content=_TOOL_OUTPUT)


def _respond(request: ModelRequest) -> list[ModelStreamEvent]:
    tool_rounds = sum(message.role is MessageRole.TOOL for message in request.messages)
    if tool_rounds >= MODEL_STEPS - 1:
        return [
            ModelStreamEvent.text_delta("No issues found. " * 40),
            ModelStreamEvent.completed({"finish_reason": "stop", "usage": _USAGE}),
        ]
    return [
        ModelStreamEvent.text_delta("Looking at the next file. " * 10),
        ModelStreamEvent.tool_call(
            id=f"call-{tool_rounds}", name="inspect", arguments={"path": f"src/{tool_rounds}.py"}
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls", "usage": _USAGE}),
    ]


def _app(store: SQLiteSessionStore) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(name=_PRICE.provider_name, response_factory=_respond), default=True
    )
    app.register_agent(AgentSpec(name="reviewer", model=_PRICE.model), tools=[_Inspect()])
    return app


def _template_id(index: int) -> str:
    return f"benchtemplate{index:04d}"


def _clone_id(index: int) -> str:
    return f"review-{index:05d}"


async def _run_templates(path: Path, templates: int) -> None:
    store = SQLiteSessionStore(path)
    try:
        app = _app(store)
        for index in range(templates):
            stream = app.run(
                RunRequest(
                    agent_name="reviewer",
                    session_id=_template_id(index),
                    title=f"Review change {index}",
                    labels={"repo": "example", "pr": str(index)},
                    messages=[Message.text("user", _PROMPT)],
                )
            )
            async for _event in stream:
                pass
    finally:
        await store.close()


def _shift(value: Any, seconds: float) -> Any:
    if type(value) is not str:
        return value
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    if parsed.tzinfo is None:
        return value
    return (parsed + timedelta(seconds=seconds)).astimezone(UTC).isoformat()


# Per-session tables none of the measured reads touch: the runtime's operation
# records and event delivery bookkeeping. Leaving them out halves the copy.
_UNREAD_TABLES = frozenset({"cayu_session_operations", "cayu_persisted_event_side_effects"})


@dataclass(frozen=True)
class _Table:
    name: str
    key: str
    columns: tuple[str, ...]
    integer_key: str | None
    times: frozenset[str]
    texts: frozenset[str]


def _unique_text_columns(connection: sqlite3.Connection, table: str, key: str) -> set[str]:
    """Text columns of a unique index that a copy under a new session ID would repeat."""

    columns: set[str] = set()
    for index in connection.execute(f'PRAGMA index_list("{table}")').fetchall():
        if not index[2]:
            continue
        indexed = {row[2] for row in connection.execute(f'PRAGMA index_info("{index[1]}")')}
        if key not in indexed and None not in indexed:
            columns |= indexed
    return columns


def _session_tables(connection: sqlite3.Connection) -> list[_Table]:
    tables = []
    for (name,) in connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'cayu_%' "
        "AND name NOT LIKE '%_fts%'"
    ).fetchall():
        info = connection.execute(f'PRAGMA table_info("{name}")').fetchall()
        columns = tuple(row[1] for row in info)
        if name in _UNREAD_TABLES or name in PROJECTION_TABLES:
            continue
        if name == "cayu_sessions":
            key = "id"
        elif "session_id" in columns:
            key = "session_id"
        else:
            continue
        keys = [row for row in info if row[5]]
        integer_key = (
            keys[0][1] if len(keys) == 1 and str(keys[0][2]).upper() == "INTEGER" else None
        )
        times = frozenset(c for c in columns if c.endswith("_at") or c == "timestamp")
        texts = frozenset(
            row[1] for row in info if str(row[2]).upper() in {"TEXT", ""} and row[1] not in times
        )
        tables.append(_Table(name, key, columns, integer_key, times, texts))
    return tables


def _renamed_columns(connection: sqlite3.Connection, tables: list[_Table]) -> frozenset[str]:
    """Columns whose values get a per-copy suffix, by name in every table alike."""

    names: set[str] = set()
    for table in tables:
        names |= _unique_text_columns(connection, table.name, table.key) & table.texts
    return frozenset(names - {"instance_id", "id", "session_id"})


def _clone(path: Path, *, sessions: int, templates: int, days: int, now: datetime) -> None:
    """Copy the template sessions' rows into ``sessions`` sessions spread over ``days``.

    The store's triggers guard runtime-owned writes, so they are set aside for
    the copy and restored unchanged. Every table the triggers maintain is copied
    as rows, except full-text indexes and ``_UNREAD_TABLES``, which the measured
    reads do not use, and the stored usage totals, rebuilt from the copied usage
    events. The copy is written without a journal, then the store's
    write-ahead log is restored.
    """

    connection = sqlite3.connect(path, isolation_level=None)
    triggers = connection.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND sql IS NOT NULL"
    ).fetchall()
    try:
        connection.create_function("bench_shift", 2, _shift, deterministic=True)
        tables = _session_tables(connection)
        renamed = _renamed_columns(connection, tables)
        sources = []
        for index in range(templates):
            session_id = _template_id(index)
            instance_id, created_at = connection.execute(
                "SELECT instance_id, created_at FROM cayu_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            sources.append((session_id, instance_id, datetime.fromisoformat(created_at)))
        stride = {
            table.name: (
                connection.execute(
                    f'SELECT COALESCE(MAX("{table.integer_key}"), 0) FROM "{table.name}"'
                ).fetchone()[0]
                + 1
            )
            for table in tables
            if table.integer_key is not None
        }
        start = now - timedelta(days=days)
        step = timedelta(days=days) / sessions
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute(
            "CREATE TEMP TABLE bench_copies (source_id TEXT, source_instance TEXT, "
            "target_id TEXT, target_instance TEXT, seconds REAL, copy INTEGER, suffix TEXT)"
        )
        connection.executemany(
            "INSERT INTO bench_copies VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    source_id,
                    source_instance,
                    _clone_id(index),
                    str(uuid.uuid4()),
                    (start + step * index - source_created).total_seconds(),
                    index + 1,
                    f"-{index}",
                )
                for index in range(sessions)
                for source_id, source_instance, source_created in (sources[index % templates],)
            ],
        )
        connection.execute("BEGIN IMMEDIATE")
        try:
            for name, _sql in triggers:
                connection.execute(f'DROP TRIGGER "{name}"')

            # One statement per table copies every template row into every copy.
            for table in tables:
                selected = []
                for column in table.columns:
                    quoted = f'source."{column}"'
                    if column == table.integer_key:
                        selected.append(f"{quoted} + copies.copy * {stride[table.name]}")
                    elif column in table.times:
                        selected.append(f"bench_shift({quoted}, copies.seconds)")
                    elif column in renamed:
                        selected.append(f"{quoted} || copies.suffix")
                    elif column in table.texts:
                        selected.append(
                            f"REPLACE(REPLACE({quoted}, copies.source_id, copies.target_id), "
                            "copies.source_instance, copies.target_instance)"
                        )
                    else:
                        selected.append(quoted)
                columns = ", ".join(f'"{column}"' for column in table.columns)
                connection.execute(
                    f'INSERT INTO "{table.name}" ({columns}) SELECT {", ".join(selected)} '
                    f'FROM "{table.name}" AS source JOIN bench_copies AS copies '
                    f'ON source."{table.key}" = copies.source_id'
                )
            for table in tables:
                if table.integer_key is None:
                    continue
                top = connection.execute(
                    f'SELECT MAX("{table.integer_key}") FROM "{table.name}"'
                ).fetchone()[0]
                connection.execute(
                    "UPDATE sqlite_sequence SET seq = MAX(seq, ?) WHERE name = ?",
                    (top, table.name),
                )
            # Stored usage totals are rebuilt from the copied usage events.
            for statement in _iter_statements(SQLITE_USAGE_PROJECTION_REBUILD):
                connection.execute(statement)
            for _name, sql in triggers:
                connection.execute(sql)
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA optimize")
    finally:
        connection.close()


async def _add_running(path: Path, running: int, now: datetime) -> None:
    """Sessions still running, a few started long before the window."""

    clock = {"now": now}
    store = SQLiteSessionStore(path, ownership_clock=lambda: clock["now"])
    try:
        for index in range(running):
            clock["now"] = now - timedelta(days=3 * (index + 1))
            session_id = f"running-{index}"
            await store.create(
                RunRequest(session_id=session_id, agent_name="reviewer", messages=[]),
                identity=SessionIdentity(provider_name=_PRICE.provider_name, model=_PRICE.model),
            )
            await store.update_status(session_id, SessionStatus.RUNNING)
    finally:
        await store.close()


def build_store(
    path: Path,
    *,
    sessions: int = SESSIONS,
    days: int = DAYS,
    templates: int = TEMPLATES,
    running: int = RUNNING,
) -> dict[str, Any]:
    """Build the benchmark store at ``path`` and describe what it holds."""

    started = time.perf_counter()
    asyncio.run(_run_templates(path, templates))
    now = datetime.now(UTC)
    _clone(path, sessions=sessions, templates=templates, days=days, now=now)
    asyncio.run(_add_running(path, running, now))
    with sqlite3.connect(path) as connection:
        events = connection.execute("SELECT COUNT(*) FROM cayu_events").fetchone()[0]
        total = connection.execute("SELECT COUNT(*) FROM cayu_sessions").fetchone()[0]
    return {
        "sessions": total,
        "events": events,
        "bytes": path.stat().st_size,
        "build_seconds": round(time.perf_counter() - started, 2),
    }


@asynccontextmanager
async def _measuring_server(path: Path) -> AsyncIterator[httpx.AsyncClient]:
    store = SQLiteSessionStore(path)
    try:
        config = ServerConfig.local_development(
            dashboard=DashboardConfig(
                runtime_config={"priceBook": default_price_book().model_dump(mode="json")}
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_server(_app(store), config=config)),
            base_url="http://bench",
            timeout=600,
        ) as client:
            yield client
    finally:
        await store.close()


async def _time(client: httpx.AsyncClient, path: str, params: Any, samples: int) -> dict:
    elapsed = []
    body: dict[str, Any] = {}
    for _ in range(samples):
        started = time.perf_counter()
        response = await client.get(path, params=params)
        elapsed.append(time.perf_counter() - started)
        if response.status_code != 200:
            raise RuntimeError(f"{path} returned {response.status_code}: {response.text[:500]}")
        body = response.json()
    return {
        "median_seconds": round(statistics.median(elapsed), 3),
        "max_seconds": round(max(elapsed), 3),
        "samples": samples,
        "body": body,
    }


def measure(path: Path, *, samples: int = 3) -> dict[str, Any]:
    """Time the dashboard's counts and list reads against the store at ``path``."""

    async def run(client: httpx.AsyncClient) -> dict[str, Any]:
        counts = await _time(client, "/api/sessions/overview/counts", {}, samples)
        listing = await _time(
            client, "/api/sessions", {"include": "overview", "limit": PAGE}, samples
        )
        dashboard = await _time(
            client,
            "/api/sessions",
            {"include": ["overview", "first_message"], "limit": PAGE},
            samples,
        )
        body = counts.pop("body")
        listed = listing.pop("body")
        dashboard.pop("body")
        return {
            "counts": {
                **counts,
                "total_count": body["total_count"],
                "running_now": body["running_now"],
                "input_tokens": body["tokens"]["input"],
            },
            "list_overview": {**listing, "rows": len(listed["overviews"])},
            "list_overview_first_message": dashboard,
        }

    async def measured() -> dict[str, Any]:
        async with _measuring_server(path) as client:
            return await run(client)

    return asyncio.run(measured())


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--sessions", type=int, default=SESSIONS)
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--store", type=Path, help="Measure an existing benchmark store.")
    parser.add_argument("--keep", type=Path, help="Build the store here and keep it.")
    args = parser.parse_args()
    report: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as scratch:
        path = args.store or args.keep or Path(scratch) / "sessions.db"
        if args.store is None:
            report["store"] = build_store(path, sessions=args.sessions, days=args.days)
        report.update(measure(path, samples=args.samples))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
