from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

POSTGRES_CONTENTION_TABLES = (
    "cayu_budget_reservations",
    "cayu_knowledge_embeddings",
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_chunks",
    "cayu_knowledge_entries",
    "cayu_event_watcher_dead_letters",
    "cayu_event_watcher_state",
    "cayu_events",
    "cayu_session_labels",
    "cayu_transcript_messages",
    "cayu_checkpoints",
    "cayu_tasks",
    "cayu_sessions",
    "cayu_schema_migrations",
)


async def drop_cayu_tables(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            for table in POSTGRES_CONTENTION_TABLES:
                await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await conn.commit()


async def recorded_revisions(dsn: str) -> list[tuple[int, str, int]]:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT revision, kind, compatible_from FROM cayu_schema_migrations "
            "ORDER BY revision ASC"
        )
        return [tuple(row) for row in await cur.fetchall()]


async def assert_waiting(task: asyncio.Task[Any], *, seconds: float = 0.1) -> None:
    await asyncio.sleep(seconds)
    assert task.done() is False


@asynccontextmanager
async def hold_advisory_xact_lock(dsn: str, lock_key: int) -> AsyncIterator[None]:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key,))
            yield
        await conn.commit()
