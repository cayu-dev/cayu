from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

POSTGRES_CONTENTION_TABLES = (
    "cayu_budget_settlements",
    "cayu_budget_reservations",
    "cayu_knowledge_embeddings",
    "cayu_task_terminalization_receipts",
    "cayu_completion_decision_application_receipts",
    "cayu_completion_decisions",
    "cayu_completion_verification_claims",
    "cayu_completion_proposals",
    "cayu_work_attempts",
    "cayu_task_session_execution_authority",
    "cayu_work_contracts",
    "cayu_recall_item_exposures",
    "cayu_context_exposures",
    "cayu_recall_receipts",
    "cayu_knowledge_publication_receipts",
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_chunks",
    "cayu_knowledge_revisions",
    "cayu_knowledge_entries",
    "cayu_event_watcher_dead_letters",
    "cayu_event_watcher_state",
    "cayu_budget_reservation_identities",
    "cayu_events",
    "cayu_session_labels",
    "cayu_public_authority_aliases",
    "cayu_public_authority_alias_keys",
    "cayu_public_authority_alias_config",
    "cayu_transcript_search_configuration",
    "cayu_transcript_messages",
    "cayu_session_message_queue",
    "cayu_persisted_event_side_effects",
    "cayu_mcp_manifest_baselines",
    "cayu_checkpoints",
    "cayu_session_operations",
    "cayu_tasks",
    "cayu_sessions",
    "cayu_eval_baseline_mutations",
    "cayu_eval_baselines",
    "cayu_eval_result_records",
    "cayu_eval_results",
    "cayu_eval_runs",
    "cayu_eval_cases",
    "cayu_eval_suites",
    "cayu_eval_corpora",
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
