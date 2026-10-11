"""PostgreSQL publication uses explicit capabilities and native transactions."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import partial

import pytest
from tests.core.test_native_publication_ownership import (
    TABLES,
    assert_imports,
    bind,
    exercise_publication,
    exercise_stage_promotion,
    preparation,
)
from tests.core.test_postgres_model_stage_ownership import postgres_composition


@asynccontextmanager
async def postgres_publication(postgres_dsn, error):
    from psycopg import sql
    from psycopg.errors import UniqueViolation

    from cayu.storage import _postgres_publication as owner
    from cayu.storage import postgres as adapter

    async with postgres_composition(postgres_dsn, error) as (stages, session, fault, clean, _):
        caps = dict(stages.prepare.keywords)
        readers = caps["load_for_update"].args[0]
        caps.update(
            stages=stages,
            closure_owners=partial(adapter.PostgresSessionStore._closure_lineage_owners, readers),
            public_authority_alias_codec=None,
            checkpoint_row_values=adapter._checkpoint_row_values,
            unique_violation=UniqueViolation,
        )

        async def snapshot():
            results = []
            async with caps["connect"]() as conn, conn.cursor() as cur:
                for table in TABLES:
                    await cur.execute(
                        sql.SQL("SELECT * FROM {} ORDER BY 1, 2").format(sql.Identifier(table))
                    )
                    results.append(await cur.fetchall())
            return results

        yield owner, caps, session, fault, clean, snapshot


def test_postgres_publication_imports_without_adapters_or_driver():
    assert_imports("postgres")


def test_postgres_stage_promotion_composes_real_publication(postgres_dsn):
    async def run():
        async with postgres_publication(postgres_dsn, RuntimeError) as (
            owner,
            caps,
            session,
            _,
            clean,
            snapshot,
        ):
            await exercise_stage_promotion(owner, caps, session, clean, snapshot)

    asyncio.run(run())


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_postgres_publication_composes_and_rolls_back_failed_writes(postgres_dsn, error):
    async def run():
        async with postgres_publication(postgres_dsn, error) as parts:
            await exercise_publication(*parts)

    asyncio.run(run())


def test_postgres_publication_cancellation_rolls_back_transaction(postgres_dsn):
    async def run():
        async with postgres_publication(postgres_dsn, RuntimeError) as (
            owner,
            caps,
            session,
            fault,
            clean,
            snapshot,
        ):
            prepared = preparation(session)
            publish = bind(owner, "publish_runtime", caps)
            before = await snapshot()
            fault.enabled, fault.block = True, True
            task = asyncio.create_task(publish(prepared))
            try:
                await asyncio.wait_for(fault.started.wait(), 5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 5)
            finally:
                fault.release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            clean()
            fault.enabled = False
            assert await snapshot() == before
            assert not (await publish(prepared)).replayed

    asyncio.run(run())
