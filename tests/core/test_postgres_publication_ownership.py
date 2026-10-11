"""PostgreSQL publication uses explicit capabilities and native transactions."""

from __future__ import annotations

import asyncio

import pytest
from tests.core._native_publication_support import (
    assert_imports,
    bind,
    exercise_checkpoint_publication,
    exercise_publication,
    exercise_stage_promotion,
    postgres_publication,
    preparation,
)


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


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_postgres_checkpoint_publication_composes_and_rolls_back_commit_guard(postgres_dsn, error):
    async def run():
        async with postgres_publication(postgres_dsn, error) as (
            owner,
            caps,
            session,
            _,
            clean,
            snapshot,
        ):
            await exercise_checkpoint_publication(owner, caps, session, error, clean, snapshot)

    asyncio.run(run())
