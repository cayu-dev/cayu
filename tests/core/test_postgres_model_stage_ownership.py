"""PostgreSQL model-stage composition owns native transactions and cancellation."""

from __future__ import annotations

import asyncio

import pytest
from tests.core._native_model_stage_support import (
    assert_owner_imports,
    exercise,
    postgres_composition,
    preparation,
)


def test_postgres_model_stage_imports_without_adapters_or_driver():
    assert_owner_imports("postgres")


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_postgres_model_stage_composes_and_rolls_back_failed_writes(postgres_dsn, error):
    async def run():
        async with postgres_composition(postgres_dsn, error) as parts:
            await exercise(*parts)

    asyncio.run(run())


def test_postgres_model_stage_cancellation_rolls_back_preparation(postgres_dsn):
    async def run():
        async with postgres_composition(postgres_dsn, RuntimeError) as (
            ops,
            session,
            fault,
            clean,
            snapshot,
        ):
            prepared = preparation(session)
            before = await snapshot()
            fault.enabled, fault.block = True, True
            task = asyncio.create_task(ops.prepare(prepared))
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
            assert not (await ops.prepare(prepared)).replayed
            clean()

    asyncio.run(run())
