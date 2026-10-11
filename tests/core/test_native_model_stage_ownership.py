"""Native model-stage operations compose without retaining a session store."""

from __future__ import annotations

import asyncio

import pytest
from tests.core._native_model_stage_support import (
    assert_owner_imports,
    exercise,
    preparation,
    sqlite_composition,
)


def test_sqlite_model_stage_imports_without_adapters_or_driver():
    assert_owner_imports("sqlite")


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_sqlite_model_stage_composes_and_rolls_back_failed_writes(tmp_path, error):
    async def run():
        async with sqlite_composition(tmp_path, error) as parts:
            await exercise(*parts[:-1])

    asyncio.run(run())


def test_sqlite_model_stage_cancellation_retains_writer_until_commit(tmp_path):
    async def run():
        async with sqlite_composition(tmp_path, RuntimeError) as (
            ops,
            session,
            fault,
            clean,
            _snapshot,
            lock,
        ):
            prepared = preparation(session)
            fault.enabled, fault.block = True, True
            task = asyncio.create_task(ops.prepare(prepared))
            try:
                await asyncio.wait_for(fault.started.wait(), 5)
                task.cancel()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert not task.done() and lock.locked()
            finally:
                fault.release.set()
                await asyncio.gather(task, return_exceptions=True)
            assert task.cancelled()
            clean()
            fault.enabled = False
            # Caller cancellation retains the physical worker through commit.
            assert (await ops.prepare(prepared)).replayed
            clean()

    asyncio.run(run())
