"""Native publication composes with real SQL after the seeding store is closed."""

from __future__ import annotations

import asyncio

import pytest
from tests.core._native_publication_support import (
    assert_imports,
    bind,
    exercise_checkpoint_publication,
    exercise_publication,
    exercise_stage_promotion,
    preparation,
    sqlite_publication,
)


def test_sqlite_publication_imports_without_adapters_or_driver():
    assert_imports("sqlite")


def test_sqlite_stage_promotion_composes_real_publication(tmp_path):
    async def run():
        async with sqlite_publication(tmp_path, RuntimeError) as (
            owner,
            caps,
            session,
            _,
            clean,
            snapshot,
            _lock,
        ):
            await exercise_stage_promotion(owner, caps, session, clean, snapshot)

    asyncio.run(run())


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_sqlite_publication_composes_and_rolls_back_failed_writes(tmp_path, error):
    async def run():
        async with sqlite_publication(tmp_path, error) as parts:
            await exercise_publication(*parts[:-1])

    asyncio.run(run())


def test_sqlite_publication_cancellation_retains_transaction_until_commit(tmp_path):
    async def run():
        async with sqlite_publication(tmp_path, RuntimeError) as (
            owner,
            caps,
            session,
            fault,
            clean,
            _,
            lock,
        ):
            prepared = preparation(session)
            publish = bind(owner, "publish_runtime", caps)
            fault.enabled, fault.block = True, True
            task = asyncio.create_task(publish(prepared))
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
            assert (await publish(prepared)).replayed

    asyncio.run(run())


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_sqlite_checkpoint_publication_composes_and_rolls_back_commit_guard(tmp_path, error):
    async def run():
        async with sqlite_publication(tmp_path, error) as (
            owner,
            caps,
            session,
            _,
            clean,
            snapshot,
            _lock,
        ):
            await exercise_checkpoint_publication(owner, caps, session, error, clean, snapshot)

    asyncio.run(run())
