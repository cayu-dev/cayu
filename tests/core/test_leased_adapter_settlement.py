from __future__ import annotations

import asyncio

import pytest

from cayu._task_wait import CapturedAwaitableOutcome
from cayu.runtime._leased_adapter_runner import LeasedAdapterRunner, LeasedAdapterSettlement


def test_settlement_starts_once_and_retains_the_exact_task_until_completion() -> None:
    async def scenario() -> None:
        runner = LeasedAdapterRunner[LeasedAdapterSettlement[str]]()
        drain = LeasedAdapterSettlement[str]()
        runner.retain_drain("proposal", drain)
        entered = asyncio.Event()
        release = asyncio.Event()
        starts = []
        observations = []

        async def settle() -> str:
            entered.set()
            await release.wait()
            return "settled"

        def operation():
            starts.append("started")
            return settle()

        def failure_for(result):
            observations.append(result)
            return None

        task = runner.start_settlement(
            "proposal", drain, operation, name="test-exact-settlement", failure_for=failure_for
        )
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert not runner.finalize_settlement("proposal", drain, task, failure_for=failure_for)
            assert not drain.settlement_processed
            assert (
                runner.start_settlement(
                    "proposal", drain, operation, name="unused-successor", failure_for=failure_for
                )
                is task
            )
            assert starts == ["started"]
            assert runner.draining("proposal") is drain
            release.set()
            assert await task == "settled"
            await asyncio.sleep(0)
            assert drain.settlement_task is task
            assert drain.settlement_processed
            assert observations == ["settled"]
            assert runner.draining("proposal") is None
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("failed", [False, True])
def test_completed_settlement_is_observed_once_even_before_its_callback(failed: bool) -> None:
    async def scenario() -> None:
        runner = LeasedAdapterRunner[LeasedAdapterSettlement[CapturedAwaitableOutcome[None]]]()
        drain = LeasedAdapterSettlement[CapturedAwaitableOutcome[None]]()
        runner.retain_drain("proposal", drain)
        failure = ValueError("late settlement failure") if failed else None
        observations = []
        releases = []

        async def settle():
            return CapturedAwaitableOutcome[None](error=failure)

        def failure_for(result):
            observations.append(result)
            return result.error

        task = asyncio.create_task(settle())
        await task
        runner.adopt_settlement(
            "proposal", drain, task, failure_for=failure_for, on_settled=lambda: releases.append(1)
        )
        # Neither observation may register another callback or start another task.
        assert runner.finalize_settlement("proposal", drain, task, failure_for=failure_for)
        assert runner.finalize_settlement("proposal", drain, task, failure_for=failure_for)
        runner.adopt_settlement(
            "proposal", drain, task, failure_for=failure_for, on_settled=lambda: releases.append(2)
        )
        assert len(observations) == 1
        assert releases == []
        assert drain.settlement_failure is failure
        assert runner.draining("proposal") is (drain if failed else None)
        await asyncio.sleep(0)
        assert releases == [1]
        assert len(observations) == 1
        runner.acknowledge_drain("proposal", drain)
        assert runner.draining("proposal") is None

    asyncio.run(scenario())


@pytest.mark.parametrize("failed", [False, True])
def test_superseded_settlement_releases_only_its_original_capacity(failed: bool) -> None:
    async def scenario() -> None:
        runner = LeasedAdapterRunner[LeasedAdapterSettlement[CapturedAwaitableOutcome[None]]]()
        original = LeasedAdapterSettlement[CapturedAwaitableOutcome[None]]()
        successor = LeasedAdapterSettlement[CapturedAwaitableOutcome[None]]()
        release = asyncio.Event()
        released = []

        def exhausted():
            return RuntimeError("capacity exhausted")

        old_capacity = runner.reserve_capacity(2, exhausted)
        new_capacity = runner.reserve_capacity(2, exhausted)
        runner.retain_drain("decision", original)

        async def settle():
            await release.wait()
            return CapturedAwaitableOutcome[None](
                error=ValueError("late release failure") if failed else None
            )

        def release_old_capacity():
            released.append(old_capacity)
            runner.release_capacity(old_capacity)

        task = asyncio.create_task(settle())
        runner.adopt_settlement(
            "decision",
            original,
            task,
            failure_for=lambda result: result.error,
            on_settled=release_old_capacity,
        )
        runner.retain_drain("decision", successor)
        try:
            with pytest.raises(RuntimeError, match="capacity exhausted"):
                runner.reserve_capacity(2, exhausted)
            release.set()
            await task
            await asyncio.sleep(0)
            assert released == [old_capacity]
            assert runner.draining("decision") is successor
            runner.acknowledge_drain("decision", original)
            assert runner.draining("decision") is successor
            with pytest.raises(RuntimeError, match="capacity exhausted"):
                runner.reserve_capacity(1, exhausted)
            replacement_capacity = runner.reserve_capacity(2, exhausted)
            runner.release_capacity(replacement_capacity)
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            runner.release_capacity(old_capacity)
            runner.release_capacity(new_capacity)

    asyncio.run(scenario())


@pytest.mark.parametrize("cancelled", [False, True])
def test_unsuccessful_settlement_task_remains_owned(cancelled: bool) -> None:
    async def scenario() -> None:
        runner = LeasedAdapterRunner[LeasedAdapterSettlement[None]]()
        drain = LeasedAdapterSettlement[None]()
        runner.retain_drain("proposal", drain)
        releases = []

        async def settle():
            if cancelled:
                raise asyncio.CancelledError("settlement cancelled")
            raise RuntimeError("settlement failed before returning an outcome")

        def failure_for(_result):
            raise AssertionError("A failed task has no returned outcome to classify.")

        task = asyncio.create_task(settle())
        await asyncio.gather(task, return_exceptions=True)
        runner.adopt_settlement(
            "proposal", drain, task, failure_for=failure_for, on_settled=lambda: releases.append(1)
        )
        await asyncio.sleep(0)
        assert releases == [1]
        assert not runner.finalize_settlement("proposal", drain, task, failure_for=failure_for)
        assert not drain.settlement_processed
        assert drain.settlement_task is task
        assert runner.draining("proposal") is drain

    asyncio.run(scenario())


def test_adoption_and_observation_cannot_replace_the_owned_settlement_task() -> None:
    async def scenario() -> None:
        runner = LeasedAdapterRunner[LeasedAdapterSettlement[None]]()
        drain = LeasedAdapterSettlement[None]()
        runner.retain_drain("proposal", drain)
        release = asyncio.Event()

        async def settle():
            await release.wait()

        original = asyncio.create_task(settle())
        other = asyncio.create_task(settle())
        try:
            runner.adopt_settlement("proposal", drain, original, failure_for=lambda result: None)
            with pytest.raises(RuntimeError, match="already owns a different settlement"):
                runner.adopt_settlement("proposal", drain, other, failure_for=lambda result: None)
            assert not runner.finalize_settlement(
                "proposal", drain, other, failure_for=lambda result: None
            )
            assert drain.settlement_task is original
            assert not drain.settlement_processed
            assert runner.draining("proposal") is drain
        finally:
            release.set()
            await asyncio.gather(original, other, return_exceptions=True)

    asyncio.run(scenario())
