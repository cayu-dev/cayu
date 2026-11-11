"""Finishing a claimed eval trial must not swallow a cancellation of its worker."""

from __future__ import annotations

import asyncio
import types
from typing import Any, cast

import pytest

from cayu.evals.store import EvalRunStatus, EvalRunTrialClaim
from cayu.server._eval_trial_claims import StoreClaimedTrial


class _Store:
    def __init__(self, renewing: asyncio.Event, release_renewal: asyncio.Event) -> None:
        self.renewing = renewing
        self.release_renewal = release_renewal
        self.released: list[EvalRunTrialClaim] = []

    async def heartbeat_run_trial(self, claim, *, extend_seconds):
        # A renewal in flight finishes even when the renewal is stopped.
        self.renewing.set()
        while not self.release_renewal.is_set():
            try:
                await self.release_renewal.wait()
            except asyncio.CancelledError:
                continue
        raise asyncio.CancelledError

    async def release_run_trial(self, claim) -> None:
        self.released.append(claim)


@pytest.mark.parametrize("retained", [False, True])
def test_finishing_a_trial_releases_its_claim_and_keeps_the_cancellation(
    retained: bool,
) -> None:
    async def scenario() -> None:
        renewing = asyncio.Event()
        release_renewal = asyncio.Event()
        store = _Store(renewing, release_renewal)
        claim = EvalRunTrialClaim.model_construct(
            run_id="run", run_epoch=1, case_id="case", trial_number=1, claim_id="claim"
        )
        trial = StoreClaimedTrial(
            store=cast("Any", store),
            claim=claim,
            lease_seconds=4,
            poll_interval_seconds=0.001,
            stop=asyncio.Event(),
        )
        await asyncio.wait_for(renewing.wait(), 5)
        finishing = asyncio.create_task(trial._finish(retained=retained))
        await asyncio.sleep(0)
        finishing.cancel("worker stopping")
        await asyncio.sleep(0)
        release_renewal.set()
        with pytest.raises(asyncio.CancelledError, match="worker stopping"):
            await finishing
        assert store.released == ([] if retained else [claim])

    asyncio.run(scenario())


def test_stopping_an_unfinished_trial_keeps_the_cancellation() -> None:
    """The claim's renewal ended first (the worker is stopping); the trial still
    running is stopped, and a cancellation arriving meanwhile ends ``settle``."""

    class _RunningStore:
        def __init__(self) -> None:
            self.released: list[EvalRunTrialClaim] = []

        async def heartbeat_run_trial(self, claim, *, extend_seconds):
            return types.SimpleNamespace(status=EvalRunStatus.RUNNING)

        async def release_run_trial(self, claim) -> None:
            self.released.append(claim)

    async def scenario() -> None:
        store = _RunningStore()
        stop = asyncio.Event()
        claim = EvalRunTrialClaim.model_construct(
            run_id="run", run_epoch=1, case_id="case", trial_number=1, claim_id="claim"
        )
        trial = StoreClaimedTrial(
            store=cast("Any", store),
            claim=claim,
            lease_seconds=4,
            poll_interval_seconds=0.001,
            stop=stop,
        )
        trial.bind(lambda value: value)
        stopping = asyncio.Event()
        release = asyncio.Event()

        async def execute():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stopping.set()
                await release.wait()
                raise

        settling = asyncio.create_task(trial.settle(execute))
        await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(stopping.wait(), 5)
        settling.cancel("worker stopping")
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError, match="worker stopping"):
            await settling
        assert store.released == [claim]

    asyncio.run(scenario())


def test_stopping_the_trial_keeps_the_first_cancellation() -> None:
    """The worker's cancellation already ends ``settle``; a second one while the
    trial stops does not replace it, and the claim is still released."""

    class _RunningStore:
        def __init__(self) -> None:
            self.released: list[EvalRunTrialClaim] = []

        async def heartbeat_run_trial(self, claim, *, extend_seconds):
            return types.SimpleNamespace(status=EvalRunStatus.RUNNING)

        async def release_run_trial(self, claim) -> None:
            self.released.append(claim)

    async def scenario() -> None:
        store = _RunningStore()
        claim = EvalRunTrialClaim.model_construct(
            run_id="run", run_epoch=1, case_id="case", trial_number=1, claim_id="claim"
        )
        trial = StoreClaimedTrial(
            store=cast("Any", store),
            claim=claim,
            lease_seconds=4,
            poll_interval_seconds=0.001,
            stop=asyncio.Event(),
        )
        trial.bind(lambda value: value)
        started = asyncio.Event()
        stopping = asyncio.Event()
        release = asyncio.Event()

        async def execute():
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stopping.set()
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        continue
                raise

        settling = asyncio.create_task(trial.settle(execute))
        await asyncio.wait_for(started.wait(), 5)
        settling.cancel("first")
        await asyncio.wait_for(stopping.wait(), 5)
        settling.cancel("second")
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError, match="first"):
            await settling
        assert store.released == [claim]

    asyncio.run(scenario())


def test_a_renewal_ending_in_process_control_propagates_from_finish() -> None:
    class _ExitingStore:
        def __init__(self) -> None:
            self.released: list[EvalRunTrialClaim] = []

        async def heartbeat_run_trial(self, claim, *, extend_seconds):
            raise GeneratorExit("process control")

        async def release_run_trial(self, claim) -> None:
            self.released.append(claim)

    async def scenario() -> None:
        store = _ExitingStore()
        claim = EvalRunTrialClaim.model_construct(
            run_id="run", run_epoch=1, case_id="case", trial_number=1, claim_id="claim"
        )
        trial = StoreClaimedTrial(
            store=cast("Any", store),
            claim=claim,
            lease_seconds=4,
            poll_interval_seconds=0.001,
            stop=asyncio.Event(),
        )
        trial._heartbeat_seconds = 0.001
        await asyncio.sleep(0.05)
        with pytest.raises(GeneratorExit):
            await trial._finish(retained=False)

    asyncio.run(scenario())


class _RenewalInFlightStore:
    """Answers the start check, then holds every renewal until released, as a
    renewal whose request is in flight does even when it is stopped."""

    def __init__(self) -> None:
        self.calls = 0
        self.renewing = asyncio.Event()
        self.release = asyncio.Event()
        self.released: list[EvalRunTrialClaim] = []

    async def heartbeat_run_trial(self, claim, *, extend_seconds):
        self.calls += 1
        if self.calls == 1:
            return types.SimpleNamespace(status=EvalRunStatus.RUNNING)
        self.renewing.set()
        stopped = False
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                stopped = True
        if stopped:
            raise asyncio.CancelledError
        return types.SimpleNamespace(status=EvalRunStatus.RUNNING)

    async def release_run_trial(self, claim) -> None:
        self.released.append(claim)


def _claimed(store, stop: asyncio.Event | None = None) -> StoreClaimedTrial:
    claim = EvalRunTrialClaim.model_construct(
        run_id="run", run_epoch=1, case_id="case", trial_number=1, claim_id="claim"
    )
    trial = StoreClaimedTrial(
        store=cast("Any", store),
        claim=claim,
        lease_seconds=4,
        poll_interval_seconds=0.001,
        stop=stop or asyncio.Event(),
    )
    trial.bind(lambda value: value)
    return trial


def test_a_second_cancellation_while_the_renewal_stops_keeps_the_first() -> None:
    async def scenario() -> None:
        store = _RenewalInFlightStore()
        trial = _claimed(store)
        trial._heartbeat_seconds = 0.001
        started = asyncio.Event()

        async def execute():
            started.set()
            await asyncio.Event().wait()

        settling = asyncio.create_task(trial.settle(execute))
        await asyncio.wait_for(started.wait(), 5)
        await asyncio.wait_for(store.renewing.wait(), 5)
        settling.cancel("first")
        await asyncio.sleep(0.01)
        assert not settling.done()
        settling.cancel("second")
        await asyncio.sleep(0)
        store.release.set()
        with pytest.raises(asyncio.CancelledError, match="first"):
            await settling
        assert len(store.released) == 1

    asyncio.run(scenario())


def test_a_stopped_trials_process_control_exit_is_raised_after_the_claim_is_released() -> None:
    class _Store:
        def __init__(self) -> None:
            self.released: list[EvalRunTrialClaim] = []

        async def heartbeat_run_trial(self, claim, *, extend_seconds):
            return types.SimpleNamespace(status=EvalRunStatus.RUNNING)

        async def release_run_trial(self, claim) -> None:
            self.released.append(claim)

    async def scenario() -> None:
        store = _Store()
        stop = asyncio.Event()
        trial = _claimed(store, stop)
        started = asyncio.Event()

        async def execute():
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise GeneratorExit("trial cleanup") from None

        settling = asyncio.create_task(trial.settle(execute))
        await asyncio.wait_for(started.wait(), 5)
        stop.set()
        [outcome] = await asyncio.gather(settling, return_exceptions=True)
        assert isinstance(outcome, GeneratorExit), repr(outcome)
        assert len(store.released) == 1

    asyncio.run(scenario())


def test_a_stopped_trials_failure_does_not_replace_the_exit_already_leaving() -> None:
    """The worker is cancelled while the trial runs; the trial's cleanup then
    ends with a group of cancellations. The worker's cancellation is raised."""

    class _Store:
        def __init__(self) -> None:
            self.released: list[EvalRunTrialClaim] = []

        async def heartbeat_run_trial(self, claim, *, extend_seconds):
            return types.SimpleNamespace(status=EvalRunStatus.RUNNING)

        async def release_run_trial(self, claim) -> None:
            self.released.append(claim)

    async def scenario() -> None:
        store = _Store()
        trial = _claimed(store)
        started = asyncio.Event()

        async def execute():
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as primary:
                raise BaseExceptionGroup(
                    "cleanup received concurrent control", [primary, asyncio.CancelledError()]
                ) from None

        settling = asyncio.create_task(trial.settle(execute))
        await asyncio.wait_for(started.wait(), 5)
        settling.cancel("worker stopping")
        with pytest.raises(asyncio.CancelledError, match="worker stopping"):
            await settling
        assert len(store.released) == 1

    asyncio.run(scenario())
