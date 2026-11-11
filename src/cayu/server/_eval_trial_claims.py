"""The trials of one durable eval run, claimed one by one from its store.

A run's owner opens its trials, and then every worker with the run's target
registered, the owner included, claims them one at a time. A claim's lease is
kept alive from the moment the trial is claimed, the claim is confirmed again
right before the trial starts, and the trial's terminal result is retained as
the run's ordinary checkpoint. A trial whose worker is lost keeps nothing: its
lease expires and the next worker claims it again under a new epoch.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from cayu._task_wait import cancel_and_join
from cayu.evals.runner import TrialExecution
from cayu.evals.store import (
    EvalRunClaim,
    EvalRunClaimLost,
    EvalRunStateConflict,
    EvalRunStatus,
    EvalRunTrialCheckpoint,
    EvalRunTrialClaim,
    EvalStore,
    EvalStoreTransientContention,
)

logger = logging.getLogger(__name__)

# A trial claim is renewed this often at most, however long its lease.
MAX_TRIAL_HEARTBEAT_INTERVAL_SECONDS = 30.0
# Store retries wait at least this long and at most the poll interval.
MIN_TRIAL_RETRY_SECONDS = 0.05


class _TrialStop(StrEnum):
    CLAIM_LOST = "claim_lost"
    CANCELLING = "cancelling"
    STOPPING = "stopping"


class StoreClaimedTrial:
    """One trial this process claimed. Its lease is kept alive until it settles.

    ``settle`` runs the trial under the claim; ``abandon`` gives up a trial
    that will not run here.
    """

    def __init__(
        self,
        store: EvalStore,
        claim: EvalRunTrialClaim,
        *,
        lease_seconds: int,
        poll_interval_seconds: float,
        stop: asyncio.Event,
    ) -> None:
        self._store = store
        self._claim = claim
        self._lease_seconds = lease_seconds
        self._poll_seconds = max(poll_interval_seconds, MIN_TRIAL_RETRY_SECONDS)
        self._heartbeat_seconds = min(lease_seconds / 4, MAX_TRIAL_HEARTBEAT_INTERVAL_SECONDS)
        self._stop = stop
        self._redact_json: Callable[[Any], Any] | None = None
        # Kept alive from the claim on, through any preparation before the trial runs.
        self._keep_alive = asyncio.ensure_future(self._renew_until_stopped())

    @property
    def case_id(self) -> str:
        return self._claim.case_id

    @property
    def trial_number(self) -> int:
        return self._claim.trial_number

    def bind(self, redact_json: Callable[[Any], Any]) -> None:
        """Set the redaction boundary its result is retained behind."""

        self._redact_json = redact_json

    async def abandon(self) -> None:
        """Give the trial back without running it."""

        await self._finish(retained=False)

    async def settle(self, execute: Callable[[], Awaitable[TrialExecution]]) -> None:
        if self._redact_json is None:
            raise RuntimeError("A claimed eval trial settles only behind a redaction boundary.")
        trial: asyncio.Future[TrialExecution] | None = None
        retained = False
        # A cancellation or process-control exit already ending the trial stays
        # the one that propagates while the trial stops.
        leaving: BaseException | None = None
        try:
            if not await self._still_held():
                return
            trial = asyncio.ensure_future(execute())
            done, _ = await asyncio.wait(
                {trial, self._keep_alive},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if trial not in done:
                logger.info(
                    "Stopped an eval trial before it finished.",
                    extra={
                        "eval_run_id": self._claim.run_id,
                        "eval_trial_case_id": self._claim.case_id,
                        "eval_trial_number": self._claim.trial_number,
                        "eval_trial_stop": str(self._keep_alive.result()),
                    },
                )
                return
            result, public_data = trial.result()
            if public_data is None:
                raise RuntimeError("Durable trial execution lost its public projection.")
            retained = await self._retain(
                EvalRunTrialCheckpoint(
                    case_id=self._claim.case_id,
                    result=result,
                    public_data=public_data,
                ),
                redact_json=self._redact_json,
            )
        except BaseException as exiting:
            if not isinstance(exiting, Exception):
                leaving = exiting
            raise
        finally:
            # A cancellation of this worker while the trial stops, or a
            # process-control exit of the stopped trial, is kept: the claim is
            # still finished, then it is raised.
            stopped: BaseException | None = None
            if trial is not None and not trial.done():
                try:
                    failure = await cancel_and_join(trial, leaving=leaving)
                except asyncio.CancelledError as cancellation:
                    stopped = cancellation
                else:
                    # An exit already leaving ``settle`` stays the one raised.
                    if (
                        leaving is None
                        and failure is not None
                        and not isinstance(failure, Exception)
                    ):
                        stopped = failure
            await self._finish(retained=retained, leaving=leaving or stopped)
            if stopped is not None:
                raise stopped

    async def _still_held(self) -> bool:
        """Confirm the claim right before the trial starts, so a lapsed claim never runs.

        A store error is retried while the claim's renewal still holds it; the
        renewal gives up once the lease has run out without one.
        """

        while not self._keep_alive.done():
            try:
                observation = await self._store.heartbeat_run_trial(
                    self._claim,
                    extend_seconds=self._lease_seconds,
                )
            except asyncio.CancelledError:
                raise
            except EvalRunClaimLost:
                return False
            except Exception:
                await asyncio.sleep(self._poll_seconds)
                continue
            return observation.status is EvalRunStatus.RUNNING
        return False

    async def _renew_until_stopped(self) -> _TrialStop:
        loop = asyncio.get_running_loop()
        renewed_at = loop.time()
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=self._heartbeat_seconds)
            if self._stop.is_set():
                return _TrialStop.STOPPING
            try:
                observation = await self._store.heartbeat_run_trial(
                    self._claim,
                    extend_seconds=self._lease_seconds,
                )
            except asyncio.CancelledError:
                raise
            except EvalRunClaimLost:
                return _TrialStop.CLAIM_LOST
            except Exception:
                # A store that fails for a moment may answer the next renewal;
                # the claim is lost once its lease ran out without one.
                if loop.time() - renewed_at >= self._lease_seconds:
                    return _TrialStop.CLAIM_LOST
                continue
            renewed_at = loop.time()
            if observation.status is not EvalRunStatus.RUNNING:
                return _TrialStop.CANCELLING

    async def _retain(
        self,
        checkpoint: EvalRunTrialCheckpoint,
        *,
        redact_json: Callable[[Any], Any],
    ) -> bool:
        """Retain the trial's result; False when another worker owns the trial now."""

        while True:
            try:
                await self._store.save_claimed_trial_checkpoint(
                    self._claim,
                    checkpoint,
                    redact_json=redact_json,
                )
                return True
            except EvalStoreTransientContention:
                logger.warning(
                    "Durable eval trial checkpoint was deferred by transient store "
                    "contention; the claimed trial will be retried."
                )
                await asyncio.sleep(self._poll_seconds)
            except (EvalRunClaimLost, EvalRunStateConflict):
                return False

    async def _finish(self, *, retained: bool, leaving: BaseException | None = None) -> None:
        # The renewal's own ordinary failure only ends it; a process-control exit
        # from it propagates. ``leaving`` is the exit already ending the trial and
        # stays the one that propagates. Otherwise a cancellation of this worker
        # while the renewal stops is kept: the claim is still released, then it
        # is raised.
        stopped: asyncio.CancelledError | None = None
        try:
            failure = await cancel_and_join(self._keep_alive, leaving=leaving)
        except asyncio.CancelledError as cancellation:
            stopped = cancellation
        else:
            if failure is not None and not isinstance(failure, Exception):
                raise failure
        if not retained:
            try:
                await self._store.release_run_trial(self._claim)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Failed to release an eval trial claim; its lease will expire.")
        if stopped is not None:
            raise stopped


class StoreTrialClaims:
    """Hand out the trials of one run that this process claims from the store.

    The run's owner passes its run ``owner`` claim and no ``stop``: it keeps
    claiming until every trial has a checkpoint, its own or another worker's,
    and picks up the trials of workers that were lost; its run's own claim
    monitor stops it. Any other worker stops claiming as soon as no trial is
    open or once ``claiming`` turns false, and releases its trials once
    ``stop`` is set. A store error while claiming is retried at the next poll:
    it never ends the run's work by itself.
    """

    def __init__(
        self,
        *,
        store: EvalStore,
        target_key: str,
        run_id: str,
        lease_seconds: int,
        poll_interval_seconds: float,
        redact_json: Callable[[Any], Any],
        stop: asyncio.Event | None,
        claiming: Callable[[], bool],
        owner: EvalRunClaim | None = None,
        first: StoreClaimedTrial | None = None,
    ) -> None:
        if first is not None and first._claim.run_id != run_id:
            raise ValueError("The first claimed trial must belong to the run.")
        self._store = store
        self._target_keys = (target_key,)
        self._run_id = run_id
        self._lease_seconds = lease_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._poll_seconds = max(poll_interval_seconds, MIN_TRIAL_RETRY_SECONDS)
        self._redact_json = redact_json
        self._stop = asyncio.Event() if stop is None else stop
        self._claiming = claiming
        self._owner = owner
        self._first = first
        self._settled = False
        if first is not None:
            first.bind(redact_json)

    async def close(self) -> None:
        """Give back a first trial that was never handed out."""

        first, self._first = self._first, None
        if first is not None:
            await first.abandon()

    def taken(self, claim: EvalRunTrialClaim) -> StoreClaimedTrial:
        """Keep a trial this process just claimed for this run."""

        trial = StoreClaimedTrial(
            self._store,
            claim,
            lease_seconds=self._lease_seconds,
            poll_interval_seconds=self._poll_interval_seconds,
            stop=self._stop,
        )
        trial.bind(self._redact_json)
        return trial

    async def claim(self) -> StoreClaimedTrial | None:
        if self._first is not None:
            first, self._first = self._first, None
            return first
        if self._settled or self._stop.is_set() or not self._claiming():
            return None
        try:
            lease = await self._store.claim_run_trial(
                self._target_keys,
                run_id=self._run_id,
                lease_seconds=self._lease_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Failed to claim a trial of a durable eval run; it will be retried.",
                extra={"eval_run_id": self._run_id},
            )
            return None
        return None if lease is None else self.taken(lease.claim)

    async def wait(self) -> bool:
        if self._owner is None or self._settled or self._stop.is_set():
            return False
        # Every open trial is held, by this process or another worker. Wait
        # for them; a lost worker's trial becomes claimable again.
        try:
            unfinished = await self._store.unfinished_run_trials(self._owner)
        except asyncio.CancelledError:
            raise
        except EvalRunClaimLost:
            # The run has another owner, which aggregates its trials.
            raise
        except Exception:
            logger.warning(
                "Failed to read a durable eval run's open trials; it will be retried.",
                extra={"eval_run_id": self._run_id},
            )
            unfinished = None
        if unfinished == 0:
            self._settled = True
            return False
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=self._poll_seconds)
        return not self._stop.is_set()


__all__ = ["StoreClaimedTrial", "StoreTrialClaims"]
