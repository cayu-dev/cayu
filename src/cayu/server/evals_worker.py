"""Focused durable worker for server-attached eval execution."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from enum import StrEnum

from cayu.evals.corpus import EvalCorpusDocument
from cayu.evals.execution import (
    CompiledCorpusSuite,
    CorpusTarget,
    _run_compiled_corpus_suite,
    compile_corpus_suite,
    evaluation_target_identity,
)
from cayu.evals.store import (
    EvalRunClaim,
    EvalRunClaimLost,
    EvalRunFailureCode,
    EvalRunLease,
    EvalRunStateConflict,
    EvalRunStatus,
)
from cayu.server.config import EvalsConfig
from cayu.server.evals_registry import (
    EvalTargetRegistration,
    ResolvedEvalsRuntime,
    resolved_evals_runtime,
    target_for_eval_invocation,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _PreparedEvalRun:
    target: CorpusTarget
    compiled: CompiledCorpusSuite


class _ClaimMonitorOutcome(StrEnum):
    STOPPING = "stopping"
    CANCELLING = "cancelling"
    CLAIM_LOST = "claim_lost"


class EvalRunCoordinator:
    """Run one restart-safe embedded worker over a durable ``EvalStore``.

    The store remains authoritative for admission, ownership, cancellation,
    and terminal publication. This coordinator never retains a second job
    state and never publishes after its fenced claim is lost.
    """

    def __init__(self, config: EvalsConfig | ResolvedEvalsRuntime) -> None:
        if type(config) is EvalsConfig:
            resolved = resolved_evals_runtime(
                explicit=config,
                registry=None,
                automatic_store=None,
            )
            assert resolved is not None
            config = resolved
        elif type(config) is not ResolvedEvalsRuntime:
            raise TypeError("config must be an exact EvalsConfig or ResolvedEvalsRuntime.")
        self._config = config
        self._stop_requested = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the single embedded worker in the current event loop."""

        if self._task is not None:
            raise RuntimeError("Eval run coordinator has already been started.")
        self._task = asyncio.create_task(
            self._run_forever(),
            name="cayu-eval-run-coordinator",
        )

    async def stop(self) -> None:
        """Stop dispatch within the configured grace period.

        Cooperative shutdown releases owned work while the grace period remains.
        After the deadline, cancellation stops local execution and the durable
        lease provides the recovery boundary; shutdown never waits indefinitely
        for a stalled store operation.
        """

        task = self._task
        if task is None:
            return
        self._stop_requested.set()
        done, _ = await asyncio.wait(
            {task},
            timeout=self._config.shutdown_grace_seconds,
        )
        self._task = None
        if done:
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return
        task.cancel()
        task.add_done_callback(self._consume_detached_task)

    async def _run_forever(self) -> None:
        while not self._stop_requested.is_set():
            try:
                lease = await self._claim_run()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error("Failed to claim durable eval work; dispatch will retry.")
                await self._wait_for_work()
                continue
            if lease is None:
                await self._wait_for_work()
                continue
            try:
                await self._run_lease(lease)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Per-lease handlers are deliberately defensive so this branch
                # indicates an implementation fault, not a workload diagnostic.
                logger.error("Unexpected failure while coordinating durable eval work.")
                await self._release_after_interruption(lease.claim)

    async def _claim_run(self) -> EvalRunLease | None:
        target_keys = self._config.registry.target_keys
        if len(target_keys) == 1:
            return await self._config.store.claim_run(
                target_key=target_keys[0],
                lease_seconds=self._config.lease_seconds,
            )
        return await self._config.store.claim_run_for_targets(
            target_keys,
            lease_seconds=self._config.lease_seconds,
        )

    async def _wait_for_work(self) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                self._stop_requested.wait(),
                timeout=self._config.poll_interval_seconds,
            )

    async def _run_lease(self, lease: EvalRunLease) -> None:
        if lease.run.status is EvalRunStatus.CANCELLING:
            await self._finish_cancel(lease.claim)
            return
        registration = self._config.registry.registration(lease.run.spec.target_key)
        if registration is None:
            await self._finalize_failure(
                lease.claim,
                EvalRunFailureCode.TARGET_UNAVAILABLE,
            )
            return
        await self._run_owned_lease(lease, registration)

    async def _run_owned_lease(
        self,
        lease: EvalRunLease,
        registration: EvalTargetRegistration,
    ) -> None:
        preflight = asyncio.create_task(
            self._preflight_lease(lease, registration),
            name=f"cayu-eval-preflight-{lease.run.id}",
        )
        monitor = asyncio.create_task(
            self._monitor_claim(lease.claim),
            name=f"cayu-eval-preflight-claim-{lease.run.id}",
        )
        try:
            done, _ = await asyncio.wait(
                {preflight, monitor},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if monitor in done:
                outcome = monitor.result()
                await self._cancel_task(preflight)
                if outcome is _ClaimMonitorOutcome.CANCELLING:
                    await self._finish_cancel(lease.claim)
                elif outcome is _ClaimMonitorOutcome.STOPPING:
                    await self._release_after_interruption(lease.claim)
                return

            preflight_result = preflight.result()
            if isinstance(preflight_result, EvalRunFailureCode):
                failure = asyncio.create_task(
                    self._finalize_failure(lease.claim, preflight_result),
                    name=f"cayu-eval-preflight-failure-{lease.run.id}",
                )
                await self._await_owned_action(lease.claim, failure, monitor)
                return
            if self._stop_requested.is_set():
                await self._release_after_interruption(lease.claim)
                return

            current = await self._refresh_claim(lease.claim)
            if current is None:
                return
            if current.status is EvalRunStatus.CANCELLING:
                await self._finish_cancel(lease.claim)
                return
            if self._stop_requested.is_set():
                await self._release_after_interruption(lease.claim)
                return

            await self._execute_compiled_lease(lease, registration, preflight_result, monitor)
        finally:
            await self._cancel_task(preflight)
            await self._cancel_task(monitor)

    async def _preflight_lease(
        self,
        lease: EvalRunLease,
        registration: EvalTargetRegistration,
    ) -> _PreparedEvalRun | EvalRunFailureCode:
        try:
            corpus = await self._config.store.load_corpus(lease.run.spec.corpus_revision)
        except asyncio.CancelledError:
            raise
        except Exception:
            return EvalRunFailureCode.CORPUS_UNAVAILABLE
        if corpus is None:
            return EvalRunFailureCode.CORPUS_UNAVAILABLE

        return await asyncio.to_thread(
            self._compile_loaded_corpus,
            corpus,
            lease,
            registration,
        )

    def _compile_loaded_corpus(
        self,
        corpus: EvalCorpusDocument,
        lease: EvalRunLease,
        registration: EvalTargetRegistration,
    ) -> _PreparedEvalRun | EvalRunFailureCode:
        try:
            target = target_for_eval_invocation(
                registration.target,
                lease.run.spec.invocation,
            )
        except Exception:
            return EvalRunFailureCode.TARGET_UNAVAILABLE
        try:
            identity = evaluation_target_identity(
                target,
                project_root=registration.manifest_project_root,
            )
        except Exception:
            return EvalRunFailureCode.TARGET_UNAVAILABLE
        if identity.app_manifest_fingerprint != registration.catalog_entry.app_manifest_fingerprint:
            return EvalRunFailureCode.TARGET_UNAVAILABLE
        try:
            compiled = compile_corpus_suite(
                corpus,
                target,
                lease.run.spec.suite_id,
            )
            if (
                compiled.run_contract.corpus_revision != lease.run.spec.corpus_revision
                or compiled.run_contract.suite_id != lease.run.spec.suite_id
                or compiled.run_contract.suite_revision != lease.run.spec.suite_revision
            ):
                raise ValueError("Persisted eval run does not match its compiled suite.")
        except Exception:
            return EvalRunFailureCode.CORPUS_UNAVAILABLE
        return _PreparedEvalRun(target=target, compiled=compiled)

    async def _execute_compiled_lease(
        self,
        lease: EvalRunLease,
        registration: EvalTargetRegistration,
        prepared: _PreparedEvalRun,
        monitor: asyncio.Task[_ClaimMonitorOutcome],
    ) -> None:
        target = prepared.target
        execution = asyncio.create_task(
            _run_compiled_corpus_suite(
                target,
                prepared.compiled,
                max_concurrency=lease.run.spec.max_concurrency,
                manifest_project_root=registration.manifest_project_root,
                expected_app_manifest_fingerprint=(
                    registration.catalog_entry.app_manifest_fingerprint
                ),
            ),
            name=f"cayu-eval-run-{lease.run.id}",
        )
        try:
            done, _ = await asyncio.wait(
                {execution, monitor},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if monitor in done:
                outcome = monitor.result()
                await self._cancel_task(execution)
                await self._handle_monitor_outcome(lease.claim, outcome)
                return

            current = await self._refresh_claim(lease.claim)
            if current is None:
                return
            if current.status is EvalRunStatus.CANCELLING:
                await self._finish_cancel(lease.claim)
                return

            publication = asyncio.create_task(
                self._publish_execution_outcome(lease.claim, target, execution),
                name=f"cayu-eval-publication-{lease.run.id}",
            )
            await self._await_owned_action(lease.claim, publication, monitor)
        finally:
            await self._cancel_task(execution)

    async def _publish_execution_outcome(
        self,
        claim: EvalRunClaim,
        target: CorpusTarget,
        execution: asyncio.Task,
    ) -> None:
        if execution.cancelled():
            await self._finalize_failure(
                claim,
                EvalRunFailureCode.WORKER_INTERRUPTED,
                refresh=False,
            )
            return
        if execution.exception() is not None:
            await self._finalize_failure(
                claim,
                EvalRunFailureCode.EXECUTION_FAILED,
                refresh=False,
            )
            return
        try:
            await self._config.store.publish_result(
                claim,
                execution.result(),
                redact_json=target.app.redact_json,
            )
        except EvalRunClaimLost:
            return
        except EvalRunStateConflict:
            await self._finish_cancel_if_requested(claim)
        except Exception:
            await self._finalize_failure(
                claim,
                EvalRunFailureCode.EXECUTION_FAILED,
            )

    async def _await_owned_action(
        self,
        claim: EvalRunClaim,
        action: asyncio.Task[None],
        monitor: asyncio.Task[_ClaimMonitorOutcome],
    ) -> None:
        try:
            done, _ = await asyncio.wait(
                {action, monitor},
                return_when=asyncio.FIRST_COMPLETED,
            )
            # A completed fenced transition is authoritative even when the
            # monitor observes that terminal record in the same loop turn.
            if action in done:
                await action
                return
            outcome = monitor.result()
            await self._cancel_task(action)
            await self._handle_monitor_outcome(claim, outcome)
        finally:
            await self._cancel_task(action)

    async def _handle_monitor_outcome(
        self,
        claim: EvalRunClaim,
        outcome: _ClaimMonitorOutcome,
    ) -> None:
        if outcome is _ClaimMonitorOutcome.CANCELLING:
            await self._finish_cancel(claim)
        elif outcome is _ClaimMonitorOutcome.STOPPING:
            await self._release_after_interruption(claim)

    async def _monitor_claim(self, claim: EvalRunClaim) -> _ClaimMonitorOutcome:
        heartbeat_interval = min(self._config.lease_seconds / 4, 30.0)
        loop = asyncio.get_running_loop()
        next_heartbeat = loop.time() + heartbeat_interval
        while True:
            wait_seconds = min(
                self._config.poll_interval_seconds,
                max(0.0, next_heartbeat - loop.time()),
            )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_requested.wait(), timeout=wait_seconds)
            if self._stop_requested.is_set():
                return _ClaimMonitorOutcome.STOPPING
            try:
                if loop.time() >= next_heartbeat:
                    record = await self._config.store.heartbeat_run(
                        claim,
                        extend_seconds=self._config.lease_seconds,
                    )
                    next_heartbeat = loop.time() + heartbeat_interval
                else:
                    record = await self._config.store.load_run(claim.run_id)
                    if (
                        record is None
                        or record.ownership is None
                        or record.ownership.epoch != claim.epoch
                    ):
                        return _ClaimMonitorOutcome.CLAIM_LOST
            except asyncio.CancelledError:
                raise
            except Exception:
                return _ClaimMonitorOutcome.CLAIM_LOST
            if record.status is EvalRunStatus.CANCELLING:
                return _ClaimMonitorOutcome.CANCELLING
            if record.status is not EvalRunStatus.RUNNING:
                return _ClaimMonitorOutcome.CLAIM_LOST

    async def _refresh_claim(self, claim: EvalRunClaim):
        try:
            return await self._config.store.heartbeat_run(
                claim,
                extend_seconds=self._config.lease_seconds,
            )
        except Exception:
            return None

    async def _finalize_failure(
        self,
        claim: EvalRunClaim,
        code: EvalRunFailureCode,
        *,
        refresh: bool = True,
    ) -> None:
        if refresh:
            current = await self._refresh_claim(claim)
            if current is None:
                return
            if current.status is EvalRunStatus.CANCELLING:
                await self._finish_cancel(claim)
                return
        try:
            await self._config.store.fail_run(claim, code)
        except EvalRunClaimLost:
            return
        except EvalRunStateConflict:
            await self._finish_cancel_if_requested(claim)
        except Exception:
            logger.error("Failed to persist a terminal eval diagnostic.")

    async def _finish_cancel_if_requested(self, claim: EvalRunClaim) -> None:
        try:
            record = await self._config.store.load_run(claim.run_id)
        except Exception:
            return
        if (
            record is not None
            and record.status is EvalRunStatus.CANCELLING
            and record.ownership is not None
            and record.ownership.epoch == claim.epoch
        ):
            await self._finish_cancel(claim)

    async def _finish_cancel(self, claim: EvalRunClaim) -> None:
        try:
            await self._config.store.finish_cancel(claim)
        except (EvalRunClaimLost, EvalRunStateConflict):
            return
        except Exception:
            logger.error("Failed to persist terminal eval cancellation.")

    async def _release_after_interruption(self, claim: EvalRunClaim) -> None:
        try:
            await self._config.store.release_run(claim)
        except (EvalRunClaimLost, EvalRunStateConflict):
            return
        except Exception:
            logger.error("Failed to release interrupted eval work; its lease will expire.")

    @staticmethod
    def _consume_detached_task(task: asyncio.Task[None]) -> None:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.result()

    @staticmethod
    async def _cancel_task(task: asyncio.Task) -> None:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
