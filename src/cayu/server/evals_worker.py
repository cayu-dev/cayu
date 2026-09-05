"""Focused durable worker for server-attached eval execution."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from enum import StrEnum

from cayu._exception_groups import iter_exception_tree
from cayu.evals._execution_profile_errors import EvalExecutionProfileChangedError
from cayu.evals.corpus import (
    EvalCorpusDocument,
    ModelJudgeAssertionSpec,
    StructuredModelJudgeAssertionSpec,
    pricing_profile_identity,
)
from cayu.evals.execution import (
    CompiledCorpusSuite,
    CorpusTarget,
    _run_compiled_corpus_suite,
    compile_corpus_suite,
    evaluation_target_identity,
)
from cayu.evals.scenario import EvalScenarioDocumentV2
from cayu.evals.scenario_execution import (
    ScenarioExecutionError,
    corpus_for_eval_scenario,
    run_compiled_eval_scenario,
    scenario_launch_settings_from_invocation,
)
from cayu.evals.scenario_preflight import ScenarioLaunchBindingV2, preflight_eval_scenario
from cayu.evals.store import (
    EVAL_RUN_MAX_OBSERVATION_INTERVAL_SECONDS,
    EVAL_RUN_MIN_OBSERVATION_INTERVAL_SECONDS,
    EvalRunClaim,
    EvalRunClaimLost,
    EvalRunFailureCode,
    EvalRunFailureDiagnostic,
    EvalRunFailureReason,
    EvalRunLease,
    EvalRunStateConflict,
    EvalRunStatus,
    EvalRunTrialCheckpoint,
    EvalScenarioTrialFailureCode,
    EvalStoreTransientContention,
)
from cayu.evals.suite_authoring import EvalSuiteDocument
from cayu.evals.suite_execution import corpus_for_authored_scenario_case
from cayu.evals.trial_policy import EvalCandidateCostBudgetV1
from cayu.runtime.execution_profiles import (
    ExecutionProfileIdentity,
    ExecutionProfileMismatchError,
)
from cayu.server.config import EvalsConfig
from cayu.server.evals_registry import (
    EvalTargetRegistration,
    ResolvedEvalsRuntime,
    resolved_evals_runtime,
    target_for_eval_invocation,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _EvalFailure:
    code: EvalRunFailureCode
    diagnostic: EvalRunFailureDiagnostic


def _failure(code: EvalRunFailureCode, reason: EvalRunFailureReason) -> _EvalFailure:
    return _EvalFailure(code, EvalRunFailureDiagnostic(reason=reason))


def _execution_diagnostic(error: BaseException) -> EvalRunFailureDiagnostic:
    from cayu.providers._openai_protocol import protocol_diagnostic_fields
    from cayu.providers.openai import OpenAIProtocolError

    if _is_execution_profile_failure(error):
        return EvalRunFailureDiagnostic(reason=EvalRunFailureReason.EXECUTION_PROFILE_REJECTED)
    for candidate in iter_exception_tree(error):
        if type(candidate) is OpenAIProtocolError:
            return EvalRunFailureDiagnostic(
                reason=EvalRunFailureReason.PROVIDER_PROTOCOL_FAILED,
                provider_protocol_reason=protocol_diagnostic_fields(candidate.reason_code)[
                    "provider_protocol_reason"
                ],
            )
    return EvalRunFailureDiagnostic(reason=EvalRunFailureReason.EXECUTION_FAILED)


def _is_execution_profile_failure(error: BaseException) -> bool:
    return any(
        isinstance(candidate, EvalExecutionProfileChangedError | ExecutionProfileMismatchError)
        or (
            isinstance(candidate, ScenarioExecutionError)
            and candidate.code is EvalScenarioTrialFailureCode.EXECUTION_PROFILE_CHANGED
        )
        for candidate in iter_exception_tree(error)
    )


def _accepted_candidate_pricing_matches_target(lease: EvalRunLease, target: CorpusTarget) -> bool:
    exposure = lease.run.spec.invocation.authored_suite_exposure
    if exposure is None:
        return True
    current = (
        ()
        if target.price_book is None
        else (pricing_profile_identity(target.price_book).fingerprint,)
    )
    return exposure.candidate_cost.pricing_profile_fingerprints == current


def _accepted_authored_work_matches_runtime(
    lease: EvalRunLease,
    compiled: CompiledCorpusSuite,
    registration: EvalTargetRegistration,
) -> bool:
    """Rebind accepted candidate and judge authority to this exact durable run."""

    exposure = lease.run.spec.invocation.authored_suite_exposure
    snapshot = lease.run.spec.invocation.execution_profile_snapshot
    if exposure is None or snapshot is None:
        return False
    case_ids = tuple(case.case_id for case in compiled.run_contract.cases)
    execution_binding = next(
        (item for item in exposure.execution_profiles if item.case_ids == case_ids),
        None,
    )
    if (
        execution_binding is None
        or snapshot.revision != execution_binding.execution_profile_revision
        or snapshot.comparison_revision != execution_binding.execution_profile_comparison_revision
    ):
        return False
    accepted_budget = execution_binding.candidate_cost_budget
    current_budget = lease.run.spec.invocation.cost_budget
    if accepted_budget != (
        None
        if current_budget is None
        else EvalCandidateCostBudgetV1.from_decimal(
            currency=current_budget.currency,
            amount=current_budget.max_estimated_cost,
        )
    ):
        return False

    judge_keys: set[str] = set()
    for case in compiled.corpus.cases:
        if case.suite_id != compiled.run_contract.suite_id:
            continue
        for assertion in case.assertions:
            if type(assertion) is ModelJudgeAssertionSpec:
                judge_keys.add(assertion.evaluator_key)
            elif type(assertion) is StructuredModelJudgeAssertionSpec:
                judge_keys.add(assertion.judge_profile_key)
    current_profiles = {
        profile.key: profile for profile in registration.catalog_entry.judge_profiles
    }
    accepted_profiles = {profile.profile_key: profile for profile in exposure.judge_profiles}
    if not judge_keys.issubset(accepted_profiles):
        return False
    return all(
        (current := current_profiles.get(key)) is not None
        and accepted_profiles[key].judge_profile_revision == current.revision
        and accepted_profiles[key].judge_profile_comparison_revision == current.comparison_revision
        for key in accepted_profiles
    )


@dataclass(frozen=True, slots=True)
class _PreparedEvalRun:
    target: CorpusTarget
    compiled: CompiledCorpusSuite
    execution_profile: ExecutionProfileIdentity
    scenario: EvalScenarioDocumentV2 | None = None
    scenario_binding: ScenarioLaunchBindingV2 | None = None


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
                diagnostic=EvalRunFailureDiagnostic(
                    reason=EvalRunFailureReason.TARGET_REGISTRATION_MISSING
                ),
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
            if isinstance(preflight_result, _EvalFailure):
                failure = asyncio.create_task(
                    self._finalize_failure(
                        lease.claim, preflight_result.code, diagnostic=preflight_result.diagnostic
                    ),
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
    ) -> _PreparedEvalRun | _EvalFailure:
        try:
            corpus = await self._config.store.load_corpus(lease.run.spec.corpus_revision)
        except asyncio.CancelledError:
            raise
        except Exception:
            return _failure(
                EvalRunFailureCode.CORPUS_UNAVAILABLE, EvalRunFailureReason.CORPUS_LOAD_FAILED
            )
        if corpus is None:
            return _failure(
                EvalRunFailureCode.CORPUS_UNAVAILABLE, EvalRunFailureReason.CORPUS_MISSING
            )

        try:
            target = target_for_eval_invocation(
                registration.execution_target(),
                lease.run.spec.invocation,
            )
            expected_profile = lease.run.spec.invocation.execution_profile
            expected_snapshot = lease.run.spec.invocation.execution_profile_snapshot
            if expected_profile is None:
                return _failure(
                    EvalRunFailureCode.TARGET_UNAVAILABLE,
                    EvalRunFailureReason.EXECUTION_PROFILE_MISSING,
                )
            prepared_profile = await self._config.registry.prepare_execution_profile(
                target.key,
                effective_target=target,
            )
            if prepared_profile.binding != expected_profile or (
                expected_snapshot is not None and prepared_profile.snapshot != expected_snapshot
            ):
                return _failure(
                    EvalRunFailureCode.TARGET_UNAVAILABLE,
                    EvalRunFailureReason.EXECUTION_PROFILE_CHANGED,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            return _failure(
                EvalRunFailureCode.TARGET_UNAVAILABLE, EvalRunFailureReason.TARGET_PREFLIGHT_FAILED
            )

        scenario_invocation = lease.run.spec.invocation.scenario
        if scenario_invocation is not None:
            try:
                scenario = await self._config.store.load_scenario(
                    scenario_invocation.scenario_revision
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                return _failure(
                    EvalRunFailureCode.CORPUS_UNAVAILABLE, EvalRunFailureReason.CORPUS_LOAD_FAILED
                )
            if scenario is None:
                return _failure(
                    EvalRunFailureCode.CORPUS_UNAVAILABLE, EvalRunFailureReason.CORPUS_MISSING
                )
            authored_suite: EvalSuiteDocument | None = None
            if scenario_invocation.authored_suite_revision is not None:
                try:
                    authored_suite = await self._config.store.load_authored_suite(
                        scenario_invocation.authored_suite_revision
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    return _failure(
                        EvalRunFailureCode.CORPUS_UNAVAILABLE,
                        EvalRunFailureReason.CORPUS_LOAD_FAILED,
                    )
                if authored_suite is None:
                    return _failure(
                        EvalRunFailureCode.CORPUS_UNAVAILABLE, EvalRunFailureReason.CORPUS_MISSING
                    )
            try:
                settings = scenario_launch_settings_from_invocation(
                    scenario_invocation,
                    max_concurrency=lease.run.spec.max_concurrency,
                    max_steps=lease.run.spec.invocation.max_steps,
                    limits=lease.run.spec.invocation.limits,
                    cost_budget=lease.run.spec.invocation.cost_budget,
                )
                preflight = await preflight_eval_scenario(
                    scenario,
                    registration.execution_target(),
                    settings,
                    actor_authorized=True,
                    project_root=registration.manifest_project_root,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                return _failure(
                    EvalRunFailureCode.TARGET_UNAVAILABLE,
                    EvalRunFailureReason.TARGET_PREFLIGHT_FAILED,
                )
            binding = preflight.binding
            if (
                not preflight.ready
                or binding is None
                or binding.revision != scenario_invocation.binding_revision
            ):
                return _failure(
                    EvalRunFailureCode.TARGET_UNAVAILABLE,
                    EvalRunFailureReason.TARGET_PREFLIGHT_FAILED,
                )
            prepared = await asyncio.to_thread(
                self._compile_loaded_scenario,
                corpus,
                scenario,
                binding,
                authored_suite,
                lease,
                registration,
                target,
                prepared_profile.binding.runtime_execution_profile,
            )
            return prepared

        return await asyncio.to_thread(
            self._compile_loaded_corpus,
            corpus,
            lease,
            registration,
            target,
            prepared_profile.binding.runtime_execution_profile,
        )

    def _compile_loaded_corpus(
        self,
        corpus: EvalCorpusDocument,
        lease: EvalRunLease,
        registration: EvalTargetRegistration,
        target: CorpusTarget,
        execution_profile: ExecutionProfileIdentity,
    ) -> _PreparedEvalRun | _EvalFailure:
        try:
            identity = evaluation_target_identity(
                target,
                project_root=registration.manifest_project_root,
            )
        except Exception:
            return _failure(
                EvalRunFailureCode.TARGET_UNAVAILABLE, EvalRunFailureReason.TARGET_IDENTITY_FAILED
            )
        if identity.app_manifest_fingerprint != registration.catalog_entry.app_manifest_fingerprint:
            return _failure(
                EvalRunFailureCode.TARGET_UNAVAILABLE, EvalRunFailureReason.TARGET_IDENTITY_CHANGED
            )
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
            exposure = lease.run.spec.invocation.authored_suite_exposure
            snapshot = lease.run.spec.invocation.execution_profile_snapshot
            if lease.run.spec.invocation.authored_suite_revision is not None and (
                exposure is None
                or snapshot is None
                or exposure.trial_policy_revision != compiled.run_contract.trial_policy.revision
                or not _accepted_authored_work_matches_runtime(lease, compiled, registration)
                or lease.run.spec.max_concurrency
                > compiled.run_contract.trial_policy.max_concurrency
                or not _accepted_candidate_pricing_matches_target(lease, target)
            ):
                raise ValueError("Persisted authored run lost its accepted work exposure.")
        except Exception:
            return _failure(
                EvalRunFailureCode.CORPUS_UNAVAILABLE,
                EvalRunFailureReason.CORPUS_COMPILATION_FAILED,
            )
        return _PreparedEvalRun(
            target=target,
            compiled=compiled,
            execution_profile=execution_profile,
        )

    def _compile_loaded_scenario(
        self,
        corpus: EvalCorpusDocument,
        scenario: EvalScenarioDocumentV2,
        binding: ScenarioLaunchBindingV2,
        authored_suite: EvalSuiteDocument | None,
        lease: EvalRunLease,
        registration: EvalTargetRegistration,
        target: CorpusTarget,
        execution_profile: ExecutionProfileIdentity,
    ) -> _PreparedEvalRun | _EvalFailure:
        try:
            scenario_invocation = lease.run.spec.invocation.scenario
            if scenario_invocation is None:
                return _failure(
                    EvalRunFailureCode.CORPUS_UNAVAILABLE,
                    EvalRunFailureReason.CORPUS_COMPILATION_FAILED,
                )
            if authored_suite is None:
                expected_corpus = corpus_for_eval_scenario(
                    scenario,
                    binding,
                    target,
                    project_root=registration.manifest_project_root,
                )
            else:
                authored_case = (
                    next(
                        (item for item in authored_suite.cases if item.id == corpus.cases[0].id),
                        None,
                    )
                    if len(corpus.cases) == 1
                    else None
                )
                if (
                    authored_suite.revision != scenario_invocation.authored_suite_revision
                    or authored_case is None
                    or authored_case.revision != scenario_invocation.authored_case_revision
                ):
                    return _failure(
                        EvalRunFailureCode.CORPUS_UNAVAILABLE,
                        EvalRunFailureReason.CORPUS_COMPILATION_FAILED,
                    )
                expected_corpus = corpus_for_authored_scenario_case(
                    authored_suite,
                    corpus.cases[0].id,
                    scenario,
                    binding,
                    target,
                    project_root=registration.manifest_project_root,
                )
            if expected_corpus != corpus:
                return _failure(
                    EvalRunFailureCode.CORPUS_UNAVAILABLE,
                    EvalRunFailureReason.CORPUS_COMPILATION_FAILED,
                )
            compiled = compile_corpus_suite(corpus, target, lease.run.spec.suite_id)
            if (
                compiled.run_contract.corpus_revision != lease.run.spec.corpus_revision
                or compiled.run_contract.suite_revision != lease.run.spec.suite_revision
            ):
                return _failure(
                    EvalRunFailureCode.CORPUS_UNAVAILABLE,
                    EvalRunFailureReason.CORPUS_COMPILATION_FAILED,
                )
            exposure = lease.run.spec.invocation.authored_suite_exposure
            snapshot = lease.run.spec.invocation.execution_profile_snapshot
            if authored_suite is not None and (
                exposure is None
                or snapshot is None
                or exposure.trial_policy_revision != compiled.run_contract.trial_policy.revision
                or not _accepted_authored_work_matches_runtime(lease, compiled, registration)
                or lease.run.spec.max_concurrency
                > compiled.run_contract.trial_policy.max_concurrency
                or not _accepted_candidate_pricing_matches_target(lease, target)
            ):
                return _failure(
                    EvalRunFailureCode.CORPUS_UNAVAILABLE,
                    EvalRunFailureReason.CORPUS_COMPILATION_FAILED,
                )
        except Exception:
            return _failure(
                EvalRunFailureCode.CORPUS_UNAVAILABLE,
                EvalRunFailureReason.CORPUS_COMPILATION_FAILED,
            )
        return _PreparedEvalRun(
            target=target,
            compiled=compiled,
            execution_profile=execution_profile,
            scenario=scenario,
            scenario_binding=binding,
        )

    async def _execute_compiled_lease(
        self,
        lease: EvalRunLease,
        registration: EvalTargetRegistration,
        prepared: _PreparedEvalRun,
        monitor: asyncio.Task[_ClaimMonitorOutcome],
    ) -> None:
        target = prepared.target

        async def persist_trial_checkpoint(checkpoint: EvalRunTrialCheckpoint) -> None:
            retry_seconds = min(max(self._config.poll_interval_seconds, 0.05), 1.0)
            while True:
                try:
                    await self._config.store.save_trial_checkpoint(
                        lease.claim,
                        checkpoint,
                        redact_json=target.app.redact_json,
                    )
                    return
                except EvalStoreTransientContention:
                    logger.warning(
                        "Durable eval checkpoint publication exhausted its transient "
                        "contention budget; the completed trial remains attached and "
                        "will be retried."
                    )
                    await asyncio.sleep(retry_seconds)

        async def save_trial_checkpoint(case_id, result, public_data) -> None:
            started_at = asyncio.get_running_loop().time()
            try:
                await persist_trial_checkpoint(
                    EvalRunTrialCheckpoint(
                        case_id=case_id,
                        result=result,
                        public_data=public_data,
                    )
                )
            finally:
                logger.debug(
                    "Durable eval checkpoint write finished.",
                    extra={
                        "cayu_eval_store_event": "checkpoint_write",
                        "eval_run_id": lease.claim.run_id,
                        "duration_seconds": (asyncio.get_running_loop().time() - started_at),
                    },
                )

        if prepared.scenario is None:
            checkpoints = await self._config.store.load_trial_checkpoints(lease.claim)
            completed_trials = {
                (checkpoint.case_id, checkpoint.trial_number): (
                    checkpoint.result,
                    checkpoint.public_data,
                )
                for checkpoint in checkpoints
            }

            execution_coro = _run_compiled_corpus_suite(
                target,
                prepared.compiled,
                max_concurrency=lease.run.spec.max_concurrency,
                manifest_project_root=registration.manifest_project_root,
                expected_app_manifest_fingerprint=(
                    registration.catalog_entry.app_manifest_fingerprint
                ),
                expected_execution_profile=prepared.execution_profile,
                native_run_id=lease.run.id,
                execution_capacity=self._config.execution_capacity,
                accepted_exposure=lease.run.spec.invocation.authored_suite_exposure,
                completed_trials=completed_trials,
                trial_completed=save_trial_checkpoint,
            )
        else:
            if prepared.scenario_binding is None:
                raise RuntimeError("Prepared scenario execution lost its launch binding.")
            execution_coro = run_compiled_eval_scenario(
                target,
                prepared.compiled,
                prepared.scenario,
                prepared.scenario_binding,
                store=self._config.store,
                claim=lease.claim,
                max_concurrency=lease.run.spec.max_concurrency,
                poll_seconds=self._config.poll_interval_seconds,
                manifest_project_root=registration.manifest_project_root,
                expected_app_manifest_fingerprint=(
                    registration.catalog_entry.app_manifest_fingerprint
                ),
                expected_execution_profile=prepared.execution_profile,
                execution_capacity=self._config.execution_capacity,
                accepted_exposure=lease.run.spec.invocation.authored_suite_exposure,
            )
        execution = asyncio.create_task(
            execution_coro,
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
                diagnostic=EvalRunFailureDiagnostic(
                    reason=EvalRunFailureReason.EXECUTION_CANCELLED
                ),
                refresh=False,
            )
            return
        failure = execution.exception()
        if failure is not None:
            await self._finalize_failure(
                claim,
                (
                    EvalRunFailureCode.TARGET_UNAVAILABLE
                    if _is_execution_profile_failure(failure)
                    else EvalRunFailureCode.EXECUTION_FAILED
                ),
                diagnostic=_execution_diagnostic(failure),
                refresh=False,
            )
            return
        try:
            started_at = asyncio.get_running_loop().time()
            try:
                await self._config.store.publish_result(
                    claim,
                    execution.result(),
                    redact_json=target.app.redact_json,
                )
            finally:
                logger.debug(
                    "Durable eval result publication finished.",
                    extra={
                        "cayu_eval_store_event": "result_publication",
                        "eval_run_id": claim.run_id,
                        "duration_seconds": asyncio.get_running_loop().time() - started_at,
                    },
                )
        except EvalRunClaimLost:
            return
        except EvalRunStateConflict:
            await self._finish_cancel_if_requested(claim)
        except EvalStoreTransientContention:
            logger.warning(
                "Durable eval result publication was deferred by transient "
                "store contention; retained checkpoints will be reclaimed."
            )
            await self._release_after_interruption(claim)
        except Exception:
            await self._finalize_failure(
                claim,
                EvalRunFailureCode.EXECUTION_FAILED,
                diagnostic=EvalRunFailureDiagnostic(
                    reason=EvalRunFailureReason.RESULT_PUBLICATION_FAILED
                ),
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
        observation_interval = max(
            self._config.poll_interval_seconds,
            EVAL_RUN_MIN_OBSERVATION_INTERVAL_SECONDS,
        )
        maximum_observation_interval = max(
            observation_interval,
            EVAL_RUN_MAX_OBSERVATION_INTERVAL_SECONDS,
        )
        loop = asyncio.get_running_loop()
        next_heartbeat = loop.time() + heartbeat_interval
        while True:
            wait_seconds = min(
                observation_interval,
                max(0.0, next_heartbeat - loop.time()),
            )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_requested.wait(), timeout=wait_seconds)
            if self._stop_requested.is_set():
                return _ClaimMonitorOutcome.STOPPING
            try:
                if loop.time() >= next_heartbeat:
                    record = await self._config.store.heartbeat_run_observation(
                        claim,
                        extend_seconds=self._config.lease_seconds,
                    )
                    next_heartbeat = loop.time() + heartbeat_interval
                else:
                    record = await self._config.store.load_run_observation(claim.run_id)
                    if (
                        record is None
                        or record.ownership is None
                        or record.ownership.epoch != claim.epoch
                    ):
                        return _ClaimMonitorOutcome.CLAIM_LOST
            except asyncio.CancelledError:
                raise
            except EvalStoreTransientContention:
                logger.warning(
                    "Durable eval heartbeat was deferred by transient store contention; "
                    "the live fenced claim will be retried."
                )
                continue
            except Exception:
                return _ClaimMonitorOutcome.CLAIM_LOST
            if record.status is EvalRunStatus.CANCELLING:
                return _ClaimMonitorOutcome.CANCELLING
            if record.status is not EvalRunStatus.RUNNING:
                return _ClaimMonitorOutcome.CLAIM_LOST
            observation_interval = min(
                observation_interval * 2,
                maximum_observation_interval,
            )

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
        diagnostic: EvalRunFailureDiagnostic | None = None,
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
            record = await self._config.store.fail_run(claim, code, diagnostic=diagnostic)
            # Log only the fenced record after durable publication succeeds. The
            # same closed evidence is returned by load_run after worker restart.
            if record.failure_diagnostic is not None:
                logger.warning(
                    "Durable eval failed.",
                    extra={
                        "eval_run_id": record.id,
                        "eval_claim_epoch": claim.epoch,
                        "eval_failure_code": str(record.failure_code),
                        "eval_failure_phase": record.failure_diagnostic.phase,
                        "eval_failure_diagnostic": record.failure_diagnostic.model_dump(
                            mode="json"
                        ),
                    },
                )
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
