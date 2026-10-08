"""Runtime-owned run-once execution of independent completion evaluators.

This owner runs inside the completion-verifier coordinator's adapter slot, before
the verifier, so the verification claim's heartbeat, caller cancellation and
cancellation-resistant draining cover the evaluation as well. It adds the
evaluation lifecycle: an existing receipt is returned without another effect, an
earlier owner's unfinished run is reconciled through the evaluator, and a new
run's intent is durable before the evaluator is called and settled exactly once
afterwards.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from cayu._task_wait import await_shielded_task_outcome
from cayu.runtime._task_store_operation_boundary import (
    capture_task_store_operation,
    raise_task_store_operation_failure,
)
from cayu.tasks.completion_evaluations import (
    CompletionEvaluationBudgetExhausted,
    CompletionEvaluationFailure,
    CompletionEvaluationOutcome,
    CompletionEvaluationReceipt,
    CompletionEvaluationRun,
    CompletionEvaluationRunRequest,
    CompletionEvaluationSettlementRequest,
    completion_evaluation_effect_id,
    completion_evaluation_evidence_sha256,
    completion_evaluation_receipt,
)
from cayu.tasks.contracts import CompletionEvaluationPolicy, WorkCompletionConflict
from cayu.tasks.store import TaskStore
from cayu.vaults.redaction import SecretRedactor
from cayu.verification.completion_evaluators import (
    CompletionEvaluationExecutionError,
    CompletionEvaluationRequest,
    CompletionEvaluationResult,
    CompletionEvaluator,
    CompletionEvaluatorBudgetExhausted,
    CompletionEvaluatorUnavailable,
    copy_completion_evaluation_result,
)
from cayu.verification.completion_verifiers import CompletionVerifierRequest


@dataclass(frozen=True, slots=True)
class EvaluationExecutionAuthority:
    """The live claim tuple one evaluation runs under."""

    proposal_id: str
    claim_id: str
    worker_id: str
    execution_owner_id: str
    claim_attempt_number: int
    evaluator_profile_fingerprint: str


@dataclass(frozen=True, slots=True)
class _RunOutcome:
    settlement: CompletionEvaluationSettlementRequest
    failure: BaseException | None = None


def _completed_settlement(
    effect_id: str,
    result: CompletionEvaluationResult,
    *,
    latency_ms: int,
    reconciled: bool,
) -> CompletionEvaluationSettlementRequest:
    return CompletionEvaluationSettlementRequest(
        effect_id=effect_id,
        outcome=CompletionEvaluationOutcome.COMPLETED,
        latency_ms=latency_ms,
        reconciled=reconciled,
        evidence=result.evidence,
        evidence_sha256=completion_evaluation_evidence_sha256(result.evidence),
        summary=result.summary,
        reported_usage=result.reported_usage,
    )


def _failed_settlement(
    effect_id: str,
    outcome: CompletionEvaluationOutcome,
    code: str,
    *,
    latency_ms: int,
    reconciled: bool = False,
) -> CompletionEvaluationSettlementRequest:
    return CompletionEvaluationSettlementRequest(
        effect_id=effect_id,
        outcome=outcome,
        latency_ms=latency_ms,
        reconciled=reconciled,
        failure=CompletionEvaluationFailure(code=code),
    )


class CompletionEvaluationRuntime:
    """Obtain one proposal's evaluation receipt, running the evaluator at most as budgeted."""

    def __init__(self, *, secret_redactor: SecretRedactor) -> None:
        self._secret_redactor = secret_redactor

    async def obtain_receipt(
        self,
        *,
        store: TaskStore,
        evaluator: CompletionEvaluator,
        policy: CompletionEvaluationPolicy,
        request: CompletionVerifierRequest,
        authority: EvaluationExecutionAuthority,
    ) -> CompletionEvaluationReceipt:
        runs = await self._list_runs(store, authority.proposal_id)
        for run in runs:
            if run.request.evaluator_profile_fingerprint != authority.evaluator_profile_fingerprint:
                raise CompletionEvaluatorUnavailable(
                    "The recorded evaluation requires a different evaluator profile."
                )
        for run in runs:
            if (
                run.settlement is not None
                and run.settlement.outcome is CompletionEvaluationOutcome.COMPLETED
            ):
                # Exact replay: the receipt is durable, so never run the effect again.
                return completion_evaluation_receipt(run)
        if runs and runs[-1].settlement is None:
            receipt = await self._reconcile(store, evaluator, runs[-1], request)
            if receipt is not None:
                return receipt
            runs = await self._list_runs(store, authority.proposal_id)
        if len(runs) >= policy.max_runs:
            raise CompletionEvaluatorBudgetExhausted(
                "The evaluation run budget for this proposal is exhausted."
            )
        run_ordinal = len(runs) + 1
        effect_id = completion_evaluation_effect_id(
            proposal_id=authority.proposal_id,
            evaluator=policy.evaluator,
            run_ordinal=run_ordinal,
        )
        run = await self._record_intent(
            store,
            CompletionEvaluationRunRequest(
                effect_id=effect_id,
                proposal_id=authority.proposal_id,
                claim_id=authority.claim_id,
                worker_id=authority.worker_id,
                execution_owner_id=authority.execution_owner_id,
                claim_attempt_number=authority.claim_attempt_number,
                policy=policy,
                evaluator_profile_fingerprint=authority.evaluator_profile_fingerprint,
                run_ordinal=run_ordinal,
            ),
        )
        evaluation_request = self._evaluation_request(run, request)
        task = asyncio.create_task(
            self._run(evaluator.evaluate, evaluation_request, policy, reconciled=False),
            name="cayu-completion-evaluation",
        )
        outcome, cancellation = await self._await_run(task, effect_id)
        settled = await self._settle(store, outcome.settlement, cancellation=cancellation)
        if isinstance(outcome.failure, asyncio.CancelledError):
            raise CompletionEvaluationExecutionError(
                "The evaluation run was cancelled without caller cancellation."
            ) from None
        if outcome.failure is not None:
            raise outcome.failure
        return completion_evaluation_receipt(settled)

    @staticmethod
    def _evaluation_request(
        run: CompletionEvaluationRun,
        request: CompletionVerifierRequest,
    ) -> CompletionEvaluationRequest:
        return CompletionEvaluationRequest(
            effect_id=run.effect_id,
            run_ordinal=run.run_ordinal,
            contract=request.contract,
            attempt=request.attempt,
            proposal=request.proposal,
        )

    async def _reconcile(
        self,
        store: TaskStore,
        evaluator: CompletionEvaluator,
        run: CompletionEvaluationRun,
        request: CompletionVerifierRequest,
    ) -> CompletionEvaluationReceipt | None:
        """Settle an earlier owner's unfinished run from the evaluator's own records."""

        policy = run.request.policy
        task = asyncio.create_task(
            self._run(
                evaluator.reconcile,
                self._evaluation_request(run, request),
                policy,
                reconciled=True,
            ),
            name="cayu-completion-evaluation-reconcile",
        )
        outcome, cancellation = await self._await_run(task, run.effect_id)
        # A failed lookup says nothing about the original effect's outcome.
        # Keep its intent unsettled so recovery can reconcile the same effect.
        if outcome.failure is not None:
            if cancellation is not None:
                raise cancellation
            if isinstance(outcome.failure, asyncio.CancelledError):
                raise CompletionEvaluationExecutionError(
                    "Evaluation reconciliation was cancelled without caller cancellation."
                ) from None
            raise outcome.failure
        try:
            settled = await self._settle(store, outcome.settlement, cancellation=cancellation)
        except WorkCompletionConflict:
            # The earlier owner settled first; its observed outcome wins.
            settled = next(
                item
                for item in await self._list_runs(store, run.proposal_id)
                if item.effect_id == run.effect_id
            )
        if settled.settlement is not None and (
            settled.settlement.outcome is CompletionEvaluationOutcome.COMPLETED
        ):
            return completion_evaluation_receipt(settled)
        return None

    async def _run(
        self,
        operation,
        request: CompletionEvaluationRequest,
        policy: CompletionEvaluationPolicy,
        *,
        reconciled: bool,
    ) -> _RunOutcome:
        started = time.monotonic()

        def elapsed() -> int:
            return max(0, int((time.monotonic() - started) * 1000))

        try:
            async with asyncio.timeout(policy.timeout_seconds):
                result = await operation(request)
        except TimeoutError:
            return _RunOutcome(
                settlement=_failed_settlement(
                    request.effect_id,
                    CompletionEvaluationOutcome.TIMED_OUT,
                    "evaluation.timed_out",
                    latency_ms=elapsed(),
                    reconciled=reconciled,
                ),
                failure=CompletionEvaluationExecutionError(
                    "The evaluation run exceeded its timeout."
                ),
            )
        except asyncio.CancelledError as cancellation:
            return _RunOutcome(
                settlement=_failed_settlement(
                    request.effect_id,
                    CompletionEvaluationOutcome.CANCELLED,
                    "evaluation.cancelled",
                    latency_ms=elapsed(),
                    reconciled=reconciled,
                ),
                failure=cancellation,
            )
        except Exception:
            # Evaluator diagnostics can contain anything; keep them out of evidence.
            return _RunOutcome(
                settlement=_failed_settlement(
                    request.effect_id,
                    CompletionEvaluationOutcome.FAILED,
                    "evaluation.failed",
                    latency_ms=elapsed(),
                    reconciled=reconciled,
                ),
                failure=CompletionEvaluationExecutionError("The evaluation run failed."),
            )
        if result is None and reconciled:
            return _RunOutcome(
                settlement=_failed_settlement(
                    request.effect_id,
                    CompletionEvaluationOutcome.OUTCOME_UNKNOWN,
                    "evaluation.unreconciled",
                    latency_ms=elapsed(),
                    reconciled=True,
                )
            )
        try:
            copied = copy_completion_evaluation_result(result)
        except (TypeError, ValueError):
            return _RunOutcome(
                settlement=_failed_settlement(
                    request.effect_id,
                    CompletionEvaluationOutcome.FAILED,
                    "evaluation.invalid_result",
                    latency_ms=elapsed(),
                    reconciled=reconciled,
                ),
                failure=CompletionEvaluationExecutionError(
                    "The evaluator returned an invalid result."
                ),
            )
        return _RunOutcome(
            settlement=_completed_settlement(
                request.effect_id, copied, latency_ms=elapsed(), reconciled=reconciled
            )
        )

    @staticmethod
    async def _await_run(
        task: asyncio.Task[_RunOutcome],
        effect_id: str,
    ) -> tuple[_RunOutcome, asyncio.CancelledError | None]:
        """Wait for the run; caller cancellation cancels only the run."""

        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                return await asyncio.shield(task), cancellation
            except asyncio.CancelledError as delivered:
                if cancellation is None:
                    cancellation = delivered
                if task.done():
                    break
                task.cancel()
        try:
            return task.result(), cancellation
        except asyncio.CancelledError:
            return (
                _RunOutcome(
                    settlement=_failed_settlement(
                        effect_id,
                        CompletionEvaluationOutcome.CANCELLED,
                        "evaluation.cancelled",
                        latency_ms=0,
                    ),
                    failure=cancellation,
                ),
                cancellation,
            )

    async def _list_runs(
        self, store: TaskStore, proposal_id: str
    ) -> tuple[CompletionEvaluationRun, ...]:
        outcome = await capture_task_store_operation(
            lambda: store.list_completion_evaluation_runs(proposal_id),
            operation_name="Completion evaluation run lookup",
            redactor=self._secret_redactor,
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        result = outcome.result
        if type(result) is not tuple or any(
            type(item) is not CompletionEvaluationRun or item.proposal_id != proposal_id
            for item in result
        ):
            raise CompletionEvaluationExecutionError(
                "Task store returned invalid completion evaluation runs."
            )
        return result

    async def _record_intent(
        self, store: TaskStore, request: CompletionEvaluationRunRequest
    ) -> CompletionEvaluationRun:
        outcome = await capture_task_store_operation(
            lambda: store.record_completion_evaluation_run(request),
            operation_name="Completion evaluation intent",
            redactor=self._secret_redactor,
            mutation_store=store,
            mutation_method_name="record_completion_evaluation_run",
        )
        failure = outcome.failure
        if failure is not None:
            if type(failure) is CompletionEvaluationBudgetExhausted:
                raise CompletionEvaluatorBudgetExhausted(str(failure)) from None
            raise_task_store_operation_failure(failure)
        run = outcome.result
        if (
            type(run) is not CompletionEvaluationRun
            or run.effect_id != request.effect_id
            or run.request != request
        ):
            raise CompletionEvaluationExecutionError(
                "Task store returned an evaluation run other than the exact intent."
            )
        if run.settlement is not None:
            raise CompletionEvaluationExecutionError(
                "The evaluation run already settled without a receipt."
            )
        return run

    async def _settle(
        self,
        store: TaskStore,
        request: CompletionEvaluationSettlementRequest,
        *,
        cancellation: asyncio.CancelledError | None,
    ) -> CompletionEvaluationRun:
        """Write the settlement in its own task so cancellation cannot skip it."""

        async def settle() -> CompletionEvaluationRun | None:
            outcome = await capture_task_store_operation(
                lambda: store.settle_completion_evaluation_run(request),
                operation_name="Completion evaluation settlement",
                redactor=self._secret_redactor,
                mutation_store=store,
                mutation_method_name="settle_completion_evaluation_run",
            )
            if outcome.failure is not None:
                raise_task_store_operation_failure(outcome.failure)
            return outcome.result

        task = asyncio.create_task(settle(), name="cayu-completion-evaluation-settlement")
        shielded = await await_shielded_task_outcome(task)
        settlement_failure = shielded.error
        recorded = shielded.result
        if settlement_failure is None and (
            type(recorded) is not CompletionEvaluationRun
            or recorded.effect_id != request.effect_id
            or recorded.settlement is None
        ):
            settlement_failure = CompletionEvaluationExecutionError(
                "Task store returned an invalid evaluation settlement."
            )
        cancellation = cancellation or shielded.cancellation
        if cancellation is not None:
            if settlement_failure is not None:
                raise cancellation from settlement_failure
            raise cancellation
        if settlement_failure is not None:
            raise settlement_failure
        assert type(recorded) is CompletionEvaluationRun
        return recorded


__all__ = ["CompletionEvaluationRuntime", "EvaluationExecutionAuthority"]
