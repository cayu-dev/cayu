from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from tests.core.task_invocation_fixtures import unattributed_session_invocation_binding

from cayu.applications import CayuApp
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.base import InMemoryTaskStore, TaskStore
from cayu.tasks.completion_evaluations import (
    CompletionEvaluationBudgetExhausted,
    CompletionEvaluationFailure,
    CompletionEvaluationOutcome,
    CompletionEvaluationRunRequest,
    CompletionEvaluationSettlementRequest,
    completion_evaluation_effect_id,
    completion_evaluation_evidence_sha256,
)
from cayu.tasks.contracts import (
    CompletionCriterionOutcome,
    CompletionEvaluationPolicy,
    CompletionEvaluatorRef,
    CompletionGap,
    CompletionProposalCreate,
    CompletionResultReference,
    CompletionResultResolverRef,
    CompletionSatisfactionBasis,
    CompletionVerdict,
    CompletionVerificationClaimLost,
    CompletionVerifierDecision,
    CompletionVerifierKind,
    CompletionVerifierRef,
    CriterionOutcomeStatus,
    WorkAttemptCreate,
    WorkCompletionConflict,
    WorkContract,
    WorkContractDraft,
    WorkCriterion,
    work_contract_from_draft,
)
from cayu.tasks.creation import TaskCreate
from cayu.verification.completion_evaluators import (
    CompletionEvaluationExecutionError,
    CompletionEvaluationRequest,
    CompletionEvaluationResult,
    CompletionEvaluator,
    CompletionEvaluatorBudgetExhausted,
    CompletionEvaluatorUnavailable,
)
from cayu.verification.completion_verifiers import (
    CompletionVerifierExecutionError,
    CompletionVerifierExecutionRequest,
    CompletionVerifierRequest,
    DeterministicCompletionVerifier,
)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _identity(name: str, version: str = "1") -> ExecutionProfileBehaviorIdentity:
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version=version, implementation_version=version
    )


def _verifier_reference() -> CompletionVerifierRef:
    return CompletionVerifierRef(
        verifier_id="promotion-gate",
        version="v1",
        kind=CompletionVerifierKind.DETERMINISTIC,
        configuration_fingerprint=_digest("promotion-gate-v1"),
    )


def _evaluator_reference() -> CompletionEvaluatorRef:
    return CompletionEvaluatorRef(
        evaluator_id="tau-bench",
        version="v1",
        configuration_fingerprint=_digest("tau-bench-banking-k4"),
    )


def _policy(**overrides: Any) -> CompletionEvaluationPolicy:
    values: dict[str, Any] = {
        "evaluator": _evaluator_reference(),
        "max_runs": 2,
        "timeout_seconds": 5.0,
    }
    values.update(overrides)
    return CompletionEvaluationPolicy(**values)


def _contract(suffix: str, *, policy: CompletionEvaluationPolicy | None = None) -> WorkContract:
    return work_contract_from_draft(
        WorkContractDraft(
            contract_id=f"promotion-{suffix}",
            version=1,
            objective="Promote the candidate only if it beats the champion.",
            criteria=(
                WorkCriterion(
                    criterion_id="beats-parent",
                    ordinal=1,
                    description="Pass^k exceeds the parent's.",
                ),
            ),
            verifier=_verifier_reference(),
            result_resolver=CompletionResultResolverRef(
                resolver_id="promotion-result",
                version="v1",
                configuration_fingerprint=_digest("promotion-result-v1"),
            ),
            evaluation=policy if policy is not None else _policy(),
        )
    )


async def _proposal(store: TaskStore, contract: WorkContract, suffix: str) -> str:
    await store.publish_work_contract(contract)
    session_id = f"promotion-session-{suffix}"
    await store.create_running_task(
        TaskCreate(
            task_id=f"promotion-task-{suffix}",
            type="promotion",
            session_id=session_id,
            work_contract=contract.reference(),
        ),
        session_invocation=unattributed_session_invocation_binding(session_id),
    )
    attempt = await store.begin_work_attempt(
        WorkAttemptCreate(
            attempt_id=f"promotion-attempt-{suffix}",
            task_id=f"promotion-task-{suffix}",
            session_id=session_id,
            contract=contract.reference(),
            execution_profile_fingerprint=_digest("worker-profile"),
        )
    )
    proposal = await store.submit_completion_proposal(
        CompletionProposalCreate(
            proposal_id=f"promotion-proposal-{suffix}",
            attempt_id=attempt.attempt_id,
            result=CompletionResultReference(
                kind="candidate.prompt",
                reference_id=f"candidate-{suffix}",
                digest=_digest(f"candidate-{suffix}"),
            ),
        )
    )
    return proposal.proposal_id


def _execution(proposal_id: str, *, claim: str = "claim-1") -> CompletionVerifierExecutionRequest:
    return CompletionVerifierExecutionRequest(
        proposal_id=proposal_id,
        claim_id=f"{proposal_id}:{claim}",
        decision_id=f"{proposal_id}:decision",
        worker_id="promotion-worker",
        lease_seconds=60,
        execution_timeout_seconds=30.0,
    )


class Gate(DeterministicCompletionVerifier):
    """Deterministic gate over the evaluation receipt: beat the parent's score."""

    def __init__(self, parent_score: float = 0.5) -> None:
        self.parent_score = parent_score
        self.requests: list[CompletionVerifierRequest] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return _identity("promotion-gate")

    async def verify(self, request: CompletionVerifierRequest) -> CompletionVerifierDecision:
        self.requests.append(request)
        assert request.evaluation is not None
        score = request.evaluation.evidence["pass_k"]
        assert isinstance(score, float)
        if score > self.parent_score:
            return CompletionVerifierDecision(
                verdict=CompletionVerdict.ACCEPTED,
                criterion_outcomes=(
                    CompletionCriterionOutcome(
                        criterion_id="beats-parent",
                        status=CriterionOutcomeStatus.SATISFIED,
                        reason_code="score.improved",
                        satisfaction_basis=CompletionSatisfactionBasis.VERIFIER_ASSERTION,
                    ),
                ),
            )
        return CompletionVerifierDecision(
            verdict=CompletionVerdict.REJECTED,
            criterion_outcomes=(
                CompletionCriterionOutcome(
                    criterion_id="beats-parent",
                    status=CriterionOutcomeStatus.UNSATISFIED,
                    reason_code="score.not_improved",
                ),
            ),
            gaps=(CompletionGap(criterion_id="beats-parent", code="score.not_improved"),),
        )


class Bench(CompletionEvaluator):
    def __init__(
        self,
        *results: CompletionEvaluationResult | BaseException,
        reconciled: CompletionEvaluationResult | None = None,
        block: bool = False,
    ) -> None:
        self.results = list(results)
        self.reconciled = reconciled
        self.calls: list[CompletionEvaluationRequest] = []
        self.reconcile_calls: list[CompletionEvaluationRequest] = []
        self.block = block
        self.started = asyncio.Event()

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return _identity("tau-bench-runner")

    async def evaluate(self, request: CompletionEvaluationRequest) -> CompletionEvaluationResult:
        self.calls.append(request)
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    async def reconcile(
        self, request: CompletionEvaluationRequest
    ) -> CompletionEvaluationResult | None:
        self.reconcile_calls.append(request)
        return self.reconciled


def _score(value: float) -> CompletionEvaluationResult:
    return CompletionEvaluationResult(
        evidence={"pass_k": value, "tasks": 20},
        summary=f"pass^4={value}",
        reported_usage={"cost_usd": "1.25", "episodes": 80},
    )


def _app(store: TaskStore, bench: Bench, gate: Gate | None = None) -> CayuApp:
    app = CayuApp(task_store=store, enable_logging=False)
    app.register_completion_verifier(_verifier_reference(), gate or Gate())
    app.register_completion_evaluator(_evaluator_reference(), bench)
    return app


StoreFactory = Callable[[], TaskStore]


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store_factory(request: pytest.FixtureRequest, tmp_path: Path) -> StoreFactory:
    if request.param == "memory":
        store = InMemoryTaskStore()
        return lambda: store
    if request.param == "sqlite":
        path = tmp_path / "tasks.db"
        return lambda: SQLiteTaskStore(path)
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresTaskStore

    dsn = request.getfixturevalue("postgres_dsn")
    return lambda: PostgresTaskStore(dsn, min_size=1, max_size=2, schema_mode=SchemaMode.MIGRATE)


async def _close(store: TaskStore) -> None:
    if not isinstance(store, InMemoryTaskStore):
        await store.close()  # type: ignore[attr-defined]


def test_contracts_without_evaluation_keep_their_canonical_definition() -> None:
    contract = work_contract_from_draft(
        WorkContractDraft(
            contract_id="plain",
            version=1,
            objective="Plain.",
            criteria=(WorkCriterion(criterion_id="done", ordinal=1, description="Done."),),
            verifier=_verifier_reference(),
            result_resolver=CompletionResultResolverRef(
                resolver_id="r", version="v1", configuration_fingerprint=_digest("r")
            ),
        )
    )
    assert "evaluation" not in contract.model_dump(mode="json")
    evaluated = _contract("fingerprint")
    assert evaluated.evaluation == _policy()
    assert (
        evaluated.fingerprint
        != work_contract_from_draft(
            WorkContractDraft(
                **{
                    **evaluated.model_dump(mode="python", exclude={"fingerprint"}),
                    "evaluation": _policy(max_runs=3),
                }
            )
        ).fingerprint
    )


def test_evaluation_receipt_is_trusted_input_to_the_verifier(store_factory: StoreFactory) -> None:
    async def scenario() -> None:
        suffix = uuid4().hex[:12]
        store = store_factory()
        try:
            contract = _contract(suffix)
            proposal_id = await _proposal(store, contract, suffix)
            bench = Bench(_score(0.75))
            gate = Gate()
            app = _app(store, bench, gate)
            decision = await app.verify_completion_proposal(_execution(proposal_id))
            assert decision.verdict is CompletionVerdict.ACCEPTED

            assert len(bench.calls) == 1
            call = bench.calls[0]
            assert call.run_ordinal == 1
            assert call.effect_id == completion_evaluation_effect_id(
                proposal_id=proposal_id, evaluator=_evaluator_reference(), run_ordinal=1
            )
            receipt = gate.requests[0].evaluation
            assert receipt is not None
            assert receipt.effect_id == call.effect_id
            assert receipt.evidence == {"pass_k": 0.75, "tasks": 20}
            assert receipt.evidence_sha256 == completion_evaluation_evidence_sha256(
                receipt.evidence
            )
            assert receipt.proposal_id == proposal_id
            assert receipt.reconciled is False

            (run,) = await app.list_completion_evaluation_runs(proposal_id)
            assert run.settlement is not None
            assert run.settlement.outcome is CompletionEvaluationOutcome.COMPLETED
            assert run.settlement.reported_usage == {"cost_usd": "1.25", "episodes": 80}

            restarted = CayuApp(task_store=store, enable_logging=False)
            assert await restarted.verify_completion_proposal(_execution(proposal_id)) == decision
            assert len(bench.calls) == 1
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_receipt_survives_a_lost_decision_write_without_running_again() -> None:
    class LosesFirstDecision(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True
        failures = 1

        async def record_completion_decision(self, request):
            if self.failures:
                self.failures -= 1
                raise ConnectionError("decision write lost")
            return await super().record_completion_decision(request)

    async def scenario() -> None:
        store = LosesFirstDecision()
        contract = _contract("lost-decision")
        proposal_id = await _proposal(store, contract, "lost-decision")
        bench = Bench(_score(0.25))
        app = _app(store, bench)
        with pytest.raises(ConnectionError):
            await app.verify_completion_proposal(_execution(proposal_id))
        decision = await app.verify_completion_proposal(_execution(proposal_id))
        assert decision.verdict is CompletionVerdict.REJECTED
        assert len(bench.calls) == 1

    asyncio.run(scenario())


def test_evaluator_failure_is_typed_and_runs_are_budgeted() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("budget")
        proposal_id = await _proposal(store, contract, "budget")
        bench = Bench(RuntimeError("benchmark crashed: secret-ish details"), _score(0.1))
        app = _app(store, bench)
        with pytest.raises(CompletionEvaluationExecutionError) as raised:
            await app.verify_completion_proposal(_execution(proposal_id))
        assert "secret-ish" not in str(raised.value)
        assert await store.load_completion_decision_for_proposal(proposal_id) is None
        (first,) = await app.list_completion_evaluation_runs(proposal_id)
        assert first.settlement is not None
        assert first.settlement.outcome is CompletionEvaluationOutcome.FAILED
        assert first.settlement.failure == CompletionEvaluationFailure(code="evaluation.failed")

        decision = await app.verify_completion_proposal(_execution(proposal_id))
        assert decision.verdict is CompletionVerdict.REJECTED
        assert [call.run_ordinal for call in bench.calls] == [1, 2]
        assert bench.calls[0].effect_id != bench.calls[1].effect_id

    asyncio.run(scenario())


def test_exhausted_run_budget_fails_before_another_effect() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("exhausted", policy=_policy(max_runs=1))
        proposal_id = await _proposal(store, contract, "exhausted")
        bench = Bench(RuntimeError("down"), _score(0.9))
        app = _app(store, bench)
        with pytest.raises(CompletionEvaluationExecutionError):
            await app.verify_completion_proposal(_execution(proposal_id))
        with pytest.raises(CompletionEvaluatorBudgetExhausted):
            await app.verify_completion_proposal(_execution(proposal_id))
        assert len(bench.calls) == 1

    asyncio.run(scenario())


class LosesFirstSettlement(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True
    failures = 1

    async def settle_completion_evaluation_run(self, request):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("settlement lost")
        return await super().settle_completion_evaluation_run(request)


def test_unsettled_run_is_reconciled_instead_of_run_again() -> None:
    async def scenario() -> None:
        store = LosesFirstSettlement()
        contract = _contract("reconcile")
        proposal_id = await _proposal(store, contract, "reconcile")
        bench = Bench(_score(0.9), reconciled=_score(0.9))
        app = _app(store, bench)
        with pytest.raises(CompletionVerifierExecutionError, match="settlement lost"):
            await app.verify_completion_proposal(_execution(proposal_id))
        (run,) = await app.list_completion_evaluation_runs(proposal_id)
        assert run.settlement is None

        decision = await app.verify_completion_proposal(_execution(proposal_id))
        assert decision.verdict is CompletionVerdict.ACCEPTED
        assert len(bench.calls) == 1
        assert [call.effect_id for call in bench.reconcile_calls] == [run.effect_id]
        (run,) = await app.list_completion_evaluation_runs(proposal_id)
        assert run.settlement is not None and run.settlement.reconciled is True

    asyncio.run(scenario())


def test_unreconciled_run_is_recorded_unknown_and_the_next_run_proceeds() -> None:
    async def scenario() -> None:
        store = LosesFirstSettlement()
        contract = _contract("unknown")
        proposal_id = await _proposal(store, contract, "unknown")
        bench = Bench(_score(0.9), _score(0.8))
        app = _app(store, bench)
        with pytest.raises(CompletionVerifierExecutionError, match="settlement lost"):
            await app.verify_completion_proposal(_execution(proposal_id))
        decision = await app.verify_completion_proposal(_execution(proposal_id))
        assert decision.verdict is CompletionVerdict.ACCEPTED
        first, second = await app.list_completion_evaluation_runs(proposal_id)
        assert first.settlement is not None
        assert first.settlement.outcome is CompletionEvaluationOutcome.OUTCOME_UNKNOWN
        assert second.settlement is not None
        assert second.settlement.outcome is CompletionEvaluationOutcome.COMPLETED
        assert [call.run_ordinal for call in bench.calls] == [1, 2]

    asyncio.run(scenario())


def test_evaluation_timeout_is_settled_and_typed() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("timeout", policy=_policy(timeout_seconds=0.05))
        proposal_id = await _proposal(store, contract, "timeout")
        bench = Bench(block=True)
        app = _app(store, bench)
        with pytest.raises(CompletionEvaluationExecutionError, match="timeout"):
            await app.verify_completion_proposal(_execution(proposal_id))
        (run,) = await app.list_completion_evaluation_runs(proposal_id)
        assert run.settlement is not None
        assert run.settlement.outcome is CompletionEvaluationOutcome.TIMED_OUT

    asyncio.run(scenario())


def test_caller_cancellation_settles_the_run_and_publishes_nothing() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("cancel")
        proposal_id = await _proposal(store, contract, "cancel")
        bench = Bench(block=True)
        app = _app(store, bench)
        running = asyncio.create_task(app.verify_completion_proposal(_execution(proposal_id)))
        await bench.started.wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert await app.drain_verified_completions(timeout_s=5.0)
        (run,) = await app.list_completion_evaluation_runs(proposal_id)
        assert run.settlement is not None
        assert run.settlement.outcome is CompletionEvaluationOutcome.CANCELLED
        assert await store.load_completion_decision_for_proposal(proposal_id) is None

    asyncio.run(scenario())


def test_missing_or_changed_evaluator_fails_before_claim() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("unregistered")
        proposal_id = await _proposal(store, contract, "unregistered")
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(_verifier_reference(), Gate())
        with pytest.raises(CompletionEvaluatorUnavailable, match="not registered"):
            await app.verify_completion_proposal(_execution(proposal_id))
        assert await store.load_completion_verification_claim(proposal_id) is None

        class Drifting(Bench):
            version = "1"

            @property
            def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
                return _identity("tau-bench-runner", self.version)

        bench = Drifting(_score(0.9))
        app.register_completion_evaluator(_evaluator_reference(), bench)
        with pytest.raises(ValueError, match="already registered"):
            app.register_completion_evaluator(_evaluator_reference(), Bench())
        bench.version = "2"
        with pytest.raises(CompletionEvaluatorUnavailable, match="changed"):
            await app.verify_completion_proposal(_execution(proposal_id))
        assert await store.load_completion_verification_claim(proposal_id) is None

    asyncio.run(scenario())


def test_verifier_request_rejects_a_receipt_from_another_chain() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        first = _contract("chain-a")
        second = _contract("chain-b")
        first_id = await _proposal(store, first, "chain-a")
        second_id = await _proposal(store, second, "chain-b")
        bench = Bench(_score(0.9))
        gate = Gate()
        app = _app(store, bench, gate)
        await app.verify_completion_proposal(_execution(first_id))
        receipt = gate.requests[0].evaluation
        attempt = await store.load_work_attempt("promotion-attempt-chain-b")
        proposal = await store.load_completion_proposal(second_id)
        assert attempt is not None and proposal is not None
        with pytest.raises(ValueError, match="evaluation receipt conflicts"):
            CompletionVerifierRequest(
                contract=second, attempt=attempt, proposal=proposal, evaluation=receipt
            )

    asyncio.run(scenario())


def test_store_fences_evaluation_intents_and_settles_once(store_factory: StoreFactory) -> None:
    async def scenario() -> None:
        suffix = uuid4().hex[:12]
        store = store_factory()
        try:
            contract = _contract(suffix)
            proposal_id = await _proposal(store, contract, suffix)

            app = CayuApp(task_store=store, enable_logging=False)
            # Take a live claim; the first run fails so no receipt exists yet.
            app.register_completion_verifier(_verifier_reference(), Gate())

            class Refuses(Bench):
                @property
                def execution_profile_identity(self):
                    return _identity("tau-bench-runner")

            refusing = Refuses()

            async def refuse(request):
                raise RuntimeError("stop")

            refusing.evaluate = refuse  # type: ignore[method-assign]
            app.register_completion_evaluator(_evaluator_reference(), refusing)
            with pytest.raises(CompletionEvaluationExecutionError):
                await app.verify_completion_proposal(_execution(proposal_id))
            claim = await store.load_completion_verification_claim(proposal_id)
            assert claim is not None and claim.execution_owner_id is not None
            owner = claim.execution_owner_id
            (failed,) = await store.list_completion_evaluation_runs(proposal_id)
            fingerprint = failed.request.evaluator_profile_fingerprint

            def intent(run_ordinal: int = 2, **overrides: Any) -> CompletionEvaluationRunRequest:
                policy = overrides.pop("policy", _policy())
                values: dict[str, Any] = {
                    "effect_id": completion_evaluation_effect_id(
                        proposal_id=proposal_id,
                        evaluator=policy.evaluator,
                        run_ordinal=run_ordinal,
                    ),
                    "proposal_id": proposal_id,
                    "claim_id": claim.claim_id,
                    "worker_id": claim.worker_id,
                    "execution_owner_id": owner,
                    "claim_attempt_number": claim.attempt_number,
                    "policy": policy,
                    "evaluator_profile_fingerprint": fingerprint,
                    "run_ordinal": run_ordinal,
                }
                values.update(overrides)
                return CompletionEvaluationRunRequest(**values)

            with pytest.raises(CompletionVerificationClaimLost):
                await store.record_completion_evaluation_run(intent(worker_id="intruder"))
            with pytest.raises(WorkCompletionConflict, match="contract evaluation policy"):
                await store.record_completion_evaluation_run(
                    intent(policy=_policy(timeout_seconds=9.0))
                )
            with pytest.raises(WorkCompletionConflict, match="one evaluator profile"):
                await store.record_completion_evaluation_run(
                    intent(evaluator_profile_fingerprint=_digest("other"))
                )
            second = await store.record_completion_evaluation_run(intent())
            assert await store.record_completion_evaluation_run(intent()) == second
            with pytest.raises(ValueError):
                intent(run_ordinal=3)  # outside the contract's two-run budget
            settlement = CompletionEvaluationSettlementRequest(
                effect_id=second.effect_id,
                outcome=CompletionEvaluationOutcome.COMPLETED,
                latency_ms=7,
                evidence={"pass_k": 0.7},
                evidence_sha256=completion_evaluation_evidence_sha256({"pass_k": 0.7}),
            )
            settled = await store.settle_completion_evaluation_run(settlement)
            assert await store.settle_completion_evaluation_run(settlement) == settled
            with pytest.raises(WorkCompletionConflict):
                await store.settle_completion_evaluation_run(
                    settlement.model_copy(update={"latency_ms": 8})
                )
            assert [
                run.run_ordinal for run in await store.list_completion_evaluation_runs(proposal_id)
            ] == [1, 2]
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_store_budget_is_enforced_atomically() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("store-budget", policy=_policy(max_runs=1))
        proposal_id = await _proposal(store, contract, "store-budget")
        bench = Bench(RuntimeError("down"))
        app = _app(store, bench)
        with pytest.raises(CompletionEvaluationExecutionError):
            await app.verify_completion_proposal(_execution(proposal_id))
        claim = await store.load_completion_verification_claim(proposal_id)
        assert claim is not None and claim.execution_owner_id is not None
        (run,) = await store.list_completion_evaluation_runs(proposal_id)
        with pytest.raises(ValueError):
            CompletionEvaluationRunRequest(
                **{
                    **run.request.model_dump(mode="python"),
                    "run_ordinal": 2,
                    "effect_id": completion_evaluation_effect_id(
                        proposal_id=proposal_id,
                        evaluator=_evaluator_reference(),
                        run_ordinal=2,
                    ),
                }
            )
        assert CompletionEvaluationBudgetExhausted.__mro__[1] is ValueError

    asyncio.run(scenario())


def test_provider_verifier_prompt_carries_the_receipt_as_trusted_evidence() -> None:
    from cayu.messages import Message, MessageRole
    from cayu.verification.provider_completion_verifiers import (
        compose_provider_completion_verifier_messages,
    )

    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("prompt")
        proposal_id = await _proposal(store, contract, "prompt")
        gate = Gate()
        app = _app(store, Bench(_score(0.9)), gate)
        await app.verify_completion_proposal(_execution(proposal_id))
        request = gate.requests[0]
        (message,) = compose_provider_completion_verifier_messages(
            [Message.text(MessageRole.USER, "Judge.")], request
        )
        text = message.content[-1].text  # type: ignore[union-attr]
        assert "trusted evidence" in text
        document = json.loads(text.split("Work contract:\n", 1)[1])
        assert document["independent_evaluation"]["evidence"] == {"pass_k": 0.9, "tasks": 20}

    asyncio.run(scenario())


def test_evaluation_failures_are_detached_and_secret_safe() -> None:
    from tests.core.test_completion_verifier_adapters import _assert_secret_absent_from_cayu_error

    from cayu.vaults.redaction import SecretRedactor

    secret = "completion-evaluation-secret-canary"

    async def scenario() -> BaseException:
        store = InMemoryTaskStore()
        contract = work_contract_from_draft(
            WorkContractDraft(
                **{
                    **_contract("secret").model_dump(mode="python", exclude={"fingerprint"}),
                    "objective": f"Private objective containing {secret}",
                }
            )
        )
        proposal_id = await _proposal(store, contract, "secret")
        app = CayuApp(
            task_store=store, secret_redactor=SecretRedactor(secret), enable_logging=False
        )
        app.register_completion_verifier(_verifier_reference(), Gate())
        app.register_completion_evaluator(
            _evaluator_reference(), Bench(RuntimeError(f"benchmark exposed {secret}"))
        )
        with pytest.raises(CompletionEvaluationExecutionError) as raised:
            await app.verify_completion_proposal(_execution(proposal_id))
        return raised.value

    _assert_secret_absent_from_cayu_error(asyncio.run(scenario()), secret)


@pytest.mark.parametrize("completed", [False, True])
def test_recovery_rejects_a_different_evaluator_profile(completed: bool) -> None:
    async def scenario() -> None:
        now = [datetime(2026, 10, 7, tzinfo=UTC)]
        store = LosesFirstSettlement(clock=lambda: now[0], ownership_clock=lambda: now[0])
        proposal_id = await _proposal(store, _contract("recovery-profile"), "recovery-profile")
        app = _app(store, Bench(_score(0.1)))
        with pytest.raises(CompletionVerifierExecutionError, match="settlement lost"):
            await app.verify_completion_proposal(_execution(proposal_id))
        if completed:
            (run,) = await store.list_completion_evaluation_runs(proposal_id)
            evidence = _score(0.1).evidence
            await store.settle_completion_evaluation_run(
                CompletionEvaluationSettlementRequest(
                    effect_id=run.effect_id,
                    outcome=CompletionEvaluationOutcome.COMPLETED,
                    latency_ms=0,
                    evidence=evidence,
                    evidence_sha256=completion_evaluation_evidence_sha256(evidence),
                )
            )
        before = await store.list_completion_evaluation_runs(proposal_id)
        now[0] += timedelta(minutes=5)

        class Replacement(Bench):
            @property
            def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
                return _identity("tau-bench-runner", "2")

        bench = Replacement(_score(0.9), reconciled=_score(0.9))
        replacement = _app(store, bench)
        with pytest.raises(CompletionEvaluatorUnavailable, match="different evaluator profile"):
            await replacement.verify_completion_proposal(_execution(proposal_id, claim="claim-2"))
        assert not bench.calls and not bench.reconcile_calls
        assert await store.list_completion_evaluation_runs(proposal_id) == before
        assert await store.load_completion_decision_for_proposal(proposal_id) is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure", ["exception", "timeout", "invalid", "self_cancel", "caller_cancel"]
)
def test_failed_reconciliation_preserves_original_effect(failure: str) -> None:
    async def scenario() -> None:
        store = LosesFirstSettlement()
        policy = _policy(timeout_seconds=0.05)
        proposal_id = await _proposal(
            store, _contract("reconcile-failure", policy=policy), "reconcile-failure"
        )

        class RecoveringBench(Bench):
            fail = True
            reconciling = asyncio.Event()

            async def reconcile(self, request):
                self.reconcile_calls.append(request)
                self.reconciling.set()
                if self.fail:
                    if failure == "exception":
                        raise ConnectionError("lookup unavailable")
                    if failure in {"timeout", "caller_cancel"}:
                        await asyncio.Event().wait()
                    if failure == "self_cancel":
                        raise asyncio.CancelledError()
                    return object()
                return _score(0.1)

        bench = RecoveringBench(_score(0.1), _score(0.9))
        app = _app(store, bench)
        with pytest.raises(CompletionVerifierExecutionError, match="settlement lost"):
            await app.verify_completion_proposal(_execution(proposal_id))
        before = await store.list_completion_evaluation_runs(proposal_id)
        if failure == "caller_cancel":
            running = asyncio.create_task(app.verify_completion_proposal(_execution(proposal_id)))
            await bench.reconciling.wait()
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            assert await app.drain_verified_completions(timeout_s=5.0)
        else:
            with pytest.raises(CompletionEvaluationExecutionError):
                await app.verify_completion_proposal(_execution(proposal_id))
        assert await store.list_completion_evaluation_runs(proposal_id) == before
        assert await store.load_completion_decision_for_proposal(proposal_id) is None
        assert len(bench.calls) == 1
        bench.fail = False
        decision = await app.verify_completion_proposal(_execution(proposal_id))
        assert decision.verdict is CompletionVerdict.REJECTED
        assert len(bench.calls) == 1
        assert {call.effect_id for call in bench.reconcile_calls} == {before[0].effect_id}

    asyncio.run(scenario())
