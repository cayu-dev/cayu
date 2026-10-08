from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from tests.core.task_invocation_fixtures import unattributed_session_invocation_binding

from cayu.applications import CayuApp
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.budgets.usage import UsageMetrics
from cayu.messages import Message, MessageRole, TextPart, ToolResultPart
from cayu.providers import ModelProvider, ModelProviderError, ModelRequest, ModelStreamEvent
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.base import InMemoryTaskStore, TaskStore
from cayu.tasks.completion_verifier_dispatches import (
    CompletionVerifierDecodeStatus,
    CompletionVerifierDispatchBudget,
    CompletionVerifierDispatchBudgetExhausted,
    CompletionVerifierDispatchFailure,
    CompletionVerifierDispatchOutcome,
    CompletionVerifierDispatchRequest,
    CompletionVerifierDispatchSettlementRequest,
    CompletionVerifierUsageStatus,
    completion_verifier_dispatch_id,
)
from cayu.tasks.contracts import (
    CompletionProposalCreate,
    CompletionResultReference,
    CompletionResultResolverRef,
    CompletionSatisfactionBasis,
    CompletionVerdict,
    CompletionVerificationClaimLost,
    CompletionVerificationClaimRequest,
    CompletionVerifierKind,
    CompletionVerifierRef,
    CriterionOutcomeStatus,
    WorkAttemptCreate,
    WorkCompletionConflict,
    WorkContract,
    WorkContractDraft,
    WorkCriterion,
    WorkEvidenceReference,
    WorkEvidenceRequirement,
    work_contract_from_draft,
)
from cayu.tasks.creation import TaskCreate
from cayu.verification.completion_verifiers import (
    CompletionVerifierExecutionError,
    CompletionVerifierExecutionRequest,
    CompletionVerifierRequest,
    CompletionVerifierUnavailable,
    DeterministicCompletionVerifier,
)
from cayu.verification.provider_completion_verifiers import (
    ProviderCompletionVerifier,
    ProviderCompletionVerifierBudgetExhausted,
    ProviderCompletionVerifierDecodingError,
    ProviderCompletionVerifierDispatchError,
    ProviderCompletionVerifierTarget,
    decode_provider_completion_verifier_decision,
    summarize_completion_verifier_dispatches,
)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _identity(name: str = "judge", version: str = "1") -> ExecutionProfileBehaviorIdentity:
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version=version, implementation_version=version
    )


def _target(**overrides: object) -> ProviderCompletionVerifierTarget:
    values: dict[str, Any] = {
        "provider_name": "scripted",
        "model": "judge-model",
        "max_input_tokens": 1_000,
        "max_output_tokens": 500,
        "attempt_timeout_seconds": 5.0,
        "retry_policy": RetryPolicy(
            max_attempts=2, max_unknown_attempts=1, initial_delay_s=0.0, jitter_s=0.0
        ),
    }
    values.update(overrides)
    return ProviderCompletionVerifierTarget(**values)


class Judge(ProviderCompletionVerifier):
    def __init__(self, target: ProviderCompletionVerifierTarget | None = None) -> None:
        self._target = target or _target()
        self.requests: list[CompletionVerifierRequest] = []
        self.messages: list[Message] = [Message.text(MessageRole.USER, "Judge the package.")]

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return _identity()

    @property
    def target(self) -> ProviderCompletionVerifierTarget:
        return self._target

    async def build_messages(self, request: CompletionVerifierRequest) -> list[Message]:
        self.requests.append(request)
        return self.messages


def _reference() -> CompletionVerifierRef:
    return CompletionVerifierRef(
        verifier_id="bid-judge",
        version="v1",
        kind=CompletionVerifierKind.PROVIDER,
        configuration_fingerprint=_digest("bid-judge-v1"),
    )


def _contract(suffix: str, *, with_evidence: bool = False) -> WorkContract:
    return work_contract_from_draft(
        WorkContractDraft(
            contract_id=f"judge-contract-{suffix}",
            version=1,
            objective="Publish a ready bid package.",
            criteria=(
                WorkCriterion(
                    criterion_id="ready",
                    ordinal=1,
                    description="The package is ready.",
                    evidence_requirement_ids=("tests",) if with_evidence else (),
                ),
            ),
            evidence_requirements=(
                (
                    WorkEvidenceRequirement(
                        requirement_id="tests", kind="test.report", description="Tests ran."
                    ),
                )
                if with_evidence
                else ()
            ),
            verifier=_reference(),
            result_resolver=CompletionResultResolverRef(
                resolver_id="bid-result",
                version="v1",
                configuration_fingerprint=_digest("bid-result-v1"),
            ),
        )
    )


async def _proposal(
    store: TaskStore,
    contract: WorkContract,
    suffix: str,
    *,
    evidence: tuple[WorkEvidenceReference, ...] = (),
) -> str:
    await store.publish_work_contract(contract)
    task_id = f"judge-task-{suffix}"
    session_id = f"judge-session-{suffix}"
    await store.create_running_task(
        TaskCreate(
            task_id=task_id,
            type="bid",
            session_id=session_id,
            work_contract=contract.reference(),
        ),
        session_invocation=unattributed_session_invocation_binding(session_id),
    )
    attempt = await store.begin_work_attempt(
        WorkAttemptCreate(
            attempt_id=f"judge-attempt-{suffix}",
            task_id=task_id,
            session_id=session_id,
            contract=contract.reference(),
            execution_profile_fingerprint=_digest("worker-profile"),
        )
    )
    proposal = await store.submit_completion_proposal(
        CompletionProposalCreate(
            proposal_id=f"judge-proposal-{suffix}",
            attempt_id=attempt.attempt_id,
            result=CompletionResultReference(
                kind="task.result",
                reference_id=f"result-{suffix}",
                digest=_digest(f"result-{suffix}"),
            ),
            evidence_references=evidence,
        )
    )
    return proposal.proposal_id


def _execution(
    proposal_id: str,
    *,
    claim: str = "claim-1",
    decision: str = "decision-1",
    lease_seconds: int = 60,
    timeout_seconds: float = 30.0,
) -> CompletionVerifierExecutionRequest:
    return CompletionVerifierExecutionRequest(
        proposal_id=proposal_id,
        claim_id=f"{proposal_id}:{claim}",
        decision_id=f"{proposal_id}:{decision}",
        worker_id="judge-worker",
        lease_seconds=lease_seconds,
        execution_timeout_seconds=timeout_seconds,
    )


def _answer(
    verdict: str = "accepted",
    status: str = "satisfied",
    *,
    evidence: list[str] | None = None,
    gap_code: str | None = None,
) -> str:
    outcome: dict[str, object] = {"id": "ready", "status": status, "reason_code": "package.ready"}
    if evidence is not None:
        outcome["evidence"] = evidence
    if gap_code is not None:
        outcome["gap_code"] = gap_code
    return json.dumps({"verdict": verdict, "criteria": [outcome], "constraints": []})


def _completion(text: str, *, input_tokens: int = 120, output_tokens: int = 30):
    return [
        ModelStreamEvent.text_delta(text),
        ModelStreamEvent.completed(
            {"usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}
        ),
    ]


def _rate_limited() -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.error(
            "rate limited",
            cause=ModelProviderError(
                "rate limited", provider="scripted", status_code=429, retryable=True
            ),
        )
    ]


def _rejected_request() -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.error(
            "bad request",
            cause=ModelProviderError(
                "bad request", provider="scripted", status_code=400, retryable=False
            ),
        )
    ]


class CountingProvider(ModelProvider):
    """Scripted provider with stable identity whose batches may end in errors."""

    name = "scripted"

    def __init__(self, *batches: list[ModelStreamEvent]) -> None:
        self._batches = list(batches)
        self.calls = 0
        self.requests: list[ModelRequest] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return _identity("counting-provider")

    def prepare_auxiliary_request(
        self, request: ModelRequest, *, max_output_tokens: int
    ) -> ModelRequest:
        return self._prepare_auxiliary_request(
            request,
            max_output_tokens=max_output_tokens,
            output_option_path=("scripted", "max_output_tokens"),
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        self.requests.append(request)
        for event in self._batches.pop(0):
            yield event


class BlockingProvider(ModelProvider):
    name = "blocking"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return _identity("blocking-provider")

    def prepare_auxiliary_request(
        self, request: ModelRequest, *, max_output_tokens: int
    ) -> ModelRequest:
        return self._prepare_auxiliary_request(
            request,
            max_output_tokens=max_output_tokens,
            output_option_path=("blocking", "max_output_tokens"),
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.started.set()
        await self.release.wait()
        yield ModelStreamEvent.text_delta(_answer())
        yield ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}})


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
    close = getattr(store, "close", None)
    if close is not None and not isinstance(store, InMemoryTaskStore):
        await close()


def _app(store: TaskStore, provider: ModelProvider, judge: ProviderCompletionVerifier) -> CayuApp:
    app = CayuApp(task_store=store, enable_logging=False)
    app.register_provider(provider)
    app.register_completion_verifier(_reference(), judge)
    return app


def test_provider_verifier_publishes_decision_with_separate_accounting(
    store_factory: StoreFactory,
) -> None:
    async def scenario() -> None:
        suffix = uuid4().hex[:12]
        store = store_factory()
        try:
            contract = _contract(suffix)
            proposal_id = await _proposal(store, contract, suffix)
            provider = CountingProvider(_completion(_answer()))
            judge = Judge()
            app = _app(store, provider, judge)
            decision = await app.verify_completion_proposal(_execution(proposal_id))

            assert decision.verdict is CompletionVerdict.ACCEPTED
            assert decision.verifier.kind is CompletionVerifierKind.PROVIDER
            outcome = decision.criterion_outcomes[0]
            assert outcome.satisfaction_basis is CompletionSatisfactionBasis.VERIFIER_ASSERTION
            assert provider.calls == 1
            assert judge.requests[0].proposal.proposal_id == proposal_id

            dispatches = await app.list_completion_verifier_dispatches(proposal_id)
            assert len(dispatches) == 1
            dispatch = dispatches[0]
            assert dispatch.ordinal == 1
            assert dispatch.request.provider_name == "scripted"
            assert dispatch.request.model == "judge-model"
            assert dispatch.request.verifier_profile_fingerprint == (
                decision.verifier_profile_fingerprint
            )
            settlement = dispatch.settlement
            assert settlement is not None
            assert settlement.outcome is CompletionVerifierDispatchOutcome.COMPLETED
            assert settlement.decode_status is CompletionVerifierDecodeStatus.DECODED
            assert settlement.usage is not None
            assert (settlement.usage.input_tokens, settlement.usage.output_tokens) == (120, 30)

            # Exact replay after restart needs neither the registration nor the provider.
            restarted = CayuApp(task_store=store, enable_logging=False)
            assert await restarted.verify_completion_proposal(_execution(proposal_id)) == decision
            assert provider.calls == 1
            assert len(await app.list_completion_verifier_dispatches(proposal_id)) == 1
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_rejected_verdict_derives_gaps_from_model_outcome() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("reject")
        proposal_id = await _proposal(store, contract, "reject")
        provider = CountingProvider(
            _completion(_answer("rejected", "unsatisfied", gap_code="package.incomplete"))
        )
        app = _app(store, provider, Judge())
        decision = await app.verify_completion_proposal(_execution(proposal_id))
        assert decision.verdict is CompletionVerdict.REJECTED
        assert decision.criterion_outcomes[0].status is CriterionOutcomeStatus.UNSATISFIED
        assert [(gap.criterion_id, gap.code) for gap in decision.gaps] == [
            ("ready", "package.incomplete")
        ]

    asyncio.run(scenario())


def test_cited_worker_evidence_is_bound_from_the_proposal_not_the_model() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("evidence", with_evidence=True)
        evidence = (
            WorkEvidenceReference(
                kind="test.report",
                reference_id="report-1",
                requirement_id="tests",
                digest=_digest("report-1"),
            ),
        )
        proposal_id = await _proposal(store, contract, "evidence", evidence=evidence)
        provider = CountingProvider(_completion(_answer(evidence=["E1"])))
        app = _app(store, provider, Judge())
        decision = await app.verify_completion_proposal(_execution(proposal_id))
        outcome = decision.criterion_outcomes[0]
        assert outcome.satisfaction_basis is CompletionSatisfactionBasis.EVIDENCE
        assert outcome.evidence_references == evidence

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "text",
    [
        "The package looks ready to me.",
        '{"verdict": "accepted", "criteria": [], "constraints": []}',
        _answer(evidence=["E1"]),
        _answer("accepted", "unsatisfied", gap_code="package.incomplete"),
        _answer("rejected", "satisfied"),
        _answer().replace('"constraints": []', '"constraints": [], "extra": 1'),
        '{"verdict": "accepted", "verdict": "rejected", "criteria": [], "constraints": []}',
    ],
)
def test_undecodable_responses_are_execution_failures_not_rejections(text: str) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("decode")
        proposal_id = await _proposal(store, contract, "decode")
        provider = CountingProvider(_completion(text))
        app = _app(store, provider, Judge())
        with pytest.raises(ProviderCompletionVerifierDecodingError) as raised:
            await app.verify_completion_proposal(_execution(proposal_id))
        assert text not in str(raised.value)
        assert await store.load_completion_decision_for_proposal(proposal_id) is None
        (dispatch,) = await app.list_completion_verifier_dispatches(proposal_id)
        assert dispatch.settlement is not None
        assert dispatch.settlement.outcome is CompletionVerifierDispatchOutcome.COMPLETED
        assert dispatch.settlement.decode_status is CompletionVerifierDecodeStatus.INVALID
        assert dispatch.settlement.decision is None
        assert provider.calls == 1

    asyncio.run(scenario())


def test_required_evidence_must_be_cited_for_a_satisfied_outcome() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("required", with_evidence=True)
        proposal_id = await _proposal(store, contract, "required")
        attempt = await store.load_work_attempt("judge-attempt-required")
        proposal = await store.load_completion_proposal(proposal_id)
        assert attempt is not None and proposal is not None
        request = CompletionVerifierRequest(contract=contract, attempt=attempt, proposal=proposal)
        with pytest.raises(ProviderCompletionVerifierDecodingError):
            decode_provider_completion_verifier_decision(_answer(), request)

    asyncio.run(scenario())


def test_retryable_provider_failure_retries_inside_one_execution(
    store_factory: StoreFactory,
) -> None:
    async def scenario() -> None:
        suffix = uuid4().hex[:12]
        store = store_factory()
        try:
            contract = _contract(suffix)
            proposal_id = await _proposal(store, contract, suffix)
            provider = CountingProvider(_rate_limited(), _completion(_answer()))
            app = _app(store, provider, Judge())
            decision = await app.verify_completion_proposal(_execution(proposal_id))
            assert decision.verdict is CompletionVerdict.ACCEPTED
            assert provider.calls == 2
            first, second = await app.list_completion_verifier_dispatches(proposal_id)
            assert (first.request.provider_attempt, second.request.provider_attempt) == (1, 2)
            assert first.request.claim_attempt_number == second.request.claim_attempt_number
            assert first.settlement is not None
            assert first.settlement.outcome is CompletionVerifierDispatchOutcome.FAILED
            assert first.settlement.failure == CompletionVerifierDispatchFailure(
                code="provider.rate_limited", status_code=429, retryable=True
            )
            summary = summarize_completion_verifier_dispatches((first, second))
            assert summary.attempts == 2
            assert summary.outcomes == {"completed": 1, "failed": 1}
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_terminal_provider_failure_is_typed_and_publishes_no_decision() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("terminal")
        proposal_id = await _proposal(store, contract, "terminal")
        provider = CountingProvider(_rejected_request())
        app = _app(store, provider, Judge())
        with pytest.raises(ProviderCompletionVerifierDispatchError):
            await app.verify_completion_proposal(_execution(proposal_id))
        assert provider.calls == 1
        assert await store.load_completion_decision_for_proposal(proposal_id) is None

    asyncio.run(scenario())


def test_verifier_budget_bounds_attempts_across_executions() -> None:
    async def scenario() -> None:
        clock_now = [datetime(2026, 10, 1, tzinfo=UTC)]
        store = InMemoryTaskStore(ownership_clock=lambda: clock_now[0])
        contract = _contract("budget")
        proposal_id = await _proposal(store, contract, "budget")
        provider = CountingProvider(_rate_limited(), _rate_limited(), _completion(_answer()))
        target = _target(budget=CompletionVerifierDispatchBudget(max_attempts=2))
        app = _app(store, provider, Judge(target))
        with pytest.raises(ProviderCompletionVerifierDispatchError):
            await app.verify_completion_proposal(_execution(proposal_id))
        assert provider.calls == 2

        # A later verifier execution under a fresh claim cannot exceed the budget.
        clock_now[0] += timedelta(minutes=5)
        with pytest.raises(ProviderCompletionVerifierBudgetExhausted):
            await app.verify_completion_proposal(
                _execution(proposal_id, claim="claim-2", decision="decision-2")
            )
        assert provider.calls == 2
        assert len(await app.list_completion_verifier_dispatches(proposal_id)) == 2

    asyncio.run(scenario())


def test_committed_provider_outcome_is_reused_instead_of_judging_again() -> None:
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
        contract = _contract("reuse")
        proposal_id = await _proposal(store, contract, "reuse")
        provider = CountingProvider(_completion(_answer()))
        app = _app(store, provider, Judge())
        with pytest.raises(ConnectionError):
            await app.verify_completion_proposal(_execution(proposal_id))
        decision = await app.verify_completion_proposal(_execution(proposal_id))
        assert decision.verdict is CompletionVerdict.ACCEPTED
        assert provider.calls == 1
        assert len(await app.list_completion_verifier_dispatches(proposal_id)) == 1

    asyncio.run(scenario())


def test_caller_cancellation_settles_the_attempt_and_publishes_nothing() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("cancel")
        proposal_id = await _proposal(store, contract, "cancel")
        provider = BlockingProvider()
        app = _app(store, provider, Judge(_target(provider_name="blocking")))
        running = asyncio.create_task(app.verify_completion_proposal(_execution(proposal_id)))
        await provider.started.wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert await app.drain_verified_completions(timeout_s=5.0)
        assert await store.load_completion_decision_for_proposal(proposal_id) is None
        (dispatch,) = await app.list_completion_verifier_dispatches(proposal_id)
        assert dispatch.settlement is not None
        assert dispatch.settlement.outcome is CompletionVerifierDispatchOutcome.CANCELLED
        assert dispatch.settlement.usage_status is CompletionVerifierUsageStatus.MISSING

    asyncio.run(scenario())


def test_attempt_timeout_is_a_typed_failure_with_durable_evidence() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("timeout")
        proposal_id = await _proposal(store, contract, "timeout")
        provider = BlockingProvider()
        target = _target(provider_name="blocking", attempt_timeout_seconds=0.05)
        app = _app(store, provider, Judge(target))
        with pytest.raises(ProviderCompletionVerifierDispatchError, match="timeout"):
            await app.verify_completion_proposal(_execution(proposal_id))
        (dispatch,) = await app.list_completion_verifier_dispatches(proposal_id)
        assert dispatch.settlement is not None
        assert dispatch.settlement.outcome is CompletionVerifierDispatchOutcome.TIMED_OUT
        assert await store.load_completion_decision_for_proposal(proposal_id) is None

    asyncio.run(scenario())


def test_registration_requires_matching_kind_and_a_registered_provider() -> None:
    class Deterministic(DeterministicCompletionVerifier):
        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return _identity("deterministic")

        async def verify(self, request):  # pragma: no cover - never dispatched
            raise AssertionError

    app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
    with pytest.raises(CompletionVerifierUnavailable, match="not registered"):
        app.register_completion_verifier(_reference(), Judge())
    app.register_provider(CountingProvider(_completion(_answer())))
    with pytest.raises(ValueError, match="kind does not match"):
        app.register_completion_verifier(_reference(), Deterministic())
    assert app.register_completion_verifier(_reference(), Judge()) == _reference()


def test_profile_binds_runtime_provider_components_and_detects_target_drift() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("drift")
        proposal_id = await _proposal(store, contract, "drift")
        judge = Judge()
        app = _app(store, CountingProvider(_completion(_answer())), judge)
        judge._target = _target(max_output_tokens=400)
        with pytest.raises(CompletionVerifierUnavailable, match="target changed"):
            await app.verify_completion_proposal(_execution(proposal_id))
        assert await store.load_completion_verification_claim(proposal_id) is None

        judge._target = _target()
        await app.verify_completion_proposal(_execution(proposal_id))
        profile = await store.load_completion_verifier_profile(proposal_id)
        assert profile is not None
        component_ids = {item.component_id for item in profile.profile.components}
        assert {"adapter", "runtime.provider-adapter", "runtime.provider-target"} <= component_ids

    asyncio.run(scenario())


def test_messages_must_be_text_only() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("tools")
        proposal_id = await _proposal(store, contract, "tools")
        judge = Judge()
        judge.messages = [
            Message(
                role=MessageRole.TOOL,
                content=(ToolResultPart(tool_call_id="call-1", tool_name="lookup", content="ok"),),
            )
        ]
        provider = CountingProvider(_completion(_answer()))
        app = _app(store, provider, judge)
        with pytest.raises(CompletionVerifierExecutionError, match="tool turns"):
            await app.verify_completion_proposal(_execution(proposal_id))
        assert provider.calls == 0
        assert await app.list_completion_verifier_dispatches(proposal_id) == ()

    asyncio.run(scenario())


def test_store_fences_dispatch_intents_and_settles_once(store_factory: StoreFactory) -> None:
    async def scenario() -> None:
        suffix = uuid4().hex[:12]
        store = store_factory()
        try:
            contract = _contract(suffix)
            proposal_id = await _proposal(store, contract, suffix)
            # Prepare the profile and a live claim through the public app path.

            class Capture(Judge):
                async def build_messages(self, request):
                    raise RuntimeError("stop before dispatch")

            app = CayuApp(task_store=store, enable_logging=False)
            app.register_provider(CountingProvider(_completion(_answer())))
            app.register_completion_verifier(_reference(), Capture())
            with pytest.raises(Exception, match="stop before dispatch"):
                await app.verify_completion_proposal(_execution(proposal_id))
            claim = await store.load_completion_verification_claim(proposal_id)
            assert claim is not None and claim.execution_owner_id is not None
            owner = claim.execution_owner_id

            def intent(provider_attempt: int = 1, **overrides: Any):
                values: dict[str, Any] = {
                    "dispatch_id": completion_verifier_dispatch_id(
                        proposal_id=proposal_id,
                        claim_id=claim.claim_id,
                        claim_attempt_number=claim.attempt_number,
                        provider_attempt=provider_attempt,
                    ),
                    "proposal_id": proposal_id,
                    "claim_id": claim.claim_id,
                    "worker_id": claim.worker_id,
                    "execution_owner_id": owner,
                    "claim_attempt_number": claim.attempt_number,
                    "verifier": claim.verifier,
                    "verifier_profile_fingerprint": claim.verifier_profile_fingerprint,
                    "provider_attempt": provider_attempt,
                    "provider_name": "scripted",
                    "pricing_provider_name": "scripted",
                    "model": "judge-model",
                    "request_sha256": _digest("request"),
                    "max_input_tokens": 100,
                    "max_output_tokens": 10,
                    "timeout_seconds": 5.0,
                    "budget": CompletionVerifierDispatchBudget(max_attempts=3),
                }
                values.update(overrides)
                return CompletionVerifierDispatchRequest(**values)

            with pytest.raises(CompletionVerificationClaimLost):
                await store.record_completion_verifier_dispatch(intent(worker_id="another-worker"))
            with pytest.raises(WorkCompletionConflict):
                await store.record_completion_verifier_dispatch(intent(provider_attempt=2))

            first = await store.record_completion_verifier_dispatch(intent())
            assert first.ordinal == 1
            assert await store.record_completion_verifier_dispatch(intent()) == first
            with pytest.raises(WorkCompletionConflict):
                await store.record_completion_verifier_dispatch(
                    intent(request_sha256=_digest("other"))
                )
            with pytest.raises(WorkCompletionConflict):
                await store.record_completion_verifier_dispatch(intent(provider_attempt=2))

            settlement = CompletionVerifierDispatchSettlementRequest(
                dispatch_id=first.dispatch_id,
                outcome=CompletionVerifierDispatchOutcome.FAILED,
                usage_status=CompletionVerifierUsageStatus.OBSERVED,
                usage=UsageMetrics(
                    provider_name="scripted",
                    requested_model="judge-model",
                    model="judge-model",
                    input_tokens=40,
                    output_tokens=2,
                    total_tokens=42,
                ),
                latency_ms=12,
                failure=CompletionVerifierDispatchFailure(code="provider.error"),
            )
            settled = await store.settle_completion_verifier_dispatch(settlement)
            assert settled.settlement is not None
            assert await store.settle_completion_verifier_dispatch(settlement) == settled
            with pytest.raises(WorkCompletionConflict):
                await store.settle_completion_verifier_dispatch(
                    settlement.model_copy(update={"latency_ms": 13})
                )

            second = await store.record_completion_verifier_dispatch(
                intent(provider_attempt=2, budget=CompletionVerifierDispatchBudget(max_attempts=3))
            )
            assert second.ordinal == 2
            with pytest.raises(WorkCompletionConflict, match="share one profile and budget"):
                await store.record_completion_verifier_dispatch(
                    intent(
                        provider_attempt=3, budget=CompletionVerifierDispatchBudget(max_attempts=4)
                    )
                )
            assert [
                item.dispatch_id
                for item in await store.list_completion_verifier_dispatches(proposal_id)
            ] == [first.dispatch_id, second.dispatch_id]
            del app
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_token_budget_counts_unknown_usage_at_its_envelope() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("tokens")
        proposal_id = await _proposal(store, contract, "tokens")

        class Stop(Judge):
            async def build_messages(self, request):
                raise RuntimeError("stop")

        app = CayuApp(task_store=store, enable_logging=False)
        app.register_provider(CountingProvider(_completion(_answer())))
        app.register_completion_verifier(_reference(), Stop())
        with pytest.raises(Exception, match="stop"):
            await app.verify_completion_proposal(_execution(proposal_id))
        claim = await store.load_completion_verification_claim(proposal_id)
        assert claim is not None and claim.execution_owner_id is not None
        owner = claim.execution_owner_id
        budget = CompletionVerifierDispatchBudget(max_attempts=5, max_input_tokens=150)

        def intent(provider_attempt: int) -> CompletionVerifierDispatchRequest:
            return CompletionVerifierDispatchRequest(
                dispatch_id=completion_verifier_dispatch_id(
                    proposal_id=proposal_id,
                    claim_id=claim.claim_id,
                    claim_attempt_number=claim.attempt_number,
                    provider_attempt=provider_attempt,
                ),
                proposal_id=proposal_id,
                claim_id=claim.claim_id,
                worker_id=claim.worker_id,
                execution_owner_id=owner,
                claim_attempt_number=claim.attempt_number,
                verifier=claim.verifier,
                verifier_profile_fingerprint=claim.verifier_profile_fingerprint,
                provider_attempt=provider_attempt,
                provider_name="scripted",
                pricing_provider_name="scripted",
                model="judge-model",
                request_sha256=_digest("request"),
                max_input_tokens=100,
                max_output_tokens=10,
                timeout_seconds=5.0,
                budget=budget,
            )

        first = await store.record_completion_verifier_dispatch(intent(1))
        await store.settle_completion_verifier_dispatch(
            CompletionVerifierDispatchSettlementRequest(
                dispatch_id=first.dispatch_id,
                outcome=CompletionVerifierDispatchOutcome.OUTCOME_UNKNOWN,
                usage_status=CompletionVerifierUsageStatus.MISSING,
                latency_ms=1,
                failure=CompletionVerifierDispatchFailure(code="provider.stream_failed"),
            )
        )
        # 100 unknown tokens count at the envelope, so another 100 would exceed 150.
        with pytest.raises(CompletionVerifierDispatchBudgetExhausted):
            await store.record_completion_verifier_dispatch(intent(2))

    asyncio.run(scenario())


def test_usage_summary_prices_verifier_attempts_separately() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("summary")
        proposal_id = await _proposal(store, contract, "summary")
        provider = CountingProvider(
            _completion(_answer(), input_tokens=1_000_000, output_tokens=1_000_000)
        )
        target = _target(max_input_tokens=2_000_000, max_output_tokens=2_000_000)
        app = _app(store, provider, Judge(target))
        await app.verify_completion_proposal(_execution(proposal_id))
        dispatches = await app.list_completion_verifier_dispatches(proposal_id)
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="scripted",
                    model="judge-model",
                    input_per_million=Decimal("2"),
                    output_per_million=Decimal("8"),
                ),
            )
        )
        summary = summarize_completion_verifier_dispatches(dispatches, pricing=pricing)
        assert (summary.input_tokens, summary.output_tokens) == (1_000_000, 1_000_000)
        assert summary.cost == Decimal("10")
        assert summary.currency == "USD"
        assert summary.unpriced_attempts == 0
        assert summarize_completion_verifier_dispatches(dispatches).cost is None

    asyncio.run(scenario())


def test_text_parts_are_preserved_when_contract_is_appended() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("compose")
        proposal_id = await _proposal(store, contract, "compose")
        provider = CountingProvider(_completion(_answer()))
        app = _app(store, provider, Judge())
        await app.verify_completion_proposal(_execution(proposal_id))
        (request,) = provider.requests
        assert request.tools == []
        assert request.messages[-1].role is MessageRole.USER
        parts = request.messages[-1].content
        assert isinstance(parts[0], TextPart) and parts[0].text == "Judge the package."
        assert isinstance(parts[-1], TextPart) and '"criteria"' in parts[-1].text

    asyncio.run(scenario())


def test_claim_request_type_is_unchanged_for_provider_verifiers() -> None:
    # Provider claims use the ordinary claim contract; only the kind differs.
    request = CompletionVerificationClaimRequest(
        claim_id="claim",
        proposal_id="proposal",
        worker_id="worker",
        verifier=_reference(),
        verifier_profile_fingerprint=_digest("profile"),
    )
    assert request.verifier.kind is CompletionVerifierKind.PROVIDER


def test_retry_under_the_same_claim_continues_from_the_next_provider_attempt() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract("same-claim")
        proposal_id = await _proposal(store, contract, "same-claim")
        provider = CountingProvider(_completion("not json"), _completion(_answer()))
        app = _app(store, provider, Judge())
        with pytest.raises(ProviderCompletionVerifierDecodingError):
            await app.verify_completion_proposal(_execution(proposal_id))
        decision = await app.verify_completion_proposal(_execution(proposal_id))
        assert decision.verdict is CompletionVerdict.ACCEPTED
        first, second = await app.list_completion_verifier_dispatches(proposal_id)
        assert (first.request.provider_attempt, second.request.provider_attempt) == (1, 2)
        assert first.request.claim_id == second.request.claim_id

        # The execution's provider retries are spent; a third call cannot dispatch.
        assert provider.calls == 2

    asyncio.run(scenario())
