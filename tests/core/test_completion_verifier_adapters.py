from __future__ import annotations

import asyncio
import multiprocessing
import traceback as traceback_module
import warnings
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest
from tests.core.completion_verifier_profile_fixtures import (
    prepare_test_completion_verifier_profile,
    retask_test_completion_verifier_profile,
)
from tests.core.task_invocation_fixtures import unattributed_session_invocation_binding
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import (
    CayuApp,
    CompletionContinuationPolicy,
    CompletionCriterionOutcome,
    CompletionDecision,
    CompletionDecisionApplicationRequest,
    CompletionDecisionCreate,
    CompletionGap,
    CompletionProposalCreate,
    CompletionRejectionAction,
    CompletionResultReference,
    CompletionResultResolverRef,
    CompletionSatisfactionBasis,
    CompletionVerdict,
    CompletionVerificationClaimLost,
    CompletionVerificationClaimRequest,
    CompletionVerifierDecision,
    CompletionVerifierExecutionError,
    CompletionVerifierExecutionRequest,
    CompletionVerifierKind,
    CompletionVerifierProfileAdoptionDecision,
    CompletionVerifierProfileComponentDeclaration,
    CompletionVerifierProfilePolicy,
    CompletionVerifierProfilePolicyRequest,
    CompletionVerifierRef,
    CompletionVerifierRequest,
    CompletionVerifierUnavailable,
    CriterionOutcomeStatus,
    DeterministicCompletionVerifier,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileBehaviorIdentity,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyResult,
    InMemoryTaskStore,
    ResolutionActor,
    ResolutionActorSource,
    SecretRedactor,
    SQLiteTaskStore,
    TaskCreate,
    WorkAttemptCreate,
    WorkCompletionConflict,
    WorkContract,
    WorkContractDraft,
    WorkContractRef,
    WorkCriterion,
    work_contract_from_draft,
)
from cayu.runtime._diagnostics import (
    MAX_DIAGNOSTIC_EXCEPTION_GROUP_NODES,
    MAX_DIAGNOSTIC_UTF8_BYTES,
)
from cayu.runtime.completion_verifier_profiles import (
    COMPLETION_VERIFIER_PROFILE_COMPONENT_MAX_ITEMS,
    COMPLETION_VERIFIER_PROFILE_TEXT_MAX_CHARS,
    build_completion_verifier_execution_profile,
    changed_completion_verifier_profile_components,
)
from cayu.runtime.execution_profiles import EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS
from cayu.runtime.work_contracts import (
    WORK_COMPLETION_OUTCOME_MAX_EVIDENCE_REFERENCES,
    WORK_COMPLETION_VERIFIER_DECISION_MAX_BYTES,
    WORK_CONTRACT_MAX_CRITERIA,
    WORK_CONTRACT_MAX_EVIDENCE_REFERENCES,
    completion_decision_request_sha256,
    completion_gap_fingerprint,
)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


class _TestCompletionVerifier(DeterministicCompletionVerifier):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name=type(self).__name__,
            behavior_version="v1",
            implementation_version="test-v1",
        )


def _verifier_reference(
    *,
    fingerprint: str = "reference-configuration",
    kind: CompletionVerifierKind = CompletionVerifierKind.DETERMINISTIC,
) -> CompletionVerifierRef:
    return CompletionVerifierRef(
        verifier_id="bid-readiness",
        version="v1",
        kind=kind,
        configuration_fingerprint=_digest(fingerprint),
    )


def _resolver_reference() -> CompletionResultResolverRef:
    return CompletionResultResolverRef(
        resolver_id="bid-result",
        version="v1",
        configuration_fingerprint=_digest("bid-result-v1"),
    )


def _contract(
    *,
    verifier: CompletionVerifierRef | None = None,
    objective: str = "Publish a ready bid package.",
    continuation_policy: CompletionContinuationPolicy | None = None,
) -> WorkContract:
    return work_contract_from_draft(
        WorkContractDraft(
            contract_id="bid-contract",
            version=1,
            objective=objective,
            criteria=(
                WorkCriterion(
                    criterion_id="ready",
                    ordinal=1,
                    description="The package is ready.",
                ),
            ),
            verifier=verifier or _verifier_reference(),
            result_resolver=_resolver_reference(),
            continuation_policy=continuation_policy or CompletionContinuationPolicy(),
        )
    )


def _accepted_decision() -> CompletionVerifierDecision:
    return CompletionVerifierDecision(
        verdict=CompletionVerdict.ACCEPTED,
        criterion_outcomes=(
            CompletionCriterionOutcome(
                criterion_id="ready",
                status=CriterionOutcomeStatus.SATISFIED,
                reason_code="package.ready",
                satisfaction_basis=CompletionSatisfactionBasis.VERIFIER_ASSERTION,
            ),
        ),
    )


def _rejected_decision() -> CompletionVerifierDecision:
    return CompletionVerifierDecision(
        verdict=CompletionVerdict.REJECTED,
        criterion_outcomes=(
            CompletionCriterionOutcome(
                criterion_id="ready",
                status=CriterionOutcomeStatus.UNSATISFIED,
                reason_code="package.missing",
            ),
        ),
        gaps=(
            CompletionGap(
                criterion_id="ready",
                code="package.missing",
            ),
        ),
    )


class RecordingVerifier(_TestCompletionVerifier):
    def __init__(self, decision: CompletionVerifierDecision) -> None:
        self.decision = decision
        self.requests: list[CompletionVerifierRequest] = []

    async def verify(self, request: CompletionVerifierRequest) -> CompletionVerifierDecision:
        self.requests.append(request)
        return self.decision


def _assert_secret_absent_from_cayu_error(error: BaseException, secret: str) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        assert secret not in str(current)
        assert secret not in repr(current)
        assert secret not in "".join(traceback_module.format_exception(current))
        current_traceback = current.__traceback__
        while current_traceback is not None:
            if is_cayu_source_filename(current_traceback.tb_frame.f_code.co_filename):
                assert all(
                    secret not in repr(value)
                    for value in current_traceback.tb_frame.f_locals.values()
                )
            current_traceback = current_traceback.tb_next
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


def _exception_leaf_types(error: BaseException) -> tuple[type[BaseException], ...]:
    leaves: list[type[BaseException]] = []
    pending = [error]
    while pending:
        current = pending.pop()
        if isinstance(current, BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
        else:
            leaves.append(type(current))
    return tuple(leaves)


async def _proposal(
    store: InMemoryTaskStore,
    contract: WorkContract,
    *,
    suffix: str = "1",
) -> str:
    await store.publish_work_contract(contract)
    task_id = f"task-{suffix}"
    session_id = f"session-{suffix}"
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
            attempt_id=f"attempt-{suffix}",
            task_id=task_id,
            session_id=session_id,
            contract=contract.reference(),
            execution_profile_fingerprint=_digest("worker-profile"),
        )
    )
    proposal = await store.submit_completion_proposal(
        CompletionProposalCreate(
            proposal_id=f"proposal-{suffix}",
            attempt_id=attempt.attempt_id,
            result=CompletionResultReference(
                kind="task.result",
                reference_id=f"result-{suffix}",
                digest=_digest(f"result-{suffix}"),
            ),
        )
    )
    return proposal.proposal_id


def _execution_request(
    proposal_id: str,
    *,
    suffix: str = "1",
    lease_seconds: int = 30,
    timeout_seconds: float = 5.0,
    profile_adoption: ExecutionProfileAdoptionIntent | None = None,
) -> CompletionVerifierExecutionRequest:
    return CompletionVerifierExecutionRequest(
        proposal_id=proposal_id,
        claim_id=f"claim-{suffix}",
        decision_id=f"decision-{suffix}",
        worker_id="verifier-worker",
        lease_seconds=lease_seconds,
        execution_timeout_seconds=timeout_seconds,
        profile_adoption=profile_adoption,
    )


def test_public_adapter_persists_decision_and_replays_after_restart() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(task_store=store, enable_logging=False)
        task_before = await store.load_task("task-1")
        assert task_before is not None

        assert app.register_completion_verifier(contract.verifier, verifier) == contract.verifier
        request = _execution_request(proposal_id)
        decision = await app.verify_completion_proposal(request)

        assert decision.verdict is CompletionVerdict.ACCEPTED
        assert decision.proposal_id == proposal_id
        assert decision.claim_id == request.claim_id
        assert decision.worker_id == request.worker_id
        assert decision.verifier == contract.verifier
        assert len(verifier.requests) == 1
        context = verifier.requests[0]
        assert context.contract == contract
        assert context.proposal.proposal_id == proposal_id
        assert context.attempt.attempt_id == "attempt-1"
        assert await store.load_task("task-1") == task_before

        # Durable reconciliation does not require the process-local registration.
        restarted = CayuApp(task_store=store, enable_logging=False)
        assert await restarted.verify_completion_proposal(request) == decision
        assert len(verifier.requests) == 1

    asyncio.run(scenario())


def test_profile_is_durable_before_adapter_dispatch_and_bound_to_public_evidence() -> None:
    class ProfileObservingVerifier(_TestCompletionVerifier):
        def __init__(self, store: InMemoryTaskStore) -> None:
            self.store = store
            self.profile_seen = None

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            self.profile_seen = await self.store.load_completion_verifier_profile(
                request.proposal.proposal_id
            )
            assert self.profile_seen is not None
            return _accepted_decision()

    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = ProfileObservingVerifier(store)
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)

        decision = await app.verify_completion_proposal(_execution_request(proposal_id))
        profile = verifier.profile_seen
        assert profile is not None
        claim = await store.load_completion_verification_claim(proposal_id)
        assert claim is not None
        assert claim.verifier_profile_fingerprint == profile.profile.fingerprint
        assert decision.verifier_profile_fingerprint == profile.profile.fingerprint
        assert profile.profile.verifier == contract.verifier
        assert profile.source_execution_profile_fingerprint == _digest("worker-profile")
        durable_profile_json = profile.model_dump_json()
        assert contract.objective not in durable_profile_json
        assert contract.criteria[0].description not in durable_profile_json
        assert "task.result" not in durable_profile_json

    asyncio.run(scenario())


def test_registered_verifier_profile_is_snapshotted_and_live_drift_fails_before_claim() -> None:
    class MutableIdentityVerifier(RecordingVerifier):
        def __init__(self) -> None:
            super().__init__(_accepted_decision())
            self.behavior_version = "v1"

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:mutable-completion-verifier",
                behavior_version=self.behavior_version,
                implementation_version="1",
            )

    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = MutableIdentityVerifier()
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        verifier.behavior_version = "v2"

        # Repeated pre-dispatch profile failures must not consume the bounded
        # adapter-dispatch capacity. The 65th retry crosses the runtime's
        # capacity bound and would expose a leaked reservation.
        for _ in range(65):
            with pytest.raises(CompletionVerifierUnavailable, match="identity changed"):
                await app.verify_completion_proposal(_execution_request(proposal_id))
        assert verifier.requests == []
        assert await store.load_completion_verifier_profile(proposal_id) is None
        assert await store.load_completion_verification_claim(proposal_id) is None

    asyncio.run(scenario())


def test_live_verifier_profile_drift_during_preparation_fails_before_claim() -> None:
    class BlockingProfileStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.preparation_started = asyncio.Event()
            self.release_preparation = asyncio.Event()

        async def prepare_completion_verifier_profile(self, request):
            self.preparation_started.set()
            await self.release_preparation.wait()
            return await super().prepare_completion_verifier_profile(request)

    class MutableIdentityVerifier(RecordingVerifier):
        def __init__(self) -> None:
            super().__init__(_accepted_decision())
            self.behavior_version = "v1"

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:preparation-race-completion-verifier",
                behavior_version=self.behavior_version,
                implementation_version="1",
            )

    async def scenario() -> None:
        store = BlockingProfileStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = MutableIdentityVerifier()
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)

        execution = asyncio.create_task(
            app.verify_completion_proposal(_execution_request(proposal_id))
        )
        await store.preparation_started.wait()
        verifier.behavior_version = "v2"
        store.release_preparation.set()

        with pytest.raises(CompletionVerifierUnavailable, match="identity changed"):
            await execution
        assert verifier.requests == []
        assert await store.load_completion_verification_claim(proposal_id) is None

    asyncio.run(scenario())


def test_duplicate_verifier_component_identity_is_rejected_at_registration() -> None:
    component = CompletionVerifierProfileComponentDeclaration(
        component_id="rules",
        identity=ExecutionProfileBehaviorIdentity(
            name="tests:verifier-rules",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    class DuplicateComponentVerifier(RecordingVerifier):
        @property
        def execution_profile_components(
            self,
        ) -> tuple[CompletionVerifierProfileComponentDeclaration, ...]:
            return (component, component)

    app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
    verifier = DuplicateComponentVerifier(_accepted_decision())
    assert verifier.execution_profile_components == (component, component)
    with pytest.raises(ValueError, match="component identities are invalid"):
        app.register_completion_verifier(
            _verifier_reference(),
            verifier,
        )


def test_adoption_evidence_accepts_the_complete_bounded_component_union() -> None:
    adapter_identity = ExecutionProfileBehaviorIdentity(
        name="tests:maximum-component-union",
        behavior_version="1",
        implementation_version="1",
    )

    def declarations(prefix: str) -> tuple[CompletionVerifierProfileComponentDeclaration, ...]:
        return tuple(
            CompletionVerifierProfileComponentDeclaration(
                component_id=f"{prefix}-{index:02d}",
                identity=ExecutionProfileBehaviorIdentity(
                    name=f"tests:{prefix}:{index:02d}",
                    behavior_version="1",
                    implementation_version="1",
                ),
            )
            for index in range(COMPLETION_VERIFIER_PROFILE_COMPONENT_MAX_ITEMS)
        )

    expected = build_completion_verifier_execution_profile(
        verifier=_verifier_reference(),
        adapter_identity=adapter_identity,
        component_declarations=declarations("expected"),
    )
    candidate = build_completion_verifier_execution_profile(
        verifier=_verifier_reference(),
        adapter_identity=adapter_identity,
        component_declarations=declarations("candidate"),
    )
    changed = changed_completion_verifier_profile_components(expected, candidate)

    assert len(changed) == COMPLETION_VERIFIER_PROFILE_COMPONENT_MAX_ITEMS * 2
    adoption = CompletionVerifierProfileAdoptionDecision(
        expected_profile_fingerprint=expected.fingerprint,
        candidate_profile_fingerprint=candidate.fingerprint,
        changed_component_ids=changed,
        policy_identity="tests:maximum-component-union-policy:v1",
        authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
        idempotency_key="maximum-component-union-adoption",
        requested_by=ResolutionActor(
            subject="operator-1",
            source=ResolutionActorSource.REQUEST,
        ),
        reason="Adopt the expanded verifier component profile.",
        policy_reason="The expanded verifier component profile is authorized.",
        request_sha256=_digest("maximum-component-union-adoption"),
    )
    assert adoption.changed_component_ids == changed

    with pytest.raises(ValueError, match="must not exceed"):
        CompletionVerifierProfileAdoptionDecision(
            expected_profile_fingerprint=adoption.expected_profile_fingerprint,
            candidate_profile_fingerprint=adoption.candidate_profile_fingerprint,
            changed_component_ids=("x" * (COMPLETION_VERIFIER_PROFILE_TEXT_MAX_CHARS + 1),),
            policy_identity=adoption.policy_identity,
            authority_decision=adoption.authority_decision,
            idempotency_key=adoption.idempotency_key,
            requested_by=adoption.requested_by,
            reason=adoption.reason,
            policy_reason=adoption.policy_reason,
            request_sha256=adoption.request_sha256,
        )


def test_verifier_registration_requires_nonsecret_stable_component_authority(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class UnidentifiedVerifier(DeterministicCompletionVerifier):
        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            return _accepted_decision()

    app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
    with pytest.raises(ValueError, match="stable execution-profile identity"):
        app.register_completion_verifier(_verifier_reference(), UnidentifiedVerifier())

    secret = "private-verifier-component"
    component = CompletionVerifierProfileComponentDeclaration(
        component_id=secret,
        identity=ExecutionProfileBehaviorIdentity(
            name="tests:verifier-component",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    class SecretComponentVerifier(RecordingVerifier):
        @property
        def execution_profile_components(
            self,
        ) -> tuple[CompletionVerifierProfileComponentDeclaration, ...]:
            return (component,)

    secret_app = CayuApp(
        task_store=InMemoryTaskStore(),
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="component identities are invalid") as exc:
            secret_app.register_completion_verifier(
                _verifier_reference(),
                SecretComponentVerifier(_accepted_decision()),
            )
    _assert_secret_absent_from_cayu_error(exc.value, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_changed_profile_requires_adoption_and_exact_adoption_replay_is_durable(
    store_kind: str,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy_reason_canary = "~"
    policy_reason = policy_reason_canary * 300
    adoption_secret = "PRIVATE_VERIFIER_ADOPTION_SECRET_CANARY"

    class VersionedVerifier(RecordingVerifier):
        def __init__(
            self,
            decision: CompletionVerifierDecision,
            *,
            behavior_version: str,
        ) -> None:
            super().__init__(decision)
            self.behavior_version = behavior_version

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:versioned-completion-verifier",
                behavior_version=self.behavior_version,
                implementation_version="1",
            )

    class AdoptionPolicy(CompletionVerifierProfilePolicy):
        def __init__(self) -> None:
            self.requests: list[CompletionVerifierProfilePolicyRequest] = []

        @property
        def identity(self) -> str:
            return "tests:completion-verifier-adoption:v1"

        async def decide(
            self,
            request: CompletionVerifierProfilePolicyRequest,
        ) -> ExecutionProfilePolicyResult:
            self.requests.append(request)
            return ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason=policy_reason,
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            )

    async def scenario() -> None:
        sqlite_path = tmp_path / "verifier-profile-adoption.sqlite"
        store = InMemoryTaskStore() if store_kind == "memory" else SQLiteTaskStore(sqlite_path)
        contract = _contract(
            continuation_policy=CompletionContinuationPolicy(
                rejection_action=CompletionRejectionAction.CONTINUE,
            )
        )
        first_proposal_id = await _proposal(store, contract)
        first = VersionedVerifier(_rejected_decision(), behavior_version="v1")
        first_app = CayuApp(task_store=store, enable_logging=False)
        first_app.register_completion_verifier(contract.verifier, first)
        first_decision = await first_app.verify_completion_proposal(
            _execution_request(first_proposal_id)
        )
        await store.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id="task-1",
                decision_id=first_decision.decision_id,
                idempotency_key="apply-first-rejection",
            )
        )

        second_attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id="attempt-2",
                task_id="task-1",
                session_id="session-1",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("worker-profile-2"),
            )
        )
        second_proposal = await store.submit_completion_proposal(
            CompletionProposalCreate(
                proposal_id="proposal-2",
                attempt_id=second_attempt.attempt_id,
                result=CompletionResultReference(
                    kind="task.result",
                    reference_id="result-2",
                    digest=_digest("result-2"),
                ),
            )
        )
        changed = VersionedVerifier(_rejected_decision(), behavior_version="v2")
        without_policy = CayuApp(task_store=store, enable_logging=False)
        without_policy.register_completion_verifier(contract.verifier, changed)
        request = _execution_request(second_proposal.proposal_id, suffix="2")
        with pytest.raises(WorkCompletionConflict, match="explicit authorized adoption"):
            await without_policy.verify_completion_proposal(request)
        assert changed.requests == []
        assert await store.load_completion_verifier_profile("proposal-2") is None

        policy = AdoptionPolicy()
        adopting_app = CayuApp(
            task_store=store,
            completion_verifier_profile_policy=policy,
            secret_redactor=SecretRedactor([adoption_secret, policy_reason_canary]),
            enable_logging=False,
        )
        adopting_app.register_completion_verifier(contract.verifier, changed)
        secret_intent = ExecutionProfileAdoptionIntent(
            idempotency_key="reject-secret-verifier-adoption",
            reason=f"Use verifier v2 with {adoption_secret} authority.",
            requested_by=ResolutionActor(
                subject="operator-1",
                source=ResolutionActorSource.REQUEST,
            ),
        )
        with (
            warnings.catch_warnings(record=True) as secret_warnings,
            pytest.raises(WorkCompletionConflict, match="workload secret") as raised,
        ):
            warnings.simplefilter("always")
            await adopting_app.verify_completion_proposal(
                _execution_request(
                    second_proposal.proposal_id,
                    suffix="2",
                    profile_adoption=secret_intent,
                )
            )
        _assert_secret_absent_from_cayu_error(raised.value, adoption_secret)
        secret_output = capsys.readouterr()
        assert adoption_secret not in caplog.text
        assert adoption_secret not in secret_output.out
        assert adoption_secret not in secret_output.err
        assert all(adoption_secret not in str(item.message) for item in secret_warnings)
        assert policy.requests == []
        assert changed.requests == []
        assert await store.load_completion_verifier_profile("proposal-2") is None
        assert await store.load_completion_verification_claim("proposal-2") is None

        intent = ExecutionProfileAdoptionIntent(
            idempotency_key="adopt-verifier-v2",
            reason="Use verifier v2 for the next attempt.",
            requested_by=ResolutionActor(
                subject="operator-1",
                source=ResolutionActorSource.REQUEST,
            ),
        )
        adopted_request = _execution_request(
            second_proposal.proposal_id,
            suffix="2",
            profile_adoption=intent,
        )
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            decision = await adopting_app.verify_completion_proposal(adopted_request)
        assert len(policy.requests) == 1
        profile = await store.load_completion_verifier_profile("proposal-2")
        assert profile is not None
        assert profile.adoption is not None
        assert profile.adoption.changed_component_ids == ("adapter",)
        assert profile.adoption.reason == intent.reason
        assert "[REDACTED_SECRET]" in profile.adoption.policy_reason
        assert (
            len(profile.adoption.policy_reason.encode("utf-8"))
            <= EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS
        )
        assert policy_reason_canary not in profile.model_dump_json()
        assert policy_reason_canary not in decision.model_dump_json()
        captured = capsys.readouterr()
        assert policy_reason_canary not in caplog.text
        assert policy_reason_canary not in captured.out
        assert policy_reason_canary not in captured.err
        assert all(policy_reason_canary not in str(item.message) for item in caught_warnings)

        # For SQLite, cross the actual durable reconstruction boundary instead
        # of merely creating another app over the same live store object.
        replay_store = store
        if isinstance(store, SQLiteTaskStore):
            await store.close()
            replay_store = SQLiteTaskStore(sqlite_path)
        try:
            # Exact decision replay is authorized by the durable adoption record
            # and neither needs nor re-runs the application policy.
            restarted = CayuApp(task_store=replay_store, enable_logging=False)
            assert await restarted.verify_completion_proposal(adopted_request) == decision
            assert len(policy.requests) == 1
            reconstructed = await replay_store.load_completion_verifier_profile("proposal-2")
            assert reconstructed is not None
            assert reconstructed.adoption == profile.adoption
            conflicting = _execution_request(
                second_proposal.proposal_id,
                suffix="2",
                profile_adoption=ExecutionProfileAdoptionIntent(
                    idempotency_key=intent.idempotency_key,
                    reason="A different adoption request.",
                    requested_by=intent.requested_by,
                ),
            )
            with pytest.raises(WorkCompletionConflict, match="adoption retry conflicts"):
                await restarted.verify_completion_proposal(conflicting)

            await replay_store.apply_completion_decision(
                CompletionDecisionApplicationRequest(
                    task_id="task-1",
                    decision_id=decision.decision_id,
                    idempotency_key="apply-second-rejection",
                )
            )
            third_attempt = await replay_store.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id="attempt-3",
                    task_id="task-1",
                    session_id="session-1",
                    contract=contract.reference(),
                    execution_profile_fingerprint=_digest("worker-profile-3"),
                )
            )
            third_proposal = await replay_store.submit_completion_proposal(
                CompletionProposalCreate(
                    proposal_id="proposal-3",
                    attempt_id=third_attempt.attempt_id,
                    result=CompletionResultReference(
                        kind="task.result",
                        reference_id="result-3",
                        digest=_digest("result-3"),
                    ),
                )
            )
            third_policy = AdoptionPolicy()
            third_verifier = VersionedVerifier(_accepted_decision(), behavior_version="v3")
            third_app = CayuApp(
                task_store=replay_store,
                completion_verifier_profile_policy=third_policy,
                secret_redactor=SecretRedactor([adoption_secret, policy_reason_canary]),
                enable_logging=False,
            )
            third_app.register_completion_verifier(contract.verifier, third_verifier)
            reused_identity = ExecutionProfileAdoptionIntent(
                idempotency_key=intent.idempotency_key,
                reason="Use verifier v3 for the next attempt.",
                requested_by=intent.requested_by,
            )
            with pytest.raises(WorkCompletionConflict, match="idempotency key"):
                await third_app.verify_completion_proposal(
                    _execution_request(
                        third_proposal.proposal_id,
                        suffix="3",
                        profile_adoption=reused_identity,
                    )
                )
            assert len(third_policy.requests) == 1
            assert third_verifier.requests == []
            assert await replay_store.load_completion_verifier_profile("proposal-3") is None
        finally:
            if replay_store is not store:
                await replay_store.close()

    asyncio.run(scenario())


def test_prior_profile_requires_one_canonical_proposal_attempt_chain() -> None:
    forged_task_id = "unrelated-prior-task"

    class CorruptPriorAuthorityStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True
        corrupt_prior = False

        async def load_prior_completion_verifier_profile(self, proposal_id):
            profile = await super().load_prior_completion_verifier_profile(proposal_id)
            if not self.corrupt_prior or profile is None:
                return profile
            return retask_test_completion_verifier_profile(
                profile,
                task_id=forged_task_id,
            )

        async def load_completion_proposal(self, proposal_id):
            proposal = await super().load_completion_proposal(proposal_id)
            if self.corrupt_prior and proposal is not None and proposal.proposal_id == "proposal-1":
                return proposal.model_copy(update={"task_id": forged_task_id})
            return proposal

    async def scenario() -> None:
        store = CorruptPriorAuthorityStore()
        contract = _contract(
            continuation_policy=CompletionContinuationPolicy(
                rejection_action=CompletionRejectionAction.CONTINUE,
            )
        )
        first_proposal_id = await _proposal(store, contract)
        first_verifier = RecordingVerifier(_rejected_decision())
        first_app = CayuApp(task_store=store, enable_logging=False)
        first_app.register_completion_verifier(contract.verifier, first_verifier)
        first_decision = await first_app.verify_completion_proposal(
            _execution_request(first_proposal_id)
        )
        await store.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id="task-1",
                decision_id=first_decision.decision_id,
                idempotency_key="apply-prior-chain-rejection",
            )
        )
        second_attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id="attempt-2",
                task_id="task-1",
                session_id="session-1",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("worker-profile-2"),
            )
        )
        second_proposal = await store.submit_completion_proposal(
            CompletionProposalCreate(
                proposal_id="proposal-2",
                attempt_id=second_attempt.attempt_id,
                result=CompletionResultReference(
                    kind="task.result",
                    reference_id="result-2",
                    digest=_digest("result-2"),
                ),
            )
        )
        second_verifier = RecordingVerifier(_accepted_decision())
        second_app = CayuApp(task_store=store, enable_logging=False)
        second_app.register_completion_verifier(contract.verifier, second_verifier)
        store.corrupt_prior = True

        with pytest.raises(WorkCompletionConflict, match="work attempt"):
            await second_app.verify_completion_proposal(
                _execution_request(second_proposal.proposal_id, suffix="2")
            )
        assert second_verifier.requests == []
        assert await store.load_completion_verification_claim("proposal-2") is None

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_restart_requires_the_exact_profile_before_replacement_dispatch(
    store_kind: str,
    tmp_path,
) -> None:
    class VersionedVerifier(RecordingVerifier):
        def __init__(
            self,
            decision: CompletionVerifierDecision,
            *,
            behavior_version: str,
            fail: bool = False,
        ) -> None:
            super().__init__(decision)
            self.behavior_version = behavior_version
            self.fail = fail

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:restart-completion-verifier",
                behavior_version=self.behavior_version,
                implementation_version="1",
            )

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            self.requests.append(request)
            if self.fail:
                raise RuntimeError("first verifier worker failed")
            return self.decision

    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        sqlite_path = tmp_path / "verifier-profile-replacement.sqlite"
        store = (
            InMemoryTaskStore(clock=lambda: now[0])
            if store_kind == "memory"
            else SQLiteTaskStore(sqlite_path, clock=lambda: now[0])
        )
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        initial = VersionedVerifier(
            _accepted_decision(),
            behavior_version="v1",
            fail=True,
        )
        first_app = CayuApp(task_store=store, enable_logging=False)
        first_app.register_completion_verifier(contract.verifier, initial)
        original = _execution_request(
            proposal_id,
            lease_seconds=1,
            timeout_seconds=0.5,
        )
        with pytest.raises(CompletionVerifierExecutionError):
            await first_app.verify_completion_proposal(original)

        profile = await store.load_completion_verifier_profile(proposal_id)
        original_claim = await store.load_completion_verification_claim(proposal_id)
        assert profile is not None
        assert original_claim is not None
        assert original_claim.verifier_profile_fingerprint == profile.profile.fingerprint

        if isinstance(store, SQLiteTaskStore):
            await store.close()
            store = SQLiteTaskStore(sqlite_path, clock=lambda: now[0])

        missing = CayuApp(task_store=store, enable_logging=False)
        with pytest.raises(CompletionVerifierUnavailable, match="not registered"):
            await missing.verify_completion_proposal(original)
        assert await store.load_completion_verification_claim(proposal_id) == original_claim

        changed = VersionedVerifier(_accepted_decision(), behavior_version="v2")
        changed_app = CayuApp(task_store=store, enable_logging=False)
        changed_app.register_completion_verifier(contract.verifier, changed)
        with pytest.raises(CompletionVerifierUnavailable, match="durable profile"):
            await changed_app.verify_completion_proposal(original)
        assert changed.requests == []
        assert await store.load_completion_verification_claim(proposal_id) == original_claim

        now[0] += timedelta(seconds=2)
        exact = VersionedVerifier(_accepted_decision(), behavior_version="v1")
        exact_app = CayuApp(task_store=store, enable_logging=False)
        exact_app.register_completion_verifier(contract.verifier, exact)
        replacement = _execution_request(
            proposal_id,
            suffix="replacement",
            lease_seconds=1,
            timeout_seconds=0.5,
        )
        decision = await exact_app.verify_completion_proposal(replacement)
        replacement_claim = await store.load_completion_verification_claim(proposal_id)
        assert replacement_claim is not None
        assert replacement_claim.attempt_number == original_claim.attempt_number + 1
        assert replacement_claim.verifier_profile_fingerprint == profile.profile.fingerprint
        assert decision.verifier_profile_fingerprint == profile.profile.fingerprint
        assert len(exact.requests) == 1

        if isinstance(store, SQLiteTaskStore):
            await store.close()

    asyncio.run(scenario())


def test_registration_uses_complete_reference_and_missing_registration_does_not_claim() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(task_store=store, enable_logging=False)
        different = _verifier_reference(fingerprint="different-configuration")
        app.register_completion_verifier(different, RecordingVerifier(_accepted_decision()))

        with pytest.raises(CompletionVerifierUnavailable, match="exact deterministic"):
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert await store.load_completion_verification_claim(proposal_id) is None

        exact = RecordingVerifier(_accepted_decision())
        app.register_completion_verifier(contract.verifier, exact)
        with pytest.raises(ValueError, match="already registered"):
            app.register_completion_verifier(contract.verifier, exact)
        assert (await app.verify_completion_proposal(_execution_request(proposal_id))).verdict is (
            CompletionVerdict.ACCEPTED
        )

    asyncio.run(scenario())


def test_provider_kind_fails_before_claim_or_adapter_execution() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract(verifier=_verifier_reference(kind=CompletionVerifierKind.PROVIDER))
        proposal_id = await _proposal(store, contract)
        app = CayuApp(task_store=store, enable_logging=False)

        with pytest.raises(ValueError, match="Only deterministic"):
            app.register_completion_verifier(
                contract.verifier,
                RecordingVerifier(_accepted_decision()),
            )
        with pytest.raises(CompletionVerifierUnavailable, match="Provider-backed"):
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert await store.load_completion_verification_claim(proposal_id) is None

    asyncio.run(scenario())


def test_rejected_adapter_outcome_is_bound_without_applying_task_state() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, RecordingVerifier(_rejected_decision()))
        task_before = await store.load_task("task-1")
        assert task_before is not None

        decision = await app.verify_completion_proposal(_execution_request(proposal_id))
        assert decision.verdict is CompletionVerdict.REJECTED
        assert tuple(gap.code for gap in decision.gaps) == ("package.missing",)
        assert await store.load_task("task-1") == task_before

    asyncio.run(scenario())


def test_identical_concurrent_execution_is_single_flight() -> None:
    class BarrierVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            self.calls += 1
            self.started.set()
            await self.release.wait()
            return _accepted_decision()

    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = BarrierVerifier()
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        request = _execution_request(proposal_id)

        first = asyncio.create_task(app.verify_completion_proposal(request))
        await verifier.started.wait()
        second = asyncio.create_task(app.verify_completion_proposal(request))
        await asyncio.sleep(0)
        assert verifier.calls == 1
        verifier.release.set()
        first_decision, second_decision = await asyncio.gather(first, second)
        assert first_decision == second_decision
        assert verifier.calls == 1

    asyncio.run(scenario())


def test_single_flight_waiter_cancellation_is_secret_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "completion-verifier-lock-cancellation-secret"

    class BarrierVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            self.calls += 1
            self.started.set()
            await self.release.wait()
            return _accepted_decision()

    async def scenario() -> BaseException:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = BarrierVerifier()
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_completion_verifier(contract.verifier, verifier)
        request = _execution_request(proposal_id)

        owner = asyncio.create_task(app.verify_completion_proposal(request))
        await verifier.started.wait()
        waiter = asyncio.create_task(app.verify_completion_proposal(request))
        await asyncio.sleep(0)

        assert waiter.cancel(secret)
        assert waiter.cancelling() == 1
        try:
            await waiter
        except asyncio.CancelledError as cancellation:
            safe_cancellation: BaseException = cancellation
        else:  # pragma: no cover - cancellation is the tested delivery mechanism
            raise AssertionError("Queued verifier cancellation was lost.")
        assert waiter.cancelled()
        assert verifier.calls == 1
        assert await store.load_completion_decision("decision-1") is None

        verifier.release.set()
        assert (await owner).decision_id == "decision-1"
        return safe_cancellation

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_separate_apps_cannot_dispatch_from_the_same_live_claim() -> None:
    class BarrierVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            self.calls += 1
            self.started.set()
            await self.release.wait()
            return _accepted_decision()

    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = BarrierVerifier()
        first_app = CayuApp(task_store=store, enable_logging=False)
        second_app = CayuApp(task_store=store, enable_logging=False)
        first_app.register_completion_verifier(contract.verifier, verifier)
        second_app.register_completion_verifier(contract.verifier, verifier)
        request = _execution_request(proposal_id)

        owner = asyncio.create_task(first_app.verify_completion_proposal(request))
        await verifier.started.wait()
        with pytest.raises(WorkCompletionConflict, match="another request"):
            await second_app.verify_completion_proposal(request)
        claim = await store.load_completion_verification_claim(proposal_id)
        assert claim is not None
        assert claim.execution_owner_id is not None
        assert verifier.calls == 1

        verifier.release.set()
        decision = await owner
        restarted = CayuApp(task_store=store, enable_logging=False)
        assert await restarted.verify_completion_proposal(request) == decision
        assert verifier.calls == 1

    asyncio.run(scenario())


def test_preforked_app_mints_process_local_execution_owners_and_dispatches_once() -> None:
    try:
        process_context = multiprocessing.get_context("fork")
    except ValueError:
        pytest.skip("This regression requires operating-system fork support.")

    shared_owner_token = process_context.Value("Q", 0)
    shared_claim_lock = process_context.Lock()
    claim_started = process_context.Event()
    release_verifier = process_context.Event()
    dispatch_count = process_context.Value("i", 0)
    owner_attempts = process_context.Queue()
    outcomes = process_context.Queue()

    class SharedClaimStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def claim_completion_verification(self, request):
            owner_id = request.execution_owner_id
            assert owner_id is not None
            owner_token = int.from_bytes(sha256(owner_id.encode()).digest()[:8], "big") or 1
            with shared_claim_lock:
                owner_attempts.put(owner_id)
                current_owner_token = shared_owner_token.value
                if current_owner_token == 0:
                    claim = await super().claim_completion_verification(request)
                    shared_owner_token.value = owner_token
                    claim_started.set()
                    return claim
                if current_owner_token != owner_token:
                    raise WorkCompletionConflict(
                        "Verification-claim identity is already bound to another request."
                    )
            return await super().claim_completion_verification(request)

    class ProcessBarrierVerifier(_TestCompletionVerifier):
        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            with dispatch_count.get_lock():
                dispatch_count.value += 1
            while not release_verifier.is_set():
                await asyncio.sleep(0.01)
            return _accepted_decision()

    store = SharedClaimStore()
    contract = _contract()
    proposal_id = asyncio.run(_proposal(store, contract))
    app = CayuApp(task_store=store, enable_logging=False)
    app.register_completion_verifier(contract.verifier, ProcessBarrierVerifier())
    request = _execution_request(proposal_id)

    def run_worker(label: str) -> None:
        async def execute() -> None:
            try:
                decision = await app.verify_completion_proposal(request)
            except BaseException as error:
                outcomes.put((label, "error", type(error).__name__))
            else:
                outcomes.put((label, "decision", decision.decision_id))

        asyncio.run(execute())

    first = process_context.Process(target=run_worker, args=("first",))
    second = process_context.Process(target=run_worker, args=("second",))
    first.start()
    try:
        assert claim_started.wait(timeout=5)
        second.start()
        second.join(timeout=3)
        second_blocked_on_duplicate_dispatch = second.is_alive()
        release_verifier.set()
        first.join(timeout=5)
        second.join(timeout=5)
        assert not second_blocked_on_duplicate_dispatch
        assert first.exitcode == 0
        assert second.exitcode == 0
        assert dispatch_count.value == 1
        attempted_owners = {owner_attempts.get(timeout=1), owner_attempts.get(timeout=1)}
        assert len(attempted_owners) == 2
        observed = {outcomes.get(timeout=1), outcomes.get(timeout=1)}
        assert ("first", "decision", request.decision_id) in observed
        assert ("second", "error", "WorkCompletionConflict") in observed
    finally:
        release_verifier.set()
        for process in (first, second):
            if process.pid is None:
                continue
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            process.close()
        owner_attempts.close()
        owner_attempts.join_thread()
        outcomes.close()
        outcomes.join_thread()


def test_inherited_active_verifier_state_fails_before_store_mutation() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(
            contract.verifier,
            RecordingVerifier(_accepted_decision()),
        )
        coordinator = app._completion_verifier_coordinator
        coordinator._execution_owner_process_id = -1
        coordinator._adapter_capacity_reservations.add(object())

        with pytest.raises(
            CompletionVerifierExecutionError,
            match="inherited active execution state",
        ):
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert await store.load_completion_verification_claim(proposal_id) is None

    asyncio.run(scenario())


def test_proposal_scoped_single_flight_rejects_a_second_decision_identity() -> None:
    class BarrierVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            self.calls += 1
            self.started.set()
            await self.release.wait()
            return _accepted_decision()

    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = BarrierVerifier()
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)

        first = asyncio.create_task(app.verify_completion_proposal(_execution_request(proposal_id)))
        await verifier.started.wait()
        conflicting = asyncio.create_task(
            app.verify_completion_proposal(
                _execution_request(proposal_id).model_copy(
                    update={"decision_id": "decision-conflicting"}
                )
            )
        )
        await asyncio.sleep(0)
        assert verifier.calls == 1
        verifier.release.set()
        assert (await first).decision_id == "decision-1"
        with pytest.raises(WorkCompletionConflict, match="conflicting authority"):
            await conflicting
        assert verifier.calls == 1
        assert await store.load_completion_decision("decision-conflicting") is None

    asyncio.run(scenario())


def test_proposal_scoped_drain_blocks_replacement_claim_after_lease_expiry() -> None:
    class ResistantVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = asyncio.Event()
            self.calls = 0

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            self.calls += 1
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()
            self.finished.set()
            return _accepted_decision()

    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        store = InMemoryTaskStore(clock=lambda: now[0])
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = ResistantVerifier()
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)

        with pytest.raises(CompletionVerifierExecutionError, match="bounded execution timeout"):
            await app.verify_completion_proposal(
                _execution_request(
                    proposal_id,
                    lease_seconds=1,
                    timeout_seconds=0.01,
                )
            )
        await verifier.cancelled.wait()
        now[0] += timedelta(seconds=2)
        replacement = _execution_request(
            proposal_id,
            suffix="2",
            lease_seconds=1,
            timeout_seconds=0.5,
        )
        with pytest.raises(CompletionVerifierExecutionError, match="still draining"):
            await app.verify_completion_proposal(replacement)
        assert verifier.calls == 1
        current_claim = await store.load_completion_verification_claim(proposal_id)
        assert current_claim is not None
        assert current_claim.claim_id == "claim-1"

        verifier.release.set()
        await verifier.finished.wait()
        await asyncio.sleep(0)
        assert (await app.verify_completion_proposal(replacement)).decision_id == "decision-2"
        assert verifier.calls == 2

    asyncio.run(scenario())


def test_retry_cannot_steal_a_completed_adapter_drain_before_its_callback() -> None:
    class ResistantVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = asyncio.Event()
            self.calls = 0

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            self.calls += 1
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()
            self.finished.set()
            return _accepted_decision()

    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        store = InMemoryTaskStore(clock=lambda: now[0])
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = ResistantVerifier()
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        coordinator = app._completion_verifier_coordinator
        original_release = coordinator._release_adapter_task
        callback_started = asyncio.Event()
        callback_release = asyncio.Event()
        callback_waiters: list[asyncio.Task[None]] = []

        def delayed_release(operation_key, completed) -> None:
            async def release_after_barrier() -> None:
                callback_started.set()
                await callback_release.wait()
                original_release(operation_key, completed)

            callback_waiters.append(asyncio.create_task(release_after_barrier()))

        object.__setattr__(coordinator, "_release_adapter_task", delayed_release)

        with pytest.raises(CompletionVerifierExecutionError, match="bounded execution timeout"):
            await app.verify_completion_proposal(
                _execution_request(
                    proposal_id,
                    lease_seconds=1,
                    timeout_seconds=0.01,
                )
            )
        await verifier.cancelled.wait()
        replacement = _execution_request(
            proposal_id,
            suffix="2",
            lease_seconds=1,
            timeout_seconds=0.5,
        )

        async def retry_when_adapter_finishes() -> CompletionDecision:
            await verifier.finished.wait()
            await callback_started.wait()
            return await app.verify_completion_proposal(replacement)

        immediate_retry = asyncio.create_task(retry_when_adapter_finishes())
        await asyncio.sleep(0)
        verifier.release.set()
        with pytest.raises(CompletionVerifierExecutionError, match="still draining"):
            await immediate_retry
        assert verifier.calls == 1

        object.__setattr__(coordinator, "_release_adapter_task", original_release)
        callback_release.set()
        await asyncio.gather(*callback_waiters)

        now[0] += timedelta(seconds=2)
        for _ in range(20):
            try:
                decision = await app.verify_completion_proposal(replacement)
            except CompletionVerifierExecutionError as error:
                assert "still draining" in str(error)
                await asyncio.sleep(0)
                continue
            break
        else:  # pragma: no cover - bounded settlement should finish promptly
            raise AssertionError("Completion-verifier drain settlement did not finish.")

        assert decision.decision_id == replacement.decision_id
        assert verifier.calls == 2
        await asyncio.sleep(0)
        assert not coordinator._draining_adapter_tasks
        assert not coordinator._claim_heartbeat_tasks

    asyncio.run(scenario())


def test_caller_cancellation_cancels_adapter_and_publishes_no_decision() -> None:
    class CancellableVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            self.started.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = CancellableVerifier()
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        owner = asyncio.create_task(app.verify_completion_proposal(_execution_request(proposal_id)))
        await verifier.started.wait()
        owner.cancel("stop verifier")
        with pytest.raises(asyncio.CancelledError, match="stop verifier"):
            await owner
        assert owner.cancelled()
        assert owner.cancelling() == 1
        await verifier.cancelled.wait()
        assert await store.load_completion_decision("decision-1") is None

    asyncio.run(scenario())


def test_concurrent_caller_cancellation_and_adapter_fatal_signal_are_both_preserved() -> None:
    class ConcurrentFatalVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.owner: asyncio.Task[CompletionDecision] | None = None

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            owner = self.owner
            assert owner is not None
            owner.cancel("stop concurrent verifier")
            raise SystemExit("adapter process-control signal")

    async def scenario() -> tuple[BaseExceptionGroup, asyncio.Task[CompletionDecision]]:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = ConcurrentFatalVerifier()
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        owner = asyncio.create_task(app.verify_completion_proposal(_execution_request(proposal_id)))
        verifier.owner = owner
        with pytest.raises(BaseExceptionGroup) as captured:
            await owner
        assert await store.load_completion_decision("decision-1") is None
        return captured.value, owner

    failure, owner = asyncio.run(scenario())
    assert owner.cancelling() == 1
    assert not owner.cancelled()
    assert any(isinstance(item, SystemExit) for item in failure.exceptions)
    assert any(isinstance(item, asyncio.CancelledError) for item in failure.exceptions)


def test_adapter_and_claim_renewal_failures_are_both_preserved() -> None:
    class RenewalFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.renewal_count = 0

        async def renew_completion_verification_claim(self, request):
            self.renewal_count += 1
            if self.renewal_count == 1:
                return await super().renew_completion_verification_claim(request)
            raise ConnectionError("claim renewal failed")

    class FatalAfterRenewalFailureVerifier(_TestCompletionVerifier):
        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            try:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            except asyncio.CancelledError:
                raise SystemExit("adapter failed after claim renewal") from None

    async def scenario() -> BaseExceptionGroup:
        store = RenewalFailureStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(
            contract.verifier,
            FatalAfterRenewalFailureVerifier(),
        )

        with pytest.raises(BaseExceptionGroup) as captured:
            await app.verify_completion_proposal(
                _execution_request(
                    proposal_id,
                    lease_seconds=1,
                    timeout_seconds=0.9,
                )
            )
        assert await store.load_completion_decision("decision-1") is None
        return captured.value

    failure = asyncio.run(scenario())
    assert tuple(type(item) for item in failure.exceptions) == (
        SystemExit,
        ConnectionError,
    )


def test_in_memory_claim_renewal_is_nondecreasing_across_backward_clock_steps() -> None:
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        store = InMemoryTaskStore(clock=lambda: now[0])
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        request = _execution_request(
            proposal_id,
            lease_seconds=300,
            timeout_seconds=30.0,
        )
        verifier_profile = await prepare_test_completion_verifier_profile(
            store,
            proposal_id,
        )
        claim_request = CompletionVerificationClaimRequest(
            claim_id=request.claim_id,
            proposal_id=request.proposal_id,
            worker_id=request.worker_id,
            execution_owner_id="cver_test_owner",
            verifier=contract.verifier,
            verifier_profile_fingerprint=verifier_profile.profile.fingerprint,
            lease_seconds=request.lease_seconds,
            execution_timeout_seconds=request.execution_timeout_seconds,
        )
        claimed = await store.claim_completion_verification(claim_request)

        now[0] -= timedelta(seconds=60)
        small_skew = await store.renew_completion_verification_claim(claim_request)
        assert small_skew.claimed_at == claimed.claimed_at
        assert small_skew.attempt_number == claimed.attempt_number
        assert small_skew.lease_expires_at == claimed.lease_expires_at

        now[0] -= timedelta(seconds=600)
        large_skew = await store.renew_completion_verification_claim(claim_request)
        assert large_skew == small_skew
        assert await store.load_completion_verification_claim(proposal_id) == claimed

    asyncio.run(scenario())


def test_nonextending_initial_claim_renewal_fails_before_adapter_dispatch() -> None:
    class NonextendingRenewalStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def renew_completion_verification_claim(self, request):
            renewed = await super().renew_completion_verification_claim(request)
            return renewed.model_copy(
                update={"lease_expires_at": renewed.lease_expires_at - timedelta(seconds=1)}
            )

    async def scenario() -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        store = NonextendingRenewalStore(clock=lambda: now)
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)

        with pytest.raises(WorkCompletionConflict, match="compare-and-extend"):
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert verifier.requests == []
        assert await store.load_completion_decision("decision-1") is None

    asyncio.run(scenario())


def test_nonextending_heartbeat_renewal_cancels_adapter_before_publication() -> None:
    class NonextendingHeartbeatStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self, *, clock) -> None:
            super().__init__(clock=clock)
            self.renewal_count = 0

        async def renew_completion_verification_claim(self, request):
            renewed = await super().renew_completion_verification_claim(request)
            self.renewal_count += 1
            if self.renewal_count == 1:
                return renewed
            return renewed.model_copy(
                update={"lease_expires_at": renewed.lease_expires_at - timedelta(milliseconds=100)}
            )

    class CancellableVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    async def scenario() -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        store = NonextendingHeartbeatStore(clock=lambda: now)
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = CancellableVerifier()
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)

        with pytest.raises(WorkCompletionConflict, match="compare-and-extend"):
            await app.verify_completion_proposal(
                _execution_request(
                    proposal_id,
                    lease_seconds=1,
                    timeout_seconds=0.9,
                )
            )
        assert verifier.started.is_set()
        assert verifier.cancelled.is_set()
        assert store.renewal_count == 2
        assert await store.load_completion_decision("decision-1") is None

    asyncio.run(scenario())


def test_renewal_adapter_and_caller_cancellation_failures_are_all_preserved() -> None:
    class RenewalFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.renewal_count = 0

        async def renew_completion_verification_claim(self, request):
            self.renewal_count += 1
            if self.renewal_count == 1:
                return await super().renew_completion_verification_claim(request)
            raise ConnectionError("claim renewal failed during cancellation")

    class CancellingFatalVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.owner: asyncio.Task[CompletionDecision] | None = None

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            try:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            except asyncio.CancelledError:
                owner = self.owner
                assert owner is not None
                owner.cancel("stop while claim renewal fails")
                raise SystemExit("adapter failed during cancellation") from None

    async def scenario() -> tuple[BaseExceptionGroup, asyncio.Task[CompletionDecision]]:
        store = RenewalFailureStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = CancellingFatalVerifier()
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        owner = asyncio.create_task(
            app.verify_completion_proposal(
                _execution_request(
                    proposal_id,
                    lease_seconds=1,
                    timeout_seconds=0.9,
                )
            )
        )
        verifier.owner = owner

        with pytest.raises(BaseExceptionGroup) as captured:
            await owner
        assert await store.load_completion_decision("decision-1") is None
        return captured.value, owner

    failure, owner = asyncio.run(scenario())
    assert tuple(type(item) for item in failure.exceptions) == (
        SystemExit,
        ConnectionError,
        asyncio.CancelledError,
    )
    assert owner.cancelling() == 1
    assert not owner.cancelled()


def test_claim_renewal_failure_does_not_duplicate_its_adapter_cancellation() -> None:
    class RenewalFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.renewal_count = 0

        async def renew_completion_verification_claim(self, request):
            self.renewal_count += 1
            if self.renewal_count == 1:
                return await super().renew_completion_verification_claim(request)
            raise ConnectionError("claim renewal failed")

    class CancellationRespectingVerifier(_TestCompletionVerifier):
        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def scenario() -> None:
        store = RenewalFailureStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(
            contract.verifier,
            CancellationRespectingVerifier(),
        )

        with pytest.raises(ConnectionError, match="claim renewal failed"):
            await app.verify_completion_proposal(
                _execution_request(
                    proposal_id,
                    lease_seconds=1,
                    timeout_seconds=0.9,
                )
            )
        assert await store.load_completion_decision("decision-1") is None

    asyncio.run(scenario())


def test_claim_renewal_and_real_caller_cancellation_remain_distinct() -> None:
    class RenewalFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.renewal_count = 0

        async def renew_completion_verification_claim(self, request):
            self.renewal_count += 1
            if self.renewal_count == 1:
                return await super().renew_completion_verification_claim(request)
            raise ConnectionError("claim renewal failed during cancellation")

    class CallerCancellingVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.owner: asyncio.Task[CompletionDecision] | None = None

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            try:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            except asyncio.CancelledError:
                owner = self.owner
                assert owner is not None
                owner.cancel("caller cancellation during renewal failure")
                raise

    async def scenario() -> tuple[BaseExceptionGroup, asyncio.Task[CompletionDecision]]:
        store = RenewalFailureStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = CallerCancellingVerifier()
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        owner = asyncio.create_task(
            app.verify_completion_proposal(
                _execution_request(
                    proposal_id,
                    lease_seconds=1,
                    timeout_seconds=0.9,
                )
            )
        )
        verifier.owner = owner

        with pytest.raises(BaseExceptionGroup) as captured:
            await owner
        assert await store.load_completion_decision("decision-1") is None
        return captured.value, owner

    failure, owner = asyncio.run(scenario())
    assert tuple(type(item) for item in failure.exceptions) == (
        ConnectionError,
        asyncio.CancelledError,
    )
    assert owner.cancelling() == 1
    assert not owner.cancelled()


def test_adapter_timeout_retains_cancellation_resistant_child_without_publication() -> None:
    class ResistantVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = asyncio.Event()
            self.calls = 0

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            self.calls += 1
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()
            self.finished.set()
            return _accepted_decision()

    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = ResistantVerifier()
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)

        request = _execution_request(
            proposal_id,
            lease_seconds=2,
            timeout_seconds=0.01,
        )
        with pytest.raises(CompletionVerifierExecutionError, match="bounded execution timeout"):
            await app.verify_completion_proposal(request)
        await verifier.cancelled.wait()
        with pytest.raises(CompletionVerifierExecutionError, match="still draining"):
            await app.verify_completion_proposal(request)
        assert verifier.calls == 1
        assert await store.load_completion_decision("decision-1") is None
        verifier.release.set()
        await verifier.finished.wait()
        await asyncio.sleep(0)
        assert (await app.verify_completion_proposal(request)).verdict is CompletionVerdict.ACCEPTED
        assert verifier.calls == 2

    asyncio.run(scenario())


def test_profile_preparation_and_reconciliation_failures_remain_ordered() -> None:
    class ReconciliationFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.fail_reconciliation = False

        async def prepare_completion_verifier_profile(self, request):
            await super().prepare_completion_verifier_profile(request)
            self.fail_reconciliation = True
            raise ConnectionError("profile preparation acknowledgement lost")

        async def load_completion_verifier_profile(self, proposal_id):
            if self.fail_reconciliation:
                raise RuntimeError("profile reconciliation unavailable")
            return await super().load_completion_verifier_profile(proposal_id)

    async def scenario() -> BaseExceptionGroup:
        store = ReconciliationFailureStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)

        with pytest.raises(ExceptionGroup) as captured:
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert verifier.requests == []
        assert await store.load_completion_verification_claim(proposal_id) is None
        durable = await InMemoryTaskStore.load_completion_verifier_profile(store, proposal_id)
        assert durable is not None
        return captured.value

    failure = asyncio.run(scenario())
    assert len(failure.exceptions) == 2
    assert isinstance(failure.exceptions[0], ConnectionError)
    assert isinstance(failure.exceptions[1], RuntimeError)


def test_decision_publication_acknowledgement_loss_reconciles_without_second_call() -> None:
    class AckLossStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.fail_once = True

        async def record_completion_decision(self, request):
            decision = await super().record_completion_decision(request)
            if self.fail_once:
                self.fail_once = False
                raise ConnectionError("decision acknowledgement lost")
            return decision

    async def scenario() -> None:
        store = AckLossStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)

        decision = await app.verify_completion_proposal(_execution_request(proposal_id))
        assert decision.decision_id == "decision-1"
        assert len(verifier.requests) == 1
        assert await app.verify_completion_proposal(_execution_request(proposal_id)) == decision
        assert len(verifier.requests) == 1

    asyncio.run(scenario())


def test_decision_publication_process_control_is_not_suppressed_by_reconciliation() -> None:
    class ProcessControlAfterCommitStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def record_completion_decision(self, request):
            await super().record_completion_decision(request)
            raise SystemExit("stop after completion decision commit")

    async def scenario() -> None:
        store = ProcessControlAfterCommitStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        request = _execution_request(proposal_id)

        with pytest.raises(SystemExit, match="stop after completion decision commit"):
            await app.verify_completion_proposal(request)
        durable = await store.load_completion_decision(request.decision_id)
        assert durable is not None
        restarted = CayuApp(task_store=store, enable_logging=False)
        assert await restarted.verify_completion_proposal(request) == durable
        assert len(verifier.requests) == 1

    asyncio.run(scenario())


def test_publication_and_reconciliation_failures_remain_ordered() -> None:
    class ReconciliationFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.fail_reconciliation = False

        async def record_completion_decision(self, request):
            await super().record_completion_decision(request)
            self.fail_reconciliation = True
            raise ConnectionError("publication acknowledgement lost")

        async def load_completion_decision_for_proposal(self, proposal_id):
            if self.fail_reconciliation:
                raise RuntimeError("decision reconciliation unavailable")
            return await super().load_completion_decision_for_proposal(proposal_id)

    async def scenario() -> BaseExceptionGroup:
        store = ReconciliationFailureStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(
            contract.verifier,
            RecordingVerifier(_accepted_decision()),
        )

        with pytest.raises(ExceptionGroup) as captured:
            await app.verify_completion_proposal(_execution_request(proposal_id))
        return captured.value

    failure = asyncio.run(scenario())
    assert len(failure.exceptions) == 2
    assert isinstance(failure.exceptions[0], ConnectionError)
    assert isinstance(failure.exceptions[1], RuntimeError)


def test_cancellation_after_decision_commit_preserves_signal_and_exact_replay() -> None:
    class CommitBarrierStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.committed = asyncio.Event()

        async def record_completion_decision(self, request):
            decision = await super().record_completion_decision(request)
            self.committed.set()
            await asyncio.Event().wait()
            return decision

    async def scenario() -> None:
        store = CommitBarrierStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        request = _execution_request(proposal_id)

        owner = asyncio.create_task(app.verify_completion_proposal(request))
        await store.committed.wait()
        owner.cancel("stop after decision commit")
        with pytest.raises(asyncio.CancelledError, match="stop after decision commit"):
            await owner
        assert owner.cancelled()
        assert owner.cancelling() == 1

        durable = await store.load_completion_decision(request.decision_id)
        assert durable is not None
        assert await app.verify_completion_proposal(request) == durable
        assert len(verifier.requests) == 1

    asyncio.run(scenario())


def test_post_commit_failure_cannot_turn_caller_cancellation_into_success() -> None:
    class TranslatedCancellationStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.committed = asyncio.Event()

        async def record_completion_decision(self, request):
            await super().record_completion_decision(request)
            self.committed.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise ConnectionError("store translated forwarded cancellation") from None

    async def scenario() -> tuple[BaseExceptionGroup, asyncio.Task[CompletionDecision]]:
        store = TranslatedCancellationStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(
            contract.verifier,
            RecordingVerifier(_accepted_decision()),
        )
        request = _execution_request(proposal_id)
        owner = asyncio.create_task(app.verify_completion_proposal(request))
        await store.committed.wait()
        owner.cancel("stop after translated commit")
        with pytest.raises(BaseExceptionGroup) as captured:
            await owner
        assert await store.load_completion_decision(request.decision_id) is not None
        return captured.value, owner

    failure, owner = asyncio.run(scenario())
    assert owner.cancelling() == 1
    assert not owner.cancelled()
    assert any(isinstance(item, ConnectionError) for item in failure.exceptions)
    assert any(isinstance(item, asyncio.CancelledError) for item in failure.exceptions)


def test_decision_publication_cancellation_retains_store_settlement_evidence(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "completion-publication-settlement-secret"

    class SettlementEvidenceStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.decision_committed = asyncio.Event()

        async def record_completion_decision(self, request):
            decision = await super().record_completion_decision(request)
            self.decision_committed.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                raise cancellation from BaseExceptionGroup(
                    "Postgres mutation cancellation settlement failed.",
                    [
                        ConnectionError(secret),
                        RuntimeError("connection abort failed"),
                    ],
                )
            return decision  # pragma: no cover - the test always cancels publication

    async def scenario() -> tuple[asyncio.CancelledError, asyncio.Task[CompletionDecision]]:
        store = SettlementEvidenceStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_completion_verifier(contract.verifier, verifier)
        request = _execution_request(proposal_id)

        owner = asyncio.create_task(app.verify_completion_proposal(request))
        await store.decision_committed.wait()
        owner.cancel("stop during decision publication settlement")
        with pytest.raises(asyncio.CancelledError) as captured:
            await owner
        assert await store.load_completion_decision(request.decision_id) is not None
        assert len(verifier.requests) == 1
        return captured.value, owner

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        cancellation, owner = asyncio.run(scenario())

    assert owner.cancelled()
    assert owner.cancelling() == 1
    assert isinstance(cancellation.__cause__, BaseExceptionGroup)
    assert _exception_leaf_types(cancellation.__cause__) == (ConnectionError, RuntimeError)
    _assert_secret_absent_from_cayu_error(cancellation, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_public_cancellation_cause_chain_cannot_reconstruct_split_secret(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cause_fragment = "settlement-left-edge"
    cancellation_fragment = "caller-right-edge"
    probe = asyncio.CancelledError(cancellation_fragment)
    probe.__cause__ = ConnectionError(cause_fragment)
    probe_rendering = "".join(traceback_module.format_exception(probe))
    secret_start = probe_rendering.index(cause_fragment)
    secret_end = probe_rendering.index(cancellation_fragment) + len(cancellation_fragment)
    split_secret = probe_rendering[secret_start:secret_end]

    class SettlementEvidenceStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.decision_committed = asyncio.Event()

        async def record_completion_decision(self, request):
            decision = await super().record_completion_decision(request)
            self.decision_committed.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                raise cancellation from ConnectionError(cause_fragment)
            return decision  # pragma: no cover - the test always cancels publication

    async def scenario() -> tuple[asyncio.CancelledError, asyncio.Task[CompletionDecision]]:
        store = SettlementEvidenceStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(split_secret),
            enable_logging=False,
        )
        app.register_completion_verifier(
            contract.verifier,
            RecordingVerifier(_accepted_decision()),
        )
        request = _execution_request(proposal_id)

        owner = asyncio.create_task(app.verify_completion_proposal(request))
        await store.decision_committed.wait()
        owner.cancel(cancellation_fragment)
        with pytest.raises(asyncio.CancelledError) as captured:
            await owner
        assert await store.load_completion_decision(request.decision_id) is not None
        return captured.value, owner

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        cancellation, owner = asyncio.run(scenario())

    assert owner.cancelled()
    assert owner.cancelling() == 1
    assert isinstance(cancellation.__cause__, ConnectionError)
    assert cause_fragment not in str(cancellation.__cause__)
    _assert_secret_absent_from_cayu_error(cancellation, split_secret)
    captured = capsys.readouterr()
    assert split_secret not in caplog.text
    assert split_secret not in captured.out
    assert split_secret not in captured.err
    assert all(split_secret not in str(item.message) for item in caught_warnings)


def test_claim_failure_after_caller_cancellation_is_bounded_and_split_secret_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = (
        "BaseExceptionGroup('Completion verification claim failed after caller cancellation.', "
        "[ConnectionError('token"
    )

    class TranslatedCancellationStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.claim_started = asyncio.Event()

        async def claim_completion_verification(self, request):
            del request
            self.claim_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise ConnectionError("token" + ("x" * 8_192)) from None

    async def scenario() -> tuple[BaseExceptionGroup, asyncio.Task[CompletionDecision]]:
        store = TranslatedCancellationStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_completion_verifier(contract.verifier, verifier)

        owner = asyncio.create_task(app.verify_completion_proposal(_execution_request(proposal_id)))
        await store.claim_started.wait()
        owner.cancel("stop during claim")
        with pytest.raises(BaseExceptionGroup) as captured:
            await owner
        assert await store.load_completion_verification_claim(proposal_id) is None
        assert verifier.requests == []
        return captured.value, owner

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        failure, owner = asyncio.run(scenario())

    assert owner.cancelling() == 1
    assert not owner.cancelled()
    assert _exception_leaf_types(failure) == (ConnectionError, asyncio.CancelledError)
    assert len(str(failure).encode("utf-8")) <= MAX_DIAGNOSTIC_UTF8_BYTES
    assert len(repr(failure).encode("utf-8")) <= MAX_DIAGNOSTIC_UTF8_BYTES
    _assert_secret_absent_from_cayu_error(failure, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_claim_cancellation_retains_store_settlement_evidence_through_coordinator(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "completion-claim-settlement-secret"

    class SettlementEvidenceStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.claim_started = asyncio.Event()

        async def claim_completion_verification(self, request):
            del request
            self.claim_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                raise cancellation from BaseExceptionGroup(
                    "Postgres mutation cancellation settlement failed.",
                    [
                        ConnectionError(secret),
                        RuntimeError("connection abort failed"),
                    ],
                )

    async def scenario() -> tuple[asyncio.CancelledError, asyncio.Task[CompletionDecision]]:
        store = SettlementEvidenceStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_completion_verifier(contract.verifier, verifier)

        owner = asyncio.create_task(app.verify_completion_proposal(_execution_request(proposal_id)))
        await store.claim_started.wait()
        owner.cancel("stop during claim settlement")
        with pytest.raises(asyncio.CancelledError) as captured:
            await owner
        assert await store.load_completion_verification_claim(proposal_id) is None
        assert verifier.requests == []
        return captured.value, owner

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        cancellation, owner = asyncio.run(scenario())

    assert owner.cancelled()
    assert owner.cancelling() == 1
    assert isinstance(cancellation.__cause__, BaseExceptionGroup)
    assert _exception_leaf_types(cancellation.__cause__) == (ConnectionError, RuntimeError)
    _assert_secret_absent_from_cayu_error(cancellation, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_invalid_adapter_result_publishes_no_decision() -> None:
    class InvalidVerifier(_TestCompletionVerifier):
        async def verify(self, request):
            del request
            return object()

    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, InvalidVerifier())

        with pytest.raises(CompletionVerifierExecutionError, match="invalid decision"):
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert await store.load_completion_decision("decision-1") is None

    asyncio.run(scenario())


def test_malformed_exact_adapter_result_is_item_bounded_before_copying() -> None:
    class EndlessOutcomes:
        def __init__(self) -> None:
            self.iterated = 0

        def __iter__(self):
            while True:
                self.iterated += 1
                yield _accepted_decision().criterion_outcomes[0]

    class MalformedVerifier(_TestCompletionVerifier):
        def __init__(self, outcome: CompletionVerifierDecision) -> None:
            self.outcome = outcome

        async def verify(self, request):
            del request
            return self.outcome

    async def scenario() -> int:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        values = EndlessOutcomes()
        malformed = _accepted_decision()
        object.__setattr__(malformed, "criterion_outcomes", values)
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, MalformedVerifier(malformed))

        with pytest.raises(CompletionVerifierExecutionError, match="invalid decision"):
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert await store.load_completion_decision("decision-1") is None
        return values.iterated

    assert asyncio.run(scenario()) == WORK_CONTRACT_MAX_CRITERIA + 1


def test_malformed_nested_adapter_result_is_item_bounded_before_copying() -> None:
    class EndlessEvidence:
        def __init__(self) -> None:
            self.iterated = 0

        def __iter__(self):
            while True:
                self.iterated += 1
                yield object()

    class MalformedVerifier(_TestCompletionVerifier):
        def __init__(self, outcome: CompletionVerifierDecision) -> None:
            self.outcome = outcome

        async def verify(self, request):
            del request
            return self.outcome

    async def scenario() -> int:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        values = EndlessEvidence()
        criterion = _accepted_decision().criterion_outcomes[0]
        object.__setattr__(criterion, "evidence_references", values)
        malformed = _accepted_decision().model_copy(update={"criterion_outcomes": (criterion,)})
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, MalformedVerifier(malformed))

        with pytest.raises(CompletionVerifierExecutionError, match="invalid decision"):
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert await store.load_completion_decision("decision-1") is None
        return values.iterated

    assert asyncio.run(scenario()) == WORK_COMPLETION_OUTCOME_MAX_EVIDENCE_REFERENCES + 1


def test_malformed_store_proposal_is_item_bounded_before_context_copy() -> None:
    class EndlessEvidence:
        def __init__(self) -> None:
            self.iterated = 0

        def __iter__(self):
            while True:
                self.iterated += 1
                yield object()

    class MalformedProposalStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.values = EndlessEvidence()

        async def load_completion_proposal(self, proposal_id):
            proposal = await super().load_completion_proposal(proposal_id)
            if proposal is not None:
                object.__setattr__(proposal, "evidence_references", self.values)
            return proposal

    async def scenario() -> int:
        store = MalformedProposalStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, RecordingVerifier(_accepted_decision()))

        with pytest.raises(WorkCompletionConflict, match="proposal.*authority"):
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert await store.load_completion_verification_claim(proposal_id) is None
        return store.values.iterated

    assert asyncio.run(scenario()) == WORK_CONTRACT_MAX_EVIDENCE_REFERENCES + 1


def test_malformed_store_contract_is_item_bounded_before_context_copy() -> None:
    class EndlessCriteria:
        def __init__(self) -> None:
            self.iterated = 0

        def __iter__(self):
            while True:
                self.iterated += 1
                yield object()

    class MalformedContractStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.values = EndlessCriteria()

        async def load_work_contract(self, reference):
            contract = await super().load_work_contract(reference)
            if contract is not None:
                object.__setattr__(contract, "criteria", self.values)
            return contract

    async def scenario() -> int:
        store = MalformedContractStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, RecordingVerifier(_accepted_decision()))

        with pytest.raises(WorkCompletionConflict, match="contract.*authority"):
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert await store.load_completion_verification_claim(proposal_id) is None
        return store.values.iterated

    assert asyncio.run(scenario()) == WORK_CONTRACT_MAX_CRITERIA + 1


def test_malformed_store_decision_is_item_bounded_before_replay_copy() -> None:
    class EndlessOutcomes:
        def __init__(self) -> None:
            self.iterated = 0

        def __iter__(self):
            while True:
                self.iterated += 1
                yield _accepted_decision().criterion_outcomes[0]

    class MalformedDecisionStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.values = EndlessOutcomes()
            self.malformed = False

        async def load_completion_decision(self, decision_id):
            decision = await super().load_completion_decision(decision_id)
            if decision is not None and self.malformed:
                object.__setattr__(decision, "criterion_outcomes", self.values)
            return decision

    async def scenario() -> int:
        store = MalformedDecisionStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, RecordingVerifier(_accepted_decision()))
        request = _execution_request(proposal_id)
        await app.verify_completion_proposal(request)

        store.malformed = True
        restarted = CayuApp(task_store=store, enable_logging=False)
        with pytest.raises(WorkCompletionConflict, match="decision.*authority"):
            await restarted.verify_completion_proposal(request)
        return store.values.iterated

    assert asyncio.run(scenario()) == WORK_CONTRACT_MAX_CRITERIA + 1


def test_verifier_decision_headroom_covers_worst_case_authority_escaping() -> None:
    def accepted_with_summary(summary_bytes: int) -> CompletionVerifierDecision:
        return CompletionVerifierDecision(
            verdict=CompletionVerdict.ACCEPTED,
            criterion_outcomes=tuple(
                CompletionCriterionOutcome(
                    criterion_id=f"criterion-{index}",
                    status=CriterionOutcomeStatus.SATISFIED,
                    reason_code="package.ready",
                    satisfaction_basis=(CompletionSatisfactionBasis.VERIFIER_ASSERTION),
                    summary="x" * summary_bytes,
                )
                for index in range(WORK_CONTRACT_MAX_CRITERIA)
            ),
        )

    lower = 0
    upper = 16 * 1024
    while lower + 1 < upper:
        candidate = (lower + upper) // 2
        try:
            accepted_with_summary(candidate)
        except ValueError:
            upper = candidate
        else:
            lower = candidate
    outcome = accepted_with_summary(lower)
    assert len(outcome.model_dump_json().encode()) <= (WORK_COMPLETION_VERIFIER_DECISION_MAX_BYTES)

    maximally_escaped_identity = "\x01" * 255 + "a"
    verifier = CompletionVerifierRef(
        verifier_id=maximally_escaped_identity,
        version=maximally_escaped_identity,
        kind=CompletionVerifierKind.DETERMINISTIC,
        configuration_fingerprint=_digest("escaped-authority"),
    )
    publication = CompletionDecisionCreate(
        decision_id=maximally_escaped_identity,
        proposal_id=maximally_escaped_identity,
        claim_id=maximally_escaped_identity,
        worker_id=maximally_escaped_identity,
        verifier=verifier,
        verifier_profile_fingerprint=_digest("escaped-verifier-profile"),
        verdict=outcome.verdict,
        criterion_outcomes=outcome.criterion_outcomes,
    )
    published = CompletionDecision(
        decision_id=publication.decision_id,
        proposal_id=publication.proposal_id,
        claim_id=publication.claim_id,
        worker_id=publication.worker_id,
        verifier=publication.verifier,
        verifier_profile_fingerprint=publication.verifier_profile_fingerprint,
        verdict=publication.verdict,
        criterion_outcomes=publication.criterion_outcomes,
        task_id="\x01" * 2047 + "a",
        attempt_id=maximally_escaped_identity,
        contract=WorkContractRef(
            contract_id=maximally_escaped_identity,
            version=1,
            fingerprint=_digest("escaped-contract"),
        ),
        claim_authority_sha256=_digest("escaped-claim-authority"),
        request_sha256=completion_decision_request_sha256(publication),
        gap_fingerprint=completion_gap_fingerprint(publication),
        decided_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert published.verifier == verifier


def test_adapter_decision_is_checked_against_contract_before_store_mutation() -> None:
    class RecordingDecisionStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.decision_writes = 0

        async def record_completion_decision(self, request):
            self.decision_writes += 1
            return await super().record_completion_decision(request)

    async def scenario() -> None:
        store = RecordingDecisionStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        incompatible = CompletionVerifierDecision(
            verdict=CompletionVerdict.ACCEPTED,
            criterion_outcomes=(
                CompletionCriterionOutcome(
                    criterion_id="another-criterion",
                    status=CriterionOutcomeStatus.SATISFIED,
                    reason_code="package.ready",
                    satisfaction_basis=CompletionSatisfactionBasis.VERIFIER_ASSERTION,
                ),
            ),
        )
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, RecordingVerifier(incompatible))

        with pytest.raises(WorkCompletionConflict, match="frozen contract"):
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert store.decision_writes == 0
        assert await store.load_completion_decision("decision-1") is None

    asyncio.run(scenario())


def test_store_returned_decision_content_must_match_its_integrity_evidence() -> None:
    class AlteredReturnStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def record_completion_decision(self, request):
            durable = await super().record_completion_decision(request)
            altered_outcome = durable.criterion_outcomes[0].model_copy(
                update={"summary": "content not present in the durable publication"}
            )
            return durable.model_copy(update={"criterion_outcomes": (altered_outcome,)})

    async def scenario() -> None:
        store = AlteredReturnStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)

        with pytest.raises(WorkCompletionConflict, match="exact publication"):
            await app.verify_completion_proposal(_execution_request(proposal_id))
        durable = await store.load_completion_decision("decision-1")
        assert durable is not None
        assert durable.criterion_outcomes[0].summary is None
        assert len(verifier.requests) == 1

    asyncio.run(scenario())


def test_corrupt_durable_proposal_integrity_fails_before_claim_or_adapter() -> None:
    class CorruptProposalStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def load_completion_proposal(self, proposal_id):
            proposal = await super().load_completion_proposal(proposal_id)
            if proposal is None:
                return None
            return proposal.model_copy(update={"request_sha256": _digest("corrupt-proposal")})

    async def scenario() -> None:
        store = CorruptProposalStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)

        with pytest.raises(WorkCompletionConflict, match="proposal.*integrity"):
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert verifier.requests == []
        assert await store.load_completion_verification_claim(proposal_id) is None

    asyncio.run(scenario())


def test_corrupt_replayed_decision_integrity_fails_without_adapter_rerun() -> None:
    class CorruptDecisionStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        corrupt_decision = False

        async def load_completion_decision(self, decision_id):
            decision = await super().load_completion_decision(decision_id)
            if decision is None or not self.corrupt_decision:
                return decision
            return decision.model_copy(update={"gap_fingerprint": _digest("corrupt-decision")})

    async def scenario() -> None:
        store = CorruptDecisionStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        request = _execution_request(proposal_id)
        await app.verify_completion_proposal(request)

        store.corrupt_decision = True
        restarted = CayuApp(task_store=store, enable_logging=False)
        with pytest.raises(WorkCompletionConflict, match="conflicting authority"):
            await restarted.verify_completion_proposal(request)
        assert len(verifier.requests) == 1

    asyncio.run(scenario())


def test_replayed_decision_requires_its_exact_final_claim_snapshot() -> None:
    class CorruptClaimStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        corrupt_claim = False

        async def load_completion_verification_claim(self, proposal_id):
            claim = await super().load_completion_verification_claim(proposal_id)
            if claim is None or not self.corrupt_claim:
                return claim
            return claim.model_copy(update={"attempt_number": claim.attempt_number + 1})

    async def scenario() -> None:
        store = CorruptClaimStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        request = _execution_request(proposal_id)
        await app.verify_completion_proposal(request)

        store.corrupt_claim = True
        restarted = CayuApp(task_store=store, enable_logging=False)
        with pytest.raises(WorkCompletionConflict, match="conflicting authority"):
            await restarted.verify_completion_proposal(request)
        assert len(verifier.requests) == 1

    asyncio.run(scenario())


def test_adapter_failure_is_detached_and_secret_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "completion-verifier-adapter-secret-canary"

    class FailingVerifier(_TestCompletionVerifier):
        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            raise RuntimeError(f"adapter exposed {secret}")

    async def scenario() -> BaseException:
        store = InMemoryTaskStore()
        contract = _contract(objective=f"Private objective containing {secret}")
        proposal_id = await _proposal(store, contract)
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_completion_verifier(contract.verifier, FailingVerifier())

        with pytest.raises(CompletionVerifierExecutionError) as exc:
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert await store.load_completion_decision("decision-1") is None
        return exc.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_adapter_failure_final_composition_is_bounded_and_split_secret_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "CompletionVerifierExecutionError('RuntimeError: token"

    class FailingVerifier(_TestCompletionVerifier):
        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            raise RuntimeError("token" + ("x" * 8_192))

    async def scenario() -> BaseException:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_completion_verifier(contract.verifier, FailingVerifier())

        with pytest.raises(CompletionVerifierExecutionError) as exc:
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert await store.load_completion_decision("decision-1") is None
        return exc.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_cayu_error(error, secret)
    assert len(str(error).encode("utf-8")) <= MAX_DIAGNOSTIC_UTF8_BYTES
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_adapter_failure_traceback_composition_is_split_secret_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "CompletionVerifierExecutionError: RuntimeError: token"

    class FailingVerifier(_TestCompletionVerifier):
        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            raise RuntimeError("token")

    async def scenario() -> BaseException:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_completion_verifier(contract.verifier, FailingVerifier())

        with pytest.raises(CompletionVerifierExecutionError) as exc:
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert await store.load_completion_decision("decision-1") is None
        return exc.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_adapter_failure_group_is_bounded_and_final_composition_is_secret_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = (
        "ExceptionGroup('Completion verifier reported multiple failures.', "
        "[CompletionVerifierExecutionError('RuntimeError: token-0')"
    )

    class GroupingVerifier(_TestCompletionVerifier):
        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            raise ExceptionGroup(
                "adapter-owned group",
                [RuntimeError(f"token-{index}") for index in range(128)],
            )

    async def scenario() -> BaseException:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_completion_verifier(contract.verifier, GroupingVerifier())

        with pytest.raises(BaseExceptionGroup) as exc:
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert await store.load_completion_decision("decision-1") is None
        return exc.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_cayu_error(error, secret)
    assert len(_exception_leaf_types(error)) <= MAX_DIAGNOSTIC_EXCEPTION_GROUP_NODES
    assert len(str(error).encode("utf-8")) <= MAX_DIAGNOSTIC_UTF8_BYTES
    assert len(repr(error).encode("utf-8")) <= MAX_DIAGNOSTIC_UTF8_BYTES
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_adapter_failure_group_traceback_composition_is_split_secret_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "| CompletionVerifierExecutionError: RuntimeError: token"

    class GroupingVerifier(_TestCompletionVerifier):
        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            raise ExceptionGroup("adapter-owned group", [RuntimeError("token")])

    async def scenario() -> BaseException:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_completion_verifier(contract.verifier, GroupingVerifier())

        with pytest.raises(BaseExceptionGroup) as exc:
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert await store.load_completion_decision("decision-1") is None
        return exc.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_duplicate_registration_final_composition_is_split_secret_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "ValueError('Completion verifier identity is already registered.')"
    app = CayuApp(
        task_store=InMemoryTaskStore(),
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    reference = _verifier_reference()
    verifier = RecordingVerifier(_accepted_decision())
    app.register_completion_verifier(reference, verifier)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as exc:
            app.register_completion_verifier(reference, verifier)

    _assert_secret_absent_from_cayu_error(exc.value, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


@pytest.mark.parametrize(
    ("verifier_kind", "message"),
    [
        (
            CompletionVerifierKind.DETERMINISTIC,
            "The exact deterministic completion verifier required by the work contract "
            "is not registered.",
        ),
        (
            CompletionVerifierKind.PROVIDER,
            "Provider-backed completion verifiers are not supported by this runtime slice.",
        ),
    ],
)
def test_unavailable_verifier_final_composition_is_split_secret_safe(
    verifier_kind: CompletionVerifierKind,
    message: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = f"CompletionVerifierUnavailable({message!r})"

    async def scenario() -> BaseException:
        store = InMemoryTaskStore()
        contract = _contract(verifier=_verifier_reference(kind=verifier_kind))
        proposal_id = await _proposal(store, contract)
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )

        with pytest.raises(CompletionVerifierUnavailable) as captured:
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert await store.load_completion_verification_claim(proposal_id) is None
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        failure = asyncio.run(scenario())

    _assert_secret_absent_from_cayu_error(failure, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_custom_store_failure_final_composition_is_split_secret_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "RuntimeError('SplitStoreFailure: token"

    class SplitStoreFailure(Exception):
        pass

    class FailingStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def load_completion_proposal(self, proposal_id):
            del proposal_id
            raise SplitStoreFailure("token" + ("x" * 8_192))

    async def scenario() -> BaseException:
        app = CayuApp(
            task_store=FailingStore(),
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        with pytest.raises(RuntimeError) as exc:
            await app.verify_completion_proposal(_execution_request("proposal-1"))
        return exc.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_cayu_error(error, secret)
    assert len(str(error).encode("utf-8")) <= MAX_DIAGNOSTIC_UTF8_BYTES
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_custom_store_failure_traceback_composition_is_split_secret_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "RuntimeError: SplitStoreFailure: token"

    class SplitStoreFailure(Exception):
        pass

    class FailingStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def load_completion_proposal(self, proposal_id):
            del proposal_id
            raise SplitStoreFailure("token")

    async def scenario() -> BaseException:
        app = CayuApp(
            task_store=FailingStore(),
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        with pytest.raises(RuntimeError) as exc:
            await app.verify_completion_proposal(_execution_request("proposal-1"))
        return exc.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


@pytest.mark.parametrize(
    "failure_phase",
    ("decision_id", "decision_proposal", "claim_replay", "claim_renewal"),
)
def test_custom_store_repr_is_detached_from_every_verifier_store_failure_frame(
    failure_phase: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "completion-verifier-store-repr-secret"

    class FailingStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.failure_phase: str | None = None

        def __repr__(self) -> str:
            return f"FailingStore({secret})"

        async def load_completion_decision(self, decision_id):
            if self.failure_phase == "decision_id":
                raise ConnectionError("completion decision lookup failed")
            return await super().load_completion_decision(decision_id)

        async def load_completion_decision_for_proposal(self, proposal_id):
            if self.failure_phase == "decision_proposal":
                raise ConnectionError("completion proposal decision lookup failed")
            return await super().load_completion_decision_for_proposal(proposal_id)

        async def load_completion_verification_claim(self, proposal_id):
            if self.failure_phase == "claim_replay":
                raise ConnectionError("completion verification claim lookup failed")
            return await super().load_completion_verification_claim(proposal_id)

        async def renew_completion_verification_claim(self, request):
            if self.failure_phase == "claim_renewal":
                raise ConnectionError("completion verification claim renewal failed")
            return await super().renew_completion_verification_claim(request)

    async def scenario() -> BaseException:
        store = FailingStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        request = _execution_request(proposal_id)
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_completion_verifier(
            contract.verifier,
            RecordingVerifier(_accepted_decision()),
        )
        if failure_phase == "claim_replay":
            await app.verify_completion_proposal(request)
            app = CayuApp(
                task_store=store,
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
        store.failure_phase = failure_phase
        with pytest.raises(ConnectionError) as exc:
            await app.verify_completion_proposal(request)
        return exc.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_custom_store_failure_group_is_bounded_and_final_composition_is_secret_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = (
        "ExceptionGroup('Completion proposal lookup reported multiple failures.', "
        "[RuntimeError('token-0')"
    )

    class FailingStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def load_completion_proposal(self, proposal_id):
            del proposal_id
            raise ExceptionGroup(
                "store-owned group",
                [RuntimeError(f"token-{index}") for index in range(128)],
            )

    async def scenario() -> BaseException:
        app = CayuApp(
            task_store=FailingStore(),
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        with pytest.raises(BaseExceptionGroup) as exc:
            await app.verify_completion_proposal(_execution_request("proposal-1"))
        return exc.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_cayu_error(error, secret)
    assert len(_exception_leaf_types(error)) <= MAX_DIAGNOSTIC_EXCEPTION_GROUP_NODES
    assert len(str(error).encode("utf-8")) <= MAX_DIAGNOSTIC_UTF8_BYTES
    assert len(repr(error).encode("utf-8")) <= MAX_DIAGNOSTIC_UTF8_BYTES
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_rejected_public_verifier_inputs_are_detached_and_secret_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "completion-verifier-public-input-secret-canary"

    class SecretVerifier(_TestCompletionVerifier):
        def __repr__(self) -> str:
            return f"SecretVerifier({secret})"

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            return _accepted_decision()

    async def scenario() -> list[BaseException]:
        app = CayuApp(
            task_store=InMemoryTaskStore(),
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        errors: list[BaseException] = []

        reference = _verifier_reference()
        object.__setattr__(reference, "verifier_id", secret)
        with pytest.raises(ValueError) as registration_error:
            app.register_completion_verifier(reference, SecretVerifier())
        errors.append(registration_error.value)

        request = _execution_request("proposal-1")
        object.__setattr__(request, "proposal_id", secret)
        with pytest.raises(ValueError) as execution_error:
            await app.verify_completion_proposal(request)
        errors.append(execution_error.value)
        return errors

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        errors = asyncio.run(scenario())

    for error in errors:
        _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_oversized_mutated_execution_identity_fails_before_store_lookup() -> None:
    class LookupCountingStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.proposal_lookups = 0

        async def load_completion_proposal(self, proposal_id):
            self.proposal_lookups += 1
            return await super().load_completion_proposal(proposal_id)

    async def scenario() -> int:
        store = LookupCountingStore()
        app = CayuApp(task_store=store, enable_logging=False)
        request = _execution_request("proposal-1")
        object.__setattr__(request, "proposal_id", "x" * 1_000_000)

        with pytest.raises(ValueError, match="execution request is invalid"):
            await app.verify_completion_proposal(request)
        return store.proposal_lookups

    assert asyncio.run(scenario()) == 0


def test_capacity_exhaustion_precedes_claim_and_is_secret_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "completion-verifier-capacity-secret-canary"

    class CapacityVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.all_started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            self.calls += 1
            if self.calls == 64:
                self.all_started.set()
            await self.release.wait()
            return _accepted_decision()

    async def scenario() -> BaseException:
        store = InMemoryTaskStore()
        contract = _contract(objective=f"Private objective containing {secret}")
        proposal_ids = [
            await _proposal(store, contract, suffix=str(index)) for index in range(1, 66)
        ]
        verifier = CapacityVerifier()
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_completion_verifier(contract.verifier, verifier)
        owners = [
            asyncio.create_task(
                app.verify_completion_proposal(_execution_request(proposal_id, suffix=str(index)))
            )
            for index, proposal_id in enumerate(proposal_ids[:64], start=1)
        ]
        await asyncio.wait_for(verifier.all_started.wait(), timeout=5)
        with pytest.raises(CompletionVerifierExecutionError, match="capacity is exhausted") as exc:
            await app.verify_completion_proposal(_execution_request(proposal_ids[64], suffix="65"))
        assert await store.load_completion_verification_claim(proposal_ids[64]) is None
        verifier.release.set()
        decisions = await asyncio.gather(*owners)
        assert len(decisions) == 64
        return exc.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_conflicting_exact_replay_rejects_changed_claim_tuple() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        original = _execution_request(proposal_id)
        await app.verify_completion_proposal(original)

        with pytest.raises(WorkCompletionConflict, match="exact execution request"):
            await app.verify_completion_proposal(
                original.model_copy(update={"lease_seconds": original.lease_seconds + 1})
            )
        with pytest.raises(WorkCompletionConflict, match="exact execution request"):
            await app.verify_completion_proposal(
                original.model_copy(
                    update={"execution_timeout_seconds": original.execution_timeout_seconds + 1}
                )
            )

    asyncio.run(scenario())


def test_unfinished_exact_retry_rejects_changed_execution_timeout() -> None:
    class CooperativelyTimedOutVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.calls = 0
            self.finished = asyncio.Event()

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            self.calls += 1
            try:
                await asyncio.Event().wait()
            finally:
                self.finished.set()

    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = CooperativelyTimedOutVerifier()
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        original = _execution_request(
            proposal_id,
            lease_seconds=2,
            timeout_seconds=0.01,
        )

        with pytest.raises(CompletionVerifierExecutionError, match="bounded execution timeout"):
            await app.verify_completion_proposal(original)
        await verifier.finished.wait()
        await asyncio.sleep(0)

        with pytest.raises(WorkCompletionConflict, match="another request"):
            await app.verify_completion_proposal(
                original.model_copy(update={"execution_timeout_seconds": 1.0})
            )
        assert verifier.calls == 1
        assert await store.load_completion_decision(original.decision_id) is None

    asyncio.run(scenario())


def test_cross_app_drain_renews_claim_until_the_original_adapter_settles() -> None:
    class RenewalStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self, *, clock) -> None:
            super().__init__(clock=clock)
            self.renewal_count = 0
            self.background_renewed = asyncio.Event()

        async def renew_completion_verification_claim(self, request):
            renewed = await super().renew_completion_verification_claim(request)
            self.renewal_count += 1
            if self.renewal_count >= 2:
                self.background_renewed.set()
            return renewed

    class ResistantVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = asyncio.Event()
            self.calls = 0

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            self.calls += 1
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()
            self.finished.set()
            return _accepted_decision()

    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        store = RenewalStore(clock=lambda: now[0])
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = ResistantVerifier()
        first_app = CayuApp(task_store=store, enable_logging=False)
        second_app = CayuApp(task_store=store, enable_logging=False)
        first_app.register_completion_verifier(contract.verifier, verifier)
        second_app.register_completion_verifier(contract.verifier, verifier)

        with pytest.raises(CompletionVerifierExecutionError, match="bounded execution timeout"):
            await first_app.verify_completion_proposal(
                _execution_request(
                    proposal_id,
                    lease_seconds=1,
                    timeout_seconds=0.01,
                )
            )
        await verifier.cancelled.wait()
        assert store.renewal_count == 1

        # Move the durable clock close to the original expiry. The retained
        # owner renews while its cancellation-resistant adapter is draining.
        now[0] += timedelta(milliseconds=750)
        await asyncio.wait_for(store.background_renewed.wait(), timeout=1)
        now[0] += timedelta(milliseconds=500)
        replacement = _execution_request(
            proposal_id,
            suffix="2",
            lease_seconds=1,
            timeout_seconds=0.5,
        )
        with pytest.raises(
            CompletionVerificationClaimLost,
            match="another live verifier claim",
        ):
            await second_app.verify_completion_proposal(replacement)
        assert verifier.calls == 1

        verifier.release.set()
        await verifier.finished.wait()
        await asyncio.sleep(0)
        now[0] += timedelta(seconds=2)
        assert (
            await second_app.verify_completion_proposal(replacement)
        ).decision_id == "decision-2"
        assert verifier.calls == 2

    asyncio.run(scenario())


def test_grouped_live_claim_loss_keeps_its_public_classification(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = (
        "BaseExceptionGroup('Completion verifier execution and claim renewal failed.', [SystemExit"
    )

    class RenewalClaimLossStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.renewal_count = 0

        async def renew_completion_verification_claim(self, request):
            self.renewal_count += 1
            if self.renewal_count == 1:
                return await super().renew_completion_verification_claim(request)
            raise CompletionVerificationClaimLost("another live verifier claim owns renewal")

    class FatalAfterClaimLossVerifier(_TestCompletionVerifier):
        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise SystemExit("adapter stopped after live claim loss") from None

    async def scenario() -> BaseExceptionGroup:
        store = RenewalClaimLossStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_completion_verifier(contract.verifier, FatalAfterClaimLossVerifier())
        with pytest.raises(BaseExceptionGroup) as captured:
            await app.verify_completion_proposal(
                _execution_request(
                    proposal_id,
                    lease_seconds=1,
                    timeout_seconds=0.9,
                )
            )
        assert await store.load_completion_decision("decision-1") is None
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        failure = asyncio.run(scenario())

    assert _exception_leaf_types(failure) == (
        SystemExit,
        CompletionVerificationClaimLost,
    )
    _assert_secret_absent_from_cayu_error(failure, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_cross_app_takeover_after_renewal_loss_discards_the_stale_result() -> None:
    class RenewalLossStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self, *, clock) -> None:
            super().__init__(clock=clock)
            self.renewal_count = 0
            self.background_renewal_failed = asyncio.Event()

        async def renew_completion_verification_claim(self, request):
            self.renewal_count += 1
            if self.renewal_count == 2:
                self.background_renewal_failed.set()
                raise ConnectionError("claim renewal authority was lost")
            return await super().renew_completion_verification_claim(request)

    class FirstCallResistsCancellationVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.first_cancelled = asyncio.Event()
            self.first_release = asyncio.Event()
            self.first_finished = asyncio.Event()
            self.calls = 0

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            self.calls += 1
            if self.calls != 1:
                return _accepted_decision()
            while not self.first_release.is_set():
                try:
                    await self.first_release.wait()
                except asyncio.CancelledError:
                    self.first_cancelled.set()
            self.first_finished.set()
            return _accepted_decision()

    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        store = RenewalLossStore(clock=lambda: now[0])
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = FirstCallResistsCancellationVerifier()
        first_app = CayuApp(task_store=store, enable_logging=False)
        second_app = CayuApp(task_store=store, enable_logging=False)
        first_app.register_completion_verifier(contract.verifier, verifier)
        second_app.register_completion_verifier(contract.verifier, verifier)

        original = _execution_request(
            proposal_id,
            lease_seconds=1,
            timeout_seconds=0.5,
        )
        with pytest.raises(BaseExceptionGroup):
            await first_app.verify_completion_proposal(original)
        await verifier.first_cancelled.wait()
        await store.background_renewal_failed.wait()
        profile_before_takeover = await store.load_completion_verifier_profile(proposal_id)
        assert profile_before_takeover is not None
        assert verifier.calls == 1
        assert not verifier.first_finished.is_set()

        now[0] += timedelta(seconds=2)
        replacement = _execution_request(
            proposal_id,
            suffix="2",
            lease_seconds=1,
            timeout_seconds=0.5,
        )
        decision = await second_app.verify_completion_proposal(replacement)
        assert decision.decision_id == replacement.decision_id
        replacement_claim = await store.load_completion_verification_claim(proposal_id)
        assert replacement_claim is not None
        assert replacement_claim.verifier_profile_fingerprint == (
            profile_before_takeover.profile.fingerprint
        )
        assert decision.verifier_profile_fingerprint == (
            profile_before_takeover.profile.fingerprint
        )
        assert await store.load_completion_verifier_profile(proposal_id) == (
            profile_before_takeover
        )
        assert verifier.calls == 2

        verifier.first_release.set()
        await verifier.first_finished.wait()
        await asyncio.sleep(0)
        assert await store.load_completion_decision(original.decision_id) is None
        assert await store.load_completion_decision(replacement.decision_id) == decision
        assert await store.load_completion_decision_for_proposal(proposal_id) == decision

    asyncio.run(scenario())


def test_decision_publication_requires_both_store_indexes_to_converge() -> None:
    class MissingIndexStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self, missing: str) -> None:
            super().__init__()
            self.missing = missing

        async def load_completion_decision(self, decision_id):
            if self.missing == "identity":
                return None
            return await super().load_completion_decision(decision_id)

        async def load_completion_decision_for_proposal(self, proposal_id):
            if self.missing == "proposal":
                return None
            return await super().load_completion_decision_for_proposal(proposal_id)

    async def scenario(missing: str) -> None:
        store = MissingIndexStore(missing)
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        request = _execution_request(proposal_id)

        with pytest.raises(
            WorkCompletionConflict, match=f"omitted the completion decision.*{missing}"
        ):
            await app.verify_completion_proposal(request)
        assert (
            await InMemoryTaskStore.load_completion_decision(store, request.decision_id) is not None
        )
        with pytest.raises(
            WorkCompletionConflict, match=f"omitted the completion decision.*{missing}"
        ):
            await app.verify_completion_proposal(request)
        assert len(verifier.requests) == 1

    for missing_index in ("identity", "proposal"):
        asyncio.run(scenario(missing_index))


class _CountingMapping(Mapping[str, object]):
    def __init__(self, size: int) -> None:
        self._keys = tuple(f"field-{index}" for index in range(size))
        self.iterated = 0

    def __getitem__(self, key: str) -> object:
        if key not in self._keys:
            raise KeyError(key)
        return object()

    def __iter__(self) -> Iterator[str]:
        for key in self._keys:
            self.iterated += 1
            yield key

    def __len__(self) -> int:
        return len(self._keys)


def test_raw_nested_adapter_outcome_is_field_bounded_before_copying() -> None:
    class MalformedVerifier(_TestCompletionVerifier):
        def __init__(self, outcome: CompletionVerifierDecision) -> None:
            self.outcome = outcome

        async def verify(self, request):
            del request
            return self.outcome

    async def scenario() -> int:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        mapping = _CountingMapping(7)
        malformed = _accepted_decision()
        object.__setattr__(malformed, "criterion_outcomes", (mapping,))
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, MalformedVerifier(malformed))

        with pytest.raises(CompletionVerifierExecutionError, match="invalid decision"):
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert await store.load_completion_decision("decision-1") is None
        return mapping.iterated

    assert asyncio.run(scenario()) == 7


def test_store_returned_attempt_and_claim_are_field_bounded_before_copying() -> None:
    class MalformedAttemptStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.mapping = _CountingMapping(4)

        async def load_work_attempt(self, attempt_id):
            attempt = await super().load_work_attempt(attempt_id)
            if attempt is not None:
                object.__setattr__(attempt, "contract", self.mapping)
            return attempt

    class MalformedClaimStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.mapping = _CountingMapping(5)

        async def claim_completion_verification(self, request):
            claim = await super().claim_completion_verification(request)
            object.__setattr__(claim, "verifier", self.mapping)
            return claim

    async def attempt_scenario() -> int:
        store = MalformedAttemptStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        with pytest.raises(WorkCompletionConflict, match="attempt.*authority"):
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert verifier.requests == []
        return store.mapping.iterated

    async def claim_scenario() -> int:
        store = MalformedClaimStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, verifier)
        with pytest.raises(WorkCompletionConflict, match="invalid completion verification claim"):
            await app.verify_completion_proposal(_execution_request(proposal_id))
        assert verifier.requests == []
        return store.mapping.iterated

    assert asyncio.run(attempt_scenario()) == 4
    assert asyncio.run(claim_scenario()) == 5


def test_pre_owner_claim_decision_replays_without_process_local_registration() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        request = _execution_request(proposal_id)
        verifier_profile = await prepare_test_completion_verifier_profile(
            store,
            proposal_id,
        )
        claim = await store.claim_completion_verification(
            CompletionVerificationClaimRequest(
                claim_id=request.claim_id,
                proposal_id=request.proposal_id,
                worker_id=request.worker_id,
                verifier=contract.verifier,
                verifier_profile_fingerprint=verifier_profile.profile.fingerprint,
                lease_seconds=request.lease_seconds,
            )
        )
        assert claim.execution_owner_id is None
        outcome = _accepted_decision()
        decision = await store.record_completion_decision(
            CompletionDecisionCreate(
                decision_id=request.decision_id,
                proposal_id=request.proposal_id,
                claim_id=request.claim_id,
                worker_id=request.worker_id,
                verifier=contract.verifier,
                verifier_profile_fingerprint=verifier_profile.profile.fingerprint,
                verdict=outcome.verdict,
                criterion_outcomes=outcome.criterion_outcomes,
            )
        )

        restarted = CayuApp(task_store=store, enable_logging=False)
        assert await restarted.verify_completion_proposal(request) == decision

    asyncio.run(scenario())


def test_completed_verifier_settles_an_inflight_claim_renewal_before_return() -> None:
    class BlockingRenewalStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.renewal_count = 0
            self.background_started = asyncio.Event()
            self.background_cancelled = asyncio.Event()
            self.cleanup_release = asyncio.Event()

        async def renew_completion_verification_claim(self, request):
            self.renewal_count += 1
            if self.renewal_count == 1:
                return await super().renew_completion_verification_claim(request)
            self.background_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.background_cancelled.set()
                await self.cleanup_release.wait()
                raise

    class CompleteDuringRenewalVerifier(_TestCompletionVerifier):
        def __init__(self, store: BlockingRenewalStore) -> None:
            self.store = store

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            await self.store.background_started.wait()
            return _accepted_decision()

    async def scenario() -> None:
        store = BlockingRenewalStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(
            contract.verifier,
            CompleteDuringRenewalVerifier(store),
        )
        owner = asyncio.create_task(
            app.verify_completion_proposal(
                _execution_request(
                    proposal_id,
                    lease_seconds=1,
                    timeout_seconds=0.9,
                )
            )
        )

        await asyncio.wait_for(store.background_cancelled.wait(), timeout=1)
        assert not owner.done()
        store.cleanup_release.set()
        assert (await owner).decision_id == "decision-1"
        await asyncio.sleep(0)
        assert not any(
            task.get_name() == "cayu-completion-verifier-claim-heartbeat"
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
        )

    asyncio.run(scenario())


def test_claim_renewal_failure_after_publication_is_observable_and_secret_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "settlement.secret.canary"

    class FailingShutdownRenewalStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.renewal_count = 0
            self.background_started = asyncio.Event()

        async def renew_completion_verification_claim(self, request):
            self.renewal_count += 1
            if self.renewal_count == 1:
                return await super().renew_completion_verification_claim(request)
            self.background_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise ConnectionError("renewal failed during heartbeat shutdown") from None

    class CompleteDuringRenewalVerifier(_TestCompletionVerifier):
        def __init__(self, store: FailingShutdownRenewalStore) -> None:
            self.store = store

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            await self.store.background_started.wait()
            return CompletionVerifierDecision(
                verdict=CompletionVerdict.ACCEPTED,
                criterion_outcomes=(
                    CompletionCriterionOutcome(
                        criterion_id="ready",
                        status=CriterionOutcomeStatus.SATISFIED,
                        reason_code=secret,
                        satisfaction_basis=CompletionSatisfactionBasis.VERIFIER_ASSERTION,
                    ),
                ),
            )

    async def scenario() -> BaseException:
        store = FailingShutdownRenewalStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_completion_verifier(
            contract.verifier,
            CompleteDuringRenewalVerifier(store),
        )
        with pytest.raises(BaseExceptionGroup) as captured:
            await app.verify_completion_proposal(
                _execution_request(
                    proposal_id,
                    lease_seconds=1,
                    timeout_seconds=0.9,
                )
            )
        assert await store.load_completion_decision("decision-1") is not None
        restarted = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        reconciled = await restarted.verify_completion_proposal(
            _execution_request(
                proposal_id,
                lease_seconds=1,
                timeout_seconds=0.9,
            )
        )
        assert reconciled.criterion_outcomes[0].reason_code == secret
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        failure = asyncio.run(scenario())

    assert _exception_leaf_types(failure) == (ConnectionError,)
    _assert_secret_absent_from_cayu_error(failure, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_claim_renewal_cancellation_retains_settlement_evidence_after_publication(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "heartbeat-settlement-secret"

    class SettlementFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.renewal_count = 0
            self.background_started = asyncio.Event()

        async def renew_completion_verification_claim(self, request):
            self.renewal_count += 1
            if self.renewal_count == 1:
                return await super().renew_completion_verification_claim(request)
            self.background_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                raise cancellation from BaseExceptionGroup(
                    "Postgres mutation cancellation settlement failed.",
                    [
                        ConnectionError(secret),
                        RuntimeError("connection abort failed"),
                    ],
                )

    class CompleteDuringRenewalVerifier(_TestCompletionVerifier):
        def __init__(self, store: SettlementFailureStore) -> None:
            self.store = store

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            await self.store.background_started.wait()
            return _accepted_decision()

    async def scenario() -> BaseExceptionGroup:
        store = SettlementFailureStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        request = _execution_request(
            proposal_id,
            lease_seconds=1,
            timeout_seconds=0.9,
        )
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_completion_verifier(
            contract.verifier,
            CompleteDuringRenewalVerifier(store),
        )
        with pytest.raises(BaseExceptionGroup) as captured:
            await app.verify_completion_proposal(request)
        assert await store.load_completion_decision(request.decision_id) is not None

        restarted = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        assert await restarted.verify_completion_proposal(request) == (
            await store.load_completion_decision(request.decision_id)
        )
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        failure = asyncio.run(scenario())

    assert _exception_leaf_types(failure) == (ConnectionError, RuntimeError)
    _assert_secret_absent_from_cayu_error(failure, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_grouped_adapter_failure_prunes_only_heartbeat_cancellation() -> None:
    class RenewalFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.renewal_count = 0

        async def renew_completion_verification_claim(self, request):
            self.renewal_count += 1
            if self.renewal_count == 1:
                return await super().renew_completion_verification_claim(request)
            raise ConnectionError("grouped claim renewal failure")

    class GroupingVerifier(_TestCompletionVerifier):
        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                raise BaseExceptionGroup(
                    "adapter grouped heartbeat cancellation",
                    [cancellation, SystemExit("independent adapter fatal signal")],
                ) from None

    async def scenario() -> BaseException:
        store = RenewalFailureStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(task_store=store, enable_logging=False)
        app.register_completion_verifier(contract.verifier, GroupingVerifier())
        with pytest.raises(BaseExceptionGroup) as captured:
            await app.verify_completion_proposal(
                _execution_request(
                    proposal_id,
                    lease_seconds=1,
                    timeout_seconds=0.9,
                )
            )
        assert await store.load_completion_decision("decision-1") is None
        return captured.value

    failure = asyncio.run(scenario())
    assert _exception_leaf_types(failure) == (SystemExit, ConnectionError)


def test_exact_retry_converges_when_dual_index_reads_straddle_publication() -> None:
    class RacingReadStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.first_identity_read = asyncio.Event()
            self.release_first_read = asyncio.Event()
            self.blocked = False

        async def load_completion_decision(self, decision_id):
            result = await super().load_completion_decision(decision_id)
            if not self.blocked and result is None:
                self.blocked = True
                self.first_identity_read.set()
                await self.release_first_read.wait()
            return result

    async def scenario() -> None:
        store = RacingReadStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = RecordingVerifier(_accepted_decision())
        first_app = CayuApp(task_store=store, enable_logging=False)
        second_app = CayuApp(task_store=store, enable_logging=False)
        first_app.register_completion_verifier(contract.verifier, verifier)
        second_app.register_completion_verifier(contract.verifier, verifier)
        request = _execution_request(proposal_id)

        first = asyncio.create_task(first_app.verify_completion_proposal(request))
        await store.first_identity_read.wait()
        published = await second_app.verify_completion_proposal(request)
        store.release_first_read.set()
        assert await first == published
        assert len(verifier.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("signal_type", "secret"),
    [
        (SystemExit, "SystemExit('token')"),
        (KeyboardInterrupt, "KeyboardInterrupt('token')"),
    ],
)
def test_fatal_verifier_wrapper_cannot_reconstruct_a_registered_secret(
    signal_type: type[BaseException],
    secret: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FatalVerifier(_TestCompletionVerifier):
        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            raise signal_type("token")

    async def scenario() -> BaseException:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_completion_verifier(contract.verifier, FatalVerifier())
        with pytest.raises(signal_type) as captured:
            await app.verify_completion_proposal(_execution_request(proposal_id))
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        failure = asyncio.run(scenario())
    _assert_secret_absent_from_cayu_error(failure, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_caller_cancellation_wrapper_cannot_reconstruct_a_registered_secret(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "CancelledError('token')"

    class BlockingVerifier(_TestCompletionVerifier):
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            del request
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def scenario() -> BaseException:
        store = InMemoryTaskStore()
        contract = _contract()
        proposal_id = await _proposal(store, contract)
        verifier = BlockingVerifier()
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_completion_verifier(contract.verifier, verifier)
        owner = asyncio.create_task(app.verify_completion_proposal(_execution_request(proposal_id)))
        await verifier.started.wait()
        owner.cancel("token")
        with pytest.raises(asyncio.CancelledError) as captured:
            await owner
        assert owner.cancelled()
        assert owner.cancelling() == 1
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        failure = asyncio.run(scenario())
    _assert_secret_absent_from_cayu_error(failure, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)
