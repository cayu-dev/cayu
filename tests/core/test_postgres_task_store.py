"""Postgres TaskStore parity tests.

Mirror the conformance assertions in ``test_task_store.py`` against a real
Dockerized Postgres. They skip automatically when Docker is unavailable.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal
from uuid import uuid4

import pytest
from psycopg import errors as psycopg_errors
from pydantic import ValidationError
from tests.core.completion_verifier_profile_fixtures import (
    prepare_test_completion_verifier_profile,
)
from tests.core.task_invocation_fixtures import (
    task_backed_session_invocation,
    unattributed_session_invocation_binding,
)
from tests.core.task_store_conformance import (
    assert_task_claim_lost_conformance,
    assert_task_session_invocation_binding_conformance,
)
from tests.core.task_terminalization_conformance import (
    assert_task_terminalization_acknowledgement_conformance,
)
from tests.core.task_topology_conformance import (
    assert_task_topology_bounded_projection_conformance,
    assert_task_topology_store_conformance,
)
from tests.core.test_verified_work_contracts import (
    _accepted_decision,
    _approval_evidence,
    _artifact_evidence,
    _contract,
    _digest,
    _rejected_decision,
    _result_reference,
    _task_result,
    _verifier_profile_fingerprint,
)

from cayu import (
    CayuApp,
    CompletionContinuationPolicy,
    CompletionDecisionApplicationRequest,
    CompletionProposalCreate,
    CompletionRejectionAction,
    CompletionVerificationClaimLost,
    CompletionVerificationClaimRequest,
    CompletionVerifierDecision,
    CompletionVerifierExecutionError,
    CompletionVerifierExecutionRequest,
    CompletionVerifierProfileAdoptionDecision,
    CompletionVerifierProfilePreparationRequest,
    CompletionVerifierRequest,
    CompletionVerifierUnavailable,
    DeterministicCompletionVerifier,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileBehaviorIdentity,
    InvocationOrigin,
    InvocationOriginClaim,
    InvocationOriginTrust,
    ResolutionActor,
    ResolutionActorSource,
    Task,
    TaskClaimLost,
    TaskCompletionDecisionRequired,
    TaskCreate,
    TaskExecutionSource,
    TaskInvocation,
    TaskOrder,
    TaskQuery,
    TaskRetryAttemptDisposition,
    TaskRetryCancellationReconciliationConflict,
    TaskRetryCancellationReconciliationEvidence,
    TaskRetryCancellationReconciliationOutcome,
    TaskRetryCancellationReconciliationRejected,
    TaskRetryCancellationReconciliationRequest,
    TaskRetryPolicy,
    TaskRetrySeriesDisposition,
    TaskRetrySettlementRequest,
    TaskStatus,
    TaskTerminalizationConflict,
    TaskTerminalizationReceipt,
    TaskTerminalizationRequest,
    TaskTerminalizationRetryPolicy,
    TaskTerminalKind,
    TaskTopologyQuery,
    WorkAttemptCreate,
    WorkCompletionConflict,
    task_create_with_execution_source,
    terminalize_task_with_retry,
)
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    DurableValueError,
    extract_durable_value_error,
)
from cayu.runtime.completion_verifier_profiles import (
    build_completion_verifier_execution_profile,
    changed_completion_verifier_profile_components,
)
from cayu.runtime.sessions import InMemorySessionStore
from cayu.runtime.work_contracts import completion_verification_claim_authority_sha256

pytestmark = pytest.mark.usefixtures("postgres_dsn")

_TABLES = (
    "cayu_knowledge_embeddings",
    "cayu_knowledge_index_readiness_current",
    "cayu_knowledge_index_readiness_events",
    "cayu_task_retry_reconciliation_rejections",
    "cayu_task_retry_settlements",
    "cayu_task_terminalization_receipts",
    "cayu_completion_decision_application_receipts",
    "cayu_completion_decisions",
    "cayu_completion_verification_claims",
    "cayu_completion_verifier_profiles",
    "cayu_completion_proposals",
    "cayu_work_attempts",
    "cayu_task_session_execution_authority",
    "cayu_work_contracts",
    "cayu_recall_item_exposures",
    "cayu_context_exposures",
    "cayu_recall_receipts",
    "cayu_knowledge_change_acknowledgements",
    "cayu_knowledge_change_consumers",
    "cayu_knowledge_change_labels",
    "cayu_knowledge_change_audiences",
    "cayu_knowledge_changes",
    "cayu_knowledge_evidence",
    "cayu_knowledge_publication_receipts",
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_chunks",
    "cayu_knowledge_revisions",
    "cayu_knowledge_entries",
    "cayu_event_watcher_state",
    "cayu_budget_reservation_identities",
    "cayu_events",
    "cayu_targeted_tool_grant_uses",
    "cayu_targeted_tool_grants",
    "cayu_public_authority_aliases",
    "cayu_public_authority_alias_keys",
    "cayu_transcript_search_configuration",
    "cayu_transcript_messages",
    "cayu_session_message_queue",
    "cayu_persisted_event_side_effects",
    "cayu_mcp_manifest_baselines",
    "cayu_checkpoints",
    "cayu_session_operations",
    "cayu_tasks",
    "cayu_sessions",
    "cayu_eval_baseline_mutations",
    "cayu_eval_baselines",
    "cayu_eval_result_records",
    "cayu_eval_results",
    "cayu_eval_runs",
    "cayu_eval_cases",
    "cayu_eval_suites",
    "cayu_eval_corpora",
    "cayu_schema_migrations",
)


async def _claim_completion_verification(store, request):
    profile = await prepare_test_completion_verifier_profile(
        store,
        request.proposal_id,
    )
    assert request.verifier_profile_fingerprint == profile.profile.fingerprint
    return await store.claim_completion_verification(request)


def test_postgres_verified_work_lifecycle_survives_restart(postgres_dsn):
    async def run() -> None:
        await _truncate(postgres_dsn)
        contract = _contract(contract_id="postgres-restart-contract")
        store = _new_store(postgres_dsn)
        try:
            assert await store.publish_work_contract(contract) == contract
            task = await store.create_running_task(
                TaskCreate(
                    task_id="postgres-verified-task",
                    type="bid",
                    session_id="session:postgres:verified",
                    work_contract=contract.reference(),
                ),
                session_invocation=unattributed_session_invocation_binding(
                    "session:postgres:verified"
                ),
            )
            attempt = await store.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id="postgres-attempt-1",
                    task_id=task.id,
                    session_id="session:postgres:verified",
                    contract=contract.reference(),
                    execution_profile_fingerprint=_digest("postgres-worker-profile"),
                )
            )
            proposal = await store.submit_completion_proposal(
                CompletionProposalCreate(
                    proposal_id="postgres-proposal-1",
                    attempt_id=attempt.attempt_id,
                    result=_result_reference("postgres"),
                    evidence_references=(_artifact_evidence(), _approval_evidence()),
                )
            )
            claim_request = CompletionVerificationClaimRequest(
                claim_id="postgres-claim-1",
                proposal_id=proposal.proposal_id,
                worker_id="postgres-verifier",
                verifier=contract.verifier,
                verifier_profile_fingerprint=_verifier_profile_fingerprint(contract.verifier),
            )
            claim = await _claim_completion_verification(store, claim_request)
            verifier_profile = await store.load_completion_verifier_profile(proposal.proposal_id)
            assert verifier_profile is not None
            assert claim.verifier_profile_fingerprint == verifier_profile.profile.fingerprint
            decision_request = _rejected_decision(
                proposal_id=proposal.proposal_id,
                claim_id=claim.claim_id,
                worker_id=claim.worker_id,
            )
            decision = await store.record_completion_decision(decision_request)
            assert decision.claim_authority_sha256 == (
                completion_verification_claim_authority_sha256(claim)
            )
            with pytest.raises(TaskCompletionDecisionRequired):
                await store.complete_task(task.id, _task_result("postgres"))
            rejection_application = CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=decision.decision_id,
                idempotency_key="postgres-apply-rejected",
            )
            still_running = await store.apply_completion_decision(rejection_application)
            assert still_running.status is TaskStatus.RUNNING
        finally:
            await store.close()

        reopened = _new_store(postgres_dsn)
        try:
            assert await reopened.load_work_contract(contract.reference()) == contract
            assert await reopened.load_work_attempt(attempt.attempt_id) == attempt
            assert await reopened.load_completion_proposal(proposal.proposal_id) == proposal
            assert await reopened.load_completion_verification_claim(proposal.proposal_id) == claim
            assert await reopened.load_completion_decision(decision.decision_id) == decision
            assert (
                await reopened.load_completion_verifier_profile(proposal.proposal_id)
                == verifier_profile
            )
            assert await reopened.record_completion_decision(decision_request) == decision
            assert await reopened.apply_completion_decision(rejection_application) == still_running
            with pytest.raises(
                WorkCompletionConflict,
                match="already applied under another identity",
            ):
                await reopened.apply_completion_decision(
                    rejection_application.model_copy(
                        update={"task_id": "missing-postgres-application-task"}
                    )
                )
            attempt_two = await reopened.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id="postgres-attempt-2",
                    task_id=task.id,
                    session_id="session:postgres:verified",
                    contract=contract.reference(),
                    execution_profile_fingerprint=_digest("postgres-worker-profile"),
                )
            )
            proposal_two = await reopened.submit_completion_proposal(
                CompletionProposalCreate(
                    proposal_id="postgres-proposal-2",
                    attempt_id=attempt_two.attempt_id,
                    result=_result_reference("postgres"),
                    evidence_references=(_artifact_evidence(), _approval_evidence()),
                )
            )
            changed_profile = build_completion_verifier_execution_profile(
                verifier=contract.verifier,
                adapter_identity=ExecutionProfileBehaviorIdentity(
                    name="tests:postgres-completion-verifier-v2",
                    behavior_version="2",
                    implementation_version="1",
                ),
            )
            inexact_adoption = CompletionVerifierProfileAdoptionDecision(
                expected_profile_fingerprint=verifier_profile.profile.fingerprint,
                candidate_profile_fingerprint=changed_profile.fingerprint,
                changed_component_ids=("unrelated-component",),
                policy_identity="tests:postgres-completion-verifier-policy:v1",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
                idempotency_key="postgres-adopt-verifier-v2",
                requested_by=ResolutionActor(
                    subject="postgres-operator",
                    source=ResolutionActorSource.REQUEST,
                ),
                reason="Adopt the PostgreSQL verifier profile.",
                policy_reason="The PostgreSQL verifier transition is authorized.",
                request_sha256=_digest("postgres-adopt-verifier-v2-request"),
            )
            with pytest.raises(WorkCompletionConflict, match="exact durable adoption"):
                await reopened.prepare_completion_verifier_profile(
                    CompletionVerifierProfilePreparationRequest(
                        proposal_id=proposal_two.proposal_id,
                        task_id=proposal_two.task_id,
                        attempt_id=attempt_two.attempt_id,
                        attempt_request_sha256=attempt_two.request_sha256,
                        source_execution_profile_fingerprint=(
                            attempt_two.execution_profile_fingerprint
                        ),
                        proposal_request_sha256=proposal_two.request_sha256,
                        contract=contract.reference(),
                        profile=changed_profile,
                        expected_prior_proposal_id=proposal.proposal_id,
                        expected_prior_profile_fingerprint=(verifier_profile.profile.fingerprint),
                        adoption=inexact_adoption,
                    )
                )
            assert await reopened.load_completion_verifier_profile(proposal_two.proposal_id) is None
            claim_two = await _claim_completion_verification(
                reopened,
                CompletionVerificationClaimRequest(
                    claim_id="postgres-claim-2",
                    proposal_id=proposal_two.proposal_id,
                    worker_id="postgres-verifier",
                    verifier=contract.verifier,
                    verifier_profile_fingerprint=_verifier_profile_fingerprint(contract.verifier),
                ),
            )
            verifier_profile_two = await reopened.load_completion_verifier_profile(
                proposal_two.proposal_id
            )
            assert verifier_profile_two is not None
            assert verifier_profile_two.profile == verifier_profile.profile
            assert verifier_profile_two.expected_prior_proposal_id == proposal.proposal_id
            assert (
                verifier_profile_two.expected_prior_profile_fingerprint
                == verifier_profile.profile.fingerprint
            )
            assert claim_two.verifier_profile_fingerprint == (
                verifier_profile_two.profile.fingerprint
            )
            accepted_request = _accepted_decision(
                proposal_id=proposal_two.proposal_id,
                claim_id=claim_two.claim_id,
                worker_id=claim_two.worker_id,
            )
            accepted = await reopened.record_completion_decision(accepted_request)
            application = CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=accepted.decision_id,
                idempotency_key="postgres-apply-accepted",
                result=_task_result("postgres"),
                result_reference=proposal_two.result,
            )
            app = CayuApp(
                session_store=InMemorySessionStore(),
                task_store=reopened,
                enable_logging=False,
            )
            completed = await app.apply_completion_decision(application)
            assert completed.status is TaskStatus.COMPLETED
            assert await app.apply_completion_decision(application) == completed
            receipt = await reopened.load_completion_decision_application_receipt(
                task.id,
                application.idempotency_key,
            )
            assert receipt is not None
            assert receipt.verifier_profile_fingerprint == (accepted.verifier_profile_fingerprint)
            assert receipt.task == completed
            assert (
                await reopened.load_active_work_contract_task_for_session(
                    "session:postgres:verified"
                )
                == completed
            )
            with pytest.raises(TaskCompletionDecisionRequired):
                await reopened.admit_ordinary_session_execution("session:postgres:verified")
        finally:
            await reopened.close()

    asyncio.run(run())


def test_postgres_verifier_profile_adoption_identity_is_task_scoped(postgres_dsn):
    async def run() -> None:
        await _truncate(postgres_dsn)
        contract = _contract(
            contract_id="postgres-verifier-adoption-contract",
            continuation_policy=CompletionContinuationPolicy(
                rejection_action=CompletionRejectionAction.CONTINUE,
                max_attempts=4,
                max_repeated_gap_count=4,
            ),
        )
        store = _new_store(postgres_dsn)
        try:
            await store.publish_work_contract(contract)
            task = await store.create_running_task(
                TaskCreate(
                    task_id="postgres-verifier-adoption-task",
                    type="bid",
                    session_id="session:postgres:verifier-adoption",
                    work_contract=contract.reference(),
                ),
                session_invocation=unattributed_session_invocation_binding(
                    "session:postgres:verifier-adoption"
                ),
            )
            first_attempt = await store.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id="postgres-verifier-adoption-attempt-1",
                    task_id=task.id,
                    session_id=task.session_id or "missing-session",
                    contract=contract.reference(),
                    execution_profile_fingerprint=_digest("postgres-adoption-worker-1"),
                )
            )
            first_proposal = await store.submit_completion_proposal(
                CompletionProposalCreate(
                    proposal_id="postgres-verifier-adoption-proposal-1",
                    attempt_id=first_attempt.attempt_id,
                    result=_result_reference("postgres-adoption-1"),
                )
            )
            first_profile = await prepare_test_completion_verifier_profile(
                store,
                first_proposal.proposal_id,
                identity_name="tests:postgres-verifier-adoption-v1",
            )
            first_claim = await store.claim_completion_verification(
                CompletionVerificationClaimRequest(
                    claim_id="postgres-verifier-adoption-claim-1",
                    proposal_id=first_proposal.proposal_id,
                    worker_id="postgres-verifier-adoption-worker",
                    verifier=contract.verifier,
                    verifier_profile_fingerprint=first_profile.profile.fingerprint,
                )
            )
            first_decision = await store.record_completion_decision(
                _rejected_decision(
                    proposal_id=first_proposal.proposal_id,
                    claim_id=first_claim.claim_id,
                    worker_id=first_claim.worker_id,
                    decision_id="postgres-verifier-adoption-decision-1",
                ).model_copy(
                    update={"verifier_profile_fingerprint": first_profile.profile.fingerprint}
                )
            )
            await store.apply_completion_decision(
                CompletionDecisionApplicationRequest(
                    task_id=task.id,
                    decision_id=first_decision.decision_id,
                    idempotency_key="postgres-apply-verifier-adoption-1",
                )
            )

            second_attempt = await store.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id="postgres-verifier-adoption-attempt-2",
                    task_id=task.id,
                    session_id=task.session_id or "missing-session",
                    contract=contract.reference(),
                    execution_profile_fingerprint=_digest("postgres-adoption-worker-2"),
                )
            )
            second_proposal = await store.submit_completion_proposal(
                CompletionProposalCreate(
                    proposal_id="postgres-verifier-adoption-proposal-2",
                    attempt_id=second_attempt.attempt_id,
                    result=_result_reference("postgres-adoption-2"),
                )
            )
            second_profile_value = build_completion_verifier_execution_profile(
                verifier=contract.verifier,
                adapter_identity=ExecutionProfileBehaviorIdentity(
                    name="tests:postgres-verifier-adoption-v2",
                    behavior_version="2",
                    implementation_version="1",
                ),
            )
            adoption_key = "postgres-verifier-adoption-key"
            second_adoption = CompletionVerifierProfileAdoptionDecision(
                expected_profile_fingerprint=first_profile.profile.fingerprint,
                candidate_profile_fingerprint=second_profile_value.fingerprint,
                changed_component_ids=changed_completion_verifier_profile_components(
                    first_profile.profile,
                    second_profile_value,
                ),
                policy_identity="tests:postgres-verifier-adoption-policy:v1",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
                idempotency_key=adoption_key,
                requested_by=ResolutionActor(
                    subject="postgres-operator",
                    source=ResolutionActorSource.REQUEST,
                ),
                reason="Adopt PostgreSQL verifier profile v2.",
                policy_reason="PostgreSQL verifier profile v2 is authorized.",
                request_sha256=_digest("postgres-verifier-adoption-request-2"),
            )
            second_profile = await store.prepare_completion_verifier_profile(
                CompletionVerifierProfilePreparationRequest(
                    proposal_id=second_proposal.proposal_id,
                    task_id=second_proposal.task_id,
                    attempt_id=second_attempt.attempt_id,
                    attempt_request_sha256=second_attempt.request_sha256,
                    source_execution_profile_fingerprint=(
                        second_attempt.execution_profile_fingerprint
                    ),
                    proposal_request_sha256=second_proposal.request_sha256,
                    contract=contract.reference(),
                    profile=second_profile_value,
                    expected_prior_proposal_id=first_profile.proposal_id,
                    expected_prior_profile_fingerprint=first_profile.profile.fingerprint,
                    adoption=second_adoption,
                )
            )
            assert second_profile.adoption == second_adoption
            second_claim = await store.claim_completion_verification(
                CompletionVerificationClaimRequest(
                    claim_id="postgres-verifier-adoption-claim-2",
                    proposal_id=second_proposal.proposal_id,
                    worker_id="postgres-verifier-adoption-worker",
                    verifier=contract.verifier,
                    verifier_profile_fingerprint=second_profile.profile.fingerprint,
                )
            )
            second_decision = await store.record_completion_decision(
                _rejected_decision(
                    proposal_id=second_proposal.proposal_id,
                    claim_id=second_claim.claim_id,
                    worker_id=second_claim.worker_id,
                    decision_id="postgres-verifier-adoption-decision-2",
                ).model_copy(
                    update={"verifier_profile_fingerprint": second_profile.profile.fingerprint}
                )
            )
            await store.apply_completion_decision(
                CompletionDecisionApplicationRequest(
                    task_id=task.id,
                    decision_id=second_decision.decision_id,
                    idempotency_key="postgres-apply-verifier-adoption-2",
                )
            )
        finally:
            await store.close()

        reopened = _new_store(postgres_dsn)
        try:
            assert (
                await reopened.load_completion_verifier_profile(second_proposal.proposal_id)
                == second_profile
            )
            third_attempt = await reopened.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id="postgres-verifier-adoption-attempt-3",
                    task_id=task.id,
                    session_id=task.session_id or "missing-session",
                    contract=contract.reference(),
                    execution_profile_fingerprint=_digest("postgres-adoption-worker-3"),
                )
            )
            third_proposal = await reopened.submit_completion_proposal(
                CompletionProposalCreate(
                    proposal_id="postgres-verifier-adoption-proposal-3",
                    attempt_id=third_attempt.attempt_id,
                    result=_result_reference("postgres-adoption-3"),
                )
            )
            third_profile_value = build_completion_verifier_execution_profile(
                verifier=contract.verifier,
                adapter_identity=ExecutionProfileBehaviorIdentity(
                    name="tests:postgres-verifier-adoption-v3",
                    behavior_version="3",
                    implementation_version="1",
                ),
            )
            reused_adoption = CompletionVerifierProfileAdoptionDecision(
                expected_profile_fingerprint=second_profile.profile.fingerprint,
                candidate_profile_fingerprint=third_profile_value.fingerprint,
                changed_component_ids=changed_completion_verifier_profile_components(
                    second_profile.profile,
                    third_profile_value,
                ),
                policy_identity="tests:postgres-verifier-adoption-policy:v1",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
                idempotency_key=adoption_key,
                requested_by=second_adoption.requested_by,
                reason="Adopt PostgreSQL verifier profile v3.",
                policy_reason="PostgreSQL verifier profile v3 is authorized.",
                request_sha256=_digest("postgres-verifier-adoption-request-3"),
            )
            with pytest.raises(WorkCompletionConflict, match="idempotency key"):
                await reopened.prepare_completion_verifier_profile(
                    CompletionVerifierProfilePreparationRequest(
                        proposal_id=third_proposal.proposal_id,
                        task_id=third_proposal.task_id,
                        attempt_id=third_attempt.attempt_id,
                        attempt_request_sha256=third_attempt.request_sha256,
                        source_execution_profile_fingerprint=(
                            third_attempt.execution_profile_fingerprint
                        ),
                        proposal_request_sha256=third_proposal.request_sha256,
                        contract=contract.reference(),
                        profile=third_profile_value,
                        expected_prior_proposal_id=second_profile.proposal_id,
                        expected_prior_profile_fingerprint=second_profile.profile.fingerprint,
                        adoption=reused_adoption,
                    )
                )
            assert (
                await reopened.load_completion_verifier_profile(third_proposal.proposal_id) is None
            )
        finally:
            await reopened.close()

    asyncio.run(run())


def test_postgres_verifier_profile_restart_requires_exact_registration_for_replacement(
    postgres_dsn,
) -> None:
    class VersionedVerifier(DeterministicCompletionVerifier):
        def __init__(self, *, behavior_version: str, fail: bool = False) -> None:
            self.behavior_version = behavior_version
            self.fail = fail
            self.requests: list[CompletionVerifierRequest] = []

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:postgres-restart-completion-verifier",
                behavior_version=self.behavior_version,
                implementation_version="1",
            )

        async def verify(
            self,
            request: CompletionVerifierRequest,
        ) -> CompletionVerifierDecision:
            self.requests.append(request)
            if self.fail:
                raise RuntimeError("first postgres verifier worker failed")
            complete = _accepted_decision(
                proposal_id=request.proposal.proposal_id,
                claim_id="adapter-outcome-has-no-claim-authority",
                worker_id="adapter-outcome-has-no-worker-authority",
            )
            return CompletionVerifierDecision(
                verdict=complete.verdict,
                criterion_outcomes=complete.criterion_outcomes,
                constraint_outcomes=complete.constraint_outcomes,
                gaps=complete.gaps,
                evidence_references=complete.evidence_references,
            )

    async def run() -> None:
        await _truncate(postgres_dsn)
        clock = _MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
        contract = _contract(contract_id="postgres-profile-replacement-contract")
        store = _new_store(postgres_dsn, clock=clock)
        try:
            await store.publish_work_contract(contract)
            task = await store.create_running_task(
                TaskCreate(
                    task_id="postgres-profile-replacement-task",
                    type="verified-work",
                    session_id="session:postgres:profile-replacement",
                    work_contract=contract.reference(),
                ),
                session_invocation=unattributed_session_invocation_binding(
                    "session:postgres:profile-replacement"
                ),
            )
            attempt = await store.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id="postgres-profile-replacement-attempt",
                    task_id=task.id,
                    session_id=task.session_id or "missing-session",
                    contract=contract.reference(),
                    execution_profile_fingerprint=_digest("postgres-profile-replacement-worker"),
                )
            )
            proposal = await store.submit_completion_proposal(
                CompletionProposalCreate(
                    proposal_id="postgres-profile-replacement-proposal",
                    attempt_id=attempt.attempt_id,
                    result=_result_reference("postgres-profile-replacement"),
                )
            )
            initial = VersionedVerifier(behavior_version="v1", fail=True)
            first_app = CayuApp(task_store=store, enable_logging=False)
            first_app.register_completion_verifier(contract.verifier, initial)
            original = CompletionVerifierExecutionRequest(
                proposal_id=proposal.proposal_id,
                claim_id="postgres-profile-replacement-claim",
                decision_id="postgres-profile-replacement-decision",
                worker_id="postgres-profile-replacement-verifier",
                lease_seconds=1,
                execution_timeout_seconds=0.5,
            )
            with pytest.raises(CompletionVerifierExecutionError):
                await first_app.verify_completion_proposal(original)
            profile = await store.load_completion_verifier_profile(proposal.proposal_id)
            original_claim = await store.load_completion_verification_claim(proposal.proposal_id)
            assert profile is not None
            assert original_claim is not None
            assert original_claim.verifier_profile_fingerprint == profile.profile.fingerprint
        finally:
            await store.close()

        reopened = _new_store(postgres_dsn, clock=clock)
        try:
            missing_app = CayuApp(task_store=reopened, enable_logging=False)
            with pytest.raises(CompletionVerifierUnavailable, match="not registered"):
                await missing_app.verify_completion_proposal(original)
            assert (
                await reopened.load_completion_verification_claim(proposal.proposal_id)
                == original_claim
            )

            changed = VersionedVerifier(behavior_version="v2")
            changed_app = CayuApp(task_store=reopened, enable_logging=False)
            changed_app.register_completion_verifier(contract.verifier, changed)
            with pytest.raises(CompletionVerifierUnavailable, match="durable profile"):
                await changed_app.verify_completion_proposal(original)
            assert changed.requests == []
            assert (
                await reopened.load_completion_verification_claim(proposal.proposal_id)
                == original_claim
            )

            clock.value += timedelta(seconds=2)
            exact = VersionedVerifier(behavior_version="v1")
            exact_app = CayuApp(task_store=reopened, enable_logging=False)
            exact_app.register_completion_verifier(contract.verifier, exact)
            replacement = original.model_copy(
                update={
                    "claim_id": "postgres-profile-replacement-claim-2",
                    "decision_id": "postgres-profile-replacement-decision-2",
                }
            )
            decision = await exact_app.verify_completion_proposal(replacement)
            replacement_claim = await reopened.load_completion_verification_claim(
                proposal.proposal_id
            )
            assert replacement_claim is not None
            assert replacement_claim.attempt_number == original_claim.attempt_number + 1
            assert replacement_claim.verifier_profile_fingerprint == profile.profile.fingerprint
            assert decision.verifier_profile_fingerprint == profile.profile.fingerprint
            assert await reopened.load_completion_verifier_profile(proposal.proposal_id) == profile
            assert len(exact.requests) == 1
        finally:
            await reopened.close()

    asyncio.run(run())


def test_postgres_revision_58_rejects_unprofiled_verification_records(postgres_dsn):
    async def run() -> None:
        import psycopg

        from cayu import PostgresTaskStore
        from cayu.storage.migrations import SchemaMode

        await _truncate(postgres_dsn)
        store = _new_store(postgres_dsn)
        contract = _contract(contract_id="postgres-revision-58-contract")
        try:
            await store.publish_work_contract(contract)
            await store.create_running_task(
                TaskCreate(
                    task_id="postgres-revision-58-task",
                    type="verified-work",
                    session_id="session:postgres:revision-58",
                    work_contract=contract.reference(),
                ),
                session_invocation=unattributed_session_invocation_binding(
                    "session:postgres:revision-58"
                ),
            )
            attempt = await store.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id="postgres-revision-58-attempt",
                    task_id="postgres-revision-58-task",
                    session_id="session:postgres:revision-58",
                    contract=contract.reference(),
                    execution_profile_fingerprint=_digest("postgres-revision-58-worker"),
                )
            )
            proposal = await store.submit_completion_proposal(
                CompletionProposalCreate(
                    proposal_id="postgres-revision-58-proposal",
                    attempt_id=attempt.attempt_id,
                    result=_result_reference("postgres-revision-58"),
                )
            )
            await _claim_completion_verification(
                store,
                CompletionVerificationClaimRequest(
                    claim_id="postgres-revision-58-claim",
                    proposal_id=proposal.proposal_id,
                    worker_id="postgres-revision-58-verifier",
                    verifier=contract.verifier,
                    verifier_profile_fingerprint=_verifier_profile_fingerprint(contract.verifier),
                ),
            )
        finally:
            await store.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                with pytest.raises(psycopg.Error):
                    await cur.execute(
                        "UPDATE cayu_completion_verification_claims "
                        "SET verifier_profile_fingerprint = NULL"
                    )
                await conn.rollback()
                await cur.execute("DROP TABLE cayu_completion_verifier_profiles")
                await cur.execute(
                    "ALTER TABLE cayu_completion_verification_claims "
                    "DROP COLUMN verifier_profile_fingerprint"
                )
                await cur.execute(
                    "ALTER TABLE cayu_completion_decisions DROP COLUMN verifier_profile_fingerprint"
                )
                await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision = 58")
            await conn.commit()

        migrator = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            with pytest.raises(RuntimeError, match="cannot attribute existing completion"):
                await migrator.ensure_schema()
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("SELECT 1 FROM cayu_schema_migrations WHERE revision = 58")
            assert await cur.fetchone() is None
            await cur.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'cayu_completion_verification_claims' "
                "AND column_name = 'verifier_profile_fingerprint'"
            )
            assert await cur.fetchone() is None

    asyncio.run(run())


def test_postgres_verified_work_authority_races_are_single_winner(postgres_dsn):
    async def run() -> None:
        await _truncate(postgres_dsn)
        first = _new_store(postgres_dsn)
        second = _new_store(postgres_dsn)
        contract = _contract(contract_id="postgres-authority-race")
        try:
            await first.publish_work_contract(contract)

            async def ordinary() -> str:
                try:
                    await first.admit_ordinary_session_execution("session:postgres:race")
                except TaskCompletionDecisionRequired:
                    return "contracted"
                return "ordinary"

            async def contracted() -> str:
                try:
                    await second.create_running_task(
                        TaskCreate(
                            task_id="postgres-authority-race-task",
                            type="bid",
                            session_id="session:postgres:race",
                            work_contract=contract.reference(),
                        ),
                        session_invocation=unattributed_session_invocation_binding(
                            "session:postgres:race"
                        ),
                    )
                except WorkCompletionConflict:
                    return "ordinary"
                return "contracted"

            outcomes = await asyncio.gather(ordinary(), contracted())
            assert outcomes[0] == outcomes[1]
            task = await first.load_task("postgres-authority-race-task")
            if outcomes[0] == "ordinary":
                assert task is None
            else:
                assert task is not None
                with pytest.raises(TaskCompletionDecisionRequired):
                    await first.admit_ordinary_session_execution("session:postgres:race")

            verified = await first.create_running_task(
                TaskCreate(
                    task_id="postgres-claim-race-task",
                    type="bid",
                    session_id="session:postgres:claim-race",
                    work_contract=contract.reference(),
                ),
                session_invocation=unattributed_session_invocation_binding(
                    "session:postgres:claim-race"
                ),
            )
            attempt = await first.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id="postgres-claim-race-attempt",
                    task_id=verified.id,
                    session_id="session:postgres:claim-race",
                    contract=contract.reference(),
                    execution_profile_fingerprint=_digest("claim-race-profile"),
                )
            )
            proposal = await first.submit_completion_proposal(
                CompletionProposalCreate(
                    proposal_id="postgres-claim-race-proposal",
                    attempt_id=attempt.attempt_id,
                    result=_result_reference("claim-race"),
                    evidence_references=(_artifact_evidence(),),
                )
            )

            async def claim(store, suffix: str):
                try:
                    return await _claim_completion_verification(
                        store,
                        CompletionVerificationClaimRequest(
                            claim_id=f"postgres-claim-race-{suffix}",
                            proposal_id=proposal.proposal_id,
                            worker_id=f"verifier-{suffix}",
                            verifier=contract.verifier,
                            verifier_profile_fingerprint=_verifier_profile_fingerprint(
                                contract.verifier
                            ),
                        ),
                    )
                except CompletionVerificationClaimLost as exc:
                    return exc

            claims = await asyncio.gather(claim(first, "a"), claim(second, "b"))
            assert sum(not isinstance(result, BaseException) for result in claims) == 1
            assert (
                sum(isinstance(result, CompletionVerificationClaimLost) for result in claims) == 1
            )
        finally:
            await first.close()
            await second.close()

    asyncio.run(run())


def test_postgres_cancelled_worker_claim_aborts_close_return_pool_connection(
    postgres_dsn,
    monkeypatch,
):
    async def run() -> None:
        import psycopg
        from psycopg_pool import AsyncConnectionPool

        from cayu import PostgresTaskStore
        from cayu.storage import _postgres_verified_work as postgres_verified_work
        from cayu.storage.migrations import SchemaMode

        await _truncate(postgres_dsn)
        monkeypatch.setattr(
            postgres_verified_work,
            "_POSTGRES_MUTATION_CANCELLATION_GRACE_SECONDS",
            0.0,
        )
        pool = AsyncConnectionPool(
            postgres_dsn,
            min_size=1,
            max_size=2,
            open=False,
            close_returns=True,
        )
        await pool.open()
        store = PostgresTaskStore(pool=pool, schema_mode=SchemaMode.CREATE)
        lock_connection = await psycopg.AsyncConnection.connect(postgres_dsn)
        caller: asyncio.Task[object] | None = None
        try:
            await store.create_task(TaskCreate(task_id="claim-cancellation", type="job"))
            await lock_connection.execute("LOCK TABLE cayu_tasks IN ACCESS EXCLUSIVE MODE")
            caller = asyncio.create_task(
                store.claim_task("claim-cancellation-worker"),
                name="postgres-claim-cancellation-caller",
            )
            await asyncio.sleep(0.1)
            caller.cancel("stop waiting after dispatch")
            assert caller.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(caller), timeout=5)
            assert caller.cancelled()
            assert not any(
                task.get_name() == "cayu-postgres-store-mutation" for task in asyncio.all_tasks()
            )

            # The cancelled mutation is already quiescent while the competing
            # transaction still owns the table lock. Releasing that lock must
            # not allow a delayed stale claim to commit.
            await lock_connection.commit()
            claimed = await store.load_task("claim-cancellation")
            assert claimed is not None
            assert claimed.status is TaskStatus.PENDING
            assert claimed.worker_id is None

            retry = await store.claim_task("claim-cancellation-successor")
            assert retry is not None
            assert retry.status is TaskStatus.CLAIMED
            assert retry.worker_id == "claim-cancellation-successor"
        finally:
            await lock_connection.rollback()
            if caller is not None and not caller.done():
                caller.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await caller
            await lock_connection.close()
            await store.close()
            await pool.close()

    asyncio.run(run())


def test_postgres_rollback_failure_physically_discards_close_return_connection(
    postgres_dsn,
    monkeypatch,
):
    async def run() -> None:
        from psycopg import AsyncConnection
        from psycopg_pool import AsyncConnectionPool

        from cayu import PostgresTaskStore
        from cayu.storage.migrations import SchemaMode

        await _truncate(postgres_dsn)
        pool = AsyncConnectionPool(
            postgres_dsn,
            min_size=1,
            max_size=2,
            open=False,
            close_returns=True,
        )
        await pool.open()
        store = PostgresTaskStore(pool=pool, schema_mode=SchemaMode.CREATE)
        contract = _contract(contract_id="postgres-rollback-abort")
        primary_failure = RuntimeError("commit failed before acknowledgement")
        rollback_failure = RuntimeError("rollback outcome remained uncertain")
        failed_connections: list[AsyncConnection] = []
        return_counts: dict[int, int] = {}
        try:
            # Complete lazy schema setup before faulting the mutation transaction.
            assert await store.load_work_contract(contract.reference()) is None
            original_putconn = pool.putconn

            async def count_putconn(connection: AsyncConnection) -> None:
                identity = id(connection)
                return_counts[identity] = return_counts.get(identity, 0) + 1
                await original_putconn(connection)

            async def fail_commit(connection: AsyncConnection) -> None:
                failed_connections.append(connection)
                raise primary_failure

            async def fail_rollback(connection: AsyncConnection) -> None:
                del connection
                raise rollback_failure

            with monkeypatch.context() as faults:
                faults.setattr(pool, "putconn", count_putconn)
                faults.setattr(AsyncConnection, "commit", fail_commit)
                faults.setattr(AsyncConnection, "rollback", fail_rollback)

                with pytest.raises(BaseExceptionGroup) as captured:
                    await store.publish_work_contract(contract)

            assert captured.value.exceptions == (primary_failure, rollback_failure)
            assert len(failed_connections) == 1
            failed_connection = failed_connections[0]
            assert failed_connection.closed
            assert return_counts[id(failed_connection)] == 1

            # The failed transaction was not published, and the pool replaces
            # rather than reuses the physically revoked connection.
            assert await store.load_work_contract(contract.reference()) is None
            async with pool.connection() as successor:
                assert successor is not failed_connection
        finally:
            await store.close()
            await pool.close()

    asyncio.run(run())


def test_postgres_first_claim_cancellation_settles_lazy_schema_readiness(postgres_dsn):
    async def run() -> None:
        import psycopg

        from cayu.storage.postgres import _SCHEMA_ADVISORY_LOCK_KEY

        await _truncate(postgres_dsn)
        bootstrap = _new_store(postgres_dsn)
        try:
            await bootstrap.create_task(
                TaskCreate(task_id="claim-readiness-cancellation", type="job")
            )
        finally:
            await bootstrap.close()

        lock_connection = await psycopg.AsyncConnection.connect(postgres_dsn)
        store = _new_store(postgres_dsn)
        caller: asyncio.Task[object] | None = None
        try:
            await lock_connection.execute(
                "SELECT pg_advisory_lock(%s)",
                (_SCHEMA_ADVISORY_LOCK_KEY,),
            )
            caller = asyncio.create_task(
                store.claim_task("claim-readiness-worker"),
                name="postgres-first-claim-readiness-caller",
            )

            async def wait_until_pool_opens() -> None:
                while not store._opened:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_until_pool_opens(), timeout=5)
            await asyncio.sleep(0.05)
            assert store._schema_ready is False

            caller.cancel("stop first claim during schema readiness")
            assert caller.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(caller), timeout=5)
            assert caller.cancelled()
            assert not any(
                task.get_name() == "cayu-postgres-store-mutation" for task in asyncio.all_tasks()
            )

            await lock_connection.execute(
                "SELECT pg_advisory_unlock(%s)",
                (_SCHEMA_ADVISORY_LOCK_KEY,),
            )
            pending = await store.load_task("claim-readiness-cancellation")
            assert pending is not None
            assert pending.status is TaskStatus.PENDING
            assert pending.worker_id is None

            claimed = await store.claim_task("claim-readiness-successor")
            assert claimed is not None
            assert claimed.status is TaskStatus.CLAIMED
            assert claimed.worker_id == "claim-readiness-successor"
        finally:
            await lock_connection.execute("SELECT pg_advisory_unlock_all()")
            if caller is not None and not caller.done():
                caller.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await caller
            await lock_connection.close()
            await store.close()

    asyncio.run(run())


def test_postgres_verified_work_clock_does_not_expire_task_worker_lease(postgres_dsn):
    async def run() -> None:
        await _truncate(postgres_dsn)
        verifier_now = datetime(2100, 1, 1, tzinfo=UTC)
        store = _new_store(postgres_dsn, clock=lambda: verifier_now)
        contract = _contract(contract_id="postgres-separate-clock-domains")
        try:
            await store.publish_work_contract(contract)
            task = await store.create_task(
                TaskCreate(
                    task_id="postgres-separate-clock-task",
                    type="verified-work",
                    work_contract=contract.reference(),
                )
            )
            claimed = await store.claim_task(
                "postgres-task-worker",
                TaskQuery(type=task.type),
                lease_seconds=30,
            )
            assert claimed is not None
            running = await store.attach_task(
                task.id,
                session_id="session:postgres:separate-clock",
                session_invocation=await task_backed_session_invocation(
                    store,
                    task.id,
                    "session:postgres:separate-clock",
                ),
                worker_id="postgres-task-worker",
            )
            attempt = await store.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id="postgres-separate-clock-attempt",
                    task_id=task.id,
                    session_id=running.session_id or "missing-session",
                    contract=contract.reference(),
                    execution_profile_fingerprint=_digest("postgres-separate-clock-profile"),
                    worker_id="postgres-task-worker",
                )
            )

            assert attempt.started_at == verifier_now
            assert attempt.worker_id == "postgres-task-worker"
            assert running.updated_at < verifier_now
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_verified_work_lease_checks_use_post_lock_time(postgres_dsn):
    async def run() -> None:
        import psycopg

        await _truncate(postgres_dsn)
        store = _new_store(postgres_dsn)
        lock_connection = await psycopg.AsyncConnection.connect(postgres_dsn)
        contract = _contract(contract_id="postgres-post-lock-clock")
        attempt_caller: asyncio.Task[object] | None = None
        claim_caller: asyncio.Task[object] | None = None
        renewal_caller: asyncio.Task[object] | None = None
        task_row_renewal_caller: asyncio.Task[object] | None = None
        decision_caller: asyncio.Task[object] | None = None
        try:
            await store.publish_work_contract(contract)
            task = await store.create_task(
                TaskCreate(
                    task_id="postgres-post-lock-task-lease",
                    type="verified-work",
                    work_contract=contract.reference(),
                )
            )
            claimed = await store.claim_task("postgres-post-lock-worker", lease_seconds=1)
            assert claimed is not None
            await store.attach_task(
                task.id,
                session_id="session:postgres:post-lock-task-lease",
                session_invocation=await task_backed_session_invocation(
                    store,
                    task.id,
                    "session:postgres:post-lock-task-lease",
                ),
                worker_id="postgres-post-lock-worker",
            )
            await lock_connection.execute(
                "SELECT id FROM cayu_tasks WHERE id = %s FOR UPDATE",
                (task.id,),
            )
            attempt_caller = asyncio.create_task(
                store.begin_work_attempt(
                    WorkAttemptCreate(
                        attempt_id="postgres-post-lock-expired-attempt",
                        task_id=task.id,
                        session_id="session:postgres:post-lock-task-lease",
                        contract=contract.reference(),
                        execution_profile_fingerprint=_digest("post-lock-task-profile"),
                        worker_id="postgres-post-lock-worker",
                    )
                )
            )
            await asyncio.sleep(1.1)
            await lock_connection.commit()
            with pytest.raises(TaskClaimLost):
                await attempt_caller
            assert await store.load_work_attempt("postgres-post-lock-expired-attempt") is None

            verifier_task = await store.create_running_task(
                TaskCreate(
                    task_id="postgres-post-lock-verifier-lease",
                    type="verified-work",
                    session_id="session:postgres:post-lock-verifier-lease",
                    work_contract=contract.reference(),
                ),
                session_invocation=unattributed_session_invocation_binding(
                    "session:postgres:post-lock-verifier-lease"
                ),
            )
            attempt = await store.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id="postgres-post-lock-verifier-attempt",
                    task_id=verifier_task.id,
                    session_id=verifier_task.session_id or "missing-session",
                    contract=contract.reference(),
                    execution_profile_fingerprint=_digest("post-lock-verifier-profile"),
                )
            )
            proposal = await store.submit_completion_proposal(
                CompletionProposalCreate(
                    proposal_id="postgres-post-lock-verifier-proposal",
                    attempt_id=attempt.attempt_id,
                    result=_result_reference("post-lock-verifier"),
                    evidence_references=(_artifact_evidence(),),
                )
            )
            claim_request = CompletionVerificationClaimRequest(
                claim_id="postgres-post-lock-verifier-claim",
                proposal_id=proposal.proposal_id,
                worker_id="postgres-post-lock-verifier",
                verifier=contract.verifier,
                verifier_profile_fingerprint=_verifier_profile_fingerprint(contract.verifier),
                lease_seconds=1,
            )
            await prepare_test_completion_verifier_profile(store, proposal.proposal_id)
            await lock_connection.execute(
                "SELECT id FROM cayu_tasks WHERE id = %s FOR UPDATE",
                (verifier_task.id,),
            )
            claim_caller = asyncio.create_task(store.claim_completion_verification(claim_request))
            await asyncio.sleep(1.1)
            await lock_connection.commit()
            claim = await claim_caller
            assert claim.lease_expires_at > datetime.now(UTC)
            await lock_connection.execute(
                "SELECT claim_id FROM cayu_completion_verification_claims "
                "WHERE claim_id = %s FOR UPDATE",
                (claim.claim_id,),
            )
            renewal_caller = asyncio.create_task(
                store.renew_completion_verification_claim(claim_request)
            )
            await asyncio.sleep(1.1)
            await lock_connection.commit()
            with pytest.raises(CompletionVerificationClaimLost):
                await renewal_caller
            assert await store.load_completion_verification_claim(proposal.proposal_id) == claim

            task_row_renewal_request = claim_request.model_copy(
                update={"claim_id": "postgres-post-lock-task-row-renewal-claim"}
            )
            task_row_renewal_claim = await _claim_completion_verification(
                store, task_row_renewal_request
            )
            await lock_connection.execute(
                "SELECT id FROM cayu_tasks WHERE id = %s FOR UPDATE",
                (verifier_task.id,),
            )
            task_row_renewal_caller = asyncio.create_task(
                store.renew_completion_verification_claim(task_row_renewal_request)
            )
            await asyncio.sleep(1.1)
            await lock_connection.commit()
            with pytest.raises(CompletionVerificationClaimLost):
                await task_row_renewal_caller
            assert (
                await store.load_completion_verification_claim(proposal.proposal_id)
                == task_row_renewal_claim
            )

            decision_claim_request = claim_request.model_copy(
                update={"claim_id": "postgres-post-lock-decision-claim"}
            )
            decision_claim = await _claim_completion_verification(store, decision_claim_request)
            decision_request = _rejected_decision(
                proposal_id=proposal.proposal_id,
                claim_id=decision_claim.claim_id,
                worker_id=decision_claim.worker_id,
                decision_id="postgres-post-lock-expired-decision",
            )
            await lock_connection.execute(
                "SELECT id FROM cayu_tasks WHERE id = %s FOR UPDATE",
                (verifier_task.id,),
            )
            decision_caller = asyncio.create_task(
                store.record_completion_decision(decision_request)
            )
            await asyncio.sleep(1.1)
            await lock_connection.commit()
            with pytest.raises(CompletionVerificationClaimLost):
                await decision_caller
            assert (
                await store.load_completion_decision("postgres-post-lock-expired-decision") is None
            )
        finally:
            await lock_connection.rollback()
            for caller in (
                attempt_caller,
                claim_caller,
                renewal_caller,
                task_row_renewal_caller,
                decision_caller,
            ):
                if caller is not None and not caller.done():
                    caller.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await caller
            await lock_connection.close()
            await store.close()

    asyncio.run(run())


def test_postgres_verified_work_global_identity_races_remain_typed(postgres_dsn):
    async def run() -> None:
        await _truncate(postgres_dsn)
        first = _new_store(postgres_dsn)
        second = _new_store(postgres_dsn)
        contract = _contract(contract_id="postgres-global-identity-races")
        try:
            await first.publish_work_contract(contract)

            async def running_task(suffix: str):
                return await first.create_running_task(
                    TaskCreate(
                        task_id=f"postgres-global-identity-task-{suffix}",
                        type="verified-work",
                        session_id=f"session:postgres:global-identity:{suffix}",
                        work_contract=contract.reference(),
                    ),
                    session_invocation=unattributed_session_invocation_binding(
                        f"session:postgres:global-identity:{suffix}"
                    ),
                )

            async def unique_attempt(suffix: str):
                task = await running_task(suffix)
                attempt = await first.begin_work_attempt(
                    WorkAttemptCreate(
                        attempt_id=f"postgres-global-identity-attempt-{suffix}",
                        task_id=task.id,
                        session_id=task.session_id or "missing-session",
                        contract=contract.reference(),
                        execution_profile_fingerprint=_digest(f"profile-{suffix}"),
                    )
                )
                return task, attempt

            async def unique_proposal(suffix: str):
                task, attempt = await unique_attempt(suffix)
                proposal = await first.submit_completion_proposal(
                    CompletionProposalCreate(
                        proposal_id=f"postgres-global-identity-proposal-{suffix}",
                        attempt_id=attempt.attempt_id,
                        result=_result_reference(suffix),
                        evidence_references=(_artifact_evidence(),),
                    )
                )
                return task, proposal

            attempt_tasks = await asyncio.gather(
                running_task("attempt-a"),
                running_task("attempt-b"),
            )
            attempt_outcomes = await asyncio.gather(
                first.begin_work_attempt(
                    WorkAttemptCreate(
                        attempt_id="postgres-global-shared-attempt",
                        task_id=attempt_tasks[0].id,
                        session_id=attempt_tasks[0].session_id or "missing-session",
                        contract=contract.reference(),
                        execution_profile_fingerprint=_digest("shared-attempt-a"),
                    )
                ),
                second.begin_work_attempt(
                    WorkAttemptCreate(
                        attempt_id="postgres-global-shared-attempt",
                        task_id=attempt_tasks[1].id,
                        session_id=attempt_tasks[1].session_id or "missing-session",
                        contract=contract.reference(),
                        execution_profile_fingerprint=_digest("shared-attempt-b"),
                    )
                ),
                return_exceptions=True,
            )
            assert sum(isinstance(value, WorkCompletionConflict) for value in attempt_outcomes) == 1
            assert sum(not isinstance(value, BaseException) for value in attempt_outcomes) == 1

            proposal_chains = await asyncio.gather(
                unique_attempt("proposal-a"),
                unique_attempt("proposal-b"),
            )
            proposal_outcomes = await asyncio.gather(
                first.submit_completion_proposal(
                    CompletionProposalCreate(
                        proposal_id="postgres-global-shared-proposal",
                        attempt_id=proposal_chains[0][1].attempt_id,
                        result=_result_reference("shared-proposal-a"),
                        evidence_references=(_artifact_evidence(),),
                    )
                ),
                second.submit_completion_proposal(
                    CompletionProposalCreate(
                        proposal_id="postgres-global-shared-proposal",
                        attempt_id=proposal_chains[1][1].attempt_id,
                        result=_result_reference("shared-proposal-b"),
                        evidence_references=(_artifact_evidence(),),
                    )
                ),
                return_exceptions=True,
            )
            assert (
                sum(isinstance(value, WorkCompletionConflict) for value in proposal_outcomes) == 1
            )
            assert sum(not isinstance(value, BaseException) for value in proposal_outcomes) == 1

            claim_chains = await asyncio.gather(
                unique_proposal("claim-a"),
                unique_proposal("claim-b"),
            )
            claim_requests = tuple(
                CompletionVerificationClaimRequest(
                    claim_id="postgres-global-shared-claim",
                    proposal_id=chain[1].proposal_id,
                    worker_id=f"postgres-global-claim-worker-{index}",
                    verifier=contract.verifier,
                    verifier_profile_fingerprint=_verifier_profile_fingerprint(contract.verifier),
                )
                for index, chain in enumerate(claim_chains)
            )
            await asyncio.gather(
                prepare_test_completion_verifier_profile(first, claim_chains[0][1].proposal_id),
                prepare_test_completion_verifier_profile(second, claim_chains[1][1].proposal_id),
            )
            claim_outcomes = await asyncio.gather(
                first.claim_completion_verification(claim_requests[0]),
                second.claim_completion_verification(claim_requests[1]),
                return_exceptions=True,
            )
            assert sum(isinstance(value, WorkCompletionConflict) for value in claim_outcomes) == 1
            assert sum(not isinstance(value, BaseException) for value in claim_outcomes) == 1

            decision_chains = await asyncio.gather(
                unique_proposal("decision-a"),
                unique_proposal("decision-b"),
            )
            await asyncio.gather(
                prepare_test_completion_verifier_profile(first, decision_chains[0][1].proposal_id),
                prepare_test_completion_verifier_profile(second, decision_chains[1][1].proposal_id),
            )
            decision_claims = await asyncio.gather(
                first.claim_completion_verification(
                    CompletionVerificationClaimRequest(
                        claim_id="postgres-global-decision-claim-a",
                        proposal_id=decision_chains[0][1].proposal_id,
                        worker_id="postgres-global-decision-worker-a",
                        verifier=contract.verifier,
                        verifier_profile_fingerprint=_verifier_profile_fingerprint(
                            contract.verifier
                        ),
                    )
                ),
                second.claim_completion_verification(
                    CompletionVerificationClaimRequest(
                        claim_id="postgres-global-decision-claim-b",
                        proposal_id=decision_chains[1][1].proposal_id,
                        worker_id="postgres-global-decision-worker-b",
                        verifier=contract.verifier,
                        verifier_profile_fingerprint=_verifier_profile_fingerprint(
                            contract.verifier
                        ),
                    )
                ),
            )
            decision_outcomes = await asyncio.gather(
                first.record_completion_decision(
                    _rejected_decision(
                        proposal_id=decision_chains[0][1].proposal_id,
                        claim_id=decision_claims[0].claim_id,
                        worker_id=decision_claims[0].worker_id,
                        decision_id="postgres-global-shared-decision",
                    )
                ),
                second.record_completion_decision(
                    _rejected_decision(
                        proposal_id=decision_chains[1][1].proposal_id,
                        claim_id=decision_claims[1].claim_id,
                        worker_id=decision_claims[1].worker_id,
                        decision_id="postgres-global-shared-decision",
                    )
                ),
                return_exceptions=True,
            )
            assert (
                sum(isinstance(value, WorkCompletionConflict) for value in decision_outcomes) == 1
            )
            assert sum(not isinstance(value, BaseException) for value in decision_outcomes) == 1

            with pytest.raises(WorkCompletionConflict):
                await first.begin_work_attempt(
                    WorkAttemptCreate(
                        attempt_id="postgres-global-shared-attempt",
                        task_id="missing-global-identity-task",
                        session_id="session:postgres:missing-global-identity-task",
                        contract=contract.reference(),
                        execution_profile_fingerprint=_digest("missing-attempt-parent"),
                    )
                )
            with pytest.raises(WorkCompletionConflict):
                await first.submit_completion_proposal(
                    CompletionProposalCreate(
                        proposal_id="postgres-global-shared-proposal",
                        attempt_id="missing-global-identity-attempt",
                        result=_result_reference("missing-proposal-parent"),
                        evidence_references=(_artifact_evidence(),),
                    )
                )
            with pytest.raises(WorkCompletionConflict):
                # This deliberately omits the parent proposal so the store's
                # global claim-identity conflict remains the authoritative
                # failure. The normal helper prepares profile authority and
                # therefore cannot represent this malformed-parent case.
                await first.claim_completion_verification(
                    CompletionVerificationClaimRequest(
                        claim_id="postgres-global-shared-claim",
                        proposal_id="missing-global-identity-proposal",
                        worker_id="missing-global-identity-verifier",
                        verifier=contract.verifier,
                        verifier_profile_fingerprint=_verifier_profile_fingerprint(
                            contract.verifier
                        ),
                    ),
                )
            with pytest.raises(WorkCompletionConflict):
                await first.record_completion_decision(
                    _rejected_decision(
                        proposal_id="missing-global-identity-proposal",
                        claim_id="missing-global-identity-claim",
                        worker_id="missing-global-identity-verifier",
                        decision_id="postgres-global-shared-decision",
                    )
                )
        finally:
            await first.close()
            await second.close()

    asyncio.run(run())


def test_postgres_decision_application_and_claim_replay_do_not_deadlock(postgres_dsn):
    async def run() -> None:
        await _truncate(postgres_dsn)
        first = _new_store(postgres_dsn)
        second = _new_store(postgres_dsn)
        contract = _contract(contract_id="postgres-application-lock-order")
        try:
            await first.publish_work_contract(contract)
            task = await first.create_running_task(
                TaskCreate(
                    task_id="postgres-application-lock-order-task",
                    type="verified-work",
                    session_id="session:postgres:application-lock-order",
                    work_contract=contract.reference(),
                ),
                session_invocation=unattributed_session_invocation_binding(
                    "session:postgres:application-lock-order"
                ),
            )
            attempt = await first.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id="postgres-application-lock-order-attempt",
                    task_id=task.id,
                    session_id=task.session_id or "missing-session",
                    contract=contract.reference(),
                    execution_profile_fingerprint=_digest("application-lock-order-profile"),
                )
            )
            proposal = await first.submit_completion_proposal(
                CompletionProposalCreate(
                    proposal_id="postgres-application-lock-order-proposal",
                    attempt_id=attempt.attempt_id,
                    result=_result_reference("application-lock-order"),
                    evidence_references=(_artifact_evidence(),),
                )
            )
            claim_request = CompletionVerificationClaimRequest(
                claim_id="postgres-application-lock-order-claim",
                proposal_id=proposal.proposal_id,
                worker_id="postgres-application-lock-order-verifier",
                verifier=contract.verifier,
                verifier_profile_fingerprint=_verifier_profile_fingerprint(contract.verifier),
            )
            claim = await _claim_completion_verification(first, claim_request)
            decision = await first.record_completion_decision(
                _rejected_decision(
                    proposal_id=proposal.proposal_id,
                    claim_id=claim.claim_id,
                    worker_id=claim.worker_id,
                    decision_id="postgres-application-lock-order-decision",
                )
            )
            application = CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=decision.decision_id,
                idempotency_key="postgres-application-lock-order-apply",
            )

            applied, replayed_claim = await asyncio.wait_for(
                asyncio.gather(
                    first.apply_completion_decision(application),
                    second.claim_completion_verification(claim_request),
                ),
                timeout=5,
            )
            assert applied.status is TaskStatus.RUNNING
            assert replayed_claim == claim
        finally:
            await first.close()
            await second.close()

    asyncio.run(run())


def _retry_causal_budget_id(task: Task) -> str:
    assert task.retry_series is not None
    return task.retry_series.causal_budget_id


def _postgres_retry_cancellation_reconciliation_request(
    task: Task,
) -> TaskRetryCancellationReconciliationRequest:
    assert task.retry_series is not None
    assert task.worker_id is not None
    assert task.lease_expires_at is not None
    assert task.status_payload is not None
    cancellation_key = task.status_payload["settlement_idempotency_key"]
    event = task.status_payload["event"]
    assert isinstance(cancellation_key, str)
    assert isinstance(event, dict)
    requested_at = event["occurred_at"]
    assert isinstance(requested_at, str)
    # Use the store-authored cancellation timestamp so the test does not assume
    # the host and Dockerized PostgreSQL clocks are perfectly synchronized.
    reconciliation_requested_at = task.updated_at
    return TaskRetryCancellationReconciliationRequest(
        task_id=task.id,
        series_id=task.retry_series.series_id,
        attempt=task.retry_series.attempt,
        causal_budget_id=task.retry_series.causal_budget_id,
        original_worker_id=task.worker_id,
        original_lease_expires_at=task.lease_expires_at,
        cancellation_requested_at=datetime.fromisoformat(requested_at),
        cancellation_idempotency_key=cancellation_key,
        reconciliation_idempotency_key="postgres-reconciliation-1",
        reconciliation_requested_at=reconciliation_requested_at,
        reconciled_by=ResolutionActor(
            subject="operator:postgres-reconciler",
            source=ResolutionActorSource.REQUEST,
        ),
        evidence=TaskRetryCancellationReconciliationEvidence(
            outcome=TaskRetryCancellationReconciliationOutcome.EFFECT_COMPLETED,
            validator_id="postgres.effect-receipt",
            validator_version="1",
            evidence_id="effect-receipt-1",
            evidence_sha256="a" * 64,
            validated_at=reconciliation_requested_at,
            effect_fingerprint="b" * 64,
        ),
        expected_effect_fingerprint="b" * 64,
    )


def test_postgres_task_store_replays_terminalization_and_receipt(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_terminal", type="review"))
        assert await store.claim_task("worker_a") is not None
        request = TaskTerminalizationRequest(
            task_id="task_terminal",
            worker_id="worker_a",
            kind=TaskTerminalKind.COMPLETED,
            result={"summary": "done", "metrics": {"changed": 2, "checked": 4}},
            idempotency_key="terminal-attempt-1",
        )

        first = await store.terminalize_task(request)
        replayed = await store.terminalize_task(request)
        receipt = await store.load_task_terminalization_receipt(
            "task_terminal", "terminal-attempt-1"
        )

        assert replayed == first
        assert type(receipt) is TaskTerminalizationReceipt
        assert receipt.task == first
        assert receipt.worker_id == "worker_a"
        assert receipt.request_sha256 == (
            "f44314f4f13d93a708c544e83a90ecb2e2dea4d6dd7f4ceb0512b2f895d364a8"
        )

    _run(postgres_dsn, ops)


def test_postgres_task_retry_series_success_budget_and_duplicate_conformance(
    postgres_dsn,
):
    async def run() -> None:
        await _truncate(postgres_dsn)
        clock = _MutableClock(datetime(2026, 8, 19, 12, tzinfo=UTC))
        store = _new_store(postgres_dsn, clock=clock)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="retry-success",
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=2,
                        initial_backoff_seconds=0,
                        max_total_tokens=10,
                        max_estimated_cost=Decimal("1.00"),
                    ),
                )
            )
            first_claim = await store.claim_task("worker-a")
            assert first_claim is not None
            retry_request = TaskRetrySettlementRequest(
                task_id="retry-success",
                worker_id="worker-a",
                idempotency_key="attempt-one",
                causal_budget_id=_retry_causal_budget_id(first_claim),
                disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                error={"code": "temporary"},
                token_count=3,
                estimated_cost=Decimal("0.25"),
            )
            receipts = await asyncio.gather(
                *(store.settle_task_retry_attempt(retry_request) for _ in range(4))
            )
            assert all(receipt == receipts[0] for receipt in receipts)
            assert receipts[0].successor is not None
            second = await store.claim_task("worker-b")
            assert second is not None
            completed = await store.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id=second.id,
                    worker_id="worker-b",
                    idempotency_key="attempt-two",
                    causal_budget_id=_retry_causal_budget_id(second),
                    disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                    result={"ok": True},
                    token_count=2,
                    estimated_cost=Decimal("0.10"),
                )
            )
            assert completed.task.retry_series is not None
            assert completed.task.retry_series.disposition is TaskRetrySeriesDisposition.SUCCEEDED
            assert completed.task.retry_series.cumulative_tokens == 5

            await store.create_task(
                TaskCreate(
                    task_id="retry-budget",
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=3,
                        initial_backoff_seconds=0,
                        max_total_tokens=4,
                    ),
                )
            )
            budget_claim = await store.claim_task("worker-c")
            assert budget_claim is not None
            exhausted = await store.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id="retry-budget",
                    worker_id="worker-c",
                    idempotency_key="budget-attempt",
                    causal_budget_id=_retry_causal_budget_id(budget_claim),
                    disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                    error={"code": "temporary"},
                    token_count=4,
                )
            )
            assert exhausted.successor is None
            assert exhausted.task.retry_series is not None
            assert (
                exhausted.task.retry_series.disposition
                is TaskRetrySeriesDisposition.TOKENS_EXHAUSTED
            )

            await store.create_task(
                TaskCreate(
                    task_id="retry-overspend",
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=2,
                        max_total_tokens=5,
                        max_estimated_cost=Decimal("0.25"),
                    ),
                )
            )
            overspend_claim = await store.claim_task("worker-d")
            assert overspend_claim is not None
            overspend = await store.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id="retry-overspend",
                    worker_id="worker-d",
                    idempotency_key="overspend-attempt",
                    causal_budget_id=_retry_causal_budget_id(overspend_claim),
                    disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                    error={"code": "temporary"},
                    token_count=6,
                )
            )
            assert overspend.successor is None
            assert overspend.task.retry_series is not None
            assert (
                overspend.task.retry_series.disposition
                is TaskRetrySeriesDisposition.TOKENS_EXHAUSTED
            )
            assert overspend.task.retry_series.cumulative_tokens == 6
            assert overspend.task.retry_series.tokens_remaining == 0
            assert (
                await store.load_task_retry_settlement(
                    "retry-overspend",
                    "overspend-attempt",
                )
                == overspend
            )
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_task_retry_series_restart_cancellation_and_claim_race(
    postgres_dsn,
):
    async def run() -> None:
        await _truncate(postgres_dsn)
        started_at = datetime(2026, 8, 19, 13, tzinfo=UTC)
        clock = _MutableClock(started_at)
        producer = _new_store(postgres_dsn, clock=clock)
        try:
            await producer.create_task(
                TaskCreate(
                    task_id="retry-restart",
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=2,
                        initial_backoff_seconds=30,
                    ),
                )
            )
            claimed = await producer.claim_task("worker-a")
            assert claimed is not None
            retry = await producer.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id="retry-restart",
                    worker_id="worker-a",
                    idempotency_key="restart-attempt",
                    causal_budget_id=_retry_causal_budget_id(claimed),
                    disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                    error={"code": "temporary"},
                )
            )
            assert retry.successor is not None
            successor_id = retry.successor.id
        finally:
            await producer.close()

        clock.value = started_at + timedelta(seconds=30)
        first = _new_store(postgres_dsn, clock=clock)
        second = _new_store(postgres_dsn, clock=clock)
        try:
            claims = await asyncio.gather(
                first.claim_task("worker-b"),
                second.claim_task("worker-c"),
            )
            assert sum(claim is not None for claim in claims) == 1
            winner = next(claim for claim in claims if claim is not None)
            assert winner.id == successor_id
            await (first if winner.worker_id == "worker-b" else second).release_task(
                successor_id,
                winner.worker_id,
            )
            cancelled = await first.cancel_task(successor_id, {"code": "operator"})
            assert cancelled.retry_series is not None
            assert cancelled.retry_series.disposition is TaskRetrySeriesDisposition.CANCELLED
            assert cancelled.status_payload is not None
            receipt_key = cancelled.status_payload["settlement_idempotency_key"]
            assert isinstance(receipt_key, str)
            receipt = await second.load_task_retry_settlement(successor_id, receipt_key)
            assert receipt is not None
            assert receipt.task == cancelled
            assert receipt.successor is None
            assert await second.claim_task("worker-d") is None
        finally:
            await first.close()
            await second.close()

    asyncio.run(run())


def test_postgres_task_retry_active_cancellation_retains_owner_until_settlement(
    postgres_dsn,
) -> None:
    async def run() -> None:
        await _truncate(postgres_dsn)
        store = _new_store(postgres_dsn)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="retry-active-cancellation",
                    type="job",
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
            claimed = await store.claim_task("worker-a")
            assert claimed is not None
            requested = await store.cancel_task(
                claimed.id,
                {"code": "operator"},
            )
            assert requested.status is TaskStatus.CLAIMED
            assert requested.worker_id == "worker-a"
            assert requested.status_reason == "retry_cancellation_requested"
            assert requested.status_payload is not None
            key = requested.status_payload["settlement_idempotency_key"]
            assert isinstance(key, str)

            for action in (
                store.pause_task,
                store.block_task,
                store.mark_task_needs_attention,
            ):
                with pytest.raises(TaskTerminalizationConflict, match="still draining"):
                    await action(claimed.id)
                assert await store.load_task(claimed.id) == requested
            with pytest.raises(TaskTerminalizationConflict, match="still draining"):
                await store.release_task(claimed.id, "worker-a")
            assert await store.load_task(claimed.id) == requested

            receipt = await store.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id=claimed.id,
                    worker_id="worker-a",
                    idempotency_key=key,
                    causal_budget_id=_retry_causal_budget_id(claimed),
                    disposition=TaskRetryAttemptDisposition.CANCELLED,
                    error={"code": "operator"},
                )
            )
            assert receipt.task.status is TaskStatus.CANCELLED
            assert receipt.task.worker_id is None
            assert receipt.task.retry_series is not None
            assert receipt.task.retry_series.disposition is TaskRetrySeriesDisposition.CANCELLED
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_task_retry_owner_lost_cancellation_reconciliation_and_late_worker_race(
    postgres_dsn,
) -> None:
    async def run() -> None:
        await _truncate(postgres_dsn)
        reconciler = _new_store(postgres_dsn)
        late_worker = _new_store(postgres_dsn)
        try:
            await reconciler.create_task(
                TaskCreate(
                    task_id="postgres-owner-lost-reconciliation",
                    type="job",
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
            claimed = await reconciler.claim_task("lost-worker", lease_seconds=1)
            assert claimed is not None
            requested = await reconciler.cancel_task(
                claimed.id,
                {"code": "operator"},
            )
            request = _postgres_retry_cancellation_reconciliation_request(requested)
            await asyncio.sleep(1.05)

            async def settle_late_worker():
                await asyncio.sleep(0.05)
                return await late_worker.settle_task_retry_attempt(
                    TaskRetrySettlementRequest(
                        task_id=request.task_id,
                        worker_id=request.original_worker_id,
                        idempotency_key=request.cancellation_idempotency_key,
                        causal_budget_id=request.causal_budget_id,
                        disposition=TaskRetryAttemptDisposition.CANCELLED,
                        error={"code": "operator"},
                    )
                )

            reconciled, worker_result = await asyncio.gather(
                reconciler.reconcile_task_retry_cancellation(request),
                settle_late_worker(),
                return_exceptions=True,
            )
            assert not isinstance(reconciled, BaseException)
            assert isinstance(worker_result, TaskTerminalizationConflict)
            assert reconciled.task.status is TaskStatus.CANCELLED
            assert reconciled.task.retry_series is not None
            assert reconciled.task.retry_series.disposition is TaskRetrySeriesDisposition.CANCELLED
            assert reconciled.successor is None
            assert await late_worker.reconcile_task_retry_cancellation(request) == reconciled

            changed = request.model_copy(
                update={
                    "evidence": request.evidence.model_copy(update={"evidence_sha256": "c" * 64})
                }
            )
            with pytest.raises(TaskRetryCancellationReconciliationConflict):
                await late_worker.reconcile_task_retry_cancellation(changed)
            assert (
                await reconciler.load_task_retry_settlement(
                    request.task_id,
                    request.cancellation_idempotency_key,
                )
                == reconciled
            )

            await reconciler.create_task(
                TaskCreate(
                    task_id="postgres-worker-wins-reconciliation-race",
                    type="job",
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
            worker_claim = await reconciler.claim_task("live-worker", lease_seconds=5)
            assert worker_claim is not None
            worker_cancel = await reconciler.cancel_task(
                worker_claim.id,
                {"code": "operator"},
            )
            losing_reconciliation = _postgres_retry_cancellation_reconciliation_request(
                worker_cancel
            )
            worker_receipt = await late_worker.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id=worker_claim.id,
                    worker_id="live-worker",
                    idempotency_key=(losing_reconciliation.cancellation_idempotency_key),
                    causal_budget_id=losing_reconciliation.causal_budget_id,
                    disposition=TaskRetryAttemptDisposition.CANCELLED,
                    error={"code": "operator"},
                )
            )
            with pytest.raises(TaskRetryCancellationReconciliationConflict):
                await reconciler.reconcile_task_retry_cancellation(losing_reconciliation)
            assert worker_receipt.reconciliation is None
            assert (
                await reconciler.load_task_retry_settlement(
                    worker_claim.id,
                    losing_reconciliation.cancellation_idempotency_key,
                )
                == worker_receipt
            )

            await reconciler.create_task(
                TaskCreate(
                    task_id="postgres-rejected-reconciliation",
                    type="job",
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
            rejected_claim = await reconciler.claim_task(
                "rejected-worker",
                lease_seconds=5,
            )
            assert rejected_claim is not None
            rejected_cancel = await reconciler.cancel_task(
                rejected_claim.id,
                {"code": "operator"},
            )
            valid_request = _postgres_retry_cancellation_reconciliation_request(rejected_cancel)
            unsupported_request = valid_request.model_copy(
                update={
                    "evidence": valid_request.evidence.model_copy(
                        update={"outcome": (TaskRetryCancellationReconciliationOutcome.UNSUPPORTED)}
                    )
                }
            )
            with pytest.raises(TaskRetryCancellationReconciliationRejected):
                await reconciler.reconcile_task_retry_cancellation(unsupported_request)
            with pytest.raises(TaskRetryCancellationReconciliationRejected):
                await late_worker.reconcile_task_retry_cancellation(unsupported_request)
            changed_request = unsupported_request.model_copy(
                update={
                    "evidence": unsupported_request.evidence.model_copy(
                        update={"outcome": TaskRetryCancellationReconciliationOutcome.QUIESCENT}
                    )
                }
            )
            with pytest.raises(TaskRetryCancellationReconciliationConflict):
                await late_worker.reconcile_task_retry_cancellation(changed_request)
            stale_request = valid_request.model_copy(update={"original_worker_id": "stale-worker"})
            with pytest.raises(TaskRetryCancellationReconciliationConflict):
                await reconciler.reconcile_task_retry_cancellation(stale_request)
            assert await reconciler.load_task(rejected_claim.id) == rejected_cancel
        finally:
            await reconciler.close()
            await late_worker.close()

    asyncio.run(run())


def test_postgres_task_retry_worker_identity_bound_precedes_claim(postgres_dsn) -> None:
    async def run() -> None:
        await _truncate(postgres_dsn)
        store = _new_store(postgres_dsn)
        try:
            created = await store.create_task(
                TaskCreate(
                    task_id="postgres-bounded-retry-worker",
                    type="job",
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
            ordinary = await store.create_task(
                TaskCreate(task_id="postgres-ordinary-long-worker", type="job")
            )
            long_worker_id = "w" * 1025
            claimed = await store.claim_task(long_worker_id)
            assert claimed is not None
            assert claimed.id == ordinary.id
            assert await store.claim_task(long_worker_id) is None
            assert await store.load_task(created.id) == created
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_task_retry_concurrent_cancellation_is_first_writer_wins(
    postgres_dsn,
) -> None:
    async def run() -> None:
        await _truncate(postgres_dsn)
        store = _new_store(postgres_dsn)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="retry-concurrent-cancellation",
                    type="job",
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
            claimed = await store.claim_task("worker-a")
            assert claimed is not None

            first, second = await asyncio.gather(
                store.cancel_task(claimed.id, {"code": "first"}),
                store.cancel_task(claimed.id, {"code": "second"}),
            )

            assert first == second
            assert first.status_reason == "retry_cancellation_requested"
            assert first.status_payload is not None
            accepted_error = first.status_payload["error"]
            assert accepted_error in ({"code": "first"}, {"code": "second"})
            persisted = await store.load_task(claimed.id)
            assert persisted == first
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_task_retry_series_expires_before_late_claim(postgres_dsn):
    async def run() -> None:
        await _truncate(postgres_dsn)
        started_at = datetime(2026, 8, 19, 14, tzinfo=UTC)
        clock = _MutableClock(started_at)
        store = _new_store(postgres_dsn, clock=clock)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="retry-elapsed",
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=3,
                        max_elapsed_seconds=10,
                        initial_backoff_seconds=2,
                    ),
                )
            )
            claimed = await store.claim_task("worker-a")
            assert claimed is not None
            first = await store.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id="retry-elapsed",
                    worker_id="worker-a",
                    idempotency_key="elapsed-attempt-one",
                    causal_budget_id=_retry_causal_budget_id(claimed),
                    disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                    error={"code": "temporary"},
                )
            )
            assert first.successor is not None
            successor_id = first.successor.id

            clock.value = started_at + timedelta(seconds=11)
            assert await store.claim_task("late-worker") is None
            expired = await store.load_task(successor_id)
            assert expired is not None
            assert expired.status is TaskStatus.FAILED
            assert expired.retry_series is not None
            assert expired.retry_series.disposition is TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
            assert expired.status_payload is not None
            receipt_key = expired.status_payload["settlement_idempotency_key"]
            assert isinstance(receipt_key, str)
            receipt = await store.load_task_retry_settlement(successor_id, receipt_key)
            assert receipt is not None
            assert receipt.task == expired
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_task_retry_series_enforces_active_deadline_with_store_clock(postgres_dsn):
    async def run() -> None:
        await _truncate(postgres_dsn)
        started_at = datetime(2026, 8, 19, 14, tzinfo=UTC)
        clock = _MutableClock(started_at)
        store = _new_store(postgres_dsn, clock=clock)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="retry-active-elapsed",
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=2,
                        max_elapsed_seconds=10,
                    ),
                )
            )
            claimed = await store.claim_task("worker-a")
            assert claimed is not None
            assert not await store.task_retry_deadline_elapsed(claimed.id, "worker-a")
            assert await store.enforce_task_retry_deadline(claimed.id, "worker-a") is None

            clock.value = started_at + timedelta(seconds=11)
            assert await store.task_retry_deadline_elapsed(claimed.id, "worker-a")
            still_owned = await store.load_task(claimed.id)
            assert still_owned is not None
            assert still_owned.status is TaskStatus.CLAIMED
            assert still_owned.worker_id == "worker-a"
            receipt = await store.enforce_task_retry_deadline(
                claimed.id,
                "worker-a",
                token_count=17,
                estimated_cost=Decimal("4.25"),
            )
            assert receipt is not None
            assert receipt.successor is None
            assert receipt.task.status is TaskStatus.FAILED
            assert receipt.task.retry_series is not None
            assert (
                receipt.task.retry_series.disposition
                is TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
            )
            assert receipt.task.retry_series.cumulative_tokens == 17
            assert receipt.task.retry_series.cumulative_estimated_cost == Decimal("4.25")
            assert (
                await store.load_task_retry_settlement(
                    claimed.id,
                    receipt.idempotency_key,
                )
                == receipt
            )
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_task_retry_deadline_probe_rechecks_lease_after_lock_wait(postgres_dsn):
    async def run() -> None:
        import psycopg

        await _truncate(postgres_dsn)
        store = _new_store(postgres_dsn)
        try:
            await store.create_task(
                TaskCreate(
                    task_id="retry-deadline-lock-wait",
                    type="job",
                    retry_policy=TaskRetryPolicy(
                        max_attempts=2,
                        max_elapsed_seconds=0.1,
                    ),
                )
            )
            claimed = await store.claim_task("worker-a", lease_seconds=1)
            assert claimed is not None

            async with await psycopg.AsyncConnection.connect(postgres_dsn) as lock_conn:
                async with lock_conn.cursor() as lock_cur:
                    await lock_cur.execute(
                        "SELECT id FROM cayu_tasks WHERE id = %s FOR UPDATE",
                        (claimed.id,),
                    )
                    probe = asyncio.create_task(
                        store.task_retry_deadline_elapsed(claimed.id, "worker-a")
                    )
                    await asyncio.sleep(1.1)
                    assert not probe.done()
                await lock_conn.commit()

            with pytest.raises(TaskClaimLost):
                await probe
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_task_store_terminalization_acknowledgement_conformance(postgres_dsn):
    async def ops(store):
        await assert_task_terminalization_acknowledgement_conformance(store)

    _run(postgres_dsn, ops)


def test_postgres_connection_failure_subclass_is_acknowledgement_ambiguous(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_connection_failure", type="review"))
        assert await store.claim_task("worker_a") is not None
        terminalize = store.terminalize_task
        calls = 0

        async def fail_before_commit_once(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise psycopg_errors.ConnectionFailure("acknowledgement lost")
            return await terminalize(request)

        store.terminalize_task = fail_before_commit_once
        outcome = await terminalize_task_with_retry(
            store,
            TaskTerminalizationRequest(
                task_id="task_connection_failure",
                worker_id="worker_a",
                kind=TaskTerminalKind.COMPLETED,
                result={"summary": "done"},
                idempotency_key="connection-failure",
            ),
            policy=TaskTerminalizationRetryPolicy(
                max_attempts=2,
                initial_backoff_seconds=0,
                max_backoff_seconds=0,
            ),
        )

        assert outcome.attempt_count == 2
        assert calls == 2

    _run(postgres_dsn, ops)


def test_postgres_task_store_terminalization_rejects_wrong_worker_and_changed_intent(
    postgres_dsn,
):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_wrong_worker", type="review"))
        assert await store.claim_task("worker_a") is not None
        winner = TaskTerminalizationRequest(
            task_id="task_wrong_worker",
            worker_id="worker_a",
            kind=TaskTerminalKind.COMPLETED,
            result={"summary": "done"},
            idempotency_key="terminal-key",
        )
        with pytest.raises(TaskClaimLost):
            await store.terminalize_task(winner.model_copy(update={"worker_id": "worker_b"}))
        assert (
            await store.load_task_terminalization_receipt("task_wrong_worker", "terminal-key")
            is None
        )

        terminal = await store.terminalize_task(winner)
        conflicts = (
            winner.model_copy(update={"worker_id": "worker_b"}),
            winner.model_copy(update={"result": {"summary": "changed"}}),
            TaskTerminalizationRequest(
                task_id="task_wrong_worker",
                worker_id="worker_a",
                kind=TaskTerminalKind.FAILED,
                error={"message": "changed"},
                idempotency_key="terminal-key",
            ),
        )
        for conflicting in conflicts:
            with pytest.raises(TaskTerminalizationConflict):
                await store.terminalize_task(conflicting)
        assert await store.load_task("task_wrong_worker") == terminal

    _run(postgres_dsn, ops)


def test_postgres_task_store_terminalization_concurrency_converges_or_conflicts(
    postgres_dsn,
):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_exact_race", type="review"))
        assert await store.claim_task("worker_a") is not None
        exact = TaskTerminalizationRequest(
            task_id="task_exact_race",
            worker_id="worker_a",
            kind=TaskTerminalKind.COMPLETED,
            result={"summary": "done"},
            idempotency_key="race-key",
        )
        exact_results = await asyncio.gather(*(store.terminalize_task(exact) for _ in range(8)))
        assert all(result == exact_results[0] for result in exact_results)

        await store.create_task(TaskCreate(task_id="task_conflict_race", type="review"))
        assert await store.claim_task("worker_b") is not None
        requests = (
            TaskTerminalizationRequest(
                task_id="task_conflict_race",
                worker_id="worker_b",
                kind=TaskTerminalKind.COMPLETED,
                result={"winner": "completed"},
                idempotency_key="conflict-key",
            ),
            TaskTerminalizationRequest(
                task_id="task_conflict_race",
                worker_id="worker_b",
                kind=TaskTerminalKind.FAILED,
                error={"winner": "failed"},
                idempotency_key="conflict-key",
            ),
        )

        async def apply(request: TaskTerminalizationRequest):
            try:
                return await store.terminalize_task(request)
            except TaskTerminalizationConflict as exc:
                return exc

        outcomes = await asyncio.gather(*(apply(request) for request in requests))
        assert sum(type(outcome) is Task for outcome in outcomes) == 1
        assert sum(isinstance(outcome, TaskTerminalizationConflict) for outcome in outcomes) == 1

    _run(postgres_dsn, ops)


def test_postgres_task_store_replays_terminalization_after_reconstruction(postgres_dsn):
    async def run() -> None:
        await _truncate(postgres_dsn)
        request = TaskTerminalizationRequest(
            task_id="task_restart",
            worker_id="worker_a",
            kind=TaskTerminalKind.COMPLETED,
            result={"summary": "done"},
            idempotency_key="restart-key",
        )
        first_store = _new_store(postgres_dsn)
        try:
            await first_store.create_task(TaskCreate(task_id="task_restart", type="review"))
            assert await first_store.claim_task("worker_a") is not None
            terminal = await first_store.terminalize_task(request)
        finally:
            await first_store.close()

        reconstructed = _new_store(postgres_dsn)
        try:
            assert await reconstructed.terminalize_task(request) == terminal
            receipt = await reconstructed.load_task_terminalization_receipt(
                "task_restart", "restart-key"
            )
            assert receipt is not None
            assert receipt.task == terminal
        finally:
            await reconstructed.close()

    asyncio.run(run())


async def _truncate(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            for table in _TABLES:
                await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await conn.commit()


class _MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def _new_store(dsn: str, *, clock=None):
    from cayu import PostgresTaskStore
    from cayu.storage.migrations import SchemaMode

    # Tests own a throwaway database and (re)create the schema each run.
    return PostgresTaskStore(
        dsn,
        min_size=1,
        max_size=4,
        clock=clock,
        schema_mode=SchemaMode.CREATE,
    )


def _run(dsn: str, coro_factory) -> object:
    async def runner():
        await _truncate(dsn)
        store = _new_store(dsn)
        try:
            return await coro_factory(store)
        finally:
            await store.close()

    return asyncio.run(runner())


def test_postgres_task_store_task_claim_lost_conformance(postgres_dsn):
    async def ops(store):
        await assert_task_claim_lost_conformance(store)

    _run(postgres_dsn, ops)


def test_postgres_task_store_binds_session_identity_to_invocation(postgres_dsn):
    async def ops(store):
        await assert_task_session_invocation_binding_conformance(store)

    _run(postgres_dsn, ops)


def test_postgres_task_store_persists_and_inherits_invocation_provenance(postgres_dsn):
    async def ops(store):
        root = await store.create_task(
            task_create_with_execution_source(
                TaskCreate(
                    task_id="provenance-root",
                    type="webhook",
                    invocation_origin=InvocationOriginClaim(
                        subject="github:org/repo",
                        tenant="customer-a",
                    ),
                ),
                source=TaskExecutionSource.WEBHOOK,
            )
        )
        child = await store.create_task(
            TaskCreate(
                task_id="provenance-child",
                type="step",
                parent_task_id=root.id,
            )
        )
        assert root.invocation.origin.trust is InvocationOriginTrust.HOST_ASSERTED
        assert child.invocation.origin == root.invocation.origin
        assert child.invocation.root_invocation_id == root.invocation.root_invocation_id
        assert child.invocation.source is TaskExecutionSource.SDK_TASK

        snapshot = await store.load_invocation_snapshot(child.id)
        assert snapshot is not None
        assert snapshot.id == child.id
        assert snapshot.session_id == child.session_id
        assert snapshot.invocation == child.invocation
        assert await store.load_invocation_snapshot("missing") is None

        reopened = _new_store(postgres_dsn)
        try:
            loaded = await reopened.load_task(child.id)
            assert loaded is not None
            assert loaded.invocation == child.invocation
        finally:
            await reopened.close()

    _run(postgres_dsn, ops)


def test_postgres_persists_availability_and_claims_once_at_exact_boundary(
    postgres_dsn,
):
    async def run() -> None:
        await _truncate(postgres_dsn)
        boundary = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
        clock = _MutableClock(boundary - timedelta(microseconds=1))
        creator = _new_store(postgres_dsn, clock=clock)
        try:
            await creator.create_task(
                TaskCreate(
                    task_id="durable-future",
                    type="scheduled",
                    available_at=boundary,
                )
            )
            await creator.create_task(
                TaskCreate(
                    task_id="after-boundary",
                    type="scheduled",
                    available_at=boundary + timedelta(microseconds=1),
                )
            )
            before = await creator.aggregate_operational_snapshot()
            assert before.claimable_pending_count == 0
            assert before.scheduled_pending_count == 2
            assert await creator.claim_task("worker-before") is None
        finally:
            await creator.close()

        reconstructed = _new_store(postgres_dsn, clock=clock)
        try:
            loaded = await reconstructed.load_task("durable-future")
            assert loaded is not None
            assert loaded.available_at == boundary

            clock.value = boundary
            at_boundary = await reconstructed.aggregate_operational_snapshot()
            assert at_boundary.claimable_pending_count == 1
            assert at_boundary.scheduled_pending_count == 1
            claims = await asyncio.gather(
                *(reconstructed.claim_task(f"worker-{index}") for index in range(8))
            )
            winners = [claim for claim in claims if claim is not None]
            assert len(winners) == 1
            assert winners[0].id == "durable-future"
            assert winners[0].available_at == boundary

            clock.value = boundary + timedelta(microseconds=1)
            after = await reconstructed.claim_task("worker-after")
            assert after is not None
            assert after.id == "after-boundary"
        finally:
            await reconstructed.close()

    asyncio.run(run())


def test_postgres_injected_availability_clock_does_not_expire_new_task_lease(
    postgres_dsn,
):
    async def run() -> None:
        await _truncate(postgres_dsn)
        boundary = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
        store = _new_store(postgres_dsn, clock=_MutableClock(boundary))
        try:
            await store.create_task(
                TaskCreate(
                    task_id="lease-clock",
                    type="scheduled",
                    available_at=boundary,
                )
            )
            claimed = await store.claim_task("worker")
            assert claimed is not None
            released = await store.release_task("lease-clock", "worker")
            assert released.status is TaskStatus.PENDING
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_production_claim_uses_database_clock(postgres_dsn):
    async def run() -> None:
        import psycopg

        await _truncate(postgres_dsn)
        store = _new_store(postgres_dsn)
        try:
            await store.ensure_schema()
            async with (
                await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
                conn.cursor() as cur,
            ):
                await cur.execute("SELECT transaction_timestamp()")
                row = await cur.fetchone()
                assert row is not None
                database_now = row[0]

            available_at = database_now + timedelta(minutes=5)
            await store.create_task(
                TaskCreate(
                    task_id="database-clock-authority",
                    type="scheduled",
                    available_at=available_at,
                )
            )

            # Simulate a worker process whose wall clock is far ahead. The
            # production claim path must not consult it for eligibility.
            store._clock = lambda: available_at + timedelta(hours=1)
            assert await store.claim_task("clock-skewed-worker") is None
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_reclaim_ignores_a_fast_process_clock(postgres_dsn, monkeypatch):
    async def run() -> None:
        from cayu.storage import postgres as postgres_module

        await _truncate(postgres_dsn)
        store = _new_store(postgres_dsn)
        try:
            await store.create_task(TaskCreate(task_id="database-reclaim-clock", type="job"))
            claimed = await store.claim_task("worker-a", lease_seconds=300)
            assert claimed is not None

            real_datetime = datetime

            class FastProcessDatetime(datetime):
                @classmethod
                def now(cls, tz=None):
                    value = real_datetime.now(UTC) + timedelta(days=1)
                    return value if tz is None else value.astimezone(tz)

            monkeypatch.setattr(postgres_module, "datetime", FastProcessDatetime)
            assert await store.reclaim_expired(query=TaskQuery(type="job")) == []
            assert await store.load_task(claimed.id) == claimed
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_task_store_task_topology_conformance(postgres_dsn):
    async def ops(store):
        await assert_task_topology_store_conformance(store)

    _run(postgres_dsn, ops)


def test_postgres_task_topology_bounded_projection_conformance(postgres_dsn):
    async def ops(store):
        await assert_task_topology_bounded_projection_conformance(store)

    _run(postgres_dsn, ops)


def test_postgres_task_topology_uses_canonical_id_ordering(postgres_dsn):
    async def ops(store):
        import psycopg

        await store.ensure_schema()
        invocation = TaskInvocation(
            origin=InvocationOrigin(trust=InvocationOriginTrust.UNATTRIBUTED),
            root_invocation_id=str(uuid4()),
            root_session_id="collation-session",
            source=TaskExecutionSource.SDK_TASK,
        ).model_dump_json()
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO cayu_tasks (
                        id, type, status, session_id, input, metadata,
                        created_at, updated_at, invocation
                    )
                    VALUES
                        (
                            'collation-a', 'step', 'pending', 'collation-session',
                            '{}'::jsonb, '{}'::jsonb,
                            TIMESTAMPTZ '2026-01-01T00:00:00Z',
                            TIMESTAMPTZ '2026-01-01T00:00:00Z',
                            %s::jsonb
                        ),
                        (
                            'collation-B', 'step', 'pending', 'collation-session',
                            '{}'::jsonb, '{}'::jsonb,
                            TIMESTAMPTZ '2026-01-01T00:00:00Z',
                            TIMESTAMPTZ '2026-01-01T00:00:00Z',
                            %s::jsonb
                        )
                    """,
                    (invocation, invocation),
                )
            await conn.commit()

        first = await store.query_task_topology(
            TaskTopologyQuery(
                linked_session_ids=("collation-session",),
                session_task_limit=1,
            )
        )
        first_branch = first.session_branches[0]
        assert [task.id for task in first_branch.tasks] == ["collation-B"]
        assert first_branch.has_more is True
        assert first_branch.next_cursor is not None

        continuation = await store.query_task_topology(
            TaskTopologyQuery(
                linked_session_ids=("collation-session",),
                session_cursors={"collation-session": first_branch.next_cursor},
                session_task_limit=1,
            )
        )
        continuation_branch = continuation.session_branches[0]
        assert [task.id for task in continuation_branch.tasks] == ["collation-a"]
        assert continuation_branch.has_more is False
        assert continuation_branch.next_cursor is None

    _run(postgres_dsn, ops)


def test_postgres_task_topology_branch_plan_is_bounded(postgres_dsn):
    async def ops(store):
        import psycopg

        parent = await store.create_task(
            TaskCreate(
                task_id="topology-plan-parent",
                type="workflow",
                session_id="topology-plan-session",
            )
        )
        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                """
                INSERT INTO cayu_tasks (
                    id, type, status, session_id, parent_task_id,
                    input, metadata, created_at, updated_at, invocation
                )
                SELECT
                    'topology-plan-child-' || lpad(value::text, 6, '0'),
                    'step',
                    'pending',
                    'topology-plan-session',
                    'topology-plan-parent',
                    '{}'::jsonb,
                    '{}'::jsonb,
                    TIMESTAMPTZ '2026-01-01T00:00:00Z',
                    TIMESTAMPTZ '2026-01-01T00:00:00Z',
                    %s::jsonb
                FROM generate_series(0, 99999) AS value
                """,
                (parent.invocation.model_dump_json(),),
            )
            await conn.commit()
            await cur.execute("SET LOCAL enable_seqscan = off")

            async def explain(
                scope_column: Literal["session_id", "parent_task_id"],
                branch_id: str,
            ):
                await cur.execute(
                    f"""
                    EXPLAIN (ANALYZE, COSTS OFF, FORMAT JSON)
                    WITH requested_branches AS (
                        SELECT branch_id, cursor_created_at, cursor_id,
                               candidate_limit, branch_order
                        FROM unnest(
                            %s::text[],
                            %s::timestamptz[],
                            %s::text[],
                            %s::integer[]
                        ) WITH ORDINALITY AS requested(
                            branch_id,
                            cursor_created_at,
                            cursor_id,
                            candidate_limit,
                            branch_order
                        )
                    )
                    SELECT child.*
                    FROM requested_branches AS requested
                    CROSS JOIN LATERAL (
                        SELECT id, created_at
                        FROM cayu_tasks
                        WHERE cayu_tasks.{scope_column} = requested.branch_id
                          AND (
                              requested.cursor_created_at IS NULL
                              OR cayu_tasks.created_at > requested.cursor_created_at
                              OR (
                                  cayu_tasks.created_at = requested.cursor_created_at
                                  AND cayu_tasks.id COLLATE "C" >
                                      requested.cursor_id COLLATE "C"
                              )
                          )
                        ORDER BY cayu_tasks.created_at ASC,
                                 cayu_tasks.id COLLATE "C" ASC
                        LIMIT requested.candidate_limit
                    ) AS child
                    ORDER BY requested.branch_order ASC,
                             child.created_at ASC,
                             child.id COLLATE "C" ASC
                    """,
                    ([branch_id], [None], [None], [26]),
                )
                return (await cur.fetchone())[0][0]["Plan"]

            parent_plan = await explain("parent_task_id", "topology-plan-parent")
            session_plan = await explain("session_id", "topology-plan-session")

        def plan_nodes(node):
            yield node
            for child in node.get("Plans", []):
                yield from plan_nodes(child)

        for plan, index_name in (
            (parent_plan, "idx_cayu_tasks_parent_created_id"),
            (session_plan, "idx_cayu_tasks_session_created_id"),
        ):
            index_nodes = [
                node for node in plan_nodes(plan) if node.get("Index Name") == index_name
            ]
            assert index_nodes
            assert all(node["Actual Rows"] <= 26 for node in index_nodes)

    _run(postgres_dsn, ops)


def test_postgres_task_store_create_load_and_copy_boundary(postgres_dsn):
    async def ops(store):
        request_input = {"invoice_id": "inv_123", "lines": [{"amount": 25}]}
        task = await store.create_task(
            TaskCreate(
                task_id="task_invoice",
                type="process_invoice",
                title="Process invoice",
                description="Extract and post invoice fields.",
                session_id="sess_invoice",
                assigned_agent_name="invoice_agent",
                input=request_input,
                metadata={"source": "webhook"},
            )
        )
        request_input["lines"][0]["amount"] = 999

        loaded = await store.load_task("task_invoice")
        assert loaded is not None
        assert task.status == TaskStatus.PENDING
        assert loaded.input == {"invoice_id": "inv_123", "lines": [{"amount": 25}]}
        assert loaded.metadata == {"source": "webhook"}

        loaded.input["invoice_id"] = "mutated"
        loaded_again = await store.load_task("task_invoice")
        assert loaded_again is not None
        assert loaded_again.input["invoice_id"] == "inv_123"

    _run(postgres_dsn, ops)


def test_postgres_task_store_creates_running_task_atomically(postgres_dsn):
    async def ops(store):
        running = await store.create_running_task(
            TaskCreate(
                task_id="task_atomic_run",
                type="run",
                session_id="sess_atomic_run",
                input={"prompt": "hello"},
            ),
            session_invocation=unattributed_session_invocation_binding("sess_atomic_run"),
        )

        assert running.status is TaskStatus.RUNNING
        assert running.session_id == "sess_atomic_run"
        assert running.started_at is not None
        assert running.completed_at is None
        assert await store.claim_task("worker_a") is None

        with pytest.raises(ValueError, match="Task already exists"):
            await store.create_running_task(
                TaskCreate(
                    task_id="task_atomic_run",
                    type="duplicate",
                    session_id="sess_other",
                ),
                session_invocation=unattributed_session_invocation_binding("sess_other"),
            )
        with pytest.raises(ValueError, match="session_id is required"):
            await store.create_running_task(
                TaskCreate(task_id="task_missing_session", type="run"),
                session_invocation=unattributed_session_invocation_binding("sess_missing"),
            )
        assert await store.load_task("task_missing_session") is None

    _run(postgres_dsn, ops)


def test_postgres_task_store_lifecycle_and_terminal_guards(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_lifecycle", type="analyze_repository"))

        running = await store.start_task(
            "task_lifecycle",
            session_id="sess_analysis",
            session_invocation=await task_backed_session_invocation(
                store, "task_lifecycle", "sess_analysis"
            ),
        )
        assert running.status == TaskStatus.RUNNING
        assert running.session_id == "sess_analysis"
        assert running.started_at is not None
        assert running.completed_at is None

        completed = await store.complete_task("task_lifecycle", {"summary": "done"})
        assert completed.status == TaskStatus.COMPLETED
        assert completed.result == {"summary": "done"}
        assert completed.error is None
        assert completed.completed_at is not None

        with pytest.raises(ValueError, match="already terminal"):
            await store.fail_task("task_lifecycle", {"message": "too late"})

        with pytest.raises(KeyError, match="Task not found"):
            await store.start_task("missing_task")

    _run(postgres_dsn, ops)


def test_postgres_task_store_hold_resume_and_attention_states(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_blocked", type="review"))
        await store.create_task(TaskCreate(task_id="task_attention", type="review"))
        await store.create_task(TaskCreate(task_id="task_pause_claim", type="review"))

        blocked = await store.block_task(
            "task_blocked",
            reason="Waiting on vendor API",
            payload={"dependency": "vendor_api"},
        )
        assert blocked.status == TaskStatus.BLOCKED
        assert blocked.status_reason == "Waiting on vendor API"
        assert blocked.status_payload == {"dependency": "vendor_api"}

        attention = await store.mark_task_needs_attention(
            "task_attention",
            reason="Operator approval required",
            payload={"field": "amount"},
        )
        assert attention.status == TaskStatus.NEEDS_ATTENTION

        claimed = await store.claim_task(
            "worker_a",
            TaskQuery(type="review", order_by=TaskOrder.CREATED_AT_ASC),
        )
        assert claimed is not None
        assert claimed.id == "task_pause_claim"

        paused = await store.pause_task("task_pause_claim", reason="Worker shutting down")
        assert paused.status == TaskStatus.PAUSED
        assert paused.worker_id is None
        assert paused.lease_expires_at is None

        assert await store.claim_task("worker_b", TaskQuery(type="review")) is None

        resumed = await store.resume_task("task_blocked")
        assert resumed.status == TaskStatus.PENDING
        assert resumed.status_reason is None
        assert resumed.status_payload is None

        claimed_after_resume = await store.claim_task("worker_c", TaskQuery(type="review"))
        assert claimed_after_resume is not None
        assert claimed_after_resume.id == "task_blocked"

        with pytest.raises(ValueError, match="not paused, blocked, or waiting"):
            await store.resume_task("task_blocked")

        escalated = await store.block_task(
            "task_attention",
            reason="Waiting on supervisor decision",
        )
        assert escalated.status == TaskStatus.BLOCKED
        assert escalated.status_reason == "Waiting on supervisor decision"
        assert escalated.status_payload is None

    _run(postgres_dsn, ops)


def test_postgres_task_store_does_not_hold_attached_running_tasks(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_attached_hold", type="review"))
        await store.start_task(
            "task_attached_hold",
            session_id="sess_attached_hold",
            session_invocation=await task_backed_session_invocation(
                store, "task_attached_hold", "sess_attached_hold"
            ),
        )

        with pytest.raises(ValueError, match="already attached to session sess_attached_hold"):
            await store.pause_task("task_attached_hold", reason="not allowed")
        with pytest.raises(ValueError, match="already attached to session sess_attached_hold"):
            await store.block_task("task_attached_hold", reason="not allowed")
        with pytest.raises(ValueError, match="already attached to session sess_attached_hold"):
            await store.mark_task_needs_attention("task_attached_hold", reason="not allowed")

        loaded = await store.load_task("task_attached_hold")
        assert loaded is not None
        assert loaded.status == TaskStatus.RUNNING
        assert loaded.status_reason is None
        assert loaded.status_payload is None

    _run(postgres_dsn, ops)


def test_postgres_task_store_list_tasks_with_filters_and_pagination(postgres_dsn):
    async def ops(store):
        await store.create_task(
            TaskCreate(
                task_id="task_1",
                type="process_invoice",
                session_id="sess_1",
                assigned_agent_name="invoice_agent",
            )
        )
        await store.create_task(
            TaskCreate(
                task_id="task_2",
                type="process_invoice",
                session_id="sess_2",
                assigned_agent_name="invoice_agent",
            )
        )
        await store.create_task(
            TaskCreate(
                task_id="task_3",
                type="review_report",
                parent_task_id="task_2",
                assigned_agent_name="reviewer",
            )
        )
        await store.start_task(
            "task_1",
            session_invocation=await task_backed_session_invocation(
                store,
                "task_1",
                "sess_1",
            ),
        )
        await store.complete_task("task_2", {"posted": True})

        invoice_tasks = await store.list_tasks(
            TaskQuery(type="process_invoice", order_by=TaskOrder.CREATED_AT_ASC)
        )
        invoice_agent_tasks = await store.list_tasks(
            TaskQuery(assigned_agent_name="invoice_agent", order_by=TaskOrder.CREATED_AT_ASC)
        )
        completed_tasks = await store.list_tasks(TaskQuery(status=TaskStatus.COMPLETED))
        child_tasks = await store.list_tasks(TaskQuery(parent_task_id="task_2"))
        search_tasks = await store.list_tasks(
            TaskQuery(q="invoice", order_by=TaskOrder.CREATED_AT_ASC)
        )
        search_parent_tasks = await store.list_tasks(
            TaskQuery(q="TASK_2", order_by=TaskOrder.CREATED_AT_ASC)
        )
        paged_tasks = await store.list_tasks(
            TaskQuery(limit=1, offset=1, order_by=TaskOrder.CREATED_AT_ASC)
        )

        assert [t.id for t in invoice_tasks] == ["task_1", "task_2"]
        assert [t.id for t in invoice_agent_tasks] == ["task_1", "task_2"]
        assert [t.id for t in completed_tasks] == ["task_2"]
        assert [t.id for t in child_tasks] == ["task_3"]
        assert [t.id for t in search_tasks] == ["task_1", "task_2"]
        assert [t.id for t in search_parent_tasks] == ["task_2", "task_3"]
        assert [t.id for t in paged_tasks] == ["task_2"]

    _run(postgres_dsn, ops)


def test_postgres_task_store_reject_duplicate_tasks_and_invalid_payloads(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_duplicate", type="demo"))

        with pytest.raises(ValueError, match="Task already exists"):
            await store.create_task(TaskCreate(task_id="task_duplicate", type="demo"))

        with pytest.raises(ValueError, match="JSON-compatible"):
            await store.complete_task("task_duplicate", {"bad": object()})

        with pytest.raises(ValueError, match="JSON object"):
            await store.fail_task("task_duplicate", ["not", "an", "object"])  # type: ignore[arg-type]

    _run(postgres_dsn, ops)


def test_postgres_task_store_revalidates_portable_values_before_atomic_mutation(postgres_dsn):
    async def ops(store):
        poisoned_create = TaskCreate(
            task_id="task_poisoned_create",
            type="demo",
            input={"safe": True},
        )
        poisoned_create.input["bad"] = float("nan")
        with pytest.raises((DurableValueError, ValidationError)) as invalid_create:
            await store.create_task(poisoned_create)
        create_error = extract_durable_value_error(invalid_create.value)
        assert create_error is not None
        assert create_error.code == "non_finite_number"
        assert await store.load_task("task_poisoned_create") is None

        request = TaskCreate(
            task_id="task_portable_numbers",
            type="demo",
            input={"numbers": {"safe": True}},
            metadata={"numbers": {"safe": True}},
        )
        numbers = {
            "ordinary": 1.0,
            "negative_zero": -0.0,
            "large": 1e18,
            "fractional": 1e-7,
        }
        request.input["numbers"] = dict(numbers)
        request.metadata["numbers"] = dict(numbers)
        await store.create_task(request)

        with pytest.raises(DurableValueError) as invalid_result:
            await store.complete_task(
                "task_portable_numbers",
                {"bad": MAX_DURABLE_JSON_INTEGER + 1},
            )
        assert invalid_result.value.code == "integer_out_of_range"
        pending = await store.load_task("task_portable_numbers")
        assert pending is not None
        assert pending.status is TaskStatus.PENDING

        with pytest.raises(DurableValueError) as invalid_reason:
            await store.pause_task("task_portable_numbers", reason="poisoned\x00reason")
        assert invalid_reason.value.code == "nul_character"
        pending = await store.load_task("task_portable_numbers")
        assert pending is not None
        assert pending.status is TaskStatus.PENDING

        for invalid_text, code in (
            ("workload-secret-value\x00", "nul_character"),
            ("workload-secret-value\ud800", "unicode_surrogate"),
        ):
            with pytest.raises(DurableValueError) as invalid_session_id:
                await store.start_task(
                    "task_portable_numbers",
                    session_id=invalid_text,
                )
            assert invalid_session_id.value.code == code
            assert "workload-secret-value" not in str(invalid_session_id.value)

            forged_query = TaskQuery(q="safe")
            forged_query.q = invalid_text
            with pytest.raises(ValidationError) as invalid_query:
                await store.list_tasks(forged_query)
            query_error = extract_durable_value_error(invalid_query.value)
            assert query_error is not None
            assert query_error.code == code
            assert "workload-secret-value" not in str(invalid_query.value)

            pending = await store.load_task("task_portable_numbers")
            assert pending is not None
            assert pending.status is TaskStatus.PENDING
            assert pending.session_id is None

        forged_query = TaskQuery()
        forged_query.offset = MAX_DURABLE_JSON_INTEGER + 1
        with pytest.raises(ValidationError):
            await store.list_tasks(forged_query)
        pending = await store.load_task("task_portable_numbers")
        assert pending is not None
        assert pending.status is TaskStatus.PENDING

        await store.complete_task("task_portable_numbers", {"numbers": numbers})
        loaded = await store.load_task("task_portable_numbers")
        assert loaded is not None
        assert loaded.result is not None
        for value in (
            loaded.input["numbers"],
            loaded.metadata["numbers"],
            loaded.result["numbers"],
        ):
            assert value == {
                "ordinary": 1,
                "negative_zero": 0,
                "large": 1_000_000_000_000_000_000,
                "fractional": 1e-7,
            }
            assert type(value["ordinary"]) is int
            assert type(value["negative_zero"]) is int
            assert type(value["large"]) is int
            assert type(value["fractional"]) is float

    _run(postgres_dsn, ops)


def test_postgres_task_store_claim_heartbeat_and_release_task(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_a", type="review"))
        await store.create_task(TaskCreate(task_id="task_b", type="review"))
        await store.create_task(
            TaskCreate(task_id="task_session_linked", type="review", session_id="sess_linked")
        )

        first = await store.claim_task(
            "worker_a",
            TaskQuery(type="review", order_by=TaskOrder.CREATED_AT_ASC),
            lease_seconds=300,
        )
        assert first is not None
        assert first.id == "task_a"
        assert first.status == TaskStatus.CLAIMED
        assert first.worker_id == "worker_a"
        assert first.lease_expires_at is not None
        assert first.started_at is None

        second = await store.claim_task(
            "worker_b",
            TaskQuery(type="review", order_by=TaskOrder.CREATED_AT_ASC),
            lease_seconds=300,
        )
        assert second is not None
        assert second.id == "task_b"
        assert second.worker_id == "worker_b"

        assert await store.claim_task("worker_c", TaskQuery(type="review")) is None
        linked = await store.load_task("task_session_linked")
        assert linked is not None
        assert linked.status == TaskStatus.PENDING
        assert linked.worker_id is None

        heartbeat = await store.heartbeat("task_a", "worker_a", extend_seconds=600)
        assert heartbeat.lease_expires_at is not None
        assert heartbeat.lease_expires_at > first.lease_expires_at

        with pytest.raises(ValueError, match="does not own"):
            await store.heartbeat("task_a", "worker_b")

        released = await store.release_task("task_a", "worker_a")
        assert released.status == TaskStatus.PENDING
        assert released.worker_id is None
        assert released.lease_expires_at is None

        reclaimed = await store.claim_task("worker_c", TaskQuery(type="review"))
        assert reclaimed is not None
        assert reclaimed.id == "task_a"
        assert reclaimed.worker_id == "worker_c"

        completed = await store.complete_task("task_a", {"ok": True})
        assert completed.status == TaskStatus.COMPLETED
        assert completed.worker_id is None
        assert completed.lease_expires_at is None

    _run(postgres_dsn, ops)


def test_postgres_task_store_attach_task_starts_claimed_task(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_claimed", type="review"))

        with pytest.raises(ValueError, match="not claimed by worker worker_a"):
            await store.attach_task(
                "task_claimed",
                session_id="sess_unclaimed",
                session_invocation=await task_backed_session_invocation(
                    store,
                    "task_claimed",
                    "sess_unclaimed",
                ),
                worker_id="worker_a",
            )
        unclaimed = await store.load_task("task_claimed")
        assert unclaimed is not None
        assert unclaimed.status == TaskStatus.PENDING
        assert unclaimed.session_id is None

        claimed = await store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None
        assert claimed.status == TaskStatus.CLAIMED
        assert claimed.worker_id == "worker_a"
        assert claimed.session_id is None
        assert claimed.lease_expires_at is not None

        with pytest.raises(ValueError, match="session_id"):
            await store.attach_task(
                "task_claimed",
                session_id="",
                session_invocation=unattributed_session_invocation_binding("unused_session"),
                worker_id="worker_a",
            )
        with pytest.raises(ValueError, match="cannot transition to running from claimed"):
            await store.start_task("task_claimed", session_id="sess_wrong")
        with pytest.raises(ValueError, match="does not own"):
            await store.attach_task(
                "task_claimed",
                session_id="sess_wrong",
                session_invocation=await task_backed_session_invocation(
                    store,
                    "task_claimed",
                    "sess_wrong",
                ),
                worker_id="worker_b",
            )

        started = await store.attach_task(
            "task_claimed",
            session_id="sess_claimed",
            session_invocation=await task_backed_session_invocation(
                store, "task_claimed", "sess_claimed"
            ),
            worker_id="worker_a",
        )
        assert started.status == TaskStatus.RUNNING
        assert started.session_id == "sess_claimed"
        assert started.worker_id == "worker_a"
        assert started.lease_expires_at == claimed.lease_expires_at

    _run(postgres_dsn, ops)


def test_postgres_task_store_rejects_expired_claim_handoff(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_expired_handoff", type="review"))

        claimed = await store.claim_task("worker_a", lease_seconds=1)
        assert claimed is not None

        await asyncio.sleep(1.05)
        with pytest.raises(ValueError, match="cannot transition to running from claimed"):
            await store.start_task("task_expired_handoff", session_id="sess_expired")
        with pytest.raises(TaskClaimLost, match="lease for worker worker_a has expired"):
            await store.heartbeat("task_expired_handoff", "worker_a")

    _run(postgres_dsn, ops)


def test_postgres_heartbeat_rechecks_lease_after_row_lock_wait(postgres_dsn):
    async def run() -> None:
        import psycopg

        await _truncate(postgres_dsn)
        store = _new_store(postgres_dsn)
        blocker = await psycopg.AsyncConnection.connect(postgres_dsn)
        try:
            await store.ensure_schema()
            await store.create_task(TaskCreate(task_id="heartbeat-lock-expiry", type="job"))
            claimed = await store.claim_task("worker-a", lease_seconds=300)
            assert claimed is not None

            async with blocker.cursor() as cur:
                await cur.execute(
                    "UPDATE cayu_tasks "
                    "SET lease_expires_at = clock_timestamp() + INTERVAL '250 milliseconds' "
                    "WHERE id = %s",
                    (claimed.id,),
                )
            heartbeat = asyncio.create_task(
                store.heartbeat(claimed.id, "worker-a", extend_seconds=300)
            )
            await asyncio.sleep(0.05)
            assert not heartbeat.done()
            await asyncio.sleep(0.30)
            await blocker.commit()

            with pytest.raises(TaskClaimLost, match="has expired"):
                await heartbeat
            unchanged = await store.load_task(claimed.id)
            assert unchanged is not None
            assert unchanged.worker_id == "worker-a"
            assert unchanged.lease_expires_at is not None
            assert unchanged.lease_expires_at <= datetime.now(UTC)
        finally:
            await blocker.close()
            await store.close()

    asyncio.run(run())


def test_postgres_retry_settlement_rechecks_lease_after_row_lock_wait(postgres_dsn):
    async def run() -> None:
        import psycopg

        await _truncate(postgres_dsn)
        store = _new_store(postgres_dsn)
        blocker = await psycopg.AsyncConnection.connect(postgres_dsn)
        try:
            await store.ensure_schema()
            await store.create_task(
                TaskCreate(
                    task_id="retry-settlement-lock-expiry",
                    type="job",
                    retry_policy=TaskRetryPolicy(max_attempts=2),
                )
            )
            claimed = await store.claim_task("worker-a", lease_seconds=300)
            assert claimed is not None
            request = TaskRetrySettlementRequest(
                task_id=claimed.id,
                worker_id="worker-a",
                idempotency_key="lock-expiry-attempt",
                causal_budget_id=_retry_causal_budget_id(claimed),
                disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                error={"code": "temporary"},
            )

            async with blocker.cursor() as cur:
                await cur.execute(
                    "UPDATE cayu_tasks "
                    "SET lease_expires_at = clock_timestamp() + INTERVAL '250 milliseconds' "
                    "WHERE id = %s",
                    (claimed.id,),
                )
            settlement = asyncio.create_task(store.settle_task_retry_attempt(request))
            await asyncio.sleep(0.05)
            assert not settlement.done()
            await asyncio.sleep(0.30)
            await blocker.commit()

            with pytest.raises(TaskClaimLost, match="has expired"):
                await settlement
            assert (
                await store.load_task_retry_settlement(
                    claimed.id,
                    request.idempotency_key,
                )
                is None
            )
            unchanged = await store.load_task(claimed.id)
            assert unchanged is not None
            assert unchanged.status is TaskStatus.CLAIMED
            assert unchanged.worker_id == "worker-a"
        finally:
            await blocker.close()
            await store.close()

    asyncio.run(run())


def test_postgres_task_store_rejects_release_after_session_attachment(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_attached_release", type="review"))

        claimed = await store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None
        await store.attach_task(
            "task_attached_release",
            session_id="sess_attached",
            session_invocation=await task_backed_session_invocation(
                store, "task_attached_release", "sess_attached"
            ),
            worker_id="worker_a",
        )

        with pytest.raises(ValueError, match="already attached to session sess_attached"):
            await store.release_task("task_attached_release", "worker_a")

        loaded = await store.load_task("task_attached_release")
        assert loaded is not None
        assert loaded.status == TaskStatus.RUNNING
        assert loaded.session_id == "sess_attached"
        assert loaded.worker_id == "worker_a"

    _run(postgres_dsn, ops)


def test_postgres_task_store_releases_attached_worker_without_requeueing(postgres_dsn):
    async def ops(store):
        await store.create_task(
            TaskCreate(
                task_id="task_attached_handoff",
                type="review",
                assigned_agent_name="reviewer",
                metadata={"tenant": "acme"},
            )
        )
        await store.claim_task("worker_a", lease_seconds=300)
        attached = await store.attach_task(
            "task_attached_handoff",
            session_id="sess_attached_handoff",
            session_invocation=await task_backed_session_invocation(
                store, "task_attached_handoff", "sess_attached_handoff"
            ),
            worker_id="worker_a",
        )

        released = await store.release_attached_task_worker(
            "task_attached_handoff",
            "worker_a",
        )

        assert released.status == TaskStatus.RUNNING
        assert released.session_id == "sess_attached_handoff"
        assert released.worker_id is None
        assert released.lease_expires_at is None
        assert released.assigned_agent_name == "reviewer"
        assert released.metadata == {"tenant": "acme"}
        assert released.created_at == attached.created_at
        assert released.started_at == attached.started_at
        assert released.updated_at >= attached.updated_at
        assert await store.claim_task("worker_b", TaskQuery(type="review")) is None
        assert await store.reclaim_expired(query=TaskQuery(type="review")) == []

    _run(postgres_dsn, ops)


def test_postgres_task_store_rejects_invalid_attached_worker_release(postgres_dsn):
    async def ops(store):
        with pytest.raises(KeyError, match="Task not found"):
            await store.release_attached_task_worker("missing", "worker_a")

        await store.create_task(TaskCreate(task_id="task_unattached", type="review"))
        await store.claim_task("worker_a", lease_seconds=300)
        unattached_before = await store.load_task("task_unattached")
        assert unattached_before is not None
        with pytest.raises(ValueError, match="not running"):
            await store.release_attached_task_worker("task_unattached", "worker_a")

        await store.create_task(TaskCreate(task_id="task_wrong_worker", type="review"))
        await store.claim_task("worker_a", lease_seconds=300)
        await store.attach_task(
            "task_wrong_worker",
            session_id="sess_wrong_worker",
            session_invocation=await task_backed_session_invocation(
                store, "task_wrong_worker", "sess_wrong_worker"
            ),
            worker_id="worker_a",
        )
        wrong_worker_before = await store.load_task("task_wrong_worker")
        assert wrong_worker_before is not None
        with pytest.raises(ValueError, match="does not own"):
            await store.release_attached_task_worker("task_wrong_worker", "worker_b")

        await store.create_task(TaskCreate(task_id="task_expired_worker", type="review"))
        await store.claim_task("worker_a", lease_seconds=1)
        await store.attach_task(
            "task_expired_worker",
            session_id="sess_expired_worker",
            session_invocation=await task_backed_session_invocation(
                store, "task_expired_worker", "sess_expired_worker"
            ),
            worker_id="worker_a",
        )
        await asyncio.sleep(1.05)
        expired_before = await store.load_task("task_expired_worker")
        assert expired_before is not None
        with pytest.raises(TaskClaimLost, match="lease for worker worker_a has expired"):
            await store.release_attached_task_worker("task_expired_worker", "worker_a")

        await store.create_task(TaskCreate(task_id="task_terminal", type="review"))
        terminal_before = await store.complete_task(
            "task_terminal",
            {"winner": "terminal-state"},
        )
        with pytest.raises(ValueError, match="running"):
            await store.release_attached_task_worker("task_terminal", "worker_a")

        unattached_after = await store.load_task("task_unattached")
        wrong_worker_after = await store.load_task("task_wrong_worker")
        expired_after = await store.load_task("task_expired_worker")
        terminal_after = await store.load_task("task_terminal")
        assert unattached_after == unattached_before
        assert wrong_worker_after == wrong_worker_before
        assert expired_after == expired_before
        assert terminal_after is not None
        assert terminal_after == terminal_before

    _run(postgres_dsn, ops)


def test_postgres_task_store_does_not_reclaim_attached_expired_leases(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_attached_expired", type="review"))

        claimed = await store.claim_task("worker_a", lease_seconds=1)
        assert claimed is not None
        await store.attach_task(
            "task_attached_expired",
            session_id="sess_attached_expired",
            session_invocation=await task_backed_session_invocation(
                store, "task_attached_expired", "sess_attached_expired"
            ),
            worker_id="worker_a",
        )

        await asyncio.sleep(1.05)
        reclaimed = await store.reclaim_expired(query=TaskQuery(type="review"))
        assert reclaimed == []

        loaded = await store.load_task("task_attached_expired")
        assert loaded is not None
        assert loaded.status == TaskStatus.RUNNING
        assert loaded.session_id == "sess_attached_expired"
        assert loaded.worker_id == "worker_a"

    _run(postgres_dsn, ops)


def test_postgres_task_store_reclaim_expired_leases(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_expired", type="demo"))
        await store.create_task(TaskCreate(task_id="task_waiting", type="demo"))
        await store.claim_task(
            "worker_a",
            TaskQuery(type="demo", order_by=TaskOrder.CREATED_AT_ASC),
            lease_seconds=1,
        )

        await asyncio.sleep(1.05)
        reclaimed = await store.reclaim_expired(
            query=TaskQuery(type="demo"),
            max_reclaims=1,
        )
        assert [task.id for task in reclaimed] == ["task_expired"]
        assert reclaimed[0].status == TaskStatus.PENDING
        assert reclaimed[0].worker_id is None
        assert reclaimed[0].lease_expires_at is None

        loaded = await store.load_task("task_expired")
        assert loaded is not None
        assert loaded.status == TaskStatus.PENDING

        assert await store.reclaim_expired(query=TaskQuery(status=TaskStatus.PENDING)) == []

    _run(postgres_dsn, ops)


def test_postgres_task_store_validate_worker_lease_inputs(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_validate_worker", type="demo"))

        with pytest.raises(ValueError, match="lease_seconds must be >= 1"):
            await store.claim_task("worker_a", lease_seconds=0)
        with pytest.raises(TypeError, match="lease_seconds must be an integer"):
            await store.claim_task("worker_a", lease_seconds=True)  # type: ignore[arg-type]

        claimed = await store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None

        with pytest.raises(ValueError, match="extend_seconds must be >= 1"):
            await store.heartbeat("task_validate_worker", "worker_a", extend_seconds=0)
        with pytest.raises(ValueError, match="max_reclaims must be >= 1"):
            await store.reclaim_expired(max_reclaims=0)
        with pytest.raises(ValueError, match="do not support session_id"):
            await store.claim_task("worker_b", TaskQuery(session_id="sess_1"))
        with pytest.raises(ValueError, match="do not support session_id"):
            await store.reclaim_expired(query=TaskQuery(session_id="sess_1"))
        with pytest.raises(ValueError, match="do not support limit"):
            await store.claim_task("worker_b", TaskQuery(limit=2))
        with pytest.raises(ValueError, match="do not support offset"):
            await store.reclaim_expired(query=TaskQuery(offset=1))

    _run(postgres_dsn, ops)


def test_postgres_task_store_concurrent_claims_do_not_duplicate_tasks(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_a", type="review"))
        await store.create_task(TaskCreate(task_id="task_b", type="review"))

        second = _new_store(postgres_dsn)
        try:
            claimed = await asyncio.gather(
                store.claim_task(
                    "worker_a",
                    TaskQuery(type="review", order_by=TaskOrder.CREATED_AT_ASC),
                ),
                second.claim_task(
                    "worker_b",
                    TaskQuery(type="review", order_by=TaskOrder.CREATED_AT_ASC),
                ),
            )
            claimed_ids = sorted(task.id for task in claimed if task is not None)
            worker_ids = sorted(task.worker_id for task in claimed if task is not None)

            assert claimed_ids == ["task_a", "task_b"]
            assert worker_ids == ["worker_a", "worker_b"]

            loaded_a = await store.load_task("task_a")
            loaded_b = await second.load_task("task_b")
            assert loaded_a is not None
            assert loaded_b is not None
            assert {loaded_a.worker_id, loaded_b.worker_id} == {"worker_a", "worker_b"}
            assert loaded_a.id != loaded_b.id
        finally:
            await second.close()

    _run(postgres_dsn, ops)


def test_postgres_task_store_cancel_and_persistence(postgres_dsn):
    async def ops(store):
        await store.create_task(
            TaskCreate(
                task_id="task_cancel",
                type="process_invoice",
                assigned_agent_name="invoice_agent",
            )
        )
        await store.start_task(
            "task_cancel",
            session_id="sess_cancel",
            session_invocation=await task_backed_session_invocation(
                store, "task_cancel", "sess_cancel"
            ),
        )
        cancelled = await store.cancel_task("task_cancel", {"reason": "operator stop"})
        assert cancelled.status == TaskStatus.CANCELLED
        assert cancelled.error == {"reason": "operator stop"}
        assert cancelled.started_at is not None
        assert cancelled.completed_at is not None

        # Reload from a fresh store/pool to confirm durability.
        reopened = _new_store(postgres_dsn)
        try:
            loaded = await reopened.load_task("task_cancel")
            assert loaded is not None
            assert loaded.status == TaskStatus.CANCELLED
            assert loaded.session_id == "sess_cancel"
            assert loaded.error == {"reason": "operator stop"}
        finally:
            await reopened.close()

    _run(postgres_dsn, ops)


def test_postgres_task_store_rejects_stale_cross_pool_transitions(postgres_dsn):
    async def ops(store):
        await store.create_task(TaskCreate(task_id="task_claim", type="demo"))

        second = _new_store(postgres_dsn)
        try:
            await store.start_task(
                "task_claim",
                session_id="session_one",
                session_invocation=await task_backed_session_invocation(
                    store, "task_claim", "session_one"
                ),
            )
            with pytest.raises(ValueError, match="cannot transition to running"):
                await second.start_task("task_claim", session_id="session_two")

            completed = await second.complete_task("task_claim", {"ok": True})
            assert completed.status == TaskStatus.COMPLETED
            assert completed.session_id == "session_one"

            with pytest.raises(ValueError, match="already terminal"):
                await store.fail_task("task_claim", {"message": "too late"})
        finally:
            await second.close()

    _run(postgres_dsn, ops)
