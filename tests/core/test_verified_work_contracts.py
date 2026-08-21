from __future__ import annotations

import asyncio
import warnings
from collections.abc import AsyncIterator, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from types import MethodType
from uuid import uuid4

import pytest
from pydantic import ValidationError
from tests.core._execution_profile_fixtures import profiled_session_identity
from tests.core.task_invocation_fixtures import (
    task_backed_session_invocation,
    unattributed_session_invocation_binding,
)
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import (
    AgentSpec,
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactSessionRequest,
    CompletionConstraintOutcome,
    CompletionContinuationPolicy,
    CompletionCriterionOutcome,
    CompletionDecisionApplicationReceipt,
    CompletionDecisionApplicationRequest,
    CompletionDecisionCreate,
    CompletionGap,
    CompletionProposal,
    CompletionProposalCreate,
    CompletionRejectionAction,
    CompletionResultReference,
    CompletionSatisfactionBasis,
    CompletionVerdict,
    CompletionVerificationClaimLost,
    CompletionVerificationClaimRequest,
    CompletionVerifierRef,
    CriterionOutcomeStatus,
    Dispatcher,
    DispatchHandle,
    DispatchRequest,
    DispatchStatus,
    Environment,
    EnvironmentSpec,
    EventType,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileBehaviorIdentity,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
    ForkExecutionProfileSelection,
    ForkGroupBranchSpec,
    ForkGroupCheckpointSelector,
    ForkGroupEvaluatorSpec,
    ForkGroupRequest,
    ForkSessionRequest,
    ForkSystemPromptPolicy,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionsRecoveryRequest,
    InMemorySessionStore,
    InMemoryTaskStore,
    InvocationOrigin,
    InvocationOriginClaim,
    InvocationOriginTrust,
    Message,
    ModelCompactor,
    PendingToolApprovalEventView,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RunRequest,
    SecretRedactor,
    Session,
    SessionExecutionSource,
    SessionIdentity,
    SessionStatus,
    SQLiteTaskStore,
    StructuredOutputSpec,
    StructuredOutputStrategy,
    Task,
    TaskClaimLost,
    TaskCompletionDecisionRequired,
    TaskCreate,
    TaskExecutionSource,
    TaskInvocation,
    TaskQuery,
    TaskRetryPolicy,
    TaskStatus,
    TaskStoreDispatcher,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolCapabilityCeiling,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    ToolResult,
    ToolSpec,
    WorkAttemptCreate,
    WorkCompletionConflict,
    WorkConstraint,
    WorkContract,
    WorkContractConflict,
    WorkContractDraft,
    WorkContractRef,
    WorkCriterion,
    WorkEvidenceReference,
    WorkEvidenceRequirement,
    completion_gap_fingerprint,
    completion_result_sha256,
    run_task_worker,
    work_contract_from_draft,
)
from cayu._validation import FrozenJsonDict, FrozenJsonList, canonical_durable_json_bytes
from cayu.core.tools import Tool, ToolContext
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import tasks as tasks_module
from cayu.runtime import work_contracts as work_contracts_module
from cayu.runtime.sessions import (
    PROMPT_ANATOMY_TRANSITION_METADATA_KEY,
    ModelCompletionStageRequest,
    run_request_with_runtime_invocation,
)
from cayu.runtime.tasks import copy_task
from cayu.runtime.work_contracts import (
    WORK_COMPLETION_APPLICATION_MAX_BYTES,
    WORK_COMPLETION_APPLICATION_MAX_ITEMS,
    WORK_COMPLETION_APPLICATION_RECEIPT_MAX_BYTES,
    WORK_COMPLETION_LINKED_ID_MAX_BYTES,
    WORK_CONTRACT_MAX_CRITERIA,
    WORK_CONTRACT_TASK_CREATION_MAX_BYTES,
    WORK_CONTRACT_TASK_MAX_BYTES,
    completion_decision_application_request_sha256,
    completion_decision_request_sha256,
    completion_proposal_request_sha256,
    completion_verification_claim_request_sha256,
    work_attempt_request_sha256,
)
from cayu.runtime.workspace_observation_recovery import (
    workspace_observation_pending_cancellation_requests,
)
from cayu.tools import UserInputTool


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


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
        traceback = current.__traceback__
        while traceback is not None:
            if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
                retained = {
                    name: type(value).__name__
                    for name, value in traceback.tb_frame.f_locals.items()
                    if secret in repr(value)
                }
                assert retained == {}, (
                    traceback.tb_frame.f_code.co_filename,
                    traceback.tb_frame.f_code.co_name,
                    retained,
                )
            traceback = traceback.tb_next
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


class _RecordingProvider(ModelProvider):
    name = "verified-work-test-provider"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:verified-work:recording-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelStreamEvent.text_delta("verified work test response")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _dispatch_test_app() -> tuple[
    CayuApp,
    InMemorySessionStore,
    InMemoryTaskStore,
    TaskStoreDispatcher,
    _RecordingProvider,
]:
    session_store = InMemorySessionStore()
    task_store = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(task_store)
    provider = _RecordingProvider()
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        dispatcher=dispatcher,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="verified-work-test-model"))
    return app, session_store, task_store, dispatcher, provider


async def _create_dispatch_test_session(app: CayuApp, session_id: str) -> None:
    # Seed a durable session as if it predated adoption of the verified-work
    # task store. The app under test must make its own first admission decision.
    seed_app = CayuApp(session_store=app.session_store, enable_logging=False)
    seed_app.register_provider(_RecordingProvider(), default=True)
    seed_app.register_agent(AgentSpec(name="assistant", model="verified-work-test-model"))
    async for _ in seed_app.run(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "Create a resumable session.")],
        )
    ):
        pass


def _verifier() -> CompletionVerifierRef:
    return CompletionVerifierRef(
        verifier_id="bid-readiness",
        version="v1",
        configuration_fingerprint=_digest("bid-readiness-v1"),
    )


def _contract(
    *,
    contract_id: str = "bid-package",
    objective: str = "Publish a complete and approved bid package.",
    version: int = 1,
    supersedes=None,
    continuation_policy: CompletionContinuationPolicy | None = None,
    constraint_evidence_requirement_ids: tuple[str, ...] = ("approval",),
) -> WorkContract:
    return work_contract_from_draft(
        WorkContractDraft(
            contract_id=contract_id,
            version=version,
            supersedes=supersedes,
            objective=objective,
            criteria=(
                WorkCriterion(
                    criterion_id="coverage",
                    ordinal=1,
                    description="All required bid sections are present.",
                    evidence_requirement_ids=("artifact",),
                ),
                WorkCriterion(
                    criterion_id="approval",
                    ordinal=2,
                    description="The package has the required business approval.",
                    evidence_requirement_ids=("approval",),
                ),
            ),
            constraints=(
                WorkConstraint(
                    constraint_id="no-unapproved-send",
                    description="Do not send a package before approval.",
                    evidence_requirement_ids=constraint_evidence_requirement_ids,
                ),
            ),
            evidence_requirements=(
                WorkEvidenceRequirement(
                    requirement_id="approval",
                    kind="business.approval",
                    description="A durable approval decision.",
                ),
                WorkEvidenceRequirement(
                    requirement_id="artifact",
                    kind="artifact.version",
                    description="An immutable artifact version.",
                ),
            ),
            verifier=_verifier(),
            continuation_policy=(
                continuation_policy
                or CompletionContinuationPolicy(
                    rejection_action=CompletionRejectionAction.CONTINUE,
                    max_attempts=3,
                    max_repeated_gap_count=2,
                )
            ),
        )
    )


def _artifact_evidence(version: str = "7") -> WorkEvidenceReference:
    return WorkEvidenceReference(
        kind="artifact.version",
        reference_id=f"artifact:bid-package:{version}",
        requirement_id="artifact",
        version=version,
        digest=_digest(f"artifact-version-{version}"),
    )


def _approval_evidence() -> WorkEvidenceReference:
    return WorkEvidenceReference(
        kind="business.approval",
        reference_id="approval:bid-package:3",
        requirement_id="approval",
        version="3",
        digest=_digest("approval-version-3"),
    )


def _shared_evidence_contract() -> WorkContract:
    return work_contract_from_draft(
        WorkContractDraft(
            contract_id="shared-artifact",
            version=1,
            objective="Verify two properties of one immutable artifact version.",
            criteria=(
                WorkCriterion(
                    criterion_id="format",
                    ordinal=1,
                    description="The artifact has the required format.",
                    evidence_requirement_ids=("artifact-format",),
                ),
                WorkCriterion(
                    criterion_id="contents",
                    ordinal=2,
                    description="The artifact has the required contents.",
                    evidence_requirement_ids=("artifact-contents",),
                ),
            ),
            evidence_requirements=(
                WorkEvidenceRequirement(
                    requirement_id="artifact-contents",
                    kind="artifact.version",
                    description="The immutable artifact used for content verification.",
                ),
                WorkEvidenceRequirement(
                    requirement_id="artifact-format",
                    kind="artifact.version",
                    description="The immutable artifact used for format verification.",
                ),
            ),
            verifier=_verifier(),
        )
    )


def _shared_evidence_decision(
    *,
    proposal_id: str,
    claim_id: str,
    worker_id: str,
    decision_id: str = "shared-evidence-decision",
) -> CompletionDecisionCreate:
    shared = {
        "kind": "artifact.version",
        "reference_id": "artifact:shared:1",
        "version": "1",
        "digest": _digest("shared-artifact-version-1"),
    }
    format_evidence = WorkEvidenceReference(
        **shared,
        requirement_id="artifact-format",
    )
    contents_evidence = WorkEvidenceReference(
        **shared,
        requirement_id="artifact-contents",
    )
    return CompletionDecisionCreate(
        decision_id=decision_id,
        proposal_id=proposal_id,
        claim_id=claim_id,
        worker_id=worker_id,
        verifier=_verifier(),
        verdict=CompletionVerdict.ACCEPTED,
        criterion_outcomes=(
            CompletionCriterionOutcome(
                criterion_id="format",
                status=CriterionOutcomeStatus.SATISFIED,
                reason_code="artifact.format.valid",
                satisfaction_basis=CompletionSatisfactionBasis.EVIDENCE,
                evidence_references=(format_evidence,),
            ),
            CompletionCriterionOutcome(
                criterion_id="contents",
                status=CriterionOutcomeStatus.SATISFIED,
                reason_code="artifact.contents.valid",
                satisfaction_basis=CompletionSatisfactionBasis.EVIDENCE,
                evidence_references=(contents_evidence,),
            ),
        ),
    )


def _task_result(suffix: str = "1") -> dict[str, object]:
    return {
        "session_id": f"session:bid:{suffix}",
        "artifact_id": f"artifact:bid-package:{suffix}",
    }


def _result_reference(
    suffix: str = "1",
    *,
    result: dict[str, object] | None = None,
) -> CompletionResultReference:
    result = _task_result(suffix) if result is None else result
    return CompletionResultReference(
        kind="session.output",
        reference_id=f"session:bid:{suffix}",
        digest=completion_result_sha256(result),
    )


def _rejected_decision(
    *,
    proposal_id: str,
    claim_id: str,
    worker_id: str,
    decision_id: str = "decision-rejected",
    gap_summary: str = "Approval evidence is missing.",
    artifact_version: str = "7",
) -> CompletionDecisionCreate:
    artifact_evidence = _artifact_evidence(artifact_version)
    return CompletionDecisionCreate(
        decision_id=decision_id,
        proposal_id=proposal_id,
        claim_id=claim_id,
        worker_id=worker_id,
        verifier=_verifier(),
        verdict=CompletionVerdict.REJECTED,
        criterion_outcomes=(
            CompletionCriterionOutcome(
                criterion_id="coverage",
                status=CriterionOutcomeStatus.SATISFIED,
                reason_code="artifact.valid",
                satisfaction_basis=CompletionSatisfactionBasis.EVIDENCE,
                evidence_references=(artifact_evidence,),
            ),
            CompletionCriterionOutcome(
                criterion_id="approval",
                status=CriterionOutcomeStatus.UNVERIFIABLE,
                reason_code="approval.missing",
            ),
        ),
        constraint_outcomes=(
            CompletionConstraintOutcome(
                constraint_id="no-unapproved-send",
                status=CriterionOutcomeStatus.UNVERIFIABLE,
                reason_code="approval.missing",
            ),
        ),
        gaps=(
            CompletionGap(
                criterion_id="approval",
                code="approval.missing",
                evidence_requirement_ids=("approval",),
                summary=gap_summary,
            ),
            CompletionGap(
                constraint_id="no-unapproved-send",
                code="approval.missing",
                evidence_requirement_ids=("approval",),
                summary=gap_summary,
            ),
        ),
        evidence_references=(artifact_evidence,),
    )


def _accepted_decision(
    *,
    proposal_id: str,
    claim_id: str,
    worker_id: str,
) -> CompletionDecisionCreate:
    return CompletionDecisionCreate(
        decision_id="decision-accepted",
        proposal_id=proposal_id,
        claim_id=claim_id,
        worker_id=worker_id,
        verifier=_verifier(),
        verdict=CompletionVerdict.ACCEPTED,
        criterion_outcomes=(
            CompletionCriterionOutcome(
                criterion_id="coverage",
                status=CriterionOutcomeStatus.SATISFIED,
                reason_code="artifact.valid",
                satisfaction_basis=CompletionSatisfactionBasis.EVIDENCE,
                evidence_references=(_artifact_evidence(),),
            ),
            CompletionCriterionOutcome(
                criterion_id="approval",
                status=CriterionOutcomeStatus.SATISFIED,
                reason_code="approval.valid",
                satisfaction_basis=CompletionSatisfactionBasis.EVIDENCE,
                evidence_references=(_approval_evidence(),),
            ),
        ),
        constraint_outcomes=(
            CompletionConstraintOutcome(
                constraint_id="no-unapproved-send",
                status=CriterionOutcomeStatus.SATISFIED,
                reason_code="approval.valid",
                satisfaction_basis=CompletionSatisfactionBasis.EVIDENCE,
                evidence_references=(_approval_evidence(),),
            ),
        ),
        evidence_references=(_artifact_evidence(), _approval_evidence()),
    )


def _held_decision(
    *,
    verdict: CompletionVerdict,
    proposal_id: str,
    claim_id: str,
    worker_id: str,
) -> CompletionDecisionCreate:
    return CompletionDecisionCreate(
        decision_id=f"decision-{verdict.value}",
        proposal_id=proposal_id,
        claim_id=claim_id,
        worker_id=worker_id,
        verifier=_verifier(),
        verdict=verdict,
        criterion_outcomes=(
            CompletionCriterionOutcome(
                criterion_id="coverage",
                status=CriterionOutcomeStatus.SATISFIED,
                reason_code="artifact.valid",
                satisfaction_basis=CompletionSatisfactionBasis.EVIDENCE,
                evidence_references=(_artifact_evidence(),),
            ),
            CompletionCriterionOutcome(
                criterion_id="approval",
                status=CriterionOutcomeStatus.UNVERIFIABLE,
                reason_code="approval.unavailable",
            ),
        ),
        constraint_outcomes=(
            CompletionConstraintOutcome(
                constraint_id="no-unapproved-send",
                status=CriterionOutcomeStatus.UNVERIFIABLE,
                reason_code="approval.unavailable",
            ),
        ),
        gaps=(
            CompletionGap(
                criterion_id="approval",
                code="approval.unavailable",
                evidence_requirement_ids=("approval",),
            ),
            CompletionGap(
                constraint_id="no-unapproved-send",
                code="approval.unavailable",
                evidence_requirement_ids=("approval",),
            ),
        ),
        evidence_references=(_artifact_evidence(),),
    )


def test_work_contract_fingerprint_covers_the_complete_canonical_definition() -> None:
    contract = _contract()
    reconstructed = WorkContract.model_validate(contract.model_dump(mode="json", warnings=False))

    assert reconstructed == contract
    assert contract.reference().fingerprint == contract.fingerprint
    assert _contract().fingerprint == contract.fingerprint
    assert (
        _contract(objective="Publish a reviewed bid package.").fingerprint != contract.fingerprint
    )
    assert _contract(constraint_evidence_requirement_ids=()).fingerprint != contract.fingerprint

    with pytest.raises(ValidationError, match="fingerprint conflicts"):
        WorkContract.model_validate(
            {
                **contract.model_dump(mode="python", warnings=False),
                "objective": "A silently weakened objective.",
            }
        )


def test_work_contract_rejects_noncanonical_or_ambiguous_inputs() -> None:
    with pytest.raises(ValidationError, match="integer"):
        WorkContractDraft(
            contract_id="invalid",
            version=True,
            objective="Invalid version.",
            criteria=(
                WorkCriterion(
                    criterion_id="criterion",
                    ordinal=1,
                    description="One criterion.",
                ),
            ),
            verifier=_verifier(),
        )

    oversized_references = tuple(
        WorkEvidenceReference(
            kind="artifact.version",
            reference_id=f"artifact:{index:03d}:" + ("x" * 2000),
        )
        for index in range(128)
    )
    with pytest.raises(ValidationError, match="Completion proposal must not exceed"):
        CompletionProposalCreate(
            proposal_id="oversized-proposal",
            attempt_id="oversized-attempt",
            result=_result_reference("oversized"),
            evidence_references=oversized_references,
        )

    with pytest.raises(ValidationError, match="canonical order"):
        WorkContractDraft(
            contract_id="invalid-order",
            version=1,
            objective="Invalid requirement ordering.",
            criteria=(
                WorkCriterion(
                    criterion_id="criterion",
                    ordinal=1,
                    description="One criterion.",
                ),
            ),
            evidence_requirements=(
                WorkEvidenceRequirement(
                    requirement_id="z",
                    kind="artifact.version",
                    description="Last.",
                ),
                WorkEvidenceRequirement(
                    requirement_id="a",
                    kind="artifact.version",
                    description="First.",
                ),
            ),
            verifier=_verifier(),
        )

    with pytest.raises(ValidationError, match="must be assigned"):
        WorkContractDraft(
            contract_id="orphaned-requirement",
            version=1,
            objective="Reject evidence that has no acceptance owner.",
            criteria=(
                WorkCriterion(
                    criterion_id="criterion",
                    ordinal=1,
                    description="One criterion.",
                ),
            ),
            evidence_requirements=(
                WorkEvidenceRequirement(
                    requirement_id="orphan",
                    kind="artifact.version",
                    description="This requirement has no criterion or constraint owner.",
                ),
            ),
            verifier=_verifier(),
        )


def test_work_contract_collection_preflight_stops_at_the_declared_bound() -> None:
    class CriteriaInput:
        def __init__(self, count: int | None) -> None:
            self.count = count
            self.consumed = 0

        def __iter__(self) -> Iterator[WorkCriterion]:
            ordinal = 1
            while self.count is None or ordinal <= self.count:
                self.consumed += 1
                if self.consumed > WORK_CONTRACT_MAX_CRITERIA + 1:
                    raise AssertionError("Bounded validation exhausted an untrusted iterable.")
                yield WorkCriterion(
                    criterion_id=(
                        f"criterion-{ordinal:03d}" if self.count is not None else "criterion"
                    ),
                    ordinal=ordinal if self.count is not None else 1,
                    description=f"Acceptance criterion {ordinal}.",
                )
                ordinal += 1

    exact = CriteriaInput(WORK_CONTRACT_MAX_CRITERIA)
    contract = WorkContractDraft.model_validate(
        {
            "contract_id": "bounded-exact-contract",
            "version": 1,
            "objective": "Accept the exact declared number of criteria.",
            "criteria": exact,
            "verifier": _verifier(),
        }
    )
    assert len(contract.criteria) == WORK_CONTRACT_MAX_CRITERIA
    assert exact.consumed == WORK_CONTRACT_MAX_CRITERIA

    excessive = CriteriaInput(None)
    with pytest.raises(ValidationError, match="criteria must contain at most 64 values"):
        WorkContractDraft.model_validate(
            {
                "contract_id": "bounded-excessive-contract",
                "version": 1,
                "objective": "Reject before exhausting an untrusted criteria iterable.",
                "criteria": excessive,
                "verifier": _verifier(),
            }
        )
    assert excessive.consumed == WORK_CONTRACT_MAX_CRITERIA + 1


def test_work_contract_evidence_assignments_fit_completion_decision_bounds() -> None:
    def requirements(count: int) -> tuple[WorkEvidenceRequirement, ...]:
        return tuple(
            WorkEvidenceRequirement(
                requirement_id=f"requirement-{index:03d}",
                kind="artifact.version",
                description=f"Required evidence {index}.",
            )
            for index in range(count)
        )

    def criterion(
        ordinal: int,
        requirement_ids: tuple[str, ...],
    ) -> WorkCriterion:
        return WorkCriterion(
            criterion_id=f"criterion-{ordinal:03d}",
            ordinal=ordinal,
            description=f"Criterion {ordinal}.",
            evidence_requirement_ids=requirement_ids,
        )

    thirty_two_requirements = requirements(32)
    thirty_two_ids = tuple(item.requirement_id for item in thirty_two_requirements)
    exact_subject_limit = WorkContractDraft(
        contract_id="exact-subject-evidence-limit",
        version=1,
        objective="Remain representable by one completion outcome.",
        criteria=(criterion(1, thirty_two_ids),),
        evidence_requirements=thirty_two_requirements,
        verifier=_verifier(),
    )
    assert work_contract_from_draft(exact_subject_limit).criteria[0].evidence_requirement_ids == (
        thirty_two_ids
    )

    thirty_three_requirements = requirements(33)
    with pytest.raises(ValidationError, match="may assign at most 32 evidence requirements"):
        WorkContractDraft(
            contract_id="unrepresentable-subject-evidence",
            version=1,
            objective="Reject an outcome that cannot cite all required evidence.",
            criteria=(
                criterion(
                    1,
                    tuple(item.requirement_id for item in thirty_three_requirements),
                ),
            ),
            evidence_requirements=thirty_three_requirements,
            verifier=_verifier(),
        )

    exact_aggregate_limit = WorkContractDraft(
        contract_id="exact-aggregate-evidence-limit",
        version=1,
        objective="Remain representable by one completion decision.",
        criteria=tuple(criterion(ordinal, thirty_two_ids) for ordinal in range(1, 9)),
        evidence_requirements=thirty_two_requirements,
        verifier=_verifier(),
    )
    assert len(work_contract_from_draft(exact_aggregate_limit).criteria) == 8

    with pytest.raises(ValidationError, match="must not exceed 256 in aggregate"):
        WorkContractDraft(
            contract_id="unrepresentable-aggregate-evidence",
            version=1,
            objective="Reject assignments that cannot fit one completion decision.",
            criteria=tuple(criterion(ordinal, thirty_two_ids) for ordinal in range(1, 9)),
            constraints=(
                WorkConstraint(
                    constraint_id="one-more-assignment",
                    description="One assignment beyond the aggregate decision limit.",
                    evidence_requirement_ids=(thirty_two_ids[0],),
                ),
            ),
            evidence_requirements=thirty_two_requirements,
            verifier=_verifier(),
        )


def test_completion_decision_rejects_conflicting_evidence_representations() -> None:
    accepted = _accepted_decision(
        proposal_id="evidence-conflict-proposal",
        claim_id="evidence-conflict-claim",
        worker_id="evidence-conflict-worker",
    )

    # Exact overlap between the decision summary and outcome-specific evidence
    # is intentional and remains a single unambiguous authority record.
    assert (
        CompletionDecisionCreate.model_validate(accepted.model_dump(mode="python", warnings=False))
        == accepted
    )

    approval = _approval_evidence()
    unavailable_approval = approval.model_copy(
        update={
            "available": False,
            "unavailable_reason": "store.unavailable",
        }
    )
    conflicting_availability = accepted.model_copy(
        update={
            "evidence_references": (
                _artifact_evidence(),
                unavailable_approval,
            )
        }
    )
    with pytest.raises(ValidationError, match="conflicting representations"):
        CompletionDecisionCreate.model_validate(
            conflicting_availability.model_dump(mode="python", warnings=False)
        )

    conflicting_digest = accepted.model_copy(
        update={
            "evidence_references": (
                _artifact_evidence(),
                approval.model_copy(update={"digest": _digest("conflicting-approval")}),
            )
        }
    )
    with pytest.raises(ValidationError, match="conflicting representations"):
        CompletionDecisionCreate.model_validate(
            conflicting_digest.model_dump(mode="python", warnings=False)
        )


def test_completion_decision_reconciles_shared_evidence_across_requirements() -> None:
    decision = _shared_evidence_decision(
        proposal_id="shared-evidence-proposal",
        claim_id="shared-evidence-claim",
        worker_id="shared-evidence-worker",
    )

    # One immutable evidence version may satisfy multiple requirements when its
    # authority-bearing representation is identical under every binding.
    assert (
        CompletionDecisionCreate.model_validate(decision.model_dump(mode="python", warnings=False))
        == decision
    )

    contents_outcome = decision.criterion_outcomes[1]
    contents_evidence = contents_outcome.evidence_references[0]
    conflicting_digest = decision.model_copy(
        update={
            "criterion_outcomes": (
                decision.criterion_outcomes[0],
                contents_outcome.model_copy(
                    update={
                        "evidence_references": (
                            contents_evidence.model_copy(
                                update={"digest": _digest("conflicting-artifact")}
                            ),
                        )
                    }
                ),
            )
        }
    )
    with pytest.raises(ValidationError, match="conflicting representations"):
        CompletionDecisionCreate.model_validate(
            conflicting_digest.model_dump(mode="python", warnings=False)
        )

    conflicting_availability = decision.model_copy(
        update={
            "criterion_outcomes": (
                decision.criterion_outcomes[0],
                contents_outcome.model_copy(
                    update={
                        "satisfaction_basis": CompletionSatisfactionBasis.VERIFIER_ASSERTION,
                        "evidence_references": (),
                    }
                ),
            ),
            "evidence_references": (
                contents_evidence.model_copy(
                    update={
                        "available": False,
                        "unavailable_reason": "artifact.unavailable",
                    }
                ),
            ),
        }
    )
    with pytest.raises(ValidationError, match="conflicting representations"):
        CompletionDecisionCreate.model_validate(
            conflicting_availability.model_dump(mode="python", warnings=False)
        )


def test_gap_fingerprint_tracks_only_stable_unresolved_gap_identity() -> None:
    first = _rejected_decision(
        proposal_id="proposal",
        claim_id="claim",
        worker_id="verifier",
        gap_summary="First explanation.",
    )
    second = _rejected_decision(
        proposal_id="proposal",
        claim_id="claim",
        worker_id="verifier",
        gap_summary="Different prose for the same durable gap.",
        artifact_version="8",
    )
    second = CompletionDecisionCreate.model_validate(
        second.model_copy(
            update={
                "criterion_outcomes": (
                    second.criterion_outcomes[0].model_copy(
                        update={"reason_code": "artifact.refreshed"}
                    ),
                    second.criterion_outcomes[1],
                )
            }
        ).model_dump(mode="python", warnings=False)
    )

    assert completion_gap_fingerprint(first) == completion_gap_fingerprint(second)

    changed = second.model_copy(
        update={
            "criterion_outcomes": (
                second.criterion_outcomes[0],
                second.criterion_outcomes[1].model_copy(
                    update={"reason_code": "approval.unavailable"}
                ),
            )
        }
    )
    changed = CompletionDecisionCreate.model_validate(
        changed.model_dump(mode="python", warnings=False)
    )
    assert completion_gap_fingerprint(first) != completion_gap_fingerprint(changed)


def test_exact_operation_digests_cover_every_decision_bearing_field() -> None:
    contract = _contract()
    other_contract = _contract(contract_id="other-contract")
    attempt = WorkAttemptCreate(
        attempt_id="attempt",
        task_id="task",
        session_id="session",
        contract=contract.reference(),
        execution_profile_fingerprint=_digest("profile"),
        worker_id="task-worker",
    )
    attempt_variants = (
        attempt.model_copy(update={"task_id": "other-task"}),
        attempt.model_copy(update={"session_id": "other-session"}),
        attempt.model_copy(update={"contract": other_contract.reference()}),
        attempt.model_copy(update={"execution_profile_fingerprint": _digest("other-profile")}),
        attempt.model_copy(update={"worker_id": "other-task-worker"}),
    )
    assert all(
        work_attempt_request_sha256(variant) != work_attempt_request_sha256(attempt)
        for variant in attempt_variants
    )

    proposal = CompletionProposalCreate(
        proposal_id="proposal",
        attempt_id=attempt.attempt_id,
        result=_result_reference(),
        evidence_references=(_artifact_evidence(),),
    )
    proposal_variants = (
        proposal.model_copy(update={"attempt_id": "other-attempt"}),
        proposal.model_copy(update={"result": _result_reference("other")}),
        proposal.model_copy(
            update={"evidence_references": (_artifact_evidence(), _approval_evidence())}
        ),
    )
    assert all(
        completion_proposal_request_sha256(variant) != completion_proposal_request_sha256(proposal)
        for variant in proposal_variants
    )

    claim = CompletionVerificationClaimRequest(
        claim_id="claim",
        proposal_id=proposal.proposal_id,
        worker_id="verifier-worker",
        verifier=contract.verifier,
        lease_seconds=60,
    )
    other_verifier = CompletionVerifierRef(
        verifier_id=contract.verifier.verifier_id,
        version="v2",
        configuration_fingerprint=_digest("bid-readiness-v2"),
    )
    claim_variants = (
        claim.model_copy(update={"proposal_id": "other-proposal"}),
        claim.model_copy(update={"worker_id": "other-verifier-worker"}),
        claim.model_copy(update={"verifier": other_verifier}),
        claim.model_copy(update={"lease_seconds": 61}),
    )
    assert all(
        completion_verification_claim_request_sha256(variant)
        != completion_verification_claim_request_sha256(claim)
        for variant in claim_variants
    )

    decision = _rejected_decision(
        proposal_id=proposal.proposal_id,
        claim_id=claim.claim_id,
        worker_id=claim.worker_id,
    )
    changed_outcome = decision.criterion_outcomes[1].model_copy(
        update={"summary": "Verifier-owned bounded explanation."}
    )
    changed_gap = decision.gaps[0].model_copy(
        update={"summary": "A different bounded actionable explanation."}
    )
    changed_constraint = decision.constraint_outcomes[0].model_copy(
        update={"summary": "A bounded constraint explanation."}
    )
    decision_variants = (
        decision.model_copy(update={"proposal_id": "other-proposal"}),
        decision.model_copy(update={"claim_id": "other-claim"}),
        decision.model_copy(update={"worker_id": "other-verifier-worker"}),
        decision.model_copy(update={"verifier": other_verifier}),
        decision.model_copy(
            update={
                "criterion_outcomes": (
                    decision.criterion_outcomes[0],
                    changed_outcome,
                )
            }
        ),
        decision.model_copy(update={"constraint_outcomes": (changed_constraint,)}),
        decision.model_copy(update={"gaps": (changed_gap, decision.gaps[1])}),
        decision.model_copy(
            update={"evidence_references": (_artifact_evidence(), _approval_evidence())}
        ),
        _accepted_decision(
            proposal_id=proposal.proposal_id,
            claim_id=claim.claim_id,
            worker_id=claim.worker_id,
        ).model_copy(update={"decision_id": decision.decision_id}),
    )
    assert all(
        completion_decision_request_sha256(variant) != completion_decision_request_sha256(decision)
        for variant in decision_variants
    )

    application = CompletionDecisionApplicationRequest(
        task_id=attempt.task_id,
        decision_id=decision.decision_id,
        idempotency_key="apply-decision",
    )
    application_variants = (
        application.model_copy(update={"task_id": "other-task"}),
        application.model_copy(update={"decision_id": "other-decision"}),
        application.model_copy(update={"idempotency_key": "other-application"}),
        application.model_copy(
            update={
                "result": _task_result("unexpected"),
                "result_reference": _result_reference("unexpected"),
            }
        ),
    )
    assert all(
        completion_decision_application_request_sha256(variant)
        != completion_decision_application_request_sha256(application)
        for variant in application_variants
    )
    with pytest.raises(ValidationError, match="verified result digest"):
        CompletionDecisionApplicationRequest(
            task_id=attempt.task_id,
            decision_id=decision.decision_id,
            idempotency_key="apply-mismatched-result",
            result=_task_result("one-result"),
            result_reference=_result_reference("another-result"),
        )


def test_decision_application_request_enforces_exact_byte_and_item_bounds() -> None:
    empty_result: dict[str, object] = {"payload": ""}
    empty_reference = CompletionResultReference(
        kind="session.output",
        reference_id="bounded-result",
        digest=completion_result_sha256(empty_result),
    )
    empty_document = {
        "task_id": "bounded-task",
        "decision_id": "bounded-decision",
        "idempotency_key": "bounded-application",
        "result": empty_result,
        "result_reference": empty_reference.model_dump(mode="json", warnings=False),
    }
    payload_bytes = WORK_COMPLETION_APPLICATION_MAX_BYTES - len(
        canonical_durable_json_bytes(empty_document, "completion_decision_application")
    )
    exact_result: dict[str, object] = {"payload": "x" * payload_bytes}
    exact = CompletionDecisionApplicationRequest(
        task_id="bounded-task",
        decision_id="bounded-decision",
        idempotency_key="bounded-application",
        result=exact_result,
        result_reference=CompletionResultReference(
            kind="session.output",
            reference_id="bounded-result",
            digest=completion_result_sha256(exact_result),
        ),
    )
    assert (
        len(
            canonical_durable_json_bytes(
                exact.model_dump(mode="json", warnings=False),
                "completion_decision_application",
            )
        )
        == WORK_COMPLETION_APPLICATION_MAX_BYTES
    )

    empty_float_result: dict[str, object] = {"float": 1e-7, "payload": ""}
    empty_float_document = {
        "task_id": "bounded-float-task",
        "decision_id": "bounded-float-decision",
        "idempotency_key": "bounded-float-application",
        "result": empty_float_result,
        "result_reference": CompletionResultReference(
            kind="session.output",
            reference_id="bounded-float-result",
            digest=completion_result_sha256(empty_float_result),
        ).model_dump(mode="json", warnings=False),
    }
    float_payload_bytes = WORK_COMPLETION_APPLICATION_MAX_BYTES - len(
        canonical_durable_json_bytes(
            empty_float_document,
            "completion_decision_application",
        )
    )
    exact_float_result: dict[str, object] = {
        "float": 1e-7,
        "payload": "x" * float_payload_bytes,
    }
    exact_float = CompletionDecisionApplicationRequest(
        task_id="bounded-float-task",
        decision_id="bounded-float-decision",
        idempotency_key="bounded-float-application",
        result=exact_float_result,
        result_reference=CompletionResultReference(
            kind="session.output",
            reference_id="bounded-float-result",
            digest=completion_result_sha256(exact_float_result),
        ),
    )
    assert (
        len(
            canonical_durable_json_bytes(
                exact_float.model_dump(mode="json", warnings=False),
                "completion_decision_application",
            )
        )
        == WORK_COMPLETION_APPLICATION_MAX_BYTES
    )

    oversized_result = {"payload": "x" * (payload_bytes + 1)}
    with pytest.raises(ValidationError, match="must not exceed"):
        CompletionDecisionApplicationRequest(
            task_id="bounded-task",
            decision_id="bounded-decision",
            idempotency_key="oversized-application",
            result=oversized_result,
            result_reference=CompletionResultReference(
                kind="session.output",
                reference_id="oversized-result",
                digest=completion_result_sha256(oversized_result),
            ),
        )

    exact_item_result = {
        "items": [None] * (WORK_COMPLETION_APPLICATION_MAX_ITEMS - 10),
    }
    CompletionDecisionApplicationRequest(
        task_id="bounded-task",
        decision_id="bounded-decision",
        idempotency_key="exact-item-application",
        result=exact_item_result,
        result_reference=CompletionResultReference(
            kind="session.output",
            reference_id="exact-item-result",
            digest=completion_result_sha256(exact_item_result),
        ),
    )
    high_cardinality_result = {
        "items": [None] * WORK_COMPLETION_APPLICATION_MAX_ITEMS,
    }
    with pytest.raises(ValidationError, match="JSON values"):
        CompletionDecisionApplicationRequest(
            task_id="bounded-task",
            decision_id="bounded-decision",
            idempotency_key="high-cardinality-application",
            result=high_cardinality_result,
            result_reference=CompletionResultReference(
                kind="session.output",
                reference_id="high-cardinality-result",
                digest=completion_result_sha256(high_cardinality_result),
            ),
        )


def test_decision_application_preflights_frozen_json_before_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copy_reached = False

    def reject_copy(*_args: object, **_kwargs: object) -> None:
        nonlocal copy_reached
        copy_reached = True
        raise AssertionError("bounded frozen input reached the defensive copier")

    monkeypatch.setattr(
        work_contracts_module,
        "copy_durable_json_object",
        reject_copy,
    )
    reference = CompletionResultReference(
        kind="session.output",
        reference_id="frozen-preflight-result",
        digest=_digest("frozen-preflight-result"),
    )
    cases = (
        (
            FrozenJsonDict(
                {"items": FrozenJsonList([None] * (WORK_COMPLETION_APPLICATION_MAX_ITEMS + 1))}
            ),
            "JSON values",
        ),
        (
            FrozenJsonDict(
                {"items": FrozenJsonList(["x" * WORK_COMPLETION_APPLICATION_MAX_BYTES])}
            ),
            "must not exceed",
        ),
    )
    for index, (result, message) in enumerate(cases):
        with pytest.raises(ValidationError, match=message):
            CompletionDecisionApplicationRequest(
                task_id="frozen-preflight-task",
                decision_id=f"frozen-preflight-decision-{index}",
                idempotency_key=f"frozen-preflight-application-{index}",
                result=result,
                result_reference=reference,
            )
        assert copy_reached is False


@pytest.mark.parametrize("field_name", ["result", "result_reference"])
def test_decision_application_rejects_mutated_mapping_before_materialization(
    field_name: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = f"hostile-application-{field_name}-secret"

    class HostileMapping(Mapping[str, object]):
        touched = False

        def __getitem__(self, key: str) -> object:
            type(self).touched = True
            raise AssertionError(f"mapping lookup exposed {secret}: {key}")

        def __iter__(self) -> Iterator[str]:
            type(self).touched = True
            raise AssertionError(f"mapping iteration exposed {secret}")

        def __len__(self) -> int:
            type(self).touched = True
            return WORK_COMPLETION_APPLICATION_MAX_ITEMS + 1

        def __repr__(self) -> str:
            return f"HostileMapping({secret})"

    async def scenario() -> BaseException:
        store = InMemoryTaskStore()
        task = await store.create_task(
            TaskCreate(task_id=f"hostile-{field_name}-task", type="verified-work")
        )
        valid = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=f"hostile-{field_name}-decision",
            idempotency_key=f"hostile-{field_name}-application",
        )
        invalid = valid.model_copy(update={field_name: HostileMapping()})

        with pytest.raises(TypeError, match=field_name) as exc:
            await store.apply_completion_decision(invalid)

        assert HostileMapping.touched is False
        assert await store.load_task(task.id) == task
        assert (
            await store.load_completion_decision_application_receipt(
                task.id,
                valid.idempotency_key,
            )
            is None
        )
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


def test_contract_bound_task_and_application_receipt_are_bounded_together() -> None:
    async def create_base_task() -> Task:
        store = InMemoryTaskStore()
        contract = _contract(contract_id="bounded-task-representation")
        await store.publish_work_contract(contract)
        return await store.create_running_task(
            TaskCreate(
                task_id="bounded-representation-task",
                type="verified-work",
                session_id="session:bounded-representation",
                input={"payload": ""},
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(
                "session:bounded-representation"
            ),
        )

    base = asyncio.run(create_base_task())
    base_document = base.model_dump(mode="python", warnings=False)
    base_bytes = len(
        canonical_durable_json_bytes(
            base.model_dump(mode="json", warnings=False),
            "contract_bound_task",
        )
    )
    payload_bytes = WORK_CONTRACT_TASK_MAX_BYTES - base_bytes
    exact_document = {
        **base_document,
        "input": {"payload": "x" * payload_bytes},
    }
    exact = Task.model_validate(exact_document)
    assert (
        len(
            canonical_durable_json_bytes(
                exact.model_dump(mode="json", warnings=False),
                "contract_bound_task",
            )
        )
        == WORK_CONTRACT_TASK_MAX_BYTES
    )
    receipt = CompletionDecisionApplicationReceipt(
        task_id=exact.id,
        decision_id="bounded-representation-decision",
        idempotency_key="bounded-representation-application",
        request_sha256=_digest("bounded-representation-application"),
        task=exact,
        applied_at=datetime.now(UTC),
    )
    assert (
        len(
            canonical_durable_json_bytes(
                receipt.model_dump(mode="json", warnings=False),
                "completion_decision_application_receipt",
            )
        )
        <= WORK_COMPLETION_APPLICATION_RECEIPT_MAX_BYTES
    )

    exact_document["input"] = {"payload": "x" * (payload_bytes + 1)}
    with pytest.raises(ValidationError, match="Contract-bound task must not exceed"):
        Task.model_validate(exact_document)
    oversized_title_document = {
        **base_document,
        "title": "t" * WORK_CONTRACT_TASK_MAX_BYTES,
    }
    with pytest.raises(ValidationError, match="Contract-bound task must not exceed"):
        Task.model_validate(oversized_title_document)

    async def reject_split_boundary_before_store_mutation() -> None:
        store = InMemoryTaskStore()
        contract = _contract(contract_id="split-bounded-task-representation")
        await store.publish_work_contract(contract)
        request = TaskCreate(
            task_id="split-bounded-representation-task",
            type="verified-work",
            work_contract=contract.reference(),
        ).model_copy(
            update={
                "input": {"left": "l" * 600_000},
                "metadata": {"right": "r" * 600_000},
            },
            deep=True,
        )

        with pytest.raises(
            ValidationError,
            match="Contract-bound task creation request must not exceed",
        ):
            await store.create_task(request)
        assert await store.load_task(request.task_id or "missing-task") is None

    asyncio.run(reject_split_boundary_before_store_mutation())


def test_contract_bound_creation_reserves_claim_lifecycle_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_now = datetime(2026, 8, 20, 12, 34, 56, 123456, tzinfo=UTC)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now if tz is not None else fixed_now.replace(tzinfo=None)

    monkeypatch.setattr(tasks_module, "datetime", FixedDatetime)

    async def scenario() -> None:
        contract = _contract(contract_id="claim-lifecycle-headroom")

        async def create(payload_bytes: int) -> tuple[InMemoryTaskStore, Task]:
            store = InMemoryTaskStore()
            await store.publish_work_contract(contract)
            task = await store.create_task(
                TaskCreate(
                    task_id="claim-lifecycle-headroom-task",
                    type="verified-work",
                    input={"payload": "x" * payload_bytes},
                    work_contract=contract.reference(),
                )
            )
            return store, task

        _, base = await create(0)
        base_bytes = len(
            canonical_durable_json_bytes(
                base.model_dump(mode="json", warnings=False),
                "contract_bound_task_creation_snapshot",
            )
        )
        payload_bytes = WORK_CONTRACT_TASK_CREATION_MAX_BYTES - base_bytes
        store, exact = await create(payload_bytes)
        assert (
            len(
                canonical_durable_json_bytes(
                    exact.model_dump(mode="json", warnings=False),
                    "contract_bound_task_creation_snapshot",
                )
            )
            == WORK_CONTRACT_TASK_CREATION_MAX_BYTES
        )

        # Control characters have the largest canonical JSON expansion among
        # permitted durable linked identities, so this exercises the reserve's
        # worst supported worker-id growth rather than only an ASCII happy path.
        worker_id = "w" + ("\x01" * (WORK_COMPLETION_LINKED_ID_MAX_BYTES - 2)) + "w"
        claimed = await store.claim_task(worker_id)
        assert claimed is not None
        assert claimed.status is TaskStatus.CLAIMED
        assert claimed.worker_id == worker_id
        assert (
            len(
                canonical_durable_json_bytes(
                    claimed.model_dump(mode="json", warnings=False),
                    "contract_bound_claimed_task",
                )
            )
            <= WORK_CONTRACT_TASK_MAX_BYTES
        )

        session_id = "s" + ("\x01" * (WORK_COMPLETION_LINKED_ID_MAX_BYTES - 2)) + "s"
        attached = await store.attach_task(
            claimed.id,
            session_id=session_id,
            session_invocation=await task_backed_session_invocation(
                store,
                claimed.id,
                session_id,
            ),
            worker_id=worker_id,
        )
        assert attached.status is TaskStatus.RUNNING
        assert attached.session_id == session_id
        assert attached.worker_id == worker_id
        assert (
            len(
                canonical_durable_json_bytes(
                    attached.model_dump(mode="json", warnings=False),
                    "contract_bound_attached_task",
                )
            )
            <= WORK_CONTRACT_TASK_MAX_BYTES
        )

        oversized_store = InMemoryTaskStore()
        await oversized_store.publish_work_contract(contract)
        oversized_task_id = "oversized-claim-lifecycle-headroom-task"
        with pytest.raises(
            ValueError,
            match="Contract-bound task creation snapshot must not exceed",
        ):
            await oversized_store.create_task(
                TaskCreate(
                    task_id=oversized_task_id,
                    type="verified-work",
                    input={"payload": "x" * (payload_bytes + 1)},
                    work_contract=contract.reference(),
                )
            )
        assert await oversized_store.load_task(oversized_task_id) is None

    asyncio.run(scenario())


def test_public_contracted_creation_enforces_headroom_for_custom_stores() -> None:
    class RecordingStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.create_called = False

        async def create_task(self, request: TaskCreate) -> Task:
            self.create_called = True
            return await super().create_task(request)

    class InflatingStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def create_task(self, request: TaskCreate) -> Task:
            task = await super().create_task(request)
            base_bytes = len(
                canonical_durable_json_bytes(
                    task.model_dump(mode="json", warnings=False),
                    "custom_store_creation_snapshot",
                )
            )
            inflated = task.model_copy(
                update={
                    "input": {
                        "payload": "x" * (WORK_CONTRACT_TASK_CREATION_MAX_BYTES - base_bytes + 1)
                    }
                },
                deep=True,
            )
            self._tasks[inflated.id] = inflated
            return inflated

    async def scenario() -> None:
        contract = _contract(contract_id="custom-store-creation-headroom")
        recording_store = RecordingStore()
        await recording_store.publish_work_contract(contract)
        recording_app = CayuApp(task_store=recording_store, enable_logging=False)
        oversized_task_id = "custom-store-preflight-headroom-task"
        with pytest.raises(
            ValueError,
            match="Contract-bound task creation snapshot must not exceed",
        ):
            await recording_app.create_task(
                TaskCreate(
                    task_id=oversized_task_id,
                    type="verified-work",
                    input={"payload": "x" * WORK_CONTRACT_TASK_CREATION_MAX_BYTES},
                    work_contract=contract.reference(),
                )
            )
        assert recording_store.create_called is False
        assert await recording_store.load_task(oversized_task_id) is None

        inflating_store = InflatingStore()
        await inflating_store.publish_work_contract(contract)
        inflating_app = CayuApp(task_store=inflating_store, enable_logging=False)
        inflated_task_id = "custom-store-return-headroom-task"
        with pytest.raises(
            WorkContractConflict,
            match="outside the creation-snapshot bounds",
        ):
            await inflating_app.create_task(
                TaskCreate(
                    task_id=inflated_task_id,
                    type="verified-work",
                    work_contract=contract.reference(),
                )
            )
        persisted = await inflating_store.load_task(inflated_task_id)
        assert persisted is not None
        assert (
            len(
                canonical_durable_json_bytes(
                    persisted.model_dump(mode="json", warnings=False),
                    "nonconforming_custom_store_creation_snapshot",
                )
            )
            > WORK_CONTRACT_TASK_CREATION_MAX_BYTES
        )

    asyncio.run(scenario())


def test_in_memory_verified_work_lifecycle_rejects_then_accepts_exactly() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        assert await store.publish_work_contract(contract) == contract
        assert await store.publish_work_contract(contract) == contract
        assert await store.load_work_contract(contract.reference()) == contract

        task = await store.create_running_task(
            TaskCreate(
                task_id="contracted-task",
                type="bid",
                session_id="session:bid:1",
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding("session:bid:1"),
        )
        assert task.work_contract == contract.reference()
        assert copy_task(task).work_contract == contract.reference()

        first_attempt_request = WorkAttemptCreate(
            attempt_id="attempt-1",
            task_id=task.id,
            session_id="session:bid:1",
            contract=contract.reference(),
            execution_profile_fingerprint=_digest("worker-profile"),
        )
        first_attempt = await store.begin_work_attempt(first_attempt_request)
        assert first_attempt.ordinal == 1
        assert await store.begin_work_attempt(first_attempt_request) == first_attempt

        with pytest.raises(WorkCompletionConflict, match="another request"):
            await store.begin_work_attempt(
                first_attempt_request.model_copy(
                    update={"execution_profile_fingerprint": _digest("other-profile")}
                )
            )

        first_proposal_request = CompletionProposalCreate(
            proposal_id="proposal-1",
            attempt_id=first_attempt.attempt_id,
            result=_result_reference(),
            evidence_references=(_artifact_evidence(),),
        )
        first_proposal = await store.submit_completion_proposal(first_proposal_request)
        assert await store.submit_completion_proposal(first_proposal_request) == first_proposal

        with pytest.raises(WorkCompletionConflict, match="another request"):
            await store.submit_completion_proposal(
                first_proposal_request.model_copy(update={"result": _result_reference("other")})
            )

        first_claim_request = CompletionVerificationClaimRequest(
            claim_id="claim-1",
            proposal_id=first_proposal.proposal_id,
            worker_id="verifier-worker",
            verifier=contract.verifier,
        )
        first_claim = await store.claim_completion_verification(first_claim_request)
        assert await store.claim_completion_verification(first_claim_request) == first_claim

        with pytest.raises(WorkCompletionConflict, match="another request"):
            await store.claim_completion_verification(
                first_claim_request.model_copy(update={"worker_id": "other-verifier"})
            )

        rejected_request = _rejected_decision(
            proposal_id=first_proposal.proposal_id,
            claim_id=first_claim.claim_id,
            worker_id=first_claim.worker_id,
        )
        rejected = await store.record_completion_decision(rejected_request)
        assert await store.record_completion_decision(rejected_request) == rejected
        assert await store.claim_completion_verification(first_claim_request) == first_claim

        with pytest.raises(WorkCompletionConflict, match="another request"):
            await store.record_completion_decision(
                _rejected_decision(
                    proposal_id=first_proposal.proposal_id,
                    claim_id=first_claim.claim_id,
                    worker_id=first_claim.worker_id,
                    gap_summary="Conflicting explanation under the same decision identity.",
                )
            )

        rejected_application = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=rejected.decision_id,
            idempotency_key="apply-rejected",
        )
        still_running = await store.apply_completion_decision(rejected_application)
        assert still_running.status is TaskStatus.RUNNING
        assert await store.apply_completion_decision(rejected_application) == still_running

        second_attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id="attempt-2",
                task_id=task.id,
                session_id="session:bid:1",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("worker-profile"),
            )
        )
        assert second_attempt.ordinal == 2
        second_proposal = await store.submit_completion_proposal(
            CompletionProposalCreate(
                proposal_id="proposal-2",
                attempt_id=second_attempt.attempt_id,
                result=_result_reference("2"),
                evidence_references=(_artifact_evidence(), _approval_evidence()),
            )
        )
        second_claim = await store.claim_completion_verification(
            CompletionVerificationClaimRequest(
                claim_id="claim-2",
                proposal_id=second_proposal.proposal_id,
                worker_id="verifier-worker",
                verifier=contract.verifier,
            )
        )
        accepted_request = _accepted_decision(
            proposal_id=second_proposal.proposal_id,
            claim_id=second_claim.claim_id,
            worker_id=second_claim.worker_id,
        )
        accepted = await store.record_completion_decision(accepted_request)
        with pytest.raises(WorkCompletionConflict, match="accepted proposal"):
            await store.apply_completion_decision(
                CompletionDecisionApplicationRequest(
                    task_id=task.id,
                    decision_id=accepted.decision_id,
                    idempotency_key="apply-wrong-result",
                    result=_task_result("other"),
                    result_reference=_result_reference("other"),
                )
            )
        for suffix, substituted_reference in (
            (
                "kind",
                second_proposal.result.model_copy(update={"kind": "task.other-result"}),
            ),
            (
                "identity",
                second_proposal.result.model_copy(update={"reference_id": "result:substituted"}),
            ),
        ):
            with pytest.raises(WorkCompletionConflict, match="accepted proposal"):
                await store.apply_completion_decision(
                    CompletionDecisionApplicationRequest(
                        task_id=task.id,
                        decision_id=accepted.decision_id,
                        idempotency_key=f"apply-wrong-result-{suffix}",
                        result=_task_result("2"),
                        result_reference=substituted_reference,
                    )
                )
        unchanged = await store.load_task(task.id)
        assert unchanged is not None
        assert unchanged.status is TaskStatus.RUNNING
        assert (
            await store.load_completion_decision_application_receipt(
                task.id,
                "apply-wrong-result",
            )
            is None
        )
        completed_request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=accepted.decision_id,
            idempotency_key="apply-accepted",
            result=_task_result("2"),
            result_reference=second_proposal.result,
        )
        completed = await store.apply_completion_decision(completed_request)

        assert completed.status is TaskStatus.COMPLETED
        assert completed.result == _task_result("2")
        assert await store.apply_completion_decision(completed_request) == completed
        receipt = await store.load_completion_decision_application_receipt(
            task.id,
            "apply-accepted",
        )
        assert receipt is not None
        assert receipt.decision_id == accepted.decision_id
        assert receipt.task == completed

        with pytest.raises(WorkCompletionConflict, match="another request"):
            await store.apply_completion_decision(
                completed_request.model_copy(
                    update={
                        "result": _task_result("different"),
                        "result_reference": _result_reference("different"),
                    }
                )
            )

    asyncio.run(scenario())


def test_verified_work_lifecycle_rejects_conflicting_shared_evidence_authority() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _shared_evidence_contract()
        await store.publish_work_contract(contract)
        task = await store.create_running_task(
            TaskCreate(
                task_id="shared-evidence-task",
                type="artifact-check",
                session_id="session:shared-evidence",
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding("session:shared-evidence"),
        )
        attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id="shared-evidence-attempt",
                task_id=task.id,
                session_id="session:shared-evidence",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("shared-evidence-profile"),
            )
        )
        result = _task_result("shared-evidence")
        proposal = await store.submit_completion_proposal(
            CompletionProposalCreate(
                proposal_id="shared-evidence-proposal",
                attempt_id=attempt.attempt_id,
                result=_result_reference("shared-evidence", result=result),
            )
        )
        claim = await store.claim_completion_verification(
            CompletionVerificationClaimRequest(
                claim_id="shared-evidence-claim",
                proposal_id=proposal.proposal_id,
                worker_id="shared-evidence-worker",
                verifier=contract.verifier,
            )
        )
        decision = _shared_evidence_decision(
            proposal_id=proposal.proposal_id,
            claim_id=claim.claim_id,
            worker_id=claim.worker_id,
        )
        contents_outcome = decision.criterion_outcomes[1]
        conflicting = decision.model_copy(
            update={
                "criterion_outcomes": (
                    decision.criterion_outcomes[0],
                    contents_outcome.model_copy(
                        update={
                            "evidence_references": (
                                contents_outcome.evidence_references[0].model_copy(
                                    update={"digest": _digest("conflicting-artifact")}
                                ),
                            )
                        }
                    ),
                )
            }
        )

        with pytest.raises(ValidationError, match="conflicting representations"):
            await store.record_completion_decision(conflicting)
        assert await store.load_completion_decision(decision.decision_id) is None
        unchanged = await store.load_task(task.id)
        assert unchanged is not None
        assert unchanged.status is TaskStatus.RUNNING

        accepted = await store.record_completion_decision(decision)
        completed = await store.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=accepted.decision_id,
                idempotency_key="apply-shared-evidence",
                result=result,
                result_reference=proposal.result,
            )
        )
        assert completed.status is TaskStatus.COMPLETED
        assert completed.result == result

    asyncio.run(scenario())


def test_verified_work_lifecycle_preserves_valid_inherited_task_and_session_identities() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract(contract_id="long-inherited-identities")
        await store.publish_work_contract(contract)
        task_id = "task:" + ("t" * 300)
        session_id = "session:" + ("s" * 300)
        task = await store.create_running_task(
            TaskCreate(
                task_id=task_id,
                type="verified-work",
                session_id=session_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(session_id),
        )

        attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id="long-inherited-identities-attempt",
                task_id=task.id,
                session_id=session_id,
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("worker-profile"),
            )
        )

        assert attempt.task_id == task_id
        assert attempt.session_id == session_id

        result = {"long-identities": True}
        proposal = await store.submit_completion_proposal(
            CompletionProposalCreate(
                proposal_id="long-inherited-identities-proposal",
                attempt_id=attempt.attempt_id,
                result=_result_reference("long-identities", result=result),
                evidence_references=(_artifact_evidence(), _approval_evidence()),
            )
        )
        claim = await store.claim_completion_verification(
            CompletionVerificationClaimRequest(
                claim_id="long-inherited-identities-claim",
                proposal_id=proposal.proposal_id,
                worker_id="verifier-worker",
                verifier=contract.verifier,
            )
        )
        decision = await store.record_completion_decision(
            _accepted_decision(
                proposal_id=proposal.proposal_id,
                claim_id=claim.claim_id,
                worker_id=claim.worker_id,
            )
        )
        completed = await store.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=decision.decision_id,
                idempotency_key="long-inherited-identities-application",
                result=result,
                result_reference=proposal.result,
            )
        )
        assert completed.status is TaskStatus.COMPLETED
        assert completed.id == task_id

    asyncio.run(scenario())


def test_oversized_contracted_identity_is_rejected_before_task_publication() -> None:
    contract = _contract(contract_id="bounded-contracted-identities")
    oversized_task_id = "t" * 2049

    with pytest.raises(ValidationError, match="when bound to a work contract"):
        TaskCreate(
            task_id=oversized_task_id,
            type="verified-work",
            work_contract=contract.reference(),
        )

    ordinary = TaskCreate(task_id=oversized_task_id, type="ordinary-work")
    assert ordinary.task_id == oversized_task_id


def test_contracted_task_transitions_require_a_bounded_session_binding() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract(contract_id="bounded-transition-session")
        await store.publish_work_contract(contract)
        pending = await store.create_task(
            TaskCreate(
                task_id="bounded-transition-pending",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )

        with pytest.raises(WorkCompletionConflict, match="session binding"):
            await store.start_task(pending.id)
        unchanged = await store.load_task(pending.id)
        assert unchanged is not None
        assert unchanged.status is TaskStatus.PENDING
        assert unchanged.session_id is None

        oversized_session_id = "s" * (WORK_COMPLETION_LINKED_ID_MAX_BYTES + 1)
        with pytest.raises(ValueError, match="when bound to a work contract"):
            await store.start_task(
                pending.id,
                session_id=oversized_session_id,
                session_invocation=await task_backed_session_invocation(
                    store,
                    pending.id,
                    oversized_session_id,
                ),
            )
        unchanged = await store.load_task(pending.id)
        assert unchanged is not None
        assert unchanged.status is TaskStatus.PENDING
        assert unchanged.session_id is None

        exact_session_id = "s" * WORK_COMPLETION_LINKED_ID_MAX_BYTES
        started = await store.start_task(
            pending.id,
            session_id=exact_session_id,
            session_invocation=await task_backed_session_invocation(
                store,
                pending.id,
                exact_session_id,
            ),
        )
        assert started.status is TaskStatus.RUNNING
        assert started.session_id == exact_session_id

        claimable = await store.create_task(
            TaskCreate(
                task_id="bounded-transition-claimed",
                type="claimed-verified-work",
                work_contract=contract.reference(),
            )
        )
        claimed = await store.claim_task(
            "bounded-transition-worker",
            TaskQuery(type=claimable.type),
            lease_seconds=30,
        )
        assert claimed is not None
        with pytest.raises(ValueError, match="when bound to a work contract"):
            await store.attach_task(
                claimed.id,
                session_id=oversized_session_id,
                session_invocation=await task_backed_session_invocation(
                    store,
                    claimed.id,
                    oversized_session_id,
                ),
                worker_id="bounded-transition-worker",
            )
        unchanged_claim = await store.load_task(claimed.id)
        assert unchanged_claim is not None
        assert unchanged_claim.status is TaskStatus.CLAIMED
        assert unchanged_claim.session_id is None
        assert unchanged_claim.worker_id == "bounded-transition-worker"

    asyncio.run(scenario())


def test_task_worker_lease_uses_wall_clock_not_verification_clock() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore(clock=lambda: datetime(2100, 1, 1, tzinfo=UTC))
        contract = _contract(contract_id="separate-clock-domains")
        await store.publish_work_contract(contract)
        task = await store.create_task(
            TaskCreate(
                task_id="separate-clock-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        claimed = await store.claim_task(
            "task-worker",
            TaskQuery(type=task.type),
            lease_seconds=30,
        )
        assert claimed is not None
        running = await store.attach_task(
            task.id,
            session_id="session:separate-clock",
            session_invocation=await task_backed_session_invocation(
                store,
                task.id,
                "session:separate-clock",
            ),
            worker_id="task-worker",
        )

        attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id="separate-clock-attempt",
                task_id=task.id,
                session_id=running.session_id or "missing",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("worker-profile"),
                worker_id="task-worker",
            )
        )

        assert attempt.started_at == datetime(2100, 1, 1, tzinfo=UTC)
        assert attempt.worker_id == "task-worker"

    asyncio.run(scenario())


def test_durable_decision_applies_after_originating_task_lease_expires() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract(contract_id="decision-after-worker-expiry")
        await store.publish_work_contract(contract)
        task = await store.create_task(
            TaskCreate(
                task_id="decision-after-worker-expiry-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        claimed = await store.claim_task(
            "task-worker",
            TaskQuery(type=task.type),
            lease_seconds=1,
        )
        assert claimed is not None
        running = await store.attach_task(
            task.id,
            session_id="session:decision-after-worker-expiry",
            session_invocation=await task_backed_session_invocation(
                store,
                task.id,
                "session:decision-after-worker-expiry",
            ),
            worker_id="task-worker",
        )
        attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id="decision-after-worker-expiry-attempt",
                task_id=task.id,
                session_id=running.session_id or "missing",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("worker-profile"),
                worker_id="task-worker",
            )
        )
        result = {"accepted": True}
        proposal = await store.submit_completion_proposal(
            CompletionProposalCreate(
                proposal_id="decision-after-worker-expiry-proposal",
                attempt_id=attempt.attempt_id,
                result=_result_reference("expired-worker", result=result),
                evidence_references=(_artifact_evidence(), _approval_evidence()),
            )
        )
        claim = await store.claim_completion_verification(
            CompletionVerificationClaimRequest(
                claim_id="decision-after-worker-expiry-claim",
                proposal_id=proposal.proposal_id,
                worker_id="verifier-worker",
                verifier=contract.verifier,
            )
        )
        decision = await store.record_completion_decision(
            _accepted_decision(
                proposal_id=proposal.proposal_id,
                claim_id=claim.claim_id,
                worker_id=claim.worker_id,
            )
        )
        assert running.lease_expires_at is not None
        await asyncio.sleep(
            max(0.0, (running.lease_expires_at - datetime.now(UTC)).total_seconds()) + 0.02
        )

        completed = await store.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=decision.decision_id,
                idempotency_key="apply-after-worker-expiry",
                result=result,
                result_reference=proposal.result,
            )
        )

        assert completed.status is TaskStatus.COMPLETED
        assert completed.worker_id is None
        assert completed.result == result

    asyncio.run(scenario())


def test_rejected_continue_decision_fences_live_attempt_worker() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract(contract_id="continue-fences-attempt-worker")
        await store.publish_work_contract(contract)
        task = await store.create_task(
            TaskCreate(
                task_id="continue-fences-attempt-worker-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        claimed = await store.claim_task(
            "originating-worker",
            TaskQuery(type=task.type),
            lease_seconds=60,
        )
        assert claimed is not None
        running = await store.attach_task(
            task.id,
            session_id="session:continue-fences-attempt-worker",
            session_invocation=await task_backed_session_invocation(
                store,
                task.id,
                "session:continue-fences-attempt-worker",
            ),
            worker_id="originating-worker",
        )
        attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id="continue-fences-attempt-worker-attempt-1",
                task_id=task.id,
                session_id=running.session_id or "missing-session",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("worker-profile-1"),
                worker_id="originating-worker",
            )
        )
        proposal = await store.submit_completion_proposal(
            CompletionProposalCreate(
                proposal_id="continue-fences-attempt-worker-proposal-1",
                attempt_id=attempt.attempt_id,
                result=_result_reference("continue-fences-worker-1"),
            )
        )
        claim = await store.claim_completion_verification(
            CompletionVerificationClaimRequest(
                claim_id="continue-fences-attempt-worker-claim-1",
                proposal_id=proposal.proposal_id,
                worker_id="verifier-worker",
                verifier=contract.verifier,
            )
        )
        decision = await store.record_completion_decision(
            _rejected_decision(
                proposal_id=proposal.proposal_id,
                claim_id=claim.claim_id,
                worker_id=claim.worker_id,
                decision_id="continue-fences-attempt-worker-decision-1",
            )
        )

        continued = await store.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=decision.decision_id,
                idempotency_key="continue-fences-attempt-worker-application-1",
            )
        )

        assert continued.status is TaskStatus.RUNNING
        assert continued.worker_id is None
        assert continued.lease_expires_at is None
        with pytest.raises(TaskClaimLost):
            await store.heartbeat(task.id, "originating-worker")
        with pytest.raises(TaskClaimLost):
            await store.release_attached_task_worker(task.id, "originating-worker")

        next_attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id="continue-fences-attempt-worker-attempt-2",
                task_id=task.id,
                session_id=continued.session_id or "missing-session",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("worker-profile-2"),
            )
        )
        assert next_attempt.ordinal == 2
        assert next_attempt.worker_id is None

    asyncio.run(scenario())


def test_accepted_decision_requires_exact_constraint_and_available_evidence_coverage() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        await store.publish_work_contract(contract)
        task = await store.create_running_task(
            TaskCreate(
                task_id="evidence-gated-task",
                type="bid",
                session_id="session:evidence-gated",
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding("session:evidence-gated"),
        )
        attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id="evidence-gated-attempt",
                task_id=task.id,
                session_id="session:evidence-gated",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("worker-profile"),
            )
        )
        proposal = await store.submit_completion_proposal(
            CompletionProposalCreate(
                proposal_id="evidence-gated-proposal",
                attempt_id=attempt.attempt_id,
                result=_result_reference("evidence-gated"),
                evidence_references=(_artifact_evidence(), _approval_evidence()),
            )
        )
        claim = await store.claim_completion_verification(
            CompletionVerificationClaimRequest(
                claim_id="evidence-gated-claim",
                proposal_id=proposal.proposal_id,
                worker_id="verifier-worker",
                verifier=contract.verifier,
            )
        )
        accepted = _accepted_decision(
            proposal_id=proposal.proposal_id,
            claim_id=claim.claim_id,
            worker_id=claim.worker_id,
        )

        missing_approval = CompletionCriterionOutcome(
            criterion_id="approval",
            status=CriterionOutcomeStatus.SATISFIED,
            reason_code="approval.asserted",
            satisfaction_basis=CompletionSatisfactionBasis.VERIFIER_ASSERTION,
        )
        with pytest.raises(WorkCompletionConflict, match="required evidence"):
            await store.record_completion_decision(
                accepted.model_copy(
                    update={
                        "criterion_outcomes": (
                            accepted.criterion_outcomes[0],
                            missing_approval,
                        )
                    }
                )
            )

        unavailable_approval = WorkEvidenceReference(
            kind="business.approval",
            reference_id="approval:unavailable",
            requirement_id="approval",
            available=False,
            unavailable_reason="store.unavailable",
        )
        supplemental_evidence = WorkEvidenceReference(
            kind="verifier.diagnostic",
            reference_id="diagnostic:available",
        )
        with pytest.raises(WorkCompletionConflict, match="required evidence"):
            await store.record_completion_decision(
                accepted.model_copy(
                    update={
                        "criterion_outcomes": (
                            accepted.criterion_outcomes[0],
                            CompletionCriterionOutcome(
                                criterion_id="approval",
                                status=CriterionOutcomeStatus.SATISFIED,
                                reason_code="approval.unavailable",
                                satisfaction_basis=CompletionSatisfactionBasis.EVIDENCE,
                                evidence_references=(
                                    unavailable_approval,
                                    supplemental_evidence,
                                ),
                            ),
                        )
                    }
                )
            )

        wrong_kind = WorkEvidenceReference(
            kind="artifact.version",
            reference_id="artifact:not-an-approval",
            requirement_id="approval",
        )
        with pytest.raises(WorkCompletionConflict, match="frozen contract requirements"):
            await store.record_completion_decision(
                accepted.model_copy(
                    update={
                        "criterion_outcomes": (
                            accepted.criterion_outcomes[0],
                            CompletionCriterionOutcome(
                                criterion_id="approval",
                                status=CriterionOutcomeStatus.SATISFIED,
                                reason_code="approval.wrong-kind",
                                satisfaction_basis=CompletionSatisfactionBasis.EVIDENCE,
                                evidence_references=(wrong_kind,),
                            ),
                        )
                    }
                )
            )

        missing_constraint_evidence = CompletionConstraintOutcome(
            constraint_id="no-unapproved-send",
            status=CriterionOutcomeStatus.SATISFIED,
            reason_code="constraint.asserted",
            satisfaction_basis=CompletionSatisfactionBasis.VERIFIER_ASSERTION,
        )
        with pytest.raises(WorkCompletionConflict, match="required evidence"):
            await store.record_completion_decision(
                accepted.model_copy(update={"constraint_outcomes": (missing_constraint_evidence,)})
            )

        with pytest.raises(WorkCompletionConflict, match="every contract constraint"):
            await store.record_completion_decision(
                accepted.model_copy(update={"constraint_outcomes": ()})
            )

        decision = await store.record_completion_decision(accepted)
        assert decision.verdict is CompletionVerdict.ACCEPTED

    asyncio.run(scenario())


def test_explicit_verifier_assertions_cover_outcomes_without_evidence_requirements() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = work_contract_from_draft(
            WorkContractDraft(
                contract_id="assertion-contract",
                version=1,
                objective="Verify an application-owned domain fact.",
                criteria=(
                    WorkCriterion(
                        criterion_id="domain-ready",
                        ordinal=1,
                        description="The application readiness rule passes.",
                    ),
                ),
                constraints=(
                    WorkConstraint(
                        constraint_id="domain-safe",
                        description="The application safety rule passes.",
                    ),
                ),
                verifier=_verifier(),
            )
        )
        await store.publish_work_contract(contract)
        task = await store.create_running_task(
            TaskCreate(
                task_id="assertion-task",
                type="assertion",
                session_id="session:assertion",
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding("session:assertion"),
        )
        attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id="assertion-attempt",
                task_id=task.id,
                session_id="session:assertion",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("worker-profile"),
            )
        )
        proposal = await store.submit_completion_proposal(
            CompletionProposalCreate(
                proposal_id="assertion-proposal",
                attempt_id=attempt.attempt_id,
                result=_result_reference("assertion"),
            )
        )
        claim = await store.claim_completion_verification(
            CompletionVerificationClaimRequest(
                claim_id="assertion-claim",
                proposal_id=proposal.proposal_id,
                worker_id="verifier-worker",
                verifier=contract.verifier,
            )
        )
        decision = await store.record_completion_decision(
            CompletionDecisionCreate(
                decision_id="assertion-decision",
                proposal_id=proposal.proposal_id,
                claim_id=claim.claim_id,
                worker_id=claim.worker_id,
                verifier=contract.verifier,
                verdict=CompletionVerdict.ACCEPTED,
                criterion_outcomes=(
                    CompletionCriterionOutcome(
                        criterion_id="domain-ready",
                        status=CriterionOutcomeStatus.SATISFIED,
                        reason_code="domain.ready",
                        satisfaction_basis=(CompletionSatisfactionBasis.VERIFIER_ASSERTION),
                    ),
                ),
                constraint_outcomes=(
                    CompletionConstraintOutcome(
                        constraint_id="domain-safe",
                        status=CriterionOutcomeStatus.SATISFIED,
                        reason_code="domain.safe",
                        satisfaction_basis=(CompletionSatisfactionBasis.VERIFIER_ASSERTION),
                    ),
                ),
            )
        )
        assert decision.verdict is CompletionVerdict.ACCEPTED

    asyncio.run(scenario())


def test_rejection_policy_interrupts_and_enforces_attempt_and_repeated_gap_limits() -> None:
    async def reject_attempt(
        store: InMemoryTaskStore,
        contract: WorkContract,
        task: Task,
        *,
        ordinal: int,
        artifact_version: str = "7",
    ) -> Task:
        attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id=f"{task.id}-attempt-{ordinal}",
                task_id=task.id,
                session_id=task.session_id or "missing-session",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("worker-profile"),
            )
        )
        proposal = await store.submit_completion_proposal(
            CompletionProposalCreate(
                proposal_id=f"{task.id}-proposal-{ordinal}",
                attempt_id=attempt.attempt_id,
                result=_result_reference(f"{task.id}-{ordinal}"),
            )
        )
        claim = await store.claim_completion_verification(
            CompletionVerificationClaimRequest(
                claim_id=f"{task.id}-claim-{ordinal}",
                proposal_id=proposal.proposal_id,
                worker_id="verifier-worker",
                verifier=contract.verifier,
            )
        )
        decision = await store.record_completion_decision(
            _rejected_decision(
                proposal_id=proposal.proposal_id,
                claim_id=claim.claim_id,
                worker_id=claim.worker_id,
                decision_id=f"{task.id}-decision-{ordinal}",
                artifact_version=artifact_version,
            )
        )
        return await store.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=decision.decision_id,
                idempotency_key=f"{task.id}-apply-{ordinal}",
            )
        )

    async def running_task(
        store: InMemoryTaskStore,
        contract: WorkContract,
        task_id: str,
    ) -> Task:
        await store.publish_work_contract(contract)
        session_id = f"session:{task_id}"
        return await store.create_running_task(
            TaskCreate(
                task_id=task_id,
                type="bid",
                session_id=session_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(session_id),
        )

    async def scenario() -> None:
        interrupt_store = InMemoryTaskStore()
        interrupt_contract = _contract(
            contract_id="interrupt-contract",
            continuation_policy=CompletionContinuationPolicy(
                rejection_action=CompletionRejectionAction.INTERRUPT,
                max_attempts=3,
                max_repeated_gap_count=2,
            ),
        )
        interrupt_task = await running_task(
            interrupt_store,
            interrupt_contract,
            "interrupt-task",
        )
        interrupted = await reject_attempt(
            interrupt_store,
            interrupt_contract,
            interrupt_task,
            ordinal=1,
        )
        assert interrupted.status is TaskStatus.PAUSED
        assert interrupted.status_reason == "work_contract_rejected"

        attempt_limit_store = InMemoryTaskStore()
        attempt_limit_contract = _contract(
            contract_id="attempt-limit-contract",
            continuation_policy=CompletionContinuationPolicy(
                rejection_action=CompletionRejectionAction.CONTINUE,
                max_attempts=1,
                max_repeated_gap_count=2,
            ),
        )
        attempt_limit_task = await running_task(
            attempt_limit_store,
            attempt_limit_contract,
            "attempt-limit-task",
        )
        attempt_limited = await reject_attempt(
            attempt_limit_store,
            attempt_limit_contract,
            attempt_limit_task,
            ordinal=1,
        )
        assert attempt_limited.status is TaskStatus.NEEDS_ATTENTION
        assert attempt_limited.status_reason == "work_contract_attempt_limit"
        resumed = await attempt_limit_store.resume_task(attempt_limited.id)
        restarted = await attempt_limit_store.start_task(
            resumed.id,
            session_id=resumed.session_id,
            session_invocation=await task_backed_session_invocation(
                attempt_limit_store,
                resumed.id,
                resumed.session_id or "missing-session",
            ),
        )
        with pytest.raises(WorkCompletionConflict, match="attempt limit"):
            await attempt_limit_store.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id="attempt-limit-task-attempt-2",
                    task_id=restarted.id,
                    session_id=restarted.session_id or "missing-session",
                    contract=attempt_limit_contract.reference(),
                    execution_profile_fingerprint=_digest("worker-profile"),
                )
            )

        repeated_gap_store = InMemoryTaskStore()
        repeated_gap_contract = _contract(
            contract_id="repeated-gap-contract",
            continuation_policy=CompletionContinuationPolicy(
                rejection_action=CompletionRejectionAction.CONTINUE,
                max_attempts=3,
                max_repeated_gap_count=1,
            ),
        )
        repeated_gap_task = await running_task(
            repeated_gap_store,
            repeated_gap_contract,
            "repeated-gap-task",
        )
        first_rejection = await reject_attempt(
            repeated_gap_store,
            repeated_gap_contract,
            repeated_gap_task,
            ordinal=1,
        )
        assert first_rejection.status is TaskStatus.RUNNING
        repeated = await reject_attempt(
            repeated_gap_store,
            repeated_gap_contract,
            first_rejection,
            ordinal=2,
            artifact_version="8",
        )
        assert repeated.status is TaskStatus.NEEDS_ATTENTION
        assert repeated.status_reason == "work_contract_repeated_gap_limit"

    asyncio.run(scenario())


def test_decision_application_prepares_receipt_before_publishing_task_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        now = datetime(2026, 8, 19, tzinfo=UTC)
        lifecycle_clock_fails = [False]

        class ApplicationDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                if lifecycle_clock_fails[0]:
                    raise RuntimeError("injected application lifecycle clock failure")
                return now if tz is not None else now.replace(tzinfo=None)

        monkeypatch.setattr(tasks_module, "datetime", ApplicationDatetime)
        store = InMemoryTaskStore(clock=lambda: now)
        contract = _contract()
        await store.publish_work_contract(contract)
        task = await store.create_running_task(
            TaskCreate(
                task_id="atomic-application-task",
                type="bid",
                session_id="session:atomic-application",
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(
                "session:atomic-application"
            ),
        )
        attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id="atomic-application-attempt",
                task_id=task.id,
                session_id="session:atomic-application",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("worker-profile"),
            )
        )
        accepted_result: dict[str, object] = {"accepted": True}
        proposal = await store.submit_completion_proposal(
            CompletionProposalCreate(
                proposal_id="atomic-application-proposal",
                attempt_id=attempt.attempt_id,
                result=_result_reference("atomic-application", result=accepted_result),
                evidence_references=(_artifact_evidence(), _approval_evidence()),
            )
        )
        claim = await store.claim_completion_verification(
            CompletionVerificationClaimRequest(
                claim_id="atomic-application-claim",
                proposal_id=proposal.proposal_id,
                worker_id="verifier-worker",
                verifier=contract.verifier,
            )
        )
        decision = await store.record_completion_decision(
            _accepted_decision(
                proposal_id=proposal.proposal_id,
                claim_id=claim.claim_id,
                worker_id=claim.worker_id,
            )
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision.decision_id,
            idempotency_key="atomic-application",
            result=accepted_result,
            result_reference=proposal.result,
        )

        lifecycle_clock_fails[0] = True
        with pytest.raises(RuntimeError, match="injected application lifecycle clock failure"):
            await store.apply_completion_decision(request)
        unchanged = await store.load_task(task.id)
        assert unchanged is not None
        assert unchanged.status is TaskStatus.RUNNING
        assert unchanged.result is None
        assert (
            await store.load_completion_decision_application_receipt(
                task.id,
                request.idempotency_key,
            )
            is None
        )

        lifecycle_clock_fails[0] = False
        completed = await store.apply_completion_decision(request)
        assert completed.status is TaskStatus.COMPLETED
        assert completed.result == {"accepted": True}

    asyncio.run(scenario())


def test_oversized_completed_task_rejects_before_task_or_receipt_mutation() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract(contract_id="bounded-application-transition")
        await store.publish_work_contract(contract)
        task = await store.create_running_task(
            TaskCreate(
                task_id="bounded-application-transition-task",
                type="verified-work",
                session_id="session:bounded-application-transition",
                input={"source": "s" * 700_000},
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(
                "session:bounded-application-transition"
            ),
        )
        attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id="bounded-application-transition-attempt",
                task_id=task.id,
                session_id=task.session_id or "missing-session",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("bounded-application-profile"),
            )
        )
        result: dict[str, object] = {"output": "r" * 400_000}
        proposal = await store.submit_completion_proposal(
            CompletionProposalCreate(
                proposal_id="bounded-application-transition-proposal",
                attempt_id=attempt.attempt_id,
                result=_result_reference("bounded-transition", result=result),
                evidence_references=(_artifact_evidence(), _approval_evidence()),
            )
        )
        claim = await store.claim_completion_verification(
            CompletionVerificationClaimRequest(
                claim_id="bounded-application-transition-claim",
                proposal_id=proposal.proposal_id,
                worker_id="bounded-application-verifier",
                verifier=contract.verifier,
            )
        )
        decision = await store.record_completion_decision(
            _accepted_decision(
                proposal_id=proposal.proposal_id,
                claim_id=claim.claim_id,
                worker_id=claim.worker_id,
            )
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision.decision_id,
            idempotency_key="bounded-application-transition",
            result=result,
            result_reference=proposal.result,
        )

        with pytest.raises(ValidationError, match="Contract-bound task must not exceed"):
            await store.apply_completion_decision(request)

        assert await store.load_task(task.id) == task
        assert (
            await store.load_completion_decision_application_receipt(
                task.id,
                request.idempotency_key,
            )
            is None
        )

    asyncio.run(scenario())


def test_decision_application_receipt_requires_exact_embedded_task_identity() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        expected = await store.create_task(
            TaskCreate(task_id="receipt-expected-task", type="verified-work")
        )
        substituted = await store.create_task(
            TaskCreate(task_id="receipt-substituted-task", type="verified-work")
        )

        with pytest.raises(
            ValidationError,
            match="Decision-application receipt conflicts with its task",
        ):
            CompletionDecisionApplicationReceipt(
                task_id=expected.id,
                decision_id="receipt-decision",
                idempotency_key="receipt-task-mismatch",
                request_sha256=_digest("receipt-task-mismatch"),
                task=substituted,
                applied_at=datetime.now(UTC),
            )
        with pytest.raises(ValidationError, match="requires a contract-bound task"):
            CompletionDecisionApplicationReceipt(
                task_id=expected.id,
                decision_id="receipt-decision",
                idempotency_key="receipt-uncontracted-task",
                request_sha256=_digest("receipt-uncontracted-task"),
                task=expected,
                applied_at=datetime.now(UTC),
            )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("verdict", "expected_status"),
    [
        (CompletionVerdict.REJECTED, TaskStatus.PAUSED),
        (CompletionVerdict.BLOCKED, TaskStatus.BLOCKED),
        (CompletionVerdict.NEEDS_REVIEW, TaskStatus.NEEDS_ATTENTION),
    ],
)
def test_in_memory_decision_application_holds_attached_tasks(
    verdict: CompletionVerdict,
    expected_status: TaskStatus,
) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract(
            continuation_policy=CompletionContinuationPolicy(
                rejection_action=CompletionRejectionAction.INTERRUPT,
                max_attempts=3,
                max_repeated_gap_count=2,
            )
        )
        await store.publish_work_contract(contract)
        task = await store.create_task(
            TaskCreate(
                task_id=f"held-{verdict.value}",
                type=f"held-{verdict.value}",
                work_contract=contract.reference(),
            )
        )
        claimed = await store.claim_task(
            "task-worker",
            TaskQuery(type=f"held-{verdict.value}"),
        )
        assert claimed is not None
        running = await store.attach_task(
            task.id,
            session_id=f"session:{verdict.value}",
            session_invocation=await task_backed_session_invocation(
                store,
                task.id,
                f"session:{verdict.value}",
            ),
            worker_id="task-worker",
        )
        attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id=f"attempt-{verdict.value}",
                task_id=task.id,
                session_id=running.session_id or "missing",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("worker-profile"),
                worker_id="task-worker",
            )
        )
        proposal = await store.submit_completion_proposal(
            CompletionProposalCreate(
                proposal_id=f"proposal-{verdict.value}",
                attempt_id=attempt.attempt_id,
                result=_result_reference(verdict.value),
            )
        )
        claim = await store.claim_completion_verification(
            CompletionVerificationClaimRequest(
                claim_id=f"claim-{verdict.value}",
                proposal_id=proposal.proposal_id,
                worker_id="verifier-worker",
                verifier=contract.verifier,
            )
        )
        decision = await store.record_completion_decision(
            _held_decision(
                verdict=verdict,
                proposal_id=proposal.proposal_id,
                claim_id=claim.claim_id,
                worker_id=claim.worker_id,
            )
        )
        held = await store.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=decision.decision_id,
                idempotency_key=f"apply-{verdict.value}",
            )
        )

        assert held.status is expected_status
        assert held.session_id == running.session_id
        assert held.worker_id is None
        assert held.lease_expires_at is None
        assert held.status_payload == {
            "completion_decision_id": decision.decision_id,
            "gap_fingerprint": decision.gap_fingerprint,
            "verdict": verdict.value,
        }

        session_store = InMemorySessionStore()
        app = CayuApp(
            session_store=session_store,
            task_store=store,
            enable_logging=False,
        )
        await session_store.create(
            RunRequest(
                agent_name="unresolved-agent",
                session_id=running.session_id,
                messages=[Message.text("user", "Stored session input.")],
            ),
            identity=SessionIdentity(
                provider_name="unresolved-provider",
                model="unresolved-model",
            ),
        )
        resume_stream = app.resume(
            ResumeRequest(
                session_id=running.session_id or "missing-session",
                messages=[Message.text("user", "Do not bypass the verifier hold.")],
            )
        )
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await anext(resume_stream)
        still_held = await store.load_task(task.id)
        assert still_held is not None
        assert still_held.status is expected_status

        resumed = await store.resume_task(task.id)
        restarted = await store.start_task(
            resumed.id,
            session_id=resumed.session_id,
            session_invocation=await task_backed_session_invocation(
                store,
                resumed.id,
                resumed.session_id or "missing-session",
            ),
        )
        next_attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id=f"attempt-{verdict.value}-after-resume",
                task_id=restarted.id,
                session_id=restarted.session_id or "missing-session",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("worker-profile-after-resume"),
            )
        )
        accepted_result = _task_result(f"{verdict.value}-after-resume")
        accepted_reference = _result_reference(
            f"{verdict.value}-after-resume",
            result=accepted_result,
        )
        next_proposal = await store.submit_completion_proposal(
            CompletionProposalCreate(
                proposal_id=f"proposal-{verdict.value}-after-resume",
                attempt_id=next_attempt.attempt_id,
                result=accepted_reference,
            )
        )
        next_claim = await store.claim_completion_verification(
            CompletionVerificationClaimRequest(
                claim_id=f"claim-{verdict.value}-after-resume",
                proposal_id=next_proposal.proposal_id,
                worker_id="verifier-worker-after-resume",
                verifier=contract.verifier,
            )
        )
        accepted_decision = await store.record_completion_decision(
            _accepted_decision(
                proposal_id=next_proposal.proposal_id,
                claim_id=next_claim.claim_id,
                worker_id=next_claim.worker_id,
            )
        )
        completed = await store.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=accepted_decision.decision_id,
                idempotency_key=f"apply-{verdict.value}-after-resume",
                result=accepted_result,
                result_reference=accepted_reference,
            )
        )
        assert completed.status is TaskStatus.COMPLETED
        assert completed.result == accepted_result

    asyncio.run(scenario())


def test_decision_requires_complete_contract_coverage_and_current_claim() -> None:
    async def scenario() -> None:
        now = [datetime(2026, 8, 19, tzinfo=UTC)]
        store = InMemoryTaskStore(clock=lambda: now[0])
        contract = _contract()
        await store.publish_work_contract(contract)
        task = await store.create_running_task(
            TaskCreate(
                task_id="claim-task",
                type="bid",
                session_id="session:claim",
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding("session:claim"),
        )
        attempt = await store.begin_work_attempt(
            WorkAttemptCreate(
                attempt_id="claim-attempt",
                task_id=task.id,
                session_id="session:claim",
                contract=contract.reference(),
                execution_profile_fingerprint=_digest("worker-profile"),
            )
        )
        proposal = await store.submit_completion_proposal(
            CompletionProposalCreate(
                proposal_id="claim-proposal",
                attempt_id=attempt.attempt_id,
                result=_result_reference("claim"),
            )
        )
        expired_claim = await store.claim_completion_verification(
            CompletionVerificationClaimRequest(
                claim_id="expired-claim",
                proposal_id=proposal.proposal_id,
                worker_id="verifier-a",
                verifier=contract.verifier,
                lease_seconds=1,
            )
        )
        now[0] += timedelta(seconds=2)
        replacement = await store.claim_completion_verification(
            CompletionVerificationClaimRequest(
                claim_id="replacement-claim",
                proposal_id=proposal.proposal_id,
                worker_id="verifier-b",
                verifier=contract.verifier,
                lease_seconds=30,
            )
        )
        assert replacement.attempt_number == 2

        with pytest.raises(CompletionVerificationClaimLost, match="current live"):
            await store.record_completion_decision(
                _rejected_decision(
                    proposal_id=proposal.proposal_id,
                    claim_id=expired_claim.claim_id,
                    worker_id=expired_claim.worker_id,
                )
            )

        incomplete = _rejected_decision(
            proposal_id=proposal.proposal_id,
            claim_id=replacement.claim_id,
            worker_id=replacement.worker_id,
        ).model_copy(
            update={
                "criterion_outcomes": (
                    CompletionCriterionOutcome(
                        criterion_id="coverage",
                        status=CriterionOutcomeStatus.UNSATISFIED,
                        reason_code="artifact.missing",
                    ),
                ),
                "gaps": (
                    CompletionGap(
                        criterion_id="coverage",
                        code="artifact.missing",
                        evidence_requirement_ids=("artifact",),
                    ),
                    CompletionGap(
                        constraint_id="no-unapproved-send",
                        code="approval.missing",
                        evidence_requirement_ids=("approval",),
                    ),
                ),
            }
        )
        incomplete = CompletionDecisionCreate.model_validate(
            incomplete.model_dump(mode="python", warnings=False)
        )
        with pytest.raises(WorkCompletionConflict, match="every contract criterion"):
            await store.record_completion_decision(incomplete)

        wrong_gap = _rejected_decision(
            proposal_id=proposal.proposal_id,
            claim_id=replacement.claim_id,
            worker_id=replacement.worker_id,
        ).model_copy(
            update={
                "gaps": (
                    CompletionGap(
                        criterion_id="coverage",
                        code="artifact.missing",
                        evidence_requirement_ids=("artifact",),
                    ),
                    CompletionGap(
                        constraint_id="no-unapproved-send",
                        code="approval.missing",
                        evidence_requirement_ids=("approval",),
                    ),
                )
            }
        )
        with pytest.raises(ValidationError, match="exactly the unresolved outcomes"):
            CompletionDecisionCreate.model_validate(
                wrong_gap.model_dump(mode="python", warnings=False)
            )

    asyncio.run(scenario())


def test_terminal_task_rejects_new_verifier_authority_but_preserves_exact_replay() -> None:
    async def scenario() -> None:
        now = [datetime(2026, 8, 19, tzinfo=UTC)]
        store = InMemoryTaskStore(clock=lambda: now[0])
        contract = _contract(contract_id="terminal-verifier-authority-contract")
        await store.publish_work_contract(contract)

        async def prepare(suffix: str) -> tuple[Task, CompletionProposal]:
            session_id = f"session:terminal-verifier-authority:{suffix}"
            task = await store.create_running_task(
                TaskCreate(
                    task_id=f"terminal-verifier-authority-task-{suffix}",
                    type="bid",
                    session_id=session_id,
                    work_contract=contract.reference(),
                ),
                session_invocation=unattributed_session_invocation_binding(session_id),
            )
            attempt = await store.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id=f"terminal-verifier-authority-attempt-{suffix}",
                    task_id=task.id,
                    session_id=session_id,
                    contract=contract.reference(),
                    execution_profile_fingerprint=_digest(f"worker-profile-{suffix}"),
                )
            )
            proposal = await store.submit_completion_proposal(
                CompletionProposalCreate(
                    proposal_id=f"terminal-verifier-authority-proposal-{suffix}",
                    attempt_id=attempt.attempt_id,
                    result=_result_reference(suffix),
                )
            )
            return task, proposal

        unclaimed_task, unclaimed_proposal = await prepare("fresh")
        await store.cancel_task(unclaimed_task.id)
        fresh_request = CompletionVerificationClaimRequest(
            claim_id="terminal-verifier-authority-fresh-claim",
            proposal_id=unclaimed_proposal.proposal_id,
            worker_id="verifier-fresh",
            verifier=contract.verifier,
        )
        with pytest.raises(WorkCompletionConflict, match="live task session"):
            await store.claim_completion_verification(fresh_request)
        assert (
            await store.load_completion_verification_claim(unclaimed_proposal.proposal_id) is None
        )

        claimed_task, claimed_proposal = await prepare("claimed")
        claim_request = CompletionVerificationClaimRequest(
            claim_id="terminal-verifier-authority-existing-claim",
            proposal_id=claimed_proposal.proposal_id,
            worker_id="verifier-existing",
            verifier=contract.verifier,
            lease_seconds=30,
        )
        claim = await store.claim_completion_verification(claim_request)
        await store.cancel_task(claimed_task.id)

        # A live exact retry remains an immutable audit replay, but it cannot
        # authorize a new decision after the attempt loses its running task.
        assert await store.claim_completion_verification(claim_request) == claim
        decision_request = _accepted_decision(
            proposal_id=claimed_proposal.proposal_id,
            claim_id=claim.claim_id,
            worker_id=claim.worker_id,
        )
        with pytest.raises(WorkCompletionConflict, match="live task session"):
            await store.record_completion_decision(decision_request)
        assert await store.load_completion_decision(decision_request.decision_id) is None

        now[0] += timedelta(seconds=31)
        replacement_request = claim_request.model_copy(
            update={
                "claim_id": "terminal-verifier-authority-replacement-claim",
                "worker_id": "verifier-replacement",
            }
        )
        with pytest.raises(WorkCompletionConflict, match="live task session"):
            await store.claim_completion_verification(replacement_request)
        assert await store.load_completion_verification_claim(claimed_proposal.proposal_id) == claim

    asyncio.run(scenario())


def test_ordinary_worker_and_run_entrances_reject_contracted_tasks_before_execution() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        await store.publish_work_contract(contract)
        worker_task = await store.create_task(
            TaskCreate(
                task_id="ordinary-worker-contract-task",
                type="bid-worker",
                work_contract=contract.reference(),
            )
        )
        app = CayuApp(
            session_store=InMemorySessionStore(),
            task_store=store,
            enable_logging=False,
        )
        handler_called = False

        async def handler(_app: CayuApp, _task: Task, _worker_id: str) -> None:
            nonlocal handler_called
            handler_called = True

        handled = await run_task_worker(
            app,
            store,
            handler,
            worker_id="ordinary-worker",
            query=TaskQuery(type=worker_task.type),
            poll_interval_s=0.001,
            max_tasks=1,
        )
        assert handled == 1
        assert handler_called is False
        held = await store.load_task(worker_task.id)
        assert held is not None
        assert held.status is TaskStatus.NEEDS_ATTENTION
        assert held.status_reason == "verified_work_contract_runner_required"
        assert held.status_payload == {
            "contract_id": contract.contract_id,
            "contract_version": contract.version,
        }

        direct_task = await store.create_task(
            TaskCreate(
                task_id="ordinary-run-contract-task",
                type="bid-run",
                work_contract=contract.reference(),
            )
        )
        stream = app.run(
            RunRequest(
                agent_name="unresolved-agent",
                messages=[Message.text("user", "Do not execute this contracted task.")],
                task_id=direct_task.id,
            )
        )
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await anext(stream)
        unchanged = await store.load_task(direct_task.id)
        assert unchanged is not None
        assert unchanged.status is TaskStatus.PENDING
        assert unchanged.session_id is None

        resume_session_id = "session:ordinary-resume-contract-task"
        session = await app.session_store.create(
            RunRequest(
                agent_name="unresolved-agent",
                session_id=resume_session_id,
                messages=[Message.text("user", "Stored session input.")],
            ),
            identity=SessionIdentity(
                provider_name="unresolved-provider",
                model="unresolved-model",
            ),
        )
        resume_task = await store.create_running_task(
            TaskCreate(
                task_id="ordinary-resume-contract-task",
                type="bid-resume",
                session_id=resume_session_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(session.id),
        )
        resume_stream = app.resume(
            ResumeRequest(
                session_id=resume_session_id,
                messages=[Message.text("user", "Do not resume this contracted task.")],
            )
        )
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await anext(resume_stream)
        resume_unchanged = await store.load_task(resume_task.id)
        assert resume_unchanged is not None
        assert resume_unchanged.status is TaskStatus.RUNNING

    asyncio.run(scenario())


def test_fork_group_evaluator_preparation_rejects_contracted_session_authority() -> None:
    async def scenario() -> None:
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        provider = _RecordingProvider()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="source", model="verified-work-test-model"))
        app.register_agent(AgentSpec(name="evaluator", model="verified-work-test-model"))

        source_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="source",
                    session_id="verified-work-fork-group-source",
                    causal_budget_id="verified-work-fork-group-budget",
                    messages=[Message.text("user", "Prepare a fork-group source.")],
                )
            )
        ]
        assert source_events[-1].type is EventType.SESSION_COMPLETED
        source_event_count = len(await session_store.load_events("verified-work-fork-group-source"))

        contract = _contract(contract_id="fork-group-evaluator-contract")
        await task_store.publish_work_contract(contract)
        evaluator_session_id = "verified-work-fork-group-evaluator"
        await task_store.create_running_task(
            TaskCreate(
                task_id="fork-group-evaluator-contract-task",
                type="verified-work",
                session_id=evaluator_session_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(evaluator_session_id),
        )
        candidate_output = StructuredOutputSpec(
            name="verified-work-fork-candidate",
            json_schema={
                "type": "object",
                "properties": {"candidate": {"type": "string"}},
                "required": ["candidate"],
                "additionalProperties": False,
            },
        )
        request = ForkGroupRequest(
            group_id="verified-work-fork-group",
            source_session_id="verified-work-fork-group-source",
            source_checkpoint=ForkGroupCheckpointSelector(),
            causal_budget_id="verified-work-fork-group-budget",
            branches=(
                ForkGroupBranchSpec(
                    branch_id="alpha",
                    session_id="verified-work-fork-group-alpha",
                    messages=(Message.text("user", "Candidate alpha."),),
                    structured_output=candidate_output,
                ),
                ForkGroupBranchSpec(
                    branch_id="beta",
                    session_id="verified-work-fork-group-beta",
                    messages=(Message.text("user", "Candidate beta."),),
                    structured_output=candidate_output,
                ),
            ),
            evaluator=ForkGroupEvaluatorSpec(
                session_id=evaluator_session_id,
                agent_name="evaluator",
            ),
        )

        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await app.run_fork_group(request)

        assert len(provider.requests) == 1
        assert await session_store.load(evaluator_session_id) is None
        assert await session_store.load("verified-work-fork-group-alpha") is None
        assert await session_store.load("verified-work-fork-group-beta") is None
        assert (
            len(await session_store.load_events("verified-work-fork-group-source"))
            == source_event_count
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_evaluator", ["missing_agent", "contracted_session"])
def test_invalid_fork_group_evaluator_does_not_admit_source(
    invalid_evaluator: str,
) -> None:
    async def scenario() -> None:
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(_RecordingProvider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="verified-work-test-model"))
        app.register_agent(AgentSpec(name="evaluator", model="verified-work-test-model"))

        source_id = f"verified-work-invalid-{invalid_evaluator}-group-source"
        group_id = f"verified-work-invalid-{invalid_evaluator}-group"
        evaluator_session_id = f"{group_id}-evaluator"
        await _create_dispatch_test_session(app, source_id)
        source = await session_store.load(source_id)
        assert source is not None
        source_events_before = await session_store.load_events(source_id)
        source_checkpoint_before = await session_store.load_checkpoint(source_id)

        contract = _contract(contract_id=f"invalid-{invalid_evaluator}-group-contract")
        await task_store.publish_work_contract(contract)
        if invalid_evaluator == "contracted_session":
            await task_store.create_running_task(
                TaskCreate(
                    task_id=f"{group_id}-evaluator-task",
                    type="verified-work",
                    session_id=evaluator_session_id,
                    work_contract=contract.reference(),
                ),
                session_invocation=unattributed_session_invocation_binding(evaluator_session_id),
            )
        candidate_output = StructuredOutputSpec(
            name=f"{group_id}-candidate",
            json_schema={
                "type": "object",
                "properties": {"candidate": {"type": "string"}},
                "required": ["candidate"],
                "additionalProperties": False,
            },
        )
        request = ForkGroupRequest(
            group_id=group_id,
            source_session_id=source_id,
            source_checkpoint=ForkGroupCheckpointSelector(),
            causal_budget_id=source.causal_budget_id,
            branches=(
                ForkGroupBranchSpec(
                    branch_id="alpha",
                    session_id=f"{group_id}-alpha",
                    messages=(Message.text("user", "Candidate alpha."),),
                    structured_output=candidate_output,
                ),
                ForkGroupBranchSpec(
                    branch_id="beta",
                    session_id=f"{group_id}-beta",
                    messages=(Message.text("user", "Candidate beta."),),
                    structured_output=candidate_output,
                ),
            ),
            evaluator=ForkGroupEvaluatorSpec(
                session_id=evaluator_session_id,
                agent_name=(
                    "missing-evaluator" if invalid_evaluator == "missing_agent" else "evaluator"
                ),
            ),
        )

        expected_error = (
            pytest.raises(KeyError, match="missing-evaluator")
            if invalid_evaluator == "missing_agent"
            else pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware")
        )
        with expected_error:
            await app.run_fork_group(request)

        assert await session_store.load_checkpoint(source_id) == source_checkpoint_before
        assert await session_store.load_events(source_id) == source_events_before
        for suffix in ("alpha", "beta", "evaluator"):
            assert await session_store.load(f"{group_id}-{suffix}") is None

        attached = await task_store.create_running_task(
            TaskCreate(
                task_id=f"{group_id}-source-task",
                type="verified-work",
                session_id=source_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(source_id),
        )
        assert attached.session_id == source_id

    asyncio.run(scenario())


def test_active_model_stage_fork_group_does_not_admit_source() -> None:
    async def scenario() -> None:
        app, session_store, task_store, _dispatcher, _provider = _dispatch_test_app()
        app.register_agent(AgentSpec(name="evaluator", model="verified-work-test-model"))
        source_id = "verified-work-active-stage-group-source"
        group_id = "verified-work-active-stage-group"
        await _create_dispatch_test_session(app, source_id)
        source = await session_store.load(source_id)
        assert source is not None
        transcript = await session_store.load_transcript_snapshot(source_id)
        await session_store.prepare_model_completion_stage(
            source_id,
            request=ModelCompletionStageRequest(
                stage_id="verified-work-active-stage",
                logical_step_id="verified-work-active-stage-step",
                dispatch_ordinal=0,
                intent={},
            ),
            expected_statuses={SessionStatus.COMPLETED},
            expected_run_epoch=source.run_epoch,
            expected_transcript_cursor=transcript.cursor,
        )
        candidate_output = StructuredOutputSpec(
            name="verified-work-active-stage-candidate",
            json_schema={
                "type": "object",
                "properties": {"candidate": {"type": "string"}},
                "required": ["candidate"],
                "additionalProperties": False,
            },
        )
        request = ForkGroupRequest(
            group_id=group_id,
            source_session_id=source_id,
            source_checkpoint=ForkGroupCheckpointSelector(),
            causal_budget_id=source.causal_budget_id,
            branches=(
                ForkGroupBranchSpec(
                    branch_id="alpha",
                    session_id=f"{group_id}-alpha",
                    messages=(Message.text("user", "Candidate alpha."),),
                    structured_output=candidate_output,
                ),
                ForkGroupBranchSpec(
                    branch_id="beta",
                    session_id=f"{group_id}-beta",
                    messages=(Message.text("user", "Candidate beta."),),
                    structured_output=candidate_output,
                ),
            ),
            evaluator=ForkGroupEvaluatorSpec(
                session_id=f"{group_id}-evaluator",
                agent_name="evaluator",
            ),
        )

        with pytest.raises(ValueError, match="active model-completion stage"):
            await app.run_fork_group(request)

        operation_id = "fork-group:" + sha256(group_id.encode("utf-8")).hexdigest()
        assert await session_store.load_session_operation(source_id, operation_id) is None
        for suffix in ("alpha", "beta", "evaluator"):
            assert await session_store.load(f"{group_id}-{suffix}") is None

        contract = _contract(contract_id="active-stage-group-contract")
        await task_store.publish_work_contract(contract)
        attached = await task_store.create_running_task(
            TaskCreate(
                task_id="active-stage-group-contract-task",
                type="verified-work",
                session_id=source_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(source_id),
        )
        assert attached.session_id == source_id

    asyncio.run(scenario())


def test_direct_and_hook_forks_reject_contracted_sources_without_mutation() -> None:
    async def scenario() -> None:
        app, session_store, task_store, _dispatcher, _provider = _dispatch_test_app()
        contract = _contract(contract_id="fork-source-contract")
        await task_store.publish_work_contract(contract)

        for entrance in ("public", "hook"):
            source_id = f"verified-work-{entrance}-fork-source"
            child_id = f"verified-work-{entrance}-fork-child"
            await _create_dispatch_test_session(app, source_id)
            task = await task_store.create_running_task(
                TaskCreate(
                    task_id=f"{entrance}-fork-source-contract-task",
                    type="verified-work",
                    session_id=source_id,
                    work_contract=contract.reference(),
                ),
                session_invocation=unattributed_session_invocation_binding(source_id),
            )
            source_before = await session_store.load(source_id)
            checkpoint_before = await session_store.load_checkpoint(source_id)
            events_before = await session_store.load_events(source_id)
            request = ForkSessionRequest(source_session_id=source_id, session_id=child_id)
            stream = (
                app.fork_session(request)
                if entrance == "public"
                else app._fork_session_from_runtime_context(
                    request,
                    source_session_id=source_id,
                )
            )

            with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
                await anext(stream)

            assert await session_store.load(source_id) == source_before
            assert await session_store.load_checkpoint(source_id) == checkpoint_before
            assert await session_store.load_events(source_id) == events_before
            assert await session_store.load(child_id) is None
            assert await session_store.load_checkpoint(child_id) is None
            assert await task_store.load_task(task.id) == task

    asyncio.run(scenario())


def test_direct_fork_validates_source_authority_before_contract_store_preflight(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "verified-work-fork-source-secret-canary"

    class LookupRecordingStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.session_lookups: list[str] = []

        async def load_active_work_contract_task_for_session(
            self,
            session_id: str,
        ) -> Task | None:
            self.session_lookups.append(session_id)
            return await super().load_active_work_contract_task_for_session(session_id)

    async def scenario() -> tuple[BaseException, LookupRecordingStore]:
        task_store = LookupRecordingStore()
        app = CayuApp(
            session_store=InMemorySessionStore(),
            task_store=task_store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        stream = app.fork_session(
            ForkSessionRequest(
                source_session_id=f"source-{secret}",
                session_id="safe-fork-child",
            )
        )
        with pytest.raises(ValueError, match="source_session_id contains a workload secret") as exc:
            await anext(stream)
        assert await app.session_store.load("safe-fork-child") is None
        return exc.value, task_store

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error, task_store = asyncio.run(scenario())

    assert task_store.session_lookups == []
    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_invalid_direct_fork_does_not_consume_source_contract_authority() -> None:
    class PendingInputProvider(_RecordingProvider):
        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.tool_call(
                id="verified-work-pending-input-call",
                name="ask_user",
                arguments={"question": "Proceed?"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

    async def scenario() -> None:
        source_id = "verified-work-invalid-fork-source"
        child_id = "verified-work-invalid-fork-child"
        session_store = InMemorySessionStore()
        provider = PendingInputProvider()
        seed_app = CayuApp(session_store=session_store, enable_logging=False)
        seed_app.register_provider(provider, default=True)
        seed_app.register_agent(
            AgentSpec(name="assistant", model="verified-work-test-model"),
            tools=[UserInputTool()],
        )
        async for _ in seed_app.run(
            RunRequest(
                agent_name="assistant",
                session_id=source_id,
                messages=[Message.text("user", "Ask for input.")],
            )
        ):
            pass
        await session_store.update_status(source_id, SessionStatus.FAILED)
        checkpoint_before = await session_store.load_checkpoint(source_id)
        assert checkpoint_before is not None and "pending_user_input" in checkpoint_before
        events_before = await session_store.load_events(source_id)

        task_store = InMemoryTaskStore()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(PendingInputProvider(), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="verified-work-test-model"),
            tools=[UserInputTool()],
        )

        with pytest.raises(RuntimeError, match="awaiting user input cannot be forked"):
            async for _ in app.fork_session(
                ForkSessionRequest(source_session_id=source_id, session_id=child_id)
            ):
                pass

        assert await session_store.load_checkpoint(source_id) == checkpoint_before
        assert await session_store.load_events(source_id) == events_before
        assert await session_store.load(child_id) is None

        contract = _contract(contract_id="invalid-fork-source-contract")
        await task_store.publish_work_contract(contract)
        attached = await task_store.create_running_task(
            TaskCreate(
                task_id="invalid-fork-source-contract-task",
                type="verified-work",
                session_id=source_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(source_id),
        )
        assert attached.session_id == source_id

    asyncio.run(scenario())


def test_invalid_fork_group_source_does_not_consume_contract_authority() -> None:
    class PendingInputProvider(_RecordingProvider):
        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.tool_call(
                id="verified-work-group-pending-input-call",
                name="ask_user",
                arguments={"question": "Proceed?"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

    async def scenario() -> None:
        source_id = "verified-work-invalid-group-source"
        group_id = "verified-work-invalid-source-group"
        session_store = InMemorySessionStore()
        seed_app = CayuApp(session_store=session_store, enable_logging=False)
        seed_app.register_provider(PendingInputProvider(), default=True)
        seed_app.register_agent(
            AgentSpec(name="assistant", model="verified-work-test-model"),
            tools=[UserInputTool()],
        )
        async for _ in seed_app.run(
            RunRequest(
                agent_name="assistant",
                session_id=source_id,
                causal_budget_id="verified-work-invalid-group-budget",
                messages=[Message.text("user", "Ask for input.")],
            )
        ):
            pass
        await session_store.update_status(source_id, SessionStatus.FAILED)
        source = await session_store.load(source_id)
        assert source is not None
        checkpoint_before = await session_store.load_checkpoint(source_id)
        assert checkpoint_before is not None and "pending_user_input" in checkpoint_before
        events_before = await session_store.load_events(source_id)

        task_store = InMemoryTaskStore()
        provider = PendingInputProvider()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="verified-work-test-model"),
            tools=[UserInputTool()],
        )
        candidate_output = StructuredOutputSpec(
            name="verified-work-invalid-source-candidate",
            json_schema={
                "type": "object",
                "properties": {"candidate": {"type": "string"}},
                "required": ["candidate"],
                "additionalProperties": False,
            },
        )
        request = ForkGroupRequest(
            group_id=group_id,
            source_session_id=source_id,
            source_checkpoint=ForkGroupCheckpointSelector(),
            causal_budget_id=source.causal_budget_id,
            branches=(
                ForkGroupBranchSpec(
                    branch_id="alpha",
                    session_id=f"{group_id}-alpha",
                    messages=(Message.text("user", "Candidate alpha."),),
                    structured_output=candidate_output,
                ),
                ForkGroupBranchSpec(
                    branch_id="beta",
                    session_id=f"{group_id}-beta",
                    messages=(Message.text("user", "Candidate beta."),),
                    structured_output=candidate_output,
                ),
            ),
            evaluator=ForkGroupEvaluatorSpec(
                session_id=f"{group_id}-evaluator",
                agent_name="assistant",
            ),
        )

        with pytest.raises(RuntimeError, match="awaiting user input cannot be forked"):
            await app.run_fork_group(request)

        assert provider.requests == []
        assert await session_store.load_checkpoint(source_id) == checkpoint_before
        assert await session_store.load_events(source_id) == events_before
        for suffix in ("alpha", "beta", "evaluator"):
            assert await session_store.load(f"{group_id}-{suffix}") is None

        contract = _contract(contract_id="invalid-group-source-contract")
        await task_store.publish_work_contract(contract)
        attached = await task_store.create_running_task(
            TaskCreate(
                task_id="invalid-group-source-contract-task",
                type="verified-work",
                session_id=source_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(source_id),
        )
        assert attached.session_id == source_id

    asyncio.run(scenario())


def test_exact_fork_replay_cannot_weaken_a_later_source_contract() -> None:
    async def scenario() -> None:
        session_store = InMemorySessionStore()
        source_id = "verified-work-legacy-fork-source"
        child_id = "verified-work-legacy-fork-child"
        seed_app = CayuApp(session_store=session_store, enable_logging=False)
        seed_app.register_provider(_RecordingProvider(), default=True)
        seed_app.register_agent(AgentSpec(name="assistant", model="verified-work-test-model"))
        await _create_dispatch_test_session(seed_app, source_id)
        request = ForkSessionRequest(source_session_id=source_id, session_id=child_id)
        first_events = [event async for event in seed_app.fork_session(request)]
        assert first_events[-1].type is EventType.SESSION_FORKED

        task_store = InMemoryTaskStore()
        contract = _contract(contract_id="legacy-fork-source-contract")
        await task_store.publish_work_contract(contract)
        task = await task_store.create_running_task(
            TaskCreate(
                task_id="legacy-fork-source-contract-task",
                type="verified-work",
                session_id=source_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(source_id),
        )
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(_RecordingProvider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="verified-work-test-model"))
        source_before = await session_store.load(source_id)
        source_checkpoint_before = await session_store.load_checkpoint(source_id)
        source_events_before = await session_store.load_events(source_id)
        child_before = await session_store.load(child_id)
        child_events_before = await session_store.load_events(child_id)

        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await anext(app.fork_session(request))

        assert await session_store.load(source_id) == source_before
        assert await session_store.load_checkpoint(source_id) == source_checkpoint_before
        assert await session_store.load_events(source_id) == source_events_before
        assert await session_store.load(child_id) == child_before
        assert await session_store.load_events(child_id) == child_events_before
        assert await task_store.load_task(task.id) == task

    asyncio.run(scenario())


def test_invalid_prompt_succession_replay_does_not_admit_source() -> None:
    class AllowForkProfileAdoption(ExecutionProfilePolicy):
        @property
        def identity(self) -> str:
            return "test:invalid-prompt-succession-adoption:v1"

        async def decide(
            self,
            request: ExecutionProfilePolicyRequest,
        ) -> ExecutionProfilePolicyResult:
            return ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason="The test authorizes its explicit child profile.",
                authority_decision=(
                    ExecutionProfileAuthorityDecision.AUTHORIZED
                    if request.authority_review_required
                    else ExecutionProfileAuthorityDecision.NOT_REQUIRED
                ),
            )

    async def scenario() -> None:
        source_id = "invalid-prompt-succession-source"
        child_id = "invalid-prompt-succession-child"
        session_store = InMemorySessionStore()
        seed_app = CayuApp(
            session_store=session_store,
            execution_profile_policy=AllowForkProfileAdoption(),
            enable_logging=False,
        )
        seed_app.register_provider(_RecordingProvider(), default=True)
        seed_app.register_environment(
            Environment(EnvironmentSpec(name="body")),
            default=True,
        )
        seed_app.register_agent(
            AgentSpec(
                name="assistant",
                model="verified-work-test-model",
                system_prompt="stable prompt succession",
            )
        )
        async for _ in seed_app.run(
            RunRequest(
                agent_name="assistant",
                session_id=source_id,
                environment_name="body",
                messages=[Message.text("user", "Create a prompt-succession source.")],
            )
        ):
            pass
        request = ForkSessionRequest(
            source_session_id=source_id,
            session_id=child_id,
            copy_checkpoint=False,
            system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
            execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
            profile_adoption=ExecutionProfileAdoptionIntent(
                idempotency_key="invalid-prompt-succession-profile",
                reason="Exercise exact prompt-succession replay validation.",
                requested_by=ResolutionActor(
                    subject="verified-work-test",
                    source=ResolutionActorSource.REQUEST,
                ),
            ),
        )
        first_events = [event async for event in seed_app.fork_session(request)]
        assert first_events[-1].type is EventType.SESSION_FORKED

        async with session_store._lock:
            child = session_store._sessions[child_id]
            corrupted_metadata = dict(child.metadata)
            assert corrupted_metadata.pop(PROMPT_ANATOMY_TRANSITION_METADATA_KEY, None) is not None
            session_store._sessions[child_id] = child.model_copy(
                update={"metadata": corrupted_metadata},
                deep=True,
            )

        task_store = InMemoryTaskStore()
        replay_app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            execution_profile_policy=AllowForkProfileAdoption(),
            enable_logging=False,
        )
        replay_app.register_provider(_RecordingProvider(), default=True)
        replay_app.register_environment(
            Environment(EnvironmentSpec(name="body")),
            default=True,
        )
        replay_app.register_agent(
            AgentSpec(
                name="assistant",
                model="verified-work-test-model",
                system_prompt="stable prompt succession",
            )
        )

        with pytest.raises(RuntimeError, match="prompt-anatomy descendant conflicts"):
            await anext(replay_app.fork_session(request))

        contract = _contract(contract_id="invalid-prompt-succession-contract")
        await task_store.publish_work_contract(contract)
        attached = await task_store.create_running_task(
            TaskCreate(
                task_id="invalid-prompt-succession-contract-task",
                type="verified-work",
                session_id=source_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(source_id),
        )
        assert attached.session_id == source_id

    asyncio.run(scenario())


def test_direct_fork_contract_attachment_race_fails_before_child_publication() -> None:
    class AdmissionBarrierStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True
        blocked_session_id: str | None = None
        admission_started: asyncio.Event | None = None
        allow_admission: asyncio.Event | None = None

        async def admit_ordinary_session_execution(self, session_id: str) -> None:
            if session_id == self.blocked_session_id:
                if self.admission_started is None or self.allow_admission is None:
                    raise AssertionError("Fork admission barrier was not configured.")
                self.admission_started.set()
                await self.allow_admission.wait()
            await super().admit_ordinary_session_execution(session_id)

    async def scenario() -> None:
        source_id = "verified-work-racing-fork-source"
        child_id = "verified-work-racing-fork-child"
        session_store = InMemorySessionStore()
        task_store = AdmissionBarrierStore()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(_RecordingProvider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="verified-work-test-model"))
        await _create_dispatch_test_session(app, source_id)
        contract = _contract(contract_id="racing-fork-source-contract")
        await task_store.publish_work_contract(contract)
        task_store.blocked_session_id = source_id
        task_store.admission_started = asyncio.Event()
        task_store.allow_admission = asyncio.Event()
        source_before = await session_store.load(source_id)
        checkpoint_before = await session_store.load_checkpoint(source_id)
        events_before = await session_store.load_events(source_id)

        async def fork() -> list[object]:
            return [
                event
                async for event in app.fork_session(
                    ForkSessionRequest(source_session_id=source_id, session_id=child_id)
                )
            ]

        fork_task = asyncio.create_task(fork())
        await task_store.admission_started.wait()
        contracted_task = await task_store.create_running_task(
            TaskCreate(
                task_id="racing-fork-source-contract-task",
                type="verified-work",
                session_id=source_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(source_id),
        )
        task_store.allow_admission.set()

        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await fork_task
        assert await session_store.load(source_id) == source_before
        assert await session_store.load_checkpoint(source_id) == checkpoint_before
        assert await session_store.load_events(source_id) == events_before
        assert await session_store.load(child_id) is None
        assert await session_store.load_checkpoint(child_id) is None
        assert await task_store.load_task(contracted_task.id) == contracted_task

    asyncio.run(scenario())


def test_fork_group_source_contract_and_attachment_race_leave_no_group_record() -> None:
    class AdmissionBarrierStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True
        blocked_session_id: str | None = None
        admission_started: asyncio.Event | None = None
        allow_admission: asyncio.Event | None = None

        async def admit_ordinary_session_execution(self, session_id: str) -> None:
            if session_id == self.blocked_session_id:
                if self.admission_started is None or self.allow_admission is None:
                    raise AssertionError("Fork-group admission barrier was not configured.")
                self.admission_started.set()
                await self.allow_admission.wait()
            await super().admit_ordinary_session_execution(session_id)

    async def request_for(
        app: CayuApp,
        source_id: str,
        group_id: str,
    ) -> ForkGroupRequest:
        source = await app.session_store.load(source_id)
        assert source is not None
        candidate_output = StructuredOutputSpec(
            name=f"{group_id}-candidate",
            json_schema={
                "type": "object",
                "properties": {"candidate": {"type": "string"}},
                "required": ["candidate"],
                "additionalProperties": False,
            },
        )
        return ForkGroupRequest(
            group_id=group_id,
            source_session_id=source_id,
            source_checkpoint=ForkGroupCheckpointSelector(),
            causal_budget_id=source.causal_budget_id,
            branches=(
                ForkGroupBranchSpec(
                    branch_id="alpha",
                    session_id=f"{group_id}-alpha",
                    messages=(Message.text("user", "Candidate alpha."),),
                    structured_output=candidate_output,
                ),
                ForkGroupBranchSpec(
                    branch_id="beta",
                    session_id=f"{group_id}-beta",
                    messages=(Message.text("user", "Candidate beta."),),
                    structured_output=candidate_output,
                ),
            ),
            evaluator=ForkGroupEvaluatorSpec(
                session_id=f"{group_id}-evaluator",
                agent_name="assistant",
            ),
        )

    async def scenario() -> None:
        session_store = InMemorySessionStore()
        task_store = AdmissionBarrierStore()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(_RecordingProvider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="verified-work-test-model"))
        contract = _contract(contract_id="fork-group-source-contract")
        await task_store.publish_work_contract(contract)

        contracted_source_id = "verified-work-contracted-group-source"
        await _create_dispatch_test_session(app, contracted_source_id)
        contracted = await task_store.create_running_task(
            TaskCreate(
                task_id="contracted-group-source-task",
                type="verified-work",
                session_id=contracted_source_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(contracted_source_id),
        )
        contracted_request = await request_for(
            app,
            contracted_source_id,
            "verified-work-contracted-group",
        )
        contracted_checkpoint_before = await session_store.load_checkpoint(contracted_source_id)
        contracted_events_before = await session_store.load_events(contracted_source_id)
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await app.run_fork_group(contracted_request)
        assert (
            await session_store.load_checkpoint(contracted_source_id)
            == contracted_checkpoint_before
        )
        assert await session_store.load_events(contracted_source_id) == contracted_events_before
        for suffix in ("alpha", "beta", "evaluator"):
            assert await session_store.load(f"verified-work-contracted-group-{suffix}") is None
        assert await task_store.load_task(contracted.id) == contracted

        racing_source_id = "verified-work-racing-group-source"
        await _create_dispatch_test_session(app, racing_source_id)
        racing_request = await request_for(
            app,
            racing_source_id,
            "verified-work-racing-group",
        )
        checkpoint_before = await session_store.load_checkpoint(racing_source_id)
        events_before = await session_store.load_events(racing_source_id)
        task_store.blocked_session_id = racing_source_id
        task_store.admission_started = asyncio.Event()
        task_store.allow_admission = asyncio.Event()
        group_task = asyncio.create_task(app.run_fork_group(racing_request))
        await task_store.admission_started.wait()
        racing_contract_task = await task_store.create_running_task(
            TaskCreate(
                task_id="racing-group-source-contract-task",
                type="verified-work",
                session_id=racing_source_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(racing_source_id),
        )
        task_store.allow_admission.set()

        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await group_task
        assert await session_store.load_checkpoint(racing_source_id) == checkpoint_before
        assert await session_store.load_events(racing_source_id) == events_before
        for suffix in ("alpha", "beta", "evaluator"):
            child_id = f"verified-work-racing-group-{suffix}"
            assert await session_store.load(child_id) is None
        assert await task_store.load_task(racing_contract_task.id) == racing_contract_task
        evaluator_id = "verified-work-racing-group-evaluator"
        evaluator_contract_task = await task_store.create_running_task(
            TaskCreate(
                task_id="rejected-group-evaluator-contract-task",
                type="verified-work",
                session_id=evaluator_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(evaluator_id),
        )
        assert await task_store.load_task(evaluator_contract_task.id) == evaluator_contract_task

    asyncio.run(scenario())


def test_ordinary_continuation_admission_follows_validation_and_wins_races_atomically() -> None:
    class AdmissionBarrierStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True
        blocked_session_id: str | None = None
        admission_started: asyncio.Event | None = None
        allow_admission: asyncio.Event | None = None

        async def admit_ordinary_session_execution(self, session_id: str) -> None:
            if session_id == self.blocked_session_id:
                if self.admission_started is None or self.allow_admission is None:
                    raise AssertionError("Resume admission barrier was not configured.")
                self.admission_started.set()
                await self.allow_admission.wait()
            await super().admit_ordinary_session_execution(session_id)

    async def attach_contract(
        store: InMemoryTaskStore,
        contract: WorkContract,
        *,
        session_id: str,
        task_id: str,
    ) -> Task:
        return await store.create_running_task(
            TaskCreate(
                task_id=task_id,
                type="verified-work",
                session_id=session_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(session_id),
        )

    async def scenario() -> None:
        session_store = InMemorySessionStore()
        task_store = AdmissionBarrierStore()
        provider = _RecordingProvider()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="verified-work-test-model"))
        contract = _contract(contract_id="validated-continuation-admission")
        await task_store.publish_work_contract(contract)

        missing_session_id = "session:missing-before-admission"
        missing_stream = app.resume(
            ResumeRequest(
                session_id=missing_session_id,
                messages=[Message.text("user", "This session does not exist.")],
            )
        )
        with pytest.raises(KeyError, match="Session not found"):
            await anext(missing_stream)
        missing_task = await attach_contract(
            task_store,
            contract,
            session_id=missing_session_id,
            task_id="missing-resume-contract-task",
        )
        assert missing_task.work_contract == contract.reference()

        stale_session_id = "session:stale-approval-before-admission"
        await _create_dispatch_test_session(app, stale_session_id)
        stale_stream = app._resolve_tool_approval_private(
            ToolApprovalRequest(
                session_id=stale_session_id,
                approval_id="missing-approval",
                tool_round_id="missing-round",
                tool_call_id="missing-call",
                decision=ToolApprovalDecision.APPROVE,
            )
        )
        with pytest.raises(RuntimeError, match="no pending tool approval"):
            await anext(stale_stream)
        stale_task = await attach_contract(
            task_store,
            contract,
            session_id=stale_session_id,
            task_id="stale-approval-contract-task",
        )
        assert stale_task.work_contract == contract.reference()

        racing_session_id = "session:resume-contract-race"
        await _create_dispatch_test_session(app, racing_session_id)
        session_before = await session_store.load(racing_session_id)
        checkpoint_before = await session_store.load_checkpoint(racing_session_id)
        events_before = await session_store.load_events(racing_session_id)
        task_store.blocked_session_id = racing_session_id
        task_store.admission_started = asyncio.Event()
        task_store.allow_admission = asyncio.Event()
        racing_stream = app.resume(
            ResumeRequest(
                session_id=racing_session_id,
                messages=[Message.text("user", "Race the final ordinary admission.")],
            )
        )

        async def receive_first_resume_event():
            return await anext(racing_stream)

        racing_resume = asyncio.create_task(receive_first_resume_event())
        await task_store.admission_started.wait()
        racing_task = await attach_contract(
            task_store,
            contract,
            session_id=racing_session_id,
            task_id="racing-resume-contract-task",
        )
        task_store.allow_admission.set()
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await racing_resume

        assert await session_store.load(racing_session_id) == session_before
        assert await session_store.load_checkpoint(racing_session_id) == checkpoint_before
        assert await session_store.load_events(racing_session_id) == events_before
        assert await task_store.load_task(racing_task.id) == racing_task
        assert provider.requests == []

    asyncio.run(scenario())


def test_initial_run_rejects_session_contract_before_provider_preflight(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "verified-work-provider-preflight-secret-canary"

    class PreflightRecordingProvider(_RecordingProvider):
        supports_native_structured_output = True

        def __init__(self) -> None:
            super().__init__()
            self.preflight_schemas: list[dict[str, object]] = []

        def __repr__(self) -> str:
            return f"PreflightRecordingProvider({secret})"

        def preflight_native_structured_output_schema(
            self,
            json_schema: dict[str, object],
        ) -> None:
            self.preflight_schemas.append(json_schema)

    async def scenario() -> BaseException:
        session_id = "session:contract-before-provider-preflight"
        task_store = InMemoryTaskStore()
        contract = _contract(
            contract_id="contract-before-provider-preflight",
            objective=f"Do not expose {secret} while rejecting ordinary execution.",
        )
        await task_store.publish_work_contract(contract)
        contracted_task = await task_store.create_running_task(
            TaskCreate(
                task_id="contract-before-provider-preflight-task",
                type="verified-work",
                session_id=session_id,
                input={"private": secret},
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(session_id),
        )
        provider = PreflightRecordingProvider()
        app = CayuApp(
            session_store=InMemorySessionStore(),
            task_store=task_store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="verified-work-test-model"))

        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "Do not run provider preparation.")],
                structured_output=StructuredOutputSpec(
                    json_schema={"type": "object"},
                    strategy=StructuredOutputStrategy.NATIVE,
                ),
            )
        )
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware") as exc:
            await anext(stream)

        assert provider.preflight_schemas == []
        assert provider.requests == []
        assert await task_store.load_task(contracted_task.id) == contracted_task
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


def test_duplicate_initial_run_does_not_consume_legacy_session_contract_authority() -> None:
    async def scenario() -> None:
        session_id = "session:duplicate-before-verified-work-admission"
        session_store = InMemorySessionStore()
        seed_app = CayuApp(session_store=session_store, enable_logging=False)
        seed_app.register_provider(_RecordingProvider(), default=True)
        seed_app.register_agent(AgentSpec(name="assistant", model="verified-work-test-model"))
        await _create_dispatch_test_session(seed_app, session_id)
        session_before = await session_store.load(session_id)
        checkpoint_before = await session_store.load_checkpoint(session_id)
        events_before = await session_store.load_events(session_id)

        task_store = InMemoryTaskStore()
        provider = _RecordingProvider()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="verified-work-test-model"))
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "This duplicate must not run.")],
            )
        )

        with pytest.raises(ValueError, match="Session already exists"):
            await anext(stream)

        assert provider.requests == []
        assert await session_store.load(session_id) == session_before
        assert await session_store.load_checkpoint(session_id) == checkpoint_before
        assert await session_store.load_events(session_id) == events_before

        contract = _contract(contract_id="duplicate-session-contract")
        await task_store.publish_work_contract(contract)
        attached = await task_store.create_running_task(
            TaskCreate(
                task_id="duplicate-session-contract-task",
                type="verified-work",
                session_id=session_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(session_id),
        )
        assert attached.session_id == session_id

    asyncio.run(scenario())


def test_materialized_contracts_are_rejected_when_store_capability_is_false() -> None:
    class AdoptedContractStore(InMemoryTaskStore):
        supports_verified_work_contracts = False
        verified_work_mutations_are_cancellation_quiescent = True

    async def scenario() -> None:
        task_store = AdoptedContractStore()
        contract = _contract(contract_id="adopted-store-contract")
        await task_store.publish_work_contract(contract)
        task = await task_store.create_task(
            TaskCreate(
                task_id="adopted-store-contracted-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        session_id = "session:adopted-store-contract"
        session_task = await task_store.create_task(
            TaskCreate(
                task_id="adopted-store-session-task",
                type="verified-work",
                session_id=session_id,
                work_contract=contract.reference(),
            )
        )
        provider = _RecordingProvider()
        app = CayuApp(
            session_store=InMemorySessionStore(),
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="verified-work-test-model"))

        stream = app.run(
            RunRequest(
                agent_name="assistant",
                task_id=task.id,
                messages=[Message.text("user", "Do not bypass the adopted contract.")],
            )
        )
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await anext(stream)

        session_stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "Do not bypass session contract authority.")],
            )
        )
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await anext(session_stream)

        assert provider.requests == []
        assert await task_store.load_task(task.id) == task
        assert await task_store.load_task(session_task.id) == session_task

    asyncio.run(scenario())


def test_initial_run_rejects_caller_owned_missing_task_identity() -> None:
    async def scenario() -> None:
        app, session_store, _task_store, _dispatcher, provider = _dispatch_test_app()
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="session:missing-caller-task",
                task_id="missing-caller-task",
                messages=[Message.text("user", "Do not run without durable task authority.")],
            )
        )

        with pytest.raises(KeyError, match="Task not found"):
            await anext(stream)

        assert provider.requests == []
        assert await session_store.load("session:missing-caller-task") is None

    asyncio.run(scenario())


def test_explicit_compaction_obeys_atomic_verified_work_admission() -> None:
    class CompactionAdmissionBarrierStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True
        blocked_session_id: str | None = None
        admission_started: asyncio.Event | None = None
        allow_admission: asyncio.Event | None = None

        async def admit_ordinary_session_execution(self, session_id: str) -> None:
            if session_id == self.blocked_session_id:
                if self.admission_started is None or self.allow_admission is None:
                    raise AssertionError("Compaction admission barrier was not configured.")
                self.admission_started.set()
                await self.allow_admission.wait()
            await super().admit_ordinary_session_execution(session_id)

    async def seed_compactable_session(
        app: CayuApp,
        session_store: InMemorySessionStore,
        session_id: str,
    ) -> tuple[Session, int]:
        session = await session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
            ),
            identity=profiled_session_identity(
                provider_name="verified-work-test-provider",
                model="verified-work-test-model",
                app=app,
            ),
        )
        transcript = [
            Message.text("user", "Old context to compact."),
            Message.text("assistant", "Old response."),
            Message.text("user", "Current context to retain."),
        ]
        await session_store.append_transcript_messages(session.id, transcript)
        completed = await session_store.update_status(session.id, SessionStatus.COMPLETED)
        return completed, len(transcript)

    async def scenario() -> None:
        session_store = InMemorySessionStore()
        task_store = CompactionAdmissionBarrierStore()
        provider = _RecordingProvider()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="verified-work-test-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="verified-work-test-model",
                    max_input_chars=1_000,
                ),
                max_user_turns=1,
            ),
        )
        contract = _contract(contract_id="compaction-admission-contract")
        await task_store.publish_work_contract(contract)

        contracted_session, contracted_cursor = await seed_compactable_session(
            app,
            session_store,
            "session:contracted-compaction",
        )
        contracted_task = await task_store.create_running_task(
            TaskCreate(
                task_id="contracted-compaction-task",
                type="verified-work",
                session_id=contracted_session.id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(contracted_session.id),
        )
        session_before = await session_store.load(contracted_session.id)
        checkpoint_before = await session_store.load_checkpoint(contracted_session.id)
        events_before = await session_store.load_events(contracted_session.id)

        rejected_stream = app.compact_session(
            CompactSessionRequest(
                session_id=contracted_session.id,
                idempotency_key="contracted-compaction",
                expected_run_epoch=contracted_session.run_epoch,
                expected_transcript_cursor=contracted_cursor,
            )
        )
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await anext(rejected_stream)

        assert provider.requests == []
        assert await session_store.load(contracted_session.id) == session_before
        assert await session_store.load_checkpoint(contracted_session.id) == checkpoint_before
        assert await session_store.load_events(contracted_session.id) == events_before
        assert await task_store.load_task(contracted_task.id) == contracted_task

        for suffix, epoch_delta, cursor_delta in (
            ("stale-epoch", 1, 0),
            ("stale-cursor", 0, 1),
        ):
            invalid_session, invalid_cursor = await seed_compactable_session(
                app,
                session_store,
                f"session:invalid-compaction:{suffix}",
            )
            invalid_session_before = await session_store.load(invalid_session.id)
            invalid_checkpoint_before = await session_store.load_checkpoint(invalid_session.id)
            invalid_events_before = await session_store.load_events(invalid_session.id)
            provider_request_count = len(provider.requests)

            invalid_stream = app.compact_session(
                CompactSessionRequest(
                    session_id=invalid_session.id,
                    idempotency_key=f"invalid-compaction-{suffix}",
                    expected_run_epoch=invalid_session.run_epoch + epoch_delta,
                    expected_transcript_cursor=invalid_cursor + cursor_delta,
                )
            )
            with pytest.raises(ValueError, match="stale"):
                await anext(invalid_stream)

            assert len(provider.requests) == provider_request_count
            assert await session_store.load(invalid_session.id) == invalid_session_before
            assert (
                await session_store.load_checkpoint(invalid_session.id) == invalid_checkpoint_before
            )
            assert await session_store.load_events(invalid_session.id) == invalid_events_before
            attached = await task_store.create_running_task(
                TaskCreate(
                    task_id=f"invalid-compaction-contract-task-{suffix}",
                    type="verified-work",
                    session_id=invalid_session.id,
                    work_contract=contract.reference(),
                ),
                session_invocation=unattributed_session_invocation_binding(invalid_session.id),
            )
            assert attached.work_contract == contract.reference()

        racing_session, racing_cursor = await seed_compactable_session(
            app,
            session_store,
            "session:racing-compaction-admission",
        )
        racing_session_before = await session_store.load(racing_session.id)
        racing_checkpoint_before = await session_store.load_checkpoint(racing_session.id)
        racing_events_before = await session_store.load_events(racing_session.id)
        provider_request_count = len(provider.requests)
        task_store.blocked_session_id = racing_session.id
        task_store.admission_started = asyncio.Event()
        task_store.allow_admission = asyncio.Event()

        async def collect_racing_compaction() -> list[object]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=racing_session.id,
                        idempotency_key="racing-compaction-admission",
                        expected_run_epoch=racing_session.run_epoch,
                        expected_transcript_cursor=racing_cursor,
                    )
                )
            ]

        racing_compaction = asyncio.create_task(collect_racing_compaction())
        await task_store.admission_started.wait()
        racing_task = await task_store.create_running_task(
            TaskCreate(
                task_id="racing-compaction-contract-task",
                type="verified-work",
                session_id=racing_session.id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(racing_session.id),
        )
        task_store.allow_admission.set()
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await racing_compaction

        assert len(provider.requests) == provider_request_count
        assert await session_store.load(racing_session.id) == racing_session_before
        assert await session_store.load_checkpoint(racing_session.id) == racing_checkpoint_before
        assert await session_store.load_events(racing_session.id) == racing_events_before
        assert await task_store.load_task(racing_task.id) == racing_task

        ordinary_session, ordinary_cursor = await seed_compactable_session(
            app,
            session_store,
            "session:ordinary-compaction",
        )
        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=ordinary_session.id,
                    idempotency_key="ordinary-compaction",
                    expected_run_epoch=ordinary_session.run_epoch,
                    expected_transcript_cursor=ordinary_cursor,
                )
            )
        ]
        assert provider.requests
        assert any(event.type is EventType.CONTEXT_COMPACTION_COMPLETED for event in events)
        with pytest.raises(WorkCompletionConflict, match="prior ordinary session execution"):
            await task_store.create_running_task(
                TaskCreate(
                    task_id="late-contracted-compaction-task",
                    type="verified-work",
                    session_id=ordinary_session.id,
                    work_contract=contract.reference(),
                ),
                session_invocation=unattributed_session_invocation_binding(ordinary_session.id),
            )

    asyncio.run(scenario())


def test_generic_recovery_rejects_contracted_sessions_before_durable_mutation() -> None:
    async def seed_contracted_incomplete_session(
        session_store: InMemorySessionStore,
        task_store: InMemoryTaskStore,
        provider: _RecordingProvider,
        *,
        suffix: str,
    ) -> Task:
        session_id = f"session:generic-recovery-contract:{suffix}"
        await session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "Recover only through verified work.")],
            ),
            identity=SessionIdentity(
                provider_name=provider.name,
                model="verified-work-test-model",
            ),
        )
        await session_store.update_status(session_id, SessionStatus.RUNNING)
        contract = _contract(contract_id=f"generic-recovery-contract-{suffix}")
        await task_store.publish_work_contract(contract)
        return await task_store.create_running_task(
            TaskCreate(
                task_id=f"generic-recovery-contract-task-{suffix}",
                type="verified-work",
                session_id=session_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(session_id),
        )

    async def assert_unchanged(
        session_store: InMemorySessionStore,
        task_store: InMemoryTaskStore,
        task: Task,
        *,
        session_before: object,
        checkpoint_before: object,
        events_before: object,
    ) -> None:
        assert await session_store.load(task.session_id or "missing-session") == session_before
        assert (
            await session_store.load_checkpoint(task.session_id or "missing-session")
            == checkpoint_before
        )
        assert (
            await session_store.load_events(task.session_id or "missing-session") == events_before
        )
        assert await task_store.load_task(task.id) == task

    async def scenario() -> None:
        single_app, single_sessions, single_tasks, _dispatcher, single_provider = (
            _dispatch_test_app()
        )
        single_task = await seed_contracted_incomplete_session(
            single_sessions,
            single_tasks,
            single_provider,
            suffix="single",
        )
        single_session_before = await single_sessions.load(
            single_task.session_id or "missing-session"
        )
        single_checkpoint_before = await single_sessions.load_checkpoint(
            single_task.session_id or "missing-session"
        )
        single_events_before = await single_sessions.load_events(
            single_task.session_id or "missing-session"
        )

        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await single_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=single_task.session_id or "missing-session"
                )
            )
        await assert_unchanged(
            single_sessions,
            single_tasks,
            single_task,
            session_before=single_session_before,
            checkpoint_before=single_checkpoint_before,
            events_before=single_events_before,
        )
        assert single_provider.requests == []

        batch_app, batch_sessions, batch_tasks, _dispatcher, batch_provider = _dispatch_test_app()
        batch_task = await seed_contracted_incomplete_session(
            batch_sessions,
            batch_tasks,
            batch_provider,
            suffix="batch",
        )
        batch_session_before = await batch_sessions.load(batch_task.session_id or "missing-session")
        batch_checkpoint_before = await batch_sessions.load_checkpoint(
            batch_task.session_id or "missing-session"
        )
        batch_events_before = await batch_sessions.load_events(
            batch_task.session_id or "missing-session"
        )

        page = await batch_app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.RUNNING},
                limit=10,
            )
        )

        assert len(page.results) == 1
        assert page.results[0].session_id == batch_task.session_id
        assert page.results[0].actions == (IncompleteSessionRecoveryAction.FAILED,)
        await assert_unchanged(
            batch_sessions,
            batch_tasks,
            batch_task,
            session_before=batch_session_before,
            checkpoint_before=batch_checkpoint_before,
            events_before=batch_events_before,
        )
        assert batch_provider.requests == []

    asyncio.run(scenario())


def test_ordinary_session_admission_and_contract_attachment_are_atomic() -> None:
    async def create_contracted_task(
        store: InMemoryTaskStore,
        contract: WorkContract,
        *,
        session_id: str,
        task_id: str,
    ) -> Task:
        return await store.create_running_task(
            TaskCreate(
                task_id=task_id,
                type="verified-work",
                session_id=session_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(session_id),
        )

    async def scenario() -> None:
        admitted_store = InMemoryTaskStore()
        admitted_contract = _contract(contract_id="admission-wins")
        await admitted_store.publish_work_contract(admitted_contract)
        await admitted_store.admit_ordinary_session_execution("session:admission-wins")
        with pytest.raises(WorkCompletionConflict, match="prior ordinary session execution"):
            await create_contracted_task(
                admitted_store,
                admitted_contract,
                session_id="session:admission-wins",
                task_id="late-contracted-task",
            )
        assert await admitted_store.load_task("late-contracted-task") is None

        contract_store = InMemoryTaskStore()
        attached_contract = _contract(contract_id="contract-wins")
        await contract_store.publish_work_contract(attached_contract)
        attached = await create_contracted_task(
            contract_store,
            attached_contract,
            session_id="session:contract-wins",
            task_id="early-contracted-task",
        )
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await contract_store.admit_ordinary_session_execution("session:contract-wins")
        assert await contract_store.load_task(attached.id) == attached

        racing_store = InMemoryTaskStore()
        racing_contract = _contract(contract_id="atomic-race")
        await racing_store.publish_work_contract(racing_contract)
        admission_result, attachment_result = await asyncio.gather(
            racing_store.admit_ordinary_session_execution("session:atomic-race"),
            create_contracted_task(
                racing_store,
                racing_contract,
                session_id="session:atomic-race",
                task_id="racing-contracted-task",
            ),
            return_exceptions=True,
        )
        successes = sum(
            not isinstance(result, BaseException)
            for result in (admission_result, attachment_result)
        )
        assert successes == 1
        failures = [
            result
            for result in (admission_result, attachment_result)
            if isinstance(result, BaseException)
        ]
        assert len(failures) == 1
        assert isinstance(
            failures[0],
            (TaskCompletionDecisionRequired, WorkCompletionConflict),
        )

    asyncio.run(scenario())


def test_stale_ordinary_worker_cannot_park_successor_claim() -> None:
    class BarrierTaskStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.parking_started = asyncio.Event()
            self.allow_parking = asyncio.Event()

        async def hold_claimed_work_contract_task(
            self,
            task_id: str,
            *,
            worker_id: str,
            contract: WorkContractRef,
        ) -> Task:
            self.parking_started.set()
            await self.allow_parking.wait()
            return await super().hold_claimed_work_contract_task(
                task_id,
                worker_id=worker_id,
                contract=contract,
            )

    async def scenario() -> None:
        store = BarrierTaskStore()
        contract = _contract(contract_id="claim-fenced-worker-parking")
        await store.publish_work_contract(contract)
        task = await store.create_task(
            TaskCreate(
                task_id="claim-fenced-worker-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        app = CayuApp(task_store=store, enable_logging=False)

        stale_worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                lambda _app, _task, _worker_id: asyncio.sleep(0),
                worker_id="stale-worker",
                query=TaskQuery(type=task.type),
                lease_seconds=1,
                poll_interval_s=0.001,
                max_tasks=1,
            )
        )
        await store.parking_started.wait()
        stale_claim = await store.load_task(task.id)
        assert stale_claim is not None
        assert stale_claim.lease_expires_at is not None
        await asyncio.sleep(
            max(
                0.0,
                (stale_claim.lease_expires_at - datetime.now(UTC)).total_seconds(),
            )
            + 0.02
        )
        reclaimed = await store.reclaim_expired(query=TaskQuery(type=task.type))
        assert [item.id for item in reclaimed] == [task.id]
        successor = await store.claim_task(
            "successor-worker",
            TaskQuery(type=task.type),
            lease_seconds=30,
        )
        assert successor is not None

        store.allow_parking.set()
        with pytest.raises(TaskClaimLost):
            await stale_worker

        current = await store.load_task(task.id)
        assert current is not None
        assert current.status is TaskStatus.CLAIMED
        assert current.worker_id == "successor-worker"
        assert current.lease_expires_at == successor.lease_expires_at

    asyncio.run(scenario())


def test_approval_continuation_rejects_session_contract_before_effects() -> None:
    class AdoptedContractStore(InMemoryTaskStore):
        supports_verified_work_contracts = False
        verified_work_mutations_are_cancellation_quiescent = True

    class ApprovalProvider(ModelProvider):
        name = "verified-work-approval-provider"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelStreamEvent.tool_call(
                    id="contract-gated-call",
                    name="contract_gated_tool",
                    arguments={"value": "must-not-run"},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            yield ModelStreamEvent.text_delta("unexpected continuation")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    class RecordingTool(Tool):
        spec = ToolSpec(
            name="contract_gated_tool",
            description="Record whether an approved effect executes.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            effect="external",
        )

        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, object]] = []

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx
            self.calls.append(dict(args))
            return ToolResult(content="effect committed")

    class RequireApproval(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            del request
            return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)

    async def scenario() -> None:
        session_id = "session:contract-gated-approval"
        session_store = InMemorySessionStore()
        task_store = AdoptedContractStore()
        provider = ApprovalProvider()
        tool = RecordingTool()
        app = CayuApp(
            session_store=session_store,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="verified-work-approval-model"),
            tools=[tool],
            tool_policy=RequireApproval(),
        )

        paused = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "Request the guarded effect.")],
                )
            )
        ]
        assert any(event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED for event in paused)
        durable_approval_event = next(
            event
            for event in await session_store.load_events(session_id)
            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval = PendingToolApprovalEventView.from_event(durable_approval_event)

        # Re-open the paused session with the verified-work store, matching a
        # process/store adoption boundary rather than attempting a forbidden
        # late contract attachment to an already admitted ordinary session.
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="verified-work-approval-model"),
            tools=[tool],
            tool_policy=RequireApproval(),
        )

        contract = _contract(contract_id="approval-continuation-contract")
        await task_store.publish_work_contract(contract)
        contracted_task = await task_store.create_running_task(
            TaskCreate(
                task_id="approval-continuation-task",
                type="verified-work",
                session_id=session_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(session_id),
        )
        session_before = await session_store.load(session_id)
        checkpoint_before = await session_store.load_checkpoint(session_id)
        events_before = await session_store.load_events(session_id)

        stream = app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
            )
        )
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await anext(stream)

        assert tool.calls == []
        assert len(provider.requests) == 1
        assert await session_store.load(session_id) == session_before
        assert await session_store.load_checkpoint(session_id) == checkpoint_before
        assert await session_store.load_events(session_id) == events_before
        unchanged = await task_store.load_task(contracted_task.id)
        assert unchanged is not None
        assert unchanged.status is TaskStatus.RUNNING

        failed = await task_store.fail_task(
            contracted_task.id,
            {"message": "The contracted work stopped before approval."},
        )
        assert failed.status is TaskStatus.FAILED

        # Terminal task state is not evidence that the paused session's work was
        # released from its contract. The pending external effect remains fenced.
        stream = app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
            )
        )
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await anext(stream)

        assert tool.calls == []
        assert len(provider.requests) == 1
        assert await session_store.load(session_id) == session_before
        assert await session_store.load_checkpoint(session_id) == checkpoint_before
        assert await session_store.load_events(session_id) == events_before
        assert await task_store.load_task(contracted_task.id) == failed

    asyncio.run(scenario())


def test_custom_dispatcher_cannot_bypass_contract_authority_or_attachment_admission() -> None:
    class RecordingDispatcher(Dispatcher):
        def __init__(self) -> None:
            self.requests: list[DispatchRequest] = []

        async def submit(self, runtime, request: DispatchRequest) -> DispatchHandle:
            del runtime
            self.requests.append(request)
            return DispatchHandle(
                dispatch_id=request.dispatch_id,
                session_id=request.session_id,
                task_id=request.task_id,
                backend="recording",
                status=DispatchStatus.SUBMITTED,
            )

    async def scenario() -> None:
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        dispatcher = RecordingDispatcher()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        contract = _contract(contract_id="custom-dispatch-contract")
        await task_store.publish_work_contract(contract)

        contracted_session_id = "session:custom-dispatch:contracted"
        await _create_dispatch_test_session(app, contracted_session_id)
        await task_store.create_running_task(
            TaskCreate(
                task_id="custom-dispatch-session-task",
                type="verified-work",
                session_id=contracted_session_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(contracted_session_id),
        )
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await app.dispatch(
                DispatchRequest(
                    session_id=contracted_session_id,
                    dispatch_id="custom-dispatch-contracted-session",
                    messages=[Message.text("user", "Do not submit contracted session work.")],
                )
            )

        contracted_task = await task_store.create_task(
            TaskCreate(
                task_id="custom-dispatch-direct-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        direct_task_session_id = "session:custom-dispatch:direct-task"
        await _create_dispatch_test_session(app, direct_task_session_id)
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await app.dispatch(
                DispatchRequest(
                    session_id=direct_task_session_id,
                    dispatch_id="custom-dispatch-contracted-task",
                    task_id=contracted_task.id,
                    messages=[Message.text("user", "Do not submit contracted task work.")],
                )
            )
        assert dispatcher.requests == []

        ordinary_session_id = "session:custom-dispatch:ordinary"
        await _create_dispatch_test_session(app, ordinary_session_id)
        handle = await app.dispatch(
            DispatchRequest(
                session_id=ordinary_session_id,
                dispatch_id="custom-dispatch-ordinary",
                messages=[Message.text("user", "Submit ordinary work.")],
            )
        )
        assert handle.status is DispatchStatus.SUBMITTED
        assert [request.dispatch_id for request in dispatcher.requests] == [
            "custom-dispatch-ordinary"
        ]
        with pytest.raises(WorkCompletionConflict, match="prior ordinary session execution"):
            await task_store.create_running_task(
                TaskCreate(
                    task_id="custom-dispatch-late-contract-task",
                    type="verified-work",
                    session_id=ordinary_session_id,
                    work_contract=contract.reference(),
                ),
                session_invocation=unattributed_session_invocation_binding(ordinary_session_id),
            )

    asyncio.run(scenario())


def test_custom_dispatcher_requires_resolved_task_authority_before_submission() -> None:
    task_id = "custom-dispatch-racing-task"

    class StaleLookupTaskStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.lookup_completed = asyncio.Event()
            self.release_lookup = asyncio.Event()

        async def load_task(self, requested_task_id: str) -> Task | None:
            task = await super().load_task(requested_task_id)
            if requested_task_id == task_id and not self.lookup_completed.is_set():
                assert task is None
                self.lookup_completed.set()
                await self.release_lookup.wait()
            return task

    class RecordingDispatcher(Dispatcher):
        def __init__(self) -> None:
            self.requests: list[DispatchRequest] = []

        async def submit(self, runtime, request: DispatchRequest) -> DispatchHandle:
            del runtime
            self.requests.append(request)
            return DispatchHandle(
                dispatch_id=request.dispatch_id,
                session_id=request.session_id,
                task_id=request.task_id,
                backend="recording",
                status=DispatchStatus.SUBMITTED,
            )

    async def scenario() -> None:
        dispatcher_without_store = RecordingDispatcher()
        app_without_store = CayuApp(
            dispatcher=dispatcher_without_store,
            enable_logging=False,
        )
        with pytest.raises(RuntimeError, match="task_store is required"):
            await app_without_store.dispatch(
                DispatchRequest(
                    session_id="session:custom-dispatch:no-store",
                    task_id="custom-dispatch-no-store-task",
                    messages=[Message.text("user", "Do not submit unverified task authority.")],
                )
            )
        assert dispatcher_without_store.requests == []

        session_store = InMemorySessionStore()
        task_store = StaleLookupTaskStore()
        dispatcher = RecordingDispatcher()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        contract = _contract(contract_id="custom-dispatch-racing-contract")
        await task_store.publish_work_contract(contract)
        session_id = "session:custom-dispatch:racing-task"
        await _create_dispatch_test_session(app, session_id)

        dispatch_call = asyncio.create_task(
            app.dispatch(
                DispatchRequest(
                    session_id=session_id,
                    dispatch_id="custom-dispatch-racing-authority",
                    task_id=task_id,
                    messages=[Message.text("user", "Do not submit stale task authority.")],
                )
            )
        )
        await asyncio.wait_for(task_store.lookup_completed.wait(), timeout=1)
        contracted_task = await task_store.create_task(
            TaskCreate(
                task_id=task_id,
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        task_store.release_lookup.set()

        with pytest.raises(KeyError, match="Task not found"):
            await dispatch_call
        assert dispatcher.requests == []

        # Rejection happened before ordinary-session admission. The now-durable
        # contracted task can still claim the same session authority.
        started = await task_store.start_task(
            contracted_task.id,
            session_id=session_id,
            session_invocation=await task_backed_session_invocation(
                task_store,
                contracted_task.id,
                session_id,
            ),
        )
        assert started.status is TaskStatus.RUNNING
        assert started.session_id == session_id

    asyncio.run(scenario())


def test_custom_dispatcher_rejects_substituted_task_lookup_authority() -> None:
    contracted_task_id = "custom-dispatch-substituted-contracted-task"
    substitute_task_id = "custom-dispatch-substituted-ordinary-task"

    class SubstitutingTaskLookupStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def load_task(self, requested_task_id: str) -> Task | None:
            if requested_task_id == contracted_task_id:
                return await super().load_task(substitute_task_id)
            return await super().load_task(requested_task_id)

    class RecordingDispatcher(Dispatcher):
        def __init__(self) -> None:
            self.requests: list[DispatchRequest] = []

        async def submit(self, runtime, request: DispatchRequest) -> DispatchHandle:
            del runtime
            self.requests.append(request)
            return DispatchHandle(
                dispatch_id=request.dispatch_id,
                session_id=request.session_id,
                task_id=request.task_id,
                backend="recording",
                status=DispatchStatus.SUBMITTED,
            )

    async def scenario() -> None:
        session_store = InMemorySessionStore()
        task_store = SubstitutingTaskLookupStore()
        dispatcher = RecordingDispatcher()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        contract = _contract(contract_id="custom-dispatch-substituted-contract")
        await task_store.publish_work_contract(contract)
        await task_store.create_task(
            TaskCreate(
                task_id=contracted_task_id,
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        await task_store.create_task(TaskCreate(task_id=substitute_task_id, type="ordinary-work"))
        session_id = "session:custom-dispatch:substituted-task"
        await _create_dispatch_test_session(app, session_id)

        with pytest.raises(RuntimeError, match="exact requested task"):
            await app.dispatch(
                DispatchRequest(
                    session_id=session_id,
                    dispatch_id="custom-dispatch-substituted-authority",
                    task_id=contracted_task_id,
                    messages=[Message.text("user", "Do not trust a substituted task row.")],
                )
            )

        assert dispatcher.requests == []
        started = await task_store.create_running_task(
            TaskCreate(
                task_id="custom-dispatch-substitution-session-owner",
                type="verified-work",
                session_id=session_id,
                work_contract=contract.reference(),
            ),
            session_invocation=await app.session_invocation_for_dispatch(session_id),
        )
        assert started.status is TaskStatus.RUNNING
        assert started.session_id == session_id

    asyncio.run(scenario())


@pytest.mark.parametrize("use_uncontracted_task_id", [False, True])
def test_inline_dispatch_rejects_session_linked_contract_authority(
    use_uncontracted_task_id: bool,
) -> None:
    async def scenario() -> None:
        app, _session_store, task_store, _dispatcher, provider = _dispatch_test_app()
        session_id = f"session:inline-contract:{use_uncontracted_task_id}"
        await _create_dispatch_test_session(app, session_id)
        contract = _contract(contract_id=f"inline-contract-{use_uncontracted_task_id}")
        await task_store.publish_work_contract(contract)
        contracted_task = await task_store.create_running_task(
            TaskCreate(
                task_id=f"inline-contracted-task-{use_uncontracted_task_id}",
                type="verified-work",
                session_id=session_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(session_id),
        )
        uncontracted_task = None
        if use_uncontracted_task_id:
            uncontracted_task = await task_store.create_task(
                TaskCreate(
                    task_id="inline-uncontracted-task",
                    type="ordinary-work",
                )
            )

        stream = app.dispatch_inline(
            DispatchRequest(
                session_id=session_id,
                task_id=None if uncontracted_task is None else uncontracted_task.id,
                messages=[Message.text("user", "Do not bypass contract authority.")],
            )
        )
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await anext(stream)

        unchanged_contract_task = await task_store.load_task(contracted_task.id)
        assert unchanged_contract_task is not None
        assert unchanged_contract_task.status is TaskStatus.RUNNING
        if uncontracted_task is not None:
            unchanged_uncontracted_task = await task_store.load_task(uncontracted_task.id)
            assert unchanged_uncontracted_task is not None
            assert unchanged_uncontracted_task.status is TaskStatus.PENDING
            assert unchanged_uncontracted_task.session_id is None
        assert len(provider.requests) == 0

    asyncio.run(scenario())


def test_queued_dispatch_rejects_contracted_sessions_without_requeue() -> None:
    async def attach_contract_task(
        task_store: InMemoryTaskStore,
        contract: WorkContract,
        *,
        session_id: str,
        task_id: str,
    ) -> Task:
        return await task_store.create_running_task(
            TaskCreate(
                task_id=task_id,
                type="verified-work",
                session_id=session_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(session_id),
        )

    async def scenario() -> None:
        app, _session_store, task_store, dispatcher, provider = _dispatch_test_app()
        contract = _contract(contract_id="queued-dispatch-contract")
        await task_store.publish_work_contract(contract)

        rejected_before_publish_session = "session:queued-contract:preflight"
        await _create_dispatch_test_session(app, rejected_before_publish_session)
        await attach_contract_task(
            task_store,
            contract,
            session_id=rejected_before_publish_session,
            task_id="queued-contract-preflight-task",
        )
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await app.dispatch(
                DispatchRequest(
                    session_id=rejected_before_publish_session,
                    dispatch_id="queued-contract-preflight",
                    messages=[Message.text("user", "Reject before queue publication.")],
                )
            )
        assert await task_store.list_tasks(TaskQuery(type=dispatcher.task_type)) == []

        persisted_session = "session:queued-contract:persisted"
        await _create_dispatch_test_session(app, persisted_session)
        submitted = await app.dispatch(
            DispatchRequest(
                session_id=persisted_session,
                dispatch_id="queued-contract-persisted",
                messages=[Message.text("user", "Persist before contract attachment.")],
            )
        )
        with pytest.raises(WorkCompletionConflict, match="prior ordinary session execution"):
            await attach_contract_task(
                task_store,
                contract,
                session_id=persisted_session,
                task_id="queued-contract-persisted-task",
            )

        completed = await dispatcher.process_next(
            app,
            worker_id="queued-contract-worker",
        )
        assert completed is not None
        assert completed.status is DispatchStatus.COMPLETED
        queue_task_id = submitted.metadata["queue_task_id"]
        queue_task = await task_store.load_task(queue_task_id)
        assert queue_task is not None
        assert queue_task.status is TaskStatus.COMPLETED
        assert (
            await dispatcher.process_next(
                app,
                worker_id="queued-contract-worker-retry",
            )
            is None
        )
        assert len(provider.requests) == 1

    asyncio.run(scenario())


def test_contracted_task_rejects_every_legacy_completion_entrance() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract()
        await store.publish_work_contract(contract)
        await store.create_task(
            TaskCreate(
                task_id="completion-bypass",
                type="bid-bypass",
                work_contract=contract.reference(),
            )
        )
        claimed = await store.claim_task(
            "worker-a",
            TaskQuery(type="bid-bypass"),
        )
        assert claimed is not None
        running = await store.attach_task(
            claimed.id,
            session_id="session:bypass",
            session_invocation=await task_backed_session_invocation(
                store,
                claimed.id,
                "session:bypass",
            ),
            worker_id="worker-a",
        )

        with pytest.raises(TaskCompletionDecisionRequired):
            await store.complete_task(running.id, {"unauthorized": True}, worker_id="worker-a")
        with pytest.raises(TaskCompletionDecisionRequired):
            await store.terminalize_task(
                TaskTerminalizationRequest(
                    task_id=running.id,
                    worker_id="worker-a",
                    kind=TaskTerminalKind.COMPLETED,
                    result={"unauthorized": True},
                    idempotency_key="legacy-completion",
                )
            )

        unchanged = await store.load_task(running.id)
        assert unchanged is not None
        assert unchanged.status is TaskStatus.RUNNING
        assert unchanged.result is None
        assert (
            await store.load_task_terminalization_receipt(running.id, "legacy-completion") is None
        )

        failed = await store.fail_task(
            running.id,
            {"message": "Independent execution failure remains allowed."},
            worker_id="worker-a",
        )
        assert failed.status is TaskStatus.FAILED

    asyncio.run(scenario())


def test_unmigrated_persistent_store_fails_closed_before_contract_binding(tmp_path) -> None:
    async def scenario() -> None:
        store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
        try:
            assert store.supports_verified_work_contracts is False
            contract = _contract()
            with pytest.raises(NotImplementedError, match="verified work contracts"):
                await store.publish_work_contract(contract)
            with pytest.raises(NotImplementedError, match="work-contract task bindings"):
                await store.create_task(
                    TaskCreate(
                        task_id="unsupported-contract-task",
                        type="bid",
                        work_contract=contract.reference(),
                    )
                )
            assert await store.load_task("unsupported-contract-task") is None

            provider = _RecordingProvider()
            app = CayuApp(
                session_store=InMemorySessionStore(),
                task_store=store,
                enable_logging=False,
            )
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="verified-work-test-model"))
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        messages=[Message.text("user", "Run ordinary uncontracted work.")],
                    )
                )
            ]
            assert provider.requests
            assert events[-1].type is EventType.SESSION_COMPLETED
        finally:
            await store.close()

    asyncio.run(scenario())


def test_contract_publication_requires_exact_content_and_lineage() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        first = _contract()
        await store.publish_work_contract(first)

        with pytest.raises(WorkContractConflict, match="different content"):
            await store.publish_work_contract(_contract(objective="Conflicting objective."))

        second = _contract(
            objective="Publish a complete approved package with final pricing.",
            version=2,
            supersedes=first.reference(),
        )
        assert await store.publish_work_contract(second) == second

        missing_predecessor = _contract(
            contract_id="unpublished-contract",
            objective="Another second version.",
            version=2,
            supersedes=WorkContractRef(
                contract_id="unpublished-contract",
                version=1,
                fingerprint=_digest("unpublished-contract-v1"),
            ),
        )
        with pytest.raises(WorkContractConflict, match="predecessor has not been published"):
            await store.publish_work_contract(missing_predecessor)

    asyncio.run(scenario())


def test_application_creates_and_loads_contract_before_binding_task() -> None:
    async def scenario() -> None:
        task_store = InMemoryTaskStore()
        app = CayuApp(
            session_store=InMemorySessionStore(),
            task_store=task_store,
            enable_logging=False,
        )
        draft = WorkContractDraft.model_validate(
            _contract().model_dump(
                mode="python",
                warnings=False,
                exclude={"fingerprint"},
            )
        )

        contract = await app.create_work_contract(draft)
        assert await app.load_work_contract(contract.reference()) == contract
        task = await app.create_task(
            TaskCreate(
                task_id="app-contracted-task",
                type="bid",
                work_contract=contract.reference(),
            )
        )
        assert task.work_contract == contract.reference()

    asyncio.run(scenario())


def test_application_rejects_retry_series_with_verified_work_contract_before_mutation() -> None:
    contract = _contract(contract_id="retry-series-contract-conflict")
    policy = TaskRetryPolicy(max_attempts=2)

    class RecordingTaskStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.create_called = False

        async def create_task(self, request: TaskCreate) -> Task:
            self.create_called = True
            return await super().create_task(request)

    with pytest.raises(ValidationError, match="cannot use verified work contracts"):
        TaskCreate(
            task_id="retry-series-contract-construction",
            type="verified-work",
            retry_policy=policy,
            work_contract=contract.reference(),
        )

    async def scenario() -> None:
        task_store = RecordingTaskStore()
        await task_store.publish_work_contract(contract)
        app = CayuApp(task_store=task_store, enable_logging=False)
        request = TaskCreate(
            task_id="retry-series-contract-public-boundary",
            type="verified-work",
            work_contract=contract.reference(),
        )
        object.__setattr__(request, "retry_policy", policy)

        with pytest.raises(ValueError, match="Task creation request is invalid"):
            await app.create_task(request)

        assert await task_store.load_task(request.task_id or "") is None
        assert await task_store.list_tasks() == []
        assert task_store.create_called is False

    asyncio.run(scenario())


def test_application_authenticates_contracted_task_parent_and_session_provenance() -> None:
    async def scenario() -> None:
        app, _session_store, task_store, _dispatcher, _provider = _dispatch_test_app()
        contract = _contract(contract_id="contracted-create-provenance-contract")
        await task_store.publish_work_contract(contract)

        parent = await app.create_task(
            TaskCreate(
                task_id="contracted-create-provenance-parent",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        child = await app.create_task(
            TaskCreate(
                task_id="contracted-create-provenance-child",
                type="verified-work",
                parent_task_id=parent.id,
                work_contract=contract.reference(),
            )
        )
        assert child.invocation.origin == parent.invocation.origin
        assert child.invocation.root_invocation_id == parent.invocation.root_invocation_id

        session_id = "session:contracted-create-provenance"
        await _create_dispatch_test_session(app, session_id)
        session_snapshot = await app.session_store.load_invocation_snapshot(session_id)
        assert session_snapshot is not None
        attached = await app.create_task(
            TaskCreate(
                task_id="contracted-create-provenance-session-task",
                type="verified-work",
                session_id=session_id,
                work_contract=contract.reference(),
            )
        )
        assert attached.invocation.origin == session_snapshot.invocation.origin
        assert (
            attached.invocation.root_invocation_id == session_snapshot.invocation.root_invocation_id
        )
        assert attached.invocation.root_session_id == session_snapshot.invocation.root_session_id

    asyncio.run(scenario())


def test_application_rejects_substituted_contracted_task_creation_results() -> None:
    contract = _contract(contract_id="contracted-create-result-contract")
    alternate_contract = _contract(contract_id="contracted-create-result-alternate")
    mutations: tuple[tuple[str, dict[str, object]], ...] = (
        ("id", {"id": "different-created-task"}),
        ("type", {"type": "different-type"}),
        ("title", {"title": "Different title"}),
        ("description", {"description": "Different description"}),
        ("session", {"session_id": "session:different-created-task"}),
        ("parent", {"parent_task_id": "different-parent"}),
        ("assignment", {"assigned_agent_name": "different-agent"}),
        ("availability", {"available_at": datetime.now(UTC) + timedelta(days=1)}),
        ("input", {"input": {"different": True}}),
        ("metadata", {"metadata": {"different": True}}),
        ("contract", {"work_contract": alternate_contract.reference()}),
        (
            "invocation",
            {
                "invocation": TaskInvocation(
                    origin=InvocationOrigin(trust=InvocationOriginTrust.UNATTRIBUTED),
                    root_invocation_id=str(uuid4()),
                    source=TaskExecutionSource.SDK_TASK,
                )
            },
        ),
        (
            "lifecycle",
            {
                "status": TaskStatus.COMPLETED,
                "result": {"unauthorized": True},
                "completed_at": datetime.now(UTC),
            },
        ),
    )

    class SubstitutingTaskCreationStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self, update: dict[str, object]) -> None:
            super().__init__()
            self.update = update

        async def create_task(self, request: TaskCreate) -> Task:
            task = await super().create_task(request)
            return task.model_copy(update=self.update)

    async def scenario() -> None:
        for suffix, update in mutations:
            task_store = SubstitutingTaskCreationStore(update)
            await task_store.publish_work_contract(contract)
            app = CayuApp(task_store=task_store, enable_logging=False)
            task_id = f"contracted-create-result-{suffix}"

            with pytest.raises(
                WorkContractConflict,
                match="exact contracted task creation request",
            ):
                await app.create_task(
                    TaskCreate(
                        task_id=task_id,
                        type="verified-work",
                        input={"expected": True},
                        metadata={"expected": True},
                        work_contract=contract.reference(),
                    )
                )

            persisted = await InMemoryTaskStore.load_task(task_store, task_id)
            assert persisted is not None
            assert persisted.id == task_id
            assert persisted.status is TaskStatus.PENDING
            assert persisted.work_contract == contract.reference()

    asyncio.run(scenario())


def test_application_rejects_secret_bearing_contracted_task_result_provenance(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "contracted-task-result-provenance-secret"

    class SubstitutingTaskCreationStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def create_task(self, request: TaskCreate) -> Task:
            task = await super().create_task(request)
            invocation = task.invocation.model_copy(
                update={
                    "origin": InvocationOrigin(
                        trust=InvocationOriginTrust.HOST_ASSERTED,
                        subject=secret,
                    )
                }
            )
            return task.model_copy(update={"invocation": invocation})

    async def scenario() -> BaseException:
        task_store = SubstitutingTaskCreationStore()
        contract = _contract(contract_id="secret-result-provenance-contract")
        await task_store.publish_work_contract(contract)
        app = CayuApp(
            task_store=task_store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )

        with pytest.raises(
            WorkContractConflict,
            match="exact contracted task creation request",
        ) as exc:
            await app.create_task(
                TaskCreate(
                    task_id="secret-result-provenance-task",
                    type="verified-work",
                    work_contract=contract.reference(),
                )
            )
        persisted = await InMemoryTaskStore.load_task(
            task_store,
            "secret-result-provenance-task",
        )
        assert persisted is not None
        assert persisted.invocation.origin.trust is InvocationOriginTrust.UNATTRIBUTED
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


def test_application_rejects_secret_bearing_work_contract_public_identities() -> None:
    async def scenario() -> None:
        secret = "work-contract-secret"
        task_store = InMemoryTaskStore()
        secret_contract = _contract(contract_id=f"bid-{secret}")
        await task_store.publish_work_contract(secret_contract)
        app = CayuApp(
            task_store=task_store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        draft = WorkContractDraft.model_validate(
            secret_contract.model_dump(
                mode="python",
                warnings=False,
                exclude={"fingerprint"},
            )
        )

        with pytest.raises(ValueError, match="public identity contains a workload secret"):
            await app.create_work_contract(draft)
        with pytest.raises(ValueError, match="workload secret.*lookup"):
            await app.load_work_contract(secret_contract.reference())
        with pytest.raises(ValueError, match="identity contains a workload secret"):
            await app.create_task(
                TaskCreate(
                    task_id="secret-contract-task",
                    type="bid",
                    work_contract=secret_contract.reference(),
                )
            )
        assert await task_store.load_task("secret-contract-task") is None

    asyncio.run(scenario())


def test_application_preserves_ordinary_task_validation_details_only() -> None:
    async def scenario() -> None:
        app_without_store = CayuApp(enable_logging=False)
        invalid_without_store = TaskCreate(
            task_id="invalid-ordinary-task-without-store",
            type="ordinary",
        )
        object.__setattr__(invalid_without_store, "type", "")
        with pytest.raises(ValidationError, match="cannot be blank"):
            await app_without_store.create_task(invalid_without_store)

        task_store = InMemoryTaskStore()
        contract = _contract(contract_id="task-validation-contract")
        await task_store.publish_work_contract(contract)
        app = CayuApp(task_store=task_store, enable_logging=False)

        ordinary_request = TaskCreate(task_id="invalid-ordinary-task", type="ordinary")
        object.__setattr__(ordinary_request, "type", "")
        with pytest.raises(ValidationError, match="cannot be blank"):
            await app.create_task(ordinary_request)

        contracted_request = TaskCreate(
            task_id="invalid-contracted-task",
            type="contracted",
            work_contract=contract.reference(),
        )
        object.__setattr__(contracted_request, "type", "")
        with pytest.raises(ValueError, match="Task creation request is invalid"):
            await app.create_task(contracted_request)

        assert await task_store.load_task("invalid-ordinary-task") is None
        assert await task_store.load_task("invalid-contracted-task") is None

    asyncio.run(scenario())


def test_work_contract_rejections_do_not_retain_secrets_in_diagnostics(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "verified-work-contract-diagnostic-secret-canary"

    class SecretRenderingValue:
        def __repr__(self) -> str:
            return f"SecretRenderingValue({secret})"

        def __str__(self) -> str:
            return secret

    async def scenario() -> list[BaseException]:
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        secret_contract = _contract(contract_id=f"bid-{secret}")
        await task_store.publish_work_contract(secret_contract)

        loaded_contract_document = _contract(contract_id="loaded-secret-contract").model_dump(
            mode="python", warnings=False, exclude={"fingerprint"}
        )
        loaded_contract_document["verifier"]["verifier_id"] = f"verifier-{secret}"
        loaded_secret_contract = work_contract_from_draft(
            WorkContractDraft.model_validate(loaded_contract_document)
        )
        await task_store.publish_work_contract(loaded_secret_contract)
        safe_task_contract = _contract(contract_id="diagnostic-safe-task-contract")
        await task_store.publish_work_contract(safe_task_contract)

        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        secret_draft = WorkContractDraft.model_validate(
            secret_contract.model_dump(
                mode="python",
                warnings=False,
                exclude={"fingerprint"},
            )
        )
        hostile_draft = WorkContractDraft.model_validate(
            _contract(contract_id="hostile-draft-contract").model_dump(
                mode="python",
                warnings=False,
                exclude={"fingerprint"},
            )
        )
        object.__setattr__(hostile_draft, "contract_id", SecretRenderingValue())
        hostile_reference = _contract(contract_id="hostile-reference-contract").reference()
        object.__setattr__(hostile_reference, "contract_id", SecretRenderingValue())
        hostile_task_request = TaskCreate(
            task_id="hostile-contract-reference-task",
            type="bid",
            input={"private": secret},
            work_contract=_contract(contract_id="hostile-task-contract").reference(),
        )
        assert hostile_task_request.work_contract is not None
        object.__setattr__(
            hostile_task_request.work_contract,
            "contract_id",
            SecretRenderingValue(),
        )

        errors: list[BaseException] = []
        with pytest.raises(ValueError, match="public identity contains a workload secret") as exc:
            await app.create_work_contract(secret_draft)
        errors.append(exc.value)
        with pytest.raises(ValueError, match="creation request is invalid") as exc:
            await app.create_work_contract(hostile_draft)
        errors.append(exc.value)
        with pytest.raises(ValueError, match="workload secret.*lookup") as exc:
            await app.load_work_contract(secret_contract.reference())
        errors.append(exc.value)
        with pytest.raises(ValueError, match="lookup reference is invalid") as exc:
            await app.load_work_contract(hostile_reference)
        errors.append(exc.value)
        with pytest.raises(ValueError, match="Loaded work contract contains") as exc:
            await app.load_work_contract(loaded_secret_contract.reference())
        errors.append(exc.value)
        with pytest.raises(ValueError, match="identity contains a workload secret") as exc:
            await app.create_task(
                TaskCreate(
                    task_id="diagnostic-secret-contract-task",
                    type="bid",
                    input={"private": secret},
                    metadata={"private": secret},
                    work_contract=secret_contract.reference(),
                )
            )
        errors.append(exc.value)
        secret_task_id = f"diagnostic-{secret}-task"
        with pytest.raises(ValueError, match="Task identity contains a workload secret") as exc:
            await app.create_task(
                TaskCreate(
                    task_id=secret_task_id,
                    type="bid",
                    input={"private": secret},
                    metadata={"private": secret},
                    work_contract=safe_task_contract.reference(),
                )
            )
        errors.append(exc.value)
        secret_root_session_task_id = "diagnostic-secret-root-session-task"
        with pytest.raises(ValueError, match="Task invocation identity contains") as exc:
            await app.create_task(
                TaskCreate(
                    task_id=secret_root_session_task_id,
                    type="bid",
                    session_id=secret,
                    work_contract=safe_task_contract.reference(),
                )
            )
        errors.append(exc.value)
        secret_session_invocation_task_id = "diagnostic-secret-session-invocation-task"
        await session_store.create(
            run_request_with_runtime_invocation(
                RunRequest(
                    agent_name="assistant",
                    session_id="diagnostic-safe-session",
                    messages=[],
                ),
                source=SessionExecutionSource.HTTP_RUN,
                verified_origin=InvocationOrigin(
                    trust=InvocationOriginTrust.SERVER_VERIFIED,
                    subject=secret,
                ),
            ),
            identity=SessionIdentity(
                provider_name="diagnostic-provider",
                model="diagnostic-model",
            ),
        )
        with pytest.raises(ValueError, match="Task invocation identity contains") as exc:
            await app.create_task(
                TaskCreate(
                    task_id=secret_session_invocation_task_id,
                    type="bid",
                    session_id="diagnostic-safe-session",
                    work_contract=safe_task_contract.reference(),
                )
            )
        errors.append(exc.value)
        secret_parent_task_id = "diagnostic-secret-parent-invocation"
        await task_store.create_task(
            TaskCreate(
                task_id=secret_parent_task_id,
                type="bid",
                invocation_origin=InvocationOriginClaim(subject=secret),
            )
        )
        secret_parent_child_task_id = "diagnostic-secret-parent-child"
        with pytest.raises(ValueError, match="Task invocation identity contains") as exc:
            await app.create_task(
                TaskCreate(
                    task_id=secret_parent_child_task_id,
                    type="bid",
                    parent_task_id=secret_parent_task_id,
                    work_contract=safe_task_contract.reference(),
                )
            )
        errors.append(exc.value)
        with pytest.raises(ValueError, match="Task creation request is invalid") as exc:
            await app.create_task(hostile_task_request)
        errors.append(exc.value)
        assert await task_store.load_task("diagnostic-secret-contract-task") is None
        assert await task_store.load_task(secret_task_id) is None
        assert await task_store.load_task(secret_root_session_task_id) is None
        assert await task_store.load_task(secret_session_invocation_task_id) is None
        assert await task_store.load_task(secret_parent_child_task_id) is None
        assert await task_store.load_task("hostile-contract-reference-task") is None
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


def test_fatal_public_model_copy_failures_are_detached_and_redacted(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "verified-work-fatal-validation-secret-canary"

    class FatalMapping(Mapping[str, object]):
        def __init__(self, *, grouped: bool) -> None:
            self.grouped = grouped

        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            if self.grouped:
                raise BaseExceptionGroup(
                    f"Grouped validation failure containing {secret}",
                    [
                        SystemExit(f"Fatal validation signal containing {secret}"),
                        RuntimeError(f"Validation failure containing {secret}"),
                    ],
                )
            raise SystemExit(f"Fatal validation signal containing {secret}")

        def __len__(self) -> int:
            return 1

        def __repr__(self) -> str:
            return f"FatalMapping({secret})"

    class HostileResultStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def publish_work_contract(self, contract: WorkContract) -> WorkContract:
            published = await super().publish_work_contract(contract)
            if contract.contract_id == "fatal-store-contract":
                object.__setattr__(published, "verifier", FatalMapping(grouped=True))
            return published

    def draft(contract: WorkContract) -> WorkContractDraft:
        return WorkContractDraft.model_validate(
            contract.model_dump(mode="python", warnings=False, exclude={"fingerprint"})
        )

    async def scenario() -> list[BaseException]:
        task_store = HostileResultStore()
        app = CayuApp(
            task_store=task_store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )

        errors: list[BaseException] = []
        hostile_draft = draft(_contract(contract_id="fatal-caller-contract"))
        object.__setattr__(hostile_draft, "verifier", FatalMapping(grouped=False))
        with pytest.raises(SystemExit) as exc:
            await app.create_work_contract(hostile_draft)
        errors.append(exc.value)

        with pytest.raises(BaseExceptionGroup) as exc:
            await app.create_work_contract(draft(_contract(contract_id="fatal-store-contract")))
        errors.append(exc.value)
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


def test_work_contract_store_conflicts_do_not_retain_sensitive_payloads(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "verified-work-store-conflict-secret-canary"

    async def scenario() -> list[BaseException]:
        task_store = InMemoryTaskStore()
        published = _contract(contract_id="diagnostic-store-conflict")
        await task_store.publish_work_contract(published)
        conflicting = _contract(
            contract_id=published.contract_id,
            objective=f"Reject this conflicting definition containing {secret}.",
        )
        conflicting_draft = WorkContractDraft.model_validate(
            conflicting.model_dump(
                mode="python",
                warnings=False,
                exclude={"fingerprint"},
            )
        )
        admitted_session_id = "session:diagnostic-contract-attachment-conflict"
        await task_store.admit_ordinary_session_execution(admitted_session_id)
        app = CayuApp(
            task_store=task_store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )

        errors: list[BaseException] = []
        with pytest.raises(WorkContractConflict, match="durable identity conflicts") as exc:
            await app.create_work_contract(conflicting_draft)
        errors.append(exc.value)
        with pytest.raises(WorkCompletionConflict, match="session binding conflicts") as exc:
            await app.create_task(
                TaskCreate(
                    task_id="diagnostic-contract-attachment-conflict-task",
                    type="verified-work",
                    session_id=admitted_session_id,
                    input={"private": secret},
                    metadata={"private": secret},
                    work_contract=published.reference(),
                )
            )
        errors.append(exc.value)
        assert await task_store.load_task("diagnostic-contract-attachment-conflict-task") is None
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


def test_verified_work_mutations_require_concrete_cancellation_quiescence() -> None:
    class LookupOnlyStore(InMemoryTaskStore):
        async def load_task(self, task_id: str) -> Task | None:
            return await super().load_task(task_id)

    class OpaqueMutationStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.mutation_dispatched = False

        async def create_task(self, request: TaskCreate) -> Task:
            def begin_opaque_mutation() -> None:
                self.mutation_dispatched = True

            await asyncio.to_thread(begin_opaque_mutation)
            return await super().create_task(request)

    class RevokedProofStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = False

    async def scenario() -> None:
        inherited_store = LookupOnlyStore()
        inherited_contract = _contract(contract_id="inherited-mutation-settlement-contract")
        await inherited_store.publish_work_contract(inherited_contract)
        inherited_app = CayuApp(task_store=inherited_store, enable_logging=False)
        inherited_task = await inherited_app.create_task(
            TaskCreate(
                task_id="inherited-mutation-settlement-task",
                type="verified-work",
                work_contract=inherited_contract.reference(),
            )
        )
        assert inherited_task.work_contract == inherited_contract.reference()

        task_store = OpaqueMutationStore()
        contract = _contract(contract_id="unproven-mutation-settlement-contract")
        await InMemoryTaskStore.publish_work_contract(task_store, contract)
        app = CayuApp(task_store=task_store, enable_logging=False)

        with pytest.raises(ValueError, match="caller-stable task_id"):
            await app.create_task(
                TaskCreate(
                    type="verified-work",
                    work_contract=contract.reference(),
                )
            )
        with pytest.raises(NotImplementedError, match="cancellation-quiescent"):
            await app.create_task(
                TaskCreate(
                    task_id="unproven-mutation-settlement-task",
                    type="verified-work",
                    work_contract=contract.reference(),
                )
            )

        assert task_store.mutation_dispatched is False
        assert await task_store.load_task("unproven-mutation-settlement-task") is None

        revoked_store = RevokedProofStore()
        revoked_contract = _contract(contract_id="revoked-mutation-settlement-contract")
        await InMemoryTaskStore.publish_work_contract(revoked_store, revoked_contract)
        revoked_app = CayuApp(task_store=revoked_store, enable_logging=False)
        with pytest.raises(NotImplementedError, match="cancellation-quiescent"):
            await revoked_app.create_task(
                TaskCreate(
                    task_id="revoked-mutation-settlement-task",
                    type="verified-work",
                    work_contract=revoked_contract.reference(),
                )
            )
        assert await revoked_store.load_task("revoked-mutation-settlement-task") is None

        shadowed_store = InMemoryTaskStore()
        shadowed_contract = _contract(contract_id="shadowed-mutation-settlement-contract")
        await shadowed_store.publish_work_contract(shadowed_contract)
        shadowed_dispatched = False

        async def shadowed_create_task(
            self: InMemoryTaskStore,
            request: TaskCreate,
        ) -> Task:
            nonlocal shadowed_dispatched
            shadowed_dispatched = True
            return await InMemoryTaskStore.create_task(self, request)

        object.__getattribute__(shadowed_store, "__dict__")["create_task"] = MethodType(
            shadowed_create_task,
            shadowed_store,
        )
        shadowed_app = CayuApp(task_store=shadowed_store, enable_logging=False)
        with pytest.raises(NotImplementedError, match="cancellation-quiescent"):
            await shadowed_app.create_task(
                TaskCreate(
                    task_id="shadowed-mutation-settlement-task",
                    type="verified-work",
                    work_contract=shadowed_contract.reference(),
                )
            )
        assert shadowed_dispatched is False
        assert await shadowed_store.load_task("shadowed-mutation-settlement-task") is None

    asyncio.run(scenario())


def test_declared_quiescent_task_creation_settles_before_cancellation_returns() -> None:
    class QuiescentCreationStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.creation_started = asyncio.Event()

        async def create_task(self, request: TaskCreate) -> Task:
            self.creation_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                return await super().create_task(request)

    async def scenario() -> None:
        task_store = QuiescentCreationStore()
        contract = _contract(contract_id="quiescent-mutation-settlement-contract")
        await task_store.publish_work_contract(contract)
        app = CayuApp(task_store=task_store, enable_logging=False)
        task_id = "quiescent-mutation-settlement-task"
        creation = asyncio.create_task(
            app.create_task(
                TaskCreate(
                    task_id=task_id,
                    type="verified-work",
                    work_contract=contract.reference(),
                )
            )
        )
        await task_store.creation_started.wait()
        assert creation.cancel("cancel contracted creation")

        with pytest.raises(asyncio.CancelledError):
            await creation

        assert creation.cancelling() == 1
        assert creation.cancelled()
        persisted = await task_store.load_task(task_id)
        assert persisted is not None
        assert persisted.work_contract == contract.reference()

    asyncio.run(scenario())


def test_task_store_extension_cannot_erase_or_forge_caller_cancellation() -> None:
    class CancellationManipulatingStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.erase_started = asyncio.Event()
            self.extension_tasks: dict[str, asyncio.Task[object]] = {}

        async def publish_work_contract(self, contract: WorkContract) -> WorkContract:
            extension_task = asyncio.current_task()
            assert extension_task is not None
            self.extension_tasks[contract.contract_id] = extension_task
            if contract.contract_id == "extension-uncancel-contract":
                self.erase_started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    extension_task.uncancel()
            elif contract.contract_id == "extension-self-cancel-contract":
                extension_task.cancel("extension-generated cancellation")
                try:
                    await asyncio.sleep(0)
                except asyncio.CancelledError as error:
                    assert error.args == ("extension-generated cancellation",)
            return await super().publish_work_contract(contract)

    def draft(contract: WorkContract) -> WorkContractDraft:
        return WorkContractDraft.model_validate(
            contract.model_dump(mode="python", warnings=False, exclude={"fingerprint"})
        )

    async def scenario() -> None:
        task_store = CancellationManipulatingStore()
        app = CayuApp(task_store=task_store, enable_logging=False)

        erase_contract = _contract(contract_id="extension-uncancel-contract")
        caller = asyncio.create_task(app.create_work_contract(draft(erase_contract)))
        await task_store.erase_started.wait()
        assert task_store.extension_tasks[erase_contract.contract_id] is not caller
        assert caller.cancel("authentic caller cancellation")
        assert caller.cancel("authentic caller cancellation")
        with pytest.raises(asyncio.CancelledError) as exc:
            await caller
        assert exc.value.args == ("authentic caller cancellation",)
        assert workspace_observation_pending_cancellation_requests(exc.value) == 2
        assert caller.cancelling() == 2
        assert caller.cancelled()
        assert await task_store.load_work_contract(erase_contract.reference()) == erase_contract

        forged_contract = _contract(contract_id="extension-self-cancel-contract")
        forged_caller = asyncio.create_task(app.create_work_contract(draft(forged_contract)))
        assert await forged_caller == forged_contract
        assert task_store.extension_tasks[forged_contract.contract_id] is not forged_caller
        assert forged_caller.cancelling() == 0
        assert not forged_caller.cancelled()
        assert await task_store.load_work_contract(forged_contract.reference()) == forged_contract

    asyncio.run(scenario())


def test_grouped_store_cleanup_preserves_forwarded_caller_cancellation_once(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "grouped-store-cancellation-secret-canary"

    class GroupedCancellationStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.publication_started = asyncio.Event()

        async def publish_work_contract(self, contract: WorkContract) -> WorkContract:
            self.publication_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError as error:
                if contract.contract_id == "grouped-duplicated-cancellation-contract":
                    raise BaseExceptionGroup(
                        f"Duplicated store cancellation containing {secret}",
                        [
                            error,
                            error,
                            ConnectionError(f"Store cleanup failed containing {secret}"),
                        ],
                    ) from None
                if contract.contract_id == "grouped-shared-cancellation-contract":
                    shared = BaseExceptionGroup(
                        f"Shared store cancellation containing {secret}",
                        [
                            error,
                            ConnectionError(f"Store cleanup failed containing {secret}"),
                        ],
                    )
                    raise BaseExceptionGroup(
                        f"Repeated store group containing {secret}",
                        [shared, shared],
                    ) from None
                cancellation = (
                    error
                    if contract.contract_id == "grouped-forwarded-cancellation-contract"
                    else asyncio.CancelledError(
                        f"Extension-generated cancellation containing {secret}"
                    )
                )
                raise BaseExceptionGroup(
                    f"Grouped store cleanup containing {secret}",
                    [
                        cancellation,
                        ConnectionError(f"Store cleanup failed containing {secret}"),
                    ],
                ) from None

    def draft(contract: WorkContract) -> WorkContractDraft:
        return WorkContractDraft.model_validate(
            contract.model_dump(mode="python", warnings=False, exclude={"fingerprint"})
        )

    async def run_case(
        contract_id: str,
    ) -> tuple[BaseExceptionGroup, asyncio.Task[WorkContract]]:
        task_store = GroupedCancellationStore()
        app = CayuApp(
            task_store=task_store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        contract = _contract(contract_id=contract_id)
        caller = asyncio.create_task(app.create_work_contract(draft(contract)))
        await task_store.publication_started.wait()
        assert caller.cancel(f"Caller cancellation containing {secret}")
        with pytest.raises(BaseExceptionGroup) as exc:
            await caller
        assert caller.cancelling() == 1
        assert not caller.cancelled()
        assert workspace_observation_pending_cancellation_requests(exc.value) == 1
        assert await task_store.load_work_contract(contract.reference()) is None
        return exc.value, caller

    async def scenario() -> list[BaseExceptionGroup]:
        forwarded, _ = await run_case("grouped-forwarded-cancellation-contract")
        assert len(forwarded.exceptions) == 2
        assert isinstance(forwarded.exceptions[0], asyncio.CancelledError)
        assert isinstance(forwarded.exceptions[1], ConnectionError)

        extension_owned, _ = await run_case("grouped-extension-cancellation-contract")
        assert len(extension_owned.exceptions) == 2
        detached_child = extension_owned.exceptions[0]
        assert isinstance(detached_child, BaseExceptionGroup)
        assert len(detached_child.exceptions) == 2
        assert isinstance(detached_child.exceptions[0], RuntimeError)
        assert "without distinct caller cancellation" in str(detached_child.exceptions[0])
        assert isinstance(detached_child.exceptions[1], ConnectionError)
        assert isinstance(extension_owned.exceptions[1], asyncio.CancelledError)

        def leaf_occurrences(error: BaseException) -> list[BaseException]:
            pending = [error]
            leaves: list[BaseException] = []
            while pending:
                candidate = pending.pop()
                if isinstance(candidate, BaseExceptionGroup):
                    pending.extend(reversed(candidate.exceptions))
                else:
                    leaves.append(candidate)
            return leaves

        duplicated, _ = await run_case("grouped-duplicated-cancellation-contract")
        duplicated_leaves = leaf_occurrences(duplicated)
        assert sum(isinstance(item, asyncio.CancelledError) for item in duplicated_leaves) == 1
        assert sum(type(item) is ConnectionError for item in duplicated_leaves) == 1
        assert sum("repeated failure evidence" in str(item) for item in duplicated_leaves) == 1

        shared, _ = await run_case("grouped-shared-cancellation-contract")
        shared_leaves = leaf_occurrences(shared)
        assert sum(isinstance(item, asyncio.CancelledError) for item in shared_leaves) == 1
        assert sum(type(item) is ConnectionError for item in shared_leaves) == 1
        assert sum("repeated failure evidence" in str(item) for item in shared_leaves) == 1
        return [forwarded, extension_owned, duplicated, shared]

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


def test_task_store_mutation_proof_is_revalidated_after_cancellation_checkpoint() -> None:
    class ReplaceableStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.replacement_called = False

        async def publish_work_contract(self, contract: WorkContract) -> WorkContract:
            return await super().publish_work_contract(contract)

    def draft(contract: WorkContract) -> WorkContractDraft:
        return WorkContractDraft.model_validate(
            contract.model_dump(mode="python", warnings=False, exclude={"fingerprint"})
        )

    async def scenario() -> None:
        task_store = ReplaceableStore()
        app = CayuApp(task_store=task_store, enable_logging=False)
        contract = _contract(contract_id="replaced-after-proof-contract")

        async def replacement(
            self: ReplaceableStore,
            replacement_contract: WorkContract,
        ) -> WorkContract:
            self.replacement_called = True
            return await InMemoryTaskStore.publish_work_contract(self, replacement_contract)

        loop = asyncio.get_running_loop()
        loop.call_soon(
            object.__getattribute__(task_store, "__dict__").__setitem__,
            "publish_work_contract",
            MethodType(replacement, task_store),
        )
        with pytest.raises(NotImplementedError, match="cancellation-quiescent"):
            await app.create_work_contract(draft(contract))

        assert task_store.replacement_called is False
        assert await InMemoryTaskStore.load_work_contract(task_store, contract.reference()) is None

    asyncio.run(scenario())


def test_unproven_worker_parking_is_rejected_before_claim_mutation() -> None:
    class OpaqueParkingStore(InMemoryTaskStore):
        async def hold_claimed_work_contract_task(
            self,
            task_id: str,
            *,
            worker_id: str,
            contract: WorkContractRef,
        ) -> Task:
            await asyncio.to_thread(lambda: None)
            return await super().hold_claimed_work_contract_task(
                task_id,
                worker_id=worker_id,
                contract=contract,
            )

    class AdoptedOpaqueParkingStore(OpaqueParkingStore):
        supports_verified_work_contracts = False

    async def scenario() -> None:
        for identity, task_store in (
            ("advertised", OpaqueParkingStore()),
            ("adopted", AdoptedOpaqueParkingStore()),
        ):
            contract = _contract(contract_id=f"unproven-{identity}-parking-contract")
            await task_store.publish_work_contract(contract)
            ordinary = await task_store.create_task(
                TaskCreate(
                    task_id=f"unproven-{identity}-parking-task",
                    type="verified-work",
                    work_contract=contract.reference(),
                )
            )
            app = CayuApp(task_store=task_store, enable_logging=False)

            with pytest.raises(NotImplementedError, match="parking implementation"):
                await run_task_worker(
                    app,
                    task_store,
                    lambda _app, _task, _worker_id: asyncio.sleep(0),
                    worker_id=f"unproven-{identity}-parking-worker",
                    poll_interval_s=0.001,
                    max_tasks=1,
                )

            unchanged = await task_store.load_task(ordinary.id)
            assert unchanged is not None
            assert unchanged.status is TaskStatus.PENDING
            assert unchanged.worker_id is None

    asyncio.run(scenario())


def test_unproven_worker_claim_mutations_are_rejected_before_dispatch() -> None:
    class OpaqueClaimStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.claim_called = False

        async def claim_task(
            self,
            worker_id: str,
            query: TaskQuery | None = None,
            *,
            lease_seconds: int = 300,
        ) -> Task | None:
            self.claim_called = True
            await asyncio.to_thread(lambda: None)
            return await super().claim_task(
                worker_id,
                query,
                lease_seconds=lease_seconds,
            )

    class OpaqueReclaimStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.reclaim_called = False

        async def reclaim_expired(
            self,
            *,
            query: TaskQuery | None = None,
            max_reclaims: int = 100,
        ) -> list[Task]:
            self.reclaim_called = True
            await asyncio.to_thread(lambda: None)
            return await super().reclaim_expired(
                query=query,
                max_reclaims=max_reclaims,
            )

    async def scenario() -> None:
        claim_store = OpaqueClaimStore()
        reclaim_store = OpaqueReclaimStore()
        for task_store, message, identity in (
            (claim_store, "task claiming", "claim"),
            (reclaim_store, "expired-claim reclamation", "reclaim"),
        ):
            ordinary = await task_store.create_task(
                TaskCreate(task_id=f"unproven-{identity}-task", type="ordinary")
            )
            app = CayuApp(task_store=task_store, enable_logging=False)

            assert (
                await run_task_worker(
                    app,
                    task_store,
                    lambda _app, _task, _worker_id: asyncio.sleep(0),
                    worker_id=f"no-op-{identity}-worker",
                    poll_interval_s=0.001,
                    max_tasks=0,
                )
                == 0
            )

            with pytest.raises(NotImplementedError, match=message):
                await run_task_worker(
                    app,
                    task_store,
                    lambda _app, _task, _worker_id: asyncio.sleep(0),
                    worker_id=f"unproven-{identity}-worker",
                    poll_interval_s=0.001,
                    max_tasks=1,
                )

            unchanged = await task_store.load_task(ordinary.id)
            assert unchanged is not None
            assert unchanged.status is TaskStatus.PENDING
            assert unchanged.worker_id is None
        assert claim_store.claim_called is False
        assert reclaim_store.reclaim_called is False

    asyncio.run(scenario())


def test_declared_quiescent_task_claim_settles_before_cancellation_returns() -> None:
    class QuiescentClaimStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.claim_started = asyncio.Event()

        async def claim_task(
            self,
            worker_id: str,
            query: TaskQuery | None = None,
            *,
            lease_seconds: int = 300,
        ) -> Task | None:
            self.claim_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                return await super().claim_task(
                    worker_id,
                    query,
                    lease_seconds=lease_seconds,
                )

    async def scenario() -> None:
        task_store = QuiescentClaimStore()
        contract = _contract(contract_id="quiescent-claim-settlement-contract")
        await task_store.publish_work_contract(contract)
        task = await task_store.create_task(
            TaskCreate(
                task_id="quiescent-claim-settlement-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        app = CayuApp(task_store=task_store, enable_logging=False)
        worker = asyncio.create_task(
            run_task_worker(
                app,
                task_store,
                lambda _app, _task, _worker_id: asyncio.sleep(0),
                worker_id="quiescent-claim-settlement-worker",
                poll_interval_s=0.001,
                reclaim=False,
                max_tasks=1,
            )
        )
        await task_store.claim_started.wait()
        assert worker.cancel("cancel task claiming")

        with pytest.raises(asyncio.CancelledError):
            await worker

        assert worker.cancelling() == 1
        assert worker.cancelled()
        persisted = await task_store.load_task(task.id)
        assert persisted is not None
        assert persisted.status is TaskStatus.CLAIMED
        assert persisted.worker_id == "quiescent-claim-settlement-worker"

    asyncio.run(scenario())


def test_unexpected_work_contract_store_failures_are_detached_and_redacted(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "verified-work-store-operation-secret-canary"

    class OperationalFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True
        publication_started: asyncio.Event | None = None

        async def publish_work_contract(self, contract: WorkContract) -> WorkContract:
            if contract.contract_id == "operational-publication-failure":
                raise ConnectionError(f"Publication failed for {contract.objective}")
            if contract.contract_id == "group-publication-failure":
                raise ExceptionGroup(
                    f"Grouped publication failure containing {secret}",
                    [
                        ConnectionError(f"First grouped failure containing {secret}"),
                        ValueError(f"Second grouped failure containing {secret}"),
                    ],
                )
            if contract.contract_id == "cancelled-publication":
                if self.publication_started is None:
                    raise AssertionError("Cancellation barrier was not configured.")
                self.publication_started.set()
                await asyncio.Future()
            if contract.contract_id == "child-cancelled-publication":
                raise asyncio.CancelledError(f"Store-generated cancellation containing {secret}")
            if contract.contract_id in {
                "swallowed-cancellation-publication",
                "cancelled-failure-publication",
            }:
                if self.publication_started is None:
                    raise AssertionError("Cancellation barrier was not configured.")
                self.publication_started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    if contract.contract_id == "cancelled-failure-publication":
                        raise ConnectionError(
                            f"Publication failed after cancellation containing {secret}"
                        ) from None
                    return await super().publish_work_contract(contract)
            return await super().publish_work_contract(contract)

        async def load_work_contract(
            self,
            reference: WorkContractRef,
        ) -> WorkContract | None:
            loaded = await super().load_work_contract(reference)
            if reference.contract_id == "operational-lookup-failure":
                raise ConnectionError(
                    f"Lookup failed after loading {None if loaded is None else loaded.objective}"
                )
            return loaded

        async def create_task(self, request: TaskCreate) -> Task:
            if request.task_id == "operational-task-creation-failure":
                raise ConnectionError(
                    f"Task creation failed for {request.input} and {request.metadata}"
                )
            return await super().create_task(request)

    def draft(contract: WorkContract) -> WorkContractDraft:
        return WorkContractDraft.model_validate(
            contract.model_dump(
                mode="python",
                warnings=False,
                exclude={"fingerprint"},
            )
        )

    async def scenario() -> list[BaseException]:
        task_store = OperationalFailureStore()
        lookup_contract = _contract(
            contract_id="operational-lookup-failure",
            objective=f"Stored lookup objective containing {secret}.",
        )
        task_contract = _contract(contract_id="operational-task-contract")
        await task_store.publish_work_contract(lookup_contract)
        await task_store.publish_work_contract(task_contract)
        app = CayuApp(
            task_store=task_store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )

        errors: list[BaseException] = []
        publication_contract = _contract(
            contract_id="operational-publication-failure",
            objective=f"Rejected publication objective containing {secret}.",
        )
        with pytest.raises(ConnectionError) as exc:
            await app.create_work_contract(draft(publication_contract))
        errors.append(exc.value)

        with pytest.raises(ConnectionError) as exc:
            await app.load_work_contract(lookup_contract.reference())
        errors.append(exc.value)

        with pytest.raises(ConnectionError) as exc:
            await app.create_task(
                TaskCreate(
                    task_id="operational-task-creation-failure",
                    type="verified-work",
                    input={"private": secret},
                    metadata={"private": secret},
                    work_contract=task_contract.reference(),
                )
            )
        errors.append(exc.value)

        group_contract = _contract(
            contract_id="group-publication-failure",
            objective=f"Grouped publication objective containing {secret}.",
        )
        with pytest.raises(ExceptionGroup) as exc:
            await app.create_work_contract(draft(group_contract))
        errors.append(exc.value)

        task_store.publication_started = asyncio.Event()
        cancellation_contract = _contract(
            contract_id="cancelled-publication",
            objective=f"Cancelled publication objective containing {secret}.",
        )
        publication_task = asyncio.create_task(
            app.create_work_contract(draft(cancellation_contract))
        )
        await task_store.publication_started.wait()
        assert publication_task.cancel(secret)
        with pytest.raises(asyncio.CancelledError) as exc:
            await publication_task
        assert publication_task.cancelling() == 1
        assert publication_task.cancelled()
        errors.append(exc.value)

        child_cancellation_contract = _contract(
            contract_id="child-cancelled-publication",
            objective=f"Child-only cancellation objective containing {secret}.",
        )
        with pytest.raises(RuntimeError, match="without caller cancellation") as exc:
            await app.create_work_contract(draft(child_cancellation_contract))
        errors.append(exc.value)

        task_store.publication_started = asyncio.Event()
        swallowed_cancellation_contract = _contract(
            contract_id="swallowed-cancellation-publication",
            objective=f"Swallowed cancellation objective containing {secret}.",
        )
        swallowed_task = asyncio.create_task(
            app.create_work_contract(draft(swallowed_cancellation_contract))
        )
        await task_store.publication_started.wait()
        assert swallowed_task.cancel(secret)
        with pytest.raises(asyncio.CancelledError) as exc:
            await swallowed_task
        assert swallowed_task.cancelling() == 1
        assert swallowed_task.cancelled()
        errors.append(exc.value)
        assert (
            await task_store.load_work_contract(swallowed_cancellation_contract.reference())
            == swallowed_cancellation_contract
        )

        task_store.publication_started = asyncio.Event()
        cancelled_failure_contract = _contract(
            contract_id="cancelled-failure-publication",
            objective=f"Cancelled failure objective containing {secret}.",
        )
        cancelled_failure_task = asyncio.create_task(
            app.create_work_contract(draft(cancelled_failure_contract))
        )
        await task_store.publication_started.wait()
        assert cancelled_failure_task.cancel(secret)
        with pytest.raises(BaseExceptionGroup) as exc:
            await cancelled_failure_task
        assert cancelled_failure_task.cancelling() == 1
        assert not cancelled_failure_task.cancelled()
        assert len(exc.value.exceptions) == 2
        assert isinstance(exc.value.exceptions[0], ConnectionError)
        assert isinstance(exc.value.exceptions[1], asyncio.CancelledError)
        errors.append(exc.value)

        assert await task_store.load_task("operational-task-creation-failure") is None
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


def test_runtime_admission_and_worker_parking_store_failures_are_detached(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "verified-work-runtime-store-failure-secret-canary"

    class OperationalFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def admit_ordinary_session_execution(self, session_id: str) -> None:
            loaded = await super().load_task("admission-diagnostic-secret-task")
            raise ConnectionError(
                f"Admission failed for {session_id} after loading {loaded} and {secret}"
            )

        async def hold_claimed_work_contract_task(
            self,
            task_id: str,
            *,
            worker_id: str,
            contract: WorkContractRef,
        ) -> Task:
            loaded = await super().load_task(task_id)
            raise ConnectionError(
                "Parking failed after loading "
                f"{loaded} for {worker_id} and {contract.contract_id} with {secret}"
            )

    async def scenario() -> list[BaseException]:
        session_store = InMemorySessionStore()
        task_store = OperationalFailureStore()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        session_id = "session:operational-admission-failure"
        await _create_dispatch_test_session(app, session_id)
        await task_store.create_task(
            TaskCreate(
                task_id="admission-diagnostic-secret-task",
                type="diagnostic",
                input={"private": secret},
            )
        )

        errors: list[BaseException] = []
        with pytest.raises(ConnectionError) as exc:
            await app.dispatch(
                DispatchRequest(
                    session_id=session_id,
                    dispatch_id="operational-admission-failure",
                    messages=[Message.text("user", f"Do not retain {secret}.")],
                )
            )
        errors.append(exc.value)

        contract = _contract(contract_id="operational-worker-parking-contract")
        await task_store.publish_work_contract(contract)
        worker_task = await task_store.create_task(
            TaskCreate(
                task_id="operational-worker-parking-task",
                type="verified-work",
                input={"private": secret},
                work_contract=contract.reference(),
            )
        )
        with pytest.raises(ConnectionError) as exc:
            await run_task_worker(
                app,
                task_store,
                lambda _app, _task, _worker_id: asyncio.sleep(0),
                worker_id="operational-worker",
                query=TaskQuery(type=worker_task.type),
                poll_interval_s=0.001,
                max_tasks=1,
            )
        errors.append(exc.value)
        claimed = await task_store.load_task(worker_task.id)
        assert claimed is not None
        assert claimed.status is TaskStatus.CLAIMED
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


def test_work_contract_lookup_conflict_does_not_retain_stored_contract(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "verified-work-lookup-conflict-secret-canary"

    async def scenario() -> BaseException:
        task_store = InMemoryTaskStore()
        contract = _contract(
            contract_id="diagnostic-lookup-conflict",
            objective=f"Keep the stored objective containing {secret} private.",
        )
        await task_store.publish_work_contract(contract)
        app = CayuApp(
            task_store=task_store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        conflicting_reference = WorkContractRef(
            contract_id=contract.contract_id,
            version=contract.version,
            fingerprint=_digest("conflicting-lookup-fingerprint"),
        )

        with pytest.raises(WorkContractConflict, match="durable identity.*conflicts") as exc:
            await app.load_work_contract(conflicting_reference)
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


def test_ordinary_execution_rejection_does_not_retain_sensitive_task(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "verified-work-task-diagnostic-secret-canary"

    async def scenario() -> list[BaseException]:
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        contract = _contract(
            contract_id="diagnostic-runtime-contract",
            objective=f"Keep {secret} out of rejection diagnostics.",
        )
        await task_store.publish_work_contract(contract)
        direct_task = await task_store.create_task(
            TaskCreate(
                task_id="diagnostic-ordinary-run-task",
                type="verified-work",
                input={"private": secret},
                metadata={"private": secret},
                work_contract=contract.reference(),
            )
        )

        recovery_session_id = "session:diagnostic-ordinary-recovery"
        await session_store.create(
            RunRequest(
                agent_name="unresolved-agent",
                session_id=recovery_session_id,
                messages=[Message.text("user", "Recover only through verified work.")],
            ),
            identity=SessionIdentity(
                provider_name="unresolved-provider",
                model="unresolved-model",
            ),
        )
        await session_store.update_status(recovery_session_id, SessionStatus.RUNNING)
        recovery_task = await task_store.create_running_task(
            TaskCreate(
                task_id="diagnostic-ordinary-recovery-task",
                type="verified-work",
                session_id=recovery_session_id,
                input={"private": secret},
                metadata={"private": secret},
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(recovery_session_id),
        )
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )

        errors: list[BaseException] = []
        stream = app.run(
            RunRequest(
                agent_name="unresolved-agent",
                task_id=direct_task.id,
                messages=[
                    Message.text(
                        "user",
                        f"Do not retain {secret} while rejecting ordinary execution.",
                    )
                ],
            )
        )
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware") as exc:
            await anext(stream)
        errors.append(exc.value)
        resume_stream = app.resume(
            ResumeRequest(
                session_id=recovery_session_id,
                messages=[
                    Message.text(
                        "user",
                        f"Do not retain {secret} while rejecting ordinary resume.",
                    )
                ],
            )
        )
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware") as exc:
            await anext(resume_stream)
        errors.append(exc.value)
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware") as exc:
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=recovery_session_id,
                    metadata={"private": secret},
                )
            )
        errors.append(exc.value)
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware") as exc:
            await app.dispatch(
                DispatchRequest(
                    session_id=recovery_session_id,
                    dispatch_id="diagnostic-contracted-dispatch",
                    messages=[Message.text("user", f"Do not retain {secret} in diagnostics.")],
                )
            )
        errors.append(exc.value)

        assert await task_store.load_task(direct_task.id) == direct_task
        assert await task_store.load_task(recovery_task.id) == recovery_task
        recovered_session = await session_store.load(recovery_session_id)
        assert recovered_session is not None
        assert recovered_session.status is SessionStatus.RUNNING
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


def test_startup_interruption_recovery_preserves_contracted_parent_for_verifier() -> None:
    async def scenario() -> None:
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        contract = _contract(contract_id="startup-interruption-contract")
        await task_store.publish_work_contract(contract)
        session_id = "session:startup-interruption-contract"
        interrupt_payload = {
            "reason": "operator stop",
            "metadata": {"ticket": "verified-work"},
            "requested_by": None,
            "interruption_type": "operator_requested",
        }
        await session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "Preserve verifier authority on restart.")],
            ),
            identity=SessionIdentity(
                provider_name="unresolved-provider",
                model="unresolved-model",
            ),
        )
        await session_store.update_status(session_id, SessionStatus.INTERRUPTING)
        await session_store.checkpoint(
            session_id,
            {
                "pending_session_interrupt": interrupt_payload,
                "pending_interruption_cascade": {
                    "attempt_id": "startup-interruption-attempt",
                    "interrupt_payload": interrupt_payload,
                },
            },
        )
        task = await task_store.create_running_task(
            TaskCreate(
                task_id="startup-interruption-contract-task",
                type="verified-work",
                session_id=session_id,
                work_contract=contract.reference(),
            ),
            session_invocation=unattributed_session_invocation_binding(session_id),
        )
        session_before = await session_store.load(session_id)
        checkpoint_before = await session_store.load_checkpoint(session_id)
        events_before = await session_store.load_events(session_id)

        scheduled = await app.resume_pending_interruption_cascades(
            interrupting_inactive_before=datetime.now(UTC)
        )

        assert scheduled == 0
        assert await session_store.load(session_id) == session_before
        assert await session_store.load_checkpoint(session_id) == checkpoint_before
        assert await session_store.load_events(session_id) == events_before
        assert await task_store.load_task(task.id) == task

    asyncio.run(scenario())


def test_application_rejects_work_contract_substitution_by_custom_store() -> None:
    class SubstitutingWorkContractStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self, substituted: WorkContract) -> None:
            super().__init__()
            self.substituted = substituted

        async def publish_work_contract(self, contract: WorkContract) -> WorkContract:
            del contract
            return self.substituted

        async def load_work_contract(self, reference: WorkContractRef) -> WorkContract | None:
            del reference
            return self.substituted

    async def scenario() -> None:
        expected = _contract()
        substituted = _contract(contract_id="substituted-contract")
        app = CayuApp(
            task_store=SubstitutingWorkContractStore(substituted),
            enable_logging=False,
        )
        draft = WorkContractDraft.model_validate(
            expected.model_dump(
                mode="python",
                warnings=False,
                exclude={"fingerprint"},
            )
        )

        with pytest.raises(WorkContractConflict, match="exact published definition"):
            await app.create_work_contract(draft)
        with pytest.raises(WorkContractConflict, match="exact requested version"):
            await app.load_work_contract(expected.reference())

    asyncio.run(scenario())
