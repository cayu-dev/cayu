"""Compose verified-worker decision phases through their existing owners.

This private phase does not claim queue work, run handlers, or elect verifier
retries. Those remain with the worker. No new terminalization algorithm or
process-local completion authority is introduced here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Protocol, TypeVar

from cayu._validation import canonical_durable_json_bytes
from cayu.core.messages import Message
from cayu.deadlines import ExecutionDeadline, ExecutionDeadlineExceeded
from cayu.runtime._completion_verifier_coordinator import CompletionVerifierOwnedExecution
from cayu.runtime._invocation_lifecycle import retire_released_invocation_context
from cayu.runtime._task_store_operation_boundary import (
    capture_sensitive_result_validation,
    capture_task_store_operation,
    raise_task_store_operation_failure,
)
from cayu.runtime._verified_work_authority import require_completion_proposal_integrity
from cayu.runtime.completion_result_resolvers import CompletionResultResolutionRequest
from cayu.runtime.completion_verifiers import (
    CompletionVerifierExecutionRequest,
    copy_completion_verifier_execution_request,
)
from cayu.runtime.invocation_release import InvocationReleaseEvidence
from cayu.runtime.sessions import ResumeRequest
from cayu.runtime.tasks import (
    CompletionDecisionApplicationReceipt,
    Task,
    TaskStatus,
    TaskStore,
    WorkAttemptLifecycleReceipt,
    copy_task,
)
from cayu.runtime.work_attempt_admission import (
    WORK_ATTEMPT_ADMISSION_LEASE_MAX_SECONDS,
    WorkAttemptAdmission,
    WorkAttemptAdmissionState,
    WorkAttemptExecutionRequest,
    WorkAttemptRecoveryRequired,
    require_work_attempt_admission_result,
)
from cayu.runtime.work_attempt_lifecycle import (
    WorkAttemptLifecycleSettlement,
    work_attempt_admission_authority_sha256,
    work_attempt_lifecycle_settlement_sha256,
)
from cayu.runtime.work_contracts import (
    CompletionDecision,
    CompletionDecisionApplicationRequest,
    CompletionProposal,
    CompletionVerdict,
    WorkCompletionConflict,
    completion_decision_application_request_sha256,
    copy_completion_decision,
    copy_completion_proposal,
    validate_work_completion_linked_id,
)
from cayu.vaults import SecretRedactor

_ResultT = TypeVar("_ResultT")


class VerifiedTaskAdmissionCallback(Protocol):
    async def __call__(
        self, request: ResumeRequest, *, execution: WorkAttemptExecutionRequest
    ) -> WorkAttemptAdmission: ...


@dataclass(frozen=True, slots=True)
class VerifiedTaskDecisionDependencies:
    store: TaskStore
    redactor: SecretRedactor
    verify: Callable[
        [CompletionVerifierExecutionRequest, ExecutionDeadline], Awaitable[CompletionDecision]
    ]
    start_verify: Callable[
        [CompletionVerifierExecutionRequest, ExecutionDeadline], CompletionVerifierOwnedExecution
    ]
    resolve: Callable[[CompletionResultResolutionRequest], Awaitable[Task]]
    apply: Callable[[CompletionDecisionApplicationRequest], Awaitable[Task]]
    release: Callable[[WorkAttemptAdmission], Awaitable[InvocationReleaseEvidence]]
    admit: VerifiedTaskAdmissionCallback


@dataclass(frozen=True, slots=True)
class VerifiedTaskDecisionResult:
    """Receipt-backed phase outcome; no receipt means an explicit continuation."""

    decision: CompletionDecision
    application: CompletionDecisionApplicationReceipt
    settlement: WorkAttemptLifecycleReceipt | None


@dataclass(frozen=True, slots=True)
class _PreparedDecision:
    admission: WorkAttemptAdmission = field(repr=False)
    verification: CompletionVerifierExecutionRequest = field(repr=False)
    proposal: CompletionProposal = field(repr=False)
    prior: WorkAttemptLifecycleReceipt | None = field(repr=False)
    settlement_request: WorkAttemptLifecycleSettlement = field(repr=False)


@dataclass(frozen=True, slots=True)
class _ContinuationDeadlineExpired:
    """Typed denial from admission, not inferred from detached error text."""


@dataclass(frozen=True, slots=True)
class VerifiedTaskDecisionExecution:
    """Private started phase with independently reachable verifier settlement."""

    verification: CompletionVerifierOwnedExecution = field(repr=False)
    _prepared: _PreparedDecision = field(repr=False)
    _coordinator: VerifiedTaskDecisionCoordinator = field(repr=False)

    async def result(self) -> VerifiedTaskDecisionResult:
        outcome = await self.verification.operation
        if outcome.error is not None:
            raise_task_store_operation_failure(outcome.error)
        if outcome.result is None:
            raise WorkCompletionConflict("Verified worker decision returned no authority.")
        return await self._coordinator._complete(self._prepared, outcome.result)


def _copy_settlement(value: object) -> WorkAttemptLifecycleReceipt | None:
    if value is None:
        return None
    if type(value) is not WorkAttemptLifecycleReceipt:
        raise TypeError("Verified worker settlement lookup returned an invalid receipt.")
    return WorkAttemptLifecycleReceipt.model_validate(
        value.model_dump(mode="python", warnings=False)
    )


def verified_task_operation_id(kind: str, *identities: str) -> str:
    digest = sha256(
        canonical_durable_json_bytes(
            {
                "domain": "cayu.verified-task-worker.v1",
                "kind": kind,
                "identities": list(identities),
            },
            "verified_task_operation",
        )
    ).hexdigest()
    return f"verified-worker:{kind}:{digest}"


class VerifiedTaskDecisionCoordinator:
    def __init__(self, dependencies: VerifiedTaskDecisionDependencies) -> None:
        self._dependencies = dependencies

    async def _read(
        self,
        operation: Callable[[], Awaitable[object]],
        validate: Callable[[object], _ResultT],
        name: str,
    ) -> _ResultT:
        outcome = await capture_task_store_operation(
            operation, operation_name=name, redactor=self._dependencies.redactor
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        validation = capture_sensitive_result_validation(
            lambda value=outcome.result: (validate(value),),
            operation_name=f"{name} validation",
            redactor=self._dependencies.redactor,
        )
        del outcome
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        if validation.result is None:
            raise RuntimeError("Verified worker read returned no validated result.")
        return validation.result[0]

    async def continue_attempt(
        self,
        admission_id: str,
        decision_id: str,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> WorkAttemptAdmission:
        """Schedule one exact successor using the existing admission transaction."""
        dependencies = self._dependencies
        store = dependencies.store

        def prepare_input(
            admission_id=admission_id,
            decision_id=decision_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        ):
            identities = tuple(
                validate_work_completion_linked_id(value, name)
                for value, name in (
                    (admission_id, "admission_id"),
                    (decision_id, "decision_id"),
                    (worker_id, "worker_id"),
                )
            )
            if (
                type(lease_seconds) is not int
                or not 1 <= lease_seconds <= WORK_ATTEMPT_ADMISSION_LEASE_MAX_SECONDS
            ):
                raise ValueError("Worker continuation lease is outside its supported bounds.")
            document = list(identities)
            if dependencies.redactor.redact_json(document) != document:
                raise ValueError("Worker continuation identity contains a workload secret.")
            return (*identities, lease_seconds)

        validation = capture_sensitive_result_validation(
            prepare_input,
            operation_name="Worker continuation request",
            redactor=dependencies.redactor,
        )
        del prepare_input, admission_id, decision_id, worker_id, lease_seconds
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        selected = validation.result
        if selected is None:
            raise ValueError("Worker continuation request is invalid.")
        admission_id, decision_id, worker_id, lease_seconds = selected

        def require_predecessor(value: object) -> WorkAttemptAdmission:
            previous = require_work_attempt_admission_result(value, operation_name="Continuation")
            if (
                previous.admission_id != admission_id
                or previous.state is not WorkAttemptAdmissionState.RELEASED
                or previous.execution_entry is None
                or previous.execution_stop is not None
            ):
                raise WorkCompletionConflict("Worker continuation predecessor conflicts.")
            return previous

        previous = await self._read(
            lambda: store.load_work_attempt_admission(admission_id),
            require_predecessor,
            "Worker continuation predecessor lookup",
        )

        def require_decision(value: object) -> CompletionDecision:
            if type(value) is not CompletionDecision:
                raise WorkCompletionConflict("Worker continuation has no durable decision.")
            decision = copy_completion_decision(value)
            if (
                decision.decision_id != decision_id
                or decision.task_id != previous.task_id
                or decision.attempt_id != previous.attempt_id
                or decision.contract != previous.contract
                or decision.verdict is not CompletionVerdict.REJECTED
            ):
                raise WorkCompletionConflict("Worker continuation decision conflicts.")
            return decision

        decision = await self._read(
            lambda: store.load_completion_decision(decision_id),
            require_decision,
            "Worker continuation decision lookup",
        )
        application_key = verified_task_operation_id("application", admission_id, decision_id)
        expected_application = CompletionDecisionApplicationRequest(
            task_id=previous.task_id,
            decision_id=decision_id,
            idempotency_key=application_key,
        )
        expected_digest = completion_decision_application_request_sha256(expected_application)

        def require_application(value: object) -> CompletionDecisionApplicationReceipt:
            if type(value) is not CompletionDecisionApplicationReceipt:
                raise WorkCompletionConflict("Worker continuation has no application receipt.")
            receipt = CompletionDecisionApplicationReceipt.model_validate(
                value.model_dump(mode="python", warnings=False)
            )
            if (
                receipt.task_id != previous.task_id
                or receipt.decision_id != decision_id
                or receipt.idempotency_key != application_key
                or receipt.request_sha256 != expected_digest
                or receipt.verifier_profile_fingerprint != decision.verifier_profile_fingerprint
                or receipt.task.status is not TaskStatus.RUNNING
            ):
                raise WorkCompletionConflict("Worker continuation application conflicts.")
            return receipt

        application = await self._read(
            lambda: store.load_completion_decision_application_receipt(
                previous.task_id, application_key
            ),
            require_application,
            "Worker continuation application lookup",
        )
        identity = (admission_id, decision_id, application.request_sha256)
        successor_id = verified_task_operation_id("admission", *identity)
        execution = WorkAttemptExecutionRequest(
            admission_id=successor_id,
            claim_id=verified_task_operation_id("execution-claim", successor_id, "1"),
            attempt_id=verified_task_operation_id("attempt", *identity),
            interaction_id=verified_task_operation_id("interaction", *identity),
            worker_id=worker_id,
            generation=1,
            lease_seconds=lease_seconds,
            task_id=previous.task_id,
            predecessor_admission_id=admission_id,
        )
        message = canonical_durable_json_bytes(
            {
                "type": "cayu.verified-task-continuation.v1",
                "decision_id": decision_id,
                "gaps": [gap.model_dump(mode="json", warnings=False) for gap in decision.gaps],
            },
            "verified_task_continuation",
        ).decode("utf-8")

        def require_successor(value: object) -> WorkAttemptAdmission:
            successor = require_work_attempt_admission_result(value, operation_name="Continuation")
            context = successor.continuation
            if (
                successor.admission_id != successor_id
                or successor.state is not WorkAttemptAdmissionState.ACTIVE
                or successor.kind != "continuation"
                or successor.task_id != previous.task_id
                or successor.contract != previous.contract
                or successor.session_invocation != previous.session_invocation
                or successor.source_execution_profile_fingerprint
                != previous.source_execution_profile_fingerprint
                or successor.run_semantics != previous.run_semantics
                or successor.attempt_id != execution.attempt_id
                or successor.interaction_id != execution.interaction_id
                or successor.claim.claim_id != execution.claim_id
                or successor.claim.worker_id != worker_id
                or successor.claim.generation != 1
                or context is None
                or context.prior_admission_id != admission_id
                or context.prior_attempt_id != previous.attempt_id
                or context.proposal_id != decision.proposal_id
                or context.decision != decision
                or context.application_idempotency_key != application_key
                or context.gap_fingerprint != decision.gap_fingerprint
            ):
                raise WorkCompletionConflict("Worker continuation returned conflicting authority.")
            return successor

        async def admit_or_expired() -> WorkAttemptAdmission | _ContinuationDeadlineExpired:
            try:
                return await dependencies.admit(
                    ResumeRequest(
                        session_id=previous.session_id, messages=[Message.text("user", message)]
                    ),
                    execution=execution,
                )
            except ExecutionDeadlineExceeded:
                # Admission owns replay-before-expiry and the final deadline
                # check. Preserve its typed denial without carrying raw errors
                # through the credential-safe store-result boundary.
                return _ContinuationDeadlineExpired()

        result = await self._read(
            admit_or_expired,
            lambda value: (
                value if type(value) is _ContinuationDeadlineExpired else require_successor(value)
            ),
            "Worker continuation admission",
        )
        if isinstance(result, _ContinuationDeadlineExpired):
            assert previous.run_semantics is not None
            raise ExecutionDeadlineExceeded(
                previous.run_semantics.deadline, "work_attempt_continuation"
            )
        return result

    async def settle(
        self, admission_id: str, verification: CompletionVerifierExecutionRequest
    ) -> VerifiedTaskDecisionResult:
        preparation = self._prepare(admission_id, verification)
        del admission_id, verification
        prepared = await preparation
        assert prepared.admission.run_semantics is not None
        decision = await self._dependencies.verify(
            prepared.verification, prepared.admission.run_semantics.deadline
        )
        return await self._complete(prepared, decision)

    async def start(
        self, admission_id: str, verification: CompletionVerifierExecutionRequest
    ) -> VerifiedTaskDecisionExecution:
        preparation = self._prepare(admission_id, verification)
        del admission_id, verification
        prepared = await preparation
        assert prepared.admission.run_semantics is not None
        execution = self._dependencies.start_verify(
            prepared.verification, prepared.admission.run_semantics.deadline
        )
        # No suspension between scheduling verification and handing ownership
        # to the caller. All admission/release checks precede scheduling.
        return VerifiedTaskDecisionExecution(execution, prepared, self)

    async def _prepare(
        self, admission_id: str, verification: CompletionVerifierExecutionRequest
    ) -> _PreparedDecision:
        dependencies = self._dependencies
        store = dependencies.store

        def prepare_input(
            admission_id=admission_id, verification=verification
        ) -> tuple[str, CompletionVerifierExecutionRequest]:
            identity = validate_work_completion_linked_id(admission_id, "admission_id")
            request = copy_completion_verifier_execution_request(verification)
            document = {"admission_id": identity, "verification": request.model_dump(mode="json")}
            if dependencies.redactor.redact_json(document) != document:
                raise ValueError("Verified worker decision request contains a workload secret.")
            return identity, request

        validation = capture_sensitive_result_validation(
            prepare_input,
            operation_name="Verified worker decision request",
            redactor=dependencies.redactor,
        )
        del prepare_input, admission_id, verification
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        if validation.result is None:
            raise ValueError("Verified worker decision request is invalid.")
        admission_id, verification = validation.result

        def require_admission(value: object) -> WorkAttemptAdmission:
            admission = require_work_attempt_admission_result(
                value, operation_name="Worker decision"
            )
            if (
                admission.admission_id != admission_id
                or admission.state is not WorkAttemptAdmissionState.RELEASED
                or admission.execution_entry is None
                or admission.execution_stop is not None
                or admission.run_semantics is None
            ):
                raise WorkAttemptRecoveryRequired(
                    "Worker decision requires a released proposed attempt."
                )
            return admission

        admission = await self._read(
            lambda: store.load_work_attempt_admission(admission_id),
            require_admission,
            "Verified worker admission lookup",
        )

        def require_proposal(value: object) -> CompletionProposal:
            if type(value) is not CompletionProposal:
                raise TypeError("Verified worker proposal lookup returned an invalid record.")
            proposal = copy_completion_proposal(value)
            if proposal.proposal_id != verification.proposal_id or admission.attempt is None:
                raise WorkCompletionConflict("Verified worker proposal identity conflicts.")
            require_completion_proposal_integrity(proposal=proposal, attempt=admission.attempt)
            return proposal

        proposal = await self._read(
            lambda: store.load_completion_proposal(verification.proposal_id),
            require_proposal,
            "Verified worker proposal lookup",
        )
        application_key = verified_task_operation_id(
            "application", admission_id, verification.decision_id
        )
        settlement_key = verified_task_operation_id("settlement", admission_id)

        prior = await self._read(
            lambda: store.load_work_attempt_lifecycle_receipt(admission_id),
            _copy_settlement,
            "Verified worker settlement lookup",
        )
        release = (
            prior.request.release_evidence
            if prior is not None
            else await dependencies.release(admission)
        )
        settlement_request = WorkAttemptLifecycleSettlement(
            settlement_id=settlement_key,
            task_id=admission.task_id,
            admission_id=admission_id,
            expected_admission_sha256=work_attempt_admission_authority_sha256(admission),
            release_evidence=release,
            kind="decision_application",
            decision_id=verification.decision_id,
            application_idempotency_key=application_key,
        )
        if prior is not None and prior.request != settlement_request:
            raise WorkCompletionConflict(
                "Verified worker settlement conflicts with its exact request."
            )

        # The exact release above proves this execution ended. An inherited
        # caller epoch (including a losing replacement's copy) must not fence
        # this subsequent verifier/application phase. Newer epochs survive.
        retire_released_invocation_context(release)

        return _PreparedDecision(admission, verification, proposal, prior, settlement_request)

    async def _complete(
        self, prepared: _PreparedDecision, raw_decision: CompletionDecision
    ) -> VerifiedTaskDecisionResult:
        dependencies = self._dependencies
        store = dependencies.store
        admission = prepared.admission
        verification = prepared.verification
        proposal = prepared.proposal
        prior = prepared.prior
        settlement_request = prepared.settlement_request
        application_key = settlement_request.application_idempotency_key
        assert application_key is not None
        decision_validation = capture_sensitive_result_validation(
            lambda value=raw_decision: copy_completion_decision(value),
            operation_name="Verified worker decision result validation",
            redactor=dependencies.redactor,
        )
        del raw_decision
        if decision_validation.failure is not None:
            raise_task_store_operation_failure(decision_validation.failure)
        decision = decision_validation.result
        if decision is None:
            raise WorkCompletionConflict("Verified worker decision returned no authority.")
        if (
            decision.decision_id != verification.decision_id
            or decision.proposal_id != proposal.proposal_id
            or decision.attempt_id != admission.attempt_id
            or decision.task_id != admission.task_id
            or decision.contract != admission.contract
        ):
            raise WorkCompletionConflict("Verified worker decision returned conflicting authority.")
        if decision.verdict is CompletionVerdict.ACCEPTED:
            raw_task = await dependencies.resolve(
                CompletionResultResolutionRequest(
                    task_id=admission.task_id,
                    decision_id=decision.decision_id,
                    idempotency_key=application_key,
                )
            )
        else:
            raw_task = await dependencies.apply(
                CompletionDecisionApplicationRequest(
                    task_id=admission.task_id,
                    decision_id=decision.decision_id,
                    idempotency_key=application_key,
                )
            )
        task_validation = capture_sensitive_result_validation(
            lambda value=raw_task: copy_task(value),
            operation_name="Verified worker application result validation",
            redactor=dependencies.redactor,
        )
        del raw_task
        if task_validation.failure is not None:
            raise_task_store_operation_failure(task_validation.failure)
        task = task_validation.result
        if task is None:
            raise WorkCompletionConflict("Verified worker application returned no task.")
        expected_application = CompletionDecisionApplicationRequest(
            task_id=admission.task_id,
            decision_id=decision.decision_id,
            idempotency_key=application_key,
            result=task.result if decision.verdict is CompletionVerdict.ACCEPTED else None,
            result_reference=proposal.result
            if decision.verdict is CompletionVerdict.ACCEPTED
            else None,
        )

        def require_application(value: object) -> CompletionDecisionApplicationReceipt:
            if type(value) is not CompletionDecisionApplicationReceipt:
                raise TypeError("Verified worker application has no durable receipt.")
            receipt = CompletionDecisionApplicationReceipt.model_validate(
                value.model_dump(mode="python", warnings=False)
            )
            if (
                receipt.task_id != admission.task_id
                or receipt.decision_id != decision.decision_id
                or receipt.idempotency_key != application_key
                or receipt.verifier_profile_fingerprint != decision.verifier_profile_fingerprint
                or receipt.request_sha256
                != completion_decision_application_request_sha256(expected_application)
                or receipt.task != task
            ):
                raise WorkCompletionConflict("Verified worker application receipt conflicts.")
            return receipt

        application = await self._read(
            lambda: store.load_completion_decision_application_receipt(
                admission.task_id, application_key
            ),
            require_application,
            "Verified worker application receipt lookup",
        )
        if decision.verdict is CompletionVerdict.REJECTED and task.status is TaskStatus.RUNNING:
            if prior is not None:
                raise WorkCompletionConflict(
                    "A continuing attempt already has terminal settlement."
                )
            return VerifiedTaskDecisionResult(decision, application, None)
        outcome = await capture_task_store_operation(
            lambda: store.settle_work_attempt_lifecycle(settlement_request),
            operation_name="Verified worker lifecycle settlement",
            redactor=dependencies.redactor,
            mutation_store=store,
            mutation_method_name="settle_work_attempt_lifecycle",
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)

        def require_settlement(value: object) -> WorkAttemptLifecycleReceipt:
            receipt = _copy_settlement(value)
            if (
                receipt is None
                or receipt.request != settlement_request
                or receipt.request_sha256
                != work_attempt_lifecycle_settlement_sha256(settlement_request)
                or receipt.task != application.task
                or receipt.retired_contract_binding
                != (decision.verdict is CompletionVerdict.ACCEPTED)
            ):
                raise WorkCompletionConflict("Verified worker lifecycle receipt conflicts.")
            return receipt

        validation = capture_sensitive_result_validation(
            lambda value=outcome.result: require_settlement(value),
            operation_name="Verified worker lifecycle receipt validation",
            redactor=dependencies.redactor,
        )
        del outcome
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        if validation.result is None:
            raise RuntimeError("Verified worker lifecycle returned no receipt.")
        return VerifiedTaskDecisionResult(decision, application, validation.result)
