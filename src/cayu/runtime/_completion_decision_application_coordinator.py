"""Runtime-owned verified-work decision-application boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import NoReturn, TypeVar, cast

from pydantic import BaseModel

from cayu._exception_groups import (
    exception_cause,
    exception_group_children,
    iter_exception_tree,
    set_exception_cause,
)
from cayu._validation import canonical_durable_json_bytes, revalidate_model_input
from cayu.runtime._diagnostics import (
    MAX_DIAGNOSTIC_EXCEPTION_GROUP_NODES,
    credential_safe_runtime_exception,
    credential_safe_runtime_exception_group,
    runtime_owned_exception_renderings_are_credential_safe,
)
from cayu.runtime._task_store_operation_boundary import (
    TaskStoreOperationOutcome,
    capture_sensitive_validation,
    capture_task_store_operation,
    raise_task_store_operation_failure,
)
from cayu.runtime._verified_work_authority import (
    completion_decision_claim_authority_matches,
    invocation_contains_secret_public_identity,
    require_completion_decision_integrity,
    require_completion_verifier_profile_integrity,
)
from cayu.runtime.completion_verifier_profiles import (
    CompletionVerifierProfilePreparationRequest,
    CompletionVerifierProfileRecord,
    changed_completion_verifier_profile_components,
    completion_verifier_profile_preparation_request_sha256,
    copy_completion_verifier_profile_record,
)
from cayu.runtime.tasks import (
    CompletionDecisionApplicationReceipt,
    Task,
    TaskStatus,
    TaskStore,
    copy_task,
)
from cayu.runtime.work_contracts import (
    CompletionDecision,
    CompletionDecisionApplicationRequest,
    CompletionProposal,
    CompletionRejectionAction,
    CompletionVerdict,
    CompletionVerificationClaim,
    WorkAttempt,
    WorkCompletionConflict,
    WorkContract,
    completion_decision_application_request_sha256,
    copy_completion_decision,
    copy_completion_decision_application_request,
    copy_completion_proposal,
    copy_completion_verification_claim,
    copy_work_attempt,
    copy_work_contract,
)
from cayu.runtime.workspace_observation_recovery import (
    retain_workspace_observation_pending_cancellation_requests,
    workspace_observation_pending_cancellation_requests,
)
from cayu.vaults import SecretRedactor

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _durable_json_equal(left: object, right: object, *, field_name: str) -> bool:
    left_encoded: bytes | None = None
    right_encoded: bytes | None = None
    try:
        left_encoded = canonical_durable_json_bytes(left, f"{field_name} left")
        right_encoded = canonical_durable_json_bytes(right, f"{field_name} right")
        return left_encoded == right_encoded
    finally:
        del left, right, left_encoded, right_encoded


def _task_snapshots_equal(left: Task, right: Task) -> bool:
    left_document: object | None = None
    right_document: object | None = None
    try:
        left_document = left.model_dump(mode="json", warnings=False)
        right_document = right.model_dump(mode="json", warnings=False)
        return _durable_json_equal(
            left_document,
            right_document,
            field_name="completion decision application task snapshot",
        )
    finally:
        del left, right, left_document, right_document


def _copy_exact_model(
    value: object,
    model_type: type[_ModelT],
    *,
    operation_name: str,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[_ModelT]:
    if type(value) is not model_type:
        return TaskStoreOperationOutcome()
    copier: Callable[[object], object]
    if model_type is WorkContract:
        copier = cast("Callable[[object], object]", copy_work_contract)
    elif model_type is WorkAttempt:
        copier = cast("Callable[[object], object]", copy_work_attempt)
    elif model_type is CompletionProposal:
        copier = cast("Callable[[object], object]", copy_completion_proposal)
    elif model_type is CompletionDecision:
        copier = cast("Callable[[object], object]", copy_completion_decision)
    elif model_type is CompletionVerificationClaim:
        copier = cast("Callable[[object], object]", copy_completion_verification_claim)
    elif model_type is CompletionVerifierProfileRecord:
        copier = cast("Callable[[object], object]", copy_completion_verifier_profile_record)
    elif model_type is Task:
        copier = cast("Callable[[object], object]", copy_task)
    else:

        def copier(candidate: object) -> object:
            return revalidate_model_input(candidate, model_type)

    return capture_sensitive_validation(
        lambda: cast("_ModelT", copier(value)),
        operation_name=operation_name,
        redactor=redactor,
    )


def _safe_failure_group(
    message: str,
    first: BaseException,
    second: BaseException,
    *,
    redactor: SecretRedactor,
) -> BaseExceptionGroup:
    ordered = [first] if first is second else [first, second]
    group: BaseExceptionGroup = (
        ExceptionGroup(message, cast("list[Exception]", ordered))
        if all(isinstance(failure, Exception) for failure in ordered)
        else BaseExceptionGroup(message, ordered)
    )
    cancellation_requests = max(
        (workspace_observation_pending_cancellation_requests(failure) for failure in ordered),
        default=0,
    )
    safe = credential_safe_runtime_exception_group(
        group,
        group_message=message,
        leaf_mapper=lambda leaf: leaf,
        invalid_leaf_factory=lambda: RuntimeError(
            "Decision application reported invalid concurrent failure evidence."
        ),
        truncated_leaf_factory=lambda: RuntimeError(
            "Additional decision-application failures were omitted."
        ),
        fallback_leaf_mapper=lambda _leaf: RuntimeError(
            "Decision application failure diagnostic was redacted."
        ),
        redactor=redactor,
    )
    if cancellation_requests:
        retain_workspace_observation_pending_cancellation_requests(
            safe,
            cancellation_requests,
        )
    return safe


def _attach_safe_failure_cause(
    error: BaseException,
    evidence: BaseException,
    *,
    redactor: SecretRedactor,
) -> BaseException | None:
    """Attach optional evidence without publishing an unsafe composed rendering."""

    if set_exception_cause(
        error, evidence
    ) and runtime_owned_exception_renderings_are_credential_safe(
        error,
        redactor=redactor,
    ):
        return evidence
    fallback = credential_safe_runtime_exception(
        RuntimeError,
        "Completion decision application failure evidence was redacted.",
        redactor=redactor,
        fallback_message="Completion decision application failure evidence was withheld.",
    )
    if set_exception_cause(
        error, fallback
    ) and runtime_owned_exception_renderings_are_credential_safe(
        error,
        redactor=redactor,
    ):
        return fallback
    set_exception_cause(error, None)
    return None


def _raise_reconciliation_signal(
    signal: BaseException,
    prior_failure: Exception,
    *,
    redactor: SecretRedactor,
) -> NoReturn:
    """Keep scalar cancellation/process control authoritative after reconciliation."""

    existing = exception_cause(signal)
    evidence = (
        prior_failure
        if existing is None
        else _safe_failure_group(
            "Completion decision application and reconciliation settlement failed.",
            prior_failure,
            existing,
            redactor=redactor,
        )
    )
    cause = _attach_safe_failure_cause(signal, evidence, redactor=redactor)
    del prior_failure, evidence, existing
    if isinstance(signal, asyncio.CancelledError):
        cancellation_requests = workspace_observation_pending_cancellation_requests(signal)
        if cancellation_requests:
            retain_workspace_observation_pending_cancellation_requests(
                signal,
                cancellation_requests,
            )
    raise signal from cause


def _without_cancellation_evidence(
    error: BaseException,
    *,
    remaining_nodes: list[int],
    visited: set[int],
) -> BaseException | None:
    """Retain ordered reconciliation failures without duplicating cancellation."""

    if remaining_nodes[0] < 1:
        return RuntimeError("Additional reconciliation failures were omitted.")
    remaining_nodes[0] -= 1
    if isinstance(error, asyncio.CancelledError):
        return None
    if not isinstance(error, BaseExceptionGroup):
        return error
    if id(error) in visited:
        return RuntimeError("Cyclic reconciliation failure evidence was omitted.")
    visited.add(id(error))
    children = exception_group_children(error)
    if children is None:
        return RuntimeError("Invalid reconciliation failure evidence was omitted.")
    filtered = [
        retained
        for child in children
        if (
            retained := _without_cancellation_evidence(
                child,
                remaining_nodes=remaining_nodes,
                visited=visited,
            )
        )
        is not None
    ]
    if not filtered:
        return None
    return BaseExceptionGroup(
        "Completion decision reconciliation failed.",
        filtered,
    )


def _raise_authenticated_store_cancellation(
    failure_group: BaseExceptionGroup,
    *,
    redactor: SecretRedactor,
    prior_failure: Exception | None = None,
) -> NoReturn:
    """Restore current caller cancellation from authenticated store-boundary evidence."""

    cancellation_requests = workspace_observation_pending_cancellation_requests(failure_group)
    cancellation = next(
        (
            candidate
            for candidate in iter_exception_tree(failure_group)
            if isinstance(candidate, asyncio.CancelledError)
        ),
        None,
    )
    if cancellation_requests < 1 or cancellation is None:
        if prior_failure is not None:
            raise _safe_failure_group(
                "Completion decision application and reconciliation failed.",
                prior_failure,
                failure_group,
                redactor=redactor,
            ) from None
        raise_task_store_operation_failure(failure_group)

    reconciliation_evidence = _without_cancellation_evidence(
        failure_group,
        remaining_nodes=[MAX_DIAGNOSTIC_EXCEPTION_GROUP_NODES],
        visited=set(),
    )
    existing = exception_cause(cancellation)
    evidence: BaseException | None = prior_failure
    if reconciliation_evidence is not None:
        evidence = (
            reconciliation_evidence
            if evidence is None
            else _safe_failure_group(
                "Completion decision application and store settlement failed.",
                evidence,
                reconciliation_evidence,
                redactor=redactor,
            )
        )
    if existing is not None:
        evidence = (
            existing
            if evidence is None
            else _safe_failure_group(
                "Completion decision cancellation retained additional settlement failure evidence.",
                evidence,
                existing,
                redactor=redactor,
            )
        )
    cause = (
        None
        if evidence is None
        else _attach_safe_failure_cause(cancellation, evidence, redactor=redactor)
    )
    retain_workspace_observation_pending_cancellation_requests(
        cancellation,
        cancellation_requests,
    )
    del failure_group, prior_failure, reconciliation_evidence, existing, evidence
    raise cancellation from cause


class CompletionDecisionApplicationCoordinator:
    """Apply one durable verifier decision through exact receipt authority."""

    def __init__(
        self,
        *,
        task_store: TaskStore | None,
        secret_redactor: SecretRedactor,
    ) -> None:
        if task_store is not None and not isinstance(task_store, TaskStore):
            raise TypeError("Completion decision application requires a TaskStore.")
        if not isinstance(secret_redactor, SecretRedactor):
            raise TypeError("Completion decision application requires a SecretRedactor.")
        self._task_store = task_store
        self._secret_redactor = secret_redactor

    async def apply(self, request: CompletionDecisionApplicationRequest) -> Task:
        if type(request) is not CompletionDecisionApplicationRequest:
            del request
            raise TypeError(
                "Decision application requires a CompletionDecisionApplicationRequest."
            ) from None
        validation = capture_sensitive_validation(
            lambda value=request: copy_completion_decision_application_request(value),
            operation_name="Completion decision application request validation",
            redactor=self._secret_redactor,
        )
        del request
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        copied = validation.result
        del validation
        if copied is None:
            raise ValueError("Completion decision application request is invalid.") from None
        request = copied
        del copied
        store: TaskStore | None = None
        existing: CompletionDecisionApplicationReceipt | None = None
        decision: CompletionDecision | None = None
        claim: CompletionVerificationClaim | None = None
        proposal: CompletionProposal | None = None
        attempt: WorkAttempt | None = None
        contract: WorkContract | None = None
        verifier_profile: CompletionVerifierProfileRecord | None = None
        authority_task: Task | None = None
        outcome: TaskStoreOperationOutcome[Task] | None = None
        dispatch_validation: (
            TaskStoreOperationOutcome[CompletionDecisionApplicationRequest] | None
        ) = None
        dispatched_request: CompletionDecisionApplicationRequest | None = None
        returned_validation: TaskStoreOperationOutcome[Task] | None = None
        returned: Task | None = None
        receipt: CompletionDecisionApplicationReceipt | None = None
        receipt_task: Task | None = None
        try:
            self._require_safe_public_identity(request)
            store = self._require_store()
            request_sha256 = completion_decision_application_request_sha256(request)

            existing = await self._load_receipt(
                store,
                task_id=request.task_id,
                idempotency_key=request.idempotency_key,
            )
            if existing is not None:
                self._require_exact_receipt_identity(
                    existing,
                    request=request,
                    request_sha256=request_sha256,
                )
                (
                    decision,
                    claim,
                    proposal,
                    attempt,
                    contract,
                    verifier_profile,
                    authority_task,
                ) = await self._load_authority(store, request)
                self._require_application_binding(
                    request=request,
                    decision=decision,
                    proposal=proposal,
                )
                return self._task_from_exact_receipt(
                    existing,
                    request=request,
                    decision=decision,
                    attempt=attempt,
                    contract=contract,
                    authority_task=authority_task,
                )

            (
                decision,
                claim,
                proposal,
                attempt,
                contract,
                verifier_profile,
                authority_task,
            ) = await self._load_authority(store, request)
            self._require_application_binding(
                request=request,
                decision=decision,
                proposal=proposal,
            )
            dispatch_validation = capture_sensitive_validation(
                lambda: copy_completion_decision_application_request(request),
                operation_name="Completion decision application dispatch request validation",
                redactor=self._secret_redactor,
            )
            if dispatch_validation.failure is not None:
                raise_task_store_operation_failure(dispatch_validation.failure)
            dispatched_request = dispatch_validation.result
            dispatch_validation = None
            if dispatched_request is None:
                raise WorkCompletionConflict(
                    "Completion decision application dispatch request is invalid."
                ) from None
            outcome = await capture_task_store_operation(
                lambda: store.apply_completion_decision(
                    cast("CompletionDecisionApplicationRequest", dispatched_request)
                ),
                operation_name="Completion decision application",
                redactor=self._secret_redactor,
                mutation_store=store,
                mutation_method_name="apply_completion_decision",
            )
            dispatched_request = None
            if outcome.failure is not None:
                failure = outcome.failure
                if isinstance(failure, BaseExceptionGroup) and (
                    workspace_observation_pending_cancellation_requests(failure) > 0
                ):
                    _raise_authenticated_store_cancellation(
                        failure,
                        redactor=self._secret_redactor,
                    )
                if not isinstance(failure, Exception):
                    raise_task_store_operation_failure(failure)
                return await self._reconcile_after_failure(
                    store,
                    request=request,
                    request_sha256=request_sha256,
                    failure=failure,
                    decision=decision,
                    attempt=attempt,
                    contract=contract,
                    authority_task=authority_task,
                )
            returned_validation = _copy_exact_model(
                outcome.result,
                Task,
                operation_name="Completion decision application result validation",
                redactor=self._secret_redactor,
            )
            outcome = None
            if returned_validation.failure is not None:
                raise_task_store_operation_failure(returned_validation.failure)
            returned = returned_validation.result
            returned_validation = None
            if returned is None:
                raise WorkCompletionConflict(
                    "Task store returned an invalid completion decision application result."
                ) from None
            receipt = await self._load_receipt(
                store,
                task_id=request.task_id,
                idempotency_key=request.idempotency_key,
            )
            if receipt is None:
                raise WorkCompletionConflict(
                    "Task store did not publish the decision-application receipt."
                ) from None
            self._require_exact_receipt_identity(
                receipt,
                request=request,
                request_sha256=request_sha256,
            )
            receipt_task = self._task_from_exact_receipt(
                receipt,
                request=request,
                decision=decision,
                attempt=attempt,
                contract=contract,
                authority_task=authority_task,
            )
            if not _task_snapshots_equal(returned, receipt_task):
                raise WorkCompletionConflict(
                    "Task store returned a task that conflicts with its application receipt."
                ) from None
            return receipt_task
        except BaseException:
            del request, store, existing, decision, claim, proposal, attempt, contract
            del verifier_profile
            del authority_task
            del dispatch_validation, dispatched_request, outcome
            del returned_validation, returned, receipt, receipt_task
            raise

    def _require_store(self) -> TaskStore:
        store = self._task_store
        if store is None:
            raise RuntimeError("task_store is required to apply completion decisions.") from None
        if not store.supports_verified_work_contracts:
            raise NotImplementedError(
                f"{type(store).__name__} does not support verified work contracts."
            ) from None
        return store

    def _require_safe_public_identity(
        self,
        request: CompletionDecisionApplicationRequest,
    ) -> None:
        identities = [request.task_id, request.decision_id, request.idempotency_key]
        if request.result_reference is not None:
            identities.extend(
                [
                    request.result_reference.kind,
                    request.result_reference.reference_id,
                    request.result_reference.digest,
                ]
            )
        if any(self._secret_redactor.redact_text(value) != value for value in identities):
            del request, identities
            raise ValueError(
                "Completion decision application contains a workload secret in public identity."
            ) from None

    async def _load_receipt(
        self,
        store: TaskStore,
        *,
        task_id: str,
        idempotency_key: str,
    ) -> CompletionDecisionApplicationReceipt | None:
        outcome = await capture_task_store_operation(
            lambda: store.load_completion_decision_application_receipt(
                task_id,
                idempotency_key,
            ),
            operation_name="Completion decision application receipt lookup",
            redactor=self._secret_redactor,
        )
        if outcome.failure is not None:
            if isinstance(outcome.failure, BaseExceptionGroup) and (
                workspace_observation_pending_cancellation_requests(outcome.failure) > 0
            ):
                _raise_authenticated_store_cancellation(
                    outcome.failure,
                    redactor=self._secret_redactor,
                )
            raise_task_store_operation_failure(outcome.failure)
        if outcome.result is None:
            return None
        validation = _copy_exact_model(
            outcome.result,
            CompletionDecisionApplicationReceipt,
            operation_name="Completion decision application receipt validation",
            redactor=self._secret_redactor,
        )
        del outcome
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        receipt = validation.result
        del validation
        if receipt is None:
            raise WorkCompletionConflict(
                "Task store returned an invalid decision-application receipt."
            ) from None
        return receipt

    async def _load_authority(
        self,
        store: TaskStore,
        request: CompletionDecisionApplicationRequest,
    ) -> tuple[
        CompletionDecision,
        CompletionVerificationClaim,
        CompletionProposal,
        WorkAttempt,
        WorkContract,
        CompletionVerifierProfileRecord,
        Task,
    ]:
        decision: CompletionDecision | None = None
        indexed: CompletionDecision | None = None
        claim: CompletionVerificationClaim | None = None
        proposal: CompletionProposal | None = None
        attempt: WorkAttempt | None = None
        contract: WorkContract | None = None
        verifier_profile: CompletionVerifierProfileRecord | None = None
        prior_verifier_profile: CompletionVerifierProfileRecord | None = None
        prior_proposal: CompletionProposal | None = None
        prior_attempt: WorkAttempt | None = None
        authority_task: Task | None = None
        try:
            decision = await self._load_required(
                lambda: store.load_completion_decision(request.decision_id),
                CompletionDecision,
                operation_name="Completion decision lookup",
            )
            indexed = await self._load_required(
                lambda: store.load_completion_decision_for_proposal(decision.proposal_id),
                CompletionDecision,
                operation_name="Completion proposal decision lookup",
            )
            if indexed != decision:
                raise WorkCompletionConflict(
                    "Task store returned non-convergent completion decision indexes."
                ) from None
            proposal = await self._load_required(
                lambda: store.load_completion_proposal(decision.proposal_id),
                CompletionProposal,
                operation_name="Completion proposal lookup",
            )
            claim = await self._load_required(
                lambda: store.load_completion_verification_claim(decision.proposal_id),
                CompletionVerificationClaim,
                operation_name="Completion verification claim lookup",
                missing_conflict_message=(
                    "Durable completion decision has no verification-claim authority."
                ),
            )
            attempt = await self._load_required(
                lambda: store.load_work_attempt(proposal.attempt_id),
                WorkAttempt,
                operation_name="Work attempt lookup",
            )
            contract = await self._load_required(
                lambda: store.load_work_contract(proposal.contract),
                WorkContract,
                operation_name="Work contract lookup",
            )
            verifier_profile = await self._load_required(
                lambda: store.load_completion_verifier_profile(decision.proposal_id),
                CompletionVerifierProfileRecord,
                operation_name="Completion verifier profile lookup",
                missing_conflict_message=(
                    "Durable completion decision has no verifier-profile authority."
                ),
            )
            prior_verifier_profile = await self._load_optional(
                lambda: store.load_prior_completion_verifier_profile(decision.proposal_id),
                CompletionVerifierProfileRecord,
                operation_name="Prior completion verifier profile lookup",
            )
            if prior_verifier_profile is not None:
                prior_proposal = await self._load_required(
                    lambda: store.load_completion_proposal(prior_verifier_profile.proposal_id),
                    CompletionProposal,
                    operation_name="Prior completion proposal lookup",
                )
                prior_attempt = await self._load_required(
                    lambda: store.load_work_attempt(prior_proposal.attempt_id),
                    WorkAttempt,
                    operation_name="Prior work attempt lookup",
                )
            authority_task = await self._load_required(
                lambda: store.load_task(decision.task_id),
                Task,
                operation_name="Completion decision task lookup",
                missing_conflict_message=("Durable completion decision has no task authority."),
            )
            if contract.reference() != proposal.contract:
                raise WorkCompletionConflict(
                    "Task store returned a work contract other than the proposal authority."
                ) from None
            self._require_safe_loaded_authority(
                decision=decision,
                claim=claim,
                proposal=proposal,
                attempt=attempt,
                contract=contract,
                verifier_profile=verifier_profile,
                authority_task=authority_task,
            )
            self._require_loaded_authority_integrity(
                decision=decision,
                claim=claim,
                proposal=proposal,
                attempt=attempt,
                contract=contract,
                verifier_profile=verifier_profile,
                prior_verifier_profile=prior_verifier_profile,
                prior_proposal=prior_proposal,
                prior_attempt=prior_attempt,
                authority_task=authority_task,
            )
            return (
                decision,
                claim,
                proposal,
                attempt,
                contract,
                verifier_profile,
                authority_task,
            )
        except BaseException:
            del store, request, decision, indexed, claim, proposal, attempt, contract
            del verifier_profile
            del prior_verifier_profile, prior_proposal, prior_attempt
            del authority_task
            raise

    def _require_loaded_authority_integrity(
        self,
        *,
        decision: CompletionDecision,
        claim: CompletionVerificationClaim,
        proposal: CompletionProposal,
        attempt: WorkAttempt,
        contract: WorkContract,
        verifier_profile: CompletionVerifierProfileRecord,
        prior_verifier_profile: CompletionVerifierProfileRecord | None,
        prior_proposal: CompletionProposal | None,
        prior_attempt: WorkAttempt | None,
        authority_task: Task,
    ) -> None:
        validation = capture_sensitive_validation(
            lambda: require_completion_decision_integrity(
                decision=decision,
                proposal=proposal,
                attempt=attempt,
                contract=contract,
            ),
            operation_name="Completion decision authority integrity validation",
            redactor=self._secret_redactor,
        )
        failure = validation.failure
        validated = validation.result
        del validation
        if failure is not None:
            del decision, claim, proposal, attempt, contract, authority_task, validated
            raise_task_store_operation_failure(failure)
        del failure
        if validated is None:
            del decision, claim, proposal, attempt, contract, verifier_profile, authority_task
            raise WorkCompletionConflict(
                "Durable completion-decision authority has invalid integrity evidence."
            ) from None
        claim_validation = capture_sensitive_validation(
            lambda: completion_decision_claim_authority_matches(
                decision=decision,
                claim=claim,
                proposal=proposal,
                contract=contract,
            ),
            operation_name="Completion decision verification-claim authority validation",
            redactor=self._secret_redactor,
        )
        claim_failure = claim_validation.failure
        claim_matches = claim_validation.result
        del claim_validation
        if claim_failure is not None:
            del decision, claim, proposal, attempt, contract, authority_task, validated
            del claim_matches
            raise_task_store_operation_failure(claim_failure)
        del claim_failure
        if claim_matches is not True:
            del decision, claim, proposal, attempt, contract, authority_task, validated
            raise WorkCompletionConflict(
                "Durable completion decision conflicts with its verification-claim authority."
            ) from None
        del claim_matches
        expected_profile_request = CompletionVerifierProfilePreparationRequest(
            proposal_id=verifier_profile.proposal_id,
            task_id=verifier_profile.task_id,
            attempt_id=verifier_profile.attempt_id,
            attempt_request_sha256=verifier_profile.attempt_request_sha256,
            source_execution_profile_fingerprint=(
                verifier_profile.source_execution_profile_fingerprint
            ),
            proposal_request_sha256=verifier_profile.proposal_request_sha256,
            contract=verifier_profile.contract,
            profile=verifier_profile.profile,
            expected_prior_proposal_id=verifier_profile.expected_prior_proposal_id,
            expected_prior_profile_fingerprint=(
                verifier_profile.expected_prior_profile_fingerprint
            ),
            adoption=verifier_profile.adoption,
        )
        if (
            verifier_profile.proposal_id != proposal.proposal_id
            or verifier_profile.task_id != proposal.task_id
            or verifier_profile.attempt_id != attempt.attempt_id
            or verifier_profile.attempt_request_sha256 != attempt.request_sha256
            or verifier_profile.source_execution_profile_fingerprint
            != attempt.execution_profile_fingerprint
            or verifier_profile.proposal_request_sha256 != proposal.request_sha256
            or verifier_profile.contract != contract.reference()
            or verifier_profile.profile.verifier != contract.verifier
            or verifier_profile.request_sha256
            != completion_verifier_profile_preparation_request_sha256(expected_profile_request)
            or claim.verifier_profile_fingerprint != verifier_profile.profile.fingerprint
            or decision.verifier_profile_fingerprint != verifier_profile.profile.fingerprint
        ):
            raise WorkCompletionConflict(
                "Durable completion decision conflicts with verifier-profile authority."
            ) from None
        self._require_prior_profile_integrity(
            verifier_profile=verifier_profile,
            prior_verifier_profile=prior_verifier_profile,
            proposal=proposal,
            attempt=attempt,
            contract=contract,
            prior_proposal=prior_proposal,
            prior_attempt=prior_attempt,
        )
        if (
            authority_task.id != decision.task_id
            or authority_task.work_contract != decision.contract
        ):
            del decision, claim, proposal, attempt, contract, authority_task, validated
            raise WorkCompletionConflict(
                "Durable task authority conflicts with the completion decision."
            ) from None
        del validated

    def _require_prior_profile_integrity(
        self,
        *,
        verifier_profile: CompletionVerifierProfileRecord,
        prior_verifier_profile: CompletionVerifierProfileRecord | None,
        proposal: CompletionProposal,
        attempt: WorkAttempt,
        contract: WorkContract,
        prior_proposal: CompletionProposal | None,
        prior_attempt: WorkAttempt | None,
    ) -> None:
        if prior_verifier_profile is None:
            if (
                attempt.ordinal != 1
                or verifier_profile.expected_prior_proposal_id is not None
                or verifier_profile.expected_prior_profile_fingerprint is not None
                or verifier_profile.adoption is not None
            ):
                raise WorkCompletionConflict(
                    "Durable verifier profile has no valid prior-profile authority."
                ) from None
            return
        if prior_proposal is None or prior_attempt is None:
            raise WorkCompletionConflict(
                "Durable verifier profile has incomplete prior-profile authority."
            ) from None
        require_completion_verifier_profile_integrity(
            profile=prior_verifier_profile,
            proposal=prior_proposal,
            attempt=prior_attempt,
            contract=contract,
        )
        adoption = verifier_profile.adoption
        if (
            prior_verifier_profile.task_id != proposal.task_id
            or prior_attempt.ordinal != attempt.ordinal - 1
            or verifier_profile.expected_prior_proposal_id != prior_verifier_profile.proposal_id
            or verifier_profile.expected_prior_profile_fingerprint
            != prior_verifier_profile.profile.fingerprint
        ):
            raise WorkCompletionConflict(
                "Durable completion decision conflicts with prior verifier-profile authority."
            ) from None
        if prior_verifier_profile.profile == verifier_profile.profile:
            if adoption is not None:
                raise WorkCompletionConflict(
                    "Exact verifier-profile reuse has unexpected adoption authority."
                ) from None
            return
        if adoption is None or adoption.changed_component_ids != (
            changed_completion_verifier_profile_components(
                prior_verifier_profile.profile,
                verifier_profile.profile,
            )
        ):
            raise WorkCompletionConflict(
                "Changed verifier profile has invalid adoption authority."
            ) from None

    async def _load_optional(
        self,
        operation,
        model_type: type[_ModelT],
        *,
        operation_name: str,
    ) -> _ModelT | None:
        outcome = await capture_task_store_operation(
            operation,
            operation_name=operation_name,
            redactor=self._secret_redactor,
        )
        if outcome.failure is not None:
            if isinstance(outcome.failure, BaseExceptionGroup) and (
                workspace_observation_pending_cancellation_requests(outcome.failure) > 0
            ):
                _raise_authenticated_store_cancellation(
                    outcome.failure,
                    redactor=self._secret_redactor,
                )
            raise_task_store_operation_failure(outcome.failure)
        if outcome.result is None:
            return None
        validation = _copy_exact_model(
            outcome.result,
            model_type,
            operation_name=f"{operation_name} result validation",
            redactor=self._secret_redactor,
        )
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        if validation.result is None:
            raise WorkCompletionConflict(
                f"Task store returned an invalid {operation_name.lower()} result."
            ) from None
        return validation.result

    async def _load_required(
        self,
        operation,
        model_type: type[_ModelT],
        *,
        operation_name: str,
        missing_conflict_message: str | None = None,
    ) -> _ModelT:
        outcome = await capture_task_store_operation(
            operation,
            operation_name=operation_name,
            redactor=self._secret_redactor,
        )
        if outcome.failure is not None:
            if isinstance(outcome.failure, BaseExceptionGroup) and (
                workspace_observation_pending_cancellation_requests(outcome.failure) > 0
            ):
                _raise_authenticated_store_cancellation(
                    outcome.failure,
                    redactor=self._secret_redactor,
                )
            raise_task_store_operation_failure(outcome.failure)
        if outcome.result is None:
            if missing_conflict_message is not None:
                raise WorkCompletionConflict(missing_conflict_message) from None
            raise KeyError(f"{operation_name.removesuffix(' lookup')} not found.") from None
        validation = _copy_exact_model(
            outcome.result,
            model_type,
            operation_name=f"{operation_name} result validation",
            redactor=self._secret_redactor,
        )
        del outcome
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        result = validation.result
        del validation
        if result is None:
            raise WorkCompletionConflict(
                f"Task store returned an invalid {operation_name.lower()} result."
            ) from None
        return result

    def _require_safe_loaded_authority(
        self,
        *,
        decision: CompletionDecision,
        claim: CompletionVerificationClaim,
        proposal: CompletionProposal,
        attempt: WorkAttempt,
        contract: WorkContract,
        verifier_profile: CompletionVerifierProfileRecord,
        authority_task: Task,
    ) -> None:
        contract_references = (
            decision.contract,
            proposal.contract,
            attempt.contract,
            contract.reference(),
            *(() if contract.supersedes is None else (contract.supersedes,)),
        )
        verifier_references = (decision.verifier, claim.verifier, contract.verifier)
        evidence_references = (
            *proposal.evidence_references,
            *decision.evidence_references,
            *(
                reference
                for outcome in (*decision.criterion_outcomes, *decision.constraint_outcomes)
                for reference in outcome.evidence_references
            ),
        )
        identities = [
            decision.decision_id,
            decision.proposal_id,
            decision.claim_id,
            decision.worker_id,
            decision.task_id,
            decision.attempt_id,
            decision.claim_authority_sha256,
            decision.request_sha256,
            decision.gap_fingerprint,
            claim.claim_id,
            claim.proposal_id,
            claim.worker_id,
            claim.request_sha256,
            proposal.proposal_id,
            proposal.attempt_id,
            proposal.task_id,
            proposal.request_sha256,
            proposal.result.kind,
            proposal.result.reference_id,
            proposal.result.digest,
            attempt.attempt_id,
            attempt.task_id,
            attempt.session_id,
            attempt.execution_profile_fingerprint,
            attempt.request_sha256,
            contract.contract_id,
            contract.fingerprint,
            authority_task.id,
            verifier_profile.proposal_id,
            verifier_profile.task_id,
            verifier_profile.attempt_id,
            verifier_profile.request_sha256,
            verifier_profile.profile.fingerprint,
            *(component.component_id for component in verifier_profile.profile.components),
            *(component.fingerprint for component in verifier_profile.profile.components),
            *(
                ()
                if verifier_profile.adoption is None
                else (
                    verifier_profile.adoption.policy_identity,
                    verifier_profile.adoption.idempotency_key,
                    verifier_profile.adoption.requested_by.subject,
                    verifier_profile.adoption.requested_by.tenant or "",
                )
            ),
            *(reference.contract_id for reference in contract_references),
            *(reference.fingerprint for reference in contract_references),
            *(reference.verifier_id for reference in verifier_references),
            *(reference.version for reference in verifier_references),
            *(reference.configuration_fingerprint for reference in verifier_references),
            *(criterion.criterion_id for criterion in contract.criteria),
            *(
                requirement_id
                for criterion in contract.criteria
                for requirement_id in criterion.evidence_requirement_ids
            ),
            *(constraint.constraint_id for constraint in contract.constraints),
            *(
                requirement_id
                for constraint in contract.constraints
                for requirement_id in constraint.evidence_requirement_ids
            ),
            *(requirement.requirement_id for requirement in contract.evidence_requirements),
            *(requirement.kind for requirement in contract.evidence_requirements),
            *(outcome.criterion_id for outcome in decision.criterion_outcomes),
            *(outcome.reason_code for outcome in decision.criterion_outcomes),
            *(outcome.constraint_id for outcome in decision.constraint_outcomes),
            *(outcome.reason_code for outcome in decision.constraint_outcomes),
            *(
                subject_id
                for gap in decision.gaps
                for subject_id in (gap.criterion_id, gap.constraint_id)
                if subject_id is not None
            ),
            *(gap.code for gap in decision.gaps),
            *(
                requirement_id
                for gap in decision.gaps
                for requirement_id in gap.evidence_requirement_ids
            ),
            *(reference.kind for reference in evidence_references),
            *(reference.reference_id for reference in evidence_references),
            *(
                value
                for reference in evidence_references
                for value in (
                    reference.requirement_id,
                    reference.version,
                    reference.digest,
                    reference.unavailable_reason,
                )
                if value is not None
            ),
        ]
        identities.extend(
            value
            for value in (
                claim.execution_owner_id,
                authority_task.session_id,
                authority_task.parent_task_id,
                authority_task.assigned_agent_name,
            )
            if value is not None
        )
        if attempt.worker_id is not None:
            identities.append(attempt.worker_id)
        if any(
            self._secret_redactor.redact_text(value) != value for value in identities
        ) or invocation_contains_secret_public_identity(
            authority_task.invocation,
            self._secret_redactor,
        ):
            del decision, claim, proposal, attempt, contract, verifier_profile, authority_task
            del contract_references, verifier_references, evidence_references, identities
            raise ValueError(
                "Durable completion-decision authority contains a workload secret in public "
                "identity."
            ) from None

    @staticmethod
    def _require_application_binding(
        *,
        request: CompletionDecisionApplicationRequest,
        decision: CompletionDecision,
        proposal: CompletionProposal,
    ) -> None:
        if request.task_id != decision.task_id or request.decision_id != decision.decision_id:
            del request, decision, proposal
            raise WorkCompletionConflict(
                "Decision application conflicts with its durable task or decision."
            ) from None
        if decision.verdict is CompletionVerdict.ACCEPTED:
            if request.result is None or request.result_reference is None:
                del request, decision, proposal
                raise ValueError(
                    "Accepted completion decisions require a verified task result."
                ) from None
            if request.result_reference != proposal.result:
                del request, decision, proposal
                raise WorkCompletionConflict(
                    "Decision application result conflicts with the accepted proposal."
                ) from None
        elif request.result is not None or request.result_reference is not None:
            del request, decision, proposal
            raise ValueError("Non-accepted completion decisions cannot carry a result.") from None

    @staticmethod
    def _require_exact_receipt_identity(
        receipt: CompletionDecisionApplicationReceipt,
        *,
        request: CompletionDecisionApplicationRequest,
        request_sha256: str,
    ) -> None:
        if (
            receipt.task_id != request.task_id
            or receipt.decision_id != request.decision_id
            or receipt.idempotency_key != request.idempotency_key
            or receipt.request_sha256 != request_sha256
        ):
            del receipt, request
            raise WorkCompletionConflict(
                "Decision-application identity is already bound to another request."
            ) from None

    def _task_from_exact_receipt(
        self,
        receipt: CompletionDecisionApplicationReceipt,
        *,
        request: CompletionDecisionApplicationRequest,
        decision: CompletionDecision,
        attempt: WorkAttempt,
        contract: WorkContract,
        authority_task: Task,
    ) -> Task:
        task_validation: TaskStoreOperationOutcome[Task] | None = None
        task: Task | None = None
        try:
            task_validation = _copy_exact_model(
                receipt.task,
                Task,
                operation_name="Completion decision application task snapshot validation",
                redactor=self._secret_redactor,
            )
            if task_validation.failure is not None:
                raise_task_store_operation_failure(task_validation.failure)
            task = task_validation.result
            task_validation = None
            if task is None or task.id != request.task_id:
                raise WorkCompletionConflict(
                    "Decision-application receipt contains an invalid task snapshot."
                ) from None
            if receipt.verifier_profile_fingerprint != decision.verifier_profile_fingerprint:
                raise WorkCompletionConflict(
                    "Decision-application receipt conflicts with verifier-profile authority."
                ) from None
            if invocation_contains_secret_public_identity(
                task.invocation,
                self._secret_redactor,
            ):
                raise ValueError(
                    "Decision-application receipt task contains a workload secret in public "
                    "invocation identity."
                ) from None
            self._require_receipt_task_semantics(
                task=task,
                authority_task=authority_task,
                request=request,
                decision=decision,
                attempt=attempt,
                contract=contract,
                applied_at=receipt.applied_at,
            )
            return task
        except BaseException:
            del receipt, request, decision, attempt, contract, authority_task
            del task_validation, task
            raise

    @staticmethod
    def _require_receipt_task_semantics(
        *,
        task: Task,
        authority_task: Task,
        request: CompletionDecisionApplicationRequest,
        decision: CompletionDecision,
        attempt: WorkAttempt,
        contract: WorkContract,
        applied_at: datetime,
    ) -> None:
        if (
            task.id != authority_task.id
            or task.type != authority_task.type
            or task.title != authority_task.title
            or task.description != authority_task.description
            or task.parent_task_id != authority_task.parent_task_id
            or task.assigned_agent_name != authority_task.assigned_agent_name
            or task.available_at != authority_task.available_at
            or not _durable_json_equal(
                task.input,
                authority_task.input,
                field_name="completion decision application task input",
            )
            or not _durable_json_equal(
                task.metadata,
                authority_task.metadata,
                field_name="completion decision application task metadata",
            )
            or task.created_at != authority_task.created_at
            or task.started_at != authority_task.started_at
            or task.invocation != authority_task.invocation
            or task.work_contract != authority_task.work_contract
        ):
            del task, authority_task, request, decision, attempt, contract
            raise WorkCompletionConflict(
                "Decision-application receipt task conflicts with durable immutable task authority."
            ) from None
        if task.work_contract != decision.contract:
            del task, authority_task, request, decision, attempt, contract
            raise WorkCompletionConflict(
                "Decision-application receipt task conflicts with its work contract."
            ) from None
        if task.worker_id is not None or task.lease_expires_at is not None:
            del task, authority_task, request, decision, attempt, contract
            raise WorkCompletionConflict(
                "Decision-application receipt retained stale worker authority."
            ) from None
        if task.session_id != attempt.session_id:
            del task, authority_task, request, decision, attempt, contract
            raise WorkCompletionConflict(
                "Decision-application receipt task conflicts with its work-attempt session."
            ) from None

        if decision.verdict is CompletionVerdict.ACCEPTED:
            if (
                task.status is not TaskStatus.COMPLETED
                or not _durable_json_equal(
                    task.result,
                    request.result,
                    field_name="completion decision application result",
                )
                or task.error is not None
                or task.status_reason is not None
                or task.status_payload is not None
                or task.completed_at != applied_at
                or task.updated_at != applied_at
            ):
                del task, authority_task, request, decision, attempt, contract
                raise WorkCompletionConflict(
                    "Accepted decision receipt contains a conflicting task snapshot."
                ) from None
            return

        if task.result is not None or task.error is not None or task.completed_at is not None:
            del task, authority_task, request, decision, attempt, contract
            raise WorkCompletionConflict(
                "Non-accepted decision receipt contains a completed task snapshot."
            ) from None
        expected_payload = {
            "completion_decision_id": decision.decision_id,
            "gap_fingerprint": decision.gap_fingerprint,
            "verifier_profile_fingerprint": decision.verifier_profile_fingerprint,
            "verdict": decision.verdict.value,
        }
        if decision.verdict is CompletionVerdict.BLOCKED:
            expected = (TaskStatus.BLOCKED, "work_contract_blocked")
        elif decision.verdict is CompletionVerdict.NEEDS_REVIEW:
            expected = (TaskStatus.NEEDS_ATTENTION, "work_contract_needs_review")
        else:
            expected = CompletionDecisionApplicationCoordinator._expected_rejected_receipt(
                task=task,
                attempt=attempt,
                contract=contract,
            )
            if expected is None:
                if task.status_payload is not None or task.status_reason is not None:
                    del task, authority_task, request, decision, attempt, contract
                    raise WorkCompletionConflict(
                        "Rejected decision receipt contains a conflicting continuation snapshot."
                    ) from None
                return
        if (
            (task.status, task.status_reason) != expected
            or task.status_payload != expected_payload
            or task.updated_at != applied_at
        ):
            del task, authority_task, request, decision, attempt, contract
            raise WorkCompletionConflict(
                "Decision-application receipt contains a conflicting held task snapshot."
            ) from None

    @staticmethod
    def _expected_rejected_receipt(
        *,
        task: Task,
        attempt: WorkAttempt,
        contract: WorkContract,
    ) -> tuple[TaskStatus, str] | None:
        policy = contract.continuation_policy
        if attempt.ordinal >= policy.max_attempts:
            return (TaskStatus.NEEDS_ATTENTION, "work_contract_attempt_limit")
        repeated_gap_possible = attempt.ordinal - 1 >= policy.max_repeated_gap_count
        if (
            repeated_gap_possible
            and task.status is TaskStatus.NEEDS_ATTENTION
            and task.status_reason == "work_contract_repeated_gap_limit"
        ):
            return (TaskStatus.NEEDS_ATTENTION, "work_contract_repeated_gap_limit")
        if policy.rejection_action is CompletionRejectionAction.INTERRUPT:
            return (TaskStatus.PAUSED, "work_contract_rejected")
        if task.status is not TaskStatus.RUNNING:
            del task, attempt, contract
            raise WorkCompletionConflict(
                "Rejected decision receipt contains an unsupported task transition."
            ) from None
        return None

    async def _reconcile_after_failure(
        self,
        store: TaskStore,
        *,
        request: CompletionDecisionApplicationRequest,
        request_sha256: str,
        failure: Exception,
        decision: CompletionDecision,
        attempt: WorkAttempt,
        contract: WorkContract,
        authority_task: Task,
    ) -> Task:
        receipt: CompletionDecisionApplicationReceipt | None = None
        reconciliation_failure: BaseException | None = None
        try:
            try:
                receipt = await self._load_receipt(
                    store,
                    task_id=request.task_id,
                    idempotency_key=request.idempotency_key,
                )
            except BaseException as caught_reconciliation_failure:
                reconciliation_failure = caught_reconciliation_failure
            if reconciliation_failure is not None:
                if isinstance(reconciliation_failure, BaseExceptionGroup) and (
                    workspace_observation_pending_cancellation_requests(reconciliation_failure) > 0
                ):
                    _raise_authenticated_store_cancellation(
                        reconciliation_failure,
                        redactor=self._secret_redactor,
                        prior_failure=failure,
                    )
                if not isinstance(reconciliation_failure, Exception | BaseExceptionGroup):
                    _raise_reconciliation_signal(
                        reconciliation_failure,
                        failure,
                        redactor=self._secret_redactor,
                    )
                raise _safe_failure_group(
                    "Completion decision application and reconciliation failed.",
                    failure,
                    reconciliation_failure,
                    redactor=self._secret_redactor,
                ) from None
            if receipt is None:
                raise_task_store_operation_failure(failure)
            try:
                self._require_exact_receipt_identity(
                    receipt,
                    request=request,
                    request_sha256=request_sha256,
                )
                return self._task_from_exact_receipt(
                    receipt,
                    request=request,
                    decision=decision,
                    attempt=attempt,
                    contract=contract,
                    authority_task=authority_task,
                )
            except Exception as receipt_failure:
                cause = _attach_safe_failure_cause(
                    receipt_failure,
                    failure,
                    redactor=self._secret_redactor,
                )
                raise receipt_failure from cause
            except BaseException as receipt_signal:
                _raise_reconciliation_signal(
                    receipt_signal,
                    failure,
                    redactor=self._secret_redactor,
                )
        except BaseException:
            del store, request, request_sha256, failure, decision, attempt, contract, receipt
            del authority_task
            del reconciliation_failure
            raise


__all__ = ["CompletionDecisionApplicationCoordinator"]
