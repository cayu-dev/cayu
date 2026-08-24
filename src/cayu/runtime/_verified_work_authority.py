"""Shared immutable-authority validation for verified-work runtime owners."""

from __future__ import annotations

from cayu.runtime.completion_verifier_profiles import (
    CompletionVerifierProfilePreparationRequest,
    CompletionVerifierProfileRecord,
    completion_verifier_profile_preparation_request_sha256,
)
from cayu.runtime.invocation import SessionInvocation, TaskInvocation
from cayu.runtime.work_contracts import (
    CompletionDecision,
    CompletionDecisionCreate,
    CompletionProposal,
    CompletionProposalCreate,
    CompletionVerificationClaim,
    WorkAttempt,
    WorkAttemptCreate,
    WorkCompletionConflict,
    WorkContract,
    completion_decision_request_sha256,
    completion_gap_fingerprint,
    completion_proposal_request_sha256,
    completion_verification_claim_authority_sha256,
    validate_completion_decision_contract,
    work_attempt_request_sha256,
)
from cayu.vaults import SecretRedactor


def invocation_contains_secret_public_identity(
    invocation: SessionInvocation | TaskInvocation,
    redactor: SecretRedactor,
) -> bool:
    """Return whether immutable invocation authority contains a workload secret."""

    public_identities = (
        invocation.origin.subject,
        invocation.origin.tenant,
        invocation.root_invocation_id,
        invocation.root_session_id,
    )
    return any(
        value is not None and redactor.redact_text(value) != value for value in public_identities
    )


def completion_decision_request_from_record(
    decision: CompletionDecision,
) -> CompletionDecisionCreate:
    """Reconstruct the exact request material committed by one decision."""

    try:
        return CompletionDecisionCreate(
            decision_id=decision.decision_id,
            proposal_id=decision.proposal_id,
            claim_id=decision.claim_id,
            worker_id=decision.worker_id,
            verifier=decision.verifier,
            verifier_profile_fingerprint=decision.verifier_profile_fingerprint,
            decision_version=decision.decision_version,
            verdict=decision.verdict,
            criterion_outcomes=decision.criterion_outcomes,
            constraint_outcomes=decision.constraint_outcomes,
            gaps=decision.gaps,
            evidence_references=decision.evidence_references,
        )
    except BaseException:
        del decision
        raise


def require_completion_proposal_integrity(
    *,
    proposal: CompletionProposal,
    attempt: WorkAttempt,
) -> None:
    """Require one proposal and attempt to retain their canonical request authority."""

    attempt_request: WorkAttemptCreate | None = None
    proposal_request: CompletionProposalCreate | None = None
    try:
        attempt_request = WorkAttemptCreate(
            attempt_id=attempt.attempt_id,
            task_id=attempt.task_id,
            session_id=attempt.session_id,
            contract=attempt.contract,
            execution_profile_fingerprint=attempt.execution_profile_fingerprint,
            worker_id=attempt.worker_id,
        )
        proposal_request = CompletionProposalCreate(
            proposal_id=proposal.proposal_id,
            attempt_id=proposal.attempt_id,
            result=proposal.result,
            evidence_references=proposal.evidence_references,
        )
    except BaseException:
        del proposal, attempt, attempt_request, proposal_request
        raise
    if attempt.request_sha256 != work_attempt_request_sha256(attempt_request):
        del proposal, attempt, attempt_request, proposal_request
        raise WorkCompletionConflict(
            "Durable work attempt has conflicting request integrity evidence."
        ) from None
    if proposal.request_sha256 != completion_proposal_request_sha256(proposal_request):
        del proposal, attempt, attempt_request, proposal_request
        raise WorkCompletionConflict(
            "Durable completion proposal has conflicting request integrity evidence."
        ) from None
    if (
        proposal.attempt_id != attempt.attempt_id
        or proposal.task_id != attempt.task_id
        or proposal.contract != attempt.contract
    ):
        del proposal, attempt, attempt_request, proposal_request
        raise WorkCompletionConflict(
            "Durable completion proposal conflicts with its work attempt."
        ) from None


def require_completion_verifier_profile_integrity(
    *,
    profile: CompletionVerifierProfileRecord,
    proposal: CompletionProposal,
    attempt: WorkAttempt,
    contract: WorkContract,
) -> None:
    """Require one verifier profile to bind a canonical proposal authority chain."""

    require_completion_proposal_integrity(proposal=proposal, attempt=attempt)
    preparation = CompletionVerifierProfilePreparationRequest(
        proposal_id=profile.proposal_id,
        task_id=profile.task_id,
        attempt_id=profile.attempt_id,
        attempt_request_sha256=profile.attempt_request_sha256,
        source_execution_profile_fingerprint=profile.source_execution_profile_fingerprint,
        proposal_request_sha256=profile.proposal_request_sha256,
        contract=profile.contract,
        profile=profile.profile,
        expected_prior_proposal_id=profile.expected_prior_proposal_id,
        expected_prior_profile_fingerprint=profile.expected_prior_profile_fingerprint,
        adoption=profile.adoption,
    )
    if (
        profile.proposal_id != proposal.proposal_id
        or profile.task_id != proposal.task_id
        or profile.attempt_id != attempt.attempt_id
        or profile.attempt_request_sha256 != attempt.request_sha256
        or profile.source_execution_profile_fingerprint != attempt.execution_profile_fingerprint
        or profile.proposal_request_sha256 != proposal.request_sha256
        or profile.contract != contract.reference()
        or proposal.contract != contract.reference()
        or profile.profile.verifier != contract.verifier
        or profile.request_sha256
        != completion_verifier_profile_preparation_request_sha256(preparation)
    ):
        raise WorkCompletionConflict(
            "Durable completion-verifier profile conflicts with proposal authority."
        ) from None


def require_completion_decision_integrity(
    *,
    decision: CompletionDecision,
    proposal: CompletionProposal,
    attempt: WorkAttempt,
    contract: WorkContract,
) -> CompletionDecisionCreate:
    """Require one decision to remain bound to its immutable proposal chain."""

    try:
        require_completion_proposal_integrity(proposal=proposal, attempt=attempt)
    except BaseException:
        del decision, proposal, attempt, contract
        raise
    if (
        decision.proposal_id != proposal.proposal_id
        or decision.task_id != proposal.task_id
        or decision.attempt_id != attempt.attempt_id
        or decision.contract != contract.reference()
        or proposal.contract != contract.reference()
        or decision.verifier != contract.verifier
    ):
        del decision, proposal, attempt, contract
        raise WorkCompletionConflict(
            "Durable completion decision conflicts with its proposal authority."
        ) from None
    try:
        durable_request = completion_decision_request_from_record(decision)
    except BaseException:
        del decision, proposal, attempt, contract
        raise
    if decision.request_sha256 != completion_decision_request_sha256(
        durable_request
    ) or decision.gap_fingerprint != completion_gap_fingerprint(durable_request):
        del decision, proposal, attempt, contract, durable_request
        raise WorkCompletionConflict(
            "Durable completion decision has conflicting integrity evidence."
        ) from None
    try:
        validate_completion_decision_contract(contract, durable_request)
    except BaseException:
        del decision, proposal, attempt, contract, durable_request
        raise
    return durable_request


def completion_decision_claim_authority_matches(
    *,
    decision: CompletionDecision,
    claim: CompletionVerificationClaim,
    proposal: CompletionProposal,
    contract: WorkContract,
) -> bool:
    """Return whether a decision binds the complete final durable claim."""

    claim_authority_sha256 = completion_verification_claim_authority_sha256(claim)
    return (
        decision.claim_id == claim.claim_id
        and claim.proposal_id == proposal.proposal_id
        and decision.worker_id == claim.worker_id
        and decision.verifier == claim.verifier
        and decision.verifier_profile_fingerprint == claim.verifier_profile_fingerprint
        and claim.verifier == contract.verifier
        and decision.decided_at >= claim.claimed_at
        and decision.decided_at < claim.lease_expires_at
        and decision.claim_authority_sha256 == claim_authority_sha256
    )


__all__ = [
    "completion_decision_claim_authority_matches",
    "completion_decision_request_from_record",
    "invocation_contains_secret_public_identity",
    "require_completion_decision_integrity",
    "require_completion_proposal_integrity",
    "require_completion_verifier_profile_integrity",
]
