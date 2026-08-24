from __future__ import annotations

from cayu import ExecutionProfileBehaviorIdentity
from cayu.runtime.completion_verifier_profiles import (
    CompletionVerifierProfilePreparationRequest,
    CompletionVerifierProfileRecord,
    build_completion_verifier_execution_profile,
    completion_verifier_profile_preparation_request_sha256,
)
from cayu.runtime.tasks import TaskStore
from cayu.runtime.work_contracts import WorkCompletionConflict


async def prepare_test_completion_verifier_profile(
    store: TaskStore,
    proposal_id: str,
    *,
    identity_name: str = "tests:completion-verifier",
) -> CompletionVerifierProfileRecord:
    proposal = await store.load_completion_proposal(proposal_id)
    if proposal is None:
        raise AssertionError("test completion proposal is missing")
    attempt = await store.load_work_attempt(proposal.attempt_id)
    if attempt is None:
        raise AssertionError("test work attempt is missing")
    contract = await store.load_work_contract(proposal.contract)
    if contract is None:
        raise AssertionError("test work contract is missing")
    prior = await store.load_prior_completion_verifier_profile(proposal_id)
    profile = build_completion_verifier_execution_profile(
        verifier=contract.verifier,
        adapter_identity=ExecutionProfileBehaviorIdentity(
            name=identity_name,
            behavior_version="1",
            implementation_version="1",
        ),
    )
    if prior is not None and prior.profile != profile:
        raise WorkCompletionConflict(
            "The test helper cannot implicitly authorize a verifier-profile change."
        )
    return await store.prepare_completion_verifier_profile(
        CompletionVerifierProfilePreparationRequest(
            proposal_id=proposal.proposal_id,
            task_id=proposal.task_id,
            attempt_id=attempt.attempt_id,
            attempt_request_sha256=attempt.request_sha256,
            source_execution_profile_fingerprint=attempt.execution_profile_fingerprint,
            proposal_request_sha256=proposal.request_sha256,
            contract=contract.reference(),
            profile=profile,
            expected_prior_proposal_id=None if prior is None else prior.proposal_id,
            expected_prior_profile_fingerprint=(
                None if prior is None else prior.profile.fingerprint
            ),
        )
    )


def retask_test_completion_verifier_profile(
    profile: CompletionVerifierProfileRecord,
    *,
    task_id: str,
) -> CompletionVerifierProfileRecord:
    """Forge internally coherent profile evidence for an unrelated task boundary."""

    request = CompletionVerifierProfilePreparationRequest(
        proposal_id=profile.proposal_id,
        task_id=task_id,
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
    return CompletionVerifierProfileRecord(
        **request.model_dump(mode="python"),
        request_sha256=completion_verifier_profile_preparation_request_sha256(request),
        prepared_at=profile.prepared_at,
    )


__all__ = [
    "prepare_test_completion_verifier_profile",
    "retask_test_completion_verifier_profile",
]
