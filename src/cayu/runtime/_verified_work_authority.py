"""Compatibility imports for contracts owned by ``cayu.tasks._verified_work_authority``."""

from cayu.tasks._verified_work_authority import (
    completion_decision_claim_authority_matches,
    completion_decision_request_from_record,
    invocation_contains_secret_public_identity,
    require_completion_decision_integrity,
    require_completion_proposal_integrity,
    require_completion_verifier_profile_integrity,
)

__all__ = [
    "completion_decision_claim_authority_matches",
    "completion_decision_request_from_record",
    "invocation_contains_secret_public_identity",
    "require_completion_decision_integrity",
    "require_completion_proposal_integrity",
    "require_completion_verifier_profile_integrity",
]
