"""Execution-profile admission checks against the active egress transition."""

from __future__ import annotations

from cayu.egress.authority import EgressAuthorityTransitionState
from cayu.egress.transitions import EgressAuthorityTransitionRecord
from cayu.execution_profiles import ExecutionProfileDecision
from cayu.sessions.records import Session


def _active_transition_matches_profile_decision(
    *,
    transition: EgressAuthorityTransitionRecord | None,
    session: Session,
    decision: ExecutionProfileDecision,
) -> bool:
    """Verify the complete durable activation tuple for profile admission."""

    return bool(
        transition is not None
        and transition.state is EgressAuthorityTransitionState.ACTIVE
        and transition.session_id == session.id
        and transition.environment_name == session.environment_name
        and transition.expected_authority == decision.expected_profile.egress_authority
        and transition.target_authority == decision.candidate_profile.egress_authority
        and transition.receipt is not None
    )
