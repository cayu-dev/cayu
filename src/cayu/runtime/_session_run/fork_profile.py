"""The exact execution profile a fork freezes for its first invocation."""

from __future__ import annotations

from cayu._validation import require_clean_nonblank
from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
from cayu.execution_profiles import ExecutionProfileIdentity
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.forks import session_fork_profile_relationship
from cayu.sessions.records import Session
from cayu.sessions.requests import ResumeRequest, _fork_initial_invocation_request_sha256

_FORK_INITIAL_PROFILE_POLICY_ID = "cayu:fork-initial-invocation:v1"


def _fork_initial_profile_actor() -> ResolutionActor:
    return ResolutionActor(
        subject="cayu:fork-initial-invocation",
        source=ResolutionActorSource.SYSTEM,
    )


def _require_exact_fork_initial_invocation_profile(
    session: Session,
    request: ResumeRequest,
    *,
    queued_dispatch_id: str | None,
) -> ExecutionProfileIdentity | None:
    """Authenticate the dispatch-only first invocation frozen with an exact fork."""

    relationship = session_fork_profile_relationship(session)
    if (
        session.run_epoch != 0
        or relationship is None
        or relationship.initial_invocation_request_sha256 is None
    ):
        return None
    if queued_dispatch_id is None:
        raise SessionRunFenced(
            "Session fork first invocation must enter through its exact durable dispatch."
        )
    queued_dispatch_id = require_clean_nonblank(
        queued_dispatch_id,
        "queued_dispatch_id",
    )
    if relationship.initial_dispatch_id != queued_dispatch_id:
        raise SessionRunFenced(
            "Session fork first invocation dispatch identity does not match the exact fork."
        )
    if relationship.initial_invocation_request_sha256 != (
        _fork_initial_invocation_request_sha256(request)
    ):
        raise SessionRunFenced(
            "Session fork first invocation does not match the exact request frozen "
            "before child publication."
        )
    if relationship.initial_invocation_profile is None:  # pragma: no cover - model invariant
        raise RuntimeError("Session fork first invocation lost its execution profile.")
    return relationship.initial_invocation_profile
