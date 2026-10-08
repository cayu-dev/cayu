"""Bounded lifecycle checks for workflow children whose payloads were not captured."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from cayu.applications import CayuApp
from cayu.evals.trajectory import (
    _SESSION_TRAJECTORY_HARD_MAX_DEPTH,
    _SESSION_TRAJECTORY_HARD_MAX_SESSIONS,
    SessionTrajectoryBounds,
    SessionTrajectoryError,
    SessionTrajectoryErrorCode,
    _CaptureState,
    _strict_child_nodes,
)
from cayu.sessions.base import (
    SessionInspectionIdentity,
    SessionLineageNode,
    TerminalSessionEvidenceErrorCode,
)
from cayu.sessions.records import SessionStatus


@dataclass(frozen=True)
class _ChildLifecycle:
    origin: SessionLineageNode
    status: SessionStatus
    run_epoch: int
    updated_at: datetime
    last_activity_at: datetime


@dataclass
class _WorkflowChildLifecycle:
    root_session_id: str
    completion_sequence: int
    snapshot: tuple[_ChildLifecycle, ...] | None = None


async def _check_workflow_child_lifecycle(
    app: CayuApp,
    lifecycle: _WorkflowChildLifecycle,
    *,
    before_store_read: Callable[[], None],
) -> None:
    """Capture, then revalidate a settled tree without reading event payloads.

    Payload capture bounds may omit a subtree; they cannot waive lifecycle checks.
    Use the existing hard topology ceilings independently of the retention policy.
    Unknown child origins are included conservatively. Below the workflow root we
    inspect all descendants, since omitted terminal events provide no cutoff.
    """

    state = _CaptureState(
        bounds=SessionTrajectoryBounds(
            max_sessions=_SESSION_TRAJECTORY_HARD_MAX_SESSIONS,
            max_depth=_SESSION_TRAJECTORY_HARD_MAX_DEPTH,
        ),
        strict=True,
    )
    pending = deque([(lifecycle.root_session_id, 1)])
    seen = {lifecycle.root_session_id}
    observed: list[_ChildLifecycle] = []
    while pending:
        parent_id, depth = pending.popleft()
        nodes = await _strict_child_nodes(
            app, parent_id, state=state, before_store_read=before_store_read
        )
        for node in sorted(nodes, key=lambda node: node.id):
            if (
                parent_id == lifecycle.root_session_id
                and len(node.origin_events) == 1
                and node.origin_events[0].sequence > lifecycle.completion_sequence
            ):
                continue
            if node.id in seen:
                raise SessionTrajectoryError(
                    SessionTrajectoryErrorCode.CYCLE_DETECTED, session_id=node.id
                )
            if depth >= _SESSION_TRAJECTORY_HARD_MAX_DEPTH:
                raise SessionTrajectoryError(
                    SessionTrajectoryErrorCode.DEPTH_LIMIT_EXCEEDED,
                    session_id=node.id,
                    limit=_SESSION_TRAJECTORY_HARD_MAX_DEPTH,
                    observed=depth + 1,
                )
            seen.add(node.id)
            before_store_read()
            try:
                identity = await app.session_store.inspect_identity(node.id)
                if type(identity) is not SessionInspectionIdentity:
                    raise TypeError("Invalid child lifecycle identity.")
                identity = SessionInspectionIdentity.model_validate(identity.model_dump())
            except Exception:
                raise SessionTrajectoryError(
                    SessionTrajectoryErrorCode.EVIDENCE_READ_FAILED, session_id=node.id
                ) from None
            if (
                identity.id != node.id
                or identity.parent_session_id != parent_id
                or identity.created_at != node.created_at
            ):
                raise SessionTrajectoryError(
                    SessionTrajectoryErrorCode.PARENT_CONTRADICTION, session_id=node.id
                )
            if identity.status not in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
                raise SessionTrajectoryError(
                    SessionTrajectoryErrorCode.TERMINAL_EVIDENCE_REJECTED,
                    session_id=node.id,
                    terminal_code=TerminalSessionEvidenceErrorCode.SESSION_INTERRUPTED
                    if identity.status is SessionStatus.INTERRUPTED
                    else TerminalSessionEvidenceErrorCode.SESSION_NOT_TERMINAL,
                )
            observed.append(
                _ChildLifecycle(
                    origin=node,
                    status=identity.status,
                    run_epoch=identity.run_epoch,
                    updated_at=identity.updated_at,
                    last_activity_at=identity.last_activity_at,
                )
            )
            pending.append((node.id, depth + 1))
    snapshot = tuple(sorted(observed, key=lambda child: child.origin.id))
    if lifecycle.snapshot is None:
        lifecycle.snapshot = snapshot
    elif snapshot != lifecycle.snapshot:
        raise SessionTrajectoryError(
            SessionTrajectoryErrorCode.CLOSURE_CHANGED, session_id=lifecycle.root_session_id
        )
