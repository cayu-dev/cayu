"""Settle a denial that a changed execution profile won't let continue.

``IncompleteSessionRecovery.deny_stranded_approval`` ends the session instead of
continuing the denied round. These wrap approval resolution: one replays a
denial that already ended its session, the other falls back to it when the
continuation is refused before anything resumed.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from cayu.approvals.tools import (
    ToolApprovalDecision,
    ToolApprovalProfileAdoptionRefused,
    ToolApprovalRequest,
)
from cayu.events import Event, EventType
from cayu.runtime._session_abandonment import stranded_approval_denial
from cayu.runtime.execution_profiles import ExecutionProfileMismatchError
from cayu.sessions.records import Session, SessionStatus

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.runtime._recovery_coordinator import RecoveryCoordinator


def require_stranded_approval_review(
    coordinator: RecoveryCoordinator,
    request: ToolApprovalRequest,
    session: Session,
    checkpoint: dict[str, Any] | None,
    *,
    accepted_request: bool = False,
) -> None:
    """Validate human review before settling a denial outside the approval claim."""
    coordinator._require_human_review_decision(
        request.review_reference,
        session,
        checkpoint,
        denying=request.decision is ToolApprovalDecision.DENY,
        accepted_request=accepted_request,
    )


async def replay_stranded_denial(
    app: CayuApp, session: Session | None, request: ToolApprovalRequest
) -> tuple[Event, ...] | None:
    """Replay a denial that already ended its session, or ``None`` if it didn't."""

    if (
        session is None
        or session.status is not SessionStatus.FAILED
        or request.decision is not ToolApprovalDecision.DENY
        or stranded_approval_denial(
            await app.session_store.load_checkpoint(session.id), request.approval_id
        )
        is None
    ):
        return None
    return await app._incomplete_recovery.deny_stranded_approval(
        request,
        refusal=RuntimeError("Tool approval was already denied with a different request."),
        require_review=app._recovery_coordinator.require_tool_approval_review,
    )


async def deny_when_stranded(
    stream: AsyncGenerator[Event, None],
    app: CayuApp,
    request: ToolApprovalRequest | None,
) -> AsyncGenerator[Event, None]:
    """Pass the resolution through; end the session if its denial can't continue.

    ``request`` is ``None`` for a task-owned continuation, which keeps its task's
    own refusal and failure handling.
    """

    denial = (
        request if request is not None and request.decision is ToolApprovalDecision.DENY else None
    )

    resumed = False
    try:
        async with contextlib.aclosing(stream):
            async for event in stream:
                resumed = resumed or event.type is EventType.SESSION_RESUMED
                yield event
    except (ExecutionProfileMismatchError, ToolApprovalProfileAdoptionRefused) as refusal:
        if denial is None or resumed:
            raise
        for event in await app._incomplete_recovery.deny_stranded_approval(
            denial,
            refusal=refusal,
            require_review=app._recovery_coordinator.require_tool_approval_review,
        ):
            yield event


__all__ = ["deny_when_stranded", "replay_stranded_denial"]
