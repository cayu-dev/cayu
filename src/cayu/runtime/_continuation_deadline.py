"""Pick and store the execution deadline a stored session continues under.

A session keeps the deadline it started with, with two exceptions applied here
before it continues:

- the resume, approval or answer carries ``execution_deadline``: that deadline
  replaces the stored one after a fresh admission check;
- a person settles an approval or answers a question on a top-level session
  whose deadline was only the default for runs started inside a task handler
  (origin ``run_default``): the session gets back the time it had left when it
  paused, so the wait for the person doesn't count.

Explicit and enclosing-scope deadlines are never extended without a request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu.approvals.tools import ToolApprovalRequest
from cayu.approvals.user_input import UserInputResponse
from cayu.deadlines import (
    ExecutionDeadline,
    ExecutionDeadlineRenewal,
    current_execution_deadline,
    renewal_from_metadata,
    resumed_execution_deadline,
    waiting_for_a_person_counts,
)
from cayu.events import Event, EventType
from cayu.sessions._deadline_replacement import RUN_IN_PROGRESS_STATUSES
from cayu.sessions._pending_approval_reader import PENDING_TOOL_APPROVAL_CHECKPOINT_KEY
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.records import Session
from cayu.sessions.requests import ResumeRequest
from cayu.sessions.store import SessionStore

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.collaboration.access import CollaborationAccessContext

ContinuationRequest = ResumeRequest | ToolApprovalRequest | UserInputResponse

HUMAN_WAIT_INTERRUPTIONS = frozenset({"tool_approval_required", "user_input_required"})
_LIFECYCLE_EVENTS = (
    EventType.SESSION_STARTED,
    EventType.SESSION_RESUMED,
    EventType.SESSION_COMPLETED,
    EventType.SESSION_FAILED,
    EventType.SESSION_INTERRUPTED,
)


async def human_wait_pause(store: SessionStore, session: Session) -> Event | None:
    """The session's pause for a person, if that pause is its latest lifecycle event."""

    records = await store.query_events(
        EventQuery(
            session_id=session.id,
            event_types=_LIFECYCLE_EVENTS,
            order_by=EventOrder.SEQUENCE_DESC,
            limit=1,
        )
    )
    if not records:
        return None
    event = records[0].event
    if (
        event.type is not EventType.SESSION_INTERRUPTED
        or event.payload.get("interruption_type") not in HUMAN_WAIT_INTERRUPTIONS
    ):
        return None
    return event


async def renewed_human_wait_deadline(
    store: SessionStore, session: Session
) -> tuple[ExecutionDeadline, ExecutionDeadlineRenewal] | None:
    """The deadline that leaves out a person's wait, or ``None`` when the wait counts.

    The time left is fixed the first time a pause is settled and stored with
    the renewed deadline, so settling the same pause again gives back the
    same time rather than the earlier renewal's.
    """

    stored = session.execution_deadline
    if stored.expires_at is None or waiting_for_a_person_counts(
        stored, top_level=session.parent_session_id is None
    ):
        return None
    pause = await human_wait_pause(store, session)
    if pause is None:
        return None
    recorded = renewal_from_metadata(session.metadata)
    if recorded is not None and recorded.pause_event_id == pause.id:
        left_at_pause = recorded.seconds_left
    else:
        left_at_pause = (stored.expires_at - pause.timestamp).total_seconds()
    if left_at_pause <= 0:
        # It had already run out when the session paused; nothing to give back.
        return None
    renewed = ExecutionDeadline.after(left_at_pause, source=stored.source, scope=stored.scope)
    enclosing = current_execution_deadline()
    assert renewed.expires_at is not None
    if enclosing.expires_at is not None and enclosing.expires_at < renewed.expires_at:
        # Never outlive the scope the decision is settled in.
        renewed = enclosing.model_copy(update={"source": stored.source, "scope": stored.scope})
    return renewed.with_origin("run_default"), ExecutionDeadlineRenewal(
        pause_event_id=pause.id, seconds_left=left_at_pause
    )


async def continuation_deadline_replacement(
    store: SessionStore,
    session: Session,
    requested: ExecutionDeadline | None,
    *,
    human_decision: bool,
) -> tuple[ExecutionDeadline, ExecutionDeadlineRenewal | None] | None:
    """The admitted deadline to store before ``session`` continues; ``None`` keeps its own.

    ``requested`` is the request's ``execution_deadline``. ``human_decision``
    says the continuation settles an approval or answers a question. Returns
    the deadline with the renewal record to store beside it. Nothing is
    replaced while a run is in progress: that run already holds the deadline
    it was admitted with, and the request is a replay or loses the race.
    """

    if session.status in RUN_IN_PROGRESS_STATUSES:
        return None
    renewal: ExecutionDeadlineRenewal | None = None
    if requested is not None:
        replacement = requested.with_origin("explicit")
    elif human_decision:
        renewed = await renewed_human_wait_deadline(store, session)
        if renewed is None:
            return None
        replacement, renewal = renewed
    else:
        return None
    # The same admission a new run gets: refused when already expired or when
    # an enclosing scope is shorter, before anything is stored.
    resumed_execution_deadline(replacement).require_admission("resume")
    stored = session.execution_deadline
    if stored.expires_at == replacement.expires_at and stored.origin == replacement.origin:
        return None
    return replacement, renewal


async def _continues_the_session_as_it_is(
    store: SessionStore, session: Session, request: ContinuationRequest
) -> bool:
    """Whether the request's own fences and pause identity match the session now.

    A resume needs its expected incarnation and run epoch to match and no
    approval or question pending; an approval or answer must name the exact
    pause the session waits on.
    """

    checkpoint = await store.load_checkpoint(session.id) or {}
    if type(request) is ResumeRequest:
        return (
            request.expected_session_instance_id in (None, session.instance_id)
            and request.expected_run_epoch in (None, session.run_epoch)
            and not checkpoint.get(PENDING_TOOL_APPROVAL_CHECKPOINT_KEY)
            and not checkpoint.get("pending_user_input")
        )
    if type(request) is ToolApprovalRequest:
        pending = checkpoint.get(PENDING_TOOL_APPROVAL_CHECKPOINT_KEY)
        return (
            isinstance(pending, dict)
            and pending.get("approval_id") == request.approval_id
            and pending.get("tool_round_id") == request.tool_round_id
            and pending.get("tool_call_id") == request.tool_call_id
        )
    assert type(request) is UserInputResponse
    pending = checkpoint.get("pending_user_input")
    return isinstance(pending, dict) and pending.get("input_id") == request.input_id


async def prepare_continuation_deadline(
    app: CayuApp,
    session: Session,
    request: ContinuationRequest,
    *,
    participant_context: CollaborationAccessContext | None,
) -> Session:
    """Authorize and store the deadline a session continues under; return it as stored.

    ``request`` is the resume, approval or answer about to continue the
    session, with private session and action identities. The replacement is
    stored only when the request matches the session as it is now (its
    fences and the pause it names), so a stale or mistaken request doesn't
    change the deadline, and only with the same authority as continuing it.
    """

    current = await app.session_store.load(session.id)
    if current is None:
        return session
    replacement = await continuation_deadline_replacement(
        app.session_store,
        current,
        request.execution_deadline,
        human_decision=type(request) is not ResumeRequest,
    )
    if replacement is None or not await _continues_the_session_as_it_is(
        app.session_store, current, request
    ):
        return current
    from cayu.resource_access import current_binding

    binding = current_binding()
    if binding is not None and binding != current.invocation.resource_access:
        # Only the subject that owns the session may change its deadline.
        from cayu._resource_access_errors import ResourceAccessDenied

        raise ResourceAccessDenied()
    # Changing the deadline needs the same authority as continuing the session.
    await app._require_participant_execution(current, participant_context)
    deadline, renewal = replacement
    try:
        return await app.session_store.replace_execution_deadline(
            current.id,
            deadline,
            expected_session_instance_id=current.instance_id,
            expected_run_epoch=current.run_epoch,
            renewal=renewal,
        )
    except NotImplementedError:
        if request.execution_deadline is not None:
            raise
        # Automatic renewal is optional for external stores. Keep enforcing
        # their stored deadline when the adapter cannot replace it.
        return current
