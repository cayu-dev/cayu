"""Pick and store the execution deadline a stored session continues under.

A session keeps the deadline it started with, with these changes applied here
before it continues:

- the resume, approval or answer carries ``execution_deadline``: that deadline
  replaces the stored one after a fresh admission check;
- a person settles an approval or answers a question on a top-level session
  whose deadline was only the default for runs started inside a task handler
  (origin ``run_default``): the session gets back the time it had left when it
  paused, so the wait for the person doesn't count;
- a task event durably bounds any existing session it continues, including
  approvals and answers, before work runs under that bound.

Explicit and enclosing-scope deadlines are never extended without a request.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cayu.approvals.tools import ToolApprovalRequest
from cayu.approvals.user_input import UserInputResponse
from cayu.deadlines import (
    ExecutionDeadline,
    ExecutionDeadlineRenewal,
    current_execution_deadline,
    current_task_event_deadline,
    effective_deadline,
    renewal_from_metadata,
    resumed_execution_deadline,
    task_event_deadline_from_metadata,
    waiting_for_a_person_counts,
)
from cayu.events import Event, EventType
from cayu.runtime._recovery_admission import RecoveryAdmissionHook
from cayu.runtime._run_limit_stream import DeadlineOutcomeAdmission
from cayu.sessions._deadline_replacement import (
    RUN_IN_PROGRESS_STATUSES,
    continuation_event_metadata,
    metadata_with_execution_deadline,
)
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
    renewed = effective_deadline(
        renewed.with_origin("run_default"), task_event_deadline_from_metadata(session.metadata)
    )
    return renewed, ExecutionDeadlineRenewal(pause_event_id=pause.id, seconds_left=left_at_pause)


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
    checkpoint = await store.load_checkpoint(session.id) or {}
    allow_rebind = not human_decision and not (
        checkpoint.get(PENDING_TOOL_APPROVAL_CHECKPOINT_KEY) or checkpoint.get("pending_user_input")
    )
    metadata = continuation_event_metadata(
        session.metadata, session.status, explicit=requested is not None, allow_rebind=allow_rebind
    )
    event_rebound = metadata is not session.metadata
    if event_rebound:
        session = session.model_copy(update={"metadata": metadata})
    event = current_task_event_deadline()
    stored_event = task_event_deadline_from_metadata(session.metadata)
    event_changed = event_rebound or (
        event.expires_at is not None
        and (stored_event.expires_at is None or event.expires_at < stored_event.expires_at)
    )
    renewal: ExecutionDeadlineRenewal | None = None
    if requested is not None:
        replacement = requested.with_origin("explicit")
    elif human_decision:
        renewed = await renewed_human_wait_deadline(store, session)
        if renewed is None:
            if not event_changed:
                return None
            replacement = session.execution_deadline
        else:
            replacement, renewal = renewed
    elif event_changed:
        replacement = session.execution_deadline
    else:
        return None
    # An event may admit a pre-existing session, but its outer bound must be
    # durable before any resumed work runs. Ordinary transient scopes still
    # cannot silently tighten a stored execution.
    replacement = effective_deadline(replacement, stored_event, event)
    resumed_execution_deadline(replacement).require_admission("resume")
    stored = session.execution_deadline
    if (
        requested is None
        and not event_changed
        and stored.expires_at == replacement.expires_at
        and stored.origin == replacement.origin
    ):
        return None
    # Stores apply the event bound atomically. Preserve an explicit independent
    # deadline too, so the next event cannot discard that session-level limit.
    retained = requested.with_origin("explicit") if requested is not None else replacement
    retained = retained.model_copy()
    retained._event_rebinding_allowed = event_rebound and allow_rebind
    return retained, renewal


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


@dataclass(frozen=True)
class ContinuationDeadlinePlan:
    """The session as loaded for a continuation, and the deadline to store at admission.

    ``deadline`` is ``None`` when nothing is left to store at admission: the
    stored deadline stays, or a store without in-run replacement already
    stored it (see ``plan_continuation_deadline``).
    """

    session: Session
    deadline: ExecutionDeadline | None = None
    renewal: ExecutionDeadlineRenewal | None = None

    def timed_session(self) -> Session:
        """The session as it will be stored, for timing the continuation before admission."""
        if self.deadline is None:
            return self.session
        return self.session.model_copy(
            update={
                "metadata": metadata_with_execution_deadline(
                    self.session.metadata, self.deadline, self.renewal, status=self.session.status
                )
            }
        )

    async def commit(self, store: SessionStore, admission: DeadlineOutcomeAdmission) -> Session:
        """Store the deadline from inside the run its decision was admitted as."""
        if (
            self.deadline is None
            or admission.session_id != self.session.id
            or admission.run_epoch is None
        ):
            raise RuntimeError("A settle's deadline can be stored only by its admitted run.")
        return await store.replace_execution_deadline(
            self.session.id,
            self.deadline,
            expected_session_instance_id=admission.session_instance_id,
            expected_run_epoch=admission.run_epoch,
            renewal=self.renewal,
        )


def after_settle_admission(
    store: SessionStore,
    plan: ContinuationDeadlinePlan | None,
    admission: DeadlineOutcomeAdmission,
    authority: Callable[[], Awaitable[None]] | None,
) -> RecoveryAdmissionHook | None:
    """The hook a settle runs once its decision is admitted: authority, then the deadline.

    Storing the deadline here, after every decision check passed and inside
    the admitted run, means a refused or stale decision never changes it.
    """

    if plan is None or plan.deadline is None:
        return authority

    async def admitted() -> Session:
        if authority is not None:
            await authority()
        return await plan.commit(store, admission)

    return admitted


def _replaces_deadlines(store: SessionStore) -> bool:
    from cayu.sessions.store import SessionStore as BaseSessionStore

    # The bound method, so an instance-level override counts too.
    method = getattr(store.replace_execution_deadline, "__func__", None)
    return method is not BaseSessionStore.replace_execution_deadline


async def _store_before_admission(
    store: SessionStore,
    current: Session,
    deadline: ExecutionDeadline,
    renewal: ExecutionDeadlineRenewal | None,
    *,
    explicit: bool,
) -> Session:
    try:
        return await store.replace_execution_deadline(
            current.id,
            deadline,
            expected_session_instance_id=current.instance_id,
            expected_run_epoch=current.run_epoch,
            renewal=renewal,
        )
    except NotImplementedError:
        if explicit:
            raise
        # Automatic renewal is optional for external stores. Keep enforcing
        # their stored deadline when the adapter cannot replace it.
        return current


async def plan_continuation_deadline(
    app: CayuApp,
    session: Session,
    request: ContinuationRequest,
    *,
    participant_context: CollaborationAccessContext | None,
) -> ContinuationDeadlinePlan | None:
    """Check a settle's deadline replacement and plan it for admission.

    ``request`` is the approval or answer about to continue the session, with
    private session and action identities. A replacement is planned only when
    the request names the pause the session waits on now and the caller has
    the authority to continue it; the plan always carries the session as
    loaded now. Stores that set ``replaces_execution_deadline_in_admitted_run``
    (the built-in ones) store it after admission. Other stores keep the
    "between runs only" contract, so their replacement is stored here, before
    admission, as it was before; an adapter that raises
    ``NotImplementedError`` does so before the decision is claimed.
    """

    current = await app.session_store.load(session.id)
    if current is None:
        return None
    replacement = await continuation_deadline_replacement(
        app.session_store,
        current,
        request.execution_deadline,
        human_decision=type(request) is not ResumeRequest,
    )
    if replacement is None or not await _continues_the_session_as_it_is(
        app.session_store, current, request
    ):
        return ContinuationDeadlinePlan(session=current)
    explicit = request.execution_deadline is not None
    if not _replaces_deadlines(app.session_store):
        if explicit:
            raise NotImplementedError(
                "This SessionStore does not support replace_execution_deadline."
            )
        return ContinuationDeadlinePlan(session=current)
    from cayu.resource_access import current_binding

    binding = current_binding()
    if binding is not None and binding != current.invocation.resource_access:
        # Only the subject that owns the session may change its deadline.
        from cayu._resource_access_errors import ResourceAccessDenied

        raise ResourceAccessDenied()
    # Changing the deadline needs the same authority as continuing the session.
    await app._require_participant_execution(current, participant_context)
    deadline, renewal = replacement
    if type(request) is ResumeRequest or not getattr(
        app.session_store, "replaces_execution_deadline_in_admitted_run", False
    ):
        stored = await _store_before_admission(
            app.session_store, current, deadline, renewal, explicit=explicit
        )
        return ContinuationDeadlinePlan(session=stored)
    return ContinuationDeadlinePlan(session=current, deadline=deadline, renewal=renewal)


async def prepare_continuation_deadline(
    app: CayuApp,
    session: Session,
    request: ResumeRequest,
    *,
    participant_context: CollaborationAccessContext | None,
) -> Session:
    """Store a resume's new deadline before it is admitted; return the session as stored.

    A resume's own preconditions (its fences, no pending approval or
    question, the caller's authority) are checked first. Settles store their
    deadline only once their decision is admitted; see
    ``plan_continuation_deadline``.
    """

    plan = await plan_continuation_deadline(
        app, session, request, participant_context=participant_context
    )
    return session if plan is None else plan.session
