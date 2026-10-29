"""``run_once`` — run a named session, or return the outcome it already reached.

A handler that names its sessions deterministically (``<run>-<finding>-verify``)
can call ``run_once`` for every step. After a restart it gets each finished
step's stored outcome back without a model call and pays only for steps that
never finished.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from cayu.events import Event, EventType, event_with_durable_sequence
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.outcomes import RunOutcome, run_to_completion
from cayu.sessions.records import Session, SessionStatus

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.sessions.requests import RunRequest

# The statuses ``run_to_completion`` can return. A stored session in one of
# them has a settled outcome; any other status means a run still owns it or was
# lost before it ended.
_SETTLED_STATUSES = frozenset(
    {SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.INTERRUPTED}
)
_INVOCATION_OPENING_EVENTS = frozenset(
    {EventType.INTERACTION_STARTED, EventType.INTERACTION_RESUMED}
)
# Within one invocation the session starts or resumes, then settles with a
# terminal event; ``None`` marks an opening.
_SESSION_BOUNDARY_EVENTS: dict[str, SessionStatus | None] = {
    EventType.SESSION_STARTED: None,
    EventType.SESSION_RESUMED: None,
    EventType.SESSION_COMPLETED: SessionStatus.COMPLETED,
    EventType.SESSION_FAILED: SessionStatus.FAILED,
    EventType.SESSION_INTERRUPTED: SessionStatus.INTERRUPTED,
}
_EVENT_PAGE_SIZE = 5000


class RunOnceRefused(ValueError):
    """``run_once`` found a session under the request's id that it will not reuse."""

    def __init__(self, session_id: str, message: str) -> None:
        self.session_id = session_id
        super().__init__(message)


class RunOnceMismatch(RunOnceRefused):
    """The stored session was started for a different agent or structured output."""

    def __init__(
        self,
        session_id: str,
        field: Literal["agent_name", "structured_output"],
        message: str,
    ) -> None:
        self.field = field
        super().__init__(session_id, message)


class RunOnceSessionNotFinished(RunOnceRefused):
    """The stored session has no settled outcome; ``run_once`` left it unchanged.

    The session is either still running somewhere or was abandoned when its
    worker died, possibly after its status changed but before its terminal
    event was published. Start a fresh attempt under a new session id, and
    settle the old one with the session recovery APIs if it was abandoned.
    """

    def __init__(
        self, session_id: str, status: SessionStatus, *, reason: str | None = None
    ) -> None:
        self.status = status
        state = status.value if reason is None else f"{status.value} but {reason}"
        super().__init__(
            session_id,
            f"Session {session_id!r} exists but is {state}, so it has not finished; run_once "
            "left it unchanged. Start a new attempt under a different session id, and "
            "recover or interrupt the old session if its run was abandoned.",
        )


async def run_once(app: CayuApp, request: RunRequest) -> RunOutcome:
    """Run ``request`` unless its session already ran, and return the outcome.

    ``request.session_id`` is required and names the work:

    - No session with that id: run it with :func:`run_to_completion`.
    - A session with that id ended ``COMPLETED``, ``FAILED`` or ``INTERRUPTED``:
      return the :class:`RunOutcome` of its latest run, resume or approval
      continuation, rebuilt from stored events (status, final text, structured
      output, usage, tool calls and error) without calling the model. The session's agent and structured-output spec must match the
      request, or :class:`RunOnceMismatch` is raised.
    - A session with that id is still pending, running or interrupting, or its
      terminal event was never published: raise
      :class:`RunOnceSessionNotFinished` and leave the session unchanged.

    A failed or interrupted session is returned as it ended; check
    ``outcome.ok`` and use a new session id to redo it. Messages and other request fields are not compared, so give each distinct
    piece of work its own session id. A setup failure is not stored, so its
    reused outcome has no ``error_detail``.
    """
    session_id = request.session_id
    if session_id is None:
        raise ValueError("run_once needs request.session_id to name the session.")
    session = await app.session_store.load(session_id)
    if session is None:
        outcome = await run_to_completion(app, request)
        if outcome.events:
            return outcome
        # Setup failed before any event. Another caller may have created the
        # session first; if so, answer from that session instead.
        session = await app.session_store.load(session_id)
        if session is None:
            return outcome
    return await _stored_outcome(app, session, request)


async def _stored_outcome(app: CayuApp, session: Session, request: RunRequest) -> RunOutcome:
    from cayu.runtime._session_queries import query_all_event_records

    _require_matching_request(session, request)
    if session.status not in _SETTLED_STATUSES:
        raise RunOnceSessionNotFinished(session.id, session.status)
    records = await query_all_event_records(
        app.session_store, EventQuery(session_id=session.id, limit=_EVENT_PAGE_SIZE)
    )
    events: list[Event] = []
    for record in records:
        # Match the events ``app.run`` yields, so the outcome is the one the
        # original caller saw.
        projected = app._project_persisted_event_record_for_exposure(record)
        events.append(event_with_durable_sequence(projected.event, projected.sequence))
    # A resumed session holds several invocations (a run, a resume, an approval
    # continuation), each opening with an interaction start or resume. Rebuild
    # only the latest one, as the stream that produced it would have.
    start = next(
        (
            index
            for index in range(len(events) - 1, -1, -1)
            if events[index].type in _INVOCATION_OPENING_EVENTS
        ),
        0,
    )
    events = events[start:]
    # The status is written before the terminal event is published. Reuse only
    # when the latest invocation's own terminal event matches it; otherwise the
    # outcome (an approval id, say) would be incomplete.
    last = next(
        (event.type for event in reversed(events) if event.type in _SESSION_BOUNDARY_EVENTS),
        None,
    )
    if last is None or _SESSION_BOUNDARY_EVENTS[last] is not session.status:
        raise RunOnceSessionNotFinished(
            session.id,
            session.status,
            reason="its latest run's terminal event was never published",
        )
    return RunOutcome.from_events(events, session_id=session.id)


def _require_matching_request(session: Session, request: RunRequest) -> None:
    from cayu.execution_profiles import ExecutionProfileComponentClass
    from cayu.runtime._resume_configuration import _structured_component
    from cayu.sessions._execution_profile_checkpoint import (
        execution_profile_from_session_metadata,
    )

    if session.agent_name != request.agent_name:
        raise RunOnceMismatch(
            session.id,
            "agent_name",
            f"Session {session.id!r} was run by agent {session.agent_name!r}, not "
            f"{request.agent_name!r}. Use a different session id for different work.",
        )
    try:
        profile = execution_profile_from_session_metadata(session.metadata)
        stored = profile.component(ExecutionProfileComponentClass.STRUCTURED_OUTPUT)
    except (KeyError, ValueError):
        stored = None
    if stored is None or stored != _structured_component(request.structured_output):
        reason = (
            "has no recorded structured output to compare with this request"
            if stored is None
            else "was run with a different structured output than this request (schema, "
            "name, strategy, retries or repair prompt)"
        )
        raise RunOnceMismatch(
            session.id,
            "structured_output",
            f"Session {session.id!r} {reason}. Use a different session id for different work.",
        )
