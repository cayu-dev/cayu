from __future__ import annotations

from collections.abc import Iterable

from cayu.core.events import Event
from cayu.core.messages import Message, MessageRole, TextPart
from cayu.evals.models import Trajectory
from cayu.runtime.app import CayuApp
from cayu.runtime.sessions import (
    Session,
    SessionOrder,
    SessionQuery,
    TerminalSessionEvidence,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
)
from cayu.runtime.usage import SessionUsageSummary, session_usage_summary

# Fresh evals retain descendant evidence for assertions and replay. Page the durable
# parent index instead of assuming the first page is complete, while retaining a hard
# stop so a corrupt or adversarial tree cannot keep a trial alive indefinitely.
_CHILD_TRAJECTORY_PAGE_SIZE = 1000
_CHILD_TRAJECTORY_MAX_PAGES = 100


class _IncompleteFlag:
    """Node-local signal that a best-effort descendant walk was incomplete."""

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = False


async def _load_terminal_evidence(app: CayuApp, session_id: str) -> TerminalSessionEvidence:
    if not app.session_store.supports_terminal_session_evidence:
        raise NotImplementedError
    return await app.session_store.load_terminal_session_evidence(session_id)


def _project_terminal_evidence(
    evidence: TerminalSessionEvidence,
) -> tuple[Session, tuple[Event, ...], tuple[Message, ...], SessionUsageSummary]:
    """Build one trajectory node's assertion substrate from exact terminal evidence."""

    events = tuple(record.event for record in evidence.events)
    transcript = tuple(record.message for record in evidence.transcript)
    usage_summary = session_usage_summary(evidence.session.id, list(events))
    return evidence.session, events, transcript, usage_summary


async def _build_child_trajectories(
    app: CayuApp,
    parent_session_id: str | None,
    *,
    visited: set[str],
    incomplete: _IncompleteFlag | None = None,
) -> tuple[Trajectory, ...]:
    """Best-effort descendant capture used by an ordinary fresh eval.

    Production-session promotion adds a fail-closed policy around this shared
    construction seam. Fresh evals preserve their existing behavior: a store failure,
    contradictory child, or exhausted page bound marks the node incomplete instead of
    aborting an otherwise completed trial.
    """

    if parent_session_id is None:
        return ()
    children: list[Trajectory] = []
    cursor: str | None = None
    for _ in range(_CHILD_TRAJECTORY_MAX_PAGES):
        try:
            result = await app.session_store.list_sessions(
                SessionQuery(
                    parent_session_id=parent_session_id,
                    limit=_CHILD_TRAJECTORY_PAGE_SIZE,
                    cursor=cursor,
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
        except Exception:
            if incomplete is not None:
                incomplete.value = True
            return tuple(children)
        for child_session in result.sessions:
            if child_session.id in visited:
                if incomplete is not None:
                    incomplete.value = True
                continue
            visited.add(child_session.id)
            child = await _load_child_trajectory(
                app,
                child_session,
                expected_parent_session_id=parent_session_id,
                visited=visited,
            )
            if child is not None:
                children.append(child)
                if child.children_incomplete and incomplete is not None:
                    incomplete.value = True
            elif incomplete is not None:
                incomplete.value = True
        cursor = result.next_cursor
        if cursor is None:
            return tuple(children)
    if incomplete is not None:
        incomplete.value = True
    return tuple(children)


async def _load_child_trajectory(
    app: CayuApp,
    session: Session,
    *,
    expected_parent_session_id: str,
    visited: set[str],
) -> Trajectory | None:
    try:
        evidence = await _load_terminal_evidence(app, session.id)
    except TerminalSessionEvidenceError as exc:
        if (
            exc.code != TerminalSessionEvidenceErrorCode.SESSION_INTERRUPTED
            or not app.session_store.supports_runner_owned_interrupted_evidence
        ):
            return None
        try:
            evidence = await app.session_store.load_runner_owned_interrupted_evidence(
                session.id,
                expected_parent_session_id=expected_parent_session_id,
            )
        except Exception:
            return None
    except Exception:
        return None
    terminal_session, events, transcript, usage_summary = _project_terminal_evidence(evidence)
    if terminal_session.parent_session_id != expected_parent_session_id:
        return None
    grandchildren_incomplete = _IncompleteFlag()
    grandchildren = await _build_child_trajectories(
        app,
        session.id,
        visited=visited,
        incomplete=grandchildren_incomplete,
    )
    return Trajectory(
        session=terminal_session,
        events=events,
        transcript=transcript,
        usage_summary=usage_summary,
        final_output=final_output_text(transcript),
        children=grandchildren,
        children_incomplete=grandchildren_incomplete.value,
    )


def final_output_text(transcript: Iterable[Message]) -> str:
    """Return the text of the last assistant message in ``transcript``."""

    for message in reversed(tuple(transcript)):
        if message.role != MessageRole.ASSISTANT:
            continue
        text = "".join(part.text for part in message.content if type(part) is TextPart)
        if text:
            return text
    return ""
