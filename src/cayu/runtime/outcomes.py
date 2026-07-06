"""``run_to_completion`` — collapse an agent run's event stream into a result.

``app.run``/``app.resume`` never raise on a model or tool failure: a failed run
ends in a terminal ``session.failed`` event, not an exception. That is the right
contract for a streaming runtime, but it means the naive "await it and read the
answer" path has to inspect the event stream by hand. This helper does that once,
correctly, and returns the answer most callers want.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cayu.core.events import Event, EventType
from cayu.runtime.sessions import SessionStatus

if TYPE_CHECKING:
    from cayu.runtime.app import CayuApp
    from cayu.runtime.sessions import RunRequest


@dataclass(frozen=True)
class RunOutcome:
    """The terminal result of an agent run.

    - ``status`` is ``SessionStatus.COMPLETED``, ``SessionStatus.FAILED``, or
      ``SessionStatus.INTERRUPTED``.
    - ``final_text`` is the last completed model turn's text output (``""`` if
      the latest model turn completed without text or failed before completion).
    - ``error`` is the failure message when ``status`` is ``SessionStatus.FAILED``, else ``None``.
    - ``events`` is the full event stream, if you need more than the summary.
    """

    session_id: str
    status: SessionStatus
    final_text: str
    error: str | None
    events: tuple[Event, ...]

    @property
    def ok(self) -> bool:
        """True only if the run reached ``session.completed``."""
        return self.status is SessionStatus.COMPLETED


async def run_to_completion(app: CayuApp, request: RunRequest) -> RunOutcome:
    """Run an agent to a terminal state and return a :class:`RunOutcome`.

    Consumes ``app.run(request)`` and returns the final text, terminal status, and
    error (if any), so you branch on ``outcome.ok`` / ``outcome.status`` instead of
    hand-inspecting events. A model/tool failure surfaces as
    ``status == SessionStatus.FAILED`` with ``error`` set. Setup-time exceptions
    before a terminal session event are also converted into a failed outcome.
    """
    events: list[Event] = []
    current_turn_text: list[str] = []
    final_text = ""
    status = SessionStatus.INTERRUPTED
    error: str | None = None
    session_id = request.session_id or ""

    try:
        async for event in app.run(request):
            events.append(event)
            session_id = event.session_id
            payload = event.payload or {}
            if event.type is EventType.MODEL_STARTED:
                current_turn_text = []
                final_text = ""
            elif event.type is EventType.MODEL_TEXT_DELTA:
                delta = payload.get("delta")
                if isinstance(delta, str):
                    current_turn_text.append(delta)
            elif event.type is EventType.MODEL_COMPLETED:
                final_text = "".join(current_turn_text)
            elif event.type is EventType.SESSION_COMPLETED:
                status = SessionStatus.COMPLETED
            elif event.type is EventType.SESSION_FAILED:
                status = SessionStatus.FAILED
                failure = payload.get("error")
                error = failure if isinstance(failure, str) else None
            elif event.type is EventType.SESSION_INTERRUPTED:
                status = SessionStatus.INTERRUPTED
    except Exception as exc:
        status = SessionStatus.FAILED
        error = f"{type(exc).__name__}: {exc}"

    return RunOutcome(
        session_id=session_id,
        status=status,
        final_text=final_text,
        error=error,
        events=tuple(events),
    )
