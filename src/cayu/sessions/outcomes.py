"""``run_to_completion`` — collapse an agent run's event stream into a result.

``app.run``/``app.resume`` never raise on a model or tool failure: a failed run
ends in a terminal ``session.failed`` event, not an exception. That is the right
contract for a streaming runtime, but it means the naive "await it and read the
answer" path has to inspect the event stream by hand. This helper does that once,
correctly, and returns the answer most callers want.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cayu._validation import copy_json_value
from cayu.events import Event, EventType, copy_event
from cayu.runtime._exception_detail import (
    MAX_EXCEPTION_SUMMARY_UTF8_BYTES,
    ExceptionDetail,
    exception_detail,
)
from cayu.runtime.tool_completion import ToolCompletionResult
from cayu.sessions.base import ResumeRequest
from cayu.sessions.records import SessionStatus
from cayu.vaults import SecretRedactor

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.sessions.base import RunRequest


@dataclass(frozen=True)
class StructuredOutputResult:
    """The last successfully validated structured output from a completed run.

    The wrapper distinguishes a valid JSON ``null`` value (``output is None``)
    from a run that did not validate structured output at all
    (``RunOutcome.structured_output is None``).
    """

    output: Any
    name: str | None
    attempt: int
    max_retries: int


@dataclass(frozen=True)
class RunOutcome:
    """The terminal result of an agent run.

    - ``status`` is ``SessionStatus.COMPLETED``, ``SessionStatus.FAILED``, or
      ``SessionStatus.INTERRUPTED``.
    - ``final_text`` is the last completed model turn's text output (``""`` if
      the latest model turn completed without text or failed before completion).
    - ``error`` is the failure message when ``status`` is ``SessionStatus.FAILED``, else ``None``.
      When the run raised instead of ending in ``session.failed``, it also names
      an exception group's leaf errors and the cause chain.
    - ``error_detail`` is the structured, redacted form of that raised exception
      (type, message, causes and group leaves), or ``None`` when the failure came
      from a ``session.failed`` event or the run did not fail.
    - ``events`` is the full event stream, if you need more than the summary.
    - ``structured_output`` is the last successfully validated structured value,
      including valid JSON ``null``, or ``None`` when the run had no validated output.
    - ``tool_completion`` is the successful final tool's durable host-rendering basis,
      or ``None`` for ordinary model completion. It does not synthesize ``final_text``.
    - ``interaction_id`` identifies the interaction that produced ``final_text`` or
      the final interaction-scoped terminal outcome.
    """

    session_id: str
    status: SessionStatus
    final_text: str
    error: str | None
    events: tuple[Event, ...]
    structured_output: StructuredOutputResult | None = None
    interaction_id: str | None = None
    tool_completion: ToolCompletionResult | None = None
    error_detail: ExceptionDetail | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(copy_event(event) for event in self.events))
        if self.error_detail is not None and type(self.error_detail) is not ExceptionDetail:
            raise TypeError("error_detail must be an ExceptionDetail.")
        if self.tool_completion is not None:
            if type(self.tool_completion) is not ToolCompletionResult:
                raise TypeError("tool_completion must be a ToolCompletionResult.")
            object.__setattr__(
                self,
                "tool_completion",
                ToolCompletionResult.model_validate(
                    self.tool_completion.model_dump(mode="python", warnings=False)
                ),
            )
        if self.structured_output is not None:
            if type(self.structured_output) is not StructuredOutputResult:
                raise TypeError("structured_output must be a StructuredOutputResult.")
            object.__setattr__(
                self,
                "structured_output",
                StructuredOutputResult(
                    output=copy_json_value(
                        self.structured_output.output,
                        "structured_output.output",
                    ),
                    name=self.structured_output.name,
                    attempt=self.structured_output.attempt,
                    max_retries=self.structured_output.max_retries,
                ),
            )

    @property
    def ok(self) -> bool:
        """True only if the run reached ``session.completed``."""
        return self.status is SessionStatus.COMPLETED


async def run_to_completion(app: CayuApp, request: RunRequest | ResumeRequest) -> RunOutcome:
    """Run an agent to a terminal state and return a :class:`RunOutcome`.

    Consumes ``app.run(request)``, or ``app.resume(request)`` for a
    :class:`ResumeRequest`, and returns the final text, terminal status, and
    error (if any), so you branch on ``outcome.ok`` / ``outcome.status`` instead of
    hand-inspecting events. To continue a conversation, pass
    ``ResumeRequest(session_id=outcome.session_id, messages=[...])``; a new
    ``RunRequest`` always starts an empty session. A model/tool failure surfaces as
    ``status == SessionStatus.FAILED`` with ``error`` set. Setup-time exceptions
    before a terminal session event are also converted into a failed outcome,
    with ``error_detail`` describing the exception, its causes and, for an
    exception group, its leaf errors. Both are redacted with the app's secret
    redactor.
    """
    events: list[Event] = []
    current_turn_text: list[str] = []
    final_text = ""
    status = SessionStatus.INTERRUPTED
    error: str | None = None
    error_detail: ExceptionDetail | None = None
    tool_completion: ToolCompletionResult | None = None
    structured_output: StructuredOutputResult | None = None
    interaction_id: str | None = None
    session_id = request.session_id or ""

    try:
        stream = app.resume(request) if isinstance(request, ResumeRequest) else app.run(request)
        async for event in stream:
            events.append(event)
            session_id = event.session_id
            if event.interaction_id is not None:
                interaction_id = event.interaction_id
            payload = event.payload or {}
            # EventType is a StrEnum — compare with == never `is` (see the note on Event.type).
            if event.type == EventType.MODEL_STARTED:
                current_turn_text = []
                final_text = ""
            elif event.type == EventType.MODEL_TEXT_DELTA:
                delta = payload.get("delta")
                if isinstance(delta, str):
                    current_turn_text.append(delta)
            elif event.type == EventType.MODEL_COMPLETED:
                final_text = "".join(current_turn_text)
            elif event.type == EventType.STRUCTURED_OUTPUT_VALIDATED:
                structured_output = StructuredOutputResult(
                    output=copy_json_value(payload.get("output"), "structured_output"),
                    name=payload.get("name") if isinstance(payload.get("name"), str) else None,
                    attempt=payload["attempt"],
                    max_retries=payload["max_retries"],
                )
            elif event.type == EventType.SESSION_COMPLETED:
                status = SessionStatus.COMPLETED
                value = payload.get("tool_completion")
                tool_completion = (
                    None if value is None else ToolCompletionResult.model_validate(value)
                )
            elif event.type == EventType.SESSION_FAILED:
                status = SessionStatus.FAILED
                failure = payload.get("error")
                error = failure if isinstance(failure, str) else None
            elif event.type == EventType.SESSION_INTERRUPTED:
                status = SessionStatus.INTERRUPTED
    except Exception as exc:
        status = SessionStatus.FAILED
        redactor = getattr(app, "_secret_redactor", None)
        if not isinstance(redactor, SecretRedactor):
            redactor = SecretRedactor()
        error_detail = exception_detail(exc, redactor=redactor)
        error = redactor.redact_text_bounded(
            error_detail.summary(),
            max_bytes=MAX_EXCEPTION_SUMMARY_UTF8_BYTES,
        )

    return RunOutcome(
        session_id=session_id,
        status=status,
        final_text=final_text,
        error=error,
        events=tuple(events),
        structured_output=structured_output,
        tool_completion=tool_completion,
        interaction_id=interaction_id,
        error_detail=error_detail,
    )
