"""``run_to_completion`` — collapse an agent run's event stream into a result.

``app.run``/``app.resume`` never raise on a model or tool failure: a failed run
ends in a terminal ``session.failed`` event, not an exception. That is the right
contract for a streaming runtime, but it means the naive "await it and read the
answer" path has to inspect the event stream by hand. This helper does that once,
correctly, and returns the answer most callers want.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from cayu._validation import copy_json_value
from cayu.budgets.usage import AggregateUsageMetrics, build_aggregate_usage_metrics
from cayu.events import Event, EventType, copy_event
from cayu.runtime._exception_detail import (
    MAX_EXCEPTION_SUMMARY_UTF8_BYTES,
    ExceptionDetail,
    exception_detail,
)
from cayu.runtime.tool_completion import ToolCompletionResult
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import ResumeRequest
from cayu.tools.outcomes import ToolCallSummary
from cayu.vaults import SecretRedactor

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.collaboration.access import CollaborationAccessContext
    from cayu.sessions.requests import RunRequest


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


class RunFailureCode(StrEnum):
    """Stable codes for failures a caller may want to handle without parsing text.

    ``RunOutcome.failure_code`` and the ``failure_code`` field of a
    ``session.failed`` event carry a session-level code.
    ``ToolCallSummary.failure_code`` carries a tool-call code. Both are plain
    strings, so a code added in a later release still reads; compare against
    these members.
    """

    PROVIDER_STREAM_CAPACITY_EXHAUSTED = "provider_stream_capacity_exhausted"
    """A model call found ``ProviderStreamDeadlines.max_concurrent_streams``
    streams already open in the process and failed before reaching the provider."""

    TOOL_OUTPUT_TOO_LARGE = "tool_output_too_large"
    """A tool returned a result over its ``ToolSpec.max_terminal_payload_bytes``
    bound or the durable record limit, and the call failed with a bounded result."""


def failure_code_of(error: BaseException) -> RunFailureCode | None:
    """The stable code of a runtime-classified failure in ``error``, if any.

    Looks through exception groups and explicit ``raise ... from`` causes. Only
    the runtime's own exception types carry a code; a subclass does not.
    """

    from cayu._exception_groups import exception_cause, iter_exception_tree
    from cayu.providers.deadlines import ProviderStreamCapacityExhausted

    codes: dict[type[BaseException], RunFailureCode] = {
        ProviderStreamCapacityExhausted: RunFailureCode.PROVIDER_STREAM_CAPACITY_EXHAUSTED,
    }
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending and len(seen) < 64:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for node in iter_exception_tree(current):
            code = codes.get(type(node))
            if code is not None:
                return code
            seen.add(id(node))
            cause = exception_cause(node)
            if cause is not None and id(cause) not in seen:
                pending.append(cause)
    return None


class RunInterruption(BaseModel):
    """Observed interruption; resumability is a hint, never admission authority.

    ``None`` means the stream does not establish ordinary resumability. False
    requires a dedicated approval/input/recovery operation. Even a true hint
    remains subject to the current store, checkpoint and execution-profile checks.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[
        "tool_approval",
        "user_input",
        "manual_recovery",
        "operator",
        "limit",
        "delegated_action",
        "unclassified",
    ]
    reason: str | None = None
    resumable: bool | None = None
    approval_id: str | None = None
    input_id: str | None = None
    tool_round_id: str | None = None
    tool_call_id: str | None = None
    limit: str | None = None


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
    - ``failure_code`` is a stable :class:`RunFailureCode` value for a failure the
      runtime classifies, or ``None``. It is read from the ``session.failed``
      event, or from the exception when the run raised.
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
    provider_name: str | None = None
    requested_model: str | None = None
    model: str | None = None
    model_steps: int = 0
    unmeasured_model_steps: int = 0
    usage: AggregateUsageMetrics = field(default_factory=build_aggregate_usage_metrics)
    tool_calls: tuple[ToolCallSummary, ...] = ()
    interruption: RunInterruption | None = None
    failure_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(copy_event(event) for event in self.events))
        object.__setattr__(
            self,
            "usage",
            AggregateUsageMetrics.model_validate(self.usage.model_dump(mode="python")),
        )
        object.__setattr__(
            self,
            "tool_calls",
            tuple(call.model_copy(deep=True) for call in self.tool_calls),
        )
        if self.interruption is not None:
            object.__setattr__(
                self, "interruption", RunInterruption.model_validate(self.interruption)
            )
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
    def tool_call_count(self) -> int:
        """Distinct calls observed in this invocation, including blocked calls."""
        return len(self.tool_calls)

    @property
    def limit_reached(self) -> str | None:
        """The run or budget limit that stopped this run, or ``None``.

        It is the interruption's ``limit`` (``"output_tokens"``,
        ``"estimated_cost"`` and so on) when the run ended ``INTERRUPTED`` at a
        limit, including one reached while waiting for structured output.
        """
        if (
            self.status is not SessionStatus.INTERRUPTED
            or self.interruption is None
            or self.interruption.kind != "limit"
        ):
            return None
        return self.interruption.limit

    @property
    def resumable(self) -> bool | None:
        if self.status is not SessionStatus.INTERRUPTED or self.interruption is None:
            return None
        return self.interruption.resumable

    @classmethod
    def from_events(cls, events: Iterable[Event], *, session_id: str = "") -> RunOutcome:
        """Summarize a collected run, approval, input-resolution or recovery stream.

        Only supplied events contribute usage and call outcomes. This performs
        no store reads and makes no claim that an incomplete stream is terminal.
        """
        collector = _OutcomeCollector(session_id)
        for event in events:
            collector.observe(event)
        return collector.finish()

    def summary(
        self,
        *,
        max_bytes: int = 16_384,
        include_events: bool = False,
        include_arguments: bool = False,
        include_structured: bool = False,
        include_artifacts: bool = False,
    ) -> dict[str, Any]:
        """Export bounded presentation data; this is not an authorization boundary.

        The byte bound is measured as compact UTF-8 JSON with ensure_ascii=False.
        Omitted or shortened requested values are listed in ``omitted_fields``.
        Raw events and full argument/result/artifact payloads require opt-in and
        still obey the byte bound. Raw events can themselves contain all payloads.
        """
        from cayu.sessions._outcome_summary import summarize_outcome

        return summarize_outcome(
            self,
            max_bytes=max_bytes,
            include_events=include_events,
            include_arguments=include_arguments,
            include_structured=include_structured,
            include_artifacts=include_artifacts,
        )

    @property
    def ok(self) -> bool:
        """True only if the run reached ``session.completed``."""
        return self.status is SessionStatus.COMPLETED


async def run_to_completion(
    app: CayuApp,
    request: RunRequest | ResumeRequest,
    *,
    context: CollaborationAccessContext | None = None,
) -> RunOutcome:
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
    return await _run_to_completion(app, request, context=context)


async def _run_to_completion(
    app: CayuApp,
    request: RunRequest | ResumeRequest,
    *,
    context: CollaborationAccessContext | None = None,
    propagate_exceptions: bool = False,
) -> RunOutcome:
    collector = _OutcomeCollector(request.session_id or "")
    try:
        if isinstance(request, ResumeRequest):
            stream = (
                app.resume(request) if context is None else app.resume(request, context=context)
            )
        else:
            stream = app.run(request)
        async for event in stream:
            collector.observe(event)
    except Exception as exc:
        if propagate_exceptions:
            raise
        redactor = getattr(app, "_secret_redactor", None)
        if not isinstance(redactor, SecretRedactor):
            redactor = SecretRedactor()
        collector.error_detail = exception_detail(exc, redactor=redactor)
        collector.error = redactor.redact_text_bounded(
            collector.error_detail.summary(),
            max_bytes=MAX_EXCEPTION_SUMMARY_UTF8_BYTES,
        )
        collector.status = SessionStatus.FAILED
        code = failure_code_of(exc)
        collector.failure_code = None if code is None else code.value
    return collector.finish()


class _OutcomeCollector:
    """One incremental projection for live streams and already collected events."""

    def __init__(self, session_id: str) -> None:
        from cayu.sessions._outcome_projection import InvocationProjection

        self.session_id = session_id
        self.events: list[Event] = []
        self.current_turn_text: list[str] = []
        self.final_text = ""
        self.status = SessionStatus.INTERRUPTED
        self.error: str | None = None
        self.error_detail: ExceptionDetail | None = None
        self.failure_code: str | None = None
        self.tool_completion: ToolCompletionResult | None = None
        self.structured_output: StructuredOutputResult | None = None
        self.projection = InvocationProjection()

    def observe(self, event: Event) -> None:
        self.events.append(event)
        self.session_id = event.session_id
        if not self.projection.observe(event, len(self.events) - 1):
            return
        payload = event.payload
        if event.type in {EventType.SESSION_STARTED, EventType.SESSION_RESUMED}:
            self.status = SessionStatus.INTERRUPTED
            self.error = self.error_detail = self.failure_code = None
            self.tool_completion = self.structured_output = None
            self.current_turn_text = []
            self.final_text = ""
        elif event.type == EventType.MODEL_STARTED:
            self.current_turn_text = []
            self.final_text = ""
        elif event.type == EventType.MODEL_TEXT_DELTA:
            delta = payload.get("delta")
            if isinstance(delta, str):
                self.current_turn_text.append(delta)
        elif event.type == EventType.MODEL_COMPLETED:
            self.final_text = "".join(self.current_turn_text)
        elif event.type == EventType.STRUCTURED_OUTPUT_VALIDATED:
            self.structured_output = StructuredOutputResult(
                output=copy_json_value(payload.get("output"), "structured_output"),
                name=payload.get("name") if isinstance(payload.get("name"), str) else None,
                attempt=payload["attempt"],
                max_retries=payload["max_retries"],
            )
        elif event.type == EventType.SESSION_COMPLETED:
            self.status = SessionStatus.COMPLETED
            self.error = self.failure_code = None
            value = payload.get("tool_completion")
            self.tool_completion = (
                None if value is None else ToolCompletionResult.model_validate(value)
            )
        elif event.type == EventType.SESSION_FAILED:
            self.status = SessionStatus.FAILED
            failure = payload.get("error")
            self.error = failure if isinstance(failure, str) else None
            code = payload.get("failure_code")
            self.failure_code = code if isinstance(code, str) else None
        elif event.type == EventType.SESSION_INTERRUPTED:
            self.status = SessionStatus.INTERRUPTED
            self.error = self.failure_code = None

    def finish(self) -> RunOutcome:
        projection = self.projection.result()
        interruption = projection.pop("interruption")
        return RunOutcome(
            session_id=self.session_id,
            status=self.status,
            final_text=self.final_text,
            error=self.error,
            events=tuple(self.events),
            structured_output=self.structured_output,
            tool_completion=self.tool_completion,
            error_detail=self.error_detail,
            failure_code=self.failure_code,
            interruption=None if interruption is None else RunInterruption(**interruption),
            **projection,
        )
