"""Bounded observations of invocation health and typed operation outcomes.

These observations never determine eval scores or retry decisions. Source events
remain the authority; report counts describe only the supplied evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from hashlib import sha256
from typing import TYPE_CHECKING, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    model_validator,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.core.events import Event, EventType
from cayu.tools.web_access import WebAccessEvidence

if TYPE_CHECKING:
    from cayu.evals.models import Trajectory

_MAX_REFERENCES = 256


OperationOutcomeState = Literal[
    "completed",
    "failed",
    "blocked",
    "cancelled",
    "not_reported",
    "zero_exit",
    "nonzero_exit",
    "timed_out",
    "response",
    "http_error",
]


class HttpOperationOutcomeV1(BaseModel):
    """Opt-in ToolResult.structured['operation_outcome'] HTTP response contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    protocol: Literal["http"] = "http"
    status_code: StrictInt = Field(ge=100, le=599)


class OperationOutcomeEvidence(BaseModel):
    """Content-free reference to one observation in the original event stream."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    session_id: str | None = Field(default=None, max_length=1024)
    event_id: str = Field(max_length=1024)
    tool_call_id: str | None = Field(default=None, max_length=1024)
    execution_id: str | None = Field(default=None, max_length=1024)
    dimension: Literal["invocation", "command", "http"]
    state: OperationOutcomeState
    exit_code: StrictInt | None = None
    timed_out: StrictBool | None = None
    cancelled: StrictBool | None = None
    http_status_code: StrictInt | None = Field(default=None, ge=100, le=599)

    @model_validator(mode="after")
    def validate_observation(self) -> OperationOutcomeEvidence:
        command_fields = (self.exit_code, self.timed_out, self.cancelled)
        if self.dimension == "command":
            expected = (
                "timed_out"
                if self.timed_out is True
                else "cancelled"
                if self.cancelled is True
                else "not_reported"
                if self.exit_code is None
                else "zero_exit"
                if self.exit_code == 0
                else "nonzero_exit"
            )
            if self.state != expected or self.http_status_code is not None:
                raise ValueError("Command classification contradicts its reported outcome.")
        elif self.dimension == "http":
            expected = (
                "not_reported"
                if self.http_status_code is None
                else "http_error"
                if self.http_status_code >= 400
                else "response"
            )
            if self.state != expected or any(value is not None for value in command_fields):
                raise ValueError("HTTP classification contradicts its reported outcome.")
        elif (
            self.state not in {"completed", "failed", "blocked", "cancelled", "not_reported"}
            or self.http_status_code is not None
            or any(value is not None for value in command_fields)
        ):
            raise ValueError("Invocation observations must retain only lifecycle classification.")
        return self


class OperationOutcomeCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    invocation_completed: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    invocation_failed: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    invocation_blocked: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    invocation_cancelled: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    invocation_not_reported: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    command_zero_exit: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    command_nonzero_exit: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    command_timed_out: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    command_cancelled: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    command_not_reported: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    http_response: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    http_error: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    http_not_reported: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)


class OperationOutcomeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    evidence_state: Literal["not_reported", "observed", "incomplete"] = "not_reported"
    counts: OperationOutcomeCounts = Field(default_factory=OperationOutcomeCounts)
    evidence: tuple[OperationOutcomeEvidence, ...] = Field(default=(), max_length=_MAX_REFERENCES)
    omitted_evidence_count: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)

    @model_validator(mode="after")
    def validate_counts(self) -> OperationOutcomeSummary:
        total = sum(self.counts.model_dump().values())
        if total != len(self.evidence) + self.omitted_evidence_count:
            raise ValueError("Outcome counts must match retained and omitted observations.")
        retained = Counter(
            "http_error" if row.state == "http_error" else f"{row.dimension}_{row.state}"
            for row in self.evidence
        )
        counts = self.counts.model_dump()
        if any(key not in counts or value > counts[key] for key, value in retained.items()):
            raise ValueError("Outcome references must agree with dimension counts.")
        if self.evidence_state == "not_reported" and total:
            raise ValueError("Not-reported summaries cannot carry observations.")
        return self


def _identity(event: Event) -> tuple[object, ...]:
    payload = event.payload
    # The round disambiguates provider call IDs reused in subsequent turns. Older
    # events can use the stable invocation idempotency key instead.
    return (
        event.session_id,
        _text(payload.get("tool_round_id")),
        _text(payload.get("idempotency_key")),
        _text(payload.get("tool_call_id")) or event.id,
    )


def _text(value: object) -> str | None:
    return value if type(value) is str and 0 < len(value) <= 1024 else None


def _reference(value: str | None) -> str | None:
    if value is None or len(value) <= 1024:
        return value
    return "sha256:" + sha256(value.encode()).hexdigest()


def _command(event: Event, payload: dict) -> OperationOutcomeEvidence:
    exit_code = payload.get("exit_code")
    timeout = payload.get("timed_out")
    cancelled = payload.get("cancelled")
    exit_code = exit_code if type(exit_code) is int else None
    timeout = timeout if type(timeout) is bool else None
    cancelled = cancelled if type(cancelled) is bool else None
    state = (
        "timed_out"
        if timeout is True
        else "cancelled"
        if cancelled is True
        else "not_reported"
        if exit_code is None
        else "zero_exit"
        if exit_code == 0
        else "nonzero_exit"
    )
    return OperationOutcomeEvidence(
        session_id=_reference(event.session_id),
        event_id=_reference(event.id) or event.id,
        tool_call_id=_text(event.payload.get("tool_call_id")),
        execution_id=_text(event.payload.get("execution_id")),
        dimension="command",
        state=state,
        exit_code=exit_code,
        timed_out=timeout,
        cancelled=cancelled,
    )


def _http_status(structured: dict) -> int | None:
    try:
        if "operation_outcome" in structured:
            return HttpOperationOutcomeV1.model_validate(
                structured["operation_outcome"]
            ).status_code
        if "access" in structured:
            return WebAccessEvidence.model_validate(structured["access"]).status_code
    except ValidationError:
        return None
    # Existing built-in web/browser fetch error contract (not arbitrary stdout).
    status = structured.get("status_code")
    if structured.get("error") == "http_status" and type(status) is int and 100 <= status <= 599:
        return status
    return None


def summarize_operation_outcomes(
    events: Iterable[Event],
    *,
    evidence_complete: bool = False,
) -> OperationOutcomeSummary:
    """Project supplied events, counting runner observations once per execution.

    `evidence_complete` describes capture coverage, not operation success. Runner
    events take precedence over enclosing command-tool results. Legacy runner
    events without execution IDs are counted by event ID and retain unknown flags.
    """
    unique = {(event.session_id, event.id): event for event in events}
    invocations: dict[tuple[object, ...], Event] = {}
    starts: dict[tuple[object, ...], Event] = {}
    runners: dict[tuple[object, ...], Event] = {}
    runner_starts: dict[tuple[object, ...], Event] = {}
    legacy_starts: dict[tuple[object, ...], list[Event]] = {}
    legacy_completions: Counter = Counter()
    runner_invocations: set[tuple[object, ...]] = set()
    terminal_types: dict[str, OperationOutcomeState] = {
        EventType.TOOL_CALL_COMPLETED: "completed",
        EventType.TOOL_CALL_FAILED: "failed",
        EventType.TOOL_CALL_BLOCKED: "blocked",
    }
    for event in unique.values():
        identity = _identity(event)
        if event.type in terminal_types:
            invocations[identity] = event
        elif event.type == EventType.TOOL_CALL_STARTED:
            starts[identity] = event
        elif event.type in {EventType.RUNNER_EXEC_STARTED, EventType.RUNNER_EXEC_COMPLETED}:
            runner_invocations.add(identity)
            execution = (*identity, _text(event.payload.get("execution_id")) or event.id)
            if event.type == EventType.RUNNER_EXEC_COMPLETED:
                runners[execution] = event
                if _text(event.payload.get("execution_id")) is None:
                    legacy_completions[identity] += 1
            elif _text(event.payload.get("execution_id")) is not None:
                runner_starts[execution] = event
            else:
                legacy_starts.setdefault(identity, []).append(event)
    # Legacy starts have no exact pairing identity. Retain excess starts as unknown,
    # and explicitly mark coverage incomplete rather than inventing a pairing.
    for identity, observed_starts in legacy_starts.items():
        evidence_complete = False
        for event in observed_starts[legacy_completions[identity] :]:
            runner_starts[(*identity, event.id)] = event

    counts = dict.fromkeys(OperationOutcomeCounts.model_fields, 0)
    references: list[OperationOutcomeEvidence] = []
    observed_count = 0

    def retain(row: OperationOutcomeEvidence) -> None:
        nonlocal observed_count
        observed_count += 1
        key = "http_error" if row.state == "http_error" else f"{row.dimension}_{row.state}"
        counts[key] += 1
        if len(references) < _MAX_REFERENCES:
            references.append(row)

    for identity, event in {**starts, **invocations}.items():
        state = terminal_types.get(event.type, "not_reported")
        if event.payload.get("interrupted") is True:
            state = "cancelled"
        retain(
            OperationOutcomeEvidence(
                session_id=_reference(event.session_id),
                event_id=_reference(event.id) or event.id,
                tool_call_id=_text(event.payload.get("tool_call_id")),
                dimension="invocation",
                state=state,
            )
        )
        result = event.payload.get("result")
        structured = result.get("structured") if type(result) is dict else None
        structured = structured if type(structured) is dict else {}
        if identity not in runner_invocations and (
            event.tool_name in {"exec_command", "run_command"}
            or all(key in structured for key in ("exit_code", "timed_out", "cancelled"))
        ):
            retain(_command(event, structured))
        status = _http_status(structured)
        retain(
            OperationOutcomeEvidence(
                session_id=_reference(event.session_id),
                event_id=_reference(event.id) or event.id,
                tool_call_id=_text(event.payload.get("tool_call_id")),
                dimension="http",
                state="not_reported"
                if status is None
                else "http_error"
                if status >= 400
                else "response",
                http_status_code=status,
            )
        )
    for execution, event in {**runner_starts, **runners}.items():
        retain(_command(event, event.payload if execution in runners else {}))
    return OperationOutcomeSummary(
        evidence_state="observed" if evidence_complete else "incomplete",
        counts=OperationOutcomeCounts(**counts),
        evidence=tuple(references),
        omitted_evidence_count=observed_count - len(references),
    )


def trajectory_operation_outcomes(trajectory: Trajectory | None) -> OperationOutcomeSummary:
    """Summarize all retained workflow/agent descendants without counting a tree twice."""
    if trajectory is None:
        return OperationOutcomeSummary()
    events: list[Event] = []
    complete = True
    pending = [trajectory]
    while pending:
        node = pending.pop()
        events.extend(node.events)
        complete = complete and not node.children_incomplete
        pending.extend(reversed(node.children))
    return summarize_operation_outcomes(events, evidence_complete=complete)


def operation_outcome_summary_text(summary: OperationOutcomeSummary) -> str:
    c = summary.counts
    return (
        f"Operation evidence: {summary.evidence_state}; invocations: "
        f"{c.invocation_completed} completed, {c.invocation_failed} failed, "
        f"{c.invocation_blocked} blocked, {c.invocation_cancelled} cancelled, "
        f"{c.invocation_not_reported} not reported; commands: "
        f"{c.command_zero_exit} zero exit, {c.command_nonzero_exit} nonzero exit, "
        f"{c.command_timed_out} timed out, {c.command_cancelled} cancelled, "
        f"{c.command_not_reported} not reported; HTTP: {c.http_response} responses below 400, "
        f"{c.http_error} errors, {c.http_not_reported} not reported. "
        f"{summary.omitted_evidence_count} evidence references omitted. "
        "Nonzero exits may be expected probes; these counts do not determine scores."
    )
