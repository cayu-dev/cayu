from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any

from cayu.artifacts import ArtifactScope
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, TextPart, ToolCallPart, ToolResultPart
from cayu.evals.corpus import (
    EVIDENCE_MAX_CHILD_SESSIONS,
    EVIDENCE_MAX_FINAL_OUTPUT_CHARS,
    EVIDENCE_MAX_MODEL_STEPS,
    EVIDENCE_MAX_TOOL_CALLS,
    EVIDENCE_MAX_TOTAL_TOKENS,
)
from cayu.evals.evidence import (
    AssertionCostEvidenceV1,
    EvidenceState,
    _canonical_decimal,
    _project_tool_evidence,
)
from cayu.evals.models import EvalAssertionResult, EvalContext, EvalOutcome, ProbeRequirements
from cayu.evals.portable_evaluation import (
    _evaluate_child_status,
    _evaluate_final_output,
    _evaluate_max_cost,
    _evaluate_maximum,
    _evaluate_root_status,
    _evaluate_tool_called,
    _evaluate_tools_in_order,
    _evaluate_usage_recorded,
)
from cayu.runtime.costs import PriceBook, SessionCostSummary, estimate_session_cost
from cayu.runtime.sessions import SessionStatus

_TOOL_ARGUMENT_TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }
)


class EvalAssertion(ABC):
    """Assertion over a completed Cayu eval case."""

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def assertion_revision(self) -> str | None:
        """Stable definition revision carried by results when one exists."""

        return None

    @property
    def evaluates_failed_session(self) -> bool:
        """Whether a failed root session is assertion evidence instead of a runner error."""

        return False

    @abstractmethod
    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        """Evaluate this assertion against a Cayu runtime context."""

    def required_probes(self) -> ProbeRequirements:
        """Live-environment data this assertion needs captured into the Trajectory.

        Default: none. Workspace/artifact assertions override this so the runner snapshots
        exactly what they read while the environment is still live, keeping evaluation offline.
        """
        return ProbeRequirements()

    def passed(
        self,
        message: str = "",
        *,
        metadata: dict[str, Any] | None = None,
        cost_summary: SessionCostSummary | None = None,
    ) -> EvalAssertionResult:
        return EvalAssertionResult(
            name=self.name,
            assertion_revision=self.assertion_revision,
            outcome=EvalOutcome.PASSED,
            score=1.0,
            message=message,
            metadata={} if metadata is None else metadata,
            cost_summary=cost_summary,
        )

    def failed(
        self,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
        cost_summary: SessionCostSummary | None = None,
    ) -> EvalAssertionResult:
        return EvalAssertionResult(
            name=self.name,
            assertion_revision=self.assertion_revision,
            outcome=EvalOutcome.FAILED,
            score=0.0,
            message=message,
            metadata={} if metadata is None else metadata,
            cost_summary=cost_summary,
        )

    def unavailable(
        self,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
        cost_summary: SessionCostSummary | None = None,
    ) -> EvalAssertionResult:
        return EvalAssertionResult(
            name=self.name,
            assertion_revision=self.assertion_revision,
            outcome=EvalOutcome.UNAVAILABLE,
            message=message,
            metadata={} if metadata is None else metadata,
            cost_summary=cost_summary,
        )

    def error(
        self,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> EvalAssertionResult:
        return EvalAssertionResult(
            name=self.name,
            assertion_revision=self.assertion_revision,
            outcome=EvalOutcome.ERROR,
            message=message,
            metadata={} if metadata is None else metadata,
        )

    def score_result(
        self,
        score: float,
        *,
        threshold: float = 1.0,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> EvalAssertionResult:
        # Seam for graded / model-judge checks: pass/fail is derived from the
        # threshold, while the continuous score is preserved on the result.
        return EvalAssertionResult(
            name=self.name,
            assertion_revision=self.assertion_revision,
            score=score,
            threshold=threshold,
            outcome=EvalOutcome.PASSED if score >= threshold else EvalOutcome.FAILED,
            message=message,
            metadata={} if metadata is None else metadata,
        )


class SessionStatusIs(EvalAssertion):
    def __init__(self, status: SessionStatus | str) -> None:
        self.status = SessionStatus(status)

    @property
    def evaluates_failed_session(self) -> bool:
        return True

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        return _evaluate_root_status(
            name=self.name,
            expected=self.status.value,
            actual=(context.session.status.value if context.session is not None else None),
        )


class SessionCompleted(SessionStatusIs):
    def __init__(self) -> None:
        super().__init__(SessionStatus.COMPLETED)


class SessionFailed(SessionStatusIs):
    def __init__(self) -> None:
        super().__init__(SessionStatus.FAILED)


class SessionInterrupted(SessionStatusIs):
    def __init__(self) -> None:
        super().__init__(SessionStatus.INTERRUPTED)


class ChildSessionCompleted(EvalAssertion):
    def __init__(self, *, agent_name: str | None = None, min_count: int = 1) -> None:
        self.agent_name = None if agent_name is None else _require_text(agent_name, "agent_name")
        self.min_count = _nonnegative_int(min_count, "min_count")

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        if not context.root_evidence_available:
            return _evaluate_child_status(
                name=self.name,
                expected=SessionStatus.COMPLETED.value,
                statuses=(),
                state="unavailable",
                minimum=self.min_count,
                maximum=None,
            )
        if self.agent_name is None:
            retained_children = context.trajectory.children[:EVIDENCE_MAX_CHILD_SESSIONS]
            statuses = tuple(
                child.session.status.value
                for child in retained_children
                if child.session is not None
            )
            if context.trajectory.children_incomplete or len(statuses) != len(retained_children):
                state: EvidenceState = "unavailable"
                statuses = ()
            elif len(context.trajectory.children) > EVIDENCE_MAX_CHILD_SESSIONS:
                state = "limit_exceeded"
            else:
                state = "complete"
            return _evaluate_child_status(
                name=self.name,
                expected=SessionStatus.COMPLETED.value,
                statuses=statuses,
                state=state,
                minimum=self.min_count,
                maximum=None,
            )
        observed_children = [
            {
                "session_id": child.session.id if child.session is not None else None,
                "status": child.session.status.value if child.session is not None else None,
                "agent_name": child.session.agent_name if child.session is not None else None,
            }
            for child in context.trajectory.children
        ]
        matching_session_ids = [
            child.session.id
            for child in context.trajectory.children
            if child.session is not None
            and child.session.status == SessionStatus.COMPLETED
            and (self.agent_name is None or child.session.agent_name == self.agent_name)
        ]
        matching_count = len(matching_session_ids)
        metadata = {
            "agent_name": self.agent_name,
            "minimum": self.min_count,
            "matching_count": matching_count,
            "matching_session_ids": matching_session_ids,
            "observed_children": observed_children,
            "children_incomplete": context.trajectory.children_incomplete,
        }
        agent_filter = f" for agent {self.agent_name}" if self.agent_name is not None else ""
        if matching_count >= self.min_count:
            return self.passed(
                f"Observed {matching_count} completed direct child session(s){agent_filter}.",
                metadata=metadata,
            )
        message = (
            f"Expected at least {self.min_count} completed direct child session(s){agent_filter}, "
            f"got {matching_count}."
        )
        if context.trajectory.children_incomplete:
            return self.unavailable(
                message + " Child capture is incomplete; additional matching sessions may exist.",
                metadata=metadata,
            )
        return self.failed(message, metadata=metadata)


class FinalOutputContains(EvalAssertion):
    def __init__(self, text: str) -> None:
        self.text = _require_text(text, "text")

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        if not context.root_evidence_available:
            state: EvidenceState = "unavailable"
            actual = ""
        elif len(context.final_output) > EVIDENCE_MAX_FINAL_OUTPUT_CHARS:
            state = "limit_exceeded"
            actual = context.final_output[:EVIDENCE_MAX_FINAL_OUTPUT_CHARS]
        else:
            state = "complete"
            actual = context.final_output
        return _evaluate_final_output(
            name=self.name,
            expected=self.text,
            actual=actual,
            state=state,
            contains=True,
        )


class FinalOutputMatches(EvalAssertion):
    def __init__(self, pattern: str) -> None:
        self.pattern = _require_text(pattern, "pattern")
        self._compiled = re.compile(pattern)

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        if self._compiled.search(context.final_output):
            return self.passed("Final output matched expected pattern.")
        return self.failed(
            "Final output did not match expected pattern.",
            metadata={"pattern": self.pattern, "final_output": context.final_output},
        )


class TranscriptContains(EvalAssertion):
    def __init__(self, text: str) -> None:
        self.text = _require_text(text, "text")

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        transcript_text = "\n".join(_message_text(message) for message in context.transcript)
        if self.text in transcript_text:
            return self.passed("Transcript contains expected text.")
        return self.failed(
            "Transcript did not contain expected text.",
            metadata={"expected": self.text},
        )


class EventOccurred(EvalAssertion):
    def __init__(
        self,
        event_type: EventType | str,
        *,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> None:
        self.event_type = _event_type_value(event_type)
        self.min_count = _nonnegative_int(min_count, "min_count")
        self.max_count = None if max_count is None else _nonnegative_int(max_count, "max_count")

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        count = sum(
            1 for event in context.events if _event_type_value(event.type) == self.event_type
        )
        if count < self.min_count:
            return self.failed(
                f"Expected at least {self.min_count} {self.event_type} event(s), got {count}.",
                metadata={"event_type": self.event_type, "count": count},
            )
        if self.max_count is not None and count > self.max_count:
            return self.failed(
                f"Expected at most {self.max_count} {self.event_type} event(s), got {count}.",
                metadata={"event_type": self.event_type, "count": count},
            )
        return self.passed(
            f"Observed {count} {self.event_type} event(s).",
            metadata={"event_type": self.event_type, "count": count},
        )


class EventNotOccurred(EventOccurred):
    def __init__(self, event_type: EventType | str) -> None:
        super().__init__(event_type, min_count=0, max_count=0)

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        result = await super().evaluate(context)
        if result.outcome in {EvalOutcome.UNAVAILABLE, EvalOutcome.ERROR}:
            return result
        if result.passed:
            return self.passed(
                f"Event {self.event_type} did not occur, as expected.",
                metadata=result.metadata,
            )
        return self.failed(
            f"Event {self.event_type} occurred but was expected not to.",
            metadata=result.metadata,
        )


class EventPayloadContains(EvalAssertion):
    def __init__(
        self,
        event_type: EventType | str,
        expected: Mapping[str, Any],
        *,
        min_count: int = 1,
    ) -> None:
        self.event_type = _event_type_value(event_type)
        if not isinstance(expected, Mapping):
            raise TypeError("expected must be a mapping.")
        self.expected = dict(expected)
        self.min_count = _nonnegative_int(min_count, "min_count")

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        observed_payloads = [
            event.payload
            for event in context.events
            if _event_type_value(event.type) == self.event_type
        ]
        matching_count = sum(
            1 for payload in observed_payloads if _mapping_contains(payload, self.expected)
        )
        metadata = {
            "expected": self.expected,
            "event_type": self.event_type,
            "matching_count": matching_count,
        }
        if matching_count >= self.min_count:
            return self.passed(
                f"Observed {matching_count} {self.event_type} event payload(s) "
                "containing expected values.",
                metadata=metadata,
            )
        return self.failed(
            f"Expected at least {self.min_count} {self.event_type} event payload(s) "
            f"containing expected values, got {matching_count}.",
            metadata={**metadata, "observed_payloads": observed_payloads},
        )


class ToolCalled(EvalAssertion):
    def __init__(
        self,
        tool_name: str,
        *,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> None:
        self.tool_name = _require_text(tool_name, "tool_name")
        self.min_count = _nonnegative_int(min_count, "min_count")
        self.max_count = None if max_count is None else _nonnegative_int(max_count, "max_count")

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        _, started_tool_names, _, state = _direct_tool_evidence(context)
        return _evaluate_tool_called(
            name=self.name,
            tool_name=self.tool_name,
            started_tool_names=started_tool_names,
            state=state,
            minimum=self.min_count,
            maximum=self.max_count,
        )


class ToolNotCalled(ToolCalled):
    def __init__(self, tool_name: str) -> None:
        super().__init__(tool_name, min_count=0, max_count=0)

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        result = await super().evaluate(context)
        if result.outcome in {EvalOutcome.UNAVAILABLE, EvalOutcome.ERROR}:
            return result
        if result.passed:
            return self.passed(
                f"Tool {self.tool_name} was not called, as expected.",
                metadata=result.metadata,
            )
        return self.failed(
            f"Tool {self.tool_name} was called but was expected not to be.",
            metadata=result.metadata,
        )


class ToolsCalledInOrder(EvalAssertion):
    """Require exact model-requested tool names in transcript order."""

    def __init__(self, tool_names: Iterable[str]) -> None:
        if isinstance(tool_names, str | bytes):
            raise TypeError("tool_names must be an iterable of tool names, not text.")
        try:
            values = tuple(tool_names)
        except TypeError as exc:
            raise TypeError("tool_names must be an iterable of tool names.") from exc
        self.tool_names = tuple(
            _require_text(tool_name, f"tool_names[{index}]")
            for index, tool_name in enumerate(values)
        )

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        actual, _, _, state = _direct_tool_evidence(context)
        return _evaluate_tools_in_order(
            name=self.name,
            expected=self.tool_names,
            actual=actual,
            state=state,
        )


class ToolArgsContain(EvalAssertion):
    def __init__(self, tool_name: str, expected: Mapping[str, Any]) -> None:
        self.tool_name = _require_text(tool_name, "tool_name")
        if not isinstance(expected, Mapping):
            raise TypeError("expected must be a mapping.")
        self.expected = dict(expected)

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        observed_arguments: list[dict[str, Any]] = []
        starts = _tool_start_events(context.events, self.tool_name)
        started_idempotency_keys: set[str] = set()
        for event in starts:
            arguments_state = event.payload.get("arguments_state")
            idempotency_key = event.payload.get("idempotency_key")
            if arguments_state == "quarantined" and (
                type(idempotency_key) is str and idempotency_key
            ):
                started_idempotency_keys.add(idempotency_key)
                continue
            # Compatibility with evaluations over historical event-only fixtures.
            if arguments_state is not None:
                continue
            arguments = event.payload.get("arguments")
            if isinstance(arguments, Mapping):
                observed_arguments.append(dict(arguments))
            if isinstance(arguments, Mapping) and _mapping_contains(arguments, self.expected):
                return self.passed(
                    f"Tool {self.tool_name} arguments contained expected values.",
                    metadata={"expected": self.expected, "actual": dict(arguments)},
                )
        for event in _tool_terminal_events(context.events, self.tool_name):
            idempotency_key = event.payload.get("idempotency_key")
            if (
                type(idempotency_key) is not str
                or idempotency_key not in started_idempotency_keys
                or event.payload.get("arguments_state") != "finalized"
            ):
                continue
            arguments = event.payload.get("arguments")
            if not isinstance(arguments, Mapping):
                continue
            observed_arguments.append(dict(arguments))
            if _mapping_contains(arguments, self.expected):
                return self.passed(
                    f"Tool {self.tool_name} arguments contained expected values.",
                    metadata={"expected": self.expected, "actual": dict(arguments)},
                )
        return self.failed(
            f"No {self.tool_name} call contained expected arguments.",
            metadata={
                "expected": self.expected,
                "actual": observed_arguments,
            },
        )


class ToolResultContains(EvalAssertion):
    def __init__(self, tool_name: str, text: str) -> None:
        self.tool_name = _require_text(tool_name, "tool_name")
        self.text = _require_text(text, "text")

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        for message in context.transcript:
            for part in message.content:
                if (
                    type(part) is ToolResultPart
                    and part.tool_name == self.tool_name
                    and self.text in part.content
                ):
                    return self.passed(f"Tool {self.tool_name} result contained expected text.")
        return self.failed(
            f"No {self.tool_name} tool result contained expected text.",
            metadata={"expected": self.text},
        )


class MaxToolCalls(EvalAssertion):
    def __init__(self, maximum: int) -> None:
        self.maximum = _nonnegative_int(maximum, "maximum")

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        _, _, count, state = _direct_tool_evidence(context)
        return _evaluate_maximum(
            name=self.name,
            evidence_area="tool call",
            actual=count,
            state=state,
            maximum=self.maximum,
        )


class MaxModelSteps(EvalAssertion):
    def __init__(self, maximum: int) -> None:
        self.maximum = _nonnegative_int(maximum, "maximum")

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        count = (
            context.usage_summary.model_steps
            if context.usage_summary is not None
            else _event_count(context.events, EventType.MODEL_COMPLETED)
        )
        if not context.root_evidence_available:
            state: EvidenceState = "unavailable"
            count = None
        elif count > EVIDENCE_MAX_MODEL_STEPS:
            state = "limit_exceeded"
        else:
            state = "complete"
        return _evaluate_maximum(
            name=self.name,
            evidence_area="model step",
            actual=count,
            state=state,
            maximum=self.maximum,
        )


class UsageRecorded(EvalAssertion):
    def __init__(self, *, min_total_tokens: int = 1) -> None:
        self.min_total_tokens = _nonnegative_int(min_total_tokens, "min_total_tokens")

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        usage = context.usage_summary if context.root_evidence_available else None
        total_tokens = None if usage is None else usage.usage.total_tokens
        state: EvidenceState = (
            "unavailable"
            if total_tokens is None
            else ("limit_exceeded" if total_tokens > EVIDENCE_MAX_TOTAL_TOKENS else "complete")
        )
        return _evaluate_usage_recorded(
            name=self.name,
            total_tokens=total_tokens,
            state=state,
            minimum=self.min_total_tokens,
        )


class MaxTotalTokens(EvalAssertion):
    def __init__(self, maximum: int) -> None:
        self.maximum = _nonnegative_int(maximum, "maximum")

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        usage = context.usage_summary if context.root_evidence_available else None
        total_tokens = None if usage is None else usage.usage.total_tokens
        state: EvidenceState = (
            "unavailable"
            if total_tokens is None
            else ("limit_exceeded" if total_tokens > EVIDENCE_MAX_TOTAL_TOKENS else "complete")
        )
        return _evaluate_maximum(
            name=self.name,
            evidence_area="total token",
            actual=total_tokens,
            state=state,
            maximum=self.maximum,
        )


class MaxEstimatedCost(EvalAssertion):
    def __init__(
        self,
        maximum: Decimal | str | int | float,
        pricing: PriceBook,
        *,
        currency: str = "USD",
    ) -> None:
        self.maximum = Decimal(str(maximum))
        self.pricing = pricing
        self.currency = currency

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        model_steps = (
            context.usage_summary.model_steps
            if context.usage_summary is not None
            else _event_count(context.events, EventType.MODEL_COMPLETED)
        )
        if (
            context.session is None
            or not context.root_evidence_available
            or model_steps > EVIDENCE_MAX_MODEL_STEPS
        ):
            return _evaluate_max_cost(
                name=self.name,
                maximum=self.maximum,
                currency=self.currency,
                cost=None,
            )
        # Cost is a pure function of the durable events + pricing, so it reads straight off
        # the trajectory — same estimator the app uses, no live handle required.
        summary = estimate_session_cost(
            session_id=context.session.id,
            events=list(context.events),
            pricing=self.pricing,
            currency=self.currency,
        )
        result = _evaluate_max_cost(
            name=self.name,
            maximum=self.maximum,
            currency=summary.currency,
            cost=AssertionCostEvidenceV1(
                currency=summary.currency,
                total_cost=_canonical_decimal(summary.total_cost),
                model_steps=summary.model_steps,
                priced_model_steps=summary.priced_model_steps,
                unpriced_model_steps=summary.unpriced_model_steps,
            ),
        )
        return EvalAssertionResult.model_validate(
            {
                **result.model_dump(mode="python"),
                "cost_summary": summary,
            }
        )


class WorkspaceFileExists(EvalAssertion):
    """Assert a path is present in the workspace when the case finishes.

    This means "present now", not "this case created it": cases in a suite share the app's
    workspace unless the caller registers an environment factory (fresh workspace per case),
    so a file left by an earlier case will satisfy this. See docs/evals.md "Workspace isolation".
    """

    def __init__(self, path: str) -> None:
        self.path = _require_text(path, "path")

    def required_probes(self) -> ProbeRequirements:
        return ProbeRequirements(workspace_paths=frozenset({self.path}))

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        probes = context.probes
        if not probes.workspace_available:
            return self.unavailable("Workspace evidence was not captured for this trajectory.")
        if self.path in probes.workspace_unavailable_paths:
            return self.unavailable(f"Workspace path capture was unavailable: {self.path}")
        # A captured path is always a key (None = observed absent). A missing key means the
        # original trajectory never probed the assertion's requested path.
        if self.path not in probes.workspace_files:
            return self.unavailable(
                f"Workspace path was not captured in this trajectory: {self.path}"
            )
        if probes.workspace_files[self.path] is None:
            return self.failed(f"Workspace file not found: {self.path}")
        return self.passed(f"Workspace file exists: {self.path}")


class WorkspaceFileContains(EvalAssertion):
    """Assert a workspace file contains expected text when the case finishes.

    Like WorkspaceFileExists, this reads the workspace as it stands at case end, not "produced
    by this case": a file left by an earlier case in a shared workspace counts. Register an
    environment factory for per-case isolation. See docs/evals.md "Workspace isolation".
    """

    def __init__(self, path: str, text: str, *, encoding: str = "utf-8") -> None:
        self.path = _require_text(path, "path")
        self.text = _require_text(text, "text")
        self.encoding = _require_text(encoding, "encoding")

    def required_probes(self) -> ProbeRequirements:
        return ProbeRequirements(workspace_paths=frozenset({self.path}))

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        probes = context.probes
        if not probes.workspace_available:
            return self.unavailable("Workspace evidence was not captured for this trajectory.")
        if self.path in probes.workspace_unavailable_paths:
            return self.unavailable(f"Workspace path capture was unavailable: {self.path}")
        if self.path not in probes.workspace_files:
            return self.unavailable(
                f"Workspace path was not captured in this trajectory: {self.path}"
            )
        content_bytes = probes.workspace_files[self.path]
        if content_bytes is None:
            return self.failed(f"Workspace file not found: {self.path}")
        stat = probes.workspace_file_stats[self.path]
        try:
            content = content_bytes.decode(self.encoding)
        except Exception as exc:
            if stat.truncated:
                return self.unavailable(
                    f"Workspace file {self.path} was truncated before it could be decoded.",
                    metadata={
                        "captured_bytes": len(content_bytes),
                        "total_bytes": stat.total_bytes,
                        "error": str(exc),
                    },
                )
            return self.failed(
                f"Could not decode workspace file: {self.path}",
                metadata={"error": str(exc)},
            )
        if self.text in content:
            return self.passed(f"Workspace file {self.path} contains expected text.")
        if stat.truncated:
            return self.unavailable(
                f"Workspace file {self.path} was truncated before the expected text was found.",
                metadata={
                    "captured_bytes": len(content_bytes),
                    "total_bytes": stat.total_bytes,
                },
            )
        return self.failed(
            f"Workspace file {self.path} did not contain expected text.",
            metadata={"expected": self.text},
        )


class ArtifactCreated(EvalAssertion):
    def __init__(
        self,
        *,
        filename: str | None = None,
        content_type: str | None = None,
        scope: ArtifactScope | str | None = None,
        min_count: int = 1,
    ) -> None:
        self.filename = filename
        self.content_type = content_type
        self.scope = None if scope is None else ArtifactScope(scope)
        self.min_count = _nonnegative_int(min_count, "min_count")

    def required_probes(self) -> ProbeRequirements:
        scope = ArtifactScope.SESSION if self.scope is None else self.scope
        return ProbeRequirements(artifact_scopes=frozenset({scope}))

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        probes = context.probes
        if not probes.artifacts_available:
            return self.unavailable("Artifact evidence was not captured for this trajectory.")
        session_id = context.session.id if context.session is not None else None
        environment_name = context.session.environment_name if context.session is not None else None
        # scope=None means "an artifact this session created", so it must not match a
        # prior case's ENVIRONMENT-scoped artifact (those persist across cases in the
        # same environment). Request scope=ArtifactScope.ENVIRONMENT explicitly for that.
        scope = ArtifactScope.SESSION if self.scope is None else self.scope
        if scope in probes.artifact_scopes_unavailable:
            return self.unavailable(f"Artifact capture was unavailable for scope {scope.value}.")
        scope_captured = scope in probes.artifact_scopes_captured
        scope_truncated = scope in probes.artifact_scopes_truncated
        if not scope_captured and not scope_truncated:
            return self.unavailable(
                f"Artifact scope was not captured in this trajectory: {scope.value}"
            )
        artifacts = [
            artifact
            for artifact in probes.artifacts
            if artifact.scope == scope
            and (
                scope != ArtifactScope.SESSION
                or session_id is None
                or artifact.session_id == session_id
            )
            and (
                scope != ArtifactScope.ENVIRONMENT
                or environment_name is None
                or artifact.environment_name == environment_name
            )
            and (self.filename is None or artifact.filename == self.filename)
            and (self.content_type is None or artifact.content_type == self.content_type)
        ]
        count = len(artifacts)
        if count >= self.min_count:
            return self.passed(
                f"Observed {count} matching artifact(s).",
                metadata={"artifact_ids": [artifact.id for artifact in artifacts]},
            )
        if scope_truncated:
            return self.unavailable(
                f"Artifact listing for scope {scope.value} was truncated before enough "
                "matching artifacts were observed.",
                metadata={"count": count, "minimum": self.min_count},
            )
        return self.failed(
            f"Expected at least {self.min_count} matching artifact(s), got {count}.",
            metadata={"count": count},
        )


def _require_text(value: str, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _nonnegative_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return value


def _event_type_value(event_type: EventType | str) -> str:
    return event_type.value if isinstance(event_type, EventType) else str(event_type)


def _event_count(events: tuple[Event, ...], event_type: EventType | str) -> int:
    expected = _event_type_value(event_type)
    return sum(1 for event in events if _event_type_value(event.type) == expected)


def _tool_start_events(events: tuple[Event, ...], tool_name: str) -> list[Event]:
    return [
        event
        for event in events
        if event.type == EventType.TOOL_CALL_STARTED and event.tool_name == tool_name
    ]


def _direct_tool_evidence(
    context: EvalContext,
) -> tuple[tuple[str, ...], tuple[str, ...], int | None, EvidenceState]:
    """Apply the portable tool completeness and cardinality contract to a direct assertion."""

    return _project_tool_evidence(
        context.trajectory,
        max_tool_calls=EVIDENCE_MAX_TOOL_CALLS,
        app=None,
        root_evidence_available=context.root_evidence_available,
        allow_event_count_fallback=True,
    )


def _tool_terminal_events(events: tuple[Event, ...], tool_name: str) -> list[Event]:
    return [
        event
        for event in events
        if event.type in _TOOL_ARGUMENT_TERMINAL_EVENT_TYPES and event.tool_name == tool_name
    ]


def _message_text(message: Message) -> str:
    parts: list[str] = []
    for part in message.content:
        if type(part) is TextPart:
            parts.append(part.text)
        elif type(part) is ToolCallPart:
            parts.append(f"{part.tool_name}({part.arguments})")
        elif type(part) is ToolResultPart:
            parts.append(part.content)
    return "\n".join(parts)


def _mapping_contains(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                return False
            if not _mapping_contains(actual_value, expected_value):
                return False
        elif actual_value != expected_value:
            return False
    return True
