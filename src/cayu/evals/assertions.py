from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from cayu.artifacts import ArtifactScope
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, TextPart, ToolCallPart, ToolResultPart
from cayu.evals.models import EvalAssertionResult, EvalContext, ProbeRequirements
from cayu.runtime.costs import ModelCatalog, PricingCatalog, estimate_session_cost
from cayu.runtime.sessions import SessionStatus


class EvalAssertion(ABC):
    """Assertion over a completed Cayu eval case."""

    @property
    def name(self) -> str:
        return type(self).__name__

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
    ) -> EvalAssertionResult:
        return EvalAssertionResult(
            name=self.name,
            passed=True,
            message=message,
            metadata={} if metadata is None else metadata,
        )

    def failed(
        self,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> EvalAssertionResult:
        return EvalAssertionResult(
            name=self.name,
            passed=False,
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
            score=score,
            threshold=threshold,
            passed=score >= threshold,
            message=message,
            metadata={} if metadata is None else metadata,
        )


class SessionStatusIs(EvalAssertion):
    def __init__(self, status: SessionStatus | str) -> None:
        self.status = SessionStatus(status)

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        actual = context.session.status if context.session is not None else None
        if actual == self.status:
            return self.passed(f"Session status is {self.status.value}.")
        return self.failed(
            f"Expected session status {self.status.value}, got "
            f"{actual.value if actual is not None else 'none'}.",
            metadata={
                "expected": self.status.value,
                "actual": actual.value if actual is not None else None,
            },
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
            message += " Child capture is incomplete; additional matching sessions may exist."
        return self.failed(message, metadata=metadata)


class FinalOutputContains(EvalAssertion):
    def __init__(self, text: str) -> None:
        self.text = _require_text(text, "text")

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        if self.text in context.final_output:
            return self.passed("Final output contains expected text.")
        return self.failed(
            "Final output did not contain expected text.",
            metadata={"expected": self.text, "final_output": context.final_output},
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
        count = len(_tool_start_events(context.events, self.tool_name))
        if count < self.min_count:
            return self.failed(
                f"Expected tool {self.tool_name} at least {self.min_count} time(s), got {count}.",
                metadata={"tool_name": self.tool_name, "count": count},
            )
        if self.max_count is not None and count > self.max_count:
            return self.failed(
                f"Expected tool {self.tool_name} at most {self.max_count} time(s), got {count}.",
                metadata={"tool_name": self.tool_name, "count": count},
            )
        return self.passed(
            f"Observed tool {self.tool_name} {count} time(s).",
            metadata={"tool_name": self.tool_name, "count": count},
        )


class ToolNotCalled(ToolCalled):
    def __init__(self, tool_name: str) -> None:
        super().__init__(tool_name, min_count=0, max_count=0)

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        result = await super().evaluate(context)
        if result.passed:
            return self.passed(
                f"Tool {self.tool_name} was not called, as expected.",
                metadata=result.metadata,
            )
        return self.failed(
            f"Tool {self.tool_name} was called but was expected not to be.",
            metadata=result.metadata,
        )


class ToolArgsContain(EvalAssertion):
    def __init__(self, tool_name: str, expected: Mapping[str, Any]) -> None:
        self.tool_name = _require_text(tool_name, "tool_name")
        if not isinstance(expected, Mapping):
            raise TypeError("expected must be a mapping.")
        self.expected = dict(expected)

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        starts = _tool_start_events(context.events, self.tool_name)
        for event in starts:
            arguments = event.payload.get("arguments")
            if isinstance(arguments, Mapping) and _mapping_contains(arguments, self.expected):
                return self.passed(
                    f"Tool {self.tool_name} arguments contained expected values.",
                    metadata={"expected": self.expected, "actual": dict(arguments)},
                )
        return self.failed(
            f"No {self.tool_name} call contained expected arguments.",
            metadata={
                "expected": self.expected,
                "actual": [
                    dict(event.payload.get("arguments", {}))
                    for event in starts
                    if isinstance(event.payload.get("arguments"), Mapping)
                ],
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
        count = _tool_call_count(context)
        if count <= self.maximum:
            return self.passed(
                f"Tool call count {count} is within limit {self.maximum}.",
                metadata={"tool_calls": count},
            )
        return self.failed(
            f"Tool call count {count} exceeded limit {self.maximum}.",
            metadata={"tool_calls": count, "maximum": self.maximum},
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
        if count <= self.maximum:
            return self.passed(
                f"Model step count {count} is within limit {self.maximum}.",
                metadata={"model_steps": count},
            )
        return self.failed(
            f"Model step count {count} exceeded limit {self.maximum}.",
            metadata={"model_steps": count, "maximum": self.maximum},
        )


class UsageRecorded(EvalAssertion):
    def __init__(self, *, min_total_tokens: int = 1) -> None:
        self.min_total_tokens = _nonnegative_int(min_total_tokens, "min_total_tokens")

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        if context.usage_summary is None:
            return self.failed("Cannot verify token usage because no usage summary was recorded.")
        total_tokens = context.usage_summary.usage.total_tokens
        metadata = {"total_tokens": total_tokens, "minimum": self.min_total_tokens}
        if total_tokens >= self.min_total_tokens:
            return self.passed(
                f"Total tokens {total_tokens} meets minimum {self.min_total_tokens}.",
                metadata=metadata,
            )
        return self.failed(
            f"Total tokens {total_tokens} is below minimum {self.min_total_tokens}.",
            metadata=metadata,
        )


class MaxTotalTokens(EvalAssertion):
    def __init__(self, maximum: int) -> None:
        self.maximum = _nonnegative_int(maximum, "maximum")

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        if context.usage_summary is None:
            return self.failed("Cannot verify total tokens because no usage was recorded.")
        total = context.usage_summary.usage.total_tokens
        if total <= self.maximum:
            return self.passed(
                f"Total tokens {total} is within limit {self.maximum}.",
                metadata={"total_tokens": total},
            )
        return self.failed(
            f"Total tokens {total} exceeded limit {self.maximum}.",
            metadata={"total_tokens": total, "maximum": self.maximum},
        )


class MaxEstimatedCost(EvalAssertion):
    def __init__(
        self,
        maximum: Decimal | str | int | float,
        pricing: PricingCatalog | ModelCatalog,
        *,
        currency: str = "USD",
    ) -> None:
        self.maximum = Decimal(str(maximum))
        self.pricing = pricing
        self.currency = currency

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        if context.session is None:
            return self.failed("Cannot estimate cost because no session was created.")
        # Cost is a pure function of the durable events + pricing, so it reads straight off
        # the trajectory — same estimator the app uses, no live handle required.
        summary = estimate_session_cost(
            session_id=context.session.id,
            events=list(context.events),
            pricing=self.pricing,
            currency=self.currency,
        )
        actual = summary.total_cost
        if actual <= self.maximum:
            return self.passed(
                f"Estimated cost {actual} {self.currency} is within limit {self.maximum}.",
                metadata={"estimated_cost": str(actual), "maximum": str(self.maximum)},
            )
        return self.failed(
            f"Estimated cost {actual} {self.currency} exceeded limit {self.maximum}.",
            metadata={"estimated_cost": str(actual), "maximum": str(self.maximum)},
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
            return self.failed("No workspace is configured for the eval session.")
        # A probed path is always a key (None = absent/unreadable); a missing key means the
        # path was never captured — only reachable when replaying a trajectory against an
        # assertion whose path the original run didn't probe. Report that distinctly.
        if self.path not in probes.workspace_files:
            return self.failed(f"Workspace path was not captured in this trajectory: {self.path}")
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
            return self.failed("No workspace is configured for the eval session.")
        if self.path not in probes.workspace_files:
            return self.failed(f"Workspace path was not captured in this trajectory: {self.path}")
        content_bytes = probes.workspace_files[self.path]
        if content_bytes is None:
            return self.failed(f"Could not read workspace file: {self.path}")
        try:
            content = content_bytes.decode(self.encoding)
        except Exception as exc:
            return self.failed(
                f"Could not decode workspace file: {self.path}",
                metadata={"error": str(exc)},
            )
        if self.text in content:
            return self.passed(f"Workspace file {self.path} contains expected text.")
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
            return self.failed("No artifact store is configured for the eval session.")
        session_id = context.session.id if context.session is not None else None
        environment_name = context.session.environment_name if context.session is not None else None
        # scope=None means "an artifact this session created", so it must not match a
        # prior case's ENVIRONMENT-scoped artifact (those persist across cases in the
        # same environment). Request scope=ArtifactScope.ENVIRONMENT explicitly for that.
        scope = ArtifactScope.SESSION if self.scope is None else self.scope
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


def _tool_call_count(context: EvalContext) -> int:
    if context.usage_summary is not None:
        return context.usage_summary.tool_calls
    return _event_count(context.events, EventType.TOOL_CALL_STARTED)


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
