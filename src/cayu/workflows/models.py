"""Typed result objects for workflow helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from cayu.failure_evidence import FailureEvidence, exception_evidence
from cayu.providers.base import ModelFinishReason


@dataclass(frozen=True, slots=True)
class StepUsage:
    """Token usage summed over the model calls of one ``step``.

    Counts come from the child session's ``model.completed`` events for the run
    that completed the step. ``model_calls`` counts every completed model call,
    including context compaction; ``unmeasured_model_calls`` counts calls whose
    provider reported no usable usage, so the token totals undercount by them.
    ``cached_input_tokens`` is the part of ``input_tokens`` the provider served
    from its prompt cache.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    total_tokens: int = 0
    model_calls: int = 0
    unmeasured_model_calls: int = 0


@dataclass(frozen=True, slots=True)
class StepResult:
    """Outcome of one successful ``step``.

    ``finish_reason`` is the normalized reason the step's final model call ended
    (for example ``ModelFinishReason.LENGTH`` when the reply hit the output cap),
    and ``raw_finish_reason`` is the provider's own value. Both are ``None`` when
    the child recorded no model call. ``usage`` sums the step's model calls.
    """

    step_id: str
    session_id: str
    text: str = ""
    output: Any = None
    finish_reason: ModelFinishReason | None = None
    raw_finish_reason: str | None = None
    usage: StepUsage = field(default_factory=StepUsage)

    @property
    def has_output(self) -> bool:
        return self.output is not None


@dataclass(frozen=True, slots=True)
class StepFailure:
    """A failed ``parallel`` branch."""

    error: str
    error_type: str
    step_id: str | None = None
    session_id: str | None = None
    evidence: FailureEvidence = field(default_factory=FailureEvidence)
    workflow_attempt_id: str | None = None


@dataclass(frozen=True, slots=True)
class ParallelResult:
    """Outcomes of a :func:`parallel` fan-out, in submission order."""

    results: tuple[StepResult | StepFailure, ...] = ()

    @property
    def successes(self) -> tuple[StepResult, ...]:
        return tuple(item for item in self.results if isinstance(item, StepResult))

    @property
    def failures(self) -> tuple[StepFailure, ...]:
        return tuple(item for item in self.results if isinstance(item, StepFailure))

    @property
    def ok(self) -> bool:
        return not self.failures

    def raise_for_failures(self) -> None:
        """Raise :class:`ParallelStepError` if any branch failed; else no-op."""
        failures = self.failures
        if failures:
            raise ParallelStepError(failures)

    @property
    def outputs(self) -> tuple[Any, ...]:
        """Validated outputs in order; raises if any branch failed."""
        self.raise_for_failures()
        return tuple(result.output for result in self.successes)


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """Verdict of a :func:`gated_loop` gate for one item."""

    passed: bool
    detail: Any = None


class StepError(Exception):
    """A ``step`` failed and carries ids for correlation."""

    def __init__(
        self,
        message: str,
        *,
        step_id: str | None = None,
        session_id: str | None = None,
        evidence: FailureEvidence | None = None,
        workflow_attempt_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.step_id = step_id
        self.session_id = session_id
        self._evidence = evidence
        self.workflow_attempt_id = workflow_attempt_id

    @property
    def evidence(self) -> FailureEvidence:
        """Structured snapshot, including a direct exception cause when present."""
        return (
            self._evidence
            if self._evidence is not None
            else exception_evidence(self.__cause__ or self)
        )


class ParallelStepError(Exception):
    """Raised by the fail-closed ``ParallelResult`` accessors when branches failed."""

    def __init__(self, failures: Iterable[StepFailure]) -> None:
        self.failures: tuple[StepFailure, ...] = tuple(failures)
        summary = ", ".join(
            f"{failure.step_id or '?'}: {failure.error}" for failure in self.failures
        )
        super().__init__(f"{len(self.failures)} parallel step(s) failed: {summary}")


def normalize_gate_outcome(value: GateOutcome | CommandResult | bool) -> GateOutcome:
    """Accept a :class:`GateOutcome`, a :class:`CommandResult`, or a bare bool from a gate."""
    if isinstance(value, GateOutcome):
        return value
    if isinstance(value, CommandResult):
        return value.gate_outcome()
    if isinstance(value, bool):
        return GateOutcome(passed=value)
    raise TypeError("A gate must return a GateOutcome, a CommandResult, or a bool.")


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Outcome of one :func:`~cayu.workflows.command` step, as journaled.

    ``stdout`` and ``stderr`` are the bounded, redacted text the journal holds,
    so a live run and a resumed run return the same value. ``replayed`` is true
    when the result came from the journal instead of a new execution.
    """

    step_id: str
    argv: tuple[str, ...]
    exit_code: int
    timed_out: bool = False
    cancelled: bool = False
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    duration_ms: int = 0
    environment_name: str | None = None
    replayed: bool = False

    @property
    def passed(self) -> bool:
        """True when the command exited 0 and was neither timed out nor cancelled."""
        return self.exit_code == 0 and not self.timed_out and not self.cancelled

    def gate_outcome(self) -> GateOutcome:
        """Return this result as a :func:`gated_loop` gate verdict."""
        return GateOutcome(
            passed=self.passed,
            detail={
                "step_id": self.step_id,
                "exit_code": self.exit_code,
                "timed_out": self.timed_out,
                "cancelled": self.cancelled,
            },
        )
