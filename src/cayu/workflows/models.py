"""Typed result objects for workflow helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StepResult:
    """Outcome of one successful ``step``."""

    step_id: str
    session_id: str
    text: str = ""
    output: Any = None

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
    ) -> None:
        super().__init__(message)
        self.message = message
        self.step_id = step_id
        self.session_id = session_id


class ParallelStepError(Exception):
    """Raised by the fail-closed ``ParallelResult`` accessors when branches failed."""

    def __init__(self, failures: Iterable[StepFailure]) -> None:
        self.failures: tuple[StepFailure, ...] = tuple(failures)
        summary = ", ".join(
            f"{failure.step_id or '?'}: {failure.error}" for failure in self.failures
        )
        super().__init__(f"{len(self.failures)} parallel step(s) failed: {summary}")


def normalize_gate_outcome(value: GateOutcome | bool) -> GateOutcome:
    """Accept either a :class:`GateOutcome` or a bare bool from a gate callable."""
    if isinstance(value, GateOutcome):
        return value
    if isinstance(value, bool):
        return GateOutcome(passed=value)
    raise TypeError("A gate must return a GateOutcome or a bool.")
