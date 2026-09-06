"""Bounded workflow diagnostics; settlement remains owned by Runtime recovery."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from cayu.deadlines import ExecutionDeadline, ExecutionDeadlineExceeded


class FailureEvidence(BaseModel):
    """Portable evidence, never a certificate that external effects have stopped.

    ``session_id``/``run_epoch``/``terminal_event_id`` reference Runtime session
    events and the existing IncompleteSessionRecoveryResult contract. Missing
    evidence is unknown, including for old durable events.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    classification: Literal["deadline", "timeout", "interruption", "failure", "unknown"] = "unknown"
    deadline: ExecutionDeadline | None = None
    deadline_phase: Literal["admission", "in_flight"] | None = None
    exception_types: tuple[
        Annotated[str, Field(max_length=128, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")], ...
    ] = Field(default=(), max_length=16)
    truncated: bool = False
    secondary_failures: bool = False
    session_id: str | None = Field(default=None, max_length=512)
    run_epoch: int | None = Field(default=None, ge=0)
    terminal_event_id: str | None = Field(default=None, max_length=512)
    settlement: Literal["unknown"] = "unknown"

    @model_validator(mode="after")
    def validate_deadline_evidence(self) -> FailureEvidence:
        if self.classification == "deadline":
            if (
                self.deadline is None
                or self.deadline.expires_at is None
                or self.deadline_phase is None
            ):
                raise ValueError("Deadline classification requires an expiry and phase.")
        elif self.deadline is not None or self.deadline_phase is not None:
            raise ValueError("Deadline evidence requires deadline classification.")
        return self


def exception_evidence(exc: BaseException) -> FailureEvidence:
    """Inspect only bounded type identifiers and Runtime deadline metadata."""
    pending = [exc]
    seen: set[int] = set()
    names: list[str] = []
    deadline = None
    phase = None
    interrupted = False
    timeout = False
    truncated = False
    secondary = False
    while pending and len(seen) < 16:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        name = type(item).__name__
        names.append(name[:128] if name.isascii() and name.isidentifier() else "Exception")
        interrupted |= isinstance(item, asyncio.CancelledError)
        timeout |= isinstance(item, TimeoutError)
        raw = getattr(item, "execution_deadline", None)
        item_phase = "in_flight"
        if raw is None and isinstance(item, ExecutionDeadlineExceeded):
            raw = item.diagnostics
            item_phase = (
                "in_flight" if raw.get("denied_stage") == "execution_completion" else "admission"
            )
        if deadline is None and isinstance(raw, dict):
            try:
                candidate = ExecutionDeadline.model_validate(
                    {key: raw[key] for key in ("expires_at", "source", "scope")}
                )
                if candidate.expires_at is not None:
                    deadline, phase = candidate, item_phase
            except (KeyError, ValidationError):
                pass
        if isinstance(item, BaseExceptionGroup):
            truncated |= len(item.exceptions) > 16
            secondary = True
            pending.extend(reversed(item.exceptions[:16]))
        cause = item.__cause__ or item.__context__
        if cause is not None:
            pending.append(cause)
    return FailureEvidence(
        classification="deadline"
        if deadline
        else "timeout"
        if timeout
        else "interruption"
        if interrupted
        else "failure",
        deadline=deadline,
        deadline_phase=phase,
        exception_types=tuple(names),
        truncated=bool(pending) or truncated,
        secondary_failures=secondary
        or (deadline is not None and not isinstance(exc, (TimeoutError, asyncio.CancelledError))),
    )


def event_failure_evidence(
    payload: dict, *, session_id: str, event_id: str, interrupted: bool
) -> FailureEvidence:
    raw = payload.get("failure_evidence")
    if isinstance(raw, dict):
        try:
            evidence = FailureEvidence.model_validate(raw)
            if evidence.session_id == session_id:
                return evidence.model_copy(update={"terminal_event_id": event_id})
        except ValidationError:
            pass
    return FailureEvidence(
        classification="interruption" if interrupted else "unknown",
        session_id=session_id,
        terminal_event_id=event_id,
    )
