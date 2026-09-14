"""Bounded workflow diagnostics; settlement remains owned by Runtime recovery."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from cayu.deadlines import ExecutionDeadline, ExecutionDeadlineExceeded


class _FailureEvidenceFields(BaseModel):
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
    # Native session IDs allow up to 2048 UTF-8 bytes; never truncate identity.
    session_id: str | None = Field(default=None, max_length=2048)
    run_epoch: int | None = Field(default=None, ge=0)
    terminal_event_id: str | None = Field(default=None, max_length=512)
    settlement: Literal["unknown"] = "unknown"

    @model_validator(mode="after")
    def validate_deadline_evidence(self) -> _FailureEvidenceFields:
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


class FailureEvidence(_FailureEvidenceFields):
    """Bounded failure snapshot with flat, ordered parallel branch evidence.

    Branches carry the same diagnostic fields but cannot recursively contain
    branches. Nested fan-outs are flattened in submission order, up to 16 leaves.
    Settlement is always unknown, including when a deadline is retained.
    """

    branch_failures: tuple[_FailureEvidenceFields, ...] = Field(
        default=(), max_length=16, exclude_if=lambda value: not value
    )


@dataclass(frozen=True)
class _ChildFailureIdentity:
    # Process-local diagnostic provenance, never continuation authority.
    evidence: FailureEvidence


def retain_child_failure_identity(exc: BaseException, evidence: FailureEvidence) -> None:
    """Carry a workflow's observed child through cancellation/timeout wrapping."""
    if evidence.session_id is not None or evidence.secondary_failures:
        exc.__dict__["_cayu_child_failure_identity"] = _ChildFailureIdentity(evidence)


def exception_evidence(exc: BaseException) -> FailureEvidence:
    """Inspect only bounded type identifiers and Runtime deadline metadata."""
    # Import locally: workflow models themselves depend on this contract.
    from cayu.workflows.models import ParallelStepError, StepError

    pending = [exc]
    snapshots: list[FailureEvidence] = []
    branches: list[_FailureEvidenceFields] = []
    parallel = False
    ordinary_failure = False
    seen: set[int] = set()
    names: list[str] = []
    deadline = None
    phase = None
    interrupted = False
    timeout = False
    truncated = False
    secondary = False
    child_identities: dict[tuple[str, int | None, str | None], FailureEvidence] = {}
    while pending and len(seen) < 16:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        name = type(item).__name__
        names.append(name[:128] if name.isascii() and name.isidentifier() else "Exception")
        if isinstance(item, StepError):
            if item._evidence is not None:
                snapshots.append(item._evidence)
                if item._evidence.branch_failures:
                    parallel = True
                    available = 16 - len(branches)
                    truncated |= len(item._evidence.branch_failures) > available
                    branches.extend(item._evidence.branch_failures[:available])
            elif item.__cause__ is None and item.__context__ is None:
                snapshots.append(FailureEvidence())
        if isinstance(item, ParallelStepError):
            parallel = True
            truncated |= len(item.failures) > 16
            for failure in item.failures[:16]:
                snapshot = failure.evidence
                secondary |= snapshot.secondary_failures
                truncated |= snapshot.truncated
                leaves = snapshot.branch_failures or (
                    _FailureEvidenceFields.model_validate(
                        snapshot.model_dump(exclude={"branch_failures"})
                    ),
                )
                available = 16 - len(branches)
                truncated |= len(leaves) > available
                branches.extend(leaves[:available])
        ordinary_failure |= not isinstance(
            item,
            (
                StepError,
                ParallelStepError,
                BaseExceptionGroup,
                TimeoutError,
                asyncio.CancelledError,
            ),
        )
        retained = item.__dict__.get("_cayu_child_failure_identity")
        if type(retained) is _ChildFailureIdentity:
            child = retained.evidence
            secondary |= child.secondary_failures
            if child.session_id is not None:
                child_identities[(child.session_id, child.run_epoch, child.terminal_event_id)] = (
                    child
                )
        interrupted |= isinstance(item, asyncio.CancelledError)
        timeout |= isinstance(item, TimeoutError)
        if isinstance(item, asyncio.CancelledError):
            from cayu.providers._credential_boundary import (
                provider_cancellation_admission_deadline,
                provider_cancellation_failures,
            )

            secondary |= bool(provider_cancellation_failures(item))
            native_admission = provider_cancellation_admission_deadline(item)
            if deadline is None and native_admission is not None:
                deadline, phase = native_admission, "admission"
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
    for snapshot in snapshots:
        secondary |= snapshot.secondary_failures
        truncated |= snapshot.truncated
        for name in snapshot.exception_types:
            if name not in names:
                if len(names) < 16:
                    names.append(name)
                else:
                    truncated = True
        if snapshot.session_id is not None:
            child_identities[
                (snapshot.session_id, snapshot.run_epoch, snapshot.terminal_event_id)
            ] = snapshot
    # A group can contain multiple children. Never select a sibling's identity
    # or a partial traversal as the sole terminal reference for the group.
    child = (
        next(iter(child_identities.values()))
        if len(child_identities) == 1 and not parallel and not pending and not truncated
        else None
    )
    typed = snapshots[0] if len(snapshots) == 1 and not parallel else None
    if typed is not None or parallel:
        secondary |= ordinary_failure
    if typed is not None and len(child_identities) == 1 and not pending:
        child = next(iter(child_identities.values()))
    if typed is not None and deadline is None:
        deadline, phase = typed.deadline, typed.deadline_phase
    return FailureEvidence(
        branch_failures=tuple(branches),
        session_id=child.session_id if child is not None else None,
        run_epoch=child.run_epoch if child is not None else None,
        terminal_event_id=child.terminal_event_id if child is not None else None,
        classification="deadline"
        if deadline
        else "timeout"
        if timeout
        else "interruption"
        if interrupted
        else typed.classification
        if typed is not None
        else "failure",
        deadline=deadline,
        deadline_phase=phase,
        exception_types=tuple(names),
        truncated=bool(pending) or truncated,
        secondary_failures=secondary
        or (
            deadline is not None
            and typed is None
            and not isinstance(exc, (TimeoutError, asyncio.CancelledError))
        ),
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
