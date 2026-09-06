"""Bounded incremental capture and deterministic scoring of saved workflow attempts.

The returned document is a verified summary, never a truncated Trajectory. Source
payloads exist only in one private, owned spill at a time and are removed before
publication. Revalidation repeats the exact closure and compares content seals.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import Counter
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from cayu._validation import canonical_durable_json_bytes
from cayu.core.messages import Message
from cayu.evals.capture_policy import SessionTrajectoryBounds, WorkflowAttemptAnchor
from cayu.evals.corpus import (
    AssertionSpec,
    ChildStatusAssertionSpec,
    FinalOutputContainsAssertionSpec,
    FinalOutputEqualsAssertionSpec,
    ProcessEventAssertionSpec,
    RootStatusAssertionSpec,
    _validated_assertion_spec,
    assertion_spec_revision,
)
from cayu.evals.evidence import _PORTABLE_PROCESS_EVENT_KINDS, _redacted_text
from cayu.evals.execution import WorkflowEvalTarget
from cayu.evals.models import EvalAssertionResult, EvalOutcome, EvalTrialResult
from cayu.evals.portable_evaluation import (
    _evaluate_child_status,
    _evaluate_final_output,
    _evaluate_root_status,
    _result,
)
from cayu.evals.trajectory import (
    _ORIGIN_EVENT_TYPES,
    SessionTrajectoryError,
    _CaptureState,
    _child_origin,
    _strict_child_nodes,
    _validate_terminal_origin_lineage,
)
from cayu.evals.workflow_recovery import (
    _prepare_workflow_attempt_capture,
    _read_workflow_attempt_root,
)
from cayu.evals.workflow_target import WorkflowEvalFailure, WorkflowEvalResult
from cayu.runtime.evidence_spool import (
    EvidenceSpool,
    IncrementalEvidenceAdmission,
    IncrementalEvidenceError,
    IncrementalEvidenceLimits,
    _settled_evidence_reads,
)
from cayu.runtime.sessions import (
    EventQueryResultTooLarge,
    SessionStatus,
    TerminalSessionEvidenceError,
)
from cayu.workflows.journal import WORKFLOW_ATTEMPT_EVENT_TYPE


class IncrementalCaptureProgress(BaseModel):
    """Payload-free progress, inspectable without source I/O after expiry."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    stage: str
    session_id: str | None = None
    processed_events: StrictInt = Field(default=0, ge=0)
    processed_transcript_records: StrictInt = Field(default=0, ge=0)
    processed_bytes: StrictInt = Field(default=0, ge=0)
    peak_record_bytes: StrictInt = Field(default=0, ge=0)
    peak_spill_bytes: StrictInt = Field(default=0, ge=0)
    peak_page_transport_bytes: StrictInt = Field(default=0, ge=0)
    peak_page_records: StrictInt = Field(default=0, ge=0)
    spill_records_read: StrictInt = Field(default=0, ge=0)


class IncrementalWorkflowCaptureError(IncrementalEvidenceError):
    def __init__(
        self, code: str, *, progress: IncrementalCaptureProgress, limits: IncrementalEvidenceLimits
    ) -> None:
        super().__init__(code)
        self.progress = progress
        self.limits = limits


class IncrementalSessionSeal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    session_id: str
    parent_session_id: str
    status: Literal["completed", "failed"]
    first_event_sequence: int
    terminal_event_sequence: int
    evidence_sha256: str
    event_count: int
    transcript_count: int
    evidence_bytes: int


class SavedIncrementalWorkflowCapture(BaseModel):
    """Independent capture revision with full-scope verification and summary retention."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    capture_id: str = Field(default_factory=lambda: str(uuid4()))
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_attempt: WorkflowAttemptAnchor
    limits: IncrementalEvidenceLimits
    evidence_digest_scheme: Literal["cayu.incremental-workflow-evidence.v1"] = (
        "cayu.incremental-workflow-evidence.v1"
    )
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_policy_revision: str
    evidence_complete: Literal[True] = True
    retained_evidence: Literal["summary_only"] = "summary_only"
    sessions: tuple[IncrementalSessionSeal, ...] = Field(max_length=500)
    process_event_counts: dict[str, int]
    progress: IncrementalCaptureProgress
    reserved_buffer_bytes: int
    model_calls: Literal[0] = 0

    @model_validator(mode="after")
    def validate_policy_revision(self) -> SavedIncrementalWorkflowCapture:
        if self.capture_policy_revision != _policy_revision(self.limits):
            raise ValueError("Incremental capture policy revision does not match its limits.")
        return self


class SavedIncrementalWorkflowScore(BaseModel):
    """Scoring revision linked to a separately preserved capture revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    score_id: str = Field(default_factory=lambda: str(uuid4()))
    scored_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_capture_id: str
    revalidation: SavedIncrementalWorkflowCapture
    source_evidence_sha256: str
    target_revision: str
    evidence_policy_revision: str
    scoring_semantics: Literal["cayu.incremental-assertions.v1"] = "cayu.incremental-assertions.v1"
    assertion_revisions: tuple[str, ...]
    assertions: tuple[EvalAssertionResult, ...]
    score: float | None
    model_calls: Literal[0] = 0


def _policy_revision(limits: IncrementalEvidenceLimits) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            canonical_durable_json_bytes(
                {
                    "semantics": "cayu.incremental-capture.v1",
                    "limits": limits.model_dump(mode="json"),
                },
                "incremental capture policy",
            )
        ).hexdigest()
    )


def _workflow_digest(anchor: WorkflowAttemptAnchor, sessions: list[IncrementalSessionSeal]) -> str:
    digest = hashlib.sha256(b"cayu.incremental-workflow-evidence.v1\0")
    digest.update(bytes.fromhex(anchor.root_sha256))
    for seal in sessions:
        digest.update(canonical_durable_json_bytes(seal.model_dump(mode="json"), "session seal"))
    return digest.hexdigest()


async def capture_incremental_workflow_eval_attempt(
    target: WorkflowEvalTarget,
    source_trial: EvalTrialResult,
    *,
    messages: tuple[Message, ...],
    output: WorkflowEvalResult,
    limits: IncrementalEvidenceLimits,
    admission: IncrementalEvidenceAdmission,
    expected_evidence_sha256: str | None = None,
) -> SavedIncrementalWorkflowCapture:
    """Capture and revalidate one saved attempt without providers, tools or callbacks.

    ``admission`` must be shared across concurrent callers. Limits are aggregate
    across the selected descendants; two full reads are charged as processing work.
    Root journal reads retain their separate existing finite bound. The first seal
    cannot establish the contents of records deleted before capture began.
    """
    target, source_trial, output, anchor = _prepare_workflow_attempt_capture(
        target,
        source_trial,
        messages,
        output,
        omit_retained_trajectory=True,
    )
    limits = IncrementalEvidenceLimits.model_validate(limits.model_dump())
    if expected_evidence_sha256 is not None and (
        len(expected_evidence_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_evidence_sha256)
    ):
        raise ValueError("Expected evidence digest must be lowercase SHA-256 hex.")
    progress = IncrementalCaptureProgress(stage="admission", session_id=anchor.session_id)
    try:
        reservation = admission.reserve(limits)
    except IncrementalEvidenceError as exc:
        raise IncrementalWorkflowCaptureError(exc.code, progress=progress, limits=limits) from None

    active_spool: EvidenceSpool | None = None
    try:
        with _settled_evidence_reads():
            async with asyncio.timeout(limits.max_seconds):
                if not target.app.session_store.supports_incremental_terminal_evidence:
                    raise IncrementalEvidenceError("incremental_store_unsupported")
                if not target.app.session_store.supports_session_lineage:
                    raise IncrementalEvidenceError("lineage_capability_unsupported")
                progress = progress.model_copy(update={"stage": "root_read"})
                _, root_records = await _read_workflow_attempt_root(
                    target,
                    anchor,
                    root_max_bytes=limits.max_root_bytes,
                    max_record_bytes=limits.max_record_bytes,
                    root_max_events=limits.max_root_events,
                )
                attempt_start = next(
                    record.sequence
                    for record in root_records
                    if record.event.type == WORKFLOW_ATTEMPT_EVENT_TYPE
                )
                process_counts: Counter[str] = Counter()
                for record in root_records:
                    kind = _PORTABLE_PROCESS_EVENT_KINDS.get(str(record.event.type))
                    if kind is not None:
                        process_counts[kind] += 1

                async def read_pass(*, revalidation: bool) -> list[IncrementalSessionSeal]:
                    nonlocal progress, active_spool
                    state = _CaptureState(
                        bounds=SessionTrajectoryBounds(
                            max_sessions=limits.max_sessions,
                            max_depth=limits.max_depth,
                        ),
                        strict=True,
                    )
                    selected: list[IncrementalSessionSeal] = []
                    visited = {anchor.session_id}
                    events = transcript = evidence_bytes = 0

                    async def visit(parent_id: str, terminal_sequence: int, depth: int) -> None:
                        nonlocal progress, active_spool, events, transcript, evidence_bytes
                        progress = progress.model_copy(
                            update={
                                "stage": "lineage_revalidation" if revalidation else "lineage",
                                "session_id": parent_id,
                            }
                        )
                        candidates = await _strict_child_nodes(target.app, parent_id, state=state)
                        origins = sorted(
                            (_child_origin(node) for node in candidates),
                            key=lambda item: (item.origin.sequence, item.node.id),
                        )
                        for origin in origins:
                            if origin.origin.sequence > terminal_sequence:
                                continue
                            session_id = origin.node.id
                            if session_id in visited:
                                raise IncrementalEvidenceError("cycle_detected")
                            if depth + 1 > limits.max_depth:
                                raise IncrementalEvidenceError("depth_limit_exceeded")
                            if len(selected) >= limits.max_sessions:
                                raise IncrementalEvidenceError("session_limit_exceeded")
                            visited.add(session_id)
                            if (
                                events >= limits.max_events
                                or evidence_bytes >= limits.max_total_bytes
                            ):
                                raise IncrementalEvidenceError("aggregate_work_limit_exceeded")
                            remaining = limits.model_copy(
                                update={
                                    "max_events": limits.max_events - events,
                                    "max_transcript_records": limits.max_transcript_records
                                    - transcript,
                                    "max_total_bytes": limits.max_total_bytes - evidence_bytes,
                                }
                            )
                            progress = progress.model_copy(
                                update={
                                    "stage": "snapshot_revalidation"
                                    if revalidation
                                    else "snapshot_copy",
                                    "session_id": session_id,
                                }
                            )
                            with EvidenceSpool(remaining) as spool:
                                active_spool = spool
                                try:
                                    await target.app.session_store.export_terminal_session_evidence(
                                        session_id,
                                        spool=spool,
                                    )
                                except NotImplementedError:
                                    raise IncrementalEvidenceError(
                                        "incremental_store_unsupported"
                                    ) from None
                                await spool.run_owned(
                                    asyncio.to_thread(
                                        _validate_terminal_origin_lineage,
                                        spool,
                                    )
                                )
                                # There is exactly one origin after the shared validator.
                                first_origin = next(
                                    record
                                    for record in spool.events
                                    if record.event.type in _ORIGIN_EVENT_TYPES
                                )
                                if (
                                    spool.session.id != session_id
                                    or spool.session.parent_session_id != parent_id
                                    or spool.session.created_at != origin.node.created_at
                                    or first_origin.sequence != origin.origin.sequence
                                    or first_origin.event.id != origin.origin.event_id
                                    or first_origin.event.type != origin.origin.event_type
                                    or spool.boundary.first_event_sequence <= attempt_start
                                    or spool.boundary.terminal_event_sequence
                                    >= anchor.completion_sequence
                                    or spool.boundary.terminal_event_sequence > terminal_sequence
                                ):
                                    raise IncrementalEvidenceError("attempt_or_lineage_mismatch")
                                if not revalidation:

                                    def count_process_events() -> None:
                                        for record in spool.events:
                                            kind = _PORTABLE_PROCESS_EVENT_KINDS.get(
                                                str(record.event.type)
                                            )
                                            if kind is not None:
                                                process_counts[kind] += 1

                                    await spool.run_owned(asyncio.to_thread(count_process_events))
                                boundary = spool.boundary
                                seal = IncrementalSessionSeal(
                                    session_id=session_id,
                                    parent_session_id=parent_id,
                                    status="completed"
                                    if spool.session.status is SessionStatus.COMPLETED
                                    else "failed",
                                    first_event_sequence=boundary.first_event_sequence,
                                    terminal_event_sequence=boundary.terminal_event_sequence,
                                    evidence_sha256=spool.evidence_sha256,
                                    event_count=boundary.event_count,
                                    transcript_count=boundary.transcript_count,
                                    evidence_bytes=boundary.total_bytes,
                                )
                                events += boundary.event_count
                                transcript += boundary.transcript_count
                                evidence_bytes += boundary.total_bytes
                                progress = progress.model_copy(
                                    update={
                                        "processed_events": progress.processed_events
                                        + boundary.event_count,
                                        "processed_transcript_records": progress.processed_transcript_records
                                        + boundary.transcript_count,
                                        "processed_bytes": progress.processed_bytes
                                        + boundary.total_bytes,
                                        "peak_record_bytes": max(
                                            progress.peak_record_bytes, spool.peak_record_bytes
                                        ),
                                        "peak_page_transport_bytes": max(
                                            progress.peak_page_transport_bytes,
                                            spool.peak_page_transport_bytes,
                                        ),
                                        "peak_page_records": max(
                                            progress.peak_page_records, spool.peak_page_records
                                        ),
                                        "spill_records_read": progress.spill_records_read
                                        + spool.spill_records_read,
                                        "peak_spill_bytes": max(
                                            progress.peak_spill_bytes, spool.path.stat().st_size
                                        ),
                                    }
                                )
                                active_spool = None
                            selected.append(seal)
                            await visit(session_id, seal.terminal_event_sequence, depth + 1)

                    await visit(anchor.session_id, anchor.completion_sequence, 1)
                    return selected

                selected = await read_pass(revalidation=False)
                if await read_pass(revalidation=True) != selected:
                    raise IncrementalEvidenceError("closure_changed")
                progress = progress.model_copy(
                    update={"stage": "root_revalidation", "session_id": anchor.session_id}
                )
                await _read_workflow_attempt_root(
                    target,
                    anchor,
                    root_max_bytes=limits.max_root_bytes,
                    max_record_bytes=limits.max_record_bytes,
                    root_max_events=limits.max_root_events,
                )
                digest = _workflow_digest(anchor, selected)
                if expected_evidence_sha256 is not None and digest != expected_evidence_sha256:
                    raise IncrementalEvidenceError("evidence_changed")
                return SavedIncrementalWorkflowCapture(
                    source_attempt=anchor,
                    limits=limits,
                    evidence_sha256=digest,
                    capture_policy_revision=_policy_revision(limits),
                    sessions=tuple(selected),
                    process_event_counts=dict(process_counts),
                    progress=progress.model_copy(update={"stage": "complete"}),
                    reserved_buffer_bytes=admission.buffer_envelope(limits),
                )
    except Exception as exc:
        if active_spool is not None:
            progress = progress.model_copy(
                update={
                    "stage": active_spool.phase,
                    "processed_events": progress.processed_events + len(active_spool.events),
                    "processed_transcript_records": progress.processed_transcript_records
                    + len(active_spool.transcript),
                    "processed_bytes": progress.processed_bytes + active_spool.processed_bytes,
                    "peak_record_bytes": max(
                        progress.peak_record_bytes, active_spool.peak_record_bytes
                    ),
                }
            )
        if isinstance(exc, TimeoutError):
            code = "deadline_exceeded"
        elif isinstance(exc, NotImplementedError):
            code = "incremental_store_unsupported"
        elif isinstance(exc, EventQueryResultTooLarge):
            code = "root_bytes_exceeded"
        elif isinstance(
            exc,
            (
                IncrementalEvidenceError,
                TerminalSessionEvidenceError,
                SessionTrajectoryError,
                WorkflowEvalFailure,
            ),
        ):
            code = str(exc.code)
        else:
            code = "evidence_read_failed"
        raise IncrementalWorkflowCaptureError(code, progress=progress, limits=limits) from None
    finally:
        reservation.close()


_SUPPORTED_ASSERTIONS = frozenset(
    {
        FinalOutputEqualsAssertionSpec,
        FinalOutputContainsAssertionSpec,
        RootStatusAssertionSpec,
        ChildStatusAssertionSpec,
        ProcessEventAssertionSpec,
    }
)


async def score_incremental_workflow_eval_capture(
    target: WorkflowEvalTarget,
    source_trial: EvalTrialResult,
    capture: SavedIncrementalWorkflowCapture,
    assertions: tuple[AssertionSpec, ...],
    *,
    messages: tuple[Message, ...],
    output: WorkflowEvalResult,
    admission: IncrementalEvidenceAdmission,
) -> SavedIncrementalWorkflowScore:
    """Freshly validate a capture seal, then apply closed incremental assertions.

    Unsupported full-trace, probe and model-backed assertions fail before source
    reads. The output projection retains its existing redaction and finite bound.
    Process-event counts examine the entire scope with constant-size counters;
    their semantics are versioned separately from eager display-event retention.
    """
    specs = tuple(_validated_assertion_spec(spec) for spec in assertions)
    if not specs or len(specs) > 100:
        raise ValueError("Incremental scoring requires 1..100 assertions.")
    if any(type(spec) not in _SUPPORTED_ASSERTIONS for spec in specs):
        raise IncrementalEvidenceError("assertion_requires_unsupported_evidence")
    capture = SavedIncrementalWorkflowCapture.model_validate(capture.model_dump(mode="python"))
    if capture.source_attempt != source_trial.workflow_attempt:
        raise IncrementalEvidenceError("source_attempt_mismatch")
    fresh = await capture_incremental_workflow_eval_attempt(
        target,
        source_trial,
        messages=messages,
        output=output,
        limits=capture.limits,
        admission=admission,
        expected_evidence_sha256=capture.evidence_sha256,
    )
    final_output = _redacted_text(target.app, output.final_output, "final output")
    output_state = (
        "complete"
        if len(final_output) <= target.evidence_policy.max_final_output_chars
        else "limit_exceeded"
    )
    statuses = tuple(
        seal.status
        for seal in fresh.sessions
        if seal.parent_session_id == capture.source_attempt.session_id
    )
    results: list[EvalAssertionResult] = []
    for spec in specs:
        if isinstance(spec, RootStatusAssertionSpec):
            result = _evaluate_root_status(name=spec.id, expected=spec.expected, actual="completed")
        elif isinstance(spec, ChildStatusAssertionSpec):
            result = _evaluate_child_status(
                name=spec.id,
                expected=spec.expected,
                statuses=statuses,
                state="complete",
                minimum=spec.min_count,
                maximum=spec.max_count,
            )
        elif isinstance(spec, FinalOutputEqualsAssertionSpec | FinalOutputContainsAssertionSpec):
            result = _evaluate_final_output(
                name=spec.id,
                expected=spec.expected,
                actual=final_output,
                state=output_state,
                contains=isinstance(spec, FinalOutputContainsAssertionSpec),
            )
        elif isinstance(spec, ProcessEventAssertionSpec):
            count = fresh.process_event_counts.get(spec.event, 0)
            passed = count >= spec.min_count and (spec.max_count is None or count <= spec.max_count)
            result = _result(
                spec.id,
                EvalOutcome.PASSED if passed else EvalOutcome.FAILED,
                f"Observed process event {spec.event} {count} time(s)."
                if passed
                else f"Process event {spec.event} count {count} is outside the required range.",
                metadata={
                    "event": spec.event,
                    "count": count,
                    "minimum": spec.min_count,
                    "maximum": spec.max_count,
                },
            )
        else:
            raise AssertionError("Unreachable unsupported assertion")
        results.append(
            result.model_copy(update={"assertion_revision": assertion_spec_revision(spec)})
        )
    scores = [result.score for result in results]
    score = None if any(value is None for value in scores) else sum(scores) / len(scores)
    return SavedIncrementalWorkflowScore(
        source_capture_id=capture.capture_id,
        revalidation=fresh,
        source_evidence_sha256=fresh.evidence_sha256,
        target_revision=capture.source_attempt.target_revision,
        evidence_policy_revision=target.evidence_policy.revision,
        assertion_revisions=tuple(assertion_spec_revision(spec) for spec in specs),
        assertions=tuple(results),
        score=score,
    )
