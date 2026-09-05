"""Read-only capture and deterministic scoring of an explicitly selected saved attempt.

No factory, projector, workflow, tool, or judge callback is invoked by this module.
The first capture seals the descendant evidence visible in the saved store; subsequent
captures can require that seal. It cannot attest to bytes deleted before the first seal.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal, Never
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from cayu._validation import canonical_durable_json_bytes
from cayu.core.messages import Message
from cayu.evals.capture_policy import SessionTrajectoryBounds, WorkflowAttemptAnchor
from cayu.evals.corpus import AssertionSpec, assertion_spec_revision, pricing_profile_identity
from cayu.evals.execution import WorkflowEvalTarget, _copy_corpus_target
from cayu.evals.models import EvalAssertionResult, EvalRun, EvalTrialResult, Trajectory
from cayu.evals.portable_assertions import compile_assertion_spec
from cayu.evals.runner import (
    _current_workflow_completion,
    _load_workflow_eval_records,
    _workflow_root_sha256,
    _workflow_structured_sha256,
    evaluate_assertions,
)
from cayu.evals.trajectory import (
    SessionTrajectoryError,
    SessionTrajectoryErrorCode,
    _build_child_trajectories,
    _CaptureState,
    _IncompleteFlag,
    _revalidate_fresh_capture,
    _workflow_trajectory_from_session,
)
from cayu.evals.workflow_target import (
    WorkflowEvalOutputEvidenceV1,
    WorkflowEvalResult,
    workflow_eval_input_messages_sha256,
    workflow_eval_output_sha256,
    workflow_eval_trial_session_id,
)
from cayu.runtime.sessions import SessionStatus
from cayu.workflows.journal import WORKFLOW_ATTEMPT_EVENT_TYPE


class SavedWorkflowEvalCapture(BaseModel):
    """An independently inspectable capture revision linked to the original report trial."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    capture_id: str = Field(default_factory=lambda: str(uuid4()))
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_trial: EvalTrialResult
    source_attempt: WorkflowAttemptAnchor
    bounds: SessionTrajectoryBounds
    evidence_sha256: str
    trajectory: Trajectory


class SavedWorkflowEvalScore(BaseModel):
    """Separate deterministic scoring revision; source execution cost is not incurred again."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    score_id: str = Field(default_factory=lambda: str(uuid4()))
    scored_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_capture_id: str
    target_revision: str
    evidence_policy_revision: str
    pricing_profile_fingerprint: str | None = None
    source_evidence_sha256: str
    assertion_revisions: tuple[str, ...]
    assertions: tuple[EvalAssertionResult, ...]
    score: float | None
    model_calls: Literal[0] = 0


def _reject(anchor: WorkflowAttemptAnchor) -> Never:
    raise SessionTrajectoryError(
        SessionTrajectoryErrorCode.EVIDENCE_INCONSISTENT, session_id=anchor.session_id
    )


async def import_workflow_eval_attempt(
    target: WorkflowEvalTarget,
    source_run: EvalRun,
    *,
    case_id: str,
    trial_number: int,
    attempt_id: str,
    completion_event_id: str,
    messages: tuple[Message, ...],
    output: WorkflowEvalResult,
) -> EvalTrialResult:
    """Explicitly adopt a legacy report's saved completion, without rewriting that report.

    The caller supplies the original target/input and an explicitly projected result,
    plus the exact persisted attempt and completion IDs. This import establishes an
    anchor now, not a claim of an earlier full-evidence seal. Keep source_run unchanged.
    """

    if type(target) is not WorkflowEvalTarget:
        raise TypeError("target must be an exact WorkflowEvalTarget")
    copied_target = _copy_corpus_target(target)
    assert isinstance(copied_target, WorkflowEvalTarget)
    target = copied_target
    source_run = EvalRun.model_validate(source_run.model_dump(mode="python"))
    output = WorkflowEvalResult.model_validate(output.model_dump(mode="python"))
    matches = [
        trial
        for case in source_run.cases
        if case.case_id == case_id
        for trial in case.trials
        if trial.trial_number == trial_number
    ]
    if len(matches) != 1:
        raise ValueError("The original report must contain exactly the selected trial.")
    trial = matches[0]
    if trial.workflow_attempt is not None:
        raise ValueError("An existing attempt anchor must be used, not replaced by an import.")
    identity = target.identity()
    root_id = workflow_eval_trial_session_id(
        target_revision=identity.revision,
        run_id=source_run.run_id,
        suite_id=source_run.suite_id,
        case_id=case_id,
        trial_number=trial_number,
    )
    if trial.session_id != root_id:
        raise ValueError("The target and report identities do not select this workflow root.")
    session = await target.app.session_store.load(root_id)
    records = await _load_workflow_eval_records(
        target.app,
        session_id=root_id,
        workflow_name=target.workflow_spec.name,
    )
    selected_attempt, completion = _current_workflow_completion(
        records, workflow_name=target.workflow_spec.name
    )
    if (
        session is None
        or session.status is not SessionStatus.COMPLETED
        or selected_attempt != attempt_id
        or completion.event.id != completion_event_id
    ):
        raise ValueError("The exact saved workflow completion is missing or changed.")
    anchor = WorkflowAttemptAnchor(
        origin="saved_store_import",
        source_report_sha256=hashlib.sha256(
            canonical_durable_json_bytes(source_run.model_dump(mode="json"), "source report")
        ).hexdigest(),
        run_id=source_run.run_id,
        suite_id=source_run.suite_id,
        case_id=case_id,
        trial_number=trial_number,
        session_id=root_id,
        target_revision=identity.revision,
        projector_revision=identity.result_projector_revision,
        input_messages_sha256=workflow_eval_input_messages_sha256(messages),
        attempt_id=attempt_id,
        completion_event_id=completion_event_id,
        completion_sequence=completion.sequence,
        root_sha256=_workflow_root_sha256(session, records),
        final_output_sha256=workflow_eval_output_sha256(output.final_output),
        structured_output_sha256=_workflow_structured_sha256(output.structured_output),
    )
    return EvalTrialResult.model_validate(
        trial.model_dump(mode="python")
        | {
            "execution_status": "completed",
            "workflow_attempt": anchor,
        }
    )


async def capture_workflow_eval_attempt(
    target: WorkflowEvalTarget,
    source_trial: EvalTrialResult,
    *,
    messages: tuple[Message, ...],
    output: WorkflowEvalResult,
    bounds: SessionTrajectoryBounds,
    expected_evidence_sha256: str | None = None,
) -> SavedWorkflowEvalCapture:
    """Recapture an exact completed attempt without invoking application callbacks.

    Supply the original target (including its original capture policy), exact input,
    and original projected output. ``bounds`` controls only this new capture revision.
    To reject changes since a prior successful capture, supply its evidence_sha256.
    Persist the returned document separately from the original EvalRun.
    """

    if type(target) is not WorkflowEvalTarget:
        raise TypeError("target must be an exact WorkflowEvalTarget")
    # Reconstruct public models: model_copy/model_construct are not validation.
    copied_target = _copy_corpus_target(target)
    assert isinstance(copied_target, WorkflowEvalTarget)
    target = copied_target
    bounds = SessionTrajectoryBounds.model_validate(bounds.model_dump(mode="python"))
    if bounds.memory_attribution_bounds != SessionTrajectoryBounds().memory_attribution_bounds:
        raise ValueError(
            "Workflow capture bounds cannot override the separate Evals memory policy."
        )
    source_trial = EvalTrialResult.model_validate(source_trial.model_dump(mode="python"))
    output = WorkflowEvalResult.model_validate(output.model_dump(mode="python"))
    anchor = source_trial.workflow_attempt
    if anchor is None or source_trial.execution_status != "completed":
        raise ValueError("Recovery requires an original completed workflow attempt anchor.")
    identity = target.identity()
    if (
        source_trial.session_id != anchor.session_id
        or source_trial.trial_number != anchor.trial_number
        or identity.revision != anchor.target_revision
        or identity.result_projector_revision != anchor.projector_revision
        or workflow_eval_input_messages_sha256(messages) != anchor.input_messages_sha256
        or workflow_eval_output_sha256(output.final_output) != anchor.final_output_sha256
        or _workflow_structured_sha256(output.structured_output) != anchor.structured_output_sha256
        or workflow_eval_trial_session_id(
            target_revision=anchor.target_revision,
            run_id=anchor.run_id,
            suite_id=anchor.suite_id,
            case_id=anchor.case_id,
            trial_number=anchor.trial_number,
        )
        != anchor.session_id
    ):
        _reject(anchor)

    async def read_root():
        session = await target.app.session_store.load(anchor.session_id)
        records = await _load_workflow_eval_records(
            target.app, session_id=anchor.session_id, workflow_name=target.workflow_spec.name
        )
        attempt, completion = _current_workflow_completion(
            records, workflow_name=target.workflow_spec.name
        )
        if sum(record.event.type == WORKFLOW_ATTEMPT_EVENT_TYPE for record in records) != 1:
            _reject(anchor)
        if (
            session is None
            or session.status is not SessionStatus.COMPLETED
            or attempt != anchor.attempt_id
            or completion.event.id != anchor.completion_event_id
            or completion.sequence != anchor.completion_sequence
            or _workflow_root_sha256(session, records) != anchor.root_sha256
        ):
            _reject(anchor)
        return session, records

    session, records = await read_root()
    state = _CaptureState(bounds=bounds, strict=False, fail_closed=True)
    incomplete = _IncompleteFlag()
    children = await _build_child_trajectories(
        target.app,
        anchor.session_id,
        visited={anchor.session_id},
        incomplete=incomplete,
        parent_terminal_sequence=anchor.completion_sequence,
        state=state,
    )
    if incomplete.value:
        _reject(anchor)
    attempt_start = next(
        record.sequence for record in records if record.event.type == WORKFLOW_ATTEMPT_EVENT_TYPE
    )
    for evidence in state.evidence_by_session_id.values():
        if (
            evidence.boundary.first_event_sequence <= attempt_start
            or evidence.boundary.terminal_event_sequence >= anchor.completion_sequence
        ):
            _reject(anchor)
    await _revalidate_fresh_capture(
        target.app, state, root_session_id=anchor.session_id, root_interrupted_observed_events=()
    )
    await read_root()
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(anchor.root_sha256))
    for session_id, evidence in sorted(state.evidence_by_session_id.items()):
        digest.update(
            canonical_durable_json_bytes(
                {
                    "session_id": session_id,
                    "evidence": evidence.model_dump(mode="json"),
                },
                "saved workflow evidence",
            )
        )
    evidence_sha256 = digest.hexdigest()
    if expected_evidence_sha256 is not None and evidence_sha256 != expected_evidence_sha256:
        _reject(anchor)
    trajectory = _workflow_trajectory_from_session(
        session,
        workflow_events=tuple(record.event for record in records),
        input_messages=messages,
        output_evidence=WorkflowEvalOutputEvidenceV1(
            target_revision=anchor.target_revision,
            projector_revision=anchor.projector_revision,
            workflow_name=target.workflow_spec.name,
            attempt_id=anchor.attempt_id,
            completion_event_id=anchor.completion_event_id,
            input_message_count=len(messages),
            input_messages_sha256=anchor.input_messages_sha256,
            final_output_sha256=anchor.final_output_sha256,
            structured_output=output.structured_output,
        ),
        final_output=output.final_output,
        children=children,
        children_incomplete=False,
        metadata={},
    )
    return SavedWorkflowEvalCapture(
        source_trial=source_trial,
        source_attempt=anchor,
        bounds=bounds,
        evidence_sha256=evidence_sha256,
        trajectory=trajectory,
    )


async def score_workflow_eval_capture(
    target: WorkflowEvalTarget,
    capture: SavedWorkflowEvalCapture,
    assertions: tuple[AssertionSpec, ...],
) -> SavedWorkflowEvalScore:
    """Freshly validate a sealed attempt, then run deterministic portable assertions only.

    Model judge specs are rejected before store reads or scoring. Environment probes
    absent from the saved evidence remain unavailable; no tools run to recreate them.
    """

    compiled = tuple(
        compile_assertion_spec(
            spec,
            app=target.app,
            evidence_policy=target.evidence_policy,
            trusted_pricing=target.price_book,
        )
        for spec in assertions
    )
    if not compiled:
        raise ValueError("Scoring requires at least one deterministic assertion.")
    capture = SavedWorkflowEvalCapture.model_validate(capture.model_dump(mode="python"))
    trajectory = capture.trajectory
    if capture.source_attempt != capture.source_trial.workflow_attempt:
        _reject(capture.source_attempt)
    if trajectory.workflow_output is None:
        _reject(capture.source_attempt)
    fresh = await capture_workflow_eval_attempt(
        target,
        capture.source_trial,
        messages=tuple(trajectory.transcript),
        output=WorkflowEvalResult(
            final_output=trajectory.final_output,
            structured_output=trajectory.workflow_output.structured_output,
        ),
        bounds=capture.bounds,
        expected_evidence_sha256=capture.evidence_sha256,
    )
    results = await evaluate_assertions(
        fresh.trajectory,
        compiled,
        suite_id=capture.source_attempt.suite_id,
        case_id=capture.source_attempt.case_id,
    )
    scores = [result.score for result in results]
    score = None if any(value is None for value in scores) else sum(scores) / len(scores)
    return SavedWorkflowEvalScore(
        source_capture_id=capture.capture_id,
        target_revision=target.identity().revision,
        evidence_policy_revision=target.evidence_policy.revision,
        pricing_profile_fingerprint=None
        if target.price_book is None
        else pricing_profile_identity(target.price_book).fingerprint,
        source_evidence_sha256=capture.evidence_sha256,
        assertion_revisions=tuple(assertion_spec_revision(spec) for spec in assertions),
        assertions=results,
        score=score,
    )
