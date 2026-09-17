from __future__ import annotations

import asyncio
import hashlib

import pytest
from tests.evals.test_workflow_eval_target import (
    _NoChildWorkflow,
    _register_app,
    _suite,
    _target,
)

from cayu import (
    FinalOutputContains,
    SessionTrajectoryBounds,
    capture_workflow_eval_attempt,
    run_workflow_eval_suite,
    score_workflow_eval_capture,
)
from cayu._validation import (
    DURABLE_DOCUMENT_LIMITS,
    DurableValueError,
    canonical_durable_json_bytes,
)
from cayu.evals.corpus import FinalOutputEqualsAssertionSpec
from cayu.evals.incremental_recovery import (
    IncrementalWorkflowCaptureError,
    capture_incremental_workflow_eval_attempt,
)
from cayu.evals.models import EvalRun, EvalStatus
from cayu.evals.runner import _load_workflow_eval_records, _workflow_root_sha256
from cayu.evals.trajectory import SessionTrajectoryError
from cayu.events import Event
from cayu.runtime.evidence_spool import IncrementalEvidenceAdmission, IncrementalEvidenceLimits
from cayu.sessions.base import EventRecord
from cayu.workflows.base import WorkflowSpec
from cayu.workflows.workflow import WorkflowBase


async def _root(app, target, trial):
    session = await app.session_store.load(trial.session_id)
    records = await _load_workflow_eval_records(
        app, session_id=trial.session_id, workflow_name=target.workflow_spec.name
    )
    return session, records


@pytest.fixture
def root():
    async def create():
        app = _register_app()
        target = _target(app, _NoChildWorkflow)
        run = await run_workflow_eval_suite(target, _suite())
        return await _root(app, target, run.cases[0].trials[0])

    return asyncio.run(create())


def test_framed_root_identity_and_record_at_a_time_encoding(root, monkeypatch):
    import cayu._validation as validation

    session, records = root
    values = []

    def encode(value, field_name):
        values.append(value)
        return canonical_durable_json_bytes(value, field_name)

    monkeypatch.setattr(validation, "canonical_durable_json_bytes", encode)
    original = _workflow_root_sha256(session, records)
    assert values == [session.model_dump(mode="json")] + [
        record.model_dump(mode="json") for record in records
    ]
    expected = hashlib.sha256(b"cayu.workflow-attempt-root.framed-v2\0")
    for value in values:
        encoded = canonical_durable_json_bytes(value, "test")
        expected.update(len(encoded).to_bytes(8, "big") + encoded)
    assert original == expected.hexdigest()
    for changed_session, changed_records in (
        (session.model_copy(update={"id": "changed"}), records),
        (session.model_copy(update={"metadata": {"changed": True}}), records),
        (session, records[::-1]),
        (session, records[:-1]),
        (session, (*records, records[-1])),
        (session, (records[0].model_copy(update={"sequence": 100}), *records[1:])),
        (
            session,
            (
                records[0].model_copy(
                    update={"event": records[0].event.model_copy(update={"payload": {"x": 1}})}
                ),
                *records[1:],
            ),
        ),
    ):
        assert _workflow_root_sha256(changed_session, changed_records) != original
    with pytest.raises(ValueError, match="Unsupported"):
        _workflow_root_sha256(session, records, version="future")


def test_root_retains_canonical_numeric_semantics(root):
    session, _ = root
    event = Event(type="custom.test", session_id=session.id, payload={"value": 1.0})
    record = EventRecord(sequence=1, event=event)
    changed = record.model_copy(
        update={"event": event.model_copy(update={"payload": {"value": 1}})}
    )
    assert _workflow_root_sha256(session, (record,)) == _workflow_root_sha256(session, (changed,))


@pytest.mark.parametrize("payload", [{"value": float("nan")}, {"x": "x" * (3 << 20)}])
def test_invalid_individual_document_is_still_rejected(root, payload):
    session, records = root
    invalid = records[0].model_copy(
        update={"event": records[0].event.model_copy(update={"payload": payload})}
    )
    with pytest.raises(DurableValueError):
        _workflow_root_sha256(session, (invalid,))


class _LargeHistoryWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="large-history")

    async def run(self, session_id):
        ctx = self.context(session_id)
        yield await ctx.start()
        for index in range(18):
            yield await ctx.emit_custom_event(
                "custom.large", payload={"index": index, "text": "x" * (1 << 20)}
            )
        yield await ctx.completed({"answer": "done"})


def test_completed_large_history_evaluates_and_recovers(monkeypatch):
    async def exercise():
        app = _register_app()
        target = _target(app, _LargeHistoryWorkflow)
        suite = _suite(FinalOutputContains("done"))
        run = await run_workflow_eval_suite(target, suite)
        trial = run.cases[0].trials[0]
        assert trial.status is EvalStatus.PASSED, trial.error
        assert trial.execution_status == "completed"
        assert trial.workflow_attempt.root_hash_version == "framed-v2"
        session, records = await _root(app, target, trial)
        assert sum(len(r.model_dump_json()) for r in records) > DURABLE_DOCUMENT_LIMITS.max_bytes
        assert sum(r.event.type == "custom.large" for r in records) == 18
        with pytest.raises(DurableValueError, match="json_value_too_large"):
            _workflow_root_sha256(session, records, version="document-v1")
        assert _workflow_root_sha256(session, records) == trial.workflow_attempt.root_sha256
        restored = EvalRun.model_validate_json(run.model_dump_json()).cases[0].trials[0]

        def forbidden(*args, **kwargs):
            pytest.fail("Recovery dispatched application execution")

        monkeypatch.setattr(app, "run", forbidden)
        target = target.model_copy(
            update={"workflow_factory": forbidden, "result_projector": forbidden}
        )
        messages = tuple(suite.cases[0].request.messages)
        capture = await capture_workflow_eval_attempt(
            target, restored, messages=messages, bounds=SessionTrajectoryBounds()
        )
        specs = (FinalOutputEqualsAssertionSpec(id="answer", expected="done"),)
        score = await score_workflow_eval_capture(target, capture, specs)
        assert score.score == 1.0 and score.model_calls == 0
        # An explicit root byte budget still rejects the otherwise valid history.
        with pytest.raises(IncrementalWorkflowCaptureError):
            await capture_incremental_workflow_eval_attempt(
                target,
                restored,
                messages=messages,
                limits=IncrementalEvidenceLimits(max_root_bytes=1 << 20),
                admission=IncrementalEvidenceAdmission(),
            )
        await app.session_store.update_metadata(session.id, {"tampered": True})
        with pytest.raises(SessionTrajectoryError):
            await capture_workflow_eval_attempt(
                target, restored, messages=messages, bounds=SessionTrajectoryBounds()
            )
        with pytest.raises(SessionTrajectoryError):
            await score_workflow_eval_capture(target, capture, specs)

    asyncio.run(exercise())


def test_unversioned_saved_anchor_keeps_legacy_digest():
    async def exercise():
        app = _register_app()
        target = _target(app, _NoChildWorkflow)
        suite = _suite()
        run = await run_workflow_eval_suite(target, suite)
        trial = run.cases[0].trials[0]
        session, records = await _root(app, target, trial)
        legacy = hashlib.sha256(
            canonical_durable_json_bytes(
                {
                    "session": session.model_dump(mode="json"),
                    "records": [r.model_dump(mode="json") for r in records],
                },
                "legacy",
            )
        ).hexdigest()
        assert _workflow_root_sha256(session, records, version="document-v1") == legacy
        data = run.model_dump(mode="json")
        saved = data["cases"][0]["trials"][0]
        for anchor in (saved["workflow_attempt"], saved["retained_workflow_output"]["anchor"]):
            anchor.pop("root_hash_version")
            anchor["root_sha256"] = legacy
        restored = EvalRun.model_validate(data).cases[0].trials[0]
        assert restored.workflow_attempt.root_hash_version == "document-v1"
        capture = await capture_workflow_eval_attempt(
            target,
            restored,
            messages=tuple(suite.cases[0].request.messages),
            bounds=SessionTrajectoryBounds(),
        )
        score = await score_workflow_eval_capture(
            target, capture, (FinalOutputEqualsAssertionSpec(id="answer", expected="done"),)
        )
        assert score.score == 1.0
        # Version changes never silently reinterpret an existing digest.
        for anchor in (saved["workflow_attempt"], saved["retained_workflow_output"]["anchor"]):
            anchor["root_hash_version"] = "framed-v2"
        with pytest.raises(SessionTrajectoryError):
            await capture_workflow_eval_attempt(
                target,
                EvalRun.model_validate(data).cases[0].trials[0],
                messages=tuple(suite.cases[0].request.messages),
                bounds=SessionTrajectoryBounds(),
            )
        saved["workflow_attempt"]["root_hash_version"] = "future"
        with pytest.raises(ValueError):
            EvalRun.model_validate(data)

    asyncio.run(exercise())


def test_large_history_changed_during_scoring_cannot_publish(monkeypatch):
    import cayu.evals.runner as runner

    async def exercise():
        app = _register_app()
        target = _target(app, _LargeHistoryWorkflow)
        evaluate = runner._evaluate_assertions_with_prepared_evidence
        scored = False

        async def change_after_scoring(assertions, context, **kwargs):
            nonlocal scored
            results = await evaluate(assertions, context, **kwargs)
            assert all(result.score == 1.0 for result in results)
            scored = True
            session_id = context.trajectory.session.id
            await app.session_store.append_event(
                session_id,
                Event(
                    type="custom.late",
                    session_id=session_id,
                    workflow_name=target.workflow_spec.name,
                    payload={"attempt_id": context.trajectory.workflow_output.attempt_id},
                ),
            )
            return results

        monkeypatch.setattr(
            runner, "_evaluate_assertions_with_prepared_evidence", change_after_scoring
        )
        run = await run_workflow_eval_suite(target, _suite(FinalOutputContains("done")))
        trial = run.cases[0].trials[0]
        assert scored
        assert trial.status is not EvalStatus.PASSED
        assert trial.score is None
        assert not trial.evidence_complete
        assert "continues after completion" in trial.error
        assert trial.final_output == ""

    asyncio.run(exercise())
