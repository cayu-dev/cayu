"""Portable assertions on one JSON-pointer field of a run's structured output."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError
from tests._session_provenance import fixture_session_invocation
from tests.evals.test_corpus_execution import _source
from tests.evals.test_external_process_target import _target as _external_target_identity
from tests.evals.test_workflow_eval_target import _register_app

from cayu import AgentSpec, Message, RunRequest
from cayu._validation import copy_durable_json_object
from cayu.applications import CayuApp
from cayu.budgets.usage import session_usage_summary
from cayu.context import StructuredOutputSpec
from cayu.evals import (
    ApplicationEvaluatorAssertionSpec,
    PublishedStructuredOutputFieldDetail,
    StructuredOutputFieldAssertionSpec,
)
from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_DETAIL_BYTES,
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    RunInputSpec,
    TrialRequestSpec,
    assertion_spec_revision,
    eval_corpus_from_json,
    eval_corpus_to_json,
)
from cayu.evals.evidence import project_assertion_evidence_view
from cayu.evals.execution import CorpusTarget, WorkflowEvalTarget, run_corpus_suite
from cayu.evals.execution_comparison import (
    CorpusComparisonReason,
    corpus_execution_compatibility,
)
from cayu.evals.execution_reporting import (
    corpus_execution_result_from_json,
    corpus_execution_result_to_json,
)
from cayu.evals.external import ExternalProcessTargetIdentityV1
from cayu.evals.models import EvalOutcome, Trajectory
from cayu.evals.portable_assertions import compile_assertion_spec
from cayu.evals.portable_evaluation import evaluate_assertion_spec
from cayu.evals.published import PublishedAssertionResult
from cayu.evals.runner import EvalCase, run_eval_case
from cayu.evals.testing import ScriptedModelProvider, scripted_structured_output
from cayu.evals.workflow_target import (
    WorkflowEvalExecution,
    WorkflowEvalInstanceScope,
    WorkflowEvalResult,
)
from cayu.events import Event, EventType
from cayu.providers import ModelStreamEvent
from cayu.sessions.records import Session, SessionStatus
from cayu.vaults import REDACTED_SECRET, SecretRedactor
from cayu.workflows.base import WorkflowSpec
from cayu.workflows.workflow import WorkflowBase

_REVISION = "sha256:" + "a" * 64
_SUITE_ID = "verdicts"
_SECRET = "s3cr3t-verifier-token"
_VERDICT: dict[str, Any] = {
    "verdict": "confirmed",
    "confidence": 0.82,
    "attempts": 3,
    "reference": "CVE-2026-1234",
    "tags": ["race", "memory"],
    "reviewer": None,
    "paths": {"a/b": {"~k": 1}},
}


def _spec(**fields: Any) -> StructuredOutputFieldAssertionSpec:
    return StructuredOutputFieldAssertionSpec(id=fields.pop("id", "field-check"), **fields)


def _run_events(session_id: str, outputs: tuple[object, ...]) -> list[Event]:
    return [
        *(
            Event(
                type=EventType.STRUCTURED_OUTPUT_VALIDATED,
                session_id=session_id,
                payload={"name": "verdict", "attempt": 1, "max_retries": 0, "output": output},
            )
            for output in outputs
        ),
        Event(type=EventType.SESSION_COMPLETED, session_id=session_id),
    ]


def _trajectory(*outputs: object, earlier_run: tuple[object, ...] | None = None) -> Trajectory:
    """One completed session; ``earlier_run`` adds a run the session was resumed after."""

    session_id = "structured-output-session"
    events = _run_events(session_id, outputs)
    if earlier_run is not None:
        events = [
            Event(type=EventType.SESSION_STARTED, session_id=session_id),
            *_run_events(session_id, earlier_run),
            Event(type=EventType.SESSION_RESUMED, session_id=session_id),
            *events,
        ]
    return Trajectory(
        session=Session(
            id=session_id,
            agent_name="verifier",
            provider_name="fixture",
            model="fixture-model",
            invocation=fixture_session_invocation(session_id),
            status=SessionStatus.COMPLETED,
        ),
        events=tuple(events),
        usage_summary=session_usage_summary(session_id, events),
    )


def _evaluate(spec, trajectory: Trajectory, *, app: CayuApp | None = None):
    evidence = project_assertion_evidence_view(
        app or CayuApp(enable_logging=False),
        trajectory,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    return evaluate_assertion_spec(spec, evidence)


def _corpus(*assertions, trials: int = 1) -> EvalCorpusDocument:
    suite = EvalSuiteSpec.create(
        id=_SUITE_ID,
        name="Verdicts",
        trial_request=TrialRequestSpec(trials=trials, timeout_seconds=30),
    )
    case = EvalCaseSpec.create(
        id="race-report",
        suite_id=suite.id,
        name="Race report",
        source=_source(),
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Verify the race report."),)),
        assertions=assertions,
    )
    return EvalCorpusDocument.create(
        target_key="verifier",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        suites=(suite,),
        cases=(case,),
    )


class _VerifierWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="verifier-eval")

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        yield await ctx.completed({**_VERDICT, "notes": "n" * 2_000})


def _verdict_schema(*properties: str) -> StructuredOutputSpec:
    return StructuredOutputSpec(
        name="verdict",
        json_schema={
            "type": "object",
            "properties": {name: {"type": "string"} for name in ("verdict", *properties)},
            "required": ["verdict"],
        },
    )


def _agent_target(structured_output: StructuredOutputSpec | None) -> CorpusTarget:
    """An agent target whose trusted request base may ask for the verifier's verdict."""

    script = (
        scripted_structured_output({"verdict": "confirmed"})
        if structured_output is not None
        else (
            ModelStreamEvent.text_delta("confirmed"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([script]), default=True)
    app.register_agent(AgentSpec(name="verifier", model="scripted-model"))
    return CorpusTarget(
        key="verifier",
        app=app,
        request_base=RunRequest(
            agent_name="verifier",
            messages=[],
            structured_output=structured_output,
        ),
        application_release_id="verifier-release",
    )


def _workflow_target(
    *,
    project_structured_output: bool = True,
    structured_output: StructuredOutputSpec | None = None,
) -> WorkflowEvalTarget:
    app = _register_app()

    def execution(invocation) -> WorkflowEvalExecution:
        return WorkflowEvalExecution(app=app, workflow=_VerifierWorkflow(app))

    def project(evidence) -> WorkflowEvalResult:
        payload = dict(evidence.completion_event.payload)
        return WorkflowEvalResult(
            final_output=payload["verdict"],
            structured_output=payload if project_structured_output else None,
        )

    return WorkflowEvalTarget(
        key="verifier",
        app=app,
        request_base=RunRequest(
            agent_name="first", messages=[], structured_output=structured_output
        ),
        application_release_id="verifier-release",
        workflow_spec=_VerifierWorkflow.spec,
        implementation_revision=_REVISION,
        result_projector_revision=_REVISION,
        execution_scope_revision=_REVISION,
        instance_scope=WorkflowEvalInstanceScope.SHARED,
        workflow_factory=execution,
        result_projector=project,
    )


def _published_assertions(result) -> tuple[PublishedAssertionResult, ...]:
    (case,) = result.run.cases
    (trial,) = case.trials
    return trial.assertions


@pytest.mark.parametrize(
    ("fields", "message"),
    (
        ({"pointer": "verdict", "operator": "exists"}, "must be empty or start with '/'"),
        ({"pointer": "/a~2b", "operator": "exists"}, "may escape only"),
        ({"pointer": "/a~", "operator": "exists"}, "may escape only"),
        ({"pointer": "/" + "a" * 1_024, "operator": "exists"}, "at most 1024 characters"),
        ({"pointer": "/verdict", "operator": "contains", "value": "x"}, "operator"),
        ({"pointer": "/verdict", "operator": "exists", "value": "x"}, "takes no value"),
        ({"pointer": "/verdict", "operator": "in", "value": []}, "non-empty array"),
        ({"pointer": "/verdict", "operator": "in", "value": "confirmed"}, "non-empty array"),
        ({"pointer": "/verdict", "operator": "in"}, "non-empty array"),
        ({"pointer": "/attempts", "operator": "gt", "value": "1"}, "requires a JSON number"),
        ({"pointer": "/attempts", "operator": "lte", "value": True}, "requires a JSON number"),
        ({"pointer": "/reference", "operator": "matches", "value": 1}, "pattern string"),
        ({"pointer": "/reference", "operator": "matches", "value": "(?=CVE)"}, "support only"),
        ({"pointer": "/verdict", "operator": "equals", "value": "x" * 4_097}, "byte limit"),
    ),
)
def test_spec_validates_pointer_operator_and_value_shape(fields, message):
    with pytest.raises(ValidationError, match=message):
        _spec(**fields)


def test_spec_round_trips_through_corpus_json_and_revisions_cover_the_comparison():
    specs = (
        _spec(id="verdict", pointer="/verdict", operator="equals", value="confirmed"),
        _spec(id="decided", pointer="/verdict", operator="in", value=["confirmed", "refuted"]),
        _spec(id="whole-output", pointer="", operator="exists"),
        _spec(id="reference", pointer="/reference", operator="matches", value=r"^CVE-\d+"),
        _spec(id="confidence", pointer="/confidence", operator="gte", value=0.8),
        _spec(id="no-reviewer", pointer="/reviewer", operator="equals", value=None),
    )
    corpus = _corpus(*specs)

    loaded = eval_corpus_from_json(eval_corpus_to_json(corpus))

    assert loaded == corpus
    assert loaded.cases[0].assertions == specs
    document = json.loads(eval_corpus_to_json(corpus))["cases"][0]["assertions"]
    assert document[0] == {
        "description": None,
        "id": "verdict",
        "kind": "structured_output_field",
        "operator": "equals",
        "pointer": "/verdict",
        "value": "confirmed",
    }
    assert "value" not in document[2]
    revisions = {
        assertion_spec_revision(spec)
        for spec in (
            specs[0],
            _spec(id="verdict", pointer="/verdict", operator="equals", value="refuted"),
            _spec(id="verdict", pointer="/outcome", operator="equals", value="confirmed"),
            _spec(id="verdict", pointer="/verdict", operator="in", value=["confirmed"]),
        )
    }
    assert len(revisions) == 4


@pytest.mark.parametrize(
    ("fields", "outcome", "observed"),
    (
        (
            {"pointer": "/verdict", "operator": "equals", "value": "confirmed"},
            "passed",
            "confirmed",
        ),
        ({"pointer": "/verdict", "operator": "equals", "value": "refuted"}, "failed", "confirmed"),
        ({"pointer": "/attempts", "operator": "equals", "value": 3.0}, "passed", 3),
        ({"pointer": "/attempts", "operator": "equals", "value": True}, "failed", 3),
        ({"pointer": "/tags", "operator": "equals", "value": ["race", "memory"]}, "passed", None),
        ({"pointer": "/reviewer", "operator": "equals", "value": None}, "passed", None),
        (
            {"pointer": "/verdict", "operator": "in", "value": ["refuted", "confirmed"]},
            "passed",
            "confirmed",
        ),
        ({"pointer": "/verdict", "operator": "in", "value": ["refuted"]}, "failed", "confirmed"),
        ({"pointer": "/tags/1", "operator": "exists"}, "passed", "memory"),
        ({"pointer": "/paths/a~1b/~0k", "operator": "exists"}, "passed", 1),
        ({"pointer": "", "operator": "exists"}, "passed", None),
        (
            {"pointer": "/reference", "operator": "matches", "value": r"^CVE-\d{4}-\d+$"},
            "passed",
            "CVE-2026-1234",
        ),
        ({"pointer": "/reference", "operator": "matches", "value": "^GHSA"}, "failed", None),
        ({"pointer": "/attempts", "operator": "matches", "value": "3"}, "failed", 3),
        ({"pointer": "/confidence", "operator": "gt", "value": 0.8}, "passed", 0.82),
        ({"pointer": "/confidence", "operator": "gte", "value": 0.82}, "passed", 0.82),
        ({"pointer": "/confidence", "operator": "lt", "value": 0.82}, "failed", 0.82),
        ({"pointer": "/attempts", "operator": "lte", "value": 3}, "passed", 3),
        ({"pointer": "/attempts", "operator": "gt", "value": 3}, "failed", 3),
        ({"pointer": "/verdict", "operator": "gt", "value": 0}, "failed", "confirmed"),
    ),
)
def test_operators_compare_the_selected_field(fields, outcome, observed):
    result = _evaluate(_spec(**fields), _trajectory(_VERDICT))

    assert result.outcome == EvalOutcome(outcome)
    assert result.score == (1.0 if outcome == "passed" else 0.0)
    assert result.metadata["observation_state"] == "available"
    assert result.metadata["matched"] is (outcome == "passed")
    if observed is not None:
        assert result.metadata["observed"] == observed


@pytest.mark.parametrize("pointer", ("/missing", "/tags/2", "/tags/01", "/tags/-", "/verdict/x"))
def test_an_unresolved_pointer_is_a_conclusive_failure(pointer):
    result = _evaluate(_spec(pointer=pointer, operator="exists"), _trajectory(_VERDICT))

    assert result.outcome is EvalOutcome.FAILED
    assert result.metadata == {
        "pointer": pointer,
        "operator": "exists",
        "observation_state": "unresolved",
        "matched": False,
    }


def test_no_validated_structured_output_is_unavailable_with_its_reason():
    result = _evaluate(
        _spec(pointer="/verdict", operator="equals", value="confirmed"),
        _trajectory(),
    )

    assert result.outcome is EvalOutcome.UNAVAILABLE
    assert result.message == (
        "Structured output evidence is unavailable because no structured output was validated."
    )
    assert result.metadata["observation_state"] == "not_validated"


def test_the_last_validated_structured_output_wins():
    trajectory = _trajectory({"verdict": "refuted"}, {"verdict": "confirmed"})

    result = _evaluate(_spec(pointer="/verdict", operator="equals", value="confirmed"), trajectory)

    assert result.outcome is EvalOutcome.PASSED
    assert result.metadata["observed"] == "confirmed"


def test_only_the_current_run_of_a_resumed_session_counts():
    spec = _spec(pointer="/verdict", operator="equals", value="confirmed")

    earlier_only = _evaluate(spec, _trajectory(earlier_run=({"verdict": "confirmed"},)))
    both = _evaluate(
        spec,
        _trajectory({"verdict": "confirmed"}, earlier_run=({"verdict": "refuted"},)),
    )

    assert earlier_only.outcome is EvalOutcome.UNAVAILABLE
    assert earlier_only.metadata["observation_state"] == "not_validated"
    assert both.outcome is EvalOutcome.PASSED
    assert both.metadata["observed"] == "confirmed"


def test_redaction_makes_compared_values_and_hidden_keys_unavailable():
    app = CayuApp(enable_logging=False, secret_redactor=SecretRedactor(_SECRET))
    trajectory = _trajectory({"verdict": "confirmed", "token": _SECRET, _SECRET: 1})

    redacted_value = _evaluate(
        _spec(pointer="/token", operator="equals", value=_SECRET), trajectory, app=app
    )
    hidden_key = _evaluate(_spec(pointer=f"/{_SECRET}", operator="exists"), trajectory, app=app)
    present = _evaluate(_spec(pointer="/token", operator="exists"), trajectory, app=app)
    unaffected = _evaluate(
        _spec(pointer="/verdict", operator="equals", value="confirmed"), trajectory, app=app
    )

    assert redacted_value.outcome is EvalOutcome.UNAVAILABLE
    assert redacted_value.metadata["observation_state"] == "redacted"
    assert hidden_key.outcome is EvalOutcome.UNAVAILABLE
    assert hidden_key.metadata["observation_state"] == "redacted"
    assert present.outcome is EvalOutcome.PASSED
    assert _SECRET not in json.dumps(present.metadata)
    assert unaffected.outcome is EvalOutcome.PASSED


def test_oversized_outputs_and_pattern_subjects_are_unavailable_not_failed():
    oversized = _evaluate(
        _spec(pointer="/verdict", operator="exists"),
        _trajectory({"verdict": "confirmed", "log": "x" * (256 << 10)}),
    )
    long_subject = _evaluate(
        _spec(pointer="/log", operator="matches", value="x"),
        _trajectory({"log": "x" * 4_097}),
    )

    assert oversized.outcome is EvalOutcome.UNAVAILABLE
    assert oversized.metadata["observation_state"] == "limit_exceeded"
    assert long_subject.outcome is EvalOutcome.UNAVAILABLE
    assert long_subject.metadata["observation_state"] == "limit_exceeded"


def test_agent_suites_read_the_root_sessions_validated_structured_output():
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider([scripted_structured_output({"verdict": "confirmed"})]),
        default=True,
    )
    app.register_agent(AgentSpec(name="verifier", model="scripted-model"))
    spec = _spec(pointer="/verdict", operator="equals", value="confirmed")
    case = EvalCase(
        id="race-report",
        request=RunRequest(
            agent_name="verifier",
            messages=[Message.text("user", "Verify the race report.")],
            structured_output=StructuredOutputSpec(
                name="verdict",
                json_schema={
                    "type": "object",
                    "properties": {"verdict": {"type": "string"}},
                    "required": ["verdict"],
                },
            ),
        ),
        assertions=[
            compile_assertion_spec(
                spec,
                app=app,
                evidence_policy=EvaluationEvidencePolicySpec.standard(),
                trusted_pricing=None,
            )
        ],
    )

    result = asyncio.run(run_eval_case(app, case, suite_id=_SUITE_ID))

    (trial,) = result.trials
    (assertion,) = trial.assertions
    assert assertion.outcome is EvalOutcome.PASSED
    assert assertion.assertion_revision == assertion_spec_revision(spec)
    assert assertion.metadata["observed"] == "confirmed"


def test_workflow_targets_publish_the_projected_field_and_its_observed_value():
    corpus = _corpus(
        _spec(id="verdict-confirmed", pointer="/verdict", operator="equals", value="confirmed"),
        _spec(id="notes-present", pointer="/notes", operator="exists"),
    )

    result = asyncio.run(run_corpus_suite(_workflow_target(), corpus, _SUITE_ID))

    verdict, notes = _published_assertions(result)
    assert result.run.status == "passed"
    assert verdict.outcome == "passed"
    assert type(verdict.detail) is PublishedStructuredOutputFieldDetail
    assert verdict.detail.model_dump(mode="json") == {
        "kind": "structured_output_field",
        "pointer": "/verdict",
        "operator": "equals",
        "value": "confirmed",
        "structured_output_revision": None,
        "observation_state": "available",
        "observed": {"value": "confirmed"},
        "matched": True,
    }
    # A field over the 1 KiB publication bound is compared but not published.
    assert notes.outcome == "passed"
    assert notes.detail.observation_state == "available"
    assert notes.detail.observed is None
    assert notes.detail.matched is True
    assert corpus_execution_result_from_json(corpus_execution_result_to_json(result)) == result


def test_agent_corpus_targets_assert_on_the_structured_output_their_request_base_asks_for():
    corpus = _corpus(
        _spec(id="verdict-confirmed", pointer="/verdict", operator="equals", value="confirmed")
    )

    result = asyncio.run(run_corpus_suite(_agent_target(_verdict_schema()), corpus, _SUITE_ID))

    (assertion,) = _published_assertions(result)
    assert result.run.status == "passed"
    assert assertion.outcome == "passed"
    assert assertion.detail.observation_state == "available"
    assert assertion.detail.observed is not None
    assert assertion.detail.observed.value == "confirmed"
    assert assertion.detail.structured_output_revision is not None
    assert corpus_execution_result_from_json(corpus_execution_result_to_json(result)) == result


def test_a_changed_requested_schema_makes_field_results_incomparable():
    corpus = _corpus(_spec(pointer="/verdict", operator="equals", value="confirmed"))

    def run(structured_output: StructuredOutputSpec | None):
        return asyncio.run(run_corpus_suite(_agent_target(structured_output), corpus, _SUITE_ID))

    baseline = run(_verdict_schema())
    same = run(_verdict_schema())
    changed = run(_verdict_schema("reason"))
    unrequested = run(None)

    (unrequested_assertion,) = _published_assertions(unrequested)
    assert unrequested_assertion.outcome == "unavailable"
    assert unrequested_assertion.detail.observation_state == "not_validated"
    assert unrequested_assertion.detail.structured_output_revision is None
    assert corpus_execution_compatibility(baseline, same).comparable is True
    for current in (changed, unrequested):
        compatibility = corpus_execution_compatibility(baseline, current)
        assert compatibility.comparable is False
        assert compatibility.reasons == (CorpusComparisonReason.ASSERTION_CONTRACT_MISMATCH,)


def test_targets_that_cannot_honor_a_requested_structured_output_reject_it():
    policy = EvaluationEvidencePolicySpec.standard()
    base = _external_target_identity()
    external = ExternalProcessTargetIdentityV1.create(
        body=base.body,
        evaluator_runtime_revision=base.evaluator_runtime_revision,
        target_implementation_revision=base.target_implementation_revision,
        runner_revision=base.runner_revision,
        environment_revision=base.environment_revision,
        reset_contract_revision=base.reset_contract_revision,
        evidence_policy_revision=policy.revision,
    )

    with pytest.raises(ValidationError, match="through its result projector"):
        _workflow_target(structured_output=_verdict_schema())
    with pytest.raises(ValidationError, match="External process targets cannot request"):
        CorpusTarget(
            key="verifier",
            app=CayuApp(enable_logging=False),
            request_base=RunRequest(
                agent_name="verifier", messages=[], structured_output=_verdict_schema()
            ),
            application_release_id="verifier-release",
            evidence_policy=policy,
            external_process=external,
        )


def test_a_workflow_that_projects_no_structured_output_is_unavailable():
    corpus = _corpus(_spec(pointer="/verdict", operator="equals", value="confirmed"))

    result = asyncio.run(
        run_corpus_suite(_workflow_target(project_structured_output=False), corpus, _SUITE_ID)
    )

    (assertion,) = _published_assertions(result)
    assert result.run.status == "unavailable"
    assert assertion.outcome == "unavailable"
    assert assertion.detail.observation_state == "not_validated"
    assert assertion.detail.matched is None


def test_published_details_reject_contradictory_observations():
    corpus = _corpus(_spec(pointer="/verdict", operator="equals", value="refuted"))
    result = asyncio.run(run_corpus_suite(_workflow_target(), corpus, _SUITE_ID))
    (assertion,) = _published_assertions(result)
    document = assertion.model_dump(mode="json")
    assert document["outcome"] == "failed"

    def with_detail(**changes: Any) -> dict[str, Any]:
        return {**document, "detail": {**document["detail"], **changes}}

    assert PublishedAssertionResult.model_validate(document) == assertion
    with pytest.raises(ValidationError, match="contradicts its observed value"):
        PublishedAssertionResult.model_validate(with_detail(matched=True))
    with pytest.raises(ValidationError, match="Only an available"):
        PublishedAssertionResult.model_validate(
            with_detail(observation_state="redacted", matched=None)
        )
    with pytest.raises(ValidationError, match="conclusive mismatch"):
        PublishedAssertionResult.model_validate(
            with_detail(observation_state="unresolved", observed=None, matched=True)
        )
    with pytest.raises(ValidationError, match="requires a JSON number"):
        PublishedAssertionResult.model_validate(with_detail(operator="gt"))


def test_changing_the_comparison_makes_results_incomparable():
    target = _workflow_target()
    baseline = asyncio.run(
        run_corpus_suite(
            target,
            _corpus(_spec(pointer="/verdict", operator="equals", value="confirmed")),
            _SUITE_ID,
        )
    )
    same = asyncio.run(
        run_corpus_suite(
            target,
            _corpus(_spec(pointer="/verdict", operator="equals", value="confirmed")),
            _SUITE_ID,
        )
    )
    changed = asyncio.run(
        run_corpus_suite(
            target,
            _corpus(_spec(pointer="/verdict", operator="in", value=["confirmed"])),
            _SUITE_ID,
        )
    )

    assert corpus_execution_compatibility(baseline, same).comparable is True
    compatibility = corpus_execution_compatibility(baseline, changed)
    assert compatibility.comparable is False
    assert CorpusComparisonReason.ASSERTION_CONTRACT_MISMATCH in compatibility.reasons


def test_fields_and_application_evaluators_share_one_published_detail_budget():
    choices = ["x" * 4_000]
    params = {"expected": "x" * 2_000}
    fields = tuple(
        _spec(id=f"field-{index}", pointer="/verdict", operator="in", value=choices)
        for index in range(32)
    )
    evaluators = tuple(
        ApplicationEvaluatorAssertionSpec(
            id=f"evaluator-{index}", evaluator_key="verdict-rules", params=params
        )
        for index in range(8)
    )
    # Each trial may publish every field's pointer, value, and a 1 KiB observation,
    # and every evaluator's parameters with a 4 KiB message and 4 KiB of metadata.
    field_bytes = 32 * (len(json.dumps("/verdict")) + len(json.dumps(choices)) + 1_024)
    evaluator_bytes = 8 * (len(json.dumps(params, separators=(",", ":"))) + 8_192)
    trials = 60
    assert field_bytes * trials < EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_DETAIL_BYTES
    assert evaluator_bytes * trials < EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_DETAIL_BYTES
    assert (
        field_bytes + evaluator_bytes
    ) * trials > EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_DETAIL_BYTES

    _corpus(*fields, trials=trials)
    _corpus(*evaluators, trials=trials)
    with pytest.raises(ValueError, match="published assertion-detail bytes"):
        _corpus(*fields, *evaluators, trials=trials)


def test_selected_redaction_marker_key_is_unavailable():
    app = CayuApp(enable_logging=False, secret_redactor=SecretRedactor(_SECRET))
    result = _evaluate(
        _spec(pointer=f"/{REDACTED_SECRET}", operator="equals", value=42),
        _trajectory({_SECRET: 42}),
        app=app,
    )
    assert result.outcome is EvalOutcome.UNAVAILABLE
    assert result.metadata["observation_state"] == "redacted"


def test_deep_observation_is_compared_without_retaining_unpublishable_metadata():
    value: Any = 1
    for _ in range(60):
        value = [value]
    result = _evaluate(_spec(pointer="", operator="exists"), _trajectory(value))
    assert result.outcome is EvalOutcome.PASSED
    assert "observed" not in result.metadata
    copy_durable_json_object({"assertion": result.metadata}, "record")
