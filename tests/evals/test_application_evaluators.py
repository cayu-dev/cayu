"""Application-registered evaluators addressed by key from portable corpora."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

import pytest
from pydantic import ValidationError
from tests.evals.test_corpus_execution import _provider, _source
from tests.evals.test_workflow_eval_target import _register_app

from cayu import AgentSpec, Message, RunRequest
from cayu.applications import CayuApp
from cayu.budgets.pricing import SessionCostSummary
from cayu.evals import (
    ApplicationEvaluator,
    ApplicationEvaluatorAssertionSpec,
    EvalAssertion,
    EvalOutcome,
    PublishedApplicationEvaluatorDetail,
)
from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    RootStatusAssertionSpec,
    RunInputSpec,
    TrialRequestSpec,
    assertion_spec_revision,
    eval_corpus_from_json,
    eval_corpus_to_json,
)
from cayu.evals.execution import (
    CorpusTarget,
    WorkflowEvalTarget,
    _copy_corpus_target,
    compile_corpus_suite,
    run_corpus_suite,
)
from cayu.evals.execution_comparison import (
    CorpusComparisonReason,
    corpus_execution_compatibility,
)
from cayu.evals.execution_reporting import (
    corpus_execution_result_from_json,
    corpus_execution_result_to_json,
    render_corpus_execution_html,
)
from cayu.evals.memory_attribution import EvalMemoryAttributionEvidenceV1
from cayu.evals.models import EvalAssertionResult, EvalContext
from cayu.evals.portable_assertions import compile_assertion_spec
from cayu.evals.published import PublishedAssertionResult, _published_detail
from cayu.evals.result_presentation import present_eval_result
from cayu.evals.runner import _blocked_assertion_results
from cayu.evals.testing import ScriptedModelProvider
from cayu.evals.workflow_target import (
    WorkflowEvalExecution,
    WorkflowEvalInstanceScope,
    WorkflowEvalResult,
)
from cayu.providers import ModelProvider
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.vaults import SecretRedactor
from cayu.workflows.base import WorkflowSpec
from cayu.workflows.workflow import WorkflowBase

_REVISION = "sha256:" + "a" * 64
_SECOND_REVISION = "sha256:" + "b" * 64
_SUITE_ID = "answers"
_CASE_ID = "total-cost"


class _AnswerCheck(EvalAssertion):
    """Compare a numeric answer with an expected value within a tolerance."""

    def __init__(self, params: Mapping[str, Any]) -> None:
        self._expected = Decimal(str(params["expected"]))
        self._tolerance = Decimal(str(params.get("tolerance", 0)))
        self._field = params.get("field")

    @property
    def reads_final_output_only(self) -> bool:
        return True

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        if self._field is None:
            answer: object = context.final_output
        else:
            workflow_output = context.trajectory.workflow_output
            if workflow_output is None or workflow_output.structured_output is None:
                return self.unavailable("Structured output was not retained.")
            answer = workflow_output.structured_output.get(self._field)
        metadata = {"expected": str(self._expected), "observed": str(answer)}
        try:
            observed = Decimal(str(answer).strip())
        except InvalidOperation:
            return self.failed("The answer is not numeric.", metadata=metadata)
        if abs(observed - self._expected) <= self._tolerance:
            return self.passed("The answer is within tolerance.", metadata=metadata)
        return self.failed("The answer is outside tolerance.", metadata=metadata)


class _ResultAssertion(EvalAssertion):
    """Return a fixed result so publication boundaries can be exercised directly."""

    def __init__(self, build: Callable[[EvalAssertion], EvalAssertionResult]) -> None:
        self._build = build

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        del context
        return self._build(self)


class _StableScriptedProvider(ScriptedModelProvider):
    """A scripted provider whose behavior identity is the same in every app."""

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:application-evaluator-provider",
            behavior_version="1",
            implementation_version="1",
        )


class _AnswerWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="answer-eval")

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        yield await ctx.completed(
            {"answer": "42.75", "explanation": "Sum of the three listed amounts."}
        )


def _answer_check(
    *,
    revision: str = _REVISION,
    factory: Callable[[Mapping[str, Any]], EvalAssertion] = _AnswerCheck,
) -> ApplicationEvaluator:
    return ApplicationEvaluator(
        key="answer-check", implementation_revision=revision, factory=factory
    )


def _spec(
    *,
    evaluator_key: str = "answer-check",
    params: dict[str, Any] | None = None,
) -> ApplicationEvaluatorAssertionSpec:
    return ApplicationEvaluatorAssertionSpec(
        id="answer-correct",
        evaluator_key=evaluator_key,
        params={"expected": 42.75, "tolerance": 0.005} if params is None else params,
    )


def _corpus(
    *assertions,
    target_key: str = "answer-agent",
) -> EvalCorpusDocument:
    suite = EvalSuiteSpec.create(
        id=_SUITE_ID,
        name="Answers",
        trial_request=TrialRequestSpec(trials=1, timeout_seconds=30),
    )
    case = EvalCaseSpec.create(
        id=_CASE_ID,
        suite_id=suite.id,
        name="Total cost",
        source=_source(),
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="What is the total cost?"),)),
        assertions=assertions or (_spec(),),
    )
    return EvalCorpusDocument.create(
        target_key=target_key,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        suites=(suite,),
        cases=(case,),
    )


def _target(
    provider: ModelProvider,
    *,
    evaluators: tuple[ApplicationEvaluator, ...] = (),
    secret_redactor: SecretRedactor | None = None,
) -> CorpusTarget:
    app = CayuApp(enable_logging=False, secret_redactor=secret_redactor)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="agent", model="fixture-model"))
    return CorpusTarget(
        key="answer-agent",
        app=app,
        request_base=RunRequest(agent_name="agent", messages=[], max_steps=1),
        bootstrap_messages=(Message.text("system", "Answer with one number."),),
        application_release_id="answer-release",
        application_evaluators=evaluators,
    )


def _workflow_target(
    app: CayuApp,
    evaluators: tuple[ApplicationEvaluator, ...],
    *,
    trial_app: Callable[[], CayuApp] | None = None,
):
    """A target over ``app``, or over a fresh ``trial_app()`` for each trial."""

    def execution(invocation) -> WorkflowEvalExecution:
        runtime_app = app if trial_app is None else trial_app()
        return WorkflowEvalExecution(app=runtime_app, workflow=_AnswerWorkflow(runtime_app))

    return WorkflowEvalTarget(
        key="answer-agent",
        app=app,
        request_base=RunRequest(agent_name="first", messages=[]),
        application_release_id="answer-release",
        workflow_spec=_AnswerWorkflow.spec,
        implementation_revision=_REVISION,
        result_projector_revision=_REVISION,
        execution_scope_revision=_REVISION,
        instance_scope=(
            WorkflowEvalInstanceScope.SHARED
            if trial_app is None
            else WorkflowEvalInstanceScope.PER_TRIAL
        ),
        workflow_factory=execution,
        result_projector=lambda evidence: WorkflowEvalResult(
            final_output=evidence.completion_event.payload["explanation"],
            structured_output=dict(evidence.completion_event.payload),
        ),
        application_evaluators=evaluators,
    )


def _run(target: CorpusTarget, corpus: EvalCorpusDocument):
    return asyncio.run(run_corpus_suite(target, corpus, _SUITE_ID))


def _only_assertion(result):
    (case,) = result.run.cases
    (trial,) = case.trials
    (assertion,) = trial.assertions
    return assertion


def test_application_evaluator_spec_round_trips_as_bounded_authority_free_data():
    corpus = _corpus()

    restored = eval_corpus_from_json(eval_corpus_to_json(corpus))

    assert restored == corpus
    (assertion,) = restored.cases[0].assertions
    assert type(assertion) is ApplicationEvaluatorAssertionSpec
    assert assertion.kind == "application_evaluator"
    assert assertion.params == {"expected": 42.75, "tolerance": 0.005}
    assert assertion_spec_revision(assertion) != assertion_spec_revision(
        _spec(params={"expected": 42.75, "tolerance": 0.01})
    )
    assert assertion_spec_revision(assertion) != assertion_spec_revision(
        _spec(evaluator_key="other-check")
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"evaluator_key": "Answer Check"},
        {"evaluator_key": ""},
        {"params": ["not", "an", "object"]},
        {"params": {"value": "x" * (17 * 1024)}},
        {"params": {"value": [1] * 1025}},
        {"params": {"value": float("nan")}},
        {"implementation_revision": _REVISION},
    ],
)
def test_application_evaluator_spec_rejects_invalid_fields(fields):
    document = {
        "id": "answer-correct",
        "evaluator_key": "answer-check",
        "params": {"expected": 1},
        **fields,
    }

    with pytest.raises(ValidationError):
        ApplicationEvaluatorAssertionSpec.model_validate(document)


def test_application_evaluator_params_reject_excessive_nesting():
    nested: dict[str, Any] = {}
    cursor = nested
    for _ in range(30):
        cursor["child"] = {}
        cursor = cursor["child"]

    with pytest.raises(ValidationError):
        _spec(params=nested)


@pytest.mark.parametrize(
    ("params", "assertions"),
    [({"expected": "x" * 15_000}, 8), ({"x": "\N{GRINNING FACE}" * 4_000}, 16)],
    ids=["ascii", "four-byte-characters"],
)
def test_a_suite_whose_evaluator_payloads_could_not_be_published_is_rejected(params, assertions):
    # Each trial publishes every evaluator's parameters and its largest observation again,
    # counted in the UTF-8 bytes the published run's own limit counts.
    suite = EvalSuiteSpec.create(
        id=_SUITE_ID,
        name="Answers",
        trial_request=TrialRequestSpec(trials=100, timeout_seconds=30),
    )
    case = EvalCaseSpec.create(
        id=_CASE_ID,
        suite_id=suite.id,
        name="Total cost",
        source=_source(),
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="What is the total cost?"),)),
        assertions=[
            ApplicationEvaluatorAssertionSpec(
                id=f"check-{index}", evaluator_key="answer-check", params=params
            )
            for index in range(assertions)
        ],
    )

    with pytest.raises(ValueError, match="published assertion-detail bytes"):
        EvalCorpusDocument.create(
            target_key="answer-agent",
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            suites=(suite,),
            cases=(case,),
        )


def test_corpus_schema_v6_rejects_earlier_documents():
    document = json.loads(eval_corpus_to_json(_corpus()))
    assert document["schema_version"] == 6
    document["schema_version"] = 5

    with pytest.raises(ValueError, match="unsupported schema_version 5"):
        eval_corpus_from_json(json.dumps(document))


@pytest.mark.parametrize(
    "fields",
    [
        {"key": "Answer Check"},
        {"implementation_revision": "v1"},
        {"implementation_revision": "sha256:" + "A" * 64},
        {"factory": "not callable"},
    ],
)
def test_application_evaluator_requires_portable_key_declared_revision_and_factory(fields):
    values = {"key": "answer-check", "implementation_revision": _REVISION, "factory": _AnswerCheck}

    with pytest.raises(ValidationError):
        ApplicationEvaluator(**{**values, **fields})


def test_targets_copy_application_evaluators_and_reject_duplicate_keys():
    evaluator = _answer_check()
    target = _target(_provider(trials=1), evaluators=(evaluator,))

    assert target.application_evaluators == (evaluator,)
    assert _copy_corpus_target(target).application_evaluators == (evaluator,)
    workflow_target = _workflow_target(_register_app(), (evaluator,))
    assert _copy_corpus_target(workflow_target).application_evaluators == (evaluator,)
    assert "factory" not in evaluator.model_dump()

    with pytest.raises(ValidationError, match="keys must be unique"):
        _target(_provider(trials=1), evaluators=(evaluator, _answer_check()))
    with pytest.raises(TypeError, match="exact ApplicationEvaluator"):
        _target(_provider(trials=1), evaluators=({"key": "answer-check"},))


def test_compilation_builds_one_assertion_per_case_with_a_parameter_copy():
    received: list[Mapping[str, Any]] = []

    def factory(params: Mapping[str, Any]) -> EvalAssertion:
        received.append(params)
        params["expected"] = 0
        return _AnswerCheck(params)

    corpus = _corpus()
    compiled = compile_corpus_suite(
        corpus,
        _target(_provider(trials=1), evaluators=(_answer_check(factory=factory),)),
        _SUITE_ID,
    )

    (assertion,) = compiled.suite.cases[0].assertions
    assert received == [{"expected": 0, "tolerance": 0.005}]
    assert corpus.cases[0].assertions[0].params == {"expected": 42.75, "tolerance": 0.005}
    assert assertion.name == "answer-correct"
    assert assertion.assertion_revision == assertion_spec_revision(corpus.cases[0].assertions[0])
    assert assertion.reads_final_output_only is True


def test_compilation_rejects_an_unknown_evaluator_key_before_dispatch():
    provider = _provider(trials=1)
    corpus = _corpus(_spec(evaluator_key="missing-check"))

    with pytest.raises(ValueError, match="requires application evaluator 'missing-check'"):
        compile_corpus_suite(corpus, _target(provider, evaluators=(_answer_check(),)), _SUITE_ID)
    with pytest.raises(ValueError, match="requires application evaluator 'missing-check'"):
        asyncio.run(
            run_corpus_suite(_target(provider, evaluators=(_answer_check(),)), corpus, _SUITE_ID)
        )
    assert provider.requests == []


def test_factory_failure_fails_compilation_without_echoing_its_message():
    def factory(params: Mapping[str, Any]) -> EvalAssertion:
        raise KeyError("private-reference-value")

    with pytest.raises(ValueError) as caught:
        compile_corpus_suite(
            _corpus(),
            _target(_provider(trials=1), evaluators=(_answer_check(factory=factory),)),
            _SUITE_ID,
        )

    message = str(caught.value)
    assert "'answer-check'" in message
    assert "'answer-correct'" in message
    assert "KeyError" in message
    assert "private-reference-value" not in message


def test_factory_must_build_an_eval_assertion():
    with pytest.raises(ValueError, match="must build an EvalAssertion"):
        compile_corpus_suite(
            _corpus(),
            _target(_provider(trials=1), evaluators=(_answer_check(factory=lambda params: None),)),
            _SUITE_ID,
        )


def test_offline_evaluation_rejects_application_evaluators():
    with pytest.raises(ValueError, match="trusted CorpusTarget evaluator binding"):
        compile_assertion_spec(
            _spec(),
            app=CayuApp(enable_logging=False),
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            trusted_pricing=None,
        )


@pytest.mark.parametrize(
    "output,outcome,message",
    [
        ("42.75", "passed", "The answer is within tolerance."),
        ("70", "failed", "The answer is outside tolerance."),
    ],
)
def test_corpus_target_publishes_the_evaluator_identity_and_observation(output, outcome, message):
    result = _run(
        _target(_provider(trials=1, output=output), evaluators=(_answer_check(),)),
        _corpus(),
    )

    assertion = _only_assertion(result)
    detail = assertion.detail
    assert result.run.status == outcome
    assert assertion.outcome == outcome
    assert assertion.score == (1.0 if outcome == "passed" else 0.0)
    assert type(detail) is PublishedApplicationEvaluatorDetail
    assert detail.evaluator_key == "answer-check"
    assert detail.evaluator_implementation_revision == _REVISION
    assert detail.params == {"expected": 42.75, "tolerance": 0.005}
    assert detail.observation is not None
    assert detail.observation.message == message
    assert detail.observation.metadata == {"expected": "42.75", "observed": output}
    assert corpus_execution_result_from_json(corpus_execution_result_to_json(result)) == result
    assert message in render_corpus_execution_html(result)
    (case,) = present_eval_result(result).cases
    (presented,) = case.trials[0].assertions
    assert presented.kind == "application_evaluator"


def test_published_application_evaluator_results_reject_contradictory_observations():
    result = _run(
        _target(_provider(trials=1), evaluators=(_answer_check(),)),
        _corpus(_spec(params={"expected": 0})),
    )
    document = _only_assertion(result).model_dump(mode="json")
    assert document["outcome"] == "failed"
    errored = {
        **document,
        "outcome": "error",
        "code": "error",
        "message": "Assertion evaluation failed.",
        "score": None,
    }
    reserved = {
        **document,
        "detail": {
            **document["detail"],
            "observation": {"message": "", "metadata": {"cayu.application_evaluator.result": {}}},
        },
    }

    with pytest.raises(ValidationError, match="score is inconsistent"):
        PublishedAssertionResult.model_validate({**document, "score": 1.0})
    with pytest.raises(ValidationError, match="cannot carry observed evidence"):
        PublishedAssertionResult.model_validate(errored)
    with pytest.raises(ValidationError, match="reserved metadata"):
        PublishedAssertionResult.model_validate(reserved)


def test_workflow_target_evaluator_reads_the_projected_structured_output():
    app = _register_app()
    corpus = _corpus(_spec(params={"field": "answer", "expected": 42.75, "tolerance": 0.005}))

    result = _run(_workflow_target(app, (_answer_check(),)), corpus)

    assertion = _only_assertion(result)
    assert result.run.status == "passed"
    assert assertion.detail.evaluator_implementation_revision == _REVISION
    assert assertion.detail.observation.metadata == {"expected": "42.75", "observed": "42.75"}


def test_workflow_target_rejects_an_unknown_evaluator_key():
    corpus = _corpus(_spec(evaluator_key="missing-check"))

    with pytest.raises(ValueError, match="requires application evaluator 'missing-check'"):
        compile_corpus_suite(corpus, _workflow_target(_register_app(), ()), _SUITE_ID)


def test_reserved_metadata_from_the_evaluator_is_replaced_by_the_trusted_identity():
    def factory(params: Mapping[str, Any]) -> EvalAssertion:
        return _ResultAssertion(
            lambda assertion: assertion.passed(
                "checked",
                metadata={
                    "cayu.application_evaluator.result": {
                        "evaluator_key": "forged-check",
                        "implementation_revision": _SECOND_REVISION,
                    },
                    "cayu.model_judge.result": {"judge_profile": {}},
                    "note": "kept",
                },
            )
        )

    result = _run(
        _target(_provider(trials=1), evaluators=(_answer_check(factory=factory),)),
        _corpus(),
    )

    detail = _only_assertion(result).detail
    assert detail.evaluator_key == "answer-check"
    assert detail.evaluator_implementation_revision == _REVISION
    assert detail.observation.metadata == {"note": "kept"}


def test_graded_evaluator_scores_publish_their_threshold():
    def factory(params: Mapping[str, Any]) -> EvalAssertion:
        return _ResultAssertion(
            lambda assertion: assertion.score_result(0.75, threshold=0.5, message="mostly right")
        )

    result = _run(
        _target(_provider(trials=1), evaluators=(_answer_check(factory=factory),)),
        _corpus(),
    )

    assertion = _only_assertion(result)
    assert assertion.outcome == "passed"
    assert assertion.score == 0.75
    assert assertion.detail.threshold == 0.5


def test_evaluator_messages_are_redacted_before_publication():
    def factory(params: Mapping[str, Any]) -> EvalAssertion:
        return _ResultAssertion(
            lambda assertion: assertion.failed(
                "Saw secret-token in the answer.",
                metadata={"observed": "secret-token"},
            )
        )

    result = _run(
        _target(
            _provider(trials=1),
            evaluators=(_answer_check(factory=factory),),
            secret_redactor=SecretRedactor("secret-token"),
        ),
        _corpus(),
    )

    observation = _only_assertion(result).detail.observation
    assert "secret-token" not in observation.message
    assert "secret-token" not in json.dumps(observation.metadata)


def test_evaluator_results_are_redacted_by_the_app_that_ran_the_trial():
    def factory(params: Mapping[str, Any]) -> EvalAssertion:
        return _ResultAssertion(
            lambda assertion: assertion.failed(
                "Saw trial-secret in the answer.",
                metadata={"observed": "trial-secret"},
            )
        )

    def app(secret_redactor: SecretRedactor | None = None) -> CayuApp:
        built = CayuApp(enable_logging=False, secret_redactor=secret_redactor)
        built.register_provider(_StableScriptedProvider([]), default=True)
        built.register_agent(AgentSpec(name="first", model="scripted-model"))
        return built

    # Only the app each trial runs in knows the secret.
    target = _workflow_target(
        app(),
        (_answer_check(factory=factory),),
        trial_app=lambda: app(SecretRedactor("trial-secret")),
    )

    result = _run(target, _corpus())

    assertion = _only_assertion(result)
    assert assertion.outcome == "failed"
    assert "trial-secret" not in assertion.detail.observation.message
    assert "trial-secret" not in json.dumps(assertion.detail.observation.metadata)


def test_a_cost_summary_of_another_session_publishes_an_evaluator_error():
    def factory(params: Mapping[str, Any]) -> EvalAssertion:
        return _ResultAssertion(
            lambda assertion: assertion.passed(
                cost_summary=SessionCostSummary(
                    session_id="another-session",
                    currency="USD",
                    model_steps=0,
                    priced_model_steps=0,
                    unpriced_model_steps=0,
                    total_cost=Decimal("0"),
                )
            )
        )

    result = _run(
        _target(_provider(trials=1), evaluators=(_answer_check(factory=factory),)),
        _corpus(),
    )

    assertion = _only_assertion(result)
    assert assertion.outcome == "error"
    assert assertion.detail.evaluator_key == "answer-check"
    assert assertion.detail.evaluator_implementation_revision == _REVISION


@pytest.mark.parametrize(
    "build",
    [
        lambda assertion: (_ for _ in ()).throw(RuntimeError("evaluator crashed")),
        lambda assertion: assertion.passed("x" * 2_049),
        lambda assertion: assertion.passed("ok", metadata={"blob": "x" * (17 * 1024)}),
        lambda assertion: assertion.passed("ok", metadata={"blob": "x" * (5 * 1024)}),
        lambda assertion: assertion.passed("\N{GRINNING FACE}" * 2_048),
    ],
    ids=[
        "raises",
        "oversized-message",
        "oversized-metadata",
        "metadata-over-evaluator-bound",
        "message-over-its-bytes",
    ],
)
def test_evaluator_failures_publish_errors_with_the_trusted_contract(build):
    def factory(params: Mapping[str, Any]) -> EvalAssertion:
        return _ResultAssertion(build)

    result = _run(
        _target(_provider(trials=1), evaluators=(_answer_check(factory=factory),)),
        _corpus(),
    )

    assertion = _only_assertion(result)
    assert assertion.outcome == "error"
    assert assertion.score is None
    assert assertion.detail.evaluator_implementation_revision == _REVISION
    assert assertion.detail.observation is None


@pytest.mark.parametrize("outcome", [EvalOutcome.ERROR, EvalOutcome.UNAVAILABLE])
def test_blocked_application_evaluator_retains_its_publication_contract(outcome):
    corpus = _corpus(RootStatusAssertionSpec(id="completed", expected="completed"), _spec())
    compiled = compile_corpus_suite(
        corpus,
        _target(_provider(trials=1), evaluators=(_answer_check(),)),
        _SUITE_ID,
    )

    results = _blocked_assertion_results(
        compiled.suite.cases[0].assertions,
        outcome,
        "candidate execution did not produce evaluable evidence",
        memory_attribution_evidence=EvalMemoryAttributionEvidenceV1.unavailable(),
    )
    detail = _published_detail(corpus.cases[0].assertions[1], results[1])

    assert type(detail) is PublishedApplicationEvaluatorDetail
    assert detail.evaluator_implementation_revision == _REVISION
    assert detail.observation is None
    assert detail.threshold is None


def test_comparison_rejects_results_from_a_different_evaluator_revision():
    corpus = _corpus()
    baseline = _run(_target(_provider(trials=1), evaluators=(_answer_check(),)), corpus)
    same = _run(_target(_provider(trials=1), evaluators=(_answer_check(),)), corpus)
    changed = _run(
        _target(_provider(trials=1), evaluators=(_answer_check(revision=_SECOND_REVISION),)),
        corpus,
    )

    assert corpus_execution_compatibility(baseline, same).comparable is True
    compatibility = corpus_execution_compatibility(baseline, changed)
    assert compatibility.comparable is False
    assert CorpusComparisonReason.ASSERTION_CONTRACT_MISMATCH in compatibility.reasons
    assert CorpusComparisonReason.CORPUS_REVISION_MISMATCH not in compatibility.reasons
