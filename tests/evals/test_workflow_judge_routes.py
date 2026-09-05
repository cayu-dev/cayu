"""Workflow judge admission must follow executed routes, independently of the probe."""

from __future__ import annotations

import asyncio

import pytest
from tests.evals.test_corpus_execution import _model_judge_corpus, _model_judge_target
from tests.evals.test_structured_model_judge import _corpus as _structured_corpus
from tests.evals.test_structured_model_judge import _judge, _judgment
from tests.evals.test_workflow_eval_target import (
    _corpus,
    _NoChildWorkflow,
    _target,
    _TwoChildWorkflow,
)

from cayu import (
    AgentSpec,
    CayuApp,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    corpus_execution_result_from_json,
    corpus_execution_result_to_json,
)
from cayu.evals.execution import compile_corpus_suite, run_corpus_suite


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("probe_model", ["model-a", "scripted-model"])
@pytest.mark.parametrize(
    "judge_model,allow_same,second_model,no_children,relation,dispatched",
    [
        ("scripted-model", False, "scripted-model", False, "same_model", False),
        ("scripted-model", True, "scripted-model", False, "same_model", True),
        ("model-c", False, "scripted-model", False, "independent_model", True),
        ("scripted-model", False, "model-c", False, "same_model", False),
        ("model-d", False, "model-c", False, "independent_model", True),
        ("scripted-model", False, "scripted-model", True, "unknown", False),
        ("scripted-model", True, "scripted-model", True, "unknown", False),
    ],
)
def test_workflow_judges_use_executed_routes(
    structured,
    probe_model,
    judge_model,
    allow_same,
    second_model,
    no_children,
    relation,
    dispatched,
):
    if structured:
        judge, judge_provider = _judge(_judgment(), model=judge_model, allow_same_model=allow_same)
        spec = _structured_corpus(judge).cases[0].assertions[0]
    else:
        judge, judge_provider = _model_judge_target(model=judge_model, allow_same_model=allow_same)
        spec = _model_judge_corpus(judge).cases[0].assertions[0]
    candidate_provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
            for _ in range(2)
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(candidate_provider, default=True)
    app.register_agent(AgentSpec(name="first", model="scripted-model"))
    app.register_agent(AgentSpec(name="second", model=second_model))
    app.register_agent(AgentSpec(name="probe", model=probe_model))
    # The workflow still runs first and second; the probe is never executed.
    target = _target(app, _NoChildWorkflow if no_children else _TwoChildWorkflow).model_copy(
        update={
            "request_base": RunRequest(agent_name="probe", messages=[]),
            "model_judges": (judge,),
        }
    )
    corpus = _corpus(spec)
    compile_corpus_suite(corpus, target, "workflow-suite")
    result = asyncio.run(run_corpus_suite(target, corpus, "workflow-suite"))
    assertion = result.run.cases[0].trials[0].assertions[0]
    assert assertion.detail.candidate_route_relation == relation
    assert assertion.outcome == ("passed" if dispatched else "unavailable")
    assert assertion.detail.diagnostic == (
        "judgment_recorded" if dispatched else "evidence_unavailable"
    )
    assert len(candidate_provider.requests) == (0 if no_children else 2)
    assert len(judge_provider.requests) == int(dispatched)
    assert corpus_execution_result_from_json(corpus_execution_result_to_json(result)) == result
    if dispatched:
        detail = assertion.detail.model_dump(mode="python")
        with pytest.raises(ValueError, match="forbidden"):
            type(assertion.detail).model_validate({**detail, "candidate_route_relation": "unknown"})
        if not allow_same:
            with pytest.raises(ValueError, match="forbidden"):
                type(assertion.detail).model_validate(
                    {**detail, "candidate_route_relation": "same_model"}
                )


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("missing", ["child_tree", "child_session", "model_start", "route_field"])
def test_workflow_judge_replay_withholds_dispatch_for_incomplete_routes(structured, missing):
    from tests.evals.test_workflow_eval_target import _register_app, _suite

    from cayu import evaluate_assertions, run_workflow_eval_suite
    from cayu.core.events import EventType

    if structured:
        judge, judge_provider = _judge(_judgment(), allow_same_model=True)
        spec = _structured_corpus(judge).cases[0].assertions[0]
    else:
        judge, judge_provider = _model_judge_target(allow_same_model=True)
        spec = _model_judge_corpus(judge).cases[0].assertions[0]
    app = _register_app(
        [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
            for _ in range(2)
        ]
    )
    target = _target(app, _TwoChildWorkflow).model_copy(update={"model_judges": (judge,)})
    result = asyncio.run(run_workflow_eval_suite(target, _suite(), retain_trajectory=True))
    trajectory = result.cases[0].trials[0].trajectory
    assert trajectory is not None
    child = trajectory.children[0]
    if missing == "child_tree":
        child = child.model_copy(update={"children_incomplete": True})
    elif missing == "child_session":
        child = child.model_copy(update={"session": None})
    elif missing == "model_start":
        child = child.model_copy(
            update={
                "events": tuple(
                    event for event in child.events if event.type != EventType.MODEL_STARTED
                )
            }
        )
    else:
        child = child.model_copy(
            update={
                "events": tuple(
                    event.model_copy(update={"payload": {**event.payload, "model": None}})
                    if event.type == EventType.MODEL_STARTED
                    else event
                    for event in child.events
                )
            }
        )
    trajectory = trajectory.model_copy(update={"children": (child, *trajectory.children[1:])})
    compiled = compile_corpus_suite(_corpus(spec), target, "workflow-suite")
    results = asyncio.run(evaluate_assertions(trajectory, compiled.suite.cases[0].assertions))
    assert results[0].outcome == "unavailable"
    record = next(iter(results[0].metadata.values()))
    assert record["candidate_route_relation"] == "unknown"
    assert judge_provider.requests == []
