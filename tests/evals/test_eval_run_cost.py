from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from decimal import Decimal
from pathlib import Path

import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets import default_price_book
from cayu.cli import main
from cayu.cli._eval_processes import _worker_price_book
from cayu.evals.assertions import FinalOutputContains
from cayu.evals.corpus import EvaluationEvidencePolicySpec, pricing_profile_identity
from cayu.evals.evidence import _validated_pricing
from cayu.evals.execution import WorkflowEvalTarget
from cayu.evals.models import EvalRun
from cayu.evals.reporting import (
    compare_eval_runs,
    comparison_to_json,
    eval_run_to_json,
    load_eval_run,
    render_comparison_html,
    render_html_report,
)
from cayu.evals.run_cost import (
    EvalRunCost,
    EvalTrialCost,
    eval_cost_difference,
    eval_run_cost,
    eval_trial_cost,
    plain_cost_amount,
)
from cayu.evals.runner import (
    EvalCase,
    EvalPlan,
    EvalSuite,
    run_eval_plan,
    run_eval_suite,
    run_workflow_eval_suite,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.evals.workflow_target import WorkflowEvalExecution, WorkflowEvalResult
from cayu.events import Event, EventType
from cayu.messages import Message, TextPart
from cayu.providers.base import ModelRequest, ModelStreamEvent, UsageDialect
from cayu.sessions.requests import RunRequest
from cayu.workflows.base import WorkflowSpec
from cayu.workflows.models import StepError
from cayu.workflows.workflow import WorkflowBase, step

# A model the bundled price book prices; the scripted route stands in for it.
_PRICED_MODEL = "gpt-5.6-luna"
_REVISION = "sha256:" + "1" * 64


def _completion(input_tokens: int, output_tokens: int) -> tuple[ModelStreamEvent, ...]:
    return (
        ModelStreamEvent.text_delta("done"),
        ModelStreamEvent.completed(
            {
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
            }
        ),
    )


def _completion_without_usage() -> tuple[ModelStreamEvent, ...]:
    return (ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed())


def _app(*responses: tuple[ModelStreamEvent, ...], priced: bool = True) -> CayuApp:
    batches = [list(response) for response in responses]
    provider = (
        ScriptedModelProvider(
            batches, billing_provider_name="openai", usage_dialect=UsageDialect.OPENAI
        )
        if priced
        else ScriptedModelProvider(batches)
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="agent", model=_PRICED_MODEL))
    return app


def _suite() -> EvalSuite:
    return EvalSuite(
        id="cost-suite",
        cases=[
            EvalCase(
                id="answer",
                request=RunRequest(agent_name="agent", messages=[Message.text("user", "hi")]),
                assertions=[FinalOutputContains("done")],
            )
        ],
    )


def _session_cost(app: CayuApp, session_id: str | None) -> Decimal:
    assert session_id is not None
    return asyncio.run(app.get_session_cost(session_id, default_price_book())).total_cost


def _priced_run(input_tokens: int, output_tokens: int) -> EvalRun:
    app = _app(_completion(input_tokens, output_tokens))
    return asyncio.run(run_eval_suite(app, _suite(), price_book=default_price_book()))


def test_a_priced_run_prices_each_trial_and_carries_the_total() -> None:
    app = _app(_completion(1000, 100), _completion(3000, 300))

    run = asyncio.run(run_eval_suite(app, _suite(), trials=2, price_book=default_price_book()))

    identity = pricing_profile_identity(default_price_book())
    amounts = []
    for trial in run.cases[0].trials:
        assert trial.cost is not None
        assert trial.cost.availability == "observed"
        assert trial.cost.model_steps == trial.cost.priced_model_steps == 1
        # The same estimate the application's own session cost reports.
        assert trial.cost.estimated_cost == _session_cost(app, trial.session_id)
        amounts.append(trial.cost.estimated_cost)
    first, second = amounts
    assert first is not None and second is not None
    assert 0 < first < second
    assert run.cost == EvalRunCost(
        availability="observed",
        price_book_version=identity.price_book_version,
        pricing_profile_fingerprint=identity.fingerprint,
        currency="USD",
        estimated_cost=first + second,
        observed_trials=2,
    )


def test_without_a_price_book_cost_is_unavailable_for_that_reason() -> None:
    run = asyncio.run(run_eval_suite(_app(_completion(1000, 100)), _suite()))

    assert run.cost.availability == "unavailable"
    assert run.cost.unavailable_reason == "no_price_book"
    assert run.cases[0].trials[0].cost == EvalTrialCost(
        availability="unavailable", unavailable_reason="no_price_book"
    )
    document = json.loads(eval_run_to_json(run))
    assert document["cost"]["unavailable_reason"] == "no_price_book"
    assert document["cases"][0]["trials"][0]["cost"]["unavailable_reason"] == "no_price_book"
    assert "Cost: unavailable: no price book was selected" in render_html_report(run)


def test_usage_the_price_book_cannot_price_leaves_the_cost_unavailable() -> None:
    run = asyncio.run(
        run_eval_suite(
            _app(_completion(1000, 100), priced=False), _suite(), price_book=default_price_book()
        )
    )

    trial_cost = run.cases[0].trials[0].cost
    assert trial_cost is not None
    assert trial_cost.unavailable_reason == "unpriced_usage"
    assert (trial_cost.model_steps, trial_cost.unpriced_model_steps) == (1, 1)
    assert run.cost.unavailable_reason == "unpriced_usage"
    assert run.cost.estimated_cost is None
    assert run.cost.observed_trials == 0


def test_a_trial_without_recorded_usage_keeps_the_rest_of_the_run_observed() -> None:
    app = _app(_completion(1000, 100), _completion_without_usage())

    run = asyncio.run(run_eval_suite(app, _suite(), trials=2, price_book=default_price_book()))

    priced, unrecorded = run.cases[0].trials
    assert priced.cost is not None and priced.cost.availability == "observed"
    assert priced.cost.estimated_cost is not None
    assert unrecorded.cost is not None
    assert unrecorded.cost.unavailable_reason == "usage_unavailable"
    assert run.cost.unavailable_reason == "usage_unavailable"
    assert run.cost.observed_trials == 1
    assert run.cost.estimated_cost == priced.cost.estimated_cost
    report = render_html_report(run)
    assert (
        "Cost: unavailable: usage wasn&#x27;t completely recorded; "
        f"{plain_cost_amount(priced.cost.estimated_cost)} USD observed across 1 of 2 trials"
    ) in report


def test_a_standalone_hosted_tool_charge_without_a_price_leaves_the_cost_unavailable() -> None:
    # A hosted web search that no model completion carries is priced on its own.
    search = Event(
        type=EventType.MODEL_HOSTED_TOOL_CALL,
        session_id="session",
        payload={
            "tool_type": "web_search",
            "status": "completed",
            "provider_name": "openai",
            "model": "unpriced-model",
        },
    )

    cost = eval_trial_cost(
        _validated_pricing(default_price_book()),
        session_id="session",
        events=[search],
        evidence_complete=True,
    )

    assert (cost.model_steps, cost.unpriced_model_steps) == (0, 0)
    assert cost.unavailable_reason == "unpriced_usage"


def test_a_saved_run_keeps_its_cost_and_rejects_a_changed_total(tmp_path: Path) -> None:
    run = _priced_run(1000, 100)
    path = tmp_path / "run.json"
    path.write_text(eval_run_to_json(run), encoding="utf-8")

    assert load_eval_run(path).cost == run.cost

    document = json.loads(path.read_text(encoding="utf-8"))
    document["cost"]["estimated_cost"] = "1"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Run cost does not match its trials"):
        load_eval_run(path)


def test_a_run_saved_before_cost_was_recorded_says_so(tmp_path: Path) -> None:
    document = json.loads(eval_run_to_json(_priced_run(1000, 100)))
    del document["cost"]
    for case in document["cases"]:
        for trial in case["trials"]:
            del trial["cost"]
    path = tmp_path / "older.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    assert load_eval_run(path).cost == EvalRunCost.not_recorded()


def test_a_saved_cost_cannot_carry_an_unbounded_amount(tmp_path: Path) -> None:
    run = asyncio.run(
        run_eval_suite(_app(_completion_without_usage()), _suite(), price_book=default_price_book())
    )
    document = json.loads(eval_run_to_json(run))
    trial_cost = document["cases"][0]["trials"][0]["cost"]
    assert trial_cost["unavailable_reason"] == "usage_unavailable"
    # A few bytes that plain notation would expand to a billion characters.
    trial_cost["estimated_cost"] = "1e-1000000000"
    path = tmp_path / "huge.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="within its digit bounds"):
        load_eval_run(path)
    with pytest.raises(ValueError, match="within its digit bounds"):
        plain_cost_amount(Decimal("1e+1000000000"))


def test_trials_priced_with_different_books_cannot_form_one_run() -> None:
    priced = _priced_run(1000, 100).cases[0].trials[0].cost
    assert priced is not None
    other_book = priced.model_copy(update={"pricing_profile_fingerprint": _REVISION})

    with pytest.raises(ValueError, match="same price book"):
        eval_run_cost([priced, other_book])


def test_compare_shows_both_costs_and_the_difference() -> None:
    baseline = _priced_run(4000, 400)
    current = _priced_run(1000, 100)

    comparison = compare_eval_runs(baseline, current)

    assert comparison.baseline_cost == baseline.cost
    assert comparison.current_cost == current.cost
    assert baseline.cost.estimated_cost is not None and current.cost.estimated_cost is not None
    difference = current.cost.estimated_cost - baseline.cost.estimated_cost
    assert difference < 0
    assert comparison.cost_difference.availability == "observed"
    assert comparison.cost_difference.estimated_cost == difference
    assert json.loads(comparison_to_json(comparison))["cost_difference"]["availability"] == (
        "observed"
    )
    assert f"Cost difference: {plain_cost_amount(difference)} USD (observed)" in (
        render_comparison_html(comparison)
    )


def test_the_difference_is_exact_beyond_the_default_decimal_precision() -> None:
    cost = _priced_run(1000, 100).cost.model_copy(
        update={"estimated_cost": Decimal("0.12345678901234567890123456789")}
    )

    assert eval_cost_difference(cost, cost).estimated_cost == 0


def test_compare_says_why_the_difference_is_unavailable() -> None:
    baseline = asyncio.run(run_eval_suite(_app(_completion(1000, 100)), _suite()))
    current = _priced_run(1000, 100)

    comparison = compare_eval_runs(baseline, current)

    assert comparison.cost_difference.unavailable_reason == "baseline_unavailable"
    rendered = render_comparison_html(comparison)
    assert "Cost: unavailable: no price book was selected -&gt; " in rendered
    assert "Cost difference: unavailable: the baseline cost is unavailable" in rendered


def test_a_workflow_plan_prices_with_its_target_book() -> None:
    plan = EvalPlan(workflow_target=_workflow_target(_workflow_app()), suite=_suite())

    with pytest.raises(ValueError, match="prices with its target's price_book"):
        asyncio.run(run_eval_plan(plan, price_book=default_price_book()))


class _OneStepWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="one-step-cost")

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        answer = await step(ctx, agent="agent", step_id="answer", prompt="answer")
        yield await ctx.completed({"answer": answer.text})


def _workflow_app() -> CayuApp:
    # Workflow trials own their routes, so the script answers per request.
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            response_factory=lambda request: _completion(1000, 100),
            billing_provider_name="openai",
            usage_dialect=UsageDialect.OPENAI,
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="agent", model=_PRICED_MODEL))
    return app


def _workflow_target(
    app: CayuApp,
    *,
    price_book=None,
    workflow_type: type[WorkflowBase] = _OneStepWorkflow,
) -> WorkflowEvalTarget:
    return WorkflowEvalTarget(
        key="workflow-cost",
        app=app,
        request_base=RunRequest(agent_name="agent", messages=[]),
        application_release_id="workflow-release",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        price_book=price_book,
        workflow_spec=workflow_type.spec,
        implementation_revision=_REVISION,
        result_projector_revision=_REVISION,
        execution_scope_revision=_REVISION,
        workflow_factory=lambda invocation: WorkflowEvalExecution(
            app=app, workflow=workflow_type(app)
        ),
        result_projector=lambda evidence: WorkflowEvalResult(
            final_output=evidence.completion_event.payload["answer"]
        ),
    )


def test_a_workflow_run_prices_its_child_sessions_with_the_target_book() -> None:
    app = _workflow_app()

    run = asyncio.run(
        run_workflow_eval_suite(
            _workflow_target(app, price_book=default_price_book()),
            _suite(),
            retain_trajectory=True,
        )
    )

    trial = run.cases[0].trials[0]
    assert trial.trajectory is not None
    (child,) = trial.trajectory.children
    assert child.session is not None
    assert trial.cost is not None and trial.cost.availability == "observed"
    assert trial.cost.estimated_cost == _session_cost(app, child.session.id)
    assert run.cost.availability == "observed"
    assert run.cost.estimated_cost == trial.cost.estimated_cost


class _ToleratesAFailedStepWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="tolerates-failed-step")

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        with suppress(StepError):
            await step(ctx, agent="agent", step_id="flaky", prompt="fail")
        answer = await step(ctx, agent="agent", step_id="answer", prompt="answer")
        yield await ctx.completed({"answer": answer.text})


def _fail_the_flaky_step(request: ModelRequest) -> tuple[ModelStreamEvent, ...]:
    prompt = request.messages[-1].content[0]
    if isinstance(prompt, TextPart) and prompt.text == "fail":
        return (ModelStreamEvent.error("provider failed"),)
    return _completion(1000, 100)


def test_a_workflow_model_call_without_recorded_usage_leaves_the_cost_unavailable() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            response_factory=_fail_the_flaky_step,
            billing_provider_name="openai",
            usage_dialect=UsageDialect.OPENAI,
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="agent", model=_PRICED_MODEL))
    target = _workflow_target(
        app, price_book=default_price_book(), workflow_type=_ToleratesAFailedStepWorkflow
    )

    run = asyncio.run(run_workflow_eval_suite(target, _suite()))

    trial = run.cases[0].trials[0]
    assert trial.status.value == "passed"
    assert trial.cost is not None
    # The failed call's spend is unknown, so only the completed call is priced.
    assert trial.cost.unavailable_reason == "usage_unavailable"
    assert run.cost.unavailable_reason == "usage_unavailable"


def test_process_workers_identify_a_workflow_target_by_its_own_book() -> None:
    target = _workflow_target(_workflow_app(), price_book=default_price_book())

    assert _worker_price_book(EvalPlan(workflow_target=target, suite=_suite())) == (
        target.price_book
    )


def build_priced_eval_plan() -> EvalPlan:
    return EvalPlan(app=_app(_completion(1000, 100)), suite=_suite())


def test_eval_run_prices_with_the_project_eval_price_book(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target = f"{__name__}:build_priced_eval_plan"
    assert main(["eval", "run", target, "--output", "unpriced.json"]) == 0
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu.evals]\nprice_book = "bundled-public"\n', encoding="utf-8"
    )
    assert main(["eval", "run", target, "--output", "priced.json"]) == 0

    assert load_eval_run(tmp_path / "unpriced.json").cost.unavailable_reason == "no_price_book"
    priced = load_eval_run(tmp_path / "priced.json").cost
    assert priced.availability == "observed"
    assert priced.price_book_version == default_price_book().price_book_version
    assert main(["eval", "compare", "priced.json", "priced.json", "--output", "same.json"]) == 0
    difference = json.loads((tmp_path / "same.json").read_text(encoding="utf-8"))
    assert difference["cost_difference"] == {
        "availability": "observed",
        "currency": "USD",
        "estimated_cost": "0",
        "unavailable_reason": None,
    }
