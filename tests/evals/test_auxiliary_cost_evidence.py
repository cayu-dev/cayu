from __future__ import annotations

import asyncio

import pytest
from examples.runtime_auxiliary_inference import MODEL, Summarize

from cayu import AgentSpec, CayuApp, Message, ModelStreamEvent, RunRequest, ScriptedModelProvider
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.evals.assertions import MaxEstimatedCost
from cayu.evals.corpus import EvaluationEvidencePolicySpec, MaxEstimatedCostAssertionSpec
from cayu.evals.evidence import project_assertion_evidence_view
from cayu.evals.models import EvalOutcome
from cayu.evals.portable_assertions import compile_assertion_spec
from cayu.evals.portable_evaluation import evaluate_assertion_spec
from cayu.evals.runner import evaluate_assertions
from cayu.evals.trajectory import trajectory_from_session
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "observed,maximum,expected",
    [
        (False, "14", EvalOutcome.UNAVAILABLE),
        (True, "12", EvalOutcome.FAILED),
        (True, "13", EvalOutcome.PASSED),
        (True, "14", EvalOutcome.PASSED),
    ],
)
def test_managed_auxiliary_cost_survives_eval_projection(
    sqlite_resources, backend, observed, maximum, expected
):
    async def scenario():
        path = sqlite_resources.path("auxiliary-eval.sqlite")
        store = sqlite_resources.own(SQLiteSessionStore(path)) if backend == "sqlite" else None
        app = CayuApp(session_store=store, enable_logging=False)
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        name="summarize", id="parent", arguments={"text": "Long input"}
                    ),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 2, "output_tokens": 1}}),
                ],
                [
                    ModelStreamEvent.text_delta("Short summary"),
                    ModelStreamEvent.completed(
                        {"usage": {"input_tokens": 5, "output_tokens": 2}} if observed else {}
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("Finished"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 2, "output_tokens": 1}}),
                ],
            ]
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model=MODEL), tools=[Summarize()])
        async for _ in app.run(
            RunRequest(
                session_id="auxiliary-eval",
                agent_name="assistant",
                messages=[Message.text("user", "Summarize")],
            )
        ):
            pass
        assert len(provider.requests) == 3
        if store is not None:
            await store.close()
            store = sqlite_resources.own(SQLiteSessionStore(path))
            app = CayuApp(session_store=store, enable_logging=False)
        try:
            trajectory = await trajectory_from_session(app, "auxiliary-eval")
            assert trajectory.usage_summary.model_steps == 2
            pricing = PriceBook(
                prices=(
                    ModelPrice.fixed(
                        provider_name=provider.name,
                        model=MODEL,
                        input_per_million=1_000_000,
                        output_per_million=1_000_000,
                    ),
                )
            )
            policy = EvaluationEvidencePolicySpec.standard()
            spec = MaxEstimatedCostAssertionSpec(id="cost", maximum=maximum, currency="USD")
            compiled = compile_assertion_spec(
                spec, app=app, evidence_policy=policy, trusted_pricing=pricing
            )
            direct, projected = await evaluate_assertions(
                trajectory, (MaxEstimatedCost(maximum, pricing), compiled)
            )
            assert direct.outcome is expected
            assert projected.outcome is expected
            assert direct.cost_summary.model_steps == 2
            assert direct.cost_summary.auxiliary_attempts == 1
            assert direct.cost_summary.unpriced_auxiliary_attempts == int(not observed)
            evidence = project_assertion_evidence_view(
                app, trajectory, evidence_policy=policy, pricing=pricing, cost_currencies=("USD",)
            )
            assert evaluate_assertion_spec(spec, evidence).outcome is expected
            assert len(evidence.costs) == int(observed)
        finally:
            if store is not None:
                await store.close()

    async def run():
        async with sqlite_resources:
            await scenario()

    asyncio.run(run())
