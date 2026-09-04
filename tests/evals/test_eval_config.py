from __future__ import annotations

import asyncio

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    CayuConfig,
    EvalCase,
    EvalConfig,
    EvalPlan,
    EvalSuite,
    EvalSuiteTrialPolicyV1,
    FinalOutputContains,
    Message,
    ModelProvider,
    ModelStreamEvent,
    RunRequest,
    run_eval_plan,
    run_eval_suite,
)


class _ConcurrentProvider(ModelProvider):
    name = "concurrent"

    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.active = 0
        self.peak = 0
        self.ready = asyncio.Event()

    async def stream(self, request):
        self.active += 1
        self.peak = max(self.peak, self.active)
        if self.active == self.expected:
            self.ready.set()
        try:
            await asyncio.wait_for(self.ready.wait(), timeout=5)
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
        finally:
            self.active -= 1


@pytest.mark.parametrize(
    ("configured", "explicit", "policy_limit", "expected"),
    [
        (None, None, None, 1),
        (2, None, None, 2),
        (2, 1, None, 1),
        (1, None, 2, 2),
        (2, None, 1, 1),
        (2, 1, 2, 1),
    ],
)
def test_suite_concurrency_precedence_reaches_dispatch(
    configured,
    explicit,
    policy_limit,
    expected,
):
    provider = _ConcurrentProvider(expected)
    app = CayuApp(
        config=None
        if configured is None
        else CayuConfig(evals=EvalConfig(max_concurrency=configured)),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="agent", model="fake"))
    suite = EvalSuite(
        id="concurrency",
        cases=[
            EvalCase(
                id=f"case-{i}",
                request=RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "go")],
                ),
                assertions=[FinalOutputContains("done")],
            )
            for i in range(2)
        ],
    )
    policy = (
        None
        if policy_limit is None
        else EvalSuiteTrialPolicyV1.create(max_concurrency=policy_limit)
    )
    result = asyncio.run(
        run_eval_suite(
            app,
            suite,
            max_concurrency=explicit,
            trial_policy=policy,
        )
    )
    assert result.status == "passed"
    assert provider.peak == expected
    assert all(
        case.trial_policy.max_concurrency == (policy_limit or expected) for case in result.cases
    )


@pytest.mark.parametrize("explicit", [None, 1])
def test_direct_plan_inherits_application_concurrency(explicit):
    expected = 2 if explicit is None else explicit
    provider = _ConcurrentProvider(expected)
    app = CayuApp(config=CayuConfig(evals=EvalConfig(max_concurrency=2)), enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="agent", model="fake"))
    suite = EvalSuite(
        id="plan",
        cases=[
            EvalCase(
                id=f"case-{i}",
                request=RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "go")],
                ),
                assertions=[FinalOutputContains("done")],
            )
            for i in range(2)
        ],
    )
    result = asyncio.run(run_eval_plan(EvalPlan(app=app, suite=suite), max_concurrency=explicit))
    assert result.status == "passed"
    assert provider.peak == expected


def test_explicit_concurrency_cannot_exceed_suite_policy():
    app = CayuApp(config=CayuConfig(evals=EvalConfig(max_concurrency=100)), enable_logging=False)
    suite = EvalSuite(
        id="bounded",
        cases=[
            EvalCase(
                id="case",
                request=RunRequest(agent_name="agent", messages=[]),
                assertions=[FinalOutputContains("done")],
            )
        ],
    )
    with pytest.raises(ValueError, match="exceeds its trial policy"):
        asyncio.run(
            run_eval_suite(
                app,
                suite,
                max_concurrency=2,
                trial_policy=EvalSuiteTrialPolicyV1.create(max_concurrency=1),
            )
        )
