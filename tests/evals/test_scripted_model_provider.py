from __future__ import annotations

import asyncio

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    ModelRequest,
    ModelStreamEvent,
    ModelTarget,
    RunRequest,
    ScriptedModelProvider,
    StructuredOutputSpec,
    Tool,
    ToolResult,
    ToolSpec,
)
from cayu.evals import EvalCase, EvalPlan, EvalSuite, run_eval_case, run_eval_plan, run_eval_suite
from cayu.evals.assertions import ChildSessionCompleted, FinalOutputContains, SessionCompleted
from cayu.evals.judges import LLMJudge
from cayu.providers import ProviderOperationStartRequest
from cayu.tools.subagents import (
    BackgroundSubagentTaskRegistry,
    SubagentExecutionMode,
    SubagentSpec,
    SubagentTool,
)


def completed(text: str) -> tuple[ModelStreamEvent, ...]:
    return (ModelStreamEvent.text_delta(text), ModelStreamEvent.completed())


def request(text: str) -> ModelRequest:
    return ModelRequest(model="test", messages=[Message.text("user", text)])


def test_request_aware_factory_selects_from_request_and_records_copy() -> None:
    provider = ScriptedModelProvider(
        response_factory=lambda req: completed(req.messages[-1].content[0].text.upper()),
        supports_native_structured_output=True,
    )
    first = request("alpha")

    async def collect():
        return [event async for event in provider.stream(first)]

    events = asyncio.run(collect())
    assert events[0].delta == "ALPHA"
    assert provider.supports_native_structured_output is True
    assert provider.requests == [first]
    assert provider.requests[0] is not first


async def collect_bad(provider, req):
    return [event async for event in provider.stream(req)]


def test_request_aware_factory_requires_complete_batch() -> None:
    provider = ScriptedModelProvider(
        response_factory=lambda req: (ModelStreamEvent.text_delta("bad"),)
    )
    with pytest.raises(ValueError, match="end with a COMPLETED"):
        asyncio.run(collect_bad(provider, request("x")))


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("cancel_child", [False, True])
def test_nested_drain_restores_live_ancestor_reservation(background, cancel_child):
    async def exercise():
        children = []
        observed = []

        class Outer(SessionCompleted):
            async def evaluate(self, context):
                provider = ScriptedModelProvider(
                    [completed("middle"), completed("next")], background=background
                )
                observed.append(provider)
                root = provider._active_eval_invocation
                rejections = []
                original_consume = provider._consume_batch

                def observe_consumption(req):
                    try:
                        return original_consume(req)
                    except ValueError as exc:
                        rejections.append(str(exc))
                        raise

                provider._consume_batch = observe_consumption
                app = app_for(provider)
                entered, release = asyncio.Event(), asyncio.Event()
                original_run = app.run

                async def blocked(req):
                    if req.messages[-1].content[0].text == "draining":
                        entered.set()
                        await release.wait()
                    async for event in original_run(req):
                        yield event

                app.run = blocked

                class Middle(SessionCompleted):
                    async def evaluate(self, ctx):
                        child = asyncio.create_task(
                            run_eval_case(
                                app,
                                EvalCase(
                                    id="draining",
                                    request=run_request("draining"),
                                    assertions=[SessionCompleted()],
                                ),
                                suite_id="draining",
                            )
                        )
                        children.append(child)
                        await asyncio.wait_for(entered.wait(), 10)
                        return await super().evaluate(ctx)

                result = await run_eval_case(
                    app,
                    EvalCase(id="middle", request=run_request("middle"), assertions=[Middle()]),
                    suite_id="middle",
                )
                assert result.status.value == "passed"
                assert provider._active_eval_invocation is not root
                with pytest.raises(ValueError, match="Concurrent eval invocations"):
                    await anext(provider.stream(request("cannot steal while draining")))
                child = children[0]
                if cancel_child:
                    child.cancel()
                    assert child.cancelling() == 1
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(child, 10)
                    assert child.cancelled()
                else:
                    rejections.clear()
                    release.set()
                    rejected = await asyncio.wait_for(child, 10)
                    assert rejected.status.value == "failed"
                    assert rejections == [
                        "A completed eval trial cannot consume positional scripted batches."
                    ]
                    assert len(provider.requests) == 1
                assert provider._active_eval_invocation is root
                assert (
                    await run_eval_case(
                        app,
                        EvalCase(
                            id="next", request=run_request("next"), assertions=[SessionCompleted()]
                        ),
                        suite_id="next",
                    )
                ).status.value == "passed"
                return await super().evaluate(context)

        try:
            result = await run_eval_suite(
                app_for(ScriptedModelProvider(response_factory=lambda req: completed("outer"))),
                EvalSuite(
                    id="outer",
                    cases=[
                        EvalCase(id="outer", request=run_request("outer"), assertions=[Outer()])
                    ],
                ),
            )
            assert result.status.value == "passed", result.cases[0].trials[0].error
            assert observed[0]._active_eval_invocation is None
        finally:
            for child in children:
                if not child.done():
                    child.cancel()
            await asyncio.gather(*children, return_exceptions=True)

    asyncio.run(exercise())


def test_events_and_factory_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="events or response_factory"):
        ScriptedModelProvider(completed("x"), response_factory=lambda req: completed("y"))


def test_native_support_defaults_to_false() -> None:
    assert ScriptedModelProvider(completed("x")).supports_native_structured_output is False


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("cancel_child", [False, True])
def test_nested_eval_delegates_exclusively_and_restores_parent(background, cancel_child):
    observed = []

    class Nested(SessionCompleted):
        async def evaluate(self, context):
            provider = ScriptedModelProvider(
                [completed("child"), completed("child")], background=background
            )
            observed.append(provider)
            parent = provider._active_eval_invocation
            inner = app_for(provider)
            entered, release = asyncio.Event(), asyncio.Event()
            original_run = inner.run

            async def blocked(req):
                entered.set()
                await release.wait()
                async for event in original_run(req):
                    yield event

            inner.run = blocked
            case = EvalCase(
                id="child", request=run_request("child"), assertions=[SessionCompleted()]
            )
            child = asyncio.create_task(run_eval_case(inner, case, suite_id="nested"))
            try:
                await asyncio.wait_for(entered.wait(), 10)
                assert provider._active_eval_invocation is not parent
                with pytest.raises(ValueError, match="Concurrent eval invocations"):
                    await run_eval_suite(inner, EvalSuite(id="sibling", cases=[case]))
                with pytest.raises(ValueError, match="Concurrent eval invocations"):
                    await anext(provider.stream(request("parent cannot steal")))
                assert provider.requests == []
                if cancel_child:
                    child.cancel()
                    assert child.cancelling() == 1
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(child, 10)
                    assert child.cancelled()
                else:
                    release.set()
                    assert (await asyncio.wait_for(child, 10)).status.value == "passed"
                assert provider._active_eval_invocation is parent
                release.set()
                assert (
                    await run_eval_suite(inner, EvalSuite(id="next-child", cases=[case]))
                ).status.value == "passed"
                assert provider._active_eval_invocation is parent
                return await super().evaluate(context)
            finally:
                release.set()
                if not child.done():
                    child.cancel()
                await asyncio.gather(child, return_exceptions=True)

    outer = app_for(ScriptedModelProvider(response_factory=lambda req: completed("outer")))
    result = asyncio.run(
        run_eval_suite(
            outer,
            EvalSuite(
                id="outer",
                cases=[EvalCase(id="outer", request=run_request("outer"), assertions=[Nested()])],
            ),
        )
    )
    assert result.status.value == "passed", result.cases[0].trials[0].error
    assert len(observed) == 1
    assert observed[0]._active_eval_invocation is None


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_checkpoint_recovery_ignores_busy_completed_only_providers(
    monkeypatch, background, partial
):
    from cayu.evals.runner import _run_eval_suite_with_public_projection
    from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1

    async def exercise():
        first = ScriptedModelProvider([completed("a"), completed("busy")], background=background)
        second = ScriptedModelProvider(completed("b"), name="second", background=background)
        app = app_for(first)
        app.register_provider(second)
        suite = EvalSuite(
            id="recovery",
            cases=[
                EvalCase(id="a", request=run_request("a"), assertions=[SessionCompleted()]),
                EvalCase(
                    id="b",
                    request=run_request("b").model_copy(
                        update={"target": ModelTarget(provider_name="second", model="test")}
                    ),
                    assertions=[SessionCompleted()],
                ),
            ],
        )
        retained = {}

        async def checkpoint(case_id, result, public_data):
            retained[(case_id, result.trial_number)] = (result, public_data)
            if case_id == ("a" if partial else "b"):
                raise RuntimeError("publication lost after checkpoint")

        async def recover(**kwargs):
            return await _run_eval_suite_with_public_projection(
                app,
                suite,
                max_concurrency=1,
                case_timeout_seconds=15,
                trials=1,
                trial_policy=EvalSuiteTrialPolicyV1.create(trial_count=1, max_concurrency=1),
                output_preview_bytes=100,
                **kwargs,
            )

        with pytest.raises(RuntimeError, match="publication lost"):
            await recover(trial_completed=checkpoint)
        entered, release = asyncio.Event(), asyncio.Event()
        original_run = app.run

        async def blocked(req):
            if req.messages[-1].content[0].text == "busy":
                entered.set()
                await release.wait()
            async for event in original_run(req):
                yield event

        monkeypatch.setattr(app, "run", blocked)
        busy = asyncio.create_task(
            run_eval_case(
                app,
                EvalCase(id="busy", request=run_request("busy"), assertions=[SessionCompleted()]),
                suite_id="busy",
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 10)
            owner = first._active_eval_invocation
            result, public = await recover(completed_trials=retained)
            assert result.status.value == "passed"
            assert set(public) == {"a", "b"}
            assert first._active_eval_invocation is owner
            assert len(first.requests) == 1
            assert len(second.requests) == 1
            release.set()
            assert (await asyncio.wait_for(busy, 10)).status.value == "passed"
        finally:
            release.set()
            if not busy.done():
                busy.cancel()
            await asyncio.gather(busy, return_exceptions=True)

    asyncio.run(exercise())


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("corrupt", [None, "boolean_key", "boolean_result"])
def test_recovery_sharing_counts_only_validated_pending_trials(background, corrupt):
    from cayu.evals.runner import _run_eval_suite_with_public_projection
    from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1

    async def exercise():
        provider = ScriptedModelProvider([completed("ok"), completed("ok")], background=background)
        app = app_for(provider)
        suite = EvalSuite(
            id="repeated",
            cases=[EvalCase(id="a", request=run_request("a"), assertions=[SessionCompleted()])],
        )
        retained = {}

        async def checkpoint(case_id, result, public_data):
            retained[(case_id, result.trial_number)] = (result, public_data)
            raise RuntimeError("checkpoint persisted")

        async def execute(concurrency, **kwargs):
            return await _run_eval_suite_with_public_projection(
                app,
                suite,
                max_concurrency=concurrency,
                case_timeout_seconds=15,
                trials=2,
                trial_policy=EvalSuiteTrialPolicyV1.create(trial_count=2, max_concurrency=2),
                output_preview_bytes=100,
                **kwargs,
            )

        with pytest.raises(RuntimeError, match="checkpoint persisted"):
            await execute(1, trial_completed=checkpoint)
        assert len(provider.requests) == 1
        if corrupt == "boolean_key":
            retained = {("a", True): retained[("a", 1)]}
        elif corrupt == "boolean_result":
            result, public = retained[("a", 1)]
            retained[("a", 1)] = (result.model_copy(update={"trial_number": True}), public)
        if corrupt is not None:
            with pytest.raises((ValueError, TypeError)):
                await execute(2, completed_trials=retained)
            assert len(provider.requests) == 1
            assert provider._active_eval_invocation is None
        else:
            result, _ = await execute(2, completed_trials=retained)
            assert result.status.value == "passed"
            assert len(provider.requests) == 2

    asyncio.run(exercise())


@pytest.mark.parametrize("background", [False, True])
def test_factory_mutation_cannot_rewrite_request_history(background):
    def respond(req):
        req.options["nested"]["value"] = "changed"
        req.messages.clear()
        return completed("done")

    provider = ScriptedModelProvider(response_factory=respond, background=background)
    req = request("original")
    req.options = {"nested": {"value": "original"}}

    async def exercise():
        if background:
            connection = await provider.provider_operations.start(
                ProviderOperationStartRequest(request=req, idempotency_key="copy-test")
            )
            return [event async for event in connection.events]
        return [event async for event in provider.stream(req)]

    assert asyncio.run(exercise())[0].delta == "done"
    assert req.options == {"nested": {"value": "original"}}
    assert provider.requests[0].options == req.options
    assert provider.requests[0].messages == req.messages


@pytest.mark.parametrize("background", [False, True])
def test_invalid_factory_batch_has_no_partial_public_output(background):
    provider = ScriptedModelProvider(
        response_factory=lambda req: [ModelStreamEvent.text_delta("must not escape")],
        background=background,
    )
    app = app_for(provider)

    async def exercise():
        events = [event async for event in app.run(run_request("invalid"))]
        # Background start failures use the runtime's conservative interrupted
        # classification, since no operation acknowledgement was returned.
        terminal = EventType.SESSION_INTERRUPTED if background else EventType.SESSION_FAILED
        assert any(event.type is terminal for event in events)
        assert not any(event.type is EventType.SESSION_COMPLETED for event in events)
        assert all("must not escape" not in str(event.payload) for event in events)
        assert len(provider.requests) == 1
        assert provider.background_operation_ids == ()

    asyncio.run(exercise())


def app_for(provider):
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="test"))
    return app


def run_request(text):
    return RunRequest(agent_name="assistant", messages=[Message.text("user", text)])


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("repeated", [False, True])
def test_finished_trial_revokes_background_subagent_consumption(monkeypatch, background, repeated):
    async def exercise():
        provider = ScriptedModelProvider(
            [
                (
                    ModelStreamEvent.tool_call(
                        id="spawn", name="subagent", arguments={"agent": "worker", "task": "work"}
                    ),
                    ModelStreamEvent.completed(),
                ),
                completed("A"),
                completed("B"),
            ],
            background=background,
        )
        app = CayuApp(enable_logging=False)
        registry = BackgroundSubagentTaskRegistry()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="test"))
        app.register_agent(
            AgentSpec(name="assistant", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={
                        "worker": SubagentSpec(
                            agent_name="worker", mode=SubagentExecutionMode.BACKGROUND
                        )
                    },
                    background_registry=registry,
                )
            ],
        )
        entered, release = asyncio.Event(), asyncio.Event()
        root_ids, child_events, rejections = [], [], []
        original_run = app.run
        original_consume = provider._consume_batch

        def observe_consumption(req):
            try:
                return original_consume(req)
            except ValueError as exc:
                rejections.append(str(exc))
                raise

        async def controlled_run(req):
            if req.agent_name == "worker":
                async for event in original_run(req):
                    child_events.append(event)
                    yield event
                    if event.type is EventType.SESSION_STARTED:
                        entered.set()
                        await release.wait()
                        # Starting a nested invocation must not renew a stale
                        # trial's authority by borrowing the suite reservation.
                        with pytest.raises(ValueError, match="completed eval trial"):
                            await run_eval_case(
                                app,
                                EvalCase(id="stale-nested", request=run_request("stale")),
                                suite_id="stale-nested",
                            )
            else:
                root_ids.append(req.session_id)
                if len(root_ids) == 2:
                    await asyncio.wait_for(entered.wait(), 10)
                    tasks = registry.active_tasks(root_ids[0])
                    assert len(tasks) == 1
                    assert len(provider.requests) == 2
                    operations = provider.background_operation_ids
                    release.set()
                    await asyncio.wait_for(asyncio.gather(*tasks), 10)
                    assert rejections == [
                        "A completed eval trial cannot consume positional scripted batches."
                    ]
                    assert len(provider.requests) == 2
                    assert provider.background_operation_ids == operations
                async for event in original_run(req):
                    yield event

        monkeypatch.setattr(provider, "_consume_batch", observe_consumption)
        monkeypatch.setattr(app, "run", controlled_run)
        case = EvalCase(id="a", request=run_request("run"), assertions=[SessionCompleted()])
        try:
            if repeated:
                result = await run_eval_case(app, case, suite_id="revocation", trials=2)
                first, second = result.trials
            else:
                result = await run_eval_suite(
                    app,
                    EvalSuite(
                        id="revocation",
                        cases=[case, case.model_copy(update={"id": "b"})],
                    ),
                    max_concurrency=1,
                )
                first, second = (item.trials[0] for item in result.cases)
            assert first.status.value == "unavailable"
            assert second.status.value == "passed", second.error
            assert second.final_output == "B"
            assert len(provider.requests) == 3
            assert not any(event.type is EventType.SESSION_COMPLETED for event in child_events)
            assert registry.active_tasks(root_ids[0]) == ()
            assert provider._active_eval_invocation is None
        finally:
            release.set()
            for parent in root_ids:
                await registry.cancel_parent(parent)

    asyncio.run(exercise())


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("max_concurrency", [1, 2])
@pytest.mark.parametrize("factory", [False, True])
def test_ambiguous_routing_is_a_case_error(background, max_concurrency, factory):
    app = CayuApp(enable_logging=False)
    provider = ScriptedModelProvider(
        None if factory else completed("good"),
        response_factory=(lambda req: completed("good")) if factory else None,
        background=background,
    )
    other = ScriptedModelProvider(completed("must not run"), name="other")
    app.register_provider(provider, model_patterns=["*"])
    app.register_provider(other, model_patterns=["bad-*"])
    for name in ("good", "bad"):
        app.register_agent(AgentSpec(name=name, model=f"{name}-model"))
    cases = [
        EvalCase(
            id=name,
            request=RunRequest(agent_name=name, messages=[Message.text("user", name)]),
            assertions=[SessionCompleted()],
        )
        for name in ("bad", "good")
    ]
    result = asyncio.run(
        run_eval_suite(app, EvalSuite(id="routing", cases=cases), max_concurrency=max_concurrency)
    )
    bad, good = result.cases
    assert bad.status.value == "error"
    assert "Model matches multiple registered providers" in bad.trials[0].error
    assert good.status.value == "passed"
    assert len(provider.requests) == 1
    assert other.requests == []
    assert other.background_operation_ids == ()
    single = asyncio.run(run_eval_case(app, cases[0], suite_id="single-routing"))
    assert single.status.value == "error"
    assert "Model matches multiple registered providers" in single.trials[0].error
    assert len(provider.requests) == 1


@pytest.mark.parametrize("suite", [False, True])
def test_missing_provider_is_a_case_error(suite):
    app = CayuApp(enable_logging=False)
    app.register_agent(AgentSpec(name="assistant", model="test"))
    case = EvalCase(id="missing", request=run_request("missing"), assertions=[SessionCompleted()])
    if suite:
        result = asyncio.run(run_eval_suite(app, EvalSuite(id="missing", cases=[case])))
        case_result = result.cases[0]
    else:
        case_result = asyncio.run(run_eval_case(app, case, suite_id="missing"))
    assert case_result.status.value == "error"
    assert "No model provider registered" in case_result.trials[0].error


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_concurrent_suite_attributes_judge_providers(background, shared):
    candidate = ScriptedModelProvider(response_factory=lambda req: completed("answer"))
    app = app_for(candidate)
    providers = []
    judges = []
    for _key in ("a", "b"):
        if shared and judges:
            judges.append(judges[0])
            continue
        provider = ScriptedModelProvider(
            [completed('{"score": 1.0, "rationale": "correct"}')], background=background
        )
        judge_app = app_for(provider)
        providers.append(provider)
        judges.append(LLMJudge(judge_app, agent_name="assistant", rubric="Is the answer correct?"))
    suite = EvalSuite(
        id="judge-isolation",
        cases=[
            EvalCase(id=key, request=run_request(key), assertions=[judge])
            for key, judge in zip(("a", "b"), judges, strict=True)
        ],
    )
    if shared:
        with pytest.raises(ValueError, match="Concurrent eval suites"):
            asyncio.run(run_eval_suite(app, suite, max_concurrency=2))
        assert candidate.requests == []
        assert all(provider.requests == [] for provider in providers)
        assert all(provider.background_operation_ids == () for provider in providers)
    else:
        result = asyncio.run(run_eval_suite(app, suite, max_concurrency=2))
        assert all(case.status.value == "passed" for case in result.cases)
        assert len(candidate.requests) == 2
        assert all(len(provider.requests) == 1 for provider in providers)


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("repeated", [False, True])
def test_concurrent_suite_rejects_positional_before_any_dispatch(background, repeated) -> None:
    provider = ScriptedModelProvider([completed("a"), completed("b")], background=background)
    app = app_for(provider)
    suite = EvalSuite(
        id="isolation",
        cases=[
            EvalCase(id=key, request=run_request(key))
            for key in (("a",) if repeated else ("a", "b"))
        ],
    )
    with pytest.raises(ValueError, match="Concurrent eval suites"):
        asyncio.run(run_eval_suite(app, suite, max_concurrency=2, trials=2 if repeated else 1))
    assert provider.requests == []
    assert provider.background_operation_ids == ()


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("factory_roots", [False, True])
def test_concurrent_plan_attributes_child_providers(background, shared, factory_roots):
    async def exercise():
        app = CayuApp(enable_logging=False)
        providers = []
        cases = []
        for key in ("a", "b"):
            spawn = (
                ModelStreamEvent.tool_call(
                    id="spawn", name="subagent", arguments={"agent": "reviewer", "task": "review"}
                ),
                ModelStreamEvent.completed(),
            )

            def factory(req, spawn=spawn):
                return (
                    completed("parent done")
                    if any(m.role == "tool" for m in req.messages)
                    else spawn
                )

            parent = ScriptedModelProvider(
                None if factory_roots else [spawn, completed("parent done")],
                response_factory=factory if factory_roots else None,
                name=f"parent-{key}",
                background=background,
            )
            app.register_provider(parent)
            providers.append(parent)
            child_name = "child-a" if shared else f"child-{key}"
            if not shared or key == "a":
                child = ScriptedModelProvider(
                    completed("review done"), name=child_name, background=background
                )
                app.register_provider(child)
                providers.append(child)
                app.register_agent(
                    AgentSpec(name=child_name, model="test", provider_name=child_name),
                    # Cyclic declarations must terminate without double-counting
                    # the provider. This route is not invoked by the script.
                    tools=[SubagentTool(app, agents={"self": child_name})],
                )
            app.register_agent(
                AgentSpec(name=f"parent-{key}", model="test", provider_name=parent.name),
                tools=[SubagentTool(app, agents={"reviewer": child_name, "alias": child_name})],
            )
            cases.append(
                EvalCase(
                    id=key,
                    request=RunRequest(
                        agent_name=f"parent-{key}", messages=[Message.text("user", key)]
                    ),
                    assertions=[SessionCompleted(), ChildSessionCompleted(agent_name=child_name)],
                )
            )
        plan = EvalPlan(app=app, suite=EvalSuite(id="descendants", cases=cases))
        if shared:
            with pytest.raises(ValueError, match="Concurrent eval suites"):
                await run_eval_plan(plan, max_concurrency=2)
            assert all(provider.requests == [] for provider in providers)
            assert all(provider.background_operation_ids == () for provider in providers)
        else:
            result = await run_eval_plan(plan, max_concurrency=2)
            assert result.status.value == "passed", [c.trials[0].error for c in result.cases]
            for provider in providers:
                assert len(provider.requests) == (2 if provider.name.startswith("parent") else 1)
        assert all(provider._active_eval_invocation is None for provider in providers)

    asyncio.run(exercise())


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("isolated", [False, True])
def test_concurrent_suite_ignores_unused_and_accepts_isolated_providers(
    monkeypatch, background, isolated
):
    class UnhashableScriptedProvider(ScriptedModelProvider):
        __hash__ = None

    unused = ScriptedModelProvider(completed("unused"), name="unused")
    provider = UnhashableScriptedProvider(
        completed("A") if isolated else None,
        response_factory=None
        if isolated
        else lambda req: completed(req.messages[-1].content[0].text.upper()),
        background=background,
    )
    app = app_for(provider)
    app.register_provider(unused)
    second = UnhashableScriptedProvider(completed("B"), name="second", background=background)
    if isolated:
        app.register_provider(second)
    cases = [
        EvalCase(
            id=key,
            request=run_request(key).model_copy(
                update={"target": ModelTarget(provider_name="second", model="test")}
                if isolated and key == "b"
                else {}
            ),
            assertions=[SessionCompleted(), FinalOutputContains(key.upper())],
        )
        for key in ("a", "b")
    ]

    async def exercise():
        second_finished = asyncio.Event()
        original_run = app.run

        async def inverted_run(req):
            key = req.messages[-1].content[0].text
            if key == "a":
                await asyncio.wait_for(second_finished.wait(), 10)
            try:
                async for event in original_run(req):
                    yield event
            finally:
                if key == "b":
                    second_finished.set()

        monkeypatch.setattr(app, "run", inverted_run)
        return await run_eval_suite(app, EvalSuite(id="isolated", cases=cases), max_concurrency=2)

    result = asyncio.run(exercise())
    assert result.status.value == "passed"
    assert unused.requests == []
    assert len(provider.requests) == (1 if isolated else 2)
    assert len(second.requests) == (1 if isolated else 0)


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("single_case_api", [False, True])
@pytest.mark.parametrize("cancel_first", [False, True])
def test_independent_sequential_evals_reserve_before_setup(
    monkeypatch, background, single_case_api, cancel_first
):
    async def exercise():
        provider = ScriptedModelProvider([completed("A"), completed("B")], background=background)
        app = app_for(provider)
        entered, release = asyncio.Event(), asyncio.Event()
        original_run = app.run

        async def blocked_run(req):
            if req.messages[-1].content[0].text == "a":
                entered.set()
                await release.wait()
            async for event in original_run(req):
                yield event

        monkeypatch.setattr(app, "run", blocked_run)

        async def evaluate(key):
            case = EvalCase(id=key, request=run_request(key), assertions=[SessionCompleted()])
            if single_case_api:
                return await run_eval_case(app, case, suite_id=key)
            return await run_eval_suite(app, EvalSuite(id=key, cases=[case]), max_concurrency=1)

        first = asyncio.create_task(evaluate("a"))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            assert provider.requests == []
            with pytest.raises(ValueError, match="Concurrent eval invocations"):
                await evaluate("b")
            assert provider.requests == []
            assert provider._active_eval_invocation is not None
            if cancel_first:
                first.cancel()
                assert first.cancelling() == 1
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(first, 10)
                assert first.cancelled()
            else:
                release.set()
                assert (await asyncio.wait_for(first, 10)).status.value == "passed"
            assert provider._active_eval_invocation is None
            assert (await evaluate("b")).status.value == "passed"
            assert len(provider.requests) == (1 if cancel_first else 2)
        finally:
            release.set()
            if not first.done():
                first.cancel()
            await asyncio.gather(first, return_exceptions=True)

    asyncio.run(exercise())


@pytest.mark.parametrize("background", [False, True])
def test_independent_sequential_suites_cannot_take_batches_between_tool_rounds(background):
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()

        class Pause(Tool):
            spec = ToolSpec(
                name="pause", description="Pause between rounds", input_schema={"type": "object"}
            )

            async def run(self, ctx, args):
                entered.set()
                await release.wait()
                return ToolResult(content="ready")

        provider = ScriptedModelProvider(
            [
                (
                    ModelStreamEvent.tool_call(id="pause-a", name="pause", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ),
                completed("finished-a"),
                completed("finished-b"),
            ],
            background=background,
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="test"), tools=[Pause()])

        async def evaluate(key):
            return await run_eval_suite(
                app,
                EvalSuite(
                    id=key,
                    cases=[
                        EvalCase(
                            id=key,
                            request=run_request(key),
                            assertions=[SessionCompleted(), FinalOutputContains(f"finished-{key}")],
                        )
                    ],
                ),
                max_concurrency=1,
            )

        first = asyncio.create_task(evaluate("a"))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            assert len(provider.requests) == 1
            with pytest.raises(ValueError, match="Concurrent eval invocations"):
                await evaluate("b")
            assert len(provider.requests) == 1
            release.set()
            assert (await asyncio.wait_for(first, 10)).status.value == "passed"
            assert (await evaluate("b")).status.value == "passed"
            assert len(provider.requests) == 3
        finally:
            release.set()
            if not first.done():
                first.cancel()
            await asyncio.gather(first, return_exceptions=True)

    asyncio.run(exercise())


@pytest.mark.parametrize("background", [False, True])
def test_native_factory_executes_through_public_run(background) -> None:
    def respond(req):
        assert req.options["structured_output"]["strategy"] == "native"
        return completed('{"answer":"done"}')

    provider = ScriptedModelProvider(
        response_factory=respond, supports_native_structured_output=True, background=background
    )
    app = app_for(provider)

    async def run() -> None:
        events = [
            event
            async for event in app.run(
                run_request("answer").model_copy(
                    update={
                        "structured_output": StructuredOutputSpec(
                            strategy="native",
                            json_schema={
                                "type": "object",
                                "properties": {"answer": {"type": "string"}},
                                "required": ["answer"],
                            },
                        )
                    }
                )
            )
        ]
        assert any(event.type is EventType.SESSION_COMPLETED for event in events)
        assert not any(event.type is EventType.SESSION_FAILED for event in events)
        assert len(provider.requests) == 1

    asyncio.run(run())


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("first", ["a", "b"])
def test_factory_concurrent_cases_have_stable_outputs_with_inverted_setup(
    monkeypatch, background, first
):
    async def exercise(concurrency):
        provider = ScriptedModelProvider(
            response_factory=lambda req: completed(req.messages[-1].content[0].text.upper()),
            background=background,
        )
        app = app_for(provider)
        original_run = app.run
        finished = asyncio.Event()

        async def scheduled_run(req):
            key = req.messages[-1].content[0].text
            if concurrency > 1 and key != first:
                await finished.wait()
            try:
                async for event in original_run(req):
                    yield event
            finally:
                if key == first:
                    finished.set()

        monkeypatch.setattr(app, "run", scheduled_run)
        suite = EvalSuite(
            id="keyed",
            cases=[
                EvalCase(
                    id=k,
                    request=run_request(k),
                    assertions=[SessionCompleted(), FinalOutputContains(k.upper())],
                )
                for k in ("a", "b")
            ],
        )
        result = await run_eval_suite(
            app, suite, max_concurrency=concurrency, retain_trajectory=True
        )
        assert len(provider.requests) == 2
        if concurrency > 1:
            assert provider.requests[0].messages[-1].content[0].text == first
        return [
            (case.case_id, case.trials[0].final_output, case.trials[0].status)
            for case in result.cases
        ]

    baseline = asyncio.run(exercise(1))
    assert [item[1] for item in baseline] == ["A", "B"]
    for _ in range(2):
        assert asyncio.run(exercise(2)) == baseline


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("failing_case", [None, "b"])
def test_concurrent_factory_keeps_tool_rounds_and_errors_case_owned(
    background, failing_case, monkeypatch
):
    async def exercise():
        reached = set()
        both = asyncio.Event()
        calls = []

        class Rendezvous(Tool):
            spec = ToolSpec(
                name="rendezvous",
                description="Join concurrent cases",
                parallel_safe=True,
                input_schema={
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            )

            async def run(self, ctx, args):
                key = args["key"]
                calls.append(key)
                return ToolResult(content=key)

        def respond(req):
            key = next(
                message.content[0].text for message in req.messages if message.role == "user"
            )
            if not any(
                part.type == "tool_result" for message in req.messages for part in message.content
            ):
                reached.add(key)
                if len(reached) == 2:
                    both.set()
                return [
                    ModelStreamEvent.tool_call(
                        id=f"call-{key}", name="rendezvous", arguments={"key": key}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            if key == failing_case:
                return [ModelStreamEvent.error(f"failure-{key}"), ModelStreamEvent.completed()]
            return completed(f"finished-{key}")

        provider = ScriptedModelProvider(response_factory=respond, background=background)

        async def wait_for_other_case(req):
            if any(
                part.type == "tool_result" for message in req.messages for part in message.content
            ):
                await asyncio.wait_for(both.wait(), 10)

        if background:
            adapter = provider.provider_operations
            original_start = adapter.start

            async def start(req):
                await wait_for_other_case(req.request)
                return await original_start(req)

            monkeypatch.setattr(adapter, "start", start)
        else:
            original_stream = provider.stream

            async def stream(req):
                await wait_for_other_case(req)
                async for event in original_stream(req):
                    yield event

            monkeypatch.setattr(provider, "stream", stream)
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="test"), tools=[Rendezvous()])
        suite = EvalSuite(
            id="rounds",
            cases=[
                EvalCase(id=k, request=run_request(k), assertions=[SessionCompleted()])
                for k in ("a", "b")
            ],
        )
        result = await run_eval_suite(app, suite, max_concurrency=2, retain_trajectory=True)
        assert sorted(calls) == ["a", "b"], [
            (case.case_id, case.trials[0].error, case.trials[0].final_output)
            for case in result.cases
        ]
        assert len(provider.requests) == 4
        for case in result.cases:
            trial = case.trials[0]
            if case.case_id == failing_case:
                assert trial.status.value == "failed"
                failed = next(
                    event
                    for event in trial.trajectory.events
                    if event.type is EventType.SESSION_FAILED
                )
                assert f"failure-{failing_case}" in failed.payload["error"]
            else:
                assert trial.status.value == "passed"
                assert trial.final_output == f"finished-{case.case_id}"

    asyncio.run(exercise())
