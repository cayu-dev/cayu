"""How a direct eval trial reports what happened to the candidate's run."""

from __future__ import annotations

import asyncio
from typing import get_args

import pytest
from tests.core.postgres_contention_support import drop_cayu_tables
from tests.sandbox_broker.fake_broker import FakeBroker

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.budgets.run_limits import RunLimits
from cayu.environments import EnvironmentLifecyclePolicy, EnvironmentSpec
from cayu.evals import (
    EvalAssertion,
    EvalAssertionResult,
    EvalCase,
    EvalContext,
    EvalOutcome,
    EvalStatus,
    EvalSuite,
    FinalOutputContains,
    ScriptedModelProvider,
    render_html_report,
    run_eval_suite,
)
from cayu.evals.models import EvalExecutionLimit, ProbeRequirements
from cayu.evals.runner import _TRACEBACK_TRUNCATION_MARKER, _TRIAL_TRACEBACK_MAX_CHARS
from cayu.events import Event, EventType
from cayu.observability.hooks import RuntimeHook, RuntimeHookContext
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.stop_policy import StopLimit
from cayu.sandbox_broker import BrokeredSandboxCodingEnvironmentFactory
from cayu.sessions import InMemorySessionStore
from cayu.storage import PostgresSessionStore, SQLiteSessionStore
from cayu.storage.migrations import SchemaMode
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workspaces import LocalWorkspace

# Longer than the one-second run limit, so the second model step crosses it.
_SLOW_STEP_SECONDS = 1.2


class _SlowSecondStepProvider(ScriptedModelProvider):
    async def stream(self, request):
        if self.requests:
            await asyncio.sleep(_SLOW_STEP_SECONDS)
        async for event in super().stream(request):
            yield event


class _NoteTool(Tool):
    spec = ToolSpec(
        name="note",
        description="Record a note.",
        input_schema={"type": "object", "properties": {}},
        max_terminal_payload_bytes=1024,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="noted")


def _note_step(call_id: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(id=call_id, name="note", arguments={}),
        ModelStreamEvent.completed(
            {
                "finish_reason": "tool_calls",
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            }
        ),
    ]


@pytest.fixture(params=("sqlite", pytest.param("postgres", marks=pytest.mark.postgres)))
def store_case(request, tmp_path):
    if request.param == "postgres":
        return request.param, tmp_path, request.getfixturevalue("postgres_dsn")
    return request.param, tmp_path, None


async def _open_store(case):
    kind, tmp_path, postgres_dsn = case
    if kind == "sqlite":
        return SQLiteSessionStore(tmp_path / "sessions.sqlite")
    await drop_cayu_tables(postgres_dsn)
    return PostgresSessionStore(
        postgres_dsn,
        min_size=1,
        max_size=4,
        schema_mode=SchemaMode.CREATE,
    )


class _ExplodingProvider(ModelProvider):
    name = "exploding"

    async def stream(self, request):
        if request is not None:
            raise RuntimeError("model exploded")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _RaisingAssertion(EvalAssertion):
    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        raise RuntimeError("assertion exploded")


def _case(request: RunRequest, *assertions: EvalAssertion) -> EvalSuite:
    return EvalSuite(
        id="outcomes", cases=[EvalCase(id="case", request=request, assertions=list(assertions))]
    )


def _request(**updates) -> RunRequest:
    return RunRequest(agent_name="agent", messages=[Message.text("user", "Take notes.")], **updates)


def _app(provider: ModelProvider) -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="agent", model="fake-model"))
    return app


def _sandboxed_app(store, broker: FakeBroker, tmp_path) -> CayuApp:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("project\n")
    app = CayuApp(enable_logging=False, session_store=store)
    app.register_provider(
        _SlowSecondStepProvider([_note_step("call-1"), _note_step("call-2")]),
        default=True,
    )
    app.register_environment_factory(
        EnvironmentSpec(
            name="sandbox",
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="brokered-sandbox", behavior_version="1", implementation_version="1"
            ),
            lifecycle_policy=EnvironmentLifecyclePolicy(),
        ),
        BrokeredSandboxCodingEnvironmentFactory(
            broker=broker.client(),
            source_workspace=LocalWorkspace(source, workspace_id="project"),
            egress="none",
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="agent", model="fake-model"), tools=[_NoteTool()])
    return app


def test_unregistered_environment_is_a_setup_failure_not_an_evaluator_error():
    app = _app(ScriptedModelProvider([_note_step("call-1")]))
    suite = _case(_request(environment_name="no_environment"), FinalOutputContains("done"))

    result = asyncio.run(run_eval_suite(app, suite))

    trial = result.cases[0].trials[0]
    assert trial.status is EvalStatus.ERROR
    assert trial.execution_status == "failed"
    assert trial.execution_failure_category == "setup_failure"
    assert trial.session_id is None
    assert trial.events_count == 0
    assert trial.error is not None
    assert trial.error.endswith("KeyError: 'Environment not registered: no_environment'")
    assert len(trial.error) <= len(_TRACEBACK_TRUNCATION_MARKER) + _TRIAL_TRACEBACK_MAX_CHARS
    (assertion,) = trial.assertions
    assert assertion.outcome is EvalOutcome.UNAVAILABLE
    assert assertion.message == (
        "Not evaluated (candidate_did_not_run): the candidate's run failed before it started."
    )
    assert "Candidate run: failed (setup_failure)." in render_html_report(result)


def test_traceback_is_kept_once_with_its_last_lines():
    class _SetupFailureApp(CayuApp):
        def run(self, request):
            # The cause's long message makes the traceback longer than the bound.
            raise KeyError("Environment not registered: deep") from ValueError("x" * 8192)

    app = _SetupFailureApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([_note_step("call-1")]), default=True)
    app.register_agent(AgentSpec(name="agent", model="fake-model"))

    result = asyncio.run(run_eval_suite(app, _case(_request(), FinalOutputContains("done"))))

    trial = result.cases[0].trials[0]
    assert trial.execution_failure_category == "setup_failure"
    assert trial.error is not None
    assert trial.error.startswith(_TRACEBACK_TRUNCATION_MARKER)
    assert len(trial.error) == len(_TRACEBACK_TRUNCATION_MARKER) + _TRIAL_TRACEBACK_MAX_CHARS
    assert trial.error.endswith("KeyError: 'Environment not registered: deep'")
    assert "Traceback" not in trial.assertions[0].message


@pytest.mark.store_semantics
def test_environment_refused_at_setup_is_an_environment_failure(store_case):
    _, tmp_path, _ = store_case
    broker = FakeBroker(tmp_path / "broker")
    broker.refuse_create = (429, "sandbox_quota_exceeded")
    suite = _case(_request(), FinalOutputContains("done"))

    async def scenario():
        store = await _open_store(store_case)
        try:
            app = _sandboxed_app(store, broker, tmp_path)
            result = await run_eval_suite(app, suite)
            return result, await store.load_events(result.cases[0].trials[0].session_id)
        finally:
            await store.close()

    result, stored_events = asyncio.run(scenario())

    trial = result.cases[0].trials[0]
    assert trial.status is EvalStatus.ERROR
    assert trial.execution_status == "failed"
    assert trial.execution_failure_category == "environment_failure"
    assert trial.error is not None
    assert trial.error.startswith("Session failed: Sandbox broker answered HTTP 429")
    assert "sandbox_quota_exceeded" in trial.error
    assert trial.events_count == len(stored_events)
    assert stored_events[-1].type == EventType.SESSION_FAILED
    assert EventType.ENVIRONMENT_FACTORY_FAILED in {event.type for event in stored_events}
    assert trial.usage_summary is None
    assert trial.evidence_complete is False
    (assertion,) = trial.assertions
    assert assertion.outcome is EvalOutcome.UNAVAILABLE
    assert "candidate_did_not_run" in assertion.message


def test_run_that_raises_after_it_started_leaves_its_assertions_unevaluated():
    class _LostStreamApp(CayuApp):
        async def run(self, request):
            yield Event(type=EventType.SESSION_STARTED, session_id=request.session_id)
            raise RuntimeError("runner lost its stream")

    app = _LostStreamApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([_note_step("call-1")]), default=True)
    app.register_agent(AgentSpec(name="agent", model="fake-model"))

    result = asyncio.run(run_eval_suite(app, _case(_request(), FinalOutputContains("done"))))

    trial = result.cases[0].trials[0]
    assert trial.status is EvalStatus.ERROR
    assert trial.execution_status == "failed"
    assert trial.execution_failure_category == "execution_failure"
    assert trial.events_count == 1
    assert trial.error is not None
    assert trial.error.endswith("RuntimeError: runner lost its stream")
    (assertion,) = trial.assertions
    assert assertion.outcome is EvalOutcome.UNAVAILABLE
    assert assertion.message == "Not evaluated (candidate_run_failed): the candidate's run failed."


def test_failed_session_with_its_snapshot_keeps_the_failure_on_its_assertions():
    result = asyncio.run(
        run_eval_suite(_app(_ExplodingProvider()), _case(_request(), FinalOutputContains("done")))
    )

    trial = result.cases[0].trials[0]
    assert trial.status is EvalStatus.ERROR
    assert trial.evidence_complete is True
    assert trial.execution_status is None
    assert trial.execution_failure_category == "provider_failure"
    (assertion,) = trial.assertions
    assert assertion.outcome is EvalOutcome.ERROR
    assert assertion.message == trial.error


class _UndeclarableProbesAssertion(EvalAssertion):
    def required_probes(self) -> ProbeRequirements:
        raise RuntimeError("probe declaration exploded")

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        return self.passed()


def test_failed_evidence_preparation_after_a_raising_run_is_an_evaluator_error():
    class _RaisesAfterCompletionApp(CayuApp):
        async def run(self, request):
            async for event in super().run(request):
                yield event
            raise RuntimeError("runner failed after the run completed")

    app = _RaisesAfterCompletionApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="agent", model="fake-model"))

    result = asyncio.run(run_eval_suite(app, _case(_request(), _UndeclarableProbesAssertion())))

    # The terminal snapshot loaded, so the run is not recorded as a failed execution.
    trial = result.cases[0].trials[0]
    assert trial.status is EvalStatus.ERROR
    assert trial.evidence_complete is False
    assert trial.final_output == "done"
    assert trial.execution_status is None
    assert trial.execution_failure_category is None
    assert trial.error is not None
    assert trial.error.endswith("RuntimeError: runner failed after the run completed")
    (assertion,) = trial.assertions
    assert assertion.outcome is EvalOutcome.ERROR


def test_assertion_that_raises_on_an_executed_run_is_an_evaluator_error():
    app = _app(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        )
    )

    result = asyncio.run(run_eval_suite(app, _case(_request(), _RaisingAssertion())))

    trial = result.cases[0].trials[0]
    assert trial.status is EvalStatus.ERROR
    assert trial.execution_status is None
    assert trial.execution_failure_category is None
    (assertion,) = trial.assertions
    assert assertion.outcome is EvalOutcome.ERROR
    assert assertion.message == "Assertion raised RuntimeError: assertion exploded"


def test_execution_limits_name_every_run_limit():
    assert set(get_args(EvalExecutionLimit)) == {limit.value for limit in StopLimit}


@pytest.mark.store_semantics
def test_trial_stopped_by_a_run_limit_keeps_its_evidence(store_case):
    _, tmp_path, _ = store_case
    broker = FakeBroker(tmp_path / "broker")
    suite = _case(_request(limits=RunLimits(max_elapsed_seconds=1)), FinalOutputContains("done"))

    async def scenario():
        store = await _open_store(store_case)
        try:
            app = _sandboxed_app(store, broker, tmp_path)
            result = await run_eval_suite(app, suite, retain_trajectory=True)
            # The sandbox's cleanup appends lifecycle records after the terminal event.
            assert await app.drain_environment_cleanups(timeout_s=5)
            return result, await store.load_events(result.cases[0].trials[0].session_id)
        finally:
            await store.close()

    result, stored_events = asyncio.run(scenario())

    trial = result.cases[0].trials[0]
    assert trial.status is EvalStatus.FAILED
    assert trial.unavailable_reason is None
    assert trial.execution_failure_category == "limit_reached"
    assert trial.execution_limit == "elapsed_seconds"
    assert trial.execution_status is None
    assert trial.assertions[0].outcome is EvalOutcome.FAILED
    assert trial.evidence_complete is True
    assert trial.trajectory is not None
    events = trial.trajectory.events
    assert trial.events_count == len(events)
    assert events[-1].type == EventType.SESSION_INTERRUPTED
    assert events[-1].payload["reason"] == "limit_reached"
    assert events[-1].payload["limit"] == "elapsed_seconds"
    assert len(stored_events) > len(events)
    # Sandbox setup counts toward the limit and may consume it before any model step.
    assert trial.usage_summary is not None
    model_steps = trial.usage_summary["model_steps"]
    assert model_steps == sum(event.type is EventType.MODEL_COMPLETED for event in events)
    assert trial.usage_summary["usage"]["total_tokens"] == 5 * model_steps
    assert "Candidate run: failed (limit_reached: elapsed_seconds)." in render_html_report(result)


class _AfterInterruptionHook(RuntimeHook):
    async def after_session_interrupted(self, context: RuntimeHookContext) -> None:
        return None


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_hook_after_a_run_limit_keeps_the_trial_evidence(store_kind, tmp_path):
    suite = _case(_request(max_steps=1), FinalOutputContains("done"))

    async def scenario():
        store = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.sqlite")
        )
        try:
            app = CayuApp(
                enable_logging=False,
                session_store=store,
                runtime_hooks=[_AfterInterruptionHook()],
            )
            app.register_provider(
                ScriptedModelProvider([_note_step("call-1")]),
                default=True,
            )
            app.register_agent(AgentSpec(name="agent", model="fake-model"), tools=[_NoteTool()])
            result = await run_eval_suite(app, suite, retain_trajectory=True)
            return result, await store.load_events(result.cases[0].trials[0].session_id)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    result, stored_events = asyncio.run(scenario())

    trial = result.cases[0].trials[0]
    # The hook's records follow the terminal event in the emitted stream and the store.
    terminal = next(
        index
        for index, event in enumerate(stored_events)
        if event.type == EventType.SESSION_INTERRUPTED
    )
    assert {event.type for event in stored_events[terminal + 1 :]} >= {
        EventType.HOOK_STARTED,
        EventType.HOOK_COMPLETED,
    }
    assert trial.unavailable_reason is None
    assert trial.evidence_complete is True
    assert trial.status is EvalStatus.FAILED
    assert trial.execution_failure_category == "limit_reached"
    assert trial.execution_limit == "model_steps"
    assert trial.trajectory is not None
    assert trial.trajectory.events[-1].type == EventType.SESSION_INTERRUPTED
    assert trial.events_count == terminal + 1
