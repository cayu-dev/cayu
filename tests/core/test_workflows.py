"""Adversarial behavioral tests for the ``cayu.workflows`` orchestration layer.

Independent of the smoke test: every case here tries to *break* one of the three
shapes (``parallel`` / ``pipeline`` / ``gated_loop``), the ``step`` contract, the
resume substrate, or the ``workflow.*`` event envelope. Everything is driven with
``ScriptedModelProvider`` (no API key) or hand-built ``StepResult`` values, so the
assertions are deterministic.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    ScriptedModelProvider,
    WorkflowSpec,
)
from cayu.core.events import Event
from cayu.core.messages import Message
from cayu.providers import ModelStreamEvent
from cayu.runtime import RunRequest, StructuredOutputSpec, StructuredOutputStrategy
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.workflows import (
    GateOutcome,
    ParallelResult,
    ParallelStepError,
    StepError,
    StepResult,
    StepRunOptions,
    WorkflowBase,
    gated_loop,
    normalize_gate_outcome,
    parallel,
    pipeline,
    step,
)

COUNT_SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
    "additionalProperties": False,
}


def _submit(output: dict[str, Any]) -> list[ModelStreamEvent]:
    """One scripted model step that submits `output` via the structured-output tool."""
    return [
        ModelStreamEvent.tool_call(
            id="call_out",
            name=STRUCTURED_OUTPUT_TOOL_NAME,
            arguments={"output": output},
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def _text(delta: str) -> list[ModelStreamEvent]:
    """One scripted model step that emits plain assistant text and stops."""
    return [
        ModelStreamEvent.text_delta(delta),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def _fail(message: str) -> list[ModelStreamEvent]:
    """A scripted model step whose error event drives the child session to failure."""
    return [
        ModelStreamEvent.error(message),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


class _ContextHarness(WorkflowBase):
    """Concrete ``WorkflowBase`` used only to mint a ``WorkflowContext``.

    Direct-helper tests (``step`` / ``gated_loop`` / ``pipeline``) need a context
    but not a full authored ``run`` body; this harness supplies one.
    """

    spec = WorkflowSpec(name="context-harness")

    async def run(self, session_id: str) -> AsyncIterator[Event]:  # pragma: no cover
        ctx = self.context(session_id)
        yield await ctx.start()
        yield await ctx.completed()


def _context(app: CayuApp, session_id: str, *, name: str):
    return _ContextHarness(app, spec=WorkflowSpec(name=name)).context(session_id)


# --------------------------------------------------------------------------- #
# 1. gated_loop pass path                                                      #
# --------------------------------------------------------------------------- #
def test_gated_loop_pass_path_runs_on_pass_and_journals_verdict():
    app = CayuApp(enable_logging=False)
    ctx = _context(app, "wf-pass", name="pass-wf")
    passes: list[str] = []
    fails: list[str] = []

    async def do(item: Any) -> StepResult:
        return StepResult(step_id=f"do-{item}", session_id=f"child-{item}", output={"item": item})

    async def gate(item: Any, result: StepResult) -> GateOutcome:
        return GateOutcome(passed=True, detail="clean")

    async def on_pass(item: Any, result: StepResult, outcome: GateOutcome) -> None:
        passes.append(item)

    async def on_fail(item: Any, result: StepResult, outcome: GateOutcome) -> None:
        fails.append(item)

    async def drive() -> list[Event]:
        return [
            event
            async for event in gated_loop(
                ctx,
                ["one", "two"],
                do=do,
                gate=gate,
                on_pass=on_pass,
                on_fail=on_fail,
                key=lambda item: str(item),
            )
        ]

    events = asyncio.run(drive())

    # on_pass fired for every item; on_fail never.
    assert passes == ["one", "two"]
    assert fails == []

    # Events yield started/completed per item, in order.
    assert [event.type for event in events] == [
        EventType.WORKFLOW_STEP_STARTED,
        EventType.WORKFLOW_STEP_COMPLETED,
        EventType.WORKFLOW_STEP_STARTED,
        EventType.WORKFLOW_STEP_COMPLETED,
    ]
    completed_events = [e for e in events if e.type == EventType.WORKFLOW_STEP_COMPLETED]
    assert [e.payload["passed"] for e in completed_events] == [True, True]
    assert [e.payload["outcome"] for e in completed_events] == ["pass", "pass"]

    # Completions are journaled under the namespaced per-item key.
    journaled = asyncio.run(ctx.journal.completed_step_ids(attempt_id=ctx.attempt_id))
    assert journaled == {"gated-loop:loop0:one", "gated-loop:loop0:two"}


# --------------------------------------------------------------------------- #
# 2. gated_loop fail path                                                      #
# --------------------------------------------------------------------------- #
def test_gated_loop_fail_path_runs_on_fail_journals_fail_and_continues():
    app = CayuApp(enable_logging=False)
    ctx = _context(app, "wf-fail-path", name="fail-path-wf")
    passes: list[str] = []
    fails: list[str] = []

    async def do(item: Any) -> StepResult:
        return StepResult(step_id=f"do-{item}", session_id=f"child-{item}", output={"item": item})

    async def gate(item: Any, result: StepResult) -> bool:
        # A bare bool exercises normalize_gate_outcome inside the loop.
        return False

    async def on_pass(item: Any, result: StepResult, outcome: GateOutcome) -> None:
        passes.append(item)

    async def on_fail(item: Any, result: StepResult, outcome: GateOutcome) -> None:
        fails.append(item)

    async def drive() -> list[Event]:
        return [
            event
            async for event in gated_loop(
                ctx,
                ["a", "b"],
                do=do,
                gate=gate,
                on_pass=on_pass,
                on_fail=on_fail,
                key=lambda item: str(item),
            )
        ]

    events = asyncio.run(drive())

    # on_fail fired for both; on_pass never. The loop did NOT stop after the
    # first failing item — the second item was still processed.
    assert fails == ["a", "b"]
    assert passes == []

    completed_events = [e for e in events if e.type == EventType.WORKFLOW_STEP_COMPLETED]
    assert [e.payload["passed"] for e in completed_events] == [False, False]
    assert [e.payload["outcome"] for e in completed_events] == ["fail", "fail"]

    # A failing verdict still counts as *completed* for resume purposes.
    journaled = asyncio.run(ctx.journal.completed_step_ids(attempt_id=ctx.attempt_id))
    assert journaled == {"gated-loop:loop0:a", "gated-loop:loop0:b"}


# --------------------------------------------------------------------------- #
# 3 & 4. gated_loop resume: skip journaled, retry uncommitted                  #
# --------------------------------------------------------------------------- #
def _crash_then_resume() -> dict[str, Any]:
    """Run a gated_loop that crashes on the 2nd item, then resume it fresh.

    Returns the recorded ``do``/``on_pass`` calls and journal snapshots of both
    processes so the resume guarantees can be asserted precisely.
    """
    app = CayuApp(enable_logging=False)
    items = ["alpha", "beta", "gamma"]

    # ---- process 1: crashes inside `do` on the second item ("beta") ----
    ctx1 = _context(app, "wf-resume", name="resume-wf")
    do_calls_1: list[str] = []
    pass_calls_1: list[str] = []

    async def do1(item: Any) -> StepResult:
        do_calls_1.append(item)
        if item == "beta":
            raise RuntimeError("simulated crash inside do on beta")
        return StepResult(step_id=f"do-{item}", session_id=f"child-{item}", output={"item": item})

    async def gate1(item: Any, result: StepResult) -> GateOutcome:
        return GateOutcome(passed=True)

    async def on_pass1(item: Any, result: StepResult, outcome: GateOutcome) -> None:
        pass_calls_1.append(item)

    async def run1() -> None:
        async for _event in gated_loop(
            ctx1, items, do=do1, gate=gate1, on_pass=on_pass1, key=lambda item: str(item)
        ):
            pass

    with pytest.raises(RuntimeError):
        asyncio.run(run1())
    completed_after_1 = asyncio.run(ctx1.journal.completed_step_ids(attempt_id=ctx1.attempt_id))

    # ---- process 2: brand-new workflow + journal, SAME app + session id ----
    ctx2 = _context(app, "wf-resume", name="resume-wf")
    do_calls_2: list[str] = []
    pass_calls_2: list[str] = []

    async def do2(item: Any) -> StepResult:
        do_calls_2.append(item)
        return StepResult(step_id=f"do-{item}", session_id=f"child-{item}", output={"item": item})

    async def gate2(item: Any, result: StepResult) -> GateOutcome:
        return GateOutcome(passed=True)

    async def on_pass2(item: Any, result: StepResult, outcome: GateOutcome) -> None:
        pass_calls_2.append(item)

    async def run2() -> None:
        async for _event in gated_loop(
            ctx2, items, do=do2, gate=gate2, on_pass=on_pass2, key=lambda item: str(item)
        ):
            pass

    asyncio.run(run2())
    completed_after_2 = asyncio.run(ctx2.journal.completed_step_ids(attempt_id=ctx2.attempt_id))

    return {
        "do_calls_1": do_calls_1,
        "pass_calls_1": pass_calls_1,
        "completed_after_1": completed_after_1,
        "do_calls_2": do_calls_2,
        "pass_calls_2": pass_calls_2,
        "completed_after_2": completed_after_2,
    }


def test_gated_loop_resume_skips_journaled_item():
    obs = _crash_then_resume()

    # Process 1 got as far as completing "alpha" before crashing on "beta".
    assert obs["do_calls_1"] == ["alpha", "beta"]
    assert obs["pass_calls_1"] == ["alpha"]
    assert obs["completed_after_1"] == {"gated-loop:loop0:alpha"}

    # Process 2 (resume) SKIPS the already-journaled "alpha": its `do` and
    # `on_pass` never fire for it again.
    assert "alpha" not in obs["do_calls_2"]
    assert "alpha" not in obs["pass_calls_2"]


def test_gated_loop_do_raise_leaves_item_unjournaled_and_retries_on_resume():
    obs = _crash_then_resume()

    # "beta" raised inside `do`, so it was never journaled as completed...
    assert "gated-loop:loop0:beta" not in obs["completed_after_1"]

    # ...and on resume it is retried (not skipped), together with the untouched
    # "gamma". Exactly the two remaining items run, in order.
    assert obs["do_calls_2"] == ["beta", "gamma"]
    assert obs["pass_calls_2"] == ["beta", "gamma"]
    assert obs["completed_after_2"] == {
        "gated-loop:loop0:alpha",
        "gated-loop:loop0:beta",
        "gated-loop:loop0:gamma",
    }


# --------------------------------------------------------------------------- #
# 5. parallel failure semantics                                                #
# --------------------------------------------------------------------------- #
def test_parallel_surfaces_failure_as_stepfailure_and_fails_closed():
    app = CayuApp(enable_logging=False)
    # Distinct named providers so gather order can't scramble which branch fails.
    app.register_provider(ScriptedModelProvider([_submit({"n": 7})], name="ok-prov"), default=True)
    app.register_provider(ScriptedModelProvider([_fail("branch boom")], name="fail-prov"))
    app.register_agent(AgentSpec(name="w", model="scripted-model"))
    ctx = _context(app, "wf-par", name="par-wf")

    async def go():
        return await parallel(
            [
                step(
                    ctx,
                    agent="w",
                    step_id="good",
                    prompt="x",
                    schema=COUNT_SCHEMA,
                    run_options=StepRunOptions(provider_name="ok-prov"),
                ),
                step(
                    ctx,
                    agent="w",
                    step_id="bad",
                    prompt="x",
                    schema=COUNT_SCHEMA,
                    run_options=StepRunOptions(provider_name="fail-prov"),
                ),
            ]
        )

    result = asyncio.run(go())

    # The failing branch is surfaced, never dropped; submission order preserved.
    assert result.ok is False
    assert len(result.results) == 2
    assert len(result.successes) == 1
    assert len(result.failures) == 1

    (success,) = result.successes
    assert success.step_id == "good"
    assert success.output == {"n": 7}

    (failure,) = result.failures
    assert failure.step_id == "bad"
    assert failure.session_id is not None  # correlated back to its child run
    assert "branch boom" in failure.error
    # A session.failed carries no Python cause, so the type is StepError itself.
    assert failure.error_type == "StepError"

    # Fail-closed accessors both raise rather than let a partial fan-out pass.
    with pytest.raises(ParallelStepError):
        result.raise_for_failures()
    with pytest.raises(ParallelStepError):
        _ = result.outputs


def test_parallel_runs_steps_concurrently():
    started: list[str] = []

    async def branch(name: str, both_started: asyncio.Event) -> StepResult:
        started.append(name)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        return StepResult(step_id=name, session_id=f"child-{name}")

    async def go() -> ParallelResult:
        both_started = asyncio.Event()
        return await asyncio.wait_for(
            parallel([branch("a", both_started), branch("b", both_started)]),
            timeout=1,
        )

    result = asyncio.run(go())

    assert result.ok is True
    assert {item.step_id for item in result.successes} == {"a", "b"}


def test_parallel_propagates_external_cancellation():
    async def run():
        started = asyncio.Event()

        async def branch() -> StepResult:
            started.set()
            await asyncio.Event().wait()
            return StepResult(step_id="a", session_id="child-a")

        fanout = asyncio.create_task(parallel([branch()]))
        await started.wait()
        fanout.cancel()
        with pytest.raises(asyncio.CancelledError):
            await fanout

    asyncio.run(run())


def test_parallel_cancels_siblings_when_fanout_is_cancelled():
    async def run():
        sibling_cancelled: list[bool] = []
        started = asyncio.Event()

        async def sibling() -> StepResult:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                sibling_cancelled.append(True)
            return StepResult(step_id="sibling", session_id="child-sibling")

        fanout = asyncio.create_task(parallel([sibling()]))
        await started.wait()
        fanout.cancel()
        with pytest.raises(asyncio.CancelledError):
            await fanout
        assert sibling_cancelled == [True]

    asyncio.run(run())


def test_parallel_collects_plain_exception_as_stepfailure():
    async def good() -> StepResult:
        return StepResult(step_id="good", session_id="child-good", output={"ok": True})

    async def bad() -> StepResult:
        raise RuntimeError("plain boom")

    result = asyncio.run(parallel([good(), bad()]))

    assert result.ok is False
    assert result.results[0] == StepResult(
        step_id="good",
        session_id="child-good",
        output={"ok": True},
    )
    assert len(result.failures) == 1
    assert result.failures[0].error == "plain boom"
    assert result.failures[0].error_type == "RuntimeError"


# --------------------------------------------------------------------------- #
# 6. pipeline typed edge                                                       #
# --------------------------------------------------------------------------- #
def test_pipeline_feeds_prior_stepresult_and_rejects_empty():
    received: list[Any] = []
    r1 = StepResult(step_id="s1", session_id="c1", output={"a": 1})
    r2 = StepResult(step_id="s2", session_id="c2", output={"b": 2})

    async def stage_one(value: Any) -> StepResult:
        received.append(value)
        return r1

    async def stage_two(value: Any) -> StepResult:
        received.append(value)
        return r2

    result = asyncio.run(pipeline("seed", [stage_one, stage_two]))

    # First stage sees `initial`; the later stage receives the PRIOR StepResult
    # object itself (a typed Python edge — identity, not a re-serialized string).
    assert received[0] == "seed"
    assert received[1] is r1
    # pipeline returns the final stage's result.
    assert result is r2

    # No stages → fail-closed.
    with pytest.raises(ValueError):
        asyncio.run(pipeline("seed", []))


def test_pipeline_rejects_stage_that_returns_non_stepresult():
    async def bad_stage(value: Any) -> Any:
        return "not a step result"

    with pytest.raises(TypeError):
        asyncio.run(pipeline("seed", [bad_stage]))


# --------------------------------------------------------------------------- #
# 7. step contract                                                             #
# --------------------------------------------------------------------------- #
def test_step_schema_populates_output():
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([_submit({"n": 42})]), default=True)
    app.register_agent(AgentSpec(name="w", model="scripted-model"))
    ctx = _context(app, "wf-step-out", name="step-out-wf")

    result = asyncio.run(step(ctx, agent="w", step_id="s", prompt="go", schema=COUNT_SCHEMA))
    assert result.output == {"n": 42}
    assert result.has_output is True
    assert result.step_id == "s"


def test_step_session_failed_raises_step_error_with_ids():
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([_fail("kaboom")]), default=True)
    app.register_agent(AgentSpec(name="w", model="scripted-model"))
    ctx = _context(app, "wf-step-fail", name="step-fail-wf")

    with pytest.raises(StepError) as excinfo:
        asyncio.run(
            step(ctx, agent="w", step_id="the-step", prompt="go", session_id="pinned-child")
        )
    assert excinfo.value.step_id == "the-step"
    assert excinfo.value.session_id == "pinned-child"
    assert "kaboom" in str(excinfo.value)


def test_step_rejects_prompt_and_messages_together_or_neither():
    app = CayuApp(enable_logging=False)
    ctx = _context(app, "wf-step-args", name="step-args-wf")

    with pytest.raises(ValueError):
        asyncio.run(
            step(
                ctx,
                agent="w",
                step_id="s",
                prompt="x",
                messages=[Message.text("user", "y")],
            )
        )
    with pytest.raises(ValueError):
        asyncio.run(step(ctx, agent="w", step_id="s"))


def test_step_rejects_schema_and_structured_output_together():
    app = CayuApp(enable_logging=False)
    ctx = _context(app, "wf-step-so", name="step-so-wf")
    spec = StructuredOutputSpec(
        name="dup",
        json_schema=COUNT_SCHEMA,
        strategy=StructuredOutputStrategy.TOOL,
    )

    with pytest.raises(ValueError):
        asyncio.run(
            step(
                ctx,
                agent="w",
                step_id="s",
                prompt="x",
                schema=COUNT_SCHEMA,
                structured_output=spec,
            )
        )


# --------------------------------------------------------------------------- #
# 8. per-agent model tiering                                                   #
# --------------------------------------------------------------------------- #
def test_per_agent_model_tiering_records_intended_model_per_child_session():
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider([_submit({"n": 1}), _submit({"n": 2})]), default=True
    )
    app.register_agent(AgentSpec(name="cheap", model="cheap-model"))
    app.register_agent(AgentSpec(name="premium", model="premium-model"))
    ctx = _context(app, "wf-tier", name="tier-wf")

    async def go() -> tuple[StepResult, StepResult]:
        cheap = await step(ctx, agent="cheap", step_id="s1", prompt="x", schema=COUNT_SCHEMA)
        premium = await step(ctx, agent="premium", step_id="s2", prompt="x", schema=COUNT_SCHEMA)
        return cheap, premium

    cheap_result, premium_result = asyncio.run(go())

    cheap_session = asyncio.run(app.session_store.load(cheap_result.session_id))
    premium_session = asyncio.run(app.session_store.load(premium_result.session_id))
    assert cheap_session is not None
    assert premium_session is not None
    # Tiering is expressed purely through agent identity: the child session each
    # step ran under recorded the model of the agent it named.
    assert cheap_session.model == "cheap-model"
    assert cheap_session.agent_name == "cheap"
    assert premium_session.model == "premium-model"
    assert premium_session.agent_name == "premium"


# --------------------------------------------------------------------------- #
# 9. escape hatch                                                              #
# --------------------------------------------------------------------------- #
def test_escape_hatch_direct_app_run_inside_run():
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([_text("done")]), default=True)
    app.register_agent(AgentSpec(name="w", model="scripted-model"))

    class EscapeWorkflow(WorkflowBase):
        spec = WorkflowSpec(name="escape-wf")

        async def run(self, session_id: str) -> AsyncIterator[Event]:
            ctx = self.context(session_id)
            yield await ctx.start()
            seen: list[Any] = []
            async for event in self.app.run(
                RunRequest(
                    agent_name="w",
                    session_id="escape-child",
                    messages=[Message.text("user", "hi")],
                )
            ):
                seen.append(event.type)
            self.seen = seen
            yield await ctx.completed()

    workflow = EscapeWorkflow(app)

    async def drive() -> list[Event]:
        return [event async for event in workflow.run("wf-escape")]

    events = asyncio.run(drive())

    # The directly-driven child run completed via the escape hatch...
    assert EventType.SESSION_COMPLETED in workflow.seen
    # ...and the workflow still brackets itself.
    assert events[0].type == EventType.WORKFLOW_STARTED
    assert events[-1].type == EventType.WORKFLOW_COMPLETED


# --------------------------------------------------------------------------- #
# 10. workflow.* event envelope                                                #
# --------------------------------------------------------------------------- #
def test_workflow_started_first_completed_last_carry_workflow_name():
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([_submit({"n": 5})]), default=True)
    app.register_agent(AgentSpec(name="w", model="scripted-model"))

    class BracketWorkflow(WorkflowBase):
        spec = WorkflowSpec(name="bracket-wf")

        async def run(self, session_id: str) -> AsyncIterator[Event]:
            ctx = self.context(session_id)
            yield await ctx.start()
            mid = await step(ctx, agent="w", step_id="mid", prompt="go", schema=COUNT_SCHEMA)
            yield await ctx.completed({"n": mid.output})

    workflow = BracketWorkflow(app)

    async def drive() -> list[Event]:
        return [event async for event in workflow.run("wf-bracket")]

    events = asyncio.run(drive())

    assert events[0].type == EventType.WORKFLOW_STARTED
    assert events[0].workflow_name == "bracket-wf"
    assert events[0].payload["workflow"] == "bracket-wf"

    assert events[-1].type == EventType.WORKFLOW_COMPLETED
    assert events[-1].workflow_name == "bracket-wf"
    assert events[-1].payload["n"] == {"n": 5}


# --------------------------------------------------------------------------- #
# bonus: normalize_gate_outcome contract                                       #
# --------------------------------------------------------------------------- #
def test_normalize_gate_outcome_accepts_bool_and_gateoutcome_rejects_other():
    coerced = normalize_gate_outcome(True)
    assert isinstance(coerced, GateOutcome)
    assert coerced.passed is True

    passthrough = GateOutcome(passed=False, detail="reason")
    assert normalize_gate_outcome(passthrough) is passthrough

    not_a_gate: Any = "nope"
    with pytest.raises(TypeError):
        normalize_gate_outcome(not_a_gate)
