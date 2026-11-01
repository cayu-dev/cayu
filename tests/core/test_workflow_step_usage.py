"""``StepResult`` carries the child's finish reason and token usage, live and on replay."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cayu import AgentSpec, CayuApp
from cayu.evals import ScriptedModelProvider
from cayu.events import Event, EventType
from cayu.providers import ModelFinishReason, ModelStreamEvent
from cayu.workflows import StepResult, StepUsage, WorkflowBase, WorkflowSpec, step
from cayu.workflows.workflow import _StepRunEventState


class _Workflow(WorkflowBase):
    spec = WorkflowSpec(name="step-usage")

    async def run(self, session_id: str) -> AsyncIterator[Event]:  # pragma: no cover
        ctx = self.context(session_id)
        yield await ctx.start()
        yield await ctx.completed()


def _usage(input_tokens: int, output_tokens: int, *, cached: int, reasoning: int) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens_details": {"reasoning_tokens": reasoning},
    }


def _app() -> tuple[CayuApp, ScriptedModelProvider]:
    app = CayuApp(enable_logging=False)
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("Part one. "),
                # end_turn=False asks the runtime for another model call in the same run.
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "end_turn": False,
                        "usage": _usage(100, 20, cached=60, reasoning=5),
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("Part two, cut"),
                ModelStreamEvent.completed(
                    {
                        "stop_reason": "max_tokens",
                        "usage": _usage(150, 40, cached=100, reasoning=10),
                    }
                ),
            ],
        ],
        # The OpenAI usage dialect reads the nested cached and reasoning counters.
        name="openai",
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="writer", model="scripted-model"))
    return app, provider


def test_step_result_reports_final_finish_reason_and_usage_summed_over_calls() -> None:
    app, provider = _app()

    result = asyncio.run(
        step(_Workflow(app).context("wf-usage"), agent="writer", step_id="draft", prompt="go")
    )

    assert len(provider.requests) == 2
    assert result.text == "Part two, cut"
    assert result.finish_reason is ModelFinishReason.LENGTH
    assert result.raw_finish_reason == "max_tokens"
    assert result.usage == StepUsage(
        input_tokens=250,
        output_tokens=60,
        reasoning_tokens=15,
        cached_input_tokens=160,
        total_tokens=310,
        model_calls=2,
    )


def test_step_replayed_from_the_journal_carries_the_same_finish_reason_and_usage() -> None:
    app, provider = _app()
    workflow = _Workflow(app)

    first = asyncio.run(
        step(workflow.context("wf-usage-replay"), agent="writer", step_id="draft", prompt="go")
    )
    # A new attempt of the same run finds `draft` journaled and rebuilds its result
    # from the completed child instead of calling the model again.
    replayed = asyncio.run(
        step(workflow.context("wf-usage-replay"), agent="writer", step_id="draft", prompt="go")
    )

    assert len(provider.requests) == 2
    assert replayed == first
    assert replayed.usage.model_calls == 2


def _completed(payload: dict) -> Event:
    return Event(type=EventType.MODEL_COMPLETED, session_id="child", payload=payload)


def _metrics(input_tokens: int, output_tokens: int) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cache": {"read_tokens": 0, "write_tokens": 0},
    }


def test_step_usage_spans_a_resumed_run_and_restarts_with_a_new_run() -> None:
    state = _StepRunEventState()
    for event in (
        Event(type=EventType.SESSION_STARTED, session_id="child", payload={"run_epoch": 0}),
        _completed({"completion": {"finish_reason": "stop"}, "usage_metrics": _metrics(1, 1)}),
        Event(type=EventType.SESSION_STARTED, session_id="child", payload={"run_epoch": 1}),
        _completed(
            {
                "completion": {"finish_reason": "tool_calls", "raw_finish_reason": "tool_use"},
                "usage_metrics": _metrics(10, 2),
            }
        ),
        Event(type=EventType.SESSION_RESUMED, session_id="child", payload={"run_epoch": 2}),
        # Compaction tokens count, but its finish reason is not the step's.
        _completed(
            {
                "purpose": "context_compaction",
                "completion": {"finish_reason": "length"},
                "usage_metrics": _metrics(5, 1),
            }
        ),
        _completed({"completion": {"finish_reason": "stop", "raw_finish_reason": "end_turn"}}),
    ):
        state.record(event)

    assert (state.finish_reason, state.raw_finish_reason) == (ModelFinishReason.STOP, "end_turn")
    assert state.step_usage() == StepUsage(
        input_tokens=15,
        output_tokens=3,
        total_tokens=18,
        model_calls=3,
        unmeasured_model_calls=1,
    )


def test_step_finish_reason_is_none_for_completions_recorded_without_one() -> None:
    state = _StepRunEventState()
    state.record(_completed({"usage_metrics": _metrics(1, 1)}))
    assert state.finish_reason is None
    assert state.raw_finish_reason is None

    state.record(_completed({"completion": {"finish_reason": "something-new"}}))
    assert state.finish_reason is ModelFinishReason.UNKNOWN


def test_step_result_defaults_keep_hand_built_results_valid() -> None:
    result = StepResult(step_id="s", session_id="child")

    assert result.finish_reason is None
    assert result.raw_finish_reason is None
    assert result.usage == StepUsage()
