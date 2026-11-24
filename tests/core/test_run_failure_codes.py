"""Stable failure codes for provider stream capacity and oversized tool output."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cayu import AgentSpec, CayuApp, Event, Message, RunRequest, SessionStatus
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent, ProviderStreamCapacityExhausted
from cayu.providers.deadlines import ProviderStreamDeadlineAdmission, ProviderStreamDeadlines
from cayu.sessions import RunFailureCode
from cayu.sessions.outcomes import RunOutcome, failure_code_of, run_to_completion
from cayu.tools import Tool, ToolEffect, ToolResult, ToolSpec


class _OneStreamProvider(ScriptedModelProvider):
    @property
    def stream_deadlines(self) -> ProviderStreamDeadlines:
        return ProviderStreamDeadlines(max_concurrent_streams=1)


class _OversizedTool(Tool):
    spec = ToolSpec(
        name="export",
        input_schema={"type": "object"},
        effect=ToolEffect.NONE,
        max_terminal_payload_bytes=64 * 1024,
    )

    async def run(self, ctx, args):
        del ctx, args
        return ToolResult(content="x" * (100 * 1024))


def _request() -> RunRequest:
    return RunRequest(
        agent_name="assistant", session_id="s1", messages=[Message.text("user", "hi")]
    )


def test_exhausted_stream_capacity_fails_the_session_with_a_stable_code() -> None:
    async def scenario() -> RunOutcome:
        held = ProviderStreamDeadlineAdmission(ProviderStreamDeadlines(max_concurrent_streams=1))
        try:
            app = CayuApp()
            app.register_provider(
                _OneStreamProvider(
                    events=[[ModelStreamEvent.text_delta("hi"), ModelStreamEvent.completed()]]
                ),
                default=True,
            )
            app.register_agent(AgentSpec(name="assistant", model="model"))
            return await run_to_completion(app, _request())
        finally:
            held.close()

    outcome = asyncio.run(scenario())

    assert outcome.status is SessionStatus.FAILED
    assert outcome.failure_code == RunFailureCode.PROVIDER_STREAM_CAPACITY_EXHAUSTED
    failed = [event for event in outcome.events if event.type == "session.failed"]
    assert failed[-1].payload["failure_code"] == "provider_stream_capacity_exhausted"
    assert failed[-1].payload["error_type"] == "ProviderStreamCapacityExhausted"
    assert outcome.summary()["failure_code"] == "provider_stream_capacity_exhausted"


def test_oversized_tool_output_names_the_size_code_on_the_call() -> None:
    async def scenario() -> RunOutcome:
        app = CayuApp()
        app.register_provider(
            ScriptedModelProvider(
                events=[
                    [
                        ModelStreamEvent.tool_call(name="export", arguments={}, id="c1"),
                        ModelStreamEvent.completed(),
                    ],
                    [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="model"), tools=[_OversizedTool()])
        return await run_to_completion(app, _request())

    outcome = asyncio.run(scenario())

    assert outcome.status is SessionStatus.COMPLETED
    assert outcome.failure_code is None
    (call,) = outcome.tool_calls
    assert call.disposition == "failed"
    assert call.failure_code == RunFailureCode.TOOL_OUTPUT_TOO_LARGE
    assert call.result is not None
    assert call.result.structured is not None
    assert call.result.structured["durable_value_error_code"] == "json_value_too_large"
    assert outcome.summary()["tool_calls"][0]["failure_code"] == "tool_output_too_large"


def test_a_tool_cannot_claim_a_failure_code_through_its_own_result() -> None:
    forged = Event(
        type="tool.call.failed",
        session_id="s1",
        payload={
            "tool_call_id": "c1",
            "result": {
                "content": "nope",
                "structured": {
                    "terminal_outcome": "invalid_tool_output",
                    "durable_value_error_code": "json_value_too_large",
                },
                "is_error": True,
            },
        },
    )

    (call,) = RunOutcome.from_events([forged]).tool_calls

    assert call.failure_code is None


def test_failure_code_is_found_in_groups_and_causes_but_not_in_subclasses() -> None:
    exhausted = ProviderStreamCapacityExhausted("capacity is exhausted")

    class _Lookalike(ProviderStreamCapacityExhausted):
        pass

    try:
        try:
            raise exhausted
        except ProviderStreamCapacityExhausted as error:
            raise RuntimeError("model step failed") from error
    except RuntimeError as wrapped:
        chained = wrapped

    assert failure_code_of(exhausted) is RunFailureCode.PROVIDER_STREAM_CAPACITY_EXHAUSTED
    assert failure_code_of(chained) is RunFailureCode.PROVIDER_STREAM_CAPACITY_EXHAUSTED
    assert (
        failure_code_of(ExceptionGroup("round", [ValueError("x"), exhausted]))
        is RunFailureCode.PROVIDER_STREAM_CAPACITY_EXHAUSTED
    )
    assert failure_code_of(_Lookalike("capacity is exhausted")) is None
    assert failure_code_of(RuntimeError("capacity is exhausted")) is None


def test_run_to_completion_codes_a_raised_failure() -> None:
    class _RaisingApp:
        async def run(self, request: RunRequest) -> AsyncIterator[Event]:
            del request
            raise ProviderStreamCapacityExhausted("capacity is exhausted")
            yield  # pragma: no cover

    outcome = asyncio.run(run_to_completion(_RaisingApp(), _request()))  # ty: ignore[invalid-argument-type]

    assert outcome.status is SessionStatus.FAILED
    assert outcome.failure_code == "provider_stream_capacity_exhausted"
