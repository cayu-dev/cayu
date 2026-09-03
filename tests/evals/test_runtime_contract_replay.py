from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest
from pydantic import SecretStr

from cayu.core.agents import AgentSpec
from cayu.core.billing import BillingIdentity, PricingContext
from cayu.core.events import EventType
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.messages import Message, MessageRole
from cayu.core.thinking import ThinkingConfig
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.evals import runtime_replay as runtime_replay_module
from cayu.evals.models import Trajectory, _trajectory_promotion_capture_sha256
from cayu.evals.runtime_replay import (
    RuntimeReplayBoundaryKind,
    RuntimeReplayBounds,
    RuntimeReplayDisposition,
    RuntimeReplayDivergenceKind,
    RuntimeReplayReason,
    RuntimeReplayRequest,
    replay_session,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.evals.trajectory import trajectory_from_session
from cayu.providers import ModelStreamEvent, ProviderStreamDeadlines
from cayu.runtime import CayuApp, RequestFootprintConfig, RunRequest
from cayu.runtime.budgets import BudgetLimit, BudgetPolicy, BudgetReservation
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.request_footprints import (
    RequestFingerprint,
    RequestFingerprintAvailability,
)
from cayu.runtime.sessions import ModelTarget
from cayu.runtime.tool_policy import StaticToolPolicy


class _WeatherTool(Tool):
    spec = ToolSpec(
        name="weather",
        description="Return recorded weather.",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        effect=ToolEffect.EXTERNAL,
    )

    def __init__(
        self,
        *,
        implementation_version: str | None = None,
        return_error: bool = False,
    ) -> None:
        if implementation_version is None:
            super().__init__()
        else:
            super().__init__(
                ToolSpec(
                    name=self.spec.name,
                    description=self.spec.description,
                    input_schema=self.spec.input_schema,
                    effect=self.spec.effect,
                    execution_profile_identity=ExecutionProfileBehaviorIdentity(
                        name="tests:runtime-replay:weather",
                        behavior_version="1",
                        implementation_version=implementation_version,
                    ),
                )
            )
        self.calls = 0
        self.return_error = return_error

    async def run(self, ctx: ToolContext, args: dict[str, object]) -> ToolResult:
        del ctx, args
        self.calls += 1
        return ToolResult(
            content="weather unavailable" if self.return_error else "sunny",
            structured=None if self.return_error else {"temperature_c": 21},
            is_error=self.return_error,
        )


class _RoundWeatherTool(Tool):
    def __init__(
        self,
        *,
        parallel_safe: bool,
        error_cities: frozenset[str] = frozenset(),
        include_artifacts: bool = False,
    ) -> None:
        super().__init__(
            ToolSpec(
                name="weather",
                description="Return recorded weather for one city.",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
                parallel_safe=parallel_safe,
                effect=ToolEffect.EXTERNAL,
            )
        )
        self.error_cities = error_cities
        self.include_artifacts = include_artifacts
        self.calls: list[str] = []
        self.active_calls = 0
        self.max_active_calls = 0

    async def run(self, ctx: ToolContext, args: dict[str, object]) -> ToolResult:
        del ctx
        city = args["city"]
        assert type(city) is str
        self.calls.append(city)
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            await asyncio.sleep(0.02 if city == "Bishkek" else 0.01)
            is_error = city in self.error_cities
            return ToolResult(
                content=f"{city} unavailable" if is_error else f"{city} sunny",
                structured=None if is_error else {"city": city, "temperature_c": 21},
                artifacts=(
                    [{"type": "weather-observation", "city": city}]
                    if self.include_artifacts
                    else []
                ),
                is_error=is_error,
            )
        finally:
            self.active_calls -= 1


def _candidate_app(
    *,
    system_prompt: str = "Answer using the weather tool.",
    tool_policy: StaticToolPolicy | None = None,
    fingerprint_key_id: str = "test-key",
    fingerprint_key: str = "x" * 32,
    tool_implementation_version: str | None = None,
    provider: ScriptedModelProvider | None = None,
    budget_policy: BudgetPolicy | None = None,
) -> tuple[CayuApp, _WeatherTool]:
    app = CayuApp(
        budget_policy=budget_policy,
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id=fingerprint_key_id,
            fingerprint_key=SecretStr(fingerprint_key),
        ),
        enable_logging=False,
    )
    app.register_provider(provider or ScriptedModelProvider(()), default=True)
    tool = _WeatherTool(
        implementation_version=tool_implementation_version,
    )
    app.register_agent(
        AgentSpec(
            name="weather-agent",
            model="test-model",
            system_prompt=system_prompt,
        ),
        tools=[tool],
        tool_policy=tool_policy,
    )
    return app, tool


async def _captured_multi_tool_round(
    *,
    parallel_safe: bool,
    error_cities: frozenset[str] = frozenset(),
    include_artifacts: bool = False,
    cities: tuple[str, ...] = ("Bishkek", "Osh"),
) -> tuple[CayuApp, _RoundWeatherTool, Trajectory]:
    provider = ScriptedModelProvider(
        (
            (
                *(
                    ModelStreamEvent.tool_call(
                        id=f"weather-call-{index}",
                        name="weather",
                        arguments={"city": city},
                    )
                    for index, city in enumerate(cities, start=1)
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ),
            (
                ModelStreamEvent.text_delta("Recorded both cities."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
        )
    )
    app = CayuApp(
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="test-key",
            fingerprint_key=SecretStr("x" * 32),
        ),
        max_parallel_tool_calls=4,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    tool = _RoundWeatherTool(
        parallel_safe=parallel_safe,
        error_cities=error_cities,
        include_artifacts=include_artifacts,
    )
    app.register_agent(
        AgentSpec(
            name="weather-agent",
            model="test-model",
            system_prompt="Answer using the weather tool.",
        ),
        tools=[tool],
    )
    session_id = (
        f"runtime-replay-{len(cities)}-call-{'parallel' if parallel_safe else 'sequential'}"
    )
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="weather-agent",
                session_id=session_id,
                messages=[Message.text(MessageRole.USER, "Compare both cities.")],
            )
        )
    ]
    assert events[-1].type is EventType.SESSION_COMPLETED
    return app, tool, await trajectory_from_session(app, session_id)


async def _captured_tool_round(
    *,
    tool_error: bool = False,
    run_request_options: Mapping[str, Any] | None = None,
    include_usage: bool = False,
    stream_deadlines: ProviderStreamDeadlines | None = None,
) -> tuple[CayuApp, _WeatherTool, Trajectory]:
    usage = {"input_tokens": 1, "output_tokens": 1}
    provider_type: type[ScriptedModelProvider] = ScriptedModelProvider
    if stream_deadlines is not None:

        class _DeadlineScriptedModelProvider(ScriptedModelProvider):
            @property
            def stream_deadlines(self) -> ProviderStreamDeadlines:
                return stream_deadlines

        provider_type = _DeadlineScriptedModelProvider
    provider = provider_type(
        (
            (
                ModelStreamEvent.tool_call(
                    id="weather-call-1",
                    name="weather",
                    arguments={"city": "Bishkek"},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        **({"usage": usage} if include_usage else {}),
                    }
                ),
            ),
            (
                ModelStreamEvent.text_delta("It is sunny."),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        **({"usage": usage} if include_usage else {}),
                    }
                ),
            ),
        )
    )
    app = CayuApp(
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="test-key",
            fingerprint_key=SecretStr("x" * 32),
        ),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    tool = _WeatherTool(return_error=tool_error)
    app.register_agent(
        AgentSpec(
            name="weather-agent",
            model="test-model",
            system_prompt="Answer using the weather tool.",
        ),
        tools=[tool],
    )
    session_id = "runtime-replay-source"
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="weather-agent",
                session_id=session_id,
                messages=[Message.text(MessageRole.USER, "How is the weather?")],
                **dict(run_request_options or {}),
            )
        )
    ]
    assert events
    return app, tool, await trajectory_from_session(app, session_id)


def test_runtime_replay_request_schema_declares_but_does_not_dump_trajectory() -> None:
    trajectory = Trajectory()
    request = RuntimeReplayRequest(trajectory=trajectory)
    schema = RuntimeReplayRequest.model_json_schema()

    assert request.trajectory is trajectory
    assert "trajectory" in schema["required"]
    assert schema["properties"]["trajectory"]["$ref"].endswith("/Trajectory")
    assert "trajectory" not in request.model_dump(mode="json")


def test_runtime_contract_replay_matches_without_reinvoking_external_tool() -> None:
    async def scenario():
        app, tool, trajectory = await _captured_tool_round()
        source_session_before = await app.session_store.load(trajectory.session.id)
        source_events_before = tuple(trajectory.events)
        source_transcript_before = await app.session_store.load_transcript(trajectory.session.id)
        source_provider = app._providers["scripted"].provider
        provider_requests_before = len(source_provider.requests)
        report = await app.replay_session(RuntimeReplayRequest(trajectory=trajectory))
        source_session_after = await app.session_store.load(trajectory.session.id)
        source_transcript_after = await app.session_store.load_transcript(trajectory.session.id)
        source_trajectory_after = await trajectory_from_session(app, trajectory.session.id)
        return (
            tool,
            report,
            source_session_before,
            source_session_after,
            source_events_before,
            source_trajectory_after.events,
            source_transcript_before,
            source_transcript_after,
            provider_requests_before,
            len(source_provider.requests),
        )

    (
        tool,
        report,
        source_session_before,
        source_session_after,
        source_events_before,
        source_events_after,
        source_transcript_before,
        source_transcript_after,
        provider_requests_before,
        provider_requests_after,
    ) = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.MATCHED, (
        report.first_divergence,
        report.changed_execution_profile_components,
    )
    assert report.reason is None
    assert report.compared_model_steps == 2
    assert report.compared_tool_rounds == 1
    assert len(report.request_attempts) == 2
    assert all(attempt.matched for attempt in report.request_attempts)
    assert all(
        attempt.source_provider_name == attempt.candidate_provider_name == "scripted"
        and attempt.source_model == attempt.candidate_model == "test-model"
        for attempt in report.request_attempts
    )
    assert report.source_execution_profile is not None
    assert report.candidate_execution_profile is not None
    assert tool.calls == 1
    assert provider_requests_before == provider_requests_after == 2
    assert "How is the weather?" not in report.model_dump_json()
    assert "Bishkek" not in report.model_dump_json()
    assert "sunny" not in report.model_dump_json()
    assert source_session_after == source_session_before
    assert source_transcript_after == source_transcript_before
    assert source_events_after == source_events_before


def test_runtime_contract_replay_preserves_nondefault_provider_stream_deadlines() -> None:
    deadlines = ProviderStreamDeadlines(
        transport_idle_timeout_s=11.0,
        protocol_idle_timeout_s=12.0,
        semantic_progress_timeout_s=13.0,
        absolute_stream_timeout_s=14.0,
    )

    async def scenario():
        app, _tool, trajectory = await _captured_tool_round(stream_deadlines=deadlines)
        return await replay_session(app, RuntimeReplayRequest(trajectory=trajectory))

    report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.MATCHED
    assert report.reason is None
    assert report.source_execution_profile == report.candidate_execution_profile


def test_runtime_contract_replay_substitutes_recorded_error_result() -> None:
    async def scenario():
        app, tool, trajectory = await _captured_tool_round(tool_error=True)
        report = await replay_session(app, RuntimeReplayRequest(trajectory=trajectory))
        return tool, report

    tool, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.MATCHED
    assert tool.calls == 1
    assert "weather unavailable" not in report.model_dump_json()


def test_runtime_contract_replay_matches_every_sequential_multi_call() -> None:
    async def scenario():
        app, tool, trajectory = await _captured_multi_tool_round(parallel_safe=False)
        terminal_events = tuple(
            event for event in trajectory.events if event.type is EventType.TOOL_CALL_COMPLETED
        )
        report = await replay_session(app, RuntimeReplayRequest(trajectory=trajectory))
        return app, tool, terminal_events, report

    app, tool, terminal_events, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.MATCHED
    assert tool.calls == ["Bishkek", "Osh"]
    assert tool.max_active_calls == 1
    assert [event.payload["arguments"] for event in terminal_events] == [
        {"city": "Bishkek"},
        {"city": "Osh"},
    ]
    assert all(event.payload["arguments_exact"] is True for event in terminal_events)
    assert len(app._providers["scripted"].provider.requests) == 2
    assert "Bishkek" not in report.model_dump_json()
    assert "Osh" not in report.model_dump_json()


def test_runtime_contract_replay_matches_parallel_mixed_tool_results() -> None:
    async def scenario():
        app, tool, trajectory = await _captured_multi_tool_round(
            parallel_safe=True,
            error_cities=frozenset({"Osh"}),
        )
        report = await replay_session(app, RuntimeReplayRequest(trajectory=trajectory))
        return tool, trajectory, report

    tool, trajectory, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.MATCHED
    assert set(tool.calls) == {"Bishkek", "Osh"}
    assert tool.max_active_calls == 2
    terminals = tuple(
        event
        for event in trajectory.events
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
    )
    assert {event.payload["arguments"]["city"]: event.type for event in terminals} == {
        "Bishkek": EventType.TOOL_CALL_COMPLETED,
        "Osh": EventType.TOOL_CALL_FAILED,
    }
    assert all(event.payload["arguments_exact"] is True for event in terminals)
    serialized = report.model_dump_json()
    assert "sunny" not in serialized
    assert "Osh unavailable" not in serialized


def test_runtime_contract_replay_rejects_nonexact_tool_argument_evidence() -> None:
    async def scenario():
        app, _tool, trajectory = await _captured_multi_tool_round(parallel_safe=False)
        terminal_index = next(
            index
            for index, event in enumerate(trajectory.events)
            if event.type is EventType.TOOL_CALL_COMPLETED
        )
        changed_events = list(trajectory.events)
        terminal = changed_events[terminal_index]
        changed_events[terminal_index] = terminal.model_copy(
            update={"payload": {**terminal.payload, "arguments_exact": False}},
            deep=True,
        )
        trajectory.events = tuple(changed_events)
        trajectory._promotion_capture_sha256 = _trajectory_promotion_capture_sha256(trajectory)
        return await replay_session(app, RuntimeReplayRequest(trajectory=trajectory))

    report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.UNAVAILABLE
    assert report.reason is (RuntimeReplayReason.SOURCE_TOOL_ARGUMENT_EVIDENCE_UNAVAILABLE)


def test_runtime_contract_replay_substitutes_recorded_artifact_results() -> None:
    async def scenario():
        app, tool, trajectory = await _captured_multi_tool_round(
            parallel_safe=True,
            include_artifacts=True,
        )
        report = await replay_session(app, RuntimeReplayRequest(trajectory=trajectory))
        return tool, report

    tool, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.MATCHED
    assert len(tool.calls) == 2
    serialized = report.model_dump_json()
    assert "weather-observation" not in serialized
    assert "Bishkek" not in serialized
    assert "Osh" not in serialized


def test_runtime_contract_replay_supports_rounds_above_the_default_call_bound() -> None:
    cities = tuple(f"City-{index}" for index in range(17))

    async def scenario():
        app, tool, trajectory = await _captured_multi_tool_round(
            parallel_safe=True,
            cities=cities,
        )
        report = await replay_session(
            app,
            RuntimeReplayRequest(
                trajectory=trajectory,
                bounds=RuntimeReplayBounds(max_tool_calls=len(cities)),
            ),
        )
        return tool, report

    tool, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.MATCHED
    assert set(tool.calls) == set(cities)
    assert len(report.supporting_source_event_ids) == 2 * len(cities) + 4
    assert "City-16" not in report.model_dump_json()


def test_runtime_contract_replay_preserves_recorded_contextual_billing_identity() -> None:
    identity = BillingIdentity(
        provider_name="scripted",
        resource_id="test-model",
        pricing_contexts=(PricingContext(dimensions={"zone": "north"}),),
    )

    class _ContextualBillingProvider(ScriptedModelProvider):
        def __init__(self, events):
            super().__init__(events)
            self.billing_request_calls = 0
            self.billing_completion_calls = 0

        async def billing_identity_for_request(self, request):
            assert request.model == "test-model"
            self.billing_request_calls += 1
            return identity

        def billing_identity_for_completion(self, request_identity, payload):
            del payload
            assert request_identity == identity
            self.billing_completion_calls += 1
            return request_identity

    async def scenario():
        provider = _ContextualBillingProvider(
            (
                (
                    ModelStreamEvent.tool_call(
                        id="weather-call-1",
                        name="weather",
                        arguments={"city": "Bishkek"},
                    ),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        }
                    ),
                ),
                (
                    ModelStreamEvent.text_delta("It is sunny."),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "stop",
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        }
                    ),
                ),
            )
        )
        price = ModelPrice.fixed(
            provider_name="scripted",
            model="test-model",
            match="exact",
            input_per_million=Decimal("1"),
            output_per_million=Decimal("1"),
            pricing_context={"zone": ("north",)},
        )
        price = price.model_copy(
            update={
                "schedules": tuple(
                    schedule.model_copy(update={"effective_from": date(2026, 1, 1)})
                    for schedule in price.schedules
                )
            },
            deep=True,
        )
        pricing = PriceBook(prices=(price,))
        app = CayuApp(
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("10"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=100,
                            max_output_tokens=100,
                        ),
                    ),
                )
            ),
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-key",
                fingerprint_key=SecretStr("x" * 32),
            ),
            enable_logging=False,
            clock=lambda: datetime(2026, 9, 1, tzinfo=UTC),
        )
        app.register_provider(provider, default=True)
        tool = _WeatherTool()
        app.register_agent(
            AgentSpec(name="weather-agent", model="test-model"),
            tools=[tool],
        )
        session_id = "runtime-replay-contextual-billing"
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="weather-agent",
                    session_id=session_id,
                    messages=[Message.text(MessageRole.USER, "How is the weather?")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        trajectory = await trajectory_from_session(app, session_id)
        report = await replay_session(app, RuntimeReplayRequest(trajectory=trajectory))
        changed_events = []
        for event in trajectory.events:
            if event.type is not EventType.BUDGET_RESERVED:
                changed_events.append(event)
                continue
            payload = dict(event.payload)
            payload.pop("billing_identity", None)
            changed_events.append(event.model_copy(update={"payload": payload}, deep=True))
        trajectory.events = tuple(changed_events)
        trajectory._promotion_capture_sha256 = _trajectory_promotion_capture_sha256(trajectory)
        unavailable = await replay_session(app, RuntimeReplayRequest(trajectory=trajectory))
        return provider, tool, report, unavailable

    provider, tool, report, unavailable = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.MATCHED
    assert unavailable.disposition is RuntimeReplayDisposition.UNAVAILABLE
    assert unavailable.reason is RuntimeReplayReason.SOURCE_BILLING_EVIDENCE_UNAVAILABLE
    assert len(provider.requests) == 2
    assert provider.billing_request_calls == 2
    assert provider.billing_completion_calls == 2
    assert tool.calls == 1


def test_runtime_contract_replay_preserves_source_resolved_model_target() -> None:
    async def scenario():
        default_provider = ScriptedModelProvider((), name="default")
        selected_provider = ScriptedModelProvider(
            (
                (
                    ModelStreamEvent.tool_call(
                        id="weather-call-1",
                        name="weather",
                        arguments={"city": "Bishkek"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ),
                (
                    ModelStreamEvent.text_delta("It is sunny."),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ),
            ),
            name="selected",
        )
        app = CayuApp(
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-key",
                fingerprint_key=SecretStr("x" * 32),
            ),
            enable_logging=False,
        )
        app.register_provider(default_provider, default=True)
        app.register_provider(selected_provider)
        tool = _WeatherTool()
        app.register_agent(
            AgentSpec(name="weather-agent", model="default-model"),
            tools=[tool],
        )
        session_id = "runtime-replay-explicit-target"
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="weather-agent",
                    session_id=session_id,
                    target=ModelTarget(
                        provider_name="selected",
                        model="selected-model",
                    ),
                    messages=[Message.text(MessageRole.USER, "How is the weather?")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        trajectory = await trajectory_from_session(app, session_id)
        report = await replay_session(app, RuntimeReplayRequest(trajectory=trajectory))
        return default_provider, selected_provider, tool, report

    default_provider, selected_provider, tool, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.MATCHED
    assert default_provider.requests == []
    assert len(selected_provider.requests) == 2
    assert tool.calls == 1


def test_runtime_contract_replay_does_not_invent_missing_invocation_semantics() -> None:
    request_budget = BudgetLimit(
        scope="session",
        max_estimated_cost=Decimal("10"),
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="scripted",
                    model="test-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        ),
    )

    async def scenario():
        cases = (
            (
                {"budget_limits": (request_budget,)},
                True,
                "invocation_budget_policy",
            ),
            ({"max_steps": 2}, False, "finalization"),
            (
                {"thinking": ThinkingConfig(effort="low")},
                False,
                "provider_request_policy",
            ),
        )
        results = []
        for options, include_usage, expected_component in cases:
            app, tool, trajectory = await _captured_tool_round(
                run_request_options=options,
                include_usage=include_usage,
            )
            report = await replay_session(app, RuntimeReplayRequest(trajectory=trajectory))
            results.append((app, tool, report, expected_component))
        return results

    results = asyncio.run(scenario())

    for app, tool, report, expected_component in results:
        assert report.disposition is RuntimeReplayDisposition.UNAVAILABLE
        assert report.reason is RuntimeReplayReason.SOURCE_INVOCATION_EVIDENCE_UNAVAILABLE
        assert report.changed_execution_profile_components == (expected_component,)
        provider = app._providers["scripted"].provider
        assert isinstance(provider, ScriptedModelProvider)
        assert len(provider.requests) == 2
        assert tool.calls == 1


def test_runtime_contract_replay_reports_first_request_context_drift() -> None:
    async def scenario():
        _source_app, source_tool, trajectory = await _captured_tool_round()
        candidate, candidate_tool = _candidate_app(
            system_prompt="Answer briefly, then use the weather tool."
        )
        report = await replay_session(candidate, RuntimeReplayRequest(trajectory=trajectory))
        return source_tool, candidate_tool, report

    source_tool, candidate_tool, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.DIVERGED
    assert report.first_divergence is not None
    assert report.first_divergence.kind is RuntimeReplayDivergenceKind.REQUEST_FOOTPRINT_MISMATCH
    assert report.first_divergence.boundary is RuntimeReplayBoundaryKind.MODEL_REQUEST
    assert report.first_divergence.index == 1
    assert source_tool.calls == 1
    assert candidate_tool.calls == 0


def test_runtime_contract_replay_preserves_first_request_drift_after_partial_execution() -> None:
    price_book = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="scripted",
                model="test-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
        )
    )
    budget_policy = BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("0.000003"),
                pricing=price_book,
                reservation=BudgetReservation(
                    max_input_tokens=1,
                    max_output_tokens=1,
                ),
            ),
        )
    )

    async def scenario():
        _source_app, _source_tool, trajectory = await _captured_tool_round(include_usage=True)
        candidate, candidate_tool = _candidate_app(
            system_prompt="Answer briefly, then use the weather tool.",
            budget_policy=budget_policy,
        )
        report = await replay_session(candidate, RuntimeReplayRequest(trajectory=trajectory))
        return candidate_tool, report

    candidate_tool, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.DIVERGED
    assert report.reason is None
    assert report.first_divergence is not None
    assert report.first_divergence.kind is (RuntimeReplayDivergenceKind.REQUEST_FOOTPRINT_MISMATCH)
    assert report.first_divergence.index == 1
    assert report.compared_model_steps == 1
    assert candidate_tool.calls == 0


def test_runtime_contract_replay_reports_removed_source_tool_as_profile_drift() -> None:
    async def scenario():
        _source_app, _source_tool, trajectory = await _captured_tool_round()
        candidate = CayuApp(
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-key",
                fingerprint_key=SecretStr("x" * 32),
            ),
            enable_logging=False,
        )
        candidate.register_provider(ScriptedModelProvider(()), default=True)
        candidate.register_agent(
            AgentSpec(
                name="weather-agent",
                model="test-model",
                system_prompt="Answer using the weather tool.",
            )
        )
        return await replay_session(candidate, RuntimeReplayRequest(trajectory=trajectory))

    report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.DIVERGED
    assert report.reason is None
    assert report.first_divergence is not None
    assert report.first_divergence.kind is (RuntimeReplayDivergenceKind.EXECUTION_PROFILE_MISMATCH)
    assert report.first_divergence.boundary is RuntimeReplayBoundaryKind.EXECUTION_PROFILE
    assert report.candidate_execution_profile is not None
    assert report.compared_model_steps == 0
    assert report.compared_tool_rounds == 0


def test_runtime_contract_replay_reports_policy_drift_before_second_request() -> None:
    async def scenario():
        _source_app, source_tool, trajectory = await _captured_tool_round()
        candidate, candidate_tool = _candidate_app(tool_policy=StaticToolPolicy(deny={"weather"}))
        report = await replay_session(candidate, RuntimeReplayRequest(trajectory=trajectory))
        return source_tool, candidate_tool, report

    source_tool, candidate_tool, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.DIVERGED
    assert report.first_divergence is not None
    assert report.first_divergence.kind is RuntimeReplayDivergenceKind.POLICY_DECISION_MISMATCH
    assert report.first_divergence.boundary is RuntimeReplayBoundaryKind.TOOL_POLICY
    assert source_tool.calls == 1
    assert candidate_tool.calls == 0


def test_runtime_contract_replay_identifies_the_exact_multi_call_policy_drift() -> None:
    class _NamedTool(Tool):
        def __init__(self, name: str) -> None:
            super().__init__(
                ToolSpec(
                    name=name,
                    description=f"Return the recorded {name} result.",
                    input_schema={"type": "object", "additionalProperties": False},
                    effect=ToolEffect.EXTERNAL,
                )
            )
            self.calls = 0

        async def run(self, ctx: ToolContext, args: dict[str, object]) -> ToolResult:
            del ctx, args
            self.calls += 1
            return ToolResult(content=f"{self.name} complete")

    async def scenario():
        provider = ScriptedModelProvider(
            (
                (
                    ModelStreamEvent.tool_call(id="first-call", name="first", arguments={}),
                    ModelStreamEvent.tool_call(id="second-call", name="second", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ),
                (
                    ModelStreamEvent.text_delta("Both complete."),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ),
            )
        )
        source = CayuApp(
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-key",
                fingerprint_key=SecretStr("x" * 32),
            ),
            enable_logging=False,
        )
        source.register_provider(provider, default=True)
        source_tools = (_NamedTool("first"), _NamedTool("second"))
        source.register_agent(
            AgentSpec(name="weather-agent", model="test-model"),
            tools=source_tools,
        )
        session_id = "runtime-replay-second-call-policy-drift"
        events = [
            event
            async for event in source.run(
                RunRequest(
                    agent_name="weather-agent",
                    session_id=session_id,
                    messages=[Message.text(MessageRole.USER, "Run both.")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        trajectory = await trajectory_from_session(source, session_id)

        candidate = CayuApp(
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-key",
                fingerprint_key=SecretStr("x" * 32),
            ),
            enable_logging=False,
        )
        candidate.register_provider(ScriptedModelProvider(()), default=True)
        candidate_tools = (_NamedTool("first"), _NamedTool("second"))
        candidate.register_agent(
            AgentSpec(name="weather-agent", model="test-model"),
            tools=candidate_tools,
            tool_policy=StaticToolPolicy(deny={"second"}),
        )
        report = await replay_session(
            candidate,
            RuntimeReplayRequest(trajectory=trajectory),
        )
        return candidate_tools, report

    candidate_tools, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.DIVERGED
    assert report.first_divergence is not None
    assert report.first_divergence.kind is (RuntimeReplayDivergenceKind.POLICY_DECISION_MISMATCH)
    assert report.first_divergence.boundary is RuntimeReplayBoundaryKind.TOOL_POLICY
    assert report.first_divergence.index == 2
    assert all(tool.calls == 0 for tool in candidate_tools)


def test_runtime_contract_replay_fails_closed_when_footprint_evidence_is_missing() -> None:
    async def scenario():
        app, _tool, trajectory = await _captured_tool_round()
        trajectory.events = tuple(
            event for event in trajectory.events if event.type != "request.footprint.recorded"
        )
        trajectory._promotion_capture_sha256 = _trajectory_promotion_capture_sha256(trajectory)
        return await replay_session(app, RuntimeReplayRequest(trajectory=trajectory))

    report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.UNAVAILABLE
    assert report.reason is RuntimeReplayReason.SOURCE_REQUEST_FOOTPRINT_UNAVAILABLE


def test_runtime_contract_replay_rejects_incomplete_available_fingerprint() -> None:
    malformed = RequestFingerprint.model_construct(
        availability=RequestFingerprintAvailability.AVAILABLE,
        value="a" * 64,
        algorithm=None,
        key_id="test-key",
        canonicalization_version=1,
        unavailable_reason=None,
    )

    with pytest.raises(runtime_replay_module._ReplayUnavailable) as exc_info:
        runtime_replay_module._available_identity(malformed)

    assert exc_info.value.reason is RuntimeReplayReason.SOURCE_REQUEST_FOOTPRINT_UNAVAILABLE


def test_runtime_contract_replay_rejects_detached_trajectory_without_input_authority() -> None:
    async def scenario():
        app, _tool, trajectory = await _captured_tool_round()
        detached = Trajectory.model_validate(trajectory.model_dump(mode="python"))
        return await replay_session(app, RuntimeReplayRequest(trajectory=detached))

    report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.UNAVAILABLE
    assert report.reason is RuntimeReplayReason.SOURCE_INPUT_EVIDENCE_UNAVAILABLE


def test_runtime_contract_replay_serializes_typed_failure_for_malformed_trajectory() -> None:
    async def scenario():
        app, _tool, trajectory = await _captured_tool_round()
        trajectory.metadata = {"attacker-marker": object()}
        return await replay_session(app, RuntimeReplayRequest(trajectory=trajectory))

    report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.UNAVAILABLE
    assert report.reason is RuntimeReplayReason.SOURCE_TRAJECTORY_INVALID
    assert report.trajectory_identity is None
    assert "attacker-marker" not in report.model_dump_json()


def test_runtime_contract_replay_enforces_source_event_bound_before_execution() -> None:
    async def scenario():
        app, tool, trajectory = await _captured_tool_round()
        report = await replay_session(
            app,
            RuntimeReplayRequest(
                trajectory=trajectory,
                bounds=RuntimeReplayBounds(max_events=len(trajectory.events) - 1),
            ),
        )
        return tool, report

    tool, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.BOUNDS_EXCEEDED
    assert report.reason is RuntimeReplayReason.EVENT_BOUND_EXCEEDED
    assert tool.calls == 1


def test_runtime_contract_replay_rejects_incomparable_fingerprint_keys() -> None:
    async def scenario():
        _source_app, _source_tool, trajectory = await _captured_tool_round()
        candidate, candidate_tool = _candidate_app(
            fingerprint_key_id="different-key",
            fingerprint_key="y" * 32,
        )
        report = await replay_session(candidate, RuntimeReplayRequest(trajectory=trajectory))
        return candidate_tool, report

    candidate_tool, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.UNAVAILABLE
    assert report.reason is RuntimeReplayReason.REQUEST_FINGERPRINT_INCOMPARABLE
    assert candidate_tool.calls == 0


def test_runtime_contract_replay_rejects_conflicting_recorded_tool_identity() -> None:
    async def scenario():
        app, source_tool, trajectory = await _captured_tool_round()
        completed_index = next(
            index
            for index, event in enumerate(trajectory.events)
            if event.type is EventType.TOOL_CALL_COMPLETED
        )
        completed = trajectory.events[completed_index]
        changed_events = list(trajectory.events)
        changed_events[completed_index] = completed.model_copy(
            update={
                "payload": {
                    **completed.payload,
                    "arguments": {"city": "Osh"},
                }
            },
            deep=True,
        )
        trajectory.events = tuple(changed_events)
        trajectory._promotion_capture_sha256 = _trajectory_promotion_capture_sha256(trajectory)
        report = await replay_session(app, RuntimeReplayRequest(trajectory=trajectory))
        return source_tool, report

    source_tool, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.UNAVAILABLE
    assert report.reason is RuntimeReplayReason.SOURCE_TOOL_EVIDENCE_UNAVAILABLE
    assert source_tool.calls == 1


def test_runtime_contract_replay_rejects_duplicate_recorded_tool_outcome() -> None:
    async def scenario():
        app, source_tool, trajectory = await _captured_tool_round()
        completed_index = next(
            index
            for index, event in enumerate(trajectory.events)
            if event.type is EventType.TOOL_CALL_COMPLETED
        )
        duplicate = trajectory.events[completed_index].model_copy(
            update={"id": "duplicate-tool-terminal"},
            deep=True,
        )
        changed_events = list(trajectory.events)
        changed_events.insert(completed_index + 1, duplicate)
        trajectory.events = tuple(changed_events)
        trajectory._promotion_capture_sha256 = _trajectory_promotion_capture_sha256(trajectory)
        report = await replay_session(app, RuntimeReplayRequest(trajectory=trajectory))
        return source_tool, report

    source_tool, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.UNAVAILABLE
    assert report.reason is RuntimeReplayReason.SOURCE_TOOL_EVIDENCE_UNAVAILABLE
    assert source_tool.calls == 1


def test_runtime_contract_replay_reports_profile_only_drift_after_matching_path() -> None:
    async def scenario():
        _source_app, _source_tool, trajectory = await _captured_tool_round()
        candidate, candidate_tool = _candidate_app(tool_implementation_version="2")
        report = await replay_session(candidate, RuntimeReplayRequest(trajectory=trajectory))
        return candidate_tool, report

    candidate_tool, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.DIVERGED
    assert report.first_divergence is not None
    assert report.first_divergence.kind is (RuntimeReplayDivergenceKind.EXECUTION_PROFILE_MISMATCH)
    assert report.first_divergence.boundary is RuntimeReplayBoundaryKind.EXECUTION_PROFILE
    assert all(attempt.matched for attempt in report.request_attempts)
    assert candidate_tool.calls == 0


def test_runtime_contract_replay_reports_candidate_execution_failure() -> None:
    class _FailingPreflightProvider(ScriptedModelProvider):
        def preflight_model_target(self, *, model: str) -> None:
            del model
            raise RuntimeError("sensitive provider failure")

    async def scenario():
        _source_app, _source_tool, trajectory = await _captured_tool_round()
        candidate, candidate_tool = _candidate_app(provider=_FailingPreflightProvider(()))
        report = await replay_session(candidate, RuntimeReplayRequest(trajectory=trajectory))
        return candidate_tool, report

    candidate_tool, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.FAILED
    assert report.reason is RuntimeReplayReason.CANDIDATE_EXECUTION_FAILED
    assert "sensitive provider failure" not in report.model_dump_json()
    assert candidate_tool.calls == 0


def test_runtime_contract_replay_enforces_every_source_work_bound() -> None:
    async def scenario():
        app, tool, trajectory = await _captured_tool_round()
        cases = (
            (
                RuntimeReplayBounds(max_transcript_messages=len(trajectory.transcript) - 1),
                RuntimeReplayReason.TRANSCRIPT_BOUND_EXCEEDED,
            ),
            (
                RuntimeReplayBounds(max_model_steps=1),
                RuntimeReplayReason.MODEL_STEP_BOUND_EXCEEDED,
            ),
            (
                RuntimeReplayBounds(max_tool_calls=0),
                RuntimeReplayReason.TOOL_CALL_BOUND_EXCEEDED,
            ),
        )
        reports = [
            await replay_session(
                app,
                RuntimeReplayRequest(trajectory=trajectory, bounds=bounds),
            )
            for bounds, _reason in cases
        ]
        return tool, cases, reports

    tool, cases, reports = asyncio.run(scenario())

    assert [report.disposition for report in reports] == [
        RuntimeReplayDisposition.BOUNDS_EXCEEDED
    ] * len(cases)
    assert [report.reason for report in reports] == [reason for _bounds, reason in cases]
    assert tool.calls == 1


def test_runtime_contract_replay_enforces_wall_clock_bound() -> None:
    async def scenario():
        app, tool, trajectory = await _captured_tool_round()
        report = await replay_session(
            app,
            RuntimeReplayRequest(
                trajectory=trajectory,
                bounds=RuntimeReplayBounds(timeout_seconds=0.000_001),
            ),
        )
        return tool, report

    tool, report = asyncio.run(scenario())

    assert report.disposition is RuntimeReplayDisposition.BOUNDS_EXCEEDED
    assert report.reason is RuntimeReplayReason.WALL_CLOCK_BOUND_EXCEEDED
    assert tool.calls == 1
