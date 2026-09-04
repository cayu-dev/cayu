from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, StrictBool, StrictInt

from cayu import (
    AgentSpec,
    CacheBreakpoint,
    CachePolicy,
    CayuApp,
    Event,
    EventType,
    Message,
    ModelPrice,
    ModelRequest,
    ModelStreamEvent,
    PriceBook,
    RequestFingerprintAvailability,
    RequestFootprint,
    RequestFootprintConfig,
    RunRequest,
    ScriptedModelProvider,
    StaticToolExposurePolicy,
    Tool,
    ToolContext,
    ToolEffect,
    ToolExposureDecision,
    ToolExposurePolicy,
    ToolExposurePolicyRequest,
    ToolResult,
    ToolSpec,
)
from cayu.evals.runner import final_output_text

_PROVIDER_NAME = "tool-exposure-economics"
_MODEL = "fixture-model"
_EXPECTED_OUTPUT = "quality-ok"


class ExposureEconomicsSide(BaseModel):
    """Bounded evidence for one side of the deterministic comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: Literal["stable_broad", "changing_narrow"]
    requests: StrictInt = Field(ge=0)
    retries: StrictInt = Field(ge=0)
    exposure_profiles: tuple[str, ...]
    profile_changes: StrictInt = Field(ge=0)
    tool_manifest_fingerprints: tuple[str, ...]
    cache_prefix_fingerprints: tuple[str | None, ...]
    input_tokens: StrictInt = Field(ge=0)
    output_tokens: StrictInt = Field(ge=0)
    cache_read_input_tokens: StrictInt = Field(ge=0)
    cache_write_input_tokens: StrictInt = Field(ge=0)
    uncached_input_tokens: StrictInt = Field(ge=0)
    quality_passed: StrictBool
    estimated_cost: Decimal = Field(ge=0)
    currency: str


class ToolExposureEconomicsReport(BaseModel):
    """Paired fixture result; deliberately not a universal savings claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    evidence_scope: Literal["deterministic_fixture"] = "deterministic_fixture"
    universal_savings_claimed: Literal[False] = False
    stable_broad: ExposureEconomicsSide
    changing_narrow: ExposureEconomicsSide


class _FixtureTool(Tool):
    def __init__(self, name: str) -> None:
        self.spec = ToolSpec(
            name=name,
            description=f"Fixture {name} capability.",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.NONE,
        )
        super().__init__()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        return ToolResult(content=f"{self.name}-ok")


class _ChangingNarrowPolicy(ToolExposurePolicy):
    def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
        if request.step == 1:
            return ToolExposureDecision(
                profile_id="narrow-phase-one",
                tool_names=("inspect",),
            )
        return ToolExposureDecision(
            profile_id="narrow-phase-two",
            tool_names=("publish",),
        )


class _EconomicsProvider(ScriptedModelProvider):
    def request_cache_policy(self, request: ModelRequest) -> CachePolicy:
        del request
        return CachePolicy(
            breakpoints=(
                CacheBreakpoint.SYSTEM_PROMPT,
                CacheBreakpoint.TOOL_DEFINITIONS,
                CacheBreakpoint.CONVERSATION_PREFIX,
            ),
            conversation_prefix_strategy="all_but_last",
        )


def _completed_usage(
    *,
    finish_reason: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> ModelStreamEvent:
    return ModelStreamEvent.completed(
        {
            "finish_reason": finish_reason,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read_tokens,
                "cache_creation_input_tokens": cache_write_tokens,
            },
        }
    )


def _provider(strategy: Literal["stable_broad", "changing_narrow"]) -> _EconomicsProvider:
    if strategy == "stable_broad":
        first_usage = (200, 8, 0, 800)
        second_usage = (200, 24, 800, 0)
    else:
        first_usage = (650, 8, 0, 0)
        second_usage = (700, 24, 0, 0)
    return _EconomicsProvider(
        (
            (
                ModelStreamEvent.tool_call(id="inspect-1", name="inspect", arguments={}),
                _completed_usage(
                    finish_reason="tool_calls",
                    input_tokens=first_usage[0],
                    output_tokens=first_usage[1],
                    cache_read_tokens=first_usage[2],
                    cache_write_tokens=first_usage[3],
                ),
            ),
            (
                ModelStreamEvent.text_delta(_EXPECTED_OUTPUT),
                _completed_usage(
                    finish_reason="stop",
                    input_tokens=second_usage[0],
                    output_tokens=second_usage[1],
                    cache_read_tokens=second_usage[2],
                    cache_write_tokens=second_usage[3],
                ),
            ),
        ),
        name=_PROVIDER_NAME,
    )


def _price_book() -> PriceBook:
    return PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name=_PROVIDER_NAME,
                model=_MODEL,
                match="exact",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
                cache_read_input_per_million=Decimal("0.1"),
                cache_write_input_per_million=Decimal("1.25"),
            ),
        )
    )


def _available_fingerprint(payload: dict, field_name: str) -> str:
    fingerprint = payload["fingerprints"][field_name]
    if fingerprint["availability"] != RequestFingerprintAvailability.AVAILABLE.value:
        raise RuntimeError(f"Expected an available {field_name} fingerprint.")
    return str(fingerprint["value"])


async def _run_side(
    strategy: Literal["stable_broad", "changing_narrow"],
) -> ExposureEconomicsSide:
    provider = _provider(strategy)
    app = CayuApp(
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="tool-exposure-fixture",
            fingerprint_key=SecretStr("fixture-tool-exposure-key-material-0001"),
        ),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model=_MODEL,
            system_prompt="Complete the deterministic fixture.",
        ),
        tools=[_FixtureTool("inspect"), _FixtureTool("publish")],
        tool_exposure_policy=(
            StaticToolExposurePolicy(
                profile_id="stable-broad",
                tools=("inspect", "publish"),
            )
            if strategy == "stable_broad"
            else _ChangingNarrowPolicy()
        ),
    )
    session_id = f"tool-exposure-economics-{strategy}"
    events: list[Event] = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "Run the paired fixture.")],
                max_steps=2,
            )
        )
    ]
    transcript = await app.session_store.load_transcript(session_id)
    quality_passed = final_output_text(transcript) == _EXPECTED_OUTPUT
    footprints = [
        RequestFootprint.model_validate(event.payload)
        for event in events
        if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
    ]
    profiles = [
        str(event.payload["profile_id"])
        for event in events
        if event.type is EventType.TOOL_EXPOSURE_RECORDED
    ]
    profile_changes = sum(
        event.payload["profile_changed"] is True
        for event in events
        if event.type is EventType.TOOL_EXPOSURE_RECORDED
    )
    usage = await app.get_session_usage(session_id)
    cost = await app.get_session_cost(session_id, _price_book())
    cache_prefixes: list[str | None] = []
    for footprint in footprints:
        cache_prefix_breakpoints = [
            item
            for item in footprint.cache_breakpoints
            if item.kind is CacheBreakpoint.CONVERSATION_PREFIX
        ]
        if len(cache_prefix_breakpoints) != 1:
            raise RuntimeError("Expected one conversation-prefix cache breakpoint.")
        fingerprint = cache_prefix_breakpoints[0].fingerprint
        cache_prefixes.append(
            fingerprint.value
            if fingerprint.availability is RequestFingerprintAvailability.AVAILABLE
            else None
        )
    return ExposureEconomicsSide(
        strategy=strategy,
        requests=len(provider.requests),
        retries=sum(event.type is EventType.MODEL_RETRY for event in events),
        exposure_profiles=tuple(profiles),
        profile_changes=profile_changes,
        tool_manifest_fingerprints=tuple(
            _available_fingerprint(event.payload, "tool_manifest")
            for event in events
            if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
        ),
        cache_prefix_fingerprints=tuple(cache_prefixes),
        input_tokens=usage.usage.input_tokens,
        output_tokens=usage.usage.output_tokens,
        cache_read_input_tokens=usage.usage.cache.read_tokens,
        cache_write_input_tokens=usage.usage.cache.write_tokens,
        uncached_input_tokens=usage.usage.cache.uncached_input_tokens,
        quality_passed=quality_passed,
        estimated_cost=cost.total_cost,
        currency=cost.currency,
    )


async def run_scenario() -> ToolExposureEconomicsReport:
    """Run both sides under one explicit deterministic fixture contract."""

    stable_broad, changing_narrow = await asyncio.gather(
        _run_side("stable_broad"),
        _run_side("changing_narrow"),
    )
    return ToolExposureEconomicsReport(
        stable_broad=stable_broad,
        changing_narrow=changing_narrow,
    )
