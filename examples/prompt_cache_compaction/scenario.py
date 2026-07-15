from __future__ import annotations

from collections.abc import AsyncIterator
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from examples._advanced_support import (
    ScenarioResult,
    collect_events,
    paired_cost_evidence,
    session_evidence,
)

from cayu import (
    AgentSpec,
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    Event,
    EventType,
    Message,
    ModelCompactor,
    PriceBook,
    PromptCacheCompactor,
    ResumeRequest,
    RunRequest,
    TextPart,
    ThinkingConfig,
    Tool,
    ToolContext,
    ToolResult,
    ToolResultPart,
    ToolSpec,
    estimate_session_cost,
)
from cayu.artifacts import RESOLVED_FILE_ATTACHMENTS_OPTION
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent

RETENTION_TOKEN = "CACHE_RETENTION_OK"
_CACHE_COMPACTION_INSTRUCTION = (
    "Summarize the conversation above so a future agent step can continue. "
    "Preserve the exact value following MANDATORY_RETENTION_TOKEN when it appears. "
    "Preserve concrete user requests, decisions, tool results, errors, and pending work. "
    "Do not invent facts. Do not call tools. Return only the concise summary text."
)
_BOUNDED_BASELINE_SYSTEM_PROMPT = (
    "Summarize prior session context for a future model call. "
    "Preserve the exact value following MANDATORY_RETENTION_TOKEN when it appears. "
    "Return only the compact summary. Do not call tools."
)


class RecordingProvider(ModelProvider):
    """Records the provider-neutral request while preserving adapter behavior."""

    def __init__(self, delegate: ModelProvider) -> None:
        self.delegate = delegate
        self.name = delegate.name
        self.usage_dialect = delegate.usage_dialect
        self.supports_native_structured_output = delegate.supports_native_structured_output
        self.requests: list[ModelRequest] = []

    @property
    def context_pressure_profile(self):
        return self.delegate.context_pressure_profile

    def preflight_native_structured_output_schema(self, json_schema: dict[str, Any]) -> None:
        self.delegate.preflight_native_structured_output_schema(json_schema)

    async def count_input_tokens(self, request: ModelRequest):
        return await self.delegate.count_input_tokens(request)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request.model_copy(deep=True))
        async for event in self.delegate.stream(request):
            yield event


class _RecordingModelCompactor(ModelCompactor):
    """Records which public compaction request the bounded control received."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.request: CompactionRequest | None = None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.request = request
        return await super().compact(request)


class PairedPromptCacheCompactor(PromptCacheCompactor):
    """Captures the first cache-aware source for a post-session bounded control.

    Only the cache-aware result runs inside the session and becomes checkpoint
    state. The scenario harness runs the bounded control after the durable
    session completes, so comparison-only spend cannot be mistaken for a model
    step governed or attributed by the candidate session.
    """

    def __init__(self, *, provider: ModelProvider) -> None:
        super().__init__(
            provider=provider,
            compaction_instruction=_CACHE_COMPACTION_INSTRUCTION,
        )
        self.paired_request: CompactionRequest | None = None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        if request.existing_summary is None and self.paired_request is None:
            self.paired_request = request
        return await super().compact(request)


class LoadStableContextTool(Tool):
    spec = ToolSpec(
        name="load_stable_context",
        description="Load the stable context used by the cache-compaction verification.",
        input_schema={
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
            "additionalProperties": False,
        },
    )

    def __init__(self, stable_context: str) -> None:
        self.stable_context = stable_context
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls += 1
        return ToolResult(content=self.stable_context)


def stable_context(*, lines: int) -> str:
    if lines < 1:
        raise ValueError("lines must be positive")
    return "\n".join(
        [f"MANDATORY_RETENTION_TOKEN: {RETENTION_TOKEN}"]
        + [
            f"CACHE_SENTINEL_{index:04d}: stable provider context group {index % 17}."
            for index in range(lines)
        ]
    )


def _compaction_model_events(events: list[Event]) -> list[Event]:
    return [
        event
        for event in events
        if event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]


def _cache_read_tokens(event: Event) -> int:
    usage_metrics = event.payload.get("usage_metrics")
    if not isinstance(usage_metrics, dict):
        return 0
    cache = usage_metrics.get("cache")
    if not isinstance(cache, dict):
        return 0
    value = cache.get("read_tokens")
    return value if type(value) is int else 0


def _usage_snapshot_payload(payload: dict[str, Any]) -> dict[str, int | bool | None]:
    usage_metrics = payload.get("usage_metrics")
    if not isinstance(usage_metrics, dict):
        return {
            "usage_available": False,
            "cache_usage_available": False,
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_output_tokens": None,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
            "uncached_input_tokens": None,
        }
    input_tokens = usage_metrics.get("input_tokens")
    output_tokens = usage_metrics.get("output_tokens")
    reasoning_output_tokens = usage_metrics.get("reasoning_output_tokens")
    cache = usage_metrics.get("cache")
    cache_usage_available = _raw_cache_counters_reported(payload)
    cache_read_tokens = cache.get("read_tokens") if isinstance(cache, dict) else None
    cache_write_tokens = cache.get("write_tokens") if isinstance(cache, dict) else None
    uncached_input_tokens = cache.get("uncached_input_tokens") if isinstance(cache, dict) else None
    return {
        "usage_available": True,
        "cache_usage_available": cache_usage_available,
        "input_tokens": input_tokens if type(input_tokens) is int else None,
        "output_tokens": output_tokens if type(output_tokens) is int else None,
        "reasoning_output_tokens": (
            reasoning_output_tokens if type(reasoning_output_tokens) is int else None
        ),
        "cache_read_tokens": cache_read_tokens if type(cache_read_tokens) is int else None,
        "cache_write_tokens": cache_write_tokens if type(cache_write_tokens) is int else None,
        "uncached_input_tokens": (
            uncached_input_tokens if type(uncached_input_tokens) is int else None
        ),
    }


def _raw_cache_counters_reported(payload: dict[str, Any]) -> bool:
    raw_usage = payload.get("usage")
    if not isinstance(raw_usage, dict):
        return False
    if any(
        key in raw_usage
        for key in (
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "cacheReadInputTokens",
            "cacheWriteInputTokens",
        )
    ):
        return True
    if isinstance(raw_usage.get("cache_creation"), dict):
        return True
    for details_key in ("input_tokens_details", "prompt_tokens_details"):
        details = raw_usage.get(details_key)
        if isinstance(details, dict) and "cached_tokens" in details:
            return True
    return False


def _usage_snapshot(event: Event) -> dict[str, int | bool | None]:
    return _usage_snapshot_payload(event.payload)


def _paired_cost_evidence(
    *,
    candidate_event: Event | None,
    baseline_payload: dict[str, Any] | None,
    price_book: PriceBook | None,
) -> dict[str, Any]:
    if candidate_event is None or baseline_payload is None:
        return paired_cost_evidence(
            candidate=None,
            baseline=None,
            price_book=price_book,
            baseline_cost_field="bounded_baseline_cost",
        )
    if price_book is None:
        return paired_cost_evidence(
            candidate=(),
            baseline=(),
            price_book=None,
            baseline_cost_field="bounded_baseline_cost",
        )

    baseline_event = Event(
        type=EventType.MODEL_COMPLETED,
        session_id="prompt-cache-bounded-baseline",
        payload=baseline_payload,
    )
    candidate_cost = estimate_session_cost(
        session_id=candidate_event.session_id,
        events=[candidate_event],
        pricing=price_book,
    )
    baseline_cost = estimate_session_cost(
        session_id=baseline_event.session_id,
        events=[baseline_event],
        pricing=price_book,
    )
    return paired_cost_evidence(
        candidate=(candidate_cost,),
        baseline=(baseline_cost,),
        price_book=price_book,
        baseline_cost_field="bounded_baseline_cost",
    )


def _retry_inclusive_cost_evidence(
    *,
    events: list[Event],
    provider_attempts: int,
    input_tokens: int,
    output_tokens: int,
    price_book: PriceBook | None,
    session_id: str,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_steps": sum(event.type == EventType.MODEL_COMPLETED for event in events),
        "provider_attempts": provider_attempts,
    }
    if price_book is None:
        return {
            **evidence,
            "cost_status": "unpriced",
            "cost": None,
            "unpriced_or_missing_usage_attempts": provider_attempts,
            "reason": "no caller-supplied price book",
        }

    summary = estimate_session_cost(session_id=session_id, events=events, pricing=price_book)
    unpriced_or_missing = summary.unpriced_model_steps + max(
        0, provider_attempts - summary.model_steps
    )
    if unpriced_or_missing:
        return {
            **evidence,
            "cost_status": "unpriced",
            "cost": None,
            "unpriced_or_missing_usage_attempts": unpriced_or_missing,
            "reason": "not every provider attempt has priced completion usage",
        }
    return {
        **evidence,
        "cost_status": "priced",
        "cost": str(summary.total_cost),
        "unpriced_or_missing_usage_attempts": 0,
    }


def _text(message: Message) -> str:
    return "\n".join(part.text for part in message.content if type(part) is TextPart)


def _model_text(events: list[Event]) -> str:
    return "".join(
        str(event.payload.get("delta", ""))
        for event in events
        if event.type == EventType.MODEL_TEXT_DELTA
    )


async def run_scenario(
    root: Path,
    *,
    provider: ModelProvider,
    baseline_provider: ModelProvider,
    model: str,
    mode: str,
    stable_context_lines: int,
    provider_options: dict[str, Any] | None = None,
    system_prompt_suffix: str = "",
    thinking: ThinkingConfig | None = None,
    price_book: PriceBook | None = None,
) -> ScenarioResult:
    run_id = uuid4().hex[:12]
    session_id = f"prompt-cache-compaction-{run_id}"
    thinking = thinking or ThinkingConfig(effort="low")
    recorder = RecordingProvider(provider)
    baseline_recorder = RecordingProvider(baseline_provider)
    tool = LoadStableContextTool(stable_context(lines=stable_context_lines))
    cache_fixture = tool.stable_context
    app = CayuApp(enable_logging=False)
    app.register_provider(recorder, default=True)
    paired_compactor = PairedPromptCacheCompactor(provider=recorder)
    baseline_options = deepcopy(provider_options) if provider_options is not None else {}
    baseline_options["thinking"] = thinking.model_dump()
    baseline_compactor = _RecordingModelCompactor(
        provider=baseline_recorder,
        model=model,
        system_prompt=_BOUNDED_BASELINE_SYSTEM_PROMPT,
        options=baseline_options,
    )
    app.register_agent(
        AgentSpec(
            name="cache-assistant",
            model=model,
            system_prompt=(
                "You verify prompt-cache compaction. On the first user turn, call "
                "load_stable_context exactly once with topic='cache contracts'. After the tool "
                "returns, answer with a concise acknowledgement. On later turns, answer directly."
                + system_prompt_suffix
            ),
            provider_options=provider_options or {},
        ),
        tools=[tool],
        context_policy=CheckpointCompactionContextPolicy(
            compactor=paired_compactor,
            max_user_turns=1,
            compact_after_messages=5,
        ),
    )
    first_events = await collect_events(
        app.run(
            RunRequest(
                agent_name="cache-assistant",
                session_id=session_id,
                messages=[
                    Message.text(
                        "user",
                        "Load the stable context, then acknowledge it without reproducing it.",
                    )
                ],
                thinking=thinking,
                max_steps=3,
            )
        )
    )
    second_events = await collect_events(
        app.resume(
            ResumeRequest(
                session_id=session_id,
                messages=[
                    Message.text(
                        "user",
                        "State the first retained constraint briefly. Do not call any tools.",
                    )
                ],
                thinking=thinking,
                max_steps=1,
            )
        )
    )
    third_events = await collect_events(
        app.resume(
            ResumeRequest(
                session_id=session_id,
                messages=[
                    Message.text(
                        "user",
                        "State the next retained constraint briefly. Do not call any tools.",
                    )
                ],
                thinking=thinking,
                max_steps=1,
            )
        )
    )
    fourth_events = await collect_events(
        app.resume(
            ResumeRequest(
                session_id=session_id,
                messages=[
                    Message.text(
                        "user",
                        "Return the mandatory retention token from the original tool context. "
                        "Do not call any tools or add other text.",
                    )
                ],
                thinking=thinking,
                max_steps=1,
            )
        )
    )
    if paired_compactor.paired_request is None:
        raise RuntimeError("The cache-aware run did not produce a paired compaction source.")
    baseline_result = await baseline_compactor.compact(paired_compactor.paired_request)
    baseline_received_same_request = baseline_compactor.request is paired_compactor.paired_request
    events = first_events + second_events + third_events + fourth_events
    compaction_events = _compaction_model_events(events)
    requests = recorder.requests

    first_prefix_preserved = False
    second_compaction_bounded = False
    if len(requests) == 7:
        last_warm_request = requests[2]
        first_compaction = requests[3]
        first_prefix_preserved = (
            first_compaction.messages[: len(last_warm_request.messages)]
            == last_warm_request.messages
            and any(
                any(
                    type(part) is ToolResultPart and "CACHE_SENTINEL_0000" in part.content
                    for part in message.content
                )
                for message in last_warm_request.messages
            )
            and first_compaction.tools == last_warm_request.tools
            and first_compaction.options.get("thinking")
            == last_warm_request.options.get("thinking")
        )
        incremental_compaction = requests[5]
        incremental_prompt = _text(incremental_compaction.messages[-1])
        second_compaction_bounded = (
            incremental_compaction.tools == []
            and len(incremental_compaction.messages) == 2
            and "Existing summary:" in incremental_prompt
            and "State the next retained constraint briefly. Do not call any tools."
            in incremental_prompt
            and len(incremental_prompt) < len(cache_fixture)
        )

    compactor_names = [event.payload.get("compactor") for event in compaction_events]
    sessions = await session_evidence(app, {session_id: "cache-compaction"})
    session = sessions[0]
    cache_read_tokens = _cache_read_tokens(compaction_events[0]) if compaction_events else 0
    compaction_usage = [_usage_snapshot(event) for event in compaction_events]
    baseline_payload = (
        baseline_result.model_completed_payloads[0]
        if baseline_result is not None and baseline_result.model_completed_payloads
        else None
    )
    baseline_usage = (
        _usage_snapshot_payload(baseline_payload)
        if baseline_payload is not None
        else _usage_snapshot_payload({})
    )
    baseline_requests = baseline_recorder.requests
    baseline_event = (
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="prompt-cache-bounded-baseline",
            payload=baseline_payload,
        )
        if baseline_payload is not None
        else None
    )
    candidate_session_evidence = _retry_inclusive_cost_evidence(
        events=events,
        provider_attempts=len(requests),
        input_tokens=session.usage["input_tokens"],
        output_tokens=session.usage["output_tokens"],
        price_book=price_book,
        session_id=session_id,
    )
    baseline_input_tokens = baseline_usage.get("input_tokens")
    baseline_output_tokens = baseline_usage.get("output_tokens")
    harness_events = [*events, *([baseline_event] if baseline_event is not None else [])]
    harness_evidence = _retry_inclusive_cost_evidence(
        events=harness_events,
        provider_attempts=len(requests) + len(baseline_requests),
        input_tokens=session.usage["input_tokens"]
        + (baseline_input_tokens if type(baseline_input_tokens) is int else 0),
        output_tokens=session.usage["output_tokens"]
        + (baseline_output_tokens if type(baseline_output_tokens) is int else 0),
        price_book=price_book,
        session_id=f"{session_id}-benchmark-harness",
    )
    bounded_baseline_shape = (
        len(baseline_requests) == 1
        and baseline_requests[0].tools == []
        and len(baseline_requests[0].messages) == 2
        and RESOLVED_FILE_ATTACHMENTS_OPTION not in baseline_requests[0].options
    )
    paired_model_configuration_matches = (
        len(requests) == 7
        and len(baseline_requests) == 1
        and baseline_requests[0].options.get("thinking") == requests[3].options.get("thinking")
    )
    paired_cost = _paired_cost_evidence(
        candidate_event=compaction_events[0] if compaction_events else None,
        baseline_payload=baseline_payload,
        price_book=price_book,
    )
    provenance_gated = (
        paired_cost.get("status") == "unpriced"
        and paired_cost.get("candidate_cost") is None
        and paired_cost.get("bounded_baseline_cost") is None
    ) or (
        paired_cost.get("status") == "priced"
        and isinstance(paired_cost.get("pricing_provenance"), dict)
    )
    assertions = {
        "tool_session_exercised": tool.calls == 1 and session.tool_calls == 1,
        "first_compaction_extended_exact_request_prefix": first_prefix_preserved,
        "first_compaction_reused_provider_cache": cache_read_tokens > 0,
        "bounded_baseline_used_same_compactable_source": (
            baseline_received_same_request and len(paired_compactor.paired_request.messages) > 0
        ),
        "bounded_baseline_stripped_exact_request_shape": bounded_baseline_shape,
        "paired_cache_counters_reported_separately": (
            bool(compaction_usage)
            and compaction_usage[0]["usage_available"] is True
            and compaction_usage[0]["cache_usage_available"] is True
            and type(compaction_usage[0]["cache_read_tokens"]) is int
            and compaction_usage[0]["cache_read_tokens"] > 0
            and baseline_usage["usage_available"] is True
            and baseline_usage["cache_usage_available"] is True
            and baseline_usage["cache_read_tokens"] == 0
        ),
        "paired_model_configuration_matches": paired_model_configuration_matches,
        "paired_cost_claim_is_provenance_gated": provenance_gated,
        "paired_summaries_pass_quality_floor": (
            RETENTION_TOKEN in baseline_result.summary
            and RETENTION_TOKEN in _model_text(fourth_events)
        ),
        "second_compaction_used_bounded_delta": second_compaction_bounded,
        "compaction_modes_recorded": compactor_names
        == ["PairedPromptCacheCompactor", "ModelCompactor"],
        "compaction_spend_persisted": len(compaction_events) == 2
        and session.model_steps == len(requests),
        "comparison_spend_reported_separately": (
            len(baseline_requests) == 1 and session.model_steps == len(requests)
        ),
        "session_completed": session.status == "completed",
    }
    result = ScenarioResult(
        scenario="prompt-cache-compaction",
        mode=mode,
        status="verified" if all(assertions.values()) else "failed",
        assertions=assertions,
        sessions=sessions,
        provider_name=recorder.name,
        model=model,
        run_id=run_id,
        metrics={
            "model_requests": len(requests),
            "compaction_model_steps": len(compaction_events),
            "first_compaction_cache_read_tokens": cache_read_tokens,
            "first_compaction_attempt": (
                compaction_usage[0] if compaction_usage else _usage_snapshot_payload({})
            ),
            "bounded_baseline_first_compaction_attempt": baseline_usage,
            "paired_first_compaction_cost": paired_cost,
            "all_compaction_completions": compaction_usage,
            "retry_inclusive_session_total": session.usage,
            "retry_inclusive_candidate_session": candidate_session_evidence,
            "benchmark_harness": harness_evidence,
            "provider_attempts": len(requests),
            "provider_attempts_beyond_completed_steps": len(requests) - session.model_steps,
            "comparison_provider_attempts": len(baseline_requests),
            "total_provider_attempts_including_comparison": len(requests) + len(baseline_requests),
            "paired_source_message_count": len(paired_compactor.paired_request.messages),
            "stable_cache_fixture_chars": len(cache_fixture),
        },
        outputs={"compactors": compactor_names},
    )
    result.write(root)
    result.require_verified()
    return result
