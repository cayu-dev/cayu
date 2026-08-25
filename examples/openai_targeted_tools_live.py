"""Spend-bounded live OpenAI ``additional_tools`` and cache-prefix contract.

Run explicitly with:

    CAYU_OPENAI_TARGETED_LIVE=1 \
      PYTHONPATH=src python examples/openai_targeted_tools_live.py

The exact model id can be overridden with ``CAYU_OPENAI_TARGETED_MODEL``.
"""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    ForkSessionRequest,
    Message,
    OpenAIProvider,
    ResumeRequest,
    RunRequest,
    StaticToolExposurePolicy,
    TargetedToolGrant,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)


class RememberKnowledgeTool(Tool):
    spec = ToolSpec(
        name="remember_knowledge",
        description="Save one reviewed durable fact.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"fact": {"type": "string", "minLength": 1}},
            "required": ["fact"],
        },
        effect=ToolEffect.NONE,
    )

    def __init__(self) -> None:
        super().__init__()
        self.facts: list[str] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        fact = str(args["fact"])
        self.facts.append(fact)
        return ToolResult(content="saved")


def _failure_evidence(events) -> list[tuple[str, dict]]:
    return [
        (event.type.value, event.payload)
        for event in events
        if event.type in {EventType.MODEL_ERROR, EventType.SESSION_FAILED}
    ]


def _first_completion_usage(events) -> dict | None:
    for event in events:
        if event.type is EventType.MODEL_COMPLETED:
            usage = event.payload.get("usage_metrics")
            if isinstance(usage, dict):
                return usage
    return None


def _long_stable_system_prompt() -> str:
    cache_material = "\n".join(
        f"Stable cache-boundary sentence {index:04d}: preserve this exact prefix."
        for index in range(220)
    )
    return (
        "Follow the latest user request. Keep ordinary answers to one short sentence. "
        "When the user explicitly asks to save the cache canary and the "
        "remember_knowledge function is available, call it exactly once with fact "
        "equal to 'openai-targeted-cache-canary', then answer 'saved'.\n\n" + cache_material
    )


async def main() -> None:
    if os.environ.get("CAYU_OPENAI_TARGETED_LIVE") != "1":
        print("Set CAYU_OPENAI_TARGETED_LIVE=1 to authorize this bounded live check.")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to run this live OpenAI contract.")

    model = os.environ.get("CAYU_OPENAI_TARGETED_MODEL", "gpt-5.6")
    cache_key = f"cayu-openai-targeted-live:{uuid4().hex}"
    provider = OpenAIProvider(additional_tools_models=(model,))
    remember = RememberKnowledgeTool()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="reviewer",
            model=model,
            system_prompt=_long_stable_system_prompt(),
            provider_options={
                "openai": {
                    "max_output_tokens": 128,
                    "prompt_cache_key": cache_key,
                    "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
                }
            },
        ),
        tools=(remember,),
        tool_exposure_policy=StaticToolExposurePolicy(
            profile_id="targeted-only",
            tools=(),
        ),
        targeted_tool_mode="openai_additional_tools",
    )

    parent_id = "openai_targeted_live_parent"
    warm_id = "openai_targeted_live_warm"
    child_id = "openai_targeted_live_child"
    parent_events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="reviewer",
                session_id=parent_id,
                messages=[Message.text("user", "Reply exactly: baseline-ready")],
                # Forks retain the source execution profile. Keep this aligned
                # with the child's two-step tool-call-and-result turn.
                max_steps=2,
            )
        )
    ]
    if parent_events[-1].type is not EventType.SESSION_COMPLETED:
        raise RuntimeError(
            f"Parent did not complete: {parent_events[-1].type}; "
            f"evidence={_failure_evidence(parent_events)!r}"
        )

    for fork_id in (warm_id, child_id):
        fork_events = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(source_session_id=parent_id, session_id=fork_id)
            )
        ]
        if [event.type for event in fork_events] != [EventType.SESSION_FORKED]:
            raise RuntimeError("Fork did not produce its canonical durable event.")

    target_message = Message.text(
        "user",
        "Save the cache canary with the available targeted function.",
    )
    warm_events = [
        event
        async for event in app.resume(
            ResumeRequest(
                session_id=warm_id,
                messages=[target_message],
                max_steps=2,
            )
        )
    ]
    if warm_events[-1].type is not EventType.SESSION_COMPLETED:
        raise RuntimeError(
            f"Cache warmer did not complete: {warm_events[-1].type}; "
            f"evidence={_failure_evidence(warm_events)!r}"
        )

    child_events = [
        event
        async for event in app.resume(
            ResumeRequest(
                session_id=child_id,
                messages=[target_message],
                tool_grants=(
                    TargetedToolGrant(
                        request_id="save-cache-canary",
                        tool_id="cayu:remember_knowledge",
                        max_calls=1,
                        lifetime_seconds=120,
                    ),
                ),
                max_steps=2,
            )
        )
    ]
    if child_events[-1].type is not EventType.SESSION_COMPLETED:
        raise RuntimeError(
            f"Child did not complete: {child_events[-1].type}; "
            f"evidence={_failure_evidence(child_events)!r}"
        )
    if remember.facts != ["openai-targeted-cache-canary"]:
        raise RuntimeError(f"Native tool call mismatch: {remember.facts!r}")

    [grant] = await app.session_store.list_targeted_tool_grants(child_id)
    if grant.used_calls != 1 or grant.remaining_calls != 0:
        raise RuntimeError("Native invocation did not consume exactly one grant use.")
    native_footprints = [
        event.payload
        for event in child_events
        if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
        and event.payload.get("targeted_native_item_active") is True
    ]
    if not native_footprints:
        raise RuntimeError("No native targeted request footprint was recorded.")

    child_model_completions = [
        event for event in child_events if event.type is EventType.MODEL_COMPLETED
    ]
    if not child_model_completions:
        raise RuntimeError("The child produced no model completion evidence.")
    first_usage = child_model_completions[0].payload.get("usage_metrics")
    first_cache = first_usage.get("cache") if isinstance(first_usage, dict) else None
    first_cached_tokens = (
        first_cache.get("cached_input_tokens") if isinstance(first_cache, dict) else None
    )
    if not isinstance(first_cached_tokens, int) or first_cached_tokens <= 0:
        raise RuntimeError(
            "OpenAI reported no cached inherited-prefix tokens on the child's first request; "
            f"parent_usage={_first_completion_usage(parent_events)!r}; "
            f"warm_usage={_first_completion_usage(warm_events)!r}; "
            f"child_usage={first_usage!r}"
        )

    usage = await app.get_session_usage(child_id)
    cached_tokens = usage.usage.cache.cached_input_tokens
    await provider.aclose()

    print("model", model)
    print("tool_calls", len(remember.facts))
    print("grant_uses", grant.used_calls)
    print("native_model_steps", len(native_footprints))
    print("first_child_cached_input_tokens", first_cached_tokens)
    print("cached_input_tokens", cached_tokens)


if __name__ == "__main__":
    asyncio.run(main())
