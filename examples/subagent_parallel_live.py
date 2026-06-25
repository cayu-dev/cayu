from __future__ import annotations

import asyncio
import os

from cayu import (
    AgentSpec,
    AnthropicProvider,
    CayuApp,
    Message,
    OpenAIProvider,
    RunRequest,
    SessionQuery,
    SubagentExecutionMode,
    SubagentResultTool,
    SubagentSpec,
    SubagentTool,
)


async def main() -> None:
    provider_name = _provider_name()
    model = _model(provider_name)

    app = CayuApp()
    if provider_name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("Set OPENAI_API_KEY or choose CAYU_PROVIDER=anthropic.")
            return
        app.register_provider(OpenAIProvider(), default=True)
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Set ANTHROPIC_API_KEY or choose CAYU_PROVIDER=openai.")
            return
        app.register_provider(AnthropicProvider(), default=True)

    subagents = SubagentTool(
        app,
        agents={
            "security_reviewer": SubagentSpec(
                agent_name="security_reviewer",
                description="Review security and abuse risks.",
                mode=SubagentExecutionMode.BACKGROUND,
                result_max_chars=4_000,
            ),
            "product_reviewer": SubagentSpec(
                agent_name="product_reviewer",
                description="Review product and user-experience risks.",
                mode=SubagentExecutionMode.BACKGROUND,
                result_max_chars=4_000,
            ),
        },
    )
    subagent_result = SubagentResultTool(app.session_store, default_timeout_s=90)

    app.register_agent(
        AgentSpec(
            name="builder",
            model=model,
            system_prompt=(
                "You are testing Cayu parallel background subagents. First, call "
                "the subagent tool twice in the same assistant step: once with "
                "agent='security_reviewer' and once with agent='product_reviewer'. "
                "Both tasks should review the user's plan from their specialty. "
                "After both subagent tool results return with child_session_id values, "
                "continue with one short sentence saying what you can draft while they "
                "run. Then call subagent_result with all=true and wait=true. After "
                "subagent_result returns, give a concise final answer combining both "
                "reviewers' most important points."
            ),
        ),
        tools=[subagents, subagent_result],
    )
    app.register_agent(
        AgentSpec(
            name="security_reviewer",
            model=model,
            system_prompt=(
                "You are a security reviewer. Return two concrete security risks "
                "and one mitigation. Be concise."
            ),
        )
    )
    app.register_agent(
        AgentSpec(
            name="product_reviewer",
            model=model,
            system_prompt=(
                "You are a product reviewer. Return two concrete user-experience "
                "or workflow risks and one mitigation. Be concise."
            ),
        )
    )

    session_id = f"demo_{provider_name}_parallel_subagents"
    print("provider", provider_name)
    print("model", model)
    print("session_id", session_id)

    async for event in app.run(
        RunRequest(
            agent_name="builder",
            session_id=session_id,
            causal_budget_id=f"job_{provider_name}_parallel_subagents",
            messages=[
                Message.text(
                    "user",
                    (
                        "Plan: build an invoice ingestion agent that accepts PDFs and "
                        "images, stores files durably, extracts invoice fields, asks "
                        "for approval before sending payment reminders, and lets "
                        "operators inspect the trace later."
                    ),
                )
            ],
        )
    ):
        print(
            event.type,
            event.session_id,
            event.tool_name or "-",
            event.payload,
        )

    children = (
        await app.session_store.list_sessions(SessionQuery(parent_session_id=session_id))
    ).sessions
    print("child_sessions", [child.id for child in children])
    for child in children:
        events = await app.session_store.load_events(child.id)
        print(
            "child",
            child.id,
            "agent",
            child.agent_name,
            "status",
            child.status,
            "events",
            len(events),
            "parent",
            child.parent_session_id,
            "causal_budget",
            child.causal_budget_id,
        )


def _provider_name() -> str:
    requested = os.environ.get("CAYU_PROVIDER")
    if requested is not None:
        requested = requested.strip().lower()
        if requested in {"openai", "anthropic"}:
            return requested
        raise RuntimeError("CAYU_PROVIDER must be openai or anthropic.")
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


def _model(provider_name: str) -> str:
    if provider_name == "openai":
        return os.environ.get("CAYU_OPENAI_MODEL", "gpt-5.5")
    return os.environ.get("CAYU_ANTHROPIC_MODEL", "claude-sonnet-4-6")


if __name__ == "__main__":
    asyncio.run(main())
