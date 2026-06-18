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
            "reviewer": SubagentSpec(
                agent_name="reviewer",
                description="Review short implementation plans and return concrete risks.",
                result_max_chars=4_000,
            )
        },
    )
    app.register_agent(
        AgentSpec(
            name="builder",
            model=model,
            system_prompt=(
                "You are testing Cayu subagents. Call the subagent tool exactly once "
                "with agent='reviewer' to review the user's plan. After the tool result "
                "returns, give a concise final answer that includes one point from the "
                "reviewer. Do not call any other tools."
            ),
        ),
        tools=[subagents],
    )
    app.register_agent(
        AgentSpec(
            name="reviewer",
            model=model,
            system_prompt=(
                "You are a focused reviewer. Return two concise risks and one practical "
                "suggestion. Do not mention that you are a subagent."
            ),
        )
    )

    session_id = f"demo_{provider_name}_subagent"
    print("provider", provider_name)
    print("model", model)
    print("session_id", session_id)

    async for event in app.run(
        RunRequest(
            agent_name="builder",
            session_id=session_id,
            causal_budget_id=f"job_{provider_name}_subagent",
            messages=[
                Message.text(
                    "user",
                    (
                        "Plan: add a file upload endpoint that stores PDFs, lets the "
                        "agent inspect them, and summarizes the document for users. "
                        "Ask the reviewer to review this plan, then answer with the "
                        "reviewer's most important point."
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

    children = await app.session_store.list_sessions(SessionQuery(parent_session_id=session_id))
    print("child_sessions", [child.id for child in children])
    for child in children:
        print(
            "child",
            child.id,
            "agent",
            child.agent_name,
            "status",
            child.status,
            "parent",
            child.parent_session_id,
            "causal_budget",
            child.causal_budget_id,
        )
        events = await app.session_store.load_events(child.id)
        print("child_event_count", len(events))


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
