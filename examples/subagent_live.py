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
    mode = _mode()

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
                mode=mode,
                result_max_chars=4_000,
            )
        },
    )
    subagent_result = SubagentResultTool(app.session_store)
    if mode == SubagentExecutionMode.BACKGROUND:
        builder_prompt = (
            "You are testing Cayu background subagents. Call the subagent tool exactly "
            "once with agent='reviewer' to review the user's plan. The subagent tool "
            "returns a child_session_id immediately. Then call subagent_result with "
            "that child_session_id and wait=true. After subagent_result returns, give "
            "a concise final answer that includes one point from the reviewer."
        )
        tools = [subagents, subagent_result]
    else:
        builder_prompt = (
            "You are testing Cayu foreground subagents. Call the subagent tool exactly "
            "once with agent='reviewer' to review the user's plan. After the tool "
            "result returns, give a concise final answer that includes one point from "
            "the reviewer. Do not call any other tools."
        )
        tools = [subagents]
    app.register_agent(
        AgentSpec(
            name="builder",
            model=model,
            system_prompt=builder_prompt,
        ),
        tools=tools,
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
    print("subagent_mode", mode)
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

    children = (
        await app.session_store.list_sessions(SessionQuery(parent_session_id=session_id))
    ).sessions
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

    if mode == SubagentExecutionMode.BACKGROUND:
        for _ in range(100):
            children = (
                await app.session_store.list_sessions(SessionQuery(parent_session_id=session_id))
            ).sessions
            if children and all(child.status != "running" for child in children):
                break
            await asyncio.sleep(0.1)
        children = (
            await app.session_store.list_sessions(SessionQuery(parent_session_id=session_id))
        ).sessions
        print(
            "background_child_statuses",
            [(child.id, str(child.status)) for child in children],
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


def _mode() -> SubagentExecutionMode:
    mode = os.environ.get("CAYU_SUBAGENT_MODE", "foreground").strip().lower()
    if mode not in {"foreground", "background"}:
        raise RuntimeError("CAYU_SUBAGENT_MODE must be foreground or background.")
    return SubagentExecutionMode(mode)


if __name__ == "__main__":
    asyncio.run(main())
