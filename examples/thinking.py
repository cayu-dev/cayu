from __future__ import annotations

import asyncio
import os

from cayu import (
    AgentSpec,
    AnthropicProvider,
    CayuApp,
    Message,
    RunRequest,
    ThinkingConfig,
)
from cayu.core.events import EventType


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run this live thinking example.")
        return

    model = os.environ.get("CAYU_ANTHROPIC_MODEL", "claude-opus-4-8")

    app = CayuApp()
    app.register_provider(AnthropicProvider(), default=True)
    # The agent's default thinking behavior. `effort` maps to adaptive thinking on
    # current Claude models; use `max_tokens=...` instead for the older budgeted path.
    app.register_agent(
        AgentSpec(
            name="thinker",
            model=model,
            system_prompt="Think carefully, then answer concisely.",
            thinking=ThinkingConfig(effort="high"),
        )
    )

    request = RunRequest(
        agent_name="thinker",
        session_id="demo_thinking",
        messages=[
            Message.text(
                "user",
                "A farmer has 17 sheep. All but 9 run away. How many are left? Explain briefly.",
            )
        ],
        # Per-run override (wins over the agent default) — uncomment to try the
        # legacy budgeted path on an older model:
        # thinking=ThinkingConfig(max_tokens=4096),
    )

    async for event in app.run(request):
        if event.type == EventType.MODEL_THINKING_DELTA:
            print("[thinking]", event.payload.get("delta"))
        elif event.type == EventType.MODEL_TEXT_DELTA:
            print("[answer]", event.payload.get("delta"))
        elif event.type in {EventType.SESSION_COMPLETED, EventType.SESSION_FAILED}:
            print(event.type)

    transcript = await app.session_store.load_transcript("demo_thinking")
    thinking_parts = [
        part
        for message in transcript
        if message.role.value == "assistant"
        for part in message.content
        if part.type == "thinking"
    ]
    print("persisted thinking parts:", len(thinking_parts))


if __name__ == "__main__":
    asyncio.run(main())
