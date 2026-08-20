from __future__ import annotations

import asyncio
import os

from cayu import (
    AgentSpec,
    CayuApp,
    CitationPart,
    HostedToolCallPart,
    Message,
    OpenAIProvider,
    OpenAIWebSearch,
    RunRequest,
)


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this live OpenAI example.")
        return

    session_id = "demo_openai_hosted_web_search"
    app = CayuApp()
    app.register_provider(OpenAIProvider(), default=True)
    app.register_agent(
        AgentSpec(
            name="researcher",
            model=os.environ.get("CAYU_OPENAI_MODEL", "gpt-5.6-luna"),
        ),
        hosted_tools=[
            OpenAIWebSearch(
                search_context_size="low",
                allowed_domains=("python.org",),
                include_sources=True,
            )
        ],
    )

    async for event in app.run(
        RunRequest(
            agent_name="researcher",
            session_id=session_id,
            messages=[
                Message.text(
                    "user",
                    "Search python.org and summarize the latest Python release.",
                )
            ],
            max_steps=1,
        )
    ):
        print(event.type)

    for message in await app.session_store.load_transcript(session_id):
        for part in message.content:
            if type(part) is CitationPart:
                print("citation", part.title, part.url)
            elif type(part) is HostedToolCallPart and part.action is not None:
                for source in part.action.sources:
                    print("source", source.title, source.url)

    usage = await app.get_session_usage(session_id)
    print("input_tokens", usage.usage.input_tokens)
    print("output_tokens", usage.usage.output_tokens)
    print("web_search_calls", usage.usage.hosted_tools.web_search_calls)


if __name__ == "__main__":
    asyncio.run(main())
