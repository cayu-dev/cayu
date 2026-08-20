"""Opt-in live acceptance for OpenAI-hosted web search.

The tests retain and assert only Cayu's bounded structural evidence. They never
print response prose, credentials, account identity, or raw provider envelopes.
"""

from __future__ import annotations

import os

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    CitationPart,
    EventType,
    HostedToolCallPart,
    Message,
    OpenAIProvider,
    OpenAISubscriptionProvider,
    OpenAIWebSearch,
    RunRequest,
)
from cayu.providers.openai_subscription import (
    OpenAISubscriptionAuth,
    OpenAISubscriptionAuthError,
)


async def _assert_live_search(provider, *, model: str) -> None:
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="live-search", model=model),
        hosted_tools=[
            OpenAIWebSearch(
                search_context_size="low",
                allowed_domains=("python.org",),
                include_sources=True,
            )
        ],
    )

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="live-search",
                messages=[
                    Message.text(
                        "user",
                        "Search python.org once and answer which Python release is latest.",
                    )
                ],
                max_steps=1,
            )
        )
    ]

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert any(
        event.type == EventType.MODEL_HOSTED_TOOL_CALL
        and event.payload.get("status") == "completed"
        for event in events
    )
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["hosted_tool_usage"]["web_search_calls"] >= 1
    transcript = await app.session_store.load_transcript(events[-1].session_id)
    assert any(
        type(part) is HostedToolCallPart
        and part.status == "completed"
        and part.action is not None
        and part.action.sources
        for message in transcript
        for part in message.content
    )
    assert any(type(part) is CitationPart for message in transcript for part in message.content)


@pytest.mark.anyio
async def test_openai_api_hosted_web_search_live() -> None:
    if os.environ.get("CAYU_OPENAI_HOSTED_WEB_SEARCH_API_LIVE") != "1":
        pytest.skip("set CAYU_OPENAI_HOSTED_WEB_SEARCH_API_LIVE=1 to spend API credits")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured")

    await _assert_live_search(
        OpenAIProvider(),
        model=os.environ.get("CAYU_OPENAI_HOSTED_WEB_SEARCH_MODEL", "gpt-5.6-luna"),
    )


@pytest.mark.anyio
async def test_openai_subscription_hosted_web_search_live() -> None:
    if os.environ.get("CAYU_OPENAI_HOSTED_WEB_SEARCH_SUBSCRIPTION_LIVE") != "1":
        pytest.skip("set CAYU_OPENAI_HOSTED_WEB_SEARCH_SUBSCRIPTION_LIVE=1 to use local login")
    auth = OpenAISubscriptionAuth()
    try:
        await auth.credentials()
    except OpenAISubscriptionAuthError:
        pytest.skip("no usable local OpenAI subscription login")

    await _assert_live_search(
        OpenAISubscriptionProvider(auth=auth),
        model=os.environ.get("CAYU_OPENAI_HOSTED_WEB_SEARCH_MODEL", "gpt-5.6-luna"),
    )
