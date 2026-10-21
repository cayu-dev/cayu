"""Both OpenAI base URL forms reach the same request URL.

The OpenAI SDK's ``OPENAI_BASE_URL`` ends in ``/v1``, while ``OpenAIProvider``
defaults to the bare API root. These tests send one model step through each
provider's own HTTP transport, backed by a fake httpx transport, and check the
URL that would have gone on the wire.
"""

from __future__ import annotations

import httpx
import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.messages import Message
from cayu.providers import ChatCompletionsProvider, OpenAIProvider
from cayu.providers.chat_completions import HttpxChatCompletionsTransport
from cayu.providers.openai import HttpxOpenAITransport
from cayu.providers.retry_policy import RetryPolicy
from cayu.sessions.requests import RunRequest


async def _request_urls(provider: OpenAIProvider | ChatCompletionsProvider) -> list[str]:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(400, json={"error": {"message": "fake"}}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider.transport._client._client = client
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="test-model"))
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
                max_steps=1,
                retry_policy=RetryPolicy(max_attempts=1),
            )
        ):
            pass
    return urls


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("base_url", "expected_base", "expected_url"),
    [
        ("https://host.example", "https://host.example", "https://host.example/v1/responses"),
        ("https://host.example/v1", "https://host.example", "https://host.example/v1/responses"),
        ("https://host.example/v1/", "https://host.example", "https://host.example/v1/responses"),
        (
            "https://gateway.example/openai/v1",
            "https://gateway.example/openai",
            "https://gateway.example/openai/v1/responses",
        ),
    ],
)
async def test_openai_provider_accepts_base_url_with_or_without_v1(
    base_url: str, expected_base: str, expected_url: str
) -> None:
    provider = OpenAIProvider(
        api_key="test-key", base_url=base_url, transport=HttpxOpenAITransport()
    )

    assert provider.base_url == expected_base
    assert await _request_urls(provider) == [expected_url]


def test_openai_provider_treats_official_v1_url_as_the_default_route() -> None:
    provider = OpenAIProvider(api_key="test-key", base_url="https://api.openai.com/v1")

    assert provider.base_url == "https://api.openai.com"
    assert provider.hosted_web_search_supported is True
    OpenAIProvider(api_key="test-key", base_url="https://api.openai.com/v1/", background=True)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        ("https://host.example/v1", "https://host.example/v1/chat/completions"),
        ("https://api.openai.com/v1", "https://api.openai.com/v1/chat/completions"),
        ("https://api.openai.com", "https://api.openai.com/v1/chat/completions"),
        ("https://eu.api.openai.com/", "https://eu.api.openai.com/v1/chat/completions"),
        # Other services keep the base exactly as given: some put the version
        # elsewhere, and some serve /chat/completions at the root.
        ("https://host.example", "https://host.example/chat/completions"),
        (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        ),
    ],
)
async def test_chat_completions_provider_base_url_includes_the_version_path(
    base_url: str, expected_url: str
) -> None:
    provider = ChatCompletionsProvider(
        api_key="test-key", base_url=base_url, transport=HttpxChatCompletionsTransport()
    )

    assert await _request_urls(provider) == [expected_url]
