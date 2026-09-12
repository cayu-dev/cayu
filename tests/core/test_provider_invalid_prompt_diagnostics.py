"""Error identity is diagnostic evidence, not permission to retry a request."""

import asyncio
import json

import pytest
from tests.core.test_openai_subscription_provider import StaticSubscriptionAuth

from cayu import Message
from cayu.providers import ModelRequest, OpenAIProvider, OpenAISubscriptionProvider
from cayu.providers.openai import _openai_stream_error_exception


@pytest.mark.parametrize("subscription", [False, True])
@pytest.mark.parametrize("event_type", ["error", "response.failed"])
def test_invalid_prompt_survives_provider_projection(subscription, event_type):
    error = {
        "type": "invalid_request_error",
        "code": "invalid_prompt",
        "message": "private prompt text",
        "param": "private parameter",
        "request_id": "private request id",
    }
    event = (
        {"type": "error", "error": error}
        if event_type == "error"
        else {"type": event_type, "response": {"error": error}}
    )

    class Transport:
        async def stream_response_events(self, **kwargs):
            yield event

    provider = (
        OpenAISubscriptionProvider(auth=StaticSubscriptionAuth(), transport=Transport())
        if subscription
        else OpenAIProvider(api_key="test-key", transport=Transport())
    )

    async def run():
        return [
            item
            async for item in provider.stream(
                ModelRequest(model="test-model", messages=[Message.text("user", "test")])
            )
        ]

    events = asyncio.run(run())
    failure = events[-1].payload
    assert failure["provider_error_code"] == "invalid_prompt"
    assert failure["provider_error_type"] == "invalid_request_error"
    assert failure["retryable"] is False
    assert "private" not in json.dumps(failure)


def test_nested_stream_status_conflicts_do_not_authorize_retry_or_overflow():
    failure = _openai_stream_error_exception(
        {
            "type": "error",
            "status_code": 503,
            "error": {
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "status_code": 400,
            },
        }
    )
    assert failure.retryable is False
    assert failure.status_code is None
    assert failure.classification_reason == "explicit_status_conflict"
