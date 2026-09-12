"""Shared credential-free native API/subscription ordering fixture."""

import httpx

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    OpenAIProvider,
    OpenAIWebSearch,
    RetryPolicy,
    RunRequest,
    SQLiteSessionStore,
    Tool,
    ToolResult,
    ToolSpec,
)
from cayu.providers import HttpxOpenAITransport
from cayu.providers.openai_subscription import OpenAISubscriptionProvider
from tests.core.test_openai_subscription_provider import StaticSubscriptionAuth
from tests.providers._responses_sse import ChunkedSSE


async def run_ordering_attempts(tmp_path, adapter, attempts):
    calls = []
    requests = []

    class Echo(Tool):
        spec = ToolSpec(
            name="echo",
            description="Synthetic invocation recorder",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        )

        async def run(self, ctx, args):
            calls.append(args)
            return ToolResult(content="fixture complete")

    async def handler(request):
        assert len(requests) < len(attempts), "unexpected native provider dispatch"
        raw = attempts[len(requests)]
        requests.append(None)
        return httpx.Response(
            200, stream=ChunkedSSE(raw), headers={"content-type": "text/event-stream"}
        )

    transport = HttpxOpenAITransport()
    store = SQLiteSessionStore(tmp_path / f"{adapter}.sqlite")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport._client._client = client
        provider = (
            OpenAIProvider(api_key="synthetic-api-key", transport=transport)
            if adapter == "api"
            else OpenAISubscriptionProvider(auth=StaticSubscriptionAuth(), transport=transport)
        )
        app = CayuApp(enable_logging=False, session_store=store)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="worker", model="gpt-5.6"),
            tools=[Echo()],
            hosted_tools=[OpenAIWebSearch()],
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="worker",
                        session_id="native-ordering",
                        messages=[Message.text("user", "fixture request")],
                        retry_policy=RetryPolicy(
                            max_attempts=5, max_unknown_attempts=2, initial_delay_s=0
                        ),
                    )
                )
            ]
            durable = await store.load_events("native-ordering")
            assert len(requests) == len(attempts)
            return events, durable, calls
        finally:
            await provider.aclose()
            await store.close()
