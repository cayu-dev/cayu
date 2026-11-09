"""Projection cancellation stays authoritative during later audit publication."""

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, Tool, ToolSpec
from cayu.environments import Environment, EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.proxies.passthrough import PassthroughProxy
from cayu.sessions import InMemorySessionStore
from cayu.tools.base import ToolEffect
from cayu.tools.result_projection import (
    ArtifactExternalizingToolResultPolicy,
    ToolResultProjectionPolicy,
)
from cayu.vaults import StaticVault


@pytest.mark.parametrize("failure_boundary", [None, "terminal", "proxy"])
def test_projection_cancellation_survives_publication_failure(monkeypatch, failure_boundary):
    async def scenario():
        ready = asyncio.Event()
        release = asyncio.Event()
        inner = ArtifactExternalizingToolResultPolicy()

        class PausedProjection(ToolResultProjectionPolicy):
            @property
            def identity(self):
                return inner.identity

            async def project(self, request):
                projection = await inner.project(request)
                ready.set()
                await release.wait()
                return projection

        class Fetch(Tool):
            spec = ToolSpec(name="fetch", effect=ToolEffect.IDEMPOTENT)

            async def run(self, ctx, args):
                await ctx.proxy.authorize_request(destination="https://example.test/fetch")
                raise ConnectionError("fetch failed")

        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            tool_result_projection_policy=PausedProjection(),
        )
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(id="fetch-1", name="fetch", arguments={}),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})],
                ]
            ),
            default=True,
        )
        vault = StaticVault({})
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                proxy=PassthroughProxy(vault),
                vault=vault,
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="worker", model="scripted-model"), tools=[Fetch()])
        original_emit = app._event_writer.emit
        publication_attempts = []

        async def emit(event, *args, **kwargs):
            boundary = (
                "terminal"
                if event.type is EventType.TOOL_CALL_FAILED
                else "proxy"
                if event.type is EventType.CREDENTIAL_PROXY_CHECKED
                else None
            )
            if boundary is not None:
                publication_attempts.append(boundary)
                if failure_boundary == boundary:
                    raise OSError(f"{boundary} store unavailable")
            return await original_emit(event, *args, **kwargs)

        monkeypatch.setattr(app._event_writer, "emit", emit)

        async def consume():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="projection-cancel",
                        agent_name="worker",
                        messages=[Message.text("user", "fetch")],
                    )
                )
            ]

        consumer = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(ready.wait(), 10)
            consumer.cancel("caller stopped during projection")
            await asyncio.sleep(0)
            release.set()
            try:
                events = await asyncio.wait_for(consumer, 10)
            except asyncio.CancelledError as cancellation:
                assert str(cancellation) == "caller stopped during projection"
            else:
                pytest.fail(
                    f"Caller cancellation was replaced by {events[-1].type.value}: "
                    f"{events[-1].payload}"
                )
            assert consumer.cancelling() == 1
            assert "terminal" in publication_attempts
            if failure_boundary != "terminal":
                assert "proxy" in publication_attempts
        finally:
            release.set()
            if not consumer.done():
                consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)

    asyncio.run(scenario())
