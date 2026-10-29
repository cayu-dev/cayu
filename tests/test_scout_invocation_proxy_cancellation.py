"""A completed external effect must survive cancellation of later telemetry."""

import asyncio
from contextlib import aclosing

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, Tool, ToolResult, ToolSpec
from cayu.environments import Environment, EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.proxies.passthrough import PassthroughProxy
from cayu.sessions import InMemorySessionStore
from cayu.sessions.requests import ResumeRequest
from cayu.tools.base import ToolEffect
from cayu.vaults import SecretRef, StaticVault


async def _scenario(monkeypatch):
    effects = []

    class Charge(Tool):
        spec = ToolSpec(name="charge", effect=ToolEffect.EXTERNAL)

        async def run(self, ctx, args):
            await ctx.proxy.authorize_request(
                destination="https://api.example.test/charge",
                credential=SecretRef(name="service_key"),
                action="charge",
            )
            effects.append("receipt-1")
            return ToolResult(content="charged", structured={"receipt_id": "receipt-1"})

    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="charge-1", name="charge", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})],
            ]
        ),
        default=True,
    )
    vault = StaticVault({"service_key": "private-key"})
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), vault=vault, proxy=PassthroughProxy(vault)),
        default=True,
    )
    app.register_agent(AgentSpec(name="worker", model="scripted-model"), tools=[Charge()])
    telemetry_started = asyncio.Event()
    release = asyncio.Event()
    append_events = store.append_events

    async def blocked_append(session_id, events):
        if any(event.type is EventType.CREDENTIAL_PROXY_CHECKED for event in events):
            telemetry_started.set()
            await release.wait()
        return await append_events(session_id, events)

    monkeypatch.setattr(store, "append_events", blocked_append)

    async def drain(stream):
        async with aclosing(stream):
            return [event async for event in stream]

    task = asyncio.create_task(
        drain(
            app.run(
                RunRequest(
                    session_id="proxy-cancel",
                    agent_name="worker",
                    messages=[Message.text("user", "charge")],
                )
            )
        )
    )
    try:
        await asyncio.wait_for(telemetry_started.wait(), 20)
        assert effects == ["receipt-1"]
        task.cancel("stop after the charge completed")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 20)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        monkeypatch.setattr(store, "append_events", append_events)

    await drain(
        app.resume(
            ResumeRequest(session_id="proxy-cancel", messages=[Message.text("user", "continue")])
        )
    )
    assert effects == ["receipt-1"]
    durable = await store.load_events("proxy-cancel")
    completed = [event for event in durable if event.type is EventType.TOOL_CALL_COMPLETED]
    assert len(completed) == 1, [
        (event.type.value, event.payload)
        for event in durable
        if event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
    ]
    assert completed[0].payload["result"]["structured"] == {"receipt_id": "receipt-1"}


def test_completed_effect_is_not_lost_when_proxy_telemetry_is_cancelled(monkeypatch):
    asyncio.run(_scenario(monkeypatch))
