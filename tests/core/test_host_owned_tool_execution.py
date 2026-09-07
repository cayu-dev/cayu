"""Characterize host execution through the existing, unmodified native tool API.

This is a live-process transport, deliberately not a durable delegation API. A
host performs the action; Cayu owns policy, model iteration and publication.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from cayu import SQLiteSessionStore
from cayu.core import AgentSpec, Event, EventType, Message, ToolResultPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    InMemorySessionStore,
    RunRequest,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)


@dataclass
class _HostCall:
    session_id: str
    idempotency_key: str
    arguments: dict[str, Any]
    result: asyncio.Future[ToolResult]


class _HostTool(Tool):
    spec = ToolSpec(
        name="host_action",
        description="Have the application host execute an action.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        parallel_safe=False,
    )

    def __init__(self, calls: asyncio.Queue[_HostCall]) -> None:
        super().__init__()
        self.calls = calls

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        # Native tools own argument validation; schema exposure alone does not
        # validate an ordinary model call at dispatch time.
        if set(args) != {"value"} or type(args["value"]) is not int:
            return ToolResult(content="Expected an integer value.", is_error=True)
        assert ctx.idempotency_key is not None
        result = asyncio.get_running_loop().create_future()
        await self.calls.put(_HostCall(ctx.session_id, ctx.idempotency_key, args, result))
        return await result


class _Provider(ModelProvider):
    name = "host-test"

    def __init__(self, arguments: dict[str, Any] | None = None) -> None:
        self.arguments = {"value": 7} if arguments is None else arguments
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelStreamEvent.tool_call(
                id="host-call", name="host_action", arguments=self.arguments
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
        else:
            yield ModelStreamEvent.text_delta("Host result received.")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _Deny(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="Host disabled.")


async def _drain(app: CayuApp) -> list[Event]:
    return [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="host-session",
                messages=[Message.text("user", "Run the host action.")],
            )
        )
    ]


@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "sqlite"])
def test_live_host_execution_retains_cayu_model_loop_and_checkpoint(tmp_path, persistent):
    async def scenario():
        store = (
            SQLiteSessionStore(tmp_path / "host.sqlite3") if persistent else InMemorySessionStore()
        )
        calls: asyncio.Queue[_HostCall] = asyncio.Queue()
        provider = _Provider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="scripted"), tools=[_HostTool(calls)])
        task = asyncio.create_task(_drain(app))
        try:
            async with asyncio.timeout(10):
                call = await calls.get()
                assert call.session_id == "host-session"
                assert call.arguments == {"value": 7}
                assert call.idempotency_key
                assert len(provider.requests) == 1
                assert not task.done()
                checkpoint = await store.load_checkpoint("host-session")
                assert checkpoint is not None
                assert checkpoint.get("pending_tool_round") is not None

                # Only this host owns execution. The runtime cannot invent its result.
                call.result.set_result(ToolResult(content="Host executed value 7."))
                events = await task

            assert len(provider.requests) == 2
            results = [
                part
                for message in provider.requests[1].messages
                for part in message.content
                if isinstance(part, ToolResultPart)
            ]
            assert len(results) == 1
            assert results[0].content == "Host executed value 7."
            assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in events)
            assert any(event.type == EventType.SESSION_COMPLETED for event in events)
            assert calls.empty()
            checkpoint = await store.load_checkpoint("host-session")
            assert checkpoint is not None
            assert checkpoint.get("pending_tool_round") is None
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if persistent:
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("rejection", ["policy", "schema"])
def test_rejected_host_call_never_reaches_execution_owner(rejection):
    async def scenario():
        calls: asyncio.Queue[_HostCall] = asyncio.Queue()
        provider = _Provider({"value": "invalid"} if rejection == "schema" else None)
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="scripted"),
            tools=[_HostTool(calls)],
            tool_policy=_Deny() if rejection == "policy" else None,
        )
        async with asyncio.timeout(10):
            await _drain(app)
        assert calls.empty()
        assert len(provider.requests) == 2
        results = [
            part
            for message in provider.requests[1].messages
            for part in message.content
            if isinstance(part, ToolResultPart)
        ]
        assert len(results) == 1
        assert results[0].is_error

    asyncio.run(scenario())
