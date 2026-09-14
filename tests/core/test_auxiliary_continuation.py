from __future__ import annotations

import asyncio

import pytest

from cayu import (
    AgentSpec,
    AuxiliaryInferencePolicy,
    CayuApp,
    EventType,
    ExecutionProfileBehaviorIdentity,
    InferenceLimits,
    Message,
    ModelRequest,
    ModelStreamEvent,
    RunLimits,
    RunRequest,
    ScriptedModelProvider,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyResult,
    ToolResult,
    ToolSpec,
    UserInputResponse,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.user_input import UserInputTool


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("pause", ["approval", "input"])
@pytest.mark.parametrize("maximum,allowed", [(21, False), (22, True)])
def test_resumed_tool_inherits_auxiliary_capability_and_original_limit(
    sqlite_resources, backend, pause, maximum, allowed
):
    async def scenario():
        bounds = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=10)
        entered = []
        refused = []

        class Summarize(Tool):
            spec = ToolSpec(
                name="summarize",
                description="Auxiliary request after a durable pause",
                input_schema={"type": "object"},
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="tests:auxiliary-continuation-tool",
                    behavior_version="1",
                    implementation_version="1",
                ),
                auxiliary_inference=AuxiliaryInferencePolicy(
                    limits=bounds, purposes=("tool.summary",)
                ),
            )

            async def run(self, ctx, args):
                assert ctx.inference is not None
                entered.append(ctx)
                try:
                    response = await ctx.inference.invoke(
                        ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                        purpose="tool.summary",
                        limits=bounds,
                    )
                except RuntimeError as exc:
                    refused.append(exc)
                    return ToolResult(content="refused", is_error=True)
                return ToolResult(content=response.text)

        class Approval(ToolPolicy):
            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:auxiliary-continuation-policy",
                    behavior_version="1",
                    implementation_version="1",
                )

            async def authorize(self, request):
                return ToolPolicyResult(
                    decision=ToolPolicyDecision.REQUIRE_APPROVAL, reason="Review"
                )

        outer = [ModelStreamEvent.tool_call(name="summarize", id="parent", arguments={})]
        if pause == "input":
            outer.insert(
                0,
                ModelStreamEvent.tool_call(
                    name="ask_user", id="ask", arguments={"question": "Continue?"}
                ),
            )
        outer.append(ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}))
        batches = [outer]
        if allowed:
            batches.append(
                [
                    ModelStreamEvent.text_delta("summary"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 3, "output_tokens": 2}}),
                ]
            )
        batches.append(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
            ]
        )
        provider = ScriptedModelProvider(batches)
        tool = Summarize()
        policy = Approval() if pause == "approval" else None
        tools = [tool] if pause == "approval" else [UserInputTool(), tool]
        path = sqlite_resources.path("continuation.sqlite")
        store = sqlite_resources.own(SQLiteSessionStore(path)) if backend == "sqlite" else None

        def build_app(store):
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="model"), tools=tools, tool_policy=policy
            )
            return app

        app = build_app(store)
        try:
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="auxiliary-continuation",
                        agent_name="assistant",
                        messages=[Message.text("user", "go")],
                        limits=RunLimits(max_total_tokens=maximum),
                    )
                )
            ]
            assert not entered and len(provider.requests) == 1
            if store is not None:
                await store.close()
                store = sqlite_resources.own(SQLiteSessionStore(path))
                app = build_app(store)
            if pause == "approval":
                event = next(
                    event
                    for event in paused
                    if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                )
                continuation = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id="auxiliary-continuation",
                        approval_id=event.payload["approval_id"],
                        tool_round_id=event.payload["tool_round_id"],
                        tool_call_id=event.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            else:
                event = next(
                    event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
                )
                continuation = app.resolve_user_input(
                    UserInputResponse(
                        session_id="auxiliary-continuation",
                        input_id=event.payload["input_id"],
                        answer="yes",
                    )
                )
            resumed = [event async for event in continuation]
            assert resumed[-1].type is EventType.SESSION_COMPLETED
            assert len(entered) == 1
            assert len(provider.requests) == (3 if allowed else 2)
            assert len(refused) == int(not allowed)
            assert all("invocation limits" in str(error) for error in refused)
            stored = await app.session_store.load_events("auxiliary-continuation")
            auxiliary = [
                event for event in stored if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
            ]
            assert len(auxiliary) == int(allowed)
            if allowed:
                assert auxiliary[0].payload["auxiliary_inference"]["tool_call_id"] == "parent"
                assert auxiliary[0].payload["auxiliary_inference"]["purpose"] == "tool.summary"
            usage = await app.get_session_usage("auxiliary-continuation")
            assert usage.model_steps == 2 and usage.usage.total_tokens == (9 if allowed else 4)
        finally:
            if store is not None:
                await store.close()

    async def run():
        async with sqlite_resources:
            await scenario()

    asyncio.run(run())
