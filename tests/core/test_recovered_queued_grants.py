"""Final-tool approval completion must retire interaction-scoped grant state."""

import asyncio

import pytest
from tests.core.test_targeted_tool_grants import _codec, _RememberTool
from tests.core.test_tool_completion import FinalTool

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus
from cayu.approvals import ToolApprovalDecision, ToolApprovalRequest
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions import EnqueueSessionMessageRequest, InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.grants import TargetedToolGrant
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("with_grant", [False, True])
@pytest.mark.parametrize("queued", [False, True])
def test_final_tool_approval_finishes_with_targeted_grants(tmp_path, backend, with_grant, queued):
    async def scenario():
        store = (
            InMemorySessionStore(public_authority_alias_codec=_codec())
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "recovery.db", public_authority_alias_codec=_codec())
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="first", name="ask_customer", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.text_delta("Second answer"), ModelStreamEvent.completed()],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        tool = FinalTool()
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            targeted_tool_mode="call_tool",
            tools=[tool, _RememberTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )
        session_id = "recovered-queue-grant"
        try:
            initial = []
            async for event in app.run(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "Ask about the first question")],
                    tool_completion={"tool_names": ["ask_customer"]},
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="remember-fact", tool_id="cayu:remember", max_calls=1
                        ),
                    )
                    if with_grant
                    else (),
                )
            ):
                initial.append(event)
                if queued and event.type is EventType.MODEL_STARTED:
                    await app.enqueue_session_message(
                        EnqueueSessionMessageRequest(
                            session_id=session_id,
                            idempotency_key="follow-up",
                            content="Answer the second question without the old grant",
                            delivery_mode="on_idle",
                        )
                    )
            approval = next(e for e in initial if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
            events = [
                event
                async for event in app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=approval.payload["approval_id"],
                        tool_round_id=approval.payload["tool_round_id"],
                        tool_call_id=approval.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            ]
            assert tool.calls == 1
            assert any(event.type is EventType.INTERACTION_COMPLETED for event in events)
            deliveries = [e for e in events if e.type is EventType.SESSION_MESSAGE_DELIVERED]
            assert len(deliveries) == int(queued)
            assert (await store.load(session_id)).status is SessionStatus.COMPLETED, [
                event.payload for event in events if event.type is EventType.SESSION_FAILED
            ]
            assert len(provider.requests) == 1 + int(queued)
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
