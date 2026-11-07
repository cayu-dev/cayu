"""Continuations after a pause must use the switched model's transcript view."""

import asyncio

import pytest
from tests.core.test_session_model_switch import _app, _EchoTool, _NamedProvider

from cayu import EventType, Message, ResumeRequest, RunRequest, SessionStatus
from cayu.approvals import ToolApprovalDecision, ToolApprovalRequest, UserInputResponse
from cayu.messages import ProviderStatePart, ThinkingPart
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore, ModelTarget
from cayu.sessions.pending_action_contracts import PendingActionQuery
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools import UserInputTool

_HISTORY = [pytest.param(False, id="text-bearing"), pytest.param(True, id="opaque-only")]


def _source(opaque_only: bool) -> _NamedProvider:
    first_answer = [] if opaque_only else [ModelStreamEvent.text_delta("first answer")]
    first_answer.append(
        ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "provider_state": [{"provider": "source", "state": {"id": "response-1"}}],
            }
        )
    )
    return _NamedProvider("source", [first_answer])


def _store(backend: str, tmp_path):
    if backend == "memory":
        return InMemorySessionStore()
    return SQLiteSessionStore(tmp_path / "continuation-history.sqlite")


async def _switch(app, session_id: str, *, opening: str = "initial"):
    initial = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", opening)],
            )
        )
    ]
    assert initial[-1].type is EventType.SESSION_COMPLETED
    paused = [
        event
        async for event in app.resume(
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "switch and call")],
                target=ModelTarget(provider_name="target", model="target-model"),
            )
        )
    ]
    assert paused[-1].type is EventType.SESSION_INTERRUPTED
    return paused


def _assert_portable(request) -> None:
    assert all(
        type(part) not in {ProviderStatePart, ThinkingPart}
        for message in request.messages
        for part in message.content
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("opaque_only", _HISTORY)
def test_model_switch_tool_approval_continues_with_projected_history(
    tmp_path, backend, opaque_only
):
    async def scenario():
        store = _store(backend, tmp_path)
        target = _NamedProvider(
            "target",
            [
                [
                    ModelStreamEvent.tool_call(
                        id="echo-call", name="echo", arguments={"value": "ok"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
            ],
        )
        tool = _EchoTool()
        app, _ = _app(_source(opaque_only), target, store=store, require_approval=True, tool=tool)
        try:
            paused = await _switch(app, "approval-history")
            assert len(target.requests) == 1
            approval = next(
                event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            resolved = [
                event
                async for event in app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id="approval-history",
                        approval_id=approval.payload["approval"]["approval_id"],
                        tool_round_id=approval.payload["tool_round_id"],
                        tool_call_id=approval.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            ]
            assert tool.calls == [{"value": "ok"}]
            assert resolved[-1].type is EventType.SESSION_COMPLETED, [
                event.payload for event in resolved if event.type is EventType.SESSION_FAILED
            ]
            assert len(target.requests) == 2
            _assert_portable(target.requests[1])
            assert target.requests[1].messages[:-2] == target.requests[0].messages
            assert (await store.load("approval-history")).status is SessionStatus.COMPLETED
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("opaque_only", _HISTORY)
def test_model_switch_user_input_continues_with_projected_history(tmp_path, backend, opaque_only):
    async def scenario():
        store = _store(backend, tmp_path)
        target = _NamedProvider(
            "target",
            [
                [
                    ModelStreamEvent.tool_call(
                        id="ask-call", name="ask_user", arguments={"question": "Which one?"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
            ],
        )
        app, _ = _app(_source(opaque_only), target, store=store, tool=UserInputTool())
        try:
            await _switch(app, "input-history")
            assert len(target.requests) == 1
            page = await store.query_pending_actions(PendingActionQuery(session_id="input-history"))
            (question,) = page.actions
            assert question.input_id is not None
            resolved = [
                event
                async for event in app.resolve_user_input(
                    UserInputResponse(
                        session_id="input-history", input_id=question.input_id, answer="That one"
                    )
                )
            ]
            assert resolved[-1].type is EventType.SESSION_COMPLETED, [
                event.payload for event in resolved if event.type is EventType.SESSION_FAILED
            ]
            assert len(target.requests) == 2
            _assert_portable(target.requests[1])
            assert target.requests[1].messages[:-2] == target.requests[0].messages
            assert (await store.load("input-history")).status is SessionStatus.COMPLETED
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
