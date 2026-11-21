"""Approval-limit recovery must not replace durable executed effects with skips."""

from datetime import UTC, datetime, timedelta

import pytest
from tests.core.session_store_conformance.support import _SimulatedProcessLoss
from tests.runtime.runtime_provider_support import VersionedFakeProvider
from tests.runtime.runtime_tool_support import SideEffectTool

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus
from cayu.approvals import ToolApprovalDecision, ToolApprovalRequest
from cayu.budgets.run_limits import RunLimits
from cayu.messages import ToolResultPart
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore
from cayu.storage import SQLiteSessionStore
from cayu.tools import AlwaysRequireApprovalToolPolicy


class _SerialSideEffectTool(SideEffectTool):
    spec = SideEffectTool.spec.model_copy(
        update={"parallel_safe": False, "max_terminal_payload_bytes": 4096}
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("limit_expired", [False, True], ids=["within-limit", "expired-limit"])
async def test_approval_limit_preserves_staged_executed_result(
    tmp_path, monkeypatch, backend, limit_expired
):
    now = [datetime.now(UTC)]
    store = (
        InMemorySessionStore()
        if backend == "memory"
        else SQLiteSessionStore(tmp_path / "approval-limit-staged-effect.sqlite")
    )
    provider = VersionedFakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="first", name="side_effect", arguments={"value": "first"}
                ),
                ModelStreamEvent.tool_call(
                    id="second", name="side_effect", arguments={"value": "second"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
        ]
    )
    tool = _SerialSideEffectTool()

    def make_app():
        app = CayuApp(session_store=store, clock=lambda: now[0], enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )
        return app

    app = make_app()
    recovery_app = None
    original_append = store.append_event
    lose_process = True

    async def lose_before_first_terminal(session_id, event):
        if (
            lose_process
            and event.type is EventType.TOOL_CALL_COMPLETED
            and event.payload.get("tool_call_id") == "first"
        ):
            # Effects and their private terminal stages are durable. The first
            # result's hooks have completed, but its public append has not.
            raise _SimulatedProcessLoss("worker died before terminal append")
        return await original_append(session_id, event)

    monkeypatch.setattr(store, "append_event", lose_before_first_terminal)
    try:
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="approval-limit-staged-effect",
                    messages=[Message.text("user", "run both calls")],
                    limits=RunLimits(max_elapsed_seconds=600, scope="session"),
                )
            )
        ]
        pending = (await store.load_checkpoint("approval-limit-staged-effect"))[
            "pending_tool_approval"
        ]
        request = ToolApprovalRequest(
            session_id="approval-limit-staged-effect",
            approval_id=pending["approval_id"],
            tool_round_id=pending["tool_round_id"],
            tool_call_id=pending["tool_call_id"],
            decision=ToolApprovalDecision.APPROVE,
        )
        with pytest.raises(_SimulatedProcessLoss):
            _ = [event async for event in app.resolve_tool_approval(request)]
        assert tool.calls == [{"value": "first"}, {"value": "second"}]
        checkpoint = await store.load_checkpoint(request.session_id)
        assert "pending_tool_approval" in checkpoint
        stages = checkpoint["pending_tool_round"]["staged_terminals"]
        assert len(stages) == 2
        first = next(stage for stage in stages if stage["tool_call_id"] == "first")
        assert first["hooks_state"] == "completed"
        staged_event = first["event"]
        assert staged_event["type"] == EventType.TOOL_CALL_COMPLETED
        assert staged_event["payload"]["result"]["content"] == "recorded"
        assert not any(
            event.type is EventType.TOOL_CALL_COMPLETED
            for event in await store.load_events(request.session_id)
        )
        await app.aclose()

        lose_process = False
        if limit_expired:
            now[0] = (await store.load(request.session_id)).created_at + timedelta(seconds=601)
        recovery_app = make_app()
        recovered = [event async for event in recovery_app.resolve_tool_approval(request)]
        expected_status = SessionStatus.INTERRUPTED if limit_expired else SessionStatus.COMPLETED
        assert (await store.load(request.session_id)).status is expected_status
        assert recovered[-1].type is (
            EventType.SESSION_INTERRUPTED if limit_expired else EventType.SESSION_COMPLETED
        )
        assert tool.calls == [{"value": "first"}, {"value": "second"}]
        assert len(provider.requests) == (1 if limit_expired else 2)
        assert "pending_tool_round" not in await store.load_checkpoint(request.session_id)

        results = [
            part
            for message in await store.load_transcript(request.session_id)
            for part in message.content
            if isinstance(part, ToolResultPart)
        ]
        assert [part.tool_call_id for part in results] == ["first", "second"]
        assert results[0].content == "recorded", (
            "Approval limit closure discarded an already-executed, hook-complete "
            f"staged result and replaced it with: {results[0].content!r}"
        )
        assert not results[0].is_error
        durable = await store.load_events(request.session_id)
        assert any(event.id == staged_event["id"] for event in durable)
        assert not any(
            event.type is EventType.TOOL_CALL_FAILED
            and event.payload.get("tool_call_id") == "first"
            for event in durable
        )
    finally:
        if recovery_app is not None:
            await recovery_app.aclose()
        await app.aclose()
        if backend == "sqlite":
            await store.close()
