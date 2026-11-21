"""Limited approval recovery merges public and staged results in round order."""

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
async def test_limited_approval_merges_public_and_staged_results(tmp_path, monkeypatch, backend):
    now = [datetime.now(UTC)]
    store = (
        InMemorySessionStore()
        if backend == "memory"
        else SQLiteSessionStore(tmp_path / "mixed-approval-results.sqlite")
    )
    provider = VersionedFakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="first", name="side_effect", arguments={}),
                ModelStreamEvent.tool_call(id="second", name="side_effect", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
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

    async def lose_before_second_terminal(session_id, event):
        if (
            event.type is EventType.TOOL_CALL_COMPLETED
            and event.payload.get("tool_call_id") == "second"
        ):
            raise _SimulatedProcessLoss("worker died after publishing the first result")
        return await original_append(session_id, event)

    try:
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="mixed-approval-results",
                    messages=[Message.text("user", "run both calls")],
                    limits=RunLimits(max_elapsed_seconds=600, scope="session"),
                )
            )
        ]
        pending = (await store.load_checkpoint("mixed-approval-results"))["pending_tool_approval"]
        request = ToolApprovalRequest(
            session_id="mixed-approval-results",
            approval_id=pending["approval_id"],
            tool_round_id=pending["tool_round_id"],
            tool_call_id=pending["tool_call_id"],
            decision=ToolApprovalDecision.APPROVE,
        )
        monkeypatch.setattr(store, "append_event", lose_before_second_terminal)
        with pytest.raises(_SimulatedProcessLoss):
            _ = [event async for event in app.resolve_tool_approval(request)]
        await app.aclose()
        monkeypatch.setattr(store, "append_event", original_append)
        now[0] = (await store.load(request.session_id)).created_at + timedelta(seconds=601)
        recovery_app = make_app()
        recovered = [event async for event in recovery_app.resolve_tool_approval(request)]

        assert recovered[-1].type is EventType.SESSION_INTERRUPTED
        assert (await store.load(request.session_id)).status is SessionStatus.INTERRUPTED
        assert len(tool.calls) == 2
        assert len(provider.requests) == 1
        results = [
            part
            for message in await store.load_transcript(request.session_id)
            for part in message.content
            if isinstance(part, ToolResultPart)
        ]
        assert [part.tool_call_id for part in results] == ["first", "second"]
        assert all(part.content == "recorded" and not part.is_error for part in results)
        terminals = [
            event
            for event in await store.load_events(request.session_id)
            if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        ]
        assert [event.payload["tool_call_id"] for event in terminals] == ["first", "second"]
        assert all(event.type is EventType.TOOL_CALL_COMPLETED for event in terminals)
        assert "pending_tool_round" not in await store.load_checkpoint(request.session_id)
    finally:
        if recovery_app is not None:
            await recovery_app.aclose()
        await app.aclose()
        if backend == "sqlite":
            await store.close()
