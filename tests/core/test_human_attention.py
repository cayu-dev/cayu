from __future__ import annotations

# ruff: noqa: F401, F811
import asyncio

import pytest
from examples.human_attention.app import build
from examples.human_attention.consumer import reconcile
from tests.core.test_session_store_shared_conformance import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)

from cayu import (
    HumanAttentionRequest,
    InterruptSessionRequest,
    Message,
    PendingActionQuery,
    RunRequest,
    ToolApprovalDecision,
    ToolApprovalRequest,
    UserInputResponse,
)


@pytest.mark.parametrize("kind", ["user_input", "tool_approval"])
def test_attention_native_pause_resolution_and_identity(session_store_case, kind, tmp_path):
    async def run():
        store = await _open_store(session_store_case)
        app, _, inbox = build(tmp_path, phase="pause", kind=kind, store=store)
        try:
            session_id = "native-attention-" + kind
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="attention-demo",
                        session_id=session_id,
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            assert str(events[-1].type) == "session.interrupted"
            page = await store.query_pending_actions(PendingActionQuery(session_id=session_id))
            assert not page.issues
            action = page.actions[0]
            session = await store.load(session_id)
            assert action.session.instance_id == session.instance_id
            attention = HumanAttentionRequest.from_pending_action(action)
            assert attention is not None
            assert attention.reference.attention_id == action.attention_id
            from cayu.sessions.base import PendingActionRecord

            restored = PendingActionRecord.model_validate_json(action.model_dump_json())
            assert restored.attention_id == action.attention_id
            cached = action.model_dump()
            cached["attention_id"] = "untrusted-cached-projection"
            assert PendingActionRecord.model_validate(cached).attention_id == action.attention_id
            assert "Which environment?" not in attention.model_dump_json()
            # Different source boundaries for the same logical pause do not
            # create a second attention request.
            alternate = action.model_copy(
                update={
                    "id": "another-source",
                    "event": action.event.model_copy(
                        update={"sequence": action.event.sequence + 1}
                    ),
                }
            )
            assert alternate.attention_id == action.attention_id
            assert (
                action.model_copy(
                    update={
                        "session": action.session.model_copy(update={"instance_id": "replacement"})
                    }
                ).attention_id
                != action.attention_id
            )
            observation = await app.get_human_attention_state(attention.reference)
            assert observation.state == "active"
            await reconcile(app, inbox, session_id=session_id)
            await reconcile(app, inbox, session_id=session_id)
            assert len(inbox.inspect()["notifications"]) == 1
            ref = attention.reference
            if kind == "user_input":
                stream = app.resolve_user_input(
                    UserInputResponse(
                        session_id=session_id, input_id=ref.action_id, answer="staging"
                    )
                )
            else:
                stream = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=ref.action_id,
                        tool_round_id=ref.round_id,
                        tool_call_id=ref.tool_call_id,
                        decision=ToolApprovalDecision.DENY,
                    )
                )
            settled = [event async for event in stream]
            assert str(settled[-1].type) == "session.completed"
            expected = "resolved" if kind == "user_input" else "cancelled"
            assert (await app.get_human_attention_state(ref)).state == expected
            inbox.accept_hint(session_id, "out-of-order-opening-event")
            await reconcile(app, inbox, session_id=session_id)
            assert inbox.inspect()["notifications"][0]["state"] == expected
            inbox.accept(attention)
            inbox.set_state(ref.attention_id, "active")
            assert inbox.inspect()["notifications"][0]["state"] == expected
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_attention_supersession_is_not_an_answer(tmp_path):
    async def run():
        app, store, _inbox = build(tmp_path, phase="pause")
        try:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="attention-demo",
                        session_id="superseded",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            action = (
                await store.query_pending_actions(PendingActionQuery(session_id="superseded"))
            ).actions[0]
            ref = HumanAttentionRequest.from_pending_action(action).reference
            _ = [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(session_id="superseded", reason="operator cancellation")
                )
            ]
            assert (await app.get_human_attention_state(ref)).state == "superseded"
        finally:
            await store.close()

    asyncio.run(run())


def test_attention_failed_and_incomplete_reads_remain_unavailable(tmp_path, monkeypatch):
    from cayu.sessions.base import PendingActionListResult

    async def run():
        app, store, _inbox = build(tmp_path, phase="pause")
        try:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="attention-demo",
                        session_id="unknown",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            action = (
                await store.query_pending_actions(PendingActionQuery(session_id="unknown"))
            ).actions[0]
            ref = HumanAttentionRequest.from_pending_action(action).reference
            original = store.query_pending_actions

            async def truncated(query, **kwargs):
                return PendingActionListResult(
                    actions=[],
                    issues=[],
                    next_cursor="truncated",
                    has_more=True,
                    total_count=None,
                    inspected_candidate_count=1,
                )

            monkeypatch.setattr(store, "query_pending_actions", truncated)
            result = await app.get_human_attention_state(ref)
            assert result.state == "unavailable" and result.reason == "incomplete_query"

            async def failed(query, **kwargs):
                raise OSError("secret-token")

            monkeypatch.setattr(store, "query_pending_actions", failed)
            result = await app.get_human_attention_state(ref)
            assert result.state == "unavailable" and "secret-token" not in result.model_dump_json()
            monkeypatch.setattr(store, "query_pending_actions", original)
            assert (await app.get_human_attention_state(ref)).state == "active"
        finally:
            await store.close()

    asyncio.run(run())


def test_attention_reconciliation_pages_and_concurrent_consumers(tmp_path):
    from examples.human_attention.consumer import Inbox
    from tests.core.pending_action_conformance import (
        _input_checkpoint,
        _tool_round_identity_payload,
    )

    from cayu import CayuApp, Event, EventType, InMemorySessionStore, SessionIdentity, SessionStatus

    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        inbox = Inbox(tmp_path / "pages.sqlite")
        for index in range(205):
            session_id = f"attention-page-{index:04d}"
            input_id, tool_call_id = f"input-{index}", f"call-{index}"
            session = await store.create(
                RunRequest(
                    agent_name="test", session_id=session_id, messages=[Message.text("user", "go")]
                ),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            checkpoint = _input_checkpoint(input_id, tool_call_id, "private question")
            checkpoint["pending_user_input"].update(
                session_id=session_id,
                session_instance_id=session.instance_id,
                source_interaction_id="interaction-" + session_id,
                source_run_epoch=session.run_epoch,
                execution_profile_fingerprint="e" * 64,
            )
            await store.append_event(
                session_id,
                Event(
                    type=EventType.SESSION_AWAITING_USER_INPUT,
                    session_id=session_id,
                    payload={
                        **_tool_round_identity_payload(),
                        "input_id": input_id,
                        "tool_call_id": tool_call_id,
                    },
                ),
            )
            await store.checkpoint(session_id, checkpoint)
            await store.update_status(session_id, SessionStatus.INTERRUPTED)
        inbox.accept_hint("scope", "enrollment")
        incomplete = await reconcile(app, inbox, max_pages=1)
        assert not incomplete["complete"]
        assert incomplete["pending_event_hints"] == 1
        results = await asyncio.gather(reconcile(app, inbox), reconcile(app, inbox))
        assert all(result["complete"] for result in results)
        assert len(inbox.inspect()["notifications"]) == 205
        assert inbox.inspect()["pending_event_hints"] == 0

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["completed", "failed"])
def test_attention_manual_recovery_remains_runtime_owned(session_store_case, outcome, monkeypatch):
    from tests.core.test_user_input import (
        _build,
        _collect,
        _CountingTool,
        _crashed_user_input_resume_events,
        _drain,
        private_events_for_public_events,
    )

    from cayu import ToolApprovalRecoveryOutcome, UserInputRecoveryRequest
    from cayu.tools.user_input import UserInputTool

    async def run():
        store = await _open_store(session_store_case)
        tool = _CountingTool()
        app, _ = _build(
            [("call_1", "count", {}), ("call_2", "ask_user", {"question": "q"})],
            tools=[UserInputTool(), tool],
            store=store,
        )
        try:
            session_id = "manual-attention"
            paused = await _collect(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "go")],
                ),
            )
            original = HumanAttentionRequest.from_pending_action(
                (
                    await store.query_pending_actions(PendingActionQuery(session_id=session_id))
                ).actions[0]
            )
            await store.append_events(
                session_id,
                _crashed_user_input_resume_events(
                    await private_events_for_public_events(store, paused),
                    session_id=session_id,
                    tool_call_id="call_1",
                ),
            )
            stuck = await _drain(
                app.resolve_user_input(
                    UserInputResponse(
                        session_id=session_id, input_id=original.reference.action_id, answer="a"
                    )
                )
            )
            assert stuck[-1].payload.get("manual_recovery_required") is True
            action = (
                await store.query_pending_actions(PendingActionQuery(session_id=session_id))
            ).actions[0]
            manual = HumanAttentionRequest.from_pending_action(action)
            assert manual.reference.kind == "manual_recovery"
            assert manual.reference.attention_id != original.reference.attention_id
            assert (await app.get_human_attention_state(manual.reference)).state == "active"
            await _drain(
                app.recover_user_input(
                    UserInputRecoveryRequest(
                        session_id=session_id,
                        input_id=original.reference.action_id,
                        answer="a",
                        tool_call_id="call_1",
                        outcome=ToolApprovalRecoveryOutcome(outcome),
                        message="Independent external receipt verified.",
                    )
                )
            )
            assert tool.calls == 0
            assert (await app.get_human_attention_state(manual.reference)).state == "resolved"
            if outcome == "failed":
                original_query = store.query_events

                for mutation in (
                    "not_manual",
                    "unknown_result",
                    "unknown_control",
                    "needs_reconciliation",
                ):

                    async def ambiguous(query, *, _mutation=mutation):
                        rows = await original_query(query)
                        changed = []
                        for row in rows:
                            event = row.event.model_copy(deep=True)
                            if event.payload.get("manual_recovery") is True:
                                if _mutation == "not_manual":
                                    event.payload.pop("manual_recovery")
                                elif _mutation == "unknown_result":
                                    event.payload["result"]["structured"] = {
                                        "outcome_unknown": True
                                    }
                                elif _mutation == "unknown_control":
                                    event.payload["outcome_unknown"] = True
                                else:
                                    event.payload["manual_reconciliation_required"] = True
                            changed.append(row.model_copy(update={"event": event}))
                        return changed

                    monkeypatch.setattr(store, "query_events", ambiguous)
                    observed = await app.get_human_attention_state(manual.reference)
                    assert observed.state == "unavailable", mutation
                    assert observed.reason == "no_terminal_evidence", mutation
                monkeypatch.setattr(store, "query_events", original_query)
            assert (await app.get_human_attention_state(original.reference)).state == "resolved"
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_attention_expiry_is_confirmed_by_runtime_closure():
    from examples.human_attention.app import ApprovalTool

    from cayu import (
        AgentSpec,
        AlwaysRequireApprovalToolPolicy,
        CayuApp,
        ModelStreamEvent,
        ScriptedModelProvider,
    )

    async def run():
        app = CayuApp(enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(id="expire-call", name="record", arguments={}),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="expiry", model="fake"),
            tools=[ApprovalTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(expires_in_seconds=0.01),
        )
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="expiry", session_id="expiry", messages=[Message.text("user", "go")]
                )
            )
        ]
        pending = (
            await app.session_store.query_pending_actions(PendingActionQuery(session_id="expiry"))
        ).actions[0]
        ref = HumanAttentionRequest.from_pending_action(pending).reference
        await asyncio.sleep(0.02)
        _ = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="expiry",
                    approval_id=ref.action_id,
                    tool_round_id=ref.round_id,
                    tool_call_id=ref.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        result = await app.get_human_attention_state(ref)
        assert result.state == "expired"

    asyncio.run(run())
