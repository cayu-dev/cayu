from __future__ import annotations

import asyncio

import pytest
from tests.core.test_approval_lifecycle_execution_identities import (
    _FailingTerminalToolEventStore,
)
from tests.core.test_human_review import (
    CONTEXT,
    ApprovalPolicy,
    Provider,
    RecordingTool,
    ReviewPolicy,
    decision,
    pause,
    resolve,
)

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    SQLiteSessionStore,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    UserInputRecoveryRequest,
)
from cayu.runtime.human_review import (
    HumanReviewConflict,
    HumanReviewDenied,
    HumanReviewDisclosure,
    HumanReviewField,
)
from cayu.runtime.user_input import (
    user_input_answer_request_digest,
    user_input_resolution_request_digest,
)
from cayu.tools.user_input import UserInputTool


class RecoveryPolicy(ReviewPolicy):
    def authorize(self, context, *, session_id, session_metadata, action):
        return (
            context.tenant == CONTEXT.tenant
            and context.purpose == CONTEXT.purpose
            and context.recipient in {CONTEXT.recipient, "recovery-operator"}
            and session_id == "review-session"
            and (action == "inspect" or self.can_decide)
        )

    def project(self, context, source):
        return HumanReviewDisclosure(
            status="permitted",
            fields=(HumanReviewField(label="Proposal", text="Run the two fixture actions."),),
            sensitive_content="application_attested",
        )


def make_recovery_app(*, approval, store=None, resumed=False):
    store = _FailingTerminalToolEventStore() if store is None else store
    policy = RecoveryPolicy()
    app = CayuApp(session_store=store, human_review_policy=policy, enable_logging=False)
    calls = [
        ("call_first", "side_effect", {"value": "first"}),
        ("call_second", "side_effect", {"value": "second"}),
    ]
    if not approval:
        calls.append(("call_question", "ask_user", {"question": "Which window?"}))
    provider = Provider(calls)
    if resumed:
        provider.requests.append(None)
    tool = RecordingTool()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool] if approval else [tool, UserInputTool()],
        tool_policy=ApprovalPolicy() if approval else None,
    )
    return app, store, policy, tool


async def interrupted_decision(*, approval, store=None):
    app, store, policy, tool = make_recovery_app(approval=approval, store=store)
    await pause(app)
    view = await app.inspect_human_review("review-session", context=CONTEXT)
    assert view.status == "permitted"
    request = decision(view, approval=approval)
    events = await resolve(app, request)
    assert events[-1].type is EventType.SESSION_INTERRUPTED
    assert tool.calls == [{"value": "first"}]
    recovery_view = await app.inspect_human_review("review-session", context=CONTEXT)
    assert recovery_view.status == "unavailable"
    assert recovery_view.reference is not None
    assert recovery_view.reference != view.reference
    assert not recovery_view.fields
    assert "cannot authorize execution" in recovery_view.guidance
    return app, store, policy, tool, request, recovery_view


@pytest.mark.parametrize("approval", [False, True])
def test_interrupted_review_recovers_without_replaying_the_effect(approval):
    async def scenario():
        app, store, policy, tool, original, view = await interrupted_decision(approval=approval)
        checkpoint = await store.load_checkpoint("review-session")
        # A fresh recovery view is not a replacement answer or approval grant.
        with pytest.raises((ValueError, RuntimeError)):
            await resolve(app, original.model_copy(update={"review_reference": view.reference}))
        assert await store.load_checkpoint("review-session") == checkpoint

        if approval:
            recovery = ToolApprovalRecoveryRequest(
                session_id=view.session_id,
                approval_id=view.interaction_id,
                tool_round_id=view.tool_round_id,
                tool_call_id="call_first",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="Completed externally.",
                review_reference=view.reference,
            )
            with pytest.raises(HumanReviewConflict):
                _ = [
                    event
                    async for event in app.recover_tool_approval(
                        recovery.model_copy(update={"review_reference": original.review_reference})
                    )
                ]
            recovered = [event async for event in app.recover_tool_approval(recovery)]
            assert recovered[-1].type is EventType.SESSION_INTERRUPTED
            assert "cannot authorize pending sibling execution" in recovered[-1].payload["error"]
            assert tool.calls == [{"value": "first"}]

            policy.can_decide = False
            with pytest.raises(HumanReviewDenied):
                await resolve(app, original)
            policy.can_decide = True
            completed = await resolve(app, original)
        else:
            recovery = UserInputRecoveryRequest(
                session_id=view.session_id,
                input_id=view.interaction_id,
                answer=original.answer,
                tool_call_id="call_first",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="Completed externally.",
                review_reference=view.reference,
                answer_review_reference=original.review_reference,
            )
            assert user_input_answer_request_digest(recovery) == user_input_answer_request_digest(
                original
            )
            missing_answer_reference = recovery.model_copy(update={"answer_review_reference": None})
            assert user_input_resolution_request_digest(
                recovery
            ) != user_input_resolution_request_digest(missing_answer_reference)
            with pytest.raises((ValueError, RuntimeError)):
                _ = [event async for event in app.recover_user_input(missing_answer_reference)]
            assert await store.load_checkpoint("review-session") == checkpoint
            completed = [event async for event in app.recover_user_input(recovery)]
            # Lost-ack replay uses the accepted recovery request, not a new view.
            replay = [event async for event in app.recover_user_input(recovery)]
            assert replay

        assert completed[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == [{"value": "first"}, {"value": "second"}]

    asyncio.run(scenario())


@pytest.mark.parametrize("approval", [False, True])
def test_recovery_view_rejects_policy_drift_and_permission_revocation(approval):
    async def scenario():
        app, store, policy, tool, original, view = await interrupted_decision(approval=approval)
        checkpoint = await store.load_checkpoint("review-session")
        common = {
            "session_id": view.session_id,
            "tool_call_id": "call_first",
            "outcome": ToolApprovalRecoveryOutcome.COMPLETED,
            "message": "Completed externally.",
            "review_reference": view.reference,
        }
        if approval:
            request = ToolApprovalRecoveryRequest(
                **common, approval_id=view.interaction_id, tool_round_id=view.tool_round_id
            )
            recover = app.recover_tool_approval
        else:
            request = UserInputRecoveryRequest(
                **common,
                input_id=view.interaction_id,
                answer=original.answer,
                answer_review_reference=original.review_reference,
            )
            recover = app.recover_user_input
        policy.can_decide = False
        with pytest.raises(HumanReviewDenied):
            _ = [event async for event in recover(request)]
        policy.can_decide = True
        policy.version = "changed"
        with pytest.raises(HumanReviewConflict):
            _ = [event async for event in recover(request)]
        assert await store.load_checkpoint("review-session") == checkpoint
        assert tool.calls == [{"value": "first"}]

    asyncio.run(scenario())


@pytest.mark.parametrize("approval", [False, True])
def test_exact_accepted_retry_keeps_original_authority_without_redispatch(approval):
    async def scenario():
        app, store, policy, tool, original, _view = await interrupted_decision(approval=approval)
        before = await store.load_checkpoint("review-session")
        policy.can_decide = False
        with pytest.raises(HumanReviewDenied):
            await resolve(app, original)
        policy.can_decide = True
        with pytest.raises((ValueError, RuntimeError)):
            await resolve(app, original.model_copy(update={"metadata": {"changed": True}}))
        assert await store.load_checkpoint("review-session") == before
        events = await resolve(app, original)
        assert events[-1].type is EventType.SESSION_INTERRUPTED
        assert tool.calls == [{"value": "first"}]

    asyncio.run(scenario())


@pytest.mark.parametrize("approval", [False, True])
@pytest.mark.parametrize("drift", ["content", "intent"])
def test_recovery_reference_binds_current_content_and_intent(approval, drift):
    async def scenario():
        app, store, _policy, tool, original, view = await interrupted_decision(approval=approval)

        def change(_session, checkpoint):
            if drift == "intent":
                key = "approval_resolution_intent" if approval else "user_input_resolution_intent"
                checkpoint[key]["resolution_request_digest"] = "f" * 64
            else:
                key = "pending_tool_approval" if approval else "pending_user_input"
                checkpoint[key]["tool_calls"][1]["arguments"]["value"] = "changed"
                if approval:
                    checkpoint["pending_tool_round"]["tool_calls"][1]["arguments"]["value"] = (
                        "changed"
                    )
            return checkpoint

        if not approval and drift == "content":
            before = await store.load_checkpoint("review-session")
            # User-input claims already bind the complete pending pause. The
            # store refuses this drift even before the review claim is reached.
            with pytest.raises(RuntimeError, match="intent conflicts with its pending pause"):
                await store.transform_checkpoint("review-session", change)
            assert await store.load_checkpoint("review-session") == before
            assert tool.calls == [{"value": "first"}]
            return
        await store.transform_checkpoint("review-session", change)
        changed = await store.load_checkpoint("review-session")
        if approval and drift == "content":
            # Matching the accepted request is insufficient if its approved
            # arguments changed. The durable content binding must still match.
            with pytest.raises(RuntimeError, match="intent conflicts with its pending approval"):
                await resolve(app, original)
        common = {
            "session_id": view.session_id,
            "tool_call_id": "call_first",
            "outcome": ToolApprovalRecoveryOutcome.COMPLETED,
            "message": "Completed externally.",
            "review_reference": view.reference,
        }
        if approval:
            request = ToolApprovalRecoveryRequest(
                **common, approval_id=view.interaction_id, tool_round_id=view.tool_round_id
            )
            recover = app.recover_tool_approval
        else:
            request = UserInputRecoveryRequest(
                **common,
                input_id=view.interaction_id,
                answer=original.answer,
                answer_review_reference=original.review_reference,
            )
            recover = app.recover_user_input
        with pytest.raises((HumanReviewConflict, RuntimeError)):
            _ = [event async for event in recover(request)]
        assert await store.load_checkpoint("review-session") == changed
        assert tool.calls == [{"value": "first"}]

    asyncio.run(scenario())


class FailingSQLiteStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1
    failed_terminal_once = False

    async def append_events(self, session_id, events):
        if not self.failed_terminal_once and any(
            event.type is EventType.TOOL_CALL_COMPLETED for event in events
        ):
            self.failed_terminal_once = True
            raise RuntimeError("terminal tool event unavailable")
        await super().append_events(session_id, events)


@pytest.mark.parametrize("approval", [False, True])
def test_sqlite_restart_recovers_with_fresh_reference_and_original_decision(tmp_path, approval):
    async def scenario():
        path = tmp_path / "sessions.sqlite"
        first_store = FailingSQLiteStore(path)
        _app, _, _, first_tool, original, previous = await interrupted_decision(
            approval=approval, store=first_store
        )
        await first_store.close()
        store = SQLiteSessionStore(path)
        try:
            app, _, _, tool = make_recovery_app(approval=approval, store=store, resumed=True)
            view = await app.inspect_human_review("review-session", context=CONTEXT)
            assert view == previous
            if approval:
                recovered = [
                    event
                    async for event in app.recover_tool_approval(
                        ToolApprovalRecoveryRequest(
                            session_id=view.session_id,
                            approval_id=view.interaction_id,
                            tool_round_id=view.tool_round_id,
                            tool_call_id="call_first",
                            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                            message="Completed externally.",
                            review_reference=view.reference,
                        )
                    )
                ]
                assert recovered[-1].type is EventType.SESSION_INTERRUPTED
                assert not tool.calls
                completed = await resolve(app, original)
            else:
                completed = [
                    event
                    async for event in app.recover_user_input(
                        UserInputRecoveryRequest(
                            session_id=view.session_id,
                            input_id=view.interaction_id,
                            answer=original.answer,
                            tool_call_id="call_first",
                            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                            message="Completed externally.",
                            review_reference=view.reference,
                            answer_review_reference=original.review_reference,
                        )
                    )
                ]
            assert completed[-1].type is EventType.SESSION_COMPLETED
            assert first_tool.calls == [{"value": "first"}]
            assert tool.calls == [{"value": "second"}]
        finally:
            await store.close()

    asyncio.run(scenario())


def test_user_input_recovery_route_preserves_answer_identity_and_checks_recipient():
    from fastapi.testclient import TestClient

    from cayu.server import AuthContext, ServerConfig, create_server

    app, _store, _policy, tool = make_recovery_app(approval=False)
    asyncio.run(pause(app))

    def auth(request):
        return AuthContext(subject=request.headers["x-recipient"], tenant=CONTEXT.tenant)

    with TestClient(create_server(app, config=ServerConfig.protected(auth))) as client:
        headers = {"x-recipient": "operator"}
        initial = client.get(
            "/api/sessions/review-session/human-review",
            params={"purpose": CONTEXT.purpose},
            headers=headers,
        ).json()
        accepted = client.post(
            "/api/user-input/resolve",
            json={
                "session_id": "review-session",
                "input_id": initial["interaction_id"],
                "answer": "morning",
                "review_reference": initial["reference"],
            },
            headers=headers,
        )
        assert accepted.status_code == 200, accepted.text
        assert tool.calls == [{"value": "first"}]
        inspected = client.get(
            "/api/sessions/review-session/human-review",
            params={"purpose": CONTEXT.purpose},
            headers=headers,
        )
        assert inspected.status_code == 200
        view = inspected.json()
        assert view["status"] == "unavailable" and view["reference"]
        body = {
            "session_id": "review-session",
            "input_id": view["interaction_id"],
            "tool_call_id": "call_first",
            "outcome": "completed",
            "message": "Completed externally.",
            "answer": "morning",
            "review_reference": view["reference"],
            "answer_review_reference": initial["reference"],
        }
        rejected = client.post(
            "/api/user-input/recover", json=body, headers={"x-recipient": "recovery-operator"}
        )
        assert rejected.status_code == 403
        assert tool.calls == [{"value": "first"}]
        recovered = client.post("/api/user-input/recover", json=body, headers=headers)
        assert recovered.status_code == 200, recovered.text
        assert tool.calls == [{"value": "first"}, {"value": "second"}]
