from __future__ import annotations

import asyncio
import html
from datetime import UTC, datetime

import pytest
from tests.core.test_approval_lifecycle_execution_identities import (
    _RecordingTool,
    _RequireApprovalPolicy,
)
from tests.core.test_user_input import _ScriptedProvider

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    Message,
    RunRequest,
    SQLiteSessionStore,
    ToolApprovalDecision,
    ToolApprovalRequest,
    UserInputResponse,
)
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.environments.factory import EnvironmentFactory, EnvironmentFactoryResult
from cayu.runtime.human_review import (
    HumanReviewContext,
    HumanReviewDenied,
    HumanReviewDisclosure,
    HumanReviewField,
    HumanReviewPolicy,
    HumanReviewSource,
    build_review,
)
from cayu.tools.user_input import UserInputTool
from cayu.vaults import SecretRedactor

QUESTION = "Which delivery window should I use: morning or afternoon?"
CONTEXT = HumanReviewContext(recipient="operator", tenant="tenant-a", purpose="delivery")


class ReviewPolicy(HumanReviewPolicy):
    version = "delivery-v1"
    binding_key = b"test-fixture-only-binding-key-32-bytes"
    can_decide = True
    attest = True

    def authorize(self, context, *, session_id, session_metadata, action):
        return (
            context == CONTEXT
            and session_id == "review-session"
            and (action == "inspect" or self.can_decide)
        )

    def project(self, context, source):
        # Application-owned vocabulary attestation, not a current-secret scan.
        if source.kind == "user_input":
            if source.question != QUESTION:
                return HumanReviewDisclosure(status="redacted")
            text = source.question
        else:
            if any(args != {"value": "morning"} for args in source.arguments_by_call.values()):
                return HumanReviewDisclosure(status="redacted")
            text = "Schedule delivery in the morning."
        return HumanReviewDisclosure(
            status="permitted",
            fields=(HumanReviewField(label="Proposal", text=text),),
            sensitive_content="application_attested" if self.attest else "reject_unknown_scope",
        )


def identity(name):
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version="1", implementation_version="1"
    )


class Provider(_ScriptedProvider):
    execution_profile_identity = identity("review-provider")


class RecordingTool(_RecordingTool):
    execution_profile_identity = identity("review-tool")


class ApprovalPolicy(_RequireApprovalPolicy):
    execution_profile_identity = identity("review-approval")


class DynamicFactory(EnvironmentFactory):
    execution_profile_identity = identity("review-factory")

    async def create(self, request):
        return EnvironmentFactoryResult(
            environment=Environment(EnvironmentSpec(name=request.environment_name))
        )


def make_app(store, policy, *, approval=False, resumed=False):
    provider = Provider(
        [
            ("call-1", "side_effect", {"value": "morning"})
            if approval
            else ("call-1", "ask_user", {"question": QUESTION}),
        ]
    )
    if resumed:
        provider.requests.append(None)
    app = CayuApp(session_store=store, human_review_policy=policy, enable_logging=False)
    app.register_provider(provider, default=True)
    tool = RecordingTool() if approval else UserInputTool()
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=ApprovalPolicy() if approval else None,
    )
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic", execution_profile_identity=identity("review-environment")),
        DynamicFactory(),
        default=True,
    )
    return app, provider, tool


async def pause(app):
    return [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="review-session",
                messages=[Message.text("user", "go")],
            )
        )
    ]


def decision(view, *, approval=False):
    if approval:
        return ToolApprovalRequest(
            session_id=view.session_id,
            approval_id=view.interaction_id,
            tool_round_id=view.tool_round_id,
            tool_call_id=view.tool_call_id,
            decision=ToolApprovalDecision.APPROVE,
            review_reference=view.reference,
        )
    return UserInputResponse(
        session_id=view.session_id,
        input_id=view.interaction_id,
        answer="morning",
        review_reference=view.reference,
    )


async def resolve(app, request):
    stream = (
        app.resolve_tool_approval(request)
        if isinstance(request, ToolApprovalRequest)
        else app.resolve_user_input(request)
    )
    return [event async for event in stream]


@pytest.mark.parametrize("approval", [False, True])
def test_sqlite_restart_review_and_exact_decision(tmp_path, approval):
    async def scenario():
        path = tmp_path / "sessions.sqlite"
        store = SQLiteSessionStore(path)
        policy = ReviewPolicy()
        app, provider, _ = make_app(store, policy, approval=approval)
        events = await pause(app)
        assert QUESTION not in str([event.model_dump() for event in events])
        before = await store.load_checkpoint("review-session")
        view = await app.inspect_human_review("review-session", context=CONTEXT)
        assert view.status == "permitted", view
        assert view.reference is not None
        assert view.calls[0].on_grant == "eligible"
        assert len(provider.requests) == 1
        assert await store.load_checkpoint("review-session") == before
        await store.close()
        store = SQLiteSessionStore(path)
        app, provider, tool = make_app(store, ReviewPolicy(), approval=approval, resumed=True)
        again = await app.inspect_human_review("review-session", context=CONTEXT)
        assert again == view
        request = decision(again, approval=approval)
        result = await resolve(app, request)
        assert result
        replay = await resolve(app, request)
        assert replay
        if approval:
            assert tool.calls == [{"value": "morning"}]
        assert (
            await app.inspect_human_review("review-session", context=CONTEXT)
        ).status == "unavailable"
        assert QUESTION not in str([event.model_dump() for event in result + replay])
        await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("approval", [False, True])
@pytest.mark.parametrize("drift", ["policy", "key", "projection", "permission"])
def test_review_drift_rejected_without_execution(tmp_path, approval, drift):
    async def scenario():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        policy = ReviewPolicy()
        app, provider, tool = make_app(store, policy, approval=approval)
        await pause(app)
        view = await app.inspect_human_review("review-session", context=CONTEXT)
        assert view.status == "permitted"
        if drift == "policy":
            policy.version = "delivery-v2"
        elif drift == "key":
            policy.binding_key = b"different-key-material-32-bytes-long"
        elif drift == "projection":
            policy.attest = False
        else:
            policy.can_decide = False
        with pytest.raises((ValueError, PermissionError)):
            await resolve(app, decision(view, approval=approval))
        assert len(provider.requests) == 1
        if approval:
            assert not tool.calls
        await store.close()

    asyncio.run(scenario())


def test_unauthorized_and_unknown_scope_fail_closed(tmp_path):
    async def scenario():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        policy = ReviewPolicy()
        app, _, _ = make_app(store, policy)
        await pause(app)
        for context in [
            CONTEXT.model_copy(update={"tenant": "tenant-b"}),
            CONTEXT.model_copy(update={"recipient": "intruder"}),
            CONTEXT.model_copy(update={"purpose": "export"}),
        ]:
            with pytest.raises(HumanReviewDenied):
                await app.inspect_human_review("review-session", context=context)
        with pytest.raises(HumanReviewDenied):
            await app.inspect_human_review("another-session", context=CONTEXT)
        policy.attest = False
        view = await app.inspect_human_review("review-session", context=CONTEXT)
        assert view.status == "redacted"
        assert not view.fields
        assert "withheld" in view.guidance
        await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "text,secret,status",
    [
        ("<img src=x onerror=alert(1)>", None, "permitted"),
        ("secret-value", "secret-value", "redacted"),
        ("x" * 4097, None, "unavailable"),
    ],
)
def test_display_bounds_escaping_and_secret_checks(text, secret, status):
    class Projection(ReviewPolicy):
        def project(self, context, source):
            return HumanReviewDisclosure(
                status="permitted", fields=(HumanReviewField(label="Question", text=text),)
            )

    view = build_review(
        policy=Projection(),
        context=CONTEXT,
        source=HumanReviewSource(
            kind="user_input",
            interaction_id="input",
            tool_round_id="round",
            tool_call_id="call",
            secret_resolution_scope="static",
            calls=(),
            arguments_by_call={},
        ),
        session_id="review-session",
        session_instance_id="incarnation",
        authoritative_content={},
        redactor=SecretRedactor() if secret is None else SecretRedactor().with_secret(secret),
        now=datetime.now(UTC),
    )
    assert view.status == status
    if status == "permitted":
        assert view.fields[0].text == html.escape(text)
    else:
        assert not view.fields


@pytest.mark.parametrize("approval", [False, True])
def test_competing_resolvers_do_not_duplicate_execution(tmp_path, approval):
    async def scenario():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        app, provider, tool = make_app(store, ReviewPolicy(), approval=approval)
        await pause(app)
        view = await app.inspect_human_review("review-session", context=CONTEXT)
        first = decision(view, approval=approval)
        second = (
            first.model_copy(update={"decision": ToolApprovalDecision.DENY})
            if approval
            else first.model_copy(update={"answer": "afternoon"})
        )
        results = await asyncio.gather(
            resolve(app, first), resolve(app, second), return_exceptions=True
        )
        assert sum(not isinstance(result, BaseException) for result in results) == 1
        assert len(provider.requests) <= 2
        if approval:
            assert len(tool.calls) <= 1
        await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("approval", [False, True])
def test_separate_worker_inspects_and_resolves(tmp_path, approval):
    import os
    import subprocess
    import sys
    from pathlib import Path

    path = tmp_path / "sessions.sqlite"

    async def setup():
        store = SQLiteSessionStore(path)
        app, _, _ = make_app(store, ReviewPolicy(), approval=approval)
        await pause(app)
        view = await app.inspect_human_review("review-session", context=CONTEXT)
        await store.close()
        return view.reference.content_tag

    tag = asyncio.run(setup())
    script = """
import asyncio, sys
from tests.core.test_human_review import *
async def main():
    store = SQLiteSessionStore(sys.argv[1])
    approval = sys.argv[2] == "True"
    app, provider, tool = make_app(store, ReviewPolicy(), approval=approval, resumed=True)
    view = await app.inspect_human_review("review-session", context=CONTEXT)
    assert view.status == "permitted"
    assert view.reference.content_tag == sys.argv[3]
    await resolve(app, decision(view, approval=approval))
    await resolve(app, decision(view, approval=approval))
    if approval:
        assert len(tool.calls) == 1
    await store.close()
asyncio.run(main())
"""
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path), str(approval), tag],
        cwd=root,
        env={**os.environ, "PYTHONPATH": f"{root / 'src'}:{root}"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def test_protected_review_route_and_inspect_only_recipient(tmp_path):
    from fastapi.testclient import TestClient

    from cayu.server import AuthContext, ServerConfig, create_server

    store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    policy = ReviewPolicy()
    policy.can_decide = False
    app, provider, _ = make_app(store, policy)
    asyncio.run(pause(app))

    def auth(request):
        return AuthContext(subject="operator", tenant=request.headers.get("x-tenant", "tenant-a"))

    with TestClient(create_server(app, config=ServerConfig.protected(auth))) as client:
        response = client.get(
            "/api/sessions/review-session/human-review", params={"purpose": "delivery"}
        )
        assert response.status_code == 200, response.text
        view = response.json()
        assert view["status"] == "permitted"
        assert response.headers["cache-control"] == "no-store"
        assert (
            client.get(
                "/api/sessions/review-session/human-review",
                params={"purpose": "delivery"},
                headers={"x-tenant": "tenant-b"},
            ).status_code
            == 403
        )
        request = {
            "session_id": "review-session",
            "input_id": view["interaction_id"],
            "answer": "morning",
            "review_reference": view["reference"],
        }
        response = client.post("/api/user-input/resolve", json=request)
        assert response.status_code == 403, response.text
        request.pop("review_reference")
        assert client.post("/api/user-input/resolve", json=request).status_code == 403
        assert len(provider.requests) == 1
    asyncio.run(store.close())
