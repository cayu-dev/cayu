from __future__ import annotations

import asyncio

from examples.human_attention.app import build
from examples.human_attention.server import create_demo_server
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    RunRequest,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.approvals import ToolApprovalDecision, ToolApprovalRequest, UserInputResponse
from cayu.evals import ScriptedModelProvider
from cayu.messages import MessageRole
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.runtime import HumanAttentionReference, HumanAttentionRequest
from cayu.server import AuthContext, ServerConfig, create_server
from cayu.sessions import InterruptSessionRequest, InvocationOriginClaim, PendingActionQuery
from cayu.tools import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult
from cayu.tools.user_input import UserInputTool


def test_optional_server_attention_uses_protected_current_resolution(tmp_path, monkeypatch):
    async def pause():
        app, store, inbox = build(tmp_path, phase="pause")
        try:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="attention-demo",
                        session_id="attention-example",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            action = (await store.query_pending_actions(PendingActionQuery())).actions[0]
            request = HumanAttentionRequest.from_pending_action(action)
            inbox.accept(request)
            return request
        finally:
            await store.close()

    request = asyncio.run(pause())
    monkeypatch.setenv("ATTENTION_STATE", str(tmp_path))
    monkeypatch.setenv("ATTENTION_SERVER_USERNAME", "operator")
    monkeypatch.setenv("ATTENTION_SERVER_PASSWORD", "local-fixture-password")
    with TestClient(create_demo_server()) as client:
        assert client.get("/api/pending-actions").status_code == 401
        response = client.get("/api/pending-actions", auth=("operator", "local-fixture-password"))
        assert response.status_code == 200, response.text
        action = response.json()["actions"][0]
        assert action["attention_id"] == request.reference.attention_id
        assert action["arguments"] is None
        answer = {
            "session_id": action["session"]["id"],
            "input_id": action["input_id"],
            "answer": "staging",
        }
        assert client.post("/api/user-input/resolve", json=answer).status_code == 401
        resolved = client.post(
            "/api/user-input/resolve", json=answer, auth=("operator", "local-fixture-password")
        )
        assert resolved.status_code == 200, resolved.text
        assert "session.completed" in resolved.text

    async def observe():
        app, store, _inbox = build(tmp_path, phase="consume")
        try:
            assert (await app.get_human_attention_state(request.reference)).state == "resolved"
        finally:
            await store.close()

    asyncio.run(observe())


_QUESTION = "Which environment-question-canary?"
_ARGUMENT = "approval-argument-canary"
_TOKEN = "attention-state-token"


class _NoteTool(Tool):
    spec = ToolSpec(
        name="record",
        description="Record a note.",
        input_schema={
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "additionalProperties": False,
        },
        effect=ToolEffect.IDEMPOTENT,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="Recorded.")


class _ApproveNotes(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL
            if request.tool_name == "record"
            else ToolPolicyDecision.ALLOW
        )


def _respond(request: ModelRequest) -> list[ModelStreamEvent]:
    if any(message.role is MessageRole.TOOL for message in request.messages):
        return [
            ModelStreamEvent.text_delta("Finished."),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    if request.model == "ask":
        call = ModelStreamEvent.tool_call(
            id="attention-call", name="ask_user", arguments={"question": _QUESTION}
        )
    else:
        call = ModelStreamEvent.tool_call(
            id="attention-call", name="record", arguments={"note": _ARGUMENT}
        )
    return [call, ModelStreamEvent.completed({"finish_reason": "tool_calls"})]


def _tenant_auth(request: Request) -> AuthContext:
    if request.headers.get("Authorization") != f"Bearer {_TOKEN}":
        raise HTTPException(status_code=401, detail="Missing or invalid credentials.")
    return AuthContext(subject="attention-reader", tenant=request.headers.get("X-Tenant"))


def _headers(tenant: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {_TOKEN}"}
    if tenant is not None:
        headers["X-Tenant"] = tenant
    return headers


def _app() -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider(response_factory=_respond), default=True)
    app.register_agent(AgentSpec(name="asker", model="ask"), tools=[UserInputTool()])
    app.register_agent(
        AgentSpec(name="approver", model="approve"),
        tools=[_NoteTool()],
        tool_policy=_ApproveNotes(),
    )
    return app


def _call(client: TestClient, function, *args):
    # SDK work shares the server's event loop and in-memory store.
    portal = client.portal
    assert portal is not None
    return portal.call(function, *args)


def _pause(client: TestClient, app: CayuApp, session_id: str, agent: str, tenant: str | None):
    async def run() -> HumanAttentionReference:
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name=agent,
                    session_id=session_id,
                    messages=[Message.text("user", "go")],
                    invocation_origin=None
                    if tenant is None
                    else InvocationOriginClaim(subject="requester", tenant=tenant),
                )
            )
        ]
        page = await app.session_store.query_pending_actions(
            PendingActionQuery(session_id=session_id)
        )
        request = HumanAttentionRequest.from_pending_action(page.actions[0])
        assert request is not None
        return request.reference

    return _call(client, run)


def _drain(client: TestClient, stream_factory) -> None:
    async def run() -> None:
        _ = [event async for event in stream_factory()]

    _call(client, run)


def _state(client: TestClient, reference: HumanAttentionReference, *, tenant=None, **overrides):
    params = reference.model_dump(mode="json", exclude={"attention_id"}, exclude_none=True)
    params.update(overrides)
    return client.get(
        f"/api/human-attention/{reference.attention_id}",
        params=params,
        headers=_headers(tenant),
    )


def _body(response) -> dict:
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    body = response.json()
    assert set(body) == {"attention_id", "state", "observed_at", "reason"}
    assert _QUESTION not in response.text and _ARGUMENT not in response.text
    return {key: value for key, value in body.items() if key != "observed_at"}


def test_human_attention_state_route_reads_each_state_without_content() -> None:
    app = _app()
    server = create_server(app, config=ServerConfig.protected(_tenant_auth))
    with TestClient(server) as client:
        answered = _pause(client, app, "attention-answered", "asker", None)
        denied = _pause(client, app, "attention-denied", "approver", None)
        interrupted = _pause(client, app, "attention-interrupted", "asker", None)
        for reference in (answered, denied, interrupted):
            assert _body(_state(client, reference)) == {
                "attention_id": reference.attention_id,
                "state": "active",
                "reason": "current_action",
            }

        _drain(
            client,
            lambda: app.resolve_user_input(
                UserInputResponse(
                    session_id=answered.session_id, input_id=answered.action_id, answer="staging"
                )
            ),
        )
        _drain(
            client,
            lambda: app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=denied.session_id,
                    approval_id=denied.action_id,
                    tool_round_id=denied.round_id,
                    tool_call_id=denied.tool_call_id,
                    decision=ToolApprovalDecision.DENY,
                )
            ),
        )
        _drain(
            client,
            lambda: app.interrupt_session(
                InterruptSessionRequest(session_id=interrupted.session_id, reason="cancelled")
            ),
        )
        for reference, state in (
            (answered, "resolved"),
            (denied, "cancelled"),
            (interrupted, "superseded"),
        ):
            assert _body(_state(client, reference)) == {
                "attention_id": reference.attention_id,
                "state": state,
                "reason": "durable_supersession" if state == "superseded" else "durable_closure",
            }
            # The route is a thin read over the SDK observation.
            sdk = _call(client, app.get_human_attention_state, reference)
            assert (sdk.state, sdk.reason) == (state, _body(_state(client, reference))["reason"])


def test_human_attention_state_route_requires_auth_and_hides_unknown_and_foreign_ids() -> None:
    app = _app()
    server = create_server(app, config=ServerConfig.protected(_tenant_auth))
    with TestClient(server) as client:
        reference = _pause(client, app, "attention-tenant-a", "asker", "tenant-a")
        unattributed = _pause(client, app, "attention-unattributed", "asker", None)
        params = reference.model_dump(mode="json", exclude={"attention_id"}, exclude_none=True)
        path = f"/api/human-attention/{reference.attention_id}"

        assert client.get(path, params=params).status_code == 401
        assert client.get(path, params=params, headers={"X-Tenant": "tenant-a"}).status_code == 401

        active = {
            "attention_id": reference.attention_id,
            "state": "active",
            "reason": "current_action",
        }
        assert _body(_state(client, reference, tenant="tenant-a")) == active
        # Without a verified tenant the route stays an operator read.
        assert _body(_state(client, reference)) == active

        hidden = {
            "attention_id": reference.attention_id,
            "state": "unavailable",
            "reason": "session_unavailable",
        }
        alias_codec = app.session_store.public_authority_alias_codec
        assert alias_codec is not None
        unresolved_alias = alias_codec.encode("missing-attention-session", field_name="session_id")
        for response in (
            _state(client, reference, tenant="tenant-b"),
            _state(client, reference, session_id="missing-attention-session"),
            _state(client, reference, session_id=unresolved_alias),
            _state(client, reference, session_id="cayu_authority_malformed"),
            _state(client, reference, action_id="another-input"),
            _state(client, reference, tenant="tenant-b", action_id="another-input"),
        ):
            assert _body(response) == hidden
        # A tenant-bound reader cannot see sessions with no tenant on their origin.
        assert _body(_state(client, unattributed, tenant="tenant-a")) == {
            **hidden,
            "attention_id": unattributed.attention_id,
        }
        assert _body(_state(client, unattributed))["state"] == "active"

        malformed = client.get(
            "/api/human-attention/attention_not-a-digest", params=params, headers=_headers()
        )
        assert malformed.status_code == 422
