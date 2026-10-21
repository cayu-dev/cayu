"""The agent's inbox: one record per approval, question or proposal, settled once.

These tests settle items over HTTP as named people and check what the other
surfaces then report: the inbox item, pending actions, human-attention state
and supervision events.
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from tests.core._event_projection_support import private_events_for_public_events
from tests.core.test_user_input import (
    _CountingTool,
    _crashed_user_input_resume_events,
    _ScriptedProvider,
)

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    RunRequest,
    SessionStatus,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.approvals import HumanReviewDisclosure, HumanReviewField, HumanReviewPolicy
from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.approvals.user_input import UserInputResponse
from cayu.evals import ScriptedModelProvider
from cayu.events import Event, EventType
from cayu.messages import MessageRole
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.runtime import HumanAttentionReference, HumanAttentionRequest
from cayu.runtime._event_projection import public_event_linkage_id
from cayu.server import AuthContext, ServerConfig, create_server
from cayu.server._inbox_routes import _event_reference
from cayu.server.inbox import InboxItemScope
from cayu.sessions import InvocationOriginClaim, PendingActionQuery
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.pending_action_contracts import human_attention_id
from cayu.sessions.records import EventRecord, PendingActionKind
from cayu.storage import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeActorType,
    KnowledgeEntry,
    KnowledgeStatus,
    SQLiteSessionStore,
)
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import TaskStatus
from cayu.tasks.worker import TaskHandlerOutcome, run_task_worker
from cayu.tools import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult
from cayu.tools.user_input import UserInputTool
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

_QUESTION = "Which environment should I deploy to?"
_PEOPLE = {
    "priya-token": ("priya", None),
    "tom-token": ("tom", None),
    "acme-token": ("acme-controller", "acme"),
    "globex-token": ("globex-controller", "globex"),
}


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
        return ToolResult(content=f"Recorded {args['note']}.")


class _ApproveNotes(ToolPolicy):
    def __init__(self, expires_in_seconds: float | None = None) -> None:
        self._expires_in_seconds = expires_in_seconds

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.tool_name != "record":
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            reason="Approve the September correction package.",
            approval_expires_in_seconds=self._expires_in_seconds,
        )


def _respond(request: ModelRequest) -> list[ModelStreamEvent]:
    if any(message.role is MessageRole.TOOL for message in request.messages):
        return [
            ModelStreamEvent.text_delta("Finished."),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    if request.model == "ask":
        call = ModelStreamEvent.tool_call(
            id="inbox-call", name="ask_user", arguments={"question": _QUESTION}
        )
    else:
        call = ModelStreamEvent.tool_call(
            id="inbox-call", name="record", arguments={"note": "proposed"}
        )
    return [call, ModelStreamEvent.completed({"finish_reason": "tool_calls"})]


def _authenticate(request: Request) -> AuthContext:
    token = request.headers.get("Authorization", "").removeprefix("Bearer ")
    if token not in _PEOPLE:
        raise HTTPException(status_code=401, detail="Missing or invalid credentials.")
    subject, tenant = _PEOPLE[token]
    return AuthContext(subject=subject, tenant=tenant)


def _as(person: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {person}-token"}


def _app(**options) -> CayuApp:
    expires_in_seconds = options.pop("expires_in_seconds", None)
    app = CayuApp(enable_logging=False, **options)
    app.register_provider(ScriptedModelProvider(response_factory=_respond), default=True)
    app.register_agent(AgentSpec(name="asker", model="ask"), tools=[UserInputTool()])
    app.register_agent(
        AgentSpec(name="approver", model="approve"),
        tools=[_NoteTool()],
        tool_policy=_ApproveNotes(expires_in_seconds),
    )
    return app


def _pause(client: TestClient, app: CayuApp, session_id: str, agent: str, tenant=None) -> None:
    async def run() -> None:
        async for _event in app.run(
            RunRequest(
                agent_name=agent,
                session_id=session_id,
                messages=[Message.text("user", "go")],
                invocation_origin=(
                    None
                    if tenant is None
                    else InvocationOriginClaim(subject="requester", tenant=tenant)
                ),
            )
        ):
            pass

    _portal(client).call(run)


def _portal(client: TestClient):
    portal = client.portal
    assert portal is not None
    return portal


def _wait_for_status(client: TestClient, app: CayuApp, session_id: str) -> None:
    """Settling returns once the decision is durable; the run finishes in the background."""

    async def wait() -> None:
        for _attempt in range(500):
            session = await app.session_store.load(session_id)
            if session is not None and session.status is SessionStatus.COMPLETED:
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"{session_id} did not complete after its decision.")

    _portal(client).call(wait)


def _items(client: TestClient, person: str) -> list[dict]:
    response = client.get("/api/inbox", headers=_as(person))
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    return response.json()["items"]


def _settle(client: TestClient, item_id: str, person: str, **body):
    return client.post(f"/api/inbox/{item_id}/settle", json=body, headers=_as(person))


def _supervision(client: TestClient, person: str = "tom") -> list[dict]:
    now = datetime.now(UTC)
    response = client.post(
        "/api/supervision-events/query",
        json={
            "start_at": (now - timedelta(hours=1)).isoformat(),
            "end_at": (now + timedelta(hours=1)).isoformat(),
        },
        headers=_as(person),
    )
    assert response.status_code == 200, response.text
    return response.json()["events"]


def _durable_app(store_kind: str, tmp_path) -> CayuApp:
    if store_kind == "memory":
        return _app()
    return _app(session_store=SQLiteSessionStore(tmp_path / "inbox.db"))


@pytest.mark.parametrize(
    "agent,body",
    [("approver", {"decision": "approve"}), ("asker", {"decision": "answer", "answer": "staging"})],
)
def test_inbox_settles_worker_handoff_with_exact_continuation_authority(agent, body) -> None:
    tasks = InMemoryTaskStore()
    app = _app(task_store=tasks)

    async def pause(runtime, task, worker_id):
        async for _event in runtime.run(
            RunRequest(
                agent_name=agent,
                session_id="worker-inbox",
                task_id=task.id,
                task_worker_id=worker_id,
                task_lease_expires_at=task.lease_expires_at,
                messages=[Message.text("user", "go")],
            )
        ):
            pass
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    async def handoff():
        await tasks.create_task(
            TaskCreate(task_id="inbox-task", type="job", assigned_agent_name=agent)
        )
        assert (
            await run_task_worker(
                app,
                tasks,
                pause,
                worker_id="original-worker",
                query=TaskQuery(type="job"),
                max_tasks=1,
                reclaim=False,
            )
            == 1
        )
        elected = (
            await tasks.claim_interrupted_task_continuation(
                "elected-worker", TaskQuery(type="job"), handoff_id=str(uuid4())
            )
        ).task
        assert elected is not None and elected.interrupted_handoff_id is not None
        return elected.interrupted_handoff_id

    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        handoff_id = _portal(client).call(handoff)
        [item] = _items(client, "priya")
        # Merely seeing an item never grants or claims the worker's authority.
        assert _settle(client, item["id"], "priya", **body).status_code == 409
        assert (
            _settle(
                client,
                item["id"],
                "priya",
                **body,
                task_worker_id="elected-worker",
                task_handoff_id="stale-handoff",
            ).status_code
            == 409
        )
        assert (
            _settle(
                client,
                item["id"],
                "priya",
                **body,
                task_worker_id="wrong-worker",
                task_handoff_id=handoff_id,
            ).status_code
            == 409
        )
        authority = {"task_worker_id": "elected-worker", "task_handoff_id": handoff_id}
        settled = _settle(client, item["id"], "priya", **body, **authority)
        assert settled.status_code == 200, settled.text
        assert settled.json()["item"]["settlement"]["decided_by"]["subject"] == "priya"
        _wait_for_status(client, app, "worker-inbox")
        task = _portal(client).call(tasks.load_task, "inbox-task")
        assert task is not None and task.status is TaskStatus.COMPLETED
        replay = _settle(client, item["id"], "priya", **body, **authority)
        assert replay.status_code == 200, replay.text
        assert replay.json()["replayed"] is True


def test_knowledge_previews_redact_before_truncation_in_list_and_detail() -> None:
    secret = "preview-secret-canary-token"
    scope = KnowledgeAccessScope.privileged()
    entry = KnowledgeEntry(
        id="preview",
        title=f"Use {secret}",
        text="x" * 490 + secret + " trailing text",
        status=KnowledgeStatus.PENDING,
    )
    store = InMemoryKnowledgeStore([entry], access_scope=scope)
    app = _app(
        knowledge_store=store, knowledge_access_scope=scope, secret_redactor=SecretRedactor(secret)
    )
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        [item] = _items(client, "priya")
        detail = client.get(f"/api/inbox/{item['id']}", headers=_as("priya"))
        assert detail.status_code == 200
        assert detail.json()["item"] == item
        assert item["title"] == f"Use {REDACTED_SECRET}"
        assert item["detail"] == ("x" * 490 + REDACTED_SECRET)[:499] + "…"
        assert secret[:9] not in json.dumps(item)


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_approval_settles_once_and_every_surface_names_the_person(store_kind, tmp_path) -> None:
    app = _durable_app(store_kind, tmp_path)
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, app, "close-september", "approver")

        [item] = _items(client, "priya")
        assert item["kind"] == "approval"
        assert item["decisions"] == ["approve", "deny"]
        assert item["tool_name"] == "record"
        assert item["detail"] == "Approve the September correction package."
        assert item["session_id"] == "close-september"
        [action] = client.get("/api/pending-actions", headers=_as("tom")).json()["actions"]
        assert item["attention_id"] == action["attention_id"]

        settled = _settle(client, item["id"], "priya", decision="approve", reason="Checked it.")
        assert settled.status_code == 200, settled.text
        body = settled.json()
        assert body["replayed"] is False
        assert body["item"]["status"] == "settled"
        assert body["item"]["settlement"]["decision"] == "approve"
        assert body["item"]["settlement"]["decided_by"] == {
            "subject": "priya",
            "tenant": None,
            "source": "http_auth",
        }
        _wait_for_status(client, app, "close-september")

        # Every other place shows the same settled record, naming the person.
        state = client.get(f"/api/inbox/{item['id']}", headers=_as("tom"))
        assert state.status_code == 200
        assert state.json() == body["item"]
        assert _items(client, "tom") == []
        assert client.get("/api/pending-actions", headers=_as("tom")).json()["actions"] == []
        [touch] = _supervision(client)
        assert touch["kind"] == "approve"
        assert touch["attention_id"] == item["attention_id"]
        assert touch["actor"] == {"subject": "priya", "tenant": None, "source": "http_auth"}
        assert touch["reason"] == "Checked it."

        # The same decision again is a replay; anything else is refused with who decided.
        again = _settle(client, item["id"], "priya", decision="approve", reason="Checked it.")
        assert again.status_code == 200, again.text
        assert again.json() == {"replayed": True, "item": body["item"]}
        for person, decision in (("priya", "deny"), ("tom", "approve")):
            refused = _settle(client, item["id"], person, decision=decision)
            assert refused.status_code == 409, refused.text
            detail = refused.json()["detail"]
            assert detail["code"] == "already_settled"
            assert detail["item"]["settlement"]["decided_by"]["subject"] == "priya"


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_question_answer_and_approval_with_changes(store_kind, tmp_path) -> None:
    app = _durable_app(store_kind, tmp_path)
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, app, "anomaly-review", "asker")
        [question] = _items(client, "priya")
        assert question["kind"] == "question"
        assert question["question"] == _QUESTION
        assert question["decisions"] == ["answer"]

        refused = _settle(client, question["id"], "priya", decision="approve")
        assert refused.status_code == 422
        answered = _settle(client, question["id"], "priya", decision="answer", answer="staging")
        assert answered.status_code == 200, answered.text
        assert answered.json()["item"]["settlement"]["decision"] == "answer"
        assert answered.json()["item"]["settlement"]["decided_by"]["subject"] == "priya"
        _wait_for_status(client, app, "anomaly-review")
        replayed = _settle(client, question["id"], "priya", decision="answer", answer="staging")
        assert replayed.status_code == 200 and replayed.json()["replayed"] is True
        changed = _settle(client, question["id"], "priya", decision="answer", answer="production")
        assert changed.status_code == 409
        assert changed.json()["detail"]["code"] == "already_settled"

        _pause(client, app, "booking", "approver")
        [approval] = _items(client, "tom")
        edited = _settle(
            client, approval["id"], "tom", decision="approve", arguments={"note": "corrected"}
        )
        assert edited.status_code == 200, edited.text
        _wait_for_status(client, app, "booking")
        kinds = {touch["session_id"]: touch["kind"] for touch in _supervision(client)}
        assert kinds == {"anomaly-review": "answer", "booking": "edit"}


def test_denial_and_expiry_are_settlements() -> None:
    app = _app(expires_in_seconds=0.01)
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, app, "late-booking", "approver")
        [item] = _items(client, "priya")
        _portal(client).call(asyncio.sleep, 0.05)
        expired = _settle(client, item["id"], "priya", decision="approve")
        assert expired.status_code == 409, expired.text
        detail = expired.json()["detail"]
        assert detail["code"] == "expired"
        assert detail["item"]["settlement"]["decision"] == "expire"
        assert detail["item"]["settlement"]["decided_by"]["subject"] == "cayu:approval-expiry"

    app = _app()
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, app, "denied-booking", "approver")
        [item] = _items(client, "priya")
        denied = _settle(client, item["id"], "priya", decision="deny", reason="Wrong vendor.")
        assert denied.status_code == 200, denied.text
        assert denied.json()["item"]["settlement"]["decision"] == "deny"
        _wait_for_status(client, app, "denied-booking")
        [touch] = _supervision(client)
        assert (touch["kind"], touch["reason"]) == ("deny", "Wrong vendor.")


def test_refused_edit_leaves_the_approval_pending() -> None:
    app = _app()
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, app, "booking", "approver")
        [item] = _items(client, "priya")
        assert (
            _settle(client, item["id"], "priya", decision="deny", arguments={"note": "x"})
        ).status_code == 422
        refused = _settle(client, item["id"], "priya", decision="approve", arguments={"memo": "x"})
        assert refused.status_code == 409, refused.text
        detail = refused.json()["detail"]
        assert detail["code"] == "conflict"
        assert detail["item"]["status"] == "pending"
        approved = _settle(client, item["id"], "priya", decision="approve")
        assert approved.status_code == 200, approved.text


def test_decision_in_the_developer_dashboard_shows_settled_in_the_inbox() -> None:
    app = _app()
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, app, "dashboard-decided", "approver")
        [item] = _items(client, "priya")
        [action] = client.get("/api/pending-actions", headers=_as("tom")).json()["actions"]
        resolved = client.post(
            "/api/tool-approvals/resolve",
            json={
                "session_id": action["session"]["id"],
                "approval_id": action["approval_id"],
                "tool_round_id": action["round_id"],
                "tool_call_id": action["tool_call_id"],
                "decision": "approve",
            },
            headers=_as("tom"),
        )
        assert resolved.status_code == 200, resolved.text

        state = client.get(f"/api/inbox/{item['id']}", headers=_as("priya")).json()
        assert state["status"] == "settled"
        assert state["settlement"]["decided_by"]["subject"] == "tom"
        refused = _settle(client, item["id"], "priya", decision="approve")
        assert refused.status_code == 409
        assert refused.json()["detail"]["item"] == state


async def _own_tenant_and_not_tom(person: AuthContext, item: InboxItemScope) -> bool:
    """A rule keeping each tenant to the sessions started under it, and Tom to nothing."""

    return person.subject != "tom" and (
        person.tenant is None or item.origin_tenant == person.tenant
    )


def test_inbox_requires_auth_and_an_access_rule_hides_items() -> None:
    app = _app()
    server = create_server(
        app, config=ServerConfig.protected(_authenticate), inbox_access=_own_tenant_and_not_tom
    )
    with TestClient(server) as client:
        _pause(client, app, "acme-run", "approver", tenant="acme")
        assert client.get("/api/inbox").status_code == 401

        [item] = _items(client, "acme")
        assert _items(client, "globex") == []
        assert _items(client, "tom") == []
        assert [listed["id"] for listed in _items(client, "priya")] == [item["id"]]
        for person in ("globex", "tom"):
            assert client.get(f"/api/inbox/{item['id']}", headers=_as(person)).status_code == 404
            assert _settle(client, item["id"], person, decision="deny").status_code == 404

        forged = item["id"][:-4] + "AAAA"
        assert client.get(f"/api/inbox/{forged}", headers=_as("priya")).status_code == 404
        assert client.get("/api/inbox/not-an-item", headers=_as("priya")).status_code == 422
        assert client.get("/api/inbox?cursor=junk", headers=_as("priya")).status_code == 422

    with pytest.raises(ValueError, match="inbox access rule"):
        create_server(
            app, config=ServerConfig.local_development(), inbox_access=_own_tenant_and_not_tom
        )


def _reference_params(app: CayuApp, client: TestClient, session_id: str) -> tuple[str, dict]:
    async def read() -> HumanAttentionReference:
        page = await app.session_store.query_pending_actions(
            PendingActionQuery(session_id=session_id)
        )
        request = HumanAttentionRequest.from_pending_action(page.actions[0])
        assert request is not None
        return request.reference

    reference = _portal(client).call(read)
    params = reference.model_dump(mode="json", exclude={"attention_id"}, exclude_none=True)
    return reference.attention_id, params


def test_the_access_rule_refuses_a_hidden_caller_on_every_route() -> None:
    app, _store = _knowledge_app()
    server = create_server(
        app, config=ServerConfig.protected(_authenticate), inbox_access=_own_tenant_and_not_tom
    )
    with TestClient(server) as client:
        _pause(client, app, "close", "approver")
        _pause(client, app, "ask", "asker")
        actions = client.get("/api/pending-actions", headers=_as("priya")).json()["actions"]
        approval = next(action for action in actions if action["kind"] == "tool_approval")
        question = next(action for action in actions if action["kind"] == "user_input")
        approval_ids = {
            "session_id": approval["session"]["id"],
            "approval_id": approval["approval_id"],
            "tool_round_id": approval["round_id"],
            "tool_call_id": approval["tool_call_id"],
        }
        question_ids = {
            "session_id": question["session"]["id"],
            "input_id": question["input_id"],
        }
        recovery = {"outcome": "completed", "message": "Checked in the mailbox."}
        [inbox_item, *_] = _items(client, "priya")
        [entry, *_] = client.get("/api/knowledge/pending", headers=_as("priya")).json()["entries"]
        attention_id, attention_params = _reference_params(app, client, "close")

        tom = _as("tom")
        assert _items(client, "tom") == []
        assert client.get(f"/api/inbox/{inbox_item['id']}", headers=tom).status_code == 404
        assert _settle(client, inbox_item["id"], "tom", decision="approve").status_code == 404
        listed = client.get("/api/pending-actions", headers=tom).json()
        assert (listed["actions"], listed["issues"], listed["total_count"]) == ([], [], None)
        refused = {
            "/api/tool-approvals/resolve": {**approval_ids, "decision": "approve"},
            "/api/tool-approvals/recover": {**approval_ids, **recovery},
            "/api/tool-rounds/recover": {
                "session_id": approval_ids["session_id"],
                "round_id": approval_ids["tool_round_id"],
                "tool_call_id": approval_ids["tool_call_id"],
                **recovery,
            },
            "/api/user-input/resolve": {**question_ids, "answer": "staging"},
            "/api/user-input/recover": {
                **question_ids,
                "answer": "staging",
                "tool_call_id": question["tool_call_id"],
                **recovery,
            },
        }
        for path, body in refused.items():
            assert client.post(path, json=body, headers=tom).status_code == 404, path
        attention = client.get(
            f"/api/human-attention/{attention_id}", params=attention_params, headers=tom
        )
        assert attention.json()["state"] == "unavailable"
        assert client.get("/api/knowledge/pending", headers=tom).json()["entries"] == []
        for request in (
            client.get(f"/api/knowledge/pending/{entry['entry_id']}", headers=tom),
            client.post(f"/api/knowledge/{entry['entry_id']}/approve", headers=tom),
            client.post(f"/api/knowledge/{entry['entry_id']}/reject", headers=tom),
        ):
            assert request.status_code == 404

        # Nothing Tom sent settled anything; Priya, whom the rule allows, still can.
        assert len(client.get("/api/pending-actions", headers=_as("priya")).json()["actions"]) == 2
        resolved = client.post(
            "/api/tool-approvals/resolve",
            json={**approval_ids, "decision": "approve"},
            headers=_as("priya"),
        )
        assert resolved.status_code == 200, resolved.text
        _wait_for_status(client, app, "close")
        assert [touch["kind"] for touch in _supervision(client, "priya")] == ["approve"]
        assert _supervision(client, "tom") == []


def test_the_access_rule_guards_human_review() -> None:
    class _Review(HumanReviewPolicy):
        version = "inbox-review-v1"

        @property
        def binding_key(self) -> bytes:
            return b"k" * 32

        def authorize(self, context, *, session_id, session_metadata, action) -> bool:
            return context.purpose == "inbox"

        def project(self, context, source) -> HumanReviewDisclosure:
            return HumanReviewDisclosure(
                status="permitted",
                sensitive_content="application_attested",
                fields=(HumanReviewField(label="Note", text="Record the note."),),
            )

    scopes: list[InboxItemScope] = []

    async def access(person, scope):
        scopes.append(scope)
        return scope.attention_id is not None and await _own_tenant_and_not_tom(person, scope)

    app = _app(human_review_policy=_Review())
    server = create_server(app, config=ServerConfig.protected(_authenticate), inbox_access=access)
    with TestClient(server) as client:
        _pause(client, app, "reviewed", "approver")
        [item] = _items(client, "priya")
        expected = scopes[-1]
        assert expected.attention_id == item["attention_id"]
        for person, status in (("tom", 403), ("priya", 200)):
            review = client.get(
                "/api/sessions/reviewed/human-review",
                params={"purpose": "inbox"},
                headers=_as(person),
            )
            assert review.status_code == status, review.text
        assert all(scope == expected for scope in scopes)


def test_human_review_policy_requires_its_reference() -> None:
    class _Review(HumanReviewPolicy):
        version = "inbox-review-v1"

        @property
        def binding_key(self) -> bytes:
            return b"k" * 32

        def authorize(self, context, *, session_id, session_metadata, action) -> bool:
            return context.recipient == "priya" and context.purpose == "inbox"

        def project(self, context, source) -> HumanReviewDisclosure:
            return HumanReviewDisclosure(
                status="permitted",
                sensitive_content="application_attested",
                fields=(HumanReviewField(label="Note", text="Record the note."),),
            )

    app = _app(human_review_policy=_Review())
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, app, "reviewed", "approver")
        [item] = _items(client, "priya")
        assert _settle(client, item["id"], "priya", decision="approve").status_code == 403

        review = client.get(
            "/api/sessions/reviewed/human-review", params={"purpose": "inbox"}, headers=_as("priya")
        )
        assert review.status_code == 200, review.text
        reference = review.json()["reference"]
        stolen = _settle(client, item["id"], "tom", decision="approve", review_reference=reference)
        assert stolen.status_code == 403
        approved = _settle(
            client, item["id"], "priya", decision="approve", review_reference=reference
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["item"]["settlement"]["decided_by"]["subject"] == "priya"


def _knowledge_app() -> tuple[CayuApp, InMemoryKnowledgeStore]:
    scope = KnowledgeAccessScope.privileged()
    entries = [
        KnowledgeEntry(
            id=entry_id,
            title=title,
            text=f"{title} from the September close.",
            status=KnowledgeStatus.PENDING,
            created_by_type=KnowledgeActorType.MODEL,
            created_by="close-agent",
        )
        for entry_id, title in (
            ("vendor-codes", "Brightline travel codes to 6410"),
            ("freight-accruals", "Norfreight accruals go to 2130"),
        )
    ]
    store = InMemoryKnowledgeStore(entries, access_scope=scope)
    return _app(knowledge_store=store, knowledge_access_scope=scope), store


def test_knowledge_proposals_are_kept_or_dropped_once() -> None:
    app, store = _knowledge_app()
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        items = {item["knowledge"]["entry_id"]: item for item in _items(client, "priya")}
        assert set(items) == {"vendor-codes", "freight-accruals"}
        assert all(item["decisions"] == ["keep", "drop"] for item in items.values())

        keep = items["vendor-codes"]["id"]
        kept = _settle(client, keep, "priya", decision="keep")
        assert kept.status_code == 200, kept.text
        settlement = kept.json()["item"]["settlement"]
        assert settlement["decision"] == "keep"
        assert settlement["decided_by"] == {
            "subject": "priya",
            "tenant": None,
            "source": "http_auth",
        }
        assert _settle(client, keep, "priya", decision="keep").json()["replayed"] is True
        other = _settle(client, keep, "tom", decision="keep")
        assert other.status_code == 409
        assert other.json()["detail"]["code"] == "already_settled"

        drop = items["freight-accruals"]["id"]
        dropped = _settle(client, drop, "tom", decision="drop")
        assert dropped.status_code == 200, dropped.text
        assert dropped.json()["item"]["settlement"]["decision"] == "drop"
        assert _settle(client, drop, "tom", decision="keep").status_code == 409
        assert _items(client, "priya") == []

    async def statuses() -> dict[str, KnowledgeStatus]:
        scope = KnowledgeAccessScope.privileged()
        return {
            entry_id: (await store.get_entry(entry_id, access_scope=scope)).status
            for entry_id in ("vendor-codes", "freight-accruals")
        }

    assert asyncio.run(statuses()) == {
        "vendor-codes": KnowledgeStatus.ACTIVE,
        "freight-accruals": KnowledgeStatus.ARCHIVED,
    }


def test_revised_knowledge_proposal_is_a_new_item() -> None:
    app, store = _knowledge_app()
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        [stale] = [
            item
            for item in _items(client, "priya")
            if item["knowledge"]["entry_id"] == "vendor-codes"
        ]

        async def revise() -> None:
            scope = KnowledgeAccessScope.privileged()
            entry = await store.get_entry("vendor-codes", access_scope=scope)
            assert entry is not None
            await store.append_entry_revision(
                entry.model_copy(
                    update={"revision": entry.revision + 1, "text": "Travel codes to 6420."}
                ),
                expected_revision=entry.revision,
                access_scope=scope,
            )

        _portal(client).call(revise)
        refused = _settle(client, stale["id"], "priya", decision="keep")
        assert refused.status_code == 409, refused.text
        assert refused.json()["detail"]["code"] == "superseded"
        [fresh] = [
            item
            for item in _items(client, "priya")
            if item["knowledge"]["entry_id"] == "vendor-codes"
        ]
        assert fresh["id"] != stale["id"]
        assert fresh["knowledge"]["revision"] == stale["knowledge"]["revision"] + 1

        # Dropping the revision doesn't settle the one it replaced.
        assert _settle(client, fresh["id"], "tom", decision="drop").status_code == 200
        state = client.get(f"/api/inbox/{stale['id']}", headers=_as("priya")).json()
        assert state["status"] == "superseded"
        stale_drop = _settle(client, stale["id"], "priya", decision="drop")
        assert stale_drop.status_code == 409
        assert stale_drop.json()["detail"]["code"] == "superseded"


def _cursor_text(cursor: str) -> str:
    """Every layer of an inbox cursor, decoded, so a test can see what it reveals."""

    def unwrap(value: str) -> str:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode("utf-8")

    layers = [unwrap(cursor.removeprefix("inboxcursor_"))]
    inner = json.loads(layers[0])[2]
    if inner is not None:
        layers.append(unwrap(inner))
        layers.append(unwrap(json.loads(layers[1])["session_id_b64"]))
    return "\n".join(layers)


def test_scoped_listing_pages_from_the_callers_own_items() -> None:
    app = _app()
    server = create_server(
        app, config=ServerConfig.protected(_authenticate), inbox_access=_own_tenant_and_not_tom
    )
    with TestClient(server) as client:
        _pause(client, app, "acme-oldest", "approver", tenant="acme")
        for index in range(11):
            _pause(client, app, f"globex-{index}", "approver", tenant="globex")
        _pause(client, app, "acme-newest", "asker", tenant="acme")

        # Pages for a tenant continue after its own items and never point into
        # another tenant's sessions.
        seen: list[str] = []
        cursor = None
        for _page in range(5):
            params = {"limit": 1} if cursor is None else {"limit": 1, "cursor": cursor}
            page = client.get("/api/inbox", params=params, headers=_as("acme")).json()
            seen.extend(item["session_id"] for item in page["items"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
            assert "globex" not in _cursor_text(cursor)
        assert seen == ["acme-newest"]
        # Ten pages of other people's items end the scan with an issue, not a
        # position among them; a bigger page reaches the older item.
        assert [issue["code"] for issue in page["issues"]] == ["inbox_truncated"]
        assert page["next_cursor"] is None
        wide = client.get("/api/inbox", params={"limit": 20}, headers=_as("acme")).json()
        assert [item["session_id"] for item in wide["items"]] == ["acme-newest", "acme-oldest"]
        assert wide["issues"] == []


def test_knowledge_proposals_page_after_the_session_items() -> None:
    async def anyone(person: AuthContext, item: InboxItemScope) -> bool:
        return True

    for inbox_access in (None, anyone):
        scope = KnowledgeAccessScope.privileged()
        store = InMemoryKnowledgeStore(
            [
                KnowledgeEntry(
                    id=f"proposal-{index}",
                    title=f"Proposal {index}",
                    text=f"Proposal {index} from the September close.",
                    status=KnowledgeStatus.PENDING,
                    created_by_type=KnowledgeActorType.MODEL,
                    created_by="close-agent",
                )
                for index in range(3)
            ],
            access_scope=scope,
        )
        app = _app(knowledge_store=store, knowledge_access_scope=scope)
        server = create_server(
            app,
            config=ServerConfig.protected(_authenticate),
            inbox_access=inbox_access,
            inbox_cursor_key=b"K" * 32,
        )
        with TestClient(server) as client:
            _pause(client, app, "close", "approver")
            pages = []
            cursor = None
            while True:
                params = {"limit": 2} if cursor is None else {"limit": 2, "cursor": cursor}
                page = client.get("/api/inbox", params=params, headers=_as("priya")).json()
                pages.append(
                    ([item["kind"] for item in page["items"]], [i["code"] for i in page["issues"]])
                )
                cursor = page["next_cursor"]
                if cursor is None:
                    break
            # The approval ends its page; proposals get independent bounded pages.
            assert pages == [
                (["approval"], []),
                (["knowledge", "knowledge"], []),
                (["knowledge"], []),
            ], inbox_access


def test_knowledge_pages_advance_past_hidden_entries_without_exposing_them(monkeypatch) -> None:
    import cayu.server._inbox_routes as inbox_routes

    monkeypatch.setattr(inbox_routes, "_KNOWLEDGE_SCAN_PAGES", 1)
    scope = KnowledgeAccessScope.privileged()
    entries = [
        KnowledgeEntry(
            id=f"hidden-{index}",
            title="hidden title",
            text="hidden text",
            importance=1,
            status=KnowledgeStatus.PENDING,
        )
        for index in range(201)
    ]
    entries.extend(
        KnowledgeEntry(
            id=f"visible-{index}",
            text="Visible proposal",
            importance=0,
            status=KnowledgeStatus.PENDING,
        )
        for index in range(3)
    )
    store = InMemoryKnowledgeStore(entries, access_scope=scope)
    app = _app(knowledge_store=store, knowledge_access_scope=scope)
    allowed = True

    async def access(person, item):
        return (
            allowed
            and item.knowledge is not None
            and item.knowledge.entry_id.startswith("visible-")
        )

    server = create_server(
        app,
        config=ServerConfig.protected(_authenticate),
        inbox_access=access,
        inbox_cursor_key=b"K" * 32,
    )
    with TestClient(server) as client:
        first = client.get("/api/inbox", params={"limit": 1}, headers=_as("acme"))
        assert first.status_code == 200, first.text
        assert first.json()["items"] == []
        assert first.json()["has_more"] is True
        cursor = first.json()["next_cursor"]
        outer = base64.urlsafe_b64decode(
            cursor.removeprefix("inboxcursor_")
            + "=" * (-len(cursor.removeprefix("inboxcursor_")) % 4)
        )
        encrypted = json.loads(outer)[2]
        raw = base64.urlsafe_b64decode(encrypted + "=" * (-len(encrypted) % 4))
        assert b"hidden" not in outer + raw
        assert "hidden" not in first.text
        for person in ("globex", "priya"):
            denied = client.get("/api/inbox", params={"cursor": cursor}, headers=_as(person))
            assert denied.status_code == 422
            assert (
                denied.json()["detail"]
                == "Invalid or expired inbox cursor. Restart the inbox listing."
            )
        # Authorization is rerun after decoding; a valid token never freezes access.
        allowed = False
        revoked = client.get("/api/inbox", params={"cursor": cursor}, headers=_as("acme"))
        assert revoked.status_code == 200 and revoked.json()["items"] == []
        allowed = True
        seen = []
        while cursor is not None:
            response = client.get(
                "/api/inbox", params={"limit": 1, "cursor": cursor}, headers=_as("acme")
            )
            assert response.status_code == 200, response.text
            page = response.json()
            for item in page["items"]:
                seen.append(item["knowledge"]["entry_id"])
                assert _settle(client, item["id"], "acme", decision="drop").status_code == 200
            cursor = page["next_cursor"]
        # Removing each preceding proposal cannot shift the following page past it.
        assert sorted(seen) == ["visible-0", "visible-1", "visible-2"]


def test_knowledge_cursor_requires_configured_key_and_binds_query_identity_and_lifetime(
    monkeypatch,
) -> None:
    import cayu.server._inbox_cursor as cursor_module
    from cayu.knowledge.search import encode_knowledge_list_cursor

    position = encode_knowledge_list_cursor(KnowledgeEntry(id="hidden", text="Private"))
    binding = {"subject": "same-person", "tenant": "one", "query": {"namespace": "one"}}
    clock = 1000
    monkeypatch.setattr(cursor_module.time, "time", lambda: clock)
    key = b"K" * 32
    codec = cursor_module.KnowledgeInboxCursor(key)
    token = codec.encode(position, binding)
    assert cursor_module.KnowledgeInboxCursor(key).decode(token, binding) == position
    # The next page can reach another host whose clock trails the issuer.
    other_worker = cursor_module.KnowledgeInboxCursor(key)
    for skew in (0.05, 1, 60):
        clock = 1000 - skew
        assert other_worker.decode(token, binding) == position
    clock = 1000 - 60.001
    with pytest.raises(HTTPException, match="Invalid or expired inbox cursor"):
        other_worker.decode(token, binding)
    clock = 1000
    for changed in (
        {**binding, "subject": "another-person"},
        {**binding, "tenant": "two"},
        {**binding, "query": {"namespace": "two"}},
    ):
        with pytest.raises(HTTPException, match="Invalid or expired inbox cursor"):
            codec.decode(token, changed)
    for bad in ("malformed", token[:-2] + "AA"):
        with pytest.raises(HTTPException, match="Invalid or expired inbox cursor"):
            codec.decode(bad, binding)
    for rotated in (None, b"R" * 32):
        with pytest.raises(HTTPException, match="Invalid or expired inbox cursor"):
            cursor_module.KnowledgeInboxCursor(rotated).decode(token, binding)
    clock += 900
    assert other_worker.decode(token, binding) == position
    clock += 0.001
    with pytest.raises(HTTPException, match="Invalid or expired inbox cursor"):
        codec.decode(token, binding)
    with pytest.raises(HTTPException, match="configured inbox_cursor_key"):
        cursor_module.KnowledgeInboxCursor(None).encode(position, binding)


def test_settled_sibling_recovery_keeps_the_crashed_call() -> None:
    """A sibling that crashed in an ask_user round is named by its own call, not the question's."""

    app = _app()
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        _pause(client, app, "ask", "asker")
        session = _portal(client).call(app.session_store.load, "ask")
        assert session is not None and session.instance_id is not None
        payload = {
            "interruption_type": "manual_recovery_required",
            "manual_recovery_required": True,
            "tool_round_id": "r1",
            "tool_call_id": "sibling",
            "user_input": {"input_id": "i1", "tool_call_id": "askcall"},
        }
        record = EventRecord(
            sequence=40,
            event=Event(type=EventType.SESSION_INTERRUPTED, session_id="ask", payload=payload),
        )
        attention_id = human_attention_id(
            session_id=session.id,
            session_instance_id=session.instance_id,
            kind=PendingActionKind.MANUAL_RECOVERY,
            discriminator="r1",
            tool_call_id="sibling",
        )
        reference = _event_reference("recovery", session, record, attention_id)
        assert reference is not None
        assert (reference.action_id, reference.round_id, reference.tool_call_id) == (
            "r1",
            "r1",
            "sibling",
        )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_assigned_sibling_recovery_keeps_scope_and_decider(store_kind, tmp_path) -> None:
    """The same exact item is allowed before recovery, on retry, and after settlement."""

    app = CayuApp(
        enable_logging=False,
        **(
            {"session_store": SQLiteSessionStore(tmp_path / "recovery.db")}
            if store_kind == "sqlite"
            else {}
        ),
    )
    tool = _CountingTool()
    app.register_provider(
        _ScriptedProvider(
            [
                ("sibling", "count", {}),
                ("question", "ask_user", {"question": _QUESTION}),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="asker", model="fake-model"), tools=[tool, UserInputTool()])
    assigned: set[str] = set()
    seen: list[InboxItemScope] = []

    async def access(person, scope):
        seen.append(scope)
        return person.subject == "priya" and scope.attention_id in assigned

    with TestClient(
        create_server(app, config=ServerConfig.protected(_authenticate), inbox_access=access)
    ) as client:

        async def crash():
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="asker",
                        session_id="crashed",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            awaiting = next(
                event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            await app.session_store.append_events(
                "crashed",
                _crashed_user_input_resume_events(
                    await private_events_for_public_events(app.session_store, paused),
                    session_id="crashed",
                    tool_call_id="sibling",
                ),
            )
            stopped = [
                event
                async for event in app.resolve_user_input(
                    UserInputResponse(
                        session_id="crashed",
                        input_id=awaiting.payload["input_id"],
                        answer="staging",
                        resolved_by=ResolutionActor(
                            subject="priya", source=ResolutionActorSource.HTTP_AUTH
                        ),
                    )
                )
            ]
            assert any(event.payload.get("manual_recovery_required") for event in stopped)
            page = await app.session_store.query_pending_actions(
                PendingActionQuery(session_id="crashed")
            )
            [action] = page.actions
            assert action.kind is PendingActionKind.MANUAL_RECOVERY
            records = await app.session_store.query_events(EventQuery(session_id="crashed"))
            started = next(
                record
                for record in records
                if record.event.type is EventType.TOOL_CALL_STARTED
                and record.event.payload.get("tool_call_id") == "sibling"
            )
            return action, public_event_linkage_id(started.sequence, "tool_call_id")

        action, sibling_alias = _portal(client).call(crash)
        assigned.add(action.attention_id)
        [item] = _items(client, "priya")
        expected_scope = seen[-1]
        assert expected_scope.attention_id == item["attention_id"]
        seen.clear()
        [public_action] = client.get("/api/pending-actions", headers=_as("priya")).json()["actions"]
        attention_id, attention_params = _reference_params(app, client, "crashed")
        assert (
            client.get(
                f"/api/human-attention/{attention_id}",
                params=attention_params,
                headers=_as("priya"),
            ).json()["state"]
            == "active"
        )
        body = {
            "session_id": public_action["session"]["id"],
            "input_id": public_action["input_id"],
            "answer": "staging",
            # The handoff also names ask_user's call, so use the sibling's
            # unambiguous tool-start alias for this exact recovery request.
            "tool_call_id": sibling_alias,
            "outcome": "completed",
            "message": "Verified externally.",
        }
        denied = client.post("/api/user-input/recover", headers=_as("tom"), json=body)
        assert denied.status_code == 404, denied.text
        recovered = client.post("/api/user-input/recover", headers=_as("priya"), json=body)
        assert recovered.status_code == 200, recovered.text
        _wait_for_status(client, app, "crashed")
        assert tool.calls == 0
        state = client.get(f"/api/inbox/{item['id']}", headers=_as("priya")).json()
        assert state["status"] == "settled"
        assert state["settlement"]["decision"] == "recover"
        assert state["settlement"]["decided_by"] == {
            "subject": "priya",
            "tenant": None,
            "source": "http_auth",
        }
        events = _portal(client).call(app.session_store.load_events, "crashed")
        terminal = next(
            event
            for event in events
            if event.type is EventType.TOOL_CALL_COMPLETED
            and event.payload.get("manual_recovery") is True
        )
        assert datetime.fromisoformat(state["settlement"]["decided_at"]) == terminal.timestamp
        replay = client.post("/api/user-input/recover", headers=_as("priya"), json=body)
        assert replay.status_code == 200, replay.text
        assert (
            client.get(
                f"/api/human-attention/{attention_id}",
                params=attention_params,
                headers=_as("priya"),
            ).json()["state"]
            == "resolved"
        )
        assert [touch["kind"] for touch in _supervision(client, "priya")] == ["recover"]
        recovery_scopes = [scope for scope in seen if scope.kind == "recovery"]
        assert recovery_scopes and all(scope == expected_scope for scope in recovery_scopes)
        assert all(scope.attention_id is not None for scope in seen)


@pytest.mark.parametrize("agent,kind", [("approver", "approval"), ("asker", "question")])
def test_item_scoped_rule_agrees_across_decision_routes(agent, kind) -> None:
    app = _app()
    scopes: list[InboxItemScope] = []

    async def access(person, scope):
        scopes.append(scope)
        return scope.attention_id is not None and scope.kind == kind and person.subject == "priya"

    with TestClient(
        create_server(app, config=ServerConfig.protected(_authenticate), inbox_access=access)
    ) as client:
        _pause(client, app, "assigned", agent)
        [item] = _items(client, "priya")
        expected = scopes[-1]
        [action] = client.get("/api/pending-actions", headers=_as("priya")).json()["actions"]
        attention_id, params = _reference_params(app, client, "assigned")
        assert (
            client.get(
                f"/api/human-attention/{attention_id}", params=params, headers=_as("priya")
            ).json()["state"]
            == "active"
        )
        if kind == "approval":
            path = "/api/tool-approvals/resolve"
            body = {
                "session_id": "assigned",
                "approval_id": action["approval_id"],
                "tool_round_id": action["round_id"],
                "tool_call_id": action["tool_call_id"],
                "decision": "approve",
            }
        else:
            path = "/api/user-input/resolve"
            body = {"session_id": "assigned", "input_id": action["input_id"], "answer": "staging"}
        resolved = client.post(path, json=body, headers=_as("priya"))
        assert resolved.status_code == 200, resolved.text
        _wait_for_status(client, app, "assigned")
        assert (
            client.get(f"/api/inbox/{item['id']}", headers=_as("priya")).json()["status"]
            == "settled"
        )
        replay = client.post(path, json=body, headers=_as("priya"))
        assert replay.status_code == 200, replay.text
        assert _supervision(client, "priya")
        assert scopes and all(scope == expected for scope in scopes)
