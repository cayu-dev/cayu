"""A person's inbox in the resource router, scoped by the resource access policy."""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from tests.core.test_human_review import identity
from tests.server.test_server_inbox import _respond

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
from cayu.evals import ScriptedModelProvider
from cayu.resource_access import ResourceAccessPolicy
from cayu.server import ProductPrincipal, ServerConfig, create_server
from cayu.server._decision_only import _EXECUTION_OVERRIDES
from cayu.server.auth import AuthContext
from cayu.server.inbox import InboxItemScope, InboxSettleRequest, resource_access_rule
from cayu.server.resource_access import create_resource_router
from cayu.server.scoped_inbox import _OPERATOR_FIELDS, InboxRefused
from cayu.sessions._access_rules import SessionAccessRule, SessionAccessScope, SessionAccessSelector
from cayu.tools import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult
from cayu.tools.user_input import UserInputTool
from cayu.vaults.redaction import SecretRedactor

PREFIX = "/app/resources"
#: Apps a served test project built from ``_app`` (``tests/cli/test_serve_inbox_access.py``).
SERVED_APPS: list[CayuApp] = []
_PEOPLE = {"alice": "acme", "bob": "globex", "carol": "acme"}


def _rule(org: str) -> tuple[SessionAccessRule, ...]:
    return (SessionAccessRule(selectors=(SessionAccessSelector(key="org", values=(org,)),)),)


def _member(org: str) -> SessionAccessScope:
    rule = _rule(org)
    return SessionAccessScope(read=rule, create=rule, execute=rule, inspect_state=rule)


class People(ResourceAccessPolicy):
    """alice and bob run conversations in their organizations; carol only reads acme's."""

    authority = "scoped-inbox-test-v1"

    def __init__(self) -> None:
        self.scopes: dict[str, SessionAccessScope] = {
            "alice": _member("acme"),
            "bob": _member("globex"),
            "carol": SessionAccessScope(read=_rule("acme")),
        }

    async def resolve(self, subject: str) -> SessionAccessScope:
        return self.scopes.get(subject, SessionAccessScope())


class _Record(Tool):
    """Records a note; with ``revoke``, alice loses her access while it runs."""

    execution_profile_identity = identity("scoped-inbox-record")
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

    def __init__(self, policy: People) -> None:
        self.policy = policy
        self.revoke = False
        self.ran_with: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.ran_with.append(dict(args))
        if self.revoke:
            self.policy.scopes["alice"] = SessionAccessScope()
        return ToolResult(content="Recorded.")


class _ApproveNotes(ToolPolicy):
    execution_profile_identity = identity("scoped-inbox-approval")

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.tool_name != "record":
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)
        return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL, reason="Approve it.")


class _Review(HumanReviewPolicy):
    version = "scoped-review-v1"

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


class _Provider(ScriptedModelProvider):
    # Stable across restarts, so a reopened app continues what it paused.
    execution_profile_identity = identity("scoped-inbox-provider")


def _app(
    provider: ScriptedModelProvider | None = None, **options: Any
) -> tuple[CayuApp, People, _Record, ScriptedModelProvider]:
    policy = People()
    tool = _Record(policy)
    provider = provider or _Provider(response_factory=_respond)
    app = CayuApp(enable_logging=False, resource_access_policy=policy, **options)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="asker", model="ask"), tools=[UserInputTool()])
    app.register_agent(
        AgentSpec(name="approver", model="approve"), tools=[tool], tool_policy=_ApproveNotes()
    )
    return app, policy, tool, provider


async def _authenticate(request: Request) -> ProductPrincipal:
    person = request.headers.get("x-person", "")
    if person not in _PEOPLE:
        raise HTTPException(status_code=401, detail="Sign-in required.")
    return ProductPrincipal(tenant_id=_PEOPLE[person], subject_id=person)


def _subject(principal: ProductPrincipal) -> str:
    # Each test person belongs to one organization, so their subject alone names them.
    return principal.subject_id


def _client(app: CayuApp, *, inbox: bool = True) -> TestClient:
    server = FastAPI()
    server.include_router(
        create_resource_router(
            app, authenticate=_authenticate, principal_subject=_subject, prefix=PREFIX, inbox=inbox
        )
    )
    return TestClient(server)


def _as(person: str) -> dict[str, str]:
    return {"x-person": person}


def _pause(client: TestClient, app: CayuApp, person: str, session_id: str, agent="approver"):
    async def run() -> None:
        access = await app.access(person)
        async for _event in access.run(
            RunRequest(
                agent_name=agent,
                session_id=session_id,
                messages=[Message.text("user", "go")],
                labels={"org": _PEOPLE[person]},
            )
        ):
            pass

    assert client.portal is not None
    client.portal.call(run)


def _status(client: TestClient, app: CayuApp, session_id: str) -> SessionStatus:
    async def wait() -> SessionStatus:
        for _attempt in range(500):
            session = await app.session_store.load(session_id)
            assert session is not None
            if session.status not in {SessionStatus.PENDING, SessionStatus.RUNNING}:
                return session.status
            await asyncio.sleep(0.01)
        raise AssertionError(f"{session_id} is still running.")

    assert client.portal is not None
    return client.portal.call(wait)


def _items(client: TestClient, person: str, **params: str) -> list[dict]:
    response = client.get(f"{PREFIX}/inbox", headers=_as(person), params=params)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    return response.json()["items"]


def _settle(client: TestClient, item_id: str, person: str, **body: Any):
    return client.post(f"{PREFIX}/inbox/{item_id}/settle", json=body, headers=_as(person))


def test_alice_approves_her_approval_as_herself() -> None:
    app, _policy, _tool, _provider = _app()
    with _client(app) as client:
        _pause(client, app, "alice", "alice-1")
        [item] = _items(client, "alice")
        assert (item["kind"], item["session_id"]) == ("approval", "alice-1")
        counts = client.get(f"{PREFIX}/inbox/counts", headers=_as("alice")).json()
        assert counts["complete"] is True
        assert counts["counts"] == {"approval": 1, "question": 0}

        settled = _settle(client, item["id"], "alice", decision="approve")

        assert settled.status_code == 200, settled.text
        decided_by = settled.json()["item"]["settlement"]["decided_by"]
        assert decided_by == {"subject": "alice", "tenant": "acme", "source": "http_auth"}
        assert _status(client, app, "alice-1") is SessionStatus.COMPLETED
        state = client.get(f"{PREFIX}/inbox/{item['id']}", headers=_as("alice")).json()
        assert state["status"] == "settled"
        # The same decision again is a replay; another one is the /api conflict.
        assert _settle(client, item["id"], "alice", decision="approve").json()["replayed"]
        conflict = _settle(client, item["id"], "alice", decision="deny")
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["code"] == "already_settled"


def test_alice_answers_her_question() -> None:
    app, _policy, _tool, _provider = _app()
    with _client(app) as client:
        _pause(client, app, "alice", "alice-q", agent="asker")
        [item] = _items(client, "alice", kind="question")

        answered = _settle(client, item["id"], "alice", decision="answer", answer="staging")

        assert answered.status_code == 200, answered.text
        assert answered.json()["item"]["settlement"]["decided_by"]["subject"] == "alice"


def test_another_organization_gets_the_same_404_as_a_missing_item() -> None:
    app, _policy, _tool, _provider = _app()
    with _client(app) as client:
        _pause(client, app, "alice", "alice-1")
        [item] = _items(client, "alice")
        missing = "inbox_" + "A" * 40

        assert _items(client, "bob") == []
        for item_id in (item["id"], missing):
            for response in (
                client.get(f"{PREFIX}/inbox/{item_id}", headers=_as("bob")),
                _settle(client, item_id, "bob", decision="approve"),
            ):
                assert response.status_code == 404
                assert response.json() == {"detail": "Resource unavailable."}


def test_a_reader_who_cant_execute_doesnt_see_the_items() -> None:
    app, _policy, _tool, _provider = _app()
    with _client(app) as client:
        _pause(client, app, "alice", "alice-1")
        [item] = _items(client, "alice")

        # carol can read the conversation, but not continue it.
        assert client.get(f"{PREFIX}/sessions/alice-1", headers=_as("carol")).status_code == 200
        assert _items(client, "carol") == []
        assert client.get(f"{PREFIX}/inbox/{item['id']}", headers=_as("carol")).status_code == 404
        counts = client.get(f"{PREFIX}/inbox/counts", headers=_as("carol")).json()
        assert counts["total_count"] == 0


def test_revoking_between_list_and_settle_hides_the_item() -> None:
    app, policy, _tool, _provider = _app()
    with _client(app) as client:
        _pause(client, app, "alice", "alice-1")
        [item] = _items(client, "alice")

        policy.scopes["alice"] = SessionAccessScope(read=_rule("acme"))

        assert _settle(client, item["id"], "alice", decision="approve").status_code == 404
        assert _status(client, app, "alice-1") is SessionStatus.INTERRUPTED


def test_revoking_during_the_continuation_stops_it() -> None:
    app, _policy, tool, provider = _app()
    with _client(app) as client:
        _pause(client, app, "alice", "alice-1")
        [item] = _items(client, "alice")
        tool.revoke = True

        assert _settle(client, item["id"], "alice", decision="approve").status_code == 200

        assert _status(client, app, "alice-1") is not SessionStatus.COMPLETED
        # The continuation ran under alice's stored binding: no model call after the tool.
        assert len(provider.requests) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"decision": "approve", "arguments": {"note": "chosen by the member"}},
        {
            "decision": "approve",
            "approval_revision": {"proposed_digest": "proposed", "approved_digest": "edited"},
        },
        {"decision": "approve", "task_continuation": "claim"},
        {"decision": "approve", "task_worker_id": "worker-1"},
        {
            "decision": "approve",
            "profile_adoption": {"idempotency_key": "adopt-1", "reason": "Upgrade."},
        },
        {"decision": "dismiss", "reason": "Stale."},
        {"decision": "keep"},
        {"decision": "drop"},
    ],
)
def test_operator_decisions_are_refused(body: dict) -> None:
    app, _policy, _tool, _provider = _app()
    with _client(app) as client:
        _pause(client, app, "alice", "alice-1")
        [item] = _items(client, "alice")

        refused = _settle(client, item["id"], "alice", **body)

        assert refused.status_code == 422, refused.text
        assert "operator" in refused.json()["detail"]
        assert _items(client, "alice") == [item]


def test_a_member_cant_change_what_the_call_runs_with() -> None:
    # Every settle field /api keeps from read-and-decide roles is an operator field here.
    overrides = _EXECUTION_OVERRIDES & set(InboxSettleRequest.model_fields)
    assert overrides == {"arguments", "approval_revision"}
    assert overrides <= set(_OPERATOR_FIELDS)
    app, _policy, tool, _provider = _app()
    with _client(app) as client:
        _pause(client, app, "alice", "alice-1")
        [item] = _items(client, "alice")
        proposed = _settle(client, item["id"], "alice", decision="approve", arguments={"note": "x"})
        assert proposed.status_code == 422

        assert _settle(client, item["id"], "alice", decision="approve").status_code == 200
        assert _status(client, app, "alice-1") is SessionStatus.COMPLETED
        assert tool.ran_with == [{"note": "proposed"}]


def test_recovery_and_knowledge_items_are_never_listed() -> None:
    app, _policy, _tool, _provider = _app()
    with _client(app) as client:
        _pause(client, app, "alice", "alice-1")
        for kind in ("recovery", "knowledge"):
            assert _items(client, "alice", kind=kind) == []

    rule = resource_access_rule(app)
    alice = AuthContext(subject="alice", tenant="acme")
    for scope in (
        InboxItemScope(kind="recovery", session_id="alice-1"),
        InboxItemScope(kind="knowledge"),
    ):
        assert asyncio.run(rule(alice, scope)) is False


def test_a_review_reference_for_alice_is_refused_to_bob() -> None:
    app, _policy, _tool, _provider = _app(human_review_policy=_Review())
    with _client(app) as client:
        _pause(client, app, "alice", "alice-1")
        [item] = _items(client, "alice")
        review = client.get(
            f"{PREFIX}/sessions/alice-1/human-review",
            params={"purpose": "inbox"},
            headers=_as("alice"),
        )
        assert review.status_code == 200, review.text
        reference = review.json()["reference"]
        assert reference["context"]["recipient"] == "alice"
        # bob can't read her review, nor use the reference she was issued.
        assert (
            client.get(
                f"{PREFIX}/sessions/alice-1/human-review",
                params={"purpose": "inbox"},
                headers=_as("bob"),
            ).status_code
            == 404
        )
        stolen = client.post(
            f"{PREFIX}/sessions/alice-1/human-review/long-field",
            json={"review_reference": reference, "key": "note"},
            headers=_as("bob"),
        )
        assert stolen.status_code == 403
        # Without the reference the policy requires, alice's approval is refused.
        assert _settle(client, item["id"], "alice", decision="approve").status_code == 403
        approved = _settle(
            client, item["id"], "alice", decision="approve", review_reference=reference
        )
        assert approved.status_code == 200, approved.text


def test_without_inbox_there_are_no_inbox_routes() -> None:
    app, _policy, _tool, _provider = _app()
    plain = create_resource_router(
        app, authenticate=_authenticate, principal_subject=_subject, prefix=PREFIX
    )
    paths = {getattr(route, "path", "") for route in plain.routes}
    assert not any("/inbox" in path or "human-review" in path for path in paths)
    with_inbox = create_resource_router(
        app, authenticate=_authenticate, principal_subject=_subject, prefix=PREFIX, inbox=True
    )
    assert f"{PREFIX}/inbox/{{item_id}}/settle" in {
        getattr(route, "path", "") for route in with_inbox.routes
    }


def test_the_python_inbox_lists_counts_and_settles() -> None:
    async def run() -> None:
        app, _policy, _tool, _provider = _app()
        alice = await app.access("alice")
        async for _event in alice.run(
            RunRequest(
                agent_name="approver",
                session_id="alice-1",
                messages=[Message.text("user", "go")],
                labels={"org": "acme"},
            )
        ):
            pass
        page = await alice.inbox.list()
        [item] = page.items
        assert page.next_cursor is None and not page.has_more
        assert (await alice.inbox.counts()).total_count == 1
        assert (await (await app.access("bob")).inbox.list()).items == []
        with pytest.raises(InboxRefused) as refused:
            await alice.inbox.settle(item.id, InboxSettleRequest(decision="keep"))
        assert refused.value.status_code == 422
        settled = await alice.inbox.settle(
            item.id, InboxSettleRequest(decision="deny", reason="Not now."), tenant="acme"
        )
        assert settled.item.settlement is not None
        assert settled.item.settlement.decided_by is not None
        assert settled.item.settlement.decided_by.subject == "alice"

    asyncio.run(run())


def test_counts_match_the_listing_for_a_search() -> None:
    app, _policy, _tool, _provider = _app()
    with _client(app) as client:
        _pause(client, app, "alice", "invoice-1")
        _pause(client, app, "alice", "payroll-1")
        assert len(_items(client, "alice")) == 2
        [match] = _items(client, "alice", q="invoice")
        assert match["session_id"] == "invoice-1"
        for params in ({"q": "invoice"}, {"q": "invoice", "session_id": "invoice-1"}):
            counts = client.get(f"{PREFIX}/inbox/counts", headers=_as("alice"), params=params)
            assert counts.json()["counts"] == {"approval": 1, "question": 0}, params
            assert counts.json()["complete"] is True
        none = client.get(
            f"{PREFIX}/inbox/counts",
            headers=_as("alice"),
            params={"q": "invoice", "session_id": "payroll-1"},
        )
        assert none.json()["total_count"] == 0


def test_a_filtered_page_keeps_to_its_limit_across_session_pages() -> None:
    async def run() -> None:
        app, _policy, _tool, _provider = _app()
        alice = await app.access("alice")
        # Either listing order puts one approval in the first two sessions and two in
        # the next two.
        for index, agent in enumerate(("asker", "approver", "approver", "approver", "asker")):
            async for _event in alice.run(
                RunRequest(
                    agent_name=agent,
                    session_id=f"alice-{index}",
                    messages=[Message.text("user", "go")],
                    labels={"org": "acme"},
                )
            ):
                pass
        first = await alice.inbox.list(kind="approval", limit=2)
        assert len(first.items) == 2
        assert first.next_cursor is not None
        rest = await alice.inbox.list(kind="approval", limit=2, cursor=first.next_cursor)
        listed = [item.session_id for item in (*first.items, *rest.items)]
        assert sorted(listed) == ["alice-1", "alice-2", "alice-3"]

    asyncio.run(run())


def test_a_session_named_by_its_public_alias_is_read_through_the_router() -> None:
    # A session started before its ID's secret was configured is exposed by an alias.
    secret = "scoped-inbox-secret-canary"
    earlier, *_rest = _app()
    with _client(earlier) as client:
        _pause(client, earlier, "alice", f"alice-{secret}")
    app, _policy, _tool, _provider = _app(
        session_store=earlier.session_store, secret_redactor=SecretRedactor(secret)
    )
    with _client(app) as client:
        [item] = _items(client, "alice")
        alias = item["session_id"]
        assert secret not in alias

        assert _settle(client, item["id"], "alice", decision="approve").status_code == 200
        assert _status(client, app, f"alice-{secret}") is SessionStatus.COMPLETED
        read = client.get(f"{PREFIX}/sessions/{alias}", headers=_as("alice"))
        assert read.status_code == 200, read.text
        assert read.json()["status"] == "completed"
        records = client.get(f"{PREFIX}/sessions/{alias}/records/events", headers=_as("alice"))
        assert records.status_code == 200, records.text
        for person in ("bob", "carol"):
            hidden = client.get(f"{PREFIX}/sessions/{alias}", headers=_as(person))
            assert (person, hidden.status_code) == (person, 404 if person == "bob" else 200)
        unknown = alias[:-4] + ("AAAA" if not alias.endswith("AAAA") else "BBBB")
        assert client.get(f"{PREFIX}/sessions/{unknown}", headers=_as("alice")).status_code == 404


def test_the_inbox_cursor_never_carries_a_private_session_id() -> None:
    # Two waiting sessions whose IDs hold a secret configured after they started.
    secret = "scoped-inbox-cursor-canary"
    earlier, *_rest = _app()
    with _client(earlier) as client:
        _pause(client, earlier, "alice", f"alice-a-{secret}")
        _pause(client, earlier, "alice", f"alice-b-{secret}")
    app, _policy, _tool, _provider = _app(
        session_store=earlier.session_store, secret_redactor=SecretRedactor(secret)
    )
    with _client(app) as client:
        first = client.get(f"{PREFIX}/inbox", headers=_as("alice"), params={"limit": 1})
        assert first.status_code == 200, first.text
        page = first.json()
        cursor = page["next_cursor"]
        assert page["has_more"] is True and cursor
        assert secret not in cursor
        decoded = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        assert secret.encode() not in decoded
        assert base64.b64encode(f"alice-a-{secret}".encode()) not in decoded
        assert base64.b64encode(f"alice-b-{secret}".encode()) not in decoded

        # The cursor continues alice's listing.
        second = client.get(
            f"{PREFIX}/inbox", headers=_as("alice"), params={"limit": 1, "cursor": cursor}
        )
        assert second.status_code == 200, second.text
        listed = [item["session_id"] for item in page["items"] + second.json()["items"]]
        assert len(set(listed)) == 2 and not any(secret in alias for alias in listed)
        # Nobody else's, and not with other filters.
        for person, params in (("carol", {}), ("bob", {}), ("alice", {"kind": "question"})):
            refused = client.get(
                f"{PREFIX}/inbox", headers=_as(person), params={"cursor": cursor, **params}
            )
            assert (person, refused.status_code) == (person, 422)


_KEY = b"K" * 32


def _operator(_request: Request) -> AuthContext:
    return AuthContext(subject="operator")


def _replica(store: Any, *, key: bytes | None, through_server: bool) -> TestClient:
    """One replica of a deployment: its own app and process key, the shared store."""

    app, *_rest = _app(session_store=store)
    router_key = None if through_server else key
    router = create_resource_router(
        app,
        authenticate=_authenticate,
        principal_subject=_subject,
        prefix=PREFIX,
        inbox=True,
        inbox_cursor_key=router_key,
    )
    if through_server:
        # The key create_server is given reaches the member inbox too.
        server = create_server(app, config=ServerConfig.protected(_operator), inbox_cursor_key=key)
    else:
        server = FastAPI()
    server.include_router(router)
    return TestClient(server)


@pytest.mark.parametrize("through_server", [False, True])
def test_replicas_with_the_same_configured_key_page_each_others_cursors(
    through_server: bool,
) -> None:
    earlier, *_rest = _app()
    with _client(earlier) as client:
        _pause(client, earlier, "alice", "alice-1")
        _pause(client, earlier, "alice", "alice-2")
    store = earlier.session_store
    with (
        _replica(store, key=_KEY, through_server=through_server) as first,
        _replica(store, key=_KEY, through_server=through_server) as second,
        _replica(store, key=None, through_server=through_server) as unconfigured,
    ):
        page = first.get(f"{PREFIX}/inbox", headers=_as("alice"), params={"limit": 1}).json()
        cursor = page["next_cursor"]
        assert cursor

        rest = second.get(
            f"{PREFIX}/inbox", headers=_as("alice"), params={"limit": 1, "cursor": cursor}
        )
        assert rest.status_code == 200, rest.text
        listed = [item["session_id"] for item in page["items"] + rest.json()["items"]]
        assert sorted(listed) == ["alice-1", "alice-2"]
        # A replica without the key can't open it.
        refused = unconfigured.get(
            f"{PREFIX}/inbox", headers=_as("alice"), params={"cursor": cursor}
        )
        assert refused.status_code == 422


def test_resource_access_rule_needs_a_policy() -> None:
    with pytest.raises(ValueError, match="resource_access_policy"):
        resource_access_rule(CayuApp(enable_logging=False))
