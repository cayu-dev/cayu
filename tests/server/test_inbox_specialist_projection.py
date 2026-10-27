"""Public Inbox names never replace private settlement identities."""

import asyncio
import base64

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from tests.server.test_server_inbox import (
    _app,
    _ApproveNotes,
    _as,
    _authenticate,
    _items,
    _NoteTool,
    _pause,
    _portal,
    _respond,
    _settle,
    _wait_for_status,
)

from cayu import AgentSpec, CayuApp
from cayu.evals import ScriptedModelProvider
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.server import ServerConfig, create_server
from cayu.server.config import DashboardConfig
from cayu.sessions.memory import InMemorySessionStore
from cayu.storage import SQLiteSessionStore
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_inbox_current_specialist_projection_preserves_exact_settlement(backend, tmp_path):
    secrets = ("specialist-secret-one", "specialist-secret-two")
    names = [f"Reviewer-{secret}" for secret in secrets]
    codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={
                "test": SecretStr(base64.urlsafe_b64encode(bytes(range(32))).decode().rstrip("="))
            },
        )
    )

    def open_store():
        return (
            InMemorySessionStore(public_authority_alias_codec=codec)
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "names.db", public_authority_alias_codec=codec)
        )

    store = open_store()

    identity = ExecutionProfileBehaviorIdentity(
        name="inbox-projection-fixture", behavior_version="1", implementation_version="1"
    )

    class Provider(ScriptedModelProvider):
        @property
        def execution_profile_identity(self):
            return identity

    class Policy(_ApproveNotes):
        @property
        def execution_profile_identity(self):
            return identity

    class Note(_NoteTool):
        spec = _NoteTool.spec.model_copy(update={"execution_profile_identity": identity})

    def application(store, redactor=None):
        app = CayuApp(session_store=store, secret_redactor=redactor, enable_logging=False)
        app.register_provider(Provider(response_factory=_respond), default=True)
        for name in names:
            app.register_agent(
                AgentSpec(name=name, model="approve"),
                tools=[Note()],
                tool_policy=Policy(),
            )
        return app

    writer = application(store)
    config = ServerConfig.protected(_authenticate, dashboard=DashboardConfig(enabled=False))
    with TestClient(create_server(writer, config=config)) as client:
        for index, name in enumerate(names):
            _pause(client, writer, f"private-{index}", name)
        before = {item["session_id"]: item["id"] for item in _items(client, "priya")}
    if backend == "sqlite":
        asyncio.run(store.close())
        store = open_store()
    reader = application(store, SecretRedactor(secrets))
    try:
        with TestClient(create_server(reader, config=config)) as client:
            public_name = f"Reviewer-{REDACTED_SECRET}"
            pending = client.get("/api/pending-actions", headers=_as("priya")).json()
            assert {a["session"]["agent_name"] for a in pending["actions"]} == {public_name}
            listed = _items(client, "priya")
            assert {item["agent_name"] for item in listed} == {public_name}
            assert {item["session_id"]: item["id"] for item in listed} == before
            for query, expected in [(secrets[0], 0), (secrets[1], 0), (public_name, 2)]:
                page = client.get("/api/inbox", params={"q": query}, headers=_as("priya"))
                counts = client.get("/api/inbox/counts", params={"q": query}, headers=_as("priya"))
                assert len(page.json()["items"]) == expected
                assert counts.json()["total_count"] == expected
            for session, item_id in before.items():
                state = client.get(f"/api/inbox/{item_id}", headers=_as("priya")).json()
                assert state["agent_name"] == state["item"]["agent_name"] == public_name
                # Current secrets also fence execution of the old checkpoint. Projection
                # must not weaken that runtime guard or dispatch using the public name.
                refused = _settle(client, item_id, "priya", decision="approve")
                assert refused.status_code == 409, refused.text
                assert refused.json()["detail"]["item"]["agent_name"] == public_name
                private = _portal(client).call(store.load, session)
                assert private.agent_name == names[int(session[-1])]
        trusted_writer = application(store)
        with TestClient(create_server(trusted_writer, config=config)) as client:
            for session, item_id in before.items():
                response = _settle(client, item_id, "priya", decision="approve")
                assert response.status_code == 200, response.text
                _wait_for_status(client, trusted_writer, session)
        reader = application(store, SecretRedactor(secrets))
        with TestClient(create_server(reader, config=config)) as client:
            for item_id in before.values():
                state = client.get(f"/api/inbox/{item_id}", headers=_as("priya")).json()
                assert state["agent_name"] == public_name
                assert state["status"] == "settled"
                again = _settle(client, item_id, "priya", decision="approve")
                assert again.status_code == 200, again.text
                assert again.json()["replayed"] is True
                assert again.json()["item"] == state
            assert _items(client, "priya") == []
    finally:
        if backend == "sqlite":
            asyncio.run(store.close())


def test_inbox_source_issue_uses_public_text_and_preserves_session_alias(monkeypatch):
    from cayu.sessions.pending_action_contracts import PendingActionIssue, PendingActionIssueCode

    secret = "issue-specialist-secret"
    app = _app(secret_redactor=SecretRedactor(secret))
    original = app.session_store.query_pending_actions

    async def with_issue(query, **kwargs):
        page = await original(query, **kwargs)
        action = page.actions[0]
        issue = PendingActionIssue(
            code=PendingActionIssueCode.SOURCE_INVALID,
            session_id=f"issue-{secret}",
            agent_name=f"Reviewer-{secret}",
            status=action.session.status,
            updated_at=action.session.updated_at,
            detail=f"Cannot inspect {secret}",
        )
        return page.model_copy(update={"issues": [issue]})

    with TestClient(
        create_server(
            app,
            config=ServerConfig.protected(_authenticate, dashboard=DashboardConfig(enabled=False)),
        )
    ) as client:
        _pause(client, app, "issue-session", "approver")
        monkeypatch.setattr(app.session_store, "query_pending_actions", with_issue)
        pending = client.get("/api/pending-actions", headers=_as("priya")).json()["issues"][0]
        response = client.get("/api/inbox", headers=_as("priya"))
        assert response.status_code == 200, response.text
        issue = response.json()["issues"][0]
        assert issue["agent_name"] == pending["agent_name"] == f"Reviewer-{REDACTED_SECRET}"
        assert issue["detail"] == pending["detail"] == f"Cannot inspect {REDACTED_SECRET}"
        assert issue["session_id"] == app.project_session_id_for_exposure(f"issue-{secret}")
        assert secret not in response.text
        counts = client.get("/api/inbox/counts", headers=_as("priya")).json()
        assert counts["total_count"] is None and counts["unavailable_reason"] == "source_incomplete"
