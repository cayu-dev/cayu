"""Public event views name a workflow step and its child session; other `*_id` keys stay private."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr

from cayu import AgentSpec, CayuApp
from cayu._event_schema import event_payload_policy
from cayu.evals import ScriptedModelProvider
from cayu.events import Event, EventType
from cayu.providers import ModelStreamEvent
from cayu.runtime._event_projection import (
    PRIVATE_EVENT_AUTHORITY,
    project_persisted_runtime_event,
    public_event_envelope_alias,
)
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.server import ServerConfig, create_server
from cayu.storage import SQLiteSessionStore
from cayu.vaults.redaction import SecretRedactor
from cayu.workflows import WorkflowBase, WorkflowSpec, step

_SECRET = "hunter2-workflow-token"


def _codec() -> PublicAuthorityAliasCodec:
    return PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={
                "test": SecretStr(base64.urlsafe_b64encode(bytes(range(32))).decode().rstrip("="))
            },
        )
    )


def _project(event: Event, redactor: SecretRedactor | None = None) -> dict:
    return project_persisted_runtime_event(
        event,
        sequence=7,
        redactor=redactor or SecretRedactor(),
        public_authority_alias_codec=_codec(),
    ).payload


def test_only_workflow_events_expose_step_and_child_session_ids() -> None:
    for event_type in (EventType.WORKFLOW_STEP_STARTED, EventType.WORKFLOW_STEP_COMPLETED):
        policy = event_payload_policy(event_type)
        assert "step_id" in policy.public_authority_keys
        assert policy.session_aliased_authority_keys == {"child_session_id"}
        payload = _project(
            Event(
                type=event_type,
                session_id="run-1",
                payload={"step_id": "plan", "child_session_id": "child-1", "attempt_id": "a1"},
            )
        )
        assert (payload["step_id"], payload["child_session_id"]) == ("plan", "child-1")
        # The workflow attempt fence stays private.
        assert payload["attempt_id"] == PRIVATE_EVENT_AUTHORITY

    # The same keys on other events keep the general rule: `*_id` is private authority.
    delegated = _project(
        Event(
            type=EventType.SESSION_DELEGATED_ACTION_UPDATED,
            session_id="run-1",
            payload={
                "action_id": "act-1",
                "action_kind": "subagent",
                "child_session_id": "child-1",
                "status": "running",
            },
        )
    )
    assert delegated["child_session_id"] != "child-1"
    assert delegated["action_id"] != "act-1"
    for event_type, policy_key in (
        (EventType.SESSION_DELEGATED_ACTION_UPDATED, "child_session_id"),
        (EventType.MODEL_STARTED, "model_step_id"),
    ):
        policy = event_payload_policy(event_type)
        assert not policy.session_aliased_authority_keys
        assert policy_key not in policy.public_authority_keys


def test_a_redacted_child_session_id_becomes_its_public_session_alias() -> None:
    child = f"child-{_SECRET}"
    payload = _project(
        Event(
            type=EventType.WORKFLOW_STEP_COMPLETED,
            session_id="run-1",
            payload={"step_id": "plan", "child_session_id": child},
        ),
        redactor=SecretRedactor(_SECRET),
    )

    assert _SECRET not in str(payload)
    assert payload["child_session_id"] == public_event_envelope_alias(
        child, field_name="session_id", codec=_codec()
    )


class _OneStep(WorkflowBase):
    spec = WorkflowSpec(name="one-step")

    async def run(self, session_id):
        ctx = self.context(session_id)
        yield await ctx.start()
        await step(
            ctx, agent="planner", step_id="plan", prompt="plan", session_id=f"child-{_SECRET}"
        )
        yield await ctx.completed()


def test_agent_names_a_redacted_child_by_the_alias_its_session_api_resolves(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "cayu.db", public_authority_alias_codec=_codec())
    writer = CayuApp(session_store=store, enable_logging=False)
    writer.register_provider(
        ScriptedModelProvider(
            response_factory=lambda request: [
                ModelStreamEvent.text_delta("ok"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    writer.register_agent(AgentSpec(name="planner", model="scripted-model"))

    async def run() -> None:
        async for _ in _OneStep(writer).run("run-1"):
            pass

    asyncio.run(run())
    reader = CayuApp(
        session_store=store, secret_redactor=SecretRedactor(_SECRET), enable_logging=False
    )
    with TestClient(create_server(reader, config=ServerConfig.local_development())) as client:
        events = client.get(
            "/api/sessions/run-1/events", params={"event_type": "workflow.step.completed"}
        ).json()["events"]
        (completed,) = events
        child = completed["payload"]["child_session_id"]
        assert _SECRET not in child and child != PRIVATE_EVENT_AUTHORITY
        assert completed["payload"]["step_id"] == "plan"
        # The same public id the session list shows under the run, and one it resolves.
        listed = client.get("/api/sessions", params={"parent_session_id": "run-1"}).json()
        assert [row["id"] for row in listed["sessions"]] == [child]
        detail = client.get(f"/api/sessions/{child}")
        assert detail.status_code == 200, detail.text
        assert detail.json()["agent_name"] == "planner"
