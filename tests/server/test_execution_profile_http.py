from __future__ import annotations

import asyncio
import logging

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileIdentity,
)
from cayu.server import ServerConfig, create_server
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.checkpoints import CheckpointCompatibilityError
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult


class _EffectTool(Tool):
    def __init__(self) -> None:
        self.spec = ToolSpec(
            name="side_effect",
            description="Governed effect.",
            input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
            effect=ToolEffect.EXTERNAL,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:http-effect", behavior_version="1", implementation_version="1"
            ),
        )
        super().__init__()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="done")


class _RequireApproval(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)


class _FailingCheckpointStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.checkpoint_failures: dict[str, Exception] = {}

    async def load_checkpoint(self, session_id: str):
        failure = self.checkpoint_failures.get(session_id)
        if failure is not None:
            raise failure
        return await super().load_checkpoint(session_id)


def _paused_client(
    store: InMemorySessionStore | None = None,
) -> tuple[CayuApp, TestClient]:
    store = InMemorySessionStore() if store is None else store
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call-1", name="side_effect", arguments={"value": "x"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            name="fake",
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model", system_prompt="Use the tool."),
        tools=[_EffectTool()],
        tool_policy=_RequireApproval(),
    )

    async def seed() -> None:
        async for _event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="paused",
                messages=[Message.text("user", "go")],
            )
        ):
            pass
        # A session created outside Runtime has no stored profile.
        await store.create(
            RunRequest(
                session_id="legacy",
                agent_name="assistant",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

    asyncio.run(seed())
    return app, TestClient(create_server(app, config=ServerConfig.local_development()))


def test_session_and_pending_action_lists_include_stored_execution_profiles() -> None:
    app, client = _paused_client()
    stored = asyncio.run(app.inspect_session_execution_profiles("paused"))
    assert stored.active_invocation is not None
    assert stored.boundary == "continuation"

    plain = client.get("/api/sessions", params={"order_by": "created_at_asc"})
    assert plain.status_code == 200
    assert plain.json()["execution_profiles"] is None

    listed = client.get(
        "/api/sessions",
        params={"include": ["execution_profile", "usage"], "order_by": "created_at_asc"},
    )
    assert listed.status_code == 200
    body = listed.json()
    assert [row["id"] for row in body["sessions"]] == ["paused", "legacy"]
    paused_profiles, legacy_profiles = body["execution_profiles"]
    assert paused_profiles == stored.model_dump(mode="json")
    assert legacy_profiles == {
        "boundary": None,
        "expected": None,
        "active_invocation": None,
        "issues": [],
    }
    assert len(body["usage"]) == 2
    expected = ExecutionProfileIdentity.model_validate(paused_profiles["expected"])
    assert (
        expected.component(ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION).fingerprint
        is not None
    )
    assert "Use the tool." not in listed.text
    assert "Governed effect." not in listed.text

    actions = client.get("/api/pending-actions")
    assert actions.status_code == 200
    assert actions.json()["execution_profiles"] is None
    included = client.get("/api/pending-actions", params={"include": "execution_profile"})
    assert included.status_code == 200
    payload = included.json()
    assert [action["kind"] for action in payload["actions"]] == ["tool_approval"]
    assert payload["execution_profiles"] == [stored.model_dump(mode="json")]

    assert client.get("/api/pending-actions", params={"include": "usage"}).status_code == 422


def test_one_unreadable_session_does_not_fail_the_listing(caplog) -> None:
    store = _FailingCheckpointStore()
    _app, client = _paused_client(store)
    store.checkpoint_failures = {
        "paused": RuntimeError("backend unavailable"),
        "legacy": CheckpointCompatibilityError(
            reason="checkpoint_schema_version_too_new",
            session_id="legacy",
            observed_version=999,
        ),
    }

    with caplog.at_level(logging.ERROR, logger="cayu.server.routes"):
        listed = client.get(
            "/api/sessions",
            params={"include": "execution_profile", "order_by": "created_at_asc"},
        )
    assert listed.status_code == 200
    body = listed.json()
    assert [row["id"] for row in body["sessions"]] == ["paused", "legacy"]
    assert [item["issues"] for item in body["execution_profiles"]] == [
        ["load_failed"],
        ["checkpoint_incompatible"],
    ]
    for item in body["execution_profiles"]:
        assert (item["boundary"], item["expected"], item["active_invocation"]) == (
            None,
            None,
            None,
        )
    # Only the unexpected failure is logged, with its redacted diagnostic.
    (record,) = [r for r in caplog.records if "Execution-profile inspection" in r.getMessage()]
    assert "backend unavailable" in record.getMessage()


def test_listing_reports_a_session_that_disappeared() -> None:
    store = _FailingCheckpointStore()
    _app, client = _paused_client(store)
    store.checkpoint_failures = {"paused": KeyError("paused")}

    listed = client.get(
        "/api/sessions",
        params={"include": "execution_profile", "order_by": "created_at_asc"},
    )
    assert listed.status_code == 200
    missing, legacy = listed.json()["execution_profiles"]
    assert missing["issues"] == ["session_not_found"]
    assert legacy["issues"] == []

    # A KeyError about anything but the listed session is an ordinary failure.
    store.checkpoint_failures = {"paused": KeyError("internal-record")}
    listed = client.get(
        "/api/sessions",
        params={"include": "execution_profile", "order_by": "created_at_asc"},
    )
    assert listed.json()["execution_profiles"][0]["issues"] == ["load_failed"]
