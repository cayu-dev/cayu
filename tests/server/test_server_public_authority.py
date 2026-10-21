"""HTTP server public authority contracts."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
pytest.importorskip("httpx")

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
from tests.server.server_test_support import (
    _LOCAL_SERVER_CONFIG,
    OneShotProvider,
    _collect_run,
    _price_book_payload,
    _sse_events,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.artifacts.attachments import FileAttachment, FileAttachmentKind
from cayu.events import EVENT_ID_MAX_CHARS, Event, EventType
from cayu.messages import FilePart, Message, MessageRole, ProviderStatePart
from cayu.runtime._event_projection import (
    public_event_envelope_alias_field,
    public_event_linkage_id,
)
from cayu.server import create_server
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.pending_action_contracts import (
    PendingActionIssue,
    PendingActionIssueCode,
    PendingActionListResult,
)
from cayu.sessions.queries import SessionListResult
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import ForkSessionRequest, ResumeRequest, RunRequest
from cayu.tasks.creation import TaskCreate
from cayu.tasks.dispatch import Dispatcher, DispatchHandle, DispatchRequest, DispatchStatus
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor


def test_control_plane_redacts_legacy_session_event_transcript_pending_and_task_data() -> None:
    secret = "legacy-control-plane-secret-canary"
    task_store = InMemoryTaskStore()
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    session_id = f"legacy-{secret}-session"
    model_step_id = f"mstep_{'1' * 32}"
    model_attempt_id = f"matt_{'2' * 32}"
    tool_round_id = f"tround_{'3' * 32}"

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
                metadata={secret: f"session value {secret}"},
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)
        await app.session_store.append_events(
            session_id,
            [
                Event(
                    id="legacy_secret_approval",
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id=session_id,
                    agent_name="assistant",
                    tool_name="deploy",
                    payload={
                        "approval_id": "legacy_approval",
                        "tool_call_id": "legacy_call",
                        "model_step_id": model_step_id,
                        "model_attempt_id": model_attempt_id,
                        "tool_round_id": tool_round_id,
                        "approval": {
                            "approval_id": "legacy_approval",
                            "tool_round_id": tool_round_id,
                            "model_step_id": model_step_id,
                            "model_attempt_id": model_attempt_id,
                            "tool_call_id": "legacy_call",
                            "tool_name": "deploy",
                            "arguments": {secret: f"checkpoint value {secret}"},
                            "agent_name": "assistant",
                            "tool_calls": [
                                {
                                    "tool_call_id": "legacy_call",
                                    "tool_name": "deploy",
                                    "arguments": {secret: f"checkpoint value {secret}"},
                                    "policy_decision": None,
                                    "reason": None,
                                    "metadata": {},
                                    "active_taint_labels": [],
                                }
                            ],
                        },
                    },
                )
            ],
        )
        await app.session_store.append_transcript_messages(
            session_id,
            [
                Message.text("user", f"legacy transcript {secret}"),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        ProviderStatePart(
                            provider="fake",
                            state={secret: f"provider state {secret}"},
                        ),
                    ),
                ),
            ],
        )
        await app.session_store.checkpoint(
            session_id,
            {
                "pending_tool_approval": {
                    "approval_id": "legacy_approval",
                    "tool_round_id": tool_round_id,
                    "model_step_id": model_step_id,
                    "model_attempt_id": model_attempt_id,
                    "tool_call_id": "legacy_call",
                    "tool_name": "deploy",
                    "arguments": {secret: f"checkpoint value {secret}"},
                    "agent_name": "assistant",
                    "publish_arguments": True,
                    "tool_calls": [
                        {
                            "tool_call_id": "legacy_call",
                            "tool_name": "deploy",
                            "arguments": {secret: f"checkpoint value {secret}"},
                            "policy_decision": None,
                            "reason": None,
                            "metadata": {},
                            "active_taint_labels": [],
                        }
                    ],
                }
            },
        )
        await task_store.create_task(
            TaskCreate(
                task_id="legacy_secret_task",
                type="review",
                title=f"legacy task {secret}",
                description=f"legacy task description {secret}",
                input={secret: f"task value {secret}"},
                metadata={secret: f"task metadata {secret}"},
            )
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    public_session_id = app.project_session_id_for_exposure(session_id)
    assert public_session_id != session_id

    responses = [
        client.get(f"/api/sessions/{public_session_id}"),
        client.get(f"/api/sessions/{public_session_id}/events"),
        client.get(f"/api/sessions/{public_session_id}/transcript"),
        client.get(f"/api/pending-actions?session_id={public_session_id}"),
        client.get("/api/tasks"),
        client.get("/api/tasks/legacy_secret_task"),
        client.get("/api/sessions"),
    ]

    for response in responses:
        assert response.status_code == 200
        rendered = json.dumps(response.json(), sort_keys=True)
        assert secret not in rendered

    assert REDACTED_SECRET in "".join(
        json.dumps(response.json(), sort_keys=True) for response in responses
    )

    pending_body = responses[3].json()
    assert pending_body["actions"][0]["session"]["id"] == public_session_id
    assert pending_body["actions"][0]["arguments"] is None
    assert pending_body["next_cursor"] is None
    assert responses[0].json()["id"] == public_session_id
    assert responses[-1].json()["sessions"][0]["id"] == public_session_id


def test_control_plane_rejects_secret_bearing_session_cursor_authority() -> None:
    secret = "cursor-secret-canary"
    secret_session_id = f"session-{secret}"
    safe_session_id = "session-safe-cursor-boundary"
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    async def seed_pending_approval(session_id: str, suffix: str) -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        await store.append_event(
            session_id,
            Event(
                id=f"approval-event-{suffix}",
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id=session_id,
                tool_name="deploy",
                payload={
                    "approval": {
                        "approval_id": f"approval-{suffix}",
                        "tool_name": "deploy",
                        "arguments": {},
                    }
                },
            ),
        )
        await store.checkpoint(
            session_id,
            {
                "pending_tool_approval": {
                    "approval_id": f"approval-{suffix}",
                    "tool_call_id": f"call-{suffix}",
                    "tool_name": "deploy",
                    "arguments": {},
                    "agent_name": "assistant",
                    "tool_calls": [
                        {
                            "tool_call_id": f"call-{suffix}",
                            "tool_name": "deploy",
                            "arguments": {},
                            "policy_decision": None,
                            "reason": None,
                            "metadata": {},
                            "active_taint_labels": [],
                        }
                    ],
                }
            },
        )

    async def seed() -> None:
        # Pending-action and session listings sort newest first by default, so
        # the second record becomes the keyset boundary for a one-item page.
        await seed_pending_approval(safe_session_id, "safe")
        await seed_pending_approval(secret_session_id, "legacy")

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    responses = [
        client.get("/api/sessions", params={"limit": 1}),
        client.post("/api/sessions/summary", params={"limit": 1}),
        client.get("/api/pending-actions", params={"limit": 1}),
    ]

    for response in responses:
        assert response.status_code == 409
        rendered = json.dumps(response.json(), sort_keys=True)
        assert secret not in rendered
        assert response.json()["detail"] == (
            "Session pagination cannot continue because its cursor authority "
            "contains a configured workload secret."
        )


def test_control_plane_preserves_secret_free_opaque_custom_store_cursor() -> None:
    class OpaqueCursorStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def list_sessions(self, query=None):
            del query
            return SessionListResult(
                sessions=[],
                next_cursor="custom-store:opaque-page-2",
                total_count=0,
            )

    app = CayuApp(
        session_store=OpaqueCursorStore(),
        enable_logging=False,
    )
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.get("/api/sessions")

    assert response.status_code == 200
    assert response.json()["next_cursor"] == "custom-store:opaque-page-2"


def test_control_plane_hides_private_run_lineage_for_root_and_child() -> None:
    app = CayuApp(enable_logging=False)

    async def seed():
        for session_id, parent in (("root", None), ("child", "root")):
            await app.session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    parent_session_id=parent,
                    messages=[],
                    metadata={"note": "visible"},
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
        child = await app.session_store.load("child")
        assert child.metadata["cayu:run_lineage"]["parent_session_id"] == "root"

    asyncio.run(seed())
    with TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)) as client:
        for session_id in ("root", "child"):
            response = client.get(f"/api/sessions/{session_id}")
            assert response.status_code == 200
            expected_metadata = {"note": "visible"}
            if session_id == "child":
                expected_metadata["cayu:fork_lineage"] = {"controller": None, "forked": False}
            assert response.json()["metadata"] == expected_metadata
            assert "cayu:run_lineage" not in response.text


def test_control_plane_short_secret_preserves_typed_response_envelopes() -> None:
    secret = "a"
    task_store = InMemoryTaskStore()
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    session_id = "sess_001"

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="bot",
                session_id=session_id,
                messages=[],
                metadata={"data": "safe"},
            ),
            identity=SessionIdentity(provider_name="prov", model="mdl"),
        )
        await app.session_store.append_events(
            session_id,
            [
                Event(
                    id=secret * 100,
                    type=EventType.SESSION_STARTED,
                    session_id=session_id,
                    agent_name="bot",
                    payload={"data": "safe"},
                )
            ],
        )
        await app.session_store.append_transcript_messages(
            session_id,
            [Message.text("user", "safe")],
        )
        await task_store.create_task(
            TaskCreate(
                task_id="tsk_001",
                type="review",
                title="safe",
                input={"data": "safe"},
            )
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    session_response = client.get(f"/api/sessions/{session_id}")
    events_response = client.get(f"/api/sessions/{session_id}/events")
    transcript_response = client.get(f"/api/sessions/{session_id}/transcript")
    task_list_response = client.get("/api/tasks")
    task_detail_response = client.get("/api/tasks/tsk_001")

    assert session_response.status_code == 200
    assert events_response.status_code == 200
    assert transcript_response.status_code == 200
    assert task_list_response.status_code == 200
    assert task_detail_response.status_code == 200

    session_body = session_response.json()
    assert "agent_name" in session_body
    assert session_body["status"] == "pending"
    redacted_untrusted = app.redact_json({"data": "safe"})
    assert session_body["metadata"] == redacted_untrusted

    event_body = events_response.json()["events"][0]
    assert event_body["type"] == "session.started"
    assert len(event_body["id"]) <= EVENT_ID_MAX_CHARS
    assert "payload" in event_body

    transcript_body = transcript_response.json()["messages"][0]
    assert transcript_body["role"] == "user"
    assert transcript_body["content"][0]["type"] == "text"

    assert task_list_response.json()[0]["status"] == "pending"
    assert task_detail_response.json()["input"] == redacted_untrusted


def test_control_plane_short_secret_preserves_file_attachment_protocol() -> None:
    secret = "image"
    session_id = "legacy_file_attachment_protocol"
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    attachment = FileAttachment(
        artifact_id="safe-artifact",
        kind=FileAttachmentKind.IMAGE,
        filename=f"private-{secret}-file.png",
        content_type="image/png",
        size_bytes=12,
        metadata={f"private-{secret}-key": f"private {secret} value"},
    )

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_transcript_messages(
            session_id,
            [
                Message(
                    role=MessageRole.USER,
                    content=(
                        FilePart(
                            attachment=attachment.model_dump(mode="json"),
                        ),
                    ),
                )
            ],
        )

    asyncio.run(seed())
    response = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)).get(
        f"/api/sessions/{session_id}/transcript"
    )

    assert response.status_code == 200
    payload = response.json()["messages"][0]["content"][0]["attachment"]
    serialized = FileAttachment.model_validate(payload)
    assert serialized.kind is FileAttachmentKind.IMAGE
    assert serialized.content_type == "image/png"
    assert serialized.type == attachment.type
    assert secret not in serialized.filename
    assert REDACTED_SECRET in serialized.filename
    assert secret not in repr(serialized.metadata)
    assert REDACTED_SECRET in repr(serialized.metadata)


def test_pending_action_issue_preserves_typed_timestamp_for_short_secret() -> None:
    observed_at = datetime(2026, 7, 27, tzinfo=UTC)

    class IssueStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def query_pending_actions(self, query=None, *, checkpoint_root_guard=None):
            del query
            del checkpoint_root_guard
            return PendingActionListResult(
                issues=[
                    PendingActionIssue(
                        code=PendingActionIssueCode.SOURCE_INVALID,
                        session_id="safe-session",
                        agent_name="safe-agent",
                        status=SessionStatus.INTERRUPTED,
                        updated_at=observed_at,
                        detail="safe detail",
                    )
                ],
                total_count=1,
                inspected_candidate_count=1,
            )

    app = CayuApp(
        session_store=IssueStore(),
        secret_redactor=SecretRedactor("2"),
        enable_logging=False,
    )
    response = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)).get(
        "/api/pending-actions"
    )

    assert response.status_code == 200
    assert (
        datetime.fromisoformat(response.json()["issues"][0]["updated_at"].replace("Z", "+00:00"))
        == observed_at
    )


def test_control_plane_preserves_runtime_timestamps_for_short_secret() -> None:
    task_store = InMemoryTaskStore()
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor("2"),
        enable_logging=False,
    )
    session_id = "safe-session"

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="bot",
                session_id=session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            session_id,
            [
                Event(
                    id="safe-event",
                    type=EventType.SESSION_STARTED,
                    session_id=session_id,
                )
            ],
        )
        task = await task_store.create_task(
            TaskCreate(
                task_id="safe-task",
                type="review",
            )
        )
        claimed = await task_store.claim_task("safe-worker")
        assert claimed is not None
        await task_store.complete_task(
            task.id,
            {"status": "done"},
            worker_id="safe-worker",
            lease_expires_at=claimed.lease_expires_at,
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    session = client.get(f"/api/sessions/{session_id}")
    event = client.get(f"/api/sessions/{session_id}/events")
    tasks = client.get("/api/tasks")
    task = client.get("/api/tasks/safe-task")

    assert session.status_code == 200
    assert event.status_code == 200
    assert tasks.status_code == 200
    assert task.status_code == 200
    for value in (
        session.json()["created_at"],
        session.json()["updated_at"],
        event.json()["events"][0]["timestamp"],
        tasks.json()[0]["created_at"],
        tasks.json()[0]["updated_at"],
        tasks.json()[0]["completed_at"],
        task.json()["started_at"],
    ):
        assert value is not None
        datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_generated_session_and_interaction_aliases_remain_distinct_and_addressable() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("-"), enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fakemodel"))

    async def seed() -> tuple[str, str, str]:
        first_session_events = await _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "first")],
                max_steps=20,
            ),
        )
        first_session_id = first_session_events[0].session_id
        assert {event.session_id for event in first_session_events} == {first_session_id}
        assert (
            len(
                {
                    event.interaction_id
                    for event in first_session_events
                    if event.interaction_id is not None
                }
            )
            == 1
        )
        second_session_events = await _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "other")],
                max_steps=20,
            ),
        )
        return first_session_id, first_session_events[0].id, second_session_events[0].session_id

    first_session_id, first_event_id, second_session_id = asyncio.run(seed())
    assert public_event_envelope_alias_field(first_session_id) == "session_id"
    assert public_event_envelope_alias_field(second_session_id) == "session_id"
    assert first_session_id != second_session_id

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    replay = client.post(
        "/api/resume",
        json={"session_id": first_session_id, "prompt": "must not dispatch"},
        headers={"Last-Event-ID": f"{first_session_id}:{first_event_id}"},
    )
    assert replay.status_code == 200, replay.text

    response = client.get(f"/api/sessions/{first_session_id}/interactions?limit=10")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == first_session_id
    assert len(body["interactions"]) == 1
    interaction_ids = [item["interaction_id"] for item in body["interactions"]]
    assert len(set(interaction_ids)) == 1
    assert all(
        public_event_envelope_alias_field(interaction_id) == "interaction_id"
        for interaction_id in interaction_ids
    )

    interaction = body["interactions"][0]
    detail = client.get(
        f"/api/sessions/{first_session_id}/interactions/{interaction['interaction_id']}"
    )
    assert detail.status_code == 200
    assert detail.json() == interaction

    events = client.get(
        f"/api/sessions/{first_session_id}/events",
        params={"interaction_id": interaction["interaction_id"]},
    )
    assert events.status_code == 200
    event_body = events.json()
    assert event_body["session_id"] == first_session_id
    assert event_body["events"]
    assert {event["session_id"] for event in event_body["events"]} == {first_session_id}
    assert {event["interaction_id"] for event in event_body["events"]} == {
        interaction["interaction_id"]
    }

    transcript = client.get(
        f"/api/sessions/{first_session_id}/transcript",
        params={"interaction_id": interaction["interaction_id"]},
    )
    assert transcript.status_code == 200
    assert transcript.json()["session_id"] == first_session_id
    assert transcript.json()["messages"]
    assert {
        message["interaction_id"]
        for message in transcript.json()["messages"]
        if message["interaction_id"] is not None
    } == {interaction["interaction_id"]}
    assert (
        client.get(
            f"/api/sessions/{second_session_id}/interactions/{interaction['interaction_id']}"
        ).status_code
        == 409
    )

    resumed = client.post(
        "/api/resume",
        json={"session_id": first_session_id, "prompt": "continue normally", "max_steps": 20},
    )
    assert resumed.status_code == 200
    assert '"type":"session.completed"' in resumed.text


def test_alias_shaped_legacy_private_authority_requires_positive_projection() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("_"), enable_logging=False)
    private_session_id = "cayu_authority_v1.legacy.session_id." + "A" * 43
    private_interaction_id = "cayu_authority_v1.legacy.interaction_id." + "B" * 43

    async def scenario() -> tuple[str, str]:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=private_session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="model"),
        )
        await app.session_store.append_transcript_messages(
            private_session_id,
            [Message.text("user", "legacy")],
            interaction_id=private_interaction_id,
        )
        return (
            app.project_session_id_for_exposure(private_session_id),
            app.project_interaction_id_for_exposure(
                private_interaction_id,
                session_id=private_session_id,
            ),
        )

    public_session_id, public_interaction_id = asyncio.run(scenario())
    assert public_session_id != private_session_id
    assert public_interaction_id != private_interaction_id
    assert public_event_envelope_alias_field(public_session_id) == "session_id"
    assert public_event_envelope_alias_field(public_interaction_id) == "interaction_id"


def test_transcript_only_raw_interaction_alias_conflict_is_rejected() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("private"), enable_logging=False)
    session_id = "session"
    private_interaction_id = "other-private-interaction"
    conflicting_raw_id = app.project_interaction_id_for_exposure(
        private_interaction_id,
        session_id=session_id,
    )

    async def scenario() -> None:
        await app.session_store.create(
            RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
            identity=SessionIdentity(provider_name="fake", model="model"),
        )
        await app.session_store.append_event(
            session_id,
            Event(
                type=EventType.INTERACTION_COMPLETED,
                session_id=session_id,
                interaction_id=private_interaction_id,
            ),
        )
        await app.session_store.append_transcript_messages(
            session_id,
            [Message.text("user", "legacy")],
            interaction_id=conflicting_raw_id,
        )
        with pytest.raises(ValueError, match="ambiguous"):
            await app._resolve_public_interaction_id(
                session_id=session_id,
                value=conflicting_raw_id,
            )

    asyncio.run(scenario())


def test_in_memory_nested_turn_interaction_alias_round_trips() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("private"), enable_logging=False)
    session_id = "session"
    private_interaction_id = "nested-private-interaction"

    async def scenario() -> None:
        await app.session_store.create(
            RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
            identity=SessionIdentity(provider_name="fake", model="model"),
        )
        await app.session_store.append_event(
            session_id,
            Event(
                type=EventType.TURN_COMPLETED,
                session_id=session_id,
                payload={"interaction_ids": [private_interaction_id]},
            ),
        )
        public_interaction_id = app.project_interaction_id_for_exposure(
            private_interaction_id,
            session_id=session_id,
        )
        assert (
            await app._resolve_public_interaction_id(
                session_id=session_id,
                value=public_interaction_id,
            )
            == private_interaction_id
        )

    asyncio.run(scenario())


def test_server_generated_session_authority_survives_short_secret_collision() -> None:
    store = InMemorySessionStore()
    task_store = InMemoryTaskStore()
    app = CayuApp(
        session_store=store,
        task_store=task_store,
        secret_redactor=SecretRedactor("-"),
        enable_logging=False,
    )
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fakemodel"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    events = _sse_events(client, "/api/run", {"agent": "assistant", "prompt": "first"})
    public_session_id = events[0]["session_id"]
    public_interaction_id = next(
        event["interaction_id"] for event in events if event["type"] == "interaction.started"
    )
    turn_completed = next(event for event in events if event["type"] == "turn.completed")

    async def private_state() -> tuple[str, str, list[str]]:
        sessions = await store.list_sessions()
        assert len(sessions.sessions) == 1
        private_session_id = sessions.sessions[0].id
        turn_records = await store.query_events(
            EventQuery(
                session_id=private_session_id,
                event_type=EventType.TURN_COMPLETED,
            )
        )
        interaction_records = await store.query_events(
            EventQuery(
                session_id=private_session_id,
                event_type=EventType.INTERACTION_STARTED,
            )
        )
        assert len(turn_records) == 1
        assert len(interaction_records) == 1
        private_interaction_id = interaction_records[0].event.interaction_id
        assert private_interaction_id is not None
        private_interaction_ids = turn_records[0].event.payload["interaction_ids"]
        assert type(private_interaction_ids) is list
        assert all(type(value) is str for value in private_interaction_ids)
        return private_session_id, private_interaction_id, private_interaction_ids

    private_session_id, private_interaction_id, private_interaction_ids = asyncio.run(
        private_state()
    )
    tasks = asyncio.run(task_store.list_tasks())

    assert private_session_id.startswith("session-")
    assert public_session_id == app.project_session_id_for_exposure(private_session_id)
    assert private_interaction_ids == [private_interaction_id]
    assert turn_completed["payload"]["interaction_ids"] == [public_interaction_id]
    assert len(tasks) == 1
    assert tasks[0].session_id == private_session_id
    assert tasks[0].status is TaskStatus.COMPLETED

    resumed = _sse_events(
        client,
        "/api/resume",
        {"session_id": public_session_id, "prompt": "second"},
    )
    assert resumed[-1]["type"] == "session.completed"
    assert {event["session_id"] for event in resumed} == {public_session_id}


def test_generated_session_causal_budget_alias_round_trips_through_http() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("private"), enable_logging=False)
    private_session_id = "generated-private-session"

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=private_session_id,
                causal_budget_id=private_session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fakemodel"),
        )

    asyncio.run(seed())
    public_session_id = app.project_session_id_for_exposure(private_session_id)
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    pricing_body = {"pricing": _price_book_payload(model="fakemodel")}
    responses = (
        client.get(f"/api/causal-budgets/{public_session_id}/usage"),
        client.post(
            f"/api/causal-budgets/{public_session_id}/cost",
            json=pricing_body,
        ),
        client.post(
            f"/api/causal-budgets/{public_session_id}/summary",
            json=pricing_body,
        ),
    )
    for response in responses:
        assert response.status_code == 200, response.text
        assert response.json()["causal_budget_id"] == public_session_id
    assert responses[0].json()["session_ids"] == [public_session_id]
    assert responses[1].json()["session_ids"] == [public_session_id]
    assert responses[2].json()["usage"]["causal_budget_id"] == public_session_id
    assert responses[2].json()["cost"]["causal_budget_id"] == public_session_id


def test_public_session_lineage_aliases_round_trip_through_list_filters() -> None:
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor("private"),
        enable_logging=False,
    )
    private_root_id = "private-root-session"
    child_id = "child-session"

    async def seed() -> None:
        identity = SessionIdentity(provider_name="fake", model="fakemodel")
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=private_root_id,
                messages=[],
            ),
            identity=identity,
        )
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=child_id,
                parent_session_id=private_root_id,
                causal_budget_id=private_root_id,
                messages=[],
            ),
            identity=identity,
        )

    asyncio.run(seed())
    public_root_id = app.project_session_id_for_exposure(private_root_id)
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    unfiltered = client.get("/api/sessions", params={"order_by": "created_at_asc"})
    assert unfiltered.status_code == 200, unfiltered.text
    sessions = unfiltered.json()["sessions"]
    assert [session["id"] for session in sessions] == [public_root_id, child_id]
    assert sessions[0]["causal_budget_id"] == public_root_id
    assert sessions[1]["parent_session_id"] == public_root_id
    assert sessions[1]["causal_budget_id"] == public_root_id

    parent_list = client.get(
        "/api/sessions",
        params={"parent_session_id": public_root_id},
    )
    parent_summary = client.post(
        "/api/sessions/summary",
        params={"parent_session_id": public_root_id},
    )
    causal_list = client.get(
        "/api/sessions",
        params={"causal_budget_id": public_root_id, "order_by": "created_at_asc"},
    )
    causal_summary = client.post(
        "/api/sessions/summary",
        params={"causal_budget_id": public_root_id, "order_by": "created_at_asc"},
    )

    assert parent_list.status_code == 200, parent_list.text
    assert [session["id"] for session in parent_list.json()["sessions"]] == [child_id]
    assert parent_summary.status_code == 200, parent_summary.text
    assert [item["session"]["id"] for item in parent_summary.json()["sessions"]] == [child_id]
    assert causal_list.status_code == 200, causal_list.text
    assert [session["id"] for session in causal_list.json()["sessions"]] == [
        public_root_id,
        child_id,
    ]
    assert causal_summary.status_code == 200, causal_summary.text
    assert [item["session"]["id"] for item in causal_summary.json()["sessions"]] == [
        public_root_id,
        child_id,
    ]


def test_alias_shaped_raw_causal_budget_round_trips_through_all_http_summaries() -> None:
    session_store = InMemorySessionStore()
    raw_causal_budget_id = session_store.public_authority_alias_codec.encode(
        "not-a-session",
        field_name="session_id",
    )
    app = CayuApp(session_store=session_store, enable_logging=False)

    async def seed() -> None:
        await session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="raw-causal-budget-session",
                causal_budget_id=raw_causal_budget_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fakemodel"),
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    pricing_body = {"pricing": _price_book_payload(model="fakemodel")}
    responses = (
        client.get(f"/api/causal-budgets/{raw_causal_budget_id}/usage"),
        client.post(
            f"/api/causal-budgets/{raw_causal_budget_id}/cost",
            json=pricing_body,
        ),
        client.post(
            f"/api/causal-budgets/{raw_causal_budget_id}/summary",
            json=pricing_body,
        ),
    )
    for response in responses:
        assert response.status_code == 200, response.text
        assert response.json()["causal_budget_id"] == raw_causal_budget_id


def test_non_session_causal_budget_secret_is_redacted_from_all_http_summaries() -> None:
    private_causal_budget_id = "legacy-secret-budget"
    app = CayuApp(secret_redactor=SecretRedactor("secret"), enable_logging=False)

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session-one",
                causal_budget_id=private_causal_budget_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fakemodel"),
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    pricing_body = {"pricing": _price_book_payload(model="fakemodel")}
    responses = (
        client.get(f"/api/causal-budgets/{private_causal_budget_id}/usage"),
        client.post(
            f"/api/causal-budgets/{private_causal_budget_id}/cost",
            json=pricing_body,
        ),
        client.post(
            f"/api/causal-budgets/{private_causal_budget_id}/summary",
            json=pricing_body,
        ),
    )
    public_causal_budget_id = f"legacy-{REDACTED_SECRET}-budget"

    for response in responses:
        assert response.status_code == 200, response.text
        assert private_causal_budget_id not in response.text
        assert response.json()["causal_budget_id"] == public_causal_budget_id
    assert responses[2].json()["usage"]["causal_budget_id"] == public_causal_budget_id
    assert responses[2].json()["cost"]["causal_budget_id"] == public_causal_budget_id


def test_direct_app_resume_accepts_a_generated_public_session_alias() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("-"), enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fakemodel"))

    async def scenario() -> tuple[list[Event], list[Event]]:
        initial = await _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "first")],
            ),
        )
        resumed = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=initial[0].session_id,
                    messages=[Message.text("user", "second")],
                )
            )
        ]
        return initial, resumed

    initial, resumed = asyncio.run(scenario())

    assert public_event_envelope_alias_field(initial[0].session_id) == "session_id"
    assert resumed[-1].type is EventType.SESSION_COMPLETED
    assert {event.session_id for event in resumed} == {initial[0].session_id}


def test_direct_app_fork_and_dispatch_accept_generated_public_session_alias() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("-"), enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fakemodel"))

    async def scenario() -> tuple[str, list[Event], list[Event], list[Event], Any, Any]:
        initial = await _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "first")],
            ),
        )
        public_session_id = initial[0].session_id
        forked = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(source_session_id=public_session_id)
            )
        ]
        grandchild = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(source_session_id=forked[0].session_id)
            )
        ]
        inline = [
            event
            async for event in app.dispatch_inline(
                DispatchRequest(
                    session_id=public_session_id,
                    dispatch_id="inline_dispatch",
                    messages=[Message.text("user", "inline dispatch")],
                )
            )
        ]
        handle = await app.dispatch(
            DispatchRequest(
                session_id=public_session_id,
                dispatch_id="submitted_dispatch",
                messages=[Message.text("user", "submitted dispatch")],
            )
        )
        stored_fork_id = await app._resolve_public_session_id(forked[0].session_id)
        stored_fork = await app.session_store.load(stored_fork_id)
        assert stored_fork is not None
        return public_session_id, forked, grandchild, inline, handle, stored_fork

    public_session_id, forked, grandchild, inline, handle, stored_fork = asyncio.run(scenario())

    assert public_event_envelope_alias_field(public_session_id) == "session_id"
    assert [event.type for event in forked] == [EventType.SESSION_FORKED]
    assert public_event_envelope_alias_field(forked[0].session_id) == "session_id"
    assert [event.type for event in grandchild] == [EventType.SESSION_FORKED]
    assert public_event_envelope_alias_field(grandchild[0].session_id) == "session_id"
    assert grandchild[0].session_id != forked[0].session_id
    assert inline[-1].type is EventType.SESSION_COMPLETED, inline[-1].payload
    assert {event.session_id for event in inline} == {public_session_id}
    assert handle.status is DispatchStatus.COMPLETED
    assert handle.session_id == public_session_id
    assert stored_fork.runtime_build_provenance.origin.value != "legacy_record"
    assert "-" in stored_fork.runtime_build_provenance.recipe


def test_exact_secret_source_alias_rejects_fork_before_child_creation() -> None:
    private_session_id = "exact-session-secret"
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(private_session_id),
        enable_logging=False,
    )
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fakemodel"))

    async def scenario() -> tuple[str, ...]:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=private_session_id,
                messages=[Message.text("user", "legacy")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fakemodel"),
        )
        await store.update_status(private_session_id, SessionStatus.COMPLETED)
        public_session_id = app.project_session_id_for_exposure(private_session_id)
        with pytest.raises(ValueError, match="source_session.id contains a workload secret"):
            _ = [
                event
                async for event in app.fork_session(
                    ForkSessionRequest(source_session_id=public_session_id)
                )
            ]
        sessions = await store.list_sessions()
        return tuple(session.id for session in sessions.sessions)

    assert asyncio.run(scenario()) == (private_session_id,)


def test_dispatch_projects_a_private_session_id_returned_by_a_dispatcher() -> None:
    class EchoDispatcher(Dispatcher):
        async def submit(self, runtime, request):
            del runtime
            return DispatchHandle(
                dispatch_id=request.dispatch_id,
                session_id=request.session_id,
                backend="echo",
            )

    app = CayuApp(
        dispatcher=EchoDispatcher(),
        secret_redactor=SecretRedactor("-"),
        enable_logging=False,
    )
    handle = asyncio.run(
        app.dispatch(
            DispatchRequest(
                session_id="private-session-id",
                dispatch_id="safe_dispatch",
                messages=[Message.text("user", "queued")],
            )
        )
    )

    assert handle.session_id == app.project_session_id_for_exposure("private-session-id")


def test_public_session_alias_resolution_uses_the_store_index_without_a_scan() -> None:
    class EndlessSessionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.list_calls = 0

        async def list_sessions(self, query=None):
            del query
            self.list_calls += 1
            return SessionListResult(
                sessions=[],
                next_cursor=f"opaque-page-{self.list_calls}",
            )

    store = EndlessSessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor("private"),
        enable_logging=False,
    )
    alias = app.project_session_id_for_exposure("private-session")

    async def resolve() -> None:
        with pytest.raises(ValueError, match="was not found"):
            await app._resolve_public_session_id(alias)

    asyncio.run(resolve())

    assert store.list_calls == 0


def test_action_routes_resolve_schema_owned_nested_public_linkage_aliases() -> None:
    app = CayuApp(enable_logging=False)

    async def seed() -> None:
        for session_id, payload in (
            (
                "session_nested_approval_alias",
                {
                    "interruption_type": "tool_approval_required",
                    "approval": {
                        "approval_id": "approval-private",
                        "tool_round_id": "round-private",
                        "tool_call_id": "call-private",
                    },
                },
            ),
            (
                "session_nested_input_alias",
                {
                    "interruption_type": "user_input_required",
                    "user_input": {
                        "input_id": "input-private",
                        "tool_call_id": "input-call-private",
                    },
                },
            ),
        ):
            await app.session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await app.session_store.append_event(
                session_id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session_id,
                    payload=payload,
                ),
            )
            await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)

    asyncio.run(seed())
    approval_sequence = asyncio.run(
        app.session_store.query_events(
            EventQuery(session_id="session_nested_approval_alias", limit=1)
        )
    )[0].sequence
    input_sequence = asyncio.run(
        app.session_store.query_events(EventQuery(session_id="session_nested_input_alias", limit=1))
    )[0].sequence
    approvals = []
    inputs = []

    async def resolve_approval(request):
        approvals.append(request)
        yield await app.emit_event(
            Event(
                type=EventType.SESSION_RESUMED,
                session_id=request.session_id,
            )
        )

    async def resolve_input(request):
        inputs.append(request)
        yield await app.emit_event(
            Event(
                type=EventType.SESSION_RESUMED,
                session_id=request.session_id,
            )
        )

    app.resolve_tool_approval = resolve_approval
    app.resolve_user_input = resolve_input
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_nested_approval_alias",
            "approval_id": public_event_linkage_id(approval_sequence, "approval_id"),
            "tool_round_id": public_event_linkage_id(approval_sequence, "tool_round_id"),
            "tool_call_id": public_event_linkage_id(approval_sequence, "tool_call_id"),
            "decision": "approve",
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    with client.stream(
        "POST",
        "/api/user-input/resolve",
        json={
            "session_id": "session_nested_input_alias",
            "input_id": public_event_linkage_id(input_sequence, "input_id"),
            "answer": "continue",
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    malformed_alias = client.post(
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_nested_approval_alias",
            "approval_id": f"cayu_event_{MAX_DURABLE_JSON_INTEGER + 1}:approval_id",
            "tool_round_id": public_event_linkage_id(
                approval_sequence,
                "tool_round_id",
            ),
            "tool_call_id": public_event_linkage_id(
                approval_sequence,
                "tool_call_id",
            ),
            "decision": "approve",
        },
    )

    assert malformed_alias.status_code == 409
    assert "malformed" in malformed_alias.json()["detail"]
    assert (
        approvals[0].approval_id,
        approvals[0].tool_round_id,
        approvals[0].tool_call_id,
    ) == (
        "approval-private",
        "round-private",
        "call-private",
    )
    assert inputs[0].input_id == "input-private"
    assert len(approvals) == 1


def test_action_linkage_disambiguates_legacy_raw_values_from_public_aliases() -> None:
    app = CayuApp(enable_logging=False)

    async def seed(
        session_id: str,
        *,
        approval_id: str,
        alias_target: str | None = None,
    ) -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        if alias_target is not None:
            await app.session_store.append_event(
                session_id,
                Event(
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id=session_id,
                    payload={
                        "approval_id": alias_target,
                        "tool_round_id": "alias-round",
                        "tool_call_id": "alias-call",
                    },
                ),
            )
        await app.session_store.append_event(
            session_id,
            Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id=session_id,
                payload={
                    "interruption_type": "tool_approval_required",
                    "approval": {
                        "approval_id": approval_id,
                        "tool_round_id": "legacy-round",
                        "tool_call_id": "legacy-call",
                    },
                },
            ),
        )
        await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)

    asyncio.run(
        seed(
            "session_linkage_raw_only",
            approval_id="cayu_event_01:approval_id",
        )
    )
    asyncio.run(
        seed(
            "session_linkage_same",
            approval_id="cayu_event_2:approval_id",
        )
    )
    asyncio.run(
        seed(
            "session_linkage_conflict",
            approval_id="cayu_event_3:approval_id",
            alias_target="different-private-approval",
        )
    )
    legacy_approval_ids = {
        "session_linkage_raw_only": "cayu_event_01:approval_id",
        "session_linkage_same": "cayu_event_2:approval_id",
        "session_linkage_conflict": "cayu_event_3:approval_id",
    }

    async def pending_actions(query):
        approval_id = legacy_approval_ids[query.session_id]
        return SimpleNamespace(actions=[SimpleNamespace(approval_id=approval_id)])

    app.session_store.query_pending_actions = pending_actions
    captured = []

    async def resolve(request):
        captured.append(request)
        yield await app.emit_event(
            Event(type=EventType.SESSION_RESUMED, session_id=request.session_id)
        )

    app.resolve_tool_approval = resolve
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    for session_id, approval_id in (
        ("session_linkage_raw_only", "cayu_event_01:approval_id"),
        ("session_linkage_same", "cayu_event_2:approval_id"),
    ):
        with client.stream(
            "POST",
            "/api/tool-approvals/resolve",
            json={
                "session_id": session_id,
                "approval_id": approval_id,
                "tool_round_id": "legacy-round",
                "tool_call_id": "legacy-call",
                "decision": "approve",
            },
        ) as response:
            if response.status_code != 200:
                response.read()
            assert response.status_code == 200, f"{session_id}: {response.text}"
            list(response.iter_lines())

    conflict = client.post(
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_linkage_conflict",
            "approval_id": "cayu_event_3:approval_id",
            "tool_round_id": "legacy-round",
            "tool_call_id": "legacy-call",
            "decision": "approve",
        },
    )

    assert [request.approval_id for request in captured] == [
        "cayu_event_01:approval_id",
        "cayu_event_2:approval_id",
    ]
    assert conflict.status_code == 409
    assert "ambiguous" in conflict.json()["detail"]


def test_dev_mode_resolution_restamps_body_resolved_by_as_request_source() -> None:
    from cayu.approvals.tools import ResolutionActorSource

    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_interrupted_session(session_id: str) -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)

    asyncio.run(create_interrupted_session("session_dev_actor"))
    asyncio.run(create_interrupted_session("session_dev_tool_round_actor"))

    captured = []
    captured_tool_round = []

    async def resolve_tool_approval(request):
        captured.append(request)
        yield Event(
            type=EventType.SESSION_RESUMED,
            session_id=request.session_id,
            agent_name="assistant",
        )

    async def recover_tool_round(request):
        captured_tool_round.append(request)
        yield Event(
            type=EventType.SESSION_RESUMED,
            session_id=request.session_id,
            agent_name="assistant",
        )

    app.resolve_tool_approval = resolve_tool_approval
    app.recover_tool_round = recover_tool_round
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    # An open-access body can assert an identity but never verified/system
    # provenance: the server re-stamps the source as "request".
    with client.stream(
        "POST",
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_dev_actor",
            "approval_id": "approval_1",
            "tool_round_id": "round_1",
            "tool_call_id": "call_1",
            "decision": "approve",
            "resolved_by": {"subject": "operator@example.com", "source": "system"},
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    actor = captured[0].resolved_by
    assert actor is not None
    assert actor.subject == "operator@example.com"
    assert actor.source is ResolutionActorSource.REQUEST

    with client.stream(
        "POST",
        "/api/tool-rounds/recover",
        json={
            "session_id": "session_dev_tool_round_actor",
            "round_id": "round_1",
            "tool_call_id": "call_1",
            "outcome": "completed",
            "message": "verified externally",
            "resolved_by": {"subject": "round-operator@example.com", "source": "system"},
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    round_actor = captured_tool_round[0].resolved_by
    assert round_actor is not None
    assert round_actor.subject == "round-operator@example.com"
    assert round_actor.source is ResolutionActorSource.REQUEST

    # Reserved system subjects cannot be claimed through the body: the
    # request-source re-stamp trips the reserved-prefix validation.
    response = client.post(
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_dev_actor",
            "approval_id": "approval_1",
            "tool_round_id": "round_1",
            "tool_call_id": "call_1",
            "decision": "approve",
            "resolved_by": {"subject": "cayu:approval-expiry", "source": "system"},
        },
    )
    assert response.status_code == 400
    assert "reserved for system actors" in response.json()["detail"]
    assert len(captured) == 1

    # No body actor means no provenance under open access.
    with client.stream(
        "POST",
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_dev_actor",
            "approval_id": "approval_1",
            "tool_round_id": "round_1",
            "tool_call_id": "call_1",
            "decision": "approve",
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())
    assert captured[1].resolved_by is None
