"""HTTP server session actions contracts."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
pytest.importorskip("httpx")

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from pydantic import SecretStr
from tests.server.server_test_support import (
    _LOCAL_SERVER_CONFIG,
    OneShotProvider,
    _create_session,
    _lifecycle_store_and_client,
    _sse_events,
    _sse_frames,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER, canonical_durable_json_bytes
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.context.base import CheckpointCompactionContextPolicy, TranscriptDigestCompactor
from cayu.events import Event, EventType, event_with_runtime_payload_authority
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime._event_projection import (
    PRIVATE_EVENT_AUTHORITY,
    public_event_id,
    public_event_linkage_id,
    public_event_sequence,
)
from cayu.runtime.provider_operations import PROVIDER_OPERATION_RESOLUTION_METADATA_MAX_BYTES
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.server import ServerConfig, create_server
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.user_input import UserInputTool
from cayu.vaults.redaction import SecretRedactor


def test_server_resolves_provider_operation_with_bounded_audited_request() -> None:
    from cayu.approvals.tools import ResolutionActorSource

    app = CayuApp()

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="provider_resolution",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(
            "provider_resolution",
            SessionStatus.INTERRUPTED,
        )

    asyncio.run(seed())
    captured = []

    async def resolve_provider_operation(request):
        captured.append(request)
        yield Event(
            type=EventType.PROVIDER_OPERATION_RESOLVED,
            session_id=request.session_id,
            payload={"resolution_action": request.action.value},
        )

    app.resolve_provider_operation = resolve_provider_operation  # type: ignore[method-assign]
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/provider-operations/resolve",
        json={
            "session_id": "provider_resolution",
            "stage_id": "stage_1",
            "expected_run_epoch": 0,
            "action": "fail",
            "reason": "Operator confirmed the provider request cannot be recovered.",
            "metadata": {"ticket": "INC-757"},
            "resolved_by": {"subject": "operator-a"},
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    assert len(captured) == 1
    request = captured[0]
    assert request.session_id == "provider_resolution"
    assert request.stage_id == "stage_1"
    assert request.expected_run_epoch == 0
    assert request.action.value == "fail"
    assert request.metadata == {"ticket": "INC-757"}
    assert request.resolved_by is not None
    assert request.resolved_by.subject == "operator-a"
    assert request.resolved_by.source is ResolutionActorSource.REQUEST


def test_provider_operation_resolution_endpoint_enforces_runtime_bounds() -> None:
    app = CayuApp()

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="provider_resolution_bounds",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(
            "provider_resolution_bounds",
            SessionStatus.INTERRUPTED,
        )

    asyncio.run(seed())
    captured = []

    async def resolve_provider_operation(request):
        captured.append(request)
        yield Event(
            type=EventType.PROVIDER_OPERATION_RESOLVED,
            session_id=request.session_id,
            payload={"resolution_action": request.action.value},
        )

    app.resolve_provider_operation = resolve_provider_operation  # type: ignore[method-assign]
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    metadata_overhead = len(canonical_durable_json_bytes({"note": ""}, "metadata"))
    value = "x" * (PROVIDER_OPERATION_RESOLUTION_METADATA_MAX_BYTES - metadata_overhead)
    at_limit_metadata = {"note": value}
    assert (
        len(canonical_durable_json_bytes(at_limit_metadata, "metadata"))
        == PROVIDER_OPERATION_RESOLUTION_METADATA_MAX_BYTES
    )
    base_request = {
        "session_id": "provider_resolution_bounds",
        "stage_id": "stage_1",
        "expected_run_epoch": MAX_DURABLE_JSON_INTEGER,
        "action": "fail",
        "metadata": at_limit_metadata,
    }

    with client.stream(
        "POST",
        "/api/provider-operations/resolve",
        json=base_request,
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())
    assert len(captured) == 1

    canary = "provider-resolution-boundary-secret"
    oversized_value = value[: -len(canary)] + canary + "x"
    metadata_response = client.post(
        "/api/provider-operations/resolve",
        json={**base_request, "metadata": {"note": oversized_value}},
    )
    assert metadata_response.status_code == 422
    assert canary not in metadata_response.text

    epoch_response = client.post(
        "/api/provider-operations/resolve",
        json={**base_request, "expected_run_epoch": MAX_DURABLE_JSON_INTEGER + 1},
    )
    assert epoch_response.status_code == 422
    assert value[:128] not in epoch_response.text
    assert len(captured) == 1


@pytest.mark.parametrize("secret", ["metadata", "unavailable", "fail"])
def test_server_projects_provider_recovery_event_contract_with_schema_secret_collision(
    tmp_path,
    secret: str,
) -> None:
    session_id = "provider-recovery-event-projection"
    alias_codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={
                "test": SecretStr(
                    base64.urlsafe_b64encode(bytes(range(32))).decode("ascii").rstrip("=")
                )
            },
        )
    )
    store = SQLiteSessionStore(
        tmp_path / "provider-recovery-event-projection.sqlite3",
        public_authority_alias_codec=alias_codec,
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        required = event_with_runtime_payload_authority(
            Event(
                type=EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED,
                session_id=session_id,
                payload={
                    "provider": "fake",
                    "model": "fake-model",
                    "step": 1,
                    "attempt": 1,
                    "max_attempts": 1,
                    "model_step_id": "model-step-private",
                    "model_attempt_id": "model-attempt-private",
                    "source_run_epoch": 0,
                    "run_epoch": 1,
                    "operation_id": "response-public",
                    "stream_protocol": "responses-v1",
                    "status": "unavailable",
                    "recovery_reason": "unavailable",
                },
            ),
            "model_step_id",
            "model_attempt_id",
            "operation_id",
            "stream_protocol",
        )
        resolved = event_with_runtime_payload_authority(
            Event(
                type=EventType.PROVIDER_OPERATION_RESOLVED,
                session_id=session_id,
                payload={
                    "provider": "fake",
                    "model": "fake-model",
                    "step": 1,
                    "attempt": 1,
                    "max_attempts": 1,
                    "model_step_id": "model-step-private",
                    "model_attempt_id": "model-attempt-private",
                    "source_run_epoch": 0,
                    "run_epoch": 1,
                    "operation_id": "response-public",
                    "stream_protocol": "responses-v1",
                    "status": "unavailable",
                    "stage_id": "stage-private",
                    "resolution_id": "resolution-private",
                    "resolution_action": "fail",
                    "recovery_reason": "unavailable",
                    "duplicate_request_risk": True,
                    "reason": "operator decision",
                    "metadata": {"ticket": "INC-757"},
                    "resolved_by": {
                        "source": "request",
                        "subject": "operator",
                        "tenant": "tenant-a",
                    },
                },
            ),
            "model_step_id",
            "model_attempt_id",
            "operation_id",
            "resolution_id",
            "stage_id",
            "stream_protocol",
        )
        await app._event_writer.emit(required)
        await app._event_writer.emit(resolved)

    try:
        asyncio.run(seed())
        client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
        response = client.get(f"/api/sessions/{session_id}/events?limit=10")

        assert response.status_code == 200
        events = response.json()["events"]
        required_payload = events[0]["payload"]
        resolved_payload = events[1]["payload"]
        assert required_payload["operation_id"] == "response-public"
        assert required_payload["stream_protocol"] == "responses-v1"
        assert required_payload["model_step_id"] == PRIVATE_EVENT_AUTHORITY
        assert required_payload["model_attempt_id"] == PRIVATE_EVENT_AUTHORITY
        assert required_payload["status"] == "unavailable"
        assert required_payload["recovery_reason"] == "unavailable"
        assert resolved_payload["operation_id"] == "response-public"
        assert resolved_payload["stream_protocol"] == "responses-v1"
        assert "stage_id" not in resolved_payload
        assert "resolution_id" not in resolved_payload
        assert resolved_payload["status"] == "unavailable"
        assert resolved_payload["recovery_reason"] == "unavailable"
        assert resolved_payload["resolution_action"] == "fail"
        assert resolved_payload["duplicate_request_risk"] is True
        assert resolved_payload["metadata"] == {"ticket": "INC-757"}
        assert resolved_payload["resolved_by"] == {
            "source": "request",
            "subject": "operator",
            "tenant": "tenant-a",
        }
    finally:
        asyncio.run(store.close())


def test_tool_approval_endpoints_preserve_metadata() -> None:
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

    asyncio.run(create_interrupted_session("session_resolve_metadata"))
    asyncio.run(create_interrupted_session("session_recover_metadata"))

    resolved_requests = []
    recovered_requests = []

    async def resolve_tool_approval(request):
        resolved_requests.append(request)
        yield Event(
            type=EventType.SESSION_RESUMED,
            session_id=request.session_id,
            agent_name="assistant",
        )

    async def recover_tool_approval(request):
        recovered_requests.append(request)
        yield Event(
            type=EventType.SESSION_RESUMED,
            session_id=request.session_id,
            agent_name="assistant",
        )

    app.resolve_tool_approval = resolve_tool_approval
    app.recover_tool_approval = recover_tool_approval
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_resolve_metadata",
            "approval_id": "approval_1",
            "tool_round_id": "round_1",
            "tool_call_id": "call_1",
            "decision": "approve",
            "metadata": {"actor": "operator"},
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    with client.stream(
        "POST",
        "/api/tool-approvals/recover",
        json={
            "session_id": "session_recover_metadata",
            "approval_id": "approval_2",
            "tool_round_id": "round_2",
            "tool_call_id": "call_1",
            "outcome": "completed",
            "message": "confirmed externally",
            "metadata": {"actor": "operator", "source": "dashboard"},
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())

    assert resolved_requests[0].metadata == {"actor": "operator"}
    assert recovered_requests[0].metadata == {
        "actor": "operator",
        "source": "dashboard",
    }

    mismatched_alias = client.post(
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_resolve_metadata",
            "approval_id": public_event_linkage_id(1, "tool_call_id"),
            "tool_round_id": "round_1",
            "tool_call_id": "call_1",
            "decision": "approve",
        },
    )
    assert mismatched_alias.status_code == 409
    assert "field-mismatched" in mismatched_alias.json()["detail"]
    assert len(resolved_requests) == 1


def test_interrupt_session_endpoint_streams_interrupted_event() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_pending_session() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_interrupt_endpoint",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

    asyncio.run(create_pending_session())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/sessions/session_interrupt_endpoint/interrupt",
        json={
            "reason": "operator requested stop",
            "metadata": {"ticket": "incident-42"},
            "requested_by": {
                "subject": "dev-operator",
                "source": "http_auth",
                "claims": {"role": "operator"},
            },
        },
    ) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())

    body = "\n".join(lines)
    assert "session.interrupted" in body
    assert "operator requested stop" in body
    data_line = next(line for line in lines if line.startswith("data: "))
    event = json.loads(data_line.removeprefix("data: "))
    assert event["payload"]["requested_by"] == {
        "subject": "dev-operator",
        "tenant": None,
        "source": "request",
    }
    assert "claims" not in event["payload"]["requested_by"]

    session = asyncio.run(app.session_store.load("session_interrupt_endpoint"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_interrupt_session_endpoint_rejects_completed_session_before_streaming() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_completed_session() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_interrupt_completed",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(
            "session_interrupt_completed", SessionStatus.COMPLETED
        )

    asyncio.run(create_completed_session())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post("/api/sessions/session_interrupt_completed/interrupt")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Session cannot be interrupted from status: completed",
    }


def test_interrupt_session_endpoint_rejects_completion_race_before_streaming() -> None:
    class CompletingRaceStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.loads = 0

        async def load(self, session_id: str):
            self.loads += 1
            if session_id == "session_interrupt_race" and self.loads == 2:
                await self.update_status(session_id, SessionStatus.COMPLETED)
            return await super().load(session_id)

    store = CompletingRaceStore()
    app = CayuApp(session_store=store)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_running_session() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_interrupt_race",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status("session_interrupt_race", SessionStatus.RUNNING)

    asyncio.run(create_running_session())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post("/api/sessions/session_interrupt_race/interrupt")

    assert response.status_code == 409
    assert response.json()["detail"] == "Session cannot be interrupted from status: completed"


def test_interrupt_session_endpoint_returns_conflict_while_interruption_finalizes() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_interrupting_session() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_interrupt_finalizing",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(
            "session_interrupt_finalizing",
            SessionStatus.INTERRUPTING,
        )

    asyncio.run(create_interrupting_session())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post("/api/sessions/session_interrupt_finalizing/interrupt")

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Session interruption is still finalizing: session_interrupt_finalizing",
    }


def test_interrupt_session_endpoint_is_idempotent_for_interrupted_session() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def create_interrupted_session() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_interrupt_idempotent",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(
            "session_interrupt_idempotent", SessionStatus.INTERRUPTED
        )
        await app.session_store.append_event(
            "session_interrupt_idempotent",
            Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id="session_interrupt_idempotent",
                agent_name="assistant",
                payload={"reason": "already interrupted", "metadata": {}},
            ),
        )

    asyncio.run(create_interrupted_session())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/sessions/session_interrupt_idempotent/interrupt",
    ) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())

    body = "\n".join(lines)
    assert "session.interrupted" in body
    assert "already interrupted" in body


def test_server_deletes_session_and_is_idempotent() -> None:
    async def seed(store):
        await _create_session(store, "sess_del")

    _, client = _lifecycle_store_and_client(seed)

    assert client.delete("/api/sessions/sess_del").status_code == 204
    assert client.get("/api/sessions/sess_del").status_code == 404
    # Idempotent: deleting a missing session is still 204.
    assert client.delete("/api/sessions/sess_del").status_code == 204


def test_server_delete_running_session_conflicts() -> None:
    async def seed(store):
        await _create_session(store, "sess_run")
        await store.update_status("sess_run", SessionStatus.RUNNING)

    _, client = _lifecycle_store_and_client(seed)

    response = client.delete("/api/sessions/sess_run")
    assert response.status_code == 409


def test_server_delete_active_durable_operation_conflicts() -> None:
    async def seed(store):
        await _create_session(store, "sess_active_operation")
        expires_at = datetime.now(UTC) + timedelta(minutes=5)
        await store.checkpoint(
            "sess_active_operation",
            {
                "session_operations": {
                    "version": 1,
                    "active_operation_id": "operation-1",
                    "records": {
                        "request-1": {
                            "operation_id": "operation-1",
                            "status": "running",
                            "claim_expires_at": expires_at.isoformat(),
                        }
                    },
                }
            },
        )

    _, client = _lifecycle_store_and_client(seed)

    response = client.delete("/api/sessions/sess_active_operation")
    assert response.status_code == 409
    assert "durable operation operation-1 is active" in response.json()["detail"]


def test_server_updates_session_labels() -> None:
    async def seed(store):
        await _create_session(store, "sess_lab", labels={"team": "research"})

    _, client = _lifecycle_store_and_client(seed)

    response = client.patch("/api/sessions/sess_lab/labels", json={"labels": {"stage": "review"}})
    assert response.status_code == 200
    # Full replacement: the old "team" label is gone.
    assert response.json()["labels"] == {"stage": "review"}
    missing = client.patch("/api/sessions/sess_missing/labels", json={"labels": {}})
    assert missing.status_code == 404
    # An invalid label (blank value) is a client error (422), not an unhandled 500.
    invalid = client.patch("/api/sessions/sess_lab/labels", json={"labels": {"k": "   "}})
    assert invalid.status_code == 422
    non_durable = client.patch(
        "/api/sessions/sess_lab/labels", json={"labels": {"stage": "review\x00hidden"}}
    )
    assert non_durable.status_code == 422
    # A typo'd key must 422 (extra="forbid"), NOT silently replace all labels with {}.
    typo = client.patch("/api/sessions/sess_lab/labels", json={"lables": {"a": "b"}})
    assert typo.status_code == 422
    # A missing required field must 422, not default to an empty (wiping) replacement.
    empty_body = client.patch("/api/sessions/sess_lab/labels", json={})
    assert empty_body.status_code == 422
    # The labels were not wiped by any of the rejected requests.
    assert client.get("/api/sessions/sess_lab").json()["labels"] == {"stage": "review"}


def test_server_updates_session_metadata() -> None:
    async def seed(store):
        await _create_session(
            store,
            "sess_meta",
            metadata={
                "a": 1,
                "subagent": {"mode": "background"},
                "cayu:taint_labels": ["untrusted"],
            },
        )

    _, client = _lifecycle_store_and_client(seed)

    response = client.patch("/api/sessions/sess_meta/metadata", json={"metadata": {"b": [1, 2]}})
    assert response.status_code == 200
    expected = {
        "b": [1, 2],
        "subagent": {"mode": "background"},
        "cayu:taint_labels": ["untrusted"],
    }
    assert response.json()["metadata"] == expected
    missing = client.patch("/api/sessions/sess_missing/metadata", json={"metadata": {}})
    assert missing.status_code == 404
    # Typo'd key / missing field must 422, never silently wipe metadata.
    assert client.patch("/api/sessions/sess_meta/metadata", json={"metadat": {}}).status_code == 422
    assert client.patch("/api/sessions/sess_meta/metadata", json={}).status_code == 422
    assert (
        client.patch(
            "/api/sessions/sess_meta/metadata",
            json={"metadata": {"subagent": {}}},
        ).status_code
        == 422
    )
    assert (
        client.patch(
            "/api/sessions/sess_meta/metadata",
            json={"metadata": {"cayu:taint_labels": []}},
        ).status_code
        == 422
    )
    assert (
        client.patch(
            "/api/sessions/sess_meta/metadata", json={"metadata": {"nested": ["value\x00"]}}
        ).status_code
        == 422
    )
    assert (
        client.patch(
            "/api/sessions/sess_meta/metadata",
            content=b'{"metadata":{"nested":"\\ud800"}}',
            headers={"content-type": "application/json"},
        ).status_code
        == 422
    )
    assert client.get("/api/sessions/sess_meta").json()["metadata"] == expected


def test_explicit_compaction_endpoint_uses_replayable_mutation_contract() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=TranscriptDigestCompactor(),
            max_user_turns=1,
        ),
    )

    async def prepare() -> tuple[int, int]:
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-explicit-compact-endpoint",
                    messages=[Message.text("user", "create only")],
                )
            )
        ]
        session = await app.session_store.load("session-explicit-compact-endpoint")
        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await app.session_store.append_transcript_messages(session.id, transcript)
        persisted_transcript = await app.session_store.load_transcript(session.id)
        return session.run_epoch, len(persisted_transcript)

    run_epoch, transcript_cursor = asyncio.run(prepare())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    session_id = "session-explicit-compact-endpoint"
    body = {
        "idempotency_key": "compact-endpoint-1",
        "expected_run_epoch": run_epoch,
        "expected_transcript_cursor": transcript_cursor,
        "instructions": "Keep decisions.",
        "requested_by": {"subject": "operator@example.com"},
    }
    with client.stream(
        "POST",
        f"/api/sessions/{session_id}/compact",
        json=body,
        headers={"Cayu-Mutation-ID": "mutation-compact-1"},
    ) as response:
        assert response.status_code == 200
        frames = [frame for frame in _sse_frames(response) if "data" in frame]

    event_types = [frame["data"]["type"] for frame in frames]
    assert event_types[:3] == [
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.SESSION_CHECKPOINTED,
    ]
    accepted_events = client.get(
        f"/api/sessions/{session_id}/events",
        params={"event_type": EventType.SERVER_MUTATION_ACCEPTED},
    ).json()["events"]
    assert len(accepted_events) == 1
    accepted = accepted_events[0]
    assert accepted["payload"] == {
        "mutation_id": "mutation-compact-1",
        "mutation_kind": "session.compact",
        "accepted_event_id": frames[0]["data"]["id"],
        "accepted_event_sequence": public_event_sequence(frames[0]["data"]["id"]),
        "accepted_event_type": EventType.CONTEXT_COMPACTION_STARTED,
    }
    assert frames[0]["data"]["payload"]["actor"] == {
        "subject": "operator@example.com",
        "tenant": None,
        "source": "request",
    }
    assert "/api/sessions/{session_id}/compact" in client.get("/openapi.json").json()["paths"]


@pytest.mark.parametrize(
    ("path", "idempotency_key", "location"),
    [
        ("/api/sessions/missing/compact", "invalid-\x00key", ["body", "idempotency_key"]),
        ("/api/sessions/invalid%00id/compact", "compact-1", ["path", "session_id"]),
    ],
)
def test_explicit_compaction_endpoint_rejects_unpersistable_identifiers(
    path: str,
    idempotency_key: str,
    location: list[str],
) -> None:
    client = TestClient(create_server(CayuApp(enable_logging=False), config=_LOCAL_SERVER_CONFIG))

    response = client.post(
        path,
        json={
            "idempotency_key": idempotency_key,
            "expected_run_epoch": 0,
            "expected_transcript_cursor": 0,
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == location


def test_explicit_compaction_endpoint_rejects_unpersistable_nested_text() -> None:
    client = TestClient(create_server(CayuApp(enable_logging=False), config=_LOCAL_SERVER_CONFIG))

    response = client.post(
        "/api/sessions/missing/compact",
        json={
            "idempotency_key": "compact-1",
            "expected_run_epoch": 0,
            "expected_transcript_cursor": 0,
            "instructions": "invalid-\x00instructions",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body"]


def test_sse_replay_preserves_canonical_policy_denial_attribution() -> None:
    session_id = "session_policy_denial_replay"
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "push")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_events(
            session_id,
            [
                Event(
                    id="event_tool_started",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="exec_command",
                    payload={"tool_call_id": "call_1"},
                ),
                Event(
                    id="event_tool_blocked",
                    type=EventType.TOOL_CALL_BLOCKED,
                    session_id=session_id,
                    tool_name="exec_command",
                    payload={
                        "tool_name": "exec_command",
                        "tool_call_id": "call_1",
                        "tool_round_id": "round_1",
                        "idempotency_key": "cayu-tool:v1:call_1",
                        "denied_by": "command_policy",
                        "decision": "deny",
                        "reason": "Remote mutation is not allowed.",
                        "result": {
                            "content": "Command denied by policy.",
                            "structured": {"error": "command_denied"},
                            "artifacts": [],
                            "is_error": True,
                        },
                    },
                ),
                Event(
                    id="event_session_completed",
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                ),
            ],
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "ignored during replay", "session_id": session_id},
        headers={"Last-Event-ID": f"{session_id}:"},
    ) as response:
        assert response.status_code == 200
        frames = [frame for frame in _sse_frames(response) if "data" in frame]

    blocked = next(
        frame["data"] for frame in frames if frame["data"]["type"] == "tool.call.blocked"
    )
    assert blocked["payload"]["denied_by"] == "command_policy"
    assert blocked["payload"]["decision"] == "deny"
    assert blocked["payload"]["tool_name"] == "exec_command"
    assert [frame["data"]["id"] for frame in frames] == [
        public_event_id(1),
        public_event_id(2),
        public_event_id(3),
    ]


def _authorized_message_endpoint():
    from cayu.server import AuthContext
    from cayu.sessions.messaging import SessionMessageAccessPolicy

    class MessagePolicy(SessionMessageAccessPolicy):
        def authorize(self, context, *, session_id, session_instance_id, action):
            return (
                context.subject == "operator@example.com"
                and session_id == "session-message-endpoint"
                and action == "enqueue"
            )

    async def authenticate(_request):
        return AuthContext(subject="operator@example.com")

    app = CayuApp(session_message_access_policy=MessagePolicy(), enable_logging=False)
    return app, TestClient(create_server(app, config=ServerConfig.protected(authenticate)))


def test_enqueue_session_message_endpoint_uses_replayable_mutation_contract() -> None:
    app, client = _authorized_message_endpoint()

    async def prepare() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session-message-endpoint",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

    asyncio.run(prepare())
    session_id = "session-message-endpoint"
    with client.stream(
        "POST",
        f"/api/sessions/{session_id}/messages",
        json={
            "idempotency_key": "message-endpoint-1",
            "content": "Please prioritize the failing deployment.",
            "delivery_mode": "next_turn",
        },
        headers={"Cayu-Mutation-ID": "mutation-message-1"},
    ) as response:
        assert response.status_code == 200
        frames = [frame for frame in _sse_frames(response) if "data" in frame]

    queued = frames[0]["data"]
    assert queued["type"] == EventType.SESSION_MESSAGE_QUEUED
    assert queued["payload"]["delivery_mode"] == "next_turn"
    assert queued["payload"]["actor"] == {
        "subject": "operator@example.com",
        "tenant": None,
        "source": "http_auth",
    }
    assert "content" not in queued["payload"]

    persisted_events = client.get(f"/api/sessions/{session_id}/events").json()["events"]
    persisted_queued = next(
        event for event in persisted_events if event["type"] == EventType.SESSION_MESSAGE_QUEUED
    )
    assert persisted_queued["id"] == queued["id"]
    accepted = next(
        event for event in persisted_events if event["type"] == EventType.SERVER_MUTATION_ACCEPTED
    )
    assert accepted["payload"] == {
        "mutation_id": "mutation-message-1",
        "mutation_kind": "session.message.enqueue",
        "accepted_event_id": queued["id"],
        "accepted_event_sequence": public_event_sequence(queued["id"]),
        "accepted_event_type": EventType.SESSION_MESSAGE_QUEUED,
    }
    assert "/api/sessions/{session_id}/messages" in client.app.openapi()["paths"]


def test_enqueue_session_message_endpoint_rejects_nonportable_text() -> None:
    _, client = _authorized_message_endpoint()

    response = client.post(
        "/api/sessions/session_1/messages",
        json={
            "idempotency_key": "message-1",
            "content": "hello\u0000",
            "delivery_mode": "next_turn",
        },
    )

    assert response.status_code == 422


class AskUserProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests += 1
        if self.requests == 1:
            yield ModelStreamEvent.tool_call(
                id="call_1", name="ask_user", arguments={"question": "which env?"}
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
        else:
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _ask_user_client() -> TestClient:
    app = CayuApp()
    app.register_provider(AskUserProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[UserInputTool()])
    return TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))


def test_server_resolve_user_input_resumes_paused_session() -> None:
    client = _ask_user_client()
    run_events = _sse_events(client, "/api/run", {"prompt": "deploy"})
    awaiting = next(e for e in run_events if e["type"] == "session.awaiting_user_input")
    session_id = awaiting["session_id"]
    input_id = awaiting["payload"]["input_id"]

    resolved = _sse_events(
        client,
        "/api/user-input/resolve",
        {"session_id": session_id, "input_id": input_id, "answer": "staging"},
    )
    assert resolved[-1]["type"] == "session.completed"
    tool_completed = next(
        e for e in resolved if e["type"] == "tool.call.completed" and e["tool_name"] == "ask_user"
    )
    assert tool_completed["payload"]["result"]["content"] == "staging"


def test_server_resolve_user_input_unknown_session_returns_404() -> None:
    client = _ask_user_client()
    response = client.post(
        "/api/user-input/resolve",
        json={"session_id": "missing", "input_id": "x", "answer": "y"},
    )
    assert response.status_code == 404


def test_server_recover_user_input_route_is_registered() -> None:
    client = _ask_user_client()
    # Unknown session → 404 (route exists and validates the session before streaming).
    response = client.post(
        "/api/user-input/recover",
        json={
            "session_id": "missing",
            "input_id": "x",
            "answer": "y",
            "tool_call_id": "call_1",
            "outcome": "completed",
            "message": "recovered",
        },
    )
    assert response.status_code == 404


def test_server_recover_tool_round_route_is_registered() -> None:
    client = _ask_user_client()
    # Unknown session → 404 (route exists and validates the session before streaming).
    response = client.post(
        "/api/tool-rounds/recover",
        json={
            "session_id": "missing",
            "round_id": "round_1",
            "tool_call_id": "call_1",
            "outcome": "completed",
            "message": "recovered",
        },
    )
    assert response.status_code == 404
