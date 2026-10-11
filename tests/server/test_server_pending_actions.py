"""HTTP server pending actions contracts."""

from __future__ import annotations

# ruff: noqa: E402
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
httpx = pytest.importorskip("httpx")

import asyncio
from unittest.mock import patch

from fastapi.testclient import TestClient
from tests.server.server_test_support import _LOCAL_SERVER_CONFIG

from cayu.applications import CayuApp
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY, public_event_linkage_id
from cayu.server import create_server
from cayu.server.routes import _serialize_pending_action
from cayu.sessions import _checkpoint_preservation as checkpoint_preservation
from cayu.sessions.checkpoints import CURRENT_CHECKPOINT_SCHEMA_VERSION
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.pending_action_contracts import PendingActionListResult, PendingActionQuery
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest


def test_server_pending_actions_lists_blocking_session_work() -> None:
    app = CayuApp()
    private_approval_reason = "pending-approval-policy-secret-canary"
    approval_round_id = f"tround_{'3' * 32}"
    user_input_round_id = f"tround_{'4' * 32}"
    recovery_round_id = f"tround_{'5' * 32}"

    def execution_identity(tool_round_id: str) -> dict[str, str]:
        return {
            "model_step_id": f"mstep_{'1' * 32}",
            "model_attempt_id": f"matt_{'2' * 32}",
            "tool_round_id": tool_round_id,
        }

    def pending_tool_call(tool_call_id: str, tool_name: str) -> dict[str, object]:
        return {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": {},
            "policy_decision": None,
            "reason": None,
            "metadata": {},
            "active_taint_labels": [],
        }

    def approval_checkpoint(
        *,
        approval_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, object] | None = None,
        reason: str | None = None,
    ) -> dict[str, object]:
        pending_call = {
            **pending_tool_call(tool_call_id, tool_name),
            "arguments": arguments or {},
            "policy_decision": "require_approval",
        }
        return {
            "pending_tool_approval": {
                **execution_identity(approval_round_id),
                "approval_id": approval_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": arguments or {},
                "agent_name": "assistant",
                "publish_arguments": True,
                "reason": reason,
                "tool_calls": [pending_call],
            },
            "pending_tool_round": {
                **execution_identity(approval_round_id),
                "agent_name": "assistant",
                "tool_calls": [pending_call],
                "policy_state": "planned",
                "policy_context_version": 1,
            },
        }

    def user_input_checkpoint(
        *,
        input_id: str,
        tool_call_id: str,
        tool_name: str,
        question: str,
        options: list[str],
    ) -> dict[str, object]:
        return {
            "pending_user_input": {
                **execution_identity(user_input_round_id),
                "input_id": input_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "question": question,
                "options": options,
                "arguments": {},
                "agent_name": "assistant",
                "tool_calls": [pending_tool_call(tool_call_id, tool_name)],
            }
        }

    def tool_round_checkpoint(
        *,
        round_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "pending_tool_round": {
                **execution_identity(round_id),
                "agent_name": "assistant",
                "tool_calls": [
                    {
                        **pending_tool_call(tool_call_id, tool_name),
                        "arguments": arguments or {},
                    }
                ],
            }
        }

    async def create(
        session_id: str,
        status: SessionStatus,
        events: list[Event],
        checkpoint: dict[str, object] | None = None,
    ) -> None:
        created = await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(session_id, status)
        await app.session_store.append_events(session_id, events)
        if checkpoint is not None:
            pending_user_input = checkpoint.get("pending_user_input")
            if type(pending_user_input) is dict:
                pending_user_input.setdefault("session_id", session_id)
                pending_user_input.setdefault("session_instance_id", created.instance_id)
                pending_user_input.setdefault(
                    "source_interaction_id",
                    f"interaction:{session_id}",
                )
                pending_user_input.setdefault("source_run_epoch", created.run_epoch)
                pending_user_input.setdefault(
                    "execution_profile_fingerprint",
                    "e" * 64,
                )
            await app.session_store.checkpoint(session_id, checkpoint)

    async def seed() -> None:
        await create(
            "pending_approval",
            SessionStatus.INTERRUPTED,
            [
                Event(
                    id="approval_requested",
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id="pending_approval",
                    agent_name="assistant",
                    tool_name="deploy",
                    payload={
                        **execution_identity(approval_round_id),
                        "approval_id": "approval_1",
                        "tool_call_id": "call_deploy",
                        "approval": {
                            **execution_identity(approval_round_id),
                            "approval_id": "approval_1",
                            "tool_call_id": "call_deploy",
                            "tool_name": "deploy",
                            "reason": private_approval_reason,
                            "arguments": {"service": "api"},
                            "agent_name": "assistant",
                            "tool_calls": [
                                {
                                    **pending_tool_call("call_deploy", "deploy"),
                                    "arguments": {"service": "api"},
                                    "policy_decision": "require_approval",
                                }
                            ],
                        },
                    },
                )
            ],
            checkpoint=approval_checkpoint(
                approval_id="approval_1",
                tool_call_id="call_deploy",
                tool_name="deploy",
                arguments={"service": "api"},
                reason=private_approval_reason,
            ),
        )
        await create(
            "pending_user_input",
            SessionStatus.INTERRUPTED,
            [
                Event(
                    id="awaiting_user_input",
                    type=EventType.SESSION_AWAITING_USER_INPUT,
                    session_id="pending_user_input",
                    payload={
                        **execution_identity(user_input_round_id),
                        "input_id": "input_1",
                        "tool_call_id": "call_ask",
                        "question": "pending-user-input-secret-canary",
                        "options": ["yes", "pending-user-input-secret-canary"],
                    },
                )
            ],
            checkpoint=user_input_checkpoint(
                input_id="input_1",
                tool_call_id="call_ask",
                tool_name="ask_user",
                question="pending-user-input-secret-canary",
                options=["yes", "pending-user-input-secret-canary"],
            ),
        )
        await create(
            "manual_recovery",
            SessionStatus.INTERRUPTED,
            [
                Event(
                    id="manual_recovery_event",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="manual_recovery",
                    payload={
                        **execution_identity(approval_round_id),
                        "interruption_type": "tool_approval_required",
                        "manual_recovery_required": True,
                        "approval_id": "approval_2",
                        "tool_call_id": "call_refund",
                        "tool_name": "refund",
                        "error": "tool outcome unknown",
                    },
                )
            ],
            checkpoint=approval_checkpoint(
                approval_id="approval_2",
                tool_call_id="call_refund",
                tool_name="refund",
                arguments={"invoice_id": "inv_123"},
            ),
        )
        await create(
            "manual_tool_round_recovery",
            SessionStatus.FAILED,
            [
                Event(
                    id="manual_tool_round_started",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="manual_tool_round_recovery",
                    tool_name="charge_card",
                    payload={
                        **execution_identity(recovery_round_id),
                        "tool_call_id": "call_charge",
                    },
                ),
                Event(
                    id="manual_tool_round_recovery_event",
                    type=EventType.SESSION_FAILED,
                    session_id="manual_tool_round_recovery",
                    payload={
                        **execution_identity(recovery_round_id),
                        "interruption_type": "runtime_interrupted",
                        "manual_recovery_required": True,
                        "tool_call_id": "call_charge",
                        "tool_name": "charge_card",
                        "error": "tool outcome unknown",
                    },
                ),
            ],
            checkpoint=tool_round_checkpoint(
                round_id=recovery_round_id,
                tool_call_id="call_charge",
                tool_name="charge_card",
                arguments={"amount": 42},
            ),
        )
        await create(
            "missing_checkpoint",
            SessionStatus.INTERRUPTED,
            [
                Event(
                    id="missing_checkpoint_event",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="missing_checkpoint",
                    payload={
                        "interruption_type": "tool_approval_required",
                        "manual_recovery_required": True,
                        "approval_id": "approval_missing",
                        "tool_call_id": "call_missing",
                        "tool_name": "refund",
                        "error": "tool outcome unknown",
                    },
                )
            ],
        )
        await create(
            "resumed_approval",
            SessionStatus.INTERRUPTED,
            [
                Event(
                    id="old_approval",
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id="resumed_approval",
                    payload={"approval": {"approval_id": "old_approval", "tool_name": "deploy"}},
                ),
                Event(
                    id="resumed_after_old_approval",
                    type=EventType.SESSION_RESUMED,
                    session_id="resumed_approval",
                ),
            ],
        )

    asyncio.run(seed())
    recovered_rounds = []

    async def recover_tool_round(request):
        recovered_rounds.append(request)
        yield Event(
            type=EventType.SESSION_RESUMED,
            session_id=request.session_id,
        )

    app.recover_tool_round = recover_tool_round
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with patch.object(
        app.session_store,
        "query_events",
        side_effect=AssertionError("pending-action serialization must use its bounded event"),
    ):
        response = client.get("/api/pending-actions")
    assert response.status_code == 200
    body = response.json()
    assert private_approval_reason not in response.text
    assert body["inspected_candidate_count"] == 4
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert body["total_count"] == 4
    actions_by_session = {action["session"]["id"]: action for action in body["actions"]}
    assert set(actions_by_session) == {
        "manual_recovery",
        "manual_tool_round_recovery",
        "pending_user_input",
        "pending_approval",
    }
    approval = actions_by_session["pending_approval"]
    assert approval["kind"] == "tool_approval"
    assert approval["approval_id"] == public_event_linkage_id(
        approval["event"]["sequence"],
        "approval_id",
    )
    assert approval["arguments"] is None
    assert approval["session"]["runtime_build_provenance"]["availability"] == "unavailable"
    user_input = actions_by_session["pending_user_input"]
    assert user_input["kind"] == "user_input"
    assert user_input["input_id"] == public_event_linkage_id(
        user_input["event"]["sequence"],
        "input_id",
    )
    assert user_input["detail"] == "Input required"
    assert user_input["question"] is None
    assert user_input["options"] == []
    assert "pending-user-input-secret-canary" not in response.text
    tool_round = actions_by_session["manual_tool_round_recovery"]
    assert tool_round["kind"] == "manual_recovery"
    assert tool_round["round_id"] == public_event_linkage_id(
        tool_round["event"]["sequence"],
        "tool_round_id",
    )
    assert tool_round["tool_call_id"] == public_event_linkage_id(
        tool_round["event"]["sequence"],
        "tool_call_id",
    )
    assert tool_round["approval_id"] is None
    assert tool_round["input_id"] is None
    assert tool_round["arguments"] is None
    assert tool_round["event"]["type"] == EventType.SESSION_FAILED
    assert tool_round["event"]["payload"] == {}

    pending_approval = asyncio.run(
        app._runtime_session_store.query_pending_actions(
            PendingActionQuery(session_id="pending_approval", limit=1)
        )
    ).actions[0]
    checkpoint_only_approval = pending_approval.model_copy(
        update={
            "source_linkage": {},
            "event": pending_approval.event.model_copy(
                update={
                    "event": pending_approval.event.event.model_copy(update={"payload": {}}),
                },
                deep=True,
            ),
        },
        deep=True,
    )
    checkpoint_only_payload = _serialize_pending_action(app, checkpoint_only_approval)
    assert checkpoint_only_payload["approval_id"] is None
    assert checkpoint_only_payload["round_id"] is None
    assert checkpoint_only_payload["tool_call_id"] is None

    with patch.object(
        app.session_store,
        "query_events",
        side_effect=AssertionError("pending-action serialization must not query per action"),
    ):
        full_page = [_serialize_pending_action(app, pending_approval) for _ in range(200)]
        assert len(full_page) == 200

    contradictory_approval = pending_approval.model_copy(
        update={"approval_id": "different-approval"},
        deep=True,
    )
    with pytest.raises(RuntimeError, match="disagrees with its durable event authority"):
        _serialize_pending_action(app, contradictory_approval)
    approval_recovery = actions_by_session["manual_recovery"]
    assert approval_recovery["kind"] == "manual_recovery"
    assert approval_recovery["arguments"] is None

    filtered = client.get("/api/pending-actions?kind=user_input&q=input")
    assert filtered.status_code == 200
    filtered_body = filtered.json()
    assert filtered_body["total_count"] == 1
    assert filtered_body["actions"][0]["session"]["id"] == "pending_user_input"

    exact = client.get("/api/pending-actions?session_id=manual_recovery")
    assert exact.status_code == 200
    exact_body = exact.json()
    assert exact_body["inspected_candidate_count"] == 1
    assert exact_body["total_count"] == 1
    assert exact_body["actions"][0]["kind"] == "manual_recovery"

    tool_round_exact = client.get("/api/pending-actions?session_id=manual_tool_round_recovery")
    assert tool_round_exact.status_code == 200
    tool_round_exact_body = tool_round_exact.json()
    assert tool_round_exact_body["inspected_candidate_count"] == 1
    assert tool_round_exact_body["total_count"] == 1
    exact_tool_round = tool_round_exact_body["actions"][0]
    assert exact_tool_round["round_id"] == public_event_linkage_id(
        exact_tool_round["event"]["sequence"],
        "tool_round_id",
    )

    with client.stream(
        "POST",
        "/api/tool-rounds/recover",
        json={
            "session_id": "manual_tool_round_recovery",
            "round_id": exact_tool_round["round_id"],
            "tool_call_id": exact_tool_round["tool_call_id"],
            "outcome": "completed",
            "message": "verified externally",
        },
    ) as recovery_response:
        assert recovery_response.status_code == 200
        list(recovery_response.iter_lines())

    assert len(recovered_rounds) == 1
    assert recovered_rounds[0].round_id == recovery_round_id
    assert recovered_rounds[0].tool_call_id == "call_charge"

    history = client.get("/api/sessions/manual_tool_round_recovery/events").json()["events"]
    failed = next(event for event in history if event["type"] == EventType.SESSION_FAILED)
    assert failed["payload"]["model_step_id"] == PRIVATE_EVENT_AUTHORITY
    assert failed["payload"]["model_attempt_id"] == PRIVATE_EVENT_AUTHORITY
    assert failed["payload"]["tool_round_id"] == tool_round["round_id"]
    assert failed["payload"]["tool_call_id"] == tool_round["tool_call_id"]
    assert recovery_round_id not in repr(failed)

    stale_exact = client.get("/api/pending-actions?session_id=missing_checkpoint")
    assert stale_exact.status_code == 200
    stale_body = stale_exact.json()
    assert stale_body["inspected_candidate_count"] == 0
    assert stale_body["total_count"] == 0


def test_server_pending_actions_uses_one_store_native_query() -> None:
    class PendingActionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.pending_query_count = 0

        async def query_pending_actions(self, query=None, *, checkpoint_root_guard=None):
            del checkpoint_root_guard
            self.pending_query_count += 1
            return PendingActionListResult()

        async def list_sessions(self, query=None):
            raise AssertionError("pending-action route must not list candidate sessions")

        async def query_events(self, query=None):
            raise AssertionError("pending-action route must not query per-session events")

        async def load_checkpoint(self, session_id: str):
            raise AssertionError("pending-action route must not load per-session checkpoints")

    store = PendingActionStore()
    client = TestClient(create_server(CayuApp(session_store=store), config=_LOCAL_SERVER_CONFIG))

    response = client.get("/api/pending-actions")

    assert response.status_code == 200
    assert response.json() == {
        "actions": [],
        "issues": [],
        "next_cursor": None,
        "has_more": False,
        "total_count": None,
        "inspected_candidate_count": 0,
        "execution_profiles": None,
    }
    assert store.pending_query_count == 1


def test_server_pending_actions_returns_413_for_oversized_page() -> None:
    class OversizedPendingActionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def query_pending_actions(self, query=None, *, checkpoint_root_guard=None):
            del checkpoint_root_guard
            from cayu.sessions.pending_action_contracts import PendingActionResultTooLarge

            raise PendingActionResultTooLarge(2 * 1024 * 1024)

    client = TestClient(
        create_server(
            CayuApp(session_store=OversizedPendingActionStore()), config=_LOCAL_SERVER_CONFIG
        )
    )

    response = client.get("/api/pending-actions")

    assert response.status_code == 413
    assert "2097152-byte result limit" in response.json()["detail"]


def test_server_pending_actions_reports_future_checkpoint_without_exposing_contents() -> None:
    app = CayuApp()
    session_id = "future_checkpoint"

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)
        with checkpoint_preservation._invocation_lifecycle_authority_mutation_scope():
            await app.session_store.checkpoint(
                session_id,
                {
                    "checkpoint_schema_version": CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
                    "pending_user_input": {"secret": "must-not-escape"},
                },
            )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.get("/api/pending-actions")

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "checkpoint_kind": "root",
        "observed_version": CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
        "reason": "checkpoint_schema_version_too_new",
        "recovery_disposition": "cannot_migrate",
        "resumable_in_place": False,
        "session_id": session_id,
        "supported_max_version": CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "supported_min_version": 1,
    }
    assert "must-not-escape" not in response.text
    error_schema = client.get("/openapi.json").json()["paths"]["/api/pending-actions"]["get"][
        "responses"
    ]["409"]["content"]["application/json"]["schema"]
    assert error_schema == {"$ref": "#/components/schemas/CheckpointCompatibilityErrorResponse"}


def test_server_pending_actions_rejects_invalid_cursor_as_400() -> None:
    client = TestClient(create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG))

    response = client.get("/api/pending-actions?cursor=not-a-cursor")

    assert response.status_code == 400
    assert "Invalid session cursor" in response.json()["detail"]


def test_server_pending_actions_does_not_misclassify_store_failure_as_400() -> None:
    class FailingPendingActionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def query_pending_actions(self, query=None, *, checkpoint_root_guard=None):
            del checkpoint_root_guard
            raise ValueError("persisted pending-action projection is corrupt")

    client = TestClient(
        create_server(
            CayuApp(session_store=FailingPendingActionStore()), config=_LOCAL_SERVER_CONFIG
        ),
        raise_server_exceptions=False,
    )

    response = client.get("/api/pending-actions")

    assert response.status_code == 500
