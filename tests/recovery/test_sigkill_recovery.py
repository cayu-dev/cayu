from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from uuid import uuid4

import pytest
from worker_harness import BackendConfig, RecoveryHarness

from cayu.core import EventType, ToolResultPart
from cayu.runtime import SessionStatus, TaskStatus

pytestmark = [
    pytest.mark.sigkill_recovery,
    pytest.mark.skipif(
        os.name != "posix" or not hasattr(signal, "SIGKILL"),
        reason="real SIGKILL recovery tests require a POSIX host",
    ),
]


def _postgres_recovery_requested() -> bool:
    required = os.environ.get("CAYU_REQUIRE_POSTGRES", "").strip().lower()
    return bool(os.environ.get("CAYU_TEST_POSTGRES_DSN")) or required in {
        "1",
        "true",
        "yes",
        "on",
    }


@pytest.fixture(
    params=[
        pytest.param("sqlite", id="sqlite"),
        pytest.param("postgres", id="postgres", marks=pytest.mark.postgres_recovery),
    ]
)
def recovery_backend(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> BackendConfig:
    if request.param == "sqlite":
        return BackendConfig.sqlite(tmp_path)
    if not _postgres_recovery_requested():
        pytest.skip("Postgres SIGKILL recovery runs in the required-Postgres lane")
    return BackendConfig.postgres(request.getfixturevalue("postgres_dsn"))


def test_worker_failure_reports_durable_state_and_cleans_control_artifacts(tmp_path) -> None:
    backend = BackendConfig.sqlite(tmp_path)

    with (
        pytest.raises(AssertionError) as captured,
        RecoveryHarness(tmp_path, backend) as harness,
    ):
        worker = harness.launch(
            scenario="ordinary_tool",
            action="automatic",
            session_id="missing-session",
        )
        worker.wait_for_phase("unreachable")

    message = str(captured.value)
    assert "durable state:" in message
    assert '"session": null' in message
    assert '"checkpoint": {}' in message
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("recovery_action", ["automatic", "manual"])
def test_sigkill_during_ordinary_tool_execution_recovers_without_reexecution(
    tmp_path,
    recovery_action: str,
) -> None:
    backend = BackendConfig.sqlite(tmp_path)
    session_id = f"sigkill_tool_{recovery_action}"

    with RecoveryHarness(tmp_path, backend) as harness:
        killed_worker = harness.launch(
            scenario="ordinary_tool",
            action="start",
            session_id=session_id,
        )
        phase = killed_worker.wait_for_phase("tool_side_effect_recorded")

        pre_kill = asyncio.run(harness.load_session_state(session_id))
        assert pre_kill.session is not None
        assert pre_kill.session.status == SessionStatus.RUNNING
        assert "pending_tool_round" in pre_kill.checkpoint
        assert [message.role for message in pre_kill.transcript] == ["user", "assistant"]
        started = [event for event in pre_kill.events if event.type == EventType.TOOL_CALL_STARTED]
        assert len(started) == 1
        assert not any(
            event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
            for event in pre_kill.events
        )
        marker = harness.read_marker()
        assert marker == [
            {
                "idempotency_key": started[0].payload["idempotency_key"],
                "session_id": session_id,
                "tool_call_id": "call_side_effect",
            }
        ]
        assert phase["idempotency_key"] == started[0].payload["idempotency_key"]

        killed_worker.sigkill()

        recovery_worker = harness.launch(
            scenario="ordinary_tool",
            action=recovery_action,
            session_id=session_id,
        )
        recovery_worker.wait_success()

        recovered = asyncio.run(harness.load_session_state(session_id))
        assert recovered.session is not None
        assert recovered.session.status == SessionStatus.COMPLETED
        assert recovered.checkpoint == {}
        assert harness.read_marker() == marker

        started = [event for event in recovered.events if event.type == EventType.TOOL_CALL_STARTED]
        terminals = [
            event
            for event in recovered.events
            if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
            and event.payload.get("tool_call_id") == "call_side_effect"
        ]
        assert len(started) == 1
        assert len(terminals) == 1
        assert terminals[0].payload["idempotency_key"] == marker[0]["idempotency_key"]

        if recovery_action == "automatic":
            assert terminals[0].type == EventType.TOOL_CALL_FAILED
            assert terminals[0].payload["recovered"] is True
            assert terminals[0].payload["result"]["structured"]["outcome_unknown"] is True
        else:
            assert terminals[0].type == EventType.TOOL_CALL_COMPLETED
            assert terminals[0].payload["manual_recovery"] is True
            assert terminals[0].payload["result"]["content"] == (
                "External marker verified the side effect."
            )

        tool_results = [
            part
            for message in recovered.transcript
            for part in message.content
            if isinstance(part, ToolResultPart) and part.tool_call_id == "call_side_effect"
        ]
        assert len(tool_results) == 1

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("decision", ["approve", "deny"])
def test_sigkill_after_durable_approval_request_preserves_resolution(
    tmp_path,
    decision: str,
) -> None:
    backend = BackendConfig.sqlite(tmp_path)
    session_id = f"sigkill_approval_{decision}"

    with RecoveryHarness(tmp_path, backend) as harness:
        killed_worker = harness.launch(
            scenario="approval",
            action="start",
            session_id=session_id,
        )
        phase = killed_worker.wait_for_phase("approval_requested_persisted")

        pre_kill = asyncio.run(harness.load_session_state(session_id))
        assert pre_kill.session is not None
        assert pre_kill.session.status == SessionStatus.RUNNING
        pending = pre_kill.checkpoint["pending_tool_approval"]
        assert pending["approval_id"] == phase["approval_id"]
        assert [event.type for event in pre_kill.events][-2:] == [
            EventType.SESSION_CHECKPOINTED,
            EventType.TOOL_CALL_APPROVAL_REQUESTED,
        ]
        assert EventType.SESSION_INTERRUPTED not in [event.type for event in pre_kill.events]
        assert harness.read_marker() == []

        killed_worker.sigkill()

        recovery_worker = harness.launch(
            scenario="approval",
            action=decision,
            session_id=session_id,
        )
        result = recovery_worker.wait_success()
        assert result["retry_rejected"] is True

        recovered = asyncio.run(harness.load_session_state(session_id))
        assert recovered.session is not None
        assert recovered.session.status == SessionStatus.COMPLETED
        assert recovered.checkpoint == {}
        assert len(harness.read_marker()) == (1 if decision == "approve" else 0)

        event_types = [event.type for event in recovered.events]
        checkpoint_index = event_types.index(EventType.SESSION_CHECKPOINTED)
        requested_index = event_types.index(EventType.TOOL_CALL_APPROVAL_REQUESTED)
        interrupted_index = event_types.index(EventType.SESSION_INTERRUPTED)
        resumed_index = event_types.index(EventType.SESSION_RESUMED)
        decision_type = (
            EventType.TOOL_CALL_APPROVED
            if decision == "approve"
            else EventType.TOOL_CALL_APPROVAL_DENIED
        )
        decision_index = event_types.index(decision_type)
        completed_index = event_types.index(EventType.SESSION_COMPLETED)
        assert (
            checkpoint_index
            < requested_index
            < interrupted_index
            < resumed_index
            < decision_index
            < completed_index
        )

        decision_events = [event for event in recovered.events if event.type == decision_type]
        assert len(decision_events) == 1
        assert event_types.count(EventType.SESSION_CHECKPOINTED) == 2
        for event_type in (
            EventType.TOOL_CALL_APPROVAL_REQUESTED,
            EventType.SESSION_INTERRUPTED,
            EventType.SESSION_RESUMED,
            EventType.SESSION_COMPLETED,
        ):
            assert event_types.count(event_type) == 1
        assert decision_events[0].payload["approval_id"] == pending["approval_id"]
        assert decision_events[0].payload["resolved_by"] == {
            "subject": "approval-operator",
            "tenant": None,
            "source": "request",
        }


def test_sigkill_during_background_subagent_spawn_reattaches_one_child(tmp_path) -> None:
    backend = BackendConfig.sqlite(tmp_path)
    parent_session_id = "sigkill_subagent_parent"

    with RecoveryHarness(tmp_path, backend) as harness:
        killed_worker = harness.launch(
            scenario="background_subagent",
            action="start",
            session_id=parent_session_id,
        )
        phase = killed_worker.wait_for_phase("subagent_child_started")

        pre_kill = asyncio.run(harness.load_session_state(parent_session_id))
        assert pre_kill.session is not None
        assert pre_kill.session.status == SessionStatus.RUNNING
        assert "pending_tool_round" in pre_kill.checkpoint
        parent_started = [
            event
            for event in pre_kill.events
            if event.type == EventType.TOOL_CALL_STARTED
            and event.payload.get("tool_call_id") == "call_background_subagent"
        ]
        assert len(parent_started) == 1
        assert not any(
            event.payload.get("tool_call_id") == "call_background_subagent"
            and event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
            for event in pre_kill.events
        )

        children = asyncio.run(harness.list_child_sessions(parent_session_id))
        assert len(children) == 1
        child = children[0]
        assert child.id == phase["child_session_id"]
        assert child.parent_session_id == parent_session_id
        assert child.causal_budget_id == "sigkill-subagent-causal"
        assert child.status == SessionStatus.RUNNING
        assert child.metadata["subagent"]["mode"] == "background"
        assert (
            child.metadata["subagent"]["idempotency_key"]
            == (parent_started[0].payload["idempotency_key"])
        )

        killed_worker.sigkill()

        recovery_worker = harness.launch(
            scenario="background_subagent",
            action="recover",
            session_id=parent_session_id,
            child_session_id=child.id,
        )
        recovery_worker.wait_success()

        recovered_parent = asyncio.run(harness.load_session_state(parent_session_id))
        recovered_children = asyncio.run(harness.list_child_sessions(parent_session_id))
        assert recovered_parent.session is not None
        assert recovered_parent.session.status == SessionStatus.COMPLETED
        assert len(recovered_children) == 1
        assert recovered_children[0].id == child.id
        assert recovered_children[0].status == SessionStatus.INTERRUPTED

        reattached = [
            event
            for event in recovered_parent.events
            if event.type == EventType.TOOL_CALL_FAILED
            and event.payload.get("tool_call_id") == "call_background_subagent"
            and event.payload.get("recovered") is True
        ]
        assert len(reattached) == 1
        structured = reattached[0].payload["result"]["structured"]
        assert structured["recovery_reason"] == "pending_tool_round_reattached_subagent"
        assert structured["child_session_id"] == child.id
        assert structured["parent_session_id"] == parent_session_id
        assert structured["outcome_unknown"] is True

        tool_results = [
            part
            for message in recovered_parent.transcript
            for part in message.content
            if isinstance(part, ToolResultPart) and part.tool_call_id == "call_background_subagent"
        ]
        assert len(tool_results) == 1
        assert tool_results[0].structured["child_session_id"] == child.id


def test_sigkill_reclaims_only_an_unattached_expired_task_claim(
    tmp_path: Path,
    recovery_backend: BackendConfig,
) -> None:
    suffix = uuid4().hex[:10]
    task_id = f"sigkill_unattached_task_{suffix}"
    session_id = f"sigkill_unattached_session_{suffix}"
    task_type = f"sigkill.unattached.{suffix}"

    with RecoveryHarness(tmp_path, recovery_backend) as harness:
        killed_worker = harness.launch(
            scenario="task_claim",
            action="start_unattached",
            session_id=session_id,
            task_id=task_id,
            task_type=task_type,
        )
        killed_worker.wait_for_phase("unattached_task_claimed")

        pre_kill_task = asyncio.run(harness.load_task(task_id))
        assert pre_kill_task is not None
        assert pre_kill_task.status == TaskStatus.CLAIMED
        assert pre_kill_task.worker_id == "worker-a"
        assert pre_kill_task.session_id is None
        assert pre_kill_task.lease_expires_at is not None

        killed_worker.sigkill()

        recovery_worker = harness.launch(
            scenario="task_claim",
            action="recover_unattached",
            session_id=session_id,
            task_id=task_id,
            task_type=task_type,
        )
        result = recovery_worker.wait_success()
        assert result["reclaimed_task_ids"] == [task_id]
        assert result["worker_b_claimed"] == task_id

        recovered_task = asyncio.run(harness.load_task(task_id))
        recovered_session = asyncio.run(harness.load_session_state(session_id))
        assert recovered_task is not None
        assert recovered_task.status == TaskStatus.COMPLETED
        assert recovered_task.session_id == session_id
        assert recovered_task.worker_id is None
        assert recovered_task.lease_expires_at is None
        assert recovered_task.result == {
            "session_id": session_id,
            "agent_name": "recovery-agent",
            "environment_name": None,
        }
        assert recovered_session.session is not None
        assert recovered_session.session.status == SessionStatus.COMPLETED
        task_event_types = [
            event.type
            for event in recovered_session.events
            if event.type in {EventType.TASK_STARTED, EventType.TASK_COMPLETED}
        ]
        assert task_event_types == [EventType.TASK_STARTED, EventType.TASK_COMPLETED]
        assert asyncio.run(harness.list_causal_sessions(task_id)) == [recovered_session.session]


def test_sigkill_preserves_attached_task_ownership_and_recovers_linked_session(
    tmp_path: Path,
    recovery_backend: BackendConfig,
) -> None:
    suffix = uuid4().hex[:10]
    task_id = f"sigkill_attached_task_{suffix}"
    session_id = f"sigkill_attached_session_{suffix}"
    task_type = f"sigkill.attached.{suffix}"

    with RecoveryHarness(tmp_path, recovery_backend) as harness:
        killed_worker = harness.launch(
            scenario="task_claim",
            action="start_attached",
            session_id=session_id,
            task_id=task_id,
            task_type=task_type,
        )
        killed_worker.wait_for_phase("attached_tool_side_effect_recorded")

        pre_kill_task = asyncio.run(harness.load_task(task_id))
        pre_kill_session = asyncio.run(harness.load_session_state(session_id))
        assert pre_kill_task is not None
        assert pre_kill_task.status == TaskStatus.RUNNING
        assert pre_kill_task.worker_id == "worker-a"
        assert pre_kill_task.session_id == session_id
        assert pre_kill_session.session is not None
        assert pre_kill_session.session.status == SessionStatus.RUNNING
        assert "pending_tool_round" in pre_kill_session.checkpoint
        assert len(harness.read_marker()) == 1

        killed_worker.sigkill()

        recovery_worker = harness.launch(
            scenario="task_claim",
            action="recover_attached",
            session_id=session_id,
            task_id=task_id,
            task_type=task_type,
        )
        result = recovery_worker.wait_success()
        assert result["reclaimed_task_ids"] == []
        assert result["worker_b_claimed"] is None

        recovered_task = asyncio.run(harness.load_task(task_id))
        recovered_session = asyncio.run(harness.load_session_state(session_id))
        assert recovered_task is not None
        assert recovered_task.status == TaskStatus.COMPLETED
        assert recovered_task.session_id == session_id
        assert recovered_task.worker_id is None
        assert recovered_task.lease_expires_at is None
        assert recovered_task.result == {
            "session_id": session_id,
            "agent_name": "recovery-agent",
            "environment_name": None,
        }
        assert recovered_session.session is not None
        assert recovered_session.session.status == SessionStatus.COMPLETED
        assert recovered_session.checkpoint == {}
        assert len(harness.read_marker()) == 1
        assert asyncio.run(harness.list_causal_sessions(task_id)) == [recovered_session.session]

        tool_terminals = [
            event
            for event in recovered_session.events
            if event.payload.get("tool_call_id") == "call_side_effect"
            and event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        ]
        assert len(tool_terminals) == 1
        assert tool_terminals[0].type == EventType.TOOL_CALL_FAILED
        assert tool_terminals[0].payload["result"]["structured"]["outcome_unknown"] is True

        event_types = [event.type for event in recovered_session.events]
        for event_type in (
            EventType.TASK_STARTED,
            EventType.TASK_COMPLETED,
            EventType.SESSION_INTERRUPTED,
            EventType.SESSION_RESUMED,
            EventType.SESSION_COMPLETED,
        ):
            assert event_types.count(event_type) == 1
        assert (
            event_types.index(EventType.TASK_STARTED)
            < event_types.index(EventType.TOOL_CALL_FAILED)
            < event_types.index(EventType.SESSION_INTERRUPTED)
            < event_types.index(EventType.SESSION_RESUMED)
            < event_types.index(EventType.TASK_COMPLETED)
            < event_types.index(EventType.SESSION_COMPLETED)
        )
