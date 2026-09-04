from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest
from worker_harness import (
    BackendConfig,
    RecoveryHarness,
    TerminalRaceHook,
    TerminalRaceProvider,
    terminal_race_tools,
)

from cayu import (
    AgentSpec,
    CayuApp,
    EnqueueSessionMessageRequest,
    EventType,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionsRecoveryRequest,
    InterruptSessionRequest,
    Message,
    ResumeRequest,
    SessionMessageDeliveryMode,
    SessionMessageQueueStatus,
    SessionStatus,
    TaskQuery,
    TaskStatus,
)
from cayu.runtime._invocation_terminal_decision import (
    invocation_terminal_decision_from_checkpoint,
    settled_invocation_terminal_decision_from_checkpoint,
)

pytestmark = pytest.mark.process


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
def terminal_race_backend(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> BackendConfig:
    if request.param == "sqlite":
        return BackendConfig.sqlite(tmp_path)
    if not _postgres_recovery_requested():
        pytest.skip("Postgres process recovery runs in the required-Postgres lane")
    return BackendConfig.postgres(request.getfixturevalue("postgres_dsn"))


@pytest.mark.parametrize(
    ("race_mode", "expected_phase", "expected_provider_requests"),
    [
        pytest.param(
            "provider_failure",
            "provider_started_after_tool",
            2,
            id="provider-already-dispatched",
        ),
        pytest.param(
            "before_next_provider",
            "tool_completed_before_next_provider",
            1,
            id="interrupt-before-next-provider",
        ),
    ],
)
def test_remote_interrupt_and_linked_task_failure_elect_one_terminal_outcome(
    tmp_path: Path,
    terminal_race_backend: BackendConfig,
    race_mode: str,
    expected_phase: str,
    expected_provider_requests: int,
) -> None:
    suffix = uuid4().hex
    session_id = f"terminal-race-session-{suffix}"
    task_id = f"terminal-race-task-{suffix}"
    task_type = f"terminal-race-{suffix}"

    async def scenario(harness: RecoveryHarness) -> None:
        worker = harness.launch(
            scenario="terminal_race",
            action="run",
            session_id=session_id,
            task_id=task_id,
            task_type=task_type,
            race_mode=race_mode,
        )
        phase = await asyncio.to_thread(worker.wait_for_phase, expected_phase)
        if race_mode == "provider_failure":
            assert phase["request_count"] == 2
        assert len(harness.read_marker()) == 1

        session_store = harness.session_store()
        task_store = harness.task_store()
        provider = TerminalRaceProvider()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            runtime_hooks=[TerminalRaceHook()],
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="recovery-agent", model="recovery-model"),
            tools=terminal_race_tools(harness.marker_path),
        )
        try:
            accepted = await app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session_id,
                    idempotency_key=f"queued-{suffix}",
                    content="deliver after the interruption",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )
            with pytest.raises(TimeoutError, match="interruption is still finalizing"):
                _ = [
                    event
                    async for event in app.interrupt_session(
                        InterruptSessionRequest(
                            session_id=session_id,
                            reason="remote operator wins",
                        )
                    )
                ]

            interrupting = await session_store.load(session_id)
            assert interrupting is not None
            assert interrupting.status is SessionStatus.INTERRUPTING
            worker.signal("release_provider_failure")
            worker_result = await asyncio.to_thread(worker.wait_success)
            assert worker_result["processed"] == 1
            assert worker_result["provider_request_count"] == expected_provider_requests

            interrupted = await session_store.load(session_id)
            task = await task_store.load_task(task_id)
            durable_events = await session_store.load_events(session_id)
            assert interrupted is not None
            assert interrupted.status is SessionStatus.INTERRUPTED
            assert task is not None
            assert task.status is TaskStatus.RUNNING
            assert task.worker_id is None
            assert sum(event.type is EventType.SESSION_INTERRUPTED for event in durable_events) == 1
            assert not any(event.type is EventType.SESSION_FAILED for event in durable_events)
            assert not any(event.type is EventType.TASK_FAILED for event in durable_events)
            marker_phases = [
                record["phase"] for record in harness.read_marker() if "phase" in record
            ]
            assert marker_phases == ["terminal_hook_completed"]

            queued_replay = await app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session_id,
                    idempotency_key=f"queued-{suffix}",
                    content="deliver after the interruption",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )
            assert queued_replay.replayed is True
            assert queued_replay.message.queue_id == accepted.message.queue_id
            assert queued_replay.message.status is SessionMessageQueueStatus.QUEUED

            continuation = await task_store.claim_interrupted_task_continuation(
                "terminal-race-recovery",
                TaskQuery(type=task_type),
                handoff_id=f"handoff-{suffix}",
            )
            assert continuation.task is not None
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        task_worker_id="terminal-race-recovery",
                        task_handoff_id=continuation.task.interrupted_handoff_id,
                        messages=[Message.text("user", "resume")],
                    )
                )
            ]
            assert resumed[-1].type is EventType.SESSION_COMPLETED
            assert provider.request_count == 1
            resumed_checkpoint = await session_store.load_checkpoint(session_id)
            assert settled_invocation_terminal_decision_from_checkpoint(resumed_checkpoint) is None
            completed_task = await task_store.load_task(task_id)
            assert completed_task is not None
            assert completed_task.status is TaskStatus.COMPLETED
            delivered = [
                event
                for event in await session_store.load_events(session_id)
                if event.type is EventType.SESSION_MESSAGE_DELIVERED
                and event.payload.get("queue_id") == accepted.message.queue_id
            ]
            assert len(delivered) == 1
        finally:
            await session_store.close()
            await task_store.close()

    with RecoveryHarness(tmp_path, terminal_race_backend) as harness:
        asyncio.run(scenario(harness))


def test_process_loss_after_failure_decision_recovers_exact_terminal_outcome(
    tmp_path: Path,
    terminal_race_backend: BackendConfig,
) -> None:
    suffix = uuid4().hex
    session_id = f"terminal-decision-loss-session-{suffix}"
    task_id = f"terminal-decision-loss-task-{suffix}"
    task_type = f"terminal-decision-loss-{suffix}"

    async def scenario(harness: RecoveryHarness) -> None:
        worker = harness.launch(
            scenario="terminal_race",
            action="run",
            session_id=session_id,
            task_id=task_id,
            task_type=task_type,
            race_mode="failure_decision_loss",
        )
        await asyncio.to_thread(worker.wait_for_phase, "failure_decision_committed")
        worker.sigkill()
        await asyncio.sleep(1.1)

        session_store = harness.session_store()
        task_store = harness.task_store()
        provider = TerminalRaceProvider()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            runtime_hooks=[TerminalRaceHook()],
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="recovery-agent", model="recovery-model"),
            tools=terminal_race_tools(harness.marker_path),
        )
        try:
            recovered = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    reason="worker_lost_after_terminal_decision",
                )
            )
            session = await session_store.load(session_id)
            task = await task_store.load_task(task_id)
            events = await session_store.load_events(session_id)
            assert recovered.actions == (
                IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_OWNERSHIP,
            )
            assert session is not None and session.status is SessionStatus.FAILED
            assert task is not None and task.status is TaskStatus.FAILED
            assert provider.request_count == 0
            assert sum(event.type is EventType.TASK_FAILED for event in events) == 1
            assert sum(event.type is EventType.INTERACTION_FAILED for event in events) == 1
            assert sum(event.type is EventType.SESSION_FAILED for event in events) == 1
        finally:
            await session_store.close()
            await task_store.close()

    with RecoveryHarness(tmp_path, terminal_race_backend) as harness:
        asyncio.run(scenario(harness))


@pytest.mark.parametrize("recovery_entrance", ["single", "batch"])
def test_process_loss_after_claimed_task_failure_recovers_exact_terminal_outcome(
    tmp_path: Path,
    terminal_race_backend: BackendConfig,
    recovery_entrance: str,
) -> None:
    suffix = uuid4().hex
    session_id = f"terminal-task-commit-loss-session-{suffix}"
    task_id = f"terminal-task-commit-loss-task-{suffix}"
    task_type = f"terminal-task-commit-loss-{suffix}"

    async def scenario(harness: RecoveryHarness) -> None:
        worker = harness.launch(
            scenario="terminal_race",
            action="run",
            session_id=session_id,
            task_id=task_id,
            task_type=task_type,
            race_mode="failure_task_commit_loss",
        )
        await asyncio.to_thread(worker.wait_for_phase, "task_failure_committed")
        worker.sigkill()

        session_store = harness.session_store()
        task_store = harness.task_store()
        provider = TerminalRaceProvider()
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            runtime_hooks=[TerminalRaceHook()],
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="recovery-agent", model="recovery-model"),
            tools=terminal_race_tools(harness.marker_path),
        )
        try:
            task_before = await task_store.load_task(task_id)
            session_before = await session_store.load(session_id)
            checkpoint_before = await session_store.load_checkpoint(session_id)
            assert task_before is not None and task_before.status is TaskStatus.FAILED
            assert task_before.worker_id is None
            assert task_before.lease_expires_at is None
            assert task_before.interrupted_handoff_id is None
            assert session_before is not None and session_before.status is SessionStatus.RUNNING
            assert checkpoint_before is not None
            decision_before = invocation_terminal_decision_from_checkpoint(checkpoint_before)
            assert decision_before is not None
            assert decision_before.task_terminalization_request_sha256 is not None
            events_before = await session_store.load_events(session_id)
            assert not any(event.type is EventType.TASK_FAILED for event in events_before)
            assert not any(event.type is EventType.INTERACTION_FAILED for event in events_before)
            assert not any(event.type is EventType.SESSION_FAILED for event in events_before)

            if recovery_entrance == "single":
                recovered = await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id=session_id,
                        inactive_for_seconds=None,
                        reason="worker_lost_after_claimed_task_failure",
                    )
                )
            else:
                page = await app.recover_incomplete_sessions(
                    IncompleteSessionsRecoveryRequest(
                        statuses={SessionStatus.RUNNING},
                        inactive_for_seconds=None,
                        reason="worker_lost_after_claimed_task_failure",
                    )
                )
                recovered = next(
                    result for result in page.results if result.session_id == session_id
                )
            session = await session_store.load(session_id)
            task = await task_store.load_task(task_id)
            events = await session_store.load_events(session_id)
            assert recovered.actions == (
                IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_OWNERSHIP,
            )
            assert session is not None and session.status is SessionStatus.FAILED
            assert task == task_before
            assert provider.request_count == 0
            assert sum(event.type is EventType.TASK_FAILED for event in events) == 1
            assert sum(event.type is EventType.INTERACTION_FAILED for event in events) == 1
            assert sum(event.type is EventType.SESSION_FAILED for event in events) == 1
        finally:
            await session_store.close()
            await task_store.close()

    with RecoveryHarness(tmp_path, terminal_race_backend) as harness:
        asyncio.run(scenario(harness))
