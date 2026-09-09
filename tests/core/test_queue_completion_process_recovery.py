"""Fresh-process recovery after rejected-only session completion commits."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tests.core.test_queued_session_messages import BlockingTwoTurnProvider

from cayu import (
    AgentSpec,
    CayuApp,
    EnqueueSessionMessageRequest,
    EventType,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    Message,
    RunRequest,
    SessionStatus,
    SQLiteSessionStore,
)
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
from cayu.runtime.session_message_lifecycle import SessionMessageConditions
from cayu.runtime.sessions import (
    _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY,
    _SESSION_RUN_OPERATION_CHECKPOINT_KEY,
    _interaction_transition_storage_key,
)

_SESSION = "rejected-only-process-completion"
_EXIT = 73


class _CrashAfterCompletionStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1
    observation_directory: Path

    async def transition_status_if_no_queued_messages(self, session_id, **kwargs):
        session = await super().transition_status_if_no_queued_messages(session_id, **kwargs)
        assert session.status is SessionStatus.COMPLETED
        assert kwargs.get("checkpoint_mutation") is None
        # A separate connection proves this is after durable commit, not merely
        # after an in-process mutation. No runtime finalizer gets to execute.
        observer = SQLiteSessionStore(self.observation_directory / "session.sqlite")
        snapshot = await _snapshot(observer, self.observation_directory)
        assert snapshot["status"] == "completed"
        assert snapshot["completed_events"] == 0
        assert snapshot["receipt"]["status_changed"] is False
        (self.observation_directory / "before.json").write_text(json.dumps(snapshot))
        os._exit(_EXIT)


async def _snapshot(store, directory):
    session = await store.load(_SESSION)
    assert session is not None
    checkpoint = await store.load_checkpoint(_SESSION)
    profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    events = await store.load_events(_SESSION)
    predecessor = next(event for event in events if event.type == EventType.INTERACTION_COMPLETED)
    connection = sqlite3.connect(f"{(directory / 'session.sqlite').as_uri()}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT record_json FROM cayu_session_operations "
            "WHERE session_id = ? AND idempotency_key = ?",
            (_SESSION, _interaction_transition_storage_key(predecessor.id)),
        ).fetchone()
        assert row is not None
        receipt = json.loads(row[0])
    finally:
        connection.close()
    return {
        "status": session.status.value,
        "epoch": session.run_epoch,
        "profile": profile.model_dump(mode="json") if profile is not None else None,
        "receipt": receipt,
        "receipt_json": row[0],
        "run_operation_present": _SESSION_RUN_OPERATION_CHECKPOINT_KEY in (checkpoint or {}),
        "recovery_claim_present": _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY in (checkpoint or {}),
        "completed_ids": [
            event.id for event in events if event.type == EventType.SESSION_COMPLETED
        ],
        "transcript": [
            message.model_dump(mode="json") for message in await store.load_transcript(_SESSION)
        ],
        "completed_events": sum(event.type == EventType.SESSION_COMPLETED for event in events),
        "started_events": sum(event.type == EventType.INTERACTION_STARTED for event in events),
        "expired_events": sum(event.type == EventType.SESSION_MESSAGE_EXPIRED for event in events),
        "delivered_events": sum(
            event.type == EventType.SESSION_MESSAGE_DELIVERED for event in events
        ),
    }


async def _worker(directory: Path, mode: str) -> None:
    store = (
        _CrashAfterCompletionStore(directory / "session.sqlite")
        if mode == "crash"
        else SQLiteSessionStore(directory / "session.sqlite")
    )
    if isinstance(store, _CrashAfterCompletionStore):
        store.observation_directory = directory
    provider = BlockingTwoTurnProvider()
    original_stream = provider.stream

    async def traced_stream(request):
        with (directory / "provider.calls").open("a") as output:
            output.write("dispatch\n")
            output.flush()
            os.fsync(output.fileno())
        assert mode == "crash", "terminal repair must not dispatch a provider"
        async for event in original_stream(request):
            yield event

    provider.stream = traced_stream
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    if mode == "crash":

        async def execute():
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=_SESSION,
                    messages=[Message.text("user", "initial")],
                    max_steps=1,
                )
            ):
                pass

        task = asyncio.create_task(execute())
        await asyncio.wait_for(provider.first_started.wait(), 10)
        await app.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id=_SESSION,
                idempotency_key="expired",
                content="must never be delivered",
                delivery_mode="on_idle",
                conditions=SessionMessageConditions(
                    expires_at=datetime.now(UTC) - timedelta(seconds=1)
                ),
            )
        )
        provider.release_first.set()
        await asyncio.wait_for(task, 20)
        raise AssertionError(
            "session-only completion checkpoint was not reached: "
            + repr(
                [
                    event.model_dump()
                    for event in await store.load_events(_SESSION)
                    if event.type == EventType.SESSION_FAILED
                ]
            )
        )
    try:
        result = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=_SESSION, inactive_for_seconds=0)
        )
        assert result.status is SessionStatus.COMPLETED
        expected_action = (
            IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE
            if mode == "recover"
            else IncompleteSessionRecoveryAction.SKIPPED_TERMINAL
        )
        assert expected_action in result.actions
        after = await _snapshot(store, directory)
        (directory / f"{mode}.json").write_text(json.dumps(after))
    finally:
        await store.close()


def test_rejected_only_completion_recovers_after_sqlite_process_loss(tmp_path):
    def launch(mode):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.core.test_queue_completion_process_recovery",
                str(tmp_path),
                mode,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    crashed = launch("crash")
    assert crashed.returncode == _EXIT, crashed.stdout + crashed.stderr
    before = json.loads((tmp_path / "before.json").read_text())
    assert before["profile"] is not None
    assert before["profile"]["run_epoch"] == before["epoch"]
    assert before["receipt"]["session"]["status"] == "running"
    assert before["receipt"]["only_if_no_queued_messages"] is True
    assert before["recovery_claim_present"] is False
    assert before["expired_events"] == 1
    assert before["started_events"] == 1
    assert before["delivered_events"] == 0
    for mode in ("recover", "retry"):
        recovered = launch(mode)
        assert recovered.returncode == 0, recovered.stdout + recovered.stderr
        after = json.loads((tmp_path / f"{mode}.json").read_text())
        assert after["status"] == "completed"
        assert after["receipt"] == before["receipt"]
        assert after["receipt_json"] == before["receipt_json"]
        assert after["transcript"] == before["transcript"]
        assert after["run_operation_present"] is False
        assert after["recovery_claim_present"] is False
        assert after["epoch"] > before["epoch"]
        assert after["completed_events"] == 1
        assert after["started_events"] == 1
        assert after["expired_events"] == 1
        assert after["delivered_events"] == 0
        assert after["profile"] is None or after["epoch"] > after["profile"]["run_epoch"]
    assert json.loads((tmp_path / "recover.json").read_text()) == after
    assert (tmp_path / "provider.calls").read_text() == "dispatch\n"


if __name__ == "__main__":
    asyncio.run(_worker(Path(sys.argv[1]), sys.argv[2]))
