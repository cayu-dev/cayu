"""Expired-owner recovery must finish cancellation after replaying a failed session."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus
from cayu.tasks.worker import _recover_expired_interrupted_task_handoffs


class _WorkerLost(BaseException):
    """Simulate process loss after the failure decision, before the task write."""


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("cancel_pending", [False, True])
async def test_replayed_failure_does_not_strand_an_expired_task_cancellation(
    backend: str, cancel_pending: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [datetime.now(UTC)]
    if backend == "memory":
        tasks = InMemoryTaskStore(ownership_clock=lambda: now[0])
        sessions = InMemorySessionStore()
    else:
        tasks = SQLiteTaskStore(tmp_path / "tasks.db", ownership_clock=lambda: now[0])
        sessions = SQLiteSessionStore(tmp_path / "sessions.db")
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.error("source failed"),
                ModelStreamEvent.completed({"finish_reason": "error"}),
            ]
        ]
    )

    def make_app() -> CayuApp:
        app = CayuApp(task_store=tasks, session_store=sessions, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))
        return app

    app = make_app()
    recovered_app = make_app()
    try:
        await tasks.create_task(TaskCreate(task_id="lost-job", type="job"))
        claim = await tasks.claim_task("lost-worker", lease_seconds=30)
        assert claim is not None and claim.lease_expires_at is not None

        async def die_before_task_write(**_kwargs):
            raise _WorkerLost()

        with monkeypatch.context() as patch:
            patch.setattr(app._session_finalization, "fail_task", die_before_task_write)
            with pytest.raises(_WorkerLost):
                async for _ in app.run(
                    RunRequest(
                        agent_name="worker-agent",
                        session_id="lost-session",
                        task_id=claim.id,
                        task_worker_id="lost-worker",
                        task_lease_expires_at=claim.lease_expires_at,
                        messages=[Message.text("user", "process this job")],
                    )
                ):
                    pass

        assert (await tasks.load_task(claim.id)).status is TaskStatus.RUNNING
        assert (await sessions.load("lost-session")).status is SessionStatus.RUNNING
        if cancel_pending:
            requested = await tasks.cancel_task(claim.id, {"code": "operator_cancelled"})
            assert requested.status_reason == "cancellation_requested"
        now[0] = claim.lease_expires_at + timedelta(seconds=1)

        # This is the scanner called by run_task_worker on every recovery pass.
        for _ in range(2):
            await _recover_expired_interrupted_task_handoffs(
                recovered_app, tasks, worker_id="recovery-worker", after=None, limit=10, stop=None
            )
        assert (await sessions.load("lost-session")).status is SessionStatus.FAILED
        assert len(provider.requests) == 1  # recovery must not redispatch model work
        settled = await tasks.load_task(claim.id)
        assert settled.status is (TaskStatus.CANCELLED if cancel_pending else TaskStatus.FAILED), (
            "Recovery ended the session but abandoned its dead owner's cancellation",
            settled.status,
            settled.status_reason,
            settled.worker_id,
        )
    finally:
        await recovered_app.aclose()
        await app.aclose()
        if backend == "sqlite":
            await sessions.close()
            await tasks.close()
