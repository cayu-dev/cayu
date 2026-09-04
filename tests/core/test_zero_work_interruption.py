from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.core.test_recovery_plans import _app, _create_running_session, _FakeProvider

from cayu import SQLiteSessionStore
from cayu.core import AgentSpec, Event, EventType
from cayu.runtime import (
    CayuApp,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    InterruptSessionRequest,
    RecoveryExecutionRequest,
    RecoveryItemExecutionStatus,
    RecoveryPlanAction,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
    SessionStatus,
)
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._zero_work_interruption import ZeroWorkInterruptionRequest


async def _orphan(store):
    app = _app(store)
    await _create_running_session(store, app, "zero-work")

    async def interrupt():
        async for _ in app.interrupt_session(InterruptSessionRequest(session_id="zero-work")):
            pass

    task = asyncio.create_task(interrupt())
    try:
        async with asyncio.timeout(10):
            while (await store.load("zero-work")).status is not SessionStatus.INTERRUPTING:
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert (await store.load("zero-work")).status is SessionStatus.INTERRUPTING


class _DriftedProvider(_FakeProvider):
    @property
    def execution_profile_identity(self):
        from tests.core._execution_profile_fixtures import versioned_test_provider_identity

        return versioned_test_provider_identity(self, behavior_version="2")


def _drifted_app(store):
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(_DriftedProvider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model", system_prompt="changed profile")
    )
    return app


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path: Path):
    if request.param == "postgres":
        from tests.core.test_postgres_session_store import _new_store, _truncate

        dsn = request.getfixturevalue("postgres_dsn")
        asyncio.run(_truncate(dsn))
        return _new_store(dsn)
    return (
        InMemorySessionStore()
        if request.param == "memory"
        else SQLiteSessionStore(tmp_path / "sessions.db")
    )


def _run(store, exercise):
    async def run():
        try:
            await exercise()
        finally:
            if not isinstance(store, InMemorySessionStore):
                await store.close()

    asyncio.run(run())


def test_profile_drift_terminalizes_and_replays_exactly(store):
    async def exercise():
        await _orphan(store)
        before = await store.load("zero-work")
        app = _drifted_app(store)
        request = IncompleteSessionRecoveryRequest(
            session_id="zero-work", inactive_for_seconds=None
        )
        result = await app.recover_incomplete_session(request)
        assert result.status is SessionStatus.INTERRUPTED
        assert result.actions == (IncompleteSessionRecoveryAction.TERMINALIZED_ZERO_WORK,)
        from cayu.runtime.interactions import InteractionStatus, InteractionSummaryEvidence

        summary = InteractionSummaryEvidence.model_validate(result.events[0].payload)
        assert summary.status is InteractionStatus.INTERRUPTED
        assert summary.model_step_count == summary.tool_call_count == 0
        assert (await store.load("zero-work")).run_epoch == before.run_epoch + 1
        replay = await app.recover_incomplete_session(request)
        assert [e.id for e in replay.events] == [e.id for e in result.events]
        assert (
            sum(
                e.type is EventType.SESSION_INTERRUPTED
                for e in await store.load_events("zero-work")
            )
            == 1
        )

    _run(store, exercise)


def test_plan_exposes_terminalization_without_validating_drifted_profile(store):
    async def exercise():
        await _orphan(store)
        app = _drifted_app(store)
        plan = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(
                    session_ids=("zero-work",), inactive_for_seconds=None
                )
            )
        )
        assert RecoveryPlanAction.TERMINALIZE_ZERO_WORK in plan.items[0].allowed_actions
        assert plan.items[0].registration.validated_execution_profile_fingerprint is None
        receipt = await app.execute_recovery(
            RecoveryExecutionRequest(plan=plan, execution_id="zero-execution")
        )
        assert receipt.items[0].status is RecoveryItemExecutionStatus.EXECUTED
        assert receipt.items[0].final_session_status is SessionStatus.INTERRUPTED

    _run(store, exercise)


def test_unknown_checkpoint_evidence_fails_closed(store):
    async def exercise():
        await _orphan(store)
        await runtime_checkpoint_session_store(store).transform_checkpoint(
            "zero-work",
            lambda session, cp: {**(cp or {}), "future_unknown_effect": {"pending": True}},
        )
        session = await store.load("zero-work")
        cp = await store.load_checkpoint("zero-work")
        result = await store._terminalize_zero_work_interruption(
            ZeroWorkInterruptionRequest(session, cp, None, True)
        )
        assert result is None
        assert await store.load_checkpoint("zero-work") == cp
        current = await store.load("zero-work")
        assert current is not None and current.status is SessionStatus.INTERRUPTING

    _run(store, exercise)


def test_model_event_and_stale_epoch_cannot_terminalize(store):
    async def exercise():
        await _orphan(store)
        session = await store.load("zero-work")
        cp = await store.load_checkpoint("zero-work")
        stale = session.model_copy(update={"run_epoch": session.run_epoch + 1})
        assert (
            await store._terminalize_zero_work_interruption(
                ZeroWorkInterruptionRequest(stale, cp, None, True)
            )
            is None
        )
        await store.append_event(
            "zero-work", Event(type=EventType.MODEL_STARTED, session_id="zero-work")
        )
        assert (
            await store._terminalize_zero_work_interruption(
                ZeroWorkInterruptionRequest(session, cp, None, True)
            )
            is None
        )

    _run(store, exercise)


def test_concurrent_terminalizers_and_acknowledgement_loss(store, monkeypatch):
    async def exercise():
        await _orphan(store)
        session = await store.load("zero-work")
        cp = await store.load_checkpoint("zero-work")
        request = ZeroWorkInterruptionRequest(session, cp, None, True)
        first, second = await asyncio.gather(
            store._terminalize_zero_work_interruption(request),
            store._terminalize_zero_work_interruption(request),
        )
        assert first is not None and second is not None
        assert sorted((first.replayed, second.replayed)) == [False, True]
        assert first.events == second.events
        assert (await store.load("zero-work")).run_epoch == session.run_epoch + 1
        assert (
            await store._terminalize_zero_work_interruption(
                ZeroWorkInterruptionRequest(
                    session.model_copy(update={"run_epoch": session.run_epoch - 1}), cp, None, True
                )
            )
            is None
        )

    _run(store, exercise)


def test_public_recovery_reconciles_commit_acknowledgement_loss(store, monkeypatch):
    async def exercise():
        await _orphan(store)
        original = store._terminalize_zero_work_interruption
        lost = False

        async def lose_ack(request):
            nonlocal lost
            result = await original(request)
            if request.commit and result is not None and not result.replayed and not lost:
                lost = True
                raise OSError("committed acknowledgement lost")
            return result

        monkeypatch.setattr(store, "_terminalize_zero_work_interruption", lose_ack)
        result = await _drifted_app(store).recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="zero-work", inactive_for_seconds=None)
        )
        assert lost
        assert result.status is SessionStatus.INTERRUPTED
        assert (
            sum(
                e.type is EventType.SESSION_INTERRUPTED
                for e in await store.load_events("zero-work")
            )
            == 1
        )

    _run(store, exercise)


def test_terminalization_preserves_ordinary_resume_admission(store):
    from cayu.core import Message
    from cayu.runtime import ResumeRequest

    async def exercise():
        await _orphan(store)
        await _drifted_app(store).recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="zero-work", inactive_for_seconds=None)
        )
        events = [
            e
            async for e in _app(store).resume(
                ResumeRequest(session_id="zero-work", messages=[Message.text("user", "continue")])
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED

    _run(store, exercise)


@pytest.mark.parametrize(
    "key",
    [
        "pending_approval",
        "pending_tool_round",
        "pending_user_input",
        "workspace_observations",
        "environment_factory_allocation_intents",
        "incomplete_recovery_claim",
        "pending_provider_operation_disposition",
        "pending_completion_finalization",
    ],
)
def test_pending_evidence_never_uses_profile_independent_terminalization(store, key):
    async def exercise():
        await _orphan(store)
        from cayu.runtime.sessions import _workspace_observation_authority_mutation_scope

        with _workspace_observation_authority_mutation_scope():
            await runtime_checkpoint_session_store(store).transform_checkpoint(
                "zero-work",
                lambda session, cp: {
                    **(cp or {}),
                    "checkpoint_schema_version": 7,
                    key: {"pending": True},
                },
            )
        session = await store.load("zero-work")
        checkpoint = await store.load_checkpoint("zero-work")
        assert key in checkpoint
        assert (
            await store._terminalize_zero_work_interruption(
                ZeroWorkInterruptionRequest(session, checkpoint, None, True)
            )
            is None
        )
        assert await store.load_checkpoint("zero-work") == checkpoint
        assert (await store.load("zero-work")).run_epoch == session.run_epoch

    _run(store, exercise)


def test_custom_store_without_atomic_protocol_retains_profile_admission(monkeypatch):
    from cayu.runtime.sessions import SessionStore

    store = InMemorySessionStore()

    async def exercise():
        await _orphan(store)
        before = await store.load_checkpoint("zero-work")
        monkeypatch.setattr(
            store,
            "_terminalize_zero_work_interruption",
            SessionStore._terminalize_zero_work_interruption.__get__(store),
        )
        with pytest.raises(Exception, match="[Pp]rofile"):
            await _drifted_app(store).recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="zero-work", inactive_for_seconds=None)
            )
        assert await store.load_checkpoint("zero-work") == before
        current = await store.load("zero-work")
        assert current is not None and current.status is SessionStatus.INTERRUPTING

    _run(store, exercise)


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_fresh_process_terminalization_and_replay(backend, tmp_path, request):
    import json
    import os
    import subprocess
    import sys

    if backend == "postgres":
        from tests.core.test_postgres_session_store import _truncate

        location = request.getfixturevalue("postgres_dsn")
        asyncio.run(_truncate(location))
    else:
        location = str(tmp_path / "fresh.db")
    script = """
import asyncio, json, sys
from cayu import SQLiteSessionStore
from tests.core.test_zero_work_interruption import _orphan, _drifted_app
from tests.core.test_postgres_session_store import _new_store
from cayu.runtime import IncompleteSessionRecoveryRequest
async def main():
    store = SQLiteSessionStore(sys.argv[2]) if sys.argv[1] == 'sqlite' else _new_store(sys.argv[2])
    try:
        if sys.argv[3] == 'create':
            await _orphan(store)
        else:
            result = await _drifted_app(store).recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id='zero-work', inactive_for_seconds=None))
            print(json.dumps({'status': str(result.status), 'events': [e.id for e in result.events]}))
    finally:
        await store.close()
asyncio.run(main())
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[2] / "src")
        + os.pathsep
        + str(Path(__file__).resolve().parents[2])
    )

    def child(mode):
        result = subprocess.run(
            [sys.executable, "-c", script, backend, location, mode],
            env=env,
            text=True,
            capture_output=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    child("create")
    # Independent interpreters and independent store connections race the same
    # accepted interruption. Both must observe the one committed receipt.
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, backend, location, "recover"],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    results = []
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=45)
            assert process.returncode == 0, stderr
            results.append(json.loads(stdout))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    first = results[0]
    assert first["status"] == "interrupted"
    assert len(first["events"]) == 2
    assert results[1] == first
    assert json.loads(child("recover")) == first


def test_live_recovery_lease_blocks_and_expired_lease_can_be_terminalized(store):
    from datetime import UTC, datetime, timedelta

    async def exercise():
        await _orphan(store)
        now = datetime.now(UTC)

        async def claim(start, end):
            await runtime_checkpoint_session_store(store).transform_checkpoint(
                "zero-work",
                lambda s, cp: {
                    **(cp or {}),
                    "incomplete_session_recovery_claim": {
                        "version": 1,
                        "claim_id": "old-owner",
                        "claimed_at": start.isoformat(),
                        "claim_expires_at": end.isoformat(),
                    },
                },
            )

        await claim(now, now + timedelta(minutes=5))
        session = await store.load("zero-work")
        checkpoint = await store.load_checkpoint("zero-work")
        assert (
            await store._terminalize_zero_work_interruption(
                ZeroWorkInterruptionRequest(session, checkpoint, None, True)
            )
            is None
        )
        await claim(now - timedelta(minutes=10), now - timedelta(minutes=5))
        result = await _drifted_app(store).recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="zero-work", inactive_for_seconds=None)
        )
        assert result.status is SessionStatus.INTERRUPTED
        assert "incomplete_session_recovery_claim" not in await store.load_checkpoint("zero-work")

    _run(store, exercise)


def test_publication_failure_rolls_back_status_epoch_and_events(store, monkeypatch):
    async def exercise():
        await _orphan(store)
        session = await store.load("zero-work")
        checkpoint = await store.load_checkpoint("zero-work")
        events = await store.load_events("zero-work")
        if isinstance(store, InMemorySessionStore):

            def fail(*args, **kwargs):
                raise OSError("before memory publication")

            monkeypatch.setattr(store, "_prepare_checkpoint_store_unlocked", fail)
        elif isinstance(store, SQLiteSessionStore):
            import cayu.storage.sqlite as sqlite_module

            original = sqlite_module._append_events_in_transaction

            def fail(*args, **kwargs):
                original(*args, **kwargs)
                raise OSError("after transactional event append")

            monkeypatch.setattr(sqlite_module, "_append_events_in_transaction", fail)
        else:

            async def fail(*args, **kwargs):
                raise OSError("after transactional event append")

            monkeypatch.setattr(store, "_upsert_checkpoint", fail)
        with pytest.raises(OSError):
            await store._terminalize_zero_work_interruption(
                ZeroWorkInterruptionRequest(session, checkpoint, None, True)
            )
        assert await store.load("zero-work") == session
        assert await store.load_checkpoint("zero-work") == checkpoint
        assert await store.load_events("zero-work") == events

    _run(store, exercise)


def test_replay_does_not_hide_new_pending_evidence(store):
    async def exercise():
        await _orphan(store)
        await _drifted_app(store).recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="zero-work", inactive_for_seconds=None)
        )
        await runtime_checkpoint_session_store(store).transform_checkpoint(
            "zero-work",
            lambda session, cp: {**(cp or {}), "pending_tool_round": {"unresolved": True}},
        )
        session = await store.load("zero-work")
        checkpoint = await store.load_checkpoint("zero-work")
        assert (
            await store._terminalize_zero_work_interruption(
                ZeroWorkInterruptionRequest(session, checkpoint, None, True)
            )
            is None
        )
        assert await store.load_checkpoint("zero-work") == checkpoint

    _run(store, exercise)
