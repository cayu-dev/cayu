from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest
from tests.core.test_abandoned_model_call_resume import _app, _consume, _LostProvider, _open

from cayu import EventType, Message, ModelStreamEvent, ResumeRequest, SessionStatus
from cayu.runtime._run_limits import UNKNOWN_POST_DISPATCH_BUDGET_REASON
from cayu.sessions.recovery import IncompleteSessionRecoveryAction


async def _start_child(backend, location, session_id, *, mode):
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "from tests.core.test_abandoned_model_call_resume import _run_child; _run_child()",
        env={
            **os.environ,
            "CAYU_MODEL_CRASH_TEST_BACKEND": backend,
            "CAYU_MODEL_CRASH_TEST_STORE": location,
            "CAYU_MODEL_CRASH_TEST_SESSION": session_id,
            "CAYU_MODEL_CRASH_TEST_CRASH": mode,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_abandoned_model_sweep_settles_only_after_native_inactivity_check(
    backend, request, tmp_path, monkeypatch
):
    async def scenario():
        location = (
            str(tmp_path / "model-sweep")
            if backend == "sqlite"
            else request.getfixturevalue("postgres_dsn")
        )
        sid = "model-sweep"
        child = await _start_child(backend, location, sid, mode="1")
        try:
            stdout, stderr = await asyncio.wait_for(child.communicate(), 30)
            assert child.returncode == 137, (stdout, stderr)
        finally:
            if child.returncode is None:
                child.kill()
                await child.wait()

        store, ledger = _open(backend, location)
        try:
            provider = _LostProvider(
                [
                    ModelStreamEvent.text_delta("Continued."),
                    ModelStreamEvent.completed(
                        {"finish_reason": "stop", "usage": {"input_tokens": 1, "output_tokens": 1}}
                    ),
                ]
            )
            app = _app(store, ledger, provider)
            original = await store.load(sid)
            active = await store.load_active_model_completion_stage(sid)
            assert original.status is SessionStatus.RUNNING
            assert active is not None and active.stage.state == "in_flight"
            assert (await app.inspect_session_execution(sid)).state == "owner_lost"
            reservations = [
                await ledger.load_reservation(rid) for rid in active.stage.reservation_ids
            ]

            # A fresh conversation remains available for continuation takeover.
            assert await app.recover_abandoned_sessions() == ()

            # A stale listing is not mutation authority: the native reservation
            # must check inactivity again before changing the stage or its budget.
            list_sessions = store.list_sessions

            async def list_even_if_fresh(query):
                return await list_sessions(query.model_copy(update={"inactive_for_seconds": 0}))

            with monkeypatch.context() as patch:
                patch.setattr(store, "list_sessions", list_even_if_fresh)
                [skipped] = await app.recover_abandoned_sessions()
            assert skipped.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
            assert (await store.load(sid)).run_epoch == original.run_epoch
            assert await store.load_active_model_completion_stage(sid) == active
            assert [
                await ledger.load_reservation(rid) for rid in active.stage.reservation_ids
            ] == reservations

            [result] = await app.recover_abandoned_sessions(inactive_for_seconds=0)
            assert result.previous_status is SessionStatus.RUNNING
            assert result.status is SessionStatus.INTERRUPTED
            assert IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED in result.actions
            current = await store.load(sid)
            assert current.status is SessionStatus.INTERRUPTED
            assert current.run_epoch > original.run_epoch
            assert await store.load_active_model_completion_stage(sid) is None
            settlement = await store.load_model_completion_stage_settlement(
                sid, active.stage.stage_id
            )
            assert settlement.disposition.value == "provider_effect_outcome_unknown"
            charged = [await ledger.load_reservation(rid) for rid in active.stage.reservation_ids]
            assert charged and all(record.status == "reconciled" for record in charged)
            assert all(record.reason == UNKNOWN_POST_DISPATCH_BUDGET_REASON for record in charged)
            assert all(record.actual_amount == record.reserved_amount for record in charged)
            assert provider.requests == []
            [interrupted] = [
                event
                for event in await store.load_events(sid)
                if event.type == EventType.SESSION_INTERRUPTED
            ]
            assert interrupted.payload["terminal_evidence_repaired"] is True
            [public_interruption] = [
                event for event in result.events if event.type == EventType.SESSION_INTERRUPTED
            ]
            assert public_interruption.payload == interrupted.payload
            assert await app.recover_abandoned_sessions(inactive_for_seconds=0) == ()
            assert [
                await ledger.load_reservation(rid) for rid in active.stage.reservation_ids
            ] == charged
            resumed = await _consume(
                app.resume(ResumeRequest(session_id=sid, messages=[Message.text("user", "Next")]))
            )
            assert (await store.load(sid)).status is SessionStatus.COMPLETED, [
                (event.type, event.payload) for event in resumed
            ]
            assert len(provider.requests) == 1
        finally:
            await store.close()
            await ledger.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_abandoned_model_sweep_preserves_live_dispatch_and_budget(backend, request, tmp_path):
    async def scenario():
        location = (
            str(tmp_path / "live-model")
            if backend == "sqlite"
            else request.getfixturevalue("postgres_dsn")
        )
        sid = "live-model"
        child = await _start_child(backend, location, sid, mode="block")
        store, ledger = _open(backend, location)
        try:
            ready = await asyncio.wait_for(child.stdout.readline(), 30)
            assert ready == b"dispatched\n", ready if ready else await child.stderr.read()
            app = _app(store, ledger, _LostProvider([]))
            original = await store.load(sid)
            active = await store.load_active_model_completion_stage(sid)
            assert active is not None
            reservations = [
                await ledger.load_reservation(rid) for rid in active.stage.reservation_ids
            ]
            [result] = await app.recover_abandoned_sessions(inactive_for_seconds=0)
            assert result.actions == (IncompleteSessionRecoveryAction.SKIPPED_EXECUTION_OWNER,)
            current = await store.load(sid)
            assert current.status is SessionStatus.RUNNING
            assert current.run_epoch == original.run_epoch
            assert (await app.inspect_session_execution(sid)).state == "executing"
            assert await store.load_active_model_completion_stage(sid) == active
            assert [
                await ledger.load_reservation(rid) for rid in active.stage.reservation_ids
            ] == reservations
        finally:
            if child.returncode is None:
                child.kill()
            await child.wait()
            await store.close()
            await ledger.close()

    asyncio.run(scenario())
