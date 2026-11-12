"""A decided lost model outcome retains the saved environment cleanup owner."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.recovery import test_named_check_durable_recovery as fixture
from tests.runtime.test_orphaned_session_reconciliation import database_url as database_url

from cayu import EventType, Message, RunRequest, SessionStatus
from cayu.sessions import IncompleteSessionRecoveryRequest


def _application(root, patch, *, register_environment=True):
    from cayu.runners import LocalRunner

    make_runner = fixture.make_runner

    def local_transport(source):
        runner = make_runner(source)
        (root / "guest").mkdir(exist_ok=True)
        local = LocalRunner(root / "guest", inherit_env=False)
        # Only the container transport is substituted. Factory allocation,
        # binding ownership, native records and disposal remain real owners.
        patch.setattr(runner, "exec", local.exec)
        patch.setattr(runner, "exec_stream", local.exec_stream)
        return runner

    patch.setattr(fixture, "make_runner", local_transport)
    if not register_environment:
        patch.setattr(fixture.CayuApp, "register_environment_factory", lambda *args, **kwargs: None)
    return fixture.build_app(root, patch)


def _crash_model(root: Path) -> None:
    async def run():
        with pytest.MonkeyPatch.context() as patch:
            app, _store, provider = _application(root, patch)

            async def stream(_request):
                assert (root / "allocated").exists()
                os._exit(137)
                yield  # pragma: no cover

            patch.setattr(provider, "stream", stream)
            events = []
            async for event in app.run(
                RunRequest(
                    agent_name="coding",
                    session_id="lost-model",
                    messages=[Message.text("user", "work")],
                )
            ):
                events.append(event)
            raise AssertionError([(event.type, event.payload) for event in events])

    asyncio.run(run())


def _crash_model_during_operator_interrupt(root: Path) -> None:
    from cayu.sessions.requests import InterruptSessionRequest

    async def run():
        with pytest.MonkeyPatch.context() as patch:
            app, _store, provider = _application(root, patch)
            started = asyncio.Event()

            async def stream(_request):
                assert (root / "allocated").exists()
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    os._exit(137)
                yield  # pragma: no cover

            patch.setattr(provider, "stream", stream)

            async def interrupt_started_model():
                await started.wait()
                async for _ in app.interrupt_session(
                    InterruptSessionRequest(session_id="lost-model", reason="operator stop")
                ):
                    pass

            interruption = asyncio.create_task(interrupt_started_model())
            async for _ in app.run(
                RunRequest(
                    agent_name="coding",
                    session_id="lost-model",
                    messages=[Message.text("user", "work")],
                )
            ):
                pass
            await interruption
            raise AssertionError("The provider did not receive the operator cancellation")

    asyncio.run(run())


def _run_crashing_process(root, entry):
    worker = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; from {__name__} import {entry}; import sys; {entry}(Path(sys.argv[1]))",
            str(root),
        ],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(item for item in sys.path if item)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert worker.returncode == 137, (worker.stdout, worker.stderr)


def _database(root, database_url):
    (root / "factory-mode").touch()
    if database_url.startswith("postgresql:"):
        (root / "postgres-dsn").write_text(database_url)


@pytest.mark.parametrize("runtime_changed", [False, True])
@pytest.mark.parametrize("lose_ack", [False, True])
def test_owed_model_outcome_keeps_registered_binding_cleanup_reachable(
    tmp_path, database_url, lose_ack, runtime_changed
):
    _database(tmp_path, database_url)
    _run_crashing_process(tmp_path, "_crash_model")

    async def recover():
        with pytest.MonkeyPatch.context() as patch:
            if runtime_changed:
                import cayu.runtime._execution_profile_continuation as continuation

                patch.setattr(continuation, "_runtime_version", lambda: "999.0.0")
            app, store, provider = _application(tmp_path, patch)
            fan_out = app._event_writer.fan_out_persisted
            faults = []

            async def lost_after_model_commit(events, *args, **kwargs):
                if any(event.type is EventType.INTERACTION_INTERRUPTED for event in events):
                    faults.append(True)
                    raise OSError("model outcome committed before acknowledgement")
                return await fan_out(events, *args, **kwargs)

            if lose_ack:
                patch.setattr(app._event_writer, "fan_out_persisted", lost_after_model_commit)
            try:
                await app.reconcile_orphaned_sessions()
                assert faults == ([True] if lose_ack else [])
                assert (await store.load("lost-model")).status is SessionStatus.INTERRUPTED
                if not lose_ack:
                    assert (tmp_path / "disposed").exists()
                patch.setattr(app._event_writer, "fan_out_persisted", fan_out)
                await app.reconcile_orphaned_sessions()
                assert (tmp_path / "disposed").exists()
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id="lost-model", inactive_for_seconds=0
                    )
                )
                assert (tmp_path / "disposed").exists()
                assert not provider.requests
                events = await store.load_events("lost-model")
                assert sum(event.type is EventType.SESSION_INTERRUPTED for event in events) == 1
            finally:
                await app.aclose()
                await store.close()
                await app.task_store.close()

    asyncio.run(recover())


def test_recorded_operator_interruption_keeps_binding_cleanup_reachable(tmp_path, database_url):
    _database(tmp_path, database_url)
    _run_crashing_process(tmp_path, "_crash_model_during_operator_interrupt")
    _run_crashing_process(tmp_path, "_crash_settlement")

    async def recover():
        with pytest.MonkeyPatch.context() as patch:
            app, store, provider = _application(tmp_path, patch)
            try:
                await _settlement_claim_expired(store)
                assert (await store.load("lost-model")).status is SessionStatus.INTERRUPTED
                assert not (tmp_path / "disposed").exists()
                checkpoint = await store.load_checkpoint("lost-model")
                assert checkpoint["pending_session_interrupt"]["reason"] == "operator stop"
                await app.reconcile_orphaned_sessions()
                assert (tmp_path / "disposed").exists()
                assert not provider.requests
                events = await store.load_events("lost-model")
                assert sum(event.type is EventType.SESSION_INTERRUPTED for event in events) == 1
            finally:
                await app.aclose()
                await store.close()
                await app.task_store.close()

    asyncio.run(recover())


async def _settlement_claim_expired(store):
    from cayu.sessions.recovery import _incomplete_recovery_claim_from_checkpoint

    async with asyncio.timeout(10):
        while True:
            claim = _incomplete_recovery_claim_from_checkpoint(
                await store.load_checkpoint("lost-model")
            )
            if claim is None or claim[1] <= datetime.now(UTC):
                break
            await asyncio.sleep(0.05)


def _crash_settlement(root):
    import cayu.runtime._recovery_ownership as recovery_ownership

    recovery_ownership._INCOMPLETE_RECOVERY_CLAIM_LEASE = timedelta(seconds=1)

    async def run():
        with pytest.MonkeyPatch.context() as patch:
            app, _store, _provider = _application(root, patch)
            fan_out = app._event_writer.fan_out_persisted

            async def stop_after_commit(events, *args, **kwargs):
                if any(event.type is EventType.INTERACTION_INTERRUPTED for event in events):
                    os._exit(137)
                return await fan_out(events, *args, **kwargs)

            patch.setattr(app._event_writer, "fan_out_persisted", stop_after_commit)
            await app.reconcile_orphaned_sessions()
            raise AssertionError("Recovery did not reach the committed model outcome")

    asyncio.run(run())


def test_retained_binding_can_be_finalized_when_environment_registration_returns(
    tmp_path, database_url
):
    _database(tmp_path, database_url)
    _run_crashing_process(tmp_path, "_crash_model")
    _run_crashing_process(tmp_path, "_crash_settlement")

    async def recover():
        with pytest.MonkeyPatch.context() as patch:
            app, store, provider = _application(tmp_path, patch, register_environment=False)
            try:
                await _settlement_claim_expired(store)
                await app.reconcile_orphaned_sessions()
                assert (await store.load("lost-model")).status is SessionStatus.INTERRUPTED
                assert not (tmp_path / "disposed").exists()
                assert not (tmp_path / "reconnected").exists()
                assert not provider.requests
                events = await store.load_events("lost-model")
                assert sum(event.type is EventType.SESSION_INTERRUPTED for event in events) == 1
            finally:
                await app.aclose()
                await store.close()
                await app.task_store.close()
        with pytest.MonkeyPatch.context() as patch:
            app, store, provider = _application(tmp_path, patch)
            try:
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id="lost-model", inactive_for_seconds=0
                    )
                )
                assert (tmp_path / "disposed").exists()
                assert not provider.requests
                events = await store.load_events("lost-model")
                assert sum(event.type is EventType.SESSION_INTERRUPTED for event in events) == 1
            finally:
                await app.aclose()
                await store.close()
                await app.task_store.close()

    asyncio.run(recover())
