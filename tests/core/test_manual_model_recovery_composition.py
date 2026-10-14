"""Manual recovery owns settlement independently of session turn execution."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_durable_model_terminalization import _store_for_backend
from tests.core.test_model_completion_recovery import (
    _RecordingProvider,
    _stage_in_flight_model_boundary,
)
from tests.core.test_recovery_composition import _run_without_owners

from cayu import AgentSpec, CayuApp, EventType
from cayu.sessions import EventQuery
from cayu.sessions._model_stage import ModelCompletionManualRecoveryRequest
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.tasks.memory import InMemoryTaskStore


def test_manual_model_recovery_imports_without_application_or_engine() -> None:
    _run_without_owners(
        ("cayu.applications", "cayu.runtime._session_engine"),
        "from cayu.runtime._manual_model_completion_recovery import ManualModelCompletionRecovery\n",
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "mode", ["registered", "terminal-failed", "terminal-interrupted", "abandoned"]
)
def test_manual_model_recovery_and_replay_without_engine(
    backend, mode, tmp_path, request, monkeypatch
) -> None:
    async def scenario() -> None:
        store = _store_for_backend(backend, tmp_path / "manual-recovery.db", request)
        app = CayuApp(session_store=store, enable_logging=False)
        try:
            terminalization = mode.startswith("terminal-")
            status = (
                SessionStatus.INTERRUPTED
                if mode in {"terminal-interrupted", "abandoned"}
                else SessionStatus.FAILED
            )
            if not terminalization:
                app.register_provider(_RecordingProvider([]), default=True)
                app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            session, _, stage = await _stage_in_flight_model_boundary(
                store,
                session_id="manual-composition-" + mode,
                provider_name=_RecordingProvider.name,
                reservation_ids=(),
            )
            command = ModelCompletionManualRecoveryRequest(
                session_id=session.id,
                stage_id=stage.stage_id,
                expected_run_epoch=session.run_epoch,
                expected_session_instance_id=session.instance_id,
                terminal_status=status,
                terminalization_only=terminalization,
                inactive_for_seconds=0,
            )
            owner = app._manual_model_completion_recovery
            monkeypatch.delattr(app, "_session_engine")
            monkeypatch.delattr(app._session_recovery, "_engine")
            if mode == "abandoned":
                assert await app._session_recovery._settle_abandoned_model_dispatch(session) == ()
                first = await app.recover_model_completion_stage(command)
                assert first.replayed
            else:
                first = await app.recover_model_completion_stage(command)
                assert not first.replayed
            assert first.session.status is status
            assert await store.load_active_model_completion_stage(session.id) is None

            # The owner keeps its construction-time store even if the public
            # application facade is later pointed at another backend.
            app.session_store = InMemorySessionStore()
            direct_replay = await owner.recover_model_completion_stage(command)
            assert direct_replay.replayed
            assert direct_replay.settlement == first.settlement
            assert await app.session_store.load(session.id) is None
            app.session_store = store
            replay = await app.recover_model_completion_stage(command)
            assert replay.replayed
            assert replay.settlement == first.settlement
            events = await store.query_events(
                EventQuery(
                    session_id=session.id,
                    event_types={EventType.INTERACTION_FAILED, EventType.INTERACTION_INTERRUPTED},
                )
            )
            assert len(events) == 1
        finally:
            await app.aclose()
            if backend != "memory":
                await store.close()

    asyncio.run(scenario())


def test_manual_recovery_keeps_captured_task_admission_before_settlement(monkeypatch) -> None:
    class UnavailableTaskStore(InMemoryTaskStore):
        async def load_active_work_contract_task_for_session(self, session_id):
            raise RuntimeError("task authority unavailable")

    async def scenario() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, task_store=UnavailableTaskStore(), enable_logging=False)
        try:
            session, _, stage = await _stage_in_flight_model_boundary(
                store,
                session_id="manual-admission-failure",
                provider_name=_RecordingProvider.name,
                reservation_ids=(),
            )
            monkeypatch.delattr(app, "_session_engine")
            app.task_store = InMemoryTaskStore()
            before = await store.load(session.id)
            with pytest.raises(RuntimeError, match="task authority unavailable"):
                await app.recover_model_completion_stage(
                    ModelCompletionManualRecoveryRequest(
                        session_id=session.id,
                        stage_id=stage.stage_id,
                        expected_run_epoch=session.run_epoch,
                        expected_session_instance_id=session.instance_id,
                        terminal_status=SessionStatus.FAILED,
                        terminalization_only=True,
                        inactive_for_seconds=0,
                    )
                )
            active = await store.load_active_model_completion_stage(session.id)
            assert active is not None and active.stage == stage
            assert await store.load(session.id) == before
            assert (
                await store.load_model_completion_stage_settlement(session.id, stage.stage_id)
                is None
            )
        finally:
            await app.aclose()

    asyncio.run(scenario())
