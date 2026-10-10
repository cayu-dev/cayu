from __future__ import annotations

import asyncio

import pytest
from tests.core._execution_profile_fixtures import versioned_test_provider_identity
from tests.core.test_recovery_plans import _create_running_session

from cayu import (
    AgentSpec,
    CayuApp,
    InMemorySessionStore,
    RecoveryBlockerCode,
    RecoveryExecutionRequest,
    RecoveryPlanAction,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
    SessionStatus,
    SQLiteSessionStore,
)
from cayu.providers.base import ModelProvider
from cayu.runtime._model_completion_contracts import ModelCompletionManualRecoveryRequired


class _Provider(ModelProvider):
    name = "fake"

    def __init__(self, version):
        self.version = version

    @property
    def execution_profile_identity(self):
        return versioned_test_provider_identity(self, behavior_version=self.version)

    async def stream(self, request):
        raise AssertionError("Startup recovery must not execute a new model request")
        yield  # pragma: no cover


def _app(store, version):
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(_Provider(version), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return app


async def _seed(store, app, session_id, status=SessionStatus.INTERRUPTING, *, create=True):
    if create:
        await _create_running_session(store, app, session_id)
    payload = {
        "reason": "operator stop",
        "metadata": {},
        "requested_by": None,
        "interruption_type": "operator_requested",
    }
    await store.transform_checkpoint(
        session_id,
        lambda _session, checkpoint: {
            **checkpoint,
            "pending_session_interrupt": payload,
            "pending_interruption_cascade": {
                "attempt_id": "cascade-" + session_id,
                "interrupt_payload": payload,
            },
            # A retained application checkpoint prevents the pristine zero-work shortcut.
            "private_payload": "startup-private-canary",
        },
    )
    await store.update_status(session_id, status)


async def _state(store, session_id):
    return (
        await store.load(session_id),
        await store.load_checkpoint(session_id),
        await store.load_events(session_id),
    )


async def _assert_isolation_and_restored_registration(store):
    old = _app(store, "3")
    replacement = _app(store, "4")
    await _seed(store, old, "blocked-old")
    await _seed(store, replacement, "later-interrupting")
    await _seed(store, replacement, "later-interrupted", SessionStatus.INTERRUPTED)
    before = await _state(store, "blocked-old")
    assert (
        await replacement.resume_pending_interruption_cascades(interrupting_inactive_for_seconds=0)
        == 2
    )
    assert await replacement.drain_background_interruptions(timeout_s=5)
    for session_id in ("later-interrupting", "later-interrupted"):
        assert (await store.load(session_id)).status == SessionStatus.INTERRUPTED
        assert "pending_interruption_cascade" not in await store.load_checkpoint(session_id)
    report = await replacement.get_startup_recovery_status()
    assert report.completed
    assert report.scheduled_roots == 2
    assert report.blocked_session_count == 1
    assert report.blocked_sessions[0].session_id == "blocked-old"
    assert report.blocked_sessions[0].blocker_codes == (
        RecoveryBlockerCode.REGISTRATION_INCOMPATIBLE,
    )
    assert "startup-private-canary" not in report.model_dump_json()
    assert before == await _state(store, "blocked-old")
    assert (
        await replacement.resume_pending_interruption_cascades(interrupting_inactive_for_seconds=0)
        == 0
    )
    assert before == await _state(store, "blocked-old")

    restored = _app(store, "3")
    plan = await restored.plan_recovery(
        RecoveryPlanRequest(
            selection=RecoveryPlanSelection(
                session_ids=("blocked-old",),
                inactive_for_seconds=0,
            )
        )
    )
    assert RecoveryPlanAction.AUTOMATIC_REPAIR in plan.items[0].allowed_actions
    receipt = await restored.execute_recovery(
        RecoveryExecutionRequest(
            plan=plan,
            execution_id="restore-compatible-registration",
        )
    )
    assert receipt.items[0].final_session_status == SessionStatus.INTERRUPTED
    assert await restored.drain_background_interruptions(timeout_s=5)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_startup_continues_and_restored_registration_recovers(backend, sqlite_resources):
    async def scenario():
        async with sqlite_resources as resources:
            store = (
                InMemorySessionStore()
                if backend == "memory"
                else resources.own(SQLiteSessionStore(resources.path("sessions.sqlite")))
            )
            await _assert_isolation_and_restored_registration(store)

    asyncio.run(scenario())


def test_postgres_startup_continues_and_restored_registration_recovers(postgres_dsn):
    async def scenario():
        from cayu import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        store = PostgresSessionStore(
            postgres_dsn, min_size=1, max_size=2, schema_mode=SchemaMode.CREATE
        )
        try:
            await _assert_isolation_and_restored_registration(store)
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_type", [OSError, RuntimeError])
def test_unclassified_preflight_failure_still_fails_startup(monkeypatch, failure_type):
    async def scenario():
        store = InMemorySessionStore()
        app = _app(store, "3")
        await _seed(store, app, "store-unavailable")

        async def fail(**kwargs):
            raise failure_type("store unavailable")

        monkeypatch.setattr(app._incomplete_recovery, "preflight_incomplete_session", fail)
        with pytest.raises(failure_type, match="store unavailable"):
            await app.resume_pending_interruption_cascades(interrupting_inactive_for_seconds=0)
        assert not (await app.get_startup_recovery_status()).completed
        assert (await app.get_startup_recovery_status()).status == "failed"
        assert (await app.get_startup_recovery_status()).blocked_session_count == 0

    asyncio.run(scenario())


def test_blocked_summary_is_bounded_and_invalid_markers_do_not_stop_the_sweep():
    async def scenario():
        from cayu import Message, RunRequest, SessionIdentity

        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        for index in range(105):
            session_id = f"invalid-cascade-{index}"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "private-startup-message")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await store.update_status(session_id, SessionStatus.INTERRUPTED)
            await store.checkpoint(
                session_id,
                {
                    "pending_interruption_cascade": {"attempt_id": " ", "interrupt_payload": {}},
                },
            )
        assert await app.resume_pending_interruption_cascades() == 0
        result = await app.get_startup_recovery_status()
        assert result.completed
        assert result.blocked_session_count == 105
        assert len(result.blocked_sessions) == 100
        assert result.blocked_sessions_truncated
        assert all(
            item.blocker_codes == (RecoveryBlockerCode.INVALID_DURABLE_STATE,)
            for item in result.blocked_sessions
        )
        assert "private-startup-message" not in result.model_dump_json()

    asyncio.run(scenario())


def test_missing_registration_is_reported_without_mutating_the_session():
    async def scenario():
        store = InMemorySessionStore()
        original = _app(store, "3")
        await _seed(store, original, "missing-registration")
        before = await _state(store, "missing-registration")
        unregistered = CayuApp(session_store=store, enable_logging=False)
        assert (
            await unregistered.resume_pending_interruption_cascades(
                interrupting_inactive_for_seconds=0
            )
            == 0
        )
        report = await unregistered.get_startup_recovery_status()
        assert report.blocked_sessions[0].blocker_codes == (
            RecoveryBlockerCode.REGISTRATION_UNAVAILABLE,
        )
        assert before == await _state(store, "missing-registration")

    asyncio.run(scenario())


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def startup_store(request):
    from contextlib import asynccontextmanager

    dsn = None
    if request.param == "postgres":
        from uuid import uuid4

        import psycopg
        from psycopg import sql
        from psycopg.conninfo import make_conninfo

        parent_dsn = request.getfixturevalue("postgres_dsn")
        database = "startup_" + uuid4().hex
        with psycopg.connect(parent_dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
        dsn = make_conninfo(parent_dsn, dbname=database)

        def drop_database():
            with psycopg.connect(parent_dsn, autocommit=True) as connection:
                connection.execute(
                    sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
                )

        request.addfinalizer(drop_database)
    # Request this only after the PostgreSQL case has had its chance to skip.
    sqlite_resources = request.getfixturevalue("sqlite_resources")

    @asynccontextmanager
    async def acquire():
        async with sqlite_resources as resources:
            if dsn is not None:
                from cayu import PostgresSessionStore
                from cayu.storage.migrations import SchemaMode

                store = PostgresSessionStore(
                    dsn, min_size=1, max_size=2, schema_mode=SchemaMode.CREATE
                )
                try:
                    yield store
                finally:
                    await store.close()
            elif request.param == "sqlite":
                yield resources.own(SQLiteSessionStore(resources.path("startup.sqlite")))
            else:
                yield InMemorySessionStore()

    return acquire


def test_manual_model_decision_does_not_block_later_startup_roots(startup_store):
    from tests.core.test_model_completion_recovery import (
        _RecordingProvider,
        _register_runtime,
        _stage_in_flight_model_boundary,
    )

    async def scenario():
        async with startup_store() as store:
            provider = _RecordingProvider()
            app = _register_runtime(store, provider)
            await asyncio.create_task(
                _stage_in_flight_model_boundary(
                    store,
                    session_id="manual-model",
                    provider_name=provider.name,
                    reservation_ids=(),
                )
            )
            await _seed(store, app, "manual-model", create=False)
            await _seed(store, app, "later-model-root", SessionStatus.INTERRUPTED)
            before = await _state(store, "manual-model")
            assert (
                await app.resume_pending_interruption_cascades(interrupting_inactive_for_seconds=0)
                == 1
            )
            assert await app.drain_background_interruptions(timeout_s=5)
            report = await app.get_startup_recovery_status()
            assert report.status == "completed"
            assert report.blocked_sessions[0].blocker_codes == (
                RecoveryBlockerCode.MODEL_EFFECT_OUTCOME_UNKNOWN,
            )
            assert before == await _state(store, "manual-model")
            assert provider.requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize("race", ["deleted", "changing"])
def test_startup_skips_concurrent_snapshot_changes(startup_store, monkeypatch, race):
    async def scenario():
        async with startup_store() as store:
            app = _app(store, "3")
            await _seed(store, app, f"racing-root-{race}")
            await _seed(store, app, f"later-race-{race}", SessionStatus.INTERRUPTED)
            coordinator = app._recovery_plan_coordinator
            original = coordinator._project_item_state
            load = store.load
            removed = False

            async def concurrent_load(session_id):
                if removed and session_id == f"racing-root-{race}":
                    return None
                return await load(session_id)

            monkeypatch.setattr(store, "load", concurrent_load)

            async def change(session, checkpoint, **kwargs):
                nonlocal removed
                item = await original(session, checkpoint, **kwargs)
                if session.id == f"racing-root-{race}":
                    if race == "deleted":
                        # Simulate another owner completing deletion between snapshot reads.
                        removed = True
                    else:
                        await store.transform_checkpoint(
                            session.id,
                            lambda _session, checkpoint: {
                                **checkpoint,
                                "private_counter": checkpoint.get("private_counter", 0) + 1,
                            },
                        )
                return item

            monkeypatch.setattr(coordinator, "_project_item_state", change)
            assert (
                await app.resume_pending_interruption_cascades(interrupting_inactive_for_seconds=0)
                == 1
            )
            assert await app.drain_background_interruptions(timeout_s=5)
            report = await app.get_startup_recovery_status()
            assert report.status == "completed"
            assert report.skipped_session_count == 1
            assert report.blocked_session_count == 0

    asyncio.run(scenario())


def test_startup_isolates_registration_change_after_preflight(startup_store, monkeypatch):
    async def scenario():
        async with startup_store() as store:
            app = _app(store, "3")
            changed = _app(store, "4")
            await _seed(store, app, "registration-race")
            await _seed(store, app, "later-registration-root", SessionStatus.INTERRUPTED)
            original = app._recovery_plan_coordinator.startup_interruption_blockers

            async def change(session_id, inactive_for_seconds):
                codes = await original(session_id, inactive_for_seconds)
                app._provider_registry._providers["fake"] = (
                    changed._provider_registry.registrations["fake"]
                )
                return codes

            monkeypatch.setattr(
                app._recovery_plan_coordinator, "startup_interruption_blockers", change
            )
            assert (
                await app.resume_pending_interruption_cascades(interrupting_inactive_for_seconds=0)
                == 1
            )
            assert await app.drain_background_interruptions(timeout_s=5)
            report = await app.get_startup_recovery_status()
            assert report.blocked_sessions[0].blocker_codes == (
                RecoveryBlockerCode.REGISTRATION_INCOMPATIBLE,
            )

    asyncio.run(scenario())


def test_preflight_manual_model_recovery_blocks_without_planner_blocker(monkeypatch):
    async def scenario():
        store = InMemorySessionStore()
        app = _app(store, "3")
        await _seed(store, app, "preflight-manual")
        await _seed(store, app, "later-preflight-root", SessionStatus.INTERRUPTED)
        before = await _state(store, "preflight-manual")

        async def manual(**kwargs):
            raise ModelCompletionManualRecoveryRequired("private-preflight-detail")

        async def unexpected(*args, **kwargs):
            raise AssertionError("A manual model decision must not reach automatic recovery.")

        monkeypatch.setattr(app._incomplete_recovery, "preflight_incomplete_session", manual)
        monkeypatch.setattr(app._incomplete_recovery, "recover_incomplete_session", unexpected)
        assert (
            await app.resume_pending_interruption_cascades(interrupting_inactive_for_seconds=0) == 1
        )
        assert await app.drain_background_interruptions(timeout_s=5)
        report = await app.get_startup_recovery_status()
        assert report.status == "completed"
        assert report.blocked_sessions[0].session_id == "preflight-manual"
        assert report.blocked_sessions[0].blocker_codes == (
            RecoveryBlockerCode.MODEL_EFFECT_OUTCOME_UNKNOWN,
        )
        assert before == await _state(store, "preflight-manual")
        assert "private-preflight-detail" not in report.model_dump_json()

    asyncio.run(scenario())


def test_startup_isolates_manual_model_recovery_after_preflight(monkeypatch):
    async def scenario():
        store = InMemorySessionStore()
        app = _app(store, "3")
        await _seed(store, app, "manual-after-preflight")
        await _seed(store, app, "later-manual-root", SessionStatus.INTERRUPTED)

        async def allowed(*args):
            return ()

        async def manual(*args, **kwargs):
            raise ModelCompletionManualRecoveryRequired("model outcome changed")

        monkeypatch.setattr(
            app._recovery_plan_coordinator, "startup_interruption_blockers", allowed
        )
        monkeypatch.setattr(app._incomplete_recovery, "recover_incomplete_session", manual)
        assert (
            await app.resume_pending_interruption_cascades(interrupting_inactive_for_seconds=0) == 1
        )
        assert await app.drain_background_interruptions(timeout_s=5)
        report = await app.get_startup_recovery_status()
        assert report.status == "completed"
        assert report.blocked_sessions[0].blocker_codes == (
            RecoveryBlockerCode.MODEL_EFFECT_OUTCOME_UNKNOWN,
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("deleted", ["before_recovery", "after_recovery"])
def test_startup_skips_root_deleted_after_preflight(monkeypatch, deleted):
    async def scenario():
        store = InMemorySessionStore()
        app = _app(store, "3")
        await _seed(store, app, "deleted-root")
        await _seed(store, app, "later-deleted-root", SessionStatus.INTERRUPTED)
        load = store.load

        async def concurrent_load(session_id):
            if session_id == "deleted-root":
                return None
            return await load(session_id)

        preflight = app._recovery_plan_coordinator.startup_interruption_blockers
        recover = app._incomplete_recovery.recover_incomplete_session

        async def delete_after_preflight(session_id, inactive_for_seconds):
            codes = await preflight(session_id, inactive_for_seconds)
            if session_id == "deleted-root" and deleted == "before_recovery":
                # The root is still INTERRUPTING and cannot be deleted through the
                # store API; hide it from reads as a concurrent deletion would.
                monkeypatch.setattr(store, "load", concurrent_load)
            return codes

        async def delete_after_recovery(request, **kwargs):
            result = await recover(request, **kwargs)
            if request.session_id == "deleted-root" and deleted == "after_recovery":
                await store.delete_session(request.session_id)
            return result

        monkeypatch.setattr(
            app._recovery_plan_coordinator, "startup_interruption_blockers", delete_after_preflight
        )
        monkeypatch.setattr(
            app._incomplete_recovery, "recover_incomplete_session", delete_after_recovery
        )
        assert (
            await app.resume_pending_interruption_cascades(interrupting_inactive_for_seconds=0) == 1
        )
        assert await app.drain_background_interruptions(timeout_s=5)
        report = await app.get_startup_recovery_status()
        assert report.status == "completed"
        assert report.skipped_session_count == 1
        assert report.blocked_session_count == 0

    asyncio.run(scenario())


def test_startup_reports_live_claims_as_deferred_and_counts_sweeps(monkeypatch):
    async def scenario():
        store = InMemorySessionStore()
        app = _app(store, "3")
        assert (await app.get_startup_recovery_status()).status == "not_started"
        await _seed(store, app, "live-owner")

        async def live_claim(*args):
            return (
                RecoveryBlockerCode.ACTIVE_RECOVERY_CLAIM,
                RecoveryBlockerCode.ACTIVE_TASK_CLAIM,
            )

        monkeypatch.setattr(
            app._recovery_plan_coordinator, "startup_interruption_blockers", live_claim
        )
        for sweep_count in (1, 2):
            assert (
                await app.resume_pending_interruption_cascades(interrupting_inactive_for_seconds=0)
                == 0
            )
            report = await app.get_startup_recovery_status()
            assert report.status == "completed"
            assert report.sweep_count == sweep_count
            assert report.deferred_session_count == 1
            assert report.blocked_session_count == 0

    asyncio.run(scenario())
