from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from cayu.runtime import CayuApp
from cayu.server import BasicAuth, DashboardConfig, ServerConfig, create_server
from cayu.server._event_side_effect_health import EventSideEffectRecoveryLoop


def test_operational_health_requires_auth_and_is_read_only():
    app = CayuApp()
    server = create_server(
        app,
        config=ServerConfig.protected(
            BasicAuth(username="operator", password="health-password"),
            dashboard=DashboardConfig(enabled=False),
        ),
    )
    with TestClient(server) as client:
        assert client.get("/api/health").json() == {"ok": True}
        for path in ["/api/event-side-effects/health", "/api/event-side-effects/deliveries"]:
            assert client.get(path).status_code == 401
        response = client.get(
            "/api/event-side-effects/health", auth=("operator", "health-password")
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["durable"]["outstanding_total"] == 0
        assert body["recovery_loop"]["scope"] == "process_local_reset_on_restart"
        assert body["recovery_loop"]["last_success_at"] is not None
        assert body["recovery_loop"]["consecutive_failures"] == 0
        response = client.get(
            "/api/event-side-effects/deliveries?cursor=bad", auth=("operator", "health-password")
        )
        assert response.status_code == 400
        response = client.get(
            "/api/event-side-effects/deliveries?status=invalid",
            auth=("operator", "health-password"),
        )
        assert response.status_code == 422
    assert server.state.cayu_event_side_effect_recovery.state == "stopped"


def test_failed_health_does_not_mutate_or_expose_error():
    app = CayuApp()

    async def unavailable():
        raise OSError("password=private")

    app.get_persisted_event_side_effect_health = unavailable
    server = create_server(
        app, config=ServerConfig.local_development(dashboard=DashboardConfig(enabled=False))
    )
    client = TestClient(server)
    response = client.get("/api/event-side-effects/health")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "event_side_effect_store_unavailable"
    assert "private" not in response.text
    assert client.get("/api/health").json() == {"ok": True}

    async def invalid_stored_row(query):
        raise ValueError("private corrupt row")

    app.query_persisted_event_side_effect_deliveries = invalid_stored_row
    response = client.get("/api/event-side-effects/deliveries")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "event_side_effect_store_unavailable"
    assert "private" not in response.text
    assert client.get("/api/event-side-effects/deliveries?cursor=bad").status_code == 400


def test_loop_failure_success_saturation_and_cancellation(monkeypatch):
    import cayu.server as module

    async def run():
        app = CayuApp()
        status = EventSideEffectRecoveryLoop(interval_seconds=0.01, batch_limit=2)
        monkeypatch.setattr(module, "_PERSISTED_EVENT_SIDE_EFFECT_RECOVERY_BATCH_SIZE", 2)
        monkeypatch.setattr(module, "_PERSISTED_EVENT_SIDE_EFFECT_RECOVERY_INTERVAL_SECONDS", 0.01)
        attempts = 0
        saturated = asyncio.Event()
        proceed = asyncio.Event()

        async def sweep(*, limit):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("private error\nsecret")
            if attempts == 2:
                return [object(), object()]
            saturated.set()
            await proceed.wait()
            return []

        app.recover_persisted_event_side_effects = sweep
        await module._recover_persisted_event_side_effects_during_startup(
            app, timeout_s=1, status=status
        )
        assert status.consecutive_failures == 1
        assert "secret" not in status.model_dump_json()
        task = module._start_persisted_event_side_effect_recovery(app, status)
        await asyncio.wait_for(saturated.wait(), 3)
        assert status.last_success_saturated
        assert status.last_delivered_count == 2
        assert status.consecutive_failures == 0
        failures = status.sweep_failures
        await module._stop_persisted_event_side_effect_recovery(task)
        assert status.state == "stopped"
        assert status.sweep_failures == failures

    asyncio.run(run())


def test_saturation_counter_survives_drain_idle_and_failure(monkeypatch):
    import cayu.server as module

    async def run():
        app = CayuApp()
        status = EventSideEffectRecoveryLoop(interval_seconds=30, batch_limit=2)
        monkeypatch.setattr(module, "_PERSISTED_EVENT_SIDE_EFFECT_RECOVERY_BATCH_SIZE", 2)
        batches = iter([[object(), object()], [object(), object()], [object()], [], []])

        async def sweep(*, limit):
            assert limit == 2
            return next(batches)

        app.recover_persisted_event_side_effects = sweep
        await module._recover_persisted_event_side_effects_until_idle(app, status)
        assert status.last_delivered_count == 0
        assert not status.last_success_saturated
        assert status.delivered_rows == 5
        assert status.saturated_batches == 2
        await module._recover_persisted_event_side_effects_until_idle(app, status)
        assert status.saturated_batches == 2

        async def fail(*, limit):
            raise OSError("transient failure")

        app.recover_persisted_event_side_effects = fail
        await module._recover_persisted_event_side_effects_during_startup(
            app, timeout_s=1, status=status
        )
        assert status.consecutive_failures == 1
        assert status.saturated_batches == 2
        server = create_server(
            app, config=ServerConfig.local_development(dashboard=DashboardConfig(enabled=False))
        )
        server.state.cayu_event_side_effect_recovery = status
        response = TestClient(server).get("/api/event-side-effects/health")
        assert response.status_code == 200
        assert response.json()["recovery_loop"]["saturated_batches"] == 2
        assert (
            EventSideEffectRecoveryLoop(interval_seconds=30, batch_limit=2).saturated_batches == 0
        )

    asyncio.run(run())
