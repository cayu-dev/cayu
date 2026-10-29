"""Usage filters and batched start counts through HTTP."""

from datetime import UTC, datetime, timedelta

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient
from tests.core.test_cloud_reporting import _app, _start

from cayu.server import ServerConfig, create_server


def test_counts_filter_current_specialist_and_return_start_buckets():
    app = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "writer-run", "writer")
        client.portal.call(_start, app, "question-run", "asker")
        until = datetime.now(UTC) + timedelta(seconds=1)
        response = client.get(
            "/api/sessions/overview/counts",
            params={
                "agent_name": "writer",
                "since": (until - timedelta(days=2)).isoformat(),
                "until": until.isoformat(),
                "bucket_unit": "day",
                "timezone": "UTC",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["total_count"] == "1"
        assert sum(int(b["total_count"]) for b in body["series"]["buckets"]) == 1
        assert sum(int(b["failed"]) for b in body["series"]["buckets"]) == 0


def test_usage_series_capabilities_and_endpoint_use_current_contract():
    app = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "done", "writer")
        now = datetime.now(UTC)
        response = client.post(
            "/api/usage/rollup",
            json={
                "start_at": (now - timedelta(days=2)).isoformat(),
                "end_at": (now + timedelta(seconds=1)).isoformat(),
                "series": {"unit": "day", "timezone": "UTC"},
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["cost"] is None
        assert sum(int(b["totals"]["model_steps"]) for b in body["series"]["buckets"]) == int(
            body["totals"]["model_steps"]
        )
        assert response.headers["cache-control"] == "private, no-store"
        capabilities = client.get("/api/contract").json()["capabilities"]
        assert capabilities["surfaces"]["usage_series"]["read"]["enabled"] is True
        assert capabilities["surfaces"]["session_overview_series"]["read"]["enabled"] is True


def test_count_series_rejects_invalid_zone_and_resolution_without_echoing_it():
    app = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.get(
            "/api/sessions/overview/counts",
            params={"bucket_unit": "day", "timezone": "../secret-canary"},
        )
        assert response.status_code == 422
        assert "secret-canary" not in response.text


def test_all_current_session_filters_and_histogram_share_one_counts_traversal(monkeypatch):
    app = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "writer", "writer")
        client.portal.call(_start, app, "asker", "asker")
        store = app.session_store
        original = store.list_sessions
        calls = []

        async def observed(query=None):
            calls.append(query)
            return await original(query)

        monkeypatch.setattr(store, "list_sessions", observed)
        now = datetime.now(UTC)
        filters = {
            "agent_name": "writer",
            "provider_name": "not-the-provider",
            "model": "not-the-model",
            "environment_name": "not-the-environment",
        }
        for name, value in filters.items():
            calls.clear()
            response = client.get(
                "/api/sessions/overview/counts",
                params={
                    name: value,
                    "since": (now - timedelta(hours=24)).isoformat(),
                    "until": (now + timedelta(seconds=1)).isoformat(),
                    "bucket_unit": "hour",
                },
            )
            assert response.status_code == 200, response.text
            body = response.json()
            expected = 1 if name == "agent_name" else 0
            assert int(body["total_count"]) == expected
            assert sum(int(b["total_count"]) for b in body["series"]["buckets"]) == expected
            assert len(calls) == 1


def test_custom_store_must_advertise_series_and_cannot_omit_requested_projection():
    from cayu.applications import CayuApp
    from cayu.sessions.memory import InMemorySessionStore

    class UnsupportedSeries(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        supports_usage_series = False

    class MissingSeries(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def aggregate_usage(self, query):
            result = await super().aggregate_usage(query)
            return result.model_copy(update={"series": None})

    body = {
        "start_at": "2026-10-01T00:00:00Z",
        "end_at": "2026-10-02T00:00:00Z",
        "series": {"unit": "hour"},
    }
    for store, status in [(UnsupportedSeries(), 501), (MissingSeries(), 500)]:
        with TestClient(
            create_server(CayuApp(session_store=store), config=ServerConfig.local_development())
        ) as client:
            response = client.post("/api/usage/rollup", json=body)
            assert response.status_code == status, response.text
            assert (
                client.post(
                    "/api/usage/rollup", json={k: v for k, v in body.items() if k != "series"}
                ).status_code
                == 200
            )


def test_series_preserves_headline_pricing_and_start_failure_basis():
    from tests.server.server_test_support import _price_book_payload

    from cayu.sessions.records import SessionStatus

    app = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "done", "writer")
        now = datetime.now(UTC) + timedelta(seconds=1)
        body = {
            "start_at": (now - timedelta(days=2)).isoformat(),
            "end_at": now.isoformat(),
            "pricing": _price_book_payload(),
        }
        plain = client.post("/api/usage/rollup", json=body)
        series = client.post("/api/usage/rollup", json={**body, "series": {"unit": "day"}})
        assert plain.status_code == series.status_code == 200
        assert plain.json()["cost"] == series.json()["cost"]
        assert plain.json()["cost"] is not None
        # Counts describe the selected starts' current state, not failure events.
        client.portal.call(app.session_store.update_status, "done", SessionStatus.FAILED)
        counts = client.get(
            "/api/sessions/overview/counts",
            params={
                "since": body["start_at"],
                "until": body["end_at"],
                "agent_name": "writer",
                "bucket_unit": "day",
            },
        )
        assert counts.status_code == 200, counts.text
        result = counts.json()
        assert result["failed"] == "1"
        assert sum(int(b["failed"]) for b in result["series"]["buckets"]) == 1


def test_filtered_histogram_keeps_older_running_and_recent_usage():
    from cayu import RunRequest
    from cayu.events import Event, EventType
    from cayu.sessions.memory import InMemorySessionStore
    from cayu.sessions.records import SessionIdentity

    now = datetime.now(UTC)
    clock = now - timedelta(days=10)
    store = InMemorySessionStore(ownership_clock=lambda: clock)
    app = _app(store)

    async def seed():
        await store.create(
            RunRequest(session_id="old-running", agent_name="writer", messages=[]),
            identity=SessionIdentity(provider_name="configured", model="configured-model"),
        )
        await store.append_event(
            "old-running",
            Event(
                id="recent",
                session_id="old-running",
                type=EventType.MODEL_COMPLETED,
                timestamp=now,
                payload={
                    "usage_metrics": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
                },
            ),
        )

    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(seed)
        response = client.get(
            "/api/sessions/overview/counts",
            params={
                "since": (now - timedelta(days=1)).isoformat(),
                "until": (now + timedelta(seconds=1)).isoformat(),
                "agent_name": "writer",
                "provider_name": "configured",
                "model": "configured-model",
                "bucket_unit": "hour",
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["total_count"] == "0" and data["running_now"] == "1"
        assert data["tokens"]["input"] == "5" and data["tokens"]["output"] == "2"
        assert all(bucket["total_count"] == "0" for bucket in data["series"]["buckets"])


def test_filtered_start_histogram_preserves_eval_and_fork_exclusions():
    from tests.core.test_cloud_reporting import _drain

    from cayu.sessions import ForkSessionRequest
    from cayu.sessions._run_identity import eval_trial_scope

    app = _app()

    async def seed():
        await _start(app, "normal", "writer")
        async with eval_trial_scope():
            await _start(app, "trial", "writer")
        await _drain(
            app.fork_session(ForkSessionRequest(source_session_id="normal", session_id="fork"))
        )

    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(seed)
        for include, expected in [(False, 1), (True, 3)]:
            response = client.get(
                "/api/sessions/overview/counts",
                params={
                    "agent_name": "writer",
                    "bucket_unit": "day",
                    "include_eval": str(include).lower(),
                    "include_forks": str(include).lower(),
                },
            )
            assert response.status_code == 200, response.text
            data = response.json()
            assert int(data["total_count"]) == expected
            assert sum(int(b["total_count"]) for b in data["series"]["buckets"]) == expected


def test_histogram_never_returns_partial_success_after_later_page_failure(monkeypatch):
    from cayu import RunRequest
    from cayu.sessions.records import SessionIdentity

    app = _app()
    store = app.session_store

    async def seed():
        for i in range(201):
            await store.create(
                RunRequest(session_id=f"row-{i}", agent_name="writer", messages=[]),
                identity=SessionIdentity(provider_name="test", model="test"),
            )

    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(seed)
        original = store.list_sessions

        async def fail_second(query=None):
            if query.cursor is not None:
                raise OSError("private-series-error-canary")
            return await original(query)

        monkeypatch.setattr(store, "list_sessions", fail_second)
        response = client.get(
            "/api/sessions/overview/counts", params={"agent_name": "writer", "bucket_unit": "hour"}
        )
        assert response.status_code == 503
        assert "series" not in response.json()
        assert "private-series-error-canary" not in response.text


def test_sqlite_http_series_excludes_hosted_progress_only_bucket(tmp_path):
    from tests.core.test_usage_series import _append_hosted_progress_and_terminal

    from cayu.storage.sqlite import SQLiteSessionStore

    store = SQLiteSessionStore(tmp_path / "hosted-http.sqlite")
    with TestClient(create_server(_app(store), config=ServerConfig.local_development())) as client:
        start = client.portal.call(
            _append_hosted_progress_and_terminal, store, "hosted", "outcome_unknown"
        )
        for hours in (1, 2):
            response = client.post(
                "/api/usage/rollup",
                json={
                    "start_at": start.isoformat(),
                    "end_at": (start + timedelta(hours=hours)).isoformat(),
                    "series": {"unit": "hour"},
                    "session_group_limit": 10,
                },
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["totals"]["session_count"] == str(hours - 1)
            assert [b["totals"]["session_count"] for b in body["series"]["buckets"]] == [
                str(i) for i in range(hours)
            ]
        client.portal.call(store.close)
