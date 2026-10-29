"""Overview acceptance through the control plane and the unchanged Cloud wire."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient
from tests.core.test_cloud_reporting import _app, _start

from cayu.cloud_reporting import read_cloud_activity
from cayu.server import ServerConfig, create_server


def test_overview_shares_cloud_state_and_preserves_wire():
    app = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        for session_id, agent in [("done", "writer"), ("approve", "approver"), ("input", "asker")]:
            client.portal.call(_start, app, session_id, agent)
        cloud = client.portal.call(read_cloud_activity, app)
        response = client.get("/api/sessions", params={"include": "overview"})
        assert response.status_code == 200
        overviews = {row["session_id"]: row for row in response.json()["overviews"]}
        assert {key: row["state"] for key, row in overviews.items()} == {
            "done": "completed",
            "approve": "waiting_approval",
            "input": "waiting_input",
        }
        for run in cloud.runs:
            row = overviews[run.session_id]
            assert row["state"] == run.status.value
            assert row["run_id"] == run.run_id
            assert int(row["model_steps"]) == run.model_steps
            assert int(row["tool_calls"]) == run.tool_calls
            assert row["cost"] is None
            assert row["uses"] == {"browser": False, "subagents": "0", "artifacts": "0"}
            assert row["eval"] is False and row["forked"] is False
            assert "title" not in run.model_dump()
            assert set(run.model_dump()) == {
                "run_id",
                "root_run_id",
                "parent_run_id",
                "session_id",
                "root_session_id",
                "parent_session_id",
                "source",
                "forked",
                "fork_controller",
                "eval",
                "agent_name",
                "status",
                "reason",
                "started_by",
                "started_at",
                "ended_at",
                "release_id",
                "model_steps",
                "tool_calls",
                "cost",
                "waiting_for",
            }
        batch = client.get(
            "/api/sessions", params=[("include", "overview"), ("ids", "done"), ("ids", "done")]
        )
        assert batch.status_code == 200
        assert [row["session_id"] for row in batch.json()["overviews"]] == ["done"]
        assert (
            client.get("/api/sessions", params=[("ids", "done"), ("offset", "1")]).status_code
            == 422
        )


def test_overview_counts_and_capability_are_real_server_reads():
    app = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "done", "writer")
        client.portal.call(_start, app, "input", "asker")
        response = client.get("/api/sessions/overview/counts")
        assert response.status_code == 200
        body = response.json()
        assert body["total_count"] == "2"
        assert body["counts_by_state"]["waiting_input"] == "1"
        assert body["running_now"] == "0"
        assert response.headers["cache-control"] == "private, no-store"
        surface = client.get("/api/contract").json()["capabilities"]["surfaces"]["session_overview"]
        assert surface["read"]["enabled"] is True
        assert surface["mutate"]["enabled"] is False


def test_neutral_reader_is_cloud_status_owner():
    from cayu.cloud_reporting import CloudRunStatus
    from cayu.sessions.overview import SessionOverviewStatus

    assert CloudRunStatus is SessionOverviewStatus


def test_counts_exhaust_pages_and_never_publish_partial_success(monkeypatch):
    from cayu import RunRequest
    from cayu.sessions.records import SessionIdentity

    app = _app()
    store = app.session_store

    async def populate():
        for index in range(1001):
            await store.create(
                RunRequest(agent_name="writer", session_id=f"row-{index}", messages=[]),
                identity=SessionIdentity(provider_name="test", model="test"),
            )

    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(populate)
        read = store.list_sessions
        calls = []

        async def fail_second_page(query=None):
            calls.append(query.cursor)
            if query.cursor is not None:
                raise OSError("private-store-canary")
            return await read(query)

        monkeypatch.setattr(store, "list_sessions", fail_second_page)
        response = client.get("/api/sessions/overview/counts")
        assert response.status_code == 503
        assert "total_count" not in response.json()
        assert "private-store-canary" not in response.text
        assert len(calls) == 2
        monkeypatch.setattr(store, "list_sessions", read)
        response = client.get("/api/sessions/overview/counts")
        assert response.status_code == 200
        assert response.json()["total_count"] == "1001"
        assert response.json()["running_now"] == "1001"
        row = client.get("/api/sessions", params={"include": "overview", "ids": "row-0"})
        assert row.status_code == 200
        assert row.json()["overviews"][0]["state"] == "running"


def test_eval_and_fork_flags_and_default_count_exclusions():
    from tests.core.test_cloud_reporting import _drain

    from cayu.sessions import ForkSessionRequest
    from cayu.sessions._run_identity import eval_trial_scope

    app = _app()

    async def populate():
        await _start(app, "normal", "writer")
        async with eval_trial_scope():
            await _start(app, "eval", "writer")
        await _drain(
            app.fork_session(ForkSessionRequest(source_session_id="normal", session_id="fork"))
        )

    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(populate)
        response = client.get("/api/sessions", params={"include": "overview"})
        assert response.status_code == 200
        rows = {row["session_id"]: row for row in response.json()["overviews"]}
        assert rows["eval"]["eval"] is True
        assert rows["fork"]["forked"] is True
        assert client.get("/api/sessions/overview/counts").json()["total_count"] == "1"
        included = client.get(
            "/api/sessions/overview/counts",
            params={"include_eval": "true", "include_forks": "true"},
        )
        assert included.json()["total_count"] == "3"


def test_replaced_incarnation_is_not_joined_to_listed_row(monkeypatch):
    app = _app()
    store = app.session_store
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "reused", "writer")
        read = store.read_usage_accounting
        changed = False

        async def replace_then_read(query, **kwargs):
            nonlocal changed
            if not changed:
                changed = True
                await store.delete_session("reused")
                await _start(app, "reused", "asker")
            return await read(query, **kwargs)

        monkeypatch.setattr(store, "read_usage_accounting", replace_then_read)
        response = client.get("/api/sessions", params={"include": "overview"})
        assert response.status_code == 200
        assert response.json()["sessions"] == response.json()["overviews"] == []
        refreshed = client.get("/api/sessions", params={"include": "overview"})
        assert refreshed.json()["overviews"][0]["state"] == "waiting_input"


def test_overview_metadata_is_bounded_redacted_and_never_sent_to_cloud(monkeypatch, caplog, capsys):
    import warnings

    from cayu.vaults.redaction import SecretRedactor

    app = _app(secret_redactor=SecretRedactor(["private-canary"]))
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "named", "writer")
        metadata = {
            "cayu:title": "Hello private-canary",
            "cayu:summary": "Summary private-canary",
            "cayu:summary_tone": "attention",
        }
        # Model stored runtime-owned metadata; public update_metadata refuses these keys.
        load = app.session_store.load

        async def named_record(session_id):
            session = await load(session_id)
            if session is not None and session_id == "named":
                return session.model_copy(update={"metadata": {**session.metadata, **metadata}})
            return session

        monkeypatch.setattr(app.session_store, "load", named_record)
        response = client.get("/api/sessions", params={"include": "overview"})
        assert response.status_code == 200
        assert "private-canary" not in response.text
        row = response.json()["overviews"][0]
        assert row["title"].startswith("Hello ") and row["summary_tone"] == "attention"
        cloud = client.portal.call(read_cloud_activity, app)
        assert "Hello" not in cloud.model_dump_json()
        metadata.update(
            {
                "cayu:title": ["private-canary"],
                "cayu:summary": "x" * 4097,
                "cayu:summary_tone": True,
            }
        )
        row = client.get("/api/sessions", params={"include": "overview"}).json()["overviews"][0]
        assert row["title"] is row["summary"] is row["summary_tone"] is None

        class Hostile:
            def __repr__(self):
                return "overview-diagnostic-canary"

            __str__ = __repr__

        # Session base serialization remains an existing, separate projection.
        # Exercise the newly owned overview projector with the hostile field
        # through its real HTTP entrance while keeping that sibling projection
        # safely redacted as it normally is for persisted JSON values.
        metadata["cayu:title"] = Hostile()
        with warnings.catch_warnings(record=True) as caught:
            result = client.get("/api/sessions", params={"include": "overview"})
        captured = capsys.readouterr()
        assert result.status_code == 200
        assert "overview-diagnostic-canary" not in (
            result.text + caplog.text + captured.out + captured.err + str(caught)
        )


def test_member_can_read_counts_and_id_overviews():
    from cryptography.hazmat.primitives.asymmetric import ec
    from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

    key = ec.generate_private_key(ec.SECP256R1())
    with TestClient(
        create_server(_app(), config=ServerConfig.protected(_auth(FakeJwks(key))))
    ) as client:
        headers = _as(_token(key, role="member"))
        assert client.get("/api/sessions/overview/counts", headers=headers).status_code == 200
        assert (
            client.get("/api/sessions", params={"include": "overview"}, headers=headers).status_code
            == 200
        )
        assert client.get("/api/sessions/overview/counts").status_code == 401


def test_delegated_parent_state_and_subagent_counts_share_cloud():
    from tests.core.test_cloud_reporting import _delegating_app

    from cayu.sessions.memory import InMemorySessionStore

    app = _delegating_app(InMemorySessionStore())
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "root", "root")
        cloud = client.portal.call(read_cloud_activity, app)
        response = client.get("/api/sessions", params={"include": "overview"})
        assert response.status_code == 200
        rows = {row["session_id"]: row for row in response.json()["overviews"]}
        assert len(rows) == 3
        assert {row["state"] for row in rows.values()} == {"waiting_approval"}
        assert sorted(row["uses"]["subagents"] for row in rows.values()) == ["0", "1", "1"]
        assert {run.session_id: run.status.value for run in cloud.runs} == {
            key: row["state"] for key, row in rows.items()
        }


def test_http_count_cancellation_remains_task_cancellation(monkeypatch):
    import asyncio

    import httpx

    async def scenario():
        app = _app()
        entered = asyncio.Event()

        async def blocked_read(query=None):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(app.session_store, "list_sessions", blocked_read)
        server = create_server(app, config=ServerConfig.local_development())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            task = asyncio.create_task(client.get("/api/sessions/overview/counts"))
            await asyncio.wait_for(entered.wait(), timeout=2)
            task.cancel()
            task.cancel()
            assert task.cancelling() == 2
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
        await app.aclose()

    asyncio.run(scenario())


def test_overview_browser_requires_private_checkpoint_authority():
    from tests.core.test_browser_control import identity

    from cayu.sessions._browser_control_checkpoint import (
        BrowserControlCheckpointMutation,
        browser_control_checkpoint_mutation_scope,
    )
    from cayu.sessions.checkpoints import (
        BROWSER_CONTROLS_CHECKPOINT_KEY,
        CHECKPOINT_SCHEMA_VERSION_KEY,
        CURRENT_CHECKPOINT_SCHEMA_VERSION,
    )
    from cayu.tools.browser_control import BrowserControlCheckpoint, BrowserControlRecord

    app = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "browser", "writer")
        session = client.portal.call(app.session_store.load, "browser")
        controls = BrowserControlCheckpoint(
            records=(
                BrowserControlRecord(
                    identity=identity().model_copy(
                        update={
                            "session_id": session.id,
                            "session_instance_id": session.instance_id,
                        }
                    )
                ),
            )
        )
        publication = {
            CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
            BROWSER_CONTROLS_CHECKPOINT_KEY: controls.model_dump(mode="json"),
        }
        client.portal.call(app.session_store.checkpoint, session.id, publication)

        def browser_flag():
            response = client.get("/api/sessions", params={"include": "overview"})
            assert response.status_code == 200
            return response.json()["overviews"][0]["uses"]["browser"]

        assert browser_flag() is False

        async def publish():
            with browser_control_checkpoint_mutation_scope(
                BrowserControlCheckpointMutation(session.id, None, controls)
            ):
                await app.session_store.checkpoint(
                    session.id,
                    {
                        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
                        BROWSER_CONTROLS_CHECKPOINT_KEY: controls.model_dump(mode="json"),
                    },
                )

        client.portal.call(publish)
        assert browser_flag() is True


def test_overview_artifacts_deduplicate_registered_store_and_refuse_partial_counts(
    tmp_path, monkeypatch
):
    from cayu.artifacts import LocalArtifactStore
    from cayu.artifacts.base import ArtifactListResult
    from cayu.environments import Environment, EnvironmentSpec

    app = _app()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    app.register_environment(Environment(EnvironmentSpec(name="first"), artifact_store=artifacts))
    app.register_environment(Environment(EnvironmentSpec(name="second"), artifact_store=artifacts))

    async def populate():
        for index in range(3):
            await artifacts.put_bytes(b"content", filename=f"{index}.txt", session_id="owner")
        await artifacts.put_bytes(b"other", filename="other.txt", session_id="other")

    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "owner", "writer")
        client.portal.call(populate)
        response = client.get("/api/sessions", params={"include": "overview"})
        assert response.status_code == 200
        assert response.json()["overviews"][0]["uses"]["artifacts"] == "3"

        async def incomplete(**kwargs):
            return ArtifactListResult(artifacts=(), total_count=None, truncated=True)

        monkeypatch.setattr(artifacts, "list", incomplete)
        response = client.get("/api/sessions", params={"include": "overview"})
        assert response.status_code == 503
        assert "overviews" not in response.json()


def test_overview_runtime_interruption_and_failure_states():
    from tests.core.test_cloud_reporting import _drain

    from cayu.sessions import InterruptSessionRequest
    from cayu.sessions.records import SessionStatus

    app = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "stop", "approver")
        client.portal.call(
            _drain, app.interrupt_session(InterruptSessionRequest(session_id="stop"))
        )
        client.portal.call(_start, app, "failed", "writer")
        # Durable store status is the terminal state owner, independent of a
        # provider-specific error shape; HTTP must project that stored outcome.
        client.portal.call(app.session_store.update_status, "failed", SessionStatus.FAILED)
        response = client.get("/api/sessions", params={"include": "overview"})
        assert response.status_code == 200
        assert {r["session_id"]: r["state"] for r in response.json()["overviews"]} == {
            "stop": "interrupted",
            "failed": "failed",
        }


def test_overview_recovery_state_after_real_resume_discovers_unfinished_work():
    from tests.core.test_cloud_reporting import _drain, _pending
    from tests.core.test_foreground_subagent_recovery import _Provider
    from tests.core.test_user_input import _CountingTool, _crashed_user_input_resume_events

    from cayu import AgentSpec, CayuApp
    from cayu.approvals import UserInputResponse
    from cayu.providers import ModelStreamEvent
    from cayu.tools.user_input import UserInputTool

    app = CayuApp(enable_logging=False)
    app.register_provider(
        _Provider(
            [
                [
                    ModelStreamEvent.tool_call(id="call_1", name="count", arguments={}),
                    ModelStreamEvent.tool_call(
                        id="call_2", name="ask_user", arguments={"question": "Which?"}
                    ),
                    ModelStreamEvent.completed(),
                ]
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="worker", model="test"), tools=[_CountingTool(), UserInputTool()]
    )
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "recover", "worker")

        async def discover():
            question = await _pending(app, "recover")
            await app.session_store.append_events(
                "recover",
                _crashed_user_input_resume_events(
                    await app.session_store.load_events("recover"),
                    session_id="recover",
                    tool_call_id="call_1",
                ),
            )
            await _drain(
                app.resolve_user_input(
                    UserInputResponse(
                        session_id="recover", input_id=question.input_id, answer="Answer"
                    )
                )
            )

        client.portal.call(discover)
        response = client.get("/api/sessions", params={"include": "overview"})
        assert response.status_code == 200
        assert response.json()["overviews"][0]["state"] == "needs_recovery"
        cloud = client.portal.call(read_cloud_activity, app)
        assert cloud.runs[0].status.value == "needs_recovery"


def test_overview_window_boundaries_batch_bound_and_configured_pricing():
    from datetime import timedelta
    from decimal import Decimal

    from cayu.budgets.pricing import default_price_book
    from cayu.server.config import DashboardConfig

    app = _app()
    config = ServerConfig.local_development(
        dashboard=DashboardConfig(runtime_config={"priceBook": default_price_book()})
    )
    with TestClient(create_server(app, config=config)) as client:
        assert client.get("/api/sessions/overview/counts").json()["total_count"] == "0"
        client.portal.call(_start, app, "priced", "writer")
        session = client.portal.call(app.session_store.load, "priced")
        response = client.get("/api/sessions", params={"include": "overview", "ids": "priced"})
        assert response.status_code == 200
        cost = response.json()["overviews"][0]["cost"]
        assert Decimal(cost["amount"]) > 0 and cost["complete"] is True
        counts = client.get(
            "/api/sessions/overview/counts",
            params={
                "since": session.created_at.isoformat(),
                "until": (session.created_at + timedelta(seconds=1)).isoformat(),
            },
        )
        assert counts.json()["total_count"] == "1"
        counts = client.get(
            "/api/sessions/overview/counts",
            params={
                "since": (session.created_at - timedelta(seconds=1)).isoformat(),
                "until": session.created_at.isoformat(),
            },
        )
        assert counts.json()["total_count"] == "0"
        assert (
            client.get(
                "/api/sessions/overview/counts", params={"since": "2026-01-01T00:00:00"}
            ).status_code
            == 422
        )
        assert client.get("/api/sessions", params=[("ids", "missing")] * 201).status_code == 422
        assert (
            client.get("/api/sessions", params=[("ids", "missing")] * 200).json()["sessions"] == []
        )


@pytest.mark.parametrize(
    "error,expected",
    [
        ("Provider returned 503: secret-canary", "Provider returned 503: [REDACTED_SECRET]"),
        (None, "This session failed."),
        ({"unsafe": "secret-canary"}, "This session failed."),
        (" ", "This session failed."),
        ("Provider failed\nTry again", "Provider failed Try again"),
        ("x" * 1020 + "secret-canary", "x" * 1020 + "[RE…"),
    ],
)
def test_failed_overview_explains_only_redacted_durable_error(error, expected):
    from cayu.events import Event, EventType
    from cayu.sessions.records import SessionStatus
    from cayu.vaults.redaction import SecretRedactor

    app = _app(secret_redactor=SecretRedactor(["secret-canary"]))
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "failure", "writer")
        client.portal.call(
            app.session_store.append_events,
            "failure",
            [
                Event(
                    session_id="failure",
                    type=EventType.SESSION_FAILED,
                    payload={"error": error, "error_type": "secret-canary"},
                )
            ],
        )
        client.portal.call(app.session_store.update_status, "failure", SessionStatus.FAILED)
        response = client.get("/api/sessions", params={"include": "overview"})
        assert response.status_code == 200, response.text
        row = response.json()["overviews"][0]
        assert row["state"] == "failed"
        assert row["reason"] == expected
        assert "secret-canary" not in response.text
        assert len(row["reason"]) <= 1024
        assert client.portal.call(app.session_store.summarize_outcome, "failure").reason == "failed"
