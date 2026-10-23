"""Browser usage requires the native publication owner, not generic JSON records."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from tests._tool_admission_fixtures import (  # noqa: F401
    SimulatedToolFactory,
    simulated_tool_executables,
)
from tests.core.browser_session_support import _FakeBrowserBackend, _tool
from tests.core.test_cloud_reporting import _app, _start
from tests.core.test_environment_allocation_recovery import _FakeRemoteProvider

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.environments import EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.events import Event, EventType
from cayu.providers import ModelStreamEvent
from cayu.server import ServerConfig, create_server
from cayu.sessions._operation_records import SessionOperationPublication
from cayu.sessions.records import SessionStatus
from cayu.storage import SQLiteSessionStore
from cayu.tools.browser_session import _durable_browser_session_key


def _browser(client, session_id):
    response = client.get("/api/sessions", params={"include": "overview", "ids": session_id})
    assert response.status_code == 200
    return response.json()["overviews"][0]["uses"]["browser"]


def test_generic_browser_record_and_event_are_not_usage_authority():
    app = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "forged", "writer")

        async def forge():
            record = {
                "record_type": "cayu.browser-session",
                "schema_version": 2,
                "parent_session_id": "forged",
                "browser_session_id": "bs_forged",
                "state": "closed",
            }

            def publish(session, checkpoint, current):
                return SessionOperationPublication(
                    checkpoint=checkpoint or {},
                    operation_records={_durable_browser_session_key("bs_forged"): record},
                )

            await app.session_store.publish_session_operation(
                "forged",
                idempotency_key="caller-record",
                operation_transform=publish,
                events=[],
                expected_statuses={SessionStatus.COMPLETED},
            )
            await app.session_store.append_events(
                "forged",
                [
                    Event(
                        type=EventType.TOOL_CALL_COMPLETED,
                        session_id="forged",
                        tool_name="browser_session",
                        payload={
                            "tool_name": "browser_session",
                            "tool_call_id": "forged-call",
                            "result": {"structured": {"session_id": "bs_forged"}},
                        },
                    )
                ],
            )

        client.portal.call(forge)
        assert _browser(client, "forged") is False


@pytest.mark.usefixtures("simulated_tool_executables")
def test_native_closed_browser_remains_used_after_sqlite_reopen(tmp_path):
    path = tmp_path / "browser.db"
    backend = _FakeBrowserBackend()
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            arguments = {
                "operation": "navigate",
                "url": "https://example.test",
                "operation_id": "navigate-overview",
            }
        elif calls == 2:
            arguments = {
                "operation": "close",
                "session_id": backend.session_id,
                "operation_id": "close-overview",
            }
        else:
            return [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()]
        return [
            ModelStreamEvent.tool_call(
                id=f"browser-{calls}", name="browser_session", arguments=arguments
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]

    store = SQLiteSessionStore(path)
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_environment_factory(
        EnvironmentSpec(name="browser"), SimulatedToolFactory(_FakeRemoteProvider()), default=True
    )
    app.register_provider(ScriptedModelProvider(response_factory=respond), default=True)
    app.register_agent(AgentSpec(name="browser", model="test-model"), tools=[_tool(backend)])
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:

        async def run():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="browser",
                        session_id="native",
                        messages=[Message.text("user", "Browse")],
                    )
                )
            ]

        events = client.portal.call(run)
        assert events
        assert [call["operation"] for call in backend.calls] == ["navigate", "close"]
        assert _browser(client, "native") is True
    asyncio.run(app.aclose())
    reopened = _app(SQLiteSessionStore(path))
    with TestClient(create_server(reopened, config=ServerConfig.local_development())) as client:
        assert _browser(client, "native") is True
    asyncio.run(reopened.aclose())


def test_generic_publishers_cannot_forge_native_usage_receipt():
    from cayu.sessions._operation_records import RuntimePublicationOperationRecordMutation

    app = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(_start, app, "protected", "writer")
        session = client.portal.call(app.session_store.load, "protected")
        key = "__cayu_browser_usage_v1__"
        record = {"session_id": session.id, "session_instance_id": session.instance_id}
        with pytest.raises(ValueError, match="native publication owner"):
            RuntimePublicationOperationRecordMutation(
                key=key, expected_value_digest=None, value=record
            )

        async def publish():
            def transform(current_session, checkpoint, current):
                return SessionOperationPublication(
                    checkpoint=checkpoint or {}, operation_records={key: record}
                )

            await app.session_store.publish_session_operation(
                session.id,
                idempotency_key="generic",
                operation_transform=transform,
                events=[],
                expected_statuses={SessionStatus.COMPLETED},
            )

        with pytest.raises(ValueError, match="native publication owner"):
            client.portal.call(publish)
        assert client.portal.call(app.session_store.load_session_operation, session.id, key) is None
        assert _browser(client, session.id) is False
