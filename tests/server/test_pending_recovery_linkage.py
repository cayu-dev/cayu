"""Exact public recovery references reconstructed from bounded native evidence."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient
from tests.core._event_projection_support import private_events_for_public_events
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    _reopen_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_user_input import (
    _CountingTool,
    _crashed_user_input_resume_events,
)

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.approvals.user_input import UserInputResponse
from cayu.events import Event, EventType
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.server import ServerConfig, create_server
from cayu.sessions import InMemorySessionStore, PendingActionQuery
from cayu.sessions.event_queries import EventQuery
from cayu.storage import SQLiteSessionStore
from cayu.tools.user_input import UserInputTool

__all__ = ["conformance_postgres_dsn", "session_store_case"]

_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="tests:recovery-linkage", behavior_version="1", implementation_version="1"
)


class _RecoveryProvider(ModelProvider):
    name = "fake"

    @property
    def execution_profile_identity(self):
        return _IDENTITY

    async def stream(self, request):
        if any(message.role == "tool" for message in request.messages):
            yield ModelStreamEvent.text_delta("reconciled")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        yield ModelStreamEvent.tool_call(id="sibling", name="count", arguments={})
        yield ModelStreamEvent.tool_call(
            id="question", name="ask_user", arguments={"question": "Which environment?"}
        )
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


class _StableCountingTool(_CountingTool):
    spec = _CountingTool.spec.model_copy(update={"execution_profile_identity": _IDENTITY})


def _app(store):
    tool = _StableCountingTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(_RecoveryProvider(), default=True)
    app.register_agent(AgentSpec(name="asker", model="fake-model"), tools=[tool, UserInputTool()])
    return app, tool


async def _crash(app, *, nested_field=None, nested_value=None):
    paused = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="asker",
                session_id="crashed",
                messages=[Message.text("user", "go")],
            )
        )
    ]
    awaiting = next(
        event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
    )
    crash_events = _crashed_user_input_resume_events(
        await private_events_for_public_events(app.session_store, paused),
        session_id="crashed",
        tool_call_id="sibling",
    )
    if nested_field is not None:
        start = next(event for event in crash_events if event.type is EventType.TOOL_CALL_STARTED)
        start.payload["result"] = {"structured": {nested_field: nested_value}}
        # A caller-supplied lookalike marker must be replaced by native proof.
        start.payload["__cayu_pending_source_linkage__"] = {
            nested_field: start.payload[nested_field]
        }
    await app.session_store.append_events("crashed", crash_events)
    stopped = [
        event
        async for event in app.resolve_user_input(
            UserInputResponse(
                session_id="crashed",
                input_id=awaiting.payload["input_id"],
                answer="staging",
            )
        )
    ]
    assert any(event.payload.get("manual_recovery_required") for event in stopped)
    [action] = (
        await app.session_store.query_pending_actions(PendingActionQuery(session_id="crashed"))
    ).actions
    return action.attention_id


@pytest.mark.store_semantics
def test_public_recovery_reconstructs_linkage_after_store_reopen(session_store_case):
    async def run():
        store = await _open_store(session_store_case)
        try:
            app, tool = _app(store)
            attention_id = await _crash(app)
            assert tool.calls == 0
            store = await _reopen_store(session_store_case, store)
            app, tool = _app(store)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=create_server(app, config=ServerConfig.local_development())
                ),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/pending-actions")
                assert response.status_code == 200, response.text
                [action] = response.json()["actions"]
                assert action["attention_id"] == attention_id
                assert action["tool_call_id"] is not None
                assert "source_linkage" not in action
                recovered = await client.post(
                    "/api/user-input/recover",
                    json={
                        "session_id": action["session"]["id"],
                        "input_id": action["input_id"],
                        "tool_call_id": action["tool_call_id"],
                        "answer": "staging",
                        "outcome": "completed",
                        "message": "Verified externally after reopen.",
                    },
                )
                assert recovered.status_code == 200, recovered.text
                assert "session.completed" in recovered.text
                assert tool.calls == 0
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("extra", ["other_round", "duplicate_start"])
def test_public_recovery_never_guesses_between_start_records(store_kind, extra, tmp_path):
    store = (
        InMemorySessionStore()
        if store_kind == "memory"
        else SQLiteSessionStore(tmp_path / "evidence.db")
    )
    app, tool = _app(store)
    attention_id = asyncio.run(_crash(app))

    async def add_evidence():
        records = await store.query_events(EventQuery(session_id="crashed"))
        start = next(
            record.event
            for record in records
            if record.event.type is EventType.TOOL_CALL_STARTED
            and record.event.payload.get("tool_call_id") == "sibling"
        )
        payload = dict(start.payload)
        if extra == "other_round":
            payload.update(
                model_step_id="mstep_" + "a" * 32,
                model_attempt_id="matt_" + "b" * 32,
                tool_round_id="tround_" + "c" * 32,
            )
        await store.append_events(
            "crashed",
            [
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="crashed",
                    payload=payload,
                    tool_name=start.tool_name,
                )
            ],
        )

    asyncio.run(add_evidence())
    try:
        with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
            response = client.get("/api/pending-actions")
            assert response.status_code == 200, response.text
            page = response.json()
            if extra == "other_round":
                [action] = page["actions"]
                assert action["attention_id"] == attention_id
                assert action["tool_call_id"] is not None
            elif page["actions"]:
                [action] = page["actions"]
                assert action["tool_call_id"] is None
            else:
                assert page["issues"]
            assert tool.calls == 0
    finally:
        asyncio.run(_close_store(store))


def test_private_recovery_linkage_copy_dump_and_retarget_boundaries(capsys, caplog):
    import warnings

    from cayu.server.routes import _serialize_pending_action
    from cayu.sessions.pending_action_contracts import PendingActionRecord

    app, _ = _app(InMemorySessionStore())
    asyncio.run(_crash(app))
    [action] = asyncio.run(
        app.session_store.query_pending_actions(PendingActionQuery(session_id="crashed"))
    ).actions
    assert _serialize_pending_action(app, action.model_copy(deep=True))["tool_call_id"] is not None
    dumped = action.model_dump(mode="json")
    assert "source_linkage" not in dumped
    reconstructed = PendingActionRecord.model_validate(dumped)
    assert _serialize_pending_action(app, reconstructed)["tool_call_id"] is None
    with pytest.raises(TypeError, match="private source references"):
        PendingActionRecord.model_validate(
            {**dumped, "source_linkage": {"tool_call_id": {"sequence": 1, "value": "sibling"}}}
        )
    retargeted = action.model_copy(
        update={
            "session": action.session.model_copy(update={"instance_id": "replacement-instance"})
        },
        deep=True,
    )
    with pytest.raises(RuntimeError, match="disagrees with its durable event authority"):
        _serialize_pending_action(app, retargeted)

    class UnsafeDiagnostic:
        def __repr__(self):
            raise AssertionError("rejected-private-linkage-secret")

        __str__ = __repr__

        def __deepcopy__(self, memo):
            raise AssertionError("rejected-private-linkage-secret")

    reference = action.source_linkage["tool_call_id"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unsafe_reference = reference.model_copy(update={"value": UnsafeDiagnostic()})
        with pytest.raises(TypeError, match="invalid private source reference"):
            PendingActionRecord.model_validate(
                {
                    **dumped,
                    "source_linkage": {"tool_call_id": unsafe_reference},
                }
            )
        for changes in [
            {"sequence": True},
            {"value": UnsafeDiagnostic()},
            {"session_instance_id": UnsafeDiagnostic()},
        ]:
            changed = action.model_copy(
                update={
                    "source_linkage": {
                        **action.source_linkage,
                        "tool_call_id": reference.model_copy(update=changes),
                    }
                }
            )
            with pytest.raises(RuntimeError, match="disagrees with its durable event authority"):
                _serialize_pending_action(app, changed)
    assert not caught
    output = capsys.readouterr()
    assert "rejected-private-linkage-secret" not in output.out + output.err + caplog.text


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("field", ["tool_call_id", "tool_round_id"])
@pytest.mark.parametrize("invalid", ["conflicting", True, ""])
def test_public_recovery_preserves_original_nested_authority(store_kind, field, invalid, tmp_path):
    async def run():
        path = tmp_path / "nested.db"
        store = InMemorySessionStore() if store_kind == "memory" else SQLiteSessionStore(path)
        try:
            app, tool = _app(store)
            await _crash(app, nested_field=field, nested_value=invalid)
            if store_kind == "sqlite":
                await store.close()
                store = SQLiteSessionStore(path)
                app, tool = _app(store)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=create_server(app, config=ServerConfig.local_development())
                ),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/pending-actions")
                assert response.status_code == 200, response.text
                [action] = response.json()["actions"]
                # The original start cannot authorize this field, even though
                # its matching top-level identity survives compact projection.
                if field == "tool_call_id":
                    assert action[field] is None
                    assert action["round_id"] is not None
                    records = await store.query_events(EventQuery(session_id="crashed"))
                    start = next(
                        record
                        for record in records
                        if record.event.type is EventType.TOOL_CALL_STARTED
                        and record.event.payload.get("tool_call_id") == "sibling"
                    )
                    rejected = await client.post(
                        "/api/user-input/recover",
                        json={
                            "session_id": action["session"]["id"],
                            "input_id": action["input_id"],
                            "tool_call_id": f"cayu_event_{start.sequence}:tool_call_id",
                            "answer": "staging",
                            "outcome": "completed",
                            "message": "Must not dispatch.",
                        },
                    )
                    assert rejected.status_code == 409, rejected.text
                    unchanged = await client.get("/api/pending-actions")
                    assert unchanged.json()["actions"][0]["attention_id"] == action["attention_id"]
                else:
                    # Round authority is independently available on the action
                    # source; invalid nested round evidence must not erase the
                    # valid neighboring call reference from the start.
                    assert action["tool_call_id"] is not None
                    assert action["round_id"] is not None
                    assert action["round_id"] != action["tool_call_id"].replace(
                        ":tool_call_id", ":tool_round_id"
                    )
                    recovered = await client.post(
                        "/api/user-input/recover",
                        json={
                            "session_id": action["session"]["id"],
                            "input_id": action["input_id"],
                            "tool_call_id": action["tool_call_id"],
                            "answer": "staging",
                            "outcome": "completed",
                            "message": "Verified externally.",
                        },
                    )
                    assert recovered.status_code == 200, recovered.text
                assert tool.calls == 0
        finally:
            await _close_store(store)

    asyncio.run(run())
