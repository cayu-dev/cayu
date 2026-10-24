"""Public trace lifecycle and retained-history page invariants."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from tests.core.test_builtin_tools import RecordingRunner
from tests.core.test_tool_completion import FinalTool, app_for, call
from tests.core.test_tool_round_publication_failure_matrix import _TwoCallProvider

from cayu import CayuApp, Message, RunRequest
from cayu.environments import Environment, EnvironmentSpec
from cayu.events import Event, EventType
from cayu.providers import ModelStreamEvent
from cayu.runners import ExecCommand
from cayu.server import ServerConfig, create_server
from cayu.sessions import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.storage import SQLiteSessionStore
from cayu.tools import ToolResult


def _store(tmp_path, backend):
    return (
        InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(tmp_path / "trace.db")
    )


def _request():
    return RunRequest(
        session_id="trace", agent_name="support", messages=[Message.text("user", "run")]
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("outcome", ["exception", "cancel"])
def test_trace_real_runner_failure_and_interruption(tmp_path, backend, outcome):
    entered = asyncio.Event()

    class Runner(RecordingRunner):
        async def exec(self, command, **kwargs):
            if outcome == "exception":
                raise OSError("launch failed")
            entered.set()
            await asyncio.Event().wait()

    class CommandTool(FinalTool):
        async def run(self, ctx, args):
            result = await ctx.runner.exec(ExecCommand(argv=["echo", "hello"]))
            return ToolResult(content=result.stdout)

    app = app_for(
        _store(tmp_path, backend),
        _TwoCallProvider(
            [call(), [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()]]
        ),
        CommandTool(),
    )
    app.register_environment(
        Environment(EnvironmentSpec(name="runner"), runner=Runner()), default=True
    )

    async def scenario():
        async def run():
            return [event async for event in app.run(_request())]

        task = asyncio.create_task(run())
        if outcome == "cancel":
            await asyncio.wait_for(entered.wait(), 10)
            task.cancel()
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled() and task.cancelling() == 1
            await app.drain_background_interruptions()
        else:
            await task

    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(scenario)
        response = client.get("/api/sessions/trace/trace", params={"limit": 1000})
        assert response.status_code == 200, response.text
        spans = response.json()["spans"]
        expected = "failed" if outcome == "exception" else "interrupted"
        kinds = {"exec"} if outcome == "exception" else {"exec", "tool", "round"}
        relevant = [span for span in spans if span["kind"] in kinds]
        assert {span["kind"] for span in relevant} == kinds
        assert all(span["status"] == expected and span["ended_at"] for span in relevant), relevant


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_trace_retained_interaction_anchor_is_independent_of_page(tmp_path, backend):
    app = CayuApp(session_store=_store(tmp_path, backend), enable_logging=False)

    async def seed():
        await app.session_store.create(
            _request(), identity=SessionIdentity(provider_name="fake", model="fake")
        )
        for kind in [
            EventType.KNOWLEDGE_LOOKUP,
            EventType.KNOWLEDGE_LOOKUP,
            EventType.INTERACTION_COMPLETED,
        ]:
            await app.session_store.append_event(
                "trace", Event(type=kind, session_id="trace", interaction_id="retained-turn")
            )

    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(seed)
        whole = client.get("/api/sessions/trace/trace", params={"limit": 1000}).json()
        expected = next(span for span in whole["spans"] if span["kind"] == "interaction")
        for after in range(3):
            page = client.get(
                "/api/sessions/trace/trace",
                params={"limit": 1, "after_sequence": after, "through_sequence": 3},
            ).json()
            actual = next(span for span in page["spans"] if span["kind"] == "interaction")
            assert {k: v for k, v in actual.items() if k != "event_ids"} == {
                k: v for k, v in expected.items() if k != "event_ids"
            }
            assert all(
                span["parent_id"] == expected["id"]
                for span in page["spans"]
                if span["kind"] == "knowledge"
            )
