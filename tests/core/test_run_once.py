"""Tests for ``run_once``: reuse a finished named session instead of rerunning it."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    ResumeRequest,
    RunRequest,
    SessionStatus,
    run_to_completion,
)
from cayu.context import StructuredOutputSpec
from cayu.evals import ScriptedModelProvider, scripted_structured_output
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.sessions import (
    RunOnceMismatch,
    RunOnceRefused,
    RunOnceSessionNotFinished,
    run_once,
)
from cayu.storage import SQLiteSessionStore

_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
    "additionalProperties": False,
}


def _spec(schema: dict | None = None) -> StructuredOutputSpec:
    return StructuredOutputSpec(name="verdict", json_schema=schema or _SCHEMA)


def _request(**overrides) -> RunRequest:
    values = {
        "agent_name": "verifier",
        "session_id": "scan-7-F1-verify-1",
        "messages": [Message.text("user", "Verify finding F1.")],
        "structured_output": _spec(),
    }
    values.update(overrides)
    return RunRequest(**values)


def _app(provider: ModelProvider, *, store=None) -> CayuApp:
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="verifier", model="scripted-model"))
    app.register_agent(AgentSpec(name="fixer", model="scripted-model"))
    return app


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_run_once_reuses_a_finished_session_without_a_model_call(
    store_kind: str, tmp_path: Path
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite") if store_kind == "sqlite" else None
    provider = ScriptedModelProvider([scripted_structured_output({"verdict": "real"})])
    app = _app(provider, store=store)

    async def scenario():
        first = await run_once(app, _request())
        assert len(provider.requests) == 1
        # A restarted handler reruns its pipeline from the top.
        second = await run_once(app, _request())
        return first, second

    first, second = asyncio.run(scenario())

    assert len(provider.requests) == 1
    assert first.ok
    assert first.structured_output is not None
    assert first.structured_output.output == {"verdict": "real"}
    # The reused outcome is the one the first caller saw, event for event.
    for field in (
        "session_id",
        "status",
        "final_text",
        "error",
        "structured_output",
        "interaction_id",
        "provider_name",
        "model",
        "model_steps",
        "usage",
        "tool_calls",
        "interruption",
        "events",
    ):
        assert getattr(second, field) == getattr(first, field), field


def test_run_once_reuses_a_failed_session_as_failed() -> None:
    provider = ScriptedModelProvider([scripted_structured_output({"wrong": 1})])
    app = _app(provider)

    async def scenario():
        first = await run_once(app, _request(max_steps=1))
        second = await run_once(app, _request(max_steps=1))
        return first, second

    first, second = asyncio.run(scenario())

    assert len(provider.requests) == 1
    assert first.status is SessionStatus.FAILED
    assert second.status is SessionStatus.FAILED
    assert second.error == first.error
    assert second.error is not None


def test_run_once_runs_a_new_session_normally() -> None:
    provider = ScriptedModelProvider([scripted_structured_output({"verdict": "real"})])
    app = _app(provider)

    outcome = asyncio.run(run_once(app, _request(session_id="scan-7-F2-verify-1")))

    assert len(provider.requests) == 1
    assert outcome.ok
    assert outcome.session_id == "scan-7-F2-verify-1"
    assert outcome.structured_output is not None
    assert outcome.structured_output.output == {"verdict": "real"}


def test_run_once_requires_a_session_id() -> None:
    provider = ScriptedModelProvider([])
    app = _app(provider)

    with pytest.raises(ValueError, match="needs request.session_id"):
        asyncio.run(run_once(app, _request(session_id=None)))
    assert provider.requests == []


@pytest.mark.parametrize(
    ("overrides", "field", "message"),
    [
        ({"agent_name": "fixer"}, "agent_name", "was run by agent 'verifier', not 'fixer'"),
        (
            {"structured_output": _spec({"type": "object", "additionalProperties": True})},
            "structured_output",
            "different structured output",
        ),
        ({"structured_output": None}, "structured_output", "different structured output"),
    ],
)
def test_run_once_refuses_a_session_started_for_different_work(
    overrides: dict, field: str, message: str
) -> None:
    provider = ScriptedModelProvider([scripted_structured_output({"verdict": "real"})])
    app = _app(provider)

    async def scenario():
        await run_once(app, _request())
        events_before = await app.session_store.load_events("scan-7-F1-verify-1")
        with pytest.raises(RunOnceMismatch, match=message) as refused:
            await run_once(app, _request(**overrides))
        events_after = await app.session_store.load_events("scan-7-F1-verify-1")
        return refused.value, events_before, events_after

    refused, events_before, events_after = asyncio.run(scenario())

    assert isinstance(refused, RunOnceRefused)
    assert refused.field == field
    assert refused.session_id == "scan-7-F1-verify-1"
    assert len(provider.requests) == 1
    assert events_after == events_before


class _BlockingProvider(ModelProvider):
    """Holds the first model call open until the test releases it."""

    name = "scripted"

    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def test_run_once_reports_a_running_session_and_leaves_it_alone() -> None:
    provider = _BlockingProvider()
    app = _app(provider)
    request = _request(structured_output=None)

    async def scenario():
        running = asyncio.create_task(run_to_completion(app, request))
        await provider.started.wait()
        session_before = await app.session_store.load(request.session_id)
        events_before = await app.session_store.load_events(request.session_id)
        with pytest.raises(RunOnceSessionNotFinished) as refused:
            await run_once(app, request)
        session_after = await app.session_store.load(request.session_id)
        events_after = await app.session_store.load_events(request.session_id)
        provider.release.set()
        finished = await running
        return refused.value, session_before, session_after, events_before, events_after, finished

    refused, before, after, events_before, events_after, finished = asyncio.run(scenario())

    assert refused.status is SessionStatus.RUNNING
    assert refused.session_id == request.session_id
    assert "not finished" in str(refused)
    assert provider.calls == 1
    assert after == before
    assert events_after == events_before
    # The original run was not disturbed.
    assert finished.ok
    assert finished.final_text == "done"


def test_run_once_answers_from_a_session_another_caller_created_first(monkeypatch) -> None:
    provider = ScriptedModelProvider([scripted_structured_output({"verdict": "real"})])
    app = _app(provider)

    async def scenario():
        first = await run_once(app, _request())
        load = app.session_store.load
        calls = 0

        async def miss_once(session_id: str):
            # The first lookup races a creator that has not committed yet.
            nonlocal calls
            calls += 1
            return None if calls == 1 else await load(session_id)

        monkeypatch.setattr(app.session_store, "load", miss_once)
        second = await run_once(app, _request())
        return first, second

    first, second = asyncio.run(scenario())

    assert len(provider.requests) == 1
    assert second.ok
    assert second.events == first.events


def test_run_once_refuses_a_settled_status_without_its_terminal_event(monkeypatch) -> None:
    provider = ScriptedModelProvider([scripted_structured_output({"verdict": "real"})])
    app = _app(provider)

    async def scenario():
        await run_once(app, _request())
        query_events = app.session_store.query_events

        async def before_terminal_publication(query):
            # The worker died after writing the status but before publishing
            # the terminal event.
            records = await query_events(query)
            return [r for r in records if r.event.type != "session.completed"]

        monkeypatch.setattr(app.session_store, "query_events", before_terminal_publication)
        with pytest.raises(RunOnceSessionNotFinished, match="terminal event") as refused:
            await run_once(app, _request())
        return refused.value

    refused = asyncio.run(scenario())

    assert refused.status is SessionStatus.COMPLETED
    assert len(provider.requests) == 1


def _text_turn(text: str) -> list[ModelStreamEvent]:
    return [ModelStreamEvent.text_delta(text), ModelStreamEvent.completed()]


def test_run_once_returns_the_latest_invocation_of_a_resumed_session() -> None:
    provider = ScriptedModelProvider([_text_turn("one"), _text_turn("two")])
    app = _app(provider)
    request = _request(structured_output=None)

    async def scenario():
        await run_once(app, request)
        resumed = await run_to_completion(
            app,
            ResumeRequest(session_id=request.session_id, messages=[Message.text("user", "Again.")]),
        )
        again = await run_once(app, request)
        return resumed, again

    resumed, again = asyncio.run(scenario())

    assert len(provider.requests) == 2
    assert again.final_text == "two"
    assert again.model_steps == 1
    for field in ("status", "final_text", "interaction_id", "usage", "tool_calls", "events"):
        assert getattr(again, field) == getattr(resumed, field), field


def test_run_once_ignores_an_earlier_invocation_terminal_event(monkeypatch) -> None:
    provider = ScriptedModelProvider([_text_turn("one"), _text_turn("two")])
    app = _app(provider)
    request = _request(structured_output=None)

    async def scenario():
        await run_once(app, request)
        await run_to_completion(
            app,
            ResumeRequest(session_id=request.session_id, messages=[Message.text("user", "Again.")]),
        )
        query_events = app.session_store.query_events

        async def resume_lost_before_terminal_publication(query):
            # The resume failed before publishing session.resumed and its own
            # terminal event, so only the first run's session.completed remains.
            records = await query_events(query)
            completed = [r.sequence for r in records if r.event.type == "session.completed"]
            return [
                r
                for r in records
                if r.sequence != max(completed, default=0) and r.event.type != "session.resumed"
            ]

        monkeypatch.setattr(
            app.session_store, "query_events", resume_lost_before_terminal_publication
        )
        with pytest.raises(RunOnceSessionNotFinished, match="terminal event") as refused:
            await run_once(app, request)
        return refused.value

    refused = asyncio.run(scenario())

    assert refused.status is SessionStatus.COMPLETED
    assert len(provider.requests) == 2
