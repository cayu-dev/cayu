"""`cayu serve --dev` continues sessions started under the demonstration provider."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu import AgentSpec, CayuApp, Message, ResumeRequest, RunRequest
from cayu.budgets import BudgetLimit, default_price_book
from cayu.cli import main
from cayu.cli.serve import register_demo_provider
from cayu.context import RecentTurnsContextPolicy
from cayu.events import EventType
from cayu.messages import TextPart, ToolCallPart, ToolResultPart
from cayu.providers import (
    DEMO_MODEL,
    DEMO_PROVIDER_NAME,
    DemoModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.sessions import ForkSessionRequest, ModelTarget
from cayu.sessions.records import SessionStatus

_QUESTION = {"question": "Which week suits the auditors?", "options": ["17 Nov", "24 Nov"]}

_FACTORY = """from cayu import AgentSpec, CayuApp
from cayu.evals import ScriptedModelProvider
from cayu.providers import ProviderNotConfigured
from cayu.storage import SQLiteSessionStore
from cayu.tools.user_input import UserInputTool


class Unselected(ScriptedModelProvider):
    def __init__(self):
        super().__init__([], name="unconfigured")

    def preflight_model_target(self, *, model):
        raise ProviderNotConfigured("No model provider is set.")


def build_app():
    app = CayuApp(session_store=SQLiteSessionStore("runtime.db"), enable_logging=False)
    app.register_provider(Unselected(), default=True)
    app.register_agent(AgentSpec(name="asker", model="unset"), tools=[UserInputTool()])
    return app
"""


class _Other(DemoModelProvider):
    name = "other"


def _request(*messages: Message, tools: tuple[str, ...] = ("ask_user",)) -> ModelRequest:
    return ModelRequest(
        model=DEMO_MODEL,
        messages=list(messages),
        tools=[{"name": name, "description": name, "input_schema": {}} for name in tools],
    )


def _events(provider: DemoModelProvider, request: ModelRequest) -> list[ModelStreamEvent]:
    async def collect() -> list[ModelStreamEvent]:
        return [event async for event in provider.stream(request)]

    return asyncio.run(collect())


def test_the_script_plays_by_turn_then_canned_replies_are_marked() -> None:
    provider = DemoModelProvider(
        ["Three things are open.", ModelStreamEvent.tool_call(name="ask_user", arguments=_QUESTION)]
    )
    first = _events(provider, _request(Message.text("user", "What is open?")))
    assert first[0].delta == "Three things are open."

    replied = Message.text("assistant", "Three things are open.")
    second = _events(provider, _request(Message.text("user", "What is open?"), replied))
    assert second[0].payload == {"name": "ask_user", "arguments": _QUESTION, "id": "demo-call-1"}
    assert second[-1].payload["finish_reason"] == "tool_calls"

    called = Message(
        role="assistant",
        content=[ToolCallPart(tool_call_id="demo-call-1", tool_name="ask_user", arguments={})],
    )
    answered = Message(
        role="tool",
        content=[
            ToolResultPart(tool_call_id="demo-call-1", tool_name="ask_user", content="17 Nov")
        ],
    )
    third = _events(provider, _request(Message.text("user", "?"), replied, called, answered))
    assert third[0].delta == "Demonstration reply, no model was called. Noted: 17 Nov"

    plain = _events(DemoModelProvider(), _request(Message.text("user", "Hello there")))
    assert plain[0].delta.startswith("Demonstration reply, no model was called. You wrote: ")
    assert "Hello there" in plain[0].delta


def test_a_scripted_tool_the_agent_isnt_offered_names_the_tool() -> None:
    provider = DemoModelProvider([ModelStreamEvent.tool_call(name="ask_user", arguments={})])

    with pytest.raises(ModelProviderError, match="calls ask_user, which this agent isn't offered"):
        _events(provider, _request(Message.text("user", "hi"), tools=()))


def test_a_script_holds_only_replies_and_tool_calls() -> None:
    with pytest.raises(TypeError, match="reply text or ModelStreamEvent.tool_call"):
        DemoModelProvider([ModelStreamEvent.text_delta("not a turn")])


def _demo_app(provider: DemoModelProvider, **agent: Any) -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_provider(provider)
    app.register_agent(AgentSpec(name="agent", model=DEMO_MODEL), **agent)
    return app


def _target() -> ModelTarget:
    return ModelTarget(provider_name=DEMO_PROVIDER_NAME, model=DEMO_MODEL)


async def _replies(app: CayuApp, session_id: str) -> list[str]:
    return [
        part.text
        for message in await app.session_store.load_transcript(session_id)
        if message.role == "assistant"
        for part in message.content
        if isinstance(part, TextPart)
    ]


def test_a_cost_budget_admits_the_demo_target_at_zero() -> None:
    async def scenario():
        app = _demo_app(DemoModelProvider(["Ready."]))
        async with app:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="budgeted",
                        target=_target(),
                        messages=[Message.text("user", "Start.")],
                        budget_limits=[BudgetLimit.usd(1)],
                    )
                )
            ]
            cost = await app.get_session_cost("budgeted", default_price_book())
            replies = await _replies(app, "budgeted")
        return events, replies, cost

    events, replies, cost = asyncio.run(scenario())
    assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
    assert replies == ["Ready."]
    assert cost.total_cost == 0
    assert cost.model_steps == 1 and cost.unpriced_model_steps == 0


def test_the_script_keeps_its_place_when_context_drops_earlier_turns() -> None:
    async def scenario():
        app = _demo_app(
            DemoModelProvider(["first", "second"]),
            context_policy=RecentTurnsContextPolicy(max_user_turns=1),
        )
        async with app:
            async for _event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="windowed",
                    target=_target(),
                    messages=[Message.text("user", "One.")],
                )
            ):
                pass
            for text in ("Two.", "Three."):
                resumed = ResumeRequest(
                    session_id="windowed", messages=[Message.text("user", text)]
                )
                async for _event in app.resume(resumed):
                    pass
            replies = await _replies(app, "windowed")
        return replies

    first, second, third = asyncio.run(scenario())
    assert (first, second) == ("first", "second")
    assert third.startswith("Demonstration reply, no model was called. You wrote: ")


def test_forks_of_a_scripted_session_each_keep_their_own_place() -> None:
    async def scenario():
        app = _demo_app(DemoModelProvider(["first", "second", "third"]))
        async with app:
            async for _event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="source",
                    target=_target(),
                    messages=[Message.text("user", "One.")],
                )
            ):
                pass
            for fork in ("fork-a", "fork-b"):
                request = ForkSessionRequest(source_session_id="source", session_id=fork)
                async for _event in app.fork_session(request):
                    pass
            for fork, text in (("fork-a", "Two, A."), ("fork-b", "Two, B.")):
                resumed = ResumeRequest(session_id=fork, messages=[Message.text("user", text)])
                async for _event in app.resume(resumed):
                    pass
            return [await _replies(app, fork) for fork in ("fork-a", "fork-b")]

    # Forks share a cache-affinity key; each continues the script from the fork point,
    # and one fork's turn doesn't move the other's on.
    assert asyncio.run(scenario()) == [["first", "second"], ["first", "second"]]


def test_a_request_session_key_is_never_serialized() -> None:
    request = _request(Message.text("user", "One."))
    keyed = request.model_copy(update={"session_key": "cayu-session"})
    assert "session_key" not in keyed.model_dump(mode="json")
    assert keyed.model_dump(mode="json") == request.model_dump(mode="json")
    with pytest.raises(ValueError, match="session_key must be"):
        type(request)(model="demo", messages=[], session_key="not a key")


def test_a_retried_step_gets_the_same_turn_and_a_new_one_moves_on() -> None:
    provider = DemoModelProvider(["first", "second", "third"])
    opening = _request(Message.text("user", "One.")).model_copy(
        update={"session_key": "cayu-session", "step_key": "cayu-step-1"}
    )
    assert _events(provider, opening)[0].delta == "first"
    # A retry of the same step plays the same turn again.
    assert _events(provider, opening)[0].delta == "first"
    # The policy dropped the first turn: the count says turn 0, the session is past it.
    windowed = opening.model_copy(
        update={"messages": [Message.text("user", "Two.")], "step_key": "cayu-step-2"}
    )
    assert _events(provider, windowed)[0].delta == "second"
    # The next step moves on even when its messages are identical to the last one's.
    repeated = windowed.model_copy(update={"step_key": "cayu-step-3"})
    assert _events(provider, repeated)[0].delta == "third"


def test_repeating_the_same_message_under_a_window_still_advances_the_script() -> None:
    async def scenario():
        app = _demo_app(
            DemoModelProvider(["first", "second"]),
            context_policy=RecentTurnsContextPolicy(max_user_turns=1),
        )
        async with app:
            async for _event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="repeat",
                    target=_target(),
                    messages=[Message.text("user", "Continue")],
                )
            ):
                pass
            for _ in range(2):
                resumed = ResumeRequest(
                    session_id="repeat", messages=[Message.text("user", "Continue")]
                )
                async for _event in app.resume(resumed):
                    pass
            return await _replies(app, "repeat")

    first, second, third = asyncio.run(scenario())
    assert (first, second) == ("first", "second")
    assert third.startswith("Demonstration reply, no model was called. You wrote: ")


def test_a_request_step_key_is_never_serialized() -> None:
    request = _request(Message.text("user", "One."))
    keyed = request.model_copy(update={"step_key": "cayu-step"})
    assert keyed.model_dump(mode="json") == request.model_dump(mode="json")
    with pytest.raises(ValueError, match="step_key must be"):
        type(request)(model="demo", messages=[], step_key="not a key")


def test_serve_dev_registers_the_demo_provider_once() -> None:
    app = CayuApp(enable_logging=False)
    assert register_demo_provider(app) is True
    assert register_demo_provider(app) is False
    assert app.list_providers() == (DEMO_PROVIDER_NAME,)
    # Never the default, even with no other provider; the next one registered is.
    assert app._provider_registry.default_name is None
    app.register_provider(_Other())
    assert app._provider_registry.default_name == "other"


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "demo_project:build_app"\n\n'
        '[tool.cayu.serve]\nauth = "demo_auth:operator"\n',
        encoding="utf-8",
    )
    (project / "demo_project.py").write_text(_FACTORY, encoding="utf-8")
    (project / "demo_auth.py").write_text(
        "from cayu.server import AuthContext\n\n\n"
        "async def operator(request):\n    return AuthContext(subject='operator')\n",
        encoding="utf-8",
    )
    for module in ("demo_project", "demo_auth"):
        sys.modules.pop(module, None)
    monkeypatch.chdir(project)
    monkeypatch.syspath_prepend(str(project))
    return project


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    launched: dict[str, Any] = {}
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda server, *, host, port: launched.update(server=server)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    return launched


def _ask(session_id: str) -> None:
    """What a seeding command does: one session that asks a question, then stops."""

    import demo_project

    async def run() -> None:
        app: CayuApp = demo_project.build_app()
        app.register_provider(
            DemoModelProvider([ModelStreamEvent.tool_call(name="ask_user", arguments=_QUESTION)])
        )
        async with app:
            async for _event in app.run(
                RunRequest(
                    agent_name="asker",
                    session_id=session_id,
                    target=ModelTarget(provider_name=DEMO_PROVIDER_NAME, model=DEMO_MODEL),
                    messages=[Message.text("user", "Set up the auditor's visit.")],
                )
            ):
                pass

    asyncio.run(run())


def _completed(client: TestClient, session_id: str) -> dict[str, Any]:
    for _attempt in range(500):
        session = client.get(f"/api/sessions/{session_id}").json()
        if session.get("status") == SessionStatus.COMPLETED.value:
            break
        time.sleep(0.01)
    return session


async def _transcript(session_id: str) -> str:
    from cayu.storage import SQLiteSessionStore

    store = SQLiteSessionStore("runtime.db")
    try:
        messages = await store.load_transcript(session_id)
    finally:
        await store.close()
    return "\n".join(
        part.text for message in messages for part in message.content if isinstance(part, TextPart)
    )


def test_serve_dev_answers_and_continues_a_demo_session(
    project: Path, launched: dict[str, Any]
) -> None:
    _ask("audit")

    assert main(["serve", "--dev"]) == 0
    with TestClient(launched["server"]) as client:
        [item] = client.get("/api/inbox").json()["items"]
        assert item["kind"] == "question"
        settled = client.post(
            f"/api/inbox/{item['id']}/settle", json={"decision": "answer", "answer": "17 Nov"}
        )
        assert settled.status_code == 200, settled.text
        session = _completed(client, "audit")
    assert session["status"] == SessionStatus.COMPLETED.value, session
    transcript = asyncio.run(_transcript("audit"))
    assert "Demonstration reply, no model was called. Noted: 17 Nov" in transcript


def test_a_protected_server_without_the_demo_provider_says_so(
    project: Path, launched: dict[str, Any]
) -> None:
    _ask("audit")

    assert main(["serve", "--host", "127.0.0.1"]) == 0
    with TestClient(launched["server"]) as client:
        [item] = client.get("/api/inbox").json()["items"]
        refused = client.post(
            f"/api/inbox/{item['id']}/settle", json={"decision": "answer", "answer": "17 Nov"}
        )

    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"]["message"].startswith(
        "This session was started with provider `demo`, which this server doesn't have."
    )
