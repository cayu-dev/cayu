"""Per-run and per-agent provider selection (agent spec / run request overrides)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import CayuApp, ResumeRequest, RunRequest
from cayu.runtime.sessions import copy_run_request


class NamedFakeProvider(ModelProvider):
    def __init__(self, name: str) -> None:
        self.name = name
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelStreamEvent.text_delta(f"answered by {self.name}")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _two_provider_app() -> tuple[CayuApp, NamedFakeProvider, NamedFakeProvider]:
    app = CayuApp(enable_logging=False)
    default_provider = NamedFakeProvider("default-provider")
    other_provider = NamedFakeProvider("other-provider")
    app.register_provider(default_provider, default=True)
    app.register_provider(other_provider)
    return app, default_provider, other_provider


async def _collect_run(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


async def _collect_resume(app: CayuApp, request: ResumeRequest) -> list[Event]:
    return [event async for event in app.resume(request)]


def test_run_defaults_to_default_provider_when_nothing_pinned() -> None:
    app, default_provider, other_provider = _two_provider_app()
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_default",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(default_provider.requests) == 1
    assert other_provider.requests == []
    session = asyncio.run(app.session_store.load("sess_provider_default"))
    assert session is not None
    assert session.provider_name == "default-provider"


def test_agent_spec_provider_name_selects_non_default_provider() -> None:
    app, default_provider, other_provider = _two_provider_app()
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model", provider_name="other-provider")
    )

    events = asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_agent_pin",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert default_provider.requests == []
    assert len(other_provider.requests) == 1
    session = asyncio.run(app.session_store.load("sess_provider_agent_pin"))
    assert session is not None
    assert session.provider_name == "other-provider"


def test_run_request_provider_name_overrides_agent_spec() -> None:
    app, default_provider, other_provider = _two_provider_app()
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model", provider_name="default-provider")
    )

    events = asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_request_override",
                provider_name="other-provider",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert default_provider.requests == []
    assert len(other_provider.requests) == 1
    session = asyncio.run(app.session_store.load("sess_provider_request_override"))
    assert session is not None
    assert session.provider_name == "other-provider"


def test_run_with_unregistered_provider_name_raises_before_session_creation() -> None:
    app, _, _ = _two_provider_app()
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(KeyError, match="Provider not registered: missing-provider"):
        asyncio.run(
            _collect_run(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_provider_missing",
                    provider_name="missing-provider",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )

    assert asyncio.run(app.session_store.load("sess_provider_missing")) is None


def test_resume_honors_session_provider_not_default() -> None:
    app, default_provider, other_provider = _two_provider_app()
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model", provider_name="other-provider")
    )

    run_events = asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_resume",
                messages=[Message.text("user", "hello")],
            ),
        )
    )
    assert run_events[-1].type == EventType.SESSION_COMPLETED

    resume_events = asyncio.run(
        _collect_resume(
            app,
            ResumeRequest(
                session_id="sess_provider_resume",
                messages=[Message.text("user", "and again")],
            ),
        )
    )

    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    assert default_provider.requests == []
    assert len(other_provider.requests) == 2


def test_copy_run_request_preserves_provider_name() -> None:
    request = RunRequest(
        agent_name="assistant",
        provider_name="other-provider",
        messages=[Message.text("user", "hello")],
    )
    assert copy_run_request(request).provider_name == "other-provider"


def test_blank_provider_names_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentSpec(name="assistant", model="fake-model", provider_name="   ")
    with pytest.raises(ValidationError):
        RunRequest(
            agent_name="assistant",
            provider_name="",
            messages=[Message.text("user", "hello")],
        )
