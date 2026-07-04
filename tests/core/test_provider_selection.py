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


def _two_routed_provider_app() -> tuple[CayuApp, NamedFakeProvider, NamedFakeProvider]:
    app = CayuApp(enable_logging=False)
    openai_provider = NamedFakeProvider("openai")
    anthropic_provider = NamedFakeProvider("anthropic")
    app.register_provider(openai_provider, default=True, model_patterns=["gpt-*", "o*"])
    app.register_provider(anthropic_provider, model_patterns=["claude-*"])
    return app, openai_provider, anthropic_provider


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


def test_run_request_model_overrides_agent_default_model() -> None:
    app, default_provider, _ = _two_provider_app()
    app.register_agent(AgentSpec(name="assistant", model="agent-model"))

    events = asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_run_model_override",
                model="request-model",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert default_provider.requests[0].model == "request-model"
    session = asyncio.run(app.session_store.load("sess_run_model_override"))
    assert session is not None
    assert session.model == "request-model"


def test_model_pattern_routes_run_request_without_provider_name() -> None:
    app, openai_provider, anthropic_provider = _two_routed_provider_app()
    app.register_agent(AgentSpec(name="assistant", model="agent-model"))

    events = asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_model_pattern_route",
                model="claude-sonnet-4-6",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert openai_provider.requests == []
    assert len(anthropic_provider.requests) == 1
    assert anthropic_provider.requests[0].model == "claude-sonnet-4-6"
    session = asyncio.run(app.session_store.load("sess_model_pattern_route"))
    assert session is not None
    assert session.provider_name == "anthropic"
    assert session.model == "claude-sonnet-4-6"


def test_explicit_provider_name_wins_over_model_pattern_route() -> None:
    app, openai_provider, anthropic_provider = _two_routed_provider_app()
    app.register_agent(AgentSpec(name="assistant", model="agent-model"))

    events = asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_model_pattern_explicit_provider",
                provider_name="openai",
                model="claude-sonnet-4-6",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(openai_provider.requests) == 1
    assert openai_provider.requests[0].model == "claude-sonnet-4-6"
    assert anthropic_provider.requests == []
    session = asyncio.run(app.session_store.load("sess_model_pattern_explicit_provider"))
    assert session is not None
    assert session.provider_name == "openai"


def test_agent_provider_pin_wins_over_model_pattern_route() -> None:
    app, openai_provider, anthropic_provider = _two_routed_provider_app()
    app.register_agent(AgentSpec(name="assistant", model="agent-model", provider_name="openai"))

    events = asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_model_pattern_agent_provider",
                model="claude-sonnet-4-6",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(openai_provider.requests) == 1
    assert openai_provider.requests[0].model == "claude-sonnet-4-6"
    assert anthropic_provider.requests == []
    session = asyncio.run(app.session_store.load("sess_model_pattern_agent_provider"))
    assert session is not None
    assert session.provider_name == "openai"


def test_model_pattern_ambiguity_raises_before_session_creation() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(
        NamedFakeProvider("first"),
        default=True,
        model_patterns=["gpt-*"],
    )
    app.register_provider(NamedFakeProvider("second"), model_patterns=["gpt-5*"])
    app.register_agent(AgentSpec(name="assistant", model="agent-model"))

    with pytest.raises(
        ValueError,
        match="Model matches multiple registered providers: gpt-5.5 -> first, second",
    ):
        asyncio.run(
            _collect_run(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_model_pattern_ambiguous",
                    model="gpt-5.5",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )

    assert asyncio.run(app.session_store.load("sess_model_pattern_ambiguous")) is None


def test_provider_model_patterns_reject_string_value() -> None:
    app = CayuApp(enable_logging=False)

    with pytest.raises(TypeError, match="model_patterns must be an iterable of strings"):
        app.register_provider(
            NamedFakeProvider("bad-provider"),
            model_patterns="gpt-*",  # type: ignore[arg-type]
        )


def test_provider_model_patterns_reject_blank_pattern() -> None:
    app = CayuApp(enable_logging=False)

    with pytest.raises(ValueError):
        app.register_provider(NamedFakeProvider("bad-provider"), model_patterns=[" "])


def test_model_without_matching_pattern_uses_default_provider() -> None:
    app, openai_provider, anthropic_provider = _two_routed_provider_app()
    app.register_agent(AgentSpec(name="assistant", model="agent-model"))

    events = asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_model_pattern_default",
                model="unknown-model",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(openai_provider.requests) == 1
    assert openai_provider.requests[0].model == "unknown-model"
    assert anthropic_provider.requests == []
    session = asyncio.run(app.session_store.load("sess_model_pattern_default"))
    assert session is not None
    assert session.provider_name == "openai"


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


def test_resume_honors_session_provider_not_model_pattern_route() -> None:
    app, openai_provider, anthropic_provider = _two_routed_provider_app()
    app.register_agent(AgentSpec(name="assistant", model="gpt-5.5"))

    run_events = asyncio.run(
        _collect_run(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_model_pattern_resume",
                messages=[Message.text("user", "hello")],
            ),
        )
    )
    assert run_events[-1].type == EventType.SESSION_COMPLETED

    resume_events = asyncio.run(
        _collect_resume(
            app,
            ResumeRequest(
                session_id="sess_model_pattern_resume",
                model="claude-sonnet-4-6",
                messages=[Message.text("user", "again")],
            ),
        )
    )

    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    assert len(openai_provider.requests) == 2
    assert openai_provider.requests[1].model == "claude-sonnet-4-6"
    assert anthropic_provider.requests == []
    session = asyncio.run(app.session_store.load("sess_model_pattern_resume"))
    assert session is not None
    assert session.provider_name == "openai"
    assert session.model == "claude-sonnet-4-6"


def test_copy_run_request_preserves_provider_name() -> None:
    request = RunRequest(
        agent_name="assistant",
        provider_name="other-provider",
        model="request-model",
        messages=[Message.text("user", "hello")],
    )
    copied = copy_run_request(request)
    assert copied.provider_name == "other-provider"
    assert copied.model == "request-model"


def test_blank_provider_names_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentSpec(name="assistant", model="fake-model", provider_name="   ")
    with pytest.raises(ValidationError):
        RunRequest(
            agent_name="assistant",
            provider_name="",
            messages=[Message.text("user", "hello")],
        )
    with pytest.raises(ValidationError):
        RunRequest(
            agent_name="assistant",
            model=" ",
            messages=[Message.text("user", "hello")],
        )
