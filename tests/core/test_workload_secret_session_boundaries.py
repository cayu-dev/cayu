from __future__ import annotations

import asyncio

import pytest
from tests.core._workload_secret_support import (
    FakeProvider,
    collect_events,
    collect_fork_events,
    collect_resume_events,
)

from cayu.core import AgentSpec, EventType, Message, MessageRole, ToolCallPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    ForkSessionRequest,
    InMemorySessionStore,
    InterruptSessionRequest,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RunRequest,
    SessionIdentity,
    SessionStatus,
)
from cayu.runtime._session_engine import _with_environment_name
from cayu.runtime._session_request_boundary import (
    prepare_fork_session_request,
    prepare_run_request,
)
from cayu.runtime.sessions import run_request_with_runtime_generated_authority
from cayu.vaults import REDACTED_SECRET, SecretRedactor


def test_runtime_attested_subagent_lineage_survives_short_secret_collision() -> None:
    request = RunRequest(
        agent_name="assistant",
        session_id="root-id_subagent_child",
        parent_session_id="root-id",
        causal_budget_id="root-id",
        messages=[Message.text("user", "review")],
    )
    request = run_request_with_runtime_generated_authority(
        request,
        "session_id",
        "parent_session_id",
        "causal_budget_id",
    )

    rewritten = _with_environment_name(request, "sandbox")
    prepared = prepare_run_request(rewritten, redactor=SecretRedactor("-"))

    assert prepared.session_id == request.session_id
    assert prepared.parent_session_id == request.parent_session_id
    assert prepared.causal_budget_id == request.causal_budget_id


def test_fork_destination_rejects_reserved_public_authority_namespace() -> None:
    with pytest.raises(ValueError, match="reserved public-authority alias namespace"):
        prepare_fork_session_request(
            ForkSessionRequest(
                source_session_id="source",
                session_id="cayu_authority_v1.key.session_id." + "A" * 43,
            ),
            redactor=SecretRedactor(),
            store_resolved_source_session_id="source",
        )


def test_cayu_app_redacts_workload_secrets_at_final_model_request_boundary() -> None:
    secret = "model-request-boundary-canary"

    class SecretDefinitionTool(Tool):
        spec = ToolSpec(
            name="secret_definition",
            description=f"Authenticate with {secret}.",
            input_schema={
                "type": "object",
                "properties": {"token": {"description": f"Must equal {secret}"}},
            },
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            raise AssertionError("tool should not run")

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SecretDefinitionTool()],
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_model_request_redaction",
                messages=[Message.text("user", f"accidentally echoed {secret}")],
                metadata={"diagnostic": secret},
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_model_request_redaction"))
    session = asyncio.run(store.load("sess_model_request_redaction"))
    serialized_request = str(provider.requests[0].model_dump(mode="json"))
    serialized_transcript = str([message.model_dump(mode="json") for message in transcript])

    assert session is not None
    assert secret not in serialized_request
    assert REDACTED_SECRET in serialized_request
    assert secret not in serialized_transcript
    assert secret not in str(session.model_dump(mode="json"))


def test_cayu_app_rejects_secret_bearing_provider_tool_authority() -> None:
    secret = "secret_tool_authority_canary"

    class SecretNameTool(Tool):
        spec = ToolSpec(
            name=secret,
            description="Unsafe provider authority.",
            input_schema={"type": "object"},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            raise AssertionError("tool should not run")

    store = InMemorySessionStore()
    provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SecretNameTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_secret_tool_authority",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.requests == []
    assert events[-1].type == EventType.SESSION_FAILED
    assert secret not in str([event.model_dump(mode="json") for event in events])


def test_cayu_app_rejects_secret_bearing_model_before_session_creation() -> None:
    secret = "secret-model-authority-canary"
    store = InMemorySessionStore()
    provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model=secret))

    with pytest.raises(ValueError, match="durable session authority"):
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_secret_model_authority",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )

    assert provider.requests == []
    assert asyncio.run(store.load("sess_secret_model_authority")) is None


def test_cayu_app_rejects_secret_bearing_resume_model_before_session_mutation() -> None:
    secret = "secret-resume-model-canary"
    store = InMemorySessionStore()
    provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_resume_model_authority",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    with pytest.raises(ValueError, match="durable session authority"):
        asyncio.run(
            collect_resume_events(
                app,
                ResumeRequest(
                    session_id="sess_resume_model_authority",
                    model=secret,
                    messages=[Message.text("user", "continue")],
                ),
            )
        )

    session = asyncio.run(store.load("sess_resume_model_authority"))
    assert session is not None
    assert session.status is SessionStatus.COMPLETED
    assert session.model == "fake-model"
    assert len(provider.requests) == 1


def test_cayu_app_rejects_secret_bearing_message_authority() -> None:
    secret = "secret_message_authority_canary"
    store = InMemorySessionStore()
    provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(ValueError, match="cannot be used as execution authority"):
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_secret_message_authority",
                    messages=[
                        Message.text("user", "continue"),
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=[
                                ToolCallPart(
                                    tool_call_id="call_1",
                                    tool_name=secret,
                                    arguments={},
                                )
                            ],
                        ),
                    ],
                ),
            )
        )

    assert provider.requests == []
    assert asyncio.run(store.load("sess_secret_message_authority")) is None


def test_cayu_app_redacts_fork_metadata_before_session_creation() -> None:
    secret = "fork-metadata-boundary-canary"
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_fork_redaction_source",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    events = asyncio.run(
        collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id="sess_fork_redaction_source",
                session_id="sess_fork_redaction_child",
                metadata={"note": f"contains {secret}"},
            ),
        )
    )
    fork = asyncio.run(store.load("sess_fork_redaction_child"))

    assert fork is not None
    assert fork.metadata == {"note": f"contains {REDACTED_SECRET}"}
    assert secret not in str([event.model_dump(mode="json") for event in events])


def test_interrupt_redacts_request_before_pending_checkpoint(monkeypatch) -> None:
    secret = "interrupt-checkpoint-boundary-canary"

    async def run():
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_redaction",
                messages=[Message.text("user", "start")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        checkpoint_before_terminal = []

        async def terminal_stream(**kwargs):
            checkpoint_before_terminal.append(
                await store.load_checkpoint("sess_interrupt_redaction")
            )
            yield await app._event_writer.emit(kwargs["event"])

        monkeypatch.setattr(
            app._session_engine,
            "_emit_terminal_event_with_hooks",
            terminal_stream,
        )
        events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_interrupt_redaction",
                    reason=f"stop because {secret}",
                    metadata={"note": f"contains {secret}"},
                    requested_by=ResolutionActor(
                        subject=f"operator-{secret}",
                        source=ResolutionActorSource.REQUEST,
                        claims={"note": f"contains {secret}"},
                    ),
                )
            )
        ]
        return events, checkpoint_before_terminal[0]

    events, checkpoint = asyncio.run(run())

    serialized = str(
        {
            "events": [event.model_dump(mode="json") for event in events],
            "checkpoint": checkpoint,
        }
    )
    assert secret not in serialized
    assert REDACTED_SECRET in serialized
