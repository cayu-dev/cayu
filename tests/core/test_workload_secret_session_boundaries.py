from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest
from pydantic import SecretStr
from tests.core._workload_secret_support import (
    FakeProvider,
    collect_events,
    collect_fork_events,
    collect_resume_events,
)

from cayu.core import AgentSpec, Event, EventType, Message, MessageRole, ToolCallPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    ForkSessionRequest,
    InMemorySessionStore,
    InterruptSessionRequest,
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    Session,
    SessionIdentity,
    SessionQuery,
    SessionStatus,
)
from cayu.runtime._session_engine import _with_environment_name
from cayu.runtime._session_request_boundary import (
    prepare_derived_fork_session,
    prepare_fork_session_request,
    prepare_fork_source_session,
    prepare_run_request,
)
from cayu.runtime.event_sinks import InMemoryEventSink
from cayu.runtime.sessions import run_request_with_runtime_generated_authority
from cayu.storage import SQLiteSessionStore
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


def _safe_fork_sessions() -> tuple[Session, Session]:
    source = Session(
        id="source",
        agent_name="source-agent",
        provider_name="provider",
        model="source-model",
        causal_budget_id="source",
        runtime_name="runtime",
        environment_name="environment",
        status=SessionStatus.COMPLETED,
    )
    return source, Session(
        id="child",
        agent_name="target-agent",
        provider_name="provider",
        model="target-model",
        parent_session_id=source.id,
        causal_budget_id=source.causal_budget_id,
        runtime_name="runtime",
        environment_name="environment",
        status=SessionStatus.COMPLETED,
    )


def _assert_secret_absent_from_cayu_exception(exc: BaseException, secret: str) -> None:
    assert secret not in str(exc)
    traceback = exc.__traceback__
    while traceback is not None:
        if "/src/cayu/" in traceback.tb_frame.f_code.co_filename:
            leaked_names = [
                name for name, value in traceback.tb_frame.f_locals.items() if secret in repr(value)
            ]
            assert not leaked_names, (
                traceback.tb_frame.f_code.co_filename,
                traceback.tb_frame.f_code.co_name,
                leaked_names,
            )
        traceback = traceback.tb_next


@pytest.mark.parametrize(
    "field_name",
    (
        "agent_name",
        "provider_name",
        "model",
        "runtime_name",
        "runtime_version",
        "environment_name",
    ),
)
def test_derived_fork_rejects_secret_bearing_final_authority(field_name: str) -> None:
    secret = "derived-fork-authority-secret"
    source, fork = _safe_fork_sessions()
    fork = fork.model_copy(update={field_name: f"prefix-{secret}-suffix"})

    with pytest.raises(ValueError, match=field_name):
        prepare_derived_fork_session(
            fork,
            source_session=source,
            runtime_generated_session_id=None,
            store_resolved_source_session_id=None,
            redactor=SecretRedactor(secret),
        )


@pytest.mark.parametrize(
    ("update", "match"),
    (
        ({"labels": {"classification": "derived-fork-secret"}}, "labels"),
        ({"metadata": {"cayu:taint_labels": ["derived-fork-secret"]}}, "metadata"),
    ),
)
def test_derived_fork_rejects_secret_bearing_final_policy_state(
    update: dict,
    match: str,
) -> None:
    source, fork = _safe_fork_sessions()
    fork = fork.model_copy(update=update)

    with pytest.raises(ValueError, match=match):
        prepare_derived_fork_session(
            fork,
            source_session=source,
            runtime_generated_session_id=None,
            store_resolved_source_session_id=None,
            redactor=SecretRedactor("derived-fork-secret"),
        )


@pytest.mark.parametrize(
    "taint_label",
    (
        "fork-taint-secret",
        "prefix-fork-taint-secret-suffix",
    ),
)
def test_fork_rejects_secret_bearing_request_taint_before_any_mutation(
    taint_label: str,
) -> None:
    secret = "fork-taint-secret"

    async def run() -> None:
        store = InMemorySessionStore()
        sink = InMemoryEventSink()
        provider = FakeProvider([])
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="taint-fork-source",
                messages=[Message.text("user", "source")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fakemodel"),
        )
        await store.append_transcript_messages(
            source.id,
            [Message.text("user", "copied transcript")],
        )
        await store.checkpoint(source.id, {"safe": "checkpoint"})
        await store.update_status(source.id, SessionStatus.COMPLETED)
        source_before = await store.load(source.id)
        transcript_before = await store.load_transcript(source.id)
        checkpoint_before = await store.load_checkpoint(source.id)
        events_before = await store.load_events(source.id)
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fakemodel"))

        with pytest.raises(ValueError, match="policy authority") as exc_info:
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id="taint-fork-child",
                    metadata={"cayu:taint_labels": [taint_label]},
                ),
            )

        assert secret not in str(exc_info.value)
        assert await store.load("taint-fork-child") is None
        with pytest.raises(KeyError, match="taint-fork-child"):
            await store.load_transcript("taint-fork-child")
        assert await store.load_checkpoint("taint-fork-child") is None
        with pytest.raises(KeyError, match="taint-fork-child"):
            await store.load_events("taint-fork-child")
        assert sink.events == []
        assert provider.requests == []
        assert await store.load(source.id) == source_before
        assert await store.load_transcript(source.id) == transcript_before
        assert await store.load_checkpoint(source.id) == checkpoint_before
        assert await store.load_events(source.id) == events_before

    asyncio.run(run())


@pytest.mark.parametrize(
    ("secret", "generated_session_id"),
    (
        ("-", "00000000-0000-4000-8000-000000000000"),
        ("a", "00000000-0000-4000-8000-00000000000a"),
    ),
)
def test_derived_fork_accepts_short_secret_collision_in_generated_session_id(
    secret: str,
    generated_session_id: str,
) -> None:
    source = Session(
        id="root",
        agent_name="bot",
        provider_name="provider",
        model="model",
        causal_budget_id="root",
        runtime_name="core",
        status=SessionStatus.COMPLETED,
    )
    fork = Session(
        id=generated_session_id,
        agent_name="bot",
        provider_name="provider",
        model="model",
        parent_session_id=source.id,
        causal_budget_id=source.causal_budget_id,
        runtime_name="core",
        status=SessionStatus.COMPLETED,
    )

    prepared = prepare_derived_fork_session(
        fork,
        source_session=source,
        runtime_generated_session_id=generated_session_id,
        store_resolved_source_session_id=None,
        redactor=SecretRedactor(secret),
    )

    assert prepared.id == generated_session_id


def test_derived_fork_keeps_requested_and_generated_identity_trust_distinct() -> None:
    secret = "unsafe-child"
    source, fork = _safe_fork_sessions()
    fork = fork.model_copy(update={"id": f"prefix-{secret}-suffix"})

    with pytest.raises(ValueError, match="session_id"):
        prepare_derived_fork_session(
            fork,
            source_session=source,
            runtime_generated_session_id=None,
            store_resolved_source_session_id=None,
            redactor=SecretRedactor(secret),
        )
    with pytest.raises(ValueError, match="session_id"):
        prepare_derived_fork_session(
            fork.model_copy(update={"id": secret}),
            source_session=source,
            runtime_generated_session_id=secret,
            store_resolved_source_session_id=None,
            redactor=SecretRedactor(secret),
        )


def test_store_resolved_fork_source_rejects_exact_secret_authority() -> None:
    secret = "exact-source-secret"
    source, _fork = _safe_fork_sessions()
    source = source.model_copy(
        update={"id": secret, "causal_budget_id": secret},
    )

    with pytest.raises(ValueError, match="source_session.id"):
        prepare_fork_source_session(
            source,
            expected_source_session_id=secret,
            store_resolved_source_session_id=secret,
            redactor=SecretRedactor(secret),
        )


def test_public_alias_for_exact_secret_source_rejects_before_fork_mutation() -> None:
    secret = "exact-source-secret"

    async def run() -> None:
        store = InMemorySessionStore()
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=secret,
                messages=[Message.text("user", "source")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fakemodel"),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fakemodel"))
        public_source_id = app.project_session_id_for_exposure(source.id)
        await store.register_public_authority_alias(
            public_source_id,
            field_name="session_id",
            private_value=source.id,
        )

        with pytest.raises(ValueError, match="source_session_id"):
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=public_source_id,
                    session_id="fork-child",
                ),
            )

        assert await store.load("fork-child") is None
        assert await store.load(source.id) is not None

    asyncio.run(run())


def test_public_fork_detaches_resolved_source_from_derived_authority_rejection() -> None:
    source_secret = "resolved-fork-source-secret"
    model_secret = "derived-fork-model-secret"
    source_id = f"legacy-{source_secret}-session"

    async def run() -> None:
        store = InMemorySessionStore()
        sink = InMemoryEventSink()
        source = await store.create(
            RunRequest(
                agent_name="source-agent",
                session_id=source_id,
                messages=[Message.text("user", "source")],
            ),
            identity=SessionIdentity(provider_name="fake", model="source-model"),
        )
        await store.append_transcript_messages(
            source.id,
            [Message.text("user", "copied transcript")],
        )
        await store.checkpoint(source.id, {"safe": "checkpoint"})
        await store.update_status(source.id, SessionStatus.COMPLETED)
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            secret_redactor=SecretRedactor([source_secret, model_secret]),
            enable_logging=False,
        )
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="source-agent", model="source-model"))
        app.register_agent(AgentSpec(name="target-agent", model=model_secret))
        public_source_id = app.project_session_id_for_exposure(source.id)
        await store.register_public_authority_alias(
            public_source_id,
            field_name="session_id",
            private_value=source.id,
        )

        with pytest.raises(ValueError, match="model") as raised:
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=public_source_id,
                    session_id="fork-child",
                    agent_name="target-agent",
                ),
            )

        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None
        _assert_secret_absent_from_cayu_exception(raised.value, source_secret)
        _assert_secret_absent_from_cayu_exception(raised.value, model_secret)
        assert await store.load("fork-child") is None
        with pytest.raises(KeyError, match="fork-child"):
            await store.load_transcript("fork-child")
        assert await store.load_checkpoint("fork-child") is None
        with pytest.raises(KeyError, match="fork-child"):
            await store.load_events("fork-child")
        assert sink.events == []

    asyncio.run(run())


@pytest.mark.parametrize(
    "target_model",
    (
        "derived-target-secret",
        "prefix-derived-target-secret-suffix",
    ),
)
def test_fork_rejects_target_agent_derived_secret_before_any_publication(
    target_model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "derived-target-secret"

    async def run() -> None:
        store = InMemorySessionStore()
        sink = InMemoryEventSink()
        provider = FakeProvider([])
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="source-agent", model="source-model"))
        app.register_agent(AgentSpec(name="target-agent", model=target_model))

        async def unexpected_taint_read(**_kwargs) -> set[str]:
            raise AssertionError("derived authority must be validated before taint lookup")

        monkeypatch.setattr(
            app._tool_round_executor,
            "prior_taint_labels_for_policy",
            unexpected_taint_read,
        )
        source = await store.create(
            RunRequest(
                agent_name="source-agent",
                session_id="fork-source",
                messages=[Message.text("user", "source transcript")],
            ),
            identity=SessionIdentity(provider_name="fake", model="source-model"),
        )
        await store.append_transcript_messages(
            source.id,
            [Message.text("user", "copied transcript")],
        )
        await store.checkpoint(source.id, {"safe": "checkpoint"})
        await store.update_status(source.id, SessionStatus.COMPLETED)
        source_before = await store.load(source.id)
        transcript_before = await store.load_transcript(source.id)
        checkpoint_before = await store.load_checkpoint(source.id)
        events_before = await store.load_events(source.id)

        with pytest.raises(ValueError, match="model") as exc_info:
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id="fork-child",
                    agent_name="target-agent",
                ),
            )

        _assert_secret_absent_from_cayu_exception(exc_info.value, secret)
        assert await store.load("fork-child") is None
        with pytest.raises(KeyError, match="fork-child"):
            await store.load_transcript("fork-child")
        assert await store.load_checkpoint("fork-child") is None
        with pytest.raises(KeyError, match="fork-child"):
            await store.load_events("fork-child")
        assert sink.events == []
        assert await store.load(source.id) == source_before
        assert await store.load_transcript(source.id) == transcript_before
        assert await store.load_checkpoint(source.id) == checkpoint_before
        assert await store.load_events(source.id) == events_before
        assert provider.requests == []

    asyncio.run(run())


def test_fork_rejects_target_agent_provider_secret_before_mismatch_diagnostic() -> None:
    secret = "derived-provider-secret"

    async def run() -> None:
        store = InMemorySessionStore()
        sink = InMemoryEventSink()
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="source-agent", model="source-model"))
        app.register_agent(
            AgentSpec(
                name="target-agent",
                model="target-model",
                provider_name=f"prefix-{secret}-suffix",
            )
        )
        source = await store.create(
            RunRequest(
                agent_name="source-agent",
                session_id="provider-fork-source",
                messages=[Message.text("user", "source")],
            ),
            identity=SessionIdentity(provider_name="fake", model="source-model"),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)

        with pytest.raises(ValueError, match="provider_name") as exc_info:
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id="provider-fork-child",
                    agent_name="target-agent",
                ),
            )

        _assert_secret_absent_from_cayu_exception(exc_info.value, secret)
        assert await store.load("provider-fork-child") is None
        assert await store.load(source.id) is not None
        assert sink.events == []

    asyncio.run(run())


def test_same_agent_fork_ignores_unused_changed_provider_pin() -> None:
    secret = "unused-provider-pin-secret"

    async def run() -> None:
        store = InMemorySessionStore()
        source = await store.create(
            RunRequest(
                agent_name="historical-agent",
                session_id="historical-source",
                messages=[Message.text("user", "source")],
            ),
            identity=SessionIdentity(provider_name="fake", model="historical-model"),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(
            AgentSpec(
                name="historical-agent",
                model="current-model",
                provider_name=f"prefix-{secret}-suffix",
            )
        )

        await collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id=source.id,
                session_id="historical-child",
            ),
        )

        child = await store.load("historical-child")
        assert child is not None
        assert child.provider_name == "fake"
        assert child.model == "historical-model"

    asyncio.run(run())


def test_runtime_hook_can_fork_generated_short_secret_source_authority() -> None:
    class ForkCompletedSessionHook(RuntimeHook):
        def __init__(self) -> None:
            self.fork_events = []

        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            self.fork_events = await context.fork_session(
                ForkSessionRequest(source_session_id=context.session.id)
            )

    async def run() -> None:
        store = InMemorySessionStore()
        hook = ForkCompletedSessionHook()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor("-"),
            runtime_hooks=[hook],
            enable_logging=False,
        )
        app.register_provider(
            FakeProvider(
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="bot", model="model"))

        await collect_events(
            app,
            RunRequest(
                agent_name="bot",
                messages=[Message.text("user", "start")],
            ),
        )

        assert [event.type for event in hook.fork_events] == [EventType.SESSION_FORKED]
        sessions = (await store.list_sessions(SessionQuery(limit=10))).sessions
        assert len(sessions) == 2
        source = next(session for session in sessions if session.parent_session_id is None)
        child = next(session for session in sessions if session.parent_session_id == source.id)
        assert "-" in source.id
        assert "-" in child.id

    asyncio.run(run())


def test_runtime_hook_fork_survives_terminal_acknowledgement_loss(
    tmp_path: Path,
) -> None:
    class CommitThenLoseTerminalAcknowledgementStore(SQLiteSessionStore):
        def __init__(self) -> None:
            encoded_key = base64.urlsafe_b64encode(bytes([31]) * 32).decode("ascii").rstrip("=")
            super().__init__(
                tmp_path / "terminal-hook-ack-loss.sqlite3",
                public_authority_alias_codec=PublicAuthorityAliasCodec(
                    PublicAuthorityAliasKeyring(
                        active_key_id="test",
                        keys={"test": SecretStr(encoded_key)},
                    )
                ),
            )
            self.lost_acknowledgement = False

        async def append_event(self, session_id: str, event: Event) -> None:
            await super().append_event(session_id, event)
            if event.type == EventType.SESSION_COMPLETED and not self.lost_acknowledgement:
                self.lost_acknowledgement = True
                raise ConnectionError("terminal acknowledgement lost after commit")

    class ForkCompletedSessionHook(RuntimeHook):
        def __init__(self) -> None:
            self.calls = 0
            self.fork_events = []

        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            self.calls += 1
            self.fork_events = await context.fork_session(
                ForkSessionRequest(source_session_id=context.session.id)
            )

    async def run() -> None:
        store = CommitThenLoseTerminalAcknowledgementStore()
        hook = ForkCompletedSessionHook()
        try:
            app = CayuApp(
                session_store=store,
                secret_redactor=SecretRedactor("-"),
                runtime_hooks=[hook],
                enable_logging=False,
            )
            app.register_provider(
                FakeProvider(
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ),
                default=True,
            )
            app.register_agent(AgentSpec(name="bot", model="model"))

            events = await collect_events(
                app,
                RunRequest(
                    agent_name="bot",
                    messages=[Message.text("user", "start")],
                ),
            )

            assert store.lost_acknowledgement is True
            assert hook.calls == 1
            assert [event.type for event in hook.fork_events] == [EventType.SESSION_FORKED]
            assert [event.type for event in events].count(EventType.SESSION_COMPLETED) == 1
            assert [event.type for event in events].count(EventType.HOOK_STARTED) == 1
            assert [event.type for event in events].count(EventType.HOOK_COMPLETED) == 1

            sessions = (await store.list_sessions(SessionQuery(limit=10))).sessions
            assert len(sessions) == 2
            source = next(session for session in sessions if session.parent_session_id is None)
            child = next(session for session in sessions if session.parent_session_id == source.id)
            assert "-" in source.id
            assert "-" in child.id

            records = await store.load_events(source.id)
            assert [event.type for event in records].count(EventType.SESSION_COMPLETED) == 1
            assert [event.type for event in records].count(EventType.HOOK_STARTED) == 1
            assert [event.type for event in records].count(EventType.HOOK_COMPLETED) == 1
        finally:
            await store.close()

    asyncio.run(run())


def test_generated_short_secret_ids_support_multi_generation_forks() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor("-"),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fakemodel"))

        source_events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "start")],
            ),
        )
        child_events = await collect_fork_events(
            app,
            ForkSessionRequest(source_session_id=source_events[-1].session_id),
        )
        await collect_fork_events(
            app,
            ForkSessionRequest(source_session_id=child_events[-1].session_id),
        )

        sessions = (await store.list_sessions(SessionQuery(limit=10))).sessions
        assert len(sessions) == 3
        root = next(session for session in sessions if session.parent_session_id is None)
        child = next(session for session in sessions if session.parent_session_id == root.id)
        grandchild = next(session for session in sessions if session.parent_session_id == child.id)
        assert all("-" in session.id for session in (root, child, grandchild))
        assert child.causal_budget_id == root.id
        assert grandchild.causal_budget_id == root.id

    asyncio.run(run())


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
