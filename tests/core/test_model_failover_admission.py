"""Candidate admission through the public run boundary (feature still gated)."""

from __future__ import annotations

import asyncio
import base64
import gc
import warnings

import pytest
from tests.core._execution_profile_fixtures import versioned_test_provider_identity
from tests.core.test_model_failover_stages import _StageMemoryStore, _StageSQLiteStore
from tests.core.test_provider_operations import _ReconnectableProvider
from tests.core.test_runtime import _app_with_artifact_store, _valid_png_bytes

from cayu import (
    AgentSpec,
    CayuApp,
    EventQuery,
    EventType,
    FileAttachment,
    FileAttachmentKind,
    FilePart,
    Message,
    MessageRole,
    ModelFailoverPolicy,
    ModelTarget,
    OpenAIWebSearch,
    ProviderOperationResolutionAction,
    ProviderOperationResolutionRequest,
    RunRequest,
    ScriptedModelProvider,
    StructuredOutputSpec,
    StructuredOutputStrategy,
    Tool,
    ToolResult,
    ToolSpec,
)
from cayu.artifacts.attachments import RESOLVED_FILE_ATTACHMENTS_OPTION
from cayu.providers.base import (
    ModelProvider,
    ModelProviderError,
    ModelStreamEvent,
    _preflight_provider_portable_messages,
)
from cayu.providers.operations import ProviderOperationMode
from cayu.runtime import provider_operations
from cayu.runtime.provider_operations import (
    ProviderOperationResolutionConflict,
    inspect_provider_operation,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.sessions.base import SessionStore
from cayu.vaults.redaction import SecretRedactor


class _ModeProvider(ModelProvider):
    def __init__(self, name, mode, *, fail=False):
        self.name = name
        self.mode = mode
        self.requests = []
        self.fail = fail

    @property
    def provider_operation_mode(self):
        return self.mode

    async def stream(self, request):
        self.requests.append(request)
        if self.fail:
            raise ModelProviderError(
                "unavailable", provider=self.name, status_code=503, retryable=True
            )
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed()


class _DurableReconnectableProvider(_ReconnectableProvider):
    @property
    def execution_profile_identity(self):
        return versioned_test_provider_identity(self)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_public_failover_preserves_attached_file_bytes(monkeypatch, tmp_path, backend):

    def unavailable(_request):
        raise ModelProviderError("unavailable", provider="primary", status_code=503, retryable=True)

    async def scenario():
        store = (
            _StageMemoryStore()
            if backend == "memory"
            else _StageSQLiteStore(tmp_path / "files.sqlite")
        )
        app, artifacts = _app_with_artifact_store(tmp_path, session_store=store)
        primary = ScriptedModelProvider(response_factory=unavailable, name="primary")
        backup = ScriptedModelProvider(
            [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()], name="backup"
        )
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"))
        image = _valid_png_bytes()
        try:
            attached = await app.attach_file(
                image, filename="image.png", kind="image", session_id="file-failover"
            )
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="file-failover",
                        messages=[Message(role=MessageRole.USER, content=(attached,))],
                        retry_policy=RetryPolicy(max_attempts=1),
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                        ),
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
            artifact_id = attached.attachment["artifact_id"]
            assert (await artifacts.read_bytes(artifact_id)).content == image
            for provider in (primary, backup):
                assert len(provider.requests) == 1
                request = provider.requests[0]
                resolved = request.options[RESOLVED_FILE_ATTACHMENTS_OPTION][artifact_id]
                assert base64.b64decode(resolved["data_base64"]) == image
                assert any(
                    type(part) is FilePart and part.attachment["artifact_id"] == artifact_id
                    for message in request.messages
                    for part in message.content
                )
            assert await store.load_active_model_completion_stage("file-failover") is None
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())


def test_public_failover_preflight_cannot_mutate_or_observe_secret_request_material(monkeypatch):
    secret = "candidate-preflight-secret-canary"

    class MutatingPreflight(ScriptedModelProvider):
        def preflight_portable_messages(self, *, model, messages, tools):
            super().preflight_portable_messages(model=model, messages=messages, tools=tools)
            assert secret not in repr((messages, tools))
            messages.clear()
            if tools:
                tools[0]["name"] = "tampered"
                tools[0]["input_schema"].clear()
            tools.clear()

    class Echo(Tool):
        spec = ToolSpec(
            name="echo",
            description=secret,
            input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
        )

        async def run(self, ctx, args) -> ToolResult:
            raise AssertionError("No provider emitted a tool call.")

    def unavailable(_request):
        raise ModelProviderError("unavailable", provider="primary", status_code=503, retryable=True)

    async def scenario():
        primary = MutatingPreflight(response_factory=unavailable, name="primary")
        backup = MutatingPreflight(
            [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()], name="backup"
        )
        store = _StageMemoryStore()
        app = CayuApp(
            session_store=store, secret_redactor=SecretRedactor(secret), enable_logging=False
        )
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(
            AgentSpec(name="agent", model="small", system_prompt=secret), tools=[Echo()]
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="detached-preflight",
                    messages=[Message.text("user", "hello " + secret)],
                    retry_policy=RetryPolicy(max_attempts=1),
                    failover=ModelFailoverPolicy(
                        fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                    ),
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        for provider in (primary, backup):
            assert len(provider.requests) == 1
            actual = provider.requests[0]
            assert "hello" in repr(actual.messages)
            assert secret not in repr(actual)
            assert [tool["name"] for tool in actual.tools] == ["echo"]
            assert actual.tools[0]["input_schema"]["properties"]["value"]["type"] == "string"
        assert "tampered" not in repr(await store.load_events("detached-preflight"))

    asyncio.run(scenario())


def test_public_failover_rechecks_new_tool_history_before_backup_dispatch(monkeypatch):

    class NoToolHistory(ScriptedModelProvider):
        def preflight_portable_messages(self, *, model, messages, tools):
            _preflight_provider_portable_messages(
                model=model,
                messages=messages,
                tools=tools,
                supports_system_messages=True,
                supports_tool_history=False,
                supports_tool_definitions=True,
                supports_file_attachments=True,
            )

    tool_calls = []

    class Echo(Tool):
        spec = ToolSpec(name="echo", description="Echo", input_schema={"type": "object"})

        async def run(self, ctx, args):
            tool_calls.append("echo")
            return ToolResult(content="echoed")

    async def scenario():
        def respond(_request):
            if len(primary.requests) == 1:
                return [
                    ModelStreamEvent.tool_call(id="echo-call", name="echo", arguments={}),
                    ModelStreamEvent.completed(),
                ]
            raise ModelProviderError(
                "unavailable", provider="primary", status_code=503, retryable=True
            )

        primary = ScriptedModelProvider(response_factory=respond, name="primary")
        backup = NoToolHistory(
            [ModelStreamEvent.text_delta("unexpected"), ModelStreamEvent.completed()], name="backup"
        )
        store = _StageMemoryStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"), tools=[Echo()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="new-tool-history",
                    messages=[Message.text("user", "hello")],
                    retry_policy=RetryPolicy(max_attempts=1),
                    failover=ModelFailoverPolicy(
                        fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                    ),
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_FAILED, events[-1].payload
        assert "tool-history support" in str(events[-1].payload)
        assert len(primary.requests) == 2 and backup.requests == []
        assert tool_calls == ["echo"]
        checkpoint = await store.load_checkpoint("new-tool-history")
        assert checkpoint is not None and checkpoint["model_failover"]["candidate_index"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("material", ["system", "tools", "file"])
def test_initial_run_rejects_nonportable_fallback_before_mutation(monkeypatch, material):

    class Echo(Tool):
        spec = ToolSpec(name="echo", description="Echo", input_schema={"type": "object"})

        async def run(self, ctx, args) -> ToolResult:
            raise AssertionError("Admission must not execute tools.")

    async def scenario():
        primary = ScriptedModelProvider(
            [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()], name="primary"
        )
        backup = _ModeProvider("backup", ProviderOperationMode.SYNCHRONOUS)
        store = _StageMemoryStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(
            AgentSpec(
                name="agent",
                model="small",
                system_prompt="System" if material == "system" else None,
            ),
            tools=[Echo()] if material == "tools" else (),
        )
        message = Message.text("user", "hello")
        if material == "file":
            attachment = FileAttachment(
                artifact_id="image",
                kind=FileAttachmentKind.IMAGE,
                filename="image.png",
                content_type="image/png",
                size_bytes=1,
            )
            message = Message(
                role=MessageRole.USER,
                content=(FilePart(attachment=attachment.model_dump(mode="json")),),
            )
        with pytest.raises(ValueError, match="does not declare"):
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="nonportable",
                        messages=[message],
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                        ),
                    )
                )
            ]
        assert primary.requests == [] and backup.requests == []
        assert await store.load("nonportable") is None
        assert await store.load_checkpoint("nonportable") is None
        assert await store.query_events(EventQuery(session_id="nonportable")) == []

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("capability", ["target", "native-schema", "hosted-tools"])
def test_initial_run_preflights_every_candidate_before_mutation(
    monkeypatch, tmp_path, backend, capability
):

    class CapabilityProvider(_ModeProvider):
        supports_native_structured_output = True

        def reject(self, checked):
            if self.name == "backup" and checked == capability:
                raise ValueError("Fallback configuration is unsupported.")

        def preflight_model_target(self, *, model):
            self.reject("target")

        def preflight_native_structured_output_schema(self, json_schema):
            self.reject("native-schema")

        def preflight_hosted_tools(self, *, model, hosted_tools, options):
            self.reject("hosted-tools")

        async def stream(self, request):
            self.requests.append(request)
            yield ModelStreamEvent.text_delta('{"answer":"done"}')
            yield ModelStreamEvent.completed()

    async def scenario():
        store = (
            _StageMemoryStore()
            if backend == "memory"
            else _StageSQLiteStore(tmp_path / "candidate-preflight.sqlite")
        )
        primary = CapabilityProvider("primary", ProviderOperationMode.SYNCHRONOUS)
        backup = CapabilityProvider("backup", ProviderOperationMode.SYNCHRONOUS)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(
            AgentSpec(name="agent", model="small"),
            hosted_tools=[OpenAIWebSearch()] if capability == "hosted-tools" else (),
        )
        try:
            with pytest.raises(ValueError, match="Fallback configuration"):
                _ = [
                    event
                    async for event in app.run(
                        RunRequest(
                            agent_name="agent",
                            session_id="invalid-candidate",
                            messages=[Message.text("user", "hello")],
                            structured_output=(
                                StructuredOutputSpec(
                                    strategy=StructuredOutputStrategy.NATIVE,
                                    json_schema={
                                        "type": "object",
                                        "properties": {"answer": {"type": "string"}},
                                        "required": ["answer"],
                                    },
                                )
                                if capability == "native-schema"
                                else None
                            ),
                            failover=ModelFailoverPolicy(
                                fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                            ),
                        )
                    )
                ]
            assert primary.requests == [] and backup.requests == []
            assert await store.load("invalid-candidate") is None
            assert await store.load_checkpoint("invalid-candidate") is None
            assert await store.query_events(EventQuery(session_id="invalid-candidate")) == []
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "backup_mode",
    [ProviderOperationMode.BACKGROUND, "synchronous", True, None, "future"],
    ids=["mixed-mode", "untyped-mode", "boolean", "missing", "unknown"],
)
def test_public_run_rejects_invalid_candidate_mode_before_mutation(
    monkeypatch, tmp_path, backend, backup_mode
):

    async def scenario():
        store = (
            _StageMemoryStore()
            if backend == "memory"
            else _StageSQLiteStore(tmp_path / "invalid-mode.sqlite")
        )
        primary = _ModeProvider("primary", ProviderOperationMode.SYNCHRONOUS)
        backup = _ModeProvider("backup", backup_mode)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"))
        try:
            with pytest.raises((TypeError, ValueError), match="[Ff]ailover.*mode"):
                _ = [
                    event
                    async for event in app.run(
                        RunRequest(
                            agent_name="agent",
                            session_id="invalid-mode",
                            messages=[Message.text("user", "hello")],
                            failover=ModelFailoverPolicy(
                                fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                            ),
                        )
                    )
                ]
            assert not primary.requests and not backup.requests
            assert await store.load("invalid-mode") is None
            assert await store.load_checkpoint("invalid-mode") is None
            assert await store.query_events(EventQuery(session_id="invalid-mode")) == []
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())


def test_invalid_mode_diagnostics_do_not_format_adapter_values(
    monkeypatch, caplog, capsys, tmp_path
):
    canary = "mode-secret-canary"

    class Hostile:
        def __str__(self):
            raise AssertionError(canary)

        def __repr__(self):
            raise AssertionError(canary)

    # Collect earlier tests' cyclic resources before observing this memory-only
    # scenario. Their SQLite cleanup warnings are not adapter diagnostics.
    gc.collect()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        test_public_run_rejects_invalid_candidate_mode_before_mutation(
            monkeypatch, tmp_path, "memory", Hostile()
        )
    assert not captured
    assert canary not in caplog.text
    streams = capsys.readouterr()
    assert canary not in streams.out + streams.err


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_public_run_rechecks_mode_after_preparation_before_provider_entry(
    monkeypatch, tmp_path, backend
):
    primary = _ModeProvider("primary", ProviderOperationMode.SYNCHRONOUS, fail=True)
    backup = _ModeProvider("backup", ProviderOperationMode.SYNCHRONOUS)

    class DriftPreparation(SessionStore):
        async def _prepare_model_completion_stage_atomic(self, prepared):
            result = await super()._prepare_model_completion_stage_atomic(prepared)
            if result.stage.intent.get("provider_name") == "backup":
                # Drift after the actual durable boundary, not an invalid
                # initial configuration. Both modes still match each other.
                primary.mode = backup.mode = ProviderOperationMode.BACKGROUND
            return result

    class DriftMemory(DriftPreparation, _StageMemoryStore):
        model_failover_stage_version = 1
        invocation_lifecycle_command_version = 1

    class DriftSQLite(DriftPreparation, _StageSQLiteStore):
        model_failover_stage_version = 1
        invocation_lifecycle_command_version = 1

    async def scenario():
        store = DriftMemory() if backend == "memory" else DriftSQLite(tmp_path / "drift.sqlite")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"))
        try:
            try:
                _ = [
                    event
                    async for event in app.run(
                        RunRequest(
                            agent_name="agent",
                            session_id="mode-drift",
                            messages=[Message.text("user", "hello")],
                            retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                            failover=ModelFailoverPolicy(
                                fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                            ),
                        )
                    )
                ]
            except (ValueError, ExceptionGroup) as failure:
                assert "mode" in str(failure)
            assert len(primary.requests) == 1 and not backup.requests
            checkpoint = await store.load_checkpoint("mode-drift")
            assert checkpoint is not None
            route = checkpoint["model_failover"]
            assert route["candidate_index"] == 1
            assert route["attempts_used"] == 2
            assert all(
                item["execution_mode"] == "synchronous" for item in route["plan"]["candidates"]
            )
            events = await store.load_events("mode-drift")
            assert "mode changed" in repr([event.payload for event in events])
            assert not any(event.type is EventType.MODEL_COMPLETED for event in events)
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["completed", "service-rejection", "ambiguous-start"])
def test_background_primary_keeps_existing_operation_owner_without_fallback(monkeypatch, outcome):

    async def scenario():
        error = (
            None
            if outcome == "completed"
            else ModelProviderError(
                "unavailable", provider="reconnectable", status_code=503, retryable=True
            )
            if outcome == "service-rejection"
            else TimeoutError("submission acknowledgement lost")
        )
        provider = _ReconnectableProvider(background=True, start_error=error)
        store = _StageMemoryStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="agent", model="small"))
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="background-plan",
                    messages=[Message.text("user", "hello")],
                    retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0),
                    failover=ModelFailoverPolicy(
                        fallbacks=(ModelTarget(provider_name=provider.name, model="large"),)
                    ),
                )
            )
        ]
        assert provider.adapter.start_calls == 1
        assert provider.stream_calls == 0
        assert not any(event.type is EventType.MODEL_RETRY for event in events)
        checkpoint = await store.load_checkpoint("background-plan")
        assert checkpoint is not None
        route = checkpoint["model_failover"]
        assert route["candidate_index"] == 0 and route["attempts_used"] == 1
        assert all(item["execution_mode"] == "background" for item in route["plan"]["candidates"])
        assert sum(event.type is EventType.MODEL_FAILOVER_SELECTED for event in events) == 1
        if outcome == "completed":
            assert events[-1].type is EventType.SESSION_COMPLETED
        else:
            assert not any(event.type is EventType.MODEL_COMPLETED for event in events)
            assert await store.load_active_model_completion_stage("background-plan") is not None

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("total_attempts", [None, 1, 2])
def test_explicit_background_retry_preserves_routed_step_and_attempt_budget(
    monkeypatch, tmp_path, backend, total_attempts
):

    async def scenario():
        provider = _DurableReconnectableProvider(
            background=True, start_error=TimeoutError("submission acknowledgement lost")
        )
        path = tmp_path / "background-retry.sqlite"
        store = _StageMemoryStore() if backend == "memory" else _StageSQLiteStore(path)

        def application():
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="agent", model="small"))
            return app

        app = application()
        initial = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="background-retry",
                    messages=[Message.text("user", "hello")],
                    retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                    failover=(
                        None
                        if total_attempts is None
                        else ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name=provider.name, model="large"),),
                            max_total_attempts=total_attempts,
                        )
                    ),
                )
            )
        ]
        assert initial[-1].type is EventType.SESSION_INTERRUPTED
        if isinstance(store, _StageSQLiteStore):
            await store.close()
            store = _StageSQLiteStore(path)
            app = application()
        session = await store.load("background-retry")
        active = await store.load_active_model_completion_stage("background-retry")
        before = await store.load_checkpoint("background-retry")
        assert session is not None and active is not None and before is not None
        request = ProviderOperationResolutionRequest(
            session_id=session.id,
            stage_id=active.stage.stage_id,
            expected_run_epoch=session.run_epoch,
            action=ProviderOperationResolutionAction.FALLBACK_RETRY,
            reason="accept duplicate-request risk",
        )
        provider.adapter.start_error = None
        inspection = await inspect_provider_operation(store, session.id)
        if total_attempts == 1:
            assert inspection.allowed_resolutions == (ProviderOperationResolutionAction.FAIL,)
            before_events = await store.load_events(session.id)
            with pytest.raises(ProviderOperationResolutionConflict):
                _ = [event async for event in app.resolve_provider_operation(request)]

            async def stale_inspection(*_args):
                return inspection.model_copy(
                    update={
                        "allowed_resolutions": (
                            ProviderOperationResolutionAction.FALLBACK_RETRY,
                            ProviderOperationResolutionAction.FAIL,
                        )
                    }
                )

            # Read-only eligibility is advisory. The atomic disposition must
            # independently reject even a stale permissive inspection result.
            with monkeypatch.context() as stale:
                stale.setattr(provider_operations, "inspect_provider_operation", stale_inspection)
                with pytest.raises(ProviderOperationResolutionConflict, match="total attempt"):
                    _ = [event async for event in app.resolve_provider_operation(request)]
            assert await store.load(session.id) == session
            assert await store.load_checkpoint(session.id) == before
            assert await store.load_events(session.id) == before_events
            assert await store.load_active_model_completion_stage(session.id) == active
            assert provider.adapter.start_calls == 1
        if total_attempts != 2:
            failure_request = request.model_copy(
                update={"action": ProviderOperationResolutionAction.FAIL}
            )
            failed = [event async for event in app.resolve_provider_operation(failure_request)]
            assert failed[-1].type is EventType.SESSION_FAILED
            assert provider.adapter.start_calls == 1
            assert await store.load_active_model_completion_stage(session.id) is None
            if isinstance(store, _StageSQLiteStore):
                await store.close()
                store = _StageSQLiteStore(path)
                app = application()
            terminal_events = await store.load_events(session.id)
            replay = [event async for event in app.resolve_provider_operation(failure_request)]
            assert [event.type for event in replay] == [EventType.PROVIDER_OPERATION_RESOLVED]
            assert await store.load_events(session.id) == terminal_events
            assert provider.adapter.start_calls == 1
            if isinstance(store, _StageSQLiteStore):
                await store.close()
            return
        assert ProviderOperationResolutionAction.FALLBACK_RETRY in inspection.allowed_resolutions
        continued = [event async for event in app.resolve_provider_operation(request)]
        assert continued[-1].type is EventType.SESSION_COMPLETED, continued[-1].payload
        assert provider.adapter.start_calls == 2 and provider.stream_calls == 0
        after = await store.load_checkpoint(session.id)
        assert after is not None
        assert (
            after["model_failover"]["logical_step_id"]
            == before["model_failover"]["logical_step_id"]
        )
        assert after["model_failover"]["attempts_used"] == 2
        assert after["model_failover"]["candidate_attempt"] == 2
        assert after["model_failover"]["candidate_index"] == 0
        replay = [event async for event in app.resolve_provider_operation(request)]
        assert [event.type for event in replay] == [EventType.PROVIDER_OPERATION_RESOLVED]
        assert provider.adapter.start_calls == 2
        if isinstance(store, _StageSQLiteStore):
            await store.close()

    asyncio.run(scenario())
