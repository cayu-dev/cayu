from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import warnings
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

import cayu.runtime._execution_profile_admission as execution_profile_admission
import cayu.runtime._session_engine as session_engine_module
from cayu import (
    EXECUTION_PROFILE_METADATA_KEY,
    AgentSpec,
    BeforeStopContext,
    BeforeStopDecision,
    BrowserWebFetchAdapter,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    CacheBreakpoint,
    CachePolicy,
    CayuApp,
    CheckpointCompactionContextPolicy,
    CommandPolicy,
    CommandPolicyDecision,
    CommandPolicyResult,
    CommandRequest,
    DenyPatternRule,
    DispatchRequest,
    DockerRunner,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    Event,
    EventType,
    ExecCommandTool,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAdoptionRejected,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileBehaviorIdentity,
    ExecutionProfileComponentClass,
    ExecutionProfileDecisionKind,
    ExecutionProfileIdentity,
    ExecutionProfileIdentityAvailability,
    ExecutionProfileIdentityStrength,
    ExecutionProfileMigrationRequired,
    ExecutionProfileMismatchError,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyError,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
    InMemorySessionStore,
    KnowledgeInjectionPolicy,
    LocalRunner,
    LoopPolicy,
    Message,
    MessageWindowContextPolicy,
    ModelCompactor,
    ModelPrice,
    ModelTarget,
    ParameterConstrainedToolPolicy,
    PriceBook,
    ProcessCommandPolicy,
    PromptCacheCompactor,
    RememberKnowledgePolicy,
    RememberKnowledgeTool,
    RequiredAllowlistRule,
    RequiredFieldRule,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RetryPolicy,
    RunLimits,
    RunRequest,
    RuntimeEvidenceRequest,
    RuntimeHook,
    ScriptedModelProvider,
    SearchTextTool,
    SecretRedactor,
    SessionIdentity,
    SessionStatus,
    SQLiteSessionStore,
    StaticToolPolicy,
    StructuredOutputSpec,
    TaintAwareToolPolicy,
    ThinkingConfig,
    Tool,
    ToolCallHookContext,
    ToolContext,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    ToolResult,
    ToolSpec,
    TranscriptDigestCompactor,
    UsageTriggeredContextPolicy,
    WebFetchTool,
    estimate_session_cost,
    runtime_evidence,
)
from cayu.providers import (
    AnthropicProvider,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    OpenAIProvider,
)
from cayu.runtime._event_projection import (
    prepare_new_runtime_event,
    project_persisted_runtime_event,
)
from cayu.runtime.execution_profiles import (
    build_execution_profile_identity,
    changed_execution_profile_components,
    event_with_execution_profile_fingerprint_authority,
    execution_profile_from_session_metadata,
    execution_profile_metadata_after_adoption,
    execution_profile_session_metadata,
)
from cayu.runtime.sessions import (
    _runtime_resume_transport_metadata,
    _with_runtime_resume_transport_metadata,
    execution_profile_adoption_request_fingerprint,
)


def test_dashboard_profile_fixture_delegates_to_the_runtime_resolver() -> None:
    path = Path(__file__).resolve().parents[2] / "examples/dashboard_behavior_live.py"
    source = path.read_text(encoding="utf-8")

    assert "execution_profile_admission.resolve_execution_profile_identity(" in source
    assert "process_identity=app._execution_profile_process_identity" in source


class RecordingTool(Tool):
    def __init__(
        self,
        name: str,
        *,
        parallel_safe: bool = True,
        workspace_mutation: bool = False,
    ) -> None:
        self.spec = ToolSpec(
            name=name,
            description="Record execution.",
            input_schema={"type": "object", "properties": {}},
            parallel_safe=parallel_safe,
            workspace_mutation=workspace_mutation,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:recording-tool",
                behavior_version="1",
                implementation_version="1",
            ),
        )
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        return ToolResult(content="recorded")


class IdentityConfiguredTool(Tool):
    def __init__(
        self,
        identity: ExecutionProfileBehaviorIdentity | None,
        *,
        input_schema: dict | None = None,
        opaque_behavior: str = "default",
    ) -> None:
        self.spec = ToolSpec(
            name="identity_configured_tool",
            description="Exercise implementation identity admission.",
            input_schema=(
                {"type": "object", "properties": {}} if input_schema is None else input_schema
            ),
            execution_profile_identity=identity,
        )
        super().__init__()
        self.opaque_behavior = opaque_behavior

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="not called")


class AlternateIdentityConfiguredTool(IdentityConfiguredTool):
    """A distinct opaque implementation with the same public tool contract."""


class OpaqueWebFetchAdapter:
    async def fetch(self, ctx: ToolContext, request: object) -> ToolResult:
        return ToolResult(content="not called")


class UnversionedProvider(ModelProvider):
    name = "unversioned"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class IdentityConfiguredProvider(UnversionedProvider):
    def __init__(self, identity: ExecutionProfileBehaviorIdentity) -> None:
        super().__init__()
        self._identity = identity

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return self._identity


class IdentityConfiguredCacheProvider(IdentityConfiguredProvider):
    def request_cache_policy(self, request: ModelRequest) -> CachePolicy | None:
        raw_policy = request.options.get("cache_policy")
        if raw_policy is None:
            return None
        return CachePolicy.model_validate(raw_policy)


class IdentityConfiguredContextPolicy(MessageWindowContextPolicy):
    def __init__(self, identity: ExecutionProfileBehaviorIdentity) -> None:
        super().__init__(max_messages=4)
        self._identity = identity

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return self._identity


class IdentityConfiguredContextCompactor(TranscriptDigestCompactor):
    def __init__(self, identity: ExecutionProfileBehaviorIdentity) -> None:
        super().__init__(max_summary_chars=4096)
        self._identity = identity

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return self._identity


class IdentityConfiguredEnvironmentFactory(EnvironmentFactory):
    def __init__(self, identity: ExecutionProfileBehaviorIdentity | None = None) -> None:
        self._identity = identity

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return self._identity or _test_behavior_identity("environment-factory")

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        return EnvironmentFactoryResult(
            environment=Environment(EnvironmentSpec(name=request.environment_name)),
        )


class IdentityConfiguredToolPolicy(ToolPolicy):
    def __init__(self, identity: ExecutionProfileBehaviorIdentity) -> None:
        self._identity = identity

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return self._identity

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class IdentityConfiguredHook(RuntimeHook):
    def __init__(
        self,
        name: str,
        identity: ExecutionProfileBehaviorIdentity | None = None,
    ) -> None:
        self._name = name
        self._identity = identity

    @property
    def name(self) -> str:
        return self._name

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return self._identity or ExecutionProfileBehaviorIdentity(
            name=f"tests:identity-configured-hook:{self.name}",
            behavior_version="1",
            implementation_version="1",
        )


class IdentityConfiguredCommandPolicy(CommandPolicy):
    def __init__(self, identity: ExecutionProfileBehaviorIdentity) -> None:
        self._identity = identity

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return self._identity

    async def evaluate(
        self,
        ctx: ToolContext,
        request: CommandRequest,
    ) -> CommandPolicyResult:
        del ctx, request
        return CommandPolicyResult(decision=CommandPolicyDecision.ALLOW)


class IdentityConfiguredRunner(LocalRunner):
    def __init__(
        self,
        root: Path,
        identity: ExecutionProfileBehaviorIdentity,
    ) -> None:
        super().__init__(root)
        self._identity = identity

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return self._identity


class PreflightRecordingProvider(ScriptedModelProvider):
    supports_native_structured_output = True

    def __init__(self) -> None:
        super().__init__(
            [
                ModelStreamEvent.text_delta('{"answer":"done"}'),
                ModelStreamEvent.completed({"finish_reason": "stop", "model": "fake-model"}),
            ],
            name="fake",
        )
        self.native_preflight_calls = 0

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return _test_behavior_identity("preflight-recording-provider")

    def preflight_native_structured_output_schema(self, json_schema: dict) -> None:
        self.native_preflight_calls += 1


class VersionedScriptedProvider(ScriptedModelProvider):
    def __init__(self, behavior_version: str) -> None:
        super().__init__(
            [ModelStreamEvent.completed({"finish_reason": "stop", "model": "fake-model"})],
            name="fake",
        )
        self._behavior_version = behavior_version

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:versioned-model-provider",
            behavior_version=self._behavior_version,
            implementation_version="1",
        )


class RecordingExecutionProfilePolicy(ExecutionProfilePolicy):
    def __init__(
        self,
        result: ExecutionProfilePolicyResult,
        *,
        identity: str = "test:execution-profile-policy:v1",
    ) -> None:
        self._result = result
        self._identity = identity
        self.requests: list[ExecutionProfilePolicyRequest] = []

    @property
    def identity(self) -> str:
        return self._identity

    async def decide(
        self,
        request: ExecutionProfilePolicyRequest,
    ) -> ExecutionProfilePolicyResult:
        self.requests.append(request)
        return self._result


class ConfiguredAdoptionLoopPolicy(LoopPolicy):
    def __init__(self, identity: str) -> None:
        self._identity = identity
        self.calls = 0
        self.metadata: list[dict] = []

    @property
    def name(self) -> str:
        return "configured-adoption-loop-policy"

    @property
    def adoption_replay_identity(self) -> str:
        return self._identity

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:configured-adoption-loop-policy",
            behavior_version=self._identity,
            implementation_version="1",
        )

    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        self.calls += 1
        self.metadata.append(context.metadata)
        return await super().before_stop(context)


class OpaqueConfiguredLoopPolicy(LoopPolicy):
    def __init__(self, decision: BeforeStopDecision) -> None:
        self._decision = decision
        self.calls = 0

    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        self.calls += 1
        return self._decision


class ConcurrentCompletedAdoptionStore(InMemorySessionStore):
    """Make one adoption finish before its already-preflighted peer commits."""

    def __init__(self) -> None:
        super().__init__()
        self.adoption_count = 0
        self.both_adoptions_prepared = asyncio.Event()

    async def admit_execution_profile_resume(self, session_id: str, **kwargs):
        self.adoption_count += 1
        adoption_index = self.adoption_count
        if adoption_index == 2:
            self.both_adoptions_prepared.set()
        await self.both_adoptions_prepared.wait()
        if adoption_index == 2:
            while True:
                session = await self.load(session_id)
                if session is not None and session.status is SessionStatus.COMPLETED:
                    break
                await asyncio.sleep(0)
        return await super().admit_execution_profile_resume(session_id, **kwargs)


class CommitThenRaiseAdoptionStore(InMemorySessionStore):
    """Lose the acknowledgement after the atomic adoption commit."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_after_commit = True

    async def admit_execution_profile_resume(self, session_id: str, **kwargs):
        result = await super().admit_execution_profile_resume(session_id, **kwargs)
        if self.fail_after_commit:
            self.fail_after_commit = False
            raise RuntimeError("adoption commit acknowledgement lost")
        return result


class BlockingAliasResolutionStore(InMemorySessionStore):
    """Pause one public session-alias lookup at its real store boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.blocked_alias: str | None = None
        self.alias_resolution_started = asyncio.Event()
        self.allow_alias_resolution = asyncio.Event()

    async def resolve_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> str | None:
        if public_alias == self.blocked_alias and field_name == "session_id":
            self.alias_resolution_started.set()
            await self.allow_alias_resolution.wait()
        return await super().resolve_public_authority_alias(
            public_alias,
            field_name=field_name,
            scope_session_id=scope_session_id,
        )


def _completed_provider() -> ScriptedModelProvider:
    return ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop", "model": "fake-model"}),
        ],
        name="fake",
    )


async def _collect(events: AsyncIterator[Event]) -> list[Event]:
    return [event async for event in events]


def test_profile_adoption_intent_requires_actor_provenance_source() -> None:
    with pytest.raises(ValueError, match="requested_by.source is required"):
        ExecutionProfileAdoptionIntent(
            idempotency_key="missing-actor-source-v1",
            reason="Adopt the reviewed profile.",
            requested_by=ResolutionActor(subject="maintainer"),
        )


def test_runtime_transport_trace_is_detached_from_adoption_semantics() -> None:
    intent = ExecutionProfileAdoptionIntent(
        idempotency_key="transport-trace-retry-v1",
        reason="Adopt the reviewed profile.",
        requested_by=ResolutionActor(
            subject="maintainer",
            source=ResolutionActorSource.REQUEST,
        ),
    )

    def attested_request(traceparent: str) -> ResumeRequest:
        request = ResumeRequest(
            session_id="transport-trace-adoption",
            messages=[Message.text("user", "resume")],
            metadata={"traceparent": traceparent},
            profile_adoption=intent,
        )
        return _with_runtime_resume_transport_metadata(
            request,
            {"traceparent": traceparent},
        )

    first_traceparent = "00-11111111111111111111111111111111-2222222222222222-01"
    retry_traceparent = "00-33333333333333333333333333333333-4444444444444444-01"
    first = attested_request(first_traceparent)
    retry = attested_request(retry_traceparent)

    assert first.metadata == retry.metadata == {}
    assert _runtime_resume_transport_metadata(first) == {"traceparent": first_traceparent}
    assert _runtime_resume_transport_metadata(retry) == {"traceparent": retry_traceparent}
    assert execution_profile_adoption_request_fingerprint(
        first,
        redactor=SecretRedactor(),
    ) == execution_profile_adoption_request_fingerprint(
        retry,
        redactor=SecretRedactor(),
    )

    raw_sdk_request = ResumeRequest(
        session_id="transport-trace-adoption",
        messages=[Message.text("user", "resume")],
        metadata={"traceparent": first_traceparent},
        profile_adoption=intent,
    )
    assert execution_profile_adoption_request_fingerprint(
        raw_sdk_request,
        redactor=SecretRedactor(),
    ) != execution_profile_adoption_request_fingerprint(
        first,
        redactor=SecretRedactor(),
    )


def test_runtime_transport_trace_stays_out_of_governed_adoption_replay() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("original_tool")],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="transport-trace-governed-adoption",
                    messages=[Message.text("user", "first")],
                )
            )
        )

        profile_policy = RecordingExecutionProfilePolicy(
            ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason="Approved deployment profile.",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            )
        )
        first_provider = _completed_provider()
        first_loop_policy = ConfiguredAdoptionLoopPolicy("transport-loop-policy:v1")
        app = CayuApp(
            session_store=store,
            execution_profile_policy=profile_policy,
            enable_logging=False,
        )
        app.register_provider(first_provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("replacement_tool")],
        )
        intent = ExecutionProfileAdoptionIntent(
            idempotency_key="transport-trace-governed-v1",
            reason="Adopt the reviewed profile.",
            requested_by=ResolutionActor(
                subject="maintainer",
                source=ResolutionActorSource.REQUEST,
            ),
        )
        first_traceparent = "00-11111111111111111111111111111111-2222222222222222-01"
        first_request = _with_runtime_resume_transport_metadata(
            ResumeRequest(
                session_id="transport-trace-governed-adoption",
                messages=[Message.text("user", "second")],
                metadata={"traceparent": first_traceparent},
                profile_adoption=intent,
                loop_policies=(first_loop_policy,),
            ),
            {"traceparent": first_traceparent},
        )
        first_events = await _collect(app.resume(first_request))

        assert first_events[0].type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
        resumed = next(event for event in first_events if event.type is EventType.SESSION_RESUMED)
        assert resumed.payload["traceparent"] == first_traceparent
        assert first_loop_policy.metadata == [{}]
        assert len(profile_policy.requests) == 1
        assert len(first_provider.requests) == 1

        retry_provider = _completed_provider()
        retry_loop_policy = ConfiguredAdoptionLoopPolicy("transport-loop-policy:v1")
        retry_app = CayuApp(session_store=store, enable_logging=False)
        retry_app.register_provider(retry_provider, default=True)
        retry_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("replacement_tool")],
        )
        retry_traceparent = "00-33333333333333333333333333333333-4444444444444444-01"
        retry_request = _with_runtime_resume_transport_metadata(
            ResumeRequest(
                session_id="transport-trace-governed-adoption",
                messages=[Message.text("user", "second")],
                metadata={"traceparent": retry_traceparent},
                profile_adoption=intent,
                loop_policies=(retry_loop_policy,),
            ),
            {"traceparent": retry_traceparent},
        )
        replayed = await _collect(retry_app.resume(retry_request))

        assert [event.id for event in replayed] == [first_events[0].id]
        assert retry_loop_policy.metadata == []
        assert retry_provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize("boundary", ("request_construction", "public_resume"))
def test_profile_adoption_copy_boundaries_do_not_emit_mutated_actor_diagnostics(
    boundary: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "profile-adoption-serializer-secret-canary-ABCDEFGHIJKLMNOP"

    class SecretBearingValue:
        def __repr__(self) -> str:
            return secret

    intent = ExecutionProfileAdoptionIntent(
        idempotency_key="diagnostic-boundary-v1",
        reason="Adopt the reviewed profile.",
        requested_by=ResolutionActor(
            subject="maintainer",
            source=ResolutionActorSource.REQUEST,
        ),
    )
    if boundary == "request_construction":
        intent.requested_by.subject = SecretBearingValue()  # type: ignore[assignment]

        def invoke() -> None:
            ResumeRequest(
                session_id="profile-adoption-diagnostic-boundary",
                messages=[Message.text("user", "resume")],
                profile_adoption=intent,
            )

    else:
        request = ResumeRequest(
            session_id="profile-adoption-diagnostic-boundary",
            messages=[Message.text("user", "resume")],
            profile_adoption=intent,
        )
        assert request.profile_adoption is not None
        request.profile_adoption.requested_by.subject = SecretBearingValue()  # type: ignore[assignment]
        app = CayuApp(enable_logging=False)

        def invoke() -> None:
            asyncio.run(_collect(app.resume(request)))

    with (
        warnings.catch_warnings(record=True) as emitted,
        caplog.at_level(logging.WARNING),
        pytest.raises(ValidationError) as raised,
    ):
        warnings.simplefilter("always")
        invoke()

    captured = capsys.readouterr()
    diagnostic_output = " ".join(
        (
            str(raised.value),
            repr(raised.value),
            captured.out,
            captured.err,
            *(record.getMessage() for record in caplog.records),
            *(str(record.message) for record in emitted),
        )
    )
    assert emitted == []
    assert secret not in diagnostic_output


def test_profile_adoption_copy_does_not_emit_mutated_thinking_diagnostics(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "profile-adoption-thinking-secret-canary-ABCDEFGHIJKLMNOP"

    class SecretBearingValue:
        def __repr__(self) -> str:
            return secret

    request = ResumeRequest(
        session_id="profile-adoption-thinking-diagnostic-boundary",
        messages=[Message.text("user", "resume")],
        thinking=ThinkingConfig(),
    )
    assert request.thinking is not None
    object.__setattr__(request.thinking, "enabled", SecretBearingValue())
    app = CayuApp(enable_logging=False)

    with (
        warnings.catch_warnings(record=True) as emitted,
        caplog.at_level(logging.WARNING),
        pytest.raises(ValidationError) as raised,
    ):
        warnings.simplefilter("always")
        asyncio.run(_collect(app.resume(request)))

    captured = capsys.readouterr()
    diagnostic_output = " ".join(
        (
            str(raised.value),
            repr(raised.value),
            captured.out,
            captured.err,
            *(record.getMessage() for record in caplog.records),
            *(str(record.message) for record in emitted),
        )
    )
    assert emitted == []
    assert secret not in diagnostic_output


def test_public_resume_owns_profile_adoption_before_alias_resolution() -> None:
    async def exercise() -> None:
        store = BlockingAliasResolutionStore()
        original_app = CayuApp(
            session_store=store,
            enable_logging=False,
        )
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("original_tool")],
        )
        initial = await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        private_session_id = initial[0].session_id

        policy = RecordingExecutionProfilePolicy(
            ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason="Approved deployment profile.",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            )
        )
        provider = _completed_provider()
        app = CayuApp(
            session_store=store,
            execution_profile_policy=policy,
            secret_redactor=SecretRedactor(private_session_id[:8]),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("replacement_tool")],
        )
        public_session_id = app.project_session_id_for_exposure(private_session_id)
        assert public_session_id != private_session_id
        store.blocked_alias = public_session_id
        request = ResumeRequest(
            session_id=public_session_id,
            messages=[Message.text("user", "second")],
            profile_adoption=ExecutionProfileAdoptionIntent(
                idempotency_key="owned-before-alias-v1",
                reason="Adopt the reviewed profile.",
                requested_by=ResolutionActor(
                    subject="original-maintainer",
                    source=ResolutionActorSource.REQUEST,
                    claims={"change": "original"},
                ),
            ),
        )

        resume_task = asyncio.create_task(_collect(app.resume(request)))
        await store.alias_resolution_started.wait()
        assert request.profile_adoption is not None
        request.profile_adoption.requested_by.subject = "mutated-maintainer"
        request.profile_adoption.requested_by.claims["change"] = "mutated"
        request.profile_adoption = ExecutionProfileAdoptionIntent(
            idempotency_key="mutated-after-entry-v1",
            reason="Mutated after the public call started.",
            requested_by=ResolutionActor(
                subject="replacement-maintainer",
                source=ResolutionActorSource.REQUEST,
            ),
        )
        store.allow_alias_resolution.set()

        events = await resume_task
        assert events[0].type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
        assert len(policy.requests) == 1
        observed_intent = policy.requests[0].intent
        assert observed_intent is not None
        assert observed_intent.idempotency_key == "owned-before-alias-v1"
        assert observed_intent.reason == "Adopt the reviewed profile."
        assert observed_intent.requested_by.subject == "original-maintainer"
        assert observed_intent.requested_by.claims == {"change": "original"}
        assert len(provider.requests) == 1

    asyncio.run(exercise())


def test_app_rejects_nonportable_execution_profile_policy_identity() -> None:
    policy = RecordingExecutionProfilePolicy(
        ExecutionProfilePolicyResult(
            action=ExecutionProfilePolicyAction.REJECT,
            reason="Not used.",
        ),
        identity="policy\x00identity",
    )

    with pytest.raises(ValueError, match="must not contain NUL"):
        CayuApp(execution_profile_policy=policy, enable_logging=False)


def _test_behavior_identity(
    name: str,
    *,
    behavior_version: str = "1",
    implementation_version: str = "1",
) -> ExecutionProfileBehaviorIdentity:
    return ExecutionProfileBehaviorIdentity(
        name=f"tests:{name}",
        behavior_version=behavior_version,
        implementation_version=implementation_version,
    )


def test_profile_strengths_report_identity_provenance() -> None:
    def resolve(
        app: CayuApp,
        *,
        request_loop_policies: tuple[LoopPolicy, ...] = (),
    ) -> ExecutionProfileIdentity:
        return execution_profile_admission.resolve_execution_profile_identity(
            registered_agent=app._agents["assistant"],
            provider_name="fake",
            model="fake-model",
            durable_system_prompt=None,
            runtime_name="cayu",
            runtime_version="test",
            redactor=app._secret_redactor,
            process_identity=app._execution_profile_process_identity,
            registered_environment=app._get_registered_environment(None),
            runtime_hooks=app._runtime_hooks,
            loop_policies=app._loop_policies,
            loop_policy_identities=app._loop_policy_execution_profile_identities,
            invocation_loop_policies=request_loop_policies,
            invocation_loop_policy_identities=tuple(
                policy.execution_profile_identity for policy in request_loop_policies
            ),
        )

    cayu_owned_app = CayuApp(enable_logging=False)
    cayu_owned_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SearchTextTool()],
        tool_policy=TaintAwareToolPolicy(
            taint_sources={"search_text": ("external",)},
            protected_tools={"search_text": ("external",)},
        ),
    )
    cayu_owned_profile = resolve(cayu_owned_app)

    identity = _test_behavior_identity("declared-authority")
    declared_app = CayuApp(
        runtime_hooks=(IdentityConfiguredHook("declared-hook", identity),),
        enable_logging=False,
    )
    declared_app.register_environment(
        Environment(
            EnvironmentSpec(name="declared-environment", execution_profile_identity=identity)
        ),
        default=True,
    )
    declared_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[IdentityConfiguredTool(identity)],
        tool_policy=IdentityConfiguredToolPolicy(identity),
    )
    declared_profile = resolve(
        declared_app,
        request_loop_policies=(ConfiguredAdoptionLoopPolicy("declared-invocation"),),
    )

    structural_components = (
        ExecutionProfileComponentClass.RUNTIME,
        ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
        ExecutionProfileComponentClass.EXECUTION_POLICIES,
        ExecutionProfileComponentClass.INVOCATION_POLICIES,
        ExecutionProfileComponentClass.RUNTIME_HOOKS,
        ExecutionProfileComponentClass.EXECUTION_ENVIRONMENT,
    )
    for component_class in structural_components:
        assert (
            cayu_owned_profile.component(component_class).strength
            is ExecutionProfileIdentityStrength.STRUCTURAL
        )

    for component_class in structural_components[1:]:
        assert (
            declared_profile.component(component_class).strength
            is ExecutionProfileIdentityStrength.APPLICATION_VERSIONED
        )
    assert (
        declared_profile.component(ExecutionProfileComponentClass.RUNTIME).strength
        is ExecutionProfileIdentityStrength.STRUCTURAL
    )


def _profile_price_book(*, rate: str = "1") -> PriceBook:
    return PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal(rate),
                output_per_million=Decimal(rate),
            ),
        )
    )


def _profile_budget_limit(
    *,
    maximum: str = "10",
    scope: str = "session",
    reserve: bool = False,
) -> BudgetLimit:
    return BudgetLimit(
        scope=scope,
        max_estimated_cost=Decimal(maximum),
        pricing=_profile_price_book(),
        reservation=(
            BudgetReservation(max_input_tokens=8, max_output_tokens=8) if reserve else None
        ),
    )


def _model_semantics_profile(
    *,
    context_policy=None,
    provider: ScriptedModelProvider | None = None,
    provider_options: dict | None = None,
    budget_policy: BudgetPolicy | None = None,
    request_budget_limits: tuple[BudgetLimit, ...] = (),
    structured_output: StructuredOutputSpec | None = None,
    thinking: ThinkingConfig | None = None,
    max_steps: int = 16,
    limits: RunLimits | None = None,
    retry_policy: RetryPolicy | None = None,
) -> ExecutionProfileIdentity:
    app = CayuApp(budget_policy=budget_policy, enable_logging=False)
    app.register_provider(_completed_provider() if provider is None else provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            provider_options={} if provider_options is None else provider_options,
        ),
        context_policy=context_policy,
    )
    return session_engine_module._execution_profile_identity(
        registered_agent=app._agents["assistant"],
        provider_name="fake",
        registered_provider=app._providers["fake"],
        model="fake-model",
        durable_system_prompt="private durable instruction sentinel",
        redactor=app._secret_redactor,
        process_identity=app._execution_profile_process_identity,
        budget_policy=app.budget_policy,
        request_budget_limits=request_budget_limits,
        causal_budget_id="profile-budget",
        structured_output=structured_output,
        thinking=thinking,
        max_steps=max_steps,
        limits=limits,
        retry_policy=retry_policy,
    )


def test_schema_v3_covers_model_decision_semantics_without_raw_material() -> None:
    context_policy = KnowledgeInjectionPolicy(
        base_policy=CheckpointCompactionContextPolicy(
            compactor=TranscriptDigestCompactor(max_summary_chars=4096),
            max_user_turns=3,
            compact_after_messages=8,
        ),
        max_hits=3,
        max_bytes=4096,
    )
    profile = _model_semantics_profile(
        context_policy=context_policy,
        provider_options={"fake": {"temperature": 0.25}},
        budget_policy=BudgetPolicy(limits=(_profile_budget_limit(scope="app"),)),
        request_budget_limits=(_profile_budget_limit(),),
        structured_output=StructuredOutputSpec(
            json_schema={
                "type": "object",
                "properties": {"private_schema_field": {"type": "string"}},
            },
            strategy="tool",
            repair_prompt="private repair instruction sentinel",
        ),
        thinking=ThinkingConfig(effort="medium"),
        max_steps=7,
        limits=RunLimits(max_total_tokens=1000),
        retry_policy=RetryPolicy(max_attempts=2),
    )

    assert profile.schema_version == 3
    assert {component.component_class for component in profile.components} == set(
        ExecutionProfileComponentClass
    )
    serialized = profile.model_dump_json()
    assert "private durable instruction sentinel" not in serialized
    assert "private_schema_field" not in serialized
    assert "private repair instruction sentinel" not in serialized


def test_nested_compactor_identity_is_copied_at_agent_registration() -> None:
    class DeclaredCompactor(TranscriptDigestCompactor):
        def __init__(self, behavior_version: str) -> None:
            super().__init__(max_summary_chars=4096)
            self.behavior_version = behavior_version

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:declared-context-compactor",
                behavior_version=self.behavior_version,
                implementation_version="1",
            )

    compactor = DeclaredCompactor("1")
    policy = KnowledgeInjectionPolicy(
        base_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=2,
        )
    )
    first_app = CayuApp(enable_logging=False)
    first_app.register_provider(_completed_provider(), default=True)
    first_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    first_profile = session_engine_module._execution_profile_identity(
        registered_agent=first_app._agents["assistant"],
        provider_name="fake",
        registered_provider=first_app._providers["fake"],
        model="fake-model",
        durable_system_prompt=None,
        redactor=first_app._secret_redactor,
        process_identity=first_app._execution_profile_process_identity,
    )
    compactor.behavior_version = "2"
    repeated_profile = session_engine_module._execution_profile_identity(
        registered_agent=first_app._agents["assistant"],
        provider_name="fake",
        registered_provider=first_app._providers["fake"],
        model="fake-model",
        durable_system_prompt=None,
        redactor=first_app._secret_redactor,
        process_identity=first_app._execution_profile_process_identity,
    )

    second_app = CayuApp(enable_logging=False)
    second_app.register_provider(_completed_provider(), default=True)
    second_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )
    replacement_profile = session_engine_module._execution_profile_identity(
        registered_agent=second_app._agents["assistant"],
        provider_name="fake",
        registered_provider=second_app._providers["fake"],
        model="fake-model",
        durable_system_prompt=None,
        redactor=second_app._secret_redactor,
        process_identity=second_app._execution_profile_process_identity,
    )

    component_class = ExecutionProfileComponentClass.CONTEXT_COMPACTION
    assert (
        first_profile.component(component_class).strength
        is ExecutionProfileIdentityStrength.APPLICATION_VERSIONED
    )
    assert first_profile.component(component_class) == repeated_profile.component(component_class)
    assert first_profile.component(component_class) != replacement_profile.component(
        component_class
    )


def test_nested_model_compactor_provider_identity_is_copied_at_agent_registration() -> None:
    provider = VersionedScriptedProvider("1")
    policy = CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(provider=provider, model="summary-model"),
        max_user_turns=2,
    )

    def registered_profile(app: CayuApp) -> ExecutionProfileIdentity:
        app.register_provider(_completed_provider(), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=policy,
        )
        return session_engine_module._execution_profile_identity(
            registered_agent=app._agents["assistant"],
            provider_name="fake",
            registered_provider=app._providers["fake"],
            model="fake-model",
            durable_system_prompt=None,
            redactor=app._secret_redactor,
            process_identity=app._execution_profile_process_identity,
        )

    first_app = CayuApp(enable_logging=False)
    first_profile = registered_profile(first_app)
    provider._behavior_version = "2"
    repeated_profile = session_engine_module._execution_profile_identity(
        registered_agent=first_app._agents["assistant"],
        provider_name="fake",
        registered_provider=first_app._providers["fake"],
        model="fake-model",
        durable_system_prompt=None,
        redactor=first_app._secret_redactor,
        process_identity=first_app._execution_profile_process_identity,
    )
    replacement_profile = registered_profile(CayuApp(enable_logging=False))

    component_class = ExecutionProfileComponentClass.CONTEXT_COMPACTION
    component = first_profile.component(component_class)
    assert component.strength is ExecutionProfileIdentityStrength.APPLICATION_VERSIONED
    assert component == repeated_profile.component(component_class)
    assert component != replacement_profile.component(component_class)


def test_prompt_cache_compactor_carries_declared_fallback_identity() -> None:
    class DeclaredFallback(TranscriptDigestCompactor):
        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return _test_behavior_identity("prompt-cache-fallback")

    profile = _model_semantics_profile(
        context_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(
                provider=ScriptedModelProvider([], name="compaction-provider"),
                fallback_compactor=DeclaredFallback(max_summary_chars=4096),
            ),
            max_user_turns=2,
        )
    )

    component = profile.component(ExecutionProfileComponentClass.CONTEXT_COMPACTION)
    assert component.strength is ExecutionProfileIdentityStrength.APPLICATION_VERSIONED


def test_usage_triggered_policy_preserves_nested_application_identity_strength() -> None:
    class DeclaredContextPolicy(MessageWindowContextPolicy):
        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return _test_behavior_identity("usage-triggered-context")

    profile = _model_semantics_profile(
        context_policy=UsageTriggeredContextPolicy(
            base_policy=MessageWindowContextPolicy(max_messages=4),
            triggered_policy=DeclaredContextPolicy(max_messages=2),
            min_input_tokens=1,
        )
    )

    for component_class in (
        ExecutionProfileComponentClass.CONTEXT_SELECTION,
        ExecutionProfileComponentClass.KNOWLEDGE_INJECTION,
        ExecutionProfileComponentClass.CONTEXT_COMPACTION,
    ):
        assert (
            profile.component(component_class).strength
            is ExecutionProfileIdentityStrength.APPLICATION_VERSIONED
        )


def test_model_compactor_binds_its_snapshotted_provider_identity() -> None:
    def profile(provider_name: str) -> ExecutionProfileIdentity:
        return _model_semantics_profile(
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=ScriptedModelProvider([], name=provider_name),
                    model="summary-model",
                ),
                max_user_turns=2,
            )
        )

    first = profile("compaction-a")
    second = profile("compaction-b")

    component_class = ExecutionProfileComponentClass.CONTEXT_COMPACTION
    assert first.component(component_class) != second.component(component_class)


def _registered_context_profile(app: CayuApp) -> ExecutionProfileIdentity:
    return session_engine_module._execution_profile_identity(
        registered_agent=app._agents["assistant"],
        provider_name="fake",
        registered_provider=app._providers["fake"],
        model="fake-model",
        durable_system_prompt=None,
        redactor=app._secret_redactor,
        process_identity=app._execution_profile_process_identity,
    )


def test_private_knowledge_configuration_mutation_changes_process_local_profile() -> None:
    first_private_namespace = "private-knowledge-namespace-a"
    second_private_namespace = "private-knowledge-namespace-b"
    policy = KnowledgeInjectionPolicy(namespace=first_private_namespace, enabled=False)
    app = CayuApp(enable_logging=False)
    app.register_provider(_completed_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    first = _registered_context_profile(app)
    policy.namespace = second_private_namespace
    second = _registered_context_profile(app)

    component_class = ExecutionProfileComponentClass.KNOWLEDGE_INJECTION
    first_component = first.component(component_class)
    second_component = second.component(component_class)
    serialized = json.dumps(
        [first_component.model_dump(mode="json"), second_component.model_dump(mode="json")],
        sort_keys=True,
    )
    assert first_component.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
    assert second_component.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
    assert first_component != second_component
    assert first_private_namespace not in serialized
    assert second_private_namespace not in serialized
    assert app._execution_profile_process_identity not in serialized


@pytest.mark.parametrize("compactor_kind", ["model", "prompt_cache"])
def test_private_builtin_compactor_mutation_changes_process_local_profile(
    compactor_kind: str,
) -> None:
    first_private_option = "private-compactor-option-a"
    second_private_option = "private-compactor-option-b"
    provider = ScriptedModelProvider([], name="compaction-provider")
    if compactor_kind == "model":
        compactor = ModelCompactor(
            provider=provider,
            model="summary-model",
            options={"private": first_private_option},
        )
    else:
        compactor = PromptCacheCompactor(
            provider=provider,
            options={"private": first_private_option},
        )
    app = CayuApp(enable_logging=False)
    app.register_provider(_completed_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=compactor,
            max_user_turns=2,
        ),
    )

    first = _registered_context_profile(app)
    compactor.options["private"] = second_private_option
    second = _registered_context_profile(app)

    component_class = ExecutionProfileComponentClass.CONTEXT_COMPACTION
    first_component = first.component(component_class)
    second_component = second.component(component_class)
    serialized = json.dumps(
        [first_component.model_dump(mode="json"), second_component.model_dump(mode="json")],
        sort_keys=True,
    )
    assert first_component.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
    assert second_component.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
    assert first_component != second_component
    assert first_private_option not in serialized
    assert second_private_option not in serialized
    assert app._execution_profile_process_identity not in serialized


def test_nondefault_checkpoint_summary_prefix_is_private_process_local_material() -> None:
    summary_prefix = "tenant-a-summary"

    def component_for(app: CayuApp, prefix: str):
        app.register_provider(_completed_provider(), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(summary_prefix=prefix),
        )
        return _registered_context_profile(app).component(
            ExecutionProfileComponentClass.CONTEXT_SELECTION
        )

    first_app = CayuApp(enable_logging=False)
    second_app = CayuApp(enable_logging=False)
    first = component_for(first_app, summary_prefix)
    second = component_for(second_app, summary_prefix)
    default = component_for(
        CayuApp(enable_logging=False),
        "Previous session context summary:",
    )
    serialized = json.dumps(
        [first.model_dump(mode="json"), second.model_dump(mode="json")],
        sort_keys=True,
    )

    assert first.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
    assert second.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
    assert first != second
    assert default.strength is ExecutionProfileIdentityStrength.STRUCTURAL
    assert summary_prefix not in serialized
    assert first_app._execution_profile_process_identity not in serialized
    assert second_app._execution_profile_process_identity not in serialized


def test_queued_target_profile_uses_durable_request_controls() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                name="source",
            ),
            default=True,
        )
        app.register_provider(
            ScriptedModelProvider(
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                name="target",
            )
        )
        app.register_agent(AgentSpec(name="assistant", model="source-model"))
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="queued-target-provider-request-profile",
                    messages=[Message.text("user", "first")],
                    thinking=ThinkingConfig(effort="medium"),
                )
            )
        )
        session = await store.load("queued-target-provider-request-profile")
        assert session is not None
        source_profile = execution_profile_from_session_metadata(session.metadata)

        request = DispatchRequest(
            session_id=session.id,
            messages=[Message.text("user", "queued")],
            target=ModelTarget(provider_name="target", model="target-model"),
            thinking=ThinkingConfig(effort="low"),
            max_steps=7,
        )
        target_profile = app._session_engine._queued_dispatch_required_profile(
            session=session,
            source_profile=source_profile,
            request=request,
        )

        assert target_profile.component(
            ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY
        ) != source_profile.component(ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY)
        assert target_profile.component(
            ExecutionProfileComponentClass.FINALIZATION
        ) != source_profile.component(ExecutionProfileComponentClass.FINALIZATION)
        assert target_profile.component(
            ExecutionProfileComponentClass.PROVIDER_TARGET
        ) != source_profile.component(ExecutionProfileComponentClass.PROVIDER_TARGET)

    asyncio.run(exercise())


def test_scripted_provider_is_structural_only_for_the_exact_builtin_type() -> None:
    class DerivedScriptedProvider(ScriptedModelProvider):
        pass

    def provider_component(provider: ModelProvider):
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        return _registered_context_profile(app).component(
            ExecutionProfileComponentClass.PROVIDER_ADAPTER
        )

    built_in = _completed_provider()
    derived = DerivedScriptedProvider([], name="fake")

    assert built_in.execution_profile_identity is None
    assert derived.execution_profile_identity is None
    assert provider_component(built_in).strength is ExecutionProfileIdentityStrength.STRUCTURAL
    assert provider_component(derived).strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL


def test_opaque_provider_options_are_secret_safe_and_content_bound_within_process() -> None:
    first_secret = "private-provider-option-alpha-canary"
    second_secret = "private-provider-option-beta-canary"
    app = CayuApp(
        secret_redactor=SecretRedactor([first_secret, second_secret]),
        enable_logging=False,
    )
    app.register_provider(_completed_provider(), default=True)
    app.register_agent(
        AgentSpec(
            name="first",
            model="fake-model",
            provider_options={
                "fake": {
                    "temperature": 0.25,
                    "private_route": first_secret,
                }
            },
        )
    )
    app.register_agent(
        AgentSpec(
            name="second",
            model="fake-model",
            provider_options={
                "fake": {
                    "temperature": 0.25,
                    "private_route": second_secret,
                }
            },
        )
    )

    profiles = [
        session_engine_module._execution_profile_identity(
            registered_agent=app._agents[agent_name],
            provider_name="fake",
            registered_provider=app._providers["fake"],
            model="fake-model",
            durable_system_prompt=None,
            redactor=app._secret_redactor,
            process_identity=app._execution_profile_process_identity,
        )
        for agent_name in ("first", "second")
    ]
    components = [
        profile.component(ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY)
        for profile in profiles
    ]
    serialized = json.dumps(
        [profile.model_dump(mode="json") for profile in profiles],
        sort_keys=True,
    )

    assert components[0].strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
    assert components[0].fingerprint != components[1].fingerprint
    assert first_secret not in serialized
    assert second_secret not in serialized
    assert app._execution_profile_process_identity not in serialized


def test_runtime_owned_private_option_commitments_survive_secret_collision() -> None:
    process_identity = "fixed-private-options-profile-process-key"
    first_options = {"fake": {"temperature": 0.25, "private_route": "private-route-alpha"}}
    second_options = {"fake": {"temperature": 0.25, "private_route": "private-route-beta"}}
    projected_options = [
        session_engine_module._execution_profile_provider_options(
            options,
            process_identity=process_identity,
        )[0]
        for options in (first_options, second_options)
    ]
    commitment_secrets = [
        projected["private_configuration_hmac_sha256"] for projected in projected_options
    ]
    app = CayuApp(
        secret_redactor=SecretRedactor(commitment_secrets),
        enable_logging=False,
    )
    app.register_provider(_completed_provider(), default=True)
    for name, options in (("first", first_options), ("second", second_options)):
        app.register_agent(AgentSpec(name=name, model="fake-model", provider_options=options))

    components = [
        session_engine_module._execution_profile_identity(
            registered_agent=app._agents[name],
            provider_name="fake",
            registered_provider=app._providers["fake"],
            model="fake-model",
            durable_system_prompt=None,
            redactor=app._secret_redactor,
            process_identity=process_identity,
        ).component(ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY)
        for name in ("first", "second")
    ]
    serialized = json.dumps(
        [component.model_dump(mode="json") for component in components],
        sort_keys=True,
    )

    assert components[0].strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
    assert components[0] != components[1]
    assert "private-route-alpha" not in serialized
    assert "private-route-beta" not in serialized
    assert process_identity not in serialized


def test_provider_request_profile_uses_only_selected_adapter_effective_options() -> None:
    inactive_private_option = "inactive-anthropic-route-canary"
    app = CayuApp(
        secret_redactor=SecretRedactor([inactive_private_option]),
        enable_logging=False,
    )
    app.register_provider(OpenAIProvider(api_key="test-key"), default=True)
    option_sets = {
        "baseline": {
            "openai": {"metadata": {"route": "primary"}},
            "anthropic": {"metadata": {"route": "ignored-a"}},
        },
        "inactive_changed": {
            "openai": {"metadata": {"route": "primary"}},
            "anthropic": {"metadata": {"route": inactive_private_option}},
        },
        "active_changed": {
            "openai": {"metadata": {"route": "secondary"}},
            "anthropic": {"metadata": {"route": "ignored-a"}},
        },
    }
    for name, options in option_sets.items():
        app.register_agent(AgentSpec(name=name, model="openai-test", provider_options=options))

    components = {
        name: session_engine_module._execution_profile_identity(
            registered_agent=app._agents[name],
            provider_name="openai",
            registered_provider=app._providers["openai"],
            model="openai-test",
            durable_system_prompt=None,
            redactor=app._secret_redactor,
            process_identity=app._execution_profile_process_identity,
        ).component(ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY)
        for name in option_sets
    }
    serialized = json.dumps(
        [component.model_dump(mode="json") for component in components.values()],
        sort_keys=True,
    )

    assert components["baseline"] == components["inactive_changed"]
    assert components["baseline"] != components["active_changed"]
    assert inactive_private_option not in serialized


def test_effective_cache_policy_change_rejects_resume_before_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        requests: list[ModelRequest] = []

        async def stream(
            _provider: AnthropicProvider,
            request: ModelRequest,
        ) -> AsyncIterator[ModelStreamEvent]:
            requests.append(request)
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

        monkeypatch.setattr(AnthropicProvider, "stream", stream)
        store = InMemorySessionStore()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(
            AnthropicProvider(
                api_key="test-key",
                cache_policy=CachePolicy(
                    breakpoints=(CacheBreakpoint.SYSTEM_PROMPT,),
                    ttl="standard",
                ),
            ),
            default=True,
        )
        original_app.register_agent(
            AgentSpec(
                name="assistant",
                model="claude-test",
                provider_options={
                    "cache_policy": {
                        "breakpoints": [CacheBreakpoint.SYSTEM_PROMPT.value],
                        "ttl": "standard",
                    }
                },
            )
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-cache-policy",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        assert len(requests) == 1

        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(
            AnthropicProvider(
                api_key="test-key",
                cache_policy=CachePolicy(
                    breakpoints=(CacheBreakpoint.SYSTEM_PROMPT,),
                    ttl="standard",
                ),
            ),
            default=True,
        )
        replacement_app.register_agent(
            AgentSpec(
                name="assistant",
                model="claude-test",
                provider_options={
                    "cache_policy": {
                        "breakpoints": [CacheBreakpoint.SYSTEM_PROMPT.value],
                        "ttl": "extended",
                    }
                },
            )
        )

        with pytest.raises(ExecutionProfileMismatchError) as raised:
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-cache-policy",
                        messages=[Message.text("user", "second")],
                    )
                )
            )

        assert raised.value.changed_component_classes == (
            ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY,
        )
        assert len(requests) == 1

    asyncio.run(exercise())


def test_anthropic_cache_override_changes_effective_provider_request_profile() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(
        AnthropicProvider(
            api_key="test-key",
            cache_policy=CachePolicy(
                breakpoints=(CacheBreakpoint.SYSTEM_PROMPT,),
                ttl="standard",
            ),
        ),
        default=True,
    )
    for name, ttl in (("standard", "standard"), ("extended", "extended")):
        app.register_agent(
            AgentSpec(
                name=name,
                model="claude-test",
                provider_options={"cache_policy": {"ttl": ttl}},
            )
        )

    components = [
        session_engine_module._execution_profile_identity(
            registered_agent=app._agents[name],
            provider_name="anthropic",
            registered_provider=app._providers["anthropic"],
            model="claude-test",
            durable_system_prompt=None,
            redactor=app._secret_redactor,
            process_identity=app._execution_profile_process_identity,
        ).component(ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY)
        for name in ("standard", "extended")
    ]

    assert components[0].strength is ExecutionProfileIdentityStrength.STRUCTURAL
    assert components[1].strength is ExecutionProfileIdentityStrength.STRUCTURAL
    assert components[0] != components[1]


def test_custom_provider_cache_options_are_private_process_local_material() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(
        IdentityConfiguredCacheProvider(_test_behavior_identity("custom-cache-provider")),
        default=True,
    )
    for name, ttl in (("first", "standard"), ("second", "extended")):
        app.register_agent(
            AgentSpec(
                name=name,
                model="custom-cache-model",
                provider_options={
                    "cache_policy": {
                        "breakpoints": [CacheBreakpoint.SYSTEM_PROMPT.value],
                        "ttl": ttl,
                    }
                },
            )
        )

    components = [
        session_engine_module._execution_profile_identity(
            registered_agent=app._agents[name],
            provider_name="unversioned",
            registered_provider=app._providers["unversioned"],
            model="custom-cache-model",
            durable_system_prompt=None,
            redactor=app._secret_redactor,
            process_identity=app._execution_profile_process_identity,
        ).component(ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY)
        for name in ("first", "second")
    ]
    serialized = json.dumps(
        [component.model_dump(mode="json") for component in components],
        sort_keys=True,
    )

    assert components[0].strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
    assert components[1].strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
    assert components[0] != components[1]
    assert '"ttl": "standard"' not in serialized
    assert '"ttl": "extended"' not in serialized


def test_custom_provider_cache_policy_change_rejects_resume_before_dispatch() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        identity = _test_behavior_identity("custom-cache-provider")
        original_provider = IdentityConfiguredCacheProvider(identity)
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(original_provider, default=True)
        original_app.register_agent(
            AgentSpec(
                name="assistant",
                model="custom-cache-model",
                provider_options={
                    "cache_policy": {
                        "breakpoints": [CacheBreakpoint.SYSTEM_PROMPT.value],
                        "ttl": "standard",
                    }
                },
            )
        )

        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-custom-cache-policy",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        assert len(original_provider.requests) == 1

        replacement_provider = IdentityConfiguredCacheProvider(identity)
        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(replacement_provider, default=True)
        replacement_app.register_agent(
            AgentSpec(
                name="assistant",
                model="custom-cache-model",
                provider_options={
                    "cache_policy": {
                        "breakpoints": [CacheBreakpoint.SYSTEM_PROMPT.value],
                        "ttl": "extended",
                    }
                },
            )
        )

        with pytest.raises(ExecutionProfileMismatchError) as raised:
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-custom-cache-policy",
                        messages=[Message.text("user", "second")],
                    )
                )
            )

        assert raised.value.changed_component_classes == (
            ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY,
        )
        assert replacement_provider.requests == []

    asyncio.run(exercise())


def test_runtime_owned_private_context_commitments_survive_secret_collision() -> None:
    process_identity = "fixed-private-context-profile-process-key"
    policies = [
        KnowledgeInjectionPolicy(namespace=namespace, enabled=False)
        for namespace in ("private-namespace-alpha", "private-namespace-beta")
    ]
    commitment_secrets = []
    for policy in policies:
        projected = execution_profile_admission._cayu_context_policy_material(
            policy,
            behavior_identities={id(policy): None, id(policy.base_policy): None},
            process_identity=process_identity,
        )
        assert projected is not None
        commitment = projected.knowledge["configuration_hmac_sha256"]
        assert type(commitment) is str
        commitment_secrets.append(commitment)
    app = CayuApp(
        secret_redactor=SecretRedactor(commitment_secrets),
        enable_logging=False,
    )
    app.register_provider(_completed_provider(), default=True)
    for name, policy in zip(("first", "second"), policies, strict=True):
        app.register_agent(
            AgentSpec(name=name, model="fake-model"),
            context_policy=policy,
        )

    components = [
        session_engine_module._execution_profile_identity(
            registered_agent=app._agents[name],
            provider_name="fake",
            registered_provider=app._providers["fake"],
            model="fake-model",
            durable_system_prompt=None,
            redactor=app._secret_redactor,
            process_identity=process_identity,
        ).component(ExecutionProfileComponentClass.KNOWLEDGE_INJECTION)
        for name in ("first", "second")
    ]
    serialized = json.dumps(
        [component.model_dump(mode="json") for component in components],
        sort_keys=True,
    )

    assert components[0].strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
    assert components[0] != components[1]
    assert "private-namespace-alpha" not in serialized
    assert "private-namespace-beta" not in serialized
    assert process_identity not in serialized


def test_each_model_semantics_component_changes_independently() -> None:
    baseline = _model_semantics_profile(
        context_policy=KnowledgeInjectionPolicy(
            base_policy=CheckpointCompactionContextPolicy(
                compactor=TranscriptDigestCompactor(max_summary_chars=4096),
                max_user_turns=3,
                compact_after_messages=8,
            ),
            max_hits=3,
        ),
        provider_options={"fake": {"temperature": 0.25}},
        budget_policy=BudgetPolicy(limits=(_profile_budget_limit(scope="app"),)),
        request_budget_limits=(_profile_budget_limit(),),
        structured_output=StructuredOutputSpec(
            json_schema={"type": "object"},
            strategy="tool",
        ),
        max_steps=7,
    )
    variants = {
        ExecutionProfileComponentClass.CONTEXT_SELECTION: _model_semantics_profile(
            context_policy=MessageWindowContextPolicy(max_messages=4),
            provider_options={"fake": {"temperature": 0.25}},
            budget_policy=BudgetPolicy(limits=(_profile_budget_limit(scope="app"),)),
            request_budget_limits=(_profile_budget_limit(),),
            structured_output=StructuredOutputSpec(json_schema={"type": "object"}, strategy="tool"),
            max_steps=7,
        ),
        ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY: _model_semantics_profile(
            context_policy=KnowledgeInjectionPolicy(
                base_policy=CheckpointCompactionContextPolicy(
                    compactor=TranscriptDigestCompactor(max_summary_chars=4096),
                    max_user_turns=3,
                    compact_after_messages=8,
                ),
                max_hits=3,
            ),
            provider_options={"fake": {"temperature": 0.5}},
            budget_policy=BudgetPolicy(limits=(_profile_budget_limit(scope="app"),)),
            request_budget_limits=(_profile_budget_limit(),),
            structured_output=StructuredOutputSpec(json_schema={"type": "object"}, strategy="tool"),
            max_steps=7,
        ),
        ExecutionProfileComponentClass.PROVIDER_ADAPTER: _model_semantics_profile(
            context_policy=KnowledgeInjectionPolicy(
                base_policy=CheckpointCompactionContextPolicy(
                    compactor=TranscriptDigestCompactor(max_summary_chars=4096),
                    max_user_turns=3,
                    compact_after_messages=8,
                ),
                max_hits=3,
            ),
            provider=VersionedScriptedProvider("2"),
            provider_options={"fake": {"temperature": 0.25}},
            budget_policy=BudgetPolicy(limits=(_profile_budget_limit(scope="app"),)),
            request_budget_limits=(_profile_budget_limit(),),
            structured_output=StructuredOutputSpec(json_schema={"type": "object"}, strategy="tool"),
            max_steps=7,
        ),
        ExecutionProfileComponentClass.KNOWLEDGE_INJECTION: _model_semantics_profile(
            context_policy=KnowledgeInjectionPolicy(
                base_policy=CheckpointCompactionContextPolicy(
                    compactor=TranscriptDigestCompactor(max_summary_chars=4096),
                    max_user_turns=3,
                    compact_after_messages=8,
                ),
                max_hits=4,
            ),
            provider_options={"fake": {"temperature": 0.25}},
            budget_policy=BudgetPolicy(limits=(_profile_budget_limit(scope="app"),)),
            request_budget_limits=(_profile_budget_limit(),),
            structured_output=StructuredOutputSpec(json_schema={"type": "object"}, strategy="tool"),
            max_steps=7,
        ),
        ExecutionProfileComponentClass.CONTEXT_COMPACTION: _model_semantics_profile(
            context_policy=KnowledgeInjectionPolicy(
                base_policy=CheckpointCompactionContextPolicy(
                    compactor=TranscriptDigestCompactor(max_summary_chars=5000),
                    max_user_turns=3,
                    compact_after_messages=8,
                ),
                max_hits=3,
            ),
            provider_options={"fake": {"temperature": 0.25}},
            budget_policy=BudgetPolicy(limits=(_profile_budget_limit(scope="app"),)),
            request_budget_limits=(_profile_budget_limit(),),
            structured_output=StructuredOutputSpec(json_schema={"type": "object"}, strategy="tool"),
            max_steps=7,
        ),
        ExecutionProfileComponentClass.APPLICATION_BUDGET_POLICY: _model_semantics_profile(
            context_policy=KnowledgeInjectionPolicy(
                base_policy=CheckpointCompactionContextPolicy(
                    compactor=TranscriptDigestCompactor(max_summary_chars=4096),
                    max_user_turns=3,
                    compact_after_messages=8,
                ),
                max_hits=3,
            ),
            provider_options={"fake": {"temperature": 0.25}},
            budget_policy=BudgetPolicy(limits=(_profile_budget_limit(maximum="11", scope="app"),)),
            request_budget_limits=(_profile_budget_limit(),),
            structured_output=StructuredOutputSpec(json_schema={"type": "object"}, strategy="tool"),
            max_steps=7,
        ),
        ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY: _model_semantics_profile(
            context_policy=KnowledgeInjectionPolicy(
                base_policy=CheckpointCompactionContextPolicy(
                    compactor=TranscriptDigestCompactor(max_summary_chars=4096),
                    max_user_turns=3,
                    compact_after_messages=8,
                ),
                max_hits=3,
            ),
            provider_options={"fake": {"temperature": 0.25}},
            budget_policy=BudgetPolicy(limits=(_profile_budget_limit(scope="app"),)),
            request_budget_limits=(_profile_budget_limit(maximum="11"),),
            structured_output=StructuredOutputSpec(json_schema={"type": "object"}, strategy="tool"),
            max_steps=7,
        ),
        ExecutionProfileComponentClass.STRUCTURED_OUTPUT: _model_semantics_profile(
            context_policy=KnowledgeInjectionPolicy(
                base_policy=CheckpointCompactionContextPolicy(
                    compactor=TranscriptDigestCompactor(max_summary_chars=4096),
                    max_user_turns=3,
                    compact_after_messages=8,
                ),
                max_hits=3,
            ),
            provider_options={"fake": {"temperature": 0.25}},
            budget_policy=BudgetPolicy(limits=(_profile_budget_limit(scope="app"),)),
            request_budget_limits=(_profile_budget_limit(),),
            structured_output=StructuredOutputSpec(json_schema={"type": "array"}, strategy="tool"),
            max_steps=7,
        ),
        ExecutionProfileComponentClass.FINALIZATION: _model_semantics_profile(
            context_policy=KnowledgeInjectionPolicy(
                base_policy=CheckpointCompactionContextPolicy(
                    compactor=TranscriptDigestCompactor(max_summary_chars=4096),
                    max_user_turns=3,
                    compact_after_messages=8,
                ),
                max_hits=3,
            ),
            provider_options={"fake": {"temperature": 0.25}},
            budget_policy=BudgetPolicy(limits=(_profile_budget_limit(scope="app"),)),
            request_budget_limits=(_profile_budget_limit(),),
            structured_output=StructuredOutputSpec(json_schema={"type": "object"}, strategy="tool"),
            max_steps=8,
        ),
    }

    for component_class, variant in variants.items():
        assert baseline.component(component_class) != variant.component(component_class)


def test_live_state_profile_component_records_explicit_absence_and_future_change() -> None:
    baseline = build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version="1",
        provider_name="fake",
        model="fake-model",
        durable_system_prompt=None,
        direct_tools=[],
    )
    projected = build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version="1",
        provider_name="fake",
        model="fake-model",
        durable_system_prompt=None,
        direct_tools=[],
        live_state_projection={
            "kind": "runtime-state-snapshot",
            "version": 1,
            "policy": "bounded",
        },
    )

    component_class = ExecutionProfileComponentClass.LIVE_STATE_PROJECTION
    assert baseline.component(component_class).availability is (
        ExecutionProfileIdentityAvailability.AVAILABLE
    )
    assert baseline.component(component_class) != projected.component(component_class)
    assert changed_execution_profile_components(baseline, projected) == (component_class,)


def test_cayu_profile_material_extractors_require_exact_registered_types(tmp_path: Path) -> None:
    class DerivedSearchTextTool(SearchTextTool):
        pass

    class DerivedLocalRunner(LocalRunner):
        pass

    class DerivedStaticToolPolicy(StaticToolPolicy):
        pass

    assert execution_profile_admission._cayu_tool_material(SearchTextTool()) is not None
    assert execution_profile_admission._cayu_tool_material(DerivedSearchTextTool()) is None
    assert execution_profile_admission._cayu_runner_material(LocalRunner(tmp_path)) is not None
    assert execution_profile_admission._cayu_runner_material(DerivedLocalRunner(tmp_path)) is None
    assert (
        execution_profile_admission._cayu_policy_material(StaticToolPolicy(allow=("search_text",)))
        is not None
    )
    assert (
        execution_profile_admission._cayu_policy_material(
            DerivedStaticToolPolicy(allow=("search_text",))
        )
        is None
    )


@pytest.mark.parametrize(
    "registration_kind",
    [
        "app_runtime_hook",
        "app_loop_policy",
        "tool",
        "command_policy",
        "tool_policy",
        "agent_runtime_hook",
        "agent_loop_policy",
        "environment_spec",
        "environment_runner",
        "factory_spec",
        "environment_factory",
        "provider",
        "context_policy",
        "context_compactor",
        "model_compactor_provider",
        "prompt_cache_fallback",
    ],
)
def test_app_rejects_workload_secrets_in_behavior_identity_registrations(
    registration_kind: str,
    tmp_path: Path,
) -> None:
    secret = "execution-profile-identity-secret-canary"
    identity = _test_behavior_identity(
        registration_kind,
        behavior_version=f"release:{secret}",
    )
    redactor = SecretRedactor(secret)

    if registration_kind == "app_runtime_hook":
        with pytest.raises(ValueError, match="configured workload secret") as caught:
            CayuApp(
                runtime_hooks=(IdentityConfiguredHook("secret-hook", identity),),
                secret_redactor=redactor,
                enable_logging=False,
            )
    elif registration_kind == "app_loop_policy":
        with pytest.raises(ValueError, match="configured workload secret") as caught:
            CayuApp(
                loop_policies=(ConfiguredAdoptionLoopPolicy(f"release:{secret}"),),
                secret_redactor=redactor,
                enable_logging=False,
            )
    else:
        app = CayuApp(secret_redactor=redactor, enable_logging=False)
        with pytest.raises(ValueError, match="configured workload secret") as caught:
            if registration_kind == "tool":
                app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    tools=[IdentityConfiguredTool(identity)],
                )
            elif registration_kind == "command_policy":
                app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    tools=[ExecCommandTool(policy=IdentityConfiguredCommandPolicy(identity))],
                )
            elif registration_kind == "tool_policy":
                app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    tools=[IdentityConfiguredTool(_test_behavior_identity("safe-tool"))],
                    tool_policy=IdentityConfiguredToolPolicy(identity),
                )
            elif registration_kind == "agent_runtime_hook":
                app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    runtime_hooks=(IdentityConfiguredHook("secret-hook", identity),),
                )
            elif registration_kind == "agent_loop_policy":
                app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    loop_policies=(ConfiguredAdoptionLoopPolicy(f"release:{secret}"),),
                )
            elif registration_kind == "environment_spec":
                app.register_environment(
                    Environment(
                        EnvironmentSpec(
                            name="environment",
                            execution_profile_identity=identity,
                        )
                    )
                )
            elif registration_kind == "environment_runner":
                app.register_environment(
                    Environment(
                        EnvironmentSpec(name="environment"),
                        runner=IdentityConfiguredRunner(tmp_path, identity),
                    )
                )
            elif registration_kind == "provider":
                app.register_provider(IdentityConfiguredProvider(identity))
            elif registration_kind == "context_policy":
                app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    context_policy=IdentityConfiguredContextPolicy(identity),
                )
            elif registration_kind == "context_compactor":
                app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    context_policy=CheckpointCompactionContextPolicy(
                        compactor=IdentityConfiguredContextCompactor(identity),
                    ),
                )
            elif registration_kind == "model_compactor_provider":
                app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    context_policy=CheckpointCompactionContextPolicy(
                        compactor=ModelCompactor(
                            provider=IdentityConfiguredProvider(identity),
                            model="summary-model",
                        ),
                    ),
                )
            elif registration_kind == "prompt_cache_fallback":
                app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"),
                    context_policy=CheckpointCompactionContextPolicy(
                        compactor=PromptCacheCompactor(
                            provider=ScriptedModelProvider([], name="cache-provider"),
                            fallback_compactor=IdentityConfiguredContextCompactor(identity),
                        ),
                    ),
                )
            elif registration_kind == "factory_spec":
                app.register_environment_factory(
                    EnvironmentSpec(
                        name="environment",
                        execution_profile_identity=identity,
                    ),
                    IdentityConfiguredEnvironmentFactory(),
                )
            else:
                assert registration_kind == "environment_factory"
                app.register_environment_factory(
                    EnvironmentSpec(name="environment"),
                    IdentityConfiguredEnvironmentFactory(identity),
                )

        assert app.list_agents() == ()
        assert app.list_environments() == ()
        assert app.list_providers() == ()

    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)


def test_public_agent_registration_does_not_serialize_a_mutated_behavior_identity(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "execution-profile-registration-serializer-secret-canary-ABCDEFGHIJKLMNOP"

    class SecretBearingValue:
        def __repr__(self) -> str:
            return secret

    tool = IdentityConfiguredTool(_test_behavior_identity("mutated-registration"))
    identity = tool.spec.execution_profile_identity
    assert type(identity) is ExecutionProfileBehaviorIdentity
    object.__setattr__(identity, "behavior_version", SecretBearingValue())
    app = CayuApp(
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )

    with (
        warnings.catch_warnings(record=True) as emitted,
        caplog.at_level(logging.WARNING),
        pytest.raises(ValidationError) as raised,
    ):
        warnings.simplefilter("always")
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )

    exception_chain: list[BaseException] = []
    current: BaseException | None = raised.value
    while current is not None and all(current is not item for item in exception_chain):
        exception_chain.append(current)
        current = current.__cause__ or current.__context__
    captured = capsys.readouterr()
    diagnostic_output = " ".join(
        (
            *(value for error in exception_chain for value in (str(error), repr(error))),
            captured.out,
            captured.err,
            *(record.getMessage() for record in caplog.records),
            *(str(record.message) for record in emitted),
        )
    )
    assert emitted == []
    assert secret not in diagnostic_output
    assert app.list_agents() == ()


def test_request_loop_policy_rejects_workload_secret_identity_before_session_creation() -> None:
    async def exercise() -> None:
        secret = "execution-profile-request-identity-secret-canary"
        store = InMemorySessionStore()
        provider = _completed_provider()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(ValueError, match="configured workload secret") as caught:
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="secret-request-profile-identity",
                        messages=[Message.text("user", "first")],
                        loop_policies=(ConfiguredAdoptionLoopPolicy(f"release:{secret}"),),
                    )
                )
            )

        assert secret not in str(caught.value)
        assert secret not in repr(caught.value)
        assert await store.load("secret-request-profile-identity") is None
        assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("material_kind", "component_class"),
    [
        ("tool", ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS),
        ("policy", ExecutionProfileComponentClass.EXECUTION_POLICIES),
        ("runner", ExecutionProfileComponentClass.EXECUTION_ENVIRONMENT),
    ],
)
def test_redactor_known_builtin_material_is_process_local(
    material_kind: str,
    component_class: ExecutionProfileComponentClass,
    tmp_path: Path,
) -> None:
    secret = "execution-profile-material-secret-canary"
    runner_root = tmp_path / secret
    runner_root.mkdir()

    def configured_app() -> CayuApp:
        app = CayuApp(
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        tools: list[Tool] = []
        tool_policy: ToolPolicy | None = None
        if material_kind == "tool":
            tools.append(SearchTextTool(exclude_directories=(secret,)))
        elif material_kind == "policy":
            tool_policy = TaintAwareToolPolicy(
                taint_sources={"read_email": (secret,)},
                protected_tools={"send_email": (secret,)},
            )
        else:
            assert material_kind == "runner"
            app.register_environment(
                Environment(
                    EnvironmentSpec(
                        name="sandbox",
                        execution_profile_identity=_test_behavior_identity("sandbox"),
                    ),
                    runner=LocalRunner(runner_root),
                ),
                default=True,
            )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=tools,
            tool_policy=tool_policy,
        )
        return app

    fingerprints: list[str | None] = []
    for _ in range(2):
        app = configured_app()
        profile = session_engine_module._execution_profile_identity(
            registered_agent=app._agents["assistant"],
            provider_name="fake",
            model="fake-model",
            durable_system_prompt=None,
            redactor=app._secret_redactor,
            registered_environment=app._get_registered_environment(None),
            process_identity=app._execution_profile_process_identity,
        )
        component = profile.component(component_class)
        assert component.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
        fingerprints.append(component.fingerprint)

    assert fingerprints[0] != fingerprints[1]


def test_remember_knowledge_application_policy_is_process_local_and_secret_safe() -> None:
    private_scope = "tenant-profile-scope-canary"
    fingerprints: list[str | None] = []

    for _ in range(2):
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[
                RememberKnowledgeTool(
                    policy=RememberKnowledgePolicy(
                        default_namespace=private_scope,
                        require_labels={"tenant": private_scope},
                    )
                )
            ],
        )
        profile = session_engine_module._execution_profile_identity(
            registered_agent=app._agents["assistant"],
            provider_name="fake",
            model="fake-model",
            durable_system_prompt=None,
            redactor=app._secret_redactor,
            registered_environment=app._get_registered_environment(None),
            process_identity=app._execution_profile_process_identity,
        )
        component = profile.component(ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS)

        assert component.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
        assert private_scope not in json.dumps(component.model_dump(mode="json"), sort_keys=True)
        fingerprints.append(component.fingerprint)

    assert fingerprints[0] != fingerprints[1]


def test_remember_knowledge_default_policy_retains_structural_identity() -> None:
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[RememberKnowledgeTool()],
    )

    profile = session_engine_module._execution_profile_identity(
        registered_agent=app._agents["assistant"],
        provider_name="fake",
        model="fake-model",
        durable_system_prompt=None,
        redactor=app._secret_redactor,
        registered_environment=app._get_registered_environment(None),
        process_identity=app._execution_profile_process_identity,
    )

    assert (
        profile.component(ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS).strength
        is ExecutionProfileIdentityStrength.STRUCTURAL
    )


def test_declared_remember_knowledge_policy_identity_is_portable() -> None:
    identity = _test_behavior_identity("remember-knowledge-policy")
    component_fingerprints: list[str | None] = []

    for _ in range(2):
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[
                RememberKnowledgeTool(
                    spec=RememberKnowledgeTool.spec.model_copy(
                        update={"execution_profile_identity": identity},
                        deep=True,
                    ),
                    policy=RememberKnowledgePolicy(
                        default_namespace="tenant-knowledge",
                        require_labels={"tenant": "acme"},
                    ),
                )
            ],
        )
        profile = session_engine_module._execution_profile_identity(
            registered_agent=app._agents["assistant"],
            provider_name="fake",
            model="fake-model",
            durable_system_prompt=None,
            redactor=app._secret_redactor,
            registered_environment=app._get_registered_environment(None),
            process_identity=app._execution_profile_process_identity,
        )
        component = profile.component(ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS)

        assert component.strength is ExecutionProfileIdentityStrength.APPLICATION_VERSIONED
        component_fingerprints.append(component.fingerprint)

    assert component_fingerprints[0] == component_fingerprints[1]


def test_unversioned_custom_tool_identity_is_scoped_to_one_app_instance() -> None:
    async def exercise() -> None:
        session_id = "execution-profile-process-local-tool"
        store = InMemorySessionStore()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[IdentityConfiguredTool(None, opaque_behavior="restricted")],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )
        original = await store.load(session_id)
        assert original is not None
        profile = execution_profile_from_session_metadata(original.metadata)
        implementation = profile.component(ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS)
        assert implementation.availability is ExecutionProfileIdentityAvailability.AVAILABLE
        assert implementation.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL

        await _collect(
            original_app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "same app")],
                )
            )
        )

        replacement_app = CayuApp(session_store=store, enable_logging=False)
        restarted_provider = _completed_provider()
        replacement_app.register_provider(restarted_provider, default=True)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[IdentityConfiguredTool(None, opaque_behavior="expanded")],
        )
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "replacement app")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
        )
        assert restarted_provider.requests == []

    asyncio.run(exercise())


def test_unversioned_provider_identity_is_scoped_to_one_app_instance() -> None:
    async def exercise() -> None:
        session_id = "execution-profile-process-local-provider"
        store = InMemorySessionStore()
        provider = UnversionedProvider()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(provider, default=True)
        original_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )

        original = await store.load(session_id)
        assert original is not None
        adapter = execution_profile_from_session_metadata(original.metadata).component(
            ExecutionProfileComponentClass.PROVIDER_ADAPTER
        )
        assert adapter.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL

        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(provider, default=True)
        replacement_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "second")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.PROVIDER_ADAPTER,
        )
        assert len(provider.requests) == 1

    asyncio.run(exercise())


def test_process_local_identity_detects_a_different_custom_type_in_the_same_process() -> None:
    async def exercise() -> None:
        session_id = "execution-profile-process-local-tool-type"
        store = InMemorySessionStore()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[IdentityConfiguredTool(None)],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )

        replacement_provider = _completed_provider()
        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(replacement_provider, default=True)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[AlternateIdentityConfiguredTool(None)],
        )
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "different opaque implementation")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
        )
        assert replacement_provider.requests == []

    asyncio.run(exercise())


def test_cayu_tool_with_opaque_adapter_is_process_local_without_declared_identity() -> None:
    async def exercise() -> None:
        session_id = "execution-profile-opaque-built-in-tool-adapter"
        store = InMemorySessionStore()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WebFetchTool(adapter=OpaqueWebFetchAdapter())],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )

        original = await store.load(session_id)
        assert original is not None
        implementation = execution_profile_from_session_metadata(original.metadata).component(
            ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS
        )
        assert implementation.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL

        restarted_provider = _completed_provider()
        restarted_app = CayuApp(session_store=store, enable_logging=False)
        restarted_app.register_provider(restarted_provider, default=True)
        restarted_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WebFetchTool(adapter=OpaqueWebFetchAdapter())],
        )

        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                restarted_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "foreign process")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
        )
        assert restarted_provider.requests == []

    asyncio.run(exercise())


def test_declared_identity_makes_opaque_built_in_tool_adapter_portable() -> None:
    async def exercise() -> None:
        session_id = "execution-profile-versioned-built-in-tool-adapter"
        store = InMemorySessionStore()
        identity = _test_behavior_identity("opaque-web-fetch-adapter")

        def configured_app() -> CayuApp:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_completed_provider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[
                    WebFetchTool(
                        adapter=OpaqueWebFetchAdapter(),
                        spec=WebFetchTool.spec.model_copy(
                            update={"execution_profile_identity": identity}
                        ),
                    )
                ],
            )
            return app

        await _collect(
            configured_app().run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )
        events = await _collect(
            configured_app().resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "portable restart")],
                )
            )
        )

        assert events[0].type is EventType.INTERACTION_STARTED

    asyncio.run(exercise())


def test_browser_adapter_dom_limit_changes_implementation_profile_before_work() -> None:
    async def exercise() -> None:
        session_id = "execution-profile-browser-dom-limit"
        store = InMemorySessionStore()

        def configured_app(*, max_dom_nodes: int) -> tuple[CayuApp, ScriptedModelProvider]:
            provider = _completed_provider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[WebFetchTool(adapter=BrowserWebFetchAdapter(max_dom_nodes=max_dom_nodes))],
            )
            return app, provider

        original_app, _original_provider = configured_app(max_dom_nodes=100)
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )

        restarted_app, _restarted_provider = configured_app(max_dom_nodes=100)
        resumed = await _collect(
            restarted_app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "same browser limit")],
                )
            )
        )
        assert resumed[0].type is EventType.INTERACTION_STARTED

        changed_app, changed_provider = configured_app(max_dom_nodes=101)
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                changed_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "changed browser limit")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
        )
        assert changed_provider.requests == []

    asyncio.run(exercise())


def test_custom_browser_worker_is_app_local_without_declared_identity() -> None:
    async def exercise() -> None:
        session_id = "execution-profile-custom-browser-worker"
        store = InMemorySessionStore()

        def configured_app() -> tuple[CayuApp, ScriptedModelProvider]:
            provider = _completed_provider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[
                    WebFetchTool(
                        adapter=BrowserWebFetchAdapter(
                            worker_command=("python", "-m", "application_browser_worker")
                        )
                    )
                ],
            )
            return app, provider

        original_app, _original_provider = configured_app()
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )
        session = await store.load(session_id)
        assert session is not None
        tool_component = execution_profile_from_session_metadata(session.metadata).component(
            ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS
        )
        assert tool_component.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL

        restarted_app, restarted_provider = configured_app()
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                restarted_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "replacement app")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
        )
        assert restarted_provider.requests == []

    asyncio.run(exercise())


def test_cayu_tool_configuration_changes_implementation_profile_before_work() -> None:
    async def exercise() -> None:
        session_id = "execution-profile-built-in-tool-configuration"
        store = InMemorySessionStore()

        def configured_app(*, max_preview_bytes: int) -> tuple[CayuApp, ScriptedModelProvider]:
            provider = _completed_provider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[SearchTextTool(max_preview_bytes=max_preview_bytes)],
            )
            return app, provider

        original_app, _original_provider = configured_app(max_preview_bytes=500)
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )

        changed_app, changed_provider = configured_app(max_preview_bytes=501)
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                changed_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "changed built-in configuration")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
        )
        assert changed_provider.requests == []

    asyncio.run(exercise())


def test_versioned_custom_tool_identity_is_portable_and_detects_implementation_drift() -> None:
    async def exercise() -> None:
        session_id = "execution-profile-versioned-tool"
        store = InMemorySessionStore()
        stable_identity = _test_behavior_identity("portable-tool")
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[IdentityConfiguredTool(stable_identity)],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )

        restarted_app = CayuApp(session_store=store, enable_logging=False)
        restarted_app.register_provider(_completed_provider(), default=True)
        restarted_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[IdentityConfiguredTool(stable_identity)],
        )
        await _collect(
            restarted_app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "portable restart")],
                )
            )
        )

        changed_app = CayuApp(session_store=store, enable_logging=False)
        changed_provider = _completed_provider()
        changed_app.register_provider(changed_provider, default=True)
        changed_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[
                IdentityConfiguredTool(
                    _test_behavior_identity(
                        "portable-tool",
                        implementation_version="2",
                    )
                )
            ],
        )
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                changed_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "changed implementation")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
        )
        assert changed_provider.requests == []

    asyncio.run(exercise())


def test_public_resume_rejects_changed_tool_schema_before_work() -> None:
    async def exercise() -> None:
        session_id = "execution-profile-tool-schema"
        store = InMemorySessionStore()
        identity = _test_behavior_identity("schema-tool")
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[IdentityConfiguredTool(identity)],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )

        replacement_provider = _completed_provider()
        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(replacement_provider, default=True)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[
                IdentityConfiguredTool(
                    identity,
                    input_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                )
            ],
        )

        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "changed schema")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.DIRECT_TOOLS,
        )
        assert replacement_provider.requests == []

    asyncio.run(exercise())


def test_public_resume_rejects_changed_environment_identity_before_work() -> None:
    async def exercise() -> None:
        session_id = "execution-profile-environment-identity"
        store = InMemorySessionStore()

        def configured_app(implementation_version: str) -> tuple[CayuApp, ScriptedModelProvider]:
            provider = _completed_provider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_environment(
                Environment(
                    EnvironmentSpec(
                        name="sandbox",
                        execution_profile_identity=_test_behavior_identity(
                            "sandbox",
                            implementation_version=implementation_version,
                        ),
                    )
                ),
                default=True,
            )
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            return app, provider

        original_app, _original_provider = configured_app("1")
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )

        replacement_app, replacement_provider = configured_app("2")
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "changed environment")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.EXECUTION_ENVIRONMENT,
        )
        assert replacement_provider.requests == []

    asyncio.run(exercise())


def test_cayu_runner_configuration_is_portable_and_detects_drift(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        session_id = "execution-profile-built-in-runner-configuration"
        store = InMemorySessionStore()
        original_root = tmp_path / "original"
        changed_root = tmp_path / "changed"
        original_root.mkdir()
        changed_root.mkdir()
        environment_identity = _test_behavior_identity("local-environment")

        def configured_app(root: Path) -> tuple[CayuApp, ScriptedModelProvider]:
            provider = _completed_provider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_environment(
                Environment(
                    EnvironmentSpec(
                        name="local",
                        execution_profile_identity=environment_identity,
                    ),
                    runner=LocalRunner(root),
                ),
                default=True,
            )
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            return app, provider

        original_app, _original_provider = configured_app(original_root)
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )

        restarted_app, _restarted_provider = configured_app(original_root)
        restarted = await _collect(
            restarted_app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "portable restart")],
                )
            )
        )
        assert restarted[0].type is EventType.INTERACTION_STARTED

        changed_app, changed_provider = configured_app(changed_root)
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                changed_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "changed runner root")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.EXECUTION_ENVIRONMENT,
        )
        assert changed_provider.requests == []

    asyncio.run(exercise())


def test_docker_environment_grants_are_process_local_profile_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        canary = "docker-profile-overlay-secret-canary"
        monkeypatch.setattr(
            "cayu.runners.docker._require_docker",
            lambda path: path or "/test/docker",
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_completed_provider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(
                    name="docker",
                    execution_profile_identity=_test_behavior_identity("docker-environment"),
                ),
                runner=DockerRunner(
                    "profile-container",
                    docker_path="/test/docker",
                    env_overlay={"PRIVATE_TOKEN": canary},
                ),
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-docker-environment-grant",
                    messages=[Message.text("user", "run")],
                )
            )
        )
        session = await store.load("execution-profile-docker-environment-grant")
        assert session is not None
        component = execution_profile_from_session_metadata(session.metadata).component(
            ExecutionProfileComponentClass.EXECUTION_ENVIRONMENT
        )
        assert component.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
        assert canary not in json.dumps(session.metadata, sort_keys=True)

    asyncio.run(exercise())


def test_factory_managed_effect_authority_is_not_claimed_as_absent() -> None:
    async def profile_for(*, factory_backed: bool) -> ExecutionProfileIdentity:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_completed_provider(), default=True)
        spec = EnvironmentSpec(
            name="managed",
            execution_profile_identity=_test_behavior_identity("managed-environment"),
        )
        if factory_backed:
            app.register_environment_factory(
                spec,
                IdentityConfiguredEnvironmentFactory(),
                default=True,
            )
        else:
            app.register_environment(Environment(spec), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"factory-authority-{factory_backed}",
                    messages=[Message.text("user", "run")],
                )
            )
        )
        session = await store.load(f"factory-authority-{factory_backed}")
        assert session is not None
        return execution_profile_from_session_metadata(session.metadata)

    static_profile = asyncio.run(profile_for(factory_backed=False))
    factory_profile = asyncio.run(profile_for(factory_backed=True))

    assert static_profile.component(
        ExecutionProfileComponentClass.EFFECT_AUTHORITY
    ) != factory_profile.component(ExecutionProfileComponentClass.EFFECT_AUTHORITY)


def test_policy_and_ordered_hook_identities_are_independent_profile_components() -> None:
    async def changed_components(
        *,
        original_policy_version: str = "1",
        replacement_policy_version: str = "1",
        original_hook_order: tuple[str, ...] = ("first", "second"),
        replacement_hook_order: tuple[str, ...] = ("first", "second"),
    ) -> tuple[ExecutionProfileComponentClass, ...]:
        session_id = f"execution-profile-policy-hooks-{original_policy_version}-{uuid4().hex}"
        store = InMemorySessionStore()

        def configured_app(*, policy_version: str, hook_order: tuple[str, ...]) -> CayuApp:
            app = CayuApp(
                session_store=store,
                runtime_hooks=tuple(IdentityConfiguredHook(name) for name in hook_order),
                enable_logging=False,
            )
            app.register_provider(_completed_provider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[IdentityConfiguredTool(_test_behavior_identity("portable-tool"))],
                tool_policy=IdentityConfiguredToolPolicy(
                    _test_behavior_identity(
                        "portable-tool-policy",
                        behavior_version=policy_version,
                    )
                ),
            )
            return app

        original_app = configured_app(
            policy_version=original_policy_version,
            hook_order=original_hook_order,
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )
        replacement_app = configured_app(
            policy_version=replacement_policy_version,
            hook_order=replacement_hook_order,
        )
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "changed authority")],
                    )
                )
            )
        return caught.value.changed_component_classes

    policy_change = asyncio.run(
        changed_components(original_policy_version="1", replacement_policy_version="2")
    )
    hook_order_change = asyncio.run(changed_components(replacement_hook_order=("second", "first")))

    assert policy_change == (ExecutionProfileComponentClass.EXECUTION_POLICIES,)
    assert hook_order_change == (ExecutionProfileComponentClass.RUNTIME_HOOKS,)


@pytest.mark.parametrize("rule_kind", ["allowlist", "deny_pattern"])
def test_parameter_policy_with_exact_values_is_app_local(rule_kind: str) -> None:
    async def exercise() -> None:
        canary = "execution-profile-parameter-policy-secret-canary"
        store = InMemorySessionStore()
        session_id = f"execution-profile-parameter-policy-{rule_kind}"

        rule = (
            RequiredAllowlistRule("target", values=(canary,))
            if rule_kind == "allowlist"
            else DenyPatternRule("target", patterns=(canary,))
        )

        def configured_app() -> tuple[CayuApp, ScriptedModelProvider]:
            provider = _completed_provider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tool_policy=ParameterConstrainedToolPolicy({"identity_configured_tool": (rule,)}),
                tools=[IdentityConfiguredTool(_test_behavior_identity("parameter-tool"))],
            )
            return app, provider

        original_app, _original_provider = configured_app()
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )
        session = await store.load(session_id)
        assert session is not None
        policy_component = execution_profile_from_session_metadata(session.metadata).component(
            ExecutionProfileComponentClass.EXECUTION_POLICIES
        )
        assert policy_component.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
        assert canary not in json.dumps(session.metadata, sort_keys=True)

        restarted_app, restarted_provider = configured_app()
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                restarted_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "replacement app")],
                    )
                )
            )
        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.EXECUTION_POLICIES,
        )
        assert restarted_provider.requests == []

    asyncio.run(exercise())


def test_structural_parameter_policy_is_portable_and_detects_rule_drift() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        session_id = "execution-profile-structural-parameter-policy"

        def configured_app(*, parameter: str) -> tuple[CayuApp, ScriptedModelProvider]:
            provider = _completed_provider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tool_policy=ParameterConstrainedToolPolicy(
                    {"identity_configured_tool": (RequiredFieldRule(parameter),)}
                ),
                tools=[IdentityConfiguredTool(_test_behavior_identity("parameter-tool"))],
            )
            return app, provider

        original_app, _original_provider = configured_app(parameter="target")
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )

        restarted_app, _restarted_provider = configured_app(parameter="target")
        resumed = await _collect(
            restarted_app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "portable policy")],
                )
            )
        )
        assert resumed[0].type is EventType.INTERACTION_STARTED

        changed_app, changed_provider = configured_app(parameter="other_target")
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                changed_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "changed policy")],
                    )
                )
            )
        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.EXECUTION_POLICIES,
        )
        assert changed_provider.requests == []

    asyncio.run(exercise())


def test_process_command_policy_with_exact_environment_values_is_app_local() -> None:
    async def exercise() -> None:
        canary = "execution-profile-policy-secret-canary"
        store = InMemorySessionStore()

        def configured_app() -> tuple[CayuApp, ScriptedModelProvider]:
            provider = _completed_provider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[
                    ExecCommandTool(
                        policy=ProcessCommandPolicy(
                            allowed_executables={"git"},
                            allowed_cwds={"/workspace"},
                            allowed_env_values={"TOKEN": canary},
                        )
                    )
                ],
            )
            return app, provider

        original_app, _original_provider = configured_app()
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-secret-command-policy",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        session = await store.load("execution-profile-secret-command-policy")
        assert session is not None
        policy_component = execution_profile_from_session_metadata(session.metadata).component(
            ExecutionProfileComponentClass.EXECUTION_POLICIES
        )
        assert policy_component.strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
        assert canary not in json.dumps(session.metadata, sort_keys=True)

        replacement_app, replacement_provider = configured_app()
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-secret-command-policy",
                        messages=[Message.text("user", "replacement app")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.EXECUTION_POLICIES,
        )
        assert replacement_provider.requests == []

    asyncio.run(exercise())


def test_public_resume_rejects_changed_direct_tools_before_work() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        original_provider = _completed_provider()
        original_tool = RecordingTool("original_tool")
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(original_provider, default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[original_tool],
        )

        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-tool-drift",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        before = await store.load("execution-profile-tool-drift")
        assert before is not None

        replacement_provider = _completed_provider()
        replacement_tool = RecordingTool("replacement_tool")
        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(replacement_provider, default=True)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[replacement_tool],
        )

        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-tool-drift",
                        messages=[Message.text("user", "second")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.DIRECT_TOOLS,
            ExecutionProfileComponentClass.EFFECT_AUTHORITY,
            ExecutionProfileComponentClass.TOOL_VIEW_GRANTS,
        )
        assert replacement_provider.requests == []
        assert replacement_tool.calls == []

        with pytest.raises(ExecutionProfileMismatchError):
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-tool-drift",
                        messages=[Message.text("user", "retry")],
                    )
                )
            )

        after = await store.load("execution-profile-tool-drift")
        assert after is not None
        assert after.status == before.status
        assert after.run_epoch == before.run_epoch

        events = await store.load_events("execution-profile-tool-drift")
        rejection = events[-1]
        assert rejection.type == EventType.SESSION_EXECUTION_PROFILE_REJECTED
        assert rejection.payload["changed_component_classes"] == [
            "direct_tools",
            "effect_authority",
            "tool_view_grants",
        ]
        assert set(rejection.payload) == {
            "actor",
            "authority_decision",
            "candidate_profile",
            "changed_component_classes",
            "decision",
            "expected_profile",
            "idempotency_identity",
            "policy_identity",
            "policy_reason",
            "reason",
        }
        assert (
            sum(event.type is EventType.SESSION_EXECUTION_PROFILE_REJECTED for event in events) == 1
        )

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("original_workspace_mutation", "replacement_workspace_mutation"),
    [(False, True), (True, False)],
)
def test_public_resume_rejects_changed_workspace_mutation_declaration_before_work(
    original_workspace_mutation: bool,
    replacement_workspace_mutation: bool,
) -> None:
    async def exercise() -> None:
        session_id = (
            "execution-profile-workspace-mutation-"
            f"{int(original_workspace_mutation)}-{int(replacement_workspace_mutation)}"
        )
        store = InMemorySessionStore()
        original_provider = _completed_provider()
        original_tool = RecordingTool(
            "workspace_tool",
            parallel_safe=False,
            workspace_mutation=original_workspace_mutation,
        )
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(original_provider, default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[original_tool],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )

        replacement_provider = _completed_provider()
        replacement_tool = RecordingTool(
            "workspace_tool",
            parallel_safe=False,
            workspace_mutation=replacement_workspace_mutation,
        )
        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(replacement_provider, default=True)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[replacement_tool],
        )

        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "second")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.DIRECT_TOOLS,
            ExecutionProfileComponentClass.EFFECT_AUTHORITY,
        )
        assert replacement_provider.requests == []
        assert replacement_tool.calls == []

    asyncio.run(exercise())


def test_default_false_workspace_mutation_keeps_direct_tool_component_shape() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_completed_provider(), default=True)
        tool = RecordingTool("legacy_tool", parallel_safe=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-legacy-workspace-mutation-false",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        session = await store.load("execution-profile-legacy-workspace-mutation-false")
        assert session is not None
        return session

    session = asyncio.run(exercise())
    stored = execution_profile_from_session_metadata(session.metadata)
    legacy = build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version=session.runtime_version,
        provider_name="fake",
        model="fake-model",
        durable_system_prompt=None,
        direct_tools=[
            {
                "name": "legacy_tool",
                "description": "Record execution.",
                "schema": {"type": "object", "properties": {}},
                "parallel_safe": False,
                "effect": "external",
            }
        ],
    )

    assert stored.component(ExecutionProfileComponentClass.DIRECT_TOOLS) == legacy.component(
        ExecutionProfileComponentClass.DIRECT_TOOLS
    )


def test_explicit_authorized_tool_profile_adoption_is_atomic_and_replayable() -> None:
    async def exercise() -> None:
        secret = "profile-adoption-audit-secret"
        store = InMemorySessionStore()
        original_app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("original_tool")],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-explicit-adoption",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        before = await store.load("execution-profile-explicit-adoption")
        assert before is not None
        baseline = before.metadata[EXECUTION_PROFILE_METADATA_KEY]["baseline"]

        policy = RecordingExecutionProfilePolicy(
            ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason=f"Deployment policy approved {secret} tool profile.",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            )
        )
        provider = _completed_provider()
        app = CayuApp(
            session_store=store,
            execution_profile_policy=policy,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("replacement_tool")],
        )
        intent = ExecutionProfileAdoptionIntent(
            idempotency_key="deploy-2026-08-13",
            reason=f"Adopt the reviewed {secret} deployment profile.",
            requested_by=ResolutionActor(
                subject=f"maintainer-{secret}",
                source=ResolutionActorSource.REQUEST,
                claims={"authorization": secret},
            ),
        )
        resumed = await _collect(
            app.resume(
                ResumeRequest(
                    session_id="execution-profile-explicit-adoption",
                    messages=[Message.text("user", "second")],
                    profile_adoption=intent,
                )
            )
        )

        assert resumed[0].type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
        assert resumed[0].payload["decision"] == ExecutionProfileDecisionKind.ADOPTED
        assert "adoption_request_fingerprint" not in resumed[0].payload
        assert secret not in json.dumps(resumed[0].payload, sort_keys=True)
        assert "claims" not in resumed[0].payload["actor"]
        assert len(policy.requests) == 1
        assert len(provider.requests) == 1
        after = await store.load("execution-profile-explicit-adoption")
        assert after is not None
        assert after.metadata[EXECUTION_PROFILE_METADATA_KEY]["baseline"] == baseline
        assert (
            after.metadata[EXECUTION_PROFILE_METADATA_KEY]["expected"]
            == resumed[0].payload["candidate_profile"]
        )
        stored = await store.load_events("execution-profile-explicit-adoption")
        decision_index = next(
            index
            for index, event in enumerate(stored)
            if event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
            and event.payload["idempotency_identity"] == intent.idempotency_key
        )
        assert len(stored[decision_index].payload["adoption_request_fingerprint"]) == 64
        later_types = [event.type for event in stored[decision_index + 1 :]]
        assert EventType.INTERACTION_STARTED in later_types

        projection_secrets = (
            "direct_tools",
            resumed[0].payload["candidate_profile"]["fingerprint"][:8],
        )
        for projection_secret in projection_secrets:
            replay_provider = _completed_provider()
            replay_app = CayuApp(
                session_store=store,
                secret_redactor=SecretRedactor([secret, projection_secret]),
                enable_logging=False,
            )
            replay_app.register_provider(replay_provider, default=True)
            replay_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[RecordingTool("replacement_tool")],
            )
            replayed = await _collect(
                replay_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-explicit-adoption",
                        messages=[Message.text("user", "second")],
                        profile_adoption=intent,
                    )
                )
            )
            assert [event.id for event in replayed] == [resumed[0].id]
            expected_profile = ExecutionProfileIdentity.model_validate(
                replayed[0].payload["expected_profile"]
            )
            candidate_profile = ExecutionProfileIdentity.model_validate(
                replayed[0].payload["candidate_profile"]
            )
            assert tuple(
                ExecutionProfileComponentClass(value)
                for value in replayed[0].payload["changed_component_classes"]
            ) == changed_execution_profile_components(expected_profile, candidate_profile)
            assert "adoption_request_fingerprint" not in replayed[0].payload
            assert replay_provider.requests == []
        assert len(policy.requests) == 1
        assert len(provider.requests) == 1

        conflicting_requests = (
            ResumeRequest(
                session_id="execution-profile-explicit-adoption",
                messages=[Message.text("user", "different message")],
                profile_adoption=intent,
            ),
            ResumeRequest(
                session_id="execution-profile-explicit-adoption",
                messages=[Message.text("user", "second")],
                max_steps=17,
                profile_adoption=intent,
            ),
            ResumeRequest(
                session_id="execution-profile-explicit-adoption",
                messages=[Message.text("user", "second")],
                metadata={"traceparent": "00-11111111111111111111111111111111-2222222222222222-01"},
                profile_adoption=intent,
            ),
            ResumeRequest(
                session_id="execution-profile-explicit-adoption",
                messages=[Message.text("user", "second")],
                profile_adoption=intent.model_copy(
                    update={
                        "requested_by": ResolutionActor(
                            subject=f"maintainer-{secret}",
                            source=ResolutionActorSource.REQUEST,
                            claims={"authorization": "different-authority"},
                        )
                    }
                ),
            ),
        )
        for conflicting_request in conflicting_requests:
            with pytest.raises(ValueError, match="idempotency key"):
                await _collect(app.resume(conflicting_request))
        assert len(policy.requests) == 1
        assert len(provider.requests) == 1

        with pytest.raises(ValueError, match="idempotency key"):
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-explicit-adoption",
                        messages=[Message.text("user", "conflict")],
                        profile_adoption=intent.model_copy(
                            update={"reason": "A different adoption request."}
                        ),
                    )
                )
            )
        with pytest.raises(ValueError, match="idempotency key"):
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-explicit-adoption",
                        messages=[Message.text("user", "actor conflict")],
                        profile_adoption=intent.model_copy(
                            update={
                                "requested_by": ResolutionActor(
                                    subject="different-maintainer",
                                    source=ResolutionActorSource.REQUEST,
                                )
                            }
                        ),
                    )
                )
            )

        conflicting_provider = _completed_provider()
        conflicting_app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        conflicting_app.register_provider(conflicting_provider, default=True)
        conflicting_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("third_tool_profile")],
        )
        with pytest.raises(ValueError, match="idempotency key"):
            await _collect(
                conflicting_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-explicit-adoption",
                        messages=[Message.text("user", "candidate conflict")],
                        profile_adoption=intent,
                    )
                )
            )
        assert conflicting_provider.requests == []

        restarted_provider = _completed_provider()
        restarted_app = CayuApp(session_store=store, enable_logging=False)
        restarted_app.register_provider(restarted_provider, default=True)
        restarted_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("replacement_tool")],
        )
        await _collect(
            restarted_app.resume(
                ResumeRequest(
                    session_id="execution-profile-explicit-adoption",
                    messages=[Message.text("user", "after restart")],
                )
            )
        )
        assert len(restarted_provider.requests) == 1

    asyncio.run(exercise())


def test_adoption_replay_binds_stable_request_loop_policy_configuration() -> None:
    async def exercise() -> None:
        secret = "loop-policy-replay-secret-canary"
        store = InMemorySessionStore()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("original_tool")],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-loop-policy-replay",
                    messages=[Message.text("user", "initial")],
                )
            )
        )

        profile_policy = RecordingExecutionProfilePolicy(
            ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason="Adopt the replacement profile.",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            )
        )
        provider = _completed_provider()
        app = CayuApp(
            session_store=store,
            execution_profile_policy=profile_policy,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("replacement_tool")],
        )
        intent = ExecutionProfileAdoptionIntent(
            idempotency_key="loop-policy-configuration-v1",
            reason="Adopt this exact request configuration.",
            requested_by=ResolutionActor(
                subject="maintainer",
                source=ResolutionActorSource.REQUEST,
            ),
        )

        with pytest.raises(ValueError, match="adoption_replay_identity"):
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-loop-policy-replay",
                        messages=[Message.text("user", "adopt")],
                        profile_adoption=intent,
                        loop_policies=(LoopPolicy(),),
                    )
                )
            )
        assert profile_policy.requests == []
        assert provider.requests == []

        before_secret_rejection = await store.load("execution-profile-loop-policy-replay")
        before_secret_events = await store.load_events("execution-profile-loop-policy-replay")
        with pytest.raises(ValueError, match="workload secret"):
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-loop-policy-replay",
                        messages=[Message.text("user", "adopt")],
                        profile_adoption=intent,
                        loop_policies=(ConfiguredAdoptionLoopPolicy(f"configuration:{secret}"),),
                    )
                )
            )
        assert await store.load("execution-profile-loop-policy-replay") == before_secret_rejection
        assert (
            await store.load_events("execution-profile-loop-policy-replay") == before_secret_events
        )
        assert profile_policy.requests == []
        assert provider.requests == []

        admitted_policy = ConfiguredAdoptionLoopPolicy("configuration:v1")
        adopted = await _collect(
            app.resume(
                ResumeRequest(
                    session_id="execution-profile-loop-policy-replay",
                    messages=[Message.text("user", "adopt")],
                    profile_adoption=intent,
                    loop_policies=(admitted_policy,),
                )
            )
        )
        assert admitted_policy.calls == 1
        assert len(profile_policy.requests) == 1
        assert len(provider.requests) == 1

        replay_policy = ConfiguredAdoptionLoopPolicy("configuration:v1")
        replayed = await _collect(
            app.resume(
                ResumeRequest(
                    session_id="execution-profile-loop-policy-replay",
                    messages=[Message.text("user", "adopt")],
                    profile_adoption=intent,
                    loop_policies=(replay_policy,),
                )
            )
        )
        assert [event.id for event in replayed] == [adopted[0].id]
        assert replay_policy.calls == 0
        assert len(profile_policy.requests) == 1
        assert len(provider.requests) == 1

        with pytest.raises(ValueError, match="idempotency key"):
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-loop-policy-replay",
                        messages=[Message.text("user", "adopt")],
                        profile_adoption=intent,
                        loop_policies=(ConfiguredAdoptionLoopPolicy("configuration:v2"),),
                    )
                )
            )
        assert len(profile_policy.requests) == 1
        assert len(provider.requests) == 1

    asyncio.run(exercise())


def test_request_loop_policy_is_separate_invocation_authority() -> None:
    async def exercise() -> None:
        session_id = "execution-profile-request-loop-policy"
        store = InMemorySessionStore()

        def configured_app() -> tuple[CayuApp, ScriptedModelProvider]:
            provider = _completed_provider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            return app, provider

        original_app, _original_provider = configured_app()
        original_policy = ConfiguredAdoptionLoopPolicy("request-policy:v1")
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                    loop_policies=(original_policy,),
                )
            )
        )
        stored_session = await store.load(session_id)
        assert stored_session is not None
        invocation_policies = execution_profile_from_session_metadata(
            stored_session.metadata
        ).component(ExecutionProfileComponentClass.INVOCATION_POLICIES)
        assert (
            invocation_policies.strength is ExecutionProfileIdentityStrength.APPLICATION_VERSIONED
        )

        changed_app, changed_provider = configured_app()
        changed_policy = ConfiguredAdoptionLoopPolicy("request-policy:v2")
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                changed_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "changed request policy")],
                        loop_policies=(changed_policy,),
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.INVOCATION_POLICIES,
        )
        assert changed_policy.calls == 0
        assert changed_provider.requests == []

    asyncio.run(exercise())


def test_opaque_request_loop_policy_exact_instance_is_process_local_reusable() -> None:
    async def exercise() -> None:
        batch = [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop", "model": "fake-model"}),
        ]
        provider = ScriptedModelProvider([batch, batch], name="fake")
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        policy = OpaqueConfiguredLoopPolicy(BeforeStopDecision.complete("configured-v1"))

        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-opaque-request-policy-reuse",
                    messages=[Message.text("user", "first")],
                    loop_policies=(policy,),
                )
            )
        )
        created = await store.load("execution-profile-opaque-request-policy-reuse")
        assert created is not None
        created_profile = execution_profile_from_session_metadata(created.metadata)
        assert (
            created_profile.component(ExecutionProfileComponentClass.INVOCATION_POLICIES).strength
            is ExecutionProfileIdentityStrength.PROCESS_LOCAL
        )

        await _collect(
            app.resume(
                ResumeRequest(
                    session_id="execution-profile-opaque-request-policy-reuse",
                    messages=[Message.text("user", "second")],
                    loop_policies=(policy,),
                )
            )
        )
        resumed = await store.load("execution-profile-opaque-request-policy-reuse")
        assert resumed is not None
        assert execution_profile_from_session_metadata(resumed.metadata) == created_profile
        assert policy.calls == 2
        assert len(provider.requests) == 2

    asyncio.run(exercise())


def test_opaque_request_loop_policy_replacement_is_rejected_within_app() -> None:
    async def exercise() -> None:
        batch = [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop", "model": "fake-model"}),
        ]
        provider = ScriptedModelProvider([batch, batch], name="fake")
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        original = OpaqueConfiguredLoopPolicy(BeforeStopDecision.complete("configured-v1"))

        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-opaque-request-policy-replacement",
                    messages=[Message.text("user", "first")],
                    loop_policies=(original,),
                )
            )
        )
        session_before = await store.load("execution-profile-opaque-request-policy-replacement")
        transcript_before = await store.load_transcript(
            "execution-profile-opaque-request-policy-replacement"
        )
        replacement = OpaqueConfiguredLoopPolicy(BeforeStopDecision.fail("replacement behavior"))

        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-opaque-request-policy-replacement",
                        messages=[Message.text("user", "second")],
                        loop_policies=(replacement,),
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.INVOCATION_POLICIES,
        )
        assert original.calls == 1
        assert replacement.calls == 0
        assert len(provider.requests) == 1
        session_after = await store.load("execution-profile-opaque-request-policy-replacement")
        assert session_before is not None
        assert session_after is not None
        assert session_after.status is session_before.status
        assert session_after.run_epoch == session_before.run_epoch
        assert (
            await store.load_transcript("execution-profile-opaque-request-policy-replacement")
            == transcript_before
        )

    asyncio.run(exercise())


def test_concurrent_completed_adoption_loser_replays_durable_decision() -> None:
    async def exercise() -> None:
        store = ConcurrentCompletedAdoptionStore()
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("initial"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("winner"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ],
            name="fake",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-concurrent-completed-adoption",
                    messages=[Message.text("user", "initial")],
                )
            )
        )
        intent = ExecutionProfileAdoptionIntent(
            idempotency_key="concurrent-completed-adoption-v1",
            reason="Adopt this reviewed profile once.",
            requested_by=ResolutionActor(
                subject="maintainer",
                source=ResolutionActorSource.REQUEST,
            ),
        )

        async def resume(prompt: str) -> list[Event]:
            return await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-concurrent-completed-adoption",
                        messages=[Message.text("user", prompt)],
                        profile_adoption=intent,
                    )
                )
            )

        first_task = asyncio.create_task(resume("winner"))
        second_task = asyncio.create_task(resume("winner"))
        results = await asyncio.gather(first_task, second_task)
        winner_events = next(events for events in results if len(events) > 1)
        replayed_events = next(events for events in results if len(events) == 1)

        assert EventType.SESSION_COMPLETED in {event.type for event in winner_events}
        assert replayed_events[0].type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
        assert len(provider.requests) == 2
        stored_decisions = [
            event
            for event in await store.load_events("execution-profile-concurrent-completed-adoption")
            if event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
        ]
        assert len(stored_decisions) == 1
        assert stored_decisions[0].payload["idempotency_identity"] == intent.idempotency_key
        assert len(stored_decisions[0].payload["adoption_request_fingerprint"]) == 64
        assert "adoption_request_fingerprint" not in replayed_events[0].payload
        expected_public_payload = dict(stored_decisions[0].payload)
        expected_public_payload.pop("adoption_request_fingerprint")
        assert expected_public_payload == replayed_events[0].payload

    asyncio.run(exercise())


def test_concurrent_adoption_key_reuse_rejects_different_resume_input() -> None:
    async def exercise() -> None:
        store = ConcurrentCompletedAdoptionStore()
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("initial"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("winner"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ],
            name="fake",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-concurrent-adoption-conflict",
                    messages=[Message.text("user", "initial")],
                )
            )
        )
        intent = ExecutionProfileAdoptionIntent(
            idempotency_key="concurrent-adoption-conflict-v1",
            reason="Adopt this profile once.",
            requested_by=ResolutionActor(
                subject="maintainer",
                source=ResolutionActorSource.REQUEST,
            ),
        )

        async def resume(prompt: str) -> list[Event]:
            return await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-concurrent-adoption-conflict",
                        messages=[Message.text("user", prompt)],
                        profile_adoption=intent,
                    )
                )
            )

        outcomes = await asyncio.gather(
            asyncio.create_task(resume("winner")),
            asyncio.create_task(resume("conflicting loser")),
            return_exceptions=True,
        )
        assert sum(isinstance(outcome, ValueError) for outcome in outcomes) == 1
        conflict = next(outcome for outcome in outcomes if isinstance(outcome, ValueError))
        assert "idempotency key" in str(conflict)
        winner = next(outcome for outcome in outcomes if isinstance(outcome, list))
        assert EventType.SESSION_COMPLETED in {event.type for event in winner}
        assert len(provider.requests) == 2
        decisions = [
            event
            for event in await store.load_events("execution-profile-concurrent-adoption-conflict")
            if event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
        ]
        assert len(decisions) == 1

    asyncio.run(exercise())


def test_adoption_replay_validates_request_after_commit_acknowledgement_loss() -> None:
    async def exercise() -> None:
        store = CommitThenRaiseAdoptionStore()
        provider = _completed_provider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-adoption-ack-loss",
                    messages=[Message.text("user", "initial")],
                )
            )
        )
        provider.requests.clear()
        intent = ExecutionProfileAdoptionIntent(
            idempotency_key="adoption-ack-loss-v1",
            reason="Adopt this exact resume request.",
            requested_by=ResolutionActor(
                subject="maintainer",
                source=ResolutionActorSource.REQUEST,
            ),
        )
        exact_request = ResumeRequest(
            session_id="execution-profile-adoption-ack-loss",
            messages=[Message.text("user", "adopt")],
            profile_adoption=intent,
        )

        with pytest.raises(RuntimeError, match="acknowledgement lost"):
            await _collect(app.resume(exact_request))
        assert provider.requests == []

        replayed = await _collect(app.resume(exact_request))
        assert len(replayed) == 1
        assert replayed[0].type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
        assert provider.requests == []

        with pytest.raises(ValueError, match="idempotency key"):
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-adoption-ack-loss",
                        messages=[Message.text("user", "different")],
                        profile_adoption=intent,
                    )
                )
            )
        assert provider.requests == []

    asyncio.run(exercise())


def test_sqlite_restart_replays_adoption_and_loads_advanced_expectation(tmp_path) -> None:
    async def exercise() -> None:
        database_path = tmp_path / "execution-profile-adoption.sqlite"
        store = SQLiteSessionStore(database_path)
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("original_tool")],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-sqlite-restart",
                    messages=[Message.text("user", "first")],
                )
            )
        )

        policy = RecordingExecutionProfilePolicy(
            ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason="The persistent deployment was approved.",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            )
        )
        adopting_provider = _completed_provider()
        adopting_app = CayuApp(
            session_store=store,
            execution_profile_policy=policy,
            enable_logging=False,
        )
        adopting_app.register_provider(adopting_provider, default=True)
        adopting_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("replacement_tool")],
        )
        intent = ExecutionProfileAdoptionIntent(
            idempotency_key="sqlite-deployment-v1",
            reason="Adopt the persistent replacement profile.",
            requested_by=ResolutionActor(
                subject="maintainer",
                source=ResolutionActorSource.REQUEST,
            ),
        )
        adopted = await _collect(
            adopting_app.resume(
                ResumeRequest(
                    session_id="execution-profile-sqlite-restart",
                    messages=[Message.text("user", "adopt")],
                    profile_adoption=intent,
                )
            )
        )
        decision_event = adopted[0]
        assert decision_event.payload["decision"] == "adopted"
        await store.close()

        restarted_store = SQLiteSessionStore(database_path)
        try:
            replay_provider = _completed_provider()
            projection_secrets = [
                "direct_tools",
                decision_event.payload["candidate_profile"]["fingerprint"][:8],
            ]
            persisted_decision = next(
                event
                for event in await restarted_store.load_events("execution-profile-sqlite-restart")
                if event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
                and event.payload["idempotency_identity"] == intent.idempotency_key
            )
            projection_secrets.append(
                persisted_decision.payload["adoption_request_fingerprint"][:8]
            )
            projected_decision = project_persisted_runtime_event(
                persisted_decision,
                sequence=1,
                redactor=SecretRedactor(projection_secrets),
            )
            expected_profile = ExecutionProfileIdentity.model_validate(
                projected_decision.payload["expected_profile"]
            )
            candidate_profile = ExecutionProfileIdentity.model_validate(
                projected_decision.payload["candidate_profile"]
            )
            assert tuple(
                ExecutionProfileComponentClass(value)
                for value in projected_decision.payload["changed_component_classes"]
            ) == changed_execution_profile_components(expected_profile, candidate_profile)
            assert len(persisted_decision.payload["adoption_request_fingerprint"]) == 64
            assert "adoption_request_fingerprint" not in decision_event.payload
            assert "adoption_request_fingerprint" not in projected_decision.payload

            restarted_app = CayuApp(session_store=restarted_store, enable_logging=False)
            restarted_app.register_provider(replay_provider, default=True)
            restarted_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[RecordingTool("replacement_tool")],
            )
            replayed = await _collect(
                restarted_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-sqlite-restart",
                        messages=[Message.text("user", "adopt")],
                        profile_adoption=intent,
                    )
                )
            )
            assert [event.id for event in replayed] == [decision_event.id]
            assert replay_provider.requests == []

            with pytest.raises(ValueError, match="idempotency key"):
                await _collect(
                    restarted_app.resume(
                        ResumeRequest(
                            session_id="execution-profile-sqlite-restart",
                            messages=[Message.text("user", "different after restart")],
                            profile_adoption=intent,
                        )
                    )
                )
            assert replay_provider.requests == []

            await _collect(
                restarted_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-sqlite-restart",
                        messages=[Message.text("user", "ordinary continuation")],
                    )
                )
            )
            assert len(replay_provider.requests) == 1
        finally:
            await restarted_store.close()

    asyncio.run(exercise())


def test_profile_metadata_rejects_a_malformed_immutable_baseline() -> None:
    profile = build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version="test",
        provider_name="fake",
        model="fake-model",
        durable_system_prompt="durable instructions",
        direct_tools=[],
    )
    metadata = {
        EXECUTION_PROFILE_METADATA_KEY: execution_profile_session_metadata(profile),
    }
    metadata[EXECUTION_PROFILE_METADATA_KEY]["baseline"]["fingerprint"] = "0" * 64

    with pytest.raises(ValueError, match="fingerprint does not match"):
        execution_profile_from_session_metadata(metadata)
    with pytest.raises(ValueError, match="fingerprint does not match"):
        execution_profile_metadata_after_adoption(metadata, profile)


def test_sqlite_restart_rejects_a_malformed_execution_profile_baseline(tmp_path) -> None:
    async def exercise() -> None:
        database_path = tmp_path / "execution-profile-malformed-baseline.sqlite"
        store = SQLiteSessionStore(database_path)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_completed_provider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-malformed-baseline",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        await store.close()

        with sqlite3.connect(database_path) as connection:
            row = connection.execute(
                "SELECT metadata_json FROM cayu_sessions WHERE id = ?",
                ("execution-profile-malformed-baseline",),
            ).fetchone()
            assert row is not None
            metadata = json.loads(row[0])
            metadata[EXECUTION_PROFILE_METADATA_KEY]["baseline"]["fingerprint"] = "0" * 64
            connection.execute(
                "UPDATE cayu_sessions SET metadata_json = ? WHERE id = ?",
                (
                    json.dumps(metadata, separators=(",", ":"), sort_keys=True),
                    "execution-profile-malformed-baseline",
                ),
            )

        restarted_store = SQLiteSessionStore(database_path)
        provider = _completed_provider()
        restarted_app = CayuApp(session_store=restarted_store, enable_logging=False)
        restarted_app.register_provider(provider, default=True)
        restarted_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        try:
            with pytest.raises(ValueError, match="fingerprint does not match"):
                await _collect(
                    restarted_app.resume(
                        ResumeRequest(
                            session_id="execution-profile-malformed-baseline",
                            messages=[Message.text("user", "second")],
                        )
                    )
                )
            assert provider.requests == []
        finally:
            await restarted_store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "action",
    [
        ExecutionProfilePolicyAction.COMPATIBLE_REUSE,
        ExecutionProfilePolicyAction.ADOPT,
    ],
)
def test_tool_authority_change_requires_distinct_authorization(
    action: ExecutionProfilePolicyAction,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("original_tool")],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-compatible-authority",
                    messages=[Message.text("user", "first")],
                )
            )
        )

        policy = RecordingExecutionProfilePolicy(
            ExecutionProfilePolicyResult(
                action=action,
                reason="The profile policy did not grant execution authority.",
            )
        )
        provider = _completed_provider()
        app = CayuApp(
            session_store=store,
            execution_profile_policy=policy,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("replacement_tool")],
        )

        with pytest.raises(ExecutionProfileMismatchError):
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-compatible-authority",
                        messages=[Message.text("user", "second")],
                        profile_adoption=(
                            ExecutionProfileAdoptionIntent(
                                idempotency_key="unauthorized-tool-adoption-v1",
                                reason="Request the replacement tool profile.",
                                requested_by=ResolutionActor(
                                    subject="maintainer",
                                    source=ResolutionActorSource.REQUEST,
                                ),
                            )
                            if action is ExecutionProfilePolicyAction.ADOPT
                            else None
                        ),
                    )
                )
            )
        assert provider.requests == []
        events = await store.load_events("execution-profile-compatible-authority")
        assert events[-1].payload["decision"] == ExecutionProfileDecisionKind.REJECTED

    asyncio.run(exercise())


def test_explicit_policy_can_classify_non_authority_drift_as_compatible(
    monkeypatch,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        monkeypatch.setattr(session_engine_module, "_runtime_version", lambda: "old-runtime")
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-compatible-runtime",
                    messages=[Message.text("user", "first")],
                )
            )
        )

        monkeypatch.setattr(session_engine_module, "_runtime_version", lambda: "new-runtime")
        policy = RecordingExecutionProfilePolicy(
            ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.COMPATIBLE_REUSE,
                reason="The application explicitly supports this runtime transition.",
            )
        )
        provider = _completed_provider()
        app = CayuApp(
            session_store=store,
            execution_profile_policy=policy,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        events = await _collect(
            app.resume(
                ResumeRequest(
                    session_id="execution-profile-compatible-runtime",
                    messages=[Message.text("user", "second")],
                )
            )
        )

        assert events[0].payload["decision"] == "compatible_reuse"
        assert policy.requests[0].changed_component_classes == (
            ExecutionProfileComponentClass.RUNTIME,
        )
        assert len(provider.requests) == 1

    asyncio.run(exercise())


def test_versioned_policy_can_replace_prior_default_rejection(
    monkeypatch,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        monkeypatch.setattr(session_engine_module, "_runtime_version", lambda: "old-runtime")
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-policy-evolution",
                    messages=[Message.text("user", "first")],
                )
            )
        )

        monkeypatch.setattr(session_engine_module, "_runtime_version", lambda: "new-runtime")
        rejected_app = CayuApp(session_store=store, enable_logging=False)
        rejected_app.register_provider(_completed_provider(), default=True)
        rejected_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        with pytest.raises(ExecutionProfileMismatchError):
            await _collect(
                rejected_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-policy-evolution",
                        messages=[Message.text("user", "rejected")],
                    )
                )
            )

        policy = RecordingExecutionProfilePolicy(
            ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.COMPATIBLE_REUSE,
                reason="Policy version two permits this runtime transition.",
            ),
            identity="test:execution-profile-policy:v2",
        )
        provider = _completed_provider()
        accepted_app = CayuApp(
            session_store=store,
            execution_profile_policy=policy,
            enable_logging=False,
        )
        accepted_app.register_provider(provider, default=True)
        accepted_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        resumed = await _collect(
            accepted_app.resume(
                ResumeRequest(
                    session_id="execution-profile-policy-evolution",
                    messages=[Message.text("user", "accepted")],
                )
            )
        )

        assert resumed[0].type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
        assert resumed[0].payload["decision"] == "compatible_reuse"
        assert len(policy.requests) == 1
        assert len(provider.requests) == 1
        decisions = [
            event
            for event in await store.load_events("execution-profile-policy-evolution")
            if event.type
            in {
                EventType.SESSION_EXECUTION_PROFILE_DECIDED,
                EventType.SESSION_EXECUTION_PROFILE_REJECTED,
            }
        ]
        assert [event.payload["decision"] for event in decisions] == [
            "rejected",
            "compatible_reuse",
        ]
        assert decisions[0].id != decisions[1].id

    asyncio.run(exercise())


def test_explicit_policy_authority_denial_cannot_admit_non_authority_drift(
    monkeypatch,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        monkeypatch.setattr(session_engine_module, "_runtime_version", lambda: "old-runtime")
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-authority-denied",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        before = await store.load("execution-profile-authority-denied")
        assert before is not None

        monkeypatch.setattr(session_engine_module, "_runtime_version", lambda: "new-runtime")
        policy = RecordingExecutionProfilePolicy(
            ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason="Deployment authorization denied this transition.",
                authority_decision=ExecutionProfileAuthorityDecision.DENIED,
            )
        )
        provider = _completed_provider()
        app = CayuApp(
            session_store=store,
            execution_profile_policy=policy,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(ExecutionProfileAdoptionRejected):
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-authority-denied",
                        messages=[Message.text("user", "second")],
                        profile_adoption=ExecutionProfileAdoptionIntent(
                            idempotency_key="authority-denied-v1",
                            reason="Adopt the current deployment.",
                            requested_by=ResolutionActor(
                                subject="maintainer",
                                source=ResolutionActorSource.REQUEST,
                            ),
                        ),
                    )
                )
            )

        after = await store.load("execution-profile-authority-denied")
        assert after is not None
        assert after.status is before.status
        assert after.run_epoch == before.run_epoch
        assert after.metadata == before.metadata
        assert provider.requests == []
        stored = await store.load_events("execution-profile-authority-denied")
        assert stored[-1].type is EventType.SESSION_EXECUTION_PROFILE_REJECTED
        assert stored[-1].payload["authority_decision"] == "denied"
        assert stored[-1].payload["decision"] == "rejected"

    asyncio.run(exercise())


def test_profile_policy_can_require_migration_before_work() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("original_tool")],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-migration-required",
                    messages=[Message.text("user", "first")],
                )
            )
        )

        policy = RecordingExecutionProfilePolicy(
            ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.MIGRATION_REQUIRED,
                reason="A durable migration must run first.",
            )
        )
        provider = _completed_provider()
        app = CayuApp(
            session_store=store,
            execution_profile_policy=policy,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("replacement_tool")],
        )
        with pytest.raises(ExecutionProfileMigrationRequired):
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-migration-required",
                        messages=[Message.text("user", "second")],
                    )
                )
            )
        assert provider.requests == []
        stored = await store.load_events("execution-profile-migration-required")
        assert stored[-1].type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
        assert stored[-1].payload["decision"] == "migration_required"

    asyncio.run(exercise())


def test_real_cancellation_during_profile_policy_leaves_no_partial_admission() -> None:
    class BlockingPolicy(ExecutionProfilePolicy):
        def __init__(self) -> None:
            self.started = asyncio.Event()

        @property
        def identity(self) -> str:
            return "test:blocking-profile-policy:v1"

        async def decide(
            self,
            request: ExecutionProfilePolicyRequest,
        ) -> ExecutionProfilePolicyResult:
            del request
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def exercise() -> None:
        store = InMemorySessionStore()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("original_tool")],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-policy-cancel",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        before = await store.load("execution-profile-policy-cancel")
        before_events = await store.load_events("execution-profile-policy-cancel")
        assert before is not None

        policy = BlockingPolicy()
        provider = _completed_provider()
        app = CayuApp(
            session_store=store,
            execution_profile_policy=policy,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("replacement_tool")],
        )
        task = asyncio.create_task(
            _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-policy-cancel",
                        messages=[Message.text("user", "second")],
                        profile_adoption=ExecutionProfileAdoptionIntent(
                            idempotency_key="cancelled-adoption-v1",
                            reason="Adopt the replacement profile.",
                            requested_by=ResolutionActor(
                                subject="maintainer",
                                source=ResolutionActorSource.REQUEST,
                            ),
                        ),
                    )
                )
            )
        )
        await policy.started.wait()
        task.cancel()
        assert task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

        after = await store.load("execution-profile-policy-cancel")
        assert after is not None
        assert after.status is before.status
        assert after.run_epoch == before.run_epoch
        assert after.metadata == before.metadata
        assert await store.load_events("execution-profile-policy-cancel") == before_events
        assert provider.requests == []

    asyncio.run(exercise())


def test_profile_policy_failure_remains_authoritative_without_partial_admission() -> None:
    class FailingPolicy(ExecutionProfilePolicy):
        @property
        def identity(self) -> str:
            return "test:failing-profile-policy:v1"

        async def decide(
            self,
            request: ExecutionProfilePolicyRequest,
        ) -> ExecutionProfilePolicyResult:
            del request
            raise RuntimeError("profile policy failed")

    async def exercise() -> None:
        store = InMemorySessionStore()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("original_tool")],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-policy-failure",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        before = await store.load("execution-profile-policy-failure")
        before_events = await store.load_events("execution-profile-policy-failure")
        assert before is not None

        provider = _completed_provider()
        app = CayuApp(
            session_store=store,
            execution_profile_policy=FailingPolicy(),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("replacement_tool")],
        )
        with pytest.raises(ExecutionProfilePolicyError) as caught:
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-policy-failure",
                        messages=[Message.text("user", "second")],
                    )
                )
            )

        assert isinstance(caught.value.__cause__, RuntimeError)
        assert str(caught.value.__cause__) == "profile policy failed"
        after = await store.load("execution-profile-policy-failure")
        assert after is not None
        assert after.status is before.status
        assert after.run_epoch == before.run_epoch
        assert after.metadata == before.metadata
        assert await store.load_events("execution-profile-policy-failure") == before_events
        assert provider.requests == []

    asyncio.run(exercise())


def test_profile_policy_copy_does_not_emit_mutated_result_diagnostics(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "profile-policy-serializer-secret-canary-ABCDEFGHIJKLMNOP"

    class SecretBearingValue:
        def __repr__(self) -> str:
            return secret

    class MutatedResultPolicy(ExecutionProfilePolicy):
        @property
        def identity(self) -> str:
            return "test:mutated-result-policy:v1"

        async def decide(
            self,
            request: ExecutionProfilePolicyRequest,
        ) -> ExecutionProfilePolicyResult:
            del request
            result = ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason="Approved deployment profile.",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            )
            object.__setattr__(result, "reason", SecretBearingValue())
            return result

    async def exercise() -> ExecutionProfilePolicyError:
        store = InMemorySessionStore()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("original_tool")],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-policy-diagnostic-boundary",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        before = await store.load("execution-profile-policy-diagnostic-boundary")
        before_events = await store.load_events("execution-profile-policy-diagnostic-boundary")
        assert before is not None

        provider = _completed_provider()
        app = CayuApp(
            session_store=store,
            execution_profile_policy=MutatedResultPolicy(),
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("replacement_tool")],
        )
        try:
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-policy-diagnostic-boundary",
                        messages=[Message.text("user", "second")],
                        profile_adoption=ExecutionProfileAdoptionIntent(
                            idempotency_key="diagnostic-policy-v1",
                            reason="Adopt the reviewed profile.",
                            requested_by=ResolutionActor(
                                subject="maintainer",
                                source=ResolutionActorSource.REQUEST,
                            ),
                        ),
                    )
                )
            )
        except ExecutionProfilePolicyError as exc:
            error = exc
        else:
            raise AssertionError("The mutated policy result must fail closed.")

        assert isinstance(error.__cause__, ValidationError)
        after = await store.load("execution-profile-policy-diagnostic-boundary")
        assert after is not None
        assert after.status is before.status
        assert after.run_epoch == before.run_epoch
        assert after.metadata == before.metadata
        assert (
            await store.load_events("execution-profile-policy-diagnostic-boundary") == before_events
        )
        assert provider.requests == []
        return error

    with warnings.catch_warnings(record=True) as emitted, caplog.at_level(logging.WARNING):
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=ResourceWarning)
        raised = asyncio.run(exercise())

    captured = capsys.readouterr()
    diagnostic_output = " ".join(
        (
            str(raised),
            repr(raised),
            str(raised.__cause__),
            repr(raised.__cause__),
            captured.out,
            captured.err,
            *(record.getMessage() for record in caplog.records),
            *(str(record.message) for record in emitted),
        )
    )
    assert emitted == []
    assert secret not in diagnostic_output


def test_public_resume_rejects_reordered_direct_tools_before_work() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        original_provider = _completed_provider()
        original_search = RecordingTool("search")
        original_write = RecordingTool("write")
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(original_provider, default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[original_search, original_write],
        )

        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-tool-order-drift",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        assert [tool["name"] for tool in original_provider.requests[0].tools] == [
            "search",
            "write",
        ]

        replacement_provider = _completed_provider()
        replacement_write = RecordingTool("write")
        replacement_search = RecordingTool("search")
        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(replacement_provider, default=True)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[replacement_write, replacement_search],
        )

        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-tool-order-drift",
                        messages=[Message.text("user", "second")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.DIRECT_TOOLS,
            ExecutionProfileComponentClass.EFFECT_AUTHORITY,
            ExecutionProfileComponentClass.TOOL_VIEW_GRANTS,
        )
        assert replacement_provider.requests == []
        assert replacement_write.calls == []
        assert replacement_search.calls == []

    asyncio.run(exercise())


def test_changed_tool_profile_rejects_before_replacement_provider_preflight() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("original_tool")],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-before-provider-preflight",
                    messages=[Message.text("user", "first")],
                )
            )
        )

        replacement_provider = PreflightRecordingProvider()
        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(replacement_provider, default=True)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("replacement_tool")],
        )
        with pytest.raises(ExecutionProfileMismatchError):
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-before-provider-preflight",
                        messages=[Message.text("user", "second")],
                        structured_output=StructuredOutputSpec(
                            json_schema={"type": "object"},
                            strategy="native",
                        ),
                    )
                )
            )

        assert replacement_provider.native_preflight_calls == 0
        assert replacement_provider.requests == []

    asyncio.run(exercise())


def test_context_profile_drift_rejects_before_dispatch_and_explicit_adoption_can_authorize() -> (
    None
):
    async def exercise() -> None:
        store = InMemorySessionStore()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                system_prompt="Preserve this durable system instruction.",
            ),
            context_policy=MessageWindowContextPolicy(max_messages=4),
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-context-adoption",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        original_session = await store.load("execution-profile-context-adoption")
        assert original_session is not None
        original_profile = execution_profile_from_session_metadata(original_session.metadata)

        rejected_provider = _completed_provider()
        rejected_app = CayuApp(session_store=store, enable_logging=False)
        rejected_app.register_provider(rejected_provider, default=True)
        rejected_app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                system_prompt="A changed live prompt must not replace durable history.",
            ),
            context_policy=MessageWindowContextPolicy(max_messages=5),
        )
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                rejected_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-context-adoption",
                        messages=[Message.text("user", "second")],
                    )
                )
            )
        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.CONTEXT_SELECTION,
        )
        assert rejected_provider.requests == []

        policy = RecordingExecutionProfilePolicy(
            ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason="Reviewed context-policy migration.",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            )
        )
        adopted_provider = _completed_provider()
        adopted_app = CayuApp(
            session_store=store,
            execution_profile_policy=policy,
            enable_logging=False,
        )
        adopted_app.register_provider(adopted_provider, default=True)
        adopted_app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                system_prompt="A changed live prompt must not replace durable history.",
            ),
            context_policy=MessageWindowContextPolicy(max_messages=5),
        )
        events = await _collect(
            adopted_app.resume(
                ResumeRequest(
                    session_id="execution-profile-context-adoption",
                    messages=[Message.text("user", "second")],
                    profile_adoption=ExecutionProfileAdoptionIntent(
                        idempotency_key="adopt-context-policy-v1",
                        reason="Adopt the reviewed context policy.",
                        requested_by=ResolutionActor(
                            subject="maintainer",
                            source=ResolutionActorSource.REQUEST,
                        ),
                    ),
                )
            )
        )
        decision = next(
            event for event in events if event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
        )
        assert decision.payload["decision"] == ExecutionProfileDecisionKind.ADOPTED
        assert decision.payload["changed_component_classes"] == ["context_selection"]
        assert len(adopted_provider.requests) == 1
        adopted_session = await store.load("execution-profile-context-adoption")
        assert adopted_session is not None
        adopted_profile = execution_profile_from_session_metadata(adopted_session.metadata)
        assert adopted_profile.component(
            ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION
        ) == original_profile.component(ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION)

    asyncio.run(exercise())


def test_live_context_policy_mutation_between_model_steps_rejects_before_redispatch() -> None:
    class MutateContextPolicyTool(Tool):
        def __init__(self, policy: MessageWindowContextPolicy) -> None:
            self._policy = policy
            self.spec = ToolSpec(
                name="mutate_context_policy",
                description="Mutate the registered context policy.",
                input_schema={"type": "object", "properties": {}},
                execution_profile_identity=_test_behavior_identity("mutate-context-policy-tool"),
            )
            super().__init__()
            self.calls = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            self.calls += 1
            self._policy.max_messages = 2
            return ToolResult(content="mutated")

    async def exercise() -> None:
        policy = MessageWindowContextPolicy(max_messages=4)
        tool = MutateContextPolicyTool(policy)
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_mutate_policy",
                        name=tool.spec.name,
                        arguments={},
                    ),
                    ModelStreamEvent.completed(
                        {"finish_reason": "tool_calls", "model": "fake-model"}
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("must not dispatch"),
                    ModelStreamEvent.completed({"finish_reason": "stop", "model": "fake-model"}),
                ],
            ],
            name="fake",
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            context_policy=policy,
        )

        events = await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="live-context-policy-mutation",
                    messages=[Message.text("user", "mutate then continue")],
                )
            )
        )
        failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
        assert failed.payload["error_type"] == "ExecutionProfileMismatchError"
        assert "context_selection" in failed.payload["error"]
        assert tool.calls == 1
        assert len(provider.requests) == 1

    asyncio.run(exercise())


def test_live_provider_mutation_from_hook_rejects_before_redispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProviderMutatingHook(RuntimeHook):
        def __init__(self, provider: OpenAIProvider) -> None:
            self._provider = provider

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return _test_behavior_identity("provider-mutating-hook")

        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            del context
            self._provider.base_url = "https://changed.example.test/v1"

    async def exercise() -> None:
        dispatched: list[ModelRequest] = []

        async def unexpected_stream(
            provider: OpenAIProvider,
            request: ModelRequest,
        ) -> AsyncIterator[ModelStreamEvent]:
            del provider
            dispatched.append(request)
            if len(dispatched) == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_stable_tool",
                    name="stable_tool",
                    arguments={},
                )
                yield ModelStreamEvent.completed(
                    {"finish_reason": "tool_calls", "model": "gpt-test"}
                )
                return
            yield ModelStreamEvent.completed({"finish_reason": "stop", "model": "gpt-test"})

        monkeypatch.setattr(OpenAIProvider, "stream", unexpected_stream)
        provider = OpenAIProvider(api_key="test-key")
        tool = RecordingTool("stable_tool")
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="gpt-test"),
            tools=[tool],
            runtime_hooks=[ProviderMutatingHook(provider)],
        )

        events = await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="live-provider-mutation-before-dispatch",
                    messages=[Message.text("user", "do not dispatch")],
                )
            )
        )
        failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
        assert failed.payload["error_type"] == "ExecutionProfileMismatchError"
        assert "provider_adapter" in failed.payload["error"]
        assert tool.calls == [{}]
        assert len(dispatched) == 1

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("change_kind", "expected_component"),
    [
        ("provider_options", ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY),
        ("knowledge", ExecutionProfileComponentClass.KNOWLEDGE_INJECTION),
        ("compaction", ExecutionProfileComponentClass.CONTEXT_COMPACTION),
        ("application_budget", ExecutionProfileComponentClass.APPLICATION_BUDGET_POLICY),
        ("invocation_budget", ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY),
        ("structured_output", ExecutionProfileComponentClass.STRUCTURED_OUTPUT),
        ("finalization", ExecutionProfileComponentClass.FINALIZATION),
    ],
)
def test_model_semantics_drift_rejects_before_replacement_provider_dispatch(
    change_kind: str,
    expected_component: ExecutionProfileComponentClass,
) -> None:
    async def exercise() -> None:
        session_id = f"execution-profile-model-semantics-{change_kind}"
        store = InMemorySessionStore()
        first_budget = (
            BudgetPolicy(limits=(_profile_budget_limit(maximum="10", scope="app"),))
            if change_kind == "application_budget"
            else None
        )
        first_provider = (
            PreflightRecordingProvider()
            if change_kind == "structured_output"
            else _completed_provider()
        )
        first_context = (
            CheckpointCompactionContextPolicy(
                compactor=TranscriptDigestCompactor(max_summary_chars=4096),
                compact_after_messages=100,
            )
            if change_kind == "compaction"
            else (
                KnowledgeInjectionPolicy(enabled=False, max_hits=3)
                if change_kind == "knowledge"
                else None
            )
        )
        first_options = {"fake": {"temperature": 0.25}} if change_kind == "provider_options" else {}
        first_app = CayuApp(
            session_store=store,
            budget_policy=first_budget,
            enable_logging=False,
        )
        first_app.register_provider(first_provider, default=True)
        first_app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                provider_options=first_options,
            ),
            context_policy=first_context,
        )
        run_options = {}
        if change_kind == "invocation_budget":
            run_options["budget_limits"] = (_profile_budget_limit(maximum="10"),)
        elif change_kind == "structured_output":
            run_options["structured_output"] = StructuredOutputSpec(
                strategy="native",
                json_schema={"type": "object"},
            )
        elif change_kind == "finalization":
            run_options["max_steps"] = 16
        await _collect(
            first_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                    **run_options,
                )
            )
        )

        replacement_budget = (
            BudgetPolicy(limits=(_profile_budget_limit(maximum="11", scope="app"),))
            if change_kind == "application_budget"
            else None
        )
        replacement_provider = (
            PreflightRecordingProvider()
            if change_kind == "structured_output"
            else _completed_provider()
        )
        replacement_context = (
            CheckpointCompactionContextPolicy(
                compactor=TranscriptDigestCompactor(max_summary_chars=5000),
                compact_after_messages=100,
            )
            if change_kind == "compaction"
            else (
                KnowledgeInjectionPolicy(enabled=False, max_hits=4)
                if change_kind == "knowledge"
                else None
            )
        )
        replacement_options = (
            {"fake": {"temperature": 0.5}} if change_kind == "provider_options" else {}
        )
        replacement_app = CayuApp(
            session_store=store,
            budget_policy=replacement_budget,
            enable_logging=False,
        )
        replacement_app.register_provider(replacement_provider, default=True)
        replacement_app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                provider_options=replacement_options,
            ),
            context_policy=replacement_context,
        )
        resume_options = {}
        if change_kind == "invocation_budget":
            resume_options["budget_limits"] = (_profile_budget_limit(maximum="11"),)
        elif change_kind == "structured_output":
            resume_options["structured_output"] = StructuredOutputSpec(
                strategy="native",
                json_schema={"type": "array"},
            )
        elif change_kind == "finalization":
            resume_options["max_steps"] = 17

        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "second")],
                        **resume_options,
                    )
                )
            )

        assert caught.value.changed_component_classes == (expected_component,)
        assert replacement_provider.requests == []
        if isinstance(replacement_provider, PreflightRecordingProvider):
            assert replacement_provider.native_preflight_calls == 0

    asyncio.run(exercise())


@pytest.mark.parametrize("entrypoint", ["run", "resume"])
def test_application_budget_profile_and_dispatch_share_one_pre_yield_snapshot(
    entrypoint: str,
) -> None:
    async def exercise() -> None:
        session_id = f"execution-profile-budget-snapshot-{entrypoint}"
        store = InMemorySessionStore()
        provider = _completed_provider()
        original_limit = _profile_budget_limit(
            maximum="10",
            scope="app",
            reserve=True,
        )
        replacement_limit = _profile_budget_limit(
            maximum="11",
            scope="app",
            reserve=True,
        )
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(limits=(original_limit,)),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        if entrypoint == "resume":
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "first")],
                    )
                )
            )
            stream = app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "second")],
                )
            )
        else:
            stream = app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )

        interaction_started = await anext(stream)
        assert interaction_started.type is EventType.INTERACTION_STARTED
        assert app.budget_policy is not None
        app.budget_policy.limits = (replacement_limit,)
        remaining = [event async for event in stream]

        budget_events = [event for event in remaining if event.type is EventType.BUDGET_CHECKED]
        assert budget_events
        assert {event.payload["maximum"] for event in budget_events} == {"10"}
        session = await store.load(session_id)
        assert session is not None
        profile = execution_profile_from_session_metadata(session.metadata)
        assert {event.payload["execution_profile_fingerprint"] for event in budget_events} == {
            profile.fingerprint
        }

    asyncio.run(exercise())


def test_model_attempt_footprint_usage_cost_and_evidence_share_governing_profile() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "model": "fake-model",
                        "usage": {
                            "input_tokens": 2,
                            "output_tokens": 1,
                            "total_tokens": 3,
                        },
                    }
                ),
            ],
            name="fake",
        )
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(limits=(_profile_budget_limit(scope="app", reserve=True),)),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-model-evidence",
                    messages=[Message.text("user", "attribute this request")],
                )
            )
        )

        session = await store.load("execution-profile-model-evidence")
        assert session is not None
        profile = execution_profile_from_session_metadata(session.metadata)
        events = await store.load_events(session.id)
        governed = [
            event
            for event in events
            if event.type
            in {
                EventType.REQUEST_FOOTPRINT_RECORDED,
                EventType.MODEL_STARTED,
                EventType.MODEL_COMPLETED,
            }
        ]
        assert {event.payload.get("execution_profile_fingerprint") for event in governed} == {
            profile.fingerprint
        }
        budget_events = [
            event
            for event in events
            if event.type
            in {
                EventType.BUDGET_CHECKED,
                EventType.BUDGET_RESERVED,
                EventType.BUDGET_RECONCILED,
            }
        ]
        assert {event.type for event in budget_events} == {
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
            EventType.BUDGET_RECONCILED,
        }
        assert {event.payload.get("execution_profile_fingerprint") for event in budget_events} == {
            profile.fingerprint
        }
        footprint = next(
            event for event in governed if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
        )
        assert footprint.payload["schema_version"] == 2

        pricing = _profile_price_book()
        cost = estimate_session_cost(session_id=session.id, events=events, pricing=pricing)
        assert cost.line_items[0].execution_profile_fingerprint == profile.fingerprint
        evidence = await runtime_evidence(
            app,
            RuntimeEvidenceRequest(root_session_id=session.id, max_sessions=10, max_events=100),
        )
        assert evidence.sessions[0].attempts[0].execution_profile_fingerprint == (
            profile.fingerprint
        )

    asyncio.run(exercise())


def test_runtime_generated_model_profile_survives_exact_workload_secret_collision() -> None:
    async def exercise() -> None:
        baseline_app = CayuApp(enable_logging=False)
        baseline_app.register_provider(_completed_provider(), default=True)
        baseline_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        baseline_profile = session_engine_module._execution_profile_identity(
            registered_agent=baseline_app._agents["assistant"],
            provider_name="fake",
            registered_provider=baseline_app._providers["fake"],
            model="fake-model",
            durable_system_prompt=None,
            redactor=baseline_app._secret_redactor,
            process_identity=baseline_app._execution_profile_process_identity,
        )
        with pytest.raises(ValueError, match="contains a workload secret"):
            prepare_new_runtime_event(
                Event(
                    type=EventType.MODEL_STARTED,
                    session_id="untrusted-profile-authority",
                    payload={
                        "execution_profile_fingerprint": baseline_profile.fingerprint,
                    },
                ),
                redactor=SecretRedactor(baseline_profile.fingerprint),
            )
        checkpoint_event = event_with_execution_profile_fingerprint_authority(
            Event(
                type=EventType.SESSION_CHECKPOINTED,
                session_id="trusted-profile-checkpoint",
                payload={"checkpoint": "context_compacted"},
            ),
            baseline_profile.fingerprint,
        )
        prepared_checkpoint = prepare_new_runtime_event(
            checkpoint_event,
            redactor=SecretRedactor(baseline_profile.fingerprint),
        )
        projected_checkpoint = project_persisted_runtime_event(
            prepared_checkpoint,
            sequence=1,
            redactor=SecretRedactor(baseline_profile.fingerprint),
        )
        assert (
            prepared_checkpoint.payload["execution_profile_fingerprint"]
            == baseline_profile.fingerprint
        )
        assert (
            projected_checkpoint.payload["execution_profile_fingerprint"]
            == baseline_profile.fingerprint
        )
        with pytest.raises(ValueError, match="contains a workload secret"):
            prepare_new_runtime_event(
                Event(
                    type=EventType.SESSION_CHECKPOINTED,
                    session_id="untrusted-profile-checkpoint",
                    payload={
                        "checkpoint": "context_compacted",
                        "execution_profile_fingerprint": baseline_profile.fingerprint,
                    },
                ),
                redactor=SecretRedactor(baseline_profile.fingerprint),
            )

        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(baseline_profile.fingerprint),
            enable_logging=False,
        )
        app.register_provider(_completed_provider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        public_events = await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-secret-collision",
                    messages=[Message.text("user", "attribute this request")],
                )
            )
        )
        session = await store.load("execution-profile-secret-collision")
        assert session is not None
        admitted_profile = execution_profile_from_session_metadata(session.metadata)
        assert admitted_profile == baseline_profile
        private_events = await store.load_events(session.id)
        governed_types = {
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
        }
        for events in (public_events, private_events):
            governed = [event for event in events if event.type in governed_types]
            assert {event.type for event in governed} == governed_types, [
                (event.type, event.payload) for event in events
            ]
            assert {event.payload.get("execution_profile_fingerprint") for event in governed} == {
                baseline_profile.fingerprint
            }

    asyncio.run(exercise())


def test_model_retry_and_each_attempt_share_the_frozen_governing_profile() -> None:
    class RetryProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return _test_behavior_identity("retry-provider")

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelStreamEvent.error("OpenAI API request failed with HTTP 429: rate limit")
                return
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "model": "fake-model",
                    "usage": {
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "total_tokens": 3,
                    },
                }
            )

    async def exercise() -> None:
        store = InMemorySessionStore()
        provider = RetryProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-retry-attribution",
                    messages=[Message.text("user", "retry this request")],
                    retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
                )
            )
        )

        session = await store.load("execution-profile-retry-attribution")
        assert session is not None
        profile = execution_profile_from_session_metadata(session.metadata)
        events = await store.load_events(session.id)
        governed_types = {
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_ERROR,
            EventType.MODEL_RETRY,
            EventType.MODEL_ATTEMPT_DISCARDED,
            EventType.MODEL_COMPLETED,
        }
        governed = [event for event in events if event.type in governed_types]

        assert [event.type for event in governed].count(EventType.MODEL_STARTED) == 2
        assert [event.type for event in governed].count(EventType.REQUEST_FOOTPRINT_RECORDED) == 2
        assert {event.type for event in governed} == governed_types
        assert {event.payload.get("execution_profile_fingerprint") for event in governed} == {
            profile.fingerprint
        }

    asyncio.run(exercise())


def test_unavailable_required_component_is_not_compatible_with_itself() -> None:
    profile = build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version=None,
        provider_name="fake",
        model="fake-model",
        durable_system_prompt=None,
        direct_tools=[],
    )

    assert (
        profile.component(ExecutionProfileComponentClass.RUNTIME).availability
        is ExecutionProfileIdentityAvailability.UNAVAILABLE
    )
    assert changed_execution_profile_components(profile, profile) == (
        ExecutionProfileComponentClass.RUNTIME,
    )


def test_public_run_fails_closed_before_work_when_required_identity_is_unavailable(
    monkeypatch,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        provider = _completed_provider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        monkeypatch.setattr(session_engine_module, "_runtime_version", lambda: None)

        with pytest.raises(RuntimeError, match="unavailable required components: runtime"):
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="execution-profile-unavailable-run",
                        messages=[Message.text("user", "first")],
                    )
                )
            )

        assert await store.load("execution-profile-unavailable-run") is None
        assert provider.requests == []

    asyncio.run(exercise())


def test_creation_profile_truthfully_identifies_rendered_workspace_system_projection() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        provider = _completed_provider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="workspace"),
                workspace_instructions="Use the durable workspace contract.",
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                system_prompt="Follow the agent contract.",
            )
        )
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-system-projection",
                    messages=[Message.text("user", "first")],
                )
            )
        )

        session = await store.load("execution-profile-system-projection")
        transcript = await store.load_transcript("execution-profile-system-projection")
        assert session is not None
        rendered_system_prompt = transcript[0].content[0].text
        assert "Follow the agent contract." in rendered_system_prompt
        assert "Use the durable workspace contract." in rendered_system_prompt
        expected = build_execution_profile_identity(
            runtime_name=session.runtime_name or "cayu",
            runtime_version=session.runtime_version,
            provider_name=session.provider_name,
            model=session.model,
            durable_system_prompt=rendered_system_prompt,
            direct_tools=[],
        )
        persisted = execution_profile_from_session_metadata(session.metadata)
        assert persisted.component(
            ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION
        ) == expected.component(ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION)

    asyncio.run(exercise())


def test_public_resume_accepts_same_structural_profile_without_transition() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        first_provider = _completed_provider()
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(first_provider, default=True)
        first_app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                system_prompt="private durable instructions",
            ),
            tools=[RecordingTool("stable_tool")],
        )
        await _collect(
            first_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-exact-reuse",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        after_run = await store.load("execution-profile-exact-reuse")
        assert after_run is not None
        profile_record = after_run.metadata[EXECUTION_PROFILE_METADATA_KEY]
        serialized_record = json.dumps(profile_record, sort_keys=True)
        assert "private durable instructions" not in serialized_record
        assert "stable_tool" not in serialized_record

        second_provider = _completed_provider()
        second_app = CayuApp(session_store=store, enable_logging=False)
        second_app.register_provider(second_provider, default=True)
        second_app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                # Resumes use the already-durable projection, not this live value.
                system_prompt="different live prompt that is not re-injected",
            ),
            tools=[RecordingTool("stable_tool")],
        )
        resume_events = await _collect(
            second_app.resume(
                ResumeRequest(
                    session_id="execution-profile-exact-reuse",
                    messages=[Message.text("user", "second")],
                )
            )
        )

        assert len(second_provider.requests) == 1
        assert EventType.SESSION_EXECUTION_PROFILE_REJECTED not in {
            event.type for event in resume_events
        }
        assert EventType.SESSION_EXECUTION_PROFILE_DECIDED not in {
            event.type for event in resume_events
        }
        stored_events = await store.load_events("execution-profile-exact-reuse")
        exact_index = next(
            index
            for index, event in enumerate(stored_events)
            if event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
            and event.payload["decision"] == "exact_reuse"
        )
        assert stored_events[exact_index + 1].type is EventType.INTERACTION_STARTED
        after_resume = await store.load("execution-profile-exact-reuse")
        assert after_resume is not None
        assert after_resume.metadata[EXECUTION_PROFILE_METADATA_KEY] == profile_record

    asyncio.run(exercise())


def test_existing_model_switch_advances_expected_profile_for_later_resume() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        source = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("source"),
                ModelStreamEvent.completed({"finish_reason": "stop", "model": "source-model"}),
            ],
            name="source",
        )
        target = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("target first"),
                    ModelStreamEvent.completed({"finish_reason": "stop", "model": "target-model"}),
                ],
                [
                    ModelStreamEvent.text_delta("target second"),
                    ModelStreamEvent.completed({"finish_reason": "stop", "model": "target-model"}),
                ],
            ],
            name="target",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(source, default=True)
        app.register_provider(target)
        app.register_agent(
            AgentSpec(name="assistant", model="source-model"),
            tools=[RecordingTool("stable_tool")],
        )
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-model-switch",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        switched = await _collect(
            app.resume(
                ResumeRequest(
                    session_id="execution-profile-model-switch",
                    messages=[Message.text("user", "switch")],
                    target=ModelTarget(provider_name="target", model="target-model"),
                )
            )
        )
        resumed = await _collect(
            app.resume(
                ResumeRequest(
                    session_id="execution-profile-model-switch",
                    messages=[Message.text("user", "continue")],
                )
            )
        )

        assert EventType.SESSION_MODEL_SWITCHED in {event.type for event in switched}
        assert EventType.SESSION_EXECUTION_PROFILE_REJECTED not in {event.type for event in resumed}
        assert len(target.requests) == 2

    asyncio.run(exercise())


def test_public_resume_fails_closed_without_a_durable_profile() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="execution-profile-missing",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("execution-profile-missing", SessionStatus.COMPLETED)
        provider = _completed_provider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("stable_tool")],
        )

        with pytest.raises(ValueError, match="no durable execution-profile identity"):
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-missing",
                        messages=[Message.text("user", "continue")],
                    )
                )
            )
        assert provider.requests == []
        session = await store.load("execution-profile-missing")
        assert session is not None
        assert session.status is SessionStatus.COMPLETED

    asyncio.run(exercise())


def test_profile_resolution_failure_does_not_admit_resume(monkeypatch) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        provider = _completed_provider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-resolution-failure",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        before = await store.load("execution-profile-resolution-failure")
        assert before is not None
        provider.requests.clear()

        def fail_resolution(**_kwargs):
            raise RuntimeError("profile resolution failed")

        monkeypatch.setattr(
            session_engine_module,
            "_execution_profile_identity",
            fail_resolution,
        )
        with pytest.raises(RuntimeError, match="profile resolution failed"):
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-resolution-failure",
                        messages=[Message.text("user", "second")],
                    )
                )
            )

        after = await store.load("execution-profile-resolution-failure")
        assert after is not None
        assert after.status == before.status
        assert after.run_epoch == before.run_epoch
        assert provider.requests == []

    asyncio.run(exercise())
