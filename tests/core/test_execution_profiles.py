from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import warnings
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import ValidationError

import cayu.runtime._session_engine as session_engine_module
from cayu import (
    EXECUTION_PROFILE_METADATA_KEY,
    AgentSpec,
    BeforeStopContext,
    BeforeStopDecision,
    CayuApp,
    Environment,
    EnvironmentSpec,
    Event,
    EventType,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAdoptionRejected,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileComponentClass,
    ExecutionProfileDecisionKind,
    ExecutionProfileIdentity,
    ExecutionProfileIdentityAvailability,
    ExecutionProfileMigrationRequired,
    ExecutionProfileMismatchError,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyError,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
    InMemorySessionStore,
    LoopPolicy,
    Message,
    ModelTarget,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SecretRedactor,
    SessionIdentity,
    SessionStatus,
    SQLiteSessionStore,
    StructuredOutputSpec,
    ThinkingConfig,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelStreamEvent
from cayu.runtime._event_projection import project_persisted_runtime_event
from cayu.runtime.execution_profiles import (
    build_execution_profile_identity,
    changed_execution_profile_components,
    execution_profile_from_session_metadata,
    execution_profile_metadata_after_adoption,
    execution_profile_session_metadata,
)
from cayu.runtime.sessions import (
    _runtime_resume_transport_metadata,
    _with_runtime_resume_transport_metadata,
    execution_profile_adoption_request_fingerprint,
)


@pytest.mark.parametrize(
    "path",
    [
        "src/cayu/cli/scaffold.py",
        "examples/dashboard_behavior_live.py",
    ],
)
def test_exact_runtime_profile_mirrors_include_workspace_mutation(path: str) -> None:
    source = (Path(__file__).resolve().parents[2] / path).read_text(encoding="utf-8")

    assert '**({"workspace_mutation": True} if tool.workspace_mutation else {})' in source


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
        )
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        return ToolResult(content="recorded")


class PreflightRecordingProvider(ScriptedModelProvider):
    supports_native_structured_output = True

    def __init__(self) -> None:
        super().__init__(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop", "model": "fake-model"}),
            ],
            name="fake",
        )
        self.native_preflight_calls = 0

    def preflight_native_structured_output_schema(self, json_schema: dict) -> None:
        self.native_preflight_calls += 1


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

    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        self.calls += 1
        self.metadata.append(context.metadata)
        return await super().before_stop(context)


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
        assert rejection.payload["changed_component_classes"] == ["direct_tools"]
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
        )
        assert replacement_provider.requests == []
        assert replacement_tool.calls == []

    asyncio.run(exercise())


def test_default_false_workspace_mutation_keeps_legacy_direct_tool_profile_shape() -> None:
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

    assert stored == legacy


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
