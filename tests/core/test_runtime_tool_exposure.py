from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

import cayu.runtime.execution_profiles as execution_profiles
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.events import (
    event_with_runtime_envelope_authority,
    event_with_runtime_payload_authority,
)
from cayu.core.messages import ToolResultPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.runtime import (
    AllRegisteredToolsExposurePolicy,
    CayuApp,
    ContextCountingConfig,
    ContextCountingMode,
    EventQuery,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileComponentClass,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
    ForkExecutionProfileSelection,
    ForkSessionRequest,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    RecentTurnsContextPolicy,
    RequestFootprintConfig,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RetryPolicy,
    RunRequest,
    Session,
    SessionIdentity,
    SessionInvocationAdmission,
    SessionRunFenced,
    SessionStatus,
    StaticToolExposurePolicy,
    StructuredOutputSpec,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolCapabilityCeiling,
    ToolExposure,
    ToolExposureDecision,
    ToolExposurePolicy,
    ToolExposurePolicyRequest,
    UserInputResponse,
)
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._invocation_lifecycle import (
    ReleaseInvocationCommand,
    SettleInvocationCommand,
    _release_invocation_command_with_cleanup_authority,
    prepare_rebind_invocation_command,
)
from cayu.runtime.hooks import (
    BeforeToolCallHookContext,
    RuntimeHook,
    ToolCallHookContext,
)
from cayu.runtime.sessions import InteractionTransitionSpec
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.runtime.tool_policy import (
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.tools import UserInputTool
from cayu.vaults import SecretRedactor, StaticVault


class _RecordingTool(Tool):
    def __init__(self, name: str, *, workspace_mutation: bool = False) -> None:
        self.spec = ToolSpec(
            name=name,
            description=f"Run {name}.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
            parallel_safe=not workspace_mutation,
            workspace_mutation=workspace_mutation,
        )
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        self.calls.append(args)
        return ToolResult(content=f"ran {self.name}")


class _ScriptedProvider(ModelProvider):
    name = "tool-exposure-test"

    def __init__(self, attempts: list[list[ModelStreamEvent]]) -> None:
        self._attempts = attempts
        self.requests: list[ModelRequest] = []
        self.count_requests: list[ModelRequest] = []

    async def count_input_tokens(
        self,
        request: ModelRequest,
    ) -> InputTokenCountResult | None:
        self.count_requests.append(request)
        return InputTokenCountResult(
            input_tokens=7,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        attempt = self._attempts[len(self.requests) - 1]
        for event in attempt:
            yield event


class _CrashAfterPendingToolRoundStore(InMemorySessionStore):
    """Lose one acknowledgement after the model step committed its tool round."""

    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.crashed = False

    async def load_checkpoint(self, session_id: str) -> dict | None:
        checkpoint = await super().load_checkpoint(session_id)
        if not self.crashed and checkpoint is not None and "pending_tool_round" in checkpoint:
            self.crashed = True
            raise RuntimeError("simulated crash after pending tool-round publication")
        return checkpoint


class _CrashAfterStagedTerminalStore(InMemorySessionStore):
    """Lose one acknowledgement after the first private terminal is staged."""

    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.crashed = False

    async def load_checkpoint(self, session_id: str) -> dict | None:
        checkpoint = await super().load_checkpoint(session_id)
        pending_round = None if checkpoint is None else checkpoint.get("pending_tool_round")
        staged_terminals = (
            None if type(pending_round) is not dict else pending_round.get("staged_terminals")
        )
        if not self.crashed and staged_terminals:
            self.crashed = True
            raise RuntimeError("simulated crash after terminal staging")
        return checkpoint


class _InterleavedCompletedExposureStore(InMemorySessionStore):
    """Complete one competing invocation immediately before resume admission."""

    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.interleaved = False

    async def admit_session_invocation(
        self,
        session_id: str,
        *,
        admission: SessionInvocationAdmission,
    ) -> Session:
        if not self.interleaved:
            records = await self.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.TOOL_EXPOSURE_RECORDED,
                    limit=1,
                )
            )
            if records:
                self.interleaved = True
                running = await self.transition_status(
                    session_id,
                    from_statuses={SessionStatus.COMPLETED},
                    to_status=SessionStatus.RUNNING,
                )
                await self.transform_checkpoint(
                    session_id,
                    lambda _session, checkpoint: (
                        execution_profiles.checkpoint_with_active_invocation_execution_profile(
                            checkpoint,
                            session_id=session_id,
                            interaction_id="competing-interaction",
                            run_epoch=running.run_epoch,
                            profile=execution_profiles.execution_profile_from_session_metadata(
                                running.metadata
                            ),
                        )
                    ),
                )
                payload = {
                    **records[-1].event.payload,
                    "profile_id": "competing-phase",
                    "profile_changed": True,
                }
                competing = event_with_runtime_payload_authority(
                    Event(
                        type=EventType.TOOL_EXPOSURE_RECORDED,
                        session_id=session_id,
                        payload=payload,
                    ),
                    "catalogue_revision",
                    "execution_profile_fingerprint",
                    "exposure_fingerprint",
                    "model_step_id",
                    "profile_id",
                )
                await self.append_event(session_id, competing)
                checkpoint = await self.load_checkpoint(session_id)
                active_profile = (
                    execution_profiles.active_invocation_execution_profile_from_checkpoint(
                        checkpoint
                    )
                )
                assert active_profile is not None
                terminal = event_with_runtime_envelope_authority(
                    Event(
                        type=EventType.INTERACTION_COMPLETED,
                        session_id=session_id,
                        interaction_id=active_profile.interaction_id,
                        agent_name=running.agent_name,
                        environment_name=running.environment_name,
                    ),
                    "session_id",
                    "interaction_id",
                )
                transition = InteractionTransitionSpec(
                    event=terminal,
                    from_statuses=(SessionStatus.RUNNING,),
                    to_status=SessionStatus.COMPLETED,
                )
                runtime_store = runtime_checkpoint_session_store(self)
                completed = await runtime_store.apply_invocation_lifecycle_command(
                    SettleInvocationCommand(
                        session_id=session_id,
                        expected_session_instance_id=running.instance_id,
                        expected_run_epoch=active_profile.run_epoch,
                        expected_active_profile=active_profile,
                        transition=transition,
                    )
                )
                await runtime_store.apply_invocation_lifecycle_command(
                    _release_invocation_command_with_cleanup_authority(
                        ReleaseInvocationCommand(
                            session_id=session_id,
                            expected_session_instance_id=completed.session.instance_id,
                            expected_run_epoch=active_profile.run_epoch,
                            expected_active_profile=active_profile,
                            settlement_transition=transition,
                        )
                    )
                )
        return await super().admit_session_invocation(session_id, admission=admission)


class _RecordingApprovalPolicy(ToolPolicy):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        self.calls.append(request.tool_name)
        return ToolPolicyResult(
            decision=ToolPolicyDecision.REQUIRE_APPROVAL,
            reason="approval would be required",
        )


class _RecordingToolHook(RuntimeHook):
    def __init__(self) -> None:
        self.before_calls: list[str] = []
        self.after_calls: list[str] = []

    async def before_tool_call(self, context: BeforeToolCallHookContext):
        self.before_calls.append(context.tool_name)
        return None

    async def after_tool_call(self, context: ToolCallHookContext):
        self.after_calls.append(context.tool_name)
        return None


class _PhaseExposurePolicy(ToolExposurePolicy):
    def __init__(self) -> None:
        self.requests: list[ToolExposurePolicyRequest] = []

    def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
        self.requests.append(request)
        if request.step == 1:
            return ToolExposureDecision(profile_id="phase-one", tool_names=("alpha",))
        return ToolExposureDecision(profile_id="phase-two", tool_names=("beta",))


class _PreviousProfileRecordingPolicy(ToolExposurePolicy):
    def __init__(self) -> None:
        self.requests: list[ToolExposurePolicyRequest] = []

    def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
        self.requests.append(request)
        return ToolExposureDecision(
            profile_id=("initial-phase" if request.previous_profile_id is None else "next-phase"),
            tool_names=("alpha",),
        )


class _PreviousProfileExposurePolicy(ToolExposurePolicy):
    def __init__(
        self,
        *,
        first_profile_id: str,
        first_tools: tuple[str, ...],
        next_profile_id: str,
        next_tools: tuple[str, ...],
    ) -> None:
        self._first_profile_id = first_profile_id
        self._first_tools = first_tools
        self._next_profile_id = next_profile_id
        self._next_tools = next_tools
        self.requests: list[ToolExposurePolicyRequest] = []

    def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
        self.requests.append(request)
        if request.previous_profile_id is None:
            return ToolExposureDecision(
                profile_id=self._first_profile_id,
                tool_names=self._first_tools,
            )
        if request.previous_profile_id != self._first_profile_id:
            raise AssertionError(
                f"unexpected previous exposure profile: {request.previous_profile_id}"
            )
        return ToolExposureDecision(
            profile_id=self._next_profile_id,
            tool_names=self._next_tools,
        )


class _AllowProfileAdoption(ExecutionProfilePolicy):
    @property
    def identity(self) -> str:
        return "tests:tool-ceiling-profile-adoption:v1"

    async def decide(
        self,
        request: ExecutionProfilePolicyRequest,
    ) -> ExecutionProfilePolicyResult:
        return ExecutionProfilePolicyResult(
            action=ExecutionProfilePolicyAction.ADOPT,
            reason="The test explicitly authorizes the selected child profile.",
            authority_decision=(
                ExecutionProfileAuthorityDecision.AUTHORIZED
                if request.authority_review_required
                else ExecutionProfileAuthorityDecision.NOT_REQUIRED
            ),
        )


async def _collect(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


def _run(app: CayuApp, session_id: str, **updates) -> list[Event]:
    return asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
                **updates,
            ),
        )
    )


def _tool_names(request: ModelRequest) -> list[str]:
    return [tool["name"] for tool in request.tools]


async def _reopen_failed_session_for_recovery(
    store: InMemorySessionStore,
    session_id: str,
) -> None:
    session = await store.load(session_id)
    checkpoint = await store.load_checkpoint(session_id)
    assert session is not None
    assert checkpoint is not None
    active_profile = execution_profiles.active_invocation_execution_profile_from_checkpoint(
        checkpoint
    )
    assert active_profile is not None
    command = prepare_rebind_invocation_command(
        session,
        checkpoint,
        expected_statuses={session.status},
        target_status=SessionStatus.RUNNING,
        checkpoint_transform=lambda current_session, current_checkpoint: (
            execution_profiles.checkpoint_with_active_invocation_execution_profile(
                current_checkpoint,
                session_id=current_session.id,
                interaction_id=active_profile.interaction_id,
                run_epoch=current_session.run_epoch + 1,
                profile=active_profile.profile,
                expected=active_profile,
            )
        ),
    )
    await runtime_checkpoint_session_store(store).apply_invocation_lifecycle_command(command)


def test_static_exposure_drives_counting_and_provider_request_in_registered_order() -> None:
    provider = _ScriptedProvider(
        [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    alpha = _RecordingTool("alpha")
    beta = _RecordingTool("beta")
    gamma = _RecordingTool("gamma")
    app = CayuApp(
        context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[alpha, beta, gamma],
        tool_exposure_policy=StaticToolExposurePolicy(
            profile_id="selected",
            tools=("gamma", "alpha"),
        ),
    )

    events = _run(app, "static-exposure")

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert [_tool_names(request) for request in provider.requests] == [["alpha", "gamma"]]
    assert [_tool_names(request) for request in provider.count_requests] == [["alpha", "gamma"]]
    assert provider.requests[0].tools == provider.count_requests[0].tools
    [exposure_event] = [event for event in events if event.type is EventType.TOOL_EXPOSURE_RECORDED]
    exposure = ToolExposure.model_validate(exposure_event.payload)
    assert exposure.profile_id == "selected"
    assert exposure.registered_count == 3
    assert exposure.ceiling_count == 3
    assert exposure.exposed_count == 2
    assert exposure.profile_changed is False
    [footprint] = [event for event in events if event.type is EventType.REQUEST_FOOTPRINT_RECORDED]
    assert footprint.payload["schema_version"] == 3
    assert footprint.payload["tool_exposure"] == {
        "profile_id": exposure.profile_id,
        "exposure_fingerprint": exposure.exposure_fingerprint,
        "registered_count": exposure.registered_count,
        "ceiling_count": exposure.ceiling_count,
        "exposed_count": exposure.exposed_count,
        "profile_changed": exposure.profile_changed,
    }
    assert footprint.payload["fingerprints"]["tool_manifest"]["availability"] == "unavailable"


def test_default_exposure_preserves_all_tools_and_unbounded_run_metadata() -> None:
    provider = _ScriptedProvider(
        [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[_RecordingTool("alpha"), _RecordingTool("beta")],
    )

    events = _run(app, "default-exposure", metadata={"legacy": "x" * 5000})

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert [_tool_names(request) for request in provider.requests] == [["alpha", "beta"]]


def test_exposure_evidence_is_independent_of_request_footprint_observation() -> None:
    provider = _ScriptedProvider(
        [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = CayuApp(
        request_footprint=RequestFootprintConfig(enabled=False),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[_RecordingTool("alpha")],
    )

    events = _run(app, "exposure-without-footprint")

    assert EventType.TOOL_EXPOSURE_RECORDED in {event.type for event in events}
    assert EventType.REQUEST_FOOTPRINT_RECORDED not in {event.type for event in events}


def test_resume_retry_refreshes_profile_after_competing_completed_invocation() -> None:
    async def exercise() -> tuple[list[Event], _PreviousProfileRecordingPolicy]:
        completed = [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
        store = _InterleavedCompletedExposureStore()
        provider = _ScriptedProvider([completed, completed])
        policy = _PreviousProfileRecordingPolicy()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_RecordingTool("alpha")],
            tool_exposure_policy=policy,
        )
        await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="interleaved-exposure-resume",
                messages=[Message.text("user", "start")],
            ),
        )
        resume_request = ResumeRequest(
            session_id="interleaved-exposure-resume",
            messages=[Message.text("user", "continue")],
        )
        with pytest.raises(SessionRunFenced):
            _ = [event async for event in app.resume(resume_request)]
        resumed = [event async for event in app.resume(resume_request)]
        return resumed, policy

    resumed, policy = asyncio.run(exercise())

    assert resumed[-1].type is EventType.SESSION_COMPLETED
    assert [request.previous_profile_id for request in policy.requests] == [
        None,
        "competing-phase",
    ]
    [resumed_exposure] = [
        ToolExposure.model_validate(event.payload)
        for event in resumed
        if event.type is EventType.TOOL_EXPOSURE_RECORDED
    ]
    assert (resumed_exposure.profile_id, resumed_exposure.profile_changed) == (
        "next-phase",
        True,
    )


def test_session_without_capability_ceiling_fails_closed_on_resume_and_fork() -> None:
    async def exercise() -> None:
        completed = [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
        provider = _ScriptedProvider([completed])
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_RecordingTool("alpha")],
        )
        await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="ceiling-profile-template",
                messages=[Message.text("user", "template")],
            ),
        )
        template = await store.load("ceiling-profile-template")
        assert template is not None
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="missing-ceiling-session",
                messages=[Message.text("user", "invalid persisted session")],
            ),
            identity=SessionIdentity(
                provider_name=provider.name,
                model="fake-model",
                execution_profile=(
                    execution_profiles.execution_profile_from_session_metadata(template.metadata)
                ),
            ),
        )
        session = await store.update_status(session.id, SessionStatus.COMPLETED)

        with pytest.raises(ValueError, match="no durable tool capability ceiling"):
            _ = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session.id,
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]
        with pytest.raises(ValueError, match="no durable tool capability ceiling"):
            _ = [
                event
                async for event in app.fork_session(
                    ForkSessionRequest(
                        source_session_id=session.id,
                        session_id="missing-ceiling-child",
                    )
                )
            ]

        assert await store.load("missing-ceiling-child") is None
        assert len(provider.requests) == 1

    asyncio.run(exercise())


def test_durable_ceiling_can_narrow_on_resume_but_never_widen() -> None:
    async def exercise() -> None:
        completed = [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
        provider = _ScriptedProvider([completed, completed, completed])
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_RecordingTool("alpha"), _RecordingTool("beta")],
        )

        await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="durable-ceiling-resume",
                messages=[Message.text("user", "start")],
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("beta",)),
            ),
        )
        created = await store.load("durable-ceiling-resume")
        assert created is not None
        assert created.tool_capability_ceiling == ToolCapabilityCeiling(tool_names=("beta",))
        profile = execution_profiles.execution_profile_from_session_metadata(created.metadata)
        assert profile.component(
            ExecutionProfileComponentClass.TOOL_VIEW_GRANTS
        ) == execution_profiles.direct_tool_capability_ceiling_component(("beta",))

        narrowed_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=created.id,
                    messages=[Message.text("user", "narrow")],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                )
            )
        ]
        assert narrowed_events[-1].type is EventType.SESSION_COMPLETED
        narrowed = await store.load(created.id)
        assert narrowed is not None
        assert narrowed.tool_capability_ceiling == ToolCapabilityCeiling(tool_names=())

        with pytest.raises(ValueError, match="never widened"):
            _ = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=created.id,
                        messages=[Message.text("user", "widen")],
                        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("beta",)),
                    )
                )
            ]

        preserved_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=created.id,
                    messages=[Message.text("user", "preserve")],
                )
            )
        ]
        assert preserved_events[-1].type is EventType.SESSION_COMPLETED
        assert [_tool_names(request) for request in provider.requests] == [["beta"], [], []]
        preserved = await store.load(created.id)
        assert preserved is not None
        assert preserved.tool_capability_ceiling == ToolCapabilityCeiling(tool_names=())

    asyncio.run(exercise())


def test_forks_inherit_and_may_narrow_the_durable_ceiling() -> None:
    async def exercise() -> None:
        provider = _ScriptedProvider(
            [
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_RecordingTool("alpha"), _RecordingTool("beta")],
        )
        await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="ceiling-fork-source",
                messages=[Message.text("user", "start")],
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("alpha",)),
            ),
        )

        inherited_events = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(
                    source_session_id="ceiling-fork-source",
                    session_id="ceiling-fork-inherited",
                )
            )
        ]
        narrowed_events = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(
                    source_session_id="ceiling-fork-source",
                    session_id="ceiling-fork-narrowed",
                    transcript_cursor=1,
                    copy_checkpoint=False,
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                )
            )
        ]
        assert inherited_events[-1].type is EventType.SESSION_FORKED
        assert narrowed_events[-1].type is EventType.SESSION_FORKED
        inherited = await store.load("ceiling-fork-inherited")
        narrowed = await store.load("ceiling-fork-narrowed")
        assert inherited is not None and narrowed is not None
        assert inherited.tool_capability_ceiling == ToolCapabilityCeiling(tool_names=("alpha",))
        assert narrowed.tool_capability_ceiling == ToolCapabilityCeiling(tool_names=())

        with pytest.raises(ValueError, match="never widened"):
            _ = [
                event
                async for event in app.fork_session(
                    ForkSessionRequest(
                        source_session_id="ceiling-fork-source",
                        session_id="ceiling-fork-widened",
                        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("alpha", "beta")),
                    )
                )
            ]
        assert await store.load("ceiling-fork-widened") is None

    asyncio.run(exercise())


def test_different_agent_fork_intersects_ceiling_and_current_child_profile() -> None:
    async def exercise() -> None:
        completed = [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
        provider = _ScriptedProvider([completed, completed])
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            execution_profile_policy=_AllowProfileAdoption(),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="source", model="fake-model"),
            tools=[_RecordingTool("shared"), _RecordingTool("source_only")],
        )
        app.register_agent(
            AgentSpec(name="child", model="fake-model"),
            tools=[_RecordingTool("shared"), _RecordingTool("child_only")],
        )
        await _collect(
            app,
            RunRequest(
                agent_name="source",
                session_id="different-agent-ceiling-source",
                messages=[Message.text("user", "start")],
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("shared", "source_only")),
            ),
        )

        fork_events = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(
                    source_session_id="different-agent-ceiling-source",
                    session_id="different-agent-ceiling-child",
                    agent_name="child",
                    execution_profile_selection=(ForkExecutionProfileSelection.CURRENT_CHILD),
                    profile_adoption=ExecutionProfileAdoptionIntent(
                        idempotency_key="different-agent-ceiling-v1",
                        reason="Install the reviewed child profile.",
                        requested_by=ResolutionActor(
                            subject="test-caller",
                            source=ResolutionActorSource.REQUEST,
                        ),
                    ),
                )
            )
        ]
        assert fork_events[-1].type is EventType.SESSION_FORKED
        child = await store.load("different-agent-ceiling-child")
        assert child is not None
        assert child.tool_capability_ceiling == ToolCapabilityCeiling(tool_names=("shared",))
        child_profile = execution_profiles.execution_profile_from_session_metadata(child.metadata)
        assert child_profile.component(
            ExecutionProfileComponentClass.TOOL_VIEW_GRANTS
        ) == execution_profiles.direct_tool_capability_ceiling_component(("shared",))

        resumed_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=child.id,
                    messages=[Message.text("user", "continue")],
                )
            )
        ]
        assert resumed_events[-1].type is EventType.SESSION_COMPLETED
        assert [_tool_names(request) for request in provider.requests] == [
            ["shared", "source_only"],
            ["shared"],
        ]

    asyncio.run(exercise())


def test_changed_registration_cannot_silently_widen_a_reconstructed_session() -> None:
    async def exercise() -> None:
        completed = [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
        store = InMemorySessionStore()

        original_provider = _ScriptedProvider([completed])
        original = CayuApp(session_store=store, enable_logging=False)
        original.register_provider(original_provider, default=True)
        original.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_RecordingTool("alpha"), _RecordingTool("beta")],
        )
        await _collect(
            original,
            RunRequest(
                agent_name="assistant",
                session_id="changed-registration-ceiling",
                messages=[Message.text("user", "start")],
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("alpha",)),
            ),
        )

        added_provider = _ScriptedProvider([completed])
        with_addition = CayuApp(
            session_store=store,
            execution_profile_policy=_AllowProfileAdoption(),
            enable_logging=False,
        )
        with_addition.register_provider(added_provider, default=True)
        with_addition.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_RecordingTool("alpha"), _RecordingTool("gamma")],
        )
        added_events = [
            event
            async for event in with_addition.resume(
                ResumeRequest(
                    session_id="changed-registration-ceiling",
                    messages=[Message.text("user", "after addition")],
                    profile_adoption=ExecutionProfileAdoptionIntent(
                        idempotency_key="changed-registration-addition-v1",
                        reason="Adopt the reviewed registered catalog.",
                        requested_by=ResolutionActor(
                            subject="test-caller",
                            source=ResolutionActorSource.REQUEST,
                        ),
                    ),
                )
            )
        ]
        assert added_events[-1].type is EventType.SESSION_COMPLETED
        assert [_tool_names(request) for request in added_provider.requests] == [["alpha"]]

        removed_provider = _ScriptedProvider([completed])
        with_removal = CayuApp(
            session_store=store,
            execution_profile_policy=_AllowProfileAdoption(),
            enable_logging=False,
        )
        with_removal.register_provider(removed_provider, default=True)
        with_removal.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_RecordingTool("gamma")],
        )
        removed_events = [
            event
            async for event in with_removal.resume(
                ResumeRequest(
                    session_id="changed-registration-ceiling",
                    messages=[Message.text("user", "after removal")],
                    profile_adoption=ExecutionProfileAdoptionIntent(
                        idempotency_key="changed-registration-removal-v1",
                        reason="Adopt the reviewed registered catalog.",
                        requested_by=ResolutionActor(
                            subject="test-caller",
                            source=ResolutionActorSource.REQUEST,
                        ),
                    ),
                )
            )
        ]
        assert removed_events[-1].type is EventType.SESSION_COMPLETED
        assert [_tool_names(request) for request in removed_provider.requests] == [[]]
        reconstructed = await store.load("changed-registration-ceiling")
        assert reconstructed is not None
        assert reconstructed.tool_capability_ceiling == ToolCapabilityCeiling(tool_names=())

    asyncio.run(exercise())


def test_agent_registration_validates_the_exposure_policy_interface() -> None:
    app = CayuApp(enable_logging=False)

    with pytest.raises(TypeError, match="tool_exposure_policy must be a ToolExposurePolicy"):
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tool_exposure_policy=object(),  # type: ignore[arg-type]
        )

    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tool_exposure_policy=AllRegisteredToolsExposurePolicy(),
    )


def test_dynamic_profile_id_cannot_publish_a_workload_secret() -> None:
    provider = _ScriptedProvider([])
    app = CayuApp(
        secret_redactor=SecretRedactor("phase-one"),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[_RecordingTool("alpha")],
        tool_exposure_policy=_PhaseExposurePolicy(),
    )

    events = _run(app, "secret-profile-id")

    assert events[-1].type is EventType.SESSION_FAILED
    assert events[-1].payload["error_type"] == "ValueError"
    assert provider.requests == []


def test_empty_application_exposure_preserves_runtime_structured_output_tool() -> None:
    provider = _ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="finalizer",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "done"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    hidden = _RecordingTool("hidden")
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[hidden],
        tool_exposure_policy=StaticToolExposurePolicy(
            profile_id="tool-free",
            tools=(),
        ),
    )

    events = _run(
        app,
        "structured-output-empty-exposure",
        structured_output=StructuredOutputSpec(
            name="answer",
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        ),
        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
    )

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert _tool_names(provider.requests[0]) == [STRUCTURED_OUTPUT_TOOL_NAME]
    assert hidden.calls == []


@pytest.mark.parametrize(
    "secret_collision",
    (
        "not_exposed_in_request",
        "Tool unavailable for this model request.",
        "unavailable",
    ),
)
def test_unexposed_registered_call_is_blocked_before_policy_hooks_and_tool(
    secret_collision: str,
) -> None:
    provider = _ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="hidden-call",
                    name="hidden",
                    arguments={"value": "must-not-cross-boundary"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    visible = _RecordingTool("visible")
    hidden = _RecordingTool("hidden", workspace_mutation=True)
    tool_policy = _RecordingApprovalPolicy()
    hook = _RecordingToolHook()
    app = CayuApp(
        secret_redactor=SecretRedactor(secret_collision),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[visible, hidden],
        tool_exposure_policy=StaticToolExposurePolicy(
            profile_id="visible-only",
            tools=("visible",),
        ),
        tool_policy=tool_policy,
        runtime_hooks=[hook],
    )

    session_id = f"unexposed-call-{len(secret_collision)}"
    events = _run(app, session_id)

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert [_tool_names(request) for request in provider.requests] == [
        ["visible"],
        ["visible"],
    ]
    assert tool_policy.calls == []
    assert hook.before_calls == []
    assert hook.after_calls == []
    assert visible.calls == []
    assert hidden.calls == []
    assert EventType.TOOL_CALL_STARTED not in [event.type for event in events]
    assert EventType.TOOL_CALL_APPROVAL_REQUESTED not in [event.type for event in events]
    [blocked] = [event for event in events if event.type is EventType.TOOL_CALL_BLOCKED]
    assert blocked.tool_name == "hidden"
    assert blocked.payload["reason"] == "not_exposed_in_request"
    assert blocked.payload["profile_id"] == "visible-only"
    assert set(blocked.payload) >= {
        "tool_call_id",
        "profile_id",
        "exposure_fingerprint",
        "reason",
        "result",
    }
    assert "arguments" not in blocked.payload
    second_request_result = provider.requests[1].messages[-1].content[0]
    assert isinstance(second_request_result, ToolResultPart)
    assert second_request_result.is_error is True
    assert second_request_result.content == SecretRedactor(secret_collision).redact_text(
        "Tool unavailable for this model request."
    )
    [blocked_record] = asyncio.run(
        app.session_store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.TOOL_CALL_BLOCKED,
            )
        )
    )
    public_blocked = app.project_event_record_for_exposure(blocked_record).event
    assert public_blocked.payload["profile_id"] == "visible-only"
    assert public_blocked.payload["exposure_fingerprint"] == blocked.payload["exposure_fingerprint"]


def test_unexposed_sibling_stays_blocked_across_approval_pause() -> None:
    provider = _ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="hidden-call",
                    name="hidden",
                    arguments={"value": "must-not-execute"},
                ),
                ModelStreamEvent.tool_call(
                    id="visible-call",
                    name="visible",
                    arguments={"value": "approved"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    visible = _RecordingTool("visible")
    hidden = _RecordingTool("hidden")
    policy = _RecordingApprovalPolicy()
    exposure_policy = _PreviousProfileExposurePolicy(
        first_profile_id="visible-only",
        first_tools=("visible",),
        next_profile_id="hidden-only",
        next_tools=("hidden",),
    )
    hook = _RecordingToolHook()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[visible, hidden],
        tool_exposure_policy=exposure_policy,
        tool_policy=policy,
        runtime_hooks=[hook],
    )

    ceiling = ToolCapabilityCeiling(tool_names=("visible", "hidden"))
    pause_events = _run(
        app,
        "unexposed-approval-sibling",
        tool_capability_ceiling=ceiling,
    )
    approval = next(
        event for event in pause_events if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    resume_events = asyncio.run(
        _collect_approval(
            app,
            ToolApprovalRequest(
                session_id="unexposed-approval-sibling",
                approval_id=approval.payload["approval"]["approval_id"],
                tool_round_id=approval.payload["tool_round_id"],
                tool_call_id=approval.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
            ),
        )
    )

    assert resume_events[-1].type is EventType.SESSION_COMPLETED, resume_events[-1].payload
    assert policy.calls == ["visible"]
    assert visible.calls == [{"value": "approved"}]
    assert hidden.calls == []
    assert hook.before_calls == ["visible"]
    assert hook.after_calls == ["visible"]
    assert [_tool_names(request) for request in provider.requests] == [
        ["visible"],
        ["hidden"],
    ]
    assert [request.previous_profile_id for request in exposure_policy.requests] == [
        None,
        "visible-only",
    ]
    [initial_exposure] = [
        ToolExposure.model_validate(event.payload)
        for event in pause_events
        if event.type is EventType.TOOL_EXPOSURE_RECORDED
    ]
    [resumed_exposure] = [
        ToolExposure.model_validate(event.payload)
        for event in resume_events
        if event.type is EventType.TOOL_EXPOSURE_RECORDED
    ]
    assert (initial_exposure.profile_id, initial_exposure.profile_changed) == (
        "visible-only",
        False,
    )
    assert (resumed_exposure.profile_id, resumed_exposure.profile_changed) == (
        "hidden-only",
        True,
    )
    [blocked] = [event for event in resume_events if event.type is EventType.TOOL_CALL_BLOCKED]
    assert blocked.tool_name == "hidden"
    assert blocked.payload["reason"] == "not_exposed_in_request"
    session = asyncio.run(app.session_store.load("unexposed-approval-sibling"))
    assert session is not None
    assert session.tool_capability_ceiling == ceiling


async def _collect_approval(
    app: CayuApp,
    request: ToolApprovalRequest,
) -> list[Event]:
    return [event async for event in app.resolve_tool_approval(request)]


async def _collect_user_input(
    app: CayuApp,
    response: UserInputResponse,
) -> list[Event]:
    return [event async for event in app.resolve_user_input(response)]


def test_unexposed_sibling_stays_blocked_across_user_input_pause() -> None:
    provider = _ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="hidden-call",
                    name="hidden",
                    arguments={"value": "must-not-execute"},
                ),
                ModelStreamEvent.tool_call(
                    id="input-call",
                    name="ask_user",
                    arguments={"question": "Continue?"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    hidden = _RecordingTool("hidden")
    exposure_policy = _PreviousProfileExposurePolicy(
        first_profile_id="input-only",
        first_tools=("ask_user",),
        next_profile_id="hidden-only",
        next_tools=("hidden",),
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[UserInputTool(), hidden],
        tool_exposure_policy=exposure_policy,
    )

    ceiling = ToolCapabilityCeiling(tool_names=("ask_user", "hidden"))
    pause_events = _run(
        app,
        "unexposed-input-sibling",
        tool_capability_ceiling=ceiling,
    )
    awaiting = next(
        event for event in pause_events if event.type is EventType.SESSION_AWAITING_USER_INPUT
    )
    resume_events = asyncio.run(
        _collect_user_input(
            app,
            UserInputResponse(
                session_id="unexposed-input-sibling",
                input_id=awaiting.payload["input_id"],
                answer="yes",
            ),
        )
    )

    assert resume_events[-1].type is EventType.SESSION_COMPLETED, resume_events[-1].payload
    assert hidden.calls == []
    assert [_tool_names(request) for request in provider.requests] == [
        ["ask_user"],
        ["hidden"],
    ]
    assert [request.previous_profile_id for request in exposure_policy.requests] == [
        None,
        "input-only",
    ]
    assert not any(
        event.type is EventType.TOOL_CALL_STARTED and event.tool_name == "hidden"
        for event in resume_events
    )
    [blocked] = [event for event in resume_events if event.type is EventType.TOOL_CALL_BLOCKED]
    assert blocked.tool_name == "hidden"
    assert blocked.payload["reason"] == "not_exposed_in_request"
    session = asyncio.run(app.session_store.load("unexposed-input-sibling"))
    assert session is not None
    assert session.tool_capability_ceiling == ceiling


def test_unexposed_call_stays_blocked_during_ordinary_tool_round_recovery() -> None:
    async def scenario() -> tuple[list[Event], _RecordingTool, _RecordingApprovalPolicy]:
        store = _CrashAfterPendingToolRoundStore()
        provider = _ScriptedProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="hidden-call",
                        name="hidden",
                        arguments={"value": "must-not-execute"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
            ]
        )
        hidden = _RecordingTool("hidden")
        policy = _RecordingApprovalPolicy()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[hidden],
            tool_exposure_policy=StaticToolExposurePolicy(
                profile_id="tool-free",
                tools=(),
            ),
            tool_policy=policy,
        )
        session_id = "unexposed-ordinary-recovery"

        initial_events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
            ),
        )
        assert initial_events[-1].type is EventType.SESSION_FAILED
        await _reopen_failed_session_for_recovery(store, session_id)

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        assert len(provider.requests) == 1
        recovered = await store.load(session_id)
        assert recovered is not None
        assert recovered.tool_capability_ceiling == ToolCapabilityCeiling(tool_names=())
        return await store.load_events(session_id), hidden, policy

    events, hidden, policy = asyncio.run(scenario())

    assert events[-1].type is EventType.SESSION_INTERRUPTED
    assert hidden.calls == []
    assert policy.calls == []
    [blocked] = [event for event in events if event.type is EventType.TOOL_CALL_BLOCKED]
    assert blocked.tool_name == "hidden"
    assert blocked.payload["reason"] == "not_exposed_in_request"
    assert "arguments" not in blocked.payload


def test_staged_unexposed_call_stays_blocked_when_sibling_scope_is_incomplete() -> None:
    async def scenario() -> tuple[list[Event], _RecordingTool, _RecordingToolHook]:
        store = _CrashAfterStagedTerminalStore()
        provider = _ScriptedProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="hidden-call",
                        name="hidden",
                        arguments={"value": "must-not-execute"},
                    ),
                    ModelStreamEvent.tool_call(
                        id="visible-call",
                        name="visible",
                        arguments={"value": "not-started"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        hidden = _RecordingTool("hidden", workspace_mutation=True)
        visible = _RecordingTool("visible")
        hook = _RecordingToolHook()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="dynamic-secrets"),
                vault=StaticVault({"unused": "workload-secret"}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[visible, hidden],
            tool_exposure_policy=StaticToolExposurePolicy(
                profile_id="visible-only",
                tools=("visible",),
            ),
            runtime_hooks=[hook],
        )
        session_id = "staged-unexposed-incomplete-scope"

        initial_events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
        assert initial_events[-1].type is EventType.SESSION_FAILED
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        [staged] = checkpoint["pending_tool_round"]["staged_terminals"]
        assert staged["event"]["type"] == EventType.TOOL_CALL_BLOCKED.value

        await _reopen_failed_session_for_recovery(store, session_id)
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        assert len(provider.requests) == 1
        assert visible.calls == []
        return await store.load_events(session_id), hidden, hook

    events, hidden, hook = asyncio.run(scenario())

    assert events[-1].type is EventType.SESSION_INTERRUPTED
    assert hidden.calls == []
    assert "hidden" not in hook.before_calls
    assert "hidden" not in hook.after_calls
    [blocked] = [
        event
        for event in events
        if event.type is EventType.TOOL_CALL_BLOCKED and event.tool_name == "hidden"
    ]
    assert blocked.payload["reason"] == "not_exposed_in_request"
    assert not any(
        event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        and event.tool_name == "hidden"
        for event in events
    )


@pytest.mark.parametrize("failure_kind", ["retry", "overflow"])
def test_one_exposure_snapshot_is_reused_for_retry_and_overflow_recovery(
    failure_kind: str,
) -> None:
    class _RecoveringProvider(ModelProvider):
        name = f"exposure-{failure_kind}"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                if failure_kind == "overflow":
                    raise ModelContextOverflowError(
                        "context too large",
                        provider=self.name,
                        status_code=400,
                        error_code="context_length_exceeded",
                    )
                raise ModelProviderError(
                    "retry",
                    provider=self.name,
                    status_code=503,
                    retryable=True,
                )
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    provider = _RecoveringProvider()
    policy = _PhaseExposurePolicy()
    app = CayuApp(
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="frozen-exposure",
            fingerprint_key="frozen-exposure-test-key-material-0001",
        ),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[_RecordingTool("alpha"), _RecordingTool("beta")],
        tool_exposure_policy=policy,
        context_overflow_policy=(
            RecentTurnsContextPolicy(max_user_turns=1) if failure_kind == "overflow" else None
        ),
    )

    events = _run(
        app,
        f"frozen-{failure_kind}",
        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("alpha",)),
    )

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert len(policy.requests) == 1
    assert [_tool_names(request) for request in provider.requests] == [
        ["alpha"],
        ["alpha"],
    ]
    assert provider.requests[0].tools == provider.requests[1].tools
    [exposure_event] = [event for event in events if event.type is EventType.TOOL_EXPOSURE_RECORDED]
    footprints = [
        event.payload for event in events if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
    ]
    assert len(footprints) == 2
    assert {item["tool_exposure"]["exposure_fingerprint"] for item in footprints} == {
        exposure_event.payload["exposure_fingerprint"]
    }
    assert len({item["fingerprints"]["tool_manifest"]["value"] for item in footprints}) == 1


def test_later_model_step_receives_previous_profile_and_may_select_another() -> None:
    provider = _ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(id="alpha-call", name="alpha", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    alpha = _RecordingTool("alpha")
    beta = _RecordingTool("beta")
    policy = _PhaseExposurePolicy()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[alpha, beta],
        tool_exposure_policy=policy,
    )

    events = _run(app, "profile-transition")

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert [_tool_names(request) for request in provider.requests] == [
        ["alpha"],
        ["beta"],
    ]
    assert [request.previous_profile_id for request in policy.requests] == [
        None,
        "phase-one",
    ]
    exposure_events = [
        ToolExposure.model_validate(event.payload)
        for event in events
        if event.type is EventType.TOOL_EXPOSURE_RECORDED
    ]
    assert [(event.profile_id, event.profile_changed) for event in exposure_events] == [
        ("phase-one", False),
        ("phase-two", True),
    ]
    assert alpha.calls == [{}]
    assert beta.calls == []
