from __future__ import annotations

import asyncio
import warnings
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest
from tests.core._execution_profile_fixtures import profiled_session_identity

import cayu.runtime._session_engine as session_engine_module
from cayu import (
    EXECUTION_PROFILE_METADATA_KEY,
    AgentSpec,
    BeforeStopContext,
    BeforeStopDecision,
    BeforeToolCallHookContext,
    BudgetLimit,
    CayuApp,
    Environment,
    EnvironmentSpec,
    Event,
    EventQuery,
    EventType,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAdoptionRejected,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileBehaviorIdentity,
    ExecutionProfileComponentClass,
    ExecutionProfileDecisionKind,
    ExecutionProfileMismatchError,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
    ForkExecutionProfileSelection,
    ForkSessionRequest,
    ForkSystemPromptPolicy,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionsRecoveryRequest,
    InMemorySessionStore,
    InteractionStatus,
    InteractionSummaryEvidence,
    LoopPolicy,
    Message,
    ModelPrice,
    ModelStreamEvent,
    ModelTarget,
    PriceBook,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RetryPolicy,
    RunLimits,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    ScriptedModelProvider,
    SessionIdentity,
    SessionInvocationAdmission,
    SessionQuery,
    SessionRunFenced,
    SessionStatus,
    SessionStatusConflict,
    SQLiteSessionStore,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolCallHookContext,
    ToolContext,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    ToolResult,
    ToolSpec,
    UserInputResponse,
    session_fork_profile_relationship,
    session_prompt_anatomy_transition,
)
from cayu.environments.factory import register_environment_factory_cleanup_retry
from cayu.providers import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ProviderOperationAdapter,
    ProviderOperationConnection,
    ProviderOperationMode,
    ProviderOperationSnapshot,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
)
from cayu.runtime import SessionStore, _approval_support, _tool_round_recovery
from cayu.runtime._model_step_executor import model_completion_recovery_context_from_stage
from cayu.runtime._recovery_coordinator import RecoverySessionRunRequest
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
)
from cayu.runtime.execution_profiles import (
    ActiveInvocationExecutionProfile,
    ExecutionProfileIdentity,
    active_invocation_execution_profile_from_checkpoint,
    build_execution_profile_identity,
    checkpoint_with_active_invocation_execution_profile,
    execution_profile_from_session_metadata,
)
from cayu.runtime.user_input import pending_user_input_from_checkpoint
from cayu.tools.user_input import UserInputTool
from cayu.vaults import SecretRedactor


class RecordingExternalTool(Tool):
    def __init__(self, *, description: str, name: str = "side_effect") -> None:
        self.spec = ToolSpec(
            name=name,
            description=description,
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            effect="external",
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name=f"tests:{type(self).__name__}",
                behavior_version="1",
                implementation_version="1",
            ),
        )
        super().__init__()
        self.calls: list[dict[str, object]] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(dict(args))
        return ToolResult(content="recorded")


class BlockingExternalTool(RecordingExternalTool):
    def __init__(self, *, description: str) -> None:
        super().__init__(description=description)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.started.set()
        await self.release.wait()
        return await super().run(ctx, args)


class RequireApprovalPolicy(ToolPolicy):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:require-approval-policy",
            behavior_version="1",
            implementation_version="1",
        )

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)


class SelectiveApprovalPolicy(ToolPolicy):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:selective-approval-policy",
            behavior_version="1",
            implementation_version="1",
        )

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        decision = (
            ToolPolicyDecision.REQUIRE_APPROVAL
            if request.tool_name == "side_effect"
            else ToolPolicyDecision.ALLOW
        )
        return ToolPolicyResult(decision=decision)


class ContinueAtModelStepLimitPolicy(LoopPolicy):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:continue-at-model-step-limit-policy",
            behavior_version="1",
            implementation_version="1",
        )

    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        return BeforeStopDecision.continue_with(
            Message.text("user", "continue after the bounded step"),
            reason="exercise the model-step limit boundary",
        )


class VersionedRequestLoopPolicy(LoopPolicy):
    def __init__(self, version: str) -> None:
        self._version = version

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:request-loop-policy",
            behavior_version=self._version,
            implementation_version="1",
        )


class RecordingCompletionHook(RuntimeHook):
    def __init__(self, name: str) -> None:
        self._name = name
        self.sessions: list[str] = []
        self.before_tool_execution_profiles: list[ExecutionProfileIdentity | None] = []
        self.after_tool_execution_profiles: list[ExecutionProfileIdentity | None] = []
        self.execution_profiles: list[ExecutionProfileIdentity | None] = []
        self.interrupted_execution_profiles: list[ExecutionProfileIdentity | None] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name=f"tests:{type(self).__name__}:{self.name}",
            behavior_version="1",
            implementation_version="1",
        )

    async def before_tool_call(self, context: BeforeToolCallHookContext) -> None:
        self.before_tool_execution_profiles.append(context.execution_profile)

    async def after_tool_call(self, context: ToolCallHookContext) -> None:
        self.after_tool_execution_profiles.append(context.execution_profile)

    async def after_session_completed(self, context: RuntimeHookContext) -> None:
        self.sessions.append(context.session.id)
        self.execution_profiles.append(context.execution_profile)

    async def after_session_interrupted(self, context: RuntimeHookContext) -> None:
        self.interrupted_execution_profiles.append(context.execution_profile)


class BlockingCompletionHook(RecordingCompletionHook):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def after_session_completed(self, context: RuntimeHookContext) -> None:
        self.started.set()
        await self.release.wait()
        await super().after_session_completed(context)


class RecordingAdoptionPolicy(ExecutionProfilePolicy):
    def __init__(self) -> None:
        self.requests: list[ExecutionProfilePolicyRequest] = []

    @property
    def identity(self) -> str:
        return "test:active-invocation-adoption:v1"

    async def decide(
        self,
        request: ExecutionProfilePolicyRequest,
    ) -> ExecutionProfilePolicyResult:
        self.requests.append(request)
        return ExecutionProfilePolicyResult(
            action=ExecutionProfilePolicyAction.ADOPT,
            reason="Approved test profile adoption.",
            authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
        )


class FailFirstRunFenceReleaseStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.release_calls = 0

    async def release_run_fence(self, session_id: str) -> None:
        self.release_calls += 1
        if self.release_calls == 1:
            raise ConnectionError("simulated deferred run-fence release failure")
        await super().release_run_fence(session_id)


class MutatingRetryProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.mutate_registration: Callable[[], None] | None = None

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            if self.mutate_registration is None:
                raise AssertionError("Registration mutation callback was not configured.")
            self.mutate_registration()
            raise ModelProviderError(
                "temporary provider failure",
                provider=self.name,
                status_code=503,
                retryable=True,
            )
        if len(self.requests) == 2:
            yield ModelStreamEvent.tool_call(
                id="call-after-retry",
                name="side_effect",
                arguments={"value": "frozen"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class BlockingStreamProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.stream_waiting = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.text_delta("partial")
        self.stream_waiting.set()
        await self.release.wait()
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class BlockingProviderOperationAdapter(ProviderOperationAdapter):
    def __init__(self) -> None:
        self.start_entered = asyncio.Event()
        self.start_release = asyncio.Event()
        self.recovery_calls = 0

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        self.start_entered.set()
        await self.start_release.wait()

        async def events() -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.text_delta("unexpected recovery")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id="active-profile-operation",
                stream_protocol="active-profile-v1",
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )

    async def retrieve(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.recovery_calls += 1
        raise AssertionError("Changed-profile recovery reached the provider adapter.")

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        self.recovery_calls += 1
        raise AssertionError("Changed-profile recovery reached the provider adapter.")

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        return ProviderOperationSnapshot(state=state, status=ProviderOperationStatus.CANCELLED)


class BlockingBackgroundProvider(ModelProvider):
    name = "fake"

    def __init__(self, adapter: BlockingProviderOperationAdapter) -> None:
        self.adapter = adapter

    @property
    def provider_operation_mode(self) -> ProviderOperationMode:
        return ProviderOperationMode.BACKGROUND

    @property
    def provider_operations(self) -> ProviderOperationAdapter:
        return self.adapter

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        raise AssertionError("Background provider used the synchronous stream path.")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class CompletingBlockingProviderOperationAdapter(BlockingProviderOperationAdapter):
    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        async def events() -> AsyncIterator[ModelStreamEvent]:
            self.start_entered.set()
            await self.start_release.wait()
            yield ModelStreamEvent.text_delta("unexpected original stream completion")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id="active-profile-operation",
                stream_protocol="active-profile-v1",
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )

    async def retrieve(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.recovery_calls += 1
        return ProviderOperationSnapshot(
            state=state,
            status=ProviderOperationStatus.COMPLETED,
            events=(
                ModelStreamEvent.text_delta("recovered with frozen adapter"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
        )

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        raise AssertionError("Terminal retrieval must not reconnect the provider operation.")


async def collect(events: AsyncIterator[Event]) -> list[Event]:
    return [event async for event in events]


@pytest.mark.parametrize(
    ("checkpoint_variant", "expected_error"),
    [
        ("future-schema", CheckpointCompatibilityError),
        ("changed-interaction", RuntimeError),
    ],
)
def test_recovery_session_boundary_validates_full_active_profile_authority(
    checkpoint_variant: str,
    expected_error: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        session_id = f"recovery-boundary-{checkpoint_variant}"
        interaction_id = "expected-interaction"
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    ModelStreamEvent.text_delta("must not dispatch"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                name="fake",
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        identity = profiled_session_identity(provider_name="fake", model="fake-model")
        assert identity.execution_profile is not None
        await store.create(
            RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
            identity=identity,
        )
        session = await store.transition_status(
            session_id,
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )
        expected_profile = ActiveInvocationExecutionProfile(
            session_id=session_id,
            interaction_id=interaction_id,
            run_epoch=session.run_epoch,
            profile=identity.execution_profile,
        )
        stored_interaction_id = (
            "changed-interaction" if checkpoint_variant == "changed-interaction" else interaction_id
        )
        checkpoint = checkpoint_with_active_invocation_execution_profile(
            {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION},
            session_id=session_id,
            interaction_id=stored_interaction_id,
            run_epoch=session.run_epoch,
            profile=identity.execution_profile,
        )
        if checkpoint_variant == "future-schema":
            checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION + 1
        await store.checkpoint(session_id, checkpoint)

        async def unexpected_run_session(**_kwargs):
            raise AssertionError("Invalid recovery authority reached session execution.")
            yield

        monkeypatch.setattr(app, "_run_session", unexpected_run_session)
        request = RecoverySessionRunRequest(
            session=session,
            registered_agent=app._agents["assistant"],
            registered_provider=app._providers["fake"],
            registered_environment=None,
            active_invocation_profile=expected_profile,
            messages=[],
            messages_to_append=[],
            max_steps=1,
            limits=RunLimits(),
            budget_limits=(),
            retry_policy=RetryPolicy(),
            structured_output=None,
            thinking=None,
            request_loop_policies=(),
            request_metadata={},
            task_id=None,
            task_worker_id=None,
            start_event_type=None,
            start_event_payload={},
            start_task_on_enter=False,
            release_run_fence_on_exit=False,
        )

        with pytest.raises(
            expected_error,
            match="active invocation|checkpoint schema|newer root checkpoint",
        ):
            await collect(app._run_recovery_session(request))

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_fork_profile_inheritance_is_atomic_and_exactly_replayable(
    store_kind: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "fork-profile-inheritance.sqlite")
        )
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("source complete"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            name="fake",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="profile-fork-source",
                    messages=[Message.text("user", "create source")],
                )
            )
        )
        source = await store.load("profile-fork-source")
        assert source is not None
        expected = execution_profile_from_session_metadata(source.metadata)
        request = ForkSessionRequest(
            source_session_id=source.id,
            session_id="profile-fork-child",
        )

        first, replay = await asyncio.gather(
            collect(app.fork_session(request)),
            collect(app.fork_session(request)),
        )

        assert [event.type for event in first] == [EventType.SESSION_FORKED]
        assert [event.id for event in replay] == [event.id for event in first]
        assert first[0].payload["execution_profile_selection"] == "inherit_parent"
        assert first[0].payload["system_prompt_policy"] == "inherit_source"
        assert "fork_request_sha256" not in first[0].payload
        assert "selected_profile_fingerprint" not in first[0].payload
        assert "source_profile_fingerprint" not in first[0].payload
        child = await store.load("profile-fork-child")
        assert child is not None
        relationship = session_fork_profile_relationship(child)
        assert relationship is not None
        assert relationship.selection is ForkExecutionProfileSelection.INHERIT_PARENT
        assert relationship.source_profile == expected
        assert relationship.selected_profile == expected
        assert relationship.decision is None
        records = await store.query_events(EventQuery(session_id=child.id, limit=10))
        assert [record.event.type for record in records] == [EventType.SESSION_FORKED]
        stored_payload = records[0].event.payload
        assert stored_payload["fork_request_sha256"] == relationship.request_sha256
        assert stored_payload["selected_profile_fingerprint"] == expected.fingerprint
        assert stored_payload["source_profile_fingerprint"] == expected.fingerprint
        if isinstance(store, SQLiteSessionStore):
            assert (
                await store.prune_events(
                    before=datetime.now(UTC) + timedelta(seconds=1),
                    session_id=child.id,
                )
                == 0
            )
            replay_after_pruning = await collect(app.fork_session(request))
            assert [event.id for event in replay_after_pruning] == [event.id for event in first]
        await store.delete_session(source.id)
        surviving_child = await store.load(child.id)
        assert surviving_child is not None
        assert surviving_child.parent_session_id is None
        assert session_fork_profile_relationship(surviving_child) == relationship
        close = getattr(store, "close", None)
        if close is not None:
            await close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "store_outcome",
    [
        "lost_acknowledgement",
        "malformed_result",
        "changed_session",
        "oversized_events",
    ],
)
def test_profiled_fork_reconciles_ambiguous_custom_store_results(
    store_outcome: str,
) -> None:
    class AmbiguousForkStore(InMemorySessionStore):
        async def create_profiled_fork(self, *args, **kwargs):
            result = await super().create_profiled_fork(*args, **kwargs)
            if store_outcome == "lost_acknowledgement":
                raise ConnectionError("fork acknowledgement was lost")
            if store_outcome == "malformed_result":
                return object()
            if store_outcome == "oversized_events":
                return result.model_copy(
                    update={"events": result.events * 3},
                    deep=True,
                )
            return result.model_copy(
                update={
                    "session": result.session.model_copy(
                        update={"status": SessionStatus.RUNNING},
                        deep=True,
                    )
                },
                deep=True,
            )

    async def scenario() -> None:
        store = AmbiguousForkStore()
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=f"ambiguous-profiled-fork-source-{store_outcome}",
                messages=[],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        source = await store.update_status(source.id, SessionStatus.COMPLETED)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider([], name="fake"), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        request = ForkSessionRequest(
            source_session_id=source.id,
            session_id=f"ambiguous-profiled-fork-child-{store_outcome}",
        )

        events = await collect(app.fork_session(request))
        replayed = await collect(app.fork_session(request))

        assert [event.type for event in events] == [EventType.SESSION_FORKED]
        assert [event.id for event in replayed] == [event.id for event in events]
        child = await store.load(request.session_id or "")
        assert child is not None
        relationship = session_fork_profile_relationship(child)
        assert relationship is not None
        records = await store.query_events(EventQuery(session_id=child.id, limit=10))
        assert [record.event.type for record in records] == [EventType.SESSION_FORKED]
        assert records[0].event.id == relationship.fork_event_id

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "tampered_control",
    [
        "transcript_cursor",
        "prompt_replacement",
        "checkpoint_validation",
        "checkpoint_result",
        "transcript_validation",
    ],
)
def test_profiled_fork_store_rejects_controls_that_conflict_with_relationship(
    tampered_control: str,
) -> None:
    class TamperingForkStore(InMemorySessionStore):
        async def create_profiled_fork(self, *args, **kwargs):
            if tampered_control == "transcript_cursor":
                kwargs["transcript_cursor"] = 0
            elif tampered_control == "prompt_replacement":
                kwargs["system_prompt_replacement"] = object()
            elif tampered_control == "checkpoint_validation":
                kwargs["checkpoint_transform"] = None
            elif tampered_control == "checkpoint_result":
                kwargs["checkpoint_transform"] = lambda _session, _checkpoint: None
            else:
                kwargs["transcript_validator"] = None
            return await super().create_profiled_fork(*args, **kwargs)

    async def scenario() -> None:
        store = TamperingForkStore()
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=f"tampered-fork-source-{tampered_control}",
                messages=[],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        await store.checkpoint(source.id, {"retained": True})
        source = await store.update_status(source.id, SessionStatus.COMPLETED)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider([], name="fake"), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        child_id = f"tampered-fork-child-{tampered_control}"

        with pytest.raises(ValueError):
            await collect(
                app.fork_session(
                    ForkSessionRequest(
                        source_session_id=source.id,
                        session_id=child_id,
                    )
                )
            )

        assert await store.load(child_id) is None
        assert await store.load(source.id) == source
        assert await store.load_checkpoint(source.id) == {"retained": True}
        assert await store.query_events(EventQuery(session_id=child_id, limit=10)) == []

    asyncio.run(scenario())


def test_profiled_fork_store_child_cancellation_is_not_caller_cancellation() -> None:
    class ChildCancelledForkStore(InMemorySessionStore):
        async def create_profiled_fork(self, *args, **kwargs):
            raise asyncio.CancelledError("store child cancelled itself")

    async def scenario() -> None:
        store = ChildCancelledForkStore()
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="child-cancelled-profiled-fork-source",
                messages=[],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        source = await store.update_status(source.id, SessionStatus.COMPLETED)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider([], name="fake"), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        current_task = asyncio.current_task()
        assert current_task is not None

        with pytest.raises(RuntimeError, match="without caller cancellation"):
            await collect(
                app.fork_session(
                    ForkSessionRequest(
                        source_session_id=source.id,
                        session_id="child-cancelled-profiled-fork-child",
                    )
                )
            )

        assert current_task.cancelling() == 0
        assert current_task.cancelled() is False
        assert await store.load("child-cancelled-profiled-fork-child") is None

    asyncio.run(scenario())


def test_profiled_fork_cancellation_waits_for_dispatched_store_mutation() -> None:
    class CommitThenBlockForkStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.committed = asyncio.Event()
            self.release_acknowledgement = asyncio.Event()

        async def create_profiled_fork(self, *args, **kwargs):
            result = await super().create_profiled_fork(*args, **kwargs)
            self.committed.set()
            await self.release_acknowledgement.wait()
            return result

    async def scenario() -> None:
        store = CommitThenBlockForkStore()
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="cancelled-profiled-fork-source",
                messages=[],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        source = await store.update_status(source.id, SessionStatus.COMPLETED)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider([], name="fake"), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        request = ForkSessionRequest(
            source_session_id=source.id,
            session_id="cancelled-profiled-fork-child",
        )
        consumer = asyncio.create_task(
            collect(app.fork_session(request)),
            name="profiled-fork-cancellation-consumer",
        )
        await store.committed.wait()

        consumer.cancel("stop waiting for the fork")
        await asyncio.sleep(0)
        assert consumer.done() is False
        assert consumer.cancelling() == 0
        consumer.cancel("second cancellation while the store is settling")
        await asyncio.sleep(0)
        assert consumer.done() is False
        assert consumer.cancelling() == 0
        store.release_acknowledgement.set()

        with pytest.raises(asyncio.CancelledError) as exc_info:
            await consumer
        assert exc_info.value.args == ("stop waiting for the fork",)
        assert consumer.cancelling() == 2
        assert consumer.cancelled() is True
        child = await store.load(request.session_id or "")
        assert child is not None
        relationship = session_fork_profile_relationship(child)
        assert relationship is not None
        records = await store.query_events(EventQuery(session_id=child.id, limit=10))
        assert [record.event.type for record in records] == [EventType.SESSION_FORKED]

        replayed = await collect(app.fork_session(request))
        assert [event.type for event in replayed] == [EventType.SESSION_FORKED]
        assert records[0].event.id == relationship.fork_event_id

    asyncio.run(scenario())


def test_profiled_fork_supervisory_exit_waits_for_store_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CommitThenBlockForkStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.committed = asyncio.Event()
            self.release_acknowledgement = asyncio.Event()
            self.acknowledgement_returned = asyncio.Event()

        async def create_profiled_fork(self, *args, **kwargs):
            result = await super().create_profiled_fork(*args, **kwargs)
            self.committed.set()
            await self.release_acknowledgement.wait()
            self.acknowledgement_returned.set()
            return result

    async def scenario() -> None:
        store = CommitThenBlockForkStore()
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="abandoned-profiled-fork-source",
                messages=[],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        source = await store.update_status(source.id, SessionStatus.COMPLETED)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider([], name="fake"), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        request = ForkSessionRequest(
            source_session_id=source.id,
            session_id="abandoned-profiled-fork-child",
        )
        original_wait = session_engine_module.await_shielded_task_outcome
        supervisory_exit_delivered = False

        async def abandon_once(task, **kwargs):
            nonlocal supervisory_exit_delivered
            if (
                not supervisory_exit_delivered
                and task.get_name() == "cayu-profiled-session-fork-publication"
            ):
                await store.committed.wait()
                supervisory_exit_delivered = True
                asyncio.get_running_loop().call_soon(store.release_acknowledgement.set)
                raise GeneratorExit("abandon fork stream")
            return await original_wait(task, **kwargs)

        monkeypatch.setattr(
            session_engine_module,
            "await_shielded_task_outcome",
            abandon_once,
        )

        with pytest.raises(GeneratorExit, match="abandon fork stream"):
            await collect(app.fork_session(request))

        assert supervisory_exit_delivered is True
        assert store.acknowledgement_returned.is_set()
        child = await store.load(request.session_id or "")
        assert child is not None
        relationship = session_fork_profile_relationship(child)
        assert relationship is not None
        records = await store.query_events(EventQuery(session_id=child.id, limit=10))
        assert [record.event.type for record in records] == [EventType.SESSION_FORKED]
        assert records[0].event.id == relationship.fork_event_id

        replayed = await collect(app.fork_session(request))
        assert [event.type for event in replayed] == [EventType.SESSION_FORKED]

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_fork_current_child_profile_requires_and_records_authorized_adoption(
    store_kind: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "fork-current-profile.sqlite")
        )
        source_provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("source complete"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            name="fake",
        )
        source_app = CayuApp(session_store=store, enable_logging=False)
        source_app.register_provider(source_provider, default=True)
        source_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingExternalTool(description="Original tool")],
        )
        await collect(
            source_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="current-profile-source",
                    messages=[Message.text("user", "create source")],
                )
            )
        )
        source = await store.load("current-profile-source")
        assert source is not None
        source_profile = execution_profile_from_session_metadata(source.metadata)

        policy = RecordingAdoptionPolicy()
        fork_app = CayuApp(
            session_store=store,
            enable_logging=False,
            execution_profile_policy=policy,
        )
        fork_app.register_provider(ScriptedModelProvider([], name="fake"), default=True)
        fork_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingExternalTool(description="Authorized replacement tool")],
        )
        request = ForkSessionRequest(
            source_session_id=source.id,
            session_id="current-profile-child",
            execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
            profile_adoption=ExecutionProfileAdoptionIntent(
                idempotency_key="fork-current-profile",
                reason="Use the current child registration.",
                requested_by=ResolutionActor(
                    subject="operator",
                    source=ResolutionActorSource.REQUEST,
                ),
            ),
        )
        events = await collect(fork_app.fork_session(request))

        assert [event.type for event in events] == [EventType.SESSION_FORKED]
        child = await store.load("current-profile-child")
        assert child is not None
        relationship = session_fork_profile_relationship(child)
        assert relationship is not None
        assert relationship.selection is ForkExecutionProfileSelection.CURRENT_CHILD
        assert relationship.source_profile == source_profile
        assert relationship.selected_profile != source_profile
        assert relationship.decision is not None
        assert relationship.decision.kind is ExecutionProfileDecisionKind.ADOPTED
        assert relationship.decision.actor.subject == "operator"
        assert len(policy.requests) == 1
        parent_after_fork = await store.load(source.id)
        assert parent_after_fork is not None
        assert execution_profile_from_session_metadata(parent_after_fork.metadata) == source_profile
        records = await store.query_events(EventQuery(session_id=child.id, limit=10))
        assert [record.event.type for record in records] == [
            EventType.SESSION_EXECUTION_PROFILE_DECIDED,
            EventType.SESSION_FORKED,
        ]
        if isinstance(store, SQLiteSessionStore):
            assert (
                await store.prune_events(
                    before=datetime.now(UTC) + timedelta(seconds=1),
                    session_id=child.id,
                )
                == 0
            )

        later_policy = RecordingAdoptionPolicy()
        later_app = CayuApp(
            session_store=store,
            enable_logging=False,
            execution_profile_policy=later_policy,
        )
        later_app.register_provider(
            ScriptedModelProvider(
                [
                    ModelStreamEvent.text_delta("child adopted a later profile"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                name="fake",
            ),
            default=True,
        )
        later_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingExternalTool(description="Later child-only tool")],
        )
        resumed = await collect(
            later_app.resume(
                ResumeRequest(
                    session_id=child.id,
                    messages=[Message.text("user", "adopt only on the child")],
                    target=ModelTarget(provider_name="fake", model="later-child-model"),
                    profile_adoption=ExecutionProfileAdoptionIntent(
                        idempotency_key="later-child-profile",
                        reason="Adopt a later profile on the child only.",
                        requested_by=ResolutionActor(
                            subject="operator",
                            source=ResolutionActorSource.REQUEST,
                        ),
                    ),
                )
            )
        )
        assert any(event.type is EventType.SESSION_COMPLETED for event in resumed)
        parent_after_child_adoption = await store.load(source.id)
        adopted_child = await store.load(child.id)
        assert parent_after_child_adoption is not None
        assert adopted_child is not None
        assert (
            execution_profile_from_session_metadata(parent_after_child_adoption.metadata)
            == source_profile
        )
        relationship_after_adoption = session_fork_profile_relationship(adopted_child)
        assert relationship_after_adoption == relationship
        assert execution_profile_from_session_metadata(adopted_child.metadata) != (
            relationship.selected_profile
        )
        assert len(later_policy.requests) == 1
        replay_after_child_adoption = await collect(fork_app.fork_session(request))
        assert [event.id for event in replay_after_child_adoption] == [event.id for event in events]
        close = getattr(store, "close", None)
        if close is not None:
            await close()

    asyncio.run(scenario())


def test_fork_current_child_profile_reviews_a_different_registered_provider() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        source_app = CayuApp(session_store=store, enable_logging=False)
        source_app.register_provider(
            ScriptedModelProvider(
                [
                    ModelStreamEvent.text_delta("source complete"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                name="source-provider",
            ),
            default=True,
        )
        source_app.register_agent(
            AgentSpec(
                name="source-agent",
                model="source-model",
                provider_name="source-provider",
            )
        )
        await collect(
            source_app.run(
                RunRequest(
                    agent_name="source-agent",
                    session_id="cross-provider-fork-source",
                    messages=[Message.text("user", "create source")],
                )
            )
        )
        source = await store.load("cross-provider-fork-source")
        assert source is not None

        policy = RecordingAdoptionPolicy()
        fork_app = CayuApp(
            session_store=store,
            enable_logging=False,
            execution_profile_policy=policy,
        )
        fork_app.register_provider(
            ScriptedModelProvider([], name="source-provider"),
            default=True,
        )
        fork_app.register_provider(
            ScriptedModelProvider([], name="child-provider"),
        )
        fork_app.register_agent(
            AgentSpec(
                name="source-agent",
                model="source-model",
                provider_name="source-provider",
            )
        )
        fork_app.register_agent(
            AgentSpec(
                name="child-agent",
                model="child-model",
                provider_name="child-provider",
            )
        )
        events = await collect(
            fork_app.fork_session(
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id="cross-provider-fork-child",
                    agent_name="child-agent",
                    execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                    profile_adoption=ExecutionProfileAdoptionIntent(
                        idempotency_key="cross-provider-fork",
                        reason="Authorize the child provider target.",
                        requested_by=ResolutionActor(
                            subject="operator",
                            source=ResolutionActorSource.REQUEST,
                        ),
                    ),
                )
            )
        )

        assert [event.type for event in events] == [EventType.SESSION_FORKED]
        child = await store.load("cross-provider-fork-child")
        assert child is not None
        assert child.provider_name == "child-provider"
        assert child.model == "child-model"
        relationship = session_fork_profile_relationship(child)
        assert relationship is not None
        assert relationship.child_provider_name == "child-provider"
        assert relationship.decision is not None
        assert len(policy.requests) == 1
        assert policy.requests[0].source_provider_name == "source-provider"
        assert policy.requests[0].target_provider_name == "child-provider"

    asyncio.run(scenario())


def test_fork_current_child_provider_change_requires_application_policy() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        source_app = CayuApp(session_store=store, enable_logging=False)
        source_app.register_provider(
            ScriptedModelProvider(
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                name="source-provider",
            ),
            default=True,
        )
        source_app.register_agent(
            AgentSpec(
                name="source-agent",
                model="shared-model",
                provider_name="source-provider",
            )
        )
        await collect(
            source_app.run(
                RunRequest(
                    agent_name="source-agent",
                    session_id="unreviewed-provider-fork-source",
                    messages=[Message.text("user", "create source")],
                )
            )
        )

        fork_app = CayuApp(session_store=store, enable_logging=False)
        fork_app.register_provider(
            ScriptedModelProvider([], name="source-provider"),
            default=True,
        )
        fork_app.register_provider(ScriptedModelProvider([], name="child-provider"))
        fork_app.register_agent(
            AgentSpec(
                name="source-agent",
                model="shared-model",
                provider_name="source-provider",
            )
        )
        fork_app.register_agent(
            AgentSpec(
                name="child-agent",
                model="shared-model",
                provider_name="child-provider",
            )
        )

        with pytest.raises(ExecutionProfileAdoptionRejected):
            await collect(
                fork_app.fork_session(
                    ForkSessionRequest(
                        source_session_id="unreviewed-provider-fork-source",
                        session_id="unreviewed-provider-fork-child",
                        agent_name="child-agent",
                        execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                        profile_adoption=ExecutionProfileAdoptionIntent(
                            idempotency_key="unreviewed-provider-fork",
                            reason="Attempt an unreviewed provider change.",
                            requested_by=ResolutionActor(
                                subject="operator",
                                source=ResolutionActorSource.REQUEST,
                            ),
                        ),
                    )
                )
            )

        assert await store.load("unreviewed-provider-fork-child") is None

    asyncio.run(scenario())


def test_fork_same_model_cross_provider_preflights_before_child_creation() -> None:
    class RejectingPortableProvider(ScriptedModelProvider):
        def __init__(self) -> None:
            super().__init__([], name="child-provider")
            self.preflight_calls = 0

        def preflight_portable_messages(
            self,
            *,
            model: str,
            messages: list[Message],
            tools: list[dict[str, Any]],
        ) -> None:
            del model, messages, tools
            self.preflight_calls += 1
            raise ValueError("target provider rejected source history")

    async def scenario() -> None:
        store = InMemorySessionStore()
        source_app = CayuApp(session_store=store, enable_logging=False)
        source_app.register_provider(
            ScriptedModelProvider(
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                name="source-provider",
            ),
            default=True,
        )
        source_app.register_agent(
            AgentSpec(
                name="source-agent",
                model="shared-model",
                provider_name="source-provider",
            )
        )
        await collect(
            source_app.run(
                RunRequest(
                    agent_name="source-agent",
                    session_id="same-model-cross-provider-source",
                    messages=[Message.text("user", "source history")],
                )
            )
        )
        source_before = await store.load("same-model-cross-provider-source")
        assert source_before is not None

        target_provider = RejectingPortableProvider()
        fork_app = CayuApp(
            session_store=store,
            enable_logging=False,
            execution_profile_policy=RecordingAdoptionPolicy(),
        )
        fork_app.register_provider(
            ScriptedModelProvider([], name="source-provider"),
            default=True,
        )
        fork_app.register_provider(target_provider)
        fork_app.register_agent(
            AgentSpec(
                name="source-agent",
                model="shared-model",
                provider_name="source-provider",
            )
        )
        fork_app.register_agent(
            AgentSpec(
                name="child-agent",
                model="shared-model",
                provider_name="child-provider",
            )
        )

        with pytest.raises(ValueError, match="target provider rejected source history"):
            await collect(
                fork_app.fork_session(
                    ForkSessionRequest(
                        source_session_id=source_before.id,
                        session_id="same-model-cross-provider-child",
                        agent_name="child-agent",
                        execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                        profile_adoption=ExecutionProfileAdoptionIntent(
                            idempotency_key="same-model-cross-provider",
                            reason="Adopt the reviewed target provider.",
                            requested_by=ResolutionActor(
                                subject="operator",
                                source=ResolutionActorSource.REQUEST,
                            ),
                        ),
                    )
                )
            )

        assert target_provider.preflight_calls == 1
        assert await store.load("same-model-cross-provider-child") is None
        assert await store.load(source_before.id) == source_before

    asyncio.run(scenario())


def test_generated_current_prompt_fork_replays_one_stable_descendant() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        source_app = CayuApp(session_store=store, enable_logging=False)
        source_app.register_provider(
            ScriptedModelProvider(
                [
                    ModelStreamEvent.text_delta("source complete"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                name="fake",
            ),
            default=True,
        )
        source_app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                system_prompt="source prompt",
            )
        )
        await collect(
            source_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="generated-current-prompt-source",
                    messages=[Message.text("user", "create source")],
                )
            )
        )

        child_app = CayuApp(
            session_store=store,
            enable_logging=False,
            execution_profile_policy=RecordingAdoptionPolicy(),
        )
        child_app.register_provider(
            ScriptedModelProvider([], name="fake"),
            default=True,
        )
        child_app.register_environment(
            Environment(EnvironmentSpec(name="body")),
            default=True,
        )
        child_app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                system_prompt="child prompt",
            )
        )
        request = ForkSessionRequest(
            source_session_id="generated-current-prompt-source",
            agent_name="assistant",
            environment_name="body",
            copy_checkpoint=False,
            system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
            execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
            profile_adoption=ExecutionProfileAdoptionIntent(
                idempotency_key="generated-current-prompt",
                reason="Select the current child prompt.",
                requested_by=ResolutionActor(
                    subject="operator",
                    source=ResolutionActorSource.REQUEST,
                ),
            ),
        )

        first, replay = await asyncio.gather(
            collect(child_app.fork_session(request)),
            collect(child_app.fork_session(request)),
        )

        assert [event.type for event in first] == [EventType.SESSION_FORKED]
        assert [event.id for event in replay] == [event.id for event in first]
        child_id = first[0].session_id
        assert child_id is not None
        children = (
            await store.list_sessions(
                SessionQuery(parent_session_id="generated-current-prompt-source", limit=10)
            )
        ).sessions
        assert [child.id for child in children] == [child_id]
        receipt = session_prompt_anatomy_transition(children[0])
        assert receipt is not None
        assert receipt.descendant_session_id == child_id

    asyncio.run(scenario())


def test_fork_profile_inheritance_rejects_environment_override() -> None:
    with pytest.raises(ValueError, match="agent/model/environment overrides"):
        ForkSessionRequest(
            source_session_id="fork-environment-source",
            session_id="fork-environment-child",
            environment_name="production",
        )


def test_fork_current_child_environment_change_requires_authority_review() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        source_app = CayuApp(session_store=store, enable_logging=False)
        source_app.register_provider(
            ScriptedModelProvider(
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                name="fake",
            ),
            default=True,
        )
        source_app.register_environment(
            Environment(EnvironmentSpec(name="restricted")),
            default=True,
        )
        source_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await collect(
            source_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="fork-environment-authority-source",
                    environment_name="restricted",
                    messages=[Message.text("user", "establish restricted authority")],
                )
            )
        )

        policy = RecordingAdoptionPolicy()
        fork_app = CayuApp(
            session_store=store,
            enable_logging=False,
            execution_profile_policy=policy,
        )
        fork_app.register_provider(ScriptedModelProvider([], name="fake"), default=True)
        fork_app.register_environment(
            Environment(EnvironmentSpec(name="production")),
            default=True,
        )
        fork_app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await collect(
            fork_app.fork_session(
                ForkSessionRequest(
                    source_session_id="fork-environment-authority-source",
                    session_id="fork-environment-authority-child",
                    environment_name="production",
                    execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                    profile_adoption=ExecutionProfileAdoptionIntent(
                        idempotency_key="fork-environment-authority",
                        reason="Authorize the production environment.",
                        requested_by=ResolutionActor(
                            subject="operator",
                            source=ResolutionActorSource.REQUEST,
                        ),
                    ),
                )
            )
        )

        assert len(policy.requests) == 1
        assert policy.requests[0].authority_review_required is True
        child = await store.load("fork-environment-authority-child")
        assert child is not None
        assert child.environment_name == "production"

    asyncio.run(scenario())


@pytest.mark.parametrize("changed_authority", ["hook", "tool_policy"])
def test_fork_current_child_reviews_decision_bearing_agent_authority(
    changed_authority: str,
) -> None:
    class DenyAllPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            del request
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="Denied.")

    async def scenario() -> None:
        store = InMemorySessionStore()
        source_app = CayuApp(session_store=store, enable_logging=False)
        source_app.register_provider(
            ScriptedModelProvider(
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                name="fake",
            ),
            default=True,
        )
        source_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        source_id = f"fork-agent-authority-source-{changed_authority}"
        await collect(
            source_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "establish agent authority")],
                )
            )
        )

        policy = RecordingAdoptionPolicy()
        fork_app = CayuApp(
            session_store=store,
            enable_logging=False,
            execution_profile_policy=policy,
        )
        fork_app.register_provider(ScriptedModelProvider([], name="fake"), default=True)
        fork_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            runtime_hooks=(
                [RecordingCompletionHook("new-hook")] if changed_authority == "hook" else None
            ),
            tool_policy=(DenyAllPolicy() if changed_authority == "tool_policy" else None),
        )
        await collect(
            fork_app.fork_session(
                ForkSessionRequest(
                    source_session_id=source_id,
                    session_id=f"fork-agent-authority-child-{changed_authority}",
                    execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                    profile_adoption=ExecutionProfileAdoptionIntent(
                        idempotency_key=f"fork-agent-authority-{changed_authority}",
                        reason="Authorize the changed agent governance.",
                        requested_by=ResolutionActor(
                            subject="operator",
                            source=ResolutionActorSource.REQUEST,
                        ),
                    ),
                )
            )
        )

        assert len(policy.requests) == 1
        assert policy.requests[0].authority_review_required is True

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_fork_current_child_profile_preserves_an_explicitly_empty_prompt(
    store_kind: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "fork-empty-current-prompt.sqlite")
        )
        source_app = CayuApp(session_store=store, enable_logging=False)
        source_app.register_provider(
            ScriptedModelProvider(
                [
                    ModelStreamEvent.text_delta("source complete"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                name="fake",
            ),
            default=True,
        )
        source_app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                system_prompt="parent-only system prompt",
            )
        )
        await collect(
            source_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="empty-prompt-source",
                    messages=[Message.text("user", "create source")],
                )
            )
        )

        policy = RecordingAdoptionPolicy()
        child_app = CayuApp(
            session_store=store,
            enable_logging=False,
            execution_profile_policy=policy,
        )
        child_app.register_provider(
            ScriptedModelProvider(
                [
                    ModelStreamEvent.text_delta("child complete"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                name="fake",
            ),
            default=True,
        )
        child_app.register_environment(Environment(EnvironmentSpec(name="body")), default=True)
        child_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await collect(
            child_app.fork_session(
                ForkSessionRequest(
                    source_session_id="empty-prompt-source",
                    session_id="empty-prompt-child",
                    environment_name="body",
                    copy_checkpoint=False,
                    system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
                    execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                    profile_adoption=ExecutionProfileAdoptionIntent(
                        idempotency_key="fork-empty-current-prompt",
                        reason="Explicitly remove the parent prompt for the child.",
                        requested_by=ResolutionActor(
                            subject="operator",
                            source=ResolutionActorSource.REQUEST,
                        ),
                    ),
                )
            )
        )

        child = await store.load("empty-prompt-child")
        assert child is not None
        relationship = session_fork_profile_relationship(child)
        assert relationship is not None
        assert len(policy.requests) == 1
        assert relationship.selected_profile == policy.requests[0].candidate_profile
        assert relationship.source_profile.component(
            ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION
        ) != relationship.selected_profile.component(
            ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION
        )
        transcript = await store.load_transcript(child.id)
        assert all(message.role.value != "system" for message in transcript)

        await store.delete_session("empty-prompt-source")
        child = await store.load("empty-prompt-child")
        assert child is not None
        assert child.parent_session_id is None
        assert session_prompt_anatomy_transition(child) is not None

        resumed = await collect(
            child_app.resume(
                ResumeRequest(
                    session_id=child.id,
                    messages=[Message.text("user", "continue without a system prompt")],
                )
            )
        )
        assert any(event.type is EventType.SESSION_COMPLETED for event in resumed)
        assert len(policy.requests) == 1
        close = getattr(store, "close", None)
        if close is not None:
            await close()

    asyncio.run(scenario())


def test_fork_profile_relationship_uses_redacted_policy_decision_evidence(
    caplog,
    capsys,
) -> None:
    secret = "fork-policy-private-reason-canary"

    class SecretReasonPolicy(ExecutionProfilePolicy):
        @property
        def identity(self) -> str:
            return "test:fork-secret-reason-policy:v1"

        async def decide(
            self,
            request: ExecutionProfilePolicyRequest,
        ) -> ExecutionProfilePolicyResult:
            return ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason=f"Approved with private policy material {secret}.",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            )

    async def scenario() -> None:
        store = InMemorySessionStore()
        source_provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("source complete"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            name="fake",
        )
        source_app = CayuApp(session_store=store, enable_logging=False)
        source_app.register_provider(source_provider, default=True)
        source_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingExternalTool(description="Original tool")],
        )
        await collect(
            source_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="secret-policy-fork-source",
                    messages=[Message.text("user", "create source")],
                )
            )
        )

        fork_app = CayuApp(
            session_store=store,
            execution_profile_policy=SecretReasonPolicy(),
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        fork_app.register_provider(ScriptedModelProvider([], name="fake"), default=True)
        fork_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingExternalTool(description="Replacement tool")],
        )
        public_events = await collect(
            fork_app.fork_session(
                ForkSessionRequest(
                    source_session_id="secret-policy-fork-source",
                    session_id="secret-policy-fork-child",
                    execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                    profile_adoption=ExecutionProfileAdoptionIntent(
                        idempotency_key="secret-policy-fork",
                        reason="Adopt the reviewed child profile.",
                        requested_by=ResolutionActor(
                            subject="operator",
                            source=ResolutionActorSource.REQUEST,
                        ),
                    ),
                )
            )
        )

        child = await store.load("secret-policy-fork-child")
        assert child is not None
        relationship = session_fork_profile_relationship(child)
        assert relationship is not None and relationship.decision is not None
        stored_events = await store.load_events(child.id)
        assert "[REDACTED_SECRET]" in relationship.decision.policy_reason
        assert secret not in repr(child.metadata)
        assert secret not in repr(stored_events)
        assert secret not in repr(public_events)

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        asyncio.run(scenario())

    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in captured_warnings)


def test_fork_profile_rejection_leaves_parent_and_child_store_unchanged() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        source_provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("source complete"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            name="fake",
        )
        source_app = CayuApp(session_store=store, enable_logging=False)
        source_app.register_provider(source_provider, default=True)
        source_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingExternalTool(description="Original tool")],
        )
        await collect(
            source_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="rejected-profile-source",
                    messages=[Message.text("user", "create source")],
                )
            )
        )
        parent_before = await store.load("rejected-profile-source")
        assert parent_before is not None

        rejecting_app = CayuApp(session_store=store, enable_logging=False)
        rejecting_app.register_provider(ScriptedModelProvider([], name="fake"), default=True)
        rejecting_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingExternalTool(description="Broader unapproved tool")],
        )
        with pytest.raises(ExecutionProfileAdoptionRejected):
            await collect(
                rejecting_app.fork_session(
                    ForkSessionRequest(
                        source_session_id=parent_before.id,
                        session_id="rejected-profile-child",
                        execution_profile_selection=(ForkExecutionProfileSelection.CURRENT_CHILD),
                        profile_adoption=ExecutionProfileAdoptionIntent(
                            idempotency_key="rejected-profile-fork",
                            reason="Attempt an unapproved authority change.",
                            requested_by=ResolutionActor(
                                subject="operator",
                                source=ResolutionActorSource.REQUEST,
                            ),
                        ),
                    )
                )
            )

        assert await store.load("rejected-profile-child") is None
        assert await store.load(parent_before.id) == parent_before

    asyncio.run(scenario())


def test_fork_inheritance_preserves_unavailable_parent_profile_identity() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        unavailable = build_execution_profile_identity(
            runtime_name="cayu",
            runtime_version=None,
            provider_name="fake",
            model="fake-model",
            durable_system_prompt=None,
            direct_tools=(),
        )
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="unavailable-profile-source",
                messages=[],
            ),
            identity=SessionIdentity(
                provider_name="fake",
                model="fake-model",
                execution_profile=unavailable,
            ),
        )
        source = await store.update_status(source.id, SessionStatus.COMPLETED)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider([], name="fake"), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await collect(
            app.fork_session(
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id="unavailable-profile-child",
                )
            )
        )
        child = await store.load("unavailable-profile-child")
        assert child is not None
        relationship = session_fork_profile_relationship(child)
        assert relationship is not None
        assert relationship.source_profile == unavailable
        assert relationship.selected_profile == unavailable

    asyncio.run(scenario())


def test_fork_current_child_rejects_unavailable_identity_before_child_creation(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="unavailable-current-profile-source",
                messages=[],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        source = await store.update_status(source.id, SessionStatus.COMPLETED)
        app = CayuApp(
            session_store=store,
            execution_profile_policy=RecordingAdoptionPolicy(),
            enable_logging=False,
        )
        app.register_provider(ScriptedModelProvider([], name="fake"), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        monkeypatch.setattr(session_engine_module, "_runtime_version", lambda: None)

        with pytest.raises(
            RuntimeError,
            match="Current-child profile selection has unavailable required components: runtime",
        ):
            await collect(
                app.fork_session(
                    ForkSessionRequest(
                        source_session_id=source.id,
                        session_id="unavailable-current-profile-child",
                        execution_profile_selection=(ForkExecutionProfileSelection.CURRENT_CHILD),
                        profile_adoption=ExecutionProfileAdoptionIntent(
                            idempotency_key="unavailable-current-profile",
                            reason="Exercise unavailable child identity.",
                            requested_by=ResolutionActor(
                                subject="operator",
                                source=ResolutionActorSource.REQUEST,
                            ),
                        ),
                    )
                )
            )

        assert await store.load("unavailable-current-profile-child") is None
        assert await store.load(source.id) == source
        assert await store.load_events(source.id) == []

    asyncio.run(scenario())


def test_partial_fork_cannot_drop_inherited_durable_system_projection() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("source complete"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            name="fake",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                system_prompt="Durable parent instructions.",
            )
        )
        await collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="partial-profile-source",
                    messages=[Message.text("user", "create source")],
                )
            )
        )
        parent_before = await store.load("partial-profile-source")
        assert parent_before is not None

        with pytest.raises(
            ValueError,
            match="must retain the complete durable system projection",
        ):
            await collect(
                app.fork_session(
                    ForkSessionRequest(
                        source_session_id=parent_before.id,
                        session_id="partial-profile-child-invalid",
                        transcript_cursor=0,
                        copy_checkpoint=False,
                    )
                )
            )

        assert await store.load("partial-profile-child-invalid") is None
        assert await store.load(parent_before.id) == parent_before

        await collect(
            app.fork_session(
                ForkSessionRequest(
                    source_session_id=parent_before.id,
                    session_id="partial-profile-child-valid",
                    transcript_cursor=1,
                    copy_checkpoint=False,
                )
            )
        )
        child = await store.load("partial-profile-child-valid")
        assert child is not None
        parent_profile = execution_profile_from_session_metadata(parent_before.metadata)
        assert execution_profile_from_session_metadata(child.metadata) == parent_profile
        transcript = await store.load_transcript(child.id)
        assert len(transcript) == 1
        assert transcript[0].role.value == "system"

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_fork_inherits_active_parent_invocation_as_independent_child_baseline(
    store_kind: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "fork-active-parent-profile.sqlite")
        )
        tool = RecordingExternalTool(description="Active invocation tool")
        baseline_identity = profiled_session_identity(
            provider_name="fake",
            model="fake-model",
        )
        active_identity = profiled_session_identity(
            provider_name="fake",
            model="fake-model",
            tools=[tool],
        )
        baseline = baseline_identity.execution_profile
        active = active_identity.execution_profile
        assert baseline is not None
        assert active is not None
        assert baseline != active
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=f"active-parent-source-{store_kind}",
                messages=[],
            ),
            identity=baseline_identity,
        )
        source = await store.transition_status(
            source.id,
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )
        checkpoint = checkpoint_with_active_invocation_execution_profile(
            {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION},
            session_id=source.id,
            interaction_id=f"active-parent-interaction-{store_kind}",
            run_epoch=source.run_epoch,
            profile=active,
        )
        await store.checkpoint(source.id, checkpoint)
        source = await store.transition_status(
            source.id,
            from_statuses={SessionStatus.RUNNING},
            to_status=SessionStatus.COMPLETED,
        )

        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("child resumed"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            name="fake",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
        )
        child_id = f"active-parent-child-{store_kind}"
        await collect(
            app.fork_session(
                ForkSessionRequest(
                    source_session_id=source.id,
                    session_id=child_id,
                )
            )
        )

        unchanged_parent = await store.load(source.id)
        child = await store.load(child_id)
        assert unchanged_parent is not None
        assert child is not None
        assert execution_profile_from_session_metadata(unchanged_parent.metadata) == baseline
        assert execution_profile_from_session_metadata(child.metadata) == active
        relationship = session_fork_profile_relationship(child)
        assert relationship is not None
        assert relationship.source_profile_source.value == "active_invocation"
        assert relationship.source_profile == active
        assert relationship.selected_profile == active
        assert relationship.source_active_interaction_id == (
            f"active-parent-interaction-{store_kind}"
        )

        resumed = await collect(
            app.resume(
                ResumeRequest(
                    session_id=child_id,
                    messages=[Message.text("user", "continue under the child baseline")],
                )
            )
        )
        assert any(event.type is EventType.SESSION_COMPLETED for event in resumed)
        close = getattr(store, "close", None)
        if close is not None:
            await close()

    asyncio.run(scenario())


def test_fork_rejects_missing_parent_profile_before_child_creation() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="missing-parent-profile-source",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        source = await store.update_status(source.id, SessionStatus.COMPLETED)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider([], name="fake"), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        parent_before = await store.load(source.id)

        with pytest.raises(ValueError, match="durable execution-profile identity"):
            await collect(
                app.fork_session(
                    ForkSessionRequest(
                        source_session_id=source.id,
                        session_id="missing-parent-profile-child",
                    )
                )
            )

        assert await store.load("missing-parent-profile-child") is None
        assert await store.load(source.id) == parent_before
        assert await store.load_events(source.id) == []

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_profiled_fork_resume_freezes_profile_through_approval_continuation(
    store_kind: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        source_id = "legacy-unprofiled-fork-source"
        child_id = "legacy-unprofiled-fork-child"
        drift_child_id = "legacy-unprofiled-fork-drift-child"
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "unprofiled-fork-profile.sqlite")
        )
        tool = RecordingExternalTool(description="Original fork tool.")
        provider = ScriptedModelProvider(
            [
                (
                    ModelStreamEvent.text_delta("source complete"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ),
                (
                    ModelStreamEvent.tool_call(
                        id="fork-call",
                        name=tool.spec.name,
                        arguments={"value": "child"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ),
                (
                    ModelStreamEvent.text_delta("child complete"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ),
                (
                    ModelStreamEvent.tool_call(
                        id="fork-drift-call",
                        name=tool.spec.name,
                        arguments={"value": "drift child"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ),
            ],
            name="fake",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=RequireApprovalPolicy(),
        )
        await collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=source_id,
                    messages=[Message.text("user", "create a fork source")],
                )
            )
        )

        await collect(
            app.fork_session(ForkSessionRequest(source_session_id=source_id, session_id=child_id))
        )
        child = await store.load(child_id)
        assert child is not None
        assert EXECUTION_PROFILE_METADATA_KEY in child.metadata
        paused = await collect(
            app.resume(
                ResumeRequest(
                    session_id=child_id,
                    messages=[Message.text("user", "run the fork tool")],
                )
            )
        )
        approval_event = next(
            event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        pending = approval_event.payload["approval"]
        assert isinstance(pending, dict)
        checkpoint = await store.load_checkpoint(child_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        pending_approval = _approval_support.pending_approval_from_checkpoint(checkpoint)
        assert active_profile is not None
        assert pending_approval is not None
        assert pending_approval.execution_profile_fingerprint == active_profile.profile.fingerprint

        completed = await collect(
            app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=child_id,
                    approval_id=pending["approval_id"],
                    tool_round_id=pending["tool_round_id"],
                    tool_call_id=pending["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        )
        assert any(event.type is EventType.SESSION_COMPLETED for event in completed)
        assert tool.calls == [{"value": "child"}]

        await collect(
            app.fork_session(
                ForkSessionRequest(source_session_id=source_id, session_id=drift_child_id)
            )
        )
        drift_paused = await collect(
            app.resume(
                ResumeRequest(
                    session_id=drift_child_id,
                    messages=[Message.text("user", "run under frozen fork profile")],
                )
            )
        )
        drift_approval_event = next(
            event for event in drift_paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        drift_pending = drift_approval_event.payload["approval"]
        assert isinstance(drift_pending, dict)
        replacement_provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("must not dispatch"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            name="fake",
        )
        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(replacement_provider, default=True)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingExternalTool(description="Changed fork tool.")],
            tool_policy=RequireApprovalPolicy(),
        )
        with pytest.raises(ExecutionProfileMismatchError):
            await collect(
                replacement_app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=drift_child_id,
                        approval_id=drift_pending["approval_id"],
                        tool_round_id=drift_pending["tool_round_id"],
                        tool_call_id=drift_pending["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            )
        assert replacement_provider.requests == []
        close = getattr(store, "close", None)
        if close is not None:
            await close()

    asyncio.run(scenario())


def test_approval_continuation_rejects_changed_invocation_profile_before_work() -> None:
    async def scenario() -> None:
        session_id = "active-profile-approval-drift"
        store = InMemorySessionStore()
        original_tool = RecordingExternalTool(description="Original governed effect.")
        original_provider = ScriptedModelProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call-1",
                    name=original_tool.spec.name,
                    arguments={"value": "original"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            name="fake",
        )
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(original_provider, default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[original_tool],
            tool_policy=RequireApprovalPolicy(),
        )

        paused = await collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run the effect")],
                )
            )
        )
        approval_events = [
            event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        ]
        assert len(approval_events) == 1, [
            (event.type, event.payload)
            for event in paused
            if event.type in {EventType.SESSION_FAILED, EventType.INTERACTION_FAILED}
        ]
        approval_event = approval_events[0]
        pending = approval_event.payload["approval"]
        assert isinstance(pending, dict)
        checkpoint = await store.load_checkpoint(session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        pending_round = _tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
        pending_approval = _approval_support.pending_approval_from_checkpoint(checkpoint)
        assert active_profile is not None
        assert pending_round is not None
        assert pending_approval is not None
        assert pending_round.execution_profile_fingerprint == active_profile.profile.fingerprint
        assert pending_approval.execution_profile_fingerprint == active_profile.profile.fingerprint
        before = await store.load(session_id)
        assert before is not None
        assert before.status is SessionStatus.INTERRUPTED

        replacement_tool = RecordingExternalTool(description="Replacement governed effect.")
        replacement_provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("continued"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            name="fake",
        )
        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(replacement_provider, default=True)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[replacement_tool],
            tool_policy=RequireApprovalPolicy(),
        )

        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await collect(
                replacement_app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=pending["approval_id"],
                        tool_round_id=pending["tool_round_id"],
                        tool_call_id=pending["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.DIRECT_TOOLS,
        )
        assert replacement_tool.calls == []
        assert replacement_provider.requests == []
        after = await store.load(session_id)
        assert after is not None
        assert after.status is before.status
        assert after.run_epoch == before.run_epoch

    asyncio.run(scenario())


def test_approval_continuation_rejects_request_loop_policy_drift() -> None:
    async def scenario() -> None:
        session_id = "active-profile-approval-request-policy-drift"
        store = InMemorySessionStore()
        tool = RecordingExternalTool(description="Governed effect.")
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(
            ScriptedModelProvider(
                [
                    ModelStreamEvent.tool_call(
                        id="call-1",
                        name=tool.spec.name,
                        arguments={"value": "original"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                name="fake",
            ),
            default=True,
        )
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=RequireApprovalPolicy(),
        )
        paused = await collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run the effect")],
                    loop_policies=(VersionedRequestLoopPolicy("1"),),
                )
            )
        )
        approval_event = next(
            event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        pending = approval_event.payload["approval"]
        assert isinstance(pending, dict)

        replacement_tool = RecordingExternalTool(description="Governed effect.")
        replacement_provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("continued"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            name="fake",
        )
        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(replacement_provider, default=True)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[replacement_tool],
            tool_policy=RequireApprovalPolicy(),
        )

        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await collect(
                replacement_app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=pending["approval_id"],
                        tool_round_id=pending["tool_round_id"],
                        tool_call_id=pending["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                        loop_policies=(VersionedRequestLoopPolicy("2"),),
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.INVOCATION_POLICIES,
        )
        assert replacement_tool.calls == []
        assert replacement_provider.requests == []

    asyncio.run(scenario())


async def _assert_approval_continuation_reconstructs_profile_once(
    store: SessionStore,
    *,
    suffix: str,
) -> None:
    session_id = f"active-profile-approval-restart-{suffix}-{uuid4().hex}"
    original_tool = RecordingExternalTool(description="Governed effect.")
    original_app = CayuApp(session_store=store, enable_logging=False)
    original_app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call-1",
                    name=original_tool.spec.name,
                    arguments={"value": "original"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            name="fake",
        ),
        default=True,
    )
    original_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[original_tool],
        tool_policy=RequireApprovalPolicy(),
    )
    paused = await collect(
        original_app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run the effect")],
            )
        )
    )
    approval_event = next(
        event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    pending = approval_event.payload["approval"]
    assert isinstance(pending, dict)

    replacement_tool = RecordingExternalTool(description="Governed effect.")
    replacement_provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("continued"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        name="fake",
    )
    replacement_app = CayuApp(session_store=store, enable_logging=False)
    replacement_app.register_provider(replacement_provider, default=True)
    replacement_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[replacement_tool],
        tool_policy=RequireApprovalPolicy(),
    )
    request = ToolApprovalRequest(
        session_id=session_id,
        approval_id=pending["approval_id"],
        tool_round_id=pending["tool_round_id"],
        tool_call_id=pending["tool_call_id"],
        decision=ToolApprovalDecision.APPROVE,
    )

    completed = await collect(replacement_app.resolve_tool_approval(request))
    assert any(event.type is EventType.SESSION_COMPLETED for event in completed)
    assert replacement_tool.calls == [{"value": "original"}]
    assert len(replacement_provider.requests) == 1

    replayed = await collect(replacement_app.resolve_tool_approval(request))
    assert replayed
    assert replacement_tool.calls == [{"value": "original"}]
    assert len(replacement_provider.requests) == 1
    session = await store.load(session_id)
    assert session is not None
    assert session.status is SessionStatus.COMPLETED


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_approval_continuation_reconstructs_profile_once_in_local_stores(
    store_kind: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "active-profile-approval.sqlite")
        )
        try:
            await _assert_approval_continuation_reconstructs_profile_once(
                store,
                suffix=store_kind,
            )
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    asyncio.run(scenario())


def test_approval_continuation_reconstructs_profile_once_in_postgres(
    postgres_dsn: str,
) -> None:
    async def scenario() -> None:
        from cayu import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await _assert_approval_continuation_reconstructs_profile_once(
                store,
                suffix="postgres",
            )
        finally:
            await store.close()

    asyncio.run(scenario())


def test_new_resume_atomically_rebinds_active_profile_to_new_interaction_epoch() -> None:
    async def scenario() -> None:
        session_id = "active-profile-new-resume"
        store = InMemorySessionStore()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(
            ScriptedModelProvider(
                [
                    ModelStreamEvent.text_delta("first"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                name="fake",
            ),
            default=True,
        )
        original_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "first")],
                )
            )
        )
        first_checkpoint = await store.load_checkpoint(session_id)
        first_profile = active_invocation_execution_profile_from_checkpoint(first_checkpoint)
        assert first_profile is not None

        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(
            ScriptedModelProvider(
                [
                    ModelStreamEvent.text_delta("second"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                name="fake",
            ),
            default=True,
        )
        replacement_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        resumed = await collect(
            replacement_app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "second")],
                )
            )
        )
        interaction_started = next(
            event for event in resumed if event.type is EventType.INTERACTION_STARTED
        )
        second_checkpoint = await store.load_checkpoint(session_id)
        second_profile = active_invocation_execution_profile_from_checkpoint(second_checkpoint)
        session = await store.load(session_id)
        assert second_profile is not None
        assert session is not None
        assert second_profile.interaction_id == interaction_started.interaction_id
        assert second_profile.interaction_id != first_profile.interaction_id
        assert second_profile.run_epoch == session.run_epoch - 1
        assert second_profile.profile == first_profile.profile

    asyncio.run(scenario())


def test_user_input_continuation_rejects_changed_profile_before_provider_work() -> None:
    async def scenario() -> None:
        session_id = "active-profile-user-input-drift"
        store = InMemorySessionStore()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(
            ScriptedModelProvider(
                [
                    ModelStreamEvent.tool_call(
                        id="call-input",
                        name="ask_user",
                        arguments={"question": "Continue?"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                name="fake",
            ),
            default=True,
        )
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool()],
        )
        paused = await collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "ask")],
                )
            )
        )
        awaiting = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )
        checkpoint = await store.load_checkpoint(session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        pending_input = pending_user_input_from_checkpoint(checkpoint)
        assert active_profile is not None
        assert pending_input is not None
        assert pending_input.execution_profile_fingerprint == active_profile.profile.fingerprint

        replacement_provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("continued"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            name="fake",
        )
        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(replacement_provider, default=True)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), RecordingExternalTool(description="New tool.")],
        )
        with pytest.raises(ExecutionProfileMismatchError):
            await collect(
                replacement_app.resolve_user_input(
                    UserInputResponse(
                        session_id=session_id,
                        input_id=awaiting.payload["input_id"],
                        answer="yes",
                    )
                )
            )
        assert replacement_provider.requests == []

    asyncio.run(scenario())


def test_profile_adoption_is_rejected_while_external_effect_is_active() -> None:
    async def scenario() -> None:
        session_id = f"active-profile-effect-boundary-{uuid4().hex}"
        store = InMemorySessionStore()
        original_tool = BlockingExternalTool(description="Original governed effect.")
        provider = ScriptedModelProvider(
            [
                (
                    ModelStreamEvent.tool_call(
                        id="call-active-effect",
                        name=original_tool.spec.name,
                        arguments={"value": "original"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ),
                (
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ),
            ],
            name="fake",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[original_tool],
        )

        run_task = asyncio.create_task(
            collect(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "run the effect")],
                    )
                )
            )
        )
        await asyncio.wait_for(original_tool.started.wait(), timeout=5.0)

        replacement_app = CayuApp(enable_logging=False)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingExternalTool(description="Replacement governed effect.")],
        )
        app._agents["assistant"] = replacement_app._agents["assistant"]
        intent = ExecutionProfileAdoptionIntent(
            idempotency_key=f"active-effect-adoption-{uuid4().hex}",
            reason="Adopt the replacement only after the historical effect is terminal.",
            requested_by=ResolutionActor(
                subject="test-maintainer",
                source=ResolutionActorSource.REQUEST,
            ),
        )
        try:
            with pytest.raises(SessionStatusConflict, match="running"):
                await collect(
                    app.resume(
                        ResumeRequest(
                            session_id=session_id,
                            messages=[Message.text("user", "adopt now")],
                            profile_adoption=intent,
                        )
                    )
                )
        finally:
            original_tool.release.set()

        events = await run_task
        assert any(event.type is EventType.SESSION_COMPLETED for event in events)
        assert original_tool.calls == [{"value": "original"}]

    asyncio.run(scenario())


async def _assert_profile_adoption_waits_for_terminal_hook_release(
    store: SessionStore,
    *,
    suffix: str,
) -> None:
    session_id = f"active-profile-terminal-hook-boundary-{suffix}-{uuid4().hex}"
    hook = BlockingCompletionHook("blocking-terminal-profile-hook")
    original_provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("original complete"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        name="fake",
    )
    original_app = CayuApp(session_store=store, enable_logging=False)
    original_app.register_provider(original_provider, default=True)
    original_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        runtime_hooks=[hook],
    )
    run_task = asyncio.create_task(
        collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "finish under the original profile")],
                )
            )
        )
    )
    await asyncio.wait_for(hook.started.wait(), timeout=5.0)

    terminal = await store.load(session_id)
    checkpoint = await store.load_checkpoint(session_id)
    active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    assert terminal is not None
    assert terminal.status is SessionStatus.COMPLETED
    assert active_profile is not None
    assert active_profile.run_epoch == terminal.run_epoch

    replacement_tool = RecordingExternalTool(description="Replacement governed effect.")
    replacement_provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("replacement complete"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        name="fake",
    )
    policy = RecordingAdoptionPolicy()
    replacement_app = CayuApp(
        session_store=store,
        execution_profile_policy=policy,
        enable_logging=False,
    )
    replacement_app.register_provider(replacement_provider, default=True)
    replacement_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[replacement_tool],
    )
    intent = ExecutionProfileAdoptionIntent(
        idempotency_key=f"terminal-hook-adoption-{suffix}-{uuid4().hex}",
        reason="Adopt only after the original terminal hook settles.",
        requested_by=ResolutionActor(
            subject="test-maintainer",
            source=ResolutionActorSource.REQUEST,
        ),
    )
    try:
        with pytest.raises(SessionRunFenced, match="terminal hooks or trailing cleanup"):
            await collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "adopt replacement")],
                        profile_adoption=intent,
                    )
                )
            )
        assert policy.requests == []
        assert replacement_provider.requests == []
    finally:
        hook.release.set()

    completed = await run_task
    assert any(event.type is EventType.SESSION_COMPLETED for event in completed)
    released = await store.load(session_id)
    assert released is not None
    assert released.run_epoch == active_profile.run_epoch + 1

    resumed = await collect(
        replacement_app.resume(
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "adopt replacement")],
                profile_adoption=intent,
            )
        )
    )
    assert any(event.type is EventType.SESSION_COMPLETED for event in resumed)
    assert len(policy.requests) == 1
    assert len(replacement_provider.requests) == 1


async def _assert_profile_adoption_waits_for_deferred_environment_cleanup(
    store: SessionStore,
    *,
    suffix: str,
) -> None:
    session_id = f"active-profile-deferred-cleanup-boundary-{suffix}-{uuid4().hex}"
    hook = BlockingCompletionHook("deferred-cleanup-profile-hook")
    original_provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("original complete"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        name="fake",
    )
    original_app = CayuApp(session_store=store, enable_logging=False)
    original_app.register_provider(original_provider, default=True)
    original_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        runtime_hooks=[hook],
    )
    run_task = asyncio.create_task(
        collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "finish before cleanup settles")],
                )
            )
        )
    )
    await asyncio.wait_for(hook.started.wait(), timeout=5.0)

    terminal = await store.load(session_id)
    checkpoint = await store.load_checkpoint(session_id)
    active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    assert terminal is not None
    assert terminal.status is SessionStatus.COMPLETED
    assert active_profile is not None
    assert active_profile.run_epoch == terminal.run_epoch

    retry_started = asyncio.Event()
    retry_release = asyncio.Event()

    async def fail_deferred_cleanup() -> None:
        raise RuntimeError("initial deferred cleanup failed")

    def retry_deferred_cleanup() -> asyncio.Task[None]:
        async def settle_deferred_cleanup() -> None:
            retry_started.set()
            await retry_release.wait()

        return asyncio.create_task(settle_deferred_cleanup())

    cleanup_task = asyncio.create_task(fail_deferred_cleanup())
    register_environment_factory_cleanup_retry(cleanup_task, retry_deferred_cleanup)
    lifecycle = original_app._environment_lifecycle
    lifecycle._deferred_factory_cleanup_tasks[session_id] = cleanup_task
    cleanup_task.add_done_callback(
        lambda completed: lifecycle._deferred_factory_cleanup_completed(
            session_id,
            completed,
        )
    )

    replacement_provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("replacement complete"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        name="fake",
    )
    policy = RecordingAdoptionPolicy()
    replacement_app = CayuApp(
        session_store=store,
        execution_profile_policy=policy,
        enable_logging=False,
    )
    replacement_app.register_provider(replacement_provider, default=True)
    replacement_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[RecordingExternalTool(description="Replacement governed effect.")],
    )
    intent = ExecutionProfileAdoptionIntent(
        idempotency_key=f"deferred-cleanup-adoption-{suffix}-{uuid4().hex}",
        reason="Adopt only after original environment cleanup settles.",
        requested_by=ResolutionActor(
            subject="test-maintainer",
            source=ResolutionActorSource.REQUEST,
        ),
    )
    retry_task: asyncio.Task[None] | None = None
    try:
        [cleanup_failure] = await asyncio.gather(
            cleanup_task,
            return_exceptions=True,
        )
        assert isinstance(cleanup_failure, RuntimeError)
        hook.release.set()
        completed = await run_task
        assert any(event.type is EventType.SESSION_COMPLETED for event in completed)
        fence_release_task = lifecycle._deferred_run_fence_release_tasks[
            (session_id, active_profile.run_epoch)
        ]

        drain_task = asyncio.create_task(original_app.drain_environment_cleanups(timeout_s=0.02))
        await asyncio.wait_for(retry_started.wait(), timeout=5.0)
        retry_task = lifecycle._deferred_factory_cleanup_tasks[session_id]
        assert retry_task is not cleanup_task
        assert await drain_task is False

        fenced = await store.load(session_id)
        assert fenced is not None
        assert fenced.run_epoch == active_profile.run_epoch
        with pytest.raises(SessionRunFenced, match="terminal hooks or trailing cleanup"):
            await collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "adopt replacement")],
                        profile_adoption=intent,
                    )
                )
            )
        assert policy.requests == []
        assert replacement_provider.requests == []

        retry_release.set()
        await retry_task
        await asyncio.wait_for(fence_release_task, timeout=5.0)
        assert session_id not in lifecycle._deferred_factory_cleanup_tasks
        released = await store.load(session_id)
        assert released is not None
        assert released.run_epoch == active_profile.run_epoch + 1

        resumed = await collect(
            replacement_app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "adopt replacement")],
                    profile_adoption=intent,
                )
            )
        )
        assert any(event.type is EventType.SESSION_COMPLETED for event in resumed)
        assert len(policy.requests) == 1
        assert len(replacement_provider.requests) == 1
    finally:
        hook.release.set()
        retry_release.set()
        if not run_task.done():
            await run_task
        if not cleanup_task.done():
            await asyncio.gather(cleanup_task, return_exceptions=True)
        if retry_task is not None and not retry_task.done():
            await retry_task


def test_repaired_failed_deferred_release_does_not_poison_next_invocation() -> None:
    async def scenario() -> None:
        session_id = "active-profile-failed-deferred-release"
        store = FailFirstRunFenceReleaseStore()
        hook = BlockingCompletionHook("failed-deferred-release-hook")
        provider = ScriptedModelProvider(
            [
                (
                    ModelStreamEvent.text_delta("first invocation complete"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ),
                (
                    ModelStreamEvent.text_delta("second invocation complete"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ),
            ],
            name="fake",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            runtime_hooks=[hook],
        )

        run_task = asyncio.create_task(
            collect(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "finish the first invocation")],
                    )
                )
            )
        )
        await asyncio.wait_for(hook.started.wait(), timeout=5.0)
        checkpoint = await store.load_checkpoint(session_id)
        first_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert first_profile is not None

        cleanup_release = asyncio.Event()

        async def settle_deferred_cleanup() -> None:
            await cleanup_release.wait()

        cleanup_task = asyncio.create_task(settle_deferred_cleanup())
        lifecycle = app._environment_lifecycle
        lifecycle._deferred_factory_cleanup_tasks[session_id] = cleanup_task
        cleanup_task.add_done_callback(
            lambda completed: lifecycle._deferred_factory_cleanup_completed(
                session_id,
                completed,
            )
        )

        hook.release.set()
        completed = await run_task
        assert any(event.type is EventType.SESSION_COMPLETED for event in completed)
        first_release_key = (session_id, first_profile.run_epoch)
        first_release_task = lifecycle._deferred_run_fence_release_tasks[first_release_key]
        cleanup_release.set()
        await cleanup_task
        [release_failure] = await asyncio.gather(
            first_release_task,
            return_exceptions=True,
        )
        assert isinstance(release_failure, ConnectionError)
        await asyncio.sleep(0)
        assert lifecycle._deferred_run_fence_release_tasks[first_release_key] is first_release_task

        stranded = await store.load(session_id)
        assert stranded is not None
        assert stranded.run_epoch == first_profile.run_epoch
        recovered_page = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.COMPLETED},
                limit=10,
            )
        )
        assert len(recovered_page.results) == 1
        assert recovered_page.results[0].actions == (
            IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_OWNERSHIP,
        )
        assert first_release_key not in lifecycle._deferred_run_fence_release_tasks

        resumed = await collect(
            app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "run a second invocation")],
                )
            )
        )
        assert any(event.type is EventType.SESSION_COMPLETED for event in resumed)
        assert len(provider.requests) == 2
        assert store.release_calls == 3
        assert first_release_key not in lifecycle._deferred_run_fence_release_tasks

        final_session = await store.load(session_id)
        final_checkpoint = await store.load_checkpoint(session_id)
        final_profile = active_invocation_execution_profile_from_checkpoint(final_checkpoint)
        assert final_session is not None
        assert final_profile is not None
        assert final_profile.run_epoch == final_session.run_epoch - 1

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_profile_adoption_waits_for_terminal_hook_release_in_local_stores(
    store_kind: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "active-profile-terminal-hook.sqlite")
        )
        try:
            await _assert_profile_adoption_waits_for_terminal_hook_release(
                store,
                suffix=store_kind,
            )
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    asyncio.run(scenario())


def test_profile_adoption_waits_for_terminal_hook_release_in_postgres(
    postgres_dsn: str,
) -> None:
    async def scenario() -> None:
        from cayu import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await _assert_profile_adoption_waits_for_terminal_hook_release(
                store,
                suffix="postgres",
            )
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_profile_adoption_waits_for_deferred_environment_cleanup(
    store_kind: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "active-profile-deferred-cleanup.sqlite")
        )
        try:
            await _assert_profile_adoption_waits_for_deferred_environment_cleanup(
                store,
                suffix=store_kind,
            )
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    asyncio.run(scenario())


async def _assert_provider_retry_keeps_process_local_resolution_after_mutation(
    store: SessionStore,
    *,
    suffix: str,
) -> None:
    session_id = f"active-profile-live-retry-mutation-{suffix}-{uuid4().hex}"
    provider = MutatingRetryProvider()
    original_tool = RecordingExternalTool(description="Original governed effect.")
    replacement_tool = RecordingExternalTool(description="Replacement governed effect.")
    original_hook = RecordingCompletionHook("original_completion")
    replacement_hook = RecordingCompletionHook("replacement_completion")
    replacement_provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("replacement"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        name="fake",
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[original_tool],
        runtime_hooks=[original_hook],
    )
    model_profiles: list[ExecutionProfileIdentity | None] = []
    tool_profiles: list[ExecutionProfileIdentity | None] = []
    cleanup_profiles: list[ExecutionProfileIdentity | None] = []
    original_model_run_factory = app._model_step_executor.create_run
    original_tool_run_factory = app._tool_round_executor.create_run
    original_terminal_cleanup = app._environment_lifecycle.finalize_terminal_event

    def capture_model_run(**kwargs):
        model_profiles.append(kwargs["execution_profile"])
        return original_model_run_factory(**kwargs)

    def capture_tool_run(**kwargs):
        tool_profiles.append(kwargs["execution_profile"])
        return original_tool_run_factory(**kwargs)

    async def capture_terminal_cleanup(**kwargs):
        cleanup_profiles.append(kwargs["execution_profile"])
        return await original_terminal_cleanup(**kwargs)

    app._model_step_executor.create_run = capture_model_run
    app._tool_round_executor.create_run = capture_tool_run
    app._environment_lifecycle.finalize_terminal_event = capture_terminal_cleanup
    replacement_app = CayuApp(enable_logging=False)
    replacement_app.register_provider(replacement_provider, default=True)
    replacement_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[replacement_tool],
        runtime_hooks=[replacement_hook],
    )

    def mutate_registration() -> None:
        app._agents["assistant"] = replacement_app._agents["assistant"]
        app._providers["fake"] = replacement_app._providers["fake"]

    provider.mutate_registration = mutate_registration

    events = await collect(
        app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "retry then execute")],
                retry_policy=RetryPolicy(
                    max_attempts=2,
                    initial_delay_s=0.0,
                    max_delay_s=0.0,
                ),
            )
        )
    )
    assert any(event.type is EventType.MODEL_RETRY for event in events)
    assert any(event.type is EventType.SESSION_COMPLETED for event in events)
    assert original_tool.calls == [{"value": "frozen"}]
    assert replacement_tool.calls == []
    assert original_hook.sessions == [session_id]
    assert replacement_hook.sessions == []
    assert len(model_profiles) == len(tool_profiles) == len(cleanup_profiles) == 1
    assert model_profiles[0] is tool_profiles[0]
    assert cleanup_profiles[0] is model_profiles[0]
    assert original_hook.before_tool_execution_profiles == [model_profiles[0]]
    assert original_hook.before_tool_execution_profiles[0] is model_profiles[0]
    assert original_hook.after_tool_execution_profiles == [model_profiles[0]]
    assert original_hook.after_tool_execution_profiles[0] is model_profiles[0]
    assert original_hook.execution_profiles == [model_profiles[0]]
    assert original_hook.execution_profiles[0] is model_profiles[0]
    assert len(provider.requests) == 3
    assert replacement_provider.requests == []


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_provider_retry_keeps_process_local_resolution_after_mutation_in_local_stores(
    store_kind: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "active-profile-retry.sqlite")
        )
        try:
            await _assert_provider_retry_keeps_process_local_resolution_after_mutation(
                store,
                suffix=store_kind,
            )
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    asyncio.run(scenario())


def test_provider_retry_keeps_process_local_resolution_after_mutation_in_postgres(
    postgres_dsn: str,
) -> None:
    async def scenario() -> None:
        from cayu import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await _assert_provider_retry_keeps_process_local_resolution_after_mutation(
                store,
                suffix="postgres",
            )
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("stop_kind", ["run-limit", "budget-limit", "model-step-limit"])
def test_terminal_limit_hook_receives_active_invocation_profile(stop_kind: str) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        hook = RecordingCompletionHook(f"{stop_kind}-profile-hook")
        completion_payload: dict[str, object] = {"finish_reason": "stop"}
        if stop_kind != "model-step-limit":
            completion_payload["usage"] = {
                "input_tokens": 1000,
                "output_tokens": 100,
                "total_tokens": 1100,
            }
        provider_events = [
            ModelStreamEvent.text_delta("bounded answer"),
            ModelStreamEvent.completed(completion_payload),
        ]
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(provider_events, name="fake"),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            runtime_hooks=[hook],
        )
        model_profiles: list[ExecutionProfileIdentity | None] = []
        original_model_run_factory = app._model_step_executor.create_run

        def capture_model_run(**kwargs):
            model_profiles.append(kwargs["execution_profile"])
            return original_model_run_factory(**kwargs)

        app._model_step_executor.create_run = capture_model_run
        limits = RunLimits(max_total_tokens=1000) if stop_kind == "run-limit" else RunLimits()
        budget_limits = (
            (
                BudgetLimit(
                    max_estimated_cost=Decimal("0.002"),
                    pricing=PriceBook(
                        prices=(
                            ModelPrice.fixed(
                                provider_name="fake",
                                model="fake-model",
                                input_per_million=Decimal("1"),
                                output_per_million=Decimal("10"),
                            ),
                        )
                    ),
                ),
            )
            if stop_kind == "budget-limit"
            else ()
        )

        events = await collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"active-profile-{stop_kind}",
                    messages=[Message.text("user", "run within the configured boundary")],
                    max_steps=1,
                    limits=limits,
                    budget_limits=budget_limits,
                    loop_policies=(
                        (ContinueAtModelStepLimitPolicy(),)
                        if stop_kind == "model-step-limit"
                        else ()
                    ),
                )
            )
        )

        assert any(event.type is EventType.SESSION_INTERRUPTED for event in events)
        assert len(model_profiles) == 1
        assert model_profiles[0] is not None
        assert hook.interrupted_execution_profiles == [model_profiles[0]]
        assert hook.interrupted_execution_profiles[0] is model_profiles[0]

    asyncio.run(scenario())


def test_abandoned_recovery_terminal_hook_receives_active_invocation_profile() -> None:
    async def scenario() -> None:
        session_id = "active-profile-abandoned-terminal-hook"
        store = InMemorySessionStore()
        hook = RecordingCompletionHook("abandoned-profile-hook")
        replacement_hook = RecordingCompletionHook("abandoned-profile-hook")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    ModelStreamEvent.text_delta("partial"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            runtime_hooks=[hook],
        )
        replacement_app = CayuApp(enable_logging=False)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            runtime_hooks=[replacement_hook],
        )

        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "stop after admission")],
            )
        )
        while True:
            event = await anext(stream)
            if event.type is EventType.MODEL_TEXT_DELTA:
                break
        app._agents["assistant"] = replacement_app._agents["assistant"]
        await stream.aclose()

        session = await store.load(session_id)
        assert session is not None
        assert session.status is SessionStatus.INTERRUPTED
        checkpoint = await store.load_checkpoint(session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert active_profile is not None
        assert hook.interrupted_execution_profiles == [active_profile.profile]
        assert replacement_hook.interrupted_execution_profiles == []

    asyncio.run(scenario())


def test_cancelled_run_finalization_keeps_process_local_runtime_after_registration_mutation() -> (
    None
):
    async def scenario() -> None:
        session_id = "active-profile-cancelled-terminal-hook"
        store = InMemorySessionStore()
        provider = BlockingStreamProvider()
        hook = RecordingCompletionHook("cancelled-profile-hook")
        replacement_hook = RecordingCompletionHook("cancelled-profile-hook")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            runtime_hooks=[hook],
        )
        replacement_app = CayuApp(enable_logging=False)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            runtime_hooks=[replacement_hook],
        )

        run_task = asyncio.create_task(
            collect(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "cancel after dispatch")],
                    )
                )
            )
        )
        await asyncio.wait_for(provider.stream_waiting.wait(), timeout=1.0)
        app._agents["assistant"] = replacement_app._agents["assistant"]

        assert run_task.cancelling() == 0
        run_task.cancel("cancel frozen-profile run")
        assert run_task.cancelling() == 1
        try:
            await run_task
        except asyncio.CancelledError as cancellation:
            assert cancellation.args == ("cancel frozen-profile run",)
        else:
            pytest.fail("Run cancellation did not propagate.")
        assert run_task.cancelled() is True

        session = await store.load(session_id)
        assert session is not None
        assert session.status is SessionStatus.INTERRUPTED
        checkpoint = await store.load_checkpoint(session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert active_profile is not None
        assert hook.interrupted_execution_profiles == [active_profile.profile]
        assert replacement_hook.interrupted_execution_profiles == []

    asyncio.run(scenario())


def test_cancelled_continuation_claim_keeps_prevalidated_runtime_after_registration_mutation() -> (
    None
):
    async def scenario() -> None:
        session_id = "active-profile-cancelled-continuation-claim"
        store = InMemorySessionStore()
        hook = RecordingCompletionHook("continuation-claim-hook")
        replacement_hook = RecordingCompletionHook("continuation-claim-hook")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    ModelStreamEvent.tool_call(
                        id="call-input",
                        name="ask_user",
                        arguments={"question": "Continue?"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                name="fake",
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool()],
            runtime_hooks=[hook],
        )
        replacement_app = CayuApp(enable_logging=False)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool()],
            runtime_hooks=[replacement_hook],
        )

        paused = await collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "ask before cancellation")],
                )
            )
        )
        awaiting = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )
        hook.interrupted_execution_profiles.clear()

        original_transition = store.transition_status_and_checkpoint
        continuation_task: asyncio.Task[list[Event]] | None = None

        async def mutate_after_claim(*args, **kwargs):
            result = await original_transition(*args, **kwargs)
            if kwargs.get("to_status") is SessionStatus.RUNNING:
                app._agents["assistant"] = replacement_app._agents["assistant"]
                if continuation_task is None:
                    raise AssertionError("Continuation task was not registered.")
                continuation_task.cancel("cancel after continuation claim")
            return result

        store.transition_status_and_checkpoint = mutate_after_claim
        continuation_task = asyncio.create_task(
            collect(
                app.resolve_user_input(
                    UserInputResponse(
                        session_id=session_id,
                        input_id=awaiting.payload["input_id"],
                        answer="yes",
                    )
                )
            )
        )

        try:
            await continuation_task
        except asyncio.CancelledError as cancellation:
            assert cancellation.args == ("cancel after continuation claim",)
        else:
            pytest.fail("Continuation-claim cancellation did not propagate.")
        assert continuation_task.cancelled() is True

        session = await store.load(session_id)
        assert session is not None
        assert session.status is SessionStatus.INTERRUPTED
        checkpoint = await store.load_checkpoint(session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert active_profile is not None
        assert hook.interrupted_execution_profiles == [active_profile.profile]
        assert replacement_hook.interrupted_execution_profiles == []

    asyncio.run(scenario())


def test_failed_continuation_resume_cleanup_keeps_prevalidated_profile_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        session_id = "active-profile-failed-continuation-resume"
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    ModelStreamEvent.tool_call(
                        id="call-input",
                        name="ask_user",
                        arguments={"question": "Continue?"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                name="fake",
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool()],
        )

        paused = await collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "ask before resume failure")],
                )
            )
        )
        awaiting = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )
        checkpoint = await store.load_checkpoint(session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert active_profile is not None

        async def return_prevalidated_profile(*_args, **_kwargs):
            return active_profile

        async def fail_resume_interaction(*_args, **_kwargs):
            raise RuntimeError("simulated resume event failure")

        finalized_profiles: list[ExecutionProfileIdentity | None] = []
        released_profiles: list[ExecutionProfileIdentity | None] = []

        async def record_finalization(_session_id: str, **kwargs) -> None:
            finalized_profiles.append(kwargs.get("execution_profile"))

        async def record_release(*, session_id: str, execution_profile=None) -> None:
            assert session_id == active_profile.session_id
            released_profiles.append(execution_profile)

        coordinator = app._recovery_coordinator
        monkeypatch.setattr(
            coordinator,
            "_validate_execution_profile_continuation",
            return_prevalidated_profile,
        )
        monkeypatch.setattr(coordinator, "_resume_interaction", fail_resume_interaction)
        monkeypatch.setattr(
            coordinator,
            "finalize_abandoned_session_by_id",
            record_finalization,
        )
        monkeypatch.setattr(
            coordinator._environment_lifecycle,
            "release_run_fence_after_environment_cleanup",
            record_release,
        )

        with pytest.raises(RuntimeError, match="simulated resume event failure"):
            await collect(
                app.resolve_user_input(
                    UserInputResponse(
                        session_id=session_id,
                        input_id=awaiting.payload["input_id"],
                        answer="yes",
                    )
                )
            )

        assert finalized_profiles == [active_profile.profile]
        assert finalized_profiles[0] is active_profile.profile
        assert released_profiles == [active_profile.profile]
        assert released_profiles[0] is active_profile.profile

    asyncio.run(scenario())


def test_abandoned_user_input_continuation_hook_receives_active_invocation_profile() -> None:
    async def scenario() -> None:
        session_id = "active-profile-abandoned-user-input-continuation"
        store = InMemorySessionStore()
        hook = RecordingCompletionHook("abandoned-user-input-profile-hook")
        sibling_tool = RecordingExternalTool(
            description="Recovered sibling effect.",
            name="sibling_effect",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    (
                        ModelStreamEvent.tool_call(
                            id="call-input",
                            name="ask_user",
                            arguments={"question": "Continue?"},
                        ),
                        ModelStreamEvent.tool_call(
                            id="call-sibling",
                            name=sibling_tool.spec.name,
                            arguments={"value": "continued"},
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ),
                    (
                        ModelStreamEvent.text_delta("partial continuation"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ),
                ],
                name="fake",
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), sibling_tool],
            runtime_hooks=[hook],
        )

        paused = await collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "ask")],
                )
            )
        )
        awaiting = next(
            event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
        )
        hook.interrupted_execution_profiles.clear()
        hook.before_tool_execution_profiles.clear()
        hook.after_tool_execution_profiles.clear()
        continuation_profiles: list[ExecutionProfileIdentity | None] = []
        original_model_run_factory = app._model_step_executor.create_run

        def capture_continuation_profile(**kwargs):
            continuation_profiles.append(kwargs["execution_profile"])
            return original_model_run_factory(**kwargs)

        app._model_step_executor.create_run = capture_continuation_profile

        continuation = app.resolve_user_input(
            UserInputResponse(
                session_id=session_id,
                input_id=awaiting.payload["input_id"],
                answer="yes",
            )
        )
        while True:
            event = await anext(continuation)
            if event.type is EventType.MODEL_TEXT_DELTA:
                break
        await continuation.aclose()

        session = await store.load(session_id)
        assert session is not None
        assert session.status is SessionStatus.INTERRUPTED
        checkpoint = await store.load_checkpoint(session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert active_profile is not None
        assert hook.interrupted_execution_profiles == [active_profile.profile]
        assert len(continuation_profiles) == 1
        assert sibling_tool.calls == [{"value": "continued"}]
        assert hook.before_tool_execution_profiles == [continuation_profiles[0]]
        assert hook.before_tool_execution_profiles[0] is continuation_profiles[0]
        assert hook.after_tool_execution_profiles == [
            continuation_profiles[0],
            continuation_profiles[0],
        ]
        assert all(
            profile is continuation_profiles[0] for profile in hook.after_tool_execution_profiles
        )

    asyncio.run(scenario())


def test_approval_continuation_sibling_hooks_receive_active_invocation_profile() -> None:
    async def scenario() -> None:
        session_id = "active-profile-approval-continuation-sibling"
        store = InMemorySessionStore()
        hook = RecordingCompletionHook("approval-continuation-sibling-profile-hook")
        approved_tool = RecordingExternalTool(description="Approved effect.")
        sibling_tool = RecordingExternalTool(
            description="Allowed sibling effect.",
            name="sibling_effect",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    (
                        ModelStreamEvent.tool_call(
                            id="call-approved",
                            name=approved_tool.spec.name,
                            arguments={"value": "approved"},
                        ),
                        ModelStreamEvent.tool_call(
                            id="call-sibling",
                            name=sibling_tool.spec.name,
                            arguments={"value": "sibling"},
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ),
                    (
                        ModelStreamEvent.text_delta("continued"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ),
                ],
                name="fake",
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[approved_tool, sibling_tool],
            tool_policy=SelectiveApprovalPolicy(),
            runtime_hooks=[hook],
        )

        paused = await collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run both effects")],
                )
            )
        )
        approval_event = next(
            event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        pending = approval_event.payload["approval"]
        assert isinstance(pending, dict)
        hook.before_tool_execution_profiles.clear()
        hook.after_tool_execution_profiles.clear()
        continuation_profiles: list[ExecutionProfileIdentity | None] = []
        original_model_run_factory = app._model_step_executor.create_run

        def capture_continuation_profile(**kwargs):
            continuation_profiles.append(kwargs["execution_profile"])
            return original_model_run_factory(**kwargs)

        app._model_step_executor.create_run = capture_continuation_profile

        completed = await collect(
            app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=session_id,
                    approval_id=pending["approval_id"],
                    tool_round_id=pending["tool_round_id"],
                    tool_call_id=pending["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        )

        assert any(event.type is EventType.SESSION_COMPLETED for event in completed)
        assert approved_tool.calls == [{"value": "approved"}]
        assert sibling_tool.calls == [{"value": "sibling"}]
        assert len(continuation_profiles) == 1
        assert hook.before_tool_execution_profiles == [
            continuation_profiles[0],
            continuation_profiles[0],
        ]
        assert all(
            profile is continuation_profiles[0] for profile in hook.before_tool_execution_profiles
        )
        assert hook.after_tool_execution_profiles == [
            continuation_profiles[0],
            continuation_profiles[0],
        ]
        assert all(
            profile is continuation_profiles[0] for profile in hook.after_tool_execution_profiles
        )

    asyncio.run(scenario())


def test_recovery_operator_interruption_hook_receives_active_invocation_profile() -> None:
    async def scenario() -> None:
        session_id = "active-profile-recovery-operator-interruption"
        store = InMemorySessionStore()
        hook = RecordingCompletionHook("recovery-operator-interruption-profile-hook")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    ModelStreamEvent.tool_call(
                        id="call-input",
                        name="ask_user",
                        arguments={"question": "Continue?"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                name="fake",
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool()],
            runtime_hooks=[hook],
        )
        await collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "ask")],
                )
            )
        )
        checkpoint = await store.load_checkpoint(session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert active_profile is not None
        await store.transform_checkpoint(
            session_id,
            lambda _session, current: {
                **({} if current is None else current),
                "pending_session_interrupt": {
                    "interruption_type": "operator_requested",
                    "interruption_request_id": "profile-recovery-interruption",
                },
            },
        )
        interrupting = await store.transition_status(
            session_id,
            from_statuses={SessionStatus.INTERRUPTED},
            to_status=SessionStatus.INTERRUPTING,
        )
        hook.interrupted_execution_profiles.clear()

        events = await collect(
            app._recovery_coordinator._interrupt_for_resumable_manual_recovery(
                session=interrupting,
                registered_agent=app._agents["assistant"],
                registered_environment=None,
                execution_profile=active_profile.profile,
                payload={"interruption_type": "runtime_interrupted"},
            )
        )

        assert any(event.type is EventType.SESSION_INTERRUPTED for event in events)
        assert hook.interrupted_execution_profiles == [active_profile.profile]
        assert hook.interrupted_execution_profiles[0] is active_profile.profile

    asyncio.run(scenario())


@pytest.mark.parametrize("background", [False, True])
def test_model_recovery_record_references_active_profile(background: bool) -> None:
    async def scenario() -> None:
        session_id = "active-profile-background-model"
        store = InMemorySessionStore()
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            name="fake",
            background=background,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        events = await collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "background")],
                )
            )
        )
        assert any(event.type is EventType.MODEL_STARTED for event in events)
        durable_events = await store.load_events(session_id)
        started = next(event for event in durable_events if event.type is EventType.MODEL_STARTED)
        model_step_id = started.payload["model_step_id"]
        stage = await store.load_model_completion_stage(
            session_id,
            f"{model_step_id}:dispatch:0",
        )
        checkpoint = await store.load_checkpoint(session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert stage is not None
        assert active_profile is not None
        recovery_context = model_completion_recovery_context_from_stage(stage)
        assert recovery_context is not None
        assert recovery_context.execution_profile_fingerprint == active_profile.profile.fingerprint

    asyncio.run(scenario())


def test_model_reconciliation_uses_frozen_provider_after_registration_mutation() -> None:
    async def scenario() -> None:
        session_id = f"active-profile-frozen-recovery-{uuid4().hex}"
        store = InMemorySessionStore()
        original_adapter = CompletingBlockingProviderOperationAdapter()
        original_provider = BlockingBackgroundProvider(original_adapter)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(original_provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        frozen_agent = app._agents["assistant"]
        frozen_provider = app._providers["fake"]

        run_task = asyncio.create_task(
            collect(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "start background work")],
                    )
                )
            )
        )
        await asyncio.wait_for(original_adapter.start_entered.wait(), timeout=1.0)

        replacement_adapter = BlockingProviderOperationAdapter()
        replacement_provider = BlockingBackgroundProvider(replacement_adapter)
        replacement_app = CayuApp(enable_logging=False)
        replacement_app.register_provider(replacement_provider, default=True)
        app._providers["fake"] = replacement_app._providers["fake"]

        try:
            session = await store.load(session_id)
            assert session is not None
            assert await store.load_active_model_completion_stage(session_id) is not None
            reconciliation = await app._recovery_coordinator.reconcile_model_completion_boundary(
                session,
                registered_agent=frozen_agent,
                registered_provider=frozen_provider,
                registered_environment=None,
            )

            assert reconciliation.state == "provider_operation_reconciled"
            assert original_adapter.recovery_calls == 1
            assert replacement_adapter.recovery_calls == 0
        finally:
            if not run_task.done():
                run_task.cancel("test cleanup")
                original_adapter.start_release.set()
                await asyncio.gather(run_task, return_exceptions=True)

    asyncio.run(scenario())


async def _assert_snapshot_only_restart_profile_boundary(
    store: SessionStore,
    *,
    suffix: str,
    restart_state: str,
) -> None:
    session_id = f"active-profile-snapshot-only-{restart_state}-{suffix}-{uuid4().hex}"
    original_tool = RecordingExternalTool(description="Original snapshot-only tool.")
    original_provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("initial complete"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        name="fake",
    )
    original_app = CayuApp(session_store=store, enable_logging=False)
    original_app.register_provider(original_provider, default=True)
    original_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[original_tool],
        runtime_hooks=[RecordingCompletionHook("snapshot-only-recovery-hook")],
    )
    await collect(
        original_app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "establish the durable profile")],
            )
        )
    )

    released = await store.load(session_id)
    checkpoint = await store.load_checkpoint(session_id)
    prior_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    assert released is not None
    assert prior_profile is not None
    interaction_id = f"snapshot-only-interaction-{uuid4().hex}"
    interaction_started = original_app._session_engine._interaction_started_event(
        session=released,
        registered_agent=original_app._agents["assistant"],
        environment_name=None,
        interaction_id=interaction_id,
    )

    async def admit_without_releasing() -> None:
        running = await store.admit_session_invocation(
            session_id,
            admission=SessionInvocationAdmission(
                from_statuses=frozenset({SessionStatus.COMPLETED}),
                checkpoint_transform=lambda _session, current: current,
                execution_profile=prior_profile.profile,
                interaction_started_event=interaction_started,
                interaction_source_messages=(
                    Message.text("user", "crash before the next durable stage"),
                ),
            ),
        )
        assert running.status is SessionStatus.RUNNING
        if restart_state == "missing":
            await store.transform_checkpoint(
                session_id,
                lambda _session, current: {
                    key: value
                    for key, value in (current or {}).items()
                    if key != "active_invocation_execution_profile"
                },
            )
        elif restart_state == "malformed":
            await store.transform_checkpoint(
                session_id,
                lambda _session, current: {
                    **(current or {}),
                    "active_invocation_execution_profile": {"record_type": "invalid"},
                },
            )

    # A child task models process-local fence authority disappearing without
    # advancing the durable epoch, as it would after worker loss.
    await asyncio.create_task(admit_without_releasing())
    crashed = await store.load(session_id)
    assert crashed is not None
    assert crashed.status is SessionStatus.RUNNING
    assert await store.load_active_model_completion_stage(session_id) is None
    assert (
        _tool_round_recovery.pending_tool_round_from_checkpoint(
            await store.load_checkpoint(session_id)
        )
        is None
    )

    recovery_hook = RecordingCompletionHook("snapshot-only-recovery-hook")
    replacement_description = (
        "Changed snapshot-only tool."
        if restart_state == "changed"
        else "Original snapshot-only tool."
    )
    replacement_provider = ScriptedModelProvider([], name="fake")
    replacement_app = CayuApp(session_store=store, enable_logging=False)
    replacement_app.register_provider(replacement_provider, default=True)
    replacement_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[RecordingExternalTool(description=replacement_description)],
        runtime_hooks=[recovery_hook],
    )
    if restart_state == "unchanged":
        result = await replacement_app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        assert result.actions == (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,)
        assert result.status is SessionStatus.INTERRUPTED
        assert len(recovery_hook.interrupted_execution_profiles) == 1
        assert recovery_hook.interrupted_execution_profiles[0] == prior_profile.profile
        recovered = await store.load(session_id)
        recovered_checkpoint = await store.load_checkpoint(session_id)
        recovered_profile = active_invocation_execution_profile_from_checkpoint(
            recovered_checkpoint
        )
        assert recovered is not None
        assert recovered_profile is not None
        assert recovered_profile.run_epoch == recovered.run_epoch - 1
    else:
        expected_error: type[Exception] = (
            ExecutionProfileMismatchError
            if restart_state == "changed"
            else RuntimeError
            if restart_state == "missing"
            else ValueError
        )
        with pytest.raises(expected_error):
            await replacement_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
        assert recovery_hook.interrupted_execution_profiles == []
        after = await store.load(session_id)
        assert after is not None
        assert after.run_epoch == crashed.run_epoch
    assert replacement_provider.requests == []


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("restart_state", ["unchanged", "changed", "missing", "malformed"])
def test_snapshot_only_restart_profile_boundary_in_local_stores(
    store_kind: str,
    restart_state: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(
                tmp_path / f"active-profile-snapshot-only-{restart_state}.sqlite"
            )
        )
        try:
            await _assert_snapshot_only_restart_profile_boundary(
                store,
                suffix=store_kind,
                restart_state=restart_state,
            )
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    asyncio.run(scenario())


@pytest.mark.parametrize("restart_state", ["unchanged", "changed", "missing", "malformed"])
def test_snapshot_only_restart_profile_boundary_in_postgres(
    postgres_dsn: str,
    restart_state: str,
) -> None:
    async def scenario() -> None:
        from cayu import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await _assert_snapshot_only_restart_profile_boundary(
                store,
                suffix="postgres",
                restart_state=restart_state,
            )
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_worker_recovery_accepts_never_admitted_profiled_pending_session(
    store_kind: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        session_id = f"profiled-pending-before-admission-{store_kind}-{uuid4().hex}"
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "profiled-pending-before-admission.sqlite")
        )
        provider = ScriptedModelProvider([], name="fake")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "created but never admitted")],
                ),
                identity=profiled_session_identity(
                    provider_name="fake",
                    model="fake-model",
                ),
            )
            assert created.status is SessionStatus.PENDING
            assert created.run_epoch == 0
            assert (
                active_invocation_execution_profile_from_checkpoint(
                    await store.load_checkpoint(session_id)
                )
                is None
            )

            recovered = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
            assert recovered.actions == (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,)
            assert recovered.status is SessionStatus.INTERRUPTED
            assert provider.requests == []
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    asyncio.run(scenario())


def test_worker_recovery_rejects_malformed_never_admitted_profile_baseline() -> None:
    class MalformedBaselineStore(InMemorySessionStore):
        corrupt_reads = False

        async def load(self, session_id: str):
            session = await super().load(session_id)
            if session is None or not self.corrupt_reads:
                return session
            return session.model_copy(
                update={
                    "metadata": {
                        **session.metadata,
                        EXECUTION_PROFILE_METADATA_KEY: {"record_type": "invalid"},
                    }
                }
            )

    async def scenario() -> None:
        session_id = f"malformed-profiled-pending-{uuid4().hex}"
        store = MalformedBaselineStore()
        provider = ScriptedModelProvider([], name="fake")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "created but never admitted")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        store.corrupt_reads = True

        with pytest.raises(ValueError, match="execution-profile metadata"):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
        store.corrupt_reads = False
        unchanged = await store.load(session_id)
        assert unchanged is not None
        assert unchanged.status is SessionStatus.PENDING
        assert unchanged.run_epoch == created.run_epoch
        assert provider.requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_worker_recovery_releases_terminal_invocation_owner(
    store_kind: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        suffix = uuid4().hex
        seed_session_id = f"active-profile-terminal-owner-seed-{store_kind}-{suffix}"
        session_id = f"active-profile-terminal-owner-{store_kind}-{suffix}"
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "active-profile-terminal-owner.sqlite")
        )
        original_provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("initial complete"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            name="fake",
        )
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(original_provider, default=True)
        original_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        try:
            await collect(
                original_app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=seed_session_id,
                        messages=[Message.text("user", "establish the profile")],
                    )
                )
            )
            checkpoint = await store.load_checkpoint(seed_session_id)
            prior_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            assert prior_profile is not None
            pending = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "create a recovery target")],
                ),
                identity=SessionIdentity(
                    provider_name="fake",
                    model="fake-model",
                    execution_profile=prior_profile.profile,
                ),
            )
            interaction_id = f"terminal-owner-interaction-{uuid4().hex}"
            interaction_started = original_app._session_engine._interaction_started_event(
                session=pending,
                registered_agent=original_app._agents["assistant"],
                environment_name=None,
                interaction_id=interaction_id,
            )

            async def publish_terminal_without_releasing() -> None:
                await original_app._runtime_session_store.admit_session_invocation(
                    session_id,
                    admission=SessionInvocationAdmission(
                        from_statuses=frozenset({SessionStatus.PENDING}),
                        checkpoint_transform=lambda _session, current: current,
                        execution_profile=prior_profile.profile,
                        interaction_started_event=interaction_started,
                        interaction_source_messages=(
                            Message.text("user", "finish before worker loss"),
                        ),
                    ),
                )
                completed_at = interaction_started.timestamp
                await store.publish_interaction_transition(
                    session_id,
                    event=Event(
                        type=EventType.INTERACTION_COMPLETED,
                        session_id=session_id,
                        interaction_id=interaction_id,
                        timestamp=completed_at,
                        agent_name="assistant",
                        payload=InteractionSummaryEvidence(
                            status=InteractionStatus.COMPLETED,
                            start_event_id=interaction_started.id,
                            started_at=interaction_started.timestamp,
                            completed_at=completed_at,
                        ).model_dump(mode="json"),
                    ),
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.COMPLETED,
                )
                await store.append_event(
                    session_id,
                    Event(
                        type=EventType.SESSION_COMPLETED,
                        session_id=session_id,
                        agent_name="assistant",
                        payload={},
                    ),
                )

            await asyncio.create_task(publish_terminal_without_releasing())
            stranded = await store.load(session_id)
            stranded_checkpoint = await store.load_checkpoint(session_id)
            stranded_profile = active_invocation_execution_profile_from_checkpoint(
                stranded_checkpoint
            )
            assert stranded is not None
            assert stranded_checkpoint is not None
            assert stranded_profile is not None
            assert stranded_profile.run_epoch == stranded.run_epoch
            assert (
                stranded_checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY]
                == CURRENT_CHECKPOINT_SCHEMA_VERSION
            )

            recovery_provider = ScriptedModelProvider(
                [
                    ModelStreamEvent.text_delta("resumed after recovery"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                name="fake",
            )
            recovery_app = CayuApp(session_store=store, enable_logging=False)
            recovery_app.register_provider(recovery_provider, default=True)
            recovery_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            recovered_page = await recovery_app.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(
                    statuses={SessionStatus.COMPLETED},
                    limit=10,
                )
            )
            assert len(recovered_page.results) == 1
            recovered = recovered_page.results[0]
            assert recovered.session_id == session_id
            assert recovered.actions == (
                IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_OWNERSHIP,
            )
            assert "ownership" in recovered.message
            settled = await store.load(session_id)
            settled_checkpoint = await store.load_checkpoint(session_id)
            settled_profile = active_invocation_execution_profile_from_checkpoint(
                settled_checkpoint
            )
            assert settled is not None
            assert settled_profile is not None
            assert settled_profile.run_epoch == settled.run_epoch - 1

            healthy_page = await recovery_app.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(
                    statuses={SessionStatus.COMPLETED},
                    limit=10,
                )
            )
            assert healthy_page.results == ()

            resumed = await collect(
                recovery_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "continue")],
                    )
                )
            )
            assert any(event.type is EventType.SESSION_COMPLETED for event in resumed)
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    asyncio.run(scenario())


async def _assert_crashed_model_operation_rejects_invalid_restart_before_recovery(
    store: SessionStore,
    *,
    suffix: str,
    invalid_restart: str,
) -> None:
    session_id = f"active-profile-model-crash-drift-{suffix}-{uuid4().hex}"
    adapter = BlockingProviderOperationAdapter()
    provider = BlockingBackgroundProvider(adapter)
    original_app = CayuApp(session_store=store, enable_logging=False)
    original_app.register_provider(provider, default=True)
    original_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[RecordingExternalTool(description="Original tool.")],
    )

    run_task = asyncio.create_task(
        collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start background work")],
                )
            )
        )
    )
    await asyncio.wait_for(adapter.start_entered.wait(), timeout=1.0)

    replacement_description = "Replacement tool."
    expected_error: type[Exception] = ExecutionProfileMismatchError
    expected_message: str | None = None
    if invalid_restart == "interaction":
        checkpoint = await store.load_checkpoint(session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        assert active_profile is not None
        await store.transform_checkpoint(
            session_id,
            lambda _session, current: checkpoint_with_active_invocation_execution_profile(
                current,
                session_id=session_id,
                interaction_id="different-open-interaction",
                run_epoch=active_profile.run_epoch,
                profile=active_profile.profile,
                expected=active_profile,
            ),
        )
        await store.release_run_fence(session_id)
        replacement_description = "Original tool."
        expected_error = RuntimeError
        expected_message = "another interaction"
    else:
        run_task.cancel("simulated process loss")
        adapter.start_release.set()
        with pytest.raises(asyncio.CancelledError):
            await run_task

    replacement_app = CayuApp(session_store=store, enable_logging=False)
    replacement_app.register_provider(provider, default=True)
    replacement_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[RecordingExternalTool(description=replacement_description)],
    )
    try:
        with pytest.raises(expected_error, match=expected_message):
            if invalid_restart == "interaction":
                await replacement_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id)
                )
            else:
                await collect(
                    replacement_app.resume(
                        ResumeRequest(
                            session_id=session_id,
                            messages=[Message.text("user", "continue")],
                        )
                    )
                )
        assert adapter.recovery_calls == 0
    finally:
        if not run_task.done():
            run_task.cancel("test cleanup")
            adapter.start_release.set()
            await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("invalid_restart", ["profile", "interaction"])
def test_crashed_model_operation_rejects_invalid_restart_in_local_stores(
    store_kind: str,
    invalid_restart: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / f"active-profile-crash-{invalid_restart}.sqlite")
        )
        try:
            await _assert_crashed_model_operation_rejects_invalid_restart_before_recovery(
                store,
                suffix=store_kind,
                invalid_restart=invalid_restart,
            )
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_restart", ["profile", "interaction"])
def test_crashed_model_operation_rejects_invalid_restart_in_postgres(
    postgres_dsn: str,
    invalid_restart: str,
) -> None:
    async def scenario() -> None:
        from cayu import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await _assert_crashed_model_operation_rejects_invalid_restart_before_recovery(
                store,
                suffix="postgres",
                invalid_restart=invalid_restart,
            )
        finally:
            await store.close()

    asyncio.run(scenario())
